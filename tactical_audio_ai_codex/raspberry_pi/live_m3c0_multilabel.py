#!/usr/bin/env python3
"""Live SAFIRA multi-label inference from M3C0.

Ambient L/R are used for environmental classes (drone, gunshot, background).
The dedicated voice channel is used for the speech score. This fixes the old
behavior where channel 3 was ignored completely.

This script is still a CLASSIFIER. It does not claim to output separated audio.
"""
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import serial
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parents[1] / "training"))
sys.path.insert(0, str(Path(__file__).parent))

from audio_model import AudioCNN
from decision_logic import decide
from m3c0 import sync_packet
from self_speech import add_self_speech_arguments, detector_from_args


def feature(x, sr, n_mels):
    t = torch.from_numpy(x.astype(np.float32) / 32768.0)
    mel = torchaudio.transforms.MelSpectrogram(
        sr,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=n_mels,
        f_min=20,
        f_max=7600,
    )(t)
    z = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel)
    z = (z - z.mean()) / (z.std() + 1e-6)
    return z[None, None]


def infer(model, x, sr, n_mels):
    with torch.no_grad():
        return torch.sigmoid(model(feature(x, sr, n_mels)))[0].numpy()


def combine_channel_probabilities(classes, ambient_prob, voice_prob):
    """Speech comes from voice mic; environmental classes come from ambient."""
    return np.asarray(
        [
            voice_prob[i] if c.lower() == "speech" else ambient_prob[i]
            for i, c in enumerate(classes)
        ],
        dtype=np.float32,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyUSB1")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--hop", type=float, default=0.25)
    p.add_argument("--smoothing", type=int, default=4)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--yamnet", action="store_true")
    p.add_argument("--yamnet-handle", default="https://tfhub.dev/google/yamnet/1")
    p.add_argument("--yamnet-class-map")
    p.add_argument("--yamnet-confidence", type=float, default=0.25)
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-start-command", action="store_true")
    add_self_speech_arguments(p)
    a = p.parse_args()

    if a.channels != 3:
        raise SystemExit("M3C0 SAFIRA mode requires 3 channels: ambient L/R + voice")
    if not np.isfinite(a.hop) or a.hop <= 0:
        p.error("--hop must be positive and finite")
    detector = detector_from_args(a)

    ck = torch.load(a.model, map_location="cpu")
    if ck.get("task") != "multilabel":
        raise SystemExit("This script requires a checkpoint with task=multilabel")

    classes = ck["classes"]
    sr = int(ck.get("sample_rate", 16000))
    if sr != 16000 or "speech" not in [c.lower() for c in classes]:
        raise SystemExit("self-speech attribution requires 16 kHz and a speech class")
    n_mels = int(ck.get("n_mels", 64))
    thresholds = {
        c: (
            a.threshold
            if a.threshold is not None
            else float(ck.get("thresholds", {}).get(c, 0.5))
        )
        for c in classes
    }

    model = AudioCNN(len(classes), **ck.get("model_config", {}))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    fallback = None
    if a.yamnet:
        from audioset_fallback import YamnetFallback

        fallback = YamnetFallback(
            a.yamnet_handle,
            a.yamnet_class_map,
            a.yamnet_confidence,
        )

    ambient_ring = deque(maxlen=sr * 2)
    voice_ring = deque(maxlen=sr * 2)
    attribution_ring = deque(maxlen=sr)
    history = deque(maxlen=max(1, a.smoothing))
    since = 0
    expected = None
    hop = max(1, int(sr * a.hop))

    with serial.Serial(a.port, a.baud, timeout=2) as s:
        s.reset_input_buffer()
        if not a.no_start_command:
            s.write(b"START_INMP_ONLY\n")
            s.flush()
            time.sleep(0.2)

        while True:
            seq, n, payload = sync_packet(s, a.channels)
            if n != 320:
                raise ValueError(f"expected M3C0 320 samples, got {n}")
            if expected is not None and seq != expected:
                print(f"[WARN] sequence jump expected={expected} received={seq}", file=sys.stderr)
                ambient_ring.clear()
                voice_ring.clear()
                attribution_ring.clear()
                history.clear()
                detector.reset()
                since = 0
            expected = (seq + 1) & 0xFFFF

            pcm = np.frombuffer(payload, dtype="<i2").reshape(n, a.channels)
            ambient = (
                (
                    pcm[:, 0].astype(np.int32)
                    + pcm[:, 1].astype(np.int32)
                )
                // 2
            ).astype(np.int16)
            voice = pcm[:, 2].astype(np.int16, copy=False)

            ambient_ring.extend(ambient)
            voice_ring.extend(voice)
            attribution_ring.extend(pcm.copy())
            since += n

            if len(ambient_ring) >= sr and len(voice_ring) >= sr and since >= hop:
                elapsed_seconds = since / sr
                since = 0
                ambient_x = np.asarray(ambient_ring, dtype=np.int16)[-sr:]
                voice_x = np.asarray(voice_ring, dtype=np.int16)[-sr:]

                t = time.perf_counter()
                ambient_prob = infer(model, ambient_x, sr, n_mels)
                voice_prob = infer(model, voice_x, sr, n_mels)
                combined = combine_channel_probabilities(
                    classes, ambient_prob, voice_prob
                )

                history.append(combined)
                prob = np.mean(history, axis=0)
                out = decide(
                    classes,
                    prob,
                    thresholds,
                    fallback,
                    ambient_x,
                    sr,
                )
                out["channel_targets"] = {
                    "ambient": {
                        c: round(float(v) * 100, 1)
                        for c, v in zip(classes, ambient_prob)
                    },
                    "voice": {
                        c: round(float(v) * 100, 1)
                        for c, v in zip(classes, voice_prob)
                    },
                }
                out["speech_source"] = "voice_channel"
                detector.annotate(out, np.asarray(attribution_ring), classes,
                                  ambient_prob, voice_prob, thresholds, elapsed_seconds)
                out["environment_source"] = "ambient_lr_mix"
                out["inference_ms"] = round(
                    (time.perf_counter() - t) * 1000, 1
                )

                if a.json:
                    print(json.dumps(out, ensure_ascii=False))
                else:
                    scores = " | ".join(
                        f"{c}={out['targets'][c]:5.1f}%" for c in classes
                    )
                    print(
                        f"[AI] {scores} | => {out['final']} "
                        f"| self={out['self_speech_active']} external_candidate={out['external_speech_candidate']} "
                        f"| {out['inference_ms']:.1f} ms"
                    )


if __name__ == "__main__":
    main()
