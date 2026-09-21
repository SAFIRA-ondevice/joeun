#!/usr/bin/env python3
"""SAFIRA live prototype: M3C0 -> AI -> routing -> SPK0 full duplex.

Current limitation:
The repository has a multi-label classifier but no trained source-separation
model. Therefore local ambient processing uses the explicitly named
"classification_guided_mixed_fallback" path. It never labels classifier output
as separated PCM.

Routing policy:
- real separated gunshot x0.15 -> headset (supported by audio_routing API)
- real separated drone x1.50 -> headset (supported by audio_routing API)
- user voice mic x1.00 -> server uplink
- unknown x0.00 -> headset
- server downlink voice x1.00 -> headset

When a real separator is added, replace classification_guided_ambient() with
headset_from_separated().
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import serial
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "training"))
sys.path.insert(0, str(Path(__file__).parent))

from audio_model import AudioCNN
from audio_network import Spk0UdpReceiver, Spk0UdpSender
from audio_routing import SafiraAudioRouter
from decision_logic import decide
from live_m3c0_multilabel import (
    combine_channel_probabilities,
    infer,
)
from m3c0 import sync_packet
from spk0 import FRAME_SAMPLES, pack_spk0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyUSB1")
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--hop", type=float, default=0.25)
    p.add_argument("--smoothing", type=int, default=4)
    p.add_argument("--threshold", type=float, default=None)

    p.add_argument("--server-host")
    p.add_argument("--uplink-port", type=int)
    p.add_argument("--downlink-bind", default="0.0.0.0")
    p.add_argument("--downlink-port", type=int)
    p.add_argument("--expected-downlink-peer")

    p.add_argument(
        "--headset-serial",
        action="store_true",
        help="write final local headset PCM back to ESP32 as SPK0",
    )
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-start-command", action="store_true")
    a = p.parse_args()

    if (a.server_host is None) != (a.uplink_port is None):
        p.error("--server-host and --uplink-port must be supplied together")

    ck = torch.load(a.model, map_location="cpu")
    if ck.get("task") != "multilabel":
        raise SystemExit("This script requires task=multilabel checkpoint")

    classes = ck["classes"]
    sr = int(ck.get("sample_rate", 16000))
    if sr != 16000:
        raise SystemExit(
            f"network contract is 16 kHz but checkpoint sample_rate={sr}"
        )
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

    router = SafiraAudioRouter(sample_rate=16000, frame_samples=FRAME_SAMPLES)
    uplink = (
        Spk0UdpSender(a.server_host, a.uplink_port)
        if a.server_host is not None
        else None
    )
    downlink = (
        Spk0UdpReceiver(
            a.downlink_bind,
            a.downlink_port,
            expected_peer_host=a.expected_downlink_peer,
            timeout=0.0,
        )
        if a.downlink_port is not None
        else None
    )

    ambient_ring = deque(maxlen=sr * 2)
    voice_ring = deque(maxlen=sr * 2)
    history = deque(maxlen=max(1, a.smoothing))
    detected = []
    latest_server_voice = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    since = 0
    hop = max(1, int(sr * a.hop))
    expected_m3c0 = None
    headset_sequence = 0

    try:
        with serial.Serial(a.port, a.baud, timeout=2) as ser:
            ser.reset_input_buffer()
            if not a.no_start_command:
                ser.write(b"START_INMP_ONLY\n")
                ser.flush()
                time.sleep(0.2)

            while True:
                seq, n, payload = sync_packet(ser, channels=3)
                if n != FRAME_SAMPLES:
                    raise ValueError(
                        f"expected M3C0 {FRAME_SAMPLES} samples, got {n}"
                    )
                if expected_m3c0 is not None and seq != expected_m3c0:
                    print(
                        f"[WARN] M3C0 jump expected={expected_m3c0} got={seq}",
                        file=sys.stderr,
                    )
                expected_m3c0 = (seq + 1) & 0xFFFF

                pcm = np.frombuffer(payload, dtype="<i2").reshape(n, 3)
                ambient = (
                    (
                        pcm[:, 0].astype(np.int32)
                        + pcm[:, 1].astype(np.int32)
                    )
                    // 2
                ).astype(np.int16)
                voice = pcm[:, 2].astype(np.int16, copy=True)

                ambient_ring.extend(ambient)
                voice_ring.extend(voice)
                since += n

                # Uplink: dedicated user voice x1.0, every 20 ms frame.
                if uplink is not None:
                    uplink.send(router.server_uplink(voice))

                # Downlink: keep the newest valid 20 ms server voice frame.
                if downlink is not None:
                    while True:
                        try:
                            _, received, _ = downlink.recv()
                            latest_server_voice = received
                        except (BlockingIOError, socket.timeout):
                            break
                        except ValueError as exc:
                            print(f"[WARN] drop downlink packet: {exc}", file=sys.stderr)
                            break
                else:
                    latest_server_voice.fill(0)

                # AI semantic decision uses a 1 s window every --hop seconds.
                if (
                    len(ambient_ring) >= sr
                    and len(voice_ring) >= sr
                    and since >= hop
                ):
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
                    result = decide(classes, prob, thresholds)
                    detected = result["detected"]
                    result["speech_source"] = "voice_channel"
                    result["environment_source"] = "ambient_lr_mix"
                    result["audio_routing_mode"] = (
                        "classification_guided_mixed_fallback"
                    )
                    result["source_separation_available"] = False
                    result["inference_ms"] = round(
                        (time.perf_counter() - t) * 1000, 1
                    )

                    if a.json:
                        print(json.dumps(result, ensure_ascii=False))
                    else:
                        scores = " | ".join(
                            f"{c}={result['targets'][c]:5.1f}%"
                            for c in classes
                        )
                        print(
                            f"[AI] {scores} | => {result['final']} "
                            f"| routing=fallback | {result['inference_ms']:.1f} ms"
                        )

                # Local headset path.
                # This is intentionally a mixed fallback until a real separator exists.
                headset_pcm, _ = router.classification_guided_ambient(
                    ambient,
                    detected,
                    latest_server_voice,
                )

                if a.headset_serial:
                    ser.write(pack_spk0(headset_sequence, headset_pcm))
                    ser.flush()
                    headset_sequence = (headset_sequence + 1) & 0xFFFF

    finally:
        if uplink is not None:
            uplink.close()
        if downlink is not None:
            downlink.close()


if __name__ == "__main__":
    main()
