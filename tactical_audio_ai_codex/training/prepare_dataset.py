#!/usr/bin/env python3
import argparse, hashlib, shutil
from pathlib import Path
import soundfile as sf
import torch
import torchaudio
from audio_io import load_mono

AUDIO_EXT = {".wav", ".flac", ".ogg"}

def convert(src: Path, dst: Path, sr: int, seconds: float):
    wav, old_sr = load_mono(src)
    if old_sr != sr:
        wav = torchaudio.functional.resample(wav, old_sr, sr)
    size = int(sr * seconds)
    wav = wav[:size]
    if wav.shape[0] < size:
        wav = torch.nn.functional.pad(wav, (0, size - wav.shape[0]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, wav.numpy(), sr, subtype="PCM_16")

def main():
    p = argparse.ArgumentParser(description="Convert class folders to fixed mono PCM16 WAV files")
    p.add_argument("--source", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--classes", nargs="+", required=True); p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--seconds", type=float, default=1.0); p.add_argument("--copy-only", action="store_true")
    a = p.parse_args()
    for cls in a.classes:
        files = sorted(x for x in (a.source / cls).rglob("*") if x.suffix.lower() in AUDIO_EXT)
        for src in files:
            key = hashlib.sha1(str(src.resolve()).encode()).hexdigest()[:12]
            dst = a.output / cls / f"{src.stem}_{key}.wav"
            if a.copy_only: dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
            else: convert(src, dst, a.sample_rate, a.seconds)
        print(f"{cls}: {len(files)}")
if __name__ == "__main__": main()
