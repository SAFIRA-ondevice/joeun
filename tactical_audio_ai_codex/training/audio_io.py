"""Audio loading without TorchCodec; reliable for PCM WAV on Raspberry Pi."""
from pathlib import Path
import soundfile as sf
import torch


def load_mono(path: Path):
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.mean(axis=1))
    return waveform, int(sample_rate)
