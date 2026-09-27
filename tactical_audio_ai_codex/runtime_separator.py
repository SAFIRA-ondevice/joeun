"""Inference-only definition from source_separation.py; checkpoint weights are separate.
Four estimated stems: speech, drone, gunshot, background. Not speaker separation.
"""
import torch
from torch import nn
CLASSES = ("speech", "drone", "gunshot", "background")
FFT, HOP = 512, 128


class Separator(nn.Module):
    def __init__(self):
        super().__init__()
        bins = FFT // 2 + 1
        self.net = nn.Sequential(
            nn.Conv1d(bins, 128, 5, padding=2), nn.GELU(),
            nn.Conv1d(128, 128, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(128, 128, 5, padding=8, dilation=4), nn.GELU(),
            nn.Conv1d(128, len(CLASSES) * bins, 1),
        )
        self.register_buffer("window", torch.hann_window(FFT))

    def spectrum(self, x):
        return torch.stft(x, FFT, HOP, window=self.window, center=True,
                          pad_mode="constant", return_complex=True)

    def forward(self, mixture):
        z = self.spectrum(mixture)
        scale = z.abs().square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-5)
        features = torch.log1p(z.abs() / scale)
        masks = self.net(features).reshape(len(mixture), 4, FFT // 2 + 1, -1).softmax(1)
        spectra = masks * z[:, None]
        stems = torch.istft(spectra.flatten(0, 1), FFT, HOP, window=self.window,
                            center=True, length=mixture.shape[-1])
        return stems.reshape(len(mixture), 4, -1), masks

