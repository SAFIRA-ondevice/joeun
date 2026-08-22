import torch
from torch import nn


class AudioCNN(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 32):
        super().__init__()
        blocks = []
        in_ch = 1
        for out_ch in (base_channels, base_channels * 2, base_channels * 4):
            blocks += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.MaxPool2d(2)
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))
