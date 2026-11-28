"""Trainable LR feature branch with no dependency on the output correction."""
import torch
from torch import nn
from torch.nn import functional as F


class LRConditionBranch(nn.Module):
    def __init__(self, channels, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, hidden, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(hidden, channels, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, normalized_lr, size=(48, 64)):
        if normalized_lr.ndim != 4 or normalized_lr.shape[1:] != (1, 16, 21):
            raise ValueError('Expected normalized LR of shape Bx1x16x21.')
        resized = F.interpolate(normalized_lr, size=size, mode='bilinear', align_corners=False)
        return self.net(resized)
