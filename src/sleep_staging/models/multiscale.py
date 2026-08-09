"""Multi-scale temporal feature extraction."""

import torch
from torch import nn
from torch.nn import functional as F


class SqueezeExcitation1D(nn.Module):
    """Channel recalibration for input and output shaped ``[B,C,T]``."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, channels, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.sigmoid(self.fc2(F.gelu(self.fc1(x.mean(dim=-1)))))
        return x * weights.unsqueeze(-1), weights


class MultiScaleFeatureExtractor(nn.Module):
    """Depthwise multi-kernel encoder: ``[B,C,T] -> [B,3*branch_out,T/2]``."""
    def __init__(self, in_channels=1, branch_out=32, kernels=(7, 15, 31),
                 stride_after=2, use_se=True):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel, padding=kernel // 2,
                          groups=in_channels, bias=False),
                nn.BatchNorm1d(in_channels), nn.GELU(),
                nn.Conv1d(in_channels, branch_out, 1, bias=False),
                nn.BatchNorm1d(branch_out), nn.GELU(),
            ) for kernel in kernels
        ])
        self.out_channels = branch_out * len(kernels)
        self.reduce = nn.Conv1d(self.out_channels, self.out_channels, 3, padding=1,
                                stride=stride_after, bias=False)
        self.bn = nn.BatchNorm1d(self.out_channels)
        self.act = nn.GELU()
        self.use_se = use_se
        self.se = SqueezeExcitation1D(self.out_channels) if use_se else None

    def forward(self, x):
        z = self.act(self.bn(self.reduce(torch.cat([branch(x) for branch in self.branches], dim=1))))
        if self.se is None:
            return z, None
        return self.se(z)
