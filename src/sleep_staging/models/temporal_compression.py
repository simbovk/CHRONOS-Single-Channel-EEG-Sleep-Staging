"""Residual temporal compression."""

from torch import nn
from .multiscale import MultiScaleFeatureExtractor, SqueezeExcitation1D


class ResidualBlock1D(nn.Module):
    """Length-preserving residual convolutions followed by optional max pooling."""
    def __init__(self, in_channels, out_channels, kernel=3, dilation=1, pool_stride=1):
        super().__init__()
        padding = (kernel // 2) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel, padding=padding,
                               dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel, padding=padding,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.short = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False), nn.BatchNorm1d(out_channels))
        self.pool = nn.Identity() if pool_stride == 1 else nn.MaxPool1d(pool_stride, pool_stride)

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        return self.pool(self.act(self.bn2(self.conv2(y)) + self.short(x)))


class TemporalCompression(nn.Module):
    """Compress each 500-sample window: ``[B*W,C,500] -> [B*W,256,5]``."""
    def __init__(self, in_channels=1, branch_out=32, kernels=(7, 15, 31),
                 mid_channels=(128, 192, 256), dilations=(1, 2, 4),
                 strides=(2, 5, 5), use_se=True):
        super().__init__()
        self.stem = MultiScaleFeatureExtractor(in_channels, branch_out, kernels, 2, use_se)
        channels = (self.stem.out_channels,) + tuple(mid_channels)
        kernels2 = (5, 3, 3)
        blocks = []
        for i in range(3):
            layers = [ResidualBlock1D(channels[i], channels[i + 1], kernels2[i],
                                      dilations[i], strides[i])]
            if use_se:
                layers.append(SqueezeExcitation1D(channels[i + 1]))
            blocks.append(nn.Sequential(*layers))
        # Attribute names match the notebook checkpoints exactly.
        self.block1, self.block2, self.block3 = blocks
        self.out_channels = mid_channels[-1]

    def forward(self, x):
        z, stem_weights = self.stem(x)
        for block in (self.block1, self.block2, self.block3):
            for layer in block:
                if isinstance(layer, SqueezeExcitation1D):
                    z, _ = layer(z)
                else:
                    z = layer(z)
        return z, stem_weights
