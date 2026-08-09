"""Unused but notebook-defined additive temporal attention component."""

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    """Pool ``[B,S,D]`` into ``[B,D]`` and return weights ``[B,S]``."""
    def __init__(self, input_dim, attention_dim=64):
        super().__init__(); self.projection = nn.Linear(input_dim, attention_dim); self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x):
        weights = torch.softmax(self.score(torch.tanh(self.projection(x))).squeeze(-1), dim=1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1), weights
