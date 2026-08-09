"""Flat and hierarchical recurrent sequence encoders."""

import torch
from torch import nn


class FlatSequenceModel(nn.Module):
    """Flatten compressed window steps and return ``[B,2H]``."""
    def __init__(self, feature_dim, hidden=128):
        super().__init__(); self.lstm = nn.LSTM(feature_dim, hidden, batch_first=True, bidirectional=True)
        self.hidden = None; self._batch = None

    @torch.no_grad()
    def reset_state(self, batch_size=1, device="cpu"):
        size = self.lstm.hidden_size
        self.hidden = (torch.zeros(2, batch_size, size, device=device), torch.zeros(2, batch_size, size, device=device))
        self._batch = batch_size

    def forward(self, z, batch_size, windows, sequence_start=False):
        _, features, steps = z.shape
        if self.hidden is None or self._batch != batch_size or self.hidden[0].device != z.device or sequence_start:
            self.reset_state(batch_size, z.device)
        sequence = z.permute(0, 2, 1).view(batch_size, windows, steps, features).reshape(batch_size, windows * steps, features)
        output, state = self.lstm(sequence, self.hidden)
        self.hidden = tuple(value.detach() for value in state)
        return output[:, -1, :]


class HierarchicalSequenceModel(nn.Module):
    """Model intra-window then inter-window context, returning ``[B,2*H_inter]``."""
    def __init__(self, feature_dim, hidden_intra=64, hidden_inter=128):
        super().__init__()
        self.lstm_intra = nn.LSTM(feature_dim, hidden_intra, batch_first=True, bidirectional=True)
        self.lstm_inter = nn.LSTM(2 * hidden_intra, hidden_inter, batch_first=True, bidirectional=True)
        self.hidden_inter = None; self._batch = None

    @torch.no_grad()
    def reset_state(self, batch_size=1, device="cpu"):
        size = self.lstm_inter.hidden_size
        self.hidden_inter = (torch.zeros(2, batch_size, size, device=device), torch.zeros(2, batch_size, size, device=device))
        self._batch = batch_size

    def forward(self, z, batch_size, windows, sequence_start=False):
        intra, _ = self.lstm_intra(z.permute(0, 2, 1))
        window_sequence = intra[:, -1, :].view(batch_size, windows, -1)
        if self.hidden_inter is None or self._batch != batch_size or self.hidden_inter[0].device != z.device or sequence_start:
            self.reset_state(batch_size, z.device)
        inter, state = self.lstm_inter(window_sequence, self.hidden_inter)
        self.hidden_inter = tuple(value.detach() for value in state)
        return inter[:, -1, :]
