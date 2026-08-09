"""Complete notebook-equivalent model."""

import torch
from torch import nn
from torch.nn import functional as F
from .temporal_compression import TemporalCompression
from .sequence_model import FlatSequenceModel, HierarchicalSequenceModel


class MLPClassifier(nn.Module):
    """Notebook MLP classifier mapping ``[B,D]`` to ``[B,K]`` logits."""
    def __init__(self, feature_dim, num_classes, hidden=(512, 256), use_bn=True, dropout=0.2):
        super().__init__(); layers=[]; current=feature_dim
        for width in hidden:
            layers.append(nn.Linear(current, width, bias=not use_bn))
            if use_bn: layers.append(nn.BatchNorm1d(width))
            layers.append(nn.GELU())
            if dropout and dropout > 0: layers.append(nn.Dropout(dropout))
            current=width
        layers.append(nn.Linear(current, num_classes)); self.net=nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


class SleepStagingModel(nn.Module):
    """Classify samples ``[B,W,T,C]`` and return a notebook-compatible dictionary."""
    def __init__(self, num_classes=5, in_channels=1, branch_out=32, kernels=(7,15,31),
                 mid_channels=(128,192,256), dilations=(1,2,4), strides=(2,5,5),
                 lstm_mode="hier", lstm_hidden=128, lstm_hidden_intra=64,
                 lstm_hidden_inter=128, fc_hidden=(512,256), fc_dropout=0.2,
                 fc_use_bn=True, use_se=True):
        super().__init__()
        self.compressor = TemporalCompression(in_channels, branch_out, kernels, mid_channels, dilations, strides, use_se)
        if lstm_mode == "flat":
            self.lstm_head = FlatSequenceModel(self.compressor.out_channels, lstm_hidden); feature_dim=2*lstm_hidden
        elif lstm_mode == "hier":
            self.lstm_head = HierarchicalSequenceModel(self.compressor.out_channels, lstm_hidden_intra, lstm_hidden_inter); feature_dim=2*lstm_hidden_inter
        else: raise ValueError("lstm_mode must be 'flat' or 'hier'")
        self.post_lstm_do = nn.Dropout(0.1)
        self.classifier = MLPClassifier(feature_dim, num_classes, fc_hidden, fc_use_bn, fc_dropout)

    @classmethod
    def from_config(cls, config: dict): return cls(**config)

    @torch.no_grad()
    def reset_state(self, batch_size=1, device="cpu"): self.lstm_head.reset_state(batch_size, device)

    def forward(self, x, sequence_start=True, return_attn=True):
        batch, windows, time, channels = x.shape
        # Treat each 5-second window as an independent convolutional sample.
        z, stem_se = self.compressor(x.view(batch * windows, time, channels).permute(0, 2, 1))
        features = self.lstm_head(z, batch, windows, sequence_start)
        features = F.layer_norm(self.post_lstm_do(features), features.shape[1:])
        return {"logits": self.classifier(features), "lstm_feat": features,
                "memberships": None, "rule_strengths": None, "attn": None,
                "stem_se": stem_se}


def count_trainable_parameters(model):
    """Return the number of trainable scalar parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
