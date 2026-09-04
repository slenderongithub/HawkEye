"""The action-classification network — shared by training and inference.

Design notes, each aimed at the "violence and fall get classified as each other"
failure mode:

  * **Multi-scale temporal convolution.** A fall is a ~0.5 s ballistic event
    (collapse, impact, stillness); violence is a multi-second oscillation. One
    convolution kernel cannot see both scales well, so the stem runs kernels of
    3 / 5 / 9 frames in parallel and concatenates them. The network gets a short
    view and a long view of the same window.

  * **BiGRU + hybrid pooling.** The recurrent layers model the *order* of events
    (upright → collapse → still is a fall; still → collapse → upright is someone
    standing up). Pooling concatenates attention-weighted, mean and max pooling:
    attention finds the decisive moment, mean captures sustained behaviour, max
    captures the single most extreme frame. Fights are sustained (mean matters);
    falls are peaky (max matters).

  * **A calibrated head by default.** The previous revision always used an
    ArcFace-style angular-margin head with scale s=30, which makes softmax
    outputs saturate at ~1.0 — so the ``action_min_conf`` threshold that gates
    every downstream trigger was meaningless. Here the default head is a plain
    linear classifier whose logits are divided by a learned/fitted
    ``temperature`` (fitted post-hoc on the validation split), giving
    probabilities that actually mean something. ``use_margin=True`` remains
    available for experiments.

Heavy torch import lives at module top, but this module is only imported lazily
(from the training script and from ActionClassifier.load()), so importing the
`hawkeye` package never requires torch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMargin(nn.Module):
    """Additive angular margin head (ArcFace). Enlarges inter-class margins.

    Note: its scaled-cosine logits are deliberately over-confident, so anything
    thresholding on softmax probability should prefer the linear head plus
    temperature scaling. Kept for ablation experiments.
    """

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.35):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cos = F.linear(F.normalize(x), F.normalize(self.weight)).clamp(-1 + 1e-6, 1 - 1e-6)
        if labels is None:                      # inference: plain scaled cosine logits
            return self.s * cos
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)      # push the true class by margin m
        onehot = F.one_hot(labels, cos.size(1)).float()
        return self.s * (onehot * target + (1 - onehot) * cos)


class MultiScaleStem(nn.Module):
    """Parallel temporal convolutions at three receptive fields."""

    def __init__(self, feat_dim: int, hidden: int, kernels=(3, 5, 9)):
        super().__init__()
        per = hidden // len(kernels)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feat_dim, per, k, padding=k // 2),
                nn.BatchNorm1d(per),
                nn.GELU(),
            ) for k in kernels
        ])
        merged = per * len(kernels)
        self.merge = nn.Sequential(
            nn.Conv1d(merged, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, F, T)
        return self.merge(torch.cat([b(x) for b in self.branches], dim=1))


class ActionNet(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int, hidden: int = 192,
                 embed: int = 192, use_margin: bool = False, dropout: float = 0.3):
        super().__init__()
        self.use_margin = use_margin
        self.stem = MultiScaleStem(feat_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.attn = nn.Sequential(
            nn.Linear(hidden * 2, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )
        # attention + mean + max pooling concatenated
        self.embed = nn.Sequential(
            nn.Linear(hidden * 6, embed), nn.BatchNorm1d(embed), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = ArcMargin(embed, num_classes) if use_margin else nn.Linear(embed, num_classes)
        # Post-hoc calibration factor, fitted on the validation split after
        # training and applied to logits at inference. 1.0 = uncalibrated.
        self.register_buffer("temperature", torch.ones(1))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Sequence (B, T, F) -> embedding (B, embed)."""
        h = self.stem(x.transpose(1, 2)).transpose(1, 2)        # (B, T, hidden)
        h, _ = self.gru(h)                                      # (B, T, 2*hidden)
        w = torch.softmax(self.attn(h).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = torch.cat([(h * w).sum(1), h.mean(1), h.max(1).values], dim=1)
        return self.embed(pooled)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.features(x)
        return self.head(emb, labels) if self.use_margin else self.head(emb)

    def calibrated_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Inference logits with temperature scaling applied."""
        return self.forward(x) / self.temperature.clamp(min=1e-2)


def build_model(feat_dim: int, num_classes: int, hidden: int = 192,
                embed: int = 192, use_margin: bool = False) -> ActionNet:
    return ActionNet(feat_dim, num_classes, hidden, embed, use_margin)
