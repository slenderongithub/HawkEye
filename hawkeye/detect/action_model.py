"""The action-classification network — shared by training and inference.

Two ideas here directly target the "violence vs fall get confused" problem:

  * Temporal model (Conv1d -> BiGRU -> attention pooling): looks at HOW the
    pose evolves over ~1.5s, not a single frame. A fall is a downward collapse
    then stillness; a fight is sustained high-frequency two-person motion. Only
    a temporal model can tell those trajectories apart.

  * Optional angular-margin head (ArcFace-style, `ArcMargin`): instead of a
    plain linear classifier, it forces each class's embeddings into a tighter
    angular cone with an explicit margin `m` between classes. This is literally
    "maximize the margin between classes" — the exact fix for two classes whose
    decision boundary is fuzzy. (Same technique the identity module uses for
    faces; here we reuse it for actions.) Toggle with `use_margin`.

Heavy torch import lives at module top, but this module is only imported lazily
(from the training script and from ActionClassifier.load()), so importing the
`hawkeye` package never requires torch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMargin(nn.Module):
    """Additive angular margin head (ArcFace). Enlarges inter-class margins."""

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


class ActionNet(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int, hidden: int = 128,
                 embed: int = 128, use_margin: bool = True):
        super().__init__()
        self.use_margin = use_margin
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.3)
        self.attn = nn.Linear(hidden * 2, 1)
        self.embed = nn.Sequential(nn.Linear(hidden * 2, embed), nn.BatchNorm1d(embed))
        self.head = ArcMargin(embed, num_classes) if use_margin else nn.Linear(embed, num_classes)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, F)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)       # (B, T, hidden)
        h, _ = self.gru(h)                                     # (B, T, 2*hidden)
        w = torch.softmax(self.attn(h).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (h * w).sum(1)                                # (B, 2*hidden)
        emb = self.embed(pooled)                               # (B, embed)
        return self.head(emb, labels) if self.use_margin else self.head(emb)


def build_model(feat_dim: int, num_classes: int, hidden: int = 128,
                embed: int = 128, use_margin: bool = True) -> ActionNet:
    return ActionNet(feat_dim, num_classes, hidden, embed, use_margin)
