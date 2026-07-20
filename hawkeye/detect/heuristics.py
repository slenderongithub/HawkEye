"""Zero-training keypoint heuristics for fall and fight/violence.

These are an honest *bootstrap*: geometry + motion that works on day one so the
whole pipeline is demoable immediately. They are designed to be swapped for the
trained ST-GCN skeleton classifier (scripts/train_action.py) behind the same
`analyze()` return contract: List[(AnomalyType, score, reason)].
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..events.schema import AnomalyType
from . import geometry
from .pose import (L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, PersonDetection)

Candidate = Tuple[AnomalyType, float, str]


def fall_score(p: PersonDetection) -> float:
    """Single-frame fall/lying likelihood in [0, 1]."""
    aspect = p.width / (p.height + 1e-6)          # >1 => wider than tall (lying)
    aspect_term = np.clip((aspect - 0.8) / 1.2, 0.0, 1.0)
    sh = geometry.mid(p.kp(L_SHOULDER), p.kp(R_SHOULDER))
    hip = geometry.mid(p.kp(L_HIP), p.kp(R_HIP))
    horiz = geometry.torso_horizontalness(sh, hip)
    if horiz is None:
        return float(aspect_term * 0.7)           # geometry only
    return float(0.6 * horiz + 0.4 * aspect_term)


def analyze(persons: List[PersonDetection], motion_score: float) -> List[Candidate]:
    """Return anomaly candidates from the current frame's poses + scene motion."""
    out: List[Candidate] = []

    # --- Fall / medical (per person) ---
    best_fall, fall_person = 0.0, None
    for p in persons:
        s = fall_score(p)
        if s > best_fall:
            best_fall, fall_person = s, p
    if best_fall >= 0.55 and fall_person is not None:
        # A collapse with little scene motion => likely medical (person down, not moving).
        reason = "person in a horizontal/collapsed posture"
        out.append((AnomalyType.FALL, float(min(1.0, best_fall)), reason))

    # --- Fight / violence (needs >=2 people close + motion) ---
    if len(persons) >= 2:
        best_prox = 0.0
        best_iou = 0.0
        aggro = 0.0
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                best_prox = max(best_prox, geometry.proximity(persons[i], persons[j]))
                best_iou = max(best_iou, geometry.box_iou(persons[i].box, persons[j].box))
            aggro = max(aggro, geometry.aggressive_pose(persons[i]))
        closeness = max(best_prox, best_iou)
        violence = closeness * (0.35 + 0.65 * motion_score) + 0.15 * aggro
        violence = float(np.clip(violence, 0.0, 1.0))
        if violence >= 0.45:
            out.append((
                AnomalyType.VIOLENCE, violence,
                "two or more people in close contact with rapid motion",
            ))

    return out
