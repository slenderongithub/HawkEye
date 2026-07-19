"""Zero-training keypoint heuristics for fall and fight/violence.

These are an honest *bootstrap*: geometry + motion that works on day one so the
whole pipeline is demoable immediately. They are designed to be swapped for the
trained ST-GCN skeleton classifier (scripts/train_action.py) behind the same
`analyze()` return contract: List[(AnomalyType, score, reason)].
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from ..events.schema import AnomalyType
from .pose import (L_HIP, L_SHOULDER, L_WRIST, R_HIP, R_SHOULDER, R_WRIST,
                   PersonDetection)

Candidate = Tuple[AnomalyType, float, str]


def _mid(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def _torso_horizontalness(p: PersonDetection) -> Optional[float]:
    """0 = upright, 1 = fully horizontal, using the shoulder->hip torso vector."""
    sh = _mid(p.kp(L_SHOULDER), p.kp(R_SHOULDER))
    hip = _mid(p.kp(L_HIP), p.kp(R_HIP))
    if sh is None or hip is None:
        return None
    v = hip - sh
    if np.linalg.norm(v) < 1e-3:
        return None
    angle_from_vertical = abs(math.degrees(math.atan2(abs(v[0]), abs(v[1]) + 1e-6)))
    return min(1.0, angle_from_vertical / 90.0)


def fall_score(p: PersonDetection) -> float:
    """Single-frame fall/lying likelihood in [0, 1]."""
    aspect = p.width / (p.height + 1e-6)          # >1 => wider than tall (lying)
    aspect_term = np.clip((aspect - 0.8) / 1.2, 0.0, 1.0)
    horiz = _torso_horizontalness(p)
    if horiz is None:
        return float(aspect_term * 0.7)           # geometry only
    return float(0.6 * horiz + 0.4 * aspect_term)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-6))


def _proximity(p1: PersonDetection, p2: PersonDetection) -> float:
    """1 when two people are essentially on top of each other, decaying with distance."""
    dist = float(np.linalg.norm(p1.centroid - p2.centroid))
    scale = (p1.height + p2.height) / 2.0 + 1e-6
    return float(np.exp(-dist / scale))


def _aggressive_pose(p: PersonDetection) -> float:
    """Wrists raised near/above shoulder height — a weak aggression/defense cue."""
    cues = 0
    n = 0
    for wi, si in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)):
        w, s = p.kp(wi), p.kp(si)
        if w is not None and s is not None:
            n += 1
            if w[1] <= s[1] + 0.1 * p.height:   # wrist at/above shoulder (image y grows down)
                cues += 1
    return (cues / n) if n else 0.0


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
                best_prox = max(best_prox, _proximity(persons[i], persons[j]))
                best_iou = max(best_iou, _box_iou(persons[i].box, persons[j].box))
            aggro = max(aggro, _aggressive_pose(persons[i]))
        closeness = max(best_prox, best_iou)
        violence = closeness * (0.35 + 0.65 * motion_score) + 0.15 * aggro
        violence = float(np.clip(violence, 0.0, 1.0))
        if violence >= 0.45:
            out.append((
                AnomalyType.VIOLENCE, violence,
                "two or more people in close contact with rapid motion",
            ))

    return out
