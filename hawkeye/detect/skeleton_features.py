"""Shared skeleton feature extraction — used by BOTH training and inference.

Why this module exists (and why it's the key to fixing violence/fall confusion):

  1. A classifier can only separate classes as well as its features let it.
     Raw keypoints don't separate "fall" from "fight" well because both involve
     limbs moving fast. So we engineer features that make the distinction
     *explicit*: fall = ONE person going horizontal with a dropping hip; fight =
     TWO+ people, close together, with high mutual motion. When the signal that
     tells the classes apart is a literal feature dimension, the decision
     boundary (and its margin) is far cleaner.

  2. Training and inference MUST compute the exact same features from the same
     code, or the model that trained beautifully will behave randomly live.
     Keeping extraction in one module guarantees that.

Feature layout per frame (BASE_DIM = 44):
    [0:34]   dominant person's 17 keypoints as (x,y), centered on the hip and
             scaled by torso length -> translation & scale invariant
    [34]     torso verticality        (0 upright .. 1 horizontal)   <- fall cue
    [35]     bbox aspect ratio  w/h    (>1 = lying down)            <- fall cue
    [36]     bbox height / frame_h
    [37]     hip height in frame       (low in frame => on the ground) <- fall cue
    [38]     head-above-hip distance   (negative => head below hip)  <- fall cue
    [39]     aggressive pose           (wrists at/above shoulders)   <- fight cue
    [40]     person count (scaled)                                   <- fight cue
    [41]     closest-pair proximity    (1 = overlapping)             <- fight cue
    [42]     max pairwise box IoU                                    <- fight cue
    [43]     scene motion energy

`build_sequence()` then appends per-frame keypoint velocity (34 dims), so the
final per-frame vector fed to the model is FEATURE_DIM = 78.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from . import geometry
from .pose import L_HIP, L_SHOULDER, NOSE, R_HIP, R_SHOULDER, PersonDetection

N_KP = 17
KP_XY = N_KP * 2          # 34
BASE_DIM = KP_XY + 10     # 44
FEATURE_DIM = KP_XY + BASE_DIM  # 78 (velocity of keypoints appended)
MIN_KP_CONF = 0.3


def _kp(p: PersonDetection, idx: int):
    x, y, c = p.keypoints[idx]
    return (np.array([x, y], dtype=np.float32) if c >= MIN_KP_CONF else None)


def _mid(a, b):
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def _dominant(persons: Sequence[PersonDetection]) -> PersonDetection:
    return max(persons, key=lambda p: (p.width * p.height) * (0.5 + 0.5 * p.conf))


def frame_feature(persons: Sequence[PersonDetection], motion: float,
                  frame_w: int, frame_h: int) -> np.ndarray:
    """Compute the BASE_DIM feature vector for one frame."""
    f = np.zeros(BASE_DIM, dtype=np.float32)
    f[43] = float(np.clip(motion, 0.0, 1.0))
    if not persons:
        return f

    p = _dominant(persons)
    sh = _mid(_kp(p, L_SHOULDER), _kp(p, R_SHOULDER))
    hip = _mid(_kp(p, L_HIP), _kp(p, R_HIP))
    center = hip if hip is not None else p.centroid
    # scale = torso length, robust fallbacks so we never divide by ~0
    if sh is not None and hip is not None:
        scale = float(np.linalg.norm(sh - hip))
    else:
        scale = p.height
    scale = max(scale, 1e-3)

    # [0:34] normalized keypoints
    for i in range(N_KP):
        k = _kp(p, i)
        if k is not None:
            f[2 * i] = np.clip((k[0] - center[0]) / scale, -4, 4)
            f[2 * i + 1] = np.clip((k[1] - center[1]) / scale, -4, 4)

    # [34] torso verticality (0 upright .. 1 horizontal)
    horiz = geometry.torso_horizontalness(sh, hip)
    if horiz is not None:
        f[34] = horiz
    # [35] aspect ratio (wide => lying)
    f[35] = np.clip(p.width / (p.height + 1e-6), 0, 5)
    # [36] normalized height
    f[36] = np.clip(p.height / max(frame_h, 1), 0, 1)
    # [37] hip vertical position in frame (1 = bottom of frame => on ground)
    f[37] = np.clip(center[1] / max(frame_h, 1), 0, 1)
    # [38] head above hip (positive normal, negative => head below hip = fallen)
    nose = _kp(p, NOSE)
    if nose is not None and hip is not None:
        f[38] = np.clip((hip[1] - nose[1]) / scale, -4, 4)
    # [39] aggressive pose: wrists at/above shoulder height
    f[39] = geometry.aggressive_pose(p)

    # [40] person count (scaled), [41] proximity, [42] IoU
    f[40] = min(len(persons), 5) / 5.0
    if len(persons) >= 2:
        best_prox = best_iou = 0.0
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                pi, pj = persons[i], persons[j]
                best_prox = max(best_prox, geometry.proximity(pi, pj))
                best_iou = max(best_iou, geometry.box_iou(pi.box, pj.box))
        f[41] = best_prox
        f[42] = best_iou
    return f


def build_sequence(frame_feats: List[np.ndarray]) -> np.ndarray:
    """Turn a list of BASE_DIM frame features into a (T, FEATURE_DIM) sequence,
    appending per-frame keypoint velocity so the model sees explicit motion
    (the strongest violence cue)."""
    if not frame_feats:
        return np.zeros((1, FEATURE_DIM), dtype=np.float32)
    base = np.stack(frame_feats).astype(np.float32)          # (T, BASE_DIM)
    kp = base[:, :KP_XY]
    vel = np.zeros_like(kp)
    vel[1:] = kp[1:] - kp[:-1]
    return np.concatenate([base, vel], axis=1)               # (T, FEATURE_DIM)


def resample(seq: np.ndarray, length: int) -> np.ndarray:
    """Uniformly resample a (T, F) sequence to exactly `length` frames."""
    t = seq.shape[0]
    if t == length:
        return seq
    idx = np.linspace(0, t - 1, length).round().astype(int)
    return seq[idx]
