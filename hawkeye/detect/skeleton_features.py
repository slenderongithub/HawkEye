"""Shared skeleton feature extraction — used by BOTH training and inference.

Why this module exists, and the three rules it enforces
------------------------------------------------------

**Rule 1 — one implementation, both paths.** Training and inference MUST compute
identical features from identical code. A previous revision of this pipeline
passed ``motion=0.0`` during training and a real motion score at inference. The
standardizer then divided that dimension by its training std (≈0, floored to
1e-6), so a live motion value of 0.35 entered the network as ~350,000. The model
trained beautifully and behaved randomly in production. Everything here — including
``standardize()`` — is shared so that class of bug cannot recur, and the
standardizer floors variance and clips its output as a second line of defence.

**Rule 2 — features must make the classes separable by construction.** Violence
and falls both look like "limbs moving fast". What actually separates them is
*structure*, so we encode it explicitly:

  * A fall is ONE body on a **ballistic vertical trajectory** — the hip
    accelerates downward, the torso rotates toward horizontal, and then the body
    goes **still**. Downward velocity alone is not enough: sitting down fast has
    that too. Downward *acceleration followed by stillness* is the signature.
  * Violence is a **sustained two-body interaction** — two skeletons close
    together with high, oscillating mutual motion, wrists repeatedly closing on
    the other body's head/torso. Not "two people present" (that is just a queue),
    and not "one burst of motion" (that is a stumble).

  So the vector carries a second actor, the geometry *between* the two bodies
  (wrist→head and wrist→torso reach distances, closing speed), and explicit
  stillness/oscillation terms — not just one person's joints.

**Rule 3 — the model must know what it cannot see.** Keypoint confidences are
features. A zeroed occluded joint would otherwise read as "joint located exactly
at the hip", which is a lie the network will happily learn from.

Layout
------
Per-frame base block (``BASE_DIM``)::

    [A_KP]      34   actor A's 17 keypoints (x, y), hip-centred, torso-scaled
    [A_KPC]     17   actor A's per-keypoint confidence (occlusion awareness)
    [A_POSE]     9   actor A posture scalars (verticality, aspect, spread, ...)
    [B_KP]      34   actor B's keypoints in A's normalised frame (0 if alone)
    [B_POSE]     4   actor B presence + posture
    [INTER]     10   A↔B interaction geometry (reach, overlap, dominance)
    [SCENE]      4   person count, scene motion, crowd proximity/spread

``build_sequence()`` then appends the temporal block, which is where the fall /
violence trajectories actually live::

    [A_VEL]     34   actor A per-keypoint velocity
    [B_VEL]     34   actor B per-keypoint velocity
    [DYN]        8   hip vertical velocity & acceleration, torso angular rate,
                     A/B kinetic energy, closing speed, stillness, jerk

Final per-frame vector fed to the model: ``FEATURE_DIM``.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import geometry
from .pose import (L_ANKLE, L_EAR, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, NOSE,
                   R_ANKLE, R_EAR, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
                   PersonDetection)

N_KP = 17
KP_XY = N_KP * 2                      # 34

# --- base block slices -----------------------------------------------------
A_KP_OFF, A_KP_N = 0, KP_XY           # [0:34]
A_KPC_OFF, A_KPC_N = 34, N_KP         # [34:51]
A_POSE_OFF, A_POSE_N = 51, 9          # [51:60]
B_KP_OFF, B_KP_N = 60, KP_XY          # [60:94]
B_POSE_OFF, B_POSE_N = 94, 4          # [94:98]
INTER_OFF, INTER_N = 98, 10           # [98:108]
SCENE_OFF, SCENE_N = 108, 4           # [108:112]
BASE_DIM = 112

# --- temporal block slices (appended by build_sequence) --------------------
A_VEL_OFF, A_VEL_N = BASE_DIM, KP_XY              # [112:146]
B_VEL_OFF, B_VEL_N = BASE_DIM + KP_XY, KP_XY      # [146:180]
DYN_OFF, DYN_N = BASE_DIM + 2 * KP_XY, 8          # [180:188]
FEATURE_DIM = BASE_DIM + 2 * KP_XY + DYN_N        # 188

MIN_KP_CONF = 0.3
# Standardisation guards. MIN_STD stops a low-variance training dimension from
# becoming a divide-by-almost-zero amplifier at inference; CLIP bounds anything
# that still drifts out of the training distribution.
MIN_STD = 1e-2
STD_CLIP = 8.0

# Joints used as "strike targets" and "strike origins" for reach geometry.
_HEAD_KPS = (NOSE, L_EAR, R_EAR)
_TORSO_KPS = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
_LIMB_KPS = (L_WRIST, R_WRIST, L_ANKLE, R_ANKLE)


def _kp(p: PersonDetection, idx: int) -> Optional[np.ndarray]:
    x, y, c = p.keypoints[idx]
    return np.array([x, y], dtype=np.float32) if c >= MIN_KP_CONF else None


def _mid(a, b):
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def _salience(p: PersonDetection) -> float:
    """How much a person "owns" the frame: big and confidently detected."""
    return (p.width * p.height) * (0.5 + 0.5 * p.conf)


def _actor_frame(p: PersonDetection) -> Tuple[np.ndarray, float]:
    """Return (origin, scale) defining a person's normalised coordinate frame.

    Origin is the hip midpoint (falling back to the box centroid) and scale is
    the torso length (falling back to box height), so features are invariant to
    where the person is in frame and how far from the camera they are.
    """
    sh = _mid(_kp(p, L_SHOULDER), _kp(p, R_SHOULDER))
    hip = _mid(_kp(p, L_HIP), _kp(p, R_HIP))
    origin = hip if hip is not None else p.centroid.astype(np.float32)
    if sh is not None and hip is not None:
        scale = float(np.linalg.norm(sh - hip))
    else:
        scale = p.height
    return origin.astype(np.float32), max(scale, 1e-3)


def _norm_kps(p: PersonDetection, origin: np.ndarray, scale: float) -> np.ndarray:
    """17 keypoints expressed in the given normalised frame, flattened to 34."""
    out = np.zeros(KP_XY, dtype=np.float32)
    for i in range(N_KP):
        k = _kp(p, i)
        if k is not None:
            out[2 * i] = np.clip((k[0] - origin[0]) / scale, -6, 6)
            out[2 * i + 1] = np.clip((k[1] - origin[1]) / scale, -6, 6)
    return out


def _min_pair_distance(p_from: PersonDetection, from_idx: Sequence[int],
                       p_to: PersonDetection, to_idx: Sequence[int],
                       scale: float) -> float:
    """Smallest normalised distance between two joint groups on two people.

    This is the "reach" geometry that separates a punch from standing nearby:
    a strike drives a wrist to within roughly one torso-length of the other
    person's head or torso. Returns a large sentinel when either group is
    fully occluded, so "unknown" never reads as "touching".
    """
    best = 6.0
    for i in from_idx:
        a = _kp(p_from, i)
        if a is None:
            continue
        for j in to_idx:
            b = _kp(p_to, j)
            if b is None:
                continue
            best = min(best, float(np.linalg.norm(a - b)) / scale)
    return float(np.clip(best, 0.0, 6.0))


def _body_spread(p: PersonDetection, origin: np.ndarray, scale: float) -> float:
    """How far the limbs extend from the body centre — flailing vs compact."""
    d = [float(np.linalg.norm(k - origin)) / scale
         for k in (_kp(p, i) for i in _LIMB_KPS) if k is not None]
    return float(np.clip(max(d) if d else 0.0, 0.0, 6.0))


def _person_posture(p: PersonDetection, frame_h: int) -> Tuple[np.ndarray, Optional[float]]:
    """Posture scalars for one person + their torso verticality."""
    origin, scale = _actor_frame(p)
    sh = _mid(_kp(p, L_SHOULDER), _kp(p, R_SHOULDER))
    hip = _mid(_kp(p, L_HIP), _kp(p, R_HIP))
    horiz = geometry.torso_horizontalness(sh, hip)

    f = np.zeros(A_POSE_N, dtype=np.float32)
    f[0] = horiz if horiz is not None else 0.0
    f[1] = np.clip(p.width / (p.height + 1e-6), 0, 5)            # >1 = lying down
    f[2] = np.clip(p.height / max(frame_h, 1), 0, 1)
    f[3] = np.clip(origin[1] / max(frame_h, 1), 0, 1)            # 1 = at frame bottom
    nose = _kp(p, NOSE)
    # Head above hip: strongly positive when upright, negative when inverted.
    f[4] = np.clip((origin[1] - nose[1]) / scale, -6, 6) if nose is not None else 0.0
    f[5] = geometry.aggressive_pose(p)                            # wrists ≥ shoulders
    f[6] = _body_spread(p, origin, scale)
    knee = _mid(_kp(p, L_KNEE), _kp(p, R_KNEE))
    # Knees above hips => crouched or collapsed rather than standing.
    f[7] = np.clip((origin[1] - knee[1]) / scale, -6, 6) if knee is not None else 0.0
    f[8] = float(np.clip(p.conf, 0.0, 1.0))
    return f, horiz


def select_actors(persons: Sequence[PersonDetection]
                  ) -> Tuple[Optional[PersonDetection], Optional[PersonDetection]]:
    """Pick actor A (most salient) and actor B (A's nearest neighbour).

    B is chosen by proximity rather than salience because the second actor that
    matters is whoever A is *interacting with*, not whoever else is biggest.
    """
    if not persons:
        return None, None
    a = max(persons, key=_salience)
    others = [p for p in persons if p is not a]
    if not others:
        return a, None
    b = min(others, key=lambda p: float(np.linalg.norm(p.centroid - a.centroid)))
    return a, b


def frame_feature(persons: Sequence[PersonDetection], motion: float,
                  frame_w: int, frame_h: int) -> np.ndarray:
    """Compute the BASE_DIM feature vector for one frame.

    ``motion`` is the scene-motion score from ``MotionAnomaly.update()`` and MUST
    be the real value in both training and inference — see Rule 1 above.
    """
    f = np.zeros(BASE_DIM, dtype=np.float32)

    # --- scene block (valid even with nobody in frame) ---
    f[SCENE_OFF + 1] = float(np.clip(motion, 0.0, 1.0))
    if not persons:
        return f
    f[SCENE_OFF + 0] = min(len(persons), 6) / 6.0

    if len(persons) >= 2:
        prox = [geometry.proximity(persons[i], persons[j])
                for i in range(len(persons)) for j in range(i + 1, len(persons))]
        f[SCENE_OFF + 2] = float(np.mean(prox))
        centroids = np.stack([p.centroid for p in persons]).astype(np.float32)
        heights = float(np.mean([p.height for p in persons])) + 1e-6
        f[SCENE_OFF + 3] = float(np.clip(centroids.std(axis=0).mean() / heights, 0, 6))

    a, b = select_actors(persons)
    if a is None:
        return f

    # --- actor A ---
    origin, scale = _actor_frame(a)
    f[A_KP_OFF:A_KP_OFF + A_KP_N] = _norm_kps(a, origin, scale)
    f[A_KPC_OFF:A_KPC_OFF + A_KPC_N] = np.clip(a.keypoints[:, 2], 0.0, 1.0)
    pose_a, _ = _person_posture(a, frame_h)
    f[A_POSE_OFF:A_POSE_OFF + A_POSE_N] = pose_a

    if b is None:
        return f

    # --- actor B, expressed in A's frame so the pair geometry is explicit ---
    f[B_KP_OFF:B_KP_OFF + B_KP_N] = _norm_kps(b, origin, scale)
    pose_b, horiz_b = _person_posture(b, frame_h)
    f[B_POSE_OFF + 0] = 1.0                                   # B present
    f[B_POSE_OFF + 1] = horiz_b if horiz_b is not None else 0.0
    f[B_POSE_OFF + 2] = np.clip(b.width / (b.height + 1e-6), 0, 5)
    f[B_POSE_OFF + 3] = np.clip(b.height / (a.height + 1e-6), 0, 4)

    # --- A ↔ B interaction geometry ---
    i = INTER_OFF
    f[i + 0] = geometry.proximity(a, b)
    f[i + 1] = geometry.box_iou(a.box, b.box)
    f[i + 2] = np.clip(float(np.linalg.norm(a.centroid - b.centroid))
                       / ((a.height + b.height) / 2.0 + 1e-6), 0, 6)
    # Reach: how close each person's strike joints get to the other's head/torso.
    f[i + 3] = _min_pair_distance(a, (L_WRIST, R_WRIST), b, _TORSO_KPS, scale)
    f[i + 4] = _min_pair_distance(b, (L_WRIST, R_WRIST), a, _TORSO_KPS, scale)
    f[i + 5] = _min_pair_distance(a, (L_WRIST, R_WRIST), b, _HEAD_KPS, scale)
    f[i + 6] = _min_pair_distance(b, (L_WRIST, R_WRIST), a, _HEAD_KPS, scale)
    # Vertical dominance: one body low while the other stands over it.
    f[i + 7] = np.clip((b.centroid[1] - a.centroid[1])
                       / ((a.height + b.height) / 2.0 + 1e-6), -4, 4)
    f[i + 8] = float(pose_b[5])                               # B aggressive pose
    # Do they face each other? Shoulder-line alignment as a coarse orientation cue.
    sa = _kp(a, L_SHOULDER), _kp(a, R_SHOULDER)
    sb = _kp(b, L_SHOULDER), _kp(b, R_SHOULDER)
    if all(v is not None for v in sa + sb):
        va = sa[1] - sa[0]
        vb = sb[1] - sb[0]
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na > 1e-3 and nb > 1e-3:
            f[i + 9] = float(np.clip(np.dot(va / na, vb / nb), -1, 1))
    return f


def build_sequence(frame_feats: List[np.ndarray]) -> np.ndarray:
    """Turn a list of BASE_DIM frame features into a (T, FEATURE_DIM) sequence.

    Appends per-keypoint velocities for both actors plus the derived kinematics
    that carry the fall/violence distinction: a fall is downward hip
    *acceleration* followed by stillness; a fight is sustained, oscillating
    two-body motion with repeated closing of the reach distance.
    """
    if not frame_feats:
        return np.zeros((1, FEATURE_DIM), dtype=np.float32)

    base = np.stack(frame_feats).astype(np.float32)            # (T, BASE_DIM)
    t = base.shape[0]
    out = np.zeros((t, FEATURE_DIM), dtype=np.float32)
    out[:, :BASE_DIM] = base

    a_kp = base[:, A_KP_OFF:A_KP_OFF + A_KP_N]
    b_kp = base[:, B_KP_OFF:B_KP_OFF + B_KP_N]
    a_vel = np.zeros_like(a_kp)
    b_vel = np.zeros_like(b_kp)
    a_vel[1:] = a_kp[1:] - a_kp[:-1]
    b_vel[1:] = b_kp[1:] - b_kp[:-1]
    # Only credit B's velocity on frames where B actually exists, otherwise the
    # frame B appears/disappears reads as an enormous phantom movement.
    b_present = base[:, B_POSE_OFF]
    valid_b = (b_present[1:] > 0.5) & (b_present[:-1] > 0.5)
    b_vel[1:][~valid_b] = 0.0
    out[:, A_VEL_OFF:A_VEL_OFF + A_VEL_N] = np.clip(a_vel, -4, 4)
    out[:, B_VEL_OFF:B_VEL_OFF + B_VEL_N] = np.clip(b_vel, -4, 4)

    d = DYN_OFF
    # Hip height in frame → vertical velocity and acceleration. Positive velocity
    # means moving DOWN the image, i.e. toward the floor.
    hip_y = base[:, A_POSE_OFF + 3]
    hv = np.zeros(t, dtype=np.float32)
    hv[1:] = hip_y[1:] - hip_y[:-1]
    ha = np.zeros(t, dtype=np.float32)
    ha[1:] = hv[1:] - hv[:-1]
    out[:, d + 0] = np.clip(hv * 20.0, -4, 4)                  # scaled for range parity
    out[:, d + 1] = np.clip(ha * 100.0, -4, 4)                 # the ballistic fall cue

    # Torso angular rate: how fast the body is rotating toward horizontal.
    horiz = base[:, A_POSE_OFF + 0]
    dh = np.zeros(t, dtype=np.float32)
    dh[1:] = horiz[1:] - horiz[:-1]
    out[:, d + 2] = np.clip(dh * 20.0, -4, 4)

    # Kinetic energy of each actor's skeleton (mean joint speed).
    a_ke = np.linalg.norm(a_vel.reshape(t, N_KP, 2), axis=2).mean(axis=1)
    b_ke = np.linalg.norm(b_vel.reshape(t, N_KP, 2), axis=2).mean(axis=1)
    out[:, d + 3] = np.clip(a_ke * 5.0, 0, 4)
    out[:, d + 4] = np.clip(b_ke * 5.0, 0, 4)

    # Closing speed of the nearest reach distance — negative means a limb is
    # driving toward the other body (a strike), positive means separating.
    reach = np.minimum(base[:, INTER_OFF + 3], base[:, INTER_OFF + 5])
    dr = np.zeros(t, dtype=np.float32)
    dr[1:] = reach[1:] - reach[:-1]
    out[:, d + 5] = np.clip(dr * 10.0, -4, 4)

    # Stillness: trailing mean of actor A's kinetic energy over ~0.5 s. A fall
    # ends still; a fight does not. Computed causally so live and offline agree.
    win = min(10, t)
    csum = np.concatenate([[0.0], np.cumsum(a_ke)])
    idx = np.arange(t)
    lo = np.maximum(0, idx - win + 1)
    trailing = (csum[idx + 1] - csum[lo]) / np.maximum(idx + 1 - lo, 1)
    out[:, d + 6] = np.clip(trailing * 5.0, 0, 4)

    # Jerk of actor A's motion — impact spikes (landing, blows) rather than the
    # smooth ramps of ordinary movement.
    jerk = np.zeros(t, dtype=np.float32)
    jerk[1:] = np.abs(a_ke[1:] - a_ke[:-1])
    out[:, d + 7] = np.clip(jerk * 20.0, 0, 4)
    return out


def resample(seq: np.ndarray, length: int) -> np.ndarray:
    """Uniformly resample a (T, F) sequence to exactly `length` frames."""
    t = seq.shape[0]
    if t == length:
        return seq
    idx = np.linspace(0, t - 1, length).round().astype(int)
    return seq[idx]


def fit_standardizer(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (mean, std) over a (N, T, F) training tensor, variance-floored.

    The floor is the fix for the amplification bug described at the top of this
    module: a dimension that barely varies in training must never become a
    multiplier at inference.
    """
    flat = x.reshape(-1, x.shape[-1])
    mean = flat.mean(axis=0)
    std = np.maximum(flat.std(axis=0), MIN_STD)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(seq: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply the shared standardisation. Used identically by train and inference."""
    return np.clip((seq - mean) / std, -STD_CLIP, STD_CLIP).astype(np.float32)
