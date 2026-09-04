"""Train the HawkEye action classifier (violence / fall / normal).

This produces the weights that REPLACE the zero-training heuristics in Stage A.
It exists to fix one specific failure: *violence and falls being classified as
each other in the real world while the training run reported near-perfect
numbers*. The previous pipeline reported ~0.95 validation macro-F1 and behaved
randomly on a live camera. Three causes, all addressed here:

1. **Window-level data leakage (the reason the old numbers were fiction).**
   Windows were cut from each clip with 50% overlap and then split *randomly*,
   so window [0:32] landed in train while [16:48] — sharing half its frames —
   landed in validation. The model memorised clips and validation applauded.
   Here the split is **grouped by source clip**: every window from one video goes
   entirely to train, val, or test. Never within a clip.

2. **Train/serve feature skew.** Training passed ``motion=0.0`` while inference
   passed a real motion score; standardisation then amplified that dimension by
   ~10^5 at inference. Motion is now computed here by the *same* ``MotionAnomaly``
   class the live pipeline uses, and ``skeleton_features.standardize()`` (shared
   by both paths) floors variance and clips its output.

3. **Meaningless confidences.** The angular-margin head saturated softmax at
   ~1.0, so the ``action_min_conf`` gate downstream was decorative. Training now
   fits a **temperature** on the validation split and selects **per-class
   thresholds** at a target precision, both saved into the checkpoint.

It also reports what actually matters instead of a single averaged score:
a per-source breakdown (to catch the model learning *which dataset* a clip came
from rather than what happens in it), the violence↔fall confusion rate as a
headline number, and a final **test** set touched exactly once.

Dataset layout (see docs/training.md):
    <root>/<label>/<clip>.mp4                 e.g. data/datasets/actions/fall/x.mp4
    <root>/<label>/<source>/<clip>.mp4        source subfolder = dataset name,
                                              used for the per-source breakdown

Run (Apple Silicon):
    python scripts/train_action.py --data data/datasets/actions --epochs 40
    # weights -> models/action_gru.pt ; then set detect.action_model in the config.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hawkeye.detect.skeleton_features import (A_KP_OFF, A_KPC_OFF, A_VEL_OFF,  # noqa: E402
                                              B_KP_OFF, B_POSE_OFF, B_VEL_OFF,
                                              BASE_DIM, FEATURE_DIM, INTER_OFF,
                                              KP_XY, N_KP, build_sequence,
                                              fit_standardizer, frame_feature,
                                              resample, standardize)

log = logging.getLogger("train_action")

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
# COCO left/right mirror map for horizontal-flip augmentation.
MIRROR = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])


# --------------------------------------------------------------------------- #
# Feature extraction (shared logic with inference via skeleton_features)
# --------------------------------------------------------------------------- #
def extract_clip(video_path: str, pose, max_frames: int = 1200,
                 batch: int = 32) -> np.ndarray:
    """Run YOLO-pose + motion over a clip -> (T, BASE_DIM) per-frame features.

    ``MotionAnomaly`` is the exact class the live pipeline uses, so the motion
    dimension carries the same distribution at train and inference time. It is
    stateful (EMA baseline), so it is constructed fresh per clip — matching a
    camera that has just started streaming. Because it is stateful, motion is
    updated strictly in frame order; only **pose** is batched.

    Pose runs ``batch`` frames per forward pass. This phase dominates the runtime
    of the whole script — millions of frames across the dataset — and one
    ``predict()`` per frame leaves the GPU idle between launches.
    """
    import cv2

    from hawkeye.detect.scene_anomaly import MotionAnomaly

    cap = cv2.VideoCapture(video_path)
    motion = MotionAnomaly()
    feats: list = []
    pending: list = []          # (frame, motion_score, w, h) awaiting a pose pass
    n = 0

    def flush() -> bool:
        """Pose the buffered frames and append their features. False on failure."""
        if not pending:
            return True
        try:
            detections = pose.detect_batch([p[0] for p in pending])
            for persons, (_, m, w, h) in zip(detections, pending):
                feats.append(frame_feature(persons, m, w, h))
        except Exception as e:  # noqa: BLE001 - keep the frames we already have
            log.warning("feature extraction failed near frame %d of %s: %s",
                        len(feats), video_path, e)
            pending.clear()     # drop the bad batch so a retry cannot re-raise it
            return False
        pending.clear()
        return True

    while n < max_frames:
        try:
            ok, frame = cap.read()
        except Exception as e:
            log.warning("frame read failed in %s after %d frames: %s", video_path, n, e)
            break
        if not ok or frame is None:
            break
        try:
            h, w = frame.shape[:2]
            pending.append((frame, motion.update(frame), w, h))   # motion: in order
        except Exception as e:
            log.warning("motion update failed on frame %d of %s: %s", video_path, n, e)
            break
        n += 1
        if len(pending) >= batch and not flush():
            break
    flush()          # the last partial batch (end-of-video exits via break)
    cap.release()
    if not feats:
        return np.zeros((0, BASE_DIM), dtype=np.float32)
    return np.stack(feats).astype(np.float32)


def build_dataset(root: str, cache: str, max_frames: int = 1200, extract_batch: int = 32):
    """Extract per-clip features. Returns (clips, y, sources, labels).

    Per-*clip* feature arrays are cached rather than materialised windows: it is
    an order of magnitude less memory, and it lets each epoch draw slightly
    different windows through augmentation.
    """
    cache_p = Path(cache) if cache else None
    if cache_p and cache_p.exists():
        d = np.load(cache_p, allow_pickle=True)
        clips = list(d["clips"])
        print(f"Loaded cached features from {cache_p}: {len(clips)} clips")
        # Back to plain str: numpy round-trips these as np.str_, which then becomes
        # the key type in the JSON report and the label list in the checkpoint.
        return (clips, d["y"], [str(s) for s in d["sources"]],
                [str(l) for l in d["labels"]])

    from hawkeye.detect.pose import PoseDetector
    pose = PoseDetector().load()

    root_p = Path(root)
    labels = sorted([d.name for d in root_p.iterdir() if d.is_dir()])
    if not labels:
        sys.exit(f"No <label>/ folders under {root}. Expected <root>/<label>/<clip>.mp4")
    print(f"Classes: {labels}")

    clips, y, sources = [], [], []
    for li, lab in enumerate(labels):
        # sorted(): rglob order is filesystem-dependent, and clip *indices* are
        # what group_split() partitions. Without a stable order the same dataset
        # produces different splits on macOS and Windows, so two machines' numbers
        # would not be comparable.
        found = sorted(c for c in (root_p / lab).rglob("*")
                       if c.is_file() and c.suffix.lower() in VIDEO_EXT)
        print(f"  [{lab}] {len(found)} clips")
        for ci, clip in enumerate(found):
            feats = extract_clip(str(clip), pose, max_frames, extract_batch)
            if len(feats) < 8:
                log.warning("skipping %s (only %d usable frames)", clip, len(feats))
                continue
            # Source = the subfolder under <label>/ if there is one, else "root".
            rel = clip.relative_to(root_p / lab)
            clips.append(feats)
            y.append(li)
            sources.append(rel.parts[0] if len(rel.parts) > 1 else "root")
            print(f"    {lab} {ci + 1}/{len(found)}", end="\r", flush=True)
        print()

    if not clips:
        sys.exit("No usable clips found — check the dataset layout.")
    y = np.array(y, dtype=np.int64)
    if cache_p:
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_p, clips=np.array(clips, dtype=object), y=y,
                            sources=np.array(sources), labels=np.array(labels))
        print(f"Cached features -> {cache_p}")
    return clips, y, sources, labels


def cap_per_class(clips, y, sources, labels, max_per_class: int, seed: int):
    """Keep at most ``max_per_class`` clips of each class (deterministic).

    This is how you run the balanced-vs-imbalanced experiment *without* building a
    second copy of the dataset: the cap is applied to the already-extracted
    features, so both runs share one ``--cache``. Pose extraction is the expensive
    part of this script by orders of magnitude — it should happen once, not once
    per variant.

    Sampling is seeded, and ``build_dataset`` sorts clips, so a given
    (seed, max_per_class) selects the same clips on any machine.
    """
    if not max_per_class:
        return clips, y, sources

    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for c in range(len(labels)):
        idx = np.where(y == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.extend(int(i) for i in idx)
    keep.sort()

    y2 = y[keep]
    dist = {labels[c]: int((y2 == c).sum()) for c in range(len(labels))}
    print(f"  capped to <={max_per_class} clips/class -> {len(keep)} clips  {dist}")
    return [clips[i] for i in keep], y2, [sources[i] for i in keep]


# --------------------------------------------------------------------------- #
# Grouped split — the leakage fix
# --------------------------------------------------------------------------- #
def group_split(y: np.ndarray, val_split: float, test_split: float, seed: int):
    """Split CLIP indices (never windows) into train/val/test, stratified by class.

    Windows from one clip share frames, so splitting windows randomly leaks the
    validation set into training. Splitting whole clips is what makes the
    reported numbers mean something.
    """
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_te = max(1, int(round(n * test_split))) if n >= 3 else 0
        n_va = max(1, int(round(n * val_split))) if n - n_te >= 2 else 0
        te.extend(idx[:n_te])
        va.extend(idx[n_te:n_te + n_va])
        tr.extend(idx[n_te + n_va:])
    return np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))


def window_index(clips, clip_ids, seq_len: int, stride: int):
    """Enumerate (clip_idx, start) windows for the given clips."""
    out = []
    for ci in clip_ids:
        t = len(clips[ci])
        if t >= seq_len:
            for s in range(0, t - seq_len + 1, max(1, stride)):
                out.append((int(ci), int(s)))
        else:
            out.append((int(ci), -1))        # short clip: resample the whole thing
    return out


# --------------------------------------------------------------------------- #
# Augmentation — camera-geometry robustness, so the model learns actions and
# not which dataset a clip came from.
# --------------------------------------------------------------------------- #
def _kp_blocks(seq: np.ndarray):
    """Offsets of every (x, y) keypoint block in a sequence."""
    return (A_KP_OFF, B_KP_OFF, A_VEL_OFF, B_VEL_OFF)


def flip_sequence(seq: np.ndarray) -> np.ndarray:
    """Horizontal mirror: negate x and swap left/right joints in every block."""
    out = seq.copy()
    for off in _kp_blocks(seq):
        kp = seq[:, off:off + KP_XY]
        new = np.empty_like(kp)
        for i in range(N_KP):
            m = MIRROR[i]
            new[:, 2 * i] = -kp[:, 2 * m]
            new[:, 2 * i + 1] = kp[:, 2 * m + 1]
        out[:, off:off + KP_XY] = new
    # Keypoint confidences follow the same left/right swap.
    conf = seq[:, A_KPC_OFF:A_KPC_OFF + N_KP]
    out[:, A_KPC_OFF:A_KPC_OFF + N_KP] = conf[:, MIRROR]
    return out


def rotate_sequence(seq: np.ndarray, deg: float) -> np.ndarray:
    """Rotate keypoints about the body origin — simulates camera tilt/roll."""
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    out = seq.copy()
    for off in _kp_blocks(seq):
        blk = out[:, off:off + KP_XY].reshape(-1, N_KP, 2)
        x, y = blk[..., 0].copy(), blk[..., 1].copy()
        blk[..., 0] = c * x - s * y
        blk[..., 1] = s * x + c * y
        out[:, off:off + KP_XY] = blk.reshape(-1, KP_XY)
    return out


def scale_sequence(seq: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Anisotropic scaling — simulates viewpoint foreshortening."""
    out = seq.copy()
    for off in _kp_blocks(seq):
        blk = out[:, off:off + KP_XY].reshape(-1, N_KP, 2)
        blk[..., 0] *= sx
        blk[..., 1] *= sy
        out[:, off:off + KP_XY] = blk.reshape(-1, KP_XY)
    return out


def drop_actor_b(seq: np.ndarray) -> np.ndarray:
    """Erase the second actor.

    Without this the network can shortcut to "two skeletons ⇒ violence", which is
    exactly how a bystander standing over a collapsed person becomes an assault
    alert. Forcing it to sometimes decide with one body present makes it rely on
    kinematics instead of head count.
    """
    out = seq.copy()
    out[:, B_KP_OFF:B_KP_OFF + KP_XY] = 0.0
    out[:, B_VEL_OFF:B_VEL_OFF + KP_XY] = 0.0
    out[:, B_POSE_OFF:B_POSE_OFF + 4] = 0.0
    out[:, INTER_OFF:INTER_OFF + 10] = 0.0
    return out


def augment(seq: np.ndarray, rng) -> np.ndarray:
    if rng.random() < 0.4 and seq.shape[0] > 8:              # temporal crop / speed
        crop = max(8, int(seq.shape[0] * rng.uniform(0.7, 1.0)))
        start = rng.integers(0, seq.shape[0] - crop + 1)
        seq = resample(seq[start:start + crop], seq.shape[0])
    if rng.random() < 0.5:
        seq = flip_sequence(seq)
    if rng.random() < 0.5:                                    # camera roll
        seq = rotate_sequence(seq, float(rng.uniform(-15, 15)))
    if rng.random() < 0.4:                                    # viewpoint squash
        seq = scale_sequence(seq, float(rng.uniform(0.85, 1.15)),
                             float(rng.uniform(0.85, 1.15)))
    if rng.random() < 0.15:
        seq = drop_actor_b(seq)
    if rng.random() < 0.5:                                    # joint localisation noise
        seq = seq.copy()
        for off in (A_KP_OFF, B_KP_OFF):
            seq[:, off:off + KP_XY] += rng.normal(
                0, 0.03, (seq.shape[0], KP_XY)).astype(np.float32)
    if rng.random() < 0.4:                                    # occlusion
        seq = seq.copy()
        for i in np.where(rng.random(N_KP) < 0.15)[0]:
            for off in (A_KP_OFF, B_KP_OFF, A_VEL_OFF, B_VEL_OFF):
                seq[:, off + 2 * i:off + 2 * i + 2] = 0.0
            seq[:, A_KPC_OFF + i] = 0.0                       # and say it's missing
    return seq


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def confusion(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def prf(cm):
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return prec, rec, f1


def print_report(cm, labels, indent="  "):
    prec, rec, f1 = prf(cm)
    w = max(len(l) for l in labels) + 2
    print(f"{indent}confusion (rows=true, cols=pred):")
    print(f"{indent}  " + "".join(f"{l[:8]:>9}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{indent}  {l[:w]:<{w}}" + "".join(f"{cm[i, j]:>9}" for j in range(len(labels))))
    print(f"{indent}per-class  precision / recall / F1:")
    for i, l in enumerate(labels):
        print(f"{indent}  {l:<{w}} {prec[i]:.3f} / {rec[i]:.3f} / {f1[i]:.3f}")
    return float(f1.mean())


def cross_confusion_rate(cm, labels, a: str, b: str):
    """Fraction of true-`a` windows predicted as `b` (and vice versa).

    This is the number this whole pipeline exists to drive down, so it gets
    printed on its own rather than buried inside a macro average.
    """
    if a not in labels or b not in labels:
        return None
    ia, ib = labels.index(a), labels.index(b)
    a_as_b = cm[ia, ib] / max(cm[ia].sum(), 1)
    b_as_a = cm[ib, ia] / max(cm[ib].sum(), 1)
    return float(a_as_b), float(b_as_a)


# --------------------------------------------------------------------------- #
# Calibration + operating thresholds
# --------------------------------------------------------------------------- #
T_MIN, T_MAX = 0.5, 10.0


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Fit a single temperature T minimising NLL of softmax(logits / T).

    Without this the raw logits are over-confident and every downstream
    probability threshold is arbitrary.

    The result is clamped to [T_MIN, T_MAX]. On a validation split the model
    separates cleanly, NLL is minimised by driving T→0, which *sharpens* softmax
    into saturation — reintroducing the exact "every probability is 1.0, so the
    confidence gate does nothing" problem temperature scaling is here to fix.
    Calibration may soften confidence; it may not become an amplifier.
    """
    import torch

    lg = torch.from_numpy(logits).float()
    ty = torch.from_numpy(y).long()
    log_t = torch.zeros(1, requires_grad=True)          # optimise log T for positivity
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=80)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(lg / log_t.exp().clamp(min=1e-3), ty)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.exp().item())
    if not math.isfinite(t):
        return 1.0
    clamped = float(np.clip(t, T_MIN, T_MAX))
    if abs(clamped - t) > 1e-6:
        print(f"  (temperature {t:.3f} clamped to {clamped:.3f} — validation is "
              "near-separable, so calibration carries little information)")
    return clamped


def pick_thresholds(probs: np.ndarray, y: np.ndarray, labels, target_precision: float):
    """Per-class probability threshold: the lowest one reaching target precision.

    A single global ``min_conf`` cannot express "a false police dispatch costs
    more than a false ambulance", so each class gets its own operating point.
    Falls back to the best-F1 threshold when the precision target is unreachable.
    """
    out = {}
    for k, lab in enumerate(labels):
        p = probs[:, k]
        pos = (y == k)
        if not pos.any():
            out[lab] = 0.5
            continue
        best_f1, best_f1_t = -1.0, 0.5
        best_ok_f1, best_ok_t = -1.0, None
        for t in np.arange(0.05, 0.96, 0.01):
            pred = p >= t
            tp = float((pred & pos).sum())
            fp = float((pred & ~pos).sum())
            fn = float((~pred & pos).sum())
            if tp == 0:
                continue
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            if f1 > best_f1:
                best_f1, best_f1_t = f1, float(t)
            # Among thresholds that meet the precision target, take the best-F1
            # one rather than the lowest. When validation separates cleanly every
            # threshold meets the target, and "lowest" would hand the live system
            # a hair-trigger 0.05 gate.
            if prec >= target_precision and f1 > best_ok_f1:
                best_ok_f1, best_ok_t = f1, float(t)
        out[lab] = best_ok_t if best_ok_t is not None else best_f1_t
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    from hawkeye.detect.action_model import build_model
    from hawkeye.train.device_profile import get_profile

    clips, y, sources, labels = build_dataset(args.data, args.cache, args.max_frames,
                                              args.extract_batch)
    clips, y, sources = cap_per_class(clips, y, sources, labels,
                                      args.max_per_class, args.seed)
    n_classes = len(labels)

    tr_ids, va_ids, te_ids = group_split(y, args.val_split, args.test_split, args.seed)
    print(f"\nGrouped split by source clip (no window leakage):")
    for name, ids in (("train", tr_ids), ("val", va_ids), ("test", te_ids)):
        dist = {labels[c]: int((y[ids] == c).sum()) for c in range(n_classes)} if len(ids) else {}
        print(f"  {name:<5} {len(ids):>4} clips  {dist}")
    if len(va_ids) == 0 or len(te_ids) == 0:
        sys.exit("Not enough clips per class for a grouped train/val/test split. "
                 "Add more clips (at least 3 per class, ideally dozens).")

    tr_win = window_index(clips, tr_ids, args.seq_len, args.window_stride)
    va_win = window_index(clips, va_ids, args.seq_len, args.seq_len // 2)
    te_win = window_index(clips, te_ids, args.seq_len, args.seq_len // 2)
    print(f"  windows: train={len(tr_win)} val={len(va_win)} test={len(te_win)}")

    profile = get_profile(args.device)

    # A window is a pure function of (clip, start), yet it is currently rebuilt on
    # every access: once per epoch for training, and again for the val pass each
    # epoch. Over a 40-epoch run that is dozens of identical build_sequence() calls
    # per window, on the main thread, with the GPU waiting on them. Cache the built
    # windows when they fit in the budget.
    #
    # Only when the loader runs in-process: with forked workers each worker would
    # fill its own copy, multiplying the memory by num_workers.
    win_bytes = args.seq_len * FEATURE_DIM * 4
    n_win = len(tr_win) + len(va_win) + len(te_win)
    want_gb = win_bytes * n_win / (1 << 30)
    use_wcache = profile.num_workers == 0 and want_gb <= args.window_cache_gb
    wcache: dict[tuple[int, int], np.ndarray] = {}
    print(f"  window cache: {'ON' if use_wcache else 'off'} "
          f"({n_win} windows ≈ {want_gb:.2f} GB, budget {args.window_cache_gb:.1f} GB"
          + ("" if profile.num_workers == 0 else f", {profile.num_workers} loader workers")
          + ")")

    def make_window(ci: int, start: int) -> np.ndarray:
        if use_wcache:
            hit = wcache.get((ci, start))
            if hit is not None:
                return hit
        feats = clips[ci]
        if start < 0:
            seq = resample(build_sequence(list(feats)), args.seq_len)
        else:
            seq = build_sequence(list(feats[start:start + args.seq_len]))
        if use_wcache:
            wcache[(ci, start)] = seq
        # Safe to hand out the cached array: augment() copies before it mutates and
        # standardize() returns a new array, so no caller writes through this.
        return seq

    # Standardiser fitted on TRAIN windows only, with the variance floor that
    # prevents a near-constant dimension from exploding at inference.
    sample = tr_win if len(tr_win) <= 4000 else [tr_win[i] for i in
                                                 np.linspace(0, len(tr_win) - 1, 4000).astype(int)]
    fit_stack = np.stack([make_window(ci, s) for ci, s in sample])
    mean, std = fit_standardizer(fit_stack)
    del fit_stack

    dev = profile.device
    batch = args.batch if args.batch is not None else profile.batch_size

    if profile.is_cuda:
        # Throughput switches, set once. cudnn.benchmark pays off because every
        # batch has the same shape (fixed seq_len), so the algorithm search is
        # amortised instead of repeated. TF32 (Ampere and newer, Blackwell
        # included) trades matmul mantissa bits for speed — the standard training
        # default; use --no-tf32 for a bit-comparable run against a CPU/MPS one.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
        torch.backends.cudnn.allow_tf32 = not args.no_tf32

    class Windows(Dataset):
        def __init__(self, idx, aug, seed):
            self.idx, self.aug = idx, aug
            self.rng = np.random.default_rng(seed)

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            ci, start = self.idx[i]
            seq = make_window(ci, start)
            if self.aug:
                seq = augment(seq, self.rng)
            seq = standardize(seq, mean, std)
            return torch.from_numpy(np.ascontiguousarray(seq)).float(), int(y[ci])

    tr_dl = DataLoader(Windows(tr_win, True, 1), batch_size=batch, shuffle=True,
                       num_workers=profile.num_workers, pin_memory=profile.pin_memory,
                       drop_last=len(tr_win) > batch)
    va_dl = DataLoader(Windows(va_win, False, 2), batch_size=batch,
                       num_workers=profile.num_workers, pin_memory=profile.pin_memory)
    te_dl = DataLoader(Windows(te_win, False, 3), batch_size=batch,
                       num_workers=profile.num_workers, pin_memory=profile.pin_memory)

    use_margin = args.margin
    model = build_model(FEATURE_DIM, n_classes, hidden=profile.hidden,
                        embed=profile.embed, use_margin=use_margin).to(dev)

    counts = np.array([int((y[tr_ids] == c).sum()) for c in range(n_classes)])
    cw = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32)
    cw = (cw / cw.sum() * n_classes).to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs * max(1, len(tr_dl))),
        pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda", enabled=profile.use_amp)

    print(f"\ndevice={dev}  amp={profile.use_amp}  batch={batch}  hidden={profile.hidden}  "
          f"margin={use_margin}  classes={n_classes}  feat_dim={FEATURE_DIM}  "
          f"seq_len={args.seq_len}")

    def evaluate(dl):
        model.eval()
        logits, trues = [], []
        with torch.no_grad():
            for xb, yb in dl:
                out = model(xb.to(dev))
                logits.append(out.float().cpu().numpy())
                trues.append(yb.numpy())
        return np.concatenate(logits), np.concatenate(trues)

    best_f1, best_state, patience = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for xb, yb in tr_dl:
            xb = xb.to(dev, non_blocking=profile.pin_memory)
            yb = yb.to(dev, non_blocking=profile.pin_memory)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=profile.use_amp):
                out = model(xb, yb) if use_margin else model(xb)
                loss = loss_fn(out, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()

        lg, tv = evaluate(va_dl)
        cm = confusion(tv, lg.argmax(1), n_classes)
        print(f"\nepoch {ep + 1}/{args.epochs}  train_loss={run / max(1, len(tr_dl)):.3f}")
        macro_f1 = print_report(cm, labels)
        cc = cross_confusion_rate(cm, labels, "violence", "fall")
        if cc:
            print(f"  violence→fall={cc[0]:.1%}  fall→violence={cc[1]:.1%}")
        print(f"  val macro-F1={macro_f1:.3f}")

        if macro_f1 > best_f1:
            best_f1, patience = macro_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping (no val improvement for {args.patience} epochs).")
                break

    if best_state:
        model.load_state_dict(best_state)

    # --- calibration + operating thresholds, fitted on VAL only ---
    val_logits, val_y = evaluate(va_dl)
    temperature = fit_temperature(val_logits, val_y) if not use_margin else 1.0
    scaled = val_logits / max(temperature, 1e-2)
    val_probs = np.exp(scaled - scaled.max(1, keepdims=True))
    val_probs /= val_probs.sum(1, keepdims=True)
    thresholds = pick_thresholds(val_probs, val_y, labels, args.target_precision)
    print(f"\nCalibration: temperature={temperature:.3f}")
    print(f"Operating thresholds (target precision {args.target_precision:.2f}): "
          + ", ".join(f"{k}={v:.2f}" for k, v in thresholds.items()))

    # --- final, single evaluation on the held-out TEST clips ---
    te_logits, te_y = evaluate(te_dl)
    te_cm = confusion(te_y, te_logits.argmax(1), n_classes)
    print(f"\n{'=' * 62}\nHELD-OUT TEST (clips never seen in training or tuning)\n{'=' * 62}")
    test_f1 = print_report(te_cm, labels)
    cc = cross_confusion_rate(te_cm, labels, "violence", "fall")
    if cc:
        print(f"  violence→fall={cc[0]:.1%}  fall→violence={cc[1]:.1%}   <-- the number that matters")
    print(f"  test macro-F1={test_f1:.3f}")

    # --- per-source breakdown: is it learning the action or the dataset? ---
    by_src = defaultdict(lambda: [0, 0])
    for (ci, _), pred in zip(te_win, te_logits.argmax(1)):
        rec = by_src[sources[ci]]
        rec[1] += 1
        rec[0] += int(pred == y[ci])
    if len(by_src) > 1:
        print("\n  accuracy by source dataset (large gaps => the model learned the "
              "dataset, not the action):")
        for src, (ok, tot) in sorted(by_src.items()):
            print(f"    {src:<28} {ok / max(tot, 1):.3f}  ({tot} windows)")

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    import torch as _torch
    _torch.save({
        "state_dict": model.state_dict(),
        "labels": labels, "feat_dim": FEATURE_DIM, "seq_len": args.seq_len,
        "hidden": profile.hidden, "embed": profile.embed, "use_margin": use_margin,
        "mean": mean.tolist(), "std": std.tolist(),
        "temperature": float(temperature), "thresholds": thresholds,
        "val_macro_f1": float(best_f1), "test_macro_f1": float(test_f1),
        "test_confusion": te_cm.tolist(),
    }, out_p)
    print(f"\nSaved -> {out_p}  (val macro-F1={best_f1:.3f}, test macro-F1={test_f1:.3f})")
    print(f"Now set  detect.action_model: \"{out_p}\"  in your config and restart HawkEye.")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump({
                "labels": labels, "val_macro_f1": float(best_f1),
                "test_macro_f1": float(test_f1), "test_confusion": te_cm.tolist(),
                "temperature": float(temperature), "thresholds": thresholds,
                "by_source": {k: v[0] / max(v[1], 1) for k, v in by_src.items()},
                "clips": {"train": len(tr_ids), "val": len(va_ids), "test": len(te_ids)},
            }, f, indent=2)
        print(f"Report -> {args.report}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Train the HawkEye action classifier")
    ap.add_argument("--data", required=True, help="<root>/<label>/<clip>.mp4")
    ap.add_argument("--out", default="models/action_gru.pt")
    ap.add_argument("--cache", default="data/datasets/action_features.npz",
                    help="per-clip feature cache (skips re-extraction on re-runs)")
    ap.add_argument("--report", default="", help="optional JSON metrics report path")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=None, help="default: from the device profile")
    ap.add_argument("--device", default="auto", help="mps | cuda | cpu | auto (default)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seq-len", type=int, default=64, dest="seq_len",
                    help="frames per window (~3s at 20fps — long enough to see a "
                         "fall settle and a fight sustain)")
    ap.add_argument("--window-stride", type=int, default=24, dest="window_stride")
    ap.add_argument("--val-split", type=float, default=0.15, dest="val_split")
    ap.add_argument("--test-split", type=float, default=0.15, dest="test_split")
    ap.add_argument("--label-smoothing", type=float, default=0.05, dest="label_smoothing")
    ap.add_argument("--target-precision", type=float, default=0.90, dest="target_precision",
                    help="precision each class's operating threshold aims for")
    ap.add_argument("--max-frames", type=int, default=1200, dest="max_frames")
    ap.add_argument("--extract-batch", type=int, default=32, dest="extract_batch",
                    help="frames per pose forward pass during feature extraction. "
                         "Raise on a big GPU, lower if extraction runs out of VRAM.")
    ap.add_argument("--window-cache-gb", type=float, default=4.0, dest="window_cache_gb",
                    help="RAM budget for caching built windows so they are not "
                         "rebuilt every epoch (0 disables)")
    ap.add_argument("--no-tf32", action="store_true", dest="no_tf32",
                    help="disable TF32 matmuls on CUDA (slower; bit-comparable runs)")
    ap.add_argument("--max-per-class", type=int, default=0, dest="max_per_class",
                    help="cap each class to N clips (0 = use everything). Applied "
                         "AFTER the feature cache, so a balanced run reuses the "
                         "same --cache as the full run instead of re-extracting.")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", action="store_true",
                    help="use the ArcFace-style angular-margin head (uncalibrated "
                         "logits; the linear head + temperature scaling is the default)")
    args = ap.parse_args()
    t0 = time.time()
    train(args)
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
