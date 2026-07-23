"""Train the HawkEye action classifier (violence / fall / normal ...).

This produces the weights that REPLACE the zero-training heuristics in Stage A.
It is built specifically to fix the "violence and fall get confused" problem —
see docs/training.md for the full explanation. The five levers, all on by
default here:

  1. Rich, class-separating features (hawkeye/detect/skeleton_features.py):
     posture/verticality cues for FALL, multi-person proximity/motion cues for
     VIOLENCE — computed identically at train and inference time.
  2. Temporal model + optional angular-margin head (ArcFace-style) that
     literally maximizes the margin between classes. Toggle with --no-margin.
  3. Class balancing (inverse-frequency weights) so a small FALL set isn't
     swallowed by a large VIOLENCE set.
  4. Augmentation (mirror flip, temporal jitter, noise, keypoint dropout) for
     generalization from few clips.
  5. A per-class confusion matrix + precision/recall/F1 every epoch, so you can
     SEE the violence<->fall confusion and watch it shrink.

Dataset layout (download separately — see docs/training.md):
    <root>/<label>/<clip>.mp4      e.g.  data/datasets/actions/violence/*.mp4
                                          data/datasets/actions/fall/*.mp4
                                          data/datasets/actions/normal/*.mp4

Run (on the RTX 3050):
    python scripts/train_action.py --data data/datasets/actions --epochs 30
    # weights -> models/action_gru.pt ; then set detect.action_model in the config.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hawkeye.detect.skeleton_features import (BASE_DIM, FEATURE_DIM, KP_XY,  # noqa: E402
                                              build_sequence, frame_feature, resample)

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
# COCO left/right mirror map for horizontal-flip augmentation.
MIRROR = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])


# --------------------------------------------------------------------------- #
# Feature extraction (shared logic with inference via skeleton_features)
# --------------------------------------------------------------------------- #
def extract_clip(video_path: str, pose) -> list:
    """Run YOLO-pose over a clip -> list of BASE_DIM per-frame feature vectors."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    feats = []
    max_frames = 1000  # safety cap (~50s at 20fps)
    n = 0
    while n < max_frames:
        try:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            h, w = frame.shape[:2]
            persons = pose.detect(frame)
            feats.append(frame_feature(persons, 0.0, w, h))
            n += 1
        except Exception:
            break
    cap.release()
    return feats


def build_dataset(root: str, seq_len: int, window_stride: int, cache: str):
    cache_p = Path(cache) if cache else None
    if cache_p and cache_p.exists():
        d = np.load(cache_p, allow_pickle=True)
        print(f"Loaded cached features from {cache_p}: X={d['X'].shape}")
        return d["X"], d["y"], list(d["labels"])

    from hawkeye.detect.pose import PoseDetector
    pose = PoseDetector().load()

    root_p = Path(root)
    labels = sorted([d.name for d in root_p.iterdir() if d.is_dir()])
    if not labels:
        sys.exit(f"No <label>/ folders under {root}. Expected <root>/<label>/<clip>.mp4")
    print(f"Classes: {labels}")

    X, y = [], []
    for li, lab in enumerate(labels):
        clips = [c for c in (root_p / lab).rglob("*") if c.is_file() and c.suffix.lower() in VIDEO_EXT]
        print(f"  [{lab}] {len(clips)} clips")
        for ci, clip in enumerate(clips):
            feats = extract_clip(str(clip), pose)
            if len(feats) < 4:
                continue
            # Sliding windows -> multiple samples per clip (more data, esp. minority).
            if len(feats) >= seq_len:
                for s in range(0, len(feats) - seq_len + 1, max(1, window_stride)):
                    X.append(build_sequence(feats[s:s + seq_len]))
                    y.append(li)
            else:
                X.append(resample(build_sequence(feats), seq_len))
                y.append(li)
            print(f"    {lab} {ci + 1}/{len(clips)}", end="\r", flush=True)
        print()

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    if cache_p:
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_p, X=X, y=y, labels=np.array(labels))
        print(f"Cached features -> {cache_p}")
    return X, y, labels


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def flip_sequence(seq: np.ndarray) -> np.ndarray:
    """Horizontal mirror: negate keypoint x and swap left/right joints, on both
    the keypoint block [0:34] and the velocity block [BASE_DIM:BASE_DIM+34]."""
    out = seq.copy()
    for block in (0, BASE_DIM):
        kp = seq[:, block:block + KP_XY]
        new = np.empty_like(kp)
        for i in range(17):
            m = MIRROR[i]
            new[:, 2 * i] = -kp[:, 2 * m]
            new[:, 2 * i + 1] = kp[:, 2 * m + 1]
        out[:, block:block + KP_XY] = new
    return out


def augment(seq: np.ndarray, rng) -> np.ndarray:
    # Temporal jitter: randomly crop 75-100% of frames, then resample back
    if rng.random() < 0.4 and seq.shape[0] > 8:
        crop_ratio = rng.uniform(0.75, 1.0)
        crop_len = max(8, int(seq.shape[0] * crop_ratio))
        start = rng.integers(0, seq.shape[0] - crop_len + 1)
        seq = resample(seq[start:start + crop_len], seq.shape[0])
    if rng.random() < 0.5:
        seq = flip_sequence(seq)
    if rng.random() < 0.5:                                    # gaussian noise on keypoints
        noise = rng.normal(0, 0.03, seq[:, :KP_XY].shape).astype(np.float32)
        seq = seq.copy(); seq[:, :KP_XY] += noise
    if rng.random() < 0.4:                                    # random keypoint dropout
        seq = seq.copy()
        drop = rng.random(17) < 0.15
        for i in np.where(drop)[0]:
            seq[:, 2 * i:2 * i + 2] = 0.0
    # Random temporal scaling of velocity features
    if rng.random() < 0.3:
        seq = seq.copy()
        scale = rng.uniform(0.8, 1.2)
        seq[:, BASE_DIM:BASE_DIM + KP_XY] *= scale
    return seq


# --------------------------------------------------------------------------- #
# Metrics (numpy — no sklearn dependency)
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


def print_report(cm, labels):
    prec, rec, f1 = prf(cm)
    w = max(len(l) for l in labels) + 2
    print("  confusion (rows=true, cols=pred):")
    print("    " + "".join(f"{l[:8]:>9}" for l in labels))
    for i, l in enumerate(labels):
        print(f"    {l[:w]:<{w}}" + "".join(f"{cm[i, j]:>9}" for j in range(len(labels))))
    print("  per-class  precision / recall / F1:")
    for i, l in enumerate(labels):
        print(f"    {l:<{w}} {prec[i]:.3f} / {rec[i]:.3f} / {f1[i]:.3f}")
    return float(f1.mean())


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    from hawkeye.detect.action_model import build_model

    X, y, labels = build_dataset(args.data, args.seq_len, args.window_stride, args.cache)
    n_classes = len(labels)
    rng = np.random.default_rng(0)

    # stratified split
    idx = np.arange(len(y))
    rng.shuffle(idx)
    val_mask = np.zeros(len(y), dtype=bool)
    for c in range(n_classes):
        cidx = idx[y[idx] == c]
        val_mask[cidx[:max(1, int(len(cidx) * args.val_split))]] = True
    Xtr, ytr = X[~val_mask], y[~val_mask]
    Xva, yva = X[val_mask], y[val_mask]

    # standardize using TRAIN stats only
    mean = Xtr.reshape(-1, FEATURE_DIM).mean(0)
    std = Xtr.reshape(-1, FEATURE_DIM).std(0) + 1e-6
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std

    counts = np.bincount(ytr, minlength=n_classes)
    print(f"train={len(ytr)} val={len(yva)}  class counts (train)={dict(zip(labels, counts))}")

    class Seq(Dataset):
        def __init__(self, X, y, aug):
            self.X, self.y, self.aug = X, y, aug
            self.rng = np.random.default_rng(1)
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            s = self.X[i]
            if self.aug:
                s = augment(s, self.rng)
            return torch.from_numpy(np.ascontiguousarray(s)).float(), int(self.y[i])

    from hawkeye.train.device_profile import get_profile

    profile = get_profile(args.device)
    dev = profile.device
    batch = args.batch if args.batch is not None else profile.batch_size

    # Features are already in RAM and augmentation is cheap numpy, so worker
    # processes only add IPC overhead — and, via Seq's per-instance RNG, would
    # repeat identical augmentations across workers. Keep the loader single-process.
    tr_dl = DataLoader(Seq(Xtr, ytr, True), batch_size=batch, shuffle=True,
                       num_workers=0, pin_memory=profile.pin_memory)
    va_dl = DataLoader(Seq(Xva, yva, False), batch_size=batch,
                       num_workers=0, pin_memory=profile.pin_memory)

    use_margin = not args.no_margin
    model = build_model(FEATURE_DIM, n_classes, hidden=profile.hidden,
                        embed=profile.embed, use_margin=use_margin).to(dev)

    # inverse-frequency class weights -> the fix for imbalance
    cw = torch.tensor((counts.sum() / np.maximum(counts, 1)), dtype=torch.float32)
    cw = (cw / cw.sum() * n_classes).to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Mixed precision on CUDA (RTX 3050): faster + less VRAM. On the CPU/MPS
    # smoke-test fallback profile.use_amp is False, so both are no-ops.
    scaler = torch.amp.GradScaler("cuda", enabled=profile.use_amp)

    print(f"device={dev}  amp={profile.use_amp}  batch={batch}  "
          f"hidden={profile.hidden}  margin={use_margin}  classes={n_classes}  "
          f"feat_dim={FEATURE_DIM}")
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
            scaler.step(opt)
            scaler.update()
            run += loss.item()
        sched.step()

        # validate
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                out = model(xb.to(dev))                       # no labels -> plain logits
                preds.extend(out.argmax(1).cpu().numpy())
                trues.extend(yb.numpy())
        cm = confusion(np.array(trues), np.array(preds), n_classes)
        print(f"\nepoch {ep + 1}/{args.epochs}  train_loss={run / max(1, len(tr_dl)):.3f}")
        macro_f1 = print_report(cm, labels)
        print(f"  macro-F1={macro_f1:.3f}")

        if macro_f1 > best_f1:
            best_f1, patience = macro_f1, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping (no val improvement for {args.patience} epochs).")
                break

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state or model.state_dict(),
        "labels": labels, "feat_dim": FEATURE_DIM, "seq_len": args.seq_len,
        "hidden": profile.hidden, "embed": profile.embed, "use_margin": use_margin,
        "mean": mean.tolist(), "std": std.tolist(),
        "val_macro_f1": best_f1,
    }, out_p)
    print(f"\nBest val macro-F1={best_f1:.3f}. Saved -> {out_p}")
    print(f"Now set  detect.action_model: \"{out_p}\"  in your config and restart HawkEye.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the HawkEye action classifier")
    ap.add_argument("--data", required=True, help="<root>/<label>/<clip>.mp4")
    ap.add_argument("--out", default="models/action_gru.pt")
    ap.add_argument("--cache", default="data/datasets/action_features.npz",
                    help="feature cache (skips re-extraction on re-runs)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=None,
                    help="batch size (default: from the device profile)")
    ap.add_argument("--device", default="auto", help="cuda | cpu | auto (default)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seq-len", type=int, default=32, dest="seq_len")
    ap.add_argument("--window-stride", type=int, default=16, dest="window_stride")
    ap.add_argument("--val-split", type=float, default=0.2, dest="val_split")
    ap.add_argument("--label-smoothing", type=float, default=0.05, dest="label_smoothing")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--no-margin", action="store_true",
                    help="disable the angular-margin head (use plain softmax)")
    args = ap.parse_args()
    t0 = time.time()
    train(args)
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
