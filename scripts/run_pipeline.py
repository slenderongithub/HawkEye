"""Full training pipeline for HawkEye — run this ONE script to go from raw
video clips to a deployed action classifier.

Orchestrates every step described in docs/training.md:

    Step 1: Verify dataset layout  (data/datasets/actions/{violence,fall,normal}/)
    Step 2: Extract skeleton features via YOLO-pose → cached .npz
    Step 3: Train ActionNet (Conv1d → BiGRU → attention → ArcMargin head)
    Step 4: Evaluate — confusion matrix, per-class P/R/F1, ablation report
    Step 5: Export — save best checkpoint + wire config
    Step 6: Benchmark — measure inference FPS on a test clip

Usage (minimal):
    python scripts/run_pipeline.py --data data/datasets/actions

Full options:
    python scripts/run_pipeline.py \\
        --data data/datasets/actions \\
        --epochs 50 --batch 32 --lr 1e-3 \\
        --seq-len 32 --window-stride 16 \\
        --out models/action_gru.pt \\
        --run-ablation \\
        --benchmark-video data/datasets/sample.mp4 \\
        --skip-verify
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CLASS_DIRS = ["violence", "fall", "normal"]

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Dataset Verification
# ─────────────────────────────────────────────────────────────────────────────
def verify_dataset(root: Path) -> dict:
    """Check dataset structure and print stats. Returns class→count dict."""
    print("\n" + "═" * 65)
    print("  STEP 1 / 6 — DATASET VERIFICATION")
    print("═" * 65)

    counts = {}
    for c in CLASS_DIRS:
        d = root / c
        if d.exists():
            clips = [f for f in d.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXT]
            counts[c] = len(clips)
        else:
            counts[c] = 0

    total = sum(counts.values())
    for c in CLASS_DIRS:
        status = "✓" if counts[c] > 0 else "✗ EMPTY"
        print(f"    {c:12s} : {counts[c]:5d} clips  {status}")
    print(f"\n    Total: {total} clips")

    if total == 0:
        print("\n    ✗ ERROR: No clips found. See docs/training.md §2 for dataset sources.")
        print("    Run: python scripts/setup_datasets.py --source info")
        sys.exit(1)

    empty = [c for c in CLASS_DIRS if counts[c] == 0]
    if empty:
        print(f"\n    ⚠ WARNING: Empty classes: {empty}. Training will proceed but "
              "the model can only learn from available classes.")

    if total > 0:
        mx = max(counts.values())
        mn = min(v for v in counts.values() if v > 0)
        ratio = mx / mn
        if ratio > 5:
            print(f"    ⚠ High class imbalance ({ratio:.1f}x). Class weighting is ON, "
                  "but adding more minority-class clips will help.")
        else:
            print(f"    ✓ Balance ratio: {ratio:.1f}x (acceptable).")

    print("═" * 65)
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Feature Extraction (delegates to train_action.build_dataset)
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(args) -> tuple:
    """Run YOLO-pose over all clips → feature matrix. Uses caching."""
    print("\n" + "═" * 65)
    print("  STEP 2 / 6 — FEATURE EXTRACTION (YOLO-pose → skeleton features)")
    print("═" * 65)

    from hawkeye.detect.skeleton_features import FEATURE_DIM

    cache_p = Path(args.cache)
    if cache_p.exists():
        d = np.load(cache_p, allow_pickle=True)
        X, y, labels = d["X"], d["y"], list(d["labels"])
        print(f"    ✓ Loaded cached features from {cache_p}")
        print(f"      Shape: X={X.shape}, labels={labels}")
        print(f"      Feature dim: {FEATURE_DIM} (base 44 keypoint/posture + 34 velocity)")
        print("    To force re-extraction, delete the cache file:")
        print(f"      del {cache_p}")
        print("═" * 65)
        return X, y, labels

    print(f"    Extracting features from {args.data}/ ...")
    print(f"    Seq length: {args.seq_len} frames (~{args.seq_len/20:.1f}s at 20 fps)")
    print(f"    Window stride: {args.window_stride} frames")
    print(f"    Feature dim: {FEATURE_DIM} per frame")
    print("    This runs YOLO-pose on every frame of every clip — first run is slow.")
    print("    Subsequent runs load from cache and are instant.")
    print()

    from scripts.train_action import build_dataset
    t0 = time.time()
    X, y, labels = build_dataset(args.data, args.seq_len, args.window_stride, args.cache)
    elapsed = time.time() - t0

    print(f"\n    ✓ Extraction complete in {elapsed:.0f}s")
    print(f"      Shape: X={X.shape} ({X.nbytes / 1e6:.1f} MB), labels={labels}")
    print(f"      Cached to: {cache_p}")
    print("═" * 65)
    return X, y, labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Training
# ─────────────────────────────────────────────────────────────────────────────
def train_model(args) -> dict:
    """Train ActionNet. Returns a results dict with best F1, timing, etc."""
    print("\n" + "═" * 65)
    print("  STEP 3 / 6 — TRAINING (ActionNet: Conv1d → BiGRU → ArcMargin)")
    print("═" * 65)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    from hawkeye.detect.action_model import build_model
    from hawkeye.detect.skeleton_features import FEATURE_DIM
    from scripts.train_action import augment, confusion, print_report

    X, y, labels = extract_features(args)
    n_classes = len(labels)
    rng = np.random.default_rng(0)

    # Stratified split
    idx = np.arange(len(y))
    rng.shuffle(idx)
    val_mask = np.zeros(len(y), dtype=bool)
    for c in range(n_classes):
        cidx = idx[y[idx] == c]
        val_mask[cidx[:max(1, int(len(cidx) * args.val_split))]] = True
    Xtr, ytr = X[~val_mask], y[~val_mask]
    Xva, yva = X[val_mask], y[val_mask]

    # Standardize (train stats only)
    mean = Xtr.reshape(-1, FEATURE_DIM).mean(0)
    std = Xtr.reshape(-1, FEATURE_DIM).std(0) + 1e-6
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std

    counts = np.bincount(ytr, minlength=n_classes)
    print(f"\n    Train: {len(ytr)} samples, Val: {len(yva)} samples")
    print(f"    Class counts (train): {dict(zip(labels, counts))}")

    class SeqDataset(Dataset):
        def __init__(self, X, y, do_aug):
            self.X, self.y, self.do_aug = X, y, do_aug
            self.rng = np.random.default_rng(1)
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            s = self.X[i]
            if self.do_aug:
                s = augment(s, self.rng)
            return torch.from_numpy(np.ascontiguousarray(s)).float(), int(self.y[i])

    tr_dl = DataLoader(SeqDataset(Xtr, ytr, True), batch_size=args.batch, shuffle=True)
    va_dl = DataLoader(SeqDataset(Xva, yva, False), batch_size=args.batch)

    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    use_margin = not args.no_margin
    model = build_model(FEATURE_DIM, n_classes, use_margin=use_margin).to(dev)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n    Model: ActionNet on {dev}")
    print(f"      Parameters: {total_params:,} total ({trainable:,} trainable)")
    print(f"      Angular margin head: {'ON (ArcFace s=30 m=0.35)' if use_margin else 'OFF (plain softmax)'}")
    print(f"      Optimizer: AdamW (lr={args.lr}, wd=1e-4)")
    print(f"      Scheduler: CosineAnnealingLR (T_max={args.epochs})")
    print(f"      Label smoothing: {args.label_smoothing}")
    print(f"      Augmentation: temporal jitter, mirror flip, gaussian noise, keypoint dropout, velocity scaling")

    # Inverse-frequency class weights
    cw = torch.tensor((counts.sum() / np.maximum(counts, 1)), dtype=torch.float32)
    cw = (cw / cw.sum() * n_classes).to(dev)
    print(f"      Class weights: {dict(zip(labels, cw.cpu().numpy().round(3)))}")

    loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"\n    {'─' * 55}")
    print(f"    TRAINING — {args.epochs} epochs, batch_size={args.batch}")
    print(f"    {'─' * 55}")

    best_f1, best_state, patience, history = -1.0, None, 0, []
    t_train_start = time.time()

    for ep in range(args.epochs):
        model.train()
        run_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(xb, yb) if use_margin else model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            run_loss += loss.item()
        sched.step()

        # Validate
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                out = model(xb.to(dev))
                preds.extend(out.argmax(1).cpu().numpy())
                trues.extend(yb.numpy())

        cm = confusion(np.array(trues), np.array(preds), n_classes)
        avg_loss = run_loss / max(1, len(tr_dl))
        print(f"\n    ┌─ Epoch {ep + 1}/{args.epochs}  train_loss={avg_loss:.4f}  lr={sched.get_last_lr()[0]:.6f}")
        macro_f1 = print_report(cm, labels)
        print(f"      macro-F1 = {macro_f1:.4f}", end="")

        history.append({
            "epoch": ep + 1, "train_loss": avg_loss, "macro_f1": macro_f1,
            "confusion_matrix": cm.tolist()
        })

        if macro_f1 > best_f1:
            best_f1, patience = macro_f1, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ★ NEW BEST", end="")
        else:
            patience += 1
            print(f"  (patience {patience}/{args.patience})", end="")
            if patience >= args.patience:
                print(f"\n    Early stopping — no improvement for {args.patience} epochs.")
                break
        print()

    train_time = time.time() - t_train_start

    # Save checkpoint
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": best_state or model.state_dict(),
        "labels": labels, "feat_dim": FEATURE_DIM, "seq_len": args.seq_len,
        "hidden": 128, "embed": 128, "use_margin": use_margin,
        "mean": mean.tolist(), "std": std.tolist(),
        "val_macro_f1": best_f1,
    }
    torch.save(checkpoint, out_p)

    results = {
        "model_path": str(out_p),
        "best_val_f1": best_f1,
        "epochs_run": len(history),
        "train_time_sec": train_time,
        "n_classes": n_classes,
        "labels": labels,
        "use_margin": use_margin,
        "total_params": total_params,
        "device": dev,
        "history": history,
    }

    print(f"\n    {'═' * 55}")
    print(f"    ✓ Training complete!")
    print(f"      Best val macro-F1: {best_f1:.4f}")
    print(f"      Epochs run: {len(history)} / {args.epochs}")
    print(f"      Training time: {train_time:.0f}s ({train_time/60:.1f} min)")
    print(f"      Checkpoint saved: {out_p}")
    print(f"    {'═' * 55}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Evaluation (detailed report)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(args, train_results: dict) -> dict:
    """Generate a detailed evaluation report from training results."""
    print("\n" + "═" * 65)
    print("  STEP 4 / 6 — EVALUATION REPORT")
    print("═" * 65)

    history = train_results["history"]
    if not history:
        print("    No training history to evaluate.")
        return {}

    # Extract metrics from final epoch
    final = history[-1]
    best_epoch = max(history, key=lambda h: h["macro_f1"])
    labels = train_results["labels"]

    print(f"\n    Model: ActionNet ({'ArcMargin' if train_results['use_margin'] else 'Softmax'})")
    print(f"    Device: {train_results['device']}")
    print(f"    Parameters: {train_results['total_params']:,}")
    print(f"\n    Best epoch: {best_epoch['epoch']}  (macro-F1 = {best_epoch['macro_f1']:.4f})")
    print(f"    Final epoch: {final['epoch']}  (macro-F1 = {final['macro_f1']:.4f})")

    # Print the final confusion matrix nicely
    cm = np.array(best_epoch["confusion_matrix"])
    print(f"\n    CONFUSION MATRIX (best epoch {best_epoch['epoch']}):")
    print(f"    (rows = true class, cols = predicted)")
    w = max(len(l) for l in labels) + 2
    print("    " + " " * w + "".join(f"{l[:8]:>9}" for l in labels))
    for i, l in enumerate(labels):
        print(f"    {l:<{w}}" + "".join(f"{cm[i, j]:>9}" for j in range(len(labels))))

    # Per-class metrics from confusion matrix
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)

    print(f"\n    PER-CLASS METRICS (best epoch):")
    print(f"    {'Class':<{w}} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print(f"    {'─' * (w + 44)}")
    for i, l in enumerate(labels):
        support = int(cm[i].sum())
        print(f"    {l:<{w}} {prec[i]:>10.4f} {rec[i]:>10.4f} {f1[i]:>10.4f} {support:>10}")
    print(f"    {'─' * (w + 44)}")
    print(f"    {'macro avg':<{w}} {prec.mean():>10.4f} {rec.mean():>10.4f} {f1.mean():>10.4f} {int(cm.sum()):>10}")

    # Key diagnostic: violence↔fall confusion
    if "violence" in labels and "fall" in labels:
        vi = labels.index("violence")
        fi = labels.index("fall")
        vf = cm[vi, fi]   # violence predicted as fall
        fv = cm[fi, vi]   # fall predicted as violence
        print(f"\n    KEY DIAGNOSTIC — Violence ↔ Fall confusion:")
        print(f"      Violence clips misclassified as Fall: {vf}")
        print(f"      Fall clips misclassified as Violence: {fv}")
        if vf + fv == 0:
            print(f"      ✓ PERFECT separation!")
        elif vf + fv <= 3:
            print(f"      ✓ Minimal confusion — the model is separating these well.")
        else:
            total_confused = vf + fv
            total_samples = cm[vi].sum() + cm[fi].sum()
            print(f"      ⚠ {total_confused}/{total_samples} confused samples "
                  f"({100*total_confused/total_samples:.1f}%).")
            print(f"      Tips to improve:")
            print(f"        - Add more/cleaner data for the confused pair")
            print(f"        - Increase margin: bump m from 0.30 → 0.40")
            print(f"        - Longer clips: raise --seq-len (more temporal context)")
            print(f"        - Check for mislabeled clips")

    # Training curve summary
    losses = [h["train_loss"] for h in history]
    f1s = [h["macro_f1"] for h in history]
    print(f"\n    TRAINING CURVE SUMMARY:")
    print(f"      Loss:  {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"      F1:    {f1s[0]:.4f} → {max(f1s):.4f} (peak) → {f1s[-1]:.4f} (final)")

    eval_results = {
        "best_epoch": best_epoch["epoch"],
        "best_macro_f1": best_epoch["macro_f1"],
        "final_confusion_matrix": cm.tolist(),
        "per_class_f1": dict(zip(labels, f1.tolist())),
        "per_class_precision": dict(zip(labels, prec.tolist())),
        "per_class_recall": dict(zip(labels, rec.tolist())),
    }

    print("═" * 65)
    return eval_results


# ─────────────────────────────────────────────────────────────────────────────
# Step 4b: Ablation (ArcMargin vs plain softmax)
# ─────────────────────────────────────────────────────────────────────────────
def run_ablation(args, margin_results: dict) -> None:
    """Train a second model WITHOUT angular margin and compare."""
    print("\n" + "═" * 65)
    print("  STEP 4b — ABLATION: ArcMargin vs Plain Softmax")
    print("═" * 65)
    print("    Training a second model with --no-margin for comparison...")
    print("    (This is a key thesis result — does the margin head help?)")

    # Create a modified args for no-margin
    import copy
    ablation_args = copy.deepcopy(args)
    ablation_args.no_margin = True
    ablation_args.out = str(Path(args.out).with_suffix("")) + "_no_margin.pt"

    no_margin_results = train_model(ablation_args)

    print("\n" + "═" * 65)
    print("  ABLATION COMPARISON")
    print("═" * 65)
    margin_f1 = margin_results["best_val_f1"]
    nomargin_f1 = no_margin_results["best_val_f1"]
    diff = margin_f1 - nomargin_f1

    print(f"\n    {'Metric':<25} {'ArcMargin':>12} {'Plain Softmax':>14} {'Delta':>10}")
    print(f"    {'─' * 63}")
    print(f"    {'Best val macro-F1':<25} {margin_f1:>12.4f} {nomargin_f1:>14.4f} {diff:>+10.4f}")
    print(f"    {'Epochs to best':<25} "
          f"{max(margin_results['history'], key=lambda h: h['macro_f1'])['epoch']:>12} "
          f"{max(no_margin_results['history'], key=lambda h: h['macro_f1'])['epoch']:>14}")
    print(f"    {'Training time':<25} "
          f"{margin_results['train_time_sec']:>11.0f}s "
          f"{no_margin_results['train_time_sec']:>13.0f}s")

    if diff > 0.01:
        print(f"\n    ✓ ArcMargin head IMPROVES F1 by {diff:.4f} — use it.")
    elif diff > -0.01:
        print(f"\n    ~ Negligible difference ({diff:+.4f}). Margin head has no "
              "downside, keep it on for the angular margin safety net.")
    else:
        print(f"\n    ⚠ Plain softmax was better by {-diff:.4f}. This is unusual. "
              "Check if there's enough data for the margin to help.")
    print("═" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Config Wiring
# ─────────────────────────────────────────────────────────────────────────────
def wire_config(args, train_results: dict) -> None:
    """Update configs/default.yaml to point at the new model."""
    print("\n" + "═" * 65)
    print("  STEP 5 / 6 — CONFIG WIRING")
    print("═" * 65)

    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"

    if not config_path.exists():
        print(f"    ⚠ Config file not found at {config_path}")
        print(f"    Manually set detect.action_model in your config to: {args.out}")
        return

    config_text = config_path.read_text()
    model_path = args.out

    # Update the action_model line
    import re
    new_config = re.sub(
        r'(action_model:\s*)"[^"]*"',
        f'\\1"{model_path}"',
        config_text
    )

    if new_config == config_text:
        # Try without quotes
        new_config = re.sub(
            r'(action_model:\s*)""',
            f'\\1"{model_path}"',
            config_text
        )

    if new_config != config_text:
        config_path.write_text(new_config)
        print(f"    ✓ Updated {config_path}:")
        print(f"      detect.action_model: \"{model_path}\"")
    else:
        print(f"    ⚠ Could not auto-update config. Manually set:")
        print(f"      detect:")
        print(f"        action_model: \"{model_path}\"")

    print(f"\n    On next restart, HawkEye will log:")
    print(f"      Stage A action source: TRAINED model ({model_path})")
    print(f"    instead of:")
    print(f"      Stage A action source: heuristics")
    print("═" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Benchmark
# ─────────────────────────────────────────────────────────────────────────────
def benchmark(args, train_results: dict) -> None:
    """Run inference speed benchmark with the trained model."""
    print("\n" + "═" * 65)
    print("  STEP 6 / 6 — INFERENCE BENCHMARK")
    print("═" * 65)

    if not args.benchmark_video:
        print("    Skipped (no --benchmark-video provided).")
        print("    To benchmark, re-run with:")
        print(f"      --benchmark-video data/datasets/sample.mp4")
        print("═" * 65)
        return

    from hawkeye.config import load_config
    from hawkeye.detect import StageA

    cfg = load_config()
    cfg.detect.action_model = args.out
    stage = StageA(cfg.detect).load()

    from hawkeye.ingest import open_source
    src = open_source(args.benchmark_video, cfg.ingest.width, cfg.ingest.height)

    n, total_lat = 0, 0.0
    max_frames = 300
    try:
        while n < max_frames:
            frame = src.read()
            if frame is None:
                break
            t0 = time.time()
            event, _, _ = stage.process(frame, n, t0)
            total_lat += (time.time() - t0)
            n += 1
    finally:
        src.release()

    if n > 0:
        avg_ms = 1000 * total_lat / n
        fps = n / total_lat
        print(f"    Frames processed: {n}")
        print(f"    Avg latency: {avg_ms:.1f} ms/frame")
        print(f"    Throughput: {fps:.1f} FPS")
        print(f"    Model: {'ArcMargin' if train_results['use_margin'] else 'Softmax'}")
    else:
        print("    No frames read from benchmark video.")
    print("═" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Summary Report
# ─────────────────────────────────────────────────────────────────────────────
def save_report(args, train_results: dict, eval_results: dict) -> None:
    """Save a JSON report of the full pipeline run."""
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"training_report_{timestamp}.json"

    report = {
        "timestamp": timestamp,
        "args": {
            "data": args.data, "epochs": args.epochs, "batch": args.batch,
            "lr": args.lr, "seq_len": args.seq_len, "window_stride": args.window_stride,
            "no_margin": args.no_margin, "label_smoothing": args.label_smoothing,
        },
        "training": {
            "model_path": train_results.get("model_path"),
            "best_val_f1": train_results.get("best_val_f1"),
            "epochs_run": train_results.get("epochs_run"),
            "train_time_sec": train_results.get("train_time_sec"),
            "device": train_results.get("device"),
            "labels": train_results.get("labels"),
            "use_margin": train_results.get("use_margin"),
            "total_params": train_results.get("total_params"),
        },
        "evaluation": eval_results,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n    ✓ Full report saved to: {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="HawkEye — Full Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick run:    python scripts/run_pipeline.py --data data/datasets/actions
  Full run:     python scripts/run_pipeline.py --data data/datasets/actions --epochs 50 --run-ablation
  Benchmark:    python scripts/run_pipeline.py --data data/datasets/actions --benchmark-video test.mp4
        """
    )
    # Data
    ap.add_argument("--data", required=True, help="Root folder with <label>/ subdirs of .mp4 clips")
    ap.add_argument("--out", default="models/action_gru.pt", help="Output checkpoint path")
    ap.add_argument("--cache", default="data/datasets/action_features.npz",
                    help="Feature cache file (re-extraction skipped if present)")

    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seq-len", type=int, default=32, dest="seq_len",
                    help="Frames per sample (~1.5s at 20 fps)")
    ap.add_argument("--window-stride", type=int, default=16, dest="window_stride",
                    help="Sliding window step (smaller = more samples)")
    ap.add_argument("--val-split", type=float, default=0.2, dest="val_split")
    ap.add_argument("--label-smoothing", type=float, default=0.05, dest="label_smoothing")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--no-margin", action="store_true", dest="no_margin",
                    help="Disable angular-margin head (use plain softmax)")

    # Pipeline options
    ap.add_argument("--skip-verify", action="store_true", dest="skip_verify",
                    help="Skip dataset verification step")
    ap.add_argument("--run-ablation", action="store_true", dest="run_ablation",
                    help="Also train without margin and compare (thesis result)")
    ap.add_argument("--benchmark-video", default="", dest="benchmark_video",
                    help="Video file for inference speed benchmark")
    ap.add_argument("--no-wire", action="store_true", dest="no_wire",
                    help="Don't auto-update configs/default.yaml")

    args = ap.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          HAWKEYE — FULL TRAINING PIPELINE                   ║")
    print("║                                                             ║")
    print("║  Step 1: Verify dataset layout                              ║")
    print("║  Step 2: Extract skeleton features (YOLO-pose)              ║")
    print("║  Step 3: Train ActionNet (Conv1d → BiGRU → ArcMargin)      ║")
    print("║  Step 4: Evaluate (confusion matrix + per-class metrics)    ║")
    print("║  Step 5: Wire config (plug model into HawkEye)             ║")
    print("║  Step 6: Benchmark (inference FPS)                          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    t_start = time.time()

    # Step 1
    if not args.skip_verify:
        verify_dataset(Path(args.data))

    # Steps 2 + 3 (extraction happens inside train)
    train_results = train_model(args)

    # Step 4
    eval_results = evaluate_model(args, train_results)

    # Step 4b (optional)
    if args.run_ablation and not args.no_margin:
        run_ablation(args, train_results)

    # Step 5
    if not args.no_wire:
        wire_config(args, train_results)

    # Step 6
    benchmark(args, train_results)

    # Save report
    save_report(args, train_results, eval_results)

    total = time.time() - t_start
    print("\n" + "═" * 65)
    print(f"  ✓ PIPELINE COMPLETE — Total time: {total:.0f}s ({total/60:.1f} min)")
    print(f"    Model: {args.out}")
    print(f"    Best F1: {train_results['best_val_f1']:.4f}")
    print(f"\n    Next steps:")
    print(f"      1. Restart HawkEye to use the trained model")
    print(f"      2. Watch for 'Stage A action source: TRAINED model' in the logs")
    print(f"      3. Share the .pt file with teammates (it's gitignored)")
    print("═" * 65)


if __name__ == "__main__":
    main()
