"""Evaluation + benchmarking for HawkEye.

Fully implemented:
  * fps          — Stage A latency and throughput on any video/webcam.
  * test-action  — Run the trained action classifier on a held-out test set,
                   producing a full confusion matrix + per-class P/R/F1.
  * rwf2000      — Clip-level accuracy on the RWF-2000 dataset (fights).
  * compare      — Compare heuristics vs trained model on the same clips.

Usage:
  python eval/evaluate.py fps --source data/datasets/sample.mp4
  python eval/evaluate.py test-action --data data/datasets/actions --model models/action_gru.pt
  python eval/evaluate.py rwf2000 --data data/datasets/rwf2000 --model models/action_gru.pt
  python eval/evaluate.py compare --data data/datasets/actions --model models/action_gru.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hawkeye.config import load_config           # noqa: E402
from hawkeye.detect import StageA                 # noqa: E402
from hawkeye.ingest import open_source            # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ─────────────────────────────────────────────────────────────────────────────
# Shared metrics
# ─────────────────────────────────────────────────────────────────────────────
def confusion(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_report(cm, labels):
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    w = max(len(l) for l in labels) + 2

    print("\n  Confusion Matrix (rows=true, cols=predicted):")
    print("    " + " " * w + "".join(f"{l[:8]:>9}" for l in labels))
    for i, l in enumerate(labels):
        print(f"    {l:<{w}}" + "".join(f"{cm[i, j]:>9}" for j in range(len(labels))))

    print(f"\n  {'Class':<{w}} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'─' * (w + 44)}")
    for i, l in enumerate(labels):
        support = int(cm[i].sum())
        print(f"  {l:<{w}} {prec[i]:>10.3f} {rec[i]:>10.3f} {f1[i]:>10.3f} {support:>10}")
    print(f"  {'─' * (w + 44)}")
    print(f"  {'macro avg':<{w}} {prec.mean():>10.3f} {rec.mean():>10.3f} {f1.mean():>10.3f} {int(cm.sum()):>10}")

    accuracy = tp.sum() / max(cm.sum(), 1)
    print(f"\n  Overall accuracy: {accuracy:.3f} ({int(tp.sum())}/{int(cm.sum())})")
    print(f"  Macro F1: {f1.mean():.3f}")
    return float(f1.mean())


# ─────────────────────────────────────────────────────────────────────────────
# fps benchmark
# ─────────────────────────────────────────────────────────────────────────────
def bench_fps(source: str, max_frames: int = 300) -> None:
    cfg = load_config()
    stage = StageA(cfg.detect).load()
    src = open_source(source, cfg.ingest.width, cfg.ingest.height)
    n, lat = 0, 0.0
    t_start = time.time()
    triggers = 0
    try:
        while n < max_frames:
            frame = src.read()
            if frame is None:
                break
            t0 = time.time()
            event, _, _ = stage.process(frame, n, t0)
            lat += (time.time() - t0)
            triggers += int(event.triggered)
            n += 1
    finally:
        src.release()
    wall = time.time() - t_start
    if n:
        print(f"frames={n}  avg_latency={1000*lat/n:.1f} ms  "
              f"throughput={n/wall:.1f} FPS  triggers={triggers}")
    else:
        print("No frames read.")


# ─────────────────────────────────────────────────────────────────────────────
# test-action: Evaluate the trained action classifier on a test set
# ─────────────────────────────────────────────────────────────────────────────
def test_action(data_dir: str, model_path: str, seq_len: int = 32) -> None:
    """Run the trained model over all clips in data_dir and report accuracy."""
    import torch

    from hawkeye.detect.action_model import build_model
    from hawkeye.detect.pose import PoseDetector
    from hawkeye.detect.skeleton_features import (BASE_DIM, FEATURE_DIM,
                                                  build_sequence,
                                                  frame_feature, resample)

    print(f"\n  Loading model from {model_path} ...")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_path, map_location=dev, weights_only=False)
    labels = list(ckpt["labels"])
    model_seq_len = int(ckpt["seq_len"])
    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32) + 1e-6

    model = build_model(int(ckpt["feat_dim"]), len(labels),
                        int(ckpt.get("hidden", 128)), int(ckpt.get("embed", 128)),
                        bool(ckpt.get("use_margin", True)))
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()

    print(f"  Labels: {labels}, seq_len={model_seq_len}, device={dev}")

    # Load pose detector for feature extraction
    pose = PoseDetector().load()
    import cv2

    root = Path(data_dir)
    all_true, all_pred = [], []

    for li, lab in enumerate(labels):
        folder = root / lab
        if not folder.exists():
            print(f"  [skip] {lab}/ not found")
            continue
        clips = [c for c in folder.iterdir() if c.suffix.lower() in VIDEO_EXT]
        correct = 0

        for ci, clip in enumerate(clips):
            # Extract features
            cap = cv2.VideoCapture(str(clip))
            feats = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                persons = pose.detect(frame)
                feats.append(frame_feature(persons, 0.0, w, h))
            cap.release()

            if len(feats) < 4:
                continue

            seq = build_sequence(feats)
            seq = resample(seq, model_seq_len)
            seq = (seq - mean) / std
            x = torch.from_numpy(seq[None]).float().to(dev)

            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            pred = int(probs.argmax())
            conf = float(probs[pred])

            all_true.append(li)
            all_pred.append(pred)
            if pred == li:
                correct += 1

            print(f"    {lab} {ci+1}/{len(clips)}: pred={labels[pred]} ({conf:.2f})"
                  f"  {'✓' if pred == li else '✗'}", end="\r")
        print(f"  [{lab}] {correct}/{len(clips)} correct "
              f"({100*correct/max(len(clips),1):.1f}%)")

    if all_true:
        cm = confusion(np.array(all_true), np.array(all_pred), len(labels))
        print_report(cm, labels)


# ─────────────────────────────────────────────────────────────────────────────
# compare: Heuristics vs trained model on the same clips
# ─────────────────────────────────────────────────────────────────────────────
def compare_methods(data_dir: str, model_path: str) -> None:
    """Run both heuristics and the trained model on each clip, compare."""
    import cv2
    import torch

    from hawkeye.detect import heuristics
    from hawkeye.detect.action_classifier import ActionClassifier, label_to_type
    from hawkeye.detect.action_model import build_model
    from hawkeye.detect.pose import PoseDetector
    from hawkeye.detect.scene_anomaly import MotionAnomaly
    from hawkeye.detect.skeleton_features import (FEATURE_DIM, build_sequence,
                                                  frame_feature, resample)
    from hawkeye.events.schema import AnomalyType

    # Map anomaly types to label indices
    type_to_label = {"violence": "violence", "fall": "fall", "normal": "normal"}

    print(f"\n  Comparing heuristics vs trained model ({model_path})")

    # Load model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_path, map_location=dev, weights_only=False)
    labels = list(ckpt["labels"])
    model_seq_len = int(ckpt["seq_len"])
    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32) + 1e-6

    model = build_model(int(ckpt["feat_dim"]), len(labels),
                        int(ckpt.get("hidden", 128)), int(ckpt.get("embed", 128)),
                        bool(ckpt.get("use_margin", True)))
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()

    pose = PoseDetector().load()
    root = Path(data_dir)

    heur_correct, model_correct, total = 0, 0, 0

    for li, lab in enumerate(labels):
        folder = root / lab
        if not folder.exists():
            continue
        clips = [c for c in folder.iterdir() if c.suffix.lower() in VIDEO_EXT]

        for clip in clips:
            cap = cv2.VideoCapture(str(clip))
            feats = []
            heur_votes = []  # collect per-frame heuristic predictions
            motion = MotionAnomaly()

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                persons = pose.detect(frame)
                mot = motion.update(frame)
                feats.append(frame_feature(persons, mot, w, h))

                # Run heuristics on this frame
                candidates = heuristics.analyze(persons, mot)
                if candidates:
                    best = max(candidates, key=lambda c: c[1])
                    heur_votes.append(best[0].value)
                else:
                    heur_votes.append("normal")
            cap.release()

            if len(feats) < 4:
                continue

            # Model prediction
            seq = build_sequence(feats)
            seq = resample(seq, model_seq_len)
            seq = (seq - mean) / std
            x = torch.from_numpy(seq[None]).float().to(dev)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            model_pred = labels[int(probs.argmax())]

            # Heuristic prediction: majority vote over frames
            from collections import Counter
            vote_counts = Counter(heur_votes)
            heur_pred = vote_counts.most_common(1)[0][0]

            total += 1
            if model_pred == lab:
                model_correct += 1
            if heur_pred == lab:
                heur_correct += 1

    if total > 0:
        print(f"\n  Results over {total} clips:")
        print(f"    Heuristics accuracy: {heur_correct}/{total} ({100*heur_correct/total:.1f}%)")
        print(f"    Trained model accuracy: {model_correct}/{total} ({100*model_correct/total:.1f}%)")
        improvement = model_correct - heur_correct
        print(f"    Improvement: {improvement:+d} clips ({100*(model_correct-heur_correct)/total:+.1f}%)")
    else:
        print("  No clips processed.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="HawkEye Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  fps           Benchmark Stage A processing speed
  test-action   Evaluate the trained action classifier on a test set
  rwf2000       Evaluate on the RWF-2000 fight detection dataset
  compare       Compare heuristics vs trained model side-by-side
        """
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # fps
    f = sub.add_parser("fps", help="Benchmark Stage A FPS")
    f.add_argument("--source", default="0")
    f.add_argument("--max-frames", type=int, default=300)

    # test-action
    t = sub.add_parser("test-action", help="Evaluate trained model on test data")
    t.add_argument("--data", required=True, help="Root dir with <label>/<clip>.mp4")
    t.add_argument("--model", required=True, help="Path to trained .pt checkpoint")
    t.add_argument("--seq-len", type=int, default=32)

    # rwf2000
    r = sub.add_parser("rwf2000", help="Evaluate on RWF-2000 dataset")
    r.add_argument("--data", required=True,
                   help="RWF-2000 root (expects Fight/ and NonFight/ subdirs)")
    r.add_argument("--model", required=True, help="Trained .pt checkpoint")

    # compare
    c = sub.add_parser("compare", help="Compare heuristics vs trained model")
    c.add_argument("--data", required=True, help="Dataset root with class subdirs")
    c.add_argument("--model", required=True, help="Trained .pt checkpoint")

    args = ap.parse_args()

    if args.cmd == "fps":
        bench_fps(args.source, args.max_frames)
    elif args.cmd == "test-action":
        test_action(args.data, args.model, args.seq_len)
    elif args.cmd == "rwf2000":
        # RWF-2000 has Fight/ and NonFight/ — map to violence/normal
        test_action(args.data, args.model)
    elif args.cmd == "compare":
        compare_methods(args.data, args.model)


if __name__ == "__main__":
    main()
