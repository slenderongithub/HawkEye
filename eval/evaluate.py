"""Evaluation + benchmarking for HawkEye.

Commands:
  * fps          — Stage A latency and throughput on any video/webcam.
  * test-action  — Run the trained action classifier over a directory of clips,
                   producing a confusion matrix + per-class P/R/F1.
  * rwf2000      — Clip-level accuracy on an RWF-2000-style Fight/NonFight tree.
  * compare      — Compare zero-training heuristics vs the trained model.

Every path here goes through the SAME feature extraction and standardisation the
live pipeline uses (`hawkeye.detect.skeleton_features`), including a real scene
motion score. An earlier revision extracted features with `motion=0.0` here and
in the trainer while the live pipeline supplied a real value, and hand-rolled
`(x - mean) / std` in three separate places — which is how a model that scored
0.95 offline ended up behaving randomly on a camera. There is one implementation
now, and evaluation exercises it.

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
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hawkeye.config import load_config           # noqa: E402
from hawkeye.detect import StageA                 # noqa: E402
from hawkeye.ingest import open_source            # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
# Folder-name aliases so an untouched RWF-2000 / Hockey tree evaluates directly.
FOLDER_ALIASES = {
    "fight": "violence", "fights": "violence", "fighting": "violence", "violent": "violence",
    "nonfight": "normal", "non_fight": "normal", "nonviolence": "normal",
    "non_violent": "normal", "no_fight": "normal", "none": "normal",
    "falls": "fall", "falling": "fall", "fall_down": "fall",
}


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
        print(f"  {l:<{w}} {prec[i]:>10.3f} {rec[i]:>10.3f} {f1[i]:>10.3f} {int(cm[i].sum()):>10}")
    print(f"  {'─' * (w + 44)}")
    print(f"  {'macro avg':<{w}} {prec.mean():>10.3f} {rec.mean():>10.3f} "
          f"{f1.mean():>10.3f} {int(cm.sum()):>10}")

    accuracy = tp.sum() / max(cm.sum(), 1)
    print(f"\n  Overall accuracy: {accuracy:.3f} ({int(tp.sum())}/{int(cm.sum())})")
    print(f"  Macro F1: {f1.mean():.3f}")

    # The pair this whole pipeline exists to keep apart.
    if "violence" in labels and "fall" in labels:
        iv, ifl = labels.index("violence"), labels.index("fall")
        v_as_f = cm[iv, ifl] / max(cm[iv].sum(), 1)
        f_as_v = cm[ifl, iv] / max(cm[ifl].sum(), 1)
        print(f"  violence→fall: {v_as_f:.1%}   fall→violence: {f_as_v:.1%}")
    return float(f1.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Shared model + feature helpers (one implementation, used by every command)
# ─────────────────────────────────────────────────────────────────────────────
class LoadedModel:
    """A trained checkpoint plus the exact preprocessing it was trained with."""

    def __init__(self, model_path: str, device: str = "auto"):
        import torch

        from hawkeye.detect.action_model import build_model
        from hawkeye.detect.pose import resolve_device

        self.device = resolve_device(device)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.labels = list(ckpt["labels"])
        self.seq_len = int(ckpt["seq_len"])
        self.mean = np.asarray(ckpt["mean"], dtype=np.float32)
        self.std = np.asarray(ckpt["std"], dtype=np.float32)
        self.thresholds = dict(ckpt.get("thresholds", {}))
        self.temperature = float(ckpt.get("temperature", 1.0))
        self.reported = {
            "val_macro_f1": ckpt.get("val_macro_f1"),
            "test_macro_f1": ckpt.get("test_macro_f1"),
        }
        model = build_model(int(ckpt["feat_dim"]), len(self.labels),
                            int(ckpt.get("hidden", 192)), int(ckpt.get("embed", 192)),
                            bool(ckpt.get("use_margin", False)))
        model.load_state_dict(ckpt["state_dict"])
        model.temperature.fill_(max(self.temperature, 1e-2))
        model.to(self.device).eval()
        self.model = model

    def predict(self, feats) -> tuple:
        """Clip features -> (label, confidence). Uses the shared standardiser."""
        import torch

        from hawkeye.detect.skeleton_features import (build_sequence, resample,
                                                      standardize)

        seq = resample(build_sequence(list(feats)), self.seq_len)
        seq = standardize(seq, self.mean, self.std)
        x = torch.from_numpy(seq[None]).float().to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model.calibrated_logits(x), dim=1)[0].cpu().numpy()
        k = int(probs.argmax())
        return self.labels[k], float(probs[k])


def extract_clip_features(clip_path: str, pose, want_heuristics: bool = False):
    """Per-frame features for a clip, with the real motion score.

    Returns (features, heuristic_votes). This mirrors
    ``scripts/train_action.extract_clip`` exactly — same pose detector, same
    ``MotionAnomaly``, same ``frame_feature`` call.
    """
    import cv2

    from hawkeye.detect import heuristics
    from hawkeye.detect.scene_anomaly import MotionAnomaly
    from hawkeye.detect.skeleton_features import frame_feature

    cap = cv2.VideoCapture(str(clip_path))
    motion = MotionAnomaly()
    feats, votes = [], []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        h, w = frame.shape[:2]
        mot = motion.update(frame)
        persons = pose.detect(frame)
        feats.append(frame_feature(persons, mot, w, h))
        if want_heuristics:
            cands = heuristics.analyze(persons, mot)
            votes.append(max(cands, key=lambda c: c[1])[0].value if cands else "normal")
    cap.release()
    return feats, votes


def iter_labelled_clips(root: Path, labels):
    """Yield (label_index, clip_path) for every clip under <root>/<label>/.

    Folder names are alias-mapped, so an unmodified RWF-2000 (Fight/NonFight)
    tree can be evaluated against a violence/normal model without renaming.
    """
    canon = {l.lower(): i for i, l in enumerate(labels)}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        name = folder.name.lower()
        key = FOLDER_ALIASES.get(name, name)
        if key not in canon:
            print(f"  [skip] {folder.name}/ — not a class this model knows {labels}")
            continue
        for clip in sorted(folder.rglob("*")):
            if clip.is_file() and clip.suffix.lower() in VIDEO_EXT:
                yield canon[key], clip


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
        print(f"frames={n}  avg_latency={1000 * lat / n:.1f} ms  "
              f"throughput={n / wall:.1f} FPS  triggers={triggers}")
    else:
        print("No frames read.")


# ─────────────────────────────────────────────────────────────────────────────
# test-action
# ─────────────────────────────────────────────────────────────────────────────
def test_action(data_dir: str, model_path: str) -> None:
    from hawkeye.detect.pose import PoseDetector

    print(f"\n  Loading model from {model_path} ...")
    lm = LoadedModel(model_path)
    print(f"  Labels: {lm.labels}, seq_len={lm.seq_len}, device={lm.device}, "
          f"temperature={lm.temperature:.3f}")
    if lm.reported["test_macro_f1"] is not None:
        print(f"  Checkpoint's own held-out test macro-F1: {lm.reported['test_macro_f1']:.3f}")
    print("\n  NOTE: this scores every clip in the directory. If that directory is\n"
          "  the training set, the number below is optimistic and means little —\n"
          "  the honest figure is the checkpoint's test macro-F1 above, or point\n"
          "  --data at a dataset the model has never seen.\n")

    pose = PoseDetector().load()
    root = Path(data_dir)
    all_true, all_pred = [], []
    per_class = Counter()
    per_class_ok = Counter()

    clips = list(iter_labelled_clips(root, lm.labels))
    for n, (li, clip) in enumerate(clips, 1):
        feats, _ = extract_clip_features(clip, pose)
        if len(feats) < 8:
            continue
        label, conf = lm.predict(feats)
        pred = lm.labels.index(label)
        all_true.append(li)
        all_pred.append(pred)
        per_class[lm.labels[li]] += 1
        per_class_ok[lm.labels[li]] += int(pred == li)
        print(f"    [{n}/{len(clips)}] {clip.name[:38]:<38} true={lm.labels[li]:<9} "
              f"pred={label:<9} ({conf:.2f}) {'✓' if pred == li else '✗'}", end="\r")

    print(" " * 110, end="\r")
    for lab in lm.labels:
        if per_class[lab]:
            print(f"  [{lab}] {per_class_ok[lab]}/{per_class[lab]} correct "
                  f"({100 * per_class_ok[lab] / per_class[lab]:.1f}%)")
    if all_true:
        print_report(confusion(np.array(all_true), np.array(all_pred), len(lm.labels)),
                     lm.labels)
    else:
        print("  No clips processed.")


# ─────────────────────────────────────────────────────────────────────────────
# compare: heuristics vs trained model
# ─────────────────────────────────────────────────────────────────────────────
def compare_methods(data_dir: str, model_path: str) -> None:
    from hawkeye.detect.pose import PoseDetector

    print(f"\n  Comparing heuristics vs trained model ({model_path})")
    lm = LoadedModel(model_path)
    pose = PoseDetector().load()
    root = Path(data_dir)

    clips = list(iter_labelled_clips(root, lm.labels))
    heur_true, heur_pred, model_pred = [], [], []

    for n, (li, clip) in enumerate(clips, 1):
        feats, votes = extract_clip_features(clip, pose, want_heuristics=True)
        if len(feats) < 8:
            continue
        label, _ = lm.predict(feats)
        # Heuristic clip verdict: majority vote over its per-frame candidates.
        hv = Counter(votes).most_common(1)[0][0] if votes else "normal"
        if hv in lm.labels:
            hidx = lm.labels.index(hv)
        elif "normal" in lm.labels:
            # The heuristics can emit types this model has no class for; scoring
            # them as "normal" is the honest reading of "it found nothing we
            # asked about" — never fall back to the true label, which would
            # silently hand the baseline free correct answers.
            hidx = lm.labels.index("normal")
        else:
            continue
        heur_true.append(li)
        model_pred.append(lm.labels.index(label))
        heur_pred.append(hidx)
        print(f"    [{n}/{len(clips)}] {clip.name[:40]:<40}", end="\r")

    print(" " * 90, end="\r")
    if not heur_true:
        print("  No clips processed.")
        return

    yt = np.array(heur_true)
    print("\n  ── Zero-training heuristics ──")
    h_f1 = print_report(confusion(yt, np.array(heur_pred), len(lm.labels)), lm.labels)
    print("\n  ── Trained model ──")
    m_f1 = print_report(confusion(yt, np.array(model_pred), len(lm.labels)), lm.labels)

    print(f"\n  Macro-F1: heuristics {h_f1:.3f} → trained {m_f1:.3f} "
          f"({m_f1 - h_f1:+.3f})")
    if m_f1 <= h_f1:
        print("  ⚠  The trained model does NOT beat the zero-training baseline.\n"
              "     Do not ship it — retrain with more/cleaner data before wiring it\n"
              "     into configs/default.yaml.")


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
  test-action   Evaluate the trained action classifier on a directory of clips
  rwf2000       Evaluate on an RWF-2000-style Fight/NonFight tree
  compare       Compare heuristics vs trained model side-by-side
        """
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fps", help="Benchmark Stage A FPS")
    f.add_argument("--source", default="0")
    f.add_argument("--max-frames", type=int, default=300)

    t = sub.add_parser("test-action", help="Evaluate trained model on a clip directory")
    t.add_argument("--data", required=True, help="Root dir with <label>/<clip>.mp4")
    t.add_argument("--model", required=True, help="Path to trained .pt checkpoint")

    r = sub.add_parser("rwf2000", help="Evaluate on an RWF-2000 style dataset")
    r.add_argument("--data", required=True,
                   help="RWF-2000 root (Fight/ and NonFight/ subdirs)")
    r.add_argument("--model", required=True, help="Trained .pt checkpoint")

    c = sub.add_parser("compare", help="Compare heuristics vs trained model")
    c.add_argument("--data", required=True, help="Dataset root with class subdirs")
    c.add_argument("--model", required=True, help="Trained .pt checkpoint")

    args = ap.parse_args()

    if args.cmd == "fps":
        bench_fps(args.source, args.max_frames)
    elif args.cmd in ("test-action", "rwf2000"):
        test_action(args.data, args.model)
    elif args.cmd == "compare":
        compare_methods(args.data, args.model)


if __name__ == "__main__":
    main()
