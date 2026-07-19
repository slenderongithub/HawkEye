"""Training scaffold: skeleton-based action classifier (fight / fall / normal).

This is the upgrade path that replaces the zero-training heuristics in
`hawkeye/detect/heuristics.py`. It trains a small temporal model on YOLO-pose
keypoint sequences — the standard, camera-motion-robust approach for violence
detection (basis: RWF-2000, RTVD-Net, skeleton-based fight detection, 2023-2025).

Pipeline:
  1. Extract keypoint sequences from labeled clips (RWF-2000 fights + a fall set).
  2. Train the temporal classifier below.
  3. Export weights; wire them into StageA behind the same candidate interface.

Datasets to download (not included):
  - RWF-2000  (fights)      : 2,000 surveillance clips, violent/non-violent
  - UR Fall / Le2i          : falls vs normal
Layout expected by `build_dataset`:  <root>/<label>/<clip>.mp4

Run (once a dataset is present):
  python scripts/train_action.py --data data/datasets/rwf2000 --epochs 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEQ_LEN = 32          # frames per sample
KP = 17               # COCO keypoints
FEAT = KP * 3         # x, y, conf per keypoint


def extract_sequence(video_path: str, pose, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Run YOLO-pose over a clip -> (seq_len, FEAT) normalized keypoint sequence."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = pose.detect(frame)
        if dets:
            p = max(dets, key=lambda d: d.conf)          # dominant person
            kp = p.keypoints.copy().astype(np.float32)
            h, w = frame.shape[:2]
            kp[:, 0] /= w
            kp[:, 1] /= h                                  # normalize to [0,1]
            frames.append(kp.flatten())
        else:
            frames.append(np.zeros(FEAT, dtype=np.float32))
    cap.release()
    if not frames:
        return np.zeros((seq_len, FEAT), dtype=np.float32)
    idx = np.linspace(0, len(frames) - 1, seq_len).round().astype(int)
    return np.stack([frames[i] for i in idx])


def build_dataset(root: str, pose):
    """Yield (sequence, label_index). Expects <root>/<label>/<clip>.* layout."""
    root_p = Path(root)
    labels = sorted([d.name for d in root_p.iterdir() if d.is_dir()])
    if not labels:
        raise SystemExit(f"No label folders under {root}. Expected <root>/<label>/<clip>.mp4")
    print(f"Labels: {labels}")
    X, y = [], []
    for li, lab in enumerate(labels):
        for clip in (root_p / lab).glob("*.*"):
            X.append(extract_sequence(str(clip), pose))
            y.append(li)
            print(f"  {lab}/{clip.name}", end="\r")
    return np.stack(X), np.array(y), labels


def train(data_root: str, epochs: int, out: str) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from hawkeye.detect.pose import PoseDetector

    pose = PoseDetector().load()
    X, y, labels = build_dataset(data_root, pose)
    print(f"\nDataset: {X.shape}, classes={labels}")

    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    dl = DataLoader(ds, batch_size=16, shuffle=True)

    class GRUClassifier(nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            self.gru = nn.GRU(FEAT, 128, num_layers=2, batch_first=True, dropout=0.2)
            self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes))

        def forward(self, x):
            _, h = self.gru(x)
            return self.head(h[-1])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = GRUClassifier(len(labels)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        tot = correct = 0
        run = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            run += loss.item()
            correct += (out.argmax(1) == yb).sum().item()
            tot += len(yb)
        print(f"epoch {ep+1}/{epochs}  loss={run/len(dl):.3f}  acc={correct/tot:.3f}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "labels": labels}, out)
    print(f"Saved -> {out}. Wire this into hawkeye/detect/ as a trained candidate source.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="<root>/<label>/<clip>.mp4")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default="models/action_gru.pt")
    args = ap.parse_args()
    train(args.data, args.epochs, args.out)


if __name__ == "__main__":
    main()
