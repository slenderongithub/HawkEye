"""Evaluation + benchmarking for HawkEye.

Runnable now:
  * FPS / latency benchmark of Stage A on any video or webcam.

Scaffolded (need the datasets downloaded):
  * UCF-Crime frame-level AUC (weakly-supervised anomaly)
  * RWF-2000 clip accuracy (fights)
  * false-alarm rate with vs without Stage B  ← headline thesis result

Usage:
  python eval/evaluate.py fps --source data/datasets/sample.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hawkeye.config import load_config           # noqa: E402
from hawkeye.detect import StageA                 # noqa: E402
from hawkeye.ingest import open_source            # noqa: E402


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


def _todo(name: str) -> None:
    print(f"[scaffold] {name}: download the dataset and implement extraction here.")
    print("  UCF-Crime : https://www.crcv.ucf.edu/projects/real-world/")
    print("  RWF-2000  : fights (Kaggle / paper repo)")
    print("  Metric plan: I3D/CLIP features -> MIL scorer -> frame AUC (UCF-Crime);")
    print("  clip accuracy on RWF-2000; and false-alarm rate Stage-A-only vs A+B.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fps"); f.add_argument("--source", default="0")
    f.add_argument("--max-frames", type=int, default=300)
    sub.add_parser("ucf-crime")
    sub.add_parser("rwf2000")
    args = ap.parse_args()

    if args.cmd == "fps":
        bench_fps(args.source, args.max_frames)
    elif args.cmd == "ucf-crime":
        _todo("UCF-Crime AUC")
    elif args.cmd == "rwf2000":
        _todo("RWF-2000 accuracy")


if __name__ == "__main__":
    main()
