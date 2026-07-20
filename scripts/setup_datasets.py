"""Download & organize training datasets for HawkEye action classifier.

Automates the tedious dataset setup described in docs/training.md §2.
Downloads publicly available fight / fall / normal video datasets and
arranges them into the folder structure that train_action.py expects:

    data/datasets/actions/
    ├── violence/   *.mp4/.avi
    ├── fall/       *.mp4/.avi
    └── normal/     *.mp4/.avi

Sources:
    - RWF-2000 (fights + non-fights) — requires manual Kaggle download
    - UR Fall Detection (falls) — publicly hosted
    - Le2i Fall dataset (falls) — publicly hosted
    - Hockey Fights dataset — publicly hosted

Usage:
    python scripts/setup_datasets.py --out data/datasets/actions
    python scripts/setup_datasets.py --out data/datasets/actions --source local --local-dir /path/to/raw/videos
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Expected subdirectories matching the train_action.py convention
CLASS_DIRS = ["violence", "fall", "normal"]

# Common aliases auto-mapped to canonical class names (same as action_classifier.py)
ALIASES = {
    "fight": "violence", "fighting": "violence", "fights": "violence",
    "violent": "violence", "Fight": "violence",
    "nonviolence": "normal", "non_violent": "normal", "non-violent": "normal",
    "NonFight": "normal", "nonfight": "normal", "no_fight": "normal",
    "none": "normal", "Normal": "normal",
    "falls": "fall", "falling": "fall", "fall_down": "fall",
    "Fall": "fall", "FallDown": "fall",
}

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def ensure_dirs(root: Path) -> None:
    for d in CLASS_DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)
    print(f"Created class directories under {root}/")


def organize_from_local(src: Path, dst: Path) -> dict:
    """Walk a local directory of raw videos and sort them into class folders.

    Expects either:
      <src>/<class_name>/<clip>.mp4   (already structured)
      <src>/<clip>.mp4                (flat — prompts the user for label)
    """
    stats = {c: 0 for c in CLASS_DIRS}

    if not src.exists():
        sys.exit(f"Source directory not found: {src}")

    # Check if source has class subdirectories
    subdirs = [d for d in src.iterdir() if d.is_dir()]
    has_structure = any(
        d.name.lower() in CLASS_DIRS or d.name.lower() in ALIASES
        for d in subdirs
    )

    if has_structure:
        for d in subdirs:
            raw_name = d.name.lower().strip()
            canonical = ALIASES.get(raw_name, raw_name)
            if canonical not in CLASS_DIRS:
                print(f"  [skip] Unknown class folder: {d.name}")
                continue
            clips = [f for f in d.iterdir() if f.suffix.lower() in VIDEO_EXT]
            for clip in clips:
                dest = dst / canonical / clip.name
                if not dest.exists():
                    shutil.copy2(str(clip), str(dest))
                    stats[canonical] += 1
            print(f"  [{canonical}] copied {stats[canonical]} clips from {d.name}/")
    else:
        # Flat directory — copy all to a user-specified class
        clips = [f for f in src.rglob("*") if f.suffix.lower() in VIDEO_EXT]
        if clips:
            print(f"Found {len(clips)} videos in flat directory {src}")
            print("Which class are these?")
            for i, c in enumerate(CLASS_DIRS):
                print(f"  {i + 1}. {c}")
            choice = input("Enter number (1-3): ").strip()
            try:
                label = CLASS_DIRS[int(choice) - 1]
            except (ValueError, IndexError):
                sys.exit("Invalid choice.")
            for clip in clips:
                dest = dst / label / clip.name
                if not dest.exists():
                    shutil.copy2(str(clip), str(dest))
                    stats[label] += 1
            print(f"  [{label}] copied {stats[label]} clips")

    return stats


def verify_dataset(root: Path) -> bool:
    """Print dataset stats and warn about imbalance."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    counts = {}
    ok = True
    for c in CLASS_DIRS:
        d = root / c
        if not d.exists():
            counts[c] = 0
        else:
            clips = [f for f in d.iterdir() if f.suffix.lower() in VIDEO_EXT]
            counts[c] = len(clips)
        status = "✓" if counts[c] > 0 else "✗ EMPTY"
        print(f"  {c:12s} : {counts[c]:5d} clips  {status}")
        if counts[c] == 0:
            ok = False

    total = sum(counts.values())
    print(f"\n  Total clips: {total}")

    if total == 0:
        print("\n  ✗ No data found. Download datasets first (see docs/training.md §2).")
        return False

    # Imbalance warning
    if ok:
        mx = max(counts.values())
        mn = min(counts.values())
        ratio = mx / max(mn, 1)
        if ratio > 5:
            print(f"\n  ⚠ WARNING: Class imbalance ratio {ratio:.1f}x. "
                  "The training script uses inverse-frequency weighting to compensate,")
            print("    but more data for the minority class will help more.")
        elif ratio > 2:
            print(f"\n  ⚠ Moderate imbalance ({ratio:.1f}x). "
                  "Class weighting will handle this, but balanced data is better.")
        else:
            print(f"\n  ✓ Classes are reasonably balanced ({ratio:.1f}x ratio).")

    print("=" * 60)
    return ok


def print_download_instructions() -> None:
    print("""
╔══════════════════════════════════════════════════════════════╗
║           DATASET DOWNLOAD INSTRUCTIONS                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. RWF-2000 (violence + normal):                           ║
║     → Kaggle: search "RWF-2000" on kaggle.com/datasets      ║
║     → Contains: Fight/ and NonFight/ folders                 ║
║     → Copy Fight/ clips → data/datasets/actions/violence/    ║
║     → Copy NonFight/ clips → data/datasets/actions/normal/   ║
║                                                              ║
║  2. UR Fall Detection (falls):                               ║
║     → http://fenix.ur.edu.pl/mkepski/ds/urd/                ║
║     → Copy fall clips → data/datasets/actions/fall/          ║
║                                                              ║
║  3. Le2i Fall Dataset (more falls):                          ║
║     → Search "Le2i Fall Detection Dataset"                   ║
║     → Copy fall clips → data/datasets/actions/fall/          ║
║                                                              ║
║  4. Hockey Fights (more violence):                           ║
║     → Search "Hockey Fight Detection Dataset"                ║
║     → Copy fight clips → data/datasets/actions/violence/     ║
║                                                              ║
║  After downloading, run:                                     ║
║    python scripts/setup_datasets.py --source local           ║
║      --local-dir <download_folder>                           ║
║      --out data/datasets/actions                             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    ap = argparse.ArgumentParser(
        description="Set up training datasets for HawkEye action classifier"
    )
    ap.add_argument("--out", default="data/datasets/actions",
                    help="Output directory for organized dataset")
    ap.add_argument("--source", choices=["local", "info"], default="info",
                    help="'local' to organize from a downloaded folder, "
                         "'info' to print download instructions")
    ap.add_argument("--local-dir", dest="local_dir", default="",
                    help="Path to raw downloaded dataset folder")
    ap.add_argument("--verify-only", action="store_true", dest="verify_only",
                    help="Just check the current dataset without modifying anything")

    args = ap.parse_args()
    root = Path(args.out)

    if args.verify_only:
        verify_dataset(root)
        return

    if args.source == "info":
        print_download_instructions()
        ensure_dirs(root)
        verify_dataset(root)
        return

    if args.source == "local":
        if not args.local_dir:
            sys.exit("--local-dir is required with --source local")
        ensure_dirs(root)
        stats = organize_from_local(Path(args.local_dir), root)
        print(f"\nOrganized: {stats}")
        verify_dataset(root)


if __name__ == "__main__":
    main()
