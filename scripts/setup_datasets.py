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
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _kaggle_utils import authenticate, download  # noqa: E402

# Expected subdirectories matching the train_action.py convention
CLASS_DIRS = ["violence", "fall", "normal"]

# Kaggle dataset candidates per source. Slugs are UNSTABLE — verify with
# `kaggle datasets list -s <term>`. The downloader tries each in order and
# continues on failure, so a stale slug never aborts the run.
KAGGLE_SOURCES = {
    # violence + its own non-fight split (-> normal via ALIASES)
    "rwf2000": [
        "vulamnguyen/rwf2000",
        "iliaskomiv/rwf2000-video-database-for-fight-detection",
    ],
    # extra violence (hockey + movies fights)
    "hockey": [
        "yassershrief/hockey-fight-vidoes",
        "vulamnguyen/hockey-fight",
    ],
    # a Kaggle fall mirror (hands-off; UR Fall / Le2i stay manual)
    "fall": [
        "uttejkumarkandagatla/fall-detection-dataset",
        "tuyenldvn/falldataset",
    ],
    # normal hard-negatives (filtered to daily-living classes below)
    "ucf101": [
        "matthewjansen/ucf101-action-recognition",
    ],
}

# UCF101 action classes that read as ordinary indoor activity — good hard
# negatives for the "normal" class. Everything else in UCF101 is ignored so the
# normal class stays representative of surveillance footage, not sports/dance.
UCF101_NORMAL_CLASSES = {
    "walkingwithdog", "typing", "writingonboard", "brushingteeth", "blowdryhair",
    "applyeyemakeup", "applylipstick", "haircut", "moppingfloor", "knitting",
    "mixing", "cuttinginkitchen", "shavingbeard", "handstandwalking",
}

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

    Handles nested structures like:
      <src>/RWF-2000/train/Fight/<clip>.mp4
      <src>/RWF-2000/val/NonFight/<clip>.mp4
      <src>/<class_name>/<clip>.mp4
    """
    stats = {c: 0 for c in CLASS_DIRS}

    if not src.exists():
        sys.exit(f"Source directory not found: {src}")

    # Recursively find all subdirectories that match class names or aliases
    all_dirs = [d for d in src.rglob("*") if d.is_dir()]
    # Include src itself in case user pointed directly inside a class folder
    all_dirs.append(src)

    found_class_dirs = False
    for d in all_dirs:
        raw_name = d.name.lower().strip()
        canonical = ALIASES.get(raw_name, raw_name)
        if canonical in CLASS_DIRS:
            found_class_dirs = True
            clips = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXT]
            copied_count = 0
            for clip in clips:
                dest = dst / canonical / clip.name
                if dest.exists() and dest.resolve() != clip.resolve():
                    # Generate a short counter prefix if file with same name exists
                    cnt = 1
                    stem, ext = clip.stem, clip.suffix
                    # Keep total filename length reasonable for Windows MAX_PATH
                    if len(stem) > 50:
                        stem = stem[-50:]
                    while (dst / canonical / f"{stem}_{cnt}{ext}").exists():
                        cnt += 1
                    dest = dst / canonical / f"{stem}_{cnt}{ext}"

                if not dest.exists():
                    try:
                        shutil.copy2(str(clip), str(dest))
                        copied_count += 1
                        stats[canonical] += 1
                    except Exception as err:
                        print(f"  ⚠ Failed to copy {clip.name}: {err}")
            if clips:
                print(f"  [{canonical}] copied {copied_count} clips from {d.relative_to(src)}/")

    if not found_class_dirs:
        # Flat directory — prompt user for class label
        clips = [f for f in src.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXT]
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


def organize_ucf101_normals(raw_dir: Path, dst: Path) -> int:
    """Copy only the allowlisted UCF101 daily-living classes into normal/.

    UCF101 ships 101 class folders (nested under train/val/test in most Kaggle
    mirrors). We pull just the ordinary-activity classes so the normal class
    stays representative of indoor surveillance, not unrelated sports/dance.
    """
    copied = 0
    normal_dst = dst / "normal"
    normal_dst.mkdir(parents=True, exist_ok=True)
    for d in raw_dir.rglob("*"):
        if not d.is_dir():
            continue
        if d.name.lower() not in UCF101_NORMAL_CLASSES:
            continue
        clips = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXT]
        for clip in clips:
            dest = normal_dst / f"ucf_{d.name}_{clip.name}"
            if not dest.exists():
                try:
                    shutil.copy2(str(clip), str(dest))
                    copied += 1
                except Exception as err:  # noqa: BLE001
                    print(f"  ⚠ Failed to copy {clip.name}: {err}")
    if copied:
        print(f"  [normal] copied {copied} UCF101 daily-living clips")
    else:
        print("  ⚠ UCF101: no allowlisted daily-living class folders found "
              "(mirror layout may differ) — skipping.")
    return copied


def download_from_kaggle(root: Path, classes: list[str], config_path: Path | None) -> set[str]:
    """Auto-download the Kaggle-hosted action datasets into ``root``.

    Downloads each source into a scratch dir, then reuses ``organize_from_local``
    (RWF-2000/hockey → alias-matched class folders) and ``organize_ucf101_normals``
    (UCF101 → filtered normals). Returns the set of source keys that succeeded, so
    the caller can print manual instructions only for what's missing.
    """
    # authenticate() handles a missing token / uninstalled package itself
    # (prints instructions, returns None).
    api = authenticate(config_path)
    if api is None:
        return set()

    raw_root = root / "_raw_kaggle"
    raw_root.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()

    want_violence = "violence" in classes
    want_normal = "normal" in classes
    want_fall = "fall" in classes

    # RWF-2000: contains BOTH Fight (->violence) and NonFight (->normal).
    if want_violence or want_normal:
        got = download(api, KAGGLE_SOURCES["rwf2000"], raw_root / "rwf2000")
        if got:
            organize_from_local(got, root)
            completed.add("rwf2000")

    # Hockey/Movies fights -> extra violence.
    if want_violence:
        got = download(api, KAGGLE_SOURCES["hockey"], raw_root / "hockey")
        if got:
            organize_from_local(got, root)
            completed.add("hockey")

    # Fall mirror.
    if want_fall:
        got = download(api, KAGGLE_SOURCES["fall"], raw_root / "fall")
        if got:
            organize_from_local(got, root)
            completed.add("fall")

    # UCF101 -> filtered daily-living normals.
    if want_normal:
        got = download(api, KAGGLE_SOURCES["ucf101"], raw_root / "ucf101")
        if got and organize_ucf101_normals(got, root):
            completed.add("ucf101")

    print(f"\n  Kaggle download complete. Sources organized: {sorted(completed) or 'none'}")
    print(f"  (raw downloads kept under {raw_root}/ — delete to reclaim disk)")
    return completed


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


# Sources that are NOT on Kaggle and always require a manual download.
MANUAL_SOURCES = {
    "UR Fall Detection (fall)":
        "http://fenix.ur.edu.pl/~mkepski/ds/uf.html  ->  copy clips to <out>/fall/",
    "Le2i Fall Detection (fall)":
        "search 'Le2i Fall Detection Dataset'  ->  copy clips to <out>/fall/",
    "Surveillance Camera Fight Dataset (violence)":
        "search 'Surveillance Camera Fight Dataset' (paper GitHub/Drive)  ->  <out>/violence/",
}


def print_download_instructions(completed: "set[str] | None" = None) -> None:
    """Print manual-download steps for sources not auto-fetched from Kaggle."""
    completed = completed or set()
    print("\n" + "=" * 64)
    print("  MANUAL / FALLBACK DATASET DOWNLOADS")
    print("=" * 64)

    kaggle_missing = [k for k in KAGGLE_SOURCES if k not in completed]
    if kaggle_missing:
        print("\n  Kaggle sources NOT auto-downloaded this run "
              f"({', '.join(kaggle_missing)}):")
        print("    -> check `kaggle datasets list -s <name>` for a current slug,")
        print("       then re-run:  python scripts/setup_datasets.py --source kaggle")

    print("\n  Sources that are never on Kaggle (download by hand, then organize):")
    for name, how in MANUAL_SOURCES.items():
        print(f"    - {name}\n        {how}")

    print("\n  After a manual download, sort it into class folders with:")
    print("    python scripts/setup_datasets.py --source local \\")
    print("      --local-dir <download_folder> --out data/datasets/actions")
    print("=" * 64)


def check_disk_space(path: Path, min_free_gb: float = 30.0) -> None:
    """Warn (don't block) if free disk under ``path`` is below a threshold."""
    try:
        usage = shutil.disk_usage(path if path.exists() else (path.anchor or "."))
    except Exception:  # noqa: BLE001
        return
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_free_gb:
        print(f"\n  ⚠ Only {free_gb:.1f} GB free at {path}. The full dataset "
              f"combination can exceed {min_free_gb:.0f} GB (raw video + copies).")
        print("    Free up space or point --out at a larger drive before downloading.\n")
    else:
        print(f"\n  ✓ {free_gb:.1f} GB free at {path}.")


def main():
    ap = argparse.ArgumentParser(
        description="Set up training datasets for HawkEye action classifier"
    )
    ap.add_argument("--out", default="data/datasets/actions",
                    help="Output directory for organized dataset")
    ap.add_argument("--source", choices=["local", "info", "kaggle"], default="info",
                    help="'kaggle' to auto-download via the Kaggle API, "
                         "'local' to organize a folder you downloaded by hand, "
                         "'info' to print download instructions")
    ap.add_argument("--local-dir", dest="local_dir", default="",
                    help="Path to raw downloaded dataset folder (with --source local)")
    ap.add_argument("--classes", nargs="+", default=CLASS_DIRS, choices=CLASS_DIRS,
                    help="Subset of classes to fetch (with --source kaggle)")
    ap.add_argument("--kaggle-config", dest="kaggle_config", default="",
                    help="Path to kaggle.json (default: ~/.kaggle/kaggle.json)")
    ap.add_argument("--verify-only", action="store_true", dest="verify_only",
                    help="Just check the current dataset without modifying anything")

    args = ap.parse_args()
    root = Path(args.out)
    kaggle_cfg = Path(args.kaggle_config) if args.kaggle_config else None

    if args.verify_only:
        verify_dataset(root)
        return

    if args.source == "info":
        print_download_instructions()
        ensure_dirs(root)
        verify_dataset(root)
        return

    if args.source == "kaggle":
        ensure_dirs(root)
        check_disk_space(root)
        completed = download_from_kaggle(root, args.classes, kaggle_cfg)
        print_download_instructions(completed)
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
