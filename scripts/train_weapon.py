"""Train a YOLO weapon detector for HawkEye (knife / gun / etc.)

This script orchestrates the weapon detection model training described in
docs/training.md §5. It fine-tunes YOLO11 on a custom weapon dataset in
YOLO format and produces a .pt checkpoint that plugs into HawkEye's
detect.weapon_model config.

Dataset layout (YOLO format):
    <root>/
    ├── images/
    │   ├── train/   *.jpg/*.png
    │   └── val/     *.jpg/*.png
    ├── labels/
    │   ├── train/   *.txt (YOLO format: class cx cy w h)
    │   └── val/     *.txt
    └── weapons.yaml
        # Contents:
        # path: .
        # train: images/train
        # val: images/val
        # names: [knife, pistol, rifle]

Usage:
    python scripts/train_weapon.py --data data/datasets/weapons/weapons.yaml
    python scripts/train_weapon.py --data data/datasets/weapons/weapons.yaml --epochs 100 --imgsz 640
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def create_weapons_yaml(data_dir: str, class_names: list[str], output: str) -> str:
    """Generate a weapons.yaml data config for ultralytics YOLO training."""
    import yaml

    data_dir = str(Path(data_dir).resolve())
    config = {
        "path": data_dir,
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }

    with open(output, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"  Created data config: {output}")
    print(f"  Classes: {class_names}")
    return output


def verify_dataset(data_yaml: str) -> bool:
    """Check that the dataset is properly structured."""
    import yaml

    print("\n  Verifying dataset structure...")

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = Path(data_yaml).parent / root

    train_img = root / cfg["train"]
    val_img = root / cfg["val"]

    issues = []
    for split, folder in [("train", train_img), ("val", val_img)]:
        if not folder.exists():
            issues.append(f"    ✗ {split} images folder missing: {folder}")
        else:
            imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
            print(f"    {split}: {len(imgs)} images")
            if len(imgs) == 0:
                issues.append(f"    ✗ {split} has no images")

    if "names" in cfg:
        names = cfg["names"]
        if isinstance(names, dict):
            names = list(names.values())
        print(f"    Classes: {names}")

    if issues:
        for i in issues:
            print(i)
        return False

    print("    ✓ Dataset structure looks good")
    return True


def train(args) -> None:
    """Fine-tune YOLO11 on the weapon dataset."""
    from ultralytics import YOLO

    print("\n" + "═" * 60)
    print("  HAWKEYE — WEAPON DETECTOR TRAINING")
    print("═" * 60)

    # Verify
    if not Path(args.data).exists():
        sys.exit(f"Data config not found: {args.data}")

    if not verify_dataset(args.data):
        sys.exit("Dataset verification failed. Fix the issues above.")

    # Load base model
    print(f"\n  Base model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")
    print(f"  Device: {args.device}")

    model = YOLO(args.model)

    t0 = time.time()
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device != "auto" else None,
        project="runs/weapon_detect",
        name="train",
        exist_ok=True,
        verbose=True,
    )
    elapsed = time.time() - t0

    possible_bests = [
        Path("runs/detect/runs/weapon_detect/train/weights/best.pt"),
        Path("runs/weapon_detect/train/weights/best.pt"),
    ]
    possible_lasts = [
        Path("runs/detect/runs/weapon_detect/train/weights/last.pt"),
        Path("runs/weapon_detect/train/weights/last.pt"),
    ]

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    found = False
    for best_pt in possible_bests:
        if best_pt.exists():
            shutil.copy2(str(best_pt), str(out_p))
            print(f"\n  ✓ Best weights copied to: {out_p}")
            found = True
            break

    if not found:
        for last_pt in possible_lasts:
            if last_pt.exists():
                shutil.copy2(str(last_pt), str(out_p))
                print(f"\n  ✓ Last weights copied to: {out_p}")
                found = True
                break

    if not found:
        print("\n  ⚠ Could not find trained weights.")
        return

    print(f"  Training time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Print config instructions
    print(f"\n  ─── Wire into HawkEye ───")
    print(f"  Edit configs/default.yaml:")
    print(f"    detect:")
    print(f"      weapon_model: \"{out_p}\"")
    print(f"      weapon_conf: 0.4")
    print(f"\n  Restart HawkEye. Weapon detections will route to `police` via")
    print(f"  the decision engine (WEAPON_KEYWORDS in hawkeye/detect/weapons.py).")

    # Validate
    print(f"\n  ─── Validation Results ───")
    model_val = YOLO(str(out_p))
    val_results = model_val.val(data=args.data, imgsz=args.imgsz, verbose=False)
    if hasattr(val_results, 'box'):
        print(f"  mAP@50:    {val_results.box.map50:.3f}")
        print(f"  mAP@50-95: {val_results.box.map:.3f}")

    print("═" * 60)


def main():
    ap = argparse.ArgumentParser(description="Train a YOLO weapon detector for HawkEye")
    ap.add_argument("--data", required=True,
                    help="Path to weapons.yaml data config (YOLO format)")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="Base YOLO model to fine-tune (default: yolo11n.pt)")
    ap.add_argument("--out", default="models/weapons.pt",
                    help="Output path for trained weights")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto",
                    help="Device: auto, cuda, cpu, 0, etc.")

    # Utility: create weapons.yaml from a folder
    ap.add_argument("--create-yaml", dest="create_yaml", default="",
                    help="Create a weapons.yaml from a folder (--create-yaml <data_dir>)")
    ap.add_argument("--classes", nargs="+", default=["knife", "pistol", "rifle"],
                    help="Class names for --create-yaml")

    args = ap.parse_args()

    if args.create_yaml:
        create_weapons_yaml(args.create_yaml, args.classes,
                           str(Path(args.create_yaml) / "weapons.yaml"))
        return

    train(args)


if __name__ == "__main__":
    main()
