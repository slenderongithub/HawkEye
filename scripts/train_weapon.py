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


# Kaggle weapon-dataset candidates (YOLO or Pascal-VOC). Slugs are UNSTABLE —
# verify with `kaggle datasets list -s weapon`. Tried in order; first hit wins.
KAGGLE_WEAPON_SOURCES = [
    "snehilsanyal/weapon-detection-test",
    "abhishek4273/gun-detection",
    "issaisasank/guns-object-detection",     # Pascal-VOC XML annotations
]

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _voc_to_yolo_lines(xml_path: Path, class_map: dict) -> list[str]:
    """Parse one Pascal-VOC XML into YOLO label lines. Unmapped classes skipped."""
    import xml.etree.ElementTree as ET

    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        return []
    W = float(size.findtext("width") or 0) or 1.0
    H = float(size.findtext("height") or 0) or 1.0
    lines = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in class_map:
            continue
        b = obj.find("bndbox")
        if b is None:
            continue
        xmin, ymin = float(b.findtext("xmin")), float(b.findtext("ymin"))
        xmax, ymax = float(b.findtext("xmax")), float(b.findtext("ymax"))
        cx, cy = ((xmin + xmax) / 2) / W, ((ymin + ymax) / 2) / H
        w, h = (xmax - xmin) / W, (ymax - ymin) / H
        lines.append(f"{class_map[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def convert_voc_to_yolo(xml_dir: Path, out_labels_dir: Path, class_map: dict) -> int:
    """Convert every Pascal-VOC XML in ``xml_dir`` to a YOLO .txt in the output dir.

    Returns the number of label files written (files with no mapped objects are
    skipped). ``class_map`` maps lower-cased VOC class names -> integer class ids.
    """
    out_labels_dir = Path(out_labels_dir)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for xml in Path(xml_dir).rglob("*.xml"):
        try:
            lines = _voc_to_yolo_lines(xml, class_map)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ VOC parse failed for {xml.name}: {e}")
            continue
        if lines:
            (out_labels_dir / f"{xml.stem}.txt").write_text("\n".join(lines) + "\n")
            n += 1
    return n


def _build_yolo_from_voc(raw: Path, out_dir: Path, class_names: list[str]) -> str | None:
    """Assemble a YOLO train/val dataset (+weapons.yaml) from a VOC download."""
    import random

    class_map = {n.lower(): i for i, n in enumerate(class_names)}
    img_by_stem = {p.stem: p for p in raw.rglob("*") if p.suffix.lower() in IMG_EXT}
    xmls = [x for x in raw.rglob("*.xml") if x.stem in img_by_stem]
    if not xmls:
        print("  ⚠ VOC XMLs found but none paired with an image — cannot build dataset.")
        return None

    random.seed(0)
    random.shuffle(xmls)
    n_val = max(1, int(len(xmls) * 0.2))

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    written = 0
    for i, xml in enumerate(xmls):
        lines = _voc_to_yolo_lines(xml, class_map)
        if not lines:
            continue
        split = "val" if i < n_val else "train"
        img = img_by_stem[xml.stem]
        shutil.copy2(str(img), str(out_dir / "images" / split / img.name))
        (out_dir / "labels" / split / f"{xml.stem}.txt").write_text("\n".join(lines) + "\n")
        written += 1

    if not written:
        print(f"  ⚠ No objects matched the class list {class_names} — nothing written.")
        return None

    yaml_path = create_weapons_yaml(str(out_dir), class_names, str(out_dir / "weapons.yaml"))
    print(f"  ✓ Built YOLO dataset from VOC: {written} labeled images -> {out_dir}")
    return yaml_path


def auto_download_weapons(out_dir: Path, config_path: Path | None = None) -> str | None:
    """Download a weapon dataset from Kaggle and return a usable weapons.yaml path.

    Best-effort: prefers a download that already ships a YOLO data yaml; otherwise
    converts a Pascal-VOC layout. Weapon mirrors vary wildly, so this logs what it
    finds and returns None (with guidance) rather than guessing on an unknown shape.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _kaggle_utils import authenticate, download

    out_dir = Path(out_dir)
    # authenticate() handles a missing token / uninstalled package (returns None).
    api = authenticate(config_path)
    if api is None:
        return None

    raw = out_dir / "_raw_kaggle"
    got = download(api, KAGGLE_WEAPON_SOURCES, raw)
    if not got:
        return None

    # 1) Ready-made YOLO data config?
    for y in list(got.rglob("data.yaml")) + list(got.rglob("*.yaml")):
        print(f"  ✓ found existing YOLO data config: {y}")
        return str(y)

    # 2) Pascal-VOC layout -> convert + build.
    if list(got.rglob("*.xml")):
        print("  Detected Pascal-VOC annotations — converting to YOLO format …")
        return _build_yolo_from_voc(got, out_dir, ["pistol", "knife", "rifle", "gun", "weapon"])

    print("  ⚠ Could not auto-assemble a weapon dataset from the download.")
    print(f"    Inspect {got}/ and use --create-yaml or organize the folder manually.")
    return None


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

    from hawkeye.train.device_profile import get_profile, get_yolo_kwargs
    profile = get_profile(args.device, task="weapon")
    yolo_kwargs = get_yolo_kwargs(profile)
    if args.batch is not None:            # explicit --batch overrides auto-batch
        yolo_kwargs["batch"] = args.batch

    # Load base model
    print(f"\n  Base model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device: {profile.device}  (batch={yolo_kwargs['batch']}, "
          f"amp={yolo_kwargs['amp']}, workers={yolo_kwargs['workers']})")

    model = YOLO(args.model)

    t0 = time.time()
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project="runs/weapon_detect",
        name="train",
        exist_ok=True,
        verbose=True,
        **yolo_kwargs,
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
    ap.add_argument("--data", default="",
                    help="Path to weapons.yaml data config (YOLO format). "
                         "Optional when --auto-download is used.")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="Base YOLO model to fine-tune (default: yolo11n.pt)")
    ap.add_argument("--out", default="models/weapons.pt",
                    help="Output path for trained weights")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=None,
                    help="batch size (default: auto-batch from the device profile)")
    ap.add_argument("--device", default="auto",
                    help="Device: auto (default), mps, cuda, cpu, 0, ...")
    ap.add_argument("--auto-download", dest="auto_download", action="store_true",
                    help="Download a weapon dataset from Kaggle before training "
                         "(overrides --data with the assembled weapons.yaml)")
    ap.add_argument("--dataset-dir", dest="dataset_dir", default="data/datasets/weapons",
                    help="Where --auto-download assembles the dataset")
    ap.add_argument("--kaggle-config", dest="kaggle_config", default="",
                    help="Path to kaggle.json (default: ~/.kaggle/kaggle.json)")

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

    if args.auto_download:
        cfg = Path(args.kaggle_config) if args.kaggle_config else None
        yaml_path = auto_download_weapons(Path(args.dataset_dir), cfg)
        if not yaml_path:
            sys.exit("Auto-download did not yield a trainable dataset — see messages above.")
        args.data = yaml_path

    if not args.data:
        sys.exit("--data is required (or use --auto-download).")

    train(args)


if __name__ == "__main__":
    main()
