"""Enroll a person into the local identity DB from a folder of face images.

Usage:
    python scripts/enroll_person.py --name "Asha Rao" --images data/faces/asha \
        --meta '{"role":"staff","id_no":"EMP-014"}'

Requires insightface installed. Real Aadhaar/government-ID integration is future
work — this maintains a small, consented local dataset only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hawkeye.config import load_config           # noqa: E402
from hawkeye.identity import IdentityService      # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--images", required=True, help="Folder of face images (or a single image)")
    ap.add_argument("--meta", default="{}", help='JSON, e.g. {"role":"staff"}')
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    import cv2

    cfg = load_config(args.config)
    svc = IdentityService(cfg.identity, cfg.storage.db_path, use_gpu=False)
    if not svc.recognizer.available:
        sys.exit("insightface is not installed. `pip install insightface onnxruntime`")

    p = Path(args.images)
    paths = [p] if p.is_file() else [f for f in p.iterdir() if f.suffix.lower() in IMG_EXT]
    if not paths:
        sys.exit(f"No images found in {args.images}")

    images = [cv2.imread(str(f)) for f in paths]
    images = [im for im in images if im is not None]
    person = svc.enroll_person(args.name, json.loads(args.meta), images)
    print(f"Enrolled {person.name} (id={person.id}) with {person.num_faces} face(s). "
          f"Total enrolled: {svc.enrolled}")


if __name__ == "__main__":
    main()
