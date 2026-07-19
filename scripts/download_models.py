"""Pre-download the pretrained weights HawkEye uses, so first run is offline-fast.

    python scripts/download_models.py

- YOLO11 pose (ultralytics auto-downloads on first use; this warms it)
- InsightFace buffalo_l pack (RetinaFace + ArcFace)
"""
from __future__ import annotations


def main() -> None:
    print("→ YOLO11n-pose ...")
    try:
        from ultralytics import YOLO
        YOLO("yolo11n-pose.pt")
        print("  ok")
    except Exception as e:
        print(f"  skipped ({e})")

    print("→ InsightFace buffalo_l ...")
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("  ok")
    except Exception as e:
        print(f"  skipped ({e})")

    print("Done. (Moondream2 / Claude are downloaded/used on demand.)")


if __name__ == "__main__":
    main()
