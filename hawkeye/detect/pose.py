"""YOLO11 pose + person detection. Heavy imports (torch/ultralytics) are lazy."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

log = logging.getLogger("hawkeye.detect.pose")

# COCO-17 keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6), (0, 1), (0, 2),
    (1, 3), (2, 4),
]


def resolve_device(pref: str = "auto") -> str:
    """Resolve 'auto' to mps (Mac) / cuda (3050) / cpu."""
    if pref and pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - torch missing
        pass
    return "cpu"


@dataclass
class PersonDetection:
    box: np.ndarray            # [x1, y1, x2, y2]
    conf: float
    keypoints: np.ndarray      # (17, 3) -> x, y, kp_conf
    track_id: Optional[int] = field(default=None)

    @property
    def width(self) -> float:
        return float(self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])

    @property
    def centroid(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])

    def kp(self, idx: int, min_conf: float = 0.3) -> Optional[np.ndarray]:
        """Return (x, y) for a keypoint if confident enough, else None."""
        x, y, c = self.keypoints[idx]
        return np.array([x, y]) if c >= min_conf else None


class PoseDetector:
    def __init__(self, model_path: str = "yolo11n-pose.pt", device: str = "auto",
                 conf: float = 0.35):
        self.model_path = model_path
        self.device = resolve_device(device)
        self.conf = conf
        self._model = None

    def load(self) -> "PoseDetector":
        from ultralytics import YOLO

        log.info("Loading pose model %s on %s", self.model_path, self.device)
        self._model = YOLO(self.model_path)
        return self

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        if self._model is None:
            self.load()
        results = self._model.predict(
            frame, conf=self.conf, device=self.device, verbose=False
        )
        out: List[PersonDetection] = []
        if not results:
            return out
        r = results[0]
        if r.keypoints is None or r.boxes is None:
            return out
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        kpts = r.keypoints.data.cpu().numpy()   # (N, 17, 3)
        for i in range(len(boxes)):
            out.append(PersonDetection(box=boxes[i], conf=float(confs[i]), keypoints=kpts[i]))
        return out
