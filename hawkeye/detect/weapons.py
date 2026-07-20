"""Optional weapon detector (fine-tuned YOLO for guns/knives).

Disabled unless detect.weapon_model is set. When active, a weapon detection
raises the anomaly to type WEAPON with high priority.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

log = logging.getLogger("hawkeye.detect.weapons")

# Substrings matched (case-insensitively) against the model's class labels.
# Broadened beyond the obvious few so common alternate names on custom weapon
# models ("handgun", "blade", "machete", ...) aren't silently ignored. Override
# per-deployment via detect.weapon_labels in the config.
WEAPON_KEYWORDS = (
    "gun", "handgun", "pistol", "revolver", "rifle", "shotgun", "firearm",
    "knife", "blade", "machete", "dagger", "sword", "weapon",
)


@dataclass
class WeaponDetection:
    box: np.ndarray
    conf: float
    label: str


class WeaponDetector:
    def __init__(self, model_path: str = "", device: str = "auto", conf: float = 0.4,
                 keywords: Sequence[str] = WEAPON_KEYWORDS):
        self.model_path = model_path
        self.device = device
        self.conf = conf
        # Normalize once; empty/blank entries are dropped so a stray "" in
        # config can't match every label.
        self.keywords = tuple(k.lower() for k in keywords if k and k.strip())
        self._model = None
        self.enabled = bool(model_path)

    def load(self):
        if not self.enabled:
            return self
        from ultralytics import YOLO

        from .pose import resolve_device

        self.device = resolve_device(self.device)
        log.info("Loading weapon model %s on %s", self.model_path, self.device)
        self._model = YOLO(self.model_path)
        return self

    def detect(self, frame: np.ndarray) -> List[WeaponDetection]:
        if not self.enabled:
            return []
        if self._model is None:
            self.load()
        results = self._model.predict(frame, conf=self.conf, device=self.device, verbose=False)
        out: List[WeaponDetection] = []
        if not results or results[0].boxes is None:
            return out
        r = results[0]
        names = r.names
        for box, cls, cf in zip(r.boxes.xyxy.cpu().numpy(),
                                r.boxes.cls.cpu().numpy(),
                                r.boxes.conf.cpu().numpy()):
            label = str(names.get(int(cls), int(cls))).lower()
            if any(k in label for k in self.keywords):
                out.append(WeaponDetection(box=box, conf=float(cf), label=label))
        return out
