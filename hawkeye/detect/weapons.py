"""Optional weapon detector (fine-tuned YOLO for guns/knives).

Disabled unless detect.weapon_model is set. When active, a weapon detection
raises the anomaly to type WEAPON with high priority.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
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
        # Require the file to actually exist, not just a non-empty path — a
        # configured-but-missing checkpoint (e.g. deleted stale weights, or a
        # config committed before the model was trained) must degrade to "no
        # weapon detection" rather than crash the whole server on startup.
        self.enabled = bool(model_path) and Path(model_path).is_file()
        if model_path and not self.enabled:
            log.warning("weapon_model %r is configured but the file does not "
                       "exist — weapon detection disabled. Train one with "
                       "scripts/train_weapon.py (see docs/training.md).", model_path)

    # A compound like "handgun"/"shotgun"/"airgun" carries a real noun prefix;
    # an inflected word that merely ends the same way ("begun") does not. Three
    # characters separates the two cheaply and without a dictionary.
    MIN_COMPOUND_PREFIX = 3

    def _matches(self, label: str) -> bool:
        """Token-aware keyword match against a model class name.

        Plain substring matching fires on any label that merely contains a
        keyword. Matching whole tokens — plus genuine compounds, since
        "handgun"/"shotgun"/"machine_gun" are all guns — keeps the real naming
        variants while rejecting incidental letter matches. Compound matching is
        deliberately kept: in a security system a missed weapon costs far more
        than one extra frame for Stage B to dismiss.
        """
        tokens = [t for t in re.split(r"[^a-z0-9]+", label.lower()) if t]
        for t in tokens:
            for k in self.keywords:
                if t == k:
                    return True
                if t.endswith(k) and len(t) - len(k) >= self.MIN_COMPOUND_PREFIX:
                    return True
        return False

    def load(self):
        if not self.enabled:
            return self
        from ultralytics import YOLO

        from .pose import resolve_device

        self.device = resolve_device(self.device)
        log.info("Loading weapon model %s on %s", self.model_path, self.device)
        try:
            self._model = YOLO(self.model_path)
        except Exception as e:
            # A corrupt/incompatible checkpoint is a bad file, not a reason to
            # take down surveillance + attendance entirely. Same contract as
            # ActionClassifier and the identity model: degrade, don't crash.
            log.warning("Failed to load weapon model %s (%s) — weapon detection "
                       "disabled for this run.", self.model_path, e)
            self.enabled = False
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
            if self._matches(label):
                out.append(WeaponDetection(box=box, conf=float(cf), label=label))
        return out
