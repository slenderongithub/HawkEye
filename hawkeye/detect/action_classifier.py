"""Inference wrapper for a trained action classifier.

Maintains a rolling window of per-frame skeleton features over the live stream
and emits anomaly candidates in the SAME contract as ``heuristics.analyze()`` ->
List[(AnomalyType, score, reason)]. Stage A uses this in place of the heuristics
whenever a trained model is configured.

Three things here exist specifically to stop one class being reported as
another:

  * **Identical preprocessing to training.** Features come from the shared
    ``skeleton_features`` module and are normalised with its shared
    ``standardize()`` (variance-floored, output-clipped). A previous revision
    hand-rolled the normalisation here and trained with a different motion value,
    which amplified one dimension ~10^5x at inference and randomised every
    prediction.

  * **Calibrated probabilities with per-class thresholds.** The checkpoint
    carries a fitted temperature and a threshold per class, chosen on validation
    data at a target precision. One global "0.8 confidence" number cannot express
    that a false police dispatch costs more than a false ambulance.

  * **Temporal hysteresis.** A single window crossing a threshold is not an
    event. A class must hold across several consecutive inference ticks before it
    is emitted, which removes the frame-level flicker that heuristic guards were
    previously papering over.

torch is imported lazily (in load()), and ``available`` checks for it without
importing, so the package still imports cleanly without torch installed.
"""
from __future__ import annotations

import importlib.util
import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..events.schema import AnomalyType
from . import geometry
from .pose import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER
from .skeleton_features import (build_sequence, frame_feature, resample,
                                standardize)

log = logging.getLogger("hawkeye.detect.action")

Candidate = Tuple[AnomalyType, float, str]

# Map common dataset folder names to our anomaly types.
_ALIASES = {
    "nonviolence": "normal", "non_violent": "normal", "non-violent": "normal",
    "no_fight": "normal", "none": "normal",
    "fight": "violence", "fighting": "violence", "fights": "violence", "violent": "violence",
    "falls": "fall", "falling": "fall", "fall_down": "fall",
}


def label_to_type(label: str) -> AnomalyType:
    key = _ALIASES.get(label.strip().lower(), label.strip().lower())
    try:
        return AnomalyType(key)
    except ValueError:
        return AnomalyType.UNKNOWN


class ActionClassifier:
    def __init__(self, model_path: str = "", device: str = "auto",
                 min_conf: float = 0.5, infer_stride: int = 2,
                 violence_motion_min: float = 0.12,
                 hysteresis_window: int = 3, hysteresis_hits: int = 2):
        self.model_path = model_path
        self.device_pref = device
        self.min_conf = min_conf          # floor; per-class thresholds take precedence
        self.infer_stride = max(1, infer_stride)
        self.violence_motion_min = violence_motion_min
        self.hysteresis_window = max(1, hysteresis_window)
        self.hysteresis_hits = max(1, min(hysteresis_hits, hysteresis_window))
        self._model = None
        self._device = "cpu"
        self._labels: List[str] = []
        self._seq_len = 64
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._thresholds: Dict[str, float] = {}
        self._buf: deque = deque()
        self._votes: deque = deque(maxlen=self.hysteresis_window)
        self._i = 0
        self._last: List[Candidate] = []

    @property
    def available(self) -> bool:
        return (bool(self.model_path) and Path(self.model_path).is_file()
                and importlib.util.find_spec("torch") is not None)

    def load(self) -> "ActionClassifier":
        import torch

        from .action_model import build_model
        from .pose import resolve_device

        self._device = resolve_device(self.device_pref)
        ckpt = torch.load(self.model_path, map_location=self._device, weights_only=False)
        self._labels = list(ckpt["labels"])
        self._seq_len = int(ckpt["seq_len"])
        self._mean = np.asarray(ckpt["mean"], dtype=np.float32)
        self._std = np.asarray(ckpt["std"], dtype=np.float32)
        self._thresholds = dict(ckpt.get("thresholds", {}))
        self._buf = deque(maxlen=self._seq_len)
        model = build_model(int(ckpt["feat_dim"]), len(self._labels),
                            int(ckpt.get("hidden", 192)), int(ckpt.get("embed", 192)),
                            bool(ckpt.get("use_margin", False)))
        model.load_state_dict(ckpt["state_dict"])
        # Temperature may live in the state dict (buffer) or as a bare scalar.
        temp = float(ckpt.get("temperature", 1.0))
        model.temperature.fill_(max(temp, 1e-2))
        model.to(self._device).eval()
        self._model = model
        log.info("Action classifier loaded: labels=%s seq_len=%d temp=%.3f "
                 "thresholds=%s device=%s", self._labels, self._seq_len, temp,
                 self._thresholds or "(default)", self._device)
        return self

    # --- public API ---
    def update(self, persons, motion: float, frame_w: int, frame_h: int) -> List[Candidate]:
        if self._model is None:
            self.load()
        self._buf.append(frame_feature(persons, motion, frame_w, frame_h))
        self._i += 1
        if len(self._buf) < max(8, self._seq_len // 4):
            return []

        if self._i % self.infer_stride != 0:
            # Between inference ticks, re-apply the physical guards to the CURRENT
            # frame rather than replaying a stale verdict: if the scene has calmed
            # or the person is upright again, the candidate must retract now, not
            # `infer_stride` frames from now.
            return self._apply_guards(self._last, persons, motion)

        raw = self._infer()
        self._last = self._apply_guards(raw, persons, motion)
        return self._last

    def reset(self) -> None:
        """Clear rolling state (new stream / new clip)."""
        self._buf.clear()
        self._votes.clear()
        self._i = 0
        self._last = []

    # --- internals ---
    def _threshold_for(self, label: str) -> float:
        return float(self._thresholds.get(label, self.min_conf))

    def _infer(self) -> List[Candidate]:
        import torch

        seq = build_sequence(list(self._buf))
        seq = resample(seq, self._seq_len)
        seq = standardize(seq, self._mean, self._std)
        x = torch.from_numpy(seq[None]).float().to(self._device)
        with torch.no_grad():
            probs = torch.softmax(self._model.calibrated_logits(x), dim=1)[0].cpu().numpy()

        # Hysteresis: a class must clear its threshold on several consecutive
        # ticks before it counts as an event.
        above = {lab for k, lab in enumerate(self._labels)
                 if probs[k] >= self._threshold_for(lab)}
        self._votes.append(above)
        if len(self._votes) < self.hysteresis_hits:
            return []

        k = int(probs.argmax())
        label = self._labels[k]
        conf = float(probs[k])
        atype = label_to_type(label)
        if atype == AnomalyType.NORMAL:
            return []
        if conf < self._threshold_for(label):
            return []
        hits = sum(1 for v in self._votes if label in v)
        if hits < self.hysteresis_hits:
            return []
        return [(atype, conf, f"action model: {label} ({conf:.0%}, {hits}/{len(self._votes)} ticks)")]

    def _apply_guards(self, candidates: List[Candidate], persons,
                      motion: float) -> List[Candidate]:
        """Physical plausibility filter, applied on every frame.

        These are cheap sanity checks on scene geometry, not a second classifier:
        they only remove verdicts the physics of the current frame contradicts.
        """
        if not candidates:
            return []
        out: List[Candidate] = []
        for atype, conf, reason in candidates:
            if atype == AnomalyType.VIOLENCE:
                # A fight needs two bodies and kinetic evidence. Two people
                # standing calmly close together have neither.
                if len(persons) < 2:
                    continue
                aggressive = any(geometry.aggressive_pose(p) > 0.5 for p in persons)
                if motion < self.violence_motion_min and not aggressive:
                    continue
            elif atype == AnomalyType.FALL:
                # A fall needs a body that is actually down: near-horizontal torso
                # or a wide (lying) bounding box.
                if persons:
                    dom = max(persons, key=lambda p: p.width * p.height)
                    sh = geometry.mid(dom.kp(L_SHOULDER), dom.kp(R_SHOULDER))
                    hip = geometry.mid(dom.kp(L_HIP), dom.kp(R_HIP))
                    horiz = geometry.torso_horizontalness(sh, hip)
                    aspect = dom.width / (dom.height + 1e-6)
                    if horiz is not None and horiz < 0.35 and aspect < 0.8:
                        continue
            out.append((atype, conf, reason))
        return out
