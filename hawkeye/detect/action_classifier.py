"""Inference wrapper for a trained action classifier.

Maintains a rolling window of per-frame skeleton features over the live stream
and, once warm, emits anomaly candidates in the SAME contract as
`heuristics.analyze()` -> List[(AnomalyType, score, reason)]. Stage A uses this
in place of the heuristics whenever a trained model is configured.

torch is imported lazily (in load()), and `available` checks for it without
importing, so the package still imports cleanly without torch installed.
"""
from __future__ import annotations

import importlib.util
import logging
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..events.schema import AnomalyType
from .skeleton_features import build_sequence, frame_feature, resample

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
                 violence_motion_min: float = 0.12):
        self.model_path = model_path
        self.device_pref = device
        self.min_conf = min_conf
        self.infer_stride = max(1, infer_stride)
        self.violence_motion_min = violence_motion_min
        self._model = None
        self._device = "cpu"
        self._labels: List[str] = []
        self._seq_len = 32
        self._mean = None
        self._std = None
        self._buf: deque = deque()
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
        self._std = np.asarray(ckpt["std"], dtype=np.float32) + 1e-6
        self._buf = deque(maxlen=self._seq_len)
        model = build_model(int(ckpt["feat_dim"]), len(self._labels),
                            int(ckpt.get("hidden", 128)), int(ckpt.get("embed", 128)),
                            bool(ckpt.get("use_margin", True)))
        model.load_state_dict(ckpt["state_dict"])
        model.to(self._device).eval()
        self._model = model
        log.info("Action classifier loaded: labels=%s seq_len=%d margin=%s device=%s",
                 self._labels, self._seq_len, ckpt.get("use_margin"), self._device)
        return self

    def update(self, persons, motion: float, frame_w: int, frame_h: int) -> List[Candidate]:
        if self._model is None:
            self.load()
        self._buf.append(frame_feature(persons, motion, frame_w, frame_h))
        self._i += 1
        # Need a little context before the first prediction.
        if len(self._buf) < max(4, self._seq_len // 4):
            return []
        # Run inference every `infer_stride` frames; reuse the last verdict between.
        if self._i % self.infer_stride != 0:
            return self._last
        raw = self._infer()
        # --- False-positive guards ---
        # These guards use scene context (person count, posture) to suppress
        # physically impossible classifications.
        filtered: List[Candidate] = []
        for atype, conf, reason in raw:
            if atype == AnomalyType("violence"):
                # A fight requires at least 2 people visible in the frame.
                if len(persons) < 2:
                    continue
                # Two close/overlapping people alone isn't a fight -- require
                # kinetic evidence: sustained scene motion or an aggressive
                # pose (wrists at/above shoulder height) on at least one
                # person. Calm people standing close together have neither.
                from . import geometry
                aggressive = any(geometry.aggressive_pose(p) > 0.5 for p in persons)
                if motion < self.violence_motion_min and not aggressive:
                    continue
            elif atype == AnomalyType("fall"):
                # A fall requires the person to be in a near-horizontal posture
                # or have a wide aspect ratio (lying down). If the dominant
                # person is clearly upright, suppress the false alarm.
                if persons:
                    from . import geometry
                    from .pose import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER
                    dom = max(persons, key=lambda p: p.width * p.height)
                    sh = geometry.mid(dom.kp(L_SHOULDER), dom.kp(R_SHOULDER))
                    hip = geometry.mid(dom.kp(L_HIP), dom.kp(R_HIP))
                    horiz = geometry.torso_horizontalness(sh, hip)
                    aspect = dom.width / (dom.height + 1e-6)
                    # If torso is clearly upright AND bbox is tall (not lying),
                    # this is not a fall.
                    if horiz is not None and horiz < 0.35 and aspect < 0.8:
                        continue
            filtered.append((atype, conf, reason))
        self._last = filtered
        return self._last

    def _infer(self) -> List[Candidate]:
        import torch

        seq = build_sequence(list(self._buf))         # (t, F)
        seq = resample(seq, self._seq_len)            # (seq_len, F)
        seq = (seq - self._mean) / self._std
        x = torch.from_numpy(seq[None]).float().to(self._device)
        with torch.no_grad():
            probs = torch.softmax(self._model(x), dim=1)[0].cpu().numpy()
        k = int(probs.argmax())
        conf = float(probs[k])
        atype = label_to_type(self._labels[k])
        if atype == AnomalyType.NORMAL or conf < self.min_conf:
            return []
        return [(atype, conf, f"action model: {self._labels[k]} ({conf:.0%})")]
