"""Local offline reasoning backend — Moondream2 (2B VLM) via transformers.

The only VLM that fits comfortably in ~2 GB alongside Stage A on an 8 GB Mac
or 4 GB-VRAM 3050. Weaker than the cloud model, so it captions the clip and we
keyword-map the caption to a verdict. Enabled only if transformers is installed.
"""
from __future__ import annotations

import logging
import threading
from typing import List

import numpy as np

from ..events.schema import (ActionType, AnomalyType, Reasoning, StageAEvent)
from .base import (ACTION_BY_TYPE, Reasoner, default_severity, sample_frames)

log = logging.getLogger("hawkeye.reason.moondream")

_QUESTION = (
    "Describe any dangerous or abnormal activity in this image: a fight or "
    "assault, a weapon, a person lying on the ground or who has collapsed, an "
    "accident, or fire. If nothing unusual is happening, answer 'normal'."
)

_KEYWORDS = [
    (("fight", "punch", "assault", "attack", "hitting", "violence"), AnomalyType.VIOLENCE),
    (("gun", "knife", "weapon", "pistol", "rifle"), AnomalyType.WEAPON),
    (("fallen", "lying", "collapsed", "on the ground", "unconscious"), AnomalyType.FALL),
    (("fire", "smoke", "flame", "burning"), AnomalyType.FIRE),
    (("crash", "accident", "collision"), AnomalyType.ACCIDENT),
    (("steal", "theft", "robbery", "grabbing"), AnomalyType.THEFT),
]


class MoondreamReasoner(Reasoner):
    name = "moondream"

    def __init__(self, model_name: str = "vikhyatk/moondream2", max_frames: int = 3):
        self.model_name = model_name
        self.max_frames = max_frames
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()

    @property
    def available(self) -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except Exception:
            return False

    def preload(self) -> None:
        try:
            self._load()
        except Exception as e:
            log.warning("Moondream preload failed (%s); will retry on first trigger", e)

    def _load(self):
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:      # another thread won the race
                return
            from transformers import AutoModelForCausalLM, AutoTokenizer

            log.info("Loading Moondream2 (%s) — first run downloads weights", self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = model

    def _caption(self, frame: np.ndarray) -> str:
        from PIL import Image

        rgb = frame[:, :, ::-1]  # BGR -> RGB
        img = Image.fromarray(rgb)
        # Support both the newer .query() and older .answer_question() APIs.
        try:
            return str(self._model.query(img, _QUESTION)["answer"])
        except Exception:
            enc = self._model.encode_image(img)
            return str(self._model.answer_question(enc, _QUESTION, self._tokenizer))

    def reason(self, event: StageAEvent, frames: List[np.ndarray]) -> Reasoning:
        try:
            self._load()
            picked = sample_frames(frames, self.max_frames)
            captions = [self._caption(f) for f in picked] or [""]
            text = " ".join(captions).lower()
        except Exception as e:
            log.warning("Moondream reasoning failed (%s); heuristic fallback", e)
            text, captions = "", [event.reason]

        atype = AnomalyType.NORMAL
        for keys, t in _KEYWORDS:
            if any(k in text for k in keys):
                atype = t
                break
        if atype == AnomalyType.NORMAL and event.anomaly_type != AnomalyType.NORMAL:
            atype = event.anomaly_type  # trust Stage A if the caption is uninformative

        confirmed = atype != AnomalyType.NORMAL and "normal" not in text[:15]
        confidence = float(np.clip(0.5 * event.score + (0.4 if confirmed else 0.0), 0.0, 1.0))
        return Reasoning(
            confirmed=confirmed,
            description=captions[0][:300] if captions and captions[0] else
                        (event.reason or "Anomalous activity detected"),
            anomaly_type=atype,
            severity=default_severity(atype, max(event.score, confidence)),
            recommended_action=ACTION_BY_TYPE.get(atype, ActionType.NONE),
            confidence=confidence,
            backend=self.name,
        )
