"""Mock reasoner — zero external deps, so the pipeline explains itself offline.

Produces a templated (but sensible) incident report from the Stage-A event.
Use for development, CI, and demos without an API key or local VLM.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..events.schema import Reasoning, StageAEvent
from .base import ACTION_BY_TYPE, Reasoner, default_severity

_TEMPLATES = {
    "violence": "Two or more people are in close physical contact with rapid, aggressive motion — consistent with a fight or assault.",
    "fall": "A person is on the ground in a collapsed posture with little subsequent movement — consistent with a fall or medical emergency.",
    "weapon": "An object resembling a weapon is visible near a person.",
    "theft": "A person is interacting with property in a manner consistent with theft.",
    "intrusion": "A person is present in a monitored area.",
    "fire": "Visual cues consistent with fire or smoke are present.",
    "accident": "A sudden impact or collision-like event is visible.",
    "unknown": "Unusual activity was detected that does not match a known pattern; review recommended.",
}


class MockReasoner(Reasoner):
    name = "mock"

    def reason(self, event: StageAEvent, frames: List[np.ndarray]) -> Reasoning:
        atype = event.anomaly_type
        desc = _TEMPLATES.get(atype.value, event.reason or "Anomalous activity detected.")
        confidence = float(np.clip(event.score, 0.0, 1.0))
        return Reasoning(
            confirmed=event.score >= 0.5,
            description=desc,
            anomaly_type=atype,
            severity=default_severity(atype, event.score),
            recommended_action=ACTION_BY_TYPE.get(atype, ACTION_BY_TYPE[atype]),
            confidence=confidence,
            backend=self.name,
        )
