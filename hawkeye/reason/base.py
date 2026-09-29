"""Reasoner interface + shared helpers (frame sampling, type/severity defaults)."""
from __future__ import annotations

import base64
from typing import List

import numpy as np

from ..events.schema import (ActionType, AnomalyType, Reasoning, Severity,
                             StageAEvent)

# Canonical mapping used for the VLM's *recommended* action and as a fallback.
# The decision engine remains authoritative (config-driven).
ACTION_BY_TYPE = {
    AnomalyType.VIOLENCE: ActionType.POLICE,
    AnomalyType.ROBBERY: ActionType.POLICE,
    AnomalyType.THEFT: ActionType.POLICE,
    AnomalyType.WEAPON: ActionType.POLICE,
    AnomalyType.INTRUSION: ActionType.POLICE,
    AnomalyType.FALL: ActionType.AMBULANCE,
    AnomalyType.MEDICAL: ActionType.AMBULANCE,
    AnomalyType.ACCIDENT: ActionType.AMBULANCE,
    AnomalyType.FIRE: ActionType.FIRE,
    AnomalyType.UNKNOWN: ActionType.NONE,
    AnomalyType.NORMAL: ActionType.NONE,
}

_HIGH = {AnomalyType.VIOLENCE, AnomalyType.ROBBERY, AnomalyType.WEAPON,
         AnomalyType.FALL, AnomalyType.MEDICAL, AnomalyType.ACCIDENT, AnomalyType.FIRE}
_MED = {AnomalyType.THEFT, AnomalyType.INTRUSION, AnomalyType.UNKNOWN}


def default_severity(atype: AnomalyType, score: float) -> Severity:
    if atype in _HIGH:
        return Severity.CRITICAL if score >= 0.8 else Severity.HIGH
    if atype in _MED:
        return Severity.MEDIUM if score >= 0.6 else Severity.LOW
    return Severity.LOW


def sample_frames(frames: List[np.ndarray], k: int) -> List[np.ndarray]:
    """Evenly sample up to k frames from a clip."""
    if not frames:
        return []
    if len(frames) <= k:
        return list(frames)
    idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
    return [frames[i] for i in idx]


def encode_jpeg_b64(frame: np.ndarray, max_side: int = 768, quality: int = 80) -> str:
    """Downscale + JPEG + base64 encode a BGR frame for a vision API."""
    import cv2

    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class Reasoner:
    """Interface for Stage B. name identifies the backend on the dashboard."""

    name = "base"

    def reason(self, event: StageAEvent, frames: List[np.ndarray]) -> Reasoning:
        raise NotImplementedError

    def preload(self) -> None:
        """Optional: warm up any heavy model in the background. Backends that
        load lazily on first use (e.g. Moondream) override this so the first
        real trigger doesn't stall the worker thread."""
        pass
