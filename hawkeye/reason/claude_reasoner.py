"""Cloud reasoning backend — Claude vision (Anthropic SDK).

Sends a handful of clip frames to Claude and gets back a structured incident
report (confirmed / description / type / severity / action). Uses
`output_config.format` (structured outputs) so the JSON is guaranteed valid.

Robust by design: any SDK/network error degrades to a heuristic Reasoning built
from the Stage-A event, so a transient failure never kills the pipeline loop.
"""
from __future__ import annotations

import json
import logging
import os
from typing import List

import numpy as np

from ..events.schema import (ActionType, AnomalyType, Reasoning, Severity,
                             StageAEvent)
from .base import (ACTION_BY_TYPE, Reasoner, default_severity, encode_jpeg_b64,
                   sample_frames)

log = logging.getLogger("hawkeye.reason.claude")

_ANOMALY_VALUES = [a.value for a in AnomalyType]
_SEVERITY_VALUES = [s.value for s in Severity]
_ACTION_VALUES = [a.value for a in ActionType]

_SCHEMA = {
    "type": "object",
    "properties": {
        "confirmed": {"type": "boolean"},
        "description": {"type": "string"},
        "anomaly_type": {"type": "string", "enum": _ANOMALY_VALUES},
        "severity": {"type": "string", "enum": _SEVERITY_VALUES},
        "recommended_action": {"type": "string", "enum": _ACTION_VALUES},
        "confidence": {"type": "number"},
    },
    "required": ["confirmed", "description", "anomaly_type", "severity",
                 "recommended_action", "confidence"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are HawkEye, a surveillance anomaly analyst. You receive a few "
    "chronological frames sampled from a short clip that an automated fast "
    "detector flagged as suspicious, plus its preliminary guess. Decide whether "
    "a genuine anomaly or emergency is occurring and describe it. Anomalies "
    "include violence/assault, robbery, theft, a visible weapon, intrusion, a "
    "person who has fallen or collapsed (medical), accidents, fire — or any "
    "other clearly dangerous or abnormal event. Be precise and avoid false "
    "alarms: if the scene looks normal, set confirmed=false and anomaly_type "
    "'normal'. Write the description as a 1-2 sentence plain-language incident "
    "report (what is happening, where in the frame, and who if identifiable)."
)


class ClaudeReasoner(Reasoner):
    name = "claude"

    def __init__(self, model: str = "claude-opus-4-8",
                 api_key_env: str = "ANTHROPIC_API_KEY", max_frames: int = 6):
        self.model = model
        self.max_frames = max_frames
        self._api_key = os.environ.get(api_key_env, "")
        self._client = None

    @property
    def available(self) -> bool:
        if not self._api_key:
            return False
        try:
            import anthropic  # noqa: F401
            return True
        except Exception:
            return False

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def reason(self, event: StageAEvent, frames: List[np.ndarray]) -> Reasoning:
        try:
            return self._reason(event, frames)
        except Exception as e:  # never let Stage B crash the loop
            log.warning("Claude reasoning failed (%s); using degraded fallback", e)
            return self._fallback(event)

    def _reason(self, event: StageAEvent, frames: List[np.ndarray]) -> Reasoning:
        picked = sample_frames(frames, self.max_frames)
        if not picked:
            return self._fallback(event)

        content: list = []
        for f in picked:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": encode_jpeg_b64(f)},
            })
        content.append({
            "type": "text",
            "text": (f"Preliminary detector guess: {event.anomaly_type.value} "
                     f"(score {event.score:.2f}) — {event.reason or 'n/a'}. "
                     f"People visible: {event.person_count}. "
                     "Analyze the frames and return the structured verdict."),
        })

        client = self._get_client()
        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM,
            messages=[{"role": "user", "content": content}],
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        data = json.loads(text)
        return self._to_reasoning(data)

    def _to_reasoning(self, data: dict) -> Reasoning:
        def _enum(cls, val, default):
            try:
                return cls(val)
            except Exception:
                return default

        atype = _enum(AnomalyType, data.get("anomaly_type"), AnomalyType.UNKNOWN)
        return Reasoning(
            confirmed=bool(data.get("confirmed", False)),
            description=str(data.get("description", "")).strip(),
            anomaly_type=atype,
            severity=_enum(Severity, data.get("severity"), Severity.LOW),
            recommended_action=_enum(ActionType, data.get("recommended_action"),
                                     ACTION_BY_TYPE.get(atype, ActionType.NONE)),
            confidence=float(np.clip(data.get("confidence", 0.0), 0.0, 1.0)),
            backend=self.name,
        )

    def _fallback(self, event: StageAEvent) -> Reasoning:
        atype = event.anomaly_type
        return Reasoning(
            confirmed=event.score >= 0.5,
            description=(event.reason or "Anomalous activity detected") +
                        " (cloud reasoning unavailable; heuristic fallback).",
            anomaly_type=atype,
            severity=default_severity(atype, event.score),
            recommended_action=ACTION_BY_TYPE.get(atype, ActionType.NONE),
            confidence=float(np.clip(event.score, 0.0, 1.0)),
            backend=self.name + "(fallback)",
        )
