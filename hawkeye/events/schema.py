"""Typed events that flow through the pipeline.

Pure-python (pydantic + stdlib) — safe to import anywhere, no heavy deps.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


def _now() -> float:
    return time.time()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class AnomalyType(str, Enum):
    NORMAL = "normal"
    VIOLENCE = "violence"
    ROBBERY = "robbery"
    THEFT = "theft"
    WEAPON = "weapon"
    INTRUSION = "intrusion"
    FALL = "fall"
    MEDICAL = "medical"
    ACCIDENT = "accident"
    FIRE = "fire"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(str, Enum):
    NONE = "none"
    POLICE = "police"
    AMBULANCE = "ambulance"
    FIRE = "fire"


class IncidentStatus(str, Enum):
    DETECTED = "detected"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"


_SEVERITY_ORDER = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


def severity_rank(sev: Severity | str) -> int:
    """Ordinal rank for comparing severities (LOW=0 ... CRITICAL=3)."""
    if isinstance(sev, str):
        sev = Severity(sev)
    return _SEVERITY_ORDER[sev]


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class StageAEvent(BaseModel):
    """Output of the always-on fast detector."""
    frame_id: int
    ts: float = Field(default_factory=_now)
    anomaly_type: AnomalyType = AnomalyType.NORMAL
    score: float = 0.0              # fused anomaly score [0,1]
    motion_score: float = 0.0
    person_count: int = 0
    boxes: List[BBox] = Field(default_factory=list)
    triggered: bool = False         # crossed the trigger threshold
    reason: str = ""                # short heuristic explanation


class Reasoning(BaseModel):
    """Output of the triggered Stage B VLM layer."""
    confirmed: bool = False
    description: str = ""
    anomaly_type: AnomalyType = AnomalyType.UNKNOWN
    severity: Severity = Severity.LOW
    recommended_action: ActionType = ActionType.NONE
    confidence: float = 0.0
    backend: str = "mock"


class FaceMatch(BaseModel):
    attempted: bool = False
    matched: bool = False
    person_id: Optional[str] = None
    name: Optional[str] = None
    similarity: float = 0.0
    face_image: Optional[str] = None   # path to saved perpetrator face crop


class ActionRecord(BaseModel):
    action: ActionType = ActionType.NONE
    dispatched: bool = False
    severity: Severity = Severity.LOW
    reason: str = ""
    dispatchers: List[str] = Field(default_factory=list)
    ts: float = Field(default_factory=_now)


class Incident(BaseModel):
    """The full record of one confirmed/observed anomaly."""
    id: str = Field(default_factory=new_id)
    ts: float = Field(default_factory=_now)
    anomaly_type: AnomalyType = AnomalyType.UNKNOWN
    score: float = 0.0
    status: IncidentStatus = IncidentStatus.DETECTED
    stage_a: Optional[StageAEvent] = None
    reasoning: Optional[Reasoning] = None
    face_match: Optional[FaceMatch] = None
    action: Optional[ActionRecord] = None
    thumb: Optional[str] = None       # dashboard thumbnail path
    clip_path: Optional[str] = None   # evidence clip (frames dir or mp4)

    def headline(self) -> str:
        who = ""
        if self.face_match and self.face_match.matched:
            who = f" by {self.face_match.name}"
        elif self.face_match and self.face_match.attempted:
            who = " by an unidentified person"
        return f"{self.anomaly_type.value.title()}{who}"


class Person(BaseModel):
    id: str = Field(default_factory=new_id)
    name: str
    meta: Dict[str, str] = Field(default_factory=dict)   # e.g. {"role": "staff", "id_no": "..."}
    enrolled_at: float = Field(default_factory=_now)
    num_faces: int = 0
