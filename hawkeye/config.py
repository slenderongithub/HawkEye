"""Configuration models + loader.

Pure-python (pydantic only) so this imports without torch/cv2 present.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class IngestConfig(BaseModel):
    source: str = "0"
    width: int = 960
    height: int = 540
    target_fps: int = 20


class DetectConfig(BaseModel):
    device: str = "auto"
    pose_model: str = "yolo11n-pose.pt"
    weapon_model: str = ""            # path to a trained weapon-detection YOLO
    weapon_conf: float = 0.4
    # Class-label substrings that count as a weapon. Empty => use the built-in
    # default list (see detect/weapons.py WEAPON_KEYWORDS).
    weapon_labels: List[str] = Field(default_factory=list)
    action_model: str = ""           # path to a trained action classifier (scripts/train_action.py)
    # Fallback confidence floor. A trained checkpoint carries per-class
    # thresholds fitted at a target precision, which take precedence over this.
    action_min_conf: float = 0.5
    action_stride: int = 2           # run the action net every N frames (perf)
    # Temporal hysteresis: a class must clear its threshold on `hits` of the last
    # `window` inference ticks before it is emitted. One window crossing a
    # threshold is noise; a sustained verdict is an event.
    action_hysteresis_window: int = 3
    action_hysteresis_hits: int = 2
    # Suppress a VIOLENCE verdict when 2+ people are close/overlapping but
    # neither scene motion nor an aggressive pose backs it up (e.g. two people
    # standing still near the camera) -- see hawkeye/detect/action_classifier.py.
    violence_motion_min: float = 0.12
    person_conf: float = 0.35
    trigger_threshold: float = 0.55
    motion_weight: float = 0.35
    clip_seconds: float = 3.0
    cooldown_seconds: float = 8.0


class ReasonConfig(BaseModel):
    backend: str = "auto"
    claude_model: str = "claude-sonnet-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_frames: int = 6
    moondream_model: str = "vikhyatk/moondream2"


class IdentityConfig(BaseModel):
    enabled: bool = True
    match_threshold: float = 0.42
    model_name: str = "buffalo_l"
    det_size: int = 640
    attendance_enabled: bool = True        # continuous face-recognition attendance
    attendance_interval_sec: float = 3.0   # how often to scan the frame for members


class DecisionConfig(BaseModel):
    confidence_min: float = 0.5
    cooldown_seconds: float = 20.0
    rules: Dict[str, str] = Field(default_factory=dict)
    dispatch_min_severity: str = "medium"
    dispatchers: List[str] = Field(default_factory=lambda: ["log"])
    webhook_url: str = ""


class StorageConfig(BaseModel):
    db_path: str = "data/hawkeye.db"
    evidence_dir: str = "data/evidence"
    retention_days: int = 30       # incidents + evidence older than this get pruned


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class Config(BaseModel):
    name: str = "HawkEye"
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    detect: DetectConfig = Field(default_factory=DetectConfig)
    reason: ReasonConfig = Field(default_factory=ReasonConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


def load_config(path: str | Path | None = None) -> Config:
    """Load a Config from YAML, falling back to packaged defaults."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, "r") as f:
        data = yaml.safe_load(f) or {}
    cfg = Config(**data)
    # Relative storage paths are anchored to the project root, not the CWD the
    # process happened to be launched from.
    if not Path(cfg.storage.db_path).is_absolute():
        cfg.storage.db_path = str(PROJECT_ROOT / cfg.storage.db_path)
    if not Path(cfg.storage.evidence_dir).is_absolute():
        cfg.storage.evidence_dir = str(PROJECT_ROOT / cfg.storage.evidence_dir)
    return cfg
