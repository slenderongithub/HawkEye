"""Configuration models + loader.

Pure-python (pydantic only) so this imports without torch/cv2 present.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class IngestConfig(BaseModel):
    source: str = "0"
    width: int = 960
    height: int = 540
    target_fps: int = 20


class DetectConfig(BaseModel):
    device: str = "auto"
    pose_model: str = "yolo11n-pose.pt"
    weapon_model: str = ""
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
    return Config(**data)
