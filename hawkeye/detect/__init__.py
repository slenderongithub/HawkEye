"""Stage A — the always-on, real-time fast detector.

Composes: pose detection (YOLO11), keypoint heuristics (fall/fight), a motion
scene-anomaly score, and an optional weapon detector, fused into anomaly events.
Heuristics are a *bootstrap* that works with zero training; the ST-GCN action
classifier (scripts/train_action.py) slots in behind the same interface later.
"""
from .stage_a import StageA
from .pose import PoseDetector, PersonDetection, resolve_device

__all__ = ["StageA", "PoseDetector", "PersonDetection", "resolve_device"]
