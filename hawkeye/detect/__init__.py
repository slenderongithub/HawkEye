"""Stage A — the always-on, real-time fast detector.

Composes: pose detection (YOLO11), an action source (either the zero-training
keypoint heuristics OR a trained temporal skeleton classifier), a motion
scene-anomaly score, and an optional weapon detector, fused into anomaly events.

The heuristics are a *bootstrap* that works with zero training; once a trained
model (scripts/train_action.py) is configured it replaces them behind the same
`List[(AnomalyType, score, reason)]` contract. See docs/training.md.
"""
from .stage_a import StageA
from .pose import PoseDetector, PersonDetection, resolve_device
from .action_classifier import ActionClassifier

__all__ = ["StageA", "PoseDetector", "PersonDetection", "resolve_device", "ActionClassifier"]
