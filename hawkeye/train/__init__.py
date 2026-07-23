"""Training-time helpers for HawkEye models (kept separate from inference-time
detect/ code). Currently holds the CUDA-first device/hyper-parameter profile
shared by the action-classifier and weapon-detector trainers."""
from hawkeye.train.device_profile import (TrainProfile, get_profile,
                                          get_yolo_kwargs)

__all__ = ["TrainProfile", "get_profile", "get_yolo_kwargs"]
