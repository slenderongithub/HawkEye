"""Stage B — triggered reasoning/explanation layer.

Turns a suspicious clip into a natural-language incident report + severity +
recommended action. Backends (cloud Claude / local Moondream / mock) all
implement the same `Reasoner` interface and are chosen by config.
"""
from .base import Reasoner
from .factory import build_reasoner

__all__ = ["Reasoner", "build_reasoner"]
