"""Motion-based scene anomaly score.

A cheap, always-on "something is happening" signal from frame-to-frame motion,
with an EMA baseline so we flag *spikes* rather than constant background motion.
Upgrade path: replace with a weakly-supervised MIL scorer trained on UCF-Crime
(see scripts/train_action.py notes) behind this same .update() interface.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class MotionAnomaly:
    def __init__(self, ema_alpha: float = 0.05, gain: float = 6.0):
        self.ema_alpha = ema_alpha
        self.gain = gain
        self._prev_gray: Optional[np.ndarray] = None
        self._baseline: float = 0.0
        self._baseline_var: float = 1e-4

    def update(self, frame: np.ndarray) -> float:
        """Return a motion-anomaly score in [0, 1] for this frame."""
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0

        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray
        motion = float(diff.mean()) / 255.0        # raw motion energy

        # Update running baseline (EMA mean + variance).
        delta = motion - self._baseline
        self._baseline += self.ema_alpha * delta
        self._baseline_var = (1 - self.ema_alpha) * (self._baseline_var + self.ema_alpha * delta * delta)
        std = float(np.sqrt(self._baseline_var)) + 1e-6

        # Anomaly = how far above baseline this frame is (z-score), squashed to [0,1].
        z = max(0.0, (motion - self._baseline) / std)
        score = float(np.tanh(z / self.gain))
        return score
