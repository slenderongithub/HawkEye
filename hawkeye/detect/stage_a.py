"""Stage A orchestrator: pose + heuristics + motion + weapons -> anomaly event.

Also renders the annotated frame shown on the dashboard.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from ..config import DetectConfig
from ..events.schema import AnomalyType, BBox, StageAEvent
from . import heuristics
from .action_classifier import ActionClassifier
from .pose import SKELETON, PersonDetection, PoseDetector
from .scene_anomaly import MotionAnomaly
from .weapons import WEAPON_KEYWORDS, WeaponDetector

log = logging.getLogger("hawkeye.detect.stage_a")

# BGR colors
GREEN, RED, YELLOW, WHITE, CYAN = (80, 220, 80), (60, 60, 235), (40, 210, 235), (240, 240, 240), (235, 200, 60)


class StageA:
    def __init__(self, cfg: DetectConfig):
        self.cfg = cfg
        self.pose = PoseDetector(cfg.pose_model, cfg.device, cfg.person_conf)
        self.motion = MotionAnomaly()
        self.weapons = WeaponDetector(cfg.weapon_model, cfg.device, cfg.weapon_conf,
                                      cfg.weapon_labels or WEAPON_KEYWORDS)
        self.action = ActionClassifier(cfg.action_model, cfg.device,
                                       cfg.action_min_conf, cfg.action_stride)
        self._last_trigger: Dict[str, float] = {}

    def load(self) -> "StageA":
        # Surface the resolved compute device (auto -> cuda/mps/cpu) so a
        # "why is this slow on a GPU box" question is answerable from the logs.
        log.info("Stage A device: %s (requested '%s')", self.pose.device, self.cfg.device)
        self.pose.load()
        self.weapons.load()
        if self.action.available:
            self.action.load()
            log.info("Stage A action source: TRAINED model (%s)", self.cfg.action_model)
        else:
            log.info("Stage A action source: heuristics (no trained model configured)")
        return self

    # --- core ---
    def process(self, frame: np.ndarray, frame_id: int, ts: float
                ) -> Tuple[StageAEvent, np.ndarray, List[PersonDetection]]:
        persons = self.pose.detect(frame)
        motion = self.motion.update(frame)
        h, w = frame.shape[:2]

        # Trained action classifier takes precedence over the bootstrap
        # heuristics once weights are configured (see docs/training.md).
        if self.action.available:
            candidates = self.action.update(persons, motion, w, h)
        else:
            candidates = heuristics.analyze(persons, motion)

        weapon_dets = self.weapons.detect(frame)
        if weapon_dets:
            top = max(w.conf for w in weapon_dets)
            candidates.append((AnomalyType.WEAPON, float(min(1.0, 0.7 + 0.3 * top)),
                               f"possible weapon detected ({weapon_dets[0].label})"))

        # Open-vocabulary safety net: unusual motion with people present but no
        # specific type -> let Stage B reason about "any other bad thing".
        if motion >= 0.6 and persons:
            candidates.append((AnomalyType.UNKNOWN, float(self.cfg.motion_weight + 0.5 * motion),
                               "unusual scene motion"))

        # Fuse: highest-scoring candidate wins; motion floor keeps a baseline.
        if candidates:
            atype, cscore, reason = max(candidates, key=lambda c: c[1])
        else:
            atype, cscore, reason = AnomalyType.NORMAL, 0.0, ""
        score = float(np.clip(max(cscore, self.cfg.motion_weight * motion), 0.0, 1.0))

        triggered = self._maybe_trigger(atype, score, ts)

        event = StageAEvent(
            frame_id=frame_id, ts=ts, anomaly_type=atype, score=score,
            motion_score=float(motion), person_count=len(persons),
            boxes=[BBox(x1=float(p.box[0]), y1=float(p.box[1]),
                        x2=float(p.box[2]), y2=float(p.box[3])) for p in persons],
            triggered=triggered, reason=reason,
        )
        annotated = self._draw(frame, persons, weapon_dets, event)
        return event, annotated, persons

    def _maybe_trigger(self, atype: AnomalyType, score: float, ts: float) -> bool:
        if atype == AnomalyType.NORMAL or score < self.cfg.trigger_threshold:
            return False
        last = self._last_trigger.get(atype.value, 0.0)
        if ts - last < self.cfg.cooldown_seconds:
            return False
        self._last_trigger[atype.value] = ts
        return True

    # --- rendering ---
    def _draw(self, frame, persons, weapon_dets, event: StageAEvent) -> np.ndarray:
        import cv2

        img = frame.copy()
        alert = event.triggered
        skel_color = RED if alert else GREEN
        for p in persons:
            x1, y1, x2, y2 = map(int, p.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), skel_color, 2)
            for a, b in SKELETON:
                pa, pb = p.kp(a), p.kp(b)
                if pa is not None and pb is not None:
                    cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), skel_color, 2)
            for idx in range(17):
                x, y, c = p.keypoints[idx]
                if c >= 0.3:
                    cv2.circle(img, (int(x), int(y)), 3, YELLOW, -1)

        for w in weapon_dets:
            x1, y1, x2, y2 = map(int, w.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), RED, 3)
            cv2.putText(img, f"WEAPON {w.conf:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)

        self._draw_hud(img, event, cv2)
        return img

    def _draw_hud(self, img, event: StageAEvent, cv2) -> None:
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 40), (25, 25, 25), -1)
        label = event.anomaly_type.value.upper()
        color = RED if event.triggered else (WHITE if event.score > 0.2 else GREEN)
        cv2.putText(img, f"{label}", (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, f"people:{event.person_count}", (w - 150, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

        # score bar with trigger-threshold marker
        bx, by, bw, bh = 200, 12, 260, 16
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (70, 70, 70), -1)
        fill = int(bw * event.score)
        cv2.rectangle(img, (bx, by), (bx + fill, by + bh), color, -1)
        tx = bx + int(bw * self.cfg.trigger_threshold)
        cv2.line(img, (tx, by - 2), (tx, by + bh + 2), CYAN, 2)
        if event.triggered:
            cv2.putText(img, "! TRIGGER", (bx + bw + 12, by + bh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2)
