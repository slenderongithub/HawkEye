"""Pipeline orchestrator — the heart of HawkEye.

Capture thread: read frames -> Stage A (real-time) -> annotate -> stream to the
dashboard. On a trigger it snapshots the clip buffer and hands it to a worker
thread so the video never stalls.

Worker thread: Stage B reasoning -> (if confirmed) identity match -> decision ->
evidence + persistence -> push the incident to the dashboard.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from typing import List, Optional

import numpy as np

from .config import Config
from .decision import DecisionEngine
from .detect import StageA
from .events.evidence import EvidenceManager
from .events.schema import (ActionType, AnomalyType, FaceMatch, Incident,
                            IncidentStatus, Reasoning, StageAEvent)
from .events.store import IncidentStore
from .hub import StreamHub
from .identity import IdentityService
from .ingest import open_source
from .reason import build_reasoner

log = logging.getLogger("hawkeye.pipeline")

_SENTINEL = object()


class Pipeline:
    def __init__(self, cfg: Config, hub: StreamHub):
        self.cfg = cfg
        self.hub = hub
        use_gpu = self._detect_gpu(cfg.detect.device)

        self.stage_a = StageA(cfg.detect)
        self.reasoner = build_reasoner(cfg.reason)
        self.identity = IdentityService(cfg.identity, cfg.storage.db_path, use_gpu)
        self.decision = DecisionEngine(cfg.decision)
        self.store = IncidentStore(cfg.storage.db_path)
        self.evidence = EvidenceManager(cfg.storage.evidence_dir)

        buflen = max(8, int(cfg.detect.clip_seconds * cfg.ingest.target_fps))
        self._ring: deque = deque(maxlen=buflen)
        self._triggers: "queue.Queue" = queue.Queue(maxsize=4)
        self._running = False
        self._capture_t: Optional[threading.Thread] = None
        self._worker_t: Optional[threading.Thread] = None
        self._fps = 0.0

    # --- lifecycle ---
    def start(self) -> None:
        log.info("Loading Stage A models...")
        self.stage_a.load()
        if self.identity.available:
            # Loaded in the background (not blocking startup): the model
            # download/prepare can take minutes on first run, and the
            # dashboard/video feed should be usable immediately regardless.
            threading.Thread(target=self._load_identity, daemon=True).start()
        self._running = True
        self.hub.set_status(running=True, reasoner=self.reasoner.name,
                            identity=self.identity.available,
                            source=self.cfg.ingest.source,
                            enrolled=self.identity.enrolled)
        self._worker_t = threading.Thread(target=self._worker_loop, daemon=True)
        self._capture_t = threading.Thread(target=self._capture_loop, daemon=True)
        self._worker_t.start()
        self._capture_t.start()
        log.info("Pipeline started (reasoner=%s, identity=%s)",
                 self.reasoner.name, self.identity.available)

    def stop(self) -> None:
        self._running = False
        try:
            self._triggers.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        self.hub.set_status(running=False)

    def _load_identity(self) -> None:
        log.info("Loading identity model (InsightFace) in the background...")
        try:
            self.identity.recognizer.load()
            log.info("Identity model ready.")
        except Exception as e:
            log.warning("Identity model failed to load (%s); identity "
                       "disabled for this run.", e)
        finally:
            self.hub.set_status(identity=self.identity.available)

    # --- capture (real-time) ---
    def _capture_loop(self) -> None:
        import cv2

        try:
            source = open_source(self.cfg.ingest.source,
                                 self.cfg.ingest.width, self.cfg.ingest.height)
        except Exception as e:
            log.error("Could not open source: %s", e)
            self._running = False
            return

        interval = 1.0 / max(1, self.cfg.ingest.target_fps)
        frame_id = 0
        try:
            while self._running:
                t0 = time.time()
                frame = source.read()
                if frame is None:
                    log.info("End of stream / source read failed.")
                    break
                self._ring.append(frame.copy())

                event, annotated, _ = self.stage_a.process(frame, frame_id, t0)
                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    self.hub.set_frame(buf.tobytes())

                if event.triggered:
                    self._enqueue_trigger(event, annotated)

                frame_id += 1
                dt = time.time() - t0
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)
                self.hub.set_status(fps=round(self._fps, 1))
                if dt < interval:
                    time.sleep(interval - dt)
        finally:
            source.release()
            self.hub.set_status(running=False)
            log.info("Capture loop stopped.")

    def _enqueue_trigger(self, event: StageAEvent, annotated: np.ndarray) -> None:
        snapshot = list(self._ring)
        try:
            self._triggers.put_nowait((event, snapshot, annotated.copy()))
        except queue.Full:
            log.debug("Trigger dropped (worker busy).")

    # --- worker (heavy, triggered) ---
    def _worker_loop(self) -> None:
        while self._running:
            item = self._triggers.get()
            if item is _SENTINEL:
                break
            try:
                self._handle_trigger(*item)
            except Exception as e:
                log.exception("Trigger handling failed: %s", e)

    def _handle_trigger(self, event: StageAEvent, frames: List[np.ndarray],
                        annotated: np.ndarray) -> None:
        reasoning = self.reasoner.reason(event, frames)
        if not reasoning.confirmed or reasoning.anomaly_type == AnomalyType.NORMAL:
            log.info("Stage B dismissed a %s trigger (false alarm avoided).",
                     event.anomaly_type.value)
            return

        # Identify the perpetrator on the most recent clip frame.
        face_match: Optional[FaceMatch] = None
        face_crop = None
        if self.identity.available and frames:
            face_match, face_crop = self.identity.identify(frames[-1])

        action = self.decision.decide(reasoning, face_match)

        incident = Incident(
            anomaly_type=reasoning.anomaly_type, score=event.score,
            status=IncidentStatus.CONFIRMED, stage_a=event, reasoning=reasoning,
            face_match=face_match, action=action,
        )
        ev = self.evidence.save(incident.id, frames, annotated, face_crop)
        incident.thumb = ev.get("thumb")
        incident.clip_path = incident.id
        if face_match and ev.get("face"):
            face_match.face_image = ev["face"]

        self.store.save(incident)
        payload = json.loads(incident.model_dump_json())
        payload["headline"] = incident.headline()
        self.hub.push_incident(payload)
        log.warning("INCIDENT %s: %s [%s] -> %s", incident.id, incident.headline(),
                    reasoning.severity.value, action.action.value)

    @staticmethod
    def _detect_gpu(device: str) -> bool:
        try:
            from .detect.pose import resolve_device
            return resolve_device(device) == "cuda"
        except Exception:
            return False
