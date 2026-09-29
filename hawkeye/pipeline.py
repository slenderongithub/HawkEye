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

from .attendance import AttendanceStore
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
        self.attendance = AttendanceStore(cfg.storage.db_path)

        buflen = max(8, int(cfg.detect.clip_seconds * cfg.ingest.target_fps))
        self._ring: deque = deque(maxlen=buflen)
        self._triggers: "queue.Queue" = queue.Queue(maxsize=16)
        self._latest_raw: Optional[np.ndarray] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._stop_evt = threading.Event()
        self._capture_t: Optional[threading.Thread] = None
        self._worker_ts: List[threading.Thread] = []
        self._attendance_t: Optional[threading.Thread] = None
        self._retention_t: Optional[threading.Thread] = None
        self._fps = 0.0
        self._NUM_WORKERS = 2

    # --- lifecycle ---
    def start(self) -> None:
        log.info("Loading Stage A models...")
        self._stop_evt.clear()          # allow restart after a stop()
        self.stage_a.load()
        if self.identity.available:
            # Loaded in the background (not blocking startup): the model
            # download/prepare can take minutes on first run, and the
            # dashboard/video feed should be usable immediately regardless.
            threading.Thread(target=self._load_identity, daemon=True).start()
        # Same reasoning as identity: a lazy-loading reasoner (Moondream) can
        # take minutes on first run, and that must not stall the first trigger.
        threading.Thread(target=self.reasoner.preload, daemon=True).start()
        self._running = True
        self.hub.set_status(running=True, reasoner=self.reasoner.name,
                            identity=self.identity.available,
                            source=self.cfg.ingest.source,
                            enrolled=self.identity.enrolled,
                            attendance=self.cfg.identity.attendance_enabled,
                            present_today=self.attendance.present_count())
        self._worker_ts = [threading.Thread(target=self._worker_loop, daemon=True,
                                            name=f"worker-{i}")
                          for i in range(self._NUM_WORKERS)]
        self._capture_t = threading.Thread(target=self._capture_loop, daemon=True)
        for t in self._worker_ts:
            t.start()
        self._capture_t.start()
        if self.cfg.identity.attendance_enabled:
            self._attendance_t = threading.Thread(target=self._attendance_loop, daemon=True)
            self._attendance_t.start()
        self._retention_t = threading.Thread(target=self._retention_loop, daemon=True)
        self._retention_t.start()
        log.info("Pipeline started (reasoner=%s, identity=%s, attendance=%s)",
                 self.reasoner.name, self.identity.available,
                 self.cfg.identity.attendance_enabled)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the threads, then release resources — in that order.

        Ordering matters: `_handle_trigger` writes incidents to SQLite from the
        worker thread, so closing the stores while that thread is still draining
        loses the incident (the write raises into the worker's own except block
        and is only logged). We signal, join, and *then* close.
        """
        self._running = False
        self._stop_evt.set()
        # The sentinel is a fast-path wakeup for a worker blocked in queue.get().
        # It can be dropped if the queue is full, which is exactly why the worker
        # also polls `_stop_evt` — the stop signal must not depend on queue space.
        try:
            self._triggers.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        for t in (self._capture_t, *self._worker_ts, self._attendance_t, self._retention_t):
            if t is not None and t.is_alive():
                t.join(timeout=timeout)
                if t.is_alive():
                    log.warning("Thread %s did not stop within %.1fs; closing stores "
                                "anyway.", t.name, timeout)
        self.hub.set_status(running=False)
        self._close_stores()

    def _close_stores(self) -> None:
        """Release SQLite connections so repeated start/stop cycles (dev reloads,
        tests) don't leak file descriptors. Called only after the worker threads
        have been joined, so no in-flight write can race the close."""
        for store in (self.store, self.attendance, getattr(self.identity, "store", None)):
            if store is None:
                continue
            try:
                store.close()
            except Exception as e:      # pragma: no cover - shutdown best-effort
                log.debug("Store close failed: %s", e)

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
        try:
            source = open_source(self.cfg.ingest.source,
                                 self.cfg.ingest.width, self.cfg.ingest.height)
        except Exception as e:
            log.error("Could not open source: %s", e)
            self._running = False
            return

        # The ingest thread always reads and drains the driver's internal
        # buffer; this thread (Stage A) processes whatever is freshest. That
        # way a slow inference pass drops old frames instead of queuing them,
        # so detection is never running on a stale frame.
        ingest_t = threading.Thread(target=self._ingest_loop, args=(source,), daemon=True)
        ingest_t.start()
        try:
            self._process_loop()
        finally:
            self._running = False
            ingest_t.join(timeout=2.0)
            source.release()
            self.hub.set_status(running=False)
            log.info("Capture loop stopped.")

    def _ingest_loop(self, source) -> None:
        is_stream = source.spec.isdigit() or source.spec.startswith(
            ("http://", "https://", "rtsp://", "rtmp://"))
        backoff = 2.0
        while self._running:
            frame = source.read()
            if frame is None:
                if not is_stream:
                    log.info("End of stream / source read failed.")
                    self._running = False
                    return
                log.warning("Video source dropped; reconnecting in %.0fs...", backoff)
                source.release()
                if self._stop_evt.wait(backoff):
                    return
                try:
                    source.open()
                    log.info("Video source reconnected.")
                    backoff = 2.0
                except Exception as e:
                    log.warning("Reconnect failed: %s", e)
                    backoff = min(backoff * 2, 30.0)
                continue
            with self._frame_lock:
                self._latest_frame = frame

    def _process_loop(self) -> None:
        import cv2

        interval = 1.0 / max(1, self.cfg.ingest.target_fps)
        frame_id = 0
        last_frame = None
        while self._running:
            t0 = time.time()
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None or frame is last_frame:
                time.sleep(0.005)
                continue
            last_frame = frame
            self._ring.append(frame.copy())
            self._latest_raw = frame.copy()   # for the attendance worker

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

    def _enqueue_trigger(self, event: StageAEvent, annotated: np.ndarray) -> None:
        snapshot = list(self._ring)
        try:
            self._triggers.put_nowait((event, snapshot, annotated.copy()))
        except queue.Full:
            log.warning("Trigger dropped — all %d workers busy and the queue "
                        "(size %d) is full. An incident may have been missed.",
                        self._NUM_WORKERS, self._triggers.maxsize)

    # --- worker (heavy, triggered) ---
    def _worker_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                # Poll rather than block forever: a dropped sentinel (full queue)
                # must never be able to strand this thread.
                item = self._triggers.get(timeout=0.5)
            except queue.Empty:
                continue
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

    # --- attendance (slow cadence, off the hot path) ---
    def _attendance_loop(self) -> None:
        """Continuously (every attendance_interval_sec) match enrolled members in
        view and log presence. Independent of anomaly triggers. Reuses the
        recognizer (thread-safe via its internal inference lock)."""
        interval = max(0.5, self.cfg.identity.attendance_interval_sec)
        # Wait on the stop event instead of sleeping, so shutdown doesn't have to
        # block for a whole scan interval before this thread notices.
        while not self._stop_evt.wait(interval):
            frame = self._latest_raw
            if frame is None or not self.identity.available:
                continue
            try:
                matches = self.identity.identify_all(frame.copy())
            except Exception as e:
                log.debug("Attendance scan failed: %s", e)
                continue
            for m in matches:
                if not (m.matched and m.person_id and m.name):
                    continue
                newly = self.attendance.mark_seen(m.person_id, m.name)
                if newly:
                    log.info("ATTENDANCE: %s marked present", m.name)
                    self.hub.push_attendance({
                        "person_id": m.person_id, "name": m.name,
                        "ts": time.time(), "similarity": m.similarity,
                    })
            self.hub.set_status(present_today=self.attendance.present_count())

    # --- retention (evidence/incidents don't accumulate forever) ---
    def _retention_loop(self) -> None:
        """Prune incidents (and their evidence files) older than
        storage.retention_days. Runs once at startup, then daily — a 24/7
        deployment would otherwise fill the disk within weeks."""
        while True:
            cutoff = time.time() - self.cfg.storage.retention_days * 86400
            try:
                ids = self.store.prune_older_than(cutoff)
                for incident_id in ids:
                    self.evidence.delete(incident_id)
                if ids:
                    log.info("Retention: pruned %d incident(s) older than %d days.",
                             len(ids), self.cfg.storage.retention_days)
            except Exception as e:
                log.warning("Retention pass failed: %s", e)
            if self._stop_evt.wait(86400):
                return

    @staticmethod
    def _detect_gpu(device: str) -> bool:
        try:
            from .detect.pose import resolve_device
            return resolve_device(device) == "cuda"
        except Exception:
            return False
