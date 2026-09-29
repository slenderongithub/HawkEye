"""StreamHub — thread-safe bridge between the pipeline thread and the async server.

The pipeline (background thread) writes the latest annotated frame and pushes
incidents here; the WebSocket handlers (async) poll it. Polling avoids fragile
cross-thread asyncio scheduling.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple


class StreamHub:
    def __init__(self, max_incidents: int = 200) -> None:
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._frame_id: int = 0
        self._incidents: Deque[dict] = deque(maxlen=max_incidents)
        self._incident_version: int = 0
        self._attendance: Deque[dict] = deque(maxlen=max_incidents)
        self._attendance_version: int = 0
        self._status_version: int = 0
        # lightweight runtime status for the dashboard header
        self.status: Dict[str, object] = {
            "running": False,
            "fps": 0.0,
            "reasoner": "mock",
            "identity": False,
            "source": "",
            "present_today": 0,
        }

    # --- producer side (pipeline thread) ---
    def set_frame(self, jpeg: bytes) -> None:
        with self._lock:
            self._latest_jpeg = jpeg
            # Wraps well inside JS's Number.MAX_SAFE_INTEGER; only used as a
            # change marker on the client, so wrapping is harmless.
            self._frame_id = (self._frame_id + 1) % (2 ** 31)

    def push_incident(self, incident: dict) -> None:
        with self._lock:
            self._incidents.appendleft(incident)
            self._incident_version += 1

    def push_attendance(self, record: dict) -> None:
        with self._lock:
            self._attendance.appendleft(record)
            self._attendance_version += 1

    def set_status(self, **kwargs) -> None:
        with self._lock:
            self.status.update(kwargs)
            self._status_version += 1

    # --- consumer side (async server) ---
    def get_frame(self) -> Tuple[int, Optional[bytes]]:
        with self._lock:
            return self._frame_id, self._latest_jpeg

    def incidents_since(self, version: int) -> Tuple[int, List[dict]]:
        """Return (current_version, incidents newer than `version`)."""
        with self._lock:
            if version >= self._incident_version:
                return self._incident_version, []
            n_new = self._incident_version - version
            # deque is newest-first; take the first n_new, return oldest-first
            new = list(self._incidents)[:n_new][::-1]
            return self._incident_version, new

    def recent_incidents(self, limit: int = 50) -> List[dict]:
        with self._lock:
            return list(self._incidents)[:limit]

    def attendance_since(self, version: int) -> Tuple[int, List[dict]]:
        """Return (current_version, newly-present records since `version`)."""
        with self._lock:
            if version >= self._attendance_version:
                return self._attendance_version, []
            n_new = self._attendance_version - version
            new = list(self._attendance)[:n_new][::-1]
            return self._attendance_version, new

    def get_status(self) -> Dict[str, object]:
        with self._lock:
            return dict(self.status)

    def status_since(self, version: int) -> Tuple[int, Optional[Dict[str, object]]]:
        """Return (current_version, status) if it changed since `version`, else
        (current_version, None) — lets the WebSocket loop skip sending a frame
        that hasn't changed."""
        with self._lock:
            if version >= self._status_version:
                return self._status_version, None
            return self._status_version, dict(self.status)
