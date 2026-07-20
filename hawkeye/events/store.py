"""SQLite persistence for incidents (the audit trail)."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional

from .schema import Incident


class IncidentStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    anomaly_type TEXT,
                    dispatched INTEGER DEFAULT 0,
                    matched INTEGER DEFAULT 0,
                    data TEXT NOT NULL
                );
                """
            )

    def save(self, incident: Incident) -> None:
        dispatched = int(bool(incident.action and incident.action.dispatched))
        matched = int(bool(incident.face_match and incident.face_match.matched))
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO incidents (id, ts, anomaly_type, dispatched, matched, data)"
                " VALUES (?,?,?,?,?,?)",
                (incident.id, incident.ts, incident.anomaly_type.value,
                 dispatched, matched, incident.model_dump_json()),
            )

    def recent(self, limit: int = 50) -> List[Incident]:
        rows = self._conn.execute(
            "SELECT data FROM incidents ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [Incident.model_validate_json(r["data"]) for r in rows]

    def get(self, incident_id: str) -> Optional[Incident]:
        r = self._conn.execute(
            "SELECT data FROM incidents WHERE id=?", (incident_id,)).fetchone()
        return Incident.model_validate_json(r["data"]) if r else None

    def close(self) -> None:
        """Release the SQLite connection (call on pipeline shutdown)."""
        with self._lock:
            self._conn.close()
