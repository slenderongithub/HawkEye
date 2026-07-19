"""SQLite-backed attendance log — one row per person per day (upsert on sighting)."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional

from ..events.schema import AttendanceRecord


def date_str(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts if ts is not None else time.time()))


class AttendanceStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS attendance (
                    person_id  TEXT NOT NULL,
                    name       TEXT NOT NULL,
                    date       TEXT NOT NULL,
                    first_seen REAL NOT NULL,
                    last_seen  REAL NOT NULL,
                    sightings  INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (person_id, date)
                );
                """
            )

    def mark_seen(self, person_id: str, name: str, ts: Optional[float] = None) -> bool:
        """Record a sighting. Returns True if this is the FIRST sighting today
        (i.e. the member was just marked present), False on repeat sightings."""
        ts = ts if ts is not None else time.time()
        d = date_str(ts)
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT sightings FROM attendance WHERE person_id=? AND date=?",
                (person_id, d)).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO attendance (person_id, name, date, first_seen, last_seen, sightings)"
                    " VALUES (?,?,?,?,?,1)", (person_id, name, d, ts, ts))
                return True
            self._conn.execute(
                "UPDATE attendance SET last_seen=?, sightings=sightings+1, name=?"
                " WHERE person_id=? AND date=?", (ts, name, person_id, d))
            return False

    def on_date(self, d: Optional[str] = None) -> List[AttendanceRecord]:
        d = d or date_str()
        rows = self._conn.execute(
            "SELECT * FROM attendance WHERE date=? ORDER BY first_seen ASC", (d,)).fetchall()
        return [AttendanceRecord(**dict(r)) for r in rows]

    def present_count(self, d: Optional[str] = None) -> int:
        d = d or date_str()
        r = self._conn.execute(
            "SELECT COUNT(*) AS n FROM attendance WHERE date=?", (d,)).fetchone()
        return int(r["n"]) if r else 0
