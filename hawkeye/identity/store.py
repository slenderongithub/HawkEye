"""SQLite-backed embedding store with in-memory cosine matching.

The known-person DB is tiny (~20 people), so a numpy matmul over all embeddings
is instant — no FAISS needed. Embeddings are ArcFace 512-d, L2-normalized, so
cosine similarity is a plain dot product.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..events.schema import Person, new_id


class EmbeddingStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._matrix: Optional[np.ndarray] = None      # (N, 512)
        self._owner_ids: List[str] = []                # length N, person id per row
        self._refresh()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS people (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    meta TEXT NOT NULL DEFAULT '{}',
                    enrolled_at REAL NOT NULL,
                    num_faces INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL,
                    vec BLOB NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE
                );
                """
            )

    # --- writes ---
    def add_person(self, name: str, meta: dict, embeddings: List[np.ndarray]) -> Person:
        pid = new_id()
        person = Person(id=pid, name=name, meta={k: str(v) for k, v in (meta or {}).items()},
                        num_faces=len(embeddings))
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO people (id, name, meta, enrolled_at, num_faces) VALUES (?,?,?,?,?)",
                (pid, name, json.dumps(person.meta), person.enrolled_at, person.num_faces),
            )
            for emb in embeddings:
                v = np.asarray(emb, dtype=np.float32).tobytes()
                self._conn.execute(
                    "INSERT INTO face_embeddings (person_id, vec) VALUES (?, ?)", (pid, v))
        self._refresh()
        return person

    def delete_person(self, person_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM face_embeddings WHERE person_id=?", (person_id,))
            self._conn.execute("DELETE FROM people WHERE id=?", (person_id,))
        self._refresh()

    # --- reads ---
    def list_people(self) -> List[Person]:
        rows = self._conn.execute(
            "SELECT * FROM people ORDER BY enrolled_at DESC").fetchall()
        return [self._row_to_person(r) for r in rows]

    def get_person(self, person_id: str) -> Optional[Person]:
        r = self._conn.execute("SELECT * FROM people WHERE id=?", (person_id,)).fetchone()
        return self._row_to_person(r) if r else None

    def match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Return (person_id, cosine_similarity) of the best match, or (None, 0)."""
        with self._lock:
            matrix, owners = self._matrix, self._owner_ids
        if matrix is None or matrix.shape[0] == 0:
            return None, 0.0
        q = np.asarray(embedding, dtype=np.float32)
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        sims = matrix @ q
        idx = int(np.argmax(sims))
        return owners[idx], float(sims[idx])

    @property
    def count(self) -> int:
        return len(self._owner_ids)

    def close(self) -> None:
        """Release the SQLite connection (call on pipeline shutdown)."""
        with self._lock:
            self._conn.close()

    # --- internals ---
    def _refresh(self) -> None:
        rows = self._conn.execute(
            "SELECT person_id, vec FROM face_embeddings").fetchall()
        owners, vecs = [], []
        for r in rows:
            owners.append(r["person_id"])
            vecs.append(np.frombuffer(r["vec"], dtype=np.float32))
        with self._lock:
            self._owner_ids = owners
            self._matrix = np.vstack(vecs).astype(np.float32) if vecs else None

    @staticmethod
    def _row_to_person(r: sqlite3.Row) -> Person:
        return Person(id=r["id"], name=r["name"], meta=json.loads(r["meta"]),
                      enrolled_at=r["enrolled_at"], num_faces=r["num_faces"])
