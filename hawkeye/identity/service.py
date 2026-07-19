"""IdentityService — enrollment + perpetrator identification.

Ties the InsightFace recognizer to the embedding store. Degrades gracefully:
if the recognizer isn't available, identify() returns an un-attempted FaceMatch
and the pipeline continues without identity.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from ..config import IdentityConfig
from ..events.schema import FaceMatch, Person
from .recognizer import IdentityRecognizer
from .store import EmbeddingStore

log = logging.getLogger("hawkeye.identity.service")


class IdentityService:
    def __init__(self, cfg: IdentityConfig, db_path: str, use_gpu: bool = False):
        self.cfg = cfg
        self.recognizer = IdentityRecognizer(cfg.model_name, cfg.det_size, use_gpu)
        self.store = EmbeddingStore(db_path)

    @property
    def available(self) -> bool:
        return (self.cfg.enabled and self.recognizer.available
                and not self.recognizer.broken)

    @property
    def enrolled(self) -> int:
        return len(self.store.list_people())

    # --- enrollment ---
    def enroll_person(self, name: str, meta: dict, images: List[np.ndarray]) -> Person:
        if not self.recognizer.available:
            raise RuntimeError("InsightFace not installed — cannot enroll faces.")
        embeddings = []
        for img in images:
            face = self.recognizer.best_face(img)
            if face is not None:
                embeddings.append(face.embedding)
        if not embeddings:
            raise ValueError(f"No detectable face in any provided image for '{name}'.")
        person = self.store.add_person(name, meta, embeddings)
        log.info("Enrolled %s (%d face(s))", name, len(embeddings))
        return person

    # --- identification ---
    def identify(self, frame: np.ndarray) -> Tuple[FaceMatch, Optional[np.ndarray]]:
        """Find the subject face in `frame` and match it. Returns (FaceMatch, crop).

        Never raises: a face-recognition failure must not drop an already-
        confirmed incident (identity is an enhancement, not a gate on saving
        evidence/dispatching).
        """
        if not self.available:
            return FaceMatch(attempted=False), None
        try:
            face = self.recognizer.best_face(frame)
        except Exception as e:
            log.warning("Face detection failed (%s); saving incident without identity.", e)
            return FaceMatch(attempted=False), None
        if face is None:
            return FaceMatch(attempted=True, matched=False), None

        crop = self._crop(frame, face.bbox)
        person_id, sim = self.store.match(face.embedding)
        matched = person_id is not None and sim >= self.cfg.match_threshold
        name = None
        if matched:
            p = self.store.get_person(person_id)
            name = p.name if p else None
        return (FaceMatch(attempted=True, matched=matched,
                          person_id=person_id if matched else None,
                          name=name, similarity=round(sim, 3)), crop)

    def list_people(self) -> List[Person]:
        return self.store.list_people()

    def delete_person(self, person_id: str) -> None:
        self.store.delete_person(person_id)

    @staticmethod
    def _crop(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return frame.copy()
        return frame[y1:y2, x1:x2].copy()
