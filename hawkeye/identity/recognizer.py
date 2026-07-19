"""InsightFace wrapper (RetinaFace detector + ArcFace 512-d embeddings).

Heavy import is lazy; `available` reports whether insightface can be used.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

log = logging.getLogger("hawkeye.identity.recognizer")


@dataclass
class Face:
    bbox: np.ndarray          # [x1, y1, x2, y2]
    embedding: np.ndarray     # L2-normalized 512-d
    det_score: float

    @property
    def area(self) -> float:
        return float((self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]))


class IdentityRecognizer:
    def __init__(self, model_name: str = "buffalo_l", det_size: int = 640, use_gpu: bool = False):
        self.model_name = model_name
        self.det_size = det_size
        self.use_gpu = use_gpu
        self._app = None
        self._load_error: Optional[Exception] = None
        self._load_lock = threading.Lock()
        # Serializes calls into the onnxruntime session: the pipeline's worker
        # thread (live triggers) and the server's request threads (enroll)
        # both call detect() concurrently, and onnxruntime sessions aren't
        # guaranteed safe for concurrent .get() calls from multiple threads.
        self._infer_lock = threading.Lock()

    @property
    def available(self) -> bool:
        try:
            import insightface  # noqa: F401
            return True
        except Exception:
            return False

    @property
    def broken(self) -> bool:
        """True once a load attempt has failed — we do not keep retrying
        (each retry can re-trigger a full ~280MB model re-download)."""
        return self._load_error is not None

    def _repair_corrupt_cache(self) -> None:
        """InsightFace's downloader treats an *existing* model directory as
        'already downloaded' and silently skips the download — with no
        integrity check — so any interrupted previous download (network
        drop, killed process, corrupt zip extraction) leaves a directory
        that permanently poisons every future load with a cryptic
        AssertionError. Self-heal by wiping an incomplete cache before load.
        """
        import shutil
        from pathlib import Path

        model_dir = Path.home() / ".insightface" / "models" / self.model_name
        if model_dir.exists() and not list(model_dir.rglob("*.onnx")):
            log.warning("InsightFace model cache at %s is empty/incomplete "
                       "(an interrupted previous download) — removing it so "
                       "a fresh download runs.", model_dir)
            shutil.rmtree(model_dir, ignore_errors=True)

    def load(self) -> "IdentityRecognizer":
        with self._load_lock:
            if self._app is not None:
                return self  # already loaded by another thread
            if self._load_error is not None:
                raise RuntimeError(
                    "InsightFace previously failed to load; not retrying "
                    "automatically (restart the process after fixing the cause)."
                ) from self._load_error
            self._repair_corrupt_cache()
            from insightface.app import FaceAnalysis

            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if self.use_gpu else ["CPUExecutionProvider"])
            log.info("Loading InsightFace %s (%s)", self.model_name, providers[0])
            try:
                app_ = FaceAnalysis(name=self.model_name, providers=providers)
                app_.prepare(ctx_id=0 if self.use_gpu else -1,
                             det_size=(self.det_size, self.det_size))
            except Exception as e:
                self._load_error = e
                log.error(
                    "InsightFace failed to load (%s). Identity is now disabled "
                    "for this run. If this looks like a corrupt download, "
                    "delete ~/.insightface/models/%s and restart.",
                    e, self.model_name)
                raise
            self._app = app_
        return self

    def detect(self, frame: np.ndarray) -> List[Face]:
        if self._app is None:
            self.load()
        with self._infer_lock:
            raw = self._app.get(frame)
        out: List[Face] = []
        for f in raw:
            emb = np.asarray(f.normed_embedding, dtype=np.float32)
            out.append(Face(bbox=np.asarray(f.bbox, dtype=np.float32),
                            embedding=emb, det_score=float(f.det_score)))
        return out

    def best_face(self, frame: np.ndarray) -> Optional[Face]:
        """Largest, most-confident face in the frame — the subject to identify."""
        faces = self.detect(frame)
        if not faces:
            return None
        return max(faces, key=lambda f: f.area * (0.5 + 0.5 * f.det_score))
