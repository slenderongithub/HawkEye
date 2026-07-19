"""Evidence saver — writes clip frames, the perpetrator face crop, and a
dashboard thumbnail for each incident under evidence_dir/<incident_id>/.

Paths returned are relative to evidence_dir (the server serves them from there).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class EvidenceManager:
    def __init__(self, evidence_dir: str, clip_frames: int = 6):
        self.root = Path(evidence_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.clip_frames = clip_frames

    def save(self, incident_id: str, frames: List[np.ndarray],
             annotated: Optional[np.ndarray] = None,
             face_crop: Optional[np.ndarray] = None) -> Dict[str, object]:
        import cv2

        d = self.root / incident_id
        d.mkdir(parents=True, exist_ok=True)
        out: Dict[str, object] = {}

        # Evenly-sampled clip frames.
        clip_paths = []
        picked = self._sample(frames, self.clip_frames)
        for i, f in enumerate(picked):
            rel = f"{incident_id}/clip_{i:02d}.jpg"
            cv2.imwrite(str(self.root / rel), f)
            clip_paths.append(rel)
        out["clip_frames"] = clip_paths

        # Thumbnail (annotated preferred, else first clip frame).
        thumb_src = annotated if annotated is not None else (picked[0] if picked else None)
        if thumb_src is not None:
            h, w = thumb_src.shape[:2]
            scale = 360 / max(1, w)
            thumb = cv2.resize(thumb_src, (int(w * scale), int(h * scale)))
            rel = f"{incident_id}/thumb.jpg"
            cv2.imwrite(str(self.root / rel), thumb)
            out["thumb"] = rel

        # Perpetrator face crop.
        if face_crop is not None and face_crop.size > 0:
            rel = f"{incident_id}/face.jpg"
            cv2.imwrite(str(self.root / rel), face_crop)
            out["face"] = rel

        return out

    @staticmethod
    def _sample(frames: List[np.ndarray], k: int) -> List[np.ndarray]:
        if not frames:
            return []
        if len(frames) <= k:
            return list(frames)
        idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
        return [frames[i] for i in idx]
