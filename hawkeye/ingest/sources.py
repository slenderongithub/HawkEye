"""Video source abstraction. cv2 is imported lazily so this module stays light."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger("hawkeye.ingest")


class VideoSource:
    """Base class. Yields BGR frames (numpy HxWx3) or None at end-of-stream."""

    def __init__(self, spec: str, width: int = 0, height: int = 0, loop: bool = False):
        self.spec = spec
        self.width = width
        self.height = height
        self.loop = loop
        self._cap = None

    def open(self) -> "VideoSource":
        import cv2

        arg: object = self.spec
        if self.spec.isdigit():
            arg = int(self.spec)
        self._cap = cv2.VideoCapture(arg)
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._cap.isOpened():
            hint = ""
            if self.spec.isdigit():
                import platform
                if platform.system() == "Darwin":
                    hint = (" On macOS, the terminal/app running this needs camera "
                            "access: System Settings -> Privacy & Security -> Camera "
                            "-> enable it for your terminal app, then restart the terminal.")
            raise RuntimeError(
                f"Could not open video source '{self.spec}'. "
                f"Check the camera index / URL / file path.{hint}"
            )
        log.info("Opened video source: %s", self.spec)
        return self

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            raise RuntimeError("Source not opened; call open() first.")
        ok, frame = self._cap.read()
        if not ok:
            if self.loop:
                import cv2

                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if ok:
                    return frame
            return None
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.release()


class DroneSource(VideoSource):
    """Phase 2 placeholder — a Tello stream plugs in here (djitellopy).

    Implementation deferred to Phase 2. It only needs to override open()/read()
    to yield BGR frames; everything downstream is unchanged.
    """

    def open(self):  # pragma: no cover - Phase 2
        raise NotImplementedError(
            "DroneSource is Phase 2. See docs/architecture.md — it will wrap "
            "djitellopy's frame_read to yield BGR frames through this same interface."
        )


def open_source(spec: str, width: int = 0, height: int = 0) -> VideoSource:
    """Factory: pick a source from the spec.

    - "0", "1", ...       -> webcam index
    - http(s)/rtsp URL    -> IP camera (e.g. phone IP-webcam app)
    - "drone:..."         -> Phase 2 Tello source (not yet implemented)
    - anything else       -> video file (looped, handy for demos on datasets)
    """
    s = spec.strip()
    if s.startswith("drone:"):
        return DroneSource(s.split(":", 1)[1], width, height).open()
    is_stream = s.isdigit() or s.startswith(("http://", "https://", "rtsp://", "rtmp://"))
    return VideoSource(s, width, height, loop=not is_stream).open()
