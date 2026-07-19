"""Pluggable video sources.

The whole point of this layer: Stage A/B/identity/decision never know where
frames come from. Phase 1 uses a webcam or phone; Phase 2 drops in a Tello
`DroneSource` with zero changes downstream.
"""
from .sources import VideoSource, open_source

__all__ = ["VideoSource", "open_source"]
