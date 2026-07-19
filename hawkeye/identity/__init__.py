"""Local identity module — RetinaFace + ArcFace (InsightFace) + embedding store.

Enrolls ~20 known people and matches a captured perpetrator face against them.
Optional: if insightface isn't installed, the module reports unavailable and the
pipeline simply skips identification (with a warning) rather than crashing.
Real Aadhaar/government-ID integration is explicit future work.
"""
from .service import IdentityService

__all__ = ["IdentityService"]
