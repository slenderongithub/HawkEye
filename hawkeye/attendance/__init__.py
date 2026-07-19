"""Attendance marking — the "who's present" half of Indoor Surveillance & Attendance.

Reuses the same local face database and recognizer as the identity module
(perpetrator ID). Where identity runs only on a *confirmed incident*, attendance
runs *continuously* on a slow timer: every few seconds it matches every enrolled
face in view and logs presence (one record per member per day).
"""
from .store import AttendanceStore

__all__ = ["AttendanceStore"]
