"""HawkEye — real-time explainable anomaly detection with identity + auto-response.

Pipeline:  ingest -> Stage A (fast detect) -> Stage B (VLM reasoning, triggered)
           -> identity (face match) -> decision (dispatch) -> dashboard.
"""

__version__ = "0.1.0"
