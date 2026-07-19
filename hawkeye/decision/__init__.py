"""Decision & response engine — maps a confirmed incident to an action.

Severity- and confidence-gated, cooldown-protected, and fully auditable. In
Phase 1 dispatch is simulated (logged / webhook); real police/ambulance
integrations are guarded Phase 3 work.
"""
from .engine import DecisionEngine

__all__ = ["DecisionEngine"]
