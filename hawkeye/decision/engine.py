"""DecisionEngine — (type, severity, confidence, identity) -> gated action."""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional

from ..config import DecisionConfig
from ..events.schema import (ActionRecord, ActionType, FaceMatch, Reasoning,
                             Severity, severity_rank)
from .dispatch import build_dispatchers

log = logging.getLogger("hawkeye.decision.engine")


class DecisionEngine:
    def __init__(self, cfg: DecisionConfig):
        self.cfg = cfg
        self.dispatchers = build_dispatchers(cfg.dispatchers, cfg.webhook_url)
        self._min_rank = severity_rank(cfg.dispatch_min_severity)
        self._last_dispatch: Dict[str, float] = {}   # action -> ts

    def _action_for(self, reasoning: Reasoning) -> ActionType:
        # Config rules are authoritative; fall back to the VLM's recommendation.
        mapped = self.cfg.rules.get(reasoning.anomaly_type.value)
        if mapped:
            try:
                return ActionType(mapped)
            except ValueError:
                pass
        return reasoning.recommended_action

    def decide(self, reasoning: Reasoning, face_match: Optional[FaceMatch] = None,
               now: Optional[float] = None) -> ActionRecord:
        now = now if now is not None else time.time()
        action = self._action_for(reasoning)
        sev = reasoning.severity

        reasons = []
        gate_ok = True
        if not reasoning.confirmed:
            gate_ok, r = False, "unconfirmed"
            reasons.append(r)
        if reasoning.confidence < self.cfg.confidence_min:
            gate_ok = False
            reasons.append(f"low confidence {reasoning.confidence:.2f}")
        if severity_rank(sev) < self._min_rank:
            gate_ok = False
            reasons.append(f"severity {sev.value} below floor")
        if action == ActionType.NONE:
            gate_ok = False
            reasons.append("no action mapped")

        # Cooldown per action type.
        if gate_ok:
            last = self._last_dispatch.get(action.value, 0.0)
            if now - last < self.cfg.cooldown_seconds:
                gate_ok = False
                reasons.append("cooldown")

        who = ""
        if face_match and face_match.matched:
            who = f" (subject: {face_match.name})"
        elif face_match and face_match.attempted:
            who = " (subject unidentified)"

        summary = f"{reasoning.anomaly_type.value}{who}: {reasoning.description}"

        dispatched = False
        if gate_ok:
            self._last_dispatch[action.value] = now
            payload = {"action": action.value, "severity": sev.value,
                       "summary": summary, "confidence": reasoning.confidence}
            dispatched = any(d.send(payload) for d in self.dispatchers)
            reason = f"dispatched {action.value}"
        else:
            reason = "logged only (" + ", ".join(reasons) + ")"
            log.info("No dispatch for %s — %s", reasoning.anomaly_type.value, reason)

        return ActionRecord(action=action, dispatched=dispatched, severity=sev,
                            reason=reason,
                            dispatchers=[d.name for d in self.dispatchers] if dispatched else [],
                            ts=now)
