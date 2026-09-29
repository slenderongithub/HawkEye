"""Self-check for DecisionEngine gating logic.

Run: python tests/test_decision_engine.py

This is the last gate before a dispatch (police/ambulance/fire) fires or an
incident is merely logged. A gating bug here either spams false dispatches or
silently swallows a real one — both are worse than a detection-layer bug
because there's no second chance downstream.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hawkeye.config import DecisionConfig  # noqa: E402
from hawkeye.decision.engine import DecisionEngine  # noqa: E402
from hawkeye.events.schema import (ActionType, AnomalyType, Reasoning,  # noqa: E402
                                   Severity)


def make_engine(**overrides) -> DecisionEngine:
    cfg = DecisionConfig(**overrides)
    return DecisionEngine(cfg)


def confirmed_violence(**overrides) -> Reasoning:
    kw = dict(confirmed=True, anomaly_type=AnomalyType.VIOLENCE,
              severity=Severity.HIGH, recommended_action=ActionType.POLICE,
              confidence=0.9, description="a fight")
    kw.update(overrides)
    return Reasoning(**kw)


def test_confirmed_high_severity_dispatches():
    engine = make_engine()
    action = engine.decide(confirmed_violence())
    assert action.dispatched, f"expected dispatch, got: {action.reason}"
    assert action.action == ActionType.POLICE
    print("  ✓ a confirmed, high-severity, high-confidence event dispatches")


def test_unconfirmed_never_dispatches():
    engine = make_engine()
    action = engine.decide(confirmed_violence(confirmed=False))
    assert not action.dispatched
    assert "unconfirmed" in action.reason
    print("  ✓ an unconfirmed Stage B verdict never dispatches")


def test_low_confidence_is_gated_out():
    engine = make_engine(confidence_min=0.8)
    action = engine.decide(confirmed_violence(confidence=0.3))
    assert not action.dispatched
    assert "confidence" in action.reason
    print("  ✓ confidence below the floor is gated out")


def test_severity_below_floor_is_gated_out():
    engine = make_engine(dispatch_min_severity="high")
    action = engine.decide(confirmed_violence(severity=Severity.LOW))
    assert not action.dispatched
    assert "severity" in action.reason
    print("  ✓ severity below the dispatch floor is gated out")


def test_no_action_mapped_is_gated_out():
    engine = make_engine()
    action = engine.decide(confirmed_violence(recommended_action=ActionType.NONE))
    assert not action.dispatched
    assert "no action mapped" in action.reason
    print("  ✓ ActionType.NONE never dispatches")


def test_cooldown_suppresses_repeat_dispatch():
    engine = make_engine(cooldown_seconds=20.0)
    t0 = 1_000_000.0
    first = engine.decide(confirmed_violence(), now=t0)
    assert first.dispatched
    second = engine.decide(confirmed_violence(), now=t0 + 5.0)
    assert not second.dispatched, "a repeat within the cooldown window must not dispatch"
    assert "cooldown" in second.reason
    third = engine.decide(confirmed_violence(), now=t0 + 25.0)
    assert third.dispatched, "cooldown must lift once it elapses"
    print("  ✓ cooldown blocks a repeat dispatch, then lifts after it elapses")


def test_cooldown_is_per_action_type():
    engine = make_engine(cooldown_seconds=20.0)
    t0 = 1_000_000.0
    engine.decide(confirmed_violence(), now=t0)   # POLICE action, starts its cooldown
    fall = engine.decide(confirmed_violence(
        anomaly_type=AnomalyType.FALL, recommended_action=ActionType.AMBULANCE), now=t0 + 1.0)
    assert fall.dispatched, "a different action type must not share another's cooldown"
    print("  ✓ cooldown is scoped per action type, not global")


def test_config_rule_overrides_vlm_recommendation():
    engine = make_engine(rules={"violence": "ambulance"})
    action = engine.decide(confirmed_violence(recommended_action=ActionType.POLICE))
    assert action.action == ActionType.AMBULANCE, "config rule must take precedence"
    print("  ✓ a configured rule overrides the VLM's recommended action")


def main() -> int:
    checks = [
        test_confirmed_high_severity_dispatches,
        test_unconfirmed_never_dispatches,
        test_low_confidence_is_gated_out,
        test_severity_below_floor_is_gated_out,
        test_no_action_mapped_is_gated_out,
        test_cooldown_suppresses_repeat_dispatch,
        test_cooldown_is_per_action_type,
        test_config_rule_overrides_vlm_recommendation,
    ]
    print(f"Running {len(checks)} decision-engine self-checks...\n")
    failed = 0
    for fn in checks:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {fn.__name__}: {e}")
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
