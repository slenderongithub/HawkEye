"""Pick a Stage B reasoner from config, with graceful fallback to mock."""
from __future__ import annotations

import logging

from ..config import ReasonConfig
from .base import Reasoner
from .mock_reasoner import MockReasoner

log = logging.getLogger("hawkeye.reason.factory")


def build_reasoner(cfg: ReasonConfig) -> Reasoner:
    backend = (cfg.backend or "auto").lower()

    def _claude():
        from .claude_reasoner import ClaudeReasoner
        r = ClaudeReasoner(cfg.claude_model, cfg.api_key_env, cfg.max_frames)
        return r if r.available else None

    def _moondream():
        from .moondream_reasoner import MoondreamReasoner
        r = MoondreamReasoner(cfg.moondream_model)
        return r if r.available else None

    chosen: Reasoner | None = None
    if backend == "claude":
        chosen = _claude()
        if chosen is None:
            log.warning("claude backend requested but unavailable (no key / SDK) — using mock")
    elif backend == "moondream":
        chosen = _moondream()
        if chosen is None:
            log.warning("moondream backend requested but transformers missing — using mock")
    elif backend == "mock":
        chosen = MockReasoner()
    else:  # auto: prefer cloud if configured, else mock (offline-safe default)
        chosen = _claude() or MockReasoner()

    chosen = chosen or MockReasoner()
    log.info("Stage B reasoner: %s", chosen.name)
    return chosen
