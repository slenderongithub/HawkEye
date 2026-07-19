"""Dispatchers — how a decided action leaves the system.

Phase 1 ships a log dispatcher (always safe) and a webhook dispatcher (best
effort, stdlib only). Real SMS/police/ambulance integrations are Phase 3 and
must stay behind explicit config + disclaimers.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from typing import Dict, List

log = logging.getLogger("hawkeye.decision.dispatch")


class Dispatcher:
    name = "base"

    def send(self, payload: Dict) -> bool:
        raise NotImplementedError


class LogDispatcher(Dispatcher):
    name = "log"

    def send(self, payload: Dict) -> bool:
        log.warning("DISPATCH → %s | %s | %s", payload.get("action"),
                    payload.get("severity"), payload.get("summary"))
        return True


class WebhookDispatcher(Dispatcher):
    name = "webhook"

    def __init__(self, url: str):
        self.url = url

    def send(self, payload: Dict) -> bool:
        if not self.url:
            return False
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(self.url, data=data,
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception as e:
            log.warning("Webhook dispatch failed: %s", e)
            return False


def build_dispatchers(names: List[str], webhook_url: str = "") -> List[Dispatcher]:
    out: List[Dispatcher] = []
    for n in names:
        if n == "log":
            out.append(LogDispatcher())
        elif n == "webhook":
            out.append(WebhookDispatcher(webhook_url))
        else:
            log.warning("Unknown dispatcher '%s' — skipping", n)
    return out or [LogDispatcher()]
