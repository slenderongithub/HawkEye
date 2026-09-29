"""Self-check for Pipeline shutdown ordering.

Run: python tests/test_pipeline_shutdown.py

The bug this guards: `stop()` closed the SQLite stores without joining the
worker thread, so an incident being written by `_handle_trigger` at that moment
hit a closed connection. The write raised inside the worker's own try/except and
was merely logged — the incident silently vanished. A second failure mode sat
next to it: the stop signal was a sentinel pushed into a bounded queue, so a full
queue dropped it and the worker never exited.

These tests drive `Pipeline.stop()` against stub stores/threads (constructing a
real Pipeline would need models, a camera and a database), which is enough to
pin the two behaviours that actually broke: join-before-close, and a stop signal
that does not depend on queue space.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hawkeye.pipeline import _SENTINEL, Pipeline  # noqa: E402


class StubStore:
    """Records the order of writes and closes, and refuses writes after close."""

    def __init__(self, name, log):
        self.name = name
        self.log = log
        self.closed = False

    def save(self, _incident):
        if self.closed:
            raise RuntimeError(f"{self.name}: write after close — incident lost")
        self.log.append(f"write:{self.name}")

    def close(self):
        self.closed = True
        self.log.append(f"close:{self.name}")


class StubHub:
    def set_status(self, **kw):
        pass


def make_pipeline(log):
    """A Pipeline with only the attributes stop() touches."""
    p = object.__new__(Pipeline)
    p.hub = StubHub()
    p._running = True
    p._stop_evt = threading.Event()
    p._triggers = queue.Queue(maxsize=4)
    p._capture_t = None
    p._worker_ts = []
    p._attendance_t = None
    p._retention_t = None
    p.store = StubStore("incidents", log)
    p.attendance = StubStore("attendance", log)
    p.identity = type("I", (), {"store": StubStore("identity", log)})()
    return p


def test_join_happens_before_close():
    """A worker mid-write must finish before the stores are closed."""
    log = []
    p = make_pipeline(log)
    started = threading.Event()

    def worker():
        started.set()
        # Simulate an in-flight incident write that outlives the stop() call.
        time.sleep(0.3)
        p.store.save(object())

    p._worker_ts = [threading.Thread(target=worker, name="worker")]
    p._worker_ts[0].start()
    started.wait(2.0)

    p.stop(timeout=5.0)

    assert "write:incidents" in log, "the in-flight incident write never happened"
    assert log.index("write:incidents") < log.index("close:incidents"), (
        f"stores closed before the worker finished writing — order was {log}")
    assert not p._worker_ts[0].is_alive(), "worker thread was not joined"
    print("  ✓ in-flight incident write completes before stores close")


def test_stop_signal_survives_a_full_queue():
    """The worker must exit even when the sentinel cannot be enqueued."""
    log = []
    p = make_pipeline(log)
    # Fill the trigger queue so put_nowait(_SENTINEL) raises queue.Full.
    for _ in range(4):
        p._triggers.put_nowait(("filler",))
    assert p._triggers.full(), "test setup: queue should be full"

    exited = threading.Event()

    def worker():
        # The real _worker_loop: poll the queue but also watch the stop event.
        while not p._stop_evt.is_set():
            try:
                item = p._triggers.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                break
        exited.set()

    p._worker_ts = [threading.Thread(target=worker, name="worker")]
    p._worker_ts[0].start()

    t0 = time.time()
    p.stop(timeout=5.0)
    elapsed = time.time() - t0

    assert exited.is_set(), "worker never exited — the dropped sentinel stranded it"
    assert elapsed < 4.0, f"stop() took {elapsed:.1f}s; it should not have to wait out a timeout"
    print(f"  ✓ worker exits on a full queue via the stop event ({elapsed * 1000:.0f} ms)")


def test_stop_is_idempotent_and_restartable():
    """Calling stop() twice must not raise, and the event must be re-armable."""
    log = []
    p = make_pipeline(log)
    p.stop(timeout=1.0)
    p.stop(timeout=1.0)          # second call: stores already closed
    assert p._stop_evt.is_set()
    p._stop_evt.clear()          # what start() does
    assert not p._stop_evt.is_set(), "stop event cannot be re-armed for a restart"
    print("  ✓ stop() is idempotent and the stop event is re-armable")


def test_slow_thread_does_not_hang_shutdown():
    """A thread that ignores the stop signal must not block stop() forever."""
    log = []
    p = make_pipeline(log)
    p._capture_t = threading.Thread(target=lambda: time.sleep(3.0), name="stuck",
                                    daemon=True)
    p._capture_t.start()

    t0 = time.time()
    p.stop(timeout=0.4)
    elapsed = time.time() - t0

    assert elapsed < 2.0, f"stop() blocked for {elapsed:.1f}s on a stuck thread"
    assert "close:incidents" in log, "stores were never closed after the join timeout"
    print(f"  ✓ a stuck thread times out instead of hanging shutdown ({elapsed:.2f}s)")


def main() -> int:
    checks = [
        test_join_happens_before_close,
        test_stop_signal_survives_a_full_queue,
        test_stop_is_idempotent_and_restartable,
        test_slow_thread_does_not_hang_shutdown,
    ]
    print(f"Running {len(checks)} shutdown self-checks...\n")
    failed = 0
    for fn in checks:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {fn.__name__}: {e}")
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f"  ✗ {fn.__name__}: unexpected {type(e).__name__}: {e}")
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
