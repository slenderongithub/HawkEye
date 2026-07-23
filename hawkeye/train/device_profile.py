"""Per-platform training profile for HawkEye model training.

HawkEye's models are trained on an **NVIDIA RTX 3050 laptop (CUDA)**. This module
centralizes device selection and the hyper-parameters that depend on it so both
trainers — ``scripts/train_action.py`` (the action GRU) and
``scripts/train_weapon.py`` (the weapon YOLO) — pick consistent, hardware-aware
settings from one place.

Device selection reuses the canonical ``resolve_device()`` from
``hawkeye.detect.pose`` (cuda → mps → cpu). Training is **CUDA-first**: if CUDA is
unavailable the profile falls back to CPU with a loud warning, purely so the code
can be import- and smoke-tested on a non-CUDA machine (e.g. a Mac). The fallback
is NOT tuned for real training on MPS/CPU — it exists to prove the code runs, not
to produce usable weights.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from hawkeye.detect.pose import resolve_device


@dataclass
class TrainProfile:
    """Resolved device + the training knobs that depend on it."""
    device: str          # "cuda" (target) or "mps"/"cpu" (smoke-test fallback)
    use_amp: bool        # mixed-precision — only meaningful on CUDA
    batch_size: int      # default action-classifier batch (CLI --batch overrides)
    num_workers: int     # DataLoader workers
    pin_memory: bool
    hidden: int          # ActionNet GRU hidden size
    embed: int           # ActionNet embedding size

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"


def _cpu_count() -> int:
    return os.cpu_count() or 4


def get_profile(device: str | None = None, task: str = "action") -> TrainProfile:
    """Resolve a training profile for the given (or auto-detected) device.

    ``device``: "cuda" / "cpu" / "auto" (default). "auto" → resolve_device().
    ``task``: "action" or "weapon". Weapon batch sizing is delegated to
    ``get_yolo_kwargs`` (ultralytics auto-batch), so this argument currently does
    not change the returned profile — it is accepted for a stable call site and
    future per-task tuning.
    """
    dev = resolve_device(device or "auto")

    if dev == "cuda":
        return TrainProfile(
            device="cuda",
            use_amp=True,
            batch_size=64,                     # ActionNet is tiny; VRAM isn't the limit here
            num_workers=min(8, _cpu_count()),
            pin_memory=True,
            hidden=192,                        # a little more capacity than the CPU default
            embed=192,
        )

    # Non-CUDA path: smoke-test only, never a real training run.
    print(
        f"\n  ⚠  CUDA not found (resolved device: '{dev}'). Using a CPU/smoke-test "
        "configuration — NOT a real training run.\n"
        "     Run training on the RTX 3050 laptop to get usable weights.\n"
    )
    return TrainProfile(
        device=dev,          # "mps" or "cpu"
        use_amp=False,
        batch_size=16,
        num_workers=0,       # keep single-process; avoids macOS fork+MPS DataLoader hangs
        pin_memory=False,
        hidden=128,          # match the historical checkpoint defaults on fallback
        embed=128,
    )


def get_yolo_kwargs(profile: TrainProfile) -> dict:
    """Translate a TrainProfile into ultralytics ``model.train()`` keyword args.

    On CUDA we use ultralytics **auto-batch** (``batch=-1`` → ~60% VRAM), the
    safest choice for the RTX 3050's ~4 GB. If a run OOMs, pass an explicit
    ``--batch`` on the CLI (e.g. 8) — the caller layers that on top of these kwargs.
    """
    if profile.device == "cuda":
        return {
            "device": 0,
            "batch": -1,                 # auto-batch to ~60% VRAM
            "workers": profile.num_workers,
            "amp": True,
        }
    # Smoke-test fallback (mps/cpu). ultralytics accepts "mps"/"cpu" device strings.
    return {
        "device": profile.device,
        "batch": 4,
        "workers": 0,
        "amp": False,
    }
