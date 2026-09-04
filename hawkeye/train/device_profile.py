"""Per-platform training profile for HawkEye model training.

HawkEye's models are trained on **Apple Silicon (M2, MPS)** as the primary
target, with CUDA supported for anyone training on an NVIDIA box. This module
centralizes device selection and the hyper-parameters that depend on it so both
trainers — ``scripts/train_action.py`` (the action net) and
``scripts/train_weapon.py`` (the weapon YOLO) — pick consistent, hardware-aware
settings from one place.

Device selection reuses the canonical ``resolve_device()`` from
``hawkeye.detect.pose`` (cuda → mps → cpu).

MPS notes that shaped these numbers:
  * ``num_workers=0`` is mandatory, not conservative. macOS fork() + MPS in a
    DataLoader worker deadlocks; features are already in RAM and augmentation is
    cheap numpy, so workers would only add IPC overhead anyway.
  * The same ``num_workers=0`` applies to CUDA **on Windows**, for an unrelated
    reason: Windows spawns workers rather than forking, so the DataLoader has to
    pickle the Dataset. ``train_action.py``'s Dataset is a closure over the
    in-RAM feature cache — unpicklable, and a spawned copy per worker would
    duplicate the whole cache. Linux/CUDA forks and keeps its workers.
  * AMP is CUDA-only here. torch's MPS autocast is still uneven across ops and
    the action net is far too small for it to matter.
  * The action net is tiny (~1-2M params on a 64-frame window), so unified
    memory is not the binding constraint — batch size is chosen for gradient
    quality, not to fit in RAM.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from hawkeye.detect.pose import resolve_device


@dataclass
class TrainProfile:
    """Resolved device + the training knobs that depend on it."""
    device: str          # "mps" (primary), "cuda", or "cpu"
    use_amp: bool        # mixed-precision — only meaningful on CUDA
    batch_size: int      # default action-classifier batch (CLI --batch overrides)
    num_workers: int     # DataLoader workers
    pin_memory: bool
    hidden: int          # ActionNet recurrent hidden size
    embed: int           # ActionNet embedding size

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"


def _cpu_count() -> int:
    return os.cpu_count() or 4


def get_profile(device: str | None = None, task: str = "action") -> TrainProfile:
    """Resolve a training profile for the given (or auto-detected) device.

    ``device``: "mps" / "cuda" / "cpu" / "auto" (default). "auto" → resolve_device().
    ``task``: "action" or "weapon". Weapon batch sizing is delegated to
    ``get_yolo_kwargs`` (ultralytics), so this argument currently does not change
    the returned profile — it is accepted for a stable call site.
    """
    dev = resolve_device(device or "auto")

    if dev == "cuda":
        return TrainProfile(
            device="cuda",
            use_amp=True,
            batch_size=64,
            # 0 on Windows: spawn would have to pickle an unpicklable Dataset that
            # closes over the in-RAM feature cache (see module docstring). The
            # windows it serves are numpy slices, so in-process is not the
            # bottleneck — the GPU step is.
            num_workers=0 if os.name == "nt" else min(8, _cpu_count()),
            pin_memory=True,
            hidden=192,
            embed=192,
        )

    if dev == "mps":
        return TrainProfile(
            device="mps",
            use_amp=False,       # MPS autocast is not worth it for a model this small
            batch_size=32,
            num_workers=0,       # mandatory on macOS — fork+MPS in a worker deadlocks
            pin_memory=False,    # no-op off CUDA
            hidden=192,          # same capacity as the CUDA profile; M2 handles it fine
            embed=192,
        )

    print(
        f"\n  ⚠  No GPU backend found (resolved device: '{dev}'). Training on CPU "
        "will be slow but correct.\n"
    )
    return TrainProfile(
        device=dev,
        use_amp=False,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        hidden=128,          # trim capacity so a CPU run finishes in reasonable time
        embed=128,
    )


def get_yolo_kwargs(profile: TrainProfile) -> dict:
    """Translate a TrainProfile into ultralytics ``model.train()`` keyword args.

    On CUDA we use ultralytics **auto-batch** (``batch=-1`` → ~60% VRAM). On MPS
    ultralytics has no auto-batch, so we pick a fixed batch that fits comfortably
    in 8 GB of unified memory at 640px; raise it with ``--batch`` on a 16/24 GB
    machine (16 is usually safe there) if training is throughput-bound.
    """
    if profile.device == "cuda":
        return {
            "device": 0,
            "batch": -1,                 # auto-batch to ~60% VRAM
            # Deliberately NOT profile.num_workers, which is 0 on Windows for the
            # action trainer's sake. Ultralytics' dataset is a module-level class
            # reading images off disk: it pickles fine under spawn, and YOLO is
            # genuinely I/O-bound, so it keeps real workers on every platform.
            "workers": min(8, _cpu_count()),
            "amp": True,
        }
    if profile.device == "mps":
        return {
            "device": "mps",
            "batch": 8,                  # safe on an 8 GB M2 Air at imgsz=640
            "workers": 4,                # YOLO reads images from disk — workers do help here
            "amp": False,
        }
    return {
        "device": "cpu",
        "batch": 4,
        "workers": min(4, _cpu_count()),
        "amp": False,
    }
