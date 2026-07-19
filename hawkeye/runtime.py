"""Process-wide runtime state shared by the server and the pipeline thread.

Kept dependency-free so importing it never drags in torch/cv2.
"""
from __future__ import annotations

import logging
from typing import Optional

from .config import Config, load_config
from .hub import StreamHub

_config: Optional[Config] = None
_hub: Optional[StreamHub] = None


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_config(cfg: Config) -> None:
    global _config
    _config = cfg


def get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_hub() -> StreamHub:
    global _hub
    if _hub is None:
        _hub = StreamHub()
    return _hub
