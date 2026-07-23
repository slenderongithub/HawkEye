"""Shared Kaggle-API download helpers for HawkEye dataset setup scripts.

Used by both ``scripts/setup_datasets.py`` (action datasets) and
``scripts/train_weapon.py`` (weapon datasets). Keeps the auth + per-slug retry
logic in one place so neither script duplicates it.

Design notes:
- We import ``KaggleApi`` from ``kaggle.api.kaggle_api_extended`` rather than
  ``import kaggle`` because the top-level package authenticates on import (which
  crashes hard if no token is present). Importing the class defers auth to an
  explicit ``authenticate()`` call we control.
- Kaggle dataset slugs are unstable (mirrors get renamed/removed). Every download
  takes a LIST of candidate slugs and tries them in order, logging failures and
  continuing — a single stale slug never aborts the caller's run.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"


def check_kaggle_json(path: Path | None = None) -> bool:
    """True if a kaggle.json credential file exists at the given/default path."""
    return (path or DEFAULT_KAGGLE_JSON).is_file()


def _print_token_instructions() -> None:
    print(
        "\n  Kaggle API token not found.\n"
        "  To enable automatic downloads:\n"
        "    1. https://www.kaggle.com/settings/account → 'API' → 'Create New Token'\n"
        "    2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json\n"
        "    3. chmod 600 ~/.kaggle/kaggle.json\n"
    )


def authenticate(config_path: Path | None = None):
    """Authenticate the Kaggle API. Returns the api object, or None on any failure.

    ``config_path`` optionally points at a kaggle.json in a non-default location.
    """
    if not check_kaggle_json(config_path):
        _print_token_instructions()
        return None

    if config_path is not None:
        # Kaggle reads its token from the directory named by KAGGLE_CONFIG_DIR.
        os.environ.setdefault("KAGGLE_CONFIG_DIR", str(Path(config_path).expanduser().parent))

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("  ✗ The 'kaggle' package is not installed. Run:  pip install kaggle")
        return None

    try:
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:  # noqa: BLE001 — surface any auth error cleanly
        print(f"  ✗ Kaggle authentication failed: {e}")
        return None


def download(api, slug_candidates: list[str], dest: Path) -> Path | None:
    """Download+unzip the first working slug from ``slug_candidates`` into ``dest``.

    Returns ``dest`` on the first success, or None if every candidate failed
    (or ``api`` is None). Never raises for a bad slug — logs and continues.
    """
    if api is None:
        return None
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for slug in slug_candidates:
        try:
            print(f"  → downloading Kaggle dataset '{slug}' …")
            api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
            print(f"    ✓ {slug}  →  {dest}")
            return dest
        except Exception as e:  # noqa: BLE001 — slug may be renamed/removed; try next
            print(f"    ✗ {slug}: {e}")
            continue
    print(f"  ⚠ none of the candidate slugs resolved: {slug_candidates}")
    return None
