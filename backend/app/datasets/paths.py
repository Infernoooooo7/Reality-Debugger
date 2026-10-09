"""Where datasets live. Configurable so large data can sit on another disk."""

from __future__ import annotations

import os
from pathlib import Path

# Repository root (backend/app/datasets/paths.py -> parents[3]); no dependency on the web app.
ROOT_DIR = Path(__file__).resolve().parents[3]

LAYOUT = ("raw", "interim", "processed", "annotations", "splits", "manifests")


def data_root() -> Path:
    """``RD_DATA_DIR`` or ``<repo>/data``."""
    return Path(os.environ.get("RD_DATA_DIR") or ROOT_DIR / "data").resolve()


def manifests_dir() -> Path:
    # Manifests are versioned with the code, so they always live in the repo.
    return ROOT_DIR / "data" / "manifests"


def splits_dir() -> Path:
    return ROOT_DIR / "data" / "splits"


def raw_dir(dataset_id: str) -> Path:
    return data_root() / "raw" / dataset_id


def interim_dir(dataset_id: str) -> Path:
    return data_root() / "interim" / dataset_id


def processed_dir(dataset_id: str) -> Path:
    return data_root() / "processed" / dataset_id


def ensure_layout() -> None:
    for name in LAYOUT:
        (data_root() / name).mkdir(parents=True, exist_ok=True)
