"""Benchmark records: one JSON file per evaluation run, with provenance."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import resource
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[3]
RECORDS_DIR = ROOT_DIR / "docs" / "benchmarks" / "records"
SCHEMA_VERSION = 1
REQUIRED = ("task", "model", "dataset", "config", "environment", "metrics", "date")


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _mem_total_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return round(int(line.split()[1]) / 1e6, 1)
    except OSError:
        pass
    return None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR, capture_output=True, text=True, timeout=10, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT_DIR, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        return out.stdout.strip() + ("+uncommitted" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def environment() -> dict[str, Any]:
    return {
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "memory_gb": _mem_total_gb(),
        "gpu": None,  # no GPU in this environment; set when a GPU provider is used
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "packages": {p: _version(p) for p in ("numpy", "opencv-python-headless", "onnxruntime", "mediapipe", "pycocotools")},
        "ort_threads": os.environ.get("RD_ORT_THREADS") or "default (min(4, cpus))",
    }


def peak_rss_mb() -> float:
    # Linux reports ru_maxrss in kilobytes.
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def write(record: dict[str, Any], name: str | None = None) -> Path:
    missing = [k for k in REQUIRED if k not in record]
    if missing:
        raise ValueError(f"benchmark record misses {missing}")
    record = {"schema_version": SCHEMA_VERSION, **record}
    record.setdefault("git_commit", git_commit())
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    name = name or slug(f"{record['task']}-{record['model']['id']}-{record['dataset']['id']}-{record['dataset'].get('split', 'all')}-{record.get('config_name', 'default')}")
    path = RECORDS_DIR / f"{name}.json"
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False, default=_json_default) + "\n")
    return path


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:  # pragma: no cover
        pass
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
