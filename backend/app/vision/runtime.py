"""ONNX Runtime sessions sized for the machine they run on.

Container platforms often cap CPU below what ``os.cpu_count()`` reports
(e.g. a 0.15-CPU instance on a many-core host); running 4 inference threads
under such a quota only adds throttling. Threads therefore follow the cgroup
CPU quota when there is one (RD_ORT_THREADS overrides).

Measured on the build machine (YOLOX-S fp32, 640x640, CPU): the ONNX Runtime
memory arena keeps peak activation memory resident (217 MB vs 171 MB after
four runs without it), so servers with little memory disable it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def cpu_quota() -> float | None:
    """CPUs granted by the cgroup (v2 cpu.max or v1 CFS quota), or None when unlimited/unknown."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        quota_us = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period_us = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota_us > 0 and period_us > 0:
            return quota_us / period_us
    except (OSError, ValueError):
        pass
    return None


def memory_limit_mb() -> float | None:
    """Memory granted by the cgroup, or None when unlimited/unknown."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
            if raw != "max" and int(raw) < 1 << 60:
                return int(raw) / 2**20
        except (OSError, ValueError):
            continue
    return None


def default_threads() -> int:
    if env := os.environ.get("RD_ORT_THREADS"):
        return max(1, int(env))
    cpus = os.cpu_count() or 1
    if (quota := cpu_quota()) is not None:
        cpus = min(cpus, max(1, int(quota)))
    return max(1, min(4, cpus))


def ort_session(path: Path, *, threads: int | None = None, low_memory: bool | None = None) -> Any:
    import onnxruntime as ort

    if low_memory is None:
        limit = memory_limit_mb()
        low_memory = limit is not None and limit < 2048
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads or default_threads()
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if low_memory:
        options.enable_cpu_mem_arena = False
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
