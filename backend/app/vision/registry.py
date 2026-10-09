"""Model registry (models/registry/*.json) and a lazy, memory-bounded provider.

The registry records what each model can actually do (tasks, vocabulary type
and categories), where its weights come from (URL + SHA-256), its licence,
runtimes, evaluation results and known limitations - including models that
were researched but cannot be used here, with the reason.

``ModelProvider`` loads a model only when first used, verifies its checksum
before loading (untrusted or corrupted weights are never loaded), keeps at most
``memory_budget_mb`` of models resident (least recently used first out) and
serialises inference per model, so a small server cannot be pushed into an
out-of-memory restart by concurrent requests.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[3]
REGISTRY_DIR = ROOT_DIR / "models" / "registry"

#: Resident-memory estimate per byte of fp32 weights for ONNX Runtime sessions (weights + arena + activations).
MEMORY_FACTOR = 2.5


class ModelUnavailable(RuntimeError):
    """The model cannot be used: not downloaded, failed its checksum, unsupported, or failed to load."""

    def __init__(self, model_id: str, reason: str) -> None:
        super().__init__(f"{model_id}: {reason}")
        self.model_id = model_id
        self.reason = reason


@dataclass(slots=True)
class Availability:
    model_id: str
    available: bool
    status: str  # bundled | fetchable | deferred | unavailable | rejected
    reason: str | None
    path: str | None


def _weight_dirs() -> list[Path]:
    dirs = []
    if env := os.environ.get("RD_MODEL_DIR"):
        dirs.append(Path(env))
    dirs.append(ROOT_DIR / "models" / "weights")
    dirs.append(ROOT_DIR / "frontend" / "public" / "models")
    if dist := os.environ.get("FRONTEND_DIST"):
        dirs.append(Path(dist) / "models")
    return dirs


class ModelRegistry:
    def __init__(self, directory: Path = REGISTRY_DIR) -> None:
        self.directory = directory
        self.entries: dict[str, dict[str, Any]] = {
            p.stem: json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))
        }
        self._verified: dict[tuple[str, int, float], bool] = {}

    def get(self, model_id: str) -> dict[str, Any]:
        try:
            return self.entries[model_id]
        except KeyError:
            raise ModelUnavailable(model_id, "not in the model registry") from None

    def ids(self, task: str | None = None) -> list[str]:
        return [i for i, e in self.entries.items() if task is None or task in e.get("tasks", [])]

    def labels(self, model_id: str) -> list[str | None]:
        """The vocabulary from the model's own metadata (browser manifest), if recorded."""
        manifest = ROOT_DIR / "frontend" / "public" / "models" / f"{model_id}.manifest.json"
        if manifest.exists():
            return json.loads(manifest.read_text())["labels"]
        entry = self.get(model_id)
        family = entry.get("family")
        if family == "yolox" or entry.get("vocabulary", {}).get("dataset") == "COCO 2017":
            # Every COCO-80 model in the registry uses the same contiguous class order (verified by evaluation).
            return json.loads((ROOT_DIR / "frontend" / "public" / "models" / "yolox_s.manifest.json").read_text())["labels"]
        raise ModelUnavailable(model_id, "no label list recorded")

    def weights_path(self, model_id: str) -> Path | None:
        entry = self.get(model_id)
        weights = entry.get("weights")
        if not weights:
            return None
        for directory in _weight_dirs():
            candidate = directory / weights["file"]
            if candidate.exists():
                return candidate
        return None

    def verify(self, model_id: str, path: Path) -> bool:
        expected = self.get(model_id).get("weights", {}).get("sha256")
        if not expected:
            return False
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime)
        if key not in self._verified:
            h = hashlib.sha256()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            self._verified[key] = h.hexdigest() == expected
        return self._verified[key]

    def availability(self, model_id: str, *, verify: bool = False) -> Availability:
        entry = self.get(model_id)
        status = entry.get("status", "unknown")
        if status not in ("bundled", "fetchable"):
            return Availability(model_id, False, status, entry.get("unavailable_reason", status), None)
        path = self.weights_path(model_id)
        if path is None:
            return Availability(model_id, False, status, f"weights not downloaded (python tools/fetch_models.py --fetch {model_id})", None)
        if verify and not self.verify(model_id, path):
            return Availability(model_id, False, status, "weights failed the SHA-256 check", str(path))
        return Availability(model_id, True, status, None, str(path))


Loader = Callable[[dict[str, Any], Path], Any]


class ModelProvider:
    """Lazy, checksum-verified, memory-bounded access to loaded models."""

    def __init__(self, registry: ModelRegistry, loaders: dict[str, Loader], *, memory_budget_mb: float = 600) -> None:
        self.registry = registry
        self.loaders = loaders  # family -> loader(entry, weights_path)
        self.memory_budget_mb = memory_budget_mb
        self._loaded: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._run_locks: dict[str, threading.Lock] = {}
        self.errors: dict[str, str] = {}

    def estimate_mb(self, model_id: str) -> float:
        entry = self.registry.get(model_id)
        size = entry.get("weights", {}).get("bytes")
        if not size and (path := self.registry.weights_path(model_id)):
            size = path.stat().st_size
        return MEMORY_FACTOR * (size or 0) / 1e6

    def get(self, model_id: str) -> Any:
        with self._lock:
            if model_id in self._loaded:
                self._loaded.move_to_end(model_id)
                return self._loaded[model_id][0]
            entry = self.registry.get(model_id)
            avail = self.registry.availability(model_id, verify=True)
            if not avail.available:
                raise ModelUnavailable(model_id, avail.reason or "unavailable")
            loader = self.loaders.get(entry.get("family", ""))
            if loader is None:
                raise ModelUnavailable(model_id, f"no loader for model family '{entry.get('family')}'")
            need = self.estimate_mb(model_id)
            if need > self.memory_budget_mb:
                raise ModelUnavailable(model_id, f"needs ~{need:.0f} MB, above this server's model memory budget ({self.memory_budget_mb:.0f} MB)")
            while self._loaded and sum(mb for _, mb in self._loaded.values()) + need > self.memory_budget_mb:
                self._loaded.popitem(last=False)  # evict least recently used
            try:
                model = loader(entry, Path(avail.path))  # type: ignore[arg-type]
            except Exception as error:  # noqa: BLE001 - surface a real error instead of a fallback
                self.errors[model_id] = f"{type(error).__name__}: {error}"
                raise ModelUnavailable(model_id, f"failed to load: {self.errors[model_id]}") from error
            self._loaded[model_id] = (model, need)
            self._run_locks.setdefault(model_id, threading.Lock())
            return model

    def run_lock(self, model_id: str) -> threading.Lock:
        with self._lock:
            return self._run_locks.setdefault(model_id, threading.Lock())

    def loaded(self) -> list[str]:
        return list(self._loaded)
