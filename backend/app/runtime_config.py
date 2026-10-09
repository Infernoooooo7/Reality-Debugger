"""Shared computer-vision / diagnostics / AI-usage parameters.

The JSON files in ``config/`` (repository root) are the single source of
truth for tunable parameters; the frontend imports the same files at build
time. Each parameter is ``{"value": ..., "purpose": ..., "source": ...}``;
:func:`values` strips the documentation and keeps the values.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import ROOT_DIR

DEFAULT_CONFIG_DIR = ROOT_DIR / "config"
FILES = ("vision", "detection", "tracking", "temporal", "diagnostics", "ai", "inference")


class ConfigError(RuntimeError):
    pass


def values(node: Any) -> Any:
    """Replace every ``{"value": v, ...}`` parameter object by ``v`` (recursively)."""
    if isinstance(node, dict):
        if "value" in node and ("purpose" in node or "source" in node):
            return node["value"]
        return {k: values(v) for k, v in node.items()}
    if isinstance(node, list):
        return [values(v) for v in node]
    return node


def _undocumented(node: Any, path: str = "") -> list[str]:
    """Parameters that lack a purpose or a source (documentation is mandatory)."""
    problems: list[str] = []
    if isinstance(node, dict):
        if "value" in node:
            if not node.get("purpose") or not node.get("source"):
                problems.append(path or "<root>")
            return problems
        for k, v in node.items():
            problems += _undocumented(v, f"{path}.{k}" if path else k)
    return problems


class RuntimeConfig:
    """Loaded configuration with dotted-path access, e.g. ``cfg.get("temporal.lifecycle.resolveAfterMs")``."""

    def __init__(self, raw: dict[str, Any], source_dir: Path) -> None:
        self.raw = raw
        self.values = {name: values(doc) for name, doc in raw.items()}
        self.source_dir = source_dir

    def get(self, dotted: str, default: Any = ...) -> Any:
        node: Any = self.values
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif default is not ...:
                return default
            else:
                raise ConfigError(f"Missing configuration value '{dotted}' in {self.source_dir}")
        return node

    def documented(self, dotted: str) -> dict[str, Any] | None:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node if isinstance(node, dict) and "value" in node else None


def load_config(directory: Path | None = None) -> RuntimeConfig:
    directory = directory or DEFAULT_CONFIG_DIR
    raw: dict[str, Any] = {}
    for name in FILES:
        path = directory / f"{name}.json"
        try:
            raw[name] = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"Configuration file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    problems = [f"{name}: {p}" for name, doc in raw.items() for p in _undocumented(doc)]
    if problems:
        raise ConfigError("Parameters without purpose/source: " + ", ".join(problems))
    cfg = RuntimeConfig(raw, directory)
    _validate(cfg)
    return cfg


def _validate(cfg: RuntimeConfig) -> None:
    """Fail fast on the values the backend relies on."""
    numeric = [
        "temporal.lifecycle.confirmObservations",
        "temporal.lifecycle.confirmMinSpanMs",
        "temporal.lifecycle.trackingObservations",
        "temporal.lifecycle.resolveAfterMs",
        "temporal.lifecycle.resolveObservations",
        "diagnostics.proximity.touchGap",
        "diagnostics.proximity.nearRelativeGap",
        "diagnostics.score.floor",
        "diagnostics.score.aiBlend",
        "ai.budget.cooldownMs",
        "ai.budget.maxCallsPerMinute",
        "ai.budget.dedupeHamming",
        "ai.budget.cacheTtlMs",
    ]
    for key in numeric:
        value = cfg.get(key)
        if not isinstance(value, (int, float)) or value < 0:
            raise ConfigError(f"'{key}' must be a non-negative number (got {value!r})")
    weights = cfg.get("diagnostics.score.severityWeights")
    if set(weights) != {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"}:
        raise ConfigError("diagnostics.score.severityWeights must define every severity")


@lru_cache
def get_runtime_config() -> RuntimeConfig:
    return load_config()
