#!/usr/bin/env python3
"""Download and verify models listed in the model registry (models/registry/*.json).

Every label list is read from the model's own metadata (the labels file
packed inside a TFLite model, or the class list in the official repository an
ONNX model was exported from), never typed in by hand.

Browser models (registry entries with a "browser" block) live in
frontend/public/models/ together with a runtime manifest the frontend loads.
Backend/evaluation models live in models/weights/ (RD_MODEL_DIR overrides).

Usage:
    python tools/fetch_models.py                      download missing browser models, rebuild their manifests
    python tools/fetch_models.py --fetch              download missing default models, keep manifests (Docker build)
    python tools/fetch_models.py --fetch yolox_m ...  download specific registry models
    python tools/fetch_models.py --check [ids...]     verify files and manifests (no network)
    python tools/fetch_models.py --list               registry summary

A file whose SHA-256 differs from the registry is deleted, never used. For a
registry entry without a checksum yet, the first download records it
("trust on first use") - review the registry diff before committing.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "models" / "registry"
BROWSER_DIR = ROOT / "frontend" / "public" / "models"


def weights_dir() -> Path:
    return Path(os.environ.get("RD_MODEL_DIR") or ROOT / "models" / "weights")


def load_registry() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(REGISTRY.glob("*.json"))}


def save_entry(entry: dict) -> None:
    (REGISTRY / f"{entry['id']}.json").write_text(json.dumps(entry, indent=2, ensure_ascii=False) + "\n")


def model_path(entry: dict) -> Path:
    base = BROWSER_DIR if "browser" in entry else weights_dir()
    return base / entry["weights"]["file"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, attempts: int = 3) -> None:
    for attempt in range(1, attempts + 1):
        print(f"  downloading {url}" + (f" (attempt {attempt})" if attempt > 1 else ""))
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as out:  # noqa: S310 - registry URLs
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
            return
        except OSError as error:  # URLError, timeouts, resets
            dest.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            print(f"  {error}; retrying")
            time.sleep(2 * attempt)


def tflite_labels(path: Path) -> list[str | None]:
    """TFLite metadata packs associated files (labels.txt) as a zip appended to the model."""
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".txt"))
        lines = zf.read(name).decode("utf-8").splitlines()
    # '???' marks unused slots of the original 91-id COCO label space; keep them as
    # None so indices still line up with the model's output columns.
    return [None if line.strip() in ("", "???") else line.strip() for line in lines]


def repo_class_list(url: str) -> list[str | None]:
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https URL
        source = resp.read().decode("utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "COCO_CLASSES":
            return list(ast.literal_eval(node.value))
    raise RuntimeError(f"COCO_CLASSES not found in {url}")


def manifest_path(entry: dict) -> Path:
    return BROWSER_DIR / f"{entry['id']}.manifest.json"


def browser_manifest(entry: dict, path: Path, labels: list[str | None]) -> dict:
    """The runtime manifest the frontend loads (field order kept stable)."""
    b, w = entry["browser"], entry["weights"]
    manifest = {
        "id": entry["id"], "name": entry["name"], "role": b["role"], "file": w["file"], "url": w["url"], "sha256": w["sha256"],
        "runtime": b["runtime"], "license": b["license"], "dataset": b["dataset"], "paper": entry["paper"],
        "reported": entry["reported"], "input": entry["input"],
    }
    if "decode" in entry:
        manifest["decode"] = entry["decode"]
    manifest["postprocess"] = entry["postprocess"]
    labels_from = b["labels_from"]
    manifest.update({
        "bytes": path.stat().st_size,
        "labels_source": "labels.txt packed in the TFLite model metadata" if labels_from == "tflite-metadata" else labels_from,
        "num_classes": sum(1 for label in labels if label),
        "labels": labels,
    })
    return manifest


def ensure_file(entry: dict, *, check_only: bool) -> tuple[Path | None, int]:
    """Download if missing, verify (or record) the checksum. Returns (path, problems)."""
    path = model_path(entry)
    print(f"{entry['id']}: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    if not path.exists():
        if check_only:
            print("  MISSING")
            return None, 1
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        download(entry["weights"]["url"], tmp)
        tmp.replace(path)
    digest = sha256(path)
    expected = entry["weights"].get("sha256")
    if expected is None:
        if check_only:
            print("  no checksum recorded in the registry")
            return path, 1
        entry["weights"]["sha256"] = digest
        entry["weights"]["bytes"] = path.stat().st_size
        save_entry(entry)
        print(f"  recorded sha256 {digest} (first download; review models/registry/{entry['id']}.json)")
    elif digest != expected:
        print(f"  SHA-256 mismatch: {digest}")
        if not check_only:
            path.unlink()  # never keep (or serve) a file that is not the published model
        return None, 1
    print(f"  ok - {path.stat().st_size / 1e6:.1f} MB")
    return path, 0


def build(ids: list[str] | None, *, check_only: bool, keep_manifests: bool) -> int:
    registry = load_registry()
    if ids:
        unknown = [i for i in ids if i not in registry]
        if unknown:
            raise SystemExit(f"unknown model(s): {', '.join(unknown)}")
        selected = [registry[i] for i in ids]
    else:
        selected = [e for e in registry.values() if e.get("fetch_by_default")]
    problems = 0
    for entry in selected:
        if entry.get("status") not in ("bundled", "fetchable"):
            print(f"{entry['id']}: {entry.get('status')} - {entry.get('unavailable_reason') or entry.get('build', '')}")
            continue
        path, bad = ensure_file(entry, check_only=check_only)
        problems += bad
        if path is None or "browser" not in entry:
            continue
        if check_only or keep_manifests:
            labels = json.loads(manifest_path(entry).read_text())["labels"]
        elif entry["browser"]["labels_from"] == "tflite-metadata":
            labels = tflite_labels(path)
        else:
            labels = repo_class_list(entry["browser"]["labels_from"])
        manifest = browser_manifest(entry, path, labels)
        if not (check_only or keep_manifests):
            manifest_path(entry).write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  manifest ok - {manifest['num_classes']} classes")
    return problems


def list_registry() -> int:
    for entry in load_registry().values():
        present = "present" if entry.get("weights") and model_path(entry).exists() else "-"
        tasks = ",".join(entry.get("tasks", []))
        print(f"{entry['id']:24} {entry.get('status', '?'):10} {present:8} {tasks:28} {entry['licence']['weights']}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", nargs="*", help="registry ids (default: models marked fetch_by_default)")
    parser.add_argument("--check", action="store_true", help="verify only; no downloads or writes")
    parser.add_argument("--fetch", action="store_true", help="download missing models; keep the committed manifests")
    parser.add_argument("--list", action="store_true", help="summarise the registry")
    args = parser.parse_args()
    if args.list:
        sys.exit(list_registry())
    sys.exit(1 if build(args.ids or None, check_only=args.check, keep_manifests=args.fetch) else 0)
