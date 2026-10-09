#!/usr/bin/env python3
"""Dataset tool: list, fetch, validate, split and summarise registered datasets.

    python tools/datasets.py list
    python tools/datasets.py fetch coco_val2017 [--force]
    python tools/datasets.py validate coco_val2017
    python tools/datasets.py split visdrone_det_val
    python tools/datasets.py stats coco_val2017

Data goes to RD_DATA_DIR (default <repo>/data, git-ignored); manifests with
sources, licences and checksums go to data/manifests/ (versioned). See
docs/DATASET_CATALOG.md for what each dataset is and is not suitable for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.datasets import fetch as fetching  # noqa: E402
from app.datasets import paths, sources  # noqa: E402


def cmd_list(_: argparse.Namespace) -> int:
    for src in sources.SOURCES.values():
        manifest = fetching.load_manifest(src.id)
        state = "fetched" if manifest.get("fetched_at") else "not fetched"
        print(f"{src.id:26} {state:12} {src.approx_bytes / 1e9:6.2f} GB  {src.licence.commercial_use}  - {src.name}")
    print(f"\ndata root: {paths.data_root()}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    for dataset_id in args.ids:
        src = sources.get(dataset_id)
        print(f"{src.id}: {src.name}", flush=True)
        if not src.licence.verified:
            print(f"  note: licence not yet verified ({src.licence.commercial_use}); local evaluation only", flush=True)
        manifest = fetching.fetch(src, force=args.force)
        print(json.dumps(manifest["archives"], indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from app.datasets import validate

    report = validate.run(args.id)
    print(json.dumps(report.summary(), indent=2))
    return 0 if report.ok else 1


def cmd_split(args: argparse.Namespace) -> int:
    from app.datasets import splits

    result = splits.make(args.id)
    print(json.dumps(result, indent=2))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from app.datasets import adapters

    print(json.dumps(adapters.load(args.id).stats(), indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    p = sub.add_parser("fetch")
    p.add_argument("ids", nargs="+", choices=sorted(sources.SOURCES))
    p.add_argument("--force", action="store_true", help="download again even if extracted")
    p.set_defaults(fn=cmd_fetch)
    for name, fn in (("validate", cmd_validate), ("split", cmd_split), ("stats", cmd_stats)):
        p = sub.add_parser(name)
        p.add_argument("id", choices=sorted(sources.SOURCES))
        p.set_defaults(fn=fn)
    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
