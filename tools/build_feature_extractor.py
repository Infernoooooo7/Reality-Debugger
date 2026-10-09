#!/usr/bin/env python3
"""Derive the patch-feature extractor used for anomaly detection from ResNet-50.

PatchCore-style anomaly detection needs mid-level feature maps, not class
scores. This script cuts the registered ResNet-50 v1 ONNX model
(models/registry/resnet50_v1.json, checksum-verified) after its second and
third residual stages and makes the spatial input size dynamic:

    data [N,3,H,W] (ImageNet-normalised RGB)
      -> resnetv17_stage2_activation3  [N, 512, H/8,  W/8 ]
      -> resnetv17_stage3_activation5  [N, 1024, H/16, W/16]

The weights are not changed. The result is written to models/weights/ (git
ignored) and registered as resnet50_v1_patch_features with the source
checksum, this script and the onnx version as provenance; its own SHA-256 is
recorded on the first build and verified on later builds.

    python tools/build_feature_extractor.py          build (downloads the source model if missing)
    python tools/build_feature_extractor.py --check  verify an existing build
    python tools/build_feature_extractor.py --strict build and require the recorded checksum (Docker)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_models  # noqa: E402

SOURCE_ID = "resnet50_v1"
TARGET_ID = "resnet50_v1_patch_features"
OUTPUTS = ("resnetv17_stage2_activation3", "resnetv17_stage3_activation5")


def build(out: Path, source: Path) -> None:
    import onnx
    from onnx.utils import Extractor

    model = onnx.load(str(source))
    sub = Extractor(model).extract_model(["data"], list(OUTPUTS))
    # Fully convolutional up to stage 3: let H and W vary.
    for value in [*sub.graph.input, *sub.graph.output]:
        dims = value.type.tensor_type.shape.dim
        if len(dims) == 4:
            dims[0].dim_param = "N"
            dims[2].dim_param = "H" if value.name == "data" else f"{value.name}_h"
            dims[3].dim_param = "W" if value.name == "data" else f"{value.name}_w"
    del sub.graph.value_info[:]  # stale fixed-size shape annotations
    sub.producer_name = "reality-debugger tools/build_feature_extractor.py"
    onnx.checker.check_model(sub)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(sub, str(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify the existing build only")
    ap.add_argument("--strict", action="store_true", help="fail if the built file differs from the recorded checksum (Docker build)")
    args = ap.parse_args()

    import onnx

    registry = fetch_models.load_registry()
    src_entry, entry = registry[SOURCE_ID], registry[TARGET_ID]
    out = fetch_models.model_path(entry)
    if args.check:
        if not out.exists():
            print(f"{TARGET_ID}: missing ({out})")
            return 1
        ok = fetch_models.sha256(out) == entry["weights"].get("sha256")
        print(f"{TARGET_ID}: {'ok' if ok else 'CHECKSUM MISMATCH'}")
        return 0 if ok else 1
    source, problems = fetch_models.ensure_file(src_entry, check_only=False)
    if source is None or problems:
        print(f"{SOURCE_ID}: source model unavailable", file=sys.stderr)
        return 1
    build(out, source)
    digest = fetch_models.sha256(out)
    recorded = entry["weights"].get("sha256")
    derived = entry.setdefault("derived_from", {})
    derived.update({"id": SOURCE_ID, "sha256": src_entry["weights"]["sha256"],
                    "script": "tools/build_feature_extractor.py", "onnx_version": onnx.__version__, "outputs": list(OUTPUTS)})
    if recorded and recorded != digest and args.strict:
        out.unlink()
        print(f"{TARGET_ID}: built file does not match the registry checksum ({digest[:12]} vs {recorded[:12]})", file=sys.stderr)
        return 1
    if recorded and recorded != digest:
        # Serialisation can differ between onnx versions; the weights cannot (they are copied verbatim).
        print(f"{TARGET_ID}: built file differs from the recorded checksum ({digest[:12]} vs {recorded[:12]}); "
              "recording the new checksum - review the registry diff", file=sys.stderr)
    entry["weights"]["sha256"] = digest
    entry["weights"]["bytes"] = out.stat().st_size
    fetch_models.save_entry(entry)
    print(f"{TARGET_ID}: {out} ({out.stat().st_size / 1e6:.1f} MB) sha256 {digest[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
