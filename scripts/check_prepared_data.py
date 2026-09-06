#!/usr/bin/env python3
"""Verify the complete prepared WikiText payload before allocating GPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reproduction.spec import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_INVENTORY_SHA256,
    EXPECTED_PAYLOAD_SHA256,
    PASSES,
    TOTAL_STEPS,
)
from reproduction.wikitext103_coverage import CanonicalWikiTextData, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if sha256_file(args.manifest) != EXPECTED_MANIFEST_SHA256:
        raise SystemExit("prepared manifest SHA-256 mismatch")
    data = CanonicalWikiTextData(args.manifest, passes=PASSES)
    if data.training_steps != TOTAL_STEPS:
        raise SystemExit("prepared training schedule changed")
    if data.manifest["remote_payload"]["sha256"] != EXPECTED_PAYLOAD_SHA256:
        raise SystemExit("prepared payload SHA-256 mismatch")
    attestation_path = args.manifest.parent / "DOUBLE_BUILD.json"
    if not attestation_path.is_file():
        raise SystemExit("prepared data has no deterministic double-build attestation")
    attestation = json.loads(attestation_path.read_text())
    expected_attestation = {
        "status": "byte_identical_double_build",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "remote_payload_sha256": EXPECTED_PAYLOAD_SHA256,
        "inventory_sha256": EXPECTED_INVENTORY_SHA256,
        "passes": PASSES,
        "stream_seed": 20_260_818,
    }
    for field, expected in expected_attestation.items():
        if attestation.get(field) != expected:
            raise SystemExit(f"double-build attestation changed at {field}")
    print(f"prepared WikiText payload verified: {EXPECTED_PAYLOAD_SHA256}")


if __name__ == "__main__":
    main()
