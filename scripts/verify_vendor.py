#!/usr/bin/env python3
"""Verify the vendored SDM runtime tree recorded by its manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        relative = path.relative_to(root).as_posix()
        if relative == "manifest.json" or "__pycache__" in path.parts:
            continue
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_vendor(root: Path) -> None:
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read vendored-runtime manifest: {error}") from error
    if manifest.get("schema_version") != 3:
        raise ValueError("unexpected vendored-runtime manifest schema")
    mapping = manifest.get("upstream_to_vendored")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("vendored-runtime lineage mapping is absent")
    for relative, identities in mapping.items():
        if not isinstance(identities, dict):
            raise ValueError(f"invalid lineage mapping for {relative}")
        upstream = identities.get("official_baseline_sha256")
        vendored = identities.get("vendored_sha256")
        if not all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in (upstream, vendored)
        ):
            raise ValueError(f"invalid lineage hash for {relative}")
        path = root / relative
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != vendored
        ):
            raise ValueError(f"vendored lineage file changed: {relative}")
    observed = tree_sha256(root)
    if observed != manifest.get("vendored_tree_sha256"):
        raise ValueError("vendored SDM runtime tree does not match its manifest")


def main() -> None:
    validate_vendor(ROOT / "third_party/runtime")
    validate_babylm_vendor(ROOT / "third_party/babylm_runtime")
    print("vendored SDM runtime verified")


def validate_babylm_vendor(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("revision") != "61d29928aa7520f421e0bc39d02b4e5006ffd5a1":
        raise ValueError("BabyLM training runtime revision changed")
    observed = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*") if path.is_file()
                and path.name != "manifest.json" and "__pycache__" not in path.parts}
    if observed != manifest.get("files"):
        raise ValueError("BabyLM runtime differs from its pinned source inventory")


if __name__ == "__main__":
    main()
