#!/usr/bin/env python3
"""Validate fresh five-arm metrics and terminal checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import tempfile
import os
from typing import Any

import torch

from reproduction.spec import (
    ARM_ORDER,
    ARMS,
    EXPECTED_GAP_CLOSURE_RANGES,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_NLL_RANGES,
    GEOMETRY,
    TOTAL_STEPS,
)
from reproduction.wikitext103_coverage import VOCAB_SIZE, sha256_file
from reproduction.model import build_language_model
from shared_residual_routing.router import install_residualized_routing


PAIRS = {
    "b7a1": ("b7a1_native_sdm", "b7a1_residualized_sdm"),
    "b8": ("b8_native_sdm", "b8_residualized_sdm"),
}


def roundtrip_checkpoint(
    checkpoint_payload: dict[str, Any],
    *,
    arm_name: str,
) -> None:
    if not torch.cuda.is_available():
        raise ValueError("checkpoint inference validation requires CUDA")
    arm = ARMS[arm_name]
    model = build_language_model(
        vocab_size=VOCAB_SIZE,
        geometry=GEOMETRY,
        profile=arm.profile,
    )
    if arm.router == "residualized_read_write":
        install_residualized_routing(model)
    model.load_state_dict(checkpoint_payload["model"], strict=True)
    device = torch.device("cuda")
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    inputs = torch.arange(32, device=device).remainder(VOCAB_SIZE).view(1, -1)
    with torch.inference_mode():
        logits = model(inputs, attn_impl="sdpa")
    if logits.shape != (1, 32, VOCAB_SIZE) or not torch.isfinite(logits).all():
        raise ValueError(f"{arm_name} checkpoint inference output is invalid")
    del logits, inputs, model
    torch.cuda.empty_cache()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_result(root: Path, arm_name: str) -> dict[str, Any]:
    arm_root = root / arm_name
    result_path = arm_root / "result.json"
    if not result_path.is_file():
        raise ValueError(f"missing terminal result for {arm_name}")
    result = json.loads(result_path.read_text())
    if result.get("schema") != "residualized-routing-wikitext103-result-v1":
        raise ValueError(f"unexpected result schema for {arm_name}")
    if result.get("status") != "terminal_verified" or result.get("arm") != arm_name:
        raise ValueError(f"{arm_name} is not terminal and verified")
    config = result.get("config", {})
    expected_arm = ARMS[arm_name]
    exact = {
        "total_steps": TOTAL_STEPS,
        "reads": 8,
        "writes": 8,
        "topology": expected_arm.profile.layout,
        "router": expected_arm.router,
        "seed": 0,
    }
    for field, expected in exact.items():
        if config.get(field) != expected:
            raise ValueError(f"{arm_name} configuration changed at {field}")
    if result.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"{arm_name} used a different data manifest")
    if result.get("trainable_parameters") != expected_arm.expected_trainable_parameters:
        raise ValueError(f"{arm_name} parameter count changed")
    coverage = result.get("coverage", {})
    if (
        not coverage.get("complete")
        or coverage.get("consumed_optimizer_steps") != TOTAL_STEPS
    ):
        raise ValueError(f"{arm_name} coverage is incomplete")
    checkpoint = result.get("terminal_checkpoint", {})
    checkpoint_path = arm_root / str(checkpoint.get("path", ""))
    if not checkpoint_path.is_file():
        raise ValueError(f"{arm_name} terminal checkpoint is absent")
    if sha256_file(checkpoint_path) != checkpoint.get("sha256"):
        raise ValueError(f"{arm_name} terminal checkpoint hash mismatch")
    try:
        checkpoint_payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:
        raise ValueError(
            f"{arm_name} terminal checkpoint cannot be deserialized"
        ) from error
    if (
        not isinstance(checkpoint_payload, dict)
        or checkpoint_payload.get("schema") != "residualized-routing-terminal-model-v1"
        or json.loads(json.dumps(checkpoint_payload.get("config"))) != config
        or checkpoint_payload.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or not isinstance(checkpoint_payload.get("model"), dict)
        or not checkpoint_payload["model"]
    ):
        raise ValueError(f"{arm_name} terminal checkpoint contents are invalid")
    try:
        roundtrip_checkpoint(checkpoint_payload, arm_name=arm_name)
    except Exception as error:
        raise ValueError(
            f"{arm_name} terminal checkpoint failed reconstruction and inference"
        ) from error
    for field, range_name in (
        ("terminal_validation", "validation"),
        ("terminal_test", "test"),
    ):
        nll = result.get(field, {}).get("nll")
        if not isinstance(nll, (int, float)) or not math.isfinite(float(nll)):
            raise ValueError(f"{arm_name} {field} NLL is invalid")
        lower, upper = EXPECTED_NLL_RANGES[arm_name][range_name]
        if not lower <= float(nll) <= upper:
            raise ValueError(
                f"{arm_name} {field} NLL {float(nll):.5f} is outside "
                f"the predeclared [{lower:.2f}, {upper:.2f}] band"
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    results = {arm: load_result(root, arm) for arm in ARM_ORDER}
    dense = float(results["dense_attention"]["terminal_test"]["nll"])
    gaps: dict[str, dict[str, float]] = {}
    for topology, (native_name, residualized_name) in PAIRS.items():
        native = float(results[native_name]["terminal_test"]["nll"])
        residualized = float(results[residualized_name]["terminal_test"]["nll"])
        native_gap = native - dense
        if native_gap <= 0:
            raise ValueError(f"{topology} native SDM did not trail dense attention")
        fraction_closed = (native - residualized) / native_gap
        lower, upper = EXPECTED_GAP_CLOSURE_RANGES[topology]
        if residualized >= native or not lower <= fraction_closed <= upper:
            raise ValueError(f"{topology} did not reproduce the routing result")
        gaps[topology] = {
            "native_to_dense": native_gap,
            "residualized_to_dense": residualized - dense,
            "closed_by_residualization": native - residualized,
            "fraction_closed": fraction_closed,
        }

    rows = [
        {
            "arm": arm,
            "layout": ARMS[arm].profile.layout,
            "router": ARMS[arm].router,
            "trainable_parameters": results[arm]["trainable_parameters"],
            "validation_nll": results[arm]["terminal_validation"]["nll"],
            "test_nll": results[arm]["terminal_test"]["nll"],
            "checkpoint_sha256": results[arm]["terminal_checkpoint"]["sha256"],
            "checkpoint_roundtrip_inference_passed": True,
        }
        for arm in ARM_ORDER
    ]
    atomic_json(
        root / "measurements.json",
        {
            "schema_version": 1,
            "status": "reproduced",
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "arms": rows,
            "terminal_checkpoints_roundtrip_inference_passed": list(ARM_ORDER),
            "test_nll_gap": gaps,
        },
    )
    with (root / "measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        "fresh five-arm experiment reproduced: "
        + ", ".join(
            f"{topology} closed {values['fraction_closed']:.1%}"
            for topology, values in gaps.items()
        )
    )


if __name__ == "__main__":
    main()
