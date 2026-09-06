"""Evaluate one retained BabyLM terminal checkpoint on full EWoK."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .evaluate import evaluate_zero_shot, load_checkpoint
from .ewok import FullEWoKData
from .io import atomic_json, sha256_file
from .model import build_model, model_contract
from .spec import (
    CAMPAIGN_ID,
    CANONICAL_SDM_COMMIT,
    EVALUATOR_REVISION,
    EWOK_DATASET,
    EWOK_REVISION,
    FULL_EWOK_EVALUATION_SDM_COMMIT,
    SPEC,
    validate_arm,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--ewok-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    validate_arm(args.arm)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(SPEC.seed)
    np.random.seed(SPEC.seed)
    torch.manual_seed(SPEC.seed)
    torch.cuda.manual_seed_all(SPEC.seed)

    training_manifest_sha256 = sha256_file(args.training_manifest)
    ewok_manifest_sha256 = sha256_file(args.ewok_manifest)
    data = FullEWoKData(args.ewok_manifest)
    model = build_model(args.arm).to(device="cuda", dtype=torch.bfloat16)
    checkpoint, checkpoint_path = load_checkpoint(
        args.arm, args.output_root, model, training_manifest_sha256
    )
    output = (
        args.output_root
        / CAMPAIGN_ID
        / args.arm
        / "official-evaluation"
        / "full-ewok"
    )
    output.mkdir(parents=True, exist_ok=True)
    configuration = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "arm": args.arm,
        "model_contract": model_contract(args.arm, model),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint["sha256"],
        "training_manifest_sha256": training_manifest_sha256,
        "ewok_manifest_sha256": ewok_manifest_sha256,
        "ewok_dataset": EWOK_DATASET,
        "ewok_dataset_revision": EWOK_REVISION,
        "official_evaluator_revision": EVALUATOR_REVISION,
        "training_sdm_commit": CANONICAL_SDM_COMMIT,
        "evaluation_sdm_commit": FULL_EWOK_EVALUATION_SDM_COMMIT,
        "batch_size": args.batch_size,
        "device": torch.cuda.get_device_name(),
    }
    atomic_json(output / "CONFIG.json", configuration)
    print(json.dumps({"marker": "STARTED", **configuration}, sort_keys=True), flush=True)
    started = time.perf_counter()
    zero_shot = evaluate_zero_shot(model, data, args.batch_size, output)
    if set(zero_shot["tasks"]) != {"ewok"}:
        raise ValueError("full EWoK scorer produced the wrong task set")
    result = {
        **configuration,
        "status": "complete_full_ewok",
        "ewok": zero_shot["tasks"]["ewok"],
        "scoring": zero_shot,
        "seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    atomic_json(output / "RESULT.json", result)
    print(
        json.dumps(
            {
                "marker": "COMPLETE",
                "arm": args.arm,
                "ewok_accuracy": result["ewok"]["accuracy"],
                "examples": result["ewok"]["examples"],
                "seconds": result["seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
