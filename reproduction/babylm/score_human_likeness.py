#!/usr/bin/env python3
"""Apply the pinned official Reading and AoA aggregators to retained scores."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import tiktoken

from reproduction.babylm.io import atomic_json, sha256_file
from reproduction.babylm.spec import (
    CAMPAIGN_ID,
    CHECKPOINT_EXPOSURES_MILLIONS,
    EVALUATOR_REVISION,
)


class GPT2TokenizerAdapter:
    vocab_size = 50_257

    def __init__(self) -> None:
        self.encoder = tiktoken.get_encoding("gpt2")

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, Any]:
        if add_special_tokens:
            raise ValueError("official AoA subword counting must omit special tokens")
        return {"input_ids": self.encoder.encode_ordinary(text)}


def require_evaluator_revision(root: Path) -> None:
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != EVALUATOR_REVISION:
        raise ValueError(
            f"official evaluator changed: {revision} != {EVALUATOR_REVISION}"
        )


def load_array(path: Path, expected: str) -> np.ndarray:
    if sha256_file(path) != expected:
        raise ValueError(f"inference array changed: {path}")
    return np.load(path, allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--packed-input-root", type=Path, required=True)
    parser.add_argument("--reading-data", type=Path, required=True)
    parser.add_argument("--cdi-human", type=Path, required=True)
    parser.add_argument("--official-evaluator-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require_evaluator_revision(args.official_evaluator_root)
    sys.path.insert(0, str(args.official_evaluator_root / "strict"))
    collate = importlib.import_module("evaluation_pipeline.collate_preds")
    evaluator_module = importlib.import_module("evaluation_pipeline.utils")

    packed_manifest = json.loads(
        (args.packed_input_root / "manifest.json").read_text()
    )
    reading_records = packed_manifest["reading"]["records"]
    aoa_records = packed_manifest["aoa"]["records"]
    current_indices = np.load(
        args.packed_input_root / reading_records["current_indices"]["path"],
        allow_pickle=False,
    )
    previous_indices = np.load(
        args.packed_input_root / reading_records["previous_indices"]["path"],
        allow_pickle=False,
    )
    word_ids = np.load(
        args.packed_input_root / aoa_records["word_ids"]["path"],
        allow_pickle=False,
    )
    words = packed_manifest["aoa"]["words"]

    reading: dict[str, Any] = {}
    aoa_results = []
    checkpoints: dict[str, Any] = {}
    for exposure in CHECKPOINT_EXPOSURES_MILLIONS:
        checkpoint = json.loads(
            (args.inference_root / f"official-{exposure:04d}M.json").read_text()
        )
        hashes = checkpoint["array_sha256"]
        prefix = args.inference_root / f"official-{exposure:04d}M"
        reading_values = load_array(
            prefix.with_name(prefix.name + "-reading-surprisals.npy"),
            hashes["reading_surprisals"],
        )
        aoa_values = load_array(
            prefix.with_name(prefix.name + "-aoa-surprisals.npy"),
            hashes["aoa_surprisals"],
        )
        predictions = []
        for current, previous in zip(current_indices, previous_indices, strict=True):
            predictions.append(
                {
                    "pred": float(reading_values[int(current)]),
                    "prev_pred": (
                        float("nan")
                        if int(previous) < 0
                        else float(reading_values[int(previous)])
                    ),
                }
            )
        reading[str(exposure)] = collate._calculate_reading_results(
            {"reading": {"predictions": predictions}}, args.reading_data
        )
        for index, surprisal in enumerate(aoa_values):
            aoa_results.append(
                {
                    "target_word": words[int(word_ids[index])],
                    "step": f"{exposure}M",
                    "surprisal": float(surprisal),
                }
            )
        checkpoints[str(exposure)] = {
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "fast_zero_shot": checkpoint["fast_zero_shot"],
            "reading": reading[str(exposure)],
        }

    aoa = evaluator_module.AoAEvaluator(args.cdi_human).compute_curve_fitness(
        {"results": aoa_results}, GPT2TokenizerAdapter()
    )
    result = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "arm": args.arm,
        "status": "complete",
        "official_evaluator_revision": EVALUATOR_REVISION,
        "reading_data_sha256": sha256_file(args.reading_data),
        "cdi_human_sha256": sha256_file(args.cdi_human),
        "checkpoints": checkpoints,
        "aoa": aoa,
    }
    atomic_json(args.output, result)
    print(json.dumps({"marker": "COMPLETE", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
