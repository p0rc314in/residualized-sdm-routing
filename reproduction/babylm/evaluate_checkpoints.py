"""Evaluate every retained BabyLM checkpoint on the official fast surfaces.

The packed inputs preserve the scoring semantics of ``babylm-org/babylm-eval``
revision ``6f825c291e2c4c78ad33b1935fd64d45f52642dc``.  This module performs only
model inference: official Reading and AoA statistical aggregation is applied to
the retained surprisal arrays after artifact retrieval.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .io import atomic_json, sha256_file
from .model import build_model, model_contract
from .spec import (
    CHECKPOINT_INPUT_SHA256,
    CAMPAIGN_ID,
    CANONICAL_SDM_COMMIT,
    CHECKPOINT_EXPOSURES_MILLIONS,
    EVALUATION_SDM_COMMIT,
    EVALUATION_REVISION,
    EVALUATOR_REVISION,
    SPEC,
    validate_arm,
)


def exposures_for_shard(shard_index: int, shard_count: int) -> list[int]:
    if shard_count < 1:
        raise ValueError("shard count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard index is outside the shard count")
    return [
        exposure
        for index, exposure in enumerate(CHECKPOINT_EXPOSURES_MILLIONS)
        if index % shard_count == shard_index
    ]


def load_record(root: Path, record: dict[str, Any]) -> np.ndarray:
    path = root / str(record["path"])
    if sha256_file(path) != str(record["sha256"]):
        raise ValueError(f"packed evaluation record changed: {path}")
    shape = tuple(int(value) for value in record["shape"])
    dtype = np.dtype(str(record["dtype"]))
    if path.suffix == ".npy":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        values = np.memmap(path, mode="r", dtype=dtype, shape=shape)
    if values.shape != shape or values.dtype != dtype:
        raise ValueError(f"packed evaluation record contract changed: {path}")
    return values


class PackedEvaluation:
    def __init__(self, root: Path, expected_manifest_sha256: str) -> None:
        self.root = root
        manifest_path = root / "manifest.json"
        if expected_manifest_sha256 != CHECKPOINT_INPUT_SHA256:
            raise ValueError("checkpoint evaluation requires the recorded input identity")
        digest = sha256_file(manifest_path)
        if digest != expected_manifest_sha256:
            raise ValueError(
                f"checkpoint evaluation manifest changed: {digest} != "
                f"{expected_manifest_sha256}"
            )
        self.manifest_sha256 = digest
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("format") != (
            "babylm2026_strict_gpt2_checkpoint_evaluation_v1"
        ):
            raise ValueError("unsupported checkpoint evaluation input")
        self.sections: dict[str, dict[str, np.ndarray]] = {}
        for name in ("fast_zero_shot", "reading", "aoa"):
            self.sections[name] = {
                key: load_record(root, record)
                for key, record in self.manifest[name]["records"].items()
            }
        self.eot_token = int(self.manifest["tokenizer"]["eot_token"])

    def sequence(self, section: str, index: int) -> np.ndarray:
        arrays = self.sections[section]
        offsets = arrays["offsets"]
        return arrays["tokens"][int(offsets[index]) : int(offsets[index + 1])]


def to_device(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.int64, copy=False)).cuda(
        non_blocking=True
    )


def write_npy(path: Path, values: np.ndarray) -> str:
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


@torch.no_grad()
def score_sequences(
    model: torch.nn.Module,
    data: PackedEvaluation,
    section: str,
    batch_size: int,
    *,
    exposure_millions: int,
) -> np.ndarray:
    arrays = data.sections[section]
    score_starts = arrays["score_starts"]
    scores = np.empty(len(score_starts), dtype=np.float64)
    model.eval()
    started = time.perf_counter()
    for first in range(0, len(score_starts), batch_size):
        stop = min(first + batch_size, len(score_starts))
        sequences = [data.sequence(section, index) for index in range(first, stop)]
        lengths = np.asarray([len(row) for row in sequences], dtype=np.int64)
        padded = np.full(
            (len(sequences), int(lengths.max())), data.eot_token, dtype=np.uint16
        )
        for row, values in enumerate(sequences):
            padded[row, : len(values)] = values
        values = to_device(padded)
        targets = values[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(values[:, :-1], attn_impl="sdpa")
            losses = F.cross_entropy(
                logits.flatten(0, 1), targets.flatten(), reduction="none"
            ).reshape_as(targets)
        for row, sequence in enumerate(range(first, stop)):
            score_start = int(score_starts[sequence])
            length = int(lengths[row])
            scores[sequence] = float(
                losses[row, score_start - 1 : length - 1].sum()
            )
        if stop % 5_000 < batch_size or stop == len(score_starts):
            print(
                json.dumps(
                    {
                        "marker": "HEARTBEAT",
                        "phase": section,
                        "official_exposure_millions": exposure_millions,
                        "sequences": stop,
                        "sequences_total": len(score_starts),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return scores


def aggregate_fast(
    data: PackedEvaluation, losses: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    arrays = data.sections["fast_zero_shot"]
    example_offsets = arrays["example_offsets"]
    labels = arrays["labels"]
    task_ids = arrays["task_ids"]
    subdomain_ids = arrays["subdomain_ids"]
    normalized = arrays["length_normalized"]
    sequence_offsets = arrays["offsets"]
    score_starts = arrays["score_starts"]
    predictions = np.empty(len(labels), dtype=np.uint8)
    correct = np.empty(len(labels), dtype=np.uint8)
    ties = 0
    for example in range(len(labels)):
        first = int(example_offsets[example])
        stop = int(example_offsets[example + 1])
        scores = -losses[first:stop].copy()
        if int(normalized[example]):
            counts = np.asarray(
                [
                    int(sequence_offsets[index + 1])
                    - int(sequence_offsets[index])
                    - int(score_starts[index])
                    for index in range(first, stop)
                ],
                dtype=np.float64,
            )
            scores /= counts
        winners = np.flatnonzero(scores == scores.max())
        ties += int(len(winners) > 1)
        predictions[example] = int(winners[0])
        correct[example] = predictions[example] == int(labels[example])

    counts: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    for example, value in enumerate(correct):
        row = counts[(int(task_ids[example]), int(subdomain_ids[example]))]
        row[0] += int(value)
        row[1] += 1
    tasks: dict[str, Any] = {}
    task_names = data.manifest["fast_zero_shot"]["tasks"]
    subdomain_names = data.manifest["fast_zero_shot"]["subdomains"]
    for task_index, task in enumerate(task_names):
        subdomains = sorted(
            {
                int(subdomain_ids[index])
                for index in np.flatnonzero(task_ids == task_index)
            }
        )
        subdomain_accuracy = {
            subdomain_names[subdomain]: (
                counts[(task_index, subdomain)][0]
                / counts[(task_index, subdomain)][1]
            )
            for subdomain in subdomains
        }
        if task == "entity_tracking":
            split_accuracy = {}
            for split in ("regular", "ambiref", "move_contents"):
                values = [
                    accuracy
                    for name, accuracy in subdomain_accuracy.items()
                    if name.startswith(split)
                ]
                if values:
                    split_accuracy[split] = sum(values) / len(values)
            average = sum(split_accuracy.values()) / len(split_accuracy)
        else:
            split_accuracy = {}
            average = sum(subdomain_accuracy.values()) / len(subdomain_accuracy)
        indices = np.flatnonzero(task_ids == task_index)
        tasks[str(task)] = {
            "accuracy": average,
            "micro_accuracy": float(correct[indices].mean()),
            "examples": len(indices),
            "subdomain_accuracy": subdomain_accuracy,
            "split_accuracy": split_accuracy,
        }
    return (
        {
            "tasks": tasks,
            "macro_average": sum(row["accuracy"] for row in tasks.values())
            / len(tasks),
            "examples": len(labels),
            "candidates": len(losses),
            "exact_score_ties": ties,
        },
        predictions,
    )


def load_checkpoint(
    arm: str,
    output_root: Path,
    model: torch.nn.Module,
    training_manifest_sha256: str,
    exposure_millions: int,
) -> tuple[dict[str, Any], Path]:
    arm_root = output_root / CAMPAIGN_ID / arm
    directory = arm_root / "checkpoints" / f"official-{exposure_millions:04d}M"
    record = json.loads((directory / "CHECKPOINT.json").read_text())
    checkpoint_path = arm_root / str(record["path"])
    if sha256_file(checkpoint_path) != str(record["sha256"]):
        raise ValueError(f"checkpoint hash changed at {exposure_millions}M")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = {
        "campaign_id": CAMPAIGN_ID,
        "arm": arm,
        "official_exposure_millions": exposure_millions,
        "canonical_sdm_commit": CANONICAL_SDM_COMMIT,
        "dataset_manifest_sha256": training_manifest_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"checkpoint identity changed at {exposure_millions}M:{key}"
            )
    model.load_state_dict(payload["model"])
    return record, checkpoint_path


def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    validate_arm(args.arm)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(SPEC.seed)
    torch.cuda.manual_seed_all(SPEC.seed)
    training_manifest_sha256 = sha256_file(args.training_manifest)
    data = PackedEvaluation(args.input_root, args.input_manifest_sha256)
    model = build_model(args.arm).to(device="cuda", dtype=torch.bfloat16)
    output = (
        args.output_root
        / CAMPAIGN_ID
        / args.arm
        / "official-checkpoint-evaluation"
    )
    output.mkdir(parents=True, exist_ok=True)
    selected_exposures = exposures_for_shard(args.shard_index, args.shard_count)
    configuration = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "arm": args.arm,
        "model_contract": model_contract(args.arm, model),
        "training_manifest_sha256": training_manifest_sha256,
        "input_manifest_sha256": data.manifest_sha256,
        "evaluation_dataset_revision": EVALUATION_REVISION,
        "evaluation_implementation_revision": EVALUATOR_REVISION,
        "evaluation_sdm_commit": EVALUATION_SDM_COMMIT,
        "training_sdm_commit": CANONICAL_SDM_COMMIT,
        "checkpoint_exposures_millions": list(CHECKPOINT_EXPOSURES_MILLIONS),
        "selected_exposures_millions": selected_exposures,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "batch_size": args.batch_size,
        "device": torch.cuda.get_device_name(),
    }
    shard_name = f"shard-{args.shard_index:02d}-of-{args.shard_count:02d}"
    configuration_path = (
        output / "CONFIG.json"
        if args.shard_count == 1
        else output / f"CONFIG.{shard_name}.json"
    )
    atomic_json(configuration_path, configuration)
    print(json.dumps({"marker": "STARTED", **configuration}, sort_keys=True), flush=True)
    started = time.perf_counter()
    checkpoints: dict[str, Any] = {}
    for exposure in selected_exposures:
        result_path = output / f"official-{exposure:04d}M.json"
        if result_path.is_file():
            checkpoints[str(exposure)] = json.loads(result_path.read_text())
            continue
        checkpoint, checkpoint_path = load_checkpoint(
            args.arm,
            args.output_root,
            model,
            training_manifest_sha256,
            exposure,
        )
        checkpoint_started = time.perf_counter()
        fast_losses = score_sequences(
            model,
            data,
            "fast_zero_shot",
            args.batch_size,
            exposure_millions=exposure,
        )
        fast, predictions = aggregate_fast(data, fast_losses)
        reading = score_sequences(
            model,
            data,
            "reading",
            args.batch_size,
            exposure_millions=exposure,
        )
        aoa = score_sequences(
            model,
            data,
            "aoa",
            args.batch_size,
            exposure_millions=exposure,
        )
        prefix = output / f"official-{exposure:04d}M"
        hashes = {
            "fast_candidate_losses": write_npy(
                prefix.with_name(prefix.name + "-fast-candidate-losses.npy"),
                fast_losses,
            ),
            "fast_predictions": write_npy(
                prefix.with_name(prefix.name + "-fast-predictions.npy"),
                predictions,
            ),
            "reading_surprisals": write_npy(
                prefix.with_name(prefix.name + "-reading-surprisals.npy"), reading
            ),
            "aoa_surprisals": write_npy(
                prefix.with_name(prefix.name + "-aoa-surprisals.npy"), aoa
            ),
        }
        result = {
            "official_exposure_millions": exposure,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint["sha256"],
            "fast_zero_shot": fast,
            "reading_sequences": len(reading),
            "aoa_sequences": len(aoa),
            "array_sha256": hashes,
            "seconds": time.perf_counter() - checkpoint_started,
        }
        atomic_json(result_path, result)
        checkpoints[str(exposure)] = result
        print(
            json.dumps(
                {
                    "marker": "CHECKPOINT_COMPLETE",
                    "arm": args.arm,
                    "official_exposure_millions": exposure,
                    "fast_macro_average": fast["macro_average"],
                    "seconds": result["seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    result = {
        **configuration,
        "status": "complete_checkpoint_evaluation_shard",
        "checkpoints": checkpoints,
        "seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    result_path = (
        output / "RESULT.json"
        if args.shard_count == 1
        else output / f"RESULT.{shard_name}.json"
    )
    atomic_json(result_path, result)
    marker = "COMPLETE" if args.shard_count == 1 else "SHARD_COMPLETE"
    print(json.dumps({"marker": marker, **result}, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
