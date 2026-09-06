"""Run the official BabyLM terminal tasks on a retained campaign checkpoint.

The task packing and scoring semantics are derived from ``babylm-org/babylm-eval``
revision ``6f825c291e2c4c78ad33b1935fd64d45f52642dc``.  The packed adapter was
first implemented in ``scratch/causal-overwrite-bus`` at commit
``e5cdf6c81f9cf68746c1c89917e79ced8b815a4e``; this module is the narrow
campaign-native evaluator for the canonical Lingua model used here.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .data import BabyLMEvaluationData, TokenSplit
from .io import atomic_json, sha256_file
from .model import build_model, model_contract
from .spec import (
    CAMPAIGN_ID,
    CANONICAL_SDM_COMMIT,
    EVALUATION_SDM_COMMIT,
    EVALUATION_REVISION,
    EVALUATOR_REVISION,
    SPEC,
    validate_arm,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def hidden_states(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Return the same final hidden states consumed by the tied LM head."""
    from apps.main.transformer import create_causal_mask
    from lingua.transformer import BaseTransformer

    length = input_ids.shape[1]
    mask = {
        "attn": create_causal_mask(
            length, "sdpa", None, device=input_ids.device
        )
    }
    hidden = model.tok_embeddings(input_ids)
    hidden = BaseTransformer.forward(
        model, hidden, mask=mask, attn_impl="sdpa"
    )
    return model.norm(hidden)


@torch.no_grad()
def evaluate_zero_shot(
    model: nn.Module,
    data: BabyLMEvaluationData,
    batch_size: int,
    output: Path,
) -> dict[str, Any]:
    model.eval()
    offsets = data.zero_arrays["candidate_offsets"]
    score_starts = data.zero_arrays["score_starts"]
    candidate_scores = np.empty(len(score_starts), dtype=np.float64)
    started = time.perf_counter()
    for first in range(0, len(score_starts), batch_size):
        stop = min(first + batch_size, len(score_starts))
        sequences = [data.zero_candidate(index) for index in range(first, stop)]
        lengths = np.asarray([len(row) for row in sequences], dtype=np.int64)
        padded = np.full(
            (len(sequences), int(lengths.max())),
            data.eot_token,
            dtype=np.uint16,
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
        for row, candidate in enumerate(range(first, stop)):
            score_start = int(score_starts[candidate])
            length = int(lengths[row])
            candidate_scores[candidate] = -float(
                losses[row, score_start - 1 : length - 1].sum()
            )
        if stop % 5_000 < batch_size or stop == len(score_starts):
            print(
                json.dumps(
                    {
                        "marker": "HEARTBEAT",
                        "phase": "zero_shot",
                        "candidates": stop,
                        "candidates_total": len(score_starts),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    example_offsets = data.zero_arrays["example_offsets"]
    labels = data.zero_arrays["labels"]
    task_ids = data.zero_arrays["task_ids"]
    subdomain_ids = data.zero_arrays["subdomain_ids"]
    normalized = data.zero_arrays["length_normalized"]
    predictions = np.empty(len(labels), dtype=np.uint8)
    correct = np.empty(len(labels), dtype=np.uint8)
    ties = 0
    for example in range(len(labels)):
        first = int(example_offsets[example])
        stop = int(example_offsets[example + 1])
        scores = candidate_scores[first:stop].copy()
        if int(normalized[example]):
            token_counts = np.asarray(
                [
                    int(offsets[index + 1])
                    - int(offsets[index])
                    - int(score_starts[index])
                    for index in range(first, stop)
                ],
                dtype=np.float64,
            )
            scores /= token_counts
        winners = np.flatnonzero(scores == scores.max())
        ties += int(len(winners) > 1)
        prediction = int(winners[0])
        predictions[example] = prediction
        correct[example] = prediction == int(labels[example])

    subdomain_counts: dict[tuple[int, int], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    for example, value in enumerate(correct):
        row = subdomain_counts[
            (int(task_ids[example]), int(subdomain_ids[example]))
        ]
        row[0] += int(value)
        row[1] += 1
    tasks: dict[str, Any] = {}
    for task_index, task in enumerate(data.zero_tasks):
        task_subdomains = sorted(
            {
                int(subdomain_ids[index])
                for index in np.flatnonzero(task_ids == task_index)
            }
        )
        subdomain_accuracy = {
            data.zero_subdomains[subdomain]: (
                subdomain_counts[(task_index, subdomain)][0]
                / subdomain_counts[(task_index, subdomain)][1]
            )
            for subdomain in task_subdomains
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
        tasks[task] = {
            "accuracy": average,
            "micro_accuracy": float(correct[indices].mean()),
            "examples": len(indices),
            "subdomain_accuracy": subdomain_accuracy,
            "split_accuracy": split_accuracy,
        }
    scores_hash = write_npy(output / "zero_shot_candidate_scores.npy", candidate_scores)
    predictions_hash = write_npy(output / "zero_shot_predictions.npy", predictions)
    return {
        "tasks": tasks,
        "macro_average": sum(row["accuracy"] for row in tasks.values()) / len(tasks),
        "examples": len(labels),
        "candidates": len(score_starts),
        "exact_score_ties": ties,
        "seconds": time.perf_counter() - started,
        "candidate_scores_sha256": scores_hash,
        "predictions_sha256": predictions_hash,
    }


class ClassifierHead(nn.Module):
    def __init__(self, labels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(SPEC.width, 1e-5, elementwise_affine=False),
            nn.Linear(SPEC.width, SPEC.width),
            nn.GELU(),
            nn.LayerNorm(SPEC.width, 1e-5, elementwise_affine=False),
            nn.Dropout(0.1),
            nn.Linear(SPEC.width, labels),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class SequenceClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, labels: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = ClassifierHead(labels)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden = hidden_states(self.backbone, input_ids)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return self.classifier(hidden[rows, lengths - 1])


def pad_sequences(
    split: TokenSplit, indices: np.ndarray, eot_token: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences = [split.sequence(int(index)) for index in indices]
    lengths = np.asarray([len(values) for values in sequences], dtype=np.int64)
    padded = np.full(
        (len(sequences), int(lengths.max())), eot_token, dtype=np.uint16
    )
    for row, values in enumerate(sequences):
        padded[row, : len(values)] = values
    labels = np.asarray(split.labels[indices], dtype=np.int64)
    return padded, lengths, labels


def classification_metrics(
    predictions: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    accuracy = float((predictions == labels).mean())
    if int(labels.max()) > 1:
        return {"accuracy": accuracy}
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    f1_denominator = 2 * tp + fp + fn
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "accuracy": accuracy,
        "f1": 0.0 if not f1_denominator else 2 * tp / f1_denominator,
        "mcc": 0.0 if not mcc_denominator else (tp * tn - fp * fn) / mcc_denominator,
    }


@torch.no_grad()
def evaluate_classifier(
    classifier: SequenceClassifier,
    split: TokenSplit,
    batch_size: int,
    eot_token: int,
) -> tuple[dict[str, float], np.ndarray]:
    classifier.eval()
    predictions: list[np.ndarray] = []
    for first in range(0, len(split), batch_size):
        indices = np.arange(first, min(first + batch_size, len(split)))
        padded, lengths, _ = pad_sequences(split, indices, eot_token)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = classifier(to_device(padded), to_device(lengths))
        predictions.append(logits.argmax(dim=-1).cpu().numpy())
    result = np.concatenate(predictions).astype(np.uint8, copy=False)
    labels = np.asarray(split.labels, dtype=np.uint8)
    return classification_metrics(result, labels), result


def lr_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.1, 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def finetune_task(
    model: nn.Module,
    base_state: dict[str, torch.Tensor],
    data: BabyLMEvaluationData,
    task: str,
    *,
    micro_batch_size: int,
    eval_batch_size: int,
    output: Path,
) -> dict[str, Any]:
    model.load_state_dict(base_state)
    config = data.manifest["finetune"]["tasks"][task]
    hyper = data.manifest["finetune"]["official_hyperparameters"]
    effective_batch = int(config["batch_size"])
    if effective_batch % micro_batch_size:
        raise ValueError(f"micro batch does not divide the official {task} batch")
    train = data.finetune_split(task, "train")
    valid = data.finetune_split(task, "valid")
    order = data.finetune_order(task)
    epochs = int(config["epochs"])
    batches_per_epoch = len(train) // effective_batch
    total_steps = batches_per_epoch * epochs
    if len(order) != len(train) * epochs:
        raise ValueError(f"precomputed {task} order changed")

    seed = int(hyper["seed"])
    set_seed(seed)
    classifier = SequenceClassifier(model, int(config["num_labels"])).cuda()
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=float(hyper["learning_rate"]),
        betas=tuple(float(value) for value in hyper["betas"]),
        eps=float(hyper["epsilon"]),
        weight_decay=float(hyper["weight_decay"]),
        fused=True,
    )
    base_lr = float(hyper["learning_rate"])
    warmup_steps = int(float(hyper["warmup_proportion"]) * total_steps)
    selection_metric = str(config["selection_metric"])
    best_score: float | None = None
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epoch_metrics: list[dict[str, Any]] = []
    global_step = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        classifier.train()
        epoch_start = epoch * len(train)
        epoch_order = order[
            epoch_start : epoch_start + batches_per_epoch * effective_batch
        ]
        loss_sum = 0.0
        for batch_index in range(batches_per_epoch):
            first = batch_index * effective_batch
            indices = np.asarray(
                epoch_order[first : first + effective_batch], dtype=np.int64
            )
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for micro_start in range(0, effective_batch, micro_batch_size):
                micro = indices[micro_start : micro_start + micro_batch_size]
                padded, lengths, labels = pad_sequences(train, micro, data.eot_token)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = classifier(to_device(padded), to_device(lengths))
                    loss = F.cross_entropy(logits, to_device(labels))
                    scaled = loss * (len(micro) / effective_batch)
                scaled.backward()
                batch_loss += float(scaled.detach())
            factor = lr_factor(global_step, warmup_steps, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = base_lr * factor
            optimizer.step()
            global_step += 1
            loss_sum += batch_loss
        metrics, _ = evaluate_classifier(
            classifier, valid, eval_batch_size, data.eot_token
        )
        score = float(metrics[selection_metric])
        improved = best_score is None or score > best_score
        if improved:
            best_score = score
            best_epoch = epoch + 1
            best_state = cpu_state(classifier)
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / batches_per_epoch,
            "validation": metrics,
            "selection_metric": selection_metric,
            "selection_score": score,
            "best": improved,
            "global_step": global_step,
        }
        epoch_metrics.append(row)
        print(json.dumps({"marker": "HEARTBEAT", "phase": "finetune", "task": task, **row}, sort_keys=True), flush=True)

    if best_state is None:
        raise RuntimeError(f"{task} produced no best checkpoint")
    classifier.load_state_dict(best_state)
    final_metrics, predictions = evaluate_classifier(
        classifier, valid, eval_batch_size, data.eot_token
    )
    prediction_hash = write_npy(output / f"finetune_{task}_predictions.npy", predictions)
    result = {
        "task": task,
        "epochs": epochs,
        "effective_batch_size": effective_batch,
        "micro_batch_size": micro_batch_size,
        "training_steps": total_steps,
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_validation": final_metrics,
        "epoch_metrics": epoch_metrics,
        "seconds": time.perf_counter() - started,
        "predictions_sha256": prediction_hash,
    }
    atomic_json(output / f"finetune_{task}.json", result)
    del classifier, optimizer, best_state
    torch.cuda.empty_cache()
    return result


def load_checkpoint(
    arm: str, output_root: Path, model: nn.Module, manifest_sha256: str
) -> tuple[dict[str, Any], Path]:
    arm_root = output_root / CAMPAIGN_ID / arm
    record_path = arm_root / "checkpoints" / "official-1000M" / "CHECKPOINT.json"
    record = json.loads(record_path.read_text())
    checkpoint_path = arm_root / str(record["path"])
    if sha256_file(checkpoint_path) != str(record["sha256"]):
        raise ValueError("terminal checkpoint hash changed")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = {
        "campaign_id": CAMPAIGN_ID,
        "arm": arm,
        "official_exposure_millions": 1_000,
        "canonical_sdm_commit": CANONICAL_SDM_COMMIT,
        "dataset_manifest_sha256": manifest_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"checkpoint identity changed at {key}")
    model.load_state_dict(payload["model"])
    return record, checkpoint_path


def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    validate_arm(args.arm)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    set_seed(SPEC.seed)
    data = BabyLMEvaluationData(args.manifest, args.immutable_input_root)
    manifest_sha256 = sha256_file(args.manifest)
    model = build_model(args.arm).to(device="cuda", dtype=torch.bfloat16)
    checkpoint, checkpoint_path = load_checkpoint(
        args.arm, args.output_root, model, manifest_sha256
    )
    contract = model_contract(args.arm, model)
    output = args.output_root / CAMPAIGN_ID / args.arm / "official-evaluation"
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    configuration = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "arm": args.arm,
        "model_contract": contract,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint["sha256"],
        "evaluation_dataset_revision": EVALUATION_REVISION,
        "evaluation_implementation_revision": EVALUATOR_REVISION,
        "evaluation_sdm_commit": EVALUATION_SDM_COMMIT,
        "training_sdm_commit": CANONICAL_SDM_COMMIT,
        "dataset_manifest_sha256": manifest_sha256,
        "packed_adapter_provenance": {
            "repository": "scratch/causal-overwrite-bus",
            "commit": "e5cdf6c81f9cf68746c1c89917e79ced8b815a4e",
        },
        "zero_shot_batch_size": args.zero_shot_batch_size,
        "finetune_micro_batch_size": args.finetune_micro_batch_size,
        "finetune_eval_batch_size": args.finetune_eval_batch_size,
        "zero_shot_only": args.zero_shot_only,
        "finetune_task": args.finetune_task,
        "device": torch.cuda.get_device_name(),
    }
    configuration_path = (
        output / f"CONFIG.finetune-{args.finetune_task}.json"
        if args.finetune_task
        else output / "CONFIG.json"
    )
    if configuration_path.exists():
        previous = json.loads(configuration_path.read_text())
        for key in ("checkpoint_sha256", "dataset_manifest_sha256", "arm"):
            if previous.get(key) != configuration[key]:
                raise ValueError(f"cached evaluation belongs to different {key}")
    atomic_json(configuration_path, configuration)
    print(json.dumps({"marker": "STARTED", **configuration}, sort_keys=True), flush=True)

    if args.finetune_task:
        if args.finetune_task not in data.manifest["finetune"]["tasks"]:
            raise ValueError(f"unknown fine-tuning task: {args.finetune_task}")
        result_path = output / f"finetune_{args.finetune_task}.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
        else:
            result = finetune_task(
                model,
                cpu_state(model),
                data,
                args.finetune_task,
                micro_batch_size=args.finetune_micro_batch_size,
                eval_batch_size=args.finetune_eval_batch_size,
                output=output,
            )
        print(
            json.dumps(
                {
                    "marker": "COMPLETE",
                    "arm": args.arm,
                    "phase": "finetune",
                    **result,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    zero_path = output / "zero_shot.json"
    if zero_path.is_file():
        zero_shot = json.loads(zero_path.read_text())
    else:
        zero_shot = evaluate_zero_shot(
            model, data, args.zero_shot_batch_size, output
        )
        atomic_json(zero_path, zero_shot)
    print(
        json.dumps(
            {
                "marker": "ZERO_SHOT_COMPLETE",
                "arm": args.arm,
                "macro_average": zero_shot["macro_average"],
                "tasks": {
                    task: row["accuracy"]
                    for task, row in zero_shot["tasks"].items()
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if args.zero_shot_only:
        result = {
            **configuration,
            "status": "complete_terminal_zero_shot",
            "zero_shot": zero_shot,
            "seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }
        atomic_json(output / "ZERO_SHOT_RESULT.json", result)
        print(json.dumps({"marker": "COMPLETE", **result}, sort_keys=True), flush=True)
        return

    base_state = cpu_state(model)
    finetune: dict[str, Any] = {}
    for task in data.manifest["finetune"]["tasks"]:
        task_path = output / f"finetune_{task}.json"
        if task_path.is_file():
            finetune[task] = json.loads(task_path.read_text())
        else:
            finetune[task] = finetune_task(
                model,
                base_state,
                data,
                task,
                micro_batch_size=args.finetune_micro_batch_size,
                eval_batch_size=args.finetune_eval_batch_size,
                output=output,
            )
    model.load_state_dict(base_state)
    result = {
        **configuration,
        "status": "complete_terminal_core",
        "zero_shot": zero_shot,
        "finetune": finetune,
        "finetune_macro_primary_metric": sum(
            row["best_validation"][row["selection_metric"]] for row in finetune.values()
        ) / len(finetune),
        "human_likeness_status": "pending_separate_checkpoint_curve",
        "seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"marker": "COMPLETE", **result}, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--immutable-input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--zero-shot-batch-size", type=int, default=16)
    parser.add_argument("--finetune-micro-batch-size", type=int, default=8)
    parser.add_argument("--finetune-eval-batch-size", type=int, default=16)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--zero-shot-only", action="store_true")
    mode.add_argument("--finetune-task")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
