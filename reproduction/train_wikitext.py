#!/usr/bin/env python3
"""Train and evaluate one arm of the canonical WikiText-103 comparison."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from reproduction.model import (
    build_language_model,
    common_parameter_sha256,
    initialize_role_keyed,
    parameter_accounting,
    validate_model_contract,
)
from reproduction.spec import (
    ARMS,
    CAMPAIGN_ID,
    EXPECTED_COMMON_SDM_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PAYLOAD_SHA256,
    GEOMETRY,
    PASSES,
    TOTAL_STEPS,
    WARMUP_STEPS,
)
from reproduction.wikitext103_coverage import (
    CanonicalWikiTextData,
    CONTEXT_LENGTH,
    EFFECTIVE_BATCH_RECORDS,
    EVALUATION_STRIDE,
    OPTIMIZER_STEPS_PER_PASS,
    PROTOCOL_ID,
    STREAM_SEED,
    VOCAB_SIZE,
    sha256_file,
)
from shared_residual_routing.router import install_residualized_routing


PEAK_LEARNING_RATE = 3e-4
MINIMUM_LR_RATIO = 0.1
ADAM_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
SEED = 0
MEMORY_BLOCK_SIZE = 256
RECOVERY_INTERVAL = 2_400
HEARTBEAT_INTERVAL = 50


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


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    host = torch.from_numpy(np.ascontiguousarray(array)).pin_memory()
    return host.to(device, non_blocking=True)


class FP32MasterAdamW:
    """AdamW with explicit FP32 master parameters and moment state."""

    def __init__(self, model: nn.Module) -> None:
        self.model_parameters = [p for p in model.parameters() if p.requires_grad]
        self.master_parameters = [
            nn.Parameter(p.detach().float().clone(), requires_grad=True)
            for p in self.model_parameters
        ]
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for model_parameter, master_parameter in zip(
            self.model_parameters, self.master_parameters, strict=True
        ):
            (
                no_decay
                if getattr(model_parameter, "_no_weight_decay", False)
                else decay
            ).append(master_parameter)
        groups: list[dict[str, Any]] = [{"params": decay, "weight_decay": WEIGHT_DECAY}]
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        self.optimizer = torch.optim.AdamW(
            groups, lr=PEAK_LEARNING_RATE, betas=ADAM_BETAS, fused=True
        )

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.optimizer.param_groups

    def zero_grad(self) -> None:
        for parameter in self.model_parameters:
            parameter.grad = None
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self) -> None:
        for model_parameter, master_parameter in zip(
            self.model_parameters, self.master_parameters, strict=True
        ):
            master_parameter.grad = (
                None
                if model_parameter.grad is None
                else model_parameter.grad.detach().float()
            )
        self.optimizer.step()
        for model_parameter, master_parameter in zip(
            self.model_parameters, self.master_parameters, strict=True
        ):
            model_parameter.copy_(master_parameter.to(dtype=model_parameter.dtype))

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "optimizer": self.optimizer.state_dict(),
            "master_parameters": [p.detach().cpu() for p in self.master_parameters],
        }

    @torch.no_grad()
    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("schema_version") != 1:
            raise ValueError("unexpected optimizer checkpoint schema")
        values = payload.get("master_parameters")
        if not isinstance(values, list) or len(values) != len(self.master_parameters):
            raise ValueError("optimizer checkpoint parameter count changed")
        for destination, source in zip(self.master_parameters, values, strict=True):
            if destination.shape != source.shape:
                raise ValueError("optimizer checkpoint parameter shape changed")
            destination.copy_(source.to(device=destination.device, dtype=torch.float32))
        self.optimizer.load_state_dict(payload["optimizer"])
        for model_parameter, master_parameter in zip(
            self.model_parameters, self.master_parameters, strict=True
        ):
            model_parameter.copy_(master_parameter.to(dtype=model_parameter.dtype))


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    campaign_id: str
    arm: str
    protocol_id: str
    passes: int
    total_steps: int
    effective_batch_records: int
    micro_batch_records: int
    context_length: int
    evaluation_stride: int
    peak_learning_rate: float
    minimum_lr_ratio: float
    warmup_steps: int
    adam_betas: tuple[float, float]
    weight_decay: float
    gradient_clip: float
    activation_dtype: str
    optimizer_state_dtype: str
    seed: int
    stream_seed: int
    reads: int
    writes: int
    factors: int
    codebook_size: int
    logical_rows: int
    router: str
    topology: str


def resolved_config(arm_name: str) -> ResolvedConfig:
    arm = ARMS[arm_name]
    return ResolvedConfig(
        campaign_id=CAMPAIGN_ID,
        arm=arm.identifier,
        protocol_id=PROTOCOL_ID,
        passes=PASSES,
        total_steps=TOTAL_STEPS,
        effective_batch_records=EFFECTIVE_BATCH_RECORDS,
        micro_batch_records=arm.micro_batch_size,
        context_length=CONTEXT_LENGTH,
        evaluation_stride=EVALUATION_STRIDE,
        peak_learning_rate=PEAK_LEARNING_RATE,
        minimum_lr_ratio=MINIMUM_LR_RATIO,
        warmup_steps=WARMUP_STEPS,
        adam_betas=ADAM_BETAS,
        weight_decay=WEIGHT_DECAY,
        gradient_clip=GRADIENT_CLIP,
        activation_dtype="bfloat16",
        optimizer_state_dtype="float32",
        seed=SEED,
        stream_seed=STREAM_SEED,
        reads=GEOMETRY.reads,
        writes=GEOMETRY.writes,
        factors=GEOMETRY.factors,
        codebook_size=GEOMETRY.codebook_size,
        logical_rows=GEOMETRY.logical_capacity,
        router=arm.router,
        topology=arm.profile.layout,
    )


def learning_rate(step: int) -> float:
    if not 1 <= step <= TOTAL_STEPS:
        raise ValueError("step lies outside the training schedule")
    if step <= WARMUP_STEPS:
        return PEAK_LEARNING_RATE * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    ratio = MINIMUM_LR_RATIO + 0.5 * (1.0 - MINIMUM_LR_RATIO) * (
        1.0 + math.cos(math.pi * progress)
    )
    return PEAK_LEARNING_RATE * ratio


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: CanonicalWikiTextData,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_nll = 0.0
    correct = 0
    targets = 0
    scored_bytes = 0
    windows = 0
    started = time.perf_counter()
    for batch in data.evaluation_batches(split):
        values = device_tensor(batch.tokens.astype(np.int64, copy=False), device)
        mask = device_tensor(batch.target_mask, device)
        inputs, labels = values[:, :-1], values[:, 1:]
        logits = model(inputs, attn_impl="sdpa")
        losses = F.cross_entropy(
            logits.float().flatten(0, 1), labels.flatten(), reduction="none"
        ).view_as(labels)
        total_nll += float(losses[mask].sum())
        correct += int(logits.argmax(dim=-1).eq(labels)[mask].sum())
        targets += batch.target_count
        scored_bytes += batch.scored_bytes
        windows += 1
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    expected = data.manifest["evaluation"]["splits"][split]
    if targets != int(expected["scored_targets"]):
        raise ValueError(f"{split} target coverage changed")
    if scored_bytes != int(expected["scored_bytes"]):
        raise ValueError(f"{split} byte coverage changed")
    nll = total_nll / targets
    return {
        "split": split,
        "windows": windows,
        "scored_targets": targets,
        "scored_bytes": scored_bytes,
        "total_nll": total_nll,
        "nll": nll,
        "perplexity": math.exp(nll),
        "next_token_accuracy": correct / targets,
        "seconds": elapsed,
    }


def save_recovery(
    output: Path,
    *,
    model: nn.Module,
    optimizer: FP32MasterAdamW,
    config: ResolvedConfig,
    manifest_sha256: str,
    step: int,
    evaluations: list[dict[str, Any]],
    curve: list[dict[str, Any]],
) -> Path:
    path = output / "checkpoints" / f"recovery-step-{step:05d}.pt"
    atomic_torch_save(
        path,
        {
            "schema": "residualized-routing-recovery-v1",
            "step": step,
            "config": asdict(config),
            "manifest_sha256": manifest_sha256,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "evaluations": evaluations,
            "curve": curve,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
            },
        },
    )
    atomic_json(
        output / "LATEST_RECOVERY.json",
        {
            "step": step,
            "path": path.relative_to(output).as_posix(),
            "sha256": sha256_file(path),
        },
    )
    return path


def load_recovery(
    path: Path,
    *,
    expected_sha256: str,
    model: nn.Module,
    optimizer: FP32MasterAdamW,
    config: ResolvedConfig,
    manifest_sha256: str,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("recovery checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "residualized-routing-recovery-v1":
        raise ValueError("unexpected recovery checkpoint schema")
    if payload.get("config") != asdict(config):
        raise ValueError("recovery checkpoint configuration changed")
    if payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError("recovery checkpoint data identity changed")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
    torch.set_rng_state(payload["rng"]["torch_cpu"])
    torch.cuda.set_rng_state_all(payload["rng"]["torch_cuda"])
    return int(payload["step"]), list(payload["evaluations"]), list(payload["curve"])


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the decision-bearing experiment")
    device = torch.device("cuda")
    arm = ARMS[args.arm]
    config = resolved_config(args.arm)
    if config.reads != config.writes:
        raise AssertionError("balanced access requires W = R")
    if TOTAL_STEPS != PASSES * OPTIMIZER_STEPS_PER_PASS:
        raise AssertionError("training schedule does not cover exactly three passes")

    manifest_sha256 = sha256_file(args.manifest)
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"manifest SHA-256 is {manifest_sha256}; expected {EXPECTED_MANIFEST_SHA256}"
        )
    data = CanonicalWikiTextData(args.manifest, passes=PASSES)
    if data.manifest["remote_payload"]["sha256"] != EXPECTED_PAYLOAD_SHA256:
        raise ValueError("prepared WikiText payload identity changed")

    set_seed(SEED)
    model = build_language_model(
        vocab_size=VOCAB_SIZE,
        geometry=GEOMETRY,
        profile=arm.profile,
        memory_block_size=MEMORY_BLOCK_SIZE,
    )
    initialize_role_keyed(model, SEED)
    common_sha256 = common_parameter_sha256(model)
    base_accounting = parameter_accounting(model)
    structural = validate_model_contract(model)
    routers = ()
    if arm.router == "residualized_read_write":
        routers = install_residualized_routing(model)
    if arm.has_sdm and common_sha256 != EXPECTED_COMMON_SDM_SHA256[arm.profile.layout]:
        raise ValueError("common SDM initialization identity changed")
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_parameters != arm.expected_trainable_parameters:
        raise ValueError(
            f"{arm.identifier} has {trainable_parameters} trainable parameters; "
            f"expected {arm.expected_trainable_parameters}"
        )
    if len(routers) not in (0, len(arm.profile.memory_layers)):
        raise AssertionError("router residualization is incomplete")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError(f"result already exists: {output / 'result.json'}")

    model.to(device=device, dtype=torch.bfloat16)
    optimizer = FP32MasterAdamW(model)
    evaluations: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    start_step = 0
    if args.resume is not None:
        if args.resume_sha256 is None:
            raise ValueError("--resume requires --resume-sha256")
        start_step, evaluations, curve = load_recovery(
            args.resume,
            expected_sha256=args.resume_sha256,
            model=model,
            optimizer=optimizer,
            config=config,
            manifest_sha256=manifest_sha256,
        )
    elif args.resume_sha256 is not None:
        raise ValueError("--resume-sha256 requires --resume")

    configuration = {
        "schema_version": 1,
        "config": asdict(config),
        "manifest_sha256": manifest_sha256,
        "payload_sha256": EXPECTED_PAYLOAD_SHA256,
        "common_parameter_sha256": common_sha256,
        "trainable_parameters": trainable_parameters,
        "base_parameter_accounting": base_accounting,
        "residualized_router_layers": len(routers),
        "structural_contract": structural,
        "runtime_manifest": json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "third_party/runtime/manifest.json"
            ).read_text()
        ),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
    }
    atomic_json(output / "config.json", configuration)
    coverage = data.coverage_report()
    coverage.update(
        {
            "consumed_optimizer_steps": start_step,
            "consumed_target_presentations": data.target_presentations_through_step(
                start_step
            ),
            "complete": start_step == TOTAL_STEPS,
        }
    )
    atomic_json(output / "coverage.json", coverage)
    print(
        json.dumps(
            {
                "marker": "STARTED",
                "arm": arm.identifier,
                "step": start_step,
                "steps": TOTAL_STEPS,
                "gpu": configuration["environment"]["gpu"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    metric_path = output / "training_curve.jsonl"
    with metric_path.open(
        "a" if start_step else "w", encoding="utf-8"
    ) as metric_handle:
        started = time.perf_counter()
        interval_started = started
        interval_targets = 0
        torch.cuda.reset_peak_memory_stats(device)
        checkpoint_steps = set(
            int(step) for step in data.manifest["training"]["checkpoint_steps"]
        )
        for step_index in range(start_step, TOTAL_STEPS):
            step = step_index + 1
            lr = learning_rate(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = data.train_batch(step_index)
            model.train()
            optimizer.zero_grad()
            loss_sum = torch.zeros((), device=device)
            for micro_start in range(0, EFFECTIVE_BATCH_RECORDS, arm.micro_batch_size):
                micro_stop = micro_start + arm.micro_batch_size
                values = device_tensor(
                    batch.tokens[micro_start:micro_stop].astype(np.int64, copy=False),
                    device,
                )
                mask = device_tensor(batch.target_mask[micro_start:micro_stop], device)
                inputs, labels = values[:, :-1], values[:, 1:]
                logits = model(inputs, attn_impl="sdpa")
                losses = F.cross_entropy(
                    logits.float().flatten(0, 1), labels.flatten(), reduction="none"
                ).view_as(labels)
                selected_sum = losses[mask].sum()
                (selected_sum / batch.target_count).backward()
                loss_sum += selected_sum.detach()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP, error_if_nonfinite=True
            )
            optimizer.step()
            interval_targets += batch.target_count

            should_heartbeat = (
                step % HEARTBEAT_INTERVAL == 0 or step in checkpoint_steps
            )
            if should_heartbeat:
                torch.cuda.synchronize(device)
                now = time.perf_counter()
                row = {
                    "step": step,
                    "train_nll": float(loss_sum / batch.target_count),
                    "learning_rate": lr,
                    "gradient_norm": float(gradient_norm),
                    "target_presentations": data.target_presentations_through_step(
                        step
                    ),
                    "interval_targets_per_second": interval_targets
                    / max(now - interval_started, 1e-12),
                }
                curve.append(row)
                metric_handle.write(json.dumps(row, sort_keys=True) + "\n")
                metric_handle.flush()
                heartbeat = {
                    "marker": "HEARTBEAT",
                    "arm": arm.identifier,
                    "steps": TOTAL_STEPS,
                    "eta_seconds": (now - started)
                    / max(step - start_step, 1)
                    * (TOTAL_STEPS - step),
                    **row,
                }
                atomic_json(output / "heartbeat.json", heartbeat)
                print(json.dumps(heartbeat, sort_keys=True), flush=True)
                interval_started = time.perf_counter()
                interval_targets = 0

            if step in checkpoint_steps:
                evaluation = evaluate(model, data, "validation", device)
                record = {
                    "step": step,
                    "completed_passes": step / OPTIMIZER_STEPS_PER_PASS,
                    "validation": evaluation,
                }
                evaluations.append(record)
                atomic_json(output / f"validation-step-{step:05d}.json", record)
                print(
                    json.dumps({"marker": "EVAL", **record}, sort_keys=True), flush=True
                )
                interval_started = time.perf_counter()

            if step % RECOVERY_INTERVAL == 0 and step != TOTAL_STEPS:
                recovery = save_recovery(
                    output,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    manifest_sha256=manifest_sha256,
                    step=step,
                    evaluations=evaluations,
                    curve=curve,
                )
                print(
                    json.dumps(
                        {
                            "marker": "RECOVERY",
                            "step": step,
                            "sha256": sha256_file(recovery),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                interval_started = time.perf_counter()

    complete_passes = {
        round(float(record["completed_passes"])): record["validation"]
        for record in evaluations
        if float(record["completed_passes"]).is_integer()
    }
    if sorted(complete_passes) != [1, 2, 3]:
        raise ValueError("full validation after each pass is absent")
    terminal_test = evaluate(model, data, "test", device)
    terminal_checkpoint = output / "terminal-model.pt"
    atomic_torch_save(
        terminal_checkpoint,
        {
            "schema": "residualized-routing-terminal-model-v1",
            "arm": arm.identifier,
            "config": asdict(config),
            "manifest_sha256": manifest_sha256,
            "model": model.state_dict(),
        },
    )
    terminal_checkpoint_sha256 = sha256_file(terminal_checkpoint)
    coverage = data.coverage_report()
    coverage.update(
        {
            "consumed_optimizer_steps": TOTAL_STEPS,
            "consumed_target_presentations": data.target_presentations_through_step(
                TOTAL_STEPS
            ),
            "complete": True,
            "all_passes_exactly_once_without_replacement": True,
        }
    )
    atomic_json(output / "coverage.json", coverage)
    result = {
        "schema": "residualized-routing-wikitext103-result-v1",
        "status": "terminal_verified",
        "campaign_id": CAMPAIGN_ID,
        "arm": arm.identifier,
        "config": asdict(config),
        "manifest_sha256": manifest_sha256,
        "payload_sha256": EXPECTED_PAYLOAD_SHA256,
        "common_parameter_sha256": common_sha256,
        "trainable_parameters": trainable_parameters,
        "validation_by_pass": {
            str(index): complete_passes[index] for index in (1, 2, 3)
        },
        "terminal_validation": complete_passes[3],
        "terminal_test": terminal_test,
        "coverage": coverage,
        "terminal_checkpoint": {
            "path": terminal_checkpoint.name,
            "sha256": terminal_checkpoint_sha256,
            "bytes": terminal_checkpoint.stat().st_size,
        },
        "total_seconds": time.perf_counter() - started,
        "peak_allocated_device_memory_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
    }
    atomic_json(output / "result.json", result)
    print(
        json.dumps(
            {
                "marker": "COMPLETE",
                "arm": arm.identifier,
                "validation_nll": result["terminal_validation"]["nll"],
                "test_nll": result["terminal_test"]["nll"],
                "checkpoint_sha256": terminal_checkpoint_sha256,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    args = parser.parse_args()
    if args.resume_sha256 is not None and (
        len(args.resume_sha256) != 64
        or any(character not in "0123456789abcdef" for character in args.resume_sha256)
    ):
        parser.error("--resume-sha256 must be a lowercase SHA-256 digest")
    return args


if __name__ == "__main__":
    train(parse_args())
