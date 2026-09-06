"""Train one matched BabyLM arm and durably retain its terminal weights."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from .data import BabyLMData
from .io import atomic_json, atomic_torch_save, sha256_file
from .model import build_model, model_contract
from .spec import CAMPAIGN_ID, CANONICAL_SDM_COMMIT, MANIFEST_SHA256, PROTOCOL_ID, SPEC, validate_arm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def learning_rate(step: int, total_steps: int) -> float:
    if step <= SPEC.warmup_steps:
        return SPEC.learning_rate * step / SPEC.warmup_steps
    progress = (step - SPEC.warmup_steps) / max(1, total_steps - SPEC.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return SPEC.learning_rate * (
        SPEC.minimum_learning_rate_ratio
        + (1.0 - SPEC.minimum_learning_rate_ratio) * cosine
    )


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def optimizer_groups(model: torch.nn.Module) -> list[dict[str, Any]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if getattr(parameter, "_no_weight_decay", False) else decay).append(
            parameter
        )
    groups: list[dict[str, Any]] = [
        {"params": decay, "weight_decay": SPEC.weight_decay}
    ]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def _recovery_identity(arm: str, manifest_sha256: str, total_steps: int) -> dict[str, Any]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "arm": arm,
        "manifest_sha256": manifest_sha256,
        "canonical_sdm_commit": CANONICAL_SDM_COMMIT,
        "spec": SPEC.as_dict(),
        "total_steps": total_steps,
    }


def save_recovery(
    path: Path,
    *,
    arm: str,
    manifest_sha256: str,
    total_steps: int,
    step: int,
    training_tokens: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "kind": "babylm_training_recovery",
        **_recovery_identity(arm, manifest_sha256, total_steps),
        "step": step,
        "training_tokens": training_tokens,
        "model": cpu_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all(),
    }
    atomic_torch_save(path, payload)
    row = {
        "path": path.name,
        "step": step,
        "training_tokens": training_tokens,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    atomic_json(path.with_suffix(".json"), row)
    return row


def load_recovery(
    path: Path,
    *,
    arm: str,
    manifest_sha256: str,
    total_steps: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int] | None:
    record_path = path.with_suffix(".json")
    if not path.exists() and not record_path.exists():
        return None
    if not path.is_file() or not record_path.is_file():
        raise RuntimeError("partial durable recovery checkpoint")
    record = json.loads(record_path.read_text())
    digest = sha256_file(path)
    if digest != record["sha256"]:
        raise RuntimeError("durable recovery checkpoint hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = _recovery_identity(arm, manifest_sha256, total_steps)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"recovery identity changed at {key}")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng_state"])
    torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
    return int(payload["step"]), int(payload["training_tokens"])


def save_inference_checkpoint(
    output: Path,
    *,
    arm: str,
    model: torch.nn.Module,
    contract: dict[str, Any],
    manifest_sha256: str,
    manifest: dict[str, Any],
    step: int,
    official_exposure_millions: int,
    training_tokens: int,
) -> dict[str, Any]:
    checkpoint_dir = output / "checkpoints" / f"official-{official_exposure_millions:04d}M"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_dir / "checkpoint.pending.pt"
    atomic_torch_save(
        temporary,
        {
            "schema_version": 1,
            "kind": "canonical_babylm_inference_checkpoint",
            "campaign_id": CAMPAIGN_ID,
            "arm": arm,
            "step": step,
            "official_exposure_millions": official_exposure_millions,
            "training_tokens": training_tokens,
            "model": cpu_state_dict(model),
            "model_contract": contract,
            "model_spec": SPEC.as_dict(),
            "tokenizer": manifest["tokenizer"],
            "dataset_manifest_sha256": manifest_sha256,
            "canonical_sdm_commit": CANONICAL_SDM_COMMIT,
        },
    )
    digest = sha256_file(temporary)
    destination = checkpoint_dir / f"{digest}.pt"
    os.replace(temporary, destination)
    if sha256_file(destination) != digest:
        raise RuntimeError("terminal checkpoint failed post-write verification")
    payload = torch.load(destination, map_location="cpu", weights_only=False)
    if payload.get("kind") != "canonical_babylm_inference_checkpoint":
        raise RuntimeError("inference checkpoint cannot be reopened")
    record = {
        "schema_version": 1,
        "status": "verified",
        "campaign_id": CAMPAIGN_ID,
        "arm": arm,
        "step": step,
        "official_exposure_millions": official_exposure_millions,
        "training_tokens": training_tokens,
        "path": str(destination.relative_to(output)),
        "sha256": digest,
        "bytes": destination.stat().st_size,
        "babylm_evaluator_ready": True,
        "canonical_sdm_commit": CANONICAL_SDM_COMMIT,
        "dataset_manifest_sha256": manifest_sha256,
    }
    atomic_json(checkpoint_dir / "CHECKPOINT.json", record)
    return record


def train(args: argparse.Namespace) -> None:
    if (args.reads, args.writes) != (SPEC.reads, SPEC.writes) or args.writes != args.reads:
        raise ValueError("BabyLM requires explicit balanced R=W=32")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    validate_arm(args.arm)
    smoke = args.smoke_steps is not None
    if smoke and not 1 <= args.smoke_steps <= 3:
        raise ValueError("smoke mode is limited to 1–3 optimizer steps")
    if args.output_root.resolve() == args.immutable_input_root.resolve():
        raise ValueError("mutable output cannot be the immutable input volume")
    if args.immutable_input_root.resolve() in args.output_root.resolve().parents:
        raise ValueError("mutable output cannot live under immutable input")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Keep deterministic implementations where PyTorch provides them, while
    # permitting SDM's CUDA product reduction, which PyTorch currently marks
    # as nondeterministic rather than providing a deterministic alternative.
    torch.use_deterministic_algorithms(True, warn_only=True)
    set_seed(SPEC.seed)
    data = BabyLMData(args.manifest, args.immutable_input_root)
    checkpoint_schedule = {
        int(row["optimizer_step"]): int(row["official_exposure_millions"])
        for row in data.manifest["corpus"]["training"]["checkpoint_exposures"]
    }
    if len(checkpoint_schedule) != len(
        data.manifest["corpus"]["training"]["checkpoint_exposures"]
    ):
        raise ValueError("official checkpoint schedule contains duplicate steps")
    manifest_sha256 = sha256_file(args.manifest)
    total_steps = data.training_steps()
    output = args.output_root.resolve() / CAMPAIGN_ID / args.arm
    output.mkdir(parents=True, exist_ok=True)
    if smoke and any(output.iterdir()):
        raise FileExistsError("smoke output must be empty")
    if (output / "RESULT.json").exists():
        raise FileExistsError("completed BabyLM results are never overwritten")
    if (output / "metrics.jsonl").exists() and not (output / "recovery_checkpoint.json").exists():
        raise FileExistsError("partial BabyLM output has no verified recovery")

    model = build_model(args.arm).to(device="cuda", dtype=torch.bfloat16)
    contract = model_contract(args.arm, model)
    optimizer = torch.optim.AdamW(
        optimizer_groups(model),
        lr=SPEC.learning_rate,
        betas=(0.9, 0.95),
        fused=True,
    )
    recovery_path = output / "recovery_checkpoint.pt"
    resumed = load_recovery(
        recovery_path,
        arm=args.arm,
        manifest_sha256=manifest_sha256,
        total_steps=total_steps,
        model=model,
        optimizer=optimizer,
    )
    start_step, training_tokens = resumed or (0, 0)
    compiled_model = model if smoke else torch.compile(model, dynamic=False)
    metrics_path = output / "metrics.jsonl"
    mode = "a" if start_step else "w"
    curve_path = output / "training_curve.jsonl"
    if start_step:
        for path in (metrics_path, curve_path):
            retained = [line for line in path.read_text().splitlines()
                        if json.loads(line).get("step", 0) <= start_step]
            path.write_text("\n".join(retained) + ("\n" if retained else ""))
    recent_losses: deque[float] = deque(maxlen=SPEC.heartbeat_steps)
    if start_step:
        for line in curve_path.read_text().splitlines()[-SPEC.heartbeat_steps:]:
            recent_losses.append(float(json.loads(line)["train_nll"]))
    started = time.perf_counter()
    interval_started = started
    interval_tokens = 0
    torch.cuda.reset_peak_memory_stats()

    started_row = {
        "marker": "STARTED",
        "campaign_id": CAMPAIGN_ID,
        "arm": args.arm,
        "resumed_from_step": start_step,
        "steps": total_steps,
        "training_tokens": data.training_tokens(),
        "model_contract": contract,
        "manifest_sha256": manifest_sha256,
        "canonical_sdm_commit": CANONICAL_SDM_COMMIT,
        "deterministic_algorithms": "enabled_warn_on_unsupported",
        "device": torch.cuda.get_device_name(),
        "hardware": {"accelerator": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                     "pytorch": torch.__version__},
        "protocol_id": PROTOCOL_ID,
        "evidence_tier": "babylm_scale_comparison",
        "prerequisite": "completed_seed0_wikitext_and_adaptive_recall_in_provenance.json",
        "reads": SPEC.reads,
        "writes": SPEC.writes,
        "mode": "smoke" if smoke else "experiment",
    }
    atomic_json(output / "CONFIG.json", {**started_row, "spec": SPEC.as_dict()})
    print(json.dumps(started_row, sort_keys=True), flush=True)

    with metrics_path.open(mode, encoding="utf-8") as metrics, curve_path.open(mode, encoding="utf-8") as curve:
        for step_index in range(start_step, args.smoke_steps if smoke else total_steps):
            step = step_index + 1
            lr = learning_rate(step, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = data.train_batch(step_index, SPEC.batch_size)
            values = torch.from_numpy(batch.astype(np.int64, copy=False)).cuda()
            inputs, targets = values[:, :-1], values[:, 1:]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = compiled_model(inputs, target=targets, attn_impl="sdpa")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), SPEC.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            batch_tokens = int(inputs.numel())
            training_tokens += batch_tokens
            interval_tokens += batch_tokens
            recent_losses.append(float(loss.detach()))
            curve.write(json.dumps({"step": step, "train_nll": recent_losses[-1],
                                    "training_tokens": training_tokens, "learning_rate": lr}) + "\n")

            if step % SPEC.heartbeat_steps == 0 or step == total_steps:
                curve.flush()
                torch.cuda.synchronize()
                now = time.perf_counter()
                elapsed = now - started
                completed = step - start_step
                heartbeat = {
                    "marker": "HEARTBEAT",
                    "campaign_id": CAMPAIGN_ID,
                    "arm": args.arm,
                    "step": step,
                    "steps": total_steps,
                    "mean_train_loss": sum(recent_losses) / len(recent_losses),
                    "latest_train_loss": recent_losses[-1],
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": lr,
                    "interval_tokens_per_second": interval_tokens
                    / max(now - interval_started, 1e-9),
                    "elapsed_seconds": elapsed,
                    "eta_seconds": elapsed / max(completed, 1) * (total_steps - step),
                    "training_tokens": training_tokens,
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                }
                metrics.write(json.dumps(heartbeat, sort_keys=True) + "\n")
                metrics.flush()
                os.fsync(metrics.fileno())
                atomic_json(output / "HEARTBEAT.json", heartbeat)
                print(json.dumps(heartbeat, sort_keys=True), flush=True)
                interval_started = now
                interval_tokens = 0

            if step in checkpoint_schedule:
                checkpoint = save_inference_checkpoint(
                    output,
                    arm=args.arm,
                    model=model,
                    contract=contract,
                    manifest_sha256=manifest_sha256,
                    manifest=data.manifest,
                    step=step,
                    official_exposure_millions=checkpoint_schedule[step],
                    training_tokens=training_tokens,
                )
                row = {"marker": "INFERENCE_CHECKPOINT", **checkpoint}
                metrics.write(json.dumps(row, sort_keys=True) + "\n")
                metrics.flush()
                os.fsync(metrics.fileno())
                print(json.dumps(row, sort_keys=True), flush=True)
                interval_started = time.perf_counter()

            if step % SPEC.recovery_steps == 0 or step == total_steps:
                recovery = save_recovery(
                    recovery_path,
                    arm=args.arm,
                    manifest_sha256=manifest_sha256,
                    total_steps=total_steps,
                    step=step,
                    training_tokens=training_tokens,
                    model=model,
                    optimizer=optimizer,
                )
                print(
                    json.dumps({"marker": "RECOVERY", "arm": args.arm, **recovery}),
                    flush=True,
                )
                interval_started = time.perf_counter()

    if smoke:
        path = output / "smoke-model.pt"
        atomic_torch_save(path, {"kind": "babylm_smoke_checkpoint", "model": cpu_state_dict(model),
                                "arm": args.arm, "model_spec": SPEC.as_dict(), "step": args.smoke_steps,
                                "dataset_manifest_sha256": manifest_sha256})
        from .checkpoint import reconstruct
        model.eval()
        probe = torch.arange(32, device="cuda").view(1, -1)
        with torch.inference_mode():
            reference = model(probe, attn_impl="sdpa")
        digest = sha256_file(path)
        del compiled_model, model, optimizer
        reconstruct(path, digest, args.arm, smoke=True, expected=reference)
        atomic_json(output / "smoke.json", {"status": "smoke_passed", "steps": args.smoke_steps,
                                            "checkpoint_sha256": digest, "checkpoint_roundtrip_inference_passed": True})
        print(json.dumps({"marker": "SMOKE_COMPLETE", "arm": args.arm}), flush=True)
        return

    terminal_checkpoint = output / "checkpoints" / "official-1000M" / "CHECKPOINT.json"
    if not terminal_checkpoint.is_file():
        raise RuntimeError("terminal official-exposure checkpoint was not retained")
    terminal = json.loads(terminal_checkpoint.read_text())
    terminal_path = output / terminal["path"]
    if sha256_file(terminal_path) != terminal["sha256"]:
        raise RuntimeError("terminal checkpoint failed final hash verification")
    from .checkpoint import reconstruct
    del compiled_model, model, optimizer
    reconstruct(terminal_path, terminal["sha256"], args.arm)
    atomic_json(output / "TERMINAL_CHECKPOINT.json", terminal)
    result = {
        "schema_version": 1,
        "status": "complete",
        "campaign_id": CAMPAIGN_ID,
        "arm": args.arm,
        "steps": total_steps,
        "training_tokens": training_tokens,
        "terminal_checkpoint": terminal,
        "official_evaluation_status": "pending",
        "model_contract": contract,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "device": torch.cuda.get_device_name(),
        "terminal_train_nll": sum(recent_losses) / len(recent_losses),
        "checkpoint_roundtrip_inference_passed": True,
    }
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"marker": "COMPLETE", **result}, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--immutable-input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--reads", type=int, default=32)
    parser.add_argument("--writes", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
