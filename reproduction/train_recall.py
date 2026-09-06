"""Train and evaluate one frozen Adaptive Recall arm; smoke mode is bounded."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from reproduction.recall_model import build_adaptive_recall_model, OUTPUT_CLASSES, QUERIES
from reproduction.recall_spec import (
    ARMS, GEOMETRY, MANIFEST_SHA256, STEPS, BATCH_SIZE, EVAL_BATCH_SIZE, PARAMETERS, config,
)
from reproduction.train_wikitext import atomic_json, atomic_torch_save, set_seed
from reproduction.wikitext103_coverage import sha256_file
from shared_residual_routing.router import install_residualized_routing
from scripts.prepare_recall import check


def build_model(arm: str):
    model = build_adaptive_recall_model(geometry=GEOMETRY, profile=ARMS[arm].profile, seed=0)
    if ARMS[arm].router == "residualized_read_write":
        install_residualized_routing(model)
    observed = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if observed != PARAMETERS[arm]:
        raise ValueError(f"{arm} parameter count changed: {observed}")
    return model


def tensor(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(values, dtype=np.int64)).pin_memory().cuda(non_blocking=True)


def summarize(rows: list[dict]) -> dict:
    examples = sum(row["examples"] for row in rows)
    queries = examples * QUERIES
    return {
        "examples": examples, "queries": queries,
        "query_loss": sum(row["loss_sum"] for row in rows) / queries,
        "query_accuracy": sum(row["correct_queries"] for row in rows) / queries,
        "exact_set_accuracy": sum(row["exact_sets"] for row in rows) / examples,
    }


@torch.inference_mode()
def evaluate(model, data, split: str, output: Path, *, smoke: bool = False) -> dict:
    model.eval()
    rows = []
    for condition in data.conditions:
        inputs, labels = data.evaluation(split, condition["id"])
        if smoke:
            inputs, labels = inputs[:1], labels[:1]
        predictions = np.empty(labels.shape, dtype=np.uint8)
        row = {"condition_id": condition["id"], "family": condition["family"],
               "examples": len(labels), "loss_sum": 0.0, "correct_queries": 0, "exact_sets": 0}
        for start in range(0, len(labels), EVAL_BATCH_SIZE):
            x, y = tensor(inputs[start:start + EVAL_BATCH_SIZE]), tensor(labels[start:start + EVAL_BATCH_SIZE])
            logits = model(x, attn_impl="sdpa")[:, -QUERIES:].float()
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite Recall evaluation logits")
            row["loss_sum"] += float(F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum"))
            predicted = logits.argmax(-1)
            correct = predicted.eq(y)
            row["correct_queries"] += int(correct.sum())
            row["exact_sets"] += int(correct.all(-1).sum())
            predictions[start:start + len(y)] = predicted.cpu().numpy()
        destination = output / "predictions" / split / f"{condition['id']}.npy"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, predictions, allow_pickle=False)
        row["predictions"] = {"path": str(destination.relative_to(output)), "sha256": sha256_file(destination)}
        rows.append(row)
    return {"aggregate": summarize(rows), "by_condition": rows,
            "by_family": {family: summarize([row for row in rows if row["family"] == family])
                          for family in sorted({row["family"] for row in rows})}}


@torch.inference_mode()
def reload_checkpoint(path: Path, expected_sha256: str, arm: str, *, inputs=None, expected=None):
    if sha256_file(path) != expected_sha256:
        raise ValueError("Recall checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("config") != config(arm) or payload.get("manifest_sha256") != MANIFEST_SHA256:
        raise ValueError("Recall checkpoint configuration or input identity changed")
    if payload.get("schema") != "residualized-routing-recall-checkpoint-v1":
        raise ValueError("unexpected Recall checkpoint schema")
    model = build_model(arm)
    model.load_state_dict(payload["model"], strict=True)
    model = model.cuda().bfloat16().eval()
    if inputs is None:
        inputs = torch.arange(32, device="cuda").view(1, -1)
    logits = model(inputs, attn_impl="sdpa")[:, -QUERIES:].float()
    if logits.shape != (len(inputs), QUERIES, OUTPUT_CLASSES) or not torch.isfinite(logits).all():
        raise ValueError("Recall checkpoint inference failed")
    if expected is not None:
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    return payload, model


def train(args) -> None:
    if (args.reads, args.writes) != (GEOMETRY.reads, GEOMETRY.writes) or args.writes != args.reads:
        raise ValueError("Recall requires explicit balanced R=W=8")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; no training was started")
    if args.smoke_steps is not None and not 1 <= args.smoke_steps <= 3:
        raise ValueError("smoke mode is limited to 1–3 optimizer steps")
    data = check(args.manifest)
    output = args.output.resolve()
    if output == data.root or data.root in output.parents:
        raise ValueError("output must be outside immutable inputs")
    output.mkdir(parents=True, exist_ok=True)
    smoke = args.smoke_steps is not None
    if smoke and any(output.iterdir()):
        raise FileExistsError("smoke output must be empty")
    if (output / "result.json").exists():
        raise FileExistsError("completed results are never overwritten")
    set_seed(0)
    model = build_model(args.arm).cuda().bfloat16()
    from lingua.optim import OptimArgs, build_optimizer
    optimizer, _ = build_optimizer(model, OptimArgs(lr=3e-4, weight_decay=.01,
                                      beta1=.9, beta2=.95, scheduler="constant", warmup=100), n_steps=STEPS)
    start_step = 0
    latest = output / "LATEST_RECOVERY.json"
    if latest.exists():
        record = json.loads(latest.read_text())
        path = (output / record["path"]).resolve()
        if output not in path.parents or sha256_file(path) != record["sha256"]:
            raise ValueError("recovery path or hash changed")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("config") != config(args.arm) or payload.get("manifest_sha256") != MANIFEST_SHA256:
            raise ValueError("recovery identity changed")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["rng_cpu"])
        torch.cuda.set_rng_state_all(payload["rng_cuda"])
        start_step = payload["step"]
        if not 0 <= start_step <= STEPS:
            raise ValueError("invalid recovery step")
        # Discard only measurements newer than the durable recovery state.
        curve = output / "training_curve.jsonl"
        retained = [line for line in curve.read_text().splitlines() if json.loads(line)["step"] <= start_step]
        curve.write_text("\n".join(retained) + ("\n" if retained else ""))
    elif (output / "training_curve.jsonl").exists():
        raise FileExistsError("partial output has no verified recovery checkpoint")
    hardware = {"accelerator": torch.cuda.get_device_name(), "cuda": torch.version.cuda, "pytorch": torch.__version__}
    atomic_json(output / "config.json", {"config": config(args.arm), "hardware": hardware,
                                           "mode": "smoke" if smoke else "experiment"})
    print(json.dumps({"marker": "STARTED", "arm": args.arm, "mode": "smoke" if smoke else "experiment"}), flush=True)
    started = time.perf_counter()
    end = args.smoke_steps if smoke else STEPS
    with (output / "training_curve.jsonl").open("a") as curve:
        for step_index in range(start_step, end):
            step = step_index + 1
            batch, labels, _ = data.train_batch(step_index)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            lr = 3e-4 * min(step / 100, 1)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss_total = 0.0
            # Historical effective batch and weighting are fixed; microbatches only bound workspace.
            micro = 8 if args.arm == "dense_attention" else 1
            for first in range(0, BATCH_SIZE, micro):
                x, y = tensor(batch[first:first + micro]), tensor(labels[first:first + micro])
                logits = model(x, attn_impl="sdpa")[:, -QUERIES:].float()
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                (loss * micro / BATCH_SIZE).backward()
                loss_total += float(loss.detach()) * micro / BATCH_SIZE
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            row = {"step": step, "train_query_loss": loss_total, "learning_rate": lr, "gradient_norm": float(grad)}
            curve.write(json.dumps(row) + "\n")
            if step % 100 == 0 or step == end:
                curve.flush()
                elapsed = time.perf_counter() - started
                heartbeat = {"marker": "HEARTBEAT", **row, "elapsed_seconds": elapsed,
                             "eta_seconds": elapsed / (step - start_step) * (end - step)}
                atomic_json(output / "heartbeat.json", heartbeat)
                print(json.dumps(heartbeat), flush=True)
            if not smoke and (step % 1000 == 0 or step == end):
                path = output / "checkpoints" / f"recovery-{step}.pt"
                atomic_torch_save(path, {"config": config(args.arm), "manifest_sha256": MANIFEST_SHA256,
                                        "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                        "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()})
                atomic_json(latest, {"path": str(path.relative_to(output)), "sha256": sha256_file(path)})
    model.eval()
    probe = tensor(data.train_batch(0)[0][:1])
    with torch.inference_mode():
        expected = model(probe, attn_impl="sdpa")[:, -QUERIES:].float()
    checkpoint = output / ("smoke-model.pt" if smoke else "terminal-model.pt")
    atomic_torch_save(checkpoint, {"schema": "residualized-routing-recall-checkpoint-v1",
                                   "config": config(args.arm), "manifest_sha256": MANIFEST_SHA256,
                                   "mode": "smoke" if smoke else "experiment", "step": end,
                                   "model": model.state_dict()})
    digest = sha256_file(checkpoint)
    del model, optimizer
    _, model = reload_checkpoint(checkpoint, digest, args.arm, inputs=probe, expected=expected)
    result = {"schema": "residualized-routing-recall-result-v1", "arm": args.arm,
              "status": "smoke_passed" if smoke else "terminal_verified", "config": config(args.arm),
              "manifest_sha256": MANIFEST_SHA256, "steps": end, "hardware": hardware,
              "terminal_checkpoint": {"path": checkpoint.name, "sha256": digest},
              "checkpoint_roundtrip_inference_passed": True,
              "validation": evaluate(model, data, "validation", output, smoke=smoke),
              "test": evaluate(model, data, "test", output, smoke=smoke)}
    atomic_json(output / ("smoke.json" if smoke else "result.json"), result)
    print(json.dumps({"marker": "SMOKE_COMPLETE" if smoke else "COMPLETE", "arm": args.arm}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--reads", type=int, default=8)
    parser.add_argument("--writes", type=int, default=8)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
