#!/usr/bin/env python3
"""Verify fresh three-arm BabyLM results and reconstruct their exact checkpoints."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
import sys
ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "third_party/babylm_runtime"))

from reproduction.babylm.checkpoint import reconstruct
from reproduction.babylm.data import BabyLMData
from reproduction.babylm.io import atomic_json, sha256_file
from reproduction.babylm.metrics import terminal_metrics, primary_metric
from reproduction.babylm.spec import (PUBLIC_ARMS, CAMPAIGN_ID, SPEC, MANIFEST_SHA256,
                                      CHECKPOINT_EXPOSURES_MILLIONS, EVALUATOR_REVISION)
from shared_residual_routing.results import FINETUNE_TASKS

# Declared tolerances in NLL / absolute score fractions, before any reproduction run.
NLL_BANDS = {"a16_dense_attention": (2.60, 2.68), "b16_native_sdm": (2.69, 2.78),
             "b16_residualized_sdm": (2.60, 2.69)}
ZERO_BANDS = {"a16_dense_attention": (.44, .51), "b16_native_sdm": (.45, .53),
              "b16_residualized_sdm": (.46, .54)}
FINE_BANDS = {"a16_dense_attention": (.58, .65), "b16_native_sdm": (.57, .65),
              "b16_residualized_sdm": (.61, .69)}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def bound_file(root: Path, record: dict) -> Path:
    path = (root / record["path"]).resolve()
    if root.resolve() not in path.parents or sha256_file(path) != record["sha256"]:
        raise ValueError("checkpoint or result file identity changed")
    return path


def validate_training(result: dict, config: dict, curve: list[dict], arm: str) -> float:
    if result.get("status") != "complete" or result.get("arm") != PUBLIC_ARMS[arm]:
        raise ValueError("BabyLM training result is not complete")
    if result.get("steps") != 102852 or result.get("training_tokens") != 1685114880:
        raise ValueError("BabyLM training coverage is incomplete")
    if config.get("spec") != SPEC.as_dict() or config.get("manifest_sha256") != MANIFEST_SHA256:
        raise ValueError("BabyLM training configuration changed")
    if config.get("mode") != "experiment" or (config.get("reads"), config.get("writes")) != (32, 32):
        raise ValueError("BabyLM smoke or changed access profile cannot validate reproduction")
    if [row["step"] for row in curve] != list(range(1, 102853)):
        raise ValueError("BabyLM complete fresh training curve is missing")
    if curve[-1]["training_tokens"] != 1685114880:
        raise ValueError("BabyLM curve exposure changed")
    if not all(math.isfinite(row["train_nll"]) for row in curve):
        raise ValueError("BabyLM curve contains invalid NLL")
    nll = sum(row["train_nll"] for row in curve[-1000:]) / 1000
    if not NLL_BANDS[arm][0] <= nll <= NLL_BANDS[arm][1]:
        raise ValueError(f"{arm} training NLL is outside its predeclared band")
    return nll


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    data = BabyLMData(args.manifest, args.manifest.parent)
    rows = []
    for arm, internal in PUBLIC_ARMS.items():
        root = (args.run_root / CAMPAIGN_ID / internal).resolve()
        result = read(root / "RESULT.json")
        config = read(root / "CONFIG.json")
        curve = [json.loads(line) for line in (root / "training_curve.jsonl").read_text().splitlines()]
        nll = validate_training(result, config, curve, arm)
        checkpoint = result["terminal_checkpoint"]
        checkpoint_path = bound_file(root, checkpoint)
        reconstruct(checkpoint_path, checkpoint["sha256"], internal)
        evaluation_root = root / "official-evaluation"
        core = read(evaluation_root / "RESULT.json")
        ewok = read(evaluation_root / "full-ewok/RESULT.json")
        if core.get("status") != "complete_terminal_core" or ewok.get("status") != "complete_full_ewok":
            raise ValueError("BabyLM terminal evaluation is incomplete")
        for evaluation in (core, ewok):
            if evaluation.get("checkpoint_sha256") != checkpoint["sha256"] or evaluation.get("arm") != internal:
                raise ValueError("BabyLM scores are not bound to the reconstructed checkpoint")
        for task in FINETUNE_TASKS:
            fine = core["finetune"][task]
            epochs = 30 if task == "wsc" else 10
            if fine["epochs"] != epochs or len(fine["epoch_metrics"]) != epochs:
                raise ValueError(f"{task} fine-tuning coverage is incomplete")
            bound_file(evaluation_root, {"path": f"finetune_{task}_predictions.npy", "sha256": fine["predictions_sha256"]})
            scores = [row["validation"][primary_metric(task)] for row in fine["epoch_metrics"]]
            if fine["best_validation"][primary_metric(task)] != max(scores):
                raise ValueError(f"{task} best validation metric changed")
        for scoring, directory in ((core["zero_shot"], evaluation_root), (ewok["scoring"], evaluation_root / "full-ewok")):
            for name, field in (("zero_shot_predictions.npy", "predictions_sha256"),
                                ("zero_shot_candidate_scores.npy", "candidate_scores_sha256")):
                path = bound_file(directory, {"path": name, "sha256": scoring[field]})
                if not np.isfinite(np.load(path, allow_pickle=False)).all():
                    raise ValueError("nonfinite BabyLM predictions")
        human = read(root / "human-likeness.json")
        if human.get("status") != "complete" or human.get("official_evaluator_revision") != EVALUATOR_REVISION:
            raise ValueError("BabyLM Reading/AoA evaluation is incomplete")
        if set(human.get("checkpoints", {})) != {str(value) for value in CHECKPOINT_EXPOSURES_MILLIONS}:
            raise ValueError("BabyLM human-likeness checkpoint matrix is incomplete")
        for exposure in CHECKPOINT_EXPOSURES_MILLIONS:
            retained = read(root / f"checkpoints/official-{exposure:04d}M/CHECKPOINT.json")
            bound_file(root, retained)
            measured = read(root / f"official-checkpoint-evaluation/official-{exposure:04d}M.json")
            if measured["checkpoint_sha256"] != retained["sha256"] or human["checkpoints"][str(exposure)]["checkpoint_sha256"] != retained["sha256"]:
                raise ValueError("BabyLM checkpoint-curve evidence belongs to other weights")
        metrics = terminal_metrics(core, ewok)
        for field, bands in (("zero_shot_macro", ZERO_BANDS), ("finetune_macro_primary_metric", FINE_BANDS)):
            if not bands[arm][0] <= metrics[field] <= bands[arm][1]:
                raise ValueError(f"{arm} {field} is outside its predeclared band")
        rows.append({"arm": arm, "training_nll": nll, **metrics,
                     "checkpoint_sha256": checkpoint["sha256"], "checkpoint_roundtrip_inference_passed": True,
                     "result_sha256": sha256_file(root / "RESULT.json"),
                     "evaluation_sha256": sha256_file(evaluation_root / "RESULT.json"),
                     "full_ewok_sha256": sha256_file(evaluation_root / "full-ewok/RESULT.json"),
                     "human_likeness_sha256": sha256_file(root / "human-likeness.json")})
    dense, native, residual = rows
    gap = native["training_nll"] - dense["training_nll"]
    if gap <= 0 or not .70 <= (native["training_nll"] - residual["training_nll"]) / gap <= 1.2:
        raise ValueError("BabyLM routing NLL improvement did not reproduce")
    if residual["finetune_macro_primary_metric"] <= max(dense["finetune_macro_primary_metric"], native["finetune_macro_primary_metric"]):
        raise ValueError("BabyLM fine-tuning improvement did not reproduce")
    atomic_json(args.run_root / "measurements.json", {"status": "reproduced", "manifest_sha256": MANIFEST_SHA256,
                "arms": rows, "terminal_checkpoints_roundtrip_inference_passed": list(PUBLIC_ARMS)})
    print("fresh three-arm BabyLM results and checkpoint inference verified")


if __name__ == "__main__":
    main()
