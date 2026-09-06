#!/usr/bin/env python3
"""Check fresh complete Recall measurements against their reconstructed checkpoints."""
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
    sys.path.insert(0, str(ROOT / "third_party/runtime"))

from reproduction.recall_spec import ARM_ORDER, MANIFEST_SHA256, STEPS, EVAL_EXAMPLES, EXACT_SET_BANDS, QUERY_LOSS_BANDS, config
from reproduction.recall_generate import CONDITIONS, QUERIES
from reproduction.train_recall import reload_checkpoint, summarize
from reproduction.train_wikitext import atomic_json
from reproduction.wikitext103_coverage import sha256_file
from scripts.prepare_recall import check


def validate_result(result: dict, arm: str) -> None:
    expected = {"schema": "residualized-routing-recall-result-v1", "status": "terminal_verified",
                "arm": arm, "config": config(arm), "manifest_sha256": MANIFEST_SHA256, "steps": STEPS}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"{arm} Recall result is incomplete or changed at {key}")
    for split in ("validation", "test"):
        rows = result[split]["by_condition"]
        if [row["condition_id"] for row in rows] != [row["id"] for row in CONDITIONS]:
            raise ValueError("Recall evaluation condition matrix changed")
        for row in rows:
            if row["examples"] != EVAL_EXAMPLES or not math.isfinite(row["loss_sum"]):
                raise ValueError("Recall evaluation coverage is incomplete")
        if summarize(rows) != result[split]["aggregate"]:
            raise ValueError("Recall result arithmetic changed")
    accuracy = result["test"]["aggregate"]["exact_set_accuracy"]
    loss = result["test"]["aggregate"]["query_loss"]
    if not EXACT_SET_BANDS[arm][0] <= accuracy <= EXACT_SET_BANDS[arm][1]:
        raise ValueError(f"{arm} exact-set accuracy is outside its declared band")
    if not QUERY_LOSS_BANDS[arm][0] <= loss <= QUERY_LOSS_BANDS[arm][1]:
        raise ValueError(f"{arm} query loss is outside its declared band")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    data = check(args.manifest)
    rows = []
    for arm in ARM_ORDER:
        root = (args.run_root / arm).resolve()
        result_path = root / "result.json"
        result = json.loads(result_path.read_text())
        validate_result(result, arm)
        curve = [json.loads(line) for line in (root / "training_curve.jsonl").read_text().splitlines()]
        if [row["step"] for row in curve] != list(range(1, STEPS + 1)):
            raise ValueError(f"{arm} training trajectory is incomplete")
        for split in ("validation", "test"):
            for row in result[split]["by_condition"]:
                record = row["predictions"]
                path = (root / record["path"]).resolve()
                if root not in path.parents or sha256_file(path) != record["sha256"]:
                    raise ValueError("Recall prediction identity changed")
                values = np.load(path, allow_pickle=False)
                _, labels = data.evaluation(split, row["condition_id"])
                if values.shape != (EVAL_EXAMPLES, QUERIES):
                    raise ValueError("Recall predictions have incomplete coverage")
                correct = values == labels
                if int(correct.sum()) != row["correct_queries"] or int(correct.all(-1).sum()) != row["exact_sets"]:
                    raise ValueError("Recall metrics do not match retained predictions")
        checkpoint = result["terminal_checkpoint"]
        path = (root / checkpoint["path"]).resolve()
        if root not in path.parents:
            raise ValueError("checkpoint escapes arm output")
        payload, model = reload_checkpoint(path, checkpoint["sha256"], arm)
        if payload.get("mode") != "experiment" or payload.get("step") != STEPS:
            raise ValueError("a smoke/recovery checkpoint cannot validate reproduction")
        del payload, model
        rows.append({"arm": arm, **result["test"]["aggregate"],
                     "checkpoint_sha256": checkpoint["sha256"],
                     "checkpoint_roundtrip_inference_passed": True,
                     "result_sha256": sha256_file(result_path)})
    accuracies = {row["arm"]: row["exact_set_accuracy"] for row in rows}
    native_gap = accuracies["dense_attention"] - accuracies["b8_native_sdm"]
    if native_gap <= 0:
        raise ValueError("all-SDM native retrieval gap did not reproduce")
    closure = (accuracies["b8_residualized_sdm"] - accuracies["b8_native_sdm"]) / native_gap
    if not .65 <= closure <= 1.2:
        raise ValueError("all-SDM exact retrieval improvement did not reproduce")
    atomic_json(args.run_root / "measurements.json", {"status": "reproduced", "manifest_sha256": MANIFEST_SHA256,
                "arms": rows, "terminal_checkpoints_roundtrip_inference_passed": list(ARM_ORDER),
                "b8_exact_set_gap_fraction_closed": closure})
    print("fresh five-arm Recall measurements and checkpoint inference verified")


if __name__ == "__main__":
    main()
