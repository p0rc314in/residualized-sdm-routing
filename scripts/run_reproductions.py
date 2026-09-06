#!/usr/bin/env python3
"""Run explicit Recall or BabyLM reproduction stages on local CUDA devices."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import sys
ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "third_party/runtime"))

from reproduction.spec import ARM_ORDER
from reproduction.babylm.spec import PUBLIC_ARMS, CAMPAIGN_ID, SPEC
from scripts.run_experiments import parse_gpus
from scripts.verify_vendor import validate_vendor, validate_babylm_vendor


def environment(benchmark: str, gpu: str | None = None) -> dict:
    env = os.environ.copy()
    runtime = ROOT / "third_party" / ("babylm_runtime" if benchmark == "babylm" else "runtime")
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(runtime)))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def arm_commands(benchmark: str, stage: str, arm: str, data: Path, output: Path,
                 *, smoke_steps: int = 1, evaluator: Path | None = None) -> list[list[str]]:
    prefix = [sys.executable, "-m"]
    if benchmark == "adaptive_recall":
        command = [*prefix, "reproduction.train_recall", "--arm", arm, "--manifest", str(data / "manifest.json"),
                   "--output", str(output / arm), "--reads", "8", "--writes", "8"]
        if stage == "smoke":
            command += ["--smoke-steps", str(smoke_steps)]
        return [command]
    internal = PUBLIC_ARMS[arm]
    shared = ["--arm", internal, "--output-root", str(output)]
    manifest = ["--manifest", str(data / "manifest.json"), "--immutable-input-root", str(data)]
    if stage in ("train", "smoke"):
        command = [*prefix, "reproduction.babylm.train", *shared, *manifest, "--reads", "32", "--writes", "32"]
        if stage == "smoke":
            command += ["--smoke-steps", str(smoke_steps)]
        return [command]
    if stage == "evaluate":
        return [[*prefix, "reproduction.babylm.evaluate", *shared, *manifest],
                [*prefix, "reproduction.babylm.evaluate_full_ewok", *shared,
                 "--training-manifest", str(data / "manifest.json"),
                 "--ewok-manifest", str(data / "full-ewok/manifest.json")]]
    if stage == "evaluate-checkpoints":
        from reproduction.babylm.io import sha256_file
        return [[*prefix, "reproduction.babylm.evaluate_checkpoints", *shared,
                 "--training-manifest", str(data / "manifest.json"), "--input-root", str(data / "checkpoint-evaluation"),
                 "--input-manifest-sha256", sha256_file(data / "checkpoint-evaluation/manifest.json")]]
    if stage == "score-human":
        if evaluator is None:
            raise ValueError("score-human requires --evaluator-root")
        official = data / "official-source/evaluation_data/full_eval"
        return [[*prefix, "reproduction.babylm.score_human_likeness", "--arm", internal,
                 "--inference-root", str(output / CAMPAIGN_ID / internal / "official-checkpoint-evaluation"),
                 "--packed-input-root", str(data / "checkpoint-evaluation"),
                 "--reading-data", str(official / "reading/reading_data.csv"),
                 "--cdi-human", str(official / "aoa/cdi_human.csv"),
                 "--official-evaluator-root", str(evaluator),
                 "--output", str(output / CAMPAIGN_ID / internal / "human-likeness.json")]]
    raise ValueError(f"unsupported stage {stage}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("adaptive_recall", "babylm"))
    parser.add_argument("stage", choices=("prepare", "train", "smoke", "evaluate", "evaluate-checkpoints", "score-human", "check", "all"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpus", default=os.environ.get("GPUS", "0"))
    parser.add_argument("--smoke-steps", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--evaluator-root", type=Path)
    args = parser.parse_args()
    benchmark = args.benchmark
    data = (args.data or ROOT / "runs/data" / ("adaptive_recall" if benchmark == "adaptive_recall" else "babylm")).resolve()
    mode = "smoke" if args.stage == "smoke" else "reproduction"
    output = (args.output or ROOT / "runs" / mode / benchmark).resolve()
    gpus = parse_gpus(args.gpus)
    stages = [args.stage]
    if args.stage == "all":
        stages = (["prepare", "train", "check"] if benchmark == "adaptive_recall" else
                  ["prepare", "train", "evaluate", "evaluate-checkpoints", "score-human", "check"])
    if benchmark == "babylm":
        validate_babylm_vendor(ROOT / "third_party/babylm_runtime")
        if (SPEC.reads, SPEC.writes) != (32, 32):
            raise ValueError("BabyLM requires explicit balanced R=W=32")
        provenance = json.loads((ROOT / "provenance.json").read_text())
        if not provenance.get("wikitext103_results") or not provenance.get("adaptive_recall_results"):
            raise ValueError("BabyLM requires completed smaller-tier evidence")
        if args.stage in ("all", "prepare", "score-human") and args.evaluator_root is None:
            parser.error("this BabyLM stage requires --evaluator-root at the pinned revision")
    else:
        validate_vendor(ROOT / "third_party/runtime")
        if args.stage in ("evaluate", "evaluate-checkpoints", "score-human"):
            parser.error("Recall evaluates its terminal model in the train stage")
    output.mkdir(parents=True, exist_ok=True)

    def run(command, gpu=None):
        subprocess.run(command, cwd=ROOT, env=environment(benchmark, gpu), check=True)

    for stage in stages:
        if stage == "prepare":
            if benchmark == "adaptive_recall":
                run([sys.executable, str(ROOT / "scripts/prepare_recall.py"), "--output", str(data)])
            else:
                for part in ("full-ewok", "core", "checkpoints", "training"):
                    run([sys.executable, "-m", "reproduction.babylm.prepare", "--output", str(data),
                         "--stage", part, "--evaluator-root", str(args.evaluator_root)])
            continue
        if stage == "check":
            script = "check_recall_reproduction.py" if benchmark == "adaptive_recall" else "check_babylm_reproduction.py"
            run([sys.executable, str(ROOT / "scripts" / script), str(output), "--manifest", str(data / "manifest.json")], gpus[0])
            continue
        arms = ARM_ORDER if benchmark == "adaptive_recall" else tuple(PUBLIC_ARMS)
        commands = {arm: arm_commands(benchmark, stage, arm, data, output, smoke_steps=args.smoke_steps,
                                     evaluator=args.evaluator_root) for arm in arms}
        def worker(gpu, assigned):
            for arm in assigned:
                log = output / f"{stage}-{arm}.log"
                print(json.dumps({"marker": "STARTED", "stage": stage, "arm": arm, "gpu": gpu}), flush=True)
                with log.open("a") as handle:
                    for command in commands[arm]:
                        subprocess.run(command, cwd=ROOT, env=environment(benchmark, gpu),
                                       stdout=handle, stderr=subprocess.STDOUT, check=True)
                print(json.dumps({"marker": "SMOKE_COMPLETE" if stage == "smoke" else "STAGE_COMPLETE",
                                  "stage": stage, "arm": arm}), flush=True)
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(worker, gpu, arms[index::len(gpus)]) for index, gpu in enumerate(gpus)]
            for future in futures:
                future.result()


if __name__ == "__main__":
    main()
