#!/usr/bin/env python3
"""Run the five small-model arms across one or more CUDA devices."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from reproduction.spec import ARM_ORDER


ROOT = Path(__file__).resolve().parents[1]


def parse_gpus(value: str) -> tuple[str, ...]:
    gpus = tuple(part.strip() for part in value.split(",") if part.strip())
    if not gpus or any(not gpu.isdigit() for gpu in gpus):
        raise ValueError("GPUS must be a comma-separated list of device indices")
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPUS must not contain duplicates")
    return gpus


def process_environment(gpu: str, extensions: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["SDM_SOURCE_ROOT"] = str(ROOT / "third_party/runtime")
    environment["TORCH_EXTENSIONS_DIR"] = str(extensions)
    runtime_paths = (str(ROOT), str(ROOT / "third_party/runtime"))
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join((*runtime_paths, *((existing,) if existing else ())))
    return environment


def arm_command(
    manifest: Path,
    arm_output: Path,
    arm: str,
    *,
    recovery: dict[str, str] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "reproduction.train_wikitext",
        "--arm",
        arm,
        "--manifest",
        str(manifest),
        "--output",
        str(arm_output),
    ]
    if recovery is not None:
        command.extend(
            [
                "--resume",
                str(arm_output / recovery["path"]),
                "--resume-sha256",
                recovery["sha256"],
            ]
        )
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default=os.environ.get("GPUS", "0"))
    args = parser.parse_args()
    gpus = parse_gpus(args.gpus)
    manifest = args.manifest.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    extensions = output / ".torch-extensions"
    extensions.mkdir(exist_ok=True)

    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_prepared_data.py"), str(manifest)],
        cwd=ROOT,
        check=True,
        env=process_environment(gpus[0], extensions),
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_released_sdm_extensions.py")],
        cwd=ROOT,
        check=True,
        env=process_environment(gpus[0], extensions),
    )

    pending = list(ARM_ORDER)
    running: dict[subprocess.Popen[bytes], tuple[str, str, object]] = {}
    while pending or running:
        free = [gpu for gpu in gpus if gpu not in {row[1] for row in running.values()}]
        while pending and free:
            arm = pending.pop(0)
            gpu = free.pop(0)
            arm_output = output / arm
            arm_output.mkdir(exist_ok=True)
            if (arm_output / "result.json").is_file():
                print(json.dumps({"marker": "ALREADY_COMPLETE", "arm": arm}), flush=True)
                continue
            latest = arm_output / "LATEST_RECOVERY.json"
            recovery = json.loads(latest.read_text()) if latest.is_file() else None
            command = arm_command(manifest, arm_output, arm, recovery=recovery)
            log_handle = (logs / f"{arm}.log").open("ab")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=process_environment(gpu, extensions),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running[process] = (arm, gpu, log_handle)
            print(json.dumps({"marker": "LAUNCHED", "arm": arm, "gpu": gpu}), flush=True)
        if not running:
            continue
        time.sleep(1.0)
        for process, (arm, gpu, log_handle) in list(running.items()):
            status = process.poll()
            if status is None:
                continue
            log_handle.close()
            del running[process]
            if status != 0:
                for active, (_other_arm, _other_gpu, other_log) in running.items():
                    active.terminate()
                    other_log.close()
                raise SystemExit(f"{arm} failed on GPU {gpu}; inspect {logs / f'{arm}.log'}")
            print(json.dumps({"marker": "ARM_COMPLETE", "arm": arm, "gpu": gpu}), flush=True)

    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_reproduction.py"), str(output)],
        cwd=ROOT,
        check=True,
        env=process_environment(gpus[0], extensions),
    )


if __name__ == "__main__":
    main()
