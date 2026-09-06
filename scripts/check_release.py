#!/usr/bin/env python3
"""Reject a release while required terminal evidence or files are missing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared_residual_routing.results import load_json, validate_marquee  # noqa: E402
from scripts.verify_vendor import validate_vendor, validate_babylm_vendor  # noqa: E402


EXPECTED_REPRODUCTIONS = {
    "wikitext103": {
        "path": "REPRODUCTION_VALIDATED.json",
        "protocol_id": "wikitext103-gpt2-causal-t2048-coverage-v1",
        "input_manifest_sha256": "fc4ef13cbc38070f2d7774dffbfd5be48cab31fe45d6d9995d522fc3bac1dde6",
        "verifier": "scripts/check_reproduction.py",
        "arms": (
            "dense_attention",
            "b7a1_native_sdm",
            "b8_native_sdm",
            "b7a1_residualized_sdm",
            "b8_residualized_sdm",
        ),
    },
    "adaptive_recall": {
        "path": "RECALL_REPRODUCTION_VALIDATED.json",
        "protocol_id": "adaptive-recall-seed102337-v1",
        "input_manifest_sha256": "b0587d62c3ab709c94e37742892451a39463a7d577212b9379bd76d966f97800",
        "verifier": "scripts/check_recall_reproduction.py",
        "arms": (
            "dense_attention",
            "b7a1_native_sdm",
            "b8_native_sdm",
            "b7a1_residualized_sdm",
            "b8_residualized_sdm",
        ),
    },
    "babylm": {
        "path": "BABYLM_REPRODUCTION_VALIDATED.json",
        "protocol_id": "babylm2026-strict-gpt2-causal-t2048-10epoch-v1",
        "input_manifest_sha256": "b8d15c06233c99769a1bd5e7164c22f0ca55cd39d1a943d63e61dfd1ddda84e2",
        "verifier": "scripts/check_babylm_reproduction.py",
        "arms": (
            "a16_dense_attention",
            "b16_native_sdm",
            "b16_residualized_sdm",
        ),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packet_tree_sha256(root: Path) -> str:
    tracked = (
        subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
        )
        .stdout.decode("utf-8")
        .split("\0")
    )
    excluded = {expected["path"] for expected in EXPECTED_REPRODUCTIONS.values()}
    digest = hashlib.sha256()
    for relative in sorted(path for path in tracked if path and path not in excluded):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"tracked packet file is missing: {relative}")
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def bound_tracked_file(
    root: Path,
    record: object,
    *,
    tracked_files: set[str],
    label: str,
) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"{label} identity is absent")
    relative = record.get("path")
    if not isinstance(relative, str):
        raise ValueError(f"{label} path is absent")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} path is not packet-relative")
    if candidate.as_posix() not in tracked_files:
        raise ValueError(f"{label} is not tracked by the packet")
    path = root / candidate
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{label} file identity changed")
    return path


def validate_reproduction_record(
    path: Path,
    *,
    benchmark: str,
    protocol_id: str,
    arms: tuple[str, ...],
    input_manifest_sha256: str,
    packet_tree_sha256: str,
    verifier_sha256: str,
    root: Path,
    tracked_files: set[str],
) -> None:
    if not path.is_file():
        raise ValueError(f"missing {path.name}")
    try:
        validation = load_json(path)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path.name}: {error}") from error
    if validation.get("status") != "clean_checkout_gpu_passed":
        raise ValueError(
            f"{benchmark} clean-checkout GPU reproduction is not validated"
        )
    if validation.get("benchmark") != benchmark:
        raise ValueError(f"{benchmark} validation benchmark identity changed")
    if validation.get("protocol_id") != protocol_id:
        raise ValueError(f"{benchmark} validation protocol identity changed")
    if tuple(validation.get("arms", ())) != arms:
        raise ValueError(f"{benchmark} validation arm matrix changed")
    if validation.get("input_manifest_sha256") != input_manifest_sha256:
        raise ValueError(f"{benchmark} validation input identity changed")
    if validation.get("packet_tree_sha256") != packet_tree_sha256:
        raise ValueError(f"{benchmark} validation is stale for this packet tree")
    if validation.get("verifier_sha256") != verifier_sha256:
        raise ValueError(f"{benchmark} validation verifier identity changed")
    measurements_path = bound_tracked_file(
        root,
        validation.get("measurements"),
        tracked_files=tracked_files,
        label=f"{benchmark} fresh measurements",
    )
    measurements = load_json(measurements_path)
    if (
        tuple(measurements.get("terminal_checkpoints_roundtrip_inference_passed", ()))
        != arms
    ):
        raise ValueError(f"{benchmark} checkpoint inference round-trip is incomplete")
    measurement_rows = measurements.get("arms")
    if not isinstance(measurement_rows, list):
        raise ValueError(f"{benchmark} measurements lack per-arm checkpoint identities")
    measurement_checkpoints = {
        row.get("arm"): row.get("checkpoint_sha256")
        for row in measurement_rows
        if isinstance(row, dict)
    }
    if tuple(measurement_checkpoints) != arms or not all(
        is_sha256(measurement_checkpoints[arm]) for arm in arms
    ):
        raise ValueError(f"{benchmark} measurement checkpoint matrix changed")
    checkpoints = validation.get("terminal_checkpoints")
    if not isinstance(checkpoints, dict) or tuple(checkpoints) != arms:
        raise ValueError(f"{benchmark} validation checkpoint matrix changed")
    for arm in arms:
        checkpoint = checkpoints[arm]
        if not isinstance(checkpoint, dict) or not is_sha256(checkpoint.get("sha256")):
            raise ValueError(
                f"{benchmark} validation has an invalid checkpoint identity"
            )
        if checkpoint["sha256"] != measurement_checkpoints[arm]:
            raise ValueError(
                f"{benchmark} published checkpoint does not match inference for {arm}"
            )
        location = checkpoint.get("location")
        if not isinstance(location, str) or not location:
            raise ValueError(f"{benchmark} checkpoint location is absent for {arm}")
        parsed = urlparse(location)
        if parsed.scheme:
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError(f"{benchmark} checkpoint location is not HTTPS")
        else:
            bound_tracked_file(
                root,
                {"path": location, "sha256": checkpoint["sha256"]},
                tracked_files=tracked_files,
                label=f"{benchmark} checkpoint for {arm}",
            )
    hardware = validation.get("hardware")
    required_hardware = ("accelerator", "cuda", "pytorch")
    if not isinstance(hardware, dict) or not all(
        isinstance(hardware.get(field), str) and hardware[field]
        for field in required_hardware
    ):
        raise ValueError(f"{benchmark} validation hardware identity is incomplete")


def main() -> None:
    identity = load_json(ROOT / "PROJECT_IDENTITY.json")
    try:
        validate_reproduction_policy(identity)
    except ValueError as error:
        raise SystemExit(f"release blocked: {error}") from error
    reproduction = identity.get("reproduction", {})
    expected_status = "implemented"
    for benchmark in EXPECTED_REPRODUCTIONS:
        status = reproduction.get(benchmark, {}).get("status")
        if status != expected_status:
            raise SystemExit(
                f"release blocked: {benchmark} reproduction status is {status!r}"
            )
    try:
        validate_marquee(load_json(ROOT / "data/marquee-results.json"), release=True)
    except ValueError as error:
        raise SystemExit(f"release blocked: {error}") from error
    required = [
        ROOT / "reproduce.sh",
        ROOT / "uv.lock",
        ROOT / "data/wikitext-training-curves.csv",
        ROOT / "figures/wikitext-training.png",
        ROOT / "figures/babylm-training.png",
        ROOT / "figures/babylm-evaluation.png",
        ROOT / "figures/social-preview.png",
        ROOT / "scripts/smoke_reproduction.py",
        ROOT / "scripts/run_reproductions.py",
        *(ROOT / expected["verifier"] for expected in EXPECTED_REPRODUCTIONS.values()),
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("release blocked: missing " + ", ".join(missing))
    try:
        validate_vendor(ROOT / "third_party/runtime")
        validate_babylm_vendor(ROOT / "third_party/babylm_runtime")
    except ValueError as error:
        raise SystemExit(f"release blocked: {error}") from error
    provenance = load_json(ROOT / "provenance.json")
    for figure in provenance.get("figures", {}).values():
        path = ROOT / figure.get("path", "")
        if not path.is_file() or sha256_file(path) != figure.get("sha256"):
            raise SystemExit(
                f"release blocked: figure identity changed for {path.name}"
            )
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/verify_results.py")],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            cwd=ROOT,
            check=True,
        )
    except (subprocess.CalledProcessError, ValueError) as error:
        raise SystemExit(
            f"release blocked: packet verification failed: {error}"
        ) from error
    tracked_files = set(
        subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.splitlines()
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    if dirty:
        raise SystemExit("release blocked: packet worktree is not clean")
    for path in required:
        if path.relative_to(ROOT).as_posix() not in tracked_files:
            raise SystemExit(f"release blocked: {path.name} is not tracked")
    commit_count = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if commit_count != "1":
        raise SystemExit("release blocked: history must contain exactly one commit")
    try:
        validate_optional_babylm_checkpoint_locations(provenance)
    except ValueError as error:
        raise SystemExit(f"release blocked: {error}") from error
    print("bounded implementation and existing-result checks passed; full experiment reruns were not required")


def validate_reproduction_policy(identity: dict) -> None:
    policy = identity.get("reproduction_validation_policy", {})
    if policy.get("mode") != "bounded_implementation_checks":
        raise ValueError("reproduction validation must use bounded implementation checks")
    if policy.get("full_experiment_rerun_required") is not False:
        raise ValueError("full experiment reruns are not a release-validation requirement")


def validate_optional_babylm_checkpoint_locations(provenance: dict) -> None:
    """Check download identities when supplied; publishing weights is optional."""
    babylm = provenance.get("babylm", {})
    hashes = babylm.get("terminal_checkpoints", {})
    locations = babylm.get("terminal_checkpoint_locations", {})
    if not locations:
        return
    for arm in ("dense", "native_sdm", "residualized_sdm"):
        record = locations.get(arm, {})
        if record.get("sha256") != hashes.get(arm) or not is_sha256(record.get("sha256")):
            raise ValueError(f"existing BabyLM checkpoint location is missing or unbound for {arm}")
        parsed = urlparse(record.get("location", ""))
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError(f"existing BabyLM checkpoint needs an HTTPS location for {arm}")


if __name__ == "__main__":
    main()
