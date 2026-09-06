from __future__ import annotations

import json
import hashlib
import tempfile
from pathlib import Path
import tomllib
import unittest

from reproduction.spec import (
    ARM_ORDER,
    ARMS,
    EXPECTED_GAP_CLOSURE_RANGES,
    EXPECTED_NLL_RANGES,
    GEOMETRY,
    TOTAL_STEPS,
    WARMUP_STEPS,
)
from scripts.check_release import (
    is_sha256, validate_reproduction_record, validate_reproduction_policy,
    validate_optional_babylm_checkpoint_locations,
)
from scripts.run_experiments import arm_command
from scripts.verify_vendor import validate_vendor


ROOT = Path(__file__).resolve().parents[1]


def option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


class ReproductionTests(unittest.TestCase):
    def test_exact_five_arm_training_commands(self) -> None:
        commands = [
            arm_command(Path("manifest"), Path("output") / arm, arm)
            for arm in ARM_ORDER
        ]
        self.assertEqual(len(commands), 5)
        for arm, command in zip(ARM_ORDER, commands, strict=True):
            self.assertEqual(command[1:3], ["-m", "reproduction.train_wikitext"])
            self.assertEqual(option(command, "--arm"), arm)
            self.assertEqual(option(command, "--manifest"), "manifest")
            self.assertEqual(option(command, "--output"), str(Path("output") / arm))

    def test_balanced_canonical_configuration(self) -> None:
        self.assertEqual((GEOMETRY.reads, GEOMETRY.writes), (8, 8))
        self.assertEqual(GEOMETRY.logical_capacity, 1024)
        self.assertEqual(TOTAL_STEPS, 21_603)
        self.assertEqual(WARMUP_STEPS, 540)
        self.assertEqual(
            tuple(ARMS[arm].expected_trainable_parameters for arm in ARM_ORDER),
            (14_965_120, 15_029_660, 15_038_880, 15_087_452, 15_104_928),
        )

    def test_reproduce_all_runs_training(self) -> None:
        script = (ROOT / "reproduce.sh").read_text()
        all_block = script.split("  all)", 1)[1].split("  *)", 1)[0]
        self.assertIn("run_experiments.py", all_block)
        self.assertNotIn("verify_results.py", all_block)

    def test_fresh_result_acceptance_is_predeclared_per_arm(self) -> None:
        self.assertEqual(tuple(EXPECTED_NLL_RANGES), ARM_ORDER)
        self.assertEqual(EXPECTED_GAP_CLOSURE_RANGES["b7a1"], (0.75, 1.20))
        self.assertEqual(EXPECTED_GAP_CLOSURE_RANGES["b8"], (0.65, 1.05))

    def test_public_sources_have_no_private_absolute_path(self) -> None:
        roots = [
            ROOT / "reproduction",
            ROOT / "scripts",
            ROOT / "shared_residual_routing",
        ]
        for directory in roots:
            for path in directory.rglob("*.py"):
                self.assertNotIn("/Users/", path.read_text(), str(path))

    def test_package_identity_and_runtime_license(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(metadata["project"]["name"], "residualized-sdm-routing")
        self.assertIn(
            "third_party/runtime/LICENSE", metadata["project"]["license-files"]
        )
        self.assertTrue((ROOT / "third_party/runtime/manifest.json").is_file())
        validate_vendor(ROOT / "third_party/runtime")

    def test_bounded_validation_does_not_claim_full_rerun(self) -> None:
        self.assertFalse((ROOT / "REPRODUCTION_VALIDATED.json").exists())
        identity = json.loads((ROOT / "PROJECT_IDENTITY.json").read_text())
        validate_reproduction_policy(identity)
        for benchmark in ("wikitext103", "adaptive_recall", "babylm"):
            self.assertEqual(identity["reproduction"][benchmark]["status"], "implemented")
            self.assertEqual(identity["reproduction"][benchmark]["end_to_end_validation"], "not_run")

    def test_policy_rejects_reintroduced_full_rerun_requirement(self) -> None:
        identity = json.loads((ROOT / "PROJECT_IDENTITY.json").read_text())
        identity["reproduction_validation_policy"]["full_experiment_rerun_required"] = True
        with self.assertRaisesRegex(ValueError, "not a release-validation requirement"):
            validate_reproduction_policy(identity)

    def test_checkpoint_distribution_is_optional_but_hash_bound(self) -> None:
        hashes = {
            arm: str(index) * 64
            for index, arm in enumerate(("dense", "native_sdm", "residualized_sdm"), 1)
        }
        provenance = {"babylm": {"terminal_checkpoints": hashes}}
        validate_optional_babylm_checkpoint_locations(provenance)
        locations = {
            arm: {"sha256": digest, "location": f"https://example.org/{arm}.pt"}
            for arm, digest in hashes.items()
        }
        provenance["babylm"]["terminal_checkpoint_locations"] = locations
        validate_optional_babylm_checkpoint_locations(provenance)
        locations["dense"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint location is missing or unbound"):
            validate_optional_babylm_checkpoint_locations(provenance)

    def test_release_validation_checks_record_contents(self) -> None:
        expected_arms = ("dense_attention", "b8_native_sdm")
        baseline = {
            "status": "clean_checkout_gpu_passed",
            "benchmark": "adaptive_recall",
            "protocol_id": "adaptive-recall-seed102337-v1",
            "arms": list(expected_arms),
            "input_manifest_sha256": "1" * 64,
            "packet_tree_sha256": "2" * 64,
            "verifier_sha256": "3" * 64,
            "measurements": {
                "path": "measurements.json",
                "sha256": "pending",
            },
            "terminal_checkpoints": {
                arm: {
                    "location": f"https://example.org/checkpoints/{arm}.pt",
                    "sha256": str(index) * 64,
                }
                for index, arm in enumerate(expected_arms, 5)
            },
            "hardware": {
                "accelerator": "test GPU",
                "cuda": "12.8",
                "pytorch": "2.11.0",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            measurements = root / "measurements.json"
            measurements.write_text(
                json.dumps(
                    {
                        "status": "reproduced",
                        "terminal_checkpoints_roundtrip_inference_passed": list(
                            expected_arms
                        ),
                        "arms": [
                            {
                                "arm": arm,
                                "checkpoint_sha256": str(index) * 64,
                            }
                            for index, arm in enumerate(expected_arms, 5)
                        ],
                    }
                )
                + "\n"
            )
            baseline["measurements"]["sha256"] = hashlib.sha256(
                measurements.read_bytes()
            ).hexdigest()
            path = root / "VALIDATED.json"
            path.write_text(json.dumps(baseline))
            validate_reproduction_record(
                path,
                benchmark="adaptive_recall",
                protocol_id="adaptive-recall-seed102337-v1",
                arms=expected_arms,
                input_manifest_sha256="1" * 64,
                packet_tree_sha256="2" * 64,
                verifier_sha256="3" * 64,
                root=root,
                tracked_files={"measurements.json"},
            )
            for field in baseline:
                record = dict(baseline)
                record[field] = "changed"
                path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    validate_reproduction_record(
                        path,
                        benchmark="adaptive_recall",
                        protocol_id="adaptive-recall-seed102337-v1",
                        arms=expected_arms,
                        input_manifest_sha256="1" * 64,
                        packet_tree_sha256="2" * 64,
                        verifier_sha256="3" * 64,
                        root=root,
                        tracked_files={"measurements.json"},
                    )

    def test_sha256_validation_is_strict(self) -> None:
        self.assertTrue(is_sha256("a" * 64))
        self.assertFalse(is_sha256("A" * 64))
        self.assertFalse(is_sha256("a" * 63))


if __name__ == "__main__":
    unittest.main()
