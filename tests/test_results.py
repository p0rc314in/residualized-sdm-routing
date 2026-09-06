from __future__ import annotations

import csv
import json
from pathlib import Path
import unittest

from shared_residual_routing.results import (
    load_json,
    render_marquee,
    replace_marquee,
    validate_marquee,
)
from reproduction.spec import ARM_ORDER as SMALL_ARM_ORDER, ARMS as SMALL_ARMS, GEOMETRY


ROOT = Path(__file__).resolve().parents[1]


class ResultTests(unittest.TestCase):
    def test_small_experiment_has_exact_five_arm_matrix(self) -> None:
        self.assertEqual(
            SMALL_ARM_ORDER,
            (
                "dense_attention",
                "b7a1_native_sdm",
                "b8_native_sdm",
                "b7a1_residualized_sdm",
                "b8_residualized_sdm",
            ),
        )
        self.assertEqual(
            tuple(SMALL_ARMS[name].profile.layout for name in SMALL_ARM_ORDER),
            ("AAAAAAAA", "BBBBBBBA", "BBBBBBBB", "BBBBBBBA", "BBBBBBBB"),
        )
        self.assertEqual((GEOMETRY.reads, GEOMETRY.writes), (8, 8))

    def test_recorded_result_has_both_five_arm_benchmarks(self) -> None:
        payload = load_json(ROOT / "data/results.json")
        self.assertEqual(payload["schema_version"], 4)
        self.assertEqual(tuple(payload["arms"]), SMALL_ARM_ORDER)
        self.assertEqual(
            tuple(payload["benchmarks"]["adaptive_recall"]["arms"]),
            SMALL_ARM_ORDER,
        )
        self.assertAlmostEqual(
            payload["benchmarks"]["wikitext103"]["test_nll_gap"]["b7a1"][
                "fraction_closed"
            ],
            0.9876622588687802,
        )
        self.assertAlmostEqual(
            payload["benchmarks"]["wikitext103"]["test_nll_gap"]["b8"][
                "fraction_closed"
            ],
            0.8535491204290251,
        )
        recall_arms = payload["benchmarks"]["adaptive_recall"]["arms"]
        dense = recall_arms["dense_attention"]["test_exact_set_accuracy"]
        native = recall_arms["b8_native_sdm"]["test_exact_set_accuracy"]
        residualized = recall_arms["b8_residualized_sdm"]["test_exact_set_accuracy"]
        self.assertAlmostEqual(
            (residualized - native) / (dense - native),
            0.9241211774792796,
        )

    def test_complete_babylm_result_is_release_valid(self) -> None:
        payload = load_json(ROOT / "data/marquee-results.json")
        validate_marquee(payload)
        validate_marquee(payload, release=True)
        self.assertEqual(payload["status"], "complete")
        terminal = payload["evaluation"]["terminal"]
        self.assertEqual(terminal["zero_shot"]["task_count"], 6)
        self.assertEqual(
            tuple(terminal["zero_shot"]["arms"]["dense"]["by_task"]),
            (
                "blimp",
                "blimp_supplement",
                "comps",
                "entity_tracking",
                "ewok",
                "global_piqa",
                "global_piqa_nonparallel",
                "global_piqa_parallel",
            ),
        )
        dense_zero = terminal["zero_shot"]["arms"]["dense"]
        self.assertAlmostEqual(
            dense_zero["by_task"]["global_piqa"],
            (
                dense_zero["by_task"]["global_piqa_nonparallel"]
                + dense_zero["by_task"]["global_piqa_parallel"]
            )
            / 2,
        )
        residualized_tune = terminal["finetune"]["arms"]["residualized_sdm"]
        self.assertAlmostEqual(
            residualized_tune["macro_primary_metric"],
            sum(residualized_tune["primary_metric_by_task"].values()) / 7,
        )
        self.assertNotAlmostEqual(
            residualized_tune["macro_primary_metric"],
            sum(residualized_tune["by_task"].values()) / 7,
        )

    def test_wikitext_curve_contains_every_unsmoothed_step(self) -> None:
        with (ROOT / "data/wikitext-training-curves.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 21_603)
        self.assertEqual(
            [int(row["step"]) for row in rows],
            list(range(1, 21_604)),
        )
        self.assertAlmostEqual(
            float(rows[-1]["b7a1_residualized_sdm"]),
            4.4167022705078125,
        )
        self.assertAlmostEqual(
            float(rows[-1]["b8_residualized_sdm"]),
            4.422915458679199,
        )
        self.assertIn(
            "figures/wikitext-training.png",
            (ROOT / "README.md").read_text(),
        )

    def test_marquee_render_contains_terminal_babylm_results_only(self) -> None:
        payload = load_json(ROOT / "data/marquee-results.json")
        rendered = render_marquee(payload)
        self.assertIn("figures/babylm-training.png", rendered)
        self.assertIn("figures/babylm-evaluation.png", rendered)
        self.assertIn(
            "| Evaluation | Task | Metric | Dense attention | Native SDM | Residualized SDM |",
            rendered,
        )
        for task in ("BLiMP", "EWoK", "BoolQ", "WSC"):
            self.assertIn(f"| {task} |", rendered)
        self.assertNotIn("RULER", rendered)

    def test_readme_block_is_generated_from_training_record(self) -> None:
        readme = (ROOT / "README.md").read_text()
        payload = json.loads((ROOT / "data/marquee-results.json").read_text())
        self.assertEqual(readme, replace_marquee(readme, render_marquee(payload)))


if __name__ == "__main__":
    unittest.main()
