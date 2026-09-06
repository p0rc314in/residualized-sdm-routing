#!/usr/bin/env python3
"""Verify compact recorded results without claiming experiment reproduction."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared_residual_routing.results import (  # noqa: E402
    load_json,
    render_marquee,
    replace_marquee,
)
from scripts.verify_vendor import validate_vendor  # noqa: E402


ARM_ORDER = (
    "dense_attention",
    "b7a1_native_sdm",
    "b8_native_sdm",
    "b7a1_residualized_sdm",
    "b8_residualized_sdm",
)
PAIRS = {
    "b7a1": ("b7a1_native_sdm", "b7a1_residualized_sdm"),
    "b8": ("b8_native_sdm", "b8_residualized_sdm"),
}
EXPECTED_WIKITEXT = {
    "dense_attention": 4.362481402308412,
    "b7a1_native_sdm": 4.45031211109916,
    "b8_native_sdm": 4.482057562999111,
    "b7a1_residualized_sdm": 4.363565034856844,
    "b8_residualized_sdm": 4.379993436217285,
}
EXPECTED_RECALL_QUERY_LOSS = {
    "dense_attention": 2.0916048359601214,
    "b7a1_native_sdm": 2.0431395416129514,
    "b8_native_sdm": 2.1319903950413086,
    "b7a1_residualized_sdm": 2.033651775928926,
    "b8_residualized_sdm": 2.057441217191566,
}
EXPECTED_RECALL_EXACT_SET = {
    "dense_attention": 0.5624837239583333,
    "b7a1_native_sdm": 0.5628092447916667,
    "b8_native_sdm": 0.448583984375,
    "b7a1_residualized_sdm": 0.5545735677083333,
    "b8_residualized_sdm": 0.5538411458333333,
}
CURVE_COLUMNS = (
    "step",
    "dense_attention",
    "b7a1_native_sdm",
    "b7a1_residualized_sdm",
    "b8_native_sdm",
    "b8_residualized_sdm",
)
EXPECTED_CURVE_TERMINAL = {
    "dense_attention": 4.432816505432129,
    "b7a1_native_sdm": 4.501713275909424,
    "b7a1_residualized_sdm": 4.4167022705078125,
    "b8_native_sdm": 4.522711277008057,
    "b8_residualized_sdm": 4.422915458679199,
}
EXPECTED_CURVE_SHA256 = (
    "f95fca446e3b531c04e560b342d0d954fafea04527e522c1524d130ac4a9f1e0"
)
EXPECTED_BABYLM_ZERO_SHOT = {
    "dense": 0.4766219396606312,
    "native_sdm": 0.49016958658122484,
    "residualized_sdm": 0.49516603919573904,
}
EXPECTED_BABYLM_FINETUNE = {
    "dense": 0.6131050943641779,
    "native_sdm": 0.6091735915291369,
    "residualized_sdm": 0.6460266583044054,
}
EXPECTED_BABYLM_EWOK = {
    "dense": 0.5141085167147181,
    "native_sdm": 0.5124338067273445,
    "residualized_sdm": 0.5099625723826685,
}
EXPECTED_BABYLM_CHECKPOINT = {
    "dense": (0.45884004692100505, 0.002035710522954514, 0.00029600978135889156),
    "native_sdm": (
        0.47621779414348525,
        0.004858164377447219,
        0.00015159110443197844,
    ),
    "residualized_sdm": (
        0.46113051407598077,
        0.003782212924057458,
        0.00037691564469729985,
    ),
}


def exact(observed: float, expected: float, name: str) -> None:
    if not math.isclose(observed, expected, abs_tol=0.0, rel_tol=0.0):
        raise SystemExit(f"{name} changed")


def main() -> None:
    try:
        validate_vendor(ROOT / "third_party/runtime")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    result = load_json(ROOT / "data/results.json")
    if result.get("schema_version") != 4:
        raise SystemExit("unexpected recorded-result schema")
    if (result.get("width"), result.get("layers"), result.get("seed")) != (128, 8, 0):
        raise SystemExit("small-model identity changed")
    if tuple(result.get("arms", {})) != ARM_ORDER:
        raise SystemExit("small comparison must contain the exact five arms")
    sdm = result["sdm"]
    if (sdm["reads"], sdm["writes"]) != (8, 8):
        raise SystemExit("SDM comparison must retain R=W=8")

    benchmarks = result["benchmarks"]
    wikitext = benchmarks["wikitext103"]
    recall = benchmarks["adaptive_recall"]
    if wikitext.get("protocol_id") != "wikitext103-gpt2-causal-t2048-coverage-v1":
        raise SystemExit("WikiText protocol changed")
    if recall.get("protocol_id") != "adaptive-recall-seed102337-v1":
        raise SystemExit("Adaptive Recall protocol changed")
    if tuple(wikitext["arms"]) != ARM_ORDER or tuple(recall["arms"]) != ARM_ORDER:
        raise SystemExit("both benchmarks must contain the exact five-arm matrix")
    for arm in ARM_ORDER:
        exact(
            float(wikitext["arms"][arm]["test_nll"]),
            EXPECTED_WIKITEXT[arm],
            f"{arm} WikiText test NLL",
        )
        exact(
            float(recall["arms"][arm]["test_query_loss"]),
            EXPECTED_RECALL_QUERY_LOSS[arm],
            f"{arm} Recall test query loss",
        )
        exact(
            float(recall["arms"][arm]["test_exact_set_accuracy"]),
            EXPECTED_RECALL_EXACT_SET[arm],
            f"{arm} Recall exact-set accuracy",
        )

    dense = EXPECTED_WIKITEXT["dense_attention"]
    for topology, (native_arm, residualized_arm) in PAIRS.items():
        native = EXPECTED_WIKITEXT[native_arm]
        residualized = EXPECTED_WIKITEXT[residualized_arm]
        row = wikitext["test_nll_gap"][topology]
        exact(float(row["native_to_dense"]), native - dense, f"{topology} native gap")
        exact(
            float(row["residualized_to_dense"]),
            residualized - dense,
            f"{topology} residualized gap",
        )
        exact(
            float(row["closed_by_residualization"]),
            native - residualized,
            f"{topology} closed gap",
        )
        exact(
            float(row["fraction_closed"]),
            (native - residualized) / (native - dense),
            f"{topology} fraction closed",
        )

    dense_recall = EXPECTED_RECALL_QUERY_LOSS["dense_attention"]
    for topology, (native_arm, residualized_arm) in PAIRS.items():
        native = EXPECTED_RECALL_QUERY_LOSS[native_arm]
        residualized = EXPECTED_RECALL_QUERY_LOSS[residualized_arm]
        row = recall["test_query_loss_effect"][topology]
        exact(
            float(row["native_to_dense"]),
            native - dense_recall,
            f"{topology} Recall native gap",
        )
        exact(
            float(row["residualized_to_dense"]),
            residualized - dense_recall,
            f"{topology} Recall residualized gap",
        )
        exact(
            float(row["improvement_from_residualization"]),
            native - residualized,
            f"{topology} Recall residualization effect",
        )

    curve_path = ROOT / "data/wikitext-training-curves.csv"
    if hashlib.sha256(curve_path.read_bytes()).hexdigest() != EXPECTED_CURVE_SHA256:
        raise SystemExit("WikiText training curve identity changed")
    provenance = load_json(ROOT / "provenance.json")
    if (
        provenance["wikitext103"]["released_training_curves"]["sha256"]
        != EXPECTED_CURVE_SHA256
    ):
        raise SystemExit("WikiText training curve provenance changed")
    with curve_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CURVE_COLUMNS:
            raise SystemExit("WikiText curve columns changed")
        curves = list(reader)
    if len(curves) != 21_603:
        raise SystemExit("WikiText curves must contain all 21,603 optimizer steps")
    if [int(row["step"]) for row in curves] != list(range(1, 21_604)):
        raise SystemExit("WikiText curve steps are incomplete")
    for arm, expected in EXPECTED_CURVE_TERMINAL.items():
        values = [float(row[arm]) for row in curves]
        if not all(math.isfinite(value) for value in values):
            raise SystemExit(f"{arm} WikiText curve contains a non-finite value")
        exact(values[-1], expected, f"{arm} terminal training NLL")

    readme = (ROOT / "README.md").read_text()
    for value in (
        "98.8%",
        "85.4%",
        "92.4%",
        "44.86%",
        "55.38%",
        "94.6%",
        "64.60%",
    ):
        if value not in readme:
            raise SystemExit(f"README no longer states {value}")
    if "figures/wikitext-training.png" not in readme:
        raise SystemExit("README no longer includes the WikiText training curve")
    marquee = load_json(ROOT / "data/marquee-results.json")
    if marquee.get("status") != "complete":
        raise SystemExit("BabyLM result is not complete")
    terminal = marquee["evaluation"]["terminal"]
    for arm, expected in EXPECTED_BABYLM_ZERO_SHOT.items():
        exact(
            float(terminal["zero_shot"]["arms"][arm]["macro_accuracy"]),
            expected,
            f"{arm} BabyLM zero-shot macro",
        )
        exact(
            float(terminal["zero_shot"]["arms"][arm]["by_task"]["ewok"]),
            EXPECTED_BABYLM_EWOK[arm],
            f"{arm} BabyLM EWoK accuracy",
        )
    for arm, expected in EXPECTED_BABYLM_FINETUNE.items():
        exact(
            float(terminal["finetune"]["arms"][arm]["macro_primary_metric"]),
            expected,
            f"{arm} BabyLM fine-tune macro",
        )
    checkpoint = marquee["evaluation"]["checkpoint_evaluation"]
    for arm, expected in EXPECTED_BABYLM_CHECKPOINT.items():
        row = checkpoint["arms"][arm]
        for observed, target, metric in zip(
            (
                row["fast_zero_shot_macro_accuracy"],
                row["reading"]["eye_tracking"],
                row["reading"]["self_paced"],
            ),
            expected,
            ("fast zero-shot macro", "eye tracking", "self-paced reading"),
            strict=True,
        ):
            exact(float(observed), target, f"{arm} BabyLM checkpoint {metric}")
        exact(
            float(row["age_of_acquisition"]["curve_fitness"]),
            0.0,
            f"{arm} BabyLM Age-of-Acquisition curve fitness",
        )
    if replace_marquee(readme, render_marquee(marquee)) != readme:
        raise SystemExit("README BabyLM result section is out of date")
    print("recorded WikiText, Recall, and BabyLM results verified")


if __name__ == "__main__":
    main()
