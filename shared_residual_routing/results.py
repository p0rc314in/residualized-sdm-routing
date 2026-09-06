"""Validate and render the note's BabyLM result section."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


SECTION_HEADING = "### Scaling the comparison to BabyLM"
NEXT_HEADING = "## Why it matters"
CAMPAIGN_ID = "sdm-shared-router-babylm2026-strict-seed0-v1"
PROTOCOL_ID = "babylm2026-strict-gpt2-causal-t2048-10epoch-v1"
ARM_ORDER = ("dense", "native_sdm", "residualized_sdm")
ZERO_SHOT_TASKS = (
    "blimp",
    "blimp_supplement",
    "comps",
    "entity_tracking",
    "ewok",
    "global_piqa",
)
FINETUNE_TASKS = ("boolq", "mnli", "mrpc", "multirc", "qqp", "rte", "wsc")
TASK_LABELS = {
    "blimp": "BLiMP",
    "blimp_supplement": "BLiMP Supplement",
    "comps": "COMPS",
    "entity_tracking": "Entity Tracking",
    "ewok": "EWoK",
    "global_piqa": "Global PIQA",
    "boolq": "BoolQ",
    "mnli": "MNLI",
    "mrpc": "MRPC",
    "multirc": "MultiRC",
    "qqp": "QQP",
    "rte": "RTE",
    "wsc": "WSC",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _validate_training(payload: dict[str, Any]) -> None:
    training = payload.get("training")
    if not isinstance(training, dict) or training.get("status") != "complete":
        raise ValueError("BabyLM training is not complete")
    if training.get("metric") != "trailing_1000_step_mean_training_nll":
        raise ValueError("BabyLM training metric changed")
    terminal_step = training.get("terminal_step")
    if terminal_step != 102_852:
        raise ValueError("BabyLM terminal step changed")
    if training.get("target_presentations_per_arm") != 1_685_114_880:
        raise ValueError("BabyLM exposure count changed")

    arms = training.get("arms")
    if not isinstance(arms, dict) or tuple(arms) != ARM_ORDER:
        raise ValueError(f"BabyLM training requires arms in order {ARM_ORDER}")
    terminal: dict[str, float] = {}
    expected_steps: tuple[int, ...] | None = None
    for arm_name in ARM_ORDER:
        arm = arms[arm_name]
        if not isinstance(arm, dict) or arm.get("status") != "terminal_verified":
            raise ValueError(f"{arm_name} is not terminal and verified")
        parameters = _number(
            arm.get("trainable_parameters"), f"{arm_name}.trainable_parameters"
        )
        peak_memory = _number(
            arm.get("peak_allocated_device_memory_bytes"),
            f"{arm_name}.peak_allocated_device_memory_bytes",
        )
        if parameters <= 0 or peak_memory <= 0:
            raise ValueError(f"{arm_name} resource measurements must be positive")
        checkpoint = arm.get("checkpoint_sha256")
        if not isinstance(checkpoint, str) or len(checkpoint) != 64:
            raise ValueError(f"{arm_name} checkpoint identity is incomplete")
        trajectory = arm.get("trajectory")
        if not isinstance(trajectory, list) or len(trajectory) < 2:
            raise ValueError(f"{arm_name} trajectory is incomplete")
        steps: list[int] = []
        values: list[float] = []
        for index, row in enumerate(trajectory):
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError(f"{arm_name}.trajectory[{index}] must be [step, nll]")
            step = row[0]
            if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
                raise ValueError(f"{arm_name}.trajectory[{index}] step is invalid")
            steps.append(step)
            values.append(_number(row[1], f"{arm_name}.trajectory[{index}].nll"))
        if steps != sorted(set(steps)) or steps[-1] != terminal_step:
            raise ValueError(f"{arm_name} trajectory steps are incomplete or unordered")
        if expected_steps is None:
            expected_steps = tuple(steps)
        elif tuple(steps) != expected_steps:
            raise ValueError("BabyLM arms must use matched trajectory steps")
        terminal[arm_name] = values[-1]

    native_gap = terminal["native_sdm"] - terminal["dense"]
    residualized_gap = terminal["residualized_sdm"] - terminal["dense"]
    closed = terminal["native_sdm"] - terminal["residualized_sdm"]
    fraction = closed / native_gap
    expected = {
        "native_to_dense": native_gap,
        "residualized_to_dense": residualized_gap,
        "closed_by_residualization": closed,
        "fraction_closed": fraction,
    }
    recorded = training.get("terminal_gap")
    if not isinstance(recorded, dict):
        raise ValueError("BabyLM terminal gap accounting is absent")
    for name, value in expected.items():
        if not math.isclose(
            _number(recorded.get(name), f"training.terminal_gap.{name}"),
            value,
            abs_tol=1e-15,
            rel_tol=1e-15,
        ):
            raise ValueError(f"BabyLM terminal gap field {name} is inconsistent")


def _validate_evaluation(payload: dict[str, Any], *, release: bool) -> None:
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("BabyLM evaluation record is absent")
    status = evaluation.get("status")
    if status == "pending":
        if evaluation.get("terminal") or evaluation.get("checkpoint_evaluation"):
            raise ValueError("pending evaluation must not contain metrics")
        if release:
            raise ValueError("BabyLM evaluation is still pending")
        return
    if status != "complete":
        raise ValueError("BabyLM evaluation status is invalid")
    terminal = evaluation.get("terminal")
    if not isinstance(terminal, dict):
        raise ValueError("complete BabyLM evaluation requires terminal metrics")
    for surface, tasks in (
        ("zero_shot", ZERO_SHOT_TASKS),
        ("finetune", FINETUNE_TASKS),
    ):
        result = terminal.get(surface)
        if not isinstance(result, dict) or result.get("task_count") != len(tasks):
            raise ValueError(f"BabyLM {surface} task coverage changed")
        arms = result.get("arms")
        if not isinstance(arms, dict) or tuple(arms) != ARM_ORDER:
            raise ValueError(f"BabyLM {surface} requires all arms")
        for arm_name in ARM_ORDER:
            arm = arms[arm_name]
            if not isinstance(arm, dict):
                raise ValueError(f"BabyLM {surface}.{arm_name} is invalid")
            aggregate_key = (
                "macro_accuracy" if surface == "zero_shot" else "macro_primary_metric"
            )
            aggregate = _number(
                arm.get(aggregate_key),
                f"evaluation.terminal.{surface}.{arm_name}.{aggregate_key}",
            )
            if not 0.0 <= aggregate <= 1.0:
                raise ValueError("BabyLM score must lie in [0, 1]")
            by_task = arm.get("by_task")
            if not isinstance(by_task, dict):
                raise ValueError(f"BabyLM {surface}.{arm_name} task set changed")
            if surface == "zero_shot":
                required = (*tasks, "global_piqa_nonparallel", "global_piqa_parallel")
                if not all(task in by_task for task in required):
                    raise ValueError(f"BabyLM {surface}.{arm_name} task set changed")
                global_piqa = _number(
                    by_task["global_piqa"],
                    f"evaluation.terminal.{surface}.{arm_name}.global_piqa",
                )
                split_mean = (
                    _number(
                        by_task["global_piqa_nonparallel"],
                        f"evaluation.terminal.{surface}.{arm_name}.global_piqa_nonparallel",
                    )
                    + _number(
                        by_task["global_piqa_parallel"],
                        f"evaluation.terminal.{surface}.{arm_name}.global_piqa_parallel",
                    )
                ) / 2
                if not math.isclose(
                    global_piqa, split_mean, abs_tol=1e-15, rel_tol=1e-15
                ):
                    raise ValueError(
                        f"BabyLM {surface}.{arm_name} Global PIQA mean is inconsistent"
                    )
                score_map = by_task
            else:
                if tuple(by_task) != tasks:
                    raise ValueError(f"BabyLM {surface}.{arm_name} task set changed")
                score_map = arm.get("primary_metric_by_task")
                if not isinstance(score_map, dict) or tuple(score_map) != tasks:
                    raise ValueError(
                        f"BabyLM {surface}.{arm_name} primary metric set changed"
                    )
            task_scores: list[float] = []
            for task in tasks:
                score = _number(score_map[task], f"BabyLM {surface}.{arm_name}.{task}")
                if not 0.0 <= score <= 1.0:
                    raise ValueError("BabyLM score must lie in [0, 1]")
                task_scores.append(score)
            if not math.isclose(
                aggregate,
                sum(task_scores) / len(task_scores),
                abs_tol=1e-15,
                rel_tol=1e-15,
            ):
                raise ValueError(f"BabyLM {surface}.{arm_name} macro is inconsistent")

    validation = evaluation.get("inference_validation")
    if not isinstance(validation, dict):
        raise ValueError("official checkpoint inference validation is absent")
    if validation.get("examples") != 2_048 or validation.get("candidates") != 4_096:
        raise ValueError("official checkpoint inference oracle extent changed")
    if _number(validation.get("max_abs_score_delta"), "max_abs_score_delta") != 0.0:
        raise ValueError("optimized checkpoint inference disagrees with the oracle")
    if validation.get("prediction_matches") != 2_048:
        raise ValueError("optimized checkpoint predictions disagree with the oracle")

    checkpoint = evaluation.get("checkpoint_evaluation")
    if not isinstance(checkpoint, dict):
        raise ValueError("official checkpoint evaluation record is absent")
    if checkpoint.get("checkpoint_count_per_arm") != 28:
        raise ValueError("official checkpoint evaluation coverage changed")
    if checkpoint.get("status") != "complete":
        raise ValueError("official checkpoint evaluation status is invalid")
    if checkpoint.get("terminal_official_corpus_exposure_millions") != 1_000:
        raise ValueError("official checkpoint terminal exposure changed")
    if checkpoint.get("terminal_target_token_presentations") != 1_685_114_880:
        raise ValueError("terminal target-token presentation count changed")
    if checkpoint.get("reading_metric") != "normalized_incremental_r_squared":
        raise ValueError("official Reading metric changed")
    if (
        checkpoint.get("age_of_acquisition_metric")
        != "curve_fitness_zero_when_p_gt_0.1"
    ):
        raise ValueError("official Age-of-Acquisition metric changed")
    checkpoint_arms = checkpoint.get("arms")
    if not isinstance(checkpoint_arms, dict) or tuple(checkpoint_arms) != ARM_ORDER:
        raise ValueError("official checkpoint evaluation requires all arms")
    for arm_name in ARM_ORDER:
        arm = checkpoint_arms[arm_name]
        fast_macro = _number(
            arm.get("fast_zero_shot_macro_accuracy"),
            f"official_evaluation.checkpoint_evaluation.{arm_name}.fast_macro",
        )
        if not 0.0 <= fast_macro <= 1.0:
            raise ValueError("official checkpoint fast accuracy must lie in [0, 1]")
        reading = arm.get("reading")
        if not isinstance(reading, dict):
            raise ValueError("official checkpoint Reading result is absent")
        for surface in ("eye_tracking", "self_paced"):
            _number(
                reading.get(surface),
                f"official_evaluation.checkpoint_evaluation.{arm_name}.{surface}",
            )
        aoa = arm.get("age_of_acquisition")
        if not isinstance(aoa, dict):
            raise ValueError("official checkpoint Age-of-Acquisition result is absent")
        curve_fitness = _number(
            aoa.get("curve_fitness"),
            f"official_evaluation.checkpoint_evaluation.{arm_name}.curve_fitness",
        )
        if not 0.0 <= curve_fitness <= 1.0:
            raise ValueError("official Age-of-Acquisition fitness must lie in [0, 1]")
        words = aoa.get("words")
        if isinstance(words, bool) or not isinstance(words, int) or words <= 0:
            raise ValueError("official Age-of-Acquisition word count is invalid")


def validate_marquee(payload: dict[str, Any], *, release: bool = False) -> None:
    if payload.get("schema_version") != 5:
        raise ValueError("unsupported marquee result schema")
    if payload.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("BabyLM campaign identity changed")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("BabyLM protocol identity changed")
    _validate_training(payload)
    _validate_evaluation(payload, release=release)
    expected_status = "complete"
    if payload.get("status") != expected_status:
        raise ValueError(f"marquee status must be {expected_status}")


def render_marquee(payload: dict[str, Any]) -> str:
    validate_marquee(payload)
    training = payload["training"]
    arms = training["arms"]
    fraction = float(training["terminal_gap"]["fraction_closed"])
    parameters = [float(arms[name]["trainable_parameters"]) for name in ARM_ORDER]
    parameter_range = f"{min(parameters) / 1e6:.0f}–{max(parameters) / 1e6:.0f}M"
    presentations = float(training["target_presentations_per_arm"]) / 1e9
    lines = [
        SECTION_HEADING,
        "",
        f"At {parameter_range} parameters and {presentations:.2f}B token presentations, residualization",
        f"closes **{fraction:.1%}** of native SDM's training-NLL gap to dense attention.",
        "",
        "![BabyLM training NLL for dense attention, native SDM, and residualized SDM.](figures/babylm-training.png)",
        "",
    ]
    if payload["evaluation"]["status"] == "complete":
        terminal = payload["evaluation"]["terminal"]
        tune = terminal["finetune"]["arms"]
        task_rows = [
            "| Evaluation | Task | Metric | Dense attention | Native SDM | Residualized SDM |",
            "|---|---|---|---:|---:|---:|",
        ]
        for evaluation, tasks in (
            ("Zero-shot", ZERO_SHOT_TASKS),
            ("Fine-tuned", FINETUNE_TASKS),
        ):
            evaluation_arms = terminal[
                "zero_shot" if evaluation == "Zero-shot" else "finetune"
            ]["arms"]
            for task in tasks:
                metric = (
                    "F1"
                    if evaluation == "Fine-tuned" and task in {"mrpc", "qqp"}
                    else "Accuracy"
                )
                score_key = (
                    "primary_metric_by_task"
                    if evaluation == "Fine-tuned"
                    else "by_task"
                )
                values = {
                    arm: float(evaluation_arms[arm][score_key][task])
                    for arm in ARM_ORDER
                }
                best = max(values.values())
                formatted = {
                    arm: (f"**{value:.2%}**" if value == best else f"{value:.2%}")
                    for arm, value in values.items()
                }
                task_rows.append(
                    f"| {evaluation} | {TASK_LABELS[task]} | {metric} | {formatted['dense']} | "
                    f"{formatted['native_sdm']} | {formatted['residualized_sdm']} |"
                )
        lines.extend(
            [
                "Residualized SDM improves all seven fine-tuning tasks over native SDM. Its",
                f"mean primary task metric reaches **{float(tune['residualized_sdm']['macro_primary_metric']):.2%}**, versus",
                f"{float(tune['native_sdm']['macro_primary_metric']):.2%} for native SDM and {float(tune['dense']['macro_primary_metric']):.2%} for dense attention.",
                "",
                "Zero-shot gains are mixed: four of six tasks improve over native SDM, while",
                "EWoK and Global PIQA decline.",
                "",
                "![Terminal zero-shot and fine-tuned BabyLM local validation summaries for dense attention, native SDM, and residualized SDM.](figures/babylm-evaluation.png)",
                "",
                "Local validation results:",
                "",
                *task_rows,
                "",
                "Global PIQA averages its two English splits. Fine-tuning reports the best",
                "validation score for each task.",
            ]
        )

    return "\n".join(lines)


def replace_marquee(readme: str, rendered: str) -> str:
    if readme.count(SECTION_HEADING) != 1 or readme.count(NEXT_HEADING) != 1:
        raise ValueError("README must contain exactly one BabyLM result section")
    before, rest = readme.split(SECTION_HEADING, 1)
    _old, after = rest.split(NEXT_HEADING, 1)
    return f"{before}{rendered}\n\n{NEXT_HEADING}{after}"
