"""Evaluator task grouping and primary fine-tuning metrics."""
import math
from shared_residual_routing.results import ZERO_SHOT_TASKS, FINETUNE_TASKS


def primary_metric(task: str) -> str:
    if task not in FINETUNE_TASKS:
        raise ValueError(f"unknown fine-tuning task: {task}")
    return "f1" if task in ("mrpc", "qqp") else "accuracy"


def terminal_metrics(core: dict, ewok: dict) -> dict:
    tasks = core["zero_shot"]["tasks"]
    zero = {task: tasks[task]["accuracy"] for task in ZERO_SHOT_TASKS if task not in ("ewok", "global_piqa")}
    zero["ewok"] = ewok["ewok"]["accuracy"]
    zero["global_piqa"] = (tasks["global_piqa_parallel"]["accuracy"] + tasks["global_piqa_nonparallel"]["accuracy"]) / 2
    if set(core["finetune"]) != set(FINETUNE_TASKS):
        raise ValueError("terminal fine-tuning task matrix is incomplete")
    fine = {}
    for task in FINETUNE_TASKS:
        row = core["finetune"][task]
        metric = primary_metric(task)
        if row.get("selection_metric") != metric:
            raise ValueError(f"{task} primary metric changed")
        fine[task] = row["best_validation"][metric]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1
               for v in (*zero.values(), *fine.values())):
        raise ValueError("terminal task score is invalid")
    return {"zero_shot": {task: zero[task] for task in ZERO_SHOT_TASKS},
            "zero_shot_macro": sum(zero.values()) / 6, "finetune": fine,
            "finetune_macro_primary_metric": sum(fine.values()) / 7}
