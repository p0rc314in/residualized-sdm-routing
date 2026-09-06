"""Fixed five-arm Adaptive Recall reproduction identity."""
from reproduction.spec import ARMS, ARM_ORDER, GEOMETRY

PROTOCOL_ID = "adaptive-recall-seed102337-v1"
MANIFEST_SHA256 = "b0587d62c3ab709c94e37742892451a39463a7d577212b9379bd76d966f97800"
STEPS = 30_000
STREAM_SEED = 102337
BATCH_SIZE = 32
EVAL_EXAMPLES = 2048
EVAL_BATCH_SIZE = 8
PARAMETERS = dict(zip(ARM_ORDER, (2_138_496, 2_203_036, 2_212_256, 2_260_828, 2_278_304)))
# Absolute fractions, declared before executing any reproduction.
EXACT_SET_BANDS = dict(zip(ARM_ORDER, ((.53, .60), (.53, .60), (.41, .49), (.52, .59), (.52, .59))))
QUERY_LOSS_BANDS = dict(zip(ARM_ORDER, ((1.95, 2.25), (1.90, 2.20), (1.98, 2.29), (1.89, 2.19), (1.91, 2.22))))


def config(arm: str) -> dict:
    row = ARMS[arm]
    return {
        "benchmark": "adaptive_recall", "protocol_id": PROTOCOL_ID,
        "evidence_tier": "small_wikitext_plus_recall", "arm": arm,
        "model": row.profile.as_dict(), "router": row.router,
        "reads": GEOMETRY.reads, "writes": GEOMETRY.writes,
        "logical_rows": GEOMETRY.logical_capacity, "seed": 0,
        "steps": STEPS, "stream_seed": STREAM_SEED, "batch_size": BATCH_SIZE,
        "eval_examples": EVAL_EXAMPLES, "eval_batch_size": EVAL_BATCH_SIZE,
        "learning_rate": 3e-4, "warmup_steps": 100, "schedule": "warmup_constant",
        "adam_betas": [0.9, 0.95], "weight_decay": 0.01, "gradient_clip": 1.0,
        "activation_dtype": "bfloat16", "optimizer": "released_lingua_adamw",
    }
