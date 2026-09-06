# Reproducing the experiments

The reproduction command prepares the pinned public WikiText-103 stream, trains
and evaluates the five small-model arms, checks the fresh measurements, and
retains a terminal model checkpoint for each arm.

## Requirements

- Linux x86-64 with Python 3.12 or 3.13
- a CUDA 12.8 development environment supported by PyTorch 2.11
- one or more NVIDIA GPUs with BF16 support
- enough local space for the prepared WikiText data, five terminal model
  checkpoints, and recovery checkpoints

The heaviest observed historical arms use about 2.1 GB of allocated device
memory. Budget more for CUDA extension compilation and allocator overhead.

On a comparable recent NVIDIA GPU, allow roughly 30 hours for the five arms.
Hardware and extension compilation materially affect this estimate. The
command prints measured throughput and ETA while running.

## Run from a clean checkout

Install the locked environment:

```bash
uv sync --frozen
```

Prepare the exact public-derived stream locally. This downloads WikiText-103 at
the pinned dataset revision, serializes and tokenizes it twice, and rejects the
build unless both complete inventories are byte-identical and match the
declared manifest and payload identities:

```bash
uv run --no-sync ./reproduce.sh prepare
```

Preparation writes under `runs/data/`; allow roughly 3 GB of temporary disk
space. The immutable token-and-index payload used by training is about 227 MB.

Run all five arms in the declared order:

```bash
uv run --no-sync ./reproduce.sh train
```

The combined entry point performs preparation, training, and fresh-result
checks:

```bash
uv run --no-sync ./reproduce.sh all
```

The exact arms are:

In layout labels, A denotes dense attention and B denotes SDM. Thus A8 is eight
dense layers, B7A1 is seven SDM layers followed by dense attention, and B8 is
eight SDM layers.

| Arm ID | Layout | Router | Micro-batch |
|---|---|---|---:|
| `dense_attention` | A8 | — | 8 |
| `b7a1_native_sdm` | B7A1 | independent | 1 |
| `b8_native_sdm` | B8 | independent | 1 |
| `b7a1_residualized_sdm` | B7A1 | residualized | 1 |
| `b8_residualized_sdm` | B8 | residualized | 1 |

All SDM arms use 1,024 logical rows per layer and balanced R=W=8 access.

## Outputs and acceptance

Each arm writes under `runs/reproduction/<arm>/`:

- `config.json`: resolved model, data, optimizer, runtime, and hardware identity;
- `training_curve.jsonl` and `heartbeat.json`: measured progress;
- `validation-step-*.json`: full validation evaluations;
- `checkpoints/recovery-step-*.pt`: resumable model, optimizer, and RNG state;
- `terminal-model.pt`: the terminal inference checkpoint;
- `coverage.json` and `result.json`: complete coverage and terminal metrics.

Interrupted arms resume only from a checkpoint whose SHA-256 matches
`LATEST_RECOVERY.json`. Completed results are never overwritten.

Fresh terminal metrics must fall in these predeclared NLL bands:

| Arm | Validation | Test |
|---|---:|---:|
| Dense | 4.32–4.40 | 4.32–4.40 |
| 7:1 native | 4.40–4.49 | 4.40–4.49 |
| 8 native | 4.43–4.53 | 4.43–4.53 |
| 7:1 residualized | 4.31–4.40 | 4.31–4.40 |
| 8 residualized | 4.33–4.42 | 4.33–4.42 |

After all five arms finish, `scripts/check_reproduction.py` also requires:

- the exact arm matrix, topology, router, data identity, and 21,603-step
  coverage;
- a present, hash-matching terminal model checkpoint for every arm;
- lower test NLL for residualized routing than native routing in both B7A1 and B8;
- 75–120% of the B7A1 native gap and 65–105% of the B8 native gap closed.

It writes `runs/reproduction/measurements.json` and `measurements.csv` from the
fresh outputs. These are reproduction measurements, not replacements for the
historical values quoted in the note.

The separate `./reproduce.sh verify-results` command audits the arithmetic of
the compact recorded values and the generated BabyLM README section; it does
not run the experiment.

Adaptive Recall and BabyLM have separate staged commands below. These
implementations have passed bounded local checks; the full commands have not
been validated end to end. Their reported experiment metrics come from the
original runs.

## Pinned identities

- protocol: `wikitext103-gpt2-causal-t2048-coverage-v1`
- dataset: [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext),
  `wikitext-103-raw-v1`, revision
  `5fddba447aa4e75996922ea0d6b18b42f0a81cc4`
- tokenizer: `tiktoken:gpt2`
- prepared manifest SHA-256:
  `fc4ef13cbc38070f2d7774dffbfd5be48cab31fe45d6d9995d522fc3bac1dde6`
- prepared remote-payload SHA-256:
  `f430ce52a43a44b88f5a8ec1ec5882866daaa568595bfbd6b765e3369586f85e`

The training schedule is three complete passes, 21,603 optimizer steps, and
353,941,347 target presentations. Optimization is AdamW at peak learning rate
3×10⁻⁴, 540 linear warmup steps, cosine decay to 10% of peak, betas 0.9 and
0.95, weight decay 0.01, gradient-norm limit 1.0, BF16 activations, FP32 master
parameters and optimizer state, and effective batch eight.

## Adaptive Recall

Prepare the frozen synthetic stream locally, then run the five arms and check
their fresh results:

```bash
uv run --no-sync ./reproduce.sh recall prepare
uv run --no-sync ./reproduce.sh recall train --gpus 0
uv run --no-sync ./reproduce.sh recall check
```

Preparation generates the complete stream twice. Both builds must match
manifest `b0587d62c3ab709c94e37742892451a39463a7d577212b9379bd76d966f97800`,
including every token, label, condition, and offset array. Training consumes
these records without generating tasks on the GPU host.

The five arms use the same topologies as WikiText, width 128, R=W=8, seed 0,
30,000 updates, batch 32, and the semantic embedding described in the appendix.
Terminal validation and test each score 2,048 examples in every condition.
Outputs under `runs/reproduction/adaptive_recall/<arm>/` include the full
training curve, per-condition predictions and metrics, recovery checkpoints,
and a terminal model. The verifier reloads each model for inference, checks
prediction accuracy against the frozen labels, and writes fresh measurements.

Predeclared test exact-set accuracy bands, in arm order, are 53–60%, 53–60%,
41–49%, 52–59%, and 52–59%. The all-SDM residualized arm must close 65–120%
of the native-to-dense gap. Query-loss bands are in
`reproduction/recall_spec.py`.

Use a BF16-capable NVIDIA GPU; an RTX 4090-class device is sufficient for the
historical small-model geometry. Packaged runtime and peak memory have not
yet been measured on CUDA. Estimate cost from measured steps per second:
`5 × 30,000 / steps_per_second / 3,600 × hourly_GPU_price`, plus evaluation.
Multiple comma-separated GPU indices run different arms concurrently, one arm
per device.

## BabyLM

Install the additional locked preparation and statistical-evaluation dependencies:

```bash
uv sync --frozen --extra babylm-eval
```

Retrieve the official evaluator at revision
`6f825c291e2c4c78ad33b1935fd64d45f52642dc` into `runs/babylm-eval`:

```bash
git clone https://github.com/babylm-org/babylm-eval.git runs/babylm-eval
git -C runs/babylm-eval checkout 6f825c291e2c4c78ad33b1935fd64d45f52642dc
uv run --no-sync ./reproduce.sh babylm prepare --evaluator-root runs/babylm-eval
```

Full EWoK preparation requires a Hugging Face account with access already
granted to `ewok-core/ewok-core-1.0`. Preparation checks that surface first and
stops if access is unavailable. It also regenerates the complete training
corpus, terminal evaluation records, and checkpoint-evaluation records twice,
checking their recorded byte identities. No private source checkout or
prepacked archive is required. All inputs are prepared before GPU execution.

The explicit GPU stages are:

```bash
uv run --no-sync ./reproduce.sh babylm train --gpus 0,1,2
uv run --no-sync ./reproduce.sh babylm evaluate --gpus 0,1,2
uv run --no-sync ./reproduce.sh babylm evaluate-checkpoints --gpus 0,1,2
uv run --no-sync ./reproduce.sh babylm score-human --evaluator-root runs/babylm-eval
uv run --no-sync ./reproduce.sh babylm check
```

The three arms are `a16_dense_attention`, `b16_native_sdm`, and
`b16_residualized_sdm`. They use the recorded width-512, tied-embedding model,
R=W=32, 102,852 updates, and all 28 official exposure checkpoints. Each arm's
files live below `runs/reproduction/babylm/` using the original campaign and
checkpoint arm identifiers. `training_curve.jsonl` contains every fresh step;
`RESULT.json` records training completion; the terminal and checkpoint
evaluation directories retain predictions and per-task scores. The verifier
requires all training and evaluation stages and reloads the exact terminal
checkpoints for inference before writing `measurements.json`.

The terminal zero-shot macro has six tasks, with the two English Global PIQA
splits averaged together and complete EWoK replacing the fast checkpoint
subset. Fine-tuning uses F1 for MRPC/QQP and accuracy for the other five tasks.

The declared terminal training-NLL bands are 2.60–2.68 for dense, 2.69–2.78
for native SDM, and 2.60–2.69 for residualized SDM. The residualized model must
close 70–120% of the native loss gap and exceed both fine-tuning macros.
Absolute task-macro tolerance bands are in `scripts/check_babylm_reproduction.py`.

The historical training used H100 80GB GPUs. Allow a multi-day GPU budget for
three-arm training plus all checkpoint evaluations; the packaged implementation
has not yet been timed on CUDA. Training cost is
`3 × 102,852 / steps_per_second / 3,600 × hourly_GPU_price`, plus evaluation.
Measure the bounded startup commands below before estimating a full run.

## Bounded implementation checks

These commands perform one optimizer update per arm on the prepared streams,
then reload a smoke checkpoint. They write under `runs/smoke/`; they cannot
produce accepted full reproduction measurements:

```bash
uv run --no-sync ./reproduce.sh recall smoke
uv run --no-sync ./reproduce.sh babylm smoke
```

`--smoke-steps` accepts only 1, 2, or 3. Recall smoke evaluation uses one
example per condition; BabyLM smoke performs only training startup and a
checkpoint inference check. Linux CPU construction tests are available through
`scripts/smoke_reproduction.py`: dense models also perform one update and an
exact save/reload inference comparison; sparse inference requires CUDA.

CUDA extensions must be available before using a paid experiment host. Build
them on a separate development machine or bake them into the chosen immutable
runtime image; do not use the experiment GPU for dependency installation or
extension builds. No cloud resources are allocated by these commands.
