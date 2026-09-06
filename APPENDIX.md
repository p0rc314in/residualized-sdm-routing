# Experimental details

## Router definition

The [construction](README.md#construction) uses the mean of the two native
random weight draws, with zero biases and residuals. Native weights use a
zero-centered truncated normal with standard-deviation parameter
`1 / sqrt(width)` and bounds at three times that value. Averaging two independent
draws halves their variance; this is the initialization used in these runs.

Native routing uses two affine maps. The residualized form uses a base plus two
residual maps, adding one affine map per SDM layer: 50% more trainable router
parameters, but a small increase in total model size. The effective routes are
still two affine maps. The comparison tests the common initialization and
additional trainable parameters together; it does not isolate their effects on
optimization. Frozen native maps remain in serialized checkpoints for
compatibility.

## Small matched comparisons

In topology labels, A is dense attention and B is SDM: A8 is eight dense
layers, B7A1 is seven SDM layers followed by dense attention, and B8 is eight
SDM layers. All five seed-0 arms have width 128 and eight layers. The WikiText
experiment uses GPT-2 tokenization, context 2,048, and protocol
`wikitext103-gpt2-causal-t2048-coverage-v1`. Each arm makes three complete
passes over 117,980,449 training targets, then scores every validation and test
transition once.

Adaptive Recall is a deterministic synthetic suite defined for this experiment.
Its 30 conditions combine pointer chasing over 32, 64, 128, or 192 associations
and 1, 2, 4, or 8 hops; span recall over 16 or 32 associations and spans of 1,
4, 8, or 16 tokens; and overwrite recall over 16 or 32 associations and 1, 2,
or 4 successive values. Protocol `adaptive-recall-seed102337-v1` runs 30,000
optimizer steps on a fixed stream with seed 102337, then evaluates 2,048
examples per condition. Exact-set accuracy is the fraction of examples for
which all 16 ordered query predictions match their corresponding labels. Query
loss averages cross-entropy over those outputs.

The packed stream uses uint32 input IDs from a 170,752-symbol vocabulary and
uint8 labels over 192 output classes; sequence length ranges from 32 to 528.
Training uses batch 32 and evaluates in batches of eight. Optimization is AdamW
at learning rate 3×10⁻⁴, betas 0.9 and 0.95, weight decay 0.01, gradient-norm
limit 1.0, and a 100-step linear warmup followed by a constant rate. Models use
four attention heads and BF16 activations.

Every SDM layer has one memory head, 1,024 logical rows, product-key initial
memory, and fixed R=W=8 access. Matched topology pairs share data order,
role-keyed initialization policy, optimizer, precision, schedule, and
evaluation. Exact aggregate values are stored once in
[`data/results.json`](data/results.json).

Every reported comparison uses one matched model seed (0); the gap-closure
percentages are matched-run estimates rather than multi-seed uncertainty
estimates.

The WikiText figure reports a trailing-1,000-step mean from step 2,000 onward,
so every displayed window falls after the 540-step warmup. It is sampled every
1,000 optimizer steps and at the terminal step, and derived from the complete
per-step `train_nll` series in
[`data/wikitext-training-curves.csv`](data/wikitext-training-curves.csv).

WikiText optimization uses AdamW with peak learning rate 3×10⁻⁴, 540 warmup
steps, cosine decay to 10% of peak, betas 0.9 and 0.95, weight decay 0.01,
gradient-norm limit 1.0, BF16 activations, FP32 optimizer state, and effective
batch eight.

## BabyLM scale comparison

The three-arm scale experiment uses protocol
`babylm2026-strict-gpt2-causal-t2048-10epoch-v1`: dense A16, native-SDM B16,
and residualized-SDM B16. All arms use width 512, 16 layers, eight attention
heads, seed 0, the complete official [BabyLM 2026 Strict corpus](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict),
and ten passes ending at step 102,852 and 1,685,114,880 target presentations.
The SDM arms use 16,384 logical rows per layer and balanced R=W=32 access.

Pretraining uses effective batch eight and AdamW at peak learning rate 3×10⁻⁴,
betas 0.9 and 0.95, weight decay 0.01, and gradient-norm limit 1.0. The learning
rate warms up for 2,572 steps, then follows cosine decay to 10% of peak. Models
train with BF16 activations.

The BabyLM curve samples the trailing-1,000-step mean training NLL every 5,000
steps and at the terminal step; it is not a held-out evaluation. Sampled values
and terminal checkpoint identities are in
[`data/marquee-results.json`](data/marquee-results.json).

## BabyLM evaluation

The evaluation uses the official [BabyLM 2026 evaluator](https://github.com/babylm-org/babylm-eval)
and [Strict evaluation data](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict-Evals),
with exact revisions recorded in [`provenance.json`](provenance.json). The
zero-shot summary is the unweighted mean of the six evaluator tasks reported in
the main note; Global PIQA is the mean of its two English splits. The terminal
EWoK row uses the complete
[`ewok-core-1.0`](https://huggingface.co/datasets/ewok-core/ewok-core-1.0)
evaluation rather than the fast subset.

Fine-tuning uses seed 42, maximum sequence length 512, and AdamW at peak
learning rate 3×10⁻⁵, betas 0.9 and 0.999, weight decay 0.01, and 6% cosine
warmup followed by decay to 10% of peak. The classifier uses dropout 0.1.
BoolQ and MultiRC use effective batch 16; the other five tasks use batch 32.
Each task runs for 10 epochs except WSC, which runs for 30. The best validation
checkpoint is selected after each epoch by accuracy, except MRPC and QQP use F1.
The local summary in the main note reports those same primary metrics; it is not
a BabyLM leaderboard score.

## Checkpoint evaluation

Reading and Age of Acquisition were evaluated at all 28 retained checkpoints
per arm. The optimized SDM inference path matched the exact training recurrence
on 2,048 examples and 4,096 candidates, with zero score difference and
identical predictions.

Reading reports normalized incremental R² beyond the evaluator's lexical and
control baseline. Age-of-Acquisition curve fitness is zero when the model–child
correlation misses the evaluator's p≤0.1 significance threshold, as it did for
all three arms. Terminal scores correspond to the 1,000M official-corpus
exposure checkpoint, or 1.69B GPT-2 target presentations. These auxiliary
evaluations show no consistent router advantage; their values are retained in
[`data/marquee-results.json`](data/marquee-results.json).
