# Third-party sources

## Sparse Delta Memory

The experiment derives from the official Sparse Delta Memory implementation.

- Paper: <https://arxiv.org/abs/2607.07386>
- Official repository: <https://github.com/facebookresearch/sparse-delta-memory>
- Official baseline revision: `183e7df809131b80ad4393741029d0f20fc3640b`
- Router-residualization and runtime lineage: `p0rc314in/sparse-delta-memory`
- Capacity-independent adapter lineage: `p0rc314in/capacity-independent-sdm-execution`
- SDM upstream license: CC BY-NC 4.0
- Meta Lingua base framework: BSD 3-Clause

`third_party/runtime/` contains the minimal released model/runtime source used
by the reproduction, plus the accepted capacity-independent selected-row
adapter needed to express the experiment's fixed logical SDM geometry without
allocating a dense logical memory table. Its `manifest.json` records the owning
repository and commit for each lineage step plus a hash of the complete
vendored tree; `scripts/verify_vendor.py` checks that tree directly. These
development commits are provenance, not retrieval dependencies. The manifest
also maps the official baseline hashes of the key model files to the vendored
files used here. The vendored source—not an external checkout—is the
implementation exercised by reproduction. No model weights or generated
datasets are redistributed.

`third_party/babylm_runtime/` contains the model and kernels from training
revision `61d29928aa7520f421e0bc39d02b4e5006ffd5a1`, including its tied
embeddings and checkpoint layout. It has a separate per-file checksum
inventory and retains the same SDM and Lingua licenses. The BabyLM commands
select this runtime in a separate Python process.

The repository's MIT license applies only to original note, experiment,
validation, and integration code. Sparse Delta Memory source remains under its
upstream CC BY-NC 4.0 license. Meta Lingua-derived files under the vendored
`apps/` and `lingua/` trees retain Meta's BSD 3-Clause notice in
`third_party/runtime/LICENSE-LINGUA`.

## WikiText-103

The reproduction will download WikiText-103 raw v1 from
`Salesforce/wikitext` at revision
`5fddba447aa4e75996922ea0d6b18b42f0a81cc4`. Dataset files are not
redistributed. The dataset card lists CC BY-SA and GFDL licensing.

## BabyLM

The scale experiment uses the official BabyLM 2026 Strict corpus at revision
`9e57baaaa91ac3c638746be14d1d5fa6c789f4cf`. Corpus files are not
redistributed; users remain responsible for the terms of the underlying
sources.

Evaluation preparation downloads the pinned official evaluation data and the
two English Global PIQA splits (CC BY-SA 4.0). Full EWoK requires the account
holder's prior acknowledgement of its dataset terms; it is prepared locally
and is not included in this repository. Reading and AoA aggregation uses a
separately retrieved official evaluator at the revision in `provenance.json`.
