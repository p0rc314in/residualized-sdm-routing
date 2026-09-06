"""Canonical matched model construction."""

from __future__ import annotations

from typing import Iterator

import torch
from torch import nn

from .spec import SPEC, validate_arm


def iter_sdm_mixers(model: nn.Module) -> Iterator[nn.Module]:
    for kind, layer in zip(model.layer_types, model.layers, strict=True):
        if kind == "sdm":
            yield layer.attention


def build_model(arm: str, seed: int = SPEC.seed) -> nn.Module:
    from apps.main.transformer import LMTransformer, LMTransformerArgs
    from lingua.sparse_delta_memory import SparseDeltaMemoryArgs

    validate_arm(arm)
    dense = arm == "dense_a16"
    shared = arm == "sdm_b16_shared"
    sdm_args = SparseDeltaMemoryArgs(
        dim=SPEC.width,
        num_heads=SPEC.memory_heads,
        slots_per_head=SPEC.logical_rows,
        num_reads=SPEC.reads,
        num_writes=SPEC.writes,
        memory_block_size=SPEC.memory_block_size,
        read_act="Softmax",
        write_act="Softmax",
        normalize_readings=True,
        backprop_on_memory=True,
        output_gate=True,
        compact_execution=True,
        initial_memory_mode="product_key",
        shared_residual_router=shared,
        initialization_seed=seed,
    )
    args = LMTransformerArgs(
        dim=SPEC.width,
        n_layers=SPEC.layers,
        head_dim=SPEC.width // SPEC.attention_heads,
        n_heads=SPEC.attention_heads,
        n_kv_heads=SPEC.attention_heads,
        multiple_of=256,
        norm_eps=1e-5,
        init_std_factor="disabled",
        max_seqlen=SPEC.context_length,
        attn_at="all" if dense else [],
        swa_at=[],
        sdm_at=[] if dense else "all",
        sdm_args=sdm_args,
        vocab_size=SPEC.vocab_size,
        weight_tying=True,
        seed=seed,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = LMTransformer(args)
        model.init_weights()
    return model


def model_contract(arm: str, model: nn.Module) -> dict[str, object]:
    from lingua.sparse_delta_memory import SparseDeltaMemory

    mixers = list(iter_sdm_mixers(model))
    if arm == "dense_a16":
        if mixers or model.layer_types != ["attn"] * SPEC.layers:
            raise RuntimeError("dense topology changed")
    else:
        if len(mixers) != SPEC.layers or model.layer_types != ["sdm"] * SPEC.layers:
            raise RuntimeError("SDM topology changed")
        for mixer in mixers:
            if not isinstance(mixer, SparseDeltaMemory):
                raise RuntimeError("noncanonical SDM layer constructed")
            if mixer.args.num_reads != SPEC.reads or mixer.args.num_writes != SPEC.writes:
                raise RuntimeError("balanced access geometry changed")
            if mixer.args.slots_per_head != SPEC.logical_rows:
                raise RuntimeError("logical bank geometry changed")
            if mixer.initial_memory_mode != "product_key":
                raise RuntimeError("product-key initial memory is inactive")
            if not mixer.args.compact_execution:
                raise RuntimeError("compact execution is inactive")
            if (mixer.shared_router is not None) != (arm == "sdm_b16_shared"):
                raise RuntimeError("shared router configuration changed")
    return {
        "topology": "A16" if arm == "dense_a16" else "B16",
        "layers": SPEC.layers,
        "width": SPEC.width,
        "sdm_layers": len(mixers),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "sdm_geometry": None
        if arm == "dense_a16"
        else {
            "factors": 2,
            "codebook_size": SPEC.codebook_size,
            "logical_rows": SPEC.logical_rows,
            "reads": SPEC.reads,
            "writes": SPEC.writes,
        },
    }
