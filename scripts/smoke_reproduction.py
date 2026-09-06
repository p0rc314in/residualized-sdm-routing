#!/usr/bin/env python3
"""Bounded construction/update/checkpoint tests; never emit reproduction evidence."""
from __future__ import annotations
import argparse
import gc
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("wikitext103", "adaptive_recall", "babylm"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA smoke requested but unavailable")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    runtime = "babylm_runtime" if args.benchmark == "babylm" else "runtime"
    sys.path.insert(0, str(root / "third_party" / runtime))
    torch.set_num_threads(2)
    if args.benchmark == "babylm":
        from reproduction.babylm.model import build_model
        from reproduction.babylm.spec import ARMS
        vocab = 50257
    elif args.benchmark == "adaptive_recall":
        from reproduction.train_recall import build_model
        from reproduction.recall_spec import ARMS
        vocab = 192
    else:
        from reproduction.model import build_language_model, initialize_role_keyed
        from reproduction.spec import ARMS, GEOMETRY
        from shared_residual_routing.router import install_residualized_routing
        vocab = 50257
        def build_model(arm):
            model = build_language_model(vocab_size=vocab, geometry=GEOMETRY, profile=ARMS[arm].profile)
            initialize_role_keyed(model, 0)
            if ARMS[arm].router == "residualized_read_write":
                install_residualized_routing(model)
            return model
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for arm in ARMS:
        model = build_model(arm)
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        sparse = any(kind == "sdm" for kind in model.layer_types)
        if sparse and args.device == "cpu":
            results.append({"arm": arm, "parameters": parameters, "status": "construction_passed",
                            "inference": "requires_cuda"})
            del model
            gc.collect()
            print(json.dumps(results[-1]), flush=True)
            continue
        dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
        model.to(device=args.device, dtype=dtype)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
        values = torch.arange(32, device=args.device).view(1, -1)
        logits = model(values, attn_impl="sdpa")
        loss = F.cross_entropy(logits.float().flatten(0, 1), (values + 1).remainder(vocab).flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            reference = model(values, attn_impl="sdpa")
        checkpoint = args.output / f"{arm}.smoke.pt"
        torch.save(model.state_dict(), checkpoint)
        del model, optimizer, logits, loss
        gc.collect()
        restored = build_model(arm)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"), strict=True)
        restored.to(device=args.device, dtype=dtype).eval()
        with torch.inference_mode():
            actual = restored(values, attn_impl="sdpa")
        if not torch.isfinite(actual).all():
            raise ValueError("checkpoint inference returned nonfinite logits")
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        results.append({"arm": arm, "parameters": parameters,
                        "status": "one_update_and_checkpoint_inference_passed", "sequence_length": 32})
        checkpoint.unlink()
        del restored, actual, reference
        gc.collect()
        print(json.dumps(results[-1]), flush=True)
    (args.output / "smoke.json").write_text(json.dumps({"benchmark": args.benchmark, "device": args.device,
                "purpose": "bounded_implementation_test", "arms": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
