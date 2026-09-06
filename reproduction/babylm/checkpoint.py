"""Reconstruct a checksum-bound BabyLM model and execute inference."""
from pathlib import Path
import torch
from .io import sha256_file
from .model import build_model, model_contract
from .spec import CAMPAIGN_ID, CANONICAL_SDM_COMMIT, MANIFEST_SHA256, SPEC


@torch.inference_mode()
def reconstruct(path: Path, digest: str, arm: str, *, smoke: bool = False, expected=None):
    if sha256_file(path) != digest:
        raise ValueError("BabyLM checkpoint checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    identity = {"arm": arm, "model_spec": SPEC.as_dict(), "dataset_manifest_sha256": MANIFEST_SHA256}
    if smoke:
        identity["kind"] = "babylm_smoke_checkpoint"
        if not 1 <= payload.get("step", 0) <= 3:
            raise ValueError("invalid BabyLM smoke step count")
    else:
        identity.update({"kind": "canonical_babylm_inference_checkpoint", "campaign_id": CAMPAIGN_ID,
                         "canonical_sdm_commit": CANONICAL_SDM_COMMIT, "step": 102852,
                         "training_tokens": 1685114880, "official_exposure_millions": 1000,
                         "tokenizer": {"implementation": "tiktoken", "encoding": "gpt2",
                                       "vocab_size": 50257, "eot_token": 50256}})
    for key, value in identity.items():
        if payload.get(key) != value:
            raise ValueError(f"BabyLM checkpoint identity changed at {key}")
    model = build_model(arm)
    model.load_state_dict(payload["model"], strict=True)
    contract = model_contract(arm, model)
    if not smoke and payload.get("model_contract") != contract:
        raise ValueError("BabyLM reconstructed architecture changed")
    model.cuda().bfloat16().eval()
    probe = torch.arange(32, device="cuda").view(1, -1)
    logits = model(probe, attn_impl="sdpa")
    if logits.shape != (1, 32, SPEC.vocab_size) or not torch.isfinite(logits).all():
        raise ValueError("BabyLM checkpoint inference produced invalid logits")
    if expected is not None:
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    del model, logits, payload
    torch.cuda.empty_cache()
