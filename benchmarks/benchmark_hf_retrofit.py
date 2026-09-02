"""Measure real-model logit fidelity before/after the HF QCC retrofit.

This benchmark deliberately loads two copies of the same Hugging Face model,
patches only one, and reports mean logit cosine plus top-1 agreement on the
same prompt.  It is a 99% *fidelity gate* for a retrofit smoke test, not a
replacement for RULER/LongBench task accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Permit running this file directly from a fresh checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcc_transformer import patch_hf_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path")
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--num-codes", type=int, default=64)
    parser.add_argument("--max-position-embeddings", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--quality-gate", type=float, default=0.99)
    args = parser.parse_args()
    if args.prompt_file is not None:
        args.prompt = args.prompt_file.read_text(encoding="utf-8")
    if not args.prompt:
        raise ValueError("prompt must be non-empty")
    if not 0.0 <= args.quality_gate <= 1.0:
        raise ValueError("quality-gate must lie in [0, 1]")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("install qcc-transformer[hf] to run this benchmark") from exc

    device = torch.device(args.device)
    common = {"trust_remote_code": args.trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **common)
    baseline = AutoModelForCausalLM.from_pretrained(args.model, **common).to(device).eval()
    patched = AutoModelForCausalLM.from_pretrained(args.model, **common).to(device).eval()
    replaced = patch_hf_model(
        patched,
        window_size=args.window_size,
        num_codes=args.num_codes,
        max_position_embeddings=args.max_position_embeddings,
    )
    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        baseline_logits = baseline(**encoded, use_cache=False).logits.float()
        patched_logits = patched(**encoded, use_cache=False).logits.float()
    cosine = torch.nn.functional.cosine_similarity(
        baseline_logits.reshape(-1, baseline_logits.shape[-1]),
        patched_logits.reshape(-1, patched_logits.shape[-1]),
        dim=-1,
    )
    top1 = (
        baseline_logits.argmax(dim=-1) == patched_logits.argmax(dim=-1)
    ).float()
    result = {
        "model": args.model,
        "tokens": int(encoded["input_ids"].shape[-1]),
        "patched_layers": replaced,
        "mean_logit_cosine": float(cosine.mean().item()),
        "top1_agreement": float(top1.mean().item()),
        "quality_gate": args.quality_gate,
        "fidelity_passed": bool(
            cosine.mean().item() >= args.quality_gate
            and top1.mean().item() >= args.quality_gate
        ),
        "note": "Logit fidelity smoke test only; task quality still requires matched RULER/LongBench evaluation.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
