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
from qcc_transformer import compare_logits, patch_hf_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path")
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument(
        "--jsonl", type=Path, default=None,
        help="held-out JSONL with a 'text'/'prompt' field; each record is scored independently",
    )
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--num-codes", type=int, default=64)
    parser.add_argument("--max-position-embeddings", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--quality-gate", type=float, default=0.99)
    parser.add_argument("--kv-head-policy", choices=("reject", "repeat"), default="reject")
    args = parser.parse_args()
    if args.prompt_file is not None and args.jsonl is not None:
        raise SystemExit("use only one of --prompt-file and --jsonl")
    prompts = [args.prompt]
    if args.prompt_file is not None:
        args.prompt = args.prompt_file.read_text(encoding="utf-8")
        prompts = [args.prompt]
    if args.jsonl is not None:
        prompts = []
        for line in args.jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            value = record.get("text", record.get("prompt"))
            if not isinstance(value, str) or not value:
                raise ValueError("each JSONL record needs a non-empty 'text' or 'prompt'")
            prompts.append(value)
    if not prompts or any(not prompt for prompt in prompts):
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
    references = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            references.append((encoded, baseline(**encoded, use_cache=False).logits.cpu()))
    del baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()
    patched = AutoModelForCausalLM.from_pretrained(args.model, **common).to(device).eval()
    replaced = patch_hf_model(
        patched,
        window_size=args.window_size,
        num_codes=args.num_codes,
        max_position_embeddings=args.max_position_embeddings,
        kv_head_policy=args.kv_head_policy,
    )
    reports = []
    with torch.no_grad():
        for encoded, reference in references:
            candidate = patched(**encoded, use_cache=False).logits.cpu()
            reports.append(compare_logits(reference, candidate, quality_gate=args.quality_gate))
    cosine = sum(r.mean_logit_cosine for r in reports) / len(reports)
    top1 = sum(r.top1_agreement for r in reports) / len(reports)
    result = {
        "model": args.model,
        "tokens": int(sum(item[0]["input_ids"].shape[-1] for item in references)),
        "records": len(reports),
        "patched_layers": replaced,
        "mean_logit_cosine": float(cosine),
        "top1_agreement": float(top1),
        "quality_gate": args.quality_gate,
        "fidelity_passed": bool(cosine >= args.quality_gate and top1 >= args.quality_gate),
        "per_record": [r.as_dict() for r in reports],
        "note": "Matched Full-KV logit gate; task quality still requires held-out RULER/LongBench evaluation.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
