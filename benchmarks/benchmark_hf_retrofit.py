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
from qcc_transformer import (
    compare_logits,
    load_retrofit_adapter,
    patch_hf_model,
    reset_hf_qcc_cache,
)
from qcc_transformer.hf_loading import load_hf_causal_lm, model_input_device
from qcc_transformer.hybrid_archive import load_hybrid_retrofit_adapter


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
    parser.add_argument(
        "--archive-kernel-features",
        action="store_true",
        help="use positive random-feature softmax kernel archive (reference path)",
    )
    parser.add_argument(
        "--archive-global-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="combine code/scale numerator and denominator before normalization",
    )
    parser.add_argument(
        "--archive-position-invariant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use raw (unrotated) Q/K for archive addressing while retaining rotary local attention",
    )
    parser.add_argument("--max-position-embeddings", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true", help="load the real checkpoint through bitsandbytes NF4")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--quality-gate", type=float, default=0.99)
    parser.add_argument("--kv-head-policy", choices=("reject", "repeat"), default="reject")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="optional QCC-only adapter checkpoint produced by calibrate_hf_retrofit.py",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="load --adapter with the fixed-capacity exact-tier hybrid archive",
    )
    parser.add_argument("--exact-num-sets", type=int, default=128)
    parser.add_argument("--exact-ways", type=int, default=4)
    parser.add_argument("--exact-probe-sets", type=int, default=None)
    parser.add_argument(
        "--hybrid-replacement-policy",
        choices=("score", "fifo"),
        default="score",
        help="bounded exact-tier replacement policy",
    )
    parser.add_argument("--hybrid-admission-threshold", type=float, default=0.0)
    parser.add_argument("--hybrid-admission-bias-init", type=float, default=-4.0)
    parser.add_argument("--hybrid-max-inserts-per-chunk", type=int, default=8)
    parser.add_argument("--hybrid-exact-mix-bias-init", type=float, default=-4.0)
    parser.add_argument("--hybrid-exact-confidence-threshold", type=float, default=0.60)
    parser.add_argument(
        "--gate-bias-init",
        type=float,
        default=2.0,
        help="initial local-path gate bias used when constructing the retrofit",
    )
    parser.add_argument("--run-id", default=None, help="provenance id copied into the JSON report")
    parser.add_argument("--output", type=Path, default=None, help="also write the JSON report to this path")
    args = parser.parse_args()
    if args.prompt_file is not None and args.jsonl is not None:
        raise SystemExit("use only one of --prompt-file and --jsonl")
    if args.hybrid and args.adapter is None:
        raise SystemExit("--hybrid requires --adapter")
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
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    common = {"trust_remote_code": args.trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **common)
    # Count the actual checkpoint from the Full-KV instance that is already
    # needed for the matched reference.  A separate metadata load creates an
    # avoidable CPU/GPU peak on small cards, especially for real 1--7B models.
    baseline = load_hf_causal_lm(
        args.model,
        dtype=dtype,
        device=device,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
    )
    parameter_count = sum(parameter.numel() for parameter in baseline.parameters())
    model_device = model_input_device(baseline, device)
    references = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        with torch.no_grad():
            references.append((encoded, baseline(**encoded, use_cache=False).logits.cpu()))
    del baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()
    patched = load_hf_causal_lm(
        args.model,
        dtype=dtype,
        device=device,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
    )
    patch_kwargs = {
        "window_size": args.window_size,
        "num_codes": args.num_codes,
        "archive_kernel_features": args.archive_kernel_features,
        "archive_global_normalization": args.archive_global_normalization,
        "archive_position_invariant": args.archive_position_invariant,
        "max_position_embeddings": args.max_position_embeddings,
        "kv_head_policy": args.kv_head_policy,
        "gate_bias_init": args.gate_bias_init,
    }
    if args.adapter is None:
        replaced = patch_hf_model(patched, **patch_kwargs)
    elif args.hybrid:
        replaced = load_hybrid_retrofit_adapter(
            patched,
            args.adapter,
            hybrid_kwargs={
                "exact_num_sets": args.exact_num_sets,
                "exact_ways": args.exact_ways,
                "exact_probe_sets": args.exact_probe_sets,
                "exact_replacement_policy": args.hybrid_replacement_policy,
                "admission_threshold": args.hybrid_admission_threshold,
                "admission_bias_init": args.hybrid_admission_bias_init,
                "max_inserts_per_chunk": args.hybrid_max_inserts_per_chunk,
                "exact_mix_bias_init": args.hybrid_exact_mix_bias_init,
                "exact_confidence_threshold": args.hybrid_exact_confidence_threshold,
            },
            **patch_kwargs,
        )
    else:
        replaced = load_retrofit_adapter(patched, args.adapter, **patch_kwargs)
    patched_device = model_input_device(patched, device)
    reports = []
    with torch.no_grad():
        for encoded, reference in references:
            # Each held-out prompt is an independent request.  QCC owns its
            # history outside the framework cache, so relying on the internal
            # counter here would append later prompts to the previous one and
            # corrupt the paired fidelity report.
            reset_hf_qcc_cache(
                patched, batch_size=int(encoded["input_ids"].shape[0])
            )
            patched_encoded = {key: value.to(patched_device) for key, value in encoded.items()}
            candidate = patched(**patched_encoded, use_cache=False).logits.cpu()
            reports.append(compare_logits(reference, candidate, quality_gate=args.quality_gate))
    cosine = sum(r.mean_logit_cosine for r in reports) / len(reports)
    top1 = sum(r.top1_agreement for r in reports) / len(reports)
    result = {
        "model": args.model,
        "parameter_count": parameter_count,
        "pretrained": True,
        "real_checkpoint": True,
        "model_id": str(args.model),
        "run_id": args.run_id,
        "tokens": int(sum(item[0]["input_ids"].shape[-1] for item in references)),
        "records": len(reports),
        "patched_layers": replaced,
        "window_size": args.window_size,
        "num_codes": args.num_codes,
        "archive_kernel_features": args.archive_kernel_features,
        "archive_global_normalization": args.archive_global_normalization,
        "gate_bias_init": args.gate_bias_init,
        "archive_position_invariant": args.archive_position_invariant,
        "mean_logit_cosine": float(cosine),
        "top1_agreement": float(top1),
        "quality_gate": args.quality_gate,
        "fidelity_passed": bool(cosine >= args.quality_gate and top1 >= args.quality_gate),
        "matched_full_kv": True,
        "real_model": True,
        "official": False,
        "synthetic": False,
        "qcc_only": False,
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "hybrid": args.hybrid,
        "exact_num_sets": args.exact_num_sets if args.hybrid else None,
        "exact_ways": args.exact_ways if args.hybrid else None,
        "exact_probe_sets": args.exact_probe_sets if args.hybrid else None,
        "hybrid_admission_threshold": (
            args.hybrid_admission_threshold if args.hybrid else None
        ),
        "hybrid_admission_bias_init": (
            args.hybrid_admission_bias_init if args.hybrid else None
        ),
        "hybrid_max_inserts_per_chunk": (
            args.hybrid_max_inserts_per_chunk if args.hybrid else None
        ),
        "hybrid_replacement_policy": (
            args.hybrid_replacement_policy if args.hybrid else None
        ),
        "hybrid_exact_mix_bias_init": (
            args.hybrid_exact_mix_bias_init if args.hybrid else None
        ),
        "hybrid_exact_confidence_threshold": (
            args.hybrid_exact_confidence_threshold if args.hybrid else None
        ),
        "per_record": [r.as_dict() for r in reports],
        "note": "Matched Full-KV logit gate; task quality still requires held-out RULER/LongBench evaluation.",
    }
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
