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


def _forward_logits(
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    *,
    chunk_size: int = 0,
    logits_to_keep: int = 0,
) -> torch.Tensor:
    """Run an exact causal forward, optionally in bounded cache-backed chunks.

    Some remote-code checkpoints only expose the eager attention implementation,
    whose full-prompt score matrix can exceed a small GPU even for an otherwise
    valid 8K evaluation.  Splitting the same request into cache-backed chunks
    preserves Full-KV semantics while bounding the temporary query-by-key matrix.
    """

    input_ids = encoded.get("input_ids")
    if input_ids is None or input_ids.ndim != 2:
        raise ValueError("chunked evaluation requires 2D input_ids")
    length = int(input_ids.shape[1])
    if chunk_size <= 0 or chunk_size >= length:
        kwargs: dict[str, object] = {"use_cache": False}
        if logits_to_keep:
            kwargs["logits_to_keep"] = logits_to_keep
        return model(**encoded, **kwargs).logits

    outputs: list[torch.Tensor] = []
    qcc_model = any(hasattr(module, "qcc") for module in model.modules())
    try:
        from transformers.cache_utils import DynamicCache

        cache = None if qcc_model else DynamicCache()
    except ImportError:  # pragma: no cover - Transformers is required by main()
        cache = None
    batch = int(input_ids.shape[0])
    for start in range(0, length, chunk_size):
        end = min(length, start + chunk_size)
        chunk: dict[str, torch.Tensor] = {}
        for key, value in encoded.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key == "attention_mask":
                # With no padding, Phi's own helper constructs the correctly
                # offset causal mask from the cache length.  Passing a growing
                # 2D mask triggers shape mismatches in older remote snapshots.
                continue
            elif value.ndim >= 2 and value.shape[1] == length:
                chunk[key] = value[:, start:end]
            else:
                chunk[key] = value
        if qcc_model:
            chunk["position_ids"] = torch.arange(
                start, end, device=input_ids.device, dtype=torch.long
            ).view(1, -1).expand(batch, -1)
        kwargs = {"use_cache": not qcc_model}
        if cache is not None:
            kwargs["past_key_values"] = cache
        result = model(**chunk, **kwargs)
        outputs.append(result.logits)
        if not qcc_model:
            cache = getattr(result, "past_key_values", None)
        if not qcc_model and cache is None:
            raise RuntimeError(
                "chunked evaluation requested, but the checkpoint returned no past_key_values"
            )
    logits = torch.cat(outputs, dim=1)
    return logits[:, -logits_to_keep:] if logits_to_keep else logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path")
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument(
        "--jsonl", type=Path, default=None,
        help="held-out JSONL with a 'text'/'prompt' field; each record is scored independently",
    )
    # This runner is primarily a matched quality diagnostic. Keep a larger
    # bounded local window by default; performance experiments should pass
    # their intended smaller window explicitly.
    parser.add_argument("--window-size", type=int, default=4096)
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
    parser.add_argument(
        "--archive-query-correction-rank",
        type=int,
        default=8,
        help="rank of the calibrated query-conditioned archive residual",
    )
    parser.add_argument("--max-position-embeddings", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true", help="load the real checkpoint through bitsandbytes NF4")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="auto",
        help="attention backend for the Full-KV reference model; SDPA avoids quadratic materialization on long prompts",
    )
    parser.add_argument(
        "--use-triton",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the Triton local-attention kernel in the retrofit; disable for a numerically conservative quality control",
    )
    parser.add_argument(
        "--local-attention-backend",
        choices=("sdpa", "eager"),
        default="sdpa",
        help="bounded local attention equation for the retrofit; eager matches HF fp32-softmax ordering",
    )
    parser.add_argument(
        "--forward-chunk-size",
        type=int,
        default=0,
        help="split long reference and retrofit forwards into exact HF cache-backed chunks (0 keeps one call)",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=None,
        help="internal QCC prefill chunk bound; defaults to the archive scan block",
    )
    parser.add_argument(
        "--logits-to-keep",
        type=int,
        default=0,
        help="retain only the final N logits when the checkpoint supports it (0 keeps the full sequence)",
    )
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
    parser.add_argument(
        "--quality-first",
        action="store_true",
        help="use the bounded score-ranked exact shadow with hard nearest reads for a quality-first run",
    )
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
    if args.archive_query_correction_rank < 0:
        raise ValueError("archive-query-correction-rank must be non-negative")
    if args.logits_to_keep < 0:
        raise ValueError("logits-to-keep must be non-negative")
    if args.forward_chunk_size < 0:
        raise ValueError("forward-chunk-size must be non-negative")
    if args.prefill_chunk_size is not None and args.prefill_chunk_size <= 0:
        raise ValueError("prefill-chunk-size must be positive when provided")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("install qcc-transformer[hf] to run this benchmark") from exc

    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    # Count the actual checkpoint from the Full-KV instance that is already
    # needed for the matched reference.  A separate metadata load creates an
    # avoidable CPU/GPU peak on small cards, especially for real 1--7B models.
    baseline = load_hf_causal_lm(
        args.model,
        dtype=dtype,
        device=device,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
        **({"attn_implementation": args.attn_implementation} if args.attn_implementation != "auto" else {}),
    )
    parameter_count = sum(parameter.numel() for parameter in baseline.parameters())
    model_device = model_input_device(baseline, device)
    references = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        with torch.no_grad():
            try:
                reference_logits = _forward_logits(
                    baseline,
                    encoded,
                    chunk_size=args.forward_chunk_size,
                    logits_to_keep=args.logits_to_keep,
                )
            except TypeError:
                if not args.logits_to_keep:
                    raise
                raise TypeError(
                    "the selected checkpoint does not support --logits-to-keep"
                )
            references.append((encoded, reference_logits.cpu()))
    del baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()
    patched = load_hf_causal_lm(
        args.model,
        dtype=dtype,
        device=device,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
        **({"attn_implementation": args.attn_implementation} if args.attn_implementation != "auto" else {}),
    )
    patch_kwargs = {
        "window_size": args.window_size,
        "num_codes": args.num_codes,
        "archive_kernel_features": args.archive_kernel_features,
        "archive_global_normalization": args.archive_global_normalization,
        "archive_position_invariant": args.archive_position_invariant,
        "archive_query_correction_rank": args.archive_query_correction_rank,
        "use_triton": args.use_triton,
        "local_attention_backend": args.local_attention_backend,
        "prefill_chunk_size": args.prefill_chunk_size,
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
                "quality_first": args.quality_first,
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
            try:
                candidate = _forward_logits(
                    patched,
                    patched_encoded,
                    chunk_size=args.forward_chunk_size,
                    logits_to_keep=args.logits_to_keep,
                ).cpu()
            except TypeError:
                if not args.logits_to_keep:
                    raise
                raise TypeError(
                    "the selected checkpoint does not support --logits-to-keep"
                )
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
        "archive_query_correction_rank": args.archive_query_correction_rank,
        "archive_query_scale_selector": args.archive_query_correction_rank > 0,
        "gate_bias_init": args.gate_bias_init,
        "archive_position_invariant": args.archive_position_invariant,
        "use_triton": args.use_triton,
        "local_attention_backend": args.local_attention_backend,
        "prefill_chunk_size": args.prefill_chunk_size,
        "mean_logit_cosine": float(cosine),
        "top1_agreement": float(top1),
        "quality_gate": args.quality_gate,
        "fidelity_passed": bool(cosine >= args.quality_gate and top1 >= args.quality_gate),
        "matched_full_kv": True,
        "real_model": True,
        "official": False,
        "synthetic": False,
        "qcc_only": False,
        "logits_to_keep": args.logits_to_keep or None,
        "forward_chunk_size": args.forward_chunk_size or None,
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "hybrid": args.hybrid,
        "quality_first": args.quality_first if args.hybrid else False,
        "exact_num_sets": args.exact_num_sets if args.hybrid else None,
        "exact_ways": args.exact_ways if args.hybrid else None,
        "exact_capacity_per_head": (
            args.exact_num_sets * args.exact_ways if args.hybrid else None
        ),
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
