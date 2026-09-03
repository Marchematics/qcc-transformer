"""Measure matched streaming prefill memory for a real HF checkpoint.

The benchmark feeds the same token stream in fixed chunks to Full-KV and the
QCC retrofit.  It records peak CUDA allocation, elapsed prefill time and
whether the run reached the requested length.  Full-KV OOM is a valid,
auditable outcome at long context; it is never silently dropped.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcc_transformer import patch_hf_model, reset_hf_qcc_cache
from qcc_transformer.hybrid_archive import load_hybrid_retrofit_adapter
from qcc_transformer.hf_loading import load_hf_causal_lm, model_input_device
from qcc_transformer.production_profile import enable_qkv_only_deployment_profile


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _token_stream(tokenizer, total_tokens: int) -> torch.Tensor:
    seed = "The quick brown fox jumps over the lazy dog. QCC streaming memory benchmark. "
    encoded = tokenizer(seed * max(1, total_tokens // 8 + 4), return_tensors="pt", add_special_tokens=True)
    tokens = encoded["input_ids"][0]
    if tokens.numel() < total_tokens:
        repeats = (total_tokens + tokens.numel() - 1) // tokens.numel()
        tokens = tokens.repeat(repeats)
    return tokens[:total_tokens]


def _full_kv_attention_state_bytes(model, context_tokens: int, batch_size: int, dtype: torch.dtype) -> int:
    config = model.config
    layers = int(getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0)))
    query_heads = int(getattr(config, "num_attention_heads", getattr(config, "n_head", 0)))
    kv_heads = int(getattr(config, "num_key_value_heads", query_heads))
    hidden = int(getattr(config, "hidden_size", getattr(config, "n_embd", 0)))
    if min(layers, query_heads, kv_heads, hidden, context_tokens, batch_size) <= 0:
        raise ValueError("model config does not expose positive attention geometry")
    if hidden % query_heads or query_heads % kv_heads:
        raise ValueError("model attention geometry is not compatible with GQA")
    head_dim = hidden // query_heads
    return 2 * layers * batch_size * context_tokens * kv_heads * head_dim * torch.empty(
        (), dtype=dtype
    ).element_size()


def _qcc_attention_state_bytes(model) -> int | None:
    total = 0
    layers = 0
    for module in model.modules():
        qcc = getattr(module, "qcc", None)
        if qcc is None or not hasattr(qcc, "archive"):
            continue
        layers += 1
        archive = qcc.archive
        if hasattr(archive, "total_state_bytes"):
            total += int(archive.total_state_bytes())
        else:
            for value in (
                getattr(archive, "_numerator", None),
                getattr(archive, "_denominator", None),
                getattr(archive, "_last_step", None),
            ):
                if isinstance(value, torch.Tensor):
                    total += value.numel() * value.element_size()
        for name in (
            "_local_key_cache",
            "_local_value_cache",
            "_archive_key_cache",
            "_lexical_key_cache",
            "_lexical_value_cache",
        ):
            value = getattr(qcc, name, None)
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
    return total if layers else None


def _native_context(config) -> int | None:
    values = [
        getattr(config, name, None)
        for name in ("max_position_embeddings", "n_positions", "max_sequence_length")
    ]
    values = [int(value) for value in values if isinstance(value, int) and value > 0]
    return max(values) if values else None


def _run_stream(
    model,
    tokens: torch.Tensor,
    device: torch.device,
    chunk_size: int,
    *,
    decode_steps: int = 0,
) -> dict:
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [batch, context]")
    model.eval()
    cache = None
    chunks = 0
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for start in range(0, tokens.shape[1], chunk_size):
                chunk = tokens[:, start : start + chunk_size].to(device)
                output = model(input_ids=chunk, past_key_values=cache, use_cache=True)
                cache = getattr(output, "past_key_values", None)
                chunks += 1
            tpot_samples: list[float] = []
            if decode_steps:
                token = output.logits[:, -1:].argmax(dim=-1)
                for _ in range(decode_steps):
                    _sync(device)
                    decode_started = time.perf_counter()
                    output = model(
                        input_ids=token,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    _sync(device)
                    tpot_samples.append((time.perf_counter() - decode_started) * 1000.0)
                    cache = getattr(output, "past_key_values", None)
                    token = output.logits[:, -1:].argmax(dim=-1)
        _sync(device)
        elapsed = time.perf_counter() - started
        peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        cache_length = None
        if cache is not None and hasattr(cache, "get_seq_length"):
            cache_length = int(cache.get_seq_length())
        return {
            "status": "ok",
            "tokens": int(tokens.numel()),
            "tokens_per_request": int(tokens.shape[1]),
            "batch_size": int(tokens.shape[0]),
            "chunks": chunks,
            "elapsed_seconds": elapsed,
            "tokens_per_second": tokens.numel() / max(elapsed, 1e-12),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "cache_length": cache_length,
            "decode_steps": decode_steps,
            "tpot_ms": sum(tpot_samples) / len(tpot_samples) if tpot_samples else None,
            "tpot_samples_ms": tpot_samples,
        }
    except torch.cuda.OutOfMemoryError as exc:
        return {
            "status": "oom",
            "tokens": int(tokens.numel()),
            "tokens_per_request": int(tokens.shape[1]),
            "batch_size": int(tokens.shape[0]),
            "chunks_completed": chunks,
            "error": f"{type(exc).__name__}: {exc}",
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--total-tokens", type=int, default=131072)
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="matched concurrent requests; multiply total work but retain this context per request",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--num-codes", type=int, default=16)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--exact-num-sets", type=int, default=128)
    parser.add_argument("--exact-ways", type=int, default=4)
    parser.add_argument("--exact-probe-sets", type=int, default=None)
    parser.add_argument("--archive-mix", type=float, default=0.125)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true", help="load the real checkpoint through bitsandbytes NF4")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--kv-head-policy", choices=("reject", "repeat"), default="repeat")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--min-native-context", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--protocol-locked",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="record that this custom workload was registered before execution",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.total_tokens <= 0 or args.chunk_size <= 0 or args.batch_size <= 0 or args.decode_steps < 0:
        raise ValueError("total-tokens/chunk-size/batch-size must be positive and decode-steps non-negative")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    common = {"trust_remote_code": args.trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **common)
    stream = _token_stream(tokenizer, args.total_tokens)
    # Make rows non-identical without changing context length or token count.
    # This prevents an accidental future optimization from treating this as a
    # shared-prefix workload when the goal is independent-request concurrency.
    tokens = torch.stack([stream.roll(shifts=index) for index in range(args.batch_size)])

    baseline = load_hf_causal_lm(
        args.model,
        dtype=dtype,
        device=device,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
    )
    baseline_device = model_input_device(baseline, device)
    native_context_tokens = _native_context(baseline.config)
    if args.min_native_context is not None and (
        native_context_tokens is None or native_context_tokens < args.min_native_context
    ):
        raise RuntimeError(
            f"model native context {native_context_tokens} is below requested "
            f"{args.min_native_context}"
        )
    baseline_result = _run_stream(
        baseline, tokens, baseline_device, args.chunk_size, decode_steps=args.decode_steps
    )
    baseline_result["attention_state_bytes"] = _full_kv_attention_state_bytes(
        baseline, args.total_tokens, args.batch_size, dtype
    )
    del baseline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    patched = load_hf_causal_lm(
        args.model,
        dtype=dtype,
        device=device,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
    )
    patched_device = model_input_device(patched, device)
    if args.adapter is None:
        replaced = patch_hf_model(
            patched,
            window_size=args.window_size,
            num_codes=args.num_codes,
            kv_head_policy=args.kv_head_policy,
        )
    else:
        replaced = load_hybrid_retrofit_adapter(
            patched,
            args.adapter,
            hybrid_kwargs={
                "exact_num_sets": args.exact_num_sets,
                "exact_ways": args.exact_ways,
                "exact_probe_sets": args.exact_probe_sets,
            },
            window_size=args.window_size,
            num_codes=args.num_codes,
            archive_position_invariant=True,
            kv_head_policy=args.kv_head_policy,
        )
        enable_qkv_only_deployment_profile(patched, archive_mix=args.archive_mix)
    reset_hf_qcc_cache(patched, batch_size=args.batch_size)
    qcc_result = _run_stream(
        patched, tokens, patched_device, args.chunk_size, decode_steps=args.decode_steps
    )
    qcc_result["attention_state_bytes"] = _qcc_attention_state_bytes(patched)
    result = {
        "model": args.model,
        "model_id": str(args.model),
        "run_id": args.run_id,
        "real_model": True,
        "synthetic": False,
        "protocol_locked": args.protocol_locked,
        "total_tokens": args.total_tokens,
        "native_context_tokens": native_context_tokens,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "window_size": args.window_size,
        "num_codes": args.num_codes,
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "exact_num_sets": args.exact_num_sets if args.adapter is not None else None,
        "exact_ways": args.exact_ways if args.adapter is not None else None,
        "exact_probe_sets": args.exact_probe_sets if args.adapter is not None else None,
        "patched_layers": replaced,
        "baseline_full_kv": baseline_result,
        "qcc_retrofit": qcc_result,
        "matched": True,
        "note": "Streaming memory/prefill diagnostic; combine with task quality and versioned vLLM evidence for 99 gate.",
    }
    measurable_memory = all(
        isinstance(result.get(field), (int, float))
        for result in (baseline_result, qcc_result)
        for field in ("peak_allocated_bytes", "peak_reserved_bytes")
    )
    if baseline_result.get("status") == "ok" and qcc_result.get("status") == "ok" and measurable_memory:
        result["derived"] = {
            "prefill_speedup": baseline_result["tokens_per_second"] / max(qcc_result["tokens_per_second"], 1e-12),
            "peak_allocated_reduction": 1.0 - qcc_result["peak_allocated_bytes"] / max(baseline_result["peak_allocated_bytes"], 1),
            "peak_reserved_reduction": 1.0 - qcc_result["peak_reserved_bytes"] / max(baseline_result["peak_reserved_bytes"], 1),
        }
    else:
        result["derived"] = {
            "prefill_speedup": None,
            "peak_allocated_reduction": None,
            "peak_reserved_reduction": None,
            "note": "A matched speed/memory ratio is undefined because one side did not complete.",
        }
        if baseline_result.get("status") == "ok" and qcc_result.get("status") == "ok":
            result["derived"]["note"] = "Peak CUDA memory is unavailable on non-CUDA device; only throughput is reported."
            result["derived"]["prefill_speedup"] = baseline_result["tokens_per_second"] / max(qcc_result["tokens_per_second"], 1e-12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
