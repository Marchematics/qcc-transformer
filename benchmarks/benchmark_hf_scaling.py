#!/usr/bin/env python3
"""Measure real-HF QCC scaling at 128K/256K/512K/1M.

The script runs the same native-context checkpoint in Full-KV and QCC modes and
records failures rather than substituting a shorter context.  It is a measurement
runner; the production gate still requires every point to complete on both sides.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcc_transformer import load_hybrid_retrofit_adapter, load_retrofit_adapter, reset_hf_qcc_cache
from qcc_transformer.production_profile import enable_qkv_only_deployment_profile


def _native_context(config: Any) -> int | None:
    values = [getattr(config, name, None) for name in ("max_position_embeddings", "n_positions", "max_sequence_length")]
    values = [int(value) for value in values if isinstance(value, int) and value > 0]
    return max(values) if values else None


def _cache_bytes(cache: Any) -> int | None:
    tensors: list[torch.Tensor] = []
    seen: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            if id(value) not in seen:
                seen.add(id(value))
                tensors.append(value)
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
            return
        for attr in ("key_cache", "value_cache", "layers"):
            item = getattr(value, attr, None)
            if item is not None:
                visit(item)

    visit(cache)
    return sum(item.numel() * item.element_size() for item in tensors) if tensors else None


def _qcc_state_bytes(model: Any) -> int:
    total = 0
    for module in model.modules():
        qcc = getattr(module, "qcc", None)
        if qcc is None:
            continue
        for value in (getattr(qcc, "_local_key_cache", None), getattr(qcc, "_local_value_cache", None), getattr(qcc.archive, "_numerator", None), getattr(qcc.archive, "_denominator", None), getattr(qcc.archive, "_last_step", None)):
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
        exact_bytes = getattr(qcc.archive, "total_state_bytes", None)
        if callable(exact_bytes):
            # total_state_bytes includes the recurrent tensors; avoid double
            # counting those when a hybrid archive exposes its own accounting.
            recurrent = sum(value.numel() * value.element_size() for value in (qcc.archive._numerator, qcc.archive._denominator, qcc.archive._last_step))
            total += max(0, int(exact_bytes()) - recurrent)
    return total


def _token_stream(tokenizer: Any, total: int) -> torch.Tensor:
    seed = "Long-context scaling workload for QCC Transformer. "
    tokens = tokenizer(seed * max(1, total // 8 + 4), add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    if tokens.numel() < total:
        tokens = tokens.repeat((total + tokens.numel() - 1) // tokens.numel())
    return tokens[:total]


@torch.inference_mode()
def _run(model: Any, tokens: torch.Tensor, *, device: torch.device, chunk_size: int, qcc: bool, decode_steps: int) -> dict[str, Any]:
    if qcc:
        reset_hf_qcc_cache(model, batch_size=1)
    past = None
    last_logits = None
    started = time.perf_counter()
    try:
        for start in range(0, tokens.numel(), chunk_size):
            chunk = tokens[start : start + chunk_size].unsqueeze(0).to(device)
            positions = torch.arange(start, start + chunk.shape[1], device=device, dtype=torch.long).view(1, -1)
            output = model(input_ids=chunk, position_ids=positions, past_key_values=past, use_cache=True)
            last_logits = output.logits[:, -1]
            past = getattr(output, "past_key_values", None)
        prefill_s = time.perf_counter() - started
        if last_logits is None:
            raise RuntimeError("model produced no logits")
        token = last_logits.argmax(dim=-1, keepdim=True)
        decode_start = time.perf_counter()
        for _ in range(decode_steps):
            output = model(input_ids=token, past_key_values=past, use_cache=True)
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            past = getattr(output, "past_key_values", None)
        decode_s = time.perf_counter() - decode_start
        return {
            "status": "ok",
            "context_tokens": int(tokens.numel()),
            "prefill_seconds": prefill_s,
            "tpot_ms": decode_s * 1000.0 / max(1, decode_steps),
            "qcc_state_bytes": _qcc_state_bytes(model) if qcc else None,
            "full_kv_state_bytes": _cache_bytes(past) if not qcc else None,
        }
    except torch.cuda.OutOfMemoryError as exc:
        return {"status": "oom", "context_tokens": int(tokens.numel()), "error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        return {"status": "error", "context_tokens": int(tokens.numel()), "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--lengths", default="128000,256000,512000,1000000")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--num-codes", type=int, default=64)
    parser.add_argument("--hybrid", action="store_true", help="load a calibrated exact-tier hybrid adapter")
    parser.add_argument("--exact-num-sets", type=int, default=32)
    parser.add_argument("--exact-ways", type=int, default=4)
    parser.add_argument("--archive-mix", type=float, default=0.125)
    parser.add_argument("--kv-head-policy", choices=("reject", "repeat"), default="repeat")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.chunk_size <= 0 or args.decode_steps <= 0:
        raise ValueError("chunk-size and decode-steps must be positive")
    lengths = [int(item) for item in args.lengths.split(",") if item.strip()]
    if lengths != sorted(set(lengths)) or any(length < 128_000 for length in lengths):
        raise ValueError("scaling lengths must be sorted, unique, and >=128K")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("install qcc-transformer[hf]") from exc
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    baseline = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=args.trust_remote_code).to(device).eval()
    native = _native_context(baseline.config)
    if native is None or native < max(lengths):
        raise RuntimeError(f"model native context {native} is below requested scaling maximum {max(lengths)}")
    points: list[dict[str, Any]] = []
    for length in lengths:
        tokens = _token_stream(tokenizer, length)
        full = _run(baseline, tokens, device=device, chunk_size=args.chunk_size, qcc=False, decode_steps=args.decode_steps)
        del tokens
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        points.append({"context_tokens": length, "full_kv": full})
    del baseline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    qcc_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=args.trust_remote_code).to(device).eval()
    if args.hybrid:
        load_hybrid_retrofit_adapter(
            qcc_model,
            args.adapter,
            hybrid_kwargs={"exact_num_sets": args.exact_num_sets, "exact_ways": args.exact_ways},
            window_size=args.window_size,
            num_codes=args.num_codes,
            max_position_embeddings=max(lengths),
            archive_position_invariant=True,
            kv_head_policy=args.kv_head_policy,
        )
    else:
        load_retrofit_adapter(
            qcc_model,
            args.adapter,
            window_size=args.window_size,
            num_codes=args.num_codes,
            max_position_embeddings=max(lengths),
            archive_position_invariant=True,
            kv_head_policy=args.kv_head_policy,
        )
    enable_qkv_only_deployment_profile(qcc_model, archive_mix=args.archive_mix)
    for point in points:
        tokens = _token_stream(tokenizer, point["context_tokens"])
        point["qcc"] = _run(qcc_model, tokens, device=device, chunk_size=args.chunk_size, qcc=True, decode_steps=args.decode_steps)
        point["matched_full_kv"] = point["full_kv"].get("status") == "ok" and point["qcc"].get("status") == "ok"
        if point["matched_full_kv"]:
            point["qcc_state_bytes"] = point["qcc"].get("qcc_state_bytes")
            point["tpot_ms"] = point["qcc"].get("tpot_ms")
        else:
            point["measured"] = False
        point["measured"] = point.get("matched_full_kv") is True
    result = {
        "schema": "qcc-hf-scaling-v1",
        "run_id": args.run_id,
        "model_id": args.model,
        "real_model": True,
        "synthetic": False,
        "official": False,
        "protocol_locked": True,
        "qcc_only": False,
        "matched_full_kv": all(point.get("matched_full_kv") is True for point in points),
        "native_context_tokens": native,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
