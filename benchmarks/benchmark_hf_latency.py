"""Matched latency benchmark for a real Hugging Face causal LM.

The script measures one full prompt (TTFT/prefill) and steady one-token
decode (TPOT) for an unpatched Full-KV model and a QCC retrofit.  It is
deliberately model-agnostic and works with local snapshots, so a result can
be reproduced without relying on a synthetic QCC model.  Long-context claims
still require a checkpoint whose context window and hardware can support the
requested length.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcc_transformer import patch_hf_model, reset_hf_qcc_cache
from qcc_transformer.hf_loading import load_hf_causal_lm, model_input_device


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("latency sample list must not be empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summary(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
        "p99": _percentile(values, 99.0),
        "samples": values,
    }


def _measure(
    model,
    encoded,
    device: torch.device,
    steps: int,
    *,
    repeats: int = 1,
    warmup: int = 0,
) -> dict[str, object]:
    if repeats <= 0 or warmup < 0:
        raise ValueError("repeats must be positive and warmup must be non-negative")
    model.eval()
    qcc_model = False

    def reset_request() -> None:
        nonlocal qcc_model
        try:
            qcc_model = reset_hf_qcc_cache(
                model, batch_size=int(encoded["input_ids"].shape[0])
            ) > 0
        except ValueError:
            # The unpatched Full-KV model has no QCC state to reset.
            qcc_model = False

    def one_request() -> tuple[float, list[float]]:
        reset_request()
        _sync(device)
        started = time.perf_counter()
        with torch.no_grad():
            output = model(**encoded, use_cache=True)
        _sync(device)
        ttft = time.perf_counter() - started
        token = output.logits[:, -1:].argmax(dim=-1)
        physical_cache = None if qcc_model else getattr(output, "past_key_values", None)
        attention_mask = encoded.get("attention_mask")
        samples = []
        with torch.no_grad():
            for _ in range(steps):
                started = time.perf_counter()
                kwargs = {"input_ids": token, "use_cache": True}
                if physical_cache is not None:
                    kwargs["past_key_values"] = physical_cache
                    if attention_mask is not None:
                        attention_mask = torch.cat(
                            (attention_mask, torch.ones_like(token)), dim=-1
                        )
                        kwargs["attention_mask"] = attention_mask
                output = model(**kwargs)
                _sync(device)
                samples.append((time.perf_counter() - started) * 1000.0)
                token = output.logits[:, -1:].argmax(dim=-1)
                if physical_cache is not None:
                    physical_cache = getattr(output, "past_key_values", None)
                    if physical_cache is None:
                        raise RuntimeError("Full-KV model did not return past_key_values")
        return ttft, samples

    for _ in range(warmup):
        one_request()
    ttft_samples: list[float] = []
    tpot_samples: list[float] = []
    for _ in range(repeats):
        ttft, samples = one_request()
        ttft_samples.append(ttft)
        tpot_samples.extend(samples)
    ttft = _summary(ttft_samples)
    tpot = _summary(tpot_samples)
    return {
        "ttft_seconds": float(ttft["mean"]),
        "tpot_ms": float(tpot["mean"]),
        "ttft_p50_seconds": float(ttft["p50"]),
        "ttft_p95_seconds": float(ttft["p95"]),
        "ttft_p99_seconds": float(ttft["p99"]),
        "tpot_p50_ms": float(tpot["p50"]),
        "tpot_p95_ms": float(tpot["p95"]),
        "tpot_p99_ms": float(tpot["p99"]),
        "ttft_samples_seconds": ttft["samples"],
        "tpot_samples_ms": tpot["samples"],
        "repeats": repeats,
        "warmup": warmup,
        "decode_tokens": repeats * steps,
        "decode_tokens_per_second": 1000.0 / float(tpot["mean"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or local snapshot")
    parser.add_argument("--prompt", default="The quick brown fox " * 32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--num-codes", type=int, default=64)
    parser.add_argument(
        "--archive-position-invariant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use raw Q/K for archive addressing while local attention keeps RoPE",
    )
    parser.add_argument("--num-decode-steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1, help="measured requests for p50/p95/p99")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true", help="load the real checkpoint through bitsandbytes NF4")
    parser.add_argument("--kv-head-policy", choices=("reject", "repeat"), default="reject")
    parser.add_argument(
        "--use-triton", action=argparse.BooleanOptionalAction, default=True,
        help="use Triton local/archive kernels when available",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if args.num_decode_steps <= 0 or args.repeats <= 0 or args.warmup < 0:
        raise ValueError("num-decode-steps and repeats must be positive; warmup must be non-negative")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("install qcc-transformer[hf] to run this benchmark") from exc
    device = torch.device(args.device)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    common = {"trust_remote_code": args.trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **common)
    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True)
    baseline = load_hf_causal_lm(
        args.model,
        dtype=dtype,
        device=device,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
    )
    model_device = model_input_device(baseline, device)
    encoded = {key: value.to(model_device) for key, value in encoded.items()}
    baseline_result = _measure(
        baseline, encoded, model_device, args.num_decode_steps,
        repeats=args.repeats, warmup=args.warmup,
    )
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
    replaced = patch_hf_model(
        patched,
        window_size=args.window_size,
        num_codes=args.num_codes,
        archive_position_invariant=args.archive_position_invariant,
        kv_head_policy=args.kv_head_policy,
        use_triton=args.use_triton,
    )
    reset_hf_qcc_cache(patched, batch_size=1)
    patched_device = model_input_device(patched, device)
    patched_encoded = {key: value.to(patched_device) for key, value in encoded.items()}
    qcc_result = _measure(
        patched, patched_encoded, patched_device, args.num_decode_steps,
        repeats=args.repeats, warmup=args.warmup,
    )
    result = {
        "model": args.model,
        "tokens": int(encoded["input_ids"].shape[-1]),
        "patched_layers": replaced,
        "baseline_full_kv": baseline_result,
        "qcc_retrofit": qcc_result,
        "speedup": {
            "ttft": baseline_result["ttft_seconds"] / max(qcc_result["ttft_seconds"], 1e-12),
            "tpot": baseline_result["tpot_ms"] / max(qcc_result["tpot_ms"], 1e-12),
            "ttft_p95": baseline_result["ttft_p95_seconds"] / max(qcc_result["ttft_p95_seconds"], 1e-12),
            "ttft_p99": baseline_result["ttft_p99_seconds"] / max(qcc_result["ttft_p99_seconds"], 1e-12),
            "tpot_p95": baseline_result["tpot_p95_ms"] / max(qcc_result["tpot_p95_ms"], 1e-12),
            "tpot_p99": baseline_result["tpot_p99_ms"] / max(qcc_result["tpot_p99_ms"], 1e-12),
        },
        "note": "Matched real-HF latency measurement; p95/p99 are reported over the requested repeats.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
