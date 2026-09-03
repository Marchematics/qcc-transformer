"""Matched latency smoke for a real Hugging Face causal LM.

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


def _measure(model, encoded, device: torch.device, steps: int) -> tuple[float, float]:
    model.eval()
    _sync(device)
    started = time.perf_counter()
    with torch.no_grad():
        output = model(**encoded, use_cache=True)
    _sync(device)
    ttft = time.perf_counter() - started
    token = output.logits[:, -1:].argmax(dim=-1)
    samples = []
    with torch.no_grad():
        for _ in range(steps):
            started = time.perf_counter()
            output = model(input_ids=token, use_cache=True)
            _sync(device)
            samples.append((time.perf_counter() - started) * 1000.0)
            token = output.logits[:, -1:].argmax(dim=-1)
    return ttft, float(sum(samples) / len(samples))


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
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true", help="load the real checkpoint through bitsandbytes NF4")
    parser.add_argument("--kv-head-policy", choices=("reject", "repeat"), default="reject")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if args.num_decode_steps <= 0:
        raise ValueError("num-decode-steps must be positive")
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
    baseline_ttft, baseline_tpot = _measure(baseline, encoded, model_device, args.num_decode_steps)
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
    )
    reset_hf_qcc_cache(patched, batch_size=1)
    patched_device = model_input_device(patched, device)
    patched_encoded = {key: value.to(patched_device) for key, value in encoded.items()}
    qcc_ttft, qcc_tpot = _measure(patched, patched_encoded, patched_device, args.num_decode_steps)
    result = {
        "model": args.model,
        "tokens": int(encoded["input_ids"].shape[-1]),
        "patched_layers": replaced,
        "baseline_full_kv": {"ttft_seconds": baseline_ttft, "tpot_ms": baseline_tpot},
        "qcc_retrofit": {"ttft_seconds": qcc_ttft, "tpot_ms": qcc_tpot},
        "speedup": {
            "ttft": baseline_ttft / max(qcc_ttft, 1e-12),
            "tpot": baseline_tpot / max(qcc_tpot, 1e-12),
        },
        "note": "Matched real-HF latency smoke; evaluate held-out RULER/LongBench before quality claims.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
