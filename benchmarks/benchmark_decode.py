"""Quick CPU/GPU comparison of full causal attention and QCC streaming attention.

This is a microbenchmark, not a quality benchmark. Run with:
    python benchmarks/benchmark_decode.py --length 512 --warmup 2 --steps 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn

# Permit running this file directly from a fresh checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcc_transformer import QCCForCausalLM


class FullAttentionBaseline(QCCForCausalLM):
    """Same shell as QCC but with a local window equal to the sequence length."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs["use_archive"] = False
        kwargs["window_size"] = int(kwargs.get("max_position_embeddings", 4096))
        super().__init__(*args, **kwargs)


def timed_prefill(model: nn.Module, tokens: torch.Tensor, warmup: int, steps: int) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(tokens)
        if tokens.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            model(tokens)
        if tokens.is_cuda:
            torch.cuda.synchronize()
    return (time.perf_counter() - start) / steps


def timed_decode(model: QCCForCausalLM, tokens: torch.Tensor, warmup: int, steps: int) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model.reset_cache(tokens.shape[0])
            for t in range(tokens.shape[1]):
                model.decode_step(tokens[:, t])
        if tokens.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            model.reset_cache(tokens.shape[0])
            for t in range(tokens.shape[1]):
                model.decode_step(tokens[:, t])
        if tokens.is_cuda:
            torch.cuda.synchronize()
    return (time.perf_counter() - start) / steps


def timed_decode_chunk(
    model: QCCForCausalLM,
    tokens: torch.Tensor,
    warmup: int,
    steps: int,
    chunk_size: int,
) -> float:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model.reset_cache(tokens.shape[0])
            for start in range(0, tokens.shape[1], chunk_size):
                model.decode_chunk(tokens[:, start : start + chunk_size])
        if tokens.is_cuda:
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(steps):
            model.reset_cache(tokens.shape[0])
            for start in range(0, tokens.shape[1], chunk_size):
                model.decode_chunk(tokens[:, start : start + chunk_size])
        if tokens.is_cuda:
            torch.cuda.synchronize()
    return (time.perf_counter() - start_time) / steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--num-codes", type=int, default=16)
    parser.add_argument("--active-codes", type=int, default=None)
    parser.add_argument("--lazy-decay", action="store_true")
    parser.add_argument("--archive-read-stride", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    window_size = args.window_size or min(128, args.length - 1)
    if window_size <= 0:
        raise ValueError("window-size must be positive and smaller than length")
    common = dict(
        vocab_size=4096,
        d_model=256,
        num_layers=2,
        num_heads=8,
        max_position_embeddings=max(args.length, 512),
        window_size=window_size,
        num_codes=args.num_codes,
        active_codes=args.active_codes,
        lazy_decay=args.lazy_decay,
        archive_read_stride=args.archive_read_stride,
    )
    tokens = torch.randint(0, common["vocab_size"], (1, args.length), device=device)
    qcc = QCCForCausalLM(**common).to(device)
    full = FullAttentionBaseline(**common).to(device)
    if args.mode == "prefill":
        qcc_time = timed_prefill(qcc, tokens, args.warmup, args.steps)
        full_time = timed_prefill(full, tokens, args.warmup, args.steps)
    elif args.chunk_size == 1:
        qcc_time = timed_decode(qcc, tokens, args.warmup, args.steps)
        full_time = timed_decode(full, tokens, args.warmup, args.steps)
    else:
        qcc_time = timed_decode_chunk(qcc, tokens, args.warmup, args.steps, args.chunk_size)
        full_time = timed_decode_chunk(full, tokens, args.warmup, args.steps, args.chunk_size)
    print(f"device={device} length={args.length}")
    print(f"mode={args.mode} chunk_size={args.chunk_size} qcc_seconds={qcc_time:.4f} full_seconds={full_time:.4f} speedup={full_time / qcc_time:.2f}x")
    archive_elements = sum(
        layer.attention.archive.num_heads
        * layer.attention.archive.num_codes
        * layer.attention.archive.num_scales
        * (layer.attention.archive.head_dim + 1)
        for layer in qcc.layers
    )
    if args.lazy_decay:
        # Lazy decay carries one int64 logical timestamp per code/scale slot.
        archive_elements += sum(
            layer.attention.archive.num_heads
            * layer.attention.archive.num_codes
            * layer.attention.archive.num_scales
            for layer in qcc.layers
        )
    kv_heads = common["num_heads"]
    head_dim = common["d_model"] // kv_heads
    local_elements = 2 * len(qcc.layers) * kv_heads * common["window_size"] * head_dim
    full_elements = 2 * len(full.layers) * kv_heads * args.length * head_dim
    print(f"qcc_archive_elements_per_batch={archive_elements}")
    print(f"qcc_bounded_state_elements_per_batch={archive_elements + local_elements}")
    print(f"full_kv_cache_elements_per_batch={full_elements}")
    print(f"cache_reduction={full_elements / (archive_elements + local_elements):.2f}x")


if __name__ == "__main__":
    main()
