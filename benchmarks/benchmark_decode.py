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


def timed(model: nn.Module, tokens: torch.Tensor, warmup: int, steps: int) -> float:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    common = dict(
        vocab_size=4096,
        d_model=256,
        num_layers=2,
        num_heads=8,
        max_position_embeddings=max(args.length, 512),
        window_size=min(128, args.length - 1),
        num_codes=16,
    )
    tokens = torch.randint(0, common["vocab_size"], (1, args.length), device=device)
    qcc = QCCForCausalLM(**common).to(device)
    full = FullAttentionBaseline(**common).to(device)
    qcc_time = timed(qcc, tokens, args.warmup, args.steps)
    full_time = timed(full, tokens, args.warmup, args.steps)
    print(f"device={device} length={args.length}")
    print(f"qcc_seconds={qcc_time:.4f} full_seconds={full_time:.4f} speedup={full_time / qcc_time:.2f}x")
    print(f"qcc_archive_elements_per_batch={sum(layer.attention.archive.num_heads * layer.attention.archive.num_codes * layer.attention.archive.num_scales * (layer.attention.archive.head_dim + 1) for layer in qcc.layers)}")


if __name__ == "__main__":
    main()
