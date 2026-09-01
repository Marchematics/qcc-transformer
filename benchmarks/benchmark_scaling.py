"""Measure decode-time scaling with context length.

Example:
    python benchmarks/benchmark_scaling.py --lengths 256,512,1024,2048
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_decode import FullAttentionBaseline, timed_decode, timed_prefill
from qcc_transformer import QCCForCausalLM


def fit_log_slope(lengths: list[int], seconds: list[float]) -> float:
    x = torch.tensor([math.log(float(n)) for n in lengths])
    y = torch.tensor([math.log(max(t, 1e-12)) for t in seconds])
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    return float((centered_x * centered_y).sum() / (centered_x.square().sum()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="256,512,1024,2048")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    lengths = [int(value) for value in args.lengths.split(",") if value.strip()]
    qcc_seconds: list[float] = []
    full_seconds: list[float] = []
    for length in lengths:
        common = dict(
            vocab_size=4096,
            d_model=256,
            num_layers=2,
            num_heads=8,
            max_position_embeddings=max(length, 512),
            window_size=min(128, length - 1),
            num_codes=16,
        )
        tokens = torch.randint(0, common["vocab_size"], (1, length), device=device)
        qcc = QCCForCausalLM(**common).to(device)
        full = FullAttentionBaseline(**common).to(device)
        if args.mode == "prefill":
            qcc_seconds.append(timed_prefill(qcc, tokens, args.warmup, args.steps))
            full_seconds.append(timed_prefill(full, tokens, args.warmup, args.steps))
        else:
            qcc_seconds.append(timed_decode(qcc, tokens, args.warmup, args.steps))
            full_seconds.append(timed_decode(full, tokens, args.warmup, args.steps))
    print(f"device={device} mode={args.mode} lengths={lengths} threads={torch.get_num_threads()}")
    for length, qcc_time, full_time in zip(lengths, qcc_seconds, full_seconds):
        print(f"length={length} qcc_seconds={qcc_time:.6f} full_seconds={full_time:.6f} speedup={full_time / qcc_time:.2f}x")
    print(f"qcc_loglog_slope={fit_log_slope(lengths, qcc_seconds):.3f}")
    print(f"full_loglog_slope={fit_log_slope(lengths, full_seconds):.3f}")


if __name__ == "__main__":
    main()
