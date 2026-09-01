"""Long-context serving harness for QCC state and latency measurements.

This script intentionally separates configuration/state accounting from an
optional end-to-end run.  A 1M-token run is expected to require a CUDA device;
the default ``--state-only`` mode is cheap and does not allocate a full token
prompt or Full-KV baseline.

Examples::

    python benchmarks/benchmark_long_context.py --length 128000 --state-only
    python benchmarks/benchmark_long_context.py --length 1000000 --device cuda \
        --chunk-size 256 --run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcc_transformer import QCCForCausalLM, count_archive_elements


def state_report(model: QCCForCausalLM, length: int, dtype_bytes: int = 4) -> dict[str, float]:
    """Return persistent QCC and hypothetical full-KV storage in bytes."""

    layers = len(model.layers)
    heads = model.layers[0].attention.num_heads
    head_dim = model.layers[0].attention.head_dim
    window = model.layers[0].attention.window_size
    qcc_elements = count_archive_elements(model) + 2 * layers * heads * window * head_dim
    full_elements = 2 * layers * heads * length * head_dim
    qcc_bytes = qcc_elements * dtype_bytes
    full_bytes = full_elements * dtype_bytes
    return {
        "qcc_elements": float(qcc_elements),
        "full_kv_elements": float(full_elements),
        "qcc_bytes": float(qcc_bytes),
        "full_kv_bytes": float(full_bytes),
        "state_fraction_percent": 100.0 * qcc_bytes / max(full_bytes, 1),
        "reduction": full_bytes / max(qcc_bytes, 1),
    }


@torch.no_grad()
def run_stream(
    model: QCCForCausalLM,
    length: int,
    chunk_size: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[float, float, int]:
    """Prefill a synthetic stream and decode one next token.

    Tokens are generated chunk-by-chunk so the benchmark does not allocate a
    second copy of a million-token prompt.  Returned values are prefill
    seconds, one-token TPOT in milliseconds, and the number of processed
    tokens.
    """

    if length <= 0 or chunk_size <= 0:
        raise ValueError("length and chunk_size must be positive")
    model.eval()
    model.reset_cache(batch_size=1)
    remaining = length
    first = True
    start = time.perf_counter()
    while remaining:
        current = min(chunk_size, remaining)
        tokens = torch.randint(0, vocab_size, (1, current), device=device)
        model.decode_chunk(tokens, reset_cache=first)
        first = False
        remaining -= current
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prefill_seconds = time.perf_counter() - start
    next_token = torch.zeros(1, dtype=torch.long, device=device)
    start = time.perf_counter()
    model.decode_step(next_token)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    tpot_ms = (time.perf_counter() - start) * 1000.0
    return prefill_seconds, tpot_ms, length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=128_000)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--num-codes", type=int, default=16)
    parser.add_argument("--archive-scan-block-size", type=int, default=256)
    parser.add_argument(
        "--position-encoding",
        choices=("sinusoidal", "learned", "rope"),
        default="sinusoidal",
    )
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run", action="store_true", help="process the full synthetic stream")
    parser.add_argument("--state-only", action="store_true", help="only print storage accounting")
    args = parser.parse_args()
    if args.length <= args.window_size:
        raise ValueError("length must exceed window-size")
    if args.state_only and args.run:
        raise ValueError("choose at most one of --state-only and --run")
    if not args.state_only and not args.run:
        args.state_only = True
    device = torch.device(args.device)
    model = QCCForCausalLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        # Reserve one extra position for the post-prefill TPOT probe.
        max_position_embeddings=args.length + 1,
        window_size=args.window_size,
        num_codes=args.num_codes,
        archive_scan_block_size=args.archive_scan_block_size,
        position_encoding=args.position_encoding,
        rope_theta=args.rope_theta,
    ).to(device)
    report = state_report(model, args.length)
    print(
        f"device={device} length={args.length} position_encoding={model.position_encoding} "
        f"state_only={args.state_only}"
    )
    print(
        f"qcc_state_bytes={int(report['qcc_bytes'])} full_kv_bytes={int(report['full_kv_bytes'])} "
        f"state_fraction={report['state_fraction_percent']:.6f}% reduction={report['reduction']:.2f}x"
    )
    if args.run:
        prefill, tpot, processed = run_stream(
            model, args.length, args.chunk_size, args.vocab_size, device
        )
        print(
            f"processed_tokens={processed} prefill_seconds={prefill:.6f} "
            f"tpot_ms={tpot:.3f}"
        )


if __name__ == "__main__":
    main()
