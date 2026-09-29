"""Measure the quality/latency trade-off of sparse archive serving.

The script keeps model weights and tokens fixed, then compares approximate
decode settings against an exact ``archive_read_stride=1`` reference. It is a
diagnostic for the serving knobs, not a language-model benchmark.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcc_transformer import QCCForCausalLM


def run_decode(model: QCCForCausalLM, tokens: torch.Tensor) -> tuple[torch.Tensor, float]:
    model.eval()
    with torch.no_grad():
        model.reset_cache(tokens.shape[0])
        start = time.perf_counter()
        logits = torch.stack(
            [model.decode_step(tokens[:, index]) for index in range(tokens.shape[1])], dim=1
        )
        if tokens.is_cuda:
            torch.cuda.synchronize()
    return logits, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--num-codes", type=int, default=512)
    parser.add_argument("--active-codes", type=int, default=4)
    parser.add_argument("--strides", default="1,2,4,8")
    parser.add_argument(
        "--query-thresholds",
        default="",
        help="optional comma-separated adaptive query cosine thresholds",
    )
    parser.add_argument("--repeat-token", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.length <= args.window_size:
        raise ValueError("length must exceed window-size so the archive is exercised")
    device = torch.device(args.device)
    torch.manual_seed(123)
    common = dict(
        vocab_size=1024,
        d_model=128,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=args.length,
        window_size=args.window_size,
        num_codes=args.num_codes,
        active_codes=args.active_codes,
        lazy_decay=True,
        use_triton=False,
    )
    if args.repeat_token:
        tokens = torch.zeros((1, args.length), dtype=torch.long, device=device)
    else:
        tokens = torch.randint(0, common["vocab_size"], (1, args.length), device=device)
    reference = QCCForCausalLM(**common, archive_read_stride=1).to(device).eval()
    reference_logits, reference_time = run_decode(reference, tokens)
    print(f"device={device} length={args.length} window={args.window_size}")
    print(f"reference_stride=1 seconds={reference_time:.6f}")
    for raw_stride in args.strides.split(","):
        stride = int(raw_stride)
        if stride <= 0:
            raise ValueError("strides must be positive")
        candidate = QCCForCausalLM(**common, archive_read_stride=stride).to(device).eval()
        candidate.load_state_dict(reference.state_dict(), strict=True)
        logits, elapsed = run_decode(candidate, tokens)
        diff = logits - reference_logits
        mse = float(diff.square().mean())
        cosine = float(
            torch.nn.functional.cosine_similarity(
                logits.reshape(-1, logits.shape[-1]),
                reference_logits.reshape(-1, reference_logits.shape[-1]),
                dim=-1,
            ).mean()
        )
        print(
            f"stride={stride} seconds={elapsed:.6f} relative={reference_time / elapsed:.2f}x "
            f"logit_mse={mse:.6e} cosine={cosine:.6f}"
        )
    for raw_threshold in (value for value in args.query_thresholds.split(",") if value.strip()):
        threshold = float(raw_threshold)
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("query-thresholds values must be in [-1, 1]")
        candidate = QCCForCausalLM(
            **common,
            archive_read_stride=1,
            archive_query_cosine_threshold=threshold,
        ).to(device).eval()
        candidate.load_state_dict(reference.state_dict(), strict=True)
        logits, elapsed = run_decode(candidate, tokens)
        diff = logits - reference_logits
        mse = float(diff.square().mean())
        cosine = float(
            torch.nn.functional.cosine_similarity(
                logits.reshape(-1, logits.shape[-1]),
                reference_logits.reshape(-1, reference_logits.shape[-1]),
                dim=-1,
            ).mean()
        )
        print(
            f"query_threshold={threshold:g} seconds={elapsed:.6f} relative={reference_time / elapsed:.2f}x "
            f"logit_mse={mse:.6e} cosine={cosine:.6f}"
        )


if __name__ == "__main__":
    main()
