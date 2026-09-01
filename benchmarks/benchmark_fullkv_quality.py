"""Compare QCC logits with a matched full-KV control.

This is a fidelity diagnostic, not a language-quality benchmark. Both models
share one checkpoint and one token stream; logits are compared chunk by chunk
so QCC does not need to materialize a second sequence of outputs.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcc_transformer import QCCForCausalLM


@torch.no_grad()
def compare(
    qcc: QCCForCausalLM,
    full: QCCForCausalLM,
    tokens: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[float, float]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    qcc.eval()
    full.eval()
    qcc.reset_cache(tokens.shape[0])
    full.reset_cache(tokens.shape[0])
    cosine_sum = 0.0
    mse_sum = 0.0
    count = 0
    for start in range(0, tokens.shape[1], chunk_size):
        end = min(tokens.shape[1], start + chunk_size)
        qcc_logits = qcc.decode_chunk(
            tokens[:, start:end], reset_cache=start == 0
        )
        full_logits = full.decode_chunk(
            tokens[:, start:end], reset_cache=start == 0
        )
        qcc_flat = qcc_logits.reshape(-1, qcc_logits.shape[-1])
        full_flat = full_logits.reshape(-1, full_logits.shape[-1])
        cosine_sum += float(F.cosine_similarity(qcc_flat, full_flat, dim=-1).sum())
        mse_sum += float((qcc_logits - full_logits).square().sum())
        count += qcc_flat.shape[0]
    return cosine_sum / count, mse_sum / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="64,128,256")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--num-codes", type=int, default=16)
    parser.add_argument(
        "--position-encoding",
        choices=("sinusoidal", "learned", "rope"),
        default="sinusoidal",
    )
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    parser.add_argument("--target-cosine", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0.0 <= args.target_cosine <= 1.0:
        raise ValueError("target-cosine must be in [0, 1]")
    if args.threads is not None:
        if args.threads <= 0:
            raise ValueError("threads must be positive")
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    for raw_length in args.lengths.split(","):
        length = int(raw_length)
        if length <= args.window_size:
            raise ValueError("each length must exceed window-size")
        common = dict(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.layers,
            num_heads=args.heads,
            max_position_embeddings=length + 1,
            window_size=args.window_size,
            num_codes=args.num_codes,
            position_encoding=args.position_encoding,
            rope_theta=args.rope_theta,
        )
        qcc = QCCForCausalLM(**common, use_archive=True).to(device)
        full = QCCForCausalLM(
            **{**common, "window_size": length + 1}, use_archive=False
        ).to(device)
        full.load_state_dict(qcc.state_dict(), strict=True)
        tokens = torch.randint(0, args.vocab_size, (args.batch, length), device=device)
        cosine, mse = compare(qcc, full, tokens, chunk_size=args.chunk_size)
        print(
            f"device={device} length={length} position_encoding={args.position_encoding} "
            f"mean_logit_cosine={cosine:.6f} per_logit_mse={mse:.6e} "
            f"target={args.target_cosine:.6f} passed={cosine >= args.target_cosine}"
        )


if __name__ == "__main__":
    main()
