"""Train/evaluate the learned-landmark archive against a direct teacher.

This isolates memory approximation quality from language-model training. It is
deliberately small and CPU-friendly:

    python benchmarks/benchmark_archive_quality.py --steps 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcc_transformer import QCCArchive


def teacher_outputs(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    window_size: int,
    decay: float,
) -> torch.Tensor:
    """Direct exponentially-decayed softmax attention over evicted history."""

    batch, length, heads, dim = keys.shape
    outputs = torch.zeros(batch, length, heads, dim, device=keys.device)
    for t in range(window_size, length):
        history = keys[:, : t - window_size + 1]
        history_values = values[:, : t - window_size + 1]
        ages = decay ** (t - torch.arange(history.shape[1], device=keys.device))
        logits = torch.einsum("bhd,bihd->bhi", queries[:, t], history) / (dim**0.5)
        weights = torch.softmax(logits + ages.log()[None, None, :], dim=-1)
        outputs[:, t] = torch.einsum("bhi,bihd->bhd", weights, history_values)
    return outputs


def archive_outputs(
    archive: QCCArchive,
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    batch, length, heads, dim = keys.shape
    archive.reset_state(batch, device=keys.device)
    outputs = torch.zeros(batch, length, heads, dim, device=keys.device)
    for t in range(window_size, length):
        archive.update(keys[:, t - window_size], values[:, t - window_size])
        outputs[:, t] = archive.read(queries[:, t])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--codes", type=int, default=8)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    decay = 0.94
    keys = torch.randn(args.batch, args.length, 1, args.dim)
    values = torch.randn_like(keys)
    queries = torch.randn_like(keys)
    target = teacher_outputs(keys, values, queries, args.window, decay)
    archive = QCCArchive(
        num_heads=1,
        head_dim=args.dim,
        num_codes=args.codes,
        decay_rates=(decay,),
        window_size=args.window,
    )
    with torch.no_grad():
        initial = archive_outputs(archive, keys, values, queries, args.window)
        initial_mse = torch.mean((initial[:, args.window :] - target[:, args.window :]) ** 2).item()
    optimizer = torch.optim.Adam(archive.parameters(), lr=3e-2)
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        predicted = archive_outputs(archive, keys, values, queries, args.window)
        loss = torch.mean((predicted[:, args.window :] - target[:, args.window :]) ** 2)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = archive_outputs(archive, keys, values, queries, args.window)
        final_mse = torch.mean((final[:, args.window :] - target[:, args.window :]) ** 2).item()
    state_elements = args.codes * (args.dim + 1)
    print(f"steps={args.steps} batch={args.batch} length={args.length} dim={args.dim} codes={args.codes}")
    print(f"initial_mse={initial_mse:.6f} final_mse={final_mse:.6f} improvement={initial_mse / max(final_mse, 1e-12):.2f}x")
    print(f"archive_state_elements_per_head={state_elements} direct_history_elements_per_token={args.length - args.window}")


if __name__ == "__main__":
    main()
