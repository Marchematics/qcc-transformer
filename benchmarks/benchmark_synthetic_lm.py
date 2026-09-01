"""Matched QCC/full-KV experiment on a random long-range recall task.

The value paired with the first marker is sampled independently per example,
so a model must carry it across the filler span. This is intentionally a
diagnostic task, not a replacement for language-model evaluation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcc_transformer import QCCForCausalLM


def make_batch(
    batch_size: int,
    length: int,
    query_position: int,
    vocab_size: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query_position < 4 or query_position + 1 >= length:
        raise ValueError("query_position must leave room for a long prefix and target")
    marker = 1
    values = torch.randint(2, vocab_size, (batch_size,), device=device)
    tokens = torch.randint(2, vocab_size, (batch_size, length), device=device)
    tokens[:, 0] = marker
    tokens[:, 1] = values
    tokens[:, query_position] = marker
    tokens[:, query_position + 1] = values
    return tokens, values


def train_one(
    model: QCCForCausalLM,
    *,
    steps: int,
    batch_size: int,
    length: int,
    query_position: int,
    vocab_size: int,
    device: torch.device,
    lr: float,
) -> tuple[float, float, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    start = time.perf_counter()
    final_loss = float("nan")
    final_accuracy = float("nan")
    for _ in range(steps):
        tokens, values = make_batch(
            batch_size, length, query_position, vocab_size, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        query_logits = logits[:, query_position]
        loss = F.cross_entropy(query_logits, values)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = loss.item()
        final_accuracy = (query_logits.argmax(-1) == values).float().mean().item()
    return final_loss, final_accuracy, time.perf_counter() - start


@torch.no_grad()
def evaluate_one(
    model: QCCForCausalLM,
    *,
    batch_size: int,
    length: int,
    query_position: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    tokens, values = make_batch(
        batch_size, length, query_position, vocab_size, device=device
    )
    query_logits = model(tokens)[:, query_position]
    return (
        F.cross_entropy(query_logits, values).item(),
        (query_logits.argmax(-1) == values).float().mean().item(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--query-position", type=int, default=48)
    parser.add_argument("--vocab", type=int, default=32)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--codes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    common = dict(
        vocab_size=args.vocab,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        max_position_embeddings=args.length,
        window_size=args.window,
        num_codes=args.codes,
        dropout=0.0,
    )
    qcc = QCCForCausalLM(**common, use_archive=True).to(device)
    full_common = {**common, "window_size": args.length}
    full = QCCForCausalLM(**full_common, use_archive=False).to(device)
    qcc_train_loss, qcc_train_acc, qcc_seconds = train_one(
        qcc,
        steps=args.steps,
        batch_size=args.batch,
        length=args.length,
        query_position=args.query_position,
        vocab_size=args.vocab,
        device=device,
        lr=3e-3,
    )
    full_train_loss, full_train_acc, full_seconds = train_one(
        full,
        steps=args.steps,
        batch_size=args.batch,
        length=args.length,
        query_position=args.query_position,
        vocab_size=args.vocab,
        device=device,
        lr=3e-3,
    )
    print(f"device={device} steps={args.steps} length={args.length} query_position={args.query_position}")
    qcc_eval_loss, qcc_eval_acc = evaluate_one(
        qcc,
        batch_size=args.batch,
        length=args.length,
        query_position=args.query_position,
        vocab_size=args.vocab,
        device=device,
    )
    full_eval_loss, full_eval_acc = evaluate_one(
        full,
        batch_size=args.batch,
        length=args.length,
        query_position=args.query_position,
        vocab_size=args.vocab,
        device=device,
    )
    print(f"qcc_train_loss={qcc_train_loss:.6f} qcc_train_accuracy={qcc_train_acc:.3f} qcc_train_seconds={qcc_seconds:.2f}")
    print(f"qcc_eval_loss={qcc_eval_loss:.6f} qcc_eval_accuracy={qcc_eval_acc:.3f}")
    print(f"full_train_loss={full_train_loss:.6f} full_train_accuracy={full_train_acc:.3f} full_train_seconds={full_seconds:.2f}")
    print(f"full_eval_loss={full_eval_loss:.6f} full_eval_accuracy={full_eval_acc:.3f}")


if __name__ == "__main__":
    main()
