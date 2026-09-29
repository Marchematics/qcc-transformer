"""Tiny Shakespeare character-level QCC vs full-KV experiment.

The dataset is fetched on demand into ``work/`` and never committed. The two
models share architecture, optimizer, batches, and seed; only the attention
memory policy differs.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcc_transformer import QCCForCausalLM

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_data(path: Path, split: float = 0.9) -> tuple[torch.Tensor, torch.Tensor, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(URL, path)
    text = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    mapping = {char: index for index, char in enumerate(chars)}
    encoded = torch.tensor([mapping[char] for char in text], dtype=torch.long)
    pivot = int(len(encoded) * split)
    return encoded[:pivot], encoded[pivot:], len(chars)


def batchify(data: torch.Tensor, batch_size: int, length: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, data.numel() - length - 1, (batch_size,))
    x = torch.stack([data[start : start + length] for start in starts]).to(device)
    return x, x[:, 1:].contiguous()


def causal_loss(logits: torch.Tensor, targets: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Score next-token prediction and ignore the final context position."""

    return F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), targets.reshape(-1))


@torch.no_grad()
def evaluate(model: QCCForCausalLM, data: torch.Tensor, *, batch_size: int, length: int, batches: int, device: torch.device) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = batchify(data, batch_size, length, device=device)
        losses.append(causal_loss(model(x), y, model.lm_head.out_features).item())
    return sum(losses) / len(losses)


def train(model: QCCForCausalLM, data: torch.Tensor, *, steps: int, batch_size: int, length: int, device: torch.device, lr: float) -> float:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    start = time.perf_counter()
    for _ in range(steps):
        x, y = batchify(data, batch_size, length, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = causal_loss(logits, y, model.lm_head.out_features)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--codes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data", default="work/tinyshakespeare.txt")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    train_data, valid_data, vocab = load_data(Path(args.data))
    common = dict(
        vocab_size=vocab,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        max_position_embeddings=args.length,
        window_size=args.window,
        num_codes=args.codes,
        dropout=0.0,
    )
    qcc = QCCForCausalLM(**common, use_archive=True).to(device)
    full = QCCForCausalLM(**{**common, "window_size": args.length}, use_archive=False).to(device)
    qcc_seconds = train(qcc, train_data, steps=args.steps, batch_size=args.batch, length=args.length, device=device, lr=3e-3)
    full_seconds = train(full, train_data, steps=args.steps, batch_size=args.batch, length=args.length, device=device, lr=3e-3)
    qcc_loss = evaluate(qcc, valid_data, batch_size=args.batch, length=args.length, batches=args.eval_batches, device=device)
    full_loss = evaluate(full, valid_data, batch_size=args.batch, length=args.length, batches=args.eval_batches, device=device)
    print(f"device={device} vocab={vocab} steps={args.steps} length={args.length} window={args.window}")
    print(f"qcc_valid_loss={qcc_loss:.6f} qcc_ppl={math.exp(qcc_loss):.3f} qcc_train_seconds={qcc_seconds:.2f}")
    print(f"full_valid_loss={full_loss:.6f} full_ppl={math.exp(full_loss):.3f} full_train_seconds={full_seconds:.2f}")


if __name__ == "__main__":
    main()
