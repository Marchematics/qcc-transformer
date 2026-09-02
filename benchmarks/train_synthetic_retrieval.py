"""Train a small QCC checkpoint on a long-range marker/value task.

This is a reproducible diagnostic for the archive mechanism, not a substitute
for RULER.  Each example contains a random value immediately after a marker at
the beginning of the stream and asks the model to predict that value after a
second marker separated by filler.  The generated JSONL follows the format
consumed by :mod:`evaluate_retrieval`.

The model is trained at a short curriculum length because the differentiable
reference path intentionally keeps the recurrence explicit.  Evaluation can
then extrapolate the trained checkpoint to 128K or 1M using the no-grad
streaming/Triton path.  A checkpoint records its architecture metadata so a
run cannot silently be evaluated with incompatible dimensions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

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
    """Create marker/value streams and return tokens plus the value target."""

    if length <= query_position + 1 or query_position < 2:
        raise ValueError("length must leave a target token after query_position")
    if vocab_size < 5:
        raise ValueError("vocab_size must leave marker, filler, and value tokens")
    marker = 1
    filler = 2
    values = torch.randint(3, vocab_size, (batch_size,), device=device)
    tokens = torch.full(
        (batch_size, length), filler, dtype=torch.long, device=device
    )
    tokens[:, 0] = marker
    tokens[:, 1] = values
    tokens[:, query_position] = marker
    tokens[:, query_position + 1] = values
    return tokens, values


def _model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "num_layers": args.layers,
        "num_heads": args.heads,
        "max_position_embeddings": args.max_position_embeddings,
        "window_size": args.window_size,
        "num_codes": args.num_codes,
        "dropout": 0.0,
        "position_encoding": args.position_encoding,
        "rope_theta": args.rope_theta,
        "use_archive": True,
        "use_triton": False,
        "archive_scan_block_size": args.archive_scan_block_size,
        "archive_content_threshold": args.archive_content_threshold,
        "archive_persistent_landmark": args.archive_persistent_landmark,
        "archive_prefix_landmark": args.archive_prefix_landmark,
        "archive_prefix_pair_landmark": args.archive_prefix_pair_landmark,
    }


def train(
    model: QCCForCausalLM,
    *,
    steps: int,
    batch_size: int,
    length: int,
    query_position: int,
    query_min: int | None,
    query_max: int | None,
    vocab_size: int,
    device: torch.device,
    lr: float,
    log_every: int,
) -> dict[str, float]:
    """Train only the retrieval position and return final metrics."""

    if steps <= 0 or batch_size <= 0 or log_every <= 0:
        raise ValueError("steps, batch_size, and log_every must be positive")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    started = time.perf_counter()
    final_loss = float("nan")
    final_accuracy = float("nan")
    best_loss = float("inf")
    best_accuracy = float("nan")
    best_state: dict[str, torch.Tensor] | None = None
    for step in range(1, steps + 1):
        step_query_position = query_position
        if query_min is not None or query_max is not None:
            lower = query_min if query_min is not None else 2
            upper = query_max if query_max is not None else length - 2
            if not 2 <= lower <= upper <= length - 2:
                raise ValueError("query range must lie in [2, train_length - 2]")
            step_query_position = int(torch.randint(lower, upper + 1, ()).item())
        tokens, values = make_batch(
            batch_size,
            length,
            step_query_position,
            vocab_size,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        # When a query range is enabled, supervise the position sampled for
        # this step rather than the nominal default.  Using ``query_position``
        # here silently trained only one location and made curriculum runs
        # appear to converge while ignoring their sampled targets.
        query_logits = logits[:, step_query_position]
        loss = F.cross_entropy(query_logits, values)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.item())
        final_accuracy = float((query_logits.argmax(dim=-1) == values).float().mean().item())
        if final_loss < best_loss:
            best_loss = final_loss
            best_accuracy = final_accuracy
            # Keep the best checkpoint rather than the potentially unstable
            # final optimizer iterate; this is especially important for the
            # tiny random-value task where a late AdamW step can collapse the
            # archive gate back to the marker token.
            best_state = {
                name: parameter.detach().clone()
                for name, parameter in model.state_dict().items()
            }
        if step == 1 or step == steps or step % log_every == 0:
            print(
                f"step={step} loss={final_loss:.6f} accuracy={final_accuracy:.4f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return {
        "steps": float(steps),
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "best_loss": best_loss,
        "best_accuracy": best_accuracy,
        "train_seconds": time.perf_counter() - started,
    }


def write_dataset(
    path: Path,
    *,
    lengths: list[int],
    examples_per_length: int,
    query_fraction: float,
    vocab_size: int,
    seed: int,
) -> int:
    """Write evaluator JSONL without retaining all long records in memory."""

    if examples_per_length <= 0 or not 0.0 < query_fraction < 1.0:
        raise ValueError("examples_per_length must be positive and query_fraction in (0, 1)")
    if vocab_size < 5:
        raise ValueError("vocab_size must be at least 5")
    generator = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with path.open("w", encoding="utf-8") as stream:
        for length in lengths:
            if length < 4:
                raise ValueError("dataset lengths must be at least 4")
            query_position = max(2, min(length - 2, int(length * query_fraction)))
            for _ in range(examples_per_length):
                value = generator.randrange(3, vocab_size)
                tokens = [2] * length
                tokens[0] = 1
                tokens[1] = value
                tokens[query_position] = 1
                tokens[query_position + 1] = value
                json.dump(
                    {
                        "input_ids": tokens,
                        "target_position": query_position,
                        "answers": [value],
                    },
                    stream,
                    separators=(",", ":"),
                )
                stream.write("\n")
                total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--train-length", type=int, default=128)
    parser.add_argument("--query-position", type=int, default=96)
    parser.add_argument("--query-min", type=int, default=None)
    parser.add_argument("--query-max", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--num-codes", type=int, default=16)
    parser.add_argument("--archive-scan-block-size", type=int, default=64)
    parser.add_argument("--archive-content-threshold", type=float, default=None)
    parser.add_argument("--archive-persistent-landmark", action="store_true")
    parser.add_argument("--archive-prefix-landmark", action="store_true")
    parser.add_argument("--archive-prefix-pair-landmark", action="store_true")
    parser.add_argument("--max-position-embeddings", type=int, default=128_001)
    parser.add_argument(
        "--position-encoding",
        choices=("sinusoidal", "rope", "none"),
        default="sinusoidal",
    )
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--eval-lengths", default="128,1024,128000")
    parser.add_argument("--eval-examples", type=int, default=10)
    parser.add_argument("--query-fraction", type=float, default=0.75)
    args = parser.parse_args()
    if args.train_length <= args.query_position + 1:
        raise ValueError("train-length must exceed query-position + 1")
    eval_lengths = [int(raw.strip()) for raw in args.eval_lengths.split(",") if raw.strip()]
    if not eval_lengths or any(length <= 0 for length in eval_lengths):
        raise ValueError("eval-lengths must contain positive integers")
    if args.max_position_embeddings <= max(eval_lengths):
        raise ValueError("max-position-embeddings must exceed the largest eval length")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    model = QCCForCausalLM(**_model_kwargs(args)).to(device)
    metrics = train(
        model,
        steps=args.steps,
        batch_size=args.batch,
        length=args.train_length,
        query_position=args.query_position,
        query_min=args.query_min,
        query_max=args.query_max,
        vocab_size=args.vocab_size,
        device=device,
        lr=args.lr,
        log_every=args.log_every,
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            key: value
            for key, value in _model_kwargs(args).items()
            if key not in {"use_archive", "use_triton", "dropout"}
        },
        "training": metrics,
        "seed": args.seed,
    }
    torch.save(payload, args.checkpoint)
    examples = write_dataset(
        args.dataset,
        lengths=eval_lengths,
        examples_per_length=args.eval_examples,
        query_fraction=args.query_fraction,
        vocab_size=args.vocab_size,
        seed=args.seed + 1,
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "checkpoint": str(args.checkpoint),
                "dataset": str(args.dataset),
                "examples": examples,
                "eval_lengths": eval_lengths,
                "train": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
