"""Evaluate long-range token retrieval from a checkpoint.

The evaluator uses a deliberately small JSONL interchange format so it can be
adapted to RULER exports without coupling this repository to a particular
dataset package.  Each line must contain::

    {"input_ids": [..], "target_position": 12345, "answers": [42]}

The model output at ``target_position`` predicts the next token.  Inputs are
streamed in chunks; the complete prompt is never copied into a KV cache by the
QCC path.  A checkpoint is mandatory because an untrained model cannot provide
meaningful retrieval evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcc_transformer import QCCForCausalLM


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a state-dict mapping")
    for key in ("state_dict", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            payload = candidate
            break
    if not payload or not all(isinstance(key, str) for key in payload):
        raise ValueError("checkpoint does not look like a PyTorch state dict")
    if all(key.startswith("module.") for key in payload):
        payload = {key.removeprefix("module."): value for key, value in payload.items()}
    return payload


def _record_fields(record: dict[str, Any]) -> tuple[torch.Tensor, int, set[int]]:
    raw_tokens = record.get("input_ids")
    if not isinstance(raw_tokens, list) or not raw_tokens:
        raise ValueError("each record needs a non-empty input_ids list")
    if not all(isinstance(token, int) and token >= 0 for token in raw_tokens):
        raise ValueError("input_ids must contain non-negative integers")
    position = record.get("target_position")
    answers = record.get("answers")
    if not isinstance(position, int) or not 0 <= position < len(raw_tokens):
        raise ValueError("target_position must index input_ids")
    if isinstance(answers, int):
        answers = [answers]
    if not isinstance(answers, list) or not answers or not all(isinstance(x, int) for x in answers):
        raise ValueError("answers must be a non-empty integer list")
    return torch.tensor(raw_tokens, dtype=torch.long), position, set(answers)


def _iter_records(dataset: Path, max_examples: int | None):
    total = 0
    with dataset.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if max_examples is not None and total >= max_examples:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be a JSON object")
                fields = _record_fields(record)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid record on line {line_number}: {exc}") from exc
            total += 1
            yield line_number, fields


@torch.no_grad()
def _predict_logits(
    model: QCCForCausalLM,
    input_ids: torch.Tensor,
    target_position: int,
    *,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    if input_ids.numel() > model.max_position_embeddings:
        raise ValueError(
            f"record has {input_ids.numel()} tokens, exceeds "
            f"max_position_embeddings={model.max_position_embeddings}"
        )
    model.reset_cache(batch_size=1)
    for start in range(0, input_ids.numel(), chunk_size):
        end = min(input_ids.numel(), start + chunk_size)
        logits = model.decode_chunk(
            input_ids[start:end].unsqueeze(0).to(device), reset_cache=start == 0
        )
        if start <= target_position < end:
            return logits[0, target_position - start].detach()
    raise RuntimeError("target_position was not reached")


@torch.no_grad()
def evaluate(
    model: QCCForCausalLM,
    dataset: Path,
    *,
    chunk_size: int,
    device: torch.device,
    max_examples: int | None,
) -> tuple[int, int]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    total = 0
    correct = 0
    for _, (input_ids, target_position, answers) in _iter_records(dataset, max_examples):
        logits = _predict_logits(
            model,
            input_ids,
            target_position,
            chunk_size=chunk_size,
            device=device,
        )
        correct += int(int(logits.argmax().item()) in answers)
        total += 1
    return correct, total


@torch.no_grad()
def evaluate_pair(
    qcc: QCCForCausalLM,
    full: QCCForCausalLM,
    dataset: Path,
    *,
    chunk_size: int,
    device: torch.device,
    max_examples: int | None,
) -> tuple[int, int, int, float]:
    """Evaluate matched models and return accuracies plus mean logit cosine."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    qcc_correct = full_correct = total = 0
    cosine_sum = 0.0
    for _, (input_ids, target_position, answers) in _iter_records(dataset, max_examples):
        qcc_logits = _predict_logits(
            qcc, input_ids, target_position, chunk_size=chunk_size, device=device
        )
        full_logits = _predict_logits(
            full, input_ids, target_position, chunk_size=chunk_size, device=device
        )
        qcc_correct += int(int(qcc_logits.argmax().item()) in answers)
        full_correct += int(int(full_logits.argmax().item()) in answers)
        cosine_sum += float(
            torch.nn.functional.cosine_similarity(
                qcc_logits.unsqueeze(0), full_logits.unsqueeze(0), dim=-1
            ).item()
        )
        total += 1
    return qcc_correct, full_correct, total, cosine_sum / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--num-codes", type=int, default=16)
    parser.add_argument("--archive-content-threshold", type=float, default=None)
    parser.add_argument("--archive-persistent-landmark", action="store_true")
    parser.add_argument("--archive-prefix-landmark", action="store_true")
    parser.add_argument("--archive-prefix-pair-landmark", action="store_true")
    parser.add_argument(
        "--position-encoding",
        choices=("sinusoidal", "learned", "rope", "none"),
        default="sinusoidal",
    )
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    parser.add_argument("--max-position-embeddings", type=int, default=1_000_001)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--target-accuracy", type=float, default=0.98)
    parser.add_argument(
        "--compare-full-kv",
        action="store_true",
        help="also evaluate a matched full-KV model (feasible only at modest context lengths)",
    )
    args = parser.parse_args()
    if not 0.0 <= args.target_accuracy <= 1.0:
        raise ValueError("target-accuracy must be in [0, 1]")
    device = torch.device(args.device)
    model = QCCForCausalLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        max_position_embeddings=args.max_position_embeddings,
        window_size=args.window_size,
        num_codes=args.num_codes,
        position_encoding=args.position_encoding,
        rope_theta=args.rope_theta,
        archive_content_threshold=args.archive_content_threshold,
        archive_persistent_landmark=args.archive_persistent_landmark,
        archive_prefix_landmark=args.archive_prefix_landmark,
        archive_prefix_pair_landmark=args.archive_prefix_pair_landmark,
    ).to(device)
    state_dict = _load_checkpoint(args.checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    full: QCCForCausalLM | None = None
    if args.compare_full_kv:
        full = QCCForCausalLM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.layers,
            num_heads=args.heads,
            max_position_embeddings=args.max_position_embeddings,
            # The baseline retains all positions up to the configured limit.
            window_size=args.max_position_embeddings,
            num_codes=args.num_codes,
            position_encoding=args.position_encoding,
            rope_theta=args.rope_theta,
            archive_content_threshold=args.archive_content_threshold,
            use_archive=False,
        ).to(device)
        full.load_state_dict(state_dict, strict=True)
        full.eval()
        correct, full_correct, total, mean_cosine = evaluate_pair(
            model,
            full,
            args.dataset,
            chunk_size=args.chunk_size,
            device=device,
            max_examples=args.max_examples,
        )
    else:
        correct, total = evaluate(
            model,
            args.dataset,
            chunk_size=args.chunk_size,
            device=device,
            max_examples=args.max_examples,
        )
    accuracy = correct / total if total else 0.0
    print(
        f"device={device} examples={total} correct={correct} "
        f"retrieval_accuracy={accuracy:.6f} target={args.target_accuracy:.6f} "
        f"passed={accuracy >= args.target_accuracy}"
    )
    if full is not None:
        full_accuracy = full_correct / total if total else 0.0
        ratio = accuracy / full_accuracy if full_accuracy else 0.0
        print(
            f"full_kv_accuracy={full_accuracy:.6f} full_kv_correct={full_correct} "
            f"quality_ratio={ratio:.6f} mean_logit_cosine={mean_cosine:.6f}"
            if full_accuracy
            else f"full_kv_accuracy={full_accuracy:.6f} full_kv_correct={full_correct} "
            f"quality_ratio=nan mean_logit_cosine={mean_cosine:.6f}"
        )


if __name__ == "__main__":
    main()
