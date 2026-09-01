"""Run the QCC target matrix with one configuration and machine-readable output.

This is an audit harness, not a synthetic claim generator.  State accounting
is always available; latency requires ``--run-latency``; quality requires a
matched Full-KV run at a feasible length; and retrieval requires both a trained
checkpoint and a JSONL dataset.  Missing prerequisites are reported as
``missing`` rather than being treated as passes.

Examples::

    python benchmarks/audit_targets.py --state-only --json
    python benchmarks/audit_targets.py --run-latency --device cuda \
        --lengths 128000,1000000 --json
    python benchmarks/audit_targets.py --checkpoint qcc.pt \
        --dataset ruler_export.jsonl --device cuda --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

# Permit running this file directly from a fresh checkout.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_fullkv_quality import compare as compare_quality
from benchmark_long_context import run_stream, state_report
from evaluate_retrieval import _load_checkpoint, evaluate, evaluate_pair
from qcc_transformer import QCCForCausalLM


def _ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("length lists must contain positive integers")
    return values


def _model_kwargs(args: argparse.Namespace, length: int, *, full: bool) -> dict[str, Any]:
    return {
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "num_layers": args.layers,
        "num_heads": args.heads,
        "max_position_embeddings": length + 1,
        "window_size": length + 1 if full else args.window_size,
        "num_codes": args.num_codes,
        "archive_content_threshold": args.archive_content_threshold,
        "position_encoding": args.position_encoding,
        "rope_theta": args.rope_theta,
        "use_archive": not full,
        "archive_scan_block_size": args.archive_scan_block_size,
    }


def _gate(value: float | None, threshold: float, *, higher: bool = True) -> dict[str, Any]:
    if value is None:
        return {"value": None, "threshold": threshold, "status": "missing"}
    passed = value >= threshold if higher else value <= threshold
    return {
        "value": value,
        "threshold": threshold,
        "status": "pass" if passed else "fail",
    }


def _state_gates(args: argparse.Namespace, lengths: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for length in lengths:
        model = QCCForCausalLM(**_model_kwargs(args, length, full=False))
        report = state_report(model, length)
        result[str(length)] = {
            "report": report,
            "under_0.5pct": report["state_fraction_percent"] <= 0.5,
            "under_0.1pct": report["state_fraction_percent"] <= 0.1,
        }
    return result


@torch.no_grad()
def _latency_gates(
    args: argparse.Namespace,
    lengths: list[int],
    device: torch.device,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not args.run_latency:
        return {str(length): {"status": "not_run"} for length in lengths}
    for length in lengths:
        qcc = QCCForCausalLM(**_model_kwargs(args, length, full=False)).to(device)
        qcc_prefill, qcc_tpot, _ = run_stream(
            qcc, length, args.chunk_size, args.vocab_size, device
        )
        entry: dict[str, Any] = {
            "qcc_prefill_seconds": qcc_prefill,
            "qcc_tpot_ms": qcc_tpot,
            "full": {"status": "missing", "reason": "full-KV limit or --compare-full not set"},
        }
        if args.compare_full and length <= args.full_max_length:
            full = QCCForCausalLM(**_model_kwargs(args, length, full=True)).to(device)
            full_prefill, full_tpot, _ = run_stream(
                full, length, args.chunk_size, args.vocab_size, device
            )
            entry["full"] = {
                "prefill_seconds": full_prefill,
                "tpot_ms": full_tpot,
                "ttft_speedup": full_prefill / max(qcc_prefill, 1e-12),
                "tpot_speedup": full_tpot / max(qcc_tpot, 1e-12),
            }
        result[str(length)] = entry
    return result


@torch.no_grad()
def _quality_gate(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not args.quality_lengths:
        return result
    for length in args.quality_lengths:
        if length > args.full_max_length:
            result[str(length)] = {
                "status": "missing",
                "reason": "quality length exceeds --full-max-length",
            }
            continue
        qcc = QCCForCausalLM(**_model_kwargs(args, length, full=False)).to(device)
        full = QCCForCausalLM(**_model_kwargs(args, length, full=True)).to(device)
        full.load_state_dict(qcc.state_dict(), strict=True)
        tokens = torch.randint(0, args.vocab_size, (args.batch, length), device=device)
        cosine, mse = compare_quality(qcc, full, tokens, chunk_size=args.chunk_size)
        result[str(length)] = {
            "mean_logit_cosine": cosine,
            "per_logit_mse": mse,
            "gate": _gate(cosine, args.quality_target),
        }
    return result


@torch.no_grad()
def _retrieval_gate(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if args.checkpoint is None or args.dataset is None:
        return {
            "status": "missing",
            "reason": "--checkpoint and --dataset are both required",
        }
    model = QCCForCausalLM(**_model_kwargs(args, args.retrieval_context, full=False)).to(device)
    state_dict = _load_checkpoint(args.checkpoint)
    model.load_state_dict(state_dict, strict=True)
    if args.compare_full_retrieval and args.retrieval_context > args.full_max_length:
        return {
            "status": "missing",
            "reason": "retrieval context exceeds --full-max-length for Full-KV comparison",
        }
    if args.compare_full_retrieval:
        full = QCCForCausalLM(
            **_model_kwargs(args, args.retrieval_context, full=True)
        ).to(device)
        full.load_state_dict(state_dict, strict=True)
        qcc_correct, full_correct, total, mean_cosine = evaluate_pair(
            model,
            full,
            args.dataset,
            chunk_size=args.chunk_size,
            device=device,
            max_examples=args.max_examples,
        )
        accuracy = qcc_correct / total if total else 0.0
        full_accuracy = full_correct / total if total else 0.0
        ratio = accuracy / full_accuracy if full_accuracy else 0.0
        return {
            "correct": qcc_correct,
            "full_correct": full_correct,
            "total": total,
            "accuracy": accuracy,
            "full_accuracy": full_accuracy,
            "quality_ratio": ratio,
            "mean_logit_cosine": mean_cosine,
            "gate": _gate(ratio, args.quality_target),
        }
    correct, total = evaluate(
        model,
        args.dataset,
        chunk_size=args.chunk_size,
        device=device,
        max_examples=args.max_examples,
    )
    accuracy = correct / total if total else 0.0
    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "gate": _gate(accuracy, args.retrieval_target),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="128000,1000000,4000000")
    parser.add_argument("--quality-lengths", default="1024,2048")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--num-codes", type=int, default=16)
    parser.add_argument("--archive-content-threshold", type=float, default=None)
    parser.add_argument("--archive-scan-block-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--full-max-length", type=int, default=4096)
    parser.add_argument("--retrieval-context", type=int, default=1_000_000)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--retrieval-target", type=float, default=0.98)
    parser.add_argument("--quality-target", type=float, default=0.99)
    parser.add_argument(
        "--position-encoding",
        choices=("sinusoidal", "learned", "rope", "none"),
        default="sinusoidal",
    )
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--run-latency", action="store_true")
    parser.add_argument("--compare-full", action="store_true")
    parser.add_argument("--compare-full-retrieval", action="store_true")
    parser.add_argument("--state-only", action="store_true", help="skip latency, quality, and retrieval")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.full_max_length <= 0 or args.chunk_size <= 0:
        raise ValueError("full-max-length and chunk-size must be positive")
    if args.threads is not None:
        if args.threads <= 0:
            raise ValueError("threads must be positive")
        torch.set_num_threads(args.threads)
    lengths = _ints(args.lengths)
    args.quality_lengths = _ints(args.quality_lengths) if args.quality_lengths else []
    device = torch.device(args.device)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "config": {
            "device": str(device),
            "lengths": lengths,
            "quality_lengths": args.quality_lengths,
            "run_latency": bool(args.run_latency and not args.state_only),
            "compare_full": bool(args.compare_full and not args.state_only),
        },
        "state": _state_gates(args, lengths),
    }
    if args.state_only:
        result["latency"] = {str(length): {"status": "not_run"} for length in lengths}
        result["quality"] = {}
        result["retrieval"] = {"status": "not_run"}
    else:
        result["latency"] = _latency_gates(args, lengths, device)
        result["quality"] = _quality_gate(args, device)
        result["retrieval"] = _retrieval_gate(args, device)
    result["elapsed_seconds"] = time.perf_counter() - started
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"device={device} lengths={','.join(map(str, lengths))}")
    for length, entry in result["state"].items():
        report = entry["report"]
        print(
            f"state length={length} fraction={report['state_fraction_percent']:.6f}% "
            f"reduction={report['reduction']:.2f}x under_0.5pct={entry['under_0.5pct']}"
        )
    for length, entry in result["latency"].items():
        if entry.get("status") == "not_run":
            continue
        full = entry.get("full", {})
        print(
            f"latency length={length} qcc_prefill={entry['qcc_prefill_seconds']:.4f}s "
            f"qcc_tpot={entry['qcc_tpot_ms']:.3f}ms "
            f"ttft_speedup={full.get('ttft_speedup', 'missing')} "
            f"tpot_speedup={full.get('tpot_speedup', 'missing')}"
        )
    for length, entry in result["quality"].items():
        print(
            f"quality length={length} cosine={entry['mean_logit_cosine']:.6f} "
            f"status={entry['gate']['status']}"
        )
    retrieval = result["retrieval"]
    if retrieval.get("accuracy") is not None:
        quality_suffix = (
            f" quality_ratio={retrieval['quality_ratio']:.6f}"
            if retrieval.get("quality_ratio") is not None
            else ""
        )
        print(
            f"retrieval accuracy={retrieval['accuracy']:.6f} "
            f"status={retrieval['gate']['status']}{quality_suffix}"
        )
    else:
        print(f"retrieval status={retrieval.get('status', 'missing')}")


if __name__ == "__main__":
    main()
