"""Join real-HF memory and fixed-SLA concurrency evidence.

The streaming report supplies attention-state bytes for one matched workload;
the concurrency sweep supplies the largest batch that keeps both variants
inside the same elapsed-time SLA.  This command joins only completed,
provenanced measurements and leaves missing or failed sides unusable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return value


def _same(left: dict[str, Any], right: dict[str, Any], field: str) -> Any:
    left_value = left.get(field)
    right_value = right.get(field)
    if left_value != right_value:
        raise ValueError(f"memory and concurrency reports disagree on {field}")
    if left_value is None:
        raise ValueError(f"memory evidence requires {field}")
    return left_value


def _status_ok(report: dict[str, Any], field: str) -> dict[str, Any]:
    value = report.get(field)
    if not isinstance(value, dict) or value.get("status") != "ok":
        raise ValueError(f"memory report {field} did not complete")
    return value


def compare_reports(memory: dict[str, Any], concurrency: dict[str, Any]) -> dict[str, Any]:
    """Return a flat production-memory section from two matched HF reports."""

    if memory.get("real_model") is not True or concurrency.get("real_model") is not True:
        raise ValueError("memory and concurrency evidence must use a real model")
    if memory.get("synthetic") is not False or concurrency.get("synthetic") is not False:
        raise ValueError("memory and concurrency evidence must not be synthetic")
    if memory.get("qcc_only") is True or concurrency.get("qcc_only") is True:
        raise ValueError("memory and concurrency evidence must include Full-KV")
    model_id = _same(memory, concurrency, "model_id")
    run_id = _same(memory, concurrency, "run_id")
    total_tokens = memory.get("total_tokens")
    if total_tokens != concurrency.get("total_tokens_per_request"):
        raise ValueError("memory and concurrency reports disagree on context length")
    if total_tokens is None:
        raise ValueError("memory evidence requires total_tokens")
    if not isinstance(total_tokens, int) or total_tokens <= 0:
        raise ValueError("total_tokens must be a positive integer")
    if memory.get("matched") is not True:
        raise ValueError("memory report is not matched")
    if concurrency.get("fixed_sla") is not True:
        raise ValueError("concurrency report is not measured under a fixed SLA")
    if memory.get("protocol_locked") is not True or concurrency.get("protocol_locked") is not True:
        raise ValueError("memory and concurrency protocols must be locked before execution")
    native_context = _same(memory, concurrency, "native_context_tokens")
    if (
        isinstance(native_context, bool)
        or not isinstance(native_context, int)
        or native_context < total_tokens
    ):
        raise ValueError("memory evidence requires native context >= requested context")

    baseline = _status_ok(memory, "baseline_full_kv")
    qcc = _status_ok(memory, "qcc_retrofit")
    full_state = _positive(
        baseline.get("attention_state_bytes"),
        "baseline_full_kv.attention_state_bytes",
    )
    qcc_state = _nonnegative(
        qcc.get("attention_state_bytes"),
        "qcc_retrofit.attention_state_bytes",
    )
    full_concurrency = _positive(
        concurrency.get("max_full_kv_batch"),
        "max_full_kv_batch",
    )
    qcc_concurrency = _positive(
        concurrency.get("max_qcc_batch"),
        "max_qcc_batch",
    )
    sla_seconds = _positive(concurrency.get("sla_seconds"), "sla_seconds")
    provenance = {
        "model_id": model_id,
        "run_id": run_id,
        "real_model": True,
        "synthetic": False,
        "official": False,
        "protocol_locked": True,
        "qcc_only": False,
        "matched_full_kv": True,
    }
    return {
        "schema": "qcc-memory-comparison-v1",
        **provenance,
        "context_tokens": total_tokens,
        "native_context_tokens": native_context,
        "full_kv_attention_state_bytes": full_state,
        "qcc_attention_state_bytes": qcc_state,
        "full_kv_concurrency": full_concurrency,
        "qcc_concurrency": qcc_concurrency,
        "fixed_sla": True,
        "sla_seconds": sla_seconds,
        "attention_state_reduction": 1.0 - qcc_state / full_state,
        "concurrency_speedup": qcc_concurrency / full_concurrency,
        "sources": {
            "streaming_memory": memory,
            "concurrency_sweep": concurrency,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-report", type=Path, required=True)
    parser.add_argument("--concurrency-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare_reports(_load(args.memory_report), _load(args.concurrency_report))
    except ValueError as exc:
        print(json.dumps({"matched": False, "error": str(exc)}, indent=2))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
