"""Compare two matched stock-vLLM streaming reports.

The comparison keeps mean throughput separate from tail latency.  A QCC result is
only marked dominant when the two reports use the same model/workload/context and
QCC improves throughput, p95 and p99 TPOT without increasing p50 TTFT.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read serving report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"serving report {path} must contain a JSON object")
    if payload.get("schema") != "qcc-vllm-serving-v1":
        raise ValueError(f"unsupported serving report schema in {path}")
    if payload.get("stock_vllm") is not True or payload.get("streaming") is not True:
        raise ValueError(f"{path} is not a streaming stock-vLLM report")
    return payload


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _summary_value(report: dict[str, Any], metric: str, percentile: str) -> float:
    summary = report.get(metric)
    if not isinstance(summary, dict):
        raise ValueError(f"{metric} summary is missing")
    return _positive(summary.get(percentile), f"{metric}.{percentile}")


def _same_required_field(
    qcc: dict[str, Any], baseline: dict[str, Any], field: str
) -> Any:
    left = qcc.get(field)
    right = baseline.get(field)
    if left != right:
        raise ValueError(f"matched serving reports disagree on {field}")
    if left is None:
        raise ValueError(f"serving reports require {field}")
    return left


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 1.0e-12)


def compare_reports(qcc: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Return a machine-readable paired comparison without inventing missing data."""

    if qcc.get("label") == baseline.get("label"):
        raise ValueError("QCC and baseline labels must differ")
    model_id = qcc.get("model_id", qcc.get("model"))
    baseline_model_id = baseline.get("model_id", baseline.get("model"))
    if model_id != baseline_model_id or model_id is None:
        raise ValueError("matched serving reports disagree on model_id")
    context_length = _same_required_field(qcc, baseline, "context_length")
    concurrency = _same_required_field(qcc, baseline, "concurrency")
    max_tokens = _same_required_field(qcc, baseline, "max_tokens")
    workload = _same_required_field(qcc, baseline, "workload")
    num_requests = _same_required_field(qcc, baseline, "num_requests")
    if _same_required_field(qcc, baseline, "vllm_version") is None:
        raise ValueError("serving reports require vllm_version")
    if not isinstance(context_length, int) or context_length <= 0:
        raise ValueError("context_length must be a positive integer")
    if not isinstance(concurrency, int) or concurrency <= 0:
        raise ValueError("concurrency must be a positive integer")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if not isinstance(num_requests, int) or num_requests <= 0:
        raise ValueError("num_requests must be a positive integer")

    qcc_ttft_p50 = _summary_value(qcc, "ttft_s", "p50")
    base_ttft_p50 = _summary_value(baseline, "ttft_s", "p50")
    qcc_tpot_p50 = _summary_value(qcc, "tpot_s", "p50")
    base_tpot_p50 = _summary_value(baseline, "tpot_s", "p50")
    qcc_tpot_p95 = _summary_value(qcc, "tpot_s", "p95")
    base_tpot_p95 = _summary_value(baseline, "tpot_s", "p95")
    qcc_tpot_p99 = _summary_value(qcc, "tpot_s", "p99")
    base_tpot_p99 = _summary_value(baseline, "tpot_s", "p99")
    qcc_throughput = _positive(
        qcc.get("throughput_tokens_per_s"), "qcc.throughput_tokens_per_s"
    )
    base_throughput = _positive(
        baseline.get("throughput_tokens_per_s"),
        "baseline.throughput_tokens_per_s",
    )
    qcc_request_throughput = _positive(
        qcc.get("request_throughput_per_s"), "qcc.request_throughput_per_s"
    )
    base_request_throughput = _positive(
        baseline.get("request_throughput_per_s"),
        "baseline.request_throughput_per_s",
    )

    ttft_regression = _ratio(qcc_ttft_p50, base_ttft_p50) - 1.0
    p50_tpot_speedup = _ratio(base_tpot_p50, qcc_tpot_p50)
    p95_tpot_speedup = _ratio(base_tpot_p95, qcc_tpot_p95)
    p99_tpot_speedup = _ratio(base_tpot_p99, qcc_tpot_p99)
    throughput_speedup = _ratio(qcc_throughput, base_throughput)
    request_throughput_speedup = _ratio(qcc_request_throughput, base_request_throughput)

    qcc_failures = int(qcc.get("failed_requests", -1))
    baseline_failures = int(baseline.get("failed_requests", -1))
    all_requests_succeeded = (
        qcc_failures == 0
        and baseline_failures == 0
        and qcc.get("successful_requests") == num_requests
        and baseline.get("successful_requests") == num_requests
    )
    throughput_latency_tradeoff = (
        throughput_speedup < 1.0
        or p95_tpot_speedup < 1.0
        or p99_tpot_speedup < 1.0
    )
    qcc_dominates = (
        all_requests_succeeded
        and ttft_regression <= 0.0
        and p95_tpot_speedup >= 1.0
        and p99_tpot_speedup >= 1.0
        and throughput_speedup >= 1.0
    )

    qcc_memory = qcc.get("server_peak_gpu_memory_mib")
    baseline_memory = baseline.get("server_peak_gpu_memory_mib")
    memory_reduction = None
    if qcc_memory is not None and baseline_memory is not None:
        qcc_memory = _positive(qcc_memory, "qcc.server_peak_gpu_memory_mib")
        baseline_memory = _positive(
            baseline_memory, "baseline.server_peak_gpu_memory_mib"
        )
        memory_reduction = 1.0 - qcc_memory / baseline_memory

    derived = {
        "ttft_regression": ttft_regression,
        "p50_tpot_speedup": p50_tpot_speedup,
        "p95_tpot_speedup": p95_tpot_speedup,
        "p99_tpot_speedup": p99_tpot_speedup,
        "throughput_speedup": throughput_speedup,
        "request_throughput_speedup": request_throughput_speedup,
        "server_peak_gpu_memory_reduction": memory_reduction,
        "all_requests_succeeded": all_requests_succeeded,
        "throughput_latency_tradeoff": throughput_latency_tradeoff,
        "qcc_dominates": qcc_dominates,
    }
    provenance = {
        "model_id": model_id,
        "context_length": context_length,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "workload": workload,
        "vllm_version": qcc.get("vllm_version"),
        "run_id": qcc.get("run_id") if qcc.get("run_id") == baseline.get("run_id") else None,
        "matched_full_kv": True,
    }
    return {
        "schema": "qcc-vllm-serving-comparison-v1",
        "qcc_label": qcc.get("label"),
        "baseline_label": baseline.get("label"),
        "provenance": provenance,
        "derived": derived,
        "vllm_latency": {
            **provenance,
            "stock_vllm": True,
            "context_tokens": context_length,
            "tpot_speedup": p50_tpot_speedup,
            "throughput_speedup": throughput_speedup,
        },
        "production_latency": {
            **provenance,
            "ttft_regression": ttft_regression,
            "p95_tpot_speedup": p95_tpot_speedup,
            "p99_tpot_speedup": p99_tpot_speedup,
            "throughput_latency_tradeoff": throughput_latency_tradeoff,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qcc-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare_reports(
            load_report(args.qcc_report), load_report(args.baseline_report)
        )
    except ValueError as exc:
        print(json.dumps({"matched": False, "error": str(exc)}, indent=2))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
