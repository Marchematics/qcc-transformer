"""Pair streaming stock-vLLM reports and emit serving gate sections.

Both reports must come from the same workload, model, context length, vLLM
version, GPU label, and run.  Percentiles are computed by the client benchmark;
this utility only takes ratios and refuses missing tails or incomplete requests.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _scalar(report: dict[str, Any], field: str) -> float:
    value = report.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} is missing from serving report")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _summary_metric(report: dict[str, Any], section: str, field: str) -> float:
    summary = report.get(section)
    if not isinstance(summary, dict):
        raise ValueError(f"{section} summary is missing")
    result = summary.get(field)
    if not isinstance(result, (int, float)) or isinstance(result, bool):
        raise ValueError(f"{section}.{field} is missing from serving report")
    result = float(result)
    if not math.isfinite(result):
        raise ValueError(f"{section}.{field} must be finite")
    return result


def _resolve_report(report: dict[str, Any], name: str) -> dict[str, Any]:
    """Accept either one serving point or a fixed-SLA sweep summary."""

    if report.get("schema") == "qcc-vllm-serving-v1":
        return report
    if report.get("schema") == "qcc-vllm-sweep-v1":
        selected = report.get("selected_report")
        if not isinstance(selected, dict):
            raise ValueError(f"{name} sweep has no SLA-eligible selected report")
        return _resolve_report(selected, name)
    raise ValueError(f"{name} report has an unsupported schema")


def _sla_limits(report: dict[str, Any], name: str) -> tuple[float, float]:
    sla = report.get("sla")
    if not isinstance(sla, dict) or report.get("sla_pass") is not True:
        raise ValueError(f"{name} report is not a passing fixed-SLA point")
    ttft = sla.get("ttft_p95_limit_ms")
    tpot = sla.get("tpot_p95_limit_ms")
    if not isinstance(ttft, (int, float)) or isinstance(ttft, bool) or not isinstance(tpot, (int, float)) or isinstance(tpot, bool):
        raise ValueError(f"{name} report has no fixed SLA limits")
    if float(ttft) <= 0 or float(tpot) <= 0:
        raise ValueError(f"{name} report has invalid fixed SLA limits")
    return float(ttft), float(tpot)


def _validate_sla(report: dict[str, Any], name: str) -> tuple[float, float]:
    limits = _sla_limits(report, name)
    if int(report.get("successful_requests", 0)) != int(report.get("num_requests", -1)):
        raise ValueError(f"{name} fixed-SLA point is incomplete")
    observed_ttft = _summary_metric(report, "ttft_s", "p95") * 1000.0
    observed_tpot = _summary_metric(report, "tpot_s", "p95") * 1000.0
    if observed_ttft > limits[0] or observed_tpot > limits[1]:
        raise ValueError(f"{name} report claims SLA pass but p95 exceeds its recorded limit")
    return limits


def compare(
    full: dict[str, Any],
    qcc: dict[str, Any],
    *,
    run_id: str,
    model_id: str,
    full_attention_state_bytes: int,
    qcc_attention_state_bytes: int,
) -> dict[str, dict[str, Any]]:
    full = _resolve_report(full, "Full-KV")
    qcc = _resolve_report(qcc, "QCC")
    for name, report in (("full_kv", full), ("qcc", qcc)):
        _validate_sla(report, name)
        if report.get("run_id") != run_id or report.get("model_id") != model_id:
            raise ValueError(f"{name} report provenance does not match run/model")
        if report.get("real_model") is not True or report.get("synthetic") is not False:
            raise ValueError(f"{name} report is not real non-synthetic evidence")
        if report.get("stock_vllm") is not True or report.get("streaming") is not True:
            raise ValueError(f"{name} report is not stock streaming vLLM evidence")
        if int(report.get("failed_requests", -1)) != 0:
            raise ValueError(f"{name} report contains failed requests")
        if int(report.get("successful_requests", 0)) != int(report.get("num_requests", -1)):
            raise ValueError(f"{name} report is incomplete")
    fields = ("context_length", "workload_sha256", "vllm_version", "gpu", "num_requests", "max_tokens")
    for field in fields:
        if full.get(field) != qcc.get(field):
            raise ValueError(f"serving reports are not matched for {field}")
    if _validate_sla(full, "Full-KV") != _validate_sla(qcc, "QCC"):
        raise ValueError("serving reports are not matched for the fixed SLA")
    context = full.get("context_length")
    if not isinstance(context, int) or context < 128_000:
        raise ValueError("matched stock-vLLM serving evidence requires context_length >= 128K")
    if not isinstance(full.get("workload_sha256"), str) or len(full["workload_sha256"]) != 64:
        raise ValueError("serving report must include a workload SHA-256")
    if full.get("vllm_version") in (None, "") or full.get("gpu") in (None, ""):
        raise ValueError("vLLM version and GPU identity are required")
    if full_attention_state_bytes <= 0 or qcc_attention_state_bytes < 0:
        raise ValueError("attention state byte counts are invalid")

    full_tpot = _summary_metric(full, "tpot_s", "p50")
    qcc_tpot = _summary_metric(qcc, "tpot_s", "p50")
    full_p95_tpot = _summary_metric(full, "tpot_s", "p95")
    qcc_p95_tpot = _summary_metric(qcc, "tpot_s", "p95")
    full_p99_tpot = _summary_metric(full, "tpot_s", "p99")
    qcc_p99_tpot = _summary_metric(qcc, "tpot_s", "p99")
    full_p95_ttft = _summary_metric(full, "ttft_s", "p95")
    qcc_p95_ttft = _summary_metric(qcc, "ttft_s", "p95")
    full_p99_ttft = _summary_metric(full, "ttft_s", "p99")
    qcc_p99_ttft = _summary_metric(qcc, "ttft_s", "p99")
    throughput_full = _scalar(full, "throughput_tokens_per_s")
    throughput_qcc = _scalar(qcc, "throughput_tokens_per_s")
    peak_full = full.get("server_peak_gpu_memory_bytes")
    peak_qcc = qcc.get("server_peak_gpu_memory_bytes")
    if not isinstance(peak_full, (int, float)) or not isinstance(peak_qcc, (int, float)):
        raise ValueError("server peak GPU memory is missing")
    peak_full = float(peak_full)
    peak_qcc = float(peak_qcc)
    if peak_full <= 0 or peak_qcc < 0:
        raise ValueError("server peak GPU memory is invalid")

    common = {
        "run_id": run_id,
        "model_id": model_id,
        "matched_full_kv": True,
        "real_model": True,
        "official": False,
        "protocol_locked": True,
        "synthetic": False,
        "qcc_only": False,
        "stock_vllm": True,
        "context_tokens": context,
        "vllm_version": full["vllm_version"],
        "gpu": full["gpu"],
        "workload_sha256": full["workload_sha256"],
    }
    state_reduction = 1.0 - qcc_attention_state_bytes / full_attention_state_bytes
    peak_reduction = 1.0 - peak_qcc / peak_full
    return {
        "vllm_latency": {
            **common,
            # The headline serving gate uses p95 TPOT so a favorable median
            # cannot hide a long-tail regression. Keep p50 as a diagnostic.
            "tpot_speedup": full_p95_tpot / max(qcc_p95_tpot, 1e-12),
            "p50_tpot_speedup": full_tpot / max(qcc_tpot, 1e-12),
            "p95_tpot_speedup": full_p95_tpot / max(qcc_p95_tpot, 1e-12),
            "throughput_speedup": float(throughput_qcc) / max(float(throughput_full), 1e-12),
            "full_kv_tpot_s": full_tpot,
            "qcc_tpot_s": qcc_tpot,
            "full_kv_p95_tpot_s": full_p95_tpot,
            "qcc_p95_tpot_s": qcc_p95_tpot,
        },
        "memory": {
            **common,
            "full_kv_attention_state_bytes": full_attention_state_bytes,
            "qcc_attention_state_bytes": qcc_attention_state_bytes,
            "full_kv_peak_memory_bytes": int(peak_full),
            "qcc_peak_memory_bytes": int(peak_qcc),
            "peak_memory_reduction": peak_reduction,
            "attention_state_reduction": state_reduction,
            "full_kv_concurrency": int(full.get("concurrency", 0)),
            "qcc_concurrency": int(qcc.get("concurrency", 0)),
            "fixed_sla": True,
        },
        "production_latency": {
            **common,
            "ttft_regression": qcc_p95_ttft / max(full_p95_ttft, 1e-12) - 1.0,
            "p95_ttft_speedup": full_p95_ttft / max(qcc_p95_ttft, 1e-12),
            "p99_ttft_speedup": full_p99_ttft / max(qcc_p99_ttft, 1e-12),
            "p95_tpot_speedup": full_p95_tpot / max(qcc_p95_tpot, 1e-12),
            "p99_tpot_speedup": full_p99_tpot / max(qcc_p99_tpot, 1e-12),
            "throughput_latency_tradeoff": bool(
                throughput_qcc < throughput_full
                or qcc_p95_ttft > full_p95_ttft
                or qcc_p99_ttft > full_p99_ttft
                or qcc_p95_tpot > full_p95_tpot
                or qcc_p99_tpot > full_p99_tpot
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--full-report", type=Path, required=True)
    parser.add_argument("--qcc-report", type=Path, required=True)
    parser.add_argument("--full-attention-state-bytes", type=int, required=True)
    parser.add_argument("--qcc-attention-state-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(
            _load(args.full_report), _load(args.qcc_report), run_id=args.run_id,
            model_id=args.model_id,
            full_attention_state_bytes=args.full_attention_state_bytes,
            qcc_attention_state_bytes=args.qcc_attention_state_bytes,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "failures": [str(exc)]}, indent=2))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sidecars = {}
    for section, value in result.items():
        sidecar = args.output.with_name(f"{args.output.stem}.{section}.json")
        sidecar.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        sidecars[section] = str(sidecar)
    print(json.dumps({"output": str(args.output), "sections": sidecars}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
