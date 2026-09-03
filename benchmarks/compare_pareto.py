"""Compute matched QCC Pareto dominance over FP8 and KV-compression baselines.

Input reports are intentionally small metric records.  A dominance bit supplied
by a caller is never trusted: this utility recomputes it from quality, state,
tail latency, and throughput measured on the same workload.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _number(report: dict[str, Any], field: str) -> float:
    value = report.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} is missing or non-numeric")
    return float(value)


def _metrics(report: dict[str, Any]) -> dict[str, float]:
    result = {
        "quality_score": _number(report, "quality_score"),
        "state_bytes": _number(report, "attention_state_bytes"),
        "p95_tpot_ms": _number(report, "p95_tpot_ms"),
        "throughput_tokens_per_s": _number(report, "throughput_tokens_per_s"),
    }
    if result["quality_score"] < 0 or result["state_bytes"] < 0 or result["p95_tpot_ms"] <= 0 or result["throughput_tokens_per_s"] <= 0:
        raise ValueError("Pareto metrics must be non-negative with positive latency/throughput")
    return result


def compare(qcc: dict[str, Any], baselines: dict[str, dict[str, Any]], *, run_id: str, model_id: str) -> dict[str, Any]:
    if "fp8_full_kv" not in baselines or len(baselines) < 3:
        raise ValueError("Pareto comparison requires fp8_full_kv plus two compression baselines")
    identity_fields = ("run_id", "model_id", "workload_sha256", "context_tokens", "gpu")
    for name, report in (("qcc", qcc), *baselines.items()):
        if report.get("run_id") != run_id or report.get("model_id") != model_id:
            raise ValueError(f"{name} provenance does not match run/model")
        if report.get("real_model") is not True or report.get("synthetic") is not False:
            raise ValueError(f"{name} is not real non-synthetic evidence")
        if report.get("protocol_locked") is not True:
            raise ValueError(f"{name} protocol is not locked")
        _metrics(report)
    for name, report in baselines.items():
        for field in identity_fields[2:]:
            if qcc.get(field) != report.get(field):
                raise ValueError(f"qcc and {name} are not matched for {field}")
    qcc_metrics = _metrics(qcc)
    output = {
        "run_id": run_id,
        "model_id": model_id,
        "matched_full_kv": True,
        "real_model": True,
        "official": False,
        "protocol_locked": True,
        "synthetic": False,
        "qcc_only": False,
        "workload_sha256": qcc["workload_sha256"],
        "context_tokens": qcc["context_tokens"],
        "gpu": qcc["gpu"],
        "baselines": [],
    }
    for name, report in baselines.items():
        baseline_metrics = _metrics(report)
        quality = qcc_metrics["quality_score"] >= baseline_metrics["quality_score"]
        state = qcc_metrics["state_bytes"] <= baseline_metrics["state_bytes"]
        latency = qcc_metrics["p95_tpot_ms"] <= baseline_metrics["p95_tpot_ms"]
        throughput = qcc_metrics["throughput_tokens_per_s"] >= baseline_metrics["throughput_tokens_per_s"]
        strict = any((
            qcc_metrics["quality_score"] > baseline_metrics["quality_score"],
            qcc_metrics["state_bytes"] < baseline_metrics["state_bytes"],
            qcc_metrics["p95_tpot_ms"] < baseline_metrics["p95_tpot_ms"],
            qcc_metrics["throughput_tokens_per_s"] > baseline_metrics["throughput_tokens_per_s"],
        ))
        output["baselines"].append({
            "name": name,
            "qcc_dominates": bool(quality and state and latency and throughput and strict),
            "baseline_quality_score": baseline_metrics["quality_score"],
            "qcc_quality_score": qcc_metrics["quality_score"],
            "baseline_state_bytes": baseline_metrics["state_bytes"],
            "qcc_state_bytes": qcc_metrics["state_bytes"],
            "baseline_p95_tpot_ms": baseline_metrics["p95_tpot_ms"],
            "qcc_p95_tpot_ms": qcc_metrics["p95_tpot_ms"],
            "baseline_throughput_tokens_per_s": baseline_metrics["throughput_tokens_per_s"],
            "qcc_throughput_tokens_per_s": qcc_metrics["throughput_tokens_per_s"],
        })
    output["all_baselines_dominated"] = all(item["qcc_dominates"] for item in output["baselines"])
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--qcc", type=Path, required=True)
    parser.add_argument("--baseline", action="append", required=True, metavar="NAME=JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baselines: dict[str, dict[str, Any]] = {}
    for item in args.baseline:
        if "=" not in item:
            raise SystemExit("--baseline must be NAME=JSON")
        name, raw_path = item.split("=", 1)
        if not name or name in baselines:
            raise SystemExit("baseline names must be non-empty and unique")
        baselines[name] = _load(Path(raw_path))
    try:
        result = compare(_load(args.qcc), baselines, run_id=args.run_id, model_id=args.model_id)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "failures": [str(exc)]}, indent=2))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "all_baselines_dominated": result["all_baselines_dominated"]}, indent=2))
    return 0 if result["all_baselines_dominated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
