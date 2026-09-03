"""Aggregate matched QCC comparisons against three or more stock-vLLM baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.compare_serving_reports import compare_reports, load_report


_STRONG_COMPRESSION_BASELINES = {
    "adakv",
    "cachegen",
    "compresskv",
    "gear",
    "h2o",
    "kivi",
    "kvpress",
    "pyramidkv",
    "quest",
    "rkv",
    "scalekv",
    "snapkv",
    "squeezekv",
    "streamingllm",
}


def _baseline_name(label: Any) -> str:
    if not isinstance(label, str) or not label.strip():
        raise ValueError("each baseline report needs a non-empty label")
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"fp8_fullkv", "fp8_full_kv", "fp8"}:
        return "fp8_full_kv"
    return normalized


def compare_pareto(qcc: dict[str, Any], baselines: list[dict[str, Any]]) -> dict[str, Any]:
    if len(baselines) < 3:
        raise ValueError("Pareto comparison requires FP8 plus at least two compression baselines")
    names: set[str] = set()
    comparisons: list[dict[str, Any]] = []
    for baseline in baselines:
        name = _baseline_name(baseline.get("label"))
        if name in names:
            raise ValueError(f"duplicate baseline label {name}")
        if name != "fp8_full_kv" and name not in _STRONG_COMPRESSION_BASELINES:
            raise ValueError(
                f"unsupported compression baseline {name}; use a registered strong KV baseline"
            )
        names.add(name)
        paired = compare_reports(qcc, baseline)
        if paired["provenance"].get("run_id") != qcc.get("run_id"):
            raise ValueError(
                f"baseline {name} is not from the same matched run as QCC"
            )
        memory_reduction = paired["derived"].get("server_peak_gpu_memory_reduction")
        memory_dominates = (
            isinstance(memory_reduction, (int, float)) and memory_reduction >= 0.0
        )
        comparisons.append(
            {
                "name": name,
                "label": baseline.get("label"),
                "memory_dominates": memory_dominates,
                "qcc_dominates": paired["derived"]["qcc_dominates"] and memory_dominates,
                "metrics": paired["derived"],
            }
        )
    if "fp8_full_kv" not in names:
        raise ValueError("Pareto comparison requires a baseline labelled FP8 Full-KV")
    compression = names - {"fp8_full_kv"}
    if len(compression) < 2:
        raise ValueError("Pareto comparison requires at least two strong KV compression baselines")
    provenance = comparisons and compare_reports(qcc, baselines[0])["provenance"]
    return {
        "schema": "qcc-vllm-pareto-comparison-v1",
        "model_id": provenance["model_id"],
        "run_id": provenance.get("run_id"),
        "matched_full_kv": True,
        "real_model": qcc.get("real_model"),
        "synthetic": qcc.get("synthetic"),
        "official": False,
        "protocol_locked": qcc.get("protocol_locked"),
        "qcc_only": False,
        "baselines": comparisons,
        "all_dominated": all(item["qcc_dominates"] for item in comparisons),
        "provenance": provenance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qcc-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare_pareto(
            load_report(args.qcc_report),
            [load_report(path) for path in args.baseline_report],
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
