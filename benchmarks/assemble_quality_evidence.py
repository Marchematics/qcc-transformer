"""Build the matched official RULER/LongBench/PG-19 quality section.

The individual runners deliberately write benchmark-native reports.  This adapter
is the only place that converts those reports to the production gate schema.  It
does not infer missing scores, drop failed examples, or treat a partial suite as a
full-suite result.
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


def _common(report: dict[str, Any], name: str, *, run_id: str, model_id: str) -> None:
    if report.get("run_id") != run_id:
        raise ValueError(f"{name}.run_id does not match {run_id}")
    if report.get("model_id") != model_id:
        raise ValueError(f"{name}.model_id does not match {model_id}")
    if report.get("real_model") is not True or report.get("synthetic") is not False:
        raise ValueError(f"{name} is not real non-synthetic evidence")
    if report.get("official") is not True:
        raise ValueError(f"{name} is not marked official")


def _score(report: dict[str, Any], field: str, name: str) -> float:
    value = report.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name}.{field} must be a finite number")
    return float(value)


def _ruler(report: dict[str, Any], *, run_id: str, model_id: str) -> dict[str, Any]:
    _common(report, "ruler", run_id=run_id, model_id=model_id)
    if report.get("benchmark") != "ruler":
        raise ValueError("ruler report has the wrong benchmark name")
    baseline = report.get("baseline_full_kv")
    qcc = report.get("qcc_retrofit")
    if not isinstance(baseline, dict) or not isinstance(qcc, dict):
        raise ValueError("ruler must contain baseline_full_kv and qcc_retrofit")
    full_records = baseline.get("records")
    qcc_records = qcc.get("records")
    if not isinstance(full_records, list) or not isinstance(qcc_records, list):
        raise ValueError("ruler reports must retain per-record results for pairing")
    if report.get("full_suite") is not True:
        raise ValueError("RULER evidence must cover the full prepared split")
    if len(full_records) == 0 or len(full_records) != len(qcc_records):
        raise ValueError("ruler Full-KV/QCC record sets are incomplete or unpaired")
    for index, (full_row, qcc_row) in enumerate(zip(full_records, qcc_records)):
        if not isinstance(full_row, dict) or not isinstance(qcc_row, dict):
            raise ValueError(f"ruler record {index} is malformed")
        if full_row.get("line") != qcc_row.get("line") or full_row.get("answers") != qcc_row.get("answers"):
            raise ValueError(f"ruler record {index} is not paired")
    full_score = _score(baseline, "accuracy", "ruler.baseline_full_kv")
    qcc_score = _score(qcc, "accuracy", "ruler.qcc_retrofit")
    evaluator = report.get("official_evaluator")
    if not isinstance(evaluator, str) or not evaluator:
        raise ValueError("ruler.official_evaluator is required")
    return {
        "model_id": model_id,
        "qcc_score": qcc_score,
        "full_kv_score": full_score,
        "records": len(full_records),
        "official": True,
        "official_evaluator": evaluator,
        "matched_full_kv": True,
        "real_model": True,
        "synthetic": False,
        "qcc_only": False,
        "run_id": run_id,
    }


def _paired_metric(
    full: dict[str, Any],
    qcc: dict[str, Any],
    name: str,
    *,
    run_id: str,
    model_id: str,
) -> dict[str, Any]:
    _common(full, f"{name}.full_kv", run_id=run_id, model_id=model_id)
    _common(qcc, f"{name}.qcc", run_id=run_id, model_id=model_id)
    if full.get("benchmark") != name or qcc.get("benchmark") != name:
        raise ValueError(f"{name} reports have the wrong benchmark name")
    if full.get("mode") != "fullkv" or qcc.get("mode") != "qcc":
        raise ValueError(f"{name} must provide explicit fullkv and qcc reports")
    if name == "longbench":
        if full.get("full_suite") is not True or qcc.get("full_suite") is not True:
            raise ValueError("LongBench evidence must cover the official full suite")
        if full.get("datasets") != qcc.get("datasets"):
            raise ValueError("LongBench dataset lists are not paired")
        if full.get("generated_rows") != qcc.get("generated_rows"):
            raise ValueError("LongBench generated row counts are not paired")
        full_scores = full.get("dataset_scores")
        qcc_scores = qcc.get("dataset_scores")
        if not isinstance(full_scores, dict) or not isinstance(qcc_scores, dict):
            raise ValueError("LongBench dataset scores are missing")
        if set(full_scores) != set(qcc_scores) or set(full_scores) != set(full.get("datasets", [])):
            raise ValueError("LongBench dataset scores do not cover the same full suite")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in (*full_scores.values(), *qcc_scores.values())):
            raise ValueError("LongBench dataset scores must be numeric")
        if int(full.get("generated_rows", 0)) <= 0:
            raise ValueError("LongBench has no generated rows")
    else:
        if full.get("full_suite") is not True or qcc.get("full_suite") is not True:
            raise ValueError(f"{name} evidence must cover the full official split")
        if full.get("split") != qcc.get("split") or full.get("split") != "test":
            raise ValueError(f"{name} must use the official test split")
        if int(full.get("documents", 0)) != int(qcc.get("documents", 0)):
            raise ValueError(f"{name} document counts are not paired")
        if int(full.get("predicted_tokens", 0)) <= 0 or int(qcc.get("predicted_tokens", 0)) <= 0:
            raise ValueError(f"{name} has no scored tokens")
    full_score = _score(full, "quality_score", f"{name}.full_kv")
    qcc_score = _score(qcc, "quality_score", f"{name}.qcc")
    evaluator = full.get("official_evaluator", full.get("official_source"))
    if not isinstance(evaluator, str) or not evaluator:
        raise ValueError(f"{name} official evaluator/source is required")
    return {
        "model_id": model_id,
        "qcc_score": qcc_score,
        "full_kv_score": full_score,
        "official": True,
        "official_evaluator": evaluator,
        "matched_full_kv": True,
        "real_model": True,
        "synthetic": False,
        "qcc_only": False,
        "run_id": run_id,
        "source_reports": {"full_kv": str(full.get("output", "")), "qcc": str(qcc.get("output", ""))},
    }


def assemble(
    *,
    run_id: str,
    model_id: str,
    ruler: Path,
    longbench_full: Path,
    longbench_qcc: Path,
    pg19_full: Path,
    pg19_qcc: Path,
) -> dict[str, Any]:
    if not run_id or not model_id:
        raise ValueError("run_id and model_id are required")
    ruler_report = _load(ruler)
    longbench_full_report = _load(longbench_full)
    longbench_qcc_report = _load(longbench_qcc)
    pg19_full_report = _load(pg19_full)
    pg19_qcc_report = _load(pg19_qcc)
    return {
        "ruler": _ruler(ruler_report, run_id=run_id, model_id=model_id),
        "longbench": _paired_metric(
            longbench_full_report, longbench_qcc_report, "longbench",
            run_id=run_id, model_id=model_id,
        ),
        "pg19": _paired_metric(
            pg19_full_report, pg19_qcc_report, "pg19",
            run_id=run_id, model_id=model_id,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--ruler", type=Path, required=True)
    parser.add_argument("--longbench-full", type=Path, required=True)
    parser.add_argument("--longbench-qcc", type=Path, required=True)
    parser.add_argument("--pg19-full", type=Path, required=True)
    parser.add_argument("--pg19-qcc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = assemble(
            run_id=args.run_id,
            model_id=args.model_id,
            ruler=args.ruler,
            longbench_full=args.longbench_full,
            longbench_qcc=args.longbench_qcc,
            pg19_full=args.pg19_full,
            pg19_qcc=args.pg19_qcc,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"passed": False, "failures": [str(exc)]}, indent=2))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "benchmarks": list(result)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
