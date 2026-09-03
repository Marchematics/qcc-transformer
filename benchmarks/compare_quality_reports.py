"""Build one matched quality report from official benchmark outputs.

RULER emits both modes in one file, while LongBench and PG-19 emit one file per
mode.  This utility joins those outputs only when model, run, split and task
coverage agree, then preserves per-task ratios for tail inspection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read quality report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"quality report {path} must contain a JSON object")
    return payload


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _metadata(report: dict[str, Any], name: str) -> tuple[str, str]:
    model_id = report.get("model_id")
    run_id = report.get("run_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(f"{name}.model_id is required")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"{name}.run_id is required")
    if report.get("real_model") is not True:
        raise ValueError(f"{name} is not marked real_model")
    if report.get("synthetic") is not False:
        raise ValueError(f"{name} is marked synthetic or lacks the field")
    if report.get("official") is not True:
        raise ValueError(f"{name} is not marked official")
    return model_id, run_id


def _ruler_scores(report: dict[str, Any]) -> tuple[float, float, dict[str, float]]:
    if report.get("benchmark") != "ruler":
        raise ValueError("RULER report has the wrong benchmark name")
    qcc = report.get("qcc_score")
    full = report.get("full_kv_score")
    if qcc is None and isinstance(report.get("qcc_retrofit"), dict):
        qcc = report["qcc_retrofit"].get("accuracy")
    if full is None and isinstance(report.get("baseline_full_kv"), dict):
        full = report["baseline_full_kv"].get("accuracy")
    raw_ratios = report.get("task_ratios")
    if not isinstance(raw_ratios, dict) or not raw_ratios:
        raise ValueError("ruler.task_ratios is required for task and length-tail checks")
    task_ratios = {
        str(key): _finite(value, f"ruler.task_ratios.{key}")
        for key, value in raw_ratios.items()
    }
    return _finite(qcc, "ruler.qcc_score"), _finite(full, "ruler.full_kv_score"), task_ratios


def _paired_scores(
    full: dict[str, Any], qcc: dict[str, Any], benchmark: str
) -> tuple[float, float, dict[str, float]]:
    full_model, full_run = _metadata(full, f"{benchmark}.full_kv")
    qcc_model, qcc_run = _metadata(qcc, f"{benchmark}.qcc")
    if (full_model, full_run) != (qcc_model, qcc_run):
        raise ValueError(f"{benchmark} Full-KV/QCC provenance does not match")
    if full.get("benchmark") != benchmark or qcc.get("benchmark") != benchmark:
        raise ValueError(f"{benchmark} reports have the wrong benchmark name")
    if full.get("mode") != "fullkv" or qcc.get("mode") != "qcc":
        raise ValueError(f"{benchmark} reports must contain fullkv and qcc modes")
    full_score = _finite(full.get("quality_score"), f"{benchmark}.full_kv_score")
    qcc_score = _finite(qcc.get("quality_score"), f"{benchmark}.qcc_score")
    full_tasks = full.get("dataset_scores")
    qcc_tasks = qcc.get("dataset_scores")
    if full_tasks is None and qcc_tasks is None:
        full_tasks = full.get("bucket_scores")
        qcc_tasks = qcc.get("bucket_scores")
    task_ratios: dict[str, float] = {}
    if isinstance(full_tasks, dict) or isinstance(qcc_tasks, dict):
        if not isinstance(full_tasks, dict) or not isinstance(qcc_tasks, dict):
            raise ValueError(f"{benchmark} task score maps are incomplete")
        if set(full_tasks) != set(qcc_tasks):
            raise ValueError(f"{benchmark} task sets do not match")
        for task in sorted(full_tasks):
            task_full = _finite(full_tasks[task], f"{benchmark}.{task}.full_kv_score")
            task_qcc = _finite(qcc_tasks[task], f"{benchmark}.{task}.qcc_score")
            if task_full <= 0:
                raise ValueError(f"{benchmark}.{task} Full-KV score must be positive")
            task_ratios[task] = task_qcc / task_full
    for field in ("split", "native_context_tokens"):
        if full.get(field) != qcc.get(field):
            raise ValueError(f"{benchmark} reports disagree on {field}")
    return qcc_score, full_score, task_ratios


def assemble_quality(
    ruler: dict[str, Any],
    longbench_full: dict[str, Any],
    longbench_qcc: dict[str, Any],
    pg19_full: dict[str, Any],
    pg19_qcc: dict[str, Any],
) -> dict[str, Any]:
    ruler_model, ruler_run = _metadata(ruler, "ruler")
    ruler_qcc, ruler_full, ruler_tasks = _ruler_scores(ruler)
    longbench_qcc_score, longbench_full_score, longbench_tasks = _paired_scores(
        longbench_full, longbench_qcc, "longbench"
    )
    pg19_qcc_score, pg19_full_score, pg19_tasks = _paired_scores(
        pg19_full, pg19_qcc, "pg19"
    )
    for report, name in (
        (longbench_full, "longbench.full_kv"),
        (pg19_full, "pg19.full_kv"),
    ):
        model_id, run_id = _metadata(report, name)
        if (model_id, run_id) != (ruler_model, ruler_run):
            raise ValueError(f"{name} provenance does not match ruler")

    common = {
        "model_id": ruler_model,
        "run_id": ruler_run,
        "matched_full_kv": True,
        "real_model": True,
        "official": True,
        "synthetic": False,
        "qcc_only": False,
    }
    def section(benchmark: str, qcc: float, full: float, tasks: dict[str, float]) -> dict[str, Any]:
        if full <= 0:
            raise ValueError(f"{benchmark} Full-KV score must be positive")
        return {
            **common,
            "benchmark": benchmark,
            "qcc_score": qcc,
            "full_kv_score": full,
            "qcc_full_kv_ratio": qcc / full,
            "task_ratios": tasks,
        }

    return {
        "schema": "qcc-quality-comparison-v1",
        "model_id": ruler_model,
        "run_id": ruler_run,
        "quality": {
            "ruler": section("ruler", ruler_qcc, ruler_full, ruler_tasks),
            "longbench": section("longbench", longbench_qcc_score, longbench_full_score, longbench_tasks),
            "pg19": section("pg19", pg19_qcc_score, pg19_full_score, pg19_tasks),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruler", type=Path, required=True)
    parser.add_argument("--longbench-fullkv", type=Path, required=True)
    parser.add_argument("--longbench-qcc", type=Path, required=True)
    parser.add_argument("--pg19-fullkv", type=Path, required=True)
    parser.add_argument("--pg19-qcc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = assemble_quality(
            load_json(args.ruler),
            load_json(args.longbench_fullkv),
            load_json(args.longbench_qcc),
            load_json(args.pg19_fullkv),
            load_json(args.pg19_qcc),
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
