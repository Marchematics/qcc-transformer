"""Summarize cross-family, cross-GPU, independent real-model reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MIN_GENERALIZATION_CONTEXT = 128_000


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_context(value: Any, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < MIN_GENERALIZATION_CONTEXT
    ):
        raise ValueError(
            f"{name} must be an integer >= {MIN_GENERALIZATION_CONTEXT}"
        )
    return value


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read report {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"report {path} must contain a JSON object")
    return value


def summarize_reports(
    reports: list[dict[str, Any]], *, model_id: str, run_id: str
) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one report is required")
    _required_text(model_id, "model_id")
    _required_text(run_id, "run_id")
    evaluations: list[dict[str, Any]] = []
    families: set[str] = set()
    gpu_generations: set[str] = set()
    reproductions: set[str] = set()
    seen_reproductions: set[str] = set()
    for index, report in enumerate(reports):
        report_model_id = _required_text(
            report.get("model_id"), f"report {index}.model_id"
        )
        if report.get("run_id") != run_id:
            raise ValueError(f"report {index} has incompatible model/run provenance")
        if report.get("real_model") is not True or report.get("synthetic") is not False:
            raise ValueError(f"report {index} is not marked as real non-synthetic evidence")
        if report.get("matched_full_kv") is not True or report.get("qcc_only") is not False:
            raise ValueError(f"report {index} is not matched Full-KV/QCC evidence")
        if report.get("protocol_locked") is not True:
            raise ValueError(f"report {index} does not declare a registered protocol")
        context = _required_context(
            report.get("native_context_tokens"), f"report {index}.native_context_tokens"
        )
        family = report.get("model_family")
        gpu_generation = report.get("gpu_generation")
        reproduction = report.get("reproduction_id")
        source = _required_text(
            report.get("source", report.get("label")), f"report {index}.source"
        )
        if not all(
            isinstance(item, str) and item.strip()
            for item in (family, gpu_generation, reproduction)
        ):
            raise ValueError(
                f"report {index} requires model_family, gpu_generation, and reproduction_id"
            )
        if reproduction in seen_reproductions:
            raise ValueError(f"report {index} reuses reproduction_id {reproduction}")
        seen_reproductions.add(reproduction)
        families.add(family)
        gpu_generations.add(gpu_generation)
        reproductions.add(reproduction)
        evaluations.append(
            {
                "model_id": report_model_id,
                "model_family": family,
                "gpu": report.get("gpu"),
                "gpu_generation": gpu_generation,
                "reproduction_id": reproduction,
                "native_context_tokens": context,
                "source": source,
            }
        )
    return {
        "schema": "qcc-generalization-v2",
        "model_id": model_id,
        "run_id": run_id,
        "matched_full_kv": False,
        "real_model": True,
        "synthetic": False,
        "official": False,
        "protocol_locked": True,
        "qcc_only": False,
        "model_families": len(families),
        "gpu_generations": len(gpu_generations),
        "independent_reproductions": len(reproductions),
        "minimums_met": {
            "model_families": len(families) >= 2,
            "gpu_generations": len(gpu_generations) >= 2,
            "independent_reproductions": len(reproductions) >= 2,
        },
        "evaluations": evaluations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = summarize_reports(
            [_load(path) for path in args.report], model_id=args.model_id, run_id=args.run_id
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
