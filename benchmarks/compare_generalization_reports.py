"""Summarize cross-family, cross-GPU, independent real-model reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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
    evaluations: list[dict[str, Any]] = []
    families: set[str] = set()
    gpu_generations: set[str] = set()
    reproductions: set[str] = set()
    for index, report in enumerate(reports):
        if report.get("model_id") is None or report.get("run_id") != run_id:
            raise ValueError(f"report {index} has incompatible model/run provenance")
        if report.get("real_model") is not True or report.get("synthetic") is not False:
            raise ValueError(f"report {index} is not marked as real non-synthetic evidence")
        family = report.get("model_family")
        gpu_generation = report.get("gpu_generation")
        reproduction = report.get("reproduction_id")
        if not all(isinstance(item, str) and item for item in (family, gpu_generation, reproduction)):
            raise ValueError(
                f"report {index} requires model_family, gpu_generation, and reproduction_id"
            )
        families.add(family)
        gpu_generations.add(gpu_generation)
        reproductions.add(reproduction)
        evaluations.append(
            {
                "model_id": report["model_id"],
                "model_family": family,
                "gpu": report.get("gpu"),
                "gpu_generation": gpu_generation,
                "reproduction_id": reproduction,
                "source": report.get("source", report.get("label")),
            }
        )
    return {
        "schema": "qcc-generalization-v1",
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
