"""Assemble independently reproduced cross-model/cross-GPU evidence."""
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


def assemble(reports: list[dict[str, Any]], *, run_id: str, model_id: str) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("generalization requires at least two independent reports")
    model_ids: list[str] = []
    model_families: list[str] = []
    gpu_generations: list[str] = []
    reproduction_ids: list[str] = []
    for index, report in enumerate(reports):
        if report.get("real_model") is not True or report.get("synthetic") is not False:
            raise ValueError(f"report {index} is not real non-synthetic evidence")
        if report.get("protocol_locked") is not True or report.get("matched_full_kv") is not True:
            raise ValueError(f"report {index} is not locked matched evidence")
        for field in ("model_id", "model_family", "gpu_generation", "reproduction_run_id"):
            if not isinstance(report.get(field), str) or not report[field]:
                raise ValueError(f"report {index}.{field} is required")
        if report.get("passed") is not True:
            raise ValueError(f"report {index} did not pass its local acceptance checks")
        model_ids.append(report["model_id"])
        model_families.append(report["model_family"])
        gpu_generations.append(report["gpu_generation"])
        reproduction_ids.append(report["reproduction_run_id"])
    if model_id not in model_ids:
        raise ValueError("primary model_id is absent from generalization reports")
    if len(set(model_families)) < 2:
        raise ValueError("generalization needs two model families")
    if len(set(gpu_generations)) < 2:
        raise ValueError("generalization needs two GPU generations")
    if len(set(reproduction_ids)) < 2:
        raise ValueError("generalization needs two independent reproductions")
    return {
        "run_id": run_id,
        "model_id": model_id,
        "matched_full_kv": True,
        "real_model": True,
        "official": False,
        "protocol_locked": True,
        "synthetic": False,
        "qcc_only": False,
        "model_families": len(set(model_families)),
        "gpu_generations": len(set(gpu_generations)),
        "independent_reproductions": len(set(reproduction_ids)),
        "model_ids": sorted(set(model_ids)),
        "gpu_generation_ids": sorted(set(gpu_generations)),
        "reproduction_run_ids": sorted(set(reproduction_ids)),
        "reports": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = assemble([_load(path) for path in args.evidence], run_id=args.run_id, model_id=args.model_id)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "failures": [str(exc)]}, indent=2))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "model_families": result["model_families"], "gpu_generations": result["gpu_generations"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
