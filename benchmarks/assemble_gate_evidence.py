"""Assemble a strict 99-gate evidence bundle from section JSON files.

Each benchmark writes its own provenance record.  This utility only joins
records after checking that every section uses one ``run_id`` and
``model_id``; it never fills in metrics or marks evidence as official.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SECTIONS = (
    "model", "quality", "vllm_latency", "memory", "calibration",
    "retrieval_1m", "tail_safety", "pareto_dominance", "production_latency",
    "scaling_law", "generalization",
)


def assemble(run_id: str, model_id: str, paths: dict[str, Path]) -> dict[str, Any]:
    if not run_id or not model_id:
        raise ValueError("run_id and model_id are required")
    bundle: dict[str, Any] = {"run_id": run_id}
    for section in SECTIONS:
        path = paths.get(section)
        if path is None:
            raise ValueError(f"missing --{section.replace('_', '-')}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read {section}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{section} evidence must be a JSON object")
        # Section comparators may emit one auditable multi-section file. Allow
        # that file to be supplied directly while still storing only the
        # requested section in the final bundle.
        if section in value and isinstance(value[section], dict):
            value = value[section]
        if section != "model":
            if value.get("run_id") != run_id:
                raise ValueError(f"{section}.run_id does not match {run_id}")
            if value.get("model_id") != model_id:
                raise ValueError(f"{section}.model_id does not match {model_id}")
        elif value.get("model_id") != model_id:
            raise ValueError(f"model.model_id does not match {model_id}")
        bundle[section] = value
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    for section in SECTIONS:
        parser.add_argument(f"--{section.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args()
    paths = {section: getattr(args, section) for section in SECTIONS}
    try:
        bundle = assemble(args.run_id, args.model_id, paths)
    except ValueError as exc:
        print(json.dumps({"passed": False, "failures": [str(exc)]}, indent=2))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sections": list(SECTIONS)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
