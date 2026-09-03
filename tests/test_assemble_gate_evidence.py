import json
from pathlib import Path

import pytest

from benchmarks.assemble_gate_evidence import SECTIONS, assemble


def _write_sections(tmp_path: Path, run_id: str = "r1", model_id: str = "m1") -> dict[str, Path]:
    paths = {}
    for section in SECTIONS:
        payload = {"model_id": model_id, "run_id": run_id}
        if section == "model":
            payload.pop("run_id")
        path = tmp_path / f"{section}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[section] = path
    return paths


def test_assemble_requires_one_run_and_model(tmp_path):
    bundle = assemble("r1", "m1", _write_sections(tmp_path))
    assert bundle["run_id"] == "r1"
    assert set(bundle) == {"run_id", *SECTIONS}


def test_assemble_rejects_mismatched_section(tmp_path):
    paths = _write_sections(tmp_path)
    paths["memory"].write_text(json.dumps({"run_id": "r2", "model_id": "m1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="memory.run_id"):
        assemble("r1", "m1", paths)


def test_assemble_unwraps_comparison_section(tmp_path):
    paths = _write_sections(tmp_path)
    paths["memory"].write_text(
        json.dumps({"schema": "comparison", "memory": {"run_id": "r1", "model_id": "m1"}}),
        encoding="utf-8",
    )
    bundle = assemble("r1", "m1", paths)
    assert bundle["memory"] == {"run_id": "r1", "model_id": "m1"}
