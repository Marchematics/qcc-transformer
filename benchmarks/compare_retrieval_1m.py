"""Pair the strict 1M Full-KV and QCC retrieval runs.

The two mode-specific runners are intentionally independent processes.  This
utility joins their per-trial JSONL only when the manifest, model, and run are
identical, then recomputes both rates and every depth-bucket tail.  A missing
or failed trial remains a miss; summary numbers supplied by a caller are not
trusted.
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


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _rows(path: Path, expected: int, name: str) -> dict[int, dict[str, Any]]:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        raise ValueError(f"cannot read {name} trial output: {exc}") from exc
    result: dict[int, dict[str, Any]] = {}
    for index, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} trial line {index} is invalid JSON: {exc}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("trial"), int):
            raise ValueError(f"{name} trial line {index} is malformed")
        trial = int(row["trial"])
        if trial in result:
            raise ValueError(f"{name} contains duplicate trial {trial}")
        result[trial] = row
    if len(result) != expected:
        raise ValueError(f"{name} contains {len(result)} trials; expected {expected}")
    return result


def _validate_report(report: dict[str, Any], mode: str, *, name: str) -> None:
    if report.get("schema") != "qcc-real-retrieval-result-v1":
        raise ValueError(f"{name} has an unsupported retrieval schema")
    if report.get("mode") != mode:
        raise ValueError(f"{name} must be mode={mode}")
    if report.get("real_model") is not True or report.get("synthetic") is not False:
        raise ValueError(f"{name} is not real non-synthetic evidence")
    if report.get("pretrained") is not True or report.get("real_checkpoint") is not True:
        raise ValueError(f"{name} is not a pretrained real checkpoint")
    if report.get("protocol_locked") is not True:
        raise ValueError(f"{name} protocol is not locked")


def compare(full: dict[str, Any], qcc: dict[str, Any]) -> dict[str, Any]:
    _validate_report(full, "fullkv", name="Full-KV")
    _validate_report(qcc, "qcc", name="QCC")
    for field in ("run_id", "model_id", "manifest_sha256", "context_tokens", "trials", "native_context_tokens"):
        if full.get(field) != qcc.get(field):
            raise ValueError(f"retrieval reports are not matched for {field}")
    run_id = full.get("run_id")
    model_id = full.get("model_id")
    if not isinstance(run_id, str) or not run_id or not isinstance(model_id, str) or not model_id:
        raise ValueError("retrieval reports require run_id and model_id")
    context = _number(full.get("context_tokens"), "context_tokens")
    trials = int(_number(full.get("trials"), "trials"))
    native = _number(full.get("native_context_tokens"), "native_context_tokens")
    if context < 1_000_000 or trials < 1000 or native < context:
        raise ValueError("strict retrieval requires >=1M context, >=1000 trials, and native context")
    if full.get("random_depth") is not True or qcc.get("random_depth") is not True:
        raise ValueError("retrieval evidence lacks random-depth trials")
    if full.get("multi_needle") is not True or qcc.get("multi_needle") is not True:
        raise ValueError("retrieval evidence lacks multi-needle trials")
    if full.get("semantic_distractor") is not True or qcc.get("semantic_distractor") is not True:
        raise ValueError("retrieval evidence lacks semantic distractors")
    if qcc.get("oracle_admission") is not False:
        raise ValueError("QCC retrieval must use learned/non-oracle admission")
    full_path = full.get("output_jsonl")
    qcc_path = qcc.get("output_jsonl")
    if not isinstance(full_path, str) or not isinstance(qcc_path, str):
        raise ValueError("retrieval reports must retain output_jsonl paths")
    full_rows = _rows(Path(full_path), trials, "Full-KV")
    qcc_rows = _rows(Path(qcc_path), trials, "QCC")
    if set(full_rows) != set(qcc_rows):
        raise ValueError("Full-KV and QCC trial IDs are not paired")

    total = len(full_rows)
    full_correct = 0
    qcc_correct = 0
    catastrophic = 0
    buckets: dict[str, dict[str, int]] = {}
    for trial in sorted(full_rows):
        full_row = full_rows[trial]
        qcc_row = qcc_rows[trial]
        for field in ("expected", "target_entity", "depth_bucket", "target_depth", "input_tokens"):
            if full_row.get(field) != qcc_row.get(field):
                raise ValueError(f"trial {trial} is not paired for {field}")
        full_ok = full_row.get("correct") is True
        qcc_ok = qcc_row.get("correct") is True
        full_correct += int(full_ok)
        qcc_correct += int(qcc_ok)
        catastrophic += int(full_ok and not qcc_ok)
        bucket = full_row.get("depth_bucket")
        if not isinstance(bucket, str) or not bucket:
            bucket = "execution-failure"
        counts = buckets.setdefault(bucket, {"trials": 0, "full_kv_correct": 0, "qcc_correct": 0})
        counts["trials"] += 1
        counts["full_kv_correct"] += int(full_ok)
        counts["qcc_correct"] += int(qcc_ok)

    critical_buckets: list[dict[str, Any]] = []
    for bucket, counts in sorted(buckets.items()):
        full_rate = counts["full_kv_correct"] / counts["trials"]
        qcc_rate = counts["qcc_correct"] / counts["trials"]
        critical_buckets.append({
            "task": "1m_retrieval",
            "context_tokens": int(context),
            "bucket": bucket,
            "trials": counts["trials"],
            "full_kv_score": full_rate,
            "qcc_score": qcc_rate,
            "qcc_full_kv_ratio": qcc_rate / full_rate if full_rate > 0 else 0.0,
        })

    common = {
        "run_id": run_id,
        "model_id": model_id,
        "matched_full_kv": True,
        "real_model": True,
        "official": False,
        "protocol_locked": True,
        "synthetic": False,
        "qcc_only": False,
        "context_tokens": int(context),
        "native_context_tokens": int(native),
        "manifest_sha256": full["manifest_sha256"],
        "trials": total,
        "random_depth": True,
        "multi_needle": True,
        "semantic_distractor": True,
    }
    retrieval = {
        **common,
        "qcc_success_rate": qcc_correct / total,
        "full_kv_success_rate": full_correct / total,
        "qcc_correct": qcc_correct,
        "full_kv_correct": full_correct,
        "oracle_admission": False,
        "full_kv_output_jsonl": full_path,
        "qcc_output_jsonl": qcc_path,
    }
    tail = {
        **common,
        "catastrophic_retrieval_miss_rate": catastrophic / total,
        "critical_buckets": critical_buckets,
        "bucket_definition": "target depth in the 1M random-depth multi-needle semantic-distractor manifest",
    }
    return {"retrieval_1m": retrieval, "tail_safety": tail}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-summary", type=Path, required=True)
    parser.add_argument("--qcc-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(_load(args.full_summary), _load(args.qcc_summary))
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
