#!/usr/bin/env python3
"""Pair Full-KV/QCC real 1M runs and emit retrieval/tail gate evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_MIN_REAL_PARAMS = 1_000_000_000
_MAX_REAL_PARAMS = 7_000_000_000


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        index = int(row["trial"])
        if index in rows:
            raise ValueError(f"duplicate trial {index} in {path}")
        rows[index] = row
    return rows


def load_manifest_targets(path: Path) -> dict[int, tuple[tuple[str, str], ...]]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("retrieval manifest requires a header and trials")
    header = json.loads(lines[0])
    if header.get("schema") != "qcc-real-retrieval-manifest-v2":
        raise ValueError("unsupported retrieval manifest schema")
    if header.get("protocol_locked") is not True:
        raise ValueError("retrieval manifest is not protocol-locked")
    if int(header.get("trials", 0)) < 1000 or int(header.get("context_tokens", 0)) < 1_000_000:
        raise ValueError("retrieval manifest must contain >=1000 trials at >=1M context")
    for field in ("random_depth", "multi_needle", "semantic_distractor"):
        if header.get(field) is not True:
            raise ValueError(f"retrieval manifest is missing {field}")
    if header.get("all_needles_required") is not True or int(header.get("needles", 0)) < 2:
        raise ValueError("retrieval manifest must require every needle")
    expected_needles = int(header["needles"])
    rows: dict[int, tuple[tuple[str, str], ...]] = {}
    for line in lines[1:]:
        trial = json.loads(line)
        index = trial.get("trial")
        if not isinstance(index, int) or index in rows:
            raise ValueError("retrieval manifest trial ids must be unique integers")
        records = trial.get("records")
        targets = trial.get("targets")
        if not isinstance(records, list) or not isinstance(targets, list) or len(targets) != expected_needles:
            raise ValueError(f"manifest trial {index} does not register every needle")
        needles = {
            record.get("entity"): record.get("code")
            for record in records
            if isinstance(record, dict) and record.get("kind") == "needle"
        }
        if len(needles) != expected_needles:
            raise ValueError(f"manifest trial {index} has an invalid needle set")
        pairs = []
        for target in targets:
            if not isinstance(target, dict):
                raise ValueError(f"manifest trial {index} has an invalid needle target")
            entity = target.get("entity")
            code = target.get("code")
            if not isinstance(entity, str) or not isinstance(code, str) or needles.get(entity) != code:
                raise ValueError(f"manifest trial {index} has an unregistered needle target")
            pairs.append((entity, code))
        if len(set(pairs)) != len(pairs):
            raise ValueError(f"manifest trial {index} repeats a needle target")
        if trial.get("target_entity") != pairs[0][0] or trial.get("expected") != pairs[0][1]:
            raise ValueError(f"manifest trial {index} has an invalid compatibility target pair")
        rows[index] = tuple(pairs)
    if len(rows) != int(header.get("trials", -1)):
        raise ValueError("retrieval manifest trial count does not match its header")
    if set(rows) != set(range(len(rows))):
        raise ValueError("retrieval manifest trial ids must be the contiguous registered range")
    return rows


def row_targets(row: dict[str, Any], label: str) -> tuple[tuple[str, str], ...]:
    targets = row.get("targets")
    if not isinstance(targets, list) or len(targets) < 2:
        raise ValueError(f"{label} is missing the complete needle target list")
    pairs = []
    for target in targets:
        if not isinstance(target, dict):
            raise ValueError(f"{label} contains an invalid needle target")
        entity = target.get("entity")
        code = target.get("code")
        if not isinstance(entity, str) or not isinstance(code, str):
            raise ValueError(f"{label} contains an invalid needle target pair")
        pairs.append((entity, code))
    return tuple(pairs)


def row_needle_results(
    row: dict[str, Any], targets: tuple[tuple[str, str], ...], label: str
) -> list[dict[str, Any]]:
    results = row.get("needle_results")
    if not isinstance(results, list) or len(results) != len(targets):
        raise ValueError(f"{label} does not score every needle")
    checked = []
    for index, (entity, expected) in enumerate(targets):
        result = results[index]
        if not isinstance(result, dict):
            raise ValueError(f"{label} contains an invalid needle result")
        if result.get("entity") != entity or result.get("expected") != expected:
            raise ValueError(f"{label} needle {index} does not match the registered target")
        depth = result.get("depth")
        bucket = result.get("depth_bucket")
        if not isinstance(depth, (int, float)) or not 0.0 < float(depth) < 1.0:
            raise ValueError(f"{label} needle {index} has no valid depth")
        if not isinstance(bucket, str) or not bucket:
            raise ValueError(f"{label} needle {index} has no depth bucket")
        checked.append(result)
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--full-summary", type=Path, required=True)
    parser.add_argument("--qcc-summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-jsonl", type=Path, required=True)
    parser.add_argument("--qcc-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    full = load(args.full_summary)
    qcc = load(args.qcc_summary)
    for payload, mode in ((full, "fullkv"), (qcc, "qcc")):
        if payload.get("schema") != "qcc-real-retrieval-result-v1" or payload.get("mode") != mode:
            raise ValueError(f"invalid {mode} summary")
        if payload.get("real_model") is not True or payload.get("synthetic") is not False:
            raise ValueError(f"{mode} summary is not real non-synthetic evidence")
        if payload.get("pretrained") is not True or payload.get("real_checkpoint") is not True:
            raise ValueError(f"{mode} summary is not a pretrained real checkpoint")
        if payload.get("protocol_locked") is not True:
            raise ValueError(f"{mode} protocol is not locked")
        if payload.get("all_needles_required") is not True:
            raise ValueError(f"{mode} summary does not require every needle")
        if payload.get("run_id") != args.run_id:
            raise ValueError(f"{mode} summary run_id does not match requested run")
        for field in ("random_depth", "multi_needle", "semantic_distractor"):
            if payload.get(field) is not True:
                raise ValueError(f"{mode} summary is missing {field} evidence")
    for field in ("model_id", "context_tokens", "trials", "parameter_count", "native_context_tokens"):
        if full.get(field) != qcc.get(field):
            raise ValueError(f"matched retrieval mismatch for {field}")
    if int(full["trials"]) < 1000 or int(full["context_tokens"]) < 1_000_000:
        raise ValueError("strict comparison requires >=1000 trials at >=1M context")
    parameter_count = full.get("parameter_count")
    if (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or not _MIN_REAL_PARAMS <= parameter_count <= _MAX_REAL_PARAMS
    ):
        raise ValueError("strict comparison requires a 1B..7B pretrained checkpoint")
    native_context = full.get("native_context_tokens")
    if (
        isinstance(native_context, bool)
        or not isinstance(native_context, int)
        or native_context < int(full["context_tokens"])
    ):
        raise ValueError("strict comparison requires native context >= locked context")
    if qcc.get("oracle_admission") is not False:
        raise ValueError("QCC 1M evidence must use learned, non-oracle admission")

    full_rows = load_rows(args.full_jsonl)
    qcc_rows = load_rows(args.qcc_jsonl)
    if full_rows.keys() != qcc_rows.keys() or len(full_rows) != int(full["trials"]):
        raise ValueError("paired trial rows are incomplete or mismatched")
    manifest_targets = load_manifest_targets(args.manifest)
    if full_rows.keys() != manifest_targets.keys():
        raise ValueError("paired trial rows do not cover the registered manifest")

    full_correct = 0
    qcc_correct = 0
    full_needle_correct = 0
    qcc_needle_correct = 0
    catastrophic = 0
    catastrophic_needle = 0
    trial_buckets: dict[str, dict[str, int]] = {}
    needle_buckets: dict[str, dict[str, int]] = {}
    for index in sorted(full_rows):
        f = full_rows[index]
        q = qcc_rows[index]
        manifest_targets_for_trial = manifest_targets[index]
        f_targets = row_targets(f, f"trial {index} Full-KV")
        q_targets = row_targets(q, f"trial {index} QCC")
        if f_targets != q_targets or f_targets != manifest_targets_for_trial:
            raise ValueError(f"trial {index} does not match the registered manifest")
        if f.get("expected_codes") != [code for _, code in f_targets] or q.get("expected_codes") != [code for _, code in q_targets]:
            raise ValueError(f"trial {index} expected-code list is incomplete")
        f_results = row_needle_results(f, f_targets, f"trial {index} Full-KV")
        q_results = row_needle_results(q, q_targets, f"trial {index} QCC")
        for label, row in (("Full-KV", f), ("QCC", q)):
            if row.get("input_tokens") != full["context_tokens"]:
                raise ValueError(f"trial {index} {label} did not execute the full context length")
            depth = row.get("target_depth")
            if not isinstance(depth, (int, float)) or not 0.0 < float(depth) < 1.0:
                raise ValueError(f"trial {index} {label} has no valid target depth")
        if f.get("depth_bucket") != f_results[0].get("depth_bucket") or q.get("depth_bucket") != q_results[0].get("depth_bucket"):
            raise ValueError(f"trial {index} compatibility depth bucket is inconsistent")
        for needle_index, (f_result, q_result) in enumerate(zip(f_results, q_results)):
            if f_result.get("depth_bucket") != q_result.get("depth_bucket"):
                raise ValueError(f"trial {index} needle {needle_index} depth bucket mismatch")
            f_needle_ok = bool(f_result.get("correct"))
            q_needle_ok = bool(q_result.get("correct"))
            full_needle_correct += int(f_needle_ok)
            qcc_needle_correct += int(q_needle_ok)
            catastrophic_needle += int(f_needle_ok and not q_needle_ok)
            bucket_name = str(f_result["depth_bucket"])
            bucket = needle_buckets.setdefault(bucket_name, {"needles": 0, "full": 0, "qcc": 0})
            bucket["needles"] += 1
            bucket["full"] += int(f_needle_ok)
            bucket["qcc"] += int(q_needle_ok)
        f_ok = all(bool(result.get("correct")) for result in f_results)
        q_ok = all(bool(result.get("correct")) for result in q_results)
        if bool(f.get("correct")) != f_ok or bool(q.get("correct")) != q_ok:
            raise ValueError(f"trial {index} all-needle score is inconsistent")
        full_correct += int(f_ok)
        qcc_correct += int(q_ok)
        catastrophic += int(f_ok and not q_ok)
        bucket_name = f.get("depth_bucket") or q.get("depth_bucket") or "execution-failure"
        bucket = trial_buckets.setdefault(bucket_name, {"trials": 0, "full": 0, "qcc": 0})
        bucket["trials"] += 1
        bucket["full"] += int(f_ok)
        bucket["qcc"] += int(q_ok)

    trials = len(full_rows)
    full_rate = full_correct / trials
    qcc_rate = qcc_correct / trials
    if full.get("correct") != full_correct or qcc.get("correct") != qcc_correct:
        raise ValueError("summary correct counts do not match paired trial rows")
    needle_count = sum(counts["needles"] for counts in needle_buckets.values())
    if full.get("needle_count") != needle_count or qcc.get("needle_count") != needle_count:
        raise ValueError("summary needle counts do not match paired needle rows")
    if full.get("needle_correct") != full_needle_correct or qcc.get("needle_correct") != qcc_needle_correct:
        raise ValueError("summary needle-correct counts do not match paired needle rows")
    critical_buckets = []
    for name, counts in sorted(needle_buckets.items()):
        full_bucket_rate = counts["full"] / counts["needles"]
        qcc_bucket_rate = counts["qcc"] / counts["needles"]
        ratio = qcc_bucket_rate / full_bucket_rate if full_bucket_rate > 0 else 0.0
        critical_buckets.append(
            {
                "name": name,
                "needles": counts["needles"],
                "unit": "needle",
                "full_kv_success_rate": full_bucket_rate,
                "qcc_success_rate": qcc_bucket_rate,
                "qcc_full_kv_ratio": ratio,
            }
        )
    trial_depth_buckets = []
    for name, counts in sorted(trial_buckets.items()):
        full_bucket_rate = counts["full"] / counts["trials"]
        qcc_bucket_rate = counts["qcc"] / counts["trials"]
        ratio = qcc_bucket_rate / full_bucket_rate if full_bucket_rate > 0 else 0.0
        trial_depth_buckets.append(
            {
                "name": name,
                "trials": counts["trials"],
                "full_kv_success_rate": full_bucket_rate,
                "qcc_success_rate": qcc_bucket_rate,
                "qcc_full_kv_ratio": ratio,
            }
        )
    catastrophic_rate = catastrophic / full_correct if full_correct else 1.0
    catastrophic_rate_trials = catastrophic / trials
    catastrophic_needle_rate = catastrophic_needle / full_needle_correct if full_needle_correct else 1.0
    common = {
        "run_id": args.run_id,
        "model_id": full["model_id"],
        "matched_full_kv": True,
        "real_model": True,
        "official": False,
        "protocol_locked": True,
        "synthetic": False,
        "qcc_only": False,
    }
    result = {
        "retrieval_1m": {
            **common,
            "trials": trials,
            "context_tokens": full["context_tokens"],
            "parameter_count": parameter_count,
            "native_context_tokens": native_context,
            "qcc_success_rate": qcc_rate,
            "full_kv_success_rate": full_rate,
            "qcc_full_kv_ratio": qcc_rate / full_rate if full_rate > 0 else 0.0,
            "all_needles_required": True,
            "needles_per_trial": len(manifest_targets[0]),
            "needle_count": needle_count,
            "needle_correct": qcc_needle_correct,
            "full_kv_needle_correct": full_needle_correct,
            "needle_success_rate": qcc_needle_correct / needle_count if needle_count else 0.0,
            "full_kv_needle_success_rate": full_needle_correct / needle_count if needle_count else 0.0,
            "random_depth": full["random_depth"],
            "multi_needle": full["multi_needle"],
            "semantic_distractor": full["semantic_distractor"],
            "depth_buckets": trial_depth_buckets,
            "needle_depth_buckets": critical_buckets,
            "oracle_admission": False,
        },
        "tail_safety": {
            **common,
            "catastrophic_retrieval_misses": catastrophic,
            "catastrophic_retrieval_miss_rate": catastrophic_rate,
            "catastrophic_retrieval_miss_rate_trials": catastrophic_rate_trials,
            "catastrophic_retrieval_needle_misses": catastrophic_needle,
            "catastrophic_retrieval_needle_miss_rate": catastrophic_needle_rate,
            "critical_buckets": critical_buckets,
            "trial_depth_buckets": trial_depth_buckets,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
