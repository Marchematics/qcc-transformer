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


def load_manifest_targets(path: Path) -> dict[int, tuple[str, str]]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("retrieval manifest requires a header and trials")
    header = json.loads(lines[0])
    if header.get("schema") != "qcc-real-retrieval-manifest-v1":
        raise ValueError("unsupported retrieval manifest schema")
    if header.get("protocol_locked") is not True:
        raise ValueError("retrieval manifest is not protocol-locked")
    if int(header.get("trials", 0)) < 1000 or int(header.get("context_tokens", 0)) < 1_000_000:
        raise ValueError("retrieval manifest must contain >=1000 trials at >=1M context")
    for field in ("random_depth", "multi_needle", "semantic_distractor"):
        if header.get(field) is not True:
            raise ValueError(f"retrieval manifest is missing {field}")
    rows: dict[int, tuple[str, str]] = {}
    for line in lines[1:]:
        trial = json.loads(line)
        index = trial.get("trial")
        target = trial.get("target_entity")
        expected = trial.get("expected")
        if not isinstance(index, int) or index in rows:
            raise ValueError("retrieval manifest trial ids must be unique integers")
        if not isinstance(target, str) or not isinstance(expected, str):
            raise ValueError(f"manifest trial {index} has no target/expected pair")
        rows[index] = (target, expected)
    if len(rows) != int(header.get("trials", -1)):
        raise ValueError("retrieval manifest trial count does not match its header")
    if set(rows) != set(range(len(rows))):
        raise ValueError("retrieval manifest trial ids must be the contiguous registered range")
    return rows


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
    catastrophic = 0
    buckets: dict[str, dict[str, int]] = {}
    for index in sorted(full_rows):
        f = full_rows[index]
        q = qcc_rows[index]
        if f.get("expected") != q.get("expected") or f.get("target_entity") != q.get("target_entity"):
            raise ValueError(f"trial {index} target mismatch")
        if (f.get("target_entity"), f.get("expected")) != manifest_targets[index]:
            raise ValueError(f"trial {index} does not match the registered manifest")
        for label, row in (("Full-KV", f), ("QCC", q)):
            if row.get("input_tokens") != full["context_tokens"]:
                raise ValueError(f"trial {index} {label} did not execute the full context length")
            depth = row.get("target_depth")
            if not isinstance(depth, (int, float)) or not 0.0 < float(depth) < 1.0:
                raise ValueError(f"trial {index} {label} has no valid target depth")
        if f.get("depth_bucket") != q.get("depth_bucket"):
            raise ValueError(f"trial {index} depth bucket mismatch")
        f_ok = bool(f.get("correct"))
        q_ok = bool(q.get("correct"))
        full_correct += int(f_ok)
        qcc_correct += int(q_ok)
        catastrophic += int(f_ok and not q_ok)
        bucket_name = f.get("depth_bucket") or q.get("depth_bucket") or "execution-failure"
        bucket = buckets.setdefault(bucket_name, {"trials": 0, "full": 0, "qcc": 0})
        bucket["trials"] += 1
        bucket["full"] += int(f_ok)
        bucket["qcc"] += int(q_ok)

    trials = len(full_rows)
    full_rate = full_correct / trials
    qcc_rate = qcc_correct / trials
    if full.get("correct") != full_correct or qcc.get("correct") != qcc_correct:
        raise ValueError("summary correct counts do not match paired trial rows")
    critical_buckets = []
    for name, counts in sorted(buckets.items()):
        full_bucket_rate = counts["full"] / counts["trials"]
        qcc_bucket_rate = counts["qcc"] / counts["trials"]
        ratio = qcc_bucket_rate / full_bucket_rate if full_bucket_rate > 0 else 0.0
        critical_buckets.append(
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
            "random_depth": full["random_depth"],
            "multi_needle": full["multi_needle"],
            "semantic_distractor": full["semantic_distractor"],
            "oracle_admission": False,
        },
        "tail_safety": {
            **common,
            "catastrophic_retrieval_misses": catastrophic,
            "catastrophic_retrieval_miss_rate": catastrophic_rate,
            "catastrophic_retrieval_miss_rate_trials": catastrophic_rate_trials,
            "critical_buckets": critical_buckets,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
