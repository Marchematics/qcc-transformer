#!/usr/bin/env python3
"""Pair Full-KV/QCC real 1M runs and emit retrieval/tail gate evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--full-summary", type=Path, required=True)
    parser.add_argument("--qcc-summary", type=Path, required=True)
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
        if payload.get("protocol_locked") is not True:
            raise ValueError(f"{mode} protocol is not locked")
        if payload.get("run_id") != args.run_id:
            raise ValueError(f"{mode} summary run_id does not match requested run")
        for field in ("random_depth", "multi_needle", "semantic_distractor"):
            if payload.get(field) is not True:
                raise ValueError(f"{mode} summary is missing {field} evidence")
    for field in ("model_id", "context_tokens", "trials"):
        if full.get(field) != qcc.get(field):
            raise ValueError(f"matched retrieval mismatch for {field}")
    if int(full["trials"]) < 1000 or int(full["context_tokens"]) < 1_000_000:
        raise ValueError("strict comparison requires >=1000 trials at >=1M context")
    if qcc.get("oracle_admission") is not False:
        raise ValueError("QCC 1M evidence must use learned, non-oracle admission")

    full_rows = load_rows(args.full_jsonl)
    qcc_rows = load_rows(args.qcc_jsonl)
    if full_rows.keys() != qcc_rows.keys() or len(full_rows) != int(full["trials"]):
        raise ValueError("paired trial rows are incomplete or mismatched")

    full_correct = 0
    qcc_correct = 0
    catastrophic = 0
    buckets: dict[str, dict[str, int]] = {}
    for index in sorted(full_rows):
        f = full_rows[index]
        q = qcc_rows[index]
        if f.get("expected") != q.get("expected") or f.get("target_entity") != q.get("target_entity"):
            raise ValueError(f"trial {index} target mismatch")
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
            "critical_buckets": critical_buckets,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
