"""Does the packaged API reproduce the benchmark harness on every record?

Compares the stored harness run (``ruler_v6.json``, policy ``lex_obs`` at
budget 4096) with a packaged-API run over the same RULER split, record by
record.  Records are matched on their reference answers, which are unique
needles/UUIDs, so no ordering assumption is needed.

Reports how many predictions are byte-identical, how many scores agree under
each of the two recall conventions, and where they differ - including which
side is right, because a package that *gains* a record the harness missed is a
different statement from a regression.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", default="artifacts/bounded-decode-frontier-ruler-v6.json")
    parser.add_argument("--package", required=True)
    parser.add_argument("--policy", default="lex_obs")
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--arm", default="bounded")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def key_of(row):
    return (row.get("task"), tuple(row.get("outputs") or []))


def main():
    args = parse_args()
    harness = json.load(open(args.harness))
    package = json.load(open(args.package))

    harness_rows = [row for row in harness["results"]
                    if row.get("policy") == args.policy and row.get("budget") == args.budget]
    package_rows = [row for row in package["records"] if row.get("arm") == args.arm]
    print(f"harness {args.policy}@{args.budget}: {len(harness_rows)} rows; "
          f"package {args.arm}: {len(package_rows)} rows")

    by_key = {}
    for row in harness_rows:
        by_key.setdefault(key_of(row), []).append(row)
    ambiguous = {key: len(rows) for key, rows in by_key.items() if len(rows) > 1}
    if ambiguous:
        print(f"note: {len(ambiguous)} answer keys appear more than once in the harness "
              f"run; the first is used")

    compared, missing = [], []
    for row in package_rows:
        candidates = by_key.get(key_of(row)) or []
        if not candidates:
            missing.append(row)
            continue
        reference = candidates[0]
        compared.append((reference, row))

    identical = [pair for pair in compared if pair[0]["prediction"] == pair[1]["prediction"]]
    partial_agree = [pair for pair in compared
                     if abs(pair[0].get("answer_recall", -1) - pair[1]["recall_partial"]) < 1e-9]
    strict_agree = [pair for pair in compared
                    if abs(pair[0].get("answer_recall_strict", -1) - pair[1]["recall_strict"]) < 1e-9]
    differences = []
    for reference, row in compared:
        if reference["prediction"] == row["prediction"]:
            continue
        differences.append({
            "task": row["task"], "prompt_tokens": row["prompt_tokens"],
            "harness_prediction": reference["prediction"][:60],
            "package_prediction": row["prediction"][:60],
            "harness_partial": reference.get("answer_recall"),
            "package_partial": row["recall_partial"],
            "harness_strict": reference.get("answer_recall_strict"),
            "package_strict": row["recall_strict"],
            "outputs": row["outputs"],
        })

    def mean(rows, field):
        values = [row[field] for row in rows]
        return round(sum(values) / len(values), 4) if values else None

    summary = {
        "harness_file": args.harness, "package_file": args.package,
        "policy": args.policy, "budget": args.budget, "arm": args.arm,
        "compared": len(compared), "unmatched_package_rows": len(missing),
        "predictions_identical": len(identical),
        "partial_agreement": len(partial_agree),
        "strict_agreement": len(strict_agree),
        "package_mean_partial": mean(compared and [row for _ref, row in compared], "recall_partial"),
        "harness_mean_partial": mean([ref for ref, _row in compared], "answer_recall"),
        "per_task": {},
        "differences": differences,
    }
    for task in sorted({row["task"] for _ref, row in compared}):
        pairs = [pair for pair in compared if pair[1]["task"] == task]
        summary["per_task"][task] = {
            "records": len(pairs),
            "identical": sum(1 for ref, row in pairs if ref["prediction"] == row["prediction"]),
            "package_mean_partial": mean([row for _ref, row in pairs], "recall_partial"),
            "harness_mean_partial": mean([ref for ref, _row in pairs], "answer_recall"),
        }
    if missing:
        summary["unmatched"] = [{"task": row["task"], "outputs": row["outputs"]}
                                for row in missing]
    print(json.dumps({k: v for k, v in summary.items() if k != "differences"}, indent=1))
    for difference in differences:
        gained = difference["package_partial"] > (difference["harness_partial"] or 0)
        print(f"  [{'package better' if gained else 'harness better'}] "
              f"{difference['task']} L={difference['prompt_tokens']} "
              f"harness={difference['harness_partial']} package={difference['package_partial']} "
              f"outputs={difference['outputs']}")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))
        print(f"wrote {args.out}")
    return summary


if __name__ == "__main__":
    main()
