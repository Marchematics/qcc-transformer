"""Re-score stored RULER predictions with the *official* metric.

RULER's `scripts/eval/synthetic/constants.py` defines

    string_match_all(preds, refs) = mean over records of
        (number of reference strings found in the prediction) / (number of references)

i.e. partial recall per record, not all-or-nothing.  An earlier version of the
harness required every reference string to be present, which understates any
multi-reference task (only `vt` here).  This script recomputes both numbers from
the saved predictions so the difference is visible.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def official(pred: str, refs: list[str]) -> float:
    if not refs:
        return 0.0
    low = pred.lower()
    return sum(1.0 for r in refs if r.lower() in low) / len(refs)


def strict(pred: str, refs: list[str]) -> float:
    return 1.0 if refs and all(r.lower() in pred.lower() for r in refs) else 0.0


def main(path):
    rows = json.load(open(path))["results"]
    for r in rows:
        r["_off"] = official(r["prediction"], r["outputs"])
        r["_strict"] = strict(r["prediction"], r["outputs"])
    full = {r["row_id"]: r for r in rows if r["policy"] == "full"}
    print(f"{path}: {len(full)} records")
    print(f"{'task':<18}{'policy':<10}{'B':>6}{'official':>10}{'strict':>8}{'n':>4}")
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])  # off, strict, fk_off, fk_strict, n
    for r in rows:
        if r["policy"] == "full":
            continue
        f = full[r["row_id"]]
        k = (r["task"], r["policy"], r["budget"])
        a = agg[k]
        a[0] += r["_off"]
        a[1] += r["_strict"]
        a[2] += f["_off"]
        a[3] += f["_strict"]
        a[4] += 1
    tot = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
    for (task, pol, b), a in agg.items():
        t = tot[(pol, b)]
        for i in range(5):
            t[i] += a[i]
    for k in sorted(tot, key=lambda x: (x[1], x[0])):
        off, strict_v, fk_off, fk_strict, n = tot[k]
        print(f"{'ALL':<18}{k[0]:<10}{k[1]:>6}{off / fk_off:>10.3f}{strict_v / fk_strict:>8.3f}{n:>4}")
    print("\nper task (official metric, ratio to Full-KV):")
    tasks = sorted({k[0] for k in agg})
    budgets = sorted({k[2] for k in agg})
    for pol in sorted({k[1] for k in agg}):
        for b in budgets:
            line = []
            for t in tasks:
                a = agg.get((t, pol, b))
                if a and a[2] > 0:
                    line.append(f"{t}={a[0] / a[2]:.3f}")
            if line:
                print(f"  {pol:<10} B={b:>6}  " + "  ".join(line))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
