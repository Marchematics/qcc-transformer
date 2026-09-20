"""Summarize a RULER frontier run: aggregate and worst-task retention.

Retention is measured only on records the matched Full-KV run answered, so a
task the pretrained model itself cannot do does not flatter or penalize the
bounded cache.  Both the raw mean answer recall and the Full-KV-conditioned
retention are printed, because they answer different questions.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict


def bucket(length):
    if length is None:
        return "?"
    if length < 12_000:
        return "8K"
    if length < 24_000:
        return "16K"
    if length < 48_000:
        return "32K"
    return "64K"


def main(path):
    rows = json.load(open(path))["results"]
    full_ok = {}
    full_tot = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["policy"] == "full":
            full_ok[r["row_id"]] = r["answer_recall"] >= 1.0
            full_tot[r["task"]][0] += r["answer_recall"]
            full_tot[r["task"]][1] += 1

    print("Full-KV by task: " + "  ".join(
        f"{t}={int(v[0])}/{v[1]}" for t, v in sorted(full_tot.items())))

    agg = defaultdict(lambda: [0.0, 0.0, 0])
    by_task = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        if r["policy"] == "full":
            continue
        k = (r["policy"], r["budget"])
        ok = full_ok.get(r["row_id"], False)
        agg[k][0] += r["answer_recall"]          # raw recall sum
        agg[k][1] += r["answer_recall"] if ok else 0.0  # conditioned
        agg[k][2] += 1 if ok else 0
        by_task[(r["task"], r["policy"], r["budget"])][0] += r["answer_recall"] if ok else 0.0
        by_task[(r["task"], r["policy"], r["budget"])][1] += 1 if ok else 0

    print(f"\n{'policy':<10} {'B':>6} {'raw recall':>11} {'retention':>10} {'n(FK ok)':>9} "
          f"{'worst task':>11}")
    summary = []
    for k in sorted(agg, key=lambda x: (x[0], -1 if x[1] is None else x[1])):
        c, cond, n = agg[k]
        retention = cond / n if n else float("nan")
        per_task = {}
        for t in sorted({t for (t, p, b) in by_task if (p, b) == k}):
            cc, nn = by_task[(t, k[0], k[1])]
            per_task[t] = cc / nn if nn else float("nan")
        worst = min(per_task.values()) if per_task else float("nan")
        worst_task = min(per_task, key=per_task.get) if per_task else "-"
        print(f"{k[0]:<10} {str(k[1]):>6} {c/len(rows):>11.3f} {retention:>10.3f} {n:>9} "
              f"{worst:>11.3f} ({worst_task})")
        summary.append({"policy": k[0], "budget": k[1], "raw_recall": round(c / len(rows), 4),
                        "retention": round(retention, 4), "worst_task": worst_task,
                        "worst_task_retention": round(worst, 4), "per_task": 
                        {t: round(v, 4) for t, v in per_task.items()}})
    out = path.replace(".json", ".summary.json")
    json.dump(summary, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
