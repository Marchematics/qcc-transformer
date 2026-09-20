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
    """Official RULER scoring: `string_match_all` (partial per-record recall).

    Retention is the ratio of summed scores (QCC / matched Full-KV) over the
    same records, reported overall and per task; the strict all-or-nothing
    number is printed alongside for transparency.
    """
    rows = json.load(open(path))["results"]
    # recompute both metrics from the stored predictions so old (strict-scored)
    # and new files are handled identically
    for r in rows:
        outs = [o for o in r.get("outputs", []) if o]
        low = r["prediction"].lower()
        r["answer_recall"] = (sum(1.0 for o in outs if o.lower() in low) / len(outs)
                              if outs else 0.0)
        r["answer_recall_strict"] = (1.0 if outs and all(o.lower() in low for o in outs)
                                     else 0.0)
    full = {r["row_id"]: r for r in rows if r["policy"] == "full"}
    print(f"{len(full)} records; Full-KV by task: " + "  ".join(
        f"{t}={sum(r['answer_recall'] for r in full.values() if r['task'] == t):.1f}/"
        f"{sum(1 for r in full.values() if r['task'] == t)}"
        for t in sorted({r['task'] for r in full.values()})))

    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for r in rows:
        if r["policy"] == "full":
            continue
        f = full[r["row_id"]]
        k = (r["task"], r["policy"], r["budget"])
        a = agg[k]
        a[0] += r["answer_recall"]
        a[1] += f["answer_recall"]
        a[2] += r.get("answer_recall_strict", 0.0)
        a[3] += f.get("answer_recall_strict", 0.0)

    print(f"\n{'policy':<10}{'B':>6}{'official':>10}{'strict':>8}{'worst task':>24}")
    summary = []
    tot = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    per_task = defaultdict(lambda: [0.0, 0.0])
    for (t, pol, b), a in agg.items():
        tt = tot[(pol, b)]
        for i in range(4):
            tt[i] += a[i]
        per_task[(pol, b, t)][0] += a[0]
        per_task[(pol, b, t)][1] += a[1]
    for k in sorted(tot, key=lambda x: (x[0], x[1])):
        off, fk, st, fkst = tot[k]
        tasks = {t: (v[0] / v[1] if v[1] else float("nan"))
                 for (pol, b, t), v in per_task.items() if (pol, b) == k}
        worst_task = min(tasks, key=tasks.get) if tasks else "-"
        worst = tasks.get(worst_task, float("nan"))
        print(f"{k[0]:<10}{k[1]:>6}{off / fk if fk else float('nan'):>10.3f}"
              f"{st / fkst if fkst else float('nan'):>8.3f}"
              f"{worst:>16.3f} ({worst_task})")
        summary.append({"policy": k[0], "budget": k[1],
                        "official_retention": round(off / fk, 4) if fk else None,
                        "strict_retention": round(st / fkst, 4) if fkst else None,
                        "worst_task": worst_task, "worst_task_retention": round(worst, 4),
                        "per_task": {t: round(v, 4) for t, v in sorted(tasks.items())}})
    out = path.replace(".json", ".summary.json")
    json.dump(summary, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
