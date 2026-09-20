"""Recover results from a benchmark log even if the run was interrupted.

The frontier harness prints one ``[record]`` line per prompt and one result line
per (policy, budget).  Parsing the log yields the same fields the JSON would,
so a killed sweep still contributes evidence.

Usage: python parse_log.py expand_v1.log out.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

REC = re.compile(r"^\[record\] seed=(?P<seed>-?\d+) L=(?P<L>\d+)")
RULER = re.compile(r"^\[ruler\]\s+(?P<task>\S+)\s+L=\s*(?P<L>\d+)")
ROW = re.compile(
    r"^\s+(?P<policy>[a-z_]+)\s+B=\s*(?P<budget>None|\d+)\s+kept=\s*(?P<kept>\d+)\s+"
    r"recall=(?P<recall>\d+)(?P<rest>.*)$"
)
KV = re.compile(r"selAll=(?P<all>\d+)/(?P<units>\d+) keepFrac=(?P<frac>[0-9.]+)")


def parse(path):
    rows = []
    cur = {"seed": 0, "L": 0, "task": None}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = REC.match(line)
        if m:
            cur = {"seed": int(m.group("seed")), "L": int(m.group("L")), "task": None}
            continue
        m = RULER.match(line)
        if m:
            cur = {"seed": -1, "L": int(m.group("L")), "task": m.group("task")}
            continue
        m = ROW.match(line)
        if m:
            budget = None if m.group("budget") == "None" else int(m.group("budget"))
            row = {
                "seed": cur["seed"], "context_tokens": cur["L"], "task": cur["task"],
                "policy": m.group("policy"), "budget": budget,
                "kept_slots": int(m.group("kept")), "answer_recall": float(m.group("recall")),
            }
            k = KV.search(m.group("rest"))
            if k:
                row["selection"] = {
                    "units_with_all_answer_tokens": int(k.group("all")),
                    "units": int(k.group("units")),
                    "answer_token_keep_fraction": float(k.group("frac")),
                }
            rows.append(row)
    return rows


def main(path, out=None):
    rows = parse(path)
    if out:
        json.dump({"recovered_from": path, "results": rows}, open(out, "w"), indent=1)
    agg = defaultdict(lambda: [0.0, 0])
    for r in rows:
        key = (r["context_tokens"] // 10000, r["policy"], r["budget"])
        agg[key][0] += r["answer_recall"]
        agg[key][1] += 1
    print(f"{len(rows)} rows recovered from {path}")
    for k in sorted(agg, key=lambda x: (x[0], x[1], -1 if x[2] is None else x[2])):
        c, n = agg[k]
        print(f"  ~{k[0]*10}K {k[1]:<10} B={str(k[2]):>5} {int(c)}/{n} = {c/n:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
