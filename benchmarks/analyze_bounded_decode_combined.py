"""Combined, Full-KV-conditioned retention across several frontier runs.

Merges any number of ``benchmark_bounded_decode_frontier.py`` result JSONs, bands
records by context length, and reports retention twice:

  raw        - mean answer recall over all records of the band
  retention  - mean answer recall over the records the matched Full-KV run
               answered, which is the quantity the "quality vs Full-KV" target
               actually refers to

Usage::

    python benchmarks/analyze_bounded_decode_combined.py \
        artifacts/bounded-decode-frontier-aggregate-v2.json \
        artifacts/bounded-decode-frontier-expand-v1.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def band(L):
    if L < 50_000:
        return "~32K"
    if L < 100_000:
        return "~64K"
    return "~128K"


def main(paths):
    rows = []
    for p in paths:
        rows.extend(json.load(open(p))["results"])
    full = {}
    for r in rows:
        if r["policy"] == "full":
            full[(r.get("seed"), r["context_tokens"])] = r["answer_recall"] >= 1.0
    bands = defaultdict(lambda: [0, 0])
    for (seed, ctx), ok in full.items():
        bands[band(ctx)][0] += int(ok)
        bands[band(ctx)][1] += 1
    print("Full-KV correct: " + "  ".join(
        f"{b}={c}/{n}" for b, (c, n) in sorted(bands.items())))

    agg = defaultdict(lambda: [0.0, 0.0, 0, 0])  # raw, cond, n_all, n_cond
    for r in rows:
        if r["policy"] == "full":
            continue
        k = (band(r["context_tokens"]), r["policy"], r["budget"])
        ok = full.get((r.get("seed"), r["context_tokens"]), False)
        agg[k][0] += r["answer_recall"]
        agg[k][1] += r["answer_recall"] if ok else 0.0
        agg[k][2] += 1
        agg[k][3] += int(ok)

    print(f"\n{'band':>6} {'policy':<10} {'B':>6} {'raw':>7} {'retention':>10} {'n(FKok)':>8}")
    for k in sorted(agg, key=lambda x: (x[0], x[2] or 0, x[1])):
        raw, cond, n, nc = agg[k]
        print(f"{k[0]:>6} {k[1]:<10} {str(k[2]):>6} {raw / n:>7.3f} "
              f"{cond / nc if nc else float('nan'):>10.3f} {nc:>8}")

    print("\nTotals per (policy, budget) over all bands")
    tot = defaultdict(lambda: [0.0, 0.0, 0, 0])
    for (b, pol, budget), v in agg.items():
        t = tot[(pol, budget)]
        for i in range(4):
            t[i] += v[i]
    for k in sorted(tot, key=lambda x: (x[1] or 0, x[0])):
        raw, cond, n, nc = tot[k]
        print(f"  {k[0]:<10} B={str(k[1]):>5} raw={raw / n:.3f} "
              f"retention={cond / nc if nc else float('nan'):.3f} n={n} (FK ok {nc})")


if __name__ == "__main__":
    main(sys.argv[1:])
