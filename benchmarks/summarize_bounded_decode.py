"""Summarize frontier-harness result JSON: retention by (length, policy, budget),
cache bytes, and speed.

Usage: python benchmarks/summarize_bounded_decode.py <results.json> [...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

# Llama-3.2-1B geometry
LAYERS = 16
KV_HEADS = 8
HEAD_DIM = 64
DTYPE_BYTES = 2


def cache_bytes(budget, context_tokens):
    per_slot = LAYERS * KV_HEADS * HEAD_DIM * DTYPE_BYTES * 2  # K and V
    return per_slot * budget, per_slot * context_tokens


def main(paths):
    rows = []
    for path in paths:
        rows.extend(json.load(open(path))["results"])

    agg = defaultdict(lambda: [0.0, 0])
    for r in rows:
        key = (r["context_tokens"], r["policy"], r["budget"])
        agg[key][0] += r["answer_recall"]
        agg[key][1] += 1

    print(f"{'L':>8} {'policy':<12} {'B':>6} {'recall':>12} {'cache':>12} {'fullKV':>12} {'ratio':>10}")
    for key in sorted(agg, key=lambda k: (k[0], k[1], -1 if k[2] is None else k[2])):
        L, pol, b = key
        c, n = agg[key]
        if b is None:
            cb, fb, ratio = "-", "-", "-"
        else:
            cb, fb = cache_bytes(b, L)
            cb = f"{cb/2**20:.2f}MiB" if cb < 2**30 else f"{cb/2**30:.2f}GiB"
            fb = f"{fb/2**20:.0f}MiB" if fb < 2**30 else f"{fb/2**30:.2f}GiB"
            ratio = f"{cache_bytes(b,L)[1]/cache_bytes(b,L)[0]:.0f}x"
        print(f"{L:>8} {pol:<12} {str(b):>6} {int(c)}/{n} = {c/n:.3f} {cb:>12} {fb:>12} {ratio:>10}")

    # per-policy totals at each budget
    print("\nRetention vs Full-KV on records where Full-KV is correct")
    full_ok = {}
    for r in rows:
        if r["policy"] == "full":
            full_ok[(r.get("seed", 0), r["context_tokens"])] = r["answer_recall"] >= 1.0
    tot = defaultdict(lambda: [0.0, 0])
    for r in rows:
        if r["policy"] == "full":
            continue
        if not full_ok.get((r.get("seed", 0), r["context_tokens"]), False):
            continue
        key = (r["context_tokens"], r["policy"], r["budget"])
        tot[key][0] += r["answer_recall"]
        tot[key][1] += 1
    for key in sorted(tot, key=lambda k: (k[0], k[2] or 0, k[1])):
        c, n = tot[key]
        print(f"  L={key[0]:>7} {key[1]:<12} B={str(key[2]):>5} {int(c)}/{n} = {c/n:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
