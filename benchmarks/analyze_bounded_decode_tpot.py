"""Decode-TPOT comparison between matched Full-KV and bounded exact-KV decode.

Reads a result JSON produced by benchmark_bounded_decode_frontier.py (which
records, per record and policy, the wall time of the greedy decode loop and the
number of generated tokens) and reports per-token latency and speedup.

This is a batch-1, greedy, Python-loop HF measurement.  The bounded-cache
latency is expected to be flat in context length; the Full-KV latency grows
with it.  The floor is set by weight reads and per-step framework overhead, so
this measurement bounds what cache bounding alone can achieve on this setup.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict

LAYERS, KV_HEADS, HEAD_DIM, DTYPE_BYTES = 16, 8, 64, 2
BYTES_PER_SLOT = LAYERS * KV_HEADS * HEAD_DIM * DTYPE_BYTES * 2


def band(L):
    return "~32K" if L < 50_000 else ("~64K" if L < 100_000 else "~128K")


def main(paths, policies=("obs_last", "obs_mean")):
    rows = []
    for p in paths:
        rows.extend(json.load(open(p))["results"])
    by_record = defaultdict(dict)
    for r in rows:
        by_record[(r.get("seed"), r.get("context_tokens"))][(r["policy"], r["budget"])] = r

    out = defaultdict(lambda: defaultdict(list))
    for (seed, ctx), table in by_record.items():
        full = table.get(("full", None))
        if not full or not full.get("generated"):
            continue
        full_tpot = full["decode_s"] / full["generated"]
        for (pol, budget), r in table.items():
            if pol == "full" or pol not in policies or not r.get("generated"):
                continue
            tpot = r["decode_s"] / r["generated"]
            key = (band(ctx), pol, budget)
            out[key]["full_ms"].append(full_tpot * 1000)
            out[key]["bounded_ms"].append(tpot * 1000)
            out[key]["speedup"].append(full_tpot / tpot)
            out[key]["cache_bytes"].append(BYTES_PER_SLOT * budget)
            out[key]["full_kv_bytes"].append(BYTES_PER_SLOT * ctx)
            out[key]["recall"].append(r["answer_recall"])

    print(f"{'band':>6} {'policy':<10} {'B':>6} {'FullKV ms':>10} {'bounded ms':>11} "
          f"{'TPOT x':>7} {'state x':>8} {'recall':>7} {'n':>3}")
    summary = []
    for key in sorted(out, key=lambda k: (k[0], k[1], k[2])):
        v = out[key]
        full_ms = statistics.mean(v["full_ms"])
        bnd_ms = statistics.mean(v["bounded_ms"])
        sp = statistics.mean(v["speedup"])
        st = statistics.mean(v["full_kv_bytes"]) / statistics.mean(v["cache_bytes"])
        rec = statistics.mean(v["recall"])
        print(f"{key[0]:>6} {key[1]:<10} {key[2]:>6} {full_ms:>10.2f} {bnd_ms:>11.2f} "
              f"{sp:>6.2f}x {st:>7.0f}x {rec:>7.3f} {len(v['recall']):>3}")
        summary.append({"band": key[0], "policy": key[1], "budget": key[2],
                        "full_kv_tpot_ms": round(full_ms, 3), "bounded_tpot_ms": round(bnd_ms, 3),
                        "tpot_speedup": round(sp, 3), "decode_state_reduction": round(st, 1),
                        "mean_recall": round(rec, 4), "records": len(v["recall"])})
    print("\nNote: batch-1 greedy decode inside a Python loop on one A10G.  Bounded-cache")
    print("latency is flat in context length; the residual floor is weight reads plus")
    print("per-step framework overhead, not KV traffic.")
    print(json.dumps(summary, indent=1)[:200], "...")
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
