"""Percentiles across repeated TPOT runs, instead of one contended number.

Reads the JSONs written by ``benchmark_bounded_decode_tpot_floor.py`` (one per
repeat), collects each variant's per-token latency, and reports the median and
the 95th/99th percentile across repeats.  A variant that OOMs in a repeat is
reported as such rather than dropped, because "the baseline cannot run here" is
part of the result.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return round(ordered[index], 3)


def main():
    args = parse_args()
    grouped = {}
    for path in args.paths:
        data = json.loads(Path(path).read_text())
        length = data["context_tokens"]
        for variant, entry in data["variants"].items():
            grouped.setdefault((length, variant), []).append(entry)

    print("| context | variant | repeats | ok | p50 ms | p95 ms | min | max |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|")
    summary = {}
    for (length, variant), entries in sorted(grouped.items()):
        times = [e["tpot_ms"] for e in entries if "tpot_ms" in e]
        errors = [e.get("error", "")[:60] for e in entries if "tpot_ms" not in e]
        row = {"repeats": len(entries), "ok": len(times),
               "p50": percentile(times, 0.5), "p95": percentile(times, 0.95),
               "min": round(min(times), 3) if times else None,
               "max": round(max(times), 3) if times else None,
               "mean": round(statistics.fmean(times), 3) if times else None,
               "errors": errors[:1]}
        summary[f"{length}:{variant}"] = row
        print(f"| {length} | {variant} | {row['repeats']} | {row['ok']} | {row['p50']} "
              f"| {row['p95']} | {row['min']} | {row['max']} |")
        if errors:
            print(f"  ({variant} at {length}: {len(errors)} failed, e.g. {errors[0]})")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
