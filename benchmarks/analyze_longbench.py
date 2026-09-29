#!/usr/bin/env python
"""Attribute a LongBench retention gap to the records that cause it.

``benchmark_retention_longbench.py`` stores one row per record and arm with the
prediction and the official score.  This reader answers the questions the report
asks about the worst task: is the shortfall spread over every record or carried by
a few, is it missing content or degenerate generation, and does a larger retained
budget move it?

Three views, all recomputed from the stored predictions:

* **per task** -- mean score per arm and the retention ratio, plus how much of the
  pooled loss the worst three records carry;
* **ROUGE-L precision/recall split** for summarisation tasks (needs the cached
  dataset for the references, `--cache-dir`), because an F1 drop from covering
  less of the reference and an F1 drop from writing more off-target text look the
  same in the F column;
* **repetition** -- distinct-word ratio and the share of the summary taken by its
  most repeated 8-gram beyond its first occurrence, with the correlation between
  the change in repetition and the change in score.

Usage::

    python benchmarks/analyze_longbench.py artifacts/longbench-retention-llama32-1b.json
    python benchmarks/analyze_longbench.py artifacts/longbench-*.json --task gov_report \\
        --cache-dir /root/qcc/data/longbench
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_artifact(path):
    with open(path) as handle:
        return json.load(handle)


def pair_records(payload, task):
    """``{record id: {"full": row, "bounded": row}}`` for one task."""
    pairs = collections.defaultdict(dict)
    for row in payload.get("records", []):
        if row.get("task") == task:
            pairs[row["id"]][row["arm"]] = row
    return pairs


def repetition(text):
    """``(words, distinct ratio, repeated 8-gram share)`` of a prediction.

    The third number is the share of the text taken by its most repeated 8-gram
    *beyond its first occurrence*, so a text in which every 8-gram is distinct
    scores 0.0 rather than ``8 / words``.
    """
    words = text.split()
    if not words:
        return 0, 0.0, 0.0
    grams = collections.Counter(tuple(words[i:i + 8]) for i in range(max(0, len(words) - 7)))
    repeats = grams.most_common(1)[0][1] - 1 if grams else 0
    share = min(1.0, repeats * 8 / len(words))
    return len(words), round(len(set(words)) / len(words), 4), round(share, 4)


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
    denom = statistics.pstdev(xs) * statistics.pstdev(ys)
    return round(cov / denom, 4) if denom else None


def loss_concentration(pairs):
    """Pooled loss over records, and the share carried by the worst three."""
    losses = []
    for rid, arms in pairs.items():
        full, bounded = arms.get("full"), arms.get("bounded")
        if not full or not bounded:
            continue
        delta = bounded["score"] - full["score"]
        if delta < 0:
            losses.append((full["row"], -delta))
    losses.sort(key=lambda item: -item[1])
    pooled = sum(value for _row, value in losses)
    worst = losses[:3]
    return {
        "records": len(pairs),
        "losing_records": len(losses),
        "pooled_loss": round(pooled, 4),
        "worst": [{"row": row, "loss": round(value, 4),
                   "share": round(value / pooled, 4) if pooled else None}
                  for row, value in worst],
        "worst_three_share": (round(sum(v for _r, v in worst) / pooled, 4)
                              if pooled else None),
    }


def degenerate_generation(pairs):
    """Repetition change against score change, over the paired records."""
    deltas, repetition_deltas, identical = [], [], 0
    for rid, arms in pairs.items():
        full, bounded = arms.get("full"), arms.get("bounded")
        if not full or not bounded:
            continue
        if full["prediction"] == bounded["prediction"]:
            identical += 1
        deltas.append(bounded["score"] - full["score"])
        _n, _u_full, share_full = repetition(full["prediction"])
        _n, _u_bounded, share_bounded = repetition(bounded["prediction"])
        repetition_deltas.append(share_bounded - share_full)
    return {
        "identical_predictions": identical,
        "mean_score_delta": round(statistics.mean(deltas), 4) if deltas else None,
        "mean_repetition_delta": (round(statistics.mean(repetition_deltas), 4)
                                  if repetition_deltas else None),
        "pearson_repetition_vs_score": pearson(repetition_deltas, deltas),
        "more_repetitive_records": sum(1 for value in repetition_deltas if value > 0.02),
    }


def rouge_split(payload, task, cache_dir):
    """Precision/recall means per arm, when the references can be loaded."""
    from benchmarks.longbench_data import load_records
    from benchmarks.longbench_metrics import _sentences, rouge_l_summary_level

    references = {record["_id"]: record["answers"][0]
                  for record in load_records([task], cache_dir=cache_dir)}
    pairs = pair_records(payload, task)
    per_arm = collections.defaultdict(lambda: collections.defaultdict(list))
    for rid, arms in pairs.items():
        reference = references.get(rid)
        if reference is None:
            continue
        for arm, row in arms.items():
            scores = rouge_l_summary_level(_sentences(row["prediction"]),
                                           _sentences(reference))
            for key in ("f", "p", "r"):
                per_arm[arm][key].append(scores[key])
    return {arm: {key: round(statistics.mean(values), 4)
                  for key, values in keys.items()}
            for arm, keys in per_arm.items()}


def matched_ratio(pairs):
    """``(mean full, mean bounded, mean ratio, matched records)``.

    The ratio follows the runner's convention: only records the Full-KV arm
    scored above zero count, because a ratio against a zero denominator is not a
    measurement.
    """
    full_rows = [(rid, arms["full"]) for rid, arms in pairs.items() if "full" in arms]
    bounded_rows = [(rid, arms["bounded"]) for rid, arms in pairs.items()
                    if "bounded" in arms]
    if not full_rows or not bounded_rows:
        return None
    ratios = [row["score"] / arms["full"]["score"]
              for rid, row in bounded_rows
              if (arms := pairs[rid]).get("full", {}).get("score", 0.0) > 0]
    return (statistics.mean([row["score"] for _rid, row in full_rows]),
            statistics.mean([row["score"] for _rid, row in bounded_rows]),
            (statistics.mean(ratios) if ratios else None), len(ratios))


def arm_matrix(payloads):
    """Mean score per (artifact, task, arm) and the matched ratio against `full`.

    The LongBench runner can emit any of the published-family selection rules as
    arms, so this view keeps one column per arm instead of assuming the
    `full`/`bounded` pair.
    """
    rows = {}
    for name, payload in payloads:
        by_task = collections.defaultdict(lambda: collections.defaultdict(list))
        for row in payload.get("records", []):
            by_task[row.get("task")][row.get("arm")].append(row.get("score"))
        for task, arms in by_task.items():
            means = {arm: statistics.mean([s for s in scores if s is not None])
                     for arm, scores in arms.items() if scores}
            reference = means.get("full")
            rows.setdefault(os.path.basename(name), {})[task] = {
                "n": len(arms.get("full", [])) or max(len(v) for v in arms.values()),
                "means": means,
                "ratios": {arm: (value / reference if reference else None)
                           for arm, value in means.items()},
            }
    return rows


def format_arm_matrix(payloads):
    matrix = arm_matrix(payloads)
    arms = []
    for per_task in matrix.values():
        for entry in per_task.values():
            for arm in entry["means"]:
                if arm not in arms:
                    arms.append(arm)
    lines = ["| artifact | task | n | " + " | ".join(arms) + " |",
             "|---|---|---:|" + "---:|" * len(arms)]
    for name, per_task in matrix.items():
        for task, entry in sorted(per_task.items()):
            cells = [("-" if entry["means"].get(arm) is None
                      else f"{entry['means'][arm]:.4f}") for arm in arms]
            lines.append(f"| {name} | {task} | {entry['n']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("| artifact | task | " + " | ".join(f"{arm} ratio" for arm in arms) + " |")
    lines.append("|---|---|" + "---:|" * len(arms))
    for name, per_task in matrix.items():
        for task, entry in sorted(per_task.items()):
            cells = [("-" if entry["ratios"].get(arm) is None
                      else f"{entry['ratios'][arm]:.3f}") for arm in arms]
            lines.append(f"| {name} | {task} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_task_table(payloads):
    lines = ["| artifact | task | matched | full | bounded | ratio | losing | worst-3 share |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, payload in payloads:
        for task in sorted({row["task"] for row in payload.get("records", [])}):
            pairs = pair_records(payload, task)
            stats = matched_ratio(pairs)
            if stats is None:
                continue
            mean_full, mean_bounded, ratio, matched = stats
            concentration = loss_concentration(pairs)
            lines.append(
                f"| {os.path.basename(name)} | {task} | {matched} | "
                f"{mean_full:.4f} | {mean_bounded:.4f} | "
                f"{'n/a' if ratio is None else f'{ratio:.3f}'} | "
                f"{concentration['losing_records']} | "
                f"{concentration['worst_three_share']} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--task", default=None,
                        help="task to attribute in detail (default: the lowest ratio)")
    parser.add_argument("--view", default="attribution",
                        choices=["attribution", "arms"],
                        help="`attribution` reads the full/bounded pair; `arms` "
                             "prints one column per arm, for runs that emit the "
                             "published-family selection rules")
    parser.add_argument("--cache-dir", default=os.environ.get("QCC_LONGBENCH_DIR"),
                        help="LongBench cache with the references, for the "
                             "ROUGE-L precision/recall split")
    args = parser.parse_args(argv)

    payloads = [(path, load_artifact(path)) for path in args.paths]
    if args.view == "arms":
        print("**mean score per arm, and the matched ratio against `full`**\n")
        print(format_arm_matrix(payloads))
        return 0
    print("**per task**\n")
    print(format_task_table(payloads))

    task = args.task
    if task is None:
        ratios = []
        for name, payload in payloads:
            for row in payload.get("records", []):
                pass
            for candidate in sorted({row["task"] for row in payload.get("records", [])}):
                stats = matched_ratio(pair_records(payload, candidate))
                if stats and stats[2] is not None:
                    ratios.append((stats[2], candidate))
        task = min(ratios)[1] if ratios else None
    if task is None:
        return 0

    print(f"\n**`{task}` attribution**\n")
    for name, payload in payloads:
        pairs = pair_records(payload, task)
        if not pairs:
            continue
        concentration = loss_concentration(pairs)
        degenerate = degenerate_generation(pairs)
        print(f"* `{os.path.basename(name)}`: {concentration['records']} records, "
              f"{concentration['losing_records']} losing, pooled loss "
              f"{concentration['pooled_loss']}, worst three carry "
              f"{concentration['worst_three_share']} "
              f"({', '.join(str(item['row']) for item in concentration['worst'])}); "
              f"identical predictions {degenerate['identical_predictions']}, "
              f"r(repetition delta, score delta) = "
              f"{degenerate['pearson_repetition_vs_score']}, mean score delta "
              f"{degenerate['mean_score_delta']}")
        if args.cache_dir:
            split = rouge_split(payload, task, args.cache_dir)
            if split:
                print("  * ROUGE-L by arm: "
                      + "; ".join(f"{arm} f={values.get('f')} p={values.get('p')} "
                                  f"r={values.get('r')}" for arm, values in sorted(split.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
