"""Turn the stored benchmark JSONs into the tables the report quotes.

Two record formats are supported, detected by their keys:

* ``multimodel`` - ``benchmarks/benchmark_retention_multimodel.py`` output:
  per-record Full-KV and bounded arms, per-task retention.
* ``baselines`` - ``benchmarks/benchmark_bounded_decode_ruler.py`` output:
  several selection policies at one budget over the same records.

Decode-state bytes are recomputed from the retained slot count and the model's
own config, so the same number is comparable across models and policies rather
than being copied from a log line.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from transformers import AutoConfig

DEFAULT_MODEL = os.environ.get("QCC_MODEL", "meta-llama/Llama-3.2-1B-Instruct")


def state_bytes(model_path, slots, dtype_bytes=2):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    layers = config.num_hidden_layers
    kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    return 2 * layers * kv_heads * slots * head_dim * dtype_bytes


def mib(value):
    return round(value / 2**20, 1)


def official(prediction, outputs):
    """RULER's string_match_all: mean fraction of references present."""
    refs = [out for out in outputs if out]
    if not refs:
        return 0.0
    low = prediction.lower()
    return sum(1.0 for ref in refs if ref.lower() in low) / len(refs)


def strict(prediction, outputs):
    refs = [out for out in outputs if out]
    return 1.0 if refs and all(ref.lower() in prediction.lower() for ref in refs) else 0.0


def multimodel(paths):
    print("| model | task | records | Full-KV | bounded | retention mean | worst | slots | decode state |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    aggregates = []
    for path in paths:
        data = json.load(open(path))
        model = data["label"]
        model_path = data["config"]["model"]
        slots = None
        for task, entry in sorted(data["summary"]["tasks"].items()):
            bounded = entry["bounded"]
            slots = bounded["mean_kept_slots"]
            retention = entry.get("retention", {})
            print(f"| {model} | {task} | {bounded['records']} | {entry['full']['mean_partial']:.3f} "
                  f"| {bounded['mean_partial']:.3f} | {retention.get('mean')} "
                  f"| {retention.get('worst')} | {slots:.0f} "
                  f"| {mib(state_bytes(data['config']['model'], slots))} MiB |")
        aggregate = data["summary"]["aggregate"]
        aggregates.append((model, aggregate))
        print(f"| **{model}** | **all** | {aggregate['matched_records']} | "
              f"| | **{aggregate['mean_retention']}** | **{aggregate['worst_task_retention']}** "
              f"| {slots:.0f} | {mib(state_bytes(model_path, slots))} MiB |")
    if aggregates:
        mean = sum(a["mean_retention"] for _m, a in aggregates) / len(aggregates)
        worst = min(a["worst_task_retention"] for _m, a in aggregates)
        print(f"\nacross {len(aggregates)} models: mean retention {mean:.4f}, "
              f"worst task retention {worst:.4f}")


def baselines(path):
    data = json.load(open(path))
    rows = data["results"]
    model_path = (data.get("config") or {}).get("model") or data.get("model")
    if model_path is None:                      # partial writes carry rows only
        model_path = json.load(open(rows[0]["source_json"]))["config"]["model"] \
            if "source_json" in rows[0] else DEFAULT_MODEL
    policies = sorted({row["policy"] for row in rows})
    tasks = sorted({row["task"] for row in rows})
    full = {(row["task"], tuple(row["outputs"])): row for row in rows if row["policy"] == "full"}
    print("| policy | " + " | ".join(tasks) + " | matched | retention | worst | slots | state |")
    print("|---|" + "---:|" * (len(tasks) + 6))
    for policy in policies:
        if policy == "full":
            continue
        selected = [row for row in rows if row["policy"] == policy]
        per_task, ratios = {}, []
        for task in tasks:
            entries = [row for row in selected if row["task"] == task]
            per_task[task] = round(sum(official(row["prediction"], row["outputs"])
                                       for row in entries) / len(entries), 3)
            for row in entries:
                reference = full.get((row["task"], tuple(row["outputs"])))
                score = official(row["prediction"], row["outputs"])
                base = official(reference["prediction"], reference["outputs"]) if reference else 0.0
                if base > 0:
                    ratios.append(score / base)
        slots = sum(row["kept_slots"] for row in selected) / len(selected)
        print(f"| {policy} | " + " | ".join(f"{per_task[task]:.3f}" for task in tasks) +
              f" | {len(ratios)} | {sum(ratios) / len(ratios):.4f} | {min(ratios):.3f} "
              f"| {slots:.0f} | {mib(state_bytes(model_path, slots))} MiB |")
    reference = [row for row in rows if row["policy"] == "full"]
    per_task = {task: round(sum(official(r["prediction"], r["outputs"]) for r in reference
                                if r["task"] == task) /
                            max(1, len([r for r in reference if r["task"] == task])), 3)
                for task in tasks}
    print(f"| full (reference) | " + " | ".join(f"{per_task[task]:.3f}" for task in tasks) + " |")

    # which selection policy wins at each budget, and what the lexical anchors add
    print("\npaired comparison against the shipped policy (lex_obs), per task:")
    shipped = {(row["task"], tuple(row["outputs"])): row for row in rows if row["policy"] == "lex_obs"}
    for policy in policies:
        if policy in ("full", "lex_obs"):
            continue
        wins = losses = ties = 0
        for row in rows:
            if row["policy"] != policy:
                continue
            other = shipped.get((row["task"], tuple(row["outputs"])))
            if not other:
                continue
            score = official(row["prediction"], row["outputs"])
            other_score = official(other["prediction"], other["outputs"])
            if score > other_score:
                wins += 1
            elif score < other_score:
                losses += 1
            else:
                ties += 1
        print(f"  {policy:<12} better on {wins:>2}, worse on {losses:>2}, tied on {ties:>2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["multimodel", "baselines"])
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    if args.kind == "multimodel":
        multimodel(args.paths)
    else:
        for path in args.paths:
            print(f"## {path}")
            baselines(path)


if __name__ == "__main__":
    main()
