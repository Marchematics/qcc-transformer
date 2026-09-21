#!/usr/bin/env python
"""How much of the query the retention policy actually sees.

The selection has two query-driven channels, and both only look at the **last
``observation_window`` tokens** of the prompt:

* the observation-window scores (attention from those queries to the context), and
* the lexical anchors, whose candidate surfaces are read from that same slice.

So a question that names more items than fit in the window is only partly
visible to the policy, however large ``lex_cap`` is.  This script measures that
directly, without a model and without a GPU: it builds the same ``needles``
records ``benchmark_required_set.py`` builds, tokenizes them, and reports, per
``(k, obs, lex_cap)`` cell,

* ``named`` -- how many of the ``k`` item names appear inside the window,
* ``sites`` -- how many item sites the resulting anchor set reaches,
* ``anchor_tokens`` -- the size of the anchor set (capped by ``lex_cap``).

``named`` is the ceiling the window imposes and ``sites`` is what the cap
actually buys, so the two lines separate "the query was not visible" from "the
budget was too small".

Usage::

    python benchmarks/anchor_window_coverage.py --model /path/to/checkpoint \\
        --items 8 16 32 --obs 64 128 256 512 --lex-cap 512 2048
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.benchmark_required_set import (
    anchor_site_coverage,
    encode_ids,
    needles_record,
)
from qcc_transformer.retention import RetentionConfig, lexical_anchors


def window_start_char(tokenizer, prompt, obs):
    """Character offset where the policy's query slice begins."""
    offsets = tokenizer(prompt, add_special_tokens=False,
                        return_offsets_mapping=True).offset_mapping
    total = len(offsets)
    if obs >= total:
        return 0
    return offsets[total - obs][0]


def named_in_window(tokenizer, prompt, keys, obs):
    """How many item names appear inside the last ``obs`` tokens."""
    start = window_start_char(tokenizer, prompt, obs)
    return sum(1 for key in keys if prompt.find(key, start) >= 0)


def measure(tokenizer, items, obs, lex_cap, length, seed):
    """One ``(k, obs, lex_cap)`` cell."""
    record = needles_record(items=items, seed=seed, target_tokens=length,
                            tokenizer=tokenizer)
    prompt, keys = record["prompt"], record["keys"]
    config = RetentionConfig(observation_window=obs, lex_cap=lex_cap)
    anchors = lexical_anchors(tokenizer, prompt, config)
    sites, total = anchor_site_coverage(tokenizer, prompt, anchors, keys)
    return {
        "items": items, "obs": obs, "lex_cap": lex_cap, "seed": seed,
        "prompt_tokens": len(encode_ids(tokenizer, prompt)),
        "named": named_in_window(tokenizer, prompt, keys, obs),
        "sites": sites,
        "items_total": total,
        "anchor_tokens": len(anchors),
    }


def format_table(rows, field):
    """A markdown table of one field over ``items x (obs, lex_cap)``."""
    columns = sorted({(row["obs"], row["lex_cap"]) for row in rows})
    header = ["k"] + [f"obs={obs}, lex_cap={cap}" for obs, cap in columns]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for items in sorted({row["items"] for row in rows}):
        cells = []
        for column in columns:
            values = [row[field] for row in rows
                      if row["items"] == items
                      and (row["obs"], row["lex_cap"]) == column]
            cells.append("-" if not values
                         else f"{sum(values) / len(values):.2f}".rstrip("0").rstrip("."))
        lines.append("| " + " | ".join([str(items)] + cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True,
                        help="checkpoint or tokenizer path (tokenizer only, no weights)")
    parser.add_argument("--items", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--obs", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--lex-cap", nargs="+", type=int, default=[512, 2048])
    parser.add_argument("--length", type=int, default=32768)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model,
                                              trust_remote_code=args.trust_remote_code)
    rows = [measure(tokenizer, items, obs, lex_cap, args.length, seed)
            for items in args.items for obs in args.obs
            for lex_cap in args.lex_cap for seed in range(args.seeds)]
    print("**item names inside the query window (`named`)**\n")
    print(format_table(rows, "named"))
    print("\n**item sites the anchor set reaches (`sites`)**\n")
    print(format_table(rows, "sites"))
    print("\n**anchor set size in tokens (`anchor_tokens`)**\n")
    print(format_table(rows, "anchor_tokens"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
