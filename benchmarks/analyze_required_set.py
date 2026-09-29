#!/usr/bin/env python
"""Read ``benchmark_required_set.py`` artifacts and print the decision tables.

The runner writes one JSON per sweep with every row it measured.  This reader
turns a set of those files into the tables the mechanism sections quote:

* **strict** -- ``correct`` (all-or-nothing: every item of the required set),
* **recall** -- ``partial_recall`` (RULER ``string_match_all`` per item, averaged),
* **trunc** -- share of rows whose decode stopped only because it hit ``max_new``,
* **sites** -- mean number of item sites the lexical-anchor set reached
  (``anchor_sites``; see ``anchor_site_coverage`` in the runner).

Reading the strict and recall views side by side is what separates "the cache
lost the item" from "the model produced a partial answer", and ``trunc`` is the
readout-side control for a collapse that is not a retention failure.  Tables are
grouped by ``model x placement x lex_cap x length`` so a sweep over one knob can
be read down the page.

Usage::

    python benchmarks/analyze_required_set.py artifacts/prediction-*.json
    python benchmarks/analyze_required_set.py artifacts/prediction-needles.json \\
        --markdown
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


def load_artifact(path):
    """One artifact file as ``(config, rows)``."""
    with open(path) as handle:
        payload = json.load(handle)
    return payload.get("config", {}), payload.get("records", payload.get("rows", []))


def artifact_signature(config, rows, path):
    """The label a table groups by: model, placement, lex_cap, length."""
    model = config.get("label") or config.get("model") or os.path.basename(path)
    model = str(model).rstrip("/").split("/")[-1]
    placements = sorted({row.get("placement", "scatter") for row in rows})
    lex_caps = sorted({row["lex_cap"] for row in rows
                       if row.get("lex_cap") is not None})
    if not lex_caps and config.get("lex_cap") is not None:
        lex_caps = [int(config["lex_cap"])]      # artifacts written before the
    lengths = sorted({row["length"] for row in rows})   # per-row field existed
    return {
        "artifact": os.path.basename(path),
        "model": model,
        "family": ",".join(config.get("family", []) or sorted(
            {row["family"] for row in rows})),
        "placement": ",".join(placements),
        "lex_cap": ",".join(str(value) for value in lex_caps) or "-",
        "length": ",".join(str(value) for value in lengths),
    }


def collect(paths):
    """All rows of ``paths`` with the artifact label attached."""
    rows = []
    for path in paths:
        config, records = load_artifact(path)
        signature = artifact_signature(config, records, path)
        for record in records:
            rows.append({**record, "_signature": signature})
    return rows


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def cell(rows, field):
    """Mean of ``field`` over ``rows``: bools as a rate, numbers as a mean.

    ``field="trunc"`` is not a row field: it is ``generated_tokens >= max_new``,
    computed here so the analyzer can report the readout control from a file
    whose rows only carry the two numbers.
    """
    if field == "trunc":
        values = [row["generated_tokens"] >= row["max_new"] for row in rows
                  if row.get("generated_tokens") is not None
                  and row.get("max_new") is not None]
        return _mean(values)
    if field == "correct":
        return _mean([bool(row.get("correct")) for row in rows
                      if row.get("correct") is not None])
    return _mean([row.get(field) for row in rows])


def arms_of(rows):
    """The arm labels of a row set, Full-KV first then budgets ascending."""
    budgets = sorted({int(row["budget"]) for row in rows
                      if row.get("budget") is not None})
    return ["full"] + [f"b{budget}" for budget in budgets]


def pivot(rows, field="correct", axis="items"):
    """``{(signature, axis value): {arm: mean}}`` plus the arm order.

    ``field`` is any row field, plus the derived ``"trunc"`` (see :func:`cell`).
    """
    arms = arms_of(rows)
    grouped = defaultdict(lambda: defaultdict(list))
    signatures, seen = [], set()
    for row in rows:
        signature = row["_signature"]
        sig_key = tuple(sorted(signature.items()))
        if sig_key not in seen:
            seen.add(sig_key)
            signatures.append(signature)
        arm = "full" if row["arm"] == "full" else f"b{int(row['budget'])}"
        grouped[(sig_key, row[axis])][arm].append(row)
    out = {key: {arm: cell(arm_rows, field) for arm, arm_rows in by_arm.items()}
           for key, by_arm in grouped.items()}
    return out, arms, signatures


def format_table(rows, field="correct", axis="items", digits=3):
    """A markdown table of one field over (artifact, axis) x arm."""
    table, arms, signatures = pivot(rows, field, axis)
    if not rows:
        return "_no rows_"
    header = ["model", "placement", "lex_cap", "L", axis] + arms
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for signature in signatures:
        axis_values = sorted({key[1] for key in table
                              if key[0] == tuple(sorted(signature.items()))},
                             key=lambda value: (value is None, value))
        for value in axis_values:
            key = (tuple(sorted(signature.items())), value)
            cells = []
            for arm in arms:
                score = table.get(key, {}).get(arm)
                cells.append("-" if score is None else f"{score:.{digits}f}")
            lines.append("| " + " | ".join([
                signature["model"], signature["placement"], signature["lex_cap"],
                signature["length"], str(value)] + cells) + " |")
    return "\n".join(lines)


def report(rows, axis="items"):
    """The tables, in the order the mechanism sections quote them."""
    blocks = [
        ("strict `correct` (all-or-nothing)", format_table(rows, "correct", axis)),
        ("`partial_recall` (RULER string match per item)", format_table(rows, "partial_recall", axis)),
        ("item statements retained (`required_coverage`)", format_table(rows, "required_coverage", axis)),
        ("attention share on the statements (`required_mass`)", format_table(rows, "required_mass", axis)),
        ("truncated decode share (`generated_tokens >= max_new`)", format_table(rows, "trunc", axis)),
        ("item sites reached by the lexical anchors", format_table(rows, "anchor_sites", axis)),
    ]
    return "\n\n".join(f"**{title}**\n\n{table}" for title, table in blocks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--axis", default="items", choices=["items", "distractors"],
                        help="`items` for the k axis, `distractors` for the "
                             "competition axis of the multikey family")
    parser.add_argument("--field", default=None,
                        choices=["correct", "partial_recall", "trunc",
                                 "required_coverage", "required_mass", "items_covered",
                                 "anchor_sites", "generated_tokens", "kept_slots"],
                        help="print one table instead of the four-table report")
    parser.add_argument("--markdown", action="store_true",
                        help="accepted for symmetry; output is already markdown")
    args = parser.parse_args(argv)

    rows = collect(args.paths)
    if not rows:
        raise SystemExit("no records found in the given artifacts")
    if args.field:
        print(format_table(rows, args.field, args.axis))
    else:
        print(report(rows, args.axis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
