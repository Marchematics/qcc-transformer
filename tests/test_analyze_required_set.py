"""CPU tests for ``benchmarks/analyze_required_set.py``.

The reader decides what the mechanism tables in ``docs/MECHANISM.md`` say, so
it is tested on a hand-built artifact: the strict/partial/truncation views must
separate ("the model answered half the items" vs "the model was cut off"), the
axis must pivot per artifact, and an artifact written before the per-row
``lex_cap`` field existed must still group under its configured cap.
"""

from __future__ import annotations

import json

from benchmarks.analyze_required_set import (
    artifact_signature,
    collect,
    format_table,
    load_artifact,
    pivot,
    report,
)


def row(items, arm, correct, recall, generated, max_new, budget=None,
        anchor_sites=None):
    return {
        "family": "needles", "items": items, "seed": 0, "length": 32768,
        "arm": arm, "budget": budget, "correct": correct,
        "partial_recall": recall, "generated_tokens": generated,
        "max_new": max_new, "anchor_sites": anchor_sites,
        "placement": "scatter", "lex_cap": 512,
    }


def write_artifact(tmp_path, name, records, config=None):
    path = tmp_path / name
    path.write_text(json.dumps({
        "config": config or {"label": "/models/Qwen2.5-3B-Instruct",
                             "family": ["needles"], "lex_cap": 512},
        "records": records,
    }))
    return str(path)


def test_views_separate_partial_answers_from_truncated_decodes(tmp_path):
    records = [
        row(4, "full", True, 1.0, 32, 32),
        row(4, "b2048", False, 0.5, 24, 32, budget=2048, anchor_sites=3),
        row(4, "b2048", False, 0.0, 32, 32, budget=2048, anchor_sites=4),
    ]
    rows = collect([write_artifact(tmp_path, "a.json", records)])
    table, arms, signatures = pivot(rows, "correct")
    key = (tuple(sorted(signatures[0].items())), 4)
    assert arms == ["full", "b2048"]
    assert table[key]["full"] == 1.0
    assert table[key]["b2048"] == 0.0                      # strict: both wrong

    table, _arms, _signatures = pivot(rows, "partial_recall")
    assert table[key]["b2048"] == 0.25                     # half the items once
    table, _arms, _signatures = pivot(rows, "trunc")
    assert table[key]["b2048"] == 0.5                      # one of two cut off
    table, _arms, _signatures = pivot(rows, "anchor_sites")
    assert table[key]["b2048"] == 3.5
    assert signatures[0]["model"] == "Qwen2.5-3B-Instruct"


def test_axis_pivots_per_artifact_and_missing_lex_cap_falls_back_to_config(tmp_path):
    first = [row(8, "full", True, 1.0, 64, 64),
             row(8, "b4096", True, 1.0, 60, 64, budget=4096)]
    second = [row(16, "b4096", False, 0.25, 64, 128, budget=4096)]
    for record in first + second:                          # pre-field artifact
        del record["lex_cap"]
    rows = collect([write_artifact(tmp_path, "a.json", first),
                    write_artifact(tmp_path, "b.json", second,
                                   config={"label": "meta-llama/Llama-3.1-8B",
                                           "family": ["needles"], "lex_cap": 2048})])
    assert [entry["lex_cap"] for entry in
            (artifact_signature({"lex_cap": 512}, first, "a.json"),
             artifact_signature({"lex_cap": 2048}, second, "b.json"))] == ["512", "2048"]
    text = report(rows)
    assert "| items | full | b4096 |" in text
    assert "| Qwen2.5-3B-Instruct | scatter | 512 | 32768 | 8 | 1.000 | 1.000 |" in text
    assert "| Llama-3.1-8B | scatter | 2048 | 32768 | 16 | - | 0.000 |" in text
    assert text.count("| items |") == 6                    # six views
    assert format_table(rows, "kept_slots").count("|") > 0  # unknown field: dashes


def test_load_artifact_reads_rows_key_as_well(tmp_path):
    path = tmp_path / "older.json"
    path.write_text(json.dumps({"config": {}, "rows": [
        row(2, "b512", True, 1.0, 10, 32, budget=512)]}))
    config, records = load_artifact(str(path))
    assert config == {} and len(records) == 1
