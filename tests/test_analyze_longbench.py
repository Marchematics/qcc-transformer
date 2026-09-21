"""CPU tests for ``benchmarks/analyze_longbench.py``.

The reader decides what the report says about the worst LongBench task, so the
attribution arithmetic is pinned on hand-built rows: the ratio must follow the
runner's matched-record convention (a zero Full-KV score cannot be a denominator),
the loss concentration must count only losing records, and the repetition
statistic must separate a looping summary from a varied one.
"""

from __future__ import annotations

from benchmarks.analyze_longbench import (
    degenerate_generation,
    format_task_table,
    load_artifact,
    loss_concentration,
    matched_ratio,
    pair_records,
    repetition,
)


def row(task, rid, arm, score, prediction, prompt_tokens=1000):
    return {"task": task, "id": rid, "arm": arm, "score": score,
            "prediction": prediction, "prompt_tokens": prompt_tokens, "row": hash(rid) % 100}


def payload(rows):
    return {"records": rows}


def test_repetition_separates_a_loop_from_varied_text():
    looped = " ".join(["the report notes that"] * 20)
    varied = " ".join(f"token{i}-{i * 7 % 97}" for i in range(60))
    words, distinct, share = repetition(looped)
    assert words == 80 and distinct < 0.2 and share > 0.5
    words, distinct, share = repetition(varied)
    assert distinct == 1.0 and share == 0.0     # every word is distinct
    assert repetition("") == (0, 0.0, 0.0)


def test_matched_ratio_skips_records_the_full_arm_cannot_answer():
    rows = [row("t", "a", "full", 0.4, "x"), row("t", "a", "bounded", 0.2, "x"),
            row("t", "b", "full", 0.0, "x"), row("t", "b", "bounded", 0.9, "x"),
            row("t", "c", "full", 0.6, "x"), row("t", "c", "bounded", 0.6, "x")]
    pairs = pair_records(payload(rows), "t")
    mean_full, mean_bounded, ratio, matched = matched_ratio(pairs)
    assert (mean_full, mean_bounded, matched) == (1.0 / 3, 1.7 / 3, 2)
    assert ratio == 0.75                       # (0.5 + 1.0) / 2, the zero dropped
    assert matched_ratio({}) is None


def test_loss_concentration_and_degenerate_generation():
    rows = [
        row("t", "a", "full", 0.30, "alpha beta gamma delta epsilon zeta eta theta"),
        row("t", "a", "bounded", 0.13, " ".join(["loop"] * 16)),
        row("t", "b", "full", 0.20, "one two three four five six seven eight"),
        row("t", "b", "bounded", 0.20, "one two three four five six seven eight"),
    ]
    pairs = pair_records(payload(rows), "t")
    concentration = loss_concentration(pairs)
    assert concentration["records"] == 2 and concentration["losing_records"] == 1
    assert concentration["pooled_loss"] == 0.17
    assert concentration["worst_three_share"] == 1.0
    degenerate = degenerate_generation(pairs)
    assert degenerate["identical_predictions"] == 1
    assert degenerate["more_repetitive_records"] == 1
    assert degenerate["mean_score_delta"] == -0.085
    assert degenerate["pearson_repetition_vs_score"] == -1.0


def test_format_task_table_reports_the_ratio_and_the_losing_count():
    rows = [row("t", "a", "full", 0.4, "x"), row("t", "a", "bounded", 0.2, "x")]
    table = format_task_table([("artifact.json", payload(rows))])
    assert "| losing |" in table
    assert "| artifact.json | t | 1 | 0.4000 | 0.2000 | 0.500 | 1 | 1.0 |" in table


def test_load_artifact_reads_records(tmp_path):
    path = tmp_path / "a.json"
    path.write_text('{"records": []}')
    assert load_artifact(path)["records"] == []
