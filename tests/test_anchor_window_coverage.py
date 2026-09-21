"""CPU tests for ``benchmarks/anchor_window_coverage.py``.

The script separates the two things that can hide an item from the policy: the
item name falling outside the observation window, and the anchor budget running
out.  Both are measured here on a tiny word-level tokenizer, so the test pins the
``named >= sites`` relation and the monotonicity in ``obs`` and ``lex_cap``
without a model.
"""

from __future__ import annotations

from benchmarks.anchor_window_coverage import (
    format_table,
    measure,
    named_in_window,
    window_start_char,
)
from benchmarks.benchmark_required_set import needles_record

from test_required_set import WordTokenizer


def test_window_start_moves_back_with_a_wider_window():
    tokenizer = WordTokenizer()
    record = needles_record(items=4, seed=0, target_tokens=600, tokenizer=tokenizer)
    prompt = record["prompt"]
    assert window_start_char(tokenizer, prompt, 10_000) == 0
    narrow = window_start_char(tokenizer, prompt, 32)
    wide = window_start_char(tokenizer, prompt, 256)
    assert 0 < wide < narrow < len(prompt)


def test_named_counts_the_item_names_the_window_can_see():
    tokenizer = WordTokenizer()
    record = needles_record(items=4, seed=0, target_tokens=600, tokenizer=tokenizer)
    prompt, keys = record["prompt"], record["keys"]
    assert named_in_window(tokenizer, prompt, keys, 10_000) == 4
    assert named_in_window(tokenizer, prompt, keys, 4) <= 4
    assert (named_in_window(tokenizer, prompt, keys, 64)
            >= named_in_window(tokenizer, prompt, keys, 16))


def test_measure_separates_the_window_ceiling_from_the_anchor_cap():
    tokenizer = WordTokenizer()
    narrow = measure(tokenizer, items=12, obs=16, lex_cap=4096, length=1200, seed=0)
    wide = measure(tokenizer, items=12, obs=64, lex_cap=4096, length=1200, seed=0)
    capped = measure(tokenizer, items=12, obs=64, lex_cap=512, length=1200, seed=0)
    tiny = measure(tokenizer, items=12, obs=64, lex_cap=49, length=1200, seed=0)

    assert wide["items_total"] == 12
    # a window shorter than the question's item list cannot see every name; a
    # site can still be reached when its own statement happens to sit inside the
    # window, so `sites` is not bounded by `named`
    assert narrow["named"] == 8 < wide["named"] == 12
    assert narrow["sites"] >= narrow["named"]
    # with the whole list visible, the cap decides how many sites are reached
    assert wide["sites"] == 12 and capped["sites"] == 8 and tiny["sites"] == 1
    assert tiny["anchor_tokens"] <= 49


def test_a_window_covering_the_whole_prompt_leaves_no_context_to_anchor():
    # `observation_window >= prompt length` makes the whole prompt the query, so
    # there is no earlier context to retrieve from: the names are visible and no
    # anchor exists.  The runner never does this (obs << prompt), but the
    # semantics have to be explicit.
    tokenizer = WordTokenizer()
    record = needles_record(items=4, seed=0, target_tokens=600, tokenizer=tokenizer)
    assert measure(tokenizer, items=4, obs=10_000, lex_cap=512, length=600,
                   seed=0)["sites"] == 0
    assert named_in_window(tokenizer, record["prompt"], record["keys"], 10_000) == 4


def test_format_table_lays_the_cells_out_by_obs_and_cap():
    rows = [
        {"items": 4, "obs": 64, "lex_cap": 512, "named": 2, "sites": 2,
         "anchor_tokens": 100},
        {"items": 4, "obs": 128, "lex_cap": 2048, "named": 4, "sites": 4,
         "anchor_tokens": 250},
    ]
    table = format_table(rows, "sites")
    assert "| k | obs=64, lex_cap=512 | obs=128, lex_cap=2048 |" in table
    assert "| 4 | 2 | 4 |" in table
