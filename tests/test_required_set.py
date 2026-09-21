"""CPU tests for ``benchmarks/benchmark_required_set.py``.

They cover the parts of the runner that decide what the numbers mean: prompt
construction for every family (with hand-computed aggregate answers, a multikey
distractor and the multi-needle question), the scoring (numeric, distractor-hit
and RULER ``string_match_all`` partial recall), the ``required_budget``
derivation from a synthetic summary matrix (including the null case and the
``full_kv_unsolved`` reporting), the ``budget_covers_required_set`` conditional
block, the plan the ``--distractors`` axis generates, and the shared-prefill
fast path being bit-identical to the packaged ``compile_bounded_cache``.

Everything runs on a tiny randomly-initialised Llama, as ``test_retention.py``
does, so no download and no GPU is needed.
"""

from __future__ import annotations

import re

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from benchmarks.benchmark_required_set import (
    aggregate_record,
    build_plan,
    build_record,
    char_f1,
    clone_cache,
    decode_budget,
    encode_ids,
    exact_set_match,
    multikey_record,
    needles_record,
    parse_args,
    parse_numbers,
    reference_recall,
    required_budget_from_matrix,
    score_needles,
    score_prediction,
    score_record,
    summarize,
)
from qcc_transformer.retention import (
    RetentionConfig,
    compile_bounded_cache,
    lexical_anchors,
    observation_scores,
    prefill_capture,
    prune_cache,
    select_indices,
)


class WordTokenizer:
    """Whitespace tokenizer with HF-style offsets, ids assigned by first use."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _id(self, word):
        return self.vocab.setdefault(word, len(self.vocab))

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        tokens = text.split()
        offsets, cursor = [], 0
        for token in tokens:
            start = text.index(token, cursor)
            offsets.append((start, start + len(token)))
            cursor = start + len(token)

        class Encoded:
            pass

        encoded = Encoded()
        encoded.offset_mapping = offsets
        encoded.input_ids = [self._id(token) for token in tokens]
        return encoded

    def encode(self, text, add_special_tokens=False):
        return self(text).input_ids

    def decode(self, ids, skip_special_tokens=False):
        inverse = {index: word for word, index in self.vocab.items()}
        return " ".join(inverse.get(int(i), "?") for i in ids)


def tiny_model(layers=2, hidden=64, heads=4, kv_heads=2, vocab=512):
    config = LlamaConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=kv_heads, max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(config).eval()


# --------------------------------------------------------------------------- #
# prompt construction and scoring
# --------------------------------------------------------------------------- #

def test_multikey_prompt_has_k_keys_and_asks_for_one_value():
    tokenizer = WordTokenizer()
    record = multikey_record(items=4, seed=1, target_tokens=400, tokenizer=tokenizer)
    pairs = re.findall(r"magic numbers for (trace-[0-9a-f]{6}) is (\d+)",
                       record["prompt"])
    assert len(pairs) == 4                                   # one statement per key
    keys = {key for key, _value in pairs}
    assert len(keys) == 4                                    # all surfaces distinct
    assert record["target_key"] in keys
    assert record["prompt"].count(record["target_key"]) == 2  # statement + question
    values = dict(pairs)
    assert record["expected"] == values[record["target_key"]]
    assert record["distractors"] == [v for k, v in pairs if k != record["target_key"]]
    # the values in the context are the only integers in the prompt
    assert [int(m) for m in re.findall(r"\b\d{5,}\b", record["prompt"])] == \
        [int(v) for _k, v in pairs]
    assert abs(record["prompt_tokens"] - 400) <= 2


def test_multikey_scoring_accepts_the_answer_and_flags_a_distractor():
    tokenizer = WordTokenizer()
    record = multikey_record(items=4, seed=1, target_tokens=300, tokenizer=tokenizer)
    expected = record["expected"]
    right = score_prediction(f"The magic number is {expected}.", expected,
                             record["distractors"])
    assert right["correct"] and right["parsed"] == expected
    assert right["partial"] == 1.0 and not right["distractor_hit"]

    distractor = record["distractors"][0]
    wrong = score_prediction(f"The magic number is {distractor}.", expected,
                             record["distractors"])
    assert not wrong["correct"] and wrong["distractor_hit"]
    assert wrong["parsed"] == distractor
    assert 0.0 < wrong["partial"] < 1.0

    empty = score_prediction("", expected, record["distractors"])
    assert empty == {"parsed": None, "correct": False, "partial": 0.0,
                     "distractor_hit": False, "candidates": []}


def test_aggregate_sum_matches_a_hand_computed_total():
    tokenizer = WordTokenizer()
    record = aggregate_record(items=5, seed=3, target_tokens=400,
                              tokenizer=tokenizer, aggregate="sum")
    hand = [int(m) for m in re.findall(r"holds (\d+) units", record["prompt"])]
    assert len(hand) == 5
    assert all(1 <= value <= 9 for value in hand)             # sum is unambiguous
    assert record["expected"] == str(sum(hand)) == str(sum(int(v) for v in record["values"]))
    assert sorted(int(v) for v in record["values"]) == sorted(hand)
    assert score_prediction(f"The total is {sum(hand)}.", record["expected"])["correct"]


def test_aggregate_count_matches_a_hand_computed_count():
    tokenizer = WordTokenizer()
    record = aggregate_record(items=6, seed=2, target_tokens=400,
                              tokenizer=tokenizer, aggregate="count")
    status = record["target_status"]
    stated = re.findall(r"item has status (\w+)", record["prompt"])
    assert len(stated) == 6
    assert stated.count(status) == int(record["expected"])
    assert 0 < int(record["expected"]) < 6                    # the criterion matters
    assert len(set(stated)) > 1                               # distractors present
    assert not re.search(r"\d", record["prompt"])             # no numbers in a count task
    assert status in record["prompt"].rsplit("listed above", 1)[-1]  # asked in the question


def test_records_are_reproducible_and_seed_dependent():
    tokenizer = WordTokenizer()
    first = build_record("aggregate", 4, 7, 300, tokenizer, "sum")
    same = build_record("aggregate", 4, 7, 300, tokenizer, "sum")
    other = build_record("aggregate", 4, 8, 300, tokenizer, "sum")
    assert first["prompt"] == same["prompt"]
    assert first["expected"] == same["expected"]
    assert first["prompt"] != other["prompt"]


def test_parse_numbers_normalizes_surfaces():
    assert parse_numbers("the total is 1,234 and also -7 units") == ["1234", "-7"]
    assert parse_numbers("no digits here") == []
    assert char_f1("7549132", "7549132") == 1.0
    assert 0.0 < char_f1("7541", "7549132") < 1.0     # partially right digits
    assert char_f1("0000", "7549132") == 0.0          # no shared digits at all


# --------------------------------------------------------------------------- #
# the distractor axis (multikey competition)
# --------------------------------------------------------------------------- #

def _statement_pairs(prompt):
    return re.findall(r"magic numbers for (trace-[0-9a-f]{6}) is (\d+)", prompt)


def test_distractors_change_the_multikey_prompt():
    tokenizer = WordTokenizer()
    small = build_record("multikey", 5, 1, 400, tokenizer, distractors=4)
    big = build_record("multikey", 33, 1, 400, tokenizer, distractors=32)
    small_pairs, big_pairs = _statement_pairs(small["prompt"]), _statement_pairs(big["prompt"])

    assert len(small_pairs) == 5 and len(big_pairs) == 33      # one statement per key
    assert len({key for key, _value in big_pairs}) == 33       # surfaces all distinct
    assert len({value for _key, value in big_pairs}) == 33     # values all distinct
    assert small["distractor_count"] == 4 and big["distractor_count"] == 32
    assert len(small["distractors"]) == 4 and len(big["distractors"]) == 32
    assert big["key_count"] == 33 and big["items"] == 33

    # the question still asks for exactly one key's value
    assert big["prompt"].rstrip().endswith("Answer with a single number.")
    assert big["prompt"].count(big["target_key"]) == 2         # statement + question
    assert all(big["prompt"].count(key) == 1 for key in big["keys"]
               if key != big["target_key"])
    assert dict(big_pairs)[big["target_key"]] == big["expected"]
    assert big["expected"] not in big["distractors"]

    # more competing keys -> more statement tokens at the same token target, and
    # the two records are genuinely different tasks
    assert big["statement_tokens"] > small["statement_tokens"]
    assert abs(big["prompt_tokens"] - 400) <= 2 and abs(small["prompt_tokens"] - 400) <= 2
    assert big["prompt"] != small["prompt"]
    assert big["target_key"] != small["target_key"] or big["expected"] != small["expected"]

    # the legacy shape (no --distractors) is the same task with k = items keys
    legacy = build_record("multikey", 5, 1, 400, tokenizer)
    assert len(_statement_pairs(legacy["prompt"])) == 5
    assert legacy["distractor_count"] == 4 and legacy["key_count"] == 5


def test_build_plan_sweeps_distractors_and_keeps_legacy_items():
    args = parse_args(["--family", "multikey", "needles", "--items", "1", "2",
                       "--distractors", "4", "32", "--seeds", "2", "--out", "/tmp/x.json"])
    plan = build_plan(args, [512, 1024])
    multikey = [entry for entry in plan if entry["family"] == "multikey"]
    needles = [entry for entry in plan if entry["family"] == "needles"]
    # one entry per (distractor count, seed, length); the cell label is N + 1
    assert sorted({(e["items"], e["distractors"]) for e in multikey}) == [(5, 4), (33, 32)]
    assert len(multikey) == 2 * 2 * 2
    assert all(e["items"] == e["distractors"] + 1 for e in multikey)
    # other families still sweep --items
    assert sorted({(e["items"], e["distractors"]) for e in needles}) == [(1, None), (2, None)]
    assert len(needles) == 2 * 2 * 2

    legacy = build_plan(parse_args(["--family", "multikey", "--items", "1", "2",
                                    "--out", "/tmp/x.json"]), [512])
    assert sorted({(e["items"], e["distractors"]) for e in legacy}) == [(1, None), (2, None)]


def test_cli_accepts_needles_and_distractors_and_help_exits_cleanly(capsys):
    args = parse_args(["--family", "needles", "--distractors", "8",
                       "--max-new", "48", "--out", "/tmp/x.json"])
    assert args.family == ["needles"] and args.distractors == [8] and args.max_new == 48
    default = parse_args(["--out", "/tmp/x.json"])
    assert default.distractors is None and default.max_new is None
    # the existing default sweep is unchanged; needles is opt-in via --family
    assert default.family == ["multikey", "aggregate"]
    with pytest.raises(SystemExit) as exit_info:
        parse_args(["--help"])
    assert exit_info.value.code == 0
    assert "--distractors" in capsys.readouterr().out


def test_decode_budget_covers_the_whole_needles_answer():
    # the default must grow with k so exact-set scoring is not limited by decode
    assert decode_budget("needles", 1) == 32
    assert decode_budget("needles", 32) == 256
    assert decode_budget("multikey", 16) == 24
    assert decode_budget("aggregate", 16) == 24
    assert decode_budget("needles", 32, 10) == 10      # explicit --max-new wins


# --------------------------------------------------------------------------- #
# the needles family (capacity probe without arithmetic)
# --------------------------------------------------------------------------- #

def test_needles_prompt_places_k_needles_and_asks_for_all_of_them():
    tokenizer = WordTokenizer()
    record = needles_record(items=4, seed=0, target_tokens=600, tokenizer=tokenizer)
    pairs = _statement_pairs(record["prompt"])
    assert len(pairs) == 4
    keys = [key for key, _value in pairs]
    assert len(set(keys)) == 4 and record["references"] == [value for _key, value in pairs]
    assert record["expected"] == ", ".join(record["references"])
    assert len(set(record["references"])) == 4

    # every reference value appears in the prompt exactly once, and nowhere else
    for value in record["references"]:
        assert record["prompt"].count(value) == 1
    # the question names all k keys (each key therefore appears twice)
    question = record["prompt"].rsplit("above:", 1)[-1]
    assert question.startswith(" ") and "Answer with all 4 numbers" in question
    for key in keys:
        assert key in question
        assert record["prompt"].count(key) == 2
    assert sorted(record["ask_order"]) == sorted(keys)
    assert len(record["ask_order"]) == 4
    assert abs(record["prompt_tokens"] - 600) <= 2
    assert record["distractor_count"] is None
    assert record["key_count"] == 4


def test_needles_partial_recall_is_rulers_string_match_all():
    # hand-computed: two of three references present, matched case-insensitively
    references = ["749132", "A1B2C3", "355883"]
    assert reference_recall(references, "the codes are 749132 and a1b2c3") == round(2 / 3, 4)
    assert reference_recall(references, "749132 A1B2C3 355883") == 1.0
    assert reference_recall(references, "") == 0.0
    assert reference_recall([], "anything") == 0.0
    assert exact_set_match(references, "749132 a1b2c3 355883") is True
    assert exact_set_match(references, "749132 a1b2c3") is False
    assert exact_set_match([], "749132") is False


def test_needles_scoring_separates_partial_recall_from_exact_match():
    tokenizer = WordTokenizer()
    record = needles_record(items=4, seed=0, target_tokens=400, tokenizer=tokenizer)
    references = record["references"]

    complete = score_record(record, "The numbers are " + ", ".join(references) + ".")
    assert complete["correct"] and complete["exact_set_match"]
    assert complete["partial_recall"] == 1.0 and complete["partial"] == 1.0
    assert complete["missing"] == [] and complete["present"] == references

    # three of four: graded credit, not an exact match
    partial = score_record(record, "Only " + references[0] + " and " + references[1]
                           + " and " + references[2] + ".")
    assert partial["correct"] is False and partial["exact_set_match"] is False
    assert partial["partial_recall"] == 0.75 and partial["partial"] == 0.75
    assert partial["present"] == references[:3]
    assert partial["missing"] == references[3:]

    # a repeated reference still counts once: recall is over the references
    repeated = score_needles(" ".join([references[0]] * 4), references)
    assert repeated["partial_recall"] == 0.25 and repeated["correct"] is False

    empty = score_record(record, "")
    assert empty["partial_recall"] == 0.0 and empty["correct"] is False

    # the single-answer families expose the same fields with one reference
    single_record = multikey_record(2, 0, 200, tokenizer)
    single = score_record(single_record, "no digits here")
    assert single["partial_recall"] == 0.0 and single["exact_set_match"] is False
    assert single["references"] == [single_record["expected"]]
    assert single["missing"] == [single_record["expected"]]


# --------------------------------------------------------------------------- #
# required_budget derivation
# --------------------------------------------------------------------------- #

def test_required_budget_is_the_smallest_matching_budget():
    assert required_budget_from_matrix({512: 0.0, 1024: 0.5, 2048: 1.0}, 1.0) == 2048
    assert required_budget_from_matrix({512: 0.5, 1024: 1.0, 2048: 1.0}, 1.0) == 1024
    assert required_budget_from_matrix({512: 1.0}, 1.0) == 512


def test_required_budget_is_null_when_nothing_matches():
    # no tested budget reaches Full-KV
    assert required_budget_from_matrix({512: 0.4, 1024: 0.6}, 1.0) is None
    # nothing to match: a Full-KV arm that answers nothing never yields a budget
    assert required_budget_from_matrix({512: 0.0, 1024: 0.0}, 0.0) is None
    # a budget that answers nothing is not a match even with full tolerance
    assert required_budget_from_matrix({512: 0.0, 1024: 0.5}, 1.0, tolerance=1.0) == 1024


def test_required_budget_honours_tolerance():
    # 512 is 0.1 below Full-KV: not a match without slack, a match with it
    assert required_budget_from_matrix({512: 0.9, 1024: 1.0}, 1.0) == 1024
    assert required_budget_from_matrix({512: 0.9, 1024: 1.0}, 1.0, tolerance=0.1) == 512
    assert required_budget_from_matrix({512: 0.8}, 1.0, tolerance=0.1) is None


def _row(family, items, seed, length, arm, correct, budget=None, statement_tokens=20.0,
         distractors=None, partial_recall=None):
    return {
        "family": family, "items": items, "seed": seed, "length": length,
        "arm": arm, "budget": budget, "correct": bool(correct),
        "statement_tokens": statement_tokens, "mean_item_spacing": 100.0,
        "kept_slots": budget if budget is not None else length,
        "required_set_tokens": items * statement_tokens,
        "distractors": distractors, "partial_recall": partial_recall,
    }


def _matrix_rows(length, *, multikey_8_correct=False):
    """Full-KV right everywhere; small items need 64, big items need 128."""
    rows = []
    for seed in range(2):
        for family, items in (("aggregate", 1), ("aggregate", 4), ("multikey", 8)):
            rows.append(_row(family, items, seed, length, "full", True))
            good64 = (family, items) == ("aggregate", 1)
            good128 = good64 or (family, items) == ("aggregate", 4) or multikey_8_correct
            rows.append(_row(family, items, seed, length, "b64", good64, 64))
            rows.append(_row(family, items, seed, length, "b128", good128, 128))
    return rows


def test_summary_matrix_and_required_budget_per_cell():
    summary = summarize(_matrix_rows(512), budgets=[64, 128], tolerance=0.0)
    assert summary["accuracy"]["aggregate"]["1"]["64"] == 1.0
    assert summary["accuracy"]["aggregate"]["4"]["64"] == 0.0
    assert summary["accuracy"]["aggregate"]["4"]["128"] == 1.0
    assert summary["accuracy"]["aggregate"]["4"]["full"] == 1.0
    assert summary["required_budget"]["aggregate"]["1"] == 64
    assert summary["required_budget"]["aggregate"]["4"] == 128
    assert summary["required_budget"]["multikey"]["8"] is None
    assert summary["cells"]["aggregate"]["4"]["required_set_tokens"] == 80.0
    assert summary["prediction_checks"]["cliff"]["cells_above"] > 0
    assert summary["prediction_checks"]["cliff"]["cells_below"] > 0


def test_summary_pools_lengths_and_checks_flatness():
    rows = _matrix_rows(512) + [
        {**_row(row["family"], row["items"], row["seed"] + 10, 1024, row["arm"],
               row["correct"] if row["arm"] != "b64" else True,
               row["budget"])
         } for row in _matrix_rows(512)
    ]
    summary = summarize(rows, budgets=[64, 128], tolerance=0.0)
    assert summary["lengths"] == [512, 1024]
    assert summary["required_budget_by_length"]["512"]["aggregate"]["4"] == 128
    assert summary["required_budget_by_length"]["1024"]["aggregate"]["4"] == 64
    # top level: the budget that matched at *every* tested length
    assert summary["required_budget"]["aggregate"]["4"] == 128
    assert summary["prediction_checks"]["required_budget_flat_in_length"] is False

    flat_rows = _matrix_rows(512) + [
        {**_row(row["family"], row["items"], row["seed"] + 10, 1024, row["arm"],
               row["correct"], row["budget"])}
        for row in _matrix_rows(512)
    ]
    flat = summarize(flat_rows, budgets=[64, 128], tolerance=0.0)
    assert flat["prediction_checks"]["required_budget_flat_in_length"] is True
    assert flat["prediction_checks"]["required_budget_grows_with_items"] is True


def test_summary_reports_required_budget_against_distractors():
    """The competition axis gets its own required_budget, next to --items."""
    rows = []
    for seed in range(2):
        for distractors, needed in ((4, 64), (128, 128)):
            items = distractors + 1
            rows.append(_row("multikey", items, seed, 512, "full", True,
                             distractors=distractors))
            rows.append(_row("multikey", items, seed, 512, "b64", needed == 64, 64,
                             distractors=distractors))
            rows.append(_row("multikey", items, seed, 512, "b128", True, 128,
                             distractors=distractors))
    summary = summarize(rows, budgets=[64, 128], tolerance=0.0)

    assert summary["required_budget_by_distractors"]["multikey"]["4"] == 64
    assert summary["required_budget_by_distractors"]["multikey"]["128"] == 128
    # the same numbers are visible on the --items axis, where the label is N + 1
    assert summary["required_budget"]["multikey"] == {"5": 64, "129": 128}
    assert summary["accuracy_by_distractors"]["multikey"]["4"]["64"] == 1.0
    assert summary["accuracy_by_distractors"]["multikey"]["128"]["64"] == 0.0
    assert summary["prediction_checks"]["required_budget_grows_with_distractors"] is True
    assert summary["prediction_checks"]["required_budget_grows_with_distractors_cells"] == 1
    assert summary["cells"]["multikey"]["129"]["distractors"] == 128.0


def test_summary_reports_accuracy_conditional_on_budget_covering_the_required_set():
    """The operational form of the cliff: conditional means and cell counts."""
    rows = [
        # needles k=2 needs 2 x 40 = 80 tokens: a 128-slot budget covers it
        _row("needles", 2, 0, 512, "full", True, statement_tokens=40.0),
        _row("needles", 2, 0, 512, "b128", True, 128, statement_tokens=40.0,
             partial_recall=1.0),
        _row("needles", 2, 1, 512, "b128", True, 128, statement_tokens=40.0,
             partial_recall=1.0),
        # needles k=8 needs 320 tokens: 128 does not cover it, 512 does
        _row("needles", 8, 0, 512, "full", True, statement_tokens=40.0),
        _row("needles", 8, 0, 512, "b128", False, 128, statement_tokens=40.0,
             partial_recall=0.25),
        _row("needles", 8, 0, 512, "b512", True, 512, statement_tokens=40.0,
             partial_recall=0.75),
    ]
    summary = summarize(rows, budgets=[128, 512], tolerance=0.0)
    block = summary["budget_covers_required_set"]

    # three bounded rows are covered (2 from k=2, 1 from k=8 at 512); the Full-KV
    # arm is never part of the split
    assert block["covered"]["rows"] == 3
    assert block["covered"]["cells"] == 2
    assert block["covered"]["mean_accuracy"] == 1.0
    assert block["covered"]["mean_partial_recall"] == round((1 + 1 + 0.75) / 3, 4)
    assert block["covered"]["mean_kept_slots"] == round((128 + 128 + 512) / 3, 2)
    assert block["covered"]["mean_required_set_tokens"] == round((80 + 80 + 320) / 3, 2)
    assert block["not_covered"]["rows"] == 1
    assert block["not_covered"]["cells"] == 1
    assert block["not_covered"]["mean_accuracy"] == 0.0
    assert block["not_covered"]["mean_partial_recall"] == 0.25
    assert block["separation"]["accuracy"] == 1.0
    assert block["separation"]["partial_recall"] == round(0.9167 - 0.25, 4)

    # per family, per cell and per budget cuts
    assert block["by_family"]["needles"]["covered"]["rows"] == 3
    assert block["by_family"]["needles"]["not_covered"]["rows"] == 1
    cell = block["by_cell"]["needles"]["8"]
    assert cell["covered"]["rows"] == 1 and cell["not_covered"]["rows"] == 1
    assert block["by_budget"]["128"]["covered"]["rows"] == 2
    assert block["by_budget"]["128"]["not_covered"]["rows"] == 1
    assert block["by_budget"]["512"]["covered"]["rows"] == 1
    assert block["by_budget"]["512"]["not_covered"]["rows"] == 0

    # the same split is recorded per (family, configuration, budget) cell
    budgets_cell = summary["cells"]["needles"]["8"]["budgets"]
    assert budgets_cell["128"]["covered"]["rows"] == 0
    assert budgets_cell["128"]["not_covered"]["rows"] == 1
    assert budgets_cell["512"]["covered"]["rows"] == 1
    assert budgets_cell["512"]["not_covered"]["rows"] == 0

    # a row that carries the flag is trusted over the arithmetic
    flagged = _row("needles", 8, 0, 512, "b64", True, 64, statement_tokens=40.0)
    flagged["budget_covers_required_set"] = True
    only = summarize([flagged], budgets=[64])
    assert only["budget_covers_required_set"]["covered"]["rows"] == 1
    assert only["budget_covers_required_set"]["not_covered"]["rows"] == 0


def test_summary_marks_cells_the_full_kv_arm_cannot_solve():
    """Full-KV accuracy 0 -> required_budget null and a full_kv_unsolved entry."""
    rows = [
        # the model cannot do this task even with every slot
        _row("aggregate", 2, 0, 512, "full", False),
        _row("aggregate", 2, 1, 512, "full", False),
        _row("aggregate", 2, 0, 512, "b64", False, 64),
        _row("aggregate", 2, 1, 512, "b64", False, 64),
        # a task Full-KV solves, so a budget can be demanded of the bounded arm
        _row("multikey", 1, 0, 512, "full", True),
        _row("multikey", 1, 0, 512, "b64", True, 64),
    ]
    summary = summarize(rows, budgets=[64], tolerance=0.0)

    assert summary["required_budget"]["aggregate"]["2"] is None
    assert summary["accuracy"]["aggregate"]["2"]["full"] == 0.0
    unsolved = summary["full_kv_unsolved"]
    assert unsolved["count"] == 1
    assert set(unsolved["cells"]) == {"aggregate"}
    cell = unsolved["cells"]["aggregate"]["2"]
    assert cell["full_accuracy"] == 0.0
    assert cell["required_budget"] is None
    assert cell["records"] == 2 and cell["seeds"] == 2 and cell["lengths"] == [512]
    assert summary["cells"]["aggregate"]["2"]["full_kv_unsolved"] is True
    assert summary["cells"]["aggregate"]["2"]["full_kv_accuracy"] == 0.0
    # the solved cell is not flagged and still yields a budget
    assert summary["cells"]["multikey"]["1"]["full_kv_unsolved"] is False
    assert summary["required_budget"]["multikey"]["1"] == 64
    assert "multikey" not in unsolved["cells"]
    # per-length view carries the same flag
    assert summary["by_length"]["512"]["full_kv_unsolved"]["count"] == 1


def test_cells_block_defines_the_required_set_without_double_counting():
    """required_set_tokens is items x tokens per statement, not items x block."""
    tokenizer = WordTokenizer()
    record = needles_record(items=4, seed=0, target_tokens=400, tokenizer=tokenizer)
    each = record["statement_tokens"] / 4
    rows = [
        _row("needles", 4, 0, 512, "full", True, statement_tokens=each),
        {**_row("needles", 4, 0, 512, "b64", True, 64, statement_tokens=each),
         "required_set_tokens": round(record["statement_tokens"], 1),
         "tokens_per_item": round(each, 2)},
    ]
    summary = summarize(rows, budgets=[64], tolerance=0.0)
    cell = summary["cells"]["needles"]["4"]
    assert cell["required_set_tokens"] == round(record["statement_tokens"], 1)
    assert cell["statement_tokens"] == round(each, 2)          # mean block per row
    assert cell["tokens_per_item"] == round(each, 2)
    # the earlier revision multiplied the whole block by items a second time
    assert cell["required_set_tokens"] < 4 * record["statement_tokens"]
    assert "items x mean tokens per item statement" in cell["required_set_tokens_rule"]


# --------------------------------------------------------------------------- #
# the shared-prefill fast path
# --------------------------------------------------------------------------- #

def test_shared_prefill_selection_matches_the_packaged_compile():
    model = tiny_model()
    tokenizer = WordTokenizer()
    record = multikey_record(items=2, seed=0, target_tokens=90, tokenizer=tokenizer)
    ids = torch.tensor([encode_ids(tokenizer, record["prompt"])], dtype=torch.long)
    length = int(ids.shape[1])
    config = RetentionConfig(budget=16, lex_cap=8, observation_window=8,
                             prefill_chunk=32, key_chunk=32)

    cache, _logits, captured = prefill_capture(model, ids, config.observation_window,
                                               config.prefill_chunk)
    layers = [(layer.keys, layer.values) for layer in cache.layers]
    scores = observation_scores(model, captured, cache, length, config)
    anchors = lexical_anchors(tokenizer, record["prompt"], config)
    shared = clone_cache(layers)
    prune_cache(shared, select_indices(scores, shared, length, config, anchors))

    reference, _logits = compile_bounded_cache(model, ids, config, tokenizer=tokenizer)
    assert shared.get_seq_length() == reference.get_seq_length()
    for got, want in zip(shared.layers, reference.layers):
        assert torch.equal(got.keys, want.keys)
        assert torch.equal(got.values, want.values)


def test_greedy_decode_from_a_compiled_cache_matches_full_kv():
    """The decode loop's absolute positions are right even when nothing drops."""
    from benchmarks.benchmark_required_set import forward_kwargs_of, greedy

    model = tiny_model()
    ids = torch.randint(1, 256, (1, 64))
    length = int(ids.shape[1])
    kwargs_names = forward_kwargs_of(model)
    config = RetentionConfig(budget=length, lex_cap=0, observation_window=8,
                             prefill_chunk=32, key_chunk=32)

    full_cache, full_logits, _captured = prefill_capture(model, ids, 8, 32)
    full_tokens, _seconds = greedy(model, full_cache,
                                   full_logits[:, -1:].argmax(-1), length, 4,
                                   set(), kwargs_names)
    compiled, compiled_logits = compile_bounded_cache(model, ids, config)
    assert compiled.get_seq_length() == length          # budget covers the prompt
    bounded_tokens, _seconds = greedy(model, compiled,
                                      compiled_logits[:, -1:].argmax(-1), length, 4,
                                      set(), kwargs_names)
    assert full_tokens == bounded_tokens
