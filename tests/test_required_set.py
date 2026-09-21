"""CPU tests for ``benchmarks/benchmark_required_set.py``.

They cover the parts of the runner that decide what the numbers mean: prompt
construction for both families (with hand-computed aggregate answers and a
multikey distractor), the numeric scoring, the ``required_budget`` derivation
from a synthetic summary matrix (including the null case), and the shared-prefill
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
    build_record,
    char_f1,
    clone_cache,
    encode_ids,
    multikey_record,
    parse_numbers,
    required_budget_from_matrix,
    score_prediction,
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


def _row(family, items, seed, length, arm, correct, budget=None, statement_tokens=20.0):
    return {
        "family": family, "items": items, "seed": seed, "length": length,
        "arm": arm, "budget": budget, "correct": bool(correct),
        "statement_tokens": statement_tokens, "mean_item_spacing": 100.0,
        "kept_slots": budget if budget is not None else length,
        "required_set_tokens": items * statement_tokens,
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
