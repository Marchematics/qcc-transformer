"""The 1M harness's prompt builder: exact needle positions and an opened answer.

The 1M row depends on two properties of the prompt that are easy to get wrong and
invisible in a pass/fail aggregate:

* the recorded needle *positions* must be the token positions of the planted
  statements, because the harness hands them to the selection as the lexical-anchor
  hits and skips the (expensive) anchor scan on the strength of them;
* the prompt must end with an opened answer.  Asked cold, Qwen2.5-0.5B re-lists the
  keys from the question - the earlier 128K run scored 0 on *both* arms for exactly
  that reason, which says nothing about retention.

These run on CPU with the real tokenizer, so the token-space splice is checked the way
the harness uses it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.benchmark_million_context import build_haystack, recall_of
from test_required_set import WordTokenizer


class DictTokenizer:
    """The harness indexes the tokenizer's result like a Hugging Face one.

    The tests' toy tokenizer returns an object with an `input_ids` attribute instead,
    so this adapts the call shape only (no tokenisation change), which keeps the test
    on CPU without a checkpoint download.
    """

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, text, add_special_tokens=False):
        encoded = self.inner(text, add_special_tokens=add_special_tokens)
        return {"input_ids": list(encoded.input_ids)}

    def decode(self, ids, skip_special_tokens=False):
        return self.inner.decode(ids, skip_special_tokens=skip_special_tokens)


@pytest.fixture(scope="module")
def tokenizer():
    return DictTokenizer(WordTokenizer())


@pytest.mark.parametrize("items", [1, 4])
def test_needle_positions_are_the_statement_tokens(tokenizer, items):
    _prompt, ids, positions, values, keys = build_haystack(tokenizer, 600, items, 0)
    assert len(ids) == pytest.approx(600, abs=16)
    assert len(keys) == len(values) == items
    assert len(positions) > items                      # every statement token, not one
    assert positions == sorted(set(positions))
    assert positions[0] > 0 and positions[-1] < len(ids)
    decoded = tokenizer.decode([ids[p] for p in positions])
    for key in keys:
        assert key in decoded or key in tokenizer.decode(ids)
    for value in values:
        assert value in tokenizer.decode(ids[positions[0]:positions[-1] + 1]) or any(
            value in tokenizer.decode(ids[max(0, p - 12):p + 12]) for p in positions)


@pytest.mark.parametrize("items", [1, 4])
def test_prompt_opens_the_answer(tokenizer, items):
    prompt, ids, _positions, _values, _keys = build_haystack(tokenizer, 600, items, 0,
                                                             answer_prefix=True)
    tail = tokenizer.decode(ids[-12:])
    assert tail.rstrip().endswith(("is", "are"))       # '... are' / '... is'
    assert prompt == ""                                 # positions are exact: no text form
    cold, cold_ids, *_ = build_haystack(tokenizer, 600, items, 0, answer_prefix=False)
    # the filler absorbs the difference, so both prompts hit the requested length
    assert len(cold_ids) == pytest.approx(len(ids), abs=4)
    assert "The magic" not in tokenizer.decode(cold_ids[-12:])


def test_a_different_seed_moves_the_needles(tokenizer):
    _p, ids_a, pos_a, values_a, _k = build_haystack(tokenizer, 600, 2, 0)
    _p, ids_b, pos_b, values_b, _k = build_haystack(tokenizer, 600, 2, 1)
    assert values_a != values_b
    assert pos_a != pos_b
    assert len(ids_a) == len(ids_b) == pytest.approx(600, abs=16)


def test_recall_requires_every_value():
    assert recall_of("the numbers are 111111, 222222", ["111111", "222222"]) == 1.0
    assert recall_of("the numbers are 111111", ["111111", "222222"]) == 0.0
    assert recall_of("THE NUMBERS ARE 111111, 222222.", ["111111", "222222"]) == 1.0


def test_recall_matches_whole_words_not_substrings():
    assert recall_of("the magic words are obsidian, gondola", ["obsidian", "gondola"]) == 1.0
    assert recall_of("OBSIDIAN and Gondola", ["obsidian", "gondola"]) == 1.0
    assert recall_of("the magic words are obsidians", ["obsidian"]) == 0.0
    assert recall_of("gondola", ["obsidian", "gondola"]) == 0.0


def test_recall_on_digits_ignores_separators_but_not_missing_digits():
    values = ["536110", "509532"]
    assert recall_of("536110, 509532.", values, value_style="digits") == 1.0
    assert recall_of("536110 and 509532", values, value_style="digits") == 1.0
    # a dropped leading digit is not a separator problem, and must not score
    assert recall_of("536110, 09532.", values, value_style="digits") == 0.0
    assert recall_of("536110", values, value_style="digits") == 0.0


def test_word_values_are_single_tokens_on_the_real_tokenizer():
    """The whole point of the `words` style: one token per value, so the metric is
    not decided by multi-token digit generation."""
    checkpoint = "/root/qcc/models/Qwen2.5-0.5B-Instruct"
    if not Path(checkpoint).exists():
        pytest.skip("checkpoint not present")
    from transformers import AutoTokenizer
    from benchmarks.benchmark_million_context import _VALUE_WORDS
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    pieces = {word: tokenizer(word, add_special_tokens=False)["input_ids"]
              for word in _VALUE_WORDS}
    assert all(len(ids) == 1 for ids in pieces.values()), {
        w: ids for w, ids in pieces.items() if len(ids) != 1}
    assert len(set(pieces)) == len(_VALUE_WORDS)          # all distinct
    # a value that also occurs in the filler would score without being retrieved
    from benchmarks.benchmark_million_context import FILLER
    assert not [w for w in _VALUE_WORDS if w in FILLER.lower()]
    _p, ids, positions, values, keys = build_haystack(tokenizer, 2048, 3, 0)
    assert len(values) == len(keys) == 3 == len(set(values))
    for position in positions:
        assert 0 <= position < len(ids)
