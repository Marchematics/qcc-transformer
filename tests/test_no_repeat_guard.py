"""The no-repeat n-gram guard: a decoding setting, applied to every arm.

The LongBench losses this project keeps measuring are decoder degeneration - a repeated
n-gram that collapses the answer - not missing content.  The guard has to remove exactly that
failure mode and nothing else, so these tests pin the standard `no_repeat_ngram_size`
semantics on hand-built logits.
"""

from __future__ import annotations

import torch

from benchmarks.benchmark_retention_longbench import pick_token


def _logits(favourite, second, size=8):
    row = torch.full((1, 1, size), -10.0)
    row[0, 0, favourite] = 10.0
    row[0, 0, second] = 9.0
    return row


def test_without_the_guard_the_argmax_is_taken():
    assert pick_token(_logits(3, 5), [1, 2], no_repeat_ngram=0) == 3


def test_the_guard_blocks_a_repeated_bigram_and_takes_the_next_best():
    # the generation already contains "7 3", so predicting 3 after 7 would repeat the bigram
    assert pick_token(_logits(3, 5), [7, 3, 7], no_repeat_ngram=2) == 5


def test_the_guard_leaves_a_new_continuation_alone():
    assert pick_token(_logits(3, 5), [1, 2, 9], no_repeat_ngram=2) == 3


def test_the_guard_needs_a_full_window_before_it_can_act():
    assert pick_token(_logits(3, 5), [7], no_repeat_ngram=3) == 3


def test_a_longer_setting_only_blocks_longer_repeats():
    history = [1, 2, 3, 4, 1, 2, 3]
    # repeating the 4-gram "1 2 3 4" is blocked at n=4 ...
    assert pick_token(_logits(4, 6), history, no_repeat_ngram=4) == 6
    # ... but the trigram "2 3 4" is not, because the ban only ever applies to an n-gram
    assert pick_token(_logits(4, 6), history, no_repeat_ngram=3) == 6
    assert pick_token(_logits(4, 6), history, no_repeat_ngram=5) == 4
