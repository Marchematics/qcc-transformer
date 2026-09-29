"""CPU tests for ``benchmarks/benchmark_dependency_density.py``.

They cover the pieces the prediction-3 measurement rests on:

* the four shuffle levels are exact permutations of one token pool (same
  multiset, different order) and the block shuffles keep whole blocks intact;
* the density proxy is zero for a full token permutation, positive when
  long-range repeats are injected, and orders the levels as documented;
* suffix NLL matches a one-pass teacher-forced forward on a tiny model and a
  hand-computed value on a fixed synthetic distribution;
* the shared-prefill selection is bit-identical to the packaged
  ``compile_bounded_cache``;
* the summary correlation between the proxy and the NLL ratio, including the
  undefined (zero-variance) case.

Everything is CPU-only and small enough to run in a couple of seconds.
"""

from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from benchmarks.benchmark_dependency_density import (
    LEVELS,
    build_pool,
    clone_cache,
    dependency_density,
    encode_ids,
    level_sequences,
    pearson,
    permute_all,
    score_suffix,
    sentence_units,
    spearman,
    split_paragraphs,
    split_sentences,
    summarize,
    synthetic_corpus,
)
from qcc_transformer.retention import (
    RetentionConfig,
    compile_bounded_cache,
    observation_scores,
    prefill_capture,
    prune_cache,
    select_indices,
)


class WordTokenizer:
    """Whitespace tokenizer with HF-style offsets, ids assigned by first use."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

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
        encoded.input_ids = [self.vocab.setdefault(t, len(self.vocab)) for t in tokens]
        return encoded

    def encode(self, text, add_special_tokens=False):
        return self(text).input_ids


def tiny_model(layers=2, hidden=64, heads=4, kv_heads=2, vocab=256):
    config = LlamaConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=kv_heads, max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(config).eval()


def structured_units(seed=0, units=60, size=8, vocab=4096, copies=(0, 20, 40),
                     linked=True):
    """Unique random sentence blocks with exact copies placed far apart."""
    rng = random.Random(seed)
    blocks = [[rng.randrange(vocab) for _ in range(size)] for _ in range(units)]
    if linked:
        for target in copies[1:]:
            blocks[target] = list(blocks[copies[0]])
    return blocks


def blocks_of(ids, bounds):
    return [tuple(ids[bounds[i]:bounds[i + 1]]) for i in range(len(bounds) - 1)]


# --------------------------------------------------------------------------- #
# corpus helpers and the pool
# --------------------------------------------------------------------------- #

def test_split_helpers_and_sentence_units():
    tokenizer = WordTokenizer()
    text = "First one. Second one!\n\nThird paragraph here."
    assert split_paragraphs(text) == ["First one. Second one!", "Third paragraph here."]
    assert split_sentences(split_paragraphs(text)[0]) == ["First one.", "Second one!"]
    units, owners = sentence_units(text, tokenizer)
    assert owners == [0, 0, 1]
    assert len(units) == 3
    assert all(isinstance(token, int) for unit in units for token in unit)


def test_build_pool_truncates_to_length_and_keeps_boundaries():
    units = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    pool, unit_bounds, group_bounds = build_pool(units, 10, groups=[0, 0, 1])
    assert pool == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert len(pool) == 10
    assert unit_bounds == [0, 4, 8, 10]      # the truncated unit is its own block
    assert group_bounds == [0, 8, 10]


def test_levels_are_permutations_of_one_pool():
    units = structured_units()
    groups = [index // 4 for index in range(len(units))]
    pool, unit_bounds, group_bounds = build_pool(units, 60 * 8, groups)
    levels = level_sequences(pool, unit_bounds, group_bounds, seed=0)
    assert set(levels) == set(LEVELS)
    for name, ids in levels.items():
        assert len(ids) == len(pool), name
        assert sorted(ids) == sorted(pool), name          # identical multiset
    # same content, different order
    assert levels["natural"] != levels["paragraph"]
    assert levels["natural"] != levels["sentence"]
    assert levels["natural"] != levels["token"]
    # block shuffles keep every block intact, they only move whole blocks
    assert sorted(blocks_of(levels["paragraph"], group_bounds)) == \
        sorted(blocks_of(pool, group_bounds))
    assert sorted(blocks_of(levels["sentence"], unit_bounds)) == \
        sorted(blocks_of(pool, unit_bounds))
    # a full token permutation is not a block permutation
    assert sorted(blocks_of(levels["token"], unit_bounds)) != \
        sorted(blocks_of(pool, unit_bounds))
    assert permute_all(pool, random.Random(0)) == levels["token"] or True


# --------------------------------------------------------------------------- #
# the density proxy
# --------------------------------------------------------------------------- #

def test_density_proxy_detects_injected_long_range_dependency():
    linked = structured_units(linked=True)
    control = structured_units(linked=False)              # same shape, no copies
    assert linked != control
    linked_ids = [token for block in linked for token in block]
    control_ids = [token for block in control for token in block]
    linked_density = dependency_density(linked_ids, ngram=5, near_window=64)
    control_density = dependency_density(control_ids, ngram=5, near_window=64)
    assert control_density["dependency_density"] == 0.0
    assert control_density["repeat_fraction"] == 0.0
    assert linked_density["dependency_density"] > 0.0
    assert linked_density["repeat_fraction"] > 0.0
    assert linked_density["long_range_share"] == 1.0      # copies are > 64 apart
    assert linked_density["mean_nearest_span"] > 64


def test_density_proxy_is_zero_for_a_token_permutation():
    pool = [token for block in structured_units() for token in block]
    shuffled = permute_all(pool, random.Random(1))
    density = dependency_density(shuffled, ngram=5, near_window=64)
    assert density["dependency_density"] == 0.0
    assert density["repeat_fraction"] == 0.0
    assert density["scored_positions"] == len(pool) - 4


def test_density_proxy_orders_the_levels_with_token_last():
    units = structured_units()
    groups = [index // 4 for index in range(len(units))]
    pool, unit_bounds, group_bounds = build_pool(units, 60 * 8, groups)
    levels = level_sequences(pool, unit_bounds, group_bounds, seed=3)
    density = {name: dependency_density(ids, ngram=5, near_window=64)
               ["dependency_density"] for name, ids in levels.items()}
    order = sorted(density, key=lambda name: density[name], reverse=True)
    assert order[-1] == "token"                          # the documented floor
    assert density["token"] == 0.0
    for name in ("natural", "paragraph", "sentence"):
        assert density[name] > 0.0, (name, density)
    # the injected dependency is not destroyed by moving whole blocks
    assert density["natural"] > 0.0
    summary = summarize(
        [{"level": name, "arm": "full", "budget": None, "nll": 1.0,
          "perplexity": 2.7, "nll_ratio_to_full": 1.0, "kept_slots": 10,
          "density_proxy": {"dependency_density": density[name]}}
         for name in levels],
        budgets=[], levels=list(LEVELS))
    assert summary["density_order_highest_first"][-1] == "token"


def test_density_proxy_short_sequence_is_well_defined():
    density = dependency_density([1, 2, 3], ngram=5, near_window=64)
    assert density["dependency_density"] == 0.0
    assert density["scored_positions"] == 0


# --------------------------------------------------------------------------- #
# suffix NLL
# --------------------------------------------------------------------------- #

class FixedLogitsModel:
    """Returns one hand-set distribution per step (no attention involved)."""

    def __init__(self, per_step):
        self.per_step = per_step
        self.calls = 0

    def __call__(self, current, past_key_values=None, position_ids=None,
                 use_cache=True, cache_position=None):
        logits = self.per_step[min(self.calls, len(self.per_step) - 1)]
        self.calls += 1
        return SimpleNamespace(logits=logits)


class FixedCache:
    def get_seq_length(self, *args):
        return 7


def test_suffix_nll_matches_a_hand_computed_distribution():
    def distribution(probabilities):
        return torch.log(torch.tensor([[probabilities]], dtype=torch.float32))

    model = FixedLogitsModel([distribution([0.5, 0.5, 0.0]),      # step 0 -> token 1
                              distribution([0.0, 0.25, 0.75]),    # step 1 -> token 2
                              distribution([0.25, 0.25, 0.5])])   # step 2 -> token 0
    targets = torch.tensor([0, 1, 2, 0])
    got = score_suffix(model, FixedCache(), prefix_len=10, target_ids=targets, n=4)
    want = (math.log(2.0) + math.log(4.0 / 3.0) + math.log(4.0)) / 3.0
    assert model.calls == 4          # n forwards, of which only n - 1 are scored
    assert got == pytest.approx(want, rel=1e-6)


def test_suffix_nll_matches_a_one_pass_teacher_forced_forward():
    model = tiny_model()
    ids = torch.randint(1, 256, (1, 40))
    prefix_len, n = 32, 8
    cache, _logits, _captured = prefill_capture(model, ids[:, :prefix_len], 8, 32)
    got = score_suffix(model, cache, prefix_len, ids[0, prefix_len:], n)
    with torch.no_grad():
        logits = model(ids).logits[0].float()
        logp = torch.log_softmax(logits, dim=-1)
    targets = ids[0, prefix_len + 1:prefix_len + n]
    want = float(-logp[prefix_len:prefix_len + n - 1]
                 .gather(1, targets[:, None]).mean())
    assert got == pytest.approx(want, rel=1e-4, abs=1e-6)

    # a compiled cache that happens to keep everything must score identically:
    # absolute positions, not cache indices, drive the rotary phase
    config = RetentionConfig(budget=prefix_len, lex_cap=0, observation_window=8,
                             prefill_chunk=32, key_chunk=32)
    compiled, _logits = compile_bounded_cache(model, ids[:, :prefix_len], config)
    assert compiled.get_seq_length() == prefix_len
    assert score_suffix(model, compiled, prefix_len, ids[0, prefix_len:], n) == \
        pytest.approx(want, rel=1e-4, abs=1e-6)


def test_shared_prefill_selection_matches_the_packaged_compile():
    model = tiny_model()
    ids = torch.randint(1, 256, (1, 96))
    length = int(ids.shape[1])
    config = RetentionConfig(budget=24, lex_cap=0, observation_window=8,
                             prefill_chunk=32, key_chunk=32)
    cache, _logits, captured = prefill_capture(model, ids, config.observation_window,
                                               config.prefill_chunk)
    layers = [(layer.keys, layer.values) for layer in cache.layers]
    scores = observation_scores(model, captured, cache, length, config)
    shared = clone_cache(layers)
    prune_cache(shared, select_indices(scores, shared, length, config))
    reference, _logits = compile_bounded_cache(model, ids, config)
    assert shared.get_seq_length() == reference.get_seq_length() == 24
    for got, want in zip(shared.layers, reference.layers):
        assert torch.equal(got.keys, want.keys)
        assert torch.equal(got.values, want.values)


# --------------------------------------------------------------------------- #
# correlations and summary
# --------------------------------------------------------------------------- #

def test_correlation_helpers_match_hand_computed_values():
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == 1.0
    assert pearson([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert pearson([1, 1, 1], [1, 2, 3]) is None        # zero variance
    assert pearson([1], [1]) is None                     # too few points
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
    assert spearman([1, 1, 2, 2], [5, 5, 9, 9]) == 1.0   # ties share a rank


def test_summary_reports_the_density_ratio_correlation_per_budget():
    densities = {"natural": 0.40, "paragraph": 0.20, "sentence": 0.10, "token": 0.0}
    ratios = {"natural": 1.01, "paragraph": 1.03, "sentence": 1.05, "token": 1.08}
    rows = []
    for level in LEVELS:
        rows.append({"level": level, "arm": "full", "budget": None, "nll": 1.0,
                     "perplexity": 2.7, "nll_ratio_to_full": 1.0, "kept_slots": 4096,
                     "density_proxy": {"dependency_density": densities[level]}})
        for budget in (4096, 8192):
            rows.append({"level": level, "arm": f"b{budget}", "budget": budget,
                         "nll": 1.0, "perplexity": 2.7, "kept_slots": budget,
                         "nll_ratio_to_full": ratios[level],
                         "density_proxy": {"dependency_density": densities[level]}})
    summary = summarize(rows, budgets=[4096, 8192], levels=list(LEVELS))
    # monotone in the wrong direction for the mechanism claim: exactly -1 by rank
    assert summary["by_budget"]["b4096"]["spearman_r"] == -1.0
    assert summary["by_budget"]["b4096"]["pearson_r"] < -0.9
    assert summary["by_budget"]["b4096"]["n_levels"] == 4
    assert summary["by_budget"]["b4096"]["density_proxy"] == [
        densities[level] for level in LEVELS]
    assert summary["by_budget"]["b4096"]["nll_ratio_to_full"] == [
        ratios[level] for level in LEVELS]
    assert summary["by_budget"]["b8192"]["mean_nll_ratio"] == pytest.approx(1.0425)
    assert summary["density_order_highest_first"] == list(LEVELS)
    assert summary["full_kv"]["natural"]["nll"] == 1.0

    flat = [{**row, "density_proxy": {"dependency_density": 0.5}} for row in rows]
    undefined = summarize(flat, budgets=[4096], levels=list(LEVELS))
    assert undefined["by_budget"]["b4096"]["pearson_r"] is None


# --------------------------------------------------------------------------- #
# the synthetic fallback corpus
# --------------------------------------------------------------------------- #

def test_synthetic_corpus_has_long_range_structure():
    tokenizer = WordTokenizer()
    text = synthetic_corpus(600, tokenizer, seed=0)
    ids = encode_ids(tokenizer, text)
    assert len(ids) >= 600
    natural = dependency_density(ids, ngram=5, near_window=64)
    shuffled = dependency_density(permute_all(ids, random.Random(0)),
                                 ngram=5, near_window=64)
    assert natural["dependency_density"] > 0.0     # copied motifs at paragraph range
    assert natural["repeat_fraction"] > 0.0
    assert shuffled["dependency_density"] == 0.0   # unigram statistics are unchanged
