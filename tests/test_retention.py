"""CPU tests for the packaged retention law.

These build a tiny randomly-initialised Llama so they run without a download or
a GPU, and they assert the invariants the design depends on:

* the compiled cache has exactly ``budget + lex_cap`` slots for every head,
  layer and request (a single 2-D mask must be able to describe a batch);
* attention sinks and the recent window are always present;
* retaining everything reproduces the uncompiled prefill's tokens;
* lexical anchors locate the answer span that attention ranking can miss.
"""

from __future__ import annotations

import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from qcc_transformer.retention import (
    RetentionConfig,
    compile_bounded_cache,
    lexical_anchors,
    select_indices,
)


class StubTokenizer:
    """Whitespace tokenizer with HF-style offset mapping, enough for anchors."""

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
        encoded.input_ids = list(range(len(tokens)))
        return encoded


def tiny_model(layers=2, hidden=64, heads=4, kv_heads=2, vocab=256):
    config = LlamaConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 2,
        num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=kv_heads, max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(config).eval()


def test_compiled_cache_has_uniform_width():
    model = tiny_model()
    ids = torch.randint(0, 256, (1, 64))
    config = RetentionConfig(budget=8, lex_cap=4, observation_window=8,
                             prefill_chunk=16, key_chunk=16)
    cache, _ = compile_bounded_cache(model, ids, config)
    widths = {layer.keys.shape[2] for layer in cache.layers}
    assert widths == {12}, widths                      # budget + lex_cap
    for layer in cache.layers:
        assert layer.keys.shape[1] == model.config.num_key_value_heads
        assert layer.values.shape == layer.keys.shape


def test_keeping_everything_reproduces_full_prefill():
    model = tiny_model()
    ids = torch.randint(0, 256, (1, 48))
    config = RetentionConfig(budget=48, lex_cap=0, observation_window=8,
                             prefill_chunk=16, key_chunk=16)
    cache, logits = compile_bounded_cache(model, ids, config)
    assert cache.get_seq_length() == 48
    with torch.no_grad():
        reference = model(ids, use_cache=True)
    same = torch.equal(logits.argmax(-1), reference.logits[:, -1:].argmax(-1))
    assert same


def test_sinks_and_recent_are_always_retained():
    model = tiny_model()
    ids = torch.randint(0, 256, (1, 96))
    config = RetentionConfig(budget=16, lex_cap=0, attention_sinks=3,
                             recent_fraction=0.25, observation_window=8,
                             prefill_chunk=32, key_chunk=32)
    cache, _, captured = None, None, None
    from qcc_transformer.retention import prefill_capture, observation_scores

    cache, _, captured = prefill_capture(model, ids, config.observation_window,
                                         config.prefill_chunk)
    scores = observation_scores(model, captured, cache, ids.shape[1], config)
    indices = select_indices(scores, cache, ids.shape[1], config)
    kept = set(indices[0][0].tolist())
    assert set(range(3)) <= kept                      # attention sinks
    recent = range(ids.shape[1] - config.recent_window, ids.shape[1])
    assert set(recent) <= kept                        # recent window


def test_lexical_anchors_find_the_repeated_key():
    tokenizer = StubTokenizer()
    config = RetentionConfig(observation_window=8, lex_cap=64,
                             min_word_len=6, min_number_len=4)
    prompt = ("filler " * 20
              + "One of the special magic numbers for ruthless-hierarchy is: 7549132. "
              + "filler " * 20
              + "What is the special magic number for ruthless-hierarchy mentioned here?")
    anchors = lexical_anchors(tokenizer, prompt, config)
    words = tokenizer(prompt).offset_mapping
    covered = {prompt[a:b] for a, b in (words[i] for i in anchors)}
    assert any("ruthless-hierarchy" in word for word in covered), covered
    assert any("7549132" in word for word in covered), covered


def test_lexical_anchors_follow_assignment_chains():
    tokenizer = StubTokenizer()
    config = RetentionConfig(observation_window=10, lex_cap=128, chain_hops=8,
                             min_word_len=3, min_number_len=4, max_occurrences=8)
    chain = ["VAR A = 12859", "VAR B = VAR A", "VAR C = VAR B", "VAR D = VAR C"]
    prompt = ("filler " * 10 + "\n" + "\n".join(chain) + "\n" + "filler " * 10
              + "\nQuestion: which variables are assigned the value 12859 above?")
    anchors = lexical_anchors(tokenizer, prompt, config)
    covered = {prompt[a:b] for a, b in
               (tokenizer(prompt).offset_mapping[i] for i in anchors)}
    for name in ("A", "B", "C", "D"):
        assert any(name == word for word in covered), (name, sorted(covered))
