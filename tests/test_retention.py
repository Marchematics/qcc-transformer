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


@torch.no_grad()
def greedy_from_cache(model, cache, first_token, steps, attention_mask=None):
    """Greedy decode from a compiled cache, HF-style.

    Position ids come from the attention mask's cumulative sum, exactly as
    ``prepare_inputs_for_generation`` derives them for a padded batch.
    """
    generated = []
    tokens = first_token
    mask = attention_mask
    for _ in range(steps):
        past = cache.get_seq_length()
        if mask is None:
            positions = torch.full((tokens.shape[0], 1), past, dtype=torch.long)
        else:
            positions = (mask.long().cumsum(-1) - 1)[:, -1:]
        out = model(tokens, past_key_values=cache, position_ids=positions,
                    cache_position=torch.arange(past, past + 1), use_cache=True)
        tokens = out.logits[:, -1:].argmax(-1)
        generated.append(tokens)
        if mask is not None:
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=-1)
    return torch.cat(generated, dim=1)


def _rag_batch(side):
    """Two requests of different lengths, padded on ``side``."""
    model = tiny_model()
    ids_long = torch.randint(0, 256, (1, 96))
    ids_short = torch.randint(0, 256, (1, 60))
    width = ids_long.shape[1]
    pad = torch.zeros(1, width - ids_short.shape[1], dtype=torch.long)
    if side == "right":
        batch = torch.cat([ids_short, pad], dim=1)
    else:
        batch = torch.cat([pad, ids_short], dim=1)
    batch = torch.cat([ids_long, batch], dim=0)
    mask = (batch != 0).long()
    mask[0] = 1                                   # row 0 is full-length
    return model, ids_long, ids_short, batch, mask


def test_batched_compile_matches_per_row_compile():
    config = RetentionConfig(budget=24, lex_cap=0, observation_window=8,
                             prefill_chunk=32, key_chunk=32)
    for side in ("left", "right"):
        model, ids_long, ids_short, batch, mask = _rag_batch(side)
        cache, logits = compile_bounded_cache(
            model, batch, config, attention_mask=mask)
        assert cache.qcc_attention_mask.shape == (2, cache.get_seq_length())
        assert cache.qcc_prompt_lengths.tolist() == [96, 60]

        for row, ids in enumerate((ids_long, ids_short)):
            reference, reference_logits = compile_bounded_cache(model, ids, config)
            slots = reference.get_seq_length()
            assert cache.qcc_attention_mask[row].sum() == slots
            for got, want in zip(cache.layers, reference.layers):
                assert torch.equal(got.keys[row: row + 1], want.keys), side
                assert torch.equal(got.values[row: row + 1], want.values), side
            assert torch.equal(logits[row: row + 1], reference_logits), side

            # duplicated filler must not change what the row attends to
            mask_row = torch.zeros(1, cache.get_seq_length(), dtype=torch.long)
            mask_row[0, :slots] = 1
            batched = greedy_from_cache(
                model, _clone_cache(cache, row), logits[row: row + 1].argmax(-1),
                3, mask_row)
            alone = greedy_from_cache(
                model, reference, reference_logits.argmax(-1), 3)
            assert torch.equal(batched, alone), (side, row, batched, alone)


@torch.no_grad()
def _clone_cache(cache, row):
    """The single row's slot of a merged cache, as its own cache."""
    from transformers import DynamicCache

    clone = DynamicCache()
    for layer_idx, layer in enumerate(cache.layers):
        clone.update(layer.keys[row: row + 1].clone(),
                     layer.values[row: row + 1].clone(), layer_idx)
    return clone


def test_uniform_batch_has_no_padding_slots():
    model = tiny_model()
    ids = torch.randint(1, 256, (2, 80))
    config = RetentionConfig(budget=16, lex_cap=4, observation_window=8,
                             prefill_chunk=32, key_chunk=32)
    cache, logits = compile_bounded_cache(model, ids, config)
    assert cache.get_seq_length() == 20
    assert bool(cache.qcc_attention_mask.all())
    assert logits.shape[0] == 2
    for row in range(2):
        reference, reference_logits = compile_bounded_cache(model, ids[row: row + 1], config)
        assert torch.equal(logits[row: row + 1], reference_logits)
        assert torch.equal(cache.layers[0].keys[row: row + 1],
                           reference.layers[0].keys)


def test_query_states_reads_a_fused_qkv_projection():
    """Phi-3 has no q_proj; the Q slice of qkv_proj must be picked out."""
    import torch.nn as nn

    from qcc_transformer.retention import _query_states

    heads, head_dim = 4, 8
    attn = nn.Module()
    attn.qkv_proj = nn.Linear(heads * head_dim, 3 * heads * head_dim, bias=True)
    hidden = torch.randn(3, heads * head_dim)
    got = _query_states(attn, hidden, heads, head_dim)
    want = attn.qkv_proj(hidden)[..., : heads * head_dim]
    want = want.view(3, heads, head_dim).transpose(0, 1)
    assert got.shape == (heads, 3, head_dim)
    assert torch.allclose(got, want)


def test_fixed_rope_length_pins_only_longrope_checkpoints():
    import torch.nn as nn

    from qcc_transformer.retention import fixed_rope_length

    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, x, position_ids, seq_len=None):
            self.seen.append(seq_len)
            return x, x

    def stub(rope_scaling, original=4096):
        model = nn.Module()
        model.config = type("C", (), {"rope_scaling": rope_scaling,
                                      "original_max_position_embeddings": original})()
        model.model = nn.Module()
        model.model.layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.rotary_emb = Recorder()
            model.model.layers.append(layer)
        return model

    model = stub({"type": "longrope"})
    x, positions = torch.zeros(1, 1, 8), torch.arange(4).unsqueeze(0)
    with fixed_rope_length(model, 20000) as pinned:
        assert pinned
        model.model.layers[0].self_attn.rotary_emb(x, positions)
    assert model.model.layers[0].self_attn.rotary_emb.seen == [20000]
    assert isinstance(model.model.layers[0].self_attn.rotary_emb, Recorder)

    short = stub({"type": "longrope"})
    with fixed_rope_length(short, 2048) as pinned:
        assert not pinned                      # short prompts keep the checkpoint's own choice
    with fixed_rope_length(stub(None), 20000) as pinned:
        assert not pinned                      # not a LongRoPE checkpoint
