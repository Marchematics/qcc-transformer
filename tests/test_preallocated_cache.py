"""The preallocated cache must be a drop-in for `DynamicCache`, exactly.

`PreallocatedCache` exists for one reason - `torch.cat` needs the old and the new
cache alive at once, which is 24 GiB at 1M on the 0.5B checkpoint - so the tests have
to show that the replacement changes *only* the growth policy: identical logits,
identical cached keys and values, identical reported length, and a hard error rather
than silent truncation when the buffer is too small.
"""

from __future__ import annotations

import pytest
import torch

from qcc_transformer.preallocated_cache import (PreallocatedCache,
                                                preallocated_cache_for)

from test_required_set import tiny_model


def _ids(model, length=48):
    torch.manual_seed(0)
    return torch.randint(1, model.config.vocab_size, (1, length))


@torch.no_grad()
def _run(model, cache, ids, chunk):
    """Chunked prefill through the model, returning the last chunk's logits."""
    logits = None
    for start in range(0, ids.shape[1], chunk):
        end = min(ids.shape[1], start + chunk)
        position = torch.arange(start, end)
        out = model(ids[:, start:end], past_key_values=cache, use_cache=True,
                    cache_position=position, position_ids=position.unsqueeze(0))
        logits = out.logits[:, -1:]
    return logits


def test_matches_dynamic_cache_on_a_chunked_prefill():
    model, ids = tiny_model(), None
    ids = _ids(model)
    reference = _run(model, cache := None, ids, 16) if False else None
    from transformers import DynamicCache
    dynamic = DynamicCache()
    reference = _run(model, dynamic, ids, 16)
    preallocated = preallocated_cache_for(model, capacity=ids.shape[1])
    got = _run(model, preallocated, ids, 16)

    assert torch.equal(got, reference)
    assert preallocated.get_seq_length() == dynamic.get_seq_length() == ids.shape[1]
    assert len(preallocated.layers) == len(dynamic.layers)
    for layer, ref_layer in zip(preallocated.layers, dynamic.layers):
        assert layer.length == ids.shape[1]
        assert torch.equal(layer.keys[:, :, :ids.shape[1]], ref_layer.keys)
        assert torch.equal(layer.values[:, :, :ids.shape[1]], ref_layer.values)


def test_decode_step_appends_in_place_and_keeps_matching_logits():
    from transformers import DynamicCache
    model = tiny_model()
    ids = _ids(model)
    dynamic, preallocated = DynamicCache(), preallocated_cache_for(model, 64)
    reference = _run(model, dynamic, ids, 16)
    got = _run(model, preallocated, ids, 16)
    assert torch.equal(got, reference)

    token = reference.argmax(-1)
    for step in range(3):
        position = torch.tensor([ids.shape[1] + step])
        ref_out = model(token, past_key_values=dynamic, use_cache=True,
                        cache_position=position, position_ids=position.unsqueeze(0))
        out = model(token, past_key_values=preallocated, use_cache=True,
                    cache_position=position, position_ids=position.unsqueeze(0))
        assert torch.equal(out.logits, ref_out.logits)
        token = out.logits[:, -1:].argmax(-1)
    assert preallocated.get_seq_length() == dynamic.get_seq_length()
    assert preallocated.get_seq_length() == ids.shape[1] + 3


def test_footprint_is_exactly_capacity_and_does_not_grow():
    model = tiny_model()
    cache = preallocated_cache_for(model, capacity=512)
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    per_layer = (512 * config.num_key_value_heads * head_dim
                 * cache.layers[0].keys.element_size() * 2)      # k and v
    assert cache.nbytes == per_layer * config.num_hidden_layers
    before = cache.nbytes
    _run(model, cache, _ids(model, 64), 32)
    assert cache.nbytes == before                    # written slots reuse the buffer
    assert cache.get_seq_length() == 64
    assert cache.layers[0].keys.shape == (1, config.num_key_value_heads, 512, head_dim)


def test_overrunning_the_buffer_grows_correctly_and_says_so():
    """An undersized buffer must never truncate; it grows like `DynamicCache` and
    records that it had to, so a caller can tell the sizing was wrong."""
    from transformers import DynamicCache
    model = tiny_model()
    ids = _ids(model, 16)
    dynamic = DynamicCache()
    reference = _run(model, dynamic, ids, 16)
    cache = preallocated_cache_for(model, capacity=8)
    got = _run(model, cache, ids, 16)
    assert cache.overflowed is True
    assert cache.get_seq_length() == dynamic.get_seq_length() == 16
    assert torch.equal(got, reference)
    for layer, ref_layer in zip(cache.layers, dynamic.layers):
        assert torch.equal(layer.keys, ref_layer.keys)
        assert torch.equal(layer.values, ref_layer.values)
    cache.reset()
    assert cache.overflowed is False and cache.get_seq_length() == 0


def test_reset_reuses_the_buffer_for_a_recompile():
    model = tiny_model()
    cache = preallocated_cache_for(model, capacity=64)
    ids = _ids(model)
    first = _run(model, cache, ids, 16)
    cache.reset()
    assert cache.get_seq_length() == 0
    second = _run(model, cache, ids, 16)
    assert torch.equal(first, second)


def test_gather_into_a_smaller_selection_switches_to_append_semantics():
    """The compile path replaces ``layer.keys`` with a gathered tensor (the benchmark
    harness does exactly this), so the cache must report the compiled width and keep
    working for decode."""
    model = tiny_model()
    cache = preallocated_cache_for(model, capacity=64)
    ids = _ids(model)
    _run(model, cache, ids, 16)
    buffer = cache.layers[0].keys.clone()                  # snapshot before compiling
    slots = 4
    for layer in cache.layers:
        gather = torch.arange(slots).view(1, 1, slots, 1).expand(
            1, layer.keys.shape[1], slots, layer.keys.shape[-1])
        layer.keys = layer.keys.gather(2, gather).contiguous()
        layer.values = layer.values.gather(2, gather).contiguous()
    assert cache.get_seq_length() == slots                 # the compiled width, not 48
    assert torch.equal(cache.layers[0].keys, buffer[:, :, :slots])

    token = ids[:, :1]
    position = torch.tensor([ids.shape[1]])
    out = model(token, past_key_values=cache, use_cache=True, cache_position=position,
                position_ids=position.unsqueeze(0))
    assert out.logits.shape[-1] == model.config.vocab_size
    assert cache.get_seq_length() == slots + 1              # appended, not overwritten

    cache.reset()                                          # reuse still works
    assert all(layer.length == 0 for layer in cache.layers)
