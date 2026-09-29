"""The prefill/decode entry points must not leave an autograd graph behind.

This pins the root cause of the long-context OOM that blocked the 128K-1M rows.
The KV cache is a *persistent* structure that a chunked prefill grows with
``torch.cat``, so with gradient tracking on, every chunk's graph stays reachable
from the cache tensors the caller keeps: the same 2,201-token chunk retains
2,813.6 MiB (1.31 MiB/token) instead of 33.9 MiB (0.015 MiB/token), and the
difference accumulates per chunk until the card is full.  Nothing in the retrofit
is trainable, so the fix is `no_grad` on the entry points - and the tests below
check the mechanism directly (no `grad_fn` on cache tensors, no grad mode inside
the forward) rather than trusting the decorator.
"""

from __future__ import annotations

import torch

from benchmarks import benchmark_bounded_decode_frontier as L
from benchmarks.benchmark_bounded_decode_frontier import decode, prefill_accumulate, prefill_capture

from test_required_set import WordTokenizer, tiny_model


def _grad_mode_seen(model):
    """Record ``torch.is_grad_enabled()`` at every layer's attention forward."""
    seen = []
    handles = [layer.self_attn.register_forward_hook(
        lambda module, args, output: seen.append(torch.is_grad_enabled()))
        for layer in model.model.layers]
    return seen, handles


def _ids(model, length=64):
    torch.manual_seed(0)
    return torch.randint(1, model.config.vocab_size, (1, length))


def test_prefill_capture_runs_with_grad_tracking_off():
    model = tiny_model()
    ids = _ids(model)
    assert torch.is_grad_enabled()                            # the caller's default
    seen, handles = _grad_mode_seen(model)
    try:
        cache, logits, tails = prefill_capture(model, ids, obs=16, chunk=32)
    finally:
        for handle in handles:
            handle.remove()
    assert seen and not any(seen)                             # forward saw grad off
    for layer in cache.layers:
        assert layer.keys.grad_fn is None                     # nothing to retain
        assert layer.values.grad_fn is None
    for window in tails.values():
        assert not window.requires_grad
    assert logits is not None and not logits.requires_grad


def test_grad_enabled_prefill_would_retain_a_graph():
    """The control: without `no_grad` the cache itself carries the graph.

    This is what makes the assertion above non-vacuous - it is the retention
    mechanism, reproduced on a tiny model.
    """
    model = tiny_model()
    ids = _ids(model)
    cache = L.DynamicCache()
    pos = torch.arange(ids.shape[1])
    with torch.enable_grad():
        model(ids, past_key_values=cache, use_cache=True,
              cache_position=pos, position_ids=pos.unsqueeze(0))
    assert any(layer.keys.grad_fn is not None for layer in cache.layers)

    cache = L.DynamicCache()
    prefill_capture(model, ids, obs=16, chunk=32)
    plain = prefill_capture(model, ids, obs=16, chunk=ids.shape[1])
    assert all(layer.keys.grad_fn is None for layer in plain[0].layers)


def test_prefill_accumulate_and_decode_run_with_grad_tracking_off():
    model = tiny_model()
    ids = _ids(model)
    seen, handles = _grad_mode_seen(model)
    try:
        cache, logits, tails, mass, top1 = prefill_accumulate(
            model, ids, obs=16, chunk=32, key_chunk=32)
    finally:
        for handle in handles:
            handle.remove()
    assert seen and not any(seen)
    assert all(layer.keys.grad_fn is None for layer in cache.layers)

    L.TOKENIZER = WordTokenizer()
    seen, handles = _grad_mode_seen(model)
    try:
        text, generated, _elapsed = decode(
            model, cache, logits[:, -1:].argmax(-1), ids.shape[1], 2, set())
    finally:
        for handle in handles:
            handle.remove()
    assert seen and not any(seen)
    assert isinstance(text, str) and generated == 2
