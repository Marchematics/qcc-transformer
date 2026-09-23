"""The long-context prefill patch must not change single-token decode.

``install_flash_layer_attn`` replaces every layer's attention forward so the
multi-query prefill can go through the fused kernel.  Decode is handed back to the
module's *original* forward - that is what keeps the model's own kernel choice
(``flash_attention_2`` when the model was loaded that way) and what makes the TPOT
comparison like-for-like between arms, since both arms then decode with the same
operator.  The delegation is the part that can silently break: Hugging Face names the
cache argument ``past_key_values`` on some versions and ``past_key_value`` on others,
and a wrong name raises rather than degrading.

These tests run on CPU with a tiny Llama, where the fused prefill kernel cannot run,
so the cache is built *before* the patch is installed and only decode goes through the
patched forward - exactly the path under test.
"""

from __future__ import annotations

import copy

import pytest
import torch

from benchmarks import benchmark_bounded_decode_frontier as L

from test_required_set import tiny_model


def _ids(model, length=32):
    torch.manual_seed(0)
    return torch.randint(1, model.config.vocab_size, (1, length))


@torch.no_grad()
def _prefill(model, ids):
    cache = L.DynamicCache()
    position = torch.arange(ids.shape[1])
    out = model(ids, past_key_values=cache, use_cache=True, cache_position=position,
                position_ids=position.unsqueeze(0))
    return out.logits[:, -1:], cache


@torch.no_grad()
def _decode_step(model, cache, token, position):
    position = torch.tensor([position])
    out = model(token, past_key_values=cache, use_cache=True, cache_position=position,
                position_ids=position.unsqueeze(0))
    return out.logits[:, -1:]


@pytest.mark.parametrize("delegate", [True, False])
def test_patched_decode_matches_the_unpatched_model(delegate):
    model, ids = tiny_model(), None
    ids = _ids(model)
    reference, cache = _prefill(model, ids)
    token = reference.argmax(-1)
    unpatched = _decode_step(model, copy.deepcopy(cache), token, ids.shape[1])

    assert L.install_flash_layer_attn(model, delegate_decode=delegate) is True
    patched = _decode_step(model, cache, token, ids.shape[1])

    assert cache.get_seq_length() == ids.shape[1] + 1
    assert torch.allclose(patched, unpatched, rtol=1e-3, atol=1e-3)
    assert patched.argmax(-1).item() == unpatched.argmax(-1).item()


def test_delegation_survives_consecutive_decodes():
    """Later steps re-enter the patched forward with a growing cache."""
    model = tiny_model()
    ids = _ids(model)
    reference, cache = _prefill(model, ids)
    assert L.install_flash_layer_attn(model, delegate_decode=True) is True
    token = reference.argmax(-1)
    for offset in range(3):
        logits = _decode_step(model, cache, token, ids.shape[1] + offset)
        assert logits.shape == (1, 1, model.config.vocab_size)
        token = logits.argmax(-1)
    assert cache.get_seq_length() == ids.shape[1] + 3
