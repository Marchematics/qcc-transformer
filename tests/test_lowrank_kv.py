"""CPU tests for the low-rank latent KV benchmark.

The byte accounting is the whole point of that comparison — a rank-`r` latent keeps
*every* token at `r` coordinates, so its stored bytes are `L x r`, not `r x head_dim`.
The tests pin that identity against the bounded arm's `slots x head_dim`, check that
a full-rank compression is exact, and check that the reconstruction error falls as
the rank rises.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# the benchmark modules are written to run as scripts and fall back to a flat
# import; put benchmarks/ on the path so the package-style import wins
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from benchmarks.benchmark_bounded_decode_frontier import prefill_capture
from benchmarks.benchmark_lowrank_kv_ruler import compress_low_rank, stored_bytes

from test_required_set import tiny_model


def _cache(length=48):
    model = tiny_model()
    torch.manual_seed(0)
    ids = torch.randint(1, model.config.vocab_size, (1, length))
    cache, _logits, _tails = prefill_capture(model, ids, obs=8, chunk=16)
    return model, ids, cache


def test_stored_bytes_counts_k_and_v_per_layer_and_head():
    # 2 slots x 4 dims x 3 layers x 2 heads x 2 bytes x (K and V)
    assert stored_bytes(2, 4, 3, 2) == 2 * 4 * 3 * 2 * 2 * 2


def test_byte_matched_rank_is_the_slots_times_dim_over_length_identity():
    length, head_dim, layers, heads, slots = 48, 8, 3, 2, 12
    matched = max(1, round(slots * head_dim / length))
    assert matched == 2
    # the latent arm stores L x rank, the bounded arm stores slots x head_dim
    assert stored_bytes(length, matched, layers, heads) == stored_bytes(
        slots, head_dim, layers, heads)


def test_full_rank_is_exact_and_lower_ranks_are_approximations():
    _model, _ids, cache = _cache()
    head_dim = cache.layers[0].keys.shape[-1]
    reference = cache.layers[0].keys[0].clone().float()

    compress_low_rank(cache, head_dim)
    assert torch.allclose(cache.layers[0].keys[0].float(), reference, atol=1e-3)

    errors = []
    for rank in (1, max(1, head_dim // 2)):
        fresh = cache.layers[0].keys[0].clone()
        cache.layers[0].keys[0].copy_(reference.to(cache.layers[0].keys.dtype))
        compress_low_rank(cache, rank)
        errors.append(float((cache.layers[0].keys[0].float() - reference)
                            .pow(2).mean()))
        cache.layers[0].keys[0].copy_(fresh)
    assert errors[0] > errors[1] > 0
    assert compress_low_rank(cache, 2)[1] == 2
