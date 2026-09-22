"""CPU tests for the accumulated-attention prefill behind the H2O/TOVA baselines.

``prefill_accumulate`` has to do two jobs at once: build exactly the cache
``prefill_capture`` builds, and report the attention statistics the H2O and TOVA
policies rank keys by.  The tests pin both — bit-identical cache and logits, and
the two conservation laws that make the statistics interpretable (every query's
attention sums to one over its valid keys, and every query casts exactly one
top-1 vote).
"""

from __future__ import annotations

import torch

from benchmarks.benchmark_bounded_decode_frontier import prefill_accumulate, prefill_capture

from test_required_set import tiny_model


def _ids(model, length=96):
    torch.manual_seed(0)
    return torch.randint(1, model.config.vocab_size, (1, length))


def test_cache_and_logits_match_the_plain_prefill():
    model = tiny_model()
    ids = _ids(model)
    cache_a, logits_a, tails_a = prefill_capture(model, ids, obs=16, chunk=32)
    cache_b, logits_b, tails_b, mass, top1 = prefill_accumulate(model, ids, obs=16, chunk=32,
                                                               key_chunk=32)
    assert torch.equal(logits_a, logits_b)
    for layer_a, layer_b in zip(cache_a.layers, cache_b.layers):
        assert torch.equal(layer_a.keys, layer_b.keys)
        assert torch.equal(layer_a.values, layer_b.values)
    for idx in tails_a:
        assert torch.equal(tails_a[idx], tails_b[idx])       # same observation window

    length = ids.shape[1]
    kv_heads = model.config.num_key_value_heads
    group = model.config.num_attention_heads // kv_heads
    assert len(mass) == len(top1) == len(cache_a.layers)
    for layer_mass in mass:
        # every (kv-head, query-head, query) row is a distribution over its valid
        # keys, so the whole layer's mass is exactly the number of such rows
        assert layer_mass.shape == (kv_heads, length)
        assert abs(layer_mass.sum().item() - kv_heads * group * length) < 1e-3
        assert bool((layer_mass >= 0).all())
    for layer_top1 in top1:
        assert layer_top1.shape == (kv_heads, length)
        # every (kv-head, query-head, query) casts exactly one vote
        assert layer_top1.sum().item() == kv_heads * group * length


def test_accumulation_is_causal_and_chunking_does_not_change_it():
    model = tiny_model()
    ids = _ids(model, length=64)
    _c1, _l1, _t1, mass_one, top1_one = prefill_accumulate(model, ids, obs=8, chunk=64,
                                                           key_chunk=64)
    _c2, _l2, _t2, mass_many, top1_many = prefill_accumulate(model, ids, obs=8, chunk=16,
                                                            key_chunk=8)
    for a, b in zip(mass_one, mass_many):
        assert torch.allclose(a, b, atol=1e-4)
    for a, b in zip(top1_one, top1_many):
        assert torch.equal(a, b)

    kv_heads = model.config.num_key_value_heads
    group = model.config.num_attention_heads // kv_heads
    # with a single-token prompt the only key takes every vote
    _c3, _l3, _t3, mass_single, top1_single = prefill_accumulate(model, ids[:, :1], obs=1,
                                                                 chunk=8, key_chunk=8)
    assert torch.allclose(mass_single[0].sum(), torch.tensor(float(kv_heads * group)))
    assert top1_single[0].sum().item() == kv_heads * group


def test_quest_keeps_whole_blocks_and_pyramid_varies_the_layer_budget():
    """The two new selection policies, checked on their defining properties."""
    from benchmarks.benchmark_bounded_decode_frontier import build_idxs

    torch.manual_seed(0)
    length, kv_heads, budget = 640, 2, 160
    scores = [torch.randn(kv_heads, length) for _ in range(4)]

    idxs = build_idxs("quest", scores, None, None, budget, 4, 40, length, 1, "cpu", 0, 0)
    assert len(idxs) == 4
    for layer_idx in idxs:
        assert layer_idx.shape[0] == kv_heads
        # whole 16-token blocks: every kept index implies its block is fully kept
        for row in layer_idx.tolist():
            assert len(row) <= budget + 16
            blocks = {p // 16 for p in row}
            for block in blocks:
                span = set(range(block * 16, min(length, block * 16 + 16)))
                assert span <= set(row)

    idxs = build_idxs("pyramid", scores, None, None, budget, 4, 40, length, 1, "cpu", 0, 0)
    widths = [int(idx.shape[1]) for idx in idxs]
    assert widths == sorted(widths, reverse=True)          # lower layers keep more
    assert abs(sum(widths) / len(widths) - budget) <= 1     # mean width preserved
    assert widths[0] > budget > widths[-1]


def test_maskless_prefill_matches_the_masked_path():
    """`attention_mask=False` is an optimisation, not a different computation.

    The maskless path exists because an explicit mask of length `end` is
    materialised per chunk and forces a non-flash kernel at long context; the two
    must agree exactly or the long-context rows would be measured on a different
    function.
    """
    from benchmarks.benchmark_bounded_decode_frontier import prefill_capture

    model = tiny_model()
    ids = _ids(model, length=48)
    cache_a, logits_a, tails_a = prefill_capture(model, ids, obs=8, chunk=16,
                                                 attention_mask=True)
    cache_b, logits_b, tails_b = prefill_capture(model, ids, obs=8, chunk=16,
                                                 attention_mask=False)
    assert torch.equal(logits_a, logits_b)
    for layer_a, layer_b in zip(cache_a.layers, cache_b.layers):
        assert torch.equal(layer_a.keys, layer_b.keys)
        assert torch.equal(layer_a.values, layer_b.values)
    assert all(torch.equal(tails_a[i], tails_b[i]) for i in tails_a)


def test_flash_sdpa_patch_installs_and_only_changes_the_chunked_case():
    """The dispatch patch must touch exactly the cached-chunked-prefill case."""
    from benchmarks.benchmark_bounded_decode_frontier import install_flash_sdpa
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    assert install_flash_sdpa() is True
    patched = ALL_ATTENTION_FUNCTIONS["sdpa"]

    seen = {}

    def fake_original(module, query, key, value, attention_mask, dropout=0.0,
                      scaling=None, is_causal=None, **kwargs):
        seen["is_causal"] = is_causal
        seen["mask"] = attention_mask
        return torch.zeros_like(query), None

    import transformers.integrations.sdpa_attention as sdpa_attention
    original = sdpa_attention.sdpa_attention_forward
    sdpa_attention.sdpa_attention_forward = fake_original
    try:
        # chunked prefill: several queries, no mask -> is_causal must be forced on
        patched(None, torch.zeros(1, 2, 8, 4), torch.zeros(1, 2, 32, 4),
                torch.zeros(1, 2, 32, 4), None)
        assert seen["is_causal"] is True and seen["mask"] is None
        # decode: a single query keeps whatever Hugging Face passed (False), i.e.
        # the wrapper does not force causality onto the decode step
        patched(None, torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 32, 4),
                torch.zeros(1, 2, 32, 4), None, is_causal=False)
        assert seen["is_causal"] is False
        # an explicit mask is never overridden
        mask = torch.zeros(1, 1, 8, 32, dtype=torch.bool)
        patched(None, torch.zeros(1, 2, 8, 4), torch.zeros(1, 2, 32, 4),
                torch.zeros(1, 2, 32, 4), mask)
        assert seen["mask"] is mask
    finally:
        sdpa_attention.sdpa_attention_forward = original
