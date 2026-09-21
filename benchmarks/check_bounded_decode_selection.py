"""CPU self-checks for the bounded-decode selection primitives.

These run without a model or a GPU and assert the invariants the retention law
depends on:

1. the kept set never exceeds the budget;
2. attention sinks and the recent window are always present;
3. block pooling spreads importance to a token's neighbours, so a contiguous
   multi-token needle is selected as a unit rather than in fragments;
4. a bigger budget never loses a token that a smaller budget already kept
   (monotonicity of the top-k family);
5. sorting the kept indices preserves the relative order of the original keys.

Run: python benchmarks/check_bounded_decode_selection.py
"""

from __future__ import annotations

import torch

try:  # repo layout
    import benchmark_bounded_decode_frontier as L
except ImportError:  # flat fallback when benchmarks/ is not on sys.path
    import longctx as L


def check_budget_and_forcing():
    kv, n, budget, nsink, nrecent = 3, 500, 64, 4, 16
    scores = torch.rand(kv, n)
    idx = L.topk_indices(scores, budget, nsink, nrecent, n, pool=1)
    assert idx.shape == (kv, budget), idx.shape
    assert int(idx.max()) < n and int(idx.min()) >= 0
    for head in range(kv):
        kept = set(idx[head].tolist())
        assert set(range(nsink)) <= kept, "attention sinks must be kept"
        assert set(range(n - nrecent, n)) <= kept, "recent window must be kept"
    print("ok: budget respected, sinks and recent window always present")


def check_pooling_protects_neighbours():
    kv, n, budget, pool = 1, 512, 16, 7
    scores = torch.zeros(kv, n)
    needle = 200
    scores[0, needle] = 1.0  # a single high-scoring token
    idx_plain = L.topk_indices(scores, budget, 0, 0, n, pool=1)
    idx_pool = L.topk_indices(scores, budget, 0, 0, n, pool=pool)
    lo = (needle // pool) * pool  # pooling is block-aligned from position 0
    hi = lo + pool
    protected = set(range(lo, hi))
    kept_plain = set(idx_plain[0].tolist())
    kept_pool = set(idx_pool[0].tolist())
    assert needle in kept_plain
    assert protected <= kept_pool, (sorted(protected - kept_pool), sorted(kept_pool))
    print(f"ok: pooling={pool} keeps the full block {lo}..{hi - 1} around a hit")


def check_budget_monotonicity():
    kv, n = 2, 400
    scores = torch.rand(kv, n)
    small = {tuple(sorted(L.topk_indices(scores, 32, 4, 8, n)[h].tolist())) for h in range(kv)}
    # monotonicity holds only when the forced sets agree, which they do here
    big = [set(L.topk_indices(scores, 128, 4, 8, n)[h].tolist()) for h in range(kv)]
    small_sets = [set(L.topk_indices(scores, 32, 4, 8, n)[h].tolist()) for h in range(kv)]
    for a, b in zip(small_sets, big):
        assert a <= b, "a larger budget must be a superset for this score family"
    print("ok: larger budget is a superset of the smaller budget")


def check_gather_pruning_preserves_order():
    B, H, n, d, budget = 1, 2, 64, 8, 32
    keys = torch.randn(B, H, n, d)
    idx = L.topk_indices(torch.rand(H, n), budget, 2, 4, n, pool=1)
    ek = idx.unsqueeze(0).unsqueeze(-1).expand(B, H, idx.shape[1], d)
    pruned = keys.gather(2, ek)
    for h in range(H):
        expect = keys[0, h, idx[h], :]
        assert torch.allclose(pruned[0, h], expect, atol=0)
        assert idx[h].tolist() == sorted(idx[h].tolist()), "kept indices must be ascending"
    print("ok: gather pruning matches direct indexing and keeps ascending order")


def check_ranking_is_only_softmax_denominator_away():
    """For a fixed query the softmax denominator is constant across keys, so
    ranking by raw score equals ranking by attention weight."""
    torch.manual_seed(0)
    q = torch.randn(1, 1, 8)
    k = torch.randn(256, 8)
    raw = (q @ k.T).squeeze()
    w = torch.softmax(raw, dim=-1)
    assert torch.equal(raw.argsort(descending=True), w.argsort(descending=True))
    print("ok: final-query ranking by raw score equals ranking by attention weight")


if __name__ == "__main__":
    torch.manual_seed(1234)
    check_budget_and_forcing()
    check_pooling_protects_neighbours()
    check_budget_monotonicity()
    check_gather_pruning_preserves_order()
    check_ranking_is_only_softmax_denominator_away()
    print("all selection self-checks passed")
