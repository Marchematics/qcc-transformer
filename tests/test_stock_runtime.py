import copy
import torch

from qcc_transformer.stock_runtime import (
    FullKVGeometry,
    PackedHybridReferenceState,
    expand_gqa,
    packed_ratio_vs_full_kv,
)
from qcc_transformer.vllm_hybrid import HybridQCCVLLMState
from qcc_transformer.vllm_stock import QCCPackedStateLayout, QCCStockVLLMConfig


def test_fullkv_geometry_uses_kv_heads_not_query_heads() -> None:
    cfg = QCCStockVLLMConfig(
        num_heads=16, head_dim=64, window_size=32, num_codes=4,
        exact_num_sets=4, exact_ways=2,
    )
    layout = QCCPackedStateLayout(cfg)
    mha = packed_ratio_vs_full_kv(layout, context_tokens=128_000, num_kv_heads=16)
    gqa = packed_ratio_vs_full_kv(layout, context_tokens=128_000, num_kv_heads=4)
    assert gqa == 4.0 * mha
    assert FullKVGeometry(128_000, 4, 64, 2).bytes == 2 * 128_000 * 4 * 64 * 2


def test_expand_gqa_repeats_each_kv_head() -> None:
    x = torch.tensor([[[[1.0]], [[2.0]]]])
    y = expand_gqa(x, 4)
    torch.testing.assert_close(y[:, :, 0, 0], torch.tensor([[1.0, 1.0, 2.0, 2.0]]))


def test_packed_reference_matches_hybrid_state_across_evictions() -> None:
    torch.manual_seed(41)
    cfg = QCCStockVLLMConfig(
        num_heads=2,
        head_dim=4,
        window_size=4,
        num_codes=3,
        num_scales=4,
        exact_num_sets=3,
        exact_ways=2,
        local_element_bytes=4,
        max_position_embeddings=64,
        archive_mix=0.125,
    )
    reference = HybridQCCVLLMState(
        2,
        4,
        window_size=4,
        num_codes=3,
        max_position_embeddings=64,
        archive_mix=0.125,
        use_triton=False,
        hybrid_kwargs={"exact_num_sets": 3, "exact_ways": 2},
    )
    reference.reset(1, device=torch.device("cpu"), dtype=torch.float32)
    packed = PackedHybridReferenceState(cfg, archive=copy.deepcopy(reference.archive), use_triton=False)
    page = packed.allocate_page(dtype=torch.float32)

    for length in (3, 2, 5, 1):
        q = torch.randn(1, 2, length, 4)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        expected = reference.forward(q, k, v)
        actual = packed.forward(page, q, k, v)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    assert int(reference.seen_tokens) == 11


def test_packed_reference_accepts_gqa_inputs() -> None:
    torch.manual_seed(5)
    cfg = QCCStockVLLMConfig(
        num_heads=4,
        head_dim=2,
        window_size=3,
        num_codes=2,
        exact_num_sets=2,
        exact_ways=1,
        local_element_bytes=4,
        max_position_embeddings=32,
    )
    packed = PackedHybridReferenceState(cfg, use_triton=False)
    page = packed.allocate_page(dtype=torch.float32)
    q = torch.randn(1, 4, 5, 2)
    k = torch.randn(1, 2, 5, 2)
    v = torch.randn_like(k)
    out = packed.forward(page, q, k, v)
    assert out.shape == q.shape
