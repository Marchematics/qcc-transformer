import torch

from qcc_transformer.hybrid_archive import HybridQCCArchive
from qcc_transformer.stock_runtime import PackedHybridReferenceState
from qcc_transformer.vllm_stock import (
    QCCPackedStateLayout,
    QCCStockVLLMConfig,
    typed_segment_view,
    validate_layout,
)


def test_packed_decode_batch_matches_independent_page_references() -> None:
    torch.manual_seed(71)
    config = QCCStockVLLMConfig(
        num_heads=2,
        head_dim=4,
        window_size=3,
        num_codes=4,
        exact_num_sets=4,
        exact_ways=2,
        local_element_bytes=4,
    )
    archive = HybridQCCArchive(
        config.num_heads,
        config.head_dim,
        num_codes=config.num_codes,
        decay_rates=config.decay_rates(),
        window_size=config.window_size,
        use_triton=False,
        exact_num_sets=config.exact_num_sets,
        exact_ways=config.exact_ways,
    )
    batched = PackedHybridReferenceState(config, archive=archive, use_triton=False)
    single_a = PackedHybridReferenceState(config, archive=archive, use_triton=False)
    single_b = PackedHybridReferenceState(config, archive=archive, use_triton=False)
    pages = torch.cat(
        [batched.allocate_page(dtype=torch.float32) for _ in range(2)], dim=0
    ).view(2, -1)
    page_a = single_a.allocate_page(dtype=torch.float32)
    page_b = single_b.allocate_page(dtype=torch.float32)

    for step in range(8):
        query = torch.randn(2, config.num_heads, config.head_dim)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        actual = batched.forward_decode_batch(
            pages,
            query,
            key,
            value,
            torch.full((2,), step, dtype=torch.long),
        )
        expected_a = single_a.forward(
            page_a,
            query[:1].unsqueeze(2),
            key[:1].unsqueeze(2),
            value[:1].unsqueeze(2),
        ).squeeze(2)
        expected_b = single_b.forward(
            page_b,
            query[1:].unsqueeze(2),
            key[1:].unsqueeze(2),
            value[1:].unsqueeze(2),
        ).squeeze(2)
        expected = torch.cat((expected_a, expected_b), dim=0)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(pages[0], page_a, atol=0, rtol=0)
        torch.testing.assert_close(pages[1], page_b, atol=0, rtol=0)


def test_packed_decode_batch_handles_gqa_and_independent_progress() -> None:
    torch.manual_seed(72)
    config = QCCStockVLLMConfig(
        num_heads=4,
        head_dim=4,
        window_size=3,
        num_codes=4,
        exact_num_sets=4,
        exact_ways=2,
        local_element_bytes=4,
    )
    archive = HybridQCCArchive(
        config.num_heads,
        config.head_dim,
        num_codes=config.num_codes,
        decay_rates=config.decay_rates(),
        window_size=config.window_size,
        use_triton=False,
        exact_num_sets=config.exact_num_sets,
        exact_ways=config.exact_ways,
    )
    batched = PackedHybridReferenceState(config, archive=archive, use_triton=False)
    single_a = PackedHybridReferenceState(config, archive=archive, use_triton=False)
    single_b = PackedHybridReferenceState(config, archive=archive, use_triton=False)
    pages = torch.cat(
        [batched.allocate_page(dtype=torch.float32) for _ in range(2)], dim=0
    ).view(2, -1)
    page_a = single_a.allocate_page(dtype=torch.float32)
    page_b = single_b.allocate_page(dtype=torch.float32)

    def compare_one(
        page_slice: torch.Tensor,
        state,
        page: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position: int,
    ) -> None:
        actual = batched.forward_decode_batch(
            page_slice,
            query,
            key,
            value,
            torch.tensor([position], dtype=torch.long),
        )
        expected = state.forward(
            page,
            query.unsqueeze(2),
            key.unsqueeze(2),
            value.unsqueeze(2),
        ).squeeze(2)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(page_slice[0], page, atol=0, rtol=0)

    query = torch.randn(2, config.num_heads, config.head_dim)
    key = torch.randn(2, 2, config.head_dim)
    value = torch.randn_like(key)
    actual = batched.forward_decode_batch(
        pages, query, key, value, torch.zeros(2, dtype=torch.long)
    )
    expected = torch.cat(
        (
            single_a.forward(
                page_a, query[:1].unsqueeze(2), key[:1].unsqueeze(2), value[:1].unsqueeze(2)
            ).squeeze(2),
            single_b.forward(
                page_b, query[1:].unsqueeze(2), key[1:].unsqueeze(2), value[1:].unsqueeze(2)
            ).squeeze(2),
        ),
        dim=0,
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(pages[0], page_a, atol=0, rtol=0)
    torch.testing.assert_close(pages[1], page_b, atol=0, rtol=0)

    for position, index, state, page in (
        (1, 0, single_a, page_a),
        (1, 1, single_b, page_b),
        (2, 0, single_a, page_a),
    ):
        query = torch.randn(1, config.num_heads, config.head_dim)
        key = torch.randn(1, 2, config.head_dim)
        value = torch.randn_like(key)
        compare_one(pages[index : index + 1], state, page, query, key, value, position)

    for positions in ((3, 2), (4, 3)):
        query = torch.randn(2, config.num_heads, config.head_dim)
        key = torch.randn(2, 2, config.head_dim)
        value = torch.randn_like(key)
        actual = batched.forward_decode_batch(
            pages, query, key, value, torch.tensor(positions, dtype=torch.long)
        )
        expected = torch.cat(
            (
                single_a.forward(
                    page_a, query[:1].unsqueeze(2), key[:1].unsqueeze(2), value[:1].unsqueeze(2)
                ).squeeze(2),
                single_b.forward(
                    page_b, query[1:].unsqueeze(2), key[1:].unsqueeze(2), value[1:].unsqueeze(2)
                ).squeeze(2),
            ),
            dim=0,
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(pages[0], page_a, atol=0, rtol=0)
        torch.testing.assert_close(pages[1], page_b, atol=0, rtol=0)


def test_packed_layout_segments_do_not_overlap_and_are_aligned() -> None:
    layout = QCCPackedStateLayout(
        QCCStockVLLMConfig(num_heads=8, head_dim=16, window_size=32)
    )
    validate_layout(layout)
    previous_end = 0
    for segment in layout.segments:
        assert segment.offset >= previous_end
        assert segment.offset % layout.config.alignment == 0
        previous_end = segment.end
    assert layout.total_bytes >= previous_end


def test_packed_state_bytes_are_context_independent() -> None:
    layout = QCCPackedStateLayout(
        QCCStockVLLMConfig(
            num_heads=16,
            head_dim=64,
            window_size=128,
            num_codes=16,
            exact_num_sets=32,
            exact_ways=4,
        )
    )
    assert layout.mutable_state_bytes() == layout.mutable_state_bytes()
    ratio_128k = layout.compression_ratio_vs_full_kv(128_000)
    ratio_1m = layout.compression_ratio_vs_full_kv(1_000_000)
    assert ratio_1m < ratio_128k
    assert ratio_1m < 0.10


def test_packed_state_ratio_uses_kv_heads_for_gqa() -> None:
    layout = QCCPackedStateLayout(
        QCCStockVLLMConfig(
            num_heads=16,
            head_dim=64,
            window_size=128,
            num_codes=16,
            exact_num_sets=32,
            exact_ways=4,
        )
    )
    mha = layout.compression_ratio_vs_full_kv(128_000)
    gqa = layout.compression_ratio_vs_full_kv(128_000, num_kv_heads=4)
    assert gqa == 4.0 * mha


def test_packed_layout_round_trips_config_json() -> None:
    config = QCCStockVLLMConfig(
        num_heads=32,
        head_dim=128,
        window_size=64,
        num_codes=24,
        num_scales=3,
        exact_num_sets=48,
        exact_ways=3,
    )
    restored = QCCStockVLLMConfig.from_json(config.to_json())
    assert restored == config
    assert QCCPackedStateLayout(restored).manifest()["config"]["num_heads"] == 32


def test_typed_views_share_raw_page_storage() -> None:
    layout = QCCPackedStateLayout(
        QCCStockVLLMConfig(
            num_heads=2,
            head_dim=4,
            window_size=3,
            num_codes=2,
            num_scales=2,
            exact_num_sets=2,
            exact_ways=2,
        )
    )
    # Simulate a scheduler page whose public dtype is BF16. The packed state uses
    # raw-byte reinterpretation for FP32 accumulators and int64 counters.
    page = torch.zeros(layout.words_for_dtype(torch.bfloat16), dtype=torch.bfloat16)
    numerator = typed_segment_view(page, layout, "recurrent_numerator")
    counters = typed_segment_view(page, layout, "counters")
    scores = typed_segment_view(page, layout, "exact_scores")
    numerator.fill_(1.25)
    counters.copy_(torch.tensor([1, 2, 3, 4], dtype=torch.int64))
    scores.fill_(-7.0)
    torch.testing.assert_close(
        typed_segment_view(page, layout, "recurrent_numerator"), numerator
    )
    torch.testing.assert_close(
        typed_segment_view(page, layout, "counters"), counters
    )
    assert page.view(torch.uint8).abs().sum() > 0


def test_fullkv_geometry_reduction_exceeds_80_percent_at_128k_default_phi_shape() -> None:
    # Phi-4-mini uses 32 query heads x 96 head dim. This geometry check is not a
    # peak-memory claim; it only enforces that the packed attention state is small
    # enough to make the production >=80% memory gate physically plausible.
    layout = QCCPackedStateLayout(
        QCCStockVLLMConfig(
            num_heads=32,
            head_dim=96,
            window_size=128,
            num_codes=16,
            num_scales=4,
            exact_num_sets=32,
            exact_ways=4,
            local_element_bytes=2,
        )
    )
    ratio = layout.compression_ratio_vs_full_kv(128_000)
    assert ratio <= 0.20
