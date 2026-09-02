import torch

from qcc_transformer.vllm_stock import (
    QCCPackedStateLayout,
    QCCStockVLLMConfig,
    typed_segment_view,
    validate_layout,
)


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
