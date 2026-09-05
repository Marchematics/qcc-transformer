import torch
import pytest
from unittest.mock import patch

from qcc_transformer import QCCArchive, QCCForCausalLM, QCCSelfAttention
from qcc_transformer.triton_kernels import (
    TRITON_AVAILABLE,
    triton_lazy_update_archive,
    triton_local_chunk_attention,
    triton_read_archive,
    triton_sparse_read_archive,
    triton_sparse_update_read_archive_chunk,
    triton_update_archive,
    triton_update_read_archive_chunk,
)


def test_local_attention_promotes_large_qk_logits_before_softmax() -> None:
    """Large projected activations must not turn the fp16 softmax into NaN."""

    torch.manual_seed(122)
    attention = QCCSelfAttention(
        d_model=16,
        num_heads=4,
        window_size=8,
        num_codes=2,
        use_triton=False,
    )
    with torch.no_grad():
        attention.q_proj.weight.mul_(100.0)
        attention.k_proj.weight.mul_(100.0)
    hidden = torch.randn(1, 4, 16)
    output = attention(hidden, reset_state=True)
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not TRITON_AVAILABLE or not torch.cuda.is_available(), reason="Triton CUDA unavailable")
def test_triton_local_chunk_matches_unfolded_reference() -> None:
    torch.manual_seed(123)
    device = torch.device("cuda")
    batch, heads, old_length, query_length, dim, window = 1, 2, 7, 13, 8, 8
    query = torch.randn(batch, heads, query_length, dim, device=device)
    keys = torch.randn(batch, heads, old_length + query_length, dim, device=device)
    values = torch.randn_like(keys)
    attention = QCCSelfAttention(
        d_model=heads * dim,
        num_heads=heads,
        window_size=window,
        use_triton=False,
    ).to(device)
    expected = attention._local_window_attention(
        query, keys, values, old_length=old_length
    )
    actual = triton_local_chunk_attention(
        query, keys, values, old_length=old_length, window_size=window
    )
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not TRITON_AVAILABLE or not torch.cuda.is_available(), reason="Triton CUDA unavailable")
def test_decode_chunk_triton_local_dispatch_matches_reference() -> None:
    torch.manual_seed(124)
    device = torch.device("cuda")
    kwargs = dict(
        vocab_size=41,
        d_model=32,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=256,
        window_size=8,
        num_codes=8,
        archive_scan_block_size=16,
    )
    triton_model = QCCForCausalLM(**kwargs, use_triton=True).to(device).eval()
    reference_model = QCCForCausalLM(**kwargs, use_triton=False).to(device).eval()
    reference_model.load_state_dict(triton_model.state_dict())
    tokens = torch.randint(0, kwargs["vocab_size"], (1, 97), device=device)
    with torch.no_grad():
        actual = torch.cat(
            [
                triton_model.decode_chunk(tokens[:, :16], reset_cache=True),
                triton_model.decode_chunk(tokens[:, 16:64]),
                triton_model.decode_chunk(tokens[:, 64:]),
            ],
            dim=1,
        )
        expected = torch.cat(
            [
                reference_model.decode_chunk(tokens[:, :16], reset_cache=True),
                reference_model.decode_chunk(tokens[:, 16:64]),
                reference_model.decode_chunk(tokens[:, 64:]),
            ],
            dim=1,
        )
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


def test_archive_matches_exponential_response_for_single_code() -> None:
    torch.manual_seed(0)
    archive = QCCArchive(
        num_heads=1,
        head_dim=2,
        num_codes=1,
        decay_rates=(0.8,),
        window_size=2,
    )
    with torch.no_grad():
        archive.codes.fill_(0.0)
    keys = torch.randn(5, 1, 2)
    values = torch.randn(5, 1, 2)
    archive.reset_state(1)
    for key, value in zip(keys[:3], values[:3]):
        archive.update(key.unsqueeze(0), value.unsqueeze(0))
    # The first three tokens are already in the archive; the last two are
    # represented by the local window in the streaming attention module.
    out = archive.read(torch.zeros(1, 1, 2))[0, 0]
    weights = torch.tensor([0.8**4, 0.8**3, 0.8**2])
    expected = (weights[:, None] * values[:3, 0]).sum(0) / weights.sum()
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_query_conditioned_archive_residual_is_identity_until_calibrated() -> None:
    torch.manual_seed(17)
    base = QCCArchive(
        num_heads=2,
        head_dim=4,
        num_codes=4,
        decay_rates=(0.8, 0.95),
        window_size=2,
        use_triton=False,
        query_correction_rank=0,
    )
    corrected = QCCArchive(
        num_heads=2,
        head_dim=4,
        num_codes=4,
        decay_rates=(0.8, 0.95),
        window_size=2,
        use_triton=False,
        query_correction_rank=3,
    )
    corrected.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        corrected.query_correction_u.normal_()
        corrected.query_correction_v.zero_()
    keys = torch.randn(1, 2, 5, 4)
    values = torch.randn_like(keys)
    query = torch.randn(1, 2, 4)
    for index in range(keys.shape[2]):
        base.update(keys[:, :, index], values[:, :, index])
        corrected.update(keys[:, :, index], values[:, :, index])
    torch.testing.assert_close(base.read(query), corrected.read(query))
    with torch.no_grad():
        corrected.query_correction_v.normal_(std=0.1)
    assert not torch.equal(base.read(query), corrected.read(query))


def test_query_conditioned_scale_mixture_is_zero_initialized_and_trainable() -> None:
    torch.manual_seed(18)
    base = QCCArchive(
        num_heads=2,
        head_dim=4,
        num_codes=4,
        decay_rates=(0.8, 0.95),
        window_size=2,
        use_triton=False,
        query_correction_rank=0,
    )
    calibrated = QCCArchive(
        num_heads=2,
        head_dim=4,
        num_codes=4,
        decay_rates=(0.8, 0.95),
        window_size=2,
        use_triton=False,
        query_correction_rank=3,
    )
    calibrated.load_state_dict(base.state_dict(), strict=False)
    assert torch.count_nonzero(calibrated.query_correction_u) == 0
    assert torch.count_nonzero(calibrated.query_correction_v) > 0
    keys = torch.randn(1, 2, 6, 4)
    values = torch.randn_like(keys)
    query = torch.randn(1, 2, 4)
    with torch.no_grad():
        for index in range(keys.shape[2]):
            base.update(keys[:, :, index], values[:, :, index])
            calibrated.update(keys[:, :, index], values[:, :, index])
        reference = base.read(query)
        initial = calibrated.read(query)
    torch.testing.assert_close(initial, reference)

    calibrated.query_scale_logits.data.normal_(std=0.2)
    calibrated.query_scale_logits.requires_grad_(True)
    output = calibrated.read(query)
    assert not torch.equal(output, reference)
    output.square().mean().backward()
    assert calibrated.query_scale_logits.grad is not None
    assert torch.isfinite(calibrated.query_scale_logits.grad).all()


def test_archive_global_normalization_matches_separable_softmax_equation():
    """Code routing must weight raw numerator/denominator mass jointly."""

    archive = QCCArchive(
        num_heads=1,
        head_dim=2,
        num_codes=2,
        decay_rates=(0.9,),
        window_size=1,
        use_triton=False,
        global_normalization=True,
    )
    with torch.no_grad():
        archive.codes.copy_(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
        archive.mix_logits.zero_()
    keys = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    values = torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])
    query = torch.tensor([[[1.0, 1.0]]])
    with torch.no_grad():
        archive.reset_state(1)
        archive.update(keys[0:1], values[0:1])
        archive.update(keys[1:2], values[1:2])
        actual = archive.read(query)[0, 0]
        code_scores = torch.tensor([1.0, 1.0]) / (2.0**0.5)
        routing = code_scores.exp()
        # Each code sees the two events through its own exponential response;
        # the global form combines those masses before one final division.
        numerator = archive._numerator[0, 0, :, 0, :]
        denominator = archive._denominator[0, 0, :, 0]
        expected = (routing[:, None] * numerator).sum(0) / (routing * denominator).sum()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_persistent_landmark_survives_long_filler_updates() -> None:
    archive = QCCArchive(
        num_heads=1,
        head_dim=2,
        num_codes=1,
        decay_rates=(0.8,),
        window_size=2,
        use_triton=False,
        persistent_landmark=True,
        content_threshold=0.5,
    )
    with torch.no_grad():
        archive.codes.fill_(0.0)
        archive.codes[..., 0] = 1.0
        archive.landmark_mix_logits.fill_(20.0)
    key = torch.tensor([[[1.0, 0.0]]])
    value = torch.tensor([[[3.0, 4.0]]])
    archive.update(key, value)
    filler_key = torch.tensor([[[-1.0, 0.0]]])
    filler_value = torch.tensor([[[-7.0, -8.0]]])
    for _ in range(100):
        archive.update(filler_key, filler_value)
    out = archive.read(key)[0, 0]
    torch.testing.assert_close(out, value[0, 0], rtol=1e-5, atol=1e-5)


def test_chunked_landmark_reads_are_causal() -> None:
    kwargs = dict(
        num_heads=1,
        head_dim=2,
        num_codes=1,
        decay_rates=(0.8,),
        window_size=1,
        use_triton=False,
        persistent_landmark=True,
    )
    chunked = QCCArchive(**kwargs)
    tokenwise = QCCArchive(**kwargs)
    with torch.no_grad():
        chunked.codes.copy_(torch.tensor([[[1.0, 0.0]]]))
        tokenwise.load_state_dict(chunked.state_dict())
        chunked.landmark_mix_logits.fill_(20.0)
        tokenwise.landmark_mix_logits.fill_(20.0)
        keys = torch.tensor(
            [[[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]]
        )
        values = torch.tensor(
            [[[[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]]]
        )
        queries = keys.clone()
        actual = chunked.update_read_chunk(keys, values, queries)
        expected = []
        for index in range(keys.shape[2]):
            tokenwise.update(keys[:, :, index], values[:, :, index])
            expected.append(tokenwise.read(queries[:, :, index]))
        expected = torch.stack(expected, dim=2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        chunked._landmark_value, tokenwise._landmark_value, rtol=0, atol=0
    )


def test_prefix_landmark_keeps_first_slots_and_uses_key_routing() -> None:
    archive = QCCArchive(
        num_heads=1,
        head_dim=2,
        num_codes=2,
        decay_rates=(0.8,),
        window_size=2,
        use_triton=False,
        persistent_landmark=True,
        prefix_landmark=True,
    )
    keys = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]])
    values = torch.tensor([[[[2.0, 0.0], [0.0, 3.0], [9.0, 9.0]]]])
    queries = keys.clone()
    with torch.no_grad():
        archive.update_read_chunk(keys, values, queries)
        read = archive._landmark_read(keys[:, :, 0])[0]
    assert archive._landmark_count == 2
    # Direct key routing should prefer the first retained key for this query;
    # the softmax remains intentionally non-one-hot.
    assert float(read[0, 0, 0]) > float(read[0, 0, 1])


def test_prefix_pair_landmark_binds_key_to_successor_value() -> None:
    archive = QCCArchive(
        num_heads=1,
        head_dim=2,
        num_codes=2,
        decay_rates=(0.8,),
        window_size=1,
        use_triton=False,
        persistent_landmark=True,
        prefix_landmark=True,
        prefix_pair_landmark=True,
    )
    keys = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]])
    values = torch.tensor([[[[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]]])
    archive.update(keys[:, :, 0], values[:, :, 0])
    archive.update(keys[:, :, 1], values[:, :, 1] + 100.0)
    # Slot zero retains the first key but now carries the second event value.
    torch.testing.assert_close(
        archive._landmark_key[0, 0, 0], keys[0, 0, 0], rtol=0, atol=0
    )
    torch.testing.assert_close(
        archive._landmark_value[0, 0, 0], values[0, 0, 1] + 100.0, rtol=0, atol=0
    )
    read = archive._landmark_read(keys[:, :, 0])[0]
    assert float(read[0, 0, 0]) > float(read[0, 0, 1])


def test_model_forward_shapes_and_gradients() -> None:
    torch.manual_seed(1)
    model = QCCForCausalLM(
        vocab_size=31,
        d_model=32,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=64,
        window_size=4,
        num_codes=4,
    )
    tokens = torch.randint(0, 31, (2, 12))
    logits = model(tokens)
    assert logits.shape == (2, 12, 31)
    logits[:, -1].mean().backward()
    assert model.layers[0].attention.archive.codes.grad is not None


def test_fused_qkv_gate_projection_matches_legacy_modules() -> None:
    torch.manual_seed(11)
    attention = QCCSelfAttention(d_model=16, num_heads=4, use_triton=False)
    hidden = torch.randn(2, 3, 16)
    q, k, v, gate = attention._project_qkv_gate(hidden)
    torch.testing.assert_close(q, attention.q_proj(hidden))
    torch.testing.assert_close(k, attention.k_proj(hidden))
    torch.testing.assert_close(v, attention.v_proj(hidden))
    torch.testing.assert_close(gate, attention.gate(hidden))


def test_sinusoidal_positions_support_million_token_limits_without_table() -> None:
    model = QCCForCausalLM(
        vocab_size=11,
        d_model=16,
        num_layers=1,
        num_heads=4,
        max_position_embeddings=4_000_000,
        window_size=8,
    )
    positions = torch.tensor([[0, 128_000, 1_000_000, 3_999_999]], dtype=torch.long)
    encoding = model.position_embedding(positions)
    assert encoding.shape == (1, 4, 16)
    assert torch.isfinite(encoding).all()
    # The default stateless encoding must not allocate a parameter row per
    # supported position; the learned option remains available explicitly.
    assert not hasattr(model.position_embedding, "weight")
    learned = QCCForCausalLM(
        vocab_size=11,
        d_model=16,
        num_layers=1,
        num_heads=4,
        max_position_embeddings=32,
        position_encoding="learned",
    )
    assert hasattr(learned.position_embedding, "weight")


def test_block_streaming_matches_single_scan_across_block_boundary() -> None:
    torch.manual_seed(42)
    kwargs = dict(
        num_heads=2,
        head_dim=8,
        num_codes=16,
        decay_rates=(0.9, 0.97),
        window_size=7,
        use_triton=False,
    )
    block = QCCArchive(**kwargs, scan_block_size=256)
    single = QCCArchive(**kwargs, scan_block_size=2048)
    single.load_state_dict(block.state_dict())
    keys = torch.randn(1, 2, 513, 8)
    values = torch.randn_like(keys)
    queries = torch.randn_like(keys)
    with torch.no_grad():
        block_out = block.update_read_chunk(keys, values, queries)
        single_out = single.update_read_chunk(keys, values, queries)
    torch.testing.assert_close(block_out, single_out, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(block.state.numerator, single.state.numerator, rtol=2e-5, atol=2e-5)


def test_chunk_output_view_is_reused_by_reference_path() -> None:
    torch.manual_seed(421)
    archive = QCCArchive(
        num_heads=2,
        head_dim=8,
        num_codes=8,
        decay_rates=(0.9, 0.97),
        window_size=3,
        use_triton=False,
        scan_block_size=5,
    )
    keys = torch.randn(1, 2, 13, 8)
    values = torch.randn_like(keys)
    queries = torch.randn_like(keys)
    with torch.no_grad():
        expected = archive.update_read_chunk(keys, values, queries)
        archive.reset_state(1)
        backing = torch.empty(1, 2, 17, 8)
        target = backing[:, :, 2:15]
        actual = archive.update_read_chunk(keys, values, queries, output=target)
    assert actual.data_ptr() == target.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_rope_position_mode_matches_streaming_decode() -> None:
    torch.manual_seed(43)
    model = QCCForCausalLM(
        vocab_size=19,
        d_model=24,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=4_000_000,
        window_size=3,
        num_codes=4,
        position_encoding="rope",
    ).eval()
    tokens = torch.randint(0, 19, (1, 17))
    with torch.no_grad():
        sequence = model(tokens)
        model.reset_cache(1)
        streamed = torch.stack(
            [model.decode_step(tokens[:, index]) for index in range(tokens.shape[1])], dim=1
        )
        model.reset_cache(1)
        chunked = torch.cat(
            [
                model.decode_chunk(tokens[:, :6], reset_cache=True),
                model.decode_chunk(tokens[:, 6:]),
            ],
            dim=1,
        )
    torch.testing.assert_close(streamed, sequence, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(chunked, sequence, rtol=3e-4, atol=3e-4)


def test_position_invariant_archive_rope_matches_streaming_decode() -> None:
    """Raw archive addressing must preserve the exact serving recurrence.

    Local attention still uses rotary Q/K, while the archive receives the
    unrotated projections.  This specifically exercises a wrapped ring and
    catches accidental mixing of rotary and raw key layouts.
    """

    torch.manual_seed(291)
    model = QCCForCausalLM(
        vocab_size=37,
        d_model=24,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=512,
        window_size=5,
        num_codes=6,
        position_encoding="rope",
        archive_position_invariant=True,
        use_triton=False,
    ).eval()
    tokens = torch.randint(0, 37, (2, 23))
    with torch.no_grad():
        sequence = model(tokens)
        model.reset_cache(tokens.shape[0])
        streamed = torch.stack(
            [model.decode_step(tokens[:, t]) for t in range(tokens.shape[1])], dim=1
        )
        model.reset_cache(tokens.shape[0])
        chunked = torch.cat(
            [model.decode_chunk(tokens[:, :8], reset_cache=True), model.decode_chunk(tokens[:, 8:])],
            dim=1,
        )
    torch.testing.assert_close(streamed, sequence, rtol=3e-4, atol=3e-4)
    for layer in model.layers:
        attention = layer.attention
        assert attention.archive_position_invariant
        assert attention._archive_key_cache is not None
        assert attention._archive_key_cache.shape[2] == attention.window_size
    torch.testing.assert_close(chunked, sequence, rtol=3e-4, atol=3e-4)


def test_model_decay_schedule_reaches_configured_context_horizon() -> None:
    model = QCCForCausalLM(
        vocab_size=17,
        d_model=32,
        num_layers=1,
        num_heads=4,
        max_position_embeddings=1_000_000,
        window_size=32,
        num_codes=8,
    )
    rates = model.layers[0].attention.archive.decay_rates
    # The slowest scale is parameterized to have an approximately one-million
    # token half-life rather than underflowing after a few thousand updates.
    # Rates are persisted in fp32 for kernel compatibility.  Near one, the
    # representable spacing shifts the effective half-life by a few percent.
    assert float(rates[-1].pow(1_000_000)) == pytest.approx(0.5, abs=0.02)
    assert float(rates[0].pow(32)) == pytest.approx(0.5, abs=0.01)


def test_first_evicted_token_enters_archive_at_window_boundary() -> None:
    torch.manual_seed(2)
    model = QCCForCausalLM(
        vocab_size=19,
        d_model=16,
        num_layers=1,
        num_heads=4,
        window_size=3,
        num_codes=4,
    )
    model(torch.randint(0, 19, (1, 3)))
    empty = model.layers[0].attention.archive.state.denominator
    assert torch.count_nonzero(empty) == 0
    model(torch.randint(0, 19, (1, 4)))
    populated = model.layers[0].attention.archive.state.denominator
    assert torch.count_nonzero(populated) > 0


def test_vectorized_inference_matches_reference_path() -> None:
    torch.manual_seed(3)
    model = QCCForCausalLM(
        vocab_size=23,
        d_model=24,
        num_layers=1,
        num_heads=4,
        window_size=3,
        num_codes=4,
    )
    tokens = torch.randint(0, 23, (2, 9))
    model.train()
    with torch.enable_grad():
        reference = model(tokens)
    model.eval()
    with torch.no_grad():
        optimized = model(tokens)
    torch.testing.assert_close(optimized, reference, rtol=2e-4, atol=2e-4)


def test_persistent_decode_matches_sequence_forward() -> None:
    torch.manual_seed(4)
    model = QCCForCausalLM(
        vocab_size=29,
        d_model=24,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=32,
        window_size=3,
        num_codes=4,
    ).eval()
    tokens = torch.randint(0, 29, (2, 11))
    with torch.no_grad():
        sequence_logits = model(tokens)
        model.reset_cache(tokens.shape[0])
        streamed = torch.stack(
            [model.decode_step(tokens[:, t]) for t in range(tokens.shape[1])], dim=1
        )
    torch.testing.assert_close(streamed, sequence_logits, rtol=2e-4, atol=2e-4)


def test_query_stability_can_suppress_repeated_archive_reads() -> None:
    torch.manual_seed(41)
    attention = QCCSelfAttention(
        d_model=16,
        num_heads=4,
        window_size=2,
        num_codes=4,
        archive_query_cosine_threshold=0.99,
    ).eval()
    calls = 0
    original_read = attention.archive.read

    def counted_read(query: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_read(query)

    hidden = torch.zeros(1, 16)
    with patch.object(attention.archive, "read", counted_read):
        with torch.no_grad():
            for index in range(8):
                attention.step(hidden, reset_cache=index == 0)
    # The first post-window query refreshes the archive; identical queries can
    # reuse that response when the optional adaptive threshold is enabled.
    assert calls == 1


def test_token_archive_read_stride_refreshes_after_eviction() -> None:
    torch.manual_seed(42)
    kwargs = dict(
        d_model=16,
        num_heads=4,
        window_size=2,
        num_codes=4,
        use_triton=False,
    )
    strided = QCCSelfAttention(**kwargs, archive_read_stride=2).eval()
    reference = QCCSelfAttention(**kwargs, archive_read_stride=1).eval()
    reference.load_state_dict(strided.state_dict())
    hidden = torch.randn(1, 9, kwargs["d_model"])
    with torch.no_grad():
        for index in range(hidden.shape[1]):
            actual = strided.step(hidden[:, index], reset_cache=index == 0)
            expected = reference.step(hidden[:, index], reset_cache=index == 0)
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("use_archive", [True, False])
def test_decode_chunk_matches_token_stream(use_archive: bool) -> None:
    torch.manual_seed(6)
    model = QCCForCausalLM(
        vocab_size=29,
        d_model=24,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=32,
        window_size=3 if use_archive else 32,
        num_codes=4,
        use_archive=use_archive,
    ).eval()
    tokens = torch.randint(0, 29, (2, 13))
    with torch.no_grad():
        model.reset_cache(tokens.shape[0])
        token_logits = torch.stack(
            [model.decode_step(tokens[:, t]) for t in range(tokens.shape[1])], dim=1
        )
        model.reset_cache(tokens.shape[0])
        chunk_logits = torch.cat(
            [
                model.decode_chunk(tokens[:, :5], reset_cache=True),
                model.decode_chunk(tokens[:, 5:9]),
                model.decode_chunk(tokens[:, 9:]),
            ],
            dim=1,
        )
    torch.testing.assert_close(chunk_logits, token_logits, rtol=2e-4, atol=2e-4)


def test_lexical_landmark_decode_paths_match_and_ring_is_bounded() -> None:
    """Lexical archive addressing must be transparent to serving semantics."""

    torch.manual_seed(61)
    model = QCCForCausalLM(
        vocab_size=31,
        d_model=24,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=128,
        window_size=5,
        num_codes=4,
        use_triton=False,
        archive_lexical_landmark=True,
        position_encoding="rope",
    ).eval()
    tokens = torch.randint(0, 31, (2, 19))
    with torch.no_grad():
        sequence = model(tokens)
        model.reset_cache(tokens.shape[0])
        streamed = torch.stack(
            [model.decode_step(tokens[:, t]) for t in range(tokens.shape[1])], dim=1
        )
        model.reset_cache(tokens.shape[0])
        chunked = torch.cat(
            [model.decode_chunk(tokens[:, :7], reset_cache=True), model.decode_chunk(tokens[:, 7:])],
            dim=1,
        )
    torch.testing.assert_close(streamed, sequence, rtol=3e-4, atol=3e-4)
    torch.testing.assert_close(chunked, streamed, rtol=3e-4, atol=3e-4)
    for layer in model.layers:
        attention = layer.attention
        assert attention._lexical_key_cache is not None
        assert attention._lexical_value_cache is not None
        assert attention._cache_length <= attention.window_size
        assert attention._lexical_key_cache.shape[2] == attention.window_size


def test_lexical_triplet_is_position_free_and_matches_token_embedding() -> None:
    torch.manual_seed(62)
    attention = QCCSelfAttention(
        d_model=16,
        num_heads=4,
        window_size=3,
        num_codes=4,
        archive_lexical_landmark=True,
        use_triton=False,
    )
    hint = torch.randn(2, 7, 16)
    query, key, value = attention._lexical_archive_triplet(hint)
    assert query is not None and key is not None and value is not None
    expected = hint.view(2, 7, 4, 4).transpose(1, 2)
    torch.testing.assert_close(query, expected)
    torch.testing.assert_close(key, expected)
    torch.testing.assert_close(value, expected)


def test_sparse_archive_read_matches_dense_when_all_codes_active() -> None:
    torch.manual_seed(7)
    kwargs = dict(num_heads=2, head_dim=8, num_codes=6, window_size=3, use_triton=False)
    dense = QCCArchive(**kwargs)
    sparse = QCCArchive(**kwargs, active_codes=6)
    sparse.load_state_dict(dense.state_dict())
    keys = torch.randn(2, 2, 9, 8)
    values = torch.randn_like(keys)
    queries = torch.randn_like(keys)
    with torch.no_grad():
        dense.reset_state(2)
        sparse.reset_state(2)
        dense_out = dense.update_read_chunk(keys, values, queries)
        sparse_out = sparse.update_read_chunk(keys, values, queries)
    torch.testing.assert_close(sparse_out, dense_out, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(sparse.state.numerator, dense.state.numerator, rtol=2e-5, atol=2e-5)


def test_lazy_decay_matches_dense_recurrence_when_all_codes_active() -> None:
    torch.manual_seed(9)
    kwargs = dict(
        num_heads=2,
        head_dim=8,
        num_codes=6,
        decay_rates=(0.8, 0.95),
        window_size=3,
        use_triton=False,
    )
    dense = QCCArchive(**kwargs)
    lazy = QCCArchive(**kwargs, active_codes=6, lazy_decay=True)
    lazy.load_state_dict(dense.state_dict(), strict=False)
    keys = torch.randn(2, 2, 11, 8)
    values = torch.randn_like(keys)
    queries = torch.randn_like(keys)
    with torch.no_grad():
        dense.reset_state(2)
        lazy.reset_state(2)
        dense_out = dense.update_read_chunk(keys, values, queries)
        lazy_out = lazy.update_read_chunk(keys, values, queries)
    torch.testing.assert_close(lazy_out, dense_out, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(lazy.state.numerator, dense.state.numerator, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(lazy.state.denominator, dense.state.denominator, rtol=2e-5, atol=2e-5)


def test_sparse_decode_chunk_matches_sparse_token_stream() -> None:
    torch.manual_seed(8)
    model = QCCForCausalLM(
        vocab_size=23,
        d_model=24,
        num_layers=1,
        num_heads=4,
        max_position_embeddings=32,
        window_size=3,
        num_codes=8,
        active_codes=2,
        use_triton=False,
    ).eval()
    tokens = torch.randint(0, 23, (2, 13))
    with torch.no_grad():
        model.reset_cache(2)
        token_logits = torch.stack(
            [model.decode_step(tokens[:, t]) for t in range(tokens.shape[1])], dim=1
        )
        model.reset_cache(2)
        chunk_logits = torch.cat(
            [model.decode_chunk(tokens[:, :5], reset_cache=True), model.decode_chunk(tokens[:, 5:])],
            dim=1,
        )
    torch.testing.assert_close(chunk_logits, token_logits, rtol=2e-4, atol=2e-4)


def test_persistent_local_cache_never_exceeds_window() -> None:
    model = QCCForCausalLM(
        vocab_size=13,
        d_model=16,
        num_layers=1,
        num_heads=4,
        max_position_embeddings=64,
        window_size=3,
        num_codes=4,
    ).eval()
    with torch.no_grad():
        model.reset_cache(1)
        for token in torch.randint(0, 13, (20,)):
            model.decode_step(token[None])
    cache = model.layers[0].attention
    assert cache._cache_length <= cache.window_size
    assert cache._local_key_cache is not None
    assert cache._local_key_cache.shape[2] == cache.window_size
    assert cache._local_value_cache is not None
    assert cache._local_value_cache.shape[2] == cache.window_size


def test_archive_state_is_constant_in_sequence_length() -> None:
    model = QCCForCausalLM(
        vocab_size=17, d_model=16, num_layers=3, num_heads=4, window_size=2
    )
    short = torch.randint(0, 17, (1, 8))
    long = torch.randint(0, 17, (1, 32))
    model(short)
    short_shape = model.layers[0].attention.archive.state.numerator.shape
    model(long)
    long_shape = model.layers[0].attention.archive.state.numerator.shape
    assert short_shape == long_shape
    assert short_shape[0] == 1
    assert short_shape[2:] == (16, 4, 4)


def test_triton_kernel_is_optional_on_cpu() -> None:
    assert isinstance(TRITON_AVAILABLE, bool)
    archive = QCCArchive(num_heads=1, head_dim=2, num_codes=1, decay_rates=(0.8,), window_size=2)
    with torch.no_grad():
        try:
            triton_update_archive(
                archive.state.numerator,
                archive.state.denominator,
                torch.zeros(1, 1, 2),
                torch.zeros(1, 1, 2),
                archive.codes,
                archive.decay_rates,
                archive.window_size,
            )
        except RuntimeError as exc:
            assert "Triton CUDA runtime" in str(exc)
        else:
            raise AssertionError("CPU invocation must not execute a CUDA kernel")
    with pytest.raises(RuntimeError, match="Triton CUDA runtime"):
        triton_lazy_update_archive(
            archive.state.numerator,
            archive.state.denominator,
            archive._last_step,
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2),
            archive.codes,
            torch.zeros(1, 1, 1, dtype=torch.long),
            archive.decay_rates,
            archive.window_size,
            1,
        )
    with pytest.raises(RuntimeError, match="Triton CUDA runtime"):
        triton_sparse_read_archive(
            torch.zeros(1, 1, 2),
            archive.state.numerator,
            archive.state.denominator,
            archive._last_step,
            archive.codes,
            archive.mix_logits,
            torch.zeros(1, 1, 1, dtype=torch.long),
            torch.zeros(1, 1, 1),
            archive.decay_rates,
            1,
        )
    with pytest.raises(RuntimeError, match="Triton CUDA runtime"):
        triton_update_read_archive_chunk(
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2),
            archive.state.numerator,
            archive.state.denominator,
            archive.codes,
            archive.mix_logits,
            archive.decay_rates,
            archive.window_size,
        )
    with pytest.raises(RuntimeError, match="Triton CUDA runtime"):
        triton_sparse_update_read_archive_chunk(
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2),
            archive.state.numerator,
            archive.state.denominator,
            archive._last_step,
            archive.codes,
            archive.mix_logits,
            archive.decay_rates,
            archive.window_size,
            0,
            1,
        )


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA and Triton required")
def test_triton_update_and_read_match_reference() -> None:
    torch.manual_seed(5)
    kwargs = dict(num_heads=2, head_dim=16, num_codes=4, decay_rates=(0.9, 0.97), window_size=3)
    reference = QCCArchive(**kwargs, use_triton=False).cuda()
    fused = QCCArchive(**kwargs, use_triton=True).cuda()
    # Compare implementations under the same learned codebook/mix weights;
    # constructing two archives after one seed otherwise gives independent
    # random parameters and makes a valid kernel look numerically wrong.
    fused.load_state_dict(reference.state_dict())
    keys = torch.randn(2, 2, 16, device="cuda")
    values = torch.randn_like(keys)
    query = torch.randn_like(keys)
    with torch.no_grad():
        reference.reset_state(2, device=keys.device)
        fused.reset_state(2, device=keys.device)
        reference.update(keys, values)
        fused.update(keys, values)
        reference_out = reference.read(query)
        fused_out = fused.read(query)
    torch.testing.assert_close(fused.state.denominator, reference.state.denominator, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(fused.state.numerator, reference.state.numerator, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(fused_out, reference_out, rtol=3e-3, atol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA and Triton required")
def test_triton_sparse_lazy_matches_reference() -> None:
    torch.manual_seed(10)
    kwargs = dict(
        num_heads=2,
        head_dim=16,
        num_codes=8,
        active_codes=4,
        lazy_decay=True,
        decay_rates=(0.9, 0.97),
        window_size=3,
    )
    reference = QCCArchive(**kwargs, use_triton=False).cuda()
    fused = QCCArchive(**kwargs, use_triton=True).cuda()
    fused.load_state_dict(reference.state_dict())
    keys = torch.randn(2, 2, 7, 16, device="cuda")
    values = torch.randn_like(keys)
    queries = torch.randn_like(keys)
    with torch.no_grad():
        reference.reset_state(2, device=keys.device)
        fused.reset_state(2, device=keys.device)
        reference_out = reference.update_read_chunk(keys, values, queries)
        fused_out = fused.update_read_chunk(keys, values, queries)
    torch.testing.assert_close(fused.state.denominator, reference.state.denominator, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(fused.state.numerator, reference.state.numerator, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(fused_out, reference_out, rtol=3e-3, atol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA and Triton required")
def test_triton_fused_chunk_matches_reference() -> None:
    torch.manual_seed(11)
    kwargs = dict(
        num_heads=2,
        head_dim=16,
        num_codes=8,
        decay_rates=(0.9, 0.97, 0.995),
        window_size=3,
        scan_block_size=5,
    )
    reference = QCCArchive(**kwargs, use_triton=False).cuda()
    fused = QCCArchive(**kwargs, use_triton=True).cuda()
    fused.load_state_dict(reference.state_dict())
    keys = torch.randn(2, 2, 13, 16, device="cuda")
    values = torch.randn_like(keys)
    queries = torch.randn_like(keys)
    with torch.no_grad():
        reference.reset_state(2, device=keys.device)
        fused.reset_state(2, device=keys.device)
        reference_out = reference.update_read_chunk(keys, values, queries)
        fused_out = fused.update_read_chunk(keys, values, queries)
    torch.testing.assert_close(fused.state.denominator, reference.state.denominator, rtol=4e-3, atol=4e-3)
    torch.testing.assert_close(fused.state.numerator, reference.state.numerator, rtol=4e-3, atol=4e-3)
    torch.testing.assert_close(fused_out, reference_out, rtol=4e-3, atol=4e-3)
