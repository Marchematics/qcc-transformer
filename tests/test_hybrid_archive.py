import copy

import torch
import pytest

from qcc_transformer import (
    HybridQCCArchive,
    LandmarkAdmissionPredictor,
    QCCArchive,
    QCCSelfAttention,
    upgrade_qcc_attention,
)


def test_rotary_exact_tier_matches_chunk_and_token_decode() -> None:
    torch.manual_seed(7)
    attention = QCCSelfAttention(
        16, 2, window_size=4, num_codes=3, use_triton=False,
        rope_theta=10000, archive_position_invariant=True,
    ).eval()
    upgrade_qcc_attention(
        attention, exact_num_sets=2, exact_ways=8,
        admission_bias_init=1, max_inserts_per_chunk=16,
    )
    tokenwise = copy.deepcopy(attention)
    hidden = torch.randn(1, 13, 16)
    with torch.no_grad():
        expected = attention.step_chunk(hidden)
        actual = torch.stack([tokenwise.step(hidden[:, i]) for i in range(13)], 1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        tokenwise.archive.exact_bank._keys, attention.archive.exact_bank._keys,
        atol=1e-6, rtol=1e-5,
    )


def test_weighted_exact_archive_matches_full_kv_while_all_keys_fit() -> None:
    torch.manual_seed(113)
    attention = QCCSelfAttention(
        16, 2, window_size=4, attention_sink_size=2, num_codes=3,
        use_triton=False, rope_theta=10000, archive_position_invariant=True,
    ).eval()
    reference = copy.deepcopy(attention)
    reference.use_archive = False
    reference.window_size = 32
    upgrade_qcc_attention(
        attention, exact_num_sets=1, exact_ways=32, exact_attention=True,
        admission_bias_init=1, max_inserts_per_chunk=32,
    )
    tokenwise = copy.deepcopy(attention)
    x = torch.randn(1, 13, 16)
    assert not attention.archive.exact_bank.set_codes.requires_grad
    with torch.no_grad():
        expected = reference.step_chunk(x)
        actual = torch.cat([attention.step_chunk(x[:, :3]), attention.step_chunk(x[:, 3:7]), attention.step_chunk(x[:, 7:])], 1)
        single = torch.stack([tokenwise.step(x[:, i]) for i in range(13)], 1)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(single, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(attention(x), expected, atol=1e-6, rtol=1e-5)
    attention.reset_cache(1, device=x.device)
    differentiable = attention(x)
    torch.testing.assert_close(differentiable, expected, atol=1e-6, rtol=1e-5)
    differentiable.sum().backward()


def test_causal_coreset_matches_full_kv_before_capacity_is_reached() -> None:
    torch.manual_seed(118)
    attention = QCCSelfAttention(
        16, 2, window_size=4, attention_sink_size=2, num_codes=3,
        use_triton=False, rope_theta=10000, archive_position_invariant=True,
    ).eval()
    reference = copy.deepcopy(attention)
    reference.use_archive = False
    reference.window_size = 32
    upgrade_qcc_attention(
        attention,
        exact_attention=True,
        causal_coreset=True,
        coreset_capacity=32,
    )
    x = torch.randn(1, 19, 16)
    with torch.no_grad():
        expected = reference.step_chunk(x)
        actual = attention.step_chunk(x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert attention.archive.causal_coreset is True
    assert attention.archive.exact_bank.state_bytes() == attention.archive.exact_state_bytes()
    assert not any(parameter.requires_grad for parameter in attention.archive.parameters())
    assert not any(parameter.requires_grad for parameter in attention.gate.parameters())


@pytest.mark.parametrize('block_size', [1, 2])
def test_background_streaming_matches_chunks_after_reservoir_fills(block_size):
    torch.manual_seed(83)
    attention = QCCSelfAttention(16, 2, window_size=4, attention_sink_size=2,
                                use_triton=False, rope_theta=10000, archive_position_invariant=True).eval()
    upgrade_qcc_attention(attention, exact_num_sets=1, exact_ways=2,
                          exact_attention=True, background_size=3, admission_bias_init=1,
                          block_size=block_size,
                          max_inserts_per_chunk=32)
    reference = copy.deepcopy(attention)
    x = torch.randn(1, 18, 16)
    with torch.no_grad():
        chunks = torch.cat([attention.step_chunk(x[:, i:i+3]) for i in range(0, 18, 3)], 1)
        tokens = torch.stack([reference.step(x[:, i]) for i in range(18)], 1)
        torch.testing.assert_close(chunks, tokens, atol=1e-6, rtol=1e-5)
        assert torch.equal(attention.archive.exact_bank._background_count, torch.full((1, 2), 10))
        attention.reset_cache(1, device=x.device)
        torch.testing.assert_close(attention.step_chunk(x), tokens, atol=1e-6, rtol=1e-5)


def test_admission_predictor_is_fail_safe_before_calibration() -> None:
    predictor = LandmarkAdmissionPredictor(num_heads=2, head_dim=8)
    key = torch.randn(3, 2, 8)
    value = torch.randn_like(key)
    score = predictor(key, value)
    torch.testing.assert_close(score, torch.full_like(score, -4.0))
    block = predictor(key[:, :, None].expand(-1, -1, 5, -1), value[:, :, None].expand(-1, -1, 5, -1))
    assert block.shape == (3, 2, 5)
    torch.testing.assert_close(block, torch.full_like(block, -4.0))


def test_hybrid_upgrade_preserves_base_recurrence_when_exact_tier_rejects() -> None:
    torch.manual_seed(71)
    base = QCCArchive(
        num_heads=2,
        head_dim=8,
        num_codes=4,
        decay_rates=(0.9, 0.99),
        window_size=4,
        use_triton=False,
    )
    reference = copy.deepcopy(base)
    hybrid = HybridQCCArchive.from_archive(
        base,
        exact_num_sets=4,
        exact_ways=2,
        admission_threshold=100.0,
    )
    keys = torch.randn(1, 2, 13, 8)
    values = torch.randn_like(keys)
    query = torch.randn(1, 2, 13, 8)
    with torch.no_grad():
        reference_out = reference.update_read_chunk(keys, values, query)
        hybrid_out = hybrid.update_read_chunk(keys, values, query)
    torch.testing.assert_close(hybrid_out, reference_out, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(hybrid.state.numerator, reference.state.numerator)
    assert not torch.isfinite(hybrid.exact_bank.state.scores).any()


def test_exact_tier_rescues_salient_record_after_many_distractors() -> None:
    base = QCCArchive(
        num_heads=1,
        head_dim=2,
        num_codes=1,
        decay_rates=(0.8,),
        window_size=2,
        use_triton=False,
    )
    with torch.no_grad():
        base.codes.zero_()
    reference = copy.deepcopy(base)
    hybrid = HybridQCCArchive.from_archive(
        base,
        exact_num_sets=1,
        exact_ways=2,
        exact_confidence_threshold=0.0,
        exact_mix_bias_init=20.0,
        admission_threshold=0.0,
    )
    with torch.no_grad():
        hybrid.admission.key_weight.zero_()
        hybrid.admission.value_weight.zero_()
        hybrid.admission.bias.zero_()
        hybrid.admission.key_weight[0, 0] = 10.0

    needle_key = torch.tensor([[[1.0, 0.0]]])
    needle_value = torch.tensor([[[10.0, -3.0]]])
    reference.update(needle_key, needle_value)
    hybrid.update(needle_key, needle_value)
    for _ in range(50):
        distractor_key = torch.tensor([[[-1.0, 0.0]]])
        distractor_value = torch.tensor([[[0.0, 7.0]]])
        reference.update(distractor_key, distractor_value)
        hybrid.update(distractor_key, distractor_value)

    query = needle_key
    recurrent = reference.read(query)
    rescued = hybrid.read(query)
    recurrent_error = (recurrent - needle_value).square().sum()
    rescued_error = (rescued - needle_value).square().sum()
    assert float(rescued_error) < 1e-5
    assert float(rescued_error) < float(recurrent_error)


def test_chunk_admission_has_hard_insert_budget() -> None:
    torch.manual_seed(72)
    base = QCCArchive(
        num_heads=1,
        head_dim=4,
        num_codes=2,
        decay_rates=(0.9, 0.99),
        window_size=2,
        use_triton=False,
    )
    hybrid = HybridQCCArchive.from_archive(
        base,
        exact_num_sets=1,
        exact_ways=8,
        max_inserts_per_chunk=2,
        admission_threshold=0.0,
    )
    with torch.no_grad():
        hybrid.admission.key_weight.zero_()
        hybrid.admission.value_weight.zero_()
        hybrid.admission.bias.fill_(10.0)
    key = torch.randn(1, 1, 9, 4)
    value = torch.randn_like(key)
    query = torch.randn_like(key)
    with torch.no_grad():
        output = hybrid.update_read_chunk(key, value, query)
    assert output.shape == query.shape
    assert int(torch.isfinite(hybrid.exact_bank.state.scores).sum()) <= 2


def test_chunk_admission_reopens_budget_for_each_bounded_tile() -> None:
    torch.manual_seed(73)
    base = QCCArchive(
        num_heads=1,
        head_dim=4,
        num_codes=2,
        decay_rates=(0.9, 0.99),
        window_size=2,
        use_triton=False,
        scan_block_size=4,
    )
    hybrid = HybridQCCArchive.from_archive(
        base,
        exact_num_sets=1,
        exact_ways=16,
        max_inserts_per_chunk=2,
        admission_threshold=0.0,
    )
    with torch.no_grad():
        hybrid.admission.key_weight.zero_()
        hybrid.admission.value_weight.zero_()
        hybrid.admission.bias.fill_(10.0)
    key = torch.randn(1, 1, 10, 4)
    value = torch.randn_like(key)
    query = torch.randn_like(key)
    with torch.no_grad():
        output = hybrid.update_read_chunk(key, value, query)
    assert output.shape == query.shape
    # The admission limit is per bounded scan tile, not per entire prompt.
    # This prevents a long prefill from giving all of its exact capacity to
    # the final few global scores.
    assert int(torch.isfinite(hybrid.exact_bank.state.scores).sum()) == 6


def test_quality_first_uses_bounded_score_hard_exact_shadow():
    base = QCCArchive(
        num_heads=1,
        head_dim=4,
        num_codes=2,
        decay_rates=(0.9, 0.99),
        window_size=2,
        use_triton=False,
        scan_block_size=16,
    )
    hybrid = HybridQCCArchive.from_archive(
        base,
        exact_num_sets=1,
        exact_ways=4,
        quality_first=True,
    )
    assert hybrid.exact_bank.replacement_policy == "score"
    assert hybrid.exact_hard_read is True
    assert hybrid.max_inserts_per_chunk == 4
    assert hybrid.scan_block_size == 4
    key = torch.randn(1, 1, 9, 4)
    value = torch.randn_like(key)
    query = torch.randn_like(key)
    with torch.no_grad():
        output = hybrid.update_read_chunk(key, value, query)
    assert output.shape == query.shape
    assert int(torch.isfinite(hybrid.exact_bank.state.scores).sum()) == 4


def test_quality_first_batched_prefill_matches_independent_requests():
    torch.manual_seed(217)

    def make_archive():
        base = QCCArchive(
            num_heads=1,
            head_dim=4,
            num_codes=2,
            decay_rates=(0.9,),
            window_size=2,
            use_triton=False,
            scan_block_size=8,
        )
        return HybridQCCArchive.from_archive(
            base,
            exact_num_sets=1,
            exact_ways=2,
            quality_first=True,
            exact_attention=True,
            background_size=3,
            block_size=2,
            admission_threshold=-1.0,
            max_inserts_per_chunk=2,
        )

    batched = make_archive()
    independent = [copy.deepcopy(batched), copy.deepcopy(batched)]
    fresh = copy.deepcopy(batched)
    key = torch.randn(2, 1, 11, 4)
    value = torch.randn_like(key)
    query = torch.randn_like(key)
    with torch.no_grad():
        batched_output = batched.update_read_chunk(key, value, query)
        independent_outputs = [
            archive.update_read_chunk(key[i:i + 1], value[i:i + 1], query[i:i + 1])
            for i, archive in enumerate(independent)
        ]
    torch.testing.assert_close(
        batched_output,
        torch.cat(independent_outputs, dim=0),
        atol=1e-6,
        rtol=1e-5,
    )
    for batch_index, archive in enumerate(independent):
        torch.testing.assert_close(
            batched.exact_bank._keys[batch_index:batch_index + 1],
            archive.exact_bank._keys,
            atol=1e-6,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            batched.exact_bank._scores[batch_index:batch_index + 1],
            archive.exact_bank._scores,
            atol=1e-6,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            batched.exact_bank._background_keys[batch_index:batch_index + 1],
            archive.exact_bank._background_keys,
            atol=1e-6,
            rtol=1e-5,
        )

    # Continue past partially filled blocks and reservoir replacement in decode.
    with torch.no_grad():
        for _ in range(7):
            next_key = torch.randn(2, 1, 4)
            next_value = torch.randn_like(next_key)
            next_query = torch.randn_like(next_key)
            batched.update(next_key, next_value)
            expected = []
            for i, archive in enumerate(independent):
                archive.update(next_key[i:i + 1], next_value[i:i + 1])
                expected.append(archive.read(next_query[i:i + 1]))
            torch.testing.assert_close(
                batched.read(next_query), torch.cat(expected), atol=1e-6, rtol=1e-5,
            )

        # Reusing the archive must reproduce a new request, including RNG state.
        batched.reset_state(2, device=key.device)
        torch.testing.assert_close(
            batched.update_read_chunk(key, value, query),
            fresh.update_read_chunk(key, value, query),
            atol=1e-6, rtol=1e-5,
        )
        torch.testing.assert_close(
            batched.exact_bank._background_keys, fresh.exact_bank._background_keys,
        )


def test_quality_first_salience_is_independent_of_key_tile_boundaries():
    torch.manual_seed(220)
    archive = HybridQCCArchive(1, 4, window_size=2, use_triton=False, quality_first=True)
    key = torch.randn(1, 1, 8, 4)
    query = torch.randn(1, 1, 128, 4)
    whole = archive._quality_first_salience(key, query, key_start=0)
    split = torch.cat([
        archive._quality_first_salience(key[:, :, :3], query, key_start=0),
        archive._quality_first_salience(key[:, :, 3:], query, key_start=3),
    ], dim=2)
    torch.testing.assert_close(whole, split)


def test_quality_first_salience_respects_sliced_query_origin():
    base = QCCArchive(
        num_heads=1,
        head_dim=4,
        num_codes=2,
        decay_rates=(0.9,),
        window_size=4,
        use_triton=False,
    )
    hybrid = HybridQCCArchive.from_archive(
        base,
        exact_num_sets=2,
        exact_ways=2,
        quality_first=True,
    )
    key = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
    future_queries = torch.tensor(
        [[[
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]]]
    )
    score = hybrid._quality_first_salience(
        key,
        future_queries,
        key_start=6,
        query_start=4,
        sample_queries=3,
    )
    assert float(score.item()) > 0.99


def test_salience_excludes_queries_before_key_eviction():
    archive = HybridQCCArchive(1, 2, window_size=4, use_triton=False, quality_first=True)
    key = torch.tensor([[[[1., 0.]]]])
    queries = torch.tensor([[[[1., 0.], [0., 1.], [0., 1.]]]])
    # Event 6 is absolute query token 10. Events 4–5 occur before key 6
    # leaves the local window, so their strong match must not count.
    score = archive._quality_first_salience(key, queries, key_start=6, query_start=4)
    torch.testing.assert_close(score, torch.zeros_like(score))


def test_quality_first_exposes_exact_confidence_for_outer_gate():
    base = QCCArchive(
        num_heads=1,
        head_dim=2,
        num_codes=1,
        decay_rates=(0.9,),
        window_size=2,
        use_triton=False,
    )
    hybrid = HybridQCCArchive.from_archive(
        base,
        exact_num_sets=1,
        exact_ways=2,
        quality_first=True,
        exact_mix_bias_init=20.0,
    )
    key = torch.tensor([[[[1.0, 0.0]]]]).expand(1, 1, 4, 2).clone()
    value = torch.randn_like(key)
    query = key.clone()
    with torch.no_grad():
        hybrid.update_read_chunk(key, value, query)
    assert hybrid._last_exact_gate is not None
    assert hybrid._last_exact_gate.shape == (1, 1, 4)
    # At least the query after the first admitted record is an exact hit.  The
    # attention wrapper uses this bounded signal to avoid diluting that hit
    # with the local path.
    assert float(hybrid._last_exact_gate.max()) > 0.99


def test_quality_first_retains_future_salient_token_across_tiles():
    base = QCCArchive(
        num_heads=1,
        head_dim=2,
        num_codes=1,
        decay_rates=(0.9,),
        window_size=2,
        use_triton=False,
        scan_block_size=8,
    )
    hybrid = HybridQCCArchive.from_archive(
        base,
        exact_num_sets=1,
        exact_ways=2,
        quality_first=True,
        exact_mix_bias_init=20.0,
    )
    needle = torch.tensor([[[[1.0, 0.0]]]])
    key = torch.cat(
        (needle, torch.tensor([[[-1.0, 0.0]]]).expand(1, 1, 5, 2)), dim=2
    )
    value = torch.arange(12, dtype=torch.float32).view(1, 1, 6, 2)
    query = torch.tensor([[[[1.0, 0.0]]]]).expand(1, 1, 6, 2).clone()
    query[:, :, 1:] = torch.tensor([-1.0, 0.0])
    with torch.no_grad():
        hybrid.update_read_chunk(key, value, query)
    stored_keys = hybrid.exact_bank.state.keys.reshape(1, 1, -1, 2)
    similarity = torch.nn.functional.cosine_similarity(
        stored_keys, needle.expand_as(stored_keys), dim=-1
    )
    assert bool((similarity > 0.99).any())


def test_quality_first_successor_score_propagates_one_block_only():
    def make_archive(propagate):
        return HybridQCCArchive(
            num_heads=1,
            head_dim=2,
            num_codes=1,
            window_size=2,
            use_triton=False,
            exact_num_sets=2,
            exact_ways=2,
            quality_first=True,
            quality_block_propagation=propagate,
            exact_attention=True,
            background_size=1,
            block_size=2,
        )

    keys = [torch.tensor([[[float(i), 0.0]]]) for i in range(8)]
    values = [key.clone() for key in keys]

    def run(archive):
        with torch.no_grad():
            for block, raw_score in enumerate((0.1, 0.7, 0.8, 0.2)):
                for offset in range(2):
                    score = torch.tensor([[raw_score]])
                    archive._admit_one(keys[block * 2 + offset], values[block * 2 + offset], score)
        stored = archive.exact_bank._keys.reshape(1, 1, -1, 2)
        return bool((stored == keys[6].reshape(1, 1, 1, 2)).all(-1).any())

    assert not run(make_archive(False))
    assert run(make_archive(True))

    propagated = make_archive(True)
    with torch.no_grad():
        for i, raw_score in enumerate((0.1, 0.1, 0.2, 0.9, 0.3, 0.4, 0.1, 0.1)):
            propagated._admit_one(keys[i], values[i], torch.tensor([[raw_score]]))
            if i == 3:
                torch.testing.assert_close(propagated._quality_previous_raw_score,
                                           torch.tensor([[0.9]]))
                assert propagated.exact_bank._scores.max().item() == pytest.approx(0.9)
            if i == 5:
                torch.testing.assert_close(propagated._quality_previous_raw_score,
                                           torch.tensor([[0.4]]))
                assert propagated.exact_bank._scores.min().item() == pytest.approx(0.9)
    # The propagated 0.9 must not spread into the fourth block.
    assert not (propagated.exact_bank._keys[..., 0] == 6).any()


def test_quality_first_decode_uses_rotary_cosine_score():
    archive = HybridQCCArchive(
        num_heads=1,
        head_dim=2,
        num_codes=1,
        window_size=2,
        use_triton=False,
        exact_num_sets=1,
        exact_ways=2,
        quality_first=True,
        exact_attention=True,
        background_size=1,
        block_size=2,
    )
    key = torch.tensor([[[1.0, 0.0]]])
    value = torch.tensor([[[3.0, 4.0]]])
    with torch.no_grad():
        archive.update(
            key, value, exact_key=key, exact_query=key,
        )
        archive.update(
            key, value, exact_key=key, exact_query=key,
        )
    stored_scores = archive.exact_bank._scores[0, 0]
    assert bool(torch.isfinite(stored_scores).any())
    assert float(stored_scores[torch.isfinite(stored_scores)][0]) == pytest.approx(1.0)


def test_quality_prefill_shadow_only_keeps_recurrent_output_and_populates_bank():
    torch.manual_seed(227)
    base = QCCArchive(
        num_heads=1,
        head_dim=4,
        num_codes=2,
        decay_rates=(0.9,),
        window_size=2,
        use_triton=False,
        scan_block_size=4,
    )
    hybrid = HybridQCCArchive.from_archive(
        base,
        exact_num_sets=2,
        exact_ways=2,
        quality_first=True,
        exact_attention=True,
        quality_prefill_shadow_only=True,
        background_size=2,
        block_size=2,
    )
    reference = copy.deepcopy(base)
    key = torch.randn(1, 1, 9, 4)
    value = torch.randn_like(key)
    query = torch.randn_like(key)
    with torch.no_grad():
        expected = reference.update_read_chunk(key, value, query)
        actual = hybrid.update_read_chunk(key, value, query)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert bool(torch.isfinite(hybrid.exact_bank._scores).any())


def test_exact_query_correction_changes_the_active_exact_read_path():
    base = QCCArchive(
        num_heads=1,
        head_dim=2,
        num_codes=1,
        decay_rates=(0.9,),
        window_size=2,
        use_triton=False,
        query_correction_rank=1,
    )
    hybrid = HybridQCCArchive.from_archive(
        base,
        exact_num_sets=1,
        exact_ways=2,
        exact_attention=True,
        exact_query_correction=True,
    )
    with torch.no_grad():
        hybrid.query_correction_v.zero_()
        hybrid.query_correction_u.zero_()
        hybrid.query_correction_v[0, 0, 0] = 1.0
        hybrid.query_correction_u[0, 0, 1] = 1.0
        hybrid.exact_bank.update(
            torch.tensor([[[1.0, 1.0]]]), torch.tensor([[[10.0, 0.0]]]),
            admission_bias=torch.tensor([[1.0]]),
        )
        hybrid.exact_bank.update(
            torch.tensor([[[-1.0, -1.0]]]), torch.tensor([[[0.0, 10.0]]]),
            admission_bias=torch.tensor([[1.0]]),
        )
        query = torch.tensor([[[1.0, 0.0]]])
        corrected = hybrid._read_exact(query)[0]
        original = hybrid.exact_bank.read_attention(query)[0]
    assert not torch.equal(corrected, original)


def test_active_query_correction_changes_live_attention_query():
    attention = QCCSelfAttention(
        8,
        2,
        window_size=2,
        num_codes=1,
        use_triton=False,
        rope_theta=10000,
        archive_query_correction_rank=1,
        active_query_correction=True,
    )
    query = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.0, -1.0]]])
    with torch.no_grad():
        original = attention._apply_active_query_correction(query)
        attention.archive.query_correction_v.zero_()
        attention.archive.query_correction_u.zero_()
        attention.archive.query_correction_v[0, 0, 0] = 1.0
        attention.archive.query_correction_u[0, 0, 1] = 2.0
        corrected = attention._apply_active_query_correction(query)
    assert not torch.equal(corrected, original)


def test_hybrid_exact_shadow_uses_rotary_side_channel():
    base = QCCArchive(
        num_heads=1,
        head_dim=2,
        num_codes=1,
        decay_rates=(0.9,),
        window_size=2,
        use_triton=False,
    )
    hybrid = HybridQCCArchive.from_archive(
        base,
        exact_num_sets=1,
        exact_ways=2,
        quality_first=True,
        exact_mix_bias_init=20.0,
    )
    raw_key = torch.tensor([[[[-1.0, 0.0]]]]).expand(1, 1, 3, 2).clone()
    rotary_key = torch.tensor([[[[1.0, 0.0]]]]).expand(1, 1, 3, 2).clone()
    value = torch.randn_like(raw_key)
    query = rotary_key.clone()
    with torch.no_grad():
        hybrid.update_read_chunk(
            raw_key,
            value,
            raw_key,
            exact_key=rotary_key,
            exact_query=query,
        )
    stored_keys = hybrid.exact_bank.state.keys.reshape(1, 1, -1, 2)
    similarity = torch.nn.functional.cosine_similarity(
        stored_keys, rotary_key[:, :, :1].expand_as(stored_keys), dim=-1
    )
    assert bool((similarity > 0.99).any())


def test_upgrade_qcc_attention_is_idempotent() -> None:
    attention = QCCSelfAttention(
        d_model=16,
        num_heads=2,
        window_size=4,
        num_codes=4,
        use_triton=False,
    )
    first = upgrade_qcc_attention(attention, exact_num_sets=4, exact_ways=2)
    second = upgrade_qcc_attention(attention, exact_num_sets=4, exact_ways=2)
    assert first is second
    assert isinstance(attention.archive, HybridQCCArchive)


def test_exact_only_reclaim_preserves_future_query_selection():
    torch.manual_seed(803)
    compact = HybridQCCArchive(
        2, 4, num_codes=2, window_size=4, use_triton=False,
        exact_num_sets=2, exact_ways=2, block_size=2, background_size=3,
        quality_first=True, exact_attention=True,
    )
    assert compact.exact_only
    assert compact._numerator.numel() == 1
    legacy = copy.deepcopy(compact)
    legacy.exact_only = False
    legacy.reset_state(1)
    key = torch.randn(1, 2, 18, 4)
    value = torch.randn_like(key)
    query = torch.randn_like(key)
    tail = torch.randn(1, 2, 5, 4)
    with torch.no_grad():
        for start in (0, 6, 12):
            kwargs = dict(quality_query=tail, quality_query_start=13,
                          quality_key_start=start)
            a = compact.update_read_chunk(key[:, :, start:start+6],
                value[:, :, start:start+6], query[:, :, start:start+6], **kwargs)
            b = legacy.update_read_chunk(key[:, :, start:start+6],
                value[:, :, start:start+6], query[:, :, start:start+6], **kwargs)
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            for name in ('_keys', '_values', '_scores', '_ages',
                         '_background_keys', '_background_values'):
                torch.testing.assert_close(getattr(compact.exact_bank, name),
                    getattr(legacy.exact_bank, name), atol=0, rtol=0)
