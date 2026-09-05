import copy

import torch

from qcc_transformer import (
    HybridQCCArchive,
    LandmarkAdmissionPredictor,
    QCCArchive,
    QCCSelfAttention,
    upgrade_qcc_attention,
)


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
