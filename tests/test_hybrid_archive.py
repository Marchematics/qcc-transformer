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
