import math
import pytest
import torch

from qcc_transformer import CausalWeightedKVBank, SetAssociativeLandmarkBank
from qcc_transformer import triton_kernels


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_attention_partition_combines_with_local_softmax(device):
    torch.manual_seed(92)
    bank = SetAssociativeLandmarkBank(2, 4, num_sets=1, ways=3, probe_sets=1)
    bank.reset_state(2, device=torch.device(device))
    keys = torch.randn(2, 2, 3, 4, device=device)
    values = torch.randn_like(keys)
    for i in range(3):
        bank.update(keys[:, :, i], values[:, :, i])
    bank._scores[1, 1].fill_(-torch.inf)
    query = torch.randn(2, 2, 129, 4, device=device)
    local_keys = torch.randn(2, 2, 5, 4, device=device)
    local_values = torch.randn_like(local_keys)
    remote, remote_z = bank.read_attention(query)
    local_logits = query @ local_keys.transpose(-1, -2) / 2
    local_z = local_logits.logsumexp(-1)
    local = local_logits.softmax(-1) @ local_values
    remote_weight = torch.sigmoid(remote_z - local_z).unsqueeze(-1)
    combined = local * (1 - remote_weight) + remote * remote_weight
    all_keys = torch.cat((bank._keys.flatten(2, 3), local_keys), 2)
    all_values = torch.cat((bank._values.flatten(2, 3), local_values), 2)
    valid = torch.cat((torch.isfinite(bank._scores.flatten(2, 3)), torch.ones(2, 2, 5, device=device, dtype=torch.bool)), -1)
    expected = torch.nn.functional.scaled_dot_product_attention(query, all_keys, all_values, attn_mask=valid.unsqueeze(2))
    torch.testing.assert_close(combined, expected, atol=1e-6, rtol=1e-5)
    assert torch.isneginf(remote_z[1, 1]).all()
    assert torch.count_nonzero(remote[1, 1]) == 0
    single, single_z = bank.read_attention(query[:, :, 0])
    torch.testing.assert_close(single, remote[:, :, 0])
    torch.testing.assert_close(single_z, remote_z[:, :, 0])


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_background_reservoir_is_disjoint_bounded_and_mass_correct(device):
    torch.manual_seed(39)
    bank = SetAssociativeLandmarkBank(1, 4, num_sets=1, ways=2, probe_sets=1,
        background_size=4, diversity_weight=0)
    bank.reset_state(1, device=torch.device(device))
    state_bytes = bank.state_bytes()
    keys = torch.randn(1, 1, 6, 4, device=device)
    values = torch.arange(6., device=device).view(1, 1, 6, 1).expand_as(keys)
    for i, score in enumerate((10., 1., 20., -1., 30., 2.)):
        bank.update(keys[:, :, i], values[:, :, i], admission_bias=torch.tensor([[score]], device=device))
    stored = torch.cat((bank._values.flatten(2, 3), bank._background_values), 2)
    assert sorted(stored[0, 0, :, 0].tolist()) == list(range(6))
    assert int(bank._background_count.item()) == 4
    query = torch.randn(1, 1, 3, 4, device=device)
    actual, log_z = bank.read_attention(query)
    logits = query @ keys.transpose(-1, -2) / 2
    torch.testing.assert_close(actual, logits.softmax(-1) @ values, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(log_z, logits.logsumexp(-1), atol=1e-6, rtol=1e-5)
    bank.reset_state(1, device=torch.device(device))
    for _ in range(50):
        bank.update(torch.zeros(1, 1, 4, device=device), torch.ones(1, 1, 4, device=device),
                    write_mask=torch.zeros(1, 1, device=device, dtype=torch.bool))
    actual, log_z = bank.read_attention(query)
    torch.testing.assert_close(actual, torch.ones_like(actual))
    torch.testing.assert_close(log_z, torch.full_like(log_z, math.log(50)))
    assert bank.state_bytes() == state_bytes
    assert int(bank._background_count.item()) == 50


def test_attention_current_block_normalizes_with_empty_and_populated_heads():
    torch.manual_seed(293)
    bank = SetAssociativeLandmarkBank(2, 4, num_sets=1, ways=3, probe_sets=1)
    bank.reset_state(1, device=torch.device('cpu'))
    keys = torch.randn(1, 2, 3, 4)
    values = torch.randn_like(keys)
    for i in range(3):
        bank.update(keys[:, :, i], values[:, :, i])
    bank._scores[:, 0].fill_(-torch.inf)
    query = torch.randn(1, 2, 129, 4)
    extra_keys = torch.randn_like(query)
    extra_values = torch.randn_like(query)
    actual, log_z = bank.read_attention(
        query, extra_keys=extra_keys, extra_values=extra_values,
    )
    all_keys = torch.cat((bank._keys.flatten(2, 3), extra_keys), 2)
    all_values = torch.cat((bank._values.flatten(2, 3), extra_values), 2)
    retained = torch.isfinite(bank._scores.flatten(2, 3))
    mask = torch.cat((
        retained.unsqueeze(2).expand(-1, -1, 129, -1),
        torch.ones(129, 129, dtype=torch.bool).tril().view(1, 1, 129, 129).expand(1, 2, -1, -1),
    ), -1)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, all_keys, all_values, attn_mask=mask,
    )
    expected_z = (query @ all_keys.transpose(-1, -2) / 2).masked_fill(~mask, -torch.inf).logsumexp(-1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(log_z, expected_z)


def test_background_reservoir_sampling_is_independent_per_batch_request():
    def make_bank():
        return SetAssociativeLandmarkBank(
            1, 2, num_sets=1, ways=2, probe_sets=1,
            diversity_weight=0, background_size=3,
        )

    batched = make_bank()
    independent = [make_bank(), make_bank()]
    for bank in independent:
        bank.load_state_dict(batched.state_dict())
    keys = torch.arange(40, dtype=torch.float32).reshape(2, 1, 10, 2)
    values = keys + 100
    writes = torch.tensor(
        [[[True], [False], [False], [True], [False], [False], [True], [False], [False], [False]],
         [[False], [True], [False], [False], [True], [False], [False], [True], [False], [False]]],
        dtype=torch.bool,
    ).reshape(2, 10, 1)
    with torch.no_grad():
        for index in range(keys.shape[2]):
            score = torch.full((2, 1), 1.0)
            batched.update(
                keys[:, :, index], values[:, :, index],
                admission_bias=score, write_mask=writes[:, index],
            )
            for batch_index, bank in enumerate(independent):
                bank.update(
                    keys[batch_index:batch_index + 1, :, index],
                    values[batch_index:batch_index + 1, :, index],
                    admission_bias=score[batch_index:batch_index + 1],
                    write_mask=writes[batch_index:batch_index + 1, index],
                )
    for batch_index, bank in enumerate(independent):
        torch.testing.assert_close(
            batched._background_keys[batch_index:batch_index + 1],
            bank._background_keys,
        )
        torch.testing.assert_close(
            batched._background_values[batch_index:batch_index + 1],
            bank._background_values,
        )
        torch.testing.assert_close(
            batched._keys[batch_index:batch_index + 1], bank._keys,
        )


def test_block_retention_keeps_pending_tokens_causal_and_bounds_state():
    torch.manual_seed(115)
    bank = SetAssociativeLandmarkBank(1, 4, num_sets=1, ways=4, probe_sets=1,
        block_size=4, background_size=4, diversity_weight=0)
    initial_bytes = bank.state_bytes()
    keys = torch.randn(1, 1, 11, 4)
    values = torch.arange(11.).view(1, 1, 11, 1).expand_as(keys)
    query = torch.randn(1, 1, 4)
    for i in range(11):
        bank.update(keys[:, :, i], values[:, :, i], admission_bias=torch.tensor([[float(i)]]))
        actual, log_z = bank.read_attention(query)
        scores = query.unsqueeze(2) @ keys[:, :, :i+1].transpose(-1, -2) / 2
        expected = (scores.softmax(-1) @ values[:, :, :i+1]).squeeze(2)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(log_z, scores.logsumexp(-1).squeeze(2))
    assert bank._pending_count == 3
    assert bank._values[0, 0, 0, :, 0].tolist() == [4., 5., 6., 7.]
    for i in range(11, 50):
        bank.update(torch.zeros(1, 1, 4), torch.ones(1, 1, 4), admission_bias=torch.tensor([[float(i)]]))
    assert bank.state_bytes() == initial_bytes
    bank.reset_state(1)
    assert bank._pending_count == 0
    assert bank._background_count.item() == 0


def test_set_associative_bank_retains_multiple_records_in_same_set() -> None:
    torch.manual_seed(7)
    bank = SetAssociativeLandmarkBank(
        num_heads=1,
        head_dim=4,
        num_sets=1,
        ways=4,
        probe_sets=1,
    )
    keys = torch.eye(4).reshape(1, 1, 4, 4)
    values = (10.0 * torch.eye(4)).reshape(1, 1, 4, 4)
    for index in range(4):
        bank.update(
            keys[:, :, index],
            values[:, :, index],
            admission_bias=torch.full((1, 1), 100.0 + index),
        )
    for index in range(4):
        response, confidence = bank.read(keys[:, :, index], hard=True)
        torch.testing.assert_close(response, values[:, :, index], rtol=0, atol=0)
        assert float(confidence.item()) > 0.99


def test_set_associative_bank_state_is_context_length_independent() -> None:
    torch.manual_seed(8)
    bank = SetAssociativeLandmarkBank(
        num_heads=2,
        head_dim=8,
        num_sets=8,
        ways=2,
        probe_sets=2,
    )
    initial_bytes = bank.state_bytes()
    for _ in range(1000):
        key = torch.randn(1, 2, 8)
        value = torch.randn_like(key)
        bank.update(key, value)
    assert bank.state_bytes() == initial_bytes
    assert bank.state.keys.shape == (1, 2, 8, 2, 8)


@pytest.mark.parametrize("block_size", [1, 4])
def test_bfloat16_exact_storage_round_trips_bfloat16_sources(block_size) -> None:
    torch.manual_seed(228)
    kwargs = dict(num_heads=2, head_dim=8, num_sets=2, ways=block_size,
                  probe_sets=2, block_size=block_size, background_size=3,
                  diversity_weight=0)
    fp32 = SetAssociativeLandmarkBank(**kwargs, storage_dtype=torch.float32)
    bf16 = SetAssociativeLandmarkBank(**kwargs, storage_dtype=torch.bfloat16)
    fp32.reset_state(2)
    bf16.reset_state(2)
    with torch.no_grad():
        bf16.set_codes.copy_(fp32.set_codes)
        fp32.admission_vector.zero_()
        bf16.admission_vector.zero_()
        keys = torch.randn(2, 2, 43, 8, dtype=torch.bfloat16)
        values = torch.randn_like(keys)
        scores = 0.5 + 0.001 * torch.randn(2, 2, 43)
        query = torch.randn(2, 2, 5, 8, dtype=torch.bfloat16)
        for index in range(keys.shape[2]):
            for bank in (fp32, bf16):
                bank.update(keys[:, :, index], values[:, :, index],
                            admission_bias=scores[:, :, index])
            for name in ("_keys", "_values", "_scores", "_ages",
                         "_background_keys", "_background_values", "_background_count"):
                torch.testing.assert_close(getattr(bf16, name).float(),
                    getattr(fp32, name).float(), atol=0, rtol=0)
            fp32_out, fp32_z = fp32.read_attention(query)
            bf16_out, bf16_z = bf16.read_attention(query)
            torch.testing.assert_close(bf16_out, fp32_out, atol=0, rtol=0)
            torch.testing.assert_close(bf16_z, fp32_z, atol=0, rtol=0)
        assert bf16._background_count.max() > bf16.background_size
        assert bf16._scores.dtype == torch.float32
        assert bf16._keys.dtype == torch.bfloat16
        assert bf16.state_bytes() < fp32.state_bytes()


def test_bfloat16_exact_storage_keeps_replacement_scores_in_fp32() -> None:
    bank = SetAssociativeLandmarkBank(
        1,
        2,
        num_sets=1,
        ways=2,
        probe_sets=1,
        block_size=2,
        storage_dtype=torch.bfloat16,
    )
    old_key = torch.tensor([[[1.0, 0.0]]], dtype=torch.bfloat16)
    new_key = torch.tensor([[[0.0, 1.0]]], dtype=torch.bfloat16)
    with torch.no_grad():
        bank.update(old_key, old_key, admission_bias=torch.tensor([[0.5001]]))
        bank.update(old_key, old_key, admission_bias=torch.tensor([[0.5001]]))
        bank.update(new_key, new_key, admission_bias=torch.tensor([[0.5002]]))
        bank.update(new_key, new_key, admission_bias=torch.tensor([[0.5002]]))
    assert bank._keys.dtype == torch.bfloat16
    assert bank._scores.dtype == torch.float32
    assert bank._pending_scores.dtype == torch.float32
    assert bool((bank._keys == new_key.reshape(1, 1, 1, 2)).all(-1).any())
    assert float(bank._scores[torch.isfinite(bank._scores)][0]) == pytest.approx(0.5002)


def test_admission_bias_protects_salient_record_from_distractors() -> None:
    torch.manual_seed(9)
    bank = SetAssociativeLandmarkBank(
        num_heads=1,
        head_dim=8,
        num_sets=1,
        ways=2,
        probe_sets=1,
        diversity_weight=0.0,
    )
    needle_key = torch.randn(1, 1, 8)
    needle_value = torch.randn_like(needle_key)
    bank.update(needle_key, needle_value, admission_bias=torch.full((1, 1), 100.0))
    for _ in range(100):
        distractor = needle_key + 0.05 * torch.randn_like(needle_key)
        bank.update(
            distractor,
            torch.randn_like(distractor),
            admission_bias=torch.full((1, 1), -100.0),
        )
    response, confidence = bank.read(needle_key, hard=True)
    torch.testing.assert_close(response, needle_value, rtol=0, atol=0)
    assert float(confidence.item()) > 0.999


def test_probe_sets_recovers_second_best_routing_set() -> None:
    bank = SetAssociativeLandmarkBank(
        num_heads=1,
        head_dim=2,
        num_sets=2,
        ways=1,
        probe_sets=2,
    )
    with torch.no_grad():
        bank.set_codes.copy_(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
    key_a = torch.tensor([[[1.0, 0.0]]])
    value_a = torch.tensor([[[4.0, 5.0]]])
    key_b = torch.tensor([[[0.0, 1.0]]])
    value_b = torch.tensor([[[7.0, 8.0]]])
    bank.update(key_a, value_a, admission_bias=torch.full((1, 1), 10.0))
    bank.update(key_b, value_b, admission_bias=torch.full((1, 1), 10.0))
    query = torch.tensor([[[0.1, 1.0]]])
    response, _ = bank.read(query, hard=True)
    torch.testing.assert_close(response, value_b, rtol=0, atol=0)


def test_full_probe_bank_uses_capacity_across_sets() -> None:
    bank = SetAssociativeLandmarkBank(
        num_heads=1,
        head_dim=4,
        num_sets=4,
        ways=1,
        probe_sets=4,
        diversity_weight=0.0,
    )
    keys = torch.eye(4).reshape(1, 1, 4, 4)
    values = (10.0 * keys).clone()
    with torch.no_grad():
        bank.set_codes.zero_()
        bank.admission_vector.zero_()
    for index in range(4):
        bank.update(
            keys[:, :, index],
            values[:, :, index],
            admission_bias=torch.full((1, 1), 10.0 + index),
        )
    assert int(torch.isfinite(bank.state.scores).sum()) == 4
    for index in range(4):
        response, confidence = bank.read(keys[:, :, index], hard=True)
        torch.testing.assert_close(response, values[:, :, index], rtol=0, atol=0)
        assert float(confidence.item()) > 0.99


def test_write_mask_skips_low_salience_rows() -> None:
    torch.manual_seed(33)
    bank = SetAssociativeLandmarkBank(
        num_heads=1,
        head_dim=4,
        num_sets=1,
        ways=2,
        probe_sets=1,
    )
    with torch.no_grad():
        bank.admission_vector.zero_()
    key = torch.randn(2, 1, 4)
    value = torch.randn_like(key)
    bank.update(
        key,
        value,
        admission_bias=torch.tensor([[10.0], [-10.0]]),
        write_mask=torch.tensor([[True], [False]]),
    )
    assert torch.isfinite(bank.state.scores[0]).any()
    assert not torch.isfinite(bank.state.scores[1]).any()


def test_read_chunk_matches_individual_reads_for_frozen_state() -> None:
    torch.manual_seed(44)
    bank = SetAssociativeLandmarkBank(
        num_heads=2,
        head_dim=8,
        num_sets=4,
        ways=2,
        probe_sets=4,
    )
    with torch.no_grad():
        bank.admission_vector.zero_()
    for _ in range(5):
        key = torch.randn(1, 2, 8)
        bank.update(
            key,
            torch.randn_like(key),
            admission_bias=torch.full((1, 2), 10.0),
        )
    query = torch.randn(1, 2, 7, 8)
    block_out, block_conf = bank.read_chunk(query, hard=True)
    outputs = []
    confidences = []
    for index in range(query.shape[2]):
        output, confidence = bank.read(query[:, :, index], hard=True)
        outputs.append(output)
        confidences.append(confidence)
    torch.testing.assert_close(block_out, torch.stack(outputs, dim=2))
    torch.testing.assert_close(block_conf, torch.stack(confidences, dim=2))


def test_causal_weighted_bank_keeps_counted_identical_keys_exact() -> None:
    bank = CausalWeightedKVBank(1, 2, capacity=2)
    key = torch.tensor([[[1.0, 0.0]]])
    other = torch.tensor([[[0.0, 1.0]]])
    with torch.no_grad():
        for value in (0.0, 1.0, 0.0):
            bank.update(key, torch.full_like(key, value))
        bank.update(other, torch.full_like(other, 2.0))
    assert bank.state.counts[0, 0].tolist() == [3, 1]
    query = torch.tensor([[[4.0, 0.0]]])
    actual, log_z = bank.read_attention(query)
    keys = torch.cat((key.expand(1, 1, 3, 2), other.unsqueeze(2)), dim=2)
    values = torch.cat((torch.tensor([[[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]]),
                        torch.full((1, 1, 1, 2), 2.0)), dim=2)
    logits = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(2)
    expected = logits.softmax(-1) @ values
    expected_z = logits.logsumexp(-1)
    torch.testing.assert_close(actual, expected.squeeze(2))
    torch.testing.assert_close(log_z, expected_z.squeeze(2))


def test_causal_weighted_bank_is_chunk_invariant_and_suffix_independent() -> None:
    torch.manual_seed(221)
    keys = torch.randn(1, 1, 40, 4)
    values = torch.randn_like(keys)
    queries = torch.randn_like(keys)
    whole = CausalWeightedKVBank(1, 4, capacity=8)
    split = CausalWeightedKVBank(1, 4, capacity=8)
    with torch.no_grad():
        out_whole, z_whole = whole.update_read_chunk(keys, values, queries)
        first_out, first_z = split.update_read_chunk(keys[:, :, :17], values[:, :, :17], queries[:, :, :17])
        second_out, second_z = split.update_read_chunk(keys[:, :, 17:], values[:, :, 17:], queries[:, :, 17:])
    torch.testing.assert_close(out_whole, torch.cat((first_out, second_out), dim=2))
    torch.testing.assert_close(z_whole, torch.cat((first_z, second_z), dim=2))
    changed = keys.clone()
    changed[:, :, 20:] = torch.randn_like(changed[:, :, 20:])
    changed_values = values.clone()
    changed_values[:, :, 20:] = torch.randn_like(changed_values[:, :, 20:])
    prefix = CausalWeightedKVBank(1, 4, capacity=8)
    changed_bank = CausalWeightedKVBank(1, 4, capacity=8)
    with torch.no_grad():
        prefix_out, _ = prefix.update_read_chunk(keys[:, :, :20], values[:, :, :20], queries[:, :, :20])
        changed_out, _ = changed_bank.update_read_chunk(changed[:, :, :20], changed_values[:, :, :20], queries[:, :, :20])
    torch.testing.assert_close(prefix_out, changed_out)
    assert whole.state_bytes() == CausalWeightedKVBank(1, 4, capacity=8).state_bytes()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_kernels.TRITON_AVAILABLE,
    reason="CUDA and Triton required",
)
def test_triton_exact_read_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(45)
    bank = SetAssociativeLandmarkBank(
        num_heads=2,
        head_dim=8,
        num_sets=8,
        ways=2,
        probe_sets=3,
    ).cuda()
    bank.reset_state(1, device=torch.device("cuda"), dtype=torch.float16)
    for _ in range(6):
        key = torch.randn(1, 2, 8, device="cuda", dtype=torch.float16)
        bank.update(
            key,
            torch.randn_like(key),
            admission_bias=torch.full((1, 2), 10.0, device="cuda"),
        )
    query = torch.randn(1, 2, 11, 8, device="cuda", dtype=torch.float16)
    monkeypatch.setattr(triton_kernels, "TRITON_AVAILABLE", False)
    reference_out, reference_conf = bank.read_chunk(query, hard=True)
    monkeypatch.setattr(triton_kernels, "TRITON_AVAILABLE", True)
    fused_out, fused_conf = bank.read_chunk(query, hard=True)
    torch.testing.assert_close(fused_out, reference_out, rtol=0, atol=0)
    torch.testing.assert_close(fused_conf, reference_conf, rtol=0, atol=1e-5)
