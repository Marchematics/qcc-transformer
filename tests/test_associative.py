import torch

from qcc_transformer import SetAssociativeLandmarkBank


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
