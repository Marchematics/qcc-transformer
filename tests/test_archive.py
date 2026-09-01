import torch

from qcc_transformer import QCCArchive, QCCForCausalLM


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
