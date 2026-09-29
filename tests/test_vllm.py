import pytest
import torch

from qcc_transformer import QCCVLLMBackend, QCCVLLMState
from qcc_transformer.triton_kernels import TRITON_AVAILABLE


@pytest.mark.skipif(not TRITON_AVAILABLE or not torch.cuda.is_available(), reason="Triton CUDA unavailable")
def test_vllm_triton_block_matches_reference() -> None:
    torch.manual_seed(17)
    device = torch.device("cuda")
    state = QCCVLLMState(2, 8, window_size=8, num_codes=8, use_triton=True)
    reference = QCCVLLMState(2, 8, window_size=8, num_codes=8, use_triton=False)
    reference.archive.load_state_dict(state.archive.state_dict())
    state.reset(1, device=device, dtype=torch.float32)
    reference.reset(1, device=device, dtype=torch.float32)
    query = torch.randn(1, 2, 33, 8, device=device)
    with torch.no_grad():
        # Split to exercise both a partially filled ring and a wrapped ring.
        actual = torch.cat((state.forward(query[:, :, :13], query[:, :, :13], query[:, :, :13]),
                            state.forward(query[:, :, 13:], query[:, :, 13:], query[:, :, 13:])), dim=2)
        expected = torch.cat((reference.forward(query[:, :, :13], query[:, :, :13], query[:, :, :13]),
                              reference.forward(query[:, :, 13:], query[:, :, 13:], query[:, :, 13:])), dim=2)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


def test_vllm_state_consumes_projected_block():
    state = QCCVLLMState(2, 4, window_size=3, num_codes=4, use_triton=False)
    query = torch.randn(1, 2, 5, 4)
    output = state.forward(query, query, query)
    assert output.shape == query.shape
    assert state._seen == 5


def test_vllm_block_matches_tokenized_execution():
    torch.manual_seed(0)
    state = QCCVLLMState(2, 4, window_size=3, num_codes=4, use_triton=False)
    tokenized = state.fork()
    query = torch.randn(1, 2, 12, 4)
    block = state.forward(query, query, query)
    pieces = [
        tokenized.forward(query[:, :, i : i + 1], query[:, :, i : i + 1], query[:, :, i : i + 1])
        for i in range(query.shape[2])
    ]
    assert torch.allclose(block, torch.cat(pieces, dim=2), atol=1e-5, rtol=1e-5)


def test_vllm_state_rejects_wrong_shape():
    state = QCCVLLMState(2, 4, window_size=3, num_codes=4, use_triton=False)
    with pytest.raises(ValueError, match="shape"):
        state.forward(torch.randn(1, 2, 4), torch.randn(1, 2, 4), torch.randn(1, 2, 4))


def test_vllm_backend_tracks_and_forks_logical_requests():
    backend = QCCVLLMBackend(2, 4, window_size=3, num_codes=4, use_triton=False)
    query = torch.randn(1, 2, 2, 4)
    assert backend.forward("a", query, query, query).shape == query.shape
    backend.fork("a", "b")
    assert backend._states["a"].seen_tokens == backend._states["b"].seen_tokens == 2
    backend.drop("a")
    assert "a" not in backend._states


def test_vllm_quality_first_archive_mix_default():
    state = QCCVLLMState(2, 4, window_size=3, num_codes=4, use_triton=False)
    assert state.archive_mix == pytest.approx(0.125)
    historical = QCCVLLMState(
        2, 4, window_size=3, num_codes=4, archive_mix=0.5, use_triton=False
    )
    assert historical.archive_mix == pytest.approx(0.5)


def test_vllm_backend_forward_batch_isolates_request_state():
    torch.manual_seed(4)
    backend = QCCVLLMBackend(2, 4, window_size=3, num_codes=4, use_triton=False)
    query = torch.randn(2, 2, 5, 4)
    batched = backend.forward_batch(["a", "b"], query, query, query)
    single_a = QCCVLLMState(2, 4, window_size=3, num_codes=4, use_triton=False)
    single_b = QCCVLLMState(2, 4, window_size=3, num_codes=4, use_triton=False)
    single_a.archive.load_state_dict(backend._states["a"].archive.state_dict())
    single_b.archive.load_state_dict(backend._states["b"].archive.state_dict())
    expected = torch.cat(
        [single_a.forward(query[:1], query[:1], query[:1]),
         single_b.forward(query[1:], query[1:], query[1:])],
        dim=0,
    )
    torch.testing.assert_close(batched, expected)


def test_vllm_backend_forward_ragged_preserves_flattened_layout():
    torch.manual_seed(5)
    backend = QCCVLLMBackend(2, 4, window_size=3, num_codes=4, use_triton=False)
    query = torch.randn(5, 2, 4)
    output = backend.forward_ragged(["a", "b"], query, query, query, [2, 3])
    assert output.shape == query.shape
    assert backend._states["a"].seen_tokens == 2
    assert backend._states["b"].seen_tokens == 3
