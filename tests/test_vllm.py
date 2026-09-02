import pytest
import torch

from qcc_transformer import QCCVLLMBackend, QCCVLLMState


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
