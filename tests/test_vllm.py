import pytest
import torch

from qcc_transformer import QCCVLLMState


def test_vllm_state_consumes_projected_block():
    state = QCCVLLMState(2, 4, window_size=3, num_codes=4, use_triton=False)
    query = torch.randn(1, 2, 5, 4)
    output = state.forward(query, query, query)
    assert output.shape == query.shape
    assert state._seen == 5


def test_vllm_state_rejects_wrong_shape():
    state = QCCVLLMState(2, 4, window_size=3, num_codes=4, use_triton=False)
    with pytest.raises(ValueError, match="shape"):
        state.forward(torch.randn(1, 2, 4), torch.randn(1, 2, 4), torch.randn(1, 2, 4))
