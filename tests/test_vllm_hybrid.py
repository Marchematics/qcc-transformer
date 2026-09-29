import torch

from qcc_transformer.vllm_hybrid import (
    HybridQCCVLLMBackend,
    HybridQCCVLLMState,
)


def test_hybrid_vllm_state_bytes_do_not_grow_with_context() -> None:
    torch.manual_seed(81)
    state = HybridQCCVLLMState(
        2,
        4,
        window_size=8,
        num_codes=4,
        max_position_embeddings=1_000_000,
        use_triton=False,
        hybrid_kwargs={"exact_num_sets": 4, "exact_ways": 2},
    )
    state.reset(1, device=torch.device("cpu"), dtype=torch.float32)
    initial = state.state_bytes()
    for _ in range(50):
        q = torch.randn(1, 2, 7, 4)
        state.forward(q, q, q)
    assert state.seen_tokens == 350
    assert state.state_bytes() == initial


def test_hybrid_vllm_backend_fork_is_independent() -> None:
    torch.manual_seed(82)
    backend = HybridQCCVLLMBackend(
        1,
        4,
        window_size=4,
        num_codes=2,
        use_triton=False,
        hybrid_kwargs={"exact_num_sets": 2, "exact_ways": 2},
    )
    block = torch.randn(1, 1, 6, 4)
    backend.forward("root", block, block, block)
    backend.fork("root", "branch")
    root_before = backend.get_state("root").archive.state.numerator.clone()
    branch_block = torch.randn(1, 1, 3, 4)
    backend.forward("branch", branch_block, branch_block, branch_block)
    torch.testing.assert_close(
        backend.get_state("root").archive.state.numerator, root_before
    )
    assert backend.get_state("branch").seen_tokens == 9
    assert backend.get_state("root").seen_tokens == 6


def test_hybrid_vllm_ragged_scheduler_preserves_layout_and_request_state() -> None:
    torch.manual_seed(83)
    backend = HybridQCCVLLMBackend(
        2,
        4,
        window_size=4,
        num_codes=2,
        use_triton=False,
        hybrid_kwargs={"exact_num_sets": 2, "exact_ways": 2},
    )
    query = torch.randn(5, 2, 4)
    output = backend.forward_ragged(
        ["a", "b"], query, query, query, [2, 3]
    )
    assert output.shape == query.shape
    assert backend.get_state("a").seen_tokens == 2
    assert backend.get_state("b").seen_tokens == 3
    backend.drop("a")
    try:
        backend.get_state("a")
    except KeyError:
        pass
    else:
        raise AssertionError("drop must remove request state")


def test_hybrid_vllm_calibrated_parameters_survive_request_reset() -> None:
    backend = HybridQCCVLLMBackend(
        1,
        4,
        window_size=4,
        num_codes=2,
        use_triton=False,
        hybrid_kwargs={"exact_num_sets": 2, "exact_ways": 2},
    )
    state = backend.reset(
        "x", device=torch.device("cpu"), dtype=torch.float32
    )
    with torch.no_grad():
        state.archive.admission.bias.fill_(3.25)
    state.reset(1, device=torch.device("cpu"), dtype=torch.float32)
    torch.testing.assert_close(
        state.archive.admission.bias, torch.full_like(state.archive.admission.bias, 3.25)
    )
