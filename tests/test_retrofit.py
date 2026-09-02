from types import SimpleNamespace

import pytest
import torch
from torch import nn

from qcc_transformer.retrofit import (
    load_retrofit_adapter,
    patch_hf_model,
    retrofit_adapter_state,
)


class _Attention(nn.Module):
    def __init__(self, *, kv_heads: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.k_proj = nn.Linear(16, 16, bias=False)
        self.v_proj = nn.Linear(16, 16, bias=False)
        self.o_proj = nn.Linear(16, 16, bias=False)
        self.num_heads = 4
        self.num_key_value_heads = kv_heads


class _Model(nn.Module):
    def __init__(self, *, kv_heads: int = 4):
        super().__init__()
        self.config = SimpleNamespace(
            max_position_embeddings=64,
            rope_theta=None,
            num_attention_heads=4,
            num_key_value_heads=kv_heads,
        )
        self.attn = _Attention(kv_heads=kv_heads)


def test_patch_hf_attention_preserves_prefill_and_decode_shapes():
    model = _Model()
    assert patch_hf_model(model, window_size=4, num_codes=4, use_triton=False) == ["attn"]
    output, _, cache = model.attn(torch.randn(2, 8, 16), use_cache=True)
    assert output.shape == (2, 8, 16)
    assert cache is not None and cache.get_seq_length() == 8
    output, _, cache = model.attn(
        torch.randn(2, 1, 16), past_key_value=cache, use_cache=True
    )
    assert output.shape == (2, 1, 16)
    assert cache.get_seq_length() == 9


def test_patch_hf_supports_modern_shared_cache_call():
    model = _Model()
    patch_hf_model(model, window_size=4, num_codes=4, use_triton=False)
    output = model.attn(
        torch.randn(1, 4, 16), past_key_values=object(), use_cache=True
    )
    assert len(output) == 2
    assert output[0].shape == (1, 4, 16)


def test_patch_hf_exposes_differentiable_calibration_path():
    model = _Model()
    patch_hf_model(model, window_size=4, num_codes=4, use_triton=False)
    model.attn.train()
    output, _, _ = model.attn(torch.randn(1, 8, 16), use_cache=False)
    output.sum().backward()
    assert model.attn.qcc.archive.codes.grad is not None


def test_retrofit_adapter_round_trip(tmp_path):
    source = _Model()
    patch_hf_model(source, window_size=4, num_codes=4, use_triton=False)
    state = retrofit_adapter_state(source)
    path = tmp_path / "adapter.pt"
    torch.save({"state_dict": state}, path)
    target = _Model()
    replaced = load_retrofit_adapter(
        target, path, window_size=4, num_codes=4, use_triton=False
    )
    assert replaced == ["attn"]
    for name, value in state.items():
        assert torch.equal(value, dict(target.named_parameters())[name].cpu())


def test_patch_hf_rejects_grouped_query_attention():
    with pytest.raises(ValueError, match="GQA/MQA"):
        patch_hf_model(_Model(kv_heads=2), use_triton=False)
