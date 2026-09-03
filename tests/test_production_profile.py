import torch
from torch import nn

from qcc_transformer.model import QCCSelfAttention
from qcc_transformer.production_profile import (
    deployment_profile_matches,
    enable_qkv_only_deployment_profile,
)


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.qcc = QCCSelfAttention(16, 4, window_size=4, num_codes=2, use_triton=False)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = Wrapper()
        self.b = Wrapper()


def test_qkv_only_profile_makes_hf_gate_constant_and_frozen():
    model = Model()
    report = enable_qkv_only_deployment_profile(model, archive_mix=0.125)
    assert report.layers == 2
    assert deployment_profile_matches(model, 0.125)
    hidden = torch.randn(2, 3, 16)
    for wrapper in (model.a, model.b):
        gate = torch.sigmoid(wrapper.qcc.gate(hidden))
        torch.testing.assert_close(gate, torch.full_like(gate, 0.875), atol=1e-6, rtol=0)
        assert all(not p.requires_grad for p in wrapper.qcc.gate.parameters())
