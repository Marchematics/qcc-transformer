"""Matched HF/stock-vLLM deployment policy for QCC."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .model import QCCSelfAttention


@dataclass(frozen=True)
class DeploymentProfileReport:
    layers: int
    archive_mix: float
    frozen_gate_parameters: int


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly inside (0, 1)")
    return math.log(probability / (1.0 - probability))


def enable_qkv_only_deployment_profile(
    model: nn.Module,
    *,
    archive_mix: float = 0.125,
    freeze_gate: bool = True,
) -> DeploymentProfileReport:
    """Make HF local/archive mixing exactly reproducible by a Q/K/V-only backend.

    ``QCCSelfAttention`` historically predicts the local weight from pre-attention
    hidden states. vLLM's ``AttentionImpl`` receives only projected Q/K/V, so that
    hidden-conditioned gate cannot be reproduced without model-specific business-code
    edits. Production instead uses a fixed calibrated remote weight and realizes the
    same policy in HF by zeroing the gate weight and setting a constant bias.
    """
    if not 0.0 <= archive_mix <= 1.0:
        raise ValueError("archive_mix must lie in [0, 1]")
    eps = 1e-6
    local_weight = min(max(1.0 - archive_mix, eps), 1.0 - eps)
    bias = _logit(local_weight)
    layers = 0
    frozen = 0
    for module in model.modules():
        qcc = getattr(module, "qcc", None)
        if not isinstance(qcc, QCCSelfAttention):
            continue
        with torch.no_grad():
            qcc.gate.weight.zero_()
            qcc.gate.bias.fill_(bias)
        if freeze_gate:
            for parameter in qcc.gate.parameters():
                if parameter.requires_grad:
                    frozen += parameter.numel()
                parameter.requires_grad_(False)
        qcc._qcc_deployment_archive_mix = float(archive_mix)
        layers += 1
    if layers == 0:
        raise ValueError("model has no patched QCC attention layers")
    return DeploymentProfileReport(layers, float(archive_mix), frozen)


def deployment_profile_matches(model: nn.Module, archive_mix: float, *, atol: float = 1e-6) -> bool:
    """Audit that every QCC gate is constant and equals the stock-vLLM mix."""
    if not 0.0 <= archive_mix <= 1.0:
        return False
    expected_local = 1.0 - archive_mix
    found = 0
    for module in model.modules():
        qcc = getattr(module, "qcc", None)
        if not isinstance(qcc, QCCSelfAttention):
            continue
        found += 1
        if torch.count_nonzero(qcc.gate.weight.detach()).item() != 0:
            return False
        local = torch.sigmoid(qcc.gate.bias.detach().float())
        if not torch.allclose(local, torch.full_like(local, expected_local), atol=atol, rtol=0):
            return False
    return found > 0


__all__ = [
    "DeploymentProfileReport",
    "deployment_profile_matches",
    "enable_qkv_only_deployment_profile",
]
