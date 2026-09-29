"""Teacher-supervised targets for the hybrid exact-tier admission predictor."""
from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F


def sampled_future_attention_salience(
    query: Tensor,
    key: Tensor,
    *,
    window_size: int,
    num_queries: int = 128,
    topk: int = 8,
) -> Tensor:
    """Estimate which historical keys a Full-KV teacher retrieves later.

    Args:
        query, key: rotary-applied teacher tensors ``[batch, heads, tokens, dim]``.
        window_size: only keys already outside the exact local window are eligible.
        num_queries: number of future query positions sampled across the chunk.
        topk: for each sampled query, only its strongest historical attention weights
            contribute to salience. This makes the target focus on retrieval-critical
            events rather than diffuse background attention.

    Returns:
        ``[batch, heads, tokens]`` accumulated teacher attention mass.
    """

    if query.shape != key.shape or query.ndim != 4:
        raise ValueError("query/key must have shape [batch, heads, tokens, dim]")
    if window_size <= 0 or num_queries <= 0 or topk <= 0:
        raise ValueError("window_size, num_queries, and topk must be positive")
    batch, heads, tokens, dim = query.shape
    salience = torch.zeros(
        batch, heads, tokens, device=query.device, dtype=torch.float32
    )
    first_query = window_size
    if tokens <= first_query:
        return salience
    count = min(num_queries, tokens - first_query)
    positions = torch.linspace(
        first_query,
        tokens - 1,
        count,
        device=query.device,
        dtype=torch.float32,
    ).round().to(torch.long).unique(sorted=True)
    scale = 1.0 / math.sqrt(dim)
    for query_position in positions.tolist():
        # A key at ``query_position-window_size`` is the newest token no longer
        # represented by the exact local window.
        cutoff = query_position - window_size + 1
        if cutoff <= 0:
            continue
        score = torch.einsum(
            "bhd,bhkd->bhk",
            query[:, :, query_position].float(),
            key[:, :, :cutoff].float(),
        ) * scale
        probability = F.softmax(score, dim=-1)
        selected = min(topk, cutoff)
        weight, index = probability.topk(selected, dim=-1)
        salience[:, :, :cutoff].scatter_add_(2, index, weight)
    if positions.numel():
        salience.div_(float(positions.numel()))
    return salience


def salience_binary_labels(
    salience: Tensor,
    *,
    positive_fraction: float = 0.02,
    min_positive: int = 1,
) -> Tensor:
    """Convert teacher attention mass into a fixed-rate per-head admission target."""

    if salience.ndim != 3:
        raise ValueError("salience must have shape [batch, heads, tokens]")
    if not 0.0 < positive_fraction <= 1.0:
        raise ValueError("positive_fraction must lie in (0, 1]")
    if min_positive <= 0:
        raise ValueError("min_positive must be positive")
    tokens = salience.shape[-1]
    positives = min(tokens, max(min_positive, int(math.ceil(tokens * positive_fraction))))
    index = salience.topk(positives, dim=-1).indices
    labels = torch.zeros_like(salience, dtype=torch.float32)
    labels.scatter_(2, index, 1.0)
    return labels


def balanced_admission_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """BCE with an automatically balanced positive class weight."""

    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have the same shape")
    labels = labels.to(device=logits.device, dtype=torch.float32)
    positives = labels.sum().clamp_min(1.0)
    negatives = (labels.numel() - labels.sum()).clamp_min(1.0)
    pos_weight = (negatives / positives).detach()
    return F.binary_cross_entropy_with_logits(
        logits.float(), labels, pos_weight=pos_weight
    )


__all__ = [
    "balanced_admission_loss",
    "salience_binary_labels",
    "sampled_future_attention_salience",
]
