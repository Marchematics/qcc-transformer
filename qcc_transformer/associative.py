"""Set-associative exact landmark memory for QCC research experiments.

This module is intentionally independent from the recurrent response archive so it can
be benchmarked before being wired into every HF/vLLM path. Persistent memory is
O(heads * sets * ways * head_dim) and therefore independent of context length.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class AssociativeLandmarkState:
    keys: Tensor
    values: Tensor
    scores: Tensor
    ages: Tensor


class SetAssociativeLandmarkBank(nn.Module):
    """Small exact-KV bank with learned set routing and multi-way replacement.

    Each incoming key is routed to the strongest learned set. Every set has ``ways``
    exact slots; replacements are controlled by a learned admission score plus a small
    diversity bonus. Reads probe the top ``probe_sets`` sets and perform exact
    query/key matching inside their ways.

    The learned admission projection is deliberately tiny (one vector per head) so a
    pretrained retrofit can tune what deserves exact retention without training the
    backbone. It is a selection mechanism, not a claim that arbitrary random
    associations can be compressed losslessly.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        num_sets: int = 64,
        ways: int = 4,
        probe_sets: int = 2,
        admission_weight: float = 1.0,
        diversity_weight: float = 0.25,
        recency_weight: float = 0.0,
        temperature: float = 8.0,
    ) -> None:
        super().__init__()
        if min(num_heads, head_dim, num_sets, ways, probe_sets) <= 0:
            raise ValueError("head dimensions, sets, ways, and probes must be positive")
        if probe_sets > num_sets:
            raise ValueError("probe_sets cannot exceed num_sets")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_sets = num_sets
        self.ways = ways
        self.probe_sets = probe_sets
        self.admission_weight = admission_weight
        self.diversity_weight = diversity_weight
        self.recency_weight = recency_weight
        self.temperature = temperature

        scale = 1.0 / math.sqrt(head_dim)
        self.set_codes = nn.Parameter(torch.randn(num_heads, num_sets, head_dim) * scale)
        self.admission_vector = nn.Parameter(torch.randn(num_heads, head_dim) * scale)
        self.reset_state(1)

    def reset_state(self, batch_size: int, *, device=None, dtype=None) -> None:
        device = device or self.set_codes.device
        dtype = dtype if dtype in (torch.float32, torch.float64) else torch.float32
        shape = (batch_size, self.num_heads, self.num_sets, self.ways)
        self._keys = torch.zeros(*shape, self.head_dim, device=device, dtype=dtype)
        self._values = torch.zeros_like(self._keys)
        self._scores = torch.full(shape, -torch.inf, device=device, dtype=dtype)
        self._ages = torch.zeros(shape, device=device, dtype=torch.long)
        self._step = 0

    @property
    def state(self) -> AssociativeLandmarkState:
        return AssociativeLandmarkState(self._keys, self._values, self._scores, self._ages)

    def state_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self._keys, self._values, self._scores, self._ages)
        )

    def _ensure_state(self, key: Tensor) -> None:
        if self._keys.shape[0] != key.shape[0] or self._keys.device != key.device:
            self.reset_state(key.shape[0], device=key.device, dtype=key.dtype)

    def _set_logits(self, x: Tensor) -> Tensor:
        return torch.einsum(
            "bhd,hmd->bhm",
            x.to(self._keys.dtype),
            self.set_codes.to(x.device, self._keys.dtype),
        ) / math.sqrt(self.head_dim)

    @torch.no_grad()
    def update(
        self,
        key: Tensor,
        value: Tensor,
        *,
        admission_bias: Tensor | None = None,
    ) -> None:
        """Insert one exact association per batch/head.

        ``admission_bias`` may be supplied by a caller that has a task/model-specific
        salience signal. This is the hook used by future HF distillation: the generic
        bank itself never relies on marker IDs or benchmark labels.
        """
        if key.shape != value.shape or key.ndim != 3:
            raise ValueError("key and value must have shape [batch, heads, head_dim]")
        if key.shape[1:] != (self.num_heads, self.head_dim):
            raise ValueError("key shape does not match bank configuration")
        self._ensure_state(key)
        self._step += 1

        set_logits = self._set_logits(key)
        set_index = set_logits.argmax(dim=-1)
        batch_index = torch.arange(key.shape[0], device=key.device)[:, None]
        head_index = torch.arange(self.num_heads, device=key.device)[None, :]
        slots_key = self._keys[batch_index, head_index, set_index]
        slots_score = self._scores[batch_index, head_index, set_index]
        valid = torch.isfinite(slots_score)

        normalized_key = F.normalize(key.to(self._keys.dtype), dim=-1)
        normalized_slots = F.normalize(slots_key, dim=-1)
        similarity = torch.einsum("bhd,bhwd->bhw", normalized_key, normalized_slots)
        max_similarity = torch.where(
            valid, similarity, torch.full_like(similarity, -1.0)
        ).max(-1).values
        diversity = 1.0 - max_similarity

        admission = torch.einsum(
            "bhd,hd->bh",
            key.to(self._keys.dtype),
            self.admission_vector.to(key.device, self._keys.dtype),
        ) / math.sqrt(self.head_dim)
        if admission_bias is not None:
            if admission_bias.shape != admission.shape:
                raise ValueError("admission_bias must have shape [batch, heads]")
            admission = admission + admission_bias.to(admission.dtype)
        candidate_score = (
            self.admission_weight * admission + self.diversity_weight * diversity
        )

        empty = ~valid
        has_empty = empty.any(dim=-1)
        empty_index = empty.to(torch.int64).argmax(dim=-1)
        age = (self._step - self._ages[batch_index, head_index, set_index]).to(
            self._scores.dtype
        )
        replacement_score = slots_score - self.recency_weight / (1.0 + age)
        weakest_index = replacement_score.argmin(dim=-1)
        way_index = torch.where(has_empty, empty_index, weakest_index)
        weakest_score = replacement_score.gather(
            -1, weakest_index.unsqueeze(-1)
        ).squeeze(-1)
        should_write = has_empty | (candidate_score > weakest_score)

        write_batch = batch_index.expand_as(set_index)[should_write]
        write_head = head_index.expand_as(set_index)[should_write]
        write_set = set_index[should_write]
        write_way = way_index[should_write]
        self._keys[write_batch, write_head, write_set, write_way] = key.to(
            self._keys.dtype
        )[should_write]
        self._values[write_batch, write_head, write_set, write_way] = value.to(
            self._values.dtype
        )[should_write]
        self._scores[write_batch, write_head, write_set, write_way] = candidate_score[
            should_write
        ]
        self._ages[write_batch, write_head, write_set, write_way] = self._step

    def read(self, query: Tensor, *, hard: bool = False) -> tuple[Tensor, Tensor]:
        """Read top routed sets; return response and best cosine-match confidence."""
        if query.ndim != 3 or query.shape[1:] != (self.num_heads, self.head_dim):
            raise ValueError("query must have shape [batch, heads, head_dim]")
        self._ensure_state(query)
        set_logits = self._set_logits(query)
        probe = set_logits.topk(self.probe_sets, dim=-1).indices
        batch_index = torch.arange(query.shape[0], device=query.device)[:, None, None]
        head_index = torch.arange(self.num_heads, device=query.device)[None, :, None]
        keys = self._keys[batch_index, head_index, probe].reshape(
            query.shape[0], self.num_heads, -1, self.head_dim
        )
        values = self._values[batch_index, head_index, probe].reshape_as(keys)
        scores = self._scores[batch_index, head_index, probe].reshape(
            query.shape[0], self.num_heads, -1
        )
        valid = torch.isfinite(scores)

        normalized_query = F.normalize(query.to(keys.dtype), dim=-1)
        normalized_keys = F.normalize(keys, dim=-1)
        similarity = torch.einsum(
            "bhd,bhnd->bhn", normalized_query, normalized_keys
        )
        similarity = torch.where(
            valid, similarity, torch.full_like(similarity, -1.0e9)
        )
        confidence, best = similarity.max(dim=-1)
        if hard:
            response = values.gather(
                2, best[..., None, None].expand(-1, -1, 1, self.head_dim)
            ).squeeze(2)
        else:
            weights = F.softmax(similarity * self.temperature, dim=-1).to(values.dtype)
            response = torch.einsum("bhn,bhnd->bhd", weights, values)
        any_valid = valid.any(dim=-1)
        response = torch.where(
            any_valid.unsqueeze(-1), response, torch.zeros_like(response)
        )
        confidence = torch.where(
            any_valid, confidence, torch.full_like(confidence, -1.0)
        )
        return response.to(query.dtype), confidence
