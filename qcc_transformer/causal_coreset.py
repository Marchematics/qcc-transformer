"""Causal fixed-capacity attention-response coreset.

This module contains a small reference implementation for the causal history
operator described in the quality study.  It keeps a disjoint set of weighted
KV representatives.  A representative stores a mean key, a mean value, and
the number of original tokens it represents.  Reads use the count as a
softmax multiplicity, so the state represents attention mass rather than a
second, independently-normalized response.

The implementation intentionally favours transparent semantics over a fused
serving kernel.  It is used as an opt-in diagnostic path until its merge rule
has been validated on real model traces.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CausalWeightedKVState:
    """A view of the bounded mutable coreset state."""

    keys: Tensor
    values: Tensor
    counts: Tensor
    radii: Tensor


class CausalWeightedKVBank(nn.Module):
    """Online Ward-merged KV representatives with causal, bounded state.

    Each event is committed before the corresponding query is read.  When the
    table is full, the least-cost pair among the existing representatives and
    the new event is merged using count-weighted key/value means.  The Ward
    cost is calculated in key space and the value is never used to choose a
    merge, which keeps the write rule independent of future queries.

    This is a correctness reference.  Its per-event pairwise merge is not a
    performance claim and should be replaced by a dedicated GPU kernel only
    after the causal semantics and quality have been measured.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        capacity: int,
        storage_dtype: torch.dtype = torch.float32,
        merge_policy: str = "ward",
        query_probe_capacity: int = 8,
        response_partition_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if min(num_heads, head_dim, capacity) <= 0:
            raise ValueError("num_heads, head_dim, and capacity must be positive")
        if storage_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("storage_dtype must be a floating-point torch dtype")
        if merge_policy not in ("ward", "response"):
            raise ValueError("merge_policy must be 'ward' or 'response'")
        if query_probe_capacity < 0:
            raise ValueError("query_probe_capacity must be nonnegative")
        if not math.isfinite(response_partition_weight) or response_partition_weight < 0:
            raise ValueError("response_partition_weight must be finite and nonnegative")
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.capacity = int(capacity)
        self.storage_dtype = storage_dtype
        self.merge_policy = merge_policy
        self.query_probe_capacity = int(query_probe_capacity)
        self.response_partition_weight = float(response_partition_weight)
        # The bank has no trainable routing parameters.  A tiny non-persistent
        # anchor keeps ``to(device)`` and state inspection conventional without
        # adding to the mutable request state.
        self.register_buffer("_anchor", torch.empty(0, dtype=torch.float32), persistent=False)
        self.reset_state(1)

    def reset_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        device = device or self._anchor.device
        dtype = self.storage_dtype
        shape = (batch_size, self.num_heads, self.capacity, self.head_dim)
        self._keys = torch.zeros(shape, device=device, dtype=dtype)
        self._values = torch.zeros_like(self._keys)
        self._counts = torch.zeros(
            batch_size, self.num_heads, self.capacity, device=device, dtype=torch.long
        )
        self._radii = torch.zeros(
            batch_size, self.num_heads, self.capacity, device=device, dtype=torch.float32
        )
        self._query_probes = torch.zeros(
            batch_size,
            self.num_heads,
            self.query_probe_capacity,
            self.head_dim,
            device=device,
            dtype=torch.float32,
        )
        self._query_probe_count = torch.zeros(
            batch_size, self.num_heads, device=device, dtype=torch.long
        )

    @property
    def state(self) -> CausalWeightedKVState:
        return CausalWeightedKVState(
            self._keys, self._values, self._counts, self._radii
        )

    @property
    def _pending_count(self) -> int:
        """Compatibility attribute used by diagnostic state inspectors."""

        return 0

    def state_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self._keys,
                self._values,
                self._counts,
                self._radii,
                self._query_probes,
                self._query_probe_count,
            )
        )

    def _ensure_state(self, key: Tensor) -> None:
        if (
            self._keys.shape[0] != key.shape[0]
            or self._keys.device != key.device
        ):
            self.reset_state(key.shape[0], device=key.device)

    @torch.no_grad()
    def _append_probe(self, query: Tensor | None, batch: int, head: int) -> None:
        if query is None or self.query_probe_capacity == 0:
            return
        count = int(self._query_probe_count[batch, head].item())
        if count < self.query_probe_capacity:
            self._query_probes[batch, head, count] = query.float()
            self._query_probe_count[batch, head] = count + 1
        else:
            self._query_probes[batch, head, :-1] = self._query_probes[batch, head, 1:].clone()
            self._query_probes[batch, head, -1] = query.float()

    @staticmethod
    def _ward_pair(
        keys: Tensor, counts: Tensor
    ) -> tuple[int, int]:
        diff = keys[:, None, :] - keys[None, :, :]
        distance = diff.square().sum(-1)
        mass = counts[:, None] * counts[None, :]
        cost = distance * mass / (counts[:, None] + counts[None, :])
        cost = cost.masked_fill(
            torch.tril(torch.ones_like(cost, dtype=torch.bool)), torch.inf
        )
        first, second = torch.where(cost == cost.min())
        return int(first[0].item()), int(second[0].item())

    def _response_pair(
        self,
        keys: Tensor,
        values: Tensor,
        counts: Tensor,
        probes: Tensor,
    ) -> tuple[int, int]:
        """Choose a merge that preserves responses on causal query probes."""

        if probes.numel() == 0:
            return self._ward_pair(keys, counts)
        n = keys.shape[0]
        pairs = torch.triu_indices(n, n, offset=1, device=keys.device)
        query = probes.float()
        logits = query @ keys.transpose(0, 1) / math.sqrt(self.head_dim)
        logits = logits + counts.clamp_min(1).log().unsqueeze(0)
        scale = logits.max(dim=-1, keepdim=True).values
        weights = torch.exp(logits - scale)
        total_z = weights.sum(-1)
        total_n = weights @ values
        reference = total_n / total_z.unsqueeze(-1).clamp_min(1e-12)
        best_cost = torch.tensor(torch.inf, device=keys.device)
        best_pair = (0, 1)
        # Keep pairwise response temporaries bounded for a diagnostic capacity
        # larger than a few dozen slots. The loop is still a reference path.
        pair_batch = 4096
        for start in range(0, pairs.shape[1], pair_batch):
            i = pairs[0, start : start + pair_batch]
            j = pairs[1, start : start + pair_batch]
            total_count = counts[i] + counts[j]
            merged_keys = (
                counts[i, None] * keys[i] + counts[j, None] * keys[j]
            ) / total_count[:, None]
            merged_values = (
                counts[i, None] * values[i] + counts[j, None] * values[j]
            ) / total_count[:, None]
            merged_logits = query @ merged_keys.transpose(0, 1) / math.sqrt(self.head_dim)
            merged_z = torch.exp(
                merged_logits + total_count.clamp_min(1).log().unsqueeze(0) - scale
            )
            pair_z = weights[:, i] + weights[:, j]
            pair_n = weights[:, i, None] * values[i] + weights[:, j, None] * values[j]
            approx_z = total_z[:, None] - pair_z + merged_z
            approx_n = total_n[:, None, :] - pair_n + merged_z[:, :, None] * merged_values
            approx = approx_n / approx_z[:, :, None].clamp_min(1e-12)
            output_error = (approx - reference[:, None, :]).square().mean(-1)
            partition_error = (
                torch.log(approx_z.clamp_min(1e-12)) - torch.log(total_z)[:, None]
            ).square()
            cost = output_error + self.response_partition_weight * partition_error
            candidate_cost, candidate_index = cost.mean(0).min(0)
            if bool(candidate_cost < best_cost):
                best_cost = candidate_cost
                offset = int(candidate_index.item())
                best_pair = (int(i[offset].item()), int(j[offset].item()))
        return best_pair

    def _insert_one(
        self,
        key: Tensor,
        value: Tensor,
        batch: int,
        head: int,
        query: Tensor | None = None,
    ) -> None:
        """Commit one event for one request/head."""

        self._append_probe(query, batch, head)
        counts = self._counts[batch, head]
        valid = counts > 0
        if bool(valid.any()):
            matching = (self._keys[batch, head].float() == key.float().view(1, -1)).all(-1)
            matching &= valid
            if bool(matching.any()):
                slot = int(matching.nonzero(as_tuple=False)[0].item())
                old_count = counts[slot].float()
                new_count = old_count + 1.0
                old_value = self._values[batch, head, slot].float()
                merged_value = (old_count * old_value + value.float()) / new_count
                self._values[batch, head, slot] = merged_value.to(self._values.dtype)
                self._counts[batch, head, slot] = new_count.to(torch.long)
                return
        if not bool(valid.all()):
            slot = int((~valid).nonzero(as_tuple=False)[0].item())
            self._keys[batch, head, slot] = key.to(self._keys.dtype)
            self._values[batch, head, slot] = value.to(self._values.dtype)
            self._counts[batch, head, slot] = 1
            self._radii[batch, head, slot] = 0.0
            return

        old_keys = self._keys[batch, head].float()
        old_values = self._values[batch, head].float()
        old_counts = counts.float()
        old_radii = self._radii[batch, head].float()
        candidate_keys = torch.cat((old_keys, key.float().view(1, -1)), dim=0)
        candidate_values = torch.cat((old_values, value.float().view(1, -1)), dim=0)
        candidate_counts = torch.cat(
            (old_counts, torch.ones(1, device=key.device, dtype=torch.float32)), dim=0
        )
        candidate_radii = torch.cat(
            (old_radii, torch.zeros(1, device=key.device, dtype=torch.float32)), dim=0
        )

        if self.merge_policy == "response":
            probe_count = int(self._query_probe_count[batch, head].item())
            i, j = self._response_pair(
                candidate_keys,
                candidate_values,
                candidate_counts,
                self._query_probes[batch, head, :probe_count],
            )
        else:
            i, j = self._ward_pair(candidate_keys, candidate_counts)
        total = candidate_counts[i] + candidate_counts[j]
        merged_key = (
            candidate_counts[i] * candidate_keys[i]
            + candidate_counts[j] * candidate_keys[j]
        ) / total
        merged_value = (
            candidate_counts[i] * candidate_values[i]
            + candidate_counts[j] * candidate_values[j]
        ) / total
        merged_radius = torch.maximum(
            candidate_radii[i] + (candidate_keys[i] - merged_key).norm(),
            candidate_radii[j] + (candidate_keys[j] - merged_key).norm(),
        )
        candidate_keys[i] = merged_key
        candidate_values[i] = merged_value
        candidate_counts[i] = total
        candidate_radii[i] = merged_radius
        keep = torch.ones(candidate_keys.shape[0], device=key.device, dtype=torch.bool)
        keep[j] = False
        self._keys[batch, head] = candidate_keys[keep].to(self._keys.dtype)
        self._values[batch, head] = candidate_values[keep].to(self._values.dtype)
        self._counts[batch, head] = candidate_counts[keep].round().to(torch.long)
        self._radii[batch, head] = candidate_radii[keep]

    @torch.no_grad()
    def update(self, key: Tensor, value: Tensor, *, query: Tensor | None = None) -> None:
        """Commit one causal KV event with shape ``[batch, heads, dim]``."""

        if key.ndim != 3 or key.shape != value.shape:
            raise ValueError("key and value must have shape [batch, heads, head_dim]")
        if key.shape[1:] != (self.num_heads, self.head_dim):
            raise ValueError("key shape does not match bank configuration")
        self._ensure_state(key)
        for batch in range(key.shape[0]):
            for head in range(self.num_heads):
                probe = None if query is None else query[batch, head]
                self._insert_one(key[batch, head], value[batch, head], batch, head, probe)

    def read_attention(self, query: Tensor) -> tuple[Tensor, Tensor]:
        """Read weighted representatives and return response plus log mass."""

        single = query.ndim == 3
        block = query.unsqueeze(2) if single else query
        if (
            block.ndim != 4
            or block.shape[1] != self.num_heads
            or block.shape[-1] != self.head_dim
        ):
            raise ValueError("query must have shape [batch, heads, (tokens,) head_dim]")
        if block.shape[0] != self._keys.shape[0] or block.device != self._keys.device:
            raise ValueError("query batch and device must match bank state")
        keys = self._keys.float()
        values = self._values.float()
        counts = self._counts
        valid = counts > 0
        logits = torch.einsum("bhtd,bhcd->bhtc", block.float(), keys)
        logits = logits / math.sqrt(self.head_dim)
        log_counts = torch.where(
            valid, counts.clamp_min(1).float().log(), torch.zeros_like(counts, dtype=torch.float32)
        )
        logits = logits + log_counts.unsqueeze(2)
        logits = logits.masked_fill(~valid.unsqueeze(2), -torch.inf)
        log_z = logits.logsumexp(-1)
        safe_z = torch.where(torch.isfinite(log_z), log_z, torch.zeros_like(log_z))
        weights = torch.exp(logits - safe_z.unsqueeze(-1))
        response = torch.einsum("bhtc,bhcd->bhtd", weights, values)
        response = torch.where(
            valid.any(-1).unsqueeze(2).unsqueeze(-1), response, torch.zeros_like(response)
        )
        response = response.to(block.dtype)
        if single:
            return response.squeeze(2), log_z.squeeze(2)
        return response, log_z

    @torch.no_grad()
    def update_read_chunk(
        self, key: Tensor, value: Tensor, query: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Commit events in order and read each query after its event."""

        if key.ndim != 4 or key.shape != value.shape or query.shape != key.shape:
            raise ValueError("key, value, and query must have shape [batch, heads, events, dim]")
        if key.shape[2] == 0:
            empty = query.new_empty(query.shape)
            return empty, query.new_empty(query.shape[:-1])
        outputs: list[Tensor] = []
        partitions: list[Tensor] = []
        for index in range(key.shape[2]):
            self.update(key[:, :, index], value[:, :, index], query=query[:, :, index])
            result, log_z = self.read_attention(query[:, :, index])
            outputs.append(result)
            partitions.append(log_z)
        return torch.stack(outputs, dim=2), torch.stack(partitions, dim=2)


__all__ = ["CausalWeightedKVBank", "CausalWeightedKVState"]
