"""Dependency-free primitive for a vLLM custom attention backend.

vLLM owns projection and scheduling; this module owns only the per-sequence
bounded state.  A backend can call :meth:`QCCVLLMState.forward` with projected
Q/K/V blocks and feed the returned head-major output to vLLM's output path.
The class deliberately does not import vLLM, so installing this package does
not pin a particular vLLM release.  It is a reference integration point, not
an assertion that the upstream vLLM ABI is stable.
"""

from __future__ import annotations

import math
import copy
from collections.abc import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .model import QCCArchive


class QCCVLLMState:
    """Per-batch state used by a vLLM custom attention implementation."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        window_size: int = 128,
        num_codes: int = 16,
        max_position_embeddings: int = 131_072,
        archive_mix: float = 0.125,
        use_triton: bool = True,
    ) -> None:
        if not 0.0 <= archive_mix <= 1.0:
            raise ValueError("archive_mix must lie in [0, 1]")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        # HF retrofit initializes its learned gate with ``gate_bias_init=2``;
        # sigmoid(2) gives a local-path weight of ≈0.881 and therefore an
        # archive contribution of ≈0.119.  Keep the standalone vLLM primitive
        # aligned with that conservative quality-first default instead of the
        # historical 50/50 ablation.  Callers can still override explicitly.
        self.archive_mix = archive_mix
        scales = 4
        horizons = torch.logspace(
            math.log10(max(1.0, float(window_size))),
            math.log10(max(float(window_size), float(max_position_embeddings))),
            scales,
        )
        rates = tuple(torch.exp(-math.log(2.0) / horizons).tolist())
        self.archive = QCCArchive(
            num_heads,
            head_dim,
            num_codes=num_codes,
            decay_rates=rates,
            window_size=window_size,
            use_triton=use_triton,
        )
        self._key_ring: Tensor | None = None
        self._value_ring: Tensor | None = None
        self._ring_start = 0
        self._ring_length = 0
        self._seen = 0

    def reset(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> None:
        self.archive.reset_state(batch_size, device=device, dtype=torch.float32)
        self._key_ring = torch.empty(
            batch_size, self.num_heads, self.window_size, self.head_dim,
            device=device, dtype=dtype,
        )
        self._value_ring = torch.empty_like(self._key_ring)
        self._ring_start = 0
        self._ring_length = 0
        self._seen = 0

    def _ordered_ring(self) -> tuple[Tensor, Tensor]:
        if self._key_ring is None or self._value_ring is None:
            raise RuntimeError("state has not been reset")
        if self._ring_length == 0:
            return self._key_ring[:, :, :0], self._value_ring[:, :, :0]
        end = self._ring_start + self._ring_length
        if end <= self.window_size:
            return (
                self._key_ring[:, :, self._ring_start:end],
                self._value_ring[:, :, self._ring_start:end],
            )
        first = self.window_size - self._ring_start
        return (
            torch.cat((self._key_ring[:, :, self._ring_start:], self._key_ring[:, :, : end % self.window_size]), dim=2),
            torch.cat((self._value_ring[:, :, self._ring_start:], self._value_ring[:, :, : end % self.window_size]), dim=2),
        )

    @torch.no_grad()
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Consume a projected block and return head-major attention output.

        Args:
            query, key, value: tensors shaped ``[batch, heads, tokens, dim]``.
                The block is assumed causal and keys/values are appended in
                order.  A caller handling paged batches should keep one state
                object per logical sequence or compact/reorder states together
                with its scheduler metadata.
        """

        if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
            raise ValueError("query/key/value must all have shape [batch, heads, tokens, dim]")
        if query.shape[1:] != (self.num_heads, query.shape[2], self.head_dim):
            raise ValueError("projected tensor shape does not match QCCVLLMState")
        if (
            self._key_ring is None
            or self.archive._numerator.shape[0] != query.shape[0]
            or self.archive._numerator.device != query.device
        ):
            self.reset(query.shape[0], device=query.device, dtype=query.dtype)
        old_k, old_v = self._ordered_ring()
        old_length = old_k.shape[2]
        combined_k = torch.cat((old_k, key), dim=2)
        combined_v = torch.cat((old_v, value), dim=2)
        length = query.shape[2]
        total_length = combined_k.shape[2]
        local = None
        if self.archive.use_triton and query.is_cuda:
            from .triton_kernels import TRITON_AVAILABLE, triton_local_chunk_attention

            if TRITON_AVAILABLE:
                # Reuse the exact one-launch sliding-window kernel used by the
                # HF/standalone serving path.  The key slice already contains
                # only the bounded ring plus this block, so no unbounded KV
                # allocation is introduced in the vLLM adapter.
                local = triton_local_chunk_attention(
                    query,
                    combined_k,
                    combined_v,
                    old_length=old_length,
                    window_size=self.window_size,
                )
        if local is None:
            positions = old_length + torch.arange(length, device=query.device)
            key_positions = torch.arange(total_length, device=query.device)
            valid = (key_positions[None, :] <= positions[:, None]) & (
                key_positions[None, :] >= positions[:, None] - self.window_size + 1
            )
            local = F.scaled_dot_product_attention(
                query.contiguous(), combined_k.contiguous(), combined_v.contiguous(),
                attn_mask=valid, dropout_p=0.0,
            )
        archive_out = torch.zeros_like(local)
        event_start = max(0, self.window_size - old_length)
        event_count = length - event_start
        if event_count > 0:
            self.archive.update_read_chunk(
                combined_k[:, :, :event_count],
                combined_v[:, :, :event_count],
                query[:, :, event_start:],
                output=archive_out[:, :, event_start:],
            )
        active = (self._seen + torch.arange(length, device=query.device) >= self.window_size).view(1, 1, length, 1)
        output = torch.where(
            active,
            (1.0 - self.archive_mix) * local + self.archive_mix * archive_out,
            local,
        )
        keep = min(self.window_size, total_length)
        assert self._key_ring is not None and self._value_ring is not None
        evicted = max(0, total_length - self.window_size)
        new_start = (self._ring_start + evicted) % self.window_size
        tail_k, tail_v = combined_k[:, :, -keep:], combined_v[:, :, -keep:]
        first = min(keep, self.window_size - new_start)
        self._key_ring[:, :, new_start : new_start + first] = tail_k[:, :, :first]
        self._value_ring[:, :, new_start : new_start + first] = tail_v[:, :, :first]
        if first < keep:
            remainder = keep - first
            self._key_ring[:, :, :remainder] = tail_k[:, :, first:]
            self._value_ring[:, :, :remainder] = tail_v[:, :, first:]
        self._ring_start = new_start
        self._ring_length = keep
        self._seen += length
        return output

    @property
    def seen_tokens(self) -> int:
        """Number of tokens consumed by this logical sequence."""

        return self._seen

    def fork(self) -> "QCCVLLMState":
        """Clone state for beam/speculative branches without sharing tensors."""

        return copy.deepcopy(self)


class QCCVLLMBackend:
    """Small scheduler-facing state registry for vLLM custom backends.

    vLLM versions expose different registration ABIs, so this class stays
    dependency-free and handles the stable part: one bounded state per
    logical request, explicit lifecycle, and beam/fork support.  A backend
    adapter can call ``forward(request_id, q, k, v)`` from its attention kernel.
    """

    def __init__(self, num_heads: int, head_dim: int, **state_kwargs: object) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.state_kwargs = dict(state_kwargs)
        self._states: dict[str, QCCVLLMState] = {}

    def reset(self, request_id: str, *, device: torch.device, dtype: torch.dtype) -> QCCVLLMState:
        state = QCCVLLMState(self.num_heads, self.head_dim, **self.state_kwargs)
        state.reset(1, device=device, dtype=dtype)
        self._states[request_id] = state
        return state

    def _state_forward(self, request_id: str, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        state = self._states.get(request_id)
        if state is None:
            state = self.reset(request_id, device=query.device, dtype=query.dtype)
        return state.forward(query, key, value)

    def forward(self, request_id: str, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Serve one logical request (the historical API)."""
        if query.ndim != 4 or query.shape[0] != 1:
            raise ValueError("QCCVLLMBackend.forward expects one [batch=1] logical request")
        return self._state_forward(request_id, query, key, value)

    def forward_batch(
        self,
        request_ids: Sequence[str],
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        """Serve equal-length blocks for several logical requests.

        vLLM schedulers commonly batch decode blocks.  The archive state is
        request-local, so processing each row independently is the safe
        reference implementation; a version-specific adapter can fuse these
        calls later without changing this contract.
        """
        if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
            raise ValueError("query/key/value must all have shape [batch, heads, tokens, dim]")
        ids = list(request_ids)
        if query.shape[0] != len(ids):
            raise ValueError("request_ids length must equal query batch size")
        outputs = [
            self._state_forward(request_id, query[index : index + 1], key[index : index + 1], value[index : index + 1])
            for index, request_id in enumerate(ids)
        ]
        return torch.cat(outputs, dim=0)

    def forward_ragged(
        self,
        request_ids: Sequence[str],
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_lens: Sequence[int],
    ) -> Tensor:
        """Serve flattened variable-length blocks and preserve row order.

        ``query/key/value`` are ``[sum(query_lens), heads, dim]``.  This is
        the layout used by vLLM's token scheduler before an attention backend
        reshapes per-request rows.  The returned tensor has the same layout.
        """
        if query.ndim != 3 or key.shape != query.shape or value.shape != query.shape:
            raise ValueError("ragged query/key/value must have shape [tokens, heads, dim]")
        ids = list(request_ids)
        lens = [int(length) for length in query_lens]
        if len(ids) != len(lens) or any(length <= 0 for length in lens):
            raise ValueError("request_ids and query_lens must have equal positive entries")
        if sum(lens) != query.shape[0]:
            raise ValueError("sum(query_lens) must equal flattened token count")
        outputs = []
        offset = 0
        for request_id, length in zip(ids, lens):
            end = offset + length
            q = query[offset:end].transpose(0, 1).unsqueeze(0)
            k = key[offset:end].transpose(0, 1).unsqueeze(0)
            v = value[offset:end].transpose(0, 1).unsqueeze(0)
            outputs.append(self._state_forward(request_id, q, k, v).squeeze(0).transpose(0, 1))
            offset = end
        return torch.cat(outputs, dim=0)

    def fork(self, source_id: str, target_id: str) -> None:
        if source_id not in self._states:
            raise KeyError(f"unknown source request: {source_id}")
        self._states[target_id] = self._states[source_id].fork()

    def drop(self, request_id: str) -> None:
        self._states.pop(request_id, None)


__all__ = ["QCCVLLMBackend", "QCCVLLMState"]
