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
        archive_mix: float = 0.5,
        use_triton: bool = True,
    ) -> None:
        if not 0.0 <= archive_mix <= 1.0:
            raise ValueError("archive_mix must lie in [0, 1]")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
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
        self._keys: list[Tensor] = []
        self._values: list[Tensor] = []
        self._seen = 0

    def reset(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> None:
        self.archive.reset_state(batch_size, device=device, dtype=torch.float32)
        self._keys.clear()
        self._values.clear()
        self._seen = 0

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
        if self.archive._numerator.shape[0] != query.shape[0] or self.archive._numerator.device != query.device:
            self.reset(query.shape[0], device=query.device, dtype=query.dtype)
        outputs: list[Tensor] = []
        scale = self.head_dim**-0.5
        for index in range(query.shape[2]):
            kt = key[:, :, index]
            vt = value[:, :, index]
            if len(self._keys) >= self.window_size:
                self.archive.update(self._keys.pop(0), self._values.pop(0))
            self._keys.append(kt)
            self._values.append(vt)
            local_k = torch.stack(self._keys, dim=2)
            local_v = torch.stack(self._values, dim=2)
            local = F.scaled_dot_product_attention(
                query[:, :, index : index + 1], local_k, local_v, dropout_p=0.0
            ).squeeze(2)
            if self._seen >= self.window_size:
                remote = self.archive.read(query[:, :, index])
                local = (1.0 - self.archive_mix) * local + self.archive_mix * remote
            outputs.append(local)
            self._seen += 1
        return torch.stack(outputs, dim=2)

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

    def forward(self, request_id: str, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        if query.ndim != 4 or query.shape[0] != 1:
            raise ValueError("QCCVLLMBackend expects one [batch=1] logical request")
        state = self._states.get(request_id)
        if state is None:
            state = self.reset(request_id, device=query.device, dtype=query.dtype)
        return state.forward(query, key, value)

    def fork(self, source_id: str, target_id: str) -> None:
        if source_id not in self._states:
            raise KeyError(f"unknown source request: {source_id}")
        self._states[target_id] = self._states[source_id].fork()

    def drop(self, request_id: str) -> None:
        self._states.pop(request_id, None)


__all__ = ["QCCVLLMBackend", "QCCVLLMState"]
