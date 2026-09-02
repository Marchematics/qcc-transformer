"""Dependency-free scheduler state for the hybrid QCC archive.

This module mirrors :mod:`qcc_transformer.vllm` but upgrades the recurrent archive to
:class:`HybridQCCArchive`. It still does not claim upstream vLLM registration; its job
is to make the request lifecycle/state contract identical between HF and the eventual
stock-vLLM adapter.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from .hybrid_archive import HybridQCCArchive
from .vllm import QCCVLLMBackend, QCCVLLMState


class HybridQCCVLLMState(QCCVLLMState):
    """One logical request with local ring + recurrent + exact associative state."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        hybrid_kwargs: dict[str, object] | None = None,
        **state_kwargs: object,
    ) -> None:
        super().__init__(num_heads, head_dim, **state_kwargs)
        self.hybrid_kwargs = dict(hybrid_kwargs or {})
        self.archive = HybridQCCArchive.from_archive(
            self.archive, **self.hybrid_kwargs
        )

    def state_bytes(self) -> int:
        """Mutable attention-state bytes for this logical request."""

        local = 0
        if self._key_ring is not None:
            local += self._key_ring.numel() * self._key_ring.element_size()
        if self._value_ring is not None:
            local += self._value_ring.numel() * self._value_ring.element_size()
        return local + self.archive.total_state_bytes()


class HybridQCCVLLMBackend(QCCVLLMBackend):
    """Scheduler registry whose requests all use :class:`HybridQCCVLLMState`."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        hybrid_kwargs: dict[str, object] | None = None,
        **state_kwargs: object,
    ) -> None:
        super().__init__(num_heads, head_dim, **state_kwargs)
        self.hybrid_kwargs = dict(hybrid_kwargs or {})

    def reset(
        self,
        request_id: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> HybridQCCVLLMState:
        state = HybridQCCVLLMState(
            self.num_heads,
            self.head_dim,
            hybrid_kwargs=self.hybrid_kwargs,
            **self.state_kwargs,
        )
        state.reset(1, device=device, dtype=dtype)
        self._states[request_id] = state
        return state

    def get_state(self, request_id: str) -> HybridQCCVLLMState:
        """Return a typed request state for adapter loading/auditing."""

        state = self._states.get(request_id)
        if state is None:
            raise KeyError(f"unknown request: {request_id}")
        if not isinstance(state, HybridQCCVLLMState):
            raise TypeError("request is not backed by HybridQCCVLLMState")
        return state

    def load_archive_state_dict(
        self,
        request_id: str,
        state_dict: dict[str, Tensor],
        *,
        strict: bool = True,
    ) -> None:
        """Load calibrated archive parameters into one request state.

        Mutable serving tensors are reset after loading. A version-specific vLLM layer
        should normally keep one calibrated parameter template and clone only mutable
        state per request; this helper keeps the dependency-free reference auditable.
        """

        state = self.get_state(request_id)
        missing, unexpected = state.archive.load_state_dict(state_dict, strict=strict)
        if strict and (missing or unexpected):  # defensive; strict=True raises first
            raise ValueError(
                f"archive state mismatch: missing={missing}, unexpected={unexpected}"
            )
        device = state.archive.codes.device
        dtype = state._key_ring.dtype if state._key_ring is not None else torch.float32
        state.reset(1, device=device, dtype=dtype)


__all__ = ["HybridQCCVLLMBackend", "HybridQCCVLLMState"]
