"""Packed-state contract for a stock vLLM v1 QCC backend.

The production target is one scheduler-owned physical cache block per logical request
and attention layer.  This module is deliberately free of a vLLM import so byte
accounting, alignment and raw typed views are testable on CPU and remain stable even
when vLLM's Python ABI moves.

Only *mutable request state* lives in the packed page. Learned codebooks, archive mix
weights and hybrid-admission parameters remain ordinary model/adapter parameters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class PackedStateSegment:
    name: str
    offset: int
    size_bytes: int
    dtype: str
    shape: tuple[int, ...]

    @property
    def end(self) -> int:
        return self.offset + self.size_bytes


@dataclass(frozen=True)
class QCCStockVLLMConfig:
    """Geometry of one QCC attention layer's per-request state."""

    num_heads: int
    head_dim: int
    window_size: int = 128
    num_codes: int = 16
    num_scales: int = 4
    exact_num_sets: int = 32
    exact_ways: int = 4
    local_element_bytes: int = 2
    alignment: int = 16

    def __post_init__(self) -> None:
        positive = (
            self.num_heads,
            self.head_dim,
            self.window_size,
            self.num_codes,
            self.num_scales,
            self.exact_num_sets,
            self.exact_ways,
            self.local_element_bytes,
            self.alignment,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("all QCC stock-vLLM geometry values must be positive")
        if self.local_element_bytes not in (1, 2, 4):
            raise ValueError("local_element_bytes must be 1, 2, or 4")
        if self.alignment & (self.alignment - 1):
            raise ValueError("alignment must be a power of two")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "QCCStockVLLMConfig":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("packed-state config JSON must contain an object")
        return cls(**payload)


class QCCPackedStateLayout:
    """Byte-exact packed layout for one layer/request state page.

    Recurrent statistics and exact landmarks use fp32 accumulation to preserve the
    numerical policy of the existing HF/Triton path.  Logical counters use int64.
    The local exact ring uses the configured cache element width (normally bf16/fp16).
    """

    def __init__(self, config: QCCStockVLLMConfig) -> None:
        self.config = config
        self._segments: list[PackedStateSegment] = []
        offset = 0

        def align(value: int) -> int:
            return (value + config.alignment - 1) & ~(config.alignment - 1)

        def add(name: str, shape: tuple[int, ...], dtype: str, itemsize: int) -> None:
            nonlocal offset
            offset = align(offset)
            count = math.prod(shape)
            size = count * itemsize
            self._segments.append(
                PackedStateSegment(name, offset, size, dtype, shape)
            )
            offset += size

        h, d = config.num_heads, config.head_dim
        w, m, s = config.window_size, config.num_codes, config.num_scales
        sets, ways = config.exact_num_sets, config.exact_ways
        local_dtype = {1: "uint8", 2: "uint16", 4: "float32"}[
            config.local_element_bytes
        ]

        add("local_keys", (h, w, d), local_dtype, config.local_element_bytes)
        add("local_values", (h, w, d), local_dtype, config.local_element_bytes)
        add("recurrent_numerator", (h, m, s, d), "float32", 4)
        add("recurrent_denominator", (h, m, s), "float32", 4)
        add("recurrent_last_step", (h, m, s), "int64", 8)
        add("exact_keys", (h, sets, ways, d), "float32", 4)
        add("exact_values", (h, sets, ways, d), "float32", 4)
        add("exact_scores", (h, sets, ways), "float32", 4)
        add("exact_ages", (h, sets, ways), "int64", 8)
        # Ring/archive scalar counters are part of request state and must not live in
        # a Python-side registry in production.
        add("counters", (4,), "int64", 8)  # ring_start, ring_length, seen, step
        self.total_bytes = align(offset)

    @property
    def segments(self) -> tuple[PackedStateSegment, ...]:
        return tuple(self._segments)

    def segment(self, name: str) -> PackedStateSegment:
        for segment in self._segments:
            if segment.name == name:
                return segment
        raise KeyError(name)

    def mutable_state_bytes(self) -> int:
        return self.total_bytes

    def words_for_dtype(self, dtype: torch.dtype) -> int:
        itemsize = torch.empty((), dtype=dtype).element_size()
        return (self.total_bytes + itemsize - 1) // itemsize

    def page_bytes_for_dtype(self, dtype: torch.dtype) -> int:
        return self.words_for_dtype(dtype) * torch.empty((), dtype=dtype).element_size()

    def compression_ratio_vs_full_kv(
        self,
        context_tokens: int,
        *,
        full_kv_element_bytes: int | None = None,
    ) -> float:
        """Attention-state bytes / matched unquantized Full-KV bytes.

        This is a geometry diagnostic only. The production gate still measures actual
        peak allocated/reserved memory and concurrency under a fixed SLA.
        """

        if context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        element_bytes = full_kv_element_bytes or self.config.local_element_bytes
        full = (
            2
            * context_tokens
            * self.config.num_heads
            * self.config.head_dim
            * element_bytes
        )
        return self.total_bytes / full

    def manifest(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "total_bytes": self.total_bytes,
            "segments": [asdict(segment) for segment in self._segments],
        }


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "uint8": torch.uint8,
        "uint16": torch.uint16,
        "float32": torch.float32,
        "int64": torch.int64,
    }[name]


def typed_segment_view(
    page: Tensor,
    layout: QCCPackedStateLayout,
    name: str,
) -> Tensor:
    """Return a typed mutable view into one raw scheduler-owned page.

    ``page`` may use any vLLM cache dtype. Its underlying bytes are reinterpreted, not
    numerically converted. The caller must supply a single physical page whose byte
    capacity is at least ``layout.total_bytes``.
    """

    if not page.is_contiguous():
        raise ValueError("packed QCC page must be contiguous")
    raw = page.view(torch.uint8).reshape(-1)
    if raw.numel() < layout.total_bytes:
        raise ValueError(
            f"packed page has {raw.numel()} bytes but layout needs {layout.total_bytes}"
        )
    segment = layout.segment(name)
    data = raw.narrow(0, segment.offset, segment.size_bytes)
    dtype = _dtype_from_name(segment.dtype)
    itemsize = torch.empty((), dtype=dtype).element_size()
    if segment.offset % itemsize or segment.size_bytes % itemsize:
        raise ValueError(f"segment {name} is not aligned for {dtype}")
    return data.view(dtype).reshape(segment.shape)


def validate_layout(layout: QCCPackedStateLayout) -> None:
    """Fail closed on overlaps, misalignment, or inconsistent byte accounting."""

    previous_end = 0
    for segment in layout.segments:
        if segment.offset < previous_end:
            raise ValueError(f"packed-state segment overlaps: {segment.name}")
        if segment.offset % layout.config.alignment:
            raise ValueError(f"segment is not base-aligned: {segment.name}")
        expected = math.prod(segment.shape) * torch.empty(
            (), dtype=_dtype_from_name(segment.dtype)
        ).element_size()
        if expected != segment.size_bytes:
            raise ValueError(f"segment byte count mismatch: {segment.name}")
        previous_end = segment.end
    if previous_end > layout.total_bytes:
        raise ValueError("packed-state total size is smaller than final segment")


def validate_stock_vllm_api() -> dict[str, str]:
    """Validate the vLLM v1 ABI features QCC requires, imported lazily."""

    try:
        from vllm.v1.attention.backend import AttentionBackend, AttentionImpl
        from vllm.v1.kv_cache_interface import CircularBufferSpec, AttentionSpec
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "stock vLLM v1 integration requires qcc-transformer[vllm]"
        ) from exc
    required = {
        "AttentionBackend.customize_spec": getattr(AttentionBackend, "customize_spec", None),
        "AttentionImpl.forward": getattr(AttentionImpl, "forward", None),
        "CircularBufferSpec": CircularBufferSpec,
        "AttentionSpec.state_content_bytes": getattr(AttentionSpec, "state_content_bytes", None),
        "AttentionBackendEnum.CUSTOM": getattr(AttentionBackendEnum, "CUSTOM", None),
        "register_backend": register_backend,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError("unsupported vLLM v1 ABI; missing: " + ", ".join(missing))
    return {name: getattr(value, "__name__", type(value).__name__) for name, value in required.items()}


def register_stock_vllm_backend() -> object:
    """Register QCC as vLLM's CUSTOM attention backend."""

    validate_stock_vllm_api()
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "qcc_transformer.vllm_v1_backend.QCCV1AttentionBackend",
    )
    return AttentionBackendEnum.CUSTOM


__all__ = [
    "PackedStateSegment",
    "QCCPackedStateLayout",
    "QCCStockVLLMConfig",
    "register_stock_vllm_backend",
    "typed_segment_view",
    "validate_layout",
    "validate_stock_vllm_api",
]
