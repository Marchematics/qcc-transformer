"""Packed-state contract for a stock vLLM v1 QCC backend.

The production target is one scheduler-owned physical cache block per logical request
and attention layer. This module is deliberately free of a vLLM import so byte
accounting, alignment and raw typed views are testable on CPU and remain stable even
when vLLM's Python ABI moves.

Only *mutable request state* lives in the packed page. Learned codebooks, archive mix
weights and hybrid-admission parameters remain ordinary model/adapter parameters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from typing import Any

import torch
from torch import Tensor

STOCK_CONFIG_ENV = "QCC_STOCK_VLLM_CONFIG"
STOCK_ADAPTER_ENV = "QCC_STOCK_VLLM_ADAPTER"


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
    """Geometry and deployment policy of one QCC attention layer."""

    num_heads: int
    head_dim: int
    window_size: int = 128
    num_codes: int = 16
    num_scales: int = 4
    exact_num_sets: int = 128
    exact_ways: int = 4
    exact_probe_sets: int | None = None
    max_position_embeddings: int = 1_000_000
    local_element_bytes: int = 2
    alignment: int = 16
    archive_mix: float = 0.125
    exact_confidence_threshold: float = 0.60
    exact_confidence_temperature: float = 20.0
    max_inserts_per_chunk: int = 8
    admission_threshold: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            self.num_heads,
            self.head_dim,
            self.window_size,
            self.num_codes,
            self.num_scales,
            self.exact_num_sets,
            self.exact_ways,
            self.max_position_embeddings,
            self.local_element_bytes,
            self.alignment,
            self.exact_confidence_temperature,
            self.max_inserts_per_chunk,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("all positive QCC stock-vLLM values must be > 0")
        if self.max_position_embeddings < self.window_size:
            raise ValueError("max_position_embeddings must cover the local window")
        if self.exact_probe_sets is not None and self.exact_probe_sets <= 0:
            raise ValueError("exact_probe_sets must be positive when provided")
        if self.exact_probe_sets is not None and self.exact_probe_sets > self.exact_num_sets:
            raise ValueError("exact_probe_sets cannot exceed exact_num_sets")
        if self.local_element_bytes not in (1, 2, 4):
            raise ValueError("local_element_bytes must be 1, 2, or 4")
        if self.alignment & (self.alignment - 1):
            raise ValueError("alignment must be a power of two")
        if not 0.0 <= self.archive_mix <= 1.0:
            raise ValueError("archive_mix must lie in [0, 1]")
        if not -1.0 <= self.exact_confidence_threshold <= 1.0:
            raise ValueError("exact_confidence_threshold must lie in [-1, 1]")
        if not math.isfinite(self.admission_threshold):
            raise ValueError("admission_threshold must be finite")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "QCCStockVLLMConfig":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("packed-state config JSON must contain an object")
        return cls(**payload)

    def decay_rates(self) -> tuple[float, ...]:
        """Match the HF QCC logarithmic half-life schedule."""

        horizons = torch.logspace(
            math.log10(max(1.0, float(self.window_size))),
            math.log10(
                max(float(self.window_size), float(self.max_position_embeddings))
            ),
            self.num_scales,
        )
        return tuple(torch.exp(-math.log(2.0) / horizons).tolist())


def stock_config_from_env() -> QCCStockVLLMConfig:
    """Load the exact deployment geometry inherited by vLLM worker processes."""

    text = os.environ.get(STOCK_CONFIG_ENV)
    if not text:
        raise RuntimeError(
            f"{STOCK_CONFIG_ENV} is required for stock vLLM QCC; "
            "use configure_stock_vllm_environment() before engine startup"
        )
    return QCCStockVLLMConfig.from_json(text)


def configure_stock_vllm_environment(
    config: QCCStockVLLMConfig,
    *,
    adapter_path: str,
) -> None:
    """Set worker-inherited configuration without touching application model code."""

    if not adapter_path:
        raise ValueError("adapter_path is required")
    os.environ[STOCK_CONFIG_ENV] = config.to_json()
    os.environ[STOCK_ADAPTER_ENV] = adapter_path


class QCCPackedStateLayout:
    """Byte-exact packed layout for one layer/request state page.

    Recurrent statistics and exact landmarks use fp32 accumulation to preserve the
    numerical policy of the existing HF/Triton path. Logical counters use int64.
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
            size = math.prod(shape) * itemsize
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
        num_kv_heads: int | None = None,
        full_kv_element_bytes: int | None = None,
    ) -> float:
        """Attention-state bytes / matched unquantized Full-KV bytes.

        This is a geometry diagnostic only. The production gate still measures actual
        peak allocated/reserved memory and concurrency under a fixed SLA.
        """

        if context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        kv_heads = self.config.num_heads if num_kv_heads is None else num_kv_heads
        if kv_heads <= 0 or kv_heads > self.config.num_heads:
            raise ValueError("num_kv_heads must lie in [1, num_heads]")
        if self.config.num_heads % kv_heads:
            raise ValueError("num_kv_heads must divide num_heads for GQA")
        element_bytes = full_kv_element_bytes or self.config.local_element_bytes
        full = (
            2
            * context_tokens
            * kv_heads
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
    """Return a typed mutable view into one raw scheduler-owned page."""

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
    """Validate the exact vLLM v1 ABI features QCC requires, imported lazily."""

    try:
        from vllm.v1.attention.backend import (
            AttentionBackend,
            AttentionImpl,
            CommonAttentionMetadata,
        )
        from vllm.v1.kv_cache_interface import CircularBufferSpec, AttentionSpec
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "stock vLLM v1 integration requires qcc-transformer[vllm]"
        ) from exc
    checks = {
        "AttentionBackend.customize_spec": hasattr(AttentionBackend, "customize_spec"),
        "AttentionImpl.forward": hasattr(AttentionImpl, "forward"),
        "CircularBufferSpec.max_num_blocks_per_req": hasattr(
            CircularBufferSpec, "max_num_blocks_per_req"
        ),
        "AttentionSpec.state_content_bytes": "state_content_bytes"
        in getattr(AttentionSpec, "__dataclass_fields__", {}),
        "CommonAttentionMetadata.token_to_req_indices": hasattr(
            CommonAttentionMetadata, "token_to_req_indices"
        ),
        "AttentionBackendEnum.CUSTOM": hasattr(AttentionBackendEnum, "CUSTOM"),
        "register_backend": callable(register_backend),
    }
    missing = [name for name, present in checks.items() if not present]
    if missing:
        raise RuntimeError("unsupported vLLM v1 ABI; missing: " + ", ".join(missing))
    return {name: "ok" for name in checks}


def register_stock_vllm_backend() -> object | None:
    """Register QCC as vLLM's CUSTOM attention backend when configured.

    vLLM discovers ``vllm.general_plugins`` entry points in every process.  A
    no-op without the two QCC environment variables keeps ordinary vLLM users
    unaffected, while a configured worker gets the same registration as the
    programmatic launcher.
    """

    if not os.environ.get(STOCK_CONFIG_ENV) or not os.environ.get(STOCK_ADAPTER_ENV):
        return None

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
    "STOCK_ADAPTER_ENV",
    "STOCK_CONFIG_ENV",
    "configure_stock_vllm_environment",
    "register_stock_vllm_backend",
    "stock_config_from_env",
    "typed_segment_view",
    "validate_layout",
    "validate_stock_vllm_api",
]
