"""vLLM 0.11+ adapter using the engine's stateful ``MambaSpec`` contract.

Recent vLLM releases removed the experimental ``CircularBufferSpec`` used by
the first QCC adapter.  ``MambaSpec`` is the stable state-cache path: the
scheduler owns one opaque state page per request and passes it to the custom
attention implementation as a list of tensors.  This module keeps the same
packed page and runtime semantics while avoiding any application-model edits.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import ClassVar

import torch

try:  # vLLM 0.28+ keeps the ABI under vllm.v1.
    from vllm.v1.attention.backend import (
        AttentionBackend,
        AttentionImpl,
        AttentionType,
    )
except ImportError:  # vLLM 0.11-0.27 used the legacy abstract module.
    from vllm.attention.backends.abstract import (
        AttentionBackend,
        AttentionImpl,
        AttentionType,
    )
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import MambaSpec

from .stock_adapter import load_archive_parameters, load_checkpoint_state_dict
from .stock_runtime import PackedHybridReferenceState
from .vllm_stock import STOCK_ADAPTER_ENV, _tensor_parallel_rank, stock_config_from_env


@dataclass
class QCCModernAttentionMetadata:
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    logical_positions: torch.Tensor
    state_indices: torch.Tensor
    num_actual_tokens: int
    max_query_len: int


class QCCModernMetadataBuilder(
    AttentionMetadataBuilder[QCCModernAttentionMetadata]
):
    """Build logical positions for one opaque state page per request."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    reorder_batch_threshold = None

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if not isinstance(kv_cache_spec, MambaSpec):
            raise TypeError("QCC modern backend requires MambaSpec")

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> QCCModernAttentionMetadata:
        del common_prefix_len, fast_build
        metadata = common_attn_metadata
        starts = metadata.query_start_loc
        starts_cpu = metadata.query_start_loc_cpu
        num_tokens = int(metadata.num_actual_tokens)
        query_lens = starts[1:] - starts[:-1]
        request_ids = torch.repeat_interleave(
            torch.arange(metadata.num_reqs, device=starts.device), query_lens
        )
        rows = torch.arange(num_tokens, device=starts.device, dtype=torch.long)
        request_starts = starts.index_select(0, request_ids).to(torch.long)
        within = rows - request_starts
        bases = metadata.seq_lens.index_select(0, request_ids).to(torch.long)
        bases = bases - query_lens.index_select(0, request_ids).to(torch.long)
        logical_positions = bases + within
        state_indices = metadata.block_table_tensor[: metadata.num_reqs, 0]
        return QCCModernAttentionMetadata(
            query_start_loc=starts,
            query_start_loc_cpu=starts_cpu,
            logical_positions=logical_positions,
            state_indices=state_indices,
            num_actual_tokens=num_tokens,
            max_query_len=int(metadata.max_query_len),
        )


class QCCModernAttentionImpl(AttentionImpl[QCCModernAttentionMetadata]):
    """Packed QCC state implementation for current upstream vLLM."""

    supports_dcp = False
    supports_pcp = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("QCC backend supports decoder attention only")
        if alibi_slopes is not None or sliding_window is not None:
            raise NotImplementedError("QCC backend does not support ALiBi/sliding-window layers")
        if logits_soft_cap not in (None, 0, 0.0) or kv_sharing_target_layer_name is not None:
            raise NotImplementedError("unsupported vLLM attention feature for QCC backend")
        config = stock_config_from_env()
        if (config.num_heads, config.head_dim) != (num_heads, head_size):
            raise ValueError("QCC stock config does not match vLLM attention geometry")
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.kv_cache_dtype = kv_cache_dtype
        self.config = config
        self.runtime = PackedHybridReferenceState(config, use_triton=True)
        self._loaded_layer_name: str | None = None

    def _ensure_layer_weights(self, layer_name: str) -> None:
        if self._loaded_layer_name == layer_name:
            return
        if self._loaded_layer_name is not None:
            raise RuntimeError("one QCC attention instance was reused across layers")
        adapter_path = os.environ.get(STOCK_ADAPTER_ENV)
        if not adapter_path:
            raise RuntimeError(f"{STOCK_ADAPTER_ENV} is required for QCC vLLM")
        state = load_checkpoint_state_dict(adapter_path)
        load_archive_parameters(
            self.runtime.archive,
            state,
            layer_name,
            tensor_parallel_rank=_tensor_parallel_rank(self.config.tensor_parallel_size),
            tensor_parallel_size=self.config.tensor_parallel_size,
        )
        self._loaded_layer_name = layer_name

    @staticmethod
    def _pages(kv_cache: object) -> torch.Tensor:
        if isinstance(kv_cache, (tuple, list)):
            if len(kv_cache) != 1 or not isinstance(kv_cache[0], torch.Tensor):
                raise ValueError("QCC MambaSpec cache must contain one state tensor")
            pages = kv_cache[0]
        elif isinstance(kv_cache, torch.Tensor):
            pages = kv_cache
        else:
            raise ValueError("QCC vLLM cache must be a tensor or one-tensor list")
        if pages.ndim != 2 or not pages.is_contiguous():
            raise ValueError("QCC state pages must be contiguous [blocks, bytes]")
        return pages

    def _decode_batch(
        self,
        pages: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: QCCModernAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        num_requests = int(metadata.query_start_loc_cpu.numel() - 1)
        token_indices = metadata.query_start_loc_cpu[:-1].to(query.device, dtype=torch.long)
        physical = metadata.state_indices[:num_requests].to(pages.device, dtype=torch.long)
        gathered = pages.index_select(0, physical).contiguous()
        logical = metadata.logical_positions.index_select(0, token_indices)
        result = self.runtime.forward_decode_batch(
            gathered.view(num_requests, -1),
            query.index_select(0, token_indices),
            key.index_select(0, token_indices),
            value.index_select(0, token_indices),
            logical,
        )
        pages.index_copy_(0, physical, gathered)
        output.index_copy_(0, token_indices, result)
        return output

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: object,
        attn_metadata: QCCModernAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QCC backend does not support fused output quantization")
        if attn_metadata is None:
            return torch.zeros_like(query) if output is None else output.zero_()
        layer_name = getattr(layer, "layer_name", None)
        if not isinstance(layer_name, str) or not layer_name:
            raise RuntimeError("vLLM attention layer must expose layer_name")
        self._ensure_layer_weights(layer_name)
        pages = self._pages(kv_cache)
        num_tokens = attn_metadata.num_actual_tokens
        query = query[:num_tokens]
        key = key[:num_tokens]
        value = value[:num_tokens]
        if output is None:
            output = torch.empty_like(query)
        query_start = attn_metadata.query_start_loc_cpu
        query_end = query_start[1:]
        query_start = query_start[:-1]
        num_requests = int(query_start.numel())
        if num_requests and bool(torch.all((query_end - query_start) == 1).item()):
            return self._decode_batch(pages, query, key, value, attn_metadata, output)
        physical = attn_metadata.state_indices.to(pages.device, dtype=torch.long)
        for request_index in range(num_requests):
            start = int(query_start[request_index].item())
            end = int(query_end[request_index].item())
            if end <= start:
                continue
            page = pages[physical[request_index]]
            logical_start = int(attn_metadata.logical_positions[start].item())
            if logical_start == 0:
                self.runtime.reset_page(page)
            q = query[start:end].transpose(0, 1).unsqueeze(0)
            k = key[start:end].transpose(0, 1).unsqueeze(0)
            v = value[start:end].transpose(0, 1).unsqueeze(0)
            result = self.runtime.forward(page, q, k, v)
            output[start:end].copy_(result.squeeze(0).transpose(0, 1))
        return output


class QCCModernAttentionBackend(AttentionBackend):
    """Registry class used by vLLM 0.11 and newer."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "float16", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[QCCModernAttentionImpl]:
        return QCCModernAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[QCCModernMetadataBuilder]:
        return QCCModernMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del num_kv_heads, head_size, cache_dtype_str
        return (num_blocks, block_size)


__all__ = [
    "QCCModernAttentionBackend",
    "QCCModernAttentionImpl",
    "QCCModernAttentionMetadata",
    "QCCModernMetadataBuilder",
]
