"""Stock vLLM v1 attention backend for QCC packed state.

This first backend is a correctness reference. It uses the current vLLM v1 custom
backend and CircularBufferSpec contracts, so the scheduler owns one physical state page
per request/layer. Python request loops and scalar synchronizations are intentionally
visible; production latency claims require replacing them with the Triton fast path.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CircularBufferSpec,
    KVCacheLayout,
)

from .stock_adapter import load_archive_parameters, load_checkpoint_state_dict
from .stock_runtime import PackedHybridReferenceState
from .vllm_stock import (
    QCCPackedStateLayout,
    STOCK_ADAPTER_ENV,
    stock_config_from_env,
    typed_segment_view,
)


def _tensor_parallel_rank(size: int) -> int:
    """Return this worker's TP rank without assuming one vLLM env spelling."""

    if size == 1:
        return 0
    try:
        from vllm.distributed import parallel_state

        getter = getattr(parallel_state, "get_tensor_model_parallel_rank", None)
        if callable(getter):
            rank = int(getter())
            if 0 <= rank < size:
                return rank
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
        pass
    for name in ("QCC_STOCK_VLLM_TP_RANK", "VLLM_TP_RANK"):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            rank = int(value)
        except ValueError as exc:
            raise RuntimeError(f"{name} must be an integer") from exc
        if not 0 <= rank < size:
            raise RuntimeError(f"{name}={rank} is outside tensor-parallel size {size}")
        return rank
    raise RuntimeError(
        "cannot determine tensor-parallel rank for QCC stock vLLM; "
        "vLLM parallel state or QCC_STOCK_VLLM_TP_RANK is required"
    )


@dataclass
class QCCV1AttentionMetadata(AttentionMetadata):
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    seq_lens: torch.Tensor
    token_to_req: torch.Tensor
    logical_positions: torch.Tensor
    num_actual_tokens: int
    max_query_len: int


class QCCV1MetadataBuilder(AttentionMetadataBuilder[QCCV1AttentionMetadata]):
    """Build request ownership and logical positions for one-page QCC state."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    reorder_batch_threshold = None

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if not isinstance(kv_cache_spec, CircularBufferSpec):
            raise TypeError("QCC stock backend requires CircularBufferSpec")
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_to_req_buffer = torch.empty(max_tokens, dtype=torch.int32, device=device)
        self.logical_positions_buffer = torch.empty(max_tokens, dtype=torch.int64, device=device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> QCCV1AttentionMetadata:
        del common_prefix_len, fast_build
        num_tokens = common_attn_metadata.num_actual_tokens
        token_to_req = common_attn_metadata.token_to_req_indices(self.token_to_req_buffer)
        requests = token_to_req[:num_tokens].long()
        query_start_loc = common_attn_metadata.query_start_loc
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        rows = torch.arange(num_tokens, device=query_start_loc.device, dtype=torch.int64)
        starts = query_start_loc.index_select(0, requests).long()
        within_query = rows - starts
        bases = (
            common_attn_metadata.seq_lens.index_select(0, requests).long()
            - query_lens.index_select(0, requests).long()
        )
        logical = self.logical_positions_buffer[:num_tokens]
        logical.copy_(bases + within_query)
        return QCCV1AttentionMetadata(
            block_table=common_attn_metadata.block_table_tensor,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
            seq_lens=common_attn_metadata.seq_lens,
            token_to_req=token_to_req[:num_tokens],
            logical_positions=logical,
            num_actual_tokens=num_tokens,
            max_query_len=common_attn_metadata.max_query_len,
        )


class QCCV1AttentionImpl(AttentionImpl[QCCV1AttentionMetadata]):
    """Reference implementation backed by one packed state page per request."""

    # A page is owned by one local scheduler rank. Cross-rank context parallelism
    # would require an explicit state exchange for the recurrent archive; rejecting
    # it keeps the one-page ownership contract exact until that collective exists.
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
            raise NotImplementedError("QCC stock backend currently supports decoder attention only")
        if alibi_slopes is not None or sliding_window is not None or logits_soft_cap not in (None, 0, 0.0):
            raise NotImplementedError("QCC stock backend does not yet support ALiBi/SWA/logit soft-cap")
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("QCC stock backend does not support KV-sharing layers")
        config = stock_config_from_env()
        if config.num_heads != num_heads or config.head_dim != head_size:
            raise ValueError(
                "QCC stock config does not match vLLM attention geometry: "
                f"config=({config.num_heads},{config.head_dim}) runtime=({num_heads},{head_size})"
            )
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_heads % num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        configured_kv_heads = config.num_kv_heads
        if configured_kv_heads is not None and configured_kv_heads != num_kv_heads:
            raise ValueError(
                "QCC stock config does not match vLLM KV-head geometry: "
                f"config={configured_kv_heads}, runtime={num_kv_heads}"
            )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.kv_cache_dtype = kv_cache_dtype
        self.config = config
        # Let the runtime take the fused CUDA local/archive paths when Triton is
        # available.  The CPU/Python implementation remains the correctness
        # fallback, but silently forcing it for stock vLLM would turn the custom
        # backend into an interpreter benchmark rather than a serving backend.
        self.runtime = PackedHybridReferenceState(config, use_triton=True)
        self._loaded_layer_name: str | None = None

    def _ensure_layer_weights(self, layer_name: str) -> None:
        if self._loaded_layer_name == layer_name:
            return
        if self._loaded_layer_name is not None and self._loaded_layer_name != layer_name:
            raise RuntimeError("one QCC AttentionImpl instance was reused across different layer names")
        adapter_path = os.environ.get(STOCK_ADAPTER_ENV)
        if not adapter_path:
            raise RuntimeError(f"{STOCK_ADAPTER_ENV} is required for QCC stock vLLM")
        state = load_checkpoint_state_dict(adapter_path)
        rank = _tensor_parallel_rank(self.config.tensor_parallel_size)
        load_archive_parameters(
            self.runtime.archive,
            state,
            layer_name,
            tensor_parallel_rank=rank,
            tensor_parallel_size=self.config.tensor_parallel_size,
        )
        self._loaded_layer_name = layer_name

    def _request_page(
        self,
        kv_cache: torch.Tensor,
        metadata: QCCV1AttentionMetadata,
        request_index: int,
    ) -> torch.Tensor:
        if metadata.block_table.ndim != 2 or metadata.block_table.shape[1] < 1:
            raise RuntimeError("QCC CircularBufferSpec requires one block-table column")
        physical = int(metadata.block_table[request_index, 0].item())
        if physical < 0 or physical >= kv_cache.shape[0]:
            raise RuntimeError(f"invalid QCC physical state block: {physical}")
        page = kv_cache[physical]
        if not page.is_contiguous():
            raise RuntimeError("QCC packed state page must be contiguous under selected layout")
        return page

    def _forward_decode_batch(
        self,
        kv_cache: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: QCCV1AttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Run one-token scheduler rows through the batched packed-page path.

        vLLM's decode scheduler already lays out one token per active request.
        Gathering those pages once lets the runtime execute local attention and
        the archive over a request batch.  The page remains the ownership unit;
        the gathered tensor is written back before returning to the scheduler.
        """

        num_requests = int(metadata.query_start_loc_cpu.shape[0] - 1)
        starts_cpu = metadata.query_start_loc_cpu[:-1].to(dtype=torch.long)
        token_indices = starts_cpu.to(device=query.device)
        physical = metadata.block_table[:num_requests, 0].to(
            device=kv_cache.device, dtype=torch.long
        )
        # CircularBufferSpec guarantees valid, request-private pages.  Keep
        # those contract checks out of the decode hot path: bounds/uniqueness
        # queries force a device synchronization on every scheduler batch and
        # erase the benefit of the batched state update.
        gathered = kv_cache.index_select(0, physical).contiguous()
        pages = gathered.view(num_requests, -1)
        q = query.index_select(0, token_indices)
        k = key.index_select(0, token_indices)
        v = value.index_select(0, token_indices)
        logical = metadata.logical_positions.index_select(0, token_indices)
        result = self.runtime.forward_decode_batch(pages, q, k, v, logical)
        kv_cache.index_copy_(0, physical, pages.view_as(gathered))
        if output.ndim == 3:
            output.index_copy_(0, token_indices, result)
        elif output.ndim == 2:
            output.index_copy_(0, token_indices, result.reshape(num_requests, -1))
        else:
            raise ValueError("unexpected vLLM attention output rank")
        return output

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: QCCV1AttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QCC stock reference does not support fused output quantization")
        if attn_metadata is None:
            return output.zero_()
        layer_name = getattr(layer, "layer_name", None)
        if not isinstance(layer_name, str) or not layer_name:
            raise RuntimeError("vLLM Attention layer must expose layer_name for QCC adapter mapping")
        self._ensure_layer_weights(layer_name)
        num_tokens = attn_metadata.num_actual_tokens
        query = query[:num_tokens]
        key = key[:num_tokens]
        value = value[:num_tokens]

        num_requests = attn_metadata.query_start_loc_cpu.shape[0] - 1
        query_start = attn_metadata.query_start_loc_cpu[:-1]
        query_end = attn_metadata.query_start_loc_cpu[1:]
        if num_requests >= 1 and bool(torch.all((query_end - query_start) == 1).item()):
            return self._forward_decode_batch(
                kv_cache,
                query,
                key,
                value,
                attn_metadata,
                output,
            )
        for request_index in range(num_requests):
            start = int(attn_metadata.query_start_loc_cpu[request_index].item())
            end = int(attn_metadata.query_start_loc_cpu[request_index + 1].item())
            if end <= start:
                continue
            page = self._request_page(kv_cache, attn_metadata, request_index)
            logical_start = int(attn_metadata.logical_positions[start].item())
            counters = typed_segment_view(page, self.runtime.layout, "counters")
            seen = int(counters[2].item())
            if logical_start == 0:
                self.runtime.reset_page(page)
                seen = 0
            if seen != logical_start:
                raise RuntimeError(
                    "QCC packed-state discontinuity: scheduler must preserve the request page "
                    "or recompute the request from logical position 0"
                )
            q = query[start:end].transpose(0, 1).unsqueeze(0)
            k = key[start:end].transpose(0, 1).unsqueeze(0)
            v = value[start:end].transpose(0, 1).unsqueeze(0)
            result = self.runtime.forward(page, q, k, v)
            thd = result.squeeze(0).transpose(0, 1)
            if output.ndim == 3:
                output[start:end].copy_(thd)
            elif output.ndim == 2:
                output[start:end].copy_(thd.reshape(end - start, -1))
            else:
                raise ValueError("unexpected vLLM attention output rank")
        return output


class QCCV1AttentionBackend(AttentionBackend):
    """Out-of-tree stock-vLLM v1 backend using a one-page CircularBufferSpec."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "float16", "bfloat16"]
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_name() -> str:
        return "QCC_V1_PACKED"

    @staticmethod
    def get_impl_cls() -> type[QCCV1AttentionImpl]:
        return QCCV1AttentionImpl

    @staticmethod
    def get_builder_cls() -> type[QCCV1MetadataBuilder]:
        return QCCV1MetadataBuilder

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.BLNHC, KVCacheLayout.BLHNC)

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        config = stock_config_from_env()
        if spec.head_size != config.head_dim:
            raise ValueError(
                f"QCC stock head_dim={config.head_dim} but vLLM layer has {spec.head_size}"
            )
        if spec.head_size_v != spec.head_size:
            raise NotImplementedError("QCC stock backend currently requires equal K/V head dimensions")
        if torch.empty((), dtype=spec.dtype).element_size() != config.local_element_bytes:
            raise ValueError(
                "QCC stock local cache dtype width does not match the configured packed page"
            )
        layout = QCCPackedStateLayout(config)
        return CircularBufferSpec(
            block_size=1,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            head_size_v=spec.head_size_v,
            dtype=spec.dtype,
            kv_quant_mode=spec.kv_quant_mode,
            num_head_slots=1,
            state_content_bytes=layout.total_bytes,
            tokens_per_state=1,
        )


__all__ = [
    "QCCV1AttentionBackend",
    "QCCV1AttentionImpl",
    "QCCV1AttentionMetadata",
    "QCCV1MetadataBuilder",
]
