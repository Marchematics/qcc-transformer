"""Reference packed-page runtime shared by the stock-vLLM adapter tests.

This module is intentionally dependency-free with respect to vLLM. It proves that
one scheduler-owned page can carry exactly the mutable state used by the hybrid QCC
attention path. The implementation is correctness-first; production kernels may
replace Python-side bookkeeping as long as these semantics remain identical.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .hybrid_archive import HybridQCCArchive
from .vllm_stock import QCCPackedStateLayout, QCCStockVLLMConfig, typed_segment_view


@dataclass(frozen=True)
class FullKVGeometry:
    context_tokens: int
    num_kv_heads: int
    head_dim: int
    element_bytes: int

    @property
    def bytes(self) -> int:
        if min(self.context_tokens, self.num_kv_heads, self.head_dim, self.element_bytes) <= 0:
            raise ValueError("Full-KV geometry values must be positive")
        return 2 * self.context_tokens * self.num_kv_heads * self.head_dim * self.element_bytes


def packed_ratio_vs_full_kv(
    layout: QCCPackedStateLayout,
    *,
    context_tokens: int,
    num_kv_heads: int,
    full_kv_element_bytes: int | None = None,
) -> float:
    """GQA-aware packed-state / Full-KV byte ratio.

    QCC may keep per-query-head state while a GQA baseline stores K/V only for
    ``num_kv_heads``. Using query heads in the denominator overstates memory savings.
    """
    element_bytes = full_kv_element_bytes or layout.config.local_element_bytes
    full = FullKVGeometry(
        context_tokens=context_tokens,
        num_kv_heads=num_kv_heads,
        head_dim=layout.config.head_dim,
        element_bytes=element_bytes,
    ).bytes
    return layout.mutable_state_bytes() / full


def expand_gqa(x: Tensor, num_query_heads: int) -> Tensor:
    """Expand projected K/V from KV heads to query heads without changing values."""
    if x.ndim != 4:
        raise ValueError("projected K/V must be [batch, heads, tokens, dim]")
    kv_heads = x.shape[1]
    if num_query_heads <= 0 or num_query_heads % kv_heads:
        raise ValueError("query heads must be a positive multiple of KV heads")
    if kv_heads == num_query_heads:
        return x
    return x.repeat_interleave(num_query_heads // kv_heads, dim=1)


class PackedHybridReferenceState:
    """Correctness reference operating directly on one packed scheduler page.

    Learned archive parameters are kept in ``archive``; all request-dependent tensors
    live in ``page``. This object may be shared across requests only sequentially,
    which matches one attention layer's forward loop. The production Triton path must
    remove the small ``.item()`` synchronizations used by this reference.
    """

    def __init__(
        self,
        config: QCCStockVLLMConfig,
        *,
        archive: HybridQCCArchive | None = None,
        use_triton: bool = False,
    ) -> None:
        self.config = config
        self.layout = QCCPackedStateLayout(config)
        if archive is None:
            archive = HybridQCCArchive(
                config.num_heads,
                config.head_dim,
                num_codes=config.num_codes,
                decay_rates=config.decay_rates(),
                window_size=config.window_size,
                use_triton=use_triton,
                exact_num_sets=config.exact_num_sets,
                exact_ways=config.exact_ways,
                exact_probe_sets=config.exact_probe_sets,
                admission_threshold=config.admission_threshold,
                max_inserts_per_chunk=config.max_inserts_per_chunk,
                exact_confidence_threshold=config.exact_confidence_threshold,
                exact_confidence_temperature=config.exact_confidence_temperature,
            )
        if archive.num_heads != config.num_heads or archive.head_dim != config.head_dim:
            raise ValueError("archive geometry does not match stock config")
        if archive.num_codes != config.num_codes or archive.num_scales != config.num_scales:
            raise ValueError("archive code/scale geometry does not match stock config")
        if archive.exact_bank.num_sets != config.exact_num_sets or archive.exact_bank.ways != config.exact_ways:
            raise ValueError("archive exact-tier geometry does not match stock config")
        self.archive = copy.deepcopy(archive)
        self.archive.use_triton = use_triton

    def allocate_page(self, *, dtype: torch.dtype, device: torch.device | str = "cpu") -> Tensor:
        page = torch.empty(
            self.layout.words_for_dtype(dtype),
            dtype=dtype,
            device=device,
        )
        self.reset_page(page)
        return page

    def _local_view(self, page: Tensor, name: str, dtype: torch.dtype) -> Tensor:
        itemsize = torch.empty((), dtype=dtype).element_size()
        if itemsize != self.config.local_element_bytes:
            raise ValueError(
                f"local cache expects {self.config.local_element_bytes}-byte activations, got {dtype}"
            )
        raw_typed = typed_segment_view(page, self.layout, name)
        if raw_typed.element_size() == itemsize and raw_typed.dtype == dtype:
            return raw_typed
        return raw_typed.view(dtype)

    def reset_page(self, page: Tensor) -> None:
        """Erase mutable state before assigning a physical page to a new request."""
        raw = page.view(torch.uint8).reshape(-1)
        if raw.numel() < self.layout.total_bytes:
            raise ValueError("page is smaller than packed layout")
        # Clear the complete allocator word, including dtype-rounding padding.
        # vLLM can recycle that tail as part of the physical page, and leaving
        # it untouched makes two freshly assigned pages observably different.
        raw.zero_()
        typed_segment_view(page, self.layout, "exact_scores").fill_(-torch.inf)
        # counters = ring_start, ring_length, seen_tokens, exact_bank_step
        typed_segment_view(page, self.layout, "counters").zero_()

    @staticmethod
    def _segment_dtype(name: str) -> torch.dtype:
        return {
            "uint8": torch.uint8,
            "uint16": torch.uint16,
            "float32": torch.float32,
            "int64": torch.int64,
        }[name]

    def _batched_segment_view(
        self,
        pages: Tensor,
        name: str,
        *,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Return ``[batch, *segment.shape]`` from contiguous packed pages.

        vLLM presents a circular-buffer allocation as a page per request.  An
        indexed gather of those pages is contiguous but no longer compatible
        with :func:`typed_segment_view`, whose offsets describe one page.  This
        helper keeps the same byte layout while adding the batch dimension, so
        a decode batch can be processed by one archive call and one page write
        back instead of one Python call per request.
        """

        if pages.ndim != 2 or not pages.is_contiguous():
            raise ValueError("batched packed pages must be contiguous [batch, bytes]")
        segment = self.layout.segment(name)
        raw = pages.view(torch.uint8).reshape(pages.shape[0], -1)
        if raw.shape[1] < self.layout.total_bytes:
            raise ValueError("batched packed pages are smaller than the configured layout")
        source_dtype = self._segment_dtype(segment.dtype)
        target_dtype = source_dtype if dtype is None else dtype
        if torch.empty((), dtype=target_dtype).element_size() != torch.empty(
            (), dtype=source_dtype
        ).element_size():
            raise ValueError(f"segment {name} cannot be viewed as {target_dtype}")
        data = raw.narrow(1, segment.offset, segment.size_bytes)
        return data.view(target_dtype).reshape(pages.shape[0], *segment.shape)

    def _batched_local_view(
        self, pages: Tensor, name: str, dtype: torch.dtype
    ) -> Tensor:
        if torch.empty((), dtype=dtype).element_size() != self.config.local_element_bytes:
            raise ValueError(
                f"local cache expects {self.config.local_element_bytes}-byte activations, got {dtype}"
            )
        return self._batched_segment_view(pages, name, dtype=dtype)

    def _reset_batched_pages(self, pages: Tensor, reset_mask: Tensor) -> None:
        """Reset only rows assigned to a new logical request."""

        if reset_mask.ndim != 1 or reset_mask.shape[0] != pages.shape[0]:
            raise ValueError("reset_mask must have one entry per packed page")
        if not bool(reset_mask.any()):
            return
        raw = pages.view(torch.uint8).reshape(pages.shape[0], -1)
        raw[reset_mask].zero_()
        self._batched_segment_view(pages, "exact_scores")[reset_mask].fill_(-torch.inf)
        self._batched_segment_view(pages, "counters")[reset_mask].zero_()

    def _bind_archive_pages(self, pages: Tensor) -> tuple[Tensor, ...]:
        """Bind a gathered page batch to the archive's mutable tensors."""

        numerator = self._batched_segment_view(pages, "recurrent_numerator")
        denominator = self._batched_segment_view(pages, "recurrent_denominator")
        last_step = self._batched_segment_view(pages, "recurrent_last_step")
        exact_keys = self._batched_segment_view(pages, "exact_keys")
        exact_values = self._batched_segment_view(pages, "exact_values")
        exact_scores = self._batched_segment_view(pages, "exact_scores")
        exact_ages = self._batched_segment_view(pages, "exact_ages")
        self.archive._numerator = numerator
        self.archive._denominator = denominator
        self.archive._last_step = last_step
        self.archive._step = 0
        self.archive.exact_bank._keys = exact_keys
        self.archive.exact_bank._values = exact_values
        self.archive.exact_bank._scores = exact_scores
        self.archive.exact_bank._ages = exact_ages
        self.archive.exact_bank._step = 0
        return (
            numerator,
            denominator,
            last_step,
            exact_keys,
            exact_values,
            exact_scores,
            exact_ages,
        )

    def _flush_archive_pages(self, pages: Tensor, bound: tuple[Tensor, ...]) -> None:
        """Copy recurrence tensors that were replaced by a block scan."""

        numerator, denominator, last_step, exact_keys, exact_values, exact_scores, exact_ages = bound
        for current, target in (
            (self.archive._numerator, numerator),
            (self.archive._denominator, denominator),
            (self.archive._last_step, last_step),
            (self.archive.exact_bank._keys, exact_keys),
            (self.archive.exact_bank._values, exact_values),
            (self.archive.exact_bank._scores, exact_scores),
            (self.archive.exact_bank._ages, exact_ages),
        ):
            if current.data_ptr() != target.data_ptr():
                target.copy_(current)
        del pages

    @torch.no_grad()
    def forward_decode_batch(
        self,
        pages: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logical_positions: Tensor,
    ) -> Tensor:
        """Process one token for many scheduler-owned pages in one call.

        ``pages`` is a contiguous ``[batch, bytes]`` gather of the physical
        CircularBufferSpec pages.  Query is ``[batch, query_heads, dim]`` and
        K/V may use fewer grouped-query heads.  The operation keeps all
        request-local counters and state in their corresponding page; only the
        temporary gathered tensor is batched.

        This is the serving fast path for decode.  Prefill continues to use the
        reference block method because requests can have different prompt
        lengths and eviction boundaries.
        """

        if pages.ndim != 2 or not pages.is_contiguous():
            raise ValueError("pages must be contiguous [batch, bytes]")
        if query.ndim != 3 or key.ndim != 3 or value.shape != key.shape:
            raise ValueError("query/key/value must be [batch, heads, dim]")
        batch = pages.shape[0]
        if query.shape[0] != batch or key.shape[0] != batch:
            raise ValueError("page and Q/K/V batch sizes must match")
        if query.shape[1:] != (self.config.num_heads, self.config.head_dim):
            raise ValueError("query shape does not match packed config")
        if logical_positions.ndim != 1 or logical_positions.shape[0] != batch:
            raise ValueError("logical_positions must have one entry per page")
        if logical_positions.device != pages.device:
            logical_positions = logical_positions.to(pages.device)

        counters = self._batched_segment_view(pages, "counters")
        reset_mask = logical_positions == 0
        self._reset_batched_pages(pages, reset_mask)
        counters = self._batched_segment_view(pages, "counters")
        seen = counters[:, 2]
        if bool(torch.any(seen != logical_positions).item()):
            raise RuntimeError(
                "QCC packed-state discontinuity: scheduler must preserve each request page"
            )

        query = query.to(device=pages.device)
        key = key.to(device=pages.device)
        value = value.to(device=pages.device)
        key = expand_gqa(key.unsqueeze(2), self.config.num_heads).squeeze(2)
        value = expand_gqa(value.unsqueeze(2), self.config.num_heads).squeeze(2)
        local_keys = self._batched_local_view(pages, "local_keys", query.dtype)
        local_values = self._batched_local_view(pages, "local_values", query.dtype)
        start = counters[:, 0].to(torch.long)
        length = counters[:, 1].to(torch.long)
        if bool(torch.any(start < 0).item()) or bool(
            torch.any(start >= self.config.window_size).item()
        ):
            raise RuntimeError("corrupt packed local-ring start counters")
        if bool(torch.any(length < 0).item()) or bool(
            torch.any(length > self.config.window_size).item()
        ):
            raise RuntimeError("corrupt packed local-ring length counters")

        ordered_keys: Tensor | None = None
        ordered_values: Tensor | None = None
        local: Tensor | None = None
        if pages.is_cuda and query.is_cuda:
            try:
                from .triton_kernels import (
                    TRITON_AVAILABLE,
                    triton_local_ring_decode_attention,
                )
            except ImportError:  # pragma: no cover - optional CUDA dependency
                TRITON_AVAILABLE = False
            if TRITON_AVAILABLE:
                # The kernel reads the physical ring directly and accepts one
                # length per request, so mixed prefill/decode rows no longer
                # need a full chronological gather just to run local attention.
                local = triton_local_ring_decode_attention(
                    query,
                    local_keys,
                    local_values,
                    key,
                    value,
                    start,
                    length,
                )
        if local is None:
            ring_positions = (
                start[:, None] + torch.arange(self.config.window_size, device=pages.device)
            ) % self.config.window_size
            gather_index = ring_positions[:, None, :, None].expand(
                batch, self.config.num_heads, self.config.window_size, self.config.head_dim
            )
            ordered_keys = local_keys.gather(2, gather_index)
            ordered_values = local_values.gather(2, gather_index)
            local_keys_with_current = torch.cat((ordered_keys, key.unsqueeze(2)), dim=2)
            local_values_with_current = torch.cat((ordered_values, value.unsqueeze(2)), dim=2)
            positions = torch.arange(self.config.window_size, device=pages.device)
            lower = (length + 1 - self.config.window_size).clamp_min(0)
            valid_old = (positions[None, :] >= lower[:, None]) & (
                positions[None, :] < length[:, None]
            )
            valid = torch.cat(
                (valid_old, torch.ones((batch, 1), device=pages.device, dtype=torch.bool)),
                dim=1,
            )
            logits = torch.einsum(
                "bhd,bhkd->bhk", query, local_keys_with_current
            ) / math.sqrt(self.config.head_dim)
            logits = logits.masked_fill(
                ~valid[:, None, :], torch.finfo(logits.dtype).min
            )
            local = torch.einsum(
                "bhk,bhkd->bhd", torch.softmax(logits.float(), dim=-1).to(query.dtype),
                local_values_with_current,
            )

        archive_out = torch.zeros_like(local)
        evict_mask = length >= self.config.window_size
        active_indices = torch.nonzero(evict_mask, as_tuple=False).flatten()
        if active_indices.numel():
            active_pages = pages.index_select(0, active_indices).contiguous()
            if ordered_keys is None or ordered_values is None:
                active_ring_keys = local_keys.index_select(0, active_indices)
                active_ring_values = local_values.index_select(0, active_indices)
                active_start = start.index_select(0, active_indices)
                active_index = active_start[:, None, None, None].expand(
                    active_indices.numel(), self.config.num_heads, 1, self.config.head_dim
                )
                active_key = active_ring_keys.gather(2, active_index).squeeze(2)
                active_value = active_ring_values.gather(2, active_index).squeeze(2)
            else:
                active_key = ordered_keys.index_select(0, active_indices)[:, :, 0]
                active_value = ordered_values.index_select(0, active_indices)[:, :, 0]
            active_query = query.index_select(0, active_indices).unsqueeze(2)
            active_key_block = active_key.unsqueeze(2)
            active_value_block = active_value.unsqueeze(2)
            self.archive.to(device=pages.device)
            admission_score = self.archive.admission(active_key, active_value)
            admitted = (admission_score >= self.archive.admission_threshold).any(dim=1)
            bound = self._bind_archive_pages(active_pages)
            active_counters = self._batched_segment_view(active_pages, "counters")
            old_steps = active_counters[:, 3].clone()
            # Exact-bank ages are absolute in each page, while one batched
            # bank instance uses a shared local step. Rebase ages to zero,
            # process one event, then restore each request's own epoch. The
            # restore offset is the old epoch: existing entries keep their
            # insertion step, while newly written entries already carry step 1.
            self.archive.exact_bank._ages.sub_(old_steps[:, None, None, None])
            active_archive = self.archive.update_read_chunk(
                active_key_block,
                active_value_block,
                active_query,
                admission_score=admission_score.unsqueeze(2),
            ).squeeze(2)
            self.archive.exact_bank._ages.add_(old_steps[:, None, None, None])
            active_counters[:, 3] = old_steps + admitted.to(old_steps.dtype)
            self._flush_archive_pages(active_pages, bound)
            archive_out.index_copy_(0, active_indices, active_archive)
            pages.index_copy_(0, active_indices, active_pages)

        active = seen >= self.config.window_size
        output = torch.where(
            active[:, None, None],
            (1.0 - self.config.archive_mix) * local
            + self.config.archive_mix * archive_out,
            local,
        )

        write_index = torch.where(
            length < self.config.window_size,
            (start + length) % self.config.window_size,
            start,
        )
        batch_index = torch.arange(batch, device=pages.device)[:, None]
        head_index = torch.arange(self.config.num_heads, device=pages.device)[None, :]
        write_index = write_index[:, None].expand(batch, self.config.num_heads)
        local_keys[batch_index, head_index, write_index] = key
        local_values[batch_index, head_index, write_index] = value
        counters[:, 0] = torch.where(
            length < self.config.window_size,
            start,
            (start + 1) % self.config.window_size,
        )
        counters[:, 1] = torch.minimum(
            length + 1,
            torch.full_like(length, self.config.window_size),
        )
        counters[:, 2] = seen + 1
        return output

    def _ordered_ring(self, page: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor, int, int]:
        counters = typed_segment_view(page, self.layout, "counters")
        start = int(counters[0].item())
        length = int(counters[1].item())
        if not (0 <= start < self.config.window_size and 0 <= length <= self.config.window_size):
            raise RuntimeError("corrupt packed local-ring counters")
        keys = self._local_view(page, "local_keys", dtype)
        values = self._local_view(page, "local_values", dtype)
        if length == 0:
            return keys[:, :0].unsqueeze(0), values[:, :0].unsqueeze(0), start, length
        end = start + length
        if end <= self.config.window_size:
            return (
                keys[:, start:end].unsqueeze(0),
                values[:, start:end].unsqueeze(0),
                start,
                length,
            )
        ordered_k = torch.cat((keys[:, start:], keys[:, : end % self.config.window_size]), dim=1)
        ordered_v = torch.cat((values[:, start:], values[:, : end % self.config.window_size]), dim=1)
        return ordered_k.unsqueeze(0), ordered_v.unsqueeze(0), start, length

    @staticmethod
    def _local_attention(query: Tensor, old_k: Tensor, old_v: Tensor, key: Tensor, value: Tensor, window: int) -> Tensor:
        outputs: list[Tensor] = []
        scale = 1.0 / math.sqrt(query.shape[-1])
        for index in range(query.shape[2]):
            hist_k = torch.cat((old_k, key[:, :, : index + 1]), dim=2)
            hist_v = torch.cat((old_v, value[:, :, : index + 1]), dim=2)
            hist_k = hist_k[:, :, -window:]
            hist_v = hist_v[:, :, -window:]
            logits = torch.einsum("bhd,bhkd->bhk", query[:, :, index], hist_k) * scale
            weights = torch.softmax(logits.float(), dim=-1).to(hist_v.dtype)
            outputs.append(torch.einsum("bhk,bhkd->bhd", weights, hist_v))
        return torch.stack(outputs, dim=2)

    def _bind_archive_page(self, page: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        numerator = typed_segment_view(page, self.layout, "recurrent_numerator").unsqueeze(0)
        denominator = typed_segment_view(page, self.layout, "recurrent_denominator").unsqueeze(0)
        last_step = typed_segment_view(page, self.layout, "recurrent_last_step").unsqueeze(0)
        exact_keys = typed_segment_view(page, self.layout, "exact_keys").unsqueeze(0)
        exact_values = typed_segment_view(page, self.layout, "exact_values").unsqueeze(0)
        exact_scores = typed_segment_view(page, self.layout, "exact_scores").unsqueeze(0)
        exact_ages = typed_segment_view(page, self.layout, "exact_ages").unsqueeze(0)
        self.archive._numerator = numerator
        self.archive._denominator = denominator
        self.archive._last_step = last_step
        self.archive._step = 0  # dense recurrence does not use lazy absolute steps
        self.archive.exact_bank._keys = exact_keys
        self.archive.exact_bank._values = exact_values
        self.archive.exact_bank._scores = exact_scores
        self.archive.exact_bank._ages = exact_ages
        counters = typed_segment_view(page, self.layout, "counters")
        self.archive.exact_bank._step = int(counters[3].item())
        return numerator, denominator, last_step, exact_keys, exact_values, exact_scores, exact_ages

    def _flush_archive_page(self, page: Tensor, bound: tuple[Tensor, ...]) -> None:
        numerator, denominator, last_step, exact_keys, exact_values, exact_scores, exact_ages = bound
        for current, target in (
            (self.archive._numerator, numerator),
            (self.archive._denominator, denominator),
            (self.archive._last_step, last_step),
            (self.archive.exact_bank._keys, exact_keys),
            (self.archive.exact_bank._values, exact_values),
            (self.archive.exact_bank._scores, exact_scores),
            (self.archive.exact_bank._ages, exact_ages),
        ):
            if current.data_ptr() != target.data_ptr():
                target.copy_(current)
        typed_segment_view(page, self.layout, "counters")[3] = int(self.archive.exact_bank._step)

    def _update_ring(self, page: Tensor, key: Tensor, value: Tensor, old_start: int, old_length: int) -> None:
        keys = self._local_view(page, "local_keys", key.dtype)
        values = self._local_view(page, "local_values", value.dtype)
        length = key.shape[2]
        total = old_length + length
        evicted = max(0, total - self.config.window_size)
        new_start = (old_start + evicted) % self.config.window_size
        keep = min(self.config.window_size, total)

        # Reconstruct the old ordered ring before mutating it.
        old_k, old_v, _, _ = self._ordered_ring(page, key.dtype)
        combined_k = torch.cat((old_k, key), dim=2)[:, :, -keep:]
        combined_v = torch.cat((old_v, value), dim=2)[:, :, -keep:]
        first = min(keep, self.config.window_size - new_start)
        keys[:, new_start : new_start + first].copy_(combined_k[0, :, :first])
        values[:, new_start : new_start + first].copy_(combined_v[0, :, :first])
        if first < keep:
            rest = keep - first
            keys[:, :rest].copy_(combined_k[0, :, first:])
            values[:, :rest].copy_(combined_v[0, :, first:])
        counters = typed_segment_view(page, self.layout, "counters")
        counters[0] = new_start
        counters[1] = keep
        counters[2] += length

    @torch.no_grad()
    def forward(self, page: Tensor, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Consume one logical request block and return ``[1, Hq, T, D]`` output."""
        if query.ndim != 4 or query.shape[0] != 1:
            raise ValueError("query must be [1, query_heads, tokens, dim]")
        if key.shape != value.shape or key.ndim != 4 or key.shape[0] != 1:
            raise ValueError("key/value must be [1, kv_heads, tokens, dim]")
        if query.shape[1] != self.config.num_heads or query.shape[-1] != self.config.head_dim:
            raise ValueError("query shape does not match packed config")
        if key.shape[2:] != query.shape[2:]:
            raise ValueError("K/V token and head dimensions must match query")
        key = expand_gqa(key, self.config.num_heads)
        value = expand_gqa(value, self.config.num_heads)
        old_k, old_v, old_start, old_length = self._ordered_ring(page, query.dtype)
        key = key.to(query.dtype)
        value = value.to(query.dtype)
        local = None
        if self.archive.use_triton and query.is_cuda:
            try:
                from .triton_kernels import TRITON_AVAILABLE, triton_local_chunk_attention
            except ImportError:  # pragma: no cover - optional CUDA dependency
                TRITON_AVAILABLE = False
            if TRITON_AVAILABLE:
                local = triton_local_chunk_attention(
                    query, torch.cat((old_k, key), dim=2), torch.cat((old_v, value), dim=2),
                    old_length=old_length, window_size=self.config.window_size,
                )
        if local is None:
            local = self._local_attention(
                query, old_k, old_v, key, value, self.config.window_size
            )

        archive_out = torch.zeros_like(local)
        event_start = max(0, self.config.window_size - old_length)
        event_count = query.shape[2] - event_start
        if event_count > 0:
            combined_k = torch.cat((old_k, key), dim=2)
            combined_v = torch.cat((old_v, value), dim=2)
            bound = self._bind_archive_page(page)
            self.archive.to(device=query.device)
            self.archive.update_read_chunk(
                combined_k[:, :, :event_count],
                combined_v[:, :, :event_count],
                query[:, :, event_start:],
                output=archive_out[:, :, event_start:],
            )
            self._flush_archive_page(page, bound)

        seen = int(typed_segment_view(page, self.layout, "counters")[2].item())
        active = (seen + torch.arange(query.shape[2], device=query.device) >= self.config.window_size).view(1, 1, -1, 1)
        result = torch.where(
            active,
            (1.0 - self.config.archive_mix) * local + self.config.archive_mix * archive_out,
            local,
        )
        self._update_ring(page, key, value, old_start, old_length)
        return result


__all__ = [
    "FullKVGeometry",
    "PackedHybridReferenceState",
    "expand_gqa",
    "packed_ratio_vs_full_kv",
]
