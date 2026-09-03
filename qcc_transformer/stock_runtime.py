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
        raw[: self.layout.total_bytes].zero_()
        typed_segment_view(page, self.layout, "exact_scores").fill_(-torch.inf)
        # counters = ring_start, ring_length, seen_tokens, exact_bank_step
        typed_segment_view(page, self.layout, "counters").zero_()

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
        local = self._local_attention(
            query, old_k, old_v, key.to(query.dtype), value.to(query.dtype), self.config.window_size
        )

        archive_out = torch.zeros_like(local)
        event_start = max(0, self.config.window_size - old_length)
        event_count = query.shape[2] - event_start
        if event_count > 0:
            combined_k = torch.cat((old_k, key.to(query.dtype)), dim=2)
            combined_v = torch.cat((old_v, value.to(query.dtype)), dim=2)
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
        self._update_ring(page, key.to(query.dtype), value.to(query.dtype), old_start, old_length)
        return result


__all__ = [
    "FullKVGeometry",
    "PackedHybridReferenceState",
    "expand_gqa",
    "packed_ratio_vs_full_kv",
]
