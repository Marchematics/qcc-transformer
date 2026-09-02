"""Reference PyTorch implementation of Query-Compiled Cache attention.

This module intentionally favors readable streaming semantics over kernel-level
performance. The archive state is the object to replace with a fused Triton or
CUDA implementation once the approximation has passed quality tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class QCCState:
    """Mutable state returned by :class:`QCCArchive` for inspection/checkpointing."""

    numerator: Tensor
    denominator: Tensor


class SinusoidalPositionEmbedding(nn.Module):
    """Stateless positions that remain valid beyond a learned table limit.

    A learned ``nn.Embedding`` allocates ``max_position_embeddings * d_model``
    parameters even when serving only one token at a time.  This module builds
    the requested positions on demand, so a model configured for million-token
    streams does not reserve a million-row parameter table.  It is deliberately
    kept as a normal module to make the positional policy explicit and easy to
    swap in experiments.
    """

    def __init__(self, d_model: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if max_period <= 1.0:
            raise ValueError("max_period must be greater than one")
        half = (d_model + 1) // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / max(half, 1)
        )
        self.d_model = d_model
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, positions: Tensor) -> Tensor:
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("positions must be an integer tensor")
        angles = positions.to(dtype=torch.float32).unsqueeze(-1) * self.frequencies
        values = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return values[..., : self.d_model]


class QCCArchive(nn.Module):
    """Constant-size, multi-timescale softmax response memory.

    Args:
        num_heads: Number of key/value heads.
        head_dim: Dimension of each key/value head.
        num_codes: Number of learned long-range query prototypes.
        decay_rates: Values in ``(0, 1)``. Each rate provides a different
            exponential time scale.
        window_size: Number of exact recent tokens. A newly evicted token is
            inserted with ``decay_rate ** window_size`` age compensation.

    The archive stores a response for each (head, code, decay rate), rather
    than a key/value pair per historical token. Its memory is independent of
    sequence length.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_codes: int = 16,
        decay_rates: tuple[float, ...] = (0.995, 0.98, 0.94, 0.85),
        window_size: int = 128,
        use_triton: bool = True,
        active_codes: Optional[int] = None,
        lazy_decay: bool = False,
        scan_block_size: int = 256,
        content_threshold: Optional[float] = None,
        persistent_landmark: bool = False,
        prefix_landmark: bool = False,
        prefix_pair_landmark: bool = False,
    ) -> None:
        super().__init__()
        if num_heads <= 0 or head_dim <= 0 or num_codes <= 0:
            raise ValueError("head dimensions and num_codes must be positive")
        rates = torch.tensor(decay_rates, dtype=torch.float32)
        if rates.numel() == 0 or not bool(torch.all((rates > 0) & (rates < 1))):
            raise ValueError("decay_rates must contain values strictly between 0 and 1")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if active_codes is not None and not 0 < active_codes <= num_codes:
            raise ValueError("active_codes must be in [1, num_codes] when provided")
        if lazy_decay and active_codes is None:
            raise ValueError("lazy_decay requires active_codes to bound touched slots")
        if scan_block_size <= 0:
            raise ValueError("scan_block_size must be positive")
        if content_threshold is not None and not math.isfinite(content_threshold):
            raise ValueError("content_threshold must be finite when provided")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_codes = num_codes
        self.num_scales = int(rates.numel())
        self.window_size = window_size
        self.use_triton = use_triton
        # Inference may route to a small top-k subset while retaining an
        # overcomplete codebook for representational capacity. ``None`` keeps
        # the dense reference path.
        self.active_codes = active_codes
        self.lazy_decay = lazy_decay
        self.scan_block_size = scan_block_size
        # Optional hard event gate.  Scores below the threshold do not enter
        # the archive, which prevents long runs of uninformative filler from
        # diluting a sparse retrieval signal.  ``None`` preserves the original
        # dense recurrence exactly.
        self.content_threshold = content_threshold
        # Optional max-retained landmark slots.  Unlike the exponentially
        # decayed response state, these slots keep the highest-salience value
        # seen for each code indefinitely.  The mechanism is disabled by
        # default so existing kernels/checkpoints retain their exact state.
        self.persistent_landmark = persistent_landmark
        self.prefix_landmark = prefix_landmark
        self.prefix_pair_landmark = prefix_pair_landmark
        if prefix_landmark and not persistent_landmark:
            raise ValueError("prefix_landmark requires persistent_landmark")
        if prefix_pair_landmark and not prefix_landmark:
            raise ValueError("prefix_pair_landmark requires prefix_landmark")
        if persistent_landmark:
            self.landmark_mix_logits = nn.Parameter(torch.zeros(num_heads))
        self.register_buffer("decay_rates", rates, persistent=True)
        self.codes = nn.Parameter(torch.randn(num_heads, num_codes, head_dim) / math.sqrt(head_dim))
        self.mix_logits = nn.Parameter(torch.zeros(num_heads, num_codes, self.num_scales))
        self.reset_state(batch_size=1, device=rates.device, dtype=torch.float32)

    def reset_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Reset archive state before a new independent stream."""

        device = device or self.codes.device
        # Accumulation in fp32 avoids long-stream drift when activations are fp16/bf16.
        state_dtype = dtype if dtype in (torch.float32, torch.float64) else torch.float32
        self._numerator = torch.zeros(
            batch_size,
            self.num_heads,
            self.num_codes,
            self.num_scales,
            self.head_dim,
            device=device,
            dtype=state_dtype,
        )
        self._denominator = torch.zeros(
            batch_size,
            self.num_heads,
            self.num_codes,
            self.num_scales,
            device=device,
            dtype=state_dtype,
        )
        self._last_step = torch.zeros(
            batch_size,
            self.num_heads,
            self.num_codes,
            self.num_scales,
            device=device,
            dtype=torch.long,
        )
        self._step = 0
        if self.persistent_landmark:
            self._landmark_count = 0
            self._prefix_pending_slot = -1
            self._landmark_score = torch.full(
                (batch_size, self.num_heads, self.num_codes),
                -torch.inf,
                device=device,
                dtype=state_dtype,
            )
            self._landmark_value = torch.zeros(
                batch_size,
                self.num_heads,
                self.num_codes,
                self.head_dim,
                device=device,
                dtype=state_dtype,
            )
            self._landmark_key = torch.zeros(
                batch_size,
                self.num_heads,
                self.num_codes,
                self.head_dim,
                device=device,
                dtype=state_dtype,
            )

    def _landmark_scores(self, key: Tensor) -> Tensor:
        """Return per-code salience scores for landmark updates."""

        codes = self.codes.to(device=key.device, dtype=self._numerator.dtype)
        return torch.einsum(
            "bhd,hmd->bhm", key.to(self._numerator.dtype), codes
        ) / math.sqrt(self.head_dim)

    def _update_landmark(self, key: Tensor, value: Tensor) -> None:
        """Retain the highest-scoring value for every code."""

        if not self.persistent_landmark:
            return
        if self.prefix_landmark:
            if self._landmark_count >= self.num_codes:
                # Prefix mode is intentionally immutable after the fixed
                # prefix has been captured; later filler must never replace
                # those exact-key anchors.
                return
            slot = self._landmark_count
            if self.prefix_pair_landmark and self._prefix_pending_slot >= 0:
                # Bind the previous retained key to the current token's value.
                # This is useful for marker/value streams where the query
                # matches the marker key but the answer is its successor.
                self._landmark_value[:, :, self._prefix_pending_slot] = value.to(
                    self._landmark_value.dtype
                )
            score = self._landmark_scores(key)
            self._landmark_key[:, :, slot] = key.to(self._landmark_key.dtype)
            self._landmark_value[:, :, slot] = value.to(self._landmark_value.dtype)
            # Validity is tracked per retained slot; use the strongest code
            # response as its scalar salience while routing itself uses the
            # retained key directly at read time.
            self._landmark_score[:, :, slot] = score.max(dim=-1).values
            self._landmark_count += 1
            if self.prefix_pair_landmark:
                self._prefix_pending_slot = slot
            return
        score = self._landmark_scores(key)
        if self.content_threshold is not None:
            score = torch.where(
                score >= self.content_threshold,
                score,
                torch.full_like(score, -torch.inf),
            )
        better = score > self._landmark_score
        candidate = value.to(self._landmark_value.dtype).unsqueeze(2)
        self._landmark_value = torch.where(
            better.unsqueeze(-1), candidate, self._landmark_value
        )
        self._landmark_key = torch.where(
            better.unsqueeze(-1), key.to(self._landmark_key.dtype).unsqueeze(2), self._landmark_key
        )
        self._landmark_score = torch.where(
            better, score, self._landmark_score
        )

    def _update_landmark_chunk(self, key: Tensor, value: Tensor) -> None:
        """Vectorized landmark update for an evicted token block."""

        if not self.persistent_landmark or key.shape[2] == 0:
            return
        if self.prefix_landmark:
            take = min(key.shape[2], self.num_codes - self._landmark_count)
            for index in range(take):
                self._update_landmark(key[:, :, index], value[:, :, index])
            # Once the prefix is full, intentionally ignore later events.
            return
        self._update_landmark_chunk_max(key, value)

    def _update_landmark_chunk_max(self, key: Tensor, value: Tensor) -> None:
        """Vectorized max-salience update used after prefix slots are filled."""

        if not self.persistent_landmark or key.shape[2] == 0:
            return
        scores = torch.einsum(
            "bhed,hmd->bhem", key.to(self._numerator.dtype),
            self.codes.to(device=key.device, dtype=self._numerator.dtype),
        ) / math.sqrt(self.head_dim)
        if self.content_threshold is not None:
            scores = torch.where(
                scores >= self.content_threshold,
                scores,
                torch.full_like(scores, -torch.inf),
            )
        best_score, best_index = scores.max(dim=2)
        value_index = best_index.unsqueeze(-1).expand(
            -1, -1, -1, self.head_dim
        )
        best_value = value.to(self._landmark_value.dtype).gather(2, value_index)
        best_key = key.to(self._landmark_key.dtype).gather(2, value_index)
        better = best_score > self._landmark_score
        self._landmark_value = torch.where(
            better.unsqueeze(-1), best_value, self._landmark_value
        )
        self._landmark_key = torch.where(
            better.unsqueeze(-1), best_key, self._landmark_key
        )
        self._landmark_score = torch.where(
            better, best_score, self._landmark_score
        )

    def _landmark_read(self, query: Tensor) -> tuple[Tensor, Tensor]:
        """Read persistent landmarks and return output plus validity mask."""

        if not self.persistent_landmark:
            return torch.zeros_like(query), torch.zeros(
                query.shape[:-1], dtype=torch.bool, device=query.device
            )
        if query.ndim == 3:
            routing_logits = torch.einsum(
                "bhd,bhmd->bhm", query.to(self._landmark_key.dtype), self._landmark_key
            ) / math.sqrt(self.head_dim)
            routing_equation = "bhm,bhmd->bhd"
        elif query.ndim == 4:
            routing_logits = torch.einsum(
                "bhed,bhmd->bhem", query.to(self._landmark_key.dtype), self._landmark_key
            ) / math.sqrt(self.head_dim)
            routing_equation = "bhem,bhemd->bhed"
        else:
            raise ValueError("query must have shape [batch, heads, dim] or [batch, heads, time, dim]")
        routing_logits = torch.where(
            torch.isfinite(self._landmark_score).unsqueeze(2)
            if query.ndim == 4
            else torch.isfinite(self._landmark_score),
            routing_logits,
            # Keep the masked softmax finite when a head has not observed a
            # threshold-passing event yet.  The corresponding values are
            # zeroed below, so a uniform probability over invalid slots has
            # no effect on the response.
            torch.full_like(routing_logits, -1.0e9),
        )
        routing = F.softmax(routing_logits, dim=-1).to(self._landmark_value.dtype)
        values = torch.where(
            torch.isfinite(self._landmark_score).unsqueeze(-1),
            self._landmark_value,
            torch.zeros_like(self._landmark_value),
        )
        response = torch.einsum(routing_equation, routing, values.unsqueeze(2) if query.ndim == 4 else values)
        valid = torch.isfinite(self._landmark_score).any(dim=-1)
        if query.ndim == 4:
            valid = valid.unsqueeze(-1).expand(query.shape[:-1])
        return response.to(query.dtype), valid

    def _combine_landmark(self, query: Tensor, response: Tensor) -> Tensor:
        """Blend sticky and exponentially decayed responses when enabled."""

        if not self.persistent_landmark:
            return response
        landmark, valid = self._landmark_read(query)
        # Prefix landmarks are an exact-key fallback, so always trust them in
        # that mode.  The learned blend remains available for max-salience
        # landmarks, where mixing can preserve the exponentially averaged
        # archive's broader context.
        mix = (
            torch.ones_like(self.landmark_mix_logits)
            if self.prefix_landmark
            else torch.sigmoid(self.landmark_mix_logits)
        ).to(response.dtype)
        mix_shape = (1, -1, 1) if response.ndim == 3 else (1, -1, 1, 1)
        mix = mix.view(*mix_shape)
        mixed = (1.0 - mix) * response + mix * landmark
        return torch.where(valid.unsqueeze(-1), mixed, response)

    @property
    def state(self) -> QCCState:
        return QCCState(self._numerator, self._denominator)

    @torch.no_grad()
    def detach_state(self) -> None:
        """Detach streaming state, useful between training chunks."""

        self._numerator = self._numerator.detach()
        self._denominator = self._denominator.detach()

    def update(self, key: Tensor, value: Tensor) -> None:
        """Insert one evicted token per batch/head into the archive.

        ``key`` and ``value`` have shape ``[batch, heads, head_dim]``. Existing
        state is decayed by one step; the inserted token is aged by the exact
        local-window length.
        """

        if key.shape != value.shape or key.ndim != 3:
            raise ValueError("key and value must both have shape [batch, heads, head_dim]")
        bsz, heads, dim = key.shape
        if heads != self.num_heads or dim != self.head_dim:
            raise ValueError("key shape does not match archive configuration")
        if self._numerator.shape[0] != bsz or self._numerator.device != key.device:
            self.reset_state(bsz, device=key.device)

        if self.lazy_decay and not torch.is_grad_enabled():
            self._lazy_update(key, value)
            return

        rates = self.decay_rates.to(device=key.device, dtype=self._numerator.dtype)
        if self.use_triton and not torch.is_grad_enabled() and key.is_cuda:
            from .triton_kernels import TRITON_AVAILABLE, triton_update_archive

            if TRITON_AVAILABLE:
                triton_update_archive(
                    self._numerator,
                    self._denominator,
                    key,
                    value,
                    self.codes,
                    rates,
                    self.window_size,
                    self.content_threshold,
                )
                return

        codes = self.codes.to(device=key.device, dtype=self._numerator.dtype)
        score = torch.einsum("bhd,hmd->bhm", key.to(self._numerator.dtype), codes)
        score = score / math.sqrt(self.head_dim)
        self._update_landmark(key, value)
        # Clipping bounds the reference implementation. A fused kernel should
        # use per-code log rescaling instead of clipping for higher fidelity.
        content_weight = torch.exp(score.clamp(min=-20.0, max=10.0))
        if self.content_threshold is not None:
            content_weight = torch.where(
                score >= self.content_threshold,
                content_weight,
                torch.zeros_like(content_weight),
            )
        age = rates.pow(self.window_size).view(1, 1, 1, self.num_scales)

        denominator_decay = rates.view(1, 1, 1, self.num_scales)
        numerator_decay = rates.view(1, 1, 1, self.num_scales, 1)
        numerator_add = (
            content_weight.unsqueeze(-1).unsqueeze(-1)
            * age.unsqueeze(-1)
            * value.to(self._numerator.dtype)[:, :, None, None, :]
        )
        denominator_add = content_weight.unsqueeze(-1) * age
        if torch.is_grad_enabled():
            # Functional assignments preserve a differentiable state during
            # teacher training. Inference uses the cheaper in-place path.
            self._denominator = self._denominator * denominator_decay + denominator_add
            self._numerator = self._numerator * numerator_decay + numerator_add
        else:
            self._denominator.mul_(denominator_decay).add_(denominator_add)
            self._numerator.mul_(numerator_decay).add_(numerator_add)

    @torch.no_grad()
    def _lazy_update(self, key: Tensor, value: Tensor) -> None:
        """Update only top-k code slots and apply decay on first touch."""

        assert self.active_codes is not None
        state_dtype = self._numerator.dtype
        codes = self.codes.to(device=key.device, dtype=state_dtype)
        scores = torch.einsum("bhd,hmd->bhm", key.to(state_dtype), codes)
        self._update_landmark(key, value)
        values, indices = torch.topk(scores, self.active_codes, dim=-1)
        self._step += 1
        if (
            self.use_triton
            and key.is_cuda
            and self.active_codes & (self.active_codes - 1) == 0
        ):
            from .triton_kernels import TRITON_AVAILABLE, triton_lazy_update_archive

            if TRITON_AVAILABLE:
                rates = self.decay_rates.to(device=key.device, dtype=state_dtype)
                triton_lazy_update_archive(
                    self._numerator,
                    self._denominator,
                    self._last_step,
                    key,
                    value,
                    codes,
                    indices,
                    rates,
                    self.window_size,
                    self._step,
                    self.content_threshold,
                )
                return
        index_scales = indices.unsqueeze(-1).expand(-1, -1, -1, self.num_scales)
        old_den = self._denominator.gather(2, index_scales)
        old_num = self._numerator.gather(
            2, index_scales.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        )
        old_step = self._last_step.gather(2, index_scales)
        delta = (self._step - old_step).clamp_min(0)
        rates = self.decay_rates.to(device=key.device, dtype=state_dtype)
        decay = rates.view(1, 1, 1, -1).pow(delta)
        selected_active = (
            torch.ones_like(values, dtype=torch.bool)
            if self.content_threshold is None
            else (values / math.sqrt(self.head_dim) >= self.content_threshold)
        )
        old_den = torch.where(
            selected_active.unsqueeze(-1), old_den * decay, old_den
        )
        old_num = torch.where(
            selected_active.unsqueeze(-1).unsqueeze(-1),
            old_num * decay.unsqueeze(-1),
            old_num,
        )
        content_weight = torch.exp(
            (values / math.sqrt(self.head_dim)).clamp(min=-20.0, max=10.0)
        )
        content_weight = torch.where(
            selected_active, content_weight, torch.zeros_like(content_weight)
        )
        age = rates.pow(self.window_size).view(1, 1, 1, -1)
        denominator_add = content_weight.unsqueeze(-1) * age
        numerator_add = (
            denominator_add.unsqueeze(-1)
            * value.to(state_dtype).unsqueeze(2).unsqueeze(3)
        )
        new_den = old_den + denominator_add
        new_num = old_num + numerator_add
        self._denominator.scatter_(2, index_scales, new_den)
        self._numerator.scatter_(
            2,
            index_scales.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim),
            new_num,
        )
        self._last_step.scatter_(
            2,
            index_scales,
            torch.where(
                selected_active.unsqueeze(-1),
                torch.full_like(old_step, self._step),
                old_step,
            ),
        )

    def _parallel_decay_scan(
        self, additions: Tensor, initial: Tensor, rates: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Scan ``state[t] = rates * state[t-1] + additions[t]`` in blocks.

        Rescaling is local to each block, avoiding underflow from ``rate **
        sequence_length`` while replacing one Python operation per token with
        one operation per configured ``scan_block_size`` block.
        """

        events = additions.shape[2]
        if events == 0:
            return additions, initial
        block_size = self.scan_block_size
        states: list[Tensor] = []
        state = initial
        for start in range(0, events, block_size):
            block = additions[:, :, start : start + block_size]
            block_length = block.shape[2]
            powers = rates.view(1, -1).pow(
                torch.arange(1, block_length + 1, device=additions.device).view(-1, 1)
            )
            if additions.ndim == 5:
                powers_view = powers.view(1, 1, block_length, 1, -1)
            else:
                powers_view = powers.view(1, 1, block_length, 1, -1, 1)
            scaled = block / powers_view
            cumulative = torch.cumsum(scaled, dim=2)
            state_view = state.unsqueeze(2)
            block_states = powers_view * (state_view + cumulative)
            states.append(block_states)
            state = block_states[:, :, -1]
        return torch.cat(states, dim=2), state

    @torch.no_grad()
    def update_read_chunk(
        self,
        key: Tensor,
        value: Tensor,
        query: Tensor,
        *,
        output: Optional[Tensor] = None,
    ) -> Tensor:
        """Update and read a sequence of evicted tokens with a block scan.

        Inputs are ``[batch, heads, events, head_dim]`` and the returned
        archive responses have the same leading dimensions as ``query``.
        ``step_chunk`` calls this only in inference mode; the differentiable
        reference remains the single-token ``update``/``read`` pair.
        """

        if key.ndim != 4 or value.shape != key.shape or query.shape != key.shape:
            raise ValueError("key, value, and query must have shape [batch, heads, events, head_dim]")
        if output is not None and (output.shape != query.shape or output.device != query.device):
            raise ValueError("output must match query shape and device")
        batch, heads, events, dim = key.shape
        if heads != self.num_heads or dim != self.head_dim:
            raise ValueError("chunk shapes do not match archive configuration")
        if events == 0:
            return query.new_empty(query.shape)
        if self._numerator.shape[0] != batch or self._numerator.device != key.device:
            self.reset_state(batch, device=key.device)

        # Update sticky landmarks once per block before dispatching to the
        # dense/sparse archive kernel.  This side state is tiny (one value per
        # code) and therefore does not alter the O(1) memory bound.
        self._update_landmark_chunk(key, value)

        if self.lazy_decay and self.use_triton and key.is_cuda:
            if self.active_codes is None:
                raise RuntimeError("lazy_decay requires active_codes")
            if self.active_codes & (self.active_codes - 1) == 0:
                from .triton_kernels import (
                    TRITON_AVAILABLE,
                    triton_sparse_update_read_archive_chunk,
                )

                if TRITON_AVAILABLE:
                    output = triton_sparse_update_read_archive_chunk(
                        key,
                        value,
                        query,
                        self._numerator,
                        self._denominator,
                        self._last_step,
                        self.codes,
                        self.mix_logits,
                        self.decay_rates,
                        self.window_size,
                        self._step,
                        self.active_codes,
                        block_size=self.scan_block_size,
                        output=output,
                        content_threshold=self.content_threshold,
                    )
                    self._step += events
                    return self._combine_landmark(query, output)

        if self.lazy_decay:
            outputs = []
            for index in range(events):
                self._lazy_update(key[:, :, index], value[:, :, index])
                outputs.append(self._lazy_read(query[:, :, index]))
            return torch.stack(outputs, dim=2)

        # Dense CUDA chunks use a two-launch fused update/read path.  The
        # previous implementation issued one update and one read launch per
        # event, which made chunked serving launch-bound even when the archive
        # state itself was tiny.  Sparse/lazy archives retain their dedicated
        # top-k path for now; CPU and unsupported devices use the block scan.
        if self.use_triton and key.is_cuda and self.active_codes is None:
            from .triton_kernels import TRITON_AVAILABLE, triton_update_read_archive_chunk

            if TRITON_AVAILABLE:
                result = triton_update_read_archive_chunk(
                    key,
                    value,
                    query,
                    self._numerator,
                    self._denominator,
                    self.codes,
                    self.mix_logits,
                    self.decay_rates,
                    self.window_size,
                    block_size=self.scan_block_size,
                    output=output,
                    content_threshold=self.content_threshold,
                )
                return self._combine_landmark(query, result)

        # Sparse/lazy CUDA chunks and unsupported devices use the reference
        # event path or block scan below.
        if self.use_triton and key.is_cuda:
            for index in range(events):
                self.update(key[:, :, index], value[:, :, index])
            return torch.stack(
                [self.read(query[:, :, index]) for index in range(events)], dim=2
            )

        state_dtype = self._numerator.dtype
        rates = self.decay_rates.to(device=key.device, dtype=state_dtype)
        codes = self.codes.to(device=key.device, dtype=state_dtype)
        # Stream the recurrence in bounded blocks.  Materializing all event
        # states at once costs O(events * num_codes * scales * head_dim), which
        # defeats long-context serving even though the persistent state itself
        # is constant-size.  A block is large enough to amortize tensor launch
        # overhead while keeping the temporary working set bounded.
        outputs: list[Tensor] = []
        state_den = self._denominator
        state_num = self._numerator
        age = rates.pow(self.window_size)
        for start in range(0, events, self.scan_block_size):
            end = min(events, start + self.scan_block_size)
            block_key = key[:, :, start:end]
            block_value = value[:, :, start:end]
            score = torch.einsum("bhed,hmd->bhem", block_key.to(state_dtype), codes)
            content_weight = torch.exp(
                (score / math.sqrt(dim)).clamp(min=-20.0, max=10.0)
            )
            if self.content_threshold is not None:
                content_weight = torch.where(
                    score / math.sqrt(dim) >= self.content_threshold,
                    content_weight,
                    torch.zeros_like(content_weight),
                )
            denominator_add = content_weight.unsqueeze(-1) * age.view(1, 1, 1, 1, -1)
            numerator_add = denominator_add.unsqueeze(-1) * block_value.to(state_dtype).unsqueeze(3).unsqueeze(4)
            denominator_states, state_den = self._parallel_decay_scan(
                denominator_add, state_den, rates
            )
            numerator_states, state_num = self._parallel_decay_scan(
                numerator_add, state_num, rates
            )
            outputs.append(
                self._read_states_chunk(
                    query[:, :, start:end], numerator_states, denominator_states
                )
            )
        self._denominator = state_den
        self._numerator = state_num
        result = torch.cat(outputs, dim=2)
        if output is not None:
            output.copy_(result)
            return output
        return result

    def _read_states(self, query: Tensor, numerator: Tensor, denominator: Tensor) -> Tensor:
        """Read one or many queries from explicit archive states."""

        codes = self.codes.to(device=query.device, dtype=self._numerator.dtype)
        routing_logits = torch.einsum(
            "bhd,hmd->bhm", query.to(codes.dtype), codes
        ) / math.sqrt(self.head_dim)
        active = self.active_codes
        if active is None or active >= self.num_codes or torch.is_grad_enabled():
            denom = denominator.clamp_min(1e-8)
            response = numerator / denom.unsqueeze(-1)
            mix = F.softmax(
                self.mix_logits.to(device=query.device, dtype=response.dtype), dim=-1
            )
            response = torch.einsum("hmj,bhmjd->bhmd", mix, response)
            routing = F.softmax(routing_logits, dim=-1).to(response.dtype)
            response = torch.einsum("bhm,bhmd->bhd", routing, response).to(query.dtype)
            return self._combine_landmark(query, response)

        values, indices = torch.topk(routing_logits, active, dim=-1)
        index_scales = indices.unsqueeze(-1).expand(-1, -1, -1, self.num_scales)
        selected_num = numerator.gather(
            2, index_scales.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        )
        selected_den = denominator.gather(2, index_scales)
        selected = selected_num / selected_den.clamp_min(1e-8).unsqueeze(-1)
        mix_logits = self.mix_logits.to(device=query.device).unsqueeze(0).expand(query.shape[0], -1, -1, -1).gather(
            2, index_scales
        )
        mix = F.softmax(mix_logits, dim=-1).to(selected.dtype)
        selected = (mix.unsqueeze(-1) * selected).sum(dim=3)
        routing = F.softmax(values, dim=-1).to(selected.dtype)
        response = (routing.unsqueeze(-1) * selected).sum(dim=2).to(query.dtype)
        return self._combine_landmark(query, response)

    @torch.no_grad()
    def _lazy_read(self, query: Tensor) -> Tensor:
        """Read top-k slots, materializing their elapsed decay on demand."""

        assert self.active_codes is not None
        state_dtype = self._numerator.dtype
        codes = self.codes.to(device=query.device, dtype=state_dtype)
        routing_logits = torch.einsum(
            "bhd,hmd->bhm", query.to(state_dtype), codes
        ) / math.sqrt(self.head_dim)
        values, indices = torch.topk(routing_logits, self.active_codes, dim=-1)
        if (
            self.use_triton
            and query.is_cuda
            and self.active_codes & (self.active_codes - 1) == 0
        ):
            from .triton_kernels import TRITON_AVAILABLE, triton_sparse_read_archive

            if TRITON_AVAILABLE:
                result = triton_sparse_read_archive(
                    query,
                    self._numerator,
                    self._denominator,
                    self._last_step,
                    codes,
                    self.mix_logits,
                    indices,
                    values,
                    self.decay_rates,
                    self._step,
                )
                landmark, valid = self._landmark_read(query)
                mix = torch.sigmoid(self.landmark_mix_logits).to(result.dtype)
                return torch.where(
                    valid.unsqueeze(-1),
                    (1.0 - mix.view(1, -1, 1)) * result
                    + mix.view(1, -1, 1) * landmark,
                    result,
                )
        index_scales = indices.unsqueeze(-1).expand(-1, -1, -1, self.num_scales)
        numerator = self._numerator.gather(
            2, index_scales.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        )
        denominator = self._denominator.gather(2, index_scales)
        last_step = self._last_step.gather(2, index_scales)
        rates = self.decay_rates.to(device=query.device, dtype=state_dtype)
        delta = (self._step - last_step).clamp_min(0)
        decay = rates.view(1, 1, 1, -1).pow(delta)
        numerator = numerator * decay.unsqueeze(-1)
        denominator = denominator * decay
        response = numerator / denominator.clamp_min(1e-8).unsqueeze(-1)
        mix_logits = self.mix_logits.to(device=query.device).unsqueeze(0).expand(query.shape[0], -1, -1, -1).gather(
            2, index_scales
        )
        mix = F.softmax(mix_logits, dim=-1).to(response.dtype)
        response = (mix.unsqueeze(-1) * response).sum(dim=3)
        routing = F.softmax(values, dim=-1).to(response.dtype)
        return (routing.unsqueeze(-1) * response).sum(dim=2).to(query.dtype)

    def _read_states_chunk(self, query: Tensor, numerator: Tensor, denominator: Tensor) -> Tensor:
        """Read a query block from explicit archive states."""

        codes = self.codes.to(device=query.device, dtype=self._numerator.dtype)
        routing_logits = torch.einsum(
            "bhed,hmd->bhem", query.to(codes.dtype), codes
        ) / math.sqrt(self.head_dim)
        active = self.active_codes
        if active is None or active >= self.num_codes:
            denom = denominator.clamp_min(1e-8)
            response = numerator / denom.unsqueeze(-1)
            mix = F.softmax(
                self.mix_logits.to(device=query.device, dtype=response.dtype), dim=-1
            )
            response = torch.einsum("hmj,bhemjd->bhemd", mix, response)
            routing = F.softmax(routing_logits, dim=-1).to(response.dtype)
            response = torch.einsum("bhem,bhemd->bhed", routing, response).to(query.dtype)
            return self._combine_landmark(query, response)

        values, indices = torch.topk(routing_logits, active, dim=-1)
        index_scales = indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.num_scales)
        selected_num = numerator.gather(
            3, index_scales.unsqueeze(-1).expand(-1, -1, -1, -1, -1, self.head_dim)
        )
        selected_den = denominator.gather(3, index_scales)
        selected = selected_num / selected_den.clamp_min(1e-8).unsqueeze(-1)
        mix_logits = self.mix_logits.to(device=query.device).unsqueeze(0).unsqueeze(2).expand(
            query.shape[0], -1, query.shape[2], -1, -1
        ).gather(3, index_scales)
        mix = F.softmax(mix_logits, dim=-1).to(selected.dtype)
        selected = (mix.unsqueeze(-1) * selected).sum(dim=4)
        routing = F.softmax(values, dim=-1).to(selected.dtype)
        response = (routing.unsqueeze(-1) * selected).sum(dim=3).to(query.dtype)
        return self._combine_landmark(query, response)

    def read(self, query: Tensor) -> Tensor:
        """Read archive response for queries of shape ``[batch, heads, head_dim]``."""

        if query.ndim != 3 or query.shape[1:] != (self.num_heads, self.head_dim):
            raise ValueError("query must have shape [batch, heads, head_dim]")
        if (
            query.shape[0] != self._numerator.shape[0]
            or query.device != self._numerator.device
        ):
            self.reset_state(query.shape[0], device=query.device)

        if self.lazy_decay and not torch.is_grad_enabled():
            return self._lazy_read(query)

        if (
            self.active_codes is None
            and self.use_triton
            and not torch.is_grad_enabled()
            and query.is_cuda
        ):
            from .triton_kernels import TRITON_AVAILABLE, triton_read_archive

            if TRITON_AVAILABLE:
                result = triton_read_archive(
                    query,
                    self._numerator,
                    self._denominator,
                    self.codes,
                    self.mix_logits,
                )
                return self._combine_landmark(query, result)

        return self._read_states(query, self._numerator, self._denominator)


class QCCSelfAttention(nn.Module):
    """Causal attention with exact local window and QCC long-range archive."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        num_codes: int = 16,
        num_scales: int = 4,
        window_size: int = 128,
        use_archive: bool = True,
        use_triton: bool = True,
        active_codes: Optional[int] = None,
        lazy_decay: bool = False,
        archive_read_stride: int = 1,
        archive_query_cosine_threshold: Optional[float] = None,
        archive_scan_block_size: int = 256,
        archive_content_threshold: Optional[float] = None,
        archive_persistent_landmark: bool = False,
        archive_prefix_landmark: bool = False,
        archive_prefix_pair_landmark: bool = False,
        rope_theta: Optional[float] = None,
        max_position_embeddings: int = 4096,
        decay_rates: Optional[tuple[float, ...]] = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if num_scales <= 0:
            raise ValueError("num_scales must be positive")
        if archive_read_stride <= 0:
            raise ValueError("archive_read_stride must be positive")
        if archive_query_cosine_threshold is not None and not -1.0 <= archive_query_cosine_threshold <= 1.0:
            raise ValueError("archive_query_cosine_threshold must be in [-1, 1]")
        if rope_theta is not None and rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")
        if decay_rates is not None and len(decay_rates) != num_scales:
            raise ValueError("decay_rates must contain exactly num_scales values")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.use_archive = use_archive
        self.rope_theta = rope_theta
        rotary_dim = 2 * (self.head_dim // 2)
        half_dim = rotary_dim // 2
        rope_inv_freq = (
            rope_theta ** (-torch.arange(half_dim, dtype=torch.float32) / max(half_dim, 1))
            if rope_theta is not None
            else torch.empty(0, dtype=torch.float32)
        )
        self.register_buffer("rope_inv_freq", rope_inv_freq, persistent=False)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, num_heads)
        # Choose half-lives from the exact window to the configured context so
        # a 1M-token model does not silently forget everything after a few
        # thousand updates.  Explicit rates remain available for ablations.
        if decay_rates is None:
            min_horizon = max(1.0, float(window_size))
            max_horizon = max(min_horizon, float(max_position_embeddings))
            horizons = torch.logspace(
                math.log10(min_horizon), math.log10(max_horizon), num_scales
            )
            rates = tuple(torch.exp(-math.log(2.0) / horizons).tolist())
        else:
            rates = decay_rates
        self.archive = QCCArchive(
            num_heads,
            self.head_dim,
            num_codes,
            rates,
            window_size,
            use_triton=use_triton,
            active_codes=active_codes,
            lazy_decay=lazy_decay,
            scan_block_size=archive_scan_block_size,
            content_threshold=archive_content_threshold,
            persistent_landmark=archive_persistent_landmark,
            prefix_pair_landmark=archive_prefix_pair_landmark,
        )
        self.archive_read_stride = archive_read_stride
        # Optional adaptive remote-read suppression. ``None`` keeps exact
        # archive reads; otherwise a read is skipped when the new query is
        # cosine-close to the previous refreshed query. The state is still
        # updated every token, so this knob changes only read freshness.
        self.archive_query_cosine_threshold = archive_query_cosine_threshold
        self._local_keys: list[Tensor] = []
        self._local_values: list[Tensor] = []
        self._local_key_cache: Optional[Tensor] = None
        self._local_value_cache: Optional[Tensor] = None
        self._full_key_cache: Optional[Tensor] = None
        self._full_value_cache: Optional[Tensor] = None
        self._cache_start = 0
        self._cache_length = 0
        self._seen_tokens = 0
        self._archive_read_cache: Optional[Tensor] = None
        self._archive_query_cache: Optional[Tensor] = None

    def _apply_rope(
        self, query: Tensor, key: Tensor, positions: Optional[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Apply rotary phases to q/k for an optional relative-position path."""

        if self.rope_theta is None or query.shape[-1] < 2:
            return query, key
        if positions is None:
            if query.ndim == 4:
                positions = torch.arange(query.shape[2], device=query.device)
            else:
                positions = torch.arange(query.shape[0], device=query.device)
        positions = positions.to(device=query.device)
        if query.ndim == 4:
            if positions.ndim == 1:
                positions = positions.unsqueeze(0)
            angles = positions.to(torch.float32).unsqueeze(-1) * self.rope_inv_freq
            cos = angles.cos().unsqueeze(1).to(query.dtype)
            sin = angles.sin().unsqueeze(1).to(query.dtype)
        elif query.ndim == 3:
            if positions.ndim == 0:
                positions = positions.expand(query.shape[0])
            angles = positions.to(torch.float32).unsqueeze(-1) * self.rope_inv_freq
            cos = angles.cos().unsqueeze(1).to(query.dtype)
            sin = angles.sin().unsqueeze(1).to(query.dtype)
        else:
            raise ValueError("q/k tensors must have rank 3 or 4")
        rotary_dim = self.rope_inv_freq.numel() * 2

        def rotate(tensor: Tensor) -> Tensor:
            prefix = tensor[..., :rotary_dim]
            suffix = tensor[..., rotary_dim:]
            pairs = prefix.reshape(*prefix.shape[:-1], -1, 2)
            first, second = pairs.unbind(dim=-1)
            rotated = torch.stack(
                (first * cos - second * sin, first * sin + second * cos), dim=-1
            ).flatten(-2)
            return torch.cat((rotated, suffix), dim=-1)

        return rotate(query), rotate(key)

    def _split_heads(self, x: Tensor) -> Tensor:
        bsz, length, _ = x.shape
        return x.view(bsz, length, self.num_heads, self.head_dim).transpose(1, 2)

    def reset_cache(self, batch_size: int, *, device: torch.device) -> None:
        """Reset the persistent state used by :meth:`step`."""

        self.archive.reset_state(batch_size, device=device)
        self._local_keys = []
        self._local_values = []
        self._local_key_cache = None
        self._local_value_cache = None
        self._full_key_cache = None
        self._full_value_cache = None
        self._chunk_key_scratch = None
        self._chunk_value_scratch = None
        self._cache_start = 0
        self._cache_length = 0
        self._seen_tokens = 0
        self._archive_read_cache = None
        self._archive_query_cache = None

    def _ordered_ring(self) -> tuple[Tensor, Tensor]:
        """Return valid ring contents in chronological order."""

        if self._local_key_cache is None:
            raise RuntimeError("local cache is not initialized")
        assert self._local_value_cache is not None
        if self._cache_length == 0:
            return self._local_key_cache[:, :, :0], self._local_value_cache[:, :, :0]
        if self._cache_start == 0:
            return (
                self._local_key_cache[:, :, : self._cache_length],
                self._local_value_cache[:, :, : self._cache_length],
            )
        return (
            torch.cat(
                (
                    self._local_key_cache[:, :, self._cache_start : self._cache_length],
                    self._local_key_cache[:, :, : self._cache_start],
                ), dim=2
            ),
            torch.cat(
                (
                    self._local_value_cache[:, :, self._cache_start : self._cache_length],
                    self._local_value_cache[:, :, : self._cache_start],
                ), dim=2
            ),
        )

    def _combined_local_chunk(
        self, key: Tensor, value: Tensor
    ) -> tuple[Tensor, Tensor, int]:
        """Build a chronological local block in reusable scratch storage.

        The persistent cache is a ring to make eviction O(1).  Attention
        kernels need chronological keys, however, and allocating two fresh
        ``cat`` tensors at every decode chunk adds allocator traffic and a
        second copy at ring wrap-around.  Reusing scratch storage keeps that
        temporary allocation out of the steady-state path.
        """

        if self._local_key_cache is None or self._local_value_cache is None:
            raise RuntimeError("local cache must be initialized before combining a chunk")
        bsz, _, length, _ = key.shape
        old_length = self._cache_length
        needed = old_length + length
        if (
            self._chunk_key_scratch is None
            or self._chunk_value_scratch is None
            or self._chunk_key_scratch.shape[0] != bsz
            or self._chunk_key_scratch.shape[2] < needed
            or self._chunk_key_scratch.device != key.device
            or self._chunk_key_scratch.dtype != key.dtype
            or self._chunk_value_scratch.device != value.device
            or self._chunk_value_scratch.dtype != value.dtype
        ):
            capacity = max(needed, self.window_size + length)
            scratch_shape = (bsz, self.num_heads, capacity, self.head_dim)
            self._chunk_key_scratch = torch.empty(
                scratch_shape, device=key.device, dtype=key.dtype
            )
            self._chunk_value_scratch = torch.empty(
                scratch_shape, device=value.device, dtype=value.dtype
            )
        assert self._chunk_value_scratch is not None
        if old_length:
            start = self._cache_start
            first = min(old_length, self.window_size - start)
            self._chunk_key_scratch[:, :, :first] = self._local_key_cache[
                :, :, start : start + first
            ]
            self._chunk_value_scratch[:, :, :first] = self._local_value_cache[
                :, :, start : start + first
            ]
            if first < old_length:
                remainder = old_length - first
                self._chunk_key_scratch[:, :, first:old_length] = self._local_key_cache[
                    :, :, :remainder
                ]
                self._chunk_value_scratch[:, :, first:old_length] = self._local_value_cache[
                    :, :, :remainder
                ]
        self._chunk_key_scratch[:, :, old_length:needed] = key
        self._chunk_value_scratch[:, :, old_length:needed] = value
        return (
            self._chunk_key_scratch[:, :, :needed],
            self._chunk_value_scratch[:, :, :needed],
            old_length,
        )

    def step(
        self,
        hidden: Tensor,
        *,
        reset_cache: bool = False,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Decode one token with bounded local KV plus recurrent archive state.

        ``hidden`` is ``[batch, d_model]``. This method is intended for
        ``torch.no_grad()`` serving; unlike :meth:`forward`, it does not replay
        the prefix and therefore exposes the constant-history read path.
        """

        if hidden.ndim != 2 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [batch, d_model]")
        bsz = hidden.shape[0]
        if (
            reset_cache
            or self.archive._numerator.shape[0] != bsz
            or self.archive._numerator.device != hidden.device
        ):
            self.reset_cache(bsz, device=hidden.device)
        q = self._split_heads(self.q_proj(hidden[:, None]))[:, :, 0]
        key = self._split_heads(self.k_proj(hidden[:, None]))[:, :, 0]
        value = self._split_heads(self.v_proj(hidden[:, None]))[:, :, 0]
        if position_ids is None:
            position_ids = torch.full(
                (bsz,), self._seen_tokens, device=hidden.device, dtype=torch.long
            )
        q, key = self._apply_rope(q, key, position_ids)
        if self.use_archive:
            if self._local_key_cache is None:
                shape = (bsz, self.num_heads, self.window_size, self.head_dim)
                self._local_key_cache = torch.empty(shape, device=key.device, dtype=key.dtype)
                self._local_value_cache = torch.empty(shape, device=value.device, dtype=value.dtype)
            assert self._local_value_cache is not None
            if self._cache_length < self.window_size:
                write_index = (self._cache_start + self._cache_length) % self.window_size
                self._cache_length += 1
            else:
                write_index = self._cache_start
                self.archive.update(self._local_key_cache[:, :, write_index], self._local_value_cache[:, :, write_index])
                self._cache_start = (self._cache_start + 1) % self.window_size
            self._local_key_cache[:, :, write_index] = key
            self._local_value_cache[:, :, write_index] = value
            # Attention over one query is permutation-invariant over its K/V
            # set. Keep the physical ring order and avoid copying/rotating the
            # window on every token; ``_cache_start`` is only the eviction slot.
            local_keys = self._local_key_cache[:, :, : self._cache_length]
            local_values = self._local_value_cache[:, :, : self._cache_length]
        else:
            if self._full_key_cache is None:
                shape = (bsz, self.num_heads, self.window_size, self.head_dim)
                self._full_key_cache = torch.empty(shape, device=key.device, dtype=key.dtype)
                self._full_value_cache = torch.empty(shape, device=value.device, dtype=value.dtype)
            if self._seen_tokens >= self.window_size:
                raise ValueError("full-KV cache exceeds configured maximum length")
            assert self._full_value_cache is not None
            self._full_key_cache[:, :, self._seen_tokens] = key
            self._full_value_cache[:, :, self._seen_tokens] = value
            local_keys = self._full_key_cache[:, :, : self._seen_tokens + 1]
            local_values = self._full_value_cache[:, :, : self._seen_tokens + 1]
        if self.use_archive:
            # SDPA handles the small local window with one fused primitive;
            # this is materially cheaper than several per-token einsums on
            # both CPU flash-attention and CUDA backends.
            local_out = F.scaled_dot_product_attention(
                q.unsqueeze(2), local_keys, local_values, dropout_p=0.0
            ).squeeze(2)
        else:
            # Use the same fused SDPA primitive as the block path for an honest
            # full-KV serving baseline. All cached keys are valid for this
            # single query because they precede (or equal) the current token.
            valid = torch.ones(
                (1, local_keys.shape[2]), device=hidden.device, dtype=torch.bool
            )
            local_out = F.scaled_dot_product_attention(
                q.unsqueeze(2),
                local_keys,
                local_values,
                attn_mask=valid,
                dropout_p=0.0,
            ).squeeze(2)
        if self.use_archive and self._seen_tokens >= self.window_size:
            refresh = (
                self._archive_read_cache is None
                or self.archive_read_stride == 1
                or self._seen_tokens % self.archive_read_stride == 0
            )
            if (
                refresh
                and self.archive_query_cosine_threshold is not None
                and self._archive_query_cache is not None
            ):
                # Archive reads are batched over heads.  Skip the whole read
                # only when every head sees a stable query; this avoids mixing
                # fresh and stale heads while keeping the decision scalar.
                similarity = F.cosine_similarity(q, self._archive_query_cache, dim=-1)
                refresh = bool(
                    torch.any(similarity < self.archive_query_cosine_threshold).item()
                )
            if refresh:
                self._archive_read_cache = self.archive.read(q)
                self._archive_query_cache = q.detach()
            assert self._archive_read_cache is not None
            archive_out = self._archive_read_cache
            gate = (
                torch.zeros_like(self.gate(hidden)).unsqueeze(-1)
                if self.archive.prefix_landmark
                else torch.sigmoid(self.gate(hidden)).unsqueeze(-1)
            )
            head_out = gate * local_out + (1.0 - gate) * archive_out
        else:
            head_out = local_out
        self._seen_tokens += 1
        return self.out_proj(head_out.reshape(bsz, self.d_model))

    @torch.no_grad()
    def step_chunk(
        self,
        hidden: Tensor,
        *,
        reset_cache: bool = False,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Decode a causal block while preserving the persistent cache.

        Projections and local attention are vectorized over the block. Archive
        writes/reads remain ordered because each position may evict a different
        historical slot.
        """

        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        bsz, length, _ = hidden.shape
        if length == 0:
            return hidden
        if (
            reset_cache
            or self.archive._numerator.shape[0] != bsz
            or self.archive._numerator.device != hidden.device
        ):
            self.reset_cache(bsz, device=hidden.device)
        q = self._split_heads(self.q_proj(hidden))
        k = self._split_heads(self.k_proj(hidden))
        v = self._split_heads(self.v_proj(hidden))
        if position_ids is None:
            position_ids = self._seen_tokens + torch.arange(
                length, device=hidden.device, dtype=torch.long
            )
        q, k = self._apply_rope(q, k, position_ids)

        if self.use_archive:
            if self._local_key_cache is None:
                shape = (bsz, self.num_heads, self.window_size, self.head_dim)
                self._local_key_cache = torch.empty(shape, device=k.device, dtype=k.dtype)
                self._local_value_cache = torch.empty(shape, device=v.device, dtype=v.dtype)
            combined_k, combined_v, old_length = self._combined_local_chunk(k, v)
        else:
            if self._full_key_cache is None:
                shape = (bsz, self.num_heads, self.window_size, self.head_dim)
                self._full_key_cache = torch.empty(shape, device=k.device, dtype=k.dtype)
                self._full_value_cache = torch.empty(shape, device=v.device, dtype=v.dtype)
            if self._seen_tokens + length > self.window_size:
                raise ValueError("full-KV cache exceeds configured maximum length")
            assert self._full_value_cache is not None
            self._full_key_cache[:, :, self._seen_tokens : self._seen_tokens + length] = k
            self._full_value_cache[:, :, self._seen_tokens : self._seen_tokens + length] = v
            old_k = self._full_key_cache[:, :, : self._seen_tokens]
            old_v = self._full_value_cache[:, :, : self._seen_tokens]
            old_length = old_k.shape[2]
            combined_k = torch.cat((old_k, k), dim=2)
            combined_v = torch.cat((old_v, v), dim=2)
        total_length = combined_k.shape[2]

        if not self.use_archive:
            # The full-KV control uses PyTorch's fused SDPA with a causal mask
            # offset by the already-cached prefix. This avoids materializing a
            # quadratic [query, key, head_dim] window in the control itself.
            key_positions = torch.arange(total_length, device=hidden.device)
            query_positions = old_length + torch.arange(length, device=hidden.device)
            causal_mask = key_positions[None, :] <= query_positions[:, None]
            head_out = F.scaled_dot_product_attention(
                q,
                combined_k,
                combined_v,
                attn_mask=causal_mask,
                dropout_p=0.0,
            )
            self._seen_tokens += length
            return self.out_proj(
                head_out.transpose(1, 2).reshape(bsz, length, self.d_model)
            )

        # Feed the finite causal band directly to SDPA.  Unlike materializing
        # an unfolded [batch, head, time, window, dim] tensor, this lets the
        # backend use its fused attention implementation while the mask keeps
        # work bounded to the exact local window.  ``combined_k`` is already
        # chronological (old ring contents followed by the new block).
        positions = old_length + torch.arange(length, device=hidden.device)
        key_positions = torch.arange(total_length, device=hidden.device)
        valid = (key_positions[None, :] <= positions[:, None]) & (
            key_positions[None, :] >= positions[:, None] - self.window_size + 1
        )
        local_out = F.scaled_dot_product_attention(
            q,
            combined_k,
            combined_v,
            attn_mask=valid,
            dropout_p=0.0,
        )

        if self.use_archive:
            archive_out = torch.zeros_like(local_out)
            event_start = max(0, self.window_size - old_length)
            event_count = length - event_start
            if event_count > 0:
                evicted_k = combined_k[:, :, :event_count]
                evicted_v = combined_v[:, :, :event_count]
                self.archive.update_read_chunk(
                    evicted_k,
                    evicted_v,
                    q[:, :, event_start:],
                    output=archive_out[:, :, event_start:],
                )
            gate = (
                torch.zeros_like(self.gate(hidden)).transpose(1, 2).unsqueeze(-1)
                if self.archive.prefix_landmark
                else torch.sigmoid(self.gate(hidden)).transpose(1, 2).unsqueeze(-1)
            )
            mixed_out = gate * local_out + (1.0 - gate) * archive_out
            active = (
                self._seen_tokens + torch.arange(length, device=hidden.device)
                >= self.window_size
            ).view(1, 1, length, 1)
            head_out = torch.where(active, mixed_out, local_out)
            keep = min(self.window_size, total_length)
            assert self._local_key_cache is not None and self._local_value_cache is not None
            # Preserve the ring layout instead of clearing and rewriting the
            # entire window.  A chunk may wrap around the physical end, so
            # split the tail copy into at most two contiguous slices.  This
            # makes cache maintenance proportional to the number of retained
            # tokens rather than an additional O(window_size) memset.
            evicted = max(0, total_length - self.window_size)
            new_start = (self._cache_start + evicted) % self.window_size
            tail_k = combined_k[:, :, -keep:]
            tail_v = combined_v[:, :, -keep:]
            first = min(keep, self.window_size - new_start)
            self._local_key_cache[:, :, new_start : new_start + first] = tail_k[:, :, :first]
            self._local_value_cache[:, :, new_start : new_start + first] = tail_v[:, :, :first]
            if first < keep:
                remainder = keep - first
                self._local_key_cache[:, :, :remainder] = tail_k[:, :, first:]
                self._local_value_cache[:, :, :remainder] = tail_v[:, :, first:]
            self._cache_start = new_start
            self._cache_length = keep
        self._seen_tokens += length
        return self.out_proj(head_out.transpose(1, 2).reshape(bsz, length, self.d_model))

    def _forward_train_chunked(self, hidden: Tensor, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Differentiable bounded-memory path for long training sequences.

        The original teacher path intentionally spells out one token at a time
        so the recurrence is easy to inspect, but that makes a 128K training
        example both launch-bound and prohibitively expensive to backpropagate
        through.  This path keeps the same equations while scanning bounded
        blocks: local attention uses a causal band mask and archive states use
        :meth:`QCCArchive._parallel_decay_scan`, which is fully differentiable.
        It is selected only on CUDA; the short CPU reference remains the
        pedagogical implementation.
        """

        if not self.use_archive:
            return self.out_proj(
                F.scaled_dot_product_attention(q, k, v, is_causal=True)
                .transpose(1, 2)
                .reshape(hidden.shape[0], hidden.shape[1], self.d_model)
            )
        bsz, length, _ = hidden.shape
        window = min(self.window_size, length)
        block_size = self.archive.scan_block_size
        local_outputs: list[Tensor] = []
        for start in range(0, length, block_size):
            end = min(length, start + block_size)
            key_start = max(0, start - window + 1)
            key_positions = torch.arange(key_start, end, device=hidden.device)
            query_positions = torch.arange(start, end, device=hidden.device)
            valid = (key_positions[None, :] <= query_positions[:, None]) & (
                key_positions[None, :] >= query_positions[:, None] - window + 1
            )
            local_outputs.append(
                F.scaled_dot_product_attention(
                    q[:, :, start:end],
                    k[:, :, key_start:end],
                    v[:, :, key_start:end],
                    attn_mask=valid,
                    dropout_p=0.0,
                )
            )
        local_out = torch.cat(local_outputs, dim=2)

        state_dtype = self.archive._numerator.dtype
        rates = self.archive.decay_rates.to(device=hidden.device, dtype=state_dtype)
        codes = self.archive.codes.to(device=hidden.device, dtype=state_dtype)
        age = rates.pow(self.window_size)
        state_den = torch.zeros(
            bsz,
            self.num_heads,
            self.archive.num_codes,
            self.archive.num_scales,
            device=hidden.device,
            dtype=state_dtype,
        )
        state_num = torch.zeros(
            bsz,
            self.num_heads,
            self.archive.num_codes,
            self.archive.num_scales,
            self.head_dim,
            device=hidden.device,
            dtype=state_dtype,
        )
        archive_outputs: list[Tensor] = []
        landmark_outputs: list[Tensor] = []
        landmark_valid_outputs: list[Tensor] = []
        landmark_score_state: Optional[Tensor] = None
        landmark_value_state: Optional[Tensor] = None
        landmark_key_state: Optional[Tensor] = None
        if self.archive.persistent_landmark:
            landmark_score_state = torch.full(
                (bsz, self.num_heads, self.archive.num_codes),
                -torch.inf,
                device=hidden.device,
                dtype=state_dtype,
            )
            landmark_value_state = torch.zeros(
                bsz,
                self.num_heads,
                self.archive.num_codes,
                self.head_dim,
                device=hidden.device,
                dtype=state_dtype,
            )
            landmark_key_state = torch.zeros(
                bsz,
                self.num_heads,
                self.archive.num_codes,
                self.head_dim,
                device=hidden.device,
                dtype=state_dtype,
            )
        event_count = max(0, length - window)
        # The first ``window`` tokens are still in exact local KV and must not
        # enter the historical archive.  The inference path begins its archive
        # recurrence at the same boundary; keeping this offset here is crucial
        # for a train/inference-equivalent checkpoint.
        for start in range(0, event_count, block_size):
            end = min(event_count, start + block_size)
            block_key = k[:, :, start:end]
            block_value = v[:, :, start:end]
            score = torch.einsum(
                "bhed,hmd->bhem", block_key.to(state_dtype), codes
            ) / math.sqrt(self.head_dim)
            content_weight = torch.exp(score.clamp(min=-20.0, max=10.0))
            if self.archive.content_threshold is not None:
                # Keep a smooth surrogate while gradients are enabled so a
                # value token just below the hard inference threshold still
                # receives a signal to become salient.  No-grad/reference and
                # Triton paths retain the exact hard gate.
                if torch.is_grad_enabled():
                    smooth_gate = torch.sigmoid(
                        (score - self.archive.content_threshold) / 0.1
                    )
                    hard_gate = (score >= self.archive.content_threshold).to(
                        content_weight.dtype
                    )
                    # Straight-through estimator: preserve the exact hard
                    # threshold in the forward pass (matching inference),
                    # while exposing a smooth derivative to train scores that
                    # sit just below the threshold.
                    gate = hard_gate + smooth_gate - smooth_gate.detach()
                    content_weight = content_weight * gate
                else:
                    content_weight = torch.where(
                        score >= self.archive.content_threshold,
                        content_weight,
                        torch.zeros_like(content_weight),
                    )
            denominator_add = content_weight.unsqueeze(-1) * age.view(1, 1, 1, 1, -1)
            numerator_add = denominator_add.unsqueeze(-1) * block_value.to(
                state_dtype
            ).unsqueeze(3).unsqueeze(4)
            denominator_states, state_den = self.archive._parallel_decay_scan(
                denominator_add, state_den, rates
            )
            numerator_states, state_num = self.archive._parallel_decay_scan(
                numerator_add, state_num, rates
            )
            archive_outputs.append(
                self.archive._read_states_chunk(
                    q[:, :, window + start : window + end],
                    numerator_states,
                    denominator_states,
                )
            )
            if self.archive.persistent_landmark:
                assert (
                    landmark_score_state is not None
                    and landmark_value_state is not None
                    and landmark_key_state is not None
                )
                landmark_scores = score
                if self.archive.content_threshold is not None:
                    landmark_scores = torch.where(
                        landmark_scores >= self.archive.content_threshold,
                        landmark_scores,
                        torch.full_like(landmark_scores, -torch.inf),
                    )
                block_scores, block_indices = torch.cummax(landmark_scores, dim=2)
                bsz_, heads_, block_len, codes_ = block_scores.shape
                value_index = block_indices.unsqueeze(-1).expand(
                    bsz_, heads_, block_len, codes_, self.head_dim
                )
                value_expanded = block_value.to(state_dtype).unsqueeze(3).expand(
                    bsz_, heads_, block_len, codes_, self.head_dim
                )
                block_values = value_expanded.gather(2, value_index)
                key_expanded = block_key.to(state_dtype).unsqueeze(3).expand(
                    bsz_, heads_, block_len, codes_, self.head_dim
                )
                block_keys = key_expanded.gather(2, value_index)
                prior_scores = landmark_score_state.unsqueeze(2)
                use_block = block_scores > prior_scores
                running_scores = torch.where(use_block, block_scores, prior_scores)
                prior_values = landmark_value_state.unsqueeze(2).expand(
                    bsz_, heads_, block_len, codes_, self.head_dim
                )
                running_values = torch.where(
                    use_block.unsqueeze(-1), block_values, prior_values
                )
                prior_keys = landmark_key_state.unsqueeze(2).expand(
                    bsz_, heads_, block_len, codes_, self.head_dim
                )
                running_keys = torch.where(
                    use_block.unsqueeze(-1), block_keys, prior_keys
                )
                query_block = q[:, :, window + start : window + end]
                landmark_routing = F.softmax(
                    torch.einsum(
                        "bhed,bhemd->bhem", query_block.to(state_dtype), running_keys
                    )
                    / math.sqrt(self.head_dim),
                    dim=-1,
                ).to(state_dtype)
                landmark_outputs.append(
                    torch.einsum(
                        "bhem,bhemd->bhed", landmark_routing, running_values
                    ).to(query_block.dtype)
                )
                landmark_valid_outputs.append(
                    torch.isfinite(running_scores).any(dim=-1)
                )
                landmark_score_state = running_scores[:, :, -1]
                landmark_value_state = running_values[:, :, -1]
                landmark_key_state = running_keys[:, :, -1]
        archive_out = torch.zeros_like(local_out)
        if archive_outputs:
            archive_out[:, :, window:] = torch.cat(archive_outputs, dim=2)
        if landmark_outputs:
            landmark_out = torch.zeros_like(local_out)
            landmark_out[:, :, window:] = torch.cat(landmark_outputs, dim=2)
            landmark_valid = torch.zeros(
                (bsz, self.num_heads, length), device=hidden.device, dtype=torch.bool
            )
            landmark_valid[:, :, window:] = torch.cat(landmark_valid_outputs, dim=2)
            landmark_mix = (
                torch.ones_like(self.archive.landmark_mix_logits)
                if self.archive.prefix_landmark
                else torch.sigmoid(self.archive.landmark_mix_logits)
            ).to(archive_out.dtype).view(1, self.num_heads, 1, 1)
            archive_out = torch.where(
                landmark_valid.unsqueeze(-1),
                (1.0 - landmark_mix) * archive_out + landmark_mix * landmark_out,
                archive_out,
            )
            self.archive._landmark_score = landmark_score_state.detach()
            self.archive._landmark_value = landmark_value_state.detach()
            self.archive._landmark_key = landmark_key_state.detach()
        gate = (
            torch.zeros_like(self.gate(hidden)).transpose(1, 2).unsqueeze(-1)
            if self.archive.prefix_landmark
            else torch.sigmoid(self.gate(hidden)).transpose(1, 2).unsqueeze(-1)
        )
        mixed_out = gate * local_out + (1.0 - gate) * archive_out
        active = (torch.arange(length, device=hidden.device) >= window).view(
            1, 1, length, 1
        )
        head_out = torch.where(active, mixed_out, local_out)
        # Keep a detached snapshot for inspection without retaining the full
        # sequence graph in the mutable streaming state.
        self.archive._numerator = state_num.detach()
        self.archive._denominator = state_den.detach()
        return self.out_proj(head_out.transpose(1, 2).reshape(bsz, length, self.d_model))

    def _forward_inference(self, hidden: Tensor, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Vectorized local path used by evaluation/decode (no autograd)."""

        bsz, length, _ = hidden.shape
        window = min(self.window_size, length)
        if hidden.is_cuda or hidden.device.type == "mps":
            # On accelerator backends, use the fused SDPA primitive on bounded
            # blocks instead of materializing an unfolded [time, window, dim]
            # tensor and launching separate einsums for logits and values. The
            # key slice contains at most ``window + block_size - 1`` tokens;
            # the boolean mask keeps exact causal local-window semantics.
            block_size = self.archive.scan_block_size
            local_outputs: list[Tensor] = []
            for start in range(0, length, block_size):
                end = min(length, start + block_size)
                key_start = max(0, start - window + 1)
                key_positions = torch.arange(key_start, end, device=hidden.device)
                query_positions = torch.arange(start, end, device=hidden.device)
                valid = (key_positions[None, :] <= query_positions[:, None]) & (
                    key_positions[None, :] >= query_positions[:, None] - window + 1
                )
                local_outputs.append(
                    F.scaled_dot_product_attention(
                        q[:, :, start:end],
                        k[:, :, key_start:end],
                        v[:, :, key_start:end],
                        attn_mask=valid,
                        dropout_p=0.0,
                    )
                )
            local_out = torch.cat(local_outputs, dim=2)
        else:
            # The CPU SDPA backend currently pays a relatively high per-block
            # mask setup cost. Keep the reference's single vectorized unfold
            # there; it remains exact and avoids thousands of tiny dispatches.
            k_pad = F.pad(k.transpose(-1, -2), (window - 1, 0))
            v_pad = F.pad(v.transpose(-1, -2), (window - 1, 0))
            k_windows = k_pad.unfold(-1, window, 1).permute(0, 1, 3, 4, 2)
            v_windows = v_pad.unfold(-1, window, 1).permute(0, 1, 3, 4, 2)
            local_logits = torch.einsum("bhtd,bhtwd->bhtw", q, k_windows)
            local_logits = local_logits / math.sqrt(self.head_dim)
            valid = torch.arange(window, device=hidden.device)[None, :] >= (
                window - 1 - torch.arange(length, device=hidden.device)[:, None]
            )
            local_logits = local_logits.masked_fill(
                ~valid[None, None], torch.finfo(local_logits.dtype).min
            )
            local_prob = F.softmax(local_logits, dim=-1)
            local_out = torch.einsum("bhtw,bhtwd->bhtd", local_prob, v_windows)

        if self.use_archive:
            self.archive.reset_state(bsz, device=hidden.device)
            archive_out = torch.zeros_like(local_out)
            # The archive state is a linear recurrence.  Feed all evicted
            # tokens through the block scan at once instead of launching one
            # Python-level update/read pair per position.  This preserves the
            # post-update read semantics (the first event corresponds to
            # position ``window_size``) while making prefill overhead scale in
            # tensor blocks rather than interpreter iterations.
            event_count = length - self.window_size
            if event_count > 0:
                self.archive.update_read_chunk(
                    k[:, :, :event_count],
                    v[:, :, :event_count],
                    q[:, :, self.window_size :],
                    output=archive_out[:, :, self.window_size :],
                )
            gate = torch.sigmoid(self.gate(hidden)).transpose(1, 2).unsqueeze(-1)
            mixed_out = gate * local_out + (1.0 - gate) * archive_out
            active = (torch.arange(length, device=hidden.device) >= self.window_size).view(1, 1, length, 1)
            head_out = torch.where(active, mixed_out, local_out)
        else:
            # Full-attention baseline uses PyTorch's fused causal kernel.
            head_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(head_out.transpose(1, 2).reshape(bsz, length, self.d_model))

    def forward(
        self,
        hidden: Tensor,
        *,
        reset_state: bool = True,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        bsz, length, _ = hidden.shape
        q = self._split_heads(self.q_proj(hidden))
        k = self._split_heads(self.k_proj(hidden))
        v = self._split_heads(self.v_proj(hidden))
        if position_ids is None:
            position_ids = torch.arange(length, device=hidden.device, dtype=torch.long)
        q, k = self._apply_rope(q, k, position_ids)
        if not torch.is_grad_enabled():
            return self._forward_inference(hidden, q, k, v)
        if hidden.is_cuda and length > self.window_size and not self.archive.prefix_landmark:
            # CUDA training uses the same bounded block equations as
            # inference, but keeps the scan differentiable.  This avoids the
            # O(sequence-length) Python/autograd loop for long retrieval
            # curriculum examples.
            if (
                reset_state
                or self.archive._numerator.shape[0] != bsz
                or self.archive._numerator.device != hidden.device
            ):
                self.archive.reset_state(bsz, device=hidden.device)
            return self._forward_train_chunked(hidden, q, k, v)
        if (
            reset_state
            or self.archive._numerator.shape[0] != bsz
            or self.archive._numerator.device != hidden.device
        ):
            self.archive.reset_state(bsz, device=hidden.device)

        local_keys: list[Tensor] = []
        local_values: list[Tensor] = []
        outputs: list[Tensor] = []
        scale = 1.0 / math.sqrt(self.head_dim)
        for t in range(length):
            kt, vt = k[:, :, t], v[:, :, t]
            local_keys.append(kt)
            local_values.append(vt)
            if len(local_keys) > self.window_size:
                self.archive.update(local_keys.pop(0), local_values.pop(0))

            lk = torch.stack(local_keys, dim=2)
            lv = torch.stack(local_values, dim=2)
            local_logits = torch.einsum("bhd,bhld->bhl", q[:, :, t], lk) * scale
            local_prob = F.softmax(local_logits, dim=-1)
            local_out = torch.einsum("bhl,bhld->bhd", local_prob, lv)
            if self.use_archive and t >= self.window_size:
                archive_out = self.archive.read(q[:, :, t])
                gate = torch.sigmoid(self.gate(hidden[:, t])).unsqueeze(-1)
                head_out = gate * local_out + (1.0 - gate) * archive_out
            else:
                head_out = local_out
            # Concatenate heads in head-major order, matching the vectorized
            # inference path and standard multi-head attention layouts.
            outputs.append(head_out.reshape(bsz, self.d_model))

        return self.out_proj(torch.stack(outputs, dim=1))


class QCCDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, **attention_kwargs: object) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = QCCSelfAttention(d_model, num_heads, **attention_kwargs)
        self.norm2 = nn.LayerNorm(d_model)
        hidden_dim = 4 * d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        *,
        reset_state: bool,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.attention(
            self.norm1(x), reset_state=reset_state, position_ids=position_ids
        )
        return x + self.mlp(self.norm2(x))

    def step_chunk(
        self,
        x: Tensor,
        *,
        reset_cache: bool = False,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.attention.step_chunk(
            self.norm1(x), reset_cache=reset_cache, position_ids=position_ids
        )
        return x + self.mlp(self.norm2(x))

    def step(
        self,
        x: Tensor,
        *,
        reset_cache: bool = False,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.attention.step(
            self.norm1(x), reset_cache=reset_cache, position_ids=position_ids
        )
        return x + self.mlp(self.norm2(x))


class QCCForCausalLM(nn.Module):
    """Small decoder-only model suitable for architecture experiments."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_position_embeddings: int = 4096,
        window_size: int = 128,
        num_codes: int = 16,
        dropout: float = 0.0,
        position_encoding: str = "sinusoidal",
        rope_theta: float = 1_000_000.0,
        use_archive: bool = True,
        use_triton: bool = True,
        active_codes: Optional[int] = None,
        lazy_decay: bool = False,
        archive_read_stride: int = 1,
        archive_query_cosine_threshold: Optional[float] = None,
        archive_scan_block_size: int = 256,
        archive_content_threshold: Optional[float] = None,
        archive_persistent_landmark: bool = False,
        archive_prefix_landmark: bool = False,
        archive_prefix_pair_landmark: bool = False,
        archive_decay_rates: Optional[tuple[float, ...]] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        if position_encoding == "sinusoidal":
            self.position_embedding = SinusoidalPositionEmbedding(d_model)
        elif position_encoding == "learned":
            self.position_embedding = nn.Embedding(max_position_embeddings, d_model)
        elif position_encoding == "rope":
            if rope_theta <= 0:
                raise ValueError("rope_theta must be positive")
            self.position_embedding = None
        elif position_encoding == "none":
            # Content-only ablation for retrieval experiments.  Keeping this
            # explicit avoids silently treating a missing positional module as
            # a production-ready long-context policy.
            self.position_embedding = None
        else:
            raise ValueError(
                "position_encoding must be 'sinusoidal', 'learned', 'rope', or 'none'"
            )
        self.position_encoding = position_encoding
        self.layers = nn.ModuleList(
            QCCDecoderLayer(
                d_model,
                num_heads,
                dropout,
                window_size=window_size,
                num_codes=num_codes,
                use_archive=use_archive,
                use_triton=use_triton,
                active_codes=active_codes,
                lazy_decay=lazy_decay,
                archive_read_stride=archive_read_stride,
                archive_query_cosine_threshold=archive_query_cosine_threshold,
                archive_scan_block_size=archive_scan_block_size,
                archive_content_threshold=archive_content_threshold,
                archive_persistent_landmark=archive_persistent_landmark,
                archive_prefix_landmark=archive_prefix_landmark,
                archive_prefix_pair_landmark=archive_prefix_pair_landmark,
                rope_theta=rope_theta if position_encoding == "rope" else None,
                max_position_embeddings=max_position_embeddings,
                decay_rates=archive_decay_rates,
            )
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.max_position_embeddings = max_position_embeddings
        self._cache_position = 0

    def forward(self, input_ids: Tensor, *, reset_state: bool = True) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, length = input_ids.shape
        if length > self.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        token = self.token_embedding(input_ids)
        if self.position_embedding is None:
            x = token
        else:
            x = token + self.position_embedding(positions).to(token.dtype)
        for index, layer in enumerate(self.layers):
            x = layer(
                x,
                reset_state=reset_state or index > 0,
                position_ids=positions,
            )
        return self.lm_head(self.norm(x))

    def reset_cache(self, batch_size: int = 1) -> None:
        """Reset all layer caches before an independent generation stream."""

        for layer in self.layers:
            layer.attention.reset_cache(batch_size, device=self.token_embedding.weight.device)
        self._cache_position = 0
        self._cache_batch_size = batch_size

    @torch.no_grad()
    def decode_step(self, input_ids: Tensor, *, reset_cache: bool = False) -> Tensor:
        """Return logits for one token per batch element using persistent caches."""

        if input_ids.ndim == 2 and input_ids.shape[1] == 1:
            input_ids = input_ids[:, 0]
        if input_ids.ndim != 1:
            raise ValueError("decode_step input_ids must have shape [batch] or [batch, 1]")
        if self._cache_position >= self.max_position_embeddings:
            raise ValueError("decode position exceeds max_position_embeddings")
        bsz = input_ids.shape[0]
        cache_batch = getattr(self, "_cache_batch_size", None)
        if reset_cache or self._cache_position == 0 or cache_batch != bsz:
            for layer in self.layers:
                layer.attention.reset_cache(bsz, device=input_ids.device)
            self._cache_position = 0
            self._cache_batch_size = bsz
        position = torch.full(
            (bsz,), self._cache_position, device=input_ids.device, dtype=torch.long
        )
        token = self.token_embedding(input_ids)
        if self.position_embedding is None:
            x = token
        else:
            x = token + self.position_embedding(position).to(token.dtype)
        for layer in self.layers:
            x = layer.step(x, position_ids=position)
        self._cache_position += 1
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def decode_chunk(self, input_ids: Tensor, *, reset_cache: bool = False) -> Tensor:
        """Return logits for a token block using persistent bounded caches."""

        if input_ids.ndim != 2:
            raise ValueError("decode_chunk input_ids must have shape [batch, sequence]")
        bsz, length = input_ids.shape
        if length == 0:
            return input_ids.new_empty(
                (bsz, 0, self.lm_head.out_features), dtype=self.lm_head.weight.dtype
            )
        if self._cache_position + length > self.max_position_embeddings:
            raise ValueError("decode positions exceed max_position_embeddings")
        cache_batch = getattr(self, "_cache_batch_size", None)
        if reset_cache or self._cache_position == 0 or cache_batch != bsz:
            for layer in self.layers:
                layer.attention.reset_cache(bsz, device=input_ids.device)
            self._cache_position = 0
            self._cache_batch_size = bsz
        positions = torch.arange(
            self._cache_position,
            self._cache_position + length,
            device=input_ids.device,
        ).unsqueeze(0)
        token = self.token_embedding(input_ids)
        if self.position_embedding is None:
            x = token
        else:
            x = token + self.position_embedding(positions).to(token.dtype)
        for layer in self.layers:
            x = layer.step_chunk(x, position_ids=positions)
        self._cache_position += length
        return self.lm_head(self.norm(x))


def count_archive_elements(model: nn.Module) -> int:
    """Return persistent archive elements per batch across all attention layers."""

    return sum(
        layer.attention.archive.num_heads
        * layer.attention.archive.num_codes
        * layer.attention.archive.num_scales
        * (layer.attention.archive.head_dim + 1)
        + (
            layer.attention.archive.num_heads
            * layer.attention.archive.num_codes
            * (2 * layer.attention.archive.head_dim + 1)
            if layer.attention.archive.persistent_landmark
            else 0
        )
        + (
            layer.attention.archive.num_heads
            * layer.attention.archive.num_codes
            * layer.attention.archive.num_scales
            if layer.attention.archive.lazy_decay
            else 0
        )
        for layer in model.layers
        if isinstance(layer, QCCDecoderLayer)
    )
