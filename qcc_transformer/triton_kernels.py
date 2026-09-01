"""Optional Triton kernels for QCC inference.

The package remains importable without Triton or CUDA. The kernel is used only
for no-grad CUDA inference; training and CPU execution use the reference path.
"""

from __future__ import annotations

import torch

try:  # pragma: no cover - exercised only on a Triton-enabled GPU runner
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True

    @triton.jit
    def _qcc_update_kernel(
        numerator_ptr,
        denominator_ptr,
        key_ptr,
        value_ptr,
        codes_ptr,
        rates_ptr,
        aged_rates_ptr,
        batch_size,
        num_heads,
        stride_nb,
        stride_nh,
        stride_nm,
        stride_nj,
        stride_nd,
        stride_db,
        stride_dh,
        stride_dm,
        stride_dj,
        stride_kb,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vd,
        stride_ch,
        stride_cm,
        BLOCK_D: tl.constexpr,
        NUM_CODES: tl.constexpr,
        NUM_SCALES: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        pid = tl.program_id(0)
        codes_per_batch = num_heads * NUM_CODES
        batch = pid // codes_per_batch
        rem = pid % codes_per_batch
        head = rem // NUM_CODES
        code_id = rem % NUM_CODES
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < HEAD_DIM

        key = tl.load(
            key_ptr + batch * stride_kb + head * stride_kh + offs_d * stride_kd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        code = tl.load(
            codes_ptr + head * stride_ch + code_id * stride_cm + offs_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        score = tl.sum(key * code, axis=0) / tl.sqrt(tl.full((), HEAD_DIM, tl.float32))
        weight = tl.exp(tl.minimum(tl.maximum(score, -20.0), 10.0))

        for scale in range(NUM_SCALES):
            rate = tl.load(rates_ptr + scale).to(tl.float32)
            aged_rate = tl.load(aged_rates_ptr + scale).to(tl.float32)
            den_offset = (
                batch * stride_db
                + head * stride_dh
                + code_id * stride_dm
                + scale * stride_dj
            )
            old_den = tl.load(denominator_ptr + den_offset).to(tl.float32)
            tl.store(denominator_ptr + den_offset, old_den * rate + weight * aged_rate)
            num_offset = (
                batch * stride_nb
                + head * stride_nh
                + code_id * stride_nm
                + scale * stride_nj
                + offs_d * stride_nd
            )
            old_num = tl.load(numerator_ptr + num_offset, mask=mask_d, other=0.0).to(tl.float32)
            value = tl.load(
                value_ptr + batch * stride_vb + head * stride_vh + offs_d * stride_vd,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            tl.store(
                numerator_ptr + num_offset,
                old_num * rate + weight * aged_rate * value,
                mask=mask_d,
            )

    @triton.jit
    def _qcc_read_kernel(
        output_ptr,
        query_ptr,
        numerator_ptr,
        denominator_ptr,
        codes_ptr,
        mix_ptr,
        num_heads,
        stride_ob,
        stride_oh,
        stride_od,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_nb,
        stride_nh,
        stride_nm,
        stride_nj,
        stride_nd,
        stride_db,
        stride_dh,
        stride_dm,
        stride_dj,
        stride_ch,
        stride_cm,
        stride_cd,
        stride_mh,
        stride_mm,
        stride_mj,
        NUM_CODES: tl.constexpr,
        NUM_SCALES: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch = pid // num_heads
        head = pid % num_heads
        offs_d = tl.arange(0, BLOCK_D)
        offs_m = tl.arange(0, BLOCK_M)
        mask_d = offs_d < HEAD_DIM
        mask_m = offs_m < NUM_CODES
        query = tl.load(
            query_ptr + batch * stride_qb + head * stride_qh + offs_d * stride_qd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        logits = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        for code_id in range(NUM_CODES):
            code = tl.load(
                codes_ptr + head * stride_ch + code_id * stride_cm + offs_d * stride_cd,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            dot = tl.sum(query * code, axis=0) / tl.sqrt(tl.full((), HEAD_DIM, tl.float32))
            logits = tl.where(offs_m == code_id, dot, logits)
        max_logit = tl.max(tl.where(mask_m, logits, -float("inf")), axis=0)
        routing = tl.exp(logits - max_logit)
        routing = tl.where(mask_m, routing / tl.sum(routing, axis=0), 0.0)

        output = tl.zeros((BLOCK_D,), tl.float32)
        for code_id in range(NUM_CODES):
            response = tl.zeros((BLOCK_D,), tl.float32)
            for scale in range(NUM_SCALES):
                den_offset = (
                    batch * stride_db
                    + head * stride_dh
                    + code_id * stride_dm
                    + scale * stride_dj
                )
                den = tl.maximum(tl.load(denominator_ptr + den_offset).to(tl.float32), 1e-8)
                num_offset = (
                    batch * stride_nb
                    + head * stride_nh
                    + code_id * stride_nm
                    + scale * stride_nj
                    + offs_d * stride_nd
                )
                num = tl.load(numerator_ptr + num_offset, mask=mask_d, other=0.0).to(tl.float32)
                mix = tl.load(mix_ptr + head * stride_mh + code_id * stride_mm + scale * stride_mj).to(tl.float32)
                response += mix * num / den
            output += routing[code_id] * response
        tl.store(
            output_ptr + batch * stride_ob + head * stride_oh + offs_d * stride_od,
            output,
            mask=mask_d,
        )

    @triton.jit
    def _qcc_lazy_update_kernel(
        numerator_ptr,
        denominator_ptr,
        last_step_ptr,
        key_ptr,
        value_ptr,
        codes_ptr,
        indices_ptr,
        rates_ptr,
        aged_rates_ptr,
        current_step,
        num_heads,
        stride_nb,
        stride_nh,
        stride_nm,
        stride_nj,
        stride_nd,
        stride_db,
        stride_dh,
        stride_dm,
        stride_dj,
        stride_lb,
        stride_lh,
        stride_lm,
        stride_lj,
        stride_kb,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vd,
        stride_ch,
        stride_cm,
        stride_cd,
        stride_ib,
        stride_ih,
        stride_ir,
        BLOCK_D: tl.constexpr,
        NUM_SCALES: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ACTIVE_CODES: tl.constexpr,
    ):
        """Update selected landmark slots and lazily apply elapsed decay."""

        pid = tl.program_id(0)
        slots_per_batch = num_heads * ACTIVE_CODES
        batch = pid // slots_per_batch
        rem = pid % slots_per_batch
        head = rem // ACTIVE_CODES
        rank = rem % ACTIVE_CODES
        code_id = tl.load(
            indices_ptr + batch * stride_ib + head * stride_ih + rank * stride_ir
        ).to(tl.int32)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < HEAD_DIM
        key = tl.load(
            key_ptr + batch * stride_kb + head * stride_kh + offs_d * stride_kd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        code = tl.load(
            codes_ptr + head * stride_ch + code_id * stride_cm + offs_d * stride_cd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        score = tl.sum(key * code, axis=0) / tl.sqrt(tl.full((), HEAD_DIM, tl.float32))
        weight = tl.exp(tl.minimum(tl.maximum(score, -20.0), 10.0))
        value = tl.load(
            value_ptr + batch * stride_vb + head * stride_vh + offs_d * stride_vd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        for scale in range(NUM_SCALES):
            rate = tl.load(rates_ptr + scale).to(tl.float32)
            aged_rate = tl.load(aged_rates_ptr + scale).to(tl.float32)
            den_offset = (
                batch * stride_db + head * stride_dh + code_id * stride_dm + scale * stride_dj
            )
            last_offset = (
                batch * stride_lb + head * stride_lh + code_id * stride_lm + scale * stride_lj
            )
            old_last = tl.load(last_step_ptr + last_offset).to(tl.int64)
            delta = tl.maximum(current_step - old_last, 0).to(tl.float32)
            decay = tl.exp(delta * tl.log(rate))
            old_den = tl.load(denominator_ptr + den_offset).to(tl.float32)
            tl.store(denominator_ptr + den_offset, old_den * decay + weight * aged_rate)
            num_offset = (
                batch * stride_nb
                + head * stride_nh
                + code_id * stride_nm
                + scale * stride_nj
                + offs_d * stride_nd
            )
            old_num = tl.load(numerator_ptr + num_offset, mask=mask_d, other=0.0).to(tl.float32)
            tl.store(
                numerator_ptr + num_offset,
                old_num * decay + weight * aged_rate * value,
                mask=mask_d,
            )
            tl.store(last_step_ptr + last_offset, current_step)

    @triton.jit
    def _qcc_sparse_read_kernel(
        output_ptr,
        query_ptr,
        numerator_ptr,
        denominator_ptr,
        last_step_ptr,
        codes_ptr,
        mix_ptr,
        indices_ptr,
        route_ptr,
        rates_ptr,
        current_step,
        num_heads,
        stride_ob,
        stride_oh,
        stride_od,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_nb,
        stride_nh,
        stride_nm,
        stride_nj,
        stride_nd,
        stride_db,
        stride_dh,
        stride_dm,
        stride_dj,
        stride_lb,
        stride_lh,
        stride_lm,
        stride_lj,
        stride_ch,
        stride_cm,
        stride_cd,
        stride_mh,
        stride_mm,
        stride_mj,
        stride_ib,
        stride_ih,
        stride_ir,
        stride_rb,
        stride_rh,
        stride_rr,
        BLOCK_D: tl.constexpr,
        NUM_SCALES: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ACTIVE_CODES: tl.constexpr,
    ):
        """Read only selected slots, applying elapsed lazy decay on demand."""

        pid = tl.program_id(0)
        batch = pid // num_heads
        head = pid % num_heads
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < HEAD_DIM
        query = tl.load(
            query_ptr + batch * stride_qb + head * stride_qh + offs_d * stride_qd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        route_logits = tl.load(
            route_ptr + batch * stride_rb + head * stride_rh + tl.arange(0, ACTIVE_CODES) * stride_rr
        ).to(tl.float32)
        route_max = tl.max(route_logits, axis=0)
        route_exp = tl.exp(route_logits - route_max)
        route = route_exp / tl.sum(route_exp, axis=0)
        output = tl.zeros((BLOCK_D,), tl.float32)
        for rank in range(ACTIVE_CODES):
            code_id = tl.load(
                indices_ptr + batch * stride_ib + head * stride_ih + rank * stride_ir
            ).to(tl.int32)
            mix_max = tl.full((), -float("inf"), tl.float32)
            for scale in range(NUM_SCALES):
                mix_value = tl.load(
                    mix_ptr + head * stride_mh + code_id * stride_mm + scale * stride_mj
                ).to(tl.float32)
                mix_max = tl.maximum(mix_max, mix_value)
            mix_sum = tl.zeros((), tl.float32)
            response = tl.zeros((BLOCK_D,), tl.float32)
            for scale in range(NUM_SCALES):
                rate = tl.load(rates_ptr + scale).to(tl.float32)
                last_offset = (
                    batch * stride_lb + head * stride_lh + code_id * stride_lm + scale * stride_lj
                )
                old_last = tl.load(last_step_ptr + last_offset).to(tl.int64)
                delta = tl.maximum(current_step - old_last, 0).to(tl.float32)
                decay = tl.exp(delta * tl.log(rate))
                den_offset = (
                    batch * stride_db + head * stride_dh + code_id * stride_dm + scale * stride_dj
                )
                den = tl.maximum(tl.load(denominator_ptr + den_offset).to(tl.float32) * decay, 1e-8)
                num_offset = (
                    batch * stride_nb + head * stride_nh + code_id * stride_nm + scale * stride_nj
                    + offs_d * stride_nd
                )
                num = tl.load(numerator_ptr + num_offset, mask=mask_d, other=0.0).to(tl.float32) * decay
                mix_value = tl.load(
                    mix_ptr + head * stride_mh + code_id * stride_mm + scale * stride_mj
                ).to(tl.float32)
                mix_weight = tl.exp(mix_value - mix_max)
                mix_sum += mix_weight
                response += mix_weight * num / den
            output += route[rank] * response / mix_sum
        tl.store(
            output_ptr + batch * stride_ob + head * stride_oh + offs_d * stride_od,
            output,
            mask=mask_d,
        )
except ImportError:  # pragma: no cover - normal CPU installation
    TRITON_AVAILABLE = False


def triton_update_archive(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    codes: torch.Tensor,
    rates: torch.Tensor,
    window_size: int,
) -> None:
    """Run the fused update kernel in-place, or raise when unavailable."""

    if not TRITON_AVAILABLE or not key.is_cuda:
        raise RuntimeError("Triton CUDA runtime is unavailable")
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("key and value must have shape [batch, heads, head_dim]")
    original_numerator = numerator
    original_denominator = denominator
    numerator = numerator.contiguous()
    denominator = denominator.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    codes = codes.contiguous()
    rates = rates.to(device=key.device, dtype=torch.float32).contiguous()
    aged_rates = rates.pow(window_size).contiguous()
    bsz, heads, dim = key.shape
    _, codes_count, _ = codes.shape
    scales = rates.numel()
    block_dim = 1 << max(4, (dim - 1).bit_length())
    grid = (bsz * heads * codes_count,)
    _qcc_update_kernel[grid](
        numerator,
        denominator,
        key,
        value,
        codes,
        rates,
        aged_rates,
        bsz,
        heads,
        *numerator.stride(),
        *denominator.stride(),
        *key.stride(),
        *value.stride(),
        *codes.stride(),
        BLOCK_D=block_dim,
        NUM_CODES=codes_count,
        NUM_SCALES=scales,
        HEAD_DIM=dim,
    )
    if numerator.data_ptr() != original_numerator.data_ptr():
        original_numerator.copy_(numerator)
    if denominator.data_ptr() != original_denominator.data_ptr():
        original_denominator.copy_(denominator)


def triton_read_archive(
    query: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    codes: torch.Tensor,
    mix_logits: torch.Tensor,
) -> torch.Tensor:
    """Fuse routing, scale mixing, and response normalization for one read."""

    if not TRITON_AVAILABLE or not query.is_cuda:
        raise RuntimeError("Triton CUDA runtime is unavailable")
    query = query.contiguous()
    numerator = numerator.contiguous()
    denominator = denominator.contiguous()
    codes = codes.contiguous()
    mix = torch.softmax(mix_logits, dim=-1).to(torch.float32).contiguous()
    batch, heads, dim = query.shape
    _, codes_count, scales, _ = numerator.shape
    block_dim = 1 << max(4, (dim - 1).bit_length())
    block_m = 1 << max(0, (codes_count - 1).bit_length())
    output = torch.empty((batch, heads, dim), device=query.device, dtype=torch.float32)
    _qcc_read_kernel[(batch * heads,)](
        output,
        query,
        numerator,
        denominator,
        codes,
        mix,
        heads,
        *output.stride(),
        *query.stride(),
        *numerator.stride(),
        *denominator.stride(),
        *codes.stride(),
        *mix.stride(),
        NUM_CODES=codes_count,
        NUM_SCALES=scales,
        HEAD_DIM=dim,
        BLOCK_D=block_dim,
        BLOCK_M=block_m,
    )
    return output.to(query.dtype)


def triton_lazy_update_archive(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    last_step: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    codes: torch.Tensor,
    indices: torch.Tensor,
    rates: torch.Tensor,
    window_size: int,
    current_step: int,
) -> None:
    """Update only selected slots with timestamp-based lazy decay."""

    if not TRITON_AVAILABLE or not key.is_cuda:
        raise RuntimeError("Triton CUDA runtime is unavailable")
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("key and value must have shape [batch, heads, head_dim]")
    if indices.ndim != 3 or indices.shape[:2] != key.shape[:2]:
        raise ValueError("indices must have shape [batch, heads, active_codes]")
    active = indices.shape[-1]
    if active & (active - 1):
        raise RuntimeError("Triton sparse kernels require a power-of-two active_codes")
    numerator = numerator.contiguous()
    denominator = denominator.contiguous()
    last_step = last_step.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    codes = codes.contiguous()
    indices = indices.to(device=key.device, dtype=torch.int32).contiguous()
    rates = rates.to(device=key.device, dtype=torch.float32).contiguous()
    aged_rates = rates.pow(window_size).contiguous()
    batch, heads, dim = key.shape
    scales = rates.numel()
    block_dim = 1 << max(4, (dim - 1).bit_length())
    _qcc_lazy_update_kernel[(batch * heads * active,)](
        numerator,
        denominator,
        last_step,
        key,
        value,
        codes,
        indices,
        rates,
        aged_rates,
        current_step,
        heads,
        *numerator.stride(),
        *denominator.stride(),
        *last_step.stride(),
        *key.stride(),
        *value.stride(),
        *codes.stride(),
        *indices.stride(),
        BLOCK_D=block_dim,
        NUM_SCALES=scales,
        HEAD_DIM=dim,
        ACTIVE_CODES=active,
    )


def triton_sparse_read_archive(
    query: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    last_step: torch.Tensor,
    codes: torch.Tensor,
    mix_logits: torch.Tensor,
    indices: torch.Tensor,
    route_logits: torch.Tensor,
    rates: torch.Tensor,
    current_step: int,
) -> torch.Tensor:
    """Read selected slots with fused routing, decay, and scale mixing."""

    if not TRITON_AVAILABLE or not query.is_cuda:
        raise RuntimeError("Triton CUDA runtime is unavailable")
    query = query.contiguous()
    numerator = numerator.contiguous()
    denominator = denominator.contiguous()
    last_step = last_step.contiguous()
    codes = codes.contiguous()
    mix_logits = mix_logits.contiguous()
    indices = indices.to(device=query.device, dtype=torch.int32).contiguous()
    route_logits = route_logits.contiguous()
    rates = rates.to(device=query.device, dtype=torch.float32).contiguous()
    batch, heads, dim = query.shape
    active = indices.shape[-1]
    if active & (active - 1):
        raise RuntimeError("Triton sparse kernels require a power-of-two active_codes")
    scales = rates.numel()
    block_dim = 1 << max(4, (dim - 1).bit_length())
    output = torch.empty((batch, heads, dim), device=query.device, dtype=torch.float32)
    _qcc_sparse_read_kernel[(batch * heads,)](
        output,
        query,
        numerator,
        denominator,
        last_step,
        codes,
        mix_logits,
        indices,
        route_logits,
        rates,
        current_step,
        heads,
        *output.stride(),
        *query.stride(),
        *numerator.stride(),
        *denominator.stride(),
        *last_step.stride(),
        *codes.stride(),
        *mix_logits.stride(),
        *indices.stride(),
        *route_logits.stride(),
        BLOCK_D=block_dim,
        NUM_SCALES=scales,
        HEAD_DIM=dim,
        ACTIVE_CODES=active,
    )
    return output.to(query.dtype)
