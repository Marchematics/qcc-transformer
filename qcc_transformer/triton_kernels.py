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
        codes_count,
        scales,
        dim,
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
