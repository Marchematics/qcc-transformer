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
    def _qcc_local_decode_kernel(
        output_ptr,
        query_ptr,
        key_ptr,
        value_ptr,
        batch_size,
        num_heads,
        valid_length,
        stride_ob,
        stride_oh,
        stride_od,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kw,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vw,
        stride_vd,
        BLOCK_D: tl.constexpr,
        BLOCK_W: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """One-launch exact softmax over the bounded local decode window."""
        pid = tl.program_id(0)
        batch = pid // num_heads
        head = pid % num_heads
        offs_d = tl.arange(0, BLOCK_D)
        offs_w = tl.arange(0, BLOCK_W)
        mask_d = offs_d < HEAD_DIM
        mask_w = offs_w < valid_length
        q = tl.load(
            query_ptr + batch * stride_qb + head * stride_qh + offs_d * stride_qd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        logits = tl.zeros((BLOCK_W,), dtype=tl.float32)
        for index in range(BLOCK_W):
            key = tl.load(
                key_ptr + batch * stride_kb + head * stride_kh + index * stride_kw + offs_d * stride_kd,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            dot = tl.sum(q * key, axis=0) / tl.sqrt(tl.full((), HEAD_DIM, tl.float32))
            logits = tl.where(offs_w == index, dot, logits)
        logits = tl.where(mask_w, logits, -float("inf"))
        max_logit = tl.max(logits, axis=0)
        weights = tl.exp(logits - max_logit)
        weights = tl.where(mask_w, weights, 0.0)
        weights = weights / tl.sum(weights, axis=0)
        result = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for index in range(BLOCK_W):
            value = tl.load(
                value_ptr + batch * stride_vb + head * stride_vh + index * stride_vw + offs_d * stride_vd,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            weight = tl.sum(tl.where(offs_w == index, weights, 0.0), axis=0)
            result += weight * value
        tl.store(
            output_ptr + batch * stride_ob + head * stride_oh + offs_d * stride_od,
            result,
            mask=mask_d,
        )

    @triton.jit
    def _qcc_local_chunk_kernel(
        output_ptr,
        query_ptr,
        key_ptr,
        value_ptr,
        query_length,
        num_heads,
        old_length,
        window_size,
        stride_ob,
        stride_oh,
        stride_ot,
        stride_od,
        stride_qb,
        stride_qh,
        stride_qt,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kt,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vt,
        stride_vd,
        BLOCK_D: tl.constexpr,
        BLOCK_W: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """Chunked local attention kernel (one program per query)."""

        pid = tl.program_id(0)
        batch = pid // (num_heads * query_length)
        rem = pid % (num_heads * query_length)
        head = rem // query_length
        time = rem % query_length
        offs_d = tl.arange(0, BLOCK_D)
        offs_w = tl.arange(0, BLOCK_W)
        mask_d = offs_d < HEAD_DIM
        query = tl.load(
            query_ptr
            + batch * stride_qb
            + head * stride_qh
            + time * stride_qt
            + offs_d * stride_qd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        global_position = old_length + time
        window_start = tl.maximum(global_position - window_size + 1, 0)
        valid_length = tl.minimum(window_size, global_position + 1)
        mask_w = offs_w < valid_length
        logits = tl.zeros((BLOCK_W,), dtype=tl.float32)
        for index in range(BLOCK_W):
            key = tl.load(
                key_ptr
                + batch * stride_kb
                + head * stride_kh
                + (window_start + index) * stride_kt
                + offs_d * stride_kd,
                mask=mask_d & (index < valid_length),
                other=0.0,
            ).to(tl.float32)
            dot = tl.sum(query * key, axis=0) / tl.sqrt(
                tl.full((), HEAD_DIM, tl.float32)
            )
            logits = tl.where(offs_w == index, dot, logits)
        logits = tl.where(mask_w, logits, -float("inf"))
        max_logit = tl.max(logits, axis=0)
        weights = tl.exp(logits - max_logit)
        weights = tl.where(mask_w, weights, 0.0)
        weights = weights / tl.sum(weights, axis=0)
        result = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for index in range(BLOCK_W):
            value = tl.load(
                value_ptr
                + batch * stride_vb
                + head * stride_vh
                + (window_start + index) * stride_vt
                + offs_d * stride_vd,
                mask=mask_d & (index < valid_length),
                other=0.0,
            ).to(tl.float32)
            weight = tl.sum(tl.where(offs_w == index, weights, 0.0), axis=0)
            result += weight * value
        tl.store(
            output_ptr
            + batch * stride_ob
            + head * stride_oh
            + time * stride_ot
            + offs_d * stride_od,
            result,
            mask=mask_d,
        )

    @triton.jit
    def _qcc_update_kernel(
        numerator_ptr,
        denominator_ptr,
        key_ptr,
        value_ptr,
        codes_ptr,
        rates_ptr,
        aged_rates_ptr,
        content_threshold,
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
        stride_cd,
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
            codes_ptr + head * stride_ch + code_id * stride_cm + offs_d * stride_cd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        score = tl.sum(key * code, axis=0) / tl.sqrt(tl.full((), HEAD_DIM, tl.float32))
        weight = tl.exp(tl.minimum(tl.maximum(score, -20.0), 10.0))
        weight = tl.where(score >= content_threshold, weight, 0.0)

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
            # Triton does not support indexing a vector with the Python loop
            # variable in all compiler versions.  Select the scalar through a
            # masked reduction instead; ``code_id`` is compile-time here and
            # this lowers to the same single-lane value.
            route_weight = tl.sum(tl.where(offs_m == code_id, routing, 0.0), axis=0)
            output += route_weight * response
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
        content_threshold,
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
        selected = score >= content_threshold
        weight = tl.where(selected, weight, 0.0)
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
            tl.store(
                denominator_ptr + den_offset,
                tl.where(selected, old_den * decay + weight * aged_rate, old_den),
            )
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
                tl.where(
                    selected,
                    old_num * decay + weight * aged_rate * value,
                    old_num,
                ),
                mask=mask_d,
            )
            tl.store(last_step_ptr + last_offset, tl.where(selected, current_step, old_last))

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

    @triton.jit
    def _qcc_update_read_partial_kernel(
        partial_ptr,
        key_ptr,
        value_ptr,
        codes_ptr,
        rates_ptr,
        aged_rates_ptr,
        mix_ptr,
        numerator_ptr,
        denominator_ptr,
        content_threshold,
        num_events,
        num_heads,
        stride_pb,
        stride_ph,
        stride_pe,
        stride_pc,
        stride_pd,
        stride_kb,
        stride_kh,
        stride_ke,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_ve,
        stride_vd,
        stride_ch,
        stride_cm,
        stride_cd,
        stride_rh,
        stride_re,
        stride_rj,
        stride_nb,
        stride_nh,
        stride_nm,
        stride_nj,
        stride_nd,
        stride_db,
        stride_dh,
        stride_dm,
        stride_dj,
        BLOCK_D: tl.constexpr,
        BLOCK_S: tl.constexpr,
        NUM_CODES: tl.constexpr,
        NUM_SCALES: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """Update every scale for one code and emit its mixed response.

        One program owns a ``(batch, head, code)`` state slot and walks the
        events in order.  This preserves the recurrent update/read semantics,
        while replacing one Python/Triton launch per event with one launch for
        the complete decode block.  A second kernel performs the query routing
        reduction across codes.
        """

        pid = tl.program_id(0)
        codes_per_batch = num_heads * NUM_CODES
        batch = pid // codes_per_batch
        rem = pid % codes_per_batch
        head = rem // NUM_CODES
        code_id = rem % NUM_CODES
        offs_d = tl.arange(0, BLOCK_D)
        offs_s = tl.arange(0, BLOCK_S)
        mask_d = offs_d < HEAD_DIM
        mask_s = offs_s < NUM_SCALES
        code = tl.load(
            codes_ptr + head * stride_ch + code_id * stride_cm + offs_d * stride_cd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        rates = tl.load(rates_ptr + offs_s, mask=mask_s, other=1.0).to(tl.float32)
        aged_rates = tl.load(
            aged_rates_ptr + offs_s, mask=mask_s, other=1.0
        ).to(tl.float32)
        mix = tl.load(
            mix_ptr + head * stride_rh + code_id * stride_re + offs_s * stride_rj,
            mask=mask_s,
            other=0.0,
        ).to(tl.float32)
        state_num_offsets = (
            batch * stride_nb
            + head * stride_nh
            + code_id * stride_nm
            + offs_s[:, None] * stride_nj
            + offs_d[None, :] * stride_nd
        )
        state_num_mask = mask_s[:, None] & mask_d[None, :]
        state_num = tl.load(
            numerator_ptr + state_num_offsets, mask=state_num_mask, other=0.0
        ).to(tl.float32)
        state_den_offsets = (
            batch * stride_db
            + head * stride_dh
            + code_id * stride_dm
            + offs_s * stride_dj
        )
        state_den = tl.load(
            denominator_ptr + state_den_offsets, mask=mask_s, other=0.0
        ).to(tl.float32)

        for event in tl.range(0, num_events):
            key = tl.load(
                key_ptr
                + batch * stride_kb
                + head * stride_kh
                + event * stride_ke
                + offs_d * stride_kd,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            value = tl.load(
                value_ptr
                + batch * stride_vb
                + head * stride_vh
                + event * stride_ve
                + offs_d * stride_vd,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(key * code, axis=0) / tl.sqrt(
                tl.full((), HEAD_DIM, tl.float32)
            )
            weight = tl.exp(tl.minimum(tl.maximum(score, -20.0), 10.0))
            weight = tl.where(score >= content_threshold, weight, 0.0)
            addition = weight * aged_rates
            state_den = state_den * rates + addition
            state_num = state_num * rates[:, None] + addition[:, None] * value[None, :]
            response = tl.sum(
                mix[:, None]
                * state_num
                / tl.maximum(state_den[:, None], 1e-8),
                axis=0,
            )
            partial_offsets = (
                batch * stride_pb
                + head * stride_ph
                + event * stride_pe
                + code_id * stride_pc
                + offs_d * stride_pd
            )
            tl.store(partial_ptr + partial_offsets, response, mask=mask_d)

        tl.store(
            numerator_ptr + state_num_offsets,
            state_num,
            mask=state_num_mask,
        )
        tl.store(
            denominator_ptr + state_den_offsets,
            state_den,
            mask=mask_s,
        )

    @triton.jit
    def _qcc_route_partial_kernel(
        output_ptr,
        query_ptr,
        partial_ptr,
        codes_ptr,
        num_heads,
        stride_ob,
        stride_oh,
        stride_oe,
        stride_od,
        stride_qb,
        stride_qh,
        stride_qe,
        stride_qd,
        stride_pb,
        stride_ph,
        stride_pe,
        stride_pc,
        stride_pd,
        stride_ch,
        stride_cm,
        stride_cd,
        num_events,
        BLOCK_D: tl.constexpr,
        BLOCK_C: tl.constexpr,
        NUM_CODES: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """Route per-code responses for a query block in one GPU launch."""

        pid = tl.program_id(0)
        events_per_batch = num_heads * num_events
        batch = pid // events_per_batch
        rem = pid % events_per_batch
        head = rem // num_events
        event = rem % num_events
        offs_d = tl.arange(0, BLOCK_D)
        offs_c = tl.arange(0, BLOCK_C)
        mask_d = offs_d < HEAD_DIM
        mask_c = offs_c < NUM_CODES
        query = tl.load(
            query_ptr
            + batch * stride_qb
            + head * stride_qh
            + event * stride_qe
            + offs_d * stride_qd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        code_offsets = (
            head * stride_ch
            + offs_c[:, None] * stride_cm
            + offs_d[None, :] * stride_cd
        )
        code_mask = mask_c[:, None] & mask_d[None, :]
        code_block = tl.load(codes_ptr + code_offsets, mask=code_mask, other=0.0).to(
            tl.float32
        )
        logits = tl.sum(code_block * query[None, :], axis=1) / tl.sqrt(
            tl.full((), HEAD_DIM, tl.float32)
        )
        logits = tl.where(mask_c, logits, -float("inf"))
        max_logit = tl.max(logits, axis=0)
        routing = tl.exp(logits - max_logit)
        routing = tl.where(mask_c, routing / tl.sum(routing, axis=0), 0.0)
        partial_offsets = (
            batch * stride_pb
            + head * stride_ph
            + event * stride_pe
            + offs_c[:, None] * stride_pc
            + offs_d[None, :] * stride_pd
        )
        partial = tl.load(partial_ptr + partial_offsets, mask=code_mask, other=0.0).to(
            tl.float32
        )
        output = tl.sum(routing[:, None] * partial, axis=0)
        output_offsets = (
            batch * stride_ob
            + head * stride_oh
            + event * stride_oe
            + offs_d * stride_od
        )
        tl.store(output_ptr + output_offsets, output, mask=mask_d)

    @triton.jit
    def _qcc_sparse_update_read_chunk_kernel(
        partial_ptr,
        key_index_ptr,
        key_score_ptr,
        query_index_ptr,
        key_ptr,
        value_ptr,
        codes_ptr,
        rates_ptr,
        aged_rates_ptr,
        mix_ptr,
        numerator_ptr,
        denominator_ptr,
        last_step_ptr,
        base_step,
        content_threshold,
        num_events,
        num_heads,
        stride_pb,
        stride_ph,
        stride_pe,
        stride_pr,
        stride_pd,
        stride_ib,
        stride_ih,
        stride_ie,
        stride_ir,
        stride_sb,
        stride_sh,
        stride_se,
        stride_sr,
        stride_qib,
        stride_qih,
        stride_qie,
        stride_qir,
        stride_kb,
        stride_kh,
        stride_ke,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_ve,
        stride_vd,
        stride_ch,
        stride_cm,
        stride_cd,
        stride_mh,
        stride_mm,
        stride_mj,
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
        BLOCK_D: tl.constexpr,
        BLOCK_S: tl.constexpr,
        ACTIVE_CODES: tl.constexpr,
        NUM_CODES: tl.constexpr,
        NUM_SCALES: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """Process a sparse lazy block for one ``(batch, head, code)``.

        The top-k indices are computed for the whole block by PyTorch before
        this kernel.  A program owns one code slot, walks events in order, and
        only materializes its state when that code is selected.  Query-selected
        responses are emitted into a small ``[events, active_codes, dim]``
        tensor for the routing reduction.  This avoids one update/read launch
        per event without introducing races between repeated code selections.
        """

        pid = tl.program_id(0)
        slots_per_batch = num_heads * NUM_CODES
        batch = pid // slots_per_batch
        rem = pid % slots_per_batch
        head = rem // NUM_CODES
        code_id = rem % NUM_CODES
        offs_d = tl.arange(0, BLOCK_D)
        offs_s = tl.arange(0, BLOCK_S)
        mask_d = offs_d < HEAD_DIM
        mask_s = offs_s < NUM_SCALES
        rates = tl.load(rates_ptr + offs_s, mask=mask_s, other=1.0).to(tl.float32)
        aged_rates = tl.load(
            aged_rates_ptr + offs_s, mask=mask_s, other=1.0
        ).to(tl.float32)
        mix = tl.load(
            mix_ptr + head * stride_mh + code_id * stride_mm + offs_s * stride_mj,
            mask=mask_s,
            other=0.0,
        ).to(tl.float32)
        state_num_offsets = (
            batch * stride_nb
            + head * stride_nh
            + code_id * stride_nm
            + offs_s[:, None] * stride_nj
            + offs_d[None, :] * stride_nd
        )
        state_num_mask = mask_s[:, None] & mask_d[None, :]
        state_num = tl.load(
            numerator_ptr + state_num_offsets, mask=state_num_mask, other=0.0
        ).to(tl.float32)
        state_den_offsets = (
            batch * stride_db
            + head * stride_dh
            + code_id * stride_dm
            + offs_s * stride_dj
        )
        state_den = tl.load(
            denominator_ptr + state_den_offsets, mask=mask_s, other=0.0
        ).to(tl.float32)
        last_offsets = (
            batch * stride_lb
            + head * stride_lh
            + code_id * stride_lm
            + offs_s * stride_lj
        )
        last = tl.load(last_step_ptr + last_offsets, mask=mask_s, other=0).to(tl.int64)

        for event in tl.range(0, num_events):
            event_step = base_step + event + 1
            # Apply selected key updates in rank order.  Top-k indices are
            # unique within an event, so at most one rank matches this code.
            for rank in range(ACTIVE_CODES):
                selected_code = tl.load(
                    key_index_ptr
                    + batch * stride_ib
                    + head * stride_ih
                    + event * stride_ie
                    + rank * stride_ir
                ).to(tl.int32)
                selected = selected_code == code_id
                score = tl.load(
                    key_score_ptr
                    + batch * stride_sb
                    + head * stride_sh
                    + event * stride_se
                    + rank * stride_sr
                ).to(tl.float32)
                weight = tl.exp(tl.minimum(tl.maximum(score, -20.0), 10.0))
                selected = selected & (score >= content_threshold)
                weight = tl.where(selected, weight, 0.0)
                key_step = tl.where(selected, event_step, last)
                delta = tl.maximum(key_step - last, 0).to(tl.float32)
                decay = tl.exp(delta * tl.log(rates))
                key_value = tl.load(
                    value_ptr
                    + batch * stride_vb
                    + head * stride_vh
                    + event * stride_ve
                    + offs_d * stride_vd,
                    mask=mask_d,
                    other=0.0,
                ).to(tl.float32)
                candidate_den = state_den * decay + selected.to(tl.float32) * weight * aged_rates
                candidate_num = state_num * decay[:, None] + selected.to(tl.float32) * (
                    weight * aged_rates
                )[:, None] * key_value[None, :]
                state_den = tl.where(selected, candidate_den, state_den)
                state_num = tl.where(selected, candidate_num, state_num)
                last = tl.where(selected, event_step, last)

            # Emit a response only for query-selected ranks.  Lazy decay is
            # applied relative to the current event without mutating the
            # timestamp, so a later key update still sees the correct age.
            for rank in range(ACTIVE_CODES):
                selected_code = tl.load(
                    query_index_ptr
                    + batch * stride_qib
                    + head * stride_qih
                    + event * stride_qie
                    + rank * stride_qir
                ).to(tl.int32)
                selected = selected_code == code_id
                delta = tl.maximum(event_step - last, 0).to(tl.float32)
                decay = tl.exp(delta * tl.log(rates))
                response = tl.sum(
                    mix[:, None]
                    * (state_num * decay[:, None])
                    / tl.maximum((state_den * decay)[:, None], 1e-8),
                    axis=0,
                )
                partial_offsets = (
                    batch * stride_pb
                    + head * stride_ph
                    + event * stride_pe
                    + rank * stride_pr
                    + offs_d * stride_pd
                )
                tl.store(
                    partial_ptr + partial_offsets,
                    response,
                    mask=mask_d & selected,
                )

        tl.store(numerator_ptr + state_num_offsets, state_num, mask=state_num_mask)
        tl.store(denominator_ptr + state_den_offsets, state_den, mask=mask_s)
        tl.store(last_step_ptr + last_offsets, last, mask=mask_s)

    @triton.jit
    def _qcc_route_sparse_partial_kernel(
        output_ptr,
        partial_ptr,
        route_ptr,
        batch_size,
        num_heads,
        stride_ob,
        stride_oh,
        stride_oe,
        stride_od,
        stride_pb,
        stride_ph,
        stride_pe,
        stride_pr,
        stride_pd,
        stride_rb,
        stride_rh,
        stride_re,
        stride_rr,
        num_events,
        BLOCK_D: tl.constexpr,
        ACTIVE_CODES: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """Fuse softmax routing over sparse query-selected responses."""

        pid = tl.program_id(0)
        events_per_batch = num_heads * num_events
        batch = pid // events_per_batch
        rem = pid % events_per_batch
        head = rem // num_events
        event = rem % num_events
        offs_d = tl.arange(0, BLOCK_D)
        offs_r = tl.arange(0, ACTIVE_CODES)
        mask_d = offs_d < HEAD_DIM
        route = tl.load(
            route_ptr
            + batch * stride_rb
            + head * stride_rh
            + event * stride_re
            + offs_r * stride_rr
        ).to(tl.float32)
        route_max = tl.max(route, axis=0)
        route = tl.exp(route - route_max)
        route = route / tl.sum(route, axis=0)
        partial_offsets = (
            batch * stride_pb
            + head * stride_ph
            + event * stride_pe
            + offs_r[:, None] * stride_pr
            + offs_d[None, :] * stride_pd
        )
        partial = tl.load(
            partial_ptr + partial_offsets,
            mask=mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        output = tl.sum(route[:, None] * partial, axis=0)
        output_offsets = (
            batch * stride_ob
            + head * stride_oh
            + event * stride_oe
            + offs_d * stride_od
        )
        tl.store(output_ptr + output_offsets, output, mask=mask_d)
except ImportError:  # pragma: no cover - normal CPU installation
    TRITON_AVAILABLE = False


def triton_local_decode_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    valid_length: int,
) -> torch.Tensor:
    """Compute one exact local-window attention query in a single Triton launch."""
    if not TRITON_AVAILABLE or not query.is_cuda:
        raise RuntimeError("Triton CUDA runtime is unavailable")
    if query.ndim != 3 or keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("query must be [B,H,D], keys/values [B,H,W,D]")
    batch, heads, dim = query.shape
    if keys.shape[0] != batch or keys.shape[1] != heads or keys.shape[3] != dim:
        raise ValueError("local attention shapes do not match")
    if not 0 < valid_length <= keys.shape[2]:
        raise ValueError("valid_length must be within the key window")
    query = query.contiguous()
    keys = keys.contiguous()
    values = values.contiguous()
    output = torch.empty_like(query)
    block_dim = 1 << max(4, (dim - 1).bit_length())
    block_window = 1 << max(0, (keys.shape[2] - 1).bit_length())
    _qcc_local_decode_kernel[(batch * heads,)](
        output, query, keys, values, batch, heads, valid_length,
        *output.stride(), *query.stride(), *keys.stride(), *values.stride(),
        BLOCK_D=block_dim, BLOCK_W=block_window, HEAD_DIM=dim,
    )
    return output


def triton_local_chunk_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    old_length: int,
    window_size: int,
) -> torch.Tensor:
    """Compute exact sliding-window attention for a whole decode block.

    ``keys``/``values`` must contain the chronological ring prefix followed
    by the new block, so their length is ``old_length + query_length``.  The
    Triton kernel emits one output per query and never materializes an
    unfolded window tensor.
    """

    if not TRITON_AVAILABLE or not query.is_cuda:
        raise RuntimeError("Triton CUDA runtime is unavailable")
    if query.ndim != 4 or keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("query, keys, and values must be rank-4 tensors")
    batch, heads, query_length, dim = query.shape
    if keys.shape[:2] != (batch, heads) or keys.shape[2] != old_length + query_length:
        raise ValueError("keys must contain old_length plus query_length tokens")
    if keys.shape[3] != dim or old_length < 0 or window_size <= 0:
        raise ValueError("invalid local attention dimensions")
    query = query.contiguous()
    keys = keys.contiguous()
    values = values.contiguous()
    output = torch.empty_like(query)
    block_dim = 1 << max(4, (dim - 1).bit_length())
    block_window = 1 << max(0, (min(window_size, old_length + query_length) - 1).bit_length())
    _qcc_local_chunk_kernel[(batch * heads * query_length,)](
        output,
        query,
        keys,
        values,
        query_length,
        heads,
        old_length,
        window_size,
        *output.stride(),
        *query.stride(),
        *keys.stride(),
        *values.stride(),
        BLOCK_D=block_dim,
        BLOCK_W=block_window,
        HEAD_DIM=dim,
    )
    return output


def triton_update_archive(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    codes: torch.Tensor,
    rates: torch.Tensor,
    window_size: int,
    content_threshold: float | None = None,
) -> None:
    """Run the fused update kernel in-place, or raise when unavailable."""

    if not TRITON_AVAILABLE or not key.is_cuda:
        raise RuntimeError("Triton CUDA runtime is unavailable")
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("key and value must have shape [batch, heads, head_dim]")
    if codes.ndim != 3 or codes.shape[0] != key.shape[1] or codes.shape[2] != key.shape[2]:
        raise ValueError("codes must have shape [heads, num_codes, head_dim]")
    if numerator.ndim != 5 or denominator.ndim != 4:
        raise ValueError("archive state has an invalid rank")
    if numerator.shape[:3] != denominator.shape[:3] or numerator.shape[3] != denominator.shape[3]:
        raise ValueError("numerator and denominator shapes do not match")
    original_numerator = numerator
    original_denominator = denominator
    numerator = numerator.contiguous()
    denominator = denominator.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    codes = codes.to(device=key.device, dtype=torch.float32).contiguous()
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
        float("-inf") if content_threshold is None else float(content_threshold),
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
    codes = codes.to(device=query.device, dtype=torch.float32).contiguous()
    mix = torch.softmax(
        mix_logits.to(device=query.device), dim=-1
    ).to(torch.float32).contiguous()
    batch, heads, dim = query.shape
    _, _, codes_count, scales, _ = numerator.shape
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
    content_threshold: float | None = None,
) -> None:
    """Update only selected slots with timestamp-based lazy decay."""

    if not TRITON_AVAILABLE or not key.is_cuda:
        raise RuntimeError("Triton CUDA runtime is unavailable")
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("key and value must have shape [batch, heads, head_dim]")
    if indices.ndim != 3 or indices.shape[:2] != key.shape[:2]:
        raise ValueError("indices must have shape [batch, heads, active_codes]")
    if codes.ndim != 3 or codes.shape[0] != key.shape[1] or codes.shape[2] != key.shape[2]:
        raise ValueError("codes must have shape [heads, num_codes, head_dim]")
    active = indices.shape[-1]
    if active & (active - 1):
        raise RuntimeError("Triton sparse kernels require a power-of-two active_codes")
    original_numerator = numerator
    original_denominator = denominator
    original_last_step = last_step
    numerator = numerator.contiguous()
    denominator = denominator.contiguous()
    last_step = last_step.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    codes = codes.to(device=key.device, dtype=torch.float32).contiguous()
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
        float("-inf") if content_threshold is None else float(content_threshold),
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
    # The archive normally owns contiguous buffers, but preserving in-place
    # semantics here avoids silently dropping updates when callers pass a
    # strided view (``Tensor.contiguous()`` may allocate a copy).
    if numerator.data_ptr() != original_numerator.data_ptr():
        original_numerator.copy_(numerator)
    if denominator.data_ptr() != original_denominator.data_ptr():
        original_denominator.copy_(denominator)
    if last_step.data_ptr() != original_last_step.data_ptr():
        original_last_step.copy_(last_step)


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
    codes = codes.to(device=query.device, dtype=torch.float32).contiguous()
    mix_logits = mix_logits.to(device=query.device, dtype=torch.float32).contiguous()
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


def triton_update_read_archive_chunk(
    key: torch.Tensor,
    value: torch.Tensor,
    query: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    codes: torch.Tensor,
    mix_logits: torch.Tensor,
    rates: torch.Tensor,
    window_size: int,
    *,
    block_size: int = 256,
    output: torch.Tensor | None = None,
    content_threshold: float | None = None,
) -> torch.Tensor:
    """Fuse dense archive update/read for a block of evicted tokens.

    The recurrent state is still updated in event order, but each bounded
    block uses two Triton launches (state/partial update and code routing)
    instead of one update plus one read launch per token.  ``block_size``
    bounds the temporary per-code response tensor, which is important for
    million-token prefill where the public API may pass a very large block.
    """

    if not TRITON_AVAILABLE or not key.is_cuda:
        raise RuntimeError("Triton CUDA runtime is unavailable")
    if key.ndim != 4 or value.shape != key.shape or query.shape != key.shape:
        raise ValueError("key, value, and query must have shape [batch, heads, events, head_dim]")
    if numerator.ndim != 5 or denominator.ndim != 4:
        raise ValueError("archive state has an invalid rank")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    batch, heads, events, dim = key.shape
    if events == 0:
        return query.new_empty(query.shape)
    if output is not None:
        if output.shape != query.shape or output.device != key.device or output.dtype != query.dtype:
            raise ValueError("output must match query shape, dtype, and device")
    if numerator.shape[0] != batch or numerator.shape[1] != heads:
        raise ValueError("archive state batch/head shape does not match inputs")
    num_codes = numerator.shape[2]
    num_scales = numerator.shape[3]
    if codes.shape != (heads, num_codes, dim):
        raise ValueError("codes shape does not match archive state")
    if rates.numel() != num_scales or mix_logits.shape != (heads, num_codes, num_scales):
        raise ValueError("rates or mix_logits shape does not match archive state")

    # Triton kernels use contiguous pointers for predictable coalescing.  The
    # archive owns contiguous state, but copy back defensively for callers
    # passing a strided state view.
    original_numerator = numerator
    original_denominator = denominator
    numerator = numerator.contiguous()
    denominator = denominator.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    query = query.contiguous()
    codes = codes.to(device=key.device, dtype=torch.float32).contiguous()
    rates = rates.to(device=key.device, dtype=torch.float32).contiguous()
    mix = torch.softmax(
        mix_logits.to(device=key.device), dim=-1
    ).to(dtype=torch.float32).contiguous()
    aged_rates = rates.pow(window_size).contiguous()
    block_dim = 1 << max(4, (dim - 1).bit_length())
    block_scales = 1 << max(0, (num_scales - 1).bit_length())
    block_codes = 1 << max(0, (num_codes - 1).bit_length())
    # Keep one result tensor and one bounded partial buffer for the whole
    # call.  Allocating a partial/output pair for every block is especially
    # expensive for million-token prefill, where it turns a bounded kernel
    # into thousands of allocator operations followed by a full ``cat``.
    if output is None:
        output = torch.empty((batch, heads, events, dim), device=key.device, dtype=query.dtype)
    scratch_size = min(block_size, events)
    partial_scratch = torch.empty(
        (batch, heads, scratch_size, num_codes, dim),
        device=key.device,
        dtype=torch.float32,
    )

    for start in range(0, events, block_size):
        count = min(block_size, events - start)
        key_block = key[:, :, start : start + count]
        value_block = value[:, :, start : start + count]
        query_block = query[:, :, start : start + count]
        partial = partial_scratch[:, :, :count]
        _qcc_update_read_partial_kernel[(batch * heads * num_codes,)](
            partial,
            key_block,
            value_block,
            codes,
            rates,
            aged_rates,
            mix,
            numerator,
            denominator,
            float("-inf") if content_threshold is None else float(content_threshold),
            count,
            heads,
            *partial.stride(),
            *key_block.stride(),
            *value_block.stride(),
            *codes.stride(),
            *mix.stride(),
            *numerator.stride(),
            *denominator.stride(),
            BLOCK_D=block_dim,
            BLOCK_S=block_scales,
            NUM_CODES=num_codes,
            NUM_SCALES=num_scales,
            HEAD_DIM=dim,
        )
        output_block = output[:, :, start : start + count]
        _qcc_route_partial_kernel[(batch * heads * count,)](
            output_block,
            query_block,
            partial,
            codes,
            heads,
            *output.stride(),
            *query_block.stride(),
            *partial.stride(),
            *codes.stride(),
            count,
            BLOCK_D=block_dim,
            BLOCK_C=block_codes,
            NUM_CODES=num_codes,
            HEAD_DIM=dim,
        )

    if numerator.data_ptr() != original_numerator.data_ptr():
        original_numerator.copy_(numerator)
    if denominator.data_ptr() != original_denominator.data_ptr():
        original_denominator.copy_(denominator)
    return output


def triton_sparse_update_read_archive_chunk(
    key: torch.Tensor,
    value: torch.Tensor,
    query: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    last_step: torch.Tensor,
    codes: torch.Tensor,
    mix_logits: torch.Tensor,
    rates: torch.Tensor,
    window_size: int,
    current_step: int,
    active_codes: int,
    *,
    block_size: int = 256,
    output: torch.Tensor | None = None,
    content_threshold: float | None = None,
) -> torch.Tensor:
    """Fuse sparse lazy archive update/read for a block of evicted tokens.

    Top-k routing is computed once over the complete block.  A Triton program
    owns one code slot and walks the event timeline, applying timestamp decay
    only when that slot is selected; a second launch routes the emitted
    query-selected responses.  This preserves lazy update-before-read ordering
    while removing the per-event Python loop used by the reference path.
    """

    if not TRITON_AVAILABLE or not key.is_cuda:
        raise RuntimeError("Triton CUDA runtime is unavailable")
    if key.ndim != 4 or value.shape != key.shape or query.shape != key.shape:
        raise ValueError("key, value, and query must have shape [batch, heads, events, head_dim]")
    if numerator.ndim != 5 or denominator.ndim != 4 or last_step.ndim != 4:
        raise ValueError("archive state has an invalid rank")
    if block_size <= 0 or active_codes <= 0:
        raise ValueError("block_size and active_codes must be positive")
    if active_codes & (active_codes - 1):
        raise RuntimeError("Triton sparse kernels require a power-of-two active_codes")
    batch, heads, events, dim = key.shape
    if events == 0:
        return query.new_empty(query.shape)
    if output is not None:
        if output.shape != query.shape or output.device != key.device or output.dtype != query.dtype:
            raise ValueError("output must match query shape, dtype, and device")
    num_codes = numerator.shape[2]
    num_scales = numerator.shape[3]
    if active_codes > num_codes:
        raise ValueError("active_codes cannot exceed num_codes")
    if codes.shape != (heads, num_codes, dim):
        raise ValueError("codes shape does not match archive state")
    if rates.numel() != num_scales or mix_logits.shape != (heads, num_codes, num_scales):
        raise ValueError("rates or mix_logits shape does not match archive state")
    if denominator.shape != last_step.shape or denominator.shape != numerator.shape[:4]:
        raise ValueError("archive state shapes do not match")

    original_numerator = numerator
    original_denominator = denominator
    original_last_step = last_step
    numerator = numerator.contiguous()
    denominator = denominator.contiguous()
    last_step = last_step.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    query = query.contiguous()
    codes = codes.contiguous()
    codes_fp32 = codes.to(dtype=torch.float32)
    rates = rates.to(device=key.device, dtype=torch.float32).contiguous()
    mix = torch.softmax(
        mix_logits.to(device=key.device), dim=-1
    ).to(dtype=torch.float32).contiguous()
    aged_rates = rates.pow(window_size).contiguous()
    block_dim = 1 << max(4, (dim - 1).bit_length())
    block_scales = 1 << max(0, (num_scales - 1).bit_length())
    # Reuse bounded temporaries across blocks.  Sparse routing used to create
    # three large tensors per block (partial, output, and a transient list
    # entry); reusing storage keeps allocator traffic independent of context.
    if output is None:
        output = torch.empty((batch, heads, events, dim), device=key.device, dtype=query.dtype)
    scratch_size = min(block_size, events)
    partial_scratch = torch.empty(
        (batch, heads, scratch_size, active_codes, dim),
        device=key.device,
        dtype=torch.float32,
    )
    scale = dim**-0.5

    for start in range(0, events, block_size):
        count = min(block_size, events - start)
        key_block = key[:, :, start : start + count]
        value_block = value[:, :, start : start + count]
        query_block = query[:, :, start : start + count]
        key_logits = torch.einsum(
            "bhed,hmd->bhem", key_block.to(torch.float32), codes_fp32
        ) * scale
        key_scores, key_indices = torch.topk(key_logits, active_codes, dim=-1)
        query_logits = torch.einsum(
            "bhed,hmd->bhem", query_block.to(torch.float32), codes_fp32
        ) * scale
        query_scores, query_indices = torch.topk(query_logits, active_codes, dim=-1)
        key_indices = key_indices.to(torch.int32).contiguous()
        query_indices = query_indices.to(torch.int32).contiguous()
        key_scores = key_scores.contiguous()
        query_scores = query_scores.contiguous()
        partial = partial_scratch[:, :, :count]
        partial.zero_()
        _qcc_sparse_update_read_chunk_kernel[(batch * heads * num_codes,)](
            partial,
            key_indices,
            key_scores,
            query_indices,
            key_block,
            value_block,
            codes,
            rates,
            aged_rates,
            mix,
            numerator,
            denominator,
            last_step,
            current_step + start,
            float("-inf") if content_threshold is None else float(content_threshold),
            count,
            heads,
            *partial.stride(),
            *key_indices.stride(),
            *key_scores.stride(),
            *query_indices.stride(),
            *key_block.stride(),
            *value_block.stride(),
            *codes.stride(),
            *mix.stride(),
            *numerator.stride(),
            *denominator.stride(),
            *last_step.stride(),
            BLOCK_D=block_dim,
            BLOCK_S=block_scales,
            ACTIVE_CODES=active_codes,
            NUM_CODES=num_codes,
            NUM_SCALES=num_scales,
            HEAD_DIM=dim,
        )
        output_block = output[:, :, start : start + count]
        _qcc_route_sparse_partial_kernel[(batch * heads * count,)](
            output_block,
            partial,
            query_scores,
            batch,
            heads,
            *output.stride(),
            *partial.stride(),
            *query_scores.stride(),
            count,
            BLOCK_D=block_dim,
            ACTIVE_CODES=active_codes,
            HEAD_DIM=dim,
        )

    if numerator.data_ptr() != original_numerator.data_ptr():
        original_numerator.copy_(numerator)
    if denominator.data_ptr() != original_denominator.data_ptr():
        original_denominator.copy_(denominator)
    if last_step.data_ptr() != original_last_step.data_ptr():
        original_last_step.copy_(last_step)
    return output
