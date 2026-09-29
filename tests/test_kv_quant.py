"""CPU tests for decode-time KV quantization (``benchmarks/kv_quant.py``).

Everything here runs on a tiny randomly-initialised Llama on CPU -- no GPU, no
download, no checkpoint.  The tests pin down the parts a reviewer will check:

* the round-trip bound ``max|error| <= scale`` for int8 and int4 on realistic
  cache shapes, for both the per-channel key rule and the per-token value rule;
* the documented int4 bit layout, transcribed literally from the docstring;
* group-wise scales actually isolate an outlier (the point of grouping);
* the documented group-size fallback (``group_size >= axis_length``) and odd
  head dimensions;
* the exactness cases (all-zero, constant value) and the hand-computed
  ``state_bytes`` arithmetic;
* quantize -> dequantize -> decode against a *real* cache produced by
  ``qcc_transformer.retention.compile_bounded_cache``: the attention output obeys
  the analytic convexity bound, and greedy decode from the quantized cache
  reproduces the unquantized decode.

The module-level fixture pins the intra-op thread count to one: the tensors here
are small enough that the default thread pool costs more than it saves on a
loaded CPU box, and the module should not care how busy the machine is.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import kv_quant as Q  # noqa: E402  (path is set up above)
from qcc_transformer.retention import RetentionConfig, compile_bounded_cache  # noqa: E402

#: A realistic per-(layer) decode-cache shape: batch 1, 8 kv heads, 512 tokens,
#: head_dim 64 (Llama-3.2-1B-Instruct's head_dim).
CACHE_SHAPE = (1, 8, 512, 64)


@pytest.fixture(autouse=True, scope="module")
def _single_threaded():
    """Run this module single-threaded and restore the previous setting."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _randn(shape, dtype=torch.float32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator).to(dtype)


def _dense_cache(shapes, dtype=torch.float32, seed=0):
    cache = DynamicCache()
    for index, shape in enumerate(shapes):
        cache.update(_randn(shape, dtype, seed + 2 * index),
                     _randn(shape, dtype, seed + 2 * index + 1), index)
    return cache


def _roundtrip(x, mode, group_size, axis, dtype=None):
    """``quantize_tensor`` + ``dequantize_tensor`` with the recorded group size."""
    packed, scale, state = Q.quantize_tensor(x, mode, group_size, axis)
    effective, _ = Q.group_layout(x.shape[axis], 1 if mode == "int8" else group_size)
    dense = Q.dequantize_tensor(packed, scale, mode, effective, axis, x.shape,
                                x.dtype if dtype is None else dtype, state=state)
    return packed, scale, dense


def _max_bound_ratio(x, dense, scale, mode, group_size, axis):
    bound = Q.per_element_scale(scale, x.shape, axis, group_size, mode)
    error = (dense.float() - x.float()).abs()
    return float(error.max()), float((error / bound).max())


# ---------------------------------------------------------------------------
# 1. round-trip error bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype,ratio_cap", [(torch.float32, 0.55), (torch.bfloat16, 0.85)])
def test_int8_roundtrip_error_bound_on_cache_shapes(dtype, ratio_cap):
    """int8: per-channel keys, per-token values, ``max|error| <= scale``."""
    x = _randn(CACHE_SHAPE, dtype)
    for axis, scale_shape in ((Q.KEY_AXIS, (1, 1, 1, CACHE_SHAPE[-1], 1)),
                              (Q.VALUE_AXIS, (1, 1, CACHE_SHAPE[-2], 1, 1))):
        packed, scale, dense = _roundtrip(x, "int8", 128, axis)
        assert packed.dtype == torch.int8
        assert packed.shape == x.shape                     # no packing for int8
        assert tuple(scale.shape) == scale_shape           # documented scale layout
        assert scale.dtype == torch.float32
        max_error, ratio = _max_bound_ratio(x, dense, scale, "int8", 1, axis)
        assert max_error > 0.0                             # int8 is lossy on random data
        assert ratio <= 1.0                                # the required bound: |error| <= scale
        # away from the cast back to bf16 the bound is the usual half-step
        assert ratio <= ratio_cap


@pytest.mark.parametrize("dtype,ratio_cap", [(torch.float32, 0.55), (torch.bfloat16, 0.60)])
def test_int4_roundtrip_error_bound_and_bit_layout(dtype, ratio_cap):
    """int4: same axes grouped by 128, two codes per byte along the head axis."""
    # -- the documented bit layout, transcribed literally (amax 7 => scale 1) --
    tiny = torch.tensor([[[[-7.0, -1.0, 0.0, 7.0]]]])      # (1, 1, 1, 4)
    packed, scale, dense = _roundtrip(tiny, "int4", 4, Q.KEY_AXIS)
    assert float(scale.item()) == 1.0
    # codes are stored biased by +8, low nibble = even channel, high = odd channel:
    # byte j = ((code[2j+1] + 8) << 4) | (code[2j] + 8)
    expected_byte0 = (((-1) + 8) << 4) | ((-7) + 8)        # codes -7, -1 -> 0x71
    expected_byte1 = ((7 + 8) << 4) | (0 + 8)              # codes  0, +7 -> 0xF8
    assert packed.tolist() == [[[[expected_byte0, expected_byte1]]]]
    assert packed.tolist() == [[[[0x71, 0xF8]]]]
    assert packed.dtype == torch.uint8
    assert torch.equal(dense, tiny)                        # codes -7..7 round-trip exactly

    # -- random tensors of a realistic cache shape, both axes --
    x = _randn(CACHE_SHAPE, dtype)
    for axis in (Q.KEY_AXIS, Q.VALUE_AXIS):
        packed, scale, dense = _roundtrip(x, "int4", 128, axis)
        assert packed.dtype == torch.uint8
        assert packed.shape == (*CACHE_SHAPE[:-1], math.ceil(CACHE_SHAPE[-1] / 2))
        max_error, ratio = _max_bound_ratio(x, dense, scale, "int4", 128, axis)
        assert max_error > 0.0
        assert ratio <= 1.0                                # |error| <= scale, elementwise
        assert ratio <= ratio_cap


# ---------------------------------------------------------------------------
# 2. zero-error cases, group scales, fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode,qmax", [("int8", 127), ("int4", 7)])
def test_zero_error_cases_are_exact(mode, qmax):
    """All-zero and constant tensors quantize exactly (documented)."""
    zeros = torch.zeros(*CACHE_SHAPE)
    packed, scale, dense = _roundtrip(zeros, mode, 128, Q.KEY_AXIS)
    assert float(scale.max()) == 1.0                       # pinned scale, not 0
    if mode == "int8":
        assert int(packed.abs().max()) == 0                # two's-complement zero
    else:
        # int4 stores code + 8, so a zero code is the nibble 8 on both halves
        assert int(packed.max()) == int(packed.min()) == 0x88
    assert torch.equal(dense, zeros)

    # c/qmax exactly representable => q = +-qmax and x_hat = c exactly
    for constant in (1.0, 0.5, float(qmax), float(qmax) * 0.25):
        x = torch.full(CACHE_SHAPE, constant)
        _, scale, dense = _roundtrip(x, mode, 128, Q.KEY_AXIS)
        assert torch.equal(dense, x), (mode, constant)


def test_group_scales_isolate_an_outlier_channel_for_keys():
    """One huge channel must not inflate the scale of the *other* key groups."""
    x = _randn(CACHE_SHAPE, seed=1)
    x[:, :, :, 0] = 50.0                                   # channel 0 is a 50x outlier
    packed, scale, dense = _roundtrip(x, "int4", 16, Q.KEY_AXIS)   # 4 channel groups
    error = (dense - x).abs()
    away, inside = error[..., 16:], error[..., 1:16]
    assert float(away.max()) <= float(Q.per_element_scale(
        scale, x.shape, Q.KEY_AXIS, 16, "int4")[..., 16:].max())   # the documented bound
    assert float(away.max()) < 0.5                         # clean groups keep a small scale
    assert float(inside.max()) > 1.0                       # the outlier's own group is coarse
    assert float(inside.max()) > 5 * float(away.max())     # grouping is what separates them
    assert float(scale.max() / scale.min()) > 4.0          # scales really do vary by group

    # the same tensor with a single global scale is far worse away from the outlier
    _, _, dense_global = _roundtrip(x, "int4", CACHE_SHAPE[-1], Q.KEY_AXIS)
    global_away = (dense_global - x).abs()[..., 16:]
    assert float(global_away.max()) > 4 * float(away.max())


def test_group_scales_isolate_an_outlier_token_for_values():
    """Values group along the token axis (per-token, KIVI), so only one block suffers."""
    x = _randn(CACHE_SHAPE, seed=2)
    x[:, :, 0, :] = 50.0                                   # token 0 is the outlier
    _, scale, dense = _roundtrip(x, "int4", 128, Q.VALUE_AXIS)
    assert tuple(scale.shape) == (1, 1, 4, 1, 1)            # 512 tokens / 128 = 4 groups
    error = (dense - x).abs()
    away = error[:, :, 128:, :]
    inside = error[:, :, 1:128, :]
    assert float(away.max()) < 0.5                         # tokens away from the outlier
    assert float(inside.max()) > 1.0                       # the outlier's own 128-token block
    assert float(inside.max()) > 5 * float(away.max())


def test_group_size_fallback_and_odd_head_dimension():
    # group_size >= axis length: exactly one group, axis never padded
    assert Q.group_layout(64, 128) == (64, 1)
    assert Q.group_layout(64, 64) == (64, 1)
    assert Q.group_layout(5, 128) == (5, 1)
    # smaller group_size: ceil division, the last group is zero-padded
    assert Q.group_layout(64, 16) == (16, 4)
    assert Q.group_layout(5, 2) == (2, 3)

    x = _randn((1, 2, 512, 64), seed=3)
    _, scale, _ = _roundtrip(x, "int4", 128, Q.KEY_AXIS)
    assert tuple(scale.shape) == (1, 1, 1, 1, 1)           # documented fallback for D=64

    # odd head dim: three bytes hold five codes, the last high nibble is padding
    y = _randn((1, 2, 5, 5), seed=4)
    packed, scale, dense = _roundtrip(y, "int4", 2, Q.VALUE_AXIS)
    assert packed.shape == (1, 2, 5, math.ceil(5 / 2))
    assert dense.shape == y.shape
    _, ratio = _max_bound_ratio(y, dense, scale, "int4", 2, Q.VALUE_AXIS)
    assert ratio <= 1.0


# ---------------------------------------------------------------------------
# 3. exact state-size accounting
# ---------------------------------------------------------------------------


def test_state_bytes_matches_hand_computed_values():
    """Small enough to add up by hand, including the metadata records."""
    cache = _dense_cache([(1, 2, 8, 4)])                    # one layer, fp32
    # dense: 2 tensors * 1*2*8*4 elements * 4 bytes
    assert Q.state_bytes(cache) == 2 * 1 * 2 * 8 * 4 * 4 == 512
    assert Q.state_bytes_full((1, 2, 8, 4), torch.float32) == 512

    int4 = Q.quantize_cache(cache, "int4", 4)
    # keys   packed 1*2*8*ceil(4/2)       = 32 B   (group_size 4 on D=4 -> 1 key group)
    # values packed 1*2*8*ceil(4/2)       = 32 B
    # keys_scale   1*1*1*1*1 floats * 4 B =  4 B
    # values_scale 1*1*2*1*1 floats * 4 B =  8 B   (group_size 4 on T=8 -> 2 groups)
    # + LAYER_METADATA_BYTES 32 + CACHE_METADATA_BYTES 16
    assert Q.state_bytes(int4) == 32 + 32 + 4 + 8 + 32 + 16 == 124
    assert Q.LAYER_METADATA_BYTES == 32 and Q.CACHE_METADATA_BYTES == 16

    int8 = Q.quantize_cache(cache, "int8", 128)
    # codes: 1*2*8*4 bytes each for keys and values       = 64 + 64 B
    # keys_scale   D=4 floats, values_scale T=8 floats    = 16 + 32 B
    # + 48 B metadata
    assert Q.state_bytes(int8) == 64 + 64 + 16 + 32 + 48 == 224

    assert Q.state_bytes(int4) < Q.state_bytes(int8) < Q.state_bytes(cache)


def test_state_bytes_ordering_on_realistic_bf16_cache():
    """int4 < int8 < bf16 on the shape the benchmark actually stores."""
    shape = (1, 4, 512, 64)
    cache = _dense_cache([shape, shape], dtype=torch.bfloat16)
    per_layer_dense = 2 * shape[1] * shape[2] * shape[3] * 2      # K+V, bf16 = 524288
    assert Q.state_bytes(cache) == 2 * per_layer_dense == 1_048_576
    assert Q.state_bytes_full(shape, torch.bfloat16) == per_layer_dense

    int8 = Q.quantize_cache(cache, "int8", 128)
    # per layer: 2 * (1*4*512*64) codes + D=64 scales + T=512 scales, fp32
    #          = 262144 + 64*4 + 512*4 = 264448, plus 32 B metadata
    assert Q.state_bytes(int8) == 2 * (264_448 + 32) + 16 == 528_976

    int4 = Q.quantize_cache(cache, "int4", 128)
    # per layer: 2 * (1*4*512*32) packed + 1 key group + 4 token groups, fp32
    #          = 131072 + 4 + 16 = 131092, plus 32 B metadata
    assert Q.state_bytes(int4) == 2 * (131_092 + 32) + 16 == 262_264

    assert Q.state_bytes(int4) < Q.state_bytes(int8) < Q.state_bytes(cache)


def test_cache_roundtrip_restores_dynamic_cache_and_metadata():
    cache = _dense_cache([CACHE_SHAPE, CACHE_SHAPE], dtype=torch.bfloat16)
    cache.qcc_attention_mask = torch.ones(1, CACHE_SHAPE[2], dtype=torch.long)
    cache.qcc_prompt_lengths = torch.tensor([CACHE_SHAPE[2]])
    quantized = Q.quantize_cache(cache, "int4", 128)
    assert Q.is_quantized(quantized) and not Q.is_quantized(cache)
    assert quantized.get_seq_length() == CACHE_SHAPE[2]
    assert quantized.keys_shape() == CACHE_SHAPE
    dense = Q.dequantize_cache(quantized)
    assert isinstance(dense, DynamicCache)
    assert dense.get_seq_length() == CACHE_SHAPE[2]
    assert torch.equal(dense.qcc_attention_mask, cache.qcc_attention_mask)
    assert torch.equal(dense.qcc_prompt_lengths, cache.qcc_prompt_lengths)

    # quantization is a pure function of the cache: two runs agree layer by layer
    again = Q.dequantize_cache(Q.quantize_cache(cache, "int4", 128))
    for index, (got, want) in enumerate(zip(dense.layers, cache.layers)):
        assert got.keys.dtype == want.keys.dtype == torch.bfloat16
        assert torch.equal(got.keys, again.layers[index].keys)
        assert torch.equal(got.values, again.layers[index].values)
        assert not torch.equal(got.keys, want.keys)         # int4 is lossy on random data

    # re-quantizing a QuantizedCache is defined as dequantize-then-quantize
    requantized = Q.quantize_cache(quantized, "int8", 128)
    expected = Q.quantize_cache(Q.dequantize_cache(quantized), "int8", 128)
    assert requantized.mode == "int8" and requantized.dtype == torch.bfloat16
    for got, want in zip(requantized.layers, expected.layers):
        assert torch.equal(got.keys_packed, want.keys_packed)
        assert torch.equal(got.values_packed, want.values_packed)
        assert torch.equal(got.keys_scale, want.keys_scale)
        assert torch.equal(got.values_scale, want.values_scale)


# ---------------------------------------------------------------------------
# 4. attention output and greedy decode on a tiny Llama
# ---------------------------------------------------------------------------

TINY_CONFIG = dict(vocab_size=256, hidden_size=64, intermediate_size=128,
                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                   max_position_embeddings=4096)

PROMPT_TOKENS = 96
BUDGET = 24          # compile_bounded_cache slots (lex_cap=0)
DECODE_STEPS = 8


def tiny_model():
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(**TINY_CONFIG)).eval()


def compile_tiny_cpu_cache():
    """A real compiled cache from the shipped retention law, on CPU."""
    model = tiny_model()
    ids = torch.randint(0, TINY_CONFIG["vocab_size"], (1, PROMPT_TOKENS))
    config = RetentionConfig(budget=BUDGET, lex_cap=0, observation_window=8,
                             prefill_chunk=32, key_chunk=32)
    cache, logits = compile_bounded_cache(model, ids, config)
    assert cache.get_seq_length() == BUDGET
    return model, cache, logits, ids


def clone_cache(cache):
    """A private copy: the decode loop appends to the cache it is given."""
    clone = DynamicCache()
    for index, layer in enumerate(cache.layers):
        clone.update(layer.keys.clone(), layer.values.clone(), index)
    for name in ("qcc_attention_mask", "qcc_prompt_lengths"):
        if hasattr(cache, name):
            setattr(clone, name, getattr(cache, name))
    return clone


@torch.no_grad()
def greedy(model, cache, first_token, prompt_length, steps, forced=None):
    """The harness decode loop, on CPU: absolute positions, greedy, no EOS stop."""
    kept = cache.get_seq_length()
    mask = torch.ones(1, kept + 1, dtype=torch.long)
    current, tokens, logits = first_token, [], []
    for step in range(steps):
        out = model(current, past_key_values=cache, attention_mask=mask,
                    cache_position=torch.arange(kept + step, kept + step + 1),
                    position_ids=torch.tensor([[prompt_length + step]]), use_cache=True)
        step_logits = out.logits[0, -1]
        logits.append(step_logits)
        token = step_logits.argmax().reshape(1, 1)
        tokens.append(token)
        mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long)], dim=-1)
        current = token if forced is None else forced[:, step:step + 1]
    return torch.cat(tokens, dim=1), torch.stack(logits, dim=0)


def softmax_attention(query, keys, values, scaling):
    """Textbook exact attention; keys/values are ``(T, D)``, query ``(D,)``."""
    weights = torch.softmax(torch.einsum("d,td->t", query, keys) * scaling, dim=0)
    return weights @ values, weights


def test_quantized_attention_obeys_the_convexity_bound():
    """``||o_q - o||_2 <= max_t ||dv_t||_2 + ||p_q - p||_1 * max_t ||v_t||_2``.

    The value error enters attention through a convex combination, so its effect
    is at most the largest per-token value error; the key error can only move the
    softmax weights, and for logit perturbations bounded by ``eps`` the total
    variation distance of the two softmaxes is at most ``2 * eps``.  Both terms
    are computed from the quantized cache, so the tolerance is derived, not
    tuned.  A random probe query is used because the bound holds for any query.
    """
    model, cache, _, _ = compile_tiny_cpu_cache()
    for mode, group_size in (("int8", 128), ("int4", 128)):
        dense = Q.dequantize_cache(Q.quantize_cache(cache, mode, group_size))
        generator = torch.Generator().manual_seed(7)
        for layer_index in range(len(cache.layers)):
            keys_full = cache.layers[layer_index].keys[0]
            values_full = cache.layers[layer_index].values[0]
            keys_q = dense.layers[layer_index].keys[0]
            values_q = dense.layers[layer_index].values[0]
            scaling = float(model.model.layers[layer_index].self_attn.scaling)
            for head in range(keys_full.shape[0]):
                query = torch.randn(keys_full.shape[-1], generator=generator)
                out_full, weights_full = softmax_attention(
                    query, keys_full[head], values_full[head], scaling)
                out_q, weights_q = softmax_attention(
                    query, keys_q[head], values_q[head], scaling)
                value_term = float((values_q[head] - values_full[head]).norm(dim=-1).max())
                eps = float(((keys_q[head] - keys_full[head]).abs().max(dim=-1).values
                             * query.norm() * scaling).max())
                assert float((weights_q - weights_full).abs().sum()) <= 2 * eps + 1e-6
                assert float((out_q - out_full).norm()) <= (
                    value_term + 2 * eps * float(values_full[head].norm(dim=-1).max()) + 1e-6)
                assert value_term > 0.0                        # the comparison is not vacuous


def test_greedy_decode_from_quantized_cache_matches_unquantized():
    """The deliverable's end-to-end check, on a real ``compile_bounded_cache``."""
    model, cache, logits, _ = compile_tiny_cpu_cache()
    first_token = logits[:, -1:].argmax(-1)
    reference_tokens, reference_logits = greedy(
        model, clone_cache(cache), first_token, PROMPT_TOKENS, DECODE_STEPS)
    reference_margins = (reference_logits.topk(2, dim=-1).values[:, 0]
                         - reference_logits.topk(2, dim=-1).values[:, 1])

    for mode, tolerance in (("int8", 0.01), ("int4", 0.05)):
        quantized = Q.quantize_cache(cache, mode, 128)
        assert Q.state_bytes(quantized) < Q.state_bytes(cache)
        # free-running greedy decode (the harness comparison)
        tokens, _ = greedy(model, Q.dequantize_cache(quantized), first_token,
                           PROMPT_TOKENS, DECODE_STEPS)
        assert torch.equal(tokens, reference_tokens), (mode, tokens, reference_tokens)
        # teacher-forced on the reference tokens: logit fidelity at every step
        forced_cache = Q.dequantize_cache(Q.quantize_cache(cache, mode, 128))
        _, forced_logits = greedy(model, forced_cache, first_token, PROMPT_TOKENS,
                                  DECODE_STEPS, forced=reference_tokens)
        perturbation = (forced_logits - reference_logits).abs().max(dim=-1).values
        assert float(perturbation.max()) <= tolerance, (mode, perturbation)
        assert float(perturbation.max()) > 0.0             # quantization is really applied
        # no *unexplained* argmax flip: a step may only differ when its reference
        # margin is smaller than the measured perturbation
        for step, same in enumerate((tokens == reference_tokens)[0].tolist()):
            assert same or float(perturbation[step]) >= float(reference_margins[step])


# ---------------------------------------------------------------------------
# 5. nf4 delegation
# ---------------------------------------------------------------------------


def test_nf4_without_bitsandbytes_raises_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    monkeypatch.setitem(sys.modules, "bitsandbytes.functional", None)
    cache = _dense_cache([(1, 2, 8, 4)])
    with pytest.raises(RuntimeError, match="bitsandbytes"):
        Q.quantize_cache(cache, "nf4", 4)


def test_nf4_roundtrip_when_bitsandbytes_is_available():
    pytest.importorskip("bitsandbytes")
    cache = _dense_cache([(1, 2, 64, 64)], dtype=torch.float32)
    keys = cache.layers[0].keys
    try:
        quantized = Q.quantize_cache(cache, "nf4", 64)
    except Exception as exc:  # a CPU-only bitsandbytes build may refuse
        pytest.skip(f"bitsandbytes 4-bit unavailable here: {type(exc).__name__}: {exc}")
    dense = Q.dequantize_cache(quantized)
    assert dense.layers[0].keys.shape == keys.shape
    assert dense.layers[0].keys.dtype == torch.float32
    # NF4 is coarse (16 levels, 64-element blocks): the error stays below the amax
    assert float((dense.layers[0].keys - keys).abs().max()) <= float(keys.abs().max())
    assert Q.state_bytes(quantized) < Q.state_bytes(cache)


# ---------------------------------------------------------------------------
# 6. the RULER runner's arm pipeline (still CPU-only, tiny model, no model load)
# ---------------------------------------------------------------------------


class _IdsTokenizer:
    """Minimal stand-in for the runner's ``tokenizer.decode`` call."""

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def test_ruler_runner_arms_compose_pruning_and_quantization(tmp_path):
    """``run_arm`` + ``summarize`` for the eviction and quantization arms.

    The runner itself is a GPU script (it loads a real checkpoint), but its arm
    pipeline is device-agnostic, so this drives it with the tiny CPU model: full
    vs bounded, bf16 vs int8 vs int4, and the byte/score bookkeeping that feeds
    the Pareto plot.
    """
    import benchmark_kv_quant_ruler as runner

    model = tiny_model()
    model.lm_head = runner._LastTokenLMHead(model.lm_head)
    arms = ["full", "full_int8", "full_int4", "bounded8", "bounded8_int8", "bounded8_int4"]
    budget = 8
    args = runner.parse_args(["--out", str(tmp_path / "ruler.json"), "--device", "cpu",
                              "--dtype", "float32", "--max-new", "3", "--budget", str(budget),
                              "--lex-cap", "0", "--arms", *arms])
    assert runner.parse_arm("bounded8192_int4", 4096) == ("bounded", 8192, "int4")
    assert runner.parse_arm("full_nf4", 4096) == ("full", None, "nf4")

    retention = RetentionConfig(budget=budget, lex_cap=0, observation_window=8,
                                prefill_chunk=32, key_chunk=32)
    ids = torch.randint(0, TINY_CONFIG["vocab_size"], (1, PROMPT_TOKENS))
    row = {"task": "niah_single_1", "index": 0, "length": PROMPT_TOKENS,
           "outputs": ["0"], "input": "stub"}
    rows = []
    for arm in arms:
        entry = runner.run_arm(model, _IdsTokenizer(), ids, row, arm, args, retention,
                               {0}, ())
        assert entry["status"] == "ok", (arm, entry)
        entry["row"] = 0
        rows.append(entry)

    by_arm = {entry["arm"]: entry for entry in rows}
    assert by_arm["full"]["kept_slots"] == PROMPT_TOKENS
    assert by_arm["bounded8_int4"]["kept_slots"] == budget
    # the dense bf16/fp32 reference at the same slots is what "x full" divides by
    assert by_arm["full"]["state_bytes"] == by_arm["full"]["full_state_bytes"]
    assert by_arm["full"]["compression_vs_full"] == 1.0
    for arm in ("full_int8", "full_int4"):
        assert by_arm[arm]["state_bytes"] < by_arm["full"]["state_bytes"]
        assert by_arm[arm]["dense_state_bytes"] == by_arm["full"]["dense_state_bytes"]
    assert (by_arm["bounded8_int4"]["state_bytes"] < by_arm["bounded8_int8"]["state_bytes"]
            < by_arm["bounded8"]["state_bytes"])
    # the scored field is the documented partial recall of the decoded text
    for entry in rows:
        assert entry["recall_partial"] == runner.partial_recall(entry["outputs"],
                                                                entry["prediction"])
        assert entry["decode_s"] >= 0.0 and entry["generated"] == 3

    summary = runner.summarize(rows, args, ["niah_single_1"])
    json.dumps(summary)                                   # the payload must serialize
    assert set(summary["aggregate"]) == set(arms)
    assert summary["pareto"]["aggregate"], "the Pareto front must not be empty"
    cheapest = summary["pareto"]["aggregate"][0]
    assert cheapest["arm"] == "bounded8_int4"              # fewest bytes, no worse score
    assert all(point["score"] is not None for point in summary["pareto"]["aggregate"])
