"""Decode-time KV quantization for a Hugging Face ``DynamicCache``.

This is the *quantization* arm of the bounded-decode comparison: it compresses a
materialised decode cache (the output of an exact prefill, or of
``qcc_transformer.retention.compile_bounded_cache``) so that quality can be
plotted against decode-state bytes on the same axis as eviction policies.
Quantization and eviction compose: ``quantize_cache`` accepts a pruned cache and
does not care how many slots survived.

The module is dependency-free.  ``int8`` and ``int4`` are implemented with plain
tensor arithmetic; ``nf4`` is the single mode that delegates, to bitsandbytes.


The quantization rule (reimplementable from this docstring)
===========================================================

Cache tensors use the Hugging Face layout ``(B, H_kv, T, D)``: ``B`` requests,
``H_kv`` key/value heads, ``T`` retained tokens and ``D`` head (channel)
dimension.  Both keys and values are quantized **symmetrically with an implicit
zero-point of exactly 0**::

    qmax  = 127                (int8)      or  7            (int4)
    scale = max(|X| over the group) / qmax
    q     = clamp(round(X / scale), -qmax, +qmax)
    X_hat = q * scale

``round`` is ``torch.round`` (round-half-to-even, IEEE); ``clamp`` is applied
*after* rounding, so saturated elements sit exactly on the extreme code.  All
arithmetic is done in float32 and the result is cast back to the cache dtype
(bf16 for the benchmarks here) on dequantization.

Groups and axes (KIVI layout)
-----------------------------
A *group* is a contiguous run of ``group_size`` entries along one axis of a
tensor.  A group's scale is the maximum absolute value **over the entire
group**, i.e. over every element that shares the group index -- all of the other
axis and all members of the group.  The axis assignment is the KIVI one:

===========  ======================  =========================================
tensor       grouping axis           meaning
===========  ======================  =========================================
keys         head axis (``-1``)      scale indexed by channel group, shared by
                                     every token  ("per-channel")
values       token axis (``-2``)     scale indexed by token group, shared by
                                     every channel ("per-token")
===========  ======================  =========================================

* ``int8`` uses those two axes with an effective ``group_size = 1`` ("no
  grouping"): one scale per ``(layer, kv-head, channel)`` for keys, one scale
  per ``(layer, kv-head, token)`` for values.  ``int8`` ignores the
  ``group_size`` argument.
* ``int4`` uses ``group_size`` (default 128) along those axes.

Group-size fallback (documented, and exercised by the tests)::

    g        = min(group_size, axis_length)
    n_groups = ceil(axis_length / g)

* ``group_size >= axis_length`` => ``g = axis_length``, exactly one group per
  head/token run and the axis is never padded.  For Llama-3.2-1B-Instruct
  (``D = 64``) this makes int4 keys carry a single scale per ``(layer, kv-head)``
  and int4 values one scale per 128 retained tokens.
* otherwise the axis is zero-padded to ``n_groups * g`` before the max and the
  quantization; the padding is dropped on dequantization.  Zero padding cannot
  change a max and its reconstructed values are discarded, so padding never
  affects a real element.
* a group whose elements are all zero has ``max(|X|) = 0``; the scale is then
  pinned to exactly ``1.0`` (rather than 0 or an epsilon) so that
  ``q = clamp(round(0/1)) = 0`` and the reconstruction is exactly 0.

Scale layout: ``scale`` has the rank of the *grouped view*, i.e. ``X.dim() + 1``,
with ``n_groups`` at the position of the grouped axis and 1 everywhere else
(including the extra within-group axis immediately after it).  For a 4-D cache
tensor that is ``(1, 1, 1, n_groups, 1)`` for keys and
``(1, 1, n_groups, 1, 1)`` for values.  ``scale`` is stored in
``scale_dtype`` (default ``float32``; bf16 scales are enough to keep the
``|X - X_hat| <= scale`` bound but add their own rounding error).

Exactness: quantization is lossless when every element of a group shares one
value ``c`` with ``c / qmax`` exactly representable -- ``q = +-qmax`` and
``X_hat = c`` -- which covers constant tensors with a power-of-two value
(``c = qmax * 2**-k``; e.g. 127, 1.0, 0.5 for int8 and 7, 1.0, 0.5 for int4), and
it is exactly lossless for an all-zero tensor because of the pinned scale.
Otherwise ``|X - X_hat| <= scale`` elementwise, and ``<= scale/2`` away from the
float32/dtype-rounding terms.

Stored representation (bit layout)
==================================

* ``int8``: codes are stored as a signed two's-complement ``torch.int8`` tensor
  of the original shape ``(B, H_kv, T, D)`` -- one byte per element, no packing.
  ``keys_scale`` has shape ``(1, 1, 1, n_groups, 1)`` and ``values_scale``
  ``(1, 1, n_groups, 1, 1)``.
* ``int4``: codes are clamped to ``[-7, +7]`` (the symmetric 4-bit range; the
  code ``-8`` is deliberately unused) and packed **two per byte along the last
  (head) axis**, regardless of which axis the groups run along::

      packed[..., j] = ((code[..., 2j + 1] + 8) << 4) | (code[..., 2j] + 8)

  i.e. the low nibble holds the even channel and the high nibble the odd
  channel, and the stored nibble is the *biased* code ``code + 8`` in ``0..15``
  (``0`` would mean code ``-8``, which never occurs).  For odd ``D`` the high
  nibble of the final byte is 0 and is ignored on unpacking.  Packed storage is
  ``torch.uint8`` of shape ``(B, H_kv, T, ceil(D/2))`` -- half the dense byte
  count -- plus the scales above.
* ``nf4``: delegated to ``bitsandbytes.functional.quantize_4bit`` with
  ``quant_type="nf4"`` and ``blocksize=group_size`` (default 128) over
  bitsandbytes' own flattened, row-major block layout -- *not* the KIVI axes
  above -- so a reviewer comparing nf4 bytes to int4 bytes is comparing two
  different group layouts.  ``bitsandbytes`` must be importable; otherwise a
  ``RuntimeError`` naming the extra is raised.  Both the packed tensor and the
  fp32 per-block ``absmax`` (plus any nested ``state2`` absmax when
  ``compress_statistics=True``) are counted by :func:`state_bytes`.

Byte accounting
===============

:func:`state_bytes` returns the exact size of the stored representation: for a
plain cache the bytes of its ``keys``/``values`` tensors, and for a
:class:`QuantizedCache` the packed tensors plus every scale/absmax tensor plus a
small, documented metadata record::

    state_bytes = CACHE_METADATA_BYTES                       # 16 B, per cache
                + sum over layers of (LAYER_METADATA_BYTES   # 32 B, per layer
                                      + stored tensor bytes)

:func:`state_bytes_full(keys_shape, dtype)` gives the dense K+V reference size
for the same shape, so a Pareto plot of quality against decode-state bytes is
directly computable.

Caveat that matters for interpreting a Pareto point: this reference
implementation has no fused int8/int4 attention kernel, so
:func:`dequantize_cache` materialises a dense cache in the model dtype before
decode.  ``state_bytes`` is therefore the *stored* decode state (what a kernel
would read, and what makes the bytes axis meaningful), while the resident
footprint during decode is the dense one.  Quantized arms in the benchmark
runner measure quality and bytes, not decode speed.

Typical use::

    from kv_quant import quantize_cache, dequantize_cache, state_bytes
    qcache = quantize_cache(cache, mode="int4", group_size=128)
    bytes_ = state_bytes(qcache)
    cache = dequantize_cache(qcache)          # a real DynamicCache
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import torch

__all__ = [
    "MODES",
    "QMAX",
    "KEY_AXIS",
    "VALUE_AXIS",
    "LAYER_METADATA_BYTES",
    "CACHE_METADATA_BYTES",
    "QuantizedLayer",
    "QuantizedCache",
    "check_mode",
    "dtype_nbytes",
    "group_layout",
    "quantize_tensor",
    "dequantize_tensor",
    "per_element_scale",
    "quantize_cache",
    "dequantize_cache",
    "state_bytes",
    "state_bytes_full",
    "is_quantized",
]

#: Supported modes. ``nf4`` needs bitsandbytes; the others are self-contained.
MODES = ("int8", "int4", "nf4")

#: Symmetric quantization range per mode (zero-point is always 0).
QMAX = {"int8": 127, "int4": 7}

#: Axis of the *head* (channel) dimension and of the *token* dimension in the
#: ``(B, H_kv, T, D)`` cache layout.  Keys group along the head axis
#: (per-channel), values along the token axis (per-token), as in KIVI.
KEY_AXIS = -1
VALUE_AXIS = -2

#: Documented per-layer metadata record: 4 x int32 for the ``(B, H, T, D)``
#: shape header, 1 byte mode id, 1 byte scale-dtype id, 2 bytes effective group
#: sizes, padded to two 16-byte cache lines.
LAYER_METADATA_BYTES = 32

#: Documented per-cache metadata record: mode id, scale-dtype id, layer count,
#: default group size.
CACHE_METADATA_BYTES = 16


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def dtype_nbytes(dtype: torch.dtype) -> int:
    """Bytes per element of ``dtype`` without allocating a real tensor."""
    return int(torch.empty((), dtype=dtype).element_size())


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel()) * int(tensor.element_size())


def group_layout(size: int, group_size: int | None) -> tuple[int, int]:
    """``(g, n_groups)`` for an axis of length ``size``, with the fallback.

    ``g = min(group_size, size)`` and ``n_groups = ceil(size / g)``, so a
    ``group_size`` at least as large as the axis yields a single group covering
    the whole axis and never pads.  ``group_size`` of ``None`` or ``<= 0`` means
    "one group over the whole axis" as well.
    """
    size = int(size)
    if size <= 0:
        raise ValueError("cannot quantize an empty axis")
    if group_size is None or int(group_size) <= 0:
        return size, 1
    g = min(int(group_size), size)
    return g, math.ceil(size / g)


def _pad_axis(x: torch.Tensor, axis: int, padded: int) -> torch.Tensor:
    """Zero-pad ``axis`` to ``padded`` (no-op when already that long)."""
    if x.shape[axis] == padded:
        return x
    pad = [0] * (2 * x.dim())
    pad[2 * (x.dim() - 1 - axis) + 1] = padded - x.shape[axis]
    return torch.nn.functional.pad(x, tuple(pad), mode="constant", value=0.0)


def _grouped_view(x: torch.Tensor, axis: int, n_groups: int, g: int):
    """``(..., n_groups, g, ...)`` view of ``x`` with the axis zero-padded."""
    padded = n_groups * g
    x = _pad_axis(x, axis, padded)
    shape = list(x.shape)
    return x.reshape(*shape[:axis], n_groups, g, *shape[axis + 1:])


def _ungroup_view(grouped: torch.Tensor, axis: int, n_groups: int, g: int,
                  size: int) -> torch.Tensor:
    """Inverse of :func:`_grouped_view`: merge the group axes and drop padding."""
    shape = list(grouped.shape)
    shape[axis] = n_groups * g
    del shape[axis + 1]
    out = grouped.reshape(shape)
    return out.narrow(axis, 0, size).contiguous()


def _pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Signed int4 codes in ``[-7, 7]`` -> ``uint8``, two per byte along dim -1.

    Byte ``j`` = ``((code[2j+1] + 8) << 4) | (code[2j] + 8)``.  An odd head
    dimension pads the high nibble of the final byte with 0.
    """
    if codes.shape[-1] % 2:
        pad = codes.new_zeros(*codes.shape[:-1], 1)
        codes = torch.cat([codes, pad], dim=-1)
    biased = (codes.to(torch.int16) + 8).to(torch.uint8)
    low, high = biased[..., 0::2], biased[..., 1::2]
    return ((high.to(torch.int16) << 4) | low.to(torch.int16)).to(torch.uint8)


def _unpack_int4(packed: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Inverse of :func:`_pack_int4`; returns int8 codes of ``(..., head_dim)``."""
    high = (packed >> 4).to(torch.int16)
    low = (packed & 0x0F).to(torch.int16)
    codes = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)
    codes = codes[..., :head_dim] - 8
    return codes.to(torch.int8)


def _bitsandbytes():
    try:
        from bitsandbytes import functional as bnb_functional
    except Exception as exc:  # pragma: no cover - exercised only without bnb
        raise RuntimeError(
            "mode='nf4' needs the optional 'bitsandbytes' dependency; "
            "install it (pip install bitsandbytes) or use mode='int8'/'int4', "
            "which are implemented in this module without any dependency"
        ) from exc
    return bnb_functional


def _bnb_state_bytes(state: Any) -> int:
    """Bytes of a bitsandbytes ``QuantState`` (absmax, plus nested state2)."""
    if state is None:
        return 0
    total = _tensor_bytes(getattr(state, "absmax", None))
    nested = getattr(state, "state2", None)
    if nested is not None:
        total += _bnb_state_bytes(nested)
    return total


# ---------------------------------------------------------------------------
# tensor-level quantization
# ---------------------------------------------------------------------------


def quantize_tensor(x: torch.Tensor, mode: str = "int8", group_size: int | None = 128,
                    axis: int = KEY_AXIS, *, scale_dtype: torch.dtype = torch.float32,
                    compress_statistics: bool = False):
    """Quantize one cache tensor.

    ``x`` is a float tensor ``(..., T, D)``; ``axis`` selects the grouped axis
    (``-1`` for the KIVI per-channel key rule, ``-2`` for the per-token value
    rule).  Returns ``(packed, scale, state)`` where ``packed`` is the stored
    code tensor (``int8`` for int8, ``uint8`` for int4/nf4), ``scale`` is the
    scale tensor in ``scale_dtype`` (bitsandbytes' ``absmax`` for nf4) and
    ``state`` is ``None`` except for nf4, where it is the bitsandbytes
    ``QuantState`` needed to dequantize.

    The rule, the group fallback, the scale layout and the int4 bit layout are
    documented in the module docstring.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if x.dim() < 2:
        raise ValueError(f"cache tensors are (..., T, D); got shape {tuple(x.shape)}")
    if mode == "nf4":
        bnb_functional = _bitsandbytes()
        blocksize = int(group_size) if group_size else 128
        packed, state = bnb_functional.quantize_4bit(
            x.detach().contiguous(), blocksize=blocksize, quant_type="nf4",
            compress_statistics=compress_statistics)
        return packed, state.absmax, state

    qmax = QMAX[mode]
    effective_group = 1 if mode == "int8" else group_size
    ax = axis % x.dim()
    size = int(x.shape[ax])
    g, n_groups = group_layout(size, effective_group)

    work = x.detach().float()
    grouped = _grouped_view(work, ax, n_groups, g)
    reduce_dims = tuple(d for d in range(grouped.dim()) if d != ax)
    scale = grouped.abs().amax(dim=reduce_dims, keepdim=True) / qmax
    # all-zero group: pin the scale to exactly 1 so codes are 0 and reconstruction is exact
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.round(grouped / scale).clamp(-qmax, qmax)
    codes = _ungroup_view(codes, ax, n_groups, g, size)
    scale = scale.to(scale_dtype).contiguous()
    if mode == "int8":
        return codes.to(torch.int8).contiguous(), scale, None
    return _pack_int4(codes.to(torch.int8)), scale, None


def dequantize_tensor(packed: torch.Tensor, scale: torch.Tensor, mode: str,
                      group_size: int | None, axis: int, shape: Sequence[int],
                      dtype: torch.dtype = torch.float32, state: Any = None) -> torch.Tensor:
    """Inverse of :func:`quantize_tensor`; returns a dense ``dtype`` tensor.

    ``shape`` is the original ``(B, H, T, D)`` shape and ``group_size`` the
    *effective* group size recorded by :func:`quantize_cache` (for int4 from
    :func:`group_layout`, i.e. already clamped to the axis length).
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    shape = tuple(int(s) for s in shape)
    if mode == "nf4":
        bnb_functional = _bitsandbytes()
        dense = bnb_functional.dequantize_4bit(packed, quant_state=state)
        return dense.reshape(shape).to(dtype)

    ax = axis % len(shape)
    size = shape[ax]
    g, n_groups = group_layout(size, 1 if mode == "int8" else group_size)
    if mode == "int8":
        codes = packed.to(torch.float32)
    else:
        codes = _unpack_int4(packed, shape[-1]).to(torch.float32)
    grouped = _grouped_view(codes, ax, n_groups, g)
    out = grouped * scale.to(torch.float32)
    return _ungroup_view(out, ax, n_groups, g, size).to(dtype)


def per_element_scale(scale: torch.Tensor, shape: Sequence[int], axis: int,
                      group_size: int | None, mode: str = "int4") -> torch.Tensor:
    """Broadcast a stored ``scale`` back to the shape of the tensor it quantized.

    The result is the per-element bound ``|x - x_hat| <= per_element_scale``
    (before the cast back to the cache dtype), i.e. the documentation of the
    invariant is directly checkable.
    """
    shape = tuple(int(s) for s in shape)
    ax = axis % len(shape)
    size = shape[ax]
    g, n_groups = group_layout(size, 1 if mode == "int8" else group_size)
    grouped_shape = (*shape[:ax], n_groups, g, *shape[ax + 1:])
    return _ungroup_view(scale.expand(grouped_shape), ax, n_groups, g, size)


# ---------------------------------------------------------------------------
# cache-level container
# ---------------------------------------------------------------------------


@dataclass
class QuantizedLayer:
    """One layer's quantized decode state (no dense cache is retained)."""

    keys_packed: torch.Tensor
    values_packed: torch.Tensor
    keys_scale: torch.Tensor | None
    values_scale: torch.Tensor | None
    mode: str
    key_group_size: int          # effective g along the head axis
    value_group_size: int        # effective g along the token axis
    shape: tuple[int, int, int, int]
    dtype: torch.dtype
    keys_state: Any = None       # bitsandbytes QuantState (nf4 only)
    values_state: Any = None
    key_axis: int = KEY_AXIS
    value_axis: int = VALUE_AXIS

    @property
    def batch_size(self) -> int:
        return int(self.shape[0])

    @property
    def num_heads(self) -> int:
        return int(self.shape[1])

    @property
    def num_tokens(self) -> int:
        return int(self.shape[2])

    @property
    def head_dim(self) -> int:
        return int(self.shape[3])

    def stored_bytes(self) -> int:
        """Bytes of this layer's packed tensors plus scales/absmax."""
        total = _tensor_bytes(self.keys_packed) + _tensor_bytes(self.values_packed)
        if self.mode == "nf4":
            total += _bnb_state_bytes(self.keys_state) + _bnb_state_bytes(self.values_state)
        else:
            total += _tensor_bytes(self.keys_scale) + _tensor_bytes(self.values_scale)
        return total


@dataclass
class QuantizedCache:
    """A decode cache stored as packed codes plus scales.

    Deliberately *not* a ``DynamicCache``: the dense tensors do not exist here,
    so ``state_bytes`` cannot be confused with a resident footprint.  Call
    :func:`dequantize_cache` to get a real ``DynamicCache`` for the decode loop.
    """

    layers: list[QuantizedLayer] = field(default_factory=list)
    mode: str = "int8"
    group_size: int = 128
    scale_dtype: torch.dtype = torch.float32
    dtype: torch.dtype = torch.float32
    extras: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.layers)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Retained tokens per head (the cache's decode width)."""
        if not self.layers:
            return 0
        return self.layers[layer_idx].num_tokens

    def stored_bytes(self) -> int:
        """Exact bytes of the stored representation, metadata included."""
        total = CACHE_METADATA_BYTES
        for layer in self.layers:
            total += LAYER_METADATA_BYTES + layer.stored_bytes()
        return total

    def head_dim(self) -> int:
        return self.layers[0].head_dim if self.layers else 0

    def keys_shape(self) -> tuple[int, int, int, int]:
        return self.layers[0].shape if self.layers else (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# cache-level API
# ---------------------------------------------------------------------------


def is_quantized(cache: Any) -> bool:
    return isinstance(cache, QuantizedCache)


def check_mode(mode: str) -> None:
    """Raise if ``mode`` cannot be used in this environment.

    ``int8``/``int4`` are always available; ``nf4`` raises the documented
    ``RuntimeError`` when bitsandbytes is not importable.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if mode == "nf4":
        _bitsandbytes()


def _iter_cache_kv(cache: Any) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(keys, values)`` per layer of a DynamicCache-like object."""
    layers = getattr(cache, "layers", None)
    if layers:
        for layer in layers:
            keys, values = getattr(layer, "keys", None), getattr(layer, "values", None)
            if keys is None or values is None:
                raise TypeError("cache layer holds no keys/values (empty cache?)")
            yield keys, values
        return
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None and value_cache is not None and len(key_cache):
        for keys, values in zip(key_cache, value_cache):
            yield keys, values
        return
    raise TypeError(
        "expected a Hugging Face DynamicCache (with .layers[i].keys/.values or "
        f".key_cache/.value_cache); got {type(cache).__name__}")


def quantize_cache(cache: Any, mode: str = "int8", group_size: int = 128, *,
                   scale_dtype: torch.dtype = torch.float32,
                   compress_statistics: bool = False) -> QuantizedCache:
    """Quantize every layer of ``cache``; returns a :class:`QuantizedCache`.

    Keys use the per-channel rule (groups along the head axis) and values the
    per-token rule (groups along the token axis); see the module docstring.
    ``group_size`` is ignored by ``int8`` and used as bitsandbytes' ``blocksize``
    by ``nf4``.  Passing an already quantized cache dequantizes it first, so
    ``int8 -> int4`` re-quantization is well defined.
    """
    check_mode(mode)
    if isinstance(cache, QuantizedCache):
        cache = dequantize_cache(cache)
    extras = {name: value for name, value in vars(cache).items()
              if name.startswith("qcc_")} if hasattr(cache, "__dict__") else {}

    layers: list[QuantizedLayer] = []
    dtype = None
    for keys, values in _iter_cache_kv(cache):
        if keys.dim() != 4:
            raise ValueError(f"expected (B, H, T, D) cache tensors; got {tuple(keys.shape)}")
        if values.shape != keys.shape:
            raise ValueError(f"keys {tuple(keys.shape)} and values {tuple(values.shape)} differ")
        dtype = keys.dtype
        key_g = group_layout(keys.shape[KEY_AXIS], 1 if mode == "int8" else group_size)[0]
        value_g = group_layout(keys.shape[VALUE_AXIS], 1 if mode == "int8" else group_size)[0]
        keys_packed, keys_scale, keys_state = quantize_tensor(
            keys, mode, group_size, KEY_AXIS, scale_dtype=scale_dtype,
            compress_statistics=compress_statistics)
        values_packed, values_scale, values_state = quantize_tensor(
            values, mode, group_size, VALUE_AXIS, scale_dtype=scale_dtype,
            compress_statistics=compress_statistics)
        layers.append(QuantizedLayer(
            keys_packed=keys_packed, values_packed=values_packed,
            keys_scale=keys_scale, values_scale=values_scale, mode=mode,
            key_group_size=key_g, value_group_size=value_g,
            shape=tuple(int(s) for s in keys.shape), dtype=dtype,
            keys_state=keys_state, values_state=values_state))
    if not layers:
        raise ValueError("cache has no layers to quantize")
    return QuantizedCache(layers=layers, mode=mode, group_size=int(group_size),
                          scale_dtype=scale_dtype, dtype=dtype, extras=extras)


def dequantize_cache(quantized: QuantizedCache):
    """Rebuild a dense ``DynamicCache`` from a :class:`QuantizedCache`.

    The result is a real Hugging Face cache, so the existing decode loops (and
    ``model(..., past_key_values=cache)``) work unchanged.  ``qcc_*`` attributes
    of the cache that was quantized (for example ``qcc_attention_mask`` and
    ``qcc_prompt_lengths`` written by
    ``qcc_transformer.retention.compile_bounded_cache``) are restored.
    """
    from transformers import DynamicCache

    if not isinstance(quantized, QuantizedCache):
        raise TypeError(f"expected a QuantizedCache; got {type(quantized).__name__}")
    dense = DynamicCache()
    for index, layer in enumerate(quantized.layers):
        keys = dequantize_tensor(
            layer.keys_packed, layer.keys_scale, layer.mode, layer.key_group_size,
            layer.key_axis, layer.shape, layer.dtype, state=layer.keys_state)
        values = dequantize_tensor(
            layer.values_packed, layer.values_scale, layer.mode, layer.value_group_size,
            layer.value_axis, layer.shape, layer.dtype, state=layer.values_state)
        dense.update(keys, values, index)
    for name, value in quantized.extras.items():
        setattr(dense, name, value)
    return dense


# ---------------------------------------------------------------------------
# byte accounting
# ---------------------------------------------------------------------------


def state_bytes(cache: Any) -> int:
    """Exact bytes of the stored decode state.

    ``QuantizedCache`` counts packed codes + scales/absmax + the documented
    metadata records; a dense cache (or any object with per-layer
    ``keys``/``values``, including transformers' legacy ``key_cache`` layout)
    counts the bytes of its dense tensors.  A bare tensor counts as itself.
    """
    if isinstance(cache, QuantizedCache):
        return cache.stored_bytes()
    if torch.is_tensor(cache):
        return _tensor_bytes(cache)
    total = 0
    for keys, values in _iter_cache_kv(cache):
        total += _tensor_bytes(keys) + _tensor_bytes(values)
    return total


def state_bytes_full(keys_shape: Sequence[int], dtype: torch.dtype,
                     values_shape: Sequence[int] | None = None) -> int:
    """Dense K+V reference size: ``(numel(keys) + numel(values)) * itemsize``.

    ``values_shape`` defaults to ``keys_shape`` (they are equal in a KV cache).
    """
    keys_numel = 1
    for dim in tuple(keys_shape):
        keys_numel *= int(dim)
    if values_shape is None:
        values_numel = keys_numel
    else:
        values_numel = 1
        for dim in tuple(values_shape):
            values_numel *= int(dim)
    return (keys_numel + values_numel) * dtype_nbytes(dtype)
