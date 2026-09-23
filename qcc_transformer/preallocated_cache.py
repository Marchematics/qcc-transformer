"""A KV cache with fixed-size buffers, written in place.

`DynamicCache` grows with `torch.cat`, which is fine while the cache is small and
fatal exactly where this project needs it: at 1M tokens on Qwen2.5-0.5B the cache is
12.0 GiB, so appending a chunk needs the old 12.0 GiB *and* the new 12.0 GiB alive at
once.  Measured on a free 24 GiB A10G, a 1M exact prefill dies there -
`torch.OutOfMemoryError` inside `DynamicLayer.update`, 22.65 GiB allocated, with the
weights (0.95 GiB) and no selection structures involved.  The same doubling happens on
every decode step against a full cache.

`PreallocatedCache` instead allocates one buffer per (layer, kv-head) up front and
writes each chunk into its slice, returning the written prefix as a view.  Appending
becomes O(chunk) instead of O(sequence), the transient doubling disappears, and the
footprint is exactly the state the model needs - `capacity * kv_heads * head_dim * 2
* 2` bytes per layer, no growth, no copy.  It is the same accounting the bounded
compile path advertises, applied to the cache the *prefill* builds.

The returned keys/values are views of the buffer, so a caller that compiles the cache
can gather into new tensors (what the bounded path does) without copying the buffer.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


class PreallocatedLayer(DynamicLayer):
    """One layer's key/value buffers plus the number of slots written.

    ``update`` writes in place and returns the *written prefix*, which is what the
    attention operator must see: the rest of the buffer is uninitialised and must never
    be attended to.
    """

    def __init__(self, capacity: int, num_heads: int, head_dim: int,
                 dtype: torch.dtype = torch.bfloat16, device="cpu"):
        self.capacity = int(capacity)
        self.keys = torch.zeros(1, num_heads, self.capacity, head_dim,
                                dtype=dtype, device=device)
        self.values = torch.zeros_like(self.keys)
        self.dtype, self.device = dtype, device
        self.is_initialized = True
        self.length = 0
        self.overflowed = False
        # the buffer's address identifies "still backed by the fixed buffer": the
        # compile path *replaces* ``layer.keys`` with a gathered tensor, and attention
        # hands back views of the buffer (same storage), so an address check separates
        # the two without the caller having to say which it did
        self.buffer_ptr = self.keys.data_ptr()

    @property
    def written(self) -> int:
        """Usable slots: what was written, capped by the extent of the live tensor."""
        return min(self.length, int(self.keys.shape[-2]))

    def _grow(self, key_states: torch.Tensor, value_states: torch.Tensor):
        """Append like `DynamicCache`, for states the fixed buffer cannot hold.

        Growth starts from the *written* prefix, never from the whole buffer: unwritten
        slots are uninitialised, so concatenating them would put garbage inside the
        attention window (measured: a 16-token chunk against an 8-slot buffer reported
        24 keys before this).
        """
        written = self.written
        self.keys = torch.cat([self.keys[:, :, :written], key_states], dim=-2)
        self.values = torch.cat([self.values[:, :, :written], value_states], dim=-2)
        self.length = int(self.keys.shape[-2])
        return self.keys, self.values

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               *args, **kwargs):
        if self.keys.data_ptr() != self.buffer_ptr:
            # the caller compiled this layer down to selected slots (a new tensor), so
            # the fixed buffer no longer describes the state
            return self._grow(key_states, value_states)
        new = int(key_states.shape[-2])
        end = self.length + new
        if end > self.capacity:
            # Correctness first: a cache that dropped slots would corrupt attention
            # without an error, and one that raised would not be a drop-in for
            # `DynamicCache` (a caller that only prefills does not know the generation
            # length yet).  Grow by concatenation, and record that the sizing was
            # wrong so callers and benchmarks can see it instead of guessing.
            self.overflowed = True
            return self._grow(key_states, value_states)
        self.keys[:, :, self.length:end] = key_states
        self.values[:, :, self.length:end] = value_states
        self.length = end
        return self.keys[:, :, :end], self.values[:, :, :end]

    def reset(self) -> None:
        """Return to empty, reusing the buffer when it still backs this layer."""
        if self.keys.data_ptr() == self.buffer_ptr:
            self.length = 0
        else:
            # compiled down or grown: drop the tensors (a re-prefill needs a clear
            # cache, and appending to the old ones would attend to stale slots)
            self.keys = self.keys[:, :, :0].contiguous()
            self.values = self.values[:, :, :0].contiguous()
            self.length = 0
        self.overflowed = False

    # `DynamicLayer` derives these from the tensor shape, which for a fixed buffer is
    # the *capacity*: left alone, the mask for a 16-token chunk over a 64-slot buffer
    # comes out 80 wide and attention dies on a shape mismatch (measured before the
    # override).  Anything that asks the cache about its length must see usable slots -
    # what has been written, and what the live tensor actually holds after a compile.
    def get_seq_length(self) -> int:
        return self.written

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.written + query_length, 0


class PreallocatedCache(DynamicCache):
    """`DynamicCache` with fixed buffers; appends are O(chunk), not O(sequence)."""

    def __init__(self, capacity: int, num_layers: int, num_heads: int, head_dim: int,
                 dtype: torch.dtype = torch.bfloat16, device="cpu"):
        super().__init__()
        self.capacity = int(capacity)
        self.layers = [PreallocatedLayer(self.capacity, num_heads, head_dim,
                                         dtype=dtype, device=device)
                       for _ in range(int(num_layers))]
        # layers are created here, so the lazy-replication path must not run
        self.layer_class_to_replicate = None

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        while len(self.layers) <= layer_idx:                    # pragma: no cover
            template = self.layers[-1]
            self.layers.append(PreallocatedLayer(
                self.capacity, template.keys.shape[1], template.keys.shape[-1],
                dtype=template.dtype, device=template.device))
        return self.layers[layer_idx].update(key_states, value_states, *args, **kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not self.layers:                                     # pragma: no cover
            return 0
        return max(layer.written for layer in self.layers)

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    @property
    def overflowed(self) -> bool:
        """Whether any layer outgrew its buffer (i.e. the cache was undersized)."""
        return any(layer.overflowed for layer in self.layers)

    @property
    def nbytes(self) -> int:
        """Bytes held by the buffers, independent of how much has been written."""
        return sum(layer.keys.numel() * layer.keys.element_size()
                   + layer.values.numel() * layer.values.element_size()
                   for layer in self.layers)


def preallocated_cache_for(model, capacity: int, device=None,
                           dtype: torch.dtype | None = None) -> PreallocatedCache:
    """Size a cache for `model` to hold exactly `capacity` tokens.

    Reads the KV geometry from the config (grouped-query models store far fewer KV
    heads than query heads, which is what makes 1M affordable at all on this class of
    card) and defaults to the model's own parameter dtype and device.
    """
    config = model.config
    head_dim = int(getattr(config, "head_dim", 0)
                   or config.hidden_size // config.num_attention_heads)
    num_heads = int(getattr(config, "num_key_value_heads", None)
                    or config.num_attention_heads)
    try:
        parameter = next(model.parameters())
        default_dtype, default_device = parameter.dtype, parameter.device
    except StopIteration:                                       # pragma: no cover
        default_dtype, default_device = torch.bfloat16, "cpu"
    return PreallocatedCache(
        capacity=capacity, num_layers=int(config.num_hidden_layers),
        num_heads=num_heads, head_dim=head_dim,
        dtype=dtype or default_dtype, device=device or default_device)
