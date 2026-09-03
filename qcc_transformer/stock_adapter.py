"""Adapter conversion/loading helpers for stock vLLM QCC deployment."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Mapping

import torch
from torch import Tensor

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_ARCHIVE_MARKER = ".qcc.archive."
_REQUIRED = {
    "codes",
    "mix_logits",
    "exact_bank.set_codes",
    "admission.key_weight",
    "admission.value_weight",
    "admission.bias",
    "exact_mix_logits",
}
_HEAD_PARAMETER_KEYS = {
    "codes",
    "mix_logits",
    "exact_bank.set_codes",
    "admission.key_weight",
    "admission.value_weight",
    "admission.bias",
    "exact_mix_logits",
}


def layer_index_from_name(name: str) -> int:
    match = _LAYER_RE.search(name)
    if match is None:
        raise ValueError(f"cannot infer transformer layer index from {name!r}")
    return int(match.group(1))


def extract_archive_parameters(
    state_dict: Mapping[str, Tensor],
    layer_name: str,
    *,
    strict: bool = True,
) -> dict[str, Tensor]:
    """Map HF adapter names for one transformer layer to HybridQCCArchive names."""
    index = layer_index_from_name(layer_name)
    layer_token = f".layers.{index}."
    extracted: dict[str, Tensor] = {}
    for name, tensor in state_dict.items():
        normalized = f".{name}" if not name.startswith(".") else name
        if layer_token not in normalized or _ARCHIVE_MARKER not in normalized:
            continue
        suffix = normalized.split(_ARCHIVE_MARKER, 1)[1]
        if suffix in extracted:
            raise ValueError(f"duplicate stock adapter key for layer {index}: {suffix}")
        extracted[suffix] = tensor
    if strict:
        missing = sorted(_REQUIRED - extracted.keys())
        if missing:
            raise ValueError(
                f"hybrid stock adapter layer {index} is incomplete; missing={missing}"
            )
    return extracted


def checkpoint_state_dict(payload: object) -> dict[str, Tensor]:
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not all(isinstance(k, str) for k in payload):
        raise ValueError("QCC adapter checkpoint must contain a string-keyed state_dict")
    tensors = {k: v for k, v in payload.items() if isinstance(v, Tensor)}
    if not tensors:
        raise ValueError("QCC adapter checkpoint contains no tensors")
    return tensors


@lru_cache(maxsize=4)
def load_checkpoint_state_dict(path: str) -> dict[str, Tensor]:
    payload = torch.load(Path(path), map_location="cpu")
    return checkpoint_state_dict(payload)


def load_archive_parameters(
    archive: torch.nn.Module,
    state_dict: Mapping[str, Tensor],
    layer_name: str,
    *,
    tensor_parallel_rank: int = 0,
    tensor_parallel_size: int = 1,
) -> None:
    """Load learned archive tensors, slicing global heads for tensor parallelism.

    HF adapters are saved once with the complete model head dimension.  A stock
    vLLM worker owns only its tensor-parallel head slice, so loading the global
    tensors unchanged would either fail on shape or silently use the wrong heads.
    Mutable request state is never part of this operation.
    """
    if tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be positive")
    if not 0 <= tensor_parallel_rank < tensor_parallel_size:
        raise ValueError("tensor_parallel_rank must lie within tensor_parallel_size")
    layer_state = extract_archive_parameters(state_dict, layer_name, strict=True)
    local_heads = int(getattr(archive, "num_heads", 0))
    if local_heads <= 0:
        raise ValueError("archive must expose a positive num_heads")
    if tensor_parallel_size > 1:
        global_heads = local_heads * tensor_parallel_size
        start = tensor_parallel_rank * local_heads
        end = start + local_heads
        sharded: dict[str, Tensor] = {}
        for name, tensor in layer_state.items():
            if name not in _HEAD_PARAMETER_KEYS:
                sharded[name] = tensor
                continue
            if tensor.ndim == 0:
                raise ValueError(f"head parameter {name} is scalar")
            if tensor.shape[0] == local_heads:
                # Accept an already-local adapter for explicit per-rank
                # deployments; the normal global adapter is sliced below.
                sharded[name] = tensor
            elif tensor.shape[0] == global_heads:
                sharded[name] = tensor[start:end]
            else:
                raise ValueError(
                    f"adapter parameter {name} has {tensor.shape[0]} heads; "
                    f"expected {local_heads} or global {global_heads}"
                )
        layer_state = sharded
    missing, unexpected = archive.load_state_dict(layer_state, strict=False)
    unexpected = [name for name in unexpected if name not in {"decay_rates"}]
    if unexpected:
        raise ValueError(f"unexpected stock archive parameters: {unexpected}")
    del missing


__all__ = [
    "checkpoint_state_dict",
    "extract_archive_parameters",
    "layer_index_from_name",
    "load_archive_parameters",
    "load_checkpoint_state_dict",
]
