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
) -> None:
    """Load only learned HybridQCCArchive tensors; mutable request state is untouched."""
    layer_state = extract_archive_parameters(state_dict, layer_name, strict=True)
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
