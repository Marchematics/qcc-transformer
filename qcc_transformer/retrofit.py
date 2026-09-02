"""Opt-in retrofit adapters for Hugging Face-style decoder attention.

The core :class:`~qcc_transformer.model.QCCForCausalLM` remains dependency
free.  This module adds a small adapter for decoder blocks whose attention
module exposes ``q_proj``, ``k_proj``, ``v_proj`` and ``o_proj`` (Llama/Qwen
style MHA).  Load a model first, call :func:`patch_hf_model`, then use the
model's normal forward/generation entry point.  The adapter keeps a bounded
QCC state per replaced layer and returns a cache handle so generation can keep
passing ``past_key_values`` without materialising historical K/V tensors.

Grouped-query attention is rejected until an explicit GQA reduction policy is
provided; silently repeating or averaging KV heads would invalidate a 99%
Full-KV quality comparison.  The integration is intentionally opt-in and
experimental: real-model quality must be measured with a matched unpatched
model and a task-appropriate checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn

from .model import QCCSelfAttention


@dataclass
class QCCCacheHandle:
    """Minimal cache protocol used by common HF generation loops.

    A handle contains no historical K/V tensors.  It points at one adapted
    attention layer and exposes the sequence length methods used by recent
    ``transformers`` cache utilities.  Older tuple-style generation code can
    pass the handle back unchanged as ``past_key_value``.
    """

    attention: QCCSelfAttention

    def get_seq_length(self, cache_position: Optional[int] = None) -> int:
        del cache_position
        return int(self.attention._seen_tokens)

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        del new_seq_length, layer_idx
        return int(self.attention._seen_tokens)

    def to_legacy_cache(self) -> tuple[tuple[None, None], ...]:
        """Return a shape-compatible empty legacy entry.

        The QCC adapter does not expose physical K/V tensors.  Consumers that
        require legacy tensors should use the adapter's own ``past_key_value``
        handle and must not concatenate this placeholder.
        """

        return ((None, None),)


class HFQCCAttention(nn.Module):
    """Wrap a Llama/Qwen-style attention module with bounded QCC state."""

    def __init__(
        self,
        base_attention: nn.Module,
        *,
        num_heads: int,
        window_size: int = 128,
        num_codes: int = 16,
        max_position_embeddings: int = 131_072,
        rope_theta: Optional[float] = None,
        use_triton: bool = True,
        archive_read_stride: int = 1,
        archive_query_cosine_threshold: Optional[float] = None,
        archive_lexical_landmark: bool = False,
    ) -> None:
        super().__init__()
        required = ("q_proj", "k_proj", "v_proj", "o_proj")
        missing = [name for name in required if not hasattr(base_attention, name)]
        if missing:
            raise TypeError(f"attention module is missing projections: {missing}")
        q_proj = getattr(base_attention, "q_proj")
        k_proj = getattr(base_attention, "k_proj")
        v_proj = getattr(base_attention, "v_proj")
        if not all(isinstance(proj, nn.Linear) for proj in (q_proj, k_proj, v_proj)):
            raise TypeError("QCC retrofit currently requires nn.Linear q/k/v projections")
        if q_proj.in_features != q_proj.out_features:
            raise ValueError("QCC retrofit requires square q projection")
        d_model = int(q_proj.in_features)
        if q_proj.out_features % num_heads:
            raise ValueError("hidden size must be divisible by num_heads")
        if k_proj.out_features != d_model or v_proj.out_features != d_model:
            raise ValueError(
                "QCC retrofit currently supports equal-width MHA projections; "
                "GQA/MQA requires an explicit KV-head policy"
            )
        self.base_attention = base_attention
        self.qcc = QCCSelfAttention(
            d_model,
            num_heads,
            window_size=window_size,
            num_codes=num_codes,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            use_triton=use_triton,
            archive_read_stride=archive_read_stride,
            archive_query_cosine_threshold=archive_query_cosine_threshold,
            archive_lexical_landmark=archive_lexical_landmark,
        )
        # Share the loaded HF projections/output projection; no second copy of
        # the large model weights is created.  The archive/gate are new trainable
        # parameters and should be calibrated or fine-tuned before quality
        # claims are made.
        self.qcc.q_proj = q_proj
        self.qcc.k_proj = k_proj
        self.qcc.v_proj = v_proj
        self.qcc.out_proj = getattr(base_attention, "o_proj")
        self.num_heads = num_heads
        self.d_model = d_model

    @staticmethod
    def _positions(
        position_ids: Optional[Tensor], batch: int, length: int, device: torch.device, start: int
    ) -> Tensor:
        if position_ids is None:
            return start + torch.arange(length, device=device, dtype=torch.long).view(1, -1).expand(batch, -1)
        if position_ids.ndim == 1:
            return position_ids.view(1, -1).expand(batch, -1)
        if position_ids.ndim != 2:
            raise ValueError("position_ids must have shape [sequence] or [batch, sequence]")
        return position_ids.to(device=device)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Any = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        del attention_mask, cache_position
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        batch, length, _ = hidden_states.shape
        # Transformers >=4.57 passes the shared Cache as ``past_key_values``
        # and decoder layers unpack only ``(attn_output, attn_weights)``;
        # older releases pass singular ``past_key_value`` and expect a third
        # present-cache item.  The internal QCC state is authoritative in both
        # cases, so the modern shared Cache need not contain physical K/V.
        modern_cache_call = "past_key_values" in kwargs
        cache = kwargs.get("past_key_values", past_key_value)
        seen = int(self.qcc._seen_tokens)
        reset = cache is None or seen == 0
        positions = self._positions(
            position_ids, batch, length, hidden_states.device, 0 if reset else seen
        )
        if self.training and not use_cache:
            # Expose a differentiable path for lightweight adapter
            # calibration/fine-tuning.  Serving remains on the no-grad
            # persistent-cache path below; callers must explicitly train with
            # ``use_cache=False`` so recurrent state is not reused across
            # optimizer steps.
            output = self.qcc(
                hidden_states,
                reset_state=reset,
                position_ids=positions,
            )
        elif length == 1 and not reset:
            output = self.qcc.step(
                hidden_states[:, 0],
                reset_cache=False,
                position_ids=positions[:, 0],
            ).unsqueeze(1)
        else:
            output = self.qcc.step_chunk(
                hidden_states,
                reset_cache=reset,
                position_ids=positions,
            )
        present = QCCCacheHandle(self.qcc) if use_cache else None
        # Attention weights are intentionally unavailable: QCC computes a
        # bounded approximation and cannot expose the exact historical matrix.
        del output_attentions
        if modern_cache_call:
            return output, None
        return output, None, present


def _module_parent(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def patch_hf_model(
    model: nn.Module,
    *,
    window_size: int = 128,
    num_codes: int = 16,
    max_position_embeddings: Optional[int] = None,
    rope_theta: Optional[float] = None,
    use_triton: bool = True,
    archive_read_stride: int = 1,
    archive_query_cosine_threshold: Optional[float] = None,
    archive_lexical_landmark: bool = False,
) -> list[str]:
    """Replace compatible HF attention modules and return their module paths.

    The model must be loaded before patching.  A ``ValueError`` is raised if a
    module advertises grouped-query attention; this keeps retrofit quality
    comparisons mathematically honest instead of silently changing head
    semantics.
    """

    config = getattr(model, "config", None)
    if max_position_embeddings is None:
        max_position_embeddings = int(
            getattr(config, "max_position_embeddings", 131_072)
        )
    if rope_theta is None:
        rope_theta = getattr(config, "rope_theta", None)
    candidates: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if all(hasattr(module, field) for field in ("q_proj", "k_proj", "v_proj", "o_proj")):
            candidates.append((name, module))
    if not candidates:
        raise ValueError(
            "no compatible HF attention modules found (expected q_proj/k_proj/v_proj/o_proj)"
        )
    replaced: list[str] = []
    for name, module in candidates:
        q_heads = int(
            getattr(module, "num_heads", getattr(config, "num_attention_heads", 0))
        )
        kv_heads = int(
            getattr(module, "num_key_value_heads", getattr(config, "num_key_value_heads", q_heads))
        )
        if q_heads <= 0:
            raise ValueError(f"cannot infer num_heads for {name}")
        if kv_heads != q_heads:
            raise ValueError(
                f"{name} uses GQA/MQA ({q_heads} query vs {kv_heads} KV heads); "
                "QCC retrofit requires an explicit KV-head reduction policy"
            )
        wrapper = HFQCCAttention(
            module,
            num_heads=q_heads,
            window_size=window_size,
            num_codes=num_codes,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            use_triton=use_triton,
            archive_read_stride=archive_read_stride,
            archive_query_cosine_threshold=archive_query_cosine_threshold,
            archive_lexical_landmark=archive_lexical_landmark,
        )
        parent, attr = _module_parent(model, name)
        setattr(parent, attr, wrapper)
        replaced.append(name)
    return replaced


def retrofit_adapter_state(model: nn.Module) -> dict[str, Tensor]:
    """Extract only QCC archive/gate parameters from a patched model."""

    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if ".qcc.archive." in name or ".qcc.gate." in name
    }
    if not state:
        raise ValueError("model has no QCC retrofit parameters; call patch_hf_model first")
    return state


def load_retrofit_adapter(
    model: nn.Module,
    checkpoint: str | Path,
    **patch_kwargs: Any,
) -> list[str]:
    """Patch a HF model and load a QCC-only adapter checkpoint."""

    replaced = patch_hf_model(model, **patch_kwargs)
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError("retrofit checkpoint must contain a state_dict mapping")
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [key for key in unexpected if ".qcc.archive." not in key and ".qcc.gate." not in key]
    missing = [
        key
        for key in missing
        if (".qcc.archive." in key and not key.endswith("archive.decay_rates"))
        or ".qcc.gate." in key
    ]
    if unexpected or missing:
        raise ValueError(f"adapter mismatch: missing={missing}, unexpected={unexpected}")
    return replaced


__all__ = [
    "HFQCCAttention",
    "QCCCacheHandle",
    "load_retrofit_adapter",
    "patch_hf_model",
    "retrofit_adapter_state",
]
