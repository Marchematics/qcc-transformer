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


@dataclass(frozen=True)
class FidelityReport:
    """Matched Full-KV-vs-retrofit quality metrics and gate decision."""

    mean_logit_cosine: float
    top1_agreement: float
    quality_gate: float = 0.99

    @property
    def passed(self) -> bool:
        return (
            self.mean_logit_cosine >= self.quality_gate
            and self.top1_agreement >= self.quality_gate
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean_logit_cosine": self.mean_logit_cosine,
            "top1_agreement": self.top1_agreement,
            "quality_gate": self.quality_gate,
            "fidelity_passed": self.passed,
        }


def compare_logits(reference: Tensor, candidate: Tensor, *, quality_gate: float = 0.99) -> FidelityReport:
    """Compute the repository's 99% Full-KV fidelity gate."""

    if reference.shape != candidate.shape or reference.ndim < 2:
        raise ValueError("reference and candidate logits must have the same rank-2+ shape")
    if not 0.0 <= quality_gate <= 1.0:
        raise ValueError("quality_gate must lie in [0, 1]")
    reference = reference.float().reshape(-1, reference.shape[-1])
    candidate = candidate.float().reshape(-1, candidate.shape[-1])
    cosine = torch.nn.functional.cosine_similarity(reference, candidate, dim=-1).mean()
    top1 = (reference.argmax(dim=-1) == candidate.argmax(dim=-1)).float().mean()
    return FidelityReport(float(cosine.item()), float(top1.item()), quality_gate)


class _RepeatKVProjection(nn.Module):
    """Expand grouped-query K/V projections to query-head width.

    This is an explicit opt-in policy.  Repeating each KV head preserves the
    usual HF GQA semantics (each query group shares one projected KV head)
    while allowing QCC's per-head archive to operate without inventing an
    unvalidated averaging rule.
    """

    def __init__(self, base: nn.Module, query_heads: int, kv_heads: int, head_dim: int) -> None:
        super().__init__()
        if query_heads % kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads for repeat policy")
        self.base = base
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.in_features = int(getattr(base, "in_features"))
        self.out_features = query_heads * head_dim
        self.repeat = query_heads // kv_heads

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[Tensor]:
        return getattr(self.base, "bias", None)

    def forward(self, hidden: Tensor) -> Tensor:
        projected = self.base(hidden)
        shape = (*projected.shape[:-1], self.kv_heads, self.head_dim)
        projected = projected.view(*shape)
        projected = projected.repeat_interleave(self.repeat, dim=-2)
        return projected.reshape(*projected.shape[:-2], self.out_features)


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
        kv_head_policy: str = "reject",
        kv_heads: Optional[int] = None,
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
        inferred_kv_heads = int(k_proj.out_features // (d_model // num_heads))
        if kv_heads is None:
            kv_heads = inferred_kv_heads
        if kv_head_policy not in {"reject", "repeat"}:
            raise ValueError("kv_head_policy must be 'reject' or 'repeat'")
        if k_proj.out_features != d_model or v_proj.out_features != d_model:
            if kv_head_policy != "repeat":
                raise ValueError(
                    "QCC retrofit currently supports equal-width MHA projections; "
                    "GQA/MQA requires kv_head_policy='repeat'"
                )
            if kv_heads <= 0 or num_heads % kv_heads:
                raise ValueError("repeat policy requires query_heads divisible by kv_heads")
            head_dim = d_model // num_heads
            if k_proj.out_features != kv_heads * head_dim or v_proj.out_features != kv_heads * head_dim:
                raise ValueError("K/V projection width is inconsistent with head counts")
            k_proj_for_qcc: nn.Module = _RepeatKVProjection(k_proj, num_heads, kv_heads, head_dim)
            v_proj_for_qcc: nn.Module = _RepeatKVProjection(v_proj, num_heads, kv_heads, head_dim)
        else:
            k_proj_for_qcc = k_proj
            v_proj_for_qcc = v_proj
            kv_heads = num_heads
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
        self.qcc.k_proj = k_proj_for_qcc
        self.qcc.v_proj = v_proj_for_qcc
        self.qcc.out_proj = getattr(base_attention, "o_proj")
        # HF checkpoints may be loaded in bf16/fp16 while the standalone QCC
        # module defaults to fp32.  The gate participates in the fused
        # projection path, so keep it in the backbone projection dtype;
        # recurrent archive accumulators remain fp32 for numerical stability.
        self.qcc.gate.to(device=q_proj.weight.device, dtype=q_proj.weight.dtype)
        # ``patch_hf_model`` is commonly called after ``from_pretrained(...).
        # to(device)``.  The newly-created archive therefore must be moved
        # explicitly; otherwise Triton receives CPU codebook pointers while
        # projected Q/K/V tensors live on CUDA.
        self.qcc.archive.to(device=q_proj.weight.device)
        self.num_heads = num_heads
        self.d_model = d_model
        self.kv_head_policy = kv_head_policy
        self.kv_heads = int(kv_heads)
        self._qcc_retrofit = True

    def __getattr__(self, name: str) -> Any:
        # HF decoder layers sometimes inspect implementation-specific fields
        # such as ``layer_idx`` or ``rotary_emb`` on their attention module.
        # Delegate those reads to the original module so patching is drop-in.
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = self.__dict__.get("_modules", {}).get("base_attention")
            if base is not None and hasattr(base, name):
                return getattr(base, name)
            raise

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
    kv_head_policy: str = "reject",
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
        if getattr(module, "_qcc_retrofit", False) or getattr(module, "_qcc_retrofit_original", False):
            continue
        q_heads = int(
            getattr(module, "num_heads", getattr(config, "num_attention_heads", 0))
        )
        kv_heads = int(
            getattr(module, "num_key_value_heads", getattr(config, "num_key_value_heads", q_heads))
        )
        if q_heads <= 0:
            raise ValueError(f"cannot infer num_heads for {name}")
        if kv_heads != q_heads and kv_head_policy == "reject":
            raise ValueError(
                f"{name} uses GQA/MQA ({q_heads} query vs {kv_heads} KV heads); "
                "QCC retrofit requires kv_head_policy='repeat'"
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
            kv_head_policy=kv_head_policy,
            kv_heads=kv_heads,
        )
        # The original module remains registered below the wrapper so its
        # projection parameters are shared.  Mark it to make patching
        # idempotent instead of recursively wrapping ``base_attention``.
        setattr(module, "_qcc_retrofit_original", True)
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


def save_retrofit_adapter(model: nn.Module, checkpoint: str | Path, **metadata: Any) -> Path:
    """Save QCC-only parameters plus a reproducibility manifest.

    The base HF model is never copied into the adapter.  Metadata is JSON-like
    and can include the model id/revision, patch arguments, and calibration
    split so a deployment script can verify that the adapter is being applied
    to the intended backbone.
    """

    path = Path(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": retrofit_adapter_state(model),
        "format": "qcc-retrofit-v1",
        "metadata": metadata,
    }
    torch.save(payload, path)
    return path


def reset_hf_qcc_cache(model: nn.Module, *, batch_size: int = 1) -> int:
    """Reset all adapted attention states before a new serving request.

    Hugging Face's ``generate`` normally owns its cache object, whereas QCC
    keeps historical state inside each patched attention module.  This helper
    gives serving wrappers an explicit request boundary and returns the number
    of layers reset.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    count = 0
    for module in model.modules():
        if isinstance(module, HFQCCAttention):
            module.qcc.reset_cache(batch_size, device=module.qcc.q_proj.weight.device)
            count += 1
    if count == 0:
        raise ValueError("model has no HFQCCAttention modules; call patch_hf_model first")
    return count


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
    "FidelityReport",
    "HFQCCAttention",
    "QCCCacheHandle",
    "load_retrofit_adapter",
    "patch_hf_model",
    "retrofit_adapter_state",
    "save_retrofit_adapter",
    "compare_logits",
    "reset_hf_qcc_cache",
]
