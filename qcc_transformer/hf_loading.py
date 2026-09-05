"""Small optional loader for real Hugging Face evaluation runs.

The default path is unchanged: load the checkpoint with its requested dtype and
move it to the requested device.  ``load_in_4bit`` is an explicit opt-in for
large real checkpoints on limited GPUs; it uses Transformers' bitsandbytes
integration and never changes the model architecture or adapter state.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import torch
from torch import nn


def _compute_dtype(dtype: torch.dtype | str | None) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return dtype
    return torch.float16


def _ensure_remote_code_compat() -> None:
    """Provide type-only symbols expected by older Phi remote-code snapshots.

    Phi-3/Phi-4 checkpoints may ship ``modeling_phi3.py`` revisions that import
    ``LossKwargs`` from ``transformers.utils`` even though that symbol is not
    exported by the installed Transformers release.  The symbol is consumed only
    by the model's forward annotation, so a mapping alias is sufficient and keeps
    the checkpoint usable without pinning the host's Transformers version.
    """

    try:
        import transformers.utils as transformers_utils
    except ImportError:  # pragma: no cover - handled by the caller
        return
    if not hasattr(transformers_utils, "LossKwargs"):
        transformers_utils.LossKwargs = dict[str, Any]

    # Transformers 5.x removed the legacy-cache conversion helpers while
    # older Phi remote-code snapshots still call both methods.  The current
    # DynamicCache constructor accepts the legacy tuples directly, so a small
    # compatibility shim is sufficient and keeps the model code untouched.
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:  # pragma: no cover - handled by the caller
        return
    if not hasattr(DynamicCache, "from_legacy_cache"):
        @classmethod
        def from_legacy_cache(cls, past_key_values):
            return cls() if past_key_values is None else cls(past_key_values)

        DynamicCache.from_legacy_cache = from_legacy_cache
    if not hasattr(DynamicCache, "to_legacy_cache"):
        def to_legacy_cache(self):
            return tuple((entry[0], entry[1]) for entry in self)

        DynamicCache.to_legacy_cache = to_legacy_cache
    if not hasattr(DynamicCache, "get_usable_length"):
        def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
            del new_seq_length
            return self.get_seq_length(layer_idx)

        DynamicCache.get_usable_length = get_usable_length


def load_hf_causal_lm(
    model_id: str,
    *,
    dtype: torch.dtype | str | None = None,
    device: torch.device | str | None = None,
    trust_remote_code: bool = False,
    load_in_4bit: bool = False,
    device_map: str | dict[str, Any] | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Load a real HF causal LM with an optional 4-bit weight path.

    Quantized models are returned in the placement selected by Transformers and
    must not be followed by ``model.to(...)``.  The normal loader keeps the
    historical placement behavior so existing callers remain unchanged.
    """

    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install qcc-transformer[hf] to load a HF checkpoint") from exc

    if trust_remote_code:
        _ensure_remote_code_compat()

    load_kwargs = dict(kwargs)
    load_kwargs["trust_remote_code"] = trust_remote_code

    # Qwen2.5-VL is a causal language model wrapped in a multimodal container,
    # so it is intentionally absent from AutoModelForCausalLM's mapping in
    # several Transformers releases.  Route that architecture to its native
    # class while retaining the usual AutoModel path for text-only models.
    model_cls: Any = AutoModelForCausalLM
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        architectures = tuple(getattr(config, "architectures", ()) or ())
        if any("Qwen2_5_VL" in name for name in architectures):
            from transformers import Qwen2_5_VLForConditionalGeneration

            model_cls = Qwen2_5_VLForConditionalGeneration
    except (ImportError, OSError, TypeError, ValueError):
        # AutoModelForCausalLM below provides the historical error message for
        # ordinary checkpoints whose config cannot be inspected independently.
        model_cls = AutoModelForCausalLM

    if load_in_4bit:
        if device is None or torch.device(device).type != "cuda":
            raise ValueError("load_in_4bit requires a CUDA device")
        if importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError(
                "load_in_4bit requires bitsandbytes; install it in the evaluation environment"
            )
        from transformers import BitsAndBytesConfig

        compute_dtype = _compute_dtype(dtype)
        load_kwargs["torch_dtype"] = compute_dtype
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = device_map or "auto"
        return model_cls.from_pretrained(model_id, **load_kwargs).eval()

    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    model = model_cls.from_pretrained(model_id, **load_kwargs)
    if device is not None:
        model = model.to(device)
    return model.eval()


def model_input_device(model: nn.Module, fallback: torch.device | str) -> torch.device:
    """Return the device on which a HF model expects input IDs."""

    fallback_device = torch.device(fallback)
    try:
        parameter = next(model.parameters())
    except StopIteration:
        return fallback_device
    if parameter.device.type != "meta":
        return parameter.device
    return fallback_device


__all__ = ["load_hf_causal_lm", "model_input_device"]
