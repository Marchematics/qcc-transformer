"""Optional vLLM v1 registration hook.

This module deliberately keeps vLLM an optional dependency.  A version-specific
``AttentionBackend`` implementation can register itself with the stable vLLM
v1 registry without changing application code::

    from qcc_transformer.vllm_plugin import register_vllm_backend
    register_vllm_backend("my_package.qcc_backend.QCCAttentionBackend")

The hook only performs registry wiring; it does not pretend that the
dependency-free :class:`QCCVLLMState` is a complete vLLM ``AttentionImpl``.
Calling it without vLLM installed raises an actionable error.
"""

from __future__ import annotations

from typing import Any


def register_vllm_backend(class_path: str) -> Any:
    """Register ``class_path`` as vLLM's ``AttentionBackendEnum.CUSTOM``.

    vLLM's registry API is imported lazily so normal HF/CPU users do not need
    vLLM installed.  The target class must implement the vLLM version-specific
    ``AttentionBackend``/metadata builder contract; this function validates
    only that the registry call completed.
    """

    if not isinstance(class_path, str) or not class_path or "." not in class_path:
        raise ValueError("class_path must be a fully-qualified module.Class path")
    try:
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional vLLM
        raise RuntimeError(
            "vLLM v1 is not installed; install qcc-transformer[vllm] before "
            "registering an upstream backend"
        ) from exc
    register_backend(AttentionBackendEnum.CUSTOM, class_path)
    return AttentionBackendEnum.CUSTOM


__all__ = ["register_vllm_backend"]
