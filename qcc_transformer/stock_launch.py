"""Minimal-adoption helpers for stock vLLM QCC deployment."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .vllm_stock import (
    QCCStockVLLMConfig,
    configure_stock_vllm_environment,
    register_stock_vllm_backend,
)


def prepare_stock_vllm(
    config: QCCStockVLLMConfig,
    *,
    adapter_path: str,
) -> dict[str, Any]:
    """Configure workers/register QCC and return the one vLLM argument required.

    Application model code stays unchanged. Programmatic users merge the returned
    mapping into ``vllm.LLM(...)``; server users use :func:`stock_vllm_cli_args`.
    """
    path = Path(adapter_path)
    if not path.exists():
        raise FileNotFoundError(path)
    configure_stock_vllm_environment(config, adapter_path=str(path.resolve()))
    register_stock_vllm_backend()
    return {"attention_config": {"backend": "CUSTOM"}}


def stock_vllm_cli_args(
    config: QCCStockVLLMConfig,
    *,
    adapter_path: str,
) -> list[str]:
    """Configure inherited env/registration and return stock vLLM CLI arguments."""
    prepare_stock_vllm(config, adapter_path=adapter_path)
    return ["--attention-config.backend", "CUSTOM"]


def stock_launch_manifest(
    config: QCCStockVLLMConfig,
    *,
    adapter_path: str,
) -> dict[str, Any]:
    return {
        "qcc_config": asdict(config),
        "adapter_path": str(Path(adapter_path)),
        "vllm_programmatic": {"attention_config": {"backend": "CUSTOM"}},
        "vllm_cli": ["--attention-config.backend", "CUSTOM"],
        "application_model_code_changes": 0,
    }


__all__ = ["prepare_stock_vllm", "stock_launch_manifest", "stock_vllm_cli_args"]
