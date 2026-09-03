#!/usr/bin/env python3
"""Launch a stock vLLM server with the QCC backend configured out of process.

The application still supplies only the normal model and OpenAI-compatible server
arguments.  This helper derives attention geometry from the pretrained checkpoint,
sets the worker-inherited packed-state configuration, and then replaces itself with
``vllm serve`` so worker PIDs and signals retain normal vLLM behavior.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from qcc_transformer.vllm_stock import (
    QCCStockVLLMConfig,
    STOCK_ADAPTER_ENV,
    STOCK_CONFIG_ENV,
)


def infer_attention_geometry(config: Any) -> tuple[int, int]:
    hidden = int(getattr(config, "hidden_size", getattr(config, "n_embd", 0)))
    heads = int(
        getattr(config, "num_attention_heads", getattr(config, "n_head", 0))
    )
    head_dim = int(getattr(config, "head_dim", 0))
    if head_dim <= 0 and heads > 0 and hidden % heads == 0:
        head_dim = hidden // heads
    if hidden <= 0 or heads <= 0 or head_dim <= 0 or hidden != heads * head_dim:
        raise ValueError("checkpoint does not expose compatible attention geometry")
    return heads, head_dim


def native_context(config: Any) -> int | None:
    values = [
        getattr(config, name, None)
        for name in ("max_position_embeddings", "n_positions", "max_sequence_length")
    ]
    values = [int(value) for value in values if isinstance(value, int) and value > 0]
    return max(values) if values else None


def _has_option(arguments: list[str], name: str) -> bool:
    return any(argument == name or argument.startswith(name + "=") for argument in arguments)


def _option_value(arguments: list[str], name: str) -> str | None:
    """Return a value for either ``--name value`` or ``--name=value``."""

    for index, argument in enumerate(arguments):
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def _trust_remote_code(arguments: list[str]) -> bool:
    """Mirror the server's opt-in remote-code flag while loading AutoConfig."""

    return _has_option(arguments, "--trust-remote-code")


def _tensor_parallel_size(arguments: list[str]) -> int:
    value = _option_value(arguments, "--tensor-parallel-size")
    if value is None:
        return 1
    try:
        size = int(value)
    except ValueError as exc:
        raise ValueError("--tensor-parallel-size must be an integer") from exc
    if size <= 0:
        raise ValueError("--tensor-parallel-size must be positive")
    return size


def build_launch(
    *,
    model: str,
    adapter: Path,
    config: Any,
    context_length: int | None,
    window_size: int,
    num_codes: int,
    num_scales: int,
    exact_num_sets: int,
    exact_ways: int,
    exact_probe_sets: int | None,
    local_dtype: str,
    passthrough: list[str],
) -> tuple[list[str], dict[str, str], QCCStockVLLMConfig]:
    if not adapter.is_file():
        raise FileNotFoundError(adapter)
    query_heads, head_dim = infer_attention_geometry(config)
    tensor_parallel_size = _tensor_parallel_size(passthrough)
    if query_heads % tensor_parallel_size:
        raise ValueError(
            "checkpoint query heads must divide --tensor-parallel-size for stock QCC"
        )
    target_context = context_length or native_context(config)
    if target_context is None or target_context <= 0:
        raise ValueError("context-length is required when the checkpoint has no native context")
    if target_context < window_size:
        raise ValueError("context-length must cover window-size")
    if local_dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("local-dtype must be float16, bfloat16, or float32")
    local_bytes = {"float16": 2, "bfloat16": 2, "float32": 4}[local_dtype]
    qcc_config = QCCStockVLLMConfig(
        num_heads=query_heads // tensor_parallel_size,
        head_dim=head_dim,
        window_size=window_size,
        num_codes=num_codes,
        num_scales=num_scales,
        exact_num_sets=exact_num_sets,
        exact_ways=exact_ways,
        exact_probe_sets=exact_probe_sets,
        max_position_embeddings=target_context,
        local_element_bytes=local_bytes,
        tensor_parallel_size=tensor_parallel_size,
    )
    configured_max_len = _option_value(passthrough, "--max-model-len")
    if configured_max_len is not None:
        try:
            configured_max_len_int = int(configured_max_len)
        except ValueError as exc:
            raise ValueError("--max-model-len must be an integer") from exc
        if configured_max_len_int != target_context:
            raise ValueError(
                "--max-model-len must equal the QCC context-length so the server "
                "and packed-state decay schedule describe the same target"
            )
    configured_backend = _option_value(passthrough, "--attention-config.backend")
    if configured_backend is not None and configured_backend.upper() != "CUSTOM":
        raise ValueError("QCC launcher requires --attention-config.backend CUSTOM")
    command = ["vllm", "serve", model, *passthrough]
    if not _has_option(passthrough, "--attention-config.backend"):
        command.extend(["--attention-config.backend", "CUSTOM"])
    if not _has_option(passthrough, "--max-model-len"):
        command.extend(["--max-model-len", str(target_context)])
    environment = {
        STOCK_CONFIG_ENV: qcc_config.to_json(),
        STOCK_ADAPTER_ENV: str(adapter.resolve()),
    }
    return command, environment, qcc_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--num-codes", type=int, default=16)
    parser.add_argument("--num-scales", type=int, default=4)
    parser.add_argument("--exact-num-sets", type=int, default=128)
    parser.add_argument("--exact-ways", type=int, default=4)
    parser.add_argument("--exact-probe-sets", type=int, default=None)
    parser.add_argument("--local-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument("--print-only", action="store_true")
    args, passthrough = parser.parse_known_args()
    if args.context_length is not None and args.context_length <= 0:
        raise ValueError("context-length must be positive")
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise SystemExit("install qcc-transformer[hf] to inspect the checkpoint config") from exc
    model_config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=_trust_remote_code(passthrough),
    )
    command, qcc_environment, qcc_config = build_launch(
        model=args.model,
        adapter=args.adapter,
        config=model_config,
        context_length=args.context_length,
        window_size=args.window_size,
        num_codes=args.num_codes,
        num_scales=args.num_scales,
        exact_num_sets=args.exact_num_sets,
        exact_ways=args.exact_ways,
        exact_probe_sets=args.exact_probe_sets,
        local_dtype=args.local_dtype,
        passthrough=passthrough,
    )
    command[0] = args.vllm_bin
    environment = os.environ.copy()
    environment.update(qcc_environment)
    manifest = {
        "command": command,
        "qcc_config": json.loads(qcc_config.to_json()),
        "adapter": qcc_environment[STOCK_ADAPTER_ENV],
    }
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if args.print_only:
        return
    os.execvpe(args.vllm_bin, command, environment)


if __name__ == "__main__":
    main()
