#!/usr/bin/env python3
"""Matched real-HF PG-19 perplexity for Full-KV and QCC.

Input is JSONL exported from the official PG-19 *test* split, one document per row as
``{"text": ...}``. The evaluator streams each document in chunks while preserving
causal state. It counts every next-token prediction, including chunk boundaries.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcc_transformer.hybrid_archive import load_hybrid_retrofit_adapter
from qcc_transformer.hf_loading import load_hf_causal_lm, model_input_device
from qcc_transformer.production_profile import enable_qkv_only_deployment_profile
from qcc_transformer.retrofit import reset_hf_qcc_cache


def native_context(config) -> int | None:
    values = [
        getattr(config, name, None)
        for name in ("max_position_embeddings", "n_positions", "max_sequence_length")
    ]
    values = [int(value) for value in values if isinstance(value, int) and value > 0]
    return max(values) if values else None


def length_bucket(tokens: int) -> str:
    if tokens < 8_192:
        return "0-8K"
    if tokens < 32_768:
        return "8-32K"
    if tokens < 128_000:
        return "32-128K"
    return "128K+"


def load_documents(path: Path, limit: int | None = None) -> list[str]:
    documents: list[str] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str) or not text:
                raise ValueError("PG-19 JSONL rows must contain non-empty text")
            documents.append(text)
            if limit is not None and len(documents) >= limit:
                break
    if not documents:
        raise ValueError("PG-19 input contains no documents")
    return documents


def token_nll(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(
        F.cross_entropy(logits.float(), targets, reduction="sum").item()
    )


@torch.inference_mode()
def document_nll(model, input_ids: torch.Tensor, *, chunk_tokens: int, qcc: bool) -> tuple[float, int]:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must be [1, tokens]")
    if input_ids.shape[1] < 2:
        return 0.0, 0
    if qcc:
        reset_hf_qcc_cache(model, batch_size=1)
    past = None
    previous_next_logits: torch.Tensor | None = None
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, input_ids.shape[1], chunk_tokens):
        chunk = input_ids[:, start : start + chunk_tokens]
        positions = torch.arange(
            start, start + chunk.shape[1], device=chunk.device, dtype=torch.long
        ).view(1, -1)
        outputs = model(
            input_ids=chunk,
            position_ids=positions,
            past_key_values=past,
            use_cache=True,
        )
        logits = outputs.logits
        # First token in every noninitial chunk is predicted by the previous
        # chunk's final logit; never drop boundary losses.
        if start > 0:
            if previous_next_logits is None:
                raise RuntimeError("missing boundary logits")
            total_nll += token_nll(previous_next_logits, chunk[:, 0])
            total_tokens += 1
        if chunk.shape[1] > 1:
            total_nll += token_nll(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                chunk[:, 1:].reshape(-1),
            )
            total_tokens += chunk.shape[1] - 1
        previous_next_logits = logits[:, -1]
        past = getattr(outputs, "past_key_values", None)
        if past is None and start + chunk.shape[1] < input_ids.shape[1] and not qcc:
            raise RuntimeError("model did not return a cache required for matched streaming PG-19")
    return total_nll, total_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pg19-test-jsonl", type=Path, required=True)
    parser.add_argument("--mode", choices=("fullkv", "qcc"), required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true", help="load the real checkpoint through bitsandbytes NF4")
    parser.add_argument("--chunk-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--num-codes", type=int, default=64)
    parser.add_argument("--exact-num-sets", type=int, default=128)
    parser.add_argument("--exact-ways", type=int, default=4)
    parser.add_argument("--exact-probe-sets", type=int, default=None)
    parser.add_argument("--archive-mix", type=float, default=0.125)
    parser.add_argument("--kv-head-policy", choices=("reject", "repeat"), default="repeat")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--official-source", default="deepmind/pg19:test")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--min-native-context", type=int, default=None,
        help="require the checkpoint to natively declare at least this many tokens",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if args.mode == "qcc" and args.adapter is None:
        raise ValueError("--adapter is required for qcc mode")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("install qcc-transformer[hf]") from exc
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    common = {"trust_remote_code": args.trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **common)
    requested_device = torch.device(args.device)
    model = load_hf_causal_lm(
        args.model,
        dtype=dtype,
        device=requested_device,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
    )
    model_device = model_input_device(model, requested_device)
    native_context_tokens = native_context(model.config)
    if native_context_tokens is None:
        raise RuntimeError("model does not declare a native context length")
    if args.min_native_context is not None and native_context_tokens < args.min_native_context:
        raise RuntimeError(
            f"model native context {native_context_tokens} is below requested "
            f"{args.min_native_context}"
        )
    patched_layers: list[str] = []
    if args.mode == "qcc":
        patched_layers = load_hybrid_retrofit_adapter(
            model,
            args.adapter,
            hybrid_kwargs={
                "exact_num_sets": args.exact_num_sets,
                "exact_ways": args.exact_ways,
                "exact_probe_sets": args.exact_probe_sets,
            },
            window_size=args.window_size,
            num_codes=args.num_codes,
            archive_position_invariant=True,
            kv_head_policy=args.kv_head_policy,
        )
        enable_qkv_only_deployment_profile(model, archive_mix=args.archive_mix)

    documents = load_documents(args.pg19_test_jsonl, args.limit)
    total_nll = 0.0
    total_predicted = 0
    per_document: list[dict[str, Any]] = []
    bucket_totals: dict[str, dict[str, float]] = {}
    for index, text in enumerate(documents):
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model_device)
        if ids.shape[1] > native_context_tokens:
            raise RuntimeError(
                f"PG-19 document {index} has {ids.shape[1]} tokens above native "
                f"context {native_context_tokens}; do not truncate official evidence"
            )
        try:
            nll, predicted = document_nll(
                model, ids, chunk_tokens=args.chunk_tokens, qcc=args.mode == "qcc"
            )
            total_nll += nll
            total_predicted += predicted
            bucket = bucket_totals.setdefault(
                length_bucket(int(ids.shape[1])), {"nll": 0.0, "tokens": 0.0}
            )
            bucket["nll"] += nll
            bucket["tokens"] += predicted
            per_document.append({
                "document": index,
                "input_tokens": int(ids.shape[1]),
                "predicted_tokens": predicted,
                "nll": nll,
            })
        except Exception as exc:
            per_document.append({
                "document": index,
                "input_tokens": int(ids.shape[1]),
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
    if total_predicted <= 0:
        raise RuntimeError("PG-19 produced no scored next-token targets")
    mean_nll = total_nll / total_predicted
    perplexity = math.exp(mean_nll)
    bucket_scores = {
        name: 1.0 / math.exp(values["nll"] / values["tokens"])
        for name, values in sorted(bucket_totals.items())
        if values["tokens"] > 0
    }
    report = {
        "schema": "qcc-pg19-v1",
        "benchmark": "pg19",
        "mode": args.mode,
        "model_id": args.model,
        "run_id": args.run_id,
        "matched_full_kv": False,
        "real_model": True,
        "synthetic": False,
        "official": True,
        "full_suite": args.limit is None,
        "qcc_only": args.mode == "qcc",
        "official_source": args.official_source,
        "split": "test",
        "native_context_tokens": native_context_tokens,
        "metric": "perplexity",
        "metric_direction": "lower_is_better",
        "perplexity": perplexity,
        "mean_nll": mean_nll,
        # Gate-normalized higher-is-better score. QCC/Full ratio = FullPPL/QCCPPL.
        "quality_score": 1.0 / perplexity,
        "bucket_scores": bucket_scores,
        "documents": len(documents),
        "predicted_tokens": total_predicted,
        "chunk_tokens": args.chunk_tokens,
        "patched_layers": patched_layers,
        "per_document": per_document,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: report[k] for k in (
        "mode", "model_id", "perplexity", "mean_nll", "quality_score",
        "documents", "predicted_tokens"
    )}, indent=2))


if __name__ == "__main__":
    main()
