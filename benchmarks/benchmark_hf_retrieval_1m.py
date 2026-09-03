#!/usr/bin/env python3
"""Run the locked 1M retrieval manifest on Full-KV or learned-admission QCC.

The manifest is generated before model execution and shared verbatim between modes.
Every trial is padded to exactly the requested token context with neutral filler; needle
records are inserted at pre-registered random depths. Failures/OOM count as misses.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcc_transformer.hybrid_archive import load_hybrid_retrofit_adapter
from qcc_transformer.production_profile import enable_qkv_only_deployment_profile
from qcc_transformer.retrofit import reset_hf_qcc_cache

_CODE_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")
_FILLER = (
    "This is neutral archival background material. It contains no access code for any "
    "named registry, cluster, vault, archive, repository, station, ledger, or warehouse. "
)
_HEADER = (
    "You are reading a long collection of records. Some records are deliberately very "
    "similar. Use the exact entity name in the final query and return only its eight "
    "digit access code. Ignore codes belonging to other entities.\n"
)


def load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    lines = [line for line in content.decode().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("manifest requires header plus trials")
    header = json.loads(lines[0])
    trials = [json.loads(line) for line in lines[1:]]
    if header.get("schema") != "qcc-real-retrieval-manifest-v1":
        raise ValueError("unsupported retrieval manifest schema")
    if header.get("protocol_locked") is not True:
        raise ValueError("retrieval manifest is not protocol-locked")
    if len(trials) != int(header.get("trials", -1)) or len(trials) < 1000:
        raise ValueError("strict retrieval manifest must contain at least 1000 trials")
    return header, trials, digest


def _tile(pattern: torch.Tensor, count: int) -> torch.Tensor:
    if count < 0:
        raise ValueError("negative filler budget")
    if count == 0:
        return pattern[:0]
    repeats = math.ceil(count / pattern.numel())
    return pattern.repeat(repeats)[:count]


def build_trial_ids(tokenizer, trial: dict[str, Any], context_tokens: int) -> tuple[torch.Tensor, list[float]]:
    encode = lambda text: tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    header = encode(_HEADER)
    filler_pattern = encode(_FILLER)
    if filler_pattern.numel() == 0:
        raise RuntimeError("tokenizer produced empty filler")
    query = encode(
        "\nFinal query: What is the access code for " + trial["target_entity"] + "? Return only the eight digits.\nAnswer:"
    )
    blocks = []
    for record in trial["records"]:
        tokens = encode("\n" + record["text"] + "\n")
        blocks.append((float(record["depth"]), tokens, record["kind"]))
    blocks.sort(key=lambda item: item[0])
    fixed = header.numel() + query.numel() + sum(tokens.numel() for _, tokens, _ in blocks)
    filler_budget = context_tokens - fixed
    if filler_budget < 0:
        raise ValueError("context length is too small for retrieval protocol records")

    pieces = [header]
    filler_used = 0
    actual_depths: list[float] = []
    current_tokens = header.numel()
    for depth, tokens, _ in blocks:
        desired_filler = min(filler_budget, max(filler_used, round(depth * filler_budget)))
        gap = desired_filler - filler_used
        if gap:
            pieces.append(_tile(filler_pattern, gap))
            filler_used += gap
            current_tokens += gap
        actual_depths.append(current_tokens / context_tokens)
        pieces.append(tokens)
        current_tokens += tokens.numel()
    if filler_used < filler_budget:
        pieces.append(_tile(filler_pattern, filler_budget - filler_used))
    pieces.append(query)
    ids = torch.cat(pieces)
    if ids.numel() != context_tokens:
        raise RuntimeError(f"constructed {ids.numel()} tokens, expected {context_tokens}")
    return ids.unsqueeze(0), actual_depths


def _native_context(model) -> int | None:
    config = model.config
    candidates = []
    for name in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            candidates.append(value)
    return max(candidates) if candidates else None


def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def normalize_prediction(text: str) -> str | None:
    matches = _CODE_RE.findall(text)
    return matches[0] if matches else None


def depth_bucket(depth: float) -> str:
    if depth < 0.25:
        return "0-25%"
    if depth < 0.50:
        return "25-50%"
    if depth < 0.75:
        return "50-75%"
    return "75-100%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("fullkv", "qcc"), required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--num-codes", type=int, default=64)
    parser.add_argument("--exact-num-sets", type=int, default=32)
    parser.add_argument("--exact-ways", type=int, default=4)
    parser.add_argument("--archive-mix", type=float, default=0.125)
    parser.add_argument("--kv-head-policy", choices=("reject", "repeat"), default="repeat")
    parser.add_argument("--require-native-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "qcc" and args.adapter is None:
        raise ValueError("--adapter is required for qcc mode")

    header, trials, manifest_hash = load_manifest(args.manifest)
    context_tokens = int(header["context_tokens"])
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("install qcc-transformer[hf]") from exc

    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    common = {"trust_remote_code": args.trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **common)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, **common).to(device).eval()
    native_context = _native_context(model)
    if args.require_native_context and (native_context is None or native_context < context_tokens):
        raise RuntimeError(
            f"model native context {native_context} is below locked {context_tokens}; "
            "use a checkpoint that natively declares the target context"
        )
    patched_layers = []
    if args.mode == "qcc":
        patched_layers = load_hybrid_retrofit_adapter(
            model,
            args.adapter,
            hybrid_kwargs={"exact_num_sets": args.exact_num_sets, "exact_ways": args.exact_ways},
            window_size=args.window_size,
            num_codes=args.num_codes,
            max_position_embeddings=context_tokens,
            archive_position_invariant=True,
            kv_head_policy=args.kv_head_policy,
        )
        enable_qkv_only_deployment_profile(model, archive_mix=args.archive_mix)

    completed: dict[int, dict[str, Any]] = {}
    if args.resume and args.output_jsonl.exists():
        for line in args.output_jsonl.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed[int(row["trial"])] = row
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stream = args.output_jsonl.open("a" if args.resume else "w", buffering=1)

    try:
        for spec in trials:
            index = int(spec["trial"])
            if index in completed:
                continue
            row: dict[str, Any] = {
                "trial": index,
                "expected": spec["expected"],
                "target_entity": spec["target_entity"],
            }
            try:
                ids, actual_depths = build_trial_ids(tokenizer, spec, context_tokens)
                target_record = next(
                    record for record in spec["records"]
                    if record["kind"] == "needle" and record["entity"] == spec["target_entity"]
                )
                # actual_depths follows records sorted by requested depth.
                sorted_records = sorted(spec["records"], key=lambda item: float(item["depth"]))
                target_sorted_index = sorted_records.index(target_record)
                target_depth = actual_depths[target_sorted_index]
                row["target_depth"] = target_depth
                row["depth_bucket"] = depth_bucket(target_depth)
                encoded = ids.to(device)
                if args.mode == "qcc":
                    reset_hf_qcc_cache(model, batch_size=1)
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=encoded,
                        attention_mask=torch.ones_like(encoded),
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                continuation = generated[0, context_tokens:]
                text = tokenizer.decode(continuation, skip_special_tokens=True)
                prediction = normalize_prediction(text)
                row.update(
                    prediction=prediction,
                    raw_prediction=text,
                    correct=prediction == spec["expected"],
                    input_tokens=context_tokens,
                )
            except Exception as exc:
                row.update(correct=False, error=f"{type(exc).__name__}: {exc}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            completed[index] = row
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    finally:
        stream.close()

    ordered = [completed[int(spec["trial"])] for spec in trials if int(spec["trial"]) in completed]
    if len(ordered) != len(trials):
        raise RuntimeError("run is incomplete; resume before producing strict summary")
    correct = sum(bool(row.get("correct")) for row in ordered)
    buckets: dict[str, dict[str, int]] = {}
    for row in ordered:
        bucket = row.get("depth_bucket", "execution-failure")
        item = buckets.setdefault(bucket, {"trials": 0, "correct": 0})
        item["trials"] += 1
        item["correct"] += int(bool(row.get("correct")))
    bucket_summary = {
        name: {**counts, "success_rate": counts["correct"] / counts["trials"]}
        for name, counts in sorted(buckets.items())
    }
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    summary = {
        "schema": "qcc-real-retrieval-result-v1",
        "mode": args.mode,
        "model_id": args.model,
        "parameter_count": parameter_count,
        "pretrained": True,
        "real_checkpoint": True,
        "real_model": True,
        "synthetic": False,
        "oracle_admission": False if args.mode == "qcc" else None,
        "official": False,
        "protocol_locked": True,
        "manifest_sha256": manifest_hash,
        "context_tokens": context_tokens,
        "native_context_tokens": native_context,
        "native_context_required": args.require_native_context,
        "trials": len(ordered),
        "correct": correct,
        "success_rate": correct / len(ordered),
        "random_depth": True,
        "multi_needle": True,
        "semantic_distractor": True,
        "depth_buckets": bucket_summary,
        "patched_layers": patched_layers,
        "output_jsonl": str(args.output_jsonl),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
