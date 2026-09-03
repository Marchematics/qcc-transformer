#!/usr/bin/env python3
"""Generate matched HF LongBench predictions and invoke THUDM's official evaluator.

This runner intentionally does not reimplement LongBench metrics. It reads prompt/maxlen
configuration and executes ``eval.py`` from a user-provided THUDM/LongBench checkout.
The official dataset JSONL files must be supplied separately, one ``<dataset>.jsonl``
per task with the original ``context/input/answers/all_classes/length`` fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcc_transformer.hybrid_archive import load_hybrid_retrofit_adapter
from qcc_transformer.hf_loading import load_hf_causal_lm, model_input_device
from qcc_transformer.production_profile import enable_qkv_only_deployment_profile
from qcc_transformer.retrofit import reset_hf_qcc_cache


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def native_context(config) -> int | None:
    values = [
        getattr(config, name, None)
        for name in ("max_position_embeddings", "n_positions", "max_sequence_length")
    ]
    values = [int(value) for value in values if isinstance(value, int) and value > 0]
    return max(values) if values else None


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        required = ("context", "input", "answers", "all_classes")
        if not isinstance(row, dict) or any(field not in row for field in required):
            raise ValueError(f"{path} is not an original LongBench-style JSONL file")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--longbench-root", type=Path, required=True,
                        help="THUDM/LongBench checkout containing LongBench/eval.py")
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="directory with official <dataset>.jsonl files")
    parser.add_argument("--mode", choices=("fullkv", "qcc"), required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--label", required=True,
                        help="safe output label used under official pred/<label>/")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true", help="load the real checkpoint through bitsandbytes NF4")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--num-codes", type=int, default=64)
    parser.add_argument("--exact-num-sets", type=int, default=128)
    parser.add_argument("--exact-ways", type=int, default=4)
    parser.add_argument("--exact-probe-sets", type=int, default=None)
    parser.add_argument("--archive-mix", type=float, default=0.125)
    parser.add_argument("--kv-head-policy", choices=("reject", "repeat"), default="repeat")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "qcc" and args.adapter is None:
        raise ValueError("--adapter is required for qcc mode")
    if any(char in args.label for char in "/\\") or not args.label:
        raise ValueError("label must be a simple directory name")

    official_root = args.longbench_root / "LongBench"
    prompt_path = official_root / "config" / "dataset2prompt.json"
    maxlen_path = official_root / "config" / "dataset2maxlen.json"
    eval_path = official_root / "eval.py"
    for path in (prompt_path, maxlen_path, eval_path):
        if not path.exists():
            raise FileNotFoundError(path)
    prompts: dict[str, str] = read_json(prompt_path)
    maxlens: dict[str, int] = read_json(maxlen_path)
    datasets = list(prompts.keys())
    if set(datasets) != set(maxlens):
        raise RuntimeError("official LongBench prompt/maxlen configs disagree")

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
    max_context = native_context(model.config)
    if max_context is None:
        raise RuntimeError("model does not declare a native context length")
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
            max_position_embeddings=max_context,
            archive_position_invariant=True,
            kv_head_policy=args.kv_head_policy,
        )
        enable_qkv_only_deployment_profile(model, archive_mix=args.archive_mix)

    prediction_dir = official_root / "pred" / args.label
    prediction_dir.mkdir(parents=True, exist_ok=True)
    generated_rows = 0
    for dataset in datasets:
        source = args.dataset_dir / f"{dataset}.jsonl"
        if not source.exists():
            raise FileNotFoundError(
                f"missing official LongBench task file {source}; full-suite evidence cannot omit tasks"
            )
        rows = load_rows(source)
        if args.limit_per_dataset is not None:
            rows = rows[: args.limit_per_dataset]
        output_path = prediction_dir / f"{dataset}.jsonl"
        with output_path.open("w", buffering=1) as out:
            for row in rows:
                prompt = prompts[dataset].format(**row)
                if args.apply_chat_template:
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                encoded = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
                input_ids = encoded["input_ids"]
                if input_ids.shape[1] > max_context:
                    raise RuntimeError(
                        f"{dataset} prompt has {input_ids.shape[1]} tokens > native context {max_context}; "
                        "do not truncate official evidence"
                    )
                input_ids = input_ids.to(model_device)
                if args.mode == "qcc":
                    reset_hf_qcc_cache(model, batch_size=1)
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        max_new_tokens=int(maxlens[dataset]),
                        do_sample=False,
                        use_cache=True,
                    )
                continuation = generated[0, input_ids.shape[1]:]
                prediction = tokenizer.decode(continuation, skip_special_tokens=True)
                official_row = {
                    "pred": prediction,
                    "answers": row["answers"],
                    "all_classes": row["all_classes"],
                    "length": row.get("length", int(input_ids.shape[1])),
                }
                out.write(json.dumps(official_row, ensure_ascii=False) + "\n")
                generated_rows += 1

    # The only metric implementation used is THUDM/LongBench's own eval.py.
    proc = subprocess.run(
        [sys.executable, "eval.py", "--model", args.label],
        cwd=official_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "official LongBench evaluator failed:\nSTDOUT:\n"
            + proc.stdout + "\nSTDERR:\n" + proc.stderr
        )
    result_path = prediction_dir / "result.json"
    scores = read_json(result_path)
    if set(scores) != set(datasets):
        missing = sorted(set(datasets) - set(scores))
        raise RuntimeError(f"official evaluator did not score the full suite; missing={missing}")
    scalar_scores = []
    for dataset, value in scores.items():
        if isinstance(value, (int, float)):
            scalar_scores.append(float(value))
        else:
            raise RuntimeError(
                "LongBench-E bucketed output is not accepted by this full LongBench gate runner"
            )
    macro = statistics.fmean(scalar_scores)
    report = {
        "schema": "qcc-longbench-v1",
        "benchmark": "longbench",
        "mode": args.mode,
        "model_id": args.model,
        "real_model": True,
        "synthetic": False,
        "official": True,
        "official_evaluator": str(eval_path),
        "official_eval_stdout": proc.stdout,
        "native_context_tokens": max_context,
        "full_suite": True,
        "datasets": datasets,
        "dataset_scores": scores,
        "quality_score": macro,
        "macro_average": macro,
        "generated_rows": generated_rows,
        "prediction_dir": str(prediction_dir),
        "patched_layers": patched_layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    print(json.dumps({
        "mode": args.mode,
        "model_id": args.model,
        "quality_score": macro,
        "datasets": len(datasets),
        "generated_rows": generated_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
