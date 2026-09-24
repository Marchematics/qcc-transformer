#!/usr/bin/env python3
"""LongBench retention: bounded decode cache vs matched Full-KV, real documents.

The RULER evidence in ``docs/REPORT.md`` is
synthetic (needle-in-a-haystack and variable tracking).  This harness runs the
same shipped law on **LongBench** (THUDM, 2023): real long-document QA,
summarisation and retrieval, scored with the official metrics, so the retention
number does not depend on synthetic needle strings.

Structure (mirrors ``benchmarks/benchmark_retention_multimodel.py``):

* for every record, build the official prompt and run each arm --
  ``full`` = exact chunked Full-KV prefill (:func:`prefill_capture`),
  ``bounded`` = :func:`compile_bounded_cache` --
* decode greedily with the usual max_new/stop-at-EOS loop and absolute
  position ids, so a shorter-than-prompt cache still decodes correctly,
* score the decoded text with the official LongBench metric of the task,
* report per-task ``mean_score`` for every arm plus a matched ``retention``
  ratio ``bounded / full`` over the records the Full-KV arm scored above zero.

Truncation (explicit): a prompt longer than ``--max-input-tokens`` is truncated
the way the official LongBench harness does it -- keep the first half and the
last half of the *token ids* -- because both ends carry instruction (the head)
and question (the tail).  Token ids are kept instead of the official
detokenise/re-encode round trip.  ``--truncation skip`` drops such records
instead; either way the choice is recorded per record and in the JSON header
(``truncation_policy``).

``--tasks`` defaults to the English non-code LongBench subset.  ``vcsum`` is in
that default list, but it is skipped with an explicit reason unless ``jieba`` is
installed: its official metric is Chinese ROUGE-L
(:mod:`longbench_metrics`); the skip is recorded in ``skipped_tasks`` and never
silently scored as zero.

Example (one 1B model, one GPU)::

    python benchmarks/benchmark_retention_longbench.py \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --tasks narrativeqa qasper hotpotqa 2wikimqa gov_report multi_news \\
                triviaqa samsum vcsum passage_retrieval_en \\
        --limit 20 --budget 4096 --lex-cap 512 --hops 6 \\
        --out artifacts/longbench-retention-llama32-1b.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
for entry in (str(REPO_ROOT), str(BENCH_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import longbench_data as lbd
import longbench_metrics as lbm
import os
from qcc_transformer.hf_loading import _ensure_remote_code_compat, load_hf_causal_lm
from qcc_transformer.retention import (RetentionConfig, compile_bounded_cache,
                                       fixed_rope_length, prefill_capture)

try:  # the selection rules the published-family arms share with the RULER runner
    import benchmark_bounded_decode_frontier as frontier
except ImportError:  # pragma: no cover - benchmarks/ is not on sys.path
    frontier = None

SCHEMA = "qcc-longbench-retention-v1"
TRUNCATION_POLICIES = ("head_tail", "skip")

#: `bounded` is the shipped compile; the rest are the published families as
#: selection rules (see `benchmarks/benchmark_bounded_decode_frontier.py`)
POLICIES = ("full", "bounded", "obs_mean", "obs_max", "obs_last", "quest",
            "pyramid", "h2o", "tova",
            # anchor channel held fixed, filler signal varied
            "mean_lex", "h2o_lex", "blend_lex")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default=None,
                        help="display label; defaults to the model directory name")
    parser.add_argument("--tasks", nargs="*", default=list(lbd.DEFAULT_TASKS))
    parser.add_argument("--limit", type=int, default=None, help="records per task")
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--lex-cap", type=int, default=512)
    parser.add_argument("--hops", type=int, default=6, help="chain_hops of the law")
    parser.add_argument("--obs", type=int, default=64, help="observation_window")
    parser.add_argument("--nsink", type=int, default=4, help="attention_sinks")
    parser.add_argument("--pool", type=int, default=7)
    parser.add_argument("--dilate", type=int, default=9)
    parser.add_argument("--scoring", default="last", choices=("last", "mean", "max"))
    parser.add_argument("--max-new", type=int, default=None,
                        help="greedy decode length; default: the official "
                             "dataset2maxlen.json value of each task (128/512/32/64)")
    parser.add_argument("--blend-alpha", type=float, default=0.5,
                        help="weight of the accumulated-attention ranking in `blend_lex`")
    parser.add_argument("--max-input-tokens", type=int, default=32768,
                        help="prompts above this are truncated or skipped "
                             "(see --truncation); default 32768")
    parser.add_argument("--truncation", default="head_tail", choices=TRUNCATION_POLICIES,
                        help="head_tail: keep the first and last half of the token ids "
                             "(the official LongBench policy); skip: drop the record")
    parser.add_argument("--prefill-chunk", type=int, default=8192)
    parser.add_argument("--key-chunk", type=int, default=4096)
    parser.add_argument("--arms", nargs="*", default=["full", "bounded"],
                        choices=POLICIES,
                        help="full, the shipped bounded compile (`bounded`), or one "
                             "of the published-family selection rules implemented "
                             "in benchmark_bounded_decode_frontier.py")
    parser.add_argument("--cache-dir", default=None,
                        help=f"LongBench cache; default ${lbd.ENV_CACHE_DIR} or "
                             f"{lbd.DEFAULT_CACHE_DIR}")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device", default="cuda", help="cuda (default) or cpu")
    parser.add_argument("--attn-impl", default="sdpa",
                        help="attention implementation to request at load time")
    parser.add_argument("--tiny-random-model", action="store_true",
                        help="smoke test only: build a small randomly initialised "
                             "Llama from --model's tokenizer instead of loading its "
                             "weights (CPU, no GPU); never evidence")
    parser.add_argument("--progress", default=None,
                        help="progress JSONL; default <out>.progress.jsonl")
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def forward_kwargs_of(model):
    """Which optional forward arguments this checkpoint accepts."""
    from qcc_transformer.retention import forward_accepts

    return ("cache_position",) if forward_accepts(model, "cache_position") else ()


def eos_token_ids(model) -> set[int]:
    ids: set[int] = set()
    for source in (model.config.eos_token_id,
                   getattr(getattr(model, "generation_config", None), "eos_token_id", None)):
        if source is None:
            continue
        ids |= set(source) if isinstance(source, (list, tuple, set)) else {source}
    return ids


def truncate_ids(input_ids: torch.Tensor, max_tokens: int):
    """Official LongBench middle truncation, on token ids.

    ``prompt = decode(ids[:max//2]) + decode(ids[-max//2:])`` in the official
    harness; keeping the ids is the same cut without the re-encoding round trip.
    Returns ``(ids, truncated)``.
    """
    length = int(input_ids.shape[1])
    if max_tokens is None or length <= max_tokens:
        return input_ids, False
    half = max_tokens // 2
    return torch.cat([input_ids[:, :half], input_ids[:, length - (max_tokens - half):]],
                     dim=1), True


@torch.no_grad()
def greedy(model, cache, first, length, max_new, eos_ids, forward_kwargs=("cache_position",)):
    """The harness decode loop: absolute positions, greedy, stop at EOS."""
    kept = cache.get_seq_length()
    mask = torch.ones(1, kept + 1, device=first.device, dtype=torch.long)
    generated, current, start = [], first, time.time()
    for step in range(max_new):
        call = dict(past_key_values=cache, attention_mask=mask, use_cache=True,
                    position_ids=torch.tensor([[length + step]], device=first.device))
        if "cache_position" in forward_kwargs:
            call["cache_position"] = torch.arange(kept + step, kept + step + 1,
                                                  device=first.device)
        out = model(current, **call)
        token = out.logits[:, -1:].argmax(-1)
        generated.append(int(token))
        mask = torch.cat([mask, torch.ones(1, 1, device=first.device, dtype=torch.long)],
                         dim=-1)
        current = token
        if int(token) in eos_ids:
            break
    return generated, time.time() - start


def load_model(args):
    if args.trust_remote_code:
        _ensure_remote_code_compat()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=args.trust_remote_code)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    if args.tiny_random_model:
        # Smoke-test path: same code path, random tiny weights, no checkpoint load.
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=64,
                             intermediate_size=128, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=4096)
        model = LlamaForCausalLM(config)
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            model = model.to(args.device)
        return model.eval(), tokenizer
    model = load_hf_causal_lm(args.model, dtype=dtype, device=args.device,
                              trust_remote_code=args.trust_remote_code,
                              load_in_4bit=args.load_4bit,
                              attn_implementation=args.attn_impl).eval()
    return model, tokenizer


def screen_tasks(tasks, mapping):
    """Split the request into runnable tasks and tasks with a missing metric."""
    runnable, skipped = [], {}
    for task in tasks:
        if task not in lbd.ALL_TASKS:
            raise SystemExit(f"unknown LongBench task {task!r}; known: {list(lbd.ALL_TASKS)}")
        reason = lbm.unavailable_reason(task, mapping)
        if reason is None:
            runnable.append(task)
        else:
            skipped[task] = reason
    return runnable, skipped


def main(argv=None):
    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)

    mapping = lbm.load_dataset2metric()
    tasks, skipped_tasks = screen_tasks(args.tasks, mapping)
    if not tasks:
        raise SystemExit(f"no runnable task in {args.tasks}; skipped: {skipped_tasks}")
    for task, reason in skipped_tasks.items():
        print(f"[skip] {task}: {reason}", flush=True)

    manifest = lbd.ensure_dataset(args.cache_dir, tasks=tasks, verbose=True)
    templates = lbd.load_prompt_templates(args.cache_dir)
    maxlens = lbd.load_max_lengths(args.cache_dir)
    records = lbd.load_records(tasks, args.cache_dir, limit=args.limit,
                               auto_download=False, verbose=True)

    model, tokenizer = load_model(args)
    config = RetentionConfig(
        budget=args.budget, observation_window=args.obs, attention_sinks=args.nsink,
        pool=args.pool, dilate=args.dilate, lex_cap=args.lex_cap,
        chain_hops=args.hops, prefill_chunk=args.prefill_chunk,
        key_chunk=args.key_chunk, scoring=args.scoring, blend_alpha=args.blend_alpha)
    eos_ids = eos_token_ids(model)
    kwargs_names = forward_kwargs_of(model)
    print(f"[model] {args.model} device={device} records={len(records)} "
          f"arms={args.arms} budget={args.budget}+{args.lex_cap} "
          f"truncation={args.truncation}<={args.max_input_tokens} "
          f"forward_kwargs={kwargs_names}", flush=True)

    progress_path = Path(args.progress) if args.progress else Path(f"{args.out}.progress.jsonl")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress = progress_path.open("w", encoding="utf-8", buffering=1)

    results, skipped_records = [], []
    metric_errors = 0
    try:
        for index, record in enumerate(records):
            task = record["dataset"]
            prompt = lbd.build_prompt(record, templates)
            ids = tokenizer(prompt, return_tensors="pt",
                            add_special_tokens=False).input_ids
            before = int(ids.shape[1])
            if args.max_input_tokens is not None and before > args.max_input_tokens:
                if args.truncation == "skip":
                    skipped_records.append({"row": index, "task": task, "id": record.get("_id"),
                                            "prompt_tokens": before,
                                            "reason": f"prompt_tokens>{args.max_input_tokens}"})
                    print(f"[{index:>4d}/{len(records)}] {task:<20} L={before:>6d} "
                          f"SKIPPED (over --max-input-tokens)", flush=True)
                    continue
                ids, _ = truncate_ids(ids, args.max_input_tokens)
            ids = ids.to(device)
            length = int(ids.shape[1])
            max_new = (args.max_new if args.max_new is not None
                       else int(maxlens.get(task, 128)))

            with fixed_rope_length(model, length) as pinned:
                for arm in args.arms:
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    start = time.time()
                    if arm == "full":
                        cache, logits, captured = prefill_capture(
                            model, ids, config.observation_window, config.prefill_chunk)
                        mass = top1 = None
                    elif arm == "bounded":
                        cache, logits = compile_bounded_cache(model, ids, config,
                                                              tokenizer=tokenizer)
                    else:
                        # a published-family arm shares the exact prefill and the
                        # observation-window capture, then selects with its own rule
                        if frontier is None:
                            raise SystemExit("published-family arms need benchmarks/ "
                                             "on sys.path")
                        if arm in ("h2o", "tova", "h2o_lex", "blend_lex"):
                            cache, logits, captured, mass, top1 = frontier.prefill_accumulate(
                                model, ids, config.observation_window,
                                config.prefill_chunk, config.key_chunk)
                        else:
                            cache, logits, captured = prefill_capture(
                                model, ids, config.observation_window,
                                config.prefill_chunk)
                            mass = top1 = None
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    prefill_seconds = time.time() - start
                    kept = int(cache.get_seq_length())
                    if arm not in ("full", "bounded"):
                        if arm == "blend_lex":
                            scores = frontier.blend_scores(
                                frontier.obs_scores(model, captured, cache, length,
                                                    config.observation_window, "last",
                                                    config.key_chunk),
                                mass, config.blend_alpha)
                        elif arm in ("h2o", "h2o_lex"):
                            scores = mass
                        elif arm == "tova":
                            scores = top1
                        else:
                            mode = {"quest": "last", "pyramid": "mean",
                                    "mean_lex": "mean"}.get(
                                arm, arm.split("_")[-1] if arm.startswith("obs_") else "last")
                            scores = frontier.obs_scores(
                                model, captured, cache, length,
                                config.observation_window, mode, config.key_chunk)
                        originals = [(layer.keys, layer.values) for layer in cache.layers]
                        # the *_lex arms also need the question's rare strings
                        lexical = (frontier.question_lexical_positions(
                            tokenizer, prompt, config.observation_window, hops=config.chain_hops)
                            if arm.endswith("_lex") else [])
                        idxs = frontier.build_idxs(
                            arm, scores, cache, _PlainRecord(lexical), config.budget,
                            config.attention_sinks, config.recent_window, length,
                            config.pool, ids.device, config.dilate, config.lex_cap)
                        for layer, (ok, ov), slots in zip(cache.layers, originals, idxs):
                            heads = ok.shape[1]
                            gather = slots.unsqueeze(0).unsqueeze(-1).expand(
                                1, heads, slots.shape[1], ok.shape[-1])
                            layer.keys = ok.gather(2, gather).contiguous()
                            layer.values = ov.gather(2, gather).contiguous()
                        kept = int(cache.get_seq_length())
                        kept_mean = round(
                            sum(int(slots.shape[1]) for slots in idxs) / len(idxs), 1)
                    else:
                        kept_mean = float(kept)
                    generated, decode_seconds = greedy(
                        model, cache, logits[:, -1:].argmax(-1), length, max_new,
                        eos_ids, kwargs_names)
                    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

                    entry = {
                        "row": index, "task": task, "id": record.get("_id"),
                        "official_length": record.get("length"),
                        "prompt_tokens": length, "prompt_tokens_before": before,
                        "truncated": bool(before > length),
                        "arm": arm, "kept_slots": kept, "kept_slots_mean": kept_mean,
                        "slots_per_token": round(kept / length, 4),
                        "max_new": max_new, "generated": len(generated),
                        "prediction": text,
                    }
                    try:
                        entry["score"] = round(float(lbm.score_record(task, text, record,
                                                                      mapping)), 6)
                    except Exception as exc:  # keep the run alive, surface the record
                        entry["score"] = 0.0
                        entry["metric_error"] = f"{type(exc).__name__}: {exc}"
                        metric_errors += 1
                    entry.update({
                        "prefill_s": round(prefill_seconds, 2),
                        "decode_s": round(decode_seconds, 2),
                        "pinned_rope": bool(pinned),
                        "peak_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 2)
                                     if device.type == "cuda" else None),
                    })
                    results.append(entry)
                    progress.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    print(f"[{index:>4d}/{len(records)}] {task:<20} L={length:>6d} "
                          f"{arm:<7} pinned={int(pinned)} kept={kept:>6d} "
                          f"score={entry['score']:.3f} pred={text[:32]!r}", flush=True)
                    del cache
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    except BaseException:
        # an interrupted run still produces a usable (partial) artifact
        progress.close()
        write_report(args, manifest, mapping, results, skipped_tasks, skipped_records,
                     metric_errors, progress_path, partial=True)
        raise
    progress.close()

    summary = write_report(args, manifest, mapping, results, skipped_tasks, skipped_records,
                           metric_errors, progress_path, partial=False)
    print("\n" + json.dumps(summary, indent=1))
    return summary


class _PlainRecord:
    """Record stand-in carrying the question's rare-string positions."""

    def __init__(self, lexical_positions=None):
        self.lexical_positions = lexical_positions or []


def summarize(results, args):
    """Per-task mean score per arm and the matched bounded/full retention."""
    by_arm = defaultdict(list)
    for entry in results:
        by_arm[entry["arm"]].append(entry)
    full = {entry["row"]: entry for entry in by_arm.get("full", [])}
    summary = {"tasks": {}, "aggregate": {}}
    for task in sorted({entry["task"] for entry in results}):
        rows = [entry for entry in results if entry["task"] == task]
        arms: dict[str, dict] = {}
        for arm in args.arms:
            arm_rows = [entry for entry in rows if entry["arm"] == arm]
            if not arm_rows:
                continue
            arms[arm] = {
                "records": len(arm_rows),
                "mean_score": round(sum(e["score"] for e in arm_rows) / len(arm_rows), 4),
                "mean_kept_slots": round(sum(e["kept_slots"] for e in arm_rows) / len(arm_rows), 1),
                "mean_prompt_tokens": round(sum(e["prompt_tokens"] for e in arm_rows) / len(arm_rows), 1),
                "mean_generated": round(sum(e["generated"] for e in arm_rows) / len(arm_rows), 1),
                "positive_records": sum(1 for e in arm_rows if e["score"] > 0),
            }
        if "full" in arms and "bounded" in arms:
            ratios = []
            for entry in rows:
                if entry["arm"] != "bounded":
                    continue
                reference = full.get(entry["row"])
                if reference and reference["score"] > 0:
                    ratios.append(entry["score"] / reference["score"])
            arms["retention"] = {
                "matched_records": len(ratios),
                "skipped_zero_full": arms["full"]["records"] - len(ratios),
                "mean": round(sum(ratios) / len(ratios), 4) if ratios else None,
                "worst": round(min(ratios), 4) if ratios else None,
            }
        summary["tasks"][task] = arms
    matched = [(task, data["retention"]) for task, data in summary["tasks"].items()
               if data.get("retention", {}).get("mean") is not None]
    if matched:
        full_means = [data["full"]["mean_score"] for data in summary["tasks"].values()
                      if "full" in data]
        bounded_means = [data["bounded"]["mean_score"] for data in summary["tasks"].values()
                         if "bounded" in data]
        summary["aggregate"] = {
            "tasks": len(matched),
            "mean_retention": round(sum(r["mean"] for _t, r in matched) / len(matched), 4),
            "worst_task_retention": round(min(r["mean"] for _t, r in matched), 4),
            "worst_record_retention": round(min(r["worst"] for _t, r in matched), 4),
            "matched_records": sum(r["matched_records"] for _t, r in matched),
            "macro_mean_full_score": round(sum(full_means) / len(full_means), 4),
            "macro_mean_bounded_score": round(sum(bounded_means) / len(bounded_means), 4),
        }
    return summary


def write_report(args, manifest, mapping, results, skipped_tasks, skipped_records,
                 metric_errors, progress_path, partial):
    summary = summarize(results, args)
    report = {
        "schema": SCHEMA,
        "partial": partial,
        "model": args.model,
        "label": (("tiny-random-" if args.tiny_random_model else "")
                  + (args.label or Path(args.model).name)),
        "tiny_random_model": bool(args.tiny_random_model),
        "evidence": not args.tiny_random_model,
        "benchmark": "longbench",
        "dataset": {
            "repo": lbd.HF_REPO_ID,
            "cache_dir": manifest["cache_dir"],
            "tasks_requested": list(args.tasks),
            "tasks_run": sorted({entry["task"] for entry in results}),
            "records_run": len({entry["row"] for entry in results}),
            "records_skipped": len(skipped_records),
            "record_counts": {task: entry["records"]
                              for task, entry in manifest["tasks"].items()},
            "configs": manifest["configs"],
        },
        "skipped_tasks": skipped_tasks,
        "skipped_records": skipped_records,
        "metric_errors": metric_errors,
        "dataset2metric": {task: mapping[task] for task in sorted(mapping)},
        "truncation_policy": {
            "policy": args.truncation,
            "max_input_tokens": args.max_input_tokens,
            "head_tail": "keep the first and last half of the token ids, the "
                         "official LongBench middle truncation",
            "skip": "records over the limit are dropped and listed in skipped_records",
        },
        "decode": {
            "greedy": True,
            "max_new": ("per-task official dataset2maxlen.json"
                        if args.max_new is None else args.max_new),
            "stop_at_eos": True,
            "add_special_tokens": False,
            "device": args.device,
            "dtype": args.dtype,
            "attn_impl": args.attn_impl,
            "load_4bit": args.load_4bit,
        },
        "config": vars(args),
        "summary": summary,
        "records": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    report["progress_path"] = str(progress_path)
    return summary


if __name__ == "__main__":
    main()
