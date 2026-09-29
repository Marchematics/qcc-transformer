"""Cross-model, cross-task retention: bounded cache vs matched Full-KV.

Runs the *shipped* law (``qcc_transformer.retention.compile_bounded_cache``)
and an exact Full-KV arm over the same RULER records, on any Hugging Face causal
LM, and reports per-task retention.  The point of this script is generality: the
same code path, the same configuration and no per-model tuning, on several model
families and sizes.  ``--lex-cap 0`` turns the lexical anchors off, which is the
ablation that answers "is this only a rare-string trick?".

Retention for a record is ``bounded_recall / full_recall`` and only records the
Full-KV arm answers are counted, so the number is "how much of what Full-KV can
do survives", not an absolute score.  Both arms decode greedily with the same
loop as the benchmark harness (max_new tokens, stop at EOS), so the Llama-3.2-1B
run here is directly comparable with the stored harness run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qcc_transformer.hf_loading import _ensure_remote_code_compat, load_hf_causal_lm
from qcc_transformer.retention import (RetentionConfig, compile_bounded_cache,
                                       fixed_rope_length, prefill_capture)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--ruler-jsonl", default=os.environ.get("QCC_RULER_JSONL", ""),
                        help="RULER split JSONL (env: QCC_RULER_JSONL)")
    parser.add_argument("--tasks", nargs="*",
                        default=["niah_single_1", "niah_multikey_2", "niah_multikey_3", "vt"])
    parser.add_argument("--limit", type=int, default=None, help="records per task")
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--lex-cap", type=int, default=512)
    parser.add_argument("--hops", type=int, default=6)
    parser.add_argument("--obs", type=int, default=64)
    parser.add_argument("--nsink", type=int, default=4)
    parser.add_argument("--pool", type=int, default=7)
    parser.add_argument("--dilate", type=int, default=9)
    parser.add_argument("--scoring", default="last")
    parser.add_argument("--anchor-mode", default="pattern",
                        choices=["pattern", "rare", "both"])
    parser.add_argument("--min-anchor-len", type=int, default=3)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--prefill-chunk", type=int, default=8192)
    parser.add_argument("--key-chunk", type=int, default=4096)
    parser.add_argument("--arms", nargs="*", default=["full", "bounded"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-impl", default="sdpa",
                        help="attention implementation to request at load time")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_records(path, tasks, limit, max_prompt_tokens, tokenizer):
    picked, counts = [], defaultdict(int)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = row.get("task")
        if tasks and task not in tasks:
            continue
        if limit is not None and counts[task] >= limit:
            continue
        prompt = row["input"] + row.get("answer_prefix", "")
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        if max_prompt_tokens is not None and len(ids) > max_prompt_tokens:
            continue
        counts[task] += 1
        picked.append((row, prompt, len(ids)))
    return picked


def partial_recall(outputs, text):
    """RULER's string_match_all, averaged over references."""
    outs = [out for out in outputs if out]
    if not outs:
        return 0.0
    low = text.lower()
    return sum(1.0 for out in outs if out.lower() in low) / len(outs)


def strict_recall(outputs, text):
    outs = [out for out in outputs if out]
    if not outs:
        return 0.0
    low = text.lower()
    return 1.0 if all(out.lower() in low for out in outs) else 0.0


@torch.no_grad()
def greedy(model, cache, first, length, max_new, eos_ids, forward_kwargs=("cache_position",)):
    """The harness decode loop: absolute positions, stop at EOS, greedy."""
    kept = cache.get_seq_length()
    mask = torch.ones(1, kept + 1, device=first.device, dtype=torch.long)
    generated, current, start = [], first, time.time()
    kwargs = {}
    for step in range(max_new):
        position = torch.tensor([[length + step]], device=first.device)
        call = dict(past_key_values=cache, attention_mask=mask, position_ids=position,
                    use_cache=True)
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


def forward_kwargs_of(model):
    from qcc_transformer.retention import forward_accepts
    names = ("cache_position",) if forward_accepts(model, "cache_position") else ()
    return names


def main():
    args = parse_args()
    if not args.ruler_jsonl:
        raise SystemExit("--ruler-jsonl is required (or set QCC_RULER_JSONL): "
                         "point it at the RULER split JSONL")
    if args.trust_remote_code:
        _ensure_remote_code_compat()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=args.trust_remote_code)
    model = load_hf_causal_lm(args.model, dtype=args.dtype, device="cuda",
                              trust_remote_code=args.trust_remote_code,
                              load_in_4bit=args.load_4bit,
                              attn_implementation=args.attn_impl).eval()
    config = RetentionConfig(
        budget=args.budget, observation_window=args.obs, attention_sinks=args.nsink,
        pool=args.pool, dilate=args.dilate, lex_cap=args.lex_cap, chain_hops=args.hops,
        prefill_chunk=args.prefill_chunk, key_chunk=args.key_chunk, scoring=args.scoring,
        anchor_mode=args.anchor_mode, min_anchor_len=args.min_anchor_len)
    eos_ids = set()
    for source in (model.config.eos_token_id,
                   getattr(getattr(model, "generation_config", None), "eos_token_id", None)):
        if source is None:
            continue
        eos_ids |= set(source) if isinstance(source, (list, tuple, set)) else {source}
    kwargs_names = forward_kwargs_of(model)

    records = load_records(args.ruler_jsonl, args.tasks, args.limit,
                           args.max_prompt_tokens, tokenizer)
    print(f"[model] {args.model} records={len(records)} arms={args.arms} "
          f"budget={args.budget}+{args.lex_cap} kwargs={kwargs_names}", flush=True)

    results = []
    for index, (row, prompt, _tokens) in enumerate(records):
        ids = tokenizer(prompt, return_tensors="pt",
                        add_special_tokens=False).input_ids.cuda()
        length = int(ids.shape[1])
        # rope-identical to a one-pass prefill for chunked, LongRoPE checkpoints
        with fixed_rope_length(model, length) as pinned:
            for arm in args.arms:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.time()
                if arm == "full":
                    cache, logits, _captured = prefill_capture(
                        model, ids, config.observation_window, config.prefill_chunk)
                else:
                    cache, logits = compile_bounded_cache(model, ids, config, tokenizer=tokenizer)
                torch.cuda.synchronize()
                prefill_seconds = time.time() - start
                kept = int(cache.get_seq_length())
                generated, decode_seconds = greedy(model, cache, logits[:, -1:].argmax(-1),
                                                   length, args.max_new, eos_ids, kwargs_names)
                text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                outputs = row.get("outputs", [])
                entry = {
                    "row": index, "task": row.get("task"),
                    "ruler_index": row.get("index"), "nominal_length": row.get("length"),
                    "prompt_tokens": length, "arm": arm, "kept_slots": kept,
                    "slots_per_token": round(kept / length, 4),
                    "prediction": text, "generated": len(generated), "outputs": outputs,
                    "recall_partial": round(partial_recall(outputs, text), 4),
                    "recall_strict": round(strict_recall(outputs, text), 4),
                    "prefill_s": round(prefill_seconds, 2),
                    "decode_s": round(decode_seconds, 2),
                    "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    "pinned_rope": bool(pinned),
                }
                results.append(entry)
                print(f"[{index:>3d}/{len(records)}] {entry['task']:<16} L={length:>6d} "
                      f"{arm:<7} pinned={int(pinned)} kept={kept:>6d} "
                      f"partial={entry['recall_partial']:.2f} pred={text[:32]!r}", flush=True)
                del cache
                torch.cuda.empty_cache()

    summary = summarize(results, args)
    out = {"model": args.model, "label": args.label or Path(args.model).name,
           "config": vars(args), "summary": summary, "records": results}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("\n" + json.dumps(summary, indent=1))
    return summary


def summarize(results, args):
    by_arm = defaultdict(list)
    for entry in results:
        by_arm[entry["arm"]].append(entry)
    full = {(e["row"]): e for e in by_arm.get("full", [])}
    tasks = sorted({e["task"] for e in results})
    summary = {"tasks": {}, "aggregate": {}}
    for task in tasks:
        rows = [entry for entry in results if entry["task"] == task]
        arms = {}
        for arm in args.arms:
            arm_rows = [e for e in rows if e["arm"] == arm]
            if arm_rows:
                arms[arm] = {
                    "records": len(arm_rows),
                    "mean_partial": round(sum(e["recall_partial"] for e in arm_rows) / len(arm_rows), 4),
                    "mean_strict": round(sum(e["recall_strict"] for e in arm_rows) / len(arm_rows), 4),
                    "mean_kept_slots": round(sum(e["kept_slots"] for e in arm_rows) / len(arm_rows), 1),
                }
        if "full" in arms and "bounded" in arms:
            scores = []
            for entry in rows:
                if entry["arm"] != "bounded":
                    continue
                reference = full.get(entry["row"])
                if reference and reference["recall_partial"] > 0:
                    scores.append(entry["recall_partial"] / reference["recall_partial"])
            arms["retention"] = {
                "matched_records": len(scores),
                "mean": round(sum(scores) / len(scores), 4) if scores else None,
                "worst": round(min(scores), 4) if scores else None,
            }
        summary["tasks"][task] = arms
    matched = [(task, data["retention"]) for task, data in summary["tasks"].items()
               if data.get("retention", {}).get("mean") is not None]
    if matched:
        summary["aggregate"] = {
            "tasks": len(matched),
            "mean_retention": round(sum(r["mean"] for _t, r in matched) / len(matched), 4),
            "worst_task_retention": round(min(r["mean"] for _t, r in matched), 4),
            "worst_record_retention": round(min(r["worst"] for _t, r in matched), 4),
            "matched_records": sum(r["matched_records"] for _t, r in matched),
        }
    return summary


if __name__ == "__main__":
    main()
