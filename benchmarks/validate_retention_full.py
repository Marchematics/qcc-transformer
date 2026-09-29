"""Does the *shipped* API reproduce the benchmark's RULER numbers?

Runs N records per task through ``qcc_transformer.retention`` at the quality
budget and scores them with RULER's official metric, so the headline claim is
attached to shipped code rather than to a benchmark script.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qcc_transformer.retention import RetentionConfig, compile_bounded_cache

DEFAULT_MODEL = os.environ.get("QCC_MODEL", "meta-llama/Llama-3.2-1B-Instruct")


def official(pred: str, refs: list[str]) -> float:
    if not refs:
        return 0.0
    low = pred.lower()
    return sum(1.0 for r in refs if r.lower() in low) / len(refs)


@torch.no_grad()
def greedy(model, cache, next_id, offset, kept, max_new, eos_ids):
    mask = torch.ones(1, kept + 1, device="cuda", dtype=torch.long)
    gen, cur = [], next_id
    for step in range(max_new):
        cp = torch.tensor([offset + step], device="cuda")
        out = model(cur, past_key_values=cache, attention_mask=mask, cache_position=cp,
                    position_ids=cp.unsqueeze(0), use_cache=True)
        cur = out.logits[:, -1:].argmax(-1)
        gen.append(int(cur))
        mask = torch.cat([mask, torch.ones(1, 1, device="cuda", dtype=torch.long)], dim=1)
        if int(cur) in eos_ids:
            break
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ruler-jsonl", default=os.environ.get("QCC_RULER_JSONL", ""),
                    help="RULER split JSONL (env: QCC_RULER_JSONL)")
    ap.add_argument("--per-task", type=int, default=5)
    ap.add_argument("--budget", type=int, default=4096)
    ap.add_argument("--lex-cap", type=int, default=512)
    ap.add_argument("--hops", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.ruler_jsonl:
        ap.error("--ruler-jsonl is required (or set QCC_RULER_JSONL)")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    eos = model.config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}

    rows = [json.loads(l) for l in open(args.ruler_jsonl)]
    by_task: dict[str, list] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)

    config = RetentionConfig(budget=args.budget, lex_cap=args.lex_cap,
                             chain_hops=args.hops, prefill_chunk=8192)
    results = []
    for task, records in sorted(by_task.items()):
        for row in records[: args.per_task]:
            prompt = row["input"] + row.get("answer_prefix", "")
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            length = ids.shape[1]
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            cache, logits = compile_bounded_cache(model, ids, config, tokenizer=tok)
            kept = cache.get_seq_length()
            gen = greedy(model, cache, logits[:, -1:].argmax(-1), length, kept,
                         args.max_new, eos_ids)
            text = tok.decode(gen, skip_special_tokens=True)
            score = official(text, row["outputs"])
            results.append({"task": task, "length": length, "kept": kept,
                            "score": round(score, 4), "prediction": text[:60],
                            "seconds": round(time.time() - t0, 1),
                            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
            print(f"{task:<17} L={length:>6} kept={kept} score={score:.3f} "
                  f"{results[-1]['seconds']:>5.1f}s", flush=True)
            del cache
            torch.cuda.empty_cache()

    per_task: dict[str, list[float]] = {}
    for row in results:
        per_task.setdefault(row["task"], []).append(row["score"])
    summary = {t: round(sum(v) / len(v), 4) for t, v in sorted(per_task.items())}
    aggregate = round(sum(r["score"] for r in results) / len(results), 4)
    print("\nper-task official recall:", summary)
    print("aggregate:", aggregate)
    Path(args.out).write_text(json.dumps(
        {"model": args.model, "budget": args.budget, "lex_cap": args.lex_cap,
         "aggregate": aggregate, "per_task": summary, "results": results}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
