"""Bounded exact-KV decode frontier on the official NVIDIA RULER JSONL split.

Reuses the prefill / selection / decode primitives from
``benchmark_bounded_decode_frontier.py`` (imported here as ``longctx``) but
drives them from real RULER records instead of the synthetic generator, and
scores official answer recall (every expected output string must appear,
case-insensitively) instead of a single magic number.

Usage:
    python ruler_frontier.py --ruler-jsonl /path/ruler_subset.jsonl \
        --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt \
        --policies full obs_last obs_mean --budgets 128 256 512 1024 \
        --out ruler_v1.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # repo layout
    import benchmark_bounded_decode_frontier as L
except ImportError:  # authoring workspace layout
    import longctx as L


def load_ruler(path, tasks, lengths, max_records=None, answer_prefix=True):
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if tasks and row.get("task") not in tasks:
            continue
        if lengths and row.get("length") not in lengths:
            # allow tolerances only when the caller asked for exact lengths
            if not any(abs(row.get("length", 0) - x) <= 64 for x in lengths):
                continue
        prompt = row["input"]
        if answer_prefix and row.get("answer_prefix"):
            prompt = prompt + row["answer_prefix"]
        pos = row.get("token_position_answer")
        rec = L.Record(
            prompt=prompt,
            question_key="",
            answer=row["outputs"][0] if row.get("outputs") else "",
            meta={
                "task": row.get("task"),
                "length": row.get("length"),
                "outputs": row.get("outputs", []),
                "index": row.get("index"),
                "length_bucket": row.get("length_bucket_requested"),
            },
        )
        if isinstance(pos, int) and pos >= 0:
            rec.answer_positions = [pos]
            rec.needle_positions = [pos]
        records.append(rec)
        if max_records and len(records) >= max_records:
            break
    return records


def recall_of(rec, text):
    outs = [o for o in rec.meta.get("outputs", []) if o]
    if not outs:
        return 0.0
    low = text.lower()
    return 1.0 if all(o.lower() in low for o in outs) else 0.0


@torch.no_grad()
def run_record(model, rec, args, eos_ids, row_id):
    ids = L.TOKENIZER(rec.prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    Lc = ids.shape[1]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    cache, last_logits, captured = L.prefill_capture(model, ids, args.obs, args.prefill_chunk)
    torch.cuda.synchronize()
    prefill_s = time.time() - t0
    next_id = last_logits.argmax(-1)
    orig = [(layer.keys, layer.values) for layer in cache.layers]
    peak = torch.cuda.max_memory_allocated()
    print(f"[ruler] {rec.meta['task']:<16} L={Lc:>6d} outs={rec.meta['outputs'][:2]} "
          f"prefill={prefill_s:.1f}s peak={peak / 2**30:.2f}GiB", flush=True)

    score_cache: dict = {}

    def get_scores(policy):
        if policy not in score_cache:
            score_cache[policy] = L.obs_scores(model, captured, cache, Lc, args.obs, policy[4:], args.key_chunk)
        return score_cache[policy]

    rows = []
    del last_logits
    for policy in args.policies:
        for layer, (ok, ov) in zip(cache.layers, orig):
            layer.keys, layer.values = ok, ov
        budgets = [None] if policy == "full" else args.budgets
        scores = None if policy in ("full", "recent", "sink_recent") else get_scores(policy)
        for budget in budgets:
            eff = budget if budget is not None else Lc
            nrecent = max(1, int(eff * 0.25)) if budget is not None else 0
            idxs = L.build_idxs(policy, scores, cache, rec, eff, args.nsink, nrecent, Lc, args.pool, ids.device, args.dilate)
            diag = L.selection_stats(idxs, rec.answer_positions, Lc)
            if idxs is None:
                for layer, (ok, ov) in zip(cache.layers, orig):
                    layer.keys, layer.values = ok, ov
            else:
                for layer, (ok, ov), idx in zip(cache.layers, orig, idxs):
                    H = ok.shape[1]
                    ek = idx.unsqueeze(0).unsqueeze(-1).expand(1, H, idx.shape[1], ok.shape[-1])
                    layer.keys = ok.gather(2, ek).contiguous()
                    layer.values = ov.gather(2, ek).contiguous()
            kept = int(cache.get_seq_length())
            text, ngen, dec_s = L.decode(model, cache, next_id, Lc, args.max_new, eos_ids)
            rec_score = recall_of(rec, text)
            row = {
                "row_id": row_id,
                "task": rec.meta["task"],
                "length": rec.meta["length"],
                "length_bucket": rec.meta["length_bucket"],
                "context_tokens": Lc,
                "policy": policy,
                "budget": budget,
                "kept_slots": kept,
                "prediction": text,
                "generated": ngen,
                "outputs": rec.meta["outputs"],
                "answer_recall": rec_score,
                "selection": diag,
                "decode_s": round(dec_s, 4),
                "prefill_s": round(prefill_s, 3),
                "peak_cuda_bytes": peak,
            }
            print(f"  {policy:10s} B={str(budget):>5s} recall={rec_score:.0f} pred={text[:50]!r}", flush=True)
            rows.append(row)
    del cache, orig, captured
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/qcc/models/Llama-3.2-1B-Instruct")
    ap.add_argument("--ruler-jsonl", required=True)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--lengths", type=int, nargs="+", default=None)
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--policies", nargs="+", default=["full", "obs_last", "obs_mean"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--obs", type=int, default=64)
    ap.add_argument("--nsink", type=int, default=4)
    ap.add_argument("--pool", type=int, default=7)
    ap.add_argument("--dilate", type=int, default=0)
    ap.add_argument("--key-chunk", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--prefill-chunk", type=int, default=8192)
    ap.add_argument("--no-answer-prefix", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    L.TOKENIZER = tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    model.eval()
    inner = model.lm_head

    class _LastTokenLMHead(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x):
            return self.wrapped(x[:, -1:, :])

    model.lm_head = _LastTokenLMHead(inner)
    eos = model.config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}

    records = load_ruler(args.ruler_jsonl, args.tasks, args.lengths,
                         args.max_records, not args.no_answer_prefix)
    print(f"[setup] {len(records)} RULER records", flush=True)

    rows = []
    for i, rec in enumerate(records):
        rows.extend(run_record(model, rec, args, eos_ids, i))
        torch.cuda.empty_cache()
        if (i + 1) % 5 == 0:
            Path(args.out).write_text(json.dumps({"partial": True, "results": rows}, indent=1), encoding="utf-8")
    Path(args.out).write_text(json.dumps(
        {"model": args.model, "ruler": args.ruler_jsonl,
         "config": {k: v for k, v in vars(args).items()}, "results": rows}, indent=1))

    # aggregates
    full = {}
    for r in rows:
        if r["policy"] == "full":
            full[r["row_id"]] = r["answer_recall"] >= 1.0
    by_task = defaultdict(lambda: [0.0, 0.0, 0])
    by_budget = defaultdict(lambda: [0.0, 0, 0])
    for r in rows:
        if r["policy"] == "full":
            continue
        key = (r["task"], r["policy"], r["budget"])
        by_task[key][0] += r["answer_recall"]
        by_task[key][1] += 1.0 if full.get(r["row_id"]) else 0.0
        by_task[key][2] += 1
    print("\n== RULER retention (numerator = recall sum; denom = records where Full-KV correct) ==")
    tasks = sorted({k[0] for k in by_task})
    budgets = sorted({k[2] for k in by_task})
    for pol in args.policies:
        if pol == "full":
            continue
        for b in budgets:
            tot = [0.0, 0.0]
            parts = []
            for t in tasks:
                k = (t, pol, b)
                if k in by_task:
                    c, d, n = by_task[k]
                    tot[0] += c
                    tot[1] += d
                    parts.append(f"{t}={c:.0f}/{d:.0f}")
            agg = tot[0] / tot[1] if tot[1] else float("nan")
            print(f"  {pol:10s} B={b:>5} aggregate={agg:.3f}  " + "  ".join(parts))
    full_total = sum(full.values())
    print(f"  Full-KV correct records: {full_total}/{len(full)}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
