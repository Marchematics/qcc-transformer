#!/usr/bin/env python
"""Low-rank latent KV against bounded exact-KV selection, at matched bytes.

The frontier long-context models (MLA-style latent attention in the DeepSeek and
Kimi families) buy context with a *low-rank* KV representation: every token is kept,
but each one costs a handful of latent coordinates instead of a full key and value.
This project buys context by keeping a *subset* of tokens exactly.  Both are
training-free at this scale if the low-rank basis is fitted post hoc, so the two can
be compared on the same records and the same byte budget:

* ``bounded`` -- the shipped selection, `budget + lex_cap` slots per (layer, kv-head),
  keys and values kept exactly (the same code path the other benchmarks use);
* ``lowrank`` -- all `L` tokens kept as rank-`r` latents whose stored bytes equal the
  bounded arm's, i.e. ``r = slots * head_dim / L`` (about 9 of 64 for a 32K prompt at
  4,608 slots), with the basis shared across tokens.

The low-rank arm reconstructs its keys and values from the latent at compile time and
decodes through the same loop, so the comparison is about *what the cache keeps*, not
about a fused kernel: quality per stored byte.  Ranks are reported per record, and
the byte accounting is printed next to every arm.

Usage::

    python benchmarks/benchmark_lowrank_kv_ruler.py --model <checkpoint> \\
        --ruler-jsonl <split> --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt \\
        --budget 4096 --lex-cap 512 --out artifacts/lowrank-kv-ruler.json
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

if __package__ in (None, ""):            # run as a script from benchmarks/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import benchmark_bounded_decode_frontier as L
    import benchmark_bounded_decode_ruler as R
except ImportError:  # pragma: no cover - package layout
    from benchmarks import benchmark_bounded_decode_frontier as L
    from benchmarks import benchmark_bounded_decode_ruler as R


def stored_bytes(slots, head_dim, layers, kv_heads, itemsize=2):
    """Bytes a cache stores: `slots` x `head_dim` per (layer, kv-head) for K and V."""
    return slots * head_dim * layers * kv_heads * itemsize * 2


def compress_low_rank(cache, rank):
    """Replace every (layer, kv-head) key/value block by its best rank-`rank` form.

    Returns `(cache_like, latent_slots, head_dim, layers, kv_heads)` where
    `latent_slots` is the per-token latent width, i.e. what a real implementation
    would store (`rank` coordinates per token instead of `head_dim`).
    """
    layers = cache.layers
    kv_heads = layers[0].keys.shape[1]
    head_dim = layers[0].keys.shape[-1]
    for layer in layers:
        keys = layer.keys[0]          # (kv_heads, L, d)
        values = layer.values[0]
        for head in range(kv_heads):
            for tensor in (keys[head], values[head]):
                matrix = tensor.float()                       # (L, d)
                u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
                keep = min(rank, s.shape[0])
                approx = (u[:, :keep] * s[:keep]) @ vh[:keep]
                tensor.copy_(approx.to(tensor.dtype))
    return cache, rank, head_dim, len(layers), kv_heads


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.environ.get("QCC_MODEL",
                                                          "meta-llama/Llama-3.2-1B-Instruct"))
    parser.add_argument("--ruler-jsonl", default=os.environ.get("QCC_RULER_JSONL"),
                        help="official RULER split; $QCC_RULER_JSONL by default")
    parser.add_argument("--tasks", nargs="+",
                        default=["niah_single_1", "niah_multikey_2", "niah_multikey_3", "vt"])
    parser.add_argument("--lengths", type=int, nargs="+", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--lex-cap", type=int, default=512)
    parser.add_argument("--obs", type=int, default=64)
    parser.add_argument("--nsink", type=int, default=4)
    parser.add_argument("--pool", type=int, default=7)
    parser.add_argument("--dilate", type=int, default=9)
    parser.add_argument("--hops", type=int, default=6)
    parser.add_argument("--key-chunk", type=int, default=1024)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--prefill-chunk", type=int, default=8192)
    parser.add_argument("--arms", nargs="*",
                        default=["full", "bounded", "lowrank1", "lowrank2"],
                        help="full, the shipped bounded selection, and `lowrankN` "
                             "latents at N times the byte-matched rank")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    L.TOKENIZER = tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    eos = model.config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}

    records = R.load_ruler(args.ruler_jsonl, args.tasks, args.lengths, args.max_records,
                           True, tokenizer=tokenizer, question_tokens=args.obs, hops=args.hops)
    print(f"[setup] {len(records)} records", flush=True)

    results = []
    for index, rec in enumerate(records):
        ids = tokenizer(rec.prompt, return_tensors="pt",
                        add_special_tokens=False).input_ids.to("cuda")
        length = int(ids.shape[1])
        torch.cuda.reset_peak_memory_stats()
        cache, logits, captured = L.prefill_capture(model, ids, args.obs, args.prefill_chunk)
        next_id = logits[:, -1:].argmax(-1)
        layers = cache.layers
        kv_heads = layers[0].keys.shape[1]
        head_dim = layers[0].keys.shape[-1]
        slots = min(args.budget + args.lex_cap, length)
        # the rank whose stored bytes equal the bounded arm's: L x rank == slots x d
        rank = max(1, round(slots * head_dim / length))
        originals = [(layer.keys.clone(), layer.values.clone()) for layer in layers]

        for arm in args.arms:
            for layer, (ok, ov) in zip(layers, originals):
                layer.keys, layer.values = ok.clone(), ov.clone()
            stored = None
            if arm == "bounded":
                # exactly the shipped selection, through the shared code path
                scores = L.obs_scores(model, captured, cache, length, args.obs, "last",
                                      args.key_chunk)
                nrecent = max(1, int(args.budget * 0.25))
                target = min(slots, length)
                idxs = L.build_idxs("lex_obs", scores, cache, rec, args.budget,
                                    args.nsink, nrecent, length, args.pool, ids.device,
                                    args.dilate, args.lex_cap)
                for layer, slots_index in zip(layers, idxs):
                    gather = slots_index.unsqueeze(0).unsqueeze(-1).expand(
                        1, kv_heads, slots_index.shape[1], head_dim)
                    layer.keys = layer.keys.gather(2, gather).contiguous()
                    layer.values = layer.values.gather(2, gather).contiguous()
                stored = stored_bytes(target, head_dim, len(layers), kv_heads)
            elif arm.startswith("lowrank"):
                # `lowrank1` is the byte-matched rank, `lowrank2` twice the bytes, ...
                multiplier = int(arm[len("lowrank"):] or 1)
                this_rank = min(head_dim, max(1, rank * multiplier))
                compress_low_rank(cache, this_rank)
                # the latent is `this_rank` coordinates per token, not per slot
                stored = stored_bytes(length, this_rank, len(layers), kv_heads)
                rank = this_rank
            else:                                     # full
                stored = stored_bytes(length, head_dim, len(layers), kv_heads)

            started = time.time()
            text, ngen, _decode_s = L.decode(model, cache, next_id, length, args.max_new,
                                             eos_ids)
            score = R.recall_of(rec, text)
            row = {
                "row": index, "task": rec.meta["task"], "length": rec.meta["length"],
                "prompt_tokens": length, "arm": arm, "score": score,
                "kept_slots": int(cache.get_seq_length()),
                "stored_bytes": stored,
                "rank": rank if arm.startswith("lowrank") else None,
                "head_dim": head_dim, "kv_heads": kv_heads, "layers": len(layers),
                "generated": ngen, "prediction": text,
                "wall_s": round(time.time() - started, 2),
            }
            results.append(row)
            print(f"[{index:>4d}/{len(records)}] {rec.meta['task']:<16} L={length:>6d} "
                  f"{arm:<8} rank={row['rank']} stored={stored / 2**20:8.2f}MiB "
                  f"recall={score:.0f}", flush=True)
        del cache, originals, captured
        torch.cuda.empty_cache()
        if (index + 1) % 5 == 0:
            Path(args.out).write_text(json.dumps({"partial": True, "results": results},
                                                 indent=1))

    summary = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))
    full_ok = {(r["row"]): r["score"] >= 1.0 for r in results if r["arm"] == "full"}
    for row in results:
        if row["arm"] == "full":
            continue
        cell = summary[row["task"]][row["arm"]]
        cell[0] += row["score"]
        cell[1] += 1.0 if full_ok.get(row["row"]) else 0.0
        cell[2] += 1
    print("\n== retention (recall sum / records the Full-KV arm answers) ==")
    for task in sorted(summary):
        for arm in sorted(summary[task]):
            got, base, n = summary[task][arm]
            ratio = got / base if base else float("nan")
            print(f"  {task:<16} {arm:<8} {got:.0f}/{base:.0f} = {ratio:.4f}  (n={n})")
    Path(args.out).write_text(json.dumps({
        "model": args.model, "ruler": args.ruler_jsonl, "config": vars(args),
        "results": results,
        "summary": {task: {arm: {"recall_sum": v[0], "full_correct": v[1], "records": v[2],
                                 "retention": (v[0] / v[1]) if v[1] else None}
                           for arm, v in arms.items()}
                    for task, arms in summary.items()},
    }, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
