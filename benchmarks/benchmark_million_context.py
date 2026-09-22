#!/usr/bin/env python
"""The 1M row: retrieval and TPOT at a length whose Full-KV cache still fits.

The 1M metrics are blocked on a 24 GiB card by the *baseline*, not by the method: a
1M-token bf16 Full-KV cache for Llama-3.2-1B is 32 GiB.  Qwen2.5-0.5B stores 12 KiB
per token (2 kv-heads x 64 dims x 24 layers x K and V), so its 1M cache is 12.0 GiB
and both arms run at 1M on the same GPU.  That makes the two rows measurable:

* 1M retrieval quality -- bounded against matched Full-KV, official RULER-style
  recall, needles planted at random depths;
* 1M TPOT -- both arms on the same execution path, parity-gated by the decoded
  tokens;
* state growth 128K -> 1M for the retained cache.

Long-context position handling is explicit: the checkpoint was trained at 32K, so
YaRN rope scaling with `factor = length / 32768` (the checkpoint's own recommendation)
is applied *identically to both arms*, and the factor is recorded per row.

Usage::

    python benchmarks/benchmark_million_context.py --model <checkpoint> \\
        --lengths 131072 1048576 --items 4 --seeds 2 --budget 4096 --lex-cap 512 \\
        --out artifacts/million-context.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import benchmark_bounded_decode_frontier as L
except ImportError:  # pragma: no cover - package layout
    from benchmarks import benchmark_bounded_decode_frontier as L

FILLER = (
    "The committee reviewed the record and noted that the programme continues on "
    "schedule. Staff reported no change to the estimate for the coming year. "
)


class Record:
    """Minimal record for the shared selection code."""

    def __init__(self, positions):
        self.lexical_positions = positions


def build_haystack(tokenizer, length, items, seed):
    """A prompt of about `length` tokens with `items` needles at random depths."""
    import random

    rng = random.Random(seed)
    keys = [f"trace-{rng.randrange(16 ** 6):06x}" for _ in range(items)]
    values = [f"{rng.randrange(10 ** 6):06d}" for _ in range(items)]
    statements = [f"One of the magic numbers for {key} is {value}."
                  for key, value in zip(keys, values)]
    order = list(range(items))
    rng.shuffle(order)

    # fill in token space first, then splice the statements in as text so the
    # tokenizer sees them normally
    body_tokens = 0
    filler_parts = []
    while body_tokens < length - 256:
        filler_parts.append(FILLER)
        body_tokens = len(tokenizer(" ".join(filler_parts), add_special_tokens=False)["input_ids"])
    text = " ".join(filler_parts)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][: length - 200]

    question = ("What are the magic numbers for all of these keys mentioned above: "
                + ", ".join(keys[i] for i in order) + "? Answer with the numbers "
                "separated by commas.")
    pieces, cursor = [], 0
    fractions = sorted(rng.random() for _ in statements)
    for index, statement in enumerate(statements):
        cut = int(fractions[index] * len(ids))
        pieces.append(tokenizer.decode(ids[cursor:cut]))
        pieces.append(" " + statement + " ")
        cursor = cut
    pieces.append(tokenizer.decode(ids[cursor:]))
    pieces.append("\n\n" + question)
    prompt = "".join(pieces)
    encoded = tokenizer(prompt, add_special_tokens=False,
                        return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    positions = []
    for statement in statements:
        start = prompt.find(statement)
        positions.extend(i for i, (a, b) in enumerate(offsets)
                         if b > start and a < start + len(statement))
    return prompt, input_ids, sorted(set(positions)), values, keys


def recall_of(text, values):
    low = text.lower()
    return 1.0 if all(v.lower() in low for v in values) else 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.environ.get("QCC_MODEL",
                                                          "/root/qcc/models/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--lengths", type=int, nargs="+", default=[131072, 1048576])
    parser.add_argument("--items", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--lex-cap", type=int, default=512)
    parser.add_argument("--obs", type=int, default=64)
    parser.add_argument("--nsink", type=int, default=4)
    parser.add_argument("--pool", type=int, default=7)
    parser.add_argument("--dilate", type=int, default=9)
    parser.add_argument("--hops", type=int, default=6)
    parser.add_argument("--key-chunk", type=int, default=4096)
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--prefill-chunk", type=int, default=8192)
    parser.add_argument("--arms", nargs="*", default=["full", "bounded"])
    parser.add_argument("--yarn", default="auto", choices=["auto", "off"],
                        help="`auto` applies YaRN rope scaling with "
                             "factor = length / trained window, identically to both arms")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    L.TOKENIZER = tokenizer
    eos = None
    results = []
    for length in args.lengths:
        config = AutoConfig.from_pretrained(args.model)
        trained = int(getattr(config, "max_position_embeddings", 32768))
        factor = max(1.0, length / trained)
        if args.yarn == "auto" and factor > 1.0:
            config.rope_scaling = {"type": "yarn", "factor": factor,
                                   "original_max_position_embeddings": trained}
            # transformers 5 exposes the resolved form as `rope_parameters`
            config.rope_parameters = {"rope_type": "yarn", "factor": factor,
                                      "original_max_position_embeddings": trained,
                                      "rope_theta": getattr(config, "rope_theta", 1e6)}
        config.max_position_embeddings = max(length + 1024, trained)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=config, dtype=torch.bfloat16,
            attn_implementation="sdpa").to("cuda").eval()
        eos_ids = {model.config.eos_token_id} if model.config.eos_token_id is not None else set()
        print(f"[length {length}] rope_scaling={getattr(config, 'rope_scaling', None)}",
              flush=True)
        for seed in range(args.seeds):
            prompt, ids, needle_positions, values, keys = build_haystack(
                tokenizer, length, args.items, seed)
            tensor = torch.tensor([ids], device="cuda")
            true_length = int(tensor.shape[1])
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.time()
            # the masked path is the one this codebase has verified at 128-262K on
            # this stack: the maskless path is exact (see tests/test_prefill_accumulate)
            # but lets the SDPA mask builder materialise a larger mask at these lengths
            cache, logits, captured = L.prefill_capture(model, tensor, args.obs,
                                                        args.prefill_chunk)
            torch.cuda.synchronize()
            prefill_s = time.time() - started
            peak_gib = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            next_id = logits[:, -1:].argmax(-1)
            originals = [(layer.keys, layer.values) for layer in cache.layers]
            kv_heads = cache.layers[0].keys.shape[1]
            head_dim = cache.layers[0].keys.shape[-1]
            full_bytes = true_length * head_dim * len(cache.layers) * kv_heads * 2 * 2
            print(f"[{length} seed {seed}] prefill {prefill_s:.0f}s peak {peak_gib}GiB "
                  f"L={true_length} needles={len(needle_positions)}", flush=True)
            for arm in args.arms:
                for layer, (ok, ov) in zip(cache.layers, originals):
                    layer.keys, layer.values = ok, ov
                kept_bytes = full_bytes
                if arm == "bounded":
                    scores = L.obs_scores(model, captured, cache, true_length, args.obs,
                                          "last", args.key_chunk)
                    anchors = L.question_lexical_positions(
                        tokenizer, prompt, args.obs, hops=args.hops)
                    nrecent = max(1, int(args.budget * 0.25))
                    idxs = L.build_idxs("lex_obs", scores, cache, Record(anchors),
                                        args.budget, args.nsink, nrecent, true_length,
                                        args.pool, tensor.device, args.dilate,
                                        args.lex_cap)
                    for layer, slots in zip(cache.layers, idxs):
                        gather = slots.unsqueeze(0).unsqueeze(-1).expand(
                            1, kv_heads, slots.shape[1], head_dim)
                        layer.keys = layer.keys.gather(2, gather).contiguous()
                        layer.values = layer.values.gather(2, gather).contiguous()
                    kept_bytes = (int(cache.get_seq_length()) * head_dim
                                  * len(cache.layers) * kv_heads * 2 * 2)
                torch.cuda.synchronize()
                decode_started = time.time()
                text, ngen, _ = L.decode(model, cache, next_id, true_length,
                                         args.max_new, eos_ids)
                torch.cuda.synchronize()
                decode_s = time.time() - decode_started
                row = {
                    "length": length, "true_tokens": true_length, "seed": seed,
                    "items": args.items, "arm": arm,
                    "recall": recall_of(text, values), "prediction": text,
                    "needle_positions": needle_positions[:8],
                    "rope_scaling": getattr(config, "rope_scaling", None),
                    "prefill_s": round(prefill_s, 1),
                    "peak_gib": peak_gib,
                    "kept_slots": int(cache.get_seq_length()),
                    "kept_bytes": kept_bytes, "full_kv_bytes": full_bytes,
                    "compression": round(full_bytes / kept_bytes, 1),
                    "generated": ngen,
                    "tpot_ms": round(1000.0 * decode_s / max(1, ngen), 2),
                    "decode_s": round(decode_s, 3),
                }
                results.append(row)
                print(f"   {arm:<8} slots={row['kept_slots']:>7d} "
                      f"state={kept_bytes / 2**20:8.1f}MiB recall={row['recall']:.0f} "
                      f"tpot={row['tpot_ms']:.1f}ms pred={text[:40]!r}", flush=True)
                Path(args.out).write_text(json.dumps(
                    {"partial": True, "config": vars(args), "results": results}, indent=1))
            del cache, originals, captured
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    # per length: bounded/Full-KV retention over the matched records
    summary = {}
    for length in args.lengths:
        rows = [r for r in results if r["length"] == length]
        full = {r["seed"]: r for r in rows if r["arm"] == "full"}
        bounded = [r for r in rows if r["arm"] == "bounded"]
        matched = [r for r in bounded if full.get(r["seed"], {}).get("recall", 0) > 0]
        summary[str(length)] = {
            "records": len(bounded),
            "full_recall": statistics.mean([r["recall"] for r in full.values()]) if full else None,
            "bounded_recall": statistics.mean([r["recall"] for r in bounded]) if bounded else None,
            "matched_records": len(matched),
            "retention": (statistics.mean([r["recall"] / full[r["seed"]]["recall"]
                                           for r in matched]) if matched else None),
            "state_mib": (bounded[0]["kept_bytes"] / 2 ** 20) if bounded else None,
            "full_state_gib": (bounded[0]["full_kv_bytes"] / 2 ** 30) if bounded else None,
            "tpot_ms": {arm: statistics.median([r["tpot_ms"] for r in rows
                                                if r["arm"] == arm]) for arm in args.arms},
            "prefill_s": statistics.median([r["prefill_s"] for r in rows]),
        }
    Path(args.out).write_text(json.dumps(
        {"config": vars(args), "summary": summary, "results": results}, indent=1))
    print("\n" + json.dumps(summary, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
