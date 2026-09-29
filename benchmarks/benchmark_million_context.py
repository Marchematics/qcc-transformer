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
import re
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
from qcc_transformer.preallocated_cache import preallocated_cache_for

FILLER = (
    "The committee reviewed the record and noted that the programme continues on "
    "schedule. Staff reported no change to the estimate for the coming year. "
)


class Record:
    """Minimal record for the shared selection code."""

    def __init__(self, positions):
        self.lexical_positions = positions


# single-token English nouns, used when the answer style is `words`: a 6-digit number
# is six tokens on this tokenizer, and a 0.5B model mis-generates one digit often
# enough to hide what the row is about (measured at 32K *and* 128K, both rope
# settings: 3 of 4 values exact, one value missing its leading digit, so every record
# scored 0 while retrieval was plainly working).  A single-token answer makes the
# metric measure retention instead of digit arithmetic.
_VALUE_WORDS = (
    "apple", "orange", "banana", "bridge", "castle", "forest", "hammer", "needle",
    "rabbit", "table", "window", "yellow", "anchor", "engine", "mirror", "river",
    "silver", "wagon", "kernel", "melon", "rocket", "danger", "hunter", "ribbon",
    "badge", "fabric", "marker", "parcel", "socket", "vine",
)


def build_haystack(tokenizer, length, items, seed, answer_prefix=True,
                   value_style="digits", value_digits=2):
    """A prompt of about `length` tokens with `items` needles at random depths.

    Built in *token* space: the filler is tiled from its own token ids and the needle
    statements' token ids are spliced in, so there is no detokenise/re-tokenise round
    trip over a megabyte-scale string.  The text-space builder this replaces did not
    return within seven minutes for a 128K prompt (report 3.35), while the model work
    that follows it is 3.2 s.  The needle positions are exact by construction, which is
    what the selection needs.

    ``answer_prefix`` opens the answer for the model ("The magic words are"), which is
    what the working long-context protocols do (RULER ships an `answer_prefix` per
    record) and it is not cosmetic on this checkpoint: asked cold, Qwen2.5-0.5B
    re-lists the keys from the question instead of answering, so every arm scores zero
    and the row measures nothing about retention.  With the prefix the model only has
    to emit the values.

    ``value_style``/``value_digits`` pick what the needle holds.  `digits` is the
    default because the number phrasing is what this checkpoint actually answers, and
    the width matters: on this tokenizer *every* digit is its own token, so a six-digit
    value is a six-token answer that a 0.5B model reproduces with an occasional dropped
    digit (measured: 3 of 4 values exact at both 32K and 128K, which scored the whole
    record 0 while retrieval was plainly working).  Two digits keeps the same task one
    or two tokens wide.  `words` uses single-token nouns instead (see `_VALUE_WORDS`).
    """
    import random

    rng = random.Random(seed)
    keys = [f"trace-{rng.randrange(16 ** 6):06x}" for _ in range(items)]
    if value_style == "digits":
        width = int(value_digits)
        values = [f"{rng.randrange(10 ** width):0{width}d}" for _ in range(items)]
        noun, noun_plural = "number", "numbers"
    else:
        values = rng.sample(_VALUE_WORDS, items)
        noun, noun_plural = "word", "words"
    # the statements carry their own surrounding spaces: the filler is spliced in
    # *token* space, so a bare statement glues to the neighbouring filler tokens
    # (measured: "...estimate for the coming year. TheOne of the magic numbers...")
    statements = [f" One of the magic {noun_plural} for {key} is {value}. "
                  for key, value in zip(keys, values)]
    if items == 1:
        question = (f"What is the magic {noun} for {keys[0]} mentioned above? "
                    f"Answer with the {noun}.")
        opening = f"The magic {noun} is "
    else:
        question = (f"What are the magic {noun_plural} for all of these keys mentioned "
                    "above: " + ", ".join(keys)
                    + f"? Answer with the {noun_plural} separated by commas.")
        opening = f"The magic {noun_plural} are "
    if not answer_prefix:
        opening = ""
    question = question + ("\n" + opening if opening else "")

    filler = tokenizer(FILLER, add_special_tokens=False)["input_ids"]
    question_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
    statement_ids = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in statements]
    body_target = max(1, length - len(question_ids) - sum(len(s) for s in statement_ids) - 4)
    body = (filler * (body_target // max(1, len(filler)) + 2))[:body_target]

    cuts = sorted(rng.randrange(0, len(body)) for _ in statements)
    ids, positions, cursor = [], [], 0
    for cut, statement in sorted(zip(cuts, statement_ids), key=lambda pair: pair[0]):
        ids.extend(body[cursor:cut])
        positions.extend(range(len(ids), len(ids) + len(statement)))
        ids.extend(statement)
        cursor = cut
    ids.extend(body[cursor:])
    ids.extend(question_ids)
    prompt = ""                      # no text form needed: positions are known exactly
    return prompt, ids, sorted(set(positions)), values, keys


def completed_keys(path):
    """``(length, seed, arm)`` rows already present in an artifact, for resume.

    The 1M harness runs on a *shared* GPU, so an attempt can die at any record;
    resuming from the rows already written is what lets the sweep accumulate its
    result across many short windows instead of restarting each time.
    """
    try:
        payload = json.loads(Path(path).read_text())
    except Exception:
        return set(), []
    rows = payload.get("results") or []
    return {(r["length"], r["seed"], r["arm"]) for r in rows}, rows


def recall_of(text, values, value_style="words"):
    """Whether every planted value came back.

    `words` matches the value as a whole word (a substring hit would count
    "gondola" inside "gondolas"); `digits` compares digit runs, so separators and
    punctuation around the numbers do not decide the score.
    """
    if value_style == "digits":
        found = {run.lstrip("0") or "0" for run in re.findall(r"\d+", text)}
        return 1.0 if all((v.lstrip("0") or "0") in found for v in values) else 0.0
    low = text.lower()
    return 1.0 if all(re.search(rf"\b{re.escape(v.lower())}\b", low) for v in values) else 0.0


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
    parser.add_argument("--attn-impl", default="sdpa",
                        help="flash_attention_2 makes the exact long-context prefill "
                             "affordable: the fused kernel is what keeps a 128K-1M "
                             "prefill at hundreds of MiB instead of tens of GiB (3.34)")
    parser.add_argument("--yarn", default="auto",
                        choices=["auto", "off", "yarn", "linear", "dynamic"],
                        help="`auto` applies YaRN with factor = length / trained window, "
                             "identically to both arms; `linear` and `dynamic` are the "
                             "other position-interpolation mechanisms (dynamic NTK "
                             "rescales theta with the sequence length instead of "
                             "distorting attention temperatures, which is the variant "
                             "worth trying on a checkpoint that was never trained with "
                             "any scaling)")
    parser.add_argument("--value-style", default="digits", choices=["words", "digits"],
                        help="what the needle holds; `digits` is what this checkpoint "
                             "answers (see build_haystack for the width argument)")
    parser.add_argument("--value-digits", type=int, default=2,
                        help="digit width of a `digits` value: every digit is a separate "
                             "token here, so a wide value adds generation noise to a "
                             "retrieval measurement")
    parser.add_argument("--answer-prefix", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="open the answer for the model ('The magic numbers are'); "
                             "without it this checkpoint re-lists the question's keys, "
                             "so every arm scores zero and the row measures nothing")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    L.TOKENIZER = tokenizer
    eos = None
    done, results = completed_keys(args.out)
    if done:
        print(f"[resume] {len(done)} rows already in {args.out}", flush=True)
    for length in args.lengths:
        config = AutoConfig.from_pretrained(args.model)
        trained = int(getattr(config, "max_position_embeddings", 32768))
        factor = max(1.0, length / trained)
        scaling = "yarn" if args.yarn == "auto" else args.yarn
        if scaling != "off" and factor > 1.0:
            config.rope_scaling = {"type": scaling, "factor": factor,
                                   "original_max_position_embeddings": trained}
            # transformers 5 exposes the resolved form as `rope_parameters`
            config.rope_parameters = {"rope_type": scaling, "factor": factor,
                                      "original_max_position_embeddings": trained,
                                      "rope_theta": getattr(config, "rope_theta", 1e6)}
        config.max_position_embeddings = max(length + 1024, trained)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=config, dtype=torch.bfloat16,
            attn_implementation=args.attn_impl).to("cuda").eval()
        eos_ids = {model.config.eos_token_id} if model.config.eos_token_id is not None else set()
        print(f"[length {length}] rope_scaling={getattr(config, 'rope_scaling', None)}",
              flush=True)
        for seed in range(args.seeds):
            prompt, ids, needle_positions, values, keys = build_haystack(
                tokenizer, length, args.items, seed,
                answer_prefix=getattr(args, "answer_prefix", True),
                value_style=getattr(args, "value_style", "digits"),
                value_digits=getattr(args, "value_digits", 2))
            tensor = torch.tensor([ids], device="cuda")
            true_length = int(tensor.shape[1])
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.time()
            # maskless + fused kernel: Hugging Face then sets `is_causal`, which is
            # exactly the mask a chunk needs when it is the last n positions of the
            # cached keys, and the fused kernel keeps the prefill at a few hundred MiB
            # instead of tens of GiB (report 3.34)
            # a preallocated cache: `DynamicCache` grows by `torch.cat`, so at 1M it
            # needs the old 12.0 GiB and the new 12.0 GiB alive at once and dies in
            # `update` (measured, 22.65 GiB allocated); writing into a fixed buffer
            # also makes every append O(chunk) instead of O(sequence) - report 3.38
            cache = preallocated_cache_for(model, true_length + args.max_new + 8)
            cache, logits, captured = L.prefill_capture(model, tensor, args.obs,
                                                        args.prefill_chunk,
                                                        attention_mask=False, flash=True,
                                                        cache=cache)
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
                if (length, seed, arm) in done:
                    print(f"   {arm:<8} already measured, skipped", flush=True)
                    continue
                for layer, (ok, ov) in zip(cache.layers, originals):
                    layer.keys, layer.values = ok, ov
                kept_bytes = full_bytes
                if arm == "bounded":
                    scores = L.obs_scores(model, captured, cache, true_length, args.obs,
                                          "last", args.key_chunk)
                    # the needles are planted, so their token positions are known:
                    # the lexical-anchor scan (a full offset pass over a 128K-1M token
                    # prompt) is the expensive part of the harness and adds nothing to
                    # what this row measures - the anchor channel's job on a synthetic
                    # record is exactly to find these positions
                    anchors = needle_positions
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
                    "recall": recall_of(text, values,
                                        getattr(args, "value_style", "digits")),
                    "prediction": text, "values": values, "keys": keys,
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
                done.add((length, seed, arm))
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
