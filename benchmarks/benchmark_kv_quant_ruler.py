"""RULER quality versus *decode-state bytes*: eviction and quantization arms.

The existing comparison set for the bounded-cache claim is eviction-only
(Full-KV, sliding window, H2O, SnapKV, the retention law).  Quantization also
shrinks decode state and composes with eviction, so this runner adds those arms
and reports every arm at the byte count it actually stores, which is what a
Pareto plot of quality against decode-state bytes needs.

Arms (``--arms``)
-----------------
``full``                exact Full-KV prefill, dense bf16 cache (the baseline)
``full_int8``           the whole prefill cache quantized, int8
``full_int4``           the whole prefill cache quantized, int4
``bounded``             ``compile_bounded_cache`` at ``--budget + --lex-cap`` slots
``bounded4096_int8``    that compiled cache quantized, budget 4096 (number overrides)
``bounded4096_int4``    likewise, int4
``*_nf4``               bitsandbytes NF4 (needs bitsandbytes importable)

The arm grammar is ``full|bounded[<budget>][_int8|_int4|_nf4]``, so
``bounded8192_int4`` is a legal arm.  For bounded arms the number is the
*retention budget*; the retained width is ``budget + lex_cap``, exactly like
``RetentionConfig``.

What is measured
----------------
Per RULER record and arm: official partial recall (fraction of the record's
references present in the greedy output, case-insensitive substring), retained
slots, ``state_bytes`` of the stored decode state (packed codes + scales +
documented metadata for quantized arms; dense tensor bytes otherwise), the
dense byte count at the same slot count for reference, prefill and decode
seconds and peak GPU memory.  The per-arm summary adds means, the Pareto front
and pairs of arms whose byte counts match within ``--match-tolerance``.

Every arm runs its own exact prefill (timed), so an arm's ``prefill_s`` is its
own and peak memory is one cache at a time; a bigger ``--limit`` therefore costs
``len(arms)`` prefills per record.  ``--lex-cap`` counts towards a bounded arm's
retained width (``budget + lex_cap``, like ``RetentionConfig``); pass
``--lex-cap 0`` to compare with eviction budgets measured without anchors.

``state_bytes`` is the *stored* decode state, which is the true size of a
quantized cache.  This reference implementation has no fused int8/int4 attention
kernel, so ``dequantize_cache`` materialises a dense cache in the model dtype
before decoding: ``resident_bytes`` reports that footprint too, and decode
seconds for a quantized arm are the decode seconds of a dense cache.  Quality
and bytes are comparable across arms; decode speed is not.

Usage::

    python benchmarks/benchmark_kv_quant_ruler.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --ruler-jsonl ruler_subset.jsonl \
        --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt \
        --limit 4 --budget 4096 --group-size 128 --max-new 64 \
        --out artifacts/bounded-decode-frontier-kv-quant.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kv_quant as KQ  # noqa: E402  (benchmarks/ is on sys.path above)
import os
from qcc_transformer.hf_loading import _ensure_remote_code_compat, load_hf_causal_lm  # noqa: E402
from qcc_transformer.retention import (RetentionConfig, compile_bounded_cache,  # noqa: E402
                                       fixed_rope_length, prefill_capture)

DEFAULT_MODEL = os.environ.get("QCC_MODEL", "meta-llama/Llama-3.2-1B-Instruct")

ARM_PATTERN = re.compile(r"^(full|bounded(\d+)?)(?:_(int8|int4|nf4))?$")
DEFAULT_ARMS = ["full", "full_int8", "full_int4", "bounded4096_int8", "bounded4096_int4"]
SCALE_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


class _LastTokenLMHead(torch.nn.Module):
    """LM head that only projects the final position (as the harnesses do).

    Without it the last prefill chunk's ``(1, chunk, vocab)`` logits stay alive
    behind the ``logits[:, -1:]`` view that the callers keep -- several GiB at a
    32K prompt with a 128K vocabulary.
    """

    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, x):
        return self.wrapped(x[:, -1:, :])


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------


def parse_arm(spec: str, default_budget: int) -> tuple[str, int | None, str | None]:
    """``"bounded4096_int8"`` -> ``("bounded", 4096, "int8")``.

    ``"full"`` -> ``("full", None, None)``; a bounded arm without a number uses
    ``default_budget`` (``--budget``).
    """
    match = ARM_PATTERN.match(spec.strip().replace("-", "_"))
    if not match:
        raise SystemExit(
            f"unrecognised arm {spec!r}; expected full|bounded[<budget>][_int8|_int4|_nf4]")
    policy = "full" if match.group(1) == "full" else "bounded"
    mode = match.group(3)
    budget = None
    if policy.startswith("bounded"):
        budget = int(match.group(2)) if match.group(2) else int(default_budget)
    return policy, budget, mode


# ---------------------------------------------------------------------------
# data and scoring
# ---------------------------------------------------------------------------


def load_records(path, tasks, limit, max_prompt_tokens, tokenizer, answer_prefix=True):
    """RULER JSONL rows -> ``(row, prompt, prompt_tokens)``, ``limit`` per task."""
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
        prompt = row["input"] + (row.get("answer_prefix", "") if answer_prefix else "")
        tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        if max_prompt_tokens is not None and tokens > max_prompt_tokens:
            continue
        counts[task] += 1
        picked.append((row, prompt, tokens))
    return picked


def partial_recall(outputs, text):
    """Official RULER partial recall: fraction of references present (case-insensitive)."""
    references = [out for out in outputs if out]
    if not references:
        return 0.0
    lowered = text.lower()
    return sum(1.0 for out in references if out.lower() in lowered) / len(references)


def strict_recall(outputs, text):
    references = [out for out in outputs if out]
    if not references:
        return 0.0
    lowered = text.lower()
    return 1.0 if all(out.lower() in lowered for out in references) else 0.0


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


@torch.no_grad()
def greedy(model, cache, first, length, max_new, eos_ids, forward_kwargs=("cache_position",),
           device="cuda"):
    """The harness decode loop: absolute positions, greedy, stop at EOS."""
    kept = cache.get_seq_length()
    mask = torch.ones(1, kept + 1, device=first.device, dtype=torch.long)
    generated, current, start = [], first, time.time()
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
        mask = torch.cat([mask, torch.ones(1, 1, device=first.device, dtype=torch.long)], dim=-1)
        current = token
        if int(token) in eos_ids:
            break
    if device == "cuda":
        torch.cuda.synchronize()
    return generated, time.time() - start


def forward_kwargs_of(model):
    from qcc_transformer.retention import forward_accepts
    return ("cache_position",) if forward_accepts(model, "cache_position") else ()


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _peak_gib(device):
    if device != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated() / 2**30, 3)


# ---------------------------------------------------------------------------
# one record / one arm
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_arm(model, tokenizer, ids, row, spec, args, config, eos_ids, forward_kwargs):
    """Prefill, optionally prune + quantize, decode; returns one result row."""
    policy, budget, mode = parse_arm(spec, args.budget)
    device = args.device
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _sync(device)
    start = time.time()
    if policy == "full":
        cache, logits, _captured = prefill_capture(
            model, ids, config.observation_window, config.prefill_chunk)
    else:
        cache, logits = compile_bounded_cache(model, ids, config, tokenizer=tokenizer)
    _sync(device)
    prefill_seconds = time.time() - start
    kept = int(cache.get_seq_length())
    prefill_peak = _peak_gib(device)
    dense_bytes = KQ.state_bytes(cache)
    # reference size of the *whole* dense cache at this slot count, every layer
    full_bytes = sum(KQ.state_bytes_full(layer.keys.shape, layer.keys.dtype)
                     for layer in cache.layers)

    quantized = None
    if mode is not None:
        quantized = KQ.quantize_cache(cache, mode, args.group_size,
                                      scale_dtype=SCALE_DTYPES[args.scale_dtype],
                                      compress_statistics=args.nf4_compress_statistics)
        del cache                       # the dense prefill cache is not the decode state
        if device == "cuda":
            torch.cuda.empty_cache()
        cache = KQ.dequantize_cache(quantized)
    stored_bytes = KQ.state_bytes(quantized) if quantized is not None else dense_bytes
    resident_bytes = KQ.state_bytes(cache)

    generated, decode_seconds = greedy(model, cache, logits[:, -1:].argmax(-1), int(ids.shape[1]),
                                      args.max_new, eos_ids, forward_kwargs, device)
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    outputs = row.get("outputs", [])
    return {
        "row": None, "task": row.get("task"), "ruler_index": row.get("index"),
        "nominal_length": row.get("length"), "prompt_tokens": int(ids.shape[1]),
        "arm": spec, "policy": policy, "budget": budget, "mode": mode,
        "kept_slots": kept, "slots_per_token": round(kept / max(1, int(ids.shape[1])), 4),
        "state_bytes": stored_bytes, "state_mib": round(stored_bytes / 2**20, 3),
        "dense_state_bytes": dense_bytes,
        "full_state_bytes": full_bytes,
        "compression_vs_full": round(full_bytes / max(1, stored_bytes), 3),
        "quantization_ratio": round(dense_bytes / max(1, stored_bytes), 3),
        "resident_bytes": resident_bytes,
        "prediction": text, "generated": len(generated), "outputs": outputs,
        "recall_partial": round(partial_recall(outputs, text), 4),
        "recall_strict": round(strict_recall(outputs, text), 4),
        "prefill_s": round(prefill_seconds, 3), "decode_s": round(decode_seconds, 3),
        "decode_tokens_per_s": round(len(generated) / decode_seconds, 2) if decode_seconds else None,
        "prefill_peak_gib": prefill_peak, "peak_gib": _peak_gib(device),
        "status": "ok", "error": None,
    }


def error_row(row, spec, args, message):
    policy, budget, mode = parse_arm(spec, args.budget)
    return {
        "row": None, "task": row.get("task") if row else None,
        "ruler_index": row.get("index") if row else None,
        "nominal_length": row.get("length") if row else None, "prompt_tokens": None,
        "arm": spec, "policy": policy, "budget": budget, "mode": mode,
        "kept_slots": None, "slots_per_token": None, "state_bytes": None,
        "state_mib": None, "dense_state_bytes": None, "full_state_bytes": None,
        "compression_vs_full": None, "quantization_ratio": None, "resident_bytes": None,
        "prediction": None, "generated": 0, "outputs": row.get("outputs", []) if row else [],
        "recall_partial": 0.0, "recall_strict": 0.0, "prefill_s": None, "decode_s": None,
        "decode_tokens_per_s": None, "prefill_peak_gib": None, "peak_gib": _peak_gib(args.device),
        "status": "error", "error": message,
    }


# ---------------------------------------------------------------------------
# summary: means, Pareto front, matched-byte pairs
# ---------------------------------------------------------------------------


def _mean(values):
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 4) if values else None


def _arm_stats(rows):
    ok = [row for row in rows if row["status"] == "ok"]
    if not ok:
        return {"records": 0, "errors": len(rows)}
    return {
        "records": len(ok),
        "errors": len(rows) - len(ok),
        "mean_partial": _mean([r["recall_partial"] for r in ok]),
        "mean_strict": _mean([r["recall_strict"] for r in ok]),
        "mean_kept_slots": _mean([r["kept_slots"] for r in ok]),
        "mean_state_bytes": int(_mean([r["state_bytes"] for r in ok])),
        "mean_state_mib": _mean([r["state_mib"] for r in ok]),
        "mean_dense_state_bytes": int(_mean([r["dense_state_bytes"] for r in ok])),
        "mean_compression_vs_full": _mean([r["compression_vs_full"] for r in ok]),
        "mean_prefill_s": _mean([r["prefill_s"] for r in ok]),
        "mean_decode_s": _mean([r["decode_s"] for r in ok]),
        "mean_peak_gib": _mean([r["peak_gib"] for r in ok]),
        "mean_prompt_tokens": _mean([r["prompt_tokens"] for r in ok]),
    }


def pareto_front(points):
    """Non-dominated ``{arm, bytes, score}`` points, cheapest first.

    Point A dominates B when A stores no more bytes and scores at least as well,
    strictly better in one of the two.
    """
    front = []
    for point in sorted(points, key=lambda p: (p["state_bytes"], -p["score"])):
        if any(other["state_bytes"] <= point["state_bytes"] and other["score"] >= point["score"]
               and (other["state_bytes"] < point["state_bytes"]
                    or other["score"] > point["score"])
               for other in points):
            continue
        front.append(point)
    return front


def matched_pairs(stats, tolerance):
    """Pairs of arms whose mean state bytes agree within ``tolerance`` (relative)."""
    arms = [(arm, data) for arm, data in stats.items() if data.get("mean_state_bytes")]
    pairs = []
    for i, (arm_a, data_a) in enumerate(arms):
        for arm_b, data_b in arms[i + 1:]:
            bytes_a, bytes_b = data_a["mean_state_bytes"], data_b["mean_state_bytes"]
            if abs(bytes_a - bytes_b) / max(bytes_a, bytes_b) <= tolerance:
                pairs.append({
                    "arms": [arm_a, arm_b],
                    "state_bytes": [bytes_a, bytes_b],
                    "mean_partial": [data_a["mean_partial"], data_b["mean_partial"]],
                    "score_delta": round((data_b["mean_partial"] or 0.0)
                                         - (data_a["mean_partial"] or 0.0), 4),
                })
    return pairs


def summarize(results, args, task_names):
    by_arm, by_task_arm = defaultdict(list), defaultdict(list)
    for entry in results:
        by_arm[entry["arm"]].append(entry)
        by_task_arm[(entry["task"], entry["arm"])].append(entry)

    aggregate = {arm: _arm_stats(rows) for arm, rows in by_arm.items()}
    per_task = {}
    for task in task_names:
        task_rows = [entry for entry in results if entry["task"] == task]
        per_task[task] = {arm: _arm_stats([entry for entry in task_rows if entry["arm"] == arm])
                          for arm in by_arm}

    def points(stats):
        return [{"arm": arm, "state_bytes": data["mean_state_bytes"],
                 "score": data["mean_partial"], "kept_slots": data["mean_kept_slots"]}
                for arm, data in stats.items()
                if data.get("mean_state_bytes") and data.get("mean_partial") is not None]

    return {
        "aggregate": aggregate,
        "tasks": per_task,
        "pareto": {"aggregate": pareto_front(points(aggregate)),
                   "per_task": {task: pareto_front(points(stats))
                                for task, stats in per_task.items()}},
        "matched_bytes": {
            "tolerance": args.match_tolerance,
            "aggregate": matched_pairs(aggregate, args.match_tolerance),
            "per_task": {task: matched_pairs(stats, args.match_tolerance)
                         for task, stats in per_task.items()},
        },
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--label", default=None)
    parser.add_argument("--ruler-jsonl", default=os.environ.get("QCC_RULER_JSONL", ""),
                        help="RULER split JSONL (env: QCC_RULER_JSONL)")
    parser.add_argument("--tasks", nargs="*",
                        default=["niah_single_1", "niah_multikey_2", "niah_multikey_3", "vt"])
    parser.add_argument("--limit", type=int, default=None, help="records per task")
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--budget", type=int, default=4096,
                        help="retention budget for bounded arms (width = budget + lex_cap)")
    parser.add_argument("--arms", nargs="*", default=DEFAULT_ARMS)
    parser.add_argument("--group-size", type=int, default=128,
                        help="int4 group size (and bitsandbytes blocksize for nf4)")
    parser.add_argument("--scale-dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--nf4-compress-statistics", action="store_true",
                        help="bitsandbytes nested absmax quantization for nf4")
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lex-cap", type=int, default=512)
    parser.add_argument("--hops", type=int, default=6)
    parser.add_argument("--obs", type=int, default=64)
    parser.add_argument("--nsink", type=int, default=4)
    parser.add_argument("--pool", type=int, default=7)
    parser.add_argument("--dilate", type=int, default=9)
    parser.add_argument("--scoring", default="last")
    parser.add_argument("--prefill-chunk", type=int, default=8192)
    parser.add_argument("--key-chunk", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-answer-prefix", action="store_true")
    parser.add_argument("--match-tolerance", type=float, default=0.15,
                        help="relative byte gap that still counts as matched decode state")
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.ruler_jsonl:
        raise SystemExit("--ruler-jsonl is required (or set QCC_RULER_JSONL): "
                         "point it at the RULER split JSONL")
    if args.trust_remote_code:
        _ensure_remote_code_compat()
    from transformers import AutoTokenizer

    scale_dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    arms = [parse_arm(spec, args.budget) for spec in args.arms]
    for spec, (_policy, _budget, mode) in zip(args.arms, arms):
        if mode is not None:
            try:
                KQ.check_mode(mode)
            except (RuntimeError, ValueError) as exc:
                raise SystemExit(f"arm {spec!r} is unavailable: {exc}") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=args.trust_remote_code)
    model = load_hf_causal_lm(args.model, dtype=args.dtype, device=args.device,
                              trust_remote_code=args.trust_remote_code,
                              attn_implementation=args.attn_impl).eval()
    model.lm_head = _LastTokenLMHead(model.lm_head)
    config = RetentionConfig(
        budget=args.budget, observation_window=args.obs, attention_sinks=args.nsink,
        pool=args.pool, dilate=args.dilate, lex_cap=args.lex_cap, chain_hops=args.hops,
        prefill_chunk=args.prefill_chunk, key_chunk=args.key_chunk, scoring=args.scoring)
    eos_ids = set()
    for source in (model.config.eos_token_id,
                   getattr(getattr(model, "generation_config", None), "eos_token_id", None)):
        if source is None:
            continue
        eos_ids |= set(source) if isinstance(source, (list, tuple, set)) else {source}
    forward_kwargs = forward_kwargs_of(model)

    records = load_records(args.ruler_jsonl, args.tasks, args.limit,
                           args.max_prompt_tokens, tokenizer,
                           answer_prefix=not args.no_answer_prefix)
    task_names = sorted({row.get("task") for row, _p, _t in records})
    print(f"[model] {args.model} records={len(records)} tasks={task_names} "
          f"arms={args.arms} budget={args.budget}+{args.lex_cap} "
          f"group_size={args.group_size} device={args.device}", flush=True)

    results = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for index, (row, prompt, _tokens) in enumerate(records):
        ids = tokenizer(prompt, return_tensors="pt",
                        add_special_tokens=False).input_ids.to(args.device)
        with fixed_rope_length(model, int(ids.shape[1])) as pinned:
            for spec in args.arms:
                try:
                    entry = run_arm(model, tokenizer, ids, row, spec, args, config,
                                    eos_ids, forward_kwargs)
                except Exception as exc:  # noqa: BLE001 - one bad arm must not kill the sweep
                    import traceback
                    message = f"{type(exc).__name__}: {str(exc)[:200]}"
                    entry = error_row(row, spec, args, message)
                    entry["traceback"] = traceback.format_exc()[-600:]
                entry["row"] = index
                entry["pinned_rope"] = bool(pinned)
                results.append(entry)
                state = (f"{entry['state_mib']:.1f} MiB" if entry["state_bytes"]
                         else "n/a")
                print(f"[{index:>3d}/{len(records)}] {str(entry['task']):<16} "
                      f"L={entry['prompt_tokens'] or 0:>6} {spec:<20} {entry['status']:<5} "
                      f"kept={str(entry['kept_slots']):>6} bytes={state:>12} "
                      f"partial={entry['recall_partial']:.2f} "
                      f"pred={str(entry['prediction'])[:28]!r}", flush=True)
                del entry
                if args.device == "cuda":
                    torch.cuda.empty_cache()
        if (index + 1) % 5 == 0:      # incremental checkpoint for an interrupted sweep
            out_path.write_text(json.dumps(
                {"model": args.model, "config": vars(args), "partial": True,
                 "records": results}, indent=1), encoding="utf-8")

    summary = summarize(results, args, task_names)
    payload = {"model": args.model, "label": args.label or Path(args.model).name,
               "ruler": args.ruler_jsonl, "config": vars(args),
               "arms": args.arms, "summary": summary, "records": results}
    out_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print("\n== quality vs decode-state bytes (mean over records) ==")
    header = f"  {'arm':<22}{'score':>8}{'strict':>8}{'slots':>9}{'state MiB':>11}{'x full':>8}"
    print(header)
    for arm in args.arms:
        data = summary["aggregate"].get(arm, {})
        if not data.get("records"):
            print(f"  {arm:<22}{'--':>8}{'--':>8}{'--':>9}{'--':>11}{'--':>8}"
                  f"  ({data.get('errors', 0)} errors)")
            continue
        print(f"  {arm:<22}{data['mean_partial']:>8.3f}{data['mean_strict']:>8.3f}"
              f"{data['mean_kept_slots']:>9.0f}{data['mean_state_mib']:>11.2f}"
              f"{data['mean_compression_vs_full']:>8.2f}x")
    print("\n== Pareto front (aggregate) ==")
    for point in summary["pareto"]["aggregate"]:
        print(f"  {point['arm']:<22} {point['state_bytes'] / 2**20:>9.2f} MiB "
              f"score={point['score']:.3f} slots={point['kept_slots']:.0f}")
    for task, front in summary["pareto"]["per_task"].items():
        rendered = ", ".join(f"{p['arm']}@{p['state_bytes'] / 2**20:.1f}MiB={p['score']:.2f}"
                             for p in front)
        print(f"  [task {task}] {rendered}")
    matched = summary["matched_bytes"]["aggregate"]
    if matched:
        print("\n== matched decode-state bytes (within "
              f"{args.match_tolerance:.0%}) ==")
        for pair in matched:
            print(f"  {pair['arms'][0]:<22} vs {pair['arms'][1]:<22} "
                  f"bytes={pair['state_bytes']} score_delta={pair['score_delta']:+.3f}")
    print("wrote", args.out)
    return summary


if __name__ == "__main__":
    main()
