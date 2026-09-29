#!/usr/bin/env python3
"""Per-step decode cost against cache size, for the two decode operators.

The batch-1 TPOT target (>= 5x Full-KV) is a statement about the part of a decode
step that scales with the cache.  It is not a statement about the whole step: a
retrofit on a small model has a fixed per-step cost (kernel launches, the LM head,
the sampler, mask bookkeeping) that no cache policy can remove, and once the cache
is bounded that fixed cost is the floor.  Reporting a ratio without the floor hides
where the ratio comes from, so this probe measures both terms directly:

    tpot(size) = floor + slope * size

by timing single-token decode against caches of several sizes, with the flash
prefill installed and either decode operator:

* ``--decode flash``   - decode hands back to the model's own forward, so with
  ``--attn-impl flash_attention_2`` it uses the fused decode kernel;
* ``--decode sdpa``    - decode runs the local ``scaled_dot_product_attention``
  branch of the patched forward (what the harness did before).

Both arms of every quality comparison share the same operator, so the choice does
not affect any retention number - it only decides how much of the TPOT ratio is
visible.

Usage:
    python benchmarks/probe_decode_tpot.py --sizes 512 1152 4608 32768 131072 \
        --steps 24 --decode flash --out artifacts/decode-tpot-flash.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import benchmark_bounded_decode_frontier as L  # noqa: E402


def build_cache(model, size, chunk, device, headroom=64):
    """A cache holding exactly `size` keys, built by an exact chunked prefill.

    The cache is preallocated.  `DynamicCache` grows by `torch.cat`, so at 1M every chunk
    copies a growing 12 GiB tensor and needs the old and the new one alive at once: the
    1M prefill measured 850.65 s with a fixed buffer in the harness, and did not finish
    in 25 minutes here with `DynamicCache` (report 3.38, 3.40).
    """
    from qcc_transformer.preallocated_cache import preallocated_cache_for

    ids = torch.randint(1, 1000, (1, size), device=device)
    # headroom must cover every append the sweep will make: each timed step appends one
    # token, and running out makes the fixed buffer grow by concatenation - which at 1M
    # needs a second 12 GiB and OOMed this probe (the shipped cache grows rather than
    # truncating, so the failure is loud but it costs the whole 1M prefill)
    cache = preallocated_cache_for(model, size + int(headroom), device=device)
    cache, logits, _tails = L.prefill_capture(model, ids, obs=64, chunk=chunk,
                                              attention_mask=False, flash=True,
                                              cache=cache)
    return cache, logits[:, -1:].argmax(-1)


@torch.no_grad()
def time_decode(model, cache, first, steps, device, use_mask=True):
    """Per-step seconds, medians over `steps` single-token decodes.

    ``use_mask=False`` passes no attention mask.  With a cache, causality and the
    absolute positions already determine the mask, so it is redundant - and it is worth
    measuring because the harness passed one and the per-step cost turned out to be
    dominated by something that does not scale with the cache (a 4,632-slot decode costs
    the same as a 131,092-slot one).
    """
    kept = cache.get_seq_length()
    mask = (torch.ones(1, kept + 1, device=device, dtype=torch.long) if use_mask else None)
    cur, times = first, []
    for step in range(steps):
        position = torch.tensor([kept + step], device=device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = model(cur, **L.model_kwargs(
            model, past_key_values=cache, attention_mask=mask, cache_position=position,
            position_ids=position.unsqueeze(0), use_cache=True))
        cur = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        if use_mask:
            mask = torch.cat([mask, torch.ones(1, 1, device=device, dtype=torch.long)],
                             dim=1)
    times = times[1:]                                   # drop the first (warm-up) step
    return {"median_ms": 1000.0 * statistics.median(times),
            "min_ms": 1000.0 * min(times),
            "p95_ms": 1000.0 * sorted(times)[max(0, int(0.95 * len(times)) - 1)],
            "steps": len(times)}


@torch.no_grad()
def profile_decode(model, cache, first, steps, device, use_mask=True):
    """Profiler kernel time per step - an upper bound, not the GPU busy time.

    A CUDA graph would give the clean number, but capture is not permitted on this
    stack's flash decode path ("operation not permitted when stream is capturing"), so
    the kernel time is summed instead.  CUPTI serializes the ~600 launches of a decode
    step, so the sum includes the bubbles it introduces: at 1,048,583 keys it reports
    83.10 ms per step against a 25.74 ms wall step.  Treat this as a ranking with a
    known inflation, and use the wall difference between cache sizes for any absolute
    statement (see `profile_exceeds_wall` in the artifact).
    """
    from torch.profiler import ProfilerActivity, profile

    kept = cache.get_seq_length()
    mask = torch.ones(1, kept + 1, device=device, dtype=torch.long) if use_mask else None
    cur = first.clone()

    def step():
        out = model(cur, **L.model_kwargs(
            model, past_key_values=cache, attention_mask=mask,
            cache_position=torch.tensor([kept], device=device),
            position_ids=torch.tensor([[kept]], device=device), use_cache=True))
        cur.copy_(out.logits[:, -1:].argmax(-1))

    for _ in range(2):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
    # `self_device_time_total` is microseconds
    total_us = sum(event.self_device_time_total for event in prof.key_averages())
    return {"gpu_ms_per_step": total_us / 1000.0 / steps, "steps": steps}


@torch.no_grad()
def time_decode_graph(model, cache, first, steps, device):
    """Per-step seconds with the decode step captured in a CUDA graph.

    This is the measurement the TPOT question needs: the eager step is host-bound (~22
    ms/step of launch and CPU against ~1-4 ms of kernel time), so a wall-clock ratio
    between cache sizes cannot show what the cache decides.  A captured graph replays the
    same step without the host, leaving GPU time.

    Capture works on the local SDPA decode branch and is refused on the flash path
    ("operation not permitted when stream is capturing"), which is why it is opt-in and
    reported separately rather than being the default.

    Length and position are held fixed across replays: the graph's shapes and addresses
    are baked in at capture, so this measures one decode step against exactly
    `cache.get_seq_length()` keys - the quantity TPOT is about.  The cache must be
    preallocated, since a growing `DynamicCache` would change shapes mid-capture.
    """
    kept = cache.get_seq_length()
    position = torch.tensor([kept], device=device)
    mask = torch.ones(1, kept + 1, device=device, dtype=torch.long)
    cur = first.clone()

    def step():
        out = model(cur, **L.model_kwargs(
            model, past_key_values=cache, attention_mask=mask,
            cache_position=position, position_ids=position.unsqueeze(0), use_cache=True))
        return out.logits[:, -1:].argmax(-1)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):                      # warm up outside the capture
        for _ in range(3):
            cur.copy_(step())
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cur.copy_(step())
    torch.cuda.synchronize()
    graph.replay()                                     # one warm replay
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        graph.replay()
    torch.cuda.synchronize()
    elapsed = 1000.0 * (time.perf_counter() - started) / steps
    return {"graph_ms_per_step": elapsed, "steps": steps}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="/root/qcc/models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[512, 1152, 4608, 32768, 131072])
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--prefill-chunk", type=int, default=8192)
    parser.add_argument("--attn-impl", default="flash_attention_2")
    parser.add_argument("--decode", choices=["flash", "sdpa", "both"], default="both")
    parser.add_argument("--passes", type=int, default=2,
                        help="repeat the wall sweep per cache size; the step is "
                             "host-bound, so a single sweep is at the edge of run-to-run "
                             "host noise (two sweeps of the same code gave 22.6 ms flat "
                             "across 512-131,072 keys and 27-28 ms with no trend)")
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=False,
                        help="also time the step captured in a CUDA graph; this is the "
                             "launch-free measurement, and it works on the sdpa decode "
                             "branch, not on the flash one")
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True,
                        help="also report a profiler kernel sum per step; it is "
                             "CUPTI-serialized and can exceed the wall clock, so the "
                             "artifact flags those rows (profile_exceeds_wall)")
    parser.add_argument("--mask", choices=["on", "off", "both"], default="both",
                        help="whether the decode step passes an explicit attention mask")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from transformers import AutoModelForCausalLM

    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn_impl).to(device).eval()
    L.TOKENIZER = None
    rows = []
    masks = [True, False] if args.mask == "both" else [args.mask == "on"]
    operators = ["flash", "sdpa"] if args.decode == "both" else [args.decode]
    # size outer, operator inner: a cache is built once and reused for every operator and
    # every pass, because building one is the expensive part - at 1M it is a 850 s exact
    # prefill, and an earlier version of this probe paid it twice (once for the wall pass,
    # once for the profiler pass)
    headroom = args.steps * max(1, args.passes) + 64 + (32 if args.graph else 0)
    for size in args.sizes:
        torch.cuda.empty_cache()
        cache, first = build_cache(model, size, args.prefill_chunk, device, headroom)
        keys = int(cache.get_seq_length())
        for operator in operators:
            L.install_flash_layer_attn(model, delegate_decode=(operator == "flash"))
            print(f"[decode {operator}] attn_impl={args.attn_impl} keys={keys}", flush=True)
            for use_mask in masks:
                for sweep in range(max(1, args.passes)):
                    stats = time_decode(model, cache, first, args.steps, device,
                                        use_mask=use_mask)
                    rows.append({"decode": operator, "mask": use_mask, "sweep": sweep,
                                 "cache_keys": keys, **stats})
                    print(f"   {size:>8d} keys mask={int(use_mask)} sweep={sweep}  "
                          f"tpot {stats['median_ms']:8.2f} ms "
                          f"(min {stats['min_ms']:.2f}, p95 {stats['p95_ms']:.2f})",
                          flush=True)
            if args.graph:
                stats = time_decode_graph(model, cache, first, max(args.steps, 32), device)
                rows.append({"decode": operator, "mask": True, "graph": True,
                             "cache_keys": keys, **stats})
                print(f"   {size:>8d} keys graph    "
                      f"{stats['graph_ms_per_step']:8.3f} ms/step", flush=True)
            if args.profile:
                stats = profile_decode(model, cache, first, 5, device)
                rows.append({"decode": operator, "mask": True, "profile": True,
                             "cache_keys": keys, **stats})
                print(f"   {size:>8d} keys profile  "
                      f"gpu {stats['gpu_ms_per_step']:8.2f} ms", flush=True)
        if getattr(cache, "overflowed", False):
            print(f"   WARNING: cache for {size} keys overflowed its buffer; the sizing "
                  f"was wrong and these numbers include a growth copy", flush=True)
            rows.append({"decode": "sizing", "mask": True, "cache_keys": keys,
                         "overflowed": True})
        del cache
        torch.cuda.empty_cache()
    # The profiler sum is CUPTI-serialized and can exceed the wall clock it is supposed
    # to decompose (measured: 83.10 ms/step of "GPU" time at 1,048,583 keys against a
    # 25.74 ms wall step, which is impossible for one step).  Flag it in the artifact
    # rather than quoting it: the wall difference between cache sizes is the quantity
    # that survives the check.
    for row in rows:
        if not row.get("profile"):
            continue
        wall = [r["median_ms"] for r in rows
                if not r.get("profile") and r.get("cache_keys") == row["cache_keys"]]
        row["wall_ms_same_cache"] = wall[0] if wall else None
        row["profile_exceeds_wall"] = bool(wall) and row["gpu_ms_per_step"] > wall[0]
    sweeps = {}
    for row in rows:
        if row.get("profile") or row.get("graph"):
            continue
        key = (row["decode"], row["mask"], row["cache_keys"])
        sweeps.setdefault(key, []).append(row["median_ms"])
    summary = {f"{op}|mask{int(mask)}|{keys}": {
        "passes": len(values), "per_pass_ms": [round(v, 2) for v in values],
        "best_ms": round(min(values), 2),
        "spread_ms": round(max(values) - min(values), 2)}
        for (op, mask, keys), values in sorted(sweeps.items(), key=lambda kv: kv[0][2])}
    payload = {"config": vars(args), "sweep_summary": summary, "results": rows,
               "honesty": ("per-step wall time is the measured quantity; the profiler "
                           "kernel sum is retained only when it is below the wall time "
                           "for the same cache (see profile_exceeds_wall)")}
    Path(args.out).write_text(json.dumps(payload, indent=1))
    def kind_of(row):
        return ("gpu" if row.get("profile") else
                "graph" if row.get("graph") else "wall")

    for operator, mask, kind in sorted({(r["decode"], r["mask"], kind_of(r))
                                        for r in rows}):
        sizes = [r for r in rows if r["decode"] == operator and r["mask"] == mask
                 and kind_of(r) == kind]
        key = ("gpu_ms_per_step" if kind == "gpu"
               else "graph_ms_per_step" if kind == "graph" else "median_ms")
        print(f"[{operator} mask={int(mask)} {kind}] " + "  ".join(
            f"{r['cache_keys']}:{r[key]:.2f}ms" for r in sizes))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
