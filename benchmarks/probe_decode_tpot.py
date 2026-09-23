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


def build_cache(model, size, chunk, device):
    """A cache holding exactly `size` keys, built by an exact chunked prefill."""
    ids = torch.randint(1, 1000, (1, size), device=device)
    cache, logits, _tails = L.prefill_capture(model, ids, obs=64, chunk=chunk,
                                              attention_mask=False, flash=True)
    return cache, logits[:, -1:].argmax(-1)


@torch.no_grad()
def time_decode(model, cache, first, steps, device):
    """Per-step seconds, medians over `steps` single-token decodes."""
    kept = cache.get_seq_length()
    mask = torch.ones(1, kept + 1, device=device, dtype=torch.long)
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
        mask = torch.cat([mask, torch.ones(1, 1, device=device, dtype=torch.long)], dim=1)
    times = times[1:]                                   # drop the first (warm-up) step
    return {"median_ms": 1000.0 * statistics.median(times),
            "min_ms": 1000.0 * min(times),
            "p95_ms": 1000.0 * sorted(times)[max(0, int(0.95 * len(times)) - 1)],
            "steps": len(times)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="/root/qcc/models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[512, 1152, 4608, 32768, 131072])
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--prefill-chunk", type=int, default=8192)
    parser.add_argument("--attn-impl", default="flash_attention_2")
    parser.add_argument("--decode", choices=["flash", "sdpa", "both"], default="both")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from transformers import AutoModelForCausalLM

    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn_impl).to(device).eval()
    L.TOKENIZER = None
    rows = []
    for operator in (["flash", "sdpa"] if args.decode == "both" else [args.decode]):
        L.install_flash_layer_attn(model, delegate_decode=(operator == "flash"))
        print(f"[decode {operator}] attn_impl={args.attn_impl}", flush=True)
        for size in args.sizes:
            torch.cuda.empty_cache()
            cache, first = build_cache(model, size, args.prefill_chunk, device)
            stats = time_decode(model, cache, first, args.steps, device)
            row = {"decode": operator, "cache_keys": int(cache.get_seq_length()), **stats}
            rows.append(row)
            print(f"   {size:>8d} keys  tpot {stats['median_ms']:8.2f} ms "
                  f"(min {stats['min_ms']:.2f}, p95 {stats['p95_ms']:.2f})", flush=True)
            del cache
            torch.cuda.empty_cache()
    payload = {"config": vars(args), "results": rows}
    Path(args.out).write_text(json.dumps(payload, indent=1))
    for operator in sorted({r["decode"] for r in rows}):
        sizes = [r for r in rows if r["decode"] == operator]
        print(f"[{operator}] " + "  ".join(
            f"{r['cache_keys']}:{r['median_ms']:.2f}ms" for r in sizes))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
