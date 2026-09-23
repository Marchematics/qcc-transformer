#!/usr/bin/env python
"""Does the TPOT ratio grow with batch, and where does 128K reach 5x?

At batch 1 the step is dominated by the model's own weight read, which no cache policy
can shrink: on Qwen2.5-0.5B the launch-free step is 3.78 ms with a 4,632-slot cache and
23.85 ms with a 1,048,576-slot one (6.31x at 1M), but only 6.84 ms at 131,072 slots
(1.81x), because 131K keys add just 3.1 ms to a step whose floor is the weights.  Serving
does not run at batch 1, so the question the 128K target really asks is how the ratio
behaves once the weight read is amortised over a batch and the cache decides the step.

This probe measures exactly that, with the same method as `probe_decode_tpot.py`: matched
arms, same operator, CUDA-graph captured (which is where the host launch cost goes), and a
preallocated cache per row.

Usage:
    python benchmarks/probe_decode_batch.py --model <checkpoint> \
        --contexts 131072 --budget 4632 --batches 1 2 4 8 \
        --out artifacts/decode-batch-<model>.json
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
from qcc_transformer.preallocated_cache import preallocated_cache_for  # noqa: E402


@torch.no_grad()
def build_cache(model, size, batch, chunk, device):
    """A preallocated cache holding exactly `size` keys for each of `batch` rows."""
    ids = torch.randint(1, 1000, (batch, size), device=device)
    cache = preallocated_cache_for(model, size + 64, device=device, batch=batch)
    cache, logits, _tails = L.prefill_capture(model, ids, obs=64, chunk=chunk,
                                              attention_mask=False, flash=True,
                                              cache=cache)
    return cache, logits[:, -1:].argmax(-1)


@torch.no_grad()
def step_latency(model, cache, first, steps, device, graph=True):
    """Wall and (optionally) launch-free ms per decode step for the cached batch."""
    kept = cache.get_seq_length()
    batch = first.shape[0]
    position = torch.tensor([kept], device=device)
    mask = torch.ones(batch, kept + 1, device=device, dtype=torch.long)
    cur = first.clone()

    def step():
        out = model(cur, **L.model_kwargs(
            model, past_key_values=cache, attention_mask=mask,
            cache_position=position,
            position_ids=position.unsqueeze(0).expand(batch, -1), use_cache=True))
        return out.logits[:, -1:].argmax(-1)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            cur.copy_(step())
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    wall = []
    for _ in range(steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        cur.copy_(step())
        torch.cuda.synchronize()
        wall.append(1000.0 * (time.perf_counter() - started))
    out = {"wall_ms_per_step": statistics.median(wall[1:]), "steps": len(wall) - 1}

    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            cur.copy_(step())
        torch.cuda.synchronize()
        capture.replay()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(max(steps, 16)):
            capture.replay()
        torch.cuda.synchronize()
        out["graph_ms_per_step"] = 1000.0 * (time.perf_counter() - started) / max(steps, 16)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="/root/qcc/models/Qwen2.5-0.5B-Instruct")
    p.add_argument("--contexts", type=int, nargs="+", default=[131072])
    p.add_argument("--budget", type=int, default=4632,
                   help="bounded-arm slots per row (the shipped 1M harness used 4,632)")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--prefill-chunk", type=int, default=8192)
    p.add_argument("--attn-impl", default="flash_attention_2")
    p.add_argument("--arms", nargs="+", default=["full", "bounded"],
                   choices=["full", "bounded"])
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    from transformers import AutoModelForCausalLM

    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn_impl).to(device).eval()
    L.TOKENIZER = None
    L.install_flash_layer_attn(model, delegate_decode=False)   # graph-capturable branch
    rows = []
    for context in args.contexts:
        for batch in args.batches:
            for arm in args.arms:
                size = context if arm == "full" else args.budget
                torch.cuda.empty_cache()
                # an arm whose cache does not fit is *recorded* as infeasible rather than
                # extrapolated: at 128K a full cache is 1.61 GiB per row, so batch 16 needs
                # 25.8 GiB for the cache alone on a 23.55 GiB card, while the bounded arm
                # needs 0.85 GiB for the whole batch
                try:
                    cache, first = build_cache(model, size, batch, args.prefill_chunk,
                                               device)
                except torch.OutOfMemoryError:
                    rows.append({"context": context, "arm": arm, "batch": batch,
                                 "status": "infeasible",
                                 "cache_bytes": size * 12288 * batch})
                    print(f"   ctx {context:>8} {arm:<8} batch {batch}  infeasible",
                          flush=True)
                    torch.cuda.empty_cache()
                    continue
                stats = step_latency(model, cache, first, args.steps, device,
                                     graph=not args.no_graph)
                rows.append({"context": context, "arm": arm, "batch": batch,
                             "status": "measured",
                             "cache_keys": int(cache.get_seq_length()), **stats})
                print(f"   ctx {context:>8} {arm:<8} batch {batch}  wall "
                      f"{stats['wall_ms_per_step']:7.2f} ms  graph "
                      f"{stats.get('graph_ms_per_step', float('nan')):7.2f} ms", flush=True)
                del cache
                torch.cuda.empty_cache()

    ratios = {}
    for context in args.contexts:
        for batch in args.batches:
            wall = {r["arm"]: r.get("wall_ms_per_step") for r in rows
                    if r["context"] == context and r["batch"] == batch}
            graph = {r["arm"]: r.get("graph_ms_per_step") for r in rows
                     if r["context"] == context and r["batch"] == batch}
            ratios[f"ctx{context}_batch{batch}"] = {
                "wall_ratio": ((wall["full"] / wall["bounded"])
                               if wall.get("full") and wall.get("bounded") else None),
                "graph_ratio": ((graph["full"] / graph["bounded"])
                                if graph.get("full") and graph.get("bounded") else None),
            }
    payload = {"config": vars(args), "results": rows, "ratios": ratios,
               "honesty": ("both arms share the operator and the capture path; the ratio "
                           "is full-KV per-step divided by bounded per-step at the same "
                           "batch")}
    Path(args.out).write_text(json.dumps(payload, indent=1))
    for key, value in ratios.items():
        def fmt(x):
            return "n/a (an arm is infeasible)" if x is None else f"{x:.2f}x"
        print(f"[{key}] wall {fmt(value['wall_ratio'])}  graph {fmt(value['graph_ratio'])}")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
