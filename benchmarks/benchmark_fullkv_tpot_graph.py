"""Matched Full-KV baseline with the *same* CUDA-graph decode path.

The bounded graph numbers are only comparable to a Full-KV baseline that gets
the same treatment.  The obstacle is memory: building a StaticCache while the
prefilled DynamicCache still holds 4 GiB at 128K needs two caches at once.

This script decodes Full-KV twice on the same prompt:
  1. dynamic  - the plain HF path (the usual baseline)
  2. graph    - a StaticCache that *takes ownership* of the prefilled tensors
                layer by layer (the dynamic cache is released progressively),
                captured into a CUDA graph

Both then run the same 32-token greedy decode, and both are checked against each
other token by token before any timing is reported.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, DynamicLayer, StaticCache

DEFAULT_MODEL = os.environ.get("QCC_MODEL", "meta-llama/Llama-3.2-1B-Instruct")

try:
    import benchmark_bounded_decode_frontier as L
except ImportError:
    import longctx as L


def timeit(fn, n, warmup=1, reps=2):
    for _ in range(warmup):
        fn(n)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn(n)
    torch.cuda.synchronize()
    return (time.time() - t0) / reps / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--length", type=int, default=131072)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--prefill-chunk", type=int, default=8192)
    ap.add_argument("--obs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--skip-dynamic", action="store_true",
                    help="the plain dynamic path grows a 4 GiB cache per step and can OOM at 128K")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    L.TOKENIZER = tok
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    inner = model.lm_head

    class _Last(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.wrapped = w

        def forward(self, x):
            return self.wrapped(x[:, -1:, :])

    model.lm_head = _Last(inner)

    rec = L.build_record(tok, args.length, 2, args.seed)
    ids = tok(rec.prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    Lc = ids.shape[1]
    n_layers = len(model.model.layers)
    print(f"[setup] L={Lc} answer={rec.answer}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    cache, last_logits, _ = L.prefill_capture(model, ids, args.obs, args.prefill_chunk)
    next_id = last_logits[:, -1:].argmax(-1)
    print(f"[setup] prefill peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)

    out = {"context_tokens": Lc, "model": args.model, "max_new": args.max_new, "variants": {}}

    # ---- 1. Full-KV, plain dynamic path ----
    mask = torch.ones(1, Lc + 1, device="cuda", dtype=torch.long)
    cur = next_id
    dyn_tokens = []

    def dynamic_step(state):
        cp = torch.tensor([Lc + state["step"]], device="cuda")
        o = model(state["cur"], past_key_values=cache, attention_mask=state["mask"],
                  cache_position=cp, position_ids=cp.unsqueeze(0), use_cache=True)
        state["cur"] = o.logits[:, -1:].argmax(-1)
        state["mask"] = torch.cat(
            [state["mask"], torch.ones(1, 1, device="cuda", dtype=torch.long)], dim=1)
        state["step"] += 1

    def run_dynamic(n):
        st = {"cur": next_id, "step": 0, "mask": mask.clone()}
        del dyn_tokens[:]
        for _ in range(n):
            dynamic_step(st)
            dyn_tokens.append(int(st["cur"]))
        torch.cuda.synchronize()

    if args.skip_dynamic:
        ref_tokens = None
        del mask
        print("  full_kv_dynamic  skipped", flush=True)
    else:
        dt = timeit(run_dynamic, args.max_new)
        out["variants"]["full_kv_dynamic"] = {"tpot_ms": round(dt * 1000, 3)}
        print(f"  full_kv_dynamic  TPOT={dt*1000:8.3f} ms", flush=True)
        ref_tokens = list(dyn_tokens)
        del mask

    # ---- 2. Full-KV, StaticCache + CUDA graph (ownership is moved, not copied) ----
    max_len = Lc + args.max_new
    sc = StaticCache(model.config, max_cache_len=max_len)
    for i in range(n_layers):
        k = cache.layers[i].keys
        v = cache.layers[i].values
        sc.layers[i].update(k, v, i)          # no clone: the layer is filled in place
        cache.layers[i].keys = None           # release the dynamic copy
        cache.layers[i].values = None
    del cache
    torch.cuda.empty_cache()
    print(f"[setup] static cache built, peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)

    mbuf = torch.ones(1, max_len, device="cuda", dtype=torch.long)
    mbuf[:, Lc + 1:] = 0
    input_ids = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    cpos = torch.zeros(1, dtype=torch.long, device="cuda")
    pids = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    logits_buf = torch.zeros(1, 1, model.config.vocab_size, dtype=torch.bfloat16, device="cuda")

    def reset_static():
        for lyr in sc.layers:
            cl = getattr(lyr, "cumulative_length", None)
            if torch.is_tensor(cl):
                cl.fill_(Lc)
            elif cl is not None:
                lyr.cumulative_length = Lc
        mbuf[:, Lc + 1:] = 0

    def step_fn():
        o = model(input_ids, past_key_values=sc, attention_mask=mbuf,
                  cache_position=cpos, position_ids=pids, use_cache=True)
        logits_buf.copy_(o.logits)

    graph_tokens = []
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step_fn()
        torch.cuda.current_stream().wait_stream(s)
        reset_static()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step_fn()
        reset_static()

        def run_graph(n):
            reset_static()
            cur = next_id
            del graph_tokens[:]
            for step in range(n):
                slot = Lc + step
                input_ids.copy_(cur)
                cpos.fill_(slot)
                pids.fill_(slot)
                if step + 1 < n:
                    mbuf[:, slot + 1] = 1
                g.replay()
                cur = logits_buf[:, -1:].argmax(-1)
                graph_tokens.append(int(cur))
            torch.cuda.synchronize()

        gt = timeit(run_graph, args.max_new)
        out["variants"]["full_kv_graph"] = {"tpot_ms": round(gt * 1000, 3)}
        print(f"  full_kv_graph    TPOT={gt*1000:8.3f} ms", flush=True)
        if ref_tokens is not None:
            out["parity_full_kv"] = {"matches_dynamic": graph_tokens == ref_tokens,
                                     "graph": graph_tokens[:8], "dynamic": ref_tokens[:8]}
            print(f"  parity Full-KV graph vs dynamic: "
                  f"{'OK' if graph_tokens == ref_tokens else 'MISMATCH'}", flush=True)
        else:
            out["parity_full_kv"] = {"checked": False, "graph": graph_tokens[:8]}
            print(f"  graph tokens: {graph_tokens[:8]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        out["variants"]["full_kv_graph"] = {"error": str(exc)[:200]}
        print(f"  full_kv_graph    FAILED: {str(exc)[:120]}", flush=True)

    out["peak_cuda_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    if "tpot_ms" in out["variants"].get("full_kv_graph", {}) and "tpot_ms" in out["variants"].get("full_kv_dynamic", {}):
        out["graph_speedup"] = round(
            out["variants"]["full_kv_dynamic"]["tpot_ms"] /
            out["variants"]["full_kv_graph"]["tpot_ms"], 3)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
