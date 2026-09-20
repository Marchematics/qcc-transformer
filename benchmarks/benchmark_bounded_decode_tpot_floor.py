"""Attack the decode-TPOT floor: static cache and CUDA-graph decode steps.

The bounded-cache decode latency measured by the frontier harness is flat at
~13.5 ms/token from 32K to 128K while the retained KV read is tiny, so the floor
is per-step framework overhead (cache concatenation, mask construction, Python)
plus the model's weight read.

This script gives the matched Full-KV run the *same* optimizations, so a
reported speedup is not an artefact of comparing an optimized bounded path
against an unoptimized baseline.

Variants, for both the retained (bounded) set and the full KV set:
  dynamic - HF DynamicCache, mask rebuilt with torch.cat every step (the
            behaviour of benchmark_bounded_decode_frontier.py)
  static  - HF StaticCache, preallocated buffers, in-place mask
  graph   - the static path captured into a CUDA graph
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, DynamicLayer, StaticCache

try:
    import benchmark_bounded_decode_frontier as L
except ImportError:
    import longctx as L

BYTES_PER_TOKEN = 16 * 8 * 64 * 2 * 2  # Llama-3.2-1B geometry


def timeit(fn, n, warmup=1, reps=2):
    for _ in range(warmup):
        fn(n)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn(n)
    torch.cuda.synchronize()
    return (time.time() - t0) / reps / n


class Runner:
    """Decode `n` greedy steps against a fixed retained K/V set."""

    def __init__(self, model, keys, values, kept, max_new, mode):
        self.model = model
        self.kept = kept
        self.mode = mode
        self.graph = None
        self.input_ids = None
        self.cache_position = None
        self.position_ids = None
        self.logits_buf = None
        if mode == "dynamic":
            self.cache = DynamicCache()
            for k, v in zip(keys, values):
                lyr = DynamicLayer()
                lyr.keys = k.clone()
                lyr.values = v.clone()
                self.cache.layers.append(lyr)
            self.mask_buf = torch.ones(1, kept + 1, device="cuda", dtype=torch.long)
        else:
            self.cache = StaticCache(model.config, max_cache_len=kept + max_new)
            for i, layer in enumerate(self.cache.layers):
                layer.update(keys[i].clone(), values[i].clone(), i)
            self.mask_buf = torch.ones(1, kept + max_new, device="cuda", dtype=torch.long)
            self.mask_buf[:, kept + 1:] = 0

    def _reset(self):
        if self.mode == "dynamic":
            self.mask_buf = torch.ones(1, self.kept + 1, device="cuda", dtype=torch.long)
        else:
            for lyr in self.cache.layers:
                cl = getattr(lyr, "cumulative_length", None)
                if cl is None:
                    continue
                if torch.is_tensor(cl):
                    cl.fill_(self.kept)
                else:
                    lyr.cumulative_length = self.kept
            self.mask_buf[:, self.kept + 1:] = 0

    def prepare_graph(self):
        self.input_ids = torch.zeros(1, 1, dtype=torch.long, device="cuda")
        self.cache_position = torch.zeros(1, dtype=torch.long, device="cuda")
        self.position_ids = torch.zeros(1, 1, dtype=torch.long, device="cuda")
        self.logits_buf = torch.zeros(1, 1, self.model.config.vocab_size,
                                      dtype=torch.bfloat16, device="cuda")

        def step_fn():
            o = self.model(self.input_ids, past_key_values=self.cache,
                           attention_mask=self.mask_buf, cache_position=self.cache_position,
                           position_ids=self.position_ids, use_cache=True)
            self.logits_buf.copy_(o.logits)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step_fn()
        torch.cuda.current_stream().wait_stream(s)
        self._reset()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            step_fn()
        self._reset()

    def __call__(self, n, next_id, Lc):
        self._reset()
        cur = next_id
        if self.mode == "graph":
            for step in range(n):
                pos = self.kept + step
                self.input_ids.copy_(cur)
                self.cache_position.fill_(pos)
                self.position_ids.fill_(pos)
                if step + 1 < n:
                    self.mask_buf[:, pos + 1] = 1
                self.graph.replay()
                cur = self.logits_buf[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            return
        for step in range(n):
            pos = Lc + step
            cp = torch.tensor([pos], device="cuda")
            o = self.model(cur, past_key_values=self.cache, attention_mask=self.mask_buf,
                           cache_position=cp, position_ids=cp.unsqueeze(0), use_cache=True)
            cur = o.logits[:, -1:].argmax(-1)
            self.mask_buf = torch.cat(
                [self.mask_buf, torch.ones(1, 1, device="cuda", dtype=torch.long)], dim=1)
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/qcc/models/Llama-3.2-1B-Instruct")
    ap.add_argument("--length", type=int, default=32768)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--obs", type=int, default=64)
    ap.add_argument("--nsink", type=int, default=4)
    ap.add_argument("--pool", type=int, default=7)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--prefill-chunk", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--modes", nargs="+", default=["dynamic", "static"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    L.TOKENIZER = tok
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    inner = model.lm_head

    class _LastTokenLMHead(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x):
            return self.wrapped(x[:, -1:, :])

    model.lm_head = _LastTokenLMHead(inner)

    rec = L.build_record(tok, args.length, 2, args.seed)
    ids = tok(rec.prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    Lc = ids.shape[1]
    print(f"[setup] L={Lc} answer={rec.answer}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    cache, last_logits, captured = L.prefill_capture(model, ids, args.obs, args.prefill_chunk)
    next_id = last_logits.argmax(-1)
    orig = [(layer.keys, layer.values) for layer in cache.layers]
    scores = L.obs_scores(model, captured, cache, Lc, args.obs, "last")
    nrecent = max(1, int(args.budget * 0.25))
    idxs = [L.topk_indices(s, args.budget, args.nsink, nrecent, Lc, args.pool) for s in scores]
    sel_k, sel_v = [], []
    for (ok, ov), idx in zip(orig, idxs):
        H = ok.shape[1]
        ek = idx.unsqueeze(0).unsqueeze(-1).expand(1, H, idx.shape[1], ok.shape[-1])
        sel_k.append(ok.gather(2, ek).contiguous())
        sel_v.append(ov.gather(2, ek).contiguous())
    kept = sel_k[0].shape[2]
    print(f"[setup] kept={kept} peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)

    results = {"context_tokens": Lc, "budget": args.budget, "kept_slots": kept,
               "model": args.model, "max_new": args.max_new, "variants": {}}

    def record(name, state_bytes, runner):
        try:
            if name.endswith("graph"):
                runner.prepare_graph()
            tpot = timeit(lambda n: runner(n, next_id, Lc), args.max_new)
            results["variants"][name] = {"tpot_ms": round(tpot * 1000, 3), "state_bytes": state_bytes}
            print(f"  {name:22s} TPOT={tpot*1000:8.3f} ms", flush=True)
        except Exception as exc:  # noqa: BLE001
            results["variants"][name] = {"error": str(exc)[:300]}
            print(f"  {name:22s} FAILED: {str(exc)[:140]}", flush=True)

    for mode in args.modes:
        record(f"full_kv_{mode}", BYTES_PER_TOKEN * Lc,
               Runner(model, [k for k, _ in orig], [v for _, v in orig], Lc, args.max_new, mode))
        torch.cuda.empty_cache()
    for mode in args.modes:
        record(f"bounded_{mode}", BYTES_PER_TOKEN * kept,
               Runner(model, sel_k, sel_v, kept, args.max_new, mode))
        torch.cuda.empty_cache()

    fd = results["variants"].get("full_kv_dynamic", {}).get("tpot_ms")
    same = results["variants"].get("full_kv_graph") or results["variants"].get("full_kv_static")
    for name, v in results["variants"].items():
        if "tpot_ms" not in v:
            continue
        if fd:
            v["speedup_vs_full_kv_dynamic"] = round(fd / v["tpot_ms"], 3)
        if same and "tpot_ms" in same:
            v["speedup_vs_full_kv_same_optimization"] = round(same["tpot_ms"] / v["tpot_ms"], 3)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(json.dumps(results["variants"], indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
