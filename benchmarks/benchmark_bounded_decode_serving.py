"""Batched serving measurement: matched Full-KV vs bounded exact-KV decode.

Distinct prompts of the same length are prefilled together (chunked, exact),
each request's cache is then either kept whole (Full-KV) or compiled to a
bounded per-(layer, kv-head) set chosen from that request's own final-query
attention.  Decode is batched greedy; all rows share the same length, so no
padding is involved.

Reports per configuration: prefill seconds, decode seconds, decode tokens/s,
per-token latency, peak CUDA allocation, and whether the configuration fit.
A configuration that raises CUDA OOM is recorded as ``oom`` rather than
aborting the sweep, so the memory-limited concurrency comparison is explicit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

try:  # repo layout
    import benchmark_bounded_decode_frontier as L
except ImportError:  # authoring workspace layout
    import longctx as L

TOK = None


def build_batch(tokenizer, length, batch, base_seed):
    """One prompt replicated across the batch.

    The measurement is a latency/throughput measurement, so a homogeneous batch
    is the right control: every row has the same token length, so there is no
    padding, no ragged position mapping and no truncation of the question.  The
    reported recall is that of this prompt, evaluated on every row.
    """
    rec = L.build_record(tokenizer, length, 2, base_seed)
    rec.needle_positions, rec.answer_positions = L.find_needle_positions(tokenizer, rec)
    ids = tokenizer(rec.prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return [rec] * batch, ids.unsqueeze(0).repeat(batch, 1).cuda()


@torch.no_grad()
def prefill_batch(model, ids, obs, chunk):
    """Chunked exact prefill for a batch; captures observation-window inputs."""
    B = ids.shape[0]
    tails: dict[int, torch.Tensor] = {}
    handles = []
    for i, layer in enumerate(model.model.layers):
        def make(idx):
            def hook(module, args, kwargs, output):
                h = args[0] if args else kwargs.get("hidden_states")
                h = h.detach()
                prev = tails.get(idx)
                tails[idx] = h if prev is None else torch.cat([prev, h], dim=1)[:, -obs:, :]
            return hook
        handles.append(layer.self_attn.register_forward_hook(make(i), with_kwargs=True))
    Lc = ids.shape[1]
    cache = DynamicCache()
    last_logits = None
    try:
        for start in range(0, Lc, chunk):
            end = min(Lc, start + chunk)
            seg = ids[:, start:end]
            pos = torch.arange(start, end, device=ids.device)
            mask = torch.ones(B, end, device=ids.device, dtype=torch.long)
            o = model(seg, past_key_values=cache, attention_mask=mask,
                      position_ids=pos.unsqueeze(0).expand(B, -1), cache_position=pos, use_cache=True)
            last_logits = o.logits[:, -1:]
    finally:
        for h in handles:
            h.remove()
    return cache, last_logits, tails


@torch.no_grad()
def obs_scores_batched(model, captured, cache, Lc, obs, mode):
    """Loop rows through the single-row scoring primitive (selection is cheap)."""
    B = cache.layers[0].keys.shape[0]
    per_row = []
    for b in range(B):
        cap_b = {k: v[b:b + 1] for k, v in captured.items()}

        class _View:
            pass

        shim = _View()
        layers = []
        for layer in cache.layers:
            lyr = _View()
            lyr.keys = layer.keys[b:b + 1]
            lyr.values = layer.values[b:b + 1]
            layers.append(lyr)
        shim.layers = layers
        shim.get_seq_length = lambda: Lc
        per_row.append(L.obs_scores(model, cap_b, shim, Lc, obs, mode))
    merged = []
    for li in range(len(per_row[0])):
        merged.append(torch.cat([per_row[b][li] for b in range(B)], dim=0))
    return merged


def topk_indices_batched(scores, budget, nsink, nrecent, Lc, pool=1):
    """scores: (B, kv_heads, Lc) -> (B, kv_heads, <=budget) sorted indices."""
    s = scores.clone().float()
    if pool > 1:
        pad = (-Lc) % pool
        if pad:
            s = torch.cat([s, torch.full(s.shape[:-1] + (pad,), float("-inf"), device=s.device)], dim=-1)
        nb = s.shape[-1] // pool
        s = s.view(*s.shape[:-1], nb, pool).amax(dim=-1, keepdim=True).expand(*s.shape[:-1], nb, pool)
        s = s.reshape(*s.shape[:-3], nb * pool)[..., :Lc]
    forced = sorted(set(list(range(min(nsink, Lc))) + list(range(max(0, Lc - nrecent), Lc))))
    if forced:
        s[..., torch.tensor(forced, device=s.device)] = float("inf")
    order = torch.argsort(s, dim=-1, descending=True)[..., : min(budget, Lc)]
    return order.sort(dim=-1).values


@torch.no_grad()
def prune_batched(cache, idxs):
    """idxs: list over layers of (B, kv_heads, budget) index tensors."""
    for layer, idx in zip(cache.layers, idxs):
        B, H = idx.shape[0], idx.shape[1]
        ek = idx.unsqueeze(-1).expand(B, H, idx.shape[2], layer.keys.shape[-1])
        layer.keys = layer.keys.gather(2, ek).contiguous()
        layer.values = layer.values.gather(2, ek).contiguous()


@torch.no_grad()
def decode_batch(model, cache, next_id, Lc, max_new, eos_ids):
    B = next_id.shape[0]
    kept = cache.get_seq_length()
    mask = torch.ones(B, kept + 1, device=next_id.device, dtype=torch.long)
    gen = torch.empty(B, 0, dtype=torch.long, device=next_id.device)
    cur = next_id
    torch.cuda.synchronize()
    t0 = time.time()
    for step in range(max_new):
        cp = torch.tensor([Lc + step], device=next_id.device)
        o = model(cur, past_key_values=cache, attention_mask=mask, cache_position=cp,
                  position_ids=cp.unsqueeze(0).expand(B, -1), use_cache=True)
        tok = o.logits[:, -1:].argmax(-1)
        gen = torch.cat([gen, tok], dim=1)
        mask = torch.cat([mask, torch.ones(B, 1, device=next_id.device, dtype=torch.long)], dim=1)
        cur = tok
    torch.cuda.synchronize()
    return gen, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/qcc/models/Llama-3.2-1B-Instruct")
    ap.add_argument("--length", type=int, default=32768)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--policies", nargs="+", default=["full", "obs_last"])
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--obs", type=int, default=64)
    ap.add_argument("--nsink", type=int, default=4)
    ap.add_argument("--pool", type=int, default=7)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--prefill-chunk", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    global TOK
    tok = AutoTokenizer.from_pretrained(args.model)
    TOK = tok
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
    eos = model.config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}

    results = []
    for policy in args.policies:
        for batch in args.batches:
            recs, ids = build_batch(tok, args.length, batch, args.seed)
            Lc = ids.shape[1]
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            entry = {"policy": policy, "batch": batch, "context_tokens": Lc,
                     "budget": args.budget if policy != "full" else None}
            try:
                torch.cuda.synchronize()
                t0 = time.time()
                cache, last_logits, captured = prefill_batch(model, ids, args.obs, args.prefill_chunk)
                torch.cuda.synchronize()
                entry["prefill_s"] = round(time.time() - t0, 3)
                next_id = last_logits.argmax(-1)
                orig = [(layer.keys, layer.values) for layer in cache.layers]
                for layer, (ok, ov) in zip(cache.layers, orig):
                    layer.keys, layer.values = ok, ov
                if policy != "full":
                    scores = obs_scores_batched(model, captured, cache, Lc, args.obs, "last")
                    nrecent = max(1, int(args.budget * 0.25))
                    idxs = [topk_indices_batched(sc, args.budget, args.nsink, nrecent, Lc, args.pool)
                            for sc in scores]
                    prune_batched(cache, idxs)
                entry["kept_slots"] = int(cache.get_seq_length())
                gen, dec_s = decode_batch(model, cache, next_id, Lc, args.max_new, eos_ids)
                entry["decode_s"] = round(dec_s, 3)
                entry["decode_tokens_per_s"] = round(batch * args.max_new / dec_s, 2)
                entry["tpot_ms"] = round(1000 * dec_s / args.max_new, 3)
                entry["peak_cuda_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
                texts = [tok.decode(gen[b].tolist(), skip_special_tokens=True).strip() for b in range(batch)]
                entry["recall"] = [1.0 if recs[b].answer in texts[b] else 0.0 for b in range(batch)]
                entry["sample"] = texts[0][:40]
                entry["status"] = "ok"
                del cache, orig, captured, gen
            except torch.cuda.OutOfMemoryError as exc:
                entry["status"] = "oom"
                entry["error"] = str(exc)[:160]
            except RuntimeError as exc:
                entry["status"] = "error"
                entry["error"] = str(exc)[:200]
            torch.cuda.empty_cache()
            print(json.dumps(entry), flush=True)
            results.append(entry)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"model": args.model, "config": vars(args), "results": results}, indent=1))
    print("\n== batch sweep ==")
    for e in results:
        if e["status"] != "ok":
            print(f"  {e['policy']:9s} batch={e['batch']:>2} {e['status']}")
            continue
        print(f"  {e['policy']:9s} batch={e['batch']:>2} prefill={e['prefill_s']:>7.2f}s "
              f"decode={e['decode_s']:>6.2f}s tok/s={e['decode_tokens_per_s']:>8.2f} "
              f"TPOT={e['tpot_ms']:>7.2f}ms peak={e['peak_cuda_gib']:>6.2f}GiB "
              f"recall={sum(e['recall'])}/{len(e['recall'])}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
