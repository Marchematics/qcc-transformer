"""Concurrency with bounded *decode* state: sequential prefill, batched decode.

The batched sweep in benchmark_bounded_decode_serving.py prefills every request
together, so peak memory is `batch_size x full KV`, and the bounded cache buys
nothing: Full-KV and bounded both OOM at batch 8 on a 24 GiB card.  That is a
statement about prefill state, not about the retention policy.

This script measures the serving pattern the retention law actually enables:
each request is prefilled *on its own* (its full KV is transient), its cache is
then compiled down to `budget` slots per (layer, kv-head), and only the bounded
caches are kept while the next request is prefilled.  Decode is then batched
over all resident requests, which is what a continuous-batching server does.

Peak memory is therefore `one prefill transient + batch_size x bounded cache`
instead of `batch_size x full KV`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, DynamicLayer

try:
    import benchmark_bounded_decode_frontier as L
except ImportError:
    import longctx as L

BYTES_PER_TOKEN = 16 * 8 * 64 * 2 * 2


def build_one(tokenizer, length, seed):
    rec = L.build_record(tokenizer, length, 2, seed)
    rec.needle_positions, rec.answer_positions = L.find_needle_positions(tokenizer, rec)
    rec.lexical_positions = L.question_lexical_positions(tokenizer, rec.prompt, 64)
    ids = tokenizer(rec.prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    return rec, ids


@torch.no_grad()
def prefill_one(model, ids, obs, chunk):
    """Exact chunked prefill for a single request; returns cache + last logits."""
    return L.prefill_capture(model, ids, obs, chunk)


@torch.no_grad()
def decode_batched(model, batch_cache, next_ids, Lc, max_new, keep_mask=None):
    B = next_ids.shape[0]
    kept = batch_cache.layers[0].keys.shape[2]
    if keep_mask is None:
        mask = torch.ones(B, kept + 1, device="cuda", dtype=torch.long)
    else:
        mask = torch.cat([keep_mask, torch.ones(B, 1, device="cuda", dtype=torch.long)], dim=1)
    gen = torch.empty(B, 0, dtype=torch.long, device="cuda")
    cur = next_ids
    torch.cuda.synchronize()
    t0 = time.time()
    for step in range(max_new):
        cp = torch.tensor([Lc + step], device="cuda")
        o = model(cur, past_key_values=batch_cache, attention_mask=mask, cache_position=cp,
                  position_ids=cp.unsqueeze(0).expand(B, -1), use_cache=True)
        tok = o.logits[:, -1:].argmax(-1)
        gen = torch.cat([gen, tok], dim=1)
        mask = torch.cat([mask, torch.ones(B, 1, device="cuda", dtype=torch.long)], dim=1)
        cur = tok
    torch.cuda.synchronize()
    return gen, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/qcc/models/Llama-3.2-1B-Instruct")
    ap.add_argument("--length", type=int, default=32768)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--obs", type=int, default=64)
    ap.add_argument("--nsink", type=int, default=4)
    ap.add_argument("--pool", type=int, default=7)
    ap.add_argument("--dilate", type=int, default=9)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--prefill-chunk", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=7000)
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

    results = []
    for batch in args.batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        entry = {"batch": batch, "budget": args.budget, "length": args.length}
        try:
            keys, values, next_ids, recs, sizes = [], [], [], [], []
            t0 = time.time()
            for b in range(batch):
                rec, ids = build_one(tok, args.length, args.seed + 1000 * b)
                Lc = ids.shape[1]
                cache, last_logits, captured = prefill_one(model, ids, args.obs, args.prefill_chunk)
                next_id = last_logits.argmax(-1)
                scores = L.obs_scores(model, captured, cache, Lc, args.obs, "last")
                nrecent = max(1, int(args.budget * 0.25))
                idxs = [L.topk_indices(s, args.budget, args.nsink, nrecent, Lc, args.pool, args.dilate)
                        for s in scores]
                for layer, idx in zip(cache.layers, idxs):
                    merged = sorted(set(idx[0].tolist()) | {p for p in rec.lexical_positions if p < Lc})
                    midx = torch.tensor([merged], device="cuda", dtype=torch.long)
                    ek = midx.unsqueeze(0).unsqueeze(-1).expand(
                        1, layer.keys.shape[1], midx.shape[1], layer.keys.shape[-1])
                    keys.append(layer.keys.gather(2, ek).contiguous()[0])
                    values.append(layer.values.gather(2, ek).contiguous()[0])
                sizes.append(len(merged))  # one entry per request, shared by its layers
                next_ids.append(next_id)
                recs.append(rec)
                del cache, captured, last_logits, scores, idxs
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
            entry["prefill_total_s"] = round(time.time() - t0, 3)
            entry["kept_slots"] = int(max(sizes))
            # rows retain slightly different sizes (their lexical anchors differ),
            # so pad to the maximum and mask the padding out during attention
            n_layers = len(model.model.layers)
            max_kept = max(k.shape[1] for k in keys)
            keep_mask = torch.zeros(batch, max_kept, device="cuda", dtype=torch.long)
            bc = DynamicCache()
            for li in range(n_layers):
                ref = keys[li]  # entries are stored row-major: b * n_layers + layer
                kk = torch.zeros(batch, ref.shape[0], max_kept, ref.shape[-1],
                                 device="cuda", dtype=ref.dtype)
                vv = torch.zeros_like(kk)
                for b in range(batch):
                    kb = keys[b * n_layers + li]
                    vb = values[b * n_layers + li]
                    n_b = kb.shape[1]
                    kk[b, :, :n_b, :] = kb
                    vv[b, :, :n_b, :] = vb
                    keep_mask[b, :n_b] = 1
                lyr = DynamicLayer()
                lyr.keys = kk
                lyr.values = vv
                bc.layers.append(lyr)
            keep_mask = keep_mask[:, :max_kept]
            cur = torch.stack(next_ids, dim=0)
            gen, dec_s = decode_batched(model, bc, cur, Lc, args.max_new, keep_mask)
            entry["decode_s"] = round(dec_s, 3)
            entry["decode_tokens_per_s"] = round(batch * args.max_new / dec_s, 2)
            entry["tpot_ms"] = round(1000 * dec_s / args.max_new, 3)
            entry["peak_cuda_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            texts = [tok.decode(gen[b].tolist(), skip_special_tokens=True).strip() for b in range(batch)]
            entry["recall"] = [1.0 if recs[b].answer in texts[b] else 0.0 for b in range(batch)]
            entry["status"] = "ok"
            del bc, keys, values
        except Exception as exc:  # noqa: BLE001
            import traceback
            entry["status"] = "oom" if "out of memory" in str(exc).lower() else "error"
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            entry["traceback"] = traceback.format_exc()[-900:]
            for _name in ("keys", "values", "bc", "gen"):
                if _name in dir():
                    pass
            keys = values = []
            try:
                del bc
            except Exception:
                pass
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        print(json.dumps(entry), flush=True)
        results.append(entry)

    Path(args.out).write_text(json.dumps({"model": args.model, "config": vars(args),
                                          "results": results}, indent=1))
    print("\n== sequential prefill, batched bounded decode ==")
    for e in results:
        if e["status"] != "ok":
            print(f"  batch={e['batch']:>3} {e['status']} {e.get('error','')[:70]}")
            continue
        print(f"  batch={e['batch']:>3} prefill={e['prefill_total_s']:>7.2f}s "
              f"decode={e['decode_s']:>6.2f}s tok/s={e['decode_tokens_per_s']:>8.2f} "
              f"TPOT={e['tpot_ms']:>7.2f}ms peak={e['peak_cuda_gib']:>6.2f}GiB "
              f"recall={sum(e['recall'])}/{len(e['recall'])}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
