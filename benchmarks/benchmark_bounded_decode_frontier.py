"""Long-context frontier harness: bounded exact-KV decode with causal selection.

Extends frontier.py with:
  * O(obs) hidden-state capture via attention forward hooks, so prefill memory
    does not grow with the number of stored hidden states;
  * last-token-only LM head, avoiding an O(L * vocab) logits tensor;
  * multi-record aggregation so quality is reported over several seeds.

Protocol
--------
Prefill is exact Full-KV (the pretrained model's own causal attention).
Selection runs once, at the end of the prompt, using only the observation
window (the last `obs` prompt tokens, which contain the question) -> causal.
Decode then runs greedily against the bounded per-(layer, kv-head) cache.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

FILLER = [
    "The grass is green and the sky is blue in the quiet valley.",
    "A gentle breeze moved through the tall trees along the river.",
    "Researchers recorded the daily temperature in a small notebook.",
    "The old library held thousands of volumes on wooden shelves.",
    "Every morning the baker opened the shop before sunrise.",
    "Sailors told stories about distant islands and calm seas.",
    "The mountain trail was long but the view from the top was clear.",
    "Children played in the courtyard while the clock struck noon.",
]
KEYS = [
    "amber", "cobalt", "dahlia", "ember", "fable", "garnet", "harbor",
    "indigo", "jasmine", "kelvin", "lumen", "marble", "nectar", "onyx",
    "pepper", "quartz", "raven", "saffron", "topaz", "umber",
]

TOKENIZER = None


@dataclass
class Record:
    prompt: str
    question_key: str
    answer: str
    needle_positions: list[int] = field(default_factory=list)
    answer_positions: list[int] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def build_record(tokenizer, target_tokens, num_pairs, seed):
    rng = random.Random(seed)
    keys = rng.sample(KEYS, num_pairs)
    values = {k: str(rng.randint(1_000_000, 9_999_999)) for k in keys}
    question_key = rng.choice(keys)
    depressions = sorted(rng.uniform(0.05, 0.95) for _ in range(num_pairs))
    question = (f"\n\nWhat is the special magic number for {question_key}? "
                f"Answer with the number only.\nAnswer:")
    n_sent = max(16, int(target_tokens / 12))
    prompt = ""
    for _ in range(6):
        body = " ".join(rng.choice(FILLER) for _ in range(n_sent))
        pieces, cursor = [], 0
        for depth, k in zip(depressions, keys):
            cut = int(len(body) * depth)
            pieces.append(body[cursor:cut])
            pieces.append(f" One of the special magic numbers for {k} is: {values[k]}.")
            cursor = cut
        pieces.append(body[cursor:])
        prompt = "".join(pieces) + question
        actual = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        if abs(actual - target_tokens) <= max(64, 0.03 * target_tokens):
            break
        n_sent = max(16, int(n_sent * target_tokens / max(1, actual)))
    return Record(prompt=prompt, question_key=question_key, answer=values[question_key],
                  meta={"all_pairs": values, "num_pairs": num_pairs, "seed": seed,
                        "target_tokens": target_tokens})


def find_needle_positions(tokenizer, record):
    enc = tokenizer(record.prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc.offset_mapping
    all_positions, answer_positions = set(), []
    cursor = 0
    for k, v in record.meta["all_pairs"].items():
        needle = f"One of the special magic numbers for {k} is: {v}."
        start = record.prompt.find(needle, cursor)
        if start < 0:
            start = record.prompt.find(needle)
        if start < 0:
            continue
        end = start + len(needle)
        cursor = end
        all_positions.update(i for i, (a, b) in enumerate(offsets) if b > start and a < end)
        if k == record.question_key:
            vstart = record.prompt.find(v, start, end)
            answer_positions = [i for i, (a, b) in enumerate(offsets)
                                if b > vstart and a < vstart + len(v)]
    return sorted(all_positions), sorted(answer_positions)


def apply_rope(q, cos, sin):
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    return q * cos + rotate_half(q) * sin


@torch.no_grad()
def prefill_capture(model, ids, obs, chunk):
    """Exact causal prefill in bounded chunks.

    Feeding the prompt in fixed chunks with a growing KV cache and absolute
    position ids is algebraically the same causal prefill as one long forward
    pass, but keeps activation memory O(chunk) instead of O(L^2) (no explicit
    quadratic causal mask).  Only the observation-window attention inputs are
    retained per layer, so hidden-state storage stays O(obs * d).
    """
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
    L = ids.shape[1]
    cache = DynamicCache()
    last_logits = None
    try:
        for start in range(0, L, chunk):
            end = min(L, start + chunk)
            seg = ids[:, start:end]
            pos = torch.arange(start, end, device=ids.device)
            mask = torch.ones(1, end, device=ids.device, dtype=torch.long)
            o = model(seg, past_key_values=cache, attention_mask=mask,
                      position_ids=pos.unsqueeze(0), cache_position=pos, use_cache=True)
            last_logits = o.logits[:, -1:]
    finally:
        for h in handles:
            h.remove()
    return cache, last_logits, tails


@torch.no_grad()
def obs_scores(model, captured, cache, L, obs, mode, key_chunk=16384):
    """Per (layer, kv-head) key importance from the observation-window queries.

    Computed in key chunks so peak memory is O(obs * key_chunk) per layer.
    mode="last" ranks by the raw final-query score (the softmax denominator is
    constant across keys, so ranking is unchanged); "mean"/"max" run an online
    two-pass softmax over the whole key axis.
    """
    num_heads = model.config.num_attention_heads
    kv_heads = cache.layers[0].keys.shape[1]
    group = num_heads // kv_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    key_pos = torch.arange(L, device=cache.layers[0].keys.device)
    NEG = -1e30
    out = []
    for layer_idx, layer in enumerate(cache.layers):
        h = captured[layer_idx]
        n_obs = h.shape[1]
        attn = model.model.layers[layer_idx].self_attn
        q = attn.q_proj(h[0]).view(n_obs, num_heads, head_dim).transpose(0, 1)
        pos = torch.arange(L - n_obs, L, device=h.device)
        cos, sin = model.model.rotary_emb(h, pos.unsqueeze(0))
        q = apply_rope(q, cos[0].unsqueeze(0), sin[0].unsqueeze(0))
        k = layer.keys[0]

        def block_scores(ks, ke):
            qq = q if mode != "last" else q[:, -1:, :]
            qg = qq.reshape(kv_heads, group, qq.shape[1], head_dim)
            sc = torch.einsum("hgod,hld->hgol", qg.float(), k[:, ks:ke].float()) * attn.scaling
            valid = key_pos[ks:ke][None, :] <= pos[:, None]
            if mode == "last":
                valid = valid[-1:, :]
            return sc, valid

        if mode == "last":
            parts = []
            for ks in range(0, L, key_chunk):
                ke = min(L, ks + key_chunk)
                sc, valid = block_scores(ks, ke)
                sc = sc.masked_fill(~valid[None, None, :, :], NEG).amax(dim=1)  # (kv, 1, blk)
                parts.append(sc[:, 0, :])
            sc = torch.cat(parts, dim=1)
        else:
            qf = q.reshape(kv_heads, group, n_obs, head_dim)
            m = torch.full((kv_heads, group, n_obs), NEG, device=h.device)
            d = torch.zeros((kv_heads, group, n_obs), device=h.device)
            for ks in range(0, L, key_chunk):
                ke = min(L, ks + key_chunk)
                sc = torch.einsum("hgod,hld->hgol", qf.float(), k[:, ks:ke].float()) * attn.scaling
                valid = key_pos[ks:ke][None, :] <= pos[:, None]
                sc = sc.masked_fill(~valid[None, None, :, :], NEG)
                bm = sc.amax(dim=-1)
                nm = torch.maximum(m, bm)
                p = (sc - nm[..., None]).masked_fill(~valid[None, None, :, :], float("-inf"))
                d = d * torch.exp(m - nm) + torch.exp(p).sum(dim=-1)
                m = nm
            acc = torch.empty((kv_heads, L), device=h.device)
            for ks in range(0, L, key_chunk):
                ke = min(L, ks + key_chunk)
                sc = torch.einsum("hgod,hld->hgol", qf.float(), k[:, ks:ke].float()) * attn.scaling
                valid = key_pos[ks:ke][None, :] <= pos[:, None]
                sc = sc.masked_fill(~valid[None, None, :, :], NEG)
                w = torch.exp(sc - m[..., None]) / d[..., None]
                w = w.masked_fill(~valid[None, None, :, :], 0.0)
                acc[:, ks:ke] = w.mean(dim=(1, 2)) if mode == "mean" else w.amax(dim=(1, 2))
            sc = acc
        out.append(sc)
    return out


def keynorm_scores(cache):
    return [layer.keys[0].float().norm(dim=-1) for layer in cache.layers]


def topk_indices(scores_layer, budget, nsink, nrecent, L, pool=1):
    kv_heads = scores_layer.shape[0]
    idxs = []
    forced = sorted(set(list(range(min(nsink, L))) + list(range(max(0, L - nrecent), L))))
    for head in range(kv_heads):
        s = scores_layer[head].clone().float()
        if pool > 1:
            pad = (-L) % pool
            if pad:
                s = torch.cat([s, torch.full((pad,), float("-inf"), device=s.device)])
            nb = s.numel() // pool
            s = s.view(nb, pool).amax(dim=1, keepdim=True).expand(nb, pool).reshape(-1)[:L]
        if forced:
            s[torch.tensor(forced, device=s.device)] = float("inf")
        order = torch.argsort(s, descending=True)
        idxs.append(sorted(order[: min(budget, L)].tolist()))
    return torch.tensor(idxs, device=scores_layer.device, dtype=torch.long)


def selection_stats(idxs, positions, L):
    if not positions or idxs is None:
        return None
    device = idxs[0].device
    pos = torch.tensor([p for p in positions if p < L], dtype=torch.long, device=device)
    if pos.numel() == 0:
        return None
    fracs, units_all = [], 0
    for idx in idxs:
        for head in range(idx.shape[0]):
            hit = torch.isin(pos, idx[head]).float().mean().item()
            fracs.append(hit)
            units_all += int(hit == 1.0)
    return {"units": len(fracs), "units_with_all_answer_tokens": units_all,
            "answer_token_keep_fraction": sum(fracs) / len(fracs),
            "all_answer_unit_fraction": units_all / len(fracs)}


def build_idxs(policy, scores, cache, record, budget, nsink, nrecent, L, pool, device):
    if policy == "full":
        return None
    if policy == "recent":
        sel = sorted(range(max(0, L - budget), L))
        return [torch.tensor([sel] * layer.keys.shape[1], device=device) for layer in cache.layers]
    if policy == "sink_recent":
        sel = sorted(set(list(range(min(nsink, L))) + list(range(max(0, L - (budget - nsink)), L))))
        return [torch.tensor([sel] * layer.keys.shape[1], device=device) for layer in cache.layers]
    if policy == "oracle_needle":
        keep = set(record.needle_positions)
        sel = sorted(set(list(range(min(nsink, L))) + list(range(max(0, L - nrecent), L))
                         + [p for p in keep if p < L]))
        return [torch.tensor([sel] * layer.keys.shape[1], device=device) for layer in cache.layers]
    if policy == "random":
        g = torch.Generator(device="cpu").manual_seed(1234)
        sel = sorted(torch.randperm(L, generator=g)[:budget].tolist())
        return [torch.tensor([sel] * layer.keys.shape[1], device=device) for layer in cache.layers]
    return [topk_indices(s, budget, nsink, nrecent, L, pool) for s in scores]


@torch.no_grad()
def decode(model, cache, next_id, L, max_new, eos_ids):
    dev = next_id.device
    kept = cache.get_seq_length()
    mask = torch.ones(1, kept + 1, device=dev, dtype=torch.long)
    gen, cur, t0 = [], next_id, time.time()
    for step in range(max_new):
        cp = torch.tensor([L + step], device=dev)
        o = model(cur, past_key_values=cache, attention_mask=mask,
                  cache_position=cp, position_ids=cp.unsqueeze(0), use_cache=True)
        tok = o.logits[:, -1:].argmax(-1)
        gen.append(tok.item())
        mask = torch.cat([mask, torch.ones(1, 1, device=dev, dtype=torch.long)], dim=1)
        cur = tok
        if tok.item() in eos_ids:
            break
    return TOKENIZER.decode(gen, skip_special_tokens=True).strip(), len(gen), time.time() - t0


def run_one(model, tokenizer, length, args, eos_ids, seed):
    global TOKENIZER
    TOKENIZER = tokenizer
    rec = build_record(tokenizer, length, args.pairs, seed)
    rec.needle_positions, rec.answer_positions = find_needle_positions(tokenizer, rec)
    ids = tokenizer(rec.prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    L = ids.shape[1]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    cache, last_logits, captured = prefill_capture(model, ids, args.obs, args.prefill_chunk)
    torch.cuda.synchronize()
    prefill_s = time.time() - t0
    next_id = last_logits.argmax(-1)
    orig = [(layer.keys, layer.values) for layer in cache.layers]
    peak = torch.cuda.max_memory_allocated()
    print(f"[record] seed={seed} L={L} answer={rec.answer} needles={len(rec.needle_positions)} "
          f"prefill={prefill_s:.2f}s peak={peak/2**30:.2f}GiB", flush=True)

    score_cache: dict = {}

    def _rank(x):
        order = torch.argsort(x, descending=True)
        r = torch.empty_like(order)
        r[order] = torch.arange(x.numel(), device=x.device)
        return r

    def get_scores(policy):
        if policy not in score_cache:
            if policy == "obs_union":
                last = obs_scores(model, captured, cache, L, args.obs, "last")
                mean = obs_scores(model, captured, cache, L, args.obs, "mean")
                # Borda fusion: lower rank sum is better; negate so "higher is better"
                score_cache[policy] = [-(_rank(a) + _rank(b)) for a, b in zip(last, mean)]
            else:
                score_cache[policy] = obs_scores(model, captured, cache, L, args.obs, policy[4:])
        return score_cache[policy]

    rows = []
    del last_logits
    for policy in args.policies:
        for layer, (ok, ov) in zip(cache.layers, orig):
            layer.keys, layer.values = ok, ov
        budgets = [None] if policy == "full" else args.budgets
        scores = None if policy in ("full", "recent", "sink_recent", "oracle_needle") else get_scores(policy)
        for budget in budgets:
            eff = budget if budget is not None else L
            nrecent = max(1, int(eff * 0.25)) if budget is not None else 0
            idxs = build_idxs(policy, scores, cache, rec, eff, args.nsink, nrecent, L, args.pool, ids.device)
            diag = selection_stats(idxs, rec.answer_positions, L)
            if idxs is None:
                for layer, (ok, ov) in zip(cache.layers, orig):
                    layer.keys, layer.values = ok, ov
            else:
                for layer, (ok, ov), idx in zip(cache.layers, orig, idxs):
                    H = ok.shape[1]
                    ek = idx.unsqueeze(0).unsqueeze(-1).expand(1, H, idx.shape[1], ok.shape[-1])
                    layer.keys = ok.gather(2, ek).contiguous()
                    layer.values = ov.gather(2, ek).contiguous()
            kept = int(cache.get_seq_length())
            text, ngen, dec_s = decode(model, cache, next_id, L, args.max_new, eos_ids)
            row = {"seed": seed, "context_tokens": L, "policy": policy, "budget": budget,
                   "kept_slots": kept, "prediction": text, "generated": ngen,
                   "answer": rec.answer, "answer_recall": 1.0 if rec.answer in text else 0.0,
                   "selection": diag, "decode_s": round(dec_s, 4),
                   "prefill_s": round(prefill_s, 3), "peak_cuda_bytes": peak,
                   "tokens_per_second_decode": round(ngen / dec_s, 2) if dec_s > 0 else None}
            sfx = (f" selAll={diag['units_with_all_answer_tokens']}/{diag['units']}"
                   f" keepFrac={diag['answer_token_keep_fraction']:.3f}") if diag else ""
            print(f"  {policy:14s} B={str(budget):>5s} kept={kept:>7d} "
                  f"recall={row['answer_recall']:.0f}{sfx} pred={text[:40]!r}", flush=True)
            rows.append(row)
    del cache, orig, captured
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/qcc/models/Llama-3.2-1B-Instruct")
    ap.add_argument("--lengths", type=int, nargs="+", default=[32768, 65536, 131072])
    ap.add_argument("--policies", nargs="+",
                    default=["full", "recent", "obs_mean", "obs_last", "obs_max", "oracle_needle"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--seeds", type=int, nargs="+", default=[101])
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--obs", type=int, default=64)
    ap.add_argument("--nsink", type=int, default=4)
    ap.add_argument("--pool", type=int, default=7)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--prefill-chunk", type=int, default=8192)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    model.eval()

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

    rows = []
    for seed in args.seeds:
        for length in args.lengths:
            rows.extend(run_one(model, tokenizer, length, args, eos_ids, seed))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"model": args.model, "config": {k: v for k, v in vars(args).items()},
         "results": rows}, indent=2))

    # aggregate
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["policy"] == "full":
            continue
        agg[(r["context_tokens"], r["policy"], r["budget"])][0] += r["answer_recall"]
        agg[(r["context_tokens"], r["policy"], r["budget"])][1] += 1
    print("\n== aggregate recall ==")
    for (Lc, pol, b), (c, n) in sorted(agg.items()):
        print(f"  L={Lc:>7d} {pol:14s} B={str(b):>5s} {int(c)}/{n} = {c/n:.3f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
