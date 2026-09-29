"""Quality frontier of a causal, bounded, exact-KV decode cache.

Question
--------
For a real pretrained long-context LM, prefill is run with exact Full-KV
attention.  Only the *decode* cache is bounded: after prefill a fixed budget of
KV slots per (layer, kv-head) is kept and greedy decode runs against that pruned
cache.  Selection uses only information available at the end of the prompt
(never future decode tokens), so the procedure is causal and deployable.

The script sweeps selection policies and budgets on RULER-style multi-key NIAH
records and compares against unpruned Full-KV decode.  The goal is to separate
"the selection law is bad" from "bounded exact decode is fundamentally
insufficient".

Diagnostic harness; it does not modify the QCC package.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = os.environ.get("QCC_MODEL", "meta-llama/Llama-3.2-1B-Instruct")

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


def build_record(tokenizer, target_tokens: int, num_pairs: int, seed: int) -> Record:
    rng = random.Random(seed)
    keys = rng.sample(KEYS, num_pairs)
    values = {k: str(rng.randint(1_000_000, 9_999_999)) for k in keys}
    question_key = rng.choice(keys)
    depressions = sorted(rng.uniform(0.05, 0.95) for _ in range(num_pairs))
    question = (
        f"\n\nWhat is the special magic number for {question_key}? "
        f"Answer with the number only.\nAnswer:"
    )
    n_sent = max(16, int(target_tokens / 12))
    prompt = ""
    for _ in range(5):
        body = " ".join(rng.choice(FILLER) for _ in range(n_sent))
        pieces = []
        cursor = 0
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
                  meta={"all_pairs": values, "num_pairs": num_pairs, "seed": seed})


def find_needle_positions(tokenizer, record: Record):
    """Token indices of every needle sentence and of the asked answer value."""
    enc = tokenizer(record.prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc.offset_mapping
    all_positions: set[int] = set()
    answer_positions: list[int] = []
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


@torch.no_grad()
def prefill(model, ids):
    return model(ids, use_cache=True, output_hidden_states=True)


def _rope(model, x, position_ids):
    return model.model.rotary_emb(x, position_ids)


def apply_rope(q, cos, sin):
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    return q * cos + rotate_half(q) * sin


@torch.no_grad()
def obs_scores(model, hidden_states, cache, obs: int, nsink: int, mode: str):
    """Per (layer, kv-head) key importance from the question observation window."""
    L = cache.get_seq_length()
    cfg = model.config
    num_heads = cfg.num_attention_heads
    key_pos = torch.arange(L, device=cache.layers[0].keys.device)
    out = []
    for layer_idx, layer in enumerate(cache.layers):
        h = hidden_states[layer_idx][0]
        attn = model.model.layers[layer_idx].self_attn
        head_dim = attn.head_dim
        q = attn.q_proj(h).view(L, num_heads, head_dim).transpose(0, 1)
        lo = max(nsink, L - obs)
        pos = torch.arange(lo, L, device=h.device)
        qo = q[:, lo:L, :]
        cos, sin = _rope(model, h.unsqueeze(0), pos.unsqueeze(0))
        qo = apply_rope(qo, cos[0].unsqueeze(0), sin[0].unsqueeze(0))
        k = layer.keys[0]
        kv_heads = k.shape[0]
        group = num_heads // kv_heads
        qg = qo.view(kv_heads, group, qo.shape[1], head_dim)
        s = torch.einsum("hgod,hld->hgol", qg.float(), k.float()) * attn.scaling
        valid = key_pos[None, :] <= pos[:, None]
        s = s.masked_fill(~valid[None, None, :, :], float("-inf"))
        w = torch.softmax(s, dim=-1)
        if mode == "mean":
            sc = w.mean(dim=(1, 2))
        elif mode == "max":
            sc = w.amax(dim=(1, 2))
        elif mode == "last":
            sc = w[:, :, -1, :].amax(dim=1)
        else:
            raise ValueError(mode)
        out.append(sc)
        del s, w
    return out


@torch.no_grad()
def h2o_scores(model, hidden_states, cache, chunk: int | None = None):
    """Cumulative causal attention mass per key (H2O heavy hitters)."""
    L = cache.get_seq_length()
    if chunk is None:
        chunk = max(8, min(512, 8_000_000 // max(1, L)))
    cfg = model.config
    num_heads = cfg.num_attention_heads
    key_pos = torch.arange(L, device=cache.layers[0].keys.device)
    out = []
    for layer_idx, layer in enumerate(cache.layers):
        h = hidden_states[layer_idx][0]
        attn = model.model.layers[layer_idx].self_attn
        head_dim = attn.head_dim
        k = layer.keys[0]
        kv_heads = k.shape[0]
        group = num_heads // kv_heads
        acc = torch.zeros(kv_heads, L, device=h.device, dtype=torch.float32)
        for start in range(0, L, chunk):
            end = min(L, start + chunk)
            pos = torch.arange(start, end, device=h.device)
            q = attn.q_proj(h[start:end]).view(end - start, num_heads, head_dim).transpose(0, 1)
            cos, sin = _rope(model, h[start:end].unsqueeze(0), pos.unsqueeze(0))
            q = apply_rope(q, cos[0].unsqueeze(0), sin[0].unsqueeze(0))
            qg = q.view(kv_heads, group, end - start, head_dim)
            s = torch.einsum("hgod,hld->hgol", qg.float(), k.float()) * attn.scaling
            valid = key_pos[None, :] <= pos[:, None]
            s = s.masked_fill(~valid[None, None, :, :], float("-inf"))
            acc += torch.softmax(s, dim=-1).sum(dim=(1, 2))
            del s
        out.append(acc)
    return out


def keynorm_scores(cache):
    return [layer.keys[0].float().norm(dim=-1) for layer in cache.layers]


def topk_indices(scores_layer, budget, nsink, nrecent, L, pool=1):
    """Return (kv_heads, <=budget) index tensor, always including sink+recent.

    pool>1 max-pools over contiguous `pool`-token blocks before ranking, so a
    high-scoring token also protects its neighbours (SnapKV-style pooling).
    """
    kv_heads = scores_layer.shape[0]
    idxs = []
    forced = sorted(set(list(range(min(nsink, L))) + list(range(max(0, L - nrecent), L))))
    for hgt in range(kv_heads):
        s = scores_layer[hgt].clone().float()
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
    fracs = []
    units_all = 0
    for idx in idxs:
        for head in range(idx.shape[0]):
            hit = torch.isin(pos, idx[head]).float().mean().item()
            fracs.append(hit)
            units_all += int(hit == 1.0)
    return {
        "units": len(fracs),
        "units_with_all_answer_tokens": units_all,
        "answer_token_keep_fraction": sum(fracs) / len(fracs),
        "all_answer_unit_fraction": units_all / len(fracs),
    }


def policy_idxs(model, policy, scores, cache, record, budget, nsink, nrecent, L, pool, device):
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
    gen, cur = [], next_id
    t0 = time.time()
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
    text = TOKENIZER.decode(gen, skip_special_tokens=True)
    return text.strip(), len(gen), time.time() - t0


@torch.no_grad()
def run_record(model, tokenizer, length, args, eos_ids):
    global TOKENIZER
    TOKENIZER = tokenizer
    rec = build_record(tokenizer, length, args.pairs, args.seed + length)
    rec.needle_positions, rec.answer_positions = find_needle_positions(tokenizer, rec)
    ids = tokenizer(rec.prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    L = ids.shape[1]
    print(f"[record] L={L} answer={rec.answer} needle_tokens={len(rec.needle_positions)} "
          f"answer_tokens={rec.answer_positions}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = prefill(model, ids)
    prefill_s = time.time() - t0
    cache = out.past_key_values
    hidden_states = out.hidden_states
    next_id = out.logits[:, -1:].argmax(-1)
    orig = [(layer.keys, layer.values) for layer in cache.layers]
    if args.free_hidden:
        hidden_states = tuple(h.detach() for h in hidden_states)
    print(f"[prefill] {prefill_s:.2f}s peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)

    rows = []
    # cache scores per policy
    score_cache: dict[str, list] = {}

    def get_scores(policy):
        if policy in score_cache:
            return score_cache[policy]
        if policy.startswith("obs_"):
            s = obs_scores(model, hidden_states, cache, args.obs, args.nsink, policy[4:])
        elif policy == "h2o":
            s = h2o_scores(model, hidden_states, cache)
        elif policy == "keynorm":
            s = keynorm_scores(cache)
        else:
            s = None
        score_cache[policy] = s
        return s

    for policy in args.policies:
        # restore the pristine full-KV cache before any selection or scoring
        for layer, (ok, ov) in zip(cache.layers, orig):
            layer.keys, layer.values = ok, ov
        budgets = [None] if policy == "full" else args.budgets
        scores = None if policy in ("full", "recent", "sink_recent", "oracle_needle", "random") else get_scores(policy)
        for budget in budgets:
            eff = budget if budget is not None else L
            nrecent = max(1, int(eff * 0.25)) if budget is not None else 0
            idxs = policy_idxs(model, policy, scores, cache, rec, eff, args.nsink, nrecent, L, args.pool, ids.device)
            diag = selection_stats(idxs, rec.answer_positions, L)
            # (re)build cache
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
            row = {
                "policy": policy, "budget": budget, "context_tokens": L,
                "kept_slots": kept, "prediction": text, "generated": ngen,
                "answer_recall": 1.0 if rec.answer in text else 0.0,
                "selection": diag, "decode_s": round(dec_s, 3),
                "prefill_s": round(prefill_s, 3),
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                "answer": rec.answer, "seed": rec.meta["seed"], "pairs": args.pairs,
                "pool": args.pool,
            }
            sfx = ""
            if diag:
                sfx = (f" selAll={diag['units_with_all_answer_tokens']}/{diag['units']}"
                       f" keepFrac={diag['answer_token_keep_fraction']:.3f}")
            print(f"  {policy:14s} B={str(budget):>5s} kept={kept:>6d} "
                  f"recall={row['answer_recall']:.0f}{sfx} pred={text!r}", flush=True)
            rows.append(row)
    del out, cache, orig, hidden_states
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--lengths", type=int, nargs="+", default=[16384, 32768])
    ap.add_argument("--policies", nargs="+",
                    default=["full", "recent", "sink_recent", "random", "keynorm",
                             "obs_mean", "obs_max", "obs_last", "h2o", "oracle_needle"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--obs", type=int, default=64)
    ap.add_argument("--nsink", type=int, default=4)
    ap.add_argument("--pool", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--free-hidden", action="store_true", default=True)
    ap.add_argument("--last-logit-only", action="store_true", default=False)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    model.eval()
    if args.last_logit_only:
        inner = model.lm_head

        class _LastTokenLMHead(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, x):
                return self.wrapped(x[:, -1:, :])

        model.lm_head = _LastTokenLMHead(inner)
        print("[setup] lm_head wrapped to compute only the final position", flush=True)
    eos = model.config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}

    all_rows = []
    for length in args.lengths:
        all_rows.extend(run_record(model, tokenizer, length, args, eos_ids))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"model": args.model, "config": vars(args), "results": all_rows}, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
