"""Does bounded retention preserve *language modelling*, not just retrieval?

Retrieval tasks ask whether one fact survives.  This harness asks the broader
question: with an exact prefill and a bounded decode cache, how much does the
model's next-token distribution degrade on ordinary long text?

Protocol
--------
1. Build one long document from local text files (no dataset download).
2. Prefill `doc[:L-N]` exactly, in chunks.
3. Score `doc[L-N:]` by teacher forcing, one token at a time, appending each
   true token to the cache (as a real decoder would).
4. Repeat with the matched Full-KV cache and with each retention policy.

Retention for language modelling is question-free: the observation window is
simply the last `obs` tokens of the prefix, i.e. the SnapKV setting.  No lexical
anchors are used because there is no question to anchor on.  The default
``--sources`` corpus is the repository's own Markdown and Python sources plus
the running interpreter's standard library.

Reported: mean negative log-likelihood, perplexity, and the ratio to Full-KV.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, DynamicLayer

DEFAULT_MODEL = os.environ.get("QCC_MODEL", "meta-llama/Llama-3.2-1B-Instruct")

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    import benchmark_bounded_decode_frontier as L
except ImportError:
    import longctx as L


def build_document(target_tokens, tokenizer, sources, min_bytes=2000):
    """Concatenate local text files into one document of ~target_tokens."""
    chunks, total = [], 0
    for pattern in sources:
        for path in sorted(glob.glob(pattern)):
            try:
                text = Path(path).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if len(text) < min_bytes:
                continue
            chunks.append(text)
    doc = "\n\n".join(chunks)
    ids = tokenizer(doc, add_special_tokens=False).input_ids
    if len(ids) < target_tokens:
        raise SystemExit(f"only {len(ids)} tokens available, need {target_tokens}")
    return ids[:target_tokens]


@torch.no_grad()
def score_suffix(model, cache, prefix_len, target_ids, mask_len, n):
    """Teacher-forced NLL of `n` tokens, appending each to the cache."""
    dev = target_ids.device
    total_nll, count = 0.0, 0
    cur = target_ids[0].view(1, 1)
    pos = prefix_len
    for step in range(n):
        cp = torch.tensor([pos], device=dev)
        o = model(cur, past_key_values=cache, cache_position=cp,
                  position_ids=cp.unsqueeze(0), use_cache=True)
        logits = o.logits[:, -1, :].float()
        target = target_ids[step + 1] if step + 1 < n else None
        if target is not None:
            logp = torch.log_softmax(logits, dim=-1)[0, target]
            total_nll += float(-logp)
            count += 1
            cur = target.view(1, 1)
        pos += 1
    return total_nll / max(1, count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--length", type=int, default=32768)
    ap.add_argument("--suffix", type=int, default=256)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--obs", type=int, default=64)
    ap.add_argument("--nsink", type=int, default=4)
    ap.add_argument("--pool", type=int, default=7)
    ap.add_argument("--dilate", type=int, default=9)
    ap.add_argument("--policies", nargs="+", default=["obs_last", "obs_mean", "recent", "random"])
    ap.add_argument("--prefill-chunk", type=int, default=8192)
    ap.add_argument("--sources", nargs="+",
                    default=[str(REPO_ROOT / "*.md"),
                             str(REPO_ROOT / "artifacts" / "*.md"),
                             str(REPO_ROOT / "benchmarks" / "*.py"),
                             str(REPO_ROOT / "qcc_transformer" / "*.py"),
                             str(Path(os.__file__).resolve().parent / "*.py")],
                    help="glob patterns concatenated into the document; the last "
                         "default entry is the interpreter's own standard library")
    ap.add_argument("--document", default=None, help="pinned token-id JSON to reuse")
    ap.add_argument("--save-document", default=None, help="write the built document here")
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

    if args.document and Path(args.document).exists():
        ids = json.loads(Path(args.document).read_text())[: args.length]
        print(f"[doc] reused pinned document ({len(ids)} tokens)", flush=True)
    else:
        ids = build_document(args.length, tok, args.sources)
        if args.save_document:
            Path(args.save_document).write_text(json.dumps(ids))
            print(f"[doc] pinned document written to {args.save_document}", flush=True)
    full_ids = torch.tensor(ids, dtype=torch.long, device="cuda").unsqueeze(0)
    prefix_len = args.length - args.suffix
    prefix = full_ids[:, :prefix_len]
    suffix_ids = full_ids[0, prefix_len:]
    print(f"[doc] {args.length} tokens, prefix {prefix_len}, suffix {args.suffix}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    cache, last_logits, captured = L.prefill_capture(model, prefix, args.obs, args.prefill_chunk)
    torch.cuda.synchronize()
    print(f"[prefill] {time.time() - t0:.1f}s peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB",
          flush=True)
    orig = [(layer.keys, layer.values) for layer in cache.layers]

    results = {"length": args.length, "budget": args.budget, "suffix": args.suffix,
               "prefix_len": prefix_len, "variants": {}}

    def run(name, idxs):
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
        nll = score_suffix(model, cache, prefix_len, suffix_ids, kept, args.suffix)
        ppl = math.exp(min(nll, 20))
        results["variants"][name] = {"kept_slots": kept, "nll": round(nll, 5),
                                     "perplexity": round(ppl, 4)}
        print(f"  {name:12s} kept={kept:>7} NLL={nll:.5f} ppl={ppl:.3f}", flush=True)
        return nll

    # Score every policy *before* any variant runs: teacher forcing appends the
    # suffix to the cache, which would make later key indices inconsistent.
    scores = {}
    for policy in args.policies:
        mode = "last" if policy.startswith("obs_last") else (
            "mean" if policy.startswith("obs_mean") else None)
        if mode is not None:
            scores[policy] = L.obs_scores(model, captured, cache, prefix_len, args.obs, mode,
                                          key_chunk=4096)

    full_nll = run("full_kv", None)
    results["variants"]["full_kv"]["is_reference"] = True

    for policy in args.policies:
        nrecent = max(1, int(args.budget * 0.25))
        if policy == "recent":
            sel = sorted(range(max(0, prefix_len - args.budget), prefix_len))
            idxs = [torch.tensor([sel], device="cuda") for _ in cache.layers]
        elif policy == "random":
            g = torch.Generator(device="cpu").manual_seed(7)
            sel = sorted(torch.randperm(prefix_len, generator=g)[:args.budget].tolist())
            idxs = [torch.tensor([sel], device="cuda") for _ in cache.layers]
        else:
            idxs = [L.topk_indices(sc, args.budget, args.nsink, nrecent, prefix_len,
                                   args.pool, args.dilate) for sc in scores[policy]]
        nll = run(policy, idxs)
        results["variants"][policy]["nll_ratio_to_full"] = round(nll / full_nll, 5)
        results["variants"][policy]["ppl_ratio_to_full"] = round(
            math.exp(min(nll, 20)) / math.exp(min(full_nll, 20)), 5)
        del idxs
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in results["variants"].items()}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
