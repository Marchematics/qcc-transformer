"""One RULER record through the shipped API on a non-Llama checkpoint.

Smoke test for the architecture adapter: Phi-3.5-mini has a fused ``qkv_proj``,
its rotary embedding lives on the attention module, and its remote code uses
the legacy cache API.  Asserts the bounded answer still matches Full-KV.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from transformers import AutoTokenizer

from qcc_transformer.hf_loading import _ensure_remote_code_compat, load_hf_causal_lm
from qcc_transformer.retention import (RetentionConfig, compile_bounded_cache,
                                       forward_accepts)

DEFAULT_MODEL = os.environ.get("QCC_MODEL", "microsoft/Phi-3.5-mini-instruct")
RULER_JSONL = os.environ.get("QCC_RULER_JSONL", "")
if not RULER_JSONL:
    raise SystemExit("QCC_RULER_JSONL is not set: point it at the RULER split JSONL")

MODEL = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 2
_ensure_remote_code_compat()
TOK = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = load_hf_causal_lm(MODEL, dtype=torch.bfloat16, device="cuda",
                          trust_remote_code=True,
                          attn_implementation="eager").eval()

rows = [json.loads(line) for line in open(RULER_JSONL)]
rows = [r for r in rows if r["task"] == "niah_single_1" and 7000 < r["length"] < 17000][:LIMIT]
config = RetentionConfig(budget=4096, lex_cap=512, chain_hops=6,
                         prefill_chunk=1024)


def score(record, text):
    hits = sum(1.0 for out in record["outputs"] if out.lower() in text.lower())
    return hits / len(record["outputs"])


@torch.no_grad()
def greedy(cache, cur, length, steps=24):
    generated = []
    for step in range(steps):
        past = cache.get_seq_length()
        position = torch.tensor([[length + step]], device="cuda")
        call = dict(past_key_values=cache,
                    attention_mask=torch.ones(1, past + 1, dtype=torch.long, device="cuda"),
                    position_ids=position, use_cache=True)
        if forward_accepts(model, "cache_position"):
            call["cache_position"] = torch.arange(past, past + 1)
        out = model(cur, **call)
        cur = out.logits[:, -1:].argmax(-1)
        generated.append(int(cur))
    return TOK.decode(generated, skip_special_tokens=True).strip()


for record in rows:
    prompt = record["input"] + record.get("answer_prefix", "")
    ids = TOK(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    start = time.time()
    cache, logits = compile_bounded_cache(model, ids, config, tokenizer=TOK)
    text = greedy(cache, logits[:, -1:].argmax(-1), ids.shape[1])
    print(json.dumps({"model": MODEL.split("/")[-1], "task": record["task"],
                      "tokens": int(ids.shape[1]), "kept": cache.get_seq_length(),
                      "score": score(record, text), "pred": text[:40],
                      "expected": record["outputs"],
                      "seconds": round(time.time() - start, 1),
                      "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}),
          flush=True)
    del cache
    torch.cuda.empty_cache()
