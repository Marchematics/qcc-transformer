"""Does the packaged API reproduce the benchmark's answer on a real record?"""
import json, sys, time
sys.path.insert(0, "/root/qcc/repo/qcc-transformer")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qcc_transformer.retention import RetentionConfig, compile_bounded_cache

M = "/root/qcc/models/Llama-3.2-1B-Instruct"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to("cuda").eval()
rows = [json.loads(l) for l in open('/home/waas/ruler_a10_v1/ruler_subset.jsonl')]
records = [r for r in rows if r['task'] in ('niah_single_1', 'niah_multikey_2')
           and 12000 < r['length'] < 20000][:4]
out = []
for r in records:
    prompt = r['input'] + r.get('answer_prefix', '')
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    config = RetentionConfig(budget=4096, lex_cap=512, chain_hops=6)
    t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    cache, logits = compile_bounded_cache(model, ids, config, tokenizer=tok)
    kept = cache.get_seq_length()
    cur = logits[:, -1:].argmax(-1)
    mask = torch.ones(1, kept + 1, device="cuda", dtype=torch.long)
    gen = []
    with torch.no_grad():
        for step in range(24):
            cp = torch.tensor([ids.shape[1] + step], device="cuda")
            o = model(cur, past_key_values=cache, attention_mask=mask, cache_position=cp,
                      position_ids=cp.unsqueeze(0), use_cache=True)
            cur = o.logits[:, -1:].argmax(-1)
            gen.append(int(cur))
            mask = torch.cat([mask, torch.ones(1, 1, device="cuda", dtype=torch.long)], dim=1)
    text = tok.decode(gen, skip_special_tokens=True)
    score = sum(1.0 for o in r['outputs'] if o.lower() in text.lower()) / len(r['outputs'])
    out.append({"task": r['task'], "L": int(ids.shape[1]), "kept": kept,
                "score": round(score, 3), "pred": text.strip()[:40],
                "seconds": round(time.time() - t0, 1),
                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
    print(out[-1], f"expected {r['outputs']}", flush=True)
    del cache
    torch.cuda.empty_cache()
json.dump(out, open("packaged_validation.json", "w"), indent=1)
print("mean score", sum(o["score"] for o in out) / len(out))
