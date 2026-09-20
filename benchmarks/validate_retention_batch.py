"""Is a ragged batch through the packaged API the same computation?

Two RULER records of different lengths are packed into one left-padded batch,
compiled, decoded together, and compared against compiling and decoding each
record on its own.  Checked at three levels:

* **cache**  the batched compile's per-row slots are bit-identical to the solo
  compile, and the filler of the short row is exactly a duplication of that
  row's own last retained slot;
* **prefill logits**  bit-identical;
* **decode**  a row sliced out of the merged cache decodes to the solo tokens
  exactly; the full two-row batch is reported separately, because a batched
  bf16 GEMM is not bit-identical to a single-row one and greedy argmax can flip
  a near-tie.  Per-step logit deltas and the top-2 margin at any divergence are
  printed so that a difference can be attributed rather than assumed.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/qcc/repo/qcc-transformer")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from qcc_transformer.retention import RetentionConfig, compile_bounded_cache

MODEL = "/root/qcc/models/Llama-3.2-1B-Instruct"
STEPS = 24
TOK = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()

rows = [json.loads(line) for line in open("/home/waas/ruler_a10_v1/ruler_subset.jsonl")]
by_length = {}
for record in rows:
    by_length.setdefault((record["task"], record["length"]), record)
PICKS = [by_length[("niah_single_1", 8190)], by_length[("niah_single_1", 16368)]]


def tokenize(record):
    prompt = record["input"] + record.get("answer_prefix", "")
    return TOK(prompt, return_tensors="pt", add_special_tokens=False).input_ids


def score(record, text):
    hits = sum(1.0 for out in record["outputs"] if out.lower() in text.lower())
    return hits / len(record["outputs"])


@torch.no_grad()
def decode(cache, first, lengths, mask=None, steps=STEPS):
    """Greedy decode where every row keeps its own absolute position."""
    cur, generated, logits = first, [], []
    lengths = torch.tensor(lengths, device="cuda", dtype=torch.long)
    batch = cache.layers[0].keys.shape[0]
    for step in range(steps):
        past = cache.get_seq_length()
        if mask is None:
            mask = torch.ones(batch, past, dtype=torch.long, device="cuda")
        out = model(cur, past_key_values=cache, attention_mask=mask,
                    position_ids=(lengths + step).unsqueeze(1),
                    cache_position=torch.arange(past, past + 1), use_cache=True)
        logits.append(out.logits[:, -1].float())
        cur = out.logits[:, -1:].argmax(-1)
        generated.append(cur)
        mask = torch.cat([mask, torch.ones(batch, 1, dtype=mask.dtype, device="cuda")],
                         dim=-1)
    return torch.cat(generated, dim=1), logits


@torch.no_grad()
def slice_cache(cache, row):
    out = DynamicCache()
    for idx, layer in enumerate(cache.layers):
        out.update(layer.keys[row:row + 1].clone(),
                   layer.values[row:row + 1].clone(), idx)
    return out


def run(config, label):
    lengths = [int(tokenize(record).shape[1]) for record in PICKS]
    print(f"\n=== {label}: budget={config.budget} lex_cap={config.lex_cap} "
          f"lengths={lengths}", flush=True)

    solo = []
    for record in PICKS:
        ids = tokenize(record).cuda()
        cache, logits = compile_bounded_cache(model, ids, config, tokenizer=TOK)
        slots = cache.get_seq_length()
        # clone the compiled cache before decoding appends to it
        keys = [layer.keys[0].clone() for layer in cache.layers]
        values = [layer.values[0].clone() for layer in cache.layers]
        tokens, step_logits = decode(cache, logits[:, -1:].argmax(-1), [ids.shape[1]],
                                     mask=torch.ones(1, slots, dtype=torch.long,
                                                     device="cuda"))
        text = TOK.decode(tokens[0].tolist(), skip_special_tokens=True).strip()
        solo.append({"L": int(ids.shape[1]), "slots": slots, "tokens": tokens[0].tolist(),
                     "text": text, "score": score(record, text), "step_logits": step_logits,
                     "logits": logits[0, -1].float(), "keys": keys, "values": values})
        print(f"solo  L={solo[-1]['L']} slots={slots} score={solo[-1]['score']} "
              f"{text[:44]!r}", flush=True)
        del cache
        torch.cuda.empty_cache()

    width = max(lengths)
    batch = torch.zeros(len(PICKS), width, dtype=torch.long)
    mask = torch.zeros(len(PICKS), width, dtype=torch.long)
    for row, record in enumerate(PICKS):
        seq = tokenize(record).squeeze(0)
        batch[row, width - seq.shape[0]:] = seq          # left padding
        mask[row, width - seq.shape[0]:] = 1
    batch, mask = batch.cuda(), mask.cuda()

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    cache, logits = compile_bounded_cache(model, batch, config, tokenizer=TOK,
                                          attention_mask=mask)
    seconds = time.time() - start
    kept = cache.get_seq_length()
    print(f"batch kept={kept} seconds={seconds:.1f} "
          f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)

    report = {"label": label, "budget": config.budget, "lex_cap": config.lex_cap,
              "lengths": lengths, "kept": kept, "compile_seconds": round(seconds, 1),
              "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
              "prompt_lengths": cache.qcc_prompt_lengths.tolist(), "rows": []}
    checks = []
    # every check below runs on the pristine merged cache, before decoding grows it
    for row, record in enumerate(PICKS):
        slots = solo[row]["slots"]
        filler = kept - slots
        mask_row = cache.qcc_attention_mask[row].bool().cuda()
        checks.append({
            "row": row,
            "cache_identical": all(
                torch.equal(got.keys[row, :, :slots], want_key) and
                torch.equal(got.values[row, :, :slots], want_value)
                for got, want_key, want_value in zip(cache.layers, solo[row]["keys"],
                                                     solo[row]["values"])),
            "filler_is_duplicate": (all(
                torch.equal(layer.keys[row:row + 1, :, slots:],
                            layer.keys[row:row + 1, :, slots - 1:slots].expand(
                                -1, -1, filler, -1)) and
                torch.equal(layer.values[row:row + 1, :, slots:],
                            layer.values[row:row + 1, :, slots - 1:slots].expand(
                                -1, -1, filler, -1))
                for layer in cache.layers) if filler else True),
            "mask_slots": int(cache.qcc_attention_mask[row].sum()),
            "slice_tokens": decode(slice_cache(cache, row),
                                   logits[row:row + 1, -1:].argmax(-1), [lengths[row]],
                                   mask=mask_row.unsqueeze(0).clone())[0][0].tolist(),
        })
    batch_tokens, batch_logits = decode(cache, logits[:, -1:].argmax(-1), lengths,
                                        mask=cache.qcc_attention_mask.clone())
    for row, record in enumerate(PICKS):
        slots = solo[row]["slots"]
        filler = kept - slots
        check = checks[row]
        deltas = [float((batch_logits[step][row] - solo[row]["step_logits"][step][0]).abs().max())
                  for step in range(STEPS)]
        diverged = next((step for step in range(STEPS)
                         if batch_tokens[row, step].item() != solo[row]["tokens"][step]),
                        None)
        margin = None
        if diverged is not None:
            top2 = batch_logits[diverged][row].topk(2).values
            margin = float(top2[0] - top2[1])
        text = TOK.decode(batch_tokens[row].tolist(), skip_special_tokens=True).strip()
        entry = {"L": lengths[row], "slots": slots, "filler_slots": filler,
                 "mask_slots": check["mask_slots"],
                 "cache_identical_to_solo": bool(check["cache_identical"]),
                 "filler_is_duplicate_of_last_slot": bool(check["filler_is_duplicate"]),
                 "prefill_logits_identical_to_solo":
                     bool(torch.equal(logits[row, -1].float(), solo[row]["logits"])),
                 "slice_decode_matches_solo":
                     check["slice_tokens"] == solo[row]["tokens"],
                 "batch_decode_matches_solo":
                     batch_tokens[row].tolist() == solo[row]["tokens"],
                 "diverged_at_step": diverged,
                 "max_logit_delta": max(deltas),
                 "top2_margin_at_divergence": margin,
                 "score": score(record, text), "text": text[:44]}
        report["rows"].append(entry)
        print("row  ", json.dumps(entry), flush=True)
    return report


out = []
for config, label in ((RetentionConfig(budget=16384, lex_cap=0), "filler-heavy"),
                      (RetentionConfig(budget=4096, lex_cap=512, chain_hops=6), "production")):
    out.append(run(config, label))
    torch.cuda.empty_cache()
OUT = Path(__file__).resolve().parent.parent / "artifacts" / \
    "bounded-decode-frontier-batch-validation.json"
json.dump(out, open(OUT, "w"), indent=1)
print("\nsummary", json.dumps([{k: v for k, v in entry.items() if k != "rows"}
                               for entry in out]))
