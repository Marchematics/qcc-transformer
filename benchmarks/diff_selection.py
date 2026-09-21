"""Same process, same cache: is the packaged selection the harness selection?

Compares, on one RULER record: the lexical anchor positions, the observation
scores, and the final per-(layer, head) retained index sets produced by the
benchmark harness (``benchmark_bounded_decode_frontier``) and by the packaged
``qcc_transformer.retention`` code path.  This isolates a real implementation
difference from bf16 decode nondeterminism between two separate runs.
"""
import json
import sys

sys.path.insert(0, "/root/qcc/repo/qcc-transformer")
sys.path.insert(0, "/root/qcc/repo/qcc-transformer/benchmarks")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import benchmark_bounded_decode_frontier as L
from qcc_transformer.retention import (RetentionConfig, lexical_anchors,
                                       observation_scores, prefill_capture,
                                       select_indices)

MODEL = "/root/qcc/models/Llama-3.2-1B-Instruct"
TASK = sys.argv[1] if len(sys.argv) > 1 else "vt"
WANT_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 31389

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
rows = [json.loads(line) for line in open("/home/waas/ruler_a10_v1/ruler_subset.jsonl")]
row = None
for candidate in rows:
    if candidate.get("task") != TASK:
        continue
    prompt = candidate["input"] + candidate.get("answer_prefix", "")
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if len(ids) == WANT_TOKENS:
        row, prompt = candidate, prompt
        break
assert row is not None, f"no {TASK} record with {WANT_TOKENS} prompt tokens"
print("record", row["task"], row.get("length"), len(ids), row["outputs"], flush=True)

ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
Lc = int(ids.shape[1])
config = RetentionConfig(budget=4096, lex_cap=512, chain_hops=6, observation_window=64,
                         attention_sinks=4, pool=7, dilate=9, prefill_chunk=8192,
                         key_chunk=4096)
with torch.no_grad():
    cache, _logits, captured = prefill_capture(model, ids, config.observation_window,
                                               config.prefill_chunk)

harness_anchors = L.question_lexical_positions(tokenizer, prompt, 64, hops=6)
decoded = tokenizer.decode(ids[0].tolist(), skip_special_tokens=False)
packaged_anchors = lexical_anchors(tokenizer, decoded, config)
overlap = len(set(harness_anchors) & set(packaged_anchors))
print(f"anchors harness={len(harness_anchors)} packaged={len(packaged_anchors)} "
      f"shared={overlap} harness_only={len(set(harness_anchors) - set(packaged_anchors))} "
      f"packaged_only={len(set(packaged_anchors) - set(harness_anchors))} "
      f"prompt_roundtrip_identical={decoded == prompt}", flush=True)

harness_scores = L.obs_scores(model, captured, cache, Lc, 64, "last", 4096)
packaged_scores = observation_scores(model, captured, cache, Lc, config)
bitwise = [torch.equal(a, b) for a, b in zip(harness_scores, packaged_scores)]
print("scores bitwise identical per layer:", all(bitwise), bitwise[:4], flush=True)

record = L.Record(prompt=prompt, question_key="", answer=row["outputs"][0], meta={})
record.lexical_positions = harness_anchors
harness_idx = L.build_idxs("lex_obs", harness_scores, cache, record, 4096, 4, 1024, Lc, 7,
                           ids.device, 9, 512)
deltas = [float((a - b).abs().max()) for a, b in zip(harness_scores, packaged_scores)]
print("max |score delta| per layer:", [round(d, 6) for d in deltas], flush=True)
for label, anchors in (("packaged anchors", packaged_anchors),
                       ("harness anchors", harness_anchors)):
    packaged_idx = select_indices(packaged_scores, cache, Lc, config, anchors)
    identical = all(torch.equal(a, b) for a, b in zip(harness_idx, packaged_idx))
    print(f"selection identical ({label}):", identical, flush=True)
    if label == "harness anchors":
        print("scores swapped: identical:",
              all(torch.equal(a, b) for a, b in
                  zip(harness_idx, select_indices(harness_scores, cache, Lc, config, anchors))),
              flush=True)
    if not identical:
        for layer, (a, b) in enumerate(zip(harness_idx, packaged_idx)):
            for head in range(a.shape[0]):
                sa, sb = set(a[head].tolist()), set(b[head].tolist())
                if sa != sb:
                    print(f"  layer {layer} head {head}: harness_only={len(sa - sb)} "
                          f"packaged_only={len(sb - sa)} example={sorted(sa - sb)[:6]}",
                          flush=True)
                    break
            else:
                continue
            break
