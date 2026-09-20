# Bounded exact-KV decode with query-window selection

Date: 2026-09-20. Workspace: `/root/qcc/experiments/retention_frontier`.
Hardware: one NVIDIA A10G, 24 GiB. Model: `Llama-3.2-1B-Instruct`
(16 layers, 8 KV heads, head dim 64, native `max_position_embeddings` 131072),
bf16, frozen, no training.

## 1. Question

The QCC repository has, across many sessions, failed to preserve long-range
retrieval with a bounded historical state. Non-causal "quality-first" selection
recovered answer recall; every causal bounded variant failed. That left one
question open:

> Is a bounded exact-KV decode cache fundamentally unable to preserve NIAH
> retrieval, or was the *selection law* wrong?

This experiment answers it by measuring the quality frontier directly, with the
selection law isolated from any archive/recurrence/mixing implementation.

## 2. Protocol

For each record (synthetic RULER-style `niah_multikey` with 2 key/value pairs and
a question appended after the haystack):

1. **Exact prefill.** The prompt is fed in fixed 8192-token chunks with a growing
   KV cache and absolute position ids. This is algebraically identical to one
   long causal forward pass, but keeps activation memory `O(chunk)` and avoids an
   explicit quadratic causal mask. Only the last-token logits are materialised.
2. **Selection (once, at the end of the prompt).** For every `(layer, kv-head)`
   the keys are ranked by the attention they receive from the *observation
   window*: the last `obs = 64` prompt tokens, i.e. the question. The required
   query-key scores are the native ones (same RoPE phases, same scaling). No
   future token, no generated token, and no answer label is used. Ranking is
   followed by SnapKV-style max-pooling over 7-token blocks, so a high-scoring
   token also protects its neighbours inside a multi-token needle.
3. **Bounded cache.** Keep `budget` slots per `(layer, kv-head)`, always including
   4 attention sinks and a recent-window share. Drop everything else.
4. **Decode.** Greedy, 16 tokens, exact softmax attention over the retained KV
   only. Keys keep their native RoPE phases, so absolute positions are unchanged.

Policies compared:

| policy | selection law | causal? |
|---|---|---|
| `full` | no pruning (matched Full-KV reference) | yes |
| `recent` | last `budget` tokens | yes |
| `obs_mean` | mean observation-window attention weight | yes |
| `obs_last` | attention of the final prompt token (max over grouped query heads) | yes |
| `obs_max` | max observation-window attention weight | yes |
| `h2o` | cumulative causal attention mass over all prefill queries | yes |
| `keynorm` | largest key L2 norm | yes |
| `random` | uniform random slots | yes |
| `oracle_needle` | needle token positions (diagnostic upper bound; not deployable) | no |

`obs_last` ranks by the raw final-query score: the softmax denominator is
constant across keys for a fixed query, so the ranking is unchanged and no
normalisation pass is needed.

## 3. Results

### 3.1 Mechanism check (single record per length, 4 key/value pairs)

`frontier.py`, `sweep_v1.json`:

| Length | Full-KV | obs_last | obs_mean | obs_max | h2o | keynorm | recent | random | oracle_needle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15,908 | 1/1 | 5/5 | 5/5 | 5/5 | 2/5 | 0/5 | 0/5 | 0/5 | 5/5 |
| 31,821 | 1/1 | 5/5 | 5/5 | 5/5 | 2/5 | 0/5 | 0/5 | 0/5 | 5/5 |
| 63,661 | 1/1 | 5/5 | 5/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |

(budgets 256, 512, 1024, 2048, 4096)

At 64K, keeping only the needle tokens plus sinks and a recent window
(`oracle_needle`) is *not* sufficient, while 256 tokens chosen by
observation-window attention are. Retrieval needs the question-relevant slice of
the surrounding context, not just the fact itself.

### 3.2 Aggregate (6 records, 2 key/value pairs, chunked exact prefill)

`longctx.py`, `aggregate_v2.json`. Full-KV is correct on **18/18** records
(6 lengths of ~32K, 6 of ~64K, 6 of ~128K), so retention is meaningful
everywhere.

Answer recall (records correct / records), budgets are slots per (layer, kv-head):

| Length | recent 128 | recent 1024 | obs_mean 128 | obs_mean 1024 | obs_max 128 | obs_max 512 | obs_last 128 | obs_last 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ~32K | 0/6 | 0/6 | 4/6 | 6/6 | 3/6 | 3/6 | **6/6** | **6/6** |
| ~64K | 0/6 | 0/6 | 5/6 | 5/6 | 5/6 | 6/6 | 4/6 | **6/6** |
| ~128K | 0/6 | 0/6 | 3/6 | 6/6 | 3/6 | 3/6 | **6/6** | **6/6** |
| total /18 | 0 | 0 | 12 | 17 | 11 | 12 | **16** | **18** |

* `obs_last` at 1024 slots per (layer, kv-head) matches Full-KV on **18/18**
  records, i.e. 100% retention.
* `obs_last` at **128** slots still matches Full-KV on 16/18 (88.9%).
* Recency-only selection is 0/18 at every budget up to 1024, so the result is
  not a recency artefact.
* `obs_mean` (the SnapKV-style mean) reaches 17/18 at 1024.

B=128 is 4 MiB of decode cache; Full-KV at 128K is 4.00 GiB. The bounded cache
is therefore **1024x smaller at 128K** while still answering 16/18 records, and
18/18 at 32 MiB (B=1024), a 128x reduction.

### 3.3 State and speed accounting

Llama-3.2-1B cache geometry: `16 layers x 8 kv-heads x 64 dim x 2 (K,V) x 2 bytes
= 32,768 bytes per token` for a full KV cache, i.e. `32,768 x B` bytes for a
budget of `B` slots per `(layer, kv-head)`.

| Context | Full-KV cache | B=128 bounded | B=256 bounded | reduction at B=128 |
|---:|---:|---:|---:|---:|
| 32K | 1.00 GiB | 4 MiB | 8 MiB | 256x |
| 64K | 2.00 GiB | 4 MiB | 8 MiB | 512x |
| 128K | 4.00 GiB | 4 MiB | 8 MiB | 1024x |
| 1M | 32.0 GiB | 4 MiB | 8 MiB | 8192x |

Bounded decode state is independent of context length: the `128K -> 1M` state
growth of the retained decode cache is exactly `1.00x`.

## 4. What this establishes, and what it does not

Establishes:

* A **causal**, **training-free**, **zero-new-parameter** retention law
  (observation-window attention, with block pooling) preserves NIAH multi-key
  retrieval that exact Full-KV solves: **18/18 records at 1024 slots** per
  `(layer, kv-head)` and 16/18 at 128 slots, at 32K/64K/128K.
* At 128K, 128 slots is a 4 MiB decode cache against a 4.00 GiB Full-KV cache,
  a **1024x** reduction, with no loss on 6/6 records.
* The previous QCC failures on these records are attributable to the selection
  law and to the archive read/mix path, not to bounded decode state as such.
* Recency (0/18), random, key-norm and H2O-style cumulative-mass selection all
  fail; the observation-window (question-aware) law is doing real work.

Does not establish:

* **Bounded prefill state.** Prefill is exact and holds the full KV transiently
  (`O(L)`), so peak prefill memory still grows with context. Only the persistent
  decode state is bounded. The objective's "historical state O(1)" is met for
  decode; a bounded-prefill variant is still open.
* **Official benchmark quality.** These are synthetic RULER-style NIAH records,
  not NVIDIA RULER JSONL, LongBench or PG-19. Aggregate quality >= 99% and
  worst-task >= 97% are not measured here.
* **Serving performance.** No vLLM/SLA/concurrency/TPOT measurement; the
  5x TPOT, 3x throughput and 8x concurrency targets remain unproven.
* **1M retrieval.** Reported separately below if run; Llama-3.2-1B is a
  128K-native checkpoint and is out of its native range at 1M.
* **Integration.** The mechanism is not yet implemented in `qcc_transformer`;
  this is a standalone diagnostic harness.

## 5. Design implied for QCC

The result says the deployable object is a **cache policy**, not an attention
approximation. All QCC archive/recurrence/gate machinery can be bypassed:

1. Prefill: exact causal attention (chunked for bounded activations). Capture,
   per layer, the attention inputs of the last `obs` query positions. This is
   `O(obs * d)` state, independent of context length.
2. Compile: for each `(layer, kv-head)`, score every key by the attention it
   receives from those queries, max-pool over small blocks, keep the top `B`.
   Union with `s` attention sinks and a recent window. Cost is `O(obs * L)` per
   layer, i.e. `obs / L` of the prefill attention work.
3. Decode: exact attention over the retained set. Keys keep native RoPE phases.
4. No new parameters, no calibration, no fine-tuning; the pretrained checkpoint
   is untouched.

Against the objective's metric table this design directly addresses
`trainable params <= 0.5%` (it is exactly 0), `retrofit` (frozen pretrained LM),
`historical state O(1)` and `128K -> 1M growth <= 1.25x` (the retained decode
cache is bit-identical in size at every context length). It does **not** by
itself address the quality-suite, prefill-state, TPOT, throughput or
concurrency targets.

The main open engineering question is the **prefill state**: an exact prefill
holds the full KV transiently. Two candidate directions:

* keep only the observation-window queries in a bounded side buffer and run the
  prefill with a bounded *recency + sink + high-norm* cache, then re-rank the
  survivors with the question queries at the end (cheap, but the prefill hidden
  states would then be computed against a lossy cache);
* keep exact K/V in host memory (or a compressed bf16/fp8 tier) during prefill
  and only materialise the bounded GPU cache afterwards.

## 6. Next steps

1. Implement the retention law inside `qcc_transformer` as a cache policy with
   an HF-retrofit entry point; verify against this harness on the same records.
2. Move from synthetic NIAH to the official RULER JSONL split
   (`/home/waas/ruler_a10_v1/ruler_subset.jsonl`), then LongBench and PG-19, and
   report aggregate and worst-task retention rather than retrieval only.
3. Measure serving behaviour: decode TPOT and throughput at 128K and 1M,
   fixed-SLA concurrency, and peak memory with the bounded cache.
4. Resolve the prefill-state question above.

## 7. What this does not claim

No official RULER/LongBench/PG-19 result, no 1M retrieval result on a
1M-native checkpoint, no vLLM/TPOT/throughput/concurrency measurement, and no
claim about the QCC package itself: this is a standalone diagnostic harness.

## 8. Reproduction

Harness (added to this repository):

* `benchmarks/benchmark_selection_frontier.py` - policy comparison with full
  hidden-state capture (mechanism check).
* `benchmarks/benchmark_bounded_decode_frontier.py` - chunked exact prefill,
  `O(obs)` hidden capture, multi-seed aggregation.
* `benchmarks/summarize_bounded_decode.py` - result tables.

```bash
# mechanism check (full policy list), 16K/32K/64K, 4 key/value pairs
python benchmarks/benchmark_selection_frontier.py --lengths 16384 32768 65536 \
  --policies full recent sink_recent random keynorm obs_mean obs_max obs_last h2o oracle_needle \
  --budgets 256 512 1024 2048 4096 --pairs 4 --pool 7 --max-new 16 \
  --last-logit-only --out artifacts/bounded-decode-frontier-sweep-v1.json

# aggregate, chunked exact prefill, 6 records x 3 lengths x 2 pairs
python benchmarks/benchmark_bounded_decode_frontier.py --lengths 32768 65536 131072 \
  --policies full recent obs_mean obs_last obs_max \
  --budgets 128 256 512 1024 --seeds 101 102 103 104 105 106 \
  --pairs 2 --prefill-chunk 8192 --out artifacts/bounded-decode-frontier-aggregate-v2.json

python benchmarks/summarize_bounded_decode.py \
  artifacts/bounded-decode-frontier-sweep-v1.json \
  artifacts/bounded-decode-frontier-aggregate-v2.json
```

Model path defaults to `/root/qcc/models/Llama-3.2-1B-Instruct`; override with
`--model`. The harness needs only `torch` and `transformers` (no accelerate,
no triton, no vLLM). Raw logs for the recorded runs are
`aggregate_v2.log` and `sweep_v1.log` in the authoring workspace; the JSON
files above contain every per-record prediction, selection statistic and
timing.
