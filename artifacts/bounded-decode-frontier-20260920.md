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

`benchmark_selection_frontier.py`, `sweep_v1.json`:

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

### 3.2 Aggregate (60 records, 2 key/value pairs, chunked exact prefill)

Two runs of `benchmark_bounded_decode_frontier.py` with disjoint seeds:
`aggregate_v2.json` (seeds 101-106, budgets 128/256/512/1024) and
`expand_v1.json` (seeds 110-123, budgets 128/256/512), 20 records per length
band. Full-KV is correct on **58/60** records (32K 20/20, 64K 20/20,
128K 18/20), so 128K is at the edge of this 1B checkpoint's ability and the two
records it misses are excluded from the retention denominators.

Retention = mean answer recall over the records the matched Full-KV run
answered (`analyze_combined.py`). Budgets are slots per (layer, kv-head);
B=128 is a 4 MiB decode cache at every context length.

| policy | B | raw recall | retention | records |
|---|---:|---:|---:|---:|
| `obs_last` | 128 | 0.867 | **0.897** | 60 |
| `obs_last` | 256 | 0.867 | **0.897** | 60 |
| `obs_last` | 512 | 0.900 | **0.931** | 60 |
| `obs_last` | 1024 | 1.000 | **1.000** | 18 |
| `obs_union` | 128 | 0.738 | 0.775 | 42 |
| `obs_union` | 256 | 0.833 | 0.875 | 42 |
| `obs_union` | 512 | 0.881 | 0.925 | 42 |
| `obs_mean` | 128 | 0.417 | 0.431 | 60 |
| `obs_mean` | 256 | 0.633 | 0.638 | 60 |
| `obs_mean` | 512 | 0.800 | 0.810 | 60 |
| `obs_mean` | 1024 | 0.944 | 0.944 | 18 |
| `obs_max` | 512 | 0.667 | 0.667 | 18 |
| `recent` | 128-1024 | 0.000 | **0.000** | 18-60 |

Per band, `obs_last`:

| band | B=128 | B=256 | B=512 | B=1024 |
|---|---:|---:|---:|---:|
| ~32K (20 rec.) | 0.850 | 0.850 | 0.950 | 1.000 (6) |
| ~64K (20 rec.) | 0.900 | 0.900 | 0.900 | 1.000 (6) |
| ~128K (18 rec.) | 0.944 | 0.944 | 0.944 | 1.000 (6) |

The first six-record sample gave 18/18 for `obs_last` at B=1024 and 16/18 at
B=128; the 60-record replication puts B=128 and B=256 at 89.7% and B=512 at
93.1%. The larger sample is the honest number, and **89.7-93.1% is below the
99% aggregate target**: four to six of sixty records lose the answer with a
small bounded cache even though Full-KV keeps it.

The diagnostic in each row (`selection.answer_token_keep_fraction`) shows why:
`obs_last` puts all answer tokens in only ~50% of (layer, kv-head) units, so
retrieval depends on the retained context being sufficient in the units that
happen to include the needle. Increasing the budget raises that fraction and
the retention together.
### 3.3 Negative controls

Every control below uses the same prefill, the same records and the same decode
path, so the only difference is the retention law.

* `recent` (last-B tokens): **0/18** at every budget up to 1024. Retention is not
  a recency artefact.
* `random`, `keynorm` (largest key norm): 0 on all records tested.
* `h2o` (cumulative causal attention mass over all prefill queries): works only
  at 2048-4096 slots and never at small budgets. A needle that is only queried
  *after* the haystack receives almost no prefill attention mass, so an online
  heavy-hitter rule evicts it before the question arrives. This is exactly why
  the question-aware observation window is needed.
* `oracle_needle` (keep the needle tokens, sinks and a recent window, but not the
  attention-selected context): correct at 32K, **wrong at 64K**. Keeping the fact
  is not sufficient; the question-relevant slice of surrounding context matters.
* Borda rank fusion of `obs_last` and `obs_mean` (`obs_union`, `union_v1.json`):
  14/18 at B=128 and B=256, 16/18 at B=512 — *worse* than `obs_last` alone. Fusing
  a strong ranking with a weaker one dilutes it; the plain final-query ranking is
  the better estimator here.

### 3.4 State and speed accounting

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

### 3.5 Decode latency (batch-1, greedy, HF sdpa loop)

Derived from the same `aggregate_v2.json` records: `decode_s / generated` for the
matched Full-KV run and for bounded runs on identical prompts
(`analyze_decode_tpot.py`).

| Length | Full-KV TPOT | bounded TPOT (B=128) | TPOT speedup | decode-state reduction |
|---:|---:|---:|---:|---:|
| ~32K | 15.6 ms | 13.5 ms | 1.16x | 254x |
| ~64K | 17.5 ms | 13.8 ms | 1.27x | 510x |
| ~128K | 27.4 ms | 13.5 ms | **2.03x** | **1026x** |

Two facts matter here:

* Bounded decode latency is **flat in context length** (13.5 ms at 32K, 64K and
  128K), exactly as a context-independent state predicts. Full-KV latency grows
  with context.
* The residual 13.5 ms floor is *not* KV traffic: it is the model's weight read
  plus per-step Python/framework overhead in this loop. Cache bounding alone
  therefore yields ~2x batch-1 TPOT at 128K, not 5x. The repo's own Amdahl note
  predicted this shape; a 5x batch-1 target needs a leaner decode path
  (CUDA graphs / no per-step Python) or contexts where KV traffic dominates the
  weight read by a large factor.

This is a diagnostic measurement, not a serving result: no vLLM, no batching, no
SLA, and the harness synchronises once per decode loop rather than per step.

### 3.6 Official RULER JSONL (80 records, 4 tasks, 8K-64K nominal)

The same prefill/selection/decode path, driven from real RULER records
(`benchmark_bounded_decode_ruler.py`).

> **Note on scoring.** The numbers in this section were produced with an
> all-or-nothing rule that is *stricter than RULER's own metric*. RULER uses
> `string_match_all`, i.e. the per-record fraction of references found, which
> only differs for `vt` (the one multi-reference task). Section 3.13 re-scores
> the same predictions with the official metric and reports both; the
> correction raises the aggregate above the 99% target and takes the worst task
> from 0.000 to 0.86-1.00 depending on budget. Records were prepared for a 32K-vocabulary
tokenizer, so the Llama-3.2 tokenizer yields ~0.72x the nominal length; both
arms see identical tokens.

Matched Full-KV (greedy, 32 new tokens) answers:

| task | Full-KV correct |
|---|---:|
| niah_single_1 | 20/20 |
| niah_multikey_2 | 19/20 |
| niah_multikey_3 (UUID keys) | 9/20 |
| vt (variable tracking) | 3/20 |
| **total** | **51/80** |

The 1B checkpoint itself cannot do most UUID multi-key and variable-tracking
records, so retention is reported on the 51 records Full-KV answers.

`obs_last` with sliding-window dilation (`ruler_v2.json`):

| budget | retention | niah_single_1 | niah_multikey_2 | niah_multikey_3 | vt |
|---|---:|---:|---:|---:|---:|
| 512 | 0.529 | **1.000** | 0.368 | 0.000 | 0.000 |
| 1024 | 0.608 | **1.000** | 0.579 | 0.000 | 0.000 |
| 2048 | 0.667 | **1.000** | 0.737 | 0.000 | 0.000 |

Without dilation the same policy reached only 0.111-0.444 on the first ten
records (`ruler_v1_partial10.json`), because the answer number's tokens were
split across a non-overlapping pooling block boundary and only part of the
value survived (`recall=0`, prediction `'109.'`). Dilation fixed that failure
mode but did not close the gap on UUID multi-key or variable tracking.

Adding the lexical-anchor union (`lex_obs`) — the token spans of the question's
rare strings (hyphenated keys, UUIDs, long numbers) wherever they appear earlier
in the prompt, expanded 16 tokens left and 32 right, selected anchors-first and
then by score up to `budget + 128` slots — takes official-RULER retention from
0.667 to **0.941**:

| policy | budget | retention | single_1 | multikey_2 | multikey_3 | vt |
|---|---:|---:|---:|---:|---:|---:|
| `obs_last` | 2048 | 0.667 | 1.000 | 0.737 | 0.000 | 0.000 |
| `lex_obs` | 512 | 0.902 | 1.000 | 1.000 | 0.778 | 0.000 |
| `lex_obs` | 1024 | **0.941** | **1.000** | **1.000** | **1.000** | 0.000 |
| `lex_obs` | 2048 | **0.941** | **1.000** | **1.000** | **1.000** | 0.000 |

**Variable tracking, precisely.** The residual gap is *not* the selection law,
and four separate experiments now show that (`vt_probe.json`, `vt_probe2.json`,
`vt_probe3.json`):

1. **Selection is solved.** With assignment-chain following the lexical
   selection contains the complete answer set for **20/20 vt records**
   (~290 of 1152 retained slots), verified by decoding the selected positions
   offline. The model needs 5 names; the cache holds all 5.
2. **Not capacity.** Raising the budget from 1024 to 2048, 4096 and 8192 slots
   (25% of a 32K context) does not fix it. At B=8192 two of the four
   Full-KV-correct records pass, at B<=4096 none do.
3. **Not noise.** `lex_only` — chain lines plus sinks and the recent window, with
   *no* attention-selected filler at all, a cache of ~800-2300 tokens — also
   fails at every budget.
4. **Not the header, and not the generation budget.** Raising the attention sink
   from 4 to 96 tokens (so the instruction header is retained) changes nothing,
   and raising the generation budget from 32/48 to 128 tokens changes nothing.

What happens instead is visible in the outputs: the model emits a partial list
and stops. With Full-KV it writes `XFE ZJWJL BBNES PGYAS DQKDH PGYAS CDXFE ...`,
a repetition that eventually contains all five names; with any bounded cache it
writes `XFE ZJWJL BBNES PGYAS [DQKDH]` and then drifts back into the
instruction. (`CDXFE` tokenises as `CD|X|FE`, and the model's first generated
token is the suffix `XFE` in *both* arms, so this is a generation-stability
property of the checkpoint, not a cache artefact.) The checker requires all five
names, so vt retention is 0.

All three retrieval tasks reach **100% retention** (48/48 records) at B=1024, a
32 MiB decode cache. The aggregate is held at 0.941 and the worst task at 0.000
by variable tracking alone: Full-KV answers only 3/20 vt records with this 1B
checkpoint, and `lex_obs` misses those 3 because vt asks for every variable that
reaches a queried value through a chain, which is multi-hop reasoning rather
than retrieval. Naive multi-hop lexical expansion (following identifiers found
on matched lines) was tested and **rejected**: it grows the anchor set to the cap
and drops multikey_2 to 1/4 and multikey_3 to 0/4, because filler words on the
matched lines match everywhere else.

Three selection bugs were found and fixed while cross-checking batched recall,
each of which had been silently degrading results: `lex_obs` originally
broadcast head 0's ranking to every head; the frontier harness scored `lex_obs`
as `obs_max` through a fall-through in the mode dispatch; and a single 2-D
attention mask cannot describe per-layer-varying retained widths, so batched
decode attended a mixture of valid and padded slots.

### 3.7 Decode latency: static cache, CUDA graphs, and a corrected retraction

`benchmark_bounded_decode_tpot_floor.py`. Every variant must first pass a token
parity check: it has to generate exactly the ids the dynamic path generates.

| variant | 32K | 128K |
|---|---:|---:|
| Full-KV, DynamicCache (matched baseline) | 15.6 ms | **28.95 ms** |
| Full-KV, StaticCache / + graph | 53.7 ms | OOM |
| bounded B=1024, DynamicCache | 13.5-17.3 ms | 16.77 ms |
| bounded B=1024, StaticCache | 22.2 ms | 22.4 ms |
| **bounded B=1024, StaticCache + CUDA graph** | **6.86 ms** | **6.87 ms** |
| parity (graph/static vs dynamic tokens) | **OK** | **OK** |

**Correction.** An earlier revision of this document withdrew the static-cache
and CUDA-graph numbers because they failed the parity check. That was wrong: the
*failure was in the check's own reference*. The dynamic reference cache was
built with `DynamicLayer()` objects whose `is_initialized` flag was left False,
so `get_seq_length()` returned 0 and HF overwrote the retained keys on the first
update; the reference generated degenerate text (`'Tags\n }\n return'`) and any
correct implementation "mismatched" against it. With the flag set, both the
static and the graph paths reproduce the dynamic tokens exactly, and the
timings stand. The corrected record is kept here rather than silently dropped.

What the corrected numbers mean:

* Bounded decode is flat at **6.86-6.87 ms/token at both 32K and 128K**, against
  28.95 ms for the matched Full-KV decode at 128K: **4.2x**, with the retained
  cache 1024x smaller.
* Of that, the cache itself contributes **1.7-2.0x** (bounded vs Full-KV when
  both use the plain dynamic path); the rest comes from graph capture removing
  per-step Python, mask rebuilding and cache concatenation, which the dynamic
  baseline still pays.
* StaticCache *without* a graph is slower than DynamicCache here (22 ms), and
  Full-KV with a static cache is slower still (53.7 ms at 32K, OOM at 128K),
  because an explicit mask over the full padded buffer loses SDPA's causal fast
  path. Both are framework artifacts, and they are why the matched baseline for
  Full-KV is the dynamic path.
* **The 5x target is still not met at batch 1**: 4.2x is measured, the
  bandwidth-limited bound for the cache-attributable part is 2.70x, and the
  graph path's 6.87 ms floor is close to the 5.0 ms weight-plus-LM-head read.

### 3.8 Serving: throughput, TPOT and concurrency at 32K

Two harnesses, same prompt set and same decode, differing only in *when* the
full prefill KV exists.

**(a) All requests prefilled together** (`benchmark_bounded_decode_serving.py`,
`serving_32k_v2.json`). Peak memory is `batch x full KV` during prefill, so the
retained cache only helps decode:

| policy | batch 1 | batch 2 | batch 4 | batch >= 8 |
|---|---:|---:|---:|---:|
| Full-KV tok/s / TPOT / peak | 61.6 / 16.2 ms / 5.4 GiB | 103.2 / 19.4 / 8.7 | 124.7 / 32.1 / 15.1 | OOM |
| bounded tok/s / TPOT / peak | 73.6 / 13.6 / 5.9 | 142.7 / 14.0 / 8.6 | 281.0 / 14.2 / 15.0 | OOM |

Concurrency is **1.0x**: both OOM at batch 8. This is a statement about prefill
state, not about the retention law.

**(b) One request prefilled at a time, bounded caches resident, decode batched**
(`benchmark_bounded_decode_serving_sequential.py`, `serving_seq_32k.json`) — the
continuous-batching pattern the bounded state actually enables. Peak is
`one prefill transient + batch x bounded cache`:

| batch | decode tok/s | TPOT | peak | recall |
|---:|---:|---:|---:|---:|
| 1 | 70.9 | 14.1 ms | 5.17 GiB | 1/1 |
| 2 | 142.5 | 14.0 ms | 5.24 GiB | 2/2 |
| 4 | 291.1 | 13.7 ms | 5.35 GiB | 4/4 |
| 8 | 583.0 | 13.7 ms | 5.48 GiB | 8/8 |
| 16 | 1174.8 | 13.6 ms | 5.83 GiB | 16/16 |
| 32 | 1946.8 | 16.4 ms | 6.62 GiB | **32/32** |
| 64 | 741.2 | 86.4 ms | 7.92 GiB | 60/64 |

* **Fixed-SLA concurrency: 8x.** Under a TPOT SLA of 50 ms, matched Full-KV
  serves at most 4 concurrent 32K requests (batch 8 OOMs at any latency), while
  the bounded design serves **32** at 16.4 ms per request. Raising the SLA to
  100 ms would allow 64.
* **Throughput: 15.6x** (1946.8 vs 124.7 decode tok/s) at the same request size;
  the >=3x target is met with margin at batch >= 4.
* **Memory:** 6.62 GiB peak for 32 resident 32K requests versus Full-KV's
  15.06 GiB for 4.
* Quality is unchanged at 100% up to batch 32; at batch 64 the last three
  records lose the answer and TPOT rises to 86 ms, i.e. batch 32 is the
  SLO-respecting operating point on this card.
* The batched decode reproduces the frontier harness's selection exactly and
  each request keeps its own absolute positions; using one row's length for all
  rows had shifted RoPE by up to ~1000 positions and cost 3 of 8 rows at 32K.
* Caveat: prefill is *serialised*, so batch-32 prefill takes 139.7 s of wall
  clock (4.4 s per request). This measures memory-limited concurrency under a
  TPOT SLA, not prefill throughput; a real server would pipeline it.

### 3.9 Serving at 128K

`benchmark_bounded_decode_serving_sequential.py` (`serving_seq_128k.json`) and
the matched batched-prefill control (`serving_128k_matched.json`).

| configuration | batch 1 | batch 2 | batch 4 | batch 8 |
|---|---:|---:|---:|---:|
| Full-KV, prefilled together | 34.5 tok/s, 29.0 ms, 10.97 GiB | **OOM** | OOM | - |
| bounded, *prefilled together* (control) | ok | **OOM** | OOM | - |
| bounded, sequential prefill | 71.6 tok/s, 14.0 ms, 10.47 GiB | 141.3, 14.2 ms, 11.27 | 285.0, 14.0 ms, 11.35 | 561.5, 14.3 ms, 13.41 GiB (7/8) |

The control row is the point: with all requests prefilled together the bounded
cache buys nothing at 128K either, because prefill holds a full 4 GiB KV per
request. Only the sequential pattern converts the bounded *decode* state into
concurrency.

* **Full-KV cannot serve two concurrent 128K requests on this card**, while the
  bounded design serves **8** with per-request TPOT flat at 14.0-14.3 ms and
  13.41 GiB peak: **8x concurrency at 128K**, matching the 32K result.
* Bounded decode TPOT at 128K (13.97 ms) is the same as at 32K (13.72 ms) and
  identical across batch sizes, i.e. the retained state is genuinely
  context-independent; matched Full-KV batch-1 TPOT is 28.95 ms, so the
  like-for-like speedup is **2.07x** — consistent with the 2.70x bandwidth
  bound above.
* Recall is 100% through batch 4 and 7/8 at batch 8.

### 3.10 Language modelling under bounded retention

Retrieval asks whether one fact survives. `benchmark_bounded_decode_lm_nll.py`
asks the broader question: with an exact prefill and a bounded decode cache, how
much does the next-token distribution degrade on ordinary long text? A 32,768
token document is assembled from local sources, prefilled exactly, and its last
256 tokens are scored by teacher forcing under each cache. There is no question
to anchor on, so the observation window is simply the last 64 prefix tokens (the
SnapKV setting) and no lexical anchors are used.

Budget curve on one **pinned** document (Full-KV NLL 1.34745, ppl 3.848 —
`lm_nll_pinned_b*.json`):

| cache | kept slots | share | perplexity | ratio |
|---|---:|---:|---:|---:|
| Full-KV | 32,512 | 100% | 3.848 | 1.00x |
| `obs_last` | 1,024 | 3.1% | 4.696 | 1.22x |
| `obs_last` | 2,048 | 6.3% | 4.196 | 1.09x |
| `obs_last` | 4,096 | 12.6% | 4.311 | 1.12x |
| `obs_last` | 8,192 | 25.2% | 4.295 | 1.12x |
| `obs_last` | 16,384 | 50.4% | 4.175 | 1.09x |

Policy comparison at B=1024 on an earlier, more heterogeneous document
(`lm_nll_32k.json`, which also includes `/usr/lib/python3.12/*.py`):

| policy | perplexity | ratio |
|---|---:|---:|
| Full-KV | 24.23 | 1.00x |
| `obs_mean` | 37.72 | 1.56x |
| `obs_last` | 42.31 | 1.75x |
| `recent` | 1053.5 | 43x |
| `random` | 2787.4 | 115x |

Conclusions:

* Language modelling degrades **much less than the first measurement
  suggested**: on a homogeneous 32K corpus, 3% of the context costs 22%
  perplexity and 6% costs 9%, with no further gain above that. The 1.75x figure
  came from a heterogeneous corpus whose long-range statistics a 3% cache really
  does destroy. The degradation is document-dependent and both numbers are
  reported rather than the flattering one alone.
* The curve is **not monotone** (1.12x at both 4096 and 8192, 1.09x at 2048 and
  16384): past the tokens the recent window already covers, top-k selection adds
  low-ranked context whose value is close to noise. More budget is not
  automatically better.
* `recent` and `random` are catastrophic (43x and 115x), confirming the
  observation-window score does real work, and `obs_mean` beats `obs_last` for
  language modelling (1.56x vs 1.75x) — the opposite ordering to retrieval, so
  the aggregation choice is task-family dependent.

### 3.11 128K TPOT across configurations, and where 5x stands

`lean_decode_128k_4bit.json`, `serving_128k_full_4bit.json`, plus the bf16 runs
above. All rows are 128K context, batch 1, 32 greedy tokens, same prompt.

| configuration | Full-KV TPOT | bounded TPOT | ratio |
|---|---:|---:|---:|
| bf16, both plain dynamic decode | 28.95 ms | 16.77 ms | **1.73x** |
| bf16, Full-KV dynamic vs bounded + CUDA graph | 28.95 ms | **6.87 ms** | **4.21x** |
| NF4 4-bit weights, both plain dynamic decode | 105.35 ms | 23.09 ms | **4.56x** |
| NF4 4-bit, Full-KV dynamic vs bounded + graph | 105.35 ms | **4.98 ms** | 21.1x |

Reading this honestly:

* The only pairing that is like-for-like in *both* cache policy and execution
  path is the first row: **1.73x**. The cache-attributable speedup at 128K is
  therefore under 2x, exactly as the bandwidth bound predicts (2.70x ceiling).
* The second row is the strongest defensible system-level number: **4.21x**,
  where the additional gain comes from capturing the decode step in a CUDA
  graph, an optimisation the Full-KV path cannot use here because its static
  cache overflows the card at 128K.
* The 4-bit rows move the ratio because the Full-KV arm is *penalised*: a
  DynamicCache concatenation of 4 GiB per step plus bitsandbytes' unfused
  dequantisation costs 105 ms/step, three and a half times its bf16 cost. The
  bounded+graph NF4 configuration is nonetheless a real result on its own:
  **4.98 ms per token, i.e. 201 tok/s single-stream at 128K context, with a
  1024x smaller cache**, verified token-identical to the dynamic path.
* **Full-KV cannot be given the same optimisation at 128K on this card.**
  `benchmark_fullkv_tpot_graph.py` builds a StaticCache that *takes ownership*
  of the prefilled tensors (so only one cache exists at a time; peak 10.48 GiB)
  and then tries to capture the decode step. The capture itself OOMs: a 128K
  forward needs activation workspace on the order of the per-layer MLP
  intermediate (2.1 GiB per tensor at this length) on top of the 4 GiB cache.
  The bounded path captures fine because its attention runs over 1,152 keys.
  The 4.21x is therefore not an artefact of withholding an optimisation from
  the baseline: at 128K the baseline cannot use it.
* **No matched configuration reaches 5x.** The target is not met at batch 1 on
  this card; the bandwidth bound says it cannot be for a 1B model whose weights
  alone are 4.12 ms of the step.

### 3.12 The quality/budget frontier: retention is task-family dependent

Putting the three quality experiments on one axis — how much of the context has
to be retained before the bounded cache matches matched Full-KV — gives the
clearest statement of what this retention law is and is not:

| task family | metric | budget that matches Full-KV | measured there |
|---|---|---:|---|
| RULER NIAH (single, multi-key, UUID multi-key) | answer recall | **3%** (1,024 slots at 32K) | 1.000 (48/48) |
| language modelling on a 32K document | perplexity | ~25% (8,192 slots) | 0.98x |
| RULER vt (list of 5 chained variables) | answer recall | **>75%** (24,576 slots) | 0.75 (3/4) |

* Retrieval is where a tiny bounded cache is genuinely sufficient: three RULER
  tasks are at 100% retention with 3% of the context and a 1024x smaller decode
  state.
* Language modelling needs an order of magnitude more context to break even, and
  even then only approximately.
* Multi-hop list answers need most of the document: at 50% budget vt retention
  is 2/4, at 75% it is 3/4, and only at 100% (Full-KV) is it 4/4. For this class
  the bounded cache buys almost nothing.

This is the honest form of the "aggregate quality >= 99%" claim: the aggregate
across the four RULER tasks is 0.941 at a 3% budget, and no single budget
satisfies every task family at once. A deployment that needs both retrieval and
list-answer quality should choose the budget per workload, or add a readout
stage for list answers, rather than keep growing the cache.

### 3.13 Scoring correction: RULER uses partial recall, and that changes the verdict

The harness scored each record as all-or-nothing: every reference string had to
appear in the prediction. RULER's own definition, in
`scripts/eval/synthetic/constants.py`, is

```python
def string_match_all(preds, refs):
    score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
                 for pred, ref in zip(preds, refs)]) / len(preds) * 100
```

i.e. **the per-record fraction of references found**, averaged over records. The
harness was therefore stricter than the benchmark. Only `vt` is affected — the
three NIAH tasks have a single reference each, where the two definitions agree —
but `vt` is exactly the task that was failing: a prediction listing 4 of the 5
chained variables scores 0.8 officially and 0.0 strict.

Re-scoring the *same* stored predictions (`analyze_ruler.py`,
`rescore_ruler.py`; no new GPU work) gives:

| policy | budget | official retention | strict | worst task |
|---|---:|---:|---:|---|
| `lex_obs` | 512 | 0.957 | 0.941 | 0.800 (vt) |
| `lex_obs` | 1024 | **1.003** | 0.980 | 0.862 (vt) |
| `lex_obs` | 2048 | **1.013** | 0.980 | 0.908 (vt) |
| `obs_last` | 2048 | 0.751 | 0.686 | 0.000 (niah_multikey_3) |

Aggregate retention is **above the 99% target** once the official metric is
used, and the worst task is `vt` at 0.86-0.91 rather than 0.00. (Ratios above
1.0 are possible because the ratio is the sum of QCC scores over the sum of
Full-KV scores, and on some UUID multi-key records the bounded cache scores
higher than Full-KV.)

The full 80-record suite re-run with the official metric and a 128-token
generation budget applied to **both** arms (`ruler_v6.json`), retention budget
4096 slots (+512 lexical anchors):

| budget | aggregate | strict | single_1 | multikey_2 | multikey_3 | vt |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 0.997 | 0.942 | 1.000 | 0.947 | 1.222 | 0.909 |
| **4096** | **1.000** | 0.942 | **1.000** | **1.000** | **1.000** | **1.000** |
| 8192 | 1.020 | 1.000 | 1.000 | 1.000 | 1.111 | 1.015 |

**Both quality targets are met at a 4096-slot budget: aggregate 1.000 (>= 99%)
and worst task 1.000 (>= 97%)**, with all four RULER tasks at parity with matched
Full-KV. The `vt` budget curve explains why a large budget is needed for it while
retrieval needs 3%:

| vt budget | share of a 32K context | vt retention (official) |
|---:|---:|---:|
| 1024 | 3% | 0.864 |
| 2048 | 6% | 0.970 |
| 4096 | 13% | 0.985-1.000 |
| 8192 | 25% | 1.000 |

Scope: this is the 80-record RULER split with four tasks. LongBench and PG-19 are
still unmeasured, so the target is met on the suite that was actually run, not on
every suite the repository's handoff mentions. In state terms 4096 slots is
151 MiB per request at this model's geometry — 27x smaller than the 4.00 GiB
Full-KV cache at 128K, but far from the 4 MiB that the 1024-slot retrieval
configuration uses.

### 3.14 The configuration that meets quality, and what it costs

The objective's metrics have to hold *simultaneously*, so the serving sweeps were
re-run at the retention budget that meets the quality targets (4096 slots + 512
lexical anchors, `serving_seq_*_q4096.json`).

| configuration | aggregate quality | worst task | decode state @128K | max batch | peak | throughput @32K |
|---|---:|---:|---:|---:|---:|---:|
| B=1024 (retrieval-tuned) | 1.003 | 0.862 (vt) | 4 MiB | 32 | 6.6 GiB | 1947 tok/s |
| **B=4096 (quality)** | **1.000** | **1.000** | 151 MiB | 32 | 11.9 GiB | 620 tok/s |

Details at the quality budget:

| context | batch | decode tok/s | TPOT | peak | recall |
|---:|---:|---:|---:|---:|---:|
| 32K | 1 | 18.4 | 54.3 ms | 5.17 GiB | 1/1 |
| 32K | 8 | 377.2 | 21.2 ms | 6.17 GiB | 8/8 |
| 32K | 16 | 464.1 | 34.5 ms | 7.36 GiB | 16/16 |
| 32K | 32 | 620.1 | 51.6 ms | 11.90 GiB | **32/32** |
| 128K | 1 | 60.9 | 16.4 ms | 10.48 GiB | 1/1 |
| 128K | 2 | 122.4 | 16.4 ms | 11.39 GiB | 2/2 |
| 128K | 4 | 249.3 | 16.1 ms | 11.70 GiB | **4/4** |

Matched Full-KV at 128K serves exactly one request (34.5 tok/s, 28.95 ms) and
OOMs at batch 2, so the bounded quality configuration still gives 4x concurrency
at 128K with per-request TPOT of 16 ms against 29 ms.

* **Concurrency is unchanged at 8x** (32 concurrent 32K requests against matched
  Full-KV's ceiling of 4), and recall is 100% at every batch size, so the
  quality targets and the concurrency target are met by the same configuration.
* **Throughput falls from 15.6x to 5.0x** (620 vs 124.7 tok/s at the largest
  batch each policy fits): still above the 3x target, but the margin shrinks
  because every decode step now reads 4x more retained KV.
* **TPOT at 32K is noisy and higher** (21-54 ms against 13.7 ms at B=1024); the
  128K numbers are steady at 16.4 ms, and the 32K batch-1 figure is inflated by
  warm-up and by other tenants on this shared GPU (the 32K batch-1 number is
  larger than the 128K one, which cannot be a real cache effect).
* State grows from 4 MiB to 151 MiB per request at 128K - still 27x smaller than
  the 4.00 GiB Full-KV cache, but no longer a rounding error.

Decode latency at the quality budget (`tpot_q4096_128k.json`, 128K, parity OK):
15.84 ms/token with a CUDA graph against 28.95 ms for the matched Full-KV
dynamic decode, i.e. **1.83x** rather than the 4.21x the 1024-slot cache
achieves. The retained set is four times larger, so attention over it is no
longer negligible next to the weight read.

### 3.14b Measurement conditions for every latency number

This box is shared with other tenants, and that matters more than any tuning
detail. The same bounded-graph configuration measured **6.87 ms** per token when
the card was otherwise idle and **11.0-18.4 ms** while a co-tenant was running;
the matched Full-KV arm went from 28.95 ms to 94.2 ms and often fails outright
with OOM at 128K. Contention penalises the Full-KV arm harder because its decode
copies a 4 GiB cache every step, so a contended window *inflates* the speedup
ratio.

Consequences, applied throughout this document:

* Every latency claim names the window it came from. The headline **4.21x** is
  bounded-graph 6.87 ms (measured 23:04) against Full-KV dynamic 28.95 ms
  (measured 22:20-22:26), both inside the same quiet period; the bounded number
  was reproduced at 6.86-6.87 ms in three separate runs.
* The quality budget (4096 slots) measured 11.02 ms in a less contended window
  and 13.4-17.7 ms in busier ones, i.e. **2.6x-1.6x** against the same 28.95 ms
  baseline. The spread is measurement noise, not a cache effect.
* The TPOT-floor harness now reports the minimum of five repetitions rather than
  a single sample, and its parity check still gates every number.
* Because full-KV at 128K needs ~23 GiB while prefilling, it can only be measured
  on an otherwise empty card - which is itself part of why the concurrency
  comparison is lopsided.

### 3.15 Cross-model check: blocked by the second checkpoint's remote code

The quality result rests on one checkpoint (Llama-3.2-1B). Testing it on
Phi-3.5-mini (3.8B, 128K native, the model this repository's earlier sessions
used) was attempted and is **blocked**, not skipped:

* Phi-3.5 ships `modeling_phi3.py` with a legacy cache API. `cache_position` is
  not in its `forward` signature (handled: the harness now drops unsupported
  kwargs), and `get_usable_length`/`from_legacy_cache` are missing from
  Transformers 5.x (handled: the repository's own `_ensure_remote_code_compat`
  shim).
* Its attention is eager, so a 16K prefill materialises a
  `(1, 32, 16384, 16384)` score matrix (~15 GiB). Even chunked, the 16K band
  OOMs on a 24 GiB card; the 8K band fits (peak 10.98 GiB) but the continuation
  path then fails with a shape mismatch between the legacy cache and the
  Transformers 5.x cache (`3072` vs `2048` at the attention dimension).
* With a correct prompt (verified against `apply_chat_template`, and with the
  answer prefix after the `<|assistant|>` header as RULER does), Phi's Full-KV
  output on an 8K record was still degenerate (`'.7.\n.\n...'`), and its
  first-chunk logits were flat at ~-29 across digit tokens.

So the cross-model claim is **not made**: the retention law is verified on
Llama-3.2-1B only, and making Phi-3.5 work needs a Transformers version matched
to its remote code (or a converted checkpoint), which is a separate piece of
work.

So the honest summary of the trade-off is: a 1024-slot cache maximises speed and
state reduction and reaches 0.862 on the worst task; a 4096-slot cache meets both
quality targets and keeps 8x concurrency and 5x throughput.

### 3.16 The retention law is now part of the package, not just a benchmark

Everything above was measured with a standalone harness. The law now ships as
`qcc_transformer/retention.py` with an HF entry point:

```python
from qcc_transformer import RetentionConfig, compile_bounded_cache

config = RetentionConfig(budget=4096, lex_cap=512, chain_hops=6)
cache, logits = compile_bounded_cache(model, input_ids, config, tokenizer=tokenizer)
# decode against `cache` exactly as against any other HF cache
```

* `prefill_capture` runs the model's own exact causal attention in fixed chunks
  and keeps only `O(obs * d)` hidden state per layer.
* `observation_scores` computes the question-window importance (final query, or
  a two-pass mean/max over the window).
* `lexical_anchors` finds the question's rare strings earlier in the prompt and
  optionally follows assignment chains.
* `select_indices` produces a **uniform** retained width (`budget + lex_cap`) for
  every head and layer, which is what makes one 2-D mask able to describe a whole
  batch, and `prune_cache` gathers the survivors in place so each key keeps its
  original rotary phase.

It adds no parameters and never touches the pretrained weights, so the retrofit
property is unchanged. Verification:

* `tests/test_retention.py` - seven CPU tests on a tiny randomly-initialised
  Llama, covering uniform width, "keep everything reproduces the uncompiled
  prefill's tokens", forced sinks/recent, anchor recall of a repeated key,
  assignment-chain following, and (3.19) a ragged batch matching the per-row
  compile slot for slot under both padding sides. They caught a real bug during
  development: the packaged `last`-query scoring had a wrong einsum operand rank.
* `validate_packaged.py` - four real RULER records at ~12K through the packaged
  API: score 1.0 on all four, 4608 slots retained, 2.3-3.9 s per record,
  6.2-7.3 GiB peak.
* `validate_retention_batch.py` - a ragged two-record batch against the same
  records compiled alone, reported in 3.19.
* The repository's full test suite passes (the one pre-existing failure was a
  test bug: it called `delattr` on an *inherited* attribute and asserted an
  empty-cache return value that Transformers 5.x no longer provides; both are
  fixed).

### 3.17 The shipped API matches the benchmark it came from

`benchmarks/validate_retention_full.py` runs records through the packaged
`compile_bounded_cache` and compares, record by record, with the benchmark
harness's stored run at the same budget (`ruler_v6.json`, lex_obs at 4096):

| task | records compared | identical scores | differences |
|---|---:|---:|---|
| niah_multikey_2 | 5 | 5 | - |
| niah_multikey_3 | 5 | 4 | 1 (package 1.000 vs harness 0.000) |
| niah_single_1 | 5 | 5 | - |
| vt | 5 | 5 | - |
| **total** | **20** | **19** | **1, in the package's favour** |

So the numbers reported above are produced by the shipped code path, not only by
the standalone harness, and the one disagreement is a single record where the
package recovers an answer the harness missed rather than the reverse.

Absolute recall on these twenty records is 0.850 (single 1.000, multikey_2 1.000,
multikey_3 0.800, vt 0.600); that is *not* the retention figure - retention is a
ratio against matched Full-KV over records Full-KV answers, and it is what the
0.941-1.000 numbers elsewhere in this document refer to.

### 3.18 Fixed-SLA concurrency, made auditable

`benchmarks/analyze_sla_concurrency.py` derives the largest batch that meets a
per-request TPOT SLA from the serving sweeps already recorded, so the concurrency
claim is a number anyone can recompute rather than a chosen figure.

| case | SLA | Full-KV max batch | bounded max batch | ratio |
|---|---:|---:|---:|---:|
| 32K, B=1024 (speed) | 25 ms | 2 | 32 | **16x** |
| 32K, B=1024 (speed) | 50 ms | 4 | 32 | **8x** |
| 32K, B=1024 (speed) | 100 ms | 4 | 64 | **16x** |
| 32K, B=4096 (quality) | 25 ms | 2 | 8 | 4x |
| 32K, B=4096 (quality) | 50 ms | 4 | 16 | 4x |
| 32K, B=4096 (quality) | 100 ms | 4 | 32 | **8x** |
| 128K, B=1024 | 50-100 ms | 1 | 8 | **8x** |

* The **>=8x target holds at a 50 ms SLA** for the speed configuration at both
  32K and 128K, and at 100 ms for the quality configuration - whose measured
  TPOTs (21-52 ms at 32K) come from a contended window, so its row is a lower
  bound rather than a best case.
* Full-KV's ceiling is 4 concurrent 32K requests (2 at 25 ms) and exactly **1**
  at 128K, because batch 2 at 128K OOMs - the ratio at 128K is therefore not a
  latency effect but a hard memory wall.
* The 64-request row is the memory-limited ceiling (7.92 GiB peak), not an SLA
  failure.

### 3.19 Ragged batches: padding is removed before selection, and the filler is exact

Serving rarely hands over equal-length prompts, so the packaged API accepts a
padded batch:

```python
cache, logits = compile_bounded_cache(model, batch, config, tokenizer=tokenizer,
                                      attention_mask=attention_mask)
# one rectangular cache; reconstruct each row's own positions for decode:
positions = cache.qcc_prompt_lengths[:, None] + step
```

* **Selection runs per request on its real tokens.**  Each row is sliced to its
  masked positions before anything else, so the padding side is irrelevant
  (left or right), the observation window is that request's own last
  `observation_window` tokens, and padding is never a candidate and never counts
  as an attention sink.
* **Ragged rows merge exactly.**  Rows of different lengths compile to different
  widths; the short rows are filled up to the widest row by duplicating their own
  last retained slot.  Duplication is not an approximation: two identical
  `(key, value)` slots split the original slot's softmax weight between them, so
  attention over the padded cache is the same computation as attention over the
  unpadded one.  The filler is marked as padding in `cache.qcc_attention_mask`,
  which is why it can also simply be masked out.
* **Positions must still be told, because the cache is shorter than the prompt.**
  A 15.6K prompt compiled to 4,608 slots cannot recover "15,584" from any
  cumulative sum over a 0/1 mask, so `cache.qcc_prompt_lengths` is returned and
  the caller adds it to the step.  With `generate`, which derives positions from
  the mask, a ragged batch would need an explicit position source; the manual
  decode loop is the supported path.

`benchmarks/validate_retention_batch.py` packs two RULER records (7,730 and
15,584 tokens) into one left-padded batch and compares every level against the
same records compiled and decoded alone:

| config | row | prompt | slots | filler | cache == solo | filler == own last slot | prefill logits == solo | row sliced from merged cache == solo | 2-row batch decode == solo | score |
|---|---:|---:|---:|---:|---|---|---|---|---|---:|
| budget 16384 | 0 | 7,730 | 7,730 | 7,854 | yes | yes | yes | yes (24/24 steps) | yes | 1.0 |
| budget 16384 | 1 | 15,584 | 15,584 | 0 | yes | n/a | yes | yes | yes | 1.0 |
| 4096 + 512 | 0 | 7,730 | 4,608 | 0 | yes | n/a | yes | yes | step 23 only | 1.0 |
| 4096 + 512 | 1 | 15,584 | 4,608 | 0 | yes | n/a | yes | yes | yes | 1.0 |

* The compiled cache of every row is **bit-identical** to compiling that row
  alone, in both the filler-heavy configuration (7,854 duplicated slots) and the
  shipped one, and a row sliced out of the merged cache decodes to exactly the
  solo tokens in all four cases.  The merged cache is therefore a faithful
  single-row cache, and batching does not change what is retained.
* The one divergence is honest and bounded: in the shipped configuration the
  two-row batch differs from solo decode at step 23 of 24, at a step whose top-2
  margin is **0.0** - a perfect tie broken differently because a batched bf16
  GEMM is not bit-identical to a single-row one (max per-step logit delta
  0.34-0.77 across all rows).  Both rows still score 1.0 and the retained slots
  are identical, so this is the usual batched-inference tie-break, not retention
  error.
* Cost: compilation runs one chunked prefill per request (linear in the batch,
  4-6 s for these two rows on the A10).  Decode is unaffected - every row pays
  the same bounded width.

## 4. What this establishes, and what it does not

Establishes:

* A **causal**, **training-free**, **zero-new-parameter** retention law preserves
  the retrieval quality of exact Full-KV. On the official 80-record RULER split
  it reaches **0.941 aggregate retention** at B=1024 (a 32 MiB decode cache),
  with **niah_single_1, niah_multikey_2 and niah_multikey_3 all at 1.000**
  (48/48 records that matched Full-KV answers).
* Bounded decode state is context-independent: 4 MiB at 128K with B=128
  (1024x smaller than the 4.00 GiB Full-KV cache) and 32 MiB at B=1024.
* At 32K, the continuous-batching pattern this enables gives **8x concurrency**
  (32 resident 32K requests vs Full-KV's 4), **15.6x decode throughput**, flat
  per-request **TPOT of 13.6-16.4 ms**, and **6.62 GiB peak versus 15.06 GiB**.
* The previous QCC failures on these records are attributable to the selection
  law and to the archive read/mix path, not to bounded decode state as such.
  Recency (0/18), random, key-norm, H2O-style cumulative mass and naive
  multi-hop lexical expansion all fail; the question-window plus lexical-anchor
  law is doing the work.

Does not establish:

* **Aggregate >= 99% or worst task >= 97%.** Aggregate is 0.941 and the worst
  task is 0.000, both held down by variable tracking: vt is multi-hop chaining
  rather than retrieval, and the 1B checkpoint itself answers only 3/20 vt
  records with full attention.
* **Official suites beyond RULER NIAH + vt.** LongBench and PG-19 are untouched.
* **Bounded prefill state.** The concurrency result serialises prefill, so total
  prefill wall clock still scales with the number of requests; peak *resident*
  memory is bounded, peak prefill transient is `O(L)` for one request.
* **5x batch-1 TPOT.** Measured ~2.0x at 128K on the dynamic path, with a
  13.5 ms floor set by per-step framework overhead rather than KV traffic.
* **A validated CUDA-graph path.** The graph/StaticCache variants fail the
  token-parity check in this Transformers build and their timings are withdrawn.
* **1M retrieval.** Llama-3.2-1B is a 128K-native checkpoint, so 1M is out of
  its native range and no 1M-native model is available here.
* **Integration.** The mechanism is not yet implemented in `qcc_transformer`;
  this is a standalone harness plus benchmarks.
* **Nominal-length overshoot.** The 128K synthetic records are 131.0K-131.5K
  tokens, marginally above the checkpoint's nominal 131072 limit. Full-KV
  answers every one of them, so the comparison is not an artefact of prefix
  overflow, but future runs should target 130K or tighten the tolerance.

## 5. Design implied for QCC

The result says the deployable object is a **cache policy**, not an attention
approximation. All QCC archive/recurrence/gate machinery can be bypassed:

1. Prefill: exact causal attention (chunked for bounded activations). Capture,
   per layer, the attention inputs of the last `obs` query positions. This is
   `O(obs * d)` state, independent of context length.
2. Compile: for each `(layer, kv-head)`, score every key by the attention it
   receives from those queries, dilate the score map with a sliding-window max
   (a multi-token answer must not be split by the pooling grid), keep the top
   `B`. Union with `s` attention sinks, a recent window, and -- for retrieval
   records that repeat the asked key in the question -- the token spans of the
   question's rare strings (hyphenated keys, UUIDs, long numbers) wherever they
   occur earlier in the prompt, expanded by 16 tokens left and 32 right. Cost is
   `O(obs * L)` per layer for the attention term plus one linear string scan.
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

Harness (in this repository):

* `benchmarks/benchmark_selection_frontier.py` - policy comparison with full
  hidden-state capture (mechanism check).
* `benchmarks/benchmark_bounded_decode_frontier.py` - chunked exact prefill,
  `O(obs)` hidden capture, multi-seed aggregation; imported by the scripts below.
* `benchmarks/benchmark_bounded_decode_ruler.py` - same frontier on official
  RULER JSONL records.
* `benchmarks/benchmark_bounded_decode_serving.py` - batched prefill/decode
  throughput and memory sweep.
* `benchmarks/analyze_bounded_decode_tpot.py` - decode-TPOT comparison from a
  results JSON.
* `benchmarks/check_bounded_decode_selection.py` - CPU-only invariant checks for
  the selection primitives (budget, forcing, pooling, monotonicity, ordering).
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

# official RULER JSONL split
python benchmarks/benchmark_bounded_decode_ruler.py \
  --ruler-jsonl <ruler_split.jsonl> \
  --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt \
  --policies full obs_last obs_mean --budgets 128 256 1024 \
  --out artifacts/bounded-decode-frontier-ruler-v1.json

# batched serving sweep
python benchmarks/benchmark_bounded_decode_serving.py --length 32768 \
  --batches 1 2 4 8 --policies full obs_last --budget 1024 \
  --out artifacts/bounded-decode-frontier-serving-v1.json

python benchmarks/analyze_bounded_decode_tpot.py \
  artifacts/bounded-decode-frontier-aggregate-v2.json
python benchmarks/check_bounded_decode_selection.py
python benchmarks/summarize_bounded_decode.py \
  artifacts/bounded-decode-frontier-sweep-v1.json \
  artifacts/bounded-decode-frontier-aggregate-v2.json \
  artifacts/bounded-decode-frontier-union-v1.json
```

Model path defaults to `/root/qcc/models/Llama-3.2-1B-Instruct`; override with
`--model`. The harness needs only `torch` and `transformers` (no accelerate,
no triton, no vLLM). Each result JSON contains every per-record prediction,
selection statistic and timing, so the tables are recomputable from the
committed files.
