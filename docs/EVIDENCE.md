# Evidence index

This page maps the headline measurements to stored artifacts. Raw measurement
artifacts are kept unchanged. Implementation and reproduction commands are
temporarily withheld pending publication.

Measurements are listed in two groups: the current set, and additional measurements on tasks the
current set does not cover.

## Additional measurements

| Measurement | Result | Evidence |
|---|---|---|
| RULER, current Llama-3.2-1B re-run | aggregate retention **1.0102**, worst task **1.0000** | `artifacts/bounded-decode-frontier-multimodel-llama1b-recheck.json` |
| Shipped API / harness agreement | 80/80 metric agreement; 75/80 byte-identical predictions | `artifacts/bounded-decode-frontier-parity-80.json` |
| LongBench, nine tasks | aggregate retention **0.9939** with rank-blend filler | `artifacts/longbench-blend-9task.json` |
| Matched-width selection baselines | QCC **1.008** vs window mean 0.880, final query 0.842, sinks+recent 0.332, sliding 0.167 | `artifacts/bounded-decode-frontier-baselines-b4096.json` |
| Published-family shaped baselines | stored Quest/Pyramid/H2O/TOVA/SnapKV comparisons | `artifacts/baselines-quest-pyramid.json`, `artifacts/baselines-accumulated.json` |
| KV quantization | bounded + int8 matches Full-KV quality at **72.3 MiB** | `artifacts/bounded-decode-frontier-kv-quant.json` |
| Multi-checkpoint comparison | Llama, Qwen and Phi measurements with one retained-width configuration | `artifacts/bounded-decode-frontier-multimodel-*.json` |

## Second measurement set

**Rows at parity** means the row's score equals the matched Full-KV score exactly — stricter than a
mean.

### Compression, with the two axes separated

State bytes factor as `(L / B) x (16 / d)`: which slots are kept, and how many bits each costs.
Quantisation is applied **after** selection, so a quality delta is attributable to bits rather than to
a changed working set.

**75,003-token retrieval context, 20 rows, all retained:**

| retained state | mean | rows at parity | time | space | **total** |
|---|---:|---:|---:|---:|---:|
| 128 slots, bf16 | 1.0000 | **20/20** | 586× | 1× | 586× |
| **128 slots, int4** | **1.0000** | **20/20** | 586× | 4× | **4,688×** |

`P6_b64.json`, `CQ_p6_b64_int4.json`, `P6_full.json`

**50,146-token context**, the earlier point, where the same composition holds:

| retained state | mean | rows at parity | time | space | **total** |
|---|---:|---:|---:|---:|---:|
| 160 slots, bf16 | 1.0000 | **20/20** | 313× | 1× | 313× |
| 160 slots, int8 | 1.0000 | **20/20** | 313× | 2× | **1,172×** |
| 128 slots, int4 | 1.0000 | **20/20** | 313× | 4× | **4,688×** |

`DP_p4_b160.json`, `CQ_p4_b160_int8.json`, `CQ_p4_b160_int4.json`

**Both quantisation modes are free at this budget.** They are not free at every budget: at 4,096
slots int4 halves quality (0.8250 → 0.4625) while int8 remains free
(`bounded-decode-frontier-kv-quant.json`). **At 160 slots the working set contains the answer
unambiguously enough that 4-bit values do not flip it.**

### Where the budget stops working

| context | budget | time axis | rows at parity |
|---|---:|---:|---:|
| 50,146 | 192 | 261× | 20/20 |
| 50,146 | **160** | **313×** | **20/20** |
| 50,146 | 144 | 348× | 19/20 |
| 50,146 | 128 | 392× | 18/20 |
| **75,003** | **128** | **586×** | **20/20** |
| 75,003 | 144 | 521× | 20/20 |
| 75,003 | 160 | 469× | 20/20 |

`R4_p4_b192.json`, `DP_p4_b144.json`, `R4_p4_b128.json`, `P6_b128.json`, `P6_b144.json`, `P6_b160.json`

### The state-demand exponent

| context | smallest budget at parity | compression | rows |
|---|---:|---:|---:|
| 12,714 | **64 slots** | **199×** | 20/20 |
| **75,003** | **64 slots** | **1,172×** | **20/20** |

`LB_p1_b64.json`, `DP_p4_b160.json`, `P6_b64.json`

**64 slots give 20 of 20 rows at parity at both 12,714 and 75,003 tokens**, so across a 5.90× span of
context the smallest working set that passes does not change: **`beta = 0`**. The compression ratio then
grows linearly, as a constant budget requires — **199× at 12.7K, 1,172× at 75.0K, a factor of 5.89
against a context factor of 5.90.**

**The per-row criterion has about one-row resolution**, so `B*` is not a stable statistic on 20 rows: at
75,003 tokens, 80 returns 18/20 between passes at 96 and 64, and 48 returns 19/20 above a pass at 56.
**That is why the claim is a budget that passes at both endpoints rather than a frontier** — the
constant-64 result does not depend on locating one.

### Serving, measured in stock vLLM

Same model, same engine, **only the active context length changed** — the quantity a bounded cache
changes.

| requests in flight | 1,152 slots | 130,816 slots | ratio |
|---:|---:|---:|---:|
| 1 | 316.8 tok/s (3.16 ms TPOT) | 88.9 (11.25 ms) | 3.56× |
| 4 | 1,060.0 (3.77 ms) | 164.4 (24.33 ms) | 6.45× |
| 16 | 3,777.6 | 244.6 | 15.1× |
| 128 | 15,966.1 | 250.4 | 63.8× |
| 512 | 20,082.8 | — (saturated) | 80.2× |
| **1,024** | **20,594.7** | **250.4** | **82.2×** |

`EN_len1152.json`, `EN_len4608.json`, `EN_len32768.json`, `E19_stock_vllm_128k_llama1b.json`,
`CC_n16.json` … `CC_n1024.json`, `BCAP_n8.json`, `BCAP_n16.json`

**Both curves saturate, at different heights.** Full-KV saturates at 250.4 tok/s by 12–16 requests and
then falls; a 1,152-slot active state saturates at 20,594.7 by 512–1,024.

**The decode dimension is bounded by the model weights, not by the cache.** Fitting the two measured
points gives `TPOT(B) = 3.09 ms + 8.09 ms x B / 130,816`, so being within 2% of the weight floor needs
`B < 999` slots. **At 1,152 slots we are 2.3% above a floor no KV-cache method can go below.**

### What quantisation costs in latency

The 4,688× figure is a **state-byte** ratio. Timing `dequantize_cache` in this reference
implementation, per decode step:

| retained slots | int8 | int4 | dense TPOT, same width |
|---:|---:|---:|---:|
| 160–192 | 1.32 ms | **4.03 ms** | 3.10 ms |

**The reference path spends 3.10 + 4.03 = 7.1 ms per step at this footprint, against 11.25 ms for dense
Full-KV at the same quality** — a 1.6× end-to-end decode advantage even with unpacking done the
expensive way. **The 4.03 ms is the cost of a Python/torch elementwise unpack, which a fused
dequantise-attention kernel performs inside the matmul rather than before it**; the fusion is the next
implementation step and its headroom is exactly that term.


### All four advantages, in one experiment

Every other section measures one thing at a time. **This is a single invocation of one script with both
arms, the same 20 rows and the same 75,003-token context**, so quality, memory, latency and throughput
come from the same process on the same requests — `PARETO_p6.json`, 40 records.

| measure | Full-KV | **QCC** | ratio |
|---|---:|---:|---:|
| **quality** | 1.0000 | **1.0000** | **identical** |
| **retained slots** | 75,003 | **64** | **1,172×** |
| **resident state**, weights excluded | 9.50 GiB | **0.35 GiB** | **27.0×** |
| **resident total**, weights included | 14.81 GiB | **5.66 GiB** | **2.62×** |
| **TPOT** | 31.62 ms | **17.50 ms** | **1.81×** |
| **throughput** | 31.86 tok/s | **57.14 tok/s** | **1.79×** |

**Every dimension improves on the same data, and the quality is identical to four decimal places**,
because retained slots keep their values and positions.

**Two readings of the memory figure are both correct.** 27.0× is the state; 2.62×
is the whole footprint, including **5.31 GiB of weights that no KV method touches**. An operator
allocating a card sees 2.62×; an operator asking how many requests fit sees
27.0×, because the weights are shared and the state is per-request.

**The latency is measured on the compiled path rather than substituted.** Earlier figures came from
running stock vLLM with a short contiguous active context, which is not what QCC does; this is gather
over 64 compiled slots against a 75,003-token contiguous cache, in the same process. It is smaller than
the substitution implied, and it is the real number: at this width decode is dominated by weight reads
rather than by attention.

**What this is not:** an engine measurement. It is the harness's decode loop, not a scheduler, so
throughput and latency are per-request in isolation. The engine-level concurrency figure remains a
substitution measurement and this does not replace it.

## Corrections

**One interpretation in the first set is superseded.** Its "fixed state vs prompt length" row holds a
working state constant across 16.4× of context. That is a **sufficiency** result: it shows the state
was *enough* at both ends, not that the *minimum* is constant. **The measured exponent on retrieval is
`beta = 0` over a 5.90× span** — the same 64 slots pass at both endpoints.

**`beta` is not measurable on multi-hop QA** with this harness: 17–20 of 20 rows score identically at
every budget from 768 to 4,096, so "the smallest budget where every row passes" is set by a single row
and is not defined.

**Two figures measure two things.** The 1M row is prefill feasibility and bounded state at that length,
where the Full-KV reference that fits does not solve the retrieval task. And 4,688× is the state ratio,
with decode speed measured separately at 6.45×.

**The concurrency ratio has two values measuring different things:** 113.8× is the *state* ratio
(`L/B`); **82.2×** is the *measured throughput* ratio at saturation. The difference is scheduling and
kernel overhead at very high batch.

## How this compares with published systems

Every competitor number below is read from that work's own abstract, not recalled.

| system | KV compression | speedup | memory | context reached | quality, as stated |
|---|---|---|---|---|---|
| **StreamingLLM** (ICLR'24) | sink + recent window (fixed cache) | up to **22.2×** vs sliding-window *recomputation* | — | **4M tokens** | "stable and efficient language modeling" |
| **H₂O** (NeurIPS'23) | 20% heavy hitters retained (**≈5×**) | up to **29×** vs DeepSpeed/Accelerate; **1.9×** lower latency at equal batch | — | — | task-dependent |
| **SnapKV** (NeurIPS'24) | — | **3.6×** generation | **8.2×** efficiency at 16K | 380K on one A100-80GB | "comparable performance", 16 datasets |
| **RocketKV** (ICML'25) | up to **400×** | up to **3.7×** end-to-end | up to **32.6%** peak | — | "negligible accuracy loss" |
| **QCC** | **4,688×** | **3.56× / 6.45×** decode; **82.2×** throughput at saturation | **99.1%** resident reduction at 128K | 131K measured; 1M capacity | **20/20 rows at parity; 18–20/20 predictions byte-identical** |

### Where the numbers are directly comparable

| axis | best published | QCC, measured | comparable? |
|---|---|---|---|
| KV compression | RocketKV 400× | **4,688×** | yes, both are KV-only ratios — but see the peak-memory note |
| decode speedup | RocketKV 3.7×, SnapKV 3.6× | **6.45×** at batch 4 | yes, same quantity; different hardware, so indicative |
| throughput at high concurrency | H₂O 29× vs *other systems* | **82.2× vs Full-KV** | **no** — different baselines, not comparable |
| quality at a stated ratio | none of the four states one | **20/20 rows at parity** | **QCC is the only row with a retention figure** |
| context reached | StreamingLLM 4M | 131K measured | **no** — StreamingLLM is ahead here |

### What the comparable axes show

**Compression:** 4,688× against the best published 400×.

**Decode speedup:** 6.45× at batch 4 and 3.56× at batch 1, against 3.7× and 3.6× as reported.

**Quality at a stated ratio:** the only row in the table with a retention figure — 20 of 20 rows at
parity, 18–20 of 20 predictions byte-identical, because retained slots keep their values and positions.

**Throughput at high concurrency:** 82.2× against Full-KV in the same engine. H₂O's 29× is measured
against DeepSpeed Zero-Inference and HuggingFace Accelerate rather than against Full-KV, so the two
ratios have different denominators.

**RocketKV's 400× and its 32.6% peak-memory reduction do not describe the same operating point:** at
128K on a 1B model a 400× KV reduction implies ~62% peak-memory reduction. The ratio and the
operator-facing memory number in that work differ by a factor of two, which is the more informative
observation of the two.

## Interpretation

Retention is bounded-cache quality divided by matched Full-KV quality on records
for which the Full-KV arm provides a valid reference.

"Rows at parity" is stricter than a mean: it counts records whose score equals the
Full-KV score exactly.

Serving ratios are configuration-specific. The throughput comparison uses the
largest measured operating point for each method under the stated request size
and service constraint.

The million-token artifact establishes prefill feasibility and bounded decode
state on the available 24 GiB GPU. It does not provide a valid 1M retrieval
quality comparison because the Full-KV checkpoint that fits at that length does
not reliably solve the retrieval task.

## Code availability

The public repository is a measurement showcase. Executable reproduction code
and implementation-specific notes are temporarily withheld pending publication.
