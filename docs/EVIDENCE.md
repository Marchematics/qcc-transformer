# Evidence index

This page maps the headline measurements to stored artifacts. Raw measurement
artifacts are kept unchanged. Implementation and reproduction commands are
temporarily withheld pending publication.

**Two measurement sets are listed.** The first is the original set. The second is a later
re-measurement that (a) pushes several dimensions further and (b) **corrects one interpretation in the
first set** — see [Corrections](#corrections) before quoting anything from the first table.

## First measurement set

| Measurement | Result | Evidence |
|---|---|---|
| RULER, current Llama-3.2-1B re-run | aggregate retention **1.0102**, worst task **1.0000** | `artifacts/bounded-decode-frontier-multimodel-llama1b-recheck.json` |
| Shipped API / harness agreement | 80/80 metric agreement; 75/80 byte-identical predictions | `artifacts/bounded-decode-frontier-parity-80.json` |
| LongBench, nine tasks | aggregate retention **0.9939** with rank-blend filler | `artifacts/longbench-blend-9task.json` |
| Fixed state vs prompt length | 4,608 slots / 144 MiB from 8K to 256K | `artifacts/bounded-decode-frontier-state-growth.json` |
| 1M state | 54.0 MiB vs 12,288 MiB Full-KV, **227.6x** reduction | `artifacts/million-context.json` |
| 32K serving, bounded | batch 32, **2,139.17 tok/s**, 6.62 GiB peak, 32/32 recall | `artifacts/serving-seq-32k-recheck.json` |
| 32K serving, Full-KV | batch 4, **144.91 tok/s**, 15.06 GiB peak; batch 8 OOM | `artifacts/serving-32k-recheck.json` |
| 128K launch-free decode | speed configuration reaches **5.37x** at batch 8 | `artifacts/decode-batch-128k-speed.json` |
| Matched-width selection baselines | QCC **1.008** vs window mean 0.880, final query 0.842, sinks+recent 0.332, sliding 0.167 | `artifacts/bounded-decode-frontier-baselines-b4096.json` |
| Published-family shaped baselines | stored Quest/Pyramid/H2O/TOVA/SnapKV comparisons | `artifacts/baselines-quest-pyramid.json`, `artifacts/baselines-accumulated.json` |
| KV quantization | bounded + int8 matches Full-KV quality at **72.3 MiB** | `artifacts/bounded-decode-frontier-kv-quant.json` |
| Multi-checkpoint comparison | Llama, Qwen and Phi measurements with one retained-width configuration | `artifacts/bounded-decode-frontier-multimodel-*.json` |

## Second measurement set

Retrieval quality, at parity, with the budget swept down at each length. **Rows at parity** means the
row's score equals the matched Full-KV score exactly, not that the mean is close.

### Compression, with the two axes separated

State bytes factor as `(L / B) x (16 / d)` — which slots are kept, and how many bits each costs.

| retained state | context | mean | rows at parity | time | space | **total** |
|---|---:|---:|---:|---:|---:|---:|
| 192 slots, bf16 | 50,146 | 1.0000 | **20/20** | 261× | 1× | 261× |
| 192 slots, int8 | 50,146 | 1.0000 | **20/20** | 261× | 2× | **522×** |
| **192 slots, int4** | 50,146 | **1.0000** | **20/20** | 261× | 4× | **1,045×** |

`R4_p4_b192.json`, `CQ_p4_b192_int8.json`, `CQ_p4_b192_int4.json`, `BS_p4_full.json`

**Quantisation is applied after selection**, so the quality figure is attributable to bits rather
than to a changed working set. **Both modes are free at this budget.** That is not true at larger
budgets: at 4,096 slots int4 halves quality (0.8250 → 0.4625) while int8 remains free
(`bounded-decode-frontier-kv-quant.json`). **At 192 slots the working set contains the answer
unambiguously enough that 4-bit values do not flip it.**

### The state-demand exponent

| context | smallest budget at parity | compression | evidence |
|---|---:|---:|---|
| 12,714 | **≤ 96 slots** | **132×** | `LB_p1_b96.json` |
| 50,146 | **(128, 192]** — 128 gives 18/20, 192 gives 20/20 | **261×** | `R4_p4_b128.json`, `R4_p4_b192.json`, `LB_p1_b128.json`, `LB_p1_b192.json` |

**A 3.94× increase in context needs about 2.0× more state, so `beta ≈ 0.50`** — and, as a consequence,
**the compression ratio grows as `sqrt(L)`**: 132× at 12.7K, 261× at 50.1K, and `(50,146/12,714)^0.505 = 2.00` is the
observed ratio of the two.

### Serving, measured in stock vLLM

Same model, same engine, **only the active context length changed** — which is the quantity a bounded
cache changes.

| requests in flight | 1,152 slots | 130,816 slots | ratio |
|---:|---:|---:|---:|
| 1 | 316.8 tok/s (3.16 ms TPOT) | 88.9 (11.25 ms) | 3.56× |
| 4 | 1,060.0 (3.77 ms) | 164.4 (24.33 ms) | 6.45× |
| 16 | 3,777.6 | 244.6 | 15.1× |
| 128 | 15,966.1 | 250.4 | 63.8× |
| 256 | 19,240.2 | — (saturated) | 76.8× |
| 512 | 20,082.8 | — | 80.2× |
| **1,024** | **20,594.7** | **250.4** | **82.2×** |

`EN_len1152.json`, `EN_len4608.json`, `EN_len32768.json`, `E19_stock_vllm_128k_llama1b.json`,
`CC_n16.json` … `CC_n1024.json`, `BCAP_n8.json`, `BCAP_n16.json`

**Both curves saturate, at different heights.** Full-KV saturates at 250.4 tok/s by 12–16 requests and
then falls; a 1,152-slot active state saturates at 20,594.7 by 512–1,024. **The saturation ratio is
82.2×.**

**The decode dimension is bounded by the model weights, not by the cache.** Separating the two
measured points gives `TPOT(B) = 3.09 ms + 8.09 ms x B / 130,816`, so being within 2% of the weight
floor needs `B < 999` slots and within 5% needs `B < 2,498`. **At 1,152 slots we are 2.3% above a
floor no KV-cache method can go below.**

## Corrections

**One interpretation in the first set is superseded.** Its "fixed state vs prompt length" row shows a
working state held constant across 16.4× of context. That is a **sufficiency** result: it shows the
state was *enough* at both ends, not that the *minimum* is constant. **The measured exponent on
retrieval is `beta ≈ 0.50`, not 0** — the working set does grow with context, at about half the rate.
`beta` is also **not measurable on multi-hop QA** with this harness: there, 17–20 of 20 rows score
identically at every budget from 768 to 4,096, so the "smallest budget where every row passes" is set
by a single row and is not defined.

**Two figures should not be read as quality statements.** The 1M row is a capacity measurement — the
Full-KV reference at that length does not reliably solve the retrieval task, as the original note
says. And the 227.6× and the 1,045× are **state-byte ratios**; the decode speed at 1,045× is *not*
measured, because the storage is dequantised before the decode loop. A fused dequantise-attention
kernel would be required.

**The concurrency ratio has two values and they measure different things:** 113.8× is the *state*
ratio (`L/B`), and **82.2×** is the *measured throughput* ratio at saturation. The difference is
scheduling and kernel overhead at very high batch.

## How this compares with published systems

Numbers below are taken from each work's own abstract, not recalled.

| system | KV compression | speedup | memory | quality, as stated |
|---|---|---|---|---|
| H₂O (NeurIPS'23) | 20% heavy hitters retained (≈5×) | up to 29× vs DeepSpeed/Accelearate, 1.9× lower latency at equal batch | — | task-dependent |
| SnapKV (NeurIPS'24) | — | 3.6× generation | 8.2× efficiency at 16K | "comparable performance", 16 datasets |
| RocketKV (ICML'25) | up to 400× | up to 3.7× end-to-end | up to 32.6% peak, decode | "negligible accuracy loss" |
| **QCC** | **1,045×** | **3.56× / 6.45×** decode measured in vLLM; **82.2×** throughput at saturation | **99.1%** resident reduction at 128K | **20/20 rows at parity; 18–20/20 predictions byte-identical** |

**On the axes that can be compared directly:** 6.45× decode against RocketKV's 3.7× and SnapKV's 3.6×,
and the only retention figure in the table. RocketKV's 400× is smaller than 1,045×; its **32.6% peak
memory reduction** is the number that decides how many requests fit, and a 400× KV reduction at 128K
on a 1B model would imply ~62% peak reduction, so **the two are not describing the same operating
point**.

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
