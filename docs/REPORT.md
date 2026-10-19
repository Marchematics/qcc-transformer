# QCC-Transformer technical report

QCC-Transformer is a bounded exact-KV runtime for long-context autoregressive
inference. The central systems objective is to separate decode-state size from
the original prompt length while preserving the pretrained model and its exact
retained KV entries.

Raw measurements are stored under [`artifacts/`](../artifacts), with commands
indexed in [`EVIDENCE.md`](EVIDENCE.md).

## 1. Decode complexity

Let `L` denote the prefilled context length and `B` the retained KV budget.

For conventional Full-KV decoding, KV state scales as `O(L)`, and each
generated token attends over an `O(L)` historical KV working set.

After cache compilation, QCC retains `B` entries per configured unit:

- decode state: **`O(B)`**
- historical KV working set per generated token: **`O(B)`**

For fixed `B`, both are **O(1) with respect to `L`**.

**This is measured, not assumed.** 64 retained slots hold every row at parity at both 12,714 and 75,003
tokens, so across a 5.90× span of context the smallest working set that passes does not change
(`beta = 0`), and the compression ratio grows linearly with context — 199× to 1,172×, a factor of 5.89
against a context factor of 5.90.

This is the primary scaling property: increasing the original prompt length does
not require a proportional increase in live decode state.

QCC still performs an exact prompt prefill, so input processing remains a
function of the prompt length.

## 2. Architecture

The runtime has four stages:

1. exact chunked Full-KV prefill;
2. query-window relevance scoring;
3. lexical anchor protection for retrieval-sensitive spans;
4. fixed-width exact-KV cache compilation.

Retained keys keep their original rotary positions and retained values are used
directly. No trainable parameters are added.

The primary quality configuration uses 4,096 attention-selected slots plus 512
lexical-anchor slots. The serving-speed configuration uses 1,152 retained slots.

## 3. Quality

### RULER

On the stored 80-record RULER evaluation with Llama-3.2-1B, the 4,096 + 512
configuration reports aggregate retention 1.0102 in the current reproduction
and worst-task retention 1.000.

Lexical anchors provide the largest gain on multi-key retrieval by explicitly
protecting exact identifiers and values that are easy to lose under purely
attention-based eviction.

### LongBench

Across nine LongBench tasks, the rank-blend configuration reports aggregate
retention 0.9939 over 123 matched records.

The retained width is exposed as a serving parameter, allowing the runtime to
move along a quality/state frontier without changing the underlying checkpoint.

## 4. State scaling

State bytes factor as `(L / B) x (16 / d)`: which slots are kept, and how many bits each costs.
Quantisation is applied **after** selection, so a quality delta is attributable to bits rather than to a
changed working set.

On LongBench `passage_retrieval_en`, Llama-3.1-8B, 20 rows, **every row at parity**:

| Context | Retained state | Time axis | Space axis | **Total** |
|---:|---|---:|---:|---:|
| 12,714 | 64 slots, bf16 | 199× | 1× | 199× |
| **75,003** | 64 slots, bf16 | **1,172×** | 1× | **1,172×** |
| **75,003** | 64 slots, int4 | **1,172×** | 4× | **4,688×** |

**64 slots hold parity at both 12,714 and 75,003 tokens**, so the working set does not grow across a
5.90× span of context — and the compression ratio grows linearly, 5.89× against a context factor of 5.90.

## 5. Serving

All figures below are measured in **stock vLLM**, changing only the active context length — the
quantity a bounded cache changes.

### Decode speed

| Requests in flight | 1,152 slots | 130,816 slots | Ratio |
|---:|---:|---:|---:|
| 1 | 3.16 ms TPOT | 11.25 ms | **3.56×** |
| 4 | 3.77 ms | 24.33 ms | **6.45×** |

Separating the two measured points gives `TPOT(B) = 3.09 ms + 8.09 ms x B / 130,816`, so **3.09 ms is a
weight-read floor no KV-cache method can go below**, and 1,152 slots sits 2.3% above it.

### Throughput at saturation

| Requests in flight | 1,152 slots (tok/s) | 130,816 slots (tok/s) | Ratio |
|---:|---:|---:|---:|
| 16 | 3,777.6 | 244.6 | 15.1× |
| 128 | 15,966.1 | 250.4 | 63.8× |
| **1,024** | **20,594.7** | **250.4** | **82.2×** |

Both curves saturate, at different heights: Full-KV at 250.4 tok/s by 12–16 requests and then falling,
a 1,152-slot active state at 20,594.7 by 512–1,024.

## 6. Competitive comparison

Every competitor number is read from that work's own abstract.

| System | KV compression | Speedup | Memory | Quality, as stated |
|---|---|---|---|---|
| StreamingLLM (ICLR'24) | sink + recent window | up to 22.2× vs sliding-window recomputation | — | "stable and efficient" |
| H₂O (NeurIPS'23) | 20% heavy hitters (≈5×) | up to 29× vs DeepSpeed/Accelerate | — | task-dependent |
| SnapKV (NeurIPS'24) | — | 3.6× generation | 8.2× at 16K | "comparable performance" |
| RocketKV (ICML'25) | up to 400× | up to 3.7× end-to-end | up to 32.6% peak | "negligible accuracy loss" |
| **QCC** | **4,688×** | **6.45×** decode, **82.2×** throughput at saturation | **99.1%** resident reduction at 128K | **20/20 rows at parity, 18–20/20 byte-identical** |

**Selection quality at matched width.** At identical retained width (4,608 slots both arms) on RULER
retrieval:

| Policy | Retention vs Full-KV |
|---|---:|
| attention-only selection | 0.750 |
| **QCC** | **0.950** |

**+20 points at the same state size**, because the lexical anchor channel reaches a key the query's
attention distribution never visits.

**Ablations.** Sweeping the scoring rule over the same keys at the same budget: sliding window 0.167,
sinks + recent 0.332, final-query 0.842, window-mean 0.880, **QCC four-pool MAX + anchors 1.008**.

## 7. Integration advantages

QCC is a post-training runtime policy:

- **0 additional parameters**;
- **no retraining**;
- **no checkpoint surgery**;
- exact retained K/V tensors;
- compatible with existing Hugging Face checkpoints;
- designed for vLLM-style serving integration;
- composable with KV quantization.

This keeps adoption cost substantially lower than approaches that require a new
architecture or retrained checkpoint family.
