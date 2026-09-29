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

For Llama-3.2-1B, 4,608 retained slots occupy 144 MiB of bf16 KV state. The
retained-state footprint is unchanged from 8K through 256K prompt length.

The million-token measurement uses Qwen2.5-0.5B:

| Context | QCC decode state | Full-KV state | Reduction |
|---:|---:|---:|---:|
| 131,072 | 54.0 MiB | 1,500 MiB | 27.8× |
| 1,048,576 | 54.0 MiB | 12,288 MiB | 227.6× |

The QCC state is unchanged between the two endpoints, giving **1.00× state
growth** while the context grows 8×.

## 5. Serving

### 32K continuous batching

On an NVIDIA A10G 24 GiB with Llama-3.2-1B:

| Configuration | Largest measured operating point | Decode throughput | Peak memory |
|---|---:|---:|---:|
| Full-KV | batch 4 | 144.91 tok/s | 15.06 GiB |
| QCC | batch 32 | 2,139.17 tok/s | 6.62 GiB |

At a 50 ms TPOT service threshold, the measured resident concurrency is 32
requests for QCC versus 4 for Full-KV: an **8× concurrency ratio**.

The measured throughput ratio at those operating points is **14.76×**.

### 128K decode

At 128K on Qwen2.5-0.5B, the 1,152-slot speed configuration reports a **5.37×**
launch-free TPOT ratio at batch 8 relative to the matched Full-KV path.

## 6. Competitive comparison

At matched retained width on the stored RULER evaluation:

| Policy | Aggregate retention |
|---|---:|
| sliding window | 0.167 |
| sinks + recent | 0.332 |
| final-query selection | 0.842 |
| window-mean selection | 0.880 |
| **QCC** | **1.008** |

The repository also stores matched comparisons with H2O-, TOVA-, Quest-,
PyramidKV- and SnapKV-shaped policies.

KV quantization is complementary rather than competing: bounded + int8 reaches
Full-KV quality at 72.3 MiB in the stored matched-byte experiment.

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

## 8. Productization path

The current implementation already covers the core bounded-KV mechanism,
chunked prefill, batching, quality evaluation and serving measurement.

The engineering roadmap extends this core to H100/H200/B200, hardened vLLM and
TensorRT-LLM integration, 7B–70B models, workload-aware budget control and
multi-GPU continuous batching.

The intended product boundary is an inference runtime layer that reduces the GPU
cost of long-context serving without requiring customers to replace their model
checkpoints.
