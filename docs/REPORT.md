# QCC-Transformer technical report

This report summarizes the current measured system behavior. Raw per-record
outputs remain under [`artifacts/`](../artifacts), with reproduction commands
indexed in [`EVIDENCE.md`](EVIDENCE.md).

## 1. Architecture

QCC is a cache policy for autoregressive Transformer inference. It runs an exact
Full-KV prefill and then compiles the prompt state into a bounded decode cache.
Selection is performed per layer and KV head using query-side attention together
with lexical retrieval anchors.

Retained keys preserve their original rotary positions and retained values are
used directly. The runtime adds no model parameters and does not require
checkpoint training.

The primary quality configuration uses 4,096 attention-selected slots plus 512
lexical-anchor slots. A 1,152-slot configuration is used for the serving-speed
measurements.

## 2. Quality

### RULER

On the stored 80-record RULER evaluation with Llama-3.2-1B, the 4,096 + 512
configuration reports aggregate retention 1.0102 in the current reproduction
and worst-task retention 1.000.

Lexical anchors provide the largest gain on multi-key retrieval, complementing
query-window attention when exact identifiers or values must survive cache
compilation.

### LongBench

Across nine LongBench tasks, the rank-blend configuration reports aggregate
retention 0.9939 over 123 matched records.

The retained width is an operating parameter. Document-generation and
high-dependency workloads benefit from a larger budget, while retrieval tasks
reach parity at substantially smaller state than Full-KV.

## 3. State scaling

For Llama-3.2-1B, 4,608 retained slots occupy 144 MiB of bf16 KV state. The
retained-state size remains constant from 8K through 256K prompt length.

The million-token experiment uses Qwen2.5-0.5B so the corresponding Full-KV
geometry fits on a 24 GiB card:

| Context | QCC decode state | Full-KV state | Reduction |
|---:|---:|---:|---:|
| 131,072 | 54.0 MiB | 1,500 MiB | 27.8× |
| 1,048,576 | 54.0 MiB | 12,288 MiB | 227.6× |

The 1,048,576-token exact prefill completed with a measured 14.06 GiB peak.
These measurements demonstrate that QCC decode state can remain fixed while the
underlying context grows by an order of magnitude.

## 4. Serving

### 32K continuous batching

The serving experiment prefills requests sequentially, keeps compiled caches
resident, and decodes the active requests together.

On an NVIDIA A10G 24 GiB with Llama-3.2-1B:

| Configuration | Largest measured operating point | Decode throughput | Peak memory |
|---|---:|---:|---:|
| Full-KV | batch 4 | 144.91 tok/s | 15.06 GiB |
| QCC | batch 32 | 2,139.17 tok/s | 6.62 GiB |

At a 50 ms TPOT service threshold, the measured resident concurrency is 32
requests for QCC and 4 for Full-KV. Throughput at those operating points differs
by 14.76×.

### 128K decode

At 128K on Qwen2.5-0.5B, the 1,152-slot speed configuration reports a 5.37×
launch-free TPOT ratio at batch 8 relative to the matched Full-KV path.

## 5. Comparison with cache alternatives

At matched retained width on the stored RULER evaluation:

| Policy | Aggregate retention |
|---|---:|
| sliding window | 0.167 |
| sinks + recent | 0.332 |
| final-query selection | 0.842 |
| window-mean selection | 0.880 |
| QCC | 1.008 |

The repository also includes matched comparisons with H2O-, TOVA-, Quest-,
PyramidKV- and SnapKV-shaped policies.

KV quantization is complementary to selection. In the stored matched-byte
experiment, bounded int8 KV reaches Full-KV quality at 72.3 MiB.

## 6. Model coverage

The same 4,096 + 512 configuration has been measured across Llama, Qwen and Phi
checkpoints. The experiments show that the retention mechanism transfers across
architectures while the optimal state budget remains workload- and
checkpoint-dependent.

This is useful operationally: QCC exposes retained width as an explicit serving
control rather than fixing it in the model architecture.

## 7. Runtime

The repository currently includes:

- exact chunked prefill;
- query-window scoring;
- lexical retrieval anchors;
- fixed-width retained caches;
- ragged-batch handling;
- KV quantization experiments;
- Hugging Face integration;
- vLLM integration work;
- reproducible quality and serving benchmarks.

## 8. Operating envelope and next validation

The current evidence establishes the core bounded-state behavior and its serving
benefit on the measured hardware. The next validation phase extends the same
system to larger production models and accelerators.

Priority work is:

1. H100/H200/B200 serving benchmarks;
2. production vLLM and TensorRT-LLM integration;
3. 7B–70B checkpoint coverage;
4. workload-aware budget selection;
5. matched million-token quality measurements on hardware that can host an
   appropriate Full-KV reference.

Detailed measurement conditions remain in [`EVIDENCE.md`](EVIDENCE.md), so the
public project description can stay concise while the underlying results remain
auditable.
