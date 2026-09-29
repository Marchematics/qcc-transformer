# Technical report

This document summarizes the measured behavior of QCC-Transformer in the current
repository release. Raw per-record outputs are stored under
[`artifacts/`](../artifacts), and reproduction commands are indexed in
[`EVIDENCE.md`](EVIDENCE.md).

## 1. System model

QCC is a cache policy for autoregressive Transformer inference. A request first
executes an exact Full-KV prefill. After the prompt is complete, QCC scores the
prompt state from a query-side observation window and compiles the decode cache
to a fixed number of retained slots per `(layer, kv-head)`. Retained keys keep
their original rotary positions and retained values are not reconstructed.

The implementation adds no model parameters and does not require checkpoint
training.

The main quality configuration studied in this repository uses 4,096
attention-selected slots plus 512 lexical-anchor slots, together with attention
sinks and a recent-context allocation. A smaller 1,152-slot configuration is
used in the serving speed measurements.

## 2. Quality

### RULER

On the stored 80-record RULER subset with Llama-3.2-1B, the 4,096 + 512
configuration reproduces Full-KV behavior on records for which the reference
model supplies a valid answer. The current re-run reports aggregate retention
1.0102 and worst-task retention 1.000.

Lexical anchors are most useful on multi-key retrieval. At matched retained
width, pure final-query attention, window-mean selection and recent-window
policies lose more retrieval information.

### LongBench

On nine LongBench tasks, the rank-blend filler configuration reports aggregate
retention 0.9939 over 123 matched records. The residual errors are concentrated
in a small number of summarization and NarrativeQA examples.

A fixed budget is not uniformly sufficient for every task. Increasing the
retained width improves the difficult document-generation cases; in the stored
sweep, a 16,896-slot configuration reaches a 0.964 worst-task ratio.

These measurements are relative to each run's matched Full-KV arm. Absolute
scores can move with model and runtime revisions, so comparisons should be made
within the same artifact.

## 3. Decode state

For Llama-3.2-1B, 4,608 retained slots occupy 144 MiB of bf16 KV state. The
retained-state size is unchanged as the prompt grows from 8K to 256K tokens,
while Full-KV state grows with prompt length.

A separate million-token experiment uses Qwen2.5-0.5B because its Full-KV
geometry fits on a 24 GiB card:

| Context | QCC decode state | Full-KV state | Reduction |
|---:|---:|---:|---:|
| 131,072 | 54.0 MiB | 1,500 MiB | 27.8x |
| 1,048,576 | 54.0 MiB | 12,288 MiB | 227.6x |

The exact 1,048,576-token prefill completed in 850.65 s with a measured peak of
14.06 GiB in that configuration.

This experiment establishes bounded decode-state scaling. It does not establish
1M retrieval parity: the Full-KV checkpoint that fits the available hardware
does not provide a reliable retrieval reference at 1M under the tested position
scaling variants.

## 4. Serving measurements

### 32K sequential-prefill serving

The continuous-batching experiment prefills requests one at a time, keeps the
compiled caches resident, and decodes them together.

On an NVIDIA A10G 24 GiB with Llama-3.2-1B:

| Configuration | Largest measured operating point | Decode throughput | Peak memory |
|---|---:|---:|---:|
| Full-KV | batch 4 | 144.91 tok/s | 15.06 GiB |
| QCC | batch 32 | 2,139.17 tok/s | 6.62 GiB |

At a 50 ms TPOT service threshold, this corresponds to 32 resident QCC requests
versus 4 Full-KV requests, an 8x concurrency ratio. Throughput at the reported
operating points differs by 14.76x.

These figures compare the largest measured operating point that satisfies the
same request size and service constraint; they are not single-stream latency
ratios.

### 128K decode

With Qwen2.5-0.5B at 128K and the 1,152-slot speed configuration, the
launch-free batch-8 decode measurement reports a 5.37x TPOT ratio relative to
the matched Full-KV path. The larger 4,632-slot quality configuration reports a
smaller ratio.

## 5. Baseline comparisons

At a matched retained width on the stored RULER records:

| Policy | Aggregate retention |
|---|---:|
| sliding window | 0.167 |
| sinks + recent | 0.332 |
| final-query selection | 0.842 |
| window-mean selection | 0.880 |
| QCC | 1.008 |

Additional stored comparisons include H2O-, TOVA-, Quest-, PyramidKV- and
SnapKV-shaped policies. Those experiments use the same model, record set and
retained-width accounting; see `EVIDENCE.md` for artifacts and commands.

KV quantization is complementary to selection. In the stored matched-byte
experiment, bounded int8 KV reaches Full-KV quality at 72.3 MiB.

## 6. Cross-model behavior

The same 4,096 + 512 configuration was measured on checkpoints from the Llama,
Qwen and Phi families. Aggregate retention varies by checkpoint.

The repository also includes a Qwen2.5-1.5B counterexample where the same slot
budget is insufficient for tasks naming many items at once. Retained width
should therefore be sized to workload structure instead of treated as a
universal constant across models.

## 7. Implementation

The current package provides exact chunked prefill, query-window attention
scoring, lexical retrieval anchors, fixed-width retained caches, ragged-batch
handling, KV quantization experiments, Hugging Face integration, experimental
vLLM integration, CPU regression tests and benchmark analyzers.

The vLLM integration is still experimental. Production latency depends on model
size, GPU, batching policy, retained width and framework version.

## 8. Scope and limitations

The current evidence supports bounded decode state after an exact prefill,
strong retrieval retention in the tested configurations, and substantial
serving gains when Full-KV memory limits continuous batching.

The current evidence does not support universal Full-KV equivalence for every
model and task, 1M retrieval parity, a production-ready drop-in serving stack,
or hardware-independent latency ratios.

For exact commands and artifact locations, see
[`EVIDENCE.md`](EVIDENCE.md).
