# Evidence index

This page maps the headline measurements to stored artifacts. Raw measurement
artifacts are kept unchanged. Implementation and reproduction commands are
temporarily withheld pending publication.

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

## Interpretation

Retention is bounded-cache quality divided by matched Full-KV quality on records
for which the Full-KV arm provides a valid reference.

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
