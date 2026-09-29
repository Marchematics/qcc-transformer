# Evidence index

This page maps the headline measurements to stored artifacts and reproduction
commands. Raw artifacts are kept unchanged.

| Measurement | Result | Evidence | Reproduction |
|---|---|---|---|
| RULER, current Llama-3.2-1B re-run | aggregate retention **1.0102**, worst task **1.0000** | `artifacts/bounded-decode-frontier-multimodel-llama1b-recheck.json` | `python benchmarks/benchmark_retention_multimodel.py --model <checkpoint> --ruler-jsonl <ruler.jsonl> --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt --budget 4096 --lex-cap 512 --hops 6 --obs 64 --nsink 4 --pool 7 --dilate 9 --arms full bounded --out artifacts/bounded-decode-frontier-multimodel-llama1b-recheck.json` |
| Shipped API / harness agreement | 80/80 metric agreement; 75/80 byte-identical predictions | `artifacts/bounded-decode-frontier-parity-80.json` | `python benchmarks/compare_package_to_harness.py --package artifacts/bounded-decode-frontier-multimodel-llama1b.json` |
| LongBench, nine tasks | aggregate retention **0.9939** with rank-blend filler | `artifacts/longbench-blend-9task.json` | `python benchmarks/benchmark_retention_longbench.py --model <checkpoint> --tasks narrativeqa qasper hotpotqa 2wikimqa gov_report multi_news triviaqa samsum passage_retrieval_en --limit 20 --arms full blend_lex --budget 4096 --lex-cap 512 --hops 6 --blend-alpha 0.5 --cache-dir <longbench> --out artifacts/longbench-blend-9task.json` |
| Fixed state vs prompt length | 4,608 slots / 144 MiB from 8K to 256K | `artifacts/bounded-decode-frontier-state-growth.json` | `python benchmarks/benchmark_state_growth.py` |
| 1M state | 54.0 MiB vs 12,288 MiB Full-KV, **227.6x** reduction | `artifacts/million-context.json` | `python benchmarks/benchmark_million_context.py --lengths 131072 1048576 --items 1 --seeds 2 --budget 4096 --lex-cap 512 --prefill-chunk 8192 --value-digits 2 --yarn auto --attn-impl flash_attention_2 --out artifacts/million-context.json` |
| 32K serving, bounded | batch 32, **2,139.17 tok/s**, 6.62 GiB peak, 32/32 recall | `artifacts/serving-seq-32k-recheck.json` | `python benchmarks/benchmark_bounded_decode_serving_sequential.py --model <checkpoint> --length 32768 --batches 1 2 4 8 16 32 --budget 1024 --lex-cap 128 --obs 64 --nsink 4 --pool 7 --dilate 9 --max-new 32 --prefill-chunk 8192 --seed 7000 --out artifacts/serving-seq-32k-recheck.json` |
| 32K serving, Full-KV | batch 4, **144.91 tok/s**, 15.06 GiB peak; batch 8 OOM | `artifacts/serving-32k-recheck.json` | `python benchmarks/benchmark_bounded_decode_serving.py --model <checkpoint> --length 32768 --batches 1 2 4 8 --max-new 32 --out artifacts/serving-32k-recheck.json` |
| 128K launch-free decode | speed configuration reaches **5.37x** at batch 8 | `artifacts/decode-batch-128k-speed.json` | `python benchmarks/probe_decode_batch.py --contexts 131072 --budget 1152 --batches 1 2 4 8 --steps 12 --out artifacts/decode-batch-128k-speed.json` |
| Matched-width selection baselines | QCC **1.008** vs window mean 0.880, final query 0.842, sinks+recent 0.332, sliding 0.167 | `artifacts/bounded-decode-frontier-baselines-b4096.json` | `python benchmarks/analyze_results.py baselines artifacts/bounded-decode-frontier-baselines-b4096.json` |
| Published-family shaped baselines | stored Quest/Pyramid/H2O/TOVA/SnapKV comparisons | `artifacts/baselines-quest-pyramid.json`, `artifacts/baselines-accumulated.json` | `python benchmarks/analyze_results.py baselines --merge artifacts/bounded-decode-frontier-baselines-b4096.json artifacts/baselines-quest-pyramid.json artifacts/baselines-accumulated.json` |
| KV quantization | bounded + int8 matches Full-KV quality at **72.3 MiB** | `artifacts/bounded-decode-frontier-kv-quant.json` | `python benchmarks/benchmark_kv_quant_ruler.py --model <checkpoint> --ruler-jsonl <ruler.jsonl> --arms full full_int8 full_int4 bounded4096 bounded4096_int8 bounded4096_int4 --out artifacts/bounded-decode-frontier-kv-quant.json` |
| Multi-checkpoint comparison | Llama, Qwen and Phi measurements with one retained-width configuration | `artifacts/bounded-decode-frontier-multimodel-*.json` | `python benchmarks/analyze_results.py multimodel artifacts/bounded-decode-frontier-multimodel-*.json` |

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

## Tests

```bash
pytest -q
```

Benchmark-specific commands and dataset setup are documented in
[`REPRODUCING.md`](REPRODUCING.md).
