# Evidence artifacts

One JSON per measurement run. They are committed because every number in
[`../README.md`](../README.md) and [`../docs/REPORT.md`](../docs/REPORT.md) is
recomputable from them without a GPU: each file carries its own configuration
(model path, budget, selection settings) and one row per record with the
prediction, the score, the retained slot count, timings and peak memory.

Naming: `bounded-decode-frontier-<measurement>[-<variant>].json`. Files named
`exact-prefill-*`, `cross-read-*` and `teacher-*` belong to earlier exploratory
phases and are kept for the audit trail in the report's first sections.

| file | backs |
|---|---|
| `bounded-decode-frontier-multimodel-llama1b.json` | RULER retention through the packaged API (README results table) |
| `bounded-decode-frontier-parity-80.json` | package vs benchmark harness on all 80 records |
| `bounded-decode-frontier-multimodel-{llama3.1-8b,qwen2.5-3b,phi35}.json` | cross-family results |
| `longbench-retention-llama32-1b.json` | LongBench, nine tasks, official metrics |
| `bounded-decode-frontier-baselines-b4096.json`, `-lexonly-b4096.json` | matched-budget selection baselines |
| `bounded-decode-frontier-multimodel-llama1b-{nolex,nolex-mean,rare,rare-hops0,pattern-hops0,mean-anchors,b1024,b2048}.json` | anchor/scoring attribution and the quality/state curve |
| `bounded-decode-frontier-state-growth.json`, `-state-growth-long.json` | retained slots and bytes vs prompt length |
| `bounded-decode-frontier-kv-quant.json` | quality against decode-state bytes with KV quantization |
| `bounded-decode-frontier-tpot-p95.json`, `-tpot-floor-32k-r1-contended.json` | parity-gated TPOT percentiles |
| `bounded-decode-frontier-batch-validation.json` | ragged-batch exactness |
| `bounded-decode-frontier-ruler-v6.json` | the stored benchmark-harness run the package is compared against |

Regenerate any of them with the command in [`../docs/REPRODUCING.md`](../docs/REPRODUCING.md);
[`../docs/CLAIMS.md`](../docs/CLAIMS.md) states which claim each one supports.
