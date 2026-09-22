# Evidence artifacts

One JSON per measurement run. They are committed because every number in
[`../README.md`](../README.md) and [`../docs/REPORT.md`](../docs/REPORT.md) is
recomputable from them without a GPU: each file carries its own configuration
(model path, budget, selection settings) and one row per record with the
prediction, the score, the retained slot count, timings and peak memory.

Naming: `bounded-decode-frontier-<measurement>[-<variant>].json`. Files named
`exact-prefill-*`, `cross-read-*` and `teacher-*` come from the earlier archive
experiments in `qcc_transformer/` and are kept for the audit trail.

| file | backs |
|---|---|
| `bounded-decode-frontier-multimodel-llama1b.json` | RULER retention through the shipped API (README results table) |
| `bounded-decode-frontier-parity-80.json` | package vs benchmark harness on all 80 records |
| `bounded-decode-frontier-multimodel-{llama3.1-8b,qwen2.5-3b,phi35}.json` | cross-family results |
| `longbench-retention-llama32-1b.json` | LongBench, nine tasks, official metrics |
| `longbench-published-families.json` | the published families on two summarisation tasks (report 3.31) |
| `bounded-decode-frontier-baselines-b4096.json`, `-lexonly-b4096.json` | matched-budget selection baselines |
| `baselines-quest-pyramid.json`, `baselines-accumulated.json`, `baselines-pyramid-mild.json` | the published eviction families at the same retained width (Quest-shaped blocks, PyramidKV-shaped layer budgets, H2O accumulated attention, TOVA top-1 counts) |
| `bounded-decode-frontier-multimodel-llama1b-{nolex,nolex-mean,rare,rare-hops0,pattern-hops0,mean-anchors,b1024,b2048}.json` | anchor/scoring attribution and the quality/state curve |
| `bounded-decode-frontier-state-growth.json`, `-state-growth-long.json` | retained slots and bytes vs prompt length |
| `bounded-decode-frontier-kv-quant.json` | quality against decode-state bytes with KV quantization |
| `bounded-decode-frontier-tpot-p95.json`, `-tpot-floor-32k-r1-contended.json` | parity-gated TPOT percentiles |
| `serving-vllm-32k.json`, `serving-vllm-32k-concurrency.json` | vLLM's paged Full-KV cache at 32K: decode-only throughput, TPOT and the KV-capacity concurrency ceiling |
| `bounded-decode-frontier-batch-validation.json` | ragged-batch exactness |
| `bounded-decode-frontier-ruler-v6.json` | the stored benchmark-harness run the package is compared against |
| `prediction-density.json` | NLL against long-range structure destroyed in three stages (MECHANISM, prediction 3) |
| `prediction-needles*.json`, `prediction-distractors*.json`, `prediction-required-set-*.json` | the required-set probe: `k` items asked for at once, and one key against `N` competitors |
| `prediction-lexcap*.json`, `prediction-knee-*.json` | the anchor-budget sweep and its knee (`lex_cap` 512-4096 at `k=16`) |
| `prediction-obs*.json` | the query-window sweep (`obs` 64-512 at fixed budget and cap) |
| `prediction-recency*.json` | the oracle-placement control: the required set immediately before the question |
| `prediction-sites-*.json`, `prediction-mass-*.json` | item sites reached by the anchors, the retained share of the statements, and the query's attention share on them |

Regenerate any of them with the command in [`../docs/REPRODUCING.md`](../docs/REPRODUCING.md);
[`../docs/CLAIMS.md`](../docs/CLAIMS.md) states which claim each one supports.
