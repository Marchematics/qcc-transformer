# Claim ledger: what is established, by which run, and what is still open

Every row names the *shipped* code path that produced the number and the exact
command that regenerates it.  A claim without a matched measurement is listed as
open rather than estimated.  Latency numbers state their measurement window
because the card was shared while they were taken.

## A. Established

| # | claim | evidence | command |
|---|---|---|---|
| A1 | Exact Full-KV prefill, then a decode cache compiled to a fixed slot count per (layer, kv-head); decode is exact softmax over the survivors, no parameter added, checkpoint untouched | `qcc_transformer/retention.py`, `artifacts/bounded-decode-frontier-20260920.md` §3.1-3.7 | `pytest -q tests/test_retention.py` |
| A2 | RULER official metric (80 records, 4 tasks) at budget 4096: **shipped-API aggregate retention 1.0071, worst task 1.000**, per-task 1.000/1.000/1.000/1.0283; harness run agrees (1.0083) | `artifacts/bounded-decode-frontier-multimodel-llama1b.json`, `artifacts/bounded-decode-frontier-ruler-v6.json` | `python benchmarks/analyze_campaign.py multimodel artifacts/bounded-decode-frontier-multimodel-llama1b.json` |
| A3 | The shipped API reproduces the harness on **all 80 records**: partial and strict recall agree 80/80, 75/80 predictions byte-identical (the other five differ only after the answer, same score), per-task means identical; on the record where they previously differed the scores are now bitwise equal and the selected slot sets identical | `artifacts/bounded-decode-frontier-parity-80.json`, `experiments/retention_frontier/diff_selection.py` | `python benchmarks/compare_package_to_harness.py --package artifacts/bounded-decode-frontier-multimodel-llama1b.json` |
| A4 | Padded/ragged batches compile per request and merge exactly: per-row slots, values and prefill logits bit-identical to compiling each row alone (including 7,854 duplicated filler slots); a row sliced from the merged cache decodes to the solo tokens | `artifacts/bounded-decode-frontier-batch-validation.json`, §3.19 | `python benchmarks/validate_retention_batch.py` |
| A5 | The law runs on non-Llama checkpoints through one code path: fused `qkv_proj` (Phi-3), per-layer rotary, per-head `q_norm` (Qwen3) | `qcc_transformer/retention.py` helper layer; `benchmarks/smoke_retention_model.py` | `python benchmarks/smoke_retention_model.py <model>` |
| A6 | Chunked prefill must be rope-consistent for LongRoPE checkpoints: without pinning the rotary length, Phi-3.5 scored 0.0 on 16K RULER records because cached keys and observation queries disagreed about position; with `fixed_rope_length` it scores 1.0 | same smoke run; `benchmarks/benchmark_retention_multimodel.py` | `python benchmarks/benchmark_retention_multimodel.py --model <phi> --trust-remote-code --attn-impl eager --prefill-chunk 1024` |
| A7 | Bounded decode state is fixed by configuration, not by prompt length: `budget + lex_cap` slots for every (layer, kv-head, request) | `artifacts/bounded-decode-frontier-state-growth.json` | `python benchmarks/benchmark_state_growth.py` |
| A8 | Cross-model retention through the shipped API (Llama-3.2-1B, Phi-3.5-mini, Qwen2.5-3B, Llama-3.1-8B-4bit) | `artifacts/bounded-decode-frontier-multimodel-*.json` | `python benchmarks/analyze_campaign.py multimodel artifacts/bounded-decode-frontier-multimodel-*.json` |
| A9 | LongBench harness with the official per-task metrics, validated against the official scorer | `benchmarks/longbench_metrics.py`, `tests/test_longbench_metrics.py` | `pytest -q tests/test_longbench_metrics.py` |

## B. Open, with the specific experiment that closes it

| # | open question | why it matters | the run that closes it |
|---|---|---|---|
| B1 | (still open for the *generality* of the cue; the size of the effect is measured) How much does the law depend on the lexical anchors? **Measured with the fixed scorer: attention ranking alone (4,608 slots, no anchors) retains 0.817 aggregate** - single_1 1.000, vt 1.039, multikey_2 0.895, multikey_3 0.333 (a task where Full-KV itself scores 0.45) - while the shipped anchored configuration is 1.007 / 1.000. The pre-fix number (0.161) was an artefact of the einsum-shape defect, which hurt attention-ranked selection far more than anchor-dominated selection | framing: the law is attention ranking *plus* a rare-string retrieval cue, and the cue is what closes multi-key disambiguation; a reviewer will ask how it generalises | `--anchor-mode rare` (task-agnostic rarity instead of pattern classes), the 7-policy baseline sweep at matched budget, and LongBench (no synthetic needles) |
| B3 | **Closed for Llama-1B**: LongBench, 9 tasks x 20 records, 122 matched records where Full-KV scores above zero: aggregate retention **1.0049** (macro mean 0.2514 full vs 0.2527 bounded), worst task 0.897 (gov_report ROUGE-L), worst record 0.420; narrativeqa +10.7%, triviaqa/samsum/multi_news/qasper/hotpotqa/2wikimqa/passage_retrieval at parity. Real documents, official per-task metrics, no synthetic needles | the non-RULER half of the generality claim | `artifacts/longbench-retention-llama32-1b.json` |
| B8 | Whether the cross-model runs (Phi-3.5, Qwen2.5-3B, Llama-3.1-8B) hold at the same configuration | the current cross-model evidence is one Phi record plus Llama-1B | the `cross` stage of `run_campaign4.sh`, gated on free GPU memory because the card is shared |
| B4 | Pareto comparison against eviction baselines at a matched budget (sliding window, H2O-style window mean, SnapKV-style pool, last-query) and against KV quantization at matched decode-state bytes | "strong baselines" | `artifacts/bounded-decode-frontier-baselines-b4096.json`, the KV-quantization runner |
| B5 | 1M retrieval and 1M TPOT | target metric | blocked on this card: no 1M-native checkpoint is available, and a 1M bf16 Full-KV cache for Llama-3.2-1B is 32 GiB against 24 GiB of HBM.  A7 measures L-independence up to the largest length that fits |
| B6 | 128K TPOT >= 5x Full-KV | target metric | not reachable on this card: matched configuration measures 1.73x, the strongest defensible system number is 4.21x (bounded + CUDA graph vs Full-KV dynamic), and the baseline cannot be given the same optimisation at 128K because capture OOMs; the bandwidth bound for a 1B model is 2.70x before weights are even counted |
| B7 | Latency percentiles on exclusive hardware | the shared card makes TPOT vary 6.9-18.4 ms for one configuration | rerun `benchmark_fullkv_tpot_graph.py` / `benchmark_bounded_decode_tpot_floor.py` on an idle card with repeats |

## C. What is deliberately not claimed

* That the retention law is a general-purpose attention approximation: it is a
  cache policy, and its quality is task-family dependent (§3.12).
* That language-modelling perplexity is preserved: bounded retention costs NLL
  relative to Full-KV (§3.10), while the retrieval-style tasks above are at
  parity.
* That the RULER retention figure transfers to arbitrary long-context tasks
  without the LongBench run (B3).
* Any 1M number: the state-growth argument is structural plus measured up to the
  largest length the card can prefill, and says so.
