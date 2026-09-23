# Reproducing the results

Every number quoted in the project README and in [`REPORT.md`](REPORT.md) comes
from a script in `benchmarks/` and is stored as JSON under `artifacts/`.
[`CLAIMS.md`](CLAIMS.md) maps each claim to its artifact and to the exact command
below; this page is the operational half of that ledger.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev,hf]'        # add ,hf-quant for 4-bit weights, ,vllm for the serving plugin
pytest -q                         # CPU tests only; no GPU or downloads required
```

A single 24 GiB GPU is enough for every measurement quoted here except the 1M
rows (see [Limits](#limits)). Models and datasets are not bundled:

| what | where it comes from |
|---|---|
| model checkpoints | any Hugging Face causal LM; pass one with `--model` or `$QCC_MODEL`; the checkpoint used is recorded in each JSON's `config.model` |
| RULER split | an external `ruler_subset.jsonl` (80 records, 4 tasks); pass it with `--ruler-jsonl` or `$QCC_RULER_JSONL`; the path used is recorded in each JSON's `config.ruler_jsonl` |
| LongBench | downloaded on demand by `benchmarks/longbench_data.py` from `THUDM/LongBench` |

## The shipped law

```bash
python -m pytest -q tests/test_retention.py          # policy invariants, ragged batches, retrofit
python benchmarks/validate_retention_batch.py        # ragged batch vs per-row compiles (GPU)
python benchmarks/compare_package_to_harness.py \
    --package artifacts/bounded-decode-frontier-multimodel-llama1b.json
```

## Quality

```bash
# RULER, both arms, all 80 records, any HF model
python benchmarks/benchmark_retention_multimodel.py --model <checkpoint> \
    --out artifacts/bounded-decode-frontier-multimodel-<name>.json

# LongBench, nine tasks, official per-task metrics
python benchmarks/benchmark_retention_longbench.py --model <checkpoint> \
    --tasks narrativeqa qasper hotpotqa 2wikimqa gov_report multi_news triviaqa samsum \
            passage_retrieval_en --limit 20 --budget 4096 --lex-cap 512 --hops 6 \
    --out artifacts/longbench-retention-<name>.json

# tables from the JSONs
python benchmarks/analyze_campaign.py multimodel artifacts/bounded-decode-frontier-multimodel-*.json
python benchmarks/analyze_campaign.py baselines  artifacts/bounded-decode-frontier-baselines-b4096.json

# attribute a LongBench task's gap to the records that cause it (per-record loss
# concentration, ROUGE-L precision/recall split, repetition against score change)
python benchmarks/analyze_longbench.py artifacts/longbench-retention-<name>.json \
    --task gov_report --cache-dir "$QCC_LONGBENCH_DIR"
```

## Selection ablations and baselines

```bash
# seven selection policies at one budget over the same records
python benchmarks/benchmark_bounded_decode_ruler.py --ruler-jsonl <split> \
    --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt \
    --policies full recent sink_recent obs_mean obs_max obs_last lex_obs \
    --budgets 4096 --obs 64 --nsink 4 --pool 7 --dilate 9 --lex-cap 512 --hops 6 \
    --key-chunk 1024 --max-new 128 --prefill-chunk 8192 \
    --out artifacts/bounded-decode-frontier-baselines-b4096.json

# anchor/scoring attribution: --lex-cap 0, --scoring mean|max, --anchor-mode rare|both, --hops 0
python benchmarks/benchmark_retention_multimodel.py --model <checkpoint> \
    --budget 4608 --lex-cap 0 --scoring mean --out artifacts/<name>.json
```

## Head-to-head with the published eviction families

The same runner implements the recent eviction families as selection rules, so one
variable moves: which positions survive. `h2o` ranks by attention accumulated over
every prefill query, `tova` by how often a key was a query's top-1, `quest` selects
whole 16-token blocks by the observation window's best similarity, and
`pyramid`/`pyramid_mild` spend a layer-dependent budget (1.5x to 0.5x and 1.25x to
0.75x from the first layer to the last, mean preserved).

```bash
# the cheap families on all 80 records, merged with the stored baseline table
python benchmarks/benchmark_bounded_decode_ruler.py --model <checkpoint> \
    --ruler-jsonl <split> --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt \
    --policies full quest pyramid --budgets 4096 --obs 64 --nsink 4 --pool 7 \
    --dilate 9 --lex-cap 512 --hops 6 --key-chunk 1024 --max-new 128 \
    --prefill-chunk 8192 --out artifacts/baselines-quest-pyramid.json

# the accumulated-attention families need a second pass over the prompt's
# attention, so they run on a balanced subset (--max-per-task spreads the subset
# across each task's length range) with their own Full-KV and reference arms
python benchmarks/benchmark_bounded_decode_ruler.py --model <checkpoint> \
    --ruler-jsonl <split> --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt \
    --policies full obs_last obs_mean lex_obs h2o tova --budgets 4096 --obs 64 \
    --nsink 4 --pool 7 --dilate 9 --lex-cap 512 --hops 6 --key-chunk 1024 \
    --max-new 128 --prefill-chunk 8192 --max-per-task 10 \
    --out artifacts/baselines-accumulated.json

python benchmarks/analyze_campaign.py baselines --merge \
    artifacts/bounded-decode-frontier-baselines-b4096.json \
    artifacts/baselines-quest-pyramid.json
```

## Where the capacity of the policy comes from

The selection sees only the last `observation_window` tokens of the prompt, and
its anchor set is capped by `lex_cap`. Both limits are measured without a model
first (tokenizer only, CPU), then on the model:

```bash
# how many of the k item names the query window can see, and how many sites the
# anchor set reaches, per (k, obs, lex_cap)
python benchmarks/anchor_window_coverage.py --model <checkpoint> \
    --items 8 16 32 64 --obs 64 128 256 512 --lex-cap 512 2048 --length 32768

# the required-set probe itself: k needles asked for at once, Full-KV against
# bounded, with the anchor window and the anchor budget as the two knobs
python benchmarks/benchmark_required_set.py --model <checkpoint> --family needles \
    --items 8 16 32 --lex-cap 512 2048 --obs 128 --budgets 2048 4096 8192 \
    --seeds 3 --lengths 32768 --hops 6 --out artifacts/prediction-needles-<name>.json

# the oracle-placement control: the same statements immediately before the
# question, i.e. inside the recency channel of any budget that can hold them
python benchmarks/benchmark_required_set.py --model <checkpoint> --family needles \
    --items 8 16 --lex-cap 512 --placement recency --budgets 2048 4096 8192 \
    --seeds 3 --lengths 32768 --hops 6 --out artifacts/prediction-recency-<name>.json

# strict accuracy, per-item recall, truncated decode share and reached sites
python benchmarks/analyze_required_set.py artifacts/prediction-*.json
```

`analyze_required_set.py` prints the views side by side on purpose: the
all-or-nothing metric and the per-item recall answer different questions, and a
gap that is a readout limit looks like a retention limit in the first one alone.
Two of the columns are retention measurements rather than budget statements:
`required_coverage` is the share of the item statements' own tokens that survived
selection (counted over every `(layer, head)` index set the arm kept), and
`required_mass` is the share of the final query's attention that lands on those
statements.

## State size, quantization and latency

```bash
python benchmarks/benchmark_state_growth.py --lengths 8192 32768 65536 131072 262144 \
    --out artifacts/bounded-decode-frontier-state-growth.json

python benchmarks/benchmark_kv_quant_ruler.py --model <checkpoint> --ruler-jsonl <split> \
    --tasks niah_single_1 niah_multikey_2 niah_multikey_3 vt --limit 4 \
    --budget 4096 --lex-cap 512 --group-size 128 --max-new 64 \
    --arms full full_int8 full_int4 bounded4096 bounded4096_int8 bounded4096_int4 \
    --out artifacts/bounded-decode-frontier-kv-quant.json

# TPOT: both cache arms on the same execution path, every timing gated by a token-parity check
python benchmarks/benchmark_bounded_decode_tpot_floor.py --length 32768 \
    --budget 4096 --max-new 32 --modes dynamic graph --out /tmp/tpot.json
python benchmarks/analyze_latency_percentiles.py /tmp/tpot-*.json
```

## The industry serving stack (vLLM)

`benchmarks/benchmark_vllm_serving_compare.py` measures the same model, prompt
length and decode under vLLM's paged Full-KV cache, and reports decode-only
throughput and TPOT by prefilling the prompt once and reusing it through prefix
caching. The engine's own start-up line ("maximum concurrency for N tokens per
request") is the KV-capacity statement the report quotes.

```bash
python benchmarks/benchmark_vllm_serving_compare.py --model <checkpoint> \
    --length 32768 --batches 1 2 4 8 --max-new 128 \
    --gpu-memory-utilization 0.6 --max-num-seqs 8 \
    --out artifacts/serving-vllm-32k.json
```

Serving throughput and SLA concurrency are produced by
`benchmark_bounded_decode_serving.py` (a sweep over batch sizes) and summarised
by `analyze_sla_concurrency.py`; the numbers in the README are the ones those
two scripts print, with the measurement window stated next to them.

## Low-rank latent KV at matched bytes

```bash
python benchmarks/benchmark_lowrank_kv_ruler.py --model <checkpoint> \
    --ruler-jsonl <split> --tasks niah_multikey_2 niah_multikey_3 --max-records 40 \
    --budget 4096 --lex-cap 512 --obs 64 --nsink 4 --pool 7 --dilate 9 --hops 6 \
    --key-chunk 1024 --max-new 128 --arms full bounded lowrank1 lowrank2 \
    --out artifacts/lowrank-kv-ruler.json
```

`lowrankN` keeps every token at `N` times the byte-matched rank (`slots x head_dim / L`,
about 9-25 of 64 here) and reconstructs keys and values at compile time, so the
comparison is quality per stored byte rather than a fused kernel.

## The 1M harness

```bash
python benchmarks/benchmark_million_context.py --model <checkpoint with a small KV> \
    --lengths 131072 1048576 --items 1 --seeds 2 --budget 4096 --lex-cap 512 \
    --prefill-chunk 8192 --value-digits 2 --yarn auto \
    --attn-impl flash_attention_2 --out <path for the 1M rows>
```

Four things are required, and each was a measured failure before it was a flag
(report 3.34-3.39):

* **`flash-attn` installed.** The chunked prefill routes each layer through
  `flash_attn_func`; the stock SDPA path materialises an explicit `chunk x end` mask
  and needs ~22 GiB at 128K, against 3.05 GiB measured with the fused kernel.
* **`torch.no_grad` on the prefill.** Without it every chunk's autograd graph stays
  reachable from the persistent cache: 1.31 MiB retained per token instead of 0.015.
* **A preallocated cache.** `DynamicCache` grows by `torch.cat`, so a 1M append needs
  the old and the new cache at once (12.0 + 12.0 GiB) and dies inside `update`. The
  harness builds a `PreallocatedCache` automatically.
* **The retrieval protocol.** `--value-digits 2` and the answer prefix are part of the
  measurement: a 6-digit value is six tokens here and a 0.5B checkpoint drops a digit
  often enough to zero a record whose retrieval worked, and without the prefix the
  model re-lists the question's keys. `--items 1` is what this checkpoint can actually
  retrieve: the 4-needle form scores 0 on the Full-KV arm itself.

## Limits

* **The 1M Full-KV arm fits only because the checkpoint's KV is 12 KiB per token.**
  Qwen2.5-0.5B (24 layers, 2 kv heads, head_dim 64) stores 12.0 GiB at 1M, which is
  what makes an *exact* Full-KV arm possible on a 24 GiB card for the first time here.
  A 1B-class model at 32 KiB/token would need 32 GiB and does not fit.
* **The 128K TPOT ratio has no matched baseline on this hardware class.** The
  same-path harness OOMs on the Full-KV arm at 131K tokens; the bounded arm's own
  floor is reproducible (11.0 ms/token, p50 = p95).
* **Latency depends on concurrent load.** Every JSON records its own run; numbers
  taken while the GPU was shared are marked as such in `REPORT.md` §3.14b.
