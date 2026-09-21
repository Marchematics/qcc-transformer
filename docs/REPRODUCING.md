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

Serving throughput and SLA concurrency are produced by
`benchmark_bounded_decode_serving.py` (a sweep over batch sizes) and summarised
by `analyze_sla_concurrency.py`; the numbers in the README are the ones those
two scripts print, with the measurement window stated next to them.

## Limits

* **1M rows are not reproducible on a 24 GiB card.** A 1M-token bf16 Full-KV
  cache is 32 GiB for the 1B model, and the law requires an *exact* Full-KV
  prefill, so a quantized or streaming prefill would measure a different method.
* **The 128K TPOT ratio has no matched baseline on this hardware class.** The
  same-path harness OOMs on the Full-KV arm at 131K tokens; the bounded arm's own
  floor is reproducible (11.0 ms/token, p50 = p95).
* **Latency depends on concurrent load.** Every JSON records its own run; numbers
  taken while the GPU was shared are marked as such in `REPORT.md` §3.14b.
