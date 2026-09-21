# QCC-Transformer

[![test](https://github.com/Marchematics/qcc-transformer/actions/workflows/test.yml/badge.svg)](https://github.com/Marchematics/qcc-transformer/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Bounded exact-KV decode: long-context inference without history proportional to context length.**

QCC is a *cache policy*, not a new attention architecture and not a trained
module. It runs an **exact Full-KV prefill** through a stock pretrained model,
then compiles the decode cache down to a fixed number of slots per
`(layer, kv-head)`. Decode is exact softmax attention over the survivors, so
every retained key keeps its original rotary phase and its original value.

* **No training, no new parameters, no checkpoint surgery** — measured as a
  bit-identical parameter set before and after compilation.
* **Decode state is set by configuration, not by context length**: 4,608 slots
  is 144 MiB at 8K *and* at 256K, where the Full-KV cache is 8 GiB.
* **Quality parity with Full-KV** on retrieval-style tasks, through the packaged
  API, on four models across three families.

```python
from qcc_transformer import RetentionConfig, compile_bounded_cache

config = RetentionConfig(budget=4096, lex_cap=512)
cache, logits = compile_bounded_cache(model, input_ids, config, tokenizer=tokenizer)
# decode against `cache` like any other Hugging Face cache; logits are the
# prefill's last-token logits, so generation can continue immediately.
```

## Results

All numbers below are produced by the shipped `compile_bounded_cache` path and
stored per record under [`artifacts/`](artifacts); [`docs/CLAIMS.md`](docs/CLAIMS.md)
maps each one to its command. `retention` is bounded quality divided by matched
Full-KV quality on the records the Full-KV arm answers.

| measurement | configuration | result |
|---|---|---|
| RULER, 80 records, 4 tasks | budget 4,096 + 512 anchors, Llama-3.2-1B | **1.007 aggregate retention, 1.000 worst task** |
| package vs benchmark harness | same 80 records | 80/80 metric agreement, 75/80 byte-identical predictions, bitwise-equal scores |
| LongBench, 9 tasks, 122 matched records | same budget | **1.0049 aggregate, 0.897 worst task** (narrativeqa +10.7%) |
| cross-family, one configuration, no tuning | Llama-3.2-1B / Llama-3.1-8B-4bit / Qwen2.5-3B / Phi-3.5-mini | 1.007 / 1.014 / 0.975 / 0.972 aggregate |
| decode state vs prompt length | 8K → 256K | **1.00x growth** (4,608 slots, 144 MiB); Full-KV 57x larger at 256K |
| quality vs state | 48 / 80 / 144 MiB | 0.987 / 0.975 / 1.007 aggregate |
| matched-budget baselines | sliding window / sinks+recent / last-query / window-mean / **QCC** | 0.167 / 0.332 / 0.842 / 0.880 / **1.008** |
| KV quantization at matched bytes | Full-KV int8, bounded+int8 | int8 lossless at 2x; **bounded+int8 = Full-KV quality at 72.3 MiB** |
| decode throughput / SLA concurrency | 32K, batch sweep | **15.6x** throughput (speed config), **8-16x** concurrency at a 50 ms SLA |
| single-stream TPOT, both arms CUDA-graphed | 32K, parity-gated repeats | **5.0x** (p95 55.2 ms vs 11.03 ms) |
| trainable parameters added | — | **0** |

Two honest limits, stated here rather than buried: the **1M rows cannot be
measured on a 24 GiB card** (a 1M bf16 Full-KV cache is 32 GiB, and the law
requires an exact Full-KV prefill, so a quantized prefill would measure a
different method), and the **128K TPOT ratio has no matched baseline on this
hardware** (the same-path baseline OOMs at 131K tokens; the bounded arm's own
floor, 11.0 ms/token with p50 = p95, is reproducible). See
[`docs/REPORT.md`](docs/REPORT.md) §4 for the full list of what is and is not
established.

## How it works

1. **Exact chunked prefill.** The prompt is fed in fixed chunks through the
   model's own causal attention, so activation memory stays `O(chunk)`; only an
   `O(observation_window * d)` hidden-state tail per layer is kept.
2. **Observation-window scoring.** The last `observation_window` prompt tokens
   (for a question-answering prompt, the question itself) attend to every key.
   Those attention scores rank keys per `(layer, kv-head)`; a sliding-window
   dilation and a block max-pool make the ranking robust to a hit straddling a
   block boundary.
3. **Retrieval anchors.** Rare strings the question uses — with a
   task-agnostic rarity mode as well as the hyphenated-word/number/UUID patterns
   — are matched back into the context, expanded asymmetrically (the answer
   usually follows the key), and forced into the retained set. This is the step
   that closes multi-key disambiguation; on LongBench, where no synthetic needles
   exist, it is not needed for parity.
4. **Uniform width.** Every `(layer, kv-head, request)` keeps exactly
   `budget + lex_cap` slots, which is what lets one 2-D mask describe a whole
   batch; attention sinks and a recent window are always forced in.
5. **Exact merge for ragged batches.** Each request is compiled from its own
   real tokens; short rows are filled by duplicating their own last retained slot
   — two identical `(key, value)` pairs split the softmax weight, so the padded
   cache is the same computation. `cache.qcc_attention_mask` and
   `cache.qcc_prompt_lengths` are returned for decode.

Compilation costs one exact prefill per request (linear in the batch, 3.5 s at
8K and 68 s at 128K on the A10G used here); every decoded token afterwards
attends to at most 4,608 keys instead of the whole context.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev,hf]'        # extras: hf-quant (4-bit weights), vllm (serving plugin)
pytest -q                         # CPU only: no GPU, no downloads
```

## Repository layout

```
qcc_transformer/          the package
  retention.py            the retention law: scoring, anchors, selection, pruning, ragged batches
  hf_loading.py           model loading, 4-bit path, remote-code compatibility shims
  retrofit.py, model.py, hybrid_archive.py, associative.py, causal_coreset.py
                          earlier research modules kept for the experiments that use them
  vllm*.py                vLLM backend/plugin integration
benchmarks/               one script per measurement; every one writes a JSON
  benchmark_retention_multimodel.py     bounded vs matched Full-KV, any HF model
  benchmark_retention_longbench.py      LongBench with the official metrics
  benchmark_bounded_decode_ruler.py     selection-policy baselines at one budget
  benchmark_state_growth.py             retained slots and bytes vs prompt length
  kv_quant.py, benchmark_kv_quant_ruler.py   KIVI-axis KV quantization and its Pareto
  benchmark_bounded_decode_tpot_floor.py     TPOT with parity-gated execution paths
  analyze_campaign.py, analyze_latency_percentiles.py, compare_package_to_harness.py
artifacts/                one JSON per run: per-record predictions, scores, slots, timings, memory
docs/                     report, claim ledger, reproduction guide, novelty boundary
examples/                 runnable CPU quickstart
tests/                    CPU test suite (policy invariants, quantization, LongBench metrics)
```

## Documentation

| page | what it covers |
|---|---|
| [`docs/REPORT.md`](docs/REPORT.md) | the full measurement record, including sections on ragged batches, package/harness parity, LongBench and baselines, state growth, quantization, cross-family results, and what is not established |
| [`docs/CLAIMS.md`](docs/CLAIMS.md) | claim → code path → artifact → command, with the open questions and the experiment that closes each |
| [`docs/MECHANISM.md`](docs/MECHANISM.md) | why a fixed slot count can be enough, the NLL-vs-budget evidence, and three falsifiable predictions |
| [`docs/REPRODUCING.md`](docs/REPRODUCING.md) | setup and the exact command behind every artifact |
| [`docs/NOVELTY.md`](docs/NOVELTY.md) | the boundary against prior bounded-memory attention work |
| [`docs/RELEASE.md`](docs/RELEASE.md) | the snapshot bundle and how to verify it |
| [`docs/HANDOFF.zh.md`](docs/HANDOFF.zh.md) | working log (Chinese) for whoever picks this up next |

## Status

The retention law is implemented, tested and measured; the evidence is frozen in
`artifacts/` and reproducible on a single 24 GiB GPU. The project is not a
drop-in serving system: the vLLM integration is experimental, the measurements
come from a shared card (so latency numbers state their window), and the 1M-scale
questions are open for lack of hardware rather than for lack of a method.

## Citation

```bibtex
@misc{qcc_transformer,
  title  = {QCC-Transformer: bounded exact-KV decoding for long-context inference},
  author = {Marchematics},
  year   = {2026},
  url    = {https://github.com/Marchematics/qcc-transformer}
}
```

## License

MIT — see [LICENSE](LICENSE).
