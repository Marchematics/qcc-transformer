# QCC-Transformer

[![test](https://github.com/Marchematics/qcc-transformer/actions/workflows/test.yml/badge.svg)](https://github.com/Marchematics/qcc-transformer/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Bounded exact-KV decoding for long-context language-model inference.**

QCC compiles a completed Full-KV prefill into a fixed-size decode cache. The
retained keys keep their original rotary positions and values, and decoding uses
ordinary softmax attention over the retained state.

The implementation is training-free and does not add model parameters or modify
checkpoint weights.

```python
from qcc_transformer import RetentionConfig, compile_bounded_cache

config = RetentionConfig(budget=4096, lex_cap=512)
cache, logits = compile_bounded_cache(
    model,
    input_ids,
    config,
    tokenizer=tokenizer,
)
```

## Why QCC

Conventional Full-KV decoding keeps key/value state for the entire prompt. Its
decode-state footprint grows with context length, reducing the number of
long-context requests that fit on a GPU.

QCC separates the one-time prefill from the state used during generation:

1. run an exact chunked prefill;
2. score prompt keys from a query-side observation window;
3. preserve retrieval anchors, attention sinks and recent context;
4. retain a fixed number of KV slots per layer and KV head;
5. decode over the retained KV with the original positional phases intact.

The retained-state budget is configured independently of prompt length.

## Benchmark snapshot

Measurements below are stored under [`artifacts/`](artifacts) and linked to
their reproduction commands in [`docs/EVIDENCE.md`](docs/EVIDENCE.md).

| Measurement | Configuration | Result |
|---|---|---:|
| RULER retention | Llama-3.2-1B, 4,096 attention slots + 512 lexical anchors | **1.007 aggregate, 1.000 worst task** |
| LongBench retention | Llama-3.2-1B, nine-task evaluation | **0.9939 aggregate** with rank-blend filler |
| Decode-state growth | 8K -> 256K, 4,608 retained slots | **1.00x**; 144 MiB at both endpoints |
| 1M decode state | Qwen2.5-0.5B, 1,048,576 tokens | **54 MiB vs 12,288 MiB Full-KV (227.6x reduction)** |
| 32K decode throughput | Llama-3.2-1B, A10G 24 GiB | **2,139 tok/s vs 145 tok/s Full-KV (14.76x)** |
| 32K fixed-SLA concurrency | same serving experiment | **32 vs 4 resident requests (8x)** |
| 128K launch-free TPOT | Qwen2.5-0.5B, batch 8, speed configuration | **5.37x** |
| Trainable parameters added | all configurations | **0** |

These results describe the measured configurations, not a universal guarantee
across models or workloads. The full report documents the cases where a fixed
budget is insufficient and the configurations used for each comparison.

## How it works

### Exact chunked prefill

The prompt is processed through the model's causal attention in chunks. This
keeps activation memory bounded by the prefill chunk while preserving the
model's native Full-KV computation.

### Query-aware selection

The final prompt tokens form an observation window. Their attention to the
prefill keys provides a per-layer, per-KV-head relevance signal.

### Lexical anchors

Rare strings, identifiers, numbers and other query-specific spans can be
matched back into the prompt and protected from eviction. This complements the
attention score on retrieval-heavy workloads.

### Fixed-width cache

Each request is compiled to a uniform retained width. Ragged requests are
batched without changing the retained computation, and original RoPE positions
are preserved.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev,hf]'
pytest -q
```

The CPU test suite does not require a GPU or model downloads.

## Repository layout

```text
qcc_transformer/   package and Hugging Face / vLLM integration
benchmarks/        benchmark runners and analysis utilities
artifacts/         per-run predictions, scores, timings and memory records
docs/              technical report, mechanism, evidence and reproduction guide
examples/          small runnable examples
tests/             unit and regression tests
```

Useful entry points:

- [`docs/REPORT.md`](docs/REPORT.md) — complete technical measurement record
- [`docs/MECHANISM.md`](docs/MECHANISM.md) — retention mechanism and capacity analysis
- [`docs/EVIDENCE.md`](docs/EVIDENCE.md) — measurement-to-artifact-to-command index
- [`docs/REPRODUCING.md`](docs/REPRODUCING.md) — environment and reproduction commands
- [`docs/NOVELTY.md`](docs/NOVELTY.md) — relation to prior bounded-memory attention work
- [`docs/RELEASE.md`](docs/RELEASE.md) — release contents and verification notes

## Scope and limitations

QCC is currently a research and systems prototype.

The retention path is implemented and covered by reproducible benchmark
artifacts. The vLLM integration remains experimental, and serving measurements
depend on the stated model, hardware, batching policy and retained-state budget.

A fixed cache budget is not sufficient for every query. Workloads that require
many simultaneously named items or dense multi-hop dependencies can require a
larger retained set.

The repository includes a 1M-token prefill and state measurement. It does not
claim 1M retrieval parity: on the available 24 GiB hardware, the Full-KV
checkpoint that fits at 1M does not provide a reliable 1M retrieval baseline.

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
