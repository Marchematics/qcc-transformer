# QCC-Transformer

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Bounded exact-KV decoding for long-context language-model inference.**

QCC-Transformer is a long-context inference runtime technology that compiles a
completed Full-KV prefill into a fixed-size decode cache. Retained keys preserve
their original rotary positions and values, while generation continues with
ordinary softmax attention over the retained state.

The implementation is training-free, adds no model parameters, and does not
require checkpoint modification.

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

Long-context inference increasingly turns KV state into a serving bottleneck.
With conventional Full-KV decoding, every request carries history-proportional
state through generation. That directly limits GPU concurrency and increases
memory traffic as contexts grow.

QCC changes the serving model:

1. run an exact chunked prefill;
2. identify the prompt state most relevant to the current query;
3. preserve retrieval anchors, attention sinks and recent context;
4. compile a fixed-width KV working set;
5. decode over that retained state with the original positional phases intact.

The decode-state budget is configured independently of prompt length.

## Measured results

All headline results are backed by per-run artifacts and reproduction commands.

| Measurement | Configuration | Result |
|---|---|---:|
| RULER retention | Llama-3.2-1B, 4,096 attention slots + 512 lexical anchors | **1.007 aggregate, 1.000 worst task** |
| LongBench retention | Llama-3.2-1B, nine-task evaluation | **0.9939 aggregate** |
| Decode-state growth | 8K → 256K, 4,608 retained slots | **1.00×**; 144 MiB at both endpoints |
| 1M decode state | Qwen2.5-0.5B, 1,048,576 tokens | **54 MiB vs 12,288 MiB Full-KV — 227.6× reduction** |
| 32K decode throughput | Llama-3.2-1B, A10G 24 GiB | **2,139 tok/s vs 145 tok/s Full-KV — 14.76×** |
| 32K fixed-SLA concurrency | same serving experiment | **32 vs 4 resident requests — 8×** |
| 128K launch-free TPOT | Qwen2.5-0.5B, batch 8, speed configuration | **5.37×** |
| Trainable parameters added | all configurations | **0** |

The central result is the scaling behavior: prompt length can increase without
requiring the decode cache to grow proportionally.

## How it works

### Exact chunked prefill

The prompt is processed through the model's native causal attention in chunks.
This preserves Full-KV prefill semantics while controlling activation memory.

### Query-aware selection

The final prompt tokens form an observation window. Their attention to the
prefill keys provides a per-layer, per-KV-head relevance signal.

### Retrieval anchors

Rare strings, identifiers, numbers and other query-specific spans are matched
back into the prompt and protected from eviction. This complements attention
selection on retrieval-heavy workloads.

### Fixed-width decode state

Each request is compiled to a uniform retained width. Ragged requests can be
batched while preserving each request's original positions.

## Serving implications

QCC is designed for workloads where long contexts make KV state a material part
of serving cost: agents, coding systems, enterprise RAG, document intelligence,
AI search and long-lived sessions.

The current measurements show three effects that matter directly for deployment:

- **bounded decode memory** as context length grows;
- **higher resident concurrency** on the same GPU;
- **higher decode throughput** once Full-KV memory becomes the batching limit.

The design is model-retrofit friendly: existing pretrained checkpoints can be
used without retraining.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev,hf]'
pytest -q
```

## Repository layout

```text
qcc_transformer/   core package and runtime integration
benchmarks/        quality, memory and serving benchmarks
artifacts/         per-run predictions, scores, timings and memory records
docs/              technical report, mechanism and reproduction notes
examples/          runnable examples
tests/             unit and regression tests
```

Useful entry points:

- [`docs/REPORT.md`](docs/REPORT.md) — technical measurement summary
- [`docs/MECHANISM.md`](docs/MECHANISM.md) — retention mechanism and capacity analysis
- [`docs/EVIDENCE.md`](docs/EVIDENCE.md) — measurement-to-artifact-to-command index
- [`docs/REPRODUCING.md`](docs/REPRODUCING.md) — reproduction instructions
- [`docs/NOVELTY.md`](docs/NOVELTY.md) — relation to prior bounded-memory attention work

## Deployment status and roadmap

The core bounded-KV path, batch handling, long-context prefill and benchmark
infrastructure are implemented. Hugging Face integration is available and vLLM
integration is under active engineering.

The next deployment phase focuses on:

- production vLLM and TensorRT-LLM integration;
- H100/H200/B200 serving measurements;
- broader 7B–70B model coverage;
- workload-aware automatic cache budgeting;
- multi-GPU continuous batching and enterprise workload pilots.

The repository already includes million-token state measurements. The next
million-token quality study will use a checkpoint and hardware configuration
that can provide a matched Full-KV reference at that length.

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
