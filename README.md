# QCC-Transformer

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Context-length-independent decode state for long-context LLM inference.**

QCC-Transformer turns the KV cache used during generation from a
history-proportional structure into a fixed-budget working set.

After an exact Full-KV prefill, QCC compiles the prompt state into a bounded
decode cache selected for the current query. Retained keys keep their original
rotary positions and retained values are used directly. Existing pretrained
checkpoints can be used without retraining or checkpoint surgery.

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

## Core advantage

Let `L` be the prefilled context length and `B` the configured retained KV
budget.

| Decode property | Full-KV | QCC |
|---|---:|---:|
| KV state vs. context length | **O(L)** | **O(B) = O(1) in L** |
| KV attention working set per generated token | **O(L)** | **O(B) = O(1) in L** |
| KV memory traffic growth with longer history | grows with `L` | bounded by configuration |
| New model parameters | — | **0** |
| Retraining required | — | **No** |
| Checkpoint modification | — | **No** |

With a fixed retention budget, the decode-side state and the number of historical
KV entries attended by each generated token no longer grow with the original
prompt length.

This is the central systems property of QCC.

## Headline measurements

All reported numbers are backed by stored per-run artifacts and reproduction
commands.

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

The measured scaling trend is the key result: the context can become much longer
without forcing the decode cache to grow proportionally.

## Why this matters

Long-context serving is increasingly constrained by KV state rather than model
weights alone. A request with a large history consumes more GPU memory and
reduces the number of concurrent sessions a server can keep resident.

QCC is designed to change that cost structure.

The current measurements show:

- **227.6× less decode state at 1M tokens** in the measured configuration;
- **14.76× higher decode throughput** at the measured 32K serving operating point;
- **8× higher fixed-SLA concurrency** on the same A10G;
- **5.37× TPOT improvement** at 128K in the measured speed configuration;
- **0 trainable parameters** and no retraining requirement.

This directly targets the economics of agent systems, coding assistants,
enterprise RAG, document intelligence, AI search and other workloads that keep
large histories alive across generation.

## How QCC works

### 1. Exact chunked prefill

The prompt is processed through the model's native causal attention. QCC does
not replace the prefill with a learned approximation.

### 2. Query-aware compilation

The final prompt tokens form an observation window. Their attention to the
prefill keys produces a per-layer, per-KV-head relevance signal.

### 3. Retrieval anchors

Rare strings, identifiers, numbers and other query-specific spans are matched
back into the prompt and protected from eviction. This gives QCC an explicit
path for preserving exact retrieval signals alongside semantic attention scores.

### 4. Fixed-width decode state

The selected KV state is compiled to a uniform retained width. Original
positions are preserved, and ragged requests can be batched.

### 5. Exact attention over retained KV

Decode uses standard softmax attention over the retained entries. QCC selects
which exact KV entries remain live; it does not reconstruct them through a
low-rank latent approximation.

## Competitive position

QCC occupies a different point in the design space from common KV reduction
approaches.

**Sliding-window methods** bound memory by discarding distant context according
to position.

**Heavy-hitter and eviction methods** retain a bounded subset, typically using
online or historical attention statistics.

**KV quantization** reduces bytes per retained KV entry.

**Latent-KV architectures** redesign the model so compact state is learned during
training.

QCC combines query-aware exact-token retention with lexical retrieval anchors
after an exact prefill. It can also compose with KV quantization.

At matched retained width on the stored RULER evaluation:

| Policy | Aggregate retention |
|---|---:|
| sliding window | 0.167 |
| sinks + recent | 0.332 |
| final-query selection | 0.842 |
| window-mean selection | 0.880 |
| **QCC** | **1.008** |

The repository also contains matched measurements against H2O-, TOVA-, Quest-,
PyramidKV- and SnapKV-shaped policies.

## Deployment model

QCC is designed as an inference-layer technology rather than a new foundation
model.

Potential integration surfaces include:

- vLLM and TensorRT-LLM runtimes;
- inference clouds;
- private LLM serving stacks;
- agent platforms;
- coding and research assistants;
- enterprise long-document systems.

The commercial lever is straightforward: increase the amount of long-context
work served by the same GPU fleet.

## Current engineering stage

The core retention path, exact chunked prefill, fixed-width cache compilation,
ragged batching, quality benchmarks and serving benchmarks are implemented.

Hugging Face integration is available. vLLM integration is already present in
the repository and is being hardened for production deployment.

The next engineering phase expands the same runtime to:

- H100/H200/B200 deployment;
- production vLLM and TensorRT-LLM paths;
- broader 7B–70B coverage;
- workload-aware automatic budget selection;
- multi-GPU continuous batching.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev,hf]'
pytest -q
```

## Repository

```text
qcc_transformer/   core runtime and integrations
benchmarks/        quality, memory and serving benchmarks
artifacts/         per-run predictions, scores, timings and memory records
docs/              mechanism, measurements and reproduction notes
examples/          runnable examples
tests/             unit and regression tests
```

- [`docs/REPORT.md`](docs/REPORT.md) — technical measurement summary
- [`docs/MECHANISM.md`](docs/MECHANISM.md) — retention mechanism and scaling analysis
- [`docs/EVIDENCE.md`](docs/EVIDENCE.md) — measurement-to-artifact-to-command index
- [`docs/REPRODUCING.md`](docs/REPRODUCING.md) — reproduction instructions
- [`docs/NOVELTY.md`](docs/NOVELTY.md) — relation to prior bounded-memory attention work

> **Complexity convention.** The O(1) statements above are with respect to
> prefilled context length `L` at fixed retained budget `B`, and refer to
> decode state and the historical KV working set used per generated token.
> Prefill still processes the input sequence.

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
