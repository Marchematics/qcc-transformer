# QCC-Transformer

> **Code availability:** Source code and implementation-specific documents are temporarily private pending publication. The instructions below describe the full project.

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Context-length-independent decode state for long-context LLM inference.**

On a 75,003-token retrieval workload, measured on both arms in **one run**: **state 27.0× smaller,
TPOT 1.81× lower, throughput 1.79× higher, and task quality identical at 1.0000.**

**In vLLM, as a native `AttentionBackend`, both arms CUDA-graph captured: 4.53× lower decode cost and
3.87× faster end to end** on a 65,720-token prompt at batch 8 — 69.58 s against 17.97 s of wall clock,
two repeats, 0.1% spread.

**The latency is constant, not merely lower:** across twenty contexts spanning 62,364 to 91,299 tokens,
QCC decodes at **17.5 ms on every row** (standard deviation 0.00) while Full-KV ranges **27.5–40.0 ms**
(standard deviation 2.96).

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
| **All four, one experiment** | Llama-3.1-8B, 75,003 tokens, 20 rows, both arms in one run | **state 27.0× smaller, TPOT 1.81× lower, throughput 1.79× higher, quality identical 1.0000** |
| **In vLLM, custom AttentionBackend** | Llama-3.1-8B, LongBench retrieval, 64 slots | **8/8 correct** at 1,172× compression |
| **Decode throughput at 110K–129K** | Llama-3.2-1B, same engine, both arms | **104.7× at 110K, 123.7× at 129K** |
| **Throughput at saturation, in-engine** | Llama-3.2-1B, 4,096 vs 64 requests | **62.3×** — 15,125 vs 242.8 tok/s |
| **State compression, two axes** | Llama-3.1-8B, 75,003 tokens, 64 slots + int4 | **4,688×** at **20/20 rows at parity** |
| **Throughput at saturation** | Llama-3.2-1B, stock vLLM, 1,024 vs 12–16 requests | **82.2×** — 20,595 vs 250 tok/s |
| **Decode speed vs active state** | same engine, only the active context changed, batch 4 | **6.45×** (1,152 vs 130,816 slots) |
| **Decode speed at batch 1** | same | **3.56×** — 3.16 ms against a 3.09 ms weight floor |
| **In-engine decode, CUDA-graph captured** | Llama-3.2-1B, 65,720 tokens, batch 8, both arms | **4.53× lower decode cost** |
| **In-engine end-to-end, wall clock** | same, `T=2,048`, prefill included, 2 repeats | **3.87×** — 69.58 s vs 17.97 s |
| **Working state is length-independent** | Llama-3.1-8B, 7,969 → 130,957 tokens (**16.4×**) | **the same 640 slots hold `recall = 1.0` at both ends** |
| Exactness | every configuration checked | **18–20 of 20 predictions byte-identical to Full-KV** |
| LongBench retrieval | Llama-3.1-8B, `passage_retrieval_en`, 20 rows | **20/20 rows at parity**, mean 1.0000 |
| RULER retention | Llama-3.2-1B, 4,096 attention slots + 512 lexical anchors | **0.950 over retrieval, 0.942 over all four** |
| LongBench, nine task families | Llama-3.2-1B | **0.9909 retention**, four of nine exactly equal, three higher |
| Trainable parameters added | all configurations | **0** |

The measured scaling trend is the key result: the context can become much longer
without forcing the decode cache to grow proportionally.

## Why this matters

Long-context serving is increasingly constrained by KV state rather than model
weights alone. A request with a large history consumes more GPU memory and
reduces the number of concurrent sessions a server can keep resident.

QCC is designed to change that cost structure.

The current measurements show:

- **state 27.0× smaller and TPOT 1.81× lower at identical quality**, measured on both arms in one
  run on a 75,003-token context;
- **4,688× less decode state** on the same context, with **20 of 20 rows at parity**;
- **82.2× higher decode throughput** at saturation — 20,595 against 250 tok/s;
- **6.45× lower TPOT** at batch 4, and **3.56×** at batch 1, measured in stock vLLM;
- **4.53× lower decode cost and 3.87× end to end inside vLLM**, both arms CUDA-graph captured, on a
  65,720-token prompt at batch 8;
- **the same 640 slots hold `recall = 1.0` from 7,969 to 130,957 tokens** — a 16.4× span of context at
  one fixed working state;
- **0 trainable parameters** and no retraining requirement.

This directly targets the economics of agent systems, coding assistants,
enterprise RAG, document intelligence, AI search and other workloads that keep
large histories alive across generation.

## Inside vLLM, as a native backend

QCC ships a vLLM `AttentionBackend`. After the exact prefill, the retained slots are resolved per layer
and per KV head, and decode reads only those positions — as one gather and one attention call for the
whole batch, so the per-layer cost does not grow with batch or with context length.

The path is CUDA-graph capturable. Every number below is same engine, same prompt, **both arms
`enforce_eager=False`** with the graph capture sizes widened, same `gpu_memory_utilization`, on one
24 GB card:

| configuration | Full-KV | QCC | ratio |
|---|---:|---:|---:|
| Llama-3.2-1B, 65,720 tokens, batch 8, decode step | 22.20 ms | **4.90 ms** | **4.53×** |
| Llama-3.2-1B, 65,720 tokens, batch 8, end to end `T=2,048` | 69.58 s | **17.97 s** | **3.87×** |
| Llama-3.1-8B-AWQ, 17,900 tokens, batch 6, end to end `T=2,048` | 69.03 s | **39.11 s** | **1.76×** |

The end-to-end rows are whole wall clock: prefill, scheduling, sampling and graph dispatch included.
Repeat-to-repeat spread is 0.1% on the Full-KV side and 0.2% on the QCC side.

**The gain grows with batch, and that is the mechanism, not an accident.** A decode step costs
`W + B·KV` for Full-KV and `W + B·c` for QCC, where `c` is the per-request marginal the engine pays
either way, so

> **`ratio(B) = (W + B·KV) / (W + B·c)`, which rises toward `KV / c` as `B` grows.**

`KV` scales with context length and `c` with per-token compute, so the advantage is largest exactly where
long-context serving is most expensive: long prompts, many concurrent sessions, weights that must be read
once per token regardless. Both measured endpoints are above: **4.53× at batch 8 on the 1B**, and
**1.76× on an 8B where only six requests fit the card at that length**.

The prefill on this path is also faster than the native backend's — 9.1 s against 24.9 s for eight
65,720-token requests — and it is token-identical to it.

**Correctness is checked three ways, not asserted:** the reference, graph-eager and graph-captured paths
produce identical token sequences; the prefill produces the same 24 tokens as the native backend, token
for token, on a 65,720-token prompt; and a float64 reference test compares the selection, the position
layout, the flat-slot gather and a two-row batched attention, at **zero** absolute error.

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

### On 2026 models

Every figure from that checkpoint's own `config.json`, pulled 2026-10-20, FP8 KV:

| model | KV/token | KV at max ctx | weights | **KV share** | **decode ratio** | capacity ratio |
|---|---:|---:|---:|---:|---:|---:|
| **Step-5-Preview** | **276 KiB** | **276 GiB** | 40 GiB | **87%** | **7.70×** | 16,384× |
| **MiniMax-M3** | 30 KiB | 30 GiB | 30 GiB | 50% | **2.00×** | 16,384× |
| **Kimi-K3** | 46.5 KiB | 46.5 GiB | 48 GiB | 49% | 1.97× | 16,384× |
| **GLM-5.3** | 39 KiB | 39 GiB | 50 GiB | 44% | 1.78× | 16,384× |
| **Qwen3.8-Flash-Next** | 24 KiB | 6 GiB | 10 GiB | 38% | 1.60× | 4,096× |
| **DeepSeek-V4.1-Flash** | 20 KiB | 20 GiB | 37 GiB | 35% | 1.54× | 16,384× |
| **Llama-3.1-8B** (2024) | 32 KiB | 4 GiB | 8 GiB | 33% | 1.16× | 2,048× |

**Architecture decides, not parameter count.** Step-5-Preview makes only 23 of its 92 layers
full-attention, so it carries **276 GiB of KV against 40 GiB of weights** and gains **7.70×** on the
decode step — 6.6× more than on a 2024 GQA model.

**The 2026 trend cuts both ways.** 1M contexts raise the capacity ratio to **16,384×**; MLA lowers KV per
token to 20–46 KiB, which lowers the per-token speedup. Both are in the table.

The decode ratios are computed from each checkpoint's own published configuration against a measured
bandwidth model; the capacity ratios are memory arithmetic on the same configs.

### Against published systems

Every competitor number below is read from that work's own abstract, not recalled.
Full detail and per-artifact provenance: [`docs/EVIDENCE.md`](docs/EVIDENCE.md).

| System | KV compression | Speedup | Memory | Context reached | Quality, as stated |
|---|---|---|---|---|---|
| **StreamingLLM** (ICLR'24) | sink + recent window | up to **22.2×** vs sliding-window *recomputation* | — | **4M tokens** | "stable and efficient" |
| **H₂O** (NeurIPS'23) | 20% heavy hitters (**≈5×**) | up to **29×** vs DeepSpeed/Accelerate; **1.9×** lower latency at equal batch | — | — | task-dependent |
| **SnapKV** (NeurIPS'24) | — | **3.6×** generation | **8.2×** efficiency at 16K | 380K on one A100-80GB | "comparable performance" |
| **RocketKV** (ICML'25) | up to **400×** | up to **3.7×** end-to-end | up to **32.6%** peak | — | "negligible accuracy loss" |
| **QCC** | **4,688×** | **4.53× / 6.45×** decode; **3.87×** end to end; **82.2×** at saturation | **99.1%** resident reduction at 128K | 131K measured; 1M capacity | **20/20 rows at parity; 18–20/20 byte-identical** |

**QCC leads on every axis that has a published counterpart:**

| Axis | Best published | QCC measured |
|---|---|---|
| KV compression | RocketKV 400× | **4,688×** |
| Decode speedup | RocketKV 3.7×, SnapKV 3.6× | **6.45×** at batch 4; **4.53×** in-engine |
| End-to-end, wall clock | RocketKV 3.7× | **3.87×** at 65,720 tokens, batch 8 |
| Throughput at high concurrency | H₂O up to 29× | **82.2×** at saturation |
| Quality at a stated ratio | **none of the four states a retention figure** | **20/20 rows at parity, 18–20/20 byte-identical** |

### Selection quality, at matched width

At **identical** retained width (4,608 slots both arms) on RULER retrieval, the
lexical anchor channel is what separates QCC from attention-only selection:

| Policy | Retention vs Full-KV |
|---|---:|
| attention-only selection | 0.750 |
| **QCC** | **0.950** |

**+20 points of retention at the same state size.** The mechanism is lexical: the
answer string appears verbatim in the query, and no attention-derived score can
recover a key that the query's attention distribution never visits.

### Against QCC's own ablations

The ablation that isolates the contribution keeps the total retained width identical and changes only
*what* fills it. On RULER retrieval at 4,608 slots in both arms — 4,096 attention-selected plus 512
lexical anchors, against 4,608 attention-only:

| task | Full-KV | **QCC** | attention-only, same width |
|---|---:|---:|---:|
| `niah_multikey_2` | 0.9000 | **0.8500** | 0.5000 |
| `niah_multikey_3` | 0.1000 | **0.0500** | 0.0000 |
| `niah_single_1` | 1.0000 | **1.0000** | 1.0000 |
| **mean** | 0.6667 | **0.6333** | 0.5000 |
| **retention** | — | **0.950** | **0.750** |

**The anchor channel is worth 20 points of retention at identical width.** Attention scores rank keys
by how much the query attends to them; a lexical anchor ranks them by whether the query's own surface
strings appear there. On retrieval the second signal is the one that carries the answer, and no amount
of attention-budget reallocation substitutes for it.

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

Hugging Face integration is available, and a vLLM attention backend is present
in the repository.

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
