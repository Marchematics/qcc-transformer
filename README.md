# QCC-Transformer

Query-Compiled Cache (QCC) Transformer is a PyTorch research prototype for long-context decoding. It keeps an exact recent window and compiles older tokens into a constant-size set of multi-timescale softmax responses.

This repository is an implementation of a falsifiable hypothesis, not a claim of a solved production system. The reference path uses Python sequence loops so that the state equations are easy to inspect. A useful next systems step is a fused Triton/CUDA update-and-read kernel.

## Why this is different

The archive does not store one K/V pair per historical token. For each KV head, it stores responses to a learned codebook of long-range queries under several exponential decay rates. At each decode step, only the exact local window is scanned; historical state is read in constant time with respect to context length.

The intended comparison is against full KV attention and sliding-window/GQA baselines. Related work includes linear attention, recurrent memory, KV eviction, KV quantization, MLA, and Infini-attention. The novelty boundary must be checked against those families before making a publication claim.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Run tests

```bash
pytest
```

## Run the microbenchmark

```bash
python benchmarks/benchmark_decode.py --length 512 --mode decode --warmup 1 --steps 3
```

For block serving, set `--chunk-size 16` (or another positive block length) to
use the vectorized persistent `decode_chunk` API.

For the counterintuitive sparse-memory variant, keep an overcomplete landmark
bank but touch only the highest-scoring slots at inference:

```bash
python benchmarks/benchmark_decode.py --length 2048 --num-codes 512 \
  --active-codes 4 --lazy-decay --archive-read-stride 4 \
  --warmup 1 --steps 3
```

`active_codes` changes only inference reads; training remains dense so the
codebook receives gradients from every landmark. `lazy_decay` stores a logical
timestamp per slot and applies elapsed decay when that slot is touched. This
makes update cost depend on the selected slots rather than the full bank, at
the cost of an explicitly approximate top-k read and a larger persistent
state. `archive_read_stride` additionally reuses the last remote response for
intermediate decode steps; it is an approximation knob and currently applies
to the token-at-a-time `decode_step` path.

`decode_step` maintains a persistent local/archive state and processes one
token at a time. Use `--mode prefill` to benchmark the sequence-level path.

To measure asymptotic behavior across context lengths, run:

```bash
python benchmarks/benchmark_scaling.py --lengths 256,512,1024,2048
```

The script reports per-length timings and log-log slopes. A slope near 1 for
QCC and near 2 for full KV is the expected bounded-state versus quadratic
scaling signature; actual values depend on hardware and kernel implementation.

Single-thread CPU reference run (`256,512,1024,2048,4096`, warmup 1, steps 2)
against the same fused SDPA primitive and preallocated full-KV control gave
QCC/full timings of `0.164/0.100`, `0.388/0.220`, `0.844/0.496`,
`1.782/1.305`, and `3.609/3.625` seconds. The fitted slopes were `1.113`
(QCC) and `1.292` (full KV). The fixed archive overhead dominates short
contexts; the bounded path reaches parity around 4k tokens in this setup.
These are single-run CPU measurements, not a hardware-independent speedup
claim.

To isolate archive learnability from language-model quality, run the synthetic
teacher benchmark:

```bash
python benchmarks/benchmark_archive_quality.py --steps 200
```

It compares the learned-landmark response with direct exponentially-decayed
softmax attention. This is a mechanism test, not a perplexity or retrieval
benchmark.

Reference run (`--steps 200 --batch 8 --length 64 --dim 16 --codes 8`) reduced
MSE from `0.056180` to `0.030020` (1.87x), while using 136 archive elements per
head instead of 56 historical value positions per query step.

For a minimal learned long-range check, run:

```bash
python benchmarks/benchmark_synthetic_lm.py --steps 100
```

This trains matched QCC and full-KV toy decoders on a random key-value recall
task and reports loss/accuracy at the delayed query. It is a diagnostic for
long-range information retention; it is not evidence of general language
quality.

For a natural-text sanity check, run the on-demand Tiny Shakespeare experiment:

```bash
python benchmarks/benchmark_tiny_shakespeare.py --steps 100
```

The script downloads the public corpus into `work/`, trains matched QCC and
full-KV character decoders, and reports held-out loss/perplexity plus training
time. It uses strict next-token targets (the final context position is not
scored). It is deliberately small and is not a claim about scaled LLM quality.

Example CPU smoke run (`--steps 50 --eval-batches 5 --batch 4 --length 64
--window 16 --d-model 32 --layers 1 --heads 4 --codes 4`) produced validation
perplexity `28.601` for QCC and `32.695` for full KV. This single-seed result is
included for reproducibility only; report multiple seeds before interpreting a
quality difference.

One CPU run (`--steps 200 --batch 16 --length 32 --query-position 24
--window 6 --d-model 24 --layers 1 --heads 4 --codes 4 --vocab 24`) reached
held-out accuracy `1.000` for both QCC and full KV. Held-out loss was `0.001903`
for QCC and `0.000617` for full KV, so this toy result shows retention but not
parity on a real language distribution.

The benchmark reports runtime only. It is not evidence of language-model quality. For a meaningful study, train matched small models and evaluate perplexity, Needle-in-a-Haystack, RULER, PG-19, cache memory, and decode TPOT at 32k+ context.

On the reference CPU environment (PyTorch 2.9.1, one thread), a persistent
decode benchmark with the fused SDPA full-KV control measured:

| Tokens | QCC | Full KV | Relative |
|---:|---:|---:|---:|
| 1,024 | 0.844 s | 0.496 s | 0.59x |
| 2,048 | 1.782 s | 1.305 s | 0.73x |
| 4,096 | 3.609 s | 3.625 s | 1.00x |

At 1,024 tokens in this configuration, the bounded QCC state contains
164,864 elements versus 1,048,576 full-KV elements (6.36x fewer). At 2,048
tokens the reduction is 12.72x. The ratio continues to grow linearly with
context length because the QCC archive and local window are bounded.

These numbers are implementation evidence for the bounded-history trend, not
a claim of a universal speedup. GPU results, batch scaling, kernel fusion, and
language quality remain open experiments.

As a separate CPU trade-off measurement, the sparse configuration above (512
codes, top-4, lazy decay, read stride 4, one thread, two timed steps) measured
`3.637 s` for QCC versus `3.711 s` for the full-KV control at 4,096 tokens
(`1.02x`). Its bounded state was `1,245,184` elements versus `4,194,304`
full-KV elements (`3.37x` fewer), including logical timestamp slots. Latency is
workload- and seed-dependent; re-run on the target hardware before making a
systems claim.
The same configuration is slower at 1,024 tokens because the overcomplete
bank has not yet amortized its fixed state cost.

## Minimal API

```python
import torch
from qcc_transformer import QCCForCausalLM

model = QCCForCausalLM(
    vocab_size=32000,
    d_model=512,
    num_layers=8,
    num_heads=8,
    window_size=2048,
    num_codes=32,
)
input_ids = torch.randint(0, 32000, (1, 4096))
logits = model(input_ids)
```

## Caveats

The current reference implementation has three deliberate limitations: no RoPE, sequence-level Python loops, and a clipped exponential accumulator. These are appropriate for validating the architecture and its gradients, but not for production throughput. The archive state is accumulated in fp32 to reduce long-stream drift.

On a CUDA installation with Triton available, construct the model with
`use_triton=True` (the default) to dispatch the fused archive-update kernel
during `no_grad()` decoding. The kernel is optional and has not been benchmarked
in this CPU-only environment; unsupported devices automatically use PyTorch.

## License

MIT
