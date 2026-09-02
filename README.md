# QCC-Transformer

Query-Compiled Cache (QCC) Transformer is a PyTorch research prototype for long-context decoding. It keeps an exact recent window and compiles older tokens into a constant-size set of multi-timescale softmax responses.

This repository is an implementation of a falsifiable hypothesis, not a claim of a solved production system. The reference path uses Python sequence loops so that the state equations are easy to inspect. CUDA inference dispatches fused archive update/read kernels for each bounded block; projection, local-attention, and cache-maintenance fusion remain open systems work.

## Why this is different

The archive does not store one K/V pair per historical token. For each KV head, it stores responses to a learned codebook of long-range queries under several exponential decay rates. At each decode step, only the exact local window is scanned; historical state is read in constant time with respect to context length. The precise mechanism is best described as a *learned-landmark, multi-timescale separable-kernel memory*: code vectors are fixed during inference, while numerator/denominator response statistics are updated recurrently.

The intended comparison is against full KV attention and sliding-window/GQA baselines. Related work includes linear attention, recurrent memory, KV eviction, KV quantization, MLA, and Infini-attention. Local attention plus bounded recurrent state, softmax numerator/normalizer recurrences, multi-scale decay, and gated local/remote mixing are established ingredients (for example Infini-attention, ABC/GSA, RetNet/KDA, and recent compressed-KV hybrids). The defensible QCC delta is the fixed per-head landmark codebook and its response-statistics parameterization; novelty must be demonstrated by a matched state-byte/quality/latency Pareto comparison rather than by the bounded-state structure alone.

QCC does not compute an exact global softmax over all historical tokens. Each code/scale response is normalized independently before routing and scale mixing, so long-range retrieval is an approximation whose quality must be measured on trained checkpoints. The repository intentionally keeps this boundary explicit and reports missing RULER/retrieval/GPU gates as missing.

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
Its finite causal window is dispatched through fused SDPA with a band mask,
so chunked serving avoids materializing an unfolded window tensor. The archive
events remain ordered and stateful.

For the counterintuitive sparse-memory variant, keep an overcomplete landmark
bank but touch only the highest-scoring slots at inference:

```bash
python benchmarks/benchmark_decode.py --length 8192 --window-size 32 \
  --num-codes 512 --active-codes 4 --lazy-decay --archive-read-stride 8 \
  --warmup 1 --steps 1
```

`active_codes` changes only inference reads; training remains dense so the
codebook receives gradients from every landmark. `lazy_decay` stores a logical
timestamp per slot and applies elapsed decay when that slot is touched. This
makes update cost depend on the selected slots rather than the full bank, at
the cost of an explicitly approximate top-k read and a larger persistent
state. `archive_read_stride` additionally reuses the last remote response for
intermediate decode steps; it is an approximation knob and currently applies
to the token-at-a-time `decode_step` path.
An alternative adaptive knob, `archive_query_cosine_threshold`, refreshes the
remote response only when query similarity falls below a threshold; it is
disabled by default and is useful for workloads with repeated hidden states.
The archive is still updated on every eviction, so this trades read freshness
for latency and should be evaluated with a drift metric.

To measure the resulting logit drift rather than only wall time, run:

```bash
python benchmarks/benchmark_tradeoff.py --length 1024 --window-size 32 \
  --num-codes 128 --active-codes 4 --strides 1,2,4,8
```

The same harness can probe adaptive query reuse (optionally with a repeated
token stream):

```bash
python benchmarks/benchmark_tradeoff.py --length 128 --window-size 16 \
  --num-codes 32 --active-codes 4 --strides 1,2 \
  --query-thresholds 0.5 --repeat-token
```

In one CPU diagnostic run, the threshold-0.5 setting reached mean logit cosine
`0.999878` and `1.05x` relative latency. This is workload-dependent evidence
for the adaptive read knob, not a general speedup claim.

In one single-thread CPU run, stride 2/4/8 achieved `1.16x`/`1.28x`/`1.33x`
relative latency with mean logit cosine `0.999225`/`0.998860`/`0.998646`.
The exact numbers depend on model size and hardware; stride 1 is the reference
and produces zero drift.

`decode_step` maintains a persistent local/archive state and processes one
token at a time. Use `--mode prefill` to benchmark the sequence-level path.

To measure asymptotic behavior across context lengths, run:

```bash
python benchmarks/benchmark_scaling.py --lengths 256,512,1024,2048
```

Use `--mode prefill` to measure sequence-level block scanning (the optimized
archive path), and `--threads 1` to make CPU timings reproducible:

```bash
python benchmarks/benchmark_scaling.py --mode prefill \
  --lengths 256,512,1024 --threads 1
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

The sequence-level prefill path uses the same block recurrence as streaming
decode (rather than a Python update/read loop). On CUDA with Triton, dense
decode chunks use two launches per bounded block (one recurrent update/partial
response kernel and one code-routing reduction), replacing the former
per-token launch pair. On the reference CPU with
`--threads 1`, `d_model=256`, two layers, 8 heads, `window_size=32`, and
`num_codes=16`, one run with warmup 1 and steps 2 measured:

| Prefill tokens | QCC | Full KV | Relative |
|---:|---:|---:|---:|
| 512 | 0.371 s | 0.581 s | 1.57x |
| 1,024 | 0.904 s | 2.013 s | 2.23x |

The benchmark exposes `--threads` so these CPU measurements can be reproduced.
They are implementation evidence for bounded local/archive work, not a
universal hardware-independent claim.

The bounded local-SDPA path is also selected on Apple MPS; only CPU keeps the
unfolded reference fallback. A small matched MPS smoke run (`d_model=64`, one
layer, four heads, `window_size=128`, eight codes, chunk size 64) measured
`9.78x` QCC/full prefill at 16,384 tokens (`0.441 s` vs `4.308 s`). At 32,768
tokens QCC prefill completed in `0.551 s`, while the FullAttention control
failed with a 16-GiB temporary-buffer allocation. These observations validate
bounded execution and an OOM boundary on that machine; they are not CUDA or
RULER results.

With the default two-layer/256-wide configuration, the same MPS stream harness
processed 128K tokens in `41.14 s` (`190.24 ms` TPOT) and 1M tokens in
`324.96 s` (`134.61 ms` TPOT), while retaining the `0.016100%` 1M state
fraction. These are QCC-only long-stream measurements: no Full-KV control was
attempted at those lengths, and they do not establish the CUDA speedup or
retrieval-quality gates.

A fresh default-configuration 128K audit is preserved in
[`artifacts/local_mps/audit_128k_qcc_only.json`](artifacts/local_mps/audit_128k_qcc_only.json):
the QCC-only prefill took `31.032 s` and the post-prefill one-token TPOT was
`79.88 ms`. The artifact explicitly marks the Full-KV, retrieval, and quality
fields as missing; it is an execution/state record rather than a long-context
quality or speedup claim.

A matched default-configuration MPS audit (randomly initialized weights,
`chunk_size=256`) is preserved in
[`artifacts/local_mps/audit_8k_16k.json`](artifacts/local_mps/audit_8k_16k.json).
At 16K, QCC prefill was `4.116 s` versus `4.507 s` for Full-KV (`1.095x`),
while one-token TPOT was `16.23 ms` versus `4,012.18 ms` (`247.14x`). At 8K,
the corresponding factors were `0.770x` prefill and `2.08x` TPOT. The large
16K decode gap is an MPS matched-control observation, not evidence for the
128K/1M CUDA gates; the 8K prefill slowdown also shows why both TTFT and TPOT
must be reported separately.

A smaller one-layer smoke configuration (`d_model=16`, two heads, four codes)
also completed a 4M-token MPS stream in `51.57 s` with `91.70 ms` TPOT and a
`0.003425%` state fraction. This demonstrates the out-of-core streaming path
at the configured FullAttention context length; it is not a 4M FullAttention
execution or a language-quality result.

### Colab CUDA/Triton evidence (Tesla T4)

The run is reproducible with the Colab CLI (after `colab auth login`):

```bash
colab new --session qcc-gpu --gpu T4
colab install --session qcc-gpu triton
printf '%s\n' '...' | colab exec --session qcc-gpu --timeout 1800
colab stop --session qcc-gpu
```

The repository was cloned into a Colab T4 session and audited with
`python -m pytest -q`; all `24` tests passed, including matched dense and sparse
Triton-vs-reference state/output checks. The compact record is preserved in
[`artifacts/colab_gpu/t4_summary.json`](artifacts/colab_gpu/t4_summary.json),
with the raw long-stream lines in
[`artifacts/colab_gpu/long_stream.log`](artifacts/colab_gpu/long_stream.log).

Using the default two-layer/256-wide configuration (`chunk_size=256`), the
QCC-only CUDA streams measured:

| Context | Prefill | TPOT | QCC state / Full-KV | Reduction |
|---:|---:|---:|---:|---:|
| 128K | 3.831 s | 71.8 ms | 0.125781% | 795x |
| 1M | 14.312 s | 72.4 ms | 0.016100% | 6,211x |
| 4M | 54.564 s | 71.7 ms | 0.004025% | 24,845x |

The 1M/4M runs intentionally do not fabricate a Full-KV baseline: allocating
that control is not feasible on the T4. A matched random-weight logit-fidelity
control reached cosine `0.99917` at 1,024 and `0.99790` at 2,048 tokens; this is
a kernel-equivalence check, not RULER, retrieval, or pretrained-LM quality.
The target retrieval gates remain `missing` until a trained checkpoint and
JSONL evaluation set are supplied.

### Colab synthetic retrieval diagnostic (STE content gate)

The differentiable archive uses a straight-through content gate: the forward
pass is exactly the hard inference threshold, while a sigmoid surrogate supplies
gradients below the threshold.  A query-position curriculum (uniform positions
32--500 during 512-token training) was trained on a Colab Tesla T4 and then
evaluated with the persistent `decode_chunk` path.  The checkpoint and machine-
readable record are preserved in
[`artifacts/colab_gpu/checkpoints/qcc_curriculum.pt`](artifacts/colab_gpu/checkpoints/qcc_curriculum.pt)
and [`artifacts/colab_gpu/retrieval_ste_summary.json`](artifacts/colab_gpu/retrieval_ste_summary.json).

| Synthetic stream | Examples | Correct | Accuracy |
|---|---:|---:|---:|
| 512 tokens, query 384 | 50 | 50 | 100% |
| 128K tokens, query 384 | 20 | 20 | 100% |
| 128K tokens, query 96K | 20 | 20 | 100% |

These records use independently sampled marker/value pairs and are an archive
learnability diagnostic, not RULER or pretrained-language quality.  A separate
max-context=1M fixed-query checkpoint reached 5/5 at query 500K but only 1/5 at
query 900K; therefore the 1M retrieval gate is still open and is not claimed as
passed.

### Persistent/prefix landmark diagnostic

The optional `archive_persistent_landmark` mode stores a constant number of
salient key/value slots outside the decayed response archive.  The
`archive_prefix_landmark` variant keeps the first `num_codes` evicted keys and
routes to them with direct key similarity.  It is an explicit long-tail
ablation, not enabled by default.  The implementation and a T4 run are
recorded in [`artifacts/colab_gpu/landmark_retrieval_summary.json`](artifacts/colab_gpu/landmark_retrieval_summary.json).

The prefix checkpoint reached 10/10 on a 128K/query96K synthetic stream and
5/5 on a 1M/query500K stream.  After fixing prefix-slot immutability and
retraining for 3,000 steps, the all-value stress batch (IDs 3--31) improved to
16/29, while the 1M/query900K five-example probe remained 1/5.  This exposes a
remaining multi-value/generalization failure, so the 1M retrieval ≥98% gate
remains unmet.  The extra landmark state is still constant in sequence length;
with the default 16 codes and 32-dimensional heads it adds only 2,304 fp32
state elements per batch/layer.

As a targeted diagnostic, the optional `archive_prefix_pair_landmark` mode
delays each retained slot's value by one event, binding a marker key to its
successor value.  A fresh no-position checkpoint trained for 1,800 steps on a
10-way query curriculum achieved 100% training accuracy but only `1/58`
(`1.72%`) on a strict all-value stress set spanning 128K and 1M contexts.  The
negative result is recorded in
[`artifacts/remote_gpu/prefix_pair_stress.json`](artifacts/remote_gpu/prefix_pair_stress.json)
and rules out a simple marker-to-successor overwrite as the missing ingredient;
the 1M retrieval gate remains open.

### 1M-token T4 latency/state snapshot

An end-to-end QCC-only stream on a Google Colab Tesla T4 processed one million
tokens with `d_model=64`, eight heads, 32 archive codes, and a 16-token local
window.  Prefill took `11.5038 s`; after kernel warm-up, one-token decode TPOT
was `1.9816 ms` median (`1.9995 ms` mean).  The recurrent/archive state was
`54,272 B`, or `0.0212%` of the hypothetical 1M-token fp16 full-KV cache
(`256,000,000 B`, a `4,721.5×` storage reduction).  The first decode token
(`884.97 ms`) includes one-time kernel/warm-up overhead and is not representative
of steady-state TPOT.  This is a QCC-only latency and storage measurement: no
1M full-KV T4 baseline was attempted, so it does **not** establish a speedup
factor.  Synthetic retrieval at the 1M tail and RULER/pretrained-LM quality
remain open gates.  Raw values are in
[`artifacts/colab_gpu/prefix_1m_latency_state.json`](artifacts/colab_gpu/prefix_1m_latency_state.json).

For million-token serving, the default `position_encoding="sinusoidal"` is
stateless: configuring `max_position_embeddings=4_000_000` does not allocate a
four-million-row learned position table. `archive_scan_block_size` controls the
temporary prefill working set (256 by default); reducing it lowers peak memory,
while increasing it can improve throughput on a larger device.
Unless `archive_decay_rates` is supplied explicitly, each layer derives a
log-spaced set of exponential half-lives from `window_size` to
`max_position_embeddings`. This keeps the slowest archive scale aligned with
the configured context instead of silently forgetting million-token history.
The rates are stored in fp32 for the reference/Triton paths, so horizons very
close to the fp32 unit-roundoff can have a small (few-percent) effective
half-life error; pass explicit rates when an ablation needs exact control.
For relative-position experiments, set `position_encoding="rope"` (and tune
`rope_theta`, default `1_000_000`); the rotary phase is applied consistently to
prefill, token decode, and chunk decode. This is a compatibility/quality option,
not an additional novelty claim.

Use the long-context harness to audit storage at the target context before
launching an expensive run:

```bash
python benchmarks/benchmark_long_context.py --length 128000 --state-only
python benchmarks/benchmark_long_context.py --length 1000000 --state-only
python benchmarks/benchmark_long_context.py --length 4000000 --state-only
```

With the default two-layer/256-wide configuration, the reported persistent QCC
state fractions were `0.125781%`, `0.016100%`, and `0.004025%` of hypothetical
full-KV storage at 128K, 1M, and 4M tokens respectively. These are storage
accounting results, not retrieval-quality or latency scores. On a CUDA device,
add `--run --chunk-size 256` to stream a synthetic prompt and report prefill
time plus one-token TPOT.

For an actual retrieval-quality gate, use a trained checkpoint and JSONL
records of the form `{"input_ids": [...], "target_position": 12345,
"answers": [token_id]}`:

```bash
python benchmarks/evaluate_retrieval.py --dataset ruler_export.jsonl \
  --checkpoint qcc.pt --vocab-size 32000 --max-position-embeddings 1000001 \
  --chunk-size 256 --target-accuracy 0.98 --device cuda
```

For a feasible shorter-context quality control, append
`--compare-full-kv`; it evaluates the same checkpoint with the full-KV control
and reports the QCC/full accuracy ratio. Do not enable that option at 1M/4M
unless the device can hold the deliberately unbounded baseline.
For a checkpoint trained with rotary positions, pass
`--position-encoding rope --rope-theta 1000000` to both model paths.

The evaluator streams each record and reports whether the requested retrieval
threshold was met. It requires a checkpoint and cannot silently turn an
untrained model into a claimed RULER score.

For a matched quality smoke test without a checkpoint, use the deterministic
random-weight control (this measures logit fidelity, not language quality):

```bash
python benchmarks/benchmark_fullkv_quality.py \
  --lengths 64,128,256 --chunk-size 32 --threads 1
```

One CPU run reported mean logit cosine `0.999338`, `0.998819`, and `0.998802`
at 64/128/256 tokens (target `0.99`). Longer-context quality must be measured
with trained weights and the retrieval evaluator above.

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

To run all available gates in one machine-readable report, use:

```bash
python benchmarks/audit_targets.py --state-only --json
```

Add `--run-latency --device cuda` for end-to-end QCC timing. Full-KV timing is
only attempted at lengths no larger than `--full-max-length` because the
baseline is intentionally unbounded. Supplying both `--checkpoint` and
`--dataset` enables the real retrieval gate; absent prerequisites are reported
as `missing`, never as a passing score. For a modest-context matched quality
check on the same retrieval records, add `--compare-full-retrieval`; the full
baseline is automatically refused beyond `--full-max-length`.

### Colab GPU audit

[`colab_qcc_gpu.ipynb`](colab_qcc_gpu.ipynb) is a one-click CUDA/Triton entry
point for a Colab T4/A100 runtime. It installs Triton, runs the test suite,
writes a JSON audit for short matched QCC/Full-KV lengths, and can then run
explicitly requested 128K/1M/4M QCC streams. Without the notebook, run:

```bash
python benchmarks/colab_gpu_smoke.py \
  --lengths 8192,16384 --compare-full \
  --output artifacts/colab_smoke_short.json
```

Add `--run-long --long-lengths 128000,1000000,4000000` only on a runtime with
enough wall time and memory. The wrapper refuses to run without CUDA and
Triton, preserves machine-readable `missing` gates, and never treats a
QCC-only long stream as evidence for Full-KV quality, RULER, or a speedup
factor.

Each long run is isolated in its own subprocess and writes combined
stdout/stderr to `artifacts/colab_long/length_<N>.log`; an OOM or timeout is
recorded with a nonzero return code so later lengths can still be attempted.
The wrapper reports `partial` rather than silently dropping failed lengths.

### Target-gate status

The requested headline gates are intentionally tracked separately from the
available smoke evidence:

| Gate | Current evidence | Status |
|---|---|---|
| State @128K/1M/4M | `0.1258%` / `0.0161%` / `0.0040%` | measured |
| Full-KV quality | `0.9981–0.9990` logit cosine at 512–2,048 tokens | short-context only |
| RULER @128K | no trained checkpoint/dataset in this checkout | missing |
| Retrieval @1M | evaluator and gate exist; no scored checkpoint | missing |
| TTFT/TPOT speedup | MPS matched control reaches `247.14x` TPOT at 16K; CUDA and 128K/1M TTFT remain unbenchmarked | short-context only |
| FullAttention context @4M | hypothetical storage accounting only | configured, not executed |

No row marked `missing` or `short-context only` should be reported as a
production or universal speedup claim.

### Remote multi-GPU matched timing

The repository now includes matched RTX 3090 measurements in
[`artifacts/remote_gpu/`](artifacts/remote_gpu/).  At 16K and 32K tokens, the
QCC path reached `2.22×` and `5.17×` TPOT speedup against fused Full-KV, while
TTFT speedups were `1.06×` and `2.21×`.  Prefill scaling crossed Full-KV at 4K
(`1.62×`) and reached `2.91×` at 8K.  Decode scaling was only `0.70–0.89×`
through 8K because this reference path still has Python-side per-token work;
the results therefore motivate a fused persistent decode kernel rather than
supporting the 1M TPOT ≥8× gate.  The 8K audit row contains a cold Triton
compile outlier and is retained for auditability.

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

The trainability smoke test in
[`artifacts/local_cpu/synthetic_retrieval_2000_b4.txt`](artifacts/local_cpu/synthetic_retrieval_2000_b4.txt)
uses a one-layer, 16-wide model with an 8-token exact window and a random value
that must be recalled after a 40-token gap. After 2,000 optimization steps,
QCC reached `1.000` held-out retrieval accuracy (`0.0208` loss), matching the
Full-KV control's `1.000`. This is a small diagnostic task only; it does not
establish RULER, 128K/1M retrieval, or pretrained-language-model quality.

As a separate CPU trade-off measurement, the sparse configuration above (512
codes, top-4, lazy decay, read stride 8, window 32, one thread) measured
`7.018 s` for QCC versus `13.373 s` for the full-KV control at 8,192 tokens
(`1.91x`). Its bounded state was `1,146,880` elements versus `8,388,608`
full-KV elements (`7.31x` fewer), including logical timestamp slots. At 4,096
tokens the same sparse family reached parity (`1.01x` in a two-step run), so
the crossover is strongly context-length dependent. Latency is workload- and
seed-dependent; re-run on the target hardware before making a systems claim.
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

The current reference implementation has three deliberate limitations:
additive sinusoidal positions are used by default (RoPE is available as an
explicit option), the training path retains sequence-level Python loops, and
the archive uses a clipped exponential accumulator. These are appropriate for
validating the architecture and its gradients, but not for production
throughput. The archive state is accumulated in fp32 to reduce long-stream
drift.

On a CUDA installation with Triton available, construct the model with
`use_triton=True` (the default) to dispatch fused archive-update and archive-read
kernels during `no_grad()` decoding. Sparse lazy configurations with power-of-two
`active_codes` additionally dispatch fused selected-slot update/read kernels;
other shapes automatically use PyTorch. Dense `decode_chunk` calls dispatch a
two-kernel fused archive path per `archive_scan_block_size` block. Power-of-two
sparse/lazy chunks now use a vectorized top-k preparation plus a fused
timestamped update/read kernel and one routing reduction; other sparse shapes
retain the ordered reference path. The fused chunk wrappers reuse one bounded
partial-response scratch tensor and one output tensor across all blocks, which
removes per-block allocator and concatenation overhead during long prefill.
These kernels are optional and have not been benchmarked in this CPU-only
environment.

## License

MIT
