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

`decode_step` maintains a persistent local/archive state and processes one
token at a time. Use `--mode prefill` to benchmark the sequence-level path.

The benchmark reports runtime only. It is not evidence of language-model quality. For a meaningful study, train matched small models and evaluate perplexity, Needle-in-a-Haystack, RULER, PG-19, cache memory, and decode TPOT at 32k+ context.

On the reference CPU environment (PyTorch 2.9, one thread configuration not
fixed), the persistent decode benchmark measured:

| Tokens | QCC | Full KV | Relative |
|---:|---:|---:|---:|
| 1,024 | 1.32 s | 2.14 s | 1.62x |
| 2,048 | 2.51 s | 10.39 s | 4.13x |
| 4,096 | 6.42 s | 40.09 s | 6.25x |

At 1,024 tokens in this configuration, the bounded QCC state contains
164,864 elements versus 1,048,576 full-KV elements (6.36x fewer). At 2,048 and
4,096 tokens the reductions are 12.72x and 25.44x. The ratio continues to grow
linearly with context length because the QCC archive and local window are
bounded.

These numbers are implementation evidence for the bounded-history trend, not
a claim of a universal speedup. GPU results, batch scaling, kernel fusion, and
language quality remain open experiments.

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

## License

MIT
