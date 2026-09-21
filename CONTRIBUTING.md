# Contributing

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev,hf]'
pytest -q                       # must stay green; CPU only
```

## What the repository optimises for

1. **Every claim has an artifact.** If a PR changes a number quoted in
   `README.md` or `docs/REPORT.md`, it must add or update the JSON under
   `artifacts/` and the row in `docs/CLAIMS.md` in the same commit. A number
   without an artifact is treated as unverified.
2. **Measurements name their conditions.** Latency rows state the measurement
   window and whether the card was shared. Quality rows state the matched
   Full-KV baseline they are a ratio of. Never report a metric as met when the
   comparison could not be run — say what could not be measured and why.
3. **The law stays a cache policy.** No training, no added parameters, no
   modification of pretrained weights. `tests/test_retention.py::test_compile_touches_no_parameter`
   enforces this.
4. **Equivalence claims are checked on the selected set, not on the scores.**
   Two algebraically equal scorers do not select the same slots: a rewritten
   einsum once cost 7 points of aggregate retention through floating-point
   tie-flips (`docs/REPORT.md` §3.20). Compare retained index sets.

## Adding a benchmark

`benchmarks/` scripts are standalone: they parse `argparse` flags, print progress
lines, and write one self-describing JSON containing the configuration and every
per-record row (prediction, score, retained slots, timings, peak memory). Follow
`benchmarks/benchmark_retention_multimodel.py` as the reference shape, and add
the run to `docs/REPRODUCING.md`.

## Adding a test

CPU tests only for the core policy: build a tiny random `LlamaForCausalLM` as
`tests/test_retention.py` does. Anything needing a real checkpoint or a GPU
belongs in `benchmarks/` with the configuration recorded in its JSON.

## Style

Match the surrounding code: `from __future__ import annotations`, explicit
tensor shapes in comments, docstrings that say *why* rather than restating the
signature. No formatter is enforced; keep lines under ~100 characters.

## Reporting a problem

Include the JSON artifact (or the command that produced it), the model path and
the commit. For performance issues, state whether the GPU was shared.
