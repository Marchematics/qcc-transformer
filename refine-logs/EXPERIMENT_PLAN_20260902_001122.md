# Experiment Plan

**Problem**: Reduce Transformer long-context memory and serving latency while retaining retrieval quality.
**Method thesis**: A fixed per-head landmark codebook with multi-timescale response statistics can replace historical KV storage, while a fused local-window/Triton path removes token-level launch overhead.
**Date**: 2026-09-02

## Claim Map

| Claim | Why it matters | Minimum convincing evidence | Linked blocks |
|---|---|---|---|
| C1: QCC gives bounded state and accelerator speed at long context | Establishes the systems contribution | 1M and 4M stream, matched Full-KV latency at feasible length, state fraction <=0.1% | B1, B2 |
| C2: landmark response statistics preserve retrieval | Establishes usefulness beyond compression accounting | Trained checkpoint, 128K RULER >=90%, 1M retrieval >=98%, Full-KV quality ratio >=99% | B3, B4 |

## Paper Storyline

- Main paper must prove: constant-state scaling, fused CUDA serving, and trained long-range retrieval.
- Appendix can support: decay/active-code ablations, MPS/CPU diagnostics, and failure cases.
- Experiments intentionally cut: claims of exact global softmax or universal hardware-independent speedup.

## Experiment Blocks

### Block 1: CUDA systems anchor

- Claim tested: bounded state and practical long-stream execution.
- Dataset / task: synthetic streaming prompts at 128K, 1M, and 4M.
- Compared systems: QCC; matched Full-KV only at feasible lengths (<=16K on T4).
- Metrics: prefill/TTFT, TPOT, peak memory, persistent state bytes.
- Setup: Tesla T4, `d_model=256`, 2 layers, 8 heads, window 128, 16 codes, chunk 256.
- Success criterion: state <=0.1% at 1M; no OOM at 4M; report matched speedup without hiding compile warmup.
- Failure interpretation: a speed result without a matched baseline is QCC-only evidence, not a speedup claim.
- Priority: MUST-RUN.

### Block 2: Kernel equivalence and ablations

- Claim tested: fused dense/sparse Triton kernels implement the reference recurrence.
- Task: deterministic state/output comparisons over multiple event lengths and block boundaries.
- Metrics: max absolute/relative state and output error; launch count and steady-state latency.
- Success criterion: CUDA tests pass with <=1e-2 tolerance and parameter-matched controls.
- Priority: MUST-RUN.

### Block 3: Trained 128K retrieval

- Claim tested: QCC retains long-range retrieval after training rather than only matching random logits.
- Dataset: RULER-compatible JSONL (or a documented synthetic long-range split while RULER is unavailable).
- Compared systems: QCC and same-checkpoint Full-KV at feasible control length; QCC-only at 128K.
- Metrics: retrieval accuracy, Full-KV accuracy ratio, mean logit cosine, TTFT/TPOT.
- Setup: train a small checkpoint with sinusoidal/relative position support, then stream records in 256-token chunks.
- Success criterion: RULER 128K >=90%; QCC/Full-KV quality >=99% on matched control.
- Failure interpretation: synthetic success cannot be relabeled as RULER; missing checkpoint/data remains missing.
- Priority: MUST-RUN.

### Block 4: 1M retrieval and novelty isolation

- Claim tested: the landmark/multi-timescale mechanism works at million-token distance and gains are not only from fewer codes.
- Dataset: 1M retrieval JSONL with known target positions and answer sets.
- Ablations: dense vs top-k active codes, single-scale vs multi-scale, learned landmarks vs random/fixed landmarks, local-only baseline.
- Metrics: accuracy, state fraction, TTFT/TPOT, per-distance accuracy curve.
- Success criterion: retrieval >=98%, state <=0.1%, and no quality collapse relative to Full-KV control where feasible.
- Priority: MUST-RUN once data/checkpoint exist.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision gate | Cost / risk |
|---|---|---|---|---|
| M0 | Validate data/metric and overfit a short retrieval task | tiny synthetic, 64-2K | accuracy >95%; evaluator agrees with hand check | minutes; catches label/position bugs |
| M1 | Validate kernels and matched controls | pytest + 8K/16K CUDA audit | all kernels pass; quality cosine >0.99 | minutes; compile warmup can skew first timing |
| M2 | Train and evaluate 128K checkpoint | staged length curriculum, then 128K | RULER >=90% or diagnose failure | hours; T4 memory/time |
| M3 | Scale retrieval to 1M and run ablations | 1M streaming records | >=98% retrieval and <=0.1% state | hours; dataset/checkpoint availability |
| M4 | Freeze evidence and push | claim audit + artifacts | every target has pass/fail/missing evidence | low cost |

## Compute and Data Budget

- Existing T4 budget: enough for 128K/1M/4M QCC-only streams and kernel tests.
- Training budget: start with a small model and curriculum; avoid claiming RULER until a real dataset is present.
- Data preparation: convert RULER records to `input_ids`, `target_position`, and `answers` JSONL; record tokenizer/checkpoint metadata.
- Biggest bottleneck: a trained checkpoint that uses the QCC attention path at 128K/1M; random-weight cosine is insufficient.

## Risks and Mitigations

- RULER export/tokenizer mismatch: store tokenizer hash and validate target token IDs with a hand-checked record.
- Full-KV OOM at 1M: report QCC-only execution and use a smaller matched control; never infer a quality ratio from missing Full-KV.
- Triton first-launch compilation: warm up kernels before latency measurement and report both cold and warm timing.
- Novelty overclaim: scope the contribution to fixed learned landmarks plus response-statistics parameterization; do not claim first constant memory or first multi-scale decay.

## Final Checklist

- [ ] 128K trained RULER result >=90%
- [ ] 1M trained retrieval result >=98%
- [x] 1M/4M state accounting <=0.1%
- [ ] 128K TTFT >=6x matched baseline
- [ ] 1M TTFT >=10x matched baseline or explicitly marked missing
- [ ] 1M TPOT >=8x matched baseline or explicitly marked missing
- [x] CUDA/Triton correctness tests pass
- [ ] Full-KV quality >=99% on matched trained checkpoint
