# Remote RTX 3090 evidence

These measurements were collected on a ten-GPU RTX 3090 node from commit
`13b7164` with the prefix-wiring fix.  They are matched random-weight
microbenchmarks, not pretrained-LM quality results.

- `audit_8k_32k.json`: QCC vs fused Full-KV at 8K/16K/32K, plus 1K/2K logit
  cosine.  The stable 16K and 32K rows reach 2.22x and 5.17x TPOT speedup;
  TTFT speedups are 1.06x and 2.21x.  The 8K row includes a cold Triton
  compilation outlier and must not be used as a steady-state claim.
- `scaling_prefill.json`: QCC/full prefill speed ratios of 0.56x, 0.89x,
  1.62x, and 2.91x at 1K/2K/4K/8K; log-log slopes 0.758 vs 1.554.
- `scaling_decode.json`: decode ratios of 0.71x, 0.70x, 0.70x, and 0.89x;
  slopes 1.026 vs 1.124.  The Python/reference decode path measured here is a
  reference implementation, not a 1M TPOT speedup proof.

The files preserve failed and short-context evidence alongside the positive
rows.

`scaling_decode_fused.txt` is a follow-up after adding a cached one-GEMM
Q/K/V/gate projection and a one-launch Triton local decode reduction.  It
measured `0.72×/0.68×/0.69×/0.91×` QCC-vs-Full-KV ratios at 1K/2K/4K/8K
(2-step averages).  The modest change confirms that Python/module dispatch
and the full model shell, rather than only local attention, dominate this
reference implementation; it is retained as a negative systems ablation.

`retrain_20260902/summary.json` records an eight-seed retraining sweep after
the prefix-wiring fix.  The strict 128K/1M all-value stress set yielded
`18/464 = 3.88%` aggregate accuracy (best seed `4/58 = 6.90%`).  The retrieval
failure is therefore not a single unlucky initialization, and the 1M all-value
target remains unmet.

`temp_sweep_20260902/summary.json` tests landmark routing temperatures 1, 4,
and 16 on the same strict stress set.  Accuracies were `6.90%`, `3.45%`, and
`3.45%`, respectively.  Sharpening the route therefore does not repair the
long-tail failure; raw training/evaluation logs are kept beside the summary.
