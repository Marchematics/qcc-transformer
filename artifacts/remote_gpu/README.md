# Remote RTX 3090 evidence

These measurements were collected on `tsinghua3090-X11DPG-OT` (ten RTX 3090
cards) from commit `13b7164`/the subsequent wiring fix.  They are matched
random-weight microbenchmarks, not pretrained-LM quality results.

- `audit_8k_32k.json`: QCC vs fused Full-KV at 8K/16K/32K, plus 1K/2K logit
  cosine.  The stable 16K and 32K rows reach 2.22x and 5.17x TPOT speedup;
  TTFT speedups are 1.06x and 2.21x.  The 8K row includes a cold Triton
  compilation outlier and must not be used as a steady-state claim.
- `scaling_prefill.json`: QCC/full prefill speed ratios of 0.56x, 0.89x,
  1.62x, and 2.91x at 1K/2K/4K/8K; log-log slopes 0.758 vs 1.554.
- `scaling_decode.json`: decode ratios of 0.71x, 0.70x, 0.70x, and 0.89x;
  slopes 1.026 vs 1.124.  This shows the current Python/reference decode
  path is not yet a 1M TPOT speedup proof.

The files intentionally preserve failed/short-context evidence rather than
turning it into a universal acceleration claim.
