# Triton local-window kernel evidence

These records were produced on an RTX 3090 (CUDA, GPU 1) on 2026-09-02, from a
checkout separate from the measurement host.  The one-launch Triton
sliding-window kernel passed the CUDA parity test against the unfolded
reference.

The matched 131,072-token audit in `audit_131k_block1024.json` used the same
random model weights for QCC and Full-KV.  It measured QCC prefill `1.9893 s`
versus Full-KV `5.1769 s` (`2.6023x` TTFT) and QCC TPOT `1.9457 ms` versus
`43.1968 ms` (`22.2009x` TPOT).  The exact-128,000-token warm audit in
`audit_128k_block1024_warm.json` measured `2.0948 s` versus `4.7250 s`
(`2.2555x` TTFT) and `1.9228 ms` versus `42.2696 ms` (`21.9834x` TPOT).
Logit cosine at 1,024 tokens was `0.9990` and the 128K state fraction was
`0.125781%` of the hypothetical Full-KV cache.  Timing varies with CUDA
kernel-cache warmup; both raw runs are retained.

This artifact covers kernel timing only: it is not evidence for the 1M,
pretrained-LM or official RULER/LongBench/PG-19 requirements.  The block-size
logs are QCC-only timing probes except for the matched audit explicitly
identified above.
