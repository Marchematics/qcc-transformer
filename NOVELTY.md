# Novelty Boundary

QCC-Transformer is a research hypothesis, not a claim that bounded-memory
attention has never been studied. A review of recent arXiv work found direct
overlap with the following families:

- [Infini-attention (2024)](https://arxiv.org/abs/2404.07143): local attention plus a bounded recurrent memory and gated mixing.
- [ABC (2021)](https://arxiv.org/abs/2110.02488) and [GSA (2024)](https://arxiv.org/abs/2409.07146): softmax-routed bounded memory slots.
- [Trellis (COLM 2025)](https://arxiv.org/abs/2512.23852): fixed-size recurrent KV compression.
- [Key-Value Means (2026)](https://arxiv.org/abs/2605.09877): block sliding window plus compressed KV state.
- [HOLA (2026)](https://arxiv.org/abs/2607.02303) and [GLIDE (2026)](https://arxiv.org/abs/2607.24788): hybrid exact-window/recurrent long-context attention.
- [Kimi Linear/KDA (2025)](https://arxiv.org/abs/2510.26692): recurrent KV replacement with fine-grained decay.

The narrow technical delta tested by this repository is a fixed learned
landmark codebook per KV head. Each landmark maintains numerator and
denominator responses at several decay rates, and the current query performs
soft routing over landmarks. The optional sparse variant adds top-k routing
and lazy per-slot decay: an overcomplete bank is kept for capacity, while only
the selected slots are touched at inference. This is best described as a
**sparse multi-timescale learned-landmark memory**. It should not be described
as exact global softmax attention, since each landmark/scale is normalized
before the responses are mixed.

The current assessment is **proceed with caution**. A publishable contribution
would need a matched recall/latency/memory Pareto comparison against Infini,
ABC/GSA, KVM, KDA, and a strong full-KV baseline, plus fused GPU kernels. The
lazy sparse update is an implementation hypothesis, not evidence of a new
algorithmic class or a universal speedup. The repository's [review trace](.aris/traces/novelty-check/2026-09-01_run01/trace.md)
records the search boundary.
