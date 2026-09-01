# Novelty-check trace (2026-09-01, run01)

## Proposed claims searched

1. Exact recent local softmax window plus bounded historical memory.
2. Historical KV replaced by per-head, per-code numerator/denominator states.
3. Learned query-code routing, described as a compiled/discrete code response.
4. Multiple exponential decay rates and local/archive gated mixing.

## Sources and search terms

- arXiv API queries (2024-2026): `Infini-attention`, `Trellis compress key-value memory`,
  `Key-Value Means`, `Gated Slot Attention`, `Hippocampus linear attention`, `GLIDE`,
  `Kimi Delta Attention`, `Gated DeltaNet`, `Blurry Window Attention`, `hybrid attention`.
- Full-text PDF inspection for arXiv:2404.07143, 2512.23852, 2605.09877,
  2607.02303, 2607.24788, 2605.24930, 2409.07146.

## Closest papers found

- Infini-attention, arXiv:2404.07143 (2024): local causal attention + recurrent
  compressive associative matrix and normalizer; linear read/write; gated mixture.
- ABC, arXiv:2110.02488 (2021), and GSA, arXiv:2409.07146 (2024): bounded softmax
  memory slots with softmax routing and gated/forgetful recurrence.
- Trellis, arXiv:2512.23852 (COLM 2025): fixed-size KV memory, two-pass nonlinear
  recurrent compression and learned forgetting.
- Key-Value Means, arXiv:2605.09877 (2026): exact block sliding window plus compressed
  KV state and single softmax over state + window.
- HOLA, arXiv:2607.02303 (2026): bounded exact KV cache combined with recurrent
  compressive state.
- GLIDE, arXiv:2607.24788 (2026): layerwise sliding-window softmax + linear recurrence.
- Blurry Window Attention, arXiv:2606.09862 (2026): ABC/SSM-inspired frequency-window
  history and explicit relation to GSA/SWA.
- Kimi Linear/KDA, arXiv:2510.26692 (2025), and Gated DeltaNet-2, arXiv:2605.22791
  (2026): fixed recurrent KV replacement with fine-grained/channel-wise decay and
  delta updates.

## Reviewer route

Cross-model reviewer requested via Codex gpt-5.5 xhigh; raw response to be appended.

## Reviewer outcome

Assessment: proceed with caution, method novelty estimated low (3/10). The
closest prior work already covers local attention plus bounded recurrent memory,
softmax numerator/normalizer state, multi-scale decay, and gated hybrid output.
The narrow defensible delta is a fixed learned landmark codebook per KV head
with multi-timescale response statistics; the implementation must be compared
against Infini-attention, ABC/GSA, Trellis, KVM, HOLA, GLIDE, and KDA. The
reviewer also noted that the current code is soft-routed rather than discrete,
and that its per-scale normalization is an approximation to global softmax.
