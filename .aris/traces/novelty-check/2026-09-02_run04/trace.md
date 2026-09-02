# Novelty check trace — 2026-09-02 run04

## Proposed method checked

QCC Transformer: exact bounded local KV ring plus a constant-size recurrent
archive. The archive stores per-KV-head learned prototype/code responses,
independently normalized numerator/denominator statistics at multiple fixed
exponential decay rates, and query-dependent code routing. Optional variants
add raw-token lexical landmarks, immutable prefix landmarks with successor
binding, and Triton/fused serving kernels.

## Search boundary and sources

Direct arXiv abstract/HTML retrieval (2023–2026) targeted bounded-memory
attention, landmark/chunk routing, successor associative memory, KV-cache
compression and linear-attention serving. Independent cross-check agent was
asked to test overlap claim-by-claim.

## Papers and evidence

- Infini-attention, Munkhdalai, Faruqui & Gopal (2024),
  https://arxiv.org/abs/2404.07143 — local masked attention plus bounded
  associative recurrent memory; linear numerator/normalizer update; learned
  local/remote gate; 1M passkey retrieval. Strongest high-level overlap.
- ABC, Peng et al. (2021), https://arxiv.org/abs/2110.02488 — unified
  bounded-memory slots with learned contextual control and softmax routing.
  GSA, Zhang et al. (2024), https://arxiv.org/abs/2409.07146 — two-layer GLA
  linked by softmax, compact recurrent slots and adaptive forgetting.
- Landmark Attention, Mohtashami & Jaggi (2023),
  https://arxiv.org/abs/2305.16300 — a landmark token represents each block;
  attention scores on landmarks select blocks for random-access long context.
  This directly weakens any generic “landmark addressing” claim.
- PCAF, Ahmed (2026), https://arxiv.org/abs/2606.10435 — causal successor
  records `(a_i,k_i,v_i)`, with `v_i=x_{i+1}`, hash/semantic bounded top-k
  retrieval, sparse successor distribution, learned gate and fused kernels.
  This directly overlaps QCC prefix-pair marker→successor value.
- HiLS Attention, Hu et al. (2026), https://arxiv.org/abs/2607.02980 —
  learned chunk landmark/surrogate keys, hierarchical softmax/top-k routing,
  sparse kernels and ultra-long (reported 4M) extrapolation. Weakens generic
  learned-landmark/sparse-routing and 4M claims.
- MARCH, Zhang et al. (2026), https://arxiv.org/abs/2608.12435 — periodic
  recurrent-state checkpoints as content-conditioned anchors, attention over
  historical anchors. Overlaps persistent/content-routed landmarks, though
  its anchor bank grows with context while QCC codebook is fixed.
- HOLA, Cui (2026), https://arxiv.org/abs/2607.02303 — GDN recurrent state
  plus bounded exact KV cache selected by residual magnitude and sharpened
  softmax read. Makes local+recurrent+exact-memory hybrid non-novel.
- Kimi Linear/KDA, Kimi Team (2025), https://arxiv.org/abs/2510.26692 —
  fine-grained per-channel decay/delta recurrent state, hybrid local/full
  attention, 1M 6x decoding. Decay and hybrid speed pieces are established.
- Trellis, Karami et al. (2025), https://arxiv.org/abs/2512.23852 — fixed-size
  KV memory with two-pass recurrent online compression, forget gate and
  test-time updates. Direct fixed recurrent KV-compression overlap.
- KVM, Goldstein et al. (2026), https://arxiv.org/abs/2605.09877 — fixed or
  expandable block-recurrent compressed KV memory, chunkwise training/prefill.
- Titans, Behrouz, Zhong & Mirrokni (2024/25),
  https://arxiv.org/abs/2501.00663 — neural long-term memory, adaptive decay,
  persistent memory and local-attention hybrids; >2M-context retrieval.
- GLA/FlashLinearAttention, Yang et al. (2023),
  https://arxiv.org/abs/2312.06635; Griffin, De et al. (2024),
  https://arxiv.org/abs/2402.19427 — gated recurrent memory + local attention
  and hardware-efficient/chunkwise kernels.
- RetrievalAttention (2024), https://arxiv.org/abs/2409.10516; Quest (2024),
  https://arxiv.org/abs/2406.10774; SnapKV (2024),
  https://arxiv.org/abs/2404.14469; PyramidKV (2024),
  https://arxiv.org/abs/2406.02069; PyramidInfer (2024),
  https://arxiv.org/abs/2405.12532; DuoAttention (2024),
  https://arxiv.org/abs/2410.10819; MagicPIG (2024),
  https://arxiv.org/abs/2410.16179 — query/head-aware sparse retrieval or
  retention and practical kernels; not codebook recurrence, but invalidate
  broad “constant/sparse KV gives speed” novelty.
- DASC (2026), https://arxiv.org/abs/2608.30386 — decay-aware retention
  horizons and compressed recurrent-state checkpoints for GDN/KDA serving;
  overlaps multi-timescale/decay-aware state compression.

## Claim-by-claim decision

1. Constant local ring + recurrent bounded archive: **LOW novelty**;
   Infini, ABC/GSA, GLA/Griffin, Titans, KDA, Trellis/KVM/HOLA already cover
   this design space.
2. Fixed learned per-head prototype/codebook with normalized numerator and
   denominator responses at multiple decay rates and query routing:
   **MEDIUM at best**. This is the only plausible narrow delta, but it is a
   specialization/combination of ABC/Infini/linear-attention response
   statistics; requires equations and ablations against those baselines.
3. Lexical/persistent/prefix landmarks: **LOW** as a generic claim;
   Landmark Attention, HiLS and MARCH cover content-addressed anchors.
4. Prefix marker→successor value binding: **LOW**; PCAF is direct prior art.
5. Fused QKV/gate/Triton kernels: **not algorithmic novelty**; engineering
   relative to FlashAttention, FlashLinearAttention, HiLS/PCAF kernels.

## Safe positioning

Use: “We study a sparse multi-timescale learned-prototype response memory:
for each KV head and prototype `c_m`, maintain independently normalized
exponential-response statistics and mix them by query-dependent routing,
while retaining an exact recent ring.” Avoid “first constant-memory
attention,” “novel landmark/successor memory,” “exact global attention,” or
universal/first 1M speedup. Report this as a narrow hypothesis and require
matched Full-KV, Infini, ABC/GSA, KDA/Trellis/KVM/HOLA, HiLS/PCAF and real
RULER/LongBench comparisons.

## Independent cross-check

The reviewer agreed PCAF is direct overlap for successor binding, HiLS/MARCH
for landmarks/anchors, HOLA for exact-ring + recurrent hybrid, and Infini for
the main local/recurrent normalized-memory mechanism. It recommended limiting
the claim to the fixed per-head multi-scale prototype response
parameterization and treating kernels as implementation evidence only.
