# Novelty check trace — 2026-09-02 run03

## Proposed method checked

QCC Transformer: exact bounded local KV ring plus a constant-size recurrent
archive. The archive uses learned per-head codebook slots, normalized
numerator/denominator response statistics at multiple decay rates, optional
lexical raw-token landmark addressing, persistent prefix landmarks with
successor-pair binding, and fused/Triton serving paths.

## Search boundary

The review covered recent 2024–2026 arXiv work and the repository's existing
novelty boundary. Searches targeted constant-memory recurrent attention,
landmark/content-routed KV compression, successor associative memory, sparse
KV retrieval, and multi-timescale decay.

## Closest prior work and overlap

- Infini-attention (2024): masked local attention plus bounded recurrent
  compressive memory, linear numerator/normalizer recurrence, and gated mixing.
- ABC (2021), GSA (2024), GLA (2023/24): bounded learned slots and recurrent
  gated/softmax-routed states.
- Landmark Attention (2023): content-addressed landmark tokens for long-range
  block retrieval.
- SnapKV, PyramidKV, Quest, RetrievalAttention, DuoAttention (2024):
  query-aware or head-aware KV retention/retrieval.
- Kimi Linear/KDA (2025), Trellis (2025), KVM/HOLA/GLIDE (2026): bounded
  recurrent or hybrid local+compressed KV mechanisms.
- PCAF (2026): causal successor records with bounded hash-bucket retrieval;
  especially close to the successor-pair diagnostic.

## Decision

The overall local+recurrent, landmark, successor, and multi-timescale ideas
are not novel in isolation. The defensible delta is the particular
fixed-codebook, per-(KV head, code, decay) normalized response-statistics
parameterization and its fused implementation. This is a narrow novelty
hypothesis requiring matched empirical comparisons; no first-of-kind or
universal speedup claim is supported.

## Raw reviewer response

The independent reviewer reached the same conclusion and listed Infini,
Landmark Attention, KDA, PCAF, and related systems as direct or severe
overlaps. It recommended comparisons against those methods and Full-KV on
real pretrained RULER/LongBench data.
