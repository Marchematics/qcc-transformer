# Novelty Check Trace: Sparse QCC Serving

Date: 2026-09-01

## Claims searched

1. Top-k routing over a learned landmark/codebook archive.
2. Lazy per-slot exponential decay using logical timestamps.
3. Reusing a remote-memory response across consecutive decode queries.

## Search boundary

Queried the arXiv API for `lazy decay attention`, `top-k recurrent memory
transformer`, and related bounded-memory terms. The returned set included
Gated Sparse Attention (arXiv:2601.15305), StreamIndex (arXiv:2605.02568),
Recurrent Autoregressive Diffusion (arXiv:2511.12940), Recurrent Memory
Transformer (arXiv:2207.06881), and the prior bounded-memory families listed in
`NOVELTY.md`. Semantic Scholar rate-limited the unauthenticated request.

## Assessment

Top-k selection and lazy decay are established ideas in adjacent sparse and
recurrent-memory systems, so neither is claimed as independently novel. The
narrow implementation delta remains a fixed per-KV-head landmark response bank
with multi-timescale numerator/denominator states, optional top-k inference,
and logical timestamps that avoid touching inactive slots. Temporal response
reuse is exposed as an approximation knob and is not presented as a new
attention primitive.

The delegated cross-model reviewer could not run because the configured
endpoint returned HTTP 403 (insufficient balance). This trace therefore keeps
the conservative `proceed with caution` assessment from `run01` rather than
upgrading the novelty claim.
