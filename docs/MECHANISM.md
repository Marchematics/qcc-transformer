# Why a fixed number of slots can be enough — and when it must fail

This page states the mechanism behind the retention law, the measurements that
support it, and the predictions that would falsify it. The central claim is a
proposition rather than a leaderboard row:

> Long-context inference does not inherently require persistent state
> proportional to context length.

## The invariant

Decode is exact softmax attention over the retained support `S`:

```
out = sum_{i in S} softmax(q·k_i) v_i          # exact over S
full = sum_{i in 1..L} softmax(q·k_i) v_i      # the reference
```

Nothing about the arithmetic is approximate: every retained key keeps its
original rotary phase and value, so the only decision the policy makes is **which
positions are in `S`**. Quality is therefore a property of the support, not of
the attention kernel.

## The support has two parts, neither proportional to L

Writing the full attention distribution as `p` over `L` keys:

* **Concentrated mass.** For retrieval-style queries the mass concentrates on a
  handful of positions — the ones carrying the answer. Keeping the top-`k` of `p`
  makes the truncated softmax equal to the full one up to the ties, and `k`
  depends on how many distractor keys compete with the answer, not on `L`. On the
  80-record RULER split this saturates at 4,608 slots (1,536 slots already keep
  0.987 of matched quality).
* **Recency.** For next-token prediction the mass is diffuse, but the diffuse
  part is dominated by nearby tokens, which a sliding window captures in `O(1)`
  state. The residual error is the *long-range diffuse* mass: mass that is far
  away **and** individually small.

That is the whole argument for `O(1)` history: the far tail of the attention
distribution is simultaneously low-mass and locally redundant, so dropping it
costs a bounded amount, and the part that is not redundant is concentrated enough
to be kept exactly. Sinks are retained because the first tokens absorb attention
disproportionately in every transformer.

## What the measurements say about the residual

Language-modelling NLL for the 256-token suffix of a 32K prompt, Llama-3.2-1B,
no anchors, last-query scoring (`artifacts/bounded-decode-frontier-lm-nll-32k-*`
and `artifacts/lm_nll_pinned_b*`):

| corpus | Full-KV NLL | 1,024 slots | 2,048 | 4,096 | 8,192 | 16,384 |
|---|---:|---:|---:|---:|---:|---:|
| synthetic, self-similar | 3.188 | – | 1.137x | 1.091x | **0.995x** | – |
| natural text | 1.348 | 1.148x | 1.064x | 1.084x | 1.082x | 1.061x |

Read together, the NLL ratio is `1.06-1.15x` below ~8K slots and lands between
`0.99x` and `1.08x` at 8-16K. On the self-similar corpus the bounded support
*beats* Full-KV at 8K slots — a cleaner context predicts the suffix slightly
better than a 32K window padded with filler, the same effect seen on narrativeqa
(+10.7%) and on the anchors-only configuration. On natural text a 6-8% gap
persists even at half the context, so no claim is made that perplexity is
preserved: what the evidence supports is that the gap is a function of how much
genuine long-range, non-redundant mass the corpus has, and that it is small and
slowly varying with budget rather than exploding.

## Falsifiable predictions

1. **Required slots scale with the number of competing items, not with `L`.**
   Weakly supported, and the binding constraint turned out not to be the slot
   count. Two sweeps at 32K (`artifacts/prediction-distractors.json`,
   `artifacts/prediction-needles.json`, `artifacts/prediction-needles-llama8b.json`):

   * *Competition axis.* With one asked key plus 4 competing keys, a 512-slot
     budget already matches Full-KV (1.000). At 33 competing keys Full-KV is still
     1.000 while the bounded arm needs **1,024** slots; at 129 and 513 competing
     keys the model itself drops to 0.667, so the axis cannot be extended on this
     checkpoint.
   * *Required-set axis.* Asking for all `k` needle values, an 8B checkpoint
     answers k=8 perfectly with Full-KV (1.000) while the bounded arm reaches only
     0.333 - **at every budget from 1,024 to 8,192 slots**. The needle statements
     occupy about 136 tokens at k=8, so the retained width covers the required set
     by a wide margin and coverage is not the constraint. The limit is the anchor
     policy: with `lex_cap=512` the expanded anchor sets for eight needles compete
     for the same cap, so needles are dropped before the slot budget is reached.

   The operational consequence is a stated design rule to test next: scale the
   anchor budget with the number of items the question names (`lex_cap` proportional
   to `k`), not with the context length. Past k=16 both checkpoints fail the task
   with exact attention as well, so the probe is bounded by model capability before
   it is bounded by the cache.

2. **Failure is a cliff, not a slope, once the required set exceeds the budget.**
   Not measured, and the simple form of the prediction is now in question. The
   conditional analysis (accuracy on rows whose retained width covers the required
   set, against rows where it does not) has 60 covered rows and **0** not-covered
   rows in the needles sweep, because the required set is small compared with any
   tested budget; the distractor sweep does produce 18 not-covered rows, with
   accuracy 0.667 against 0.881 when covered, but in exactly those cells the
   model's own Full-KV arm is already at 0.667. The honest statement is therefore
   that the required-set framework explains the *small*-k regime and that the
   transition is governed by anchor capacity rather than by slot capacity.

## What the mechanism establishes

If predictions 1-3 hold, the claim is no longer "a cache policy matches Full-KV
on benchmarks" but a statement about where long-context state is *needed*: the
retained set is the sum of a concentrated retrieval component and a recent
component, both `O(1)` in `L`, and the tasks that break the policy are exactly
those whose answers are spread across more positions than the budget allows.
That is a claim other groups can test, extend or contradict.

## Related pages

* [`REPORT.md`](REPORT.md) §3.10 and §3.22 for the raw curves, §3.21 and §3.25-3.26
  for how much of the quality the anchors carry, and §4 for what is not claimed.
* [`CLAIMS.md`](CLAIMS.md) for the artifact behind every number above.
