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

The corpus-dependence is measured directly by destroying the long-range structure
of one natural prompt in three stages — paragraph shuffle, sentence shuffle,
token shuffle — each with its own matched Full-KV arm
(`artifacts/prediction-density.json`, Llama-3.2-1B, 32K prompt, 256-token suffix,
1.6 MB of source text):

| level | long-range repeat fraction | NLL ratio at 4,096 slots | at 8,192 |
|---|---:|---:|---:|
| natural | 0.085 | 1.215 | 1.164 |
| paragraph shuffle | 0.096 | 1.358 | 1.285 |
| sentence shuffle | 0.123 | 1.321 | 1.227 |
| token shuffle | 0.000 | **1.012** | **1.005** |

The correlation between the repeat proxy and the measured degradation is
`r=0.94` (`rho=0.80`) at 4,096 slots and `r=0.91` at 8,192; the clean end of the
axis is the token shuffle, where a prompt with no long-range repeat left is
predicted within 1.2% of Full-KV. The middle of the table is the honest caveat:
shuffling whole paragraphs or sentences *moves* repeated n-grams beyond the near
window instead of deleting them, so this proxy rates a block shuffle as denser
than the natural order it came from, and the two block levels swap order with
respect to the degradation. The axis is therefore read as *structure destroyed
implies the gap collapses* — plus *natural text degrades least* — not as a
calibrated ordering of the shuffled levels.

## What sets the policy's item capacity

The two query-driven channels — the observation-window scores and the lexical
anchors — both read the **last `observation_window` tokens** of the prompt. That
single design choice fixes how many items a request can be served with, and it is
measurable without a model at all (`benchmarks/anchor_window_coverage.py`, Qwen2.5
tokenizer, 32K prompts, three seeds; item names seen inside the window):

| items asked for (`k`) | `obs=64` | `obs=128` | `obs=256` | `obs=512` | `obs=1024` |
|---|---:|---:|---:|---:|---:|
| 8 | 6.0 | 8.0 | 8.0 | 8.0 | 8.0 |
| 16 | 6.0 | 14.3 | 16.0 | 16.0 | 16.0 |
| 32 | 6.0 | 14.3 | 30.7 | 32.0 | 32.0 |
| 64 | 6.3 | 14.3 | 30.7 | 62.3 | 64.0 |
| 128 | 6.0 | 14.0 | 30.0 | 61.7 | 126.0 |

The visible count is linear in the window — about `(obs - 20) / 8.2`, i.e. eight
tokens per named item after a twenty-token question prefix — so the shipped default
(`observation_window=64`) puts roughly six named items in front of the anchor
channel. That is a limit of the *anchors*, not of the policy: the
observation-window scores still contribute, and on the model the two channels turn
out to be complementary. Qwen2.5-3B, 32K, `lex_cap=2048`, 4,096 slots, three seeds
(Full-KV arm of the same records: 0.958 at `k=16`, 0.781 at `k=32`):

| `k` | | `obs=64` | `obs=128` | `obs=256` | `obs=512` |
|---|---:|---:|---:|---:|---:|
| 16 | coverage | 0.789 | 0.956 | 1.000 | 1.000 |
| 16 | recall | 0.875 | 0.938 | 0.917 | 0.896 |
| 32 | coverage | 0.599 | – | 0.973 | 1.000 |
| 32 | recall | 0.396 | – | **0.833** | 0.698 |

Coverage rises monotonically with the window and stops exactly where the tokenizer
table says it will; recall tracks it while coverage is the binding constraint
(`k=32`: 0.396 to 0.833 as coverage goes 0.599 to 0.973 — *above* the Full-KV arm's
0.781, the same cleaner-context effect seen on narrativeqa), but once coverage
reaches 1.0 a wider window buys nothing and slightly costs, because it also changes
which filler the score channel keeps and how the anchors compete. So the design rule
is to size the window to the query, not to maximise it.

The second limit is the anchor budget: with `lex_cap=512` the expanded spans cover
9.7 of the 16 named items at `obs=128`, and `lex_cap=2048` covers 14.3 — the
window's own ceiling, not the cap's.

Both limits move quality, and the second one has a knee where the arithmetic says
it should. `lexical_anchors` expands each hit by
`lex_context_left + lex_context_right + 1 = 49` tokens, so 16 named items predict
about 784 anchor tokens. Qwen2.5-3B, 32K, `k=16`, three seeds, mean per-item recall
against the Full-KV arm of the same records (0.958 recall / 0.667 exact-set):

| arm | kept slots | measured coverage | items covered | recall | exact-set |
|---|---:|---:|---:|---:|---:|
| `budget 2048`, `lex_cap 512` (shipped) | 2,560 | – | – | 0.729 | 0.000 |
| `budget 4096`, `lex_cap 512` | 4,608 | 0.847 | 9.3 / 16 | 0.812 | 0.000 |
| `budget 8192`, `lex_cap 512` | 8,704 | – | – | 0.896 | 0.333 |
| `budget 4096`, `lex_cap 768` | 4,864 | 0.949 | 14.0 / 16 | 0.917 | 0.333 |
| `budget 4096`, `lex_cap 1024` | 5,120 | 0.956 | 14.3 / 16 | 0.938 | 0.333 |
| `budget 4096`, `lex_cap 3072` | 7,168 | 0.964 | 14.3 / 16 | 0.938 | 0.333 |
| `budget 2048`, `lex_cap 2048` | 4,096 | – | – | **0.958** | 0.333 |
| `budget 4096`, `lex_cap 4096` | 8,192 | – | – | **0.958** | **0.667** |

`required_coverage` is measured against the slots the selection actually kept —
every token of every item statement, over every `(layer, head)` index set — rather
than inferred from a width comparison. Two things follow. First, the knee is where
the rule puts it: between `lex_cap=512` (9.3 items covered) and `lex_cap=768` (14.0),
with the predicted requirement at 784. Second, **accuracy is a saturating function
of measured coverage, not a cliff**: exact-set match looks like a cliff only because
that column demands every item at once.

## Retention is not the same as being read

The sharpest evidence is a control: put the whole required set immediately before
the question, i.e. inside the recency channel of any budget that can hold the
statement block, and retention is guaranteed by construction. Then the bounded arm
and the Full-KV arm *of that same prompt* agree exactly, at every tested budget, on
both checkpoints:

| model | `k` | placement | Full-KV recall / exact-set | bounded (2,048 slots + 512 anchors) |
|---|---:|---|---|---|
| Qwen2.5-3B | 16 | scattered | 0.958 / 0.667 | 0.729 / 0.000 |
| Qwen2.5-3B | 16 | recency | 0.875 / 0.333 | **0.875 / 0.333** |
| Qwen2.5-3B | 32 | scattered | 0.781 / 0.000 | 0.188 / 0.000 |
| Qwen2.5-3B | 32 | recency | 0.792 / 0.333 | 0.760-0.844 / 0.000-0.333 |
| Llama-3.1-8B-4bit | 8 | scattered | 1.000 / 1.000 | 0.625 / 0.333 |
| Llama-3.1-8B-4bit | 8 | recency | 1.000 / 1.000 | **1.000 / 1.000** |
| Llama-3.1-8B-4bit | 16 | scattered | 0.708 / 0.000 | 0.375 / 0.000 |
| Llama-3.1-8B-4bit | 16 | recency | 0.792 / 0.667 | **0.792 / 0.667** |

2,560 retained slots equal a 32,769-slot cache on every recency row. The 8B-4bit
checkpoint is the one that produced the original collapse, and its gap disappears
entirely — including at `k=8`, where raising `lex_cap` from 512 to 2,048 changed
nothing (0.625 recall in both arms).

Presence in the support is necessary but not sufficient, and that same checkpoint
measures the difference. At `k=8` with the shipped configuration its item statements
are *all* retained: measured coverage 1.000, i.e. every token of every statement
survives in every `(layer, head)` index set, and the anchor census reaches 8 of 8
sites — yet its per-item recall is 0.625 against a Full-KV arm that answers all
eight. (Qwen2.5-3B is the control on the other side: the same 8-of-8 coverage at
`k=8` and 1.000 recall, so presence *is* sufficient there.) So what a statement
needs is not only to be kept but to keep the query's attention; the recency
placement is what puts it where the attention is, and that is why the scatter and
recency rows above differ although both retain the answer.

The two conditions are separately measurable, which is what makes the mechanism
checkable rather than anecdotal: `required_coverage` and the anchor census measure
the first, and the share of the final query's attention that lands on the required
spans (`required_mass`) measures the second. On that 8B-4bit `k=8` record set, with
the same budget, the same `lex_cap` and the same retained content:

| arm | coverage | attention share on the statements | recall | exact-set |
|---|---:|---:|---:|---:|
| scattered | 1.000 | 0.029 (0.028-0.032) | 0.625 | 0.333 |
| recency | 1.000 | **0.063** (0.063-0.065) | **1.000** | **1.000** |

The arm whose statements sit next to the question takes 6.3% of the final query's
attention instead of 2.9%, and answers all eight instead of five, although both arms
keep every statement. Two caveats belong with that number: the share is computed at
the final prompt token and averaged over `(layer, kv-head)`, so it is an *arm-level*
statistic — the within-arm spread is small (0.028-0.032, 0.063-0.065) while recall
varies by seed inside an arm — and it is not comparable across checkpoints, since
the number of keys and the sharpness of the attention differ.

## Falsifiable predictions, and where they stand

1. **Required slots do not grow with `L`.** Measured: decode state is fixed by
   configuration (4,608 slots, 144 MiB at 8K *and* 256K, 1.00x growth against 57x
   for Full-KV), and the `required_budget` derived per record set is flat in length
   (`artifacts/bounded-decode-frontier-state-growth*.json`).
2. **What grows instead is the query.** Measured here: the item capacity of the
   policy is `min(names inside the observation window, item spans the anchor budget
   holds)`, both constants of the configuration. A request whose question names more
   items than that is served with a partly filled support, and per-item recall then
   falls smoothly with the measured coverage; the all-or-nothing metric converts
   that slope into a cliff. The refuted form of this prediction — "the retained
   *width* is what binds" — is refuted by the same table: 8,704 slots do worse than
   4,096 slots allocated differently.
3. **The `L`-dependent residue is long-range dependency density.** Measured on four
   destruction levels of one natural prompt: `r=0.94` / `r=0.91` between the
   far-repeat proxy and the NLL ratio at 4,096 / 8,192 slots, with the fully
   shuffled prompt at 1.012x Full-KV and natural text at 1.215x (table above, and
   the stated caveat that the proxy is not monotone between natural order and a
   block shuffle).

The three together say something narrower and more checkable than "fixed state is
enough": the state a request needs is `O(1)` in context length and set by its query,
and the tasks that break the policy are the ones whose query outruns the policy's
two constants — not the ones whose context is long.

## What the mechanism establishes

The claim is no longer "a cache policy matches Full-KV on benchmarks". It is a
statement about where long-context state is *needed*: the retained set is the sum
of a concentrated retrieval component and a recent component, both `O(1)` in `L`;
what a request does need is one retained span per item its question names, and both
the window the policy reads the question through and the anchor budget that holds
those spans are constants of the configuration. The tasks that break the policy are
therefore the ones whose *query* outruns those constants — not the ones whose
context is long. That is a claim other groups can test, extend or contradict, and
the two instruments behind it ship with the repository:
`benchmarks/anchor_window_coverage.py` (what the policy can see of a query) and
`benchmarks/benchmark_required_set.py` with `benchmarks/analyze_required_set.py`
(what it retained, against what the answer needed).

## Related pages

* [`REPORT.md`](REPORT.md) §3.10 and §3.22 for the raw curves, §3.21 and §3.25-3.26
  for how much of the quality the anchors carry, §3.27 for this capacity analysis,
  and §4 for what is not claimed.
* [`CLAIMS.md`](CLAIMS.md) for the artifact behind every number above.
* [`REPRODUCING.md`](REPRODUCING.md#where-the-capacity-of-the-policy-comes-from)
  for the commands that regenerate the two capacity instruments.
