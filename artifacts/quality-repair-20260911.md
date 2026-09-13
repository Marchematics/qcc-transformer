# Quality repair, 2026-09-11

Source workspace: `/root/qcc/repo/qcc-transformer`, branch `main`.
Runtime copy: `/home/waas/qcc-quality-sep10`.

## Current user targets

These targets replace the earlier 98% quality target. Quality percentages
are provisionally interpreted as retention relative to matched Full-KV.

| Metric | Target | Current evidence |
|---|---|---|
| Aggregate task quality | ≥99% | Current first-five diagnostic: QCC 2/5, Full-KV 5/5; aggregate unproven |
| Worst task quality | ≥97% | Full task coverage missing |
| 1M retrieval | ≥99.5% | No validated 1M checkpoint/run |
| Historical state | O(1), bounded | Bounded archive, recent window, and optional prefix |
| State growth, 128K→1M | ≤1.25×, preferably ≈1× | End-to-end measurement missing |
| 128K TPOT speedup | ≥5× Full-KV | Matched serving measurement missing |
| 1M TPOT speedup | ≥5× Full-KV | Missing |
| Throughput speedup | ≥3× | Missing |
| Fixed-SLA concurrency increase | ≥8× | SLA and workload need specification |
| Trainable parameter fraction | ≤0.5%, preferably ≤0.2% | Regular adapter adds 6,390,784 / 3,821,079,552 = 0.16725%; backbone is frozen before patching in evaluation |
| Retrofit | Existing pretrained LM, no retraining | Current runs use the pretrained Phi-3.5 weights |

Current hardware is one A10G. The 128K-native Phi checkpoint establishes no
1M claim. Offloaded Full-KV here is a quality reference, not a measured
serving-performance baseline. All targets remain active; a local retrieval
improvement does not establish the complete objective.

## Confirmed implementation defects

The hybrid prefill path writes rotary keys into the exact bank. Its token
decode path previously wrote raw keys and queried the same bank with raw
queries when position-invariant recurrent addressing was enabled. Decode now
passes rotary keys/queries separately, as prefill already does. Recurrent
addressing remains unchanged. A deterministic chunk-versus-token comparison
failed before the fix and passes after it.

The non-Triton scan retained its final state as a view of the entire scan
block. Persistent storage therefore included all intermediate states. The
fix copies only the final numerator and denominator. At 32 heads, 16 codes,
four scales, dimension 96, and a 512-event block, one numerator held 384 MiB
instead of 0.75 MiB per layer. The storage regression fails before the fix
and passes afterwards. This fixes retained scan storage, not all temporary
prefill memory.

## Validation and remaining work

The two existing archive test files now have 51 passing tests; the two RULER
conversion tests also pass (53 total). Two original
accelerator failures were reproduced against the unmodified source and fixed:
the per-token archive fallback returned results without writing the supplied
output buffer, so attention consumed zeros; sparse legacy kernels were also
selected for global normalization, which they do not implement. The fallback
now writes the output buffer, and sparse kernels are selected only for their
supported normalization and query-correction configuration.

`regular-state-storage-fix.json` is a matched real Phi-3.5 first-record run
with bf16, SDPA, a 4096-token window, 16 codes, 512-token prefill chunks,
and a 64-token output budget. Full-KV scores 1; regular QCC scores 0 and
generates a refusal in 29 tokens. No generation exception occurred.
The implementation fixes do not establish recovered task quality.

`regular-window8192-state-fixed.json` repeats the first two records with an
8192-token window and otherwise the same settings. Full-KV scores 2/2,
QCC 0/2; both QCC responses exhaust 64 output tokens. No generation
exception occurred. Increasing this window alone did not recover retrieval.

Previous raw-address, dot-product/sigmoid-confidence, threshold, and capacity
experiments in the runtime copy all scored zero. Their speculative changes
were removed, including the automatic low-precision exact-bank reset and
the changed default capacity. Original result JSON files remain as evidence.
An unrotated first-layer key ranking does not prove which keys the full
model needs, and cannot establish a necessary exact-bank capacity.

No new test files, branches, or pull requests were created. One A10G was used.
Further work must establish retrieval quality on matched records before
making broader quality claims. The real-model results above used non-Triton
execution and are not evidence that these accelerator fixes recover retrieval.

## Prefix retention implementation

`attention_sink_size` is an optional, default-zero argument on the core
attention and HF retrofit. It retains a bounded prefix of rotary K/V and
normalizes it together with the recent window. Prefix entries are masked
until they leave that window, and never enter the recurrent archive. Reset
clears the prefix. Streaming and differentiable reference forward agree;
sink-enabled training currently uses the sequential reference path.

A regression also exposed stale fused projections after a bias-only update.
The projection cache now tracks bias versions as well as weight versions.

The RULER benchmark in this checkout incorporates the runtime copy's matched
offloaded baseline and record selection, and exposes `--attention-sink-size`
and `--archive-mix`. Reports include both settings. Answer recall is retained
separately; a response exhausting the output budget without EOS scores zero.
EOS at the final allowed token counts as normal completion.

The local-only control without prefix retention failed row 2. The experimental
32-token-prefix control retrieved the answer but exhausted 64 tokens, so its
old recall-based score of one is not a completed quality pass. The implemented
path (`prefix32-implemented-row2.json`) used prefix 32, window 4096, archive
mix zero, bf16 SDPA, and 128 output tokens. It reproduced the Full-KV answer
exactly and ended normally after 29 tokens. This establishes a local retrieval
improvement on one record; it does not establish long-range archive quality.

Use this main checkout for further runs. The runtime copy contains the earlier
repairs but does not contain the new prefix-retention implementation.

`prefix32-default-archive-first2.json` restores the default archive mixture
and evaluates rows 1–2 with the implemented prefix cache. Both Full-KV and
QCC score 2/2, with QCC stopping normally after 33 and 29 tokens. Both
answers occur within the recent window; this does not establish retrieval
from the compressed archive.

`prefix32-default-archive-rows3to5.json` evaluates three records whose answers
lie outside the recent window. Full-KV scores 3/3 and QCC 0/3. Row 3 returns
an incorrect number; rows 4–5 exhaust the 128-token budget. Together with
the first-two diagnostic this gives QCC 2/5 versus Full-KV 5/5, not a suite
aggregate. Prefix retention fixes the near-window failure but does not
recover arbitrary associations from the recurrent archive. This remains the
primary quality problem under the updated user targets.

## Teacher archive diagnosis

`benchmarks/diagnose_hf_archive.py` compares the final prompt query of the
unmodified Phi teacher with reconstructed recurrent statistics and exact-KV
references. `teacher-archive-row3.json` records five layers of RULER row 3.
The experiment uses real hidden states and no answer labels in selection.
For layer 23 the remote-mass-weighted relative squared error is 1.557 for
the recurrent archive, 2.051 for full-history top-1, and 0.046 for top-128.
Using 512 keys selected by 32 earlier prompt queries raises the error to
0.591 (cosine retention) or 0.659 (dot retention). Layer 0 has a diffuse
response and top-k selection can be worse than the recurrent archive.

These are isolated attention-output diagnostics, not end-to-end quality
scores. Full-history top-k and offline selection are not deployed bounded
serving algorithms. The results motivate testing retention and weighted
readout together; they do not justify a global switch to dot-product top-1
or prove the new quality targets achievable.

Parameter accounting used the actual CPU-loaded pretrained model and the
regular retrofit with 16 codes, window 4096, and prefix 32. Unique introduced
parameters are counted by object identity, excluding shared pretrained
projections: 6,390,784 versus 3,821,079,552 pretrained parameters (0.16725%).
Prefix retention adds no learned parameters. The benchmark now freezes the
backbone before patching and reports the actual trainable count and fraction.
No training or optimizer step was performed.

## Weighted bounded-KV readout

`SetAssociativeLandmarkBank.read_attention` now computes scaled-dot-product
softmax over the fixed retained table and returns both response and log
partition. Queries are tiled in blocks of 128, and no persistent state or
learned parameters are added. The optional `--exact-attention` experiment now
connects it to hybrid streaming and merges retained and local KV by their
softmax masses. In this experiment the recurrent response is not used in
the output; the recurrent state is still maintained. The teacher diagnostic uses the actual method for its
retained-table comparisons and checks agreement with a direct tensor read.

The existing associative test file passes all 10 tests, including CPU and
CUDA checks that merging local and retained-table outputs by partition mass
equals a single attention call over their union. Tests cover empty heads,
single-query reads, and the query-tile boundary.

`teacher-archive-row4.json` repeats the real-teacher diagnostic on another
failed remote record. Layer 23 has relative squared error 0.886 for the
recurrent archive, 0.024 for the full-history top-128 reference, 0.328 for
512 cosine-selected slots, and 0.867 for 512 dot-selected slots. Selection
uses earlier prompt queries and excludes the probe query. This again shows
that a global cosine-to-dot switch is insufficient. The retained-table
selection remains offline and has not demonstrated end-to-end task quality.
The verified task score remains 2/5 versus Full-KV 5/5.

The weighted readout integration passes the full-history equivalence check
when all evicted keys fit in the retained table. Chunking, single-token
decode, reference forward, and backward are covered. Snapshotting retained
K/V during differentiable reads prevents later admissions from invalidating
backward. There are 62 passing archive/associative/hybrid tests and 27
passing HF retrofit tests. The real weighted-readout generation experiment
is tracked separately from the earlier task scores.

`weighted-exact-prefix32-row3` has completed. Full-KV scores 1/1; the weighted
candidate scores 0/1 and exhausts 128 tokens. Its report also records
19,172,352 trainable parameters (0.50175% of the backbone), slightly above
the user's 0.5% ceiling. The regular adapter's 0.16725% figure must not be
applied to this larger hybrid configuration.

The weighted global-table mode now freezes its unused set-routing codebook:
neither global writes nor full-table weighted reads consult that parameter.
An actual CPU-loaded Phi retrofit recount gives 6,589,440 trainable parameters
(0.17245%) with the backbone frozen. Total added parameters remain 19,270,656;
freezing is not a storage reduction. The full-KV-equivalence test still passes.
The active generation process was constructed before this trainability-only
change, so its report will retain the older parameter count. Numerical
generation does not depend on the frozen routing codebook.

Two salience-index defects were then fixed: query event j means token
j + window_size, so key i is eligible only when j >= i; the HF side channel
now supplies event zero for its stream beginning at token window_size.
Chunk-local query streams use their own event offset. The new eviction-boundary
regression failed before the fix, and 41 hybrid/retrofit tests pass afterwards.

Completed run: `weighted-exact-eviction-index-row3` (former Python PID 76165,
tool session 57198). Output:
`artifacts/weighted-exact-eviction-index-row3.json`. It evaluates row 3, prefix 32, window 4096,
16 recurrent codes, 128 exact sets × 4 ways, quality-first retention,
weighted exact attention, bf16 SDPA, 512-token prefill chunks, and a
128-token generation budget. Full-KV returned the correct answer. The
Python launch suppresses the incompatible optional `kernels` import only
for this process (`sys.modules['kernels'] = None`); installed packages are
unchanged. Candidate prefill is slow due to per-token exact-bank admission.
Process-local hooks report every four completed prefill layers; they are
not installed in the source code. The prior weighted result used the old
salience indices and does not evaluate this correction.
All 32 layers completed and the process exited. Full-KV scores 1/1; the
candidate scores 0/1, returning a refusal in 21 tokens with normal EOS.
The event-index correction did not recover the missing answer. Weighted
readout and timing correctness are insufficient with this retention policy.
No quality-repair GPU job remains active from this run.

## Retention and background diagnostics

`teacher-archive-centered-row3.json` and `teacher-archive-centered-row4.json`
contain the complete tail-query, background-sampling, and centered-background
comparisons. Retention uses the last 32 prompt queries before the probe query;
the probe query and answer labels are excluded from selection. Each background
comparison keeps 384 selected keys plus 128 uniform keys from the complement,
weighted by complement size / sample size, with seeds 11, 29, and 47.

Tail selection alone improves some late layers but worsens layer 0. On row 3,
the layer-0 relative squared error of dot-based top-512 is 3.365. Adding the
weighted background sample reduces it to 0.016–0.025; using the exact background
value mean as a control variate reduces it further to 0.0097–0.0140. On row 4,
the corresponding centered-background error is 0.0068–0.0105. The centered
estimator uses the known population mean plus the sampled weighted-minus-
unweighted value difference, with the finite-sample covariance correction.
That mean could be maintained online with fixed-size sums and counts.

This does not eliminate selection and sampling error: layer 23's centered
dot-based error still ranges from 0.098 to 0.441 on row 3 and 0.087 to 0.372 on
row 4. The tail/background experiments are offline teacher-input diagnostics,
not an online reservoir implementation and not task-accuracy measurements.
They justify preserving diffuse background mass while investigating retention;
they do not establish the user quality or performance targets.

Three intermediate tail/background reports were removed only after checking
that every JSON field and value was exactly preserved in these two combined
reports. Original uniform-query diagnostics and generation results remain.

## Online background reservoir

`background_size` enables a fixed-capacity reservoir in the global score-based
bank. Every foreground rejection or demotion enters the background exactly
once. Background entries never re-enter the foreground. Algorithm-R reservoir
sampling, reset with seed 11 per request, therefore samples the foreground
complement without duplicate counting. Weighted reads add
`log(background population / valid sample count)` to background logits.
Persistent storage includes the reservoir tensors, count, and RNG state.

This implementation uses ordinary importance weighting; the centered estimator
from the offline diagnostics is not implemented in serving. With background
sampling enabled, foreground priority uses salience alone (diversity weight
zero), matching the diagnostic selection rather than adding a separate bonus.
The optional `quality_query_tail` selects a suffix of the prompt-query stream
using archive-event offsets. Defaults remain unchanged.

CPU and CUDA tests cover disjointness, rejection and demotion, exact Full-KV
agreement before the reservoir fills, background-only softmax mass after
overflow, fixed storage, reset, and chunk/token parity. The full run before the
last selection change passed 93 tests; the subsequent 55 associative/hybrid/HF
tests and the two tail-offset cases pass after the relevant changes.

Completed run: `weighted-background-tail32-row3` (former Python PID 89957,
tool session 91518). Output: `artifacts/weighted-background-tail32-row3.json`.
Configuration: row 3, bf16 SDPA, window 4096, prefix 32, 512-token prefill chunks,
384 foreground slots (96 sets × 4 ways), 128 background slots, tail 32 queries,
quality-first admission, weighted attention, and 128 output tokens. Total KV
slots remain 512, excluding the recent window and prefix. Full-KV answered
correctly; the candidate returned 3581221 instead of 3539476, stopped normally
after 29 tokens, and scored 0/1. The returned number does not occur in the
input record. The run reported a trainable fraction of 0.17245%. It has exited;
no quality-repair GPU job remains active from this run.
This is an end-to-end experiment of the combined retention/background method,
not an isolated single-factor comparison with the previous failed variant.

The benchmark now records `qcc_runtime_state` using backing-storage sizes,
deduplicating aliased views and including background tensors, counters, RNG
state, prefix/recent KV, and request scratch. Model parameters, registered
constant buffers, and fused projection-weight copies are excluded. It also
records per-request PyTorch peak CUDA allocation. The storage-alias regression
passes. This is QCC-owned state accounting, not whole-server memory accounting
and not a measured 128K-to-1M result. The active background run was launched
before this instrumentation, so its JSON will not contain these new fields.

## Passive retention trace

Completed run: `retention-trace-background-row3` (former Python PID 99753,
tool session 45961). Output: `artifacts/retention-trace-background-row3.json`.
It repeats the same 384+128 background/tail32 configuration and a fresh
matched Full-KV reference. The reference answered correctly. The candidate
reproduced exactly the prior erroneous prediction 3581221, including normal
termination after 29 tokens. All 32 layers and generation have completed.

`--trace-candidate` in the existing diagnostic records answer-neighbour keys
as they enter the bank. Labels determine only what is observed; callbacks
forward identical inputs to the original update and do not consume random
numbers or influence selection, attention, or generation. Foreground token
identities use insertion ages; background observations require exact matches
of both stored K and V. It reports head membership after prefill and after
generation. Only evicted-token bank entries are traced; prefix/window tokens
are not interpreted as lost bank entries.

The observer keeps small additional KV copies outside the model. Peak CUDA
allocation therefore includes diagnostic overhead, while QCC-owned state
accounting excludes those callback-owned copies. This run is not a speed or
whole-server memory benchmark. Hooks and method overrides are removed at exit.

The answer occupies input token positions 9644–9650. Of the 224 answer-token
and layer combinations, 60 have no matching retained K/V in any head after
prefill. None of the tracked answer-neighbour membership entries changes
between prefill and completion. The failure is therefore not explained by
decode-time replacement within this tracked region. Remaining matches alone
do not establish that the relevant teacher retrieval heads retain the needed
information; later-token values may also encode earlier answer information.

Measured QCC-owned state after generation is 5,580,542,464 bytes, including
request scratch and RNG state, excluding model parameters and observer KV
copies. Peak CUDA allocation is 15,779,754,496 bytes and includes observer
overhead. These measurements cover this 16K record only and do not establish
the required 128K-to-1M growth bound or serving concurrency.

## Teacher retrieval-head alignment

`teacher-head-alignment-row3.json` observes the native teacher's actual rotary
Q/K during generation without changing the SDPA output. The reference again
generates the correct answer. At output indices 18–24 (the seven answer digits),
only 31.173% of teacher attention mass directed to the answer-token KV is
covered by candidate prefill membership. This is a descriptive coverage measure,
not an intervention proving causal head importance. Some heads place almost
all their attention on missing answer KV. Layer 13/head 26, for example, keeps
the name/punctuation near the answer but omits most answer-digit KV.

`teacher-head-block-alignment-row3.json` additionally compares offline selection
from the same teacher prefill: 384 individual foreground tokens versus twelve
32-token foreground blocks, each with 128 complementary background samples.
Foreground plus background capacity is 512 in both cases. Selection uses prompt
queries only; generated answer labels are used afterwards for coverage scoring.
Coverage of teacher answer attention is 34.942% for individual selection and
89.128% for block selection. Neither is a candidate generation accuracy result.

`teacher-archive-block32-row3.json` and `teacher-archive-block32-row4.json`
also show that fixed blocks improve early-layer last-prompt-query error but
can worsen later-layer error. Last-prompt-query reconstruction alone therefore
does not decide whether a policy supports future answer generation. The next
implementation to evaluate is bounded, causal block retention; it is not yet
implemented in the serving path. Current end-to-end scores remain unchanged.

## 2026-09-12 quality run and path-parity fixes

The completed run `p0-block32-background-tail128-first10` evaluates the first
10 supplied RULER records, all `niah_multikey_2`, with 16K and 32K prompts. It
uses Phi-3.5-mini, bf16 SDPA, window 4096, sink 32, 384 foreground exact slots,
128 background slots, block size 32, a 128-query prefill tail, and a 128-token
generation limit. The matched Full-KV reference scores 10/10; QCC scores 4/10
(rows 1, 2, 4, 5 correct; rows 3, 6-10 incorrect). All QCC generations ended
normally and none hit the output limit. The measured retention is 40% for this
single task and this 16K-32K subset. It is not the 128K aggregate or a worst-task
measurement. Result JSON: `p0-block32-background-tail128-first10.json`.

QCC-owned state is 5,605,839,360 bytes per request in this run, including
request scratch and RNG state; the maximum observed request peak CUDA allocation
is 16,657,575,424 bytes. Neither number establishes 128K-to-1M state growth.
The same configuration reports 6,589,440 trainable parameters out of
3,821,079,552 pretrained parameters (0.17245%). The pretrained model was not
trained. TPOT, throughput, fixed-SLA concurrency, and 1M retrieval were not
measured.

The earlier statement that bounded causal block retention was not implemented
described the state at the time of that entry. Block32 pending-state retention
is now implemented, but its admission score still uses the prompt query
side-channel and is not a causal online policy. Offline teacher attention
coverage is not end-to-end evidence; the completed 4/10 run is the current
end-to-end evidence for this candidate.

The quality-first chunk path previously used different admission behavior for
batch size one and larger batches. A new regression reproduces the mismatch;
`HybridQCCArchive._exact_chunk` now applies the same per-request salience and
top-k selection in batched prefill. Background reservoir RNG state is now
per-request, so batch rows consume independent state and batched execution
matches separate requests under the fixed seed. Related associative and hybrid
tests pass.

`benchmark_hf_ruler.py` now creates a run progress JSONL beside the final JSON
and appends the resolved command configuration plus every completed Full-KV
and QCC row. Partial runs remain inspectable if the process exits before the
aggregate JSON is written. This completed run predates the progress ledger.

## 2026-09-12 resumed path checks

Extended the existing quality-first batched-prefill regression with seven
single-token update/read steps and request reset/replay against an untouched
archive. This exercises partial block completion and background reservoir
replacement without a GPU. The focused test passes; `git diff --check` passes.
This verifies batch versus independent request decode output and reset replay
for the synthetic fixture, not equivalence between full-prefill and tokenwise
admission, suffix independence, or HF serving semantics.

The inherited tool session 63684 is unavailable in the resumed session, but
PID 69630 remains live and its CPU time advanced from 20:26 to 21:17 during
inspection. The row6 output JSON is still absent. No competing GPU job was
started. Continue monitoring this process before reading its completed trace.

The updated objective additionally requires 1M TPOT >=5x matched Full-KV,
alongside the existing 128K TPOT target. Neither has been measured; the quality
evidence remains 4/10 on the supplied 16K–32K subset.

### Key-tile sampling fix and completed row6 trace

A CPU reproduction with seed 220 found that splitting eight keys at position
three, while keeping the same 128 queries, changed salience by up to 0.19296765.
Sampling started at the key tile boundary, so scheduling changed admission
scores. Query samples now anchor to the supplied query interval; the existing
per-key eligibility mask remains. The new regression failed before the change
and passed afterwards; all 18 hybrid tests and diff whitespace checks pass.
This fixes key-tile dependence under a shared query interval. Prompt suffix
dependence and the prefill/decode admission-policy difference remain unresolved.

PID 69630 completed and wrote `retention-trace-block32-tail128-row6.json`.
Full-KV returned 8650260 correctly; QCC returned 9132055. All 32 prefill layers
captured seven answer tokens. Retained answer-token/head memberships range
from 1 to 49 per layer, showing incomplete coverage but not causal importance.
This run loaded code before the sampling fix and is evidence for that earlier
candidate. A single-GPU teacher-head alignment run was launched against this
trace, output `teacher-head-alignment-row6.json`. Its offline comparison uses
the diagnostic's fixed tail32 queries and is not a matched tail128 comparison;
the candidate membership alignment uses the actual completed trace.

The teacher alignment completed successfully: seven answer generation steps,
197.3634 total attention mass toward answer tokens summed over steps/layers/heads,
54.3209 on retained answer KV, giving 27.523% coverage. Largest missing masses
occur in layers 17, 19, 15, and 20. This is descriptive attention coverage,
not a controlled intervention on the candidate.

The diagnostic now reads foreground/background capacity and query tail from
the candidate trace, using the current fixed query-interval sampling rule.
The old tail32 result remains intact. Session 43024 writes the corrected
offline comparison to `teacher-head-alignment-tail128-row6.json`. Compilation
and diff whitespace checks pass. Offline background sampling still differs
from streaming retention, so a difference cannot be assigned solely to hidden
state drift. Removed generated pytest and test bytecode caches; retained the
experiment artifacts as evidence. Updated the external handoff's stale active
process instructions.

### Teacher KV retention replay

Matched tail128 offline comparison completed: individual selection covers
39.567% and block32 selection 74.706% of teacher answer attention, versus
27.523% for candidate membership. Background algorithm and hidden states still
differ, so this gap does not by itself identify drift.

Added an admission-only teacher-KV replay option to the existing diagnostic,
restricted to layers 15, 17, 19, and 20 (largest missing answer attention).
It calls the actual salience and admission methods with candidate capacity,
block size, query interval, and streaming reservoir; the teacher output stays
unmodified. A CPU comparison against `update_read_chunk` verifies identical
foreground keys/values/scores/ages and background keys/values across different
tile boundaries. Compilation and diff checks pass. The replay uses the current
query-sampling fix; the candidate trace predates that fix, a remaining comparison
limitation. Output will be `teacher-retention-replay-row6.json`.

Replay completed: over layers 15/17/19/20, teacher answer mass is 106.5545;
the teacher-KV replay retains 78.7408 (73.897%), whereas the actual candidate
retains 10.9020 (10.231%). This supports testing prefill hidden-state drift
before capacity changes, subject to the noted sampling-version difference.

Added a diagnostic-only exact-prefill intervention to the existing script.
Each QCC wrapper populates its bounded state, then a forward hook substitutes
the original attention's exact prefill output without retaining a Full-KV
cache. Single-token decode remains QCC. A tiny two-layer Phi CPU experiment
matches original prefill logits and verifies populated QCC token counts.
This is not an online bounded-prefill implementation or performance evidence.
The real row6 experiment is running on one GPU, output
`exact-prefill-intervention-row6.json`; capacity remains 384+128, block32,
tail128. No result is available yet.

The intervention's decode isolation was checked with a two-layer Phi CPU
model: after identical intervened prefill, retaining versus removing the hook
produces bit-identical next-token logits, and both QCC states advance from 19
to 20 tokens. This confirms the hook does not substitute exact decode.
The real run's matched Full-KV row6 has completed correctly; candidate prefill
is still running in tool session 78282. No competing GPU process was launched.

### Exact-prefill intervention outcome

`exact-prefill-intervention-row6.json` completed: matched Full-KV correct,
intervened QCC correct (8650260), 30 generated tokens, no output-limit case.
QCC state is 5,605,839,360 bytes, unchanged from the earlier candidate;
peak CUDA allocation is 16,654,749,184 bytes. All 64 prefill/completion layer
traces are present. This establishes a successful single-row diagnostic,
not aggregate quality or bounded online prefill.

The earlier failed row6 used the old query-sampling rule. To separate that
change from the prefill intervention, session 46491 now runs the current
code without intervention at identical capacity/settings, writing
`retention-trace-fixed-sampling-row6.json`. Do not attribute the improvement
solely to exact prefill until this control completes. Only one GPU job runs.

### Available next-stage data

Inspected the actual RULER subset: 80 rows, 20 each for niah_multikey_2,
niah_multikey_3, niah_single_1, and vt; maximum declared length 65,523.
It contains no 128K examples and cannot establish the requested 128K aggregate
or full-suite worst-task quality. Local generation source is available at
`/home/waas/ruler_src/scripts/data/prepare.py` with task definitions in
`/home/waas/ruler_src/scripts/synthetic.yaml`.

Before looking at additional candidate outcomes, select rows 21, 41, and 61
(the first approximately 16K records of the other three supplied tasks) for
the next cross-task diagnostic if the current intervention survives its
matched control. These are candidate-unseen within this repair sequence,
not a claim that no historical experiment has used them. They are a cheap
cross-task check, not a substitute for the fixed 128K evaluation.

Read all available `config.json` files beneath
`/datasets/ComfyUI/models/LLM`: no text configuration declares a 1M context.
The largest declared limit there is 262,144 (Qwen3-VL 4B/8B); the active Phi
declares 131,072. This local inventory does not rule out other checkpoints
elsewhere, but none in this model directory supplies the required 1M baseline.
No model limits were edited or models downloaded. Session 46491 completed.

### Matched current-code control completed

`retention-trace-fixed-sampling-row6.json` completed: Full-KV correct,
ordinary QCC wrong (9132055), same answer as the earlier failed candidate.
Compared the saved config dictionaries against the successful exact-prefill
intervention: identical, as are the QCC state bytes (5,605,839,360).
Thus the query-sampling fix does not explain the row6 success; exact prefill
rescues this sample at fixed capacity. This is a causal intervention on the
prefill computation, not proof of task-wide quality or online bounded prefill.

Started the three preselected cross-task rows 21/41/61 sequentially, each in
a separate process on the same GPU, using the identical exact-prefill
intervention and matched Full-KV reference. Outputs are
`exact-prefill-intervention-row21.json`, `...-row41.json`, and `...-row61.json`.
The sequence stops if a process fails. No concurrent GPU jobs are launched.

### Fixed-tail projection allocation

Inspection found `_bounded_prefill` projected full-prompt QKV before slicing
the quality query tail. It now slices hidden states, positions, and supplied
rotary embeddings before projection. Thus the side-channel projection is
bounded by the configured tail (the no-tail option remains length-dependent).
Extended the existing query-visibility test to check projected token counts;
all 29 retrofit tests pass, as does diff whitespace validation. This does not
make exact prefill or the full HF forward bounded-memory.

Row21 in session 98854 loaded before this allocation change; the subsequent
row41/61 subprocesses will load it. Account for this code difference when
comparing across rows; within each row the reference and candidate still use
the same checkpoint/dtype. Real-model numerical parity of this allocation
change has not yet been measured.

Row21 Full-KV emitted the correct UUID but reached the 128-token output limit;
the benchmark therefore reports answer_recall=1 and score=0/correct=false.
The candidate completed. Do not treat this baseline as a clean matched quality
denominator; retain both raw recall and output-limit information.

Row21 completed: both generations reached 128 tokens. Full-KV contains the
complete correct UUID (raw answer recall 1); exact-prefill QCC corrupts the
UUID suffix and has raw answer recall 0. Both have benchmark score 0 due to
the output limit. Thus exact prefill is insufficient for this cross-task
sample; the raw outputs establish a retrieval difference even though the
score ratio is undefined. Session 98854 has advanced to row41; row61 follows.

Extended the existing tail-query test to compare against full-prompt projection
with and without external rotary embeddings (four cases, all pass). A short
prompt below the local window also returns finite output. This provides CPU
equation parity; real-model bf16 parity remains unmeasured.

## GitHub synchronization snapshot

User requested synchronization to main for analysis. The five affected test
suites pass (103 tests). Row41/61 sequence remains active in session 98854.
Row21 answer crosses a block boundary: tokens 2277–2303 encode the UUID
prefix, tokens 2304–2309 encode `7343daf`. Several layers retain the prefix
but omit suffix KV across all heads. This suggests testing block geometry at
fixed capacity after the active sequence, not increasing capacity blindly.

Row41 completed: Full-KV and exact-prefill QCC both emit the correct number
4080114 (raw recall 1), and both hit the 128-token output limit (score 0).
QCC state remains 5,605,839,360 bytes; peak allocation 15,276,427,264 bytes.
This is retrieval success under truncation, not a valid score-ratio result.
Session 98854 has advanced to row61. Row61 starts after the GitHub main merge
and therefore includes the remote cache-compatibility and Triton-read changes;
this run still has use_triton=False. Preserve this code provenance distinction.

Row61 completed: Full-KV emitted all five expected variables in its raw output
but reached the 128-token output limit (score 0). Exact-prefill QCC emitted a
finite 65-token response with 4/5 variables, answer_recall=0.8 and score=0.8;
it omitted or corrupted one variable. QCC state is 5,605,839,360 bytes, with
peak allocation 15,333,390,336 bytes. This is the first cross-task sample where
the intervention gives a non-truncated, partially correct score, but it still
fails the task and cannot support a high aggregate-quality claim.

Across the 4 diagnostic rows, exact-prefill QCC has one exact full-score result
(row6), one partial result (row61), one raw-recall failure under truncation
(row21), and one correct answer under truncation (row41). These rows are
diagnostics only: they are approximately 16K-token examples from a 65K maximum
subset, not the requested 128K aggregate.

### Fixed-capacity block-geometry control

Ran row21 with the same 384 foreground plus 128 background slots and exact
prefill, changing retention from 12×32 blocks to 24×16 blocks. Full-KV again
returned the complete UUID; QCC returned a different UUID suffix with raw
answer recall 0 after 88 generated tokens. State was 5,593,190,912 bytes and
peak allocation 15,293,140,992 bytes. The failure persists when block boundaries
change, so the row21 loss is not explained by the 32-token block boundary.
The artifact is `exact-prefill-intervention-row21-block16.json`. This is a
single-row diagnostic; it does not justify changing the production geometry.

### Capacity and successor-score experiments

The next preselected capacity point, 768 foreground plus 128 background with
32-token blocks, succeeded on row21 under exact prefill:
`exact-prefill-intervention-row21-cap768.json` reports the complete UUID,
`answer_recall=1.0`, `score=1.0`, and 85 generated tokens. State is
5,912,547,840 bytes (5.50 GiB), peak allocation 16,123,708,416 bytes. The
384+128 exact-prefill control failed, so capacity is a demonstrated factor on
this row. The run is still a selection diagnostic; it does not establish the
minimum capacity, aggregate quality, or production speed trade-off.

An attempted ordinary-QCC cap768 benchmark was invalid: a stale external CUDA
context consumed about 16.4 GiB while `nvidia-smi` showed no visible process,
and both Full-KV and QCC generation hit OOM before producing predictions. The
failed JSON/progress files were retained as an explicit environment failure,
not quality evidence. No reset or privileged command was used.

Read the supplied CPU preexperiment package at
`/root/qcc/RULER/QCC_quality_preexperiments_20260912.zip`. Its successor
prototype propagates only the previous raw block score one step, adding one
float32 per batch/head and no trainable parameters; it improves a constructed
27/33 to 33/33 case but reduces independent short-fact retention (12 to 6).
This trade-off is mechanism evidence only.

Implemented the successor policy behind the opt-in
`quality_block_propagation` flag in `HybridQCCArchive`, the benchmark, and the
diagnostic. Raw block maxima are stored separately from propagated write scores,
and all three runtime score tensors are included in `total_state_bytes`.
An existing hybrid test verifies that the immediate successor can replace a
weaker old block while the same sequence without propagation cannot. The
focused hybrid/retrofit/benchmark tests pass (51 tests); no default behavior
changed. This is ready for the prescribed paired A/B experiment once the GPU
context is clear.

### Prefill/decode score unification

The existing quality-first prefill scores rotary K against a fixed future-query
tail, while token eviction called the untrained admission predictor (initial
bias -4). Quality-first token eviction now accepts the current rotary query and
uses the same normalized cosine score definition; the base `QCCArchive.update`
accepts and ignores the optional query for API compatibility. Both tokenwise
attention paths pass the current query with the exact rotary key. This removes
the known score-scale mismatch without changing non-hybrid behavior or adding
parameters/state. A focused test confirms a matching decode key/query is stored
with score 1.0; all 103 existing associative/hybrid/retrofit/archive/benchmark
tests pass. Real-model quality is not yet measured because the GPU has a stale
hidden allocation of roughly 16.4 GiB.

### Paired experiment scheduling note

After the GPU allocation cleared, a single-process 2×2 run for rows 6 and 21
was started with one model load and in-memory group switching. The existing
per-token exact-bank Python update path made it impractical: after about
80 minutes of CPU time it had produced only the first ordinary-QCC row6
record, while row21 was still running. The process was interrupted before
writing its aggregate JSON, so it is not evidence and no partial result is
claimed. The prescribed comparison remains pending; future runs should write
each group immediately or use a bounded diagnostic subset before committing
to multiple 16K–32K records.

### Stable-code row6 isolation result

After reverting the non-equivalent batch-read optimization, a paired row6
isolation was completed on the stable code. Group C (exact prefill, original
block score) returns the correct 8650260 with `answer_recall=1.0` and
`score=1.0`; Group D (exact prefill, one-step successor score) returns
9510896 with `answer_recall=0` and `score=0`. Both use 384+128 capacity,
block32, tail128, and the same checkpoint. The opt-in successor score rule
therefore harms this real row6 case and remains disabled by default. Artifacts:
`benchmark-exact-prefill-original-row6-current.json` and
`benchmark-exact-prefill-successor-row6-current.json`.

The earlier A/B artifacts generated while an unproven batch-read optimization
was temporarily present were not used as evidence and are excluded from the
main synchronization. This C/D result supports the report's warning that the
constructed successor benefit does not transfer automatically to real model
quality. A/B ordinary-prefill results under the stable code remain unmeasured.

### Causal prefill shadow-only result

Tested the new `quality_prefill_shadow_only` mode on real Phi row6 with the
same 384+128 capacity, block32, tail128, and matched Full-KV reference. The
mode keeps the exact tier bounded and populated but returns the recurrent QCC
output during prefill, preventing exact-tier feedback into prompt states.
Full-KV scored 1.0; shadow-only QCC generated 22 tokens saying the number was
not mentioned, with `answer_recall=0` and `score=0`. State remained
5,605,839,360 bytes. This is a valid single-row result, not an aggregate
claim, and shows that removing exact-tier feedback alone does not recover the
task. The option stays opt-in; no default behavior changed.

The run also confirms why the long 2×2 attempt was slow: even with exact reads
disabled, ordered background reservoir admission across 32 layers and 32K
tokens takes roughly 12 minutes on one A10G. A full real-model matrix must
persist each cell independently and should begin with shorter held-out rows.

### Ordinary-prefill 768-capacity result

The requested ordinary QCC prefill comparison is now valid and complete for
both selected rows, with original scoring and 768 foreground plus 128
background slots (24×32, block32, tail128). Row21: Full-KV contains the full
UUID in a 128-token truncated response; QCC emits a corrupted suffix and also
hits the limit, with raw answer recall 0 and score 0. Row6: Full-KV is correct;
QCC emits 3894566, raw answer recall 0 and score 0, without hitting the limit.
State is 5,912,547,840 bytes for both rows. Thus increasing capacity from
384 to 768 does not make ordinary prefill usable on these two rows. The
requested next focus is representation/selection drift, not another successor
score or block-width scan.

### Exact-storage dtype implementation

The exact bank now accepts an explicit `storage_dtype` and the benchmark and
diagnostic expose `--exact-storage-dtype {float32,bfloat16}`. The default stays
FP32. When BF16 is selected, K/V, pending K/V, and background K/V are stored in
BF16; scores and recurrent numerator/denominator retain their existing
precision, and reads still perform the same FP32 arithmetic. A CPU test with
BF16 source K/V confirms bit-identical FP32/BF16 read outputs and log partitions
with lower bank state bytes. This is a storage-only change and is not a quality
or speed claim. A real row21 FP32/BF16 paired run is the next measurement.

The real fixed-capacity storage pair completed on row21 with ordinary prefill,
original scoring, and otherwise identical settings. FP32 state was
5,912,547,840 bytes; BF16 state was 5,546,004,992 bytes, a reduction of
366,542,848 bytes (6.20%). Both Full-KV references reached the output limit
with raw UUID recall 1; both QCC candidates returned a corrupted UUID with
raw answer recall 0 and score 0. Exact storage dtype changed state size
without recovering this task. The BF16 artifact is
`benchmark-ordinary-cap768-bf16-row21.json`; this is one row, not a quality or
speed claim.

### BF16 score-precision correction

The preceding BF16 result is not a pure K/V-storage comparison. At that time
`_scores` and `_pending_scores` followed the BF16 K/V dtype, so close scores
could round into a tie and change replacement decisions. The state difference
therefore included 364,904,448 bytes from K/V plus 1,638,400 bytes from score
and pending-score tensors; 366,542,848 bytes is 349.5625 MiB. The old artifact
is retained as evidence of the mixed-dtype run, but must not be cited as an
FP32-score/BF16-KV result.

Fixed `reset_state` so K/V, pending K/V, and background K/V use the requested
storage dtype while `_scores` and `_pending_scores` use FP32 (or FP64 when the
state dtype is FP64). Admission/diversity score arithmetic is explicitly
FP32, and assignment casts only at the FP32 score destination. An existing
CPU replacement test reproduces 0.5001 versus 0.5002 competing for one block:
the BF16-KV bank now accepts the higher score exactly as the FP32 bank does.
All 108 focused tests pass. The corrected real pair then completed on row21:
BF16 K/V with FP32 scores,
same 768+128/original-score/ordinary-prefill configuration. QCC generated the
same corrupted UUID suffix as the FP32 run, reached the 128-token limit, and
had raw answer recall 0 and score 0. Its state is 5,547,643,392 bytes, exactly
the prior mixed-dtype state plus 1,638,400 bytes of restored FP32 score and
pending-score storage. This validates the storage-only interpretation for this
row's end-to-end output; it remains a one-row quality result, not a general
parity or performance claim.

### Fixed-prefix Q/K/V cross-read diagnostic

Added `--cross-read` to `diagnose_hf_archive.py` and ran row6 at layer17. The
teacher was generated once; its prefix through the first answer digit was then
fed identically to the student. The target is generation step 17, token `8`.
The four remote-read variants use the same fixed capacity and teacher-selected
positions where indicated; background and pending entries retain their actual
counts and weights. Mean per-head cosine to the teacher Full-KV remote output
was: A (student Q, student K/V, student positions) 0.6314; B (student Q,
student K/V, teacher-selected positions) 0.5157; C (student Q, teacher K/V,
teacher-selected positions) 0.5733; D (teacher Q, teacher K/V, teacher-selected
positions) 0.8956. Mean relative squared errors were 2.4246, 5.0004, 2.1352,
and 0.2447 respectively. These conditional read metrics are not task scores;
they exclude free-generation feedback and do not decompose additively.

For this row/layer/target, replacing student Q with teacher Q gives the largest
improvement; replacing student positions with teacher-selected positions
hurts A→B, and replacing student K/V on those positions gives only a partial
recovery B→C. This ranks query drift as the strongest of the three measured
factors for this probe, with selection and K/V representation still coupled.
The artifact is `cross-read-row6-layer17.json`; it contains the fixed prefix
metadata, top student token candidates, read metrics, positions, and state
accounting. It is a diagnostic at one layer and one target, not a system-wide
quality claim.

The same fixed-prefix cross-read then completed on row21 at layer17, targeting
the first UUID token (`target_step=49`, token `f`). Mean per-head cosine to the
teacher remote reference was A 0.7940, B 0.8085, C 0.8137, and D 0.9143;
relative squared errors were 0.4273, 0.4183, 0.4095, and 0.2224. Position and
historical K/V replacement provide small improvements on this probe, while
teacher Q remains the largest single change. The student top token at the
fixed target is `f`; this conditional prediction does not imply free-generation
success. The artifact is `cross-read-row21-layer17.json`. Together with row6,
the evidence ranks Q drift as a strong contributor, but does not isolate it
from later autoregressive feedback or prove that a Q-only repair reaches the
task targets.

The same row6/layer17 diagnostic fitted a rank-8 linear residual from student
Q to teacher Q using the common prefix (holding out the final 256 prefix
tokens). Applying it only to the student-query/student-state read raised mean
cosine A 0.6314→E 0.6605 and lowered mean relative squared error 2.4246→2.2122;
it remained far below teacher-Q D 0.8956. Query residual RMSE was 0.3819 on
the fit segment and 0.5305 on the holdout segment. Artifact
`cross-read-row6-layer17-qfit8.json` records the rank, fit split, factors,
and conditional read metrics. This is a fixed-prefix, teacher-informed
diagnostic; its coefficients are not a deployable adapter and no task score
improvement is claimed.

### Active exact-query correction hook

The existing low-rank `query_correction_*` parameters previously affected only
the recurrent archive read; quality-first exact reads bypassed them. Added an
opt-in `exact_query_correction` path that applies the same low-rank residual to
the exact-tier query before its FP32 attention calculation. The benchmark and
diagnostic expose the flag; it is disabled by default and no weights were
trained. An existing hybrid test sets a nonzero rank-1 residual and confirms
that the active exact read changes, so future calibration can target a parameter
that actually contributes to the deployed quality path. All 109 focused tests
pass. This is an effective-path plumbing change, not a quality result.

### Fixed-prefix cross-read at the final Phi layer

A second fixed-prefix cross-read completed for row6 at layer31 (the final
attention layer), using the same teacher-generated prefix and 768+128
quality-first configuration as the layer17 probe. The first answer digit is
still step17 (`8`); the student is evaluated on exactly the teacher prefix
through the token before that prediction. Mean per-head cosine to the teacher
Full-KV remote output is A (student Q, student K/V, student positions) 0.8675,
B (student Q, student K/V, teacher-selected positions) 0.8584, C (student Q,
teacher K/V, teacher-selected positions) 0.8413, D (teacher Q, teacher K/V,
teacher-selected positions) 0.9257. Mean relative squared errors are 0.6519,
0.8016, 1.3704, and 0.8160 respectively. The artifact is
`cross-read-row6-layer31.json`.

At this late layer, replacing student Q with teacher Q remains the largest
single improvement (A→D), while replacing positions or historical K/V alone
is harmful on this probe. This differs in magnitude from layer17 but preserves
the main conclusion: Q, K/V representation, and selection are coupled, and a
one-layer conditional read cannot assign a system-wide share of responsibility.
The result is a diagnostic read metric, not a free-generation score or evidence
that a Q-only adapter will meet the target.

### Causal weighted coreset reference path

Implemented an opt-in `CausalWeightedKVBank` and wired it through
`HybridQCCArchive(..., causal_coreset=True, coreset_capacity=...)`. The bank
commits each evicted KV before reading the matching query, stores a bounded
count-weighted key/value representative, and uses the count as a log-softmax
multiplicity. When full, its transparent reference merge chooses the least-cost
Ward pair among the existing representatives and the new event. It never reads
future queries, answer labels, or an external chunk boundary. The old
future-query `quality_first` table and recurrent response are not used in this
mode; it requires `exact_attention` so the retained response is combined with
the local partition by the existing exact mass equation.

The bank has fixed mutable state (`keys`, `values`, integer counts, and bounded
key radii); its merge loop is deliberately a CPU/GPU reference implementation,
not a serving-speed claim. Existing RULER CLI now exposes
`--causal-coreset` and `--coreset-capacity` for a reproducible real-model
comparison. Tests cover counted identical-key exactness, chunk invariance,
suffix independence, and an HF-attention integration case where the entire
evicted history fits in the coreset; the full local suite passes. No real-model
task score has been run with this new mode yet.

The causal bank also exposes an opt-in `merge_policy="response"`. It retains a
small FIFO set of queries observed at or before each write and chooses the pair
that minimizes a bounded joint response and log-partition error on those causal
probes; `merge_policy="ward"` remains the default transparent geometric
reference. Both policies keep the same state bound and are chunk invariant. A
past/current-query objective cannot guarantee an unseen future query, so this is
an experimental comparison hook, not a claim of future-query quality.

The causal path now freezes the obsolete recurrent codebook, admission
predictor, exact blend, and attention gate at construction. Exact mass merging
bypasses those parameters, so they are no longer counted as trainable capacity
for this deployment mode. This keeps the causal operator aligned with the
report's requirement that calibration parameters must affect the active output.

A deterministic CPU stream compared the two causal merge policies on 256
16-dimensional events from 16 key/value groups, with 8 causal query probes and
held-out queries from the same group distribution. Relative held-out output
L2 errors for Ward versus response-aware merging were 0.1900 versus 0.2184 at
capacity 8, 0.0019125 versus 0.0019125 at capacity 16, and 0.0018012 versus
0.0019009 at capacity 32. The response-aware reference was also substantially
slower (19.1 s versus 0.15 s at capacity 16) because it evaluates pairwise
response candidates. Artifact `causal_coreset_cpu.json` records the exact
stream and state bytes. This does not establish model quality; it is evidence
against promoting the response objective without a real-trace gain and a
bounded merge kernel.

### Real Phi teacher-layer causal coreset probe

A standalone layer17 probe used real Phi-3.5-mini projected rotary K/V from a
shortened row6 prompt (3,948 tokens, 3,436 historical events; target fact and
question retained). The causal Ward coreset was built online without future
queries and compared with the teacher's full historical remote read at the
final query. Mean per-head cosine was only 0.0155/0.0149/0.0152 for capacities
32/64/128, with mean relative squared error 15.56/19.43/20.36. State bytes
were 897,280/1,696,000/3,293,440 for one layer. The three capacities all
represented the same 109,952 input head-events; increasing capacity in this
range did not recover the response. Artifact
`causal-coreset-phi-layer17.json` records the prompt construction and metrics.
This is a real pretrained-layer diagnostic, not a task score or generation
result, and it rules out promoting the simple Ward mean as the quality method.

A companion real Phi layer17 selection probe used the same shortened teacher
trace but kept original K/V entries rather than averaging them. At capacities
32/64/128, deterministic uniform samples with an inverse-count mass correction
reached mean cosine 0.7354/0.7548/0.7755; online-equivalent random reservoir
samples were 0.3635/0.4444/0.4831. Farthest-point k-center selections were
0.6608/0.7413/0.7860. A post-hoc top-query selection (an oracle unavailable at
causal writes) reached 0.9370/0.9613/0.9780. Top-key-norm selection was poor
(0.2527/0.2774/0.3011). These are attention-read diagnostics, not task scores;
the artifact is `causal-selection-phi-layer17.json`. The gap between causal
static selection and the query oracle shows that preserving original K/V avoids
the Ward-mean failure but does not by itself solve unknown-future-query
selection.

### Active query correction plumbing

The existing rank-limited query factors can now optionally correct the live
projected Q before RoPE, local attention, archive addressing, and exact reads
(`active_query_correction=True`). The previous archive residual only changed a
returned archive vector, so it could not repair the Q drift observed by the
fixed-prefix probes. The new path is zero-initialized, mutually exclusive with
the exact-read-only correction to prevent double application, and exposed by
both calibration CLIs and the RULER benchmark. A focused test confirms that a
nonzero factor changes the live query; the full local suite remains green. No
weights have been trained and no quality improvement is claimed yet.

An active-query-correction calibration attempt on the existing real Phi-3.5
4K diagnostic text (50 steps, window512, rank8, bf16) was stopped after about
45 minutes while the single A10G remained occupied; it produced no adapter or
metrics artifact. It is not evidence for or against the correction. Future
calibration should first reduce the token/step budget or use the layerwise
frozen-prefix path before attempting a longer run.

### Active-query-correction calibration probe

A bounded real Phi-3.5 layerwise calibration was completed on one 1,024-token
training chunk and one independent 1,024-token held-out chunk, selecting only
layer31, rank8 active Q correction, window512, 16 codes, bf16, and 10 steps.
The best held-out calibration metrics were cosine 0.999186 and top-1 0.787109;
the fidelity gate therefore remained false. Evaluating that adapter on the
independent 4,096-token `held_4k` prompt gave mean logit cosine 0.987627 and
top-1 agreement 0.650391. The report is
`artifacts/active-query-calibration/eval_layer31_activeq_held4k.json`; the
adapter is retained locally but is ignored by Git because it is a binary
checkpoint. This is a targeted calibration diagnostic, not a task score.
The active Q hook is effective, but this small probe does not justify
promoting it or claiming that query correction alone solves the quality gap.

A larger active-Q calibration probe selected layers24–31 (rank8, window512,
16 codes, one 1,024-token train chunk, one held-out chunk, 10 steps). It used
1,597,696 trainable parameters (0.04174%). Best held-out chunk metrics were
cosine 0.999222 and top-1 0.786132; the gate remained false. On the independent
4,096-token `held_4k` prompt, the adapter reached cosine 0.982642 and top-1
0.650391, below the layer31-only probe's 0.987627/0.650391 and below the
existing all-layer archive calibration reference. Artifact
`artifacts/active-query-calibration/eval_lastquarter_activeq_held4k.json`.
This small calibration does not support promoting active-Q correction; the
feature remains available for later data-rich calibration only.

A real Phi layer17 kernel-feature probe used mathematically scaled positive
Gaussian softmax features on the same shortened teacher trace. Features 64,
128, 256, 512, and 1024 reached mean remote-read cosine 0.5451, 0.5791,
0.6051, 0.6462, and 0.6708, respectively; mean relative squared errors were
11.31, 8.30, 5.67, 4.55, and 3.08. The feature-state bytes for one layer grew
from 1.5 MiB to 24 MiB. This is a post-hoc attention-read diagnostic, not task
quality, but it rules out treating correct random-feature scaling as a cheap
replacement for the missing query-generalized response representation.
Artifact: `phi-kernel-layer17.json`.

An offline upper-bound probe ran six Lloyd iterations of per-head k-means over
the full real Phi teacher history, then used count-weighted centroid K/V in the
same final query. Mean cosine was 0.5054/0.5288/0.9019/0.6627 for 32/64/128/256
centroids; the non-monotonic result reflects centroid initialization and shows
that key-space clustering alone is unstable. This uses the complete history
and the final query for evaluation, so it is neither causal nor a task score.
Artifact: `phi-kmeans-layer17.json`. It does not justify adding a k-means
writer to the serving path without a query-domain objective and better
initialization.

### HF cache compatibility and lexical benchmark attempt

The current Phi-3.5 remote-code snapshot calls `DynamicCache.get_max_length`,
which Transformers 5 no longer provides. The loader now supplies a `None`
compatibility method, and the RULER runner probes `forward` before passing
`logits_to_keep`; this removes two setup failures encountered while wiring the
lexical-addressing diagnostic. The corresponding cache compatibility test
passes.

A real lexical-landmark RULER run could not produce a paired score on this
A10G. Phi's remote eager attention materializes the full `[heads, query,
key]` matrix: the 7,422-token NF4 row attempted another 6.57 GiB allocation
with 16.98 GiB already resident, and the 32,030-token row would require about
61 GiB. Offloading the KV cache does not bound this attention temporary. The
partial progress files were removed because they contain no paired QCC result;
no lexical quality claim is made.

### Exact chunked Phi Full-KV reference and lexical landmark task probe

The RULER runner now offers an exact online-softmax chunker for Phi's eager
remote attention. It bounds the temporary logits by query/key tiles while
preserving the full-KV softmax equation, and patches all 32 Phi attention
layers only when `--chunked-full-attention` is requested. A regression against
PyTorch causal SDPA passes. The loader also carries a logical Cache handle
across legacy Phi generation calls, so QCC no longer fabricates empty
`None`-valued legacy entries; the handle is shared by all layers and reports
one logical sequence length without physical historical KV.

A valid real-model run then evaluated the lexical hidden-state archive on RULER
row16 (7,422 tokens, `niah_multikey_2`) with a matched chunked Full-KV reference
and 4-bit weights. Full-KV returned the correct `6569343` (score 1.0); lexical
QCC generated a degenerate 128-token continuation, answer recall 0, and score
0. The QCC runtime state was 6,871,449,600 bytes and peak allocated memory
10,653,569,024 bytes. Artifact `benchmark-lexical-row15-chunked-full-v4.json`
contains the paired result. This real task result rules out promoting the
current lexical landmark path as the quality repair; chunked Full-KV is a
reference-memory fix, not a QCC quality gain.

### Paired real RULER quality-first run with chunked Full-KV

After adding the exact chunked-softmax Full-KV reference and shared legacy Cache
handle, a paired Phi-3.5-mini NF4 run completed on row16 (7,422 tokens,
`niah_multikey_2`). Full-KV returned the expected `6569343` after 30 tokens
(score 1.0). The existing future-query quality-first exact table used 24×32
foreground slots plus 128 background slots; QCC produced a punctuation-heavy
128-token continuation, answer recall 0, score 0. QCC runtime state was
5,899,964,928 bytes (including RNG state), with peak allocated
9,680,987,648 bytes. The artifact is
`benchmark-quality-first-row16-chunked-full-v1.json`. This is an independent
real task failure for the current quality-first path; it does not support
promoting larger exact capacity or the lexical variant as a general repair.

### LongRoPE boundary repair and causal parity verification

A real Phi-3.5 layer isolation found the source of the long-window failure:
raw Q/K/V projections matched the teacher exactly, but Q/K rotary outputs had
cosines 0.6781/0.6658 because the retrofit selected long factors per absolute
token position. Phi LongRoPE selects short or long factors from the current
sequence length and applies that table to every position in the call. The
retrofit now accepts an explicit sequence-length context and the HF bounded
prefill passes the complete request length to every chunk. This preserves the
semantics when a request crosses the 4,096-token original limit.

After the repair, the same layer17 real-trace attention output reached cosine
0.999984 for both one-call and 512-token chunked execution, with relative RMSE
0.00557/0.00558. The prompt/cache isolation on row16 reached prompt-logit
cosine 0.999992 and matching top-1; the one-token cached decode reached
cosine 0.988135 with matching top-1. Artifacts are
`phi-qkv-compare-row16-layer17.json`, `phi-layer17-local-isolation-row16.json`,
and `prefill-cache-isolation-row16.json`. These are implementation fidelity
checks, not aggregate task scores.

The corrected paired row16 quality-first run now returns the exact answer
(`6569343`) normally at 30 tokens, score 1.0. Artifact
`benchmark-quality-first-row16-chunked-full-v3.json` supersedes the pre-fix
row16 failure. The fixed-window row16 control also returns score 1.0 in
`benchmark-full-window-row16-chunked-v2.json`.

A four-record cross-task 8K probe (one each from niah_multikey_2,
naire_multikey_3, niah_single_1, and variable tracking) completed after the
repair. Full-KV completed 3/4 under the 128-token completion rule (the variable
tracking reference itself hit the limit). QCC completed 1/4 by the same score:
the niah_multikey_2 record finished correctly; the UUID and single-number
records contained the correct answer but continued into extra text until the
limit, and variable tracking likewise hit the limit. QCC answer recall was
1.0 on all four records. This is a four-record, 0–8K diagnostic, not a suite
result; artifact `benchmark-quality-first-selected8k-v1.json` records both
recall and completion scores.

The corrected LongRoPE path was checked against the ordinary archive on the
same row16. With window4096 and 16 recurrent codes (no exact tier), Full-KV
scored 1.0 while ordinary QCC still scored 0 and produced a semantic refusal
until the 128-token limit. State was 5,160,566,784 bytes and peak allocated
8,931,024,896 bytes. Artifact `benchmark-ordinary-row16-longrope-fix-v1.json`.
This separates the resolved rotary implementation error from the remaining
recurrent archive quality loss.

### Post-LongRoPE fixed-prefix cross-read

The fixed-prefix Q/K/V cross-read was repeated on row16 after the LongRoPE
repair (layer17, first answer digit at teacher step20). Mean per-head remote
read cosine for A/B/C/D was 0.9284/0.9317/0.9318/0.9890, with relative squared
error 0.2310/0.2179/0.2172/0.0264. Replacing student Q with teacher Q remains
the dominant improvement; teacher-selected positions and teacher K/V provide
small additional gains on this probe. The student top token at the fixed
prefix is `6`, matching the target. Artifact
`cross-read-row16-layer17-postrope.json`. This is a conditional one-layer
metric, not a task score or an additive error decomposition.

The earlier row6/row21 layer17 and row6 layer31 cross-read artifacts were
collected before the LongRoPE factor-selection repair. Their Q/K source
comparisons include the pre-fix rotary mismatch and must not be used as current
factor rankings. They remain retained for debugging history; the row16
post-repair cross-read is the current diagnostic reference.

The first corrected 16K quality-first probe completed on row1 (16,120 tokens,
`niah_multikey_2`). Full-KV and QCC both returned `7549132` normally, score
1.0. QCC state was 5,899,964,928 bytes and peak allocated 11,407,241,216
bytes. Artifact `benchmark-quality-first-row1-16k-longrope-v1.json`.

The corrected 16K UUID probe completed on row21 (`niah_multikey_3`, 16,156
tokens). Full-KV produced a 128-token answer containing the target UUID but
hit the configured completion limit (raw recall 1, score 0). Quality-first
768+128 QCC terminated after 92 tokens but returned a different UUID
(`9917b1c1-2b7d-477e-9af0-86ff63c8bdd8`), raw recall 0 and score 0. QCC state
was 5,899,964,928 bytes and peak allocated 11,414,560,256 bytes. Artifact
`benchmark-quality-first-row21-16k-longrope-v1.json`. This is the first
post-RoPE long UUID failure and shows the repaired implementation still needs
a better retention/representation path for UUID-style associations.

The post-study boundary patch is now integrated: bounded prefill computes
`rope_context_length = previous_logical_length + new_length`, and the wrapper
synchronizes QCC state only when a native Phi prepare call explicitly drops an
existing cache. CPU tests cover both behaviours; the full suite passes. The
model-level cache-rebuild path still needs a real crossing test before it is
used as a production compatibility guarantee.

### Real Phi cache-rebuild boundary check

A real Phi-3.5 NF4 QCC generation with a 4,096-token prompt and two requested
new tokens crossed the native short-to-long transition. The installed Phi
prepare path was wrapped, explicitly discarded its existing cache once, and
all 32 QCC layers ended at logical length 4,097 (rather than retaining the old
4,096-token state and appending a second full prefix). This verifies the reset
control flow in the current environment. The check measures state continuity,
not task quality or Full-KV parity; the generated continuation was not used as
a benchmark result. Artifact `cache-rebuild-phi-4096.json`.

### State reclamation: measured reduction and selection regression corrected

The first reclamation run completed on row16: runtime state fell from
5,899,964,928 to 4,162,737,152 bytes (1,737,227,776 bytes reclaimed), but the
candidate returned 6920437 instead of 6569343. The raw result is retained in
`benchmark-quality-first-row16-state-reclaim-v1.json`; it is not evidence of
quality-preserving memory reduction.

Inspection found that the exact-only chunk branch omitted quality_query,
quality_key_start and quality_query_start, changing retention scores. These
arguments, and explicit admission_score, are now forwarded. A regression
compares the reclaimed and recurrence-maintaining paths across three successive
chunks using the same future query tail: outputs, retained keys/values/scores/
ages and background entries are bit-identical. The reclamation is restricted
to quality-first exact attention and causal coreset configurations; predictor
modes still retain raw keys because their admission predictor may consume them.
The archive/attention/HF focused suite passes (97 cases). The corrected real
model run remains outstanding. Shared scratch and the 8192-slot BF16 comparison
are not implemented by this change.


### Correction of two diagnostic comparisons (2026-09-14)

Inspection of the actual generating scripts found two metric defects. In
`prefill-cache-isolation-row16.json`, next_step_logits compares the student's
next prediction against the teacher's previous prompt prediction. The 0.988135
cosine is not a matched cached-decode fidelity measurement. Prompt-logit metrics
are unaffected. No teacher decode reference was collected by that script.

In `causal-coreset-phi-layer17.json`, read_attention returned [1,32,1,96]
while the reference had shape [1,32,96]. Subtraction and cosine broadcast across
heads. The reported ~0.015 cosine and relative errors are invalid and do not
rule out Ward compression. The singleton query dimension must be removed and
same-head shapes asserted before recomputing. Original values are preserved,
with explicit validity annotations. The separate synthetic counterexamples
remain separate evidence. These corrections retract the earlier conclusions
based on these two numbers; they do not establish new quality results.

### Release per-layer exact-path scratch after execution

HF inference now drops each exact-only layer's chronological K/V scratch after
its forward completes. The persistent ring has already copied the retained
K/V; projected outputs do not alias scratch. PyTorch can reuse the released
allocation on subsequent layers on the same execution stream. This preserves
reuse within a layer's internal prefill loop without maintaining a second
K/V working window for every layer for the entire request. No allocator cache
flush or shared mutable cross-stream buffer is introduced.

An existing HF test file compares this lifecycle with a scratch-retaining
reference through a 13-token prefill, 5-token continuation and single-token
decode after eviction. Outputs match exactly and owned runtime bytes decrease;
all 33 HF retrofit tests pass. Actual GPU peak/reserved-memory and throughput
changes remain unmeasured. The currently running row16 state-reclaim-v2 job
was launched before this scratch-lifetime edit and does not test this change.

### Logical parameter denominator for NF4

The RULER runner now calls the loaded model's Transformers num_parameters()
method before patching. Its Params4bit handling counts logical weights rather
than packed storage elements. Reports retain pretrained_storage_elements
separately and identify the counting method. A real CPU NF4 Linear4bit(32,16)
check counted 272 packed elements and the correct 528 logical parameters
(including bias). Existing historical reports retain their original numbers;
for Phi, 6,589,440 / 3,821,079,552 is approximately 0.17245%, not the fraction
obtained using 2,009,140,224 packed elements. This does not establish that all
reported trainable parameters contribute to the active inference output.
The running state-reclaim-v2 process predates this metadata correction.

### Corrected reclamation real-model result

The row16 state-reclaim-v2 paired run completed: both Full-KV and QCC return
6569343 and finish normally after 30 generated tokens. QCC owns 4,162,737,152
bytes, compared with 5,899,964,928 previously (29.44% less). Peak allocated
is 7,944,146,944 bytes. This validates recurrence/raw-key reclamation on one
NF4 record with unchanged 768+128 capacity. It does not validate the subsequent
scratch release, BF16 bank storage, larger capacity, or suite-level quality.

RULER reports now additionally summarize answer recall and its paired ratios by
task and length. Existing completion-weighted score fields remain distinct.
A zero Full-KV denominator yields null rather than a misleading zero ratio.

### BF16 storage comparison through replacement and reservoir updates

Expanded the existing precision regression from a not-yet-full table to a
43-event, two-request/two-head stream with close FP32 scores, both token and
four-token block retention, and a three-slot background reservoir. After every
write, BF16-source K/V stored in BF16 or FP32 yield identical foreground ages,
scores, K/V, background contents/counts, attention outputs, and log partitions.
The stream exceeds both capacities and ends with an incomplete block. All 19
associative tests pass. This confirms the storage-only property under these
controlled BF16-source conditions; it does not substitute for the forthcoming
real-model capacity/storage pair.

### Sequential storage validation in progress

The scratch-release row16 FP32-table run remains the active GPU experiment.
A follow-up process waits for that exact process to terminate and requires its
final paired report to show correct, completed answers before launching
`benchmark-quality-first-row16-scratch-bf16-v1.json`. The follow-up keeps the
768+128 capacity and all other recorded settings; only exact_storage_dtype
changes to bfloat16. Its stdout/stderr are preserved in the matching .log file.
If the first run fails or produces no report, the follow-up stops without
starting another GPU workload. No result is claimed for either pending run.


### Scratch release verified on real row16

The scratch-release-v1 run completed with the same correct 30-token prediction
as the pre-reclamation quality-first 768+128 run. Owned runtime state is
2,350,797,824 bytes (2.1894 GiB), down from 5,899,964,928 bytes
(60.16% reduction). Relative to recurrence/raw-key reclamation alone,
this releases another 1,811,939,328 bytes, exactly the retained per-layer K/V
scratch geometry. Peak CUDA allocated is 6,131,671,040
bytes. These measurements concern one NF4 row16 request, with FP32 bank K/V;
they do not establish concurrency or serving speedups. Same-capacity BF16
validation has started sequentially and is still pending.

### Capacity budget from allocated bank tensors

Actual CPU allocations for the current bank geometry are 23,105,792 bytes per
layer at 768 FP32 foreground slots, 11,702,528 at 768 BF16, and 105,779,456
at 8192 BF16, including background/pending tensors, scores and ages. Holding
all other measured request buffers and CUDA RNG bytes fixed gives estimated
request totals of 2,350,797,824 / 1,985,893,376 / 4,996,355,072 bytes.
The 8192 BF16 estimate is about 4.65 GiB, below the original 5.50 GiB request
state. It is not a GPU peak or concurrency measurement. The allocation record
is exact-storage-capacity-budget.json; forthcoming GPU runs check the estimate.

### Same-capacity BF16 GPU comparison completed

The scratch-bf16-v1 run returned exactly the FP32 scratch-release prediction,
including normal termination after 30 tokens on row16. Owned request bytes are
1,985,893,376 (1.85 GiB), exactly matching the allocation-derived estimate;
peak CUDA allocated is 5,763,817,472 bytes. Foreground capacity remains 768
with 128 background slots and FP32 replacement scores. This validates original
BF16 K/V storage on this paired real run. It does not yet establish larger
capacity quality. The queued row21 experiment with 8192 BF16 foreground slots
has started after this process terminated, still using one GPU.
