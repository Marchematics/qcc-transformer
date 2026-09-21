# LongBench retention: bounded decode cache vs Full-KV

The RULER evidence behind the retention law
(`docs/REPORT.md`) is synthetic: needle-in-a-haystack and variable tracking.
This harness adds **LongBench** (Bai et al., 2023) — real long-document QA,
summarisation, classification and retrieval — and runs the shipped law
(`qcc_transformer.retention.compile_bounded_cache`) against a matched exact
Full-KV arm on the same records, prompts and decode loop.

## Files

| file | role |
| --- | --- |
| `benchmarks/longbench_data.py` | downloads/caches the official LongBench data + configs, loads records, builds the official prompt |
| `benchmarks/longbench_metrics.py` | official LongBench metrics, dependency-free |
| `benchmarks/benchmark_retention_longbench.py` | the runner (Full-KV arm vs bounded arm, per-task retention) |
| `tests/test_longbench_metrics.py` | CPU-only unit tests with hand-computed expectations |

## Quick start

```bash
# 0) unit tests (no network, no GPU, a few seconds)
python -m pytest -q tests/test_longbench_metrics.py

# 1) dataset check: downloads into $QCC_LONGBENCH_DIR and prints counts
python benchmarks/longbench_data.py --tasks narrativeqa qasper

# 2) the real run (GPU; 1B model, ~20 records/task)
python benchmarks/benchmark_retention_longbench.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --tasks narrativeqa qasper hotpotqa 2wikimqa gov_report multi_news \
            triviaqa samsum vcsum passage_retrieval_en \
    --limit 20 --budget 4096 --lex-cap 512 --hops 6 \
    --out artifacts/longbench-retention-llama32-1b.json

# 3) optional: same harness, no checkpoint, CPU (random 2-layer model; smoke test only)
python benchmarks/benchmark_retention_longbench.py \
    --model meta-llama/Llama-3.2-1B-Instruct --tiny-random-model \
    --device cpu --dtype float32 --tasks samsum passage_retrieval_en \
    --limit 2 --max-input-tokens 1024 --max-new 6 \
    --budget 256 --lex-cap 64 --out /tmp/longbench-smoke.json
```

`--tiny-random-model` builds a randomly initialised 2-layer Llama with the
checkpoint's tokenizer.  It exercises the whole runner (prompt build, tokenize,
truncate, both arms, greedy decode, metric, JSON) without loading weights, and
its report is marked `"evidence": false` so it can never be mistaken for a
result.

## Data

* Upstream: HF dataset repo `THUDM/LongBench` (now also served as `zai-org/LongBench`).
  The repo holds `data.zip`; the zip contains `data/<task>.jsonl` (200 records
  for most tasks) plus the LongBench-E variants `data/<task>_e.jsonl`.
* The three configs are **not** in the HF repo — they live in the GitHub
  repository under `LongBench/config/`.  `dataset2metric.json` does not exist
  upstream at all: the official `eval.py` hard-codes the table, and
  `longbench_data.py` writes that table to `dataset2metric.json` in the cache so
  the cache is self-describing.  `longbench_metrics.load_dataset2metric()`
  prefers the JSON file and falls back to the built-in official table.
* Cache: `$QCC_LONGBENCH_DIR`, default `~/.cache/qcc/longbench`
  (`--cache-dir` overrides both).
  Files: `data/<task>.jsonl`, `dataset2prompt.json`, `dataset2maxlen.json`,
  `dataset2metric.json`, `manifest.json` (counts/paths/sources), and
  `_download/data.zip` (the 109 MiB archive; delete it after extraction with
  `--prune-archive` or by hand).  A full extraction is ~300 MiB.
* Verified download (2026-09-21): **all 21 official tasks, 4 750 records** —
  every task has 200 records except `multifieldqa_en` (150), `lcc` (500) and
  `repobench-p` (500).  `manifest.json` records the per-task counts, and
  `ensure_dataset(tasks=None)` extracts/validates the whole suite, not just the
  tasks of one run.
* Network note: a `no_proxy` value containing `[::1]` makes
  `httpx`/`huggingface_hub` raise `InvalidURL: Invalid port: ':1]'`.  Both
  `longbench_data.py` and the runner call `sanitize_proxy_env()` (at import
  time) which rewrites only `NO_PROXY`/`no_proxy` to `localhost,127.0.0.1`.
* The prompt is built exactly as in the official `pred.py`:
  `template.format(context=..., input=...)` with the task template from
  `dataset2prompt.json`.

## Tasks and metrics

`--tasks` defaults to the English, real-document subset below (`vcsum` included;
it is skipped unless `jieba` is installed — see caveats).

| task | official metric | implemented here |
| --- | --- | --- |
| narrativeqa, qasper, hotpotqa, 2wikimqa, musique, multifieldqa_en | `qa_f1_score` (normalised token F1) | yes |
| gov_report, qmsum, multi_news, samsum | `rouge_score` (ROUGE-L, `rouge` package semantics) | yes |
| trec, lsht | `classification_score` | yes |
| passage_retrieval_en | `retrieval_score` (`Paragraph N`) | yes |
| passage_retrieval_zh | `retrieval_zh_score` (`段落N`) | yes |
| passage_count | `count_score` | yes |
| vcsum, dureader, multifieldqa_zh | `rouge_zh_score` / `qa_f1_zh_score` | needs `jieba` |
| lcc, repobench-p | `code_sim_score` | needs `fuzzywuzzy` |

Metric fidelity (all verified by differential testing against the official
implementations, 4 000 random cases × 6 metrics, 0 mismatches — including
against the real `rouge==1.0.1` package the official harness calls):

* `qa_f1_score` is the official normalisation (lowercase, strip
  `string.punctuation`, drop `a|an|the`, collapse whitespace) plus the official
  multiset `f1_score`.  It is order-insensitive and ignores case/punctuation.
* `rouge_score` is *summary-level* ROUGE-L with the `rouge` package's
  semantics: split on `"."`, align every reference sentence against every
  hypothesis sentence, **union** the LCS token sets, count *distinct*
  reference/hypothesis words for recall/precision, and
  `f = 2PR/(P+R+1e-8)`.  It is case- and punctuation-sensitive (unlike
  `qa_f1_score`), a perfect match is `0.999999995` rather than exactly 1, and
  the official wrapper turns every failure (empty prediction, empty reference)
  into `0.0`.
* `classification_score` reproduces the official loop that removes from the
  match list *while iterating it*; that quirk can skip a removal when a
  prediction contains several classes, and reproducing it is what keeps the
  numbers comparable with published LongBench tables.
* `retrieval_score` divides by the number of numbers in the prediction, so a
  correct id mixed with unrelated numbers scores below 1 (official behaviour).
* `score_record` mirrors `eval.py`'s `scorer`: first line only for
  `trec/triviaqa/samsum/lsht`, and the score of a record is the maximum over its
  reference answers (0.0 if it has none).  Scores are in `[0, 1]`; the official
  harness multiplies by 100 for its tables.

## What the JSON contains

`--out <file>.json` (plus a streaming `<file>.json.progress.jsonl`, one line per
arm/record, written as the run proceeds and usable by
`benchmarks/recover_bounded_decode_log.py`-style tooling or plain `jq`):

* `dataset`: repo id, cache dir, requested/run tasks, record counts, config
  sources; `skipped_tasks` (with the reason), `skipped_records`,
  `metric_errors`, `dataset2metric`;
* `truncation_policy` and `decode`: how over-long prompts were handled (see
  below) and the exact decode settings (greedy, stop at EOS, `max_new` per task,
  `add_special_tokens=false`, dtype/device/attention implementation);
* `config`: the full CLI, so a run is reproducible;
* `records`: one row per (record, arm) with `score`, `prediction`,
  `prompt_tokens_before/after`, `truncated`, `kept_slots`,
  `slots_per_token`, `generated`, `prefill_s`, `decode_s`, `peak_gib`,
  `pinned_rope`, and `metric_error` when scoring raised;
* `summary.tasks[task][arm]`: `records`, `mean_score`, `mean_kept_slots`,
  `mean_prompt_tokens`, `mean_generated`, `positive_records`;
* `summary.tasks[task].retention`: `matched_records`, `skipped_zero_full`,
  `mean` and `worst` of `bounded_score / full_score`, counted **only** over
  records whose Full-KV arm scored above zero, so the number reads as "how much
  of what Full-KV can do survives" rather than an absolute score;
* `summary.aggregate`: macro `mean_retention`, `worst_task_retention`,
  `worst_record_retention`, `macro_mean_full_score`,
  `macro_mean_bounded_score`, `matched_records`;
* `partial: true` is written if the run crashes or is interrupted, so a
  half-finished run is still an inspectable artifact.

## Caveats

* **Truncation (explicit).**  A prompt longer than `--max-input-tokens`
  (default 32 768) is truncated the way the official LongBench harness does it:
  keep the **first half and the last half** of the token ids, because the head
  carries the task framing and the tail carries the question/instruction.  Token
  ids are kept instead of the official detokenise/re-encode round trip.  Each
  affected row is flagged `truncated: true` with both token counts, and the
  policy is repeated in the JSON header.  `--truncation skip` drops such records
  into `skipped_records` instead.  Measured with the Llama-3.2 tokenizer over 20
  prompts per task, only `narrativeqa` exceeds 32 768 tokens (up to ~46.7 k);
  pass `--max-input-tokens 65536` to keep those records intact.
* **Chinese tasks are skipped, not approximated.**  `vcsum` (in the default
  list), `dureader` and `multifieldqa_zh` need `jieba`-based metrics; `lcc` and
  `repobench-p` need `fuzzywuzzy`.  Without those packages the tasks appear in
  `skipped_tasks` with the missing package named and never contribute a
  silently-zero score.  `pip install jieba` (or `fuzzywuzzy`) makes them
  runnable without touching this code.
* **Decode is the retention harness's loop, not the official generator.**  Both
  arms decode greedily with `--max-new` tokens (default: the per-task official
  `dataset2maxlen.json` value: 128/512/32/64), stop at EOS, use absolute
  position ids and no chat template, and tokenise with
  `add_special_tokens=False`.  The official `pred.py` also caps `samsum` with
  `min_length` to stop runaway `" Dialogue"` repetition; that guard is not
  reproduced, so absolute `samsum` scores may sit below published numbers for
  both arms (the retention ratio is unaffected).
* **A small budget is what makes the comparison bite.**  When a prompt is
  shorter than `budget + lex_cap` the law keeps every slot, so `bounded`
  reproduces `full` exactly and retention is 1.0 by construction; read
  `slots_per_token` per record to see how much compression actually happened.
  With the default `--budget 4096 --lex-cap 512`, `multi_news` (median ~1.7 k
  tokens with the Llama-3.2 tokenizer) is mostly uncompressed while
  `narrativeqa`, `triviaqa`, `gov_report`, `vcsum`, `samsum` and
  `passage_retrieval_en` (10 k–47 k tokens) are genuinely compressed.  Lower
  `--budget` if you want the short tasks to be informative too.
* **`mean_score` is an internal retention number**, not a leaderboard claim: it
  is the official metric (×1, not ×100) on this harness's greedy decode.  The
  headline result is `retention`, which is arm-matched by construction.
* **CPU smoke path.**  The runner is smoke-tested end to end on CPU with
  `--tiny-random-model` (both arms, truncation, skip paths, progress file,
  partial-report-on-interrupt) and its summarisation logic is unit-checked with
  synthetic rows.  A real run needs a CUDA device; the `--load-4bit` path and
  checkpoints whose forward takes no `cache_position` are not covered by the
  CPU smoke test.
