"""Does the NLL gap track long-range dependency density, or context length?

Prediction 3 of ``docs/MECHANISM.md``: at a fixed context length and a fixed
retained budget, the NLL ratio to Full-KV should track how much genuine
long-range dependency the text carries, not how long the text is.  The two
corpora already measured (natural text vs a self-similar synthetic corpus)
differ as predicted, but the density axis itself has never been swept.

This runner builds **one** 32K-token document pool per level and measures the
suffix NLL of a matched Full-KV arm and of every ``--budgets`` arm:

``natural``
    the corpus order;
``paragraph``
    whole paragraphs permuted;
``sentence``
    sentences permuted;
``token``
    every token permuted.

All four levels are permutations of the *same* pool of exactly ``--length``
tokens: the sentence blocks are tokenised once and the pool is the concatenation
of whole sentence blocks (the last one truncated), so the multiset of token ids
is identical across levels by construction and only the order -- and therefore
the dependency structure -- changes.  Unigram statistics are exactly equal, so a
difference in NLL ratio cannot be blamed on vocabulary or frequency.

The x-axis: ``dependency_density``
----------------------------------
An ordinal "natural > paragraph > sentence" is not a figure.  The proxy below is
computed from the token ids alone and is defined as:

    For each position ``i >= k - 1`` let ``g(i)`` be the ``k``-gram ending at
    ``i`` (``k = --ngram``, default 5) and let ``j(i)`` be the position of the
    **nearest earlier occurrence** of ``g(i)`` (``j(i) < i``), if any.  With
    ``W = --near-window`` (default 512):

        repeat_fraction      = #{i : j(i) exists} / N
        far_repeat_fraction  = #{i : j(i) exists and i - j(i) > W} / N
        long_range_share     = far_repeat_fraction / repeat_fraction
        mean_nearest_span    = mean{ i - j(i) : j(i) exists }
        dependency_density   = far_repeat_fraction

    (``N = len(ids) - (k - 1)``; a token with no earlier copy of its ``k``-gram
    does not count as long-range supported, so permuting every token drives the
    proxy to ~0.)

``dependency_density`` is the share of tokens whose *only* repetition support
lies beyond the near window -- precisely the support a bounded cache cannot
see, which is why it is the mechanistically relevant x-axis for prediction 3.
The other components are reported so the figure can be re-drawn with a different
proxy.  The proxy is *measured*, not assumed: the summary reports the resulting
order of the levels and the per-budget correlation (Pearson and Spearman)
between the proxy and ``nll_ratio_to_full`` across levels, whatever their sign.

Measurement
-----------
Per level: exact chunked prefill of ``doc[:L - suffix]``, then teacher-forced
scoring of the last ``--suffix`` tokens with each true token appended to the
cache (as a real decoder would).  Full-KV uses ``prefill_capture``; each bounded
arm uses the packaged selection (``observation_scores`` + ``select_indices`` +
``prune_cache``, the steps ``compile_bounded_cache`` runs internally) on one
shared exact prefill of the level, or ``compile_bounded_cache`` itself with
``--no-shared-prefill``.  No lexical anchors: there is no question to anchor on,
so ``--lex-cap`` defaults to 0.
"""

from __future__ import annotations

import os

import argparse
import glob
import json
import math
import random
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qcc_transformer.hf_loading import _ensure_remote_code_compat, load_hf_causal_lm
from qcc_transformer.retention import (RetentionConfig, compile_bounded_cache,
                                       fixed_rope_length, forward_accepts,
                                       observation_scores, prefill_capture,
                                       prune_cache, select_indices)

LEVELS = ("natural", "paragraph", "sentence", "token")

#: The same local text files ``benchmark_bounded_decode_lm_nll.py`` uses, so the
#: two harnesses can be compared on the same corpus.  No download, ever.
DEFAULT_CORPUS = (
    str(REPO_ROOT / "*.md"),
    str(REPO_ROOT / "docs" / "*.md"),
    str(REPO_ROOT / "benchmarks" / "*.py"),
    str(REPO_ROOT / "qcc_transformer" / "*.py"),
)

#: Words for the ``--synthetic`` fallback.  Long-range structure is injected by
#: reusing whole sentences at paragraph distance, so the natural level of the
#: synthetic corpus has a measurably positive ``dependency_density``.
SYNTH_SUBJECTS = ("surveyor", "archivist", "engineer", "navigator", "chemist",
                  "gardener", "locksmith", "librarian", "machinist", "observer")
SYNTH_VERBS = ("measured", "recorded", "compared", "labelled", "recovered",
               "sketched", "rechecked", "catalogued", "transported", "sealed")
SYNTH_OBJECTS = ("the intake valve", "the second sample", "the northern plot",
                 "the revised ledger", "the spare coupling", "the outer casing",
                 "the earlier reading", "the lower bracket", "the final batch",
                 "the archive box")
SYNTH_TAILS = ("before the light faded", "during the long pause",
               "while the room was quiet", "after the second attempt",
               "in the usual order", "without further comment",
               "under the same conditions", "for no particular reason")


def encode_ids(tokenizer, text):
    """Token ids of ``text`` with no special tokens, for HF or stub tokenizers."""
    try:
        return list(tokenizer(text, add_special_tokens=False).input_ids)
    except TypeError:
        return list(tokenizer.encode(text, add_special_tokens=False))


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #

def load_corpus_text(sources, min_bytes=2000):
    """Concatenate every local file matching ``sources`` into one document."""
    chunks, seen = [], 0
    for pattern in sources:
        for path in sorted(glob.glob(pattern)):
            try:
                text = Path(path).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if len(text) >= min_bytes:
                chunks.append(text)
                seen += 1
    if not chunks:
        raise SystemExit(
            "no corpus file matched " + ", ".join(sources)
            + "; pass --corpus <path/glob> or --synthetic")
    return "\n\n".join(chunks), seen


def split_paragraphs(text):
    """Blank-line separated blocks, with the separator dropped."""
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def split_sentences(text, max_chars=400):
    """Sentence-ish split: punctuation or newline, then a hard character cap.

    Code and markdown rarely have clean sentences, so an over-long piece is cut
    at the last space before ``max_chars``; the split is only used to define the
    shuffle blocks, and both shuffles use the same blocks.
    """
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = []
    for piece in pieces:
        piece = piece.strip()
        while len(piece) > max_chars:
            cut = piece.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            out.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            out.append(piece)
    return out


def synthetic_corpus(min_tokens, tokenizer, seed=0):
    """A dependency-structured pseudo-corpus, for runs with no corpus on disk.

    Paragraphs are generated from a small template grammar around a topic word;
    roughly a quarter of the sentences are verbatim copies of a sentence from an
    *earlier* paragraph, which is exactly the long-range repetition the density
    proxy is meant to quantify.
    """
    rng = random.Random(f"qcc-synthetic-corpus:{seed}")
    topics = [f"channel{index}" for index in range(24)]
    generated, motifs = [], []

    def next_paragraph():
        topic = rng.choice(topics)
        sentences = []
        for _ in range(rng.randint(4, 7)):
            if motifs and rng.random() < 0.25:
                sentence = rng.choice(motifs)
            else:
                sentence = (f"The {rng.choice(SYNTH_SUBJECTS)} "
                            f"{rng.choice(SYNTH_VERBS)} "
                            f"{rng.choice(SYNTH_OBJECTS)} on {topic} "
                            f"{rng.choice(SYNTH_TAILS)}.")
            sentences.append(sentence)
        motifs.extend(sentences)
        generated.append(" ".join(sentences))

    next_paragraph()
    total = len(encode_ids(tokenizer, " " + generated[-1]))
    while total < min_tokens * 1.05:
        next_paragraph()
        total += len(encode_ids(tokenizer, " " + generated[-1]))
    text = "\n\n".join(generated)
    while len(encode_ids(tokenizer, text)) < min_tokens:
        next_paragraph()
        text = "\n\n".join(generated)
    return text


def corpus_units(text, tokenizer):
    """Yield ``(sentence token ids, paragraph index)`` lazily, in corpus order.

    Lazy on purpose: a 32K pool needs a couple of thousand sentences, and
    tokenising the whole 1.7 MB corpus (tens of thousands of sentences) just to
    throw most of them away would dominate the startup cost.
    """
    for paragraph_index, paragraph in enumerate(split_paragraphs(text)):
        for sentence in split_sentences(paragraph):
            yield encode_ids(tokenizer, " " + sentence), paragraph_index


def sentence_units(text, tokenizer):
    """``(unit_ids, paragraph_of_unit)``: one token block per sentence."""
    pairs = list(corpus_units(text, tokenizer))
    if not pairs:
        raise SystemExit("the corpus produced no sentences to shuffle")
    return [unit for unit, _owner in pairs], [owner for _unit, owner in pairs]


# --------------------------------------------------------------------------- #
# levels: one pool, four orders
# --------------------------------------------------------------------------- #

def build_pool_from_pairs(pairs, length):
    """One token pool of exactly ``length`` ids plus its block boundaries.

    ``pairs`` yields ``(unit_ids, group)``; whole units are concatenated until
    the pool is full and the last one is truncated, so every level is an exact
    permutation of the same ids.  The iterable may be lazy: construction stops
    as soon as the pool is full.  Returns ``(ids, unit_bounds, group_bounds)``
    where each bound list starts at 0, ends at ``length`` and delimits the
    blocks of the respective granularity.
    """
    ids, unit_bounds, group_bounds = [], [0], [0]
    current_group = None
    for unit, group in pairs:
        if len(ids) >= length:
            break
        if group is not None and group != current_group:
            if group_bounds[-1] != len(ids):
                group_bounds.append(len(ids))
            current_group = group
        ids.extend(unit)
        if len(ids) <= length:
            unit_bounds.append(len(ids))
    if len(ids) < length:
        raise SystemExit(
            f"the corpus only yields {len(ids)} tokens, need {length}; "
            "add --corpus globs or use --synthetic")
    del ids[length:]
    return (ids,
            sorted({b for b in unit_bounds if b <= length} | {length}),
            sorted({b for b in group_bounds if b <= length} | {length}))


def build_pool(unit_ids, length, groups=None):
    """``build_pool_from_pairs`` with an optional parallel ``groups`` list."""
    pairs = (zip(unit_ids, groups) if groups is not None
             else ((unit, None) for unit in unit_ids))
    return build_pool_from_pairs(pairs, length)


def permute_blocks(ids, bounds, rng):
    """Permute whole ``[bounds[i], bounds[i + 1])`` blocks, keeping each intact."""
    blocks = [ids[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    rng.shuffle(blocks)
    return [token for block in blocks for token in block]


def permute_all(ids, rng):
    out = list(ids)
    rng.shuffle(out)
    return out


def level_sequences(pool_ids, unit_bounds, group_bounds, seed=0, levels=LEVELS):
    """``{level: ids}`` -- every value an exact permutation of ``pool_ids``."""
    out = {}
    if "natural" in levels:
        out["natural"] = list(pool_ids)
    if "paragraph" in levels:
        out["paragraph"] = permute_blocks(pool_ids, group_bounds,
                                          random.Random(f"{seed}:paragraph"))
    if "sentence" in levels:
        out["sentence"] = permute_blocks(pool_ids, unit_bounds,
                                         random.Random(f"{seed}:sentence"))
    if "token" in levels:
        out["token"] = permute_all(pool_ids, random.Random(f"{seed}:token"))
    return out


# --------------------------------------------------------------------------- #
# density proxy
# --------------------------------------------------------------------------- #

def dependency_density(ids, ngram=5, near_window=512):
    """The x-axis of prediction 3: see the module docstring for the formula.

    ``dependency_density`` is the fraction of scored positions whose nearest
    earlier copy of their ``ngram``-gram lies more than ``near_window`` tokens
    back -- i.e. tokens whose repetition support is *long-range only*.
    """
    n = len(ids)
    scored = max(1, n - (ngram - 1))
    empty = {"ngram": ngram, "near_window": near_window, "tokens": n,
             "scored_positions": 0, "repeat_fraction": 0.0,
             "far_repeat_fraction": 0.0, "long_range_share": 0.0,
             "mean_nearest_span": 0.0, "median_nearest_span": 0.0,
             "dependency_density": 0.0}
    if n < ngram:
        return empty
    last: dict[tuple, int] = {}
    repeats = far = 0
    spans: list[int] = []
    for index in range(ngram - 1, n):
        gram = tuple(ids[index - ngram + 1: index + 1])
        previous = last.get(gram)
        if previous is not None:
            repeats += 1
            span = index - previous
            spans.append(span)
            if span > near_window:
                far += 1
        last[gram] = index
    spans.sort()
    middle = len(spans) // 2
    median = (0.0 if not spans else
              float(spans[middle]) if len(spans) % 2 else
              (spans[middle - 1] + spans[middle]) / 2.0)
    return {
        "ngram": ngram, "near_window": near_window, "tokens": n,
        "scored_positions": n - (ngram - 1),
        "repeat_fraction": round(repeats / scored, 6),
        "far_repeat_fraction": round(far / scored, 6),
        "long_range_share": round(far / repeats, 6) if repeats else 0.0,
        "mean_nearest_span": round(sum(spans) / len(spans), 2) if spans else 0.0,
        "median_nearest_span": median,
        "dependency_density": round(far / scored, 6),
    }


def _ranks(values):
    """Average ranks (1-based) with ties shared, for Spearman."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in order[position:end + 1]:
            ranks[index] = average
        position = end + 1
    return ranks


def pearson(xs, ys):
    """Pearson correlation, or ``None`` when it is undefined."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    sx = math.sqrt(sum(v * v for v in dx))
    sy = math.sqrt(sum(v * v for v in dy))
    if sx == 0.0 or sy == 0.0:
        return None
    return round(sum(a * b for a, b in zip(dx, dy)) / (sx * sy), 6)


def spearman(xs, ys):
    """Spearman rank correlation, or ``None`` when it is undefined."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(_ranks(list(xs)), _ranks(list(ys)))


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #

def forward_kwargs_of(model):
    return ("cache_position",) if forward_accepts(model, "cache_position") else ()


@torch.no_grad()
def score_suffix(model, cache, prefix_len, target_ids, n, forward_kwargs=()):
    """Teacher-forced mean NLL of ``n`` suffix tokens against ``cache``.

    ``target_ids[0]`` is fed at absolute position ``prefix_len`` and every true
    token is appended to the cache afterwards, exactly as a decoder would see
    them.  The first token has no predecessor inside the suffix, so ``n - 1``
    tokens are scored.
    """
    device = target_ids.device
    kept = cache.get_seq_length()
    total, count = 0.0, 0
    current = target_ids[0].view(1, 1)
    for step in range(n):
        position = torch.tensor([[prefix_len + step]], device=device)
        call = dict(past_key_values=cache, position_ids=position, use_cache=True)
        if "cache_position" in forward_kwargs:
            call["cache_position"] = torch.arange(kept + step, kept + step + 1,
                                                  device=device)
        logits = model(current, **call).logits[:, -1, :].float()
        if step + 1 < n:
            target = target_ids[step + 1]
            total += float(-torch.log_softmax(logits, dim=-1)[0, target])
            count += 1
            current = target.view(1, 1)
    return total / max(1, count)


def clone_cache(layers):
    """A fresh ``DynamicCache`` holding exactly ``layers``' key/value tensors."""
    from transformers import DynamicCache

    clone = DynamicCache()
    for index, (keys, values) in enumerate(layers):
        clone.update(keys, values, index)
    return clone


def retention_config(args, budget):
    return RetentionConfig(
        budget=budget, observation_window=args.obs, attention_sinks=args.nsink,
        pool=args.pool, dilate=args.dilate, lex_cap=args.lex_cap,
        chain_hops=0, prefill_chunk=args.prefill_chunk,
        key_chunk=args.key_chunk, scoring=args.scoring)


def peak_memory_mib(device):
    """Peak device memory since the last reset (0 on CPU)."""
    if device.type == "cuda":
        return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    return 0.0


def peak_rss_mib():
    """Peak resident set size of this process, for CPU-only smoke runs."""
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def measure_level(model, ids, args, configs, forward_kwargs, device, level, density):
    """NLL of the suffix for Full-KV and for every budget on one level."""
    prefix_len = args.length - args.suffix
    if prefix_len <= 0:
        raise SystemExit("--suffix must be smaller than --length")
    prefix = ids[:, :prefix_len]
    suffix = ids[0, prefix_len:]
    budgets = sorted(configs)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    prefill_start = time.time()
    cache, _logits, captured = prefill_capture(model, prefix, args.obs,
                                               args.prefill_chunk)
    prefill_s = time.time() - prefill_start
    layers = [(layer.keys, layer.values) for layer in cache.layers]
    # The selection scores must be read before suffix scoring appends to the
    # cache; they do not depend on the budget, so one pass serves every arm.
    shared_scores = (observation_scores(model, captured, cache, prefix_len,
                                        configs[budgets[0]])
                     if args.shared_prefill else None)

    rows = []

    def record(arm, budget, kept, nll, seconds):
        row = {
            "level": level, "arm": arm, "budget": budget,
            "kept_slots": int(kept), "slots_per_token": round(kept / args.length, 5),
            "nll": round(nll, 5), "perplexity": round(math.exp(min(nll, 20)), 4),
            "prefill_s": round(seconds, 2), "peak_mib": peak_memory_mib(device),
            "peak_rss_mib": peak_rss_mib(),
            "suffix": args.suffix, "length": args.length,
            "density_proxy": density,
        }
        rows.append(row)
        return row

    full_nll = score_suffix(model, cache, prefix_len, suffix, args.suffix,
                            forward_kwargs)
    record("full", None, cache.get_seq_length(), full_nll, prefill_s)
    for budget in budgets:
        arm_start = time.time()
        if args.shared_prefill:
            working = clone_cache(layers)
            indices = select_indices(shared_scores, working, prefix_len,
                                     configs[budget])
            prune_cache(working, indices)
            arm_prefill_s = prefill_s
        else:
            working, _ignored = compile_bounded_cache(model, prefix, configs[budget])
            arm_prefill_s = time.time() - arm_start
        nll = score_suffix(model, working, prefix_len, suffix, args.suffix,
                           forward_kwargs)
        row = record(f"b{budget}", budget, working.get_seq_length(), nll,
                     arm_prefill_s)
        row["nll_ratio_to_full"] = round(nll / full_nll, 5) if full_nll else None
        row["ppl_ratio_to_full"] = round(
            math.exp(min(nll, 20)) / math.exp(min(full_nll, 20)), 5) if full_nll else None
        del working
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for row in rows:
        row.setdefault("nll_ratio_to_full", 1.0)
        row.setdefault("ppl_ratio_to_full", 1.0)
    return rows


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #

def summarize(rows, budgets, levels):
    """Per-budget correlation between the density proxy and the NLL ratio."""
    full = {row["level"]: row for row in rows if row["arm"] == "full"}
    summary = {
        "levels": list(levels),
        "budgets": sorted(budgets),
        "full_kv": {level: {"nll": full[level]["nll"],
                            "perplexity": full[level]["perplexity"]}
                    for level in levels if level in full},
        "by_budget": {},
        "density_proxy_formula": (
            "dependency_density = fraction of scored positions whose nearest "
            "earlier copy of their k-gram is more than `near_window` tokens "
            "back (see the module docstring)"),
    }
    for budget in sorted(budgets):
        arm = f"b{budget}"
        arm_rows = {row["level"]: row for row in rows if row["arm"] == arm}
        present = [level for level in levels if level in arm_rows]
        densities = [arm_rows[level]["density_proxy"]["dependency_density"]
                     for level in present]
        ratios = [arm_rows[level]["nll_ratio_to_full"] for level in present]
        summary["by_budget"][arm] = {
            "budget": int(budget),
            "levels": present,
            "density_proxy": densities,
            "nll_ratio_to_full": ratios,
            "mean_nll_ratio": (round(sum(ratios) / len(ratios), 5) if ratios else None),
            "pearson_r": pearson(densities, ratios),
            "spearman_r": spearman(densities, ratios),
            "n_levels": len(present),
        }
    if len(levels) > 1:
        def density_of(level):
            row = full.get(level)
            return row["density_proxy"]["dependency_density"] if row else 0.0

        summary["density_order_highest_first"] = sorted(
            levels, key=density_of, reverse=True)
        summary["expected_order_highest_first"] = ["natural", "paragraph",
                                                   "sentence", "token"]
        summary["expected_order_note"] = (
            "prediction 3's convention, reported next to the measured order: "
            "natural text carries the most long-range structure and a full "
            "token permutation carries none.  Note that a block-level "
            "permutation moves repeated n-grams *beyond* the near window "
            "instead of deleting them, so the far-repeat proxy can be larger "
            "for a paragraph or sentence shuffle than for the corpus order; "
            "that is a property of this proxy, and the per-budget correlation "
            "below is what the experiment reports")
    return summary


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="NLL ratio to Full-KV across dependency-density levels.")
    parser.add_argument("--model", default=os.environ.get("QCC_MODEL", "meta-llama/Llama-3.2-1B-Instruct"))
    parser.add_argument("--length", type=int, default=32768)
    parser.add_argument("--suffix", type=int, default=256)
    parser.add_argument("--budgets", nargs="+", type=int, default=[4096, 8192])
    parser.add_argument("--levels", nargs="+", default=list(LEVELS),
                        choices=list(LEVELS))
    parser.add_argument("--corpus", nargs="+", default=list(DEFAULT_CORPUS),
                        help="local text files or globs (never downloaded)")
    parser.add_argument("--synthetic", action="store_true",
                        help="use the built-in dependency-structured corpus")
    parser.add_argument("--min-bytes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ngram", type=int, default=5)
    parser.add_argument("--near-window", type=int, default=512)
    parser.add_argument("--obs", type=int, default=64)
    parser.add_argument("--nsink", type=int, default=4)
    parser.add_argument("--pool", type=int, default=7)
    parser.add_argument("--dilate", type=int, default=9)
    parser.add_argument("--lex-cap", type=int, default=0,
                        help="lexical anchors need a question; 0 for language modelling")
    parser.add_argument("--scoring", default="last", choices=["last", "mean", "max"])
    parser.add_argument("--prefill-chunk", type=int, default=8192)
    parser.add_argument("--key-chunk", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--shared-prefill", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="select every budget from one exact prefill of the "
                             "level instead of re-prefilling per budget")
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.trust_remote_code:
        _ensure_remote_code_compat()
    from transformers import AutoTokenizer

    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=args.trust_remote_code)
    model = load_hf_causal_lm(args.model, dtype=args.dtype, device=args.device,
                              trust_remote_code=args.trust_remote_code,
                              load_in_4bit=args.load_4bit,
                              attn_implementation=args.attn_impl).eval()
    forward_kwargs = forward_kwargs_of(model)
    configs = {budget: retention_config(args, budget) for budget in sorted(args.budgets)}

    if args.synthetic:
        text = synthetic_corpus(args.length + args.suffix, tokenizer, args.seed)
        corpus_info = {"synthetic": True, "sources": [], "files": 0,
                       "chars": len(text)}
    else:
        text, files = load_corpus_text(args.corpus, args.min_bytes)
        corpus_info = {"synthetic": False, "sources": list(args.corpus),
                       "files": files, "chars": len(text)}
    pool, unit_bounds, group_bounds = build_pool_from_pairs(
        corpus_units(text, tokenizer), args.length)
    sequences = level_sequences(pool, unit_bounds, group_bounds, args.seed,
                                args.levels)
    print(f"[corpus] synthetic={corpus_info['synthetic']} files={corpus_info['files']} "
          f"chars={corpus_info['chars']} pool={len(pool)} "
          f"sentence_blocks={len(unit_bounds) - 1} "
          f"paragraph_blocks={len(group_bounds) - 1}", flush=True)

    rows = []
    for level in args.levels:
        ids = torch.tensor([sequences[level]], dtype=torch.long, device=device)
        density = dependency_density(sequences[level], args.ngram, args.near_window)
        print(f"[{level}] density={density['dependency_density']:.4f} "
              f"repeat={density['repeat_fraction']:.4f} "
              f"span={density['mean_nearest_span']:.1f}", flush=True)
        with fixed_rope_length(model, args.length):
            level_rows = measure_level(model, ids, args, configs, forward_kwargs,
                                       device, level, density)
        rows.extend(level_rows)
        for row in level_rows:
            print(f"  {row['arm']:<7} kept={row['kept_slots']:>6d} "
                  f"nll={row['nll']:.5f} ppl={row['perplexity']:.3f} "
                  f"ratio={row['nll_ratio_to_full']}", flush=True)
        del ids

    summary = summarize(rows, args.budgets, args.levels)
    payload = {
        "model": args.model,
        "label": args.label or Path(args.model).name,
        "config": {**vars(args), "levels": list(args.levels),
                   "budgets": sorted(args.budgets)},
        "corpus": corpus_info,
        "density_proxy_formula": (
            "see dependency_density(): dependency_density = fraction of scored "
            "positions whose nearest earlier copy of their ngram-gram is more "
            "than near_window tokens back"),
        "summary": summary,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print("\n" + json.dumps(summary, indent=1))
    print("wrote", out)
    return summary


if __name__ == "__main__":
    main()
