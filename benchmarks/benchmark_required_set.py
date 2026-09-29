"""Where does bounded retention collapse? Required set vs retained budget.

``docs/MECHANISM.md`` states two falsifiable predictions for the main line:

1. **The required slot count scales with the number of competing items, not with
   the context length.**  Grow ``L`` at a fixed task: quality is flat.  Grow the
   number of items the answer depends on (or the number of items competing with
   it): the budget needed to match Full-KV grows with it.
2. **Failure is a cliff once the required set exceeds the budget.**  A task whose
   answer needs ``m`` scattered items must collapse when ``m x tokens-per-item``
   exceeds the retained budget.

This runner sweeps exactly those two axes on three synthetic families, all
generated from a seed so a run is reproducible:

``multikey``
    The RULER ``niah_multikey_k`` shape, parameterised by ``k``: ``k`` keys with
    values scattered through filler, all sharing the surface form
    ``trace-<hex>``, and a question about one of them.  ``--items`` is ``k``.
    Score: exact match of the asked value (the headline); character-level F1 of
    the best numeric candidate is reported as partial credit.  Two axes grow the
    competition for the asked key:

    * ``--items`` (legacy shape): ``k`` keys in the context, ``k - 1`` of them
      competing with the asked one.  On a real 1B model ``k <= 16`` is answered
      from a 512-slot cache, so this axis alone does not stress retention.
    * ``--distractors N``: one asked key plus ``N`` *other* keys of the same
      surface form (``trace-<hex>``), i.e. ``N + 1`` keys in the context.  With
      ``N`` in the hundreds the observation window alone cannot find the asked
      key, so the anchors/selection have to do the work.  When ``--distractors``
      is given, the multikey records are generated from that axis and the cell
      label ``items`` is the total key count ``N + 1``; ``--items`` is ignored
      for ``multikey`` in that mode.

``needles``
    The capacity probe that avoids arithmetic altogether: ``k`` needles are
    placed in the context, each a distinct rare key (``trace-<hex>``) with a
    distinct value, and the question names **all** ``k`` keys and asks for all
    ``k`` values.  ``--items`` is ``k``.  The answer needs exactly ``k``
    statements, so ``required_set_tokens`` (= ``k x`` tokens per statement) is a
    known quantity.  Score: partial recall, exactly RULER's ``string_match_all``
    (mean fraction of the ``k`` references present, case-insensitive substring),
    plus ``exact_set_match`` (all ``k`` present) as the strict headline.

``aggregate``
    ``m`` numbers scattered through filler; the question asks for their sum
    (small integers, so the sum is unambiguous) or for the count of items with a
    given status (``--aggregate sum|count``).  ``--items`` is ``m``.  Score:
    exact match of the numeric answer.  Kept for continuity, but note that a
    real 1B model cannot add, so ``aggregate sum`` with ``m >= 2`` measures
    arithmetic and is reported as ``full_kv_unsolved`` rather than as a cache
    result.

All families are run against a matched **Full-KV** arm through
``prefill_capture`` and against a bounded arm through the packaged retention API
(``RetentionConfig`` + ``compile_bounded_cache``), with the same greedy decode
loop for every arm.  ``--budgets`` is swept; ``--lengths`` optionally repeats the
whole sweep at several context lengths, which is what prediction 1 needs
(``required_budget`` must not grow with ``L``).

The derived field the experiment exists for is ``required_budget`` per
``(family, items)``: the smallest tested budget whose accuracy matches the
Full-KV arm's accuracy on that record set, or ``null`` when no tested budget
does.  The multikey competition axis gets the same field per distractor count in
``summary["required_budget_by_distractors"]``.  When several lengths are swept,
``summary["by_length"][L]`` carries the per-length value and the top-level
``summary["required_budget"]`` is the largest across lengths (the budget that
matched at *every* tested length).

Making the cliff measurable
---------------------------

Prediction 2 is not assumed here, it is measured.  Every bounded row carries
``budget_covers_required_set = required_set_tokens <= kept_slots`` -- the
*measured* retained width of the bounded arm, not the nominal budget -- and
``summary["budget_covers_required_set"]`` reports accuracy (and partial recall)
**conditional on that split**, overall, per family and per cell, with the cell
and row counts.  That conditional block is the operational form of prediction 2:
if the prediction holds, accuracy is high in the ``covered`` side and collapses
in the ``not_covered`` side, and the separation between the two means is the
size of the cliff.  Nothing in the derivation assumes it: the two sides are
whatever the sweep produced.

``required_set_tokens`` is the tokens the item statements occupy, i.e.
``items x tokens_per_item`` where ``tokens_per_item`` is the mean tokens of one
item statement and ``items`` counts the statements the answer's support has to
resolve (``k`` needles, ``m`` aggregate items, ``key_count`` multikey keys).
``statement_tokens`` remains the tokens of the whole statement block, so
``required_set_tokens == statement_tokens`` up to rounding; earlier revisions
multiplied the block by ``items`` a second time and so overstated the required
set by that factor, which would have made ``budget_covers_required_set`` almost
never true.  The sweep only produces a meaningful cliff where
``items x tokens_per_item`` actually crosses the budget: with ~13-20 tokens per
statement for a Llama-3.2 tokenizer, ``k = 32`` needles need ~400-600 tokens, so
budgets below that are the interesting ones and larger ``k`` values extend the
axis past 8,192 slots.

Two derived fields keep a model failure from being read as a cache result.  When
the Full-KV arm scores zero on a cell, ``required_budget`` for that cell stays
``null`` (there is nothing for a bounded budget to match) and the cell is listed
under ``summary["full_kv_unsolved"]`` with its record count and the reason.  The
same flag is on ``summary["cells"][family][items]``.

Everything here is a measurement harness: it never modifies the checkpoint, and
on CPU it can be exercised end to end with a tiny randomly-initialised model
(``--device cpu``), which is how the unit tests drive it.
"""

from __future__ import annotations

import os

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qcc_transformer.hf_loading import _ensure_remote_code_compat, load_hf_causal_lm
from qcc_transformer.retention import (RetentionConfig, compile_bounded_cache,
                                       fixed_rope_length, forward_accepts,
                                       lexical_anchors, observation_scores,
                                       prefill_capture, prune_cache, select_indices)

# --------------------------------------------------------------------------- #
# task generation
# --------------------------------------------------------------------------- #

#: Filler sentences contain no digits, so in ``aggregate`` the only integers in
#: the prompt are the item values and in ``multikey`` the only long digit runs
#: are the values.
FILLER_SENTENCES = (
    "The report continues with a description of the surrounding conditions.",
    "Several observations were recorded during the earlier part of the session.",
    "A short pause followed while the instrument settled into a stable state.",
    "The notes mention the same procedure in a slightly different wording.",
    "Nothing in this passage changes the outcome of the question that follows.",
    "The remaining details were considered unremarkable and were left out.",
    "A careful reading shows that the surrounding text is mostly decorative.",
    "The paragraph was kept short so that the sequence stays easy to follow.",
    "Further remarks about the weather were added to fill the remaining space.",
    "The assistant recorded the result and moved on to the next section.",
    "Long passages of ordinary prose surround the statements that matter here.",
    "The order of the surrounding sentences carries no information at all.",
    "A reader may safely skip the filler and look for the relevant statement.",
    "The document repeats this kind of neutral sentence many times over.",
    "Most of the words here exist only to separate the informative statements.",
    "The setting was described at length without adding anything essential.",
    "Background material of this sort is common in long collections of notes.",
    "The next few sentences describe nothing that the question will ask about.",
    "An idle remark about the schedule closes the current fragment of text.",
    "The surrounding prose is deliberately free of numbers and identifiers.",
)

PAD_WORDS = ("and", "then", "also", "with", "the", "for", "near", "after",
             "while", "before", "under", "over", "into", "onto", "beside")

ITEM_NAMES = (
    "birch", "cedar", "maple", "willow", "aspen", "hazel", "alder", "rowan",
    "larch", "spruce", "juniper", "cypress", "poplar", "sycamore", "hemlock",
    "magnolia", "dogwood", "redwood", "chestnut", "walnut", "pecan", "almond",
    "olive", "laurel", "myrtle", "linden", "hornbeam", "mulberry", "tamarack",
    "ironwood", "basswood", "butternut", "sassafras", "persimmon", "catalpa",
)

STATUSES = ("amber", "cobalt", "crimson", "emerald", "indigo", "ochre",
            "russet", "violet")

_ARITH_RE = re.compile(r"-?\d[\d,]*")
_HEX = "0123456789abcdef"


def encode_ids(tokenizer, text):
    """Token ids of ``text`` with no special tokens, for HF or stub tokenizers."""
    try:
        return list(tokenizer(text, add_special_tokens=False).input_ids)
    except TypeError:
        return list(tokenizer.encode(text, add_special_tokens=False))


def decode_ids(tokenizer, ids):
    """Inverse of :func:`encode_ids` for HF or stub tokenizers."""
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(list(ids), skip_special_tokens=False)
    return " ".join(str(i) for i in ids)


def parse_numbers(text):
    """Every integer-looking surface in ``text``, normalized (commas removed).

    ``"1,234"`` and ``"1234"`` both normalize to ``"1234"`` so a prediction that
    formats the answer differently still counts as an exact match.
    """
    out = []
    for match in _ARITH_RE.finditer(text or ""):
        raw = match.group(0).replace(",", "")
        try:
            out.append(str(int(raw)))
        except ValueError:  # pragma: no cover - regex guarantees digits
            continue
    return out


def char_f1(candidate, expected):
    """Character-multiset F1 between two numeric strings, in ``[0, 1]``.

    Partial credit for a numerically wrong but textually close answer: ``"7541"``
    against ``"7549132"`` scores well above zero, ``"0000"`` scores zero.
    """
    if not candidate or not expected:
        return 0.0
    left, right = Counter(candidate), Counter(expected)
    overlap = sum((left & right).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(left.values())
    recall = overlap / sum(right.values())
    return 2 * precision * recall / (precision + recall)


def score_prediction(prediction, expected, distractors=()):
    """Exact-match scoring of a numeric answer plus partial credit.

    ``correct`` is the headline: the expected value appears as an integer
    somewhere in the prediction.  ``parsed`` is the first integer the model
    emitted (what a strict parser would read).  ``partial`` is the best
    character F1 over the prediction's integer candidates and is only reported
    as partial credit.  ``distractor_hit`` flags the mechanism-specific error:
    answering with another item's value instead of the asked one.
    """
    candidates = parse_numbers(prediction)
    correct = expected in candidates
    partial = max((char_f1(c, expected) for c in candidates), default=0.0)
    distractor_hit = (not correct) and any(d in candidates for d in distractors)
    return {
        "parsed": candidates[0] if candidates else None,
        "correct": bool(correct),
        "partial": round(partial, 4),
        "distractor_hit": bool(distractor_hit),
        "candidates": candidates[:8],
    }


def reference_recall(references, prediction):
    """RULER's ``string_match_all``: mean fraction of the references present.

    A reference counts when it appears in the prediction as a case-insensitive
    substring, which is exactly the RULER/HuggingFace ``string_match_all`` rule
    (``mean(out in prediction for out in references)``).  Multi-reference tasks
    (``needles``) are scored with this; single-reference tasks get ``1.0`` when
    the one reference is present and ``0.0`` otherwise.
    """
    refs = [str(reference) for reference in references if str(reference)]
    if not refs:
        return 0.0
    low = (prediction or "").lower()
    return round(sum(1.0 for reference in refs if reference.lower() in low) / len(refs), 4)


def exact_set_match(references, prediction):
    """Whether **every** reference appears in the prediction (strict recall)."""
    refs = [str(reference) for reference in references if str(reference)]
    return bool(refs) and reference_recall(refs, prediction) >= 1.0


def score_needles(prediction, references):
    """Multi-reference scoring with RULER's ``string_match_all``.

    ``partial_recall`` is the headline graded number: the fraction of the ``k``
    needle values present in the prediction.  ``correct``/``exact_set_match`` is
    the strict all-or-nothing version.  ``partial`` mirrors ``partial_recall``
    so every family's rows expose the same field name for graded quality.
    """
    refs = [str(reference) for reference in references if str(reference)]
    low = (prediction or "").lower()
    present = [reference for reference in refs if reference.lower() in low]
    missing = [reference for reference in refs if reference not in present]
    recall = reference_recall(refs, prediction)
    candidates = parse_numbers(prediction)
    return {
        "parsed": candidates[0] if candidates else None,
        "correct": bool(refs) and not missing,
        "partial": recall,
        "partial_recall": recall,
        "exact_set_match": bool(refs) and not missing,
        "distractor_hit": False,
        "candidates": candidates[:8],
        "references": refs,
        "present": present,
        "missing": missing,
    }


def score_record(record, prediction):
    """Score a prediction the way its family defines the answer.

    Every branch returns the same fields, so a row always carries ``correct``
    (strict), ``partial`` (graded), ``partial_recall`` (fraction of the answer's
    references present), ``exact_set_match``, ``present`` and ``missing``.  For
    the single-answer families the answer has exactly one reference, so
    ``partial_recall`` and ``exact_set_match`` coincide with ``correct`` while
    ``partial`` keeps the character-level partial credit.
    """
    if record["family"] == "needles":
        return score_needles(prediction, record["references"])
    expected = str(record["expected"])
    scored = score_prediction(prediction, expected, record.get("distractors") or ())
    present = [expected] if scored["correct"] else []
    scored.update({
        "partial_recall": 1.0 if scored["correct"] else 0.0,
        "exact_set_match": bool(scored["correct"]),
        "references": [expected],
        "present": present,
        "missing": [] if scored["correct"] else [expected],
    })
    return scored


def record_rng(family, items, seed, aggregate="sum", distractors=None):
    """A seed-stable generator: the same arguments always rebuild the same task.

    The legacy key (no ``--distractors``) is unchanged, so an ``--items`` sweep
    rebuilds exactly the records earlier runs used; the distractor count is
    appended only when the distractor axis is active.
    """
    key = f"qcc-required-set:{family}:{items}:{aggregate}:{seed}"
    if distractors is not None:
        key += f":d{distractors}"
    return random.Random(key)


def item_names(rng, count):
    """``count`` distinct item names (extends the pool past :data:`ITEM_NAMES`)."""
    if count <= len(ITEM_NAMES):
        return rng.sample(ITEM_NAMES, count)
    names = list(ITEM_NAMES)
    index = 0
    while len(names) < count:
        names.append(f"{ITEM_NAMES[index % len(ITEM_NAMES)]}{index // len(ITEM_NAMES) + 1}")
        index += 1
    return names


def distinct_keys(rng, count):
    """``count`` distinct ``trace-<hex>`` surfaces, at least one letter each.

    The shared surface form is the point of the ``multikey`` family: every key
    looks like every other key, so selection cannot be a lexical match on the
    key name alone.
    """
    keys, seen = [], set()
    while len(keys) < count:
        suffix = "".join(rng.choice(_HEX) for _ in range(6))
        key = "trace-" + suffix
        # at least one letter keeps the key from looking like the answer's digits
        if key not in seen and any(character.isalpha() for character in suffix):
            seen.add(key)
            keys.append(key)
    return keys


def distinct_values(rng, count, low=100000, high=999999):
    """``count`` distinct six-digit values, the answer alphabet of every family."""
    values, seen = [], set()
    while len(values) < count:
        value = rng.randint(low, high)
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def _render(intro, statements, question, filler_count, fractions, extra_words,
            filler_pool):
    """Assemble the prompt text for a given amount of filler."""
    parts = [intro]
    placed = 0
    for index in range(filler_count):
        while placed < len(statements) and fractions[placed] * filler_count <= index:
            parts.append(statements[placed])
            placed += 1
        sentence = filler_pool[index % len(filler_pool)]
        if extra_words and index == filler_count - 1:
            words = [PAD_WORDS[i % len(PAD_WORDS)] for i in range(extra_words)]
            words[0] = words[0].capitalize()
            sentence = sentence + " " + " ".join(words) + "."
        parts.append(sentence)
    while placed < len(statements):
        parts.append(statements[placed])
        placed += 1
    parts.append(question)
    return " ".join(parts)


def assemble_prompt(intro, statements, question, target_tokens, tokenizer,
                    rng, filler_pool=FILLER_SENTENCES, max_filler=8000,
                    placement="scatter"):
    """Scatter ``statements`` through filler and land on ~``target_tokens``.

    The item statements are placed at random positions through the filler body
    (as in RULER's needle placement) and the question is always the last
    segment, so the observation window sees the question.  The filler count is
    binary-searched and then padded with one-token filler words, so the final
    prompt is within a couple of tokens of the target even though the fillers
    tokenize to different lengths.

    Returns ``(text, n_tokens, statement_tokens)``.
    """
    if placement == "recency":
        # Oracle-placement control: every item statement lands immediately
        # before the question, so the whole required set sits inside the recency
        # channel of any budget that can hold the statement block
        # (``recent_fraction x budget >= statement_tokens``).  Bounded accuracy
        # that still trails Full-KV under this placement cannot be explained by
        # the required tokens falling outside the retained support.
        fractions = [1.0] * len(statements)
    elif placement == "scatter":
        fractions = sorted(rng.random() for _ in statements)
    else:
        raise ValueError(f"unknown placement {placement!r}")
    base = len(encode_ids(tokenizer, intro)) + len(encode_ids(tokenizer, question))
    statement_tokens = len(encode_ids(tokenizer, " ".join(statements)))
    room = target_tokens - base - statement_tokens

    def tokens_for(filler_count, extra_words=0):
        text = _render(intro, statements, question, filler_count, fractions,
                       extra_words, filler_pool)
        return len(encode_ids(tokenizer, text))

    if room <= 0:
        # The items alone already exceed the target: keep every item and let the
        # record report its true prompt length.
        text = _render(intro, statements, question, 0, fractions, 0, filler_pool)
        return text, len(encode_ids(tokenizer, text)), statement_tokens

    low, high = 0, max_filler
    while low < high:                       # largest filler count that fits
        mid = (low + high + 1) // 2
        if tokens_for(mid) <= target_tokens:
            low = mid
        else:
            high = mid - 1
    filler_count = low
    deficit = target_tokens - tokens_for(filler_count)
    extra_words = 0
    for _ in range(4):
        if deficit <= 0:
            break
        extra_words += max(1, deficit)
        text = _render(intro, statements, question, filler_count, fractions,
                       extra_words, filler_pool)
        deficit = target_tokens - len(encode_ids(tokenizer, text))
    text = _render(intro, statements, question, filler_count, fractions,
                   extra_words, filler_pool)
    return text, len(encode_ids(tokenizer, text)), statement_tokens


def multikey_record(items, seed, target_tokens, tokenizer, distractors=None,
                    placement="scatter"):
    """Keys of one surface form, one asked value, the other keys competing.

    Two shapes, both seeded and reproducible:

    * ``distractors=None`` (the legacy ``--items`` shape): ``items`` keys in the
      context, one of them asked, so ``items - 1`` keys compete with it.
    * ``distractors=N``: one asked key plus ``N`` other keys of the same surface
      form, so the context holds ``N + 1`` keys.  ``items`` is then the total key
      count the caller wants recorded (``N + 1``); the question still asks for a
      single value, and every other key is a competitor the selection has to
      survive.

    ``record["distractors"]`` stays the list of *values* of the competing keys
    (what :func:`score_prediction` flags a distractor hit with);
    ``record["distractor_count"]`` is how many competing keys there are.
    """
    if distractors is None:
        key_count = items
        distractor_count = max(0, items - 1)
    else:
        distractor_count = int(distractors)
        key_count = distractor_count + 1
        items = key_count
    rng = record_rng("multikey", items, seed, distractors=distractors)
    names = item_names(rng, key_count)
    keys = distinct_keys(rng, key_count)
    values = distinct_values(rng, key_count)
    target_index = rng.randrange(key_count)
    statements = [f"One of the magic numbers for {key} is {value}."
                  for key, value in zip(keys, values)]
    intro = ("A list of records is given below. Each record names a key and the "
             "magic number assigned to that key.")
    question = (f"What is the magic number for {keys[target_index]} mentioned "
                "above? Answer with a single number.")
    prompt, n_tokens, statement_tokens = assemble_prompt(
        intro, statements, question, target_tokens, tokenizer, rng,
        placement=placement)
    return {
        "family": "multikey", "items": items, "seed": seed,
        "aggregate": None, "prompt": prompt, "prompt_tokens": n_tokens,
        "statement_tokens": statement_tokens, "placement": placement,
        "statements": statements,
        "expected": str(values[target_index]),
        "distractors": [str(v) for i, v in enumerate(values) if i != target_index],
        "distractor_count": distractor_count, "key_count": key_count,
        "keys": keys, "values": [str(v) for v in values],
        "target_key": keys[target_index], "target_index": target_index,
        "names": names,
    }


def needles_record(items, seed, target_tokens, tokenizer, placement="scatter"):
    """``k`` needles, each a distinct key with a distinct value; ask for all.

    This is the capacity probe: the answer needs every one of the ``k``
    statements, so ``required_set_tokens`` is a known quantity rather than a
    guess, and no arithmetic is involved -- the model only has to reproduce the
    values.  The question names all ``k`` keys (in a shuffled order, so the
    order of the answer is not a cue), which also means the observation window
    sees the keys that have to be looked up.
    """
    rng = record_rng("needles", items, seed)
    keys = distinct_keys(rng, items)
    values = distinct_values(rng, items)
    order = list(range(items))
    rng.shuffle(order)
    statements = [f"One of the magic numbers for {key} is {value}."
                  for key, value in zip(keys, values)]
    intro = ("A list of records is given below. Each record names a key and the "
             "magic number assigned to that key.")
    asked = ", ".join(keys[index] for index in order)
    question = (f"What are the magic numbers for all of these keys mentioned "
                f"above: {asked}? Answer with all {items} numbers separated by "
                "commas.")
    prompt, n_tokens, statement_tokens = assemble_prompt(
        intro, statements, question, target_tokens, tokenizer, rng,
        placement=placement)
    references = [str(value) for value in values]
    return {
        "family": "needles", "items": items, "seed": seed,
        "aggregate": None, "distractors": None, "distractor_count": None,
        "key_count": items, "prompt": prompt, "prompt_tokens": n_tokens,
        "statement_tokens": statement_tokens, "placement": placement,
        "statements": statements,
        "expected": ", ".join(references), "references": references,
        "keys": keys, "values": references,
        "ask_order": [keys[index] for index in order], "ask_order_indices": order,
    }


def aggregate_record(items, seed, target_tokens, tokenizer, aggregate="sum",
                     placement="scatter"):
    """``m`` scattered items; ask for their sum or for a status count."""
    rng = record_rng("aggregate", items, seed, aggregate)
    names = item_names(rng, items)
    if aggregate == "sum":
        values = [rng.randint(1, 9) for _ in range(items)]
        statuses = [rng.choice(STATUSES) for _ in range(items)]
        statements = [f"The {name} item holds {value} units."
                      for name, value in zip(names, values)]
        intro = "A list of items is given below."
        question = ("What is the total number of units held by all of the items "
                    "listed above? Answer with a single number.")
        expected = str(sum(values))
        distractors = [str(v) for v in values]
    elif aggregate == "count":
        target_status = rng.choice(STATUSES)
        others = [s for s in STATUSES if s != target_status]
        # at least one item carries another status, so the criterion matters
        upper = items if items < 2 else max(1, items - 1)
        count = rng.randint(1, upper)
        statuses = [target_status] * count + [
            others[i % len(others)] for i in range(items - count)]
        rng.shuffle(statuses)
        values = [rng.randint(1, 9) for _ in range(items)]
        statements = [f"The {name} item has status {status}."
                      for name, status in zip(names, statuses)]
        intro = "A list of items is given below."
        question = (f"How many of the items listed above have status "
                    f"{target_status}? Answer with a single number.")
        expected = str(statuses.count(target_status))
        # Every count other than the answer is a plausible distractor.
        distractors = [str(n) for n in range(1, items + 1) if str(n) != expected]
    else:
        raise ValueError(f"unknown aggregate mode {aggregate!r}")
    prompt, n_tokens, statement_tokens = assemble_prompt(
        intro, statements, question, target_tokens, tokenizer, rng,
        placement=placement)
    return {
        "family": "aggregate", "items": items, "seed": seed,
        "aggregate": aggregate, "prompt": prompt, "prompt_tokens": n_tokens,
        "statement_tokens": statement_tokens, "placement": placement,
        "statements": statements, "expected": expected,
        "distractors": distractors, "names": names,
        "distractor_count": None, "key_count": None,
        "values": [str(v) for v in values], "statuses": list(statuses),
        "target_status": target_status if aggregate == "count" else None,
    }


def build_record(family, items, seed, target_tokens, tokenizer, aggregate="sum",
                 distractors=None, placement="scatter"):
    """One task record of the requested family (deterministic in the seed)."""
    if family == "multikey":
        return multikey_record(items, seed, target_tokens, tokenizer, distractors,
                               placement=placement)
    if family == "needles":
        return needles_record(items, seed, target_tokens, tokenizer,
                              placement=placement)
    if family == "aggregate":
        return aggregate_record(items, seed, target_tokens, tokenizer, aggregate,
                                placement=placement)
    raise ValueError(f"unknown family {family!r}")


def decode_budget(family, items, requested=None):
    """Greedy decode budget for one record.

    ``--max-new`` wins when given.  Otherwise the default is 24 tokens for the
    single-answer families and ``max(32, 8 x k)`` for ``needles``, because the
    answer there is ``k`` values: a decode budget that cannot hold the whole
    answer would turn ``exact_set_match`` into a measurement of the decode loop
    rather than of the cache.
    """
    if requested is not None:
        return int(requested)
    if family == "needles":
        return max(32, 8 * int(items))
    return 24


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #

def forward_kwargs_of(model):
    """Decode-time kwargs the model's own forward advertises."""
    return ("cache_position",) if forward_accepts(model, "cache_position") else ()


@torch.no_grad()
def greedy(model, cache, first, length, max_new, eos_ids, forward_kwargs=()):
    """Greedy decode from a prefilled (full or compiled) cache.

    ``length`` is the true prompt length, so the next token gets its absolute
    position; ``cache_position`` indexes the cache itself, which is shorter for
    a compiled one.  Every arm uses this same loop.
    """
    kept = cache.get_seq_length()
    mask = torch.ones(1, kept + 1, device=first.device, dtype=torch.long)
    generated, current, start = [], first, time.time()
    for step in range(max_new):
        position = torch.tensor([[length + step]], device=first.device)
        call = dict(past_key_values=cache, attention_mask=mask,
                    position_ids=position, use_cache=True)
        if "cache_position" in forward_kwargs:
            call["cache_position"] = torch.arange(kept + step, kept + step + 1,
                                                  device=first.device)
        out = model(current, **call)
        token = out.logits[:, -1:].argmax(-1)
        generated.append(int(token))
        mask = torch.cat([mask, torch.ones(1, 1, device=first.device,
                                           dtype=torch.long)], dim=-1)
        current = token
        if int(token) in eos_ids:
            break
    return generated, time.time() - start


def eos_token_ids(model):
    ids = set()
    for source in (getattr(model.config, "eos_token_id", None),
                   getattr(getattr(model, "generation_config", None),
                           "eos_token_id", None)):
        if source is None:
            continue
        ids |= set(source) if isinstance(source, (list, tuple, set)) else {source}
    return ids


def peak_memory_mib(device):
    """Peak device memory since the last reset (0 on CPU)."""
    if device.type == "cuda":
        return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    return 0.0


def peak_rss_mib():
    """Peak resident set size of this process, for CPU-only smoke runs."""
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


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
        chain_hops=args.hops, prefill_chunk=args.prefill_chunk,
        key_chunk=args.key_chunk, scoring=args.scoring,
        anchor_mode=args.anchor_mode, min_anchor_len=args.min_anchor_len)


# --------------------------------------------------------------------------- #
# derivation and summary
# --------------------------------------------------------------------------- #

def required_budget_from_matrix(accuracy_by_budget, full_accuracy, tolerance=0.0):
    """Smallest tested budget that matches the Full-KV accuracy, else ``None``.

    ``accuracy_by_budget`` maps a tested budget to the bounded arm's accuracy on
    one record set.  A budget qualifies when its accuracy is within
    ``tolerance`` of ``full_accuracy`` and is strictly positive: a budget that
    answers nothing never counts as matching, even when the Full-KV arm also
    scores zero (in which case there is nothing to match and the result is
    ``None``).
    """
    if full_accuracy <= 0.0:
        return None
    target = max(0.0, full_accuracy - tolerance)
    qualifying = sorted(int(budget) for budget, accuracy in accuracy_by_budget.items()
                        if accuracy + 1e-9 >= target and accuracy > 0.0)
    return qualifying[0] if qualifying else None


def _nested_matrix(rows, budgets, field="items", metric="correct"):
    """Accuracy nested by family -> ``field`` -> budget (plus a ``full`` key).

    ``field`` is ``"items"`` for the main matrix (``k`` keys, ``m`` items, ``k``
    needles) and ``"distractors"`` for the competition axis (the number of other
    keys sharing the asked key's surface form).  Rows with no value on that axis
    are skipped.  ``metric`` is ``"correct"`` (the strict all-or-nothing flag) or
    any numeric per-row field such as ``"partial_recall"``.
    """
    matrix = {}
    for row in rows:
        key = row.get(field)
        if key is None:
            continue
        cell = matrix.setdefault(row["family"], {}).setdefault(str(key), {})
        arm = "full" if row["arm"] == "full" else str(row["budget"])
        value = row.get(metric)
        if value is None:
            continue
        cell.setdefault(arm, []).append(
            float(bool(value)) if metric == "correct" else float(value))
    out = {}
    for family, by_items in matrix.items():
        out[family] = {}
        for items, by_arm in by_items.items():
            out[family][items] = {arm: round(sum(v) / len(v), 4)
                                  for arm, v in by_arm.items()}
            for budget in budgets:
                out[family][items].setdefault(str(budget), None)
            out[family][items].setdefault("full", None)
    return out


def _required_budgets(rows, budgets, tolerance, field="items"):
    """``{family: {field: smallest matching budget or None}}``."""
    cells = {}
    for row in rows:
        key = row.get(field)
        if key is None:
            continue
        cells.setdefault((row["family"], key), []).append(row)
    out = {}
    for (family, items), cell_rows in cells.items():
        per_budget, full_scores = {}, []
        for row in cell_rows:
            if row["arm"] == "full":
                full_scores.append(float(bool(row["correct"])))
            else:
                per_budget.setdefault(int(row["budget"]), []).append(
                    float(bool(row["correct"])))
        accuracy = {budget: sum(values) / len(values)
                    for budget, values in per_budget.items()}
        full = sum(full_scores) / len(full_scores) if full_scores else None
        value = None if full is None else required_budget_from_matrix(
            {budget: accuracy.get(budget, 0.0) for budget in budgets}, full, tolerance)
        out.setdefault(family, {})[str(items)] = value
    return out


def _mean_field(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return round(sum(values) / len(values), 2) if values else None


def _mean_bool(rows, field):
    values = [float(bool(row[field])) for row in rows if row.get(field) is not None]
    return round(sum(values) / len(values), 4) if values else None


def _mean_float(rows, field, digits=4):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return round(sum(values) / len(values), digits) if values else None


def _truncation_rate(rows):
    """Share of rows whose decode stopped only because it hit ``max_new``.

    A row that generated its whole budget without emitting EOS was cut off, so a
    low score there measures the decode loop, not the cache.  The field is the
    readout-side control for any collapse that is not a retention failure.
    """
    values = [float(row["generated_tokens"] >= row["max_new"]) for row in rows
              if row.get("generated_tokens") is not None
              and row.get("max_new") is not None]
    return round(sum(values) / len(values), 4) if values else None


def required_coverage_of(indices, spans):
    """How much of every item statement survived selection, per (layer, head).

    ``indices`` is what :func:`qcc_transformer.retention.select_indices` returns:
    one ``[heads, slots]`` tensor per layer, and the sets differ per head.  A
    token is *available* to a fraction of those sets; a statement is *covered*
    when every one of its tokens is available everywhere (which is what the
    shared anchor list buys and what the per-head top-k alone does not).

    Returns ``(availability, covered_items)`` where ``availability`` is the mean
    per-token availability over the required spans, or ``(None, None)`` when
    there is nothing to measure.
    """
    pairs = [set(row) for tensor in indices for row in tensor.tolist()]
    if not pairs or not spans:
        return None, None
    total = sum(len(span) for span in spans)
    available = sum(sum(1 for pair in pairs if token in pair) for span in spans
                    for token in span)
    covered = sum(1 for span in spans
                  if all(all(token in pair for pair in pairs) for token in span))
    return (round(available / (total * len(pairs)), 4) if total else None), covered


def required_mass_last_query(scores, spans, device_softmax=True):
    """Share of the final query's attention that lands on the required spans.

    ``observation_scores`` returns, per layer, the raw final-query logits per
    key/value head (the singleton query axis makes the softmax denominator
    constant across keys, which is why the selection can rank by the raw score).
    Normalising those logits therefore gives the *actual* attention the last
    prompt token pays to each position, and summing over the item statements says
    how much of it reaches the answer.  This is the second half of the mechanism:
    a statement can be retained and still lose the competition for mass.

    Returns the mean over ``(layer, head)`` of that share, or ``None``.
    """
    if not spans or scores is None:
        return None
    required = torch.zeros(int(scores[0].shape[-1]), dtype=torch.bool)
    for span in spans:
        for token in span:
            if 0 <= token < required.numel():
                required[token] = True
    if not bool(required.any()):
        return None
    shares = []
    for layer_scores in scores:
        weights = torch.softmax(layer_scores.float(), dim=-1)
        shares.append(weights[:, required].sum(dim=-1))
    stacked = torch.cat(shares)
    return round(float(stacked.mean()), 6)


def row_fully_covers_required_set(row):
    """Whether the *selected* slots of a bounded row contain every item statement.

    ``None`` when the row has no measured coverage (the Full-KV arm, or a run
    through ``compile_bounded_cache``, which does not return its index set).  This
    is the measured counterpart of :func:`row_covers_required_set`: the width
    comparison says the statements would fit, this says they survived selection.
    """
    coverage = row.get("required_coverage")
    return None if row["arm"] == "full" or coverage is None else bool(coverage >= 1.0)


def required_span_coverage_block(rows):
    """Accuracy conditional on the item statements surviving selection.

    Prediction 2 proper: the row-level ``required_coverage`` is measured against
    the kept slots, and the two conditional means (statements fully retained /
    partly or wholly dropped) are reported with their counts.
    """
    bounded = [row for row in rows if row_fully_covers_required_set(row) is not None]
    covered = [row for row in bounded if row_fully_covers_required_set(row)]
    missing = [row for row in bounded if not row_fully_covers_required_set(row)]
    block = {
        "rule": ("`required_coverage` is the share of the item statements' own "
                 "tokens that survived selection, measured against the kept index "
                 "set of the bounded arm; a row is fully covered when every "
                 "statement kept every token"),
        "prediction": ("prediction 2 of docs/MECHANISM.md in its measured form: "
                       "accuracy is high when the item statements are in the "
                       "retained support, and the two means below are the cliff"),
        "covered": _coverage_side(covered),
        "not_covered": _coverage_side(missing),
        "separation": _separation(_coverage_side(covered), _coverage_side(missing)),
        "mean_coverage": _mean_float(bounded, "required_coverage"),
        "by_cell": {},
        "by_budget": {},
    }
    for family in sorted({row["family"] for row in bounded}):
        for items in sorted({row["items"] for row in bounded
                             if row["family"] == family}):
            cell = [row for row in bounded
                    if row["family"] == family and row["items"] == items]
            side_a = [row for row in cell if row_fully_covers_required_set(row)]
            side_b = [row for row in cell if not row_fully_covers_required_set(row)]
            block["by_cell"].setdefault(family, {})[str(items)] = {
                "mean_coverage": _mean_float(cell, "required_coverage"),
                "mean_items_covered": _mean_field(cell, "items_covered"),
                "items_required": _mean_field(cell, "items_required"),
                "covered": _coverage_side(side_a),
                "not_covered": _coverage_side(side_b),
                "separation": _separation(_coverage_side(side_a),
                                          _coverage_side(side_b)),
            }
    for budget in sorted({int(row["budget"]) for row in bounded
                          if row.get("budget") is not None}):
        per_budget = [row for row in bounded if int(row["budget"]) == budget]
        side_a = [row for row in per_budget if row_fully_covers_required_set(row)]
        side_b = [row for row in per_budget
                  if not row_fully_covers_required_set(row)]
        block["by_budget"][str(budget)] = {
            "mean_coverage": _mean_float(per_budget, "required_coverage"),
            "mean_items_covered": _mean_field(per_budget, "items_covered"),
            "covered": _coverage_side(side_a),
            "not_covered": _coverage_side(side_b),
            "separation": _separation(_coverage_side(side_a),
                                      _coverage_side(side_b)),
        }
    return block


def row_covers_required_set(row):
    """Whether a bounded row's retained width covers its required set.

    ``None`` for the Full-KV arm (which keeps every slot) and for rows without a
    ``required_set_tokens``.  The comparison uses the *measured* ``kept_slots``
    of the bounded arm rather than the nominal budget, so a budget the selection
    could not fill (a prompt shorter than the budget) does not count as covering
    anything.  A row that already carries the flag is trusted.
    """
    if row["arm"] == "full" or row.get("required_set_tokens") is None:
        return None
    flag = row.get("budget_covers_required_set")
    if flag is not None:
        return bool(flag)
    return bool(row["required_set_tokens"] <= row["kept_slots"])


def _coverage_side(rows):
    """Row/cell counts and conditional means for one side of the split."""
    return {
        "rows": len(rows),
        "cells": len({(row["family"], row["items"], row["length"], row["budget"])
                      for row in rows}),
        "mean_accuracy": _mean_bool(rows, "correct"),
        "mean_partial_recall": _mean_float(rows, "partial_recall"),
        "mean_kept_slots": _mean_field(rows, "kept_slots"),
        "mean_required_set_tokens": _mean_field(rows, "required_set_tokens"),
    }


def _separation(covered, not_covered):
    """``covered - not_covered`` for the two conditional means."""
    def difference(left, right):
        return None if left is None or right is None else round(left - right, 4)
    return {
        "accuracy": difference(covered["mean_accuracy"], not_covered["mean_accuracy"]),
        "partial_recall": difference(covered["mean_partial_recall"],
                                     not_covered["mean_partial_recall"]),
    }


def budget_coverage_block(rows):
    """Accuracy conditional on the retained budget covering the required set.

    This is prediction 2 in its operational form: every bounded row is split by
    ``required_set_tokens <= kept_slots`` and the two conditional means are
    reported with the row and cell counts, overall, per family, per cell
    (``family -> items``) and per budget.  The block measures the cliff instead
    of assuming it -- the two sides are whatever the sweep produced, and a
    positive ``separation`` is evidence for the prediction while a near-zero one
    is evidence against it.
    """
    bounded = [row for row in rows if row_covers_required_set(row) is not None]
    covered = [row for row in bounded if row_covers_required_set(row)]
    not_covered = [row for row in bounded if not row_covers_required_set(row)]
    block = {
        "rule": ("a bounded row's budget covers its required set when "
                 "required_set_tokens <= kept_slots, where kept_slots is the "
                 "measured retained width of the bounded arm (not the nominal "
                 "budget) and required_set_tokens is the mean tokens per item "
                 "statement times the number of items the answer needs"),
        "prediction": ("prediction 2 of docs/MECHANISM.md: accuracy is high when "
                       "the budget covers the required set and collapses when it "
                       "does not; `separation` is the measured size of the cliff"),
        "covered": _coverage_side(covered),
        "not_covered": _coverage_side(not_covered),
        "separation": _separation(_coverage_side(covered), _coverage_side(not_covered)),
        "by_family": {},
        "by_cell": {},
        "by_budget": {},
    }
    for family in sorted({row["family"] for row in bounded}):
        family_rows = [row for row in bounded if row["family"] == family]
        side_covered = [row for row in family_rows if row_covers_required_set(row)]
        side_missing = [row for row in family_rows if not row_covers_required_set(row)]
        block["by_family"][family] = {
            "covered": _coverage_side(side_covered),
            "not_covered": _coverage_side(side_missing),
            "separation": _separation(_coverage_side(side_covered),
                                      _coverage_side(side_missing)),
        }
    for family in sorted({row["family"] for row in bounded}):
        for items in sorted({row["items"] for row in bounded if row["family"] == family}):
            cell_rows = [row for row in bounded
                         if row["family"] == family and row["items"] == items]
            side_covered = [row for row in cell_rows if row_covers_required_set(row)]
            side_missing = [row for row in cell_rows if not row_covers_required_set(row)]
            block["by_cell"].setdefault(family, {})[str(items)] = {
                "covered": _coverage_side(side_covered),
                "not_covered": _coverage_side(side_missing),
                "separation": _separation(_coverage_side(side_covered),
                                          _coverage_side(side_missing)),
            }
    for budget in sorted({int(row["budget"]) for row in bounded
                          if row.get("budget") is not None}):
        budget_rows = [row for row in bounded if int(row["budget"]) == budget]
        side_covered = [row for row in budget_rows if row_covers_required_set(row)]
        side_missing = [row for row in budget_rows if not row_covers_required_set(row)]
        block["by_budget"][str(budget)] = {
            "covered": _coverage_side(side_covered),
            "not_covered": _coverage_side(side_missing),
            "separation": _separation(_coverage_side(side_covered),
                                      _coverage_side(side_missing)),
        }
    return block


def full_kv_unsolved_block(rows):
    """Cells the Full-KV arm itself scores zero on, so nothing can be matched.

    ``required_budget`` is ``null`` for these cells by construction, and this
    block says why: the model cannot do the task at full KV, so the cell is a
    model failure and not a cache result.  Reporting them explicitly keeps the
    summary from hiding "the model cannot add" behind a retention story.
    """
    cells = {}
    for row in rows:
        if row["arm"] != "full":
            continue
        cell = cells.setdefault((row["family"], row["items"]),
                                {"scores": [], "seeds": set(), "lengths": set()})
        cell["scores"].append(float(bool(row["correct"])))
        cell["seeds"].add(row["seed"])
        cell["lengths"].add(row["length"])
    out, count = {}, 0
    for (family, items), cell in sorted(cells.items()):
        accuracy = (sum(cell["scores"]) / len(cell["scores"])) if cell["scores"] else 0.0
        if accuracy > 0.0:
            continue
        out.setdefault(family, {})[str(items)] = {
            "full_accuracy": round(accuracy, 4),
            "required_budget": None,
            "records": len(cell["scores"]),
            "seeds": len(cell["seeds"]),
            "lengths": sorted(cell["lengths"]),
            "reason": ("Full-KV answered none of these records: the task, not the "
                       "cache, is the bottleneck, so required_budget is null"),
        }
        count += 1
    return {
        "rule": ("cells whose Full-KV accuracy is 0; required_budget stays null "
                 "for them and they must not be read as bounded-retention "
                 "failures"),
        "count": count,
        "cells": out,
    }


def _pool_across_lengths(summary, lengths, field_name):
    """A per-length ``{family: {key: budget}}`` pooled to the largest that held."""
    pooled = {}
    for family, by_key in summary["by_length"][str(lengths[0])][field_name].items():
        pooled[family] = {}
        for key in by_key:
            values = [summary["by_length"][str(length)][field_name]
                      .get(family, {}).get(key) for length in lengths]
            pooled[family][key] = (None if any(v is None for v in values)
                                   else max(values))
    return pooled


def summarize(rows, budgets, tolerance=0.0):
    """The accuracy matrix, the derived ``required_budget`` and the checks.

    ``accuracy[family][items][budget]`` (with ``"full"`` alongside the tested
    budgets) is the summary matrix; ``accuracy_by_distractors`` is the same
    matrix over the competition axis.  ``required_budget[family][items]`` is the
    smallest tested budget that matches the Full-KV arm on that record set, or
    ``None`` -- and stays ``None`` whenever the Full-KV arm scores zero, which
    :func:`full_kv_unsolved_block` reports per cell.  With more than one tested
    length the pooled matrix sits at the top level and ``by_length[str(L)]``
    repeats it per length, because the prediction-1 test is precisely
    "``required_budget`` must not grow with ``L``".

    ``budget_covers_required_set`` is the operational form of prediction 2: the
    conditional accuracy given ``required_set_tokens <= kept_slots`` with the
    cell counts (see :func:`budget_coverage_block`).
    """
    lengths = sorted({row["length"] for row in rows})
    budgets = sorted(int(b) for b in budgets)
    distractor_rows = [row for row in rows if row.get("distractors") is not None]
    summary = {
        "rows": len(rows),
        "lengths": lengths,
        "budgets": budgets,
        "required_budget_rule": (
            "smallest tested budget whose mean accuracy on the record set is "
            "within `required_tol` of the Full-KV arm's mean accuracy and "
            "strictly positive; null when Full-KV scores zero (see "
            "`full_kv_unsolved`) or no tested budget reaches it"),
        "required_tol": tolerance,
        "accuracy": _nested_matrix(rows, budgets),
        "partial_recall": _nested_matrix(rows, budgets, metric="partial_recall"),
        "required_budget": _required_budgets(rows, budgets, tolerance),
        "accuracy_by_distractors": _nested_matrix(distractor_rows, budgets,
                                                  "distractors"),
        "required_budget_by_distractors": _required_budgets(
            distractor_rows, budgets, tolerance, "distractors"),
        "full_kv_unsolved": full_kv_unsolved_block(rows),
        "by_length": {},
        "cells": {},
    }
    if len(lengths) > 1:
        summary["required_budget_by_length"] = {}
    for length in lengths:
        subset = [row for row in rows if row["length"] == length]
        per_length = {
            "accuracy": _nested_matrix(subset, budgets),
            "required_budget": _required_budgets(subset, budgets, tolerance),
            "accuracy_by_distractors": _nested_matrix(
                [row for row in subset if row.get("distractors") is not None],
                budgets, "distractors"),
            "required_budget_by_distractors": _required_budgets(
                [row for row in subset if row.get("distractors") is not None],
                budgets, tolerance, "distractors"),
            "full_kv_unsolved": full_kv_unsolved_block(subset),
        }
        summary["by_length"][str(length)] = per_length
        if len(lengths) > 1:
            summary["required_budget_by_length"][str(length)] = per_length["required_budget"]
    if len(lengths) > 1:
        # top level: the budget that matched at every tested length
        summary["required_budget"] = _pool_across_lengths(
            summary, lengths, "required_budget")
        summary["required_budget_by_distractors"] = _pool_across_lengths(
            summary, lengths, "required_budget_by_distractors")

    # per (family, items): how many tokens the answer's required set occupies
    for family in sorted({row["family"] for row in rows}):
        for items in sorted({row["items"] for row in rows if row["family"] == family}):
            cell_rows = [row for row in rows
                         if row["family"] == family and row["items"] == items]
            statement_tokens = _mean_field(cell_rows, "statement_tokens")
            spacing = _mean_field(cell_rows, "mean_item_spacing")
            bounded = [row for row in cell_rows if row["arm"] != "full"]
            full_rows = [row for row in cell_rows if row["arm"] == "full"]
            full_accuracy = _mean_bool(full_rows, "correct")
            budgets_cell = {}
            for budget in sorted({int(row["budget"]) for row in bounded
                                  if row.get("budget") is not None}):
                budget_rows = [row for row in bounded if int(row["budget"]) == budget]
                covered = [row for row in budget_rows if row_covers_required_set(row)]
                missing = [row for row in budget_rows if not row_covers_required_set(row)]
                budgets_cell[str(budget)] = {
                    "covered": _coverage_side(covered),
                    "not_covered": _coverage_side(missing),
                    "mean_kept_slots": _mean_field(budget_rows, "kept_slots"),
                    "mean_partial_recall": _mean_float(budget_rows, "partial_recall"),
                    "truncation_rate": _truncation_rate(budget_rows),
                    "mean_required_coverage": _mean_float(budget_rows,
                                                          "required_coverage"),
                    "mean_required_mass": _mean_float(budget_rows, "required_mass",
                                                      digits=6),
                    "mean_items_covered": _mean_field(budget_rows, "items_covered"),
                }
            summary["cells"].setdefault(family, {})[str(items)] = {
                "seeds": len({row["seed"] for row in cell_rows}),
                "records": len({(row["seed"], row["length"]) for row in cell_rows}),
                "distractors": _mean_field(cell_rows, "distractors"),
                "statement_tokens": statement_tokens,
                "tokens_per_item": _mean_field(cell_rows, "tokens_per_item"),
                "mean_item_spacing": spacing,
                "required_set_tokens": _mean_field(cell_rows, "required_set_tokens"),
                "required_set_tokens_rule": (
                    "items x mean tokens per item statement (the tokens the item "
                    "statements occupy); `statement_tokens` is the whole block"),
                "mean_kept_slots_bounded": _mean_field(bounded, "kept_slots"),
                "mean_kept_slots_full": _mean_field(full_rows, "kept_slots"),
                "full_kv_accuracy": full_accuracy,
                "mean_partial_recall_bounded": _mean_float(bounded, "partial_recall"),
                "mean_partial_recall_full": _mean_float(full_rows, "partial_recall"),
                "truncation_rate_bounded": _truncation_rate(bounded),
                "truncation_rate_full": _truncation_rate(full_rows),
                "mean_anchor_sites": _mean_field(bounded, "anchor_sites"),
                "lex_cap": sorted({row["lex_cap"] for row in cell_rows
                                   if row.get("lex_cap") is not None}),
                "placement": sorted({row.get("placement", "scatter")
                                     for row in cell_rows}),
                "full_kv_unsolved": bool(full_accuracy is not None
                                         and full_accuracy <= 0.0),
                "budgets": budgets_cell,
            }

    summary["budget_covers_required_set"] = budget_coverage_block(rows)
    summary["required_span_coverage"] = required_span_coverage_block(rows)
    summary["prediction_checks"] = _prediction_checks(rows, summary, budgets, lengths)
    return summary


def _prediction_checks(rows, summary, budgets, lengths):
    """The two mechanism predictions, evaluated on whatever was swept.

    Prediction 1 is checked per ``(family, items)``: are the per-length required
    budgets equal, and does the required budget grow with ``items`` (and with the
    distractor count, the competition axis)?  Prediction 2 lives in
    ``summary["budget_covers_required_set"]``; the ``cliff`` entry below keeps
    the original nominal-budget view for continuity and repeats its two means.
    The means and cell counts are reported instead of a pass/fail flag, because
    the threshold is the experimenter's to choose.
    """
    checks = {}
    flat = []
    for family, by_items in summary.get("accuracy", {}).items():
        for items in by_items:
            values = [summary["by_length"][str(length)]["required_budget"]
                      .get(family, {}).get(items) for length in lengths]
            present = [v for v in values if v is not None]
            if len(lengths) > 1 and len(present) == len(lengths):
                flat.append(len(set(present)) == 1)
    checks["required_budget_flat_in_length"] = (all(flat) if flat else None)
    checks["required_budget_flat_cells"] = len(flat)

    def grows_over(field_name):
        cells = 0
        verdicts = []
        for length in lengths:
            per_length = summary["by_length"][str(length)][field_name]
            for family, by_items in per_length.items():
                sequence = [(int(items), value) for items, value in by_items.items()
                            if value is not None]
                sequence.sort()
                if len(sequence) > 1:
                    cells += 1
                    verdicts.append(all(b[1] >= a[1]
                                        for a, b in zip(sequence, sequence[1:])))
        return (all(verdicts) if verdicts else None), cells

    checks["required_budget_grows_with_items"], checks["required_budget_grows_cells"] = \
        grows_over("required_budget")
    (checks["required_budget_grows_with_distractors"],
     checks["required_budget_grows_with_distractors_cells"]) = \
        grows_over("required_budget_by_distractors")

    above, below = [], []
    for row in rows:
        if row["arm"] == "full" or row.get("required_set_tokens") is None:
            continue
        value = float(bool(row["correct"]))
        if row["budget"] >= row["required_set_tokens"]:
            above.append(value)
        else:
            below.append(value)
    coverage = summary.get("budget_covers_required_set", {})
    checks["cliff"] = {
        "mean_accuracy_budget_covers_required_set": (round(sum(above) / len(above), 4)
                                                    if above else None),
        "mean_accuracy_budget_below_required_set": (round(sum(below) / len(below), 4)
                                                   if below else None),
        "cells_above": len(above), "cells_below": len(below),
        "required_set_tokens_rule": "items x mean tokens per item statement",
        "budget_comparison": "nominal budget, not the measured kept slots",
        "measured_kept_slots_block": "budget_covers_required_set",
        "measured_separation_accuracy": coverage.get("separation", {}).get("accuracy"),
    }
    return checks


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Required-set sweep: items the answer depends on vs retained budget.")
    parser.add_argument("--family", nargs="+", default=["multikey", "aggregate"],
                        choices=["multikey", "aggregate", "needles"])
    parser.add_argument("--items", nargs="+", type=int, default=[1, 2, 4, 8, 16],
                        help="k for multikey (total keys in the legacy shape) and "
                             "for needles (needles asked for), m for aggregate; "
                             "ignored for multikey when --distractors is given")
    parser.add_argument("--distractors", nargs="+", type=int, default=None,
                        help="multikey competition axis: the number of other keys "
                             "sharing the asked key's surface form; each count is "
                             "swept and the cell label `items` becomes count + 1 "
                             "(default: off, the legacy --items shape)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="number of seeds (0..seeds-1)")
    parser.add_argument("--length", type=int, default=32768)
    parser.add_argument("--lengths", nargs="+", type=int, default=None,
                        help="repeat the sweep at these context lengths "
                             "(prediction 1: required_budget must be flat in L)")
    parser.add_argument("--budgets", nargs="+", type=int,
                        default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--aggregate", default="sum", choices=["sum", "count"],
                        help="aggregate family: ask for the sum or for a count")
    parser.add_argument("--placement", default="scatter",
                        choices=["scatter", "recency"],
                        help="where the item statements sit.  `scatter` spreads "
                             "them through the filler (the shipped shape); "
                             "`recency` puts all of them immediately before the "
                             "question, which is the oracle-placement control: "
                             "the required set then sits inside the recency "
                             "channel of any budget whose recent window covers "
                             "the statement block")
    parser.add_argument("--obs", type=int, default=64)
    parser.add_argument("--nsink", type=int, default=4)
    parser.add_argument("--pool", type=int, default=7)
    parser.add_argument("--dilate", type=int, default=9)
    parser.add_argument("--lex-cap", type=int, default=512)
    parser.add_argument("--hops", type=int, default=6)
    parser.add_argument("--anchor-mode", default="pattern",
                        choices=["pattern", "rare", "both"])
    parser.add_argument("--min-anchor-len", type=int, default=3)
    parser.add_argument("--scoring", default="last", choices=["last", "mean", "max"])
    parser.add_argument("--prefill-chunk", type=int, default=8192)
    parser.add_argument("--key-chunk", type=int, default=4096)
    parser.add_argument("--max-new", type=int, default=None,
                        help="greedy decode budget per arm (default: 24, or "
                             "max(32, 8 x k) for needles so the whole answer fits)")
    parser.add_argument("--required-tol", type=float, default=0.0,
                        help="accuracy slack allowed when deriving required_budget")
    parser.add_argument("--model", default=os.environ.get("QCC_MODEL", "meta-llama/Llama-3.2-1B-Instruct"))
    parser.add_argument("--tokenizer", default=None,
                        help="tokenizer path (defaults to --model)")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-impl", default="sdpa",
                        help="attention implementation to request at load time")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--shared-prefill", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="prefill once per record and re-select per budget "
                             "(bit-identical to compile_bounded_cache, much faster)")
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def required_spans(tokenizer, prompt, statements):
    """Token index set of every item statement, located verbatim in the prompt.

    The required set of a record is the item statements themselves, so this is
    the *measured* form of prediction 2: a bounded row can be scored against the
    slots it actually kept (``required_coverage``) instead of against the width
    comparison ``required_set_tokens <= kept_slots``, which only says that the
    statements would fit.  A statement counts as covered when every one of its
    tokens survives.
    """
    if not statements:
        return []
    offsets = tokenizer(prompt, add_special_tokens=False,
                        return_offsets_mapping=True).offset_mapping
    spans, cursor = [], 0
    for statement in statements:
        position = prompt.find(statement, cursor)
        if position < 0:
            position = prompt.find(statement)
        if position < 0:
            continue
        cursor = position + len(statement)
        spans.append({token for token, (a, b) in enumerate(offsets)
                      if b > position and a < position + len(statement)})
    return spans


def anchor_site_coverage(tokenizer, prompt, anchors, keys):
    """How many item sites the lexical-anchor set actually reaches.

    A site counts as reached when at least one of the key's own tokens is in
    ``anchors``.  ``lexical_anchors`` expands every hit by
    ``lex_context_left + lex_context_right + 1`` tokens and stops at
    ``lex_cap``, so the number of sites the anchor channel can hold is a
    *measured* quantity: roughly ``lex_cap / (left + right + 1)``.  Returns
    ``(covered, total)``.
    """
    encoded = tokenizer(prompt, add_special_tokens=False,
                        return_offsets_mapping=True)
    offsets = encoded.offset_mapping
    anchor_set = {int(token) for token in (anchors or [])}
    covered = 0
    for key in keys:
        position = prompt.find(key)
        if position < 0:
            continue
        span = {token for token, (a, b) in enumerate(offsets)
                if b > position and a < position + len(key)}
        if span & anchor_set:
            covered += 1
    return covered, len(keys)


def build_plan(args, lengths):
    """One entry per record to build: ``(family, items, distractors, seed, length)``.

    ``multikey`` with ``--distractors`` sweeps the competition axis: the entry's
    ``items`` is the total key count ``N + 1`` (which is the cell label the
    summary groups by) and ``--items`` is not used for that family in this mode.
    Every other family sweeps ``--items``.
    """
    plan = []
    for length in lengths:
        for family in args.family:
            if family == "multikey" and args.distractors is not None:
                for count in sorted({int(value) for value in args.distractors}):
                    for seed in range(args.seeds):
                        plan.append({"family": family, "items": count + 1,
                                     "distractors": count, "seed": seed,
                                     "length": length})
            else:
                for items in args.items:
                    for seed in range(args.seeds):
                        plan.append({"family": family, "items": int(items),
                                     "distractors": None, "seed": seed,
                                     "length": length})
    return plan


def main(argv=None):
    args = parse_args(argv)
    if args.distractors is not None:
        if any(int(value) < 1 for value in args.distractors):
            raise SystemExit("--distractors counts must be >= 1")
        if "multikey" not in args.family:
            print("[plan] --distractors only affects family=multikey; ignored here",
                  flush=True)
    if args.trust_remote_code:
        _ensure_remote_code_compat()
    from transformers import AutoTokenizer

    lengths = args.lengths or [args.length]
    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model,
                                              trust_remote_code=args.trust_remote_code)
    model = load_hf_causal_lm(args.model, dtype=args.dtype, device=args.device,
                              trust_remote_code=args.trust_remote_code,
                              load_in_4bit=args.load_4bit,
                              attn_implementation=args.attn_impl).eval()
    kwargs_names = forward_kwargs_of(model)
    eos_ids = eos_token_ids(model)
    budgets = sorted(args.budgets)
    configs = {budget: retention_config(args, budget) for budget in budgets}

    plan = build_plan(args, lengths)
    if args.distractors is not None and "multikey" in args.family:
        key_counts = sorted({entry["items"] for entry in plan
                             if entry["family"] == "multikey"})
        print(f"[plan] multikey competition axis: {len(key_counts)} key counts "
              f"{key_counts} (--items is not used for multikey while --distractors "
              "is given)", flush=True)
    print(f"[plan] {len(plan)} records x (1 full + {len(budgets)} budgets) arms  "
          f"lengths={lengths} items={args.items} budgets={budgets} "
          f"distractors={args.distractors} "
          f"device={args.device} shared_prefill={args.shared_prefill}", flush=True)

    rows = []
    for index, entry in enumerate(plan):
        family, items = entry["family"], entry["items"]
        seed, length = entry["seed"], entry["length"]
        record = build_record(family, items, seed, length, tokenizer, args.aggregate,
                              distractors=entry["distractors"],
                              placement=args.placement)
        max_new = decode_budget(family, items, args.max_new)
        # ``record["statement_tokens"]`` is the whole item-statement block; the
        # required set of every family is that block (every item for needles and
        # aggregate, every competing key for multikey), i.e. items x tokens per
        # item statement.  The per-item mean is what makes the product honest.
        tokens_per_item = round(record["statement_tokens"] / max(1, items), 2)
        required_set_tokens = round(items * tokens_per_item, 1)
        ids = torch.tensor([encode_ids(tokenizer, record["prompt"])],
                           dtype=torch.long, device=device)
        prompt_tokens = int(ids.shape[1])
        mean_spacing = round(
            max(0.0, prompt_tokens - record["statement_tokens"]) / max(1, items), 1)
        required_token_set = required_spans(tokenizer, record["prompt"],
                                            record.get("statements") or [])
        required_tokens = sum(len(span) for span in required_token_set)
        anchors = None
        scores = None
        anchor_sites = None
        required_mass = None
        with fixed_rope_length(model, prompt_tokens):
            reset_peak_memory(device)
            start = time.time()
            cache, logits, captured = prefill_capture(
                model, ids, args.obs, args.prefill_chunk)
            prefill_s = time.time() - start
            layers = [(layer.keys, layer.values) for layer in cache.layers]
            if args.shared_prefill:
                # Same packaged steps compile_bounded_cache runs internally, so
                # every budget re-uses one exact prefill of the record.
                scores = observation_scores(model, captured, cache, prompt_tokens,
                                            configs[budgets[0]])
                if args.lex_cap > 0:
                    anchors = lexical_anchors(tokenizer, record["prompt"],
                                              configs[budgets[0]])
                if record.get("keys"):
                    anchor_sites = anchor_site_coverage(
                        tokenizer, record["prompt"], anchors, record["keys"])
                required_mass = required_mass_last_query(scores, required_token_set)
            arm_rows = []
            for arm, budget in [("full", None)] + [(f"b{b}", b) for b in budgets]:
                arm_prefill_s = prefill_s
                indices = None
                if arm == "full":
                    working, kept = cache, cache.get_seq_length()
                elif args.shared_prefill:
                    working = clone_cache(layers)
                    config = configs[budget]
                    indices = select_indices(scores, working, prompt_tokens,
                                             config, anchors)
                    prune_cache(working, indices)
                    kept = int(working.get_seq_length())
                else:
                    compile_start = time.time()
                    working, _logits = compile_bounded_cache(
                        model, ids, configs[budget], tokenizer=tokenizer)
                    arm_prefill_s = time.time() - compile_start
                    kept = int(working.get_seq_length())
                decode_start = time.time()
                generated, _decode_s = greedy(model, working, logits[:, -1:].argmax(-1),
                                              prompt_tokens, max_new, eos_ids,
                                              kwargs_names)
                decode_s = time.time() - decode_start
                text = decode_ids(tokenizer, generated).strip()
                scored = score_record(record, text)
                kept_required = coverage = covered_items = None
                if indices is not None and required_token_set:
                    coverage, covered_items = required_coverage_of(
                        indices, required_token_set)
                    kept_required = (None if coverage is None
                                     else round(coverage * required_tokens))
                row = {
                    "family": family, "items": items, "seed": seed,
                    "length": length, "aggregate": record["aggregate"],
                    "distractors": record.get("distractor_count"),
                    "key_count": record.get("key_count"),
                    "arm": arm, "budget": budget, "prompt_tokens": prompt_tokens,
                    "kept_slots": int(kept),
                    "prediction": text, "expected": record["expected"],
                    "references": scored["references"],
                    "correct": scored["correct"], "parsed": scored["parsed"],
                    "partial": scored["partial"],
                    "partial_recall": scored["partial_recall"],
                    "exact_set_match": scored["exact_set_match"],
                    "present": scored["present"], "missing": scored["missing"],
                    "distractor_hit": scored["distractor_hit"],
                    "candidates": scored["candidates"],
                    "generated_tokens": len(generated),
                    "max_new": max_new,
                    "statement_tokens": record["statement_tokens"],
                    "placement": record.get("placement", "scatter"),
                    "anchor_sites": None if arm == "full" or anchor_sites is None
                    else anchor_sites[0],
                    "anchor_sites_total": None if arm == "full" or anchor_sites is None
                    else anchor_sites[1],
                    "lex_cap": int(args.lex_cap),
                    "required_span_tokens": required_tokens or None,
                    "required_kept_tokens": kept_required,
                    "required_coverage": coverage,
                    "items_covered": covered_items,
                    "items_required": len(required_token_set) or None,
                    "required_mass": None if arm == "full" else required_mass,
                    "tokens_per_item": tokens_per_item,
                    "mean_item_spacing": mean_spacing,
                    "required_set_tokens": required_set_tokens,
                    "budget_covers_required_set": (
                        None if arm == "full"
                        else bool(required_set_tokens <= kept)),
                    "prefill_s": round(arm_prefill_s, 2),
                    "decode_s": round(decode_s, 2),
                    "peak_mib": peak_memory_mib(device),
                    "peak_rss_mib": peak_rss_mib(),
                    "target_key": record.get("target_key"),
                    "target_status": record.get("target_status"),
                    "shared_prefill": bool(args.shared_prefill),
                }
                arm_rows.append(row)
                print(f"[{index:>3d}/{len(plan)}] {family:<9} items={items:<3d} "
                      f"L={prompt_tokens:>6d} seed={seed} {arm:<7} kept={kept:>6d} "
                      f"covered={str(row['budget_covers_required_set']):<5} "
                      f"correct={int(row['correct'])} "
                      f"recall={row['partial_recall']:.2f} pred={text[:24]!r}",
                      flush=True)
                if arm != "full" and not args.shared_prefill:
                    del working
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            rows.extend(arm_rows)
            del cache, layers, captured, scores, anchors
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = summarize(rows, budgets, args.required_tol)
    payload = {
        "model": args.model,
        "label": args.label or Path(args.model).name,
        "config": {**vars(args), "lengths": lengths, "budgets": budgets},
        "summary": summary,
        "records": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    coverage = summary["budget_covers_required_set"]
    printed = {
        key: value for key, value in summary.items()
        if key in ("accuracy", "required_budget", "required_budget_by_length",
                   "required_budget_by_distractors", "full_kv_unsolved",
                   "prediction_checks")
    }
    printed["budget_covers_required_set"] = {
        key: coverage[key]
        for key in ("covered", "not_covered", "separation", "by_family")}
    print("\n" + json.dumps(printed, indent=1))
    print("wrote", out)
    return summary


if __name__ == "__main__":
    main()
