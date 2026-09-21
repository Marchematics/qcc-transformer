"""Where does bounded retention collapse? Required set vs retained budget.

``docs/MECHANISM.md`` states two falsifiable predictions for the main line:

1. **The required slot count scales with the number of competing items, not with
   the context length.**  Grow ``L`` at a fixed task: quality is flat.  Grow the
   number of items the answer depends on: the budget needed to match Full-KV
   grows with it.
2. **Failure is a cliff once the required set exceeds the budget.**  A task whose
   answer needs ``m`` scattered items must collapse when ``m x tokens-per-item``
   exceeds the retained budget.

This runner sweeps exactly those two axes on two synthetic families, both
generated from a seed so a run is reproducible:

``multikey``
    The RULER ``niah_multikey_k`` shape, parameterised by ``k``: ``k`` keys with
    values scattered through filler, all sharing the surface form
    ``trace-<hex>``, and a question about one of them.  ``--items`` is ``k``.
    Score: exact match of the asked value (the headline); character-level F1 of
    the best numeric candidate is reported as partial credit.

``aggregate``
    ``m`` numbers scattered through filler; the question asks for their sum
    (small integers, so the sum is unambiguous) or for the count of items with a
    given status (``--aggregate sum|count``).  ``--items`` is ``m``.  Score:
    exact match of the numeric answer.

Both families are ran against a matched **Full-KV** arm through
``prefill_capture`` and against a bounded arm through the packaged retention API
(``RetentionConfig`` + ``compile_bounded_cache``), with the same greedy decode
loop for every arm.  ``--budgets`` is swept; ``--lengths`` optionally repeats the
whole sweep at several context lengths, which is what prediction 1 needs
(``required_budget`` must not grow with ``L``).

The derived field the experiment exists for is ``required_budget`` per
``(family, items)``: the smallest tested budget whose accuracy matches the
Full-KV arm's accuracy on that record set, or ``null`` when no tested budget
does.  When several lengths are swept, ``summary["by_length"][L]`` carries the
per-length value and the top-level ``summary["required_budget"]`` is the largest
across lengths (the budget that matched at *every* tested length).

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


def record_rng(family, items, seed, aggregate="sum"):
    """A seed-stable generator: the same triple always rebuilds the same task."""
    return random.Random(f"qcc-required-set:{family}:{items}:{aggregate}:{seed}")


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
                    rng, filler_pool=FILLER_SENTENCES, max_filler=8000):
    """Scatter ``statements`` through filler and land on ~``target_tokens``.

    The item statements are placed at random positions through the filler body
    (as in RULER's needle placement) and the question is always the last
    segment, so the observation window sees the question.  The filler count is
    binary-searched and then padded with one-token filler words, so the final
    prompt is within a couple of tokens of the target even though the fillers
    tokenize to different lengths.

    Returns ``(text, n_tokens, statement_tokens)``.
    """
    fractions = sorted(rng.random() for _ in statements)
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


def multikey_record(items, seed, target_tokens, tokenizer):
    """``k`` keys of one surface form, one asked value, ``k - 1`` distractors."""
    rng = record_rng("multikey", items, seed)
    names = (rng.sample(ITEM_NAMES, items) if items <= len(ITEM_NAMES)
             else [f"{name}{index}"
                   for index, name in enumerate(ITEM_NAMES)][:items])
    keys, seen_keys = [], set()
    while len(keys) < items:
        suffix = "".join(rng.choice(_HEX) for _ in range(6))
        key = "trace-" + suffix
        # at least one letter keeps the key from looking like the answer's digits
        if key not in seen_keys and any(character.isalpha() for character in suffix):
            seen_keys.add(key)
            keys.append(key)
    values, seen_values = [], set()
    while len(values) < items:
        value = rng.randint(100000, 999999)
        if value not in seen_values:
            seen_values.add(value)
            values.append(value)
    target_index = rng.randrange(items)
    statements = [f"One of the magic numbers for {key} is {value}."
                  for key, value in zip(keys, values)]
    intro = ("A list of records is given below. Each record names a key and the "
             "magic number assigned to that key.")
    question = (f"What is the magic number for {keys[target_index]} mentioned "
                "above? Answer with a single number.")
    prompt, n_tokens, statement_tokens = assemble_prompt(
        intro, statements, question, target_tokens, tokenizer, rng)
    return {
        "family": "multikey", "items": items, "seed": seed,
        "aggregate": None, "prompt": prompt, "prompt_tokens": n_tokens,
        "statement_tokens": statement_tokens,
        "expected": str(values[target_index]),
        "distractors": [str(v) for i, v in enumerate(values) if i != target_index],
        "keys": keys, "values": [str(v) for v in values],
        "target_key": keys[target_index], "target_index": target_index,
        "names": names,
    }


def aggregate_record(items, seed, target_tokens, tokenizer, aggregate="sum"):
    """``m`` scattered items; ask for their sum or for a status count."""
    rng = record_rng("aggregate", items, seed, aggregate)
    names = (rng.sample(ITEM_NAMES, items) if items <= len(ITEM_NAMES)
             else [f"{name}{index}"
                   for index, name in enumerate(ITEM_NAMES)][:items])
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
        intro, statements, question, target_tokens, tokenizer, rng)
    return {
        "family": "aggregate", "items": items, "seed": seed,
        "aggregate": aggregate, "prompt": prompt, "prompt_tokens": n_tokens,
        "statement_tokens": statement_tokens, "expected": expected,
        "distractors": distractors, "names": names,
        "values": [str(v) for v in values], "statuses": list(statuses),
        "target_status": target_status if aggregate == "count" else None,
    }


def build_record(family, items, seed, target_tokens, tokenizer, aggregate="sum"):
    """One task record of the requested family (deterministic in the seed)."""
    if family == "multikey":
        return multikey_record(items, seed, target_tokens, tokenizer)
    if family == "aggregate":
        return aggregate_record(items, seed, target_tokens, tokenizer, aggregate)
    raise ValueError(f"unknown family {family!r}")


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


def _nested_matrix(rows, budgets):
    """Accuracy nested by family -> items -> budget (plus a ``full`` key)."""
    matrix = {}
    for row in rows:
        cell = matrix.setdefault(row["family"], {}).setdefault(str(row["items"]), {})
        arm = "full" if row["arm"] == "full" else str(row["budget"])
        cell.setdefault(arm, []).append(float(bool(row["correct"])))
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


def _required_budgets(rows, budgets, tolerance):
    """``{family: {items: smallest matching budget or None}}``."""
    cells = {}
    for row in rows:
        cells.setdefault((row["family"], row["items"]), []).append(row)
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


def summarize(rows, budgets, tolerance=0.0):
    """The accuracy matrix, the derived ``required_budget`` and the checks.

    ``accuracy[family][items][budget]`` (with ``"full"`` alongside the tested
    budgets) is the summary matrix.  ``required_budget[family][items]`` is the
    smallest tested budget that matches the Full-KV arm on that record set, or
    ``None``.  With more than one tested length the pooled matrix sits at the
    top level and ``by_length[str(L)]`` repeats it per length, because the
    prediction-1 test is precisely "``required_budget`` must not grow with
    ``L``".
    """
    lengths = sorted({row["length"] for row in rows})
    budgets = sorted(int(b) for b in budgets)
    summary = {
        "rows": len(rows),
        "lengths": lengths,
        "budgets": budgets,
        "required_budget_rule": (
            "smallest tested budget whose mean accuracy on the record set is "
            "within `required_tol` of the Full-KV arm's mean accuracy and "
            "strictly positive; null when Full-KV scores zero or no tested "
            "budget reaches it"),
        "required_tol": tolerance,
        "accuracy": _nested_matrix(rows, budgets),
        "required_budget": _required_budgets(rows, budgets, tolerance),
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
        }
        summary["by_length"][str(length)] = per_length
        if len(lengths) > 1:
            summary["required_budget_by_length"][str(length)] = per_length["required_budget"]
    if len(lengths) > 1:
        # top level: the budget that matched at every tested length
        pooled = {}
        for family, by_items in summary["by_length"][str(lengths[0])]["required_budget"].items():
            pooled[family] = {}
            for items in by_items:
                values = [summary["by_length"][str(length)]["required_budget"]
                          .get(family, {}).get(items) for length in lengths]
                pooled[family][items] = (None if any(v is None for v in values)
                                         else max(values))
        summary["required_budget"] = pooled

    # per (family, items): how many tokens the answer's required set occupies
    for family in sorted({row["family"] for row in rows}):
        for items in sorted({row["items"] for row in rows if row["family"] == family}):
            cell_rows = [row for row in rows
                         if row["family"] == family and row["items"] == items]
            statement_tokens = _mean_field(cell_rows, "statement_tokens")
            spacing = _mean_field(cell_rows, "mean_item_spacing")
            bounded = [row for row in cell_rows if row["arm"] != "full"]
            summary["cells"].setdefault(family, {})[str(items)] = {
                "seeds": len({row["seed"] for row in cell_rows}),
                "records": len({(row["seed"], row["length"]) for row in cell_rows}),
                "statement_tokens": statement_tokens,
                "mean_item_spacing": spacing,
                "required_set_tokens": (None if statement_tokens is None
                                        else round(items * statement_tokens, 1)),
                "mean_kept_slots_bounded": _mean_field(bounded, "kept_slots"),
                "mean_kept_slots_full": _mean_field(
                    [row for row in cell_rows if row["arm"] == "full"], "kept_slots"),
            }

    summary["prediction_checks"] = _prediction_checks(rows, summary, budgets, lengths)
    return summary


def _prediction_checks(rows, summary, budgets, lengths):
    """The two mechanism predictions, evaluated on whatever was swept.

    Prediction 1 is checked per ``(family, items)``: are the per-length required
    budgets equal, and does the required budget grow with ``items``?
    Prediction 2 is reported as the accuracy separation between cells whose
    budget covers ``items x statement_tokens`` and cells where it does not; the
    two means and the cell counts are reported instead of a pass/fail flag,
    because the threshold is the experimenter's to choose.
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

    grows = []
    for length in lengths:
        per_length = summary["by_length"][str(length)]["required_budget"]
        for family, by_items in per_length.items():
            sequence = [(int(items), value) for items, value in by_items.items()
                        if value is not None]
            sequence.sort()
            if len(sequence) > 1:
                grows.append(all(b[1] >= a[1] for a, b in zip(sequence, sequence[1:])))
    checks["required_budget_grows_with_items"] = (all(grows) if grows else None)
    checks["required_budget_grows_cells"] = len(grows)

    above, below = [], []
    for row in rows:
        if row["arm"] == "full" or row.get("required_set_tokens") is None:
            continue
        value = float(bool(row["correct"]))
        if row["budget"] >= row["required_set_tokens"]:
            above.append(value)
        else:
            below.append(value)
    checks["cliff"] = {
        "mean_accuracy_budget_covers_required_set": (round(sum(above) / len(above), 4)
                                                    if above else None),
        "mean_accuracy_budget_below_required_set": (round(sum(below) / len(below), 4)
                                                   if below else None),
        "cells_above": len(above), "cells_below": len(below),
        "required_set_tokens_rule": "items x mean tokens per item statement",
    }
    return checks


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Required-set sweep: items the answer depends on vs retained budget.")
    parser.add_argument("--family", nargs="+", default=["multikey", "aggregate"],
                        choices=["multikey", "aggregate"])
    parser.add_argument("--items", nargs="+", type=int, default=[1, 2, 4, 8, 16],
                        help="k for multikey, m for aggregate")
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
    parser.add_argument("--max-new", type=int, default=24)
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


def main(argv=None):
    args = parse_args(argv)
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

    plan = [(family, items, seed, length)
            for length in lengths
            for family in args.family
            for items in args.items
            for seed in range(args.seeds)]
    print(f"[plan] {len(plan)} records x (1 full + {len(budgets)} budgets) arms  "
          f"lengths={lengths} items={args.items} budgets={budgets} "
          f"device={args.device} shared_prefill={args.shared_prefill}", flush=True)

    rows = []
    for index, (family, items, seed, length) in enumerate(plan):
        record = build_record(family, items, seed, length, tokenizer, args.aggregate)
        ids = torch.tensor([encode_ids(tokenizer, record["prompt"])],
                           dtype=torch.long, device=device)
        prompt_tokens = int(ids.shape[1])
        mean_spacing = round(
            max(0.0, prompt_tokens - record["statement_tokens"]) / max(1, items), 1)
        anchors = None
        scores = None
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
            arm_rows = []
            for arm, budget in [("full", None)] + [(f"b{b}", b) for b in budgets]:
                arm_prefill_s = prefill_s
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
                                              prompt_tokens, args.max_new, eos_ids,
                                              kwargs_names)
                decode_s = time.time() - decode_start
                text = decode_ids(tokenizer, generated).strip()
                scored = score_prediction(text, record["expected"],
                                          record["distractors"])
                row = {
                    "family": family, "items": items, "seed": seed,
                    "length": length, "aggregate": record["aggregate"],
                    "arm": arm, "budget": budget, "prompt_tokens": prompt_tokens,
                    "kept_slots": int(kept),
                    "prediction": text, "expected": record["expected"],
                    "correct": scored["correct"], "parsed": scored["parsed"],
                    "partial": scored["partial"],
                    "distractor_hit": scored["distractor_hit"],
                    "candidates": scored["candidates"],
                    "generated_tokens": len(generated),
                    "statement_tokens": record["statement_tokens"],
                    "mean_item_spacing": mean_spacing,
                    "required_set_tokens": round(items * record["statement_tokens"], 1),
                    "prefill_s": round(arm_prefill_s, 2),
                    "decode_s": round(decode_s, 2),
                    "peak_mib": peak_memory_mib(device),
                    "peak_rss_mib": peak_rss_mib(),
                    "target_key": record.get("target_key"),
                    "target_status": record.get("target_status"),
                    "shared_prefill": bool(args.shared_prefill),
                }
                arm_rows.append(row)
                print(f"[{index:>3d}/{len(plan)}] {family:<9} items={items:<2d} "
                      f"L={prompt_tokens:>6d} seed={seed} {arm:<7} kept={kept:>6d} "
                      f"correct={int(row['correct'])} pred={text[:24]!r}", flush=True)
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
    print("\n" + json.dumps({k: v for k, v in summary.items()
                             if k in ("accuracy", "required_budget",
                                      "required_budget_by_length",
                                      "prediction_checks")}, indent=1))
    print("wrote", out)
    return summary


if __name__ == "__main__":
    main()
