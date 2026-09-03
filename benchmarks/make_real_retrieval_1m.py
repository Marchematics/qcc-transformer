#!/usr/bin/env python3
"""Pre-register a strict 1M real-LM retrieval workload before model execution."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

_ADJECTIVES = (
    "amber", "azure", "cobalt", "crimson", "golden", "indigo", "ivory", "jade",
    "lilac", "silver", "teal", "violet",
)
_NOUNS = (
    "archive", "cluster", "ledger", "registry", "repository", "station", "vault",
    "warehouse",
)


def _code(rng: random.Random) -> str:
    return f"{rng.randrange(0, 100_000_000):08d}"


def make_trial(index: int, seed: int, needles: int, distractors: int) -> dict:
    rng = random.Random(seed + 1_000_003 * index)
    family = rng.choice(("AURORA", "ATLAS", "ORION", "NOVA", "VEGA"))
    base_number = rng.randrange(1000, 9000)
    records = []
    used_codes: set[str] = set()
    used_entities: set[str] = set()

    def unique_code() -> str:
        while True:
            value = _code(rng)
            if value not in used_codes:
                used_codes.add(value)
                return value

    for needle_index in range(needles):
        entity = (
            f"{family} {rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)} "
            f"{base_number + 17 * needle_index}"
        )
        code = unique_code()
        used_entities.add(entity)
        records.append(
            {
                "kind": "needle",
                "entity": entity,
                "code": code,
                "text": f"Authoritative record: the access code for {entity} is {code}.",
                "depth": rng.uniform(0.01, 0.97),
            }
        )
    # The query must require every needle. Shuffle the requested order so a
    # model cannot learn that the answer is always emitted in construction
    # order, while retaining a single canonical order for paired scoring.
    target_indices = list(range(needles))
    rng.shuffle(target_indices)
    targets = [
        {"entity": records[index]["entity"], "code": records[index]["code"]}
        for index in target_indices
    ]

    # Semantic distractors share the family/category and nearby identifiers but
    # carry different codes. They are intentionally lexical near-neighbors.
    for distractor_index in range(distractors):
        near = base_number + rng.randrange(-12, 13)
        while True:
            entity = (
                f"{family} {rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)} {near}"
            )
            if entity not in used_entities:
                used_entities.add(entity)
                break
            near = base_number + rng.randrange(-12, 13)
        code = unique_code()
        records.append(
            {
                "kind": "semantic_distractor",
                "entity": entity,
                "code": code,
                "text": f"Reference record: the access code for {entity} is {code}.",
                "depth": rng.uniform(0.01, 0.97),
            }
        )
    rng.shuffle(records)
    return {
        "trial": index,
        "seed": seed + 1_000_003 * index,
        # Keep the first target fields for readers of the original protocol;
        # strict scoring uses the complete ordered target list below.
        "target_entity": targets[0]["entity"],
        "expected": targets[0]["code"],
        "targets": targets,
        "records": records,
        "random_depth": True,
        "multi_needle": needles >= 2,
        "semantic_distractor": distractors > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--needles", type=int, default=4)
    parser.add_argument("--semantic-distractors", type=int, default=12)
    parser.add_argument("--context-tokens", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1000:
        raise ValueError("strict real retrieval manifest requires at least 1000 trials")
    if args.needles < 2 or args.semantic_distractors < 1:
        raise ValueError("strict manifest requires multi-needle and semantic distractors")
    if args.context_tokens < 1_000_000:
        raise ValueError("strict manifest context must be at least 1M tokens")

    header = {
        "schema": "qcc-real-retrieval-manifest-v2",
        "protocol_locked": True,
        "trials": args.trials,
        "seed": args.seed,
        "needles": args.needles,
        "semantic_distractors": args.semantic_distractors,
        "context_tokens": args.context_tokens,
        "random_depth": True,
        "multi_needle": args.needles >= 2,
        "semantic_distractor": args.semantic_distractors > 0,
        "all_needles_required": True,
        "scoring": "exact ordered normalized access-code match for every needle in generated continuation",
    }
    rows = [json.dumps(header, sort_keys=True)]
    rows.extend(
        json.dumps(make_trial(i, args.seed, args.needles, args.semantic_distractors), sort_keys=True)
        for i in range(args.trials)
    )
    content = "\n".join(rows) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(json.dumps({**header, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
