"""Generate a strict marker/value retrieval JSONL split.

Each value ID is evaluated at every requested context length.  Marker and
filler IDs can be randomized independently per record, which tests lexical
addressing rather than memorization of one special token.  The output format
is consumed by :mod:`evaluate_retrieval`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def write_dataset(
    path: Path,
    *,
    lengths: list[int],
    vocab_size: int,
    query_fraction: float,
    seed: int,
    random_marker: bool,
    random_filler: bool,
) -> int:
    if vocab_size < 5 or not lengths:
        raise ValueError("vocab_size must be >=5 and lengths must be non-empty")
    if not 0.0 < query_fraction < 1.0:
        raise ValueError("query_fraction must lie in (0, 1)")
    generator = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    with path.open("w", encoding="utf-8") as stream:
        for length in lengths:
            if length < 4:
                raise ValueError("lengths must be >=4")
            query_position = max(2, min(length - 2, int(length * query_fraction)))
            # Cover every legal value ID exactly once per context length.
            for value in range(3, vocab_size):
                marker = generator.randrange(1, vocab_size) if random_marker else 1
                filler = generator.randrange(1, vocab_size) if random_filler else 2
                tokens = [filler] * length
                tokens[0] = marker
                tokens[1] = value
                tokens[query_position] = marker
                tokens[query_position + 1] = value
                json.dump(
                    {
                        "input_ids": tokens,
                        "target_position": query_position,
                        "answers": [value],
                    },
                    stream,
                    separators=(",", ":"),
                )
                stream.write("\n")
                records += 1
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", default="128000,1000000")
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--query-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-marker", action="store_true")
    parser.add_argument("--random-filler", action="store_true")
    args = parser.parse_args()
    lengths = [int(raw.strip()) for raw in args.lengths.split(",") if raw.strip()]
    records = write_dataset(
        args.output,
        lengths=lengths,
        vocab_size=args.vocab_size,
        query_fraction=args.query_fraction,
        seed=args.seed,
        random_marker=args.random_marker,
        random_filler=args.random_filler,
    )
    print(json.dumps({"output": str(args.output), "records": records, "lengths": lengths}))


if __name__ == "__main__":
    main()
