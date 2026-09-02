"""Convert a prepared NVIDIA RULER JSONL split to the QCC evaluator format.

RULER's prepared records contain text in ``input``, answer strings in
``outputs``, and the token offset ``token_position_answer``.  QCC's evaluator
expects integer ``input_ids`` and the answer token IDs.  This adapter keeps the
tokenizer and truncation policy explicit so a converted split can be audited;
it does not claim that a toy or randomly initialized checkpoint is a language
model quality result.

Example::

    python benchmarks/convert_ruler_json.py \
        --input ruler/niah_single_1/validation.jsonl \
        --output runs/ruler_niah_128k.jsonl \
        --tokenizer Qwen/Qwen2.5-0.5B-Instruct --max-examples 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_tokenizer(name_or_path: str, *, trust_remote_code: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "transformers is required for RULER conversion; install it first"
        ) from exc
    return AutoTokenizer.from_pretrained(
        name_or_path, use_fast=True, trust_remote_code=trust_remote_code
    )


def _answer_ids(tokenizer: Any, answer: str) -> list[int]:
    ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"answer tokenized to an empty sequence: {answer!r}")
    return [int(token) for token in ids]


def convert(
    source: Path,
    destination: Path,
    tokenizer: Any,
    *,
    max_examples: int | None,
    strict_offset: bool,
) -> int:
    """Convert records and return the number written.

    The QCC evaluator predicts one token at ``target_position``.  RULER may
    request multiple generated answer tokens; this adapter therefore records
    the first answer token and verifies that the provided RULER offset points
    at the same tokenization boundary.  Multi-token answer fidelity must be
    evaluated with RULER's generation scorer separately.
    """

    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive when provided")
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with source.open("r", encoding="utf-8") as src, destination.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_number, line in enumerate(src, start=1):
            if max_examples is not None and written >= max_examples:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: record must be an object")
            text = record.get("input")
            outputs = record.get("outputs")
            offset = record.get("token_position_answer")
            if not isinstance(text, str) or not text:
                raise ValueError(f"line {line_number}: missing non-empty input")
            if not isinstance(outputs, list) or not outputs or not all(
                isinstance(answer, str) and answer for answer in outputs
            ):
                raise ValueError(f"line {line_number}: outputs must be non-empty strings")
            if not isinstance(offset, int) or offset < 0:
                raise ValueError(f"line {line_number}: invalid token_position_answer")

            encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
            input_ids = [int(token) for token in encoded]
            answer_ids = _answer_ids(tokenizer, outputs[0])
            if offset >= len(input_ids):
                raise ValueError(
                    f"line {line_number}: answer offset {offset} outside {len(input_ids)} tokens"
                )
            if strict_offset:
                # RULER computes the offset before appending its answer prefix;
                # retaining this check catches tokenizer/template mismatches.
                prefix_ids = tokenizer(
                    text[: record.get("index", 0)]
                    if isinstance(record.get("index"), int)
                    else text,
                    add_special_tokens=False,
                )["input_ids"]
                if record.get("index") is not None and len(prefix_ids) > offset:
                    raise ValueError(
                        f"line {line_number}: tokenizer offset mismatch ({len(prefix_ids)} > {offset})"
                    )
            json.dump(
                {
                    "input_ids": input_ids,
                    "target_position": offset,
                    "answers": answer_ids,
                    "ruler_index": record.get("index"),
                    "ruler_outputs": outputs,
                },
                dst,
                separators=(",", ":"),
            )
            dst.write("\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--no-strict-offset",
        action="store_true",
        help="skip the optional tokenizer-boundary consistency check",
    )
    args = parser.parse_args()
    tokenizer = _load_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    count = convert(
        args.input,
        args.output,
        tokenizer,
        max_examples=args.max_examples,
        strict_offset=not args.no_strict_offset,
    )
    print(json.dumps({"input": str(args.input), "output": str(args.output), "records": count}))


if __name__ == "__main__":
    main()
