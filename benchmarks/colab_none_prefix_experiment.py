"""Colab CLI experiment for position-invariant prefix landmark retrieval.

The experiment intentionally uses the existing synthetic marker/value task only
as a mechanism diagnostic.  It trains with no positional embedding, then tests
the same checkpoint at 128K and million-token distances over every value ID.
Run with ``colab run --gpu T4 --timeout 1800``; the script clones the commit
under test so the remote result is reproducible.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path("/content/qcc-transformer")


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def write_all_values_dataset(path: Path, lengths: tuple[int, ...], vocab_size: int = 32) -> None:
    """Write one marker/value example for every value ID at each length."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for length in lengths:
            query = int(length * 0.9)
            for value in range(3, vocab_size):
                tokens = [2] * length
                tokens[0] = 1
                tokens[1] = value
                tokens[query] = 1
                tokens[query + 1] = value
                json.dump({"input_ids": tokens, "target_position": query, "answers": [value]}, stream, separators=(",", ":"))
                stream.write("\n")


def main() -> None:
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/Marchematics/qcc-transformer.git", str(ROOT)],
        check=True,
    )
    run([sys.executable, "-m", "pip", "install", "-q", "triton"])
    run([
        sys.executable,
        "benchmarks/train_synthetic_retrieval.py",
        "--steps", "1800",
        "--batch", "32",
        "--train-length", "128",
        "--query-position", "96",
        "--query-min", "32",
        "--query-max", "120",
        "--vocab-size", "32",
        "--d-model", "32",
        "--heads", "4",
        "--layers", "1",
        "--window-size", "16",
        "--num-codes", "32",
        "--max-position-embeddings", "1000001",
        "--position-encoding", "none",
        "--archive-persistent-landmark",
        "--archive-prefix-landmark",
        "--archive-prefix-pair-landmark",
        "--checkpoint", "artifacts/colab_none_prefix.pt",
        "--dataset", "artifacts/colab_none_prefix.jsonl",
        "--eval-lengths", "128",
        "--eval-examples", "8",
        "--seed", "46",
        "--device", "cuda",
    ])
    # Evaluate every value ID at three distances.  Records are streamed by the
    # evaluator, so only one long prompt is resident at a time.
    write_all_values_dataset(ROOT / "artifacts/colab_none_prefix_long.jsonl", (128_000, 500_000, 1_000_000))
    run([
        sys.executable,
        "benchmarks/evaluate_retrieval.py",
        "--dataset", "artifacts/colab_none_prefix_long.jsonl",
        "--checkpoint", "artifacts/colab_none_prefix.pt",
        "--vocab-size", "32",
        "--d-model", "32",
        "--layers", "1",
        "--heads", "4",
        "--window-size", "16",
        "--num-codes", "32",
        "--max-position-embeddings", "1000001",
        "--position-encoding", "none",
        "--archive-persistent-landmark",
        "--archive-prefix-landmark",
        "--archive-prefix-pair-landmark",
        "--chunk-size", "256",
        "--device", "cuda",
    ])
    print(json.dumps({"status": "completed", "note": "Position-free prefix landmark probe; not RULER."}))


if __name__ == "__main__":
    main()
