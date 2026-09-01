"""Colab-ready CUDA/Triton smoke and target audit.

Run from a fresh checkout on a Colab GPU::

    %pip install -U "torch>=2.1" triton
    !python benchmarks/colab_gpu_smoke.py --lengths 8192,32768 --compare-full

The script never converts missing evidence into a pass.  Million-token runs
require ``--run-long`` and are QCC-only unless a feasible Full-KV limit is
explicitly supplied.  Use ``--checkpoint`` and ``--dataset`` for retrieval.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def _run_logged(cmd: list[str], path: Path) -> int:
    """Run a long job while preserving stdout/stderr even when it fails."""

    print("$", " ".join(cmd), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(f"returncode={completed.returncode} log={path}", flush=True)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="8192,32768")
    parser.add_argument("--compare-full", action="store_true")
    parser.add_argument("--full-max-length", type=int, default=16384)
    parser.add_argument("--run-long", action="store_true")
    parser.add_argument("--long-lengths", default="128000,1000000,4000000")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--num-codes", type=int, default=16)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--retrieval-context", type=int, default=1_000_000)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--output", type=Path, default=Path("artifacts/colab_gpu_smoke.json"))
    parser.add_argument(
        "--long-output-dir",
        type=Path,
        default=Path("artifacts/colab_long"),
        help="directory for one stdout/stderr log per requested long length",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. In Colab select Runtime > Change runtime type > T4 GPU.")
    try:
        import triton  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Triton is unavailable; run `%pip install -U triton` first.") from exc

    print("torch=", torch.__version__, "cuda=", torch.version.cuda, "device=", torch.cuda.get_device_name(0))
    _run([sys.executable, "-m", "pytest", "-q"])
    lengths = ",".join(x.strip() for x in args.lengths.split(",") if x.strip())
    audit = [
        sys.executable,
        "benchmarks/audit_targets.py",
        "--run-latency",
        "--device",
        "cuda",
        "--lengths",
        lengths,
        "--full-max-length",
        str(args.full_max_length),
        "--chunk-size",
        str(args.chunk_size),
        "--vocab-size",
        str(args.vocab_size),
        "--d-model",
        str(args.d_model),
        "--layers",
        str(args.layers),
        "--heads",
        str(args.heads),
        "--window-size",
        str(args.window_size),
        "--num-codes",
        str(args.num_codes),
        "--quality-lengths",
        "1024,2048",
        "--json",
    ]
    if args.compare_full:
        audit.append("--compare-full")
    if args.checkpoint is not None:
        audit += ["--checkpoint", str(args.checkpoint)]
    if args.dataset is not None:
        audit += ["--dataset", str(args.dataset), "--retrieval-context", str(args.retrieval_context)]
        if args.max_examples is not None:
            audit += ["--max-examples", str(args.max_examples)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            audit,
            cwd=ROOT,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise SystemExit(
            f"audit failed with returncode={completed.returncode}; see {args.output}"
        )
    print("wrote", args.output)
    long_results: list[dict[str, object]] = []
    if args.run_long:
        for raw in args.long_lengths.split(","):
            length = raw.strip()
            if not length:
                continue
            log_path = args.long_output_dir / f"length_{length}.log"
            returncode = _run_logged([
                sys.executable,
                "benchmarks/benchmark_long_context.py",
                "--length", length,
                "--chunk-size", str(args.chunk_size),
                "--vocab-size", str(args.vocab_size),
                "--d-model", str(args.d_model),
                "--layers", str(args.layers),
                "--heads", str(args.heads),
                "--window-size", str(args.window_size),
                "--num-codes", str(args.num_codes),
                "--device", "cuda",
                "--run",
            ], log_path)
            long_results.append(
                {"length": int(length), "returncode": returncode, "log": str(log_path)}
            )
    status = "complete" if all(item["returncode"] == 0 for item in long_results) else "partial"
    print(
        json.dumps(
            {"status": status, "audit": str(args.output), "cuda": True, "long_runs": long_results},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
