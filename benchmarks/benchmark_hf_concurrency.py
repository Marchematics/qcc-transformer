"""Sweep independent-request concurrency for a real HF checkpoint.

Each point invokes the matched streaming benchmark with the same per-request
context and chunk size.  Full-KV and QCC are loaded in the same subprocess, so
CUDA peak measurements are isolated and an OOM is retained as evidence rather
than treated as a missing datapoint.  This is a diagnostic input to the 99
gate; it is not a substitute for a vLLM scheduler run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--total-tokens", type=int, default=131072)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--num-codes", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kv-head-policy", choices=("reject", "repeat"), default="repeat")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    batch_sizes = [int(raw.strip()) for raw in args.batch_sizes.split(",") if raw.strip()]
    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise ValueError("batch-sizes must contain positive integers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points: list[dict[str, object]] = []
    for batch_size in batch_sizes:
        output = args.output_dir / f"batch_{batch_size}.json"
        log = args.output_dir / f"batch_{batch_size}.log"
        command = [
            sys.executable,
            str(Path(__file__).with_name("benchmark_hf_streaming_memory.py")),
            "--model", args.model,
            "--total-tokens", str(args.total_tokens),
            "--batch-size", str(batch_size),
            "--chunk-size", str(args.chunk_size),
            "--window-size", str(args.window_size),
            "--num-codes", str(args.num_codes),
            "--device", args.device,
            "--kv-head-policy", args.kv_head_policy,
            "--output", str(output),
        ]
        if args.trust_remote_code:
            command.append("--trust-remote-code")
        completed = subprocess.run(command, text=True, stdout=log.open("w"), stderr=subprocess.STDOUT)
        point: dict[str, object] = {"batch_size": batch_size, "returncode": completed.returncode, "output": str(output), "log": str(log)}
        if output.exists():
            try:
                point["result"] = json.loads(output.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                point["parse_error"] = str(exc)
        points.append(point)
    full_ok = [p["batch_size"] for p in points if isinstance(p.get("result"), dict) and p["result"].get("baseline_full_kv", {}).get("status") == "ok"]
    qcc_ok = [p["batch_size"] for p in points if isinstance(p.get("result"), dict) and p["result"].get("qcc_retrofit", {}).get("status") == "ok"]
    summary = {
        "model": args.model,
        "total_tokens_per_request": args.total_tokens,
        "batch_sizes": batch_sizes,
        "points": points,
        "max_full_kv_batch": max(full_ok) if full_ok else None,
        "max_qcc_batch": max(qcc_ok) if qcc_ok else None,
        "concurrency_ratio": (max(qcc_ok) / max(full_ok)) if full_ok and qcc_ok else None,
        "note": "HF diagnostic only; use matched real-vLLM evidence for gate_99.",
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
