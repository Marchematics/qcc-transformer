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
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--exact-num-sets", type=int, default=128)
    parser.add_argument("--exact-ways", type=int, default=4)
    parser.add_argument("--exact-probe-sets", type=int, default=None)
    parser.add_argument("--archive-mix", type=float, default=0.125)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true", help="load the real checkpoint through bitsandbytes NF4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kv-head-policy", choices=("reject", "repeat"), default="repeat")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--decode-steps", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--min-native-context", type=int, default=None)
    parser.add_argument("--sla-seconds", type=float, default=None)
    parser.add_argument(
        "--protocol-locked",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="record that this custom workload was registered before execution",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    batch_sizes = [int(raw.strip()) for raw in args.batch_sizes.split(",") if raw.strip()]
    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise ValueError("batch-sizes must contain positive integers")
    if args.decode_steps < 0:
        raise ValueError("decode-steps must be non-negative")
    if args.sla_seconds is not None and args.sla_seconds <= 0:
        raise ValueError("sla-seconds must be positive")
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
            "--exact-num-sets", str(args.exact_num_sets),
            "--exact-ways", str(args.exact_ways),
            "--dtype", args.dtype,
            "--device", args.device,
            "--kv-head-policy", args.kv_head_policy,
            "--decode-steps", str(args.decode_steps),
            "--output", str(output),
        ]
        if args.run_id is not None:
            command.extend(["--run-id", args.run_id])
        if args.min_native_context is not None:
            command.extend(["--min-native-context", str(args.min_native_context)])
        if args.protocol_locked is not None:
            command.append("--protocol-locked" if args.protocol_locked else "--no-protocol-locked")
        if args.trust_remote_code:
            command.append("--trust-remote-code")
        if args.load_in_4bit:
            command.append("--load-in-4bit")
        if args.adapter is not None:
            command.extend([
                "--adapter", str(args.adapter),
                "--archive-mix", str(args.archive_mix),
            ])
            if args.exact_probe_sets is not None:
                command.extend(["--exact-probe-sets", str(args.exact_probe_sets)])
        completed = subprocess.run(command, text=True, stdout=log.open("w"), stderr=subprocess.STDOUT)
        point: dict[str, object] = {"batch_size": batch_size, "returncode": completed.returncode, "output": str(output), "log": str(log)}
        if output.exists():
            try:
                result = json.loads(output.read_text(encoding="utf-8"))
                point["result"] = result
                if isinstance(result, dict):
                    full = result.get("baseline_full_kv")
                    qcc = result.get("qcc_retrofit")
                    full_ok = isinstance(full, dict) and full.get("status") == "ok"
                    qcc_ok = isinstance(qcc, dict) and qcc.get("status") == "ok"
                    point["full_kv_within_sla"] = bool(
                        full_ok
                        and (
                            args.sla_seconds is None
                            or float(full.get("elapsed_seconds", float("inf"))) <= args.sla_seconds
                        )
                    )
                    point["qcc_within_sla"] = bool(
                        qcc_ok
                        and (
                            args.sla_seconds is None
                            or float(qcc.get("elapsed_seconds", float("inf"))) <= args.sla_seconds
                        )
                    )
            except json.JSONDecodeError as exc:
                point["parse_error"] = str(exc)
        points.append(point)
    full_ok = [p["batch_size"] for p in points if p.get("full_kv_within_sla") is True]
    qcc_ok = [p["batch_size"] for p in points if p.get("qcc_within_sla") is True]
    summary = {
        "model": args.model,
        "model_id": args.model_id or args.model,
        "run_id": args.run_id,
        "real_model": True,
        "synthetic": False,
        "protocol_locked": args.protocol_locked,
        "qcc_only": False,
        "total_tokens_per_request": args.total_tokens,
        "decode_steps": args.decode_steps,
        "sla_seconds": args.sla_seconds,
        "fixed_sla": args.sla_seconds is not None,
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
