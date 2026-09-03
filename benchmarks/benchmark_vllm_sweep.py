#!/usr/bin/env python3
"""Sweep stock-vLLM concurrency and select the largest fixed-SLA point.

Start one stock vLLM server for the requested mode, run this client at several
concurrency levels, and repeat it against the matched Full-KV/baseline server.
Only a point whose complete request set and p95 TTFT/TPOT satisfy the same
pre-registered SLA is eligible for the concurrency comparison.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or values != sorted(set(values)) or any(value <= 0 for value in values):
        raise ValueError("concurrencies must be sorted, unique, and positive")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--concurrencies", default="1,2,4,8,16")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--slo-ttft-ms", type=float, required=True)
    parser.add_argument("--slo-tpot-ms", type=float, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--vllm-version", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    concurrencies = _ints(args.concurrencies)
    if args.slo_ttft_ms <= 0 or args.slo_tpot_ms <= 0:
        raise ValueError("SLA limits must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("benchmark_vllm_server.py")
    points: list[dict[str, Any]] = []
    for concurrency in concurrencies:
        report_path = args.output_dir / f"concurrency_{concurrency}.json"
        log_path = args.output_dir / f"concurrency_{concurrency}.log"
        command = [
            sys.executable, str(script), "--url", args.url, "--model", args.model,
            "--model-id", args.model_id, "--workload", str(args.workload),
            "--label", args.label, "--run-id", args.run_id,
            "--concurrency", str(concurrency), "--max-tokens", str(args.max_tokens),
            "--slo-ttft-ms", str(args.slo_ttft_ms), "--slo-tpot-ms", str(args.slo_tpot_ms),
            "--timeout", str(args.timeout), "--warmup", str(args.warmup),
            "--context-length", str(args.context_length), "--gpu", args.gpu,
            "--vllm-version", args.vllm_version, "--output", str(report_path),
        ]
        if args.limit is not None:
            command.extend(("--limit", str(args.limit)))
        if args.server_pid is not None:
            command.extend(("--server-pid", str(args.server_pid)))
        completed = subprocess.run(
            command, stdout=log_path.open("w"), stderr=subprocess.STDOUT,
            text=True, check=False,
        )
        point: dict[str, Any] = {
            "concurrency": concurrency,
            "returncode": completed.returncode,
            "report": str(report_path),
            "log": str(log_path),
        }
        if report_path.exists():
            try:
                point["result"] = _load(report_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                point["parse_error"] = str(exc)
        points.append(point)
    eligible = [
        point for point in points
        if isinstance(point.get("result"), dict) and point["result"].get("sla_pass") is True
    ]
    selected = max(eligible, key=lambda point: point["concurrency"], default=None)
    result = {
        "schema": "qcc-vllm-sweep-v1",
        "run_id": args.run_id,
        "model_id": args.model_id,
        "label": args.label,
        "real_model": True,
        "synthetic": False,
        "stock_vllm": True,
        "streaming": True,
        "protocol_locked": True,
        "context_length": args.context_length,
        "gpu": args.gpu,
        "vllm_version": args.vllm_version,
        "sla": {"ttft_p95_limit_ms": args.slo_ttft_ms, "tpot_p95_limit_ms": args.slo_tpot_ms},
        "concurrencies": concurrencies,
        "points": points,
        "max_sla_concurrency": selected["concurrency"] if selected else None,
        "selected_report": selected.get("result") if selected else None,
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "max_sla_concurrency": result["max_sla_concurrency"]}, indent=2))
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
