"""Measure real-HF QCC/Full-KV scaling at several native context lengths.

Each length runs in a fresh subprocess through ``benchmark_hf_streaming_memory.py``
so one failed Full-KV allocation cannot contaminate the next point.  The summary
keeps a point matched only when both real-model runs reach the requested length.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _point(length: int, result: dict[str, Any] | None) -> dict[str, Any]:
    baseline = result.get("baseline_full_kv", {}) if result else {}
    qcc = result.get("qcc_retrofit", {}) if result else {}
    both_ok = baseline.get("status") == "ok" and qcc.get("status") == "ok"
    return {
        "context_tokens": length,
        "qcc_state_bytes": qcc.get("attention_state_bytes"),
        "full_kv_state_bytes": baseline.get("attention_state_bytes"),
        "tpot_ms": qcc.get("tpot_ms"),
        "full_kv_tpot_ms": baseline.get("tpot_ms"),
        "qcc_prefill_tokens_per_second": qcc.get("tokens_per_second"),
        "full_kv_prefill_tokens_per_second": baseline.get("tokens_per_second"),
        "matched_full_kv": both_ok,
        "measured": both_ok,
        "full_kv_status": baseline.get("status"),
        "qcc_status": qcc.get("status"),
        "result_derived": result.get("derived") if result else None,
    }


def _native_context(config: Any) -> int | None:
    values = [
        getattr(config, name, None)
        for name in ("max_position_embeddings", "n_positions", "max_sequence_length")
    ]
    values = [int(value) for value in values if isinstance(value, int) and value > 0]
    return max(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lengths", default="128000,256000,512000,1000000")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--num-codes", type=int, default=16)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--exact-num-sets", type=int, default=128)
    parser.add_argument("--exact-ways", type=int, default=4)
    parser.add_argument("--exact-probe-sets", type=int, default=None)
    parser.add_argument("--archive-mix", type=float, default=0.125)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kv-head-policy", choices=("reject", "repeat"), default="repeat")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--min-native-context", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--protocol-locked", action=argparse.BooleanOptionalAction, default=None,
        help="mark this source-controlled scaling protocol as pre-registered",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.chunk_size <= 0 or args.decode_steps < 0:
        raise ValueError("chunk-size must be positive and decode-steps non-negative")
    lengths = [int(raw.strip()) for raw in args.lengths.split(",") if raw.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("lengths must contain positive integers")
    if len(set(lengths)) != len(lengths):
        raise ValueError("lengths must not contain duplicates")
    required_lengths = {128_000, 256_000, 512_000, 1_000_000}
    if not required_lengths.issubset(lengths):
        raise ValueError("scaling protocol requires 128K, 256K, 512K, and 1M points")
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise SystemExit("install qcc-transformer[hf] to inspect native context") from exc
    config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    native_context_tokens = _native_context(config)
    target_native_context = max(lengths)
    if native_context_tokens is None or native_context_tokens < target_native_context:
        raise ValueError(
            "scaling protocol requires native context >= the longest point: "
            f"native={native_context_tokens}, requested={target_native_context}"
        )
    if args.min_native_context is not None and args.min_native_context > native_context_tokens:
        raise ValueError(
            "--min-native-context exceeds the checkpoint native context: "
            f"native={native_context_tokens}, requested={args.min_native_context}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    points: list[dict[str, Any]] = []
    for length in lengths:
        output = args.output.parent / f"{args.output.stem}_{length}.json"
        log = args.output.parent / f"{args.output.stem}_{length}.log"
        command = [
            sys.executable,
            str(Path(__file__).with_name("benchmark_hf_streaming_memory.py")),
            "--model", args.model,
            "--total-tokens", str(length),
            "--chunk-size", str(args.chunk_size),
            "--decode-steps", str(args.decode_steps),
            "--window-size", str(args.window_size),
            "--num-codes", str(args.num_codes),
            "--exact-num-sets", str(args.exact_num_sets),
            "--exact-ways", str(args.exact_ways),
            "--dtype", args.dtype,
            "--device", args.device,
            "--kv-head-policy", args.kv_head_policy,
            "--output", str(output),
        ]
        for flag in ("--load-in-4bit", "--trust-remote-code"):
            if getattr(args, flag[2:].replace("-", "_")):
                command.append(flag)
        command.extend(["--min-native-context", str(target_native_context)])
        if args.adapter is not None:
            command.extend([
                "--adapter", str(args.adapter),
                "--archive-mix", str(args.archive_mix),
            ])
            if args.exact_probe_sets is not None:
                command.extend(["--exact-probe-sets", str(args.exact_probe_sets)])
        if args.run_id is not None:
            command.extend(["--run-id", args.run_id])
        if args.protocol_locked is not None:
            command.append("--protocol-locked" if args.protocol_locked else "--no-protocol-locked")
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command, text=True, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        result = _load(output)
        point = _point(length, result)
        point.update({"returncode": completed.returncode, "output": str(output), "log": str(log)})
        points.append(point)
    result = {
        "schema": "qcc-hf-scaling-v1",
        "model_id": args.model,
        "run_id": args.run_id,
        "matched_full_kv": all(point["matched_full_kv"] for point in points),
        "real_model": True,
        "synthetic": False,
        "protocol_locked": args.protocol_locked,
        "qcc_only": False,
        "native_context_required": target_native_context,
        "native_context_tokens": native_context_tokens,
        "lengths": lengths,
        "points": points,
        "note": "Real-HF scaling diagnostic; a point is paired only when both sides complete.",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
