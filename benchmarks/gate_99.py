"""Fail-closed audit for the five-part 99 gate.

The gate is intentionally evidence-only: it never infers a result from a
checkpoint name, a synthetic benchmark, or an unpaired timing.  A JSON report
must describe one real pretrained 1--7B model and one run id shared by all
sections.  Missing fields, non-finite values, ``synthetic`` evidence, and
unmatched Full-KV controls all fail the gate.

Example::

    python benchmarks/gate_99.py --evidence artifacts/gates/run.json

The accepted schema is documented in ``README.md`` and is deliberately plain
JSON so Colab/vLLM runners can emit it without importing this package.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


MIN_PARAMS = 1_000_000_000
MAX_PARAMS = 7_000_000_000
QUALITY_RATIO = 0.98
TPOT_SPEEDUP = 5.0
THROUGHPUT_SPEEDUP = 2.0
MEMORY_REDUCTION = 0.80
CONCURRENCY_SPEEDUP = 4.0
CALIBRATION_FRACTION = 0.01


def _number(value: Any, name: str, failures: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        failures.append(f"{name} must be a finite number")
        return None
    result = float(value)
    if not math.isfinite(result):
        failures.append(f"{name} must be a finite number")
        return None
    return result


def _require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _matched(section: Any, name: str, failures: list[str]) -> None:
    _require(isinstance(section, dict), f"{name} section is missing", failures)
    if not isinstance(section, dict):
        return
    _require(bool(section.get("matched_full_kv")), f"{name} is not matched Full-KV", failures)
    _require(bool(section.get("real_model")), f"{name} is not marked real_model", failures)
    _require(bool(section.get("official")), f"{name} is not marked official", failures)
    _require(not bool(section.get("synthetic")), f"{name} is synthetic", failures)
    _require(not bool(section.get("qcc_only")), f"{name} is QCC-only", failures)


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a complete gate decision and human-readable failure reasons."""

    failures: list[str] = []
    if not isinstance(payload, dict):
        return {"passed": False, "failures": ["evidence must be a JSON object"]}

    run_id = payload.get("run_id")
    model = payload.get("model")
    _require(isinstance(run_id, str) and bool(run_id), "run_id is required", failures)
    _require(isinstance(model, dict), "model section is missing", failures)
    if isinstance(model, dict):
        _require(bool(model.get("pretrained")), "model is not marked pretrained", failures)
        _require(bool(model.get("real_checkpoint")), "real_checkpoint evidence is missing", failures)
        _require(isinstance(model.get("model_id"), str) and bool(model.get("model_id")), "model_id is required", failures)
        params = _number(model.get("parameter_count"), "model.parameter_count", failures)
        if params is not None:
            _require(
                MIN_PARAMS <= params <= MAX_PARAMS,
                "parameter_count must be within 1B..7B",
                failures,
            )

    quality = payload.get("quality")
    _require(isinstance(quality, dict), "quality section is missing", failures)
    quality_ratios: dict[str, float] = {}
    if isinstance(quality, dict):
        for benchmark in ("ruler", "longbench", "pg19"):
            section = quality.get(benchmark)
            _matched(section, f"quality.{benchmark}", failures)
            if isinstance(section, dict):
                qcc = _number(section.get("qcc_score"), f"quality.{benchmark}.qcc_score", failures)
                full = _number(section.get("full_kv_score"), f"quality.{benchmark}.full_kv_score", failures)
                if qcc is not None and full is not None:
                    _require(full > 0, f"quality.{benchmark}.full_kv_score must be > 0", failures)
                    if full > 0:
                        ratio = qcc / full
                        quality_ratios[benchmark] = ratio
                        _require(ratio >= QUALITY_RATIO, f"quality.{benchmark} ratio {ratio:.6f} < {QUALITY_RATIO:.2f}", failures)
                _require(section.get("run_id") == run_id, f"quality.{benchmark}.run_id does not match root run_id", failures)

    latency = payload.get("vllm_latency")
    _matched(latency, "vllm_latency", failures)
    latency_summary: dict[str, float] = {}
    if isinstance(latency, dict):
        context = _number(latency.get("context_tokens"), "vllm_latency.context_tokens", failures)
        if context is not None:
            _require(context >= 128_000, "vllm context_tokens is below 128K", failures)
        tpot_speed = _number(latency.get("tpot_speedup"), "vllm_latency.tpot_speedup", failures)
        throughput_speed = _number(latency.get("throughput_speedup"), "vllm_latency.throughput_speedup", failures)
        if tpot_speed is not None:
            latency_summary["tpot_speedup"] = tpot_speed
            _require(tpot_speed >= TPOT_SPEEDUP, f"vLLM TPOT speedup {tpot_speed:.6f} < {TPOT_SPEEDUP:.1f}", failures)
        if throughput_speed is not None:
            latency_summary["throughput_speedup"] = throughput_speed
            _require(throughput_speed >= THROUGHPUT_SPEEDUP, f"vLLM throughput speedup {throughput_speed:.6f} < {THROUGHPUT_SPEEDUP:.1f}", failures)
        _require(latency.get("run_id") == run_id, "vllm_latency.run_id does not match root run_id", failures)
        _require(isinstance(latency.get("vllm_version"), str) and bool(latency.get("vllm_version")), "vllm_version is required", failures)

    memory = payload.get("memory")
    _matched(memory, "memory", failures)
    memory_summary: dict[str, float] = {}
    if isinstance(memory, dict):
        full_peak = _number(memory.get("full_kv_peak_bytes"), "memory.full_kv_peak_bytes", failures)
        qcc_peak = _number(memory.get("qcc_peak_bytes"), "memory.qcc_peak_bytes", failures)
        if full_peak is not None and qcc_peak is not None:
            _require(full_peak > 0, "memory.full_kv_peak_bytes must be > 0", failures)
            reduction = 1.0 - qcc_peak / full_peak if full_peak > 0 else float("nan")
            memory_summary["peak_reduction"] = reduction
            _require(reduction >= MEMORY_REDUCTION, f"peak memory reduction {reduction:.6f} < {MEMORY_REDUCTION:.2f}", failures)
        full_concurrency = _number(memory.get("full_kv_concurrency"), "memory.full_kv_concurrency", failures)
        qcc_concurrency = _number(memory.get("qcc_concurrency"), "memory.qcc_concurrency", failures)
        if full_concurrency is not None and qcc_concurrency is not None:
            _require(full_concurrency > 0, "memory.full_kv_concurrency must be > 0", failures)
            concurrency = qcc_concurrency / full_concurrency if full_concurrency > 0 else float("nan")
            memory_summary["concurrency_speedup"] = concurrency
            _require(concurrency >= CONCURRENCY_SPEEDUP, f"long-context concurrency speedup {concurrency:.6f} < {CONCURRENCY_SPEEDUP:.1f}", failures)
        _require(memory.get("run_id") == run_id, "memory.run_id does not match root run_id", failures)

    calibration = payload.get("calibration")
    _require(isinstance(calibration, dict), "calibration section is missing", failures)
    calibration_summary: dict[str, float] = {}
    if isinstance(calibration, dict):
        fraction = _number(calibration.get("trainable_parameter_fraction"), "calibration.trainable_parameter_fraction", failures)
        if fraction is not None:
            calibration_summary["trainable_parameter_fraction"] = fraction
            _require(0.0 <= fraction <= CALIBRATION_FRACTION, f"trainable parameter fraction {fraction:.6f} > {CALIBRATION_FRACTION:.2f}", failures)
        _require(bool(calibration.get("hf_zero_code_changes")), "HF zero-code retrofit evidence is missing", failures)
        _require(bool(calibration.get("vllm_zero_code_changes")), "vLLM zero-code integration evidence is missing", failures)
        _require(calibration.get("run_id") == run_id, "calibration.run_id does not match root run_id", failures)

    return {
        "passed": not failures,
        "run_id": run_id,
        "model_id": model.get("model_id") if isinstance(model, dict) else None,
        "quality_ratios": quality_ratios,
        "latency": latency_summary,
        "memory": memory_summary,
        "calibration": calibration_summary,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = json.loads(args.evidence.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "failures": [f"cannot read evidence: {exc}"]}, indent=2))
        return 2
    result = audit(payload)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
