"""Fail-closed audit for the extended 99 gate.

The gate is intentionally evidence-only: it never infers a result from a
checkpoint name, a synthetic benchmark, or an unpaired timing.  A JSON report
must describe one real pretrained 1--7B model and one run id shared by all
sections.  Missing fields, non-finite values, ``synthetic`` evidence, and
unmatched Full-KV controls all fail the gate.

Example::

    python benchmarks/gate_99.py --evidence artifacts/gates/run.json

The accepted schema is documented in ``README.md`` and is deliberately plain
JSON so Colab/vLLM runners can emit it without importing this package.  In
addition to the original model/quality/latency/memory/calibration sections,
the final gate requires retrieval_1m, tail_safety, pareto_dominance,
production_latency, scaling_law, and generalization evidence.
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
RETRIEVAL_RATE = 0.99
TAIL_RATE = 0.95
STATE_VARIATION = 1.25
MIN_REPRODUCTIONS = 2


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
    _require(section.get("matched_full_kv") is True, f"{name} is not matched Full-KV", failures)
    _require(section.get("real_model") is True, f"{name} is not marked real_model", failures)
    _require(section.get("official") is True, f"{name} is not marked official", failures)
    _require(section.get("synthetic") is False, f"{name} is synthetic", failures)
    _require(section.get("qcc_only") is False, f"{name} is QCC-only", failures)


def _same_model(section: Any, name: str, model_id: Any, failures: list[str]) -> None:
    if not isinstance(section, dict):
        return
    _require(
        isinstance(section.get("model_id"), str) and section.get("model_id") == model_id,
        f"{name}.model_id does not match model.model_id",
        failures,
    )


def _run_id(section: Any, name: str, run_id: Any, failures: list[str]) -> None:
    if isinstance(section, dict):
        _require(section.get("run_id") == run_id, f"{name}.run_id does not match root run_id", failures)


def _extended_audit(payload: dict[str, Any], run_id: Any, model_id: Any, failures: list[str]) -> dict[str, Any]:
    """Audit the additional anti-cherry-picking and production gates.

    These sections are mandatory.  A missing section is a failure, even when
    the original five-part evidence is complete.
    """
    summary: dict[str, Any] = {}
    retrieval = payload.get("retrieval_1m")
    _matched(retrieval, "retrieval_1m", failures)
    _same_model(retrieval, "retrieval_1m", model_id, failures)
    if isinstance(retrieval, dict):
        trials = _number(retrieval.get("trials"), "retrieval_1m.trials", failures)
        qcc_rate = _number(retrieval.get("qcc_success_rate"), "retrieval_1m.qcc_success_rate", failures)
        full_rate = _number(retrieval.get("full_kv_success_rate"), "retrieval_1m.full_kv_success_rate", failures)
        if trials is not None:
            _require(trials >= 1000, "retrieval_1m requires at least 1000 trials", failures)
        if qcc_rate is not None:
            _require(qcc_rate >= RETRIEVAL_RATE, f"1M retrieval rate {qcc_rate:.6f} < {RETRIEVAL_RATE:.2f}", failures)
        if full_rate is not None:
            _require(full_rate > 0, "retrieval_1m.full_kv_success_rate must be > 0", failures)
        if qcc_rate is not None and full_rate is not None and full_rate > 0:
            ratio = qcc_rate / full_rate
            summary["retrieval_ratio"] = ratio
            _require(ratio >= RETRIEVAL_RATE, f"1M retrieval ratio {ratio:.6f} < {RETRIEVAL_RATE:.2f}", failures)
        for field in ("random_depth", "multi_needle", "semantic_distractor"):
            _require(retrieval.get(field) is True, f"retrieval_1m.{field} evidence is missing", failures)
    _run_id(retrieval, "retrieval_1m", run_id, failures)

    tail = payload.get("tail_safety")
    _matched(tail, "tail_safety", failures)
    _same_model(tail, "tail_safety", model_id, failures)
    if isinstance(tail, dict):
        miss = _number(tail.get("catastrophic_retrieval_miss_rate"), "tail_safety.catastrophic_retrieval_miss_rate", failures)
        _require(miss is not None and miss < 0.01, "catastrophic retrieval miss rate must be < 1%", failures)
        buckets = tail.get("critical_buckets")
        _require(isinstance(buckets, list) and bool(buckets), "tail_safety.critical_buckets is missing", failures)
        if isinstance(buckets, list):
            for index, bucket in enumerate(buckets):
                ratio = _number(bucket.get("qcc_full_kv_ratio") if isinstance(bucket, dict) else None, f"tail_safety.critical_buckets[{index}].qcc_full_kv_ratio", failures)
                if ratio is not None:
                    _require(ratio >= TAIL_RATE, f"tail bucket {index} ratio {ratio:.6f} < {TAIL_RATE:.2f}", failures)
    _run_id(tail, "tail_safety", run_id, failures)

    pareto = payload.get("pareto_dominance")
    _matched(pareto, "pareto_dominance", failures)
    _same_model(pareto, "pareto_dominance", model_id, failures)
    if isinstance(pareto, dict):
        baselines = pareto.get("baselines")
        _require(isinstance(baselines, list), "pareto_dominance.baselines is missing", failures)
        if isinstance(baselines, list):
            names = {item.get("name") for item in baselines if isinstance(item, dict)}
            _require("fp8_full_kv" in names, "pareto dominance must include fp8_full_kv", failures)
            _require(len(baselines) >= 3, "pareto dominance requires FP8 plus two compression baselines", failures)
            for index, item in enumerate(baselines):
                _require(isinstance(item, dict) and item.get("qcc_dominates") is True, f"pareto baseline {index} is not dominated", failures)
    _run_id(pareto, "pareto_dominance", run_id, failures)

    production = payload.get("production_latency")
    _matched(production, "production_latency", failures)
    _same_model(production, "production_latency", model_id, failures)
    if isinstance(production, dict):
        ttft = _number(production.get("ttft_regression"), "production_latency.ttft_regression", failures)
        p95 = _number(production.get("p95_tpot_speedup"), "production_latency.p95_tpot_speedup", failures)
        p99 = _number(production.get("p99_tpot_speedup"), "production_latency.p99_tpot_speedup", failures)
        tradeoff = production.get("throughput_latency_tradeoff")
        if ttft is not None:
            _require(ttft <= 0, "TTFT regression must be <= 0", failures)
        if p95 is not None:
            _require(p95 >= 1.0, "p95 TPOT must not regress", failures)
        if p99 is not None:
            _require(p99 >= 1.0, "p99 TPOT must not regress", failures)
        _require(tradeoff is False, "throughput/latency tradeoff evidence is not acceptable", failures)
    _run_id(production, "production_latency", run_id, failures)

    scaling = payload.get("scaling_law")
    _matched(scaling, "scaling_law", failures)
    _same_model(scaling, "scaling_law", model_id, failures)
    if isinstance(scaling, dict):
        points = scaling.get("points")
        _require(isinstance(points, list), "scaling_law.points is missing", failures)
        required_lengths = {128_000, 256_000, 512_000, 1_000_000}
        seen_lengths: set[int] = set()
        states: list[float] = []
        tpots: list[float] = []
        if isinstance(points, list):
            for index, point in enumerate(points):
                if not isinstance(point, dict):
                    failures.append(f"scaling_law.points[{index}] must be an object")
                    continue
                length = _number(point.get("context_tokens"), f"scaling_law.points[{index}].context_tokens", failures)
                state = _number(point.get("qcc_state_bytes"), f"scaling_law.points[{index}].qcc_state_bytes", failures)
                tpot = _number(point.get("tpot_ms"), f"scaling_law.points[{index}].tpot_ms", failures)
                if length is not None: seen_lengths.add(int(length))
                if state is not None: states.append(state)
                if tpot is not None: tpots.append(tpot)
                _require(point.get("matched_full_kv") is True, f"scaling_law.points[{index}] is not matched Full-KV", failures)
        _require(required_lengths.issubset(seen_lengths), "scaling law requires 128K/256K/512K/1M points", failures)
        for values, label in ((states, "state"), (tpots, "TPOT")):
            if values:
                variation = max(values) / max(min(values), 1e-12)
                summary[f"{label.lower()}_variation"] = variation
                _require(variation <= STATE_VARIATION, f"scaling {label} variation {variation:.6f} > {STATE_VARIATION:.2f}", failures)
    _run_id(scaling, "scaling_law", run_id, failures)

    general = payload.get("generalization")
    _matched(general, "generalization", failures)
    if isinstance(general, dict):
        for field in ("model_families", "gpu_generations", "independent_reproductions"):
            value = _number(general.get(field), f"generalization.{field}", failures)
            if value is not None:
                _require(value >= MIN_REPRODUCTIONS, f"generalization.{field} must be >= {MIN_REPRODUCTIONS}", failures)
    _run_id(general, "generalization", run_id, failures)
    summary["extended_sections"] = 6
    return summary


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a complete gate decision and human-readable failure reasons."""

    failures: list[str] = []
    if not isinstance(payload, dict):
        return {"passed": False, "failures": ["evidence must be a JSON object"]}

    run_id = payload.get("run_id")
    model = payload.get("model")
    model_id = model.get("model_id") if isinstance(model, dict) else None
    _require(isinstance(run_id, str) and bool(run_id), "run_id is required", failures)
    _require(isinstance(model, dict), "model section is missing", failures)
    if isinstance(model, dict):
        _require(model.get("pretrained") is True, "model is not marked pretrained", failures)
        _require(model.get("real_checkpoint") is True, "real_checkpoint evidence is missing", failures)
        _require(isinstance(model_id, str) and bool(model_id), "model_id is required", failures)
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
            _same_model(section, f"quality.{benchmark}", model_id, failures)
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
    _same_model(latency, "vllm_latency", model_id, failures)
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
    _same_model(memory, "memory", model_id, failures)
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
        _require(calibration.get("hf_zero_code_changes") is True, "HF zero-code retrofit evidence is missing", failures)
        _require(calibration.get("vllm_zero_code_changes") is True, "vLLM zero-code integration evidence is missing", failures)
        _require(calibration.get("run_id") == run_id, "calibration.run_id does not match root run_id", failures)

    extended_summary = _extended_audit(payload, run_id, model_id, failures)

    return {
        "passed": not failures,
        "run_id": run_id,
        "model_id": model_id,
        "quality_ratios": quality_ratios,
        "latency": latency_summary,
        "memory": memory_summary,
        "calibration": calibration_summary,
        "extended": extended_summary,
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
