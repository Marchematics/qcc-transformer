"""Fail-closed QCC production acceptance gate.

Official public benchmarks (RULER/LongBench/PG-19) must be marked ``official``.
Project-defined serving/retrieval protocols instead must be pre-registered with
``protocol_locked=true``; they must never be mislabeled official merely to pass a gate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

MIN_PARAMS = 1_000_000_000
MAX_PARAMS = 7_000_000_000
QUALITY_RATIO = 0.98
RETRIEVAL_RATE = 0.99
TAIL_RATIO = 0.95
TPOT_SPEEDUP = 5.0
THROUGHPUT_SPEEDUP = 2.0
STATE_REDUCTION = 0.80
CONCURRENCY_SPEEDUP = 4.0
TRAINABLE_FRACTION = 0.01
SCALING_VARIATION = 1.25
MIN_REPRODUCTIONS = 2


def _require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _number(value: Any, name: str, failures: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        failures.append(f"{name} must be a finite number")
        return None
    value = float(value)
    if not math.isfinite(value):
        failures.append(f"{name} must be a finite number")
        return None
    return value


def _common(
    section: Any,
    name: str,
    *,
    model_id: Any,
    run_id: Any,
    failures: list[str],
    official: bool,
    matched: bool = True,
) -> bool:
    if not isinstance(section, dict):
        failures.append(f"{name} section is missing")
        return False
    if matched:
        _require(section.get("matched_full_kv") is True, f"{name} is not matched Full-KV", failures)
    _require(section.get("real_model") is True, f"{name} is not marked real_model", failures)
    _require(section.get("synthetic") is False, f"{name} is synthetic", failures)
    _require(section.get("qcc_only") is False, f"{name} is QCC-only", failures)
    _require(section.get("model_id") == model_id, f"{name}.model_id does not match model.model_id", failures)
    _require(section.get("run_id") == run_id, f"{name}.run_id does not match root run_id", failures)
    if official:
        _require(section.get("official") is True, f"{name} is not marked official", failures)
    else:
        _require(
            section.get("protocol_locked") is True,
            f"{name} custom protocol is not locked before execution",
            failures,
        )
    return True


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    summary: dict[str, Any] = {}
    if not isinstance(payload, dict):
        return {"passed": False, "failures": ["evidence must be a JSON object"], "summary": {}}

    run_id = payload.get("run_id")
    _require(isinstance(run_id, str) and bool(run_id), "run_id is required", failures)
    model = payload.get("model")
    _require(isinstance(model, dict), "model section is missing", failures)
    model_id = model.get("model_id") if isinstance(model, dict) else None
    if isinstance(model, dict):
        _require(model.get("pretrained") is True, "model is not marked pretrained", failures)
        _require(model.get("real_checkpoint") is True, "real_checkpoint evidence is missing", failures)
        _require(isinstance(model_id, str) and bool(model_id), "model_id is required", failures)
        params = _number(model.get("parameter_count"), "model.parameter_count", failures)
        if params is not None:
            _require(MIN_PARAMS <= params <= MAX_PARAMS, "parameter_count must be within 1B..7B", failures)
        native_context = _number(model.get("native_context_tokens"), "model.native_context_tokens", failures)
        if native_context is not None:
            _require(native_context >= 128_000, "model native context is below 128K", failures)

    # 1/2: one real pretrained model, official long-context quality >=98% Full-KV.
    quality = payload.get("quality")
    _require(isinstance(quality, dict), "quality section is missing", failures)
    quality_ratios: dict[str, float] = {}
    if isinstance(quality, dict):
        for benchmark in ("ruler", "longbench", "pg19"):
            section = quality.get(benchmark)
            if _common(section, f"quality.{benchmark}", model_id=model_id, run_id=run_id, failures=failures, official=True):
                qcc = _number(section.get("qcc_score"), f"quality.{benchmark}.qcc_score", failures)
                full = _number(section.get("full_kv_score"), f"quality.{benchmark}.full_kv_score", failures)
                if qcc is not None and full is not None:
                    _require(full > 0, f"quality.{benchmark}.full_kv_score must be > 0", failures)
                    if full > 0:
                        ratio = qcc / full
                        quality_ratios[benchmark] = ratio
                        _require(ratio >= QUALITY_RATIO, f"quality.{benchmark} ratio {ratio:.6f} < {QUALITY_RATIO:.2f}", failures)
    summary["quality_ratios"] = quality_ratios

    # 3: 128K stock-vLLM real serving gains.
    latency = payload.get("vllm_latency")
    if _common(latency, "vllm_latency", model_id=model_id, run_id=run_id, failures=failures, official=False):
        _require(latency.get("stock_vllm") is True, "vllm_latency is not stock vLLM", failures)
        context = _number(latency.get("context_tokens"), "vllm_latency.context_tokens", failures)
        tpot = _number(latency.get("tpot_speedup"), "vllm_latency.tpot_speedup", failures)
        throughput = _number(latency.get("throughput_speedup"), "vllm_latency.throughput_speedup", failures)
        if context is not None:
            _require(context >= 128_000, "vllm context_tokens is below 128K", failures)
        if tpot is not None:
            _require(tpot >= TPOT_SPEEDUP, f"vLLM TPOT speedup {tpot:.6f} < {TPOT_SPEEDUP:.1f}", failures)
        if throughput is not None:
            _require(throughput >= THROUGHPUT_SPEEDUP, f"vLLM throughput speedup {throughput:.6f} < {THROUGHPUT_SPEEDUP:.1f}", failures)
        _require(isinstance(latency.get("vllm_version"), str) and bool(latency.get("vllm_version")), "vllm_version is required", failures)
        _require(isinstance(latency.get("gpu"), str) and bool(latency.get("gpu")), "vllm GPU identity is required", failures)
        workload_hash = latency.get("workload_sha256")
        _require(isinstance(workload_hash, str) and len(workload_hash) == 64, "vllm workload_sha256 is required", failures)

    # 4: actual attention-state reduction and fixed-SLA concurrency.
    memory = payload.get("memory")
    if _common(memory, "memory", model_id=model_id, run_id=run_id, failures=failures, official=False):
        full_state = _number(memory.get("full_kv_attention_state_bytes"), "memory.full_kv_attention_state_bytes", failures)
        qcc_state = _number(memory.get("qcc_attention_state_bytes"), "memory.qcc_attention_state_bytes", failures)
        full_conc = _number(memory.get("full_kv_concurrency"), "memory.full_kv_concurrency", failures)
        qcc_conc = _number(memory.get("qcc_concurrency"), "memory.qcc_concurrency", failures)
        _require(memory.get("fixed_sla") is True, "memory concurrency is not measured under a fixed SLA", failures)
        if full_state is not None and qcc_state is not None:
            _require(full_state > 0 and qcc_state >= 0, "attention-state bytes must be non-negative with Full-KV > 0", failures)
            if full_state > 0:
                reduction = 1.0 - qcc_state / full_state
                summary["attention_state_reduction"] = reduction
                _require(reduction >= STATE_REDUCTION, f"attention-state reduction {reduction:.6f} < {STATE_REDUCTION:.2f}", failures)
        full_peak = _number(memory.get("full_kv_peak_memory_bytes"), "memory.full_kv_peak_memory_bytes", failures)
        qcc_peak = _number(memory.get("qcc_peak_memory_bytes"), "memory.qcc_peak_memory_bytes", failures)
        if full_peak is not None and qcc_peak is not None:
            _require(full_peak > 0 and qcc_peak >= 0, "peak memory bytes must be non-negative with Full-KV > 0", failures)
            if full_peak > 0:
                peak_reduction = 1.0 - qcc_peak / full_peak
                summary["peak_memory_reduction"] = peak_reduction
                _require(peak_reduction >= STATE_REDUCTION, f"peak memory reduction {peak_reduction:.6f} < {STATE_REDUCTION:.2f}", failures)
        if full_conc is not None and qcc_conc is not None:
            _require(full_conc > 0, "full_kv_concurrency must be > 0", failures)
            if full_conc > 0:
                ratio = qcc_conc / full_conc
                summary["concurrency_speedup"] = ratio
                _require(ratio >= CONCURRENCY_SPEEDUP, f"concurrency speedup {ratio:.6f} < {CONCURRENCY_SPEEDUP:.1f}", failures)

    # 5: tiny retrofit and zero application model-code modifications.
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict):
        failures.append("calibration section is missing")
    else:
        _require(calibration.get("model_id") == model_id, "calibration.model_id does not match model.model_id", failures)
        _require(calibration.get("run_id") == run_id, "calibration.run_id does not match root run_id", failures)
        fraction = _number(calibration.get("trainable_parameter_fraction"), "calibration.trainable_parameter_fraction", failures)
        if fraction is not None:
            _require(fraction <= TRAINABLE_FRACTION, f"trainable parameter fraction {fraction:.6f} > {TRAINABLE_FRACTION:.2f}", failures)
        _require(calibration.get("hf_zero_code_changes") is True, "HF retrofit requires business-code changes", failures)
        _require(calibration.get("vllm_zero_code_changes") is True, "vLLM retrofit requires business-code changes", failures)

    # 0: locked, paired, real 1M retrieval; learned admission only.
    retrieval = payload.get("retrieval_1m")
    if _common(retrieval, "retrieval_1m", model_id=model_id, run_id=run_id, failures=failures, official=False):
        trials = _number(retrieval.get("trials"), "retrieval_1m.trials", failures)
        context = _number(retrieval.get("context_tokens"), "retrieval_1m.context_tokens", failures)
        qcc = _number(retrieval.get("qcc_success_rate"), "retrieval_1m.qcc_success_rate", failures)
        full = _number(retrieval.get("full_kv_success_rate"), "retrieval_1m.full_kv_success_rate", failures)
        if trials is not None:
            _require(trials >= 1000, "retrieval_1m requires at least 1000 trials", failures)
        if context is not None:
            _require(context >= 1_000_000, "retrieval_1m context is below 1M tokens", failures)
        native = _number(retrieval.get("native_context_tokens"), "retrieval_1m.native_context_tokens", failures)
        if native is not None and context is not None:
            _require(native >= context, "retrieval_1m native context is below the evaluated context", failures)
        if qcc is not None:
            _require(qcc >= RETRIEVAL_RATE, f"1M retrieval rate {qcc:.6f} < {RETRIEVAL_RATE:.2f}", failures)
        if full is not None:
            _require(full >= RETRIEVAL_RATE, f"Full-KV 1M retrieval rate {full:.6f} < {RETRIEVAL_RATE:.2f}", failures)
        if qcc is not None and full is not None and full > 0:
            ratio = qcc / full
            summary["retrieval_ratio"] = ratio
            _require(ratio >= RETRIEVAL_RATE, f"1M retrieval ratio {ratio:.6f} < {RETRIEVAL_RATE:.2f}", failures)
        for field in ("random_depth", "multi_needle", "semantic_distractor"):
            _require(retrieval.get(field) is True, f"retrieval_1m.{field} evidence is missing", failures)
        _require(retrieval.get("oracle_admission") is False, "retrieval_1m uses oracle admission", failures)
        manifest_hash = retrieval.get("manifest_sha256")
        _require(isinstance(manifest_hash, str) and len(manifest_hash) == 64, "retrieval_1m manifest_sha256 is required", failures)

    # 6: no average-score masking of catastrophic retrieval failures.
    tail = payload.get("tail_safety")
    if _common(tail, "tail_safety", model_id=model_id, run_id=run_id, failures=failures, official=False):
        miss = _number(tail.get("catastrophic_retrieval_miss_rate"), "tail_safety.catastrophic_retrieval_miss_rate", failures)
        if miss is not None:
            _require(miss < 0.01, "catastrophic retrieval miss rate must be < 1%", failures)
        buckets = tail.get("critical_buckets")
        _require(isinstance(buckets, list) and bool(buckets), "tail_safety.critical_buckets is missing", failures)
        if isinstance(buckets, list):
            names: set[str] = set()
            for index, bucket in enumerate(buckets):
                task = bucket.get("task") if isinstance(bucket, dict) else None
                context = bucket.get("context_tokens") if isinstance(bucket, dict) else None
                label = bucket.get("bucket") if isinstance(bucket, dict) else None
                _require(isinstance(task, str) and bool(task), f"tail_safety.critical_buckets[{index}].task is required", failures)
                _require(isinstance(label, str) and bool(label), f"tail_safety.critical_buckets[{index}].bucket is required", failures)
                if isinstance(label, str) and label:
                    names.add(label)
                context_value = _number(context, f"tail_safety.critical_buckets[{index}].context_tokens", failures)
                if context_value is not None:
                    _require(context_value >= 128_000, f"tail_safety bucket {index} is below 128K", failures)
                trials_value = _number(bucket.get("trials") if isinstance(bucket, dict) else None, f"tail_safety.critical_buckets[{index}].trials", failures)
                if trials_value is not None:
                    _require(trials_value > 0, f"tail_safety bucket {index} is empty", failures)
                full_score = _number(bucket.get("full_kv_score") if isinstance(bucket, dict) else None, f"tail_safety.critical_buckets[{index}].full_kv_score", failures)
                qcc_score = _number(bucket.get("qcc_score") if isinstance(bucket, dict) else None, f"tail_safety.critical_buckets[{index}].qcc_score", failures)
                ratio = _number(bucket.get("qcc_full_kv_ratio") if isinstance(bucket, dict) else None, f"tail_safety.critical_buckets[{index}].qcc_full_kv_ratio", failures)
                if full_score is not None and qcc_score is not None and ratio is not None:
                    _require(full_score > 0, f"tail_safety bucket {index} Full-KV score must be > 0", failures)
                    if full_score > 0:
                        _require(math.isclose(ratio, qcc_score / full_score, rel_tol=1e-6, abs_tol=1e-9), f"tail_safety bucket {index} ratio is not recomputed from raw scores", failures)
                if ratio is not None:
                    _require(ratio >= TAIL_RATIO, f"tail bucket {index} ratio {ratio:.6f} < {TAIL_RATIO:.2f}", failures)
            _require({"0-25%", "25-50%", "50-75%", "75-100%"}.issubset(names), "tail_safety must cover all random-depth buckets", failures)

    # 7: dominate FP8 Full-KV plus two compression alternatives on matched evidence.
    pareto = payload.get("pareto_dominance")
    if _common(pareto, "pareto_dominance", model_id=model_id, run_id=run_id, failures=failures, official=False):
        baselines = pareto.get("baselines")
        _require(isinstance(baselines, list), "pareto_dominance.baselines is missing", failures)
        if isinstance(baselines, list):
            names = {item.get("name") for item in baselines if isinstance(item, dict)}
            _require("fp8_full_kv" in names, "pareto dominance must include fp8_full_kv", failures)
            _require(len(baselines) >= 3, "pareto dominance requires FP8 plus two compression baselines", failures)
            for index, item in enumerate(baselines):
                _require(isinstance(item, dict) and item.get("qcc_dominates") is True, f"pareto baseline {index} is not dominated", failures)
                if isinstance(item, dict):
                    for field in ("baseline_quality_score", "baseline_state_bytes", "baseline_p95_tpot_ms", "baseline_throughput_tokens_per_s", "qcc_quality_score", "qcc_state_bytes", "qcc_p95_tpot_ms", "qcc_throughput_tokens_per_s"):
                        _number(item.get(field), f"pareto_dominance.baselines[{index}].{field}", failures)
                    baseline_quality = item.get("baseline_quality_score")
                    qcc_quality = item.get("qcc_quality_score")
                    baseline_state = item.get("baseline_state_bytes")
                    qcc_state = item.get("qcc_state_bytes")
                    baseline_tail = item.get("baseline_p95_tpot_ms")
                    qcc_tail = item.get("qcc_p95_tpot_ms")
                    baseline_throughput = item.get("baseline_throughput_tokens_per_s")
                    qcc_throughput = item.get("qcc_throughput_tokens_per_s")
                    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (baseline_quality, qcc_quality, baseline_state, qcc_state, baseline_tail, qcc_tail, baseline_throughput, qcc_throughput)):
                        _require(qcc_quality >= baseline_quality, f"pareto baseline {index} quality is not dominated", failures)
                        _require(qcc_state <= baseline_state, f"pareto baseline {index} state is not dominated", failures)
                        _require(qcc_tail <= baseline_tail, f"pareto baseline {index} p95 latency is not dominated", failures)
                        _require(qcc_throughput >= baseline_throughput, f"pareto baseline {index} throughput is not dominated", failures)
                        _require(any((qcc_quality > baseline_quality, qcc_state < baseline_state, qcc_tail < baseline_tail, qcc_throughput > baseline_throughput)), f"pareto baseline {index} has no strict improvement", failures)

    # 8: tail production latency must improve without hiding a throughput tradeoff.
    production = payload.get("production_latency")
    if _common(production, "production_latency", model_id=model_id, run_id=run_id, failures=failures, official=False):
        ttft = _number(production.get("ttft_regression"), "production_latency.ttft_regression", failures)
        p95 = _number(production.get("p95_tpot_speedup"), "production_latency.p95_tpot_speedup", failures)
        p99 = _number(production.get("p99_tpot_speedup"), "production_latency.p99_tpot_speedup", failures)
        if ttft is not None:
            _require(ttft <= 0, "TTFT regression must be <= 0", failures)
        if p95 is not None:
            _require(p95 >= 1.0, "p95 TPOT must not regress", failures)
        if p99 is not None:
            _require(p99 >= 1.0, "p99 TPOT must not regress", failures)
        for field in ("p95_ttft_speedup", "p99_ttft_speedup"):
            value = _number(production.get(field), f"production_latency.{field}", failures)
            if value is not None:
                _require(value >= 1.0, f"{field} must not regress", failures)
        _require(production.get("throughput_latency_tradeoff") is False, "throughput/latency tradeoff evidence is not acceptable", failures)

    # 9: constant-state/near-constant-TPOT scaling through 1M.
    scaling = payload.get("scaling_law")
    if _common(scaling, "scaling_law", model_id=model_id, run_id=run_id, failures=failures, official=False):
        points = scaling.get("points")
        _require(isinstance(points, list), "scaling_law.points is missing", failures)
        required = {128_000, 256_000, 512_000, 1_000_000}
        lengths: set[int] = set()
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
                if length is not None:
                    lengths.add(int(length))
                if state is not None:
                    states.append(state)
                if tpot is not None:
                    tpots.append(tpot)
                _require(point.get("matched_full_kv") is True, f"scaling_law.points[{index}] is not matched Full-KV", failures)
                _require(point.get("measured") is True, f"scaling_law.points[{index}] is not measured evidence", failures)
        _require(required.issubset(lengths), "scaling law requires 128K/256K/512K/1M points", failures)
        for values, label in ((states, "state"), (tpots, "TPOT")):
            if values:
                variation = max(values) / max(min(values), 1e-12)
                summary[f"scaling_{label.lower()}_variation"] = variation
                _require(variation <= SCALING_VARIATION, f"scaling {label} variation {variation:.6f} > {SCALING_VARIATION:.2f}", failures)

    # 10: independent generalization, not just reruns on one stack.
    general = payload.get("generalization")
    if _common(general, "generalization", model_id=model_id, run_id=run_id, failures=failures, official=False, matched=False):
        for field in ("model_families", "gpu_generations", "independent_reproductions"):
            value = _number(general.get(field), f"generalization.{field}", failures)
            if value is not None:
                _require(value >= MIN_REPRODUCTIONS, f"generalization.{field} must be >= {MIN_REPRODUCTIONS}", failures)
        model_ids = general.get("model_ids")
        gpu_ids = general.get("gpu_generation_ids")
        reproduction_ids = general.get("reproduction_run_ids")
        for value, label in ((model_ids, "model_ids"), (gpu_ids, "gpu_generation_ids"), (reproduction_ids, "reproduction_run_ids")):
            _require(isinstance(value, list) and len(set(value)) >= MIN_REPRODUCTIONS, f"generalization.{label} must list >= {MIN_REPRODUCTIONS} distinct entries", failures)

    return {"passed": not failures, "failures": failures, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.evidence.read_text()))
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
