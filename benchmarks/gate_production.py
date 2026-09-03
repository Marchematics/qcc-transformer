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
MIN_QUALITY_CONTEXT = 128_000
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

    # 1/2: one real pretrained model, official long-context quality >=98% Full-KV.
    quality = payload.get("quality")
    _require(isinstance(quality, dict), "quality section is missing", failures)
    quality_ratios: dict[str, float] = {}
    quality_contexts: set[int] = set()
    if isinstance(quality, dict):
        for benchmark in ("ruler", "longbench", "pg19"):
            section = quality.get(benchmark)
            if _common(section, f"quality.{benchmark}", model_id=model_id, run_id=run_id, failures=failures, official=True):
                _require(
                    section.get("benchmark") == benchmark,
                    f"quality.{benchmark}.benchmark is missing or mismatched",
                    failures,
                )
                _require(
                    section.get("full_suite") is True,
                    f"quality.{benchmark} is not a complete official suite",
                    failures,
                )
                context = _number(section.get("native_context_tokens"), f"quality.{benchmark}.native_context_tokens", failures)
                if context is not None:
                    _require(context >= MIN_QUALITY_CONTEXT, f"quality.{benchmark} native context is below 128K", failures)
                    quality_contexts.add(int(context))
                qcc = _number(section.get("qcc_score"), f"quality.{benchmark}.qcc_score", failures)
                full = _number(section.get("full_kv_score"), f"quality.{benchmark}.full_kv_score", failures)
                if qcc is not None and full is not None:
                    _require(full > 0, f"quality.{benchmark}.full_kv_score must be > 0", failures)
                    if full > 0:
                        ratio = qcc / full
                        quality_ratios[benchmark] = ratio
                        _require(ratio >= QUALITY_RATIO, f"quality.{benchmark} ratio {ratio:.6f} < {QUALITY_RATIO:.2f}", failures)
                task_ratios = section.get("task_ratios")
                _require(
                    isinstance(task_ratios, dict) and bool(task_ratios),
                    f"quality.{benchmark}.task_ratios is missing",
                    failures,
                )
                if isinstance(task_ratios, dict):
                    for task, value in task_ratios.items():
                        task_ratio = _number(
                            value,
                            f"quality.{benchmark}.task_ratios.{task}",
                            failures,
                        )
                        if task_ratio is not None:
                            _require(
                                task_ratio >= TAIL_RATIO,
                                f"quality.{benchmark} task {task} ratio {task_ratio:.6f} < {TAIL_RATIO:.2f}",
                                failures,
                            )
        _require(
            len(quality_contexts) == 1,
            "quality reports must share one native context",
            failures,
        )
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
            native = _number(
                latency.get("native_context_tokens"),
                "vllm_latency.native_context_tokens",
                failures,
            )
            if native is not None:
                _require(
                    native >= context,
                    "vllm native context is below the requested workload",
                    failures,
                )
            _require(
                latency.get("workload_context_exact") is True,
                "vllm 128K workload token count is not exact",
                failures,
            )
        if tpot is not None:
            _require(tpot >= TPOT_SPEEDUP, f"vLLM TPOT speedup {tpot:.6f} < {TPOT_SPEEDUP:.1f}", failures)
        if throughput is not None:
            _require(throughput >= THROUGHPUT_SPEEDUP, f"vLLM throughput speedup {throughput:.6f} < {THROUGHPUT_SPEEDUP:.1f}", failures)
        _require(isinstance(latency.get("vllm_version"), str) and bool(latency.get("vllm_version")), "vllm_version is required", failures)

    # 4: actual attention-state reduction and fixed-SLA concurrency.
    memory = payload.get("memory")
    if _common(memory, "memory", model_id=model_id, run_id=run_id, failures=failures, official=False):
        context = _number(memory.get("context_tokens"), "memory.context_tokens", failures)
        native = _number(
            memory.get("native_context_tokens"),
            "memory.native_context_tokens",
            failures,
        )
        if context is not None:
            _require(context >= 128_000, "memory context_tokens is below 128K", failures)
        if context is not None and native is not None:
            _require(
                native >= context,
                "memory native context is below the requested workload",
                failures,
            )
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
            native = _number(
                retrieval.get("native_context_tokens"),
                "retrieval_1m.native_context_tokens",
                failures,
            )
            if native is not None:
                _require(
                    native >= context,
                    "retrieval_1m native context is below the requested workload",
                    failures,
                )
        if qcc is not None:
            _require(qcc >= RETRIEVAL_RATE, f"1M retrieval rate {qcc:.6f} < {RETRIEVAL_RATE:.2f}", failures)
        if full is not None:
            _require(full >= RETRIEVAL_RATE, f"Full-KV 1M retrieval rate {full:.6f} < {RETRIEVAL_RATE:.2f}", failures)
        if qcc is not None and full is not None and full > 0:
            ratio = qcc / full
            summary["retrieval_ratio"] = ratio
            _require(ratio >= RETRIEVAL_RATE, f"1M retrieval ratio {ratio:.6f} < {RETRIEVAL_RATE:.2f}", failures)
        _require(
            retrieval.get("all_needles_required") is True,
            "retrieval_1m must score every needle in each trial",
            failures,
        )
        needle_qcc = _number(
            retrieval.get("needle_success_rate"),
            "retrieval_1m.needle_success_rate",
            failures,
        )
        needle_full = _number(
            retrieval.get("full_kv_needle_success_rate"),
            "retrieval_1m.full_kv_needle_success_rate",
            failures,
        )
        if needle_qcc is not None:
            _require(needle_qcc >= RETRIEVAL_RATE, f"1M needle retrieval rate {needle_qcc:.6f} < {RETRIEVAL_RATE:.2f}", failures)
        if needle_full is not None:
            _require(needle_full >= RETRIEVAL_RATE, f"Full-KV needle retrieval rate {needle_full:.6f} < {RETRIEVAL_RATE:.2f}", failures)
        if needle_qcc is not None and needle_full is not None and needle_full > 0:
            needle_ratio = needle_qcc / needle_full
            summary["needle_retrieval_ratio"] = needle_ratio
            _require(needle_ratio >= RETRIEVAL_RATE, f"1M needle retrieval ratio {needle_ratio:.6f} < {RETRIEVAL_RATE:.2f}", failures)
        for field in ("random_depth", "multi_needle", "semantic_distractor"):
            _require(retrieval.get(field) is True, f"retrieval_1m.{field} evidence is missing", failures)
        _require(retrieval.get("oracle_admission") is False, "retrieval_1m uses oracle admission", failures)

    # 6: no average-score masking of catastrophic retrieval failures.
    tail = payload.get("tail_safety")
    if _common(tail, "tail_safety", model_id=model_id, run_id=run_id, failures=failures, official=False):
        miss = _number(tail.get("catastrophic_retrieval_miss_rate"), "tail_safety.catastrophic_retrieval_miss_rate", failures)
        if miss is not None:
            _require(miss < 0.01, "catastrophic retrieval miss rate must be < 1%", failures)
        miss_trials = _number(
            tail.get("catastrophic_retrieval_miss_rate_trials"),
            "tail_safety.catastrophic_retrieval_miss_rate_trials",
            failures,
        )
        if miss_trials is not None:
            _require(
                miss_trials < 0.01,
                "catastrophic retrieval miss rate over all trials must be < 1%",
                failures,
            )
        needle_miss = _number(
            tail.get("catastrophic_retrieval_needle_miss_rate"),
            "tail_safety.catastrophic_retrieval_needle_miss_rate",
            failures,
        )
        if needle_miss is not None:
            _require(needle_miss < 0.01, "catastrophic needle retrieval miss rate must be < 1%", failures)
        buckets = tail.get("critical_buckets")
        _require(isinstance(buckets, list) and bool(buckets), "tail_safety.critical_buckets is missing", failures)
        if isinstance(buckets, list):
            for index, bucket in enumerate(buckets):
                ratio = _number(bucket.get("qcc_full_kv_ratio") if isinstance(bucket, dict) else None, f"tail_safety.critical_buckets[{index}].qcc_full_kv_ratio", failures)
                if ratio is not None:
                    _require(ratio >= TAIL_RATIO, f"tail bucket {index} ratio {ratio:.6f} < {TAIL_RATIO:.2f}", failures)

    # 7: dominate FP8 Full-KV plus two compression alternatives on matched evidence.
    pareto = payload.get("pareto_dominance")
    if _common(pareto, "pareto_dominance", model_id=model_id, run_id=run_id, failures=failures, official=False):
        baselines = pareto.get("baselines")
        _require(isinstance(baselines, list), "pareto_dominance.baselines is missing", failures)
        _require(
            pareto.get("all_dominated") is True,
            "pareto dominance is not complete across all baselines",
            failures,
        )
        if isinstance(baselines, list):
            names = {item.get("name") for item in baselines if isinstance(item, dict)}
            _require("fp8_full_kv" in names, "pareto dominance must include fp8_full_kv", failures)
            _require(len(baselines) >= 3, "pareto dominance requires FP8 plus two compression baselines", failures)
            for index, item in enumerate(baselines):
                _require(isinstance(item, dict) and item.get("qcc_dominates") is True, f"pareto baseline {index} is not dominated", failures)
                _require(isinstance(item, dict) and item.get("memory_dominates") is True, f"pareto baseline {index} lacks memory dominance", failures)

    # 8: tail production latency must improve without hiding a throughput tradeoff.
    production = payload.get("production_latency")
    if _common(production, "production_latency", model_id=model_id, run_id=run_id, failures=failures, official=False):
        ttft = _number(production.get("ttft_regression"), "production_latency.ttft_regression", failures)
        p95 = _number(production.get("p95_tpot_speedup"), "production_latency.p95_tpot_speedup", failures)
        p99 = _number(production.get("p99_tpot_speedup"), "production_latency.p99_tpot_speedup", failures)
        ttft_p95 = _number(production.get("p95_ttft_speedup"), "production_latency.p95_ttft_speedup", failures)
        ttft_p99 = _number(production.get("p99_ttft_speedup"), "production_latency.p99_ttft_speedup", failures)
        if ttft is not None:
            _require(ttft <= 0, "TTFT regression must be <= 0", failures)
        if p95 is not None:
            _require(p95 >= 1.0, "p95 TPOT must not regress", failures)
        if p99 is not None:
            _require(p99 >= 1.0, "p99 TPOT must not regress", failures)
        if ttft_p95 is not None:
            _require(ttft_p95 >= 1.0, "p95 TTFT must not regress", failures)
        if ttft_p99 is not None:
            _require(ttft_p99 >= 1.0, "p99 TTFT must not regress", failures)
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
        evaluations = general.get("evaluations")
        _require(
            isinstance(evaluations, list) and bool(evaluations),
            "generalization.evaluations is missing",
            failures,
        )
        evaluation_families: set[str] = set()
        evaluation_gpus: set[str] = set()
        evaluation_reproductions: set[str] = set()
        if isinstance(evaluations, list):
            for index, evaluation in enumerate(evaluations):
                if not isinstance(evaluation, dict):
                    failures.append(f"generalization.evaluations[{index}] must be an object")
                    continue
                for field, values in (
                    ("model_family", evaluation_families),
                    ("gpu_generation", evaluation_gpus),
                    ("reproduction_id", evaluation_reproductions),
                ):
                    value = evaluation.get(field)
                    _require(
                        isinstance(value, str) and bool(value.strip()),
                        f"generalization.evaluations[{index}].{field} is missing",
                        failures,
                    )
                    if isinstance(value, str) and value.strip():
                        values.add(value)
                context = _number(
                    evaluation.get("native_context_tokens"),
                    f"generalization.evaluations[{index}].native_context_tokens",
                    failures,
                )
                if context is not None:
                    _require(
                        context >= MIN_QUALITY_CONTEXT,
                        f"generalization.evaluations[{index}] is below 128K",
                        failures,
                    )
                _require(
                    isinstance(evaluation.get("source"), str)
                    and bool(evaluation.get("source", "").strip()),
                    f"generalization.evaluations[{index}].source is missing",
                    failures,
                )
        if isinstance(evaluations, list):
            _require(
                len(evaluation_families) >= MIN_REPRODUCTIONS,
                "generalization evaluations do not cover two model families",
                failures,
            )
            _require(
                len(evaluation_gpus) >= MIN_REPRODUCTIONS,
                "generalization evaluations do not cover two GPU generations",
                failures,
            )
            _require(
                len(evaluation_reproductions) >= MIN_REPRODUCTIONS,
                "generalization evaluations do not contain two independent reproductions",
                failures,
            )
        for field in ("model_families", "gpu_generations", "independent_reproductions"):
            value = _number(general.get(field), f"generalization.{field}", failures)
            if value is not None:
                _require(value >= MIN_REPRODUCTIONS, f"generalization.{field} must be >= {MIN_REPRODUCTIONS}", failures)

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
