import json

import pytest

from benchmarks.assemble_quality_evidence import assemble
from benchmarks.assemble_generalization import assemble as assemble_generalization
from benchmarks.compare_serving_reports import compare
from benchmarks.compare_pareto import compare as compare_pareto
from benchmarks.compare_retrieval_1m import compare as compare_retrieval


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _common(run_id="run-1", model_id="org/model"):
    return {
        "run_id": run_id,
        "model_id": model_id,
        "real_model": True,
        "synthetic": False,
        "official": True,
    }


def test_quality_assembly_requires_paired_full_suites(tmp_path):
    run_id = "run-1"
    model_id = "org/model"
    ruler = {
        **_common(run_id, model_id),
        "benchmark": "ruler",
        "official_evaluator": "test-ruler-scorer",
        "full_suite": True,
        "baseline_full_kv": {
            "accuracy": 1.0,
            "records": [{"line": 1, "correct": True}],
        },
        "qcc_retrofit": {
            "accuracy": 0.99,
            "records": [{"line": 1, "correct": True}],
        },
    }
    datasets = ["task_a", "task_b"]
    longbench_full = {
        **_common(run_id, model_id),
        "benchmark": "longbench", "mode": "fullkv", "full_suite": True,
        "datasets": datasets, "dataset_scores": {name: 0.5 for name in datasets},
        "generated_rows": 2, "quality_score": 0.5, "official_evaluator": "test-longbench-scorer",
    }
    longbench_qcc = {
        **longbench_full, "mode": "qcc",
        "dataset_scores": {name: 0.49 for name in datasets}, "quality_score": 0.49,
    }
    pg19_full = {
        **_common(run_id, model_id),
        "benchmark": "pg19", "mode": "fullkv", "split": "test",
        "documents": 2, "predicted_tokens": 20, "quality_score": 0.2, "official_source": "test-pg19", "full_suite": True,
    }
    pg19_qcc = {**pg19_full, "mode": "qcc", "quality_score": 0.19}
    paths = {
        "ruler": _write(tmp_path / "ruler.json", ruler),
        "longbench_full": _write(tmp_path / "longbench_full.json", longbench_full),
        "longbench_qcc": _write(tmp_path / "longbench_qcc.json", longbench_qcc),
        "pg19_full": _write(tmp_path / "pg19_full.json", pg19_full),
        "pg19_qcc": _write(tmp_path / "pg19_qcc.json", pg19_qcc),
    }
    result = assemble(run_id=run_id, model_id=model_id, **paths)
    assert result["ruler"]["qcc_score"] == 0.99
    assert result["longbench"]["full_kv_score"] == 0.5
    assert result["pg19"]["qcc_score"] == 0.19

    longbench_qcc["datasets"] = ["task_a"]
    _write(paths["longbench_qcc"], longbench_qcc)
    with pytest.raises(ValueError, match="dataset lists"):
        assemble(run_id=run_id, model_id=model_id, **paths)


def _serving(label, *, run_id="run-1", model_id="org/model", throughput=100.0, concurrency=8):
    return {
        "schema": "qcc-vllm-serving-v1",
        "run_id": run_id,
        "label": label,
        "model_id": model_id,
        "real_model": True,
        "synthetic": False,
        "stock_vllm": True,
        "streaming": True,
        "failed_requests": 0,
        "successful_requests": 8,
        "num_requests": 8,
        "context_length": 128_000,
        "workload_sha256": "a" * 64,
        "vllm_version": "0.10.0",
        "gpu": "A100-80GB",
        "num_requests": 8,
        "max_tokens": 32,
        "concurrency": concurrency,
        "sla": {
            "ttft_p95_limit_ms": 2000.0,
            "tpot_p95_limit_ms": 20000.0,
            "ttft_p95_ms": 1200.0,
            "tpot_p95_ms": 12.0 if label == "fullkv" else 3.0,
        },
        "sla_pass": True,
        "throughput_tokens_per_s": throughput,
        "server_peak_gpu_memory_bytes": 10_000,
        "ttft_s": {"p50": 1.0, "p95": 1.2, "p99": 1.3},
        "tpot_s": {"p50": 10.0 if label == "fullkv" else 2.0,
                   "p95": 12.0 if label == "fullkv" else 3.0,
                   "p99": 14.0 if label == "fullkv" else 4.0},
    }


def test_serving_comparison_uses_tail_percentiles_and_peak_memory():
    result = compare(
        _serving("fullkv", concurrency=1), _serving("qcc", throughput=220.0, concurrency=4),
        run_id="run-1", model_id="org/model",
        full_attention_state_bytes=1000, qcc_attention_state_bytes=100,
    )
    assert result["vllm_latency"]["tpot_speedup"] == 4.0
    assert result["vllm_latency"]["throughput_speedup"] == 2.2
    assert result["memory"]["attention_state_reduction"] == 0.9
    assert result["production_latency"]["p95_tpot_speedup"] == 4.0
    assert result["production_latency"]["p99_tpot_speedup"] == 3.5
    assert result["memory"]["qcc_concurrency"] == 4


def test_serving_comparison_rejects_failed_or_mismatched_requests():
    full = _serving("fullkv")
    qcc = _serving("qcc")
    qcc["failed_requests"] = 1
    with pytest.raises(ValueError, match="failed requests"):
        compare(
            full, qcc, run_id="run-1", model_id="org/model",
            full_attention_state_bytes=1000, qcc_attention_state_bytes=100,
        )


def test_serving_comparison_rejects_unlocked_or_false_sla():
    full = _serving("fullkv")
    qcc = _serving("qcc")
    qcc["sla_pass"] = False
    with pytest.raises(ValueError, match="fixed-SLA"):
        compare(
            full, qcc, run_id="run-1", model_id="org/model",
            full_attention_state_bytes=1000, qcc_attention_state_bytes=100,
        )


def _pareto_metrics(name, *, quality, state, latency, throughput):
    return {
        "run_id": "run-1", "model_id": "org/model", "workload_sha256": "b" * 64,
        "context_tokens": 128_000, "gpu": "A100-80GB",
        "real_model": True, "synthetic": False, "protocol_locked": True,
        "label": name, "quality_score": quality, "attention_state_bytes": state,
        "p95_tpot_ms": latency, "throughput_tokens_per_s": throughput,
    }


def test_pareto_comparison_recomputes_dominance():
    qcc = _pareto_metrics("qcc", quality=0.99, state=100, latency=2, throughput=200)
    baselines = {
        "fp8_full_kv": _pareto_metrics("fp8", quality=0.98, state=1000, latency=8, throughput=100),
        "compresskv": _pareto_metrics("compress", quality=0.97, state=500, latency=5, throughput=120),
        "h2o": _pareto_metrics("h2o", quality=0.96, state=400, latency=4, throughput=150),
    }
    result = compare_pareto(qcc, baselines, run_id="run-1", model_id="org/model")
    assert result["all_baselines_dominated"] is True
    assert all(item["qcc_dominates"] for item in result["baselines"])
    baselines["h2o"]["quality_score"] = 1.0
    result = compare_pareto(qcc, baselines, run_id="run-1", model_id="org/model")
    assert result["all_baselines_dominated"] is False


def test_generalization_requires_distinct_model_and_gpu_families():
    reports = [
        {"model_id": "org/model", "model_family": "llama", "gpu_generation": "ampere", "reproduction_run_id": "rep-a", "real_model": True, "synthetic": False, "protocol_locked": True, "matched_full_kv": True, "passed": True},
        {"model_id": "org/other", "model_family": "phi", "gpu_generation": "hopper", "reproduction_run_id": "rep-b", "real_model": True, "synthetic": False, "protocol_locked": True, "matched_full_kv": True, "passed": True},
    ]
    result = assemble_generalization(reports, run_id="run-1", model_id="org/model")
    assert result["model_families"] == 2
    assert result["gpu_generations"] == 2
    assert result["independent_reproductions"] == 2


def test_retrieval_comparison_recomputes_rates_and_all_depth_tails(tmp_path):
    rows_full = []
    rows_qcc = []
    bucket_names = ("0-25%", "25-50%", "50-75%", "75-100%")
    for trial in range(1000):
        bucket = bucket_names[trial % len(bucket_names)]
        common = {
            "trial": trial,
            "expected": f"{trial:08d}",
            "target_entity": f"entity-{trial}",
            "depth_bucket": bucket,
        }
        rows_full.append({**common, "correct": True})
        rows_qcc.append({**common, "correct": trial != 0})
    full_rows = tmp_path / "full.jsonl"
    qcc_rows = tmp_path / "qcc.jsonl"
    full_rows.write_text("\n".join(json.dumps(row) for row in rows_full) + "\n")
    qcc_rows.write_text("\n".join(json.dumps(row) for row in rows_qcc) + "\n")

    def summary(mode, output):
        return {
            "schema": "qcc-real-retrieval-result-v1",
            "mode": mode,
            "run_id": "run-1",
            "model_id": "org/model",
            "real_model": True,
            "synthetic": False,
            "pretrained": True,
            "real_checkpoint": True,
            "protocol_locked": True,
            "manifest_sha256": "c" * 64,
            "context_tokens": 1_000_000,
            "native_context_tokens": 1_000_000,
            "trials": 1000,
            "random_depth": True,
            "multi_needle": True,
            "semantic_distractor": True,
            "oracle_admission": False if mode == "qcc" else None,
            "output_jsonl": str(output),
        }

    result = compare_retrieval(summary("fullkv", full_rows), summary("qcc", qcc_rows))
    assert result["retrieval_1m"]["qcc_success_rate"] == 0.999
    assert result["retrieval_1m"]["full_kv_success_rate"] == 1.0
    assert result["tail_safety"]["catastrophic_retrieval_miss_rate"] == 0.001
    assert {row["bucket"] for row in result["tail_safety"]["critical_buckets"]} == set(bucket_names)
