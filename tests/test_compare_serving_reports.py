from benchmarks.compare_serving_reports import compare_reports


def _report(label: str, *, tpot: float, throughput: float, ttft: float) -> dict:
    return {
        "schema": "qcc-vllm-serving-v1",
        "label": label,
        "model": "org/model-1b",
        "model_id": "org/model-1b",
        "run_id": "run-1",
        "context_length": 128_000,
        "workload_context_exact": True,
        "native_context_tokens": 131_072,
        "workload": "workload.jsonl",
        "num_requests": 8,
        "successful_requests": 8,
        "failed_requests": 0,
        "concurrency": 4,
        "max_tokens": 32,
        "vllm_version": "0.10.0",
        "stock_vllm": True,
        "streaming": True,
        "real_model": True,
        "synthetic": False,
        "qcc_only": False,
        "protocol_locked": True,
        "gpu": "A100-80GB",
        "gpu_generation": "Ampere",
        "model_family": "llama",
        "throughput_tokens_per_s": throughput,
        "request_throughput_per_s": throughput / 32.0,
        "server_peak_gpu_memory_mib": 1000.0,
        "ttft_s": {"p50": ttft, "p95": ttft * 1.1, "p99": ttft * 1.2},
        "tpot_s": {"p50": tpot, "p95": tpot * 1.1, "p99": tpot * 1.2},
    }


def test_serving_comparison_keeps_tail_and_throughput_separate():
    baseline = _report("fp8_full_kv", tpot=10.0, throughput=100.0, ttft=1.0)
    qcc = _report("qcc", tpot=2.0, throughput=220.0, ttft=0.9)
    result = compare_reports(qcc, baseline)
    assert result["derived"]["p95_tpot_speedup"] == 5.0
    assert result["derived"]["p99_tpot_speedup"] == 5.0
    assert result["derived"]["throughput_speedup"] == 2.2
    assert result["derived"]["ttft_regression"] < 0.0
    assert result["derived"]["throughput_latency_tradeoff"] is False
    assert result["derived"]["qcc_dominates"] is True
    assert result["vllm_latency"]["qcc_only"] is False
    assert result["production_latency"]["real_model"] is True


def test_serving_comparison_marks_tail_regression_even_with_higher_throughput():
    baseline = _report("fp8_full_kv", tpot=10.0, throughput=100.0, ttft=1.0)
    qcc = _report("qcc", tpot=2.0, throughput=220.0, ttft=0.9)
    qcc["tpot_s"]["p99"] = 13.0
    result = compare_reports(qcc, baseline)
    assert result["derived"]["throughput_speedup"] > 2.0
    assert result["derived"]["p99_tpot_speedup"] < 1.0
    assert result["derived"]["throughput_latency_tradeoff"] is True
    assert result["derived"]["qcc_dominates"] is False
