from benchmarks.compare_pareto_reports import compare_pareto


def _report(label: str, *, tpot: float, throughput: float) -> dict:
    return {
        "schema": "qcc-vllm-serving-v1",
        "label": label,
        "model_id": "org/model-1b",
        "run_id": "run-1",
        "context_length": 128_000,
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
        "protocol_locked": True,
        "throughput_tokens_per_s": throughput,
        "request_throughput_per_s": throughput / 32.0,
        "ttft_s": {"p50": 1.0, "p95": 1.1, "p99": 1.2},
        "tpot_s": {"p50": tpot, "p95": tpot * 1.1, "p99": tpot * 1.2},
    }


def test_pareto_requires_and_joins_fp8_and_compression_baselines():
    qcc = _report("qcc", tpot=2.0, throughput=220.0)
    result = compare_pareto(
        qcc,
        [
            _report("fp8-fullkv", tpot=10.0, throughput=100.0),
            _report("snapkv", tpot=8.0, throughput=90.0),
            _report("h2o", tpot=6.0, throughput=80.0),
        ],
    )
    assert result["all_dominated"] is True
    assert result["baselines"][0]["name"] == "fp8_full_kv"
    assert len(result["baselines"]) == 3
