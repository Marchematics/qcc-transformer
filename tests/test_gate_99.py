from benchmarks.gate_99 import audit


def _custom(run_id: str, model_id: str) -> dict:
    return {
        "model_id": model_id,
        "matched_full_kv": True,
        "real_model": True,
        "official": False,
        "protocol_locked": True,
        "synthetic": False,
        "qcc_only": False,
        "run_id": run_id,
    }


def _evidence() -> dict:
    run_id = "qwen-1.5b-production-20260903"
    model_id = "org/qwen-1.5b"
    quality = {
        name: {
            "model_id": model_id,
            "qcc_score": 98.5,
            "full_kv_score": 99.0,
            "matched_full_kv": True,
            "real_model": True,
            "official": True,
            "synthetic": False,
            "qcc_only": False,
            "run_id": run_id,
            "task_ratios": {"task:all": 0.99, "length:128K+": 0.99},
        }
        for name in ("ruler", "longbench", "pg19")
    }
    custom = _custom(run_id, model_id)
    return {
        "run_id": run_id,
        "model": {
            "model_id": model_id,
            "pretrained": True,
            "real_checkpoint": True,
            "parameter_count": 1_500_000_000,
        },
        "quality": quality,
        "vllm_latency": {
            **custom,
            "stock_vllm": True,
            "context_tokens": 128_000,
            "tpot_speedup": 5.1,
            "throughput_speedup": 2.1,
            "vllm_version": "0.29.0",
        },
        "memory": {
            **custom,
            "full_kv_attention_state_bytes": 1000,
            "qcc_attention_state_bytes": 180,
            "full_kv_concurrency": 1,
            "qcc_concurrency": 4,
            "fixed_sla": True,
        },
        "calibration": {
            "model_id": model_id,
            "trainable_parameter_fraction": 0.01,
            "hf_zero_code_changes": True,
            "vllm_zero_code_changes": True,
            "run_id": run_id,
        },
        "retrieval_1m": {
            **custom,
            "trials": 1000,
            "context_tokens": 1_000_000,
            "qcc_success_rate": 0.995,
            "full_kv_success_rate": 1.0,
            "all_needles_required": True,
            "needle_success_rate": 0.997,
            "full_kv_needle_success_rate": 1.0,
            "random_depth": True,
            "multi_needle": True,
            "semantic_distractor": True,
            "oracle_admission": False,
        },
        "tail_safety": {
            **custom,
            "catastrophic_retrieval_miss_rate": 0.005,
            "catastrophic_retrieval_miss_rate_trials": 0.004,
            "catastrophic_retrieval_needle_miss_rate": 0.005,
            "critical_buckets": [
                {"name": "0-25%", "qcc_full_kv_ratio": 0.99},
                {"name": "75-100%", "qcc_full_kv_ratio": 0.98},
            ],
        },
        "pareto_dominance": {
            **custom,
            "baselines": [
                {"name": "fp8_full_kv", "qcc_dominates": True, "memory_dominates": True},
                {"name": "compresskv", "qcc_dominates": True, "memory_dominates": True},
                {"name": "h2o", "qcc_dominates": True, "memory_dominates": True},
            ],
        },
        "production_latency": {
            **custom,
            "ttft_regression": 0.0,
            "p95_ttft_speedup": 1.01,
            "p99_ttft_speedup": 1.01,
            "p95_tpot_speedup": 1.1,
            "p99_tpot_speedup": 1.02,
            "throughput_latency_tradeoff": False,
        },
        "scaling_law": {
            **custom,
            "points": [
                {
                    "context_tokens": n,
                    "qcc_state_bytes": 1000,
                    "tpot_ms": 1.0,
                    "matched_full_kv": True,
                    "measured": True,
                }
                for n in (128_000, 256_000, 512_000, 1_000_000)
            ],
        },
        "generalization": {
            **custom,
            "model_families": 2,
            "gpu_generations": 2,
            "independent_reproductions": 2,
        },
    }


def test_gate_passes_only_when_all_sections_pass() -> None:
    result = audit(_evidence())
    assert result["passed"] is True
    assert result["failures"] == []


def test_gate_accepts_locked_custom_protocols_without_false_official_label() -> None:
    evidence = _evidence()
    for name in (
        "vllm_latency", "memory", "retrieval_1m", "tail_safety",
        "pareto_dominance", "production_latency", "scaling_law", "generalization",
    ):
        assert evidence[name]["official"] is False
    assert audit(evidence)["passed"] is True


def test_gate_fails_closed_for_synthetic_unmatched_or_overbudget_evidence() -> None:
    evidence = _evidence()
    evidence["quality"]["ruler"]["synthetic"] = True
    evidence["vllm_latency"]["matched_full_kv"] = False
    evidence["calibration"]["trainable_parameter_fraction"] = 0.011
    result = audit(evidence)
    assert result["passed"] is False
    assert any("quality.ruler is synthetic" in item for item in result["failures"])
    assert any("vllm_latency is not matched Full-KV" in item for item in result["failures"])
    assert any("trainable parameter fraction" in item for item in result["failures"])


def test_gate_rejects_unlocked_custom_protocol_and_oracle_retrieval() -> None:
    evidence = _evidence()
    evidence["retrieval_1m"]["protocol_locked"] = False
    evidence["retrieval_1m"]["oracle_admission"] = True
    result = audit(evidence)
    assert result["passed"] is False
    assert any("retrieval_1m custom protocol is not locked" in item for item in result["failures"])
    assert any("oracle admission" in item for item in result["failures"])


def test_gate_rejects_wrong_parameter_scale_and_non_native_1m_shape() -> None:
    evidence = _evidence()
    evidence["model"]["parameter_count"] = 900_000_000
    evidence["retrieval_1m"]["context_tokens"] = 999_999
    result = audit(evidence)
    assert result["passed"] is False
    assert "parameter_count must be within 1B..7B" in result["failures"]
    assert "retrieval_1m context is below 1M tokens" in result["failures"]
