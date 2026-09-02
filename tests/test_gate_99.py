from benchmarks.gate_99 import audit


def _evidence() -> dict:
    run_id = "qwen-1.5b-128k-20260902"
    quality = {
        name: {
            "model_id": "org/qwen-1.5b",
            "qcc_score": 98.5,
            "full_kv_score": 99.0,
            "matched_full_kv": True,
            "real_model": True,
            "official": True,
            "synthetic": False,
            "qcc_only": False,
            "run_id": run_id,
        }
        for name in ("ruler", "longbench", "pg19")
    }
    return {
        "run_id": run_id,
        "model": {
            "model_id": "org/qwen-1.5b",
            "pretrained": True,
            "real_checkpoint": True,
            "parameter_count": 1_500_000_000,
        },
        "quality": quality,
        "vllm_latency": {
            "model_id": "org/qwen-1.5b",
            "context_tokens": 128_000,
            "tpot_speedup": 5.1,
            "throughput_speedup": 2.1,
            "matched_full_kv": True,
            "real_model": True,
            "official": True,
            "synthetic": False,
            "qcc_only": False,
            "vllm_version": "0.8.5",
            "run_id": run_id,
        },
        "memory": {
            "model_id": "org/qwen-1.5b",
            "full_kv_peak_bytes": 1000,
            "qcc_peak_bytes": 180,
            "full_kv_concurrency": 1,
            "qcc_concurrency": 4,
            "matched_full_kv": True,
            "real_model": True,
            "official": True,
            "synthetic": False,
            "qcc_only": False,
            "run_id": run_id,
        },
        "calibration": {
            "model_id": "org/qwen-1.5b",
            "trainable_parameter_fraction": 0.01,
            "hf_zero_code_changes": True,
            "vllm_zero_code_changes": True,
            "run_id": run_id,
        },
    }


def test_gate_passes_only_when_all_sections_pass() -> None:
    result = audit(_evidence())
    assert result["passed"] is True
    assert result["failures"] == []


def test_gate_fails_closed_for_synthetic_or_unmatched_evidence() -> None:
    evidence = _evidence()
    evidence["quality"]["ruler"]["synthetic"] = True
    evidence["vllm_latency"]["matched_full_kv"] = False
    evidence["calibration"]["trainable_parameter_fraction"] = 0.011
    result = audit(evidence)
    assert result["passed"] is False
    assert any("quality.ruler is synthetic" in item for item in result["failures"])
    assert any("vllm_latency is not matched Full-KV" in item for item in result["failures"])
    assert any("trainable parameter fraction" in item for item in result["failures"])


def test_gate_rejects_wrong_parameter_scale() -> None:
    evidence = _evidence()
    evidence["model"]["parameter_count"] = 900_000_000
    result = audit(evidence)
    assert result["passed"] is False
    assert "parameter_count must be within 1B..7B" in result["failures"]
