from benchmarks.compare_memory_reports import compare_reports
import pytest


def test_memory_comparison_exports_fixed_sla_metrics():
    memory = {
        "model_id": "org/model-1b",
        "run_id": "run-1",
        "real_model": True,
        "synthetic": False,
        "qcc_only": False,
        "matched": True,
        "protocol_locked": True,
        "total_tokens": 128_000,
        "baseline_full_kv": {"status": "ok", "attention_state_bytes": 1000},
        "qcc_retrofit": {"status": "ok", "attention_state_bytes": 180},
    }
    concurrency = {
        "model_id": "org/model-1b",
        "run_id": "run-1",
        "real_model": True,
        "synthetic": False,
        "qcc_only": False,
        "protocol_locked": True,
        "total_tokens_per_request": 128_000,
        "fixed_sla": True,
        "sla_seconds": 20.0,
        "max_full_kv_batch": 1,
        "max_qcc_batch": 4,
    }
    result = compare_reports(memory, concurrency)
    assert result["attention_state_reduction"] == pytest.approx(0.82)
    assert result["concurrency_speedup"] == 4.0
    assert result["fixed_sla"] is True
    assert result["matched_full_kv"] is True


def test_memory_comparison_rejects_unmatched_oom_or_missing_sla():
    memory = {
        "model_id": "m",
        "run_id": "r",
        "real_model": True,
        "synthetic": False,
        "qcc_only": False,
        "matched": True,
        "protocol_locked": True,
        "total_tokens": 128_000,
        "baseline_full_kv": {"status": "oom"},
        "qcc_retrofit": {"status": "ok", "attention_state_bytes": 1},
    }
    concurrency = {
        "model_id": "m",
        "run_id": "r",
        "real_model": True,
        "synthetic": False,
        "qcc_only": False,
        "protocol_locked": True,
        "total_tokens_per_request": 128_000,
        "fixed_sla": False,
        "sla_seconds": None,
        "max_full_kv_batch": 1,
        "max_qcc_batch": 4,
    }
    try:
        compare_reports(memory, concurrency)
    except ValueError as exc:
        assert "fixed SLA" in str(exc)
    else:
        raise AssertionError("missing fixed SLA must be rejected")
