import pytest

from benchmarks.compare_quality_reports import assemble_quality


def _common(benchmark: str, mode: str) -> dict:
    return {
        "benchmark": benchmark,
        "mode": mode,
        "model_id": "org/model-1b",
        "run_id": "run-1",
        "real_model": True,
        "synthetic": False,
        "official": True,
        "full_suite": True,
        "native_context_tokens": 131_072,
    }


def test_quality_join_preserves_task_ratios():
    ruler = {
        **_common("ruler", "paired"),
        "qcc_score": 0.99,
        "full_kv_score": 1.0,
        "task_ratios": {"task:niah": 0.99, "length:8-32K": 0.99},
    }
    longbench_full = {
        **_common("longbench", "fullkv"),
        "quality_score": 90.0,
        "dataset_scores": {"qa": 80.0, "summ": 100.0},
        "split": "test",
    }
    longbench_qcc = {
        **_common("longbench", "qcc"),
        "quality_score": 88.0,
        "dataset_scores": {"qa": 76.0, "summ": 100.0},
        "split": "test",
    }
    pg19_full = {
        **_common("pg19", "fullkv"),
        "quality_score": 0.2,
        "bucket_scores": {"8-32K": 0.2},
        "split": "test",
    }
    pg19_qcc = {
        **_common("pg19", "qcc"),
        "quality_score": 0.19,
        "bucket_scores": {"8-32K": 0.19},
        "split": "test",
    }
    result = assemble_quality(
        ruler, longbench_full, longbench_qcc, pg19_full, pg19_qcc
    )
    assert result["quality"]["ruler"]["qcc_full_kv_ratio"] == 0.99
    assert result["quality"]["longbench"]["task_ratios"]["qa"] == 0.95
    assert result["quality"]["pg19"]["qcc_full_kv_ratio"] == 0.95
    assert result["native_context_tokens"] == 131_072

    ruler["native_context_tokens"] = 64_000
    with pytest.raises(ValueError, match="native_context_tokens"):
        assemble_quality(ruler, longbench_full, longbench_qcc, pg19_full, pg19_qcc)
