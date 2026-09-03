from benchmarks.compare_generalization_reports import summarize_reports


def _report(model_id: str, family: str, gpu: str, reproduction: str) -> dict:
    return {
        "model_id": model_id,
        "run_id": "run-1",
        "real_model": True,
        "synthetic": False,
        "matched_full_kv": True,
        "qcc_only": False,
        "protocol_locked": True,
        "native_context_tokens": 131_072,
        "model_family": family,
        "gpu_generation": gpu,
        "reproduction_id": reproduction,
        "source": f"https://github.com/example/{reproduction}",
    }


def test_generalization_counts_distinct_families_gpus_and_reproductions():
    result = summarize_reports(
        [
            _report("org/qwen-1b", "qwen", "ampere", "a"),
            _report("org/phi-3b", "phi", "hopper", "b"),
            _report("org/qwen-1b", "qwen", "ampere", "c"),
        ],
        model_id="org/qwen-1b",
        run_id="run-1",
    )
    assert result["model_families"] == 2
    assert result["gpu_generations"] == 2
    assert result["independent_reproductions"] == 3
    assert result["schema"] == "qcc-generalization-v2"
    assert result["minimums_met"] == {
        "model_families": True,
        "gpu_generations": True,
        "independent_reproductions": True,
    }


def test_generalization_rejects_unpaired_or_short_context_reports():
    report = _report("org/qwen-1b", "qwen", "ampere", "a")
    report["matched_full_kv"] = False
    try:
        summarize_reports([report], model_id="org/qwen-1b", run_id="run-1")
    except ValueError as exc:
        assert "matched Full-KV/QCC" in str(exc)
    else:
        raise AssertionError("unpaired evidence must be rejected")

    report = _report("org/qwen-1b", "qwen", "ampere", "b")
    report["native_context_tokens"] = 64_000
    try:
        summarize_reports([report], model_id="org/qwen-1b", run_id="run-1")
    except ValueError as exc:
        assert "native_context_tokens" in str(exc)
    else:
        raise AssertionError("short-context evidence must be rejected")
