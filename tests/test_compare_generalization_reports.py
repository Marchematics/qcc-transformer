from benchmarks.compare_generalization_reports import summarize_reports


def _report(model_id: str, family: str, gpu: str, reproduction: str) -> dict:
    return {
        "model_id": model_id,
        "run_id": "run-1",
        "real_model": True,
        "synthetic": False,
        "model_family": family,
        "gpu_generation": gpu,
        "reproduction_id": reproduction,
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
