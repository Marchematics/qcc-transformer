import json

from benchmarks.benchmark_hf_ruler import _append_progress, _start_progress_log


def test_ruler_progress_log_preserves_each_completed_record(tmp_path):
    path = tmp_path / "run.progress.jsonl"
    _start_progress_log(path, run_id="test-run", config={"window_size": 64})
    _append_progress(path, {"event": "record", "mode": "full_kv", "line": 1})
    _append_progress(path, {"event": "record", "mode": "qcc", "line": 1})

    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events == [
        {
            "event": "run_start",
            "run_id": "test-run",
            "config": {"window_size": 64},
        },
        {"event": "record", "mode": "full_kv", "line": 1},
        {"event": "record", "mode": "qcc", "line": 1},
    ]
