import json

from benchmarks.benchmark_hf_ruler import (
    _append_progress,
    _start_progress_log,
    _supports_forward_argument,
)


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


def test_ruler_forward_argument_probe_handles_remote_model_signatures():
    class Model:
        def forward(self, input_ids, logits_to_keep=None):
            del input_ids, logits_to_keep

    class LegacyModel:
        def forward(self, input_ids):
            del input_ids

    assert _supports_forward_argument(Model(), "logits_to_keep")
    assert not _supports_forward_argument(LegacyModel(), "logits_to_keep")
