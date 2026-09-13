import json
import pytest
import torch

from benchmarks.benchmark_hf_ruler import (
    _append_progress,
    _chunked_causal_attention,
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


def test_chunked_causal_attention_matches_sdpa():
    torch.manual_seed(7)
    query = torch.randn(2, 3, 11, 4)
    key = torch.randn(2, 3, 11, 4)
    value = torch.randn_like(key)
    actual = _chunked_causal_attention(
        query,
        key,
        value,
        query_chunk_size=3,
        key_chunk_size=4,
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, is_causal=True
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_phi3_record_template_applies_answer_prefix_once():
    from benchmarks.benchmark_hf_ruler import _format_phi3_record
    record = {"input": "Find the number.", "outputs": ["123"],
              "answer_prefix": " The number is"}
    formatted = _format_phi3_record(record)
    assert formatted["input"] == (
        "<|user|>\nFind the number.<|end|>\n<|assistant|>\n The number is")
    assert _format_phi3_record(formatted) == formatted
    assert record["input"] == "Find the number."
    assert formatted["outputs"] == record["outputs"]
