import torch

from benchmarks.benchmark_hf_retrieval_1m import build_trial_ids
from benchmarks.make_real_retrieval_1m import make_trial


class FakeTokenizer:
    """Deterministic tokenizer for testing exact context construction without HF."""

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        del add_special_tokens, return_tensors
        ids = [ord(char) % 251 + 1 for char in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def test_locked_trial_is_deterministic_unique_and_has_required_stressors():
    left = make_trial(17, 20260903, needles=4, distractors=12)
    right = make_trial(17, 20260903, needles=4, distractors=12)
    assert left == right
    records = left["records"]
    assert sum(item["kind"] == "needle" for item in records) == 4
    assert sum(item["kind"] == "semantic_distractor" for item in records) == 12
    assert len({item["code"] for item in records}) == 16
    assert all(0.01 <= float(item["depth"]) <= 0.97 for item in records)
    targets = [
        item for item in records
        if item["kind"] == "needle" and item["entity"] == left["target_entity"]
    ]
    assert len(targets) == 1
    assert targets[0]["code"] == left["expected"]


def test_trial_seed_changes_every_trial_without_changing_protocol_shape():
    trials = [make_trial(index, 20260903, needles=4, distractors=12) for index in range(20)]
    assert len({trial["seed"] for trial in trials}) == 20
    assert len({trial["expected"] for trial in trials}) == 20
    assert all(len(trial["records"]) == 16 for trial in trials)


def test_real_retrieval_builder_hits_exact_token_budget_and_depth_order():
    trial = make_trial(3, 20260903, needles=4, distractors=12)
    ids, actual_depths = build_trial_ids(FakeTokenizer(), trial, context_tokens=100_000)
    assert ids.shape == (1, 100_000)
    assert len(actual_depths) == 16
    assert actual_depths == sorted(actual_depths)
    assert all(0.0 < depth < 1.0 for depth in actual_depths)
