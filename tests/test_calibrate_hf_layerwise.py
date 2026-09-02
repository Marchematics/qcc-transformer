"""Unit tests for layer selection in the optional HF calibration CLI."""

import importlib.util
from pathlib import Path

import pytest


_PATH = Path(__file__).parents[1] / "benchmarks" / "calibrate_hf_layerwise.py"
_SPEC = importlib.util.spec_from_file_location("calibrate_hf_layerwise", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
parse_layer_spec = _MODULE.parse_layer_spec
_chunked_mse = _MODULE._chunked_mse
_mean_cosine_from_cpu = _MODULE._mean_cosine_from_cpu


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("all", [0, 1, 2, 3, 4, 5, 6, 7]),
        ("last-half", [4, 5, 6, 7]),
        ("first-half", [0, 1, 2, 3]),
        ("last-quarter", [6, 7]),
        ("2-4", [2, 3, 4]),
        ("4,2,4", [2, 4]),
    ],
)
def test_parse_layer_spec(spec, expected):
    assert parse_layer_spec(spec, 8) == expected


@pytest.mark.parametrize("spec", ["8", "-1", "4-2", "0-8", "0-1-2"])
def test_parse_layer_spec_rejects_invalid_indices(spec):
    with pytest.raises(ValueError):
        parse_layer_spec(spec, 8)


def test_chunked_distillation_helpers_match_reference():
    import torch

    student = torch.randn(2, 3, 17)
    teacher = torch.randn(2, 3, 17)
    expected_mse = torch.nn.functional.mse_loss(student, teacher)
    expected_cos = torch.nn.functional.cosine_similarity(
        student.reshape(-1, 17), teacher.reshape(-1, 17), dim=-1
    ).mean()
    assert torch.allclose(_chunked_mse(student, teacher, chunk_size=5), expected_mse)
    assert torch.allclose(_mean_cosine_from_cpu(student, teacher, chunk_size=5), expected_cos)
