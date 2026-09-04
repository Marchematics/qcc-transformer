"""Regression tests for the regular HF calibration CLI helpers."""

import importlib.util
from pathlib import Path

import pytest


_PATH = Path(__file__).parents[1] / "benchmarks" / "calibrate_hf_retrofit.py"
_SPEC = importlib.util.spec_from_file_location("calibrate_hf_retrofit", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_teacher_argmax_cross_entropy_is_distinct_from_mse():
    import torch

    student = torch.tensor([[[8.0, 0.0, 0.0]]])
    teacher = torch.tensor([[[0.0, 8.0, 0.0]]])
    mse = _MODULE._chunked_mse(student, teacher, chunk_size=2)
    ce = _MODULE._teacher_argmax_cross_entropy(student, teacher)
    assert torch.isfinite(ce)
    assert not torch.allclose(ce, mse)


def test_teacher_argmax_cross_entropy_rejects_shape_mismatch():
    import torch

    with pytest.raises(ValueError, match="shape mismatch"):
        _MODULE._teacher_argmax_cross_entropy(
            torch.zeros(1, 2, 3), torch.zeros(1, 3, 3)
        )
