"""Unit tests for layer selection in the optional HF calibration CLI."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


_PATH = Path(__file__).parents[1] / "benchmarks" / "calibrate_hf_layerwise.py"
_SPEC = importlib.util.spec_from_file_location("calibrate_hf_layerwise", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
parse_layer_spec = _MODULE.parse_layer_spec
_encode_positioned_chunks = _MODULE._encode_positioned_chunks
_chunked_mse = _MODULE._chunked_mse
_mean_cosine_from_cpu = _MODULE._mean_cosine_from_cpu
_chunked_kl_divergence = _MODULE._chunked_kl_divergence
_distillation_loss = _MODULE._distillation_loss
_teacher_top2_margin_loss = _MODULE._teacher_top2_margin_loss
quality_gate_passed = _MODULE.quality_gate_passed
_teacher_logits_and_inputs = _MODULE._teacher_logits_and_inputs
_hidden_distillation_loss = _MODULE._hidden_distillation_loss
_distillation_loss = _MODULE._distillation_loss
_distillation_loss_from_hidden = _MODULE._distillation_loss_from_hidden


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.k_proj = nn.Linear(16, 16, bias=False)
        self.v_proj = nn.Linear(16, 16, bias=False)
        self.o_proj = nn.Linear(16, 16, bias=False)
        self.num_heads = 4
        self.num_key_value_heads = 4


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            max_position_embeddings=64,
            rope_theta=None,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        self.attn = _Attention()


class _TeacherAttention(_Attention):
    def forward(self, hidden_states):
        return self.o_proj(self.v_proj(hidden_states))


class _TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _TeacherAttention()
        self.lm_head = nn.Linear(16, 32, bias=False)

    def forward(self, input_ids, **_kwargs):
        hidden = torch.nn.functional.one_hot(input_ids % 16, num_classes=16).float()
        hidden = self.attn(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


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


def test_positioned_chunks_preserve_absolute_offsets():
    import torch

    class Tokenizer:
        def __call__(self, text, *, return_tensors, add_special_tokens):
            del text, return_tensors, add_special_tokens
            ids = torch.arange(30, dtype=torch.long).view(1, -1)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    chunks = _encode_positioned_chunks(
        Tokenizer(), "ignored", max_tokens=8, num_chunks=3, window_size=2
    )
    assert [int(chunk["position_ids"][0, 0]) for chunk in chunks] == [0, 11, 22]
    assert [int(chunk["position_ids"][0, -1]) for chunk in chunks] == [7, 18, 29]
    assert all(chunk["input_ids"].shape == (1, 8) for chunk in chunks)


def test_key_sample_code_init_uses_teacher_key_geometry():
    import torch

    model = _Model()
    from qcc_transformer import patch_hf_model

    patch_hf_model(model, window_size=4, num_codes=4, use_triton=False)
    hidden = torch.randn(1, 8, 16)
    original = model.attn.qcc.archive.codes.detach().clone()
    _MODULE._initialize_codebooks_from_teacher(
        model, {0: hidden}, strategy="key-sample"
    )
    assert not torch.equal(model.attn.qcc.archive.codes, original)
    assert torch.isfinite(model.attn.qcc.archive.codes).all()


def test_teacher_capture_samples_every_batch_and_selected_attention():
    import torch

    model = _TeacherModel()
    batches = [
        {"input_ids": torch.arange(12).view(1, -1)},
        {"input_ids": torch.arange(12, 24).view(1, -1)},
    ]
    teachers, snapshots, attention = _teacher_logits_and_inputs(
        model, batches, max_capture_tokens=4, selected_layers={0}
    )
    assert len(teachers) == 2
    assert snapshots[0].shape == (1, 8, 16)
    assert len(attention[0]) == 2
    assert attention[0][0].shape == (4, 16)
    assert torch.isfinite(_hidden_distillation_loss(attention[0][0], attention[0][0]))


def test_hidden_logit_loss_matches_full_logit_loss():
    torch.manual_seed(7)
    hidden = torch.randn(1, 5, 6, requires_grad=True)
    head = nn.Linear(6, 11, bias=False)
    teacher = torch.randn(1, 5, 11)
    kwargs = dict(
        chunk_size=4,
        cosine_weight=0.2,
        kl_weight=0.3,
        ce_weight=0.1,
        margin_weight=0.1,
        margin=0.05,
        kl_temperature=1.7,
    )
    tiled = _distillation_loss_from_hidden(hidden, head, teacher, **kwargs)
    full = _distillation_loss(head(hidden), teacher, **kwargs)
    torch.testing.assert_close(tiled, full, rtol=1e-5, atol=1e-5)
    tiled.backward()
    assert head.weight.grad is not None


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


def test_distillation_loss_zero_weight_matches_mse():
    import torch

    student = torch.randn(2, 3, 11)
    teacher = torch.randn(2, 3, 11)
    assert torch.allclose(
        _distillation_loss(student, teacher, chunk_size=4),
        _chunked_mse(student, teacher, chunk_size=4),
    )


def test_distillation_loss_cosine_term_is_finite():
    import torch

    student = torch.ones(1, 1, 7)
    teacher = torch.ones(1, 1, 7)
    loss = _distillation_loss(student, teacher, chunk_size=3, cosine_weight=0.5)
    assert torch.isfinite(loss)
    assert loss.item() < 1e-6


def test_distillation_loss_ce_term_is_not_short_circuited():
    import torch

    student = torch.tensor([[[8.0, 0.0, 0.0]]])
    teacher = torch.tensor([[[0.0, 8.0, 0.0]]])
    mse = _chunked_mse(student, teacher, chunk_size=2)
    loss = _distillation_loss(student, teacher, chunk_size=2, ce_weight=0.5)
    assert torch.isfinite(loss)
    assert not torch.allclose(loss, mse)


def test_top2_margin_term_penalizes_teacher_argmax_swap():
    import torch

    teacher = torch.tensor([[[4.0, 3.0, 0.0]]])
    swapped = torch.tensor([[[3.0, 4.0, 0.0]]])
    aligned = torch.tensor([[[4.0, 3.0, 0.0]]])
    assert _teacher_top2_margin_loss(swapped, teacher, margin=0.5).item() == pytest.approx(1.5)
    assert _teacher_top2_margin_loss(aligned, teacher, margin=0.5).item() == pytest.approx(0.0)


def test_distillation_loss_margin_term_is_not_short_circuited():
    import torch

    teacher = torch.tensor([[[4.0, 3.0, 0.0]]])
    student = torch.tensor([[[3.0, 4.0, 0.0]]])
    mse = _chunked_mse(student, teacher, chunk_size=2)
    loss = _distillation_loss(
        student, teacher, chunk_size=2, margin_weight=0.5, margin=0.5
    )
    assert torch.isfinite(loss)
    assert not torch.allclose(loss, mse)


def test_quality_gate_requires_cosine_and_top1():
    assert quality_gate_passed(0.999, 0.999, 0.99)
    assert not quality_gate_passed(0.999, 0.84, 0.99)
    assert not quality_gate_passed(0.84, 0.999, 0.99)


def test_chunked_kl_is_zero_for_identical_logits():
    import torch

    logits = torch.randn(2, 3, 13)
    loss = _chunked_kl_divergence(logits, logits, chunk_size=4)
    assert torch.isfinite(loss)
    assert loss.item() < 1e-5


def test_hf_loader_supplies_phi_remote_code_loss_kwargs():
    transformers = pytest.importorskip("transformers")
    from qcc_transformer.hf_loading import _ensure_remote_code_compat

    utils = transformers.utils
    had_symbol = hasattr(utils, "LossKwargs")
    previous = getattr(utils, "LossKwargs", None)
    try:
        if had_symbol:
            delattr(utils, "LossKwargs")
        _ensure_remote_code_compat()
        assert hasattr(utils, "LossKwargs")
    finally:
        if had_symbol:
            setattr(utils, "LossKwargs", previous)
        else:
            delattr(utils, "LossKwargs")


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA unavailable")
def test_chunked_kl_supports_cpu_teacher_and_cuda_student():
    import torch

    teacher = torch.randn(1, 2, 17)
    student = teacher.detach().cuda().requires_grad_()
    loss = _chunked_kl_divergence(student, teacher, chunk_size=5)
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
