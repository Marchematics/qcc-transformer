"""Layer-wise calibration for QCC HF retrofit.

Allows selective calibration of specific layers (e.g., only later layers) and
evaluates on both training and held-out text to detect overfitting.  The goal
is to reach >=0.99 cosine similarity on held-out text while keeping trainable
parameters <=1% of the base model.

Priority: later layers are closer to logits and may need less capacity to
reach the fidelity gate.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcc_transformer import patch_hf_model, reset_hf_qcc_cache, save_retrofit_adapter
from qcc_transformer.hf_loading import load_hf_causal_lm, model_input_device


def _chunked_mse(student: torch.Tensor, teacher_cpu: torch.Tensor, *, chunk_size: int = 8192) -> torch.Tensor:
    """Compute full-vocabulary MSE with teacher targets streamed from CPU."""
    if student.shape != teacher_cpu.shape:
        raise ValueError(f"student/teacher shape mismatch: {student.shape} vs {teacher_cpu.shape}")
    total = student.new_zeros(())
    for start in range(0, student.shape[-1], chunk_size):
        end = min(start + chunk_size, student.shape[-1])
        target = teacher_cpu[..., start:end].to(device=student.device, dtype=student.dtype)
        total = total + torch.nn.functional.mse_loss(
            student[..., start:end], target, reduction="sum"
        )
    return total / student.numel()


def _mean_cosine_from_cpu(student: torch.Tensor, teacher_cpu: torch.Tensor, *, chunk_size: int = 8192) -> torch.Tensor:
    """Mean per-token cosine while keeping the teacher off GPU."""
    if student.shape != teacher_cpu.shape:
        raise ValueError(f"student/teacher shape mismatch: {student.shape} vs {teacher_cpu.shape}")
    shape = student.shape[:-1]
    dot = student.new_zeros(shape)
    student_sq = student.new_zeros(shape)
    teacher_sq = student.new_zeros(shape)
    for start in range(0, student.shape[-1], chunk_size):
        end = min(start + chunk_size, student.shape[-1])
        s = student[..., start:end]
        t = teacher_cpu[..., start:end].to(device=student.device, dtype=student.dtype)
        dot = dot + (s * t).sum(dim=-1)
        student_sq = student_sq + (s * s).sum(dim=-1)
        teacher_sq = teacher_sq + (t * t).sum(dim=-1)
    return (dot / (student_sq.sqrt() * teacher_sq.sqrt()).clamp_min(1e-12)).mean()


def _chunked_kl_divergence(
    student: torch.Tensor,
    teacher_cpu: torch.Tensor,
    *,
    chunk_size: int = 8192,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Match the teacher distribution with bounded GPU temporaries."""
    if student.shape != teacher_cpu.shape:
        raise ValueError(f"student/teacher shape mismatch: {student.shape} vs {teacher_cpu.shape}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher = teacher_cpu.float() / temperature
    # Keep the full teacher logits on CPU, but move the token-wise normalizer
    # to the student device.  It is only [batch, tokens], unlike the vocab
    # dimension, and avoids a CPU/GPU operation in the KL reduction below.
    teacher_log_norm = torch.logsumexp(teacher, dim=-1).to(device=student.device)
    student_log_norm = torch.full(
        student.shape[:-1], -torch.inf, device=student.device, dtype=torch.float32
    )
    for start in range(0, student.shape[-1], chunk_size):
        end = min(start + chunk_size, student.shape[-1])
        student_slice = student[..., start:end].float() / temperature
        student_log_norm = torch.logaddexp(
            student_log_norm, torch.logsumexp(student_slice, dim=-1)
        )
    total = student.new_zeros((), dtype=torch.float32)
    for start in range(0, student.shape[-1], chunk_size):
        end = min(start + chunk_size, student.shape[-1])
        teacher_slice = teacher[..., start:end].to(device=student.device)
        student_slice = student[..., start:end].float() / temperature
        teacher_log_prob = teacher_slice - teacher_log_norm.unsqueeze(-1)
        student_log_prob = student_slice - student_log_norm.unsqueeze(-1)
        total = total + (
            teacher_log_prob.exp() * (teacher_log_prob - student_log_prob)
        ).sum()
    tokens = max(1, student.numel() // student.shape[-1])
    return total * (temperature * temperature) / tokens


def _teacher_argmax_cross_entropy(
    student: torch.Tensor, teacher_cpu: torch.Tensor
) -> torch.Tensor:
    """Match the teacher's selected token without copying its vocabulary logits."""
    if student.shape != teacher_cpu.shape:
        raise ValueError(f"student/teacher shape mismatch: {student.shape} vs {teacher_cpu.shape}")
    targets = teacher_cpu.argmax(dim=-1).to(device=student.device)
    return torch.nn.functional.cross_entropy(
        student.reshape(-1, student.shape[-1]), targets.reshape(-1)
    )


def _teacher_top2_margin_loss(
    student: torch.Tensor,
    teacher_cpu: torch.Tensor,
    *,
    margin: float = 0.0,
) -> torch.Tensor:
    """Penalize a student argmax that loses to the teacher's top token.

    MSE and cosine can be excellent while a small logit swap still changes the
    generated token.  A bounded top-2 hinge term targets that failure directly
    without moving the full teacher vocabulary back to the accelerator.
    """

    if student.shape != teacher_cpu.shape:
        raise ValueError(f"student/teacher shape mismatch: {student.shape} vs {teacher_cpu.shape}")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if student.shape[-1] < 2:
        return student.new_zeros((), dtype=torch.float32)
    top2 = teacher_cpu.topk(2, dim=-1).indices.to(device=student.device)
    selected = student.gather(-1, top2)
    return torch.relu(
        student.new_tensor(float(margin)) - selected[..., 0] + selected[..., 1]
    ).mean()


def quality_gate_passed(cosine: float, top1: float, threshold: float) -> bool:
    """Apply the Full-KV fidelity threshold to both cosine and top-1 metrics."""

    return cosine >= threshold and top1 >= threshold


def _distillation_loss(
    student: torch.Tensor,
    teacher_cpu: torch.Tensor,
    *,
    chunk_size: int = 8192,
    cosine_weight: float = 0.0,
    kl_weight: float = 0.0,
    ce_weight: float = 0.0,
    margin_weight: float = 0.0,
    margin: float = 0.0,
    kl_temperature: float = 2.0,
) -> torch.Tensor:
    """Numerically stable logit distillation objective.

    The original calibration used MSE only, which over-weights high-magnitude
    vocabulary coordinates and can yield a good train loss but poor held-out
    ranking fidelity.  A bounded cosine term directly optimizes directional
    agreement while retaining the MSE scale.  ``cosine_weight=0`` is exactly
    the historical objective for reproducibility.
    """
    if (
        not 0.0 <= cosine_weight <= 1.0
        or not 0.0 <= kl_weight <= 1.0
        or not 0.0 <= ce_weight <= 1.0
        or not 0.0 <= margin_weight <= 1.0
    ):
        raise ValueError(
            "cosine_weight, kl_weight, ce_weight, and margin_weight must lie in [0, 1]"
        )
    if cosine_weight + kl_weight + ce_weight + margin_weight > 1.0:
        raise ValueError(
            "cosine_weight + kl_weight + ce_weight + margin_weight must not exceed 1"
        )
    if margin < 0:
        raise ValueError("margin must be non-negative")
    mse = _chunked_mse(student, teacher_cpu, chunk_size=chunk_size)
    if (
        cosine_weight == 0.0
        and kl_weight == 0.0
        and ce_weight == 0.0
        and margin_weight == 0.0
    ):
        return mse
    cosine = (
        _mean_cosine_from_cpu(student, teacher_cpu, chunk_size=chunk_size)
        if cosine_weight
        else student.new_zeros((), dtype=torch.float32)
    )
    kl = (
        _chunked_kl_divergence(
            student,
            teacher_cpu,
            chunk_size=chunk_size,
            temperature=kl_temperature,
        )
        if kl_weight
        else student.new_zeros((), dtype=torch.float32)
    )
    ce = (
        _teacher_argmax_cross_entropy(student, teacher_cpu)
        if ce_weight
        else student.new_zeros((), dtype=torch.float32)
    )
    margin_loss = (
        _teacher_top2_margin_loss(student, teacher_cpu, margin=margin)
        if margin_weight
        else student.new_zeros((), dtype=torch.float32)
    )
    return (
        (1.0 - cosine_weight - kl_weight - ce_weight - margin_weight) * mse
        + cosine_weight * (1.0 - cosine)
        + kl_weight * kl
        + ce_weight * ce
        + margin_weight * margin_loss
    )


def parse_layer_spec(spec: str, num_layers: int) -> list[int]:
    """Parse layer specification like '8-15', '0,2,4', or 'last-half'."""
    spec = spec.strip().lower()
    if num_layers <= 0:
        raise ValueError("model must contain at least one compatible attention layer")
    if spec == "all":
        layers = list(range(num_layers))
    elif spec == "last-half":
        layers = list(range(num_layers // 2, num_layers))
    elif spec == "first-half":
        layers = list(range(num_layers // 2))
    elif spec == "last-quarter":
        layers = list(range(3 * num_layers // 4, num_layers))
    elif "-" in spec:
        parts = spec.split("-")
        if len(parts) != 2:
            raise ValueError(f"invalid layer range: {spec!r}")
        start, end = (int(part) for part in parts)
        layers = list(range(start, end + 1))
    elif "," in spec:
        layers = [int(x.strip()) for x in spec.split(",")]
    else:
        layers = [int(spec)]
    if not layers:
        raise ValueError("calibrate-layers must select at least one layer")
    invalid = sorted({layer for layer in layers if layer < 0 or layer >= num_layers})
    if invalid:
        raise ValueError(
            f"calibrate-layers contains invalid layer indices {invalid}; "
            f"model has layers 0 through {num_layers - 1}"
        )
    return sorted(set(layers))


def _encode_positioned_chunks(
    tokenizer,
    text: str,
    *,
    max_tokens: int,
    num_chunks: int,
    window_size: int,
) -> list[dict[str, torch.Tensor]]:
    """Tokenize evenly spaced windows and preserve each window's offset.

    Passing no ``position_ids`` made every calibration slice start at RoPE
    position zero, even when it came from the middle of a long document.  The
    student then saw a different positional distribution from serving.  Keep
    the original token offset explicit for both teacher and student.
    """

    if max_tokens <= window_size or num_chunks <= 0:
        raise ValueError(
            "max_tokens must exceed window_size and num_chunks must be positive"
        )
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    ids = encoded["input_ids"][0]
    mask = encoded.get("attention_mask")
    if ids.numel() == 0:
        raise ValueError("text did not produce any tokens")
    if ids.numel() <= max_tokens:
        starts = [0]
    else:
        max_start = int(ids.numel() - max_tokens)
        count = min(num_chunks, max_start + 1)
        starts = torch.linspace(0, max_start, count).round().to(torch.long).tolist()

    batches: list[dict[str, torch.Tensor]] = []
    for raw_start in starts:
        start = int(raw_start)
        chunk_ids = ids[start : start + max_tokens]
        if chunk_ids.numel() <= window_size:
            continue
        batch: dict[str, torch.Tensor] = {
            "input_ids": chunk_ids.unsqueeze(0),
            "position_ids": torch.arange(
                start, start + chunk_ids.numel(), dtype=torch.long
            ).view(1, -1),
        }
        if mask is not None:
            batch["attention_mask"] = mask[
                0, start : start + chunk_ids.numel()
            ].unsqueeze(0)
        batches.append(batch)
    if not batches:
        raise ValueError("text did not produce any chunk longer than window-size")
    return batches


@torch.no_grad()
def _initialize_codebooks_from_teacher(
    model,
    teacher_hidden: dict[int, torch.Tensor],
    *,
    strategy: str,
) -> None:
    """Initialize archive codes from real teacher key projections.

    The archive codebook is in key space.  Starting it from teacher key geometry
    gives the first optimizer step useful addresses and leaves the code vectors
    fully trainable afterwards.  The teacher snapshots are bounded CPU samples,
    so this does not change the bounded activation memory budget.  ``random``
    remains available as an explicit ablation.
    """

    if strategy not in {"key-sample", "kmeans", "random"}:
        raise ValueError("strategy must be 'key-sample', 'kmeans', or 'random'")
    if strategy == "random":
        return
    wrappers = [
        module
        for module in model.modules()
        if getattr(module, "qcc", None) is not None
    ]
    for wrapper in wrappers:
        qcc = wrapper.qcc
        layer_index = getattr(qcc, "_qcc_layer_index", None)
        if layer_index not in teacher_hidden:
            continue
        hidden = teacher_hidden[layer_index].to(
            device=qcc.archive.codes.device,
            dtype=qcc.q_proj.weight.dtype,
        )
        _, key, _, _ = qcc._project_qkv_gate(hidden)
        keys = qcc._split_heads(key)
        tokens = keys.shape[2]
        if tokens <= 0:
            continue
        code_count = qcc.archive.num_codes
        if strategy == "kmeans" and tokens > 1:
            # Attention routing is primarily directional.  A deterministic
            # cosine k-means++-style pass covers the teacher key manifold much
            # better than temporal sampling, while the fixed code norm keeps
            # the archive logits in the same range as the original init.
            source = F.normalize(keys[0].float(), dim=-1)
            centers = []
            for head in range(source.shape[0]):
                points = source[head]
                first = 0
                selected = [first]
                center = points[first : first + 1]
                for _ in range(1, min(code_count, tokens)):
                    distance = 1.0 - points @ center.transpose(0, 1)
                    next_index = int(distance.min(dim=1).values.argmax().item())
                    selected.append(next_index)
                    center = torch.cat((center, points[next_index : next_index + 1]), dim=0)
                center = points[torch.tensor(selected, device=points.device)]
                for _ in range(8):
                    assignment = (points @ center.transpose(0, 1)).argmax(dim=1)
                    updated = torch.zeros_like(center)
                    updated.index_add_(0, assignment, points)
                    counts = torch.zeros(
                        center.shape[0], device=assignment.device, dtype=updated.dtype
                    )
                    counts.index_add_(
                        0, assignment, torch.ones_like(assignment, dtype=updated.dtype)
                    )
                    counts = counts.unsqueeze(-1)
                    center = F.normalize(
                        torch.where(counts > 0, updated / counts.clamp_min(1.0), center),
                        dim=-1,
                    )
                if center.shape[0] < code_count:
                    center = torch.cat(
                        (center, center[torch.arange(code_count - center.shape[0], device=center.device) % center.shape[0]]),
                        dim=0,
                    )
                centers.append(center[:code_count])
            sampled = torch.stack(centers, dim=0)
        else:
            indices = torch.arange(code_count, device=keys.device) % tokens
            if tokens > 1:
                indices = torch.linspace(
                    0, tokens - 1, code_count, device=keys.device
                ).round().to(torch.long)
            sampled = keys[0, :, indices, :]
            sampled = F.normalize(sampled.float(), dim=-1)
        # Match the existing random initialization scale per head.  This keeps
        # routing logits in a comparable range while preserving key geometry.
        target_norm = qcc.archive.codes.detach().float().norm(dim=-1).mean(dim=-1)
        sampled = sampled * target_norm.view(-1, 1, 1)
        qcc.archive.codes.copy_(sampled.to(dtype=qcc.archive.codes.dtype))


def _long_range_view(tensor: torch.Tensor, window_size: int) -> torch.Tensor:
    """Return only positions whose attention can use the archive."""

    if tensor.ndim < 2:
        raise ValueError("tensor must have a sequence dimension")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if tensor.shape[-2] <= window_size:
        return tensor
    return tensor[..., window_size:, :]


@torch.no_grad()
def _teacher_logits_and_inputs(
    model,
    batches: list[dict[str, torch.Tensor]],
    *,
    max_capture_tokens: int = 256,
    selected_layers: set[int] | None = None,
    attention_start: int = 0,
) -> tuple[
    list[torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, list[torch.Tensor]],
]:
    """Run teacher batches and retain bounded K-init snapshots.

    The logits are kept for every batch because they are the distillation
    target.  Hidden states are only used to seed the codebook, so retaining a
    small, evenly spaced sample from every batch gives better coverage of a
    long calibration stream without turning the CPU staging area into another
    copy of the model activations.
    """

    if max_capture_tokens <= 0:
        raise ValueError("max_capture_tokens must be positive")
    if attention_start < 0:
        raise ValueError("attention_start must be non-negative")

    attention_modules = [
        module
        for module in model.modules()
        if (
            all(hasattr(module, field) for field in ("q_proj", "k_proj", "v_proj", "o_proj"))
            or (hasattr(module, "qkv_proj") and hasattr(module, "o_proj"))
        )
    ]
    captured: dict[int, list[torch.Tensor]] = {
        index: [] for index in range(len(attention_modules))
    }
    attention_targets: dict[int, list[torch.Tensor]] = {
        index: []
        for index in range(len(attention_modules))
        if selected_layers is None or index in selected_layers
    }
    batch_captured: dict[int, torch.Tensor] = {}
    batch_attention: dict[int, torch.Tensor] = {}
    hooks = []
    for index, module in enumerate(attention_modules):
        def capture(_module, inputs, kwargs, *, layer_index=index):
            hidden = inputs[0] if inputs else kwargs.get("hidden_states")
            if hidden is not None and layer_index not in batch_captured:
                batch_captured[layer_index] = hidden.detach()

        hooks.append(module.register_forward_pre_hook(capture, with_kwargs=True))
        if index in attention_targets:
            def capture_output(_module, _inputs, output, *, layer_index=index):
                if layer_index in batch_attention:
                    return
                value = output[0] if isinstance(output, (tuple, list)) else output
                if isinstance(value, torch.Tensor):
                    batch_attention[layer_index] = value.detach()

            hooks.append(module.register_forward_hook(capture_output))
    try:
        teachers = []
        for batch in batches:
            batch_captured = {}
            batch_attention = {}
            teachers.append(model(**batch, use_cache=False).logits.float().cpu())
            for layer_index, hidden in batch_captured.items():
                flat = hidden.reshape(-1, hidden.shape[-1]).cpu()
                take = min(max_capture_tokens, flat.shape[0])
                indices = torch.linspace(
                    0, flat.shape[0] - 1, take, dtype=torch.long
                )
                captured[layer_index].append(flat[indices])
            for layer_index, output in batch_attention.items():
                if output.ndim == 3 and attention_start and output.shape[1] > attention_start:
                    output = output[:, attention_start:]
                flat = output.reshape(-1, output.shape[-1]).cpu()
                take = min(max_capture_tokens, flat.shape[0])
                indices = torch.linspace(
                    0, flat.shape[0] - 1, take, dtype=torch.long
                )
                attention_targets[layer_index].append(flat[indices])
    finally:
        for hook in hooks:
            hook.remove()
    missing_attention = [
        layer_index
        for layer_index, targets in attention_targets.items()
        if len(targets) != len(batches)
    ]
    if missing_attention:
        raise RuntimeError(
            "teacher did not expose attention outputs for every batch in layers "
            f"{missing_attention}"
        )
    snapshots = {
        layer_index: torch.cat(chunks, dim=0).unsqueeze(0)
        for layer_index, chunks in captured.items()
        if chunks
    }
    return teachers, snapshots, attention_targets


def _hidden_distillation_loss(
    student: torch.Tensor,
    teacher_cpu: torch.Tensor,
    *,
    cosine_weight: float = 0.5,
) -> torch.Tensor:
    """Match selected attention outputs without a vocabulary-sized target."""

    if student.shape != teacher_cpu.shape:
        raise ValueError(
            f"student/teacher hidden shape mismatch: {student.shape} vs {teacher_cpu.shape}"
        )
    if not 0.0 <= cosine_weight <= 1.0:
        raise ValueError("cosine_weight must lie in [0, 1]")
    # Phi attention outputs can exceed fp16's safe square range even when the
    # final logits remain finite.  Keep this bounded auxiliary objective in
    # fp32; only the frozen backbone and trainable adapter stay in model dtype.
    student_f = student.float()
    teacher = teacher_cpu.to(device=student.device, dtype=torch.float32)
    mse = torch.nn.functional.mse_loss(student_f, teacher)
    cosine = torch.nn.functional.cosine_similarity(student_f, teacher, dim=-1).mean()
    return (1.0 - cosine_weight) * mse + cosine_weight * (1.0 - cosine)


def _forward_hidden_and_head(
    model,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.nn.Module] | None:
    """Run an HF backbone without materializing its full vocabulary logits."""

    backbone = getattr(model, "model", None)
    get_head = getattr(model, "get_output_embeddings", None)
    if backbone is None or not callable(get_head):
        return None
    head = get_head()
    if head is None:
        return None
    try:
        outputs = backbone(**batch, use_cache=False, return_dict=True)
    except TypeError:
        outputs = backbone(**batch, use_cache=False)
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None and isinstance(outputs, (tuple, list)) and outputs:
        hidden = outputs[0]
    if not isinstance(hidden, torch.Tensor):
        return None
    return hidden, head


def _lm_head_slice(
    hidden: torch.Tensor,
    head: torch.nn.Module,
    start: int,
    end: int,
) -> torch.Tensor:
    """Project one vocabulary slice, retaining a bounded student temporary."""

    weight = getattr(head, "weight", None)
    if isinstance(weight, torch.Tensor):
        bias = getattr(head, "bias", None)
        bias_slice = bias[start:end] if isinstance(bias, torch.Tensor) else None
        return torch.nn.functional.linear(hidden, weight[start:end], bias_slice)
    # Quantized/custom output heads may not expose a sliceable weight. This
    # fallback preserves correctness; normal HF Linear heads use the bounded
    # path above.
    return head(hidden)[..., start:end]


def _distillation_loss_from_hidden(
    hidden: torch.Tensor,
    head: torch.nn.Module,
    teacher_cpu: torch.Tensor,
    *,
    chunk_size: int = 8192,
    cosine_weight: float = 0.0,
    kl_weight: float = 0.0,
    ce_weight: float = 0.0,
    margin_weight: float = 0.0,
    margin: float = 0.0,
    kl_temperature: float = 2.0,
) -> torch.Tensor:
    """Compute the logit objective from hidden states in vocab-sized tiles."""

    if hidden.shape[:-1] != teacher_cpu.shape[:-1]:
        raise ValueError(
            f"student/teacher hidden shape mismatch: {hidden.shape} vs {teacher_cpu.shape}"
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if (
        not 0.0 <= cosine_weight <= 1.0
        or not 0.0 <= kl_weight <= 1.0
        or not 0.0 <= ce_weight <= 1.0
        or not 0.0 <= margin_weight <= 1.0
    ):
        raise ValueError("distillation weights must lie in [0, 1]")
    if cosine_weight + kl_weight + ce_weight + margin_weight > 1.0:
        raise ValueError("distillation weights must not sum above 1")
    if margin < 0 or kl_temperature <= 0:
        raise ValueError("margin must be non-negative and temperature must be positive")

    shape = hidden.shape[:-1]
    vocab = teacher_cpu.shape[-1]
    teacher = teacher_cpu.float()
    mse_sum = hidden.new_zeros((), dtype=torch.float32)
    dot = hidden.new_zeros(shape, dtype=torch.float32)
    student_sq = hidden.new_zeros(shape, dtype=torch.float32)
    teacher_sq = hidden.new_zeros(shape, dtype=torch.float32)
    student_log_norm = torch.full(
        shape, -torch.inf, device=hidden.device, dtype=torch.float32
    )
    student_ce_log_norm = torch.full(
        shape, -torch.inf, device=hidden.device, dtype=torch.float32
    )
    teacher_log_norm = (
        torch.logsumexp(teacher / kl_temperature, dim=-1).to(device=hidden.device)
        if kl_weight
        else None
    )
    targets = teacher.argmax(dim=-1).to(device=hidden.device)
    target_logits = hidden.new_zeros(shape, dtype=torch.float32)
    top2 = teacher.topk(2, dim=-1).indices if margin_weight and vocab >= 2 else None
    top2_logits = (
        hidden.new_zeros((*shape, 2), dtype=torch.float32)
        if top2 is not None
        else None
    )

    for start in range(0, vocab, chunk_size):
        end = min(start + chunk_size, vocab)
        student_slice = _lm_head_slice(hidden, head, start, end).float()
        teacher_slice = teacher[..., start:end].to(device=hidden.device)
        mse_sum = mse_sum + torch.nn.functional.mse_loss(
            student_slice, teacher_slice, reduction="sum"
        )
        if cosine_weight:
            dot = dot + (student_slice * teacher_slice).sum(dim=-1)
            student_sq = student_sq + student_slice.square().sum(dim=-1)
            teacher_sq = teacher_sq + teacher_slice.square().sum(dim=-1)
        if kl_weight:
            student_log_norm = torch.logaddexp(
                student_log_norm,
                torch.logsumexp(student_slice / kl_temperature, dim=-1),
            )
        if ce_weight:
            student_ce_log_norm = torch.logaddexp(
                student_ce_log_norm,
                torch.logsumexp(student_slice, dim=-1),
            )
            selected = (targets - start).clamp(0, end - start - 1)
            selected_logits = student_slice.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
            target_logits = torch.where(
                (targets >= start) & (targets < end), selected_logits, target_logits
            )
        if top2_logits is not None and top2 is not None:
            for rank in range(2):
                indices = (top2[..., rank] - start).clamp(0, end - start - 1)
                selected_logits = student_slice.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
                top2_logits[..., rank] = torch.where(
                    (top2[..., rank] >= start) & (top2[..., rank] < end),
                    selected_logits,
                    top2_logits[..., rank],
                )

    mse = mse_sum / teacher_cpu.numel()
    cosine = hidden.new_zeros((), dtype=torch.float32)
    if cosine_weight:
        cosine = (
            dot / (student_sq.sqrt() * teacher_sq.sqrt()).clamp_min(1e-12)
        ).mean()
    kl = hidden.new_zeros((), dtype=torch.float32)
    if kl_weight:
        for start in range(0, vocab, chunk_size):
            end = min(start + chunk_size, vocab)
            student_slice = _lm_head_slice(hidden, head, start, end).float()
            teacher_slice = teacher[..., start:end].to(device=hidden.device)
            assert teacher_log_norm is not None
            teacher_log_prob = teacher_slice / kl_temperature - teacher_log_norm.unsqueeze(-1)
            student_log_prob = student_slice / kl_temperature - student_log_norm.unsqueeze(-1)
            kl = kl + (
                teacher_log_prob.exp() * (teacher_log_prob - student_log_prob)
            ).sum()
        tokens = max(1, teacher_cpu.numel() // vocab)
        kl = kl * (kl_temperature * kl_temperature) / tokens
    ce = hidden.new_zeros((), dtype=torch.float32)
    if ce_weight:
        ce = (student_ce_log_norm - target_logits).mean()
    ranking = hidden.new_zeros((), dtype=torch.float32)
    if top2_logits is not None:
        ranking = torch.relu(
            hidden.new_tensor(float(margin))
            - top2_logits[..., 0]
            + top2_logits[..., 1]
        ).mean()
    return (
        (1.0 - cosine_weight - kl_weight - ce_weight - margin_weight) * mse
        + cosine_weight * (1.0 - cosine)
        + kl_weight * kl
        + ce_weight * ce
        + margin_weight * ranking
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", type=Path, required=True,
                        help="training text file")
    parser.add_argument("--held-out-file", type=Path, default=None,
                        help="held-out text file for quality gate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--num-codes", type=int, default=32)
    parser.add_argument(
        "--calibrate-layers",
        default="all",
        help="layers to calibrate: 'all', 'last-half', 'first-half', 'last-quarter', '8-15', or '0,2,4'",
    )
    parser.add_argument(
        "--archive-position-invariant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use raw Q/K for archive; local attention keeps RoPE",
    )
    parser.add_argument(
        "--archive-persistent-landmark", action="store_true",
        help="retain one exact max-salience key/value landmark per archive code",
    )
    parser.add_argument(
        "--archive-prefix-landmark", action="store_true",
        help="make persistent landmarks temporal prefix slots (requires persistent landmark)",
    )
    parser.add_argument(
        "--archive-prefix-pair-landmark", action="store_true",
        help="store paired prefix landmarks (requires prefix landmark)",
    )
    parser.add_argument(
        "--archive-landmark-temperature", type=float, default=1.0,
        help="routing temperature multiplier for persistent landmarks",
    )
    parser.add_argument(
        "--archive-norm-gating", action="store_true",
        help="parameter-free norm agreement gate for archive contribution",
    )
    parser.add_argument(
        "--archive-kernel-features", action="store_true",
        help="use positive random-feature softmax kernel archive",
    )
    parser.add_argument(
        "--archive-global-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="combine code/scale numerator and denominator before normalization",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--archive-scan-block-size",
        type=int,
        default=256,
        help="bounded archive scan block used during calibration; smaller values reduce peak memory",
    )
    parser.add_argument(
        "--num-train-chunks", type=int, default=1,
        help="number of evenly spaced training chunks to distill (cycles across chunks each step)",
    )
    parser.add_argument(
        "--num-held-out-chunks", type=int, default=1,
        help="number of evenly spaced held-out chunks used for validation",
    )
    parser.add_argument(
        "--code-init",
        choices=("key-sample", "kmeans", "random"),
        default="kmeans",
        help="initialize archive codes from teacher K projections, cosine k-means, or random initialization",
    )
    parser.add_argument(
        "--code-init-tokens",
        type=int,
        default=256,
        help="teacher tokens sampled per training chunk for codebook initialization",
    )
    parser.add_argument(
        "--attention-loss-weight",
        type=float,
        default=0.35,
        help="blend weight for selected-layer attention-output distillation",
    )
    parser.add_argument(
        "--distill-long-range-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train on positions outside the exact local window; use --no-distill-long-range-only for the legacy full-sequence loss",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--load-in-4bit", action="store_true", help="load the real checkpoint through bitsandbytes NF4")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--kv-head-policy", choices=("reject", "repeat"), default="repeat")
    parser.add_argument(
        "--gate-bias-init", type=float, default=2.0,
        help="initial local-path bias; use 0.0 for the historical 50/50 ablation",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="checkpoint frozen blocks to reduce memory",
    )
    parser.add_argument(
        "--cpu-offload-activations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="move autograd-saved activations to CPU during calibration to fit full-layer runs on small GPUs",
    )
    parser.add_argument(
        "--quality-gate",
        type=float,
        default=0.99,
        help="minimum held-out cosine and top-1 agreement for fidelity gate",
    )
    parser.add_argument(
        "--cosine-weight", type=float, default=0.0,
        help="weight of directional (1-cosine) term in calibration loss; "
             "0 preserves historical MSE-only behavior",
    )
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument(
        "--ce-weight",
        type=float,
        default=0.0,
        help="weight of teacher-argmax cross entropy; 0 preserves the historical objective",
    )
    parser.add_argument(
        "--margin-weight",
        type=float,
        default=0.0,
        help="weight of a teacher top-2 ranking hinge term; 0 preserves the historical objective",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="minimum student logit margin for the teacher's top token",
    )
    parser.add_argument("--kl-temperature", type=float, default=2.0)
    args = parser.parse_args()

    if (
        args.steps <= 0
        or args.lr <= 0
        or args.max_tokens <= 0
        or args.num_train_chunks <= 0
        or args.num_held_out_chunks <= 0
        or args.code_init_tokens <= 0
        or not 0.0 <= args.attention_loss_weight <= 1.0
    ):
        raise ValueError(
            "steps, lr, max-tokens, chunk counts, and code-init-tokens must be positive; "
            "attention-loss-weight must lie in [0, 1]"
        )
    if args.archive_scan_block_size <= 0:
        raise ValueError("archive-scan-block-size must be positive")
    if (
        not 0.0 <= args.cosine_weight <= 1.0
        or not 0.0 <= args.kl_weight <= 1.0
        or not 0.0 <= args.ce_weight <= 1.0
        or not 0.0 <= args.margin_weight <= 1.0
    ):
        raise ValueError(
            "cosine-weight, kl-weight, ce-weight, and margin-weight must lie in [0, 1]"
        )
    if args.cosine_weight + args.kl_weight + args.ce_weight + args.margin_weight > 1.0:
        raise ValueError(
            "cosine-weight + kl-weight + ce-weight + margin-weight must not exceed 1"
        )
    if args.margin < 0:
        raise ValueError("margin must be non-negative")
    if args.kl_temperature <= 0:
        raise ValueError("kl-temperature must be positive")
    if args.archive_prefix_landmark and not args.archive_persistent_landmark:
        raise ValueError("archive-prefix-landmark requires archive-persistent-landmark")
    if args.archive_prefix_pair_landmark and not args.archive_prefix_landmark:
        raise ValueError("archive-prefix-pair-landmark requires archive-prefix-landmark")
    if args.max_tokens <= args.window_size:
        raise ValueError("max-tokens must exceed window-size to exercise archive")

    train_text = args.train_file.read_text(encoding="utf-8")
    if not train_text.strip():
        raise ValueError("train-file must contain text")

    held_out_text = None
    if args.held_out_file:
        held_out_text = args.held_out_file.read_text(encoding="utf-8")
        if not held_out_text.strip():
            raise ValueError("held-out-file must contain text")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("install qcc-transformer[hf]") from exc

    requested_device = torch.device(args.device)
    dtype = None if args.dtype == "auto" else {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    device = requested_device
    common = {"trust_remote_code": args.trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **common)

    train_batches = _encode_positioned_chunks(
        tokenizer,
        train_text,
        max_tokens=args.max_tokens,
        num_chunks=args.num_train_chunks,
        window_size=args.window_size,
    )
    held_out_batches = (
        _encode_positioned_chunks(
            tokenizer,
            held_out_text,
            max_tokens=args.max_tokens,
            num_chunks=args.num_held_out_chunks,
            window_size=args.window_size,
        )
        if held_out_text
        else []
    )

    # Capture teacher logits and release immediately
    print("Loading teacher model...", file=sys.stderr)
    baseline = load_hf_causal_lm(
        args.model,
        dtype=dtype,
        device=requested_device,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
    )
    device = model_input_device(baseline, requested_device)
    train_batches = [
        {key: value.to(device) for key, value in batch.items()}
        for batch in train_batches
    ]
    held_out_batches = [
        {key: value.to(device) for key, value in batch.items()}
        for batch in held_out_batches
    ]

    teacher_attention_modules = [
        module
        for module in baseline.modules()
        if (
            all(hasattr(module, field) for field in ("q_proj", "k_proj", "v_proj", "o_proj"))
            or (hasattr(module, "qkv_proj") and hasattr(module, "o_proj"))
        )
    ]
    if not teacher_attention_modules:
        raise RuntimeError("could not locate compatible teacher attention modules")
    calibrate_layers = parse_layer_spec(
        args.calibrate_layers, len(teacher_attention_modules)
    )
    teacher_num_layers = len(teacher_attention_modules)

    with torch.no_grad():
        train_teachers, teacher_hidden, train_attention_targets = _teacher_logits_and_inputs(
            baseline,
            train_batches,
            max_capture_tokens=args.code_init_tokens,
            selected_layers=set(calibrate_layers),
            attention_start=args.window_size if args.distill_long_range_only else 0,
        )
        held_out_teachers = [
            baseline(**batch, use_cache=False).logits.float().cpu()
            for batch in held_out_batches
        ]

    # The module list contains strong references to every attention submodule.
    # Drop it before loading the student, otherwise the released teacher can
    # remain resident and a 24 GB-class GPU may OOM during checkpoint loading.
    del teacher_attention_modules
    del baseline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Loading student model...", file=sys.stderr)
    patched = load_hf_causal_lm(
        args.model,
        dtype=dtype,
        device=requested_device,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
    )
    device = model_input_device(patched, requested_device)
    train_batches = [
        {key: value.to(device) for key, value in batch.items()}
        for batch in train_batches
    ]
    held_out_batches = [
        {key: value.to(device) for key, value in batch.items()}
        for batch in held_out_batches
    ]
    replaced = patch_hf_model(
        patched,
        window_size=args.window_size,
        num_codes=args.num_codes,
        archive_position_invariant=args.archive_position_invariant,
        archive_persistent_landmark=args.archive_persistent_landmark,
        archive_prefix_landmark=args.archive_prefix_landmark,
        archive_prefix_pair_landmark=args.archive_prefix_pair_landmark,
        archive_landmark_temperature=args.archive_landmark_temperature,
        archive_norm_gating=args.archive_norm_gating,
        archive_kernel_features=args.archive_kernel_features,
        archive_scan_block_size=args.archive_scan_block_size,
        archive_global_normalization=args.archive_global_normalization,
        kv_head_policy=args.kv_head_policy,
        gate_bias_init=args.gate_bias_init,
    )
    _initialize_codebooks_from_teacher(
        patched, teacher_hidden, strategy=args.code_init
    )
    # Keep trainable adapter gates in fp32 when the frozen backbone is loaded
    # in fp16/bf16.  The attention projection path casts the gate back to the
    # backbone dtype, preserving inference behavior while avoiding half-
    # precision AdamW updates during calibration.
    for module in patched.modules():
        qcc = getattr(module, "qcc", None)
        if qcc is not None:
            qcc.gate.float()

    if args.gradient_checkpointing and hasattr(patched, "gradient_checkpointing_enable"):
        patched.config.use_cache = False
        patched.gradient_checkpointing_enable()
        if hasattr(patched, "enable_input_require_grads"):
            patched.enable_input_require_grads()

    # Determine which layers to calibrate. The teacher and patched model must
    # expose the same attention-module ordering for hidden-output supervision.
    num_layers = len(replaced)
    if num_layers != teacher_num_layers:
        raise RuntimeError(
            "teacher/student attention layer counts differ: "
            f"{teacher_num_layers} vs {num_layers}"
        )
    print(f"Calibrating layers: {calibrate_layers} out of {num_layers}", file=sys.stderr)

    # Freeze all, then selectively unfreeze
    for parameter in patched.parameters():
        parameter.requires_grad = False

    trainable = []
    for module in patched.modules():
        qcc = getattr(module, "qcc", None)
        if qcc is None:
            continue
        layer_index = getattr(qcc, "_qcc_layer_index", None)
        if layer_index is None:
            # Fallback: enable all QCC modules if layer index not tracked
            should_train = True
        else:
            should_train = layer_index in calibrate_layers

        if should_train:
            for parameter in qcc.archive.parameters():
                parameter.requires_grad = True
                trainable.append(parameter)
            for parameter in (qcc.gate.weight, qcc.gate.bias):
                parameter.requires_grad = True
                trainable.append(parameter)

    if not trainable:
        raise RuntimeError("no QCC parameters selected for training")

    # ``modules()`` may expose a wrapper and nested modules that reference the
    # same parameter.  Deduplicate before optimizer construction to prevent a
    # double update (or AdamW's duplicate-parameter error).
    unique_trainable = {id(p): p for p in trainable}
    trainable = list(unique_trainable.values())
    parameter_count = sum(p.numel() for p in patched.parameters())
    trainable_parameter_count = sum(p.numel() for p in unique_trainable.values())
    trainable_parameter_fraction = trainable_parameter_count / max(parameter_count, 1)

    print(f"Trainable: {trainable_parameter_count} / {parameter_count} "
          f"= {trainable_parameter_fraction:.4%}", file=sys.stderr)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    patched.train()

    student_attention_outputs: dict[int, torch.Tensor] = {}
    student_hooks = []
    for module in patched.modules():
        qcc = getattr(module, "qcc", None)
        layer_index = getattr(qcc, "_qcc_layer_index", None)
        if qcc is None or layer_index not in calibrate_layers:
            continue

        def capture_student(_module, _inputs, output, *, layer_index=layer_index):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(value, torch.Tensor):
                student_attention_outputs[layer_index] = value

        student_hooks.append(module.register_forward_hook(capture_student))

    try:
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            batch_index = step % len(train_batches)
            # Every calibration batch is an independent request.  QCC keeps its
            # history inside each adapted attention layer, so explicitly reset it
            # between optimizer steps instead of letting ``_seen_tokens`` turn
            # unrelated examples into one ever-growing stream.
            reset_hf_qcc_cache(
                patched, batch_size=int(train_batches[batch_index]["input_ids"].shape[0])
            )
            student_attention_outputs.clear()
            offload_context = (
                torch.autograd.graph.save_on_cpu(pin_memory=False)
                if args.cpu_offload_activations and device.type == "cuda"
                else nullcontext()
            )
            with offload_context:
                hidden_and_head = _forward_hidden_and_head(
                    patched, train_batches[batch_index]
                )
                if hidden_and_head is None:
                    student = patched(**train_batches[batch_index], use_cache=False).logits
                    teacher_target = (
                        _long_range_view(train_teachers[batch_index], args.window_size)
                        if args.distill_long_range_only
                        else train_teachers[batch_index]
                    )
                    student_target = (
                        _long_range_view(student, args.window_size)
                        if args.distill_long_range_only
                        else student
                    )
                    logit_loss = _distillation_loss(
                        student_target,
                        teacher_target,
                        cosine_weight=args.cosine_weight,
                        kl_weight=args.kl_weight,
                        ce_weight=args.ce_weight,
                        margin_weight=args.margin_weight,
                        margin=args.margin,
                        kl_temperature=args.kl_temperature,
                    )
                else:
                    student_hidden, student_head = hidden_and_head
                    teacher_target = (
                        _long_range_view(train_teachers[batch_index], args.window_size)
                        if args.distill_long_range_only
                        else train_teachers[batch_index]
                    )
                    hidden_target = (
                        _long_range_view(student_hidden, args.window_size)
                        if args.distill_long_range_only
                        else student_hidden
                    )
                    logit_loss = _distillation_loss_from_hidden(
                        hidden_target,
                        student_head,
                        teacher_target,
                        cosine_weight=args.cosine_weight,
                        kl_weight=args.kl_weight,
                        ce_weight=args.ce_weight,
                        margin_weight=args.margin_weight,
                        margin=args.margin,
                        kl_temperature=args.kl_temperature,
                    )
                attention_losses = []
                if args.attention_loss_weight:
                    for layer_index in calibrate_layers:
                        targets = train_attention_targets.get(layer_index, [])
                        if not targets:
                            continue
                        output = student_attention_outputs.get(layer_index)
                        if output is None:
                            raise RuntimeError(
                                f"student did not expose attention output for layer {layer_index}"
                            )
                        if args.distill_long_range_only and output.shape[-2] > args.window_size:
                            output = output[:, args.window_size:]
                        flat = output.reshape(-1, output.shape[-1])
                        target = targets[batch_index]
                        take = min(target.shape[0], flat.shape[0])
                        indices = torch.linspace(
                            0, flat.shape[0] - 1, take, device=flat.device, dtype=torch.long
                        )
                        attention_losses.append(
                            _hidden_distillation_loss(flat[indices], target[:take])
                        )
                    if not attention_losses:
                        raise RuntimeError(
                            "attention-output supervision requested but no targets were captured"
                        )
                attention_loss = (
                    torch.stack(attention_losses).mean()
                    if attention_losses
                    else logit_loss.new_zeros(())
                )
                loss = (
                    (1.0 - args.attention_loss_weight) * logit_loss
                    + args.attention_loss_weight * attention_loss
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "calibration diverged; lower --lr or reduce --max-tokens"
                    )
                loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            if (step + 1) % 10 == 0 or step == 0:
                print(
                    f"Step {step+1}/{args.steps}: loss={loss.item():.6f} "
                    f"logit={logit_loss.item():.6f} attention={attention_loss.item():.6f}",
                    file=sys.stderr,
                )
    finally:
        for hook in student_hooks:
            hook.remove()
        student_attention_outputs.clear()

    # Evaluate
    patched.eval()
    with torch.no_grad():
        train_cosines = []
        train_agreements = []
        for batch, teacher in zip(train_batches, train_teachers):
            reset_hf_qcc_cache(
                patched, batch_size=int(batch["input_ids"].shape[0])
            )
            train_student = patched(**batch, use_cache=False).logits
            train_cosines.append(_mean_cosine_from_cpu(train_student, teacher))
            train_agreements.append((train_student.argmax(-1).cpu() == teacher.argmax(-1)).float().mean())
        train_cosine = torch.stack(train_cosines).mean()
        train_agreement = torch.stack(train_agreements).mean()

        held_out_cosine = None
        held_out_agreement = None
        held_out_gate_passed = None

        if held_out_teachers:
            held_cosines = []
            held_agreements = []
            for batch, teacher in zip(held_out_batches, held_out_teachers):
                reset_hf_qcc_cache(
                    patched, batch_size=int(batch["input_ids"].shape[0])
                )
                held_out_student = patched(**batch, use_cache=False).logits
                held_cosines.append(_mean_cosine_from_cpu(held_out_student, teacher))
                held_agreements.append(
                    (held_out_student.argmax(-1).cpu() == teacher.argmax(-1))
                    .float()
                    .mean()
                )
            held_out_cosine = torch.stack(held_cosines).mean()
            held_out_agreement = torch.stack(held_agreements).mean()
            held_out_gate_passed = quality_gate_passed(
                float(held_out_cosine.item()),
                float(held_out_agreement.item()),
                args.quality_gate,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_retrofit_adapter(
        patched,
        args.output,
        base_model=args.model,
        model_id=str(args.model),
        run_id=args.run_id,
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        trainable_parameter_fraction=trainable_parameter_fraction,
        hf_zero_code_changes=True,
        vllm_zero_code_changes=True,
        retrofit={
            "window_size": args.window_size,
            "num_codes": args.num_codes,
            "archive_kernel_features": args.archive_kernel_features,
            "archive_scan_block_size": args.archive_scan_block_size,
            "archive_global_normalization": args.archive_global_normalization,
            "archive_position_invariant": args.archive_position_invariant,
            "patched_layers": replaced,
            "calibrated_layers": calibrate_layers,
            "kv_head_policy": args.kv_head_policy,
            "gate_bias_init": args.gate_bias_init,
            "num_train_chunks": len(train_batches),
            "num_held_out_chunks": len(held_out_batches),
            "code_init": args.code_init,
            "code_init_tokens": args.code_init_tokens,
            "attention_loss_weight": args.attention_loss_weight,
            "distill_long_range_only": args.distill_long_range_only,
            "cpu_offload_activations": args.cpu_offload_activations,
        },
    )

    result = {
        "base_model": args.model,
        "model_id": str(args.model),
        "run_id": args.run_id,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_parameter_fraction": trainable_parameter_fraction,
        "calibrated_layers": calibrate_layers,
        "total_layers": num_layers,
        "hf_zero_code_changes": True,
        "vllm_zero_code_changes": True,
        "gradient_checkpointing": args.gradient_checkpointing,
        "cpu_offload_activations": args.cpu_offload_activations,
        "output": str(args.output),
        "train_tokens": int(sum(batch["input_ids"].shape[-1] for batch in train_batches)),
        "train_chunks": len(train_batches),
        "held_out_chunks": len(held_out_batches),
        "steps": args.steps,
        "code_init": args.code_init,
        "code_init_tokens": args.code_init_tokens,
        "attention_loss_weight": args.attention_loss_weight,
        "cosine_weight": args.cosine_weight,
        "kl_weight": args.kl_weight,
        "ce_weight": args.ce_weight,
        "margin_weight": args.margin_weight,
        "margin": args.margin,
        "kl_temperature": args.kl_temperature,
        "archive_kernel_features": args.archive_kernel_features,
        "archive_scan_block_size": args.archive_scan_block_size,
        "archive_global_normalization": args.archive_global_normalization,
        "train_mean_logit_cosine": float(train_cosine.item()),
        "train_top1_agreement": float(train_agreement.item()),
    }

    if held_out_cosine is not None:
        result.update({
            "held_out_tokens": int(
                sum(batch["input_ids"].shape[-1] for batch in held_out_batches)
            ),
            "held_out_mean_logit_cosine": float(held_out_cosine.item()),
            "held_out_top1_agreement": float(held_out_agreement.item()),
            "quality_gate": args.quality_gate,
            "held_out_gate_passed": held_out_gate_passed,
        })

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
