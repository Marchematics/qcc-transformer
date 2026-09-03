"""Calibrate a QCC HF retrofit against an unpatched teacher model.

Only the newly introduced archive and gate parameters are optimized; all
pretrained weights remain frozen.  The output is a small adapter checkpoint
that can be loaded with :func:`qcc_transformer.load_retrofit_adapter`.
Calibration is a necessary setup step, not evidence of RULER/LongBench
quality; always evaluate the saved adapter on held-out real tasks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcc_transformer import patch_hf_model, save_retrofit_adapter
from qcc_transformer.hf_loading import load_hf_causal_lm, model_input_device


def _chunked_mse(student: torch.Tensor, teacher_cpu: torch.Tensor, *, chunk_size: int = 8192) -> torch.Tensor:
    """Compute MSE while keeping teacher targets off the accelerator.

    A 1.5B Qwen vocabulary produces hundreds of MB of logits even for a
    short sequence.  Keeping a second full-vocabulary teacher tensor on a
    24GB GPU causes calibration OOMs, so materialize one vocabulary slice at
    a time and accumulate the scalar loss.
    """
    if student.shape != teacher_cpu.shape:
        raise ValueError(f"student/teacher shape mismatch: {student.shape} vs {teacher_cpu.shape}")
    total = student.new_zeros(())
    vocab = student.shape[-1]
    for start in range(0, vocab, chunk_size):
        end = min(start + chunk_size, vocab)
        target = teacher_cpu[..., start:end].to(device=student.device, dtype=student.dtype)
        total = total + torch.nn.functional.mse_loss(
            student[..., start:end], target, reduction="sum"
        )
    return total / student.numel()


def _mean_cosine_from_cpu(student: torch.Tensor, teacher_cpu: torch.Tensor, *, chunk_size: int = 8192) -> torch.Tensor:
    """Mean per-token cosine without copying the full teacher to GPU."""
    if student.shape != teacher_cpu.shape:
        raise ValueError(f"student/teacher shape mismatch: {student.shape} vs {teacher_cpu.shape}")
    dot = student.new_zeros((student.shape[0], student.shape[1]))
    student_sq = student.new_zeros(dot.shape)
    teacher_sq = student.new_zeros(dot.shape)
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
    """Compute temperature-scaled teacher KL with bounded GPU temporaries."""
    if student.shape != teacher_cpu.shape:
        raise ValueError(f"student/teacher shape mismatch: {student.shape} vs {teacher_cpu.shape}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher = teacher_cpu.float() / temperature
    teacher_log_norm = torch.logsumexp(teacher, dim=-1)
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
        teacher_slice = teacher[..., start:end]
        student_slice = student[..., start:end].float() / temperature
        teacher_log_prob = teacher_slice - teacher_log_norm.unsqueeze(-1)
        student_log_prob = student_slice - student_log_norm.unsqueeze(-1)
        total = total + (
            teacher_log_prob.exp() * (teacher_log_prob - student_log_prob)
        ).sum()
    tokens = max(1, student.numel() // student.shape[-1])
    return total * (temperature * temperature) / tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--num-codes", type=int, default=64)
    parser.add_argument(
        "--archive-position-invariant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use raw (unrotated) Q/K for long-range archive addressing; local attention keeps RoPE",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--load-in-4bit", action="store_true", help="load the real checkpoint through bitsandbytes NF4")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--kv-head-policy", choices=("reject", "repeat"), default="reject")
    parser.add_argument(
        "--gate-bias-init", type=float, default=2.0,
        help="initial local-path bias; use 0.0 for the historical 50/50 ablation",
    )
    parser.add_argument("--run-id", default=None, help="provenance id shared with the gate evidence bundle")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="checkpoint frozen transformer blocks to keep calibration peak memory bounded",
    )
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--kl-temperature", type=float, default=2.0)
    args = parser.parse_args()
    if args.steps <= 0 or args.lr <= 0 or args.max_tokens <= 0:
        raise ValueError("steps, lr, and max-tokens must be positive")
    if args.max_tokens <= args.window_size:
        raise ValueError(
            "max-tokens must exceed window-size so calibration exercises the QCC archive; "
            "otherwise all outputs stay on the exact local path and have no trainable QCC graph"
        )
    if not 0.0 <= args.kl_weight <= 1.0:
        raise ValueError("kl-weight must lie in [0, 1]")
    if args.kl_temperature <= 0:
        raise ValueError("kl-temperature must be positive")
    text = args.text_file.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("text-file must contain non-whitespace text")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("install qcc-transformer[hf] to run calibration") from exc

    device = torch.device(args.device)
    dtype = None if args.dtype == "auto" else {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    common = {"trust_remote_code": args.trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **common)
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=args.max_tokens,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    # Materialize the teacher only long enough to capture logits, then release
    # its 1--7B backbone before loading the trainable retrofit copy.  Keeping
    # both models on a 24GB card doubles the checkpoint footprint and makes a
    # modest 512-token calibration look like an algorithmic OOM.
    baseline = load_hf_causal_lm(
        args.model,
        dtype=dtype,
        device=device,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
    )
    model_device = model_input_device(baseline, device)
    encoded = {key: value.to(model_device) for key, value in encoded.items()}
    with torch.no_grad():
        teacher = baseline(**encoded, use_cache=False).logits.float().cpu()
    del baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()

    patched = load_hf_causal_lm(
        args.model,
        dtype=dtype,
        device=device,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
    )
    patched_device = model_input_device(patched, device)
    patched_encoded = {key: value.to(patched_device) for key, value in encoded.items()}
    replaced = patch_hf_model(
        patched,
        window_size=args.window_size,
        num_codes=args.num_codes,
        archive_position_invariant=args.archive_position_invariant,
        kv_head_policy=args.kv_head_policy,
        gate_bias_init=args.gate_bias_init,
    )
    # The backbone may be fp16/bf16 on a single GPU, but AdamW should not
    # update a trainable gate in half precision.  Keep the tiny adapter gate
    # in fp32; QCCSelfAttention casts it back to the projection dtype for the
    # fused QKV operation, so this does not change the deployed interface.
    for module in patched.modules():
        qcc = getattr(module, "qcc", None)
        if qcc is not None:
            qcc.gate.float()
    if args.gradient_checkpointing and hasattr(patched, "gradient_checkpointing_enable"):
        # Only the small QCC archive/gate is trainable.  Checkpointing the
        # frozen backbone avoids retaining every MLP activation, while input
        # grads keep the custom attention graph connected to the adapter.
        patched.config.use_cache = False
        patched.gradient_checkpointing_enable()
        if hasattr(patched, "enable_input_require_grads"):
            patched.enable_input_require_grads()
    for parameter in patched.parameters():
        parameter.requires_grad = False
    trainable = []
    for module in patched.modules():
        qcc = getattr(module, "qcc", None)
        if qcc is None:
            continue
        for parameter in qcc.archive.parameters():
            parameter.requires_grad = True
            trainable.append(parameter)
        for parameter in (qcc.gate.weight, qcc.gate.bias):
            parameter.requires_grad = True
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("no QCC parameters found after patching")
    # The wrapper itself and its nested modules are both visited by
    # ``Module.modules()`` in some HF architectures.  Optimizers reject
    # duplicate parameters (and would otherwise apply an update twice), so
    # canonicalize the parameter list before constructing AdamW.
    unique_trainable = {id(parameter): parameter for parameter in trainable}
    trainable = list(unique_trainable.values())
    parameter_count = sum(parameter.numel() for parameter in patched.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in unique_trainable.values())
    trainable_parameter_fraction = trainable_parameter_count / max(parameter_count, 1)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    patched.train()
    last_loss = float("nan")
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        student = patched(**patched_encoded, use_cache=False).logits
        mse = _chunked_mse(student, teacher)
        kl = _chunked_kl_divergence(
            student, teacher, temperature=args.kl_temperature
        )
        loss = (1.0 - args.kl_weight) * mse + args.kl_weight * kl
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "calibration diverged; lower --lr or reduce --max-tokens"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        last_loss = float(loss.detach().item())
    patched.eval()
    with torch.no_grad():
        student = patched(**patched_encoded, use_cache=False).logits
        cosine = _mean_cosine_from_cpu(student, teacher)
        agreement = (student.argmax(-1).cpu() == teacher.argmax(-1)).float().mean()
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
            "archive_position_invariant": args.archive_position_invariant,
            "patched_layers": replaced,
            "kv_head_policy": args.kv_head_policy,
            "gate_bias_init": args.gate_bias_init,
        },
    )
    print(
        json.dumps(
            {
                "base_model": args.model,
                "model_id": str(args.model),
                "run_id": args.run_id,
                "parameter_count": parameter_count,
                "trainable_parameter_count": trainable_parameter_count,
                "trainable_parameter_fraction": trainable_parameter_fraction,
                "hf_zero_code_changes": True,
                "vllm_zero_code_changes": True,
                "gradient_checkpointing": args.gradient_checkpointing,
                "output": str(args.output),
                "tokens": int(patched_encoded["input_ids"].shape[-1]),
                "steps": args.steps,
                "kl_weight": args.kl_weight,
                "kl_temperature": args.kl_temperature,
                "final_mse": last_loss,
                "mean_logit_cosine": float(cosine.item()),
                "top1_agreement": float(agreement.item()),
                "note": "Teacher-distillation calibration only; validate on held-out RULER/LongBench tasks.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
