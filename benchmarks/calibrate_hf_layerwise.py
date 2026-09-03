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
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qcc_transformer import patch_hf_model, save_retrofit_adapter
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


def _distillation_loss(
    student: torch.Tensor,
    teacher_cpu: torch.Tensor,
    *,
    chunk_size: int = 8192,
    cosine_weight: float = 0.0,
    kl_weight: float = 0.0,
    kl_temperature: float = 2.0,
) -> torch.Tensor:
    """Numerically stable logit distillation objective.

    The original calibration used MSE only, which over-weights high-magnitude
    vocabulary coordinates and can yield a good train loss but poor held-out
    ranking fidelity.  A bounded cosine term directly optimizes directional
    agreement while retaining the MSE scale.  ``cosine_weight=0`` is exactly
    the historical objective for reproducibility.
    """
    if not 0.0 <= cosine_weight <= 1.0 or not 0.0 <= kl_weight <= 1.0:
        raise ValueError("cosine_weight and kl_weight must lie in [0, 1]")
    if cosine_weight + kl_weight > 1.0:
        raise ValueError("cosine_weight + kl_weight must not exceed 1")
    mse = _chunked_mse(student, teacher_cpu, chunk_size=chunk_size)
    if cosine_weight == 0.0 and kl_weight == 0.0:
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
    return (1.0 - cosine_weight - kl_weight) * mse + cosine_weight * (1.0 - cosine) + kl_weight * kl


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
        default="last-half",
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
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--num-train-chunks", type=int, default=1,
        help="number of sequential training chunks to distill (cycles across chunks each step)",
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
    parser.add_argument("--quality-gate", type=float, default=0.99,
                        help="minimum held-out cosine for fidelity gate")
    parser.add_argument(
        "--cosine-weight", type=float, default=0.0,
        help="weight of directional (1-cosine) term in calibration loss; "
             "0 preserves historical MSE-only behavior",
    )
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--kl-temperature", type=float, default=2.0)
    args = parser.parse_args()

    if args.steps <= 0 or args.lr <= 0 or args.max_tokens <= 0 or args.num_train_chunks <= 0:
        raise ValueError("steps, lr, max-tokens, and num-train-chunks must be positive")
    if not 0.0 <= args.cosine_weight <= 1.0 or not 0.0 <= args.kl_weight <= 1.0:
        raise ValueError("cosine-weight and kl-weight must lie in [0, 1]")
    if args.cosine_weight + args.kl_weight > 1.0:
        raise ValueError("cosine-weight + kl-weight must not exceed 1")
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

    def encode_text(text: str, *, max_length: int | None = None) -> dict:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=max_length or args.max_tokens,
        )
        return {key: value.to(device) for key, value in encoded.items()}

    if args.num_train_chunks == 1:
        train_batches = [encode_text(train_text)]
    else:
        full = tokenizer(train_text, return_tensors="pt", add_special_tokens=True, truncation=True,
                         max_length=args.max_tokens * args.num_train_chunks)
        ids = full["input_ids"][0]
        mask = full.get("attention_mask")
        train_batches = []
        for start in range(0, ids.numel(), args.max_tokens):
            chunk_ids = ids[start : start + args.max_tokens]
            if chunk_ids.numel() <= args.window_size:
                continue
            batch = {"input_ids": chunk_ids.unsqueeze(0)}
            if mask is not None:
                batch["attention_mask"] = mask[0, start : start + chunk_ids.numel()].unsqueeze(0)
            train_batches.append({key: value.to(device) for key, value in batch.items()})
        if not train_batches:
            raise ValueError("train-file did not produce any chunks longer than window-size")
    held_out_encoded = encode_text(held_out_text) if held_out_text else None

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
    train_batches = [{key: value.to(device) for key, value in batch.items()} for batch in train_batches]
    if held_out_encoded is not None:
        held_out_encoded = {key: value.to(device) for key, value in held_out_encoded.items()}

    with torch.no_grad():
        train_teachers = [baseline(**batch, use_cache=False).logits.float().cpu() for batch in train_batches]
        held_out_teacher = None
        if held_out_encoded:
            held_out_teacher = baseline(**held_out_encoded, use_cache=False).logits.float().cpu()

    del baseline
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
    train_batches = [{key: value.to(device) for key, value in batch.items()} for batch in train_batches]
    if held_out_encoded is not None:
        held_out_encoded = {key: value.to(device) for key, value in held_out_encoded.items()}
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
        kv_head_policy=args.kv_head_policy,
        gate_bias_init=args.gate_bias_init,
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

    # Determine which layers to calibrate
    num_layers = len(replaced)
    calibrate_layers = parse_layer_spec(args.calibrate_layers, num_layers)
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

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        batch_index = step % len(train_batches)
        student = patched(**train_batches[batch_index], use_cache=False).logits
        loss = _distillation_loss(
            student,
            train_teachers[batch_index],
            cosine_weight=args.cosine_weight,
            kl_weight=args.kl_weight,
            kl_temperature=args.kl_temperature,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "calibration diverged; lower --lr or reduce --max-tokens"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step {step+1}/{args.steps}: loss={loss.item():.6f}", file=sys.stderr)

    # Evaluate
    patched.eval()
    with torch.no_grad():
        train_cosines = []
        train_agreements = []
        for batch, teacher in zip(train_batches, train_teachers):
            train_student = patched(**batch, use_cache=False).logits
            train_cosines.append(_mean_cosine_from_cpu(train_student, teacher))
            train_agreements.append((train_student.argmax(-1).cpu() == teacher.argmax(-1)).float().mean())
        train_cosine = torch.stack(train_cosines).mean()
        train_agreement = torch.stack(train_agreements).mean()

        held_out_cosine = None
        held_out_agreement = None
        held_out_gate_passed = None

        if held_out_teacher is not None:
            held_out_student = patched(**held_out_encoded, use_cache=False).logits
            held_out_cosine = _mean_cosine_from_cpu(held_out_student, held_out_teacher)
            held_out_agreement = (held_out_student.argmax(-1).cpu() == held_out_teacher.argmax(-1)).float().mean()
            held_out_gate_passed = float(held_out_cosine.item()) >= args.quality_gate

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
            "calibrated_layers": calibrate_layers,
            "kv_head_policy": args.kv_head_policy,
            "gate_bias_init": args.gate_bias_init,
            "num_train_chunks": len(train_batches),
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
        "output": str(args.output),
        "train_tokens": int(sum(batch["input_ids"].shape[-1] for batch in train_batches)),
        "train_chunks": len(train_batches),
        "steps": args.steps,
        "cosine_weight": args.cosine_weight,
        "kl_weight": args.kl_weight,
        "kl_temperature": args.kl_temperature,
        "train_mean_logit_cosine": float(train_cosine.item()),
        "train_top1_agreement": float(train_agreement.item()),
    }

    if held_out_cosine is not None:
        result.update({
            "held_out_tokens": int(held_out_encoded["input_ids"].shape[-1]),
            "held_out_mean_logit_cosine": float(held_out_cosine.item()),
            "held_out_top1_agreement": float(held_out_agreement.item()),
            "quality_gate": args.quality_gate,
            "held_out_gate_passed": held_out_gate_passed,
        })

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
