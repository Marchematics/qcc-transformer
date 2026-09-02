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
from qcc_transformer import patch_hf_model, retrofit_adapter_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--num-codes", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.lr <= 0 or args.max_tokens <= 0:
        raise ValueError("steps, lr, and max-tokens must be positive")
    text = args.text_file.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("text-file must contain non-whitespace text")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("install qcc-transformer[hf] to run calibration") from exc

    device = torch.device(args.device)
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
    baseline = AutoModelForCausalLM.from_pretrained(args.model, **common).to(device).eval()
    patched = AutoModelForCausalLM.from_pretrained(args.model, **common).to(device)
    replaced = patch_hf_model(
        patched,
        window_size=args.window_size,
        num_codes=args.num_codes,
    )
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
    with torch.no_grad():
        teacher = baseline(**encoded, use_cache=False).logits.float()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    patched.train()
    last_loss = float("nan")
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        student = patched(**encoded, use_cache=False).logits.float()
        loss = torch.nn.functional.mse_loss(student, teacher)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        last_loss = float(loss.detach().item())
    patched.eval()
    with torch.no_grad():
        student = patched(**encoded, use_cache=False).logits.float()
        cosine = torch.nn.functional.cosine_similarity(
            student.reshape(-1, student.shape[-1]),
            teacher.reshape(-1, teacher.shape[-1]),
            dim=-1,
        ).mean()
        agreement = (student.argmax(-1) == teacher.argmax(-1)).float().mean()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": retrofit_adapter_state(patched),
            "base_model": args.model,
            "retrofit": {
                "window_size": args.window_size,
                "num_codes": args.num_codes,
                "patched_layers": replaced,
            },
        },
        args.output,
    )
    print(
        json.dumps(
            {
                "base_model": args.model,
                "output": str(args.output),
                "tokens": int(encoded["input_ids"].shape[-1]),
                "steps": args.steps,
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
