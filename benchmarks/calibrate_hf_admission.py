"""Train hybrid exact-tier admission from a real Full-KV HF teacher.

The script first runs the *unpatched* pretrained model and stores layer-input hidden
states for a few calibration chunks. It then patches the same model with QCC + the
hybrid exact tier, reuses the loaded projection weights to reconstruct teacher Q/K/V,
and trains only the tiny per-head admission predictors against sampled future
Full-KV attention salience.

This avoids the non-differentiable hard replacement problem: admission is supervised
explicitly before normal QCC logit distillation. The base model is never duplicated in
GPU memory.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

from qcc_transformer import (
    HFQCCAttention,
    HybridQCCArchive,
    patch_hf_model_hybrid,
    retrofit_adapter_state,
    save_retrofit_adapter,
)
from qcc_transformer.hf_loading import load_hf_causal_lm, model_input_device
from qcc_transformer.admission_training import (
    balanced_admission_loss,
    salience_binary_labels,
    sampled_future_attention_salience,
)


def _dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"calibration file is empty: {path}")
    return text


def _chunk_ids(
    tokenizer,
    text: str,
    *,
    max_tokens: int,
    num_chunks: int,
) -> list[Tensor]:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if encoded.numel() < max_tokens:
        repeats = int(math.ceil(max_tokens / max(1, encoded.numel())))
        encoded = encoded.repeat(repeats)
    max_start = max(0, int(encoded.numel()) - max_tokens)
    if num_chunks == 1:
        starts = [0]
    else:
        starts = (
            torch.linspace(0, max_start, num_chunks).round().to(torch.long).tolist()
        )
    return [encoded[start : start + max_tokens].clone() for start in starts]


def _parse_layers(spec: str, count: int) -> set[int]:
    if spec == "all":
        return set(range(count))
    if spec == "last-half":
        return set(range(count // 2, count))
    if spec == "last-quarter":
        return set(range((3 * count) // 4, count))
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            result.update(range(int(left), int(right) + 1))
        else:
            result.add(int(part))
    if not result or min(result) < 0 or max(result) >= count:
        raise ValueError(f"invalid layer selection {spec!r} for {count} layers")
    return result


@torch.no_grad()
def _collect_hidden_states(
    model,
    chunks: Iterable[Tensor],
    *,
    selected_layers: set[int],
    device: torch.device,
) -> list[dict[int, Tensor]]:
    records: list[dict[int, Tensor]] = []
    for ids in chunks:
        output = model(
            input_ids=ids.unsqueeze(0).to(device),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise RuntimeError("teacher did not return hidden states")
        records.append(
            {
                index: hidden_states[index].detach().cpu()
                for index in selected_layers
            }
        )
        del output, hidden_states
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return records


@torch.no_grad()
def _teacher_examples(
    wrapper: HFQCCAttention,
    hidden_records: list[dict[int, Tensor]],
    *,
    layer_index: int,
    window_size: int,
    num_teacher_queries: int,
    teacher_topk: int,
    positive_fraction: float,
) -> list[tuple[Tensor, Tensor, Tensor]]:
    qcc = wrapper.qcc
    projection = qcc.q_proj
    device = projection.weight.device
    dtype = getattr(projection, "compute_dtype", None)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        dtype = qcc.gate.weight.dtype
    if not dtype.is_floating_point:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    examples: list[tuple[Tensor, Tensor, Tensor]] = []
    for record in hidden_records:
        hidden = record[layer_index].to(device=device, dtype=dtype)
        q, k, v, _ = qcc._project_qkv_gate(hidden)
        qh = qcc._split_heads(q)
        kh = qcc._split_heads(k)
        vh = qcc._split_heads(v)
        positions = torch.arange(
            hidden.shape[1], device=device, dtype=torch.long
        ).view(1, -1)
        rotated_q, rotated_k = qcc._apply_rope(qh, kh, positions)
        # The teacher's actual attention uses rotary Q/K, while the deployed
        # archive may use raw position-invariant Q/K.  A predictor trained in
        # only one coordinate system can discard a token that is salient in the
        # other.  Keep the union by taking the stronger per-position signal.
        rotary_salience = sampled_future_attention_salience(
            rotated_q,
            rotated_k,
            window_size=window_size,
            num_queries=num_teacher_queries,
            topk=teacher_topk,
        )
        if qcc.archive_position_invariant:
            raw_salience = sampled_future_attention_salience(
                qh,
                kh,
                window_size=window_size,
                num_queries=num_teacher_queries,
                topk=teacher_topk,
            )
            salience = torch.maximum(rotary_salience, raw_salience)
        else:
            salience = rotary_salience
        labels = salience_binary_labels(
            salience, positive_fraction=positive_fraction, min_positive=1
        )
        examples.append((kh.detach().cpu(), vh.detach().cpu(), labels.cpu()))
        del hidden, q, k, v, qh, kh, vh, rotated_q, rotated_k, rotary_salience, salience, labels
        if qcc.archive_position_invariant:
            del raw_salience
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return examples


def _binary_metrics(logits: Tensor, labels: Tensor) -> dict[str, float]:
    prediction = logits >= 0
    target = labels >= 0.5
    tp = int((prediction & target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    tn = int((~prediction & ~target).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _train_layer(
    archive: HybridQCCArchive,
    train_examples: list[tuple[Tensor, Tensor, Tensor]],
    held_examples: list[tuple[Tensor, Tensor, Tensor]],
    *,
    steps: int,
    lr: float,
) -> dict[str, object]:
    predictor = archive.admission
    device = archive.codes.device
    for parameter in predictor.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=0.01)
    losses: list[float] = []
    predictor.train()
    for step in range(steps):
        key_cpu, value_cpu, label_cpu = train_examples[step % len(train_examples)]
        key = key_cpu.to(device)
        value = value_cpu.to(device)
        labels = label_cpu.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = predictor(key, value)
        loss = balanced_admission_loss(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    predictor.eval()

    def evaluate(examples: list[tuple[Tensor, Tensor, Tensor]]) -> dict[str, float]:
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for key_cpu, value_cpu, label_cpu in examples:
                all_logits.append(
                    predictor(key_cpu.to(device), value_cpu.to(device)).cpu()
                )
                all_labels.append(label_cpu)
        return _binary_metrics(torch.cat(all_logits, dim=-1), torch.cat(all_labels, dim=-1))

    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "train": evaluate(train_examples),
        "held_out": evaluate(held_examples or train_examples),
    }


def _load_initial_adapter(model: torch.nn.Module, path: Path) -> None:
    """Load a previously calibrated regular QCC adapter into a hybrid model."""

    payload = torch.load(path, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError("initial adapter must contain a state_dict mapping")
    missing, unexpected = model.load_state_dict(state, strict=False)
    required_missing = [
        key
        for key in missing
        if ".qcc.gate." in key
        or key.endswith(".qcc.archive.codes")
        or key.endswith(".qcc.archive.mix_logits")
    ]
    if required_missing or unexpected:
        raise ValueError(
            "initial QCC adapter does not match the hybrid model: "
            f"missing={required_missing}, unexpected={unexpected}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--held-out-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--init-adapter",
        type=Path,
        default=None,
        help="optional regular QCC adapter used to initialize the hybrid base path",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--num-train-chunks", type=int, default=4)
    parser.add_argument("--num-held-chunks", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--num-codes", type=int, default=16)
    parser.add_argument("--exact-num-sets", type=int, default=128)
    parser.add_argument("--exact-ways", type=int, default=4)
    parser.add_argument("--exact-probe-sets", type=int, default=None)
    parser.add_argument("--max-inserts-per-chunk", type=int, default=8)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--teacher-queries", type=int, default=128)
    parser.add_argument("--teacher-topk", type=int, default=8)
    parser.add_argument("--positive-fraction", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--load-in-4bit", action="store_true", help="load the real checkpoint through bitsandbytes NF4")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--kv-head-policy", choices=("reject", "repeat"), default="repeat")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--run-id", default="hf-admission")
    args = parser.parse_args()
    if args.max_tokens <= args.window_size:
        raise ValueError("max-tokens must exceed window-size to produce long-range labels")
    if args.num_train_chunks <= 0 or args.num_held_chunks <= 0 or args.steps <= 0:
        raise ValueError("chunk counts and steps must be positive")
    if not 0.0 < args.positive_fraction <= 1.0:
        raise ValueError("positive-fraction must lie in (0, 1]")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("install the HF extra: pip install -e '.[hf]'") from exc

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    model = load_hf_causal_lm(
        args.model,
        dtype=_dtype(args.dtype),
        device=device,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
    )
    device = model_input_device(model, device)
    model.eval()
    config = model.config
    layer_count = int(
        getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0))
    )
    if layer_count <= 0:
        raise ValueError("could not infer number of decoder layers from config")
    selected_layers = _parse_layers(args.layers, layer_count)

    train_chunks = _chunk_ids(
        tokenizer,
        _read_text(args.train_file),
        max_tokens=args.max_tokens,
        num_chunks=args.num_train_chunks,
    )
    held_path = args.held_out_file or args.train_file
    held_chunks = _chunk_ids(
        tokenizer,
        _read_text(held_path),
        max_tokens=args.max_tokens,
        num_chunks=args.num_held_chunks,
    )

    # Full-KV teacher pass happens before any attention module is replaced.
    train_hidden = _collect_hidden_states(
        model, train_chunks, selected_layers=selected_layers, device=device
    )
    held_hidden = _collect_hidden_states(
        model, held_chunks, selected_layers=selected_layers, device=device
    )

    replaced = patch_hf_model_hybrid(
        model,
        window_size=args.window_size,
        num_codes=args.num_codes,
        kv_head_policy=args.kv_head_policy,
        use_triton=False,
        hybrid_kwargs={
            "exact_num_sets": args.exact_num_sets,
            "exact_ways": args.exact_ways,
            "exact_probe_sets": args.exact_probe_sets,
            "max_inserts_per_chunk": args.max_inserts_per_chunk,
        },
    )
    if args.init_adapter is not None:
        _load_initial_adapter(model, args.init_adapter)
    wrappers = [
        module for module in model.modules() if isinstance(module, HFQCCAttention)
    ]
    if len(wrappers) != layer_count:
        raise ValueError(
            f"patched {len(wrappers)} attention layers but config reports {layer_count}"
        )

    # Freeze everything; only selected admission predictors receive gradients.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    layer_reports: dict[str, object] = {}
    for wrapper in wrappers:
        layer_index = int(wrapper.qcc._qcc_layer_index)
        if layer_index not in selected_layers:
            continue
        archive = wrapper.qcc.archive
        if not isinstance(archive, HybridQCCArchive):
            raise TypeError("hybrid patch did not install HybridQCCArchive")
        train_examples = _teacher_examples(
            wrapper,
            train_hidden,
            layer_index=layer_index,
            window_size=args.window_size,
            num_teacher_queries=args.teacher_queries,
            teacher_topk=args.teacher_topk,
            positive_fraction=args.positive_fraction,
        )
        held_examples = _teacher_examples(
            wrapper,
            held_hidden,
            layer_index=layer_index,
            window_size=args.window_size,
            num_teacher_queries=args.teacher_queries,
            teacher_topk=args.teacher_topk,
            positive_fraction=args.positive_fraction,
        )
        layer_reports[str(layer_index)] = _train_layer(
            archive,
            train_examples,
            held_examples,
            steps=args.steps,
            lr=args.lr,
        )

    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    adapter_state = retrofit_adapter_state(model)
    adapter_parameters = sum(tensor.numel() for tensor in adapter_state.values())
    metadata = {
        "run_id": args.run_id,
        "base_model": args.model,
        "pretrained": True,
        "real_checkpoint": True,
        "calibration": "full_kv_future_attention_salience",
        "window_size": args.window_size,
        "num_codes": args.num_codes,
        "exact_num_sets": args.exact_num_sets,
        "exact_ways": args.exact_ways,
        "exact_probe_sets": args.exact_probe_sets,
        "max_inserts_per_chunk": args.max_inserts_per_chunk,
        "selected_layers": sorted(selected_layers),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / max(1, total),
        "adapter_parameters": adapter_parameters,
        "init_adapter": str(args.init_adapter) if args.init_adapter is not None else None,
        "hf_zero_business_code": True,
    }
    save_retrofit_adapter(model, args.output, **metadata)
    report = {
        **metadata,
        "patched_layers": replaced,
        "layer_reports": layer_reports,
        "output": str(args.output),
    }
    report_path = args.report or args.output.with_suffix(".admission.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
