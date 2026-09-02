"""Procedural million-token stress for the exact associative tier.

This benchmark is intentionally *synthetic* and therefore can never satisfy the final
real-LM gate. Its purpose is narrower and falsifiable: prevent a return to the old
single-needle demonstration by exercising random 1M depths, multiple needles,
semantic near-key distractors, and at least 1000 trials.

The default ``oracle`` admission labels needles as salient and distractors as filler.
That isolates capacity/routing from the separate learned-admission problem. The final
``gate_99.py`` rejects this output because it is marked ``synthetic=true`` and
``real_model=false``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from qcc_transformer import SetAssociativeLandmarkBank


def _run_batch(
    *,
    batch_size: int,
    seed: int,
    context_tokens: int,
    num_needles: int,
    semantic_per_needle: int,
    random_distractors: int,
    head_dim: int,
    num_sets: int,
    ways: int,
    query_noise: float,
    semantic_noise: float,
) -> dict[str, float | int]:
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed + 10_000)
    needles = F.normalize(
        torch.randn(batch_size, num_needles, head_dim, generator=generator), dim=-1
    )
    needle_values = torch.randn(
        batch_size, num_needles, head_dim, generator=generator
    )
    target = torch.randint(0, num_needles, (batch_size,), generator=generator)

    total_events = (
        num_needles + num_needles * semantic_per_needle + random_distractors
    )
    positions = torch.randint(
        1,
        context_tokens - 1,
        (batch_size, total_events),
        generator=generator,
    )

    keys = [needles]
    values = [needle_values]
    admissions = [torch.full((batch_size, num_needles), 100.0)]
    flags = [
        torch.arange(num_needles).view(1, -1) == target.view(batch_size, 1)
    ]

    semantic = F.normalize(
        needles[:, :, None, :]
        + semantic_noise
        * torch.randn(
            batch_size,
            num_needles,
            semantic_per_needle,
            head_dim,
            generator=generator,
        ),
        dim=-1,
    ).reshape(batch_size, num_needles * semantic_per_needle, head_dim)
    keys.append(semantic)
    values.append(
        torch.randn(
            batch_size,
            num_needles * semantic_per_needle,
            head_dim,
            generator=generator,
        )
    )
    admissions.append(
        torch.full((batch_size, num_needles * semantic_per_needle), -100.0)
    )
    flags.append(
        torch.zeros(
            batch_size, num_needles * semantic_per_needle, dtype=torch.bool
        )
    )

    random_keys = F.normalize(
        torch.randn(
            batch_size, random_distractors, head_dim, generator=generator
        ),
        dim=-1,
    )
    keys.append(random_keys)
    values.append(
        torch.randn(
            batch_size, random_distractors, head_dim, generator=generator
        )
    )
    admissions.append(torch.full((batch_size, random_distractors), -100.0))
    flags.append(torch.zeros(batch_size, random_distractors, dtype=torch.bool))

    keys_tensor = torch.cat(keys, dim=1)
    values_tensor = torch.cat(values, dim=1)
    admission_tensor = torch.cat(admissions, dim=1)
    flag_tensor = torch.cat(flags, dim=1)

    order = positions.argsort(dim=1)
    gather = order[:, :, None].expand(-1, -1, head_dim)
    keys_tensor = keys_tensor.gather(1, gather)
    values_tensor = values_tensor.gather(1, gather)
    admission_tensor = admission_tensor.gather(1, order)
    flag_tensor = flag_tensor.gather(1, order)
    sorted_positions = positions.gather(1, order)

    bank = SetAssociativeLandmarkBank(
        num_heads=1,
        head_dim=head_dim,
        num_sets=num_sets,
        ways=ways,
        probe_sets=num_sets,
        diversity_weight=0.1,
    )
    # The benchmark isolates the external admission signal. The bank's own
    # random linear score would otherwise add irrelevant noise to an oracle test.
    with torch.no_grad():
        bank.admission_vector.zero_()
    bank.reset_state(batch_size)
    for index in range(total_events):
        score = admission_tensor[:, index].view(batch_size, 1)
        bank.update(
            keys_tensor[:, index].unsqueeze(1),
            values_tensor[:, index].unsqueeze(1),
            admission_bias=score,
            write_mask=score > 0,
        )

    batch_index = torch.arange(batch_size)
    target_key = needles[batch_index, target]
    target_value = needle_values[batch_index, target]
    query = F.normalize(
        target_key
        + query_noise
        * torch.randn(batch_size, head_dim, generator=generator),
        dim=-1,
    )

    full_similarity = torch.einsum("bpd,bd->bp", keys_tensor, query)
    full_best = full_similarity.argmax(dim=1)
    full_hit = flag_tensor[batch_index, full_best]

    response, _ = bank.read(query.unsqueeze(1), hard=True)
    qcc_hit = torch.isclose(
        response[:, 0], target_value, rtol=0, atol=0
    ).all(dim=-1)

    target_event_index = flag_tensor.to(torch.int64).argmax(dim=1)
    target_depth = (
        sorted_positions[batch_index, target_event_index].float() / context_tokens
    )
    return {
        "qcc_correct": int(qcc_hit.sum()),
        "full_correct": int(full_hit.sum()),
        "catastrophic": int((full_hit & ~qcc_hit).sum()),
        "min_depth": float(target_depth.min()),
        "max_depth": float(target_depth.max()),
        "state_bytes": int(bank.state_bytes()),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    qcc_correct = 0
    full_correct = 0
    catastrophic = 0
    min_depth = 1.0
    max_depth = 0.0
    state_bytes: int | None = None
    started = time.perf_counter()
    for offset in range(0, args.trials, args.batch_size):
        batch_size = min(args.batch_size, args.trials - offset)
        result = _run_batch(
            batch_size=batch_size,
            seed=args.seed + offset,
            context_tokens=args.context_tokens,
            num_needles=args.num_needles,
            semantic_per_needle=args.semantic_per_needle,
            random_distractors=args.random_distractors,
            head_dim=args.head_dim,
            num_sets=args.num_sets,
            ways=args.ways,
            query_noise=args.query_noise,
            semantic_noise=args.semantic_noise,
        )
        qcc_correct += int(result["qcc_correct"])
        full_correct += int(result["full_correct"])
        catastrophic += int(result["catastrophic"])
        min_depth = min(min_depth, float(result["min_depth"]))
        max_depth = max(max_depth, float(result["max_depth"]))
        state_bytes = int(result["state_bytes"])

    qcc_rate = qcc_correct / args.trials
    full_rate = full_correct / args.trials
    ratio = qcc_rate / full_rate if full_rate else 0.0
    catastrophic_rate = catastrophic / args.trials
    return {
        "benchmark": "associative_1m_procedural",
        "context_tokens": args.context_tokens,
        "trials": args.trials,
        "qcc_success_rate": qcc_rate,
        "full_kv_success_rate": full_rate,
        "qcc_full_kv_ratio": ratio,
        "catastrophic_retrieval_miss_rate": catastrophic_rate,
        "random_depth": True,
        "multi_needle": args.num_needles >= 2,
        "semantic_distractor": args.semantic_per_needle > 0,
        "min_target_depth": min_depth,
        "max_target_depth": max_depth,
        "num_needles": args.num_needles,
        "semantic_distractors_per_needle": args.semantic_per_needle,
        "random_distractors": args.random_distractors,
        "state_bytes": state_bytes,
        "synthetic": True,
        "real_model": False,
        "official": False,
        "matched_full_kv": True,
        "oracle_admission": True,
        "elapsed_seconds": time.perf_counter() - started,
        "passed_development_gate": (
            args.trials >= 1000
            and qcc_rate >= args.require_rate
            and ratio >= args.require_ratio
            and catastrophic_rate < 0.01
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--context-tokens", type=int, default=1_000_000)
    parser.add_argument("--num-needles", type=int, default=8)
    parser.add_argument("--semantic-per-needle", type=int, default=8)
    parser.add_argument("--random-distractors", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--num-sets", type=int, default=32)
    parser.add_argument("--ways", type=int, default=4)
    parser.add_argument("--query-noise", type=float, default=0.02)
    parser.add_argument("--semantic-noise", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--require-rate", type=float, default=0.99)
    parser.add_argument("--require-ratio", type=float, default=0.99)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.trials < 1000:
        raise ValueError("strict development stress requires at least 1000 trials")
    if args.context_tokens < 1_000_000:
        raise ValueError("strict development stress requires at least 1M context")
    if args.num_needles < 2:
        raise ValueError("strict development stress requires multiple needles")
    if args.semantic_per_needle <= 0:
        raise ValueError("strict development stress requires semantic distractors")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not result["passed_development_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
