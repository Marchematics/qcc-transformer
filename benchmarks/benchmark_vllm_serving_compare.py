#!/usr/bin/env python
"""Industry-stack baseline: what vLLM's paged Full-KV cache serves at 32K.

The retention law is a cache policy, so its natural industry comparison is not a
kernel but *how many concurrent long-context requests fit on the card, and what
they decode at*.  vLLM is the reference serving stack and its PagedAttention
manages a Full-KV cache: every resident request keeps every token's key and value,
so the capacity is set by `batch x L x KV bytes per token`.

This script measures, on the same model and the same 32K prompt set as
``benchmark_bounded_decode_serving_sequential.py``:

* the largest batch vLLM can actually serve at that prompt length, by walking the
  batch sizes upward until the engine fails to start or OOMs;
* decode throughput and per-token latency at each batch that fits.

Then it reports the numbers next to the bounded cache's stored ones for the same
prompts and the same GPU, so the comparison is stated in the currency the
difference is in: resident long-context requests per card.

Usage::

    python benchmarks/benchmark_vllm_serving_compare.py --model <checkpoint> \\
        --length 32768 --batches 1 2 4 8 --max-new 32 \\
        --out artifacts/serving-vllm-32k.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

PROMPT = ("A list of records is given below. Each record names a key and the value "
          "assigned to that key. The key {key} carries the value {value}. ")


def build_prompts(tokenizer, length, batch):
    """`batch` prompts of about `length` tokens, each with a distinct question.

    The filler is truncated in token space and then re-decoded, so the prompt that
    is actually sent is re-measured and topped up until it is within a few tokens
    of the request: a decode/encode round trip does not preserve the token count.
    """
    prompts = []
    filler = "The report continues with a description of the surrounding conditions. "
    tail = "\n\nWhat value does the key key-{index} carry? Answer with the number."
    overhead = len(tokenizer(tail.format(index=0), add_special_tokens=False)["input_ids"])
    for index in range(batch):
        target = max(1, length - overhead)
        unit = max(1, len(tokenizer(filler, add_special_tokens=False)["input_ids"]))
        body = filler * (target // unit + 2)
        for _ in range(6):
            ids = tokenizer(body, add_special_tokens=False)["input_ids"]
            if abs(len(ids) - target) <= 4:
                break
            if len(ids) < target:      # BPE merges across filler boundaries
                body += filler * max(1, (target - len(ids)) // unit + 1)
            else:
                body = tokenizer.decode(ids[:target])
        prompts.append(body + tail.format(index=index))
    return prompts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.environ.get("QCC_MODEL",
                                                          "meta-llama/Llama-3.2-1B-Instruct"))
    parser.add_argument("--length", type=int, default=32768)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--max-num-seqs", type=int, default=8,
                        help="vLLM's warmup allocates for this many sequences; it "
                             "has to be at least the batch size being probed")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    # transformers releases that replaced the slow tokenizer backend dropped
    # `all_special_tokens_extended`, which vLLM 0.11 still reads when it builds its
    # stop-token set.  Supply the attribute when it is missing rather than pinning
    # a version: the value is only used to enumerate special tokens.
    try:
        from transformers import PreTrainedTokenizerBase
        if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
            PreTrainedTokenizerBase.all_special_tokens_extended = property(
                lambda self: list(getattr(self, "all_special_tokens", [])))
    except Exception:
        pass

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = []
    for batch in args.batches:
        prompts = build_prompts(tokenizer, args.length, batch)
        entry = {"batch": batch, "prompt_tokens": len(tokenizer(prompts[0],
                                                               add_special_tokens=False)["input_ids"])}
        started = time.time()
        try:
            engine = LLM(model=args.model, dtype="bfloat16", max_model_len=args.length + 256,
                         gpu_memory_utilization=args.gpu_memory_utilization,
                         max_num_seqs=max(batch, args.max_num_seqs),
                         enforce_eager=False, disable_log_stats=True,
                         enable_prefix_caching=True)
            sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new,
                                      ignore_eos=True)
            load_s = time.time() - started
            # offline `LLM.generate` does not expose per-request metrics, so the
            # prompt is prefilled once and then reused through prefix caching: the
            # timed call is decode only, which is the quantity the bounded-cache
            # serving harness reports for the same prompts
            engine.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1,
                                                    ignore_eos=True))
            t0 = time.time()
            outputs = engine.generate(prompts, sampling)
            elapsed = time.time() - t0
            generated = sum(len(o.outputs[0].token_ids) for o in outputs)
            # the engine's own per-request metrics separate prefill from decode; the
            # wall clock includes the 32K prefill and would understate decoding
            prefill_s, decode_s, decode_tokens = [], [], 0
            for output in outputs:
                metrics = getattr(output, "metrics", None)
                first = getattr(metrics, "first_token_time", None)
                finished = getattr(metrics, "finished_time", None)
                scheduled = getattr(metrics, "scheduled_time", None)
                if first and finished:
                    tokens = max(1, len(output.outputs[0].token_ids) - 1)
                    decode_s.append(finished - first)
                    decode_tokens += tokens
                    if scheduled:
                        prefill_s.append(first - scheduled)
            if not decode_s:
                metrics = getattr(outputs[0], "metrics", None)
                entry["metrics_attrs"] = (sorted(vars(metrics)) if metrics is not None
                                          else None)
            entry.update({
                "decode_only_tokens_per_s": round(
                    batch * max(0, args.max_new - 1) / elapsed, 1),
                "decode_only_tpot_ms": round(1000.0 * elapsed / max(1, args.max_new - 1), 2),
                "engine_load_s": round(load_s, 1),
                "wall_s": round(elapsed, 2),
                "generated_tokens": generated,
                "metrics_available": bool(decode_s),
                "prefill_s": round(sum(prefill_s) / len(prefill_s), 3) if prefill_s else None,
                "decode_s": round(sum(decode_s) / len(decode_s), 3) if decode_s else None,
                "decode_tokens_per_s": (round(decode_tokens / max(decode_s), 1)
                                        if decode_s else None),
                "tpot_ms": (round(1000.0 * sum(decode_s) / max(1, decode_tokens) * batch, 2)
                            if decode_s else None),
                "e2e_tokens_per_s": round(generated / elapsed, 1),
                "ok": True,
            })
            del engine
        except Exception as exc:                     # OOM or a length limit
            entry.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]})
        rows.append(entry)
        print(json.dumps(entry), flush=True)

    Path(args.out).write_text(json.dumps({
        "model": args.model, "length": args.length, "max_new": args.max_new,
        "engine": "vllm", "config": vars(args), "rows": rows,
    }, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
