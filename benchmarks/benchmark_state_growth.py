"""How large is the decode state, as a function of prompt length?

The claim under test is that the compiled decode state is *independent* of how
much context was read: ``budget + lex_cap`` slots per (layer, kv-head) whatever
the prompt length, while a Full-KV cache grows linearly and is the term that
actually limits long-context serving.

For each length this records the retained slots, the compiled state in bytes,
the Full-KV state the same prompt would need, and the ratio of the two - plus
the compile time, so the one-off cost of reading a long prompt is visible next
to the recurring decode cost.  The largest length that fits on a 24 GiB card is
limited by the Full-KV prefill (a 128K bf16 Full-KV cache for Llama-3.2-1B is
4 GiB; 1M would be 32 GiB), which is itself part of the result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qcc_transformer.hf_loading import load_hf_causal_lm
from qcc_transformer.retention import RetentionConfig, compile_bounded_cache

DEFAULT_MODEL = os.environ.get("QCC_MODEL", "meta-llama/Llama-3.2-1B-Instruct")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--lengths", nargs="*", type=int,
                        default=[8192, 32768, 65536, 131072, 262144])
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--lex-cap", type=int, default=512)
    parser.add_argument("--prefill-chunk", type=int, default=8192)
    parser.add_argument("--key-chunk", type=int, default=4096)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def kv_bytes(config, tokens, dtype_bytes=2):
    layers = config.num_hidden_layers
    kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    return 2 * layers * kv_heads * tokens * head_dim * dtype_bytes


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_hf_causal_lm(args.model, dtype="bfloat16", device="cuda",
                              load_in_4bit=args.load_4bit).eval()
    config = RetentionConfig(budget=args.budget, lex_cap=args.lex_cap,
                             prefill_chunk=args.prefill_chunk, key_chunk=args.key_chunk)
    # a document-like prompt: repeated prose has no rare strings, so the anchors
    # contribute nothing and the retained width is exactly `budget + lex_cap`
    unit = ("The grass is green. The sky is blue. The sun is yellow. "
            "The sea is deep and the mountains are tall. ")
    ids = torch.tensor([tokenizer(unit, add_special_tokens=False).input_ids], device="cuda")

    rows = []
    for length in args.lengths:
        repeats = length // ids.shape[1] + 1
        prompt = ids.repeat(1, repeats)[:, :length]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        try:
            cache, _logits = compile_bounded_cache(model, prompt, config, tokenizer=tokenizer)
        except torch.OutOfMemoryError as error:  # keep the sweep going, record the wall
            rows.append({"prompt_tokens": length, "status": "oom",
                         "detail": str(error)[:120]})
            print(f"{length:>8d} OOM {str(error)[:60]}", flush=True)
            torch.cuda.empty_cache()
            continue
        torch.cuda.synchronize()
        seconds = time.time() - start
        slots = int(cache.get_seq_length())
        full = kv_bytes(model.config, length)
        bounded = kv_bytes(model.config, slots)
        rows.append({"prompt_tokens": length, "status": "ok", "retained_slots": slots,
                     "slots_per_prompt_token": round(slots / length, 6),
                     "bounded_state_mib": round(bounded / 2**20, 1),
                     "fullkv_state_mib": round(full / 2**20, 1),
                     "fullkv_over_bounded": round(full / bounded, 2),
                     "compile_seconds": round(seconds, 1),
                     "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
        print(f"{length:>8d} slots={slots:>6d} bounded={bounded / 2**20:8.1f} MiB "
              f"full={full / 2**20:9.1f} MiB ratio={full / bounded:6.2f}x "
              f"compile={seconds:5.1f}s peak={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
              flush=True)
        del cache
        torch.cuda.empty_cache()

    ok = [row for row in rows if row["status"] == "ok"]
    summary = {"model": args.model, "budget": args.budget, "lex_cap": args.lex_cap,
               "state_growth_first_to_last": (round(ok[-1]["bounded_state_mib"] /
                                                    ok[0]["bounded_state_mib"], 4)
                                              if len(ok) > 1 else None),
               "slots_growth_first_to_last": (round(ok[-1]["retained_slots"] /
                                                    ok[0]["retained_slots"], 4)
                                              if len(ok) > 1 else None)}
    out = {"config": vars(args), "summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
