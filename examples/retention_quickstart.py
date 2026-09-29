"""Compile a bounded decode cache and generate — CPU, tiny random model, no downloads.

    python examples/retention_quickstart.py

The same three lines work on a real checkpoint; only the model construction
changes.  `compile_bounded_cache` runs an exact chunked prefill and then keeps
`budget + lex_cap` slots per (layer, kv-head), so the decode state does not grow
with the prompt.
"""
from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from qcc_transformer import RetentionConfig, compile_bounded_cache


def main() -> None:
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
    )).eval()

    prompt = torch.randint(1, 256, (1, 512))
    config = RetentionConfig(budget=64, lex_cap=16, observation_window=16,
                             prefill_chunk=128, key_chunk=128)

    cache, logits = compile_bounded_cache(model, prompt, config)
    print(f"prompt {prompt.shape[1]} tokens -> {cache.get_seq_length()} retained slots "
          f"({RetentionConfig(budget=64, lex_cap=16).target_slots} configured)")

    # decode exactly as against any other Hugging Face cache
    token = logits[:, -1:].argmax(-1)
    with torch.no_grad():
        for step in range(3):
            past = cache.get_seq_length()
            position = torch.tensor([[prompt.shape[1] + step]])
            out = model(token, past_key_values=cache, position_ids=position,
                        cache_position=torch.arange(past, past + 1), use_cache=True)
            token = out.logits[:, -1:].argmax(-1)
    print("generated token ids:", token.flatten().tolist())


if __name__ == "__main__":
    main()
