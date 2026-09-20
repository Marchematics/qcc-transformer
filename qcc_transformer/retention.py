"""Query-compiled retention: a bounded exact-KV decode cache.

This module is the packaged form of the retention law measured in
``benchmarks/benchmark_bounded_decode_frontier.py`` and documented in
``artifacts/bounded-decode-frontier-20260920.md``.  It is a *cache policy*, not
an attention approximation: prefill runs the model's own exact causal attention,
the cache is then compiled down to a fixed number of slots per
``(layer, kv-head)``, and decode computes exact softmax attention over whatever
survived.  No parameter is added and the pretrained checkpoint is untouched.

Two selection signals are combined:

* an **observation window** — the attention that the last ``observation_window``
  prompt tokens (for a question-answering prompt, the question itself) pay to
  every key, which is causal because those tokens are the end of the prompt;
* **lexical anchors** — the token spans of the question's rare strings (keys,
  UUIDs, long numbers) wherever they appear earlier in the prompt, optionally
  following assignment chains for value-tracking tasks.

The retained width is uniform by construction (``budget + lex_cap`` slots for
every head, layer and request) so that a single 2-D attention mask can describe
a batch of them.

Usage::

    from qcc_transformer.retention import RetentionConfig, compile_bounded_cache
    config = RetentionConfig(budget=4096, lex_cap=512)
    cache, logits = compile_bounded_cache(model, input_ids, config)
    # decode against `cache` exactly as against any other HF cache
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

__all__ = [
    "RetentionConfig",
    "forward_accepts",
    "observation_scores",
    "lexical_anchors",
    "select_indices",
    "prune_cache",
    "compile_bounded_cache",
]


@dataclass
class RetentionConfig:
    """Retention budget and selection settings."""

    budget: int = 4096
    observation_window: int = 64
    attention_sinks: int = 4
    recent_fraction: float = 0.25
    pool: int = 7
    dilate: int = 9
    lex_cap: int = 512
    lex_context_left: int = 16
    lex_context_right: int = 32
    chain_hops: int = 0
    min_word_len: int = 6
    min_number_len: int = 4
    max_occurrences: int = 8
    prefill_chunk: int = 8192
    key_chunk: int = 4096
    scoring: str = "last"          # "last" | "mean" | "max"
    _forward_args: dict = field(default_factory=dict, repr=False)

    @property
    def target_slots(self) -> int:
        return self.budget + self.lex_cap

    @property
    def recent_window(self) -> int:
        return max(1, int(self.budget * self.recent_fraction))


_FORWARD_ARGS: dict[int, set[str]] = {}


def forward_accepts(model, name: str) -> bool:
    """Whether ``model.forward`` advertises ``name``.

    Older remote-code checkpoints (Phi-3/Phi-4) do not accept ``cache_position``
    and reject it rather than ignoring it.
    """
    key = id(model)
    if key not in _FORWARD_ARGS:
        try:
            _FORWARD_ARGS[key] = set(inspect.signature(model.forward).parameters)
        except (TypeError, ValueError):
            _FORWARD_ARGS[key] = set()
    return name in _FORWARD_ARGS[key]


def _model_kwargs(model, **kwargs):
    return {k: v for k, v in kwargs.items()
            if v is not None and forward_accepts(model, k)}


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


@torch.no_grad()
def prefill_capture(model, input_ids: torch.Tensor, observation_window: int,
                    chunk_size: int):
    """Exact chunked prefill, capturing only the observation-window inputs.

    Feeding the prompt in fixed chunks with a growing KV cache and absolute
    position ids is algebraically the same causal prefill as one long forward
    pass, but activation memory stays ``O(chunk)`` and only ``O(obs * d)`` hidden
    state per layer is retained.
    """
    from transformers import DynamicCache

    tails: dict[int, torch.Tensor] = {}
    handles = []
    for i, layer in enumerate(model.model.layers):
        def make(idx):
            def hook(module, args, kwargs, output):
                h = args[0] if args else kwargs.get("hidden_states")
                h = h.detach()
                prev = tails.get(idx)
                tails[idx] = (h if prev is None
                              else torch.cat([prev, h], dim=1)[:, -observation_window:, :])
            return hook
        handles.append(layer.self_attn.register_forward_hook(make(i), with_kwargs=True))

    total = input_ids.shape[1]
    cache = DynamicCache()
    last_logits = None
    try:
        for start in range(0, total, chunk_size):
            end = min(total, start + chunk_size)
            segment = input_ids[:, start:end]
            positions = torch.arange(start, end, device=input_ids.device)
            mask = torch.ones(1, end, device=input_ids.device, dtype=torch.long)
            out = model(segment, **_model_kwargs(
                model, past_key_values=cache, attention_mask=mask,
                position_ids=positions.unsqueeze(0), cache_position=positions,
                use_cache=True))
            last_logits = out.logits[:, -1:]
    finally:
        for handle in handles:
            handle.remove()
    return cache, last_logits, tails


@torch.no_grad()
def observation_scores(model, captured: dict[int, torch.Tensor], cache,
                       length: int, config: RetentionConfig) -> list[torch.Tensor]:
    """Per ``(layer, kv-head)`` key importance from the observation window."""
    num_heads = model.config.num_attention_heads
    kv_heads = cache.layers[0].keys.shape[1]
    group = num_heads // kv_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    key_positions = torch.arange(length, device=cache.layers[0].keys.device)
    neg = -1e30
    scores = []
    for layer_idx, layer in enumerate(cache.layers):
        hidden = captured[layer_idx]
        n_obs = hidden.shape[1]
        attn = model.model.layers[layer_idx].self_attn
        query = attn.q_proj(hidden[0]).view(n_obs, num_heads, head_dim).transpose(0, 1)
        positions = torch.arange(length - n_obs, length, device=hidden.device)
        cos, sin = model.model.rotary_emb(hidden, positions.unsqueeze(0))
        query = _apply_rope(query, cos[0].unsqueeze(0), sin[0].unsqueeze(0))
        keys = layer.keys[0]

        if config.scoring == "last":
            # for a fixed query the softmax denominator is constant across keys,
            # so ranking by the raw final-query score equals ranking by weight
            parts = []
            final_query = query[:, -1:, :].reshape(kv_heads, group, head_dim)
            for start in range(0, length, config.key_chunk):
                end = min(length, start + config.key_chunk)
                logits = torch.einsum("god,hld->hgol", final_query.float(),
                                      keys[:, start:end].float()) * attn.scaling
                valid = key_positions[start:end][None, :] <= positions[-1]
                logits = logits.masked_fill(~valid[None, None, :, :], neg).amax(dim=1)
                parts.append(logits[:, 0, :])
            scores.append(torch.cat(parts, dim=1))
            continue

        qf = query.reshape(kv_heads, group, n_obs, head_dim)
        running_max = torch.full((kv_heads, group, n_obs), neg, device=hidden.device)
        denominator = torch.zeros_like(running_max)
        for start in range(0, length, config.key_chunk):
            end = min(length, start + config.key_chunk)
            logits = torch.einsum("hgod,hld->hgol", qf.float(), keys[:, start:end].float()) * attn.scaling
            valid = key_positions[start:end][None, :] <= positions[:, None]
            logits = logits.masked_fill(~valid[None, None, :, :], neg)
            block_max = logits.amax(dim=-1)
            new_max = torch.maximum(running_max, block_max)
            probs = (logits - new_max[..., None]).masked_fill(
                ~valid[None, None, :, :], float("-inf"))
            denominator = denominator * torch.exp(running_max - new_max) + torch.exp(probs).sum(dim=-1)
            running_max = new_max
        acc = torch.empty((kv_heads, length), device=hidden.device)
        for start in range(0, length, config.key_chunk):
            end = min(length, start + config.key_chunk)
            logits = torch.einsum("hgod,hld->hgol", qf.float(), keys[:, start:end].float()) * attn.scaling
            valid = key_positions[start:end][None, :] <= positions[:, None]
            logits = logits.masked_fill(~valid[None, None, :, :], neg)
            weights = torch.exp(logits - running_max[..., None]) / denominator[..., None]
            weights = weights.masked_fill(~valid[None, None, :, :], 0.0)
            acc[:, start:end] = (weights.mean(dim=(1, 2)) if config.scoring == "mean"
                                 else weights.amax(dim=(1, 2)))
        scores.append(acc)
    return scores


def lexical_anchors(tokenizer, prompt: str, config: RetentionConfig) -> list[int]:
    """Token positions of the question's rare strings earlier in the prompt.

    Hyphenated identifiers, UUIDs and long numbers are matched verbatim; matches
    are expanded asymmetrically (the answer usually follows the key) and, when
    ``chain_hops > 0``, assignment chains are followed by walking identifiers
    that are themselves defined by an ``=`` somewhere in the context.
    """
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.offset_mapping
    total = len(offsets)
    question_start_token = max(0, total - config.observation_window)
    question_char_start = offsets[question_start_token][0] if question_start_token < total else len(prompt)
    question = prompt[question_char_start:]

    candidates: set[str] = set()
    for match in re.finditer(r"[A-Za-z0-9][A-Za-z0-9_\-]{%d,}" % (config.min_word_len - 1),
                             question):
        candidates.add(match.group(0))
    for match in re.finditer(r"\d{%d,}" % config.min_number_len, question):
        candidates.add(match.group(0))
    for match in re.finditer(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                             r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", question):
        candidates.add(match.group(0))

    rare: list[tuple[str, list[int]]] = []
    for word in sorted(candidates, key=len, reverse=True):
        occurrences, start = [], 0
        while len(occurrences) <= config.max_occurrences:
            found = prompt.find(word, start, question_char_start)
            if found < 0:
                break
            occurrences.append(found)
            start = found + 1
        if 0 < len(occurrences) <= config.max_occurrences:
            rare.append((word, occurrences))

    hits: set[int] = set()
    for word, occurrences in rare:
        for position in occurrences:
            if len(hits) >= config.lex_cap:
                break
            for token, (a, b) in enumerate(offsets):
                if b > position and a < position + len(word):
                    hits.add(token)

    chain_hits: set[int] = set()
    if config.chain_hops > 0:
        symbol = r"[A-Za-z][A-Za-z0-9_\-]{1,}"
        defined = set(re.findall(r"(" + symbol + r")\s*=", prompt[:question_char_start]))
        frontier: set[str] = set()
        for _word, occurrences in rare:
            for position in occurrences:
                line_start = prompt.rfind("\n", 0, position) + 1
                line_end = prompt.find("\n", position)
                frontier |= set(re.findall(symbol, prompt[line_start: line_end if line_end >= 0 else len(prompt)]))
        frontier &= defined
        seen: set[str] = set()
        for _hop in range(config.chain_hops):
            if not frontier or len(hits) >= config.lex_cap:
                break
            following: set[str] = set()
            for name in sorted(frontier):
                start = 0
                while len(hits) < config.lex_cap:
                    found = prompt.find(name, start, question_char_start)
                    if found < 0:
                        break
                    start = found + 1
                    line_start = prompt.rfind("\n", 0, found) + 1
                    line_end = prompt.find("\n", found)
                    line_end = line_end if line_end >= 0 else len(prompt)
                    for token, (a, b) in enumerate(offsets):
                        if b > line_start and a < line_end:
                            chain_hits.add(token)
                    following |= set(re.findall(symbol, prompt[line_start:line_end]))
            seen |= frontier
            frontier = (following & defined) - seen

    expanded: set[int] = set()
    for token in sorted(hits):
        expanded.update(range(max(0, token - config.lex_context_left),
                              min(total, token + config.lex_context_right + 1)))
        if len(expanded) >= config.lex_cap:
            break
    expanded |= chain_hits
    return sorted(expanded)[: config.lex_cap]


def _forced_positions(length: int, config: RetentionConfig) -> list[int]:
    return sorted(set(list(range(min(config.attention_sinks, length)))
                      + list(range(max(0, length - config.recent_window), length))))


def _transform(scores: torch.Tensor, length: int, config: RetentionConfig) -> torch.Tensor:
    """Sliding-window dilation, then block pooling; sinks and recent forced."""
    out = []
    for head in range(scores.shape[0]):
        s = scores[head].clone().float()
        if config.dilate > 1:
            k = config.dilate if config.dilate % 2 == 1 else config.dilate + 1
            pad = k // 2
            padded = F.pad(s.view(1, 1, -1), (pad, pad), value=float("-inf"))
            s = F.max_pool1d(padded, kernel_size=k, stride=1).view(-1)[:length]
        if config.pool > 1:
            pad = (-length) % config.pool
            if pad:
                s = torch.cat([s, torch.full((pad,), float("-inf"), device=s.device)])
            blocks = s.numel() // config.pool
            s = s.view(blocks, config.pool).amax(dim=1, keepdim=True).expand(blocks, config.pool).reshape(-1)[:length]
        forced = _forced_positions(length, config)
        if forced:
            s[torch.tensor(forced, device=s.device)] = float("inf")
        out.append(s)
    return torch.stack(out, dim=0)


def select_indices(scores: list[torch.Tensor], cache, length: int,
                   config: RetentionConfig, anchors: list[int] | None = None) -> list[torch.Tensor]:
    """Retained slot indices per layer, uniform width for every head."""
    target = min(config.target_slots, length)
    anchor_list = [p for p in (anchors or []) if p < length][:target]
    anchor_set = set(anchor_list)
    selected = []
    for layer_scores in scores:
        transformed = _transform(layer_scores, length, config)
        top = torch.topk(transformed, target, dim=-1).indices
        rows = []
        for head in range(transformed.shape[0]):
            chosen = list(anchor_list)
            if len(chosen) < target:
                for position in top[head].tolist():
                    if len(chosen) >= target:
                        break
                    if position not in anchor_set:
                        chosen.append(position)
            rows.append(sorted(chosen[:target]))
        selected.append(torch.tensor(rows, device=transformed.device, dtype=torch.long))
    return selected


@torch.no_grad()
def prune_cache(cache, indices: list[torch.Tensor]) -> None:
    """Keep only ``indices`` in place, preserving each key's rotary phase."""
    for layer, index in zip(cache.layers, indices):
        heads = layer.keys.shape[1]
        gather_index = index.unsqueeze(0).unsqueeze(-1).expand(
            1, heads, index.shape[1], layer.keys.shape[-1])
        layer.keys = layer.keys.gather(2, gather_index).contiguous()
        layer.values = layer.values.gather(2, gather_index).contiguous()


@torch.no_grad()
def _compile_one(model, input_ids: torch.Tensor, config: RetentionConfig,
                 tokenizer=None):
    """Compile a single unpadded request; ``(cache, last_logits)``."""
    length = input_ids.shape[1]
    cache, last_logits, captured = prefill_capture(
        model, input_ids, config.observation_window, config.prefill_chunk)
    scores = observation_scores(model, captured, cache, length, config)
    anchors = None
    if config.lex_cap > 0 and tokenizer is not None:
        prompt = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
        anchors = lexical_anchors(tokenizer, prompt, config)
    indices = select_indices(scores, cache, length, config, anchors)
    prune_cache(cache, indices)
    return cache, last_logits


@torch.no_grad()
def compile_bounded_cache(model, input_ids: torch.Tensor, config: RetentionConfig,
                          tokenizer=None, attention_mask: torch.Tensor | None = None):
    """Exact prefill, then compile the decode cache down to ``budget`` slots.

    ``input_ids`` is a single sequence ``(1, L)`` or a batch ``(B, L)``.
    ``attention_mask`` marks each row's real tokens; when it is given the
    padding is *removed before selection*, so it does not matter whether the
    batch is left- or right-padded: every request is compiled from exactly its
    own tokens, and its observation window is that request's own last
    ``observation_window`` tokens.

    Rows of different lengths compile to different retained widths.  The
    returned cache is a single rectangular one, so shorter rows are filled up
    to the widest row by **duplicating their own retained slots**.  That filler
    is not a hack: two identical ``(key, value)`` slots split the original
    slot's softmax weight between them, so attention over a duplicated cache is
    numerically identical to attention over the unpadded one.  The filler slots
    are nevertheless marked as padding in ``cache.qcc_attention_mask``.

    Returns ``(cache, last_logits)``.  Decode against ``cache`` exactly as
    against any other Hugging Face cache; each request keeps its own absolute
    positions because the retained keys carry their original rotary phases.
    ``cache.qcc_prompt_lengths`` (the true prompt length of every row) and
    ``cache.qcc_attention_mask`` (1 for a real slot, 0 for filler) are always
    attached, so a single-request call needs no special case.

    A ragged batch has one more requirement: the next token of row ``r`` has to
    be given the absolute position ``lengths[r] + step``, because the compiled
    cache is shorter than the prompt and no cumulative sum over a mask can
    recover that.  ``cache.qcc_prompt_lengths`` holds those true lengths, so
    decode with ``position_ids = cache.qcc_prompt_lengths[:, None] + step``
    (``cache.qcc_attention_mask`` masks the duplicated filler and may be grown
    with a one per step, but it is not a position source).

    Compilation runs one exact chunked prefill per request, so its cost is
    linear in the batch; what decode pays is the same for every row.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    mask = attention_mask.bool()
    lengths = [int(row.sum()) for row in mask]
    if not lengths or min(lengths) == 0:
        raise ValueError("every row needs at least one real token")
    prompt_lengths = torch.tensor(lengths, dtype=torch.long, device=input_ids.device)

    caches, logits = [], []
    for row, length in enumerate(lengths):
        ids = (input_ids[row: row + 1] if length == input_ids.shape[1]
               else input_ids[row][mask[row]].unsqueeze(0))
        row_cache, row_logits = _compile_one(model, ids, config, tokenizer)
        caches.append(row_cache)
        logits.append(row_logits)

    if len(caches) == 1:
        cache = caches[0]
        slots = cache.get_seq_length()
        cache.qcc_attention_mask = torch.ones(1, slots, dtype=torch.long,
                                              device=input_ids.device)
        cache.qcc_prompt_lengths = prompt_lengths
        return cache, logits[0]

    slots = [cache.layers[0].keys.shape[2] for cache in caches]
    width = max(slots)
    rows = []
    for layer_idx in range(len(caches[0].layers)):
        keys, values = [], []
        for cache, kept in zip(caches, slots):
            layer = cache.layers[layer_idx]
            k, v = layer.keys, layer.values
            if kept < width:  # exact filler: a duplicate of this row's own last slot
                duplicate = torch.arange(width, device=k.device).clamp_(max=kept - 1)
                k = k.index_select(2, duplicate)
                v = v.index_select(2, duplicate)
            keys.append(k)
            values.append(v)
        rows.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))

    from transformers import DynamicCache

    merged = DynamicCache()
    for layer_idx, (keys, values) in enumerate(rows):
        merged.update(keys, values, layer_idx)
    merged.qcc_attention_mask = torch.zeros(
        len(caches), width, dtype=torch.long, device=input_ids.device)
    for row, kept in enumerate(slots):
        merged.qcc_attention_mask[row, :kept] = 1
    merged.qcc_prompt_lengths = prompt_lengths
    return merged, torch.cat(logits, dim=0)
