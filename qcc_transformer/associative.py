"""Set-associative exact landmark memory for QCC research experiments.

The bank is intentionally independent from the recurrent response archive so it can be
benchmarked and calibrated in isolation. Persistent state is
``O(heads * sets * ways * head_dim)`` and therefore independent of context length.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class AssociativeLandmarkState:
    keys: Tensor
    values: Tensor
    scores: Tensor
    ages: Tensor


class SetAssociativeLandmarkBank(nn.Module):
    """Small exact-KV bank with learned set routing and multi-way replacement.

    Each incoming key is routed to the strongest learned set. Every set has ``ways``
    exact slots; replacement is controlled by an admission score plus a diversity
    bonus. Reads probe the top ``probe_sets`` sets and perform exact query/key matching
    inside their ways.

    ``write_mask`` is deliberately explicit. A production caller can run a learned
    salience predictor once per block and prevent low-value filler from touching the
    exact tier at all. This keeps state bounded *and* avoids turning constant memory
    into per-token Python replacement overhead.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        num_sets: int = 64,
        ways: int = 4,
        probe_sets: int = 2,
        admission_weight: float = 1.0,
        diversity_weight: float = 0.25,
        recency_weight: float = 0.0,
        temperature: float = 8.0,
        replacement_policy: str = "score",
        background_size: int = 0,
        block_size: int = 1,
        storage_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if min(num_heads, head_dim, num_sets, ways, probe_sets) <= 0:
            raise ValueError("head dimensions, sets, ways, and probes must be positive")
        if probe_sets > num_sets:
            raise ValueError("probe_sets cannot exceed num_sets")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if replacement_policy not in ("score", "fifo"):
            raise ValueError("replacement_policy must be 'score' or 'fifo'")
        if storage_dtype is not None and storage_dtype not in (
            torch.float16, torch.bfloat16, torch.float32, torch.float64
        ):
            raise ValueError("storage_dtype must be a floating-point torch dtype")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_sets = num_sets
        self.ways = ways
        self.probe_sets = probe_sets
        self.admission_weight = admission_weight
        self.diversity_weight = diversity_weight
        self.recency_weight = recency_weight
        self.temperature = temperature
        self.replacement_policy = replacement_policy
        if background_size < 0 or (background_size and (replacement_policy != 'score' or probe_sets != num_sets)):
            raise ValueError("background sampling requires a nonnegative capacity and global score replacement")
        self.background_size = background_size
        if block_size < 1 or (block_size > 1 and (ways != block_size or probe_sets != num_sets or replacement_policy != 'score')):
            raise ValueError('block retention requires ways=block_size and global score replacement')
        self.block_size = block_size
        self.storage_dtype = storage_dtype

        scale = 1.0 / math.sqrt(head_dim)
        self.set_codes = nn.Parameter(torch.randn(num_heads, num_sets, head_dim) * scale)
        self.admission_vector = nn.Parameter(torch.randn(num_heads, head_dim) * scale)
        self.reset_state(1)

    def reset_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        device = device or self.set_codes.device
        if self.storage_dtype is not None:
            dtype = self.storage_dtype
        else:
            dtype = dtype if dtype in (torch.float32, torch.float64) else torch.float32
        score_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        shape = (batch_size, self.num_heads, self.num_sets, self.ways)
        self._keys = torch.zeros(*shape, self.head_dim, device=device, dtype=dtype)
        self._values = torch.zeros_like(self._keys)
        self._scores = torch.full(shape, -torch.inf, device=device, dtype=score_dtype)
        self._ages = torch.zeros(shape, device=device, dtype=torch.long)
        self._fifo_cursor = torch.zeros(
            batch_size, self.num_heads, device=device, dtype=torch.long
        )
        self._step = 0
        self._pending_count = 0
        if self.block_size > 1:
            self._pending_keys = torch.zeros(batch_size, self.num_heads, self.block_size, self.head_dim, device=device, dtype=dtype)
            self._pending_values = torch.zeros_like(self._pending_keys)
            self._pending_scores = torch.full(
                (batch_size, self.num_heads, self.block_size),
                -torch.inf,
                device=device,
                dtype=score_dtype,
            )
        if self.background_size:
            self._background_keys = torch.zeros(batch_size, self.num_heads, self.background_size, self.head_dim, device=device, dtype=dtype)
            self._background_values = torch.zeros_like(self._background_keys)
            self._background_count = torch.zeros(batch_size, self.num_heads, device=device, dtype=torch.long)
            self._background_rngs = [
                torch.Generator(device=device).manual_seed(11)
                for _ in range(batch_size)
            ]

    @property
    def state(self) -> AssociativeLandmarkState:
        return AssociativeLandmarkState(
            self._keys, self._values, self._scores, self._ages
        )

    def state_bytes(self) -> int:
        """Return mutable serving-state bytes, excluding trainable parameters."""

        total = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self._keys, self._values, self._scores, self._ages)
        )
        if self.background_size:
            total += sum(t.numel() * t.element_size() for t in (
                self._background_keys, self._background_values, self._background_count,
                *(rng.get_state() for rng in self._background_rngs),
            ))
        if self.block_size > 1:
            total += sum(t.numel()*t.element_size() for t in (self._pending_keys, self._pending_values, self._pending_scores))
        return total

    @torch.no_grad()
    def _update_block(self, key, value, admission_bias, write_mask):
        slot = self._pending_count
        self._pending_keys[:, :, slot] = key
        self._pending_values[:, :, slot] = value
        score = (
            torch.zeros_like(self._scores[:, :, 0, 0])
            if admission_bias is None
            else admission_bias.to(self._pending_scores.dtype)
        )
        if write_mask is not None:
            if write_mask.ndim == 1:
                write_mask = write_mask[:, None]
            score = torch.where(write_mask, score, -torch.inf)
        self._pending_scores[:, :, slot] = score
        self._pending_count += 1
        if self._pending_count < self.block_size:
            return
        self._commit_pending_block()

    @torch.no_grad()
    def _commit_pending_block(self) -> None:
        score = self._pending_scores.amax(-1)
        previous = self._scores[:, :, :, 0]
        empty = ~torch.isfinite(previous)
        has_empty = empty.any(-1)
        index = torch.where(has_empty, empty.long().argmax(-1), previous.argmin(-1))
        weakest = previous.gather(-1, index[..., None]).squeeze(-1)
        write = torch.isfinite(score) & (has_empty | (score > weakest))
        device = self._keys.device
        batch = torch.arange(self._keys.shape[0], device=device)[:, None]
        head = torch.arange(self.num_heads, device=device)[None, :]
        old_keys = self._keys[batch, head, index]
        old_values = self._values[batch, head, index]
        demoted = write & ~has_empty
        if self.background_size:
            event = ~write | demoted
            block_keys = torch.where(
                demoted[..., None, None],
                old_keys,
                self._pending_keys,
            )
            block_values = torch.where(
                demoted[..., None, None],
                old_values,
                self._pending_values,
            )
            self._sample_background_block(block_keys, block_values, event)
        wb, wh = torch.where(write)
        selected = index[write]
        self._keys[wb, wh, selected] = self._pending_keys[write]
        self._values[wb, wh, selected] = self._pending_values[write]
        self._scores[wb, wh, selected] = score[write].unsqueeze(-1)
        ages = self._step-self.block_size+1+torch.arange(self.block_size, device=device)
        self._ages[wb, wh, selected] = ages
        self._pending_count = 0

    @torch.no_grad()
    def _sample_background(self, key: Tensor, value: Tensor, event: Tensor) -> None:
        """Reservoir-sample the stream of rejected or demoted foreground entries."""
        self._background_count.add_(event.to(torch.long))
        count = self._background_count
        draw = torch.stack([
            torch.rand(count.shape[1:], device=key.device, generator=rng)
            for rng in self._background_rngs
        ])
        slot = torch.where(count <= self.background_size, count - 1, (draw * count).long())
        keep = event & (slot < self.background_size)
        batch, head = torch.where(keep)
        selected = slot[keep]
        self._background_keys[batch, head, selected] = key.to(self._background_keys.dtype)[keep]
        self._background_values[batch, head, selected] = value.to(self._background_values.dtype)[keep]

    @torch.no_grad()
    def _sample_background_block(
        self, key: Tensor, value: Tensor, event: Tensor
    ) -> None:
        """Reservoir-sample one rejected block with bounded tensor work."""
        if key.ndim != 4 or value.shape != key.shape:
            raise ValueError("background block key/value must be [batch, heads, block, dim]")
        if key.shape[:2] != event.shape or key.shape[2] != self.block_size:
            raise ValueError("background block shape does not match bank")
        batch_size, heads, block, _ = key.shape
        count_before = self._background_count
        event_block = event.to(torch.long).unsqueeze(-1).expand(-1, -1, block)
        counts = count_before.unsqueeze(-1) + event_block.cumsum(-1)
        draws = torch.stack([
            torch.rand((block,), device=key.device, generator=rng)
            for rng in self._background_rngs
        ], dim=0).unsqueeze(1)
        slots = torch.where(
            counts <= self.background_size,
            counts - 1,
            (draws * counts).to(torch.long),
        )
        keep = event_block.bool() & (slots < self.background_size)
        slot_ids = torch.arange(
            self.background_size, device=key.device, dtype=torch.long
        ).view(1, 1, 1, -1)
        positions = torch.arange(block, device=key.device, dtype=torch.long).view(
            1, 1, block, 1
        )
        last = torch.where(
            keep.unsqueeze(-1) & (slots.unsqueeze(-1) == slot_ids),
            positions,
            torch.full_like(positions, -1),
        ).amax(2)
        valid = last >= 0
        gather = last.clamp_min(0).unsqueeze(-1).expand(
            batch_size, heads, self.background_size, key.shape[-1]
        )
        chosen_key = key.gather(2, gather).to(self._background_keys.dtype)
        chosen_value = value.gather(2, gather).to(self._background_values.dtype)
        batch_index = torch.arange(batch_size, device=key.device)[:, None, None]
        head_index = torch.arange(heads, device=key.device)[None, :, None]
        slot_index = torch.arange(
            self.background_size, device=key.device
        )[None, None, :]
        write_batch = batch_index.expand_as(valid)[valid]
        write_head = head_index.expand_as(valid)[valid]
        write_slot = slot_index.expand_as(valid)[valid]
        self._background_keys[write_batch, write_head, write_slot] = chosen_key[valid]
        self._background_values[write_batch, write_head, write_slot] = chosen_value[valid]
        self._background_count.add_(event_block.sum(-1))

    def _ensure_state(self, key: Tensor) -> None:
        if self._keys.shape[0] != key.shape[0] or self._keys.device != key.device:
            self.reset_state(key.shape[0], device=key.device, dtype=key.dtype)

    def _set_logits(self, x: Tensor) -> Tensor:
        # Routing must be invariant to the norm difference between a query and
        # its matching key.  A raw dot product makes the set assignment
        # depend on activation scale, which is especially unstable
        # after RoPE and across model families.
        normalized_x = F.normalize(x.to(self._keys.dtype), dim=-1)
        normalized_codes = F.normalize(
            self.set_codes.to(x.device, self._keys.dtype), dim=-1
        )
        return torch.einsum(
            "bhd,hmd->bhm",
            normalized_x,
            normalized_codes,
        ) / math.sqrt(self.head_dim)

    @torch.no_grad()
    def update(
        self,
        key: Tensor,
        value: Tensor,
        *,
        admission_bias: Tensor | None = None,
        write_mask: Tensor | None = None,
    ) -> None:
        """Insert one exact association per selected batch/head.

        Args:
            key, value: ``[batch, heads, head_dim]``.
            admission_bias: optional external salience score ``[batch, heads]``.
                A teacher-trained predictor can supply this without baking task labels
                into the bank.
            write_mask: optional bool mask ``[batch, heads]`` or ``[batch]``. False
                entries do not allocate empty slots and cannot replace existing slots.
        """

        if key.shape != value.shape or key.ndim != 3:
            raise ValueError("key and value must have shape [batch, heads, head_dim]")
        if key.shape[1:] != (self.num_heads, self.head_dim):
            raise ValueError("key shape does not match bank configuration")
        self._ensure_state(key)
        self._step += 1
        if self.block_size > 1:
            self._update_block(key, value, admission_bias, write_mask)
            return

        # FIFO is a deliberately simple bounded-quality mode.  It avoids
        # making replacement depend on an uncalibrated salience score while
        # retaining exactly the same fixed state footprint.  The caller still
        # controls admission with ``write_mask``.
        if self.replacement_policy == "fifo":
            slots = self.num_sets * self.ways
            batch_index = torch.arange(key.shape[0], device=key.device)[:, None]
            head_index = torch.arange(self.num_heads, device=key.device)[None, :]
            should_write = torch.ones(
                key.shape[0], self.num_heads, device=key.device, dtype=torch.bool
            )
            if write_mask is not None:
                if write_mask.shape == (key.shape[0],):
                    write_mask = write_mask[:, None].expand(-1, self.num_heads)
                if write_mask.shape != should_write.shape:
                    raise ValueError("write_mask must have shape [batch] or [batch, heads]")
                should_write &= write_mask.to(device=key.device, dtype=torch.bool)
            cursor = self._fifo_cursor.to(device=key.device)
            write_set = torch.div(cursor, self.ways, rounding_mode="floor")
            write_way = cursor % self.ways
            selected_batch = batch_index.expand_as(cursor)[should_write]
            selected_head = head_index.expand_as(cursor)[should_write]
            selected_set = write_set[should_write]
            selected_way = write_way[should_write]
            self._keys[selected_batch, selected_head, selected_set, selected_way] = key.to(
                self._keys.dtype
            )[should_write]
            self._values[selected_batch, selected_head, selected_set, selected_way] = value.to(
                self._values.dtype
            )[should_write]
            self._scores[selected_batch, selected_head, selected_set, selected_way] = (
                0.0 if admission_bias is None else admission_bias.to(self._scores.dtype)[should_write]
            )
            self._ages[selected_batch, selected_head, selected_set, selected_way] = self._step
            next_cursor = (cursor + should_write.to(cursor.dtype)) % slots
            self._fifo_cursor = next_cursor
            return

        batch_index = torch.arange(key.shape[0], device=key.device)[:, None]
        head_index = torch.arange(self.num_heads, device=key.device)[None, :]
        global_bank = self.probe_sets == self.num_sets
        if global_bank:
            # When every set is probed, routing a write to only one set wastes
            # capacity and makes collisions depend on an untrained codebook.
            # Treat the bank as one fixed global exact table in that mode. It
            # retains the strongest admitted associations across all slots,
            # while the bounded state size and read cost remain unchanged.
            slots_key = self._keys.reshape(
                key.shape[0], self.num_heads, self.num_sets * self.ways, self.head_dim
            )
            slots_score = self._scores.reshape(
                key.shape[0], self.num_heads, self.num_sets * self.ways
            )
        else:
            set_logits = self._set_logits(key)
            set_index = set_logits.argmax(dim=-1)
            slots_key = self._keys[batch_index, head_index, set_index]
            slots_score = self._scores[batch_index, head_index, set_index]
        valid = torch.isfinite(slots_score)

        diversity = 0.0
        if self.diversity_weight:
            normalized_key = F.normalize(key.float(), dim=-1)
            normalized_slots = F.normalize(slots_key.float(), dim=-1)
            similarity = torch.einsum("bhd,bhwd->bhw", normalized_key, normalized_slots)
            max_similarity = torch.where(
                valid, similarity, torch.full_like(similarity, -1.0)
            ).max(-1).values
            diversity = 1.0 - max_similarity

        admission = torch.einsum(
            "bhd,hd->bh",
            key.float(),
            self.admission_vector.to(key.device, dtype=torch.float32),
        ) / math.sqrt(self.head_dim)
        if admission_bias is not None:
            if admission_bias.shape != admission.shape:
                raise ValueError("admission_bias must have shape [batch, heads]")
            admission = admission + admission_bias.to(admission.dtype)
        candidate_score = (
            self.admission_weight * admission + self.diversity_weight * diversity
        )

        empty = ~valid
        has_empty = empty.any(dim=-1)
        empty_index = empty.to(torch.int64).argmax(dim=-1)
        if global_bank:
            ages = self._ages.reshape(
                key.shape[0], self.num_heads, self.num_sets * self.ways
            )
        else:
            ages = self._ages[batch_index, head_index, set_index]
        age = (self._step - ages).to(self._scores.dtype)
        replacement_score = slots_score - self.recency_weight / (1.0 + age)
        weakest_index = replacement_score.argmin(dim=-1)
        slot_index = torch.where(has_empty, empty_index, weakest_index)
        weakest_score = replacement_score.gather(
            -1, weakest_index.unsqueeze(-1)
        ).squeeze(-1)
        should_write = has_empty | (candidate_score > weakest_score)

        if write_mask is not None:
            if write_mask.shape == (key.shape[0],):
                write_mask = write_mask[:, None].expand(-1, self.num_heads)
            if write_mask.shape != should_write.shape:
                raise ValueError("write_mask must have shape [batch] or [batch, heads]")
            should_write = should_write & write_mask.to(
                device=key.device, dtype=torch.bool
            )

        if self.background_size:
            demoted = should_write & ~has_empty
            gather = slot_index[..., None, None].expand(-1, -1, 1, self.head_dim)
            old_key = slots_key.gather(2, gather).squeeze(2)
            old_value = self._values.flatten(2, 3).gather(2, gather).squeeze(2)
            self._sample_background(
                torch.where(demoted[..., None], old_key, key),
                torch.where(demoted[..., None], old_value, value),
                ~should_write | demoted,
            )
        write_batch = batch_index.expand_as(slot_index)[should_write]
        write_head = head_index.expand_as(slot_index)[should_write]
        if global_bank:
            write_set = torch.div(slot_index[should_write], self.ways, rounding_mode="floor")
            write_way = slot_index[should_write] % self.ways
        else:
            write_set = set_index[should_write]
            write_way = slot_index[should_write]
        self._keys[write_batch, write_head, write_set, write_way] = key.to(
            self._keys.dtype
        )[should_write]
        self._values[write_batch, write_head, write_set, write_way] = value.to(
            self._values.dtype
        )[should_write]
        self._scores[write_batch, write_head, write_set, write_way] = candidate_score[
            should_write
        ].to(self._scores.dtype)
        self._ages[write_batch, write_head, write_set, write_way] = self._step

    def read_attention(
        self,
        query: Tensor,
        *,
        extra_keys: Tensor | None = None,
        extra_values: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Attend over all retained slots; return response and log partition.

        The log partition lets callers combine disjoint exact KV partitions
        with softmax mass, without a cosine-confidence heuristic. Query tiles
        bound temporary score storage; persistent bank contents are unchanged.
        """
        single = query.ndim == 3
        block = query.unsqueeze(2) if single else query
        if block.ndim != 4 or block.shape[1] != self.num_heads or block.shape[-1] != self.head_dim:
            raise ValueError("query must have shape [batch, heads, (tokens,) head_dim]")
        batch, heads, tokens, dim = block.shape
        if block.shape[0] != self._keys.shape[0] or block.device != self._keys.device:
            raise ValueError("query batch and device must match bank state")
        if (extra_keys is None) != (extra_values is None):
            raise ValueError("extra_keys and extra_values must be provided together")
        if extra_keys is not None:
            if extra_keys.ndim != 4 or extra_values.shape != extra_keys.shape:
                raise ValueError("extra keys and values must have shape [batch, heads, tokens, dim]")
            if extra_keys.shape[:2] != block.shape[:2] or extra_keys.shape[2] != block.shape[2] or extra_keys.shape[3] != dim:
                raise ValueError("extra keys and values must match query batch, head, token, and dim")
        keys = self._keys.reshape(batch, heads, -1, dim).float()
        values = self._values.reshape(batch, heads, -1, dim).float()
        if torch.is_grad_enabled():
            # Later admissions mutate serving state; backward needs this read's snapshot.
            keys, values = keys.clone(), values.clone()
        valid = torch.isfinite(self._scores.reshape(batch, heads, -1))
        log_multiplicity = None
        if self.background_size:
            population = self._background_count
            sample_count = population.clamp(max=self.background_size)
            bg_valid = torch.arange(self.background_size, device=query.device)[None, None, :] < sample_count[..., None]
            log_scale = (population.clamp_min(1).float() / sample_count.clamp_min(1)).log()
            log_multiplicity = torch.cat((torch.zeros_like(valid, dtype=torch.float32), log_scale[..., None].expand_as(bg_valid)), -1)
            keys = torch.cat((keys, self._background_keys.float()), 2)
            values = torch.cat((values, self._background_values.float()), 2)
            valid = torch.cat((valid, bg_valid), -1)
        if self.block_size > 1 and self._pending_count:
            count = self._pending_count
            keys = torch.cat((keys, self._pending_keys[:, :, :count].float()), 2)
            values = torch.cat((values, self._pending_values[:, :, :count].float()), 2)
            pending_valid = torch.ones(batch, heads, count, device=query.device, dtype=torch.bool)
            valid = torch.cat((valid, pending_valid), -1)
            if log_multiplicity is not None:
                log_multiplicity = torch.cat((log_multiplicity, torch.zeros_like(pending_valid, dtype=torch.float32)), -1)
        extra_valid = None
        if extra_keys is not None:
            extra_count = extra_keys.shape[2]
            keys = torch.cat((keys, extra_keys.float()), 2)
            values = torch.cat((values, extra_values.float()), 2)
            extra_valid = torch.tril(
                torch.ones(tokens, extra_count, device=query.device, dtype=torch.bool)
            )
            if log_multiplicity is not None:
                log_multiplicity = torch.cat(
                    (log_multiplicity, torch.zeros(batch, heads, extra_count, device=query.device)), -1
                )
        response = torch.empty_like(block)
        partition = torch.empty((batch, heads, tokens), device=block.device, dtype=torch.float32)
        for start in range(0, tokens, 128):
            end = min(start + 128, tokens)
            logits = torch.matmul(block[:, :, start:end].float(), keys.transpose(-1, -2)) / math.sqrt(dim)
            if log_multiplicity is not None:
                logits = logits + log_multiplicity.unsqueeze(2)
            valid_mask = valid.unsqueeze(2).expand(-1, -1, end - start, -1)
            if extra_valid is not None:
                valid_mask = torch.cat(
                    (valid_mask, extra_valid[start:end].view(1, 1, end - start, -1).expand(batch, heads, -1, -1)),
                    dim=-1,
                )
            logits = logits.masked_fill(~valid_mask, -torch.inf)
            log_z = torch.logsumexp(logits, -1)
            safe_z = torch.where(valid_mask.any(-1), log_z, torch.zeros_like(log_z))
            weights = torch.exp(logits - safe_z.unsqueeze(-1))
            response[:, :, start:end] = torch.matmul(weights, values).to(block.dtype)
            partition[:, :, start:end] = log_z
        if single:
            return response.squeeze(2), partition.squeeze(2)
        return response, partition

    @torch.no_grad()
    def update_read_chunk(
        self, key: Tensor, value: Tensor, query: Tensor, *,
        admission_score: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Read evicted KV causally, then commit at fixed block boundaries.

        Scores must come from information available when each event arrives.
        Pending KV remain exact until the logical block's final query has read
        them. External call boundaries do not trigger retention decisions.
        """
        if key.shape != value.shape or key.shape != query.shape or key.ndim != 4:
            raise ValueError("key, value and query must share [batch, heads, tokens, dim]")
        if admission_score.shape != key.shape[:-1]:
            raise ValueError("admission_score must have shape [batch, heads, tokens]")
        if key.shape[2]:
            self._ensure_state(key[:, :, 0])
        output = torch.empty_like(query)
        partition = torch.empty(query.shape[:-1], device=query.device, dtype=torch.float32)
        start = 0
        while start < key.shape[2]:
            pending = self._pending_count if self.block_size > 1 else 0
            end = min(key.shape[2], start + self.block_size - pending)
            # Match the precision of already-pending entries across API calls.
            keys = key[:, :, start:end].to(self._keys.dtype)
            values = value[:, :, start:end].to(self._values.dtype)
            response, log_z = self.read_attention(
                query[:, :, start:end], extra_keys=keys, extra_values=values,
            )
            output[:, :, start:end] = response
            partition[:, :, start:end] = log_z
            if self.block_size > 1:
                count = end - start
                self._pending_keys[:, :, pending:pending + count] = keys
                self._pending_values[:, :, pending:pending + count] = values
                self._pending_scores[:, :, pending:pending + count] = admission_score[:, :, start:end]
                self._step += count
                self._pending_count += count
                if self._pending_count == self.block_size:
                    self._commit_pending_block()
            else:
                self.update(
                    keys[:, :, 0], values[:, :, 0],
                    admission_bias=admission_score[:, :, start],
                )
            start = end
        return output, partition

    def read(self, query: Tensor, *, hard: bool = False) -> tuple[Tensor, Tensor]:
        """Read top routed sets; return response and best cosine confidence."""

        if query.ndim != 3 or query.shape[1:] != (self.num_heads, self.head_dim):
            raise ValueError("query must have shape [batch, heads, head_dim]")
        self._ensure_state(query)
        if query.is_cuda and hard:
            response, confidence = self.read_chunk(
                query.unsqueeze(2), hard=hard
            )
            return response.squeeze(2), confidence.squeeze(2)
        set_logits = self._set_logits(query)
        probe = set_logits.topk(self.probe_sets, dim=-1).indices
        batch_index = torch.arange(query.shape[0], device=query.device)[:, None, None]
        head_index = torch.arange(self.num_heads, device=query.device)[None, :, None]
        keys = self._keys[batch_index, head_index, probe].reshape(
            query.shape[0], self.num_heads, -1, self.head_dim
        )
        values = self._values[batch_index, head_index, probe].reshape_as(keys)
        scores = self._scores[batch_index, head_index, probe].reshape(
            query.shape[0], self.num_heads, -1
        )
        valid = torch.isfinite(scores)

        normalized_query = F.normalize(query.to(keys.dtype), dim=-1)
        normalized_keys = F.normalize(keys, dim=-1)
        similarity = torch.einsum(
            "bhd,bhnd->bhn", normalized_query, normalized_keys
        )
        similarity = torch.where(
            valid, similarity, torch.full_like(similarity, -1.0e9)
        )
        confidence, best = similarity.max(dim=-1)
        if hard:
            response = values.gather(
                2, best[..., None, None].expand(-1, -1, 1, self.head_dim)
            ).squeeze(2)
        else:
            weights = F.softmax(similarity * self.temperature, dim=-1).to(values.dtype)
            response = torch.einsum("bhn,bhnd->bhd", weights, values)
        any_valid = valid.any(dim=-1)
        response = torch.where(
            any_valid.unsqueeze(-1), response, torch.zeros_like(response)
        )
        confidence = torch.where(
            any_valid, confidence, torch.full_like(confidence, -1.0)
        )
        return response.to(query.dtype), confidence

    def read_chunk(
        self, query: Tensor, *, hard: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Vectorized reads against one frozen bank state.

        ``query`` is ``[batch, heads, tokens, head_dim]``. The state must not be
        mutated during this call; a causal caller can therefore process a chunk in a
        handful of segments separated by the few admitted insertions instead of one
        Python operation per token.
        """

        if (
            query.ndim != 4
            or query.shape[1] != self.num_heads
            or query.shape[-1] != self.head_dim
        ):
            raise ValueError(
                "query must have shape [batch, heads, tokens, head_dim]"
            )
        batch, heads, tokens, dim = query.shape
        if tokens == 0:
            return query.new_empty(query.shape), query.new_empty(batch, heads, 0)
        self._ensure_state(query[:, :, 0])

        # Do not materialize [batch, heads, tokens, probe, ways, dim] for an
        # entire million-token prefill.  The exact bank is constant-size, so a
        # bounded query tile preserves the same result while keeping temporary
        # memory independent of the requested context length.
        block_size = 1024
        if tokens > block_size:
            responses = []
            confidences = []
            for start in range(0, tokens, block_size):
                response, confidence = self.read_chunk(
                    query[:, :, start : start + block_size], hard=hard
                )
                responses.append(response)
                confidences.append(confidence)
            return torch.cat(responses, dim=2), torch.cat(confidences, dim=2)

        # ``probe_sets == num_sets`` is the default quality configuration. In
        # that mode routing is only an index permutation: every slot is searched
        # anyway.  The direct path avoids materializing a per-token set-index
        # tensor and keeps the CPU fallback at [B,H,T,S*W] similarities rather
        # than [B,H,T,S*W,D] gathered keys/values.
        if self.probe_sets == self.num_sets:
            if query.is_cuda and hard:
                try:
                    from .triton_kernels import (
                        TRITON_AVAILABLE,
                        triton_exact_global_read_chunk,
                    )
                except ImportError:  # pragma: no cover - optional CUDA dependency
                    TRITON_AVAILABLE = False
                if TRITON_AVAILABLE:
                    return triton_exact_global_read_chunk(
                        query,
                        self._keys,
                        self._values,
                        self._scores,
                    )

            slots = self.num_sets * self.ways
            keys = self._keys.reshape(batch, heads, slots, dim)
            values = self._values.reshape(batch, heads, slots, dim)
            scores = self._scores.reshape(batch, heads, slots)
            valid = torch.isfinite(scores)
            normalized_query = F.normalize(query.to(keys.dtype), dim=-1)
            normalized_keys = F.normalize(keys, dim=-1)
            similarity = torch.einsum(
                "bhtd,bhsd->bhts", normalized_query, normalized_keys
            )
            similarity = torch.where(
                valid.unsqueeze(2), similarity, torch.full_like(similarity, -1.0e9)
            )
            confidence, best = similarity.max(dim=-1)
            if hard:
                # Gather along the slot dimension directly.  Expanding the
                # value table to [B,H,T,S,D] is only a view in theory, but
                # gather materializes that expanded indexing path on several
                # torch backends.  The direct rank-4 gather keeps temporary
                # memory at [B,H,T,D] while preserving the exact argmax read.
                response = values.gather(
                    2, best[..., None].expand(batch, heads, tokens, dim)
                )
            else:
                weights = F.softmax(similarity * self.temperature, dim=-1).to(
                    values.dtype
                )
                response = torch.einsum("bhts,bhsd->bhtd", weights, values)
            any_valid = valid.any(dim=-1).unsqueeze(2)
            response = torch.where(
                any_valid.unsqueeze(-1), response, torch.zeros_like(response)
            )
            confidence = torch.where(
                any_valid, confidence, torch.full_like(confidence, -1.0)
            )
            return response.to(query.dtype), confidence

        codes = F.normalize(
            self.set_codes.to(query.device, self._keys.dtype), dim=-1
        )
        normalized_query = F.normalize(query.to(self._keys.dtype), dim=-1)
        set_logits = torch.einsum(
            "bhtd,hmd->bhtm", normalized_query, codes
        ) / math.sqrt(self.head_dim)
        probe = set_logits.topk(self.probe_sets, dim=-1).indices

        # The exact tier is on the serving critical path.  When Triton is
        # available, keep set selection as one small tensor op and fuse the
        # routed cosine search/value gather into one kernel launch per query.
        if query.is_cuda and hard:
            try:
                from .triton_kernels import TRITON_AVAILABLE, triton_exact_read_chunk
            except ImportError:  # pragma: no cover - optional CUDA dependency
                TRITON_AVAILABLE = False
            if TRITON_AVAILABLE:
                return triton_exact_read_chunk(
                    query,
                    probe,
                    self._keys,
                    self._values,
                    self._scores,
                )

        keys_all = self._keys[:, :, None].expand(
            batch, heads, tokens, self.num_sets, self.ways, dim
        )
        values_all = self._values[:, :, None].expand_as(keys_all)
        scores_all = self._scores[:, :, None].expand(
            batch, heads, tokens, self.num_sets, self.ways
        )
        gather_keys = probe[..., None, None].expand(
            batch, heads, tokens, self.probe_sets, self.ways, dim
        )
        keys = keys_all.gather(3, gather_keys).reshape(
            batch, heads, tokens, -1, dim
        )
        values = values_all.gather(3, gather_keys).reshape_as(keys)
        gather_scores = probe[..., None].expand(
            batch, heads, tokens, self.probe_sets, self.ways
        )
        scores = scores_all.gather(3, gather_scores).reshape(
            batch, heads, tokens, -1
        )
        valid = torch.isfinite(scores)

        normalized_query = F.normalize(query.to(keys.dtype), dim=-1)
        normalized_keys = F.normalize(keys, dim=-1)
        similarity = torch.einsum(
            "bhtd,bhtnd->bhtn", normalized_query, normalized_keys
        )
        similarity = torch.where(
            valid, similarity, torch.full_like(similarity, -1.0e9)
        )
        confidence, best = similarity.max(dim=-1)
        if hard:
            response = values.gather(
                3, best[..., None, None].expand(batch, heads, tokens, 1, dim)
            ).squeeze(3)
        else:
            weights = F.softmax(similarity * self.temperature, dim=-1).to(
                values.dtype
            )
            response = torch.einsum("bhtn,bhtnd->bhtd", weights, values)
        any_valid = valid.any(dim=-1)
        response = torch.where(
            any_valid.unsqueeze(-1), response, torch.zeros_like(response)
        )
        confidence = torch.where(
            any_valid, confidence, torch.full_like(confidence, -1.0)
        )
        return response.to(query.dtype), confidence
