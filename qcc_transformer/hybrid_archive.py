"""Hybrid recurrent + exact associative archive for pretrained QCC retrofits.

The original QCC recurrence is excellent for constant-state averaging but can erase a
small number of retrieval-critical associations. ``HybridQCCArchive`` keeps that
recurrent path unchanged and adds a *small, fixed-capacity* exact tier. A separately
trainable admission predictor decides which evicted tokens are allowed to touch the
exact bank.

The exact tier is deliberately fail-safe:

* an uncalibrated predictor starts with a negative bias, so the base QCC behavior is
  preserved rather than randomly caching historical tokens;
* at most ``max_inserts_per_chunk`` events are admitted from a prefill block;
* exact reads are confidence gated and remain constant-size with context length;
* the helper refuses sparse/lazy base archives for now, avoiding double updates in a
  fallback path that calls virtual ``update``/``read`` methods internally.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .associative import SetAssociativeLandmarkBank
from .causal_coreset import CausalWeightedKVBank
from .model import QCCArchive, QCCSelfAttention


class LandmarkAdmissionPredictor(nn.Module):
    """Tiny per-head predictor for future retrieval salience.

    The predictor is trained with teacher-derived labels rather than through the hard
    replacement decision. It uses normalized K/V features so one calibration threshold
    transfers more cleanly across activation scales. Parameter cost is only
    ``2 * heads * head_dim + heads``.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        bias_init: float = -4.0,
    ) -> None:
        super().__init__()
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError("num_heads and head_dim must be positive")
        if not math.isfinite(bias_init):
            raise ValueError("bias_init must be finite")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.key_weight = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.value_weight = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.bias = nn.Parameter(torch.full((num_heads,), float(bias_init)))

    def forward(self, key: Tensor, value: Tensor) -> Tensor:
        if key.shape != value.shape or key.ndim not in (3, 4):
            raise ValueError(
                "key/value must be [batch, heads, dim] or [batch, heads, tokens, dim]"
            )
        if key.shape[1] != self.num_heads or key.shape[-1] != self.head_dim:
            raise ValueError("key/value shape does not match admission predictor")
        key_f = F.normalize(key.float(), dim=-1)
        value_f = F.normalize(value.float(), dim=-1)
        if key.ndim == 3:
            score = torch.einsum("bhd,hd->bh", key_f, self.key_weight.float())
            score = score + torch.einsum(
                "bhd,hd->bh", value_f, self.value_weight.float()
            )
            return score + self.bias.float().view(1, -1)
        score = torch.einsum("bhtd,hd->bht", key_f, self.key_weight.float())
        score = score + torch.einsum(
            "bhtd,hd->bht", value_f, self.value_weight.float()
        )
        return score + self.bias.float().view(1, -1, 1)


class HybridQCCArchive(QCCArchive):
    """QCC recurrent archive plus a fixed-capacity exact associative tier."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_codes: int = 16,
        decay_rates: tuple[float, ...] = (0.995, 0.98, 0.94, 0.85),
        window_size: int = 128,
        *,
        use_triton: bool = True,
        active_codes: int | None = None,
        lazy_decay: bool = False,
        scan_block_size: int = 1024,
        content_threshold: float | None = None,
        persistent_landmark: bool = False,
        prefix_landmark: bool = False,
        prefix_pair_landmark: bool = False,
        landmark_temperature: float = 1.0,
        global_normalization: bool = True,
        query_correction_rank: int = 0,
        exact_num_sets: int = 128,
        exact_ways: int = 4,
        exact_probe_sets: int | None = None,
        exact_replacement_policy: str = "score",
        admission_threshold: float = 0.0,
        admission_bias_init: float = -4.0,
        max_inserts_per_chunk: int = 8,
        exact_confidence_threshold: float = 0.60,
        exact_confidence_temperature: float = 20.0,
        exact_mix_bias_init: float = -4.0,
        quality_first: bool = False,
        quality_block_propagation: bool = False,
        exact_attention: bool = False,
        quality_prefill_shadow_only: bool = False,
        exact_storage_dtype: torch.dtype | None = None,
        exact_query_correction: bool = False,
        active_query_correction: bool = False,
        causal_coreset: bool = False,
        coreset_capacity: int | None = None,
        coreset_merge_policy: str = "ward",
        coreset_query_probes: int = 8,
        coreset_partition_weight: float = 1.0,
        background_size: int = 0,
        block_size: int = 1,
        quality_query_tail: int | None = None,
    ) -> None:
        if background_size and not exact_attention:
            raise ValueError("background sampling requires exact_attention")
        if block_size > 1 and not background_size:
            raise ValueError('hybrid block retention requires background sampling to process every eviction')
        if quality_query_tail is not None and quality_query_tail <= 0:
            raise ValueError("quality_query_tail must be positive")
        if causal_coreset:
            if not exact_attention:
                raise ValueError("causal_coreset requires exact_attention")
            if quality_first:
                raise ValueError("causal_coreset cannot use future-query quality_first selection")
            if background_size or block_size != 1:
                raise ValueError("causal_coreset does not support background or block retention")
            if exact_storage_dtype not in (None, torch.float32):
                raise ValueError("causal_coreset representative statistics require float32 storage")
            if coreset_capacity is None:
                coreset_capacity = exact_num_sets * exact_ways
            if coreset_capacity <= 0:
                raise ValueError("coreset_capacity must be positive")
        elif coreset_capacity is not None:
            raise ValueError("coreset_capacity requires causal_coreset")
        if active_query_correction and exact_query_correction:
            raise ValueError("active_query_correction and exact_query_correction cannot both be enabled")
        self.quality_query_tail = quality_query_tail
        if active_codes is not None or lazy_decay:
            raise ValueError(
                "HybridQCCArchive currently requires the dense base archive; "
                "sparse/lazy fallbacks can double-dispatch virtual updates"
            )
        if quality_first:
            # Quality-first is an explicit bounded exact shadow.  Use the
            # score-based global table rather than FIFO: a long prefill is
            # processed in many bounded tiles, and FIFO would overwrite all
            # previously useful records at every tile boundary.  The chunk
            # path below supplies a future query-key salience score so this
            # table keeps retrieval-critical history across the whole stream.
            exact_replacement_policy = "score"
            exact_probe_sets = exact_num_sets
            # Scores are bounded by the predictor's normalized features; a
            # large finite floor keeps constructor validation meaningful.
            admission_threshold = -1.0e9
            # The exact tier is a shadow for quality recovery, not a license to
            # replace the recurrent response on every moderately similar key.
            # Soft reads average unrelated values in a dense fixed table and the
            # old forced-positive mix bias made that contamination pervasive.
            # Keep nearest-neighbour reads and honor the caller's mix bias so the
            # confidence gate remains a conservative, tunable fallback.
            exact_mix_bias_init = float(exact_mix_bias_init)
            exact_hard_read = True
            max_inserts_per_chunk = exact_num_sets * exact_ways
        else:
            exact_hard_read = True
        super().__init__(
            num_heads,
            head_dim,
            num_codes=num_codes,
            decay_rates=decay_rates,
            window_size=window_size,
            use_triton=use_triton,
            active_codes=active_codes,
            lazy_decay=lazy_decay,
            scan_block_size=scan_block_size,
            content_threshold=content_threshold,
            persistent_landmark=persistent_landmark,
            prefix_landmark=prefix_landmark,
            prefix_pair_landmark=prefix_pair_landmark,
            landmark_temperature=landmark_temperature,
            global_normalization=global_normalization,
            query_correction_rank=query_correction_rank,
        )
        if max_inserts_per_chunk <= 0:
            raise ValueError("max_inserts_per_chunk must be positive")
        if not math.isfinite(admission_threshold):
            raise ValueError("admission_threshold must be finite")
        if not -1.0 <= exact_confidence_threshold <= 1.0:
            raise ValueError("exact_confidence_threshold must lie in [-1, 1]")
        if exact_confidence_temperature <= 0 or not math.isfinite(
            exact_confidence_temperature
        ):
            raise ValueError("exact_confidence_temperature must be positive and finite")
        # The exact bank is the quality-recovery tier.  By default, search the
        # complete fixed table so an untrained set codebook cannot hide a
        # salient record in an unprobed bucket.  Deployments with a measured
        # latency budget may still set ``exact_probe_sets`` explicitly.
        probes = exact_num_sets if exact_probe_sets is None else exact_probe_sets
        self.causal_coreset = bool(causal_coreset)
        if self.causal_coreset:
            self.exact_bank = CausalWeightedKVBank(
                num_heads=num_heads,
                head_dim=head_dim,
                capacity=int(coreset_capacity),
                storage_dtype=(
                    torch.float32 if exact_storage_dtype is None else exact_storage_dtype
                ),
                merge_policy=coreset_merge_policy,
                query_probe_capacity=coreset_query_probes,
                response_partition_weight=coreset_partition_weight,
            )
        else:
            self.exact_bank = SetAssociativeLandmarkBank(
                num_heads=num_heads,
                head_dim=head_dim,
                num_sets=exact_num_sets,
                ways=exact_ways,
                probe_sets=probes,
                diversity_weight=0.0 if background_size else 0.10,
                replacement_policy=exact_replacement_policy,
                background_size=background_size,
                block_size=block_size,
                storage_dtype=exact_storage_dtype,
            )
        # Hybrid admission is supplied by the teacher-trained predictor below.
        # Keep the bank's legacy internal score neutral and frozen.
        if not self.causal_coreset:
            with torch.no_grad():
                self.exact_bank.admission_vector.zero_()
            self.exact_bank.admission_vector.requires_grad_(False)
        self.admission = LandmarkAdmissionPredictor(
            num_heads, head_dim, bias_init=admission_bias_init
        )
        self.exact_mix_logits = nn.Parameter(
            torch.full((num_heads,), float(exact_mix_bias_init))
        )
        self.admission_threshold = float(admission_threshold)
        self.max_inserts_per_chunk = int(max_inserts_per_chunk)
        self.exact_confidence_threshold = float(exact_confidence_threshold)
        self.exact_confidence_temperature = float(exact_confidence_temperature)
        self.quality_first = bool(quality_first)
        self.quality_block_propagation = bool(quality_block_propagation)
        self.exact_attention = bool(exact_attention)
        self.quality_prefill_shadow_only = bool(quality_prefill_shadow_only)
        self.exact_query_correction = bool(exact_query_correction)
        self.active_query_correction = bool(active_query_correction)
        if self.causal_coreset:
            # The causal coreset is a parameter-free response operator.  The
            # recurrent codebook, admission predictor, and learned blend are
            # obsolete in this mode; keeping them trainable would report
            # adapter capacity for a path that cannot affect its output.
            for parameter in self.parameters():
                parameter.requires_grad_(False)
        # Global-table writes and weighted reads never consult set routing.
        if (
            not self.causal_coreset
            and self.exact_attention
            and probes == exact_num_sets
        ):
            self.exact_bank.set_codes.requires_grad_(False)
        self.exact_hard_read = exact_hard_read
        # Per-read confidence gate consumed by QCCSelfAttention.  Keeping it
        # separate from the blended response lets quality-first mode promote
        # a true exact hit without changing the conservative recurrent/local
        # mix for unrelated queries.
        self._last_exact_gate: Tensor | None = None
        self._quality_previous_raw_score: Tensor | None = None
        self._quality_current_raw_score: Tensor | None = None
        self._quality_current_write_score: Tensor | None = None
        # The exact tier admits a whole bounded tile in quality-first mode;
        # matching the tile to its capacity lets the global score table see
        # every candidate while retaining only the strongest fixed-capacity
        # records across the complete stream.
        if self.quality_first:
            self.scan_block_size = min(self.scan_block_size, self.max_inserts_per_chunk)

    @classmethod
    def from_archive(
        cls,
        archive: QCCArchive,
        **hybrid_kwargs: Any,
    ) -> "HybridQCCArchive":
        """Upgrade an existing dense QCC archive without changing its recurrence."""

        if isinstance(archive, cls):
            return archive
        if archive.active_codes is not None or archive.lazy_decay:
            raise ValueError("hybrid upgrade requires a dense non-lazy QCCArchive")
        upgraded = cls(
            archive.num_heads,
            archive.head_dim,
            num_codes=archive.num_codes,
            decay_rates=tuple(float(x) for x in archive.decay_rates.tolist()),
            window_size=archive.window_size,
            use_triton=archive.use_triton,
            scan_block_size=archive.scan_block_size,
            content_threshold=archive.content_threshold,
            persistent_landmark=archive.persistent_landmark,
            prefix_landmark=archive.prefix_landmark,
            prefix_pair_landmark=archive.prefix_pair_landmark,
            landmark_temperature=archive.landmark_temperature,
            global_normalization=archive.global_normalization,
            query_correction_rank=archive.query_correction_rank,
            **hybrid_kwargs,
        ).to(device=archive.codes.device)
        base_state = archive.state_dict()
        missing, unexpected = upgraded.load_state_dict(base_state, strict=False)
        unexpected = [name for name in unexpected if not name.startswith("exact_")]
        if unexpected:
            raise ValueError(f"unexpected base archive state: {unexpected}")
        # Missing keys are expected for exact_bank/admission/exact_mix_logits.
        del missing

        # Preserve in-flight recurrent state when the upgrade is applied at an
        # explicit request boundary or during a diagnostic.
        upgraded._numerator = archive._numerator.clone()
        upgraded._denominator = archive._denominator.clone()
        upgraded._last_step = archive._last_step.clone()
        upgraded._step = int(archive._step)
        if archive.persistent_landmark:
            upgraded._landmark_count = int(archive._landmark_count)
            upgraded._prefix_pending_slot = int(archive._prefix_pending_slot)
            upgraded._landmark_score = archive._landmark_score.clone()
            upgraded._landmark_value = archive._landmark_value.clone()
            upgraded._landmark_key = archive._landmark_key.clone()
        upgraded.exact_bank.to(device=archive.codes.device)
        upgraded.admission.to(device=archive.codes.device)
        return upgraded

    def reset_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().reset_state(batch_size, device=device, dtype=dtype)
        self._last_exact_gate = None
        self._last_exact_log_partition = None
        self._quality_previous_raw_score = None
        self._quality_current_raw_score = None
        self._quality_current_write_score = None
        if hasattr(self, "exact_bank"):
            self.exact_bank.reset_state(
                batch_size,
                device=device or self.codes.device,
                dtype=dtype,
            )

    def exact_state_bytes(self) -> int:
        return self.exact_bank.state_bytes()

    def total_state_bytes(self) -> int:
        recurrent = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self._numerator, self._denominator, self._last_step)
        )
        if self.persistent_landmark:
            recurrent += sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    self._landmark_key,
                    self._landmark_value,
                    self._landmark_score,
                )
            )
        quality_state = 0
        for tensor in (
            self._quality_previous_raw_score,
            self._quality_current_raw_score,
            self._quality_current_write_score,
        ):
            if tensor is not None:
                quality_state += tensor.numel() * tensor.element_size()
        return recurrent + self.exact_state_bytes() + quality_state

    def _blend_exact(
        self,
        recurrent: Tensor,
        exact: Tensor,
        confidence: Tensor,
    ) -> Tensor:
        if self.exact_attention:
            self._last_exact_log_partition = confidence
            return exact
        gate_bias = self.exact_mix_logits.to(
            device=recurrent.device, dtype=torch.float32
        )
        if recurrent.ndim == 3:
            gate_bias = gate_bias.view(1, -1)
        elif recurrent.ndim == 4:
            gate_bias = gate_bias.view(1, -1, 1)
        else:
            raise ValueError("archive response must be rank 3 or 4")
        confidence_gate = torch.sigmoid(
            gate_bias
            + self.exact_confidence_temperature
            * (confidence.float() - self.exact_confidence_threshold)
        )
        valid = confidence > -0.5
        confidence_gate = torch.where(
            valid, confidence_gate, torch.zeros_like(confidence_gate)
        ).to(recurrent.dtype)
        self._last_exact_gate = confidence_gate.detach()
        return (
            (1.0 - confidence_gate.unsqueeze(-1)) * recurrent
            + confidence_gate.unsqueeze(-1) * exact.to(recurrent.dtype)
        )

    @torch.no_grad()
    def _admit_one(
        self,
        key: Tensor,
        value: Tensor,
        score: Tensor,
        *,
        write_mask: Tensor | None = None,
    ) -> None:
        if self.quality_first and self.quality_block_propagation and self.exact_bank.block_size > 1:
            # Preserve the raw score of the completed block separately. The
            # one-step propagation applies only to the block being written;
            # carrying the propagated value forward would turn one salient
            # block into an unbounded score wave across the stream.
            if self.exact_bank._pending_count == 0:
                self._quality_current_raw_score = score.detach().clone()
                previous = self._quality_previous_raw_score
                if previous is None:
                    previous = torch.full_like(score, -torch.inf)
                self._quality_current_write_score = torch.maximum(score, previous)
            else:
                assert self._quality_current_raw_score is not None
                self._quality_current_raw_score = torch.maximum(
                    self._quality_current_raw_score, score
                )
            score = self._quality_current_write_score
        mask = score >= self.admission_threshold
        if write_mask is not None:
            if write_mask.shape == (key.shape[0],):
                write_mask = write_mask[:, None].expand_as(mask)
            if write_mask.shape != mask.shape:
                raise ValueError("write_mask must match [batch, heads]")
            mask = mask & write_mask.to(device=mask.device, dtype=torch.bool)
        if bool(mask.any()) or self.exact_bank.background_size:
            self.exact_bank.update(
                key,
                value,
                admission_bias=score,
                write_mask=mask,
            )
            if (
                self.quality_first
                and self.quality_block_propagation
                and self.exact_bank.block_size > 1
                and self.exact_bank._pending_count == 0
            ):
                assert self._quality_current_raw_score is not None
                self._quality_previous_raw_score = self._quality_current_raw_score
                self._quality_current_raw_score = None
                self._quality_current_write_score = None

    @torch.no_grad()
    def _quality_first_salience(
        self,
        key: Tensor,
        query: Tensor,
        *,
        key_start: int,
        query_start: int = 0,
        sample_queries: int = 32,
    ) -> Tensor:
        """Estimate future retrieval value for a bounded key tile.

        ``update_read_chunk`` receives a whole prefill block, so future
        queries are available even though the exact bank remains causal at
        serving time.  Sampling a small, evenly spaced set of future queries
        gives the score table a global view without materializing a
        ``tokens x tokens`` attention matrix.  The returned score is a cosine
        similarity in ``[-1, 1]`` with shape ``[batch, heads, tile]``.
        """

        if key.ndim != 4 or query.ndim != 4:
            raise ValueError("key and query must be rank-4 tensors")
        if key.shape[0] != query.shape[0] or key.shape[1] != query.shape[1]:
            raise ValueError("key and query batch/head dimensions must match")
        tile = key.shape[2]
        if tile == 0 or query.shape[2] == 0:
            return key.new_empty(key.shape[0], key.shape[1], tile, dtype=torch.float32)
        query_count = min(int(sample_queries), query.shape[2])
        if query_count <= 0:
            return key.new_full((key.shape[0], key.shape[1], tile), -1.0, dtype=torch.float32)

        # Query event j is token j + window_size. Key i first leaves the
        # local window at that token when j == i. All offsets use event units.
        query_start = int(query_start)
        if query_start < 0:
            raise ValueError("query_start must be non-negative")
        # Anchor sampling to the query interval so splitting a key tile does
        # not change a token's admission score. The per-key mask below excludes
        # queries that precede its archive admission event.
        first_query = query_start
        last_query = query_start + query.shape[2] - 1
        if int(key_start) > last_query:
            return key.new_full(
                (key.shape[0], key.shape[1], tile), -1.0, dtype=torch.float32
            )
        positions_abs = torch.linspace(
            first_query,
            last_query,
            query_count,
            device=query.device,
            dtype=torch.float32,
        ).round().to(torch.long).unique(sorted=True)
        if positions_abs.numel() == 0:
            return key.new_full((key.shape[0], key.shape[1], tile), -1.0, dtype=torch.float32)

        normalized_key = F.normalize(key.float(), dim=-1)
        positions = positions_abs - query_start
        normalized_query = F.normalize(query.index_select(2, positions).float(), dim=-1)
        similarity = torch.einsum(
            "bhtd,bhqd->bhtq", normalized_key, normalized_query
        )
        absolute_key = torch.arange(
            int(key_start), int(key_start) + tile, device=query.device
        )
        # Query event j corresponds to an absolute token at j + window_size.
        # Only queries with event j >= i can read key i from the archive.
        minimum_query = absolute_key
        valid = positions_abs.view(1, 1, 1, -1) >= minimum_query.view(1, 1, -1, 1)
        similarity = similarity.masked_fill(~valid, -1.0)
        return similarity.max(dim=-1).values

    def update(
        self,
        key: Tensor,
        value: Tensor,
        *,
        exact_key: Tensor | None = None,
        exact_query: Tensor | None = None,
    ) -> None:
        if self.causal_coreset:
            self.exact_bank.update(
                key if exact_key is None else exact_key,
                value,
                query=exact_query,
            )
            return
        super().update(key, value)
        with torch.no_grad():
            if self.quality_first and exact_key is not None and exact_query is not None:
                score = F.cosine_similarity(
                    exact_key.float(), exact_query.float(), dim=-1
                )
            else:
                score = self.admission(key, value)
            self._admit_one(key if exact_key is None else exact_key, value, score)

    def _read_exact(self, query: Tensor) -> tuple[Tensor, Tensor]:
        if self.exact_query_correction and not self.active_query_correction and self.query_correction_rank:
            query_f = query.float()
            value_v = self.query_correction_v.to(device=query.device, dtype=torch.float32)
            value_u = self.query_correction_u.to(device=query.device, dtype=torch.float32)
            if query.ndim == 3:
                latent = torch.einsum("bhd,hdr->bhr", query_f, value_v)
                delta = torch.einsum("bhr,hrd->bhd", latent, value_u)
            elif query.ndim == 4:
                latent = torch.einsum("bhtd,hdr->bhtr", query_f, value_v)
                delta = torch.einsum("bhtr,hrd->bhtd", latent, value_u)
            else:
                raise ValueError("query must have rank 3 or 4")
            query = query + delta.to(query.dtype)
        if self.exact_attention:
            return self.exact_bank.read_attention(query)
        if query.ndim == 4:
            return self.exact_bank.read_chunk(query, hard=self.exact_hard_read)
        return self.exact_bank.read(query, hard=self.exact_hard_read)

    def read(self, query: Tensor, *, exact_query: Tensor | None = None) -> Tensor:
        recurrent = super().read(query)
        exact, confidence = self._read_exact(query if exact_query is None else exact_query)
        return self._blend_exact(recurrent, exact, confidence)

    @torch.no_grad()
    def _exact_chunk(
        self,
        key: Tensor,
        value: Tensor,
        query: Tensor,
        score: Tensor,
        *,
        quality_query: Tensor | None = None,
        quality_key_start: int = 0,
        quality_query_start: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Causally read a block with a bounded number of admission events."""

        if self.causal_coreset:
            # The counted coreset commits each event immediately before the
            # matching query.  It deliberately ignores all future-query
            # salience arguments; accepting them here would reintroduce the
            # non-causal prefill side channel this mode is intended to test.
            del score, quality_query, quality_key_start, quality_query_start
            return self.exact_bank.update_read_chunk(key, value, query)

        batch, _, tokens, _ = query.shape
        exact = torch.zeros_like(query)
        confidence = torch.full(
            query.shape[:-1], -1.0, device=query.device, dtype=torch.float32
        )
        tile_size = max(1, int(self.scan_block_size))
        if batch != 1 and not self.quality_first:
            # Calibration can use larger batches. Keep each tile causal while
            # allowing the bank to update all request rows in one tensor call.
            # A tile boundary is only a scheduling boundary; the bank itself
            # remains persistent across tiles.
            for tile_start in range(0, tokens, tile_size):
                tile_end = min(tokens, tile_start + tile_size)
                for index in range(tile_start, tile_end):
                    self._admit_one(
                        key[:, :, index], value[:, :, index], score[:, :, index]
                    )
                    result, conf = self._read_exact(query[:, :, index])
                    exact[:, :, index] = result
                    confidence[:, :, index] = conf
            return exact, confidence

        # Select independently inside each bounded tile.  Selecting once across
        # a million-token prefill would spend the entire admission budget on a
        # few global maxima and could discard a real needle in an earlier region
        # before the model ever reaches its query.  Per-tile selection keeps the
        # state constant while giving every context region the same calibrated
        # opportunity to enter the exact tier.
        #
        # The budget is on *positions*, not head-position pairs.  A predictor
        # can mark the same salient token for several heads, and updating those
        # heads together is one state transition.  This caps Python/Triton
        # scheduling work at K events per tile instead of K events times the
        # number of heads during a long prefill.
        for tile_start in range(0, tokens, tile_size):
            tile_end = min(tokens, tile_start + tile_size)
            tile_score = score[:, :, tile_start:tile_end]
            if self.quality_first:
                # The admission predictor is intentionally untrained in this
                # mode.  Replace its constant bias with an oracle-free
                # future query-key score computed from the current prefill
                # block; the exact bank still only retains a fixed number of
                # slots and uses score replacement globally.
                tile_score = self._quality_first_salience(
                    key[:, :, tile_start:tile_end],
                    query if quality_query is None else quality_query,
                    key_start=quality_key_start + tile_start,
                    query_start=quality_key_start if quality_query is None else quality_query_start,
                )
            admission_score = tile_score if self.quality_first else score[:, :, tile_start:tile_end]
            eligible = tile_score >= self.admission_threshold
            position_score = torch.where(
                eligible,
                tile_score,
                torch.full_like(tile_score, -torch.inf),
            ).max(dim=1).values
            selected_positions = torch.isfinite(position_score)
            if tile_end - tile_start > self.max_inserts_per_chunk:
                top_scores, top_positions = position_score.topk(
                    self.max_inserts_per_chunk, dim=-1
                )
                selected_positions = torch.zeros_like(selected_positions)
                selected_positions.scatter_(
                    1, top_positions, torch.isfinite(top_scores)
                )
            if self.exact_bank.background_size:
                candidates = torch.arange(tile_end - tile_start, device=key.device)
            else:
                candidates = torch.nonzero(
                    selected_positions.any(dim=0), as_tuple=False
                ).flatten()

            if self.quality_prefill_shadow_only:
                # Populate the bounded tier without reading it back into the
                # prefill representation. Admission remains ordered so block
                # replacement and the background reservoir are unchanged.
                for position_tensor in candidates:
                    position = tile_start + int(position_tensor.item())
                    index = position - tile_start
                    self._admit_one(
                        key[:, :, position],
                        value[:, :, position],
                        admission_score[:, :, index],
                        write_mask=(
                            eligible[:, :, index]
                            & selected_positions[:, index, None]
                        ),
                    )
                continue

            cursor = tile_start
            for position_tensor in candidates:
                position = tile_start + int(position_tensor.item())
                if position > cursor:
                    result, conf = self._read_exact(query[:, :, cursor:position])
                    exact[:, :, cursor:position] = result
                    confidence[:, :, cursor:position] = conf
                self._admit_one(
                    key[:, :, position],
                    value[:, :, position],
                    admission_score[:, :, position - tile_start],
                    write_mask=(
                        eligible[:, :, position - tile_start]
                        & selected_positions[:, position - tile_start, None]
                    ),
                )
                cursor = position
            if cursor < tile_end:
                result, conf = self._read_exact(query[:, :, cursor:tile_end])
                exact[:, :, cursor:tile_end] = result
                confidence[:, :, cursor:tile_end] = conf
        return exact, confidence

    @torch.no_grad()
    def update_read_chunk(
        self,
        key: Tensor,
        value: Tensor,
        query: Tensor,
        *,
        output: Tensor | None = None,
        admission_score: Tensor | None = None,
        exact_key: Tensor | None = None,
        exact_query: Tensor | None = None,
        quality_query: Tensor | None = None,
        quality_key_start: int = 0,
        quality_query_start: int = 0,
    ) -> Tensor:
        if admission_score is not None and (
            admission_score.shape != key.shape[:3]
            or admission_score.device != key.device
        ):
            raise ValueError("admission_score must have shape [batch, heads, events] on key.device")
        if exact_key is None:
            exact_key = key
        if exact_query is None:
            exact_query = query
        if quality_key_start < 0:
            raise ValueError("quality_key_start must be non-negative")
        if quality_query_start < 0:
            raise ValueError("quality_query_start must be non-negative")
        if quality_query is not None and (
            quality_query.ndim != 4
            or quality_query.shape[0] != key.shape[0]
            or quality_query.shape[1] != key.shape[1]
            or quality_query.shape[-1] != key.shape[-1]
            or quality_query.device != key.device
        ):
            raise ValueError(
                "quality_query must have shape [batch, heads, tokens, head_dim] on key.device"
            )
        if (
            exact_key.shape != key.shape
            or exact_query.shape != query.shape
            or exact_key.device != key.device
            or exact_query.device != query.device
        ):
            raise ValueError(
                "exact_key/exact_query must match key/query shapes and devices"
            )
        if self.causal_coreset:
            # A causal coreset is the complete remote state for this opt-in
            # path.  Do not spend memory or compute maintaining the obsolete
            # recurrent response that the output would ignore.
            exact, partition = self._exact_chunk(
                exact_key,
                value,
                exact_query,
                self.admission(key, value),
            )
            result = self._blend_exact(torch.zeros_like(exact), exact, partition)
            if output is not None:
                if output.shape != result.shape or output.device != result.device:
                    raise ValueError("output must match query shape and device")
                output.copy_(result)
                return output
            return result

        recurrent = super().update_read_chunk(key, value, query, output=None)
        score = self.admission(key, value) if admission_score is None else admission_score
        # The recurrent path can use position-invariant raw Q/K, while the
        # bounded exact shadow should score the same rotary Q/K used by the
        # model's true local attention.  This side channel removes the phase
        # mismatch without adding any persistent position state.
        exact, confidence = self._exact_chunk(
            exact_key,
            value,
            exact_query,
            score,
            quality_query=quality_query,
            quality_key_start=quality_key_start,
            quality_query_start=quality_query_start,
        )
        if self.quality_prefill_shadow_only:
            # The exact tier is populated as a bounded shadow during prefill,
            # but its future-query selection must not feed back into the
            # representation of earlier prompt tokens.
            result = recurrent
        else:
            result = self._blend_exact(recurrent, exact, confidence)
        if output is not None:
            if output.shape != result.shape or output.device != result.device:
                raise ValueError("output must match query shape and device")
            output.copy_(result)
            return output
        return result


def upgrade_qcc_attention(
    attention: QCCSelfAttention,
    **hybrid_kwargs: Any,
) -> HybridQCCArchive:
    """Replace one attention layer's archive with the hybrid implementation."""

    if not isinstance(attention, QCCSelfAttention):
        raise TypeError("attention must be QCCSelfAttention")
    archive = HybridQCCArchive.from_archive(attention.archive, **hybrid_kwargs)
    attention.archive = archive
    if archive.causal_coreset:
        # Exact mass merging bypasses the learned local/archive gate.  Freeze
        # it alongside the obsolete recurrent parameters so calibration counts
        # only parameters that can change this deployment path.
        for parameter in attention.gate.parameters():
            parameter.requires_grad_(False)
    return archive


def enable_hybrid_retrofit(model: nn.Module, **hybrid_kwargs: Any) -> list[str]:
    """Upgrade every already-patched HF QCC attention layer in ``model``."""

    upgraded: list[str] = []
    for name, module in model.named_modules():
        qcc = getattr(module, "qcc", None)
        if isinstance(qcc, QCCSelfAttention):
            before = qcc.archive
            archive = upgrade_qcc_attention(qcc, **hybrid_kwargs)
            if archive is not before:
                upgraded.append(name)
    if not upgraded and not any(
        isinstance(getattr(module, "qcc", None), QCCSelfAttention)
        for module in model.modules()
    ):
        raise ValueError("model has no QCC retrofit layers; call patch_hf_model first")
    return upgraded


def patch_hf_model_hybrid(
    model: nn.Module,
    *,
    hybrid_kwargs: dict[str, Any] | None = None,
    **patch_kwargs: Any,
) -> list[str]:
    """One-call HF retrofit: regular QCC patch followed by the exact tier."""

    from .retrofit import patch_hf_model

    replaced = patch_hf_model(model, **patch_kwargs)
    enable_hybrid_retrofit(model, **(hybrid_kwargs or {}))
    return replaced


def load_hybrid_retrofit_adapter(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    hybrid_kwargs: dict[str, Any] | None = None,
    **patch_kwargs: Any,
) -> list[str]:
    """Patch the hybrid structure first, then load a QCC-only adapter."""

    replaced = patch_hf_model_hybrid(
        model, hybrid_kwargs=hybrid_kwargs, **patch_kwargs
    )
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError("retrofit checkpoint must contain a state_dict mapping")
    missing, unexpected = model.load_state_dict(state, strict=False)
    # A regular QCC adapter intentionally contains only the shared recurrent
    # archive/gate weights.  The exact tier is an optional runtime upgrade, so
    # its parameters must be allowed to remain at their configured defaults
    # when loading that adapter into a hybrid structure.  Base decay rates are
    # reconstructed from the patched model configuration for the same reason.
    relevant_missing = [
        key
        for key in missing
        if (".qcc.archive." in key
            and not key.endswith("archive.decay_rates")
            and "query_correction_" not in key
            and not key.endswith("archive.query_scale_logits")
            and not (
                ".qcc.archive.exact_" in key
                or ".qcc.archive.exact_bank." in key
                or ".qcc.archive.admission." in key
            ))
        or ".qcc.gate." in key
    ]
    relevant_unexpected = [
        key
        for key in unexpected
        if ".qcc.archive." in key or ".qcc.gate." in key
    ]
    if relevant_missing or relevant_unexpected:
        raise ValueError(
            "hybrid adapter mismatch: "
            f"missing={relevant_missing}, unexpected={relevant_unexpected}"
        )
    return replaced


__all__ = [
    "HybridQCCArchive",
    "LandmarkAdmissionPredictor",
    "enable_hybrid_retrofit",
    "load_hybrid_retrofit_adapter",
    "patch_hf_model_hybrid",
    "upgrade_qcc_attention",
]
