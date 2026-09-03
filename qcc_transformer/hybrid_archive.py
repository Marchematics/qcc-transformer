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
        exact_num_sets: int = 128,
        exact_ways: int = 4,
        exact_probe_sets: int | None = None,
        admission_threshold: float = 0.0,
        admission_bias_init: float = -4.0,
        max_inserts_per_chunk: int = 8,
        exact_confidence_threshold: float = 0.60,
        exact_confidence_temperature: float = 20.0,
        exact_mix_bias_init: float = -4.0,
    ) -> None:
        if active_codes is not None or lazy_decay:
            raise ValueError(
                "HybridQCCArchive currently requires the dense base archive; "
                "sparse/lazy fallbacks can double-dispatch virtual updates"
            )
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
        probes = min(4, exact_num_sets) if exact_probe_sets is None else exact_probe_sets
        self.exact_bank = SetAssociativeLandmarkBank(
            num_heads=num_heads,
            head_dim=head_dim,
            num_sets=exact_num_sets,
            ways=exact_ways,
            probe_sets=probes,
            diversity_weight=0.10,
        )
        # Hybrid admission is supplied by the teacher-trained predictor below.
        # Keep the bank's legacy internal score neutral and frozen.
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
        return recurrent + self.exact_state_bytes()

    def _blend_exact(
        self,
        recurrent: Tensor,
        exact: Tensor,
        confidence: Tensor,
    ) -> Tensor:
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
        mask = score >= self.admission_threshold
        if write_mask is not None:
            if write_mask.shape == (key.shape[0],):
                write_mask = write_mask[:, None].expand_as(mask)
            if write_mask.shape != mask.shape:
                raise ValueError("write_mask must match [batch, heads]")
            mask = mask & write_mask.to(device=mask.device, dtype=torch.bool)
        if bool(mask.any()):
            self.exact_bank.update(
                key,
                value,
                admission_bias=score,
                write_mask=mask,
            )

    def update(self, key: Tensor, value: Tensor) -> None:
        super().update(key, value)
        with torch.no_grad():
            score = self.admission(key, value)
            self._admit_one(key, value, score)

    def read(self, query: Tensor) -> Tensor:
        recurrent = super().read(query)
        exact, confidence = self.exact_bank.read(query, hard=True)
        return self._blend_exact(recurrent, exact, confidence)

    @torch.no_grad()
    def _exact_chunk(
        self,
        key: Tensor,
        value: Tensor,
        query: Tensor,
        score: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Causally read a block with at most K associative state mutations."""

        batch, _, tokens, _ = query.shape
        exact = torch.zeros_like(query)
        confidence = torch.full(
            query.shape[:-1], -1.0, device=query.device, dtype=torch.float32
        )
        if batch != 1:
            # Calibration can use larger batches. Keep this path exact and
            # simple; serving keeps one state per logical vLLM request (B=1).
            for index in range(tokens):
                self._admit_one(key[:, :, index], value[:, :, index], score[:, :, index])
                result, conf = self.exact_bank.read(query[:, :, index], hard=True)
                exact[:, :, index] = result
                confidence[:, :, index] = conf
            return exact, confidence

        # Select independently per head.  A global token top-k lets one head
        # consume the whole budget and silently drops retrieval-critical events
        # for the other heads.  The budget is fixed per head, so total mutable
        # state remains bounded by the bank geometry.
        selected = torch.zeros_like(score[0], dtype=torch.bool)
        for head in range(score.shape[1]):
            head_score = score[0, head]
            eligible = head_score >= self.admission_threshold
            indices = torch.nonzero(eligible, as_tuple=False).flatten()
            if indices.numel() > self.max_inserts_per_chunk:
                keep = head_score[indices].topk(self.max_inserts_per_chunk).indices
                indices = indices[keep]
            selected[head, indices] = True
        candidates = torch.nonzero(selected.any(dim=0), as_tuple=False).flatten()

        cursor = 0
        for position_tensor in candidates:
            position = int(position_tensor.item())
            if position > cursor:
                result, conf = self.exact_bank.read_chunk(
                    query[:, :, cursor:position], hard=True
                )
                exact[:, :, cursor:position] = result
                confidence[:, :, cursor:position] = conf
            self._admit_one(
                key[:, :, position],
                value[:, :, position],
                score[:, :, position],
                write_mask=selected[:, position].unsqueeze(0),
            )
            cursor = position
        if cursor < tokens:
            result, conf = self.exact_bank.read_chunk(query[:, :, cursor:], hard=True)
            exact[:, :, cursor:] = result
            confidence[:, :, cursor:] = conf
        return exact, confidence

    @torch.no_grad()
    def update_read_chunk(
        self,
        key: Tensor,
        value: Tensor,
        query: Tensor,
        *,
        output: Tensor | None = None,
    ) -> Tensor:
        recurrent = super().update_read_chunk(key, value, query, output=None)
        score = self.admission(key, value)
        exact, confidence = self._exact_chunk(key, value, query, score)
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
    relevant_missing = [
        key
        for key in missing
        if ".qcc.archive." in key or ".qcc.gate." in key
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
