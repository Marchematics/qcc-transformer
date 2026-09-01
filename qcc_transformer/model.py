"""Reference PyTorch implementation of Query-Compiled Cache attention.

This module intentionally favors readable streaming semantics over kernel-level
performance. The archive state is the object to replace with a fused Triton or
CUDA implementation once the approximation has passed quality tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class QCCState:
    """Mutable state returned by :class:`QCCArchive` for inspection/checkpointing."""

    numerator: Tensor
    denominator: Tensor


class QCCArchive(nn.Module):
    """Constant-size, multi-timescale softmax response memory.

    Args:
        num_heads: Number of key/value heads.
        head_dim: Dimension of each key/value head.
        num_codes: Number of learned long-range query prototypes.
        decay_rates: Values in ``(0, 1)``. Each rate provides a different
            exponential time scale.
        window_size: Number of exact recent tokens. A newly evicted token is
            inserted with ``decay_rate ** window_size`` age compensation.

    The archive stores a response for each (head, code, decay rate), rather
    than a key/value pair per historical token. Its memory is independent of
    sequence length.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_codes: int = 16,
        decay_rates: tuple[float, ...] = (0.995, 0.98, 0.94, 0.85),
        window_size: int = 128,
    ) -> None:
        super().__init__()
        if num_heads <= 0 or head_dim <= 0 or num_codes <= 0:
            raise ValueError("head dimensions and num_codes must be positive")
        rates = torch.tensor(decay_rates, dtype=torch.float32)
        if rates.numel() == 0 or not bool(torch.all((rates > 0) & (rates < 1))):
            raise ValueError("decay_rates must contain values strictly between 0 and 1")
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_codes = num_codes
        self.num_scales = int(rates.numel())
        self.window_size = window_size
        self.register_buffer("decay_rates", rates, persistent=True)
        self.codes = nn.Parameter(torch.randn(num_heads, num_codes, head_dim) / math.sqrt(head_dim))
        self.mix_logits = nn.Parameter(torch.zeros(num_heads, num_codes, self.num_scales))
        self.reset_state(batch_size=1, device=rates.device, dtype=torch.float32)

    def reset_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Reset archive state before a new independent stream."""

        device = device or self.codes.device
        # Accumulation in fp32 avoids long-stream drift when activations are fp16/bf16.
        state_dtype = dtype if dtype in (torch.float32, torch.float64) else torch.float32
        self._numerator = torch.zeros(
            batch_size,
            self.num_heads,
            self.num_codes,
            self.num_scales,
            self.head_dim,
            device=device,
            dtype=state_dtype,
        )
        self._denominator = torch.zeros(
            batch_size,
            self.num_heads,
            self.num_codes,
            self.num_scales,
            device=device,
            dtype=state_dtype,
        )

    @property
    def state(self) -> QCCState:
        return QCCState(self._numerator, self._denominator)

    @torch.no_grad()
    def detach_state(self) -> None:
        """Detach streaming state, useful between training chunks."""

        self._numerator = self._numerator.detach()
        self._denominator = self._denominator.detach()

    def update(self, key: Tensor, value: Tensor) -> None:
        """Insert one evicted token per batch/head into the archive.

        ``key`` and ``value`` have shape ``[batch, heads, head_dim]``. Existing
        state is decayed by one step; the inserted token is aged by the exact
        local-window length.
        """

        if key.shape != value.shape or key.ndim != 3:
            raise ValueError("key and value must both have shape [batch, heads, head_dim]")
        bsz, heads, dim = key.shape
        if heads != self.num_heads or dim != self.head_dim:
            raise ValueError("key shape does not match archive configuration")
        if self._numerator.shape[0] != bsz:
            self.reset_state(bsz, device=key.device)

        rates = self.decay_rates.to(device=key.device, dtype=self._numerator.dtype)
        codes = self.codes.to(dtype=self._numerator.dtype)
        score = torch.einsum("bhd,hmd->bhm", key.to(self._numerator.dtype), codes)
        score = score / math.sqrt(self.head_dim)
        # Clipping bounds the reference implementation. A fused kernel should
        # use per-code log rescaling instead of clipping for higher fidelity.
        content_weight = torch.exp(score.clamp(min=-20.0, max=10.0))
        age = rates.pow(self.window_size).view(1, 1, 1, self.num_scales)

        denominator_decay = rates.view(1, 1, 1, self.num_scales)
        numerator_decay = rates.view(1, 1, 1, self.num_scales, 1)
        numerator_add = (
            content_weight.unsqueeze(-1).unsqueeze(-1)
            * age.unsqueeze(-1)
            * value.to(self._numerator.dtype)[:, :, None, None, :]
        )
        denominator_add = content_weight.unsqueeze(-1) * age
        if torch.is_grad_enabled():
            # Functional assignments preserve a differentiable state during
            # teacher training. Inference uses the cheaper in-place path.
            self._denominator = self._denominator * denominator_decay + denominator_add
            self._numerator = self._numerator * numerator_decay + numerator_add
        else:
            self._denominator.mul_(denominator_decay).add_(denominator_add)
            self._numerator.mul_(numerator_decay).add_(numerator_add)

    def read(self, query: Tensor) -> Tensor:
        """Read archive response for queries of shape ``[batch, heads, head_dim]``."""

        if query.ndim != 3 or query.shape[1:] != (self.num_heads, self.head_dim):
            raise ValueError("query must have shape [batch, heads, head_dim]")
        if query.shape[0] != self._numerator.shape[0]:
            self.reset_state(query.shape[0], device=query.device)

        denom = self._denominator.clamp_min(1e-8)
        response = self._numerator / denom.unsqueeze(-1)
        mix = F.softmax(self.mix_logits, dim=-1).to(response.dtype)
        response = torch.einsum("hmj,bhmjd->bhmd", mix, response)
        routing = torch.einsum(
            "bhd,hmd->bhm", query.to(self.codes.dtype), self.codes
        ) / math.sqrt(self.head_dim)
        routing = F.softmax(routing, dim=-1).to(response.dtype)
        return torch.einsum("bhm,bhmd->bhd", routing, response)


class QCCSelfAttention(nn.Module):
    """Causal attention with exact local window and QCC long-range archive."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        num_codes: int = 16,
        num_scales: int = 4,
        window_size: int = 128,
        use_archive: bool = True,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if num_scales <= 0:
            raise ValueError("num_scales must be positive")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.use_archive = use_archive
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, num_heads)
        # Log-spaced rates cover short, medium, and long historical scales.
        rates = tuple(1.0 - 10.0 ** (-x) for x in torch.linspace(1.3, 3.5, num_scales).tolist())
        self.archive = QCCArchive(
            num_heads, self.head_dim, num_codes, rates, window_size
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        bsz, length, _ = x.shape
        return x.view(bsz, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden: Tensor, *, reset_state: bool = True) -> Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [batch, sequence, d_model]")
        bsz, length, _ = hidden.shape
        q = self._split_heads(self.q_proj(hidden))
        k = self._split_heads(self.k_proj(hidden))
        v = self._split_heads(self.v_proj(hidden))
        if reset_state or self.archive._numerator.shape[0] != bsz:
            self.archive.reset_state(bsz, device=hidden.device)

        local_keys: list[Tensor] = []
        local_values: list[Tensor] = []
        outputs: list[Tensor] = []
        scale = 1.0 / math.sqrt(self.head_dim)
        for t in range(length):
            kt, vt = k[:, :, t], v[:, :, t]
            local_keys.append(kt)
            local_values.append(vt)
            if len(local_keys) > self.window_size:
                self.archive.update(local_keys.pop(0), local_values.pop(0))

            lk = torch.stack(local_keys, dim=2)
            lv = torch.stack(local_values, dim=2)
            local_logits = torch.einsum("bhd,bhld->bhl", q[:, :, t], lk) * scale
            local_prob = F.softmax(local_logits, dim=-1)
            local_out = torch.einsum("bhl,bhld->bhd", local_prob, lv)
            if self.use_archive and t >= self.window_size:
                archive_out = self.archive.read(q[:, :, t])
                gate = torch.sigmoid(self.gate(hidden[:, t])).unsqueeze(-1)
                head_out = gate * local_out + (1.0 - gate) * archive_out
            else:
                head_out = local_out
            outputs.append(head_out.transpose(1, 2).reshape(bsz, self.d_model))

        return self.out_proj(torch.stack(outputs, dim=1))


class QCCDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, **attention_kwargs: object) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = QCCSelfAttention(d_model, num_heads, **attention_kwargs)
        self.norm2 = nn.LayerNorm(d_model)
        hidden_dim = 4 * d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, *, reset_state: bool) -> Tensor:
        x = x + self.attention(self.norm1(x), reset_state=reset_state)
        return x + self.mlp(self.norm2(x))


class QCCForCausalLM(nn.Module):
    """Small decoder-only model suitable for architecture experiments."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_position_embeddings: int = 4096,
        window_size: int = 128,
        num_codes: int = 16,
        dropout: float = 0.0,
        use_archive: bool = True,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_position_embeddings, d_model)
        self.layers = nn.ModuleList(
            QCCDecoderLayer(
                d_model,
                num_heads,
                dropout,
                window_size=window_size,
                num_codes=num_codes,
                use_archive=use_archive,
            )
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.max_position_embeddings = max_position_embeddings

    def forward(self, input_ids: Tensor, *, reset_state: bool = True) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, length = input_ids.shape
        if length > self.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for index, layer in enumerate(self.layers):
            x = layer(x, reset_state=reset_state or index > 0)
        return self.lm_head(self.norm(x))


def count_archive_elements(model: nn.Module) -> int:
    """Return persistent archive elements per batch across all attention layers."""

    return sum(
        layer.attention.archive.num_heads
        * layer.attention.archive.num_codes
        * layer.attention.archive.num_scales
        * (layer.attention.archive.head_dim + 1)
        for layer in model.layers
        if isinstance(layer, QCCDecoderLayer)
    )
