"""QCC-Transformer research prototype."""

from .model import (
    QCCArchive,
    QCCForCausalLM,
    QCCSelfAttention,
    SinusoidalPositionEmbedding,
    count_archive_elements,
)
from .retrofit import HFQCCAttention, QCCCacheHandle, patch_hf_model
from .vllm import QCCVLLMState

__all__ = [
    "QCCArchive",
    "QCCForCausalLM",
    "QCCSelfAttention",
    "SinusoidalPositionEmbedding",
    "count_archive_elements",
    "HFQCCAttention",
    "QCCCacheHandle",
    "patch_hf_model",
    "QCCVLLMState",
]
