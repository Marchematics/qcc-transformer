"""QCC-Transformer research prototype."""

from .model import (
    QCCArchive,
    QCCForCausalLM,
    QCCSelfAttention,
    SinusoidalPositionEmbedding,
    count_archive_elements,
)
from .retrofit import (
    HFQCCAttention,
    QCCCacheHandle,
    load_retrofit_adapter,
    patch_hf_model,
    retrofit_adapter_state,
)
from .vllm import QCCVLLMState

__all__ = [
    "QCCArchive",
    "QCCForCausalLM",
    "QCCSelfAttention",
    "SinusoidalPositionEmbedding",
    "count_archive_elements",
    "HFQCCAttention",
    "QCCCacheHandle",
    "load_retrofit_adapter",
    "patch_hf_model",
    "retrofit_adapter_state",
    "QCCVLLMState",
]
