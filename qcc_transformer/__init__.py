"""QCC-Transformer research prototype."""

from .associative import AssociativeLandmarkState, SetAssociativeLandmarkBank
from .model import (
    QCCArchive,
    QCCForCausalLM,
    QCCSelfAttention,
    SinusoidalPositionEmbedding,
    count_archive_elements,
)
from .retrofit import (
    FidelityReport,
    HFQCCAttention,
    QCCCacheHandle,
    load_retrofit_adapter,
    patch_hf_model,
    retrofit_adapter_state,
    save_retrofit_adapter,
    compare_logits,
    reset_hf_qcc_cache,
)
from .vllm import QCCVLLMBackend, QCCVLLMState
from .vllm_plugin import register_vllm_backend

__all__ = [
    "AssociativeLandmarkState",
    "SetAssociativeLandmarkBank",
    "QCCArchive",
    "QCCForCausalLM",
    "QCCSelfAttention",
    "SinusoidalPositionEmbedding",
    "count_archive_elements",
    "HFQCCAttention",
    "FidelityReport",
    "QCCCacheHandle",
    "load_retrofit_adapter",
    "patch_hf_model",
    "retrofit_adapter_state",
    "save_retrofit_adapter",
    "compare_logits",
    "reset_hf_qcc_cache",
    "QCCVLLMState",
    "QCCVLLMBackend",
    "register_vllm_backend",
]
