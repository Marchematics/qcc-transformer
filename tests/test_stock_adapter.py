import pytest
import torch

from qcc_transformer.stock_adapter import (
    checkpoint_state_dict,
    extract_archive_parameters,
    layer_index_from_name,
)


def _fake_layer(prefix: str, layer: int):
    root = f"{prefix}.layers.{layer}.self_attn.qcc.archive."
    return {
        root + "codes": torch.zeros(1),
        root + "mix_logits": torch.zeros(1),
        root + "exact_bank.set_codes": torch.zeros(1),
        root + "admission.key_weight": torch.zeros(1),
        root + "admission.value_weight": torch.zeros(1),
        root + "admission.bias": torch.zeros(1),
        root + "exact_mix_logits": torch.zeros(1),
    }


def test_layer_index_is_stable_across_hf_and_vllm_prefixes():
    assert layer_index_from_name("model.layers.7.self_attn") == 7
    assert layer_index_from_name("model.model.layers.31.self_attn") == 31


def test_extract_archive_parameters_uses_layer_index_not_full_prefix():
    state = {}
    state.update(_fake_layer("model", 2))
    state.update(_fake_layer("model", 3))
    got = extract_archive_parameters(state, "model.model.layers.3.self_attn")
    assert set(got) == {
        "codes", "mix_logits", "exact_bank.set_codes", "admission.key_weight",
        "admission.value_weight", "admission.bias", "exact_mix_logits"
    }


def test_incomplete_hybrid_adapter_fails_closed():
    state = _fake_layer("model", 1)
    state.pop("model.layers.1.self_attn.qcc.archive.exact_mix_logits")
    with pytest.raises(ValueError, match="incomplete"):
        extract_archive_parameters(state, "model.layers.1.self_attn")


def test_checkpoint_state_dict_unwraps_payload():
    payload = {"state_dict": {"x": torch.ones(1)}, "metadata": {"model": "x"}}
    assert checkpoint_state_dict(payload)["x"].item() == 1
