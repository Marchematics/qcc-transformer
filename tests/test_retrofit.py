from types import SimpleNamespace

import pytest
import torch
from torch import nn

from qcc_transformer.retrofit import (
    compare_logits,
    load_retrofit_adapter,
    patch_hf_model,
    retrofit_adapter_state,
    save_retrofit_adapter,
    reset_hf_qcc_cache,
    qcc_runtime_state_bytes,
    _synchronize_native_prefix_rebuild,
)
from qcc_transformer.hybrid_archive import patch_hf_model_hybrid
from benchmarks.calibrate_hf_admission import _load_initial_adapter


class _Attention(nn.Module):
    def __init__(self, *, kv_heads: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=False)
        kv_width = 16 if kv_heads == 4 else 8
        self.k_proj = nn.Linear(16, kv_width, bias=False)
        self.v_proj = nn.Linear(16, kv_width, bias=False)
        self.o_proj = nn.Linear(16, 16, bias=False)
        self.num_heads = 4
        self.num_key_value_heads = kv_heads


class _Model(nn.Module):
    def __init__(self, *, kv_heads: int = 4):
        super().__init__()
        self.config = SimpleNamespace(
            max_position_embeddings=64,
            rope_theta=None,
            num_attention_heads=4,
            num_key_value_heads=kv_heads,
        )
        self.attn = _Attention(kv_heads=kv_heads)


class _TwoLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            max_position_embeddings=64,
            rope_theta=None,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        self.layers = nn.ModuleList([_Attention(), _Attention()])


class _FusedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(16, 32, bias=False)  # Q=16, K=V=8
        self.o_proj = nn.Linear(16, 16, bias=False)
        self.num_key_value_heads = 1
        self.head_dim = 8


class _FusedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            max_position_embeddings=64,
            rope_theta=None,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
        self.attn = _FusedAttention()


class _ModernCache:
    def __init__(self):
        self.length = 0

    def get_seq_length(self, layer_idx=0):
        del layer_idx
        return self.length

    def get_usable_length(self, new_seq_length, layer_idx=0):
        del new_seq_length, layer_idx
        return self.length


def test_runtime_state_counts_backing_storage_once():
    model = _Model()
    patch_hf_model(model, window_size=4, num_codes=4, use_triton=False)
    before = qcc_runtime_state_bytes(model)['total_bytes']
    backing = torch.zeros(1024)
    model.attn.qcc._archive_read_cache = backing[:1]
    model.attn.qcc._archive_query_cache = backing[1:2]
    model.attn.qcc._fused_projection_weight = torch.zeros(2000)
    after = qcc_runtime_state_bytes(model)['total_bytes']
    assert after - before == backing.untyped_storage().nbytes()


def test_patch_hf_reads_transformers5_rope_parameters():
    model = _Model()
    model.config.rope_theta = None
    model.config.rope_parameters = {"rope_theta": 1_000_000.0}
    patch_hf_model(model, window_size=4, num_codes=4, use_triton=False)
    assert model.attn.qcc.rope_theta == 1_000_000.0


def test_patch_hf_assigns_stable_layer_indices():
    model = _TwoLayerModel()
    assert patch_hf_model(model, window_size=4, num_codes=4, use_triton=False) == [
        "layers.0",
        "layers.1",
    ]
    assert [layer.qcc._qcc_layer_index for layer in model.layers] == [0, 1]
    assert torch.allclose(
        model.layers[0].qcc.gate.bias,
        torch.full_like(model.layers[0].qcc.gate.bias, 2.0),
    )


def test_patch_hf_preserves_parent_training_mode():
    model = _Model().eval()
    patch_hf_model(model, window_size=4, num_codes=4, use_triton=False)
    assert model.attn.training is False
    model = _Model().train()
    patch_hf_model(model, window_size=4, num_codes=4, use_triton=False)
    assert model.attn.training is True


def test_patch_hf_gate_bias_ablation_is_explicit():
    model = _Model()
    patch_hf_model(model, window_size=4, num_codes=4, use_triton=False, gate_bias_init=0.0)
    assert torch.allclose(model.attn.qcc.gate.bias, torch.zeros_like(model.attn.qcc.gate.bias))


def test_patch_hf_archive_norm_gating_is_explicit():
    model = _Model()
    patch_hf_model(model, window_size=4, num_codes=4, use_triton=False, archive_norm_gating=True)
    assert model.attn.qcc.archive_norm_gating is True


def test_patch_hf_archive_scan_block_size_is_forwarded():
    model = _Model()
    patch_hf_model(
        model,
        window_size=4,
        num_codes=4,
        archive_scan_block_size=7,
        use_triton=False,
    )
    assert model.attn.qcc.archive.scan_block_size == 7


def test_patch_hf_local_attention_backend_is_forwarded():
    model = _Model()
    patch_hf_model(
        model,
        window_size=4,
        num_codes=4,
        local_attention_backend="eager",
        use_triton=False,
    )
    assert model.attn.qcc.local_attention_backend == "eager"


def test_patch_hf_quality_selector_uses_query_correction_rank():
    model = _Model()
    patch_hf_model(
        model,
        window_size=4,
        num_codes=4,
        archive_global_normalization=True,
        archive_position_invariant=True,
        archive_query_correction_rank=8,
        use_triton=False,
    )
    archive = model.attn.qcc.archive
    assert archive.query_correction_rank == 8
    assert archive.query_scale_logits.shape == (4, 8, archive.num_scales)


def test_archive_norm_gating_suppresses_scale_mismatch():
    model = _Model()
    patch_hf_model(model, window_size=4, num_codes=4, use_triton=False, archive_norm_gating=True)
    qcc = model.attn.qcc
    local = torch.ones(1, 4, 1, 4)
    archive = torch.full_like(local, 100.0)
    gate = torch.full_like(local, 0.5)
    mixed = qcc._mix_local_archive(local, archive, gate)
    assert mixed.max().item() < 2.0


def test_patch_hf_supports_fused_qkv_attention():
    model = _FusedModel()
    assert patch_hf_model(model, window_size=4, num_codes=4, use_triton=False, kv_head_policy="repeat") == ["attn"]
    output, _, _ = model.attn(torch.randn(1, 6, 16), use_cache=True)
    assert output.shape == (1, 6, 16)


def test_fused_qkv_projection_uses_one_source_and_repeats_gqa():
    model = _FusedModel()
    patch_hf_model(model, window_size=4, num_codes=4, use_triton=False, kv_head_policy="repeat")
    qcc = model.attn.qcc
    hidden = torch.randn(1, 3, 16)
    q, k, v, _ = qcc._project_qkv_gate(hidden)
    assert q.shape == k.shape == v.shape == (1, 3, 16)
    # One KV head is repeated across both query heads, so the two head slices
    # must be identical before any attention/position transform.
    assert torch.equal(k.view(1, 3, 2, 8)[:, :, 0], k.view(1, 3, 2, 8)[:, :, 1])


def test_quantized_projection_uses_native_forward_instead_of_weight_concat():
    model = _Model()
    patch_hf_model(model, window_size=4, num_codes=4, use_triton=False)
    # bitsandbytes Linear4bit exposes this marker while still subclassing
    # nn.Linear. The retrofit must avoid treating its packed weight as dense.
    model.attn.qcc.q_proj.quant_state = object()
    q, k, v, gate = model.attn.qcc._project_qkv_gate(torch.randn(1, 3, 16))
    assert q.shape == k.shape == v.shape == (1, 3, 16)
    assert gate.shape == (1, 3, 4)


def test_patch_hf_attention_preserves_prefill_and_decode_shapes():
    model = _Model()
    assert patch_hf_model(model, window_size=4, num_codes=4, use_triton=False) == ["attn"]
    assert model.attn.qcc.archive_position_invariant
    output, _, cache = model.attn(torch.randn(2, 8, 16), use_cache=True)
    assert output.shape == (2, 8, 16)
    assert cache is not None and cache.get_seq_length() == 8
    output, _, cache = model.attn(
        torch.randn(2, 1, 16), past_key_value=cache, use_cache=True
    )
    assert output.shape == (2, 1, 16)
    assert cache.get_seq_length() == 9


def test_hf_prefill_is_chunked_when_prompt_exceeds_bound():
    model = _Model()
    patch_hf_model(
        model,
        window_size=4,
        num_codes=4,
        use_triton=False,
        prefill_chunk_size=3,
    )
    hidden = torch.randn(1, 10, 16)
    output, _, cache = model.attn(hidden, use_cache=True)
    assert output.shape == hidden.shape
    assert cache is not None and cache.get_seq_length() == hidden.shape[1]
    assert model.attn.qcc._chunk_key_scratch is not None
    assert model.attn.qcc._chunk_key_scratch.shape[2] <= 3 + 4


@pytest.mark.parametrize('tail, expected_length, expected_start', [(None, 6, 0), (2, 2, 4)])
@pytest.mark.parametrize('external_rope', [False, True])
def test_quality_first_prefill_keeps_future_queries_visible(tail, expected_length, expected_start, external_rope):
    model = _Model()
    patch_hf_model_hybrid(
        model,
        window_size=4,
        num_codes=4,
        use_triton=False,
        hybrid_kwargs={"exact_num_sets": 2, "exact_ways": 2, "quality_first": True, "quality_query_tail": tail},
    )
    qcc = model.attn.qcc
    calls = []
    quality_queries = []
    query_starts = []
    rope_lengths = []
    projection_lengths = []
    project = qcc._project_qkv_gate
    def observed_project(hidden):
        projection_lengths.append(hidden.shape[1])
        return project(hidden)
    qcc._project_qkv_gate = observed_project
    original = qcc.step_chunk

    def counted(hidden, **kwargs):
        calls.append(int(hidden.shape[1]))
        quality_queries.append(kwargs.get("quality_query"))
        query_starts.append(kwargs.get("quality_query_start"))
        rope_lengths.append(kwargs.get("rope_sequence_length"))
        return original(hidden, **kwargs)

    qcc.step_chunk = counted
    hidden = torch.randn(1, 10, 16)
    positions = torch.arange(10).unsqueeze(0)
    angles = torch.randn(1, 10, qcc.head_dim)
    embeddings = (angles.cos(), angles.sin()) if external_rope else None
    output, _, cache = model.attn(hidden, use_cache=True, position_ids=positions,
                                 position_embeddings=embeddings)
    assert output.shape == hidden.shape
    assert cache is not None and cache.get_seq_length() == hidden.shape[1]
    # The quality-first path keeps normal chunks but exposes the complete
    # future query stream to every salience decision.
    assert calls == [4, 4, 2]
    assert [query.shape[2] for query in quality_queries] == [expected_length] * 3
    # Offsets use archive-query events, not absolute token positions.
    assert query_starts == [expected_start] * 3
    assert rope_lengths == [10] * 3
    assert projection_lengths[0] == expected_length
    assert max(projection_lengths[1:]) <= 4
    # Compare the sliced projection against the previous full-prompt equation.
    full_query = qcc._split_heads(project(hidden)[0])
    full_query, _ = qcc._apply_rope(full_query, full_query, positions, embeddings)
    for query in quality_queries:
        torch.testing.assert_close(query, full_query[:, :, -expected_length:])


def test_patch_hf_supports_modern_shared_cache_call():
    model = _Model()
    patch_hf_model(model, window_size=4, num_codes=4, use_triton=False)
    output = model.attn(
        torch.randn(1, 4, 16), past_key_values=object(), use_cache=True
    )
    assert len(output) == 2
    assert output[0].shape == (1, 4, 16)


def test_modern_generation_without_physical_cache_keeps_qcc_state():
    model = _Model()
    patch_hf_model(model, window_size=4, num_codes=4, use_triton=False)
    first = model.attn(
        torch.randn(1, 4, 16), past_key_values=object(), use_cache=True
    )
    assert len(first) == 2
    assert model.attn.qcc._seen_tokens == 4
    # A QCC retrofit intentionally returns no physical past-KV object.  A
    # subsequent framework call may therefore pass None; that must not reset
    # the bounded archive between generated tokens.
    second = model.attn(
        torch.randn(1, 1, 16), past_key_values=None, use_cache=True
    )
    assert len(second) == 2
    assert model.attn.qcc._seen_tokens == 5


def test_modern_cache_reports_logical_qcc_length_without_physical_kv():
    model = _Model()
    patch_hf_model(model, window_size=4, num_codes=4, use_triton=False)
    cache = _ModernCache()
    first = model.attn(
        torch.randn(1, 4, 16), past_key_values=cache, use_cache=True
    )
    assert len(first) == 2
    assert cache.get_seq_length() == 4
    assert cache.get_usable_length(1) == 4
    second = model.attn(
        torch.randn(1, 1, 16), past_key_values=cache, use_cache=True
    )
    assert len(second) == 2
    assert cache.get_seq_length() == 5
    assert cache.get_usable_length(1) == 5


def test_native_prepare_cache_rebuild_resets_qcc_once():
    model = _Model()
    patch_hf_model(model, window_size=4, num_codes=4, use_triton=False)
    model.attn.qcc._seen_tokens = 7
    dropped = {"value": False}

    def prepare(input_ids, past_key_values=None):
        del input_ids
        return {"past_key_values": None if dropped["value"] else past_key_values}

    model.prepare_inputs_for_generation = prepare
    _synchronize_native_prefix_rebuild(model)
    model.prepare_inputs_for_generation(torch.zeros(1, 3, dtype=torch.long), object())
    assert model.attn.qcc._seen_tokens == 7
    dropped["value"] = True
    model.prepare_inputs_for_generation(torch.zeros(1, 3, dtype=torch.long), object())
    assert model.attn.qcc._seen_tokens == 0
    model.attn.qcc._seen_tokens = 9
    model.prepare_inputs_for_generation(torch.zeros(1, 3, dtype=torch.long), None)
    assert model.attn.qcc._seen_tokens == 9


def test_patch_hf_exposes_differentiable_calibration_path():
    model = _Model()
    patch_hf_model(model, window_size=4, num_codes=4, use_triton=False)
    model.attn.train()
    output, _, _ = model.attn(torch.randn(1, 8, 16), use_cache=False)
    output.sum().backward()
    assert model.attn.qcc.archive.codes.grad is not None


def test_retrofit_adapter_round_trip(tmp_path):
    source = _Model()
    patch_hf_model(source, window_size=4, num_codes=4, use_triton=False)
    state = retrofit_adapter_state(source)
    path = tmp_path / "adapter.pt"
    torch.save({"state_dict": state}, path)
    target = _Model()
    replaced = load_retrofit_adapter(
        target, path, window_size=4, num_codes=4, use_triton=False
    )
    assert replaced == ["attn"]
    for name, value in state.items():
        assert torch.equal(value, dict(target.named_parameters())[name].cpu())


def test_legacy_adapter_without_query_residual_remains_loadable(tmp_path):
    source = _Model()
    patch_hf_model(source, window_size=4, num_codes=4, use_triton=False)
    state = retrofit_adapter_state(source)
    legacy = {
        name: value
        for name, value in state.items()
        if "query_correction_" not in name
    }
    path = tmp_path / "legacy_adapter.pt"
    torch.save({"state_dict": legacy}, path)
    target = _Model()
    load_retrofit_adapter(target, path, window_size=4, num_codes=4, use_triton=False)
    for name, value in legacy.items():
        assert torch.equal(value, dict(target.named_parameters())[name].cpu())


def test_regular_adapter_can_seed_hybrid_admission_calibration(tmp_path):
    source = _Model()
    patch_hf_model(source, window_size=4, num_codes=4, use_triton=False)
    path = tmp_path / "regular_adapter.pt"
    torch.save({"state_dict": retrofit_adapter_state(source)}, path)

    target = _Model()
    patch_hf_model_hybrid(
        target,
        window_size=4,
        num_codes=4,
        use_triton=False,
        hybrid_kwargs={"exact_num_sets": 4, "exact_ways": 2},
    )
    _load_initial_adapter(target, path)
    source_state = retrofit_adapter_state(source)
    target_state = dict(target.named_parameters())
    for name, value in source_state.items():
        torch.testing.assert_close(target_state[name].cpu(), value)


def test_patch_hf_rejects_grouped_query_attention():
    with pytest.raises(ValueError, match="GQA/MQA"):
        patch_hf_model(_Model(kv_heads=2), use_triton=False)


def test_patch_hf_gqa_repeat_policy_and_idempotence():
    model = _Model(kv_heads=2)
    replaced = patch_hf_model(model, use_triton=False, kv_head_policy="repeat")
    assert replaced == ["attn"]
    assert patch_hf_model(model, use_triton=False, kv_head_policy="repeat") == []
    assert reset_hf_qcc_cache(model) == 1


def test_gqa_projection_fuses_raw_qkv_before_repeat():
    model = _Model(kv_heads=2)
    patch_hf_model(model, use_triton=False, kv_head_policy="repeat")
    hidden = torch.randn(1, 3, 16)
    q, k, v, gate = model.attn.qcc._project_qkv_gate(hidden)
    base = model.attn.base_attention
    expected_q = base.q_proj(hidden)
    expected_k = base.k_proj(hidden).view(1, 3, 2, 4).repeat_interleave(2, dim=-2).reshape(1, 3, 16)
    expected_v = base.v_proj(hidden).view(1, 3, 2, 4).repeat_interleave(2, dim=-2).reshape(1, 3, 16)
    assert q.shape == k.shape == v.shape == (1, 3, 16)
    torch.testing.assert_close(q, expected_q)
    torch.testing.assert_close(k, expected_k)
    torch.testing.assert_close(v, expected_v)
    assert gate.shape == (1, 3, 4)


def test_fused_projection_casts_fp32_adapter_gate_to_backbone_dtype():
    model = _Model()
    patch_hf_model(model, window_size=4, num_codes=4, use_triton=False)
    model.attn.qcc.gate.double()
    hidden = torch.randn(1, 3, 16)
    q, k, v, gate = model.attn.qcc._project_qkv_gate(hidden)
    assert q.dtype == k.dtype == v.dtype == gate.dtype == hidden.dtype


def test_fidelity_gate_and_adapter_manifest(tmp_path):
    reference = torch.randn(2, 3, 7)
    assert compare_logits(reference, reference).passed
    report = compare_logits(reference, -reference)
    assert not report.passed
    model = _Model()
    patch_hf_model(model, use_triton=False)
    path = save_retrofit_adapter(model, tmp_path / "adapter.pt", base_model="fixture")
    payload = torch.load(path, map_location="cpu")
    assert payload["format"] == "qcc-retrofit-v1"
    assert payload["metadata"]["base_model"] == "fixture"


def test_exact_hf_releases_scratch_without_changing_continuation():
    import copy
    torch.manual_seed(902)
    model = _Model().eval()
    patch_hf_model_hybrid(model, window_size=4, prefill_chunk_size=3,
        use_triton=False, hybrid_kwargs=dict(quality_first=True,
        exact_attention=True, exact_num_sets=2, exact_ways=2,
        background_size=3, block_size=2))
    reference = copy.deepcopy(model.attn)
    with torch.no_grad():
        for count in (13, 5, 1):
            hidden = torch.randn(1, count, 16)
            seen = reference.qcc._seen_tokens
            if count > 1:
                expected = reference._bounded_prefill(hidden,
                    torch.arange(seen, seen + count).view(1, -1),
                    reset=seen == 0, archive_hint=None, position_embeddings=None)
            else:
                expected = reference.qcc.step(hidden[:, 0]).unsqueeze(1)
            actual = model.attn(hidden, use_cache=True)[0]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            assert model.attn.qcc._chunk_key_scratch is None
            assert model.attn.qcc._chunk_value_scratch is None
            assert model.attn.qcc._archive_key_cache is None
    assert qcc_runtime_state_bytes(model)['total_bytes'] < qcc_runtime_state_bytes(reference)['total_bytes']


@pytest.mark.parametrize('sets', [2, 16])
def test_causal_block_hf_prefill_and_decode_share_commit_semantics(sets):
    import copy
    torch.manual_seed(913)
    model = _Model().eval()
    patch_hf_model_hybrid(model, window_size=4, prefill_chunk_size=5,
        use_triton=False, hybrid_kwargs=dict(causal_block_retention=True,
        exact_attention=True, exact_num_sets=sets, exact_ways=2, block_size=2))
    split = copy.deepcopy(model)
    changed = copy.deepcopy(model)
    hidden = torch.randn(1, 29, 16)
    with torch.no_grad():
        expected = model.attn(hidden, use_cache=True)[0]
        outputs = []
        start = 0
        for end in (3, 11, 12, 17, 28, 29):
            outputs.append(split.attn(hidden[:, start:end], use_cache=True)[0])
            start = end
        torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=1e-6, rtol=1e-5)
        other = hidden.clone()
        other[:, 19:] += 10
        actual = changed.attn(other, use_cache=True)[0]
        torch.testing.assert_close(actual[:, :19], expected[:, :19], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(model.attn.qcc.archive.exact_bank._keys,
                                   split.attn.qcc.archive.exact_bank._keys)
        if sets == 16:
            qcc = model.attn.qcc
            q, k, v, _ = qcc._project_qkv_gate(hidden)
            q, k = qcc._apply_rope(qcc._split_heads(q), qcc._split_heads(k),
                torch.arange(29).view(1, -1), sequence_length=29)
            head = torch.nn.functional.scaled_dot_product_attention(q, k, qcc._split_heads(v), is_causal=True)
            full = qcc.out_proj(head.transpose(1, 2).reshape(1, 29, 16))
            torch.testing.assert_close(expected, full, atol=1e-6, rtol=1e-5)
    assert model.attn.qcc._archive_key_cache is None
    assert model.attn.qcc._chunk_key_scratch is None
