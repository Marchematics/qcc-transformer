from pathlib import Path
from types import SimpleNamespace

from benchmarks.launch_stock_vllm import (
    _option_value,
    _trust_remote_code,
    build_launch,
    infer_attention_geometry,
    native_context,
    infer_kv_heads,
)


def test_stock_launcher_derives_geometry_and_context():
    config = SimpleNamespace(hidden_size=256, num_attention_heads=8, max_position_embeddings=131072)
    assert infer_attention_geometry(config) == (8, 32)
    assert native_context(config) == 131072
    assert infer_kv_heads(config, 8) == 8


def test_stock_launcher_preserves_local_gqa_geometry_under_tp(tmp_path: Path):
    adapter = tmp_path / "adapter.pt"
    adapter.write_bytes(b"adapter")
    config = SimpleNamespace(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=131072,
    )
    _, _, packed = build_launch(
        model="org/model-1b",
        adapter=adapter,
        config=config,
        context_length=128000,
        window_size=128,
        num_codes=16,
        num_scales=4,
        exact_num_sets=32,
        exact_ways=2,
        exact_probe_sets=None,
        local_dtype="bfloat16",
        passthrough=["--tensor-parallel-size", "2"],
    )
    assert packed.num_heads == 4
    assert packed.num_kv_heads == 1


def test_stock_launcher_parses_remote_code_and_option_forms():
    arguments = ["--trust-remote-code", "--max-model-len=128000", "--dtype", "bfloat16"]
    assert _trust_remote_code(arguments) is True
    assert _option_value(arguments, "--max-model-len") == "128000"
    assert _option_value(arguments, "--dtype") == "bfloat16"


def test_stock_launcher_shards_query_heads_for_tensor_parallelism(tmp_path: Path):
    adapter = tmp_path / "adapter.pt"
    adapter.write_bytes(b"adapter")
    config = SimpleNamespace(hidden_size=256, num_attention_heads=8, max_position_embeddings=131072)
    command, _, packed = build_launch(
        model="org/model-1b",
        adapter=adapter,
        config=config,
        context_length=128000,
        window_size=128,
        num_codes=16,
        num_scales=4,
        exact_num_sets=32,
        exact_ways=2,
        exact_probe_sets=None,
        local_dtype="bfloat16",
        passthrough=["--tensor-parallel-size", "2"],
    )
    assert command[:3] == ["vllm", "serve", "org/model-1b"]
    assert packed.num_heads == 4
    assert packed.tensor_parallel_size == 2


def test_stock_launcher_builds_worker_environment_and_preserves_server_args(tmp_path: Path):
    adapter = tmp_path / "adapter.pt"
    adapter.write_bytes(b"adapter")
    config = SimpleNamespace(hidden_size=256, num_attention_heads=8, max_position_embeddings=131072)
    command, environment, packed = build_launch(
        model="org/model-1b",
        adapter=adapter,
        config=config,
        context_length=128000,
        window_size=128,
        num_codes=16,
        num_scales=4,
        exact_num_sets=32,
        exact_ways=2,
        exact_probe_sets=None,
        local_dtype="bfloat16",
        passthrough=["--dtype", "bfloat16", "--port", "9000"],
    )
    assert command[:3] == ["vllm", "serve", "org/model-1b"]
    assert "--dtype" in command and "bfloat16" in command
    assert "--attention-config.backend" in command
    assert "--max-model-len" in command
    assert environment["QCC_STOCK_VLLM_ADAPTER"].endswith("adapter.pt")
    assert packed.num_heads == 8
    assert packed.max_position_embeddings == 128000


def test_stock_launcher_rejects_server_context_or_backend_mismatch(tmp_path: Path):
    adapter = tmp_path / "adapter.pt"
    adapter.write_bytes(b"adapter")
    config = SimpleNamespace(hidden_size=256, num_attention_heads=8, max_position_embeddings=131072)
    common = dict(
        model="org/model-1b",
        adapter=adapter,
        config=config,
        context_length=128000,
        window_size=128,
        num_codes=16,
        num_scales=4,
        exact_num_sets=32,
        exact_ways=2,
        exact_probe_sets=None,
        local_dtype="bfloat16",
    )
    try:
        build_launch(**common, passthrough=["--max-model-len", "131072"])
    except ValueError as exc:
        assert "max-model-len" in str(exc)
    else:
        raise AssertionError("mismatched server context was accepted")
    try:
        build_launch(**common, passthrough=["--attention-config.backend", "FLASH_ATTN"])
    except ValueError as exc:
        assert "CUSTOM" in str(exc)
    else:
        raise AssertionError("mismatched attention backend was accepted")


def test_stock_launcher_rejects_context_above_native_limit(tmp_path: Path):
    adapter = tmp_path / "adapter.pt"
    adapter.write_bytes(b"adapter")
    config = SimpleNamespace(
        hidden_size=256,
        num_attention_heads=8,
        max_position_embeddings=131072,
    )
    try:
        build_launch(
            model="org/model-1b",
            adapter=adapter,
            config=config,
            context_length=256000,
            window_size=128,
            num_codes=16,
            num_scales=4,
            exact_num_sets=32,
            exact_ways=2,
            exact_probe_sets=None,
            local_dtype="bfloat16",
            passthrough=[],
        )
    except ValueError as exc:
        assert "native context" in str(exc)
    else:
        raise AssertionError("context above the checkpoint native limit was accepted")
