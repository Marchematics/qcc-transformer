import sys
import types

from qcc_transformer.vllm_stock import (
    STOCK_ADAPTER_ENV,
    STOCK_CONFIG_ENV,
    QCCStockVLLMConfig,
    register_stock_vllm_backend,
)


def test_stock_plugin_is_noop_without_qcc_configuration(monkeypatch):
    monkeypatch.delenv(STOCK_CONFIG_ENV, raising=False)
    monkeypatch.delenv(STOCK_ADAPTER_ENV, raising=False)
    assert register_stock_vllm_backend() is None


def test_stock_plugin_registers_when_configured(monkeypatch):
    config = QCCStockVLLMConfig(num_heads=2, head_dim=4)
    monkeypatch.setenv(STOCK_CONFIG_ENV, config.to_json())
    monkeypatch.setenv(STOCK_ADAPTER_ENV, "/tmp/qcc-adapter.pt")
    calls = []
    enum = types.SimpleNamespace(CUSTOM="CUSTOM")

    backend = types.ModuleType("vllm.v1.attention.backend")
    backend.AttentionBackend = types.SimpleNamespace(customize_spec=True)
    backend.AttentionImpl = types.SimpleNamespace(forward=True)
    backend.CommonAttentionMetadata = types.SimpleNamespace(token_to_req_indices=True)
    kv = types.ModuleType("vllm.v1.kv_cache_interface")
    kv.CircularBufferSpec = types.SimpleNamespace(max_num_blocks_per_req=True)
    kv.AttentionSpec = types.SimpleNamespace(
        __dataclass_fields__={"state_content_bytes": object()}
    )
    registry = types.ModuleType("vllm.v1.attention.backends.registry")
    registry.AttentionBackendEnum = enum
    registry.register_backend = lambda value, path: calls.append((value, path))
    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.attention": types.ModuleType("vllm.v1.attention"),
        "vllm.v1.attention.backend": backend,
        "vllm.v1.attention.backends": types.ModuleType("vllm.v1.attention.backends"),
        "vllm.v1.attention.backends.registry": registry,
        "vllm.v1.kv_cache_interface": kv,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    result = register_stock_vllm_backend()
    assert result == "CUSTOM"
    assert calls == [("CUSTOM", "qcc_transformer.vllm_v1_backend.QCCV1AttentionBackend")]


def test_stock_plugin_registers_modern_vllm_028_paths(monkeypatch):
    config = QCCStockVLLMConfig(num_heads=2, head_dim=4)
    monkeypatch.setenv(STOCK_CONFIG_ENV, config.to_json())
    monkeypatch.setenv(STOCK_ADAPTER_ENV, "/tmp/qcc-adapter.pt")
    calls = []
    enum = types.SimpleNamespace(CUSTOM="CUSTOM")

    registry = types.ModuleType("vllm.v1.attention.backends.registry")
    registry.AttentionBackendEnum = enum
    registry.register_backend = lambda value, path: calls.append((value, path))
    attention_layer = types.ModuleType(
        "vllm.model_executor.layers.attention.attention"
    )

    class Attention:
        @staticmethod
        def get_kv_cache_spec(_self, _config):
            return "stock"

    attention_layer.Attention = Attention
    kv = types.ModuleType("vllm.v1.kv_cache_interface")

    class MambaSpec:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    kv.MambaSpec = MambaSpec
    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.attention": types.ModuleType("vllm.v1.attention"),
        "vllm.v1.attention.backends": types.ModuleType(
            "vllm.v1.attention.backends"
        ),
        "vllm.v1.attention.backends.registry": registry,
        "vllm.v1.kv_cache_interface": kv,
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType(
            "vllm.model_executor.layers"
        ),
        "vllm.model_executor.layers.attention": types.ModuleType(
            "vllm.model_executor.layers.attention"
        ),
        "vllm.model_executor.layers.attention.attention": attention_layer,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    result = register_stock_vllm_backend()

    assert result == "CUSTOM"
    assert calls == [
        ("CUSTOM", "qcc_transformer.vllm_modern_backend.QCCModernAttentionBackend")
    ]
    attention = Attention()
    attention.attn_backend = types.SimpleNamespace(get_name=lambda: "CUSTOM")
    spec = Attention.get_kv_cache_spec(attention, types.SimpleNamespace())
    assert spec.kwargs["dtypes"]
    assert "mamba_type" not in spec.kwargs
