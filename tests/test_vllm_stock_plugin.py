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
