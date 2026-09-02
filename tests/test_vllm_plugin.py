import sys
import types

import pytest

from qcc_transformer.vllm_plugin import register_vllm_backend


def test_registration_requires_qualified_path():
    with pytest.raises(ValueError):
        register_vllm_backend("QCCBackend")


def test_registration_is_lazy_and_actionable_without_vllm():
    # vLLM is intentionally absent in the test environment.  If an unrelated
    # installation is present, this assertion is covered by the fake registry
    # test below instead.
    if "vllm.v1.attention.backends.registry" in sys.modules:
        pytest.skip("vLLM installed; use registry contract test")
    with pytest.raises(RuntimeError, match="vLLM v1 is not installed"):
        register_vllm_backend("pkg.QCCBackend")


def test_registration_calls_v1_registry(monkeypatch):
    calls = []
    enum = types.SimpleNamespace(CUSTOM="CUSTOM")
    registry = types.ModuleType("vllm.v1.attention.backends.registry")
    registry.AttentionBackendEnum = enum
    registry.register_backend = lambda backend, path: calls.append((backend, path))
    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.attention": types.ModuleType("vllm.v1.attention"),
        "vllm.v1.attention.backends": types.ModuleType("vllm.v1.attention.backends"),
        "vllm.v1.attention.backends.registry": registry,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    result = register_vllm_backend("pkg.QCCBackend")
    assert result == "CUSTOM"
    assert calls == [("CUSTOM", "pkg.QCCBackend")]
