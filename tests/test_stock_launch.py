from pathlib import Path

from qcc_transformer.stock_launch import stock_launch_manifest
from qcc_transformer.vllm_stock import QCCStockVLLMConfig


def test_stock_launch_manifest_is_one_config_change(tmp_path: Path):
    adapter = tmp_path / "adapter.pt"
    adapter.write_bytes(b"placeholder")
    config = QCCStockVLLMConfig(num_heads=8, head_dim=16)
    manifest = stock_launch_manifest(config, adapter_path=str(adapter))
    assert manifest["vllm_programmatic"] == {"attention_config": {"backend": "CUSTOM"}}
    assert manifest["vllm_cli"] == ["--attention-backend", "CUSTOM"]
    assert manifest["application_model_code_changes"] == 0
