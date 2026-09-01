from pathlib import Path

from benchmarks.colab_gpu_smoke import _parse_long_log


def test_parse_long_log_extracts_success_fields(tmp_path: Path) -> None:
    log = tmp_path / "length_128000.log"
    log.write_text(
        "device=cuda length=128000 position_encoding=sinusoidal state_only=False\n"
        "qcc_state_bytes=659456 full_kv_bytes=524288000 state_fraction=0.125781% "
        "reduction=795.03x under_0.5pct=True under_0.1pct=False\n"
        "processed_tokens=128000 prefill_seconds=12.5 tpot_ms=4.25\n",
        encoding="utf-8",
    )
    parsed = _parse_long_log(log)
    assert parsed["length"] == 128000
    assert parsed["qcc_state_bytes"] == 659456
    assert parsed["processed_tokens"] == 128000
    assert parsed["prefill_seconds"] == 12.5
    assert parsed["tpot_ms"] == 4.25


def test_parse_long_log_keeps_failed_log_path(tmp_path: Path) -> None:
    log = tmp_path / "length_4000000.log"
    log.write_text("CUDA out of memory\n", encoding="utf-8")
    parsed = _parse_long_log(log)
    assert parsed == {"log": str(log)}
