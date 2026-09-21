# QCC retention bundle (20260924)

中文摘要：本包是可复现的证据快照——手稿（`MANUSCRIPT.md`）、指标台账（`CLAIMS.md`）、
全部结论对应的原始结果 JSON（`data/`）、以及产生它们所需的代码与运行脚本（`code/`）。
所有数字都由已发布的 `compile_bounded_cache` 路径产生；`CLAIMS.md` 里每条结论都写明
对应命令。状态：质量类与状态类指标在参考模型上达标并有跨族证据；1M 两项与 128K TPOT
5x 受硬件限制（1M 的 bf16 Full-KV 需 32 GiB > 24 GiB 卡，本机无 1M-native 权重）。

## Contents

| path | what it is |
|---|---|
| `MANUSCRIPT.md` | the full technical report: sections 1-3 (per-claim measurements, including 3.19 ragged batches, 3.20 package/harness parity, 3.21 LongBench + baselines, 3.22 state growth, 3.23 quantization Pareto, 3.24 cross-family, 3.25 worst-task attribution), plus what is not established |
| `CLAIMS.md` | every claim mapped to the shipped code path, the artifact behind it and the command that regenerates it; open questions and the run that closes each |
| `HANDOFF.zh.md` | Chinese hand-off note: current audit, the two bugs the reproduction check caught, remaining gaps, push procedure |
| `data/` | the evidence JSONs (see table below); each contains per-record predictions, scores, retained slot counts, timings and peak memory |
| `code/` | the packaged retention path, the benchmark runners, the analysis scripts and the tests |
| `environment.txt` | git revision, library versions, GPU, and the measurement-condition caveat |
| `MANIFEST.sha256` | checksums of every file in this bundle |

## What the data shows

| claim | artifact | number |
|---|---|---|
| 80-record package/harness parity | `data/bounded-decode-frontier-parity-80.json` | 80/80 metric agreement, 75/80 byte-identical predictions |
| RULER retention through the shipped API | `data/bounded-decode-frontier-multimodel-llama1b.json` | 1.0071 aggregate, 1.000 worst task at 4,608 slots |
| Ragged/padded batches | `data/bounded-decode-frontier-batch-validation.json` | per-row slots and logits bit-identical to per-row compiles |
| LongBench (9 tasks, official metrics) | `data/longbench-retention-llama32-1b.json` | 1.0049 aggregate, 0.897 worst task, 122 matched records |
| Matched-budget baselines | `data/bounded-decode-frontier-baselines-b4096.json` | sliding window 0.167, sinks+recent 0.332, last-query 0.842, window-max 0.863, window-mean 0.880, shipped 1.008 |
| Quality/state Pareto | `data/bounded-decode-frontier-multimodel-llama1b-b1024.json`, `-b2048.json`, `data/bounded-decode-frontier-lexonly-b4096.json` | 48 MiB -> 0.987, 80 MiB -> 0.975, 144 MiB -> 1.007, 34.5 MiB anchors-only -> 0.930 |
| State growth vs prompt length | `data/bounded-decode-frontier-state-growth.json`, `-long.json` | 4,608 slots / 144 MiB at 8K-256K, growth 1.00x, Full-KV 57x larger at 256K |
| Quantization Pareto | `data/bounded-decode-frontier-kv-quant.json` | Full-KV int8 lossless at 2x; int4 0.413; bounded+int8 = Full-KV quality at 72.3 MiB |
| Cross-family generality | `data/bounded-decode-frontier-multimodel-{llama3.1-8b,qwen2.5-3b,phi35}.json` | 8B 1.014/1.000, Qwen2.5-3B 0.975/0.917, Phi-3.5-mini 0.972/0.889 |
| Anchor/scoring attribution | `data/bounded-decode-frontier-multimodel-llama1b-{nolex,nolex-mean,rare,rare-hops0,pattern-hops0,mean-anchors}.json` | ranking only 0.817 (last) / 0.870 (window mean); +rarity 0.938; +patterns 1.0065; chains add nothing |
| TPOT, both arms optimised | `data/bounded-decode-frontier-tpot-percentiles.json` | 32K: bounded+graph 11.01 ms vs Full-KV+graph 53.7-55.2 ms = 4.9-5.0x |

## Reproducing

```bash
# the shipped path and its tests
/root/qcc/venv/bin/python -m pytest -q tests/test_retention.py tests/test_kv_quant.py \
    tests/test_longbench_metrics.py

# one model, both arms, all 80 RULER records
python benchmarks/benchmark_retention_multimodel.py \
    --model /root/qcc/models/Llama-3.2-1B-Instruct \
    --out artifacts/multimodel-llama1b.json

# the report's tables from the JSONs
python benchmarks/analyze_campaign.py multimodel <json>...
python benchmarks/analyze_campaign.py baselines <json>
```

Datasets are not bundled: RULER split (`/home/waas/ruler_a10_v1/ruler_subset.jsonl`) and
LongBench (`THUDM/LongBench`, downloaded on demand by `benchmarks/longbench_data.py`).
Model checkpoints are not bundled either; the paths used are recorded in each JSON's
`config.model` field.

## Honest status

Established: quality parity on two Llama sizes, three families measured, LongBench parity,
1.00x state growth to 256K, 48 MiB caches at 0.987, eviction and quantization Pareto,
80/80 reproducibility. Not established, with the concrete reason: 1M retrieval and 1M TPOT
(a 1M bf16 Full-KV cache is 32 GiB against 24 GiB of HBM, and no 1M-native checkpoint is
available here); 128K TPOT 5x (4.9-5.0x measured at 32K with both arms graphed; at 128K the
baseline cannot run on this card); latency percentiles beyond p50 (shared card); and the
cross-family shortfall on `niah_multikey_3` (0.917 Qwen, 0.889 Phi), which five different
selection configurations leave unchanged.
