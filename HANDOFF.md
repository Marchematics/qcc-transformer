# QCC Transformer 本地交接文件

更新时间：2026-09-03（Asia/Shanghai）
交接对象：下一位继续实现、评测或部署 QCC Transformer 的工程师/模型
状态：代码已推送；99 gate 尚未通过；不要把当前结果包装成已达标结果。

## 1. 项目目标与验收口径

目标是让同一个真实 pretrained 1B–7B checkpoint，同时满足以下五项要求：

1. 官方 RULER、LongBench、PG-19 相对 matched Full-KV 的质量均达到 `>= 98%`；
2. 真实 vLLM、128K context：TPOT `>= 5x`，吞吐 `>= 2x`；
3. matched peak memory reduction `>= 80%`，并带来 long-context concurrency `>= 4x`；
4. 校准 trainable 参数 `<= 1%`；
5. HF/vLLM 基本零业务代码接入，做到 retrofit/即插即用。

`benchmarks/gate_99.py` 是 fail-closed 验收器。没有完整、同模型、同硬件、同数据和可复核原始日志的 evidence bundle，就不能声称通过 99 gate。synthetic、random-weight、QCC-only、短上下文或 unmatched 结果只能作为开发诊断。

## 2. 本地、远程与 GitHub 路径

### 本地（当前 Codex 工作区）

- 仓库根目录：`/Users/nathmath/Documents/Codex/2026-09-01/cha`
- 交接文件：`/Users/nathmath/Documents/Codex/2026-09-01/cha/HANDOFF.md`
- 工作区说明：`/Users/nathmath/Documents/Codex/2026-09-01/cha/WORKSPACE.md`
- 研究计划/实验跟踪：`/Users/nathmath/Documents/Codex/2026-09-01/cha/refine-logs/`
- 核心实现：`/Users/nathmath/Documents/Codex/2026-09-01/cha/qcc_transformer/`
- 基准与 gate：`/Users/nathmath/Documents/Codex/2026-09-01/cha/benchmarks/`
- 测试：`/Users/nathmath/Documents/Codex/2026-09-01/cha/tests/`
- 实验产物：`/Users/nathmath/Documents/Codex/2026-09-01/cha/artifacts/`

### 远程 GPU 工作区

- SSH：`ssh frankwang122222@100.127.220.34`
- 用户要求的资源根目录：`/home/frankwang122222/zjh/工作目录/工作文件`
- 实际 QCC 项目目录（当前可用的 canonical project path）：`/home/frankwang122222/zjh/zjh/工作文件/qcc-transformer-next`
- 真实 checkpoint：`/home/frankwang122222/zjh/zjh/工作文件/qcc-transformer-next/models/qwen2.5-1.5b`
- 远程实验产物：`<project>/artifacts/hf_99/`，包括 `sweep/`、`sweep2/`、`sweep3/`

中文路径在某些 shell/同步工具中会出现编码问题；远程进入项目时可以使用：

```bash
cd /home/frankwang122222/zjh/zjh/*/qcc-transformer-next
```

不要把 SSH 密码写入脚本、日志或本文件；使用现有 SSH agent/密钥或交互式认证。

### GitHub

- 仓库：<https://github.com/Marchematics/qcc-transformer>
- 分支：`main`
- 当前已推送提交：`ede94fc Fix HF RoPE retrofit and position-invariant archive`
- 本地工作树中未跟踪的实验日志/产物不属于代码提交，见第 5 节。

## 3. 已实现内容

### 3.1 Position-invariant archive

`qcc_transformer/model.py` 新增 `archive_position_invariant`：

- 局部 attention 仍使用 rotary Q/K，保持 HF/Qwen 的局部语义；
- 长程 archive 使用未旋转的 raw Q/K，避免绝对位置 RoPE 相位污染内容寻址；
- 已贯通 `forward`、streaming `step`、`step_chunk`、CUDA differentiable chunk path 与 ring wrap-around；
- HF retrofit 默认开启；使用 `--no-archive-position-invariant` 可做旧语义 ablation。

### 3.2 HF/Qwen RoPE 与投影兼容

- `_apply_rope` 使用 HF Llama/Qwen 的 half-split `rotate_half` 约定；
- 同时兼容旧版 `config.rope_theta` 与 Transformers 5.x 的 `config.rope_parameters["rope_theta"]`；
- 新建的 `rope_inv_freq` buffer 会显式移动到 projection 所在 CUDA device；
- GQA/MQA 不静默改变 head 语义，使用 `kv_head_policy="repeat"` 时才显式复制 KV heads。

### 3.3 校准、显存与审计

`benchmarks/calibrate_hf_retrofit.py` 已支持：

- 先算 teacher logits、释放 teacher，再加载 student，避免 24GB 卡双模型 OOM；
- 默认 gradient checkpointing，并启用 input grads；
- 输出模型参数量、可训练参数量、参数比例、`run_id` 以及 HF/vLLM zero-code flags；
- adapter 只保存 QCC archive/gate，不复制 pretrained backbone。

### 3.4 分层校准增量（已实现，待远程验证）

- `benchmarks/calibrate_hf_layerwise.py` 支持 `all`、`last-half`、`last-quarter`、显式范围和离散层列表；只对选定层的 archive/gate 参数开启梯度；
- `patch_hf_model` 为每个替换层记录稳定的 `_qcc_layer_index`，供校准脚本选择；
- 优化器参数在构造前去重，避免 HF wrapper/nested module 引用同一参数时发生重复更新；
- `scripts/test_single_layerwise.sh` 和 `scripts/run_layerwise_sweep.sh` 提供远程实验入口，但其中的结果尚未形成 gate 证据。
- `scripts/run_layerwise_10gpu.sh` 可将 10 组配置分配到 10 张卡；最近一次 `layerwise10_20260903_021353` 有 9 组完成、1 组（`codes=64`）OOM，最佳 held-out cosine `0.8414`，全部 `held_out_gate_passed=false`。

### 3.5 未校准安全 gate（已实现，待远程验证）

- `QCCSelfAttention` 新增 `gate_bias_init`；HF retrofit 默认值为 `2.0`，让新 adapter 初始更接近 exact local path，避免随机 archive 以 50/50 比例污染 pretrained logits；
- 传入 `--gate-bias-init 0.0` 可复现旧的 50/50 ablation；校准脚本和 adapter manifest 会记录该值；
- 该改动只改善初始化稳定性，不等于长程质量或 99 gate 证据。

### 3.6 最新远程诊断

- CPU teacher logits + 分块损失修复后，单卡 3-step smoke 可完成且不 OOM；全层 20-step smoke 也可完成；
- 10 卡 `layerwise10_20260903_021353`：9/10 配置完成，最佳 held-out cosine `0.8414`，`codes=64` 配置 OOM；所有配置均未通过 `0.99` fidelity gate；
- 新增 gate-bias smoke（全层、3 steps、bias=2.0）：held-out cosine `0.8406`，仅作初始化诊断；
- 远程当前已有 Qwen2.5-0.5B/1.5B（原生 `max_position_embeddings=32768`）和已下载的 `microsoft/Phi-4-mini-instruct`（真实 3.8B、`131072` context）；Phi-4 使用 Phi3 fused `qkv_proj` 和 partial/long RoPE，兼容实现见下一条。
- 已接入 Phi3/Phi4 fused `qkv_proj` 视图、单源 GEMM、GQA repeat 和 partial/LongRoPE 频率提取；真实 Phi-4-mini 81-token matched smoke：cosine `0.9999655`、top-1 `100%`，32/32 层成功 patch。该结果只证明短上下文路径对齐，不证明 128K archive 质量。
- 真实 Phi-4-mini 512-token long diagnostic（window 128、codes 16、未校准）：cosine `0.9646269`、top-1 `89.84375%`；50-step 全层校准后训练 cosine `0.9974645`、held-out cosine `0.9698532`、参数比例 `0.1037%`，仍未达到 `0.99`。
- Phi-4-mini 64K matched streaming（chunk 512、window 128、codes 16）：Full-KV `661.83 tok/s`、peak allocated `17.43GB`、reserved `24.94GB`；QCC `2190.75 tok/s`、peak allocated `8.59GB`、reserved `8.85GB`；速度 `3.31x`，allocated reduction `50.7%`，reserved reduction `64.5%`。两侧均完成 65536 tokens。
- Phi-4-mini 128K 同 runner：QCC 完成 `131072` tokens，`2200.34 tok/s`、peak allocated `8.59GB`；Full-KV 在第 166/256 chunk（约 `84.99K` tokens）OOM，peak allocated `20.33GB`；因此不能从该结果计算 matched speedup/80% reduction，只能记录为 QCC 可完成而 Full-KV 未完成。
- Phi-4-mini 多 chunk 校准（4 个 train chunks、100 steps、window 128、codes 16）：训练 cosine `0.996909`，held-out cosine `0.972924`，参数比例 `0.1037%`；相较单片段过拟合有所改善，但仍未达到 `0.99`。

## 4. 已验证结果（严格区分证据等级）

### 本地回归

- `python -m pytest -q`：51 个收集项，46 passed，5 个 CUDA/Triton 条件 skip；
- 分层校准解析与层索引测试已加入，当前完整回归：51 个收集项，46 passed，5 个 CUDA/Triton 条件 skip；
- `git diff --check`：通过；
- `python -m compileall qcc_transformer benchmarks tests`：通过。

### 真实 Qwen2.5-1.5B

远程模型参数量约 `1,543,714,304`。当前结果：

| 实验 | 结果 | 解释 |
|---|---:|---|
| 1001 token，window 覆盖全序列，matched Full-KV | cosine `0.9998749`，top-1 `100%` | HF projection/GQA/RoPE/local exact path 已对齐 |
| 9868 token，window 128，未校准 archive | cosine `0.5875`，top-1 `9.97%` | 长程 archive 明显不足 |
| 10-step calibration，window 64/codes 32，held-out 1001 | cosine `0.9319`，top-1 `99.9%` | 未达到 0.99 |
| 100-step calibration，window 48/codes 32，训练集 | cosine `0.9800`，top-1 `100%` | 仅训练集诊断，不是 gate 证据 |
| 同 adapter held-out 1001 | cosine `0.9686`，top-1 `100%` | 仍未达到 0.99 |
| 10 卡 sweep3 最佳 held-out | cosine `0.9370`，top-1 `100%` | 仍未达到 0.99 |

校准参数比例：`1,935,696 / 1,545,650,000 = 0.1252%`，满足 `<=1%` 这一单项限制，但不代表整体 99 gate 通过。

特别注意：Qwen2.5-1.5B 原始 `max_position_embeddings=32768`，不能直接拿它作为 128K pretrained gate 证据；128K 需要原生长上下文 checkpoint、明确 RoPE scaling 适配，或另行审计的长上下文扩展方案。

## 5. 资源、运行状态与清理

- 远端共有 10 张 RTX 3090，每张 24GB；按用户授权，Volt、roco-spring、YOLO 等其它 GPU 任务已停止；当前没有 QCC 活跃任务，10 卡可继续并行实验。
- 三轮 Qwen sweep 均为 QCC 项目实验，日志/adapter 位于远程 `artifacts/hf_99/sweep*`。
- 远程曾发生磁盘 100% 满；已清理可重建的 `~/.cache/pip`（约 3.7GB）、过期 Hugging Face `.incomplete` 文件和本次同步产生的 `._*` 文件，释放约 4.5GB。
- 未删除 HOTC2026、biohub、其它用户项目或用户已有 QCC artifacts。
- Colab CLI 曾返回 `Service Unavailable`，当前没有活动 Colab session。
- 远程未安装正式 vLLM；`qcc_transformer/vllm.py` 目前是 dependency-free primitive，不是 upstream vLLM 已注册 backend。远程项目顶层还有同名 `vllm.py`，做正式 vLLM 测试时要先确认 import 来源，避免 shadowing。

本地以下未跟踪文件是有意保留的实验结果，不要误删：

```text
artifacts/hf_99/
artifacts/local_cpu/multiseed/
artifacts/remote_gpu/retrain_20260902/
```

清理前必须确认：属于 QCC、未被活跃作业引用、超过保留期且可从源数据/脚本重建；不要按文件名批量删除压缩包或其它项目目录。

## 6. 当前困难与风险

1. 当前 archive 主要保留 code-response statistics，真实 Qwen 长程 logits 与 Full-KV 差距仍大；简单增加 codes 会显著抬高反向峰值显存。
2. calibration 已增加 CPU teacher logits 和词表分块损失，但 24GB 卡在 `max_tokens=512`、`codes=64` 时仍会 OOM；更长序列需要进一步分块 activation/逐层蒸馏。
3. gate 初始化虽已默认偏向 local path，但仍需在真实长程 held-out 数据上校准 archive，不能将初始化效果当作最终质量。
4. 尚无官方 RULER、LongBench、PG-19 的 matched Full-KV/QCC 结果；尚无真实 128K vLLM TPOT、吞吐、peak memory/concurrency 证据。
5. 尚未完成正式 vLLM backend registration；当前 primitive 不能冒充零改动 upstream backend。
6. 当前没有一个 `gate_99.py` evidence bundle 返回 `passed: true`；不得宣称“99 gate 已通过”“≥98% 全面质量”或“颠覆级加速”。

## 7. 下一阶段目标（按优先级）

1. 先提升真实长程 fidelity：实现按层/按深度 calibration（优先后半层），系统比较 `num_codes`、`window_size`、gate 初始化、persistent/prefix landmark，并固定 held-out 文本。
2. 选择原生支持 `>=128K` 的真实 1B–7B checkpoint，记录模型 hash、tokenizer、RoPE 配置和硬件信息。
3. 接入官方 RULER、LongBench、PG-19，分别跑 matched Full-KV 与 QCC，输出可复核的逐任务结果。
4. 在干净环境安装匹配版本 vLLM，完成 version-specific backend registration；跑真实 128K TPOT、吞吐、CUDA peak memory 和并发曲线。
5. 把所有验收 section 写入同一个 evidence bundle，并执行：

```bash
cd /Users/nathmath/Documents/Codex/2026-09-01/cha
python benchmarks/gate_99.py --evidence artifacts/gates/<run_id>.json
```

只有 gate 返回 `passed: true`，且原始日志、模型 hash、硬件和 benchmark 可复核，才可以对外宣称达到用户的五项目标。

## 8. 常用命令

### 最新代码变更（2026-09-03）

- `benchmarks/calibrate_hf_layerwise.py` 新增 `--cosine-weight`，可在词表分块 MSE 外加入方向一致性损失；默认 `0` 与历史目标兼容。
- `scripts/run_layerwise_10gpu.sh` 已改为 10 卡 cosine-weight 消融矩阵（`0, 0.1, 0.3, 0.5, 1.0`）。
- 该 sweep 默认模型已切换为原生 128K 的 `models/phi-4-mini-instruct`，并默认
  `--trust-remote-code --num-train-chunks 4`；不再把 32K 的 Qwen2.5-1.5B 当作
  128K gate 模型。
- `qcc_transformer/vllm.py` 的默认 `archive_mix` 从 `0.5` 调整为 `0.125`，与 HF `gate_bias_init=2` 的质量优先初始化一致；复现实验时可显式传入 `archive_mix=0.5`。
- 本地回归：`61 passed`（CUDA/Triton 条件测试按环境跳过）；最新提交 `e3d17a7` 已推送 GitHub。
- 2026-09-03 复查时远程 SSH 与 Colab 均无可用活动会话；重新启动长任务前先检查连接和 GPU 占用。
- 本地 tiny-random-Llama CPU smoke 已验证 `--cosine-weight 0.3` 与多 chunk
  参数路径可运行（1 step，held-out cosine `0.99537`；仅 API smoke，不是 gate 证据）。

### 本地回归

```bash
cd /Users/nathmath/Documents/Codex/2026-09-01/cha
python -m pytest -q
python -m compileall qcc_transformer benchmarks tests
git diff --check
```

### 远程进入项目

```bash
ssh frankwang122222@100.127.220.34
cd /home/frankwang122222/zjh/zjh/工作文件/qcc-transformer-next
```

### 真实 HF matched fidelity（示例）

```bash
CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com \
python benchmarks/benchmark_hf_retrofit.py \
  --model models/qwen2.5-1.5b --prompt-file README.md \
  --window-size 128 --num-codes 64 --kv-head-policy repeat \
  --output artifacts/hf_99/<name>.json
```

## 9. 交接原则

- 先读本文件、`README.md`、`WORKSPACE.md` 和 `refine-logs/EXPERIMENT_PLAN.md`，再启动长任务；
- 所有新实验必须记录 `run_id`、模型路径/hash、数据来源、卡号、CUDA/PyTorch/Transformers 版本、命令行、原始日志和输出 JSON；
- 不覆盖已有 artifacts；新实验使用新目录或新文件名；
- 任何性能/质量数字先跑 `gate_99.py` 和相应审计，再写入 README 或论文；
- 用户要求的“实现并推到 GitHub”已完成到 `ede94fc`，后续代码改动需单独提交并推送，避免把大模型和大日志提交进 Git。
