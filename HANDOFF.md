# QCC Transformer 交接文件

更新时间：2026-09-03（Asia/Shanghai）

## 目标与验收口径

当前严格目标是让同一个真实 pretrained 1B--7B checkpoint 同时满足：

- 官方 RULER、LongBench、PG-19 相对 matched Full-KV 的质量比均 `>= 0.98`；
- 真实 vLLM、128K context：TPOT `>= 5x`、吞吐 `>= 2x`；
- matched peak memory reduction `>= 80%`，long-context concurrency `>= 4x`；
- 校准 trainable 参数 `<= 1%`，HF/vLLM 零业务代码接入。

`benchmarks/gate_99.py` 是 fail-closed gate。当前没有任何通过 bundle，不能把
synthetic、random-weight、QCC-only、短上下文或 unmatched 结果当作 99 gate 证据。

## 代码与工作区路径

- 本地仓库：`/Users/nathmath/Documents/Codex/2026-09-01/cha`
- GitHub：<https://github.com/Marchematics/qcc-transformer>
- 当前本地分支：`main`
- 上次已推送提交：`3ae1a9a Use fused local kernel in vLLM state`
- 本次改动尚未提交：位置解耦 archive、HF RoPE 修复、校准内存优化、测试和文档。
- 远程独立工作区（实际存在路径）：
  `/home/frankwang122222/zjh/zjh/工作文件/qcc-transformer-next`
- 远程连接：`ssh frankwang122222@100.127.220.34`
- 远程真实模型：
  `/home/frankwang122222/zjh/zjh/工作文件/qcc-transformer-next/models/qwen2.5-1.5b`

用户要求的 `/zjh/工作目录/工作文件` 在该主机上没有对应的项目子目录；实际 QCC
项目在上面的 `zjh/zjh/工作文件` 路径，不能混用两个路径。

## 已实现的核心改动

### 位置解耦 archive

`qcc_transformer/model.py` 增加 `archive_position_invariant`：

- 局部 attention 继续使用 rotary Q/K；
- 长程 archive 使用 raw（未旋转）Q/K，避免不同绝对位置的 RoPE 相位污染内容寻址；
- streaming `step`、`step_chunk`、CUDA differentiable chunk path 和 ring wrap-around
  均维护独立 raw key ring。

HF retrofit 默认开启该选项，可用 `--no-archive-position-invariant` 做旧语义 ablation。

### HF/Qwen 兼容性修复

- `_apply_rope` 改为 Hugging Face Llama/Qwen 使用的 half-split `rotate_half` 约定；
- `patch_hf_model` 同时读取旧版 `config.rope_theta` 和 Transformers 5.x
  `config.rope_parameters["rope_theta"]`；
- retrofit 新建的 `rope_inv_freq` buffer 显式移动到 HF projection 所在 CUDA 设备；
- GQA/MQA 仍需显式 `kv_head_policy="repeat"`，不会静默改变 head 语义。

### 校准与审计

`benchmarks/calibrate_hf_retrofit.py` 现在：

- 先计算 teacher logits、释放 teacher，再加载 student，避免 24GB 卡双模型 OOM；
- 默认开启 gradient checkpointing，并启用 input grads；
- 输出 `parameter_count`、`trainable_parameter_count`、fraction、`run_id`、HF/vLLM
  zero-code flags；
- adapter 仍只包含 QCC archive/gate，不复制 pretrained backbone。

## 已验证结果（必须区分口径）

### 本地

- `git diff --check`、`compileall`、pytest：通过（当前 51 个收集项，46 passed、5 个 CUDA/Triton 条件 skip）。

### 远程真实 Qwen2.5-1.5B（1,543,714,304 参数）

| 实验 | 结果 | 结论 |
|---|---:|---|
| 1001 token，window 覆盖全序列，matched Full-KV | cosine `0.9998749`，top-1 `100%` | HF projection/GQA/RoPE/local exact path 已对齐 |
| 9868 token，uncalibrated archive，window 128 | cosine `0.5875`，top-1 `9.97%` | 长程 archive 质量严重不足 |
| calibration 10 steps，window 64/codes 32；held-out 1001 | cosine `0.9319`，top-1 `99.9%` | 未达到 0.99 |
| calibration 100 steps，window 48/codes 32；train 256 | cosine `0.9800`，top-1 `100%` | 训练集诊断 |
| 同一 adapter held-out 1001 token | cosine `0.9686`，top-1 `100%` | 仍未达到 0.99 |
| best of 10-card sweep3 held-out | cosine `0.9370`，top-1 `100%` | 仍未达到 0.99 |

校准 trainable 参数：`1,935,696 / 1,545,650,000 = 0.1252%`，满足参数比例限制，
但不代表质量 gate 已满足。

## 远程资源与清理状态

- 远程有 10 张 RTX 3090（每张 24GB）；2026-09-03 已按用户授权停止 Volt、roco-spring、YOLO 等其它 GPU 任务。
- 第三轮 sweep 使用 10 卡并行，所有作业属于 QCC 项目，日志和 adapter 在
  `artifacts/hf_99/sweep3/`。
- 远程 QCC 证据目录：`artifacts/hf_99/`；未把大模型、adapter 或日志推入 Git。
- 远程磁盘曾 100% 满；已清理可重建的 `~/.cache/pip`（约 3.7GB）及超过保留期的
  Hugging Face `.incomplete` 下载，释放约 4.5GB。未删除 HOTC2026、biohub 或其它项目的用户数据。
- Colab CLI 曾返回 `Service Unavailable`，当前没有活动 Colab session。
- 远程未安装正式 vLLM；`qcc_transformer/vllm.py` 是 dependency-free primitive，
  不是已注册的 upstream vLLM backend。远程工作区顶层有一个同名 `vllm.py`，测试正式
  vLLM 时应在项目目录外或先确认 import 来源，避免 shadowing。

## 当前主要困难

1. 线性/多尺度 archive 只保留 code response statistics，真实 Qwen 长程 logits 与 Full-KV 差距仍大；增加 codes 会明显推高反向峰值显存。
2. 目前 calibration 是全层 teacher-distillation，虽已用 checkpointing，但仍受 24GB 显存和 1.5B 模型训练吞吐限制。
3. Qwen2.5 原始 `max_position_embeddings=32768`；不能把它直接当作 128K pretrained gate，需要 RoPE scaling/原生长上下文 checkpoint 或明确的长上下文适配证据。
4. 尚无官方 RULER/LongBench/PG-19 的 matched Full-KV/QCC 结果；尚无真实 vLLM 128K TPOT/吞吐、peak memory/concurrency 证据。
5. 不能用当前结果宣称“99 gate 通过”或“颠覆级加速”。

## 推荐下一步

1. 读取 `artifacts/hf_99/adapter_eval_long.json`，确认 100-step adapter 的 held-out 结果。
2. 加入按层/按深度的 calibration（优先后半层），并比较 codes/window、gate 初始化和 persistent landmark；每次保留 held-out 文本。
3. 选择原生支持 >=128K 的真实 1B--7B checkpoint，准备官方 RULER/LongBench/PG-19 数据，生成同 `run_id` 的 evidence sections。
4. 在干净环境安装匹配版本 vLLM，完成 version-specific backend registration 和 128K matched benchmark；同时记录 CUDA peak memory 与并发曲线。
5. 所有 section 写入一个 JSON bundle 后运行：

   ```bash
   python benchmarks/gate_99.py --evidence artifacts/gates/<run_id>.json
   ```

6. 只有 gate 返回 `passed: true` 且原始日志、模型 hash、硬件信息可复核时，才可对外声称满足用户的五项要求。

## 常用命令

本地回归：

```bash
cd /Users/nathmath/Documents/Codex/2026-09-01/cha
python -m pytest -q
```

远程进入项目（用 glob 可规避中文路径编码问题）：

```bash
ssh frankwang122222@100.127.220.34
cd /home/frankwang122222/zjh/zjh/工作文件/qcc-transformer-next
```

真实 HF matched fidelity：

```bash
CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com \
python benchmarks/benchmark_hf_retrofit.py \
  --model models/qwen2.5-1.5b --prompt-file README.md \
  --window-size 128 --num-codes 64 --kv-head-policy repeat \
  --output artifacts/hf_99/<name>.json
```
