# QCC Transformer 本地交接文件

更新时间：2026-09-05（Asia/Shanghai）
交接对象：下一位继续实现、评测或部署 QCC Transformer 的工程师/模型
状态：代码已推送；99 gate 尚未通过；不要把当前结果包装成已达标结果。

## 当前 DSW 会话（2026-09-04）

- 网页入口：`https://dsw-gateway-cn-hangzhou.data.aliyun.com/dsw-2154359/lab`
- 内置 Terminal 主机：`dsw-2154359-8fcb79487-bnct7`，工作目录 `/mnt/workspace`。
- 规格：单卡 NVIDIA A10，显存 `23028 MiB`；检查时空闲，无 QCC/vLLM/训练进程。
- 已尝试从 GitHub 浅克隆 `main`；DSW 出站连接 GitHub 失败（先为 HTTP/2 framing error，改 HTTP/1.1 后为无法连接 443），因此当前 `/mnt/workspace/qcc-transformer` 尚未形成可用 checkout。
- 本地已生成源码包 `/tmp/qcc-transformer-src.tgz`（仅源码、约 825 KB，不含 `.git`、模型和 artifacts），必要时可通过 DSW 文件浏览器上传后在 Terminal 解包。
- 新增并推送 `scripts/dsw_prepare.sh`（提交 `e7e083b`），源码可用后执行 `bash scripts/dsw_prepare.sh /mnt/workspace/qcc-transformer` 完成 GPU、依赖和 checkpoint 探测。

### SSH A10G 实验（2026-09-04）

- 新主机：`root@93ff774ffe724492bc75389676c4d5d2.region1.waas.aigate.cc:47671`，单卡 NVIDIA A10G 24564 MiB；代码位于 `/home/waas/qcc-transformer`，使用 `/root/miniconda3/bin/python`（torch `2.7.0+cu128`、transformers `5.16.0.dev0`）。
- 可复用真实 checkpoint：`/datasets/ComfyUI/models/LLM/Phi-3.5-mini-instruct`，3.8B、原生 `131072` context。
- `ssh_phi_cal_20_fix`：512 tokens、20 steps、window 128、16 codes、bf16；adapter `artifacts/remote_gpu/ssh_runs/phi_cal_20_fix.adapter.pt`，参数 `3,825,864,704`，可训练 `4,785,152`（`0.1251%`），训练片段 cosine `0.999110`。
- 四段独立短 held-out（每段约 450 tokens）平均 cosine `0.960659`、平均 top-1 `0.617042`，`fidelity_passed=false`；这是 matched HF 诊断，不是 RULER/LongBench/PG-19 结果。
- 该次实验暴露并修复了 bf16 hidden 与 fp32 gate 的 dtype mismatch，修复提交为 `86c839e`。
- 同配置扩大到 `1024 tokens / 40 steps` 时在 `_parallel_decay_scan` 申请约 672 MiB 时 OOM（A10G 已用约 23.0 GiB）；失败日志和未生成 adapter 保留在 `artifacts/remote_gpu/ssh_runs/phi_cal_40_1024.log`。当前可复现实验上限仍是 512 tokens。

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

- 入口：阿里云 PAI DSW 的 `Terminal`，不使用 SSH。
- 实例：`dsw-7epb81cc8iok7hzw8r`，地域 `cn-shanghai`。
- 项目目录：`/mnt/workspace/qcc-transformer`。
- 真实 checkpoint：`/mnt/workspace/qcc-transformer/models/phi-4-mini-instruct-ms`。
- 远程实验产物：`<project>/artifacts/hf_99/` 和 `/tmp/phi_*.log`。
- 终端内已提供 `aliyun` CLI，并通过 DSW URI profile 获取临时凭据；不要把凭据写入脚本、日志或本文件。

### GitHub

- 仓库：<https://github.com/Marchematics/qcc-transformer>
- 分支：`main`
- 当前代码已推送到 `main`；实验日志和模型权重不进入 Git。
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
- Transformers 5.x 传入的 `position_embeddings=(cos, sin)` 现在直接用于 local Q/K，支持 batch/sequence 维度变体和 partial rotary；prefill chunk 会沿序列轴切片，避免动态/LongRoPE 被 wrapper 重算覆盖；
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
- `scripts/run_layerwise_sweep.sh` 与 `scripts/test_single_layerwise.sh` 已切换为终端可直接执行的默认值：项目目录 `/mnt/workspace/qcc-transformer`、模型 `models/phi-4-mini-instruct-ms`，并允许 `PROJECT_DIR`、`MODEL_PATH`、`OUTPUT_DIR`、`RUN_ID` 和 `HF_ENDPOINT` 环境变量覆盖；修复了未设置 `PYTHONPATH` 时在 `set -u` 下退出的问题。该修正已在提交 `10db081` 推送到 `main`。

### 3.5 Admission 标签坐标修正

`benchmarks/calibrate_hf_admission.py` 的 predictor 标签现在同时计算真实教师 RoPE Q/K 与 position-invariant raw Q/K 的未来 salience，并按位置取较强信号。这样 hybrid exact tier 不会因训练坐标与部署坐标不一致而漏掉检索关键 token；该改动只影响校准标签，不增加推理状态或参数。

该脚本的多 chunk 校准同时保留每个 chunk 的全局文本起点，并将其传入 teacher 的 `position_ids` 及后续 RoPE salience 计算；LongRoPE 模型因此不会把高位置样本错误地当作位置 0 的前缀。

teacher 特征现在通过未 patch attention 模块的 forward-pre-hook 直接捕获归一化后的 `hidden_states`（兼容位置参数和 keyword 参数），再复用原始 Q/K/V 投影；这避免用残差流替代 attention 输入训练 admission predictor。

该 hook 只用于校准 teacher，按选定层和 chunk 保存到 CPU；部署路径不注册 hook，也不改变模型参数、缓存状态或 HF/vLLM 接口。

### 3.6 未校准安全 gate（已实现，待远程验证）

- `QCCSelfAttention` 新增 `gate_bias_init`；HF retrofit 默认值为 `2.0`，让新 adapter 初始更接近 exact local path，避免随机 archive 以 50/50 比例污染 pretrained logits；
- 传入 `--gate-bias-init 0.0` 可复现旧的 50/50 ablation；校准脚本和 adapter manifest 会记录该值；
- 该改动只改善初始化稳定性，不等于长程质量或 99 gate 证据。

### 3.7 Phi 远程代码兼容

- `qcc_transformer.hf_loading.load_hf_causal_lm` 在 `trust_remote_code=True` 时为旧版 Phi3/Phi4 远程代码补充缺失的 `transformers.utils.LossKwargs` 类型别名。
- 该补丁只影响远程代码导入的类型注解，不改变模型权重或注意力计算；现有校准测试覆盖了该兼容路径。

### 3.8 最新远程诊断

- CPU teacher logits + 分块损失修复后，单卡 3-step smoke 可完成且不 OOM；全层 20-step smoke 也可完成；
- 10 卡 `layerwise10_20260903_021353`：9/10 配置完成，最佳 held-out cosine `0.8414`，`codes=64` 配置 OOM；所有配置均未通过 `0.99` fidelity gate；
- 新增 gate-bias smoke（全层、3 steps、bias=2.0）：held-out cosine `0.8406`，仅作初始化诊断；
- 远程当前已有 Qwen2.5-0.5B/1.5B（原生 `max_position_embeddings=32768`）和已下载的 `microsoft/Phi-4-mini-instruct`（真实 3.8B、`131072` context）；Phi-4 使用 Phi3 fused `qkv_proj` 和 partial/long RoPE，兼容实现见下一条。
- 已接入 Phi3/Phi4 fused `qkv_proj` 视图、单源 GEMM、GQA repeat 和 partial/LongRoPE 频率提取；真实 Phi-4-mini 81-token matched smoke：cosine `0.9999655`、top-1 `100%`，32/32 层成功 patch。该结果只证明短上下文路径对齐，不证明 128K archive 质量。
- 真实 Phi-4-mini 512-token long diagnostic（window 128、codes 16、未校准）：cosine `0.9646269`、top-1 `89.84375%`；50-step 全层校准后训练 cosine `0.9974645`、held-out cosine `0.9698532`、参数比例 `0.1037%`，仍未达到 `0.99`。
- Phi-4-mini 64K matched streaming（chunk 512、window 128、codes 16）：Full-KV `661.83 tok/s`、peak allocated `17.43GB`、reserved `24.94GB`；QCC `2190.75 tok/s`、peak allocated `8.59GB`、reserved `8.85GB`；速度 `3.31x`，allocated reduction `50.7%`，reserved reduction `64.5%`。两侧均完成 65536 tokens。
- Phi-4-mini 128K 同 runner：QCC 完成 `131072` tokens，`2200.34 tok/s`、peak allocated `8.59GB`；Full-KV 在第 166/256 chunk（约 `84.99K` tokens）OOM，peak allocated `20.33GB`；因此不能从该结果计算 matched speedup/80% reduction，只能记录为 QCC 可完成而 Full-KV 未完成。
- Phi-4-mini 多 chunk 校准（4 个 train chunks、100 steps、window 128、codes 16）：训练 cosine `0.996909`，held-out cosine `0.972924`，参数比例 `0.1037%`；相较单片段过拟合有所改善，但仍未达到 `0.99`。
- 2026-09-05 在 SSH A10G（`93ff774ffe724492bc75389676c4d5d2.region1.waas.aigate.cc:47671`）完成真实 Phi-3.5-mini 并发诊断：8K tokens/request、chunk 128、decode 4、固定 SLA 120 s、batch `1,2,4`，adapter `phi_cal_20_fix.adapter.pt`。Full-KV batch 1/2 均完成，吞吐 `1289.88/1522.60 tok/s`，TPOT `70.01/114.43 ms`，peak allocated `14.40/21.15 GB`；batch 4 在约 40/64 chunks OOM（peak `24.76 GB`）。QCC batch 1/2/4 均完成，吞吐 `1579.17/1704.02/1919.58 tok/s`，TPOT `52.43/233.14/54.99 ms`，peak allocated `7.97/8.27/8.88 GB`。按完成 batch 计 `max_full_kv_batch=2`、`max_qcc_batch=4`、并发比 `2.0x`；batch 4 因 Full-KV OOM 没有 matched speedup，不能外推为 4x 或 vLLM 结果。原始汇总已保留在 `artifacts/remote_gpu/ssh_runs/phi_concurrency_8k/summary.json`。
- 2026-09-05 质量修复：`benchmarks/calibrate_hf_layerwise.py` 现在在每个独立训练 batch、训练评估 batch 和 held-out batch 前调用 `reset_hf_qcc_cache`，避免 `_seen_tokens` 把 optimizer steps 串成一条历史流；同时修复 `ce_weight` 非零时被 MSE early-return 忽略的分支。真实 Phi-3.5-mini、window `512`、16 codes、8 个 1K train chunks、50 steps、`lr=0.002`、CE `0.4`/KL `0.3` 的校准结果：held-out cosine `1.0000`、top-1 `0.8418`；同 adapter 在独立 4K prompt 上 cosine `0.99759`、top-1 `0.69238`。训练参数 `4,785,152 / 3,825,864,704 = 0.1251%`。结果位于 `artifacts/remote_gpu/ssh_runs/quality_diag/cal_window512_ce_v4/` 和 `eval_window512_ce_v4/`。
- 同一真实 Phi-3.5-mini 4K prompt 的未校准质量曲线：window `128` / 16 codes 为 cosine `0.95244`、top-1 `0.42480`；window `512` 为 `0.99736` / `0.67163`；window `1024` 为 `0.99939` / `0.79663`；window `2048` 为 `0.99980` / `0.90894`。这证明误差主要来自历史注意力近似；这些都是 matched HF fidelity diagnostics，不是 RULER/LongBench/PG-19，也没有达到 0.99 top-1 gate。
- 实验性 `archive_kernel_features`（正随机特征 softmax archive）已贯通 reference/chunk/HF API，但在真实 Phi-3.5 4K、window `128`、16 codes 上修正尺度后仅得到 cosine `0.96385`、top-1 `0.43579`，默认保持关闭，不能作为质量方案。
- 质量校准修复（2026-09-05 后续）：普通 `calibrate_hf_retrofit.py` 现在像分层校准一样，在每个 optimizer step 和最终评估前显式 reset HF QCC state，避免独立文本 batch 被 `_seen_tokens` 串成一条流；同时支持可选 teacher-argmax `--ce-weight`，并正确参与 KL/MSE 权重归一化。HF retrofit 新增 `archive_scan_block_size`，校准 CLI 默认 `256`，可在不改变递推方程的情况下把长序列反向临时张量降到默认 `1024` 的四分之一，便于验证更大 codebook。该改动尚未产生新的真实 RULER/LongBench/PG-19 结果，不能视为质量门槛已通过。
- archive 读取质量修正（2026-09-05 后续）：新增默认开启的 `archive_global_normalization`，按可分离 softmax 的全局方程先合并 code/scale 的 numerator 与 denominator，再做一次归一化；旧的逐 code 独立归一化保留为 `--no-archive-global-normalization` 消融。为避免新旧 Triton kernel 混用不同方程，开启该模式时暂走 reference read/update path，需后续补等价 Triton kernel 后再恢复融合性能。该修正需要重新校准 adapter，不能直接把旧 adapter 的结果当作新方程的质量证据。

## 4. 已验证结果（严格区分证据等级）

- 长程质量校准修正（2026-09-05 后续）：`benchmarks/calibrate_hf_layerwise.py` 的多 chunk 校准现在从整段文本均匀取窗口，并为每个窗口显式传递原始绝对 `position_ids`；新增 `--num-held-out-chunks` 聚合多个验证窗口，避免所有 RoPE 样本从位置 0 开始。新增可选 `--margin-weight/--margin` top-2 排序损失，直接抑制 teacher argmax 交换；`held_out_gate_passed` 现在同时要求 cosine 与 top-1 达到阈值。旧 adapter 无需迁移，但要按新采样协议重校准后再比较质量。
- 2026-09-05 质量校准继续修复：分层校准的 `--code-init key-sample`（默认）现在从每个训练 chunk 均匀抽取有界数量的 teacher K 投影来初始化 codebook，而不是只使用第一个 chunk；`--code-init-tokens`（默认 256）限制 CPU staging，`--code-init random` 保留随机初始化消融。该改动不增加部署参数，仍需在真实长上下文 checkpoint 上重新校准并以 held-out/task benchmark 验证。
- 同一分层校准器新增默认 `--attention-loss-weight 0.35`：对选中层的 teacher/student attention 输出使用同一有界 token 采样监督，避免只靠最终 logits 让不同层互相补偿；设为 `0` 可复现 logit-only 消融。该局部损失不改变 adapter 参数量或最终 held-out gate 口径。

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
- Colab CLI 当前没有活动 session；本次通过已认证 OAuth 重新分配 `qcc-terminal` 时，Colab assignment API 返回 `503 Service Unavailable`。认证 scopes 正常，属于服务端资源阻塞；恢复前不要循环重试或假定 GPU 已分配。
- 2026-09-04 再次从本地 Terminal 执行 `colab new --gpu T4 --session qcc-terminal`，结果仍为 assignment API `503 Service Unavailable`；随后 `colab status` 确认 session 不存在。当前没有可运行的 Colab GPU 作业。
- 同一轮未改用 SSH；一次 `colab new --gpu L4 --session qcc-l4` 和一次 `colab new --gpu A100 --session qcc-a100` 均被后端以账户无对应 accelerator quota/entitlement 拒绝。可用路径仍是 PAI DSW Terminal 或 Colab T4 服务恢复。
- 2026-09-04 通过已打开的 DSW JupyterLab 网页入口确认实例在线：规格显示 `DSW - GPU`，剩余约 8 小时。此入口可直接从 Launcher 打开 Terminal；后续优先在该 Terminal 中执行远程实验，不需要 SSH。
- 远程尚未完成正式 vLLM 端到端运行；当前代码已提供 `vllm_modern_backend.py`，在现代 vLLM ABI（含 0.28）用 `MambaSpec` 分配每请求单页状态，并保留 0.11--0.27 的旧导入回退与 `CircularBufferSpec` 适配。`qcc_transformer/vllm.py` 仍是 dependency-free primitive，不能替代真实 serving 测量；远程项目顶层还有同名 `vllm.py`，测试时要先确认 import 来源，避免 shadowing。

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
5. 正式 vLLM backend registration 已实现为旧 `CircularBufferSpec` 与 vLLM 0.11+
   `MambaSpec` 两条路径，但远程尚未完成端到端 serving 验证；当前 primitive 不能
   冒充真实性能证据。
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
- 本地回归：全部现有测试通过（CUDA/Triton 条件测试按环境跳过）；Phi 远程代码兼容修复已推送 GitHub。
- 2026-09-04 复查时 DSW 终端连接曾短暂超时；重新启动长任务前先检查终端状态和 GPU 占用。
- 本地 tiny-random-Llama CPU smoke 已验证 `--cosine-weight 0.3` 与多 chunk
  参数路径可运行（1 step，held-out cosine `0.99537`；仅 API smoke，不是 gate 证据）。
- 新增 `benchmarks/benchmark_hf_concurrency.py`，可扫独立请求 batch；底层
  `benchmark_hf_streaming_memory.py` 现支持 `--batch-size`，并在 CPU 上只报告
  吞吐、不错误计算 CUDA 峰值显存。该工具仍是 HF diagnostic，不能替代 vLLM gate。
- `benchmarks/gate_99.py` 已扩展为严格 11 项验收：1M retrieval、tail safety、
  Pareto baseline、p95/p99 latency、scaling law、跨模型/GPU 复现均为必填；
  缺失任何 section 都 fail-closed。
- `register_stock_vllm_backend()` 现在按 ABI 选择旧 `CircularBufferSpec` 或现代
  vLLM（含 0.28）的 `MambaSpec` 适配；后者通过插件自动 patch
  `Attention.get_kv_cache_spec`，仍需在真实 GPU 环境完成端到端 benchmark，不能
  把本地 API 检查当作 serving 结果。
- 2026-09-05 修复 `benchmarks/launch_stock_vllm.py` 与 `qcc_transformer/stock_launch.py`
  的 vLLM CLI 参数：vLLM 0.28 使用 `--attention-backend CUSTOM`，旧的
  `--attention-config.backend` 只作为兼容输入解析，不再作为默认输出；JSON
  `--attention-config` 中的 backend 也会做冲突校验。该修复仅改变启动参数，不改变
  QCC 状态布局或模型接口。
- 0.28 上游 `MambaSpec.mamba_type` 是严格枚举字段；QCC 自定义 attention 不写入
  伪造的字符串类型，保留上游默认值以兼容 worker/KV-transfer 路径。
- `benchmarks/benchmark_hf_latency.py` 的每个 repeat 现在独立复制 decode
  `attention_mask`，避免前一个请求追加的 token 泄漏到后续 TPOT/p95/p99 样本。
- 现代 vLLM worker 的 `MambaSpec` cache 实际绑定为
  `[blocks, 1, 1, page_bytes]`；`QCCModernAttentionImpl` 现在零拷贝展平该视图，
  并将 TP rank helper 放入 `vllm_stock.py` 供新旧 backend 共同复用。此前这两处
  会分别导致运行时 page shape 拒绝和 modern backend 导入失败。
- `QCCSelfAttention` 新增可选 `archive_norm_gating`（参数量不变、O(1) 状态），
  按 local/archive 响应范数一致性抑制异常远程读；默认关闭，10 卡 sweep 的
  GPU9 打开该消融。
- 新增 `benchmarks/assemble_gate_evidence.py`：从 11 个 section JSON 组装最终
  bundle，并强制检查统一 `run_id`/`model_id`；它不会填充指标或伪造 provenance。

### 本地回归

```bash
cd /Users/nathmath/Documents/Codex/2026-09-01/cha
python -m pytest -q
python -m compileall qcc_transformer benchmarks tests
git diff --check
```

### 远程进入项目

在 PAI DSW 实例的 `Terminal` 中执行：

```bash
cd /mnt/workspace/qcc-transformer
git pull --ff-only origin main
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
- 用户要求的“实现并推到 GitHub”已完成；后续代码改动需单独提交并推送，避免把大模型和大日志提交进 Git。
