# QCC Transformer 本地交接文件

更新时间：2026-09-05（Asia/Shanghai）
交接对象：下一位继续实现、评测或部署 QCC Transformer 的工程师/模型
状态：代码已推送；99 gate 尚未通过；不要把当前结果包装成已达标结果。

## 当前审计（2026-09-24 凌晨）：普适性证据齐了，两个 bug 被自己的复现检查抓出来

**结论先说**：保留律不再是"RULER 上的一个高分"，它现在是四条独立证据链。全部数字
由**已发布的包路径**（`compile_bounded_cache`）产生，命令写在 `CLAIMS.md`。

### 1. 公开实现复现主结果（80/80，逐位）

之前 package 自己的 80 条跑出 aggregate retention 0.9253（harness 是 1.000）。同进程、
同 cache 的诊断脚本（`experiments/retention_frontier/diff_selection.py`）把它拆开：

| 环节 | 修前 | 修后 |
|---|---|---|
| 词法锚点 | 288 vs 288，完全一致 | 不变 |
| 每层打分 | 不同 | **逐位相同（max|Δ|=0.0）** |
| 选中的槽集合 | 不同 | **完全相同** |

根因是我自己写的 `last` query 打分用了代数等价但**形状不同**的 einsum
（`"god,hld->hgol"` vs harness 的 `"hgod,hld->hgol"`）：cuBLAS 换 kernel → 舍入不同 →
top-k 近似并列翻转。修好后 80 条记录指标 80/80 一致、75/80 生成逐字节相同，per-task
均值与 harness 完全相同，包路径自身 retention **1.0071 / 最差任务 1.000**。

**教训**（值得写进论文的 engineering note）：等价重写打分核 = 行为改变；判定标准是
**选中的集合**，不是分数。

### 2. 非 RULER 证据：LongBench 9 任务 × 20 条

真实长文档（小说/论文/政府报告/新闻/对话），官方指标，122 条 Full-KV 能答对的记录：
聚合 retention **1.0049**（macro 0.2514 vs 0.2527），最差任务 0.897（gov_report 的
ROUGE-L），narrativeqa **+10.7%**。→ "RULER 专用技巧"这条质疑被削弱：换到真实文档、
不用合成 needle，结论不变。

### 3. 跨模型普适性：4 个模型 / 3 个族 / 同一配置、零调参

| 模型 | 结构 | 匹配记录 | aggregate | 最差任务 | 槽 | decode state |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B | GQA 32:8, 16L | 68/80 | **1.007** | **1.000** | 4,608 | 144 MiB |
| Llama-3.1-8B (4bit) | GQA 32:8, 32L | 79/80 | **1.014** | **1.000** | 4,608 | 576 MiB |
| Qwen2.5-3B | GQA 16:2, 36L | 57/60 | 0.975 | 0.917 | 4,608 | 162 MiB |
| Phi-3.5-mini | MHA 32:32, LongRoPE | 39/40 | 0.972 | 0.889 | 4,608 | 1,728 MiB |

两个 Llama 全任务持平或更好；Qwen/Phi 的缺口**只在 `niah_multikey_3` 一个任务**上
（三键消歧，也是各模型自己最接近随机的地方）。换句话说："最差任务 ≥97%" 目前是
Llama 族成立、跨族待收口，缺口是**被定位的一个任务**，不是弥散的短板。

Phi 跑通需要两个非 Llama 适配：融合 `qkv_proj` 取 Q 切片 + 每层 rotary；
`fixed_rope_length`（否则分块 prefill 让 LongRoPE 前几块用 short factor，缓存 key 与
观测 query 的相位不一致，16K 记录直接 0 分）。

### 4. 质量来自哪里 + 同预算 baseline

| 配置（Llama-1B，80 条） | single_1 | multikey_2 | multikey_3 | vt | aggregate | 最差任务 |
|---|---:|---:|---:|---:|---:|---:|
| 只靠注意力排序（`lex_cap=0`） | 1.000 | 0.895 | 0.333 | 1.039 | 0.817 | 0.333 |
| 窗口 mean（SnapKV 形，harness baseline） | 1.000 | 0.800 | 0.200 | 0.640 | 0.880 | 0.000 |
| **+ 任务无关的稀有串锚点**（`anchor_mode="rare"`） | 1.000 | 0.895 | **0.889** | 0.968 | **0.938** | 0.889 |
| 出厂：模式锚点 + 赋值链 | 1.000 | 1.000 | 1.000 | 1.028 | **1.007** | 1.000 |

同预算（4,096/4,608 槽）7 策略对比（官方 `string_match_all` 口径）：滑窗 0.167、
sink+recent 0.332、last-query 0.842、窗口 max 0.863、SnapKV 形窗口 mean 0.880、
出厂 1.008——**只有出厂配置没有整任务归零**。

### 5. 状态与字节：0.93 的质量只需 34.5 MiB

- 8K→128K 实测：保留槽恒为 4,608、状态恒为 144 MiB，**增长 1.00×**；Full-KV 增长 16×。
- 只用锚点+sink+近期窗（不选注意力填充）：**1,105 槽 / 34.5 MiB → 0.930**，且在
  multikey_2 上比出厂还好（1.000 vs 0.950）。
- 量化 Pareto：Full-KV bf16 427 MiB 0.8125；Full-KV int8 215 MiB 0.8125（**无损**）；
  Full-KV int4 107 MiB **0.4125**；bounded bf16 144 MiB 0.8250；bounded+int8
  **72.3 MiB 0.8250**（= Full-KV 质量，小 5.9×）。

### 仍然没做到的（不要包装）

- **1M 两项**：无 1M-native 权重；1M 的 bf16 Full-KV 需 32 GiB > 24 GiB 卡，基线在 1M
  根本跑不起来。
- **128K TPOT ≥5×**：matched 1.73×、最强系统数字 4.21×，带宽上限 2.70×（还没算权重）。
- **跨族"最差任务 ≥97%"**：Qwen 0.917 / Phi 0.889，缺口集中在 multikey_3；已有针对该
  任务的收口实验在队列里。
- **延迟百分位**：共享卡，同一配置 6.87–18.4 ms 波动；要独占硬件 + 重复 + p50/p95/p99。

## 当前审计（2026-09-23 深夜）：保留律已进包，SLA 并发可复核

**保留律现在是包的一部分**（`qcc_transformer/retention.py`，已从 `__init__`
导出）：

```python
from qcc_transformer import RetentionConfig, compile_bounded_cache
cache, logits = compile_bounded_cache(model, input_ids,
                                      RetentionConfig(budget=4096, lex_cap=512, chain_hops=6),
                                      tokenizer=tok)
```

`attention_mask` 可以是 padded batch（左/右 padding 都可以）：选择先在每条请求自己的
真实 token 上做，行宽不同时短行用**自身最后槽的副本**补齐——两个完全相同的 `(key, value)`
槽会均分原来的 softmax 权重，所以补齐后与不补齐是同一次计算；副本在
`cache.qcc_attention_mask` 里标成 padding。ragged batch 解码时必须自己给位置：
`position_ids = cache.qcc_prompt_lengths[:, None] + step`（压缩后的 cache 比 prompt 短，
任何 mask 的 cumsum 都推不出真实长度）。验证见
`benchmarks/validate_retention_batch.py`（报告 §3.19）：7,730 与 15,584 两行打包后每行
cache 与单独编译**逐位相同**，唯一分歧是 shipped 配置下 24 步里第 23 步、top-2 margin
恰为 0.0 的 bf16 batch GEMM 平局，两行得分都仍是 1.0。

零新增参数、不动权重。验证三层：
1. `tests/test_retention.py` 七个 CPU 测试（小随机 Llama）：统一宽度、全保留时与
   未压缩 prefill 逐 token 一致、sink/近期强制保留、锚点召回、赋值链跟随，以及
   ragged batch 在两种 padding 下与逐行编译逐槽一致；开发过程中它们抓出打包版
   `last` 打分的 einsum 秩错误；
2. `benchmarks/validate_retention_api.py`：4 条真实 RULER 记录，4/4 得分 1.0，
   4608 槽，2.3–3.9 s/条，峰值 6.2–7.3 GiB；
3. `benchmarks/validate_retention_full.py`：20 条（每任务 5 条）与 benchmark
   harness 在相同预算下逐条比对，**19/20 完全一致**，唯一差异是包在一
   multikey_3 记录上答对而 harness 答错。仓库完整测试套件通过（并修掉一个
   Transformers 5.x 下必挂的既有测试 bug）。

**固定 SLA 并发现在可复核**（`benchmarks/analyze_sla_concurrency.py`）：

| 场景 | SLA | Full-KV 最大 batch | 有界 最大 batch | 比值 |
|---|---:|---:|---:|---:|
| 32K, B=1024 | 50 ms | 4 | 32 | **8x** |
| 32K, B=1024 | 25 ms | 2 | 32 | **16x** |
| 32K, B=4096（质量配置） | 100 ms | 4 | 32 | **8x** |
| 128K, B=1024 | 50 ms | 1 | 8 | **8x** |

128K 的 Full-KV 上限是 1 个请求（batch 2 直接 OOM），所以那一行不是延迟差异而是
显存硬墙。

## 当前审计（2026-09-23）：延迟数字的测量条件（重要）

本机与其它租户共享，这比任何调参都更影响 TPOT 读数：**同一个"有界+graph"
配置在空闲时是 6.87 ms/token，在有共租户负载时是 11.0–18.4 ms**；matched
Full-KV 则从 28.95 ms 涨到 94.2 ms，并且在 128K 频繁直接 OOM。争用对 Full-KV
一侧打击更大（它每步要复制 4 GiB 缓存），所以**繁忙窗口会放大加速比**——这是个
会误导结论的系统性偏差，必须写明。

已做的修正：
1. `lean_decode.py` 现在报 **5 次重复的最小值**（不是单次采样），parity 检查仍然
   是每个数字的闸门；
2. Full-KV 一侧可以用 `clone=False` 复用预填充张量，使两侧能在**同一进程、同一
   窗口**内背靠背测量（128K 下 Full-KV 仍需空卡，否则 OOM）；
3. 报告里每个延迟数字都标注测量窗口。头条 **4.21x** = 有界+graph 6.87 ms
   （23:04）× Full-KV dynamic 28.95 ms（22:20–22:26），同处一个空闲时段，且有界
   侧在三次独立运行中复现为 6.86–6.87 ms；
4. 质量预算（4096 槽）应报**区间 1.6x–2.6x**（11.02 ms 较空 vs 13.4–17.7 ms 较忙），
   而不是单一数字。

## 当前审计（2026-09-22 深夜）：达标配置的完整代价 + 跨模型检查受阻

**达标配置（4096 槽 + 512 锚点）的完整账**：

| 指标 | B=1024（速度优先） | B=4096（质量达标） |
|---|---:|---:|
| RULER 聚合质量（官方口径） | 1.003 | **1.000** |
| 最差任务 | 0.862 (vt) | **1.000** |
| 128K 解码态/请求 | 4 MiB | 151 MiB |
| 最大并发 batch（32K） | 32 | 32 |
| 32K 吞吐 | 1947 tok/s | 620 tok/s |
| 128K TPOT（CUDA graph） | 6.87 ms（4.21x Full-KV） | **15.84 ms（1.83x Full-KV）** |

即：**质量达标后，128K 的 TPOT 倍数从 4.21x 掉到 1.83x**（保留集大 4 倍，
注意力不再能被权重读取掩盖），吞吐从 15.6x 降到 5.0x，并发仍是 8x。选配置就是
在"质量"和"TPOT"之间取舍，两者不能同时取到最好。

**跨模型检查（Phi-3.5-mini）结论是"受阻"，不是"跳过"**：
1. 它的 remote code 没有 `cache_position`（harness 已改为按签名裁剪参数），且
   需要 Transformers 5.x 已删除的 legacy cache API（仓库自带的
   `_ensure_remote_code_compat` 已补 `get_usable_length` 等）；
2. 它是 eager attention：16K 预填需要 `(1,32,16384,16384)` 的分数矩阵（约
   15 GiB），24 GiB 卡即使用分块也放不下；8K 可行（峰值 10.98 GiB）；
3. 但在 8K 上继续解码时 legacy cache 与 5.x cache 形状不匹配（注意力维上
   3072 vs 2048）；且即便用正确模板（已与 `apply_chat_template` 核对、答案前缀
   放在 `<|assistant|>` 之后），Phi 的 Full-KV 输出仍然退化
   （`'.7.\n.\n...'`，首个 chunk 的 logits 在数字 token 上平坦到约 -29）。

所以**跨模型结论不做**：保留律目前只在 Llama-3.2-1B 上验证过。要让 Phi 跑通，
需要与它 remote code 匹配的 Transformers 版本（或转换后的 checkpoint）。

## 当前审计（2026-09-22）：达标配置的代价已量化

目标里的指标必须**同时**成立，所以把 serving 扫描改到"满足质量的那个预算"
（4096 槽 + 512 词面锚点）重跑：

| 配置 | 聚合质量 | 最差任务 | 128K 解码态 | 最大 batch | 峰值 | 32K 吞吐 |
|---|---:|---:|---:|---:|---:|---:|
| B=1024（速度优先） | 1.003 | 0.862 (vt) | 4 MiB | 32 | 6.6 GiB | 1947 tok/s |
| **B=4096（质量达标）** | **1.000** | **1.000** | 151 MiB | 32 | 11.9 GiB | 620 tok/s |

- **并发仍是 8x**（32 并发 32K 请求，matched Full-KV 上限 4），且每个 batch 都是
  100% recall —— 质量目标与并发目标由同一配置同时满足；
- **吞吐从 15.6x 降到 5.0x**（620 对 124.7 tok/s），仍在 3x 目标之上；
- 128K 下该配置：batch 1/2/4 = 60.9/122.4/249.3 tok/s，TPOT 16.4/16.4/16.1 ms，
  峰值 10.5/11.4/11.7 GiB，4/4 recall；matched Full-KV 在 128K 只能服务 1 个请求
  （34.5 tok/s、28.95 ms、batch 2 OOM）；
- 代价：每请求解码态从 4 MiB 涨到 151 MiB（仍比 128K Full-KV 的 4.00 GiB 小
  27 倍），32K 的 TPOT 从 13.7 ms 升到 21–54 ms（该区间受共享 GPU 上其它租户
  影响，128K 的 16 ms 更稳定）。

顺带把 `lex_obs` 的选择从"每 head 扫 O(L) 的 Python 循环"改成向量化 top-k +
锚点优先填充；旧实现在 128K、4608 槽时每条记录要花几分钟。

## 当前审计（2026-09-22）：计分口径修正后，质量两项达标

**发现并修正了一个计分错误。** 之前 harness 用"全部参考答案都出现"才算对，而
RULER 官方 `scripts/eval/synthetic/constants.py` 的 `string_match_all` 是**逐记录
部分召回**：

```python
score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
             for pred, ref in zip(preds, refs)]) / len(preds) * 100
```

即"命中的参考串数 / 参考串总数"，再对记录取平均。两者只在 `vt` 上不同（只有它是
多参考答案）——而 `vt` 正是之前被判 0 的任务：列出 5 个变量中的 4 个，官方记
0.8，旧口径记 0.0。**旧口径比基准更严，低估了质量。**

用同一批已保存的预测重算（无需 GPU）：

| 策略 | 预算 | 官方口径 retention | 旧严格口径 | 最差任务 |
|---|---:|---:|---:|---|
| `lex_obs` | 1024 | **1.003** | 0.980 | 0.862 (vt) |
| `lex_obs` | 2048 | **1.013** | 0.980 | 0.908 (vt) |
| `obs_last` | 2048 | 0.751 | 0.686 | 0.000 |

即**聚合质量已过 99% 线**（比值可 >1：比值是 QCC 得分总和 / Full-KV 得分总和，
个别 UUID 多键记录上有界缓存反而高于 Full-KV）。

`vt` 在官方口径下的预算曲线（128 token 生成预算）：3% → 0.864，6% → **0.970**，
13% → 0.985，25% → **1.000**，50% → 0.970，75% → 0.985。**因此 ≥97% 的最差任务
目标从 6% 预算起即满足。**

口径与协议都在 `analyze_ruler.py` / `rescore_ruler.py` 与报告 §3.13 里写明，旧数字
仍保留在 §3.6 并标注已被 §3.13 取代。

**最终数字（官方口径，`ruler_v6.json`，两侧都用 128 token 生成预算）**：

| 预算 | 聚合 | 旧严格口径 | single_1 | multikey_2 | multikey_3 | vt |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 0.997 | 0.942 | 1.000 | 0.947 | 1.222 | 0.909 |
| **4096** | **1.000** | 0.942 | **1.000** | **1.000** | **1.000** | **1.000** |
| 8192 | 1.020 | 1.000 | 1.000 | 1.000 | 1.111 | 1.015 |

即 **4096 槽预算下聚合 1.000（≥99%）与最差任务 1.000（≥97%）同时达标**，四个
RULER 任务都与 matched Full-KV 持平。4096 槽在本模型几何下是每请求 151 MiB
（128K 的 Full-KV 是 4.00 GiB，约小 27 倍）。范围仅限这个 80 条 RULER split；
LongBench / PG-19 仍未测。

## 当前审计（2026-09-22）：质量/预算前沿 + Full-KV 无法上 graph

**质量/预算前沿**（合并三个质量实验，回答"要让有界缓存追平 Full-KV，需要保留
多少上下文"）：

| 任务族 | 指标 | 追平 Full-KV 所需预算 | 该预算下的实测 |
|---|---|---:|---|
| RULER NIAH（单键/多键/UUID 多键） | answer recall | **3%**（32K 下 1024 槽） | 1.000（48/48） |
| 32K 文档语言建模 | perplexity | 6–25% | 1.09–1.12x |
| RULER vt（5 个链式变量的列表） | answer recall | **>75%**（24576 槽） | 0.75（3/4） |

vt 的预算曲线：≤25% 时 0/4，50% 时 2/4，75% 时 3/4，只有 100%（Full-KV）才
4/4。语言建模用的是**固定文档**重测（之前每次重建文档导致各预算不可比）：
3% 时 1.22x，≥6% 后 1.09–1.12x 且**非单调**（4096 与 8192 都是 1.12x，2048 与
16384 都是 1.09x），50% 也只到 1.09x。早先那个 1.75x 来自含
`/usr/lib/python3.12/*.py` 的异质语料，两个数字都保留在报告里。

**Full-KV 在 128K 无法使用 CUDA graph**：`benchmark_fullkv_tpot_graph.py` 先让
StaticCache 逐层"接管"预填充张量（同一时刻只有一份缓存，峰值 10.48 GiB），再
尝试捕获 decode step —— 捕获本身 OOM：128K 的前向需要与逐层 MLP 中间张量同量
级的 workspace（该长度下每张 2.1 GiB），叠加 4 GiB 缓存即超过 24 GiB。有界路径
能捕获，因为它的注意力只跑 1152 个 key。所以 4.21x 不是"只给对方优化"造成的
假象：**在这张卡、这个长度上，基线根本无法使用该优化**。

另：一次编辑事故删掉了 artifact 里的 serving 与语言建模两节，已从 git 历史恢复
（`b31f74e`），并重申：所有结论引用前先核对 artifact 结构完整。

## 当前审计（2026-09-21 深夜）：撤回被纠正，TPOT 配置全表

**纠正上一轮的撤回。** 之前说 StaticCache/CUDA-graph 路径 parity 不通过、耗时作废
——那是错的：**parity 检查的参照实现本身坏了**。参照用的 `DynamicLayer()` 没有设
`is_initialized=True`，于是 `get_seq_length()` 返回 0，HF 在第一次 update 时把保
留的 K/V 覆盖掉，参照输出退化成 `'Tags\n }\n return'`，任何正确实现都会"不匹配"。
修好该标志后 static 与 graph 两条路径都与 dynamic **逐 token 一致**（32K/128K 均
通过），之前的耗时数字成立。报告 §3.7 保留了这次纠正的完整记录。

**128K TPOT 全配置表**（batch 1，32 token，同一 prompt）：

| 配置 | Full-KV | 有界 | 比值 |
|---|---:|---:|---:|
| bf16，两侧都用普通 dynamic decode | 28.95 ms | 16.77 ms | **1.73x** |
| bf16，Full-KV dynamic vs 有界+CUDA graph | 28.95 ms | **6.87 ms** | **4.21x** |
| NF4 4bit，两侧都用普通 dynamic decode | 105.35 ms | 23.09 ms | **4.56x** |
| NF4 4bit，Full-KV dynamic vs 有界+graph | 105.35 ms | **4.98 ms** | 21.1x |

结论：**没有任何"口径一致"的配置达到 5x**。真正的 like-for-like 只有第一行
1.73x，与带宽上界 2.70x 吻合；4.21x 是"有界+graph 对 Full-KV 普通路径"，多出的
部分来自 graph 消除了每步 Python/mask/cache-concat（Full-KV 在 128K 用不了
static cache，会 OOM）；4bit 两行之所以比值大，是因为它**惩罚 Full-KV**（每步
4 GiB 的 DynamicCache 拼接 + bnb 非融合反量化 = 105 ms/步，比 bf16 慢 3.5 倍）。
但"有界+graph+NF4"本身是个真实结果：**4.98 ms/token，128K 下单流 201 tok/s，
缓存小 1024 倍，parity 已验证**。

环境：venv 已装 `bitsandbytes 0.50.2` 与 `accelerate`，两个 harness 都支持
`--load-in-4bit`。注意本机有其它用户的 GPU 作业
（`run_oracle_headroom_multitarget.py` 等），跑测量前先确认没有争用，否则会出现
假 OOM 和 5-8 倍的耗时抖动。

## 当前审计（2026-09-21）：vt 差距已定位，LM 质量曲线已测

**vt（variable tracking）的差距不是保留律造成的。** 四组实验
（`vt_probe.json` / `vt_probe2.json` / `vt_probe3.json`）：

1. 选择已经解决：加入赋值链跟随后，词面集合在 **20/20** 条 vt 记录上完整包含
   全部 5 个答案变量（约 290/1152 槽，离线解码已核对）；
2. 不是容量：budget 从 1024 提到 2048/4096/8192（占 32K 上下文 25%）都不解决；
3. 不是噪声：`lex_only`（只留链式行 + sink + 近期窗口，完全不含注意力选出的
   填充）在同样 budget 下同样失败；
4. 不是表头也不是生成长度：attention sink 从 4 提到 96、生成预算从 32/48 提到
   128，结果不变。

现象是模型只列出**部分**名单就漂移回指令；Full-KV 下它会重复整份名单、最终
包含全部 5 个名字。所以剩下的差距是 checkpoint 在压缩上下文下的生成稳定性，
不是保留律。因此官方 RULER 上聚合 retention 仍是 **0.941**、最差任务 **0.000**。

**语言建模质量曲线**（`benchmark_bounded_decode_lm_nll.py`，32768 token 文档，
最后 256 token 教师强制打分）：perplexity 相对 matched Full-KV 的比值在
B=1024（占 3.1%）是 **1.75x**、2048 是 1.55x、4096 是 1.34x、8192（25.2%）是
**0.98x**。`recent`/`random` 分别是 43x/115x（灾难级）。结论：保留律是**检索
专用**的——给出 100% NIAH retention 的 3% 预算会让通用语言建模的困惑度上升
75%；而语言建模下 `obs_mean`(1.56x) 反而优于 `obs_last`(1.75x)，与检索相反。

## 推送凭据（重要，2026-09-20）

本机 git 推送依赖 VS Code 的 askpass：`GIT_ASKPASS` 会通过
`VSCODE_GIT_IPC_HANDLE` 指向的 unix socket 向 VS Code server 要 token。会话
重启后环境变量可能仍指向**已死**的 socket（表现为
`remote: No anonymous write access` / `鉴权失败`）。修法是找到活的 socket：

```bash
python - <<'EOF'
import socket, glob, os
for p in sorted(glob.glob('/tmp/vscode-git-*.sock'), key=os.path.getmtime, reverse=True)[:5]:
    s=socket.socket(socket.AF_UNIX); s.settimeout(1.5)
    try: s.connect(p); print("ALIVE", p)
    except Exception: pass
EOF
VSCODE_GIT_IPC_HANDLE=<alive.sock> git push origin main
```

## 当前审计（2026-09-20 深夜）：检索质量达标，serving 两项达标

在第 4 轮基础上修掉三个选择实现缺陷后（lex_obs 曾把 head 0 的排序广播给
所有 head；frontier harness 把 lex_obs 误当作 obs_max 评分；单个 2-D mask
无法表达逐层不同的保留宽度，导致批量 decode 注意力混入 padding 槽），官方
RULER 的正确数字是：

| 策略 | B | retention | single_1 | multikey_2 | multikey_3 | vt |
|---|---:|---:|---:|---:|---:|---:|
| `obs_last` | 2048 | 0.667 | 1.000 | 0.737 | 0.000 | 0.000 |
| `lex_obs` | 1024 | **0.941** | **1.000** | **1.000** | **1.000** | 0.000 |

即三个检索任务在 B=1024（decode 缓存 32 MiB）上**全部 100% 保留**（48/48
条 Full-KV 答对的记录）。聚合 0.941、最差任务 0.000 完全由 vt 拖累：vt 是
多跳链式任务而非检索，1B 基座自身在 Full-KV 下也只答对 3/20，`lex_obs` 对这
3 条仍失败。朴素多跳词面扩展（沿匹配行里的标识符继续匹配）已实测为负结果，
会导致锚点集合涨到上限并把 multikey_2/3 拉垮，默认关闭（`hops=0`）。

Serving（32K，单卡，continuous batching 模式：逐请求精确 prefill + 常驻有界
缓存 + 批量 decode）：

| batch | decode tok/s | TPOT | peak | recall |
|---:|---:|---:|---:|---:|
| 4 | 291.1 | 13.7 ms | 5.35 GiB | 4/4 |
| 32 | 1946.8 | 16.4 ms | 6.62 GiB | 32/32 |
| 64 | 741.2 | 86.4 ms | 7.92 GiB | 60/64 |

matched Full-KV 在 32K 的硬上限是 batch 4（batch 8 必 OOM），124.7 tok/s、
TPOT 32.1 ms、peak 15.06 GiB。因此**固定 SLA（TPOT ≤ 50 ms）下并发 8x**
（32 对 4）、**吞吐 15.6x**（1946.8 对 124.7）。注意 prefill 是串行的：batch 32
的 prefill 墙钟 139.7s，本结果衡量的是显存受限并发而非 prefill 吞吐。batch-1
TPOT 仍是约 2.0x，5x 未达标（地板是每步框架开销。）

CUDA-graph/StaticCache 的 TPOT 数字仍然无效（parity 检查不过），不得引用。

## 当前审计（2026-09-20 晚）：官方 RULER 与 serving 的真实差距

全部新证据在 `artifacts/bounded-decode-frontier-20260920.md` 及其 JSON。
本轮做了三件事，结论都比上一节更严格：

1. **官方 RULER（80 条，4 任务）**：matched Full-KV 只答对 51/80——1B
   checkpoint 自身在 niah_multikey_3(UUID) 只有 9/20、vt 只有 3/20。在 Full-KV
   答对的 51 条上，`obs_last`+dilation 的 retention 是 0.529/0.608/0.667
   （B=512/1024/2048）；加上**词面锚点并集**（把问句里罕见串——连字符 key、
   UUID、长数字——在上下文中出现的位置连同左 16/右 32 token 强制保留，约
   50-125 个槽）后升到 **0.863/0.882/0.882**：niah_single_1 `1.000`、
   niah_multikey_2 `0.947-1.000`、niah_multikey_3 `0.667-0.778`、vt `0.000`。
   vt 是多跳链式任务而非检索，词面锚点只能保住被查值的直接赋值行。**因此
   99%/97% 目标仍未达到，最差任务是 0.000。**
2. **合成 60 条复现**：`obs_last` 在 B=128/256/512 上是 0.897/0.897/0.931
   （18 条子集 B=1024 为 1.000），recency 全 0/18。上一节"B=1024 达 100%"
   只成立于 18 条子集，60 条才是诚实数字。
3. **Serving（32K，单卡）**：Full-KV batch 1/2/4 = 61.6/103.2/124.7 tok/s、
   TPOT 16.2/19.4/32.1ms；有界 B=1024 = 73.6/142.7/281.0 tok/s、TPOT
   13.6/14.0/14.2ms；**两者 batch 8 全部 OOM，并发比 1.0x**。原因是 prefill
   仍按整批持有 full KV，有界只影响 decode。要达到 8x 并发必须让 prefill
   也有界/分阶段（`benchmarks/benchmark_bounded_decode_serving_sequential.py`
   是原型，尚未验证）。

**已撤回**：CUDA-graph / StaticCache 的 6.87ms TPOT 数字。新加的 parity 检查
要求每个解码路径逐 token 复现 dynamic 路径，static/graph 两条都不通过（输出
退化为重复串），所以其耗时无效、不得引用。仍然成立的是：有界 decode TPOT 在
32K/64K/128K 恒为 13.5ms，而 matched Full-KV 从 15.6ms 涨到 27.4ms，即
**128K batch-1 约 2.0x**，5x 未达标，地板是每步框架开销而非 KV 流量。

## 当前审计（2026-09-20）：有界 decode 缓存的选择律已找到

新证据在 `artifacts/bounded-decode-frontier-20260920.md`，原始结果在
`artifacts/bounded-decode-frontier-aggregate-v2.json` 与
`...-sweep-v1.json`，复现脚本为 `benchmarks/benchmark_selection_frontier.py`、
`benchmarks/benchmark_bounded_decode_frontier.py` 和
`benchmarks/summarize_bounded_decode.py`。

在真实 frozen `Llama-3.2-1B-Instruct`（16 层、8 KV heads、原生 128K）、
bf16、A10G 上做的独立诊断：prefill 用精确分块因果注意力（等价于一次长
forward，activation 有界），只在 prompt 末尾用**最后 64 个 token（问题本身）
的注意力**对每个 (layer, kv-head) 的 key 打分，做 7-token 块 max-pool 后保留
top-B，加上 4 个 attention sink 和一个近期窗口；decode 只对这个有界集合做精确
softmax 注意力，key 保留原生 RoPE 相位。

结果（6 条记录 × 3 个长度 × 2 个 key/value 对，Full-KV 在 18/18 条上正确）：

| 长度 | recent 1024 | obs_mean 1024 | obs_max 512 | obs_last 128 | obs_last 1024 |
|---|---:|---:|---:|---:|---:|
| ~32K | 0/6 | 6/6 | 3/6 | 6/6 | 6/6 |
| ~64K | 0/6 | 5/6 | 6/6 | 4/6 | 6/6 |
| ~128K | 0/6 | 6/6 | 3/6 | 6/6 | 6/6 |
| 合计 /18 | 0 | 17 | 12 | 16 | **18** |

B=128 即每 (layer, kv-head) 128 个槽位，decode 缓存 4 MiB；128K 的 Full-KV
是 4.00 GiB，缩小 `1024x`；B=1024 是 32 MiB，缩小 `128x`，并达到 `18/18`
（100% retention）。`128K -> 1M` 的有界 decode 状态增长是 `1.00x`。该策略
**零新增参数、无需校准或重训**，也**不使用未来 query**（观测窗口就是问题本身）。

边界（必须一起读）：prefill 仍是精确的，瞬时 KV 为 `O(L)`，所以“有界”只成立
于持久 decode 状态；这里只有合成 RULER-style NIAH，没有官方 RULER/LongBench/
PG-19，没有 1M-native checkpoint 结果，也没有 vLLM/TPOT/吞吐/并发测量。旧递推
archive、quality-first future-query 选择、H2O 累积质量、key-norm、recency 都在
同一 harness 下失败，说明此前的失败来自选择律与 archive 读出/混合路径，而不是
“有界状态”本身。

## 当前审计（2026-09-16）

最新真实 Phi-3.5-mini 32K 结果已记录在
`artifacts/quality-repair-20260911.md`。非因果 quality-first 在四条诊断
记录上达到 QCC answer recall `1.0`，但依赖未来查询。因果 16K、对齐
hidden writer、full-history core 加 bounded bank、prefill/decode 分阶段
探针均未恢复多键远程检索；当前没有证据支持 1M `99.5%`、5x TPOT、3x
吞吐或 8x 并发。主线应保持这个证据边界，优先实现 GQA-aware 的真实
物理状态和联合 prefill/读取路径，再进行长上下文评测。

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

### 3.7 Quality-first exact shadow（已实现，待远程验证）

- `HybridQCCArchive(..., quality_first=True)` 提供显式质量优先控制：固定容量 score-ranked exact tier、置信度门控的 nearest-neighbour exact read、按 bounded tile 的未来 query-key 相似度选择候选；容量仍与上下文长度无关。quality-first 默认不再强制放大 exact 分支，避免未校准近邻值污染 recurrent 输出。
- Hybrid exact tier 现在可接收独立的 rotary K/Q side-channel：recurrent archive 继续使用 raw position-invariant K/Q，exact shadow 用真实 local attention 的 rotary K/Q 做匹配；普通 QCCArchive 忽略该可选 side-channel，保持接口兼容。
- `calibrate_hf_admission.py`、`benchmark_hf_retrofit.py` 与 `benchmark_hf_retrieval_1m.py` 均支持 `--quality-first`。该模式用于先验证真实模型质量，不得直接当作 80% memory、vLLM 或最终 99 gate 证据；报告必须包含 exact tier 容量和实际峰值显存。

### 3.8 Phi 远程代码兼容

- `qcc_transformer.hf_loading.load_hf_causal_lm` 在 `trust_remote_code=True` 时为旧版 Phi3/Phi4 远程代码补充缺失的 `transformers.utils.LossKwargs` 类型别名。
- 该补丁只影响远程代码导入的类型注解，不改变模型权重或注意力计算；现有校准测试覆盖了该兼容路径。

### 3.9 最新远程诊断

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
- 2026-09-05 质量校准继续修复：分层校准的 `--code-init key-sample` 从每个训练 chunk 均匀抽取有界数量的 teacher K 投影来初始化 codebook；当前默认已切换为 `--code-init kmeans`，`--code-init random` 保留随机初始化消融。`--code-init-tokens`（默认 256）限制 CPU staging，仍需在真实长上下文 checkpoint 上重新校准并以 held-out/task benchmark 验证。
- 同一分层校准器新增默认 `--attention-loss-weight 0.35`：对选中层的 teacher/student attention 输出使用同一有界 token 采样监督，避免只靠最终 logits 让不同层互相补偿；设为 `0` 可复现 logit-only 消融。该局部损失不改变 adapter 参数量或最终 held-out gate 口径。
- 分层校准默认层选择已从 `last-half` 调整为 `all`：已有真实诊断中全层校准的 held-out fidelity 明显高于只训练后半层；后半层仍可显式指定作为参数预算消融。全层仍只开放 QCC archive/gate（约 0.1% 参数），不改变部署接口。
- 分层校准训练路径现在优先调用 HF backbone 的 `last_hidden_state`，按词表 tile 计算与原实现等价的 MSE/KL/CE/margin loss，避免 GPU 保留完整 student logits；不具备标准 `model`/output-embedding 接口的模型仍走原 full-logit fallback。该改动已在真实 HF tiny Llama 流程验证，需在真实长上下文模型上重新校准。
- 2026-09-05 后续质量校准：`--code-init kmeans` 已成为默认，采用 teacher key 的确定性 cosine k-means 初始化；`--distill-long-range-only` 默认只反向优化窗口之外的 archive 位置。两项改动尚未在真实长上下文 checkpoint 上产生新的 gate 结果。
- 2026-09-05 T4 复核暴露两项工程问题并已修复：分层校准释放 teacher 时仍保留 attention 子模块引用，加载 student 会被系统 `SIGKILL`；现保存层数整数后显式删除模块列表并执行 `gc.collect()`。Phi-3.5 attention-output 辅助损失改为 fp32 计算，避免 fp16 平方溢出；新增单元测试覆盖该数值边界。
- 同轮真实 Phi-3.5-mini T4 诊断：全层 512-token 反向仍因 activation peak OOM；last-quarter（8/32 层，0.0313% 参数）50 步可完成，但 held-out cosine `0.9148`、top-1 `0.4463`，未通过 `0.99`。固定容量 quality-first exact shadow 在 512-token matched HF 上最高记录 cosine `0.9687`、top-1 `0.5781`，仍不能替代 RULER/LongBench/PG-19。
- 2026-09-05 质量修复：quality-first exact tier 不再按 tile FIFO 覆盖，而是使用全局 score-ranked 固定表；每个 bounded prefill tile 用采样的未来 query-key cosine 选取候选，避免长流中早期 salient token 被后续 tile 无条件淘汰。普通 admission 模式保持原有 score/FIFO 配置不变；新增跨 tile salient-token 回归测试。该修复尚未产生新的真实模型 benchmark 数字，旧质量结果不自动迁移。
- 2026-09-05 质量数值路径：`QCCSelfAttention` 新增 `local_attention_backend={sdpa,eager}`。`eager` 保留 HF 的原始 dtype QK matmul、缩放、fp32 softmax、V matmul 顺序，用于 top-1 敏感的 matched Full-KV 质量对照；CUDA 非 Triton fallback 同时改用有界 SDPA，避免大窗口 unfold 产生 GiB 级临时张量。`benchmark_hf_retrofit.py` 新增 `--use-triton/--no-use-triton`、`--local-attention-backend`、`--prefill-chunk-size`，便于复核 kernel 与 chunk 边界影响。
- 2026-09-05 真实 Phi-3.5-mini-instruct A10G 质量复核：同一 3.8B pretrained checkpoint、HF `eager` Full-KV 对照、`window=4096`、bf16、4K prompt、`local_attention_backend=eager` 得到 mean logit cosine `1.0000166`、top-1 `100%`；`window=8192`、NF4（两侧同量化）、8K prompt 得到 cosine `1.0000185`、top-1 `100%`，均通过 matched `0.99` 双指标 fidelity gate。未量化 8K eager 因 A10G 两份模型加临时张量超出显存；16K NF4 探测同样 OOM。上述是 matched Full-KV fidelity diagnostic，不是 RULER/LongBench/PG-19，也不能外推到 128K/1M。
- 2026-09-05 质量对照（真实 Phi-3.5-mini-instruct，A10G，4 条独立 held-out，共 2436 tokens）：未校准 `window=1024, codes=16` 的平均 logit cosine `0.998933`、top-1 `0.978468`（2/4 记录同时达到 `0.99`）；`gate_bias_init=4` 对照为 cosine `0.998717`、top-1 `0.977664`。同一首条记录的 regular `window=128` adapter 为 cosine `0.920406`、top-1 `0.378617`，quality-first fixed exact shadow 为 cosine `0.976046`、top-1 `0.448553`；扩大 exact capacity 到 1024 slots 仍为 cosine `0.975927`、top-1 `0.449357`，说明容量不是主要瓶颈。上述均为 matched HF fidelity diagnostics，不是 RULER/LongBench/PG-19，也未通过双指标 `0.99` gate。
- 2026-09-05 质量实现修复：quality-first 改为 hard nearest read 并尊重调用方 mix bias；Hybrid exact tier 接入 rotary K/Q side-channel，避免 raw archive addressing 与 Full-KV historical matching 的相位错位。新增 `test_hybrid_exact_shadow_uses_rotary_side_channel`，本地和远端 hybrid 单测通过；该修复尚未改变官方任务缺口。
- 2026-09-05 质量校准补偿：`QCCArchive` 新增默认 rank `8` 的零初始化 query-conditioned low-rank residual，作用于 archive read、位于 local/archive mixing 之前；未校准输出严格保持不变，校准时可用约 `0.03%` 级别额外参数学习系统性长程偏差。`calibrate_hf_retrofit.py`、`calibrate_hf_layerwise.py` 和 `benchmark_hf_retrofit.py` 均支持 `--archive-query-correction-rank`；旧 adapter 缺少该键时自动保留零初始化。新增 archive identity/legacy adapter 回归测试，尚未产生真实长上下文或官方 benchmark 数字。
- 分层校准新增显式 `--cpu-offload-activations`：用 PyTorch `save_on_cpu` 将 autograd 保存张量移到主存，目标是在小显存卡上重新尝试全层校准；默认关闭，尚未形成新的真实质量结果。
- 2026-09-05 质量优先校准：默认 `--distill-long-range-only` 只对超出 exact local window 的位置计算蒸馏损失，避免已精确匹配的前缀 token 稀释 archive 梯度；`--no-distill-long-range-only` 保留旧的全序列消融。`--code-init` 默认改为确定性 cosine k-means，`key-sample` 与 `random` 仍可复现旧初始化。
- 2026-09-05 质量修复：HF quality-first hybrid 预填充现在先构造整段请求的 rotary query side-channel，再按普通 bounded chunks 执行 archive/local attention；因此 early needle 的 exact-tier salience 能看到后续 query，同时不强制保留 full-sequence archive 临时张量。`benchmark_hf_ruler.py` 新增 `--quality-first`，可直接用 matched Full-KV/QCC 跑官方 RULER split。
- 2026-09-05 T4 复核：真实 `Qwen/Qwen2.5-0.5B-Instruct`、470 tokens、quality-first exact tier（32 sets x 8 ways）、显式 128-token chunks 的 matched HF logit cosine/top-1 为 `0.83514/0.95745`；同配置旧的块内 salience 路径为 `0.78942/0.95106`。这是短上下文实现诊断，未校准、非官方 RULER/LongBench/PG-19，不能外推到长上下文质量。
- 2026-09-05 Colab T4 复核（同一真实 `Qwen/Qwen2.5-0.5B-Instruct`、470 tokens、显式 128-token prefill chunks）：regular `window=128` 为 cosine/top-1 `0.51829/0.37447`，quality-first exact tier（128 sets x 4 ways）为 `0.58413/0.37021`；将 exact local window 提高到 `512` 后 regular 与 hybrid 均为 `0.99981/0.98511`，因为该输入没有发生历史淘汰。这验证了短上下文质量失败主要由窗口不足造成，但不代表 128K/1M 或官方任务质量。
- 同轮 Qwen quality-first exact read 对照：在相同 470-token 输入和 `window=128` 下，hard read 为 cosine/top-1 `0.56581/0.36170`；临时切换 soft read 为 `0.60030/0.34894`，降低 confidence threshold 后为 `0.58329/0.37021`。cosine 与 top-1 没有一致改善，因此保留 hard-read 默认，不把该短样本当作质量结论。
- 2026-09-05 质量评测默认值：`benchmark_hf_retrofit.py` 与 `benchmark_hf_ruler.py` 默认 `--window-size` 调整为 `1024`，性能对照仍可显式传 `128`；quality-first 帮助文本更正为 hard nearest reads。更大窗口只改变有界 exact local state，不能替代长上下文 RULER/LongBench/PG-19 结果。
- 2026-09-05 Qwen2.5-1.5B NF4 质量复核暴露并修复两项数值问题：量化 Q/K 的 fp16 局部点积会溢出为 NaN，现改为 fp32 logits/softmax 累加；校准 cosine/MSE 的词表归约也改为 fp32。短文本不足以越过 `window_size` 时，校准 CLI 现在直接拒绝，避免生成无 archive 梯度的假 adapter。完整本地回归为 `89 passed, 7 skipped`，修复已推送提交 `67ffc35`。
- 2026-09-05 后续质量修复：quality-first hybrid exact tier 现在把每次 exact read 的置信度以有界 side-channel 暴露给 `QCCSelfAttention`；仅在高置信 exact 命中时动态压低 local/archive gate，未命中仍保持原保守 gate。这样不会改变普通 QCC 或未命中 query 的行为，并新增跨 tile exact-confidence 回归测试。
- 2026-09-05 长程质量增量：`QCCArchive` 复用低秩 query residual 的坐标新增零初始化的 query-conditioned decay-scale selector。校准后每个 query 可在短/长时间尺度之间动态配比，避免所有 token 共用静态 scale mix；未校准输出保持不变，旧 adapter 缺失该键时按零 selector 加载。新增 archive 回归测试，尚未产生真实 RULER/LongBench/PG-19 数字。
- 2026-09-05 后续校准修复：`calibrate_hf_layerwise.py` 新增 `--validation-interval`（默认 10），在 held-out chunks 上按 `min(cosine, top-1)` 选择并恢复最佳 trainable adapter，避免最后一步训练集过拟合覆盖更好的中间 checkpoint；报告和 adapter manifest 同时记录最佳验证指标。该流程改进尚未产生新的真实 RULER/LongBench/PG-19 结果。
- 2026-09-05 质量校准修复：普通 `calibrate_hf_retrofit.py` 现在与分层校准器一致，默认用 teacher projected-key 的确定性 cosine k-means 初始化 archive codebook（`--code-init kmeans`）；`key-sample` 与 `random` 保留为显式消融。通过 `--code-init-tokens` 限制 teacher hidden 的 CPU staging，初始化完成后 teacher 仍会在加载 student 前释放。该改动已在本地真实 HF tiny checkpoint 完成 smoke 与 adapter 回载验证，但尚未形成 Phi/Qwen 长上下文或官方 benchmark 结果。
- 2026-09-05 质量 benchmark 工程修复：`benchmarks/benchmark_hf_retrofit.py` 移除了传给 `patch_hf_model` 的不存在参数 `archive_query_scale_selector`；该参数只是 metadata 概念，selector 实际由 `archive_query_correction_rank` 控制。新增 retrofit 回归测试覆盖 rank/scale-selector 初始化，避免真实质量评测在启动阶段因接口错误失败。
- 2026-09-05 质量配置调整：`benchmark_hf_retrofit.py`、`benchmark_hf_ruler.py`、`benchmark_hf_longbench.py`、`benchmark_hf_pg19.py` 与 `benchmark_hf_retrieval_1m.py` 的默认 exact local window 统一为 `4096`；性能脚本仍保持 `128`，需显式传参才可作速度配置。该默认值调整本身不构成官方质量结果。
- 同轮真实 `Qwen/Qwen2.5-1.5B-Instruct`、T4、NF4、`window=128/codes=64` 的 5-step README 校准在训练段为 cosine `0.93330`、top-1 `0.65625`；独立 5 条 held-out（共 `4033` tokens）regular QCC 为 cosine `0.79167`、top-1 `0.47688`。quality-first exact shadow 单条 held-out（`588` tokens）在 128-slot 和 512-slot 配置分别为 `0.89995/0.64626` 与 `0.89966/0.64456`；仍未达到 `0.99` fidelity，更不是 RULER/LongBench/PG-19 证据。远端临时会话随后丢失，adapter/log 未作为本地证据归档。

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
