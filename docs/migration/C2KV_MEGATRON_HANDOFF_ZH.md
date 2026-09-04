# C2KV Megatron-SWIFT 迁移与接管报告

更新日期：2026-09-05

目标分支：`megatron`

上游基线：`s7a9/C2KV@832ccd9`

个人 fork：`flyinglandlord/C2KV`

## 1. 结论与证据边界

本分支已把 C2KV 的训练主路径重构为 Megatron Core 原生单次结构化 forward，训练入口由
Megatron-SWIFT 调度；Transformers 只保留在导出后的评测兼容层。仓库本身不引入
`pyproject.toml`、editable install、Poetry、uv 或 Hydra，全部入口均从仓库根目录直接运行。

截至本报告更新时，证据分为三层：

| 层级 | 状态 | 已验证内容 |
| --- | --- | --- |
| 本地纯逻辑 | 通过 | 配置、压缩率调度、layout、mask、RoPE 位置、蒸馏对齐、JSON/JSONL 索引，共 17 个单元测试 |
| 本地静态入口 | 通过 | 全量语法编译、训练/导出/评测 dry-run、`git diff --check` |
| H200 运行时 | 阻塞 | SSH 在跳板阶段超时，尚未形成 CUDA/MCore 训练、checkpoint、resume 或 W&B 证据 |

“本地通过”不能替代 GPU 正确性。本报告不会把尚未执行的 H200 smoke、数值 parity 或
W&B run 写成已完成。

## 2. 仓库和 Git 接管信息

```text
origin    https://github.com/flyinglandlord/C2KV.git
upstream  https://github.com/s7a9/C2KV
branch    megatron
base      832ccd9 add longmagpic preprocess file
```

推荐后续同步方式：

```bash
git fetch upstream
git switch megatron
git rebase upstream/main
git push --force-with-lease origin megatron
```

只有在明确检查过上游冲突后再 rebase。训练产物、W&B 凭据、模型权重和数据不进入 Git。

## 3. 迁移后的训练数据流

```text
JSON / JSONL
  -> JsonRecordStore（JSONL 仅保存 byte offset）
  -> C2KVExample（统一 paper-level schema）
  -> TokenizedExample
  -> LayoutCompiler
       input_ids / labels / position_ids
       token_roles / document_ids / objective_kinds
       MemorySlot(compression_source_indices, visible_document_indices)
  -> C2KVCollator
       dense blocked attention mask
       memory selector / special-token selector / loss selector
  -> C2KVGPTModel + C2KVSelfAttention
  -> C2KVMegatronTrainer
       LM + reconstruction + optional distillation + optional QKV regularization
  -> native MCore checkpoint
  -> C2KVBridge
  -> Transformers safetensors
  -> preserved evaluation scripts
```

与旧 `Trainer.compute_loss()` 内多次调用模型、拼 `DynamicCache` 相比，新路径把文档、memory、
query、response 和 reconstruction 编译到一个 forward 中。这样 pipeline scheduler 能看到完整
计算图，数据语义也不再隐藏在 Trainer 的临时缓存里。

## 4. 论文语义与代码命名

| 论文含义 | 统一代码名 | 不再复用的含糊含义 |
| --- | --- | --- |
| 压缩后的 KV token | `MEMORY`, `memory_slot`, `memory_indices` | 不用 `gist_num` 同时表示 token 和 chunk |
| 被一个 memory slot 压缩的原始 chunk | `compression_source_indices` | 不与 sink/overlap 混称 `source` |
| memory 可见的文档 token | `visible_document_indices` | 明确包含 sink、overlap、当前 chunk |
| 文档编号 | `document_id` | 不用 batch 内临时循环变量表达归属 |
| 重建起始标记 | `RECONSTRUCTION_MARKER` | 与被重建的正文 token 分离 |
| 回答损失 | `LANGUAGE_MODEL` / `language_model` | 不用笼统 `loss` 判断 token 归属 |
| 重建损失 | `RECONSTRUCTION` / `reconstruction` | 与回答 loss 分开标记和加权 |
| C2KV 投影 | MCore `c2kv_linear_qkv` | HF 导出仍映射为论文兼容的 `gist_{q,k,v}_proj` |

`LayoutPlan` 是最重要的稳定接口：任何新数据集、新模型架构或新 sparse kernel 都应消费同一份
layout 语义，不应各自重新推导 token 位置和 mask。

## 5. Attention 和位置语义

结构化 mask 的可见性规则如下：

1. system token 只看 system causal prefix；
2. 每个 document 只看 system 和自身 causal document token，不看其他文档；
3. memory slot 看 system、同文档 sink、overlap、当前 compression chunk、此前同文档 memory 和自身；
4. query/response 看 system、全部 memory，以及 query/response causal prefix，不看 raw document；
5. reconstruction 看 system、目标文档 memory，以及本 reconstruction span 的 causal prefix。

`True` 在传给 MCore 的四维 mask 中表示 blocked。dense mask 是正确性 oracle，当前硬上限由
`dense_mask_max_tokens` 控制。

位置使用“逻辑拼接”语义：每个文档的 token 和 memory 先按本地 chunk 末尾定义位置，再根据
前面文档长度整体平移；query/response 位于所有逻辑文档之后。这与评测时独立 prefill 后拼接
memory KV 的相对 RoPE 位置保持一致。

## 6. 可训练参数和初始化

默认只训练：

- 每层 `c2kv_linear_qkv`；
- 两行 `c2kv_special_embeddings`：memory marker 与 reconstruction marker。

基础 token embedding、基础 QKV、MLP、norm 和 LM head 均冻结。基础模型首次迁移时：

- `c2kv_linear_qkv` 从基础 fused QKV 精确复制；
- `residual_type=none` 时，两行特殊 embedding 从 memory token 对应的基础 embedding 复制；
- `mean` 或 `embedding_mean` 时，特殊 embedding 从零开始，chunk mean 提供残差输入；
- 从带 `c2kv_config.json` 的已导出 C2KV checkpoint 启动时，不覆盖已训练参数；
- native MCore resume 的最终权重由 checkpoint loader 恢复。

`mean` 在每一层把 compression chunk 的 hidden-state mean 加到 memory QKV 投影输入；
`embedding_mean` 只在第一层执行。残差来源使用 `compression_source_indices`，不会误把 sink 或
overlap token 混入均值。

## 7. Loss 定义

主训练返回 per-token cross entropy，由 `c2kv_objective_ids` 在 Megatron 完成 label shift 后同步
shift，再按以下权重求和：

```text
L_token = w_lm * sum(LM-token CE) + w_rec * sum(reconstruction-token CE)
```

QKV regularization 对各层 C2KV/base fused QKV 的均方差取平均，并按 active token 数加入 loss，
从而保持 Megatron token-normalized loss 约定。

self-distillation 使用去除 memory 与 reconstruction span 的完整上下文作为 teacher。只有能与
teacher 对齐的 LM token 做 KL；重建 CE 不会因开启蒸馏而丢失：

```text
L = L_non_distillable_CE
    + (1 - alpha) * L_distillable_CE
    + alpha * T^2 * KL(teacher || student)
```

当前 self-distillation 只允许 PP=1，因为两次 forward 尚未接入 Megatron pipeline schedule。

## 8. 目录职责

| 路径 | 责任 | 后续修改原则 |
| --- | --- | --- |
| `c2kv/core/` | layout、mask、位置、压缩率、稳定类型 | 不 import torch/Megatron/Transformers |
| `c2kv/data/` | schema、文件索引、adapter、collator | 新数据集优先加 adapter，不改 trainer |
| `c2kv/megatron/` | MCore model/attention/bridge/trainer | 框架版本相关代码集中在这里 |
| `c2kv/transformers/` | 旧评测模型与 cache 的隔离 facade | 训练代码禁止反向 import |
| `c2kv/eval/` | 评测 backend、配置、JSONL/metric 接口 | 新评测后端通过 registry 增加 |
| `configs/train/` | JSON 训练实验 | 每个可复现实验独立文件 |
| `configs/eval/` | JSON 评测实验 | 输出路径不写死在 Python 中 |
| `tests/unit/` | 纯 CPU/标准库语义测试 | 每次论文语义变更必须先更新 |
| `tests/parity/` | HF/MCore 数值门禁 | 必须在 GPU 环境执行 |
| `tests/integration/` | 分布式训练产物门禁 | checkpoint/resume 后执行 |
| `python/` | 冻结的 legacy Transformers 实现 | parity 完成前不删除 |

## 9. 直接运行方法

环境检查与本地门禁：

```bash
python check_environment.py \
  --config configs/train/qwen3_0_6b_smoke.json \
  --strict --require-wandb
python run_tests.py --suite all
python train_megatron.py --config configs/train/qwen3_0_6b_smoke.json --dry-run
```

单卡 smoke：

```bash
torchrun --nproc_per_node=1 train_megatron.py \
  --config configs/train/qwen3_0_6b_smoke.json
```

导出与 Transformers 评测：

```bash
torchrun --nproc_per_node=1 export_megatron.py \
  --config configs/train/qwen3_0_6b_smoke.json \
  --checkpoint outputs/qwen3-0.6b-smoke/checkpoint-2 \
  --output outputs/qwen3-0.6b-smoke-hf \
  --test-precision

python evaluate.py --config configs/eval/qwen3_4b_hotpotqa.json
```

环境由服务器独立维护；仓库只检查版本和 import，不负责安装 Python 包。

## 10. 并行能力矩阵

| 能力 | 当前状态 | 原因/下一门禁 |
| --- | --- | --- |
| DP | 设计支持，待 H200 | 标准 Megatron data parallel |
| TP | 设计支持，待 H200 | fused C2KV QKV 使用同一 ColumnParallelLinear 规格 |
| Sequence Parallel | `residual=none` 设计支持，待 H200 | memory/special selector 按 TP sequence shard 对齐 |
| SP + residual | 配置拒绝 | chunk mean 需要跨 sequence shard 聚合 |
| PP 主 loss | 设计支持，待 H200 | 单 structured forward，label 只在 last stage |
| PP + self-distill | 配置拒绝 | teacher/student 双 forward 尚未调度化 |
| PP + QKV regularization | 配置拒绝 | 需要跨 PP 聚合各层正则项 |
| CP | 配置拒绝 | arbitrary C2KV mask 尚未接入 CP split/通信语义 |
| EP/MoE | Qwen MoE 注册，待专项门禁 | attention 可复用，router/padding 行为仍需实测 |
| MTP | 配置拒绝 | MTP layer 尚未共享 C2KV structured mask/spec |

“设计支持”不是“已验证”。在大规模训练前必须按下一节逐项关闭门禁。

## 11. 验收门禁

### Gate A：单卡 source-to-runtime

- runtime imports 全部成功；
- Qwen3-0.6B 两步 BF16 smoke 无 NaN/OOM；
- 只有 `c2kv_linear_qkv` 和 `c2kv_special_embeddings` 有梯度并发生更新；
- base 权重 hash 在前后不变；
- W&B 记录 loss、tokens、学习率、吞吐、显存；
- 生成 `checkpoint-2`、HF safetensors 和 `c2kv_config.json`。

### Gate B：HF/MCore 数值 parity

- 同一小 batch 对齐 input/layout/mask/position；
- FP32 对齐 memory Q/K/V、最终 logits、LM loss、reconstruction loss；
- BF16 设定明确误差阈值并保存逐层最大/平均误差；
- HF -> MCore -> HF round trip 检查 `gist_{q,k,v}_proj` 和特殊 embedding；
- 固定 greedy generation 对齐已保留的 Transformers eval。

### Gate C：并行和恢复

- TP=2 与 TP=1 的 loss/gradient parity；
- TP=2 + SP 的 `residual=none` 门禁；
- PP=2 主 loss 的 bubble/label/selector 正确性；
- 保存后 resume 两步，loss、global step、optimizer/RNG 连续；
- 多节点前先完成 DP=2 和 NCCL failure recovery。

### Gate D：规模化性能

- 实现 sparse backend，不调大 dense cap 冒充扩展；
- 相同 tokens/global batch 比较 legacy DeepSpeed 与 MCore 的 tokens/s、MFU、peak memory；
- 衡量当前“双 QKV projection + `where`”的额外开销；
- 决定采用 grouped sparse GEMM、memory-row gather，或专用 fused projection kernel；
- 长上下文、动态 ratio、多文档数量分桶后再进入正式预训练。

## 12. H200 状态与复跑清单

本轮两次连接 `H200-2035` 均在认证提示出现前失败：

```text
Connection timed out during banner exchange
Connection to UNKNOWN port 65535 timed out
```

配置展开显示目标通过 SSH jump host 转发；对 jump host 的 TCP 22 探测同样无响应。因此当前
阻塞点是本机到跳板的网络/VPN/主机状态，不是已确认的密码错误。尚未创建远端
`c2kv-megatron` 环境，尚未上传代码，尚未产生真实 W&B run。

网络恢复后的操作顺序：

1. 登录并记录 `hostname`、GPU、driver、CUDA、Python、磁盘；
2. 创建或克隆独立 `c2kv-megatron` 环境，不修改仓库包管理文件；
3. 从 `origin/megatron` clone/pull，核对 commit；
4. 运行 `check_environment.py --strict --require-wandb`；
5. 运行全部本地测试与 dry-run；
6. W&B 通过环境变量或交互登录，密钥不落盘、不写报告；
7. 执行 Gate A，两步 smoke；
8. 设置 `C2KV_SMOKE_OUTPUT` 后跑 integration test；
9. 保存 W&B run URL、checkpoint 路径、环境版本和失败日志摘要；
10. 继续 Gate B/C，未通过前不启动大规模作业。

## 13. 已核对的外部 API 基线

静态实现对以下源码快照做过接口核对：

| 组件 | 核对 commit | 关键接口 |
| --- | --- | --- |
| ms-swift | `16e89c8b713055c8f8452c2a697e4866839172a3` | `MegatronSft`, `MegatronTrainer`, batch/label shift, save hooks |
| mcore-bridge | `4f2a95c9548bcd588b6f7763074debcdbf6b2d69` | `ModelLoader`, `GPTBridge`, HF/MCore QKV conversion |
| Megatron-LM | `6572312c971ce57a7371cbdc45e9785add82bd49` | `SelfAttention`, `GPTModel`, sequence parallel embedding |

H200 环境若不是这些接口对应的版本，应先跑 import/runtime parity；不要通过吞掉异常来兼容未知版本。

## 14. 后续论文改进接口

### 14.1 新压缩策略

扩展 `CompressionSchedule` 或新增一个 schedule 实现，让它只决定每个 sample/epoch 的 ratio；
不要让 schedule 操作 tensor 或模型。若需要 learned allocation，可把确定后的 chunk plan 写进
`LayoutPlan`，保持 mask 和评测可重放。

### 14.2 新 memory 可见性或分层 memory

扩展 `MemorySlot` 与 `allowed_key_indices()`，同时增加 dense/sparse 一致性测试。不要直接在 CUDA
kernel 里定义一套未被 CPU oracle 覆盖的语义。

### 14.3 大模型和 MoE 架构

先在 `c2kv/megatron/architectures.py` 注册 capability；标准 GQA attention 可复用现有 loader。
MLA、linear attention、hybrid Mamba、DSA 等必须增加专用 attention/module spec 与 bridge，禁止
把它们伪装为 `C2KVSelfAttention`。MoE 还要单独验证 padding mask、router loss 与 EP checkpoint。

### 14.4 Block-sparse H200 kernel

`iter_allowed_key_ranges()` 是稳定输入边界。新 backend 应：

- 保留 dense oracle；
- 明确 block size、尾块、不同 document/memory role 的布局；
- 提供 forward、dQ/dK/dV、TP/SP/CP 数值测试；
- 在 collator 中输出压缩后的 range/block metadata，不再分配 `[B,1,S,S]`；
- 记录编译缓存、吞吐和显存，不只报告最终 loss。

### 14.5 新目标函数

token 级目标先扩展 `ObjectiveKind` 和 collator selector，再在 trainer 聚合。跨层或跨 stage 的目标
必须显式定义并行归约，不能在某一 rank 本地静默计算。

### 14.6 评测管线

现有 `evaluate.py` 是稳定入口，legacy scripts 是 backend。下一步可逐个把 dataset、generation、
metric 移到 `c2kv/eval/`，每迁一个 backend 都保留固定样本输出 parity；训练侧不依赖评测实现。

## 15. 建议的后续任务拆分

1. **P0：恢复 H200 网络并关闭 Gate A。** 这是当前唯一阻止“训练正确”结论的外部条件。
2. **P0：补齐 HF/MCore parity。** 尤其是 Qwen3 q/k norm、RoPE、特殊 embedding、checkpoint round trip。
3. **P1：实现 sparse mask backend。** dense 4096 只适合 correctness，不适合更大规模预训练。
4. **P1：TP=2 + SP 与 PP=2。** 把“设计支持”转成可复现证据。
5. **P1：吞吐优化 memory-only QKV。** 避免所有 token 同时计算 base/C2KV 两套投影。
6. **P2：Qwen3-MoE/目标大模型专用 architecture gate。** 先单层、再单卡、再 EP。
7. **P2：逐步替换 legacy eval。** 以输出 parity 为删除旧代码的唯一条件。

## 16. 不应删除的兼容边界

- 在 HF/MCore parity 完成前保留 `python/`；
- 在 sparse kernel 完成前保留 dense oracle；
- native MCore checkpoint 始终是训练 resume 的权威格式；
- HF safetensors 是评测/发布格式，不替代 optimizer/RNG resume；
- 所有服务凭据只走环境变量或交互输入，不写 JSON、shell、Git、日志或报告。
