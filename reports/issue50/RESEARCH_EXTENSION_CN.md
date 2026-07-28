# YOLO-Master Issue #50：稳定性、强基线与可信结果审计扩展

## 1. 项目定位

官方 Issue #50 的基础交付由多个上游贡献逐步形成，而非由单个 PR 独立完成。本研究不重复宣称完成该任务，而是在这些官方实现之上回答三个更严格的问题：LoRA 训练为什么会失稳；它与常见微调策略相比处于什么位置；怎样保证最终结论可审计、可复现且不被失败运行或重复样本污染。

研究因此分为两个阶段。Phase 1 复现并定位 AMP 条件下的数值异常，形成 Stable LoRA 配置，并完成 rank 与初步多随机种子比较。Phase 2 增加 Head-only、Neck + Head、Last Stage + Neck + Head、Full Fine-tuning、Stable LoRA 和 Partial Fine-tuning + LoRA 等强基线，再对参数量、显存、时间、推理速度、精度和稳定性做统一审计。腾讯上游已经合并的 PR #178 是本项目衍生的实验协议澄清与 CSV 防覆盖贡献；它改善的是结果管理和复现可信度，不是训练精度优化 PR。

本文的正式数值以审计后的 `PHASE2_FINAL_RESULTS.csv`、`PHASE2_SEED_SUMMARY.csv` 和 `PHASE2_PARETO_SUMMARY.csv` 为准。旧 Markdown 只用于理解过程；若与审计 CSV 冲突，以后者为准。

### 相关上游工作

Issue #50 的上游演进包括：PR #69 提供早期场景配置、rank sweep 和历史结果；#84 建立面向 v0.10 的六组 LoRA 实验管线；#102 补充跨 rank 矩阵与双语适配指南；#135 统一 README、VisDrone `fraction` 与运行脚本；#166 增加场景化配置、Router 策略和另一组结果；#178 是本项目贡献的协议标识、CSV schema 对齐与防覆盖修复。开放的 #114 扩展其他数据集，#179 是尚未合并的另一套 Issue #50 rank sweep，本项目不依赖二者的结果。

### 与上游已有 LoRA 稳定性修复的边界

本项目不宣称首次发现或解决所有 AMP、LoRA 或 RS-LoRA 稳定性问题。上游 #124 修复的是 MoE 稀疏分派中 AMP `index_add_` 的 dtype 对齐；#125 修复 LoRA + MoE 在 DDP 中的 ready-twice；#170 让 fallback 后端真正使用 RS-LoRA scaling；#177 继续修复 fallback alpha warmup 在 EMA、验证、保存和 resume 生命周期中的同步。本项目则是在自己的实验协议中，首先观察到 LoRA Adapter 的非有限梯度，并用 AMP=False、Adapter 0.1×与 Router 0.5×分组学习率进行单变量诊断。这些问题边界相关，但不是同一个缺陷。

配置与正式汇总证明，本项目 LoRA 配置请求 `lora_backend=auto`、设置 `lora_use_rslora=True`，实际正式 rank sweep 的 `effective_lora_backend` 为 **PEFT**，不是 fallback。实验 commit 已包含 #124 和 #125，但不包含随后合并的 #170 与 #177；后两项主要针对 fallback 路径，因而不是这些 PEFT 运行的实际执行路径。尽管如此，结果仍属于当时研究分支实现，不能被解释为当前 `upstream/main` 修复后 RS-LoRA 的最终性能。本项目保留历史结果且不为此重跑训练。

## 2. 从异常到稳定配置

最初的正式链路使用 AMP。日志直接证明：在第 1 个 epoch 的第 0 个 step，首个被定位的非有限参数梯度是 `model.base_model.model.4.conv.lora_A.default.weight`；同一步的 box、cls 和 dfl loss 仍为有限值。这说明异常首先暴露在 Adapter 梯度，而不是由三个已记录检测损失先变为 NaN。源码与日志还证明训练器会检测非有限状态，并通过健康检查点进行 recovery；因此“训练最终有结果”不能自动等同于“全过程稳定”。

单变量诊断显示，关闭 AMP 后未再出现同类非有限梯度，数值稳定性明显改善。保持 FP32 时，将 Adapter 学习率倍率从 1.0 降到 0.1，Brain Tumor 诊断 mAP50-95 从 0.02245 提高到 0.06630；原始 AMP 诊断运行为 0.02424，并伴随 non-finite/recovery。该比较支持“抑制 Adapter 更新可减轻小数据场景中的特征漂移”，但它仍是诊断证据，不是跨任务定律。最终稳定方案使用 Adapter 约 0.1×基础学习率、Router 约 0.5×基础学习率，并让随机初始化的 Detection Head 保持 1.0×。这些倍率对本项目有效，不能声称是所有数据集的通用最优值。

“AMP-safe LoRA”原计划让普通网络使用混合精度、关键 Adapter 计算保持 FP32，以兼顾速度与稳定性；实际实现因缺少 `Conv2d.amp_safe_forward` 等接口在训练开始前失败。因此它的正确状态是 `implementation_failed / not_validated`，不能列为已验证创新，也不能进入正式均值或 Pareto。

## 3. 方法与公平性

Brain Tumor 使用完整训练划分；VisDrone 的 LoRA 协议使用 `fraction=0.2`，所以其结论仅适用于该数据比例。正式对比固定同一数据集内的数据划分、seed 规则、图像尺寸、epoch 上限和增强策略。比较方法包括：

- **Head-only**：仅训练重新初始化的检测头；
- **Neck + Head**：解冻特征融合颈部和检测头；
- **Last Stage + Neck + Head**：进一步解冻 Backbone 最后一阶段；
- **Full Fine-tuning**：训练全部参数；
- **Stable LoRA**：冻结主体，以稳定分组学习率训练 Adapter、Router 和 Head；
- **Partial fine-tuning + LoRA**：部分解冻与 LoRA 联合训练。

审计把 `requested_batch` 与最终 `args.yaml` 中的 `actual_batch` 分开。VisDrone Full FT 曾请求 batch=8，OOM 后自动降到 4；恢复运行被标为 `oom_recovered_diagnostic`，不与干净的正式 batch=4 运行重复聚合。Head-only 的部分 b4/b2/b1 队列项只是重复引用同一旧目录，并未真实启动，故标记为 `not_executed`。队列执行状态与正式有效性也分别记录：26 个队列任务中 15 completed、11 failed，并不等价于最终证据表中“通过稳定性门控”的数量。

### 证据层级

为避免把推断写成事实，本项目把结论分成四级。第一，日志直接证明运行时发生了什么，例如非有限梯度出现的位置、OOM 文本、恢复提示、实际训练轮数和验证指标。第二，源码直接证明机制如何实现，例如非有限状态的检测条件、`last_healthy.pt` 的保存与恢复路径、参数冻结规则和优化器分组。第三，合理推断用于解释现象，例如 Adapter 更新过快可能导致特征漂移；这类解释即使与单变量结果一致，也不能替代专门的特征统计实验。第四，尚未验证的设想包括 Adapter-only FP32、独立 GradScaler 和动态 rank，均不能进入已验证贡献。

这种分层尤其影响“恢复”的解释。健康检查点可以提高无人值守训练的容错性，却不能抹去此前发生的非有限更新。如果一次运行触发 recovery，它仍应保留为失败或诊断证据，而不是因为随后生成 `best.pt` 就进入正式比较。相同原则也用于 OOM 自动降批次：恢复能力属于工程价值，恢复后的运行却不能冒充从一开始就遵循正式 batch 协议的干净样本。

## 4. 核心结果

### 多随机种子结论

| 数据集 | 最佳绝对性能方法 | n | P | R | mAP50 | mAP50-95 | 单次最高 mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| Brain Tumor | Last Stage + Neck + Head | 3 | 0.44358 ± 0.02798 | 0.78865 ± 0.02487 | 0.54427 ± 0.02979 | **0.39172 ± 0.02316** | **0.40944** |
| VisDrone | Full Fine-tuning（actual batch=4） | 3 | 0.38211 ± 0.00387 | 0.29718 ± 0.00420 | 0.27550 ± 0.00294 | **0.15342 ± 0.00221** | **0.15527** |

标准差为样本标准差，种子均为有效的 0、1、2。Brain Tumor 的最佳方法不是 Full FT，而是 Last Stage + Neck + Head，说明小型医疗数据上适度限制可训练范围可能优于全量更新。VisDrone 则由 Full FT 取得最高绝对精度。

### seed=0 的精度与资源代表值

| 数据集 | 方法 | mAP50-95 | 可训练参数 | 峰值显存/GiB | 训练时间/s | 推理/ms·图 |
|---|---|---:|---:|---:|---:|---:|
| Brain Tumor | Head-only | 0.36997 | 347,718 | 6.21 | 306.5 | 2.0 |
| Brain Tumor | Neck + Head | 0.34770 | 901,574 | 6.47 | 278.3 | 1.8 |
| Brain Tumor | Last Stage + Neck + Head | **0.40021** | 2,114,633 | 6.51 | 385.6 | 2.4 |
| Brain Tumor | Full Fine-tuning | 0.37121 | 2,662,546 | 9.44 | 537.6 | 2.3 |
| Brain Tumor | Stable LoRA | 0.06395 | 409,174 | 7.46 | 325.1 | 3.0 |
| Brain Tumor | Partial fine-tuning + LoRA | 0.35319 | 965,574 | 7.47 | 486.7 | 2.5 |
| VisDrone | Neck + Head | 0.14359 | 903,134 | 22.50 | 1705.1 | 6.4 |
| VisDrone | Last Stage + Neck + Head | 0.14988 | 2,116,193 | 22.50 | 1746.0 | 5.4 |
| VisDrone | Full Fine-tuning | **0.15527** | 2,664,106 | 23.00 | 1912.5 | 5.6 |

推理速度来自验证日志，适合本项目内部比较，但不是独立、严格控制硬件预热和重复次数的部署基准。Brain Tumor 的 Head-only 只训练 347,718 个参数，精度接近 Full FT 且显存更低，是参数效率代表；Partial LoRA 没有超过普通部分微调。Stable LoRA 的数值稳定性得到改善，但精度明显落后。VisDrone 的正式稳定性门控下，LoRA 运行没有形成可与三种微调基线并列的有效结果；旧的稳定 LoRA 证据也不足以支持绝对精度优势。

## 5. 结果审计与真实 Pareto

Phase 2 收尾审计修复了五类会改变结论的问题：非 LoRA 方法的可训练参数曾被错误读取为验证阶段的 “0 gradients”；Stable LoRA 被误分为 Diagnostic；自动降 batch 的 b8 命名运行与干净 b4 运行可能重复计入；未启动的 Head-only 队列项可能被误写成多次 OOM；同一 seed 的恢复、重复引用或诊断运行可能被当作独立种子。修复后，无法由原始证据确定的字段使用 `unknown/evidence_missing`，而不是用 0 填充。

参数统计错误说明了为什么“从日志抓一个数字”并不可靠。模型在训练初始化时具有可训练参数，但验证阶段通常关闭梯度；若汇总器读取后者，Head-only 和 Full FT 都可能被错误写成 0。修复后的提取逻辑优先使用初始化证据、冻结配置和静态模型构建，并对正式非 LoRA 方法的零值设置断言。Partial LoRA 则必须同时计入 LoRA 参数和被解冻的普通参数，不能只计算 Adapter。

多随机种子聚合以数据集、规范化方法名、真实配置签名和 seed 为键。每个 seed 只能贡献一个符合正式协议的运行；诊断、恢复、失败和未执行条目不会因为目录中存在旧 `results.csv` 就获得正式资格。这里使用样本标准差而非总体标准差，并同时报告 `n`，以提醒读者三个种子的估计仍有较大不确定性。对于只有 seed=0 的方法，本报告只呈现代表运行，不把单次差异包装成稳定排序。

审计 CSV 的多目标 Pareto 分析先以退出码、结果文件、权重、日志无 NaN/Inf/recovery、指标非异常归零等条件做稳定性门控，再同时最大化 mAP50-95、最小化可训练参数、峰值显存和训练时间。它不是简单选“最高精度/最高比值”。Brain Tumor 的四目标前沿包含不同资源取向的 Last Stage、Head-only、Neck + Head 等运行；VisDrone 的四目标前沿包含 Full FT、Last Stage 和 Neck + Head。Stable LoRA 某个低成本点可能因训练时间形成四目标非支配关系，但低精度意味着它不自动成为实用推荐。展示图中的填充标记则独立按 mAP50-95 与可训练参数重新计算二维前沿，不直接复用四目标 designation。

## 6. 贡献边界、局限与可用价值

本项目最可靠的贡献是：建立了从非有限 Adapter 梯度定位、AMP/学习率单变量诊断、健康检查点审计，到强基线、多随机种子、OOM 恢复去重和真正 Pareto 分析的完整证据链。它展示了参数高效训练研究不能只报告“可训练参数少”，还必须同时核对精度、显存、时间、稳定性和失败样本。

相对于上游已有场景配置，本项目的额外贡献集中在强基线和多随机种子比较、Detection Head/Adapter/Router 分组学习率、`requested_batch` 与 `actual_batch` 分离、OOM/重复目录/同 seed 去重审计、formal validity registry，以及 Pareto 与语义完成门控。这些工作提升的是证据可信度与实验治理，不应与上游通用 LoRA 修复混为一谈。

从评审或简历展示角度，最合适的表述不是“提出一种全面超越全量微调的新 LoRA”，而是“完成了稳定性增强的参数高效目标检测实验体系，并通过审计发现 LoRA 在当前任务中不具备绝对精度优势”。这体现的是故障定位、实验设计、自动化容错、统计汇总和研究诚信。比赛或论文材料应把最佳绝对性能、最佳参数效率和数值稳定方案分列，展示完整失败证据，同时避免把数学上的非支配点直接称为最佳模型。

对导师或评审而言，复现价值还来自可追溯性：每个结论都能回到独立运行、配置、日志和审计表；失败实验未被删除，自动恢复也没有被隐藏。这样的材料允许第三方区分模型能力、数值稳定性和工程容错，而不是只能接受一张无法核验的最高分表格。

边界同样明确：Stable 只表示数值稳定，不表示精度最高；LoRA 在当前两个数据集上均无绝对精度优势；VisDrone LoRA 只使用 20% 训练数据；AMP-safe LoRA 尚未验证；部分方法只有 seed=0；Adapter 0.1×与 Router 0.5×只是当前任务的经验配置。若继续研究，最低成本且最有信息量的工作是先修复 AMP-safe 接口并做单变量复验，再为关键部分微调基线补种子，最后在完整 VisDrone 数据上重新比较稳定 LoRA 与强基线。
