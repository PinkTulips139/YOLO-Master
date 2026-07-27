# Issue #50 最终项目报告

## 1. 项目目标与证据范围

本项目复现并审计 YOLO-Master [Issue #50](https://github.com/Tencent/YOLO-Master/issues/50) 的 LoRA 目标检测链路，在 Brain Tumor 与 VisDrone 两个场景比较 rank=4/8/16，并研究混合精度、参数组学习率和自动恢复对训练可信度的影响。所有正式结果均要求退出码为 0、`results.csv`、`args.yaml`、`best.pt`、`last.pt`、完整日志和 manifest 齐全，且日志与指标中不存在 NaN、Inf 或 recovery。

模型为 `YOLO-Master-EsMoE-N.pt`。Brain Tumor 包含 2 类、893 张训练图和 223 张验证图；VisDrone 包含 10 类、6471/548/1610 张训练/验证/测试图，正式训练使用训练集 `fraction=0.2`、完整验证集。预训练权重成功迁移 760/811 项；其余 51 项来自类别数变化导致的检测头不匹配，因此检测头重新初始化。

## 2. 原始问题与根因

日志直接证明：原始 AMP 实验在 epoch 1、step 0 首先出现非有限梯度，首个参数为 `model.base_model.model.4.conv.lora_A.default.weight`，属于 Adapter，而当步 box/cls/dfl loss 仍为有限值。随后恢复器触发 `NaN recovery`，实验后期指标归零。

源码直接证明：参数组为普通 weight、norm/no-decay、bias、router 和两组 adapter；检测头并未独立成单一参数组。`last_healthy.pt` 在训练开始前创建，有限 epoch 后原子刷新；loss、fitness、gradient 或 EMA 非有限均可触发恢复，连续恢复上限为 3 次。恢复时会关闭 AMP，并恢复优化器、scaler 与 EMA。当前 LoRA 在线模型恢复只显式加载 adapter 张量，完整恢复一致性仍是风险。

根因排序如下：

1. **AMP 下 Adapter 梯度溢出（已验证）**：同一链路关闭 AMP 后不再出现非有限梯度。
2. **随机初始化检测头与 Adapter 更新速度失衡（已验证）**：Adapter 倍率 1.0 虽无 NaN，但性能在 epoch 4 后漂移；降至 0.1 后 mAP50-95 从 0.02245 提升至 0.06630。
3. **`optimizer=auto` 覆盖名义 `lr0`（源码与日志验证）**：原始正式链路实际选择 AdamW、基础学习率约 0.001667。
4. **低学习率诊断存在 warmup 混杂（已验证）**：`lr0=0.0003` 组首轮 bias warmup 仍达 0.0804161，且新检测头学习不足，不能简单归因为“低 LR 更差”。

## 3. 稳定方案

Brain Tumor 固定 `AdamW, lr0=8e-4, amp=False, lora_lr_mult=0.1, router_scale=0.5, batch=16, imgsz=640, epochs=40`；VisDrone 固定 `lr0=1e-3, batch=8, imgsz=768, fraction=0.2, epochs=30`，其余稳定性设置相同。rank 对比只改变 `lora_r`，并保持 `alpha=2r`。

选择完全关闭 AMP 是当前经过实验验证的安全方案。“仅 Adapter 保持 FP32、其余 AMP”没有作为正式结果：现有 PEFT Linear/Conv forward 会把 Adapter 输出转换回主分支 dtype，并共享同一个 GradScaler/backward；若不引入定制 autograd 或独立缩放，无法保证隔离，临时修改会破坏消融可信度。

## 4. Rank 正式结果

以下均为 seed=0、按 mAP50-95 选择的最佳 epoch：

| 数据集 | Rank | P | R | mAP50 | mAP50-95 | 可训练参数 | 峰值显存 | 时间(s) | 稳定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Brain Tumor | 4 | 0.39918 | 0.06107 | 0.11416 | **0.06395** | 409,174 | 7.46 GiB | 325.1 | 是 |
| Brain Tumor | 8 | 0.43074 | 0.06191 | 0.11313 | 0.06180 | 473,174 | 7.58 GiB | 271.5 | 是 |
| Brain Tumor | 16 | 0.01114 | 0.26982 | 0.09309 | 0.04821 | 601,174 | 7.58 GiB | 454.1 | 是 |
| VisDrone | 4 | 0.09139 | 0.09525 | **0.05444** | **0.02878** | 410,734 | 23.00 GiB | 1502.2 | 是 |
| VisDrone | 8 | 0.09106 | 0.09204 | 0.05286 | 0.02811 | 474,734 | 23.00 GiB | 1563.7 | 是 |
| VisDrone | 16 | 0.09090 | 0.09340 | 0.05361 | 0.02862 | 602,734 | 23.00 GiB | 1755.5 | 是 |

两个场景的单次最佳 rank 均为 4。更高 rank 增加了参数和时间，但没有带来精度收益；这是本项目中最清晰的 rank 效率结论。

日志记录的单次验证推理耗时为：Brain Tumor r4/r8/r16 分别 3.0/6.3/2.5 ms/图，VisDrone 分别 13.7/5.0/4.1 ms/图；Brain Tumor head-only/full fine-tune 分别 2.0/2.3 ms/图。这些值保留在最终 CSV，但由于输入尺寸、验证阶段 warmup 与运行抖动不同，不将其解释为 rank 带来的确定速度提升。

## 5. 多随机种子复现

最佳稳定配置 r=4 在每个数据集运行 seed=0/1。均值、样本标准差和单次最佳值如下：

| 数据集 | P（均值±SD） | R（均值±SD） | mAP50（均值±SD） | mAP50-95（均值±SD） | 单次最佳 mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Brain Tumor | 0.39094±0.01165 | 0.05304±0.01136 | 0.10096±0.01868 | 0.05741±0.00926 | 0.06395 |
| VisDrone | 0.14605±0.07730 | 0.09686±0.00228 | 0.05609±0.00233 | 0.02920±0.00059 | 0.02962 |

Brain Tumor 对随机种子较敏感，因此单次最佳值不能代表期望性能；VisDrone 的 mAP 方差较小，但 Precision 方差较大。

## 6. 核心消融

| 方案 | 唯一目标变量 | mAP50 | mAP50-95 | 数值状态 | 结论 |
|---|---|---:|---:|---|---|
| AMP on | 原始 AMP | 0.04418 | 0.02424 | Adapter 非有限梯度、recovery | 不可进入正式结果 |
| AMP off | 关闭 AMP（3 epoch） | 0.05718 | 0.03007 | 有限 | 验证稳定性，不用于长期精度比较 |
| Adapter LR ×1.0 | Adapter 倍率 1.0（10 epoch） | 0.04410 | 0.02245 | 有限但性能漂移 | 更新过快 |
| Adapter LR ×0.1 | 仅倍率改为 0.1（10 epoch） | 0.12106 | 0.06630 | 有限 | 明显抑制漂移 |
| 完整 Stable LoRA | 冻结后的正式方案 | 0.11416 | 0.06395 | 有限、无 recovery | 可用于正式材料 |

Router scale 0.25 与 0.5 的逐轮结果完全一致，说明该配置下 Router 并非主要矛盾，后续没有继续无意义搜索。

## 7. 公平基线与边界

Brain Tumor 的 head-only 与 full fine-tune 使用与 Stable LoRA 相同的数据、seed、预训练权重、epoch、batch、imgsz、优化器、基础学习率、warmup 和 AMP 策略。

| 方法 | P | R | mAP50 | mAP50-95 | 可训练参数（占比） | 峰值显存 | 时间(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Head-only | 0.45906 | 0.86525 | 0.51525 | 0.36997 | 347,718（13.06%） | 6.21 GiB | 306.5 |
| Stable LoRA r=4 | 0.39918 | 0.06107 | 0.11416 | 0.06395 | 409,174（15.01%） | 7.46 GiB | 325.1 |
| Full fine-tune | 0.44260 | 0.83539 | 0.51586 | 0.37121 | 2,662,546（100.00%） | 9.44 GiB | 537.6 |

结果不支持“当前 Stable LoRA 精度优于 head-only”。在这个小型、类别数变化的场景，重新初始化的检测头是主要学习对象，head-only 同时取得最高参数效率与接近全量微调的精度。

VisDrone head-only 在严格匹配 `batch=8, imgsz=768` 时于第 7 个 epoch 前发生 CUDA OOM（已完成 6 个 epoch，退出码 1），因此不进入正式比较；其部分指标只保留为失败证据。全量微调预计占用更高，按停止规则未启动。降低 batch 可以运行，但会破坏本轮公平协议，故没有临时拼凑结果。

## 8. Pareto 与创新证据

精度、可训练参数、显存和时间的联合 Pareto 分析表明：

- Brain Tumor：head-only 是核心 Pareto 点；full fine-tune仅提供约 0.00124 mAP50-95 增益，却显著增加参数、显存和时间。Stable LoRA r=8 因训练时间较短形成另一非支配点，但精度明显较低。
- VisDrone：在可完整比较的 LoRA rank 中，r=4 同时具有最高 mAP50-95、最少参数和最短时间，支配 r=8/16。

创新证据分级：

- **已被实验验证**：AMP 下 Adapter 首先发生非有限梯度；关闭 AMP 可消除该异常；Adapter 学习率倍率 0.1 可抑制性能漂移；健康检查点和非有限检测能够留下可审计恢复证据；r=4 在两个场景提供最佳 rank 权衡。
- **有初步证据**：随机初始化检测头与 Adapter 的更新竞争是小数据集 LoRA 表现不佳的重要机制；分组学习率对 Adapter 有效，但检测头目前仍分散在常规 weight/norm/bias 组中，尚未完成独立倍率消融。
- **后续研究设想**：真正的 Adapter-only FP32/独立 GradScaler、检测头/Router/Adapter 三路学习率、恢复后全模型一致性校验、跨更多数据集和至少 3–5 个种子的统计验证。

## 9. 局限性

只有最佳 LoRA 配置完成两个种子，不能进行强统计推断；VisDrone 非 LoRA 基线受 24 GiB 显存限制；推理速度来自训练日志的单次验证，受缓存与 warmup 影响，不是独立基准；Brain Tumor LoRA 明显弱于 head-only，因此创新价值应表述为“稳定训练与失败诊断方法”，而不是“普遍提升精度”。

## 10. 最终结论

本项目形成了可复现的稳定 LoRA 训练协议和完整失败证据链。最可信的结论是：AMP 会首先破坏 Adapter 梯度；关闭 AMP 并将 Adapter 学习率降至基础学习率的 0.1 可恢复数值稳定性并显著改善长期性能；rank=4 在两个数据集上是最有效率的 LoRA 选择。但 Brain Tumor 公平基线显示 head-only 更适合当前小数据、检测头重初始化场景。比赛或论文材料应以“稳定性增强的参数高效目标检测训练与可信恢复审计”为主线，并如实呈现精度边界。
