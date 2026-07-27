# YOLO-Master Issue #50 初学者学习指南

## 如何使用本指南

每节都按“**概念 → 本项目实例 → 应该怎么看**”组织。数字以 Phase 2 最终文件为准；若汇总字段存在缺口，会明确说明“现有证据不足”。

## 1. 文件与目录

**概念：** 源码、配置、实验产物和研究报告承担不同职责。
**本项目实例：**

- `ultralytics/`：模型、数据、训练器、Loss、验证和恢复机制源码。
- `examples/lora_examples/*.yaml`：Brain Tumor、VisDrone 的场景配置。
- `scripts/issue50/`：rank sweep、诊断、Phase 2 队列与收尾器。
- `runs/issue50/formal/`、`phase2/`、`diagnostics/`：结果、日志、配置和权重；不提交 Git。
- `reports/issue50/`：协议、CSV、结论、图表和本学习记录。

**应该怎么看：** 想确认“实际跑了什么”先看输出目录的 `args.yaml`；想看指标看 `results.csv`；想查错误看独立日志；想确认版本、命令和退出码看 `run_manifest.json`；想了解结论看 `PHASE2_FINAL_REPORT_CN.md`。

## 2. 图片如何进入模型

**概念：** 数据加载器读取图片和 YOLO 标签，完成缩放及增强，再组成 batch。模型前向传播依次提取、融合并输出预测。
**本项目实例：** Backbone 提取纹理和语义；Neck 融合多尺度特征；MoE 的 Router 选择 Expert；Detection Head 输出类别与框。Brain Tumor 类别数与预训练模型不同，因此 811 项权重中只迁移 760 项，检测头 51 项不匹配并重新初始化。
**应该怎么看：** 权重迁移不是越接近 100% 越好。类别数变化时 Head 不匹配是预期行为；关键是确认它已正确初始化并参与训练。

## 3. 前向传播、Loss 与参数更新

**概念：** 前向传播得到预测；box loss 衡量定位，cls loss 衡量类别，DFL 约束边界框分布，mixture auxiliary loss 约束专家路由。反向传播计算梯度，optimizer 按学习率更新可训练参数，EMA 保存更平滑的模型副本。
**本项目实例：** AMP 异常实验中 loss 仍有限，但 `lora_A` 梯度已经非有限，说明只看 loss 会漏掉首发故障。
**应该怎么看：** Loss 下降只说明训练目标在变化，不等于验证性能更好；还要联合看梯度有限性、验证 mAP、EMA 和多轮趋势。

## 4. 六个核心模块

| 概念 | 本项目中的作用 | 阅读重点 |
|---|---|---|
| Backbone | 从低级纹理到高级语义提取特征 | 冻结多少、最后 stage 是否解冻 |
| Neck | 融合不同分辨率特征 | 小目标 VisDrone 尤其依赖多尺度融合 |
| Detection Head | 把特征变成类别和框 | 本项目重新初始化，需要充分学习 |
| MoE Expert | 多个子网络提供额外容量 | 是否出现专家或路由失衡 |
| Router | 选择 Expert | `router_scale` 控制其更新速度 |
| LoRA Adapter | 用低秩 A/B 矩阵学习权重增量 | rank、alpha、参数量和梯度稳定性 |

## 5. 冻结与四种微调

**概念：**

- Head-only：只训练检测头；
- Partial FT：解冻 Neck 或最后 Backbone stage 加 Neck/Head；
- Full FT：几乎全部基础参数都更新；
- LoRA：冻结大部分基础权重，只训练 Adapter，并保留必要的 Head；
- Partial FT+LoRA：部分基础层和 Adapter 同时训练。

**本项目实例：** Brain Tumor Head-only 已达到 0.36997 mAP50-95，Last Stage+Neck+Head 的三种子均值为 0.39172；Stable LoRA 只有约 0.05741 的两种子均值。Partial FT+LoRA 提升到 0.35319，但仍低于最强部分微调。
**应该怎么看：** “参数更少”与“精度更高”是两件事。小数据集且 Head 重初始化时，直接训练 Head/后段网络可能比 LoRA 更合适。

## 6. 关键超参数

| 参数 | 含义 | 本项目结论 |
|---|---|---|
| `lr0` | 基础学习率 | Brain 稳定 LoRA 用 `8e-4`，VisDrone 用 `1e-3` |
| `lora_lr_mult` | Adapter 相对基础 LR | 0.1 明显优于 1.0；是已验证结论 |
| `router_scale` | Router 相对 LR | 0.25 与 0.5 逐轮一致，未证明更低更好 |
| warmup | 初期逐渐调整 LR | bias warmup 曾造成混杂，正式诊断显式控制 |
| AMP | 混合精度加速 | 原始 AMP 导致 Adapter 梯度溢出；FP32 稳定 |
| batch | 一次前向的图片数 | VisDrone OOM 时降物理 batch，并用累积保持有效 batch |
| imgsz | 输入分辨率 | OOM 时未通过降低 imgsz 掩盖问题 |
| fraction | 使用训练集比例 | VisDrone 正式结果是 0.2，不能冒充完整训练集结果 |
| rank / alpha | LoRA 容量和缩放 | 比较 r4/8/16，保持 `alpha=2r`；r4 最有效 |

**应该怎么看：** 最终值以 `args.yaml` 为准。命令、场景 YAML 与默认值冲突时，不能凭记忆判断。

## 7. 指标与随机种子

**概念：** Precision 高表示误报少；Recall 高表示漏检少；mAP50 使用 IoU=0.5；mAP50-95 在多个更严格阈值上平均，是主要指标。最佳 epoch 是验证指标最好的轮次，不一定是最后一轮。均值表示典型水平，标准差表示种子波动。
**本项目实例：** Brain 最强部分微调为 0.39172±0.02316；VisDrone Full FT 为 0.15344±0.00193。
**应该怎么看：** 不能挑最高 seed 当平均结果。只有一个 seed 的方案只能说“初步结果”，不能与三种子方案进行同等强度的稳定性比较。

## 8. NaN、Inf、recovery 与 OOM

- **NaN/Inf：** 数值已不可用；本项目首发点是 Adapter 梯度。
- **梯度溢出：** 低精度下梯度超出表示范围，可能早于 loss 异常。
- **recovery：** 从 `last_healthy.pt` 恢复并继续；它能救工程任务，但触发过 recovery 的实验不应混入干净正式比较。
- **OOM：** GPU 显存不足，不是模型精度结论。Phase 2 将 OOM 尝试保留为失败证据，并按预设 batch 降级。

**应该怎么看：** 成功必须同时满足退出码 0、产物齐全、指标可解析且没有非有限值/recovery。失败结果不删除，但不得拿部分 mAP 与完整实验比较。

## 9. 五类证据文件

1. `results.csv`：每轮 Loss、P、R、mAP 和耗时；按列名读取，不靠肉眼猜位置。
2. `args.yaml`：实际配置的最高优先级证据。
3. `run_manifest.json`：命令、commit、环境、时间、退出码和成功状态。
4. 训练日志：权重迁移、冻结层、optimizer 参数组、OOM、NaN 和 recovery。
5. 最终报告/CSV：跨实验汇总，但仍需检查解析口径。

Phase 2 非 LoRA 方法的 `trainable_params` 出现 0，显然不能解释为“没有训练参数”；这是汇总解析缺口。因此绝对精度可用，但基于这些 0 值产生的参数效率 Pareto 标记证据不足。

## 10. 公平、稳定与可复现

**公平：** 同一问题只改变目标变量；固定数据划分、权重、seed、imgsz、epoch、增强和验证集。OOM 降 batch 时要记录累积策略。
**稳定：** 不只“跑完”，还要无 NaN/Inf/recovery、指标不异常归零、权重和 CSV 完整。
**可复现：** 保存命令、args、日志、manifest、Git commit、环境和独立输出目录。
**应该怎么看：** 烟雾测试只证明链路能运行；诊断实验只回答一个机制问题；只有通过门控的正式实验才能进入结果表。

## 11. Phase 1 与 Phase 2 的实际结论

### Phase 1

日志直接证明 AMP 下 Adapter 首先溢出；关闭 AMP 能规避异常。控制实验直接证明 Adapter LR=0.1× 能抑制性能漂移。r4 在两个场景的 LoRA rank 比较中最好，但 LoRA 绝对精度明显弱于 Brain Tumor Head-only/Full FT。

### 最终封版补充：Phase 1 与 Phase 2 完整总结

Phase 2 共 26 个任务，15 成功、11 失败。Brain Tumor 最强是 Last Stage+Neck+Head；VisDrone 最强是 Full FT。Partial FT+LoRA 有明显改善，但尚未达到最佳非 LoRA 精度，也缺少多种子复验。AMP-safe LoRA 原型失败，不能当作已验证创新。项目最扎实的贡献是稳定性诊断、Adapter LR 证据、自动恢复审计、失败保留和公平比较基础设施。

## 12. 如何形成比赛或论文材料

比赛材料应分别给出：最佳绝对性能权重、资源受限方案、显存/耗时、失败边界和复现配置。论文可以围绕“参数高效检测的稳定性审计”组织：问题定义、首发梯度证据、Adapter LR 消融、rank/强基线、多种子、OOM 与恢复门控。

不能夸大的表述包括：

- 不能说 LoRA 在本项目中优于 Head-only 或 Full FT；
- 不能说 AMP-safe LoRA 已验证成功；
- 不能把 VisDrone `fraction=0.2` 写成完整训练集实验；
- 不能用解析为 0 的可训练参数宣称 Pareto 最优；
- 不能把单 seed 最高值写成稳定平均性能。

最终材料的可信度来自“结论与证据级别一致”，而不是只展示最高 mAP。
