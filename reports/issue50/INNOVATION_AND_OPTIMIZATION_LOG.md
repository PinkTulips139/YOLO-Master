# Issue #50 创新与优化记录

## 证据说明

本表以 Phase 2 最终材料为最高优先级。状态含义：**已验证**表示有日志或控制实验支持；**初步证据**表示方向有效但重复或控制不足；**未验证设想**不得写入成果结论；**被否定/不推荐**表示当前实现或数据不支持。

## 最终封版补充：Phase 1 与 Phase 2 完整总结

| 方向 | 状态 | 当前证据与创新价值 | 成本、风险与局限 | 下一步最小验证 |
|---|---|---|---|---|
| AMP-safe LoRA | 被否定/不推荐（当前实现） | Phase 1 日志证明 AMP 首先破坏 Adapter 梯度；但 Phase 2 的 AMP-safe 原型在两个数据集均失败、指标全零，不能声称成功 | 混合精度边界和 dtype 回传复杂；错误实现会制造“看似混合、实际不安全” | 不训练，先用单层 LoRA 做 forward/backward dtype、有限梯度与 GradScaler 单测 |
| Adapter LR 抑制 | 已验证 | Brain Tumor 单变量 1.0→0.1 使 mAP50-95 由 0.02245 提升至 0.06630；0.2 较弱。VisDrone 0.1 也优于 0.5。价值在抑制特征漂移 | 过低会欠适配；结论仅覆盖当前模型和数据 | 固定协议，在第三数据集仅比较 1.0 与 0.1 |
| Head/Router/Adapter 分组 LR | 初步证据 | Adapter=0.1×有效；Router 0.25×与0.5×逐轮一致。当前 Head 实际分散在普通 weight/norm/bias 组，尚无独立 Head LR 消融 | 参数漏分组或重复分组风险高 | 写参数 ID 去重测试，再只比较 Head 1.0×与一个预注册邻近值 |
| 非有限梯度检测 | 已验证 | 日志定位 epoch 1 step 0 首个异常为 `lora_A`，早于 loss 非有限。把“训练崩了”转化为可定位证据 | 监测会增加少量开销；只定位首发点，不自动证明根因 | 注入单个 Inf，验证检测位置、停止策略和日志完整性 |
| `last_healthy.pt` 与恢复审计 | 初步证据 | 源码证明启动前快照、健康 epoch 原子刷新、最多3次恢复，并恢复 optimizer/scaler/EMA | LoRA 在线模型只显式加载 Adapter，Head 与全状态一致性未被 hash 验证 | 人工注入异常，比较恢复前后 model/head/optimizer/EMA 哈希 |
| LoRA rank 效率权衡 | 已验证 | 两数据集 r4/r8/r16 均稳定；r4 精度最高且参数最少，高 rank 未带来收益 | 只覆盖两个场景；不能推断所有密集检测任务都偏好 r4 | 第三数据集固定预算复验 r4/r8/r16，不再调 LR |
| Partial FT + LoRA | 初步证据 | Brain Tumor 0.35319、VisDrone 0.08565，均明显高于 Stable LoRA，但低于最强非 LoRA 方案 | 仅 seed=0；训练参数计数在最终表中不可靠，且耗时不占优 | 先修正参数计数；若仍在 Pareto 前沿，再补两个 seed |
| 多随机种子稳定性 | 已验证（关键方案） | Last Stage+Neck+Head：Brain 0.39172±0.02316；Full FT：VisDrone 0.15344±0.00193 | 其他候选多为单 seed，不能比较方差 | 只给可能进入最终结论的 Partial FT+LoRA 补 seed，不扩大全矩阵 |
| 精度—参数—显存—时间 Pareto | 初步证据 | 精度、显存、时间均已汇总，支持多目标评价；项目不再只追求单次 mAP | Phase 2 非 LoRA 行 `trainable_params=0` 是解析缺口，因此“参数效率最优”标记暂不可信；推理速度也非独立基准 | 从 checkpoint/模型结构重新只读统计参数，再重算 Pareto |
| Adapter-only FP32 / 独立 GradScaler | 未验证设想 | 理论上可保留主体 AMP 速度并保护 Adapter；当前没有成功实验 | 需定制 autograd、独立缩放或明确 dtype 边界，工程风险高 | 先做最小模块级梯度一致性测试，不直接跑正式训练 |
| 动态或分层 rank | 未验证设想 | 有望在固定参数预算下按层分配容量 | 搜索空间大，容易与层选择、数据规模混杂 | 用 r4 总参数预算，预注册一种敏感度分配，与固定 r4 单变量比较 |
| 跨数据集与完整 VisDrone 训练 | 未验证设想 | 两类差异场景已有初步跨域证据，但 VisDrone 正式协议使用 `fraction=0.2` | 全量训练成本高；不能与当前 fraction 结果直接混用 | 固定一组最强基线和一组高效方案，在完整训练集各跑一个 seed 做可行性验证 |

## 研究价值与边界

当前 LoRA 在 Brain Tumor 的精度明显低于 Head-only、Full FT 和 Last Stage+Neck+Head。Phase 2 最强值来自 Last Stage+Neck+Head，而不是 LoRA；VisDrone 最强值来自 Full FT。项目价值主要是：

1. 建立“梯度先异常、loss 后异常”的稳定性诊断链；
2. 证明 Adapter LR 抑制能缓解有限数值下的特征漂移；
3. 把失败实验、恢复机制、参数效率和资源指标纳入同一审计协议。

因此可主张“稳定性诊断与可信实验基础设施”，不能主张“LoRA 带来普遍绝对精度提升”，也不能把尚未实现成功的 Adapter-only FP32 包装为创新成果。
