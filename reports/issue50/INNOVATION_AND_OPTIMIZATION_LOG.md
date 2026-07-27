# Issue #50 创新与优化记录

| 想法 | 状态 | 创新性 | 成本 | 风险 | 实验设计 |
|---|---|---|---|---|---|
| detection head/router/adapter 独立 LR | 正在验证 | 中高：解决小数据集 head 重初始化与 PEFT 耦合 | 中 | 分组错误造成漏训/重复参数 | 先固定 head/base，单变量 sweep adapter；再新增 head group |
| adapter LR 稳定性约束 | 正在验证 | 中 | 低 | 过低导致适配不足 | `lora_lr_mult=1.0→0.1→0.2`，固定其余条件 |
| AMP 首步安全探针 | 已实现诊断 | 中高 | 低 | 日志开销 | 记录首个非有限参数、group、step 和 loss |
| AMP 自动隔离 adapter | 值得验证 | 高 | 中 | 混合精度边界复杂 | adapter 梯度/权重保持 FP32，其余 AMP，对照全 FP32 |
| 梯度范数分组监测与裁剪 | 值得验证 | 中 | 中 | 监测影响速度 | 按 head/router/adapter 记录 norm，分别设置 clip |
| recovery 完整一致性 | 值得验证 | 高 | 中 | 修改恢复代码风险高 | 人工注入异常，比较 model/head/optimizer/EMA hash |
| 动态/分层 rank | 值得验证 | 高 | 高 | 公平比较和实现复杂 | 稳定固定-rank基线后按层敏感度分配相同参数预算 |
| rank 与数据规模自适应 | 值得验证 | 高 | 高 | 容易过拟合单数据集 | Brain Tumor/VisDrone 多规模曲线与统一预算 |
| LoRA 插入层选择消融 | 值得验证 | 中高 | 高 | 组合空间大 | Backbone、Neck、MoE expert 分区单变量消融 |
| Backbone/Neck/Head 注入策略 | 值得验证 | 中高 | 高 | head LoRA 与重初始化混杂 | 三段式注入矩阵，固定总 adapter 参数 |
| 参数/显存/速度/精度 Pareto | 正在建设 | 中 | 中 | 测量口径不一致 | 同硬件记录 mAP、参数、峰值显存、epoch time |
| 多随机种子 | 值得验证 | 必需的可信度提升 | 高 | 计算成本 | 正式候选先 seed 0，再对关键 rank 补 2 个 seed |
| 跨数据集泛化 | 计划中 | 高 | 高 | 数据协议差异 | Brain Tumor 稳定后独立校准 VisDrone，再比较趋势 |
| 小数据集 head 重初始化专用 warmup | 值得验证 | 中高 | 中 | warmup 过长欠拟合 | head 独立 LR/warmup，对比共享 pg0-2 |
| 直接保留 adapter multiplier=1.0 | 不推荐 | 低 | 低 | 已观察到性能崩塌 | 除非新机制能证明稳定，否则不进入正式结果 |

## 当前最重要的研究问题

日志事实表明，基础模型和新 detection head 与 adapter 使用相近 LR 时，即使所有数值有限，
验证性能也会崩塌；把 adapter LR 降至 head/base 的 0.1 倍后，性能持续改善。当前要验证的不是
“LoRA 是否有效”，而是“小数据集、检测头重初始化场景下，head-adapter 更新速度如何匹配”。

