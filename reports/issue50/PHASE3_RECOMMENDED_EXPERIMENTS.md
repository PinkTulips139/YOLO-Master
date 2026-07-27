# Issue #50 Phase 3 建议实验

本文件只记录 Phase 2 审计后仍需新训练才能回答的问题；当前审计未启动训练。

| 优先级 | 最小实验 | 原因 | 预计成本 | 是否影响 Phase 2 |
|---|---|---|---|---|
| P0 | 修复后的 AMP-safe LoRA：两个数据集各用最佳稳定 rank、seed=0，与 AMP=False 配对 | 现有两组在训练前因 `Conv2d.amp_safe_forward` 缺失而实现失败，无法验证混合精度安全方案 | 2 个正式训练 | 否；Phase 2 结论为“未验证/实现失败” |
| P1 | Brain Tumor Head-only seed=1、2 | 当前 Head-only 只有 seed=0，无法估计方差或与三种子 Last-stage 方案做同等强度比较 | 2 个训练 | 不改变现有单次结果，可能影响多种子排序 |
| P1 | VisDrone Neck+Head seed=1、2，固定干净 batch=4 | 当前仅 seed=0；它是参数效率候选，但缺少方差证据 | 2 个训练 | 不改变 Phase 2 单次 Pareto，可能影响稳健 Pareto 结论 |

停止条件：每项严格复用对应数据集正式协议；出现 NaN、Inf、recovery 或 OOM 即保留证据并停止同类扩展，不做额外网格搜索。
