# YOLO-Master Issue #50 初学者学习指南

## 你手上有哪些文件

- `ultralytics/`：模型、训练器、数据加载、Loss 和验证的核心源码。
- `examples/lora_examples/*.yaml`：某个实验场景的基础配置。
- `scripts/issue50/`：批量启动和汇总 Issue #50 实验。
- `runs/issue50/`：日志、曲线、`results.csv` 和权重；不提交 Git。
- `reports/issue50/`：实验协议、台账、错误、决策与学习资料。
- `weights/*.pt`：预训练或训练后权重；不提交 Git。
- 数据集目录：图片进入 `images/train|val`，标签位于 `labels/train|val`。

## 一张图片怎样完成一次训练

1. 数据加载器读取图片和 YOLO 标签，并进行缩放、翻转、mosaic 等增强。
2. 前向传播把图片依次送入 Backbone、Neck、MoE/router 和 detection head。
3. detection head 预测类别、边界框分布和位置。
4. Loss 比较预测与标签：box loss 管位置，cls loss 管类别，DFL 管边界框离散分布，
   mixture auxiliary loss 约束专家路由。
5. 反向传播计算每个可训练参数的梯度。
6. optimizer 按参数组 LR 更新 detection head、router 和 adapter。
7. EMA 保存更平滑的模型副本，验证集计算 Precision、Recall 和 mAP。

## 关键模块

- **Backbone**：提取从边缘、纹理到高级语义的特征。
- **Neck**：融合不同尺度的特征，小目标和大目标都需要它。
- **Detection head**：把特征变成类别和框。本数据集类别变化，因此它被重新初始化。
- **MoE**：多个 expert 处理输入，增加模型容量。
- **Router**：决定哪些 expert 处理当前特征；学习率过高可能导致专家选择不稳定。
- **LoRA/adapter**：冻结大部分基础权重，只训练低秩增量矩阵，减少训练参数。

## 本项目的参数组

- `pg0`：普通 weight，含 detection head 的卷积权重。
- `pg1`：BN/无衰减 weight。
- `pg2`：bias，含 detection head bias。
- `pg3`：router，默认是基础 LR 的 0.5 倍。
- `pg4/pg5`：adapter；后一个组应用 layer-wise LR decay。

Detection head 目前没有独立组，所以“基础 LR”同时影响 head 和其他解冻参数。

## 为什么 AMP 会出问题

AMP 用较低精度加速训练。当前日志证明：loss 仍是正常数字时，第一个 LoRA A 矩阵的梯度已经
出现非有限值。恢复器随后切换到 FP32。为了让正式 rank 对比不被恢复行为污染，当前候选关闭 AMP。

## 为什么当前实验降低 adapter LR

Brain Tumor 的 detection head 是新初始化的，需要较快学习；adapter 从预训练特征附近开始，
更新过快会让输入 head 的特征持续漂移。实验表明 adapter 与 head 同阶 LR 时性能在 epoch 4 崩塌，
adapter 使用 0.1 倍 LR 后，10 轮指标持续提升。这是“更新速度匹配”问题。

## 怎样看实验结果

- Precision 高：预测出来的框更少误报。
- Recall 高：真实目标更少漏检。
- mAP50：IoU=0.5 下的总体检测质量。
- mAP50-95：更严格、更重要的综合指标。
- train loss 降而 val loss 升：可能过拟合或特征/置信度崩塌。
- 单轮最好不能证明稳定；要看多轮趋势、重复种子和是否触发 recovery。

## 怎样判断项目优秀

1. 比较公平：rank 外的条件相同。
2. 可复现：命令、commit、环境、数据和随机种子齐全。
3. 数值稳定：无 NaN/Inf/recovery 污染。
4. 参数高效：在较少 trainable params 下取得有竞争力的 mAP。
5. 资源透明：同时报告显存、耗时和速度。
6. 结论克制：区分事实、机制、推断和假设。

## 最终怎样形成材料

- 工程报告：问题、修复、配置、结果表、资源表和复现命令。
- 论文：研究问题、head-adapter LR 匹配方法、rank/数据集实验、消融和统计稳定性。
- 比赛材料：最优 checkpoint、推理配置、速度/显存与错误案例可视化。

