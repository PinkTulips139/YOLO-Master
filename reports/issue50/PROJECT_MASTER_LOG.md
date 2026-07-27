# Issue #50 项目总日志

## 记录口径

- **日志事实**：由训练日志、`results.csv`、`args.yaml` 或 manifest 直接证明。
- **源码机制**：由对应 commit 的实现直接证明。
- **合理推断**：多个事实支持，但尚未完成严格控制实验。
- **尚未验证**：现有材料不足，不作为项目结论。

## 早期记录：从复现到稳定性诊断

项目最初目标是在 Brain Tumor 与 VisDrone 上复现 Issue #50 的 LoRA `r=4/8/16`，公平记录精度、参数、显存、耗时和稳定性。接管初期先解决工程链路：训练脚本改用 `pathlib` 从仓库位置推导路径，消除 Windows/Linux 硬编码；明确 `project/name`，避免 `runs/detect/runs` 嵌套；正式任务绑定分支、commit、环境和 manifest；本地作为 tracked 文件主仓库，云端通过 SSH 接管，并以 `nohup + setsid + flock` 保障断线运行。

预训练日志显示权重迁移 **760/811**。其余 51 项因数据集类别数变化与检测头形状不匹配而不加载，检测头重新初始化并解冻。这是预期行为，却改变了研究问题：随机初始化 Head 需要较快学习，而 LoRA Adapter 应尽量保持预训练特征稳定。

第一条正式链路 `brain_tumor_r4_seed0` 在 epoch 1、step 0 发生 Adapter 非有限梯度；首个异常参数是 `model.base_model.model.4.conv.lora_A.default.weight`，当时 box/cls/dfl loss 仍有限。随后触发 recovery，后期指标归零。源码审计确认：loss、fitness、gradient 或 EMA 非有限均可触发恢复；`last_healthy.pt` 启动前建立，健康 epoch 后原子刷新，最多连续恢复 3 次。恢复会处理 optimizer、scaler 和 EMA，但 LoRA 在线模型恢复一致性仍有待专门验证。

## Phase 1：形成可复现的 Stable LoRA

诊断遵守单变量原则。关闭 AMP 后异常消失；显式使用 AdamW，并把 `warmup_bias_lr` 设为 0，避免 `optimizer=auto` 覆盖名义 `lr0` 以及 bias warmup 混杂。`lr0=3e-4` 稳定但学习不足，`1.2e-3` 波动较大，最终 Brain Tumor 选 `8e-4`。更关键的是 Adapter 学习率：倍率 1.0 的 10 轮实验数值有限却发生性能漂移，倍率 0.1 将 mAP50-95 从 0.02245 提升到 0.06630；0.2 弱于 0.1。Router 0.25 与 0.5 逐轮一致，因此停止继续搜索。

冻结后的稳定协议为：Brain Tumor 使用 `amp=False, AdamW, lr0=8e-4, lora_lr_mult=0.1, router_scale=0.5`；VisDrone 独立校准为 `lr0=1e-3`，其余稳定性设置相同。两个场景的 r4/r8/r16 均完成，且 r4 的 seed=0 最佳：Brain Tumor mAP50-95=0.06395，VisDrone=0.02878。最佳 rank 各补一个 seed，结果表明 Brain Tumor 的种子敏感性高于 VisDrone。

Phase 1 强基线改变了结论：Brain Tumor Head-only 和 Full FT 的 mAP50-95 分别为 0.36997、0.37121，显著高于 Stable LoRA。故项目不再以“LoRA 提升绝对精度”为主张，而转向稳定性诊断、参数效率审计和可信失败证据链。

## 最终封版补充：Phase 1 与 Phase 2 完整总结

Phase 2 将问题扩展为强微调比较，候选包括 Head-only、Neck+Head、Last Stage+Neck+Head、Full FT、Stable LoRA、Partial FT+LoRA 和 AMP-safe LoRA。云端可恢复队列共登记 **26 个任务：15 成功、11 失败**。VisDrone 发生 OOM 时只按既定顺序降低物理 batch，并以梯度累积保持有效 batch；失败目录、日志和退出状态均保留，不用部分结果冒充正式比较。

关键结果如下：

- Brain Tumor 的 Last Stage+Neck+Head 在 seed=0 达到 mAP50-95=0.40021；3 个有效种子的均值±标准差为 **0.39172±0.02316**，是当前可信的绝对性能方案。
- VisDrone Full FT 在匹配的 batch=4 运行中达到 0.15527；多种子汇总为 **0.15344±0.00193**，是当前绝对性能方案。
- Partial FT+LoRA 相比 Stable LoRA 明显缩小差距：Brain Tumor 为 0.35319，VisDrone 为 0.08565，但仍未超过对应最强非 LoRA 方案，且只有 seed=0。
- AMP-safe LoRA 原型在两个数据集均失败并产生全零结果，不能列为已验证创新。
- Phase 2 的若干非 LoRA 行把 `trainable_params` 解析成 0，因此最终 Pareto CSV 对这些方法的“参数效率最优”标记不可直接采信；绝对精度、显存与时间仍可读取。参数效率结论必须等重新审计参数计数后再定。

最终材料已通过完整性门控：报告、3 份汇总 CSV、7 张图、归档与 SHA256 均存在，`runs/issue50/PHASE2_COMPLETE.flag` 已创建。最终结果 commit 为 `f3d9060`，当前分支已推送。失败实验没有删除，它们是 AMP、OOM 和配置边界的正式证据。

## 文件地图与当前状态

- Phase 1 正式结果：`runs/issue50/formal/`
- Phase 2 结果：`runs/issue50/phase2/`
- 诊断证据：`runs/issue50/diagnostics/`
- Phase 1 总结：`reports/issue50/FINAL_PROJECT_REPORT_CN.md`
- Phase 2 总结：`reports/issue50/PHASE2_FINAL_REPORT_CN.md`
- 逐运行结果：`reports/issue50/PHASE2_FINAL_RESULTS.csv`
- 多种子与选择表：`PHASE2_SEED_SUMMARY.csv`、`PHASE2_PARETO_SUMMARY.csv`
- 最终交接：`reports/issue50/PHASE2_FINAL_HANDOFF.md`
- 协议与错误链：`FORMAL_EXPERIMENT_PROTOCOL.md`、`ERROR_FIX_LOG.md`

当前无待运行训练；项目处于学习版封版状态。

## 最合理的三项后续工作

1. 只读修正并复核 Phase 2 非 LoRA/Partial FT 的可训练参数计数，再重算可信 Pareto。
2. 重新设计 Adapter-only FP32 或独立 GradScaler；先做最小梯度单元测试，再运行单一短消融。
3. 在第三个独立数据集上，以统一显存协议和至少 3 个种子比较 Last Stage+Neck+Head、Full FT 与 Partial FT+LoRA。
