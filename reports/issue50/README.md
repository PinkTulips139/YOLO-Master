# Issue #50 个人研究扩展

## 项目简介

本目录记录 YOLO-Master Issue #50 上游 LoRA 工作基础上的个人研究扩展。基础交付由多个 PR 逐步形成：[#69](https://github.com/Tencent/YOLO-Master/pull/69) 提供早期配置、rank sweep 与历史结果；[#84](https://github.com/Tencent/YOLO-Master/pull/84) 建立 v0.10 六组实验管线；[#102](https://github.com/Tencent/YOLO-Master/pull/102) 增加跨 rank 矩阵与双语指南；[#135](https://github.com/Tencent/YOLO-Master/pull/135) 统一 README、fraction 和脚本；[#166](https://github.com/Tencent/YOLO-Master/pull/166) 增加场景化配置、Router 策略和新结果。本项目进一步研究 AMP 数值稳定性、分组学习率、强微调基线、多随机种子和可审计的 Pareto 比较。

腾讯已合并的 [PR #178](https://github.com/Tencent/YOLO-Master/pull/178) 来自本项目的协议审计工作，贡献是澄清两套历史实验协议并防止结果 CSV 被静默覆盖，不是精度优化提交。

## 与上游稳定性修复的边界

- [#124](https://github.com/Tencent/YOLO-Master/pull/124)：AMP `index_add_` dtype alignment。
- [#125](https://github.com/Tencent/YOLO-Master/pull/125)：LoRA + MoE DDP ready-twice。
- [#170](https://github.com/Tencent/YOLO-Master/pull/170)：fallback RS-LoRA scaling。
- [#177](https://github.com/Tencent/YOLO-Master/pull/177)：fallback alpha warmup 的 EMA、验证、保存与 resume 生命周期。
- 本项目：在自身协议下定位 Adapter 非有限梯度，并比较 AMP=False 及 Detection Head/Adapter/Router 分组学习率。

本项目不宣称首次发现或解决所有 AMP、LoRA 或 RS-LoRA 稳定性问题。配置请求 `lora_backend=auto`、启用 `lora_use_rslora=True`，正式 rank sweep 实际使用 PEFT 后端。实验 commit 包含 #124/#125，但不包含稍后合并的 #170/#177；虽然后两者针对未实际使用的 fallback 路径，本文结果仍只代表当时研究分支，不能解释为当前 `upstream/main` 修复后 RS-LoRA 的最终性能。

相关开放工作中，[#114](https://github.com/Tencent/YOLO-Master/pull/114) 扩展其他数据集，[#179](https://github.com/Tencent/YOLO-Master/pull/179) 是另一套尚未合并的 Issue #50 rank sweep；本项目不依赖其结果。

## 核心研究问题

1. AMP 下 LoRA Adapter 为什么首先出现非有限梯度？
2. 降低 Adapter 学习率能否改善稳定性？
3. LoRA 与 Head-only、部分微调和全量微调相比，精度与资源位置如何？
4. 如何排除 OOM 恢复、重复目录、同 seed 重复和失败实现对结论的污染？

## 主要发现

- 首个已定位的异常是 AMP 条件下 Adapter `lora_A` 的非有限梯度；关闭 AMP 后稳定性改善。
- Adapter 采用 0.1×、Router 采用约 0.5×基础学习率是本项目的稳定配置，但不是通用最优值。
- Brain Tumor 的绝对性能最佳方法是 Last Stage + Neck + Head；VisDrone 是 Full Fine-tuning。
- Stable LoRA 只表示数值稳定，并不表示精度最高；LoRA 在两个数据集上均未取得绝对精度优势。
- AMP-safe LoRA 因实现失败而尚未验证，未进入正式结果。
- 本项目的额外贡献包括强基线、多随机种子、Adapter/Router 分组学习率、requested/actual batch 分离、OOM/重复目录/同 seed 审计、formal validity registry、Pareto 与语义完成门控。

## 核心结果

| 数据集 | 最佳方法 | 有效 seeds | mAP50-95 均值 ± 样本标准差 | 单次最高 |
|---|---|---:|---:|---:|
| Brain Tumor | Last Stage + Neck + Head | 0、1、2 | **0.39172 ± 0.02316** | **0.40944** |
| VisDrone | Full Fine-tuning（actual batch=4） | 0、1、2 | **0.15342 ± 0.00221** | **0.15527** |

VisDrone LoRA 实验使用 `fraction=0.2`。资源、单次结果和方法级比较见完整报告。

## 报告

- [中文研究扩展报告](RESEARCH_EXTENSION_CN.md)
- [English research extension](RESEARCH_EXTENSION_EN.md)
- [Phase 2 审计报告](PHASE2_FINAL_REPORT_CN.md)
- [Phase 2 最终交接](PHASE2_FINAL_HANDOFF.md)
- [项目过程总览](PROJECT_MASTER_LOG.md)

## 展示图

- [Brain Tumor 方法比较](SHOWCASE_FIGURES/brain_tumor_method_comparison.png)
- [VisDrone 方法比较](SHOWCASE_FIGURES/visdrone_method_comparison.png)
- [精度—可训练参数二维 Pareto](SHOWCASE_FIGURES/accuracy_parameter_pareto.png)
- [AMP 与 Adapter 学习率诊断](SHOWCASE_FIGURES/amp_adapter_lr_stability_diagnostic.png)
- [图表数据来源与过滤规则](SHOWCASE_FIGURES/README.md)

## 可复现文件索引

| 用途 | 文件 |
|---|---|
| 单次运行审计结果 | `PHASE2_FINAL_RESULTS.csv` |
| 多随机种子汇总 | `PHASE2_SEED_SUMMARY.csv` |
| 审计后的四目标非支配前沿 | `PHASE2_PARETO_SUMMARY.csv` |
| 完整实验登记 | `PHASE2_EXPERIMENT_REGISTRY.csv` |
| Phase 1 汇总 | `FINAL_RESULTS_SUMMARY.csv` |
| 正式实验协议 | `FORMAL_EXPERIMENT_PROTOCOL.md` |
| 图表生成脚本 | `generate_showcase_figures.py` |

## 局限性

- Stable LoRA 精度明显低于最佳传统微调基线。
- VisDrone LoRA 只覆盖 20% 训练数据。
- 部分方法只有一个随机种子。
- AMP-safe LoRA 是失败实现，不能作为已验证创新。
- 推理时间来自验证日志，不等同于严格部署基准。
- Adapter 0.1×与 Router 0.5×只得到当前任务内证据。
