# Issue #50 最终交接

## 可用于正式材料

- 最终报告：`reports/issue50/FINAL_PROJECT_REPORT_CN.md`
- 完整结构化结果：`reports/issue50/FINAL_RESULTS_SUMMARY.csv`
- 图表：`reports/issue50/FINAL_FIGURES/`
- 正式 LoRA 结果：`runs/issue50/formal/{brain_tumor,visdrone}_r{4,8,16}_stable_*_seed0`
- 第二种子：`runs/issue50/formal/brain_tumor_r4_stable_v1_seed1`、`visdrone_r4_stable_v2_seed1`
- Brain Tumor 公平基线：`runs/issue50/formal/baselines/brain_tumor_{head_only,full_finetune}_seed0`

最佳稳定 LoRA 权重：

- Brain Tumor：`runs/issue50/formal/brain_tumor_r4_stable_v1_seed0/weights/best.pt`
- VisDrone：`runs/issue50/formal/visdrone_r4_stable_v2_seed1/weights/best.pt`（单次最高）；正式 rank sweep 使用 seed=0 同名 r4 目录。

## 仅作为诊断证据

- AMP 首次溢出：`runs/issue50/diagnostics/brain_tumor_r4_amp_probe_e1`
- AMP-off 短实验：`runs/issue50/diagnostics/brain_tumor_r4_ampoff_e3`
- Adapter 倍率消融：`brain_tumor_r4_ampoff_lr8e4_e10` 与 `brain_tumor_r4_ampoff_lr8e4_adapt01_e10`
- 早期不稳定正式链路：`runs/issue50/formal/brain_tumor_r4_seed0`
- VisDrone OOM 基线：`runs/issue50/formal/baselines/visdrone_head_only_seed0`

## 复现命令

```bash
# 查看固定正式矩阵
python scripts/issue50/run_rank_sweep.py --dry-run

# 运行指定正式任务
python scripts/issue50/run_rank_sweep.py --only brain_tumor_r4_stable_v1_seed0

# 公平基线预览/可恢复串行队列
python scripts/issue50/run_final_baselines.py --dry-run
python scripts/issue50/run_final_baselines.py --device 0

# 从保存产物重建最终 CSV 与图表
python scripts/issue50/build_final_materials.py
```

云端仓库为 `/root/autodl-tmp/YOLO-Master`，本地仓库为 `D:\桌面文件\LoRA\YOLO-Master`。最终 commit、push 状态、归档路径与 SHA256 在最终提交时补入本文件。

## 结果边界

- 所有正式 LoRA rank/seed 与 Brain Tumor 基线满足完整产物、退出码 0 和非有限值门控。
- VisDrone head-only 在匹配协议下 OOM；full fine-tune未启动，二者不能作为正式精度比较。
- 选择性 Adapter FP32 尚未可信实现，不应声称已验证。

## 最值得继续的方向

实现真正的 Adapter-only FP32/独立缩放，并将随机初始化检测头、Router、Adapter 明确拆成三个学习率组；随后在显存允许的匹配协议下，以至少 3 个种子验证稳定性、参数效率和精度。
