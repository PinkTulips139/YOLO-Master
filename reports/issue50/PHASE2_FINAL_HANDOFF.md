# Phase 2 审计修正版交接

- Brain Tumor 最佳正式运行：`/root/autodl-tmp/YOLO-Master/runs/issue50/phase2/phase2_brain_tumor_last_stage_neck_head_seed2`，Last stage + Neck + Head，mAP50-95=0.40944
- Brain Tumor 最佳权重：`/root/autodl-tmp/YOLO-Master/runs/issue50/phase2/phase2_brain_tumor_last_stage_neck_head_seed2/weights/best.pt`
- VisDrone 最佳正式运行：`/root/autodl-tmp/YOLO-Master/runs/issue50/phase2/phase2_visdrone_full_finetune_seed0_b4`，Full fine-tuning，mAP50-95=0.15527
- VisDrone 最佳权重：`/root/autodl-tmp/YOLO-Master/runs/issue50/phase2/phase2_visdrone_full_finetune_seed0_b4/weights/best.pt`
- 逐运行结果：`reports/issue50/PHASE2_FINAL_RESULTS.csv`
- 多种子汇总：`reports/issue50/PHASE2_SEED_SUMMARY.csv`
- Pareto：`reports/issue50/PHASE2_PARETO_SUMMARY.csv`
- 图表：`reports/issue50/PHASE2_FINAL_FIGURES/`
- 队列状态：`runs/issue50/phase2_queue_status.json`

所有正式结论均要求 formal_validity_status=passed。自动降 batch、OOM、not_executed、诊断和实现失败运行保留为证据，
但不进入正式均值与 Pareto。复现历史命令见各 run_manifest；本次审计没有执行训练。
