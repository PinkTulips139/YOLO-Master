# Issue #50 Phase 2 Final Handoff

- Branch: `issue-50-lora-reproduction`
- Source commit before finalization: `39accf4a571dcaba7366bd6a86bcca71bb0d1b29`
- Formal results: `runs/issue50/phase2/` and reused Phase 1 formal directories
- Diagnostic results: `runs/issue50/diagnostics/`
- Reproduce queue: `python scripts/issue50/phase2_performance_queue.py --device 0 --launcher /root/miniconda3/bin/yolo`
- Resume finalizer: `nohup setsid flock -n runs/issue50/phase2_finalize.lock /root/miniconda3/bin/python scripts/issue50/phase2_finalize_after_queue.py > runs/issue50/phase2_finalize.log 2>&1 < /dev/null &`

## Best absolute checkpoints

- Brain Tumor / Last stage + Neck + Head: `/root/autodl-tmp/YOLO-Master/runs/issue50/phase2/phase2_brain_tumor_last_stage_neck_head_seed0/weights/best.pt`
- VisDrone / Full fine-tuning: `/root/autodl-tmp/YOLO-Master/runs/issue50/phase2/phase2_visdrone_full_finetune_seed0_b4/weights/best.pt`

## Next research

优先在独立数据集验证 AMP-safe Adapter 路径与三路学习率，并以至少 3 个种子复验；不要继续无界调参。
