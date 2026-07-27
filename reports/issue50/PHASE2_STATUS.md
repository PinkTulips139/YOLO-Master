# Issue #50 Phase 2 Status

- Queue: prepared for cloud launch
- Seed-0 candidates: Head-only, Neck+Head, Last-stage+Neck+Head, full fine-tuning, Stable LoRA, Neck+Head+LoRA, AMP-safe LoRA
- Existing complete Phase 1/baseline runs are inspected and skipped.
- VisDrone OOM policy: physical batch 8 → 4 → 2 → 1, while `nbs=64` preserves gradient accumulation policy.
- After seed=0, the absolute-best and mAP50-95-per-trainable-parameter candidates receive seed=1/2; completed seeds are reused.
- State: `runs/issue50/phase2_queue_status.json`
- Log: `runs/issue50/phase2_queue.log`
