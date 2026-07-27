# Issue #50 无人值守状态

- 当前任务：`visdrone_r4_stable_v2_seed0`
- 队列 PID：26359；训练链 PID：26362；后台方式：`nohup + setsid + flock`
- 已完成：Brain Tumor 与 VisDrone Stable V1 r=4/r=8/r=16；Adapter 0.1 单变量诊断
- 下一步：以 `lora_lr_mult=0.1` 完成 VisDrone Stable V2 r=4/r=8/r=16
- 状态文件：`runs/issue50/overnight_full_queue_status.json`
- 队列日志：`runs/issue50/overnight_full_queue.log`
- Git：本地与云端 commit `ac87a1b`；GitHub push 因网络失败待重试，无需人工认证
