# Issue #50 无人值守状态

- 当前任务：准备 `visdrone_r4_ampoff_lr1e3_adapt01_e10`
- 队列 PID：待启动；后台方式：`nohup + setsid + flock`
- 已完成：Brain Tumor 与 VisDrone Stable V1 r=4/r=8/r=16；VisDrone 数据预检与烟雾测试
- 下一步：单变量验证 VisDrone Adapter LR 倍率 `0.5→0.1`
- 状态文件：`runs/issue50/overnight_full_queue_status.json`
- 队列日志：`runs/issue50/overnight_full_queue.log`
- Git：本地与云端 commit `043716b`；GitHub push 因网络失败待重试，无需人工认证
