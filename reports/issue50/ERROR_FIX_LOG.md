# Issue #50 错误与修复日志

## E01 Windows 路径无法在 Linux 使用

- **现象**：早期命令包含 Windows 个人绝对路径，云端无法解析。
- **根因**：脚本未从自身位置推导仓库根目录。
- **修复**：使用 `Path(__file__).resolve().parents[2]` 和仓库相对路径。
- **预防**：正式协议禁止硬编码个人绝对路径。
- **状态**：已修复。

## E02 `runs/detect/runs` 嵌套

- **现象**：输出目录被 CLI 默认 project 与手工路径重复拼接。
- **根因**：`project/name` 的职责不清。
- **修复**：脚本显式传入唯一 `project` 与 `name`，并在启动前拒绝已有目录。
- **状态**：已修复；旧结果保留，不作为正式结果。

## E03 Git 工作区脏状态

- **现象**：实验可能无法绑定唯一源码状态。
- **修复**：正式脚本拒绝脏工作树；诊断实验把状态写入 provenance/manifest。
- **状态**：正式链路已防护。

## E04 Git 作者身份缺失

- **原始报错**：`Author identity unknown`。
- **根因**：云端仓库没有 repo-local `user.name/user.email`。
- **排查命令**：`git config --get user.name`、`git config --get user.email`。
- **修复**：复用仓库上一提交的作者身份，仅写入当前仓库配置；未记录邮箱内容。
- **状态**：已修复。若项目要求 noreply 邮箱，应由仓库维护者确认后再替换。

## E05 GitHub push 网络失败

- **原始报错**：HTTP/2 stream 未正常关闭；HTTP/1.1 重试超时。
- **根因**：云端到 GitHub HTTPS 链路不稳定，非 commit 失败。
- **修复**：保留本地 commit，定期重试，只推当前分支。
- **不确定性**：远端仍落后于本地提交。
- **状态**：未完全解决，不阻塞训练。

## E06 云端 Codex 地区限制与 SSH 接管

- **来源**：用户提供的历史背景，当前会话未独立验证地区限制页面。
- **处理**：本地 Codex 通过 SSH 别名 `autodl-yolo` 接管；不读取或记录 SSH 私钥。
- **状态**：SSH 接管稳定可用。

## E07 `tmux` 不可用与后台训练

- **现象**：环境未保证提供 tmux。
- **修复**：训练统一使用 `nohup + setsid`，保存 PID、状态、退出码和独立日志。
- **验证**：进程 SID/TTY 与 SSH 解耦；PID 14266 在 SSH 断开后完成，exit code 0。
- **状态**：已修复。

## E08 预训练权重仅迁移 760/811

- **现象**：日志显示 `Transferred 760/811 items`。
- **根因**：Brain Tumor 类别数与预训练模型不同，detection head 有 51 项形状不匹配。
- **修复**：不强行加载不匹配 head；重新初始化并解冻 detection head。
- **影响**：新增约 347734 个可训练 head 参数，必须重视 head 学习率和 warmup。
- **状态**：预期行为，已记录。

## E09 AMP 下 adapter 梯度溢出

- **时间/实验**：2026-07-27，`brain_tumor_r4_amp_probe_e1`。
- **现象**：loss 有限，但训练触发 NaN recovery。
- **首个异常**：epoch 1 step 0，
  `model.base_model.model.4.conv.lora_A.default.weight`，group=`adapter`。
- **当时 loss**：box=1.7112、cls=4.5091、dfl=1.5735、mixture=3.0，均有限。
- **修改文件**：`ultralytics/engine/trainer.py` 增加首个非有限梯度诊断。
- **修复**：候选正式配置使用 `amp=False`。
- **状态**：数值异常已规避；AMP 安全训练仍值得后续研究。

## E10 NaN recovery 与 `last_healthy.pt`

- **源码机制**：loss、fitness、gradient、EMA 任一非有限标志触发恢复；连续最多 3 次。
- **保存机制**：启动前创建完整健康快照；有限 epoch 后通过临时文件原子替换。
- **恢复机制**：恢复 optimizer/scaler/EMA；AMP loss/gradient 异常会切换 FP32。
- **风险**：LoRA 在线模型当前只载入 adapter 张量，detection head 与 optimizer 恢复一致性尚未完全证明。
- **状态**：机制已定位；一致性风险待专门测试。

## E11 `optimizer=auto` 覆盖 `lr0`

- **现象**：正式组虽然 args 中 `lr0=0.001`，实际 auto 选择 AdamW `lr≈0.001667`。
- **根因**：PEFT auto optimizer 策略按类别数计算 LR。
- **修复**：诊断和候选正式配置显式设置 AdamW、`lr0` 和 bias warmup。
- **状态**：已解决配置歧义。

## E12 LR 过高不稳定、过低欠拟合

- **事实**：adapter multiplier=1.0 时，`lr0=8e-4` 在 epoch 4 验证性能崩塌。
- **事实**：干净 `lr0=3e-4` 三轮稳定但最佳 mAP50 仅 0.02995。
- **推断**：关键不是单一基础 LR，而是 detection head 与 adapter 的更新比例。
- **修复进展**：`lr0=8e-4, lora_lr_mult=0.1` 十轮稳定，最佳 mAP50=0.12106。
- **状态**：正在验证 multiplier=0.2。

## E13 终端乱码与 ANSI 进度日志

- **现象**：PowerShell/SSH 输出出现 UTF-8 误解码、ANSI 控制字符和动态进度行。
- **根因**：Windows 控制台编码与 TQDM 行覆盖格式。
- **处理**：文件统一 UTF-8/LF；解析日志时去除 ANSI，只把原始 `train.log` 作为证据。
- **状态**：显示层缓解，原始日志保留。

## E14 后台正式脚本找不到 `yolo`

- **错误**：`FileNotFoundError: No such file or directory: 'yolo'`。
- **原因**：`nohup + setsid` 非登录环境的 PATH 不含 Conda CLI。
- **修复**：归档失败 manifest/log/PID；重启时显式传入
  `--launcher /root/miniconda3/bin/yolo`。

## E15 VisDrone 预检归档与队列读取竞态

- **错误**：烟雾下载被停止后，父队列读取已归档的 `train.log`，触发 `FileNotFoundError` 并退出。
- **原因**：归档发生在子进程退出、父队列完成结果判定之前。
- **修复**：日志缺失时安全判定为失败；后续只在父队列退出后归档。云端慢速部分下载已保留在独立归档目录。
- **防复发**：替换数据前依次执行“停止子任务 → 等待队列退出 → 归档 → 校验新数据 → 恢复队列”。

