# YOLO-Master Issue #50 æ­£å¼å®žéªŒåè®®

## ç›®æ ‡

éªŒè¯ YOLO-Master åœ¨åž‚ç±»ç›®æ ‡æ£€æµ‹åœºæ™¯ä¸­ä½¿ç”¨ LoRA/PEFT å¾®è°ƒçš„æ•ˆæžœä¸Žèµ„æºå¼€é”€ï¼Œé‡ç‚¹æ¯”è¾ƒ `rank=4/8/16`ã€‚

æœ¬åè®®åªå®šä¹‰æ­£å¼å®žéªŒæ–¹æ¡ˆï¼›å½“å‰é˜¶æ®µåªæ‰§è¡Œ `--dry-run`ï¼Œä¸å¯åŠ¨è®­ç»ƒã€‚

## å›ºå®šæ¡ä»¶

| é¡¹ç›® | è®¾ç½® |
|---|---|
| ä»“åº“ | Tencent/YOLO-Master |
| æœ¬åœ°åˆ†æ”¯ | `issue-50-lora-reproduction` |
| Conda çŽ¯å¢ƒ | `yolo-master` |
| è®­ç»ƒå…¥å£ | `scripts/issue50/run_rank_sweep.py` |
| æ±‡æ€»å…¥å£ | `scripts/issue50/summarize_runs.py` |
| å®˜æ–¹é¢„è®­ç»ƒæƒé‡ | `weights/YOLO-Master-EsMoE-N.pt` |
| è¾“å‡ºç›®å½• | `runs/issue50/formal/` |
| æ—¥å¿—ç›®å½• | `runs/issue50/formal/logs/` |
| resume | å¿…é¡»ä¸º `False` |
| è¦†ç›–æ—§å®žéªŒ | ç¦æ­¢é»˜è®¤è¦†ç›–ï¼Œ`exist_ok=False` |
| LoRA alpha | `alpha=2*r` |
| éšæœºç§å­ | `seed=0` |

## å®žéªŒçŸ©é˜µ

| åœºæ™¯ | æ•°æ®é…ç½® | Rank | Epochs | Batch | ImgSz | Fraction | Seed |
|---|---|---:|---:|---:|---:|---:|---:|
| Brain Tumor | `examples/lora_examples/yolo_master_brain_tumor_lora.yaml` | 4 | 40 | 16 | 640 | 1.0 | 0 |
| Brain Tumor | åŒä¸Š | 8 | 40 | 16 | 640 | 1.0 | 0 |
| Brain Tumor | åŒä¸Š | 16 | 40 | 16 | 640 | 1.0 | 0 |
| VisDrone | `examples/lora_examples/yolo_master_visdrone_lora.yaml` | 4 | 30 | 8 | 768 | 0.2 | 0 |
| VisDrone | åŒä¸Š | 8 | 30 | 8 | 768 | 0.2 | 0 |
| VisDrone | åŒä¸Š | 16 | 30 | 8 | 768 | 0.2 | 0 |

## æ‰§è¡Œå‘½ä»¤

å…ˆæ¿€æ´»çŽ¯å¢ƒå¹¶è¿›å…¥ä»“åº“ï¼š

```powershell
conda activate yolo-master
cd D:\æ¡Œé¢æ–‡ä»¶\LoRA\YOLO-Master
```

åªæ£€æŸ¥è®¡åˆ’ï¼Œä¸è®­ç»ƒï¼š

```powershell
python scripts/issue50/run_rank_sweep.py --dry-run
```

æ­£å¼è®­ç»ƒæ—¶ä½¿ç”¨åŒä¸€ä¸ªå…¥å£ï¼Œä½†åŽ»æŽ‰ `--dry-run`ï¼š

```powershell
python scripts/issue50/run_rank_sweep.py
```

è®­ç»ƒå®ŒæˆåŽæ±‡æ€»ç»“æžœï¼š

```powershell
python scripts/issue50/summarize_runs.py
```

## è®°å½•å­—æ®µ

`summarize_runs.py` ä¼šæ±‡æ€»ï¼š

- `mAP50`
- `mAP50_95`
- `precision`
- `recall`
- `best_epoch`
- `trainable_params`
- `adapter_params`
- `train_time_min`
- `peak_gpu_mem_gb`
- `weights_transferred`
- `weights_total`
- `nan_status`
- `gradient_checkpointing_requested`
- `gradient_checkpointing_actual`

## åˆ¤æ–­è§„åˆ™

- å…­ç»„å®žéªŒå¿…é¡»éƒ½ä»Ž `weights/YOLO-Master-EsMoE-N.pt` å¯åŠ¨ã€‚
- ä¸ä½¿ç”¨è‡ªåŠ¨ resumeï¼Œä¸è¦†ç›–æ—§ç›®å½•ï¼Œä¸æŠŠ `runs/` å’Œ `weights/` æäº¤åˆ° Gitã€‚
- rank å¯¹æ¯”åªåœ¨åŒä¸€æ•°æ®é…ç½®ã€åŒä¸€ seedã€åŒä¸€è¯„ä¼°åè®®ä¸‹æ¯”è¾ƒã€‚
- è‹¥å‡ºçŽ° OOMï¼Œå…ˆè®°å½•å¤±è´¥ï¼Œå†å¦å»ºå°æ˜¾å­˜è¡¥å……å®žéªŒï¼›ä¸è¦é™é»˜æ”¹å˜æ­£å¼å®žéªŒçŸ©é˜µã€‚
- è‹¥æ—¥å¿—å‡ºçŽ° NaNã€æƒé‡è¿ç§»å¼‚å¸¸æˆ– gradient checkpointing è¢«è·³è¿‡ï¼Œéœ€è¦åœ¨ç»“æžœè¡¨ä¸­å•ç‹¬æ ‡è®°ã€‚

## å½“å‰çŠ¶æ€åˆ¤æ–­

æ­£å¼å®žéªŒè„šæœ¬å’Œæ±‡æ€»è„šæœ¬å·²å‡†å¤‡ï¼›å½“å‰åªå…è®¸ dry-run æ£€æŸ¥å…­ç»„å‘½ä»¤ï¼Œä¸æŠŠ dry-run å½“ä½œå®žéªŒç»“æžœã€‚

## ä¸‹ä¸€æ­¥å»ºè®®

1. åœ¨æœ¬åœ°æ‰§è¡Œ `python scripts/issue50/run_rank_sweep.py --dry-run` ç¡®è®¤å‘½ä»¤ã€‚
2. äº‘ GPU å‡†å¤‡å¥½åŽï¼Œå…ˆç¡®è®¤æƒé‡å’Œæ•°æ®é›†è·¯å¾„ï¼Œå†è¿è¡Œæ­£å¼è®­ç»ƒã€‚
3. è®­ç»ƒç»“æŸåŽè¿è¡Œ `python scripts/issue50/summarize_runs.py` ç”Ÿæˆæ±‡æ€»è¡¨ã€‚
