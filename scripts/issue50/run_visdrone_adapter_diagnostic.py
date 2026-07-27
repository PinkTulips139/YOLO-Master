#!/usr/bin/env python3
"""Run the single-variable VisDrone adapter-LR diagnostic with reproducibility metadata."""

from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


RUN_NAME = "visdrone_r4_ampoff_lr1e3_adapt01_e10"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def output(root: Path, *command: str) -> str:
    result = subprocess.run(
        command,
        cwd=root,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip()


def main() -> None:
    import fcntl

    root = Path(__file__).resolve().parents[2]
    result_dir = root / "runs" / "issue50" / "diagnostics" / RUN_NAME
    log_dir = root / "runs" / "issue50" / "diagnostics" / "logs" / RUN_NAME
    status_path = root / "runs" / "issue50" / "visdrone_adapter_diagnostic_status.json"
    lock_path = root / "runs" / "issue50" / "visdrone_adapter_diagnostic.lock"
    if result_dir.exists() or log_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing diagnostic: {RUN_NAME}")
    log_dir.mkdir(parents=True)
    command = [
        "/root/miniconda3/bin/yolo",
        "train",
        f"cfg={root / 'examples/lora_examples/yolo_master_visdrone_lora.yaml'}",
        f"model={root / 'weights/YOLO-Master-EsMoE-N.pt'}",
        "pretrained=True",
        "resume=False",
        "exist_ok=False",
        "val=True",
        "plots=False",
        "deterministic=True",
        "save=True",
        "lora_r=4",
        "lora_alpha=8",
        "epochs=10",
        "batch=8",
        "imgsz=768",
        "fraction=0.2",
        "seed=0",
        "workers=8",
        "device=0",
        "amp=False",
        "optimizer=AdamW",
        "lr0=0.001",
        "warmup_bias_lr=0.0",
        "lora_lr_mult=0.1",
        "moe_router_lr_scale=0.5",
        f"project={root / 'runs/issue50/diagnostics'}",
        f"name={RUN_NAME}",
    ]
    (log_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    (log_dir / "provenance.txt").write_text(
        f"branch={output(root, 'git', 'branch', '--show-current')}\n"
        f"commit={output(root, 'git', 'rev-parse', 'HEAD')}\n",
        encoding="utf-8",
    )
    environment = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": output(root, "/root/miniconda3/bin/python", "-c", "import torch; print(torch.__version__)"),
        "cuda": output(root, "/root/miniconda3/bin/python", "-c", "import torch; print(torch.version.cuda)"),
        "gpu": output(root, "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"),
    }
    (log_dir / "environment.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in environment.items()), encoding="utf-8"
    )
    status = {"run": RUN_NAME, "status": "starting", "started_at": now(), "pid": os.getpid()}
    atomic_json(status_path, status)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("Another adapter diagnostic owns the lock.") from error
        status["status"] = "running"
        atomic_json(status_path, status)
        with (log_dir / "train.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            status["training_pid"] = process.pid
            atomic_json(status_path, status)
            return_code = process.wait()
    status.update(
        {"status": "completed" if return_code == 0 else "failed", "exit_code": return_code, "finished_at": now()}
    )
    (log_dir / "status.txt").write_text(
        f"exit_code={return_code}\nstatus={status['status']}\nstarted_at={status['started_at']}\nfinished_at={status['finished_at']}\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "/root/miniconda3/bin/python",
            "scripts/issue50/build_diagnostic_manifests.py",
            "--run",
            RUN_NAME,
        ],
        cwd=root,
        check=False,
    )
    manifest = log_dir / "run_manifest.json"
    status["manifest"] = str(manifest)
    status["artifacts_complete"] = (
        json.loads(manifest.read_text(encoding="utf-8")).get("artifacts_complete", False)
        if manifest.exists()
        else False
    )
    atomic_json(status_path, status)
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
