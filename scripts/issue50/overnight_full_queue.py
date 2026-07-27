"""Resume after the Brain queue, preflight VisDrone, then run its formal rank sweep."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def alive(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    try:
        if stat.exists() and stat.read_text(encoding="utf-8").split()[2] == "Z":
            return False
    except (FileNotFoundError, IndexError, PermissionError, ProcessLookupError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def active_training_pids() -> list[int]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if alive(int(entry.name)) and ("run_rank_sweep.py" in command or "/bin/yolo train" in command):
            found.append(int(entry.name))
    return found


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def metrics(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {}
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return {
        "epochs": len(rows),
        "best_epoch": int(best["epoch"]),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "map50": float(best["metrics/mAP50(B)"]),
        "map50_95": float(best["metrics/mAP50-95(B)"]),
        "elapsed_seconds": float(rows[-1]["time"]),
    }


def inspect_formal(repo: Path, scene: str, rank: int) -> dict:
    name = f"{scene}_r{rank}_stable_v1_seed0"
    run = repo / "runs" / "issue50" / "formal" / name
    logs = repo / "runs" / "issue50" / "formal" / "logs"
    log_path = logs / f"{name}.log"
    manifest_path = logs / name / "run_manifest.json"
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    memory = [float(value) for value in re.findall(r"\b(\d+(?:\.\d+)?)G\b", text)]
    result = {
        "name": name,
        **metrics(run / "results.csv"),
        "exit_code": manifest.get("exit_code"),
        "success": manifest.get("success") is True,
        "peak_gpu_memory_gib": max(memory, default=None),
        "unstable": any(
            marker in text
            for marker in (
                "Non-finite gradient detected",
                "NaN recovery",
                "Global nonfinite training state",
                "Training failed: NaN",
            )
        ),
        "artifacts": all(
            path.exists()
            for path in (
                run / "args.yaml",
                run / "results.csv",
                run / "weights" / "best.pt",
                run / "weights" / "last.pt",
                log_path,
                manifest_path,
            )
        ),
    }
    csv_metrics = metrics(run / "results.csv")
    result["zero_metric_tail"] = False
    if csv_metrics and (run / "results.csv").exists():
        with (run / "results.csv").open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        result["zero_metric_tail"] = len(rows) >= 3 and all(
            all(float(row[key]) == 0.0 for key in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)"))
            for row in rows[-3:]
        )
    result["passed"] = bool(
        result["success"] and result["artifacts"] and not result["unstable"] and not result["zero_metric_tail"]
    )
    return result


def dataset_counts(root: Path) -> dict:
    return {
        "root": str(root),
        "images_train": len(list((root / "images" / "train").glob("*"))),
        "images_val": len(list((root / "images" / "val").glob("*"))),
        "labels_train": len(list((root / "labels" / "train").glob("*.txt"))),
        "labels_val": len(list((root / "labels" / "val").glob("*.txt"))),
    }


def locate_visdrone(repo: Path) -> Path | None:
    candidates = (
        Path("/root/autodl-tmp/datasets/VisDrone"),
        repo.parent / "datasets" / "VisDrone",
        repo / "datasets" / "VisDrone",
    )
    return next((root for root in candidates if (root / "images" / "train").is_dir()), None)


def wait(pid: int, state: dict, state_path: Path) -> None:
    while alive(pid):
        state["heartbeat_at"] = now()
        atomic_json(state_path, state)
        time.sleep(30)


def launch(command: list[str], repo: Path, output_path: Path, state: dict, state_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=repo, stdout=output, stderr=subprocess.STDOUT)
    state["current_pid"] = process.pid
    state["current_command"] = command
    state["current_log"] = str(output_path)
    atomic_json(state_path, state)
    while process.poll() is None:
        state["heartbeat_at"] = now()
        atomic_json(state_path, state)
        time.sleep(30)
    return process.wait()


def smoke_visdrone(repo: Path, state: dict, state_path: Path) -> dict:
    name = "visdrone_r4_preflight_e1"
    run = repo / "runs" / "issue50" / "diagnostics" / name
    logs = repo / "runs" / "issue50" / "diagnostics" / "logs" / name
    log_path = logs / "train.log"
    if (run / "results.csv").exists() and (run / "weights" / "best.pt").exists():
        return {"name": name, **metrics(run / "results.csv"), "passed": True, "skipped": True}
    if run.exists() or log_path.exists():
        return {"name": name, "passed": False, "reason": "incomplete pre-existing smoke run"}
    command = [
        "/root/miniconda3/bin/yolo",
        "train",
        f"cfg={repo / 'examples/lora_examples/yolo_master_visdrone_lora.yaml'}",
        f"model={repo / 'weights/YOLO-Master-EsMoE-N.pt'}",
        "pretrained=True",
        "resume=False",
        "exist_ok=False",
        "lora_r=4",
        "lora_alpha=8",
        "epochs=1",
        "batch=2",
        "imgsz=320",
        "fraction=0.01",
        "workers=2",
        "device=0",
        "amp=False",
        "optimizer=AdamW",
        "lr0=0.001",
        "warmup_bias_lr=0.0",
        "lora_lr_mult=0.5",
        "plots=False",
        f"project={repo / 'runs/issue50/diagnostics'}",
        f"name={name}",
    ]
    logs.mkdir(parents=True)
    (logs / "command.txt").write_text(" ".join(map(str, command)) + "\n", encoding="utf-8")
    state["status"] = "visdrone_preflight"
    return_code = launch(command, repo, log_path, state, state_path)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    result = {
        "name": name,
        **metrics(run / "results.csv"),
        "exit_code": return_code,
        "passed": bool(
            return_code == 0
            and (run / "results.csv").exists()
            and (run / "weights" / "best.pt").exists()
            and "Non-finite gradient detected" not in text
            and "NaN recovery" not in text
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--handoff-queue-pid", type=int)
    parser.add_argument("--handoff-training-pid", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    runs = repo / "runs" / "issue50"
    state_path = runs / "overnight_full_queue_status.json"
    lock_path = runs / "overnight_full_queue.lock"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stages": [
                        "brain_tumor_r4_stable_v1_seed0",
                        "brain_tumor_r8_stable_v1_seed0",
                        "brain_tumor_r16_stable_v1_seed0",
                        "visdrone_r4_preflight_e1",
                        "visdrone_r4_stable_v1_seed0",
                        "visdrone_r8_stable_v1_seed0",
                        "visdrone_r16_stable_v1_seed0",
                    ],
                    "handoff_queue_pid": args.handoff_queue_pid,
                    "handoff_training_pid": args.handoff_training_pid,
                    "visdrone_yaml": (repo / "ultralytics/cfg/datasets/VisDrone.yaml").exists(),
                    "weights": (repo / "weights/YOLO-Master-EsMoE-N.pt").exists(),
                    "dataset": str(locate_visdrone(repo)) if locate_visdrone(repo) else None,
                },
                indent=2,
            )
        )
        return
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {
        "started_at": now(),
        "status": "waiting_brain_queue",
        "completed": [],
    }
    atomic_json(state_path, state)

    if args.handoff_training_pid and alive(args.handoff_training_pid):
        state["status"] = "waiting_handoff_training"
        state["waiting_for_pid"] = args.handoff_training_pid
        wait(args.handoff_training_pid, state, state_path)
    if args.handoff_queue_pid and alive(args.handoff_queue_pid):
        state["status"] = "stopping_previous_queue"
        atomic_json(state_path, state)
        os.kill(args.handoff_queue_pid, signal.SIGTERM)
        for _ in range(30):
            if not alive(args.handoff_queue_pid):
                break
            time.sleep(1)
        if alive(args.handoff_queue_pid):
            raise SystemExit("Previous queue did not stop after SIGTERM.")
    for training_pid in active_training_pids():
        state["status"] = "waiting_handoff_race"
        state["waiting_for_pid"] = training_pid
        wait(training_pid, state, state_path)

    if args.wait_pid and alive(args.wait_pid):
        state["waiting_for_pid"] = args.wait_pid
        wait(args.wait_pid, state, state_path)

    current_pid = int(state.get("current_pid") or -1)
    if alive(current_pid):
        state["status"] = "resuming_active_child"
        state["waiting_for_pid"] = current_pid
        wait(current_pid, state, state_path)

    import fcntl

    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("Another full queue owns the lock.") from error

        brain = []
        for rank in (4, 8, 16):
            result = inspect_formal(repo, "brain_tumor", rank)
            if not result["passed"]:
                name = f"brain_tumor_r{rank}_stable_v1_seed0"
                result_dir = runs / "formal" / name
                log_path = runs / "formal" / "logs" / f"{name}.log"
                if result_dir.exists() or log_path.exists():
                    state["status"] = f"blocked_incomplete_{name}"
                    atomic_json(state_path, state)
                    raise SystemExit(f"Refusing to repeat incomplete run: {name}")
                state["status"] = f"running_{name}"
                command = [
                    "/root/miniconda3/bin/python",
                    "scripts/issue50/run_rank_sweep.py",
                    "--scene",
                    "brain_tumor",
                    "--ranks",
                    str(rank),
                    "--device",
                    "0",
                    "--launcher",
                    "/root/miniconda3/bin/yolo",
                ]
                return_code = launch(
                    command, repo, runs / "formal" / "logs" / f"{name}.queue.log", state, state_path
                )
                result = inspect_formal(repo, "brain_tumor", rank)
                result["queue_exit_code"] = return_code
            brain.append(result)
            state["brain_tumor"] = brain
            atomic_json(state_path, state)
            if not result["passed"]:
                state["status"] = f"failed_brain_r{rank}"
                atomic_json(state_path, state)
                raise SystemExit(f"Brain Tumor gate failed after r={rank}.")

        yaml_path = repo / "ultralytics/cfg/datasets/VisDrone.yaml"
        weight_path = repo / "weights/YOLO-Master-EsMoE-N.pt"
        disk = shutil.disk_usage("/root/autodl-tmp")
        state["visdrone_precheck"] = {
            "yaml_exists": yaml_path.exists(),
            "weight_exists": weight_path.exists(),
            "classes": 10,
            "disk_free_gib": round(disk.free / 2**30, 2),
            "dataset_before": str(locate_visdrone(repo)) if locate_visdrone(repo) else None,
        }
        atomic_json(state_path, state)
        if not yaml_path.exists() or not weight_path.exists() or disk.free < 5 * 2**30:
            state["status"] = "blocked_visdrone_precheck"
            atomic_json(state_path, state)
            raise SystemExit("VisDrone static precheck failed.")

        smoke = smoke_visdrone(repo, state, state_path)
        state["visdrone_smoke"] = smoke
        root = locate_visdrone(repo)
        state["visdrone_precheck"]["dataset_after"] = dataset_counts(root) if root else None
        atomic_json(state_path, state)
        if not smoke["passed"] or root is None:
            state["status"] = "blocked_visdrone_smoke"
            atomic_json(state_path, state)
            raise SystemExit("VisDrone smoke gate failed.")

        for rank in (4, 8, 16):
            result = inspect_formal(repo, "visdrone", rank)
            if result["passed"]:
                state["completed"].append(result)
                atomic_json(state_path, state)
                continue
            name = f"visdrone_r{rank}_stable_v1_seed0"
            result_dir = runs / "formal" / name
            log_path = runs / "formal" / "logs" / f"{name}.log"
            if result_dir.exists() or log_path.exists():
                state["status"] = f"blocked_incomplete_{name}"
                atomic_json(state_path, state)
                raise SystemExit(f"Refusing to repeat incomplete run: {name}")
            state["status"] = f"running_{name}"
            command = [
                "/root/miniconda3/bin/python",
                "scripts/issue50/run_rank_sweep.py",
                "--scene",
                "visdrone",
                "--ranks",
                str(rank),
                "--device",
                "0",
                "--launcher",
                "/root/miniconda3/bin/yolo",
            ]
            return_code = launch(
                command, repo, runs / "formal" / "logs" / f"{name}.queue.log", state, state_path
            )
            result = inspect_formal(repo, "visdrone", rank)
            result["queue_exit_code"] = return_code
            state["completed"].append(result)
            atomic_json(state_path, state)
            if not result["passed"]:
                state["status"] = f"failed_{name}"
                atomic_json(state_path, state)
                raise SystemExit(f"VisDrone gate failed: {name}")

        state["status"] = "completed"
        state["finished_at"] = now()
        atomic_json(state_path, state)


if __name__ == "__main__":
    main()
