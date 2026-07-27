"""Gate and run the remaining Brain Tumor Stable V1 ranks sequentially."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def inspect_run(repo: Path, rank: int) -> dict[str, object]:
    name = f"brain_tumor_r{rank}_stable_v1_seed0"
    run = repo / "runs" / "issue50" / "formal" / name
    log = repo / "runs" / "issue50" / "formal" / "logs" / f"{name}.log"
    manifest = repo / "runs" / "issue50" / "formal" / "logs" / name / "run_manifest.json"
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    rows = []
    if (run / "results.csv").exists():
        with (run / "results.csv").open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"])) if rows else None
    manifest_data = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    unstable = any(
        marker in text
        for marker in (
            "Non-finite gradient detected",
            "NaN recovery",
            "Global nonfinite training state",
            "Training failed: NaN",
        )
    )
    zero_tail = len(rows) >= 3 and all(float(row["metrics/mAP50-95(B)"]) == 0.0 for row in rows[-3:])
    return {
        "run_name": name,
        "success": manifest_data.get("success") is True,
        "exit_code": manifest_data.get("exit_code"),
        "epochs": len(rows),
        "best_epoch": int(best["epoch"]) if best else None,
        "map50": float(best["metrics/mAP50(B)"]) if best else None,
        "map50_95": float(best["metrics/mAP50-95(B)"]) if best else None,
        "unstable": unstable,
        "zero_tail": zero_tail,
        "artifacts": all(
            path.exists()
            for path in (
                run / "args.yaml",
                run / "results.csv",
                run / "weights" / "best.pt",
                run / "weights" / "last.pt",
                run / "weights" / "last_healthy.pt",
                log,
                manifest,
            )
        ),
    }


def write_status(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def passed(result: dict[str, object]) -> bool:
    return bool(
        result["success"]
        and result["artifacts"]
        and not result["unstable"]
        and not result["zero_tail"]
    )


def save_result(state: dict[str, object], result: dict[str, object]) -> None:
    runs = state.setdefault("runs", [])
    runs[:] = [item for item in runs if item.get("run_name") != result["run_name"]]
    runs.append(result)


def wait_for_process(pid: int, state: dict[str, object], status_path: Path) -> None:
    while process_exists(pid):
        state["heartbeat_at"] = now()
        write_status(status_path, state)
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-pid", type=int)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    status_path = repo / "runs" / "issue50" / "overnight_brain_queue_status.json"
    if status_path.exists():
        state = json.loads(status_path.read_text(encoding="utf-8"))
        state["resumed_at"] = now()
    else:
        state = {"started_at": now(), "status": "starting", "runs": []}
    write_status(status_path, state)

    for rank in (4, 8, 16):
        result = inspect_run(repo, rank)
        if passed(result):
            save_result(state, result)
            state["status"] = f"skipped_completed_r{rank}"
            write_status(status_path, state)
            continue

        active_pid = None
        if int(state.get("current_rank", -1)) == rank:
            candidate = int(state.get("current_pid", -1))
            if candidate > 0 and process_exists(candidate):
                active_pid = candidate
        if rank == 4 and args.initial_pid and process_exists(args.initial_pid):
            active_pid = args.initial_pid

        if active_pid is None:
            name = f"brain_tumor_r{rank}_stable_v1_seed0"
            result_dir = repo / "runs" / "issue50" / "formal" / name
            log_path = repo / "runs" / "issue50" / "formal" / "logs" / f"{name}.log"
            if result_dir.exists() or log_path.exists():
                save_result(state, result)
                state["status"] = f"stopped_incomplete_r{rank}"
                state["finished_at"] = now()
                write_status(status_path, state)
                raise SystemExit(f"Refusing to repeat incomplete run r={rank}: {result}")
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
            queue_log = repo / "runs" / "issue50" / "formal" / "logs" / f"{name}.queue.log"
            with queue_log.open("a", encoding="utf-8") as output:
                process = subprocess.Popen(command, cwd=repo, stdout=output, stderr=subprocess.STDOUT)
            active_pid = process.pid

        state["status"] = f"running_r{rank}"
        state["current_rank"] = rank
        state["current_pid"] = active_pid
        state["current_log"] = str(
            repo / "runs" / "issue50" / "formal" / "logs" / f"brain_tumor_r{rank}_stable_v1_seed0.log"
        )
        write_status(status_path, state)
        wait_for_process(active_pid, state, status_path)
        result = inspect_run(repo, rank)
        save_result(state, result)
        write_status(status_path, state)
        if not passed(result):
            state["status"] = f"stopped_after_r{rank}"
            state["finished_at"] = now()
            write_status(status_path, state)
            raise SystemExit(f"Stability gate failed after r={rank}: {result}")

    state["status"] = "completed"
    state["finished_at"] = now()
    write_status(status_path, state)


if __name__ == "__main__":
    main()
