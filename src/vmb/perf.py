from __future__ import annotations

import asyncio
import os
from pathlib import Path

from .args import build_perf_command, command_to_shell
from .config import Task
from .progress import log
from .results import append_perf_result
from .util import ensure_dir


async def run_perf_task(
    task: Task,
    host: str,
    port: int,
    gpus: list[int],
    result_root: Path,
    global_config: dict,
) -> None:
    raw_dir = ensure_dir(result_root / "raw")
    csv_path = result_root / "csv" / "perf_results.csv"
    cmd, raw_json_path = build_perf_command(task, host, port, raw_dir, global_config)
    log_dir = ensure_dir(result_root / "runs" / task.id)
    client_command = command_to_shell(cmd)
    log("perf_client_command", task.id, command=client_command)
    (log_dir / "client_command.txt").write_text(client_command + "\n", encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    log("perf_start", task.id, raw_json=raw_json_path, stdout=log_dir / "bench.stdout.log", stderr=log_dir / "bench.stderr.log")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout, stderr = await proc.communicate()
    log("perf_process_done", task.id, returncode=proc.returncode)

    (log_dir / "bench.stdout.log").write_bytes(stdout)
    (log_dir / "bench.stderr.log").write_bytes(stderr)
    log("perf_logs_written", task.id, stdout=log_dir / "bench.stdout.log", stderr=log_dir / "bench.stderr.log")

    if proc.returncode == 0:
        append_perf_result(csv_path, task, gpus, raw_json_path, "ok")
        log("perf_result_written", task.id, status="ok", csv=csv_path, raw_json=raw_json_path)
    else:
        append_perf_result(
            csv_path,
            task,
            gpus,
            raw_json_path,
            "failed",
            stderr.decode("utf-8", errors="replace")[-2000:],
        )
        log("perf_result_written", task.id, status="failed", csv=csv_path, raw_json=raw_json_path)
