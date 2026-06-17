from __future__ import annotations

import asyncio
import os
from pathlib import Path

from .args import build_perf_command
from .config import Task
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

    env = os.environ.copy()
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout, stderr = await proc.communicate()

    log_dir = ensure_dir(result_root / "runs" / task.id)
    (log_dir / "bench.stdout.log").write_bytes(stdout)
    (log_dir / "bench.stderr.log").write_bytes(stderr)

    if proc.returncode == 0:
        append_perf_result(csv_path, task, gpus, raw_json_path, "ok")
    else:
        append_perf_result(
            csv_path,
            task,
            gpus,
            raw_json_path,
            "failed",
            stderr.decode("utf-8", errors="replace")[-2000:],
        )
