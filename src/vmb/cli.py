from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .args import build_server_command, command_to_text
from .config import Task, expand_tasks, load_config
from .hardware import idle_gpu_indices, list_gpus
from .perf import run_perf_task
from .scheduler import Allocation, GpuScheduler, run_scheduled
from .server import VllmServer
from .stability import run_stability_task
from .util import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(prog="vmb", description="vLLM model benchmark orchestrator")
    parser.add_argument("--config", default="bench_config.py", help="Path to bench_config.py")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Show task and resource plan")
    plan.add_argument("--kind", choices=["perf", "stability", "all"], default="all")
    plan.add_argument("--model", help="Model id, served name, or model path to include")

    run = sub.add_parser("run", help="Run benchmarks")
    run.add_argument("kind", choices=["perf", "stability", "all"])
    run.add_argument("--model", help="Model id, served name, or model path to include")
    run.add_argument("--update-baseline", action="store_true", help="Refresh stability baselines")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "plan":
        tasks = expand_tasks(config, args.kind, args.model)
        print_plan(config.global_config, tasks)
        return

    if args.command == "run":
        tasks = expand_tasks(config, args.kind, args.model)
        asyncio.run(run_tasks(config.global_config, tasks, args.update_baseline))


def print_plan(global_config: dict, tasks: list[Task]) -> None:
    idle_threshold = int(global_config.get("idle_memory_threshold_mb", 1024))
    print("GPUs:")
    for gpu in list_gpus():
        print(
            f"  gpu={gpu.index} name={gpu.name} used={gpu.memory_used_mb}MB "
            f"total={gpu.memory_total_mb}MB"
        )
    print(f"Idle GPUs: {idle_gpu_indices(idle_threshold)}")
    print("")
    print("Tasks:")
    host = global_config.get("host", "127.0.0.1")
    base_port = int(global_config.get("base_port", 8000))
    for idx, task in enumerate(tasks):
        cmd = build_server_command(task.model, host, base_port + idx)
        print(
            f"  {task.id}: kind={task.kind} model={task.model['id']} "
            f"type={task.model.get('type')} required_gpus={task.required_gpus}"
        )
        print(f"    server_cmd={command_to_text(cmd)}")


async def run_tasks(
    global_config: dict,
    tasks: list[Task],
    update_baseline: bool,
) -> None:
    result_root = ensure_dir(Path(global_config.get("result_dir", "results")).resolve())
    scheduler = GpuScheduler(
        base_port=int(global_config.get("base_port", 8000)),
        poll_interval_sec=int(global_config.get("poll_interval_sec", 30)),
        idle_memory_threshold_mb=int(global_config.get("idle_memory_threshold_mb", 1024)),
    )

    async def launch(task: Task) -> None:
        await run_scheduled(
            scheduler,
            task.required_gpus,
            lambda allocation: run_task_with_server(
                task,
                allocation,
                global_config,
                result_root,
                update_baseline,
            ),
        )

    await asyncio.gather(*(launch(task) for task in tasks))


async def run_task_with_server(
    task: Task,
    allocation: Allocation,
    global_config: dict,
    result_root: Path,
    update_baseline: bool,
) -> None:
    host = global_config.get("host", "127.0.0.1")
    run_dir = ensure_dir(result_root / "runs" / task.id)
    manifest = {
        "task_id": task.id,
        "kind": task.kind,
        "model": task.model,
        "case": task.case,
        "gpus": allocation.gpus,
        "port": allocation.port,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    server = VllmServer(
        task.model,
        host,
        allocation.port,
        allocation.gpus,
        run_dir,
        int(global_config.get("ready_timeout_sec", 1800)),
    )
    try:
        await server.start()
        if task.kind == "perf":
            await run_perf_task(task, host, allocation.port, allocation.gpus, result_root, global_config)
        elif task.kind == "stability":
            await run_stability_task(task, host, allocation.port, allocation.gpus, result_root, update_baseline)
        else:
            raise ValueError(f"Unsupported task kind: {task.kind}")
    finally:
        await server.stop()
