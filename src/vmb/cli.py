from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .args import build_server_command, command_to_shell, command_to_text
from .config import Task, expand_tasks, load_config
from .hardware import configured_gpus, configured_gpu_indices as normalize_gpu_indices, format_gpu_inventory
from .perf import run_perf_task
from .progress import log
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

    sub.add_parser("gpu", help="Show configured static GPU pool")

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
        return

    if args.command == "gpu":
        print_gpu_pool(config.global_config)
        return


def print_plan(global_config: dict, tasks: list[Task]) -> None:
    gpu_indices = normalize_gpu_indices(global_config.get("gpu_indices", []))
    gpus = configured_gpus(gpu_indices)
    print("GPUs:")
    print("  allocation_mode=static")
    print(f"  configured_gpu_indices={gpu_indices}")
    for gpu in gpus:
        print(f"  gpu={gpu.index} name={gpu.name}")
    print(f"Idle GPUs: {gpu_indices}")
    print("")
    print("Tasks:")
    host = global_config.get("host", "127.0.0.1")
    base_port = int(global_config.get("base_port", 8000))
    for idx, task in enumerate(tasks):
        cmd = build_server_command(task.model, host, base_port + idx, task.case)
        print(
            f"  {task.id}: kind={task.kind} model={task.model['id']} "
            f"type={task.model.get('type')} required_gpus={task.required_gpus}"
        )
        print(f"    server_cmd={command_to_text(cmd)}")


def print_gpu_pool(global_config: dict) -> None:
    indices = normalize_gpu_indices(global_config.get("gpu_indices", []))
    gpus = configured_gpus(indices)
    print("allocation_mode: static")
    print(f"configured_gpu_indices: {indices}")
    print(f"detected_gpus: {format_gpu_inventory(gpus)}")
    print(f"idle_gpus: {','.join(str(gpu) for gpu in indices) or 'none'}")
    print("external_usage_check: disabled")


async def run_tasks(
    global_config: dict,
    tasks: list[Task],
    update_baseline: bool,
) -> None:
    result_root = ensure_dir(Path(global_config.get("result_dir", "results")).resolve())
    log("run_start", task_count=len(tasks), result_root=result_root)
    scheduler = GpuScheduler(
        base_port=int(global_config.get("base_port", 8000)),
        poll_interval_sec=int(global_config.get("poll_interval_sec", 30)),
        configured_gpu_indices=global_config.get("gpu_indices", []),
    )

    async def launch(task: Task) -> None:
        log(
            "task_queued",
            task.id,
            kind=task.kind,
            model=task.model["id"],
            case=task.case["id"],
            required_gpus=task.required_gpus,
        )
        try:
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
                task.id,
            )
            log("task_done", task.id)
        except Exception as exc:
            log("task_failed", task.id, error=repr(exc))
            raise

    await asyncio.gather(*(launch(task) for task in tasks))
    log("run_done", task_count=len(tasks), result_root=result_root)


async def run_task_with_server(
    task: Task,
    allocation: Allocation,
    global_config: dict,
    result_root: Path,
    update_baseline: bool,
) -> None:
    host = global_config.get("host", "127.0.0.1")
    run_dir = ensure_dir(result_root / "runs" / task.id)
    task_case = dict(task.case)
    task_case["_task_id"] = task.id
    server = VllmServer(
        task.model,
        task_case,
        host,
        allocation.port,
        allocation.gpus,
        run_dir,
        int(global_config.get("ready_timeout_sec", 1800)),
        str(global_config.get("gpu_visible_devices_env", "CUDA_VISIBLE_DEVICES")),
    )
    visible_devices = ",".join(str(gpu) for gpu in allocation.gpus)
    visible_devices_env = str(global_config.get("gpu_visible_devices_env", "CUDA_VISIBLE_DEVICES"))
    server_command = f"{visible_devices_env}={visible_devices} {command_to_shell(server.command())}"
    manifest = {
        "task_id": task.id,
        "kind": task.kind,
        "model": task.model,
        "case": task.case,
        "gpus": allocation.gpus,
        "port": allocation.port,
        "server_command": server_command,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log("manifest_written", task.id, path=run_dir / "manifest.json")
    log("server_command", task.id, command=server_command)
    (run_dir / "server_command.txt").write_text(server_command + "\n", encoding="utf-8")

    try:
        log("task_start", task.id, kind=task.kind, run_dir=run_dir)
        await server.start()
        if task.kind == "perf":
            await run_perf_task(task, host, allocation.port, allocation.gpus, result_root, global_config)
        elif task.kind == "stability":
            await run_stability_task(task, host, allocation.port, allocation.gpus, result_root, update_baseline)
        else:
            raise ValueError(f"Unsupported task kind: {task.kind}")
    finally:
        await server.stop()
