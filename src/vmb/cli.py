from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .args import build_server_command, command_to_shell, command_to_text
from .config import Task, expand_tasks, load_config
from .hardware import configured_gpus, configured_gpu_indices as normalize_gpu_indices, format_gpu_inventory
from .perf import run_perf_task
from .progress import log
from .scheduler import Allocation, GpuScheduler, run_scheduled
from .server import VllmServer
from .stability import run_stability_task
from .util import ensure_dir


@dataclass(frozen=True)
class TaskGroup:
    id: str
    model: dict[str, Any]
    tasks: list[Task]
    required_gpus: int
    server_case: dict[str, Any]


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
    print("Server groups:")
    host = global_config.get("host", "127.0.0.1")
    base_port = int(global_config.get("base_port", 8000))
    groups = group_tasks_by_model(tasks)
    for idx, group in enumerate(groups):
        cmd = build_server_command(group.model, host, base_port + idx, group.server_case)
        print(
            f"  {group.id}: model={group.model['id']} type={group.model.get('type')} "
            f"required_gpus={group.required_gpus} client_tasks={len(group.tasks)}"
        )
        print(f"    server_cmd={command_to_text(cmd)}")
        for task in group.tasks:
            print(f"    client={task.id} kind={task.kind} case={task.case['id']}")


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
    groups = group_tasks_by_model(tasks)
    log("run_start", task_count=len(tasks), server_group_count=len(groups), result_root=result_root)
    scheduler = GpuScheduler(
        base_port=int(global_config.get("base_port", 8000)),
        poll_interval_sec=int(global_config.get("poll_interval_sec", 30)),
        configured_gpu_indices=global_config.get("gpu_indices", []),
    )

    async def launch(group: TaskGroup) -> None:
        log(
            "server_group_queued",
            group.id,
            model=group.model["id"],
            required_gpus=group.required_gpus,
            client_tasks=len(group.tasks),
        )
        try:
            await run_scheduled(
                scheduler,
                group.required_gpus,
                lambda allocation: run_task_group_with_server(
                    group,
                    allocation,
                    global_config,
                    result_root,
                    update_baseline,
                ),
                group.id,
            )
            log("server_group_done", group.id, client_tasks=len(group.tasks))
        except Exception as exc:
            log("server_group_failed", group.id, error=repr(exc))
            raise

    await asyncio.gather(*(launch(group) for group in groups))
    log("run_done", task_count=len(tasks), server_group_count=len(groups), result_root=result_root)


async def run_task_group_with_server(
    group: TaskGroup,
    allocation: Allocation,
    global_config: dict,
    result_root: Path,
    update_baseline: bool,
) -> None:
    host = global_config.get("host", "127.0.0.1")
    run_dir = ensure_dir(result_root / "runs" / group.id)
    server = VllmServer(
        group.model,
        group.server_case,
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
        "server_group_id": group.id,
        "client_task_ids": [task.id for task in group.tasks],
        "model": group.model,
        "server_case": group.server_case,
        "gpus": allocation.gpus,
        "port": allocation.port,
        "server_command": server_command,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log("manifest_written", group.id, path=run_dir / "manifest.json")
    log("server_command", group.id, command=server_command)

    try:
        log("server_group_start", group.id, run_dir=run_dir, client_tasks=len(group.tasks))
        await server.start()
        for task in group.tasks:
            await run_task_client(
                task,
                group,
                host,
                allocation,
                result_root,
                global_config,
                update_baseline,
                server_command,
            )
    finally:
        await server.stop()


async def run_task_client(
    task: Task,
    group: TaskGroup,
    host: str,
    allocation: Allocation,
    result_root: Path,
    global_config: dict,
    update_baseline: bool,
    server_command: str,
) -> None:
    run_dir = ensure_dir(result_root / "runs" / task.id)
    manifest = {
        "task_id": task.id,
        "server_group_id": group.id,
        "kind": task.kind,
        "model": task.model,
        "case": task.case,
        "gpus": allocation.gpus,
        "port": allocation.port,
        "server_command": server_command,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log("manifest_written", task.id, path=run_dir / "manifest.json")
    log("task_start", task.id, kind=task.kind, run_dir=run_dir, server_group=group.id)
    if task.kind == "perf":
        await run_perf_task(task, host, allocation.port, allocation.gpus, result_root, global_config)
    elif task.kind == "stability":
        await run_stability_task(task, host, allocation.port, allocation.gpus, result_root, update_baseline)
    else:
        raise ValueError(f"Unsupported task kind: {task.kind}")
    log("task_done", task.id, server_group=group.id)


def group_tasks_by_model(tasks: list[Task]) -> list[TaskGroup]:
    grouped: dict[str, list[Task]] = {}
    for task in tasks:
        grouped.setdefault(task.model["id"], []).append(task)

    return [
        TaskGroup(
            id=f"server_{model_id}",
            model=items[0].model,
            tasks=items,
            required_gpus=items[0].required_gpus,
            server_case=build_group_server_case(model_id, items),
        )
        for model_id, items in grouped.items()
    ]


def build_group_server_case(model_id: str, tasks: list[Task]) -> dict[str, Any]:
    return {
        "id": f"server_{model_id}",
        "_task_id": f"server_{model_id}",
        "server_args": merge_case_server_args(tasks),
    }


def merge_case_server_args(tasks: list[Task]) -> dict[str, Any] | list[Any]:
    merged: dict[str, Any] = {}
    shared_list: list[Any] | None = None
    for task in tasks:
        server_args = task.case.get("server_args", {})
        if not server_args:
            continue
        if isinstance(server_args, dict):
            if shared_list is not None:
                raise ValueError("Cannot mix list and dict case server_args in one server group")
            for key, value in server_args.items():
                if key not in merged:
                    merged[key] = value
                elif isinstance(value, (int, float)) and isinstance(merged[key], (int, float)):
                    merged[key] = max(merged[key], value)
                elif merged[key] != value:
                    raise ValueError(
                        f"Conflicting case server_args for '{key}' in server group: "
                        f"{merged[key]!r} vs {value!r}"
                    )
        elif isinstance(server_args, (list, tuple)):
            if merged:
                raise ValueError("Cannot mix dict and list case server_args in one server group")
            current = list(server_args)
            if shared_list is None:
                shared_list = current
            elif shared_list != current:
                raise ValueError("Conflicting list case server_args in one server group")
        else:
            raise TypeError(
                f"case server_args must be a dict/list/tuple, got: {type(server_args).__name__}"
            )
    if shared_list is not None:
        return shared_list
    return merged
