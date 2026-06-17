from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from io import StringIO


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    name: str
    memory_total_mb: int
    memory_used_mb: int


def _run_nvidia_smi(args: list[str]) -> str:
    return subprocess.check_output(["nvidia-smi", *args], text=True, stderr=subprocess.DEVNULL)


def list_gpus() -> list[GPU]:
    try:
        output = _run_nvidia_smi([
            "--query-gpu=index,uuid,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    gpus: list[GPU] = []
    reader = csv.reader(StringIO(output))
    for row in reader:
        if len(row) < 5:
            continue
        index, uuid, name, total, used = [cell.strip() for cell in row[:5]]
        gpus.append(GPU(int(index), uuid, name, int(total), int(used)))
    return gpus


def busy_gpu_uuids() -> set[str]:
    try:
        output = _run_nvidia_smi([
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return set()

    busy: set[str] = set()
    reader = csv.reader(StringIO(output))
    for row in reader:
        if len(row) >= 2 and row[0].strip():
            busy.add(row[0].strip())
    return busy


def idle_gpu_indices(idle_memory_threshold_mb: int = 1024) -> list[int]:
    gpus = list_gpus()
    busy = busy_gpu_uuids()
    idle = []
    for gpu in gpus:
        has_compute_process = gpu.uuid in busy
        under_memory_threshold = gpu.memory_used_mb <= idle_memory_threshold_mb
        if not has_compute_process and under_memory_threshold:
            idle.append(gpu.index)
    return idle
