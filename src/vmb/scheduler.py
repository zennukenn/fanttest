from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable

from .hardware import configured_gpu_indices as normalize_gpu_indices
from .progress import log


@dataclass(frozen=True)
class Allocation:
    gpus: list[int]
    port: int


class GpuScheduler:
    def __init__(
        self,
        base_port: int,
        poll_interval_sec: int,
        configured_gpu_indices: Iterable[int] | None = None,
    ) -> None:
        self.base_port = base_port
        self.poll_interval_sec = poll_interval_sec
        self.configured_gpu_indices = normalize_gpu_indices(configured_gpu_indices)
        self._reserved: set[int] = set()
        self._ports_in_use: set[int] = set()
        self._lock = asyncio.Lock()

    async def acquire(self, required_gpus: int, task_id: str | None = None) -> Allocation:
        log("waiting_gpu", task_id, required_gpus=required_gpus)
        while True:
            async with self._lock:
                idle = self.configured_gpu_indices
                candidates = [gpu for gpu in idle if gpu not in self._reserved]
                log(
                    "gpu_snapshot",
                    task_id,
                    required_gpus=required_gpus,
                    allocation_mode="static",
                    configured_gpus=",".join(str(gpu) for gpu in idle) or "none",
                    idle_gpus=",".join(str(gpu) for gpu in idle) or "none",
                    reserved_gpus=",".join(str(gpu) for gpu in sorted(self._reserved)) or "none",
                    available_gpus=",".join(str(gpu) for gpu in candidates) or "none",
                )
                if len(candidates) >= required_gpus:
                    selected = candidates[:required_gpus]
                    self._reserved.update(selected)
                    port = self._next_port()
                    self._ports_in_use.add(port)
                    log(
                        "gpu_allocated",
                        task_id,
                        gpus=",".join(str(gpu) for gpu in selected),
                        port=port,
                    )
                    return Allocation(selected, port)
                log(
                    "gpu_wait",
                    task_id,
                    required_gpus=required_gpus,
                    available_count=len(candidates),
                    sleep_sec=self.poll_interval_sec,
                )
            await asyncio.sleep(self.poll_interval_sec)

    async def release(self, allocation: Allocation, task_id: str | None = None) -> None:
        async with self._lock:
            for gpu in allocation.gpus:
                self._reserved.discard(gpu)
            self._ports_in_use.discard(allocation.port)
            log(
                "gpu_released",
                task_id,
                gpus=",".join(str(gpu) for gpu in allocation.gpus),
                port=allocation.port,
            )

    def _next_port(self) -> int:
        port = self.base_port
        while port in self._ports_in_use:
            port += 1
        return port


async def run_scheduled(
    scheduler: GpuScheduler,
    required_gpus: int,
    worker: Callable[[Allocation], Awaitable[None]],
    task_id: str | None = None,
) -> None:
    allocation = await scheduler.acquire(required_gpus, task_id)
    try:
        await worker(allocation)
    finally:
        await scheduler.release(allocation, task_id)
