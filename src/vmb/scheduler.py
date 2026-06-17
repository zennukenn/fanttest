from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from .hardware import idle_gpu_indices


@dataclass(frozen=True)
class Allocation:
    gpus: list[int]
    port: int


class GpuScheduler:
    def __init__(
        self,
        base_port: int,
        poll_interval_sec: int,
        idle_memory_threshold_mb: int,
    ) -> None:
        self.base_port = base_port
        self.poll_interval_sec = poll_interval_sec
        self.idle_memory_threshold_mb = idle_memory_threshold_mb
        self._reserved: set[int] = set()
        self._ports_in_use: set[int] = set()
        self._lock = asyncio.Lock()

    async def acquire(self, required_gpus: int) -> Allocation:
        while True:
            async with self._lock:
                idle = idle_gpu_indices(self.idle_memory_threshold_mb)
                candidates = [gpu for gpu in idle if gpu not in self._reserved]
                if len(candidates) >= required_gpus:
                    selected = candidates[:required_gpus]
                    self._reserved.update(selected)
                    port = self._next_port()
                    self._ports_in_use.add(port)
                    return Allocation(selected, port)
            await asyncio.sleep(self.poll_interval_sec)

    async def release(self, allocation: Allocation) -> None:
        async with self._lock:
            for gpu in allocation.gpus:
                self._reserved.discard(gpu)
            self._ports_in_use.discard(allocation.port)

    def _next_port(self) -> int:
        port = self.base_port
        while port in self._ports_in_use:
            port += 1
        return port


async def run_scheduled(
    scheduler: GpuScheduler,
    required_gpus: int,
    worker: Callable[[Allocation], Awaitable[None]],
) -> None:
    allocation = await scheduler.acquire(required_gpus)
    try:
        await worker(allocation)
    finally:
        await scheduler.release(allocation)
