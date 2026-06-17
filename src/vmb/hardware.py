from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GPU:
    index: int
    name: str


def configured_gpu_indices(indices: Iterable[int] | None) -> list[int]:
    return [int(index) for index in (indices or [])]


def configured_gpus(indices: Iterable[int] | None) -> list[GPU]:
    return [
        GPU(index=index, name=f"configured-gpu-{index}")
        for index in configured_gpu_indices(indices)
    ]


def format_gpu_inventory(gpus: list[GPU]) -> str:
    if not gpus:
        return "none"
    return ";".join(f"{gpu.index}:{gpu.name}" for gpu in gpus)
