from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def log(event: str, task_id: str | None = None, **fields: Any) -> None:
    parts = [datetime.now(timezone.utc).isoformat(timespec="seconds"), event]
    if task_id:
        parts.append(f"task={task_id}")
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    print(" | ".join(parts), flush=True)
