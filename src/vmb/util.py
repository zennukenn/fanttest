from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def short_hash(value: Any, length: int = 12) -> str:
    text = value if isinstance(value, str) else stable_json(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def sanitize_id(value: str) -> str:
    value = value.strip().replace("/", "_").replace(":", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "item"


def kebab(name: str) -> str:
    return name.replace("_", "-")


def normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().replace("\r\n", "\n").split("\n")]
    return "\n".join(lines)


def append_cli_value(cmd: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    cmd.extend([flag, str(value)])
