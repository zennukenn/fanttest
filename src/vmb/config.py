from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import sanitize_id, short_hash


@dataclass(frozen=True)
class Task:
    kind: str
    model: dict[str, Any]
    case: dict[str, Any]
    id: str
    required_gpus: int


@dataclass(frozen=True)
class BenchConfig:
    path: Path
    global_config: dict[str, Any]
    models: list[dict[str, Any]]
    perf_cases: list[dict[str, Any]]
    stability_cases: list[dict[str, Any]]


def load_config(path: str | Path) -> BenchConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("vmb_user_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config module: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    global_config = dict(getattr(module, "GLOBAL", {}))
    models = [normalize_model(m) for m in getattr(module, "MODELS", [])]
    perf_cases = [normalize_case(c, "perf") for c in getattr(module, "PERF_CASES", [])]
    stability_cases = [
        normalize_case(c, "stability") for c in getattr(module, "STABILITY_CASES", [])
    ]
    return BenchConfig(config_path, global_config, models, perf_cases, stability_cases)


def normalize_model(model: dict[str, Any]) -> dict[str, Any]:
    item = dict(model)
    item.setdefault("id", sanitize_id(str(item.get("model", "model"))))
    item["id"] = sanitize_id(str(item["id"]))
    item.setdefault("type", "dense")
    item.setdefault("tp", 1)
    item.setdefault("dp", 1)
    item.setdefault("pp", 1)
    item.setdefault("enable_ep", False)
    item.setdefault("server_args", {})
    item.setdefault("extra_server_args", [])
    item.setdefault("bench", {})
    if "model" not in item:
        raise ValueError(f"Model entry {item['id']} is missing required key 'model'")
    return item


def normalize_case(case: dict[str, Any], prefix: str) -> dict[str, Any]:
    item = dict(case)
    item.setdefault("id", f"{prefix}_{short_hash(item)}")
    item["id"] = sanitize_id(str(item["id"]))
    return item


def required_gpus(model: dict[str, Any]) -> int:
    tp = int(model.get("tp", 1) or 1)
    dp = int(model.get("dp", 1) or 1)
    pp = int(model.get("pp", 1) or 1)
    return max(1, tp * dp * pp)


def expand_tasks(
    config: BenchConfig,
    kind: str,
    model_filter: str | None = None,
) -> list[Task]:
    tasks: list[Task] = []
    selected = []
    for model in config.models:
        if model_filter and model_filter not in {model["id"], model["model"], model.get("served_model_name")}:
            continue
        selected.append(model)

    if kind in {"perf", "all"}:
        for model in selected:
            cases = model.get("perf_cases") or config.perf_cases
            for case in cases:
                task_id = sanitize_id(f"perf_{model['id']}_{case['id']}")
                tasks.append(Task("perf", model, case, task_id, required_gpus(model)))

    if kind in {"stability", "all"}:
        for model in selected:
            cases = model.get("stability_cases") or config.stability_cases
            for case in cases:
                task_id = sanitize_id(f"stability_{model['id']}_{case['id']}")
                tasks.append(Task("stability", model, case, task_id, required_gpus(model)))

    return tasks
