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
    models = ensure_unique_model_ids(
        [normalize_model(m) for m in getattr(module, "MODELS", [])]
    )
    perf_cases = [
        normalize_case(c, "perf", global_config)
        for c in getattr(module, "PERF_CASES", [])
    ]
    stability_cases = [
        normalize_case(c, "stability", global_config)
        for c in getattr(module, "STABILITY_CASES", [])
    ]
    return BenchConfig(config_path, global_config, models, perf_cases, stability_cases)


def normalize_model(model: dict[str, Any]) -> dict[str, Any]:
    item = dict(model)
    if "model" not in item:
        raise ValueError("Model entry is missing required key 'model'")
    item.setdefault("id", default_model_id(str(item["model"])))
    item["id"] = sanitize_id(str(item["id"]))
    item.setdefault("type", "dense")
    item.setdefault("tp", 1)
    item.setdefault("dp", 1)
    item.setdefault("pp", 1)
    item.setdefault("enable_ep", False)
    item.setdefault("server_args", {})
    item.setdefault("bench", {})
    return item


def default_model_id(model: str) -> str:
    name = Path(model.rstrip("/")).name or model
    return sanitize_id(name.lower().replace("-", "_").replace(".", "_"))


def ensure_unique_model_ids(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    out = []
    for model in models:
        item = dict(model)
        base_id = item["id"]
        count = seen.get(base_id, 0) + 1
        seen[base_id] = count
        if count > 1:
            item["id"] = f"{base_id}_{count}"
        out.append(item)
    return out


def normalize_case(
    case: Any,
    prefix: str,
    global_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if prefix == "perf":
        item = normalize_perf_case(case, global_config or {})
    else:
        item = normalize_stability_case(case, global_config or {})
    item.setdefault("id", default_case_id(item, prefix))
    item["id"] = sanitize_id(str(item["id"]))
    return item


def normalize_stability_case(case: Any, global_config: dict[str, Any]) -> dict[str, Any]:
    if isinstance(case, dict):
        item = dict(case)
    elif isinstance(case, (list, tuple)):
        if len(case) == 2:
            item = {"prompt": case[0], "expected": case[1]}
        elif len(case) == 3:
            item = {"image": case[0], "prompt": case[1], "expected": case[2]}
        else:
            raise ValueError(
                "Stability case list/tuple must be [prompt, expected] "
                "or [image, prompt, expected]"
            )
    else:
        raise TypeError(f"stability case must be a dict/list/tuple, got: {type(case).__name__}")

    default_sampling = dict(
        global_config.get(
            "default_stability_sampling",
            {"temperature": 0, "top_p": 1, "max_tokens": 128, "seed": 42},
        )
    )
    default_compare = dict(
        global_config.get("default_stability_compare", {"mode": "exact_normalized"})
    )
    sampling = dict(default_sampling)
    sampling.update(item.get("sampling", {}))
    compare = dict(default_compare)
    compare.update(item.get("compare", {}))
    item["sampling"] = sampling
    item["compare"] = compare

    if "image" in item and "messages" not in item:
        item.setdefault("endpoint", "/v1/chat/completions")
        item["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": item["prompt"]},
                    {"type": "image_url", "image_url": {"url": f"file://{item['image']}"}},
                ],
            }
        ]
    return item


def normalize_perf_case(case: Any, global_config: dict[str, Any]) -> dict[str, Any]:
    multiplier = int(global_config.get("default_num_prompts_multiplier", 2))
    if isinstance(case, dict):
        item = dict(case)
    elif isinstance(case, (list, tuple)):
        if len(case) not in {3, 4, 5}:
            raise ValueError(
                "Perf case list/tuple must be "
                "[max_concurrency, input_len, output_len], "
                "[max_concurrency, input_len, output_len, num_prompts], or "
                "[max_concurrency, input_len, output_len, num_prompts, temperature]"
            )
        item = {
            "max_concurrency": case[0],
            "input_len": case[1],
            "output_len": case[2],
            "num_prompts": case[3] if len(case) >= 4 else int(case[0]) * multiplier,
            "temperature": case[4] if len(case) >= 5 else 0,
            "server_args": {
                "max_num_seqs": case[0],
            },
        }
    else:
        raise TypeError(f"perf case must be a dict/list/tuple, got: {type(case).__name__}")

    if "num_prompts" not in item and "max_concurrency" in item:
        item["num_prompts"] = int(item["max_concurrency"]) * multiplier
    item.setdefault("temperature", 0)
    if "max_concurrency" in item:
        server_args = item.get("server_args", {})
        if isinstance(server_args, dict):
            server_args.setdefault("max_num_seqs", item["max_concurrency"])
            item["server_args"] = server_args
    return item


def default_case_id(item: dict[str, Any], prefix: str) -> str:
    if prefix == "perf" and {"max_concurrency", "input_len", "output_len"} <= set(item):
        return f"seq{item['max_concurrency']}_i{item['input_len']}_o{item['output_len']}"
    if prefix == "stability":
        if "image" in item:
            return f"vl_{short_hash({'image': item.get('image'), 'prompt': item.get('prompt'), 'expected': item.get('expected')})}"
        if "prompt" in item:
            return f"qa_{short_hash({'prompt': item.get('prompt'), 'expected': item.get('expected')})}"
    return f"{prefix}_{short_hash(item)}"


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
            cases = [
                normalize_case(c, "perf", config.global_config)
                for c in (model.get("perf_cases") or config.perf_cases)
            ]
            for case in cases:
                task_id = sanitize_id(f"perf_{model['id']}_{case['id']}")
                tasks.append(Task("perf", model, case, task_id, required_gpus(model)))

    if kind in {"stability", "all"}:
        for model in selected:
            cases = [
                normalize_case(c, "stability", config.global_config)
                for c in (model.get("stability_cases") or config.stability_cases)
            ]
            for case in cases:
                task_id = sanitize_id(f"stability_{model['id']}_{case['id']}")
                tasks.append(Task("stability", model, case, task_id, required_gpus(model)))

    return tasks
