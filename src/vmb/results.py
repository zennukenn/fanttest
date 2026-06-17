from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import Task
from .util import ensure_dir


PERF_FIELDS = [
    "task_id",
    "status",
    "error",
    "model_id",
    "model",
    "served_model_name",
    "model_type",
    "tp",
    "dp",
    "pp",
    "enable_ep",
    "dtype",
    "case_id",
    "input_len",
    "output_len",
    "num_prompts",
    "max_concurrency",
    "request_rate",
    "gpus",
    "raw_json_path",
    "successful_requests",
    "total_input_tokens",
    "total_generated_tokens",
    "request_throughput",
    "output_token_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
]


def append_perf_result(
    csv_path: Path,
    task: Task,
    gpus: list[int],
    raw_json_path: Path,
    status: str,
    error: str = "",
) -> None:
    ensure_dir(csv_path.parent)
    row = base_row(task, gpus, raw_json_path, status, error)
    if raw_json_path.exists():
        try:
            with open(raw_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            row.update(extract_metrics(data))
        except Exception as exc:
            row["status"] = "parse_error"
            row["error"] = str(exc)

    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PERF_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in PERF_FIELDS})


def base_row(
    task: Task,
    gpus: list[int],
    raw_json_path: Path,
    status: str,
    error: str,
) -> dict[str, Any]:
    model = task.model
    case = task.case
    return {
        "task_id": task.id,
        "status": status,
        "error": error,
        "model_id": model["id"],
        "model": model["model"],
        "served_model_name": model.get("served_model_name", ""),
        "model_type": model.get("type", ""),
        "tp": model.get("tp", 1),
        "dp": model.get("dp", 1),
        "pp": model.get("pp", 1),
        "enable_ep": model.get("enable_ep", False),
        "dtype": model.get("dtype", ""),
        "case_id": case["id"],
        "input_len": case.get("input_len", case.get("random_input_len", "")),
        "output_len": case.get("output_len", case.get("random_output_len", "")),
        "num_prompts": case.get("num_prompts", ""),
        "max_concurrency": case.get("max_concurrency", ""),
        "request_rate": case.get("request_rate", "inf"),
        "gpus": ",".join(str(gpu) for gpu in gpus),
        "raw_json_path": str(raw_json_path),
    }


def extract_metrics(data: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "successful_requests": ["successful_requests", "completed", "num_successful_requests"],
        "total_input_tokens": ["total_input_tokens", "total_prompt_tokens"],
        "total_generated_tokens": ["total_generated_tokens", "total_output_tokens"],
        "request_throughput": ["request_throughput", "requests_per_second"],
        "output_token_throughput": ["output_token_throughput", "output_throughput"],
        "total_token_throughput": ["total_token_throughput", "total_throughput"],
        "mean_ttft_ms": ["mean_ttft_ms", "mean_ttft"],
        "median_ttft_ms": ["median_ttft_ms", "median_ttft"],
        "p99_ttft_ms": ["p99_ttft_ms", "p99_ttft"],
        "mean_tpot_ms": ["mean_tpot_ms", "mean_tpot"],
        "median_tpot_ms": ["median_tpot_ms", "median_tpot"],
        "p99_tpot_ms": ["p99_tpot_ms", "p99_tpot"],
        "mean_itl_ms": ["mean_itl_ms", "mean_itl"],
        "median_itl_ms": ["median_itl_ms", "median_itl"],
        "p99_itl_ms": ["p99_itl_ms", "p99_itl"],
    }
    flat = flatten(data)
    out: dict[str, Any] = {}
    for field, names in aliases.items():
        for name in names:
            if name in flat:
                out[field] = flat[name]
                break
    return out


def flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten(value, full_key))
        elif not isinstance(value, list):
            out[str(key)] = value
            out[full_key] = value
    return out
