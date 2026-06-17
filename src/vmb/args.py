from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from .config import Task
from .util import append_cli_value, kebab


def build_server_command(
    model: dict[str, Any],
    host: str,
    port: int,
    case: dict[str, Any] | None = None,
) -> list[str]:
    cmd = ["vllm", "serve", str(model["model"])]
    cmd.extend(["--host", host, "--port", str(port)])

    served_model_name = model.get("served_model_name")
    if served_model_name:
        cmd.extend(["--served-model-name", str(served_model_name)])

    append_cli_value(cmd, "--tensor-parallel-size", model.get("tp", 1))
    if int(model.get("dp", 1) or 1) > 1 or model.get("enable_ep"):
        append_cli_value(cmd, "--data-parallel-size", model.get("dp", 1))
    if int(model.get("pp", 1) or 1) > 1:
        append_cli_value(cmd, "--pipeline-parallel-size", model.get("pp", 1))
    if model.get("enable_ep"):
        cmd.append("--enable-expert-parallel")

    append_cli_value(cmd, "--dtype", model.get("dtype"))

    append_server_args(cmd, model.get("server_args", {}))

    if case:
        append_server_args(cmd, case.get("server_args", {}))

    mtp = dict(model.get("mtp", {}) or {})
    if mtp.get("enabled"):
        speculative_config = mtp.get("speculative_config")
        if speculative_config is not None:
            append_cli_value(cmd, "--speculative-config", speculative_config)
        else:
            append_cli_value(cmd, "--spec-method", mtp.get("spec_method"))
            append_cli_value(cmd, "--spec-model", mtp.get("spec_model"))
            append_cli_value(cmd, "--spec-tokens", mtp.get("spec_tokens"))
        cmd.extend([str(x) for x in mtp.get("extra_args", [])])

    return cmd


def append_server_args(cmd: list[str], server_args: Any) -> None:
    if not server_args:
        return
    if isinstance(server_args, dict):
        for key, value in server_args.items():
            append_cli_value(cmd, f"--{kebab(key)}", value)
        return
    if isinstance(server_args, (list, tuple)):
        cmd.extend([str(value) for value in server_args])
        return
    raise TypeError(f"server_args must be a dict/list/tuple, got: {type(server_args).__name__}")


def build_perf_command(
    task: Task,
    host: str,
    port: int,
    result_dir: Path,
    global_config: dict[str, Any],
) -> tuple[list[str], Path]:
    model = task.model
    case = task.case
    bench_cfg = dict(model.get("bench", {}) or {})
    result_filename = f"{task.id}.json"
    result_path = result_dir / result_filename

    backend = case.get("backend") or bench_cfg.get("backend") or global_config.get("default_backend", "vllm")
    endpoint = case.get("endpoint") or bench_cfg.get("endpoint") or global_config.get("default_endpoint", "/v1/completions")
    dataset_name = case.get("dataset_name") or bench_cfg.get("dataset_name") or "random"

    cmd = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        str(backend),
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        str(model.get("served_model_name") or model["model"]),
        "--endpoint",
        str(endpoint),
        "--dataset-name",
        str(dataset_name),
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
    ]

    if global_config.get("save_detailed", True) or case.get("save_detailed"):
        cmd.append("--save-detailed")

    mapped = {
        "input_len": "--input-len",
        "output_len": "--output-len",
        "num_prompts": "--num-prompts",
        "max_concurrency": "--max-concurrency",
        "request_rate": "--request-rate",
        "num_warmups": "--num-warmups",
        "temperature": "--temperature",
        "top_p": "--top-p",
        "random_input_len": "--random-input-len",
        "random_output_len": "--random-output-len",
        "random_range_ratio": "--random-range-ratio",
    }
    for key, flag in mapped.items():
        append_cli_value(cmd, flag, case.get(key))

    metadata = {
        "task_id": task.id,
        "model_id": model["id"],
        "model_type": model.get("type"),
        "tp": model.get("tp", 1),
        "dp": model.get("dp", 1),
        "pp": model.get("pp", 1),
        "enable_ep": model.get("enable_ep", False),
        "dtype": model.get("dtype"),
        "case_id": case["id"],
    }
    for key, value in metadata.items():
        if value is not None:
            cmd.extend(["--metadata", f"{key}={value}"])

    for key, value in dict(bench_cfg.get("args", {}) or {}).items():
        append_cli_value(cmd, f"--{kebab(key)}", value)
    for key, value in dict(case.get("args", {}) or {}).items():
        append_cli_value(cmd, f"--{kebab(key)}", value)

    cmd.extend([str(x) for x in bench_cfg.get("extra_args", [])])
    cmd.extend([str(x) for x in case.get("extra_args", [])])
    return cmd, result_path


def command_to_text(cmd: list[str]) -> str:
    return json.dumps(cmd, ensure_ascii=False)


def command_to_shell(cmd: list[str]) -> str:
    return shlex.join(cmd)
