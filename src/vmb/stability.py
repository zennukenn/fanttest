from __future__ import annotations

import asyncio
import difflib
import json
import platform
import subprocess
from datetime import datetime, timezone
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import Task
from .util import ensure_dir, normalize_text, short_hash


async def run_stability_task(
    task: Task,
    host: str,
    port: int,
    gpus: list[int],
    result_root: Path,
    update_baseline: bool,
) -> None:
    completion = await asyncio.to_thread(request_completion, task, host, port)
    model_sig = model_signature(task.model)
    case_sig = case_signature(task.case)

    base_dir = ensure_dir(result_root / "stability" / "baselines" / task.model["id"])
    report_dir = ensure_dir(result_root / "stability" / "reports" / task.model["id"])
    current_dir = ensure_dir(result_root / "stability" / "current" / task.model["id"])

    baseline_path = base_dir / f"{task.case['id']}_{model_sig}_{case_sig}.json"
    legacy_baseline_path = base_dir / f"{task.case['id']}_{model_sig}_{case_sig}.txt"
    current_path = current_dir / f"{task.case['id']}_{model_sig}_{case_sig}.json"
    report_path = report_dir / f"{task.id}.json"
    diff_path = report_dir / f"{task.id}.diff"

    current_record = build_record(task, gpus, completion, model_sig, case_sig)
    write_json(current_path, current_record)

    report: dict[str, Any] = {
        "schema_version": 2,
        "task_id": task.id,
        "model_id": task.model["id"],
        "case_id": task.case["id"],
        "compare_mode": task.case.get("compare", {}).get("mode", "exact_normalized"),
        "gpus": gpus,
        "baseline_path": str(baseline_path),
        "current_path": str(current_path),
        "current_hash": current_record["output_hash"],
        "updated_baseline": False,
        "changed": False,
    }

    if update_baseline or (
        not baseline_path.exists() and not legacy_baseline_path.exists()
    ):
        write_json(baseline_path, current_record)
        report["updated_baseline"] = True
        write_json(report_path, report)
        return

    baseline_record = read_baseline(baseline_path, legacy_baseline_path)
    changed = has_changed(task.case, baseline_record, current_record)
    report["baseline_hash"] = baseline_record["output_hash"]
    report["changed"] = changed

    if changed:
        diff = difflib.unified_diff(
            baseline_record["normalized_output"].splitlines(keepends=True),
            current_record["normalized_output"].splitlines(keepends=True),
            fromfile=str(baseline_path),
            tofile=str(current_path),
        )
        diff_path.write_text("".join(diff), encoding="utf-8")
        report["diff_path"] = str(diff_path)

    write_json(report_path, report)


def request_completion(task: Task, host: str, port: int) -> dict[str, Any]:
    model = task.model
    case = task.case
    endpoint = case.get("endpoint") or model.get("bench", {}).get("endpoint")
    if endpoint is None:
        endpoint = "/v1/chat/completions" if model.get("type") in {"vl", "vl_moe"} else "/v1/completions"

    body = build_request_body(task, endpoint)
    url = f"http://{host}:{port}{endpoint}"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=int(case.get("timeout_sec", 600))) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    return {
        "endpoint": endpoint,
        "url": url,
        "request": body,
        "response": payload,
        "output": extract_text(payload),
    }


def build_request_body(task: Task, endpoint: str) -> dict[str, Any]:
    case = task.case
    body: dict[str, Any] = {
        "model": task.model.get("served_model_name") or task.model["model"],
    }
    body.update(case.get("sampling", {}))

    if "messages" in case or endpoint.endswith("/chat/completions"):
        body["messages"] = case.get("messages") or [{"role": "user", "content": case["prompt"]}]
    else:
        body["prompt"] = case["prompt"]

    body.update(case.get("extra_body", {}))
    return body


def extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    first = choices[0]
    if "text" in first:
        return first["text"]
    message = first.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=2)


def build_record(
    task: Task,
    gpus: list[int],
    completion: dict[str, Any],
    model_sig: str,
    case_sig: str,
) -> dict[str, Any]:
    raw_output = completion["output"]
    normalized_output = normalize_text(raw_output)
    request_body = completion["request"]
    response_body = completion["response"]
    return {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_id": task.id,
        "model_id": task.model["id"],
        "model": task.model["model"],
        "served_model_name": task.model.get("served_model_name") or task.model["model"],
        "model_type": task.model.get("type"),
        "model_config_hash": model_sig,
        "case_id": task.case["id"],
        "case_config_hash": case_sig,
        "request_hash": short_hash(request_body, 32),
        "output_hash": short_hash(normalized_output, 32),
        "gpus": gpus,
        "endpoint": completion["endpoint"],
        "url": completion["url"],
        "request": request_body,
        "response": response_body,
        "raw_output": raw_output,
        "normalized_output": normalized_output,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "vllm_version": vllm_version(),
        },
    }


def read_baseline(baseline_path: Path, legacy_baseline_path: Path) -> dict[str, Any]:
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("raw_output", data.get("normalized_output", ""))
        data.setdefault("normalized_output", normalize_text(data["raw_output"]))
        data.setdefault("output_hash", short_hash(data["normalized_output"], 32))
        return data

    if legacy_baseline_path.exists():
        raw_output = legacy_baseline_path.read_text(encoding="utf-8")
        normalized_output = normalize_text(raw_output)
        return {
            "schema_version": 1,
            "raw_output": raw_output,
            "normalized_output": normalized_output,
            "output_hash": short_hash(normalized_output, 32),
        }

    raise FileNotFoundError(f"Baseline not found: {baseline_path}")


def has_changed(
    case: dict[str, Any],
    baseline_record: dict[str, Any],
    current_record: dict[str, Any],
) -> bool:
    mode = case.get("compare", {}).get("mode", "exact_normalized")
    if mode == "exact_raw":
        return baseline_record.get("raw_output", "") != current_record["raw_output"]
    if mode in {"exact_normalized", "hash"}:
        return baseline_record["output_hash"] != current_record["output_hash"]
    raise ValueError(f"Unsupported stability compare mode: {mode}")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def vllm_version() -> str:
    try:
        output = subprocess.check_output(
            ["vllm", "--version"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"
    return output.strip()


def model_signature(model: dict[str, Any]) -> str:
    keys = ["type", "tp", "dp", "pp", "enable_ep", "dtype", "mtp", "server_args"]
    return short_hash({key: model.get(key) for key in keys})


def case_signature(case: dict[str, Any]) -> str:
    return short_hash({"sampling": case.get("sampling", {}), "compare": case.get("compare", {})})
