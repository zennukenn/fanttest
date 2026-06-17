from __future__ import annotations

import asyncio
import difflib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import Task
from .util import ensure_dir, normalize_text, short_hash, stable_json


async def run_stability_task(
    task: Task,
    host: str,
    port: int,
    gpus: list[int],
    result_root: Path,
    update_baseline: bool,
) -> None:
    output = await asyncio.to_thread(request_output, task, host, port)
    model_sig = model_signature(task.model)
    case_sig = case_signature(task.case)

    base_dir = ensure_dir(result_root / "stability" / "baselines" / task.model["id"])
    report_dir = ensure_dir(result_root / "stability" / "reports" / task.model["id"])
    current_dir = ensure_dir(result_root / "stability" / "current" / task.model["id"])

    baseline_path = base_dir / f"{task.case['id']}_{model_sig}_{case_sig}.txt"
    current_path = current_dir / f"{task.case['id']}_{model_sig}_{case_sig}.txt"
    report_path = report_dir / f"{task.id}.json"
    diff_path = report_dir / f"{task.id}.diff"

    current_path.write_text(output, encoding="utf-8")

    normalized_output = normalize_text(output)
    report: dict[str, Any] = {
        "task_id": task.id,
        "model_id": task.model["id"],
        "case_id": task.case["id"],
        "gpus": gpus,
        "baseline_path": str(baseline_path),
        "current_path": str(current_path),
        "current_hash": short_hash(normalized_output, 32),
        "updated_baseline": False,
        "changed": False,
    }

    if update_baseline or not baseline_path.exists():
        baseline_path.write_text(output, encoding="utf-8")
        report["updated_baseline"] = True
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    baseline = baseline_path.read_text(encoding="utf-8")
    normalized_baseline = normalize_text(baseline)
    changed = normalized_baseline != normalized_output
    report["baseline_hash"] = short_hash(normalized_baseline, 32)
    report["changed"] = changed

    if changed:
        diff = difflib.unified_diff(
            normalized_baseline.splitlines(keepends=True),
            normalized_output.splitlines(keepends=True),
            fromfile=str(baseline_path),
            tofile=str(current_path),
        )
        diff_path.write_text("".join(diff), encoding="utf-8")
        report["diff_path"] = str(diff_path)

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def request_output(task: Task, host: str, port: int) -> str:
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

    return extract_text(payload)


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


def model_signature(model: dict[str, Any]) -> str:
    keys = ["type", "tp", "dp", "pp", "enable_ep", "dtype", "mtp", "server_args"]
    return short_hash({key: model.get(key) for key in keys})


def case_signature(case: dict[str, Any]) -> str:
    return short_hash({"sampling": case.get("sampling", {}), "compare": case.get("compare", {})})
