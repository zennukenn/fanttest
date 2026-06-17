from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .args import build_server_command
from .progress import log
from .util import ensure_dir


class VllmServer:
    def __init__(
        self,
        model: dict,
        case: dict,
        host: str,
        port: int,
        gpu_indices: list[int],
        log_dir: Path,
        ready_timeout_sec: int,
        visible_devices_env: str = "CUDA_VISIBLE_DEVICES",
    ) -> None:
        self.model = model
        self.case = case
        self.host = host
        self.port = port
        self.gpu_indices = gpu_indices
        self.log_dir = ensure_dir(log_dir)
        self.ready_timeout_sec = ready_timeout_sec
        self.visible_devices_env = visible_devices_env
        self.process: subprocess.Popen | None = None
        self.stdout_file = None
        self.stderr_file = None
        self.task_id = str(case.get("_task_id", ""))

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def command(self) -> list[str]:
        return build_server_command(self.model, self.host, self.port, self.case)

    async def start(self) -> None:
        env = os.environ.copy()
        env[self.visible_devices_env] = ",".join(str(i) for i in self.gpu_indices)
        env.setdefault("VLLM_NO_USAGE_STATS", "1")

        log(
            "server_starting",
            self.task_id,
            model=self.model.get("id"),
            port=self.port,
            gpus=env[self.visible_devices_env],
            visible_devices_env=self.visible_devices_env,
            stdout=self.log_dir / "server.stdout.log",
            stderr=self.log_dir / "server.stderr.log",
        )
        self.stdout_file = open(self.log_dir / "server.stdout.log", "w", encoding="utf-8")
        self.stderr_file = open(self.log_dir / "server.stderr.log", "w", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command(),
            stdout=self.stdout_file,
            stderr=self.stderr_file,
            env=env,
            start_new_session=True,
        )
        log("server_process_started", self.task_id, pid=self.process.pid)
        await self.wait_ready()

    async def wait_ready(self) -> None:
        deadline = time.time() + self.ready_timeout_sec
        url = f"{self.base_url}/v1/models"
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            if self.process and self.process.poll() is not None:
                raise RuntimeError(f"vLLM server exited early with code {self.process.returncode}")
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if 200 <= response.status < 300:
                        log("server_ready", self.task_id, url=url, attempts=attempt)
                        return
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == 1 or attempt % 6 == 0:
                    remaining = max(0, int(deadline - time.time()))
                    log("server_wait_ready", self.task_id, url=url, attempt=attempt, remaining_sec=remaining)
                await asyncio.sleep(5)
        raise TimeoutError(f"vLLM server did not become ready within {self.ready_timeout_sec}s")

    async def stop(self) -> None:
        if self.process and self.process.poll() is None:
            try:
                log("server_stopping", self.task_id, pid=self.process.pid)
                os.killpg(self.process.pid, signal.SIGTERM)
                await asyncio.to_thread(self.process.wait, 60)
                log("server_stopped", self.task_id, returncode=self.process.returncode)
            except Exception:
                try:
                    log("server_killing", self.task_id, pid=self.process.pid)
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if self.stdout_file:
            self.stdout_file.close()
        if self.stderr_file:
            self.stderr_file.close()
