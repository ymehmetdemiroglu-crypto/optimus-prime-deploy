"""
Automation Agent — Sandboxed Script Execution

Pluggable executor architecture:
  - DockerExecutor (primary): Ephemeral containers, no network, memory-limited.
  - LocalSubprocessExecutor (fallback): For dev/trusted scripts.
  - ScriptExecutor (facade): Tries Docker first, falls back to local.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from abc import ABC, abstractmethod
from typing import Optional

from .schemas import ExecutionResult, ExecutorType, ScriptConfig, ScriptStatus

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Harness Template
# ═══════════════════════════════════════════════════════════

# This wrapper is injected around user scripts so we can pass
# parameters in and capture structured JSON output.
_PYTHON_HARNESS = '''
import json, sys
params = json.loads("""{params_json}""")
# ── User Script ──
{user_script}
# ── End User Script ──
# If the user script defined a `result` variable, output it
if "result" in dir() or "result" in locals():
    print("__AGENT_OUTPUT__" + json.dumps(result))
'''


# ═══════════════════════════════════════════════════════════
#  Base Executor
# ═══════════════════════════════════════════════════════════

class BaseExecutor(ABC):
    """Abstract base for all script executors."""

    @abstractmethod
    async def execute(
        self,
        script: ScriptConfig,
        parameters: dict,
        timeout: int,
    ) -> ExecutionResult:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...


# ═══════════════════════════════════════════════════════════
#  Docker Executor
# ═══════════════════════════════════════════════════════════

class DockerExecutor(BaseExecutor):
    """
    Runs scripts inside ephemeral Docker containers.

    Security flags:
      - network_disabled=True (no outbound requests)
      - mem_limit="128m"
      - read_only=True
      - tmpfs /tmp with 10m cap
      - auto_remove=True
    """

    def __init__(self, image: str = "python:3.11-slim"):
        self.image = image
        self._client = None

    def _get_client(self):
        if self._client is None:
            import docker
            self._client = docker.from_env()
        return self._client

    def is_available(self) -> bool:
        try:
            client = self._get_client()
            client.ping()
            return True
        except Exception:
            return False

    async def execute(
        self,
        script: ScriptConfig,
        parameters: dict,
        timeout: int,
    ) -> ExecutionResult:
        result = ExecutionResult(
            script_id=script.script_id,
            executor_used=ExecutorType.DOCKER,
            started_at=None,
        )

        params_json = json.dumps(parameters).replace('"""', r'\"\"\"')
        harness = _PYTHON_HARNESS.format(
            params_json=params_json,
            user_script=script.script_body,
        )

        def _run_container():
            client = self._get_client()
            start = time.monotonic()
            try:
                container_output = client.containers.run(
                    image=self.image,
                    command=["python", "-c", harness],
                    network_disabled=True,
                    mem_limit="128m",
                    read_only=True,
                    tmpfs={"/tmp": "size=10m"},
                    auto_remove=True,
                    stdout=True,
                    stderr=True,
                    timeout=timeout,
                )
                elapsed = (time.monotonic() - start) * 1000
                raw_out = container_output.decode("utf-8", errors="replace") if isinstance(container_output, bytes) else str(container_output)
                return raw_out, "", 0, elapsed
            except Exception as e:
                elapsed = (time.monotonic() - start) * 1000
                return "", str(e), 1, elapsed

        from datetime import datetime
        result.started_at = datetime.utcnow()

        try:
            stdout, stderr, exit_code, duration = await asyncio.to_thread(_run_container)
        except asyncio.TimeoutError:
            result.status = ScriptStatus.TIMEOUT
            result.error_message = f"Execution timed out after {timeout}s"
            result.completed_at = datetime.utcnow()
            return result

        result.raw_stdout = stdout
        result.raw_stderr = stderr
        result.exit_code = exit_code
        result.duration_ms = duration
        result.completed_at = datetime.utcnow()

        # Parse structured output
        if "__AGENT_OUTPUT__" in stdout:
            try:
                json_part = stdout.split("__AGENT_OUTPUT__", 1)[1].strip().split("\n")[0]
                result.output = json.loads(json_part)
            except (json.JSONDecodeError, IndexError):
                pass

        result.status = ScriptStatus.SUCCESS if exit_code == 0 else ScriptStatus.FAILED
        if exit_code != 0:
            result.error_message = stderr or f"Script exited with code {exit_code}"

        return result


# ═══════════════════════════════════════════════════════════
#  Local Subprocess Executor
# ═══════════════════════════════════════════════════════════

class LocalSubprocessExecutor(BaseExecutor):
    """
    Runs scripts via asyncio subprocess.
    Suitable for trusted scripts or when Docker is unavailable.
    """

    def is_available(self) -> bool:
        return True  # Always available

    async def execute(
        self,
        script: ScriptConfig,
        parameters: dict,
        timeout: int,
    ) -> ExecutionResult:
        from datetime import datetime

        result = ExecutionResult(
            script_id=script.script_id,
            executor_used=ExecutorType.LOCAL,
            started_at=datetime.utcnow(),
        )

        params_json = json.dumps(parameters).replace('"""', r'\"\"\"')
        harness = _PYTHON_HARNESS.format(
            params_json=params_json,
            user_script=script.script_body,
        )

        # Write to temp file
        tmp_dir = tempfile.mkdtemp(prefix="agent_sandbox_")
        script_path = os.path.join(tmp_dir, "script.py")
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(harness)

            start = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                "python", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmp_dir,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                result.status = ScriptStatus.TIMEOUT
                result.error_message = f"Execution timed out after {timeout}s"
                result.duration_ms = (time.monotonic() - start) * 1000
                result.completed_at = datetime.utcnow()
                return result

            elapsed = (time.monotonic() - start) * 1000
            result.raw_stdout = stdout_bytes.decode("utf-8", errors="replace")
            result.raw_stderr = stderr_bytes.decode("utf-8", errors="replace")
            result.exit_code = proc.returncode
            result.duration_ms = elapsed
            result.completed_at = datetime.utcnow()

            # Parse structured output
            if "__AGENT_OUTPUT__" in result.raw_stdout:
                try:
                    json_part = result.raw_stdout.split("__AGENT_OUTPUT__", 1)[1].strip().split("\n")[0]
                    result.output = json.loads(json_part)
                except (json.JSONDecodeError, IndexError):
                    pass

            result.status = ScriptStatus.SUCCESS if proc.returncode == 0 else ScriptStatus.FAILED
            if proc.returncode != 0:
                result.error_message = result.raw_stderr or f"Script exited with code {proc.returncode}"

        finally:
            # Clean up temp files
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return result


# ═══════════════════════════════════════════════════════════
#  Script Executor Facade
# ═══════════════════════════════════════════════════════════

class ScriptExecutor:
    """
    Facade that tries Docker first, falls back to Local.
    Supports force_executor override.
    """

    def __init__(self):
        self.docker = DockerExecutor()
        self.local = LocalSubprocessExecutor()

    async def execute(
        self,
        script: ScriptConfig,
        parameters: dict,
        force_executor: Optional[ExecutorType] = None,
    ) -> ExecutionResult:
        timeout = script.timeout_seconds

        if force_executor == ExecutorType.DOCKER:
            return await self.docker.execute(script, parameters, timeout)
        elif force_executor == ExecutorType.LOCAL:
            return await self.local.execute(script, parameters, timeout)

        # Default: try Docker, fall back to Local
        if self.docker.is_available():
            logger.info(f"[Sandbox] Using DockerExecutor for {script.script_id}")
            return await self.docker.execute(script, parameters, timeout)
        else:
            logger.warning(f"[Sandbox] Docker unavailable, falling back to LocalSubprocessExecutor for {script.script_id}")
            return await self.local.execute(script, parameters, timeout)
