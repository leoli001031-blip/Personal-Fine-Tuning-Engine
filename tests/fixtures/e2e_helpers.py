"""End-to-end test helpers for PFE integration testing.

This module provides test infrastructure for validating the complete
`collect -> curate -> train -> eval -> promote -> serve` closed loop.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator, Optional
from uuid import uuid4

import requests

# Default test configuration
DEFAULT_TEST_PORT = 9999
DEFAULT_TEST_TIMEOUT = 30


def _resolve_project_python() -> str:
    """Return the Python executable that has PFE packages installed.

    Test fixtures that spawn PFE server/daemon subprocesses need an
    interpreter where all package dependencies (fastapi, peft, trl, etc.)
    are installed. We prefer the project venv when it exists.
    """
    project_root = Path(__file__).resolve().parents[2]
    venv_python = project_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    venv_python3 = project_root / ".venv" / "bin" / "python3"
    if venv_python3.exists():
        return str(venv_python3)
    return sys.executable


_PYTHON_EXE = _resolve_project_python()


def _cleanup_port_processes(port: int) -> None:
    """Kill any process currently listening on the given port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            try:
                pid = int(line.strip())
                os.kill(pid, signal.SIGKILL)
            except (ValueError, ProcessLookupError, PermissionError):
                pass
    except Exception:
        pass


def _cleanup_residual_daemons(workspace: str) -> None:
    """Kill any lingering PFE daemon processes for the given workspace."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if "pfe_core.worker_daemon" in line and f"--workspace {workspace}" in line:
                try:
                    pid = int(line.split()[1])
                    os.kill(pid, signal.SIGKILL)
                except (ValueError, ProcessLookupError, PermissionError):
                    pass
    except Exception:
        pass


class TimeoutError(Exception):
    """Raised when a wait operation times out."""


@dataclass
class E2ETestConfig:
    """Configuration for end-to-end tests."""

    port: int = DEFAULT_TEST_PORT
    base_url: str = field(init=False)
    test_workspace: str = "test_e2e"
    min_samples_trigger: int = 10
    auto_trigger: bool = True
    auto_promote: bool = True
    test_timeout: int = 600  # 10 minutes for training

    def __post_init__(self):
        self.base_url = f"http://localhost:{self.port}"


def wait_for_port(port: int, timeout: float = 30.0, host: str = "localhost") -> bool:
    """Wait for a port to become available.

    Args:
        port: Port number to check
        timeout: Maximum time to wait in seconds
        host: Host to connect to

    Returns:
        True if port is available, False if timeout
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            time.sleep(0.1)
    return False


def wait_for(
    condition: Callable[[], bool],
    timeout: float = 30.0,
    interval: float = 0.5,
    message: str = "Condition not met",
) -> bool:
    """Wait for a condition to become true.

    Args:
        condition: Callable that returns True when condition is met
        timeout: Maximum time to wait in seconds
        interval: Time between checks in seconds
        message: Message for timeout error

    Returns:
        True if condition was met

    Raises:
        TimeoutError: If condition is not met within timeout
    """
    start = time.time()
    while time.time() - start < timeout:
        if condition():
            return True
        time.sleep(interval)
    raise TimeoutError(f"{message} after {timeout}s")


class TestServer:
    """Test server context manager.

    Manages a PFE server process for testing.
    """

    def __init__(
        self,
        port: int = DEFAULT_TEST_PORT,
        workspace: str = "test_e2e",
        config_override: Optional[dict[str, Any]] = None,
    ):
        self.port = port
        self.workspace = workspace
        self.config_override = config_override or {}
        self.proc: Optional[subprocess.Popen] = None
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def __enter__(self) -> TestServer:
        """Start the server and wait for it to be ready."""
        _cleanup_port_processes(self.port)
        _cleanup_residual_daemons(self.workspace)
        # Create temporary config if overrides provided
        env = os.environ.copy()
        env["PFE_WORKSPACE"] = self.workspace
        env["PFE_SERVER_PORT"] = str(self.port)
        if "PFE_HOME" in os.environ:
            env["PFE_HOME"] = os.environ["PFE_HOME"]

        # Apply config overrides via environment
        for key, value in self.config_override.items():
            env[f"PFE_{key.upper()}"] = json.dumps(value) if isinstance(value, (dict, list)) else str(value)

        # Start server using the resolved project Python
        cmd = [
            _PYTHON_EXE, "-m", "pfe_server",
            "--port", str(self.port),
            "--workspace", self.workspace,
        ]

        self._log_file = tempfile.NamedTemporaryFile(mode="w", suffix=".log", prefix=f"pfe_server_{self.workspace}_", delete=False)
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )

        # Wait for server to be ready
        if not wait_for_port(self.port, timeout=DEFAULT_TEST_TIMEOUT):
            log_text = Path(self._log_file.name).read_text(encoding="utf-8", errors="ignore") if hasattr(self, "_log_file") else ""
            self._cleanup()
            raise TimeoutError(f"Server failed to bind on port {self.port}. log:\n{log_text}")

        if self.proc.poll() is not None:
            log_text = Path(self._log_file.name).read_text(encoding="utf-8", errors="ignore") if hasattr(self, "_log_file") else ""
            self._cleanup()
            raise TimeoutError(f"Server exited immediately after binding to port {self.port}. log:\n{log_text}")

        # Wait for HTTP health endpoint to be responsive
        health_ready = False
        health_start = time.time()
        while time.time() - health_start < 15.0:
            if self.health_check():
                health_ready = True
                break
            time.sleep(0.5)
        if not health_ready:
            log_text = Path(self._log_file.name).read_text(encoding="utf-8", errors="ignore") if hasattr(self, "_log_file") else ""
            self._cleanup()
            raise TimeoutError(f"Server bound to port {self.port} but health check failed. log:\n{log_text}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the server."""
        self._cleanup()

    def _cleanup(self) -> None:
        """Clean up server process."""
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
        if hasattr(self, "_log_file"):
            try:
                self._log_file.close()
                Path(self._log_file.name).unlink(missing_ok=True)
            except Exception:
                pass

    def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            response = requests.get(f"http://localhost:{self.port}/healthz", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False


class TestDaemon:
    """Test daemon context manager.

    Manages a PFE worker daemon process for testing.
    """

    def __init__(
        self,
        port: int = DEFAULT_TEST_PORT,
        workspace: str = "test_e2e",
    ):
        self.port = port
        self.workspace = workspace
        self.proc: Optional[subprocess.Popen] = None
        self.daemon_id: str = str(uuid4())[:8]

    def __enter__(self) -> TestDaemon:
        """Start the daemon."""
        _cleanup_residual_daemons(self.workspace)
        env = os.environ.copy()
        env["PFE_WORKSPACE"] = self.workspace
        env["PFE_SERVER_URL"] = f"http://localhost:{self.port}"

        cmd = [
            _PYTHON_EXE, "-m", "pfe_core.worker_daemon",
            "--workspace", self.workspace,
            "--runner-max-seconds", "120",
            "--heartbeat-interval-seconds", "30",
            "--lease-timeout-seconds", "90",
            "--disable-reliability",
        ]

        self.proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Give daemon time to start
        time.sleep(1)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the daemon."""
        self._cleanup()

    def _cleanup(self) -> None:
        """Clean up daemon process."""
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None

    def kill(self, sig: int = signal.SIGKILL) -> None:
        """Send signal to daemon process (for crash simulation)."""
        if self.proc is not None:
            self.proc.send_signal(sig)

    def is_running(self) -> bool:
        """Check if daemon process is still running."""
        if self.proc is None:
            return False
        return self.proc.poll() is None


_TEMP_CONFIG_KEY_MAP: dict[str, list[str] | list[list[str]] | None] = {
    "train_trigger.enabled": ["trainer", "trigger", "enabled"],
    "train_trigger.min_samples": ["trainer", "trigger", "min_new_samples"],
    "train_trigger.min_trigger_interval_minutes": ["trainer", "trigger", "min_trigger_interval_minutes"],
    "eval_gate.auto_trigger": [
        ["trainer", "trigger", "eval_gate_policy", "auto_trigger"],
        ["trainer", "trigger", "auto_evaluate"],
    ],
    "eval_gate.trigger_delay_seconds": ["trainer", "trigger", "eval_gate_policy", "trigger_delay_seconds"],
    "promote_gate.auto_promote": [
        ["trainer", "trigger", "promote_gate_policy", "auto_promote"],
        ["trainer", "trigger", "auto_promote"],
    ],
    "promote_gate.min_quality_score": ["trainer", "trigger", "promote_gate_policy", "min_quality_score"],
    "training.backend": ["trainer", "backend"],
    "training.epochs": ["trainer", "epochs"],
    "training.max_steps": None,
    "training.method": ["trainer", "method"],
    "training.train_type": ["trainer", "train_type"],
    "training.save_steps": None,
    "dpo.beta": ["trainer", "dpo", "beta"],
    "signal_quality.minimum_confidence": ["trainer", "trigger", "train_trigger_policy", "signal_quality_gate", "minimum_confidence"],
    "signal_quality.reject_conflicted_signal_quality": ["trainer", "trigger", "train_trigger_policy", "signal_quality_gate", "reject_conflicted_signal_quality"],
    "signal_quality.require_complete_event_chain": ["trainer", "trigger", "train_trigger_policy", "signal_quality_gate", "require_complete_event_chain"],
    "train_trigger.require_holdout_split": ["trainer", "trigger", "train_trigger_policy", "require_holdout_split"],
    "train_trigger.queue_mode": ["trainer", "trigger", "queue_mode"],
    "worker_daemon.heartbeat_interval_seconds": None,
    "worker_daemon.lease_timeout_seconds": None,
    "worker_daemon.max_retries": None,
}


def _set_nested(data: dict[str, Any], path: list[str], value: Any) -> None:
    target = data
    for part in path[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[path[-1]] = value


@contextlib.contextmanager
def temp_config(overrides: dict[str, Any]) -> Generator[dict[str, Any], None, None]:
    """Context manager for temporary configuration overrides.

    Writes a temporary config.toml and points PFE_HOME at it so that
    server/daemon subprocesses pick up the overrides.

    Args:
        overrides: Dictionary of configuration overrides

    Yields:
        The merged configuration dictionary
    """
    original_env = {}
    env_prefix = "PFE_"
    temp_dir_obj: tempfile.TemporaryDirectory | None = None
    original_pfe_home = os.environ.get("PFE_HOME")

    try:
        from pfe_core.config import PFEConfig
    except Exception:
        PFEConfig = None  # type: ignore[misc]

    if PFEConfig is not None:
        # Build merged config dict
        try:
            base_config = PFEConfig.load()
        except Exception:
            base_config = PFEConfig()
        merged = base_config.to_dict()
        for key, value in overrides.items():
            paths = _TEMP_CONFIG_KEY_MAP.get(key)
            if paths is None:
                continue
            if isinstance(paths[0], str):
                paths = [paths]  # type: ignore[list-item]
            for path in paths:
                _set_nested(merged, path, value)  # type: ignore[arg-type]
        # Write temporary config.toml
        temp_dir_obj = tempfile.TemporaryDirectory()
        config_path = Path(temp_dir_obj.name) / "config.toml"
        try:
            merged_config = PFEConfig.from_dict(merged)
            config_path.write_text(merged_config.to_toml(), encoding="utf-8")
        except Exception:
            # Fallback: write a minimal toml with mapped overrides only
            toml_lines: list[str] = []
            for key, value in overrides.items():
                paths = _TEMP_CONFIG_KEY_MAP.get(key)
                if paths is None:
                    continue
                if isinstance(paths[0], str):
                    paths = [paths]  # type: ignore[list-item]
                for path in paths:
                    section = ".".join(path[:-1])  # type: ignore[index]
                    toml_lines.append(f"[{section}]")
                    toml_lines.append(f'{path[-1]} = {json.dumps(value)}')  # type: ignore[index]
            config_path.write_text("\n".join(toml_lines) + "\n", encoding="utf-8")
        os.environ["PFE_HOME"] = str(temp_dir_obj.name)

    for key in overrides:
        env_key = f"{env_prefix}{key.upper()}"
        original_env[env_key] = os.environ.get(env_key)

    for key, value in overrides.items():
        env_key = f"{env_prefix}{key.upper()}"
        os.environ[env_key] = json.dumps(value) if isinstance(value, (dict, list)) else str(value)

    try:
        yield overrides
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if original_pfe_home is None:
            os.environ.pop("PFE_HOME", None)
        else:
            os.environ["PFE_HOME"] = original_pfe_home
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


def chat_completion(
    message: str,
    port: int = DEFAULT_TEST_PORT,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Send a chat completion request.

    Args:
        message: User message
        port: Server port
        session_id: Optional session ID

    Returns:
        Response dictionary
    """
    url = f"http://localhost:{port}/v1/chat/completions"

    payload = {
        "messages": [{"role": "user", "content": message}],
        "model": "local",
        "stream": False,
    }

    if session_id:
        payload["session_id"] = session_id

    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def submit_signal(
    event_type: str,
    context: str,
    model_output: str,
    user_action: dict[str, Any],
    port: int = DEFAULT_TEST_PORT,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> dict[str, Any]:
    """Submit a signal to the server.

    Args:
        event_type: Type of event (accept, reject, edit, etc.)
        context: The input context/prompt
        model_output: Model's output
        user_action: User action details
        port: Server port
        session_id: Optional session ID
        request_id: Optional request ID

    Returns:
        Signal submission response
    """
    import uuid
    url = f"http://localhost:{port}/pfe/signal"
    _session_id = session_id or f"session-{uuid.uuid4().hex[:12]}"
    _request_id = request_id or f"req-{uuid.uuid4().hex[:12]}"
    _source_event_id = f"source-{uuid.uuid4().hex[:12]}"
    _event_id = f"event-{uuid.uuid4().hex[:12]}"

    payload = {
        "event_id": _event_id,
        "source_event_id": _source_event_id,
        "source_event_ids": [_source_event_id, _event_id],
        "event_type": event_type,
        "session_id": _session_id,
        "request_id": _request_id,
        "user_input": context,
        "model_output": model_output,
        "user_action": user_action,
        "metadata": {
            "workspace": os.environ.get("PFE_WORKSPACE", "user_default"),
        },
    }

    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.Timeout as e:
            if attempt < 2:
                print(f"Timeout submitting signal (attempt {attempt + 1}), retrying...")
                time.sleep(1.0)
                continue
            raise
        except requests.RequestException:
            raise


def get_signals(port: int = DEFAULT_TEST_PORT) -> list[dict[str, Any]]:
    """Get all signals from the server.

    Args:
        port: Server port

    Returns:
        List of signal dictionaries
    """
    url = f"http://localhost:{port}/pfe/signals"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("signals", [])
    except requests.RequestException:
        return []


def get_queue_depth(port: int = DEFAULT_TEST_PORT) -> int:
    """Get current training queue depth.

    Args:
        port: Server port

    Returns:
        Number of items in queue
    """
    url = f"http://localhost:{port}/pfe/queue/status"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("depth", 0)
    except requests.RequestException:
        return 0


def get_job_status(job_id: str, port: int = DEFAULT_TEST_PORT) -> Optional[str]:
    """Get status of a training job.

    Args:
        job_id: Job ID
        port: Server port

    Returns:
        Job status string or None if not found
    """
    url = f"http://localhost:{port}/pfe/training/jobs/{job_id}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("status")
    except requests.RequestException:
        return None


def get_retry_count(job_id: str, port: int = DEFAULT_TEST_PORT) -> int:
    """Get retry count for a job.

    Args:
        job_id: Job ID
        port: Server port

    Returns:
        Number of retries
    """
    url = f"http://localhost:{port}/pfe/training/jobs/{job_id}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("retry_count", 0)
    except requests.RequestException:
        return 0


def submit_training_job(
    method: str = "sft",
    port: int = DEFAULT_TEST_PORT,
) -> str:
    """Submit a training job.

    Args:
        method: Training method (sft or dpo)
        port: Server port

    Returns:
        Job ID
    """
    url = f"http://localhost:{port}/pfe/training/jobs"

    payload = {
        "method": method,
        "auto_trigger": False,
    }

    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()["job_id"]


def wait_for_training(
    timeout: float = 600,
    port: int = DEFAULT_TEST_PORT,
) -> Optional[dict[str, Any]]:
    """Wait for training to complete.

    Args:
        timeout: Maximum time to wait
        port: Server port

    Returns:
        Training result or None if timeout
    """
    url = f"http://localhost:{port}/pfe/training/status"

    def check_training() -> bool:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            status = response.json()
            return status.get("state") in ["completed", "failed"]
        except requests.RequestException:
            return False

    try:
        wait_for(check_training, timeout=timeout, message="Training did not complete")
    except TimeoutError:
        return None

    # Get final result
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def wait_for_eval(
    timeout: float = 300,
    port: int = DEFAULT_TEST_PORT,
) -> Optional[dict[str, Any]]:
    """Wait for evaluation to complete.

    Args:
        timeout: Maximum time to wait
        port: Server port

    Returns:
        Evaluation result or None if timeout
    """
    url = f"http://localhost:{port}/pfe/eval/status"

    def check_eval() -> bool:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            status = response.json()
            return status.get("state") in ["completed", "failed"]
        except requests.RequestException:
            return False

    try:
        wait_for(check_eval, timeout=timeout, message="Eval did not complete")
    except TimeoutError:
        return None

    # Get final result
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def get_latest_adapter(port: int = DEFAULT_TEST_PORT) -> Optional[str]:
    """Get latest promoted adapter version.

    Args:
        port: Server port

    Returns:
        Adapter version or None
    """
    url = f"http://localhost:{port}/pfe/adapters/latest"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("version")
    except requests.RequestException:
        return None


def adapter_exists(version: str, port: int = DEFAULT_TEST_PORT) -> bool:
    """Check if adapter version exists.

    Args:
        version: Adapter version
        port: Server port

    Returns:
        True if adapter exists
    """
    url = f"http://localhost:{port}/pfe/adapters/{version}"

    try:
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False


def run_eval(version: str, port: int = DEFAULT_TEST_PORT) -> dict[str, Any]:
    """Run evaluation on an adapter version.

    Args:
        version: Adapter version
        port: Server port

    Returns:
        Evaluation report
    """
    url = f"http://localhost:{port}/pfe/eval"

    payload = {"version": version}

    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def promote(version: str, port: int = DEFAULT_TEST_PORT) -> dict[str, Any]:
    """Promote an adapter version.

    Args:
        version: Adapter version
        port: Server port

    Returns:
        Promotion result
    """
    url = f"http://localhost:{port}/pfe/adapters/{version}/promote"

    response = requests.post(url, timeout=10)
    response.raise_for_status()
    return response.json()


def _extract_response_content(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message", {})
        if isinstance(msg, dict):
            return msg.get("content", "") or ""
    return response.get("content", "") or ""


def simulate_user_accept(response: dict[str, Any], context: str = "") -> dict[str, Any]:
    """Simulate user accepting a response.

    Args:
        response: Chat completion response
        context: Original prompt/context

    Returns:
        Signal submission response
    """
    content = _extract_response_content(response)
    return submit_signal(
        event_type="accept",
        context=context,
        model_output=content,
        user_action={"type": "accept", "accepted_text": content},
        session_id=response.get("session_id"),
        request_id=response.get("request_id"),
    )


def simulate_user_edit(response: dict[str, Any], edited_text: str, context: str = "") -> dict[str, Any]:
    """Simulate user editing a response.

    Args:
        response: Chat completion response
        edited_text: Edited text
        context: Original prompt/context

    Returns:
        Signal submission response
    """
    content = _extract_response_content(response)
    return submit_signal(
        event_type="edit",
        context=context,
        model_output=content,
        user_action={
            "type": "edit",
            "edited_text": edited_text,
            "original_text": content,
        },
        session_id=response.get("session_id"),
        request_id=response.get("request_id"),
    )


def simulate_user_reject(response: dict[str, Any], context: str = "") -> dict[str, Any]:
    """Simulate user rejecting a response.

    Args:
        response: Chat completion response
        context: Original prompt/context

    Returns:
        Signal submission response
    """
    content = _extract_response_content(response)
    return submit_signal(
        event_type="reject",
        context=context,
        model_output=content,
        user_action={"type": "reject", "rejected_text": content},
        session_id=response.get("session_id"),
        request_id=response.get("request_id"),
    )


def simulate_conversations(
    count: int,
    port: int = DEFAULT_TEST_PORT,
    accept_ratio: float = 0.7,
    edit_ratio: float = 0.2,
) -> list[dict[str, Any]]:
    """Simulate multiple conversations with signals.

    Args:
        count: Number of conversations
        port: Server port
        accept_ratio: Ratio of accepted responses
        edit_ratio: Ratio of edited responses

    Returns:
        List of signal responses
    """
    import random

    signals = []
    prompts = [
        "Hello, how are you?",
        "What is the weather like?",
        "Tell me a joke.",
        "Explain Python.",
        "Write a poem.",
        "What is AI?",
        "How do I cook pasta?",
        "Recommend a book.",
        "What is 2+2?",
        "Translate to French: Hello",
    ]

    for i in range(count):
        prompt = prompts[i % len(prompts)]
        response = None
        for attempt in range(3):
            try:
                response = chat_completion(prompt, port=port)
                break
            except requests.Timeout as e:
                print(f"Timeout simulating conversation {i} (attempt {attempt + 1}): {e}")
                time.sleep(1.0)
            except requests.RequestException as e:
                print(f"Failed to simulate conversation {i} (attempt {attempt + 1}): {e}")
                break
        if response is None:
            continue

        # Determine user action based on ratios
        rand = random.random()
        if rand < accept_ratio:
            signal = simulate_user_accept(response, context=prompt)
        elif rand < accept_ratio + edit_ratio:
            signal = simulate_user_edit(response, edited_text=f"Edited: {_extract_response_content(response)}", context=prompt)
        else:
            signal = simulate_user_reject(response, context=prompt)

        signals.append(signal)
        # Small delay to reduce contention with daemon
        time.sleep(0.2)

    return signals


def create_preference_pairs(
    pairs: list[tuple[str, str, str]],
    port: int = DEFAULT_TEST_PORT,
) -> list[dict[str, Any]]:
    """Create preference pairs for DPO training.

    Args:
        pairs: List of (prompt, chosen, rejected) tuples
        port: Server port

    Returns:
        List of created signal responses
    """
    signals = []

    for prompt, chosen, rejected in pairs:
        _session_id = f"session-{uuid4().hex[:12]}"
        _request_id = f"req-{uuid4().hex[:12]}"

        # Submit accepted signal (chosen)
        accept_signal = submit_signal(
            event_type="accept",
            context=prompt,
            model_output=chosen,
            user_action={"type": "accept", "accepted_text": chosen},
            session_id=_session_id,
            request_id=_request_id,
            port=port,
        )
        signals.append(accept_signal)

        # Submit rejected signal with accepted_text so it can curate into a DPO sample
        reject_signal = submit_signal(
            event_type="reject",
            context=prompt,
            model_output=rejected,
            user_action={
                "type": "reject",
                "rejected_text": rejected,
                "accepted_text": chosen,
            },
            session_id=_session_id,
            request_id=_request_id,
            port=port,
        )
        signals.append(reject_signal)

    return signals


def kill_runner(job_id: str, port: int = DEFAULT_TEST_PORT) -> bool:
    """Kill a runner process (for failure recovery testing).

    Args:
        job_id: Job ID
        port: Server port

    Returns:
        True if kill signal was sent
    """
    url = f"http://localhost:{port}/pfe/training/jobs/{job_id}/runner"

    try:
        # Get runner PID
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        pid = response.json().get("pid")

        if pid:
            os.kill(pid, signal.SIGKILL)
            return True
    except (requests.RequestException, ProcessLookupError):
        pass

    return False


def run_server(port: int = DEFAULT_TEST_PORT) -> contextlib.AbstractContextManager[TestServer]:
    """Factory function for TestServer context manager.

    Args:
        port: Server port

    Returns:
        TestServer context manager
    """
    return TestServer(port=port)


def run_daemon(port: int = DEFAULT_TEST_PORT) -> contextlib.AbstractContextManager[TestDaemon]:
    """Factory function for TestDaemon context manager.

    Args:
        port: Server port

    Returns:
        TestDaemon context manager
    """
    return TestDaemon(port=port)


def submit_feedback(
    action: str,
    session_id: str,
    request_id: str,
    user_message: str = "",
    assistant_message: str = "",
    response_time_seconds: Optional[float] = None,
    edited_text: Optional[str] = None,
    next_message: Optional[str] = None,
    port: int = DEFAULT_TEST_PORT,
) -> dict[str, Any]:
    """Submit user feedback via the /pfe/feedback endpoint.

    This function uses the new feedback endpoint that integrates with
    ChatCollector for implicit signal extraction.

    Args:
        action: Feedback action type (accept, reject, edit, regenerate, delete)
        session_id: Session identifier
        request_id: Request identifier
        user_message: Original user message
        assistant_message: Assistant's response
        response_time_seconds: Time taken for user to respond
        edited_text: Edited text (for edit actions)
        next_message: Next user message (for accept actions)
        port: Server port

    Returns:
        Feedback submission response
    """
    url = f"http://localhost:{port}/pfe/feedback"

    payload: dict[str, Any] = {
        "session_id": session_id,
        "request_id": request_id,
        "action": action,
        "user_message": user_message,
        "assistant_message": assistant_message,
    }

    if response_time_seconds is not None:
        payload["response_time_seconds"] = response_time_seconds
    if edited_text is not None:
        payload["edited_text"] = edited_text
    if next_message is not None:
        payload["next_message"] = next_message

    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def simulate_user_accept_via_feedback(
    response: dict[str, Any],
    next_message: Optional[str] = None,
    port: int = DEFAULT_TEST_PORT,
) -> dict[str, Any]:
    """Simulate user accepting a response via feedback endpoint.

    Args:
        response: Chat completion response
        next_message: Optional next message from user
        port: Server port

    Returns:
        Feedback submission response
    """
    return submit_feedback(
        action="accept",
        session_id=response.get("session_id", ""),
        request_id=response.get("request_id", ""),
        user_message=response.get("prompt", ""),
        assistant_message=response.get("content", ""),
        next_message=next_message,
        port=port,
    )


def simulate_user_edit_via_feedback(
    response: dict[str, Any],
    edited_text: str,
    port: int = DEFAULT_TEST_PORT,
) -> dict[str, Any]:
    """Simulate user editing a response via feedback endpoint.

    Args:
        response: Chat completion response
        edited_text: Edited text
        port: Server port

    Returns:
        Feedback submission response
    """
    return submit_feedback(
        action="edit",
        session_id=response.get("session_id", ""),
        request_id=response.get("request_id", ""),
        user_message=response.get("prompt", ""),
        assistant_message=response.get("content", ""),
        edited_text=edited_text,
        port=port,
    )


def simulate_user_reject_via_feedback(
    response: dict[str, Any],
    port: int = DEFAULT_TEST_PORT,
) -> dict[str, Any]:
    """Simulate user rejecting a response via feedback endpoint.

    Args:
        response: Chat completion response
        port: Server port

    Returns:
        Feedback submission response
    """
    return submit_feedback(
        action="reject",
        session_id=response.get("session_id", ""),
        request_id=response.get("request_id", ""),
        user_message=response.get("prompt", ""),
        assistant_message=response.get("content", ""),
        port=port,
    )


def simulate_user_regenerate_via_feedback(
    response: dict[str, Any],
    port: int = DEFAULT_TEST_PORT,
) -> dict[str, Any]:
    """Simulate user requesting regeneration via feedback endpoint.

    Args:
        response: Chat completion response
        port: Server port

    Returns:
        Feedback submission response
    """
    return submit_feedback(
        action="regenerate",
        session_id=response.get("session_id", ""),
        request_id=response.get("request_id", ""),
        user_message=response.get("prompt", ""),
        assistant_message=response.get("content", ""),
        port=port,
    )
