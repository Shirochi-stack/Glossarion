"""Managed local Ollama transport for the ``ollamapull/`` model route.

Only the fixed loopback Ollama API is used. Installation uses Ollama's official
platform installers; the model and chat paths use its native HTTP API.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional

import requests

import mobile_runtime


BASE_URL = "http://127.0.0.1:11434"
#: Glossarion Mobile cannot install or start Ollama: it uses an Ollama server on the user's PC
#: (``OLLAMAPULL_BASE_URL``), where models are pulled through the same API.
MOBILE_HINT = ("ollamapull/ on mobile uses the Ollama server on your PC (for a model on the phone itself, use "
               "Model manager › On-device): start Ollama there (with OLLAMA_HOST=0.0.0.0 "
               "so the phone can reach it) and set its address in Settings › Local AI")


def base_url() -> str:
    """The Ollama API this route uses: the loopback server (desktop) or ``OLLAMAPULL_BASE_URL`` (mobile)."""
    return os.environ.get("OLLAMAPULL_BASE_URL", "").strip().rstrip("/") or BASE_URL
_LATEST_RELEASE_URL = "https://api.github.com/repos/ollama/ollama/releases/latest"
_UPDATE_CHECK_SECONDS = 6 * 60 * 60
_lifecycle_lock = threading.RLock()
_release_lock = threading.Lock()
_release_checked_at = 0.0
_release_version: Optional[str] = None
_attempted_auto_updates: set[tuple[str, str]] = set()
_active_chats = 0
_managed_server_process: Optional[subprocess.Popen] = None


class OllamaPullError(RuntimeError):
    """Local Ollama installation, transport, or API failure."""


class OllamaPullCancelled(OllamaPullError):
    """The user stopped a pull or request."""


def native_model_name(model_name: str) -> str:
    name = str(model_name or "").strip()
    if name.lower().startswith("ollamapull/"):
        name = name[len("ollamapull/"):].strip()
    if not name:
        raise OllamaPullError("Enter an Ollama model after 'ollamapull/'.")
    return name


_HF_PART = r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}"
_HF_RE = re.compile(rf"^hf\.co/({_HF_PART})/({_HF_PART})(?::([A-Za-z0-9][A-Za-z0-9._-]{{0,63}}))?$")
_LIBRARY_RE = re.compile(r"^(?:[a-z0-9][a-z0-9._-]{0,63}/){0,2}[a-z0-9][a-z0-9._-]{0,95}"
                         r"(?::[A-Za-z0-9][A-Za-z0-9._-]{0,127})?$")


def normalize_pull_name(text: str) -> str:
    """The name ``ollama pull`` / ``/api/pull`` takes for what a user typed or pasted.

    Accepts an Ollama library tag (``qwen3.5:2b``, ``user/model:tag``), a Hugging Face GGUF repo in
    Ollama's syntax (``hf.co/<user>/<repo>[:<quant>]``), a ``huggingface.co`` link to one, or a bare
    ``<user>/<repo>-GGUF[:<quant>]`` id. An ``ollamapull/`` prefix is dropped. Raises
    ``OllamaPullError`` with a short reason for anything else."""
    value = str(text or "").strip()
    if value.lower().startswith("ollamapull/"):
        value = value[len("ollamapull/"):].strip()
    link = False
    for scheme in ("https://", "http://"):
        if value.lower().startswith(scheme):
            value = value[len(scheme):]
            link = True
    if link and not value.lower().startswith(("huggingface.co/", "www.huggingface.co/", "hf.co/")):
        raise OllamaPullError("Only Hugging Face links can be pulled; enter other models as name:tag.")
    if not value:
        raise OllamaPullError("Enter an Ollama model tag or a Hugging Face id.")
    if any(ch.isspace() for ch in value):
        raise OllamaPullError("A model name cannot contain spaces.")
    lowered = value.lower()
    for host in ("www.huggingface.co/", "huggingface.co/", "hf.co/"):
        if lowered.startswith(host):
            value = "hf.co/" + value[len(host):]
            break
    else:
        repo = value.split(":", 1)[0]
        if repo.count("/") == 1 and repo.lower().endswith("-gguf"):
            value = "hf.co/" + value
    if value.startswith("hf.co/"):
        # a pasted page link: hf.co/<user>/<repo>/tree/main, /blob/main/<file>.gguf ...
        head, _, quant = value.partition(":")
        parts = head.split("/")
        if len(parts) > 3:
            head = "/".join(parts[:3])
        value = f"{head}:{quant}" if quant else head
        if not _HF_RE.match(value):
            raise OllamaPullError("A Hugging Face id looks like hf.co/<user>/<repo> or hf.co/<user>/<repo>:<quant>.")
        return value
    value = value.lower()  # library names and tags are lowercase (ollama.com/library)
    if not _LIBRARY_RE.match(value):
        raise OllamaPullError("An Ollama model looks like name:tag (for example qwen3.5:2b) "
                              "or hf.co/<user>/<repo>:<quant>.")
    return value


def load_settings() -> dict[str, Any]:
    """Read the GUI's saved settings snapshot on every use."""
    raw = os.environ.get("OLLAMA_SETTINGS_JSON", "")
    if not raw:
        return {"auto_update": True, "models": {}}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise OllamaPullError(f"Invalid Ollama settings JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise OllamaPullError("Ollama settings must be a JSON object.")
    value.setdefault("auto_update", True)
    value.setdefault("models", {})
    if not isinstance(value["models"], dict):
        raise OllamaPullError("Ollama model settings must be a JSON object.")
    return value


def model_settings(model_name: str) -> dict[str, Any]:
    name = native_model_name(model_name)
    entry = load_settings()["models"].get(name, {})
    if not isinstance(entry, dict):
        raise OllamaPullError(f"Settings for {name} must be a JSON object.")
    return entry


def _progress(callback: Optional[Callable[[str], None]], message: str) -> None:
    print(f"[Ollama] {message}", flush=True)
    if callback is not None:
        try:
            callback(message)
        except Exception:
            pass


def _chat_status(callback: Optional[Callable[[str], None]], message: str) -> None:
    """Report chat progress through the caller's log without a duplicate print."""
    if callback is not None:
        try:
            callback(message)
        except Exception:
            pass


#: after a new line or closing block tag, and before an opening <p> (models that never close their paragraphs)
_STREAM_LOG_BREAK = re.compile(r"\n|</(?:h[1-6]|p|div|li|ul|ol|table|tr|blockquote|head|body|html)>|(?=<p[\s>])", re.IGNORECASE)


def _stream_log_fragment(pending: str, fragment: str, *, prefix: str = "",
                         force: bool = False) -> str:
    """Emit readable live log lines while retaining a short incomplete line: a line ends at a new line, after
    a closing block tag or before an opening <p>, and a long one at its last space, never inside a word or tag."""
    pending += fragment
    while True:
        pending = pending.lstrip("\n")
        found = _STREAM_LOG_BREAK.search(pending, 1)  # from 1: a line that starts with <p> is not an empty line
        if found is None:
            break
        line, pending = pending[:found.end()], pending[found.end():]
        if line.strip():
            print(f"{prefix}{line.rstrip()}", flush=True)
    while len(pending) >= 160:
        cut = pending.rfind(" ", 0, 160)
        if cut <= 0:
            cut = pending.find(" ", 160)
        if cut <= 0:
            break
        print(f"{prefix}{pending[:cut]}", flush=True)
        pending = pending[cut + 1:]
    if pending and force:
        print(f"{prefix}{pending}", flush=True)
        pending = ""
    return pending


def _check_stop(should_stop: Optional[Callable[[], bool]]) -> None:
    if should_stop is not None and should_stop():
        raise OllamaPullCancelled("Ollama operation cancelled")


def _http_error(response: requests.Response, action: str) -> OllamaPullError:
    try:
        body = response.json()
        detail = body.get("error") or body.get("message") if isinstance(body, dict) else None
    except (ValueError, AttributeError):
        detail = None
    if not detail:
        detail = (getattr(response, "text", "") or "").strip()[:500]
    return OllamaPullError(f"Ollama {action} failed (HTTP {response.status_code}): {detail or response.reason}")


def _api_get(path: str, timeout: float = 3.0) -> dict[str, Any]:
    response = requests.get(f"{base_url()}{path}", timeout=timeout)
    if not response.ok:
        raise _http_error(response, path)
    data = response.json()
    if not isinstance(data, dict):
        raise OllamaPullError(f"Unexpected Ollama response from {path}.")
    return data


def _api_post(path: str, payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    response = requests.post(f"{base_url()}{path}", json=payload, timeout=timeout)
    if not response.ok:
        raise _http_error(response, path)
    data = response.json()
    if not isinstance(data, dict):
        raise OllamaPullError(f"Unexpected Ollama response from {path}.")
    return data


def _ollama_executable() -> Optional[str]:
    if sys.platform == "darwin":
        managed = Path.home() / "Applications" / "Ollama.app" / "Contents" / "Resources" / "ollama"
        if managed.is_file():
            return str(managed)
    found = shutil.which("ollama")
    if found:
        return found
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "")
        candidates = [Path(local) / "Programs" / "Ollama" / "ollama.exe"] if local else []
    elif sys.platform == "darwin":
        candidates = [Path("/Applications/Ollama.app/Contents/Resources/ollama")]
    else:
        candidates = [Path("/usr/local/bin/ollama"), Path("/usr/bin/ollama")]
    return str(next((path for path in candidates if path.is_file()), "")) or None


def _hidden_subprocess_kwargs() -> dict[str, Any]:
    """Keep Ollama CLI subprocesses in Task Manager without console flashes."""
    if sys.platform != "win32":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        "startupinfo": startupinfo,
    }


def _binary_version() -> Optional[str]:
    if not mobile_runtime.subprocesses_available():
        return None
    executable = _ollama_executable()
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"], capture_output=True, text=True,
            timeout=5, **_hidden_subprocess_kwargs(),
        )
        match = re.search(r"\b(\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.]+)?)\b", result.stdout + result.stderr)
        return match.group(1) if match else None
    except (OSError, subprocess.SubprocessError):
        return None


def _server_version() -> Optional[str]:
    try:
        value = _api_get("/api/version", timeout=1.5).get("version")
        return str(value).lstrip("v") if value else None
    except (requests.RequestException, OllamaPullError, ValueError):
        return None


def _installed_version() -> Optional[str]:
    return _binary_version() or _server_version()


def _version_tuple(version: Optional[str]) -> Optional[tuple[int, int, int]]:
    match = re.match(r"^v?(\d+)\.(\d+)\.(\d+)", str(version or ""))
    return tuple(map(int, match.groups())) if match else None


def latest_version(*, force: bool = False) -> Optional[str]:
    global _release_checked_at, _release_version
    with _release_lock:
        if not force and time.monotonic() - _release_checked_at < _UPDATE_CHECK_SECONDS:
            return _release_version
        try:
            response = requests.get(_LATEST_RELEASE_URL, headers={"Accept": "application/vnd.github+json"}, timeout=5)
            response.raise_for_status()
            version = str(response.json().get("tag_name") or "").lstrip("v")
            if _version_tuple(version) is None:
                raise OllamaPullError("Latest Ollama release has no version number.")
            _release_version = version
            _release_checked_at = time.monotonic()
        except (requests.RequestException, ValueError, OllamaPullError):
            # Updating is best effort; an offline user can keep translating.
            _release_checked_at = time.monotonic()
        return _release_version


def _models() -> list[dict[str, Any]]:
    data = _api_get("/api/tags")
    value = data.get("models", [])
    return value if isinstance(value, list) else []


def _model_is_installed(model_name: str, models: list[dict[str, Any]]) -> bool:
    name = native_model_name(model_name).lower()
    aliases = {name, f"{name}:latest"} if ":" not in name else {name}
    for item in models:
        if isinstance(item, dict) and str(item.get("name") or item.get("model") or "").lower() in aliases:
            return True
    return False


def get_model_details(model_name: str) -> dict[str, Any]:
    """Return native /api/show metadata, including capabilities and defaults."""
    return _api_post("/api/show", {"model": native_model_name(model_name)})


def get_status(model_name: str = "") -> dict[str, Any]:
    """Read-only installation, server, release, and selected-model status."""
    status: dict[str, Any] = {
        "installed": False, "version": None, "server_running": False,
        "model_installed": None, "model_loaded": None, "model_info": None,
        "latest_version": None, "update_available": False,
        "installed_version": None, "server_version": None, "restart_required": False,
    }
    executable = _ollama_executable()
    status["server_version"] = _server_version()
    status["server_running"] = bool(status["server_version"])
    status["installed_version"] = _binary_version() if executable else None
    status["version"] = status["server_version"] or status["installed_version"]
    status["installed"] = bool(executable or status["server_running"])
    if status["server_running"] and model_name:
        try:
            status["model_installed"] = _model_is_installed(model_name, _models())
            if status["model_installed"]:
                status["model_info"] = get_model_details(model_name)
        except (requests.RequestException, OllamaPullError, ValueError) as exc:
            status["error"] = str(exc)
        try:
            running_models = _api_get("/api/ps").get("models", [])
            status["model_loaded"] = _model_is_installed(
                model_name, running_models if isinstance(running_models, list) else [])
        except (requests.RequestException, OllamaPullError, ValueError) as exc:
            status["loaded_error"] = str(exc)
    status["latest_version"] = latest_version()
    installed_version = _version_tuple(status["installed_version"] or status["version"])
    available_version = _version_tuple(status["latest_version"])
    status["update_available"] = bool(installed_version and available_version and available_version > installed_version)
    server_version = _version_tuple(status["server_version"])
    status["restart_required"] = bool(installed_version and server_version and installed_version > server_version)
    return status


def _installer_command(script_path: Path) -> list[str]:
    if not mobile_runtime.subprocesses_available():
        raise OllamaPullError(MOBILE_HINT)
    if sys.platform == "win32":
        shell = shutil.which("powershell.exe") or "powershell.exe"
        return [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script_path)]
    if sys.platform == "linux":
        if hasattr(os, "geteuid") and os.geteuid() != 0:
            try:
                subprocess.run(["sudo", "-n", "true"], check=True, capture_output=True, timeout=3)
            except (OSError, subprocess.SubprocessError):
                pkexec = shutil.which("pkexec")
                if pkexec and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                    return [pkexec, "/bin/sh", str(script_path)]
                raise OllamaPullError(
                    "Ollama needs administrator access on Linux, but no graphical "
                    "authorization agent is available. Run Ollama's official installer in a terminal."
                )
        return ["/bin/sh", str(script_path)]
    raise OllamaPullError(f"Automatic Ollama installation is unavailable on {sys.platform}.")


def _install_macos_app(progress: Optional[Callable[[str], None]],
                       should_stop: Optional[Callable[[], bool]]) -> None:
    """Install Ollama's official app archive in the current user's Applications."""
    if not mobile_runtime.subprocesses_available():
        raise OllamaPullError(MOBILE_HINT)
    _check_stop(should_stop)
    _progress(progress, "Downloading the official Ollama app for macOS...")
    destination = Path.home() / "Applications" / "Ollama.app"
    with tempfile.TemporaryDirectory(prefix="glossarion-ollama-") as directory:
        archive = Path(directory) / "Ollama-darwin.zip"
        try:
            with requests.get("https://ollama.com/download/Ollama-darwin.zip",
                              stream=True, timeout=(5, 30)) as response:
                response.raise_for_status()
                total = int(response.headers.get("Content-Length") or 0)
                downloaded = 0
                with archive.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        _check_stop(should_stop)
                        if chunk:
                            output.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                _progress(progress, f"Ollama app download: {downloaded / total:.0%}")
        except requests.RequestException as exc:
            _check_stop(should_stop)
            raise OllamaPullError(f"Could not download Ollama for macOS: {exc}") from exc
        _check_stop(should_stop)
        extracted = Path(directory) / "extracted"
        try:
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.namelist():
                    member_path = Path(member)
                    if member_path.is_absolute() or ".." in member_path.parts:
                        raise OllamaPullError("The Ollama app archive contains an unsafe path.")
        except zipfile.BadZipFile as exc:
            raise OllamaPullError("The Ollama app download is not a valid ZIP archive.") from exc
        try:
            subprocess.run(["/usr/bin/ditto", "-x", "-k", str(archive), str(extracted)],
                           check=True, capture_output=True, timeout=600)
        except (OSError, subprocess.SubprocessError) as exc:
            raise OllamaPullError(f"Could not extract the Ollama app: {exc}") from exc
        source = extracted / "Ollama.app"
        if not (source / "Contents" / "Resources" / "ollama").is_file():
            raise OllamaPullError("The Ollama app archive does not contain its CLI executable.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        backup = Path(directory) / "previous-Ollama.app"
        if destination.exists():
            shutil.move(str(destination), str(backup))
        try:
            shutil.move(str(source), str(destination))
        except OSError:
            if backup.exists():
                shutil.move(str(backup), str(destination))
            raise
    _progress(progress, f"Ollama installed in {destination}.")


def _run_installer(progress: Optional[Callable[[str], None]], should_stop: Optional[Callable[[], bool]]) -> None:
    if not mobile_runtime.subprocesses_available():
        raise OllamaPullError(f"{MOBILE_HINT} ({base_url()} did not answer)")
    if sys.platform == "win32":
        script_url, suffix = "https://ollama.com/install.ps1", ".ps1"
    elif sys.platform == "darwin":
        _install_macos_app(progress, should_stop)
        return
    elif sys.platform == "linux":
        script_url, suffix = "https://ollama.com/install.sh", ".sh"
    else:
        raise OllamaPullError(f"Automatic Ollama installation is unavailable on {sys.platform}.")
    _check_stop(should_stop)
    _progress(progress, "Downloading the official Ollama installer...")
    try:
        response = requests.get(script_url, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaPullError(f"Could not download Ollama's official installer: {exc}") from exc
    script = response.content
    expected_start = b"<#" if sys.platform == "win32" else b"#!/bin/sh"
    if not script.lstrip().startswith(expected_start) or len(script) < 100:
        raise OllamaPullError("The official Ollama installer response was not a valid install script.")
    _check_stop(should_stop)
    with tempfile.TemporaryDirectory(prefix="glossarion-ollama-") as directory:
        path = Path(directory) / f"install{suffix}"
        path.write_bytes(script)
        command = _installer_command(path)
        _progress(progress, "Installing Ollama; this may take several minutes...")
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                **_hidden_subprocess_kwargs(),
            )
        except OSError as exc:
            raise OllamaPullError(f"Could not start the Ollama installer: {exc}") from exc
        output: queue.Queue[Optional[str]] = queue.Queue()

        def read_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                output.put(line.strip())
            output.put(None)

        threading.Thread(target=read_output, daemon=True).start()
        last_lines: list[str] = []
        try:
            while process.poll() is None:
                _check_stop(should_stop)
                try:
                    line = output.get(timeout=0.25)
                except queue.Empty:
                    continue
                if line:
                    last_lines.append(line)
                    last_lines = last_lines[-8:]
                    _progress(progress, line)
            while not output.empty():
                line = output.get_nowait()
                if line:
                    last_lines.append(line)
                    last_lines = last_lines[-8:]
                    _progress(progress, line)
            if process.returncode != 0:
                raise OllamaPullError(
                    f"Ollama installer failed (exit {process.returncode}): {'; '.join(last_lines) or 'check system permissions and disk space'}"
                )
        except OllamaPullCancelled:
            process.terminate()
            raise


def update_ollama(*, progress: Optional[Callable[[str], None]] = None,
                  should_stop: Optional[Callable[[], bool]] = None) -> None:
    """Run the official installer, which installs or upgrades the current app."""
    with _lifecycle_lock:
        _run_installer(progress, should_stop)
        _after_update(progress, should_stop)
        _progress(progress, "Ollama installation finished.")


def _start_server(progress: Optional[Callable[[str], None]], should_stop: Optional[Callable[[], bool]]) -> None:
    global _managed_server_process
    if not mobile_runtime.subprocesses_available():
        try:
            _api_get("/api/version", timeout=3)
            return
        except (requests.RequestException, OllamaPullError, ValueError) as exc:
            raise OllamaPullError(f"{MOBILE_HINT} ({base_url()}: {exc})") from exc
    try:
        _api_get("/api/version", timeout=1)
        return
    except (requests.RequestException, OllamaPullError, ValueError):
        pass
    executable = _ollama_executable()
    if not executable:
        raise OllamaPullError("Ollama was installed, but its executable could not be found.")
    _progress(progress, "Starting the local Ollama server...")
    kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform == "win32":
        kwargs.update(_hidden_subprocess_kwargs())
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen([executable, "serve"], **kwargs)
        _managed_server_process = process
    except OSError as exc:
        raise OllamaPullError(f"Could not start Ollama: {exc}") from exc
    deadline = time.monotonic() + 30
    exited_code: Optional[int] = None
    while time.monotonic() < deadline:
        _check_stop(should_stop)
        try:
            _api_get("/api/version", timeout=1)
        except (requests.RequestException, OllamaPullError, ValueError):
            code = process.poll()
            if code is not None and exited_code is None:
                exited_code = code
                if _managed_server_process is process:
                    _managed_server_process = None
                _progress(progress, "Ollama launch exited; checking whether its app server is starting...")
            time.sleep(0.25)
        else:
            _check_stop(should_stop)
            if process.poll() is not None and _managed_server_process is process:
                # The Ollama app/installer may have started its own server while
                # this duplicate ``serve`` command was exiting on a port conflict.
                _managed_server_process = None
            return
    _check_stop(should_stop)
    # A server can become available during the final sleep or HTTP timeout.
    try:
        _api_get("/api/version", timeout=1)
    except (requests.RequestException, OllamaPullError, ValueError):
        pass
    else:
        _check_stop(should_stop)
        if process.poll() is not None and _managed_server_process is process:
            _managed_server_process = None
        return
    _check_stop(should_stop)
    code = process.poll()
    if code is not None:
        if _managed_server_process is process:
            _managed_server_process = None
        raise OllamaPullError(f"Ollama server exited (code {code}); no local server became available within 30 seconds.")
    raise OllamaPullError("Ollama did not start within 30 seconds.")


def shutdown_ollama() -> bool:
    """Stop the local Ollama server, including one started by the Ollama app.

    Only a process verified to own the loopback Ollama listener (or a process
    started by this module) is eligible. Returns whether a process was stopped.
    """
    global _managed_server_process
    _stop_ban_server()
    with _lifecycle_lock:
        targets = []
        try:
            import psutil
            executable = _ollama_executable()
            install_dir = Path(executable).resolve().parent if executable else None
            if install_dir is not None:
                for process in psutil.process_iter():
                    try:
                        if process.name().lower() not in ("llama-server", "llama-server.exe"):
                            continue
                        runner_path = Path(process.exe()).resolve()
                        if runner_path.is_relative_to(install_dir):
                            # Ollama's model runner can outlive its server and
                            # retain the model's VRAM allocation.
                            targets.append(process)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                        continue
            for connection in psutil.net_connections(kind="tcp"):
                address = connection.laddr
                if not address or address.port != 11434 or connection.status != psutil.CONN_LISTEN:
                    continue
                if address.ip not in ("127.0.0.1", "::1", "0.0.0.0", "::") or not connection.pid:
                    continue
                try:
                    process = psutil.Process(connection.pid)
                    if not process.name().lower().startswith("ollama"):
                        continue
                    targets.append(process)
                    # The desktop app can relaunch its server after a direct
                    # child shutdown. Stop Ollama parents as well.
                    parent = process.parent()
                    while parent is not None and parent.name().lower().startswith("ollama"):
                        targets.append(parent)
                        parent = parent.parent()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except ImportError:
            psutil = None
        except Exception as exc:
            raise OllamaPullError(f"Could not identify the local Ollama server: {exc}") from exc

        managed = _managed_server_process
        if managed is not None and managed.poll() is None:
            try:
                managed.terminate()
                managed.wait(timeout=2)
            except subprocess.TimeoutExpired:
                managed.kill()
                managed.wait(timeout=2)
            _managed_server_process = None
            stopped = True
        else:
            _managed_server_process = None
            stopped = False

        if psutil is not None:
            unique = {process.pid: process for process in targets}
            for process in unique.values():
                try:
                    process.terminate()
                    stopped = True
                except psutil.NoSuchProcess:
                    pass
                except psutil.AccessDenied as exc:
                    raise OllamaPullError(f"Access denied stopping Ollama (PID {process.pid}).") from exc
            if unique:
                _, alive = psutil.wait_procs(list(unique.values()), timeout=2)
                for process in alive:
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
                if alive:
                    _, alive = psutil.wait_procs(alive, timeout=2)
                    if alive:
                        raise OllamaPullError("Ollama processes did not exit after shutdown.")
        if not stopped and _server_version() is not None:
            raise OllamaPullError("Ollama is running, but its local server process could not be identified.")
        return stopped


def _after_update(progress: Optional[Callable[[str], None]],
                  should_stop: Optional[Callable[[], bool]]) -> None:
    """Restart only a server this process launched; report external restarts."""
    global _managed_server_process
    binary = _version_tuple(_binary_version())
    running = _version_tuple(_server_version())
    if not (binary and running and binary > running):
        return
    process = _managed_server_process
    if process is not None and process.poll() is None:
        _progress(progress, "Restarting the managed Ollama server to apply the update...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        _managed_server_process = None
        _start_server(progress, should_stop)
    else:
        _progress(progress, "Ollama was updated, but an existing server is still running the previous version. Restart Ollama to use the update.")


def _iter_json_lines(response: requests.Response,
                     should_stop: Optional[Callable[[], bool]]):
    for line in response.iter_lines():
        _check_stop(should_stop)
        if line:
            try:
                item = json.loads(line)
            except ValueError as exc:
                raise OllamaPullError(f"Invalid JSON from Ollama: {line[:150]!r}") from exc
            if not isinstance(item, dict):
                raise OllamaPullError("Unexpected Ollama stream item.")
            if item.get("error"):
                raise OllamaPullError(str(item["error"]))
            yield item
    _check_stop(should_stop)


def _watch_stop(response: requests.Response,
                should_stop: Optional[Callable[[], bool]]) -> threading.Event:
    """Close a blocked stream promptly when the user presses Stop."""
    finished = threading.Event()
    if should_stop is not None:
        def monitor() -> None:
            while not finished.wait(0.2):
                try:
                    if should_stop():
                        response.close()
                        return
                except Exception:
                    return
        threading.Thread(target=monitor, daemon=True).start()
    return finished


def _pull_eta(seconds: float) -> str:
    remaining = max(0, int(seconds + 0.5))
    if remaining >= 3600:
        return f"{remaining // 3600}h {(remaining % 3600) // 60}m"
    if remaining >= 60:
        return f"{remaining // 60}m {remaining % 60}s"
    return f"{remaining}s"


def pull_model(model_name: str, *, progress: Optional[Callable[[str], None]] = None,
               should_stop: Optional[Callable[[], bool]] = None) -> None:
    name = native_model_name(model_name)
    _check_stop(should_stop)
    _progress(progress, f"Pulling Ollama model {name}...")
    try:
        with requests.post(f"{base_url()}/api/pull", json={"model": name, "stream": True},
                           stream=True, timeout=(5, 30)) as response:
            finished = _watch_stop(response, should_stop)
            try:
                if not response.ok:
                    raise _http_error(response, "model pull")
                completed = False
                last_status = ""
                transfer_key = ""
                samples = deque(maxlen=32)
                for item in _iter_json_lines(response, should_stop):
                    status = str(item.get("status") or "")
                    total = item.get("total")
                    current = item.get("completed")
                    if isinstance(total, (float, int)) and total > 0 and isinstance(current, (float, int)):
                        status = f"{status} {current / total:.0%}"
                        key = str(item.get("digest") or item.get("status") or "")
                        if key != transfer_key or (samples and current < samples[-1][1]):
                            samples.clear()
                            transfer_key = key
                        now = time.monotonic()
                        samples.append((now, current))
                        while len(samples) > 2 and now - samples[0][0] > 8:
                            samples.popleft()
                        if len(samples) >= 2:
                            elapsed = now - samples[0][0]
                            transferred = current - samples[0][1]
                            if elapsed >= 0.25 and transferred > 0:
                                bytes_per_second = transferred / elapsed
                                status += f" · {bytes_per_second / 1_000_000:.1f} MB/s"
                                status += f" · ETA {_pull_eta((total - current) / bytes_per_second)}"
                    if status and status != last_status:
                        _progress(progress, f"{name}: {status}")
                        last_status = status
                    if item.get("status") == "success":
                        completed = True
                if not completed:
                    raise OllamaPullError(f"Ollama closed the pull stream before {name} completed.")
            finally:
                finished.set()
    except requests.RequestException as exc:
        _check_stop(should_stop)
        raise OllamaPullError(f"Could not pull Ollama model {name}: {exc}") from exc
    try:
        from model_options import refresh_ollamapull_model_catalog
        refresh_ollamapull_model_catalog(timeout=3)
    except Exception:
        pass  # Catalog persistence is best effort; the model itself is ready.


def ensure_ready(model_name: Optional[str] = None, *, auto_update: Optional[bool] = None,
                 progress: Optional[Callable[[str], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None) -> None:
    """Install/update Ollama, start its server, and pull a missing model."""
    with _lifecycle_lock:
        _check_stop(should_stop)
        installed = bool(_ollama_executable()) if mobile_runtime.subprocesses_available() else False
        if not installed:
            try:
                _api_get("/api/version", timeout=1 if mobile_runtime.subprocesses_available() else 5)
                installed = True
            except (requests.RequestException, OllamaPullError, ValueError):
                pass
        if not installed:
            _run_installer(progress, should_stop)
            _after_update(progress, should_stop)
        else:
            enabled = load_settings().get("auto_update", True) if auto_update is None else auto_update
            enabled = enabled and mobile_runtime.subprocesses_available()  # a phone never updates the PC's Ollama
            if enabled:
                current_version = _installed_version()
                available_version = latest_version()
                current = _version_tuple(current_version)
                available = _version_tuple(available_version)
                update_key = (str(current_version), str(available_version))
                if (current and available and available > current
                        and update_key not in _attempted_auto_updates
                        and _active_chats == 0):
                    _attempted_auto_updates.add(update_key)
                    _progress(progress, "Updating Ollama to the latest version...")
                    try:
                        _run_installer(progress, should_stop)
                        _after_update(progress, should_stop)
                    except OllamaPullCancelled:
                        raise
                    except OllamaPullError as exc:
                        _progress(progress, f"Ollama update failed; using the installed version: {exc}")
        _start_server(progress, should_stop)
        if model_name and not _model_is_installed(model_name, _models()):
            pull_model(model_name, progress=progress, should_stop=should_stop)


def _native_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    native: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {"role": message.get("role", "user")}
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts: list[str] = []
            images: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text_parts.append(str(part.get("text") or ""))
                elif part.get("type") == "image_url":
                    url = part.get("image_url")
                    if isinstance(url, dict):
                        url = url.get("url")
                    if not isinstance(url, str) or not url.startswith("data:") or "," not in url:
                        raise OllamaPullError("Ollama image messages require base64 data URLs.")
                    images.append(url.split(",", 1)[1])
            item["content"] = "\n".join(text_parts)
            if images:
                item["images"] = images
        else:
            item["content"] = str(content or "")
        for key in ("tool_calls", "tool_call_id", "name"):
            if key in message:
                item[key] = message[key]
        native.append(item)
    return native


# ---- token ban -----------------------------------------------------------------------------------------------
# "Block other scripts" must stop banned tokens while sampling, like the on-device engine on mobile. Ollama's API
# has no logit_bias (and ignores it), but it runs models with the llama-server it ships, which has one. So a banned
# request goes to our own llama-server on the same model file, with the ban built from the file's vocabulary.

#: Ollama options that are this request's sampling parameters, under the same names in llama-server's request.
_LLAMA_REQUEST_OPTIONS = ("temperature", "top_k", "top_p", "min_p", "typical_p", "repeat_last_n", "repeat_penalty",
                          "presence_penalty", "frequency_penalty", "seed", "stop", "mirostat", "mirostat_tau",
                          "mirostat_eta", "tfs_z")
#: Ollama options set when the model loads -> llama-server's flag.
_LLAMA_LOAD_OPTIONS = {"num_gpu": "-ngl", "num_thread": "-t", "num_batch": "-b"}
_LLAMA_FALLBACK_CONTEXT = 32768
#: Frees the GPU like Ollama's keep-alive; the next request wakes it.
_LLAMA_IDLE_SECONDS = 300
_LLAMA_START_SECONDS = 300
_ban_server_lock = threading.RLock()
_ban_server: Optional[dict[str, Any]] = None  # {"process", "url", "key"}
_ban_backends: dict[str, dict[str, str]] = {}  # llama-server path -> environment of its GPU backend
_ban_contexts: dict[str, int] = {}  # model -> the context size Ollama chose for it
_ban_bias_cache: dict[tuple, list] = {}  # (model file, scripts, kaomoji) -> llama-server's logit_bias
_ban_needs_cache: dict[tuple, dict] = {}  # the same key -> banned token -> what it needs from a source


def _llama_server_executable() -> Optional[str]:
    """The llama-server inside this Ollama install, None when it has none (an older Ollama)."""
    executable = _ollama_executable()
    if not executable:
        return None
    folder = Path(executable).resolve().parent
    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    for candidate in (folder / "lib" / "ollama" / name, folder.parent / "lib" / "ollama" / name, folder / name):
        if candidate.is_file():
            return str(candidate)
    return None


def _llama_backend_env(server: str) -> dict[str, str]:
    """The environment that gives ``server`` a GPU: Ollama keeps each backend in its own folder and loads the one
    that works, the fastest first; none means the CPU."""
    if not mobile_runtime.subprocesses_available():
        return {}
    if server in _ban_backends:
        return _ban_backends[server]
    folder = Path(server).parent
    chosen: dict[str, str] = {}
    for variant in ("cuda_v13", "cuda_v12", "rocm_v7_1", "vulkan"):
        libraries = sorted((folder / variant).glob("*ggml-*")) if (folder / variant).is_dir() else []
        if not libraries:
            continue
        env = {"GGML_BACKEND_PATH": str(libraries[0]),
               "PATH": os.pathsep.join((str(folder / variant), str(folder), os.environ.get("PATH", "")))}
        try:
            listed = subprocess.run([server, "--list-devices"], env={**os.environ, **env}, capture_output=True,
                                    text=True, errors="replace", timeout=30, **_hidden_subprocess_kwargs())
        except (OSError, subprocess.SubprocessError):
            continue
        if re.search(r"^\s*\w+\d+:", (listed.stdout or "") + (listed.stderr or ""), re.MULTILINE):
            chosen = env
            break
    _ban_backends[server] = chosen
    return chosen


def _gguf_tokens(path: str) -> list[str]:
    """``tokenizer.ggml.tokens`` of a GGUF file (it is read from the header, the tensors are not touched)."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with open(path, "rb") as handle:
        def number(layout: str) -> int:
            return struct.unpack(layout, handle.read(struct.calcsize(layout)))[0]

        def text() -> bytes:
            return handle.read(number("<Q"))

        def skip(kind: int) -> None:
            if kind == 8:
                handle.seek(number("<Q"), 1)
            elif kind == 9:
                inner, count = number("<I"), number("<Q")
                for _ in range(count if inner in (8, 9) else 0):
                    skip(inner)
                if inner not in (8, 9):
                    handle.seek(sizes[inner] * count, 1)
            else:
                handle.seek(sizes[kind], 1)

        if handle.read(4) != b"GGUF":
            raise OllamaPullError(f"{path} is not a GGUF model file.")
        number("<I"), number("<Q")  # version, tensors
        for _ in range(number("<Q")):
            key, kind = text(), number("<I")
            if key == b"tokenizer.ggml.tokens" and kind == 9:
                number("<I")
                return [text().decode("utf-8", "replace") for _ in range(number("<Q"))]
            skip(kind)
    raise OllamaPullError(f"{path} has no vocabulary.")


def _token_ban_bias(model_file: str, scripts: tuple, allow_kaomoji: bool, exempt_chars=frozenset()) -> list:
    """llama-server's ``logit_bias`` that forbids every token of ``scripts`` (``[id, false]``), the same tokens
    the on-device engine masks (``script_ban_core.banned_token_ids``), except those that only need
    ``exempt_chars`` (the source of an Extract Glossary request)."""
    import script_ban_core

    key = (model_file, scripts, allow_kaomoji)
    with _ban_server_lock:
        if key not in _ban_bias_cache:
            tokens = _gguf_tokens(model_file)
            byte_level = script_ban_core.is_byte_level(tokens)
            pieces = [script_ban_core.piece_bytes(token, byte_level) for token in tokens]
            banned = script_ban_core.banned_token_ids(pieces, scripts, allow_kaomoji=allow_kaomoji)
            _ban_bias_cache[key] = [[token, False] for token in banned]
            _ban_needs_cache[key] = {token: script_ban_core.token_requirements(pieces[token], allow_kaomoji)
                                     for token in banned}
        if not exempt_chars:
            return _ban_bias_cache[key]
        keys = script_ban_core.source_keys(exempt_chars)
        mask = script_ban_core.script_mask(scripts)
        return [[token, False] for token, needs in _ban_needs_cache[key].items()
                if not script_ban_core.needs_met(needs, keys, mask)]


def _model_files(name: str) -> tuple[str, str]:
    """(model file, vision projector or '') of a downloaded model, from Ollama's own record of it."""
    shown = _api_post("/api/show", {"model": name}, timeout=30.0)
    files = [line[5:].strip() for line in str(shown.get("modelfile") or "").splitlines() if line.startswith("FROM ")]
    files = [path for path in files if Path(path).is_file()]
    if not files:
        raise OllamaPullError(f"Ollama does not list a model file for {name}.")
    return files[0], (files[1] if len(files) > 1 else "")


def _ollama_context(name: str) -> int:
    """The context size Ollama loads ``name`` with (it picks by the GPU's memory): loaded once to ask it."""
    if name not in _ban_contexts:
        context = 0
        try:
            _api_post("/api/generate", {"model": name}, timeout=300.0)
            for loaded in _api_get("/api/ps").get("models") or []:
                if isinstance(loaded, dict) and loaded.get("name") == name:
                    context = int(loaded.get("context_length") or 0)
        except (OllamaPullError, requests.RequestException, TypeError, ValueError):
            pass
        _ban_contexts[name] = context or _LLAMA_FALLBACK_CONTEXT
    return _ban_contexts[name]


def _stop_ban_server() -> None:
    """Stop our llama-server, if it runs."""
    global _ban_server
    with _ban_server_lock:
        server, _ban_server = _ban_server, None
    if server is None:
        return
    process = server["process"]
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


atexit.register(_stop_ban_server)


def _start_ban_server(name: str, options: dict[str, Any], vision: bool,
                      progress: Optional[Callable[[str], None]],
                      should_stop: Optional[Callable[[], bool]]) -> tuple[str, str]:
    """(llama-server's URL, model file): the one running with these load settings, else a new one. Ollama's copy
    of the model is unloaded first, or the GPU would hold it twice."""
    global _ban_server
    if not mobile_runtime.subprocesses_available():
        raise OllamaPullError("Block other scripts needs a llama-server, which a phone cannot start.")
    executable = _llama_server_executable()
    if not executable:
        raise OllamaPullError("Block other scripts needs the llama-server of a current Ollama, and this install "
                              "has none. Update Ollama, or turn the toggle off in Ollama Settings.")
    model_file, projector = _model_files(name)
    context = int(options.get("num_ctx") or 0) or _ollama_context(name)
    args = ["-c", str(context)]
    for option, flag in _LLAMA_LOAD_OPTIONS.items():
        if options.get(option) is not None:
            args += [flag, str(options[option])]
    if vision and projector:
        args += ["--mmproj", projector]
    key = (model_file, tuple(args))
    with _ban_server_lock:
        if _ban_server is not None and _ban_server["key"] == key and _ban_server["process"].poll() is None:
            return _ban_server["url"], model_file
        _stop_ban_server()
        try:
            _api_post("/api/generate", {"model": name, "keep_alive": 0}, timeout=60.0)
        except (OllamaPullError, requests.RequestException):
            pass
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        _chat_status(progress, f"Starting llama-server for {name} (Block other scripts)...")
        process = subprocess.Popen(
            [executable, "--model", model_file, "--port", str(port), "--host", "127.0.0.1", "--no-webui", "--offline",
             "-np", "1", "--sleep-idle-seconds", str(_LLAMA_IDLE_SECONDS), *args],
            env={**os.environ, **_llama_backend_env(executable)},
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            **_hidden_subprocess_kwargs())
        url = f"http://127.0.0.1:{port}"
        _ban_server = {"process": process, "url": url, "key": key}
        deadline = time.monotonic() + _LLAMA_START_SECONDS
        while time.monotonic() < deadline:
            _check_stop(should_stop)
            if process.poll() is not None:
                _ban_server = None
                raise OllamaPullError(f"llama-server stopped while loading {name} (exit code {process.returncode}).")
            try:
                if requests.get(f"{url}/health", timeout=2).ok:
                    return url, model_file
            except requests.RequestException:
                pass
            time.sleep(0.25)
        _stop_ban_server()
        raise OllamaPullError(f"llama-server did not load {name} in {_LLAMA_START_SECONDS} seconds.")


def _token_ban_request(name: str, messages: list[dict[str, Any]], payload: dict[str, Any], scripts: tuple,
                       allow_kaomoji: bool, progress: Optional[Callable[[str], None]],
                       should_stop: Optional[Callable[[], bool]], exempt_chars=frozenset()) -> tuple[str, dict[str, Any]]:
    """(URL, body) of the banned request: the native ``payload`` as llama-server's chat request."""
    options = payload.get("options") or {}
    vision = any(isinstance(part, dict) and part.get("type") == "image_url"
                 for message in messages if isinstance(message.get("content"), list) for part in message["content"])
    url, model_file = _start_ban_server(name, options, vision, progress, should_stop)
    body: dict[str, Any] = {"messages": messages, "stream": bool(payload.get("stream")),
                            "logit_bias": _token_ban_bias(model_file, scripts, allow_kaomoji, exempt_chars)}
    if body["stream"]:
        body["stream_options"] = {"include_usage": True}
    body.update({key: options[key] for key in _LLAMA_REQUEST_OPTIONS if options.get(key) is not None})
    if options.get("num_predict") is not None:
        body["max_tokens"] = options["num_predict"]
    # Off unless Ollama Settings asks for it: qwen3.5:2b spent every token (24000) thinking under the ban and never
    # answered, in all three runs with the model's default; with thinking off all six runs were complete.
    think = payload.get("think")
    body["chat_template_kwargs"] = {"enable_thinking": think is not None and think is not False}
    response_format = payload.get("format")
    if isinstance(response_format, dict):
        body["json_schema"] = response_format
    elif response_format == "json":
        body["response_format"] = {"type": "json_object"}
    return f"{url}/v1/chat/completions", body


def _llama_item(data: dict[str, Any], delta_key: str, done: bool = False) -> dict[str, Any]:
    """A llama-server chat reply (or stream chunk) in the shape of Ollama's ``/api/chat``."""
    choices = data.get("choices") or [{}]
    delta = choices[0].get(delta_key) or {}
    usage = data.get("usage") or {}
    item: dict[str, Any] = {
        "message": {"content": delta.get("content") or "", "thinking": delta.get("reasoning_content") or ""},
        "done": done,
        "done_reason": choices[0].get("finish_reason"),
    }
    if usage:
        item["prompt_eval_count"] = usage.get("prompt_tokens")
        item["eval_count"] = usage.get("completion_tokens")
    return item


def _iter_llama_stream(response: requests.Response, should_stop: Optional[Callable[[], bool]]):
    """llama-server's chat stream as Ollama's stream items (``_iter_json_lines``)."""
    final: dict[str, Any] = {"message": {}, "done": True, "done_reason": "stop"}
    for line in response.iter_lines():
        _check_stop(should_stop)
        text = line.decode("utf-8", "replace").strip() if isinstance(line, bytes) else str(line or "").strip()
        if not text.startswith("data:"):
            continue
        text = text[5:].strip()
        if text == "[DONE]":
            break
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise OllamaPullError(f"Invalid JSON from llama-server: {text[:150]!r}") from exc
        if not isinstance(data, dict):
            raise OllamaPullError("Unexpected llama-server stream item.")
        if data.get("error"):
            raise OllamaPullError(str(data["error"]))
        item = _llama_item(data, "delta")
        if item["done_reason"]:
            final["done_reason"] = item["done_reason"]
        for key in ("prompt_eval_count", "eval_count"):
            if item.get(key) is not None:
                final[key] = item[key]
        if item["message"]["content"] or item["message"]["thinking"]:
            yield item
    _check_stop(should_stop)
    yield final


def chat(model_name: str, messages: list[dict[str, Any]], *, temperature: Optional[float] = None,
         max_tokens: Optional[int] = None, stream: bool = True, ban_scripts: tuple = (), allow_kaomoji: bool = False,
         ban_exempt_chars=frozenset(), repeat_watch: Any = None,
         log_stream: bool = False, log_thinking: bool = False,
         progress: Optional[Callable[[str], None]] = None,
         should_stop: Optional[Callable[[], bool]] = None,
         on_response_open: Optional[Callable[[requests.Response], None]] = None,
         on_response_close: Optional[Callable[[requests.Response], None]] = None) -> dict[str, Any]:
    """Send one native chat request and return text, finish reason, and usage.

    ``ban_scripts``: the ``script_ban_core.SCRIPTS`` the answer may not use: the request goes to llama-server with a
    token ban (``_token_ban_request``), except tokens that only need ``ban_exempt_chars``. Only on this PC: a
    remote Ollama's model file is out of reach. ``repeat_watch`` (``forced_truncation.RepetitionWatch``): the answer
    stops with finish reason ``length`` when it says so; the request streams even when the logs do not."""
    name = native_model_name(model_name)
    _check_stop(should_stop)
    stream = bool(stream) or repeat_watch is not None
    settings = model_settings(name)
    options: dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    configured_options = settings.get("options") or {}
    if not isinstance(configured_options, dict):
        raise OllamaPullError(f"Options for {name} must be a JSON object.")
    options.update(configured_options)
    payload: dict[str, Any] = {
        "model": name, "messages": _native_messages(messages), "stream": bool(stream),
        "options": options,
    }
    for key in ("think", "keep_alive", "format"):
        if key in settings and settings[key] is not None:
            payload[key] = settings[key]
    request_options = settings.get("request") or {}
    if not isinstance(request_options, dict):
        raise OllamaPullError(f"Additional request fields for {name} must be a JSON object.")
    forbidden = {"model", "messages", "stream"}
    overlapping = forbidden.intersection(request_options)
    if overlapping:
        raise OllamaPullError(f"Additional request fields cannot override {', '.join(sorted(overlapping))}.")
    if "options" in request_options:
        extra_options = request_options["options"]
        if not isinstance(extra_options, dict):
            raise OllamaPullError(f"Additional options for {name} must be a JSON object.")
        payload["options"].update(extra_options)
    payload.update({key: value for key, value in request_options.items() if key != "options"})
    response: Optional[requests.Response] = None
    finished: Optional[threading.Event] = None
    global _active_chats
    with _lifecycle_lock:
        _active_chats += 1
    first_event = threading.Event()
    request_started = time.monotonic()
    _chat_status(progress, f"Sending request for {name}; waiting for Ollama to respond.")
    if progress is not None:
        def report_wait() -> None:
            while not first_event.wait(15):
                try:
                    if should_stop is not None and should_stop():
                        return
                except Exception:
                    return
                elapsed = int(time.monotonic() - request_started)
                _chat_status(progress, f"Still waiting for an Ollama response ({elapsed}s); model loading or prompt processing may be underway.")
        threading.Thread(target=report_wait, daemon=True).start()
    try:
        token_ban = bool(ban_scripts)
        if token_ban and (base_url() != BASE_URL or not mobile_runtime.subprocesses_available()):
            token_ban = False
            _chat_status(progress, "Block other scripts is not applied: the token ban needs Ollama on this device.")
        chat_url = f"{base_url()}/api/chat"
        if token_ban:
            chat_url, payload = _token_ban_request(name, messages, payload, tuple(ban_scripts), allow_kaomoji,
                                                   progress, should_stop, ban_exempt_chars)
        else:
            _stop_ban_server()  # its GPU memory is Ollama's again
        response = requests.post(chat_url, json=payload, stream=stream, timeout=(5, 600))
        finished = _watch_stop(response, should_stop)
        if on_response_open:
            on_response_open(response)
        if not response.ok:
            raise _http_error(response, "chat")
        chunks: list[str] = []
        final: dict[str, Any] = {}
        if stream:
            thinking_log_buffer = ""
            content_log_buffer = ""
            saw_thinking = False
            saw_content = False
            stream_items = _iter_llama_stream if token_ban else _iter_json_lines
            for item in stream_items(response, should_stop):
                message = item.get("message")
                content = message.get("content") if isinstance(message, dict) else None
                thinking = message.get("thinking") if isinstance(message, dict) else None
                if thinking or content or item.get("done"):
                    first_event.set()
                if thinking:
                    if not saw_thinking:
                        saw_thinking = True
                        _chat_status(progress, "Ollama has started generating reasoning; response text has not begun.")
                        if log_stream and log_thinking:
                            print("🧠 [Ollama] Thinking...", flush=True)
                    if log_stream and log_thinking:
                        thinking_log_buffer = _stream_log_fragment(
                            thinking_log_buffer, str(thinking), prefix="    ")
                if content:
                    if not saw_content:
                        saw_content = True
                        if thinking_log_buffer:
                            _stream_log_fragment(thinking_log_buffer, "", prefix="    ", force=True)
                            thinking_log_buffer = ""
                        if saw_thinking and log_stream and log_thinking:
                            print("🧠 [Ollama] Thinking complete.", flush=True)
                        _chat_status(progress, "First text token received; Ollama text streaming has begun.")
                        if log_stream:
                            print("📡 [Ollama] Text streaming...", flush=True)
                    chunks.append(str(content))
                    if log_stream:
                        content_log_buffer = _stream_log_fragment(content_log_buffer, str(content))
                if item.get("done"):
                    final = item
                elif repeat_watch is not None and repeat_watch.feed(content or thinking):
                    final = {**item, "done": True, "done_reason": "length"}
                    response.close()
                    break
            if log_stream:
                if thinking_log_buffer:
                    _stream_log_fragment(thinking_log_buffer, "", prefix="    ", force=True)
                if content_log_buffer:
                    _stream_log_fragment(content_log_buffer, "", force=True)
            if not final:
                raise OllamaPullError("Ollama chat stream ended without a completion event.")
        else:
            final = response.json()
            first_event.set()
            if not isinstance(final, dict):
                raise OllamaPullError("Unexpected Ollama chat response.")
            if token_ban:
                final = _llama_item(final, "message", done=True)
            message = final.get("message") or {}
            if isinstance(message, dict):
                chunks.append(str(message.get("content") or ""))
        _check_stop(should_stop)
        reason = str(final.get("done_reason") or "stop")
        if reason in {"length", "max_tokens"}:
            reason = "length"
        usage = {
            "prompt_tokens": int(final.get("prompt_eval_count") or 0),
            "completion_tokens": int(final.get("eval_count") or 0),
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        completion_label = "Ollama stream complete" if stream else "Ollama response complete"
        if final.get("eval_count") is not None:
            _chat_status(progress, f"{completion_label} ({usage['completion_tokens']} generated tokens).")
        else:
            _chat_status(progress, f"{completion_label}.")
        return {"content": "".join(chunks), "finish_reason": reason, "usage": usage, "raw_response": final}
    except requests.RequestException as exc:
        _check_stop(should_stop)
        raise OllamaPullError(f"Ollama chat failed: {exc}") from exc
    finally:
        first_event.set()
        if finished is not None:
            finished.set()
        try:
            if response is not None:
                try:
                    if on_response_close:
                        on_response_close(response)
                finally:
                    response.close()
        finally:
            with _lifecycle_lock:
                _active_chats -= 1
