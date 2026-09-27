"""Managed local Ollama transport for the ``ollamapull/`` model route.

Only the fixed loopback Ollama API is used. Installation uses Ollama's official
platform installers; the model and chat paths use its native HTTP API.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
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


BASE_URL = "http://127.0.0.1:11434"
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


def _stream_log_fragment(pending: str, fragment: str, *, prefix: str = "",
                         force: bool = False) -> str:
    """Emit readable live log lines while retaining a short incomplete line."""
    pending += fragment
    while "\n" in pending:
        line, pending = pending.split("\n", 1)
        if line.strip():
            print(f"{prefix}{line}", flush=True)
    if pending and (force or len(pending) >= 160):
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
    response = requests.get(f"{BASE_URL}{path}", timeout=timeout)
    if not response.ok:
        raise _http_error(response, path)
    data = response.json()
    if not isinstance(data, dict):
        raise OllamaPullError(f"Unexpected Ollama response from {path}.")
    return data


def _api_post(path: str, payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    response = requests.post(f"{BASE_URL}{path}", json=payload, timeout=timeout)
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


def _binary_version() -> Optional[str]:
    executable = _ollama_executable()
    if not executable:
        return None
    try:
        result = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=5)
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
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, creationflags=flags,
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
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
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
        with requests.post(f"{BASE_URL}/api/pull", json={"model": name, "stream": True},
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
        installed = bool(_ollama_executable())
        if not installed:
            try:
                _api_get("/api/version", timeout=1)
                installed = True
            except (requests.RequestException, OllamaPullError, ValueError):
                pass
        if not installed:
            _run_installer(progress, should_stop)
            _after_update(progress, should_stop)
        else:
            enabled = load_settings().get("auto_update", True) if auto_update is None else auto_update
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


def chat(model_name: str, messages: list[dict[str, Any]], *, temperature: Optional[float] = None,
         max_tokens: Optional[int] = None, stream: bool = True,
         log_stream: bool = False, log_thinking: bool = False,
         progress: Optional[Callable[[str], None]] = None,
         should_stop: Optional[Callable[[], bool]] = None,
         on_response_open: Optional[Callable[[requests.Response], None]] = None,
         on_response_close: Optional[Callable[[requests.Response], None]] = None) -> dict[str, Any]:
    """Send one native chat request and return text, finish reason, and usage."""
    name = native_model_name(model_name)
    _check_stop(should_stop)
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
        response = requests.post(f"{BASE_URL}/api/chat", json=payload, stream=stream, timeout=(5, 600))
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
            for item in _iter_json_lines(response, should_stop):
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
