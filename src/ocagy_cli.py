# ocagy_cli.py - OpenCode Antigravity OAuth provider for Glossarion
"""Use NoeFabris/opencode-antigravity-auth through the OpenCode CLI.

This provider intentionally does not reimplement the plugin's private Google API
protocol.  Glossarion launches ``opencode run`` in an isolated, tool-disabled
workspace; OpenCode loads ``opencode-antigravity-auth`` and the plugin owns OAuth,
token refresh, quota handling, and multi-account rotation.

Glossarion model prefix:
    ocagy/<friendly-model>

Examples:
    ocagy/gemini-3.1-pro-high
    ocagy/gemini-3.1-pro-low
    ocagy/gemini-3-flash-high

Environment variables:
    OCAGY_CLI_PATH          Explicit path to opencode/opencode.exe.
    OPENCODE_CLI_PATH       Alternate explicit path.
    OCAGY_WORKSPACE         Base workspace used for isolated CLI requests.
    OCAGY_PLUGIN_PACKAGE    Plugin package (default: opencode-antigravity-auth@latest).
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

_CANCEL_EVENT = threading.Event()
_ACTIVE_PROCESSES: set[subprocess.Popen] = set()
_ACTIVE_LOCK = threading.RLock()
_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_PLUGIN_PACKAGE = str(os.environ.get("OCAGY_PLUGIN_PACKAGE", "opencode-antigravity-auth@latest") or "opencode-antigravity-auth@latest").strip()
OPENCODE_NPM_INSTALL_COMMAND = "npm install -g opencode-ai"
NODEJS_WINGET_INSTALL_COMMAND = "winget install --id OpenJS.NodeJS.LTS --exact"


class OcAgyError(RuntimeError):
    """Raised when OpenCode or the Antigravity OAuth plugin cannot complete a request."""


def get_install_instructions() -> str:
    """Return concise, copyable OpenCode setup instructions."""
    if os.name == "nt":
        return (
            "OpenCode CLI was not found.\n\n"
            "Open a new PowerShell window and run:\n"
            f"  {OPENCODE_NPM_INSTALL_COMMAND}\n\n"
            "If npm is not installed, install Node.js LTS first (it includes npm):\n"
            f"  {NODEJS_WINGET_INSTALL_COMMAND}\n\n"
            "After installing Node.js, close and reopen PowerShell, run the OpenCode command, "
            "then restart Glossarion. Advanced users may instead set OCAGY_CLI_PATH."
        )
    return (
        "OpenCode CLI was not found. Install Node.js/npm if needed, then run:\n"
        f"  {OPENCODE_NPM_INSTALL_COMMAND}\n\n"
        "Restart Glossarion afterward. Advanced users may instead set OCAGY_CLI_PATH."
    )


def cancel_stream() -> None:
    _CANCEL_EVENT.set()
    with _ACTIVE_LOCK:
        processes = list(_ACTIVE_PROCESSES)
    for proc in processes:
        _terminate_process_tree(proc)


def reset_cancel() -> None:
    if os.environ.get("TRANSLATION_CANCELLED") == "1":
        return
    _CANCEL_EVENT.clear()


def is_cancelled() -> bool:
    return _CANCEL_EVENT.is_set() or os.environ.get("TRANSLATION_CANCELLED") == "1"


def _candidate_paths() -> Iterable[Path]:
    for key in ("OCAGY_CLI_PATH", "OPENCODE_CLI_PATH"):
        raw = str(os.environ.get(key, "") or "").strip().strip('"')
        if raw:
            yield Path(raw).expanduser()

    found = shutil.which("opencode") or shutil.which("opencode.exe") or shutil.which("opencode.cmd")
    if found:
        yield Path(found)

    home = Path.home()
    local_appdata = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))

    # Native installer / common package-manager locations.
    yield home / ".opencode" / "bin" / "opencode.exe"
    yield home / ".opencode" / "bin" / "opencode"
    yield home / ".local" / "bin" / "opencode"
    yield home / ".bun" / "bin" / "opencode.exe"
    yield home / "scoop" / "apps" / "opencode" / "current" / "opencode.exe"
    yield local_appdata / "Programs" / "opencode" / "opencode.exe"
    yield local_appdata / "opencode" / "opencode.exe"
    yield appdata / "npm" / "opencode.cmd"
    yield appdata / "npm" / "node_modules" / "opencode-ai" / "node_modules" / "opencode-windows-x64" / "bin" / "opencode.exe"
    yield Path("C:/ProgramData/chocolatey/bin/opencode.exe")


def find_executable(explicit_path: Optional[str] = None) -> str:
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    candidates.extend(_candidate_paths())

    seen: set[str] = set()
    for candidate in candidates:
        try:
            normalized = str(candidate.resolve(strict=False))
        except Exception:
            normalized = str(candidate)
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return str(candidate.resolve())

    raise OcAgyError(get_install_instructions())


def _workspace_dir() -> Path:
    configured = str(os.environ.get("OCAGY_WORKSPACE", "") or "").strip()
    path = Path(configured).expanduser() if configured else Path.home() / ".glossarion" / "opencode_antigravity"
    path.mkdir(parents=True, exist_ok=True)
    _write_workspace_config(path)
    _ensure_plugin_config()
    return path


def _workspace_config() -> Dict[str, Any]:
    # Project-local config keeps this integration independent from the user's
    # normal OpenCode projects while still reusing the plugin's account store.
    return {
        "$schema": "https://opencode.ai/config.json",
        "plugin": [_PLUGIN_PACKAGE],
        "share": "disabled",
        "permission": {"*": "deny"},
        "agent": {
            "glossarion": {
                "description": "Non-interactive text generation backend for Glossarion",
                "mode": "primary",
                "prompt": (
                    "Act only as a text-generation backend for Glossarion. Never use tools, "
                    "never inspect files or the workspace, never browse, and never explain your process. "
                    "Follow the supplied instructions and return only the requested final text."
                ),
                "permission": {"*": "deny"},
            }
        },
        "provider": {
            "google": {
                "models": {
                    "antigravity-gemini-3.1-pro": {
                        "name": "Gemini 3.1 Pro (Antigravity OAuth)",
                        "limit": {"context": 1048576, "output": 65535},
                        "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
                        "variants": {
                            "low": {"thinkingLevel": "low"},
                            "high": {"thinkingLevel": "high"},
                        },
                    },
                    "antigravity-gemini-3-pro": {
                        "name": "Gemini 3 Pro (Antigravity OAuth)",
                        "limit": {"context": 1048576, "output": 65535},
                        "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
                        "variants": {
                            "low": {"thinkingLevel": "low"},
                            "high": {"thinkingLevel": "high"},
                        },
                    },
                    "antigravity-gemini-3-flash": {
                        "name": "Gemini 3 Flash (Antigravity OAuth)",
                        "limit": {"context": 1048576, "output": 65536},
                        "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
                        "variants": {
                            "minimal": {"thinkingLevel": "minimal"},
                            "low": {"thinkingLevel": "low"},
                            "medium": {"thinkingLevel": "medium"},
                            "high": {"thinkingLevel": "high"},
                        },
                    },
                    "antigravity-claude-sonnet-4-6": {
                        "name": "Claude Sonnet 4.6 (Antigravity OAuth)",
                        "limit": {"context": 200000, "output": 64000},
                        "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
                    },
                    "antigravity-claude-opus-4-6-thinking": {
                        "name": "Claude Opus 4.6 Thinking (Antigravity OAuth)",
                        "limit": {"context": 200000, "output": 64000},
                        "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
                        "variants": {
                            "low": {"thinkingConfig": {"thinkingBudget": 8192}},
                            "max": {"thinkingConfig": {"thinkingBudget": 32768}},
                        },
                    },
                }
            }
        },
    }


def _write_workspace_config(path: Path) -> Path:
    target = path / "opencode.json"
    desired = _workspace_config()
    current: Any = None
    if target.is_file():
        try:
            current = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            current = None
    if current != desired:
        target.write_text(json.dumps(desired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def _config_dir() -> Path:
    configured = str(os.environ.get("OPENCODE_CONFIG_DIR", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    modern = Path.home() / ".config" / "opencode"
    legacy = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "opencode"
    if (modern / "antigravity-accounts.json").exists() or not (legacy / "antigravity-accounts.json").exists():
        return modern
    return legacy


def _ensure_plugin_config() -> Dict[str, Any]:
    """Enable PID-based account offsets for Glossarion's parallel workers.

    Existing plugin settings are preserved. If a user's file contains JSONC or
    otherwise cannot be parsed safely, it is left untouched.
    """
    config_dir = _config_dir()
    path = config_dir / "antigravity.json"
    result: Dict[str, Any] = {"plugin_config": str(path), "pid_offset_enabled": False}
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        payload: Any = {}
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
        if "pid_offset_enabled" not in payload:
            payload["pid_offset_enabled"] = True
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["pid_offset_enabled"] = bool(payload.get("pid_offset_enabled"))
    except Exception as exc:
        result["plugin_config_error"] = str(exc)
    return result


def _account_summary() -> Dict[str, Any]:
    path = _config_dir() / "antigravity-accounts.json"
    result: Dict[str, Any] = {"accounts_file": str(path), "account_count": 0, "emails": []}
    if not path.is_file():
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        accounts = payload.get("accounts", []) if isinstance(payload, dict) else []
        enabled = [item for item in accounts if isinstance(item, dict) and item.get("enabled", True)]
        result["account_count"] = len(enabled)
        result["emails"] = [str(item.get("email", "") or "") for item in enabled if item.get("email")]
    except Exception as exc:
        result["accounts_error"] = str(exc)
    return result


def get_account_summary() -> Dict[str, Any]:
    """Return the plugin's local OAuth account summary without exposing tokens."""
    return _account_summary()


def _require_oauth_account() -> None:
    """Fail before launching OpenCode when the plugin has no usable OAuth account."""
    account = _account_summary()
    if account.get("account_count"):
        return

    detail = ""
    if account.get("accounts_error"):
        detail = f" The OAuth account store could not be read: {account['accounts_error']}"
    raise OcAgyError(
        "OpenCode Antigravity OAuth is not configured. The ocagy/ route does not use or require "
        "a Google API key. Click the OpenCode Antigravity Login button, select Google, then "
        "OAuth with Google (Antigravity), finish signing in, and retry. Do not select "
        f"Manually enter API Key.{detail}"
    )


def _creation_flags(*, visible: bool = False) -> int:
    if os.name != "nt":
        return 0
    if visible:
        return getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


def _subprocess_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("NO_COLOR", "1")
    env.setdefault("TERM", "dumb")
    env.setdefault("PYTHONUTF8", "1")
    return env


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _clean_text(value: str) -> str:
    return _ANSI_RE.sub("", str(value or "")).replace("\r\n", "\n").replace("\r", "\n").strip()


def _run_short(args: List[str], timeout: int = 60) -> subprocess.CompletedProcess:
    exe = find_executable()
    workspace = _workspace_dir()
    return subprocess.run(
        [exe, *args],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=_creation_flags(),
        check=False,
        env=_subprocess_env(),
    )


def get_status() -> Dict[str, Any]:
    """Return OpenCode installation, plugin model visibility, and account count."""
    try:
        exe = find_executable()
    except Exception as exc:
        return {"installed": False, "authenticated": False, "plugin_ready": False, "error": str(exc), "models": []}

    version_text = ""
    try:
        version = _run_short(["--version"], timeout=20)
        version_text = _clean_text(version.stdout or version.stderr)
    except Exception as exc:
        version_text = str(exc)

    models: List[str] = []
    models_error = ""
    plugin_ready = False
    try:
        listing = _run_short(["models", "google"], timeout=120)
        combined = _clean_text((listing.stdout or "") + "\n" + (listing.stderr or ""))
        if listing.returncode == 0:
            models = sorted(set(re.findall(r"google/[A-Za-z0-9._-]+", combined)))
            plugin_ready = any("antigravity-" in item for item in models)
        else:
            models_error = combined or f"opencode models exited with code {listing.returncode}"
    except Exception as exc:
        models_error = str(exc)

    account = _account_summary()
    plugin_config = _ensure_plugin_config()
    return {
        "installed": True,
        "authenticated": bool(account.get("account_count")),
        "plugin_ready": plugin_ready,
        "executable": exe,
        "version": version_text,
        "models": models,
        "error": models_error,
        **account,
        **plugin_config,
    }


def launch_login() -> Dict[str, Any]:
    """Open OpenCode's plugin-aware OAuth login in a visible terminal."""
    exe = find_executable()
    workspace = _workspace_dir()
    if os.name == "nt":
        # The project-local opencode.json already declares the plugin and models.
        command = (
            "$Host.UI.RawUI.WindowTitle='Glossarion OpenCode Antigravity Login'; "
            "Write-Host 'Glossarion OpenCode Antigravity OAuth setup' -ForegroundColor Cyan; "
            "Write-Host 'Select Google, then OAuth with Google (Antigravity).' -ForegroundColor Yellow; "
            f"Set-Location -LiteralPath {json.dumps(str(workspace))}; "
            f"& {json.dumps(exe)} auth login; "
            "Write-Host ''; Write-Host 'When login is complete, return to Glossarion and click the status button.' -ForegroundColor Green"
        )
        proc = subprocess.Popen(
            ["powershell.exe", "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", command],
            cwd=str(workspace),
            creationflags=_creation_flags(visible=True),
            env=_subprocess_env(),
        )
    else:
        shell_command = f"cd {shlex_quote(str(workspace))}; {shlex_quote(exe)} auth login; exec $SHELL"
        terminal = shutil.which("x-terminal-emulator") or shutil.which("gnome-terminal") or shutil.which("konsole")
        if terminal:
            if Path(terminal).name == "gnome-terminal":
                proc = subprocess.Popen([terminal, "--", "sh", "-lc", shell_command])
            else:
                proc = subprocess.Popen([terminal, "-e", "sh", "-lc", shell_command])
        else:
            proc = subprocess.Popen([exe, "auth", "login"], cwd=str(workspace), start_new_session=True)
    return {"started": True, "pid": proc.pid, "executable": exe, "workspace": str(workspace)}


def shlex_quote(value: str) -> str:
    import shlex
    return shlex.quote(value)


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                kind = str(item.get("type", "") or "").lower()
                if kind in {"text", "input_text", "output_text"} or "text" in item:
                    parts.append(str(item.get("text", "") or ""))
                elif kind in {"image", "image_url", "input_image"}:
                    parts.append("[Image input omitted: OpenCode Antigravity text mode]")
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text", "") or "")
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def build_prompt(messages: List[Dict[str, Any]]) -> str:
    system_sections: List[str] = []
    conversation: List[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user") or "user").strip().lower()
        text = _message_text(message.get("content"))
        if role in {"system", "developer"}:
            system_sections.append(text)
        else:
            label = "ASSISTANT" if role == "assistant" else "USER"
            conversation.append(f"<{label}>\n{text}\n</{label}>")
    return (
        "You are being called by Glossarion as a non-interactive text-generation backend.\n"
        "Never use tools. Do not inspect files, the current directory, or external resources.\n"
        "Do not explain your process. Return only the final text requested by the instructions.\n\n"
        "<SYSTEM_INSTRUCTIONS>\n"
        + "\n\n".join(section for section in system_sections if section)
        + "\n</SYSTEM_INSTRUCTIONS>\n\n<CONVERSATION>\n"
        + "\n\n".join(conversation)
        + "\n</CONVERSATION>\n"
    )


def resolve_model(model: str) -> Tuple[str, Optional[str]]:
    """Map a friendly Glossarion suffix to OpenCode's provider/model + variant."""
    value = str(model or "").strip().lower()
    if value.startswith("ocagy/"):
        value = value.split("/", 1)[1]
    elif value.startswith("ocagy"):
        value = value[len("ocagy"):].lstrip("/")

    direct = re.match(r"^google/(.+)$", value)
    if direct:
        return "google/" + direct.group(1), None

    mappings: List[Tuple[str, str, Optional[str]]] = [
        ("gemini-3.1-pro-high", "google/antigravity-gemini-3.1-pro", "high"),
        ("gemini-3.1-pro-low", "google/antigravity-gemini-3.1-pro", "low"),
        ("gemini-3-pro-high", "google/antigravity-gemini-3-pro", "high"),
        ("gemini-3-pro-low", "google/antigravity-gemini-3-pro", "low"),
        ("gemini-3-flash-minimal", "google/antigravity-gemini-3-flash", "minimal"),
        ("gemini-3-flash-low", "google/antigravity-gemini-3-flash", "low"),
        ("gemini-3-flash-medium", "google/antigravity-gemini-3-flash", "medium"),
        ("gemini-3-flash-high", "google/antigravity-gemini-3-flash", "high"),
        ("claude-sonnet-4-6", "google/antigravity-claude-sonnet-4-6", None),
        ("claude-opus-4-6-thinking-low", "google/antigravity-claude-opus-4-6-thinking", "low"),
        ("claude-opus-4-6-thinking-max", "google/antigravity-claude-opus-4-6-thinking", "max"),
    ]
    for friendly, model_id, variant in mappings:
        if value == friendly:
            return model_id, variant
    if value.startswith("antigravity-"):
        return "google/" + value, None
    if not value:
        return "google/antigravity-gemini-3.1-pro", "high"
    raise OcAgyError(
        f"Unknown ocagy model '{model}'. Use ocagy/gemini-3.1-pro-high, "
        "ocagy/gemini-3.1-pro-low, or another listed ocagy model."
    )


def _parse_json_events(stdout: str) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    events: List[Dict[str, Any]] = []
    text_parts: List[str] = []
    non_json: List[str] = []
    for raw in _clean_text(stdout).splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            non_json.append(line)
            continue
        if not isinstance(item, dict):
            continue
        events.append(item)
        if item.get("type") == "text":
            part = item.get("part")
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text:
                    text_parts.append(text)
    content = "".join(text_parts).strip()
    return content, events, non_json


def _usage_from_value(value: Any) -> Optional[Dict[str, int]]:
    if isinstance(value, dict):
        candidates = [value]
        for key in ("usage", "usageMetadata", "usage_metadata", "tokens"):
            item = value.get(key)
            if isinstance(item, dict):
                candidates.insert(0, item)
        for candidate in candidates:
            def number(*keys: str) -> int:
                for key in keys:
                    raw = candidate.get(key)
                    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                        return int(raw)
                return 0
            prompt = number("input_tokens", "prompt_tokens", "inputTokens", "promptTokenCount", "input")
            completion = number("output_tokens", "completion_tokens", "outputTokens", "candidatesTokenCount", "output")
            total = number("total_tokens", "totalTokens", "totalTokenCount", "total") or prompt + completion
            cache = number("cache_read_tokens", "cached_input_tokens", "cacheReadTokens")
            if any((prompt, completion, total, cache)):
                result = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}
                if cache:
                    result["cache_read_tokens"] = cache
                return result
        for nested in value.values():
            found = _usage_from_value(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _usage_from_value(nested)
            if found:
                return found
    return None


def _event_error(events: List[Dict[str, Any]]) -> str:
    messages: List[str] = []
    for event in events:
        if event.get("type") != "error":
            continue
        error = event.get("error")
        if isinstance(error, dict):
            data = error.get("data")
            if isinstance(data, dict) and data.get("message"):
                messages.append(str(data.get("message")))
            elif error.get("message"):
                messages.append(str(error.get("message")))
            else:
                messages.append(json.dumps(error, ensure_ascii=False))
        elif error:
            messages.append(str(error))
    return "\n".join(messages)


def _http_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    if not raw:
        return None
    return json.loads(raw.decode("utf-8", errors="replace"))


def _loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _drain_process_pipe(pipe: Any, output: List[str]) -> None:
    try:
        for line in iter(pipe.readline, ""):
            output.append(str(line))
            if len(output) > 500:
                del output[:100]
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _read_sse_events(response: Any, output: "queue.Queue[Tuple[str, Any]]") -> None:
    data_lines: List[str] = []
    try:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                if data_lines:
                    value = "\n".join(data_lines)
                    data_lines.clear()
                    try:
                        event = json.loads(value)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict):
                        output.put(("event", event))
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            try:
                event = json.loads("\n".join(data_lines))
            except json.JSONDecodeError:
                event = None
            if isinstance(event, dict):
                output.put(("event", event))
    except Exception as exc:
        output.put(("error", exc))
    finally:
        output.put(("eof", None))


def _stream_log_text(text: str, buffer: List[str], logger: Callable[[str], None]) -> None:
    """Emit readable line-sized chunks, matching the Antigravity stream logger."""
    if not text:
        return
    combined = "".join(buffer) + text
    for tag in ("</h1>", "</h2>", "</h3>", "</h4>", "</h5>", "</h6>", "</p>"):
        combined = combined.replace(tag, tag + "\n")
    if "\n" in combined:
        parts = combined.split("\n")
        for part in parts[:-1]:
            logger(part)
        buffer[:] = [parts[-1]]
    else:
        buffer[:] = [combined]
        if len(combined) > 150:
            logger(combined)
            buffer.clear()


class _OpenCodeStreamState:
    """Collect authoritative output while exposing incremental SSE deltas."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.assistant_message_ids: set[str] = set()
        self.text_order: List[str] = []
        self.text_by_part: Dict[str, str] = {}
        self.reasoning_by_part: Dict[str, str] = {}
        self.step_events: List[Dict[str, Any]] = []
        self.finish_reason = "stop"
        self.error = ""
        self.complete = False

    @staticmethod
    def _merge_part(
        part: Dict[str, Any],
        delta: Any,
        store: Dict[str, str],
    ) -> str:
        part_id = str(part.get("id", "") or "")
        if not part_id:
            return ""
        previous = store.get(part_id, "")
        full = part.get("text")
        full_text = str(full) if isinstance(full, str) else ""
        delta_text = str(delta) if isinstance(delta, str) else ""
        if full_text:
            fragment = full_text[len(previous):] if full_text.startswith(previous) else delta_text
            if not fragment and full_text != previous:
                fragment = full_text
            store[part_id] = full_text
            return fragment
        if delta_text:
            store[part_id] = previous + delta_text
            return delta_text
        return ""

    @staticmethod
    def _error_text(error: Any) -> str:
        if isinstance(error, dict):
            data = error.get("data")
            if isinstance(data, dict) and data.get("message"):
                return str(data.get("message"))
            return str(error.get("message") or error.get("name") or json.dumps(error, ensure_ascii=False))
        return str(error or "OpenCode session failed")

    def feed(self, event: Dict[str, Any]) -> List[Tuple[str, str]]:
        emitted: List[Tuple[str, str]] = []
        event_type = str(event.get("type", "") or "")
        properties = event.get("properties")
        if not isinstance(properties, dict):
            properties = {}

        if event_type == "message.updated":
            info = properties.get("info")
            if isinstance(info, dict) and info.get("role") == "assistant":
                message_id = str(info.get("id", "") or "")
                if message_id:
                    self.assistant_message_ids.add(message_id)
            return emitted

        if event_type == "message.part.updated":
            part = properties.get("part")
            if not isinstance(part, dict) or part.get("sessionID") != self.session_id:
                return emitted
            message_id = str(part.get("messageID", "") or "")
            if message_id not in self.assistant_message_ids:
                return emitted
            part_type = str(part.get("type", "") or "")
            if part_type == "text":
                part_id = str(part.get("id", "") or "")
                if part_id and part_id not in self.text_order:
                    self.text_order.append(part_id)
                fragment = self._merge_part(part, properties.get("delta"), self.text_by_part)
                if fragment:
                    emitted.append(("text", fragment))
            elif part_type == "reasoning":
                fragment = self._merge_part(part, properties.get("delta"), self.reasoning_by_part)
                if fragment:
                    emitted.append(("reasoning", fragment))
            elif part_type == "step-finish":
                self.step_events.append({"type": "step_finish", "part": part})
                reason = str(part.get("reason", "") or "")
                if reason:
                    self.finish_reason = reason
            return emitted

        if event_type == "session.error":
            event_session = str(properties.get("sessionID", "") or "")
            if not event_session or event_session == self.session_id:
                self.error = self._error_text(properties.get("error"))
                self.complete = True
            return emitted

        if event_type == "session.idle" and properties.get("sessionID") == self.session_id:
            self.complete = True
        elif event_type == "session.status" and properties.get("sessionID") == self.session_id:
            status = properties.get("status")
            if isinstance(status, dict) and status.get("type") == "idle":
                self.complete = True
        return emitted

    def content(self) -> str:
        return "".join(self.text_by_part.get(part_id, "") for part_id in self.text_order)


def _send_via_server(
    *,
    exe: str,
    request_dir: Path,
    prompt: str,
    model_id: str,
    variant: Optional[str],
    timeout_seconds: int,
    logger: Callable[[str], None],
    log_stream: bool,
) -> Dict[str, Any]:
    """Run an isolated OpenCode server and consume true token deltas over SSE."""
    port = _loopback_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [exe, "serve", "--hostname", "127.0.0.1", "--port", str(port)]
    proc = subprocess.Popen(
        command,
        cwd=str(request_dir),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_creation_flags(),
        start_new_session=(os.name != "nt"),
        env=_subprocess_env(),
    )
    with _ACTIVE_LOCK:
        _ACTIVE_PROCESSES.add(proc)

    server_stdout: List[str] = []
    server_stderr: List[str] = []
    for pipe, target in ((proc.stdout, server_stdout), (proc.stderr, server_stderr)):
        if pipe is not None:
            threading.Thread(
                target=_drain_process_pipe,
                args=(pipe, target),
                daemon=True,
            ).start()

    event_response = None
    session_id = ""
    started = time.time()
    try:
        startup_deadline = min(started + timeout_seconds, started + 45)
        while time.time() < startup_deadline:
            if is_cancelled():
                raise OcAgyError("OpenCode Antigravity request cancelled by user")
            if proc.poll() is not None:
                detail = _clean_text("".join(server_stderr + server_stdout))
                raise OcAgyError(
                    "OpenCode Antigravity server exited during startup."
                    + (f" Details: {detail}" if detail else "")
                )
            try:
                health = _http_json(base_url, "/global/health", timeout=0.75)
                if isinstance(health, dict) and health.get("healthy"):
                    break
            except Exception:
                time.sleep(0.1)
        else:
            raise OcAgyError("OpenCode Antigravity server did not become ready within 45 seconds.")

        session = _http_json(
            base_url,
            "/session",
            method="POST",
            payload={"title": "Glossarion translation"},
            timeout=10,
        )
        if not isinstance(session, dict) or not session.get("id"):
            raise OcAgyError("OpenCode Antigravity could not create a streaming session.")
        session_id = str(session["id"])

        event_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        event_request = urllib.request.Request(
            base_url + "/event",
            headers={"Accept": "text/event-stream", "Cache-Control": "no-cache"},
            method="GET",
        )
        event_response = urllib.request.urlopen(event_request, timeout=timeout_seconds)
        threading.Thread(
            target=_read_sse_events,
            args=(event_response, event_queue),
            daemon=True,
        ).start()

        connected = False
        connect_deadline = min(started + timeout_seconds, time.time() + 10)
        pending_events: List[Dict[str, Any]] = []
        while time.time() < connect_deadline:
            try:
                kind, value = event_queue.get(timeout=0.5)
            except queue.Empty:
                if is_cancelled():
                    raise OcAgyError("OpenCode Antigravity request cancelled by user")
                continue
            if kind == "event" and isinstance(value, dict):
                if value.get("type") == "server.connected":
                    connected = True
                    break
                pending_events.append(value)
            elif kind == "error":
                raise OcAgyError(f"OpenCode Antigravity event stream failed: {value}")
            elif kind == "eof":
                break
        if not connected:
            raise OcAgyError("OpenCode Antigravity event stream did not connect.")

        provider_id, model_name = model_id.split("/", 1)
        payload: Dict[str, Any] = {
            "agent": "glossarion",
            "model": {"providerID": provider_id, "modelID": model_name},
            "parts": [{"type": "text", "text": prompt}],
        }
        if variant:
            payload["variant"] = variant
        _http_json(
            base_url,
            f"/session/{session_id}/prompt_async",
            method="POST",
            payload=payload,
            timeout=15,
        )

        state = _OpenCodeStreamState(session_id)
        text_buffer: List[str] = []
        thinking_buffer: List[str] = []
        thinking_started = False
        first_text = False
        stream_thinking = os.getenv("STREAM_THINKING_LOGS", "0").strip().lower() not in (
            "", "0", "false", "no", "off",
        )

        def consume(event: Dict[str, Any]) -> None:
            nonlocal thinking_started, first_text
            for fragment_type, fragment in state.feed(event):
                if fragment_type == "reasoning" and log_stream and stream_thinking:
                    if not thinking_started:
                        thinking_started = True
                        logger("🧠 [ocagy] Thinking...")
                    _stream_log_text(fragment, thinking_buffer, logger)
                elif fragment_type == "text" and log_stream:
                    if not first_text:
                        first_text = True
                        logger(f"OpenCode Antigravity: first token in {time.time() - started:.1f}s, streaming...")
                    _stream_log_text(fragment, text_buffer, logger)

        for event in pending_events:
            consume(event)

        stream_eof = False
        while not state.complete:
            if is_cancelled():
                try:
                    _http_json(
                        base_url,
                        f"/session/{session_id}/abort",
                        method="POST",
                        payload={},
                        timeout=2,
                    )
                except Exception:
                    pass
                raise OcAgyError("OpenCode Antigravity request cancelled by user")
            if time.time() - started >= timeout_seconds:
                raise OcAgyError(
                    f"OpenCode Antigravity timed out after {timeout_seconds}s. "
                    "Increase the API timeout or use a shorter chunk."
                )
            if proc.poll() is not None:
                detail = _clean_text("".join(server_stderr + server_stdout))
                raise OcAgyError(
                    "OpenCode Antigravity server exited before the response completed."
                    + (f" Details: {detail}" if detail else "")
                )
            try:
                kind, value = event_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if kind == "event" and isinstance(value, dict):
                consume(value)
            elif kind == "error":
                raise OcAgyError(f"OpenCode Antigravity event stream failed: {value}")
            elif kind == "eof":
                stream_eof = True
                break

        if stream_eof and not state.complete:
            raise OcAgyError("OpenCode Antigravity event stream closed before completion.")
        if state.error:
            raise _classify_error(state.error, 1)

        if log_stream and thinking_buffer:
            tail = "".join(thinking_buffer).strip()
            if tail:
                logger(f"    {tail}")
        if log_stream and text_buffer:
            tail = "".join(text_buffer).strip()
            if tail:
                logger(tail)

        return {
            "content": state.content(),
            "finish_reason": state.finish_reason,
            "usage": _usage_from_value(state.step_events),
            "raw_response": state.step_events,
        }
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        raise _classify_error(detail, int(getattr(exc, "code", 1) or 1))
    except urllib.error.URLError as exc:
        raise OcAgyError(f"OpenCode Antigravity local streaming connection failed: {exc}")
    finally:
        if event_response is not None:
            try:
                event_response.close()
            except Exception:
                pass
        _terminate_process_tree(proc)
        with _ACTIVE_LOCK:
            _ACTIVE_PROCESSES.discard(proc)


def _classify_error(detail: str, returncode: int) -> OcAgyError:
    text = _clean_text(detail) or f"opencode exited with code {returncode}"
    lower = text.lower()
    if any(x in lower for x in (
        "api key is missing",
        "api key missing",
        "google_generative_ai_api_key",
    )):
        return OcAgyError(
            "OpenCode did not find an Antigravity OAuth session and fell back to its built-in "
            "Google provider. The ocagy/ route does not use or require a Google API key. Click "
            "the OpenCode Antigravity Login button, select Google, then OAuth with Google "
            "(Antigravity), finish signing in, and retry. Do not select Manually enter API Key."
        )
    if any(x in lower for x in ("invalid_grant", "not authenticated", "auth login", "oauth", "credential", "api key missing")):
        return OcAgyError(
            "OpenCode Antigravity authentication failed. Click the OpenCode Antigravity Login button, "
            "select Google → OAuth with Google (Antigravity), then retry. Details: " + text
        )
    if any(x in lower for x in ("rate limit", "rate-limited", "quota", "resource_exhausted", "too many requests", "429")):
        return OcAgyError("OpenCode Antigravity quota/rate limit: " + text)
    if any(x in lower for x in ("model not found", "unknown model", "model does not exist", "cannot be resolved")):
        return OcAgyError("OpenCode Antigravity model error: " + text)
    if any(x in lower for x in ("plugin", "opencode-antigravity-auth", "provider google")):
        return OcAgyError("OpenCode Antigravity plugin/configuration error: " + text)
    return OcAgyError(f"OpenCode Antigravity failed (exit {returncode}): {text}")


def _send_chat_completion_buffered(
    *,
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 65536,
    timeout: int = 1800,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run one isolated ``opencode run --format json`` request via the plugin."""
    if is_cancelled():
        raise OcAgyError("OpenCode Antigravity request cancelled by user")

    exe = find_executable()
    _require_oauth_account()
    model_id, variant = resolve_model(model)
    prompt = build_prompt(messages)
    logger = log_fn or (lambda _message: None)
    timeout_seconds = max(30, int(timeout or 1800))

    base = _workspace_dir()
    request_dir = Path(tempfile.mkdtemp(prefix="request-", dir=str(base)))
    _write_workspace_config(request_dir)
    command = [
        exe,
        "run",
        "--format",
        "json",
        "--model",
        model_id,
        "--agent",
        "glossarion",
        "--dir",
        str(request_dir),
        "--title",
        "Glossarion translation",
    ]
    if variant:
        command.extend(["--variant", variant])

    logger(f"🪐 OpenCode Antigravity: {Path(exe).name} run --model {model_id}" + (f" --variant {variant}" if variant else ""))
    logger(
        "🎛️ OpenCode/plugin manages temperature and output limits internally "
        f"(Glossarion requested temperature={temperature}, max_tokens={max_tokens:,})."
    )

    start = time.time()
    proc = subprocess.Popen(
        command,
        cwd=str(request_dir),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_creation_flags(),
        start_new_session=(os.name != "nt"),
        env=_subprocess_env(),
    )
    with _ACTIVE_LOCK:
        _ACTIVE_PROCESSES.add(proc)

    stdout = ""
    stderr = ""
    try:
        try:
            stdout, stderr = proc.communicate(input=prompt, timeout=1.0)
        except subprocess.TimeoutExpired:
            while True:
                if is_cancelled():
                    _terminate_process_tree(proc)
                    raise OcAgyError("OpenCode Antigravity request cancelled by user")
                if time.time() - start >= timeout_seconds:
                    _terminate_process_tree(proc)
                    raise OcAgyError(
                        f"OpenCode Antigravity timed out after {timeout_seconds}s. "
                        "Increase the API timeout or use a shorter chunk."
                    )
                try:
                    stdout, stderr = proc.communicate(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    continue
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_PROCESSES.discard(proc)
        shutil.rmtree(request_dir, ignore_errors=True)

    elapsed = time.time() - start
    content, events, non_json = _parse_json_events(stdout)
    event_error = _event_error(events)
    if proc.returncode != 0 or event_error:
        detail = "\n".join(part for part in (event_error, _clean_text(stderr), "\n".join(non_json)) if part)
        raise _classify_error(detail, int(proc.returncode or 1))

    if not content:
        # Compatibility fallback for older OpenCode versions that ignore --format json.
        fallback = "\n".join(non_json).strip()
        if fallback:
            content = fallback
    content = _clean_text(content)
    if not content:
        detail = _clean_text(stderr)
        raise OcAgyError("OpenCode Antigravity returned an empty response." + (f" Details: {detail}" if detail else ""))

    usage = _usage_from_value(events)
    logger(f"✅ OpenCode Antigravity: completed in {elapsed:.1f}s")
    return {
        "content": content,
        "finish_reason": "stop",
        "usage": usage,
        "model": model_id,
        "variant": variant,
        "provider": "ocagy",
        "elapsed_seconds": elapsed,
        "stderr": _clean_text(stderr),
        "raw_response": events,
    }


def send_chat_completion(
    *,
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 65536,
    timeout: int = 1800,
    log_fn: Optional[Callable[[str], None]] = None,
    log_stream: bool = True,
) -> Dict[str, Any]:
    """Run one isolated request through OpenCode's live local event API."""
    if is_cancelled():
        raise OcAgyError("OpenCode Antigravity request cancelled by user")

    exe = find_executable()
    _require_oauth_account()
    model_id, variant = resolve_model(model)
    prompt = build_prompt(messages)
    logger = log_fn or (lambda _message: None)
    timeout_seconds = max(30, int(timeout or 1800))

    base = _workspace_dir()
    request_dir = Path(tempfile.mkdtemp(prefix="request-", dir=str(base)))
    _write_workspace_config(request_dir)
    logger(
        f"🪐 OpenCode Antigravity: {Path(exe).name} live stream --model {model_id}"
        + (f" --variant {variant}" if variant else "")
    )
    logger(
        "🎛️ OpenCode/plugin manages temperature and output limits internally "
        f"(Glossarion requested temperature={temperature}, max_tokens={max_tokens:,})."
    )

    start = time.time()
    try:
        result = _send_via_server(
            exe=exe,
            request_dir=request_dir,
            prompt=prompt,
            model_id=model_id,
            variant=variant,
            timeout_seconds=timeout_seconds,
            logger=logger,
            log_stream=bool(log_stream),
        )
    finally:
        shutil.rmtree(request_dir, ignore_errors=True)

    elapsed = time.time() - start
    content = _clean_text(str(result.get("content", "") or ""))
    if not content:
        raise OcAgyError("OpenCode Antigravity returned an empty response.")

    logger(f"✅ OpenCode Antigravity: stream finished in {elapsed:.1f}s")
    return {
        "content": content,
        "finish_reason": str(result.get("finish_reason", "stop") or "stop"),
        "usage": result.get("usage"),
        "model": model_id,
        "variant": variant,
        "provider": "ocagy",
        "elapsed_seconds": elapsed,
        "stderr": "",
        "raw_response": result.get("raw_response"),
    }
