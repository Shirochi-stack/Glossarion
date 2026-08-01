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
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

_CANCEL_EVENT = threading.Event()
_ACTIVE_PROCESSES: set[subprocess.Popen] = set()
_ACTIVE_LOCK = threading.RLock()
_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_PLUGIN_PACKAGE = str(os.environ.get("OCAGY_PLUGIN_PACKAGE", "opencode-antigravity-auth@latest") or "opencode-antigravity-auth@latest").strip()


class OcAgyError(RuntimeError):
    """Raised when OpenCode or the Antigravity OAuth plugin cannot complete a request."""


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

    searched = "\n".join(f"  - {item}" for item in candidates)
    raise OcAgyError(
        "OpenCode CLI was not found. Install the OpenCode terminal CLI, add it to PATH, "
        "or set OCAGY_CLI_PATH. On Windows, Scoop or Chocolatey is recommended.\n"
        "Searched:\n" + searched
    )


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


def _classify_error(detail: str, returncode: int) -> OcAgyError:
    text = _clean_text(detail) or f"opencode exited with code {returncode}"
    lower = text.lower()
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


def send_chat_completion(
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

