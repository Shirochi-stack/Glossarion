# ocagy_cli.py - OpenCode Antigravity OAuth provider for Glossarion
"""Use NoeFabris/opencode-antigravity-auth through the OpenCode CLI.

This provider intentionally does not reimplement the plugin's private Google API
protocol.  Glossarion launches ``opencode run`` in an isolated, tool-disabled
workspace; OpenCode loads ``opencode-antigravity-auth`` and the plugin owns OAuth,
token refresh, quota handling, and multi-account rotation.

Glossarion model prefixes:
    ocagy0/<friendly-model>  # plugin-managed shared account pool
    ocagy/<friendly-model>   # account 1
    ocagy1/<friendly-model>  # account 2
    ocagy2/<friendly-model>  # account 3, and so on

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
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from installer_utils import run_logged_subprocess

try:
    import httpx
except ImportError:  # pragma: no cover - httpx is a required app dependency
    httpx = None

_CANCEL_EVENT = threading.Event()
_ACTIVE_PROCESSES: set[subprocess.Popen] = set()
_ACTIVE_LOCK = threading.RLock()
_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_PLUGIN_PACKAGE = str(os.environ.get("OCAGY_PLUGIN_PACKAGE", "opencode-antigravity-auth@latest") or "opencode-antigravity-auth@latest").strip()
OPENCODE_NPM_INSTALL_COMMAND = "npm install -g opencode-ai"
NODEJS_WINGET_INSTALL_COMMAND = "winget install --id OpenJS.NodeJS.LTS --exact"
OPENCODE_INSTALL_TIMEOUT_SECONDS = 600
OCAGY_PLUGIN_INSTALL_TIMEOUT_SECONDS = 300
_INSTALL_LOCK = threading.Lock()
_PLUGIN_BOOTSTRAP_LOCK = threading.Lock()
_PLUGIN_BOOTSTRAPPED_EXECUTABLES: set[str] = set()
_ANTIGRAVITY_CLOUD_CODE_BASE = "https://cloudcode-pa.googleapis.com"
_ANTIGRAVITY_FALLBACK_PROJECT_ID = "rising-fact-p41fc"
_ANTIGRAVITY_QUOTA_USER_AGENT = "antigravity/windows/amd64"

# OpenCode lists the plugin's base model IDs, while Glossarion exposes the
# selectable thinking variants as separate ocagy/ entries. Keep the expansion
# here, next to the CLI integration that owns both representations.
_CATALOG_MODEL_VARIANTS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("google/antigravity-gemini-3.1-pro", "gemini-3.1-pro", ("high", "low")),
    ("google/antigravity-gemini-3-pro", "gemini-3-pro", ("high", "low")),
    (
        "google/antigravity-gemini-3-flash",
        "gemini-3-flash",
        ("minimal", "low", "medium", "high"),
    ),
    ("google/antigravity-claude-sonnet-4-6", "claude-sonnet-4-6", ()),
    (
        "google/antigravity-claude-opus-4-6-thinking",
        "claude-opus-4-6-thinking",
        ("low", "max"),
    ),
)


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


def _candidate_command(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found

    if os.name != "nt":
        return None

    home = Path.home()
    appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
    local_appdata = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    program_files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
    candidates: List[Path] = []
    if name in {"npm", "npx"}:
        candidates.extend(
            [
                program_files / "nodejs" / f"{name}.cmd",
                local_appdata / "Programs" / "nodejs" / f"{name}.cmd",
                appdata / "npm" / f"{name}.cmd",
            ]
        )
    elif name == "winget":
        candidates.append(local_appdata / "Microsoft" / "WindowsApps" / "winget.exe")

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def _split_setup_override(value: str) -> List[str]:
    try:
        return shlex.split(value, posix=(os.name != "nt"))
    except Exception:
        return value.split()


def _format_setup_command(command: List[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def _run_setup_command(
    command: List[str],
    *,
    action: str,
    log_fn: Optional[Callable[[str], None]] = None,
    timeout: int = OPENCODE_INSTALL_TIMEOUT_SECONDS,
    cwd: Optional[Path] = None,
) -> Dict[str, Any]:
    logger = log_fn or (lambda _message: None)
    command_text = _format_setup_command(command)
    logger(f"▶ Running: {command_text}")
    try:
        result = run_logged_subprocess(
            command,
            log_fn=logger,
            timeout=max(1, int(timeout)),
            cwd=str(cwd) if cwd is not None else None,
            env=_subprocess_env(),
            popen_kwargs={"creationflags": _creation_flags()},
        )
    except Exception as exc:
        error = f"{action} could not start: {exc}"
        logger(f"❌ {error}")
        return {"ok": False, "command": command_text, "error": error}

    output = str(result.get("output") or "No installer output was returned.")
    if result.get("timed_out"):
        error = f"{action} timed out after {max(1, int(timeout))}s."
        logger(f"❌ {error}")
        return {"ok": False, "command": command_text, "error": error, "output": output}
    returncode = int(result.get("returncode", -1))
    if returncode != 0:
        error = f"{action} exited with code {returncode}: {output}"
        logger(f"❌ {error}")
        return {
            "ok": False,
            "command": command_text,
            "error": error,
            "output": output,
        }
    return {
        "ok": True,
        "command": command_text,
        "output": output,
    }


def _nodejs_install_command() -> Optional[List[str]]:
    override = str(os.environ.get("OCAGY_NODE_INSTALL_CMD", "") or "").strip()
    if override:
        return _split_setup_override(override)
    if os.name != "nt":
        return None
    winget = _candidate_command("winget")
    if not winget:
        return None
    return [
        winget,
        "install",
        "--id",
        "OpenJS.NodeJS.LTS",
        "--exact",
        "--silent",
        "--accept-package-agreements",
        "--accept-source-agreements",
        "--disable-interactivity",
    ]


def _opencode_install_command() -> Optional[List[str]]:
    override = str(os.environ.get("OCAGY_OPENCODE_INSTALL_CMD", "") or "").strip()
    if override:
        return _split_setup_override(override)

    # OpenCode recommends its user-local install script on macOS/Linux. On
    # Windows, npm is the supported native route.
    if os.name != "nt":
        curl = _candidate_command("curl")
        bash = _candidate_command("bash")
        if curl and bash:
            return [
                bash,
                "-c",
                f"{shlex.quote(curl)} -fsSL https://opencode.ai/install | {shlex.quote(bash)}",
            ]

    npm = _candidate_command("npm")
    if npm:
        return [npm, "install", "-g", "opencode-ai"]
    return None


def ensure_opencode_installed(log_fn=None) -> str:
    """Find OpenCode or install it automatically, retaining manual fallback text."""
    try:
        return find_executable()
    except OcAgyError:
        pass

    logger = log_fn or (lambda _message: None)
    with _INSTALL_LOCK:
        try:
            return find_executable()
        except OcAgyError:
            pass

        logger("🧰 OpenCode CLI was not found. Glossarion is installing it automatically...")
        details: List[str] = []
        command = _opencode_install_command()

        if command is None and os.name == "nt":
            node_command = _nodejs_install_command()
            if node_command is not None:
                logger("🧰 npm is unavailable. Installing Node.js LTS first...")
                node_result = _run_setup_command(
                    node_command,
                    action="Node.js LTS installation",
                    log_fn=logger,
                )
                if not node_result.get("ok"):
                    details.append(str(node_result.get("error") or "Node.js installation failed."))
                command = _opencode_install_command()
            else:
                details.append("npm and a supported automatic Node.js installer were not found.")

        if command is not None:
            install_result = _run_setup_command(
                command,
                action="OpenCode installation",
                log_fn=logger,
            )
            if not install_result.get("ok"):
                details.append(str(install_result.get("error") or "OpenCode installation failed."))
        elif not details:
            details.append("No supported automatic OpenCode installer was found for this system.")

        try:
            executable = find_executable()
            logger(f"✅ OpenCode installed successfully: {executable}")
            return executable
        except OcAgyError:
            pass

        detail = "\n".join(dict.fromkeys(details))
        raise OcAgyError(
            "Glossarion could not install OpenCode automatically.\n"
            f"Installer details: {detail or 'The installer completed, but opencode was not found.'}\n\n"
            + get_install_instructions()
        )


def _plugin_fallback_instructions() -> str:
    return (
        "Fallback: verify that the Glossarion workspace opencode.json contains "
        f'\"plugin\": [\"{_PLUGIN_PACKAGE}\"], then run `opencode models google` '
        "or `opencode auth login` once."
    )


def ensure_auth_plugin_installed(executable: str, log_fn=None) -> None:
    """Resolve and verify opencode-antigravity-auth through OpenCode itself."""
    logger = log_fn or (lambda _message: None)
    key = str(Path(executable).resolve(strict=False)).casefold()
    if key in _PLUGIN_BOOTSTRAPPED_EXECUTABLES:
        return

    with _PLUGIN_BOOTSTRAP_LOCK:
        if key in _PLUGIN_BOOTSTRAPPED_EXECUTABLES:
            return

        workspace = _workspace_dir()
        logger(
            "🧩 Ensuring the OpenCode Antigravity auth plugin is installed "
            f"({_PLUGIN_PACKAGE})..."
        )
        result = _run_setup_command(
            [executable, "models", "google"],
            action="OpenCode Antigravity auth plugin installation",
            log_fn=logger,
            timeout=OCAGY_PLUGIN_INSTALL_TIMEOUT_SECONDS,
            cwd=workspace,
        )
        output = str(result.get("output") or "")
        if not result.get("ok") or "google/antigravity-" not in output.lower():
            detail = str(
                result.get("error")
                or "OpenCode finished without exposing any Antigravity plugin models."
            )
            raise OcAgyError(
                "Glossarion could not install or verify opencode-antigravity-auth automatically.\n"
                f"Installer details: {detail}\n\n{_plugin_fallback_instructions()}"
            )

        _PLUGIN_BOOTSTRAPPED_EXECUTABLES.add(key)
        logger("✅ OpenCode Antigravity auth plugin is installed and ready.")


def _workspace_dir() -> Path:
    configured = str(os.environ.get("OCAGY_WORKSPACE", "") or "").strip()
    path = Path(configured).expanduser() if configured else Path.home() / ".glossarion" / "opencode_antigravity"
    path.mkdir(parents=True, exist_ok=True)
    _write_workspace_config(path)
    _ensure_plugin_config()
    return path


def _workspace_config(*, temperature: Optional[float] = None) -> Dict[str, Any]:
    # Project-local config keeps this integration independent from the user's
    # normal OpenCode projects while still reusing the plugin's account store.
    config: Dict[str, Any] = {
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
                # OpenCode only forwards the agent's numeric temperature when
                # the selected custom model advertises temperature support.
                "models": {
                    "antigravity-gemini-3.1-pro": {
                        "name": "Gemini 3.1 Pro (Antigravity OAuth)",
                        "temperature": True,
                        "limit": {"context": 1048576, "output": 65535},
                        "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
                        "variants": {
                            "low": {"thinkingLevel": "low"},
                            "high": {"thinkingLevel": "high"},
                        },
                    },
                    "antigravity-gemini-3-pro": {
                        "name": "Gemini 3 Pro (Antigravity OAuth)",
                        "temperature": True,
                        "limit": {"context": 1048576, "output": 65535},
                        "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
                        "variants": {
                            "low": {"thinkingLevel": "low"},
                            "high": {"thinkingLevel": "high"},
                        },
                    },
                    "antigravity-gemini-3-flash": {
                        "name": "Gemini 3 Flash (Antigravity OAuth)",
                        "temperature": True,
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
                        "temperature": True,
                        "limit": {"context": 200000, "output": 64000},
                        "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
                    },
                    "antigravity-claude-opus-4-6-thinking": {
                        "name": "Claude Opus 4.6 Thinking (Antigravity OAuth)",
                        "temperature": True,
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
    if temperature is not None:
        config["agent"]["glossarion"]["temperature"] = float(temperature)
    return config


def _write_workspace_config(path: Path, *, temperature: Optional[float] = None) -> Path:
    target = path / "opencode.json"
    desired = _workspace_config(temperature=temperature)
    current: Any = None
    if target.is_file():
        try:
            current = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            current = None
    if current != desired:
        target.write_text(json.dumps(desired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def _model_output_limit(model_id: str) -> Optional[int]:
    """Return the configured hard output ceiling for one OpenCode model."""
    try:
        provider_id, provider_model = str(model_id).split("/", 1)
        value = _workspace_config()["provider"][provider_id]["models"][provider_model]["limit"]["output"]
        return int(value) if int(value) > 0 else None
    except (KeyError, TypeError, ValueError):
        return None


def _effective_output_limit(model_id: str, requested_max_tokens: int) -> int:
    requested = max(1, int(requested_max_tokens))
    model_limit = _model_output_limit(model_id)
    return min(requested, model_limit) if model_limit is not None else requested


def _log_generation_controls(
    logger: Callable[[str], None],
    *,
    model_id: str,
    temperature: float,
    max_tokens: int,
) -> None:
    requested = max(1, int(max_tokens))
    effective = _effective_output_limit(model_id, requested)
    if effective < requested:
        logger(
            "🎛️ Glossarion controls OpenCode generation: "
            f"temperature={temperature}, max_tokens={effective:,} "
            f"(requested {requested:,}; model cap {effective:,})."
        )
        return
    logger(
        "🎛️ Glossarion controls OpenCode generation: "
        f"temperature={temperature}, max_tokens={effective:,}."
    )


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


def _load_account_store() -> Tuple[Path, Dict[str, Any]]:
    """Read the plugin account store used by the normal OpenCode profile."""
    path = _config_dir() / "antigravity-accounts.json"
    if not path.is_file():
        return path, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OcAgyError(f"The OpenCode Antigravity OAuth account store could not be read: {exc}") from exc
    if not isinstance(payload, dict):
        raise OcAgyError("The OpenCode Antigravity OAuth account store has an invalid format.")
    return path, payload


def _select_account_slot(account_number: int) -> Tuple[Dict[str, Any], str]:
    """Return a specific saved account using Antigravity's one-based slot order."""
    _path, payload = _load_account_store()
    accounts = payload.get("accounts", [])
    if not isinstance(accounts, list):
        accounts = []
    index = account_number - 1
    if index < 0 or index >= len(accounts):
        raise OcAgyError(
            f"OcAgy account #{account_number} is not linked. Click OCAGY Login, add that "
            "Google account, then retry. Use ocagy0/ if you want the plugin-managed shared pool."
        )
    account = accounts[index]
    if not isinstance(account, dict) or not account.get("refreshToken"):
        raise OcAgyError(
            f"OcAgy account #{account_number} is not usable because its OAuth credential is missing. "
            "Sign in again with OCAGY Login."
        )
    if not account.get("enabled", True):
        raise OcAgyError(
            f"OcAgy account #{account_number} is disabled in the OpenCode Antigravity account store. "
            "Enable it or use another numbered OcAgy prefix."
        )
    email = str(account.get("email", "") or "").strip()
    return account, email


def _require_oauth_account(account_number: int = 0) -> Optional[Tuple[Dict[str, Any], str]]:
    """Fail before launching OpenCode when the requested OAuth account is unavailable."""
    if account_number > 0:
        return _select_account_slot(account_number)

    account = _account_summary()
    if account.get("account_count"):
        return None

    detail = ""
    if account.get("accounts_error"):
        detail = f" The OAuth account store could not be read: {account['accounts_error']}"
    raise OcAgyError(
        "OpenCode Antigravity OAuth is not configured. The ocagy/ route does not use or require "
        "a Google API key. Click the OCAGY Login button, select Google, then "
        "OAuth with Google (Antigravity), finish signing in, and retry. Do not select "
        f"Manually enter API Key.{detail}"
    )


def _write_private_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _prepare_isolated_account_config(
    request_dir: Path,
    account_number: int,
    selected_account: Dict[str, Any],
) -> Path:
    """Create a request-local config containing only the explicitly selected account."""
    config_dir = request_dir / f".ocagy-account-{account_number}"
    config_dir.mkdir(parents=True, exist_ok=False)
    try:
        config_dir.chmod(0o700)
    except OSError:
        pass

    # Deep-copy through JSON so the shared account object is never mutated
    # while forming the isolated plugin store.
    account_copy = json.loads(json.dumps(selected_account))
    _source_path, source_store = _load_account_store()
    account_store = {
        "version": source_store.get("version", 1),
        "accounts": [account_copy],
        "activeIndex": 0,
        "activeIndexByFamily": {"gemini": 0, "claude": 0},
    }
    _write_private_json(config_dir / "antigravity-accounts.json", account_store)

    plugin_settings: Dict[str, Any] = {}
    shared_settings = _config_dir() / "antigravity.json"
    try:
        if shared_settings.is_file():
            loaded = json.loads(shared_settings.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                plugin_settings = loaded
    except Exception:
        # The shared file may be JSONC. The isolated request only needs the
        # deterministic account-selection overrides below.
        plugin_settings = {}
    plugin_settings["pid_offset_enabled"] = False
    plugin_settings["account_selection_strategy"] = "sticky"
    _write_private_json(config_dir / "antigravity.json", plugin_settings)
    return config_dir


def _request_subprocess_env(
    request_dir: Path,
    account_number: int,
    selected: Optional[Tuple[Dict[str, Any], str]],
    logger: Callable[[str], None],
    max_tokens: int,
) -> Dict[str, str]:
    if selected is None:
        logger("🧭 OcAgy: using shared plugin account pool (ocagy0/)")
        env = _subprocess_env()
        # OpenCode otherwise applies its own global output ceiling before the
        # model-specific limit in opencode.json.
        env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(max(1, int(max_tokens)))
        return env

    selected_account, email = selected
    isolated_config = _prepare_isolated_account_config(
        request_dir,
        account_number,
        selected_account,
    )
    logger(
        f"🧭 OcAgy: using account slot #{account_number}"
        + (f" ({email})" if email else "")
    )
    env = _subprocess_env(isolated_config)
    env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = str(max(1, int(max_tokens)))
    return env


def _creation_flags(*, visible: bool = False) -> int:
    if os.name != "nt":
        return 0
    if visible:
        return getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


def _subprocess_env(config_dir: Optional[Path] = None) -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("NO_COLOR", "1")
    env.setdefault("TERM", "dumb")
    env.setdefault("PYTHONUTF8", "1")
    if config_dir is not None:
        env["OPENCODE_CONFIG_DIR"] = str(config_dir)
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


def _quota_form_json(url: str, values: Dict[str, str], timeout: float) -> Dict[str, Any]:
    data = urllib.parse.urlencode(values).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    payload = json.loads(raw.decode("utf-8", errors="replace")) if raw else {}
    return payload if isinstance(payload, dict) else {}


def _plugin_install_candidates() -> Iterable[Path]:
    """Yield likely roots for the OpenCode-owned Antigravity auth plugin."""
    override = str(os.environ.get("OCAGY_PLUGIN_ROOT", "") or "").strip()
    if override:
        yield Path(override).expanduser()

    package_name = "opencode-antigravity-auth"
    cache_base = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ).expanduser() / "opencode"
    yield cache_base / "node_modules" / package_name
    yield cache_base / "packages" / _PLUGIN_PACKAGE / "node_modules" / package_name
    packages_dir = cache_base / "packages"
    if packages_dir.is_dir():
        yield from sorted(
            packages_dir.glob(f"{package_name}@*/node_modules/{package_name}"),
            reverse=True,
        )


def _plugin_oauth_client_credentials() -> Tuple[str, str]:
    """Read OAuth application metadata from the plugin that owns authentication.

    The desktop OAuth application's metadata ships with the installed plugin.
    Glossarion must not duplicate those values in its own source or Git history.
    """
    configured_id = str(os.environ.get("OCAGY_GOOGLE_CLIENT_ID", "") or "").strip()
    configured_secret = str(os.environ.get("OCAGY_GOOGLE_CLIENT_SECRET", "") or "").strip()
    if configured_id and configured_secret:
        return configured_id, configured_secret

    patterns = {
        "client_id": re.compile(
            r"\bANTIGRAVITY_CLIENT_ID\s*=\s*['\"]([^'\"]+)['\"]"
        ),
        "client_secret": re.compile(
            r"\bANTIGRAVITY_CLIENT_SECRET\s*=\s*['\"]([^'\"]+)['\"]"
        ),
    }
    for root in _plugin_install_candidates():
        for constants_path in (
            root / "dist" / "src" / "constants.js",
            root / "src" / "constants.ts",
        ):
            if not constants_path.is_file():
                continue
            try:
                source = constants_path.read_text(encoding="utf-8")
            except OSError:
                continue
            client_id_match = patterns["client_id"].search(source)
            client_secret_match = patterns["client_secret"].search(source)
            if client_id_match and client_secret_match:
                return client_id_match.group(1), client_secret_match.group(1)

    raise OcAgyError(
        "Could not read OAuth application metadata from the installed "
        "opencode-antigravity-auth plugin. Restart OpenCode once so the plugin "
        "is installed, or set OCAGY_PLUGIN_ROOT to its package directory."
    )


def _quota_post_json(
    url: str,
    access_token: str,
    payload: Dict[str, Any],
    timeout: float,
    *,
    user_agent: str = _ANTIGRAVITY_QUOTA_USER_AGENT,
) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "User-Agent": user_agent,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    result = json.loads(raw.decode("utf-8", errors="replace")) if raw else {}
    return result if isinstance(result, dict) else {}


def _quota_error_message(error: BaseException) -> str:
    """Return bounded quota-check detail without reflecting credentials."""
    if isinstance(error, urllib.error.HTTPError):
        detail = ""
        try:
            body = error.read(4096).decode("utf-8", errors="replace")
            payload = json.loads(body)
            candidate = payload.get("error", payload) if isinstance(payload, dict) else payload
            if isinstance(candidate, dict):
                detail = str(
                    candidate.get("message")
                    or candidate.get("error_description")
                    or candidate.get("status")
                    or ""
                )
            elif isinstance(candidate, str):
                detail = candidate
        except Exception:
            detail = ""
        clean_detail = re.sub(r"\s+", " ", detail).strip()[:200]
        suffix = f" — {clean_detail}" if clean_detail else ""
        return f"HTTP {int(getattr(error, 'code', 0) or 0)}{suffix}"
    return re.sub(r"\s+", " ", str(error or "")).strip()[:240] or type(error).__name__


def _quota_reset_timestamp(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError):
        return None


def _quota_reset_in(value: Any) -> str:
    timestamp = _quota_reset_timestamp(value)
    if timestamp is None:
        return ""
    remaining = max(0, int(round(timestamp - time.time())))
    if remaining <= 0:
        return "now"
    days, remainder = divmod(remaining, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes = remainder // 60
    if days:
        return f"{days}d {hours}h" if hours else f"{days}d"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _normalize_quota_fraction(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return max(0.0, min(1.0, float(value)))


def _aggregate_antigravity_quota(models: Any) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    if not isinstance(models, dict):
        return groups
    for model_name, raw_info in models.items():
        info = raw_info if isinstance(raw_info, dict) else {}
        combined = f"{model_name} {info.get('displayName', '')} {info.get('modelName', '')}".lower()
        if "claude" in combined:
            group = "claude"
        elif "gemini-3" in combined or "gemini 3" in combined:
            group = "gemini-flash" if "flash" in combined else "gemini-pro"
        else:
            continue
        quota = info.get("quotaInfo")
        if not isinstance(quota, dict):
            continue
        remaining = _normalize_quota_fraction(quota.get("remainingFraction"))
        reset_time = str(quota.get("resetTime", "") or "")
        existing = groups.setdefault(group, {"model_count": 0})
        existing["model_count"] += 1
        if remaining is not None:
            current = existing.get("remaining_fraction")
            existing["remaining_fraction"] = remaining if current is None else min(current, remaining)
        reset_timestamp = _quota_reset_timestamp(reset_time)
        existing_timestamp = _quota_reset_timestamp(existing.get("reset_time"))
        if reset_timestamp is not None and (
            existing_timestamp is None or reset_timestamp < existing_timestamp
        ):
            existing["reset_time"] = reset_time
            existing["reset_in"] = _quota_reset_in(reset_time)
    return groups


def _aggregate_gemini_cli_quota(buckets: Any) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    if not isinstance(buckets, list):
        return models
    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue
        model_id = str(bucket.get("modelId", "") or "").strip()
        if not (model_id.startswith("gemini-3-") or model_id == "gemini-2.5-pro"):
            continue
        remaining = _normalize_quota_fraction(bucket.get("remainingFraction"))
        entry: Dict[str, Any] = {"model_id": model_id}
        if remaining is not None:
            entry["remaining_fraction"] = remaining
        reset_time = str(bucket.get("resetTime", "") or "")
        if reset_time:
            entry["reset_time"] = reset_time
            entry["reset_in"] = _quota_reset_in(reset_time)
        models.append(entry)
    return sorted(models, key=lambda item: str(item.get("model_id", "")).casefold())


def _check_account_quota(account: Dict[str, Any], index: int, timeout: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "index": index,
        "email": str(account.get("email", "") or "") or f"Account {index + 1}",
        "disabled": account.get("enabled", True) is False,
        "status": "error",
    }
    refresh_token = str(account.get("refreshToken", "") or "").strip()
    if not refresh_token:
        result["error"] = "Stored OAuth account has no refresh token"
        return result
    try:
        client_id, client_secret = _plugin_oauth_client_credentials()
        token_payload = _quota_form_json(
            "https://oauth2.googleapis.com/token",
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout,
        )
        access_token = str(token_payload.get("access_token", "") or "").strip()
        if not access_token:
            raise OcAgyError("Google did not return an OAuth access token")

        project_id = str(
            account.get("managedProjectId") or account.get("projectId") or ""
        ).strip()
        if not project_id:
            try:
                project_payload = _quota_post_json(
                    f"{_ANTIGRAVITY_CLOUD_CODE_BASE}/v1internal:loadCodeAssist",
                    access_token,
                    {"metadata": {"ideType": "ANTIGRAVITY"}},
                    timeout,
                )
                project_value = project_payload.get("cloudaicompanionProject")
                if isinstance(project_value, str):
                    project_id = project_value
                elif isinstance(project_value, dict):
                    project_id = str(project_value.get("id", "") or "")
            except Exception:
                project_id = ""
        project_id = project_id or _ANTIGRAVITY_FALLBACK_PROJECT_ID
        body = {"project": project_id}

        try:
            antigravity_payload = _quota_post_json(
                f"{_ANTIGRAVITY_CLOUD_CODE_BASE}/v1internal:fetchAvailableModels",
                access_token,
                body,
                timeout,
            )
            result["antigravity"] = _aggregate_antigravity_quota(
                antigravity_payload.get("models")
            )
            if not result["antigravity"]:
                result["antigravity_error"] = "No Antigravity quota information returned"
        except Exception as exc:
            result["antigravity"] = {}
            result["antigravity_error"] = _quota_error_message(exc)

        try:
            cli_payload = _quota_post_json(
                f"{_ANTIGRAVITY_CLOUD_CODE_BASE}/v1internal:retrieveUserQuota",
                access_token,
                body,
                timeout,
                user_agent="GeminiCLI/1.0.0/gemini-2.5-pro",
            )
            result["gemini_cli"] = _aggregate_gemini_cli_quota(cli_payload.get("buckets"))
            if not result["gemini_cli"]:
                result["gemini_cli_error"] = "No Gemini CLI quota information returned"
        except Exception as exc:
            result["gemini_cli"] = []
            result["gemini_cli_error"] = _quota_error_message(exc)

        result["status"] = "ok"
        return result
    except Exception as exc:
        result["error"] = _quota_error_message(exc)
        return result


def get_quota_status(timeout: float = 15.0, log_fn=None) -> Dict[str, Any]:
    """Fetch sanitized live quota data using the plugin's stored OAuth accounts."""
    executable = ensure_opencode_installed(log_fn=log_fn)
    ensure_auth_plugin_installed(executable, log_fn=log_fn)
    _require_oauth_account()
    account_path = _config_dir() / "antigravity-accounts.json"
    try:
        payload = json.loads(account_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OcAgyError(f"Could not read the OpenCode Antigravity account store: {exc}") from exc
    accounts = payload.get("accounts", []) if isinstance(payload, dict) else []
    if not isinstance(accounts, list):
        accounts = []
    timeout = max(1.0, float(timeout))
    results = [
        _check_account_quota(account, index, timeout)
        for index, account in enumerate(accounts)
        if isinstance(account, dict)
    ]
    enabled_count = sum(1 for item in results if not item.get("disabled"))
    return {
        "installed": True,
        "authenticated": enabled_count > 0,
        "plugin_ready": True,
        "executable": executable,
        "account_count": enabled_count,
        "emails": [item["email"] for item in results if not item.get("disabled")],
        "quota_accounts": results,
        "checked_at": time.time(),
    }


def normalize_polled_models(raw_models: Iterable[str]) -> List[str]:
    """Convert OpenCode plugin model IDs into selectable Glossarion IDs."""
    available = {
        str(model or "").strip().casefold(): str(model or "").strip()
        for model in raw_models
        if str(model or "").strip().casefold().startswith("google/antigravity-")
    }
    result: List[str] = []
    known = set()
    for raw_model, friendly, variants in _CATALOG_MODEL_VARIANTS:
        key = raw_model.casefold()
        known.add(key)
        if key not in available:
            continue
        if variants:
            result.extend(f"ocagy/{friendly}-{variant}" for variant in variants)
        else:
            result.append(f"ocagy/{friendly}")

    # Preserve future plugin models even before Glossarion gives them a friendly
    # alias. resolve_model() already accepts the retained antigravity-* suffix.
    for key in sorted(set(available) - known):
        raw_model = available[key]
        result.append(f"ocagy/{raw_model.split('/', 1)[1]}")
    return list(dict.fromkeys(result))


def poll_models(timeout: float = 8.0) -> List[str]:
    """Poll OpenCode's Google catalog using the existing OcAgy OAuth account."""
    _require_oauth_account()
    try:
        listing = _run_short(
            ["models", "google"],
            timeout=max(1, int(round(float(timeout)))),
        )
    except subprocess.TimeoutExpired as exc:
        raise OcAgyError("OpenCode model polling timed out") from exc
    combined = _clean_text((listing.stdout or "") + "\n" + (listing.stderr or ""))
    if listing.returncode != 0:
        raise _classify_error(
            combined or f"opencode models exited with code {listing.returncode}",
            listing.returncode,
        )
    raw_models = re.findall(r"google/[A-Za-z0-9._-]+", combined)
    models = normalize_polled_models(raw_models)
    if not models:
        raise OcAgyError("OpenCode returned no usable Antigravity OAuth model IDs")
    return models


def launch_login(log_fn=None) -> Dict[str, Any]:
    """Open OpenCode's plugin-aware OAuth login in a visible terminal."""
    exe = ensure_opencode_installed(log_fn=log_fn)
    ensure_auth_plugin_installed(exe, log_fn=log_fn)
    workspace = _workspace_dir()
    if os.name == "nt":
        # The project-local opencode.json already declares the plugin and models.
        command = (
            "$Host.UI.RawUI.WindowTitle='Glossarion OCAGY Login'; "
            "Write-Host 'Glossarion OCAGY OAuth setup' -ForegroundColor Cyan; "
            "Write-Host 'Select Google, then OAuth with Google (Antigravity).' -ForegroundColor Yellow; "
            f"Set-Location -LiteralPath {json.dumps(str(workspace))}; "
            f"& {json.dumps(exe)} auth login; "
            "Write-Host ''; Write-Host 'When login is complete, return to Glossarion. The account count updates automatically; click the quota button for live usage.' -ForegroundColor Green"
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


def parse_account_route(model: str) -> Tuple[int, str]:
    """Return ``(account_number, model_suffix)`` for an OcAgy route.

    Account number 0 means the plugin-managed shared pool. Unprefixed model
    suffixes are also treated as pooled for backwards compatibility with
    callers that invoke this module directly.
    """
    value = str(model or "").strip()
    match = re.fullmatch(r"ocagy(?P<number>\d{1,4})?(?:/(?P<suffix>.*))?", value, re.IGNORECASE)
    if not match:
        return 0, value
    number = match.group("number")
    account_number = 1 if number is None else (0 if int(number) == 0 else int(number) + 1)
    return account_number, str(match.group("suffix") or "").strip()


def resolve_model(model: str) -> Tuple[str, Optional[str]]:
    """Map a friendly Glossarion route/suffix to OpenCode's model + variant."""
    _account_number, suffix = parse_account_route(model)
    value = suffix.lower()

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
        "ocagy1/gemini-3.1-pro-high, ocagy0/gemini-3.1-pro-high, or another listed model."
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


def _normalize_opencode_finish_reason(reason: Any) -> Optional[str]:
    """Normalize an authoritative OpenCode reason for Glossarion routing."""
    raw = str(reason or "").strip()
    if not raw:
        return None
    normalized = raw.lower().replace(" ", "_")
    if normalized in {
        "end_turn",
        "end-turn",
        "completed",
        "complete",
    }:
        return "stop"
    if normalized in {
        "length",
        "max_tokens",
        "max-tokens",
        "max_output_tokens",
        "max-output-tokens",
        "truncated",
    }:
        return "length"
    if normalized in {
        "content-filter",
        "content_filter",
        "prohibited-content",
        "prohibited_content",
        "censorship-blocked",
        "censorship_blocked",
        "blocked",
        "safety",
        "recitation",
        "blocklist",
        "spii",
    }:
        return "prohibited_content"
    return normalized


def _finish_reason_from_info(info: Any) -> Optional[str]:
    """Read a finish value from one persisted/streamed assistant message."""
    if not isinstance(info, dict):
        return None
    for key in (
        "finish",
        "finish_reason",
        "finishReason",
        "stop_reason",
        "stopReason",
    ):
        value = info.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _finish_reason_from_message_records(
    records: Any,
    assistant_message_ids: Optional[Iterable[str]] = None,
) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """Extract the latest server-persisted assistant finish reason."""
    if isinstance(records, dict):
        message_records = [records]
    elif isinstance(records, list):
        message_records = records
    else:
        return None, "", {}

    allowed_ids = {
        str(message_id)
        for message_id in (assistant_message_ids or [])
        if str(message_id or "").strip()
    }
    for record in reversed(message_records):
        if not isinstance(record, dict):
            continue
        info = record.get("info")
        if not isinstance(info, dict):
            info = {}
        if str(info.get("role", "") or "").lower() not in ("", "assistant"):
            continue
        message_id = str(info.get("id", "") or "")
        if allowed_ids and message_id not in allowed_ids:
            continue

        parts = record.get("parts")
        if isinstance(parts, list):
            for part in reversed(parts):
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type", "") or "").replace("_", "-").lower()
                reason = str(part.get("reason", "") or "").strip()
                if part_type == "step-finish" and reason:
                    return reason, "server_message_step_finish", {
                        "message_id": message_id,
                        "part_id": str(part.get("id", "") or ""),
                    }

        reason = _finish_reason_from_info(info)
        if reason:
            return reason, "server_message_info", {
                "message_id": message_id,
            }
    return None, "", {}


def _finish_reason_from_cli_events(
    events: Iterable[Dict[str, Any]],
) -> Tuple[Optional[str], str]:
    """Extract OpenCode's reason from buffered ``run --format json`` events."""
    for event in reversed(list(events or [])):
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type", "") or "").replace("_", "-").lower()
        part = event.get("part")
        if event_type == "step-finish" and isinstance(part, dict):
            reason = str(part.get("reason", "") or "").strip()
            if reason:
                return reason, "cli_step_finish"
        if event_type == "message.updated":
            properties = event.get("properties")
            info = properties.get("info") if isinstance(properties, dict) else None
            reason = _finish_reason_from_info(info)
            if reason:
                return reason, "cli_message_info"
    return None, ""


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
    """Forward each httpx SSE event immediately instead of buffering the body."""
    data_lines: List[str] = []
    try:
        for raw in response.iter_lines():
            line = (
                raw.decode("utf-8", errors="replace")
                if isinstance(raw, bytes)
                else str(raw)
            ).rstrip("\r\n")
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


def _forced_stream_thinking_logging_enabled(log_stream: bool) -> bool:
    """Reasoning is part of OcAgy's forced live stream visibility."""
    return bool(log_stream)


class _OpenCodeStreamState:
    """Collect authoritative output while exposing incremental SSE deltas."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.assistant_message_ids: set[str] = set()
        self.assistant_message_order: List[str] = []
        self.text_order: List[str] = []
        self.text_by_part: Dict[str, str] = {}
        self.reasoning_by_part: Dict[str, str] = {}
        self.part_types: Dict[str, str] = {}
        self.step_events: List[Dict[str, Any]] = []
        self.finish_reason: Optional[str] = None
        self.raw_finish_reason: Optional[str] = None
        self.finish_reason_source = ""
        self.finish_reason_evidence: Dict[str, Any] = {}
        self.error = ""
        self.complete = False

    def record_finish_reason(
        self,
        reason: Any,
        source: str,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Record a real OpenCode reason without inventing a default."""
        raw_reason = str(reason or "").strip()
        normalized = _normalize_opencode_finish_reason(raw_reason)
        if not raw_reason or not normalized:
            return False
        self.raw_finish_reason = raw_reason
        self.finish_reason = normalized
        self.finish_reason_source = str(source or "")
        self.finish_reason_evidence = dict(evidence or {})
        return True

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
                    if message_id not in self.assistant_message_order:
                        self.assistant_message_order.append(message_id)
                reason = _finish_reason_from_info(info)
                if reason:
                    self.record_finish_reason(
                        reason,
                        "sse_message_info",
                        {"message_id": message_id},
                    )
            return emitted

        if event_type == "message.part.updated":
            part = properties.get("part")
            if not isinstance(part, dict) or part.get("sessionID") != self.session_id:
                return emitted
            message_id = str(part.get("messageID", "") or "")
            part_type = str(part.get("type", "") or "")
            if part_type == "step-finish":
                self.step_events.append({"type": "step_finish", "part": part})
                reason = str(part.get("reason", "") or "")
                if reason:
                    self.record_finish_reason(
                        reason,
                        "sse_step_finish",
                        {
                            "message_id": message_id,
                            "part_id": str(part.get("id", "") or ""),
                        },
                    )
                return emitted
            if message_id not in self.assistant_message_ids:
                return emitted
            part_id = str(part.get("id", "") or "")
            if part_id and part_type:
                self.part_types[part_id] = part_type
            if part_type == "text":
                if part_id and part_id not in self.text_order:
                    self.text_order.append(part_id)
                fragment = self._merge_part(part, properties.get("delta"), self.text_by_part)
                if fragment:
                    emitted.append(("text", fragment))
            elif part_type == "reasoning":
                fragment = self._merge_part(part, properties.get("delta"), self.reasoning_by_part)
                if fragment:
                    emitted.append(("reasoning", fragment))
            return emitted

        # OpenCode 1.18+ publishes the actual live token fragments separately
        # from message.part.updated. The updated event identifies the part and
        # the delta event carries each new piece of its text. Ignoring this
        # event makes an otherwise-live SSE connection appear buffered until
        # OpenCode sends the final accumulated part snapshot.
        if event_type == "message.part.delta":
            if properties.get("sessionID") != self.session_id:
                return emitted
            message_id = str(properties.get("messageID", "") or "")
            if message_id not in self.assistant_message_ids:
                return emitted
            if str(properties.get("field", "") or "") != "text":
                return emitted
            part_id = str(properties.get("partID", "") or "")
            delta = properties.get("delta")
            delta_text = str(delta) if isinstance(delta, str) else ""
            if not part_id or not delta_text:
                return emitted
            part_type = self.part_types.get(part_id, "")
            if part_type == "text":
                if part_id not in self.text_order:
                    self.text_order.append(part_id)
                self.text_by_part[part_id] = self.text_by_part.get(part_id, "") + delta_text
                emitted.append(("text", delta_text))
            elif part_type == "reasoning":
                self.reasoning_by_part[part_id] = (
                    self.reasoning_by_part.get(part_id, "") + delta_text
                )
                emitted.append(("reasoning", delta_text))
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


def _retrieve_server_finish_reason(
    base_url: str,
    session_id: str,
    assistant_message_ids: Iterable[str],
    timeout: float,
) -> Tuple[Optional[str], str, Dict[str, Any], str]:
    """Query OpenCode's persisted message records for a missed finish event."""
    deadline = time.monotonic() + max(0.1, float(timeout or 0.1))
    ordered_ids = [
        str(message_id)
        for message_id in assistant_message_ids
        if str(message_id or "").strip()
    ]
    errors: List[str] = []

    def remaining() -> float:
        return max(0.1, deadline - time.monotonic())

    encoded_session = urllib.parse.quote(str(session_id), safe="")
    for message_id in reversed(ordered_ids):
        if time.monotonic() >= deadline:
            break
        encoded_message = urllib.parse.quote(message_id, safe="")
        try:
            record = _http_json(
                base_url,
                f"/session/{encoded_session}/message/{encoded_message}",
                timeout=remaining(),
            )
            reason, source, evidence = _finish_reason_from_message_records(
                record,
                [message_id],
            )
            if reason:
                return reason, source, evidence, ""
        except Exception as exc:
            errors.append(str(exc))

    if time.monotonic() < deadline:
        try:
            records = _http_json(
                base_url,
                f"/session/{encoded_session}/message?limit=20",
                timeout=remaining(),
            )
            reason, source, evidence = _finish_reason_from_message_records(
                records,
                ordered_ids,
            )
            if reason:
                return reason, source, evidence, ""
        except Exception as exc:
            errors.append(str(exc))

    return None, "", {}, "; ".join(error for error in errors if error)


def _send_via_server(
    *,
    exe: str,
    request_dir: Path,
    prompt: str,
    model_id: str,
    variant: Optional[str],
    timeout_seconds: float,
    logger: Callable[[str], None],
    log_stream: bool,
    subprocess_env: Dict[str, str],
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
        env=subprocess_env,
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
    event_stream_context = None
    session_id = ""
    started = time.time()

    def remaining_request_timeout() -> float:
        remaining = float(timeout_seconds) - (time.time() - started)
        if remaining <= 0:
            raise OcAgyError(
                f"OpenCode Antigravity timed out after {timeout_seconds}s. "
                "Increase the API timeout or use a shorter chunk."
            )
        return max(0.1, remaining)

    try:
        startup_deadline = started + timeout_seconds
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
                health = _http_json(
                    base_url,
                    "/global/health",
                    timeout=min(0.75, remaining_request_timeout()),
                )
                if isinstance(health, dict) and health.get("healthy"):
                    break
            except Exception:
                time.sleep(0.1)
        else:
            raise OcAgyError(
                f"OpenCode Antigravity server did not become ready within {timeout_seconds} seconds."
            )

        session = _http_json(
            base_url,
            "/session",
            method="POST",
            payload={"title": "Glossarion translation"},
            timeout=remaining_request_timeout(),
        )
        if not isinstance(session, dict) or not session.get("id"):
            raise OcAgyError("OpenCode Antigravity could not create a streaming session.")
        session_id = str(session["id"])

        if httpx is None:
            raise OcAgyError(
                "OcAgy real-time streaming requires httpx, but it is not installed."
            )
        event_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        connect_timeout = remaining_request_timeout()
        stream_timeout = httpx.Timeout(
            remaining_request_timeout(),
            connect=connect_timeout,
        )
        event_stream_context = httpx.stream(
            "GET",
            base_url + "/event",
            headers={
                "Accept": "text/event-stream",
                "Accept-Encoding": "identity",
                "Cache-Control": "no-cache",
            },
            timeout=stream_timeout,
        )
        event_response = event_stream_context.__enter__()
        if event_response.status_code != 200:
            try:
                detail = event_response.read().decode("utf-8", errors="replace")
            except Exception:
                detail = f"HTTP {event_response.status_code}"
            raise _classify_error(detail, int(event_response.status_code or 1))
        threading.Thread(
            target=_read_sse_events,
            args=(event_response, event_queue),
            daemon=True,
        ).start()

        connected = False
        connect_deadline = started + timeout_seconds
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
            timeout=remaining_request_timeout(),
        )

        state = _OpenCodeStreamState(session_id)
        text_buffer: List[str] = []
        thinking_buffer: List[str] = []
        thinking_started = False
        first_text = False
        stream_thinking = _forced_stream_thinking_logging_enabled(log_stream)

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

        finish_reason_lookup_error = ""
        if not state.finish_reason:
            remaining_after_stream = timeout_seconds - (time.time() - started)
            if remaining_after_stream > 0.1:
                lookup_timeout = min(3.0, remaining_after_stream)
                (
                    persisted_reason,
                    persisted_source,
                    persisted_evidence,
                    finish_reason_lookup_error,
                ) = _retrieve_server_finish_reason(
                    base_url,
                    session_id,
                    state.assistant_message_order,
                    lookup_timeout,
                )
                if persisted_reason:
                    state.record_finish_reason(
                        persisted_reason,
                        persisted_source,
                        persisted_evidence,
                    )
            else:
                finish_reason_lookup_error = (
                    "request deadline exhausted before server lookup"
                )

        finish_reason_fallback = not bool(state.finish_reason)
        if finish_reason_fallback:
            state.finish_reason = "stop"
            state.raw_finish_reason = None
            state.finish_reason_source = "fallback_stop"
            state.finish_reason_evidence = {}
            warning = (
                "⚠️ OpenCode Antigravity: no finish reason was present in "
                "the live events or persisted server message; falling back to 'stop'."
            )
            if finish_reason_lookup_error:
                warning += f" Server lookup: {finish_reason_lookup_error}"
            logger(warning)
        else:
            raw_suffix = (
                f", raw: '{state.raw_finish_reason}'"
                if state.raw_finish_reason
                and state.raw_finish_reason != state.finish_reason
                else ""
            )
            logger(
                "🏁 OpenCode Antigravity: finish reason "
                f"'{state.finish_reason}' (source: {state.finish_reason_source}"
                f"{raw_suffix})"
            )

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
            "raw_finish_reason": state.raw_finish_reason,
            "finish_reason_source": state.finish_reason_source,
            "finish_reason_fallback": finish_reason_fallback,
            "finish_reason_evidence": state.finish_reason_evidence,
            "finish_reason_lookup_error": finish_reason_lookup_error,
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
    except TimeoutError as exc:
        raise OcAgyError(
            f"OpenCode Antigravity timed out after {timeout_seconds}s. "
            "Increase the API timeout or use a shorter chunk."
        ) from exc
    except Exception as exc:
        if httpx is not None and isinstance(exc, httpx.TimeoutException):
            raise OcAgyError(
                f"OpenCode Antigravity real-time stream timed out after {timeout_seconds}s."
            ) from exc
        if httpx is not None and isinstance(exc, httpx.RequestError):
            raise OcAgyError(
                f"OpenCode Antigravity local real-time stream failed: {exc}"
            ) from exc
        raise
    finally:
        if event_response is not None:
            try:
                event_response.close()
            except Exception:
                pass
        if event_stream_context is not None:
            try:
                event_stream_context.__exit__(None, None, None)
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
            "the OCAGY Login button, select Google, then OAuth with Google "
            "(Antigravity), finish signing in, and retry. Do not select Manually enter API Key."
        )
    if any(x in lower for x in ("invalid_grant", "not authenticated", "auth login", "oauth", "credential", "api key missing")):
        return OcAgyError(
            "OpenCode Antigravity authentication failed. Click the OCAGY Login button, "
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
    timeout: float = 1800,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run one isolated ``opencode run --format json`` request via the plugin."""
    if is_cancelled():
        raise OcAgyError("OpenCode Antigravity request cancelled by user")

    logger = log_fn or (lambda _message: None)
    exe = ensure_opencode_installed(log_fn=logger)
    account_number, _model_suffix = parse_account_route(model)
    selected = _require_oauth_account(account_number)
    ensure_auth_plugin_installed(exe, log_fn=logger)
    model_id, variant = resolve_model(model)
    prompt = build_prompt(messages)
    timeout_seconds = max(1.0, float(timeout or 1800))

    base = _workspace_dir()
    request_dir = Path(tempfile.mkdtemp(prefix="request-", dir=str(base)))
    try:
        process_env = _request_subprocess_env(
            request_dir,
            account_number,
            selected,
            logger,
            max_tokens,
        )
        _write_workspace_config(request_dir, temperature=temperature)
    except Exception:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise
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
    _log_generation_controls(
        logger,
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    start = time.time()
    try:
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
            env=process_env,
        )
    except Exception:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise
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
    raw_finish_reason, finish_reason_source = _finish_reason_from_cli_events(
        events
    )
    finish_reason = _normalize_opencode_finish_reason(raw_finish_reason)
    finish_reason_fallback = not bool(finish_reason)
    if finish_reason_fallback:
        finish_reason = "stop"
        finish_reason_source = "fallback_stop"
        logger(
            "⚠️ OpenCode Antigravity: buffered events did not include a finish "
            "reason; falling back to 'stop'."
        )
    else:
        raw_suffix = (
            f", raw: '{raw_finish_reason}'"
            if raw_finish_reason != finish_reason
            else ""
        )
        logger(
            "🏁 OpenCode Antigravity: finish reason "
            f"'{finish_reason}' (source: {finish_reason_source}{raw_suffix})"
        )
    logger(f"✅ OpenCode Antigravity: completed in {elapsed:.1f}s")
    return {
        "content": content,
        "finish_reason": finish_reason,
        "raw_finish_reason": raw_finish_reason,
        "finish_reason_source": finish_reason_source,
        "finish_reason_fallback": finish_reason_fallback,
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
    timeout: float = 1800,
    log_fn: Optional[Callable[[str], None]] = None,
    log_stream: bool = True,
) -> Dict[str, Any]:
    """Run one isolated request through OpenCode's live local event API."""
    if is_cancelled():
        raise OcAgyError("OpenCode Antigravity request cancelled by user")

    logger = log_fn or (lambda _message: None)
    exe = ensure_opencode_installed(log_fn=logger)
    account_number, _model_suffix = parse_account_route(model)
    selected = _require_oauth_account(account_number)
    ensure_auth_plugin_installed(exe, log_fn=logger)
    model_id, variant = resolve_model(model)
    prompt = build_prompt(messages)
    timeout_seconds = max(1.0, float(timeout or 1800))

    base = _workspace_dir()
    request_dir = Path(tempfile.mkdtemp(prefix="request-", dir=str(base)))
    start = time.time()
    try:
        process_env = _request_subprocess_env(
            request_dir,
            account_number,
            selected,
            logger,
            max_tokens,
        )
        _write_workspace_config(request_dir, temperature=temperature)
        logger(
            f"🪐 OpenCode Antigravity: {Path(exe).name} live stream --model {model_id}"
            + (f" --variant {variant}" if variant else "")
        )
        _log_generation_controls(
            logger,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = _send_via_server(
            exe=exe,
            request_dir=request_dir,
            prompt=prompt,
            model_id=model_id,
            variant=variant,
            timeout_seconds=timeout_seconds,
            logger=logger,
            log_stream=bool(log_stream),
            subprocess_env=process_env,
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
        "finish_reason": str(result.get("finish_reason", "") or "stop"),
        "raw_finish_reason": result.get("raw_finish_reason"),
        "finish_reason_source": str(
            result.get("finish_reason_source", "")
            or (
                "send_chat_completion_fallback_stop"
                if not result.get("finish_reason")
                else ""
            )
        ),
        "finish_reason_fallback": bool(
            result.get("finish_reason_fallback", False)
            or not result.get("finish_reason")
        ),
        "finish_reason_evidence": result.get("finish_reason_evidence") or {},
        "finish_reason_lookup_error": result.get(
            "finish_reason_lookup_error", ""
        ),
        "usage": result.get("usage"),
        "model": model_id,
        "variant": variant,
        "provider": "ocagy",
        "elapsed_seconds": elapsed,
        "stderr": "",
        "raw_response": result.get("raw_response"),
    }
