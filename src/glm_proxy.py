"""Local Z.AI login proxy used by Glossarion's ``authza/`` routes.

This adapter manages the unofficial TriDefender/zcode-api runtime, including
downloading a pinned/current source archive, installing Bun and package
dependencies, opening Z.AI's browser login, and running an OpenAI-compatible
localhost server.  Each numbered ``authzaN/`` route gets isolated credentials,
configuration, port, and process state.

Upstream: https://github.com/TriDefender/zcode-api
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

try:
    import httpx
except ImportError:  # pragma: no cover - requests remains the compatibility fallback
    httpx = None

from installer_utils import run_logged_subprocess


logger = logging.getLogger(__name__)

PROXY_REPO_URL = "https://github.com/TriDefender/zcode-api"
PROXY_GITHUB_API_MASTER = "https://api.github.com/repos/TriDefender/zcode-api/commits/master"
PROXY_GITHUB_RAW_PACKAGE_URL = (
    "https://raw.githubusercontent.com/TriDefender/zcode-api/{revision}/package.json"
)
PROXY_GITHUB_REVISION_ARCHIVE_URL = (
    "https://codeload.github.com/TriDefender/zcode-api/zip/{revision}"
)
PROXY_DEFAULT_REVISION = "9cec45e7268190050b4af6074ea4d852f8241b8a"
PROXY_DEFAULT_VERSION = "2.6.0"
PROXY_UPDATE_CHECK_INTERVAL_SECONDS = 300
PROXY_ARCHIVE_DOWNLOAD_TIMEOUT_SECONDS = 90
RUNTIME_PATCH_VERSION = "2026-08-27-zcode-dual-access-v6"
ZCODE_APP_VERSION = "3.9.2"

DEFAULT_PROXY_HOST = "127.0.0.1"
DEFAULT_PROXY_PORT = 18870
CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"
MODELS_ENDPOINT = "/v1/models"
HEALTH_ENDPOINT = "/health"
LOGIN_PLAN_CHAT_ENDPOINT = (
    "https://zcode.z.ai/api/v1/zcode-plan/anthropic/v1/messages"
)
LOGIN_PLAN_MODELS_ENDPOINT = "https://zcode.z.ai/api/v1/zcode-plan/billing/balance"
GENERAL_API_BASE = "https://api.z.ai/api/paas/v4"
GENERAL_API_CHAT_ENDPOINT = f"{GENERAL_API_BASE}/chat/completions"
GENERAL_API_MODELS_ENDPOINT = f"{GENERAL_API_BASE}/models"
GENERAL_API_MODE_ENV = "AUTHZA_USE_GENERAL_API"
GENERAL_API_CATALOG_MIN_TIMEOUT_SECONDS = 45
BUN_NPM_PACKAGE = os.environ.get("GLM_PROXY_BUN_PACKAGE", "bun@latest")
BUN_INSTALL_TIMEOUT_SECONDS = 300
DEPENDENCY_INSTALL_TIMEOUT_SECONDS = 600
LOGIN_TIMEOUT_SECONDS = 360

DEFAULT_MODELS = (
    "glm-4.5-air",
    "glm-4.6",
    "glm-4.6v",
    "glm-4.7",
    "glm-5",
    "glm-5-turbo",
    "glm-5v-turbo",
    "glm-5.1",
    "glm-5.2",
    "glm-5.3",
)

# Keep the login JWT inside the JavaScript credential store. This probe mirrors
# current ZCode desktop's account-specific model discovery from
# data.balances[].capabilities entries named "model:*".
_LOGIN_PLAN_MODEL_CATALOG_SCRIPT = r'''
import { loadCredential } from "./src/auth/store.ts";

const credential = await loadCredential();
if (!credential?.jwt) throw new Error("Z.AI login JWT is unavailable; sign in again");
const appVersion = process.env.ZCODE_APP_VERSION || "3.9.2";
const endpoint = new URL(process.env.ZCODE_LOGIN_PLAN_MODELS_ENDPOINT);
endpoint.searchParams.set("app_version", appVersion);
const timeoutMs = Number(process.env.ZCODE_MODEL_CATALOG_TIMEOUT_MS || "10000");
const response = await fetch(endpoint, {
  headers: {
    "Authorization": `Bearer ${credential.jwt}`,
    "HTTP-Referer": "https://zcode.z.ai",
    "User-Agent": `ZCode/${appVersion}`,
    "X-Title": "Z Code@glossarion",
    "X-ZCode-Agent": "glm",
    "X-ZCode-App-Version": appVersion,
  },
  signal: AbortSignal.timeout(Math.max(1000, timeoutMs)),
});
const text = await response.text();
let payload = {};
try { payload = JSON.parse(text); } catch {}
if (!response.ok || (payload.code !== undefined && ![0, 200].includes(payload.code))) {
  const detail = payload.msg || payload.message || text.slice(0, 300) || "unknown error";
  throw new Error(`Z.AI model catalog HTTP ${response.status}: ${detail}`);
}
const models = [];
const seen = new Set();
for (const balance of payload?.data?.balances || []) {
  const capabilities = Array.isArray(balance?.capabilities) ? balance.capabilities : [];
  const capabilityModels = capabilities
    .filter((item) => typeof item === "string" && item.toLowerCase().startsWith("model:"))
    .map((item) => item.slice(6).trim())
    .filter(Boolean);
  const candidates = capabilityModels.length
    ? capabilityModels
    : [String(balance?.show_name || "").trim()];
  for (const model of candidates) {
    const key = model.toLowerCase();
    if (model && !seen.has(key)) {
      seen.add(key);
      models.push(model);
    }
  }
}
process.stdout.write("GLOSSARION_MODELS=" + JSON.stringify(models));
'''.strip()

# General API credentials are still encrypted by zcode-api's credential store.
# Keep the provisioned API key inside Bun while querying the OpenAI-compatible
# model catalog, just as the login-plan probe keeps the JWT outside Python.
_GENERAL_API_MODEL_CATALOG_SCRIPT = r'''
import { loadCredential } from "./src/auth/store.ts";
import { credentialString } from "./src/auth/types.ts";

const credential = await loadCredential();
if (!credential?.apiKey || credential.apiKey === "zcode-login") {
  throw new Error("Z.AI general API key is unavailable; sign in again in general API mode");
}
const endpoint = process.env.ZCODE_GENERAL_API_MODELS_ENDPOINT;
const timeoutMs = Number(process.env.ZCODE_MODEL_CATALOG_TIMEOUT_MS || "10000");
const response = await fetch(endpoint, {
  headers: {
    "Authorization": `Bearer ${credentialString(credential)}`,
    "Accept-Language": "en-US,en",
    "User-Agent": `Glossarion/${process.env.ZCODE_APP_VERSION || "3.9.2"}`,
  },
  signal: AbortSignal.timeout(Math.max(1000, timeoutMs)),
});
const text = await response.text();
let payload = {};
try { payload = JSON.parse(text); } catch {}
if (!response.ok || payload?.error) {
  const detail = payload?.error?.message || payload?.msg || payload?.message || text.slice(0, 300) || "unknown error";
  throw new Error(`Z.AI general model catalog HTTP ${response.status}: ${detail}`);
}
const entries = Array.isArray(payload?.data)
  ? payload.data
  : Array.isArray(payload?.data?.models)
    ? payload.data.models
    : Array.isArray(payload?.models)
      ? payload.models
      : [];
const models = [];
const seen = new Set();
for (const entry of entries) {
  const model = String(typeof entry === "string" ? entry : entry?.id || entry?.model || entry?.name || "").trim();
  const key = model.toLowerCase();
  if (model && !seen.has(key)) {
    seen.add(key);
    models.push(model);
  }
}
process.stdout.write("GLOSSARION_MODELS=" + JSON.stringify(models));
'''.strip()

_cancel_event = threading.Event()
_cancel_state_lock = threading.Lock()
_cancel_generation = 0
_active_response_lock = threading.Lock()
_active_responses: Dict[int, Any] = {}
_proxy_launch_lock = threading.Lock()
_proxy_processes: Dict[int, subprocess.Popen] = {}
_proxy_started_callback = None
_proxy_started_callback_lock = threading.Lock()
_update_lock = threading.Lock()
_last_release_check_at = 0.0
_cached_release: Optional[Dict[str, Any]] = None


def _log_noop(_message: str) -> None:
    return None


def _log_console(message: str) -> None:
    """Best-effort console logging that also works on legacy Windows codepages."""
    try:
        print(message)
    except UnicodeEncodeError:
        text = str(message).encode("ascii", errors="backslashreplace").decode("ascii")
        print(text)


def set_proxy_started_callback(callback) -> None:
    """Register a UI callback invoked with the ready account id."""
    global _proxy_started_callback
    with _proxy_started_callback_lock:
        _proxy_started_callback = callback


def _notify_proxy_started(account_id: int) -> None:
    with _proxy_started_callback_lock:
        callback = _proxy_started_callback
    if callback is None:
        return
    try:
        callback(int(account_id))
    except Exception as exc:
        logger.debug("GLM proxy-start callback failed: %s", exc)


def _normalize_account_id(account_id: Optional[int]) -> int:
    try:
        value = int(account_id or 0)
    except (TypeError, ValueError):
        value = 0
    if value < 0 or value > 9999:
        raise ValueError("GLM proxy account id must be between 0 and 9999")
    return value


def uses_general_api() -> bool:
    """Return whether AuthZA should use the billable general API endpoint."""
    return os.environ.get(GENERAL_API_MODE_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def set_general_api_mode(enabled: bool) -> bool:
    """Apply the AuthZA access mode and stop stale managed proxy processes.

    Login-plan and general-API credentials are stored separately, so changing
    this setting never overwrites the other login. The next request/login will
    rebuild the account config and start a proxy in the selected mode.
    """
    previous = uses_general_api()
    current = bool(enabled)
    os.environ[GENERAL_API_MODE_ENV] = "1" if current else "0"
    if previous != current:
        for account in list(_proxy_processes):
            shutdown_proxy(account)
    return current


def get_upstream_chat_endpoint() -> str:
    """Return the exact Z.AI endpoint selected for AuthZA chat requests."""
    if uses_general_api():
        return GENERAL_API_CHAT_ENDPOINT
    return LOGIN_PLAN_CHAT_ENDPOINT


def account_id_from_model(model: str) -> Optional[int]:
    """Return the isolated proxy account selected by an ``authza`` model.

    The unnumbered ``authza/`` route is account 0.  A non-AuthZA model returns
    ``None`` so GUI callers can distinguish it from that default account.
    """
    match = re.match(r"^authza(\d{0,4})(?:/|$)", str(model or "").strip(), re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1) or 0)


def _get_proxy_data_dir() -> str:
    return os.environ.get(
        "GLM_PROXY_DATA_DIR",
        os.path.join(os.path.expanduser("~"), ".config", "glossarion-glm-proxy"),
    )


def _account_dir(account_id: Optional[int] = None) -> str:
    account = _normalize_account_id(account_id)
    name = "default" if account == 0 else str(account)
    return os.path.join(_get_proxy_data_dir(), "accounts", name)


def _credentials_path(
    account_id: Optional[int] = None,
    *,
    general_api: Optional[bool] = None,
) -> str:
    use_general = uses_general_api() if general_api is None else bool(general_api)
    filename = "credentials-general-api.json" if use_general else "credentials.json"
    return os.path.join(_account_dir(account_id), filename)


def _config_path(account_id: Optional[int] = None) -> str:
    return os.path.join(_account_dir(account_id), "config.yaml")


def _secrets_path(account_id: Optional[int] = None) -> str:
    return os.path.join(_account_dir(account_id), "secrets.json")


def _account_env_name(base: str, account_id: Optional[int]) -> str:
    account = _normalize_account_id(account_id)
    return base if account == 0 else f"{base}_{account}"


def _get_proxy_port(account_id: Optional[int] = None) -> int:
    account = _normalize_account_id(account_id)
    specific = os.environ.get(_account_env_name("GLM_PROXY_PORT", account), "").strip()
    base = os.environ.get("GLM_PROXY_PORT_BASE", "").strip()
    try:
        if specific:
            port = int(specific)
        elif base:
            port = int(base) + account
        else:
            port = DEFAULT_PROXY_PORT + account
    except ValueError:
        port = DEFAULT_PROXY_PORT + account
    if not 1 <= port <= 65535:
        raise ValueError(f"Invalid GLM proxy port for account {account}: {port}")
    return port


def _external_proxy_url(account_id: Optional[int] = None) -> Optional[str]:
    value = os.environ.get(_account_env_name("GLM_PROXY_URL", account_id), "").strip()
    if not value and _normalize_account_id(account_id) != 0:
        value = os.environ.get("GLM_PROXY_URL", "").strip()
    return value.rstrip("/") or None


def get_proxy_url(account_id: Optional[int] = None) -> str:
    external = _external_proxy_url(account_id)
    if external:
        return external
    return f"http://{DEFAULT_PROXY_HOST}:{_get_proxy_port(account_id)}"


def get_local_chat_endpoint(account_id: Optional[int] = None) -> str:
    """Return the local OpenAI-compatible endpoint used by Glossarion."""
    return f"{get_proxy_url(account_id)}{CHAT_COMPLETIONS_ENDPOINT}"


def _github_headers() -> Dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Glossarion GLM Proxy Updater",
    }


def _parse_version(value: str) -> str:
    match = re.search(r"(\d+\.\d+\.\d+)", value or "")
    return match.group(1) if match else PROXY_DEFAULT_VERSION


def _latest_proxy_release() -> Dict[str, Any]:
    global _cached_release, _last_release_check_at
    revision_override = os.environ.get("GLM_PROXY_REVISION", "").strip()
    if revision_override:
        return {
            "tag": f"master-{revision_override[:12]}",
            "revision": revision_override,
            "version": os.environ.get("GLM_PROXY_VERSION", PROXY_DEFAULT_VERSION),
            "archive_url": PROXY_GITHUB_REVISION_ARCHIVE_URL.format(revision=revision_override),
            "resolved": True,
        }

    with _update_lock:
        if (
            _cached_release is not None
            and time.monotonic() - _last_release_check_at < PROXY_UPDATE_CHECK_INTERVAL_SECONDS
        ):
            return dict(_cached_release)

        release: Dict[str, Any]
        try:
            response = requests.get(PROXY_GITHUB_API_MASTER, headers=_github_headers(), timeout=15)
            response.raise_for_status()
            revision = str((response.json() or {}).get("sha") or "").strip()
            if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                raise ValueError("GitHub returned an invalid revision")
            version = PROXY_DEFAULT_VERSION
            try:
                package_response = requests.get(
                    PROXY_GITHUB_RAW_PACKAGE_URL.format(revision=revision),
                    headers=_github_headers(),
                    timeout=15,
                )
                package_response.raise_for_status()
                version = _parse_version(str((package_response.json() or {}).get("version") or ""))
            except Exception as exc:
                logger.debug("Could not read zcode-api package version: %s", exc)
            release = {
                "tag": f"master-{revision[:12]}",
                "revision": revision,
                "version": version,
                "archive_url": PROXY_GITHUB_REVISION_ARCHIVE_URL.format(revision=revision),
                "resolved": True,
            }
        except Exception as exc:
            logger.debug("Could not resolve current zcode-api revision: %s", exc)
            release = {
                "tag": f"master-{PROXY_DEFAULT_REVISION[:12]}",
                "revision": PROXY_DEFAULT_REVISION,
                "version": PROXY_DEFAULT_VERSION,
                "archive_url": PROXY_GITHUB_REVISION_ARCHIVE_URL.format(
                    revision=PROXY_DEFAULT_REVISION
                ),
                "resolved": False,
            }
        _cached_release = release
        _last_release_check_at = time.monotonic()
        return dict(release)


def _safe_runtime_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "runtime"


def _runtime_root() -> str:
    return os.path.join(_get_proxy_data_dir(), "runtime")


def _runtime_metadata_path(runtime_dir: str) -> str:
    return os.path.join(runtime_dir, ".glossarion-runtime.json")


def _runtime_entrypoint(runtime_dir: str) -> str:
    return os.path.join(runtime_dir, "src", "index.ts")


def _runtime_is_valid(runtime_dir: str, release: Optional[Dict[str, Any]] = None) -> bool:
    if not os.path.isfile(_runtime_entrypoint(runtime_dir)):
        return False
    try:
        with open(_runtime_metadata_path(runtime_dir), "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("patch_version") != RUNTIME_PATCH_VERSION:
            return False
        return release is None or metadata.get("tag") == release.get("tag")
    except Exception:
        return False


def _latest_existing_runtime() -> Optional[str]:
    root = _runtime_root()
    if not os.path.isdir(root):
        return None
    candidates = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if _runtime_is_valid(path):
            candidates.append((os.path.getmtime(path), path))
    return max(candidates, default=(0, None))[1]


def _candidate_executable(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    home = os.path.expanduser("~")
    candidates: List[str] = []
    if name == "bun":
        bun_root = os.environ.get("BUN_INSTALL", "").strip() or os.path.join(home, ".bun")
        candidates.append(os.path.join(bun_root, "bin", "bun.exe" if sys.platform == "win32" else "bun"))
    if sys.platform == "win32":
        candidates.extend(
            [
                os.path.join(home, ".bun", "bin", f"{name}.exe"),
                os.path.join(home, ".bun", "bin", f"{name}.cmd"),
                os.path.join(os.environ.get("APPDATA", ""), "npm", f"{name}.cmd"),
                os.path.join(os.environ.get("ProgramFiles", ""), "nodejs", f"{name}.cmd"),
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "nodejs", f"{name}.cmd"),
            ]
        )
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _bun_command() -> Optional[List[str]]:
    override = os.environ.get("GLM_PROXY_BUN_CMD", "").strip()
    if override:
        return shlex.split(override, posix=(sys.platform != "win32"))
    bun = _candidate_executable("bun")
    if bun:
        return [bun]
    npx = _candidate_executable("npx")
    if npx:
        return [npx, "--yes", "--package", BUN_NPM_PACKAGE, "bun"]
    return None


def _automatic_bun_install_command() -> Optional[List[str]]:
    override = os.environ.get("GLM_PROXY_BUN_INSTALL_CMD", "").strip()
    if override:
        return shlex.split(override, posix=(sys.platform != "win32"))
    if sys.platform == "win32":
        powershell = _candidate_executable("powershell") or _candidate_executable("pwsh")
        if not powershell:
            return None
        return [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "& ([scriptblock]::Create((irm bun.sh/install.ps1)))",
        ]
    shell = _candidate_executable("bash") or _candidate_executable("sh")
    curl = _candidate_executable("curl")
    if shell and curl:
        return [shell, "-c", 'curl -fsSL https://bun.sh/install | bash']
    return None


def _install_bun_automatically(log_fn=None) -> List[str]:
    command = _automatic_bun_install_command()
    if not command:
        raise RuntimeError(
            "GLM proxy needs Bun. Install Bun (or Node.js with npx), then retry."
        )
    _log = log_fn or _log_noop
    _log("📦 GLM proxy: installing the Bun JavaScript runtime...")
    kwargs: Dict[str, Any] = {}
    if sys.platform == "win32":
        try:
            from shutdown_utils import subprocess_no_window_kwargs

            kwargs.update(subprocess_no_window_kwargs())
        except Exception:
            pass
    result = run_logged_subprocess(
        command,
        log_fn=_log,
        timeout=BUN_INSTALL_TIMEOUT_SECONDS,
        popen_kwargs=kwargs,
    )
    bun = _bun_command()
    if result["returncode"] != 0 or not bun:
        raise RuntimeError(
            "Automatic Bun installation failed. " + str(result.get("output") or "")
        )
    _log("✅ GLM proxy: Bun is ready.")
    return bun


def _download_archive(url: str) -> bytes:
    errors: List[str] = []
    curl = _candidate_executable("curl")
    if curl:
        try:
            result = subprocess.run(
                [
                    curl,
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--connect-timeout",
                    "30",
                    "--max-time",
                    str(PROXY_ARCHIVE_DOWNLOAD_TIMEOUT_SECONDS),
                    url,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=PROXY_ARCHIVE_DOWNLOAD_TIMEOUT_SECONDS + 10,
            )
            data = bytes(result.stdout or b"")
            if result.returncode == 0 and data.startswith(b"PK"):
                return data
            errors.append(bytes(result.stderr or b"").decode("utf-8", errors="replace"))
        except Exception as exc:
            errors.append(str(exc))
    try:
        response = requests.get(url, headers=_github_headers(), timeout=PROXY_ARCHIVE_DOWNLOAD_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = bytes(response.content or b"")
        if not data.startswith(b"PK"):
            raise RuntimeError("response was not a ZIP archive")
        return data
    except Exception as exc:
        errors.append(str(exc))
    raise RuntimeError("Could not download zcode-api: " + "; ".join(filter(None, errors)))


def _find_archive_root(extract_dir: str) -> str:
    for root, dirs, files in os.walk(extract_dir):
        if "package.json" in files and os.path.isfile(os.path.join(root, "src", "index.ts")):
            return root
        dirs[:] = dirs[:4]
    raise RuntimeError("Downloaded zcode-api archive did not contain src/index.ts")


def _patch_credentials_store(runtime_dir: str) -> None:
    store_path = os.path.join(runtime_dir, "src", "auth", "store.ts")
    try:
        source = Path(store_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Downloaded zcode-api has no credential store: {exc}") from exc
    if "ZCODE_PROXY_CREDENTIALS_PATH" in source:
        return

    patterns = (
        (
            r"const STORE_FILE\s*=\s*join\(homedir\(\),\s*[\"']\.zcode-proxy[\"'],\s*[\"']credentials\.json[\"']\)\s*;",
            'const STORE_FILE = process.env.ZCODE_PROXY_CREDENTIALS_PATH '
            '?? join(homedir(), ".zcode-proxy", "credentials.json");',
        ),
        (
            r"const STORE_FILE\s*=\s*join\(STORE_DIR,\s*[\"']credentials\.json[\"']\)\s*;",
            'const STORE_FILE = process.env.ZCODE_PROXY_CREDENTIALS_PATH '
            '?? join(STORE_DIR, "credentials.json");',
        ),
        (
            r"const CREDENTIALS_FILE\s*=\s*join\(homedir\(\),\s*[\"']\.zcode-proxy[\"'],\s*[\"']credentials\.json[\"']\)\s*;",
            'const CREDENTIALS_FILE = process.env.ZCODE_PROXY_CREDENTIALS_PATH '
            '?? join(homedir(), ".zcode-proxy", "credentials.json");',
        ),
    )
    patched = source
    for pattern, replacement in patterns:
        patched, count = re.subn(pattern, replacement, patched, count=1)
        if count:
            break
    else:
        patched, count = re.subn(
            r"join\(homedir\(\),\s*[\"']\.zcode-proxy[\"'],\s*[\"']credentials\.json[\"']\)",
            'process.env.ZCODE_PROXY_CREDENTIALS_PATH ?? join(homedir(), ".zcode-proxy", "credentials.json")',
            patched,
            count=1,
        )
        if not count:
            raise RuntimeError("Could not patch zcode-api for isolated account credentials")
    Path(store_path).write_text(patched, encoding="utf-8")


def _patch_numbered_account_switch(runtime_dir: str) -> None:
    """Route numbered logins through Z.AI's own switch-account screen."""
    index_path = os.path.join(runtime_dir, "src", "index.ts")
    try:
        source = Path(index_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Downloaded zcode-api has no CLI entrypoint: {exc}") from exc
    marker = "GLOSSARION_AUTHZA_ACCOUNT_SWITCH"
    if marker in source:
        return

    insertion = f'''// {marker}
  // Z.AI's consent page otherwise reuses the browser's current account. Its
  // own "Switch account" control routes through this login URL while keeping
  // the original authorize target as the post-login redirect.
  if (process.env.ZCODE_OAUTH_FORCE_ACCOUNT_SELECTION === "1") {{
    try {{
      const authorizeTarget = new URL(url);
      if (authorizeTarget.protocol === "https:" && authorizeTarget.hostname === "chat.z.ai") {{
        const redirect = `${{authorizeTarget.pathname}}${{authorizeTarget.search}}`;
        url = `https://chat.z.ai/auth?redirect=${{encodeURIComponent(redirect)}}&switch_account=true`;
      }}
    }} catch {{ /* retain the original authorize URL */ }}
  }}
'''
    patched, count = re.subn(
        r"(function\s+openBrowser\(url:\s*string\):\s*void\s*\{\s*)",
        lambda match: match.group(1) + insertion,
        source,
        count=1,
    )
    if not count:
        raise RuntimeError("Could not patch zcode-api for numbered account switching")
    Path(index_path).write_text(patched, encoding="utf-8")


def _patch_login_plan_only_auth(runtime_dir: str) -> None:
    """Support JWT-only login while retaining opt-in API-key provisioning."""
    index_path = os.path.join(runtime_dir, "src", "index.ts")
    try:
        source = Path(index_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Downloaded zcode-api has no CLI entrypoint: {exc}") from exc
    marker = "GLOSSARION_ZCODE_LOGIN_PLAN_JWT_ONLY"
    if marker in source:
        return

    old_login = '''    const { accessToken, userId, jwt } = await runOAuth(provider);
    console.log("\\nResolving API key...");
    const resolver = new KeyResolver();
    cred = await resolver.resolveCodingPlanCredential(accessToken, provider, userId);
    if (jwt) cred.jwt = jwt;'''
    new_login = f'''    const {{ accessToken, userId, jwt }} = await runOAuth(provider);
    // {marker}
    if (process.env.GLOSSARION_ZCODE_LOGIN_PLAN_ONLY === "1" && provider === "zai") {{
      if (!jwt) throw new Error("ZCode login did not return a login-plan JWT");
      cred = {{ apiKey: "zcode-login", provider: "zai", userId, jwt }};
      console.log("\\nUsing ZCode login-plan credential (no API key provisioned).");
    }} else {{
      console.log("\\nResolving API key...");
      const resolver = new KeyResolver();
      cred = await resolver.resolveCodingPlanCredential(accessToken, provider, userId);
      if (jwt) cred.jwt = jwt;
    }}'''
    if old_login not in source:
        raise RuntimeError("Could not patch zcode-api's API-key-provisioning login path")
    patched = source.replace(old_login, new_login, 1)

    api_key_log = '  console.log(`  API Key: ${cred.apiKey.substring(0, 12)}...`);'
    credential_log = (
        '  if (cred.apiKey === "zcode-login" && cred.jwt) '
        'console.log("  Login credential: ZCode JWT");\n'
        '  else console.log("  API Key: stored securely");'
    )
    patched, log_count = re.subn(
        re.escape(api_key_log),
        lambda _match: credential_log,
        patched,
    )
    if log_count < 2:
        raise RuntimeError("Could not patch zcode-api's API-key login/status output")
    Path(index_path).write_text(patched, encoding="utf-8")


def _patch_zcode_login_plan_endpoint(runtime_dir: str) -> None:
    """Support current login-plan and general-API upstream wire formats.

    zcode-api 2.6.0 still points ``start-plan`` at the retired OpenAI-style
    ``/api/v1/zcode-plan/chat/completions`` route.  Current ZCode desktop uses
    the login JWT with ``/api/v1/zcode-plan/anthropic/v1/messages`` instead.
    General-API mode instead uses the provisioned project key with Z.AI's
    OpenAI-compatible ``/api/paas/v4`` endpoint. Keep one OpenAI-compatible
    local surface and select the upstream format from the managed config.
    """
    upstream_path = os.path.join(runtime_dir, "src", "proxy", "upstream.ts")
    handler_path = os.path.join(runtime_dir, "src", "proxy", "handler.ts")
    try:
        upstream = Path(upstream_path).read_text(encoding="utf-8")
        handler = Path(handler_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Downloaded zcode-api has no proxy routing sources: {exc}") from exc

    upstream_marker = "GLOSSARION_ZCODE_LOGIN_PLAN_ANTHROPIC"
    if upstream_marker not in upstream:
        upstream, base_count = re.subn(
            r'const\s+STARTPLAN_OPENAI_BASE\s*=\s*["\'][^"\']+["\']\s*;',
            f'// {upstream_marker}\nconst STARTPLAN_ANTHROPIC_BASE = '
            '"https://zcode.z.ai/api/v1/zcode-plan/anthropic";',
            upstream,
            count=1,
        )
        upstream, route_count = re.subn(
            r'return\s+`\$\{STARTPLAN_OPENAI_BASE\}/chat/completions`\s*;',
            'return `${STARTPLAN_ANTHROPIC_BASE}/v1/messages`;',
            upstream,
            count=1,
        )
        if not base_count or not route_count:
            raise RuntimeError("Could not patch zcode-api's obsolete login-plan endpoint")

    handler_marker = "GLOSSARION_ZCODE_DUAL_ACCESS_ROUTING"
    if handler_marker not in handler:
        routing_pattern = re.compile(
            r'const\s+startPlan\s*=\s*config\.plan\s*===\s*["\']start-plan["\']\s*;\s*'
            r'const\s+translateAnthropicToOpenAI\s*=\s*format\s*===\s*["\']anthropic["\']\s*&&\s*startPlan\s*;\s*'
            r'const\s+translateOpenAIToAnthropic\s*=\s*format\s*===\s*["\']openai["\']\s*&&\s*!startPlan\s*;\s*'
            r'const\s+upstreamFormat:\s*Format\s*=\s*startPlan\s*\?\s*["\']openai["\']\s*:\s*["\']anthropic["\']\s*;',
            re.MULTILINE,
        )
        replacement = (
            f'// {handler_marker}\n'
            '  const startPlan = config.plan === "start-plan";\n'
            '  const generalApi = !startPlan && '
            'config.providers[config.provider].openaiBase.includes("/api/paas/v4");\n'
            '  const translateAnthropicToOpenAI = generalApi && format === "anthropic";\n'
            '  const translateOpenAIToAnthropic = !generalApi && format === "openai";\n'
            '  const upstreamFormat: Format = generalApi ? "openai" : "anthropic";'
        )
        handler, routing_count = routing_pattern.subn(replacement, handler, count=1)
        if not routing_count:
            raise RuntimeError("Could not patch zcode-api's login-plan wire format")

    Path(upstream_path).write_text(upstream, encoding="utf-8")
    Path(handler_path).write_text(handler, encoding="utf-8")


def _write_runtime_metadata(runtime_dir: str, release: Dict[str, Any]) -> None:
    with open(_runtime_metadata_path(runtime_dir), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "tag": release["tag"],
                "revision": release["revision"],
                "version": release["version"],
                "patch_version": RUNTIME_PATCH_VERSION,
                "source": release["archive_url"],
                "updated_at": int(time.time()),
            },
            handle,
            indent=2,
        )
        handle.write("\n")


def _download_proxy_runtime(release: Dict[str, Any], runtime_dir: str, log_fn=None) -> None:
    _log = log_fn or _log_noop
    _log(f"⬇️ GLM proxy: downloading zcode-api {release['version']}...")
    archive_data = _download_archive(release["archive_url"])
    parent = os.path.dirname(runtime_dir)
    os.makedirs(parent, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="download-", dir=parent) as temp_dir:
        extract_dir = os.path.join(temp_dir, "extract")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
            archive.extractall(extract_dir)
        archive_root = _find_archive_root(extract_dir)
        _patch_credentials_store(archive_root)
        _patch_numbered_account_switch(archive_root)
        _patch_login_plan_only_auth(archive_root)
        _patch_zcode_login_plan_endpoint(archive_root)
        _write_runtime_metadata(archive_root, release)
        if os.path.exists(runtime_dir):
            shutil.rmtree(runtime_dir)
        shutil.copytree(archive_root, runtime_dir)


def _ensure_proxy_runtime(log_fn=None) -> str:
    release = _latest_proxy_release()
    runtime_dir = os.path.join(_runtime_root(), _safe_runtime_segment(release["tag"]))
    if _runtime_is_valid(runtime_dir, release):
        return runtime_dir
    try:
        _download_proxy_runtime(release, runtime_dir, log_fn=log_fn)
        return runtime_dir
    except Exception as exc:
        cached = _latest_existing_runtime()
        if cached:
            (log_fn or _log_noop)(
                f"⚠️ GLM proxy: update failed; using cached runtime. {str(exc)[:500]}"
            )
            return cached
        raise


def _dependency_marker(runtime_dir: str) -> str:
    return os.path.join(runtime_dir, ".glossarion-dependencies.json")


def _ensure_dependencies(runtime_dir: str, bun: List[str], log_fn=None) -> None:
    marker = _dependency_marker(runtime_dir)
    if os.path.isdir(os.path.join(runtime_dir, "node_modules")) and os.path.isfile(marker):
        return
    _log = log_fn or _log_noop
    _log("📦 GLM proxy: installing proxy packages...")
    command = [*bun, "install"]
    if os.path.isfile(os.path.join(runtime_dir, "bun.lock")) or os.path.isfile(
        os.path.join(runtime_dir, "bun.lockb")
    ):
        command.append("--frozen-lockfile")
    result = run_logged_subprocess(
        command,
        log_fn=_log,
        timeout=DEPENDENCY_INSTALL_TIMEOUT_SECONDS,
        cwd=runtime_dir,
    )
    if result["returncode"] != 0:
        raise RuntimeError("GLM proxy package installation failed. " + result["output"])
    with open(marker, "w", encoding="utf-8") as handle:
        json.dump({"installed_at": int(time.time())}, handle)
        handle.write("\n")
    _log("✅ GLM proxy: proxy packages are ready.")


def _ensure_runtime_and_dependencies(log_fn=None) -> Tuple[str, List[str]]:
    runtime_dir = _ensure_proxy_runtime(log_fn=log_fn)
    bun = _bun_command()
    if not bun:
        bun = _install_bun_automatically(log_fn=log_fn)
    _ensure_dependencies(runtime_dir, bun, log_fn=log_fn)
    return runtime_dir, bun


def _read_or_create_secrets(account_id: Optional[int] = None) -> Dict[str, str]:
    path = _secrets_path(account_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data: Dict[str, Any] = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("proxy_api_key", "sk-glossarion-" + secrets.token_urlsafe(24))
    data.setdefault("credential_secret", secrets.token_urlsafe(48))
    data.setdefault("device_mid", str(uuid.uuid4()))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return {
        key: str(data[key])
        for key in ("proxy_api_key", "credential_secret", "device_mid")
    }


def _proxy_api_key(account_id: Optional[int] = None) -> str:
    env_value = os.environ.get(_account_env_name("GLM_PROXY_API_KEY", account_id), "").strip()
    if not env_value and _normalize_account_id(account_id) != 0:
        env_value = os.environ.get("GLM_PROXY_API_KEY", "").strip()
    if env_value:
        return env_value
    return _read_or_create_secrets(account_id)["proxy_api_key"]


def _ensure_account_config(account_id: Optional[int] = None) -> str:
    account = _normalize_account_id(account_id)
    account_dir = _account_dir(account)
    os.makedirs(account_dir, exist_ok=True)
    secrets_data = _read_or_create_secrets(account)
    models_yaml = "\n".join(f'    - "{model}"' for model in DEFAULT_MODELS)
    general_api = uses_general_api()
    plan = "coding-plan" if general_api else "start-plan"
    enabled_flag = "false" if general_api else "true"
    config = f'''server:
  host: "{DEFAULT_PROXY_HOST}"
  port: {_get_proxy_port(account)}
auth:
  mode: "oauth"
  proxyApiKey: {json.dumps(secrets_data["proxy_api_key"])}
  oauthCredentialsPath: {json.dumps(_credentials_path(account))}
provider: "zai"
plan: "{plan}"
providers:
  zai:
    anthropicBase: "https://api.z.ai/api/anthropic"
    openaiBase: {json.dumps(GENERAL_API_BASE if general_api else "https://api.z.ai/api/coding/paas/v4")}
models:
{models_yaml}
responses:
  enabled: true
identity:
  appVersion: {json.dumps(ZCODE_APP_VERSION)}
  deviceMid: {json.dumps(secrets_data["device_mid"])}
  sourceTitle: "glossarion"
endpointRouting:
  enabled: {enabled_flag}
clientSigning:
  enabled: {enabled_flag}
mcp:
  enabled: false
'''
    path = _config_path(account)
    current = ""
    try:
        current = Path(path).read_text(encoding="utf-8")
    except OSError:
        pass
    if current != config:
        Path(path).write_text(config, encoding="utf-8")
    return path


def _runtime_env(account_id: Optional[int] = None) -> Dict[str, str]:
    account = _normalize_account_id(account_id)
    values = dict(os.environ)
    secrets_data = _read_or_create_secrets(account)
    values.update(
        {
            "ZCODE_PROXY_CONFIG": _config_path(account),
            "ZCODE_PROXY_CREDENTIALS_PATH": _credentials_path(account),
            "ZCODE_PROXY_CREDENTIAL_SECRET": secrets_data["credential_secret"],
            "GLOSSARION_ZCODE_LOGIN_PLAN_ONLY": "0" if uses_general_api() else "1",
        }
    )
    return values


def has_credentials(account_id: Optional[int] = None) -> bool:
    if _external_proxy_url(account_id):
        return bool(_proxy_api_key(account_id))
    path = _credentials_path(account_id)
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 10
    except OSError:
        return False


def can_auto_provision_general_api_key(account_id: Optional[int] = None) -> bool:
    """Return whether polling may retrieve a General API key via Z.AI login."""
    return uses_general_api() and _external_proxy_url(account_id) is None


def _build_headers(account_id: Optional[int] = None) -> Dict[str, str]:
    api_key = _proxy_api_key(account_id)
    return {
        "Authorization": f"Bearer {api_key}",
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }


def _build_stream_headers(account_id: Optional[int] = None) -> Dict[str, str]:
    """Build headers that keep SSE uncompressed and flushable end to end."""
    return {
        **_build_headers(account_id),
        "Accept": "text/event-stream",
        # The managed Node proxy forwards the client's accepted encoding to
        # Z.AI. Compressed SSE can sit in gzip/Brotli buffers until generation
        # completes, which turns a nominal stream into one final burst.
        "Accept-Encoding": "identity",
        "Cache-Control": "no-cache",
    }


def check_proxy_health(account_id: Optional[int] = None) -> Dict[str, Any]:
    url = f"{get_proxy_url(account_id)}{HEALTH_ENDPOINT}"
    try:
        response = requests.get(url, headers=_build_headers(account_id), timeout=5)
        if response.status_code == 200:
            try:
                details = response.json()
            except Exception:
                details = {}
            return {"healthy": True, "details": details}
        return {"healthy": False, "error": f"HTTP {response.status_code}"}
    except requests.ConnectionError:
        return {"healthy": False, "error": "Connection refused"}
    except Exception as exc:
        return {"healthy": False, "error": str(exc)}


def _hidden_process_kwargs() -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        try:
            from shutdown_utils import subprocess_no_window_kwargs

            kwargs.update(subprocess_no_window_kwargs())
        except Exception:
            pass
    else:
        kwargs["start_new_session"] = True
    return kwargs


def _login(account_id: Optional[int] = None, log_fn=None) -> None:
    account = _normalize_account_id(account_id)
    if _external_proxy_url(account):
        raise RuntimeError("A remote GLM_PROXY_URL cannot perform local browser login")
    runtime_dir, bun = _ensure_runtime_and_dependencies(log_fn=log_fn)
    _ensure_account_config(account)
    _log = log_fn or _log_noop
    label = "default account" if account == 0 else f"account #{account}"
    login_env = _runtime_env(account)
    # A numbered route represents a different account slot, so take the same
    # account-selection path as Z.AI's visible "Switch account" control.
    login_env["ZCODE_OAUTH_FORCE_ACCOUNT_SELECTION"] = "1" if account > 0 else "0"
    _log(f"🔐 GLM proxy: opening Z.AI login for {label}...")
    if account > 0:
        _log(f"🔁 GLM proxy: forcing Z.AI account selection for account #{account}...")
    result = run_logged_subprocess(
        [*bun, "run", _runtime_entrypoint(runtime_dir), "auth", "login", "zai"],
        log_fn=_log,
        timeout=LOGIN_TIMEOUT_SECONDS,
        cwd=_account_dir(account),
        env=login_env,
    )
    if result["returncode"] != 0 or not has_credentials(account):
        suffix = " (login timed out)" if result.get("timed_out") else ""
        raise RuntimeError(
            f"Z.AI login did not complete{suffix}. " + str(result.get("output") or "")
        )
    _log(f"✅ GLM proxy: Z.AI login saved for {label}.")


def ensure_general_api_key(
    account_id: Optional[int] = None,
    *,
    log_fn=None,
    force: bool = False,
) -> bool:
    """Retrieve or refresh the selected account's provisioned General API key.

    Returns ``True`` when the browser provisioning flow was run. The upstream
    login command exchanges the Z.AI OAuth result through ``KeyResolver`` and
    stores the resulting project API key in the isolated encrypted credential
    file used by the model poller and proxy.
    """
    account = _normalize_account_id(account_id)
    if not uses_general_api():
        return False
    if has_credentials(account) and not force:
        return False
    if not can_auto_provision_general_api_key(account):
        raise RuntimeError("An external GLM proxy cannot auto-provision a Z.AI General API key")
    _log = log_fn or _log_console
    label = "default account" if account == 0 else f"account #{account}"
    action = "refreshing" if force else "retrieving"
    _log(f"🔑 AuthZA: {action} the Z.AI General API key for {label} before model polling…")
    _login(account_id=account, log_fn=_log)
    if not has_credentials(account):
        raise RuntimeError(f"Z.AI General API key retrieval did not save a key for {label}")
    _log(f"✅ AuthZA: General API key ready for {label}; polling models now.")
    return True


def open_login(log_fn=None, account_id: Optional[int] = None) -> str:
    """Run the upstream Z.AI browser login and start that account's proxy."""
    _login(account_id=account_id, log_fn=log_fn)
    restart_proxy(account_id=account_id, log_fn=log_fn, auto_login=False)
    return get_proxy_url(account_id)


def ensure_proxy_running(
    log_fn=None,
    account_id: Optional[int] = None,
    auto_login: bool = True,
    notify_started: bool = True,
) -> Dict[str, Any]:
    account = _normalize_account_id(account_id)
    health = check_proxy_health(account)
    if health.get("healthy"):
        return {"running": True, **health}
    if _external_proxy_url(account):
        return {
            "running": False,
            "error": f"Configured GLM proxy is unavailable: {health.get('error', 'unknown error')}",
        }

    with _proxy_launch_lock:
        health = check_proxy_health(account)
        if health.get("healthy"):
            return {"running": True, **health}
        if not has_credentials(account):
            if not auto_login:
                return {"running": False, "needs_login": True, "error": "Z.AI login required"}
            _login(account, log_fn=log_fn)

        runtime_dir, bun = _ensure_runtime_and_dependencies(log_fn=log_fn)
        config_path = _ensure_account_config(account)
        previous = _proxy_processes.get(account)
        if previous is not None and previous.poll() is not None:
            _proxy_processes.pop(account, None)
        launched = False
        if account not in _proxy_processes:
            command = [*bun, "run", _runtime_entrypoint(runtime_dir), "serve", config_path]
            process = subprocess.Popen(
                command,
                cwd=_account_dir(account),
                env=_runtime_env(account),
                **_hidden_process_kwargs(),
            )
            _proxy_processes[account] = process
            launched = True
            (log_fn or _log_noop)(
                f"🟢 GLM proxy: started account {account or 'default'} on {get_proxy_url(account)} "
                f"(PID {process.pid})."
            )

        deadline = time.monotonic() + 30
        last_health = health
        while time.monotonic() < deadline:
            process = _proxy_processes.get(account)
            if process is not None and process.poll() is not None:
                _proxy_processes.pop(account, None)
                return {"running": False, "error": f"GLM proxy exited with code {process.returncode}"}
            time.sleep(0.25)
            last_health = check_proxy_health(account)
            if last_health.get("healthy"):
                if launched and notify_started:
                    _notify_proxy_started(account)
                return {"running": True, **last_health}
        return {
            "running": False,
            "error": f"GLM proxy did not become healthy: {last_health.get('error', 'timeout')}",
        }


def restart_proxy(
    account_id: Optional[int] = None,
    log_fn=None,
    auto_login: bool = True,
) -> Dict[str, Any]:
    account = _normalize_account_id(account_id)
    process = _proxy_processes.pop(account, None)
    if process is not None and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
    return ensure_proxy_running(log_fn=log_fn, account_id=account, auto_login=auto_login)


def shutdown_proxy(account_id: Optional[int] = None) -> None:
    account = _normalize_account_id(account_id)
    process = _proxy_processes.pop(account, None)
    if process is not None and process.poll() is None:
        try:
            process.terminate()
        except Exception:
            pass


def cancel_stream() -> None:
    """Signal active streams without allowing a later reset to revive them."""
    global _cancel_generation
    with _cancel_state_lock:
        _cancel_generation += 1
        _cancel_event.set()
    with _active_response_lock:
        responses = list(_active_responses.values())
    for response in responses:
        try:
            response.close()
        except Exception:
            pass


def reset_cancel() -> None:
    """Allow new streams while keeping prior cancelled generations invalid."""
    with _cancel_state_lock:
        _cancel_event.clear()


def is_cancelled() -> bool:
    return _cancel_event.is_set()


def capture_cancel_generation() -> int:
    """Return the cancellation generation a new AuthZA operation belongs to."""
    with _cancel_state_lock:
        return _cancel_generation


def is_cancel_generation_cancelled(cancel_generation: Optional[int]) -> bool:
    """Return whether Stop invalidated a particular AuthZA operation."""
    with _cancel_state_lock:
        if _cancel_event.is_set():
            return True
        return (
            cancel_generation is not None
            and int(cancel_generation) != _cancel_generation
        )


def _register_response(response: Any) -> None:
    with _active_response_lock:
        _active_responses[id(response)] = response


def _unregister_response(response: Any) -> None:
    with _active_response_lock:
        _active_responses.pop(id(response), None)


def _raise_if_cancelled(cancel_generation: Optional[int] = None) -> None:
    if is_cancel_generation_cancelled(cancel_generation):
        raise RuntimeError("GLM proxy: stream cancelled by user")


def _extract_error(response: requests.Response) -> str:
    try:
        data = response.json()
        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error)
        if error:
            return str(error)
        if isinstance(data, dict) and data.get("message"):
            return str(data["message"])
    except Exception:
        pass
    try:
        return (response.text or "").strip()[:2000] or "unknown error"
    except Exception:
        return "unknown error"


def _payload(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: Optional[float],
    max_tokens: Optional[int],
    stream: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": (model or "glm-5.3").strip(),
        "messages": messages,
        "stream": stream,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _parse_chat_response(data: Dict[str, Any]) -> Dict[str, Any]:
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("GLM proxy returned no choices")
    choice = choices[0] or {}
    message = choice.get("message") or {}
    return {
        "content": _content_text(message.get("content")),
        "finish_reason": choice.get("finish_reason") or "stop",
        "usage": data.get("usage"),
        "raw_response": data,
    }


def _ensure_for_request(account_id: int, log_fn=None, auto_login: bool = True) -> None:
    status = ensure_proxy_running(log_fn=log_fn, account_id=account_id, auto_login=auto_login)
    if not status.get("running"):
        raise RuntimeError(status.get("error") or "GLM proxy is not running")


def send_message(
    messages: List[Dict[str, Any]],
    model: str = "glm-5.3",
    temperature: Optional[float] = 0.7,
    max_tokens: int = 8192,
    timeout: float = 300,
    log_fn=None,
    account_id: Optional[int] = None,
    auto_login: bool = True,
    connect_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    account = _normalize_account_id(account_id)
    _raise_if_cancelled()
    _ensure_for_request(account, log_fn=log_fn, auto_login=auto_login)
    payload = _payload(messages, model, temperature, max_tokens, stream=False)
    try:
        response = requests.post(
            f"{get_proxy_url(account)}{CHAT_COMPLETIONS_ENDPOINT}",
            headers=_build_headers(account),
            json=payload,
            timeout=(connect_timeout or min(30, timeout), timeout),
        )
    except requests.ConnectionError as exc:
        raise RuntimeError(f"GLM proxy connection failed: {exc}") from exc
    except requests.Timeout as exc:
        raise RuntimeError(f"GLM proxy request timed out after {timeout}s") from exc
    _raise_if_cancelled()
    if response.status_code != 200:
        raise RuntimeError(f"GLM proxy: HTTP {response.status_code} - {_extract_error(response)}")
    try:
        return _parse_chat_response(response.json())
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"GLM proxy returned invalid JSON: {exc}") from exc


def _iter_sse_lines(response: Any) -> Iterable[str]:
    try:
        lines = response.iter_lines(decode_unicode=True, chunk_size=1)
    except TypeError:
        lines = response.iter_lines()
    for raw in lines:
        if isinstance(raw, bytes):
            yield raw.decode("utf-8", errors="replace")
        else:
            yield str(raw or "")


def _log_text_stream(text: str, log_buf: List[str], log_fn) -> None:
    """Emit readable live text without creating one GUI row per SSE token."""
    if not text:
        return
    combined = "".join(log_buf) + text
    for tag in ("</h1>", "</h2>", "</h3>", "</h4>", "</h5>", "</h6>", "</p>"):
        combined = combined.replace(tag, tag + "\n")
    if "\n" in combined:
        parts = combined.split("\n")
        for part in parts[:-1]:
            log_fn(part)
        log_buf[:] = [parts[-1]]
    else:
        log_buf[:] = [combined]
        if len(combined) > 150:
            log_fn(combined)
            log_buf.clear()


def _stream_error(event: Dict[str, Any]) -> Optional[str]:
    error = event.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or error)
    return str(error) if error else None


def _close_stream_response(response: Any, stream_context: Any = None) -> None:
    """Close either an httpx streaming context or a requests response."""
    try:
        if stream_context is not None:
            stream_context.__exit__(None, None, None)
        elif response is not None:
            response.close()
    except Exception:
        pass


def send_message_stream(
    messages: List[Dict[str, Any]],
    model: str = "glm-5.3",
    temperature: Optional[float] = 0.7,
    max_tokens: int = 8192,
    timeout: float = 300,
    log_fn=None,
    log_stream: bool = True,
    account_id: Optional[int] = None,
    auto_login: bool = True,
    connect_timeout: Optional[float] = None,
    cancel_generation: Optional[int] = None,
) -> Dict[str, Any]:
    """Send an always-streaming OpenAI-compatible request through AuthZA."""
    _log = log_fn or _log_noop
    request_cancel_generation = (
        capture_cancel_generation()
        if cancel_generation is None
        else int(cancel_generation)
    )
    account = _normalize_account_id(account_id)
    _raise_if_cancelled(request_cancel_generation)
    _ensure_for_request(account, log_fn=log_fn, auto_login=auto_login)
    _raise_if_cancelled(request_cancel_generation)
    payload = _payload(messages, model, temperature, max_tokens, stream=True)
    proxy_url = get_proxy_url(account)
    _log(f"🌊 AuthZA: streaming from {proxy_url} (model={payload['model']})")
    url = f"{proxy_url}{CHAT_COMPLETIONS_ENDPOINT}"
    headers = _build_stream_headers(account)
    response = None
    stream_context = None
    try:
        if httpx is not None:
            effective_connect_timeout = connect_timeout or min(30, timeout)
            timeout_config = httpx.Timeout(timeout, connect=effective_connect_timeout)
            stream_context = httpx.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=timeout_config,
            )
            response = stream_context.__enter__()
        else:
            _log("AuthZA: httpx is unavailable; falling back to requests streaming.")
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(connect_timeout or min(30, timeout), timeout),
                stream=True,
            )
    except requests.ConnectionError as exc:
        raise RuntimeError(f"GLM proxy connection failed: {exc}") from exc
    except requests.Timeout as exc:
        raise RuntimeError(f"GLM proxy request timed out after {timeout}s") from exc
    except Exception as exc:
        if httpx is not None and isinstance(exc, httpx.ConnectError):
            raise RuntimeError(f"GLM proxy connection failed: {exc}") from exc
        if httpx is not None and isinstance(exc, httpx.TimeoutException):
            raise RuntimeError(f"GLM proxy request timed out after {timeout}s") from exc
        raise
    try:
        _raise_if_cancelled(request_cancel_generation)
    except RuntimeError:
        _close_stream_response(response, stream_context)
        raise
    if response.status_code != 200:
        if stream_context is not None:
            try:
                response.read()
            except Exception:
                pass
        message = _extract_error(response)
        _close_stream_response(response, stream_context)
        raise RuntimeError(f"GLM proxy: HTTP {response.status_code} - {message}")

    content: List[str] = []
    finish_reason: Optional[str] = None
    usage = None
    got_first_data = False
    saw_done_marker = False
    content_log_buf: List[str] = []
    reasoning_log_buf: List[str] = []
    reasoning_started = False
    text_started = False
    started = time.monotonic()
    _register_response(response)
    try:
        for line in _iter_sse_lines(response):
            _raise_if_cancelled(request_cancel_generation)
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                saw_done_marker = True
                break
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            error = _stream_error(event)
            if error:
                raise RuntimeError(f"GLM proxy stream error: {error}")
            if not got_first_data:
                got_first_data = True
                _log(f"GLM proxy: first token in {time.monotonic() - started:.1f}s, streaming...")
            if event.get("usage") is not None:
                usage = event.get("usage")
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0] or {}
            delta = choice.get("delta") or {}
            reasoning = _content_text(delta.get("reasoning_content"))
            if reasoning and log_stream:
                if not reasoning_started:
                    reasoning_started = True
                    _log("[authza] Thinking...")
                _log_text_stream(reasoning, reasoning_log_buf, _log)
            text = _content_text(delta.get("content"))
            if text:
                content.append(text)
                if log_stream:
                    if not text_started:
                        text_started = True
                        if reasoning_started:
                            remainder = "".join(reasoning_log_buf).strip()
                            if remainder:
                                _log(f"    {remainder}")
                            reasoning_log_buf.clear()
                            _log("🧠 [authza] Thinking complete")
                            _log("─" * 50)
                        _log("📡 AuthZA: text streaming...")
                    _log_text_stream(text, content_log_buf, _log)
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])

        if log_stream and reasoning_log_buf:
            remainder = "".join(reasoning_log_buf).strip()
            if remainder:
                _log(f"    {remainder}")
        if log_stream and content_log_buf:
            remainder = "".join(content_log_buf).strip()
            if remainder:
                _log(remainder)
        _raise_if_cancelled(request_cancel_generation)
    finally:
        _unregister_response(response)
        _close_stream_response(response, stream_context)
    _raise_if_cancelled(request_cancel_generation)
    if finish_reason is None:
        _log("❌ GLM proxy: stream ended without an explicit finish_reason")
        raise RuntimeError("GLM proxy: stream ended without an explicit finish_reason")
    terminal_usage = {}
    if isinstance(usage, dict):
        terminal_usage = {
            key: usage.get(key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if key in usage
        }
    _log(
        "📊 AuthZA: terminal metadata "
        f"finish_reason={finish_reason!r}, max_tokens={max_tokens!r}, "
        f"usage={terminal_usage!r}"
    )
    _log(f"GLM proxy: stream finished in {time.monotonic() - started:.1f}s")
    return {
        "content": "".join(content),
        "finish_reason": finish_reason,
        "finish_reason_observed": True,
        "stream_done_observed": saw_done_marker,
        "usage": usage,
        "raw_response": None,
    }


def send_chat_completion(
    *,
    messages: List[Dict[str, Any]],
    model: str = "glm-5.3",
    temperature: Optional[float] = 0.7,
    max_tokens: int = 8192,
    timeout: float = 300,
    log_fn=None,
    connect_timeout: Optional[float] = None,
    account_id: Optional[int] = None,
    auto_login: bool = True,
    log_stream: bool = True,
    cancel_generation: Optional[int] = None,
    **_ignored: Any,
) -> Dict[str, Any]:
    """Compatibility entrypoint used by ``UnifiedClient`` (streaming by default)."""
    return send_message_stream(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        log_fn=log_fn,
        log_stream=log_stream,
        account_id=account_id,
        auto_login=auto_login,
        connect_timeout=connect_timeout,
        cancel_generation=cancel_generation,
    )


def fetch_available_models(
    account_id: Optional[int] = None,
    timeout: float = 10,
    *,
    auto_provision: bool = True,
    log_fn=None,
) -> List[str]:
    """Return model IDs visible in the selected AuthZA access mode."""
    account = _normalize_account_id(account_id)
    general_api = uses_general_api()
    if general_api and auto_provision:
        ensure_general_api_key(account, log_fn=log_fn)
    if not has_credentials(account):
        raise RuntimeError("GLM proxy has no saved Z.AI login for this account")
    runtime_dir, bun = _ensure_runtime_and_dependencies()
    env = _runtime_env(account)
    effective_timeout = max(1.0, float(timeout))
    if general_api:
        # The global /models endpoint regularly takes longer than the generic
        # eight-second provider poll budget. Keep it from failing before Z.AI
        # has returned the catalog.
        effective_timeout = max(
            effective_timeout,
            float(GENERAL_API_CATALOG_MIN_TIMEOUT_SECONDS),
        )
    env.update(
        {
            "ZCODE_APP_VERSION": ZCODE_APP_VERSION,
            "ZCODE_MODEL_CATALOG_TIMEOUT_MS": str(max(1000, int(effective_timeout * 1000))),
        }
    )
    if general_api:
        env["ZCODE_GENERAL_API_MODELS_ENDPOINT"] = GENERAL_API_MODELS_ENDPOINT
        catalog_script = _GENERAL_API_MODEL_CATALOG_SCRIPT
    else:
        env["ZCODE_LOGIN_PLAN_MODELS_ENDPOINT"] = LOGIN_PLAN_MODELS_ENDPOINT
        catalog_script = _LOGIN_PLAN_MODEL_CATALOG_SCRIPT
    _log = log_fn or _log_console
    endpoint = GENERAL_API_MODELS_ENDPOINT if general_api else LOGIN_PLAN_MODELS_ENDPOINT
    mode_label = "general API" if general_api else "login plan"
    _log(
        f"🌐 AuthZA: polling {mode_label} models from {endpoint} "
        f"(timeout={effective_timeout:g}s)…"
    )

    refreshed_key = False
    while True:
        result = run_logged_subprocess(
            [*bun, "-e", catalog_script],
            log_fn=None,
            timeout=max(2, effective_timeout + 2),
            cwd=runtime_dir,
            env=env,
        )
        if result["returncode"] == 0:
            break
        error_output = str(result.get("output") or "")
        auth_error = any(
            marker in error_output.casefold()
            for marker in (
                "http 401",
                "http 403",
                "unauthorized",
                "invalid api key",
                "api key is unavailable",
                "token expired",
            )
        )
        if general_api and auto_provision and auth_error and not refreshed_key:
            ensure_general_api_key(account, log_fn=_log, force=True)
            env = _runtime_env(account)
            env.update(
                {
                    "ZCODE_APP_VERSION": ZCODE_APP_VERSION,
                    "ZCODE_MODEL_CATALOG_TIMEOUT_MS": str(int(effective_timeout * 1000)),
                    "ZCODE_GENERAL_API_MODELS_ENDPOINT": GENERAL_API_MODELS_ENDPOINT,
                }
            )
            refreshed_key = True
            continue
        break
    if result["returncode"] != 0:
        raise RuntimeError(
            str(result.get("output") or f"Z.AI {mode_label} model catalog request failed")
        )
    try:
        output = str(result.get("output") or "")
        marker_line = next(
            (line for line in reversed(output.splitlines()) if line.startswith("GLOSSARION_MODELS=")),
            "",
        )
        payload = json.loads(marker_line.partition("=")[2])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Z.AI model catalog returned invalid JSON") from exc
    models = []
    seen = set()
    for model_id in payload if isinstance(payload, list) else []:
        value = str(model_id or "").strip()
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            models.append(value)
    if not models:
        mode_label = "general API" if general_api else "login plan"
        raise RuntimeError(f"Z.AI {mode_label} returned no model IDs")
    return models
