# antigravity_proxy.py - Antigravity Proxy integration
# Routes requests through frieser/antigravity-proxy:
# https://github.com/frieser/antigravity-proxy
#
# Glossarion keeps using model names prefixed with "antigravity/".
# This module translates them to the OpenAI-compatible model ids expected by
# frieser/antigravity-proxy and exposes the same Python API as the previous
# Antigravity adapter.

"""Antigravity Proxy adapter for Glossarion.

The current upstream proxy is an OpenAI-compatible local server. By default it
listens on http://localhost:3000 and exposes:

  - GET  /api/status
  - GET  /v1/models
  - POST /v1/chat/completions
  - GET  /oauth/start

Glossarion auto-downloads the current upstream proxy archive into the user's
Antigravity proxy data directory and launches that local runtime. This avoids
requiring users of the compiled app to install Git or manually update a checkout.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
import zipfile
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

try:
    import httpx
except ImportError:  # pragma: no cover - fallback path for stripped builds
    httpx = None

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_PROXY_URL = "http://localhost:3000"
PROXY_PACKAGE_NAME = "antigravity-proxy"
PROXY_GITHUB_API_TAGS = "https://api.github.com/repos/frieser/antigravity-proxy/tags"
PROXY_GITHUB_ARCHIVE_URL = (
    "https://github.com/frieser/antigravity-proxy/archive/refs/tags/{tag}.zip"
)
PROXY_DEFAULT_TAG = "v1.7.1"
BUN_NPM_PACKAGE = os.environ.get("ANTIGRAVITY_BUN_PACKAGE", "bun@latest")
RUNTIME_PATCH_VERSION = "2026-06-29-gemini35-flash-tiers"

ANTIGRAVITY_SITE_URL = "https://antigravity.google/changelog"
ANTIGRAVITY_CLIENT_VERSION_FALLBACK = "2.2.1"

CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"
MODELS_ENDPOINT = "/v1/models"
STATUS_ENDPOINT = "/api/status"
OAUTH_START_ENDPOINT = "/oauth/start"
CLAUDE_MAX_OUTPUT_TOKENS = 64000
# Antigravity's Cloud Code route rejects 65,536 even when the public Gemini model
# advertises that limit. Keep Gemini on the same 64k ceiling to avoid 400s.
GEMINI_MAX_OUTPUT_TOKENS = 64000

PROXY_REPO_URL = "https://github.com/frieser/antigravity-proxy"

# Module-level cancellation flag.
_cancel_event = threading.Event()

# Module-level proxy subprocess tracking.
_proxy_process: Optional[subprocess.Popen] = None
_proxy_launch_lock = threading.Lock()

# Auth browser tracking - only open the browser once per session.
_auth_browser_opened = False
_auth_browser_lock = threading.Lock()


def _log_noop(_: str) -> None:
    return None


def _proxy_command_for_humans() -> str:
    return "Glossarion auto-updates frieser/antigravity-proxy and launches it locally"


def _get_proxy_data_dir() -> str:
    """Return the stable data/config directory used for auto-launched proxy."""
    return os.environ.get(
        "ANTIGRAVITY_PROXY_DATA_DIR",
        os.path.join(os.path.expanduser("~"), ".config", "antigravity-proxy"),
    )


def _get_proxy_runtime_root(data_dir: str) -> str:
    return os.path.join(data_dir, "runtime")


def _safe_runtime_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "runtime"


def _parse_semver(value: str) -> tuple:
    match = re.search(r"v?(\d+)\.(\d+)\.(\d+)", value or "")
    if not match:
        return (0, 0, 0)
    return tuple(int(part) for part in match.groups())


def _version_to_str(version: tuple) -> str:
    return ".".join(str(part) for part in version)


def _github_headers() -> Dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Glossarion Antigravity Proxy Updater",
    }


def _latest_proxy_release() -> Dict[str, str]:
    """Return the current frieser/antigravity-proxy tag via GitHub HTTPS APIs."""
    override = os.environ.get("ANTIGRAVITY_PROXY_TAG", "").strip()
    if not override:
        version_override = os.environ.get("ANTIGRAVITY_PROXY_VERSION", "").strip()
        if version_override:
            override = version_override if version_override.startswith("v") else f"v{version_override}"

    if override:
        tag = override
        return {
            "tag": tag,
            "version": _version_to_str(_parse_semver(tag)),
            "archive_url": PROXY_GITHUB_ARCHIVE_URL.format(tag=tag),
        }

    try:
        resp = requests.get(PROXY_GITHUB_API_TAGS, headers=_github_headers(), timeout=15)
        resp.raise_for_status()
        tags = resp.json()
        candidates = []
        if isinstance(tags, list):
            for item in tags:
                if not isinstance(item, dict):
                    continue
                tag = str(item.get("name") or "").strip()
                if not tag:
                    continue
                candidates.append(
                    (
                        _parse_semver(tag),
                        {
                            "tag": tag,
                            "version": _version_to_str(_parse_semver(tag)),
                            "archive_url": PROXY_GITHUB_ARCHIVE_URL.format(tag=tag),
                        },
                    )
                )
        if candidates:
            return max(candidates, key=lambda item: item[0])[1]
    except Exception as exc:
        logger.debug("Could not resolve latest Antigravity proxy tag: %s", exc)

    return {
        "tag": PROXY_DEFAULT_TAG,
        "version": _version_to_str(_parse_semver(PROXY_DEFAULT_TAG)),
        "archive_url": PROXY_GITHUB_ARCHIVE_URL.format(tag=PROXY_DEFAULT_TAG),
    }


def _latest_proxy_version() -> str:
    return _latest_proxy_release()["version"]


def _absolute_url(base_url: str, value: str) -> str:
    if value.startswith("http://") or value.startswith("https://"):
        return value
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}/{value.lstrip('/')}"


def _script_urls_from_html(base_url: str, html: str) -> List[str]:
    urls: List[str] = []
    for match in re.finditer(r"""(?:src|href)=["']([^"']+\.js)["']""", html or "", re.IGNORECASE):
        url = _absolute_url(base_url, match.group(1))
        if url not in urls:
            urls.append(url)
    return urls


def _extract_antigravity_versions(text: str) -> List[str]:
    versions = []
    patterns = [
        r"antigravity-hub/(\d+\.\d+\.\d+)-",
        r"version:\s*[\"'](\d+\.\d+\.\d+)<br>",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text or "", re.IGNORECASE):
            versions.append(match.group(1))
    return versions


def _latest_antigravity_client_version() -> str:
    """Resolve the current official Antigravity app version for request headers."""
    override = os.environ.get("ANTIGRAVITY_CLIENT_VERSION", "").strip()
    if override:
        return override.lstrip("v")

    try:
        resp = requests.get(ANTIGRAVITY_SITE_URL, timeout=15)
        resp.raise_for_status()
        html = resp.text or ""
        candidates = _extract_antigravity_versions(html)

        for url in _script_urls_from_html(ANTIGRAVITY_SITE_URL, html)[:8]:
            try:
                script_resp = requests.get(url, timeout=15)
                script_resp.raise_for_status()
                candidates.extend(_extract_antigravity_versions(script_resp.text or ""))
            except Exception:
                continue

        if candidates:
            return max(candidates, key=_parse_semver)
    except Exception as exc:
        logger.debug("Could not resolve official Antigravity client version: %s", exc)

    return ANTIGRAVITY_CLIENT_VERSION_FALLBACK


def _write_proxy_runtime_package_json(data_dir: str, version: Optional[str] = None) -> None:
    package_json = os.path.join(data_dir, "package.json")
    version = version or _latest_proxy_version()
    try:
        existing: Dict[str, Any] = {}
        if os.path.exists(package_json):
            with open(package_json, "r", encoding="utf-8") as f:
                existing = json.load(f)

        desired = {
            **existing,
            "name": PROXY_PACKAGE_NAME,
            "private": True,
            "version": version,
        }

        if existing != desired:
            with open(package_json, "w", encoding="utf-8") as f:
                json.dump(desired, f, indent=2)
    except Exception as exc:
        logger.debug("Could not update Antigravity proxy package.json: %s", exc)


def _patch_runtime_package_json(runtime_dir: str, version: str) -> None:
    package_json = os.path.join(runtime_dir, "package.json")
    with open(package_json, "r", encoding="utf-8") as f:
        package_data = json.load(f)

    package_data["name"] = PROXY_PACKAGE_NAME
    package_data["version"] = version

    with open(package_json, "w", encoding="utf-8") as f:
        json.dump(package_data, f, indent=2)
        f.write("\n")


def _patch_runtime_antigravity_client_version(runtime_dir: str, version: str) -> bool:
    headers_path = os.path.join(runtime_dir, "src", "utils", "headers.ts")
    if not os.path.exists(headers_path):
        return False

    with open(headers_path, "r", encoding="utf-8") as f:
        content = f.read()

    updated, count = re.subn(
        r'const\s+ANTIGRAVITY_VERSION\s*=\s*["\'][^"\']+["\'];',
        f'const ANTIGRAVITY_VERSION = "{version}";',
        content,
        count=1,
    )

    if count:
        with open(headers_path, "w", encoding="utf-8") as f:
            f.write(updated)

    return bool(count)


def _patch_runtime_gemini35_flash_support(runtime_dir: str) -> bool:
    """Patch upstream transform support for Antigravity's Gemini 3.5 Flash tiers."""
    transform_path = os.path.join(runtime_dir, "src", "utils", "transform.ts")
    if not os.path.exists(transform_path):
        return False

    with open(transform_path, "r", encoding="utf-8") as f:
        content = f.read()

    updated = content
    changed = False

    updated, count = re.subn(
        r'\n\s*"gemini-3\.5-flash(?:-(?:high|medium|low))?",',
        "",
        updated,
    )
    changed = changed or count > 0

    def add_supported_models(match: re.Match) -> str:
        indent = match.group("indent")
        return (
            match.group(0)
            + f'{indent}"gemini-3.5-flash-high",\n'
            + f'{indent}"gemini-3.5-flash-medium",\n'
            + f'{indent}"gemini-3.5-flash-low",\n'
            + f'{indent}"gemini-3.5-flash",\n'
        )

    updated, count = re.subn(
        r'(?P<indent>\s*)"gemini-3\.1-pro-preview",\n',
        add_supported_models,
        updated,
        count=1,
    )
    changed = changed or count > 0

    mapping_marker = 'googleModel = `gemini-3.5-flash-${extractedTier || "medium"}`;'
    updated, count = re.subn(
        r'googleModel = "gemini-3\.5-flash-low";',
        mapping_marker,
        updated,
    )
    changed = changed or count > 0

    if mapping_marker not in updated:
        def add_flash_mapping(match: re.Match) -> str:
            if_indent = match.group("if_indent")
            body_indent = match.group("body_indent")
            original = match.group(0)
            return (
                f'{if_indent}if (baseModel.includes("gemini-3.5-flash")) {{\n'
                f'{body_indent}{mapping_marker}\n'
                f'{if_indent}}} else {original.lstrip()}'
            )

        updated, count = re.subn(
            r'(?P<if_indent>\s*)if \(baseModel\.includes\("gemini-3\.1-pro"\)\) \{\n'
            r'(?P<body_indent>\s*)googleModel = `gemini-3\.1-pro-\$\{extractedTier \|\| "high"\}`;',
            add_flash_mapping,
            updated,
            count=1,
        )
        changed = changed or count > 0

    if changed:
        with open(transform_path, "w", encoding="utf-8") as f:
            f.write(updated)

    return (
        '"gemini-3.5-flash-high"' in updated
        and '"gemini-3.5-flash-medium"' in updated
        and '"gemini-3.5-flash-low"' in updated
        and '"gemini-3.5-flash",' in updated
        and mapping_marker in updated
    )


def _runtime_has_entrypoint(runtime_dir: str) -> bool:
    return os.path.exists(os.path.join(runtime_dir, "src", "server.ts")) and os.path.exists(
        os.path.join(runtime_dir, "package.json")
    )


def _runtime_metadata_path(runtime_dir: str) -> str:
    return os.path.join(runtime_dir, ".glossarion-runtime.json")


def _read_runtime_metadata(runtime_dir: str) -> Dict[str, Any]:
    with open(_runtime_metadata_path(runtime_dir), "r", encoding="utf-8") as f:
        return json.load(f)


def _runtime_metadata_matches(runtime_dir: str, tag: str, client_version: str) -> bool:
    if not _runtime_has_entrypoint(runtime_dir):
        return False

    try:
        metadata = _read_runtime_metadata(runtime_dir)
        return (
            metadata.get("tag") == tag
            and metadata.get("client_version") == client_version
            and metadata.get("patch_version") == RUNTIME_PATCH_VERSION
        )
    except Exception:
        return False


def _write_runtime_metadata(runtime_dir: str, release: Dict[str, str], client_version: str) -> None:
    metadata = {
        "tag": release["tag"],
        "version": release["version"],
        "client_version": client_version,
        "patch_version": RUNTIME_PATCH_VERSION,
        "source": release["archive_url"],
        "updated_at": int(time.time()),
    }
    with open(_runtime_metadata_path(runtime_dir), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def _find_archive_root(extract_dir: str) -> str:
    for root, dirs, files in os.walk(extract_dir):
        if "package.json" in files and os.path.exists(os.path.join(root, "src", "server.ts")):
            return root
        dirs[:] = dirs[:3]
    raise RuntimeError("Downloaded proxy archive did not contain src/server.ts")


def _copy_runtime_tree(source_dir: str, target_dir: str) -> None:
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


def _download_proxy_runtime(
    release: Dict[str, str],
    client_version: str,
    runtime_dir: str,
    log_fn=None,
) -> None:
    _log = log_fn or _log_noop
    _log(f"Antigravity: downloading {PROXY_PACKAGE_NAME} {release['tag']}...")

    resp = requests.get(release["archive_url"], headers=_github_headers(), timeout=60)
    resp.raise_for_status()

    parent = os.path.dirname(runtime_dir)
    os.makedirs(parent, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="download-", dir=parent) as tmp_dir:
        extract_dir = os.path.join(tmp_dir, "extract")
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
            archive.extractall(extract_dir)

        archive_root = _find_archive_root(extract_dir)
        _patch_runtime_package_json(archive_root, release["version"])
        if not _patch_runtime_antigravity_client_version(archive_root, client_version):
            raise RuntimeError("Downloaded proxy archive did not contain ANTIGRAVITY_VERSION")
        if not _patch_runtime_gemini35_flash_support(archive_root):
            raise RuntimeError("Downloaded proxy archive could not be patched for Gemini 3.5 Flash")
        _write_runtime_metadata(archive_root, release, client_version)

        _copy_runtime_tree(archive_root, runtime_dir)


def _latest_existing_runtime(runtime_root: str) -> Optional[str]:
    if not os.path.isdir(runtime_root):
        return None

    candidates = []
    for name in os.listdir(runtime_root):
        path = os.path.join(runtime_root, name)
        if _runtime_has_entrypoint(path):
            candidates.append((os.path.getmtime(path), path))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _cached_runtime_needs_patch(data_dir: str) -> bool:
    runtime_dir = _latest_existing_runtime(_get_proxy_runtime_root(data_dir))
    if not runtime_dir:
        return False
    try:
        metadata = _read_runtime_metadata(runtime_dir)
    except Exception:
        return True
    return metadata.get("patch_version") != RUNTIME_PATCH_VERSION


def _patch_cached_runtime(
    runtime_dir: str,
    release: Dict[str, str],
    client_version: str,
    log_fn=None,
) -> bool:
    try:
        _patch_runtime_package_json(runtime_dir, release["version"])
        if not _patch_runtime_antigravity_client_version(runtime_dir, client_version):
            return False
        if not _patch_runtime_gemini35_flash_support(runtime_dir):
            return False
        _write_runtime_metadata(runtime_dir, release, client_version)
        (log_fn or _log_noop)("Antigravity: patched cached proxy runtime in place.")
        return True
    except Exception as exc:
        logger.debug("Could not patch cached Antigravity proxy runtime: %s", exc)
        return False


def _ensure_proxy_runtime(data_dir: str, log_fn=None, force_update: bool = False) -> str:
    release = _latest_proxy_release()
    client_version = _latest_antigravity_client_version()
    _write_proxy_runtime_package_json(data_dir, release["version"])

    runtime_root = _get_proxy_runtime_root(data_dir)
    runtime_name = (
        f"{_safe_runtime_segment(release['tag'])}-ag-{_safe_runtime_segment(client_version)}"
    )
    runtime_dir = os.path.join(runtime_root, runtime_name)

    if not force_update and _runtime_metadata_matches(runtime_dir, release["tag"], client_version):
        return runtime_dir

    try:
        _download_proxy_runtime(release, client_version, runtime_dir, log_fn=log_fn)
        return runtime_dir
    except Exception as exc:
        logger.debug("Could not update Antigravity proxy runtime: %s", exc)
        existing = _latest_existing_runtime(runtime_root)
        if existing:
            if _patch_cached_runtime(existing, release, client_version, log_fn=log_fn):
                return existing
            (log_fn or _log_noop)(
                f"Antigravity: using cached proxy runtime because update failed: {exc}"
            )
            return existing
        raise


def _ensure_proxy_config() -> str:
    """Ensure the auto-launched proxy has a stable working/data directory.

    frieser/antigravity-proxy stores config.json and reads package.json relative
    to process.cwd(). We launch from this dedicated directory while executing
    the downloaded proxy script by absolute path, so config/accounts stay stable
    across runtime updates.
    """
    data_dir = _get_proxy_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_proxy_url() -> str:
    """Get the Antigravity proxy URL from env or default."""
    return os.environ.get("ANTIGRAVITY_PROXY_URL", DEFAULT_PROXY_URL).rstrip("/")


def _get_proxy_port() -> int:
    try:
        parsed = urlparse(get_proxy_url())
        if parsed.port:
            return parsed.port
    except Exception:
        pass
    return 3000


def _build_headers() -> Dict[str, str]:
    """Build HTTP headers for the OpenAI-compatible proxy."""
    return {
        "Content-Type": "application/json",
        # The upstream server currently does not validate this, but including an
        # OpenAI-shaped Authorization header keeps generic middleware happy.
        "Authorization": "Bearer sk-antigravity",
    }


def invalidate_api_key_cache() -> None:
    """Compatibility no-op for callers from the older adapter."""
    return None


# ---------------------------------------------------------------------------
# Cancellation and auth-browser helpers
# ---------------------------------------------------------------------------

def cancel_stream() -> None:
    """Signal any active Antigravity proxy stream to abort."""
    _cancel_event.set()


def reset_cancel() -> None:
    """Clear the cancellation flag before a new request."""
    _cancel_event.clear()
    reset_auth_browser()


def reset_auth_browser() -> None:
    """Reset the auth browser flag so the browser can be re-opened."""
    global _auth_browser_opened
    with _auth_browser_lock:
        _auth_browser_opened = False


def is_cancelled() -> bool:
    return _cancel_event.is_set()


def _open_auth_browser_once(proxy_url: str, log_fn=None) -> bool:
    """Open the proxy OAuth URL in the browser, but only once per session."""
    global _auth_browser_opened
    with _auth_browser_lock:
        if _auth_browser_opened:
            return False
        _auth_browser_opened = True

    _log = log_fn or _log_noop
    auth_url = f"{proxy_url}{OAUTH_START_ENDPOINT}"
    _log(f"Antigravity: opening {auth_url} for Google account linking...")
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass
    return True


# ---------------------------------------------------------------------------
# Model and response conversion
# ---------------------------------------------------------------------------

def _normalize_model_name(model: str) -> str:
    """Convert Glossarion's model ids to frieser/antigravity-proxy ids.

    Glossarion strips the public "antigravity/" provider prefix before calling
    this module. This adapter always uses the proxy's explicit "antigravity-"
    namespace so requests stay on the upstream sandbox/agent route instead of
    accidentally falling into the CLI-pool Gemini route.
    """
    clean = (model or "").strip()
    lower = clean.lower()

    if lower.startswith("antigravity/"):
        clean = clean.split("/", 1)[1].lstrip("/")
        lower = clean.lower()

    if lower.startswith("antigravity-"):
        return clean

    # Claude, GPT-equivalent, Gemini, image, and thinking models should use the
    # explicit Antigravity model namespace from this adapter.
    if (
        lower.startswith(("claude-", "gpt-", "gemini-"))
        or "image" in lower
        or lower.startswith("anthropic/")
    ):
        return f"antigravity-{clean}"

    return clean


def _payload_for_openai_chat(
    messages: List[Dict],
    model: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> Dict[str, Any]:
    normalized_model = _normalize_model_name(model)
    return {
        "model": normalized_model,
        "messages": messages,
        "max_tokens": _clamp_max_tokens_for_model(normalized_model, max_tokens),
        "temperature": temperature,
        "stream": stream,
    }


def clamp_output_tokens_for_model(model: str, max_tokens: Any, default: int = 8192) -> int:
    try:
        requested = int(max_tokens)
    except Exception:
        requested = int(default)

    if requested <= 0:
        return requested

    model_lower = _normalize_model_name(model).lower()
    if "claude" in model_lower:
        return min(requested, CLAUDE_MAX_OUTPUT_TOKENS)
    if "gemini" in model_lower:
        return min(requested, GEMINI_MAX_OUTPUT_TOKENS)
    return requested


def _clamp_max_tokens_for_model(model: str, max_tokens: int) -> int:
    return clamp_output_tokens_for_model(model, max_tokens)


def _format_token_count(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def _token_count_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _log_payload_token_limit(log_fn, requested_max_tokens: Any, payload: Dict[str, Any]) -> None:
    if log_fn is None:
        return

    sent_max_tokens = payload.get("max_tokens")
    requested = _token_count_or_none(requested_max_tokens)
    sent = _token_count_or_none(sent_max_tokens)
    model = payload.get("model", "")

    if requested is not None and sent is not None and requested != sent:
        log_fn(
            "🎚️ Antigravity: max_tokens clamped "
            f"{_format_token_count(requested)} -> {_format_token_count(sent)} "
            f"(model={model})"
        )
        return

    log_fn(f"🎚️ Antigravity: max_tokens={_format_token_count(sent_max_tokens)} (model={model})")


def _parse_openai_chat_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an OpenAI Chat Completions response to Glossarion's shape."""
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}

    content = message.get("content") or ""
    finish_reason = choice.get("finish_reason") or "stop"
    usage = data.get("usage")

    return {
        "content": content,
        "finish_reason": finish_reason,
        "usage": usage,
        "raw_response": data,
    }


def _extract_error_message(resp: requests.Response) -> str:
    try:
        data = resp.json()
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
    except Exception:
        pass
    return (getattr(resp, "text", "") or "")[:1000]


def _health_details_have_accounts(details: Any) -> bool:
    if not isinstance(details, dict):
        return False
    accounts = details.get("accounts")
    return isinstance(accounts, list) and len(accounts) > 0


def _proxy_has_accounts() -> bool:
    health = check_proxy_health()
    return bool(health.get("healthy") and _health_details_have_accounts(health.get("details")))


def _should_wait_for_auth(resp: requests.Response) -> bool:
    """Return True when the request likely failed because no account is linked."""
    if resp.status_code in (401, 403):
        return not _proxy_has_accounts()

    if resp.status_code == 429:
        # Upstream returns a generic quota-exhausted 429 when there are no
        # accounts at all. Only treat that as auth setup when status confirms
        # an empty account list.
        return not _proxy_has_accounts()

    error_text = _extract_error_message(resp).lower()
    if "no account" in error_text or "add account" in error_text:
        return True

    return False


def _should_wait_for_auth_status_error(status_code: int, error_text: str = "") -> bool:
    if status_code in (401, 403):
        return not _proxy_has_accounts()

    if status_code == 429:
        return not _proxy_has_accounts()

    lowered = (error_text or "").lower()
    return "no account" in lowered or "add account" in lowered


class _HttpxStreamResponseAdapter:
    """Keep an httpx.stream() context alive while exposing a response-like API."""

    def __init__(self, context_manager: Any):
        self._context_manager = context_manager
        self._response = context_manager.__enter__()
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._response.close()
        finally:
            self._context_manager.__exit__(None, None, None)


def _wait_for_auth(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    proxy_url: str,
    log_fn=None,
    max_wait: int = 180,
    poll_interval: int = 5,
    stream: bool = False,
    request_timeout: float = 300,
    prefer_httpx_stream: bool = False,
):
    """Open OAuth and poll until an account is linked or timeout is reached."""
    _open_auth_browser_once(proxy_url, log_fn)
    _log = log_fn or _log_noop

    _log("")
    _log("Antigravity: Google account link required.")
    _log(f"Open {proxy_url}, add a Google account, then keep this window open.")
    _log(f"Waiting for account linking to complete... (timeout: {max_wait}s)")

    elapsed = 0
    while elapsed < max_wait:
        if _cancel_event.is_set():
            return None

        time.sleep(poll_interval)
        elapsed += poll_interval

        if _proxy_has_accounts():
            _log("Antigravity: account detected, retrying request...")
            retry_resp = None
            try:
                if stream and prefer_httpx_stream and httpx is not None:
                    retry_resp = _HttpxStreamResponseAdapter(
                        _stream_chat_with_httpx(url, payload, headers, request_timeout)
                    )
                else:
                    retry_resp = requests.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=request_timeout,
                        stream=stream,
                    )
                if retry_resp.status_code < 400:
                    return retry_resp
            except Exception:
                pass
            if retry_resp is not None:
                try:
                    retry_resp.close()
                except Exception:
                    pass

        _log(f"Still waiting for Antigravity account linking... ({elapsed}s / {max_wait}s)")

    return None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def _proxy_reports_unsupported(proxy_url: Optional[str] = None) -> bool:
    """Detect the upstream unsupported-version interstitial."""
    try:
        resp = requests.get(proxy_url or get_proxy_url(), timeout=5)
        text = (resp.text or "").lower()
        return "version of antigravity is no longer supported" in text
    except Exception:
        return False


def check_proxy_health() -> Dict[str, Any]:
    """Check if frieser/antigravity-proxy is running."""
    proxy_url = get_proxy_url()

    try:
        resp = requests.get(f"{proxy_url}{STATUS_ENDPOINT}", timeout=5)
        if resp.status_code == 200:
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            if _proxy_reports_unsupported(proxy_url):
                return {"healthy": False, "error": "Antigravity proxy version is no longer supported.", "details": data}
            return {"healthy": True, "details": data}
        status_error = f"HTTP {resp.status_code}"
    except requests.ConnectionError:
        return {"healthy": False, "error": "Connection refused - is antigravity-proxy running?"}
    except Exception as exc:
        status_error = str(exc)

    # Fallback for future-compatible OpenAI-only deployments.
    try:
        resp = requests.get(f"{proxy_url}{MODELS_ENDPOINT}", headers=_build_headers(), timeout=5)
        if resp.status_code == 200:
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            if _proxy_reports_unsupported(proxy_url):
                return {"healthy": False, "error": "Antigravity proxy version is no longer supported.", "details": data}
            return {"healthy": True, "details": data}
    except Exception:
        pass

    if _proxy_reports_unsupported(proxy_url):
        return {"healthy": False, "error": "Antigravity proxy version is no longer supported."}

    return {"healthy": False, "error": status_error}


# ---------------------------------------------------------------------------
# Auto-launch proxy
# ---------------------------------------------------------------------------

def _candidate_executable(name: str) -> Optional[str]:
    path = shutil.which(name)
    if path:
        return path

    if sys.platform == "win32":
        candidates: List[str] = []
        home = os.path.expanduser("~")
        candidates.extend(
            [
                os.path.join(home, ".bun", "bin", f"{name}.exe"),
                os.path.join(home, ".bun", "bin", f"{name}.cmd"),
                os.path.join(os.environ.get("APPDATA", ""), "npm", f"{name}.cmd"),
                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "bun", f"{name}.exe"),
            ]
        )
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate

    return None


def _find_proxy_launch_command(runtime_dir: str) -> Optional[List[str]]:
    """Return a launch command for the downloaded upstream proxy runtime."""
    override = os.environ.get("ANTIGRAVITY_PROXY_LAUNCH_CMD", "").strip()
    if override:
        override = override.format(
            runtime_dir=runtime_dir,
            server=os.path.join(runtime_dir, "src", "server.ts"),
        )
        try:
            return shlex.split(override, posix=(sys.platform != "win32"))
        except Exception:
            return override.split()

    server_script = os.path.join(runtime_dir, "src", "server.ts")

    bun = _candidate_executable("bun")
    if bun:
        return [bun, "run", server_script]

    npx = _candidate_executable("npx")
    if npx:
        return [npx, "--yes", "--package", BUN_NPM_PACKAGE, "bun", "run", server_script]

    return None


def _proxy_launch_error() -> str:
    return (
        "Could not find Bun or npx to launch the Antigravity proxy.\n"
        "Install Node.js/npm or Bun, then restart Glossarion.\n"
        "Git is not required; Glossarion downloads the proxy runtime itself."
    )


def ensure_proxy_running(log_fn=None) -> Dict[str, Any]:
    """Ensure the Antigravity proxy is running, auto-launching if needed."""
    global _proxy_process

    _log = log_fn or _log_noop
    data_dir = _ensure_proxy_config()
    force_update = False

    health = check_proxy_health()
    if health.get("healthy"):
        if _cached_runtime_needs_patch(data_dir):
            _log("Antigravity proxy runtime patch changed; restarting with updated local runtime...")
            _kill_proxy_by_port(_get_proxy_port())
            force_update = True
            time.sleep(2)
        else:
            return {"running": True, "auto_launched": False}

    with _proxy_launch_lock:
        health = check_proxy_health()
        if health.get("healthy"):
            if _cached_runtime_needs_patch(data_dir):
                _log("Antigravity proxy runtime patch changed; restarting with updated local runtime...")
                _kill_proxy_by_port(_get_proxy_port())
                force_update = True
                time.sleep(2)
            else:
                return {"running": True, "auto_launched": False}

        if "no longer supported" in str(health.get("error", "")).lower():
            _log("Antigravity proxy reports an unsupported version; updating the local proxy runtime...")
            _kill_proxy_by_port(_get_proxy_port())
            force_update = True
            time.sleep(2)

        if _proxy_process is not None:
            if _proxy_process.poll() is None:
                _log("Antigravity proxy process is running, waiting for it to become healthy...")
                for _ in range(15):
                    time.sleep(2)
                    health = check_proxy_health()
                    if health.get("healthy"):
                        return {"running": True, "auto_launched": True}
                return {
                    "running": False,
                    "auto_launched": True,
                    "error": "Proxy was launched but did not become healthy within 30s.",
                }
            _proxy_process = None

        try:
            runtime_dir = _ensure_proxy_runtime(data_dir, log_fn=_log, force_update=force_update)
        except Exception as exc:
            return {
                "running": False,
                "auto_launched": False,
                "error": f"Failed to update Antigravity proxy runtime: {exc}",
            }

        cmd = _find_proxy_launch_command(runtime_dir)
        if not cmd:
            return {"running": False, "auto_launched": False, "error": _proxy_launch_error()}

        _log(f"Auto-launching Antigravity proxy with: {' '.join(cmd)}")

        try:
            env = os.environ.copy()
            env.setdefault("BASE_URL", get_proxy_url())
            env.setdefault(
                "ACCOUNTS_FILE",
                os.path.join(data_dir, "antigravity-accounts.json"),
            )

            kwargs: Dict[str, Any] = {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "stdin": subprocess.DEVNULL,
                "cwd": data_dir,
                "env": env,
            }

            if sys.platform == "win32":
                create_new_process_group = 0x00000200
                detached_process = 0x00000008
                try:
                    from shutdown_utils import subprocess_no_window_kwargs

                    kwargs.update(
                        subprocess_no_window_kwargs(
                            creationflags=create_new_process_group | detached_process
                        )
                    )
                except Exception:
                    kwargs["creationflags"] = create_new_process_group | detached_process
            else:
                kwargs["start_new_session"] = True

            _proxy_process = subprocess.Popen(cmd, **kwargs)
            _log(f"Antigravity proxy process started (PID {_proxy_process.pid}).")
        except Exception as exc:
            return {
                "running": False,
                "auto_launched": False,
                "error": f"Failed to launch proxy: {exc}",
            }

        for _ in range(30):
            time.sleep(1)
            if _proxy_process.poll() is not None:
                _proxy_process = None
                return {
                    "running": False,
                    "auto_launched": True,
                    "error": (
                        "Proxy process exited immediately. "
                        "Check that Node.js/npm or Bun is installed."
                    ),
                }
            health = check_proxy_health()
            if health.get("healthy"):
                _log("Antigravity proxy is now running.")
                return {"running": True, "auto_launched": True}

        return {
            "running": False,
            "auto_launched": True,
            "error": "Proxy launched but did not become healthy within 30s. Check the proxy logs.",
        }


def _kill_proxy_by_port(port: int = 3000) -> None:
    """Kill any process listening on the proxy port."""
    try:
        if sys.platform == "win32":
            try:
                from shutdown_utils import run_no_window
            except Exception:
                run_no_window = subprocess.run

            result = run_no_window(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    pid = int(parts[-1])
                    run_no_window(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=5)
        else:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=5)
    except Exception:
        pass


def restart_proxy(log_fn=None) -> Dict[str, Any]:
    """Kill the running proxy and relaunch it."""
    global _proxy_process
    _log = log_fn or _log_noop
    _log("Antigravity: restarting proxy...")

    if _proxy_process is not None:
        try:
            _proxy_process.terminate()
            _proxy_process.wait(timeout=5)
        except Exception:
            try:
                _proxy_process.kill()
            except Exception:
                pass
        _proxy_process = None

    _kill_proxy_by_port(_get_proxy_port())
    time.sleep(2)
    return ensure_proxy_running(log_fn=log_fn)


# ---------------------------------------------------------------------------
# Send request helpers
# ---------------------------------------------------------------------------

def _post_chat(payload: Dict[str, Any], timeout: float, stream: bool) -> requests.Response:
    return requests.post(
        f"{get_proxy_url()}{CHAT_COMPLETIONS_ENDPOINT}",
        json=payload,
        headers=_build_headers(),
        timeout=timeout,
        stream=stream,
    )


def _stream_chat_with_httpx(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float,
):
    if httpx is None:
        raise ImportError("httpx is not installed")

    stream_headers = {
        **headers,
        "Accept": "text/event-stream",
        # Disable gzip so SSE events are yielded as the proxy flushes them.
        "Accept-Encoding": "identity",
    }
    connect_timeout = min(30.0, float(timeout)) if timeout else 30.0
    timeout_config = httpx.Timeout(timeout, connect=connect_timeout)
    return httpx.stream(
        "POST",
        url,
        json=payload,
        headers=stream_headers,
        timeout=timeout_config,
    )


def _raise_connection_refused() -> None:
    raise RuntimeError(
        "Antigravity proxy connection refused. "
        "Use Glossarion's Antigravity auto-launch/restart so it can update the proxy runtime, "
        f"Then open {get_proxy_url()} and add your Google account."
    )


def _raise_for_proxy_status(resp: requests.Response) -> None:
    error_msg = _extract_error_message(resp)
    raise RuntimeError(f"Antigravity: HTTP {resp.status_code} - {error_msg}")


def send_message(
    messages: List[Dict],
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.7,
    max_tokens: int = 8192,
    timeout: float = 300,
    log_fn=None,
) -> Dict[str, Any]:
    """Send a non-streaming message to frieser/antigravity-proxy."""
    proxy_url = get_proxy_url()
    url = f"{proxy_url}{CHAT_COMPLETIONS_ENDPOINT}"
    payload = _payload_for_openai_chat(messages, model, temperature, max_tokens, stream=False)

    _log = log_fn or _log_noop
    _log(f"🚀 Antigravity: sending to {proxy_url} (model={payload['model']})")
    _log_payload_token_limit(_log, max_tokens, payload)

    try:
        resp = _post_chat(payload, timeout=timeout, stream=False)
    except requests.ConnectionError:
        _raise_connection_refused()
    except requests.Timeout:
        raise RuntimeError(
            f"Antigravity proxy request timed out after {timeout}s. "
            "The model may need more time for long translations."
        )

    if _should_wait_for_auth(resp):
        auth_resp = _wait_for_auth(
            url,
            payload,
            _build_headers(),
            proxy_url,
            log_fn,
            stream=False,
            request_timeout=timeout,
        )
        if auth_resp is not None:
            resp = auth_resp
        else:
            raise RuntimeError(
                f"Antigravity: authentication timed out.\n"
                f"Open {proxy_url} and add your Google account, then try again."
            )

    if resp.status_code != 200:
        _raise_for_proxy_status(resp)

    try:
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Antigravity: invalid JSON response from proxy: {exc}")

    return _parse_openai_chat_response(data)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def _log_text_stream(text: str, log_buf: List[str], log_fn) -> None:
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


def _iter_stream_lines(resp: Any) -> Iterable[Any]:
    try:
        yield from resp.iter_lines(decode_unicode=True, chunk_size=1)
    except TypeError:
        yield from resp.iter_lines()


def _consume_openai_stream(resp: Any, log_fn=None, log_stream: bool = True) -> Dict[str, Any]:
    _log = log_fn or _log_noop
    collected_content: List[str] = []
    finish_reason = "stop"
    usage = None
    got_first_data = False
    t_start = time.time()
    log_buf: List[str] = []
    thinking_buf: List[str] = []
    thinking_started = False
    stream_thinking = os.getenv("STREAM_THINKING_LOGS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )

    try:
        try:
            resp.encoding = "utf-8"
        except Exception:
            pass

        for line in _iter_stream_lines(resp):
            if _cancel_event.is_set():
                resp.close()
                raise RuntimeError("Antigravity: stream cancelled by user")

            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue

            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break

            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if not got_first_data:
                got_first_data = True
                _log(f"Antigravity: first token in {time.time() - t_start:.1f}s, streaming...")

            if event.get("usage"):
                usage = event.get("usage")

            choices = event.get("choices") or []
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta") or {}

            reasoning = delta.get("reasoning_content") or ""
            if reasoning and log_stream and stream_thinking:
                if not thinking_started:
                    thinking_started = True
                    _log("[antigravity] Thinking...")
                _log_text_stream(reasoning, thinking_buf, _log)

            text = delta.get("content") or ""
            if text:
                collected_content.append(text)
                if log_stream:
                    _log_text_stream(text, log_buf, _log)

            stop = choice.get("finish_reason")
            if stop:
                finish_reason = stop

        if log_stream and thinking_buf:
            remainder = "".join(thinking_buf).strip()
            if remainder:
                _log(f"    {remainder}")

        if log_stream and log_buf:
            remainder = "".join(log_buf).strip()
            if remainder:
                _log(remainder)

    finally:
        try:
            resp.close()
        except Exception:
            pass

    _log(f"Antigravity: stream finished in {time.time() - t_start:.1f}s")
    return {
        "content": "".join(collected_content),
        "finish_reason": finish_reason,
        "usage": usage,
        "raw_response": None,
    }


def send_message_stream(
    messages: List[Dict],
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.7,
    max_tokens: int = 8192,
    timeout: float = 300,
    log_fn=None,
    log_stream: bool = True,
) -> Dict[str, Any]:
    """Send a streaming message to frieser/antigravity-proxy."""
    proxy_url = get_proxy_url()
    url = f"{proxy_url}{CHAT_COMPLETIONS_ENDPOINT}"
    payload = _payload_for_openai_chat(messages, model, temperature, max_tokens, stream=True)

    _log = log_fn or _log_noop
    _log(f"🌊 Antigravity: streaming from {proxy_url} (model={payload['model']})")
    _log_payload_token_limit(_log, max_tokens, payload)

    try:
        with _stream_chat_with_httpx(url, payload, _build_headers(), timeout) as resp:
            if resp.status_code != 200:
                try:
                    error_text = resp.read().decode("utf-8", errors="replace")[:1000]
                except Exception:
                    error_text = ""

                if _should_wait_for_auth_status_error(resp.status_code, error_text):
                    auth_resp = _wait_for_auth(
                        url,
                        payload,
                        _build_headers(),
                        proxy_url,
                        log_fn,
                        stream=True,
                        request_timeout=timeout,
                        prefer_httpx_stream=True,
                    )
                    if auth_resp is not None:
                        return _consume_openai_stream(auth_resp, log_fn=log_fn, log_stream=log_stream)
                    raise RuntimeError(
                        f"Antigravity: authentication timed out.\n"
                        f"Open {proxy_url} and add your Google account, then try again."
                    )

                raise RuntimeError(f"Antigravity: HTTP {resp.status_code} - {error_text}")

            return _consume_openai_stream(resp, log_fn=log_fn, log_stream=log_stream)

    except ImportError:
        _log("Antigravity: httpx is not installed, falling back to requests streaming.")
    except Exception as exc:
        if httpx is not None and isinstance(exc, httpx.ConnectError):
            _raise_connection_refused()
        if httpx is not None and isinstance(exc, httpx.TimeoutException):
            raise RuntimeError(f"Antigravity proxy streaming request timed out after {timeout}s.")
        raise

    try:
        resp = _post_chat(payload, timeout=timeout, stream=True)
    except requests.ConnectionError:
        _raise_connection_refused()
    except requests.Timeout:
        raise RuntimeError(f"Antigravity proxy streaming request timed out after {timeout}s.")

    if _should_wait_for_auth(resp):
        try:
            resp.close()
        except Exception:
            pass
        auth_resp = _wait_for_auth(
            url,
            payload,
            _build_headers(),
            proxy_url,
            log_fn,
            stream=True,
            request_timeout=timeout,
        )
        if auth_resp is not None:
            resp = auth_resp
        else:
            raise RuntimeError(
                f"Antigravity: authentication timed out.\n"
                f"Open {proxy_url} and add your Google account, then try again."
            )

    if resp.status_code != 200:
        try:
            error_text = resp.text[:1000]
        except Exception:
            error_text = ""
        try:
            resp.close()
        except Exception:
            pass
        raise RuntimeError(f"Antigravity: HTTP {resp.status_code} - {error_text}")

    return _consume_openai_stream(resp, log_fn=log_fn, log_stream=log_stream)
