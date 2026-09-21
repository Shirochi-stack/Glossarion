"""
opera_aria.py - Opera "Ask AI" (Aria) browser-backed route.

Companion to gemini_free.py / authnd_auth.py. Instead of extracting Opera's
obfuscated app secret, this drives a locally-installed Opera browser briefly over
CDP and lets Opera itself mint an accountless Bearer JWT via its privileged
extension API:

    opr.operaIdentityPrivate.getAccessToken(false, 'user:read shodan:aria', true)

Flow (mirrors authnd_auth.py's launch→harvest→kill pattern):
1. Launch Opera in the background with --remote-debugging-port (~2-3s).
2. Connect to the Aria extension service worker via CDP and evaluate getAccessToken.
3. Kill Opera immediately; cache the JWT to disk (valid ~1 hour).
4. Send all chat/translation requests directly over HTTP/SSE to
   composer.opera-api.com with that Bearer token. Zero continuous browser cost.

Requires Opera (or Opera GX) installed. No account, no manual token, no API key.
Public entry point mirrors gemini_free.send_chat_completion(...):
returns {"content", "finish_reason", "usage", "raw_response"}.
"""

from __future__ import annotations

import base64
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import requests

try:
    from shutdown_utils import popen_no_window, run_no_window, terminate_subprocess_tree
except Exception:  # pragma: no cover - fallback if helper missing
    def popen_no_window(args, **kwargs):
        return subprocess.Popen(args, **kwargs)

    def run_no_window(args, **kwargs):
        return subprocess.run(args, **kwargs)

    def terminate_subprocess_tree(proc, *, kill=False, timeout=3.0):
        try:
            proc.kill() if kill else proc.terminate()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Endpoints / constants
# ---------------------------------------------------------------------------
COMPOSER_BASE_URL = "https://composer.opera-api.com"
CHAT_ENDPOINT_V2 = f"{COMPOSER_BASE_URL}/api/v2/a-chat"

ARIA_EXTENSION_ID = "jifbgnmbgbdiedhdecealmlgmekpagde"
ACCESS_TOKEN_EXPRESSION = (
    "opr.operaIdentityPrivate.getAccessToken(false, 'user:read shodan:aria', true)"
)

DEFAULT_MODEL = "opera"          # search/opera -> "opera"
DEFAULT_TIMEOUT = 120
TOKEN_MARGIN_SECONDS = 120       # re-mint when <2 min remaining
CDP_TARGET_WAIT = 25.0           # seconds to wait for the Aria service worker
GENERATION_FAILURE_MARKERS = (
    "something went wrong and the content wasn't generated",
    "content wasn't generated",
    "content was not generated",
    "tos_violation",
    "token_limit_exceeded",
)


# ---------------------------------------------------------------------------
# Cancellation (parity with gemini_free.cancel_stream / reset_cancel)
# ---------------------------------------------------------------------------
_CANCEL = threading.Event()


def cancel_stream() -> None:
    _CANCEL.set()


def reset_cancel() -> None:
    _CANCEL.clear()


def _is_cancelled() -> bool:
    return _CANCEL.is_set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _log(log_fn: Optional[Callable[[str], None]], message: str) -> None:
    if log_fn is not None:
        try:
            log_fn(message)
            return
        except Exception:
            pass
    print(message)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _think_harder_enabled() -> bool:
    return _env_bool("OPERA_ARIA_THINK_HARDER", False)


def _stream_logging_enabled() -> bool:
    """Real-time log streaming, mirroring the other browser-backed routes."""
    val = os.getenv("OPERA_ARIA_STREAM")
    if val is None or val.strip() == "":
        val = os.getenv("LOG_STREAM_CHUNKS", "1")
    return str(val).strip().lower() in ("1", "true", "yes", "on")


class OperaAriaError(RuntimeError):
    def __init__(self, message: str, *, error_type: str = "api_error"):
        super().__init__(message)
        self.error_type = error_type


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "\n".join(p for p in parts if p)
    if content is None:
        return ""
    return str(content)


def _messages_to_query(messages: Iterable[Dict[str, Any]]) -> str:
    """Flatten chat messages into a single Aria query (system prepended)."""
    msgs = [m for m in messages if isinstance(m, dict)]
    if not msgs:
        return ""
    if len(msgs) == 1:
        return _content_to_text(msgs[0].get("content"))
    system_txt = "\n\n".join(
        _content_to_text(m.get("content"))
        for m in msgs if str(m.get("role", "")).lower() == "system"
    ).strip()
    non_system = [m for m in msgs if str(m.get("role", "")).lower() != "system"]
    turns: List[str] = []
    for m in non_system:
        role = str(m.get("role", "user")).lower()
        text = _content_to_text(m.get("content")).strip()
        if not text:
            continue
        turns.append(f"Assistant: {text}" if role == "assistant"
                     else (text if len(non_system) == 1 else f"User: {text}"))
    body = "\n\n".join(turns)
    return f"{system_txt}\n\n{body}".strip() if system_txt else body.strip()


def _contains_generation_failure(text: str) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in GENERATION_FAILURE_MARKERS)


# ---------------------------------------------------------------------------
# Opera binary discovery (cross-platform)
# ---------------------------------------------------------------------------
def _opera_candidates() -> List[str]:
    override = _env("OPERA_ARIA_BINARY")
    cands: List[str] = [override] if override else []

    if sys.platform.startswith("win"):
        roots = [os.environ.get("LOCALAPPDATA", ""), os.environ.get("PROGRAMFILES", ""),
                 os.environ.get("PROGRAMFILES(X86)", "")]
        names = ["Opera", "Opera GX", "Opera Developer", "Opera Beta"]
        for root in roots:
            if not root:
                continue
            for name in names:
                cands.append(os.path.join(root, "Programs", name, "opera.exe"))
                cands.append(os.path.join(root, name, "opera.exe"))
                # versioned subfolders: ...\Opera\<version>\opera.exe
                cands.extend(glob.glob(os.path.join(root, "Programs", name, "*", "opera.exe")))
        for exe in ("opera", "opera.exe"):
            found = shutil.which(exe)
            if found:
                cands.append(found)
    elif sys.platform == "darwin":
        for app in ("Opera", "Opera GX", "Opera Developer", "Opera Beta"):
            cands.append(f"/Applications/{app}.app/Contents/MacOS/Opera")
            cands.append(os.path.expanduser(f"~/Applications/{app}.app/Contents/MacOS/Opera"))
        cands.append(shutil.which("opera") or "")
    else:  # linux / other posix
        for exe in ("opera", "opera-stable", "opera-gx", "opera-beta", "opera-developer"):
            found = shutil.which(exe)
            if found:
                cands.append(found)
        cands += ["/usr/bin/opera", "/snap/bin/opera",
                  "/usr/lib/x86_64-linux-gnu/opera/opera", "/opt/opera/opera"]
    # dedupe, keep order, only existing
    seen, out = set(), []
    for c in cands:
        if c and c not in seen and os.path.exists(c):
            seen.add(c)
            out.append(c)
    return out


def _find_opera_binary() -> Optional[str]:
    cands = _opera_candidates()
    return cands[0] if cands else None


# ---------------------------------------------------------------------------
# Automatic Opera install (official archive, per-user, no admin)
# ---------------------------------------------------------------------------
OPERA_ARCHIVE = "https://get.geo.opera.com/pub/opera/desktop/"
OPERA_FALLBACK_VERSION = "136.0.6008.22"

_install_lock = threading.Lock()
_install_attempted = False
_MINT_LOCK = threading.Lock()  # serialize token minting across parallel batch threads


def _opera_arch() -> str:
    m = (platform.machine() or "").lower()
    if m in ("arm64", "aarch64"):
        return "arm64"
    return "x64"


def _latest_opera_version(log_fn=None) -> str:
    pinned = _env("OPERA_ARIA_INSTALL_VERSION")
    if pinned:
        return pinned
    try:
        with urllib.request.urlopen(OPERA_ARCHIVE, timeout=15) as r:
            html = r.read().decode("utf-8", "replace")
        vers = re.findall(r'href="(\d+\.\d+\.\d+\.\d+)/"', html)
        if vers:
            return max(vers, key=lambda v: tuple(int(x) for x in v.split(".")))
    except Exception:
        pass
    return OPERA_FALLBACK_VERSION


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_verified(url: str, dest: str, log_fn=None) -> str:
    _log(log_fn, f"⬇️ Opera Aria: downloading {url.rsplit('/', 1)[-1]}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length", 0))
    done = last = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(1 << 20):
            if _is_cancelled():
                raise OperaAriaError("Opera Aria: install cancelled", error_type="cancelled")
            if chunk:
                f.write(chunk)
                done += len(chunk)
                if total and done - last >= (24 << 20):
                    last = done
                    _log(log_fn, f"   … {done >> 20}/{total >> 20} MB")
    try:
        expected = requests.get(url + ".sha256sum", timeout=20).text.split()[0].strip().lower()
        if expected and _sha256_file(dest) != expected:
            raise OperaAriaError("Opera Aria: installer checksum mismatch — aborting",
                                 error_type="api_error")
    except OperaAriaError:
        raise
    except Exception:
        pass  # checksum unavailable — proceed best-effort
    return dest


def _auto_install_opera(log_fn=None) -> Optional[str]:
    """Download Opera's official installer and install per-user (no admin). Returns the binary path."""
    if _env("OPERA_ARIA_AUTO_INSTALL", "1") == "0":
        return None
    ver = _latest_opera_version(log_fn)
    arch = _opera_arch()
    tmp = tempfile.mkdtemp(prefix="glossarion_opera_dl_")
    try:
        if sys.platform.startswith("win"):
            fn = f"Opera_{ver}_Setup_{'arm64' if arch == 'arm64' else 'x64'}.exe"
            exe = _download_verified(f"{OPERA_ARCHIVE}{ver}/win/{fn}", os.path.join(tmp, fn), log_fn)
            args = [exe, "/silent", "/allusers=0", "/setdefaultbrowser=0", "/launchbrowser=0",
                    "/desktopshortcut=0", "/startmenushortcut=0", "/pintotaskbar=0",
                    "/import-browser-data=0", "/enable-crash-reporting=0"]
            install_dir = _env("OPERA_ARIA_INSTALL_DIR")
            if install_dir:
                args.append(f"/installfolder={install_dir}")
            _log(log_fn, "🛠️ Opera Aria: installing Opera silently (per-user, no admin)…")
            run_no_window(args, timeout=600)
        elif sys.platform == "darwin":
            fn = f"Opera_{ver}_Setup.dmg"
            dmg = _download_verified(f"{OPERA_ARCHIVE}{ver}/mac/{fn}", os.path.join(tmp, fn), log_fn)
            mount = os.path.join(tmp, "mnt")
            os.makedirs(mount, exist_ok=True)
            _log(log_fn, "🛠️ Opera Aria: installing Opera to ~/Applications…")
            subprocess.run(["hdiutil", "attach", "-nobrowse", "-quiet", "-mountpoint", mount, dmg],
                           check=True, timeout=180)
            try:
                app_src = next((os.path.join(mount, x) for x in os.listdir(mount)
                                if x.endswith(".app")), None)
                if app_src:
                    dest_dir = os.path.expanduser("~/Applications")
                    os.makedirs(dest_dir, exist_ok=True)
                    dest_app = os.path.join(dest_dir, os.path.basename(app_src))
                    if os.path.exists(dest_app):
                        shutil.rmtree(dest_app, ignore_errors=True)
                    shutil.copytree(app_src, dest_app, symlinks=True)
            finally:
                subprocess.run(["hdiutil", "detach", "-quiet", mount], timeout=60)
        else:  # linux — packaged installers need root
            raise OperaAriaError(
                "Opera Aria: automatic install on Linux requires root. Install Opera via your "
                "package manager (the .deb/.rpm at get.geo.opera.com/pub/opera/desktop) or set "
                "OPERA_ARIA_BINARY to an Opera executable.",
                error_type="config_error")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return _find_opera_binary()


def _free_port() -> int:
    port = _env("OPERA_ARIA_CDP_PORT")
    if port.isdigit():
        return int(port)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _profile_dir() -> str:
    override = _env("OPERA_ARIA_PROFILE")
    if override:
        return override
    import tempfile
    return os.path.join(tempfile.gettempdir(), "glossarion_opera_aria_profile")


# ---------------------------------------------------------------------------
# Token cache (JWT with embedded exp)
# ---------------------------------------------------------------------------
def _decode_jwt_payload(token: str) -> Dict[str, Any]:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        b64 = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
        return json.loads(base64.urlsafe_b64decode(b64.encode()).decode())
    except Exception:
        return {}


def _token_cache_path() -> Path:
    override = _env("OPERA_ARIA_TOKEN_FILE")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "opera_aria_token.json"


def _load_cached_token() -> Optional[str]:
    try:
        p = _token_cache_path()
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            token = data.get("token")
            expires_at = float(data.get("expires_at", 0))
            if token and time.time() < (expires_at - TOKEN_MARGIN_SECONDS):
                return token
    except Exception:
        pass
    return None


def _save_cached_token(token: str) -> None:
    payload = _decode_jwt_payload(token)
    exp = payload.get("exp", int(time.time()) + 3600)
    try:
        _token_cache_path().write_text(json.dumps({
            "token": token, "iat": payload.get("iat", int(time.time())),
            "expires_at": exp, "cached_at": time.time()}), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CDP token minting
# ---------------------------------------------------------------------------
def _cdp_find_aria_worker(port: int, deadline: float, log_fn=None) -> Optional[Dict[str, Any]]:
    ext_id = _env("OPERA_ARIA_EXTENSION_ID") or ARIA_EXTENSION_ID
    while time.time() < deadline:
        if _is_cancelled():
            raise OperaAriaError("Opera Aria: cancelled", error_type="cancelled")
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/json")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                targets = json.loads(resp.read().decode())
            worker = next(
                (t for t in targets
                 if ext_id in str(t.get("url", ""))
                 and t.get("type") in ("service_worker", "background_page", "worker")
                 and t.get("webSocketDebuggerUrl")),
                None,
            )
            if worker:
                return worker
        except Exception:
            pass
        time.sleep(0.3)
    return None


def _cdp_eval_token(ws_url: str, log_fn=None) -> str:
    import websocket  # websocket-client (bundled)
    ws = websocket.create_connection(ws_url, timeout=15, suppress_origin=True)
    try:
        ws.send(json.dumps({"id": 1, "method": "Runtime.enable"}))
        ws.send(json.dumps({
            "id": 2, "method": "Runtime.evaluate",
            "params": {"expression": ACCESS_TOKEN_EXPRESSION,
                       "awaitPromise": True, "returnByValue": True},
        }))
        deadline = time.time() + 20
        while time.time() < deadline:
            if _is_cancelled():
                raise OperaAriaError("Opera Aria: cancelled", error_type="cancelled")
            raw = ws.recv()
            if not raw:
                continue
            msg = json.loads(raw)
            if msg.get("id") != 2:
                continue
            if "error" in msg:
                raise OperaAriaError(
                    f"Opera Aria: getAccessToken failed: {msg['error']}",
                    error_type="auth_error")
            result = msg.get("result", {}).get("result", {}).get("value")
            if isinstance(result, dict) and result.get("token"):
                return result["token"]
            raise OperaAriaError(
                f"Opera Aria: unexpected getAccessToken return: {result}",
                error_type="auth_error")
        raise OperaAriaError("Opera Aria: timed out awaiting token", error_type="auth_error")
    finally:
        try:
            ws.close()
        except Exception:
            pass


def mint_opera_token(log_fn=None) -> str:
    """Launch Opera briefly, mint a fresh accountless JWT via CDP, kill Opera."""
    direct = _env("OPERA_ARIA_ACCESS_TOKEN")
    if direct:
        _save_cached_token(direct)
        return direct

    opera = _find_opera_binary()
    if not opera:
        global _install_attempted
        with _install_lock:
            if not _install_attempted and _env("OPERA_ARIA_AUTO_INSTALL", "1") != "0":
                _install_attempted = True
                _log(log_fn, "📦 Opera Aria: Opera not found — auto-installing from Opera's official server")
                try:
                    opera = _auto_install_opera(log_fn)
                except OperaAriaError:
                    raise
                except Exception as exc:
                    raise OperaAriaError(
                        f"Opera Aria: automatic Opera install failed: {exc}. Install Opera "
                        f"manually or set OPERA_ARIA_BINARY.", error_type="config_error")
            else:
                opera = _find_opera_binary()
    if not opera:
        raise OperaAriaError(
            "Opera Aria: Opera is not installed (or not found). Install Opera or Opera GX, set "
            "OPERA_ARIA_BINARY, or enable auto-install (OPERA_ARIA_AUTO_INSTALL=1). The route "
            "mints its token by briefly launching Opera's built-in Ask AI — no account or API key.",
            error_type="config_error")

    # Prefer headless (no visible window, far lighter on CPU/GPU). Fall back to a
    # hidden off-screen window only if the Aria service worker doesn't appear headless.
    headless_pref = _env("OPERA_ARIA_HEADLESS", "1") != "0"
    modes = [True, False] if headless_pref else [False]
    last_err: Optional[Exception] = None
    for i, headless in enumerate(modes):
        try:
            token = _launch_and_harvest(opera, headless, log_fn)
            if token:
                _log(log_fn, "🔑 Opera Aria: token minted and cached (~1h)")
                return token
        except OperaAriaError as exc:
            if getattr(exc, "error_type", "") == "cancelled":
                raise
            last_err = exc
        if i < len(modes) - 1:
            _log(log_fn, "↩️ Opera Aria: Aria worker not found headless; retrying hidden off-screen")
    if last_err:
        raise last_err
    raise OperaAriaError(
        "Opera Aria: could not locate the Aria service worker over CDP. Ensure this Opera "
        "build includes Ask AI/Aria, or set OPERA_ARIA_EXTENSION_ID.",
        error_type="auth_error")


def _launch_and_harvest(opera: str, headless: bool, log_fn=None) -> Optional[str]:
    """Launch Opera (hidden), evaluate getAccessToken over CDP, kill Opera. None if no worker."""
    port = _free_port()
    profile = _profile_dir()
    try:
        os.makedirs(profile, exist_ok=True)
    except Exception:
        pass

    args = [
        opera,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile}",
        "--remote-allow-origins=*",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-dev-shm-usage",
        "--disable-gpu",
    ]
    if headless:
        args.append("--headless=new")
    else:
        # visible-mode fallback: shove the window off-screen and make it tiny
        args += ["--window-position=-32000,-32000", "--window-size=1,1"]
    env = os.environ.copy()
    if not sys.platform.startswith("win") and sys.platform != "darwin":
        args.insert(1, "--no-sandbox")
        env.setdefault("DISPLAY", ":1")

    _log(log_fn, f"🎭 Opera Aria: minting accountless token via {os.path.basename(opera)} "
                 f"(CDP:{port}{', headless' if headless else ''})")
    proc = popen_no_window(args, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + float(_env("OPERA_ARIA_CDP_WAIT") or CDP_TARGET_WAIT)
        worker = _cdp_find_aria_worker(port, deadline, log_fn)
        if not worker:
            return None
        token = _cdp_eval_token(worker["webSocketDebuggerUrl"], log_fn)
        _save_cached_token(token)
        return token
    finally:
        terminate_subprocess_tree(proc, kill=True)


def get_token(force_refresh: bool = False, log_fn=None) -> str:
    if not force_refresh:
        cached = _load_cached_token()
        if cached:
            return cached
    # Serialize minting so parallel batch threads don't each launch a browser
    # (the thundering herd that spikes CPU and stutters the GUI).
    with _MINT_LOCK:
        if not force_refresh:
            cached = _load_cached_token()  # another thread may have just minted
            if cached:
                return cached
        return mint_opera_token(log_fn)


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------
def _stream_thinking_enabled() -> bool:
    return str(os.getenv("STREAM_THINKING_LOGS", "0")).strip().lower() not in (
        "", "0", "false", "no", "off")


def _debug_sse() -> bool:
    return _env_bool("OPERA_ARIA_DEBUG_SSE", False)


def _classify(data: Dict[str, Any], event: Optional[str]) -> tuple:
    """Return (kind, text): kind is 'text', 'thinking', or None.

    Opera's think_harder reasoning may arrive as an SSE `event: thinking_status`
    frame, a response with content_type 'thinking'/'reasoning', or a top-level
    thinking/reasoning field. We separate it so it never pollutes the answer.
    """
    ev = (event or "").lower()
    thinking_ev = any(k in ev for k in ("thinking", "reason", "thought"))
    resp = data.get("response") if isinstance(data, dict) else None
    if isinstance(resp, dict):
        ct = str(resp.get("content_type") or "").lower()
        if ct == "image":
            return (None, None)
        msg = resp.get("message") if isinstance(resp.get("message"), str) else None
        think = resp.get("thinking") or resp.get("reasoning") or resp.get("thought")
        if ct in ("thinking", "reasoning", "thought") or thinking_ev:
            return ("thinking", msg or (think if isinstance(think, str) else None))
        if isinstance(think, str) and think:
            return ("thinking", think)
        if msg is not None:
            return ("text", msg)
    top_think = data.get("thinking") or data.get("reasoning") if isinstance(data, dict) else None
    if isinstance(top_think, str) and top_think:
        return ("thinking", top_think)
    if thinking_ev:
        return ("thinking", None)  # status frame with no visible text
    return (None, None)


def _accumulate(acc: str, piece: str) -> str:
    if not acc:
        return piece
    if piece.startswith(acc):
        return piece
    if acc.endswith(piece):
        return acc
    return acc + piece


def _post_chat(token: str, query: str, timeout: int, stream: bool = True):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
        "Origin": COMPOSER_BASE_URL,
        "Referer": f"{COMPOSER_BASE_URL}/assets/aria/index.html",
        "X-Opera-Timezone": "UTC",
        "X-Opera-UI-Language": "en_US",
    }
    payload = {"query": query, "stream": stream, "request_source": "side_panel"}
    if _think_harder_enabled():
        payload["think_harder"] = True
    return requests.post(CHAT_ENDPOINT_V2, headers=headers, json=payload,
                         stream=stream, timeout=timeout)


def _run_chat(messages: Iterable[Dict[str, Any]], *, model: str, timeout: int,
              log_fn=None, log_stream: Optional[bool] = None) -> Dict[str, Any]:
    query = _messages_to_query(messages)
    if not query.strip():
        raise OperaAriaError("Opera Aria: empty query", error_type="config_error")

    token = get_token(log_fn=log_fn)
    think_on = _think_harder_enabled()
    _log(log_fn, f"🎭 Opera Aria: sending request ({len(query):,} chars, model={model}"
                 f"{', think harder' if think_on else ''})")
    if think_on:
        _log(log_fn, "🧠 Opera Aria: thinking mode enabled (think harder)")
    resp = _post_chat(token, query, timeout)

    if resp.status_code in (401, 403):
        _log(log_fn, "⚠️ Opera Aria: token rejected; re-minting and retrying")
        token = get_token(force_refresh=True, log_fn=log_fn)
        resp = _post_chat(token, query, timeout)

    if resp.status_code != 200:
        raise OperaAriaError(
            f"Opera Aria chat failed ({resp.status_code}): {resp.text[:200]}",
            error_type="auth_error" if resp.status_code in (401, 403) else "api_error")

    acc = ""
    conversation_id = None
    stream_log = _stream_logging_enabled() if log_stream is None else bool(log_stream)
    stream_thinking = _stream_thinking_enabled()
    debug_sse = _debug_sse()
    first_token = False
    thinking_started = False
    emit_buf: List[str] = []
    think_buf: List[str] = []
    current_event: Optional[str] = None

    def _flush_lines(buf: List[str], prefix: str = "") -> None:
        combined = "".join(buf)
        for tag in ("</p>", "</h1>", "</h2>", "</h3>", "</li>"):
            combined = combined.replace(tag, tag + "\n")
        if "\n" in combined:
            lines = combined.split("\n")
            for ln in lines[:-1]:
                if ln.strip():
                    _log(log_fn, f"{prefix}{ln}")
            buf[:] = [lines[-1]]
        elif len(combined) >= 160:
            _log(log_fn, f"{prefix}{combined}")
            buf.clear()

    def _emit_live(fragment: str) -> None:
        if not fragment:
            return
        emit_buf.append(fragment)
        _flush_lines(emit_buf)

    def _emit_thinking(fragment: Optional[str]) -> None:
        nonlocal thinking_started
        if not (stream_log and stream_thinking):
            return
        if not thinking_started:
            thinking_started = True
            _log(log_fn, "🧠 [opera] Thinking...")
        if fragment:
            think_buf.append(fragment)
            _flush_lines(think_buf, prefix="    ")

    for raw in resp.iter_lines(decode_unicode=True):
        if _is_cancelled():
            try:
                resp.close()
            except Exception:
                pass
            raise OperaAriaError("Opera Aria: stream cancelled", error_type="cancelled")
        if not raw:
            continue
        line = raw.strip()
        if line.startswith("event:"):
            current_event = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body in ("[DONE]", "null", ""):
            current_event = None
            continue
        if debug_sse:
            _log(log_fn, f"[opera-sse]{(' ' + current_event) if current_event else ''} {body[:400]}")
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            current_event = None
            continue
        kind, text = _classify(data, current_event)
        current_event = None
        if kind == "thinking":
            _emit_thinking(text)
        elif kind == "text" and text:
            prev_len = len(acc)
            acc = _accumulate(acc, text)
            if stream_log:
                if thinking_started and think_buf:
                    _log(log_fn, f"    {''.join(think_buf)}")
                    think_buf.clear()
                if not first_token:
                    first_token = True
                    _log(log_fn, "📡 Opera Aria: streaming response...")
                _emit_live(acc[prev_len:])
        meta = data.get("metadata")
        if isinstance(meta, dict) and meta.get("conversation_id"):
            conversation_id = meta["conversation_id"]

    if stream_log and stream_thinking and think_buf and "".join(think_buf).strip():
        _log(log_fn, f"    {''.join(think_buf)}")
    if stream_log and emit_buf and "".join(emit_buf).strip():
        _log(log_fn, "".join(emit_buf))

    if not acc.strip():
        raise OperaAriaError("Opera Aria: empty response from server", error_type="api_error")
    if _contains_generation_failure(acc):
        raise OperaAriaError(f"Opera Aria generation failed: {acc[:160]}", error_type="api_error")

    return {
        "content": acc,
        "finish_reason": "stop",
        "usage": None,
        "raw_response": {"conversation_id": conversation_id, "model": model},
    }


# ---------------------------------------------------------------------------
# Public entry point (mirrors gemini_free.send_chat_completion)
# ---------------------------------------------------------------------------
def send_chat_completion(
    *,
    messages: Iterable[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    log_stream: Optional[bool] = None,
    **_: Any,
) -> Dict[str, Any]:
    del temperature, max_tokens
    timeout_value = int(timeout or int(os.getenv("OPERA_ARIA_TIMEOUT", str(DEFAULT_TIMEOUT))))
    return _run_chat(list(messages), model=model or DEFAULT_MODEL,
                     timeout=timeout_value, log_fn=log_fn, log_stream=log_stream)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Opera Aria route probe")
    ap.add_argument("prompt", nargs="?", default="Reply with: PONG")
    ap.add_argument("--refresh", action="store_true", help="force re-mint the token")
    ap.add_argument("--which", action="store_true", help="print detected Opera binary and exit")
    ap.add_argument("--resolve", action="store_true", help="print the installer URL it would use (no download)")
    args = ap.parse_args()
    reset_cancel()
    if args.which:
        print("candidates:", _opera_candidates())
        print("selected:", _find_opera_binary())
        sys.exit(0)
    if args.resolve:
        ver = _latest_opera_version()
        arch = _opera_arch()
        print("platform:", sys.platform, "arch:", arch, "version:", ver)
        if sys.platform.startswith("win"):
            fn = f"Opera_{ver}_Setup_{'arm64' if arch == 'arm64' else 'x64'}.exe"
            print("installer:", f"{OPERA_ARCHIVE}{ver}/win/{fn}")
        elif sys.platform == "darwin":
            print("installer:", f"{OPERA_ARCHIVE}{ver}/mac/Opera_{ver}_Setup.dmg")
        else:
            print("installer: (linux needs root; .deb/.rpm)")
        sys.exit(0)
    if args.refresh:
        print("token:", get_token(force_refresh=True)[:40], "...")
        sys.exit(0)
    out = send_chat_completion(messages=[{"role": "user", "content": args.prompt}])
    print(out.get("content", ""))
