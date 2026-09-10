"""Arena sessions and the automatically provisioned LMArenaBridge runtime.

Route numbers are one-based: autharena1/ is stored slot 0. autharena0/
is the rotating pool; autharena/ is an alias for autharena1/.
This file is also the managed Python worker entry point (including frozen builds).
"""
from __future__ import annotations

import atexit
import base64
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import zipfile

import requests

REVISION = "e9655ea6d74cddabdfdd651da285aa4ca60091ad"
ADAPTER_VERSION = 2
UV_VERSION = "0.8.22"
ROUTE_RE = re.compile(r"^autharena(\d{0,4})(?:/|$)", re.I)
_lock = threading.RLock()
_file_locks = {"state": threading.RLock(), "setup": threading.RLock()}
_cancel = threading.Event()
_generation = 0
_responses = set()
_pending_requests = {}
_owned = None
_started_callback = None


def set_proxy_started_callback(callback):
    global _started_callback
    _started_callback = callback


def data_dir():
    p = Path(os.environ.get("AUTHARENA_PROXY_DATA_DIR", str(Path.home() / ".glossarion" / "autharena_proxy")))
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_route(model):
    """Return (stored slot or None for rotation, unprefixed model)."""
    match = ROUTE_RE.match(str(model).strip())
    if not match:
        raise ValueError("Expected autharena/, autharena0/, or autharenaN/.")
    number = match.group(1)
    slot = None if number and int(number) == 0 else (int(number) - 1 if number else 0)
    return slot, str(model).strip()[match.end():]


def route_for_slot(slot, model):
    return f"autharena{int(slot) + 1}/{model}"


@contextlib.contextmanager
def disk_lock(name="state"):
    """Serialize setup and encrypted state writes across translation processes."""
    with _file_locks[name], open(data_dir() / (name + ".lock"), "a+b") as f:
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(.1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _load(name):
    from token_encryption import load_encrypted_tokens
    path = data_dir() / name
    return load_encrypted_tokens(str(path)) if path.exists() else {}


def _save(name, value):
    from token_encryption import save_encrypted_tokens
    path = data_dir() / name
    temporary = path.with_suffix(".tmp")
    save_encrypted_tokens(value, str(temporary))
    temporary.chmod(0o600)
    os.replace(temporary, path)


def list_accounts():
    with disk_lock():
        accounts = _load("accounts.enc")
    return [{"slot": int(k), "email": v.get("email", ""), "user_id": v.get("user_id", "")}
            for k, v in sorted(accounts.items(), key=lambda x: int(x[0]))]


def _env():
    # Do not inherit translation prompts or API keys into compiler/runtime processes.
    keep = {"systemroot", "windir", "comspec", "path", "pathext", "temp", "tmp",
            "home", "userprofile", "localappdata", "appdata", "lang", "lc_all",
            "display", "wayland_display", "xdg_runtime_dir", "xdg_config_home", "dbus_session_bus_address"}
    env = {k: v for k, v in os.environ.items() if k.lower() in keep}
    env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", UV_PYTHON_INSTALL_DIR=str(data_dir() / "python"))
    env["AUTHARENA_PROXY_DATA_DIR"] = str(data_dir())
    env["PLAYWRIGHT_BROWSERS_PATH"] = str(data_dir() / "browsers")
    return env


def _run(args, log_fn=print):
    result = subprocess.run([str(a) for a in args], env=_env(), capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=900,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError("Arena runtime setup failed: " + result.stderr[-1800:])
    return result.stdout.strip()


def _ensure_browser(runtime, python, log_fn=print, refresh=False):
    marker = runtime / "browser-ready"
    if not refresh and marker.exists() and Path(marker.read_text(encoding="utf-8")).is_file():
        return
    log_fn("Arena: installing the internal browser…")
    _run([python, "-m", "playwright", "install", "chromium"], log_fn)
    executable = _run([python, "-c", "from playwright.sync_api import sync_playwright; p=sync_playwright().start(); print(p.chromium.executable_path); p.stop()"], log_fn)
    if not Path(executable).is_file():
        raise RuntimeError("Arena internal browser installation did not produce an executable.")
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(executable, encoding="utf-8")
    os.replace(temporary, marker)


def _download(url, target):
    with requests.get(url, stream=True, timeout=(15, 180)) as response:
        response.raise_for_status()
        with open(target, "wb") as out:
            for block in response.iter_content(1024 * 1024):
                out.write(block)


def _extract(archive, target):
    """Reject traversal and links before extracting downloaded archives."""
    target = Path(target).resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                if not (target / item.filename).resolve().is_relative_to(target):
                    raise RuntimeError("Unsafe runtime archive")
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError("Runtime archive contains a symbolic link")
            z.extractall(target)
    else:
        with tarfile.open(archive) as t:
            for item in t.getmembers():
                if not (target / item.name).resolve().is_relative_to(target) or item.issym() or item.islnk() or item.isdev():
                    raise RuntimeError("Unsafe runtime archive")
            t.extractall(target, filter="data")


def _source_file(name):
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    for p in (root / name, root / "src" / name, Path(__file__).with_name(name)):
        if p.is_file():
            return p
    raise RuntimeError(f"Arena packaged runtime source missing: {name}")


def _ensure_runtime(log_fn=print):
    root = data_dir()
    runtime = root / ("bridge-" + REVISION)
    python = runtime / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if (runtime / "ready").exists() and python.exists():
        _ensure_browser(runtime, python, log_fn)
        return runtime, python
    log_fn("Arena: downloading the managed Python runtime and LMArenaBridge…")
    machine = platform.machine().lower()
    arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
    system = platform.system()
    triples = {"Windows": f"{arch}-pc-windows-msvc", "Darwin": f"{arch}-apple-darwin", "Linux": f"{arch}-unknown-linux-gnu"}
    if system not in triples or machine not in ("amd64", "x86_64", "arm64", "aarch64"):
        raise RuntimeError("Arena automatic runtime setup does not support this OS/architecture.")
    uv = root / ("uv.exe" if os.name == "nt" else "uv")
    with tempfile.TemporaryDirectory(prefix="setup-", dir=root) as work:
        work = Path(work)
        if not uv.exists():
            asset = f"uv-{triples[system]}." + ("zip" if os.name == "nt" else "tar.gz")
            archive = work / asset
            _download(f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{asset}", archive)
            _extract(archive, work / "uv")
            candidate = next((work / "uv").rglob(uv.name))
            shutil.copy2(candidate, uv)
            uv.chmod(0o700)
        archive = work / "bridge.zip"
        _download(f"https://github.com/CloudWaddie/LMArenaBridge/archive/{REVISION}.zip", archive)
        _extract(archive, work / "bridge")
        extracted = next((work / "bridge").iterdir())
        code = (extracted / "src/main.py").read_text(encoding="utf-8")
        for symbol in ("async def api_chat_completions", "async def get_initial_data", "STRICT_BROWSER_FETCH_MODELS"):
            if symbol not in code:
                raise RuntimeError("LMArenaBridge adapter compatibility check failed.")
        # venv paths are absolute; build in a version-specific final directory.
        # The ready marker is the atomic publication boundary for other processes.
        runtime.mkdir(exist_ok=True)
        shutil.copytree(extracted, runtime / "bridge", dirs_exist_ok=True)
        _run([uv, "python", "install", "3.12"], log_fn)
        _run([uv, "venv", "--python", "3.12", runtime / "venv"], log_fn)
        log_fn("Arena: installing proxy dependencies…")
        _run([uv, "pip", "install", "--python", python, "-r", runtime / "bridge/requirements.txt", "requests", "cryptography"], log_fn)
        _ensure_browser(runtime, python, log_fn, refresh=True)
        (runtime / "ready").write_text(REVISION, encoding="ascii")
    return runtime, python


def check_proxy_health():
    try:
        state = _load("service.enc")
        response = requests.get(state["url"] + "/health", headers={"Authorization": "Bearer " + state["key"]}, timeout=2)
        compatible = response.ok and response.json().get("revision") == REVISION
        return {"running": compatible and response.json().get("adapter_version") == ADAPTER_VERSION,
                "outdated": compatible and response.json().get("adapter_version") != ADAPTER_VERSION, **state}
    except Exception:
        return {"running": False}


def ensure_proxy_running(log_fn=print, notify_started=True):
    global _owned
    with disk_lock("setup"):
        status = check_proxy_health()
        if status["running"]:
            return status
        if status.get("outdated"):
            requests.post(status["url"] + "/shutdown", headers={"Authorization": "Bearer " + status["key"]}, timeout=5).raise_for_status()
        runtime, python = _ensure_runtime(log_fn or print)
        for name in ("autharena_proxy.py", "token_encryption.py"):
            shutil.copy2(_source_file(name), runtime / name)
        key = secrets.token_urlsafe(32)
        # The child binds port 0 itself and publishes its authenticated endpoint.
        _owned = subprocess.Popen([str(python), str(runtime / "autharena_proxy.py"), "--worker"],
                                 stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 env=_env(), cwd=runtime, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        _owned.stdin.write(json.dumps({"key": key}).encode() + b"\n")
        _owned.stdin.close()
        # Worker publishes state without taking the parent-held setup lock.
        for _ in range(120):
            status = check_proxy_health()
            if status["running"] and status.get("key") == key:
                if notify_started and _started_callback:
                    _started_callback()
                return status
            if _owned.poll() is not None:
                raise RuntimeError("Arena proxy exited during initialization. Check the installed runtime dependencies.")
            time.sleep(.25)
    raise RuntimeError("Arena proxy did not become healthy.")


def _request(path, payload=None, timeout=600):
    status = ensure_proxy_running()
    method = requests.post if payload is not None else requests.get
    response = method(status["url"] + path, headers={"Authorization": "Bearer " + status["key"]},
                      **({"json": payload} if payload is not None else {}), timeout=timeout)
    if not response.ok:
        raise RuntimeError(response.json().get("detail", "Arena request failed"))
    return response.json()


def open_login(log_fn=print, account_id=0):
    ensure_proxy_running(log_fn=log_fn)
    if log_fn:
        log_fn("Arena Login: opening the internal browser. Finish sign-in on the Arena page.")
    return _request("/login", {"slot": account_id}, timeout=660)


def list_models(timeout=30):
    if not check_proxy_health().get("running"):
        return []
    return _request("/v1/models", timeout=timeout).get("data", [])


def capture_cancel_generation():
    return _generation


def is_cancel_generation_cancelled(generation):
    return _cancel.is_set() or (generation is not None and generation != _generation)


def is_cancelled():
    return _cancel.is_set()


def cancel_stream():
    global _generation
    with _lock:
        _generation += 1
        _cancel.set()
        active = list(_responses)
        pending = list(_pending_requests.items())
    def notify_worker():
        for request_id, state in pending:
            with contextlib.suppress(Exception):
                requests.post(state["url"] + "/cancel", json={"id": request_id},
                              headers={"Authorization": "Bearer " + state["key"]}, timeout=2)
    if pending:
        threading.Thread(target=notify_worker, daemon=True, name="Arena cancellation").start()
    for response in active:
        # Interrupt an outstanding read before close() waits for urllib3's lock.
        with contextlib.suppress(Exception):
            import socket
            response.raw._fp.fp.raw._sock.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(Exception):
            response.close()


def reset_cancel():
    _cancel.clear()


def visible_stream():
    name, default = ("ALLOW_AUTHGPT_BATCH_STREAM_LOGS", "0") if os.getenv("BATCH_TRANSLATION") == "1" else ("LOG_STREAM_CHUNKS", "1")
    return os.getenv(name, default).strip().lower() not in ("", "0", "false", "no", "off")


def consume_stream(lines, log_fn=print, log_stream=True, cancel_generation=None):
    text, thinking, usage, finish = [], [], None, None
    done = False
    for line in lines:
        if is_cancel_generation_cancelled(cancel_generation):
            raise RuntimeError("Arena stream cancelled")
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        event = json.loads(data)
        if event.get("error"):
            error = event["error"]
            raise RuntimeError("Arena stream failed: " + str(error.get("message", "upstream error") if isinstance(error, dict) else error))
        usage = event.get("usage") or usage
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta", {})
            for field, target in (("content", text), ("reasoning_content", thinking), ("reasoning", thinking)):
                chunk = delta.get(field)
                if isinstance(chunk, str) and chunk:
                    target.append(chunk)
                    if log_stream and log_fn:
                        log_fn(chunk)
            finish = choice.get("finish_reason") or finish
    if is_cancel_generation_cancelled(cancel_generation):
        raise RuntimeError("Arena stream cancelled")
    if not done or not finish:
        raise RuntimeError("Arena stream interrupted before its completion marker; partial output was not retried.")
    return {"content": "".join(text), "reasoning_content": "".join(thinking), "usage": usage, "finish_reason": finish}


def send_message_stream(messages, model, temperature=0.7, max_tokens=None, timeout=600,
                        log_fn=print, log_stream=None, account_id=0, cancel_generation=None):
    generation = capture_cancel_generation() if cancel_generation is None else cancel_generation
    if is_cancel_generation_cancelled(generation):
        raise RuntimeError("Arena stream cancelled")
    status = ensure_proxy_running(log_fn=log_fn)
    if is_cancel_generation_cancelled(generation):
        raise RuntimeError("Arena stream cancelled")
    payload = {"model": model, "messages": messages, "stream": True, "temperature": temperature, "account_slot": account_id}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    request_id = secrets.token_hex(16)
    payload["request_id"] = request_id
    with _lock:
        _pending_requests[request_id] = status
    response = None
    try:
        if is_cancel_generation_cancelled(generation):
            raise RuntimeError("Arena stream cancelled")
        response = requests.post(status["url"] + "/v1/chat/completions", json=payload,
                                 headers={"Authorization": "Bearer " + status["key"]}, stream=True, timeout=(15, timeout))
        with _lock:
            _responses.add(response)
        if is_cancel_generation_cancelled(generation):
            raise RuntimeError("Arena stream cancelled")
        if not response.ok:
            raise RuntimeError("Arena: " + response.json().get("detail", f"HTTP {response.status_code}"))
        response.encoding = "utf-8"
        return consume_stream(response.iter_lines(decode_unicode=True, chunk_size=1), log_fn,
                              visible_stream() if log_stream is None else log_stream, generation)
    except Exception:
        if is_cancel_generation_cancelled(generation):
            raise RuntimeError("Arena stream cancelled") from None
        raise
    finally:
        with _lock:
            _responses.discard(response)
            _pending_requests.pop(request_id, None)
        if response is not None:
            response.close()


send_message = send_message_stream


def shutdown_proxy():
    cancel_stream()
    if _owned is not None and _owned.poll() is None:
        try:
            state = _load("service.enc")
            requests.post(state["url"] + "/shutdown", headers={"Authorization": "Bearer " + state["key"]}, timeout=2)
            _owned.wait(timeout=5)
        except Exception:
            _owned.terminate()


atexit.register(shutdown_proxy)


def session_from_cookies(cookies):
    allowed = [c for c in cookies if c.get("domain", "").lstrip(".") in ("arena.ai", "lmarena.ai")]
    chunks = {c["name"]: c["value"] for c in allowed}
    token = chunks.get("arena-auth-prod-v1", "")
    if not token:
        parts = sorted((int(k.rsplit(".", 1)[1]), v) for k, v in chunks.items()
                       if re.fullmatch(r"arena-auth-prod-v1\.\d+", k))
        token = "".join(v for _, v in parts)
    try:
        raw = token.removeprefix("base64-")
        session = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
        user = session.get("user", {})
        if not user.get("id") or user.get("is_anonymous", False) or not user.get("email"):
            return None
        if session.get("expires_at", 0) <= time.time():
            return None
        return {"token": token, "cookies": allowed, "email": user["email"], "user_id": user["id"],
                "expires_at": session["expires_at"]}
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


async def _serve_worker(key):
    """Authenticated loopback broker; independent upstream module state per slot."""
    import asyncio
    import copy
    import importlib
    import socket
    import types
    from fastapi import FastAPI, HTTPException, Request
    globals()["Request"] = Request  # FastAPI resolves postponed endpoint annotations.
    from starlette.responses import JSONResponse, StreamingResponse
    from playwright.async_api import async_playwright
    import uvicorn

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    browser = None
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(data_dir() / "browsers")
    playwright = await async_playwright().start()
    slots = {}
    slot_init_lock = asyncio.Lock()
    connect_lock = asyncio.Lock()
    login_lock = asyncio.Lock()
    cursor = 0
    jobs = {}
    cancelled_jobs = set()

    @app.middleware("http")
    async def authorize(request, call_next):
        if not secrets.compare_digest(request.headers.get("authorization", ""), "Bearer " + key):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        try:
            return await call_next(request)
        except Exception:
            return JSONResponse({"detail": "Arena operation failed. Reconnect with Arena Login."}, status_code=503)

    async def connect():
        async with connect_lock:
            return await connect_once()

    async def connect_once():
        nonlocal browser
        if browser is not None and browser.is_connected():
            return browser
        try:
            browser = await playwright.chromium.launch(headless=False, timeout=60000)
        except Exception as exc:
            raise HTTPException(503, f"Arena could not open its internal browser: {exc}")
        return browser

    @app.get("/health")
    async def health():
        return {"revision": REVISION, "adapter_version": ADAPTER_VERSION}

    @app.post("/login")
    async def login(request: Request):
        body = await request.json()
        async with login_lock:
            chrome = await connect()
            # Each login owns its context; + New never inherits another account.
            context = await chrome.new_context()
            try:
                page = await context.new_page()
                await page.goto("https://arena.ai/", wait_until="domcontentloaded")
                await page.bring_to_front()
                clicked = False
                for _ in range(300):
                    cookies = await context.cookies(["https://arena.ai/", "https://lmarena.ai/"])
                    account = session_from_cookies(cookies)
                    if account and body.get("slot") is not None:
                        expected = next((item for item in list_accounts() if item["slot"] == int(body["slot"])), None)
                        if expected and expected.get("user_id") and expected["user_id"] != account["user_id"]:
                            raise HTTPException(409, f"The internal browser is signed into a different Arena account. Switch accounts on the Arena page, then reconnect account {body['slot']}.")
                    if account and body.get("slot") is None:
                        known = list_accounts()
                        if any(v["user_id"] == account["user_id"] for v in known):
                            await asyncio.sleep(1)
                            continue
                    if account:
                        with disk_lock():
                            saved = _load("accounts.enc")
                            slot = body.get("slot")
                            if slot is None:
                                slot = max([int(k) for k in saved] + [-1]) + 1
                            slot = int(slot)
                            if slot < 0 or slot > 9998:
                                raise HTTPException(400, "Invalid Arena account slot")
                            saved[str(slot)] = account
                            _save("accounts.enc", saved)
                        previous = slots.pop(slot, None)
                        if previous:
                            async with previous["lock"]:
                                await previous["context"].close()
                        return {"slot": slot, "email": account["email"]}
                    if not clicked:
                        for role in ("button", "link"):
                            control = page.get_by_role(role, name=re.compile(r"\b(sign\s*in|log\s*in|login)\b", re.I))
                            if await control.count() and await control.first.is_visible():
                                await control.first.click()
                                clicked = True
                                break
                    await asyncio.sleep(1)
                raise HTTPException(408, "Arena Login timed out before a signed-in session was available.")
            finally:
                with contextlib.suppress(Exception):
                    await context.close()

    async def get_slot(slot):
        async with slot_init_lock:
            return await initialize_slot(slot)

    async def initialize_slot(slot):
        chrome = await connect()
        with disk_lock():
            account = _load("accounts.enc").get(str(slot))
        if not account or account.get("expires_at", 0) <= time.time():
            raise HTTPException(401, f"Arena account {slot} needs Arena Login.")
        if slot in slots and slots[slot]["context"] in chrome.contexts:
            return slots[slot]
        context = await chrome.new_context()
        await context.add_cookies(account["cookies"])
        namespace = f"arena_slot_{slot}_{secrets.token_hex(4)}"
        package = types.ModuleType(namespace)
        package.__path__ = [str(Path(__file__).parent / "bridge")]
        sys.modules[namespace] = package
        main = importlib.import_module(namespace + ".src.main")
        config_module = importlib.import_module(namespace + ".src.config")
        auth = importlib.import_module(namespace + ".src.auth")
        transport = importlib.import_module(namespace + ".src.transport")
        cfg = {"auth_tokens": [account["token"]], "auth_token": account["token"], "api_keys": [{"key": key, "rpm": 100000}],
               "persist_arena_auth_cookie": False, "browser_cookies": {c["name"]: c["value"] for c in account["cookies"]}}
        config_module._apply_config_defaults(cfg)
        models = []
        def save_config(value, **kwargs):
            cfg.update(copy.deepcopy(value))
        for module in (main, config_module):
            module.get_config = lambda: copy.deepcopy(cfg)
            module.save_config = save_config
            module.get_models = lambda: list(models)
            module.save_models = lambda value: (models.clear(), models.extend(value))
        main.DEBUG = False
        main.debug_print = lambda *args, **kwargs: None
        main.print = lambda *args, **kwargs: None
        # Discovery and requests use isolated contexts in the app-owned browser.
        # No extension, personal browser profile, or CAPTCHA solver is used.
        class ContextLease:
            def __init__(self, **kwargs):
                self.before = set(context.pages)
            async def __aenter__(self):
                return context
            async def __aexit__(self, *args):
                for p in context.pages:
                    if p not in self.before:
                        await p.close()
        async def no_challenge_click(*args, **kwargs):
            return False
        main.AsyncCamoufox = ContextLease
        main.click_turnstile = no_challenge_click
        main._userscript_proxy_is_active = lambda: False
        main.find_chrome_executable = lambda: "managed-chromium"
        async def no_refresh(*args, **kwargs):
            return None
        main.refresh_recaptcha_token = no_refresh
        main.maybe_refresh_expired_auth_tokens = no_refresh
        main.maybe_refresh_expired_auth_tokens_via_lmarena_http = no_refresh
        main.get_cached_recaptcha_token = lambda: ""
        try:
            await main.get_initial_data()
        except BaseException:
            await context.close()
            raise
        if not models:
            await context.close()
            raise HTTPException(503, "Arena's model catalog was unavailable. Complete any browser challenge and retry.")
        main.STRICT_BROWSER_FETCH_MODELS = {m["publicName"] for m in models}
        state = {"main": main, "context": context, "lock": asyncio.Lock(), "models": models, "submitted": False, "usage": None}

        async def fetch(http_method, url, payload, auth_token="", timeout_seconds=120, **kwargs):
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if parsed.hostname not in ("arena.ai", "lmarena.ai"):
                raise RuntimeError("Unexpected Arena transport origin")
            page = await context.new_page()
            queue = asyncio.Queue(maxsize=32)
            headers_ready = asyncio.Event()
            done_event = asyncio.Event()
            result = {"status": 502, "headers": {}}
            async def emit(source, item):
                if "status" in item:
                    result.update(item)
                    headers_ready.set()
                elif "line" in item:
                    await queue.put(item["line"])
            await page.expose_binding("arenaEmit", emit)
            await page.goto("https://arena.ai/", wait_until="domcontentloaded")
            sitekey, action = main.get_recaptcha_settings(cfg)
            async def pump():
                try:
                    await page.evaluate("""async ({url, method, payload, sitekey, action}) => {
                        if (globalThis.grecaptcha?.enterprise && sitekey) {
                            await new Promise(resolve => grecaptcha.enterprise.ready(resolve));
                            payload.recaptchaV3Token = await grecaptcha.enterprise.execute(sitekey, {action});
                        }
                        const response = await fetch(url, {method, credentials:'include',
                            headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
                        await arenaEmit({status:response.status, headers:Object.fromEntries(response.headers)});
                        const reader = response.body.getReader(); const decoder = new TextDecoder(); let pending='';
                        while (true) { const {done,value} = await reader.read();
                            pending += decoder.decode(value || new Uint8Array(), {stream:!done});
                            let end; while ((end=pending.indexOf('\\n')) >= 0) {
                                await arenaEmit({line:pending.slice(0,end).replace(/\\r$/, '')}); pending=pending.slice(end+1);
                            }
                            if (done) break;
                        }
                        if (pending) await arenaEmit({line:pending});
                    }""", {"url": "https://arena.ai" + parsed.path, "method": http_method, "payload": payload,
                             "sitekey": sitekey, "action": action})
                finally:
                    headers_ready.set()
                    done_event.set()
            task = asyncio.create_task(asyncio.wait_for(pump(), timeout_seconds))
            try:
                await asyncio.wait_for(headers_ready.wait(), timeout_seconds)
                if task.done() and task.exception():
                    raise task.exception()
            except BaseException:
                task.cancel()
                await page.close()
                raise
            class Response(transport.BrowserFetchStreamResponse):
                async def aiter_lines(self):
                    async for line in super().aiter_lines():
                        value = line.removeprefix("data:").strip()
                        if value.startswith("ad:"):
                            metadata = json.loads(value[3:])
                            if not metadata.get("finishReason"):
                                raise RuntimeError("Arena omitted the finish reason")
                            state["usage"] = metadata.get("usage") or state["usage"]
                        elif value.startswith("{"):
                            event = json.loads(value)
                            state["usage"] = event.get("usage") or state["usage"]
                            for choice in event.get("choices", []):
                                if choice.get("finish_reason"):
                                    yield line
                                    yield "ad:" + json.dumps({"finishReason": choice["finish_reason"]})
                                    break
                            else:
                                yield line
                            continue
                        yield line
                    await task
                async def aclose(self):
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
                    await page.close()
                async def __aexit__(self, *args):
                    await self.aclose()
            return Response(result["status"], result["headers"], lines_queue=queue, done_event=done_event, method=http_method, url=url)
        async def submit_once(*args, **kwargs):
            if state["submitted"]:
                return transport.BrowserFetchStreamResponse(400, {}, text="Arena request was already submitted; reconnect with Arena Login before retrying.")
            state["submitted"] = True
            before = set(context.pages)
            try:
                return await fetch(*args, **kwargs)
            except asyncio.CancelledError:
                for page in context.pages:
                    if page not in before:
                        await page.close()
                raise
            except Exception:
                for page in context.pages:
                    if page not in before:
                        await page.close()
                # Returning an explicit response prevents upstream from falling
                # through to its unrelated HTTP/browser launch fallbacks.
                return transport.BrowserFetchStreamResponse(400, {}, text="The internal browser could not submit the Arena request. Use Arena Login to reconnect.")
        main.fetch_lmarena_stream_via_chrome = submit_once
        main.fetch_lmarena_stream_via_camoufox = submit_once
        slots[slot] = state
        return state

    @app.get("/v1/models")
    async def models_route():
        accounts = list_accounts()
        if not accounts:
            return {"data": []}
        state = await get_slot(accounts[0]["slot"])
        return {"data": [{"id": m["publicName"]} for m in state["models"]]}

    @app.post("/cancel")
    async def cancel_request(request: Request):
        request_id = str((await request.json()).get("id", ""))
        if len(cancelled_jobs) > 1024:
            cancelled_jobs.clear()
        cancelled_jobs.add(request_id)
        task = jobs.get(request_id)
        if task is not None:
            task.cancel()
        return {"cancelled": True}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        request_id = str(body.pop("request_id", secrets.token_hex(16)))
        if request_id in cancelled_jobs:
            cancelled_jobs.discard(request_id)
            raise HTTPException(409, "Arena request cancelled")
        jobs[request_id] = asyncio.current_task()
        try:
            return await prepare_chat(request, body, request_id)
        except BaseException:
            jobs.pop(request_id, None)
            raise

    async def prepare_chat(request, body, request_id):
        nonlocal cursor
        slot = body.pop("account_slot", 0)
        if slot is None:
            accounts = list_accounts()
            if not accounts:
                raise HTTPException(401, "No Arena accounts. Use Arena Login.")
            slot = accounts[cursor % len(accounts)]["slot"]
            cursor += 1
        state = await get_slot(int(slot))
        main = state["main"]
        body["stream"] = True
        body.pop("conversation_id", None)
        request._body = json.dumps(body).encode()
        request._json = body
        await state["lock"].acquire()
        main.chat_sessions.clear()
        state["submitted"] = False
        state["usage"] = None
        try:
            response = await main.api_chat_completions(request, {"key": key, "rpm": 100000})
        except BaseException:
            state["lock"].release()
            raise
        if not hasattr(response, "body_iterator"):
            state["lock"].release()
            jobs.pop(request_id, None)
            return response
        async def chunks():
            try:
                jobs[request_id] = asyncio.current_task()
                if request_id in cancelled_jobs:
                    raise asyncio.CancelledError()
                async for chunk in response.body_iterator:
                    if state["usage"] and isinstance(chunk, str) and chunk.startswith("data: {"):
                        event = json.loads(chunk[6:])
                        if any(c.get("finish_reason") for c in event.get("choices", [])):
                            event["usage"] = state["usage"]
                            chunk = "data: " + json.dumps(event) + "\n\n"
                    yield chunk
            finally:
                main.chat_sessions.clear()
                main.conversation_tokens.clear()
                main.request_failed_tokens.clear()
                state["lock"].release()
                jobs.pop(request_id, None)
                cancelled_jobs.discard(request_id)
        return StreamingResponse(chunks(), media_type="text/event-stream")

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    _save("service.enc", {"url": f"http://127.0.0.1:{sock.getsockname()[1]}", "key": key})
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    @app.post("/shutdown")
    async def shutdown():
        for task in list(jobs.values()):
            task.cancel()
        server.should_exit = True
        return {"stopping": True}
    try:
        await server.serve(sockets=[sock])
    finally:
        for state in slots.values():
            with contextlib.suppress(Exception):
                await state["context"].close()
        # This browser belongs exclusively to the Arena worker.
        if browser is not None:
            with contextlib.suppress(Exception):
                await browser.close()
        await playwright.stop()


def create_login_controls(parent, get_model, set_model, log_fn=print, on_login=None, selector_enabled=None):
    """Shared Qt controls; capture the target slot before starting background work."""
    from PySide6.QtCore import Signal, Slot
    from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QComboBox, QInputDialog, QMessageBox

    class Controls(QWidget):
        completed = Signal(object, object)
        progress = Signal(str)

        def __init__(self):
            super().__init__(parent)
            row = QHBoxLayout(self)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(3)
            self.accounts = QComboBox(self)
            self.accounts.setMinimumWidth(65)
            self.login_button = QPushButton("Arena Login", self)
            self.login_button.setToolTip("Log into Arena in the automatically installed internal browser")
            row.addWidget(self.accounts)
            row.addWidget(self.login_button)
            self.login_button.clicked.connect(self.login)
            self.accounts.activated.connect(self.select_account)
            self.completed.connect(self.finished)
            self.progress.connect(log_fn or print)
            self.busy = False
            self.refresh()

        def refresh(self, *args):
            model = str(get_model() or "")
            match = ROUTE_RE.match(model.strip())
            self.setVisible(bool(match))
            if not match:
                return
            slot, _ = parse_route(model)
            self.accounts.blockSignals(True)
            self.accounts.clear()
            ids = sorted({0, *[a["slot"] for a in list_accounts()], *([] if slot is None else [slot])})
            for aid in ids:
                self.accounts.addItem(str(aid), aid)
            self.accounts.addItem("+ New", "new")
            self.accounts.setCurrentIndex(max(0, self.accounts.findData(slot)))
            self.accounts.blockSignals(False)
            self.accounts.setVisible(bool(match.group(1)) and slot is not None and (selector_enabled is None or selector_enabled()))
            self.accounts.setEnabled(not self.busy)
            self.login_button.setEnabled(not self.busy)

        def select_account(self, index):
            selected = self.accounts.itemData(index)
            if selected == "new":
                self.start(None, True)
            else:
                _, model = parse_route(get_model())
                set_model(route_for_slot(selected, model))
                self.refresh()

        def login(self):
            slot, _ = parse_route(get_model())
            if slot is None:
                entries = list_accounts()
                labels = [str(a["slot"]) for a in entries] + ["+ New"]
                choice, ok = QInputDialog.getItem(self, "Arena Login", "Account", labels, 0, False)
                if not ok:
                    return
                slot = None if choice == "+ New" else int(choice)
            self.start(slot, False)

        def start(self, slot, update_route):
            if self.busy:
                return
            self.busy = True
            self.refresh()
            # A row/model can change while login is open. Never retarget the
            # credential write or overwrite a later model edit on completion.
            original = get_model()
            def worker():
                try:
                    if slot is None:
                        self.progress.emit("Arena: sign in with the new account in the internal browser.")
                    result = open_login(log_fn=self.progress.emit, account_id=slot)
                    self.completed.emit((result, original, update_route), None)
                except Exception as exc:
                    with contextlib.suppress(RuntimeError):
                        self.completed.emit(None, str(exc))
            threading.Thread(target=worker, daemon=True, name="Arena Login").start()

        @Slot(object, object)
        def finished(self, result, error):
            self.busy = False
            if error:
                self.progress.emit("Arena Login: " + error)
                QMessageBox.warning(self, "Arena Login", error)
            else:
                account, original, update_route = result
                if update_route and get_model() == original:
                    _, model = parse_route(original)
                    set_model(route_for_slot(account["slot"], model))
                self.progress.emit(f"Arena account {account['slot']} connected.")
                if on_login:
                    on_login()
            self.refresh()

    return Controls()


def install_combo_login(parent, combo, log_fn=print, on_login=None):
    """Attach the shared control to dynamically created multi-key model fields."""
    from PySide6.QtCore import QTimer
    if getattr(combo, "_autharena_controls", None) is not None or not combo.isEditable():
        return
    editor = combo.lineEdit()
    controls = create_login_controls(editor, combo.currentText, combo.setCurrentText, log_fn, on_login)
    combo._autharena_controls = controls
    margins = editor.textMargins()
    previous_width = 0
    def position(*args):
        nonlocal previous_width
        from shiboken6 import isValid
        if not isValid(combo) or not isValid(controls):
            return
        controls.refresh()
        width = controls.sizeHint().width() if ROUTE_RE.match(combo.currentText().strip()) else 0
        controls.resize(width, editor.height())
        controls.move(max(0, editor.width() - width), 0)
        controls.raise_()
        current = editor.textMargins()
        if width:
            editor.setTextMargins(current.left(), current.top(), width + 2, current.bottom())
        elif previous_width and current.right() == previous_width + 2:
            editor.setTextMargins(current.left(), current.top(), margins.right(), current.bottom())
        previous_width = width
    original = editor.resizeEvent
    def resized(event):
        original(event)
        position()
    editor.resizeEvent = resized
    combo.currentTextChanged.connect(position)
    QTimer.singleShot(0, position)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        import asyncio
        asyncio.run(_serve_worker(json.loads(sys.stdin.readline())["key"]))
    else:
        import argparse
        parser = argparse.ArgumentParser(description="Manage Glossarion's Arena proxy; no arguments starts the service until Ctrl+C.")
        parser.add_argument("--status", action="store_true", help="Check health without starting the service or browser")
        parser.add_argument("--login", metavar="PREFIX", help="Sign in to an account, e.g. autharena1/ for stored account 0")
        args = parser.parse_args()
        try:
            if args.status:
                print("Arena proxy is " + ("running" if check_proxy_health().get("running") else "stopped"))
            elif args.login:
                slot, _ = parse_route(args.login)
                account = open_login(account_id=slot)
                print(f"Arena account {account['slot']} connected.")
            else:
                ensure_proxy_running()
                print("Arena proxy is running. Press Ctrl+C to stop.")
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
