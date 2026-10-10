"""Host tests for the ChatGPT sign-in bridge (services/oauth.py) and the Accounts / LoginSheet UI.

The real ``authgpt_auth.begin_oauth`` / ``complete_from_redirect`` run against a fake
token endpoint on 127.0.0.1 (``OPENAI_TOKEN_URL`` patched) with the loopback callback
server on a free port (``CALLBACK_PORT`` patched); a token store in a temp folder
replaces the per-user default, so nothing touches the real ``~/.glossarion``.

Run from src/mobile with the 3.13 venv:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_oauth.py
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import json
import os
import socket
import sys
import threading
import time
import types
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for entry in (str(APP_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from glossarion_mobile.services.oauth import OAuthBridge, SignInState, provider_for_model  # noqa: E402

authgpt_auth = pytest.importorskip("authgpt_auth")
TIMEOUT = 10.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _jwt(payload: dict) -> str:
    def part(data: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")

    return f"{part({'alg': 'none'})}.{part(payload)}.sig"


class FakeTokenEndpoint:
    """POST /oauth/token -> tokens; records every exchange."""

    def __init__(self) -> None:
        self.requests: list = []
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # noqa: D401
                pass

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                form = dict(urllib.parse.parse_qsl(self.rfile.read(length).decode()))
                endpoint.requests.append(form)
                body = json.dumps({
                    "access_token": f"access-{form.get('code')}",
                    "refresh_token": "refresh-1",
                    "expires_in": 3600,
                    "id_token": _jwt({"email": "reader@example.com",
                                      "https://api.openai.com/auth": {"chatgpt_plan_type": "plus"}}),
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/oauth/token"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def oauth_env(tmp_path, monkeypatch):
    import token_encryption

    endpoint = FakeTokenEndpoint()
    port = _free_port()
    monkeypatch.setattr(authgpt_auth, "OPENAI_TOKEN_URL", endpoint.url)
    monkeypatch.setattr(authgpt_auth, "CALLBACK_PORT", port)
    monkeypatch.setenv("GLOSSARION_OAUTH_RETURN_URL", "glossarion://app/oauth/return")
    token_encryption.set_symmetric_key(os.urandom(32))
    stores: dict = {}

    def get_store(account_id=None):
        key = int(account_id or 0)
        if key not in stores:
            name = "authgpt_tokens.json" if not key else f"authgpt_tokens_{key}.json"
            stores[key] = authgpt_auth.AuthGPTTokenStore(token_file=str(tmp_path / name), account_id=key)
        return stores[key]

    module = types.SimpleNamespace(
        begin_oauth=authgpt_auth.begin_oauth,
        complete_from_redirect=authgpt_auth.complete_from_redirect,
        get_store=get_store,
    )
    env = types.SimpleNamespace(endpoint=endpoint, port=port, module=module, stores=stores, tmp=tmp_path)
    try:
        yield env
    finally:
        endpoint.close()
        token_encryption.set_symmetric_key(None)


class Browser:
    """Captures webbrowser.open(url) and plays the user finishing the ChatGPT login."""

    def __init__(self) -> None:
        self.opened: list = []

    def open(self, url: str) -> bool:
        self.opened.append(url)
        return True

    @property
    def query(self) -> dict:
        return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.opened[-1]).query))

    def finish(self, code: str = "CODE", state: str = None) -> str:
        query = self.query
        redirect = urllib.parse.urlsplit(query["redirect_uri"])
        url = (f"http://127.0.0.1:{redirect.port}{redirect.path}?code={code}"
               f"&state={urllib.parse.quote(state or query['state'])}")
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:  # follows the 302 to /success
            return response.read().decode()


class FakeNative:
    def __init__(self, running: bool = False) -> None:
        self.running = running
        self.calls: list = []

    async def is_job_service_running(self) -> bool:
        return self.running

    async def start_job_service(self, title: str, text: str) -> bool:
        self.calls.append(("start", title, text))
        return True

    async def stop_job_service(self) -> None:
        self.calls.append(("stop",))


async def _wait_for(predicate, timeout=TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


def _bridge(env, browser, **kwargs) -> OAuthBridge:
    kwargs.setdefault("safe_root", None)  # temp token stores (sign-out guard: tests_host/test_accounts_profiles.py)
    return OAuthBridge(auth_module=env.module, opener=browser.open, **kwargs)


# ==========================================================================
# Bridge
# ==========================================================================


def test_loopback_sign_in_with_return_page_and_android_fgs(oauth_env):
    browser = Browser()
    native = FakeNative()
    closed = []

    async def close_browser():
        closed.append(True)

    bridge = _bridge(oauth_env, browser, native=native, is_android=True, close_browser=close_browser)
    steps: list = []
    bridge.subscribe(lambda state: steps.append(state.step))

    async def scenario():
        task = asyncio.ensure_future(bridge.sign_in())
        assert await _wait_for(lambda: browser.opened)
        assert bridge.state.step == "waiting" and bridge.state.auth_url == browser.opened[0]
        query = browser.query
        assert query["redirect_uri"] == f"http://localhost:{oauth_env.port}/auth/callback"
        assert query["code_challenge_method"] == "S256"
        page = await asyncio.to_thread(browser.finish, "CODE")
        assert "glossarion://app/oauth/return?p=authgpt" in page  # GLOSSARION_OAUTH_RETURN_URL seam
        bridge.on_return_link("authgpt")  # the deep link the success page opens
        status = await asyncio.wait_for(task, TIMEOUT)
        await asyncio.sleep(0)
        return status, query

    status, query = asyncio.run(scenario())
    assert status["signed_in"] and status["email"] == "reader@example.com"
    assert steps[:3] == ["opening", "waiting", "exchanging"] and steps[-1] == "done"
    exchange = oauth_env.endpoint.requests[-1]
    assert exchange["code"] == "CODE" and exchange["grant_type"] == "authorization_code"
    challenge = base64.urlsafe_b64encode(hashlib.sha256(exchange["code_verifier"].encode()).digest()).decode().rstrip("=")
    assert challenge == query["code_challenge"]  # PKCE bound to this sign-in
    token_file = oauth_env.tmp / "authgpt_tokens.json"
    assert token_file.is_file() and b"access-CODE" not in token_file.read_bytes()  # encrypted at rest
    assert oauth_env.stores[0].load_tokens()["access_token"] == "access-CODE"
    assert not Path(oauth_env.stores[0].pending_oauth_file).exists()  # pending state cleared
    assert "authgpt" in bridge.signed_in
    assert native.calls[0][0] == "start" and native.calls[-1] == ("stop",)  # short sign-in FGS
    assert closed == [True]  # iOS in-app browser closed on the return link


def test_paste_redirect_url_while_waiting(oauth_env):
    browser = Browser()
    bridge = _bridge(oauth_env, browser)

    async def scenario():
        task = asyncio.ensure_future(bridge.sign_in())
        assert await _wait_for(lambda: browser.opened)
        state = browser.query["state"]
        status = await bridge.complete_with_paste(
            f"http://localhost:{oauth_env.port}/auth/callback?code=PASTED&state={urllib.parse.quote(state)}")
        assert await asyncio.wait_for(task, TIMEOUT) == {}  # the paste finished this sign-in
        return status

    status = asyncio.run(scenario())
    assert status["signed_in"] and bridge.state.step == "done"
    assert oauth_env.endpoint.requests[-1]["code"] == "PASTED"


def test_paste_after_the_app_was_killed_and_state_checks(oauth_env):
    browser = Browser()
    first = _bridge(oauth_env, browser)

    async def start_then_kill():
        task = asyncio.ensure_future(first.sign_in())
        assert await _wait_for(lambda: browser.opened)
        first.cancel()  # the loopback server dies with the app; the pending PKCE state stays on disk
        assert await asyncio.wait_for(task, TIMEOUT) == {}
        return browser.query["state"]

    state = asyncio.run(start_then_kill())
    assert first.state.step == "cancelled"
    assert oauth_env.stores[0].load_pending_oauth()["state"] == state

    second = _bridge(oauth_env, Browser())  # a fresh app process
    assert second.has_pending()
    with pytest.raises(RuntimeError, match="state mismatch"):
        asyncio.run(second.complete_with_paste("CODE2#not-the-state"))
    assert second.state.step == "error" and "state mismatch" in second.state.message
    status = asyncio.run(second.complete_with_paste(f"CODE2#{state}"))
    assert status["signed_in"] and oauth_env.endpoint.requests[-1]["code"] == "CODE2"
    with pytest.raises(RuntimeError, match="No pending ChatGPT sign-in"):
        asyncio.run(_bridge(oauth_env, Browser()).complete_with_paste("CODE3"))


def test_timeout_cancel_sign_out_and_existing_job_service(oauth_env):
    native = FakeNative(running=True)  # a translation already holds the foreground service
    bridge = _bridge(oauth_env, Browser(), native=native, is_android=True, timeout=1)
    with pytest.raises(RuntimeError, match="timed out"):
        asyncio.run(bridge.sign_in())
    assert bridge.state.step == "error" and native.calls == []
    browser = Browser()
    bridge = _bridge(oauth_env, browser)

    async def cancel():
        task = asyncio.ensure_future(bridge.sign_in())
        assert await _wait_for(lambda: browser.opened)
        assert bridge.reopen_browser() and len(browser.opened) == 2
        bridge.cancel()
        return await asyncio.wait_for(task, TIMEOUT)

    assert asyncio.run(cancel()) == {}
    assert bridge.state.step == "cancelled"
    oauth_env.stores[0].save_tokens({"access_token": "a", "refresh_token": "r", "expires_at": time.time() + 600})
    assert bridge.status()["signed_in"] and "authgpt" in bridge.signed_in
    asyncio.run(bridge.sign_out())
    assert not bridge.status()["signed_in"] and "authgpt" not in bridge.signed_in
    with pytest.raises(RuntimeError, match="Unknown sign-in provider"):  # U4: every listed provider signs in
        asyncio.run(bridge.sign_in("bogus"))
    assert provider_for_model("authgpt/gpt-6-luna") == ("authgpt", 0)
    assert provider_for_model("authgpt2/gpt-5") == ("authgpt", 2)
    assert provider_for_model("gemini/x") is None and provider_for_model("authgptx/y") is None


# ==========================================================================
# LoginPanel / Accounts screen
# ==========================================================================


@pytest.mark.skipif(importlib.util.find_spec("flet") is None, reason="flet not installed")
def test_login_panel_and_accounts_screen(oauth_env):
    import flet as ft

    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.accounts import UNAVAILABLE_ACCOUNTS, AccountsScreen, LoginPanel, LoginSheet

    browser = Browser()
    bridge = _bridge(oauth_env, browser)
    done: list = []
    panel = LoginPanel(bridge, on_done=done.append)
    assert panel.start_button.visible and not panel.reopen_button.visible
    panel.apply(SignInState(step="waiting"))
    assert panel.reopen_button.visible and panel.cancel_button.visible and not panel.start_button.visible
    assert panel.step_text.value == "Waiting for sign-in…"
    panel.apply(SignInState(step="error", message="OAuth error: access_denied"))
    assert panel.error_text.visible and panel.start_button.content == "Try again"

    async def via_panel():
        task = asyncio.ensure_future(panel.run())
        assert await _wait_for(lambda: browser.opened)
        await asyncio.to_thread(browser.finish, "PANEL")
        return await asyncio.wait_for(task, TIMEOUT)

    status = asyncio.run(via_panel())
    assert status["signed_in"] and done and done[0]["email"] == "reader@example.com"
    panel.apply(bridge.state)
    assert panel.step_text.value == "Done · signed in as reader@example.com"

    screen = AccountsScreen(parse_route("/settings/accounts"), oauth=bridge)
    body = screen.get_body()
    keys = [getattr(c, "key", None) for c in body.controls]
    assert keys[:4] == ["account-authgpt", "account-authgrok", "account-authcd", "account-authgem"]  # U4: all sign in
    assert not any(getattr(c, "key", None) == "accounts-unavailable" for c in body.controls)  # U12: not listed
    asyncio.run(screen.refresh("authgpt"))
    assert screen.status["signed_in"] and screen.slots["authgpt"] == [0]
    row = screen.slot_rows[("authgpt", 0)]
    assert row.subtitle.value.startswith("✓ Signed in · reader@example.com") and row.trailing.icon == ft.Icons.MORE_VERT
    assert asyncio.run(screen.sign_out("authgpt", 0))
    assert screen.slot_rows[("authgpt", 0)].subtitle.value == "Not signed in"
    assert screen.slot_rows[("authgpt", 0)].trailing.content == "Sign in"
    sheet = LoginSheet(bridge, autostart=False)
    assert sheet.panel in sheet.dialog.content.content.controls


def test_sign_in_again_after_a_failed_paste_frees_the_loopback_port(oauth_env):
    """A failed paste (stale URL, wrong state) left the first sign-in's loopback server on the
    callback port; "Try again" must close it before listening again ("Port ... already in use")."""
    browser = Browser()
    bridge = _bridge(oauth_env, browser)

    async def scenario():
        task = asyncio.ensure_future(bridge.sign_in())
        assert await _wait_for(lambda: browser.opened)
        with pytest.raises(RuntimeError, match="state mismatch"):
            await bridge.complete_with_paste(f"http://localhost:{oauth_env.port}/auth/callback?code=X&state=WRONG")
        assert await asyncio.wait_for(task, TIMEOUT) == {}  # the paste claimed the first sign-in
        assert bridge.state.step == "error"
        retry = asyncio.ensure_future(bridge.sign_in())
        assert await _wait_for(lambda: len(browser.opened) == 2 or retry.done())
        assert not retry.done(), retry.exception() if retry.done() else None
        assert bridge.state.step == "waiting"
        await asyncio.to_thread(browser.finish, "RETRY")
        return await asyncio.wait_for(retry, TIMEOUT)

    status = asyncio.run(scenario())
    assert status["signed_in"] and bridge.state.step == "done"
    assert oauth_env.endpoint.requests[-1]["code"] == "RETRY"
