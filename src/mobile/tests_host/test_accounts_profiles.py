"""Host tests for U4 Accounts, Profiles & prompts and the Data / About settings pages.

* ``services/oauth.py`` multi-provider OAuthBridge against fake auth modules: a loopback
  provider (Claude / Gemini contract: ``begin_oauth`` / ``complete_from_redirect``, paste
  of ``code#state``), the Grok device-code flow (``begin_device_login`` /
  ``poll_device_login`` with stop), account slots from token files / the model / key
  pools / "+ Add account", the sign-out guard for token files outside the app data folder,
  Gemini projects and status.
* Accounts screen, LoginPanel (device code), Welcome step 2.
* Profiles and assistant prefill against a fake ``prompt_profiles`` core implementing the
  U4 contract with the desktop semantics (other_settings / translator_gui); the round trip
  through the real ``MobileConfigStore`` writes exactly the desktop key set. When the real
  ``prompt_profiles`` core is importable its config output is compared with the desktop
  file writer too.
* All prompts (real settings schema), PromptEditor helpers, Appearance, Storage, Backup &
  restore (real ``config_store``), Import from desktop (desktop-key decrypt + device-key
  re-encrypt), About, Danger zone (the reset's preserved keys are executed from the
  desktop source and compared), the feature installer.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_accounts_profiles.py
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
import sys
import textwrap
import threading
import time
import types
import zipfile
from pathlib import Path
from typing import Optional

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from glossarion_mobile.services.oauth import (  # noqa: E402
    PROVIDER_INFO,
    PROVIDERS,
    OAuthBridge,
    SignInState,
    provider_for_model,
    slot_key,
)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not _has("flet"), reason="flet not installed")
needs_crypto = pytest.mark.skipif(not _has("cryptography"), reason="cryptography not installed")
TIMEOUT = 10.0


def _run(coro):
    return asyncio.run(coro)


async def _wait_for(predicate, timeout=TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


# ==========================================================================
# Fake auth modules (the U4 begin/complete and device-code contracts)
# ==========================================================================


class FakeStore:
    """Token store with the shared API (``_token_file``, load/save/clear, has_tokens, account_info)."""

    def __init__(self, token_file: Path, account_id: int, provider: str) -> None:
        self._token_file = str(token_file)
        self.account_id = account_id
        self.provider = provider
        self.pending: Optional[dict] = None
        self.cleared_logout_flag = 0

    def load_tokens(self):
        try:
            return json.loads(Path(self._token_file).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def save_tokens(self, tokens):
        Path(self._token_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self._token_file).write_text(json.dumps(tokens), encoding="utf-8")

    def clear_tokens(self):
        try:
            os.remove(self._token_file)
        except OSError:
            pass

    @property
    def has_tokens(self):
        return bool((self.load_tokens() or {}).get("access_token"))

    @property
    def account_info(self):
        tokens = self.load_tokens() or {}
        return {"email": tokens.get("email", ""), "name": tokens.get("name", ""), "source": tokens.get("_source", "")}

    def get_valid_access_token(self, auto_login=True):
        tokens = self.load_tokens() or {}
        if not tokens.get("access_token"):
            raise RuntimeError("not signed in")
        return tokens["access_token"]

    def load_pending_oauth(self):
        return self.pending

    def clear_pending_oauth(self):
        self.pending = None

    def clear_logout_flag(self):
        self.cleared_logout_flag += 1


class FakeLoopbackSession:
    def __init__(self, store: FakeStore, account_id: int, state: str) -> None:
        self.store = store
        self.account_id = account_id
        self.state = state
        self.auth_url = f"https://auth.example/authorize?state={state}&slot={account_id}"
        self._event = threading.Event()
        self.code: Optional[str] = None
        self.closed = False

    @property
    def server_running(self):
        return not self.closed

    @property
    def callback_received(self):
        return self._event.is_set()

    def wait_for_callback(self, timeout=None):
        return self._event.wait(timeout)

    def deliver(self, code: str) -> None:
        self.code = code
        self._event.set()

    def close(self):
        self.closed = True


class FakeAuthModule(types.ModuleType):
    """Loopback provider: ``begin_oauth`` / ``complete_from_redirect`` + ``get_store``."""

    def __init__(self, provider: str, token_dir: Path) -> None:
        super().__init__(f"fake_{provider}_auth")
        self.provider = provider
        self._DEFAULT_TOKEN_DIR = str(token_dir)
        self.stores: dict = {}
        self.sessions: list = []
        self.exchanges: list = []
        self._cached_project_id: dict = {}
        self._project_set_by_gui: dict = {}
        self.projects: Optional[list] = None
        self.status_result: dict = {"verified": True, "sub_label": "Pro", "credit_label": "OK",
                                    "quota_lines": ["  gemini-pro: 10%"]}

    def get_store(self, account_id=None):
        key = int(account_id or 0)
        if key not in self.stores:
            name = f"{self.provider}_tokens.json" if not key else f"{self.provider}_tokens_{key}.json"
            self.stores[key] = FakeStore(Path(self._DEFAULT_TOKEN_DIR) / name, key, self.provider)
        return self.stores[key]

    def begin_oauth(self, store=None, account_id=None, timeout=300):
        state = f"STATE{len(self.sessions)}"
        session = FakeLoopbackSession(store, int(account_id or 0), state)
        store.pending = {"state": state}
        self.sessions.append(session)
        return session

    def complete_from_redirect(self, session_or_store, value=None):
        if isinstance(session_or_store, FakeLoopbackSession):
            store, state = session_or_store.store, session_or_store.state
            code = session_or_store.code if value is None else None
        else:
            store = session_or_store
            pending = store.load_pending_oauth()
            if not pending:
                raise RuntimeError("No pending sign-in to complete")
            state = pending["state"]
            code = None
        if value is not None:
            text = str(value)
            if "#" in text:
                code, _, returned = text.partition("#")
                if returned != state:
                    raise RuntimeError("OAuth state mismatch – possible CSRF attack.")
            else:
                code = text
        if not code:
            raise RuntimeError("OAuth login timed out – no callback received.")
        self.exchanges.append(code)
        store.save_tokens({"access_token": f"access-{code}", "refresh_token": "r",
                           "expires_at": time.time() + 7200, "email": f"{code.lower()}@example.com"})
        store.pending = None
        return store.load_tokens()

    # Gemini extras (authgem_auth names)
    def list_gcp_projects(self, token, on_listed=None, http=None):
        """authgem_auth.list_gcp_projects: (billed, unbilled, unknown), None when nothing was listed."""
        self.listed_with = token
        if self.projects is None:
            return None
        lists = ([p for p, s in self.projects if s == "billed"], [p for p, s in self.projects if s == "unbilled"],
                 [p for p, s in self.projects if s not in ("billed", "unbilled")])
        if on_listed is not None:
            on_listed([], [], [p for p, _s in self.projects])
        return lists

    def check_account_status(self, token, account_id=0):
        return dict(self.status_result)


class DeviceSession:
    def __init__(self, account_id: int) -> None:
        self.account_id = account_id
        self.user_code = "WXYZ-1234"
        self.verification_uri = "https://accounts.x.ai/oauth2/device"
        self.verification_uri_complete = "https://accounts.x.ai/oauth2/device?user_code=WXYZ-1234"
        self.interval = 0.01


class FakeDeviceModule(FakeAuthModule):
    """Grok: ``begin_device_login`` / ``poll_device_login`` (RFC 8628) + slot helpers."""

    def __init__(self, token_dir: Path) -> None:
        super().__init__("authgrok", token_dir)
        self.approved = threading.Event()
        self.polls = 0
        self.email = "grok@example.com"

    def begin_device_login(self, store=None, account_id=0, timeout=300):
        return DeviceSession(int(account_id or 0))

    def poll_device_login(self, session, should_stop=None):
        while True:
            if should_stop is not None and should_stop():
                raise RuntimeError("Grok sign-in cancelled")
            self.polls += 1
            if self.approved.wait(session.interval):
                return {"access_token": f"grok-{session.account_id}", "refresh_token": "r", "email": self.email}

    def validate_account_slot_tokens(self, account_id, tokens):
        for slot, store in self.stores.items():
            if slot != account_id and (store.load_tokens() or {}).get("email") == tokens.get("email"):
                raise RuntimeError(f"{tokens.get('email')} is already saved in Grok account slot #{slot}.")

    def get_next_account_id(self, reserved_ids=()):
        used = set(int(r) for r in reserved_ids) | {0}
        return next(i for i in range(1, 100) if i not in used)


class Opener:
    def __init__(self) -> None:
        self.opened: list = []

    def __call__(self, url: str) -> bool:
        self.opened.append(url)
        return True


@pytest.fixture
def fake_auth(tmp_path):
    token_dir = tmp_path / "home" / ".glossarion"
    token_dir.mkdir(parents=True)
    modules = {
        "authgpt": FakeAuthModule("authgpt", token_dir),
        "authcd": FakeAuthModule("authcd", token_dir),
        "authgem": FakeAuthModule("authgem", token_dir),
        "authgrok": FakeDeviceModule(token_dir),
    }
    opener = Opener()
    bridge = OAuthBridge(auth_modules=modules, opener=opener, timeout=5, safe_root=str(tmp_path / "home"))
    return types.SimpleNamespace(modules=modules, opener=opener, bridge=bridge, dir=token_dir, tmp=tmp_path)


# ==========================================================================
# OAuthBridge: providers, flows, slots
# ==========================================================================


def test_provider_table_and_model_routes():
    assert [p for p, _label, _reason in PROVIDERS] == ["authgpt", "authgrok", "authcd", "authgem", "authnan"]
    assert all(reason is None for _p, _l, reason in PROVIDERS)  # every provider signs in from U4
    assert PROVIDER_INFO["authgrok"].flow == "device" and PROVIDER_INFO["authcd"].flow == "loopback"
    assert PROVIDER_INFO["authgpt"].pool_route == "authgpt0/" and PROVIDER_INFO["authgem"].pool_route == "authgem-vertex0/"
    cases = {
        "authgpt/gpt-6-luna": ("authgpt", 0), "authgpt2/o3": ("authgpt", 2), "authgpt0/gpt-6": ("authgpt", 0),
        "authgrok/grok-4": ("authgrok", 0), "authgrok3/grok-4": ("authgrok", 3), "authcd5/claude": ("authcd", 5),
        "authgem/gemini-3": ("authgem", 0), "authgem-vertex2/gemini": ("authgem", 2), "authgem7/x": ("authgem", 7),
        "authgem-key/gemini": None, "gemini/x": None, "authgptx/y": None, "": None, None: None,
    }
    for model, expected in cases.items():
        assert provider_for_model(model) == expected, model
    assert slot_key("authgpt", 0) == "authgpt" and slot_key("authgem", 3) == "authgem3"


def test_sign_in_rules_for_slots_and_pool_routes():
    """The Send gate, the drawer chip and the ModelSheet share one rule: ``authgptN/`` needs slot #N;
    the pool routes (authgpt0/, authgrok0/, authgem-vertex0/) any signed-in slot of the provider."""
    from glossarion_mobile.services.oauth import pool_route_provider, sign_in_satisfied, sign_in_slot

    assert [pool_route_provider(m) for m in ("authgpt0/x", "authgrok0/x", "authgem-vertex0/x", "authgem0/x",
                                             "authgpt/x", "authgpt10/x", "gemini/x")] == \
        ["authgpt", "authgrok", "authgem", None, None, None, None]
    cases = [
        ("authgpt2/gpt-6-luna", {"authgpt2"}, True), ("authgpt2/gpt-6-luna", {"authgpt"}, False),
        ("authgpt/gpt-6-luna", {"authgpt"}, True), ("authgpt/gpt-6-luna", {"authgpt2"}, False),
        ("authgpt0/gpt-6-luna", set(), False), ("authgpt0/gpt-6-luna", {"authgpt3"}, True),
        ("authgpt0/gpt-6-luna", {"authgem"}, False), ("authgem-vertex0/gemini", {"authgem2"}, True),
        ("authgem-vertex/gemini", {"authgem2"}, False), ("authgem-vertex2/gemini", {"authgem2"}, True),
        ("authgrok0/grok-4", {"authgrok5"}, True), ("authcd/claude", {"authcd1"}, False),
        ("gemini-3.5-flash", set(), True), ("authgem-key/gemini", set(), True),
    ]
    for model, signed, expected in cases:
        assert sign_in_satisfied(model, signed) is expected, (model, signed)
    assert [sign_in_slot(m, "authgpt") for m in ("authgpt2/x", "authgpt/x", "authgpt0/x", "gemini/x", "authgrok3/x")] \
        == [2, 0, 0, 0, 0]


def test_loopback_provider_sign_in_paste_and_logout_flag(fake_auth):
    bridge, opener = fake_auth.bridge, fake_auth.opener
    claude = fake_auth.modules["authcd"]
    steps: list = []
    bridge.subscribe(lambda state: steps.append((state.provider, state.step)))

    async def scenario():
        task = asyncio.ensure_future(bridge.sign_in("authcd", 2))
        assert await _wait_for(lambda: opener.opened)
        assert bridge.state.step == "waiting" and bridge.state.provider == "authcd" and bridge.state.account_id == 2
        assert opener.opened[-1] == claude.sessions[-1].auth_url
        claude.sessions[-1].deliver("CLAUDECODE")
        return await asyncio.wait_for(task, TIMEOUT)

    status = _run(scenario())
    assert status["signed_in"] and status["email"] == "claudecode@example.com" and status["account_id"] == 2
    assert claude.get_store(2).cleared_logout_flag == 1  # desktop: an explicit login ends the post-logout block
    assert [s for p, s in steps if p == "authcd"][:3] == ["opening", "waiting", "exchanging"] and steps[-1][1] == "done"
    assert "authcd2" in bridge.signed_in and claude.sessions[-1].closed

    # Paste fallback (the page shows code#state): wrong state, then the right one
    async def paste():
        task = asyncio.ensure_future(bridge.sign_in("authcd", 0))
        assert await _wait_for(lambda: len(opener.opened) == 2)
        state = claude.sessions[-1].state
        with pytest.raises(RuntimeError, match="state mismatch"):
            await bridge.complete_with_paste("PASTED#nope", provider="authcd")
        assert await asyncio.wait_for(task, TIMEOUT) == {}  # the paste claimed this sign-in
        return await bridge.complete_with_paste(f"PASTED#{state}", provider="authcd")

    status = _run(paste())
    assert status["signed_in"] and claude.exchanges[-1] == "PASTED" and bridge.state.step == "done"
    # device providers have nothing to paste
    with pytest.raises(RuntimeError, match="device code"):
        _run(bridge.complete_with_paste("x", provider="authgrok"))


def test_grok_device_code_flow_poll_cancel_and_duplicate_account(fake_auth):
    bridge, opener = fake_auth.bridge, fake_auth.opener
    grok = fake_auth.modules["authgrok"]

    async def approve():
        task = asyncio.ensure_future(bridge.sign_in("authgrok", 0))
        assert await _wait_for(lambda: bridge.state.user_code == "WXYZ-1234")
        assert bridge.state.step == "waiting" and bridge.state.device
        assert opener.opened[-1].endswith("user_code=WXYZ-1234")  # the app opens the verification page itself
        assert bridge.state.verification_uri == "https://accounts.x.ai/oauth2/device"
        assert bridge.reopen_browser() and len(opener.opened) == 2
        await asyncio.sleep(0.05)
        grok.approved.set()
        return await asyncio.wait_for(task, TIMEOUT)

    status = _run(approve())
    assert status["signed_in"] and status["email"] == "grok@example.com" and grok.polls >= 2
    assert grok.get_store(0).load_tokens()["access_token"] == "grok-0"

    # Cancel while polling stops the poll loop
    grok.approved.clear()

    async def cancel():
        task = asyncio.ensure_future(bridge.sign_in("authgrok", 4))
        assert await _wait_for(lambda: bridge.state.step == "waiting")
        bridge.cancel()
        return await asyncio.wait_for(task, TIMEOUT)

    assert _run(cancel()) == {} and bridge.state.step == "cancelled"
    assert not grok.get_store(4).has_tokens

    # The same xAI account in another slot is rejected (desktop validate_account_slot_tokens)
    grok.approved.set()
    with pytest.raises(RuntimeError, match="already saved in Grok account slot #0"):
        _run(bridge.sign_in("authgrok", 5))
    assert bridge.state.step == "error" and not grok.get_store(5).has_tokens


def test_account_slots_from_files_model_pools_and_add(fake_auth):
    bridge, token_dir = fake_auth.bridge, fake_auth.dir
    for name in ("authgpt_tokens.json", "authgpt_tokens_2.json", "authgpt_tokens_2.oauth_pending",
                 "authgpt_tokens_x.json", "authgem_tokens_6.json"):
        (token_dir / name).write_text("{}", encoding="utf-8")
    config = {
        "model": "authgpt3/gpt-6",
        "use_multi_api_keys": True,
        "multi_api_keys": [{"model": "authgpt5/x", "enabled": True}, {"model": "authgpt7/y", "enabled": False},
                           {"model": "authgem-vertex8/g", "enabled": True}],
        "use_fallback_keys": False,
        "fallback_keys": [{"model": "authgpt9/z", "enabled": True}],
    }
    get = lambda key, default=None: config.get(key, default)  # noqa: E731
    assert bridge.account_slots("authgpt", get) == [0, 2, 3, 5]
    assert bridge.account_slots("authgem", get) == [0, 6, 8]
    assert bridge.next_slot("authgpt", [0, 2, 3, 5]) == 6
    bridge.add_slot("authgpt", 6)
    assert bridge.account_slots("authgpt", get) == [0, 2, 3, 5, 6]
    assert bridge.next_slot("authgrok", [0, 1, 2]) == 3  # authgrok asks get_next_account_id, like desktop
    bridge.config_get = get
    assert bridge.account_slots("authcd") == [0]


def test_sign_out_guard_never_deletes_outside_app_data(fake_auth, tmp_path):
    bridge = fake_auth.bridge
    inside = fake_auth.modules["authgpt"].get_store(0)
    inside.save_tokens({"access_token": "a", "email": "me@example.com"})
    outside_dir = tmp_path / "real_user_profile" / ".glossarion"
    outside = FakeStore(outside_dir / "authcd_tokens.json", 0, "authcd")
    outside.save_tokens({"access_token": "desktop-token"})
    fake_auth.modules["authcd"].stores[0] = outside  # e.g. a Windows dev run that resolved '~' to the real profile
    with pytest.raises(RuntimeError, match="outside the app's data folder"):
        _run(bridge.sign_out(0, "authcd"))
    assert outside.load_tokens()["access_token"] == "desktop-token"  # untouched
    _run(bridge.sign_out(0, "authgpt"))
    assert not inside.has_tokens and "authgpt" not in bridge.signed_in
    # sign out everywhere clears every signed-in slot inside the data folder and reports the guard
    fake_auth.modules["authgem"].get_store(3).save_tokens({"access_token": "g3"})
    fake_auth.modules["authgrok"].get_store(0).save_tokens({"access_token": "x0"})
    with pytest.raises(RuntimeError, match="authcd"):
        bridge.sign_out_everywhere()
    assert not fake_auth.modules["authgem"].get_store(3).has_tokens
    assert not fake_auth.modules["authgrok"].get_store(0).has_tokens
    assert outside.has_tokens
    fake_auth.modules["authcd"].stores.pop(0)
    assert bridge.sign_out_everywhere() == []


def test_gemini_projects_status_and_project_cache(fake_auth):
    bridge = fake_auth.bridge
    gem = fake_auth.modules["authgem"]
    gem.get_store(0).save_tokens({"access_token": "tok"})
    assert bridge.gemini_projects(0) == [] and gem.listed_with == "tok"  # nothing listed: no fallback guess
    # the shared authgem_auth.list_gcp_projects; the desktop picker's order: billed, unknown, unbilled
    gem.projects = [("p-unbilled", "unbilled"), ("p-unknown", "unknown"), ("p-billed", "billed")]
    assert bridge.gemini_projects(0) == [("p-billed", "billed"), ("p-unknown", "unknown"), ("p-unbilled", "unbilled")]
    bridge.set_gemini_project("p-billed", 2)
    assert gem._cached_project_id[2] == "p-billed" and gem._project_set_by_gui[2] is True
    assert bridge.gemini_status(0)["sub_label"] == "Pro"
    with pytest.raises(RuntimeError, match="Not logged in"):
        bridge.gemini_status(4)


# ==========================================================================
# The real backend splits through the bridge (network faked, temp token stores)
# ==========================================================================


@pytest.fixture
def token_key():
    pytest.importorskip("cryptography")
    import token_encryption

    token_encryption.set_symmetric_key(os.urandom(32))
    yield
    token_encryption.set_symmetric_key(None)


def _hit(url: str) -> int:
    """GET the loopback callback without following redirects (the success page may redirect to glossarion://)."""
    import http.client
    import urllib.parse

    parts = urllib.parse.urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=TIMEOUT)
    try:
        conn.request("GET", parts.path + ("?" + parts.query if parts.query else ""))
        response = conn.getresponse()
        response.read()
        return response.status
    finally:
        conn.close()


def _real_module(name: str, required: tuple):
    module = pytest.importorskip(name)
    if not all(callable(getattr(module, attr, None)) for attr in required):
        pytest.skip(f"{name} has no {required} yet")
    return module


def _real_namespace(module, store_cls, provider: str, tmp_path: Path, **extra):
    stores: dict = {}

    def get_store(account_id=None):
        key = int(account_id or 0)
        if key not in stores:
            name = f"{provider}_tokens.json" if not key else f"{provider}_tokens_{key}.json"
            stores[key] = store_cls(token_file=str(tmp_path / name), account_id=key)
        return stores[key]

    ns = types.SimpleNamespace(get_store=get_store, _DEFAULT_TOKEN_DIR=str(tmp_path), stores=stores, **extra)
    for attr in ("begin_oauth", "complete_from_redirect", "begin_device_login", "poll_device_login",
                 "resume_device_login", "validate_account_slot_tokens"):
        if hasattr(module, attr) and not hasattr(ns, attr):
            setattr(ns, attr, getattr(module, attr))
    return ns


def test_real_authgem_and_authcd_loopback_through_the_bridge(tmp_path, monkeypatch, token_key):
    authgem = _real_module("authgem_auth", ("begin_oauth", "complete_from_redirect"))
    authcd = _real_module("authcd_auth", ("begin_oauth", "complete_from_redirect"))
    # the app's mobile gates (bootstrap env contract): no Claude Code CLI / ~/.claude credential import
    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    monkeypatch.setenv("GLOSSARION_OAUTH_RETURN_URL", "glossarion://app/oauth/return")
    monkeypatch.setattr(authgem, "exchange_code_for_tokens",
                        lambda code, redirect_uri: {"access_token": f"g-{code}", "refresh_token": "r",
                                                    "expires_at": time.time() + 3600})
    monkeypatch.setattr(authgem, "fetch_user_info", lambda token: {"email": "gem@example.com", "name": "Gem"})
    exchanged: list = []

    def claude_exchange(code, verifier, redirect_uri=None, state=None):
        exchanged.append((code, redirect_uri, state))
        return {"access_token": f"c-{code}", "refresh_token": "r", "expires_at": time.time() + 3600,
                "account": {"email_address": "claude@example.com"}}

    monkeypatch.setattr(authcd, "exchange_code_for_tokens", claude_exchange)
    gem_ns = _real_namespace(authgem, authgem.AuthGemTokenStore, "authgem", tmp_path)
    cd_ns = _real_namespace(authcd, authcd.AuthCDTokenStore, "authcd", tmp_path)
    opener = Opener()
    bridge = OAuthBridge(auth_modules={"authgem": gem_ns, "authcd": cd_ns}, opener=opener, timeout=20,
                         safe_root=str(tmp_path))

    async def gemini():
        task = asyncio.ensure_future(bridge.sign_in("authgem", 2))
        assert await _wait_for(lambda: opener.opened)
        session = bridge._session
        assert opener.opened[-1] == session.auth_url and "accounts.google.com" in session.auth_url
        status = await asyncio.to_thread(_hit, f"{session.redirect_uri}?state={session.state}&code=GEMCODE")
        assert status in (200, 302)  # success page (or its redirect to it)
        return await asyncio.wait_for(task, TIMEOUT)

    status = _run(gemini())
    assert status["signed_in"] and status["email"] == "gem@example.com" and status["account_id"] == 2
    assert gem_ns.stores[2].load_tokens()["access_token"] == "g-GEMCODE"
    assert b"g-GEMCODE" not in (tmp_path / "authgem_tokens_2.json").read_bytes()  # encrypted at rest

    async def claude_paste():
        task = asyncio.ensure_future(bridge.sign_in("authcd", 0))
        assert await _wait_for(lambda: len(opener.opened) == 2)
        assert bridge.state.manual_url and bridge.open_manual_page() and opener.opened[-1] == bridge.state.manual_url
        state = bridge._session.state
        status = await bridge.complete_with_paste(f"PASTEDCODE#{state}", provider="authcd")
        assert await asyncio.wait_for(task, TIMEOUT) == {}
        return status

    status = _run(claude_paste())
    assert status["signed_in"] and exchanged[-1][0] == "PASTEDCODE"
    assert exchanged[-1][1] != "" and "localhost" not in exchanged[-1][1]  # the code page's redirect URI was used
    assert cd_ns.stores[0].load_tokens()["_source"] == "glossarion_oauth"
    assert sorted(bridge.signed_in) == ["authcd", "authgem2"]
    _run(bridge.sign_out(2, "authgem"))
    assert not (tmp_path / "authgem_tokens_2.json").exists()


@needs_flet
def test_real_authgem_paste_after_the_app_was_killed(tmp_path, monkeypatch, token_key):
    """Browser opened, then the app (and its loopback listener) was killed. Reopening the slot's
    LoginSheet must not start over (that replaces the saved PKCE verifier / state): the panel offers
    the paste, and the redirect the browser showed for the first sign-in still finishes it."""
    from glossarion_mobile.ui.screens.accounts import PENDING_TEXT, LoginPanel

    authgem = _real_module("authgem_auth", ("begin_oauth", "complete_from_redirect"))
    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    monkeypatch.setattr(authgem, "exchange_code_for_tokens",
                        lambda code, redirect_uri: {"access_token": f"g-{code}", "refresh_token": "r",
                                                    "expires_at": time.time() + 3600})
    monkeypatch.setattr(authgem, "fetch_user_info", lambda token: {"email": "gem@example.com"})
    opener = Opener()

    def bridge():  # a fresh process: new bridge, new store objects over the same token folder
        ns = _real_namespace(authgem, authgem.AuthGemTokenStore, "authgem", tmp_path)
        return OAuthBridge(auth_modules={"authgem": ns}, opener=opener, timeout=20, safe_root=str(tmp_path)), ns

    first, _ns = bridge()

    async def started():
        task = asyncio.ensure_future(first.sign_in("authgem", 0))
        assert await _wait_for(lambda: opener.opened)
        session = first._session
        redirect = f"{session.redirect_uri}?state={session.state}&code=OLDCODE"
        first.cancel()  # stands in for the kill: listener gone, the pending sign-in stays saved
        assert await asyncio.wait_for(task, TIMEOUT) == {}
        return redirect

    redirect = _run(started())
    second, ns = bridge()
    assert second.has_pending(0, "authgem") and not second.has_pending(1, "authgem")

    async def reopened():
        panel = LoginPanel(second, provider="authgem", account_id=0, autostart=True)
        assert await panel.auto_start() is None  # did_mount's step: no new sign-in
        assert len(opener.opened) == 1 and panel.pending and panel.step_text.value == PENDING_TEXT
        assert panel.paste_field.visible and panel.finish_button.visible and panel.start_button.content == "Start over"
        return await panel.finish_paste(redirect)

    status = _run(reopened())
    assert status["signed_in"] and status["email"] == "gem@example.com"
    assert ns.stores[0].load_tokens()["access_token"] == "g-OLDCODE"
    assert not second.has_pending(0, "authgem")  # finished: the next sheet signs in at once

    async def fresh_start():
        panel = LoginPanel(second, provider="authgem", account_id=0, autostart=True)
        task = asyncio.ensure_future(panel.auto_start())
        assert await _wait_for(lambda: len(opener.opened) == 2)
        second.cancel()
        await asyncio.wait_for(task, TIMEOUT)

    _run(fresh_start())


def test_real_authgrok_device_login_through_the_bridge(tmp_path, monkeypatch, token_key):
    grok = _real_module("authgrok_auth", ("begin_device_login", "poll_device_login"))
    monkeypatch.setenv("GLOSSARION_MOBILE", "1")  # the app's mobile gates (bootstrap env contract)
    approved = threading.Event()
    monkeypatch.setattr(grok, "_DEFAULT_TOKEN_DIR", str(tmp_path))
    monkeypatch.setattr(grok, "_load_oidc_discovery", lambda *a, **k: {})
    monkeypatch.setattr(grok, "_warm_jwks_cache", lambda discovery: None)
    monkeypatch.setattr(grok, "load_grok_cli_credentials", lambda *a, **k: None)
    monkeypatch.setattr(grok, "request_device_code", lambda timeout=30: {
        "device_code": "dc-1", "user_code": "ABCD-1234", "verification_uri": "https://accounts.x.ai/oauth2/device",
        "verification_uri_complete": "https://accounts.x.ai/oauth2/device?user_code=ABCD-1234", "interval": 1,
        "expires_in": 600})

    def poll(device_code, timeout=300, should_stop=None):
        while not approved.wait(0.05):
            if should_stop is not None and should_stop():
                raise RuntimeError("Grok sign-in cancelled")
        return {"access_token": "xai-access", "refresh_token": "xai-refresh", "id_token": "id"}

    monkeypatch.setattr(grok, "poll_device_code_tokens", poll)
    monkeypatch.setattr(grok, "_validate_id_token", lambda token, nonce, discovery: {"email": "x@example.com", "sub": "s1"})
    ns = _real_namespace(grok, grok.AuthGrokTokenStore, "authgrok", tmp_path)
    monkeypatch.setattr(grok, "get_store", ns.get_store)  # validate_account_slot_tokens looks slots up here
    opener = Opener()
    bridge = OAuthBridge(auth_modules={"authgrok": ns}, opener=opener, timeout=20, safe_root=str(tmp_path))

    async def scenario():
        task = asyncio.ensure_future(bridge.sign_in("authgrok", 0))
        assert await _wait_for(lambda: bridge.state.user_code == "ABCD-1234")
        assert "return_to=" in opener.opened[-1]  # signs the xAI website out first, then the device page
        assert bridge.state.verification_uri == "https://accounts.x.ai/oauth2/device"
        approved.set()
        return await asyncio.wait_for(task, TIMEOUT)

    status = _run(scenario())
    assert status["signed_in"] and status["email"] == "x@example.com"
    assert ns.stores[0].load_tokens()["refresh_token"] == "xai-refresh"
    assert not ns.stores[0].load_pending_oauth()  # the saved device code is cleared
    # the same xAI account in slot #1 is rejected by the shared validate_account_slot_tokens
    with pytest.raises(RuntimeError, match="already saved in Grok account slot #0"):
        _run(bridge.sign_in("authgrok", 1))
    assert not ns.get_store(1).has_tokens


# ==========================================================================
# Accounts screen / LoginPanel / Welcome
# ==========================================================================


@needs_flet
def test_accounts_screen_cards_slots_actions_and_project_picker(fake_auth):
    import flet as ft

    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.accounts import (
        EXPERIMENTAL_ACCOUNTS,
        UNAVAILABLE_ACCOUNTS,
        AccountsScreen,
        LoginSheet,
        expiry_label,
        slot_status_line,
    )

    bridge = fake_auth.bridge
    config: dict = {"model": "authgem2/gemini-3"}
    writes: list = []
    signed: list = []
    screen = AccountsScreen(parse_route("/settings/accounts"), oauth=bridge, config_get=lambda k, d=None: config.get(k, d),
                            config_set=lambda values: (writes.append(values), config.update(values)),
                            on_signed_in_changed=signed.append)
    body = screen.get_body()
    keys = [getattr(c, "key", None) for c in body.controls]
    # U12 item 1: the routes that cannot work on mobile are not listed
    assert keys == ["account-authgpt", "account-authgrok", "account-authcd", "account-authgem", "account-authnan",
                    "account-authds", "account-antigravity", "account-zai", "accounts-experimental"]  # U13
    assert body.controls[-1].title == f"Experimental ({len(EXPERIMENTAL_ACCOUNTS)})"
    fake_auth.modules["authgem"].get_store(2).save_tokens({"access_token": "g", "email": "gem@example.com",
                                                          "expires_at": time.time() + 3 * 3600 + 60})
    out = _run(screen.refresh())
    assert set(out) == {"authgpt", "authgrok", "authcd", "authgem", "authnan"} and screen.slots["authgem"] == [0, 2]
    row = screen.slot_rows[("authgem", 2)]
    assert row.subtitle.value == "✓ Signed in · gem@example.com · token refreshes in 3 h"
    assert isinstance(row.trailing, ft.IconButton)
    assert screen.slot_rows[("authgem", 0)].trailing.content == "Sign in"
    actions = screen.slot_actions("authgem", 2)
    assert [item.label for item in actions.items] == ["Re-login", "📊 Status", "Log out"]
    dialog = screen.confirm_sign_out("authgem", 2)
    assert "Currently logged in as: gem@example.com #2" in dialog.title + dialog.dialog.content.content.controls[0].value
    # + Add account takes the next free slot and opens its LoginSheet
    assert _run(screen.add_account("authgpt")) == 1 and isinstance(screen.sheet, LoginSheet)
    assert screen.slots["authgpt"] == [0, 1] and screen.sheet.panel.account_id == 1
    assert screen.sheet.panel.start_button.content == "Sign in with ChatGPT"
    # GCP project picker: the model's signed-in Gemini slot, config authgem_project, module cache
    fake_auth.modules["authgem"].projects = [("p1", "billed"), ("p2", "unbilled")]
    projects = _run(screen.load_projects())
    assert projects == [("p1", "billed"), ("p2", "unbilled")]
    assert [o.text for o in screen.project_dropdown.options] == ["✅ p1", "⚠️ p2 (no billing)"]
    assert screen.project_note.value == "Found 1 GCP project(s) with billing enabled"
    assert screen.select_project("p1") == "p1" and writes[-1] == {"authgem_project": "p1"}
    assert fake_auth.modules["authgem"]._cached_project_id[2] == "p1"
    status_sheet = _run(screen.show_gemini_status(2))
    assert "✅ Gemini #2: Account verified" in status_sheet.body and "gemini-pro: 10%" in status_sheet.body
    fake_auth.modules["authgem"].status_result = {"verified": False, "verification_url": "https://verify.example",
                                                  "verification_message": "Verify your account"}
    status_sheet = _run(screen.show_gemini_status(2))
    assert "⚠️ Gemini #2: Verify your account" in status_sheet.body
    # sign out through the screen
    assert _run(screen.sign_out("authgem", 2)) and signed == [False]
    assert screen.slot_rows[("authgem", 2)].subtitle.value == "Not signed in"
    # helpers
    assert expiry_label(time.time() - 5) == "token expired · refreshes on next use"
    assert expiry_label((time.time() + 600) * 1000).startswith("token refreshes in ")  # milliseconds
    assert slot_status_line({"signed_in": True, "name": "N", "plan": "plus", "source": "claude_code"}) == \
        "✓ Signed in · N · plan plus · from claude_code"


@needs_flet
def test_login_panel_device_code_and_provider_filtering(fake_auth):
    from glossarion_mobile.ui.screens.accounts import LoginPanel

    copied: list = []
    panel = LoginPanel(fake_auth.bridge, provider="authgrok", account_id=0, copy_text=copied.append)
    assert panel.start_button.content == "Sign in with Grok" and not panel.paste_button.visible
    panel.apply(SignInState(provider="authgrok", step="waiting", user_code="AB-12", auth_url="https://x"))
    assert panel.device_box.visible and panel.code_text.value == "AB-12" and not panel.reopen_button.visible
    _run(panel._copy_code())
    assert copied == ["AB-12"]
    # another provider's sign-in does not show in this panel
    panel.apply(SignInState(provider="authcd", step="waiting"))
    assert not panel.device_box.visible and panel.start_button.visible
    claude = LoginPanel(fake_auth.bridge, provider="authcd", account_id=3)
    assert claude.paste_button.visible and claude.paste_field.hint_text.startswith("Paste the redirect URL")
    claude.apply(SignInState(provider="authcd", account_id=1, step="waiting"))
    assert claude.start_button.visible  # slot #1's sign-in is not slot #3's
    # Claude's code page: "Get a code to paste instead" opens it and reveals the paste field
    waiting = SignInState(provider="authcd", account_id=3, step="waiting", manual_url="https://claude.example/code")
    claude.apply(waiting)
    assert claude.manual_button.visible and not claude.paste_field.visible
    fake_auth.bridge.state = waiting
    claude._open_manual()
    assert fake_auth.opener.opened[-1] == "https://claude.example/code" and claude.paste_field.visible


@needs_flet
def test_login_sheets_target_their_own_slot_not_the_last_used_one(fake_auth):
    """Blocked Send, the Welcome panel and the ModelSheet chips open slot #0 (or the model's slot),
    whatever slot the bridge signed in last."""
    from glossarion_mobile.ui.screens.accounts import LoginPanel, LoginSheet

    bridge = fake_auth.bridge
    bridge.state = SignInState(provider="authgpt", step="done", account_id=2, email="two@example.com")
    sheet = LoginSheet(bridge, on_done=lambda status: None)  # no slot given
    assert sheet.panel._target_account() == 0
    assert not sheet.panel.step_text.value.startswith("Done")  # slot #2's sign-in is not this panel's
    bridge.state = SignInState(provider="authgem", step="done", account_id=3)
    assert LoginSheet(bridge, provider="authgem", account_id=0).panel._target_account() == 0
    assert LoginSheet(bridge, provider="authgpt", account_id=2).panel._target_account() == 2
    # a sign-in started from a slot-less panel signs in slot #0
    gpt = fake_auth.modules["authgpt"]
    bridge.state = SignInState()
    panel = LoginPanel(bridge, provider="authgpt")

    async def scenario():
        task = asyncio.ensure_future(panel.run())
        assert await _wait_for(lambda: fake_auth.opener.opened)
        assert bridge.state.account_id == 0
        gpt.sessions[-1].deliver("SLOTZERO")
        return await asyncio.wait_for(task, TIMEOUT)

    status = _run(scenario())
    assert status["account_id"] == 0 and gpt.get_store(0).has_tokens and not gpt.get_store(2).has_tokens


@needs_flet
def test_login_panel_waits_for_another_providers_sign_in(fake_auth):
    from glossarion_mobile.ui.screens.accounts import LoginPanel

    bridge = fake_auth.bridge
    panel = LoginPanel(bridge, provider="authgem", account_id=0)
    panel.apply(SignInState(provider="authgpt", account_id=2, step="waiting"))
    assert panel.start_button.visible and panel.start_button.disabled
    assert panel.step_text.value == "Finish or cancel the ChatGPT #2 sign-in first."
    panel.apply(SignInState(provider="authgpt", account_id=2, step="done"))
    assert not panel.start_button.disabled and panel.step_text.value == ""
    # a start the bridge refuses shows why (the bridge state does not change)
    bridge.state = SignInState(provider="authcd", account_id=1, step="waiting")
    assert _run(panel.run()) is None
    assert panel.error_text.visible and "already in progress" in panel.error_text.value
    # an empty paste is explained too
    bridge.state = SignInState()
    assert _run(panel.finish_paste("  ")) is None and "Paste the redirect URL" in panel.error_text.value


def test_chat_refresh_sign_in_reads_the_models_chatgpt_slot():
    from glossarion_mobile.state.app_state import AppState, ChatContext
    from glossarion_mobile.state.store import LoopGuard
    from glossarion_mobile.ui.chat.integration import ChatFeature

    class Bridge:
        def __init__(self, signed):
            self.signed = set(signed)
            self.reads = []

        async def refresh_status(self, account_id=0, provider="authgpt"):
            self.reads.append(account_id)
            return {"signed_in": account_id in self.signed, "account_id": account_id}

        def statuses(self, provider, slots=None):
            self.reads.append(provider)
            return [{"signed_in": s in self.signed, "account_id": s} for s in (0, 2, 3)]

    async def run_io(fn, *args):
        return fn(*args)

    state = AppState(guard=LoopGuard())
    state.backend.set({"ok": True})
    owner = types.SimpleNamespace(oauth=Bridge({2}), app=types.SimpleNamespace(state=state), _run_io=run_io)
    state.chat_context.set(ChatContext(model="authgpt2/gpt-6-luna"))
    assert _run(ChatFeature.refresh_sign_in(owner))["signed_in"] is False
    assert owner.oauth.reads == [0, 2] and state.signed_in.value == frozenset({"authgpt2"})
    assert state.send_block() is None  # slot #2 signed in: Send is not blocked for authgpt2/
    state.chat_context.set(ChatContext(model="authgpt/gpt-6-luna"))
    assert state.send_block().fix_action == "sign_in_chatgpt"
    owner.oauth = Bridge({3})
    state.chat_context.set(ChatContext(model="authgpt0/gpt-6-luna"))
    _run(ChatFeature.refresh_sign_in(owner))
    assert owner.oauth.reads == [0, "authgpt"] and state.signed_in.value == frozenset({"authgpt3"})
    assert state.send_block() is None  # the pool uses any ChatGPT slot


def test_grok_device_login_resumes_after_the_app_was_killed(fake_auth):
    bridge, grok = fake_auth.bridge, fake_auth.modules["authgrok"]
    resumed: list = []

    def resume_device_login(store=None, account_id=None, timeout=300):
        if not store.pending:
            return None
        resumed.append(account_id)
        session = DeviceSession(int(account_id or 0))
        session.user_code = store.pending["user_code"]
        return session

    grok.resume_device_login = resume_device_login
    grok.get_store(2).pending = {"user_code": "SAVED-CODE"}  # the previous process died while polling
    grok.approved.set()
    grok.email = "slot2@example.com"
    states: list = []
    bridge.subscribe(lambda state: states.append(state.user_code))
    status = _run(bridge.sign_in("authgrok", 2))
    assert resumed == [2] and "SAVED-CODE" in states and status["signed_in"]
    grok.get_store(3).pending = None
    grok.email = "slot3@example.com"
    _run(bridge.sign_in("authgrok", 3))  # nothing saved: a fresh device code
    assert resumed == [2] and "WXYZ-1234" in states


@needs_flet
def test_welcome_step_two_sign_ins_and_local_ai(fake_auth):
    """U11 item 1: step 1 offers every sign-in side by side; after one the provider's models are polled and
    the model becomes its most cost-efficient one (or stays the default while it is still listed)."""
    import asyncio
    import types

    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.accounts import LoginPanel, LoginSheet
    from glossarion_mobile.ui.screens.welcome import SIGN_INS, WelcomeScreen

    routes: list = []
    queries: list = []
    chosen: list = []
    polled: list = []

    class Catalog:
        snapshot = types.SimpleNamespace(models=("authgem/gemini-3.5-flash", "authgem/gemini-3.5-pro",
                                                 "authgem/gemini-3-flash-preview", "authgpt/gpt-5.6-luna"))

        async def refresh(self, provider, explicit=True):
            polled.append(provider)

    screen = WelcomeScreen(parse_route("/welcome"),
                           login_panel_factory=lambda on_done: LoginPanel(fake_auth.bridge, on_done=on_done),
                           navigate=routes.append, on_use_provider=queries.append, catalog=Catalog(),
                           current_model=lambda: "authgpt/gpt-6-luna", on_set_model=chosen.append)
    screen.get_body()
    assert screen.oauth is fake_auth.bridge  # taken from the LoginPanel factory
    keys = [getattr(c, "key", None) for c in screen.page_area.content.controls]
    assert [f"welcome-signin-{p}" for p, _label in SIGN_INS] == [k for k in keys if str(k).startswith("welcome-signin")]
    screen.go("providers")
    keys = [getattr(c, "key", None) for c in screen.page_area.content.controls]
    assert "welcome-local" in keys and not any(str(k).startswith("welcome-signin") for k in keys)
    sheet = screen.open_sign_in("authgem")
    assert isinstance(sheet, LoginSheet) and sheet.panel.provider == "authgem" and sheet.panel.account_id == 0
    screen._go("settings.endpoints")
    assert routes == ["settings.endpoints"]
    screen.go("sign_in")
    screen.provider_status["authgem"] = {"email": "gem@example.com"}
    screen.flow.signed_in = True
    assert asyncio.run(screen.check_model("authgem")) == "authgem/gemini-3.5-flash"
    assert polled == ["authgem"] and chosen == ["authgem/gemini-3.5-flash"] and queries == []
    rows = {getattr(c, "key", None): c for c in screen.page_area.content.controls}
    signed = rows["welcome-signed-authgem"]
    texts = [getattr(c, "value", "") for c in signed.controls]
    assert any("gemini-3.5-flash" in str(v) for v in texts), texts
    use = next(c for c in signed.controls[0].controls if getattr(c, "key", None) == "welcome-use-authgem")
    use.on_click(None)
    assert queries == ["authgem"] and use.content == "Change model"
    # ChatGPT: the default GPT-6 Luna is gone from the catalog -> the newest Luna, said so
    assert asyncio.run(screen.check_model("authgpt")) == "authgpt/gpt-5.6-luna"
    assert "no longer offered" in screen.model_notes["authgpt"]
    # without a bridge no sign-in buttons are offered
    bare = WelcomeScreen(parse_route("/welcome"))
    bare.get_body()
    bare.go("providers")
    assert not any(str(getattr(c, "key", "")).startswith("welcome-signin") for c in bare.page_area.content.controls)


# ==========================================================================
# Profiles: fake prompt_profiles core with the desktop semantics
# ==========================================================================


DEFAULTS = {
    "Universal": "You MUST translate to {target_lang}.\n{split_marker_instruction}\n",
    "Korean_BeautifulSoup": "Korean BS prompt",
    "Korean_html2text": "Korean h2t prompt",
}


class FakeProfileState:
    def __init__(self, config: dict) -> None:
        self.default_prompts = dict(DEFAULTS)
        self.protected = set(DEFAULTS)
        self.prompt_profiles = dict(config.get("prompt_profiles", dict(DEFAULTS)))
        active = config.get("active_profile", next(iter(self.prompt_profiles)))
        self.profile_var = active if active in self.prompt_profiles else next(iter(self.prompt_profiles))
        self._original_profile_content: dict = {}


def _fake_profiles_core() -> types.ModuleType:
    """The U4 ``prompt_profiles`` contract with other_settings / translator_gui semantics (test double)."""
    m = types.ModuleType("prompt_profiles_fake")

    def profile_state_from_config(config):
        return FakeProfileState(config)

    def select_profile(state, name, config=None):  # other_settings.on_profile_select
        if name in state.prompt_profiles:
            state.profile_var = name
            config["active_profile"] = name
            lowered = name.lower()
            if "beautifulsoup" in lowered:
                config["text_extraction_method"] = "standard"
            elif "html2text" in lowered:
                config["text_extraction_method"] = "enhanced"
            return state.prompt_profiles[name]
        return None

    def save_profile(state, name, content, source_name=None, config=None):  # other_settings.save_profile
        source = source_name or state.profile_var
        if name in state.prompt_profiles and name != source:
            return "A profile with this name already exists. Choose another name."
        renaming = source in state.prompt_profiles and name != source and source not in state.protected
        content = content.strip()
        if renaming:
            state.prompt_profiles = {(name if k == source else k): (content if k == source else v)
                                     for k, v in state.prompt_profiles.items()}
        else:
            state.prompt_profiles[name] = content
        state.profile_var = name
        return None

    def new_profile(state, config=None):  # translator_gui._quick_new_profile
        n = 1
        while f"New Profile #{n}" in state.prompt_profiles:
            n += 1
        name = f"New Profile #{n}"
        state.prompt_profiles[name] = ""
        state.profile_var = name
        return name

    def delete_or_reset_profile(state, name, config=None):  # other_settings.delete_profile
        if name not in state.prompt_profiles:
            return f"Profile '{name}' not found."
        if name in state.protected:
            state.prompt_profiles[name] = state.default_prompts[name]
            return "reset"
        del state.prompt_profiles[name]
        state.profile_var = next(iter(state.prompt_profiles), "")
        return "deleted"

    def merge_imported_profiles(state, data, config=None):  # other_settings.import_profiles
        state.prompt_profiles.update(data)

    def export_profiles_json(profiles):  # other_settings.export_profiles
        return json.dumps(profiles, ensure_ascii=False, indent=2)

    def apply_profiles_to_config(state, config):  # other_settings.save_profiles key set
        config["prompt_profiles"] = state.prompt_profiles
        config["profile_name_autofill"] = bool(config.get("profile_name_autofill", True))
        config["profile_mousewheel_locked"] = bool(config.get("profile_mousewheel_locked", True))
        config["active_profile"] = state.profile_var

    # assistant prefill (translator_gui.show_assistant_prompt_dialog)
    def prefill_state_from_config(config):
        stored = config.get("assistant_prompt_profiles", {})
        profiles = {k: v for k, v in stored.items() if isinstance(k, str) and k.strip()
                    and k.strip().casefold() != "default" and isinstance(v, str)} if isinstance(stored, dict) else {}
        active = config.get("active_assistant_prompt_profile", "")
        return types.SimpleNamespace(
            profiles=profiles,
            default_prompt=str(config.get("assistant_prompt_profile_default", config.get("assistant_prompt", "")) or ""),
            active_name=active if isinstance(active, str) and active in profiles else "",
        )

    def prefill_select(state, name):
        if name.strip().casefold() == "default":
            state.active_name = ""
        elif name in state.profiles:
            state.active_name = name

    def prefill_save(state, name, text):
        name, text = name.strip(), text.strip()
        if name.casefold() == "default" and state.active_name:
            return "Default is reserved. Choose another profile name."
        if name in state.profiles and name != state.active_name:
            return "A profile with this name already exists. Choose another name."
        if name.casefold() == "default":
            state.default_prompt = text
            state.active_name = ""
        else:
            if state.active_name in state.profiles and name != state.active_name:
                state.profiles = {(name if k == state.active_name else k): (text if k == state.active_name else v)
                                  for k, v in state.profiles.items()}
            else:
                state.profiles[name] = text
            state.active_name = name

    def prefill_new(state):
        n = 1
        while f"New Profile #{n}" in state.profiles:
            n += 1
        state.active_name = f"New Profile #{n}"
        state.profiles[state.active_name] = ""
        return state.active_name

    def prefill_delete(state, name):
        del state.profiles[name]
        state.active_name = next(iter(state.profiles), "")

    def prefill_apply_to_config(state, config, prompt_text=""):
        config.update({
            "assistant_prompt_profiles": dict(state.profiles),
            "assistant_prompt_profile_default": state.default_prompt,
            "active_assistant_prompt_profile": state.active_name,
            "assistant_prompt": str(prompt_text or "").strip(),
        })

    for fn in (profile_state_from_config, select_profile, save_profile, new_profile, delete_or_reset_profile,
               merge_imported_profiles, export_profiles_json, apply_profiles_to_config, prefill_state_from_config,
               prefill_select, prefill_save, prefill_new, prefill_delete, prefill_apply_to_config):
        setattr(m, fn.__name__, fn)
    return m


@pytest.fixture
def test_key():
    fernet = pytest.importorskip("cryptography.fernet")
    import api_key_encryption

    key = fernet.Fernet.generate_key()
    api_key_encryption.set_key_material(key)
    yield key
    api_key_encryption.set_key_material(None)


def _store(path: Path, config: Optional[dict] = None):
    from glossarion_mobile.state.config_store import MobileConfigStore

    if config is not None:
        import config_store

        config_store.save_config_file(dict(config), str(path), backup=False)
    store = MobileConfigStore(str(path), debounce=30)
    store.load()
    return store


class Ctx:
    """The SettingsContext surface the pages use."""

    def __init__(self, store=None, prefs=None, schema=None) -> None:
        self.store = store
        self.prefs = prefs
        self.schema = schema
        self.page = None
        self.tablet = False
        self.messages: list = []
        self.routes: list = []
        self.opened: list = []

    def say(self, message, *_a):
        self.messages.append(message)

    def push(self, *controls):
        pass

    def on_ui(self, fn, *args):
        fn(*args)

    async def run_io(self, fn, *args):
        return fn(*args)

    def spawn(self, coro):
        return asyncio.ensure_future(coro)

    def go(self, name, params=None, *, fragment=None):
        self.routes.append((name, params))
        return name

    def open_setting(self, section_id, key=None):
        self.opened.append((section_id, key))
        return section_id


@needs_crypto
def test_profile_service_crud_writes_only_the_desktop_key_set(tmp_path, test_key):
    from glossarion_mobile.ui.screens.profiles import PROFILE_CONFIG_KEYS, ProfilesCore, ProfileService, profile_id

    path = tmp_path / "config.json"
    store = _store(path, {"model": "authgpt/gpt-6-luna", "api_key": "sk-secret-value-123456",
                          "prompt_profiles": dict(DEFAULTS), "active_profile": "Universal", "temperature": 0.3})
    raw_before = json.loads(path.read_text(encoding="utf-8"))
    service = ProfileService(store, core=ProfilesCore(_fake_profiles_core()))
    listing = service.listing()
    assert listing.names == list(DEFAULTS) and listing.active == "Universal" and listing.is_builtin("Universal")
    assert listing.name_for(profile_id("Korean_html2text")) == "Korean_html2text"

    name = service.new()
    assert name == "New Profile #1" and store.get("active_profile") == "New Profile #1"
    assert store.get("prompt_profiles")["New Profile #1"] == ""
    assert store.get("profile_name_autofill") is True and store.get("profile_mousewheel_locked") is True
    # rename a custom profile in place (key order kept), then copy a built-in under a new name
    assert service.save("New Profile #1", "Mine", "  my prompt {target_lang}  ") == "Mine"
    assert list(store.get("prompt_profiles")) == [*DEFAULTS, "Mine"] and store.get("prompt_profiles")["Mine"] == "my prompt {target_lang}"
    assert service.save("Universal", "Universal copy", "edited universal") == "Universal copy"
    assert store.get("prompt_profiles")["Universal"] == DEFAULTS["Universal"]  # built-in kept, copy made
    with pytest.raises(ValueError, match="already exists"):
        service.save("Mine", "Universal", "x")
    assert service.duplicate("Mine") == "Mine (copy)" and service.duplicate("Mine") == "Mine (copy 2)"
    # selecting an extraction profile switches text_extraction_method like desktop on_profile_select
    service.select("Korean_BeautifulSoup")
    assert store.get("active_profile") == "Korean_BeautifulSoup" and store.get("text_extraction_method") == "standard"
    # built-ins reset, custom profiles delete
    store.set_many({"prompt_profiles": {**store.get("prompt_profiles"), "Korean_html2text": "changed"}})
    assert service.listing().is_modified("Korean_html2text")
    assert service.delete_or_reset("Korean_html2text") == "reset"
    assert store.get("prompt_profiles")["Korean_html2text"] == DEFAULTS["Korean_html2text"]
    assert service.delete_or_reset("Mine (copy 2)") == "deleted" and "Mine (copy 2)" not in store.get("prompt_profiles")
    # import / export in the desktop JSON format
    assert service.import_json(json.dumps({"Imported": "imp"})) == 1 and store.get("prompt_profiles")["Imported"] == "imp"
    with pytest.raises(ValueError, match="Failed to import profiles"):
        service.import_json("[1, 2]")
    exported = json.loads(service.export_json())
    assert exported == store.get("prompt_profiles")
    # role toggle = desktop system_prompt_to_user
    service.set_role_user(True)
    assert store.get("system_prompt_to_user") is True and service.role_is_user()

    store.flush()
    raw_after = json.loads(path.read_text(encoding="utf-8"))
    changed = {k for k in set(raw_before) | set(raw_after) if raw_before.get(k) != raw_after.get(k)}
    assert changed <= set(PROFILE_CONFIG_KEYS) | {"system_prompt_to_user"}
    assert raw_after["api_key"] == raw_before["api_key"]  # untouched secret written back byte-for-byte (ENC:)
    assert raw_after["temperature"] == 0.3 and raw_after["model"] == "authgpt/gpt-6-luna"
    store.close()


@needs_crypto
def test_profiles_core_adapter_follows_other_core_signatures(tmp_path, test_key):
    """``bind_call`` binds by parameter name: a dict-in/dict-out merge and a core that only has the
    desktop file writer (``write_profiles_to_config_file``) work without adapter changes."""
    from glossarion_mobile.ui.screens.profiles import ProfilesCore, ProfileService, ProfilesUnavailable, bind_call

    base = _fake_profiles_core()
    variant = types.ModuleType("prompt_profiles_variant")
    for name in ("profile_state_from_config", "select_profile", "save_profile", "new_profile", "delete_or_reset_profile"):
        setattr(variant, name, getattr(base, name))

    def merge_imported_profiles(prompt_profiles, data):  # returns the merged dict (other_settings.import_profiles)
        merged = dict(prompt_profiles)
        merged.update(data)
        return merged

    def export_profiles_json(prompt_profiles):
        return json.dumps(prompt_profiles, ensure_ascii=False, indent=2)

    def write_profiles_to_config_file(config_file, prompt_profiles, active_profile, config=None):  # save_profiles
        with open(config_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        data["prompt_profiles"] = prompt_profiles
        data["profile_name_autofill"] = bool((config or {}).get("profile_name_autofill", True))
        data["profile_mousewheel_locked"] = bool((config or {}).get("profile_mousewheel_locked", True))
        data["active_profile"] = active_profile
        with open(config_file, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        return True

    for fn in (merge_imported_profiles, export_profiles_json, write_profiles_to_config_file):
        setattr(variant, fn.__name__, fn)
    store = _store(tmp_path / "config.json", {"prompt_profiles": dict(DEFAULTS), "active_profile": "Universal",
                                              "profile_name_autofill": False})
    service = ProfileService(store, core=ProfilesCore(variant))
    assert service.import_json(json.dumps({"Imported": "x"})) == 1
    assert store.get("prompt_profiles")["Imported"] == "x" and store.get("profile_name_autofill") is False
    assert store.get("profile_mousewheel_locked") is True and store.get("active_profile") == "Universal"
    assert service.new() == "New Profile #1" and store.get("active_profile") == "New Profile #1"
    assert json.loads(service.export_json())["New Profile #1"] == ""
    with pytest.raises(ProfilesUnavailable, match="needs 'mystery'"):
        bind_call(lambda state, mystery: None, {"state": 1})
    assert bind_call(lambda state, extra=5: (state, extra), {"state": 1}) == (1, 5)
    store.close()


def test_profiles_without_the_core_are_read_only():
    from glossarion_mobile.ui.screens.profiles import ProfilesCore, ProfileService, ProfilesUnavailable, extraction_note

    class Store:
        def snapshot(self):
            return {"prompt_profiles": {"A": "a", "B": "b"}, "active_profile": "B"}

    core = ProfilesCore()
    core._tried, core._module, core.error = True, None, "ModuleNotFoundError"
    service = ProfileService(Store(), core=core)
    listing = service.listing()
    assert listing.names == ["A", "B"] and listing.active == "B" and not service.available
    with pytest.raises(ProfilesUnavailable):
        service.new()
    # a core whose desktop start-up cannot import the backend (missing packages) degrades the same way
    broken = types.SimpleNamespace(profile_state_from_config=lambda config: __import__("no_such_backend_package"))
    service = ProfileService(Store(), core=ProfilesCore(broken))
    assert service.listing().names == ["A", "B"] and "ModuleNotFoundError" in str(service.error)
    with pytest.raises(ProfilesUnavailable):
        service.new()
    assert extraction_note("Japanese_BeautifulSoup").startswith("Using this profile switches text extraction to Beau")
    assert extraction_note("Chinese_html2text").endswith("(enhanced).") and extraction_note("Universal") is None


@needs_flet
@needs_crypto
def test_profiles_screens_flow(tmp_path, test_key):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.profiles import (
        ProfileDetailScreen,
        ProfilesCore,
        ProfileService,
        ProfilesScreen,
        profile_id,
    )

    store = _store(tmp_path / "config.json", {"prompt_profiles": dict(DEFAULTS), "active_profile": "Universal"})
    service = ProfileService(store, core=ProfilesCore(_fake_profiles_core()))
    ctx = Ctx(store)
    shared: list = []
    screen = ProfilesScreen(parse_route("/settings/profiles"), ctx, service=service, share_file=shared.append,
                            temp_dir=str(tmp_path / "tmp"))
    screen.get_body()
    assert list(screen.rows) == list(DEFAULTS)
    assert screen.rows["Universal"].key == f"profile-{profile_id('Universal')}"
    name = screen.new_profile()
    assert name == "New Profile #1" and ctx.routes[-1] == ("settings.profiles.detail", {"pid": profile_id(name)})
    sheet = screen.row_actions("Universal")
    assert [i.label for i in sheet.items] == ["Use this profile", "Duplicate", "Reset to default"]
    assert [i.label for i in screen.row_actions(name).items][-1] == "Delete"
    assert screen.duplicate("Universal") == "Universal (copy)"
    dialog = screen.confirm_delete("Universal")
    assert dialog.title == "Reset Profile"
    # export writes the desktop JSON and shares it; import merges a file
    path = _run(screen.export_profiles())
    assert shared == [path] and json.loads(Path(path).read_text(encoding="utf-8")) == store.get("prompt_profiles")
    source = tmp_path / "import.json"
    source.write_text(json.dumps({"From desktop": "hello"}), encoding="utf-8")
    assert _run(screen.import_profiles(str(source))) == 1 and "From desktop" in screen.rows
    screen.show_segment("all")
    assert screen.content.content is screen.all_prompts and not screen.fab.visible
    screen.show_segment("profiles")

    detail = ProfileDetailScreen(parse_route(f"/settings/profiles/{profile_id('New Profile #1')}"), ctx, service=service)
    detail.get_body()
    assert detail.name == "New Profile #1" and detail.delete_button.content == "Delete"
    detail.name_field.value = "Renamed"
    detail.editor.set_value("Renamed prompt", initial=False)
    assert detail.save() == "Renamed" and ctx.routes[-1] == ("settings.profiles.detail", {"pid": profile_id("Renamed")})
    assert store.get("prompt_profiles")["Renamed"] == "Renamed prompt" and "New Profile #1" not in store.get("prompt_profiles")
    assert detail.save_as("Renamed 2") == "Renamed 2"
    detail.error_text.visible = False
    assert detail.save_as("Universal") is None and "already exists" in detail.error_text.value
    builtin = ProfileDetailScreen(parse_route(f"/settings/profiles/{profile_id('Korean_html2text')}"), ctx, service=service)
    builtin.get_body()
    assert builtin.delete_button.content == "Reset to default"
    keys = [getattr(c, "key", None) for c in builtin.body.content.controls]
    assert "profile-extraction-note" in keys
    assert builtin.use() and store.get("text_extraction_method") == "enhanced"
    missing = ProfileDetailScreen(parse_route("/settings/profiles/0123456789ab"), ctx, service=service)
    assert missing.get_body().content.key == "profile-missing"
    store.close()


@needs_flet
@needs_crypto
def test_prefill_profiles_flow(tmp_path, test_key):
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.prefill_profiles import PREFILL_CONFIG_KEYS, PrefillCore, PrefillScreen, PrefillService

    store = _store(tmp_path / "config.json", {"assistant_prompt": "legacy prefill", "temperature": 0.5})
    service = PrefillService(store, core=PrefillCore(_fake_profiles_core()))
    listing = service.listing()
    assert listing.options == ["Default"] and listing.default_text == "legacy prefill"  # seeded from the old prefill
    ctx = Ctx(store)
    screen = PrefillScreen(parse_route("/settings/prefill"), ctx, service=service)
    screen.get_body()
    assert screen.editor.value == "legacy prefill" and screen.status_text.value.startswith("✓ Prefill is active")
    assert screen.new() == "New Profile #1"
    assert store.get("active_assistant_prompt_profile") == "New Profile #1" and store.get("assistant_prompt") == ""
    screen.name_field.value = "Formal"
    screen.editor.set_value("  Reply formally.  ", initial=False)
    assert screen.save() == "Formal"
    assert store.get("assistant_prompt_profiles") == {"Formal": "Reply formally."}
    assert store.get("assistant_prompt") == "Reply formally." and store.get("active_assistant_prompt_profile") == "Formal"
    screen.select("Default")
    assert store.get("assistant_prompt") == "legacy prefill" and store.get("active_assistant_prompt_profile") == ""
    screen.name_field.value = "Default"
    with pytest.raises(ValueError, match="cannot be deleted"):
        service.delete("Default")
    assert screen.delete("Formal") == "Formal" and store.get("assistant_prompt_profiles") == {}
    assert set(PREFILL_CONFIG_KEYS) == {"assistant_prompt_profiles", "assistant_prompt_profile_default",
                                       "active_assistant_prompt_profile", "assistant_prompt"}
    store.close()


def test_real_prompt_profiles_core_matches_the_desktop_file_shape(tmp_path):
    """Once the shared core lands: saving profiles through MobileConfigStore gives config.json the same
    profile keys and values as the core's desktop file writer (other_settings.save_profiles) - for a
    built-in saved under a new name (copy), a custom rename, a new profile and a delete."""
    core_mod = pytest.importorskip("prompt_profiles")
    writer = getattr(core_mod, "write_profiles_to_config_file", None)
    if not callable(writer):
        pytest.skip("prompt_profiles has no write_profiles_to_config_file yet")
    pytest.importorskip("cryptography")
    import inspect

    import api_key_encryption
    from cryptography.fernet import Fernet

    from glossarion_mobile.ui.screens.profiles import (
        _ALIASES,
        PROFILE_CONFIG_KEYS,
        ProfilesCore,
        ProfileService,
        ProfilesUnavailable,
        bind_call,
    )

    core = ProfilesCore(core_mod)
    try:
        core.state({})
    except ProfilesUnavailable as exc:
        pytest.skip(f"prompt_profiles state factory: {exc}")
    api_key_encryption.set_key_material(Fernet.generate_key())
    try:
        start = {"prompt_profiles": {"Universal": "u", "Mine": "m"}, "active_profile": "Universal", "temperature": 0.1}
        mobile_path = tmp_path / "mobile" / "config.json"
        mobile_path.parent.mkdir()
        store = _store(mobile_path, start)
        service = ProfileService(store, core=core)
        desktop_path = tmp_path / "desktop" / "config.json"
        desktop_path.parent.mkdir()
        desktop_path.write_text(json.dumps(start, ensure_ascii=False, indent=2), encoding="utf-8")
        desktop_config = json.loads(json.dumps(start))
        state = core.state(desktop_config)
        path_name = next(n for n in inspect.signature(writer).parameters
                         if n in ("config_file", "path", "config_path", "file_path", "filename"))

        def desktop_write() -> dict:

            bind_call(writer, {"state": state, "profiles": core._profiles_of(state), "active": core._active_of(state),
                               "config": desktop_config, "path": str(desktop_path)}, {**_ALIASES, "path": (path_name,)})
            return json.loads(desktop_path.read_text(encoding="utf-8"))

        steps = [
            (lambda: service.save("Universal", "Universal copy", "copy text"),
             lambda: core.save(state, "Universal", "Universal copy", "copy text", desktop_config)),
            (lambda: service.save("Mine", "Mine renamed", "m2"),
             lambda: core.save(state, "Mine", "Mine renamed", "m2", desktop_config)),
            (lambda: service.new(), lambda: core.new(state, desktop_config)),
            (lambda: service.delete_or_reset("Mine renamed"),
             lambda: core.delete_or_reset(state, "Mine renamed", desktop_config)),
        ]
        for mobile_step, desktop_step in steps:
            mobile_step()
            store.flush()
            desktop_step()
            desktop = desktop_write()
            mobile = json.loads(mobile_path.read_text(encoding="utf-8"))
            for key in PROFILE_CONFIG_KEYS:
                if key == "text_extraction_method":
                    continue
                assert mobile.get(key) == desktop.get(key), key
            assert list(mobile["prompt_profiles"]) == list(desktop["prompt_profiles"])  # order kept
            assert mobile["temperature"] == 0.1
        store.close()
    finally:
        api_key_encryption.set_key_material(None)


# ==========================================================================
# All prompts / PromptEditor
# ==========================================================================


def test_all_prompts_index_from_the_schema():
    pytest.importorskip("settings_schema")
    from glossarion_mobile.ui.screens.all_prompts import filter_entries, prompt_index
    from glossarion_mobile.ui.settings.schema_access import SchemaAccess

    entries = prompt_index(SchemaAccess())
    keys = {entry.key for entry in entries}
    for key in ("translation_chunk_prompt", "image_chunk_prompt", "vision_ocr_prompt", "refinement_system_prompt",
                "book_title_prompt", "review_system_prompt", "glossary_refinement_system_prompt",
                "parallel_epub_glossary_wrapper_prompt", "manga_ocr_prompt", "gtool_scan_user_prompt"):
        assert key in keys, key
    assert len(keys) == len(entries) >= 30
    vision = filter_entries(entries, "vision ocr")
    assert vision and all("vision" in (e.key + e.label + e.section_title).lower() for e in vision)
    assert filter_entries(entries, "") == entries


@needs_flet
def test_all_prompts_view_opens_the_setting():
    from glossarion_mobile.ui.screens.all_prompts import AllPromptsView, PromptEntry

    entries = [PromptEntry("a_prompt", "A prompt", "sec.a", "Section A", "Translation"),
               PromptEntry("b_prompt", "B prompt", "sec.b", "Section B", "Glossary", unavailable="Not on mobile")]
    ctx = Ctx()
    view = AllPromptsView(ctx, entries=entries)
    assert list(view.tiles) == ["a_prompt", "b_prompt"] and view.count_text.value == "2 prompts"
    view.set_query("glossary")
    assert list(view.tiles) == ["b_prompt"]
    view.open(entries[0])
    assert ctx.opened == [("sec.a", "a_prompt")]


@needs_flet
def test_prompt_editor_placeholders_and_token_count():
    from glossarion_mobile.ui.screens.prompt_editor import PromptEditorPane, TokenCounter, insert_placeholder, token_label

    assert insert_placeholder("abc", "{x}") == ("abc\n{x}", 7)
    assert insert_placeholder("abc ", "{x}") == ("abc {x}", 7)
    assert insert_placeholder("", "{x}") == ("{x}", 3)
    assert insert_placeholder("hello world", "{x}", 6, 11) == ("hello {x}", 9)
    assert insert_placeholder("hello", "{x}", 2) == ("he{x}llo", 5)
    assert token_label("abcd", 3) == "4 chars · 3 tokens" and token_label("abcd", None).startswith("4 chars · ≈ ")
    pane = PromptEditorPane(value="Translate", counter=lambda text, model: len(text.split()))
    assert [chip.key for chip in pane.chips] == ["placeholder-target_lang", "placeholder-split_marker_instruction"]
    import flet as ft

    pane.field.selection = ft.TextSelection(base_offset=0, extent_offset=0)
    assert pane.insert("{target_lang}") == "{target_lang}Translate" and pane.dirty
    labels: list = []
    counter = TokenCounter(counter=lambda text, model: 42, on_count=labels.append, delay=0.01, model=lambda: "gpt-4")

    async def count():
        counter.schedule("some text")
        await asyncio.sleep(0.1)

    _run(count())
    assert labels == ["9 chars · 42 tokens"] and counter.last == 42


# ==========================================================================
# Appearance / Storage / Backup / Import / About / Danger zone
# ==========================================================================


class FakePrefs(dict):
    def set(self, key, value):
        self[key] = value


@needs_flet
def test_appearance_prefs_apply_theme_scale_and_haptics():
    import flet as ft

    from glossarion_mobile.ui import tokens
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.appearance import AppearanceScreen, apply_appearance, normalize_appearance

    assert normalize_appearance({"theme": "AMOLED", "accent": "nope", "text_scale": 9}) == {
        "theme": "amoled", "accent": "halgakos_rose", "text_scale": 1.3, "reduce_motion": False, "haptics": True}
    page = types.SimpleNamespace(theme=None, dark_theme=None, theme_mode=None, update=lambda: None)
    state = types.SimpleNamespace(text_scale=types.SimpleNamespace(value=1.0, set=lambda v: setattr(state.text_scale, "value", v)))
    haptics = types.SimpleNamespace(enabled=True)
    prefs = FakePrefs()
    ctx = Ctx(prefs=prefs)
    ctx.page = page
    screen = AppearanceScreen(parse_route("/settings/appearance"), ctx, state=state, haptics=haptics)
    screen.get_body()
    screen.set_value("theme", "dark")
    screen.set_value("accent", "desktop_blue")
    screen.set_value("text_scale", 1.15)
    screen.set_value("haptics", False)
    screen.set_value("reduce_motion", True)
    assert prefs["appearance"] == {"theme": "dark", "accent": "desktop_blue", "text_scale": 1.15, "reduce_motion": True,
                                   "haptics": False}
    assert page.theme_mode == ft.ThemeMode.DARK and page.theme.color_scheme_seed == tokens.ACCENTS["desktop_blue"]
    assert page.theme.page_transitions.android == ft.PageTransitionTheme.NONE
    assert state.text_scale.value == 1.15 and haptics.enabled is False
    assert screen.accent_chips["desktop_blue"].selected and not screen.accent_chips["halgakos_rose"].selected
    assert apply_appearance(None, {"theme": "light"})["theme"] == "light"


def test_storage_usage_clear_and_mirror(tmp_path):
    from glossarion_mobile.ui.screens.storage import clear_folder, folder_usage, mirror_output, storage_folders

    paths = types.SimpleNamespace(data=tmp_path / "data", output=tmp_path / "data" / "Output", library=tmp_path / "lib",
                                  cache=tmp_path / "cache", temp=tmp_path / "temp", logs=tmp_path / "data" / "logs")
    folders = storage_folders(paths)
    assert [f.id for f in folders] == ["data", "output", "library", "inbox", "cache", "temp", "logs", "payloads",
                                       "http_requests"]  # U9: the Logs & diagnostics dumps
    assert [f.id for f in folders if f.clearable] == ["cache", "temp", "payloads", "http_requests"]
    (tmp_path / "cache" / "tiktoken").mkdir(parents=True)
    (tmp_path / "cache" / "tiktoken" / "enc").write_bytes(b"x" * 10)
    (tmp_path / "cache" / "covers").mkdir()
    (tmp_path / "cache" / "covers" / "a.png").write_bytes(b"y" * 5)
    (tmp_path / "cache" / "model_catalog_cache.json").write_text("{}")
    assert folder_usage(str(tmp_path / "cache")) == (17, 3)
    assert clear_folder(str(tmp_path / "cache")) == 2
    assert sorted(os.listdir(tmp_path / "cache")) == ["tiktoken"]  # offline token counting keeps working
    assert folder_usage(str(tmp_path / "missing")) == (0, 0)

    out = tmp_path / "data" / "Output" / "Book"
    out.mkdir(parents=True)
    (out / "book.epub").write_bytes(b"e")
    (out / "notes.txt").write_text("n")
    saved: list = []

    async def save_to_downloads(path):
        saved.append(os.path.basename(path))
        return "content://" + os.path.basename(path)

    files = types.SimpleNamespace(platform="android", save_to_downloads=save_to_downloads)
    assert _run(mirror_output(files, str(out), FakePrefs(mirror_outputs=False))) == []
    assert _run(mirror_output(files, str(out), FakePrefs(mirror_outputs=True))) == ["content://book.epub", "content://notes.txt"]
    files.platform = "ios"
    assert _run(mirror_output(files, str(out), FakePrefs(mirror_outputs=True))) == []


@needs_crypto
def test_backup_create_list_restore_delete(tmp_path, test_key):
    from glossarion_mobile.ui.screens.backup import config_without_keys, delete_backup, list_backups, restore_backup

    path = tmp_path / "config.json"
    store = _store(path, {"model": "m1", "api_key": "sk-first-key-000000", "temperature": 0.1})
    first = store.backup_now()
    assert first and os.path.basename(first).startswith("config_")
    time.sleep(1.1)  # backup names have one-second resolution
    store.set("model", "m2")
    second = store.backup_now()
    backups = list_backups(str(path))
    assert [b["path"] for b in backups][:2] == [second, first]  # newest first
    store.set("model", "m3")  # unsaved edit: flushed into the safety backup, then replaced by the restore
    time.sleep(1.1)
    assert restore_backup(store, first) == os.path.basename(first)
    assert store.get("model") == "m1" and store.get("api_key") == "sk-first-key-000000"
    names = [b["name"] for b in list_backups(str(path))]
    assert len(names) >= 3  # the safety backup of the pre-restore config
    with pytest.raises(ValueError, match="Not a configuration backup"):
        delete_backup(str(path), "../config.json")
    delete_backup(str(path), os.path.basename(second))
    assert os.path.basename(second) not in [b["name"] for b in list_backups(str(path))]
    stripped = config_without_keys({"api_key": "x", "model": "m", "multi_api_keys": [{"api_key": "k", "model": "a"}]})
    assert stripped == {"model": "m", "multi_api_keys": [{"model": "a"}]}
    store.close()


@needs_crypto
def test_import_from_desktop_reencrypts_with_the_device_key(tmp_path, test_key):
    from cryptography.fernet import Fernet

    import api_key_encryption
    from glossarion_mobile.ui.screens.desktop_import import (
        import_glossaries,
        merge_desktop_config,
        preview_import,
        read_desktop_key,
    )

    desktop_key = Fernet.generate_key()
    cipher = Fernet(desktop_key)

    def enc(value: str) -> str:
        return "ENC:" + base64.b64encode(cipher.encrypt(value.encode())).decode()

    desktop = {
        "model": "gemini-3.5-flash", "api_key": enc("sk-desktop-main-key-123"), "temperature": 0.7,
        "multi_api_keys": [{"api_key": enc("AIza-pool-key-0001"), "model": "gemini-3.5-flash", "enabled": True}],
        "antigravity_accounts_hint": {"kept": True},  # an excluded route's value: round-trips untouched
    }
    desktop_path = tmp_path / "desktop_config.json"
    desktop_path.write_text(json.dumps(desktop), encoding="utf-8")
    key_path = tmp_path / ".glossarion_key"
    key_path.write_bytes(desktop_key)

    without = preview_import(str(desktop_path))
    assert without.secrets == 2 and without.decrypted == 0 and without.undecryptable == ["api_key", "multi_api_keys"]
    preview = preview_import(str(desktop_path), read_desktop_key(str(key_path)))
    assert preview.decrypted == 2 and not preview.undecryptable
    assert preview.config["api_key"] == "sk-desktop-main-key-123" and "2/2 API keys decrypted" in preview.summary

    mobile_path = tmp_path / "mobile" / "config.json"
    mobile_path.parent.mkdir()
    store = _store(mobile_path, {"model": "authgpt/gpt-6-luna", "phone_only": 1})
    changed = merge_desktop_config(store, preview.config)
    assert set(changed) >= {"model", "api_key", "multi_api_keys", "temperature", "antigravity_accounts_hint"}
    raw = json.loads(mobile_path.read_text(encoding="utf-8"))
    assert raw["phone_only"] == 1 and raw["antigravity_accounts_hint"] == {"kept": True}
    assert raw["api_key"].startswith("ENC:") and raw["api_key"] != desktop["api_key"]
    device = api_key_encryption.get_handler()
    assert device.decrypt_value(raw["api_key"]) == "sk-desktop-main-key-123"  # re-encrypted with the device key
    assert device.decrypt_value(raw["multi_api_keys"][0]["api_key"]) == "AIza-pool-key-0001"
    with pytest.raises(Exception):
        Fernet(desktop_key).decrypt(base64.b64decode(raw["api_key"][4:]))
    assert list((mobile_path.parent / "config_backups").glob("config_*.json.bak"))  # backed up first
    store.close()

    gloss = tmp_path / "Glossary"
    one = tmp_path / "Book_glossary.csv"
    one.write_text("type,raw_name,translated_name\n", encoding="utf-8")
    archive = tmp_path / "glossaries.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Glossary/Other/Other_glossary.csv", "a")
        zf.writestr("Glossary/readme.pdf", "skip")
        zf.writestr("../escape_glossary.csv", "evil")
    copied = import_glossaries([str(one), str(archive), str(tmp_path / "x.exe")], str(gloss))
    names = sorted(os.path.relpath(p, gloss).replace("\\", "/") for p in copied)
    assert names == ["Book_glossary.csv", "Other/Other_glossary.csv"]
    assert not (tmp_path / "escape_glossary.csv").exists()


@needs_flet
@needs_crypto
def test_desktop_import_screen_scrubs_the_picked_key(tmp_path, test_key):
    from cryptography.fernet import Fernet

    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.desktop_import import DesktopImportScreen

    inbox = tmp_path / "Inbox"
    inbox.mkdir()
    key_copy = inbox / ".glossarion_key"
    key_copy.write_bytes(Fernet.generate_key())
    config_copy = inbox / "config.json"
    config_copy.write_text(json.dumps({"model": "x"}), encoding="utf-8")
    store = _store(tmp_path / "config.json", {"model": "y"})
    screen = DesktopImportScreen(parse_route("/settings/import"), Ctx(store), scrub_dirs=(str(inbox),))
    screen.get_body()
    _run(screen.choose_config(str(config_copy)))
    assert screen.preview is not None and not screen.import_button.disabled
    _run(screen.choose_key(str(key_copy)))
    assert screen.key_bytes and not key_copy.exists()  # read once, the app-owned copy deleted
    assert _run(screen.run_import()) == ["model"] and store.get("model") == "x"
    store.close()


def test_about_bundle_info_versions_and_licences(tmp_path):
    from glossarion_mobile.ui.screens.about import bundle_info, license_rows, version_rows

    (tmp_path / "_bundle_info.py").write_text(textwrap.dedent("""\
        FORMAT = 1
        BUILD_VERSION = '9.13.6'
        GIT_SHA = 'abcdef0123456789'
        GIT_DIRTY = True
        MODULE_COUNT = 120
        TOTAL_BYTES = 2048
        BUNDLE_SHA256 = '0123456789abcdef0123'
        MODULES = ('a', 'b')
        """), encoding="utf-8")
    info = bundle_info(tmp_path)
    assert info == {"BUILD_VERSION": "9.13.6", "GIT_SHA": "abcdef0123456789", "GIT_DIRTY": True, "MODULE_COUNT": 120,
                    "TOTAL_BYTES": 2048, "BUNDLE_SHA256": "0123456789abcdef0123"}
    rows = dict(version_rows({"version": "9.13.6", "build": 9130600}, info, platform_name="android",
                             flet_version="1.0.3", backend_source="bundle"))
    assert rows["Version"] == "9.13.6" and rows["Build"] == "9130600" and rows["Commit"] == "abcdef012345 (modified)"
    assert rows["Bundle"] == "120 modules · 2.0 KB" and rows["Platform"] == "android" and rows["Backend"] == "bundle"
    assert bundle_info(tmp_path / "missing") == {} and bundle_info(None) == {}
    licences = license_rows()
    assert licences and all(len(row) == 3 for row in licences)
    assert [r[0].lower() for r in licences] == sorted(r[0].lower() for r in licences)


_RESET_ORDER = (
    "api_key", "multi_api_keys", "fallback_keys", "glossary_keys", "glossary_refinement_keys", "metadata_keys",
    "qa_scan_keys", "ai_truncation_detection_keys", "rolling_summary_keys", "truncation_retry_keys", "inpainter_keys",
    "tts_keys", "replicate_api_key", "model", "azure_vision_key", "azure_vision_endpoint",
    "azure_document_intelligence_key", "azure_document_intelligence_endpoint", "google_vision_credentials",
    "google_cloud_credentials", "prompt_profiles", "active_profile", "use_multi_api_keys", "use_fallback_keys",
    "use_glossary_keys", "use_glossary_refinement_keys", "use_metadata_keys", "use_qa_scan_keys",
    "use_ai_truncation_detection_keys", "use_rolling_summary_keys", "use_truncation_retry_keys",
    "use_inpainter_keys", "use_tts_keys",
)


def test_reset_preserved_keys_are_the_shared_desktop_block():
    """Danger zone runs ``config_store.reset_preserved_keys``, the block other_settings'
    ``_reset_config_to_defaults`` calls too (tests/parity/test_u4_dialog_parity.py compares it
    with the legacy desktop closure); no mobile copy."""
    import config_store
    from glossarion_mobile.ui.screens.danger_zone import RESET_PRESERVED_TEXT, preserved_reset_keys

    assert preserved_reset_keys is config_store.reset_preserved_keys
    assert RESET_PRESERVED_TEXT is config_store.RESET_PRESERVED_TEXT
    source = (SRC_DIR / "other_settings.py").read_text(encoding="utf-8-sig")
    start = source.index("    def _reset_config_to_defaults():")
    handler = source[start:source.index("reset_btn = QPushButton", start)]
    assert "reset_preserved_keys(current_config)" in handler and "+ RESET_PRESERVED_TEXT" in handler
    rich = {key: f"value-{key}" for key in reversed(_RESET_ORDER)}
    rich.update({"temperature": 1, "batch_size": 3, "openai_base_url": "http://x",
                 "qa_scanner_settings": {"excluded_characters": "★", "min_file_length": 9}})
    kept = preserved_reset_keys(rich)
    assert list(kept) == [*_RESET_ORDER, "qa_scanner_settings"]  # the desktop's order, not the config's
    assert kept["qa_scanner_settings"] == {"excluded_characters": "★"} and kept["model"] == "value-model"
    assert preserved_reset_keys({"qa_scanner_settings": "broken", "api_key": ""}) == {"api_key": ""}
    assert "• QA Scanner Excluded Characters" in RESET_PRESERVED_TEXT


@needs_crypto
def test_reset_settings_and_wipe(tmp_path, test_key):
    from glossarion_mobile.ui.screens.danger_zone import reset_settings, wipe_app_data, wipe_targets

    path = tmp_path / "data" / "config.json"
    path.parent.mkdir()
    store = _store(path, {"model": "m", "api_key": "sk-keep-me-0000000", "temperature": 0.9,
                          "qa_scanner_settings": {"excluded_characters": "x", "min_file_length": 5},
                          "prompt_profiles": {"A": "a"}})
    kept = reset_settings(store)
    assert kept == {"api_key": "sk-keep-me-0000000", "model": "m", "prompt_profiles": {"A": "a"},
                    "qa_scanner_settings": {"excluded_characters": "x"}}
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert set(raw) == set(kept) and raw["api_key"].startswith("ENC:") and "temperature" not in raw
    assert list((path.parent / "config_backups").glob("config_*.json.bak"))
    store.close()

    data, cache, temp = tmp_path / "data", tmp_path / "cache", tmp_path / "temp"
    for folder in (data / "logs", data / "home" / ".glossarion", data / "Inbox", cache / "tiktoken", temp / "x"):
        folder.mkdir(parents=True, exist_ok=True)
    (data / "mobile_state.json").write_text("{}")
    (data / "logs" / "run.log").write_text("log")
    paths = types.SimpleNamespace(data=data, docs=data, cache=cache, temp=temp, logs=data / "logs")
    targets = {os.path.relpath(t, tmp_path).replace("\\", "/") for t in wipe_targets(paths)}
    assert "data/logs" not in targets and {"data/config.json", "data/home", "data/Inbox", "cache/tiktoken"} <= targets
    stopped: list = []
    assert wipe_app_data(paths, stop=lambda: stopped.append(True)) == []
    assert stopped == [True] and sorted(os.listdir(data)) == ["logs"] and os.listdir(cache) == []


# ==========================================================================
# Feature installer
# ==========================================================================


_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_u4acc", Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)
storage = _TB.storage
app_env = _TB.app_env


def _foundations():
    spec = importlib.util.spec_from_file_location("_glossarion_tf_helpers_u4acc",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")
@pytest.mark.skipif(not (_has("requests") and _has("bs4")),
                    reason="the real pages drive the backend (authgem_auth, the prompt_profiles start-up): needs requests/bs4")
def test_feature_on_the_real_app_renders_every_page(app_env, caplog, monkeypatch):
    """The real app on a fake Flet session: install the feature, open every page it owns (the
    session serialises each View, so invalid control properties fail here) and the settings home."""
    from glossarion_mobile.ui.router import build_route
    from glossarion_mobile.ui.screens.pages_feature import SCREEN_ROUTES, AccountsProfilesFeature
    from glossarion_mobile.ui.screens.profiles import profile_id

    tf = _foundations()

    async def scenario():
        _m, conn, session, page, app = await tf._start("android")
        try:
            await tf._wait(lambda: app.state.engine_ready, timeout=60)
            feature = await AccountsProfilesFeature.install(app, exit_app=lambda: None)
            assert app.chat_feature.oauth.config_get is not None
            built = []
            for name in SCREEN_ROUTES:
                params = {"pid": profile_id("Universal")} if name == "settings.profiles.detail" else None
                match = await app.navigate(build_route(name, params))
                assert match is not None and match.name == name, name
                screen = app.shell.top_screen
                assert screen is not None and type(screen).__name__ != "PlaceholderScreen", name
                built.append(type(screen).__name__)
                await asyncio.sleep(0.3)  # did_show work (account status, folder usage, backups) runs off the loop
            assert built == ["AccountsScreen", "ProfilesScreen", "ProfileDetailScreen", "PrefillScreen",
                             "AppearanceScreen", "StorageScreen", "CloudSyncScreen", "BackupScreen",
                             "DesktopImportScreen", "AboutScreen", "DangerZoneScreen", "NotificationsScreen",
                             "WelcomeScreen"]  # CloudSyncScreen: Settings › Cloud sync & sharing (U10)
            assert feature.screens_built == list(SCREEN_ROUTES)
            errors = [r for r in caplog.records if r.levelno >= 40 and r.name.startswith("glossarion")]
            assert not errors, [r.getMessage() for r in errors]
            # every provider slot's status is mirrored into AppState.signed_in (ModelSheet / Send / drawer)
            import authgem_auth
            from glossarion_mobile import runtime_bootstrap as rb

            token_dir = Path(rb.get_paths().home) / ".glossarion"
            # (an earlier test in this process may have imported authgem_auth before the bootstrap env existed)
            monkeypatch.setattr(authgem_auth, "_DEFAULT_TOKEN_DIR", str(token_dir))
            monkeypatch.setattr(authgem_auth, "_account_stores", {})
            store = authgem_auth.get_store(2)
            assert Path(store._token_file).parent == token_dir
            store.save_tokens({"access_token": "t", "refresh_token": "r", "expires_at": time.time() + 3600})
            assert "authgem2" in await feature.refresh_sign_ins() and "authgem2" in app.state.signed_in.value
            await app.chat_feature.oauth.sign_out(2, "authgem")
            assert not Path(store._token_file).exists() and "authgem2" not in feature.sync_signed_in()
            await app.navigate(build_route("settings"))
            home = app.shell.top_screen
            for name in ("settings.appearance", "settings.storage", "settings.cloud", "settings.backup", "settings.import",
                         "settings.about", "settings.danger", "settings.profiles", "settings.prefill", "settings.accounts"):
                assert name in home.implemented, name
                tile = home.route_tiles.get(name)
                assert tile is None or getattr(tile.trailing, "reason", None) is None, name
        finally:
            app.jobs.close()
            await tf._stop(app)

    asyncio.run(scenario())


@needs_flet
@needs_crypto
def test_feature_installs_screens_and_extends_the_settings_home(tmp_path, test_key, fake_auth):
    from glossarion_mobile.state.prefs import Prefs
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.about import AboutScreen
    from glossarion_mobile.ui.screens.accounts import AccountsScreen
    from glossarion_mobile.ui.screens.appearance import AppearanceScreen
    from glossarion_mobile.ui.screens.backup import BackupScreen
    from glossarion_mobile.ui.screens.danger_zone import DangerZoneScreen
    from glossarion_mobile.ui.screens.desktop_import import DesktopImportScreen
    from glossarion_mobile.ui.screens.pages_feature import IMPLEMENTED_ROUTES, AccountsProfilesFeature
    from glossarion_mobile.ui.screens.prefill_profiles import PrefillScreen
    from glossarion_mobile.ui.screens.profiles import ProfileDetailScreen, ProfilesScreen, profile_id
    from glossarion_mobile.ui.screens.storage import StorageScreen
    from glossarion_mobile.ui.screens.welcome import WelcomeScreen

    store = _store(tmp_path / "config.json", {"prompt_profiles": {"Universal": "u"}})
    prefs = Prefs(tmp_path / "mobile_state.json")
    prefs.load()
    prefs.set("appearance", {"theme": "light", "accent": "library_violet"})
    fallback_calls: list = []

    class Home:
        def __init__(self, ctx):
            self.implemented = frozenset({"settings.logs"})
            self.ctx = ctx

        def _on_profiles(self, e=None):
            return "info sheet"

    ctx = Ctx(store, prefs)

    def fallback(match):
        fallback_calls.append(match.name)
        return Home(ctx) if match.name == "settings" else types.SimpleNamespace(name=match.name)

    def welcome_screen(match):
        return WelcomeScreen(match)

    page = types.SimpleNamespace(theme=None, dark_theme=None, theme_mode=None, update=lambda: None, platform=None)
    chat = types.SimpleNamespace(oauth=fake_auth.bridge, env=ctx, chats=None, welcome_screen=welcome_screen,
                                 _spawn=lambda c: c.close(), refresh_sign_in=lambda: asyncio.sleep(0))
    paths = types.SimpleNamespace(data=tmp_path, docs=tmp_path, cache=tmp_path / "c", temp=tmp_path / "t",
                                  logs=tmp_path / "l", output=tmp_path / "o", library=tmp_path / "lib",
                                  home=tmp_path / "home", backend_dir=None, backend_source="repo")
    app = types.SimpleNamespace(page=page, shell=types.SimpleNamespace(screen_factory=fallback, tablet=False),
                                chat_feature=chat, config_store=store, prefs=prefs, paths=paths, state=None,
                                haptics=None, notify=lambda *a: None, navigate_to=lambda *a, **k: None, jobs=None)
    feature = _run(AccountsProfilesFeature.install(app, profiles=None, exit_app=lambda: None))
    assert app.pages_feature is feature and fake_auth.bridge.config_get is not None
    assert page.theme_mode is not None and page.theme.color_scheme_seed == "#6C63FF"  # saved appearance applied
    factory = app.shell.screen_factory
    expected = {
        "/settings/accounts": AccountsScreen, "/settings/profiles": ProfilesScreen,
        f"/settings/profiles/{profile_id('Universal')}": ProfileDetailScreen, "/settings/prefill": PrefillScreen,
        "/settings/appearance": AppearanceScreen, "/settings/storage": StorageScreen, "/settings/backup": BackupScreen,
        "/settings/import": DesktopImportScreen, "/settings/about": AboutScreen, "/settings/danger": DangerZoneScreen,
        "/welcome": WelcomeScreen,
    }
    for route, cls in expected.items():
        assert isinstance(factory(parse_route(route)), cls), route
    welcome = factory(parse_route("/welcome"))
    assert welcome.oauth is fake_auth.bridge and welcome.page is page
    home = factory(parse_route("/settings"))
    assert IMPLEMENTED_ROUTES <= home.implemented and "settings.logs" in home.implemented
    home._on_profiles()
    assert ctx.routes[-1] == ("settings.profiles", None)
    assert factory(parse_route("/jobs")).name == "jobs" and fallback_calls == ["settings", "jobs"]
    assert "settings.profiles.detail" not in IMPLEMENTED_ROUTES and "welcome" not in IMPLEMENTED_ROUTES
    store.close()
    prefs.close()
