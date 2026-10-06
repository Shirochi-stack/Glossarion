"""Glossarion Mobile U4: begin/complete sign-in splits of authgem, authcd and authgrok.

Covered:
  * oauth_session: the shared loopback session, encrypted pending sign-in,
    pasted-redirect parser, GLOSSARION_OAUTH_RETURN_URL success-page seam and
    the GLOSSARION_TOKEN_DIR token folder
  * authgem_auth: begin_oauth() (random 127.0.0.1 port) / complete_from_redirect()
    (loopback or pasted redirect URL), run_oauth_flow() composed from them
  * authcd_auth: begin_oauth() (localhost flow, manual code-page URL) /
    complete_from_redirect() (localhost URL or code#state paste), the legacy
    paste path fixed (code#state split, state sent, CLI scopes),
    run_automatic_oauth_login() composed from them, and the Claude Code CLI is
    never run or imported on mobile
  * authgrok_auth: begin_device_login() / poll_device_login() /
    resume_device_login() (RFC 8628 device code), no browser and no Grok CLI
    import on mobile, run_device_oauth_flow() composed from them
  * desktop parity: the modules at BASE_SHA (``git show``) and the split ones
    serve the same pages and send the same requests, return the same tokens,
    print the same output and raise the same errors under the same fakes

No network: the token, userinfo and xAI endpoints are faked and the "browser"
is http.client talking to the loopback servers.
"""
import base64
import hashlib
import http.client
import importlib
import json
import os
import re
import socket
import subprocess
import sys
import textwrap
import threading
import time
import types
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import authcd_auth  # noqa: E402
import authgem_auth  # noqa: E402
import authgrok_auth  # noqa: E402
import oauth_session  # noqa: E402

#: Parent commit of the U4 auth splits (the merged U3 + CI fix); the legacy modules come from here.
BASE_SHA = "a7aa4a75847e1acdedeefcc86e69d1aab500ed78"

ENV_NAMES = (
    "GLOSSARION_MOBILE",
    "GLOSSARION_NO_PROCESSES",
    "GLOSSARION_OAUTH_RETURN_URL",
    "GLOSSARION_TOKEN_DIR",
    "FLET_PLATFORM",
    "AUTHGEM_TOKEN_FILE",
    "AUTHGROK_TOKEN_FILE",
    "AUTHCD_TOKEN_FILE",
    "AUTHGROK_ACCESS_TOKEN",
    "AUTHGROK_REFRESH_TOKEN",
    "TRANSLATION_CANCELLED",
    "SPACE_ID",
    "HF_SPACES",
    "DOCKER_CONTAINER",
    "KUBERNETES_SERVICE_HOST",
)
RETURN_URL = "glossarion://app/oauth/return"
TEST_TOKEN_KEY = base64.b64encode(bytes(range(32))).decode("ascii")

LEGACY_GEM_SUCCESS_HTML = (
    "<html><body style='font-family:sans-serif;text-align:center;padding-top:60px;"
    "background:#1a1a2e;color:#e0e0e0'>"
    "<h1 style='color:#4db8ff'>&#10004; Gemini Authenticated!</h1>"
    "<p>You can close this tab and return to Glossarion.</p>"
    "</body></html>"
)
LEGACY_GROK_SUCCESS_HTML = (
    "<html><body><h1>Grok authorization received.</h1>"
    "You can close this tab and return to Glossarion.</body></html>"
)


@pytest.fixture(autouse=True)
def _desktop_env(monkeypatch):
    """Every test starts from the desktop environment; token encryption uses a test key."""
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    # Linux/macOS: never create or read the user's real token key (Windows uses DPAPI).
    monkeypatch.setenv("GLOSSARION_TOKEN_KEY_B64", TEST_TOKEN_KEY)
    authcd_auth.reset_cancel()


def _mobile(monkeypatch):
    monkeypatch.setenv("GLOSSARION_MOBILE", "1")
    monkeypatch.setenv("GLOSSARION_NO_PROCESSES", "1")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _port_is_free(port):
    """True when nothing listens on 127.0.0.1:*port* (binds like HTTPServer, see the U1 tests)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if os.name != "nt":
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _http_get(host, port, path):
    conn = http.client.HTTPConnection(host, port, timeout=10)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        headers = {k: v for k, v in resp.getheaders() if k.lower() != "date"}  # differs between runs
        return resp.status, headers, resp.read().decode("utf-8")
    finally:
        conn.close()


def _query(url):
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


def _s256(verifier):
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")


class _Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = {} if body is None else body
        self.headers = headers or {}
        self.text = json.dumps(self._body)
        self.ok = status < 400
        self.is_redirect = False
        self.is_permanent_redirect = False

    def json(self):
        return json.loads(json.dumps(self._body))

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Net:
    """Stands in for requests.post / requests.get; answers by URL and records every call."""

    def __init__(self):
        self.calls = []
        self.routes = []

    def route(self, method, url, responder):
        self.routes.append((method, url, responder))

    def _call(self, method, url, kwargs):
        record = {"method": method, "url": url}
        for key in ("data", "json", "params", "headers", "timeout"):
            if key in kwargs:
                value = kwargs[key]
                record[key] = dict(value) if isinstance(value, dict) else value
        self.calls.append(record)
        for route_method, route_url, responder in self.routes:
            if route_method == method and url.startswith(route_url):
                return responder(record)
        raise AssertionError(f"unexpected {method} {url}")

    def post(self, url, **kwargs):
        return self._call("POST", url, kwargs)

    def get(self, url, **kwargs):
        return self._call("GET", url, kwargs)


@pytest.fixture
def net(monkeypatch):
    import requests

    fake = _Net()
    monkeypatch.setattr(requests, "post", fake.post)
    monkeypatch.setattr(requests, "get", fake.get)
    return fake


@pytest.fixture
def no_browser(monkeypatch):
    """webbrowser.open must not be called (the caller opens the page itself)."""
    import webbrowser

    opened = []

    def forbidden(url, *args, **kwargs):
        opened.append(url)
        raise AssertionError(f"webbrowser.open({url!r}) was called")

    monkeypatch.setattr(webbrowser, "open", forbidden)
    return opened


# ---------------------------------------------------------------------------
# legacy modules (BASE_SHA) for the desktop differential
# ---------------------------------------------------------------------------

_LEGACY = {}


def legacy_module(name):
    """``src/<name>.py`` at BASE_SHA, executed as a private module (shares requests/webbrowser/secrets)."""
    if name not in _LEGACY:
        try:
            data = subprocess.run(
                ["git", "show", f"{BASE_SHA}:src/{name}.py"], cwd=str(REPO_ROOT),
                check=True, capture_output=True,
            ).stdout
        except Exception as exc:  # pragma: no cover - environment dependent
            message = f"git show {BASE_SHA}:src/{name}.py unavailable: {exc}"
            if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
                pytest.fail(message + " (CI must fetch the base commit)")
            pytest.skip(message)
        source = data.decode("utf-8-sig").replace("\r\n", "\n")
        module = types.ModuleType(f"legacy_{name}")
        module.__file__ = str(SRC / f"{name}.py")
        exec(compile(source, f"<legacy {name}>", "exec"), module.__dict__)
        _LEGACY[name] = module
    return _LEGACY[name]


def _normalize(value, ports):
    """Replace the run's loopback port(s) with <PORT> so two runs compare equal."""
    if isinstance(value, dict):
        return {k: _normalize(v, ports) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_normalize(v, ports) for v in value)
    if isinstance(value, str):
        for port in ports:
            value = re.sub(rf"(?<![0-9]){port}(?![0-9])", "<PORT>", value)
        return value
    return value


@pytest.fixture
def fixed_secrets(monkeypatch):
    """Deterministic PKCE verifier and state so legacy and split runs build the same URLs."""
    import secrets

    monkeypatch.setattr(secrets, "token_urlsafe", lambda nbytes=None: "fixed-state-0123456789")
    monkeypatch.setattr(secrets, "token_bytes", lambda nbytes=None: bytes(range(nbytes or 32)))


# ===========================================================================
# oauth_session
# ===========================================================================

PASTED_CASES = [
    ("http://localhost:1455/auth/callback?code=c1&state=s1", ("c1", "s1", None)),
    ("localhost:1455/auth/callback?state=s2&code=c2", ("c2", "s2", None)),
    ("http://localhost:1455/auth/callback#code=c3&state=s3", ("c3", "s3", None)),
    ("?code=c4&state=s4", ("c4", "s4", None)),
    ("code=c5&state=s5", ("c5", "s5", None)),
    ("c6#s6", ("c6", "s6", None)),
    ("  c7  ", ("c7", None, None)),
    ("http://localhost:1455/auth/callback?error=access_denied&state=s8", (None, "s8", "access_denied")),
    ("", (None, None, None)),
]


@pytest.mark.parametrize("pasted, expected", PASTED_CASES)
def test_parse_oauth_redirect_matches_the_authgpt_parser(pasted, expected):
    authgpt_auth = importlib.import_module("authgpt_auth")
    assert oauth_session.parse_oauth_redirect(pasted) == expected
    assert authgpt_auth._parse_oauth_redirect(pasted) == expected


def test_default_token_dir_desktop_and_override(monkeypatch, tmp_path):
    assert oauth_session.default_token_dir() == os.path.join(os.path.expanduser("~"), ".glossarion")
    monkeypatch.setenv("GLOSSARION_TOKEN_DIR", f"  {tmp_path}  ")
    assert oauth_session.default_token_dir() == str(tmp_path)
    monkeypatch.setenv("GLOSSARION_TOKEN_DIR", "  ")
    assert oauth_session.default_token_dir() == os.path.join(os.path.expanduser("~"), ".glossarion")


def test_success_page_seam(monkeypatch):
    assert oauth_session.oauth_success_html("authgem", "<p>desktop</p>") == "<p>desktop</p>"
    monkeypatch.setenv("GLOSSARION_OAUTH_RETURN_URL", RETURN_URL)
    page = oauth_session.oauth_success_html("authgem", "<p>desktop</p>", "Done!")
    assert "href='glossarion://app/oauth/return?p=authgem'" in page
    assert "<h1>Done!</h1>" in page and "location.href" in page
    monkeypatch.setenv("GLOSSARION_OAUTH_RETURN_URL", RETURN_URL + "?x=1")
    assert "href='glossarion://app/oauth/return?x=1&amp;p=authcd'" in oauth_session.oauth_success_html("authcd", "")
    # Same page as authgpt's (U1) apart from the provider.
    authgpt_auth = importlib.import_module("authgpt_auth")
    monkeypatch.setenv("GLOSSARION_OAUTH_RETURN_URL", RETURN_URL)
    assert oauth_session.oauth_return_page(RETURN_URL, "authgpt") == authgpt_auth._oauth_success_html()


class _PendingStore(oauth_session.PendingOAuthMixin):
    def __init__(self, token_file):
        self._token_file = str(token_file)
        self._lock = threading.RLock()

    def _ensure_dir(self):
        os.makedirs(os.path.dirname(self._token_file), exist_ok=True)


def test_pending_store_encrypts_and_expires(tmp_path):
    store = _PendingStore(tmp_path / "sub" / "x_tokens_2.json")
    assert store.pending_oauth_file == str(tmp_path / "sub" / "x_tokens_2.oauth_pending")
    assert store.load_pending_oauth() is None
    secret = "verifier-SECRET-123"
    assert store.save_pending_oauth({"code_verifier": secret, "state": "st", "created_at": time.time()})
    raw = Path(store.pending_oauth_file).read_bytes()
    assert raw.startswith(b"GLSE1:") and secret.encode() not in raw
    assert store.load_pending_oauth()["code_verifier"] == secret

    store.save_pending_oauth({"state": "st", "created_at": time.time() - oauth_session.PENDING_OAUTH_MAX_AGE_SECONDS - 1})
    assert store.load_pending_oauth() is None and not Path(store.pending_oauth_file).exists()
    store.save_pending_oauth({"state": "st", "created_at": time.time(), "expires_at": time.time() - 1})
    assert store.load_pending_oauth() is None and not Path(store.pending_oauth_file).exists()
    Path(store.pending_oauth_file).write_bytes(b"not encrypted")
    assert store.load_pending_oauth() is None and not Path(store.pending_oauth_file).exists()
    store.clear_pending_oauth()  # nothing there: no error


def test_token_dir_override_reaches_every_slot_and_cache(tmp_path):
    """GLOSSARION_TOKEN_DIR moves default and numbered slots and the login caches (fresh imports)."""
    # Desktop: module constants only (building a store would decrypt the user's real token files).
    script = textwrap.dedent(
        """
        import json, os, sys
        sys.path.insert(0, sys.argv[1])
        import authcd_auth, authgem_auth, authgrok_auth
        files = [authgem_auth._DEFAULT_TOKEN_FILE, authgrok_auth._DEFAULT_TOKEN_FILE, authcd_auth._DEFAULT_TOKEN_FILE]
        if sys.argv[2] == "stores":
            files = [
                authgem_auth.get_store(0)._token_file, authgem_auth.get_store(3)._token_file,
                authgrok_auth.get_store(0)._token_file, authgrok_auth.get_store(3)._token_file,
                authcd_auth.get_store(0)._token_file, authcd_auth.get_store(3)._token_file,
            ]
        print(json.dumps({
            "dirs": [authgem_auth._DEFAULT_TOKEN_DIR, authgrok_auth._DEFAULT_TOKEN_DIR, authcd_auth._DEFAULT_TOKEN_DIR],
            "files": files + [authcd_auth._CLIENT_VERSION_FILE, authcd_auth.GLOSSARION_CLAUDE_CONFIG_DIR],
        }))
        """
    )
    env = {k: v for k, v in os.environ.items() if k not in ENV_NAMES}
    desktop = json.loads(subprocess.run(
        [sys.executable, "-c", script, str(SRC), "constants"], env=env, capture_output=True, text=True, check=True,
    ).stdout)
    home_dir = os.path.join(os.path.expanduser("~"), ".glossarion")
    assert desktop["dirs"] == [home_dir] * 3
    assert [os.path.relpath(p, home_dir) for p in desktop["files"]] == [
        "authgem_tokens.json", "authgrok_tokens.json", "authcd_tokens.json", "authcd_client_version.json", "claude-code",
    ]

    env.update(GLOSSARION_TOKEN_DIR=str(tmp_path / "tok"), GLOSSARION_MOBILE="1", GLOSSARION_NO_PROCESSES="1")
    mobile = json.loads(subprocess.run(
        [sys.executable, "-c", script, str(SRC), "stores"], env=env, capture_output=True, text=True, check=True,
    ).stdout)
    assert mobile["dirs"] == [str(tmp_path / "tok")] * 3
    names = [os.path.relpath(p, tmp_path / "tok") for p in mobile["files"]]
    assert names == [
        "authgem_tokens.json", "authgem_tokens_3.json", "authgrok_tokens.json", "authgrok_tokens_3.json",
        "authcd_tokens.json", "authcd_tokens_3.json", "authcd_client_version.json", "claude-code",
    ]


# ===========================================================================
# authgem_auth
# ===========================================================================

GEM_TOKENS = {"access_token": "gem-at", "refresh_token": "gem-rt", "expires_in": 3600, "scope": "x"}
GEM_USER = {"email": "gem@example.test", "name": "Gem User", "picture": "https://example.test/p.png"}


@pytest.fixture
def gem_net(net):
    net.route("POST", authgem_auth.GOOGLE_TOKEN_URL, lambda record: _Resp(200, GEM_TOKENS))
    net.route("GET", authgem_auth.GOOGLE_USERINFO_URL, lambda record: _Resp(200, GEM_USER))
    return net


def _gem_exchange(redirect_uri, code):
    return {
        "method": "POST",
        "url": authgem_auth.GOOGLE_TOKEN_URL,
        "data": {
            "grant_type": "authorization_code",
            "client_id": authgem_auth.GOOGLE_CLIENT_ID,
            "client_secret": authgem_auth.GOOGLE_CLIENT_SECRET,
            "code": code,
            "redirect_uri": redirect_uri,
        },
        "timeout": 30,
    }


def test_gem_begin_loopback_then_complete_saves_tokens(gem_net, no_browser, tmp_path, monkeypatch):
    monkeypatch.setenv("GLOSSARION_OAUTH_RETURN_URL", RETURN_URL)
    token_file = tmp_path / "authgem_tokens_2.json"
    store = authgem_auth.AuthGemTokenStore(token_file=str(token_file), account_id=2)
    session = authgem_auth.begin_oauth(store, timeout=30)
    try:
        port = session.port
        assert session.redirect_uri == f"http://127.0.0.1:{port}/oauth2callback"
        query = _query(session.auth_url)
        assert session.auth_url.startswith(authgem_auth.GOOGLE_AUTH_URL + "?")
        assert query == {
            "client_id": authgem_auth.GOOGLE_CLIENT_ID,
            "response_type": "code",
            "redirect_uri": session.redirect_uri,
            "scope": " ".join(authgem_auth.OAUTH_SCOPES),
            "state": session.state,
            "access_type": "offline",
            "prompt": "consent",
        }
        assert session.account_id == 2 and session.store is store and session.provider == "authgem"
        assert session.server_running and not session.callback_received

        pending_file = Path(store.pending_oauth_file)
        assert pending_file.name == "authgem_tokens_2.oauth_pending"
        raw = pending_file.read_bytes()
        assert raw.startswith(b"GLSE1:") and session.state.encode() not in raw
        pending = store.load_pending_oauth()
        assert pending["provider"] == "authgem" and pending["state"] == session.state
        assert pending["redirect_uri"] == session.redirect_uri and pending["code_verifier"] is None

        status, headers, _ = _http_get("127.0.0.1", port, f"/oauth2callback?state={session.state}&code=gem-code&scope=x")
        assert status == 302 and headers["Location"] == f"http://127.0.0.1:{port}/success"
        assert session.wait_for_callback(5)
        tokens = authgem_auth.complete_from_redirect(session)
        status, _, body = _http_get("127.0.0.1", port, "/success")
        assert status == 200 and "glossarion://app/oauth/return?p=authgem" in body
        assert session.wait(5) and not session.server_running
    finally:
        session.close()

    assert no_browser == []
    assert gem_net.calls[0] == _gem_exchange(session.redirect_uri, "gem-code")
    assert gem_net.calls[1]["url"] == authgem_auth.GOOGLE_USERINFO_URL
    assert tokens["access_token"] == "gem-at" and tokens["_user_info"]["email"] == "gem@example.test"
    reloaded = authgem_auth.AuthGemTokenStore(token_file=str(token_file), account_id=2)
    assert reloaded.load_tokens()["refresh_token"] == "gem-rt"
    assert reloaded.account_info == {"email": "gem@example.test", "name": "Gem User"}
    assert not pending_file.exists()
    assert _port_is_free(port)


def test_gem_paste_redirect_after_the_listener_died(gem_net, tmp_path):
    token_file = tmp_path / "authgem_tokens.json"
    store = authgem_auth.AuthGemTokenStore(token_file=str(token_file))
    session = authgem_auth.begin_oauth(store, timeout=30)
    session.close()  # app killed while the browser was open: only the encrypted pending file survives
    fresh = authgem_auth.AuthGemTokenStore(token_file=str(token_file))
    pasted = f"  {session.redirect_uri}?state={session.state}&code=pasted-code&scope=x  "
    tokens = authgem_auth.complete_from_redirect(fresh, pasted)
    assert gem_net.calls[0] == _gem_exchange(session.redirect_uri, "pasted-code")
    assert tokens["_user_info"]["name"] == "Gem User"
    assert fresh.load_tokens()["access_token"] == "gem-at"
    assert fresh.load_pending_oauth() is None


def test_gem_paste_rejections(gem_net, tmp_path):
    store = authgem_auth.AuthGemTokenStore(token_file=str(tmp_path / "authgem_tokens.json"))
    with pytest.raises(RuntimeError, match="No pending Gemini sign-in"):
        authgem_auth.complete_from_redirect(store, "?code=x&state=y")
    session = authgem_auth.begin_oauth(store, timeout=30)
    session.close()
    uri = session.redirect_uri
    with pytest.raises(RuntimeError, match="state mismatch"):
        authgem_auth.complete_from_redirect(store, f"{uri}?state=forged&code=evil")
    with pytest.raises(RuntimeError, match="whole redirect URL"):
        authgem_auth.complete_from_redirect(store, "bare-code-without-state")
    with pytest.raises(RuntimeError, match="Google OAuth error: access_denied"):
        authgem_auth.complete_from_redirect(store, f"{uri}?error=access_denied&state={session.state}")
    with pytest.raises(RuntimeError, match="authorization code"):
        authgem_auth.complete_from_redirect(store, f"{uri}?state={session.state}")
    with pytest.raises(RuntimeError, match="not a Gemini sign-in"):
        authgem_auth.complete_from_redirect(dict(store.load_pending_oauth(), provider="authcd"), "c#s")
    assert gem_net.calls == []
    assert store.load_tokens() is None
    # The genuine redirect still completes afterwards.
    authgem_auth.complete_from_redirect(store, f"{uri}?state={session.state}&code=ok")
    assert store.load_tokens()["access_token"] == "gem-at"


def test_gem_begin_with_account_id_uses_that_slot(gem_net, tmp_path, monkeypatch):
    monkeypatch.setattr(authgem_auth, "_DEFAULT_TOKEN_DIR", str(tmp_path))
    monkeypatch.setattr(authgem_auth, "_account_stores", {})
    session = authgem_auth.begin_oauth(account_id=4, timeout=30)
    try:
        assert session.store._token_file == str(tmp_path / "authgem_tokens_4.json")
        assert session.account_id == 4
        assert Path(tmp_path / "authgem_tokens_4.oauth_pending").exists()
    finally:
        session.close()


def test_gem_sign_in_keeps_the_gcp_project_selection(gem_net, tmp_path, monkeypatch):
    """A sign-in leaves the per-account project choice alone, as the desktop login does;
    signing out of that slot still resets it (clear_tokens)."""
    monkeypatch.setattr(authgem_auth, "_cached_project_id", {0: "proj-zero", 2: "proj-two"})
    monkeypatch.setattr(authgem_auth, "_project_set_by_gui", {2: True})
    store = authgem_auth.AuthGemTokenStore(token_file=str(tmp_path / "authgem_tokens_2.json"), account_id=2)
    session = authgem_auth.begin_oauth(store, timeout=30)
    session.close()
    authgem_auth.complete_from_redirect(store, f"{session.redirect_uri}?state={session.state}&code=c")
    assert authgem_auth._cached_project_id == {0: "proj-zero", 2: "proj-two"}
    assert authgem_auth._project_set_by_gui == {2: True}
    store.clear_tokens()
    assert authgem_auth._cached_project_id == {0: "proj-zero"} and authgem_auth._project_set_by_gui == {}


def test_gem_run_oauth_flow_composes_the_split(gem_net, monkeypatch):
    calls = []
    real_begin = authgem_auth.begin_oauth

    def spy(*args, **kwargs):
        session = real_begin(*args, **kwargs)
        calls.append((args, kwargs, session))
        return session

    monkeypatch.setattr(authgem_auth, "begin_oauth", spy)
    monkeypatch.setattr("webbrowser.open", _gem_browser([]))
    tokens = authgem_auth.run_oauth_flow(timeout=20)
    args, kwargs, session = calls[0]
    assert (args, kwargs) == ((), {"timeout": 20, "persist": False, "auto_close": False})
    assert session.store is None and len(session._servers) == 1
    assert tokens["access_token"] == "gem-at" and not session.server_running


def _gem_browser(pages, state=None, error=None, visit_success=True):
    """Fake browser for the Gemini loopback: hit the callback (and /success) on another thread."""

    def browser(url, *args, **kwargs):
        query = _query(url)
        port = urlparse(query["redirect_uri"]).port
        callback = f"/oauth2callback?state={state or query['state']}"
        callback += f"&error={error}" if error else "&code=gem-code&scope=x"

        def visit():
            pages.append(_http_get("127.0.0.1", port, callback))
            if visit_success:
                pages.append(_http_get("127.0.0.1", port, "/success"))

        threading.Thread(target=visit, daemon=True).start()
        return True

    return browser


@pytest.mark.parametrize(
    "scenario, browser_kwargs, timeout",
    [
        ("success", {}, 20),
        ("state mismatch", {"state": "forged"}, 20),
        ("denied", {"error": "access_denied", "visit_success": False}, 1.5),
        ("no callback", None, 0.6),
    ],
)
def test_gem_run_oauth_flow_matches_legacy(gem_net, fixed_secrets, monkeypatch, capsys, scenario, browser_kwargs, timeout):
    legacy = legacy_module("authgem_auth")
    results = {}
    for name, module in (("legacy", legacy), ("split", authgem_auth)):
        gem_net.calls.clear()
        pages = []
        port = _free_port()
        monkeypatch.setattr(module, "_find_available_port", lambda port=port: port)
        browser = (lambda url, *a, **k: True) if browser_kwargs is None else _gem_browser(pages, **browser_kwargs)
        monkeypatch.setattr("webbrowser.open", browser)
        try:
            outcome = ("tokens", {k: v for k, v in module.run_oauth_flow(timeout=timeout).items() if k != "expires_at"})
        except Exception as exc:
            outcome = ("error", type(exc).__name__, str(exc))
        _wait_until(lambda: len(pages) >= (0 if browser_kwargs is None else (2 if browser_kwargs.get("visit_success", True) else 1)))
        results[name] = _normalize(
            {"outcome": outcome, "pages": pages, "calls": list(gem_net.calls), "out": capsys.readouterr().out}, [port]
        )
    assert results["split"] == results["legacy"]
    assert results["split"]["outcome"][0] == ("tokens" if scenario == "success" else "error")


# ===========================================================================
# authcd_auth
# ===========================================================================

CD_TOKENS = {"access_token": "cd-at", "refresh_token": "cd-rt", "expires_in": 3600}


@pytest.fixture
def cd_net(net):
    net.route("POST", authcd_auth.CLAUDE_TOKEN_URL, lambda record: _Resp(200, CD_TOKENS))
    return net


def _authorize_query_of(url):
    """The authorize query inside the sign-out-first URL (claude.ai/logout?returnTo=/oauth/authorize?...)."""
    parsed = urlparse(url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == authcd_auth.CLAUDE_AI_LOGOUT_URL
    return_to = parse_qs(parsed.query)["returnTo"][0]
    assert return_to.startswith(authcd_auth.CLAUDE_AI_AUTHORIZE_PATH + "?")
    return {k: v[0] for k, v in parse_qs(return_to.split("?", 1)[1]).items()}


def _cd_exchange(code, verifier, redirect_uri, state):
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": authcd_auth.CLAUDE_CLIENT_ID,
        "code_verifier": verifier,
    }
    if state:
        payload["state"] = state
    return {"method": "POST", "url": authcd_auth.CLAUDE_TOKEN_URL, "json": payload,
            "headers": dict(authcd_auth._TOKEN_ENDPOINT_HEADERS), "timeout": 30}


@pytest.mark.parametrize("return_url", [None, RETURN_URL])
def test_cd_begin_localhost_then_complete_saves_tokens(cd_net, no_browser, tmp_path, monkeypatch, return_url):
    if return_url:
        monkeypatch.setenv("GLOSSARION_OAUTH_RETURN_URL", return_url)
    token_file = tmp_path / "authcd_tokens_3.json"
    store = authcd_auth.AuthCDTokenStore(token_file=str(token_file), account_id=3)
    session = authcd_auth.begin_oauth(store, timeout=30)
    try:
        port = session.port
        assert session.redirect_uri == f"http://localhost:{port}/callback"
        query = _authorize_query_of(session.auth_url)
        assert query == {
            "code": "true",
            "client_id": authcd_auth.CLAUDE_CLIENT_ID,
            "response_type": "code",
            "redirect_uri": session.redirect_uri,
            "scope": authcd_auth.CLAUDE_CODE_LOGIN_SCOPES,
            "code_challenge": _s256(session.code_verifier),
            "code_challenge_method": "S256",
            "state": session.state,
        }
        manual = _authorize_query_of(session.manual_auth_url)
        assert manual == dict(query, redirect_uri=authcd_auth.CLAUDE_REDIRECT_URI)
        assert session.manual_redirect_uri == authcd_auth.CLAUDE_REDIRECT_URI
        assert session.server_running and session.account_id == 3

        raw = Path(store.pending_oauth_file).read_bytes()
        assert raw.startswith(b"GLSE1:") and session.code_verifier.encode() not in raw
        pending = store.load_pending_oauth()
        assert pending["provider"] == "authcd" and pending["code_verifier"] == session.code_verifier
        assert pending["manual_redirect_uri"] == authcd_auth.CLAUDE_REDIRECT_URI
        assert pending["manual_auth_url"] == session.manual_auth_url

        status, headers, body = _http_get("127.0.0.1", port, f"/callback?code=cd-code&state={session.state}")
        if return_url:
            assert status == 200 and "glossarion://app/oauth/return?p=authcd" in body
        else:
            assert status == 302 and headers["Location"] == authcd_auth.CLAUDE_AI_SUCCESS_URL
        assert session.wait_for_callback(5)
        assert session.wait(5) and not session.server_running  # one callback ends the sign-in
        tokens = authcd_auth.complete_from_redirect(session)
    finally:
        session.close()

    assert no_browser == []
    assert cd_net.calls == [_cd_exchange("cd-code", session.code_verifier, session.redirect_uri, session.state)]
    assert tokens["access_token"] == "cd-at" and tokens["_source"] == "glossarion_oauth"
    assert authcd_auth.AuthCDTokenStore(token_file=str(token_file), account_id=3).load_tokens()["refresh_token"] == "cd-rt"
    assert store.load_pending_oauth() is None
    assert _port_is_free(port)


@pytest.mark.parametrize(
    "paste_kind",
    ["code#state", "bare code", "localhost URL", "code page URL"],
)
def test_cd_paste_fallbacks_send_the_matching_redirect_uri(cd_net, tmp_path, paste_kind):
    token_file = tmp_path / "authcd_tokens.json"
    store = authcd_auth.AuthCDTokenStore(token_file=str(token_file))
    session = authcd_auth.begin_oauth(store, timeout=30)
    session.close()
    fresh = authcd_auth.AuthCDTokenStore(token_file=str(token_file))
    pasted, redirect_uri = {
        "code#state": (f"pasted-code#{session.state}", authcd_auth.CLAUDE_REDIRECT_URI),
        "bare code": ("pasted-code", authcd_auth.CLAUDE_REDIRECT_URI),
        "localhost URL": (f"{session.redirect_uri}?code=pasted-code&state={session.state}", session.redirect_uri),
        "code page URL": (
            f"{authcd_auth.CLAUDE_REDIRECT_URI}?code=pasted-code&state={session.state}", authcd_auth.CLAUDE_REDIRECT_URI,
        ),
    }[paste_kind]
    tokens = authcd_auth.complete_from_redirect(fresh, f"  {pasted}\n")
    assert cd_net.calls == [_cd_exchange("pasted-code", session.code_verifier, redirect_uri, session.state)]
    assert tokens["_source"] == "glossarion_oauth" and fresh.load_tokens()["access_token"] == "cd-at"
    assert fresh.load_pending_oauth() is None


def test_cd_paste_with_live_session_frees_the_port(cd_net, tmp_path):
    store = authcd_auth.AuthCDTokenStore(token_file=str(tmp_path / "authcd_tokens.json"))
    session = authcd_auth.begin_oauth(store, timeout=30)
    port = session.port
    authcd_auth.complete_from_redirect(session, f"live-code#{session.state}")
    assert not session.server_running and _port_is_free(port)
    assert cd_net.calls[0]["json"]["code"] == "live-code"


def test_cd_paste_rejections(cd_net, tmp_path):
    store = authcd_auth.AuthCDTokenStore(token_file=str(tmp_path / "authcd_tokens.json"))
    with pytest.raises(RuntimeError, match="No pending Claude sign-in"):
        authcd_auth.complete_from_redirect(store, "c#s")
    session = authcd_auth.begin_oauth(store, timeout=30)
    session.close()
    with pytest.raises(RuntimeError, match="state mismatch"):
        authcd_auth.complete_from_redirect(store, "evil#forged-state")
    with pytest.raises(RuntimeError, match="AuthCD: access_denied"):
        authcd_auth.complete_from_redirect(store, f"{session.redirect_uri}?error=access_denied")
    with pytest.raises(RuntimeError, match="no authorization code"):
        authcd_auth.complete_from_redirect(store, f"?state={session.state}")
    assert cd_net.calls == [] and store.load_tokens() is None
    assert store.load_pending_oauth()["state"] == session.state  # still pasteable


def test_cd_loopback_errors_end_the_sign_in(cd_net, tmp_path):
    store = authcd_auth.AuthCDTokenStore(token_file=str(tmp_path / "authcd_tokens.json"))
    session = authcd_auth.begin_oauth(store, timeout=30)
    status, _, body = _http_get("127.0.0.1", session.port, "/callback?code=x&state=forged")
    assert status == 400 and body == "Claude sign-in failed: state mismatch in the sign-in callback"
    assert session.wait_for_callback(5)
    with pytest.raises(RuntimeError, match="AuthCD: state mismatch in the sign-in callback"):
        authcd_auth.complete_from_redirect(session)
    session.close()
    timed_out = authcd_auth.begin_oauth(timeout=30, persist=False)
    timed_out.close()
    with pytest.raises(RuntimeError, match="timed out"):
        authcd_auth.complete_from_redirect(timed_out)
    assert cd_net.calls == []


def test_cd_legacy_paste_path_splits_code_and_state(cd_net, monkeypatch):
    # The URL Anthropic's code page answers: Claude Code's manual login (code=true, CLI scopes).
    query = _query(authcd_auth.build_auth_url("challenge", "st"))
    assert query["code"] == "true" and query["redirect_uri"] == authcd_auth.CLAUDE_REDIRECT_URI
    assert query["scope"] == authcd_auth.CLAUDE_CODE_LOGIN_SCOPES == authcd_auth.SCOPES
    assert query["code_challenge"] == "challenge" and query["state"] == "st"
    assert authcd_auth.build_auth_url("c", "s").startswith(authcd_auth.CLAUDE_AUTH_URL + "?")

    authcd_auth.complete_oauth_exchange("the-code#the-state", "verifier", "the-state")
    assert cd_net.calls[-1]["json"] == _cd_exchange("the-code", "verifier", authcd_auth.CLAUDE_REDIRECT_URI, "the-state")["json"]
    authcd_auth.complete_oauth_exchange("  the-code#the-state ", "verifier")
    assert cd_net.calls[-1]["json"]["state"] == "the-state" and cd_net.calls[-1]["json"]["code"] == "the-code"
    authcd_auth.complete_oauth_exchange("bare-code", "verifier", "expected")
    assert cd_net.calls[-1]["json"]["state"] == "expected"
    authcd_auth.complete_oauth_exchange("bare-code", "verifier")  # desktop compat: nothing to send
    assert "state" not in cd_net.calls[-1]["json"] and cd_net.calls[-1]["json"]["code"] == "bare-code"
    with pytest.raises(RuntimeError, match="state mismatch"):
        authcd_auth.complete_oauth_exchange("evil#other-state", "verifier", "the-state")

    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url, *a, **k: opened.append(url) or True)
    monkeypatch.setattr("builtins.input", lambda prompt="": f"stdin-code#{_query(opened[-1])['state']}")
    tokens = authcd_auth.run_oauth_flow()
    sent = cd_net.calls[-1]["json"]
    assert sent["code"] == "stdin-code" and sent["state"] == _query(opened[-1])["state"]
    assert sent["code_verifier"] and tokens["access_token"] == "cd-at"


def test_cd_run_automatic_login_composes_the_split(cd_net, monkeypatch):
    calls = []
    real_begin = authcd_auth.begin_oauth

    def spy(*args, **kwargs):
        session = real_begin(*args, **kwargs)
        calls.append((args, kwargs, session))
        return session

    monkeypatch.setattr(authcd_auth, "begin_oauth", spy)
    tokens = authcd_auth.run_automatic_oauth_login(timeout=20, open_browser=_cd_browser([]), sign_out_first=False)
    args, kwargs, session = calls[0]
    assert args == () and kwargs == {
        "timeout": 20, "persist": False, "sign_out_first": False,
        "dual_stack": False, "auto_close": False, "serve": False,
    }
    assert session.store is None and len(session._servers) == 1 and session.closed
    assert tokens["access_token"] == "cd-at" and _port_is_free(session.port)


def _cd_browser(pages, state=None, query=None):
    """Fake browser for the Claude localhost flow (follows claude.ai/logout?returnTo=...)."""

    def browser(url):
        if url.startswith(authcd_auth.CLAUDE_AI_LOGOUT_URL):
            url = "https://claude.ai" + parse_qs(urlparse(url).query)["returnTo"][0]
        params = _query(url)
        port = urlparse(params["redirect_uri"]).port
        path = "/callback?" + (query or "code=auth-code-123&state=" + (state or params["state"]))

        threading.Thread(target=lambda: pages.append(_http_get("127.0.0.1", port, path)), daemon=True).start()
        return True

    return browser


@pytest.mark.parametrize(
    "scenario",
    ["success", "forged state", "error description", "no code", "cancelled", "timeout"],
)
def test_cd_run_automatic_login_matches_legacy(cd_net, fixed_secrets, monkeypatch, scenario):
    legacy = legacy_module("authcd_auth")
    results = {}
    for name, module in (("legacy", legacy), ("split", authcd_auth)):
        cd_net.calls.clear()
        pages = []
        opened = []
        monkeypatch.delenv("TRANSLATION_CANCELLED", raising=False)
        module.reset_cancel()
        browser = {
            "success": _cd_browser(pages),
            "forged state": _cd_browser(pages, state="forged"),
            "error description": _cd_browser(
                pages, query="error=access_denied&error_description=User+said+no&state=fixed-state-0123456789",
            ),
            "no code": _cd_browser(pages, query="state=fixed-state-0123456789"),
            "cancelled": lambda url: True,
            "timeout": lambda url: True,
        }[scenario]

        def open_browser(url, browser=browser):
            opened.append(url)
            if scenario == "cancelled":
                module.cancel_stream()
            return browser(url)

        try:
            outcome = ("tokens", {k: v for k, v in module.run_automatic_oauth_login(
                timeout=0.6 if scenario == "timeout" else 20, open_browser=open_browser, sign_out_wait=0,
            ).items() if k != "expires_at"})
        except Exception as exc:
            outcome = ("error", type(exc).__name__, str(exc))
        finally:
            module.reset_cancel()
        _wait_until(lambda: len(pages) >= (1 if scenario not in ("cancelled", "timeout") else 0))
        port = urlparse(_authorize_query_of(opened[0])["redirect_uri"]).port
        results[name] = _normalize({"outcome": outcome, "opened": opened, "pages": pages, "calls": list(cd_net.calls)}, [port])
    assert results["split"] == results["legacy"]
    expected = {"success": "tokens", "timeout": "error"}.get(scenario, "error")
    assert results["split"]["outcome"][0] == expected
    if scenario == "timeout":
        assert results["split"]["outcome"][1] == "TimeoutError"


def test_cd_mobile_never_runs_or_imports_the_claude_cli(monkeypatch, tmp_path):
    import shutil

    creds = tmp_path / ".credentials.json"
    creds.write_text(json.dumps({"claudeAiOauth": {"accessToken": "cli-at", "refreshToken": "cli-rt",
                                                   "expiresAt": 1893456000000}}), encoding="utf-8")
    monkeypatch.setattr(authcd_auth, "_CLAUDE_CODE_CREDS", str(creds))
    monkeypatch.setattr(authcd_auth, "_GLOSSARION_CLAUDE_CODE_CREDS", str(tmp_path / "none.json"))
    monkeypatch.setattr(authcd_auth, "_load_claude_code_credman_json", lambda config_dir=None: None)
    monkeypatch.setattr(authcd_auth, "_load_claude_code_keychain_json", lambda config_dir=None: None)
    monkeypatch.setattr(authcd_auth, "GLOSSARION_CLAUDE_CONFIG_DIR", str(tmp_path / "claude-code"))

    # Desktop: unchanged (reads the CLI login, asks the CLI).
    assert authcd_auth._load_claude_code_credentials()["access_token"] == "cli-at"
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: seen.append(a) or subprocess.CompletedProcess(a, 0, "{}", ""))
    assert authcd_auth.claude_cli_auth_status("claude") == {}
    assert len(seen) == 1

    _mobile(monkeypatch)

    def no_process(*args, **kwargs):
        raise AssertionError(f"process started on mobile: {args!r}")

    monkeypatch.setattr(subprocess, "Popen", no_process)
    monkeypatch.setattr(subprocess, "run", no_process)
    monkeypatch.setattr(shutil, "which", no_process)
    with pytest.raises(RuntimeError, match="not available on Glossarion Mobile"):
        authcd_auth.run_claude_cli_login("claude")
    assert authcd_auth.claude_cli_auth_status("claude") is None
    assert authcd_auth._installed_claude_code_version() is None
    assert authcd_auth._load_claude_code_credentials() is None
    assert authcd_auth.import_claude_code_login(authcd_auth.AuthCDTokenStore(token_file=str(tmp_path / "t.json"))) is None
    default_store = authcd_auth.AuthCDTokenStore(token_file=str(tmp_path / "authcd_tokens.json"), account_id=0)
    assert default_store.load_tokens() is None and not default_store.has_tokens
    assert "Claude login completed" in authcd_auth.claude_login_failure_message("claude")


# ===========================================================================
# authgrok_auth
# ===========================================================================

DEVICE_CODE = {
    "device_code": "device-SECRET-code",
    "user_code": "ABCD-EFGH",
    "verification_uri": "https://accounts.x.ai/oauth2/device",
    "verification_uri_complete": "https://accounts.x.ai/oauth2/device?user_code=ABCD-EFGH",
    "expires_in": 600,
    "interval": 1,
}
GROK_TOKENS = {"access_token": "grok-at", "refresh_token": "grok-rt", "id_token": "h.p.s", "expires_in": 3600}
GROK_CLAIMS = {"email": "grok@example.test", "name": "Grok User", "sub": "sub-1"}


@pytest.fixture
def xai(net, monkeypatch):
    """Fake xAI: device-code endpoint plus a token endpoint answering from ``xai.token_replies``."""
    net.token_replies = []

    def token(record):
        status, body = net.token_replies.pop(0) if net.token_replies else (200, GROK_TOKENS)
        return _Resp(status, body)

    net.route("POST", authgrok_auth.XAI_OAUTH_DEVICE_CODE_URL, lambda record: _Resp(200, DEVICE_CODE))
    net.route("POST", authgrok_auth.XAI_OAUTH_TOKEN_URL, token)
    net.discovery = {"jwks_uri": authgrok_auth.XAI_OAUTH_JWKS_URL, "issuer": authgrok_auth.XAI_OAUTH_ISSUER}
    monkeypatch.setattr(authgrok_auth, "_load_oidc_discovery", lambda *a, **k: dict(net.discovery))
    monkeypatch.setattr(authgrok_auth, "_warm_jwks_cache", lambda discovery: None)
    net.validated = []

    def validate(id_token, nonce, discovery, timeout=15):
        net.validated.append((id_token, nonce, discovery))
        return dict(GROK_CLAIMS)

    monkeypatch.setattr(authgrok_auth, "_validate_id_token", validate)
    return net


def test_grok_device_login_on_mobile_saves_the_slot(xai, no_browser, monkeypatch, tmp_path):
    _mobile(monkeypatch)
    slot_checks = []
    monkeypatch.setattr(authgrok_auth, "validate_account_slot_tokens", lambda *a: slot_checks.append(a))
    token_file = tmp_path / "authgrok_tokens_2.json"
    store = authgrok_auth.AuthGrokTokenStore(str(token_file), 2)
    before = time.time()
    session = authgrok_auth.begin_device_login(store=store, account_id=2, timeout=60)

    assert session.user_code == "ABCD-EFGH" and session.interval == 1
    assert session.verification_uri == DEVICE_CODE["verification_uri"]
    assert session.verification_uri_complete == DEVICE_CODE["verification_uri_complete"]
    assert session.open_url == session.signed_out_url == authgrok_auth.build_signed_out_device_url(
        DEVICE_CODE["verification_uri_complete"])
    assert before + 600 <= session.expires == session.expires_at <= time.time() + 600
    assert xai.calls[0]["url"] == authgrok_auth.XAI_OAUTH_DEVICE_CODE_URL and xai.calls[0]["timeout"] == 30

    raw = Path(store.pending_oauth_file).read_bytes()
    assert raw.startswith(b"GLSE1:") and b"device-SECRET-code" not in raw
    pending = store.load_pending_oauth()
    assert pending["flow"] == "device" and pending["device_code"]["device_code"] == "device-SECRET-code"

    xai.token_replies[:] = [(400, {"error": "authorization_pending"})]
    tokens = authgrok_auth.poll_device_login(session, should_stop=lambda: False)
    polls = [c for c in xai.calls if c["url"] == authgrok_auth.XAI_OAUTH_TOKEN_URL]
    assert len(polls) == 2 and polls[0]["data"] == {
        "grant_type": authgrok_auth.DEVICE_CODE_GRANT_TYPE,
        "device_code": "device-SECRET-code",
        "client_id": authgrok_auth.XAI_OAUTH_CLIENT_ID,
    }
    assert xai.validated == [("h.p.s", None, xai.discovery)]
    assert tokens["account"] == {"email": "grok@example.test", "name": "Grok User", "subject": "sub-1"}
    assert slot_checks == [(2, tokens)]
    saved = authgrok_auth.AuthGrokTokenStore(str(token_file), 2)
    assert saved.load_tokens()["access_token"] == "grok-at" and saved.account_info["email"] == "grok@example.test"
    assert store.load_pending_oauth() is None
    assert no_browser == []


def test_grok_device_login_can_be_cancelled(xai, monkeypatch, tmp_path):
    store = authgrok_auth.AuthGrokTokenStore(str(tmp_path / "authgrok_tokens.json"))
    session = authgrok_auth.begin_device_login(store=store, timeout=60)
    xai.token_replies[:] = [(400, {"error": "authorization_pending"})] * 50
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="cancelled"):
        authgrok_auth.poll_device_login(session, should_stop=lambda: time.monotonic() - started > 0.3)
    assert time.monotonic() - started < 2.5
    assert store.load_pending_oauth() is None and store.load_tokens() is None

    session = authgrok_auth.begin_device_login(store=store, timeout=60)
    threading.Timer(0.3, session.close).start()
    with pytest.raises(RuntimeError, match="cancelled"):
        authgrok_auth.poll_device_login(session, should_stop=lambda: False)


def test_grok_device_login_denied_or_duplicate_account(xai, monkeypatch, tmp_path):
    store = authgrok_auth.AuthGrokTokenStore(str(tmp_path / "authgrok_tokens_5.json"), 5)
    session = authgrok_auth.begin_device_login(store=store, timeout=60)
    xai.token_replies[:] = [(400, {"error": "access_denied"})]
    with pytest.raises(RuntimeError, match="denied"):
        authgrok_auth.poll_device_login(session, should_stop=lambda: False)
    assert store.load_pending_oauth() is None

    def duplicate(account_id, tokens):
        raise RuntimeError("grok@example.test is already saved in Grok account slot #1.")

    monkeypatch.setattr(authgrok_auth, "validate_account_slot_tokens", duplicate)
    session = authgrok_auth.begin_device_login(store=store, timeout=60)
    with pytest.raises(RuntimeError, match="already saved"):
        authgrok_auth.poll_device_login(session, should_stop=lambda: False)
    assert store.load_tokens() is None


def test_grok_resume_device_login_after_restart(xai, monkeypatch, tmp_path):
    token_file = tmp_path / "authgrok_tokens.json"
    store = authgrok_auth.AuthGrokTokenStore(str(token_file))
    assert authgrok_auth.resume_device_login(store) is None
    session = authgrok_auth.begin_device_login(store=store, account_id=0, timeout=60)
    fresh = authgrok_auth.AuthGrokTokenStore(str(token_file))
    resumed = authgrok_auth.resume_device_login(fresh, timeout=60)
    assert resumed.user_code == session.user_code and resumed.open_url == session.open_url
    assert resumed.expires_at == pytest.approx(session.expires_at)
    assert resumed.device_code == session.device_code and resumed.store is fresh
    monkeypatch.setattr(authgrok_auth, "validate_account_slot_tokens", lambda *a: None)
    assert authgrok_auth.poll_device_login(resumed, should_stop=lambda: False)["access_token"] == "grok-at"
    assert fresh.load_tokens()["refresh_token"] == "grok-rt"

    fresh.save_pending_oauth(dict(session.pending_state(), expires_at=time.time() - 1))
    assert authgrok_auth.resume_device_login(fresh) is None
    fresh.save_pending_oauth({"provider": "authgem", "state": "x", "created_at": time.time()})
    assert authgrok_auth.resume_device_login(fresh) is None


def test_grok_mobile_never_opens_a_browser_or_reads_the_cli_login(xai, no_browser, monkeypatch, tmp_path):
    cli = tmp_path / "auth.json"
    cli.write_text(json.dumps({"access_token": "cli-at", "refresh_token": "cli-rt"}), encoding="utf-8")
    monkeypatch.setattr(authgrok_auth, "_OFFICIAL_GROK_AUTH_FILE", str(cli))
    assert authgrok_auth.load_grok_cli_credentials()["access_token"] == "cli-at"  # desktop unchanged

    _mobile(monkeypatch)
    assert authgrok_auth.load_grok_cli_credentials() is None
    assert authgrok_auth.load_grok_cli_credentials(str(cli))["access_token"] == "cli-at"  # explicit file only
    for flow in (lambda: authgrok_auth.run_oauth_flow(), lambda: authgrok_auth.run_device_oauth_flow(),
                 lambda: authgrok_auth.run_oauth_flow(force_account_selection=False),
                 lambda: authgrok_auth._open_oauth_browser("https://accounts.x.ai/")):
        with pytest.raises(RuntimeError, match="Accounts screen"):
            flow()
    store = authgrok_auth.AuthGrokTokenStore(str(tmp_path / "authgrok_tokens.json"), 0)
    with pytest.raises(RuntimeError, match="Accounts screen"):
        store.get_valid_access_token(auto_login=True)
    assert store.load_tokens() is None
    assert xai.calls == [] and no_browser == []


def test_grok_run_device_flow_composes_the_split(xai, monkeypatch):
    calls = []
    real_begin = authgrok_auth.begin_device_login

    def spy(*args, **kwargs):
        session = real_begin(*args, **kwargs)
        calls.append((args, kwargs, session))
        return session

    opened = []
    monkeypatch.setattr(authgrok_auth, "begin_device_login", spy)
    monkeypatch.setattr(authgrok_auth, "_open_oauth_browser", opened.append)
    monkeypatch.setattr(authgrok_auth.time, "sleep", lambda seconds: None)
    tokens = authgrok_auth.run_device_oauth_flow(timeout=40)
    args, kwargs, session = calls[0]
    assert (args, kwargs) == ((), {"timeout": 40, "persist": False, "warm_jwks": False})
    assert opened == [session.open_url] and session.store is None
    assert tokens["account"]["email"] == "grok@example.test"


def _grok_stub(module, monkeypatch, record, warmed, poll_result=None):
    monkeypatch.setattr(module, "_load_oidc_discovery", lambda *a, **k: record.append(("discovery",)) or {"jwks_uri": "J"})
    monkeypatch.setattr(module, "request_device_code",
                        lambda timeout=30: record.append(("device_code", timeout)) or dict(DEVICE_CODE))
    monkeypatch.setattr(module, "_warm_jwks_cache", lambda discovery: warmed.append(discovery))
    monkeypatch.setattr(module, "_open_oauth_browser", lambda url: record.append(("open", url)))

    def poll(device_code, timeout=300):
        record.append(("poll", device_code, timeout))
        if isinstance(poll_result, Exception):
            raise poll_result
        return dict(GROK_TOKENS)

    monkeypatch.setattr(module, "poll_device_code_tokens", poll)
    monkeypatch.setattr(module, "_validate_id_token",
                        lambda id_token, nonce, discovery, timeout=15: record.append(("validate", id_token, nonce, discovery))
                        or dict(GROK_CLAIMS))


@pytest.mark.parametrize("poll_result", [None, RuntimeError("xAI device authorization was denied")])
def test_grok_run_device_flow_matches_legacy(monkeypatch, capsys, poll_result):
    legacy = legacy_module("authgrok_auth")
    results = {}
    for name, module in (("legacy", legacy), ("split", authgrok_auth)):
        record, warmed = [], []
        _grok_stub(module, monkeypatch, record, warmed, poll_result)
        try:
            outcome = ("tokens", module.run_device_oauth_flow(timeout=123))
        except Exception as exc:
            outcome = ("error", type(exc).__name__, str(exc))
        _wait_until(lambda: warmed)
        results[name] = {"outcome": outcome, "record": record, "warmed": warmed, "out": capsys.readouterr().out}
    assert results["split"] == results["legacy"]


POLL_SCRIPTS = {
    "pending then ok": [(400, {"error": "authorization_pending"}), (200, GROK_TOKENS)],
    "slow down": [(400, {"error": "slow_down"}), (400, {"error": "authorization_pending"}), (200, GROK_TOKENS)],
    "denied": [(400, {"error": "access_denied"})],
    "expired": [(400, {"error": "expired_token"})],
    "other": [(400, {"error": "server_error", "error_description": "boom"})],
    "no refresh token": [(200, {"access_token": "a", "id_token": "i"})],
}


@pytest.mark.parametrize("script", sorted(POLL_SCRIPTS))
def test_grok_poll_device_code_tokens_matches_legacy(net, monkeypatch, script):
    legacy = legacy_module("authgrok_auth")
    results = {}
    for name, module in (("legacy", legacy), ("split", authgrok_auth)):
        replies = list(POLL_SCRIPTS[script])
        net.calls.clear()
        net.routes.clear()
        net.route("POST", authgrok_auth.XAI_OAUTH_TOKEN_URL, lambda record: _Resp(*replies.pop(0)))
        sleeps = []
        monkeypatch.setattr(module.time, "sleep", sleeps.append)
        try:
            outcome = ("tokens", {k: v for k, v in module.poll_device_code_tokens(
                {"device_code": "dc", "expires_in": 600, "interval": 2}, timeout=300).items() if k != "expires_at"})
        except Exception as exc:
            outcome = ("error", type(exc).__name__, str(exc))
        results[name] = {"outcome": outcome, "sleeps": [round(s) for s in sleeps], "calls": list(net.calls)}
    assert results["split"] == results["legacy"]


@pytest.mark.parametrize("return_url", [None, RETURN_URL])
def test_grok_pkce_callback_page_seam_and_parity(xai, monkeypatch, fixed_secrets, return_url):
    """run_oauth_flow(force_account_selection=False): the success page is legacy's, or returns to the app."""
    if return_url:
        monkeypatch.setenv("GLOSSARION_OAUTH_RETURN_URL", return_url)
    legacy = legacy_module("authgrok_auth")
    results = {}
    for name, module in (("legacy", legacy), ("split", authgrok_auth)):
        pages = []
        xai.calls.clear()
        monkeypatch.setattr(module, "_load_oidc_discovery", lambda *a, **k: {"authorization_endpoint": authgrok_auth.XAI_OAUTH_AUTHORIZATION_URL})
        monkeypatch.setattr(module, "_validate_id_token", lambda *a, **k: dict(GROK_CLAIMS))

        def browser(url):
            params = _query(url)
            port = urlparse(params["redirect_uri"]).port
            threading.Thread(target=lambda: pages.append(
                _http_get("127.0.0.1", port, f"/callback?code=pkce-code&state={params['state']}")), daemon=True).start()
            return True

        monkeypatch.setattr(module, "_open_oauth_browser", browser)
        tokens = module.run_oauth_flow(timeout=20, force_account_selection=False)
        _wait_until(lambda: pages)
        port = urlparse(xai.calls[-1]["data"]["redirect_uri"]).port
        results[name] = _normalize({"tokens": {k: v for k, v in tokens.items() if k != "expires_at"},
                                    "pages": pages, "calls": list(xai.calls)}, [port])
    status, _, body = results["split"]["pages"][0]
    assert status == 200
    if return_url:
        assert "glossarion://app/oauth/return?p=authgrok" in body
        assert results["legacy"]["pages"][0][2] == LEGACY_GROK_SUCCESS_HTML
        results["split"]["pages"] = results["legacy"]["pages"]
    else:
        assert body == LEGACY_GROK_SUCCESS_HTML
    assert results["split"] == results["legacy"]


# ---------------------------------------------------------------------------
# U7: the GCP project picker's list and selection rule (authgem_auth chooser)
# ---------------------------------------------------------------------------

#: translator_gui before the U7 move of _authgem_projects_loaded's selection rule into authgem_auth.
U7_BASE_SHA = "41814faa95e273e956870bd3aac5a2c6fb7d66b1"


class _FakeProjectCombo:
    """The QComboBox surface _authgem_projects_loaded uses (items, findData, current index)."""

    def __init__(self, log):
        self.items, self.index, self.log = [], -1, log

    def blockSignals(self, value):
        self.log.append(("blockSignals", value))

    def clear(self):
        self.items, self.index = [], -1

    def addItem(self, label, data):
        self.items.append((label, data))
        if self.index < 0:
            self.index = 0

    def findData(self, data):
        return next((i for i, (_label, d) in enumerate(self.items) if d == data), -1)

    def setCurrentIndex(self, index):
        self.index = index

    def currentIndex(self):
        return self.index

    def count(self):
        return len(self.items)

    def show(self):
        self.log.append(("show",))

    def hide(self):
        self.log.append(("hide",))


def _projects_loaded_function(text):
    import ast

    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TranslatorGUI")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_authgem_projects_loaded")
    lines = text.split("\n")
    source = "class _Probe:\n" + "\n".join(lines[node.lineno - 1:node.end_lineno]) + "\n"
    ns = {"os": os, "Slot": lambda *a, **k: (lambda f: f), "__name__": "translator_gui"}
    exec(compile(source, "<_authgem_projects_loaded>", "exec"), ns)
    return vars(ns["_Probe"])["_authgem_projects_loaded"]


def test_authgem_project_picker_rule_matches_the_desktop_slot():
    """The working-tree slot (list + rule from authgem_auth) shows and selects exactly what the
    U7 parent's inline code did, for random billing results and saved projects."""
    import random

    legacy_text = subprocess.run(["git", "show", f"{U7_BASE_SHA}:src/translator_gui.py"], cwd=str(REPO_ROOT),
                                 capture_output=True).stdout.decode("utf-8-sig").replace("\r\n", "\n")
    if not legacy_text:
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            pytest.fail(f"git show {U7_BASE_SHA[:12]} unavailable")
        pytest.skip(f"git show {U7_BASE_SHA[:12]} unavailable")
    current_text = (SRC / "translator_gui.py").read_bytes().decode("utf-8-sig").replace("\r\n", "\n")
    sides = {"legacy": _projects_loaded_function(legacy_text), "new": _projects_loaded_function(current_text)}
    assert "authgem_auth.choose_authgem_project_index(" in current_text
    rng = random.Random("u7-authgem-projects")
    pool = [f"proj-{i}" for i in range(7)]
    for state in range(600):
        ids = rng.sample(pool, rng.randint(0, 6))
        billed, unbilled, unknown = [], [], []
        for pid in ids:
            rng.choice((billed, unbilled, unknown)).append(pid)
        saved = rng.choice(["", None, "elsewhere"] + pool)
        has_combo = rng.random() > 0.05
        observed = {}
        for side, fn in sides.items():
            log = []
            owner = types.SimpleNamespace(
                config={"authgem_project": saved} if saved is not None else {},
                _authgem_billed_projects=list(billed), _authgem_unbilled_projects=list(unbilled),
                _authgem_unknown_projects=list(unknown),
                _authgem_project_changed=lambda index, _log=log: _log.append(("changed", index)),
                _reposition_authgem_project_combo=lambda _log=log: _log.append(("reposition",)),
                append_log=lambda message, _log=log: _log.append(("log", message)))
            if has_combo:
                owner.authgem_project_combo = _FakeProjectCombo(log)
            fn(owner)
            combo = getattr(owner, "authgem_project_combo", None)
            observed[side] = (log, combo.items if combo else None, combo.index if combo else None)
        assert observed["legacy"] == observed["new"], (state, billed, unbilled, unknown, saved)
    # the shared helpers on their own (the mobile picker calls exactly these)
    items = authgem_auth.authgem_project_items(["b1"], ["u1"], ["k1", "k2"])
    assert items == [("✅ b1", "b1"), ("❔ k1", "k1"), ("❔ k2", "k2"), ("⚠️ u1 (no billing)", "u1")]
    assert authgem_auth.choose_authgem_project_index("k2", ["b1"], ["u1"], ["k1", "k2"]) == 2
    assert authgem_auth.choose_authgem_project_index("u1", ["b1"], ["u1"], ["k1"]) == 0  # known unbilled: first billed
    assert authgem_auth.choose_authgem_project_index("u1", [], ["u1"], ["k1"]) == 0  # else the first unknown
    assert authgem_auth.choose_authgem_project_index("", [], ["u1"], []) == -1
