"""authcd sign-in in a private window of the default browser instead of claude.ai/logout (which now
asks "Log out of Claude?"); unknown browsers fall back to the logout page."""
import os
import sys
from urllib.parse import unquote

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import authcd_auth  # noqa: E402
import oauth_session  # noqa: E402


@pytest.mark.parametrize("name,path,flag", [
    ("chrome.exe", "", "--incognito"),
    ("brave.exe", "", "--incognito"),
    ("vivaldi.exe", "", "--incognito"),
    ("msedge.exe", "", "--inprivate"),
    ("opera.exe", "", "--private"),
    ("launcher.exe", r"C:\Users\u\AppData\Local\Programs\Opera GX\launcher.exe", "--private"),  # Opera GX
    ("launcher.exe", r"C:\Tools\launcher.exe", None),  # some other launcher
    ("firefox.exe", "", "-private-window"),
    ("librewolf.exe", "", "-private-window"),
    ("com.operasoftware.OperaGX", "", "--private"),
    ("com.google.Chrome", "", "--incognito"),
    ("com.apple.Safari", "", None),  # no private-window flag: the logout page stays
    ("brave-browser.desktop", "", "--incognito"),
    ("firefox.desktop", "", "-private-window"),
    ("someotherbrowser.exe", "", None),
])
def test_private_window_flag_per_browser(name, path, flag):
    assert oauth_session._private_flag(name, path) == flag


def test_windows_default_browser_command(monkeypatch, tmp_path):
    exe = tmp_path / "Opera GX" / "launcher.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")

    class Key:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    values = {"UserChoice": "OperaGXStable", r"OperaGXStable\shell\open\command":
              f'"{exe}" -noautoupdate -- "%1"'}

    class FakeWinreg:
        HKEY_CURRENT_USER, HKEY_CLASSES_ROOT = 1, 2

        @staticmethod
        def OpenKey(_root, key):
            return Key(next(v for k, v in values.items() if key.endswith(k)))

        @staticmethod
        def QueryValueEx(handle, _name):
            return handle.value, 1

    monkeypatch.setitem(sys.modules, "winreg", FakeWinreg)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(oauth_session, "is_mobile", lambda: False)
    assert oauth_session.private_window_command() == [str(exe), "--private"]


def test_no_private_window_on_mobile(monkeypatch):
    monkeypatch.setattr(oauth_session, "is_mobile", lambda: True)
    assert oauth_session.private_window_command() is None
    assert oauth_session.open_private_window("https://claude.ai") is False


def test_private_sign_in_url_skips_the_logout(monkeypatch):
    session = authcd_auth.begin_oauth(persist=False, dual_stack=False, serve=False)
    try:
        assert session.auth_url.startswith(authcd_auth.CLAUDE_AI_LOGOUT_URL + "?returnTo=")
        assert session.private_auth_url.startswith(authcd_auth.CLAUDE_AI_AUTHORIZE_URL + "?")
        # the same authorize request: the logout page's returnTo is the private window's URL path
        assert unquote(session.auth_url.split("returnTo=", 1)[1]) in session.private_auth_url
    finally:
        session.close()


def _run_login(monkeypatch, private_ok):
    opened = []

    class Session:
        auth_url, private_auth_url, callback_received = "LOGOUT_URL", "PRIVATE_URL", True

        def close(self):
            pass

    monkeypatch.setattr(authcd_auth, "begin_oauth", lambda **_k: Session())
    monkeypatch.setattr(authcd_auth, "complete_from_redirect", lambda s: {"ok": True})
    monkeypatch.setattr(oauth_session, "open_private_window", lambda url: (opened.append(("private", url)), private_ok)[1])
    monkeypatch.setattr(authcd_auth.webbrowser, "open", lambda url: opened.append(("browser", url)))
    authcd_auth.run_automatic_oauth_login(timeout=5)
    return opened


def test_login_opens_a_private_window_without_the_logout(monkeypatch):
    assert _run_login(monkeypatch, True) == [("private", "PRIVATE_URL")]


def test_unknown_browser_falls_back_to_the_logout_page(monkeypatch):
    assert _run_login(monkeypatch, False) == [("private", "PRIVATE_URL"), ("browser", "LOGOUT_URL")]
