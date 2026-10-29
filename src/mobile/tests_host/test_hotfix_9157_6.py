"""v9.15.7 hotfix 6, owner (2026-10-29, phone, a Grok sign-in): "Grok needs to auto close this page, as login was
already successful. It just confuses the user on a page that never loads" - the 127.0.0.1 "Return to Glossarion" page
the app served after the device approval; the sign-in service stopped right after opening it, so Android froze the
app that serves it.

Fix: Android brings the app's Activity back over the Custom Tab (``bring_to_front``, closing it), iOS closes the
in-app browser; the return page only when Android could not come back.

Run from src/mobile with the mobile venv::

    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hotfix_9157_6.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

MOBILE_DIR = Path(__file__).resolve().parents[1]
for entry in (str(MOBILE_DIR / "app"), str(MOBILE_DIR.parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

_AP_SPEC = importlib.util.spec_from_file_location("_glossarion_hf9157_6_accounts",
                                                  Path(__file__).with_name("test_accounts_profiles.py"))
AP = importlib.util.module_from_spec(_AP_SPEC)
_AP_SPEC.loader.exec_module(AP)

from glossarion_mobile.services.native import NativeStub  # noqa: E402
from glossarion_mobile.services.oauth import OAuthBridge  # noqa: E402


class _Native(NativeStub):
    def __init__(self, comes_back: bool) -> None:
        super().__init__()
        self.comes_back = comes_back
        self.brought = 0

    async def bring_to_front(self) -> bool:
        self.brought += 1
        return self.comes_back


def _grok_sign_in(tmp_path, **bridge_kwargs):
    token_dir = tmp_path / "home" / ".glossarion"
    token_dir.mkdir(parents=True)
    grok = AP.FakeDeviceModule(token_dir)
    opener = AP.Opener()
    bridge = OAuthBridge(auth_modules={"authgrok": grok}, opener=opener, timeout=5,
                         safe_root=str(tmp_path / "home"), **bridge_kwargs)

    async def approve():
        task = asyncio.ensure_future(bridge.sign_in("authgrok", 0))
        assert await AP._wait_for(lambda: bridge.state.user_code == "WXYZ-1234")
        grok.approved.set()
        return await asyncio.wait_for(task, AP.TIMEOUT)

    status = AP._run(approve())
    assert status["signed_in"]
    return opener.opened


def _return_pages(opened: list) -> list:
    return [url for url in opened if url.startswith("http://127.0.0.1:")]


def test_android_comes_back_over_the_browser_instead_of_the_return_page(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOSSARION_OAUTH_RETURN_URL", "glossarion://app/oauth/return")
    native = _Native(comes_back=True)
    opened = _grok_sign_in(tmp_path, native=native, is_android=True)
    assert native.brought == 1 and not _return_pages(opened)


def test_android_without_the_native_call_still_shows_the_return_page(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOSSARION_OAUTH_RETURN_URL", "glossarion://app/oauth/return")
    native = _Native(comes_back=False)  # an older extension answers False
    opened = _grok_sign_in(tmp_path, native=native, is_android=True)
    assert native.brought == 1 and len(_return_pages(opened)) == 1


def test_ios_closes_the_in_app_browser(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOSSARION_OAUTH_RETURN_URL", "glossarion://app/oauth/return")
    closed = []

    async def close_browser():
        closed.append(True)

    opened = _grok_sign_in(tmp_path, close_browser=close_browser)
    assert closed == [True] and not _return_pages(opened)
