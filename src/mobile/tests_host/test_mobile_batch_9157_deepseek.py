"""v9.15.7 owner batch, issues 1 and 3: DeepSeek web (``authds/``) on Glossarion Mobile.

1. "I am signed in to deepseek, but the app is not detecting my sign in session".
3. authds/ jobs logged ``Unexpected error: [Errno 2] No such file or directory: '.../home/.glossarion/
   authds_browser/.signed_in'``, saved a failed-request payload and retried 7 times per chapter.

Root cause: on the phone no browser profile creates ``~/.glossarion/authds_browser``; the translate route's
in-app check (``authds_auth._wait_login``) found the DeepSeek token and then ``touch``-ed the marker in that
missing folder, so the raw ``FileNotFoundError`` failed every request of a signed-in user (the generic
"Unexpected error" branch retried and saved it) and the ``.signed_in`` hint that Accounts shows was never
written. The Accounts / Welcome check also took one look at the hidden page (a late answer is None).

Fixed: one ``_mark_session`` writes the hint with its folder; the in-app checks ask the page again until a
deadline; a missing mobile sign-in fails at once with the Accounts hint (remembered for the next chapters),
and the shared client does not retry authds/ sign-in / setup errors.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for entry in (str(APP_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import authds_auth as ds  # noqa: E402
import browser_driver  # noqa: E402

MESSAGES = [{"role": "user", "content": "안녕"}]


class FakePage:
    """A hidden in-app page on chat.deepseek.com: ``tokens`` are the answers to the session check."""

    def __init__(self, site):
        self.site = site
        self.closed = False

    def load(self, url):
        self.site.loads.append(url)

    def run_js(self, script, js_timeout_ms=15000):
        if script == ds._TOKEN_PRESENT:
            self.site.token_reads += 1
            answers = self.site.tokens
            return answers.pop(0) if len(answers) > 1 else answers[0]
        if "document.readyState" in script:
            return True
        return True

    def close(self):
        self.closed = True


class FakeDriver:
    name = "fake"

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.loads = []
        self.pages = []
        self.token_reads = 0

    def open_page(self, *, owner, user_agent=None, viewport=None, cancel_check=None):
        page = FakePage(self)
        self.pages.append(page)
        return page

    def cancel_pages(self, owner=None, reason=browser_driver.CANCELLED):
        pass


@pytest.fixture()
def phone(tmp_path, monkeypatch):
    """The phone's layout: HOME is the app's data/home, where nothing has created ``.glossarion`` yet."""
    home = tmp_path / "data" / "home"
    home.mkdir(parents=True)
    for name in ("HOME", "USERPROFILE", "OUTPUT_DIRECTORY", "GLOSSARION_LIBRARY_DIR", "GLOSSARION_DATA_DIR"):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.setenv("AUTHDS_PROFILE_DIR", str(home / ".glossarion" / "authds_browser"))
    monkeypatch.delenv("AUTHDS_MODE", raising=False)
    monkeypatch.setattr(ds, "DRIVER_SESSION_SECONDS", 0.5)
    monkeypatch.setattr(ds, "DRIVER_CHECK_SECONDS", 0.5)
    monkeypatch.setitem(ds._no_session, "at", None)
    monkeypatch.setitem(ds._muted, "until", 0.0)
    previous = browser_driver.get_driver()
    drivers = []

    def install(tokens):
        driver = FakeDriver(tokens)
        browser_driver.register_driver(driver)
        drivers.append(driver)
        return driver

    yield install
    if previous is not None:
        browser_driver.register_driver(previous)
    else:
        browser_driver.unregister_driver()


def _reply(monkeypatch):
    sent = []

    def complete(page, model, messages, *args, **kwargs):
        sent.append(model)
        return {"content": "Hello", "finish_reason": "stop"}

    monkeypatch.setattr(ds, "_complete", complete)
    return sent


def test_a_signed_in_phone_translates_although_the_profile_folder_does_not_exist(phone, monkeypatch):
    """Issue 3's log: the session was there; writing its hint into the missing folder raised Errno 2."""
    phone([True])
    sent = _reply(monkeypatch)
    assert not ds.profile_dir().exists() and not ds.has_session()
    result = ds.send_chat_completion(messages=MESSAGES, model="authds/flash", log_fn=lambda *_: None)
    assert result["content"] == "Hello" and sent == ["authds/flash"]
    # issue 1: the translate route's sighting is what Accounts shows ("✓ Signed in")
    assert ds.has_session() and (ds.profile_dir() / ".signed_in").is_file()


def test_the_sign_in_check_waits_for_a_late_answer_and_marks_the_session(phone):
    """Issue 1: Accounts / Welcome / the model menu ask ``driver_session_present``; one None (a late answer)
    or an early False no longer reads as "Not signed in"."""
    from glossarion_mobile.ui.screens.accounts import deepseek_session_present

    site = phone([None, False, True])
    assert asyncio.run(deepseek_session_present()) is True
    assert site.token_reads == 3 and all(page.closed for page in site.pages)
    assert ds.has_session()


def test_without_a_sign_in_authds_fails_fast_with_the_accounts_hint(phone, monkeypatch):
    site = phone([False])
    sent = _reply(monkeypatch)
    started = time.monotonic()
    with pytest.raises(ds.AuthDSError) as info:
        ds.send_chat_completion(messages=MESSAGES, model="authds/flash", log_fn=lambda *_: None)
    assert info.value.error_type == "auth_error" and "Sign in to DeepSeek" in str(info.value)
    assert "Errno" not in str(info.value) and not sent
    assert len(site.pages) == 1 and time.monotonic() - started < 5
    # the next chapters fail at once: no hidden DeepSeek page each
    with pytest.raises(ds.AuthDSError, match="Sign in to DeepSeek"):
        ds.send_chat_completion(messages=MESSAGES, model="authds/flash", log_fn=lambda *_: None)
    assert len(site.pages) == 1 and not ds.has_session()
    # signed in from Accounts meanwhile: its check forgets the "not signed in" and the job goes on
    site.tokens[:] = [True]
    assert ds.driver_session_present() is True
    assert ds.send_chat_completion(messages=MESSAGES, model="authds/flash",
                                   log_fn=lambda *_: None)["content"] == "Hello"


def test_the_client_does_not_retry_or_save_a_missing_deepseek_sign_in(phone, monkeypatch, tmp_path):
    """Issue 3: one clear error per request, not 7 retries each saving a failed-request payload."""
    import unified_api_client as uac

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GRACEFUL_STOP", raising=False)
    calls = []

    def send(**kwargs):
        calls.append(kwargs.get("model"))
        raise ds._not_signed_in_error()

    monkeypatch.setattr(ds, "send_chat_completion", send)
    client = uac.UnifiedClient(api_key="", model="authds/flash", output_dir=str(tmp_path / "out"))
    with pytest.raises(uac.UnifiedClientError) as info:
        client.send(MESSAGES, temperature=0.3, max_tokens=64, context="translation")
    assert "Sign in to DeepSeek" in str(info.value) and info.value.error_type == "auth_error"
    assert len(calls) == 1
    assert not list(tmp_path.rglob("failed_*authds*.json"))
