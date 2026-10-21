"""authchatgpt/ on the in-app browser: a hidden page rebuilt mid-reply (2026-10-21 report).

"AuthChatGPT page was reloaded during translation" came from two things the poll loop could
not tell apart:

* the in-app browser (``WebViewPage.run_js``) answers None whenever a page's answer is late or
  the call could not be posted, so one late poll looked like a reload;
* a real new document: Flet 1.0.3 draws ``page.overlay`` (the hidden WebViews' home) only in
  the top View, so a screen push / pop moves each hidden WebView into another widget tree and
  Flutter builds a new one, which loads its ``url`` again without the hook.

Now a None poll asks the page whether the hook is gone (``_DriverPage.hook_lost``) and keeps
polling when it is not; a page that is really gone gets the message once more on a new page
(no API error, no backoff, the watchdog / pre-call callback runs once). The desktop (CDP) keeps
its behaviour: there None always means a new document.

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore tests_host/test_authchatgpt_page_rebuilt.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = MOBILE_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import authchatgpt_auth as gpt  # noqa: E402

NODE = shutil.which("node")
REPLY = ('data: {"v": {"message": {"author": {"role": "assistant"}, "recipient": "all", "channel": null,'
         ' "content": {"content_type": "text", "parts": ["Title"]}, "metadata": {"model_slug": "gpt-x"}}}}\n\n'
         'data: {"type": "message_stream_complete"}\n\n')


class FakePage:
    """A hooked chatgpt.com page; ``script`` lists what each poll gets after the send:
    a state dict, ``None`` (a late answer) or ``"gone"`` (a new document without the hook)."""

    def __init__(self, script, *, driver=True):
        self.script = list(script)
        self.requested = False
        self.gone = False
        self.probes = 0
        if driver:
            self.driver = True
            self.hook_lost = self._hook_lost

    def _hook_lost(self):
        self.probes += 1
        return self.gone

    def evaluate(self, expression):
        if expression == gpt._COMPOSER:
            return {"text": ".", "send": True, "requested": self.requested}
        if expression == gpt._SEND:
            self.requested = True
            return True
        if expression == gpt._POLL:
            step = self.script.pop(0) if self.script else {"chunks": [], "streams": 1, "open": 0}
            if step == "gone":
                self.gone = True
            return None if step in (None, "gone") else step
        return True


def _run(monkeypatch, pages, **kwargs):
    opened = []

    @contextmanager
    def open_tab(config, cancel_check=None):
        page = pages[len(opened)]
        opened.append(config)
        yield page

    monkeypatch.setattr(gpt, "_open_tab", open_tab)
    monkeypatch.setattr(gpt, "_access_token", lambda log_fn: None)
    monkeypatch.setattr(gpt.time, "sleep", lambda s: None)
    logs, sends = [], []
    result = gpt.send_chat_completion(messages=[{"role": "user", "content": "Translate"}], model="auto",
                                      timeout=30, log_fn=logs.append,
                                      before_send_callback=lambda: sends.append(1), **kwargs)
    return result, opened, logs, sends


def _done():
    return {"chunks": [REPLY], "streams": 1, "open": 0}


def test_a_late_poll_answer_is_not_a_reload(monkeypatch):
    page = FakePage([None, None, _done()])
    result, opened, logs, sends = _run(monkeypatch, [page])
    assert result["content"] == "Title"
    assert len(opened) == 1 and sends == [1] and page.probes == 2
    assert not any("again" in line for line in logs)


def test_a_rebuilt_page_sends_once_more_on_a_new_page(monkeypatch):
    first = FakePage([{"chunks": [], "streams": 1, "open": 1}, "gone"])
    second = FakePage([_done()])
    result, opened, logs, sends = _run(monkeypatch, [first, second])
    assert result["content"] == "Title"
    assert len(opened) == 2 and opened[0] == opened[1]
    assert sends == [1]  # the watchdog claim / pre-call callback runs once per request
    assert any("sending the message again" in line for line in logs)


def test_a_second_rebuild_fails_the_request(monkeypatch):
    with pytest.raises(gpt.AuthChatGPTError, match="reloaded during translation") as info:
        _run(monkeypatch, [FakePage(["gone"]), FakePage(["gone"])])
    assert info.value.error_type == "browser_error"


def test_desktop_keeps_failing_at_once(monkeypatch):
    with pytest.raises(gpt.AuthChatGPTError, match="reloaded during translation"):
        _run(monkeypatch, [FakePage([None], driver=False), FakePage([_done()], driver=False)])


class _RunJs:
    def __init__(self, answer):
        self.answer = answer
        self.scripts = []

    def run_js(self, script, timeout):
        self.scripts.append(script)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def test_hook_lost_reads_the_top_page_and_treats_no_answer_as_unknown():
    page = gpt._DriverPage(_RunJs(True))
    page.framed = True
    assert page.hook_lost() is True
    assert page.page.scripts[0].startswith("(() => {const f = document.getElementById('glossarion-chatgpt');")
    assert gpt._DriverPage(_RunJs(None)).hook_lost() is None
    assert gpt._DriverPage(_RunJs(RuntimeError("closed"))).hook_lost() is None


@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize("framed, frame, hooked, expected", [
    (True, "hooked", None, False),      # the chat frame still runs the hook
    (True, "bare", None, True),         # the frame's app was reloaded
    (True, None, None, True),           # the host page was reloaded: no frame at all
    (True, "cross", None, True),        # the frame left chatgpt.com
    (False, None, True, False),         # logged out: the app is the top page, hooked
    (False, None, False, True),         # logged out: a new top document
])
def test_hook_lost_script_under_node(framed, frame, hooked, expected):
    runner = _RunJs(None)
    page = gpt._DriverPage(runner)
    page.framed = framed
    page.hook_lost()
    script = runner.scripts[0]
    frames = {"hooked": "{contentWindow: {__glossarionGPT: {}}}", "bare": "{contentWindow: {}}",
              "cross": "{get contentWindow() { return {get __glossarionGPT() { throw Error('cross-origin'); }}; }}"}
    harness = (f"const window = {{{'__glossarionGPT: {}' if hooked else ''}}};"
               f" const document = {{getElementById: () => ({frames.get(frame, 'null')})}};"
               f" process.stdout.write(JSON.stringify(eval({json.dumps(script)})));")
    proc = subprocess.run([NODE, "-e", harness], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) is expected
