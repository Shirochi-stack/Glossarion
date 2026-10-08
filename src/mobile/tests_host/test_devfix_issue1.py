"""Owner's device report #1 on the U8 APK (2026-10-08): "Reader pages do not turn" on Android.

What the owner saw: after the first page, a chapter change moved the Reader's index, title, "Ch N/M"
and the saved position (so every reopen started one chapter further on), but the page on screen never
changed. flet-webview 1.0.3 reads ``url`` only when Flutter builds the WebView's State (``initState``,
no ``didUpdateWidget``), so the Reader's ``webview.url = ...`` did nothing on the device, and the
event de-duplicator then dropped every event of the page still on screen (even the centre tap).

This acceptance test replays the owner's session headlessly on the real objects: the real app
(``GlossarionApp`` on the fake Flet session as Android, ``test_ui_foundations._start``) opens a Library
book through its real ReaderFeature, ReaderServer and ReaderScreen, and two client stand-ins answer the
way the Flutter side does on the phone:

* ``WebViewClient`` (flet-webview's ``webview_mobile_and_mac.dart``): a WebView control loads its
  ``url`` once, when its Flutter State is created; later ``url`` patches change nothing; ``load_request``
  loads a page; ``run_javascript`` runs in the page; the page's ``console.log`` lines come back as
  ``console_message`` events. The page runs in headless Chrome (DevTools protocol over node >= 22's
  WebSocket) when one is installed: the real shared shell JS (``reader_doc``) and Reader extras, real
  CSS-column pages, real taps and swipes, and the fetch fallback reaching the ReaderServer. Without
  Chrome, ``ShellModel`` replays the shell's events in Python.
* ``ListViewClient`` (Flutter's ListView with Android's clamping physics) for the native "Lightweight"
  reader: ``scroll_to`` reports scroll notifications (at a bound it only overscrolls) and taps arrive as
  GestureDetector ``tap_up`` events.

"What the owner sees" is the page the stand-in shows (its last load, and the text Chrome renders):
every chapter path must change it on the same WebView, the new page's events must be handled, and
the saved position must be the one on screen.

Run from src/mobile (mobile venv; ``GLOSSARION_TEST_NO_BROWSER=1`` uses ShellModel instead of Chrome):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_issue1.py
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Headless Chrome on Windows does not start with USERPROFILE pointing elsewhere; it writes nothing there
# (its profile is a --user-data-dir under tmp). Taken before any fixture points USERPROFILE at tmp.
_REAL_USERPROFILE = os.environ.get("USERPROFILE")

WIDTH, HEIGHT = 412, 860
CHAPTERS = 6
LONG = 1  # the long chapter (several pages / screens); the others are short
LINK_TARGET = 5  # chapter 1 links to chapter 5 (index 4)
MARK = re.compile(r"\b(EN|RAW)-MARK-(\d\d)\b")
ISOLATED_ENV = ("GLOSSARION_LIBRARY_DIR", "OUTPUT_DIRECTORY", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA")
RIGHT, LEFT, CENTRE, TAP_Y = WIDTH - 12, 12, WIDTH // 2, 430


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _md5(path: Path) -> Optional[str]:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else None


async def _wait(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.03)
    return predicate()


# =====================================================================================
# The owner's book: a compiled (translated) EPUB with its raw source (Original / Translated)
# =====================================================================================

_EN = "The rain kept falling on the old harbour city while the night watch walked its rounds. "
_KO = "비가 오래된 항구 도시에 계속 내렸고 야경꾼들은 밤새 순찰을 돌았다. "


def _write_epub(path: Path, *, marker: str, lang: str, filler: str, link: bool) -> Path:
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier(f"glossarion-devfix-issue1-{marker.lower()}")
    book.set_title("Owner Book")
    book.set_language(lang)
    book.add_author("Glossarion tests")
    items = []
    for number in range(1, CHAPTERS + 1):
        mark = f"{marker}-MARK-{number:02d}"
        body = [f"<h1>{marker} chapter {number}</h1>", f"<p>{mark}</p>"]
        if link and number == 1:
            body.append(f'<p><a href="chapter{LINK_TARGET:04d}.xhtml">{marker} link to chapter {LINK_TARGET}</a></p>')
        for n in range(16 if number - 1 == LONG else 1):
            body.append(f"<p>{mark} {n + 1}. {filler * 6}</p>")
        chapter = epub.EpubHtml(title=f"{marker} chapter {number}", file_name=f"chapter{number:04d}.xhtml", lang=lang)
        chapter.content = (f"<html><head><title>{marker} chapter {number}</title></head>"
                           f"<body>{''.join(body)}</body></html>")
        book.add_item(chapter)
        items.append(chapter)
    book.toc = tuple(items)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *items]
    path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(path), book)
    return path


def _owner_book(folder: Path) -> dict:
    raw = _write_epub(folder / "Owner Book.epub", marker="RAW", lang="ko", filler=_KO, link=False)
    translated = _write_epub(folder / "Owner Book_translated.epub", marker="EN", lang="en", filler=_EN, link=True)
    return {"name": "Owner Book", "path": str(translated), "raw_source_path": str(raw)}


def _marks(text: str) -> set:
    return {(kind, int(number) - 1) for kind, number in MARK.findall(text or "")}


# =====================================================================================
# Page engines: what runs the page the WebView loads
# =====================================================================================

_CDP_BRIDGE_JS = r"""
'use strict';
const { spawn } = require('child_process');
const readline = require('readline');
const [chrome, userDir, width, height] = process.argv.slice(2);
const W = parseInt(width, 10), H = parseInt(height, 10);
process.stdout.on('error', () => {});
const out = (o) => { try { process.stdout.write(JSON.stringify(o) + '\n'); } catch (e) {} };
const args = ['--headless=new', '--remote-debugging-port=0', '--user-data-dir=' + userDir, '--no-first-run',
  '--no-default-browser-check', '--disable-extensions', '--disable-background-networking', '--disable-sync',
  '--disable-component-update', '--disable-default-apps', '--disable-domain-reliability', '--no-pings',
  '--disable-client-side-phishing-detection', '--metrics-recording-only', '--mute-audio', '--disable-gpu',
  '--password-store=basic', '--use-mock-keychain', '--proxy-server=127.0.0.1:9',
  '--window-size=' + W + ',' + H, 'about:blank'];
if (process.platform === 'linux') { args.unshift('--no-sandbox'); }
const proc = spawn(chrome, args, { stdio: ['ignore', 'ignore', 'pipe'] });
let finished = false, ws = null, nextId = 1, buffer = '';
const pending = new Map(), loadWaiters = [];
function finish(code) {
  if (finished) { return; }
  finished = true;
  try { proc.kill(); } catch (e) {}
  setTimeout(() => process.exit(code), 300);
}
proc.on('exit', () => { if (!finished) { out({ fatal: 'chrome exited' }); finish(3); } });
proc.on('error', (e) => { out({ fatal: 'chrome did not start: ' + e }); finish(3); });
process.on('exit', () => { try { proc.kill(); } catch (e) {} });
setTimeout(() => { if (!ws) { out({ fatal: 'no DevTools endpoint' }); finish(4); } }, 60000);
function send(method, params) {
  const id = nextId++;
  ws.send(JSON.stringify({ id, method, params: params || {} }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
proc.stderr.on('data', async (chunk) => {
  if (ws) { return; }
  buffer += chunk.toString();
  const m = /DevTools listening on (ws:\/\/[^\s]+)/.exec(buffer);
  if (!m) { return; }
  ws = 'pending';
  try {
    const port = new URL(m[1]).port;
    let page = null;
    for (let i = 0; i < 100 && !page; i++) {
      const list = await (await fetch('http://127.0.0.1:' + port + '/json/list')).json();
      page = list.find((t) => t.type === 'page');
      if (!page) { await sleep(100); }
    }
    ws = new WebSocket(page.webSocketDebuggerUrl);
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && pending.has(msg.id)) {
        const p = pending.get(msg.id);
        pending.delete(msg.id);
        if (msg.error) { p.reject(new Error(JSON.stringify(msg.error))); } else { p.resolve(msg.result); }
      } else if (msg.method === 'Runtime.consoleAPICalled') {
        const a = (msg.params.args || [])[0];
        if (a && typeof a.value === 'string') { out({ console: a.value }); }
      } else if (msg.method === 'Page.loadEventFired') {
        while (loadWaiters.length) { loadWaiters.shift()(); }
      } else if (msg.method === 'Runtime.exceptionThrown') {
        out({ exception: JSON.stringify(msg.params.exceptionDetails || {}).slice(0, 800) });
      }
    };
    await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
    await send('Page.enable');
    await send('Runtime.enable');
    await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 2, mobile: true });
    await send('Emulation.setTouchEmulationEnabled', { enabled: true, maxTouchPoints: 5 });
    out({ ready: true });
  } catch (e) {
    out({ fatal: String(e) });
    finish(5);
  }
});
async function handle(cmd) {
  switch (cmd.cmd) {
    case 'navigate': {
      const loaded = new Promise((resolve) => loadWaiters.push(resolve));
      await send('Page.navigate', { url: cmd.url });
      await Promise.race([loaded, sleep(15000)]);
      return true;
    }
    case 'eval': {
      const r = await send('Runtime.evaluate', { expression: cmd.expr, returnByValue: true, awaitPromise: true });
      if (r.exceptionDetails) { throw new Error(JSON.stringify(r.exceptionDetails).slice(0, 800)); }
      return r.result ? r.result.value : null;
    }
    case 'tap': {
      const p = { x: cmd.x, y: cmd.y, button: 'left', clickCount: 1 };
      await send('Input.dispatchMouseEvent', Object.assign({ type: 'mousePressed' }, p));
      await send('Input.dispatchMouseEvent', Object.assign({ type: 'mouseReleased' }, p));
      return true;
    }
    case 'swipe': {
      const at = (x) => [{ x: x, y: cmd.y, id: 1 }];
      await send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: at(cmd.x0) });
      for (let i = 1; i <= 6; i++) {
        const x = cmd.x0 + (cmd.x1 - cmd.x0) * i / 6;
        await send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: at(x) });
      }
      await send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
      return true;
    }
    case 'quit':
      finish(0);
      return true;
    default:
      throw new Error('unknown command ' + cmd.cmd);
  }
}
readline.createInterface({ input: process.stdin }).on('line', async (line) => {
  let cmd;
  try { cmd = JSON.parse(line); } catch (e) { return; }
  try {
    const value = await handle(cmd);
    out({ id: cmd.id, ok: true, value: value === undefined ? null : value });
  } catch (e) {
    out({ id: cmd.id, ok: false, error: String((e && e.message) || e) });
  }
}).on('close', () => finish(0));
"""

_SHOWN_JS = (
    "JSON.stringify({text: (document.body ? document.body.innerText : '').slice(0, 20000),"
    " page: window.GLRDR ? GLRDR.page() : -1, count: window.GLRDR ? GLRDR.count() : -1,"
    " paged: !!document.getElementById('columns'), url: location.href})"
)
_LINK_POINT_JS = (
    "(function (h) { var a = Array.prototype.slice.call(document.querySelectorAll('a')).filter("
    "function (x) { return (x.getAttribute('href') || '').indexOf(h) >= 0; })[0];"
    " if (!a) { return null; } var r = a.getBoundingClientRect();"
    " return [r.left + Math.min(r.width / 2, 20), r.top + r.height / 2]; })(%s)"
)


class ChromeEngine:
    """Headless Chrome as the WebView's web engine, driven over the DevTools protocol by node."""

    name = "chrome"

    def __init__(self, workdir: Path, chrome: str, node: str) -> None:
        self.workdir, self.chrome, self.node = workdir, chrome, node
        self.on_console: Callable[[str], None] = lambda text: None
        self.proc: Optional[subprocess.Popen] = None
        self.cond = threading.Condition()
        self.results: dict = {}
        self.ready = False
        self.fatal: Optional[str] = None
        self.exceptions: list = []
        self.next_id = 0

    @staticmethod
    def locate() -> Optional[tuple]:
        """(chrome, node) when a Chromium browser and node with a global WebSocket (>= 22) exist."""
        if os.environ.get("GLOSSARION_TEST_NO_BROWSER"):
            return None
        node = shutil.which("node")
        if not node:
            return None
        try:
            kind = subprocess.run([node, "-e", "process.stdout.write(typeof WebSocket)"], capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:
            return None
        if kind != "function":
            return None
        candidates = [os.environ.get("CHROME_PATH"), shutil.which("google-chrome"),
                      shutil.which("google-chrome-stable"), shutil.which("chromium"), shutil.which("chromium-browser"),
                      r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                      r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                      r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate, node
        return None

    def start(self, timeout: float = 90.0) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        script = self.workdir / "cdp_bridge.js"
        script.write_text(_CDP_BRIDGE_JS, encoding="utf-8")
        env = dict(os.environ)
        if _REAL_USERPROFILE:
            env["USERPROFILE"] = _REAL_USERPROFILE
        self.proc = subprocess.Popen([self.node, str(script), self.chrome, str(self.workdir / "profile"), str(WIDTH),
                                      str(HEIGHT)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, encoding="utf-8", env=env)
        threading.Thread(target=self._read, name="devfix1-chrome", daemon=True).start()
        with self.cond:
            self.cond.wait_for(lambda: self.ready or self.fatal is not None, timeout)
        if not self.ready:
            self.stop()
            raise RuntimeError(f"headless Chrome did not start: {self.fatal or 'timeout'}")

    def _read(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "console" in msg:
                self.on_console(str(msg["console"]))
                continue
            with self.cond:
                if "id" in msg:
                    self.results[msg["id"]] = msg
                elif msg.get("ready"):
                    self.ready = True
                elif "fatal" in msg:
                    self.fatal = str(msg["fatal"])
                elif "exception" in msg:
                    self.exceptions.append(msg["exception"])
                self.cond.notify_all()
        with self.cond:
            self.fatal = self.fatal or "the bridge exited"
            self.cond.notify_all()

    def call(self, cmd: str, timeout: float = 30.0, **params: Any) -> Any:
        assert self.proc is not None and self.proc.stdin is not None
        with self.cond:
            self.next_id += 1
            ident = self.next_id
        self.proc.stdin.write(json.dumps({"id": ident, "cmd": cmd, **params}) + "\n")
        self.proc.stdin.flush()
        with self.cond:
            done = self.cond.wait_for(lambda: ident in self.results or self.fatal is not None, timeout)
            msg = self.results.pop(ident, None)
        if not done or msg is None:
            raise RuntimeError(f"Chrome: no answer to {cmd} ({self.fatal or 'timeout'})")
        if not msg.get("ok"):
            raise RuntimeError(f"Chrome: {cmd} failed: {msg.get('error')}")
        return msg.get("value")

    # ---- the engine interface ----
    def navigate(self, url: str) -> None:
        self.call("navigate", url=url)

    def run_js(self, code: str) -> None:
        try:
            self.call("eval", expr=code)
        except RuntimeError:
            pass  # a script for a page that is gone (run_javascript errors are only logged on the device)

    def tap(self, x: float, y: float) -> None:
        self.call("tap", x=x, y=y)

    def swipe(self, x0: float, x1: float, y: float) -> None:
        self.call("swipe", x0=x0, x1=x1, y=y)

    def click_link(self, href: str) -> None:
        point = self.call("eval", expr=_LINK_POINT_JS % json.dumps(href))
        assert point, f"no link to {href} on the page"
        self.tap(point[0], point[1])

    def shown(self) -> dict:
        return json.loads(self.call("eval", expr=_SHOWN_JS))

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"id": 0, "cmd": "quit"}) + "\n")
                proc.stdin.flush()
                proc.stdin.close()  # the bridge also quits (closing Chrome) when its stdin ends
            proc.wait(15)
        except Exception:
            proc.kill()


class ShellModel:
    """Without Chrome: the events the shared shell (``reader_doc._MOBILE_BRIDGE_JS``) and the Reader
    extras post for the page the ReaderServer serves, replayed in Python (3 pages per paged chapter)."""

    name = "shell-model"

    def __init__(self) -> None:
        self.on_console: Callable[[str], None] = lambda text: None
        self.page: dict = {}
        self.exceptions: list = []

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def _post(self, event: dict, *, doc: bool = False) -> None:
        state = self.page
        if doc:
            event["doc"] = state["cfg"].get("doc", "")
        state["seq"] += 1
        event["seq"] = state["seq"]
        if state["chapter"] is not None and "chapter" not in event:
            event["chapter"] = state["chapter"]
        self.on_console("GLRDR:" + json.dumps(event))

    def navigate(self, url: str) -> None:
        with urllib.request.urlopen(url.split("#", 1)[0], timeout=10) as response:
            html = response.read().decode("utf-8")
        chapter = re.search(r"CHAPTER = (null|-?\d+)", html).group(1)
        initial = int(re.search(r"INITIAL_PAGE = (\d+)", html).group(1))
        cfg = json.loads(html.split("window.__GLRDR_CFG=", 1)[1].split(";</script>", 1)[0])
        paged = "id='columns'" in html or 'id="columns"' in html
        count = 3 if paged else 1
        fraction = re.search(r"[#&]f=([0-9.]+)", "#" + url.partition("#")[2])
        first = round((count - 1) * float(fraction.group(1))) if fraction and paged else initial
        text = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S | re.I)
        self.page = {"url": url, "chapter": None if chapter == "null" else int(chapter), "cfg": cfg, "paged": paged,
                     "count": count, "page": max(0, min(first, count - 1)), "seq": 0,
                     "text": re.sub(r"<[^>]+>", " ", text)}
        self._post({"type": "ready", "page": self.page["page"], "count": count, "paginated": paged})
        if (cfg.get("find") or {}).get("text"):
            self._post({"type": "found", "ok": True}, doc=True)

    def _next(self) -> None:
        state = self.page
        if not state["paged"]:
            self._post({"type": "tap", "zone": "right"})
        elif state["page"] < state["count"] - 1:
            state["page"] += 1
            self._post({"type": "page", "page": state["page"], "count": state["count"], "reason": "next"})
        else:
            self._post({"type": "edge", "edge": "end", "page": state["page"], "count": state["count"]})

    def _prev(self) -> None:
        state = self.page
        if not state["paged"]:
            self._post({"type": "tap", "zone": "left"})
        elif state["page"] > 0:
            state["page"] -= 1
            self._post({"type": "page", "page": state["page"], "count": state["count"], "reason": "prev"})
        else:
            self._post({"type": "edge", "edge": "start", "page": state["page"], "count": state["count"]})

    def tap(self, x: float, y: float) -> None:
        side = x < WIDTH / 3 or x > 2 * WIDTH / 3
        if side and self.page["cfg"].get("zones") is False and self.page["paged"]:
            self._post({"type": "tap", "zone": "center"}, doc=True)
        elif x < WIDTH / 3:
            self._prev()
        elif x > 2 * WIDTH / 3:
            self._next()
        else:
            self._post({"type": "tap", "zone": "center"})

    def swipe(self, x0: float, x1: float, y: float) -> None:
        if abs(x1 - x0) > 40:
            self._next() if x1 < x0 else self._prev()

    def click_link(self, href: str) -> None:
        self._post({"type": "link", "href": href})

    def run_js(self, code: str) -> None:
        state = self.page
        if not state:
            return
        found = re.search(r"GLRDR\.(go|goLast)\((\d*)\)", code)
        if found:
            state["page"] = state["count"] - 1 if found.group(1) == "goLast" else min(int(found.group(2) or 0),
                                                                                       state["count"] - 1)
            self._post({"type": "page", "page": state["page"], "count": state["count"], "reason": "api"})
        elif "GLRDR.find(" in code:
            self._post({"type": "found", "ok": True}, doc=True)

    def shown(self) -> dict:
        state = self.page
        return {"text": state["text"], "page": state["page"], "count": state["count"], "paged": state["paged"],
                "url": state["url"]}


# =====================================================================================
# Client stand-ins (what Flutter does on the phone)
# =====================================================================================


def _invokes(conn, name: str) -> list:
    from flet.messaging.protocol import MessageAction

    return [m for m in conn.messages if m.action == MessageAction.INVOKE_METHOD and m.body.name == name]


class WebViewClient:
    """flet-webview 1.0.3 on Android (``webview_mobile_and_mac.dart``) for the Reader's WebViews.

    ``built(control)`` is Flutter creating the control's State: ``initState`` loads ``url`` (once). A
    State lives as long as the control id: a control that took over an older one's id (Flet's
    reconciliation) keeps the old State and loads nothing. ``load_request`` loads a page;
    ``run_javascript`` runs in it; ``url`` patches are ignored (no ``didUpdateWidget``). The page's
    console lines are delivered as ``console_message`` events of the WebView that loaded it."""

    def __init__(self, conn, session, engine) -> None:
        self.conn, self.session, self.engine = conn, session, engine
        self.loop = asyncio.get_running_loop()
        self.loads: list = []  # (control id, url, "init" | "load_request") in order
        self.scripts: list = []
        self.states: set = set()  # WebView control ids with a Flutter State
        self.console: list = []  # (control id, text) as delivered
        self.fail_loads = 0  # the next N load_requests fail on the platform side (an error answer)
        self.failed: list = []
        self.loading: Optional[int] = None
        self.jobs: queue.Queue = queue.Queue()
        self.engine.on_console = self._on_console
        self._send = conn.send_message
        conn.send_message = self._intercept
        self.worker = threading.Thread(target=self._work, name="devfix1-webview", daemon=True)
        self.worker.start()

    @property
    def shown_url(self) -> Optional[str]:
        return self.loads[-1][1] if self.loads else None

    def built(self, control: Any) -> None:
        ident = control._i
        if ident in self.states:
            return
        self.states.add(ident)
        self._load(ident, str(control.url), "init")

    def _intercept(self, message: Any) -> None:
        from flet.messaging.protocol import MessageAction

        body = message.body
        ours = message.action == MessageAction.INVOKE_METHOD and body.control_id in self.states
        if ours and body.name == "load_request" and self.fail_loads > 0:
            self.fail_loads -= 1  # the platform refuses: the page on screen stays, the call raises
            self.conn.messages.append(message)
            self.failed.append(str((body.args or {}).get("url")))
            self.loop.call_soon_threadsafe(self.session.handle_invoke_method_results, body.control_id, body.call_id,
                                           None, "WebView: net::ERR_FAILED")
            return
        self._send(message)
        if not ours:
            return
        args = body.args or {}
        if message.body.name == "load_request":
            self._load(message.body.control_id, str(args.get("url")), "load_request")
        elif message.body.name == "run_javascript":
            self.scripts.append(str(args.get("value")))
            self.submit("run_js", str(args.get("value")))

    def _load(self, ident: int, url: str, how: str) -> None:
        self.loads.append((ident, url, how))
        self.submit("navigate", url, ident=ident)

    def submit(self, method: str, *args: Any, ident: Optional[int] = None) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()
        self.jobs.put((future, method, args, ident))
        return future

    async def do(self, method: str, *args: Any) -> Any:
        return await asyncio.wrap_future(self.submit(method, *args))

    def _work(self) -> None:
        while True:
            item = self.jobs.get()
            if item is None:
                return
            future, method, args, ident = item
            if ident is not None:
                self.loading = ident
            try:
                future.set_result(getattr(self.engine, method)(*args))
            except BaseException as exc:  # noqa: BLE001 - handed to the awaiting test
                future.set_exception(exc)

    def _on_console(self, text: str) -> None:
        ident = self.loading
        if ident is not None:
            self.loop.call_soon_threadsafe(self.deliver, ident, text)

    def deliver(self, ident: int, text: str) -> None:
        self.console.append((ident, text))
        asyncio.ensure_future(self.session.dispatch_event(ident, "console_message",
                                                          {"message": text, "severity_level": "log"}))

    def close(self) -> None:
        self.jobs.put(None)
        self.worker.join(30)
        self.conn.send_message = self._send


class ListViewClient:
    """Flutter's ListView with Android's ClampingScrollPhysics for the native reader's list
    (flet ``scrollable_control.dart`` + ``scroll_notification_control.dart``).

    ``scroll_to(delta, duration)`` animates: a start notification at once, updates while it moves, and
    at a bound an overscroll before the end notification (pixels stay at the bound).
    ``scroll_to(offset)`` (no duration) jumps: start / update / end only when the offset changes.
    The content height is estimated from the list's texts (26 px lines of 38 characters)."""

    LINE, CHARS = 26.0, 38

    def __init__(self, conn, session, list_view: Callable[[], Any]) -> None:
        self.conn, self.session, self.list_view = conn, session, list_view
        self.loop = asyncio.get_running_loop()
        self.pixels = 0.0
        self.calls: list = []
        self.events: list = []
        self._send = conn.send_message
        conn.send_message = self._intercept

    def _intercept(self, message: Any) -> None:
        from flet.messaging.protocol import MessageAction

        self._send(message)
        target = self.list_view()
        if (message.action != MessageAction.INVOKE_METHOD or target is None
                or message.body.control_id != target._i or message.body.name != "scroll_to"):
            return
        args = message.body.args or {}
        duration = args.get("duration") or 0
        if isinstance(duration, dict):
            duration = duration.get("milliseconds", 0)
        self.calls.append((args.get("offset"), args.get("delta"), duration))
        self.loop.call_soon_threadsafe(self._scroll, target._i, args.get("offset"), args.get("delta"),
                                       float(duration or 0))

    @classmethod
    def _text(cls, control: Any) -> list:
        import flet as ft

        if isinstance(control, ft.Text):
            return [str(control.value or "") + "".join(str(s.text or "") for s in (control.spans or []))]
        found = []
        for name in ("content", "controls"):
            child = getattr(control, name, None)
            for item in (child if isinstance(child, list) else [child] if child is not None else []):
                if hasattr(item, "__dict__"):
                    found.extend(cls._text(item))
        return found

    def max_extent(self) -> float:
        target = self.list_view()
        height = 32.0  # the list's vertical padding
        for control in list(getattr(target, "controls", None) or []):
            for text in self._text(control) or [""]:
                lines = max(1, -(-len(text) // self.CHARS))
                height += lines * self.LINE + 12
        return max(0.0, height - HEIGHT)

    def _emit(self, ident: int, kind: str, maximum: float, **extra: Any) -> None:
        data = {"event_type": kind, "pixels": self.pixels, "min_scroll_extent": 0.0, "max_scroll_extent": maximum,
                "viewport_dimension": float(HEIGHT), **extra}
        self.events.append(data)
        asyncio.ensure_future(self.session.dispatch_event(ident, "scroll", data))

    def _scroll(self, ident: int, offset: Any, delta: Any, duration_ms: float) -> None:
        maximum = self.max_extent()
        self.pixels = min(max(self.pixels, 0.0), maximum)  # a relayout clamps the offset (no notification)
        before = self.pixels
        if offset is not None or duration_ms < 1:
            target = float(offset) if offset is not None else before + float(delta or 0)
            if offset is not None and target < 0:
                target = maximum + target + 1
            target = min(max(target, 0.0), maximum)
            if target != before:  # jumpTo
                self.pixels = target
                self._emit(ident, "start", maximum)
                self._emit(ident, "update", maximum, scroll_delta=target - before)
                self._emit(ident, "end", maximum)
            return
        wanted = before + float(delta or 0)
        final = min(max(wanted, 0.0), maximum)
        self._emit(ident, "start", maximum)

        def halfway() -> None:
            if final != before:
                self.pixels = before + (final - before) / 2
                self._emit(ident, "update", maximum, scroll_delta=(final - before) / 2)

        def finish() -> None:
            moved = final - self.pixels
            self.pixels = final
            if moved:
                self._emit(ident, "update", maximum, scroll_delta=moved)
            if final != wanted:
                self._emit(ident, "overscroll", maximum, overscroll=wanted - final, velocity=0.0)
            self._emit(ident, "end", maximum)

        self.loop.call_later(duration_ms / 3000.0, halfway)
        self.loop.call_later(duration_ms / 1000.0, finish)

    def close(self) -> None:
        self.conn.send_message = self._send


# =====================================================================================
# The real app on the fake Flet session as Android
# =====================================================================================

_UF = None


def _ui_helpers():
    """test_ui_foundations' app starter (one copy of the fake session helpers)."""
    global _UF
    if _UF is None:
        spec = importlib.util.spec_from_file_location("_glossarion_uf_helpers_devfix1",
                                                      Path(__file__).with_name("test_ui_foundations.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _UF = module
    return _UF


@pytest.fixture
def phone(tmp_path, monkeypatch):
    """Isolated storage (Library, output, HOME, USERPROFILE, APPDATA under tmp; src/config.json unchanged),
    the owner's book, and ``await phone.start()`` -> the real app on the fake session as Android."""
    needed = ("flet", "msgpack", "flet_webview", "ebooklib", "reader_doc", "library_core", "reader_overlay")
    if not all(_has(m) for m in needed):
        pytest.skip("flet / flet_webview / msgpack / ebooklib or the shared reader cores are missing")
    config = SRC_DIR / "config.json"
    config_md5 = _md5(config)
    for name in ISOLATED_ENV:
        folder = tmp_path / "env" / name.lower()
        folder.mkdir(parents=True)
        monkeypatch.setenv(name, str(folder))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    import reader_doc

    (tmp_path / "epubcache").mkdir()
    monkeypatch.setattr(reader_doc, "_EPUB_CACHE_DIR_OVERRIDE", str(tmp_path / "epubcache"))
    helpers = _ui_helpers()
    tb = helpers._TB
    for name in ("data", "cache", "temp"):
        folder = tmp_path / "storage" / name
        folder.mkdir(parents=True)
        monkeypatch.setenv(f"FLET_APP_STORAGE_{name.upper()}", str(folder))
    for key in tb._ENV_INPUTS:
        monkeypatch.delenv(key, raising=False)
    tb.rb.reset(restore_env=True)
    tb.secure_keys.reset()
    tb.rb.bootstrap(app_dir=APP_DIR, force=True)
    monkeypatch.setattr(tb.rb, "start_warm_import", lambda modules=None, *, on_done=None: on_done and on_done(
        {"ok": True, "modules": 0, "secs": 0.0, "failed": {}, "requested": []}))
    # sign-in status reads at start-up: token stores under tmp, also for auth modules imported earlier
    token_dir = str(tmp_path / "storage" / "data" / "home" / ".glossarion")
    for name in ("authgpt_auth", "authgem_auth", "authgrok_auth", "authcd_auth"):
        module = sys.modules.get(name)
        if module is None:
            continue
        if hasattr(module, "_DEFAULT_TOKEN_DIR"):
            monkeypatch.setattr(module, "_DEFAULT_TOKEN_DIR", token_dir)
        if hasattr(module, "_DEFAULT_TOKEN_FILE"):
            monkeypatch.setattr(module, "_DEFAULT_TOKEN_FILE",
                                os.path.join(token_dir, name.replace("_auth", "_tokens.json")))
        if hasattr(module, "_default_store"):
            monkeypatch.setattr(module, "_default_store", None)
        if hasattr(module, "_account_stores"):
            monkeypatch.setattr(module, "_account_stores", {})
    (tmp_path / "storage" / "data" / "mobile_state.json").write_text(json.dumps({"welcome_completed": True}),
                                                                       encoding="utf-8")
    from glossarion_mobile.ui.reader import reader_view

    monkeypatch.setattr(reader_view, "CHROME_AUTO_HIDE", 0.05)  # the first-open chrome fade, out of the way
    book = _owner_book(tmp_path / "books")
    state = types.SimpleNamespace(tmp=tmp_path, book=book, helpers=helpers, app=None, conn=None, session=None,
                                  page=None, closers=[], monkeypatch=monkeypatch)

    async def start(platform: str = "android"):
        _main, state.conn, state.session, state.page, state.app = await helpers._start(platform)
        state.session.apply_page_patch({"width": WIDTH, "height": HEIGHT})
        assert getattr(state.app, "reader", None) is not None, "the app did not install its ReaderFeature"
        return state.app

    async def stop():
        for close in reversed(state.closers):
            try:
                close()
            except Exception:
                pass
        if state.app is not None:
            state.app.reader.detach()
            await helpers._stop(state.app)

    state.start, state.stop = start, stop
    yield state
    tb._join_app_io_threads()
    tb.secure_keys.reset()
    tb.rb.reset(restore_env=True)
    assert _md5(config) == config_md5, "src/config.json changed"


def _engine(phone) -> Any:
    found = ChromeEngine.locate()
    if found is not None:
        engine = ChromeEngine(phone.tmp / "chrome", *found)
        try:
            engine.start()
            phone.closers.append(engine.stop)
            return engine
        except RuntimeError as exc:
            print(f"[devfix-issue1] {exc}; using ShellModel")
    return ShellModel()


def _attach_webview(phone, engine) -> WebViewClient:
    """The WebView stand-in on the app's connection; ``ReaderScreen._mount_webview`` is where a new
    WebView control reaches the client (Flutter builds its State and ``initState`` loads its url)."""
    from glossarion_mobile.ui.reader.reader_view import ReaderScreen

    client = WebViewClient(phone.conn, phone.session, engine)
    phone.closers.append(client.close)
    mount = ReaderScreen._mount_webview

    def mounted(self, url):
        mount(self, url)
        client.built(self.webview)

    phone.monkeypatch.setattr(ReaderScreen, "_mount_webview", mounted)
    return client


async def _open(phone, **kwargs) -> Any:
    from glossarion_mobile.ui.reader.reader_view import ReaderScreen

    app = phone.app
    previous = app.reader.active
    assert app.reader.open_book(dict(phone.book), **kwargs)

    def shown() -> bool:
        screen = app.reader.active
        if not isinstance(screen, ReaderScreen) or screen is previous or screen.state != "ready":
            return False
        if screen.renderer == "webview":
            return screen.webview is not None and screen.page_slot.content is screen.webview
        return screen.fallback_generation == screen.render_generation
    assert await _wait(shown, 30), "the Reader did not open"
    return app.reader.active


async def _leave(phone, screen) -> None:
    phone.app.back()
    assert await _wait(lambda: screen.disposed, 10)


async def _fire(phone, control: Any, event: str, data: Any = None) -> None:
    """A client event (a tap, a change) on a control the client shows."""
    assert phone.session.index.get(control._i) is control, f"{type(control).__name__} is not on screen"
    await phone.session.dispatch_event(control._i, event, data)


# =====================================================================================
# Issue 1 with the page view (the default on Android)
# =====================================================================================


class Turns:
    """Checks one chapter change the way the owner sees it."""

    def __init__(self, phone, client: WebViewClient, screen: Any) -> None:
        self.phone, self.client, self.screen = phone, client, screen
        self.webview = screen.webview
        self.urls = [screen.webview.url]

    def accepted_since(self, before: set) -> list:
        return [e for e in self.screen.events if id(e) not in before]

    async def page_ready(self, before: set, index: int) -> Any:
        screen = self.screen
        found: list = []

        def ready() -> bool:
            found[:] = [e for e in self.accepted_since(before) if e.type == "ready" and e.chapter == index]
            return bool(found)
        assert await _wait(ready, 20), f"the page of chapter {index + 1} never reported ready to the Reader"
        assert screen.page_count == found[-1].count  # the new page's events are handled
        return found[-1]

    async def shown(self) -> dict:
        return await self.client.do("shown")

    async def check_shown(self, index: int, *, mode: str = "EN") -> dict:
        """The page on screen is chapter ``index`` and the Reader (chrome, saved position) agrees."""
        screen, session = self.screen, self.screen.session
        shown = await self.shown()
        assert shown["url"].split("#", 1)[0] == self.client.shown_url.split("#", 1)[0]
        assert (mode, index) in _marks(shown["text"]), (mode, index, sorted(_marks(shown["text"])))
        assert {i for kind, i in _marks(shown["text"]) if kind == mode} == {index}
        assert screen.index == index
        assert screen.chrome.chapter_title.value == session.chapter_title(index)
        assert screen.chrome.progress_text.value.startswith(f"Ch {index + 1}/{session.count}")
        assert await _wait(lambda: screen.page_no == max(0, shown["page"]), 5)
        saved = screen.save_position()
        assert saved is not None and saved["href"] == session.filenames[index] and saved["chapter"] == index
        assert saved["page"] == max(0, shown["page"]) and saved["mode"] == session.flavor
        return shown

    async def turn(self, action: Callable[[], Any], index: int, *, mode: str = "EN") -> dict:
        """``action`` changes the chapter: exactly one ``load_request`` with the newly published page,
        on the same WebView; the page on screen and the Reader then agree on chapter ``index``."""
        screen, client = self.screen, self.client
        loads = len(client.loads)
        before = {id(e) for e in screen.events}
        doc = screen.current_doc
        result = action()
        if asyncio.iscoroutine(result):
            await result
        assert await _wait(lambda: len(client.loads) > loads and screen.current_doc != doc, 20), \
            "the chapter change never reached the WebView"
        await self.page_ready(before, index)
        await asyncio.sleep(0.25)  # nothing else follows
        new = client.loads[loads:]
        assert new == [(self.webview._i, screen.webview.url, "load_request")], new
        assert screen.webview is self.webview and screen.page_slot.content is self.webview
        assert new[0][1] not in self.urls and "?v=" in new[0][1]
        self.urls.append(new[0][1])
        return await self.check_shown(index, mode=mode)

    async def tap(self, x: float, y: float = TAP_Y) -> None:
        await self.client.do("tap", x, y)

    async def page_through(self) -> int:
        """Tap the right edge until the last page of the chapter on screen; returns the page count."""
        screen = self.screen
        shown = await self.shown()
        count = shown["count"]
        for page in range(shown["page"] + 1, count):
            await self.tap(RIGHT)
            assert await _wait(lambda: screen.page_no == page, 10), f"page {page + 1}/{count} never arrived"
            assert screen.chrome.page_text.value == f"Page {page + 1}/{count}"
            assert (await self.shown())["page"] == page
        return count


@pytest.mark.skipif(not _has("msgpack"), reason="msgpack not installed")
@pytest.mark.parametrize("platform", ["android", "ios"])  # iOS runs the same flet-webview Dart file
def test_owner_report_1_every_chapter_change_turns_the_page(phone, platform):
    """The owner's Reader session with the page view: page-JS edges (taps and a swipe), ◀ / ▶, the
    slider, a Chapters row, a book link, a search hit, Original, Aa layout and tap-edge changes. Each
    loads the new chapter into the same WebView, its page's events are handled (the outgoing page's are
    dropped), and the saved position is the page on screen."""
    async def scenario():
        await phone.start(platform)
        assert str(getattr(phone.page.platform, "value", phone.page.platform)) == platform
        engine = _engine(phone)
        print(f"[devfix-issue1] page engine: {engine.name}")
        client = _attach_webview(phone, engine)
        try:
            screen = await _open(phone, chapter=0)
            session = screen.session
            assert screen.renderer == "webview" and session.plan.mode == "dual" and session.count == CHAPTERS
            assert phone.app.reader.server is not None and screen.webview.url.startswith(
                phone.app.reader.server.base_url)
            # the first page: the WebView is built with its URL (initState loads it), no load_request
            assert client.loads == [(screen.webview._i, screen.webview.url, "init")]
            assert _invokes(phone.conn, "load_request") == []
            turns = Turns(phone, client, screen)
            await turns.page_ready(set(), 0)
            await turns.check_shown(0)
            # the console channel works: the page is switched to console-only (run_javascript reached it)
            assert await _wait(lambda: any("httpOff" in s for s in client.scripts), 5)

            # 1. page JS: tap through chapter 1, then past its last page -> chapter 2 (the long one)
            await turns.page_through()
            await turns.turn(lambda: turns.tap(RIGHT), LONG)
            count = await turns.page_through()
            if engine.name == "chrome":
                assert count >= 3, "the long chapter should span several CSS-column pages"
            # past the last page by a swipe (right to left) -> chapter 3
            await turns.turn(lambda: client.do("swipe", WIDTH - 60, 60, TAP_Y), 2)
            # before the first page -> the previous chapter at its last page
            shown = await turns.turn(lambda: turns.tap(LEFT), LONG)
            assert shown["page"] == shown["count"] - 1 == count - 1 and screen.last_page
            await turns.turn(lambda: turns.tap(RIGHT), 2)

            # the outgoing page's late events are dropped; the page on screen's events are handled
            stale_doc = screen.current_doc
            await turns.turn(lambda: _fire(phone, screen.chrome.next_button, "click"), 3)            # ▶
            screen.chrome.set_visible(True)
            events = len(screen.events)
            client.deliver(screen.webview._i, "GLRDR:" + json.dumps(
                {"type": "tap", "zone": "center", "seq": 70, "chapter": 2}))
            client.deliver(screen.webview._i, "GLRDR:" + json.dumps(
                {"type": "selection", "text": "rain", "seq": 71, "doc": stale_doc}))
            await asyncio.sleep(0.3)
            assert len(screen.events) == events and screen.chrome.visible and not screen.selection.visible
            await turns.tap(CENTRE)
            assert await _wait(lambda: not screen.chrome.visible, 10), "a centre tap on the new page was dropped"
            assert screen.events[-1].type == "tap" and screen.events[-1].chapter == 3

            await turns.turn(lambda: _fire(phone, screen.chrome.prev_button, "click"), 2)            # ◀
            screen.chrome.slider.value = 5                                                            # slider
            await turns.turn(lambda: _fire(phone, screen.chrome.slider, "change_end", 5), 5)
            tile = screen.toc.list_view.controls[0]                                                   # Chapters
            await turns.turn(lambda: _fire(phone, tile, "click"), 0)
            await turns.turn(lambda: client.do("click_link", f"chapter{LINK_TARGET:04d}"), LINK_TARGET - 1)  # link

            sheet = screen.open_search()                                                              # search
            sheet.start("EN-MARK-04")
            assert await _wait(lambda: sheet.state == "results", 20), sheet.status.value
            assert {int(r["chapter_idx"]) for r in sheet.rows} == {3}
            before = {id(e) for e in screen.events}
            await turns.turn(lambda: _fire(phone, sheet.results.controls[0], "click"), 3)
            assert await _wait(lambda: any(e.type == "found" and e.ok for e in turns.accepted_since(before)), 10)

            screen.chrome.mode_buttons.selected = ["original"]                                        # Original
            await turns.turn(lambda: _fire(phone, screen.chrome.mode_buttons, "change"), 3, mode="RAW")
            assert session.flavor == "original"
            screen.chrome.mode_buttons.selected = ["translated"]
            await turns.turn(lambda: _fire(phone, screen.chrome.mode_buttons, "change"), 3)

            screen.open_aa()                                                                          # Aa layout
            aa = screen.aa_sheet
            aa.layout_buttons.selected = ["scroll"]
            shown = await turns.turn(lambda: _fire(phone, aa.layout_buttons, "change"), 3)
            assert not shown["paged"] and screen.layout == "scroll"
            aa.layout_buttons.selected = ["single_page"]
            await turns.turn(lambda: _fire(phone, aa.layout_buttons, "change"), 3)
            aa.zones_switch.value = False                                                             # tap edges off
            await turns.turn(lambda: _fire(phone, aa.zones_switch, "change"), 3)
            screen.chrome.set_visible(False)
            loads = len(client.loads)
            await turns.tap(RIGHT)  # a side tap is now a centre tap: the chrome, never a page or chapter turn
            assert await _wait(lambda: screen.chrome.visible, 10)
            await asyncio.sleep(0.3)
            assert len(client.loads) == loads and screen.index == 3

            assert engine.exceptions == [], engine.exceptions  # the page scripts never threw
            screen.dispose()
            saved = phone.app.prefs.reader_position(screen.bid)
            assert saved["href"] == session.filenames[3] and saved["chapter"] == 3
        finally:
            await phone.stop()

    asyncio.run(scenario())


@pytest.mark.skipif(not _has("msgpack"), reason="msgpack not installed")
def test_owner_report_1_reopening_shows_the_saved_page(phone):
    """The saved position is the page that was on screen, so ▶ Continue reopens there (not one chapter
    further on), again and again; a plain open offers Resume, and Resume loads that page."""
    async def scenario():
        await phone.start()
        engine = _engine(phone)
        client = _attach_webview(phone, engine)
        snackbars: list = []
        app_module = sys.modules[type(phone.app).__module__]
        show_snackbar = app_module.show_snackbar

        def recording(page, message, **kwargs):
            bar = show_snackbar(page, message, **kwargs)
            snackbars.append((message, bar))
            return bar

        phone.monkeypatch.setattr(app_module, "show_snackbar", recording)
        try:
            screen = await _open(phone, chapter=LONG)
            turns = Turns(phone, client, screen)
            await turns.page_ready(set(), LONG)
            await turns.tap(RIGHT)
            assert await _wait(lambda: screen.page_no == 1, 10)
            await turns.tap(RIGHT)
            assert await _wait(lambda: screen.page_no == 2, 10)
            shown = await turns.check_shown(LONG)
            await _leave(phone, screen)
            saved = phone.app.prefs.reader_position(screen.bid)
            assert (saved["chapter"], saved["page"], saved["href"]) == (LONG, 2, screen.session.filenames[LONG])

            for _ in range(2):  # ▶ Continue, twice without moving: the same page both times
                screen = await _open(phone, resume=True)
                turns = Turns(phone, client, screen)
                assert client.loads[-1] == (screen.webview._i, screen.webview.url, "init")
                await turns.page_ready(set(), LONG)
                again = await turns.check_shown(LONG)
                assert again["page"] == shown["page"] == 2
                await _leave(phone, screen)
                assert phone.app.prefs.reader_position(screen.bid)["chapter"] == LONG

            # a plain open starts at the book's start and offers Resume; leaving untouched keeps the saved page
            for resume in (False, True):
                offers = len(snackbars)
                screen = await _open(phone)
                turns = Turns(phone, client, screen)
                await turns.page_ready(set(), 0)
                assert await _wait(lambda: any(m.startswith("Resume at Ch 2") for m, _ in snackbars[offers:]), 10)
                message, bar = [s for s in snackbars[offers:] if s[0].startswith("Resume at Ch 2")][-1]
                assert bar.action == "Resume"
                if resume:
                    shown = await turns.turn(lambda: _fire(phone, bar, "action"), LONG)
                    assert shown["page"] == 2
                await _leave(phone, screen)
                saved = phone.app.prefs.reader_position(screen.bid)
                assert (saved["chapter"], saved["page"]) == (LONG, 2)
        finally:
            await phone.stop()

    asyncio.run(scenario())


@pytest.mark.skipif(not _has("msgpack"), reason="msgpack not installed")
def test_owner_report_1_a_refused_load_or_quick_taps_still_end_on_the_right_page(phone):
    """A ``load_request`` the platform refuses builds a new WebView (a new Flutter State, so it loads
    its URL itself) that shows the chapter; two quick ▶ taps end on the second chapter's page."""
    async def scenario():
        await phone.start()
        engine = _engine(phone)
        client = _attach_webview(phone, engine)
        try:
            screen = await _open(phone, chapter=0)
            turns = Turns(phone, client, screen)
            await turns.page_ready(set(), 0)
            old = screen.webview
            client.fail_loads = 1
            before = {id(e) for e in screen.events}
            await _fire(phone, screen.chrome.next_button, "click")
            assert await _wait(lambda: screen.webview is not old and client.loads[-1][2] == "init", 10)
            new = screen.webview
            assert len(client.failed) == 1 and new._i != old._i and new.key != old.key
            assert client.loads[-1] == (new._i, new.url, "init") and screen.page_slot.content is new
            turns.webview = new
            await turns.page_ready(before, 1)
            await turns.check_shown(1)
            await turns.turn(lambda: _fire(phone, screen.chrome.next_button, "click"), 2)

            loads = len(client.loads)
            before = {id(e) for e in screen.events}
            await _fire(phone, screen.chrome.next_button, "click")
            await asyncio.sleep(0.05)  # a quick second tap (the first chapter change has started)
            await _fire(phone, screen.chrome.next_button, "click")
            await turns.page_ready(before, 4)
            await asyncio.sleep(0.3)
            assert 1 <= len(client.loads) - loads <= 2
            assert client.loads[-1] == (new._i, screen.webview.url, "load_request") and screen.webview is new
            await turns.check_shown(4)
        finally:
            await phone.stop()

    asyncio.run(scenario())


# =====================================================================================
# Issue 1 with the native ("Lightweight") reader
# =====================================================================================


@pytest.mark.skipif(not _has("msgpack"), reason="msgpack not installed")
def test_owner_report_1_lightweight_reader_turns_chapters_at_the_list_edges(phone):
    """The native reader (⋯ › Lightweight reader, or after a WebView failure): its texts are plain, so
    taps reach the tap zones and a paragraph's long-press its action sheet; an edge tap scrolls a screen
    and at the end / start of the list opens the next / previous chapter (a short chapter on the first
    tap); with tap edges off every tap is the chrome. Back to the page view, chapters load again."""
    import flet as ft

    from glossarion_mobile.ui.components.action_sheet import ActionSheet
    from glossarion_mobile.ui.reader import fallback_view

    async def scenario():
        await phone.start()
        phone.app.prefs.set("reader_prefs", {"lightweight": True})
        sheets: list = []
        show = ActionSheet.show

        def recording(self, page):
            sheets.append(self)
            return show(self, page)

        phone.monkeypatch.setattr(ActionSheet, "show", recording)
        holder: dict = {}

        def native_list() -> Any:
            screen = holder.get("screen")
            return screen.fallback.list_view if screen is not None else None

        lists = ListViewClient(phone.conn, phone.session, native_list)
        phone.closers.append(lists.close)
        try:
            screen = await _open(phone, chapter=0)
            holder["screen"] = screen
            session = screen.session
            assert screen.renderer == "native" and screen.webview is None
            assert screen.page_slot.content is screen.fallback.control

            def texts() -> list:
                return [t for c in screen.fallback.list_view.controls for t in _texts(c)]

            def shows(index: int) -> bool:
                """The native page shows chapter ``index`` (its text) and the Reader agrees."""
                body = " ".join(ListViewClient._text(screen.fallback.list_view))
                return (screen.index == index and screen.fallback_chapter == index
                        and screen.fallback_generation == screen.render_generation and ("EN", index) in _marks(body)
                        and {i for kind, i in _marks(body) if kind == "EN"} == {index})

            async def tap(x: float) -> None:  # TapUpDetails.toMap() (flet events.dart)
                point = {"x": float(x), "y": float(TAP_Y)}
                await _fire(phone, screen.fallback.control, "tap_up", {"k": "touch", "l": point, "g": dict(point)})

            def position_matches(index: int) -> None:
                saved = screen.save_position()
                assert saved is not None and saved["href"] == session.filenames[index] and saved["chapter"] == index

            assert await _wait(lambda: shows(0), 10)
            assert texts() and all(t.selectable is False for t in texts())
            # a paragraph's long-press opens its actions (a SelectableText would have taken the gesture)
            paragraph = next(c for c in screen.fallback.list_view.controls if isinstance(c, ft.GestureDetector))
            await _fire(phone, paragraph, "long_press")
            assert sheets and [i.label for i in sheets[-1].items][:1] == ["Copy"]
            sheets[-1].close()
            # the centre toggles the chrome
            screen.chrome.set_visible(True)
            await tap(CENTRE)
            assert await _wait(lambda: not screen.chrome.visible, 5)
            # chapter 1 is short: the first right tap turns to chapter 2 (it opens at its top)
            await tap(RIGHT)
            assert await _wait(lambda: shows(LONG), 10), "a right tap at the end of a short chapter did not turn"
            position_matches(LONG)
            assert await _wait(lambda: lists.pixels == 0.0 and screen.fallback.at_edge(-1) is not False, 5)
            # chapter 2 is long: right taps scroll a screen each, never skipping it; at its end the next turns
            maximum = lists.max_extent()
            assert maximum > 3 * HEIGHT * 0.9
            steps = 0
            while lists.pixels < maximum and steps < 40:
                at = lists.pixels
                await tap(RIGHT)
                assert await _wait(lambda: lists.pixels > at, 5), "a right tap did not scroll the long chapter"
                await asyncio.sleep(fallback_view.PAGE_SCROLL_MS / 1000 + fallback_view.PAGE_SETTLE + 0.15)
                assert screen.index == LONG
                steps += 1
            assert steps >= 3 and lists.pixels == maximum
            await tap(RIGHT)
            assert await _wait(lambda: shows(2), 10), "a right tap at the end of the long chapter did not turn"
            assert lists.calls[-1][:2] == (0, None)  # the next chapter opens at its top
            position_matches(2)
            # at the top a left tap opens the previous chapter at its end
            await tap(LEFT)
            assert await _wait(lambda: shows(LONG) and screen.last_page, 10)
            assert await _wait(lambda: lists.pixels == maximum, 5) and lists.calls[-1][:2] == (-1, None)
            position_matches(LONG)
            await tap(LEFT)  # a left tap at the end scrolls back a screen
            assert await _wait(lambda: lists.pixels < maximum, 5)
            await asyncio.sleep(fallback_view.PAGE_SCROLL_MS / 1000 + fallback_view.PAGE_SETTLE + 0.15)
            assert screen.index == LONG
            # tap edges off (Aa): every tap is the chrome
            screen.open_aa()
            aa = screen.aa_sheet
            aa.zones_switch.value = False
            await _fire(phone, aa.zones_switch, "change")
            assert await _wait(lambda: not screen.settings.tap_zones, 5)
            calls, visible = len(lists.calls), screen.chrome.visible
            await tap(RIGHT)
            assert await _wait(lambda: screen.chrome.visible != visible, 5)
            await asyncio.sleep(0.5)
            assert screen.index == LONG and len(lists.calls) == calls
            aa.zones_switch.value = True
            await _fire(phone, aa.zones_switch, "change")
            assert await _wait(lambda: screen.settings.tap_zones, 5)

            # ⋯ › Use the page view: a new WebView loads the chapter, and chapter changes load again
            engine = _engine(phone)
            client = _attach_webview(phone, engine)
            menu = screen.open_more()
            tile = next(t for t, item in zip(menu.tiles, menu.items) if item.label == "Use the page view")
            await _fire(phone, tile, "click")
            assert await _wait(lambda: screen.renderer == "webview" and screen.webview is not None
                               and screen.page_slot.content is screen.webview, 10)
            assert client.loads == [(screen.webview._i, screen.webview.url, "init")]
            turns = Turns(phone, client, screen)
            await turns.page_ready(set(), LONG)
            await turns.check_shown(LONG)
            await turns.turn(lambda: _fire(phone, screen.chrome.next_button, "click"), 2)
            screen.dispose()
        finally:
            await phone.stop()

    asyncio.run(scenario())


def _texts(control: Any) -> list:
    import flet as ft

    found = [control] if isinstance(control, ft.Text) else []
    for name in ("content", "controls"):
        child = getattr(control, name, None)
        for item in (child if isinstance(child, list) else [child] if child is not None else []):
            if hasattr(item, "__dict__"):
                found.extend(_texts(item))
    return found
