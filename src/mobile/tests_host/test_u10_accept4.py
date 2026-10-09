"""Acceptance test for U10 item 4: "Share links: transfer.it handoff, Gofile, Send (E2EE), pixeldrain".

The owner's decision (2026-10-09): "Share file via link" is tap-only; every provider is off until it is
turned on in Settings › Cloud sync & sharing behind a consent sheet that says the file leaves the phone
and whether the service can read it. transfer.it is a browser handoff (the EPUB made reachable, the start
page opened in the in-app browser, the link pasted back; NEVER a request to transfer.it); Gofile uses its
official API with one anonymous guest token (stored encrypted) and deletes through the folder id; Send
(send.vis.ee) encrypts on the phone with the key in the URL fragment; pixeldrain uses the user's own API key
(stored encrypted). Links are saved per book and shown on the Book page and on the chat's Result card with
Copy / Share / Delete (Remove where the service cannot delete); the only retry is Gofile's HTTP 429 with a
Retry-After (once); an upload that fails while the app is hidden posts one notification.

This file proves it end to end on the REAL app objects, headlessly, in ONE app session:

* ``GlossarionApp`` on a fake Flet session as an Android 14 phone (``test_bootstrap._fake_session``) with
  the real ``flet_glossarion_native.GlossarionNative`` service, whose platform calls (and the Clipboard,
  Share, UrlLauncher and PermissionHandler services') are answered by ``Phone`` below: the foreground
  service, notifications, MediaStore's Downloads/Glossarion (a taken name becomes "name (1).ext", as
  Android does), the clipboard and the share sheet. The UI is driven like the device flows
  (``host_tester.PyTester`` + ``ui_driver.UiDriver``): real taps on the visible tree, sheets included.
* The book is a real chat translation (New chat, ＋ › Files, Send, Start) against the offline fake model
  server (``diagnostics.fake_llm_server``); it moves into the Library by itself. A big TXT output
  (``*_translated.txt``, 12 MB) is added to its workspace, so "Which file?" and mid-upload progress show.
* The real ``ShareLinkService`` (``app.share_links``) with the real Gofile / Send / pixeldrain providers
  pointed at the local fakes of ``test_share_links.py`` (Gofile API, pixeldrain API, timvisee/send over a
  WebSocket); the Send link is decrypted with the independent reference implementation there.
* The E2E network guard (``diagnostics.e2e.ProcessGuards``) refuses every non-loopback connection for the
  whole test: nothing reaches transfer.it, gofile.io, pixeldrain.com or send.vis.ee.

Scenario: defaults off (Result card and Book page show "Turn on a service first"; nothing sent) ->
Settings: each switch opens the consent sheet (Cancel keeps it off; the checkbox gates "Turn on"), the
pixeldrain key saved encrypted and checked -> Result card: Which file? -> Send -> progress -> "Link ready"
(Copy / Share), decrypted by the reference recipient -> Book page (Open in Library › Output): the same link;
Send again offers the existing link; Gofile's 429 rule (one retry only with a short Retry-After);
Share / Delete (folder id + the guest token) and the guest token reused; pixeldrain with mid-upload progress
in the sheet and the foreground notification, and the notification's Stop; the transfer.it handoff (Downloads
copy, start page, a bad paste refused, Paste from the clipboard, Remove); a recompiled book handed off again;
an upload failing while the app is hidden -> Result card: every link of the book, Delete from the card ->
no secret in any file or log, no config.json key, the snapshots gone, nothing off the machine.

Defects that do not stop the rest of the scenario are collected and fail the test at the end, all listed.

Run from src/mobile with the mobile venv (keep the shell's PYTHONPATH):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_u10_accept4.py
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
TESTS_DIR = MOBILE_DIR / "tests"
EXTENSION_SRC = MOBILE_DIR / "extensions" / "flet_glossarion_native" / "src"
for _path in (APP_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _native_extension() -> bool:
    return _has("flet_glossarion_native") or (EXTENSION_SRC / "flet_glossarion_native" / "__init__.py").is_file()


def _tiktoken_assets() -> bool:
    folder = APP_DIR / "assets" / "tiktoken"
    return folder.is_dir() and any(p.is_file() and p.suffix == "" for p in folder.iterdir())


_NEEDED = ("flet", "msgpack", "ebooklib", "openai", "httpx", "tiktoken", "bs4", "lxml", "cryptography", "websockets")
pytestmark = [
    pytest.mark.skipif(not all(_has(m) for m in _NEEDED), reason=f"needs {', '.join(_NEEDED)} (the project venv)"),
    pytest.mark.skipif(not _native_extension(), reason="flet_glossarion_native extension sources not present"),
    pytest.mark.skipif(not _tiktoken_assets(), reason="app/assets/tiktoken is generated by tools/prepare_assets.py"),
]


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TB = _load("_glossarion_tb_helpers_u10a4", "test_bootstrap.py")
storage = _TB.storage
app_env = _TB.app_env

LEAVE = ("inactive", "hide", "pause")  # Android: Home pressed
COME_BACK = ("show", "resume")
BIG_TXT_BYTES = 12 * 1024 * 1024
PD_KEY = "pd-key-0123456789"  # PixeldrainFake's accepted key
TRANSFER_LINK = "https://transfer.it/t/AbCdEf_123-xyz"
SHARE_HOSTS = ("transfer.it", "gofile", "pixeldrain", "send.vis.ee", "mega")


# ==========================================================================
# The phone (platform side of every Flet service call)
# ==========================================================================


class Phone:
    """An Android 14 phone as the app's platform calls see it: ``GlossarionNative`` (foreground service,
    notifications, ``save_to_downloads`` with MediaStore's naming), ``Clipboard``, ``Share``,
    ``UrlLauncher`` and ``PermissionHandler``. Every call is recorded (``calls``)."""

    def __init__(self, downloads: Path) -> None:
        self.calls: list = []  # (control type, method, args)
        self.fgs: Optional[dict] = None  # the running foreground service
        self.fgs_log: list = []  # (method, title, text)
        self.posted: list = []  # show_notification args
        self.downloads = downloads  # Downloads/Glossarion
        self.entries: dict = {}  # content URI -> file
        self.saves: list = []  # (display name, uri, file, replace_uri)
        self.clipboard: Optional[str] = None
        self.lock = threading.Lock()

    def answer(self, control: str, method: str, args: Any) -> tuple:
        args = args if isinstance(args, dict) else {}
        with self.lock:
            self.calls.append((control, method, dict(args)))
        if control == "PermissionHandler":
            return ("granted" if method in ("get_status", "request") else True), None
        if control == "Clipboard":
            if method == "set":
                self.clipboard = args.get("data")
            return (self.clipboard if method == "get" else None), None
        if control == "Share":
            return {"status": "success", "raw": "test.share.target"}, None
        if control != "GlossarionNative":
            return None, None
        handler = getattr(self, "_n_" + method, None)
        return (handler(args), None) if handler is not None else (None, None)

    def calls_of(self, control: str, method: str) -> list:
        with self.lock:
            return [c[2] for c in self.calls if c[0] == control and c[1] == method]

    # ---- GlossarionNative ------------------------------------------------------------------------

    def _n_get_platform_info(self, args: dict) -> dict:
        return {"platform": "android", "sdk_int": 34, "notifications_enabled": True,
                "post_notifications_granted": True, "fgs_types": ["dataSync"], "save_to_downloads": True}

    def _n_init_notifications(self, args: dict) -> bool:
        return True

    def _n_show_notification(self, args: dict) -> bool:
        self.posted.append(dict(args))
        return True

    def _n_start_job_service(self, args: dict) -> bool:
        self.fgs = {"title": args.get("title"), "text": args.get("text")}
        self.fgs_log.append(("start", args.get("title"), args.get("text")))
        return True

    def _n_update_job_service(self, args: dict) -> bool:
        if self.fgs is None:
            return False
        self.fgs.update({k: v for k, v in (("title", args.get("title")), ("text", args.get("text"))) if v is not None})
        self.fgs_log.append(("update", args.get("title"), args.get("text")))
        return True

    def _n_stop_job_service(self, args: dict) -> bool:
        self.fgs = None
        self.fgs_log.append(("stop", None, None))
        return True

    def _n_is_job_service_running(self, args: dict) -> bool:
        return self.fgs is not None

    def _n_save_to_downloads(self, args: dict) -> Optional[str]:
        """MediaStore Downloads/<subdir>: a new entry per call (a taken name becomes "name (1).ext"); with
        ``replace_uri`` of an earlier entry that entry is overwritten in place."""
        import shutil

        source = Path(str(args.get("path") or ""))
        name = str(args.get("display_name") or source.name)
        replace = args.get("replace_uri")
        if not source.is_file() or str(args.get("subdir") or "") != "Glossarion":
            return None
        if replace and replace in self.entries:
            shutil.copyfile(source, self.entries[replace])
            self.saves.append((name, replace, self.entries[replace], replace))
            return replace
        stem, ext = os.path.splitext(name)
        target = self.downloads / name
        counter = 1
        while target.exists():
            target = self.downloads / f"{stem} ({counter}){ext}"
            counter += 1
        shutil.copyfile(source, target)
        uri = f"content://media/external/downloads/{1000 + len(self.entries)}"
        self.entries[uri] = target
        self.saves.append((name, uri, target, None))
        return uri


def _install_phone(conn: Any, session: Any, phone: Phone) -> None:
    """Answer the app's invoke_method calls from ``phone`` (the plain fake client answers None)."""
    from flet.messaging.protocol import MessageAction

    pending: dict = {}
    send = conn.send_message

    def send_message(message: Any) -> None:
        if message.action == MessageAction.INVOKE_METHOD:
            pending[message.body.call_id] = message.body
        send(message)

    real_handle = type(session).handle_invoke_method_results

    def handle(control_id: Any, call_id: Any, result: Any, error: Any) -> None:
        body = pending.pop(call_id, None)
        if body is not None:
            control = session.index.get(control_id)
            try:
                result, error = phone.answer(type(control).__name__, body.name, body.args)
            except Exception as exc:  # pragma: no cover - a broken fake must show up
                result, error = None, f"phone: {exc!r}"
        real_handle(session, control_id, call_id, result, error)

    conn.send_message = send_message
    session.handle_invoke_method_results = handle


class GatedProvider:
    """A share provider whose next upload waits (on its worker thread) until ``go`` is set: the test can
    hide the app while the upload is running. Everything else is the wrapped provider's."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.info = inner.info
        self.armed = False
        self.waiting = threading.Event()
        self.go = threading.Event()

    def upload(self, *args: Any, **kwargs: Any) -> Any:
        if self.armed:
            self.armed = False
            self.waiting.set()
            self.go.wait(60)
        return self.inner.upload(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


# ==========================================================================
# Helpers
# ==========================================================================


async def _until(predicate: Any, timeout: float = 30.0, step: float = 0.05) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value or time.monotonic() >= deadline:
            return value
        await asyncio.sleep(step)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else ""


def _inside(path: Any, root: Any) -> bool:
    try:
        a = os.path.normcase(os.path.abspath(str(path)))
        b = os.path.normcase(os.path.abspath(str(root)))
        return os.path.commonpath([a, b]) == b
    except ValueError:
        return False


def _key(control: Any) -> Any:
    key = getattr(control, "key", None)
    return getattr(key, "value", key)


def _texts_under(control: Any) -> list:
    from host_tester import _children, _texts

    out: list = []
    stack = [control]
    while stack:
        node = stack.pop()
        if node is None or getattr(node, "visible", True) is False:
            continue
        out.extend(_texts(node))
        stack.extend(reversed(list(_children(node))))  # depth first, in screen order
    return out


def _visible(tester: Any, pattern: str) -> list:
    """``[(key, control)]`` of the visible controls whose key matches ``pattern`` (per-build keys)."""
    regex = re.compile(pattern)
    return [(_key(c), c) for c in tester._walk() if isinstance(_key(c), str) and regex.match(_key(c))]


def _control(tester: Any, key: str) -> Any:
    """The first visible control keyed ``key`` (None when there is none)."""
    return next((c for c in tester._walk() if _key(c) == key), None)


def _dialog(tester: Any, key: str) -> Any:
    """The open dialog ``key`` names: the U10 sheets keep ``key`` on their frame and give the dialog itself a
    per-build key (``U10Actions._sheet``: ``<key>-<n>``), so a sheet shown again never reuses a closing one's."""
    for dialog in reversed(list(getattr(getattr(tester.page, "_dialogs", None), "controls", None) or [])):
        if getattr(dialog, "open", False) and key in (_key(dialog), _key(getattr(dialog, "content", None))):
            return dialog
    return None


async def _tap_control(tester: Any, control: Any) -> None:
    """The tap Flutter would send to ``control`` (it must be in the visible tree and enabled)."""
    assert any(c is control for c in tester._walk()), f"{type(control).__name__} is not on screen"
    node = control
    while node is not None and getattr(node, "on_click", None) is None:
        node = getattr(node, "parent", None)
    assert node is not None, f"{type(control).__name__} has no click handler"
    assert not tester._disabled(node), f"{type(control).__name__} is disabled"
    await tester._dispatch(node, "click")
    await asyncio.sleep(0.15)


def _link_rows(tester: Any, prefix: str) -> list:
    """Saved-link rows on screen: ``[(build, index, texts)]`` for keys ``<prefix>-<build>-<index>``."""
    rows = []
    for key, control in _visible(tester, rf"^{re.escape(prefix)}-(\d+)-(\d+)$"):
        build, index = re.match(rf"^{re.escape(prefix)}-(\d+)-(\d+)$", key).groups()
        rows.append((build, index, _texts_under(control)))
    return rows


def _row_button(tester: Any, prefix: str, url: str, action: str) -> Optional[str]:
    """Key of the Copy / Share / Delete button of the saved-link row showing ``url``."""
    for build, index, texts in _link_rows(tester, prefix):
        if url in texts:
            return f"{prefix}-{build}-{action}-{index}"
    return None


def _row_labels(tester: Any, prefix: str, url: str) -> list:
    for build, index, texts in _link_rows(tester, prefix):
        if url in texts:
            return [t for t in texts if t in ("Copy", "Share", "Delete", "Remove")]
    return []


def _urls_in(texts: list) -> list:
    return [t for t in texts if t.startswith(("http://", "https://"))]


def _result_card(app: Any) -> Any:
    from glossarion_mobile.ui.chat.cards import JobCard

    view = app.chat_view
    cards = [c for c in list(view.transcript.cards) + [getattr(c, "card", c) for c in view.transcript.tail]
             if isinstance(c, JobCard) and c.phase.name == "done"]
    return cards[-1] if cards else None


def _card_share_button(card: Any) -> Any:
    return card.action_buttons.get("share_link") if card is not None else None


def _card_share_reason(card: Any) -> Optional[str]:
    from glossarion_mobile.ui.components.reason_chip import ReasonChip

    button = _card_share_button(card)
    if button is None:
        return "missing"
    for child in getattr(button, "controls", None) or []:
        if isinstance(child, ReasonChip):
            return child.reason
    return None


def _leaked(blob: str, secrets: dict) -> list:
    return [name for name, value in secrets.items() if value and value in blob]


# ==========================================================================
# The test
# ==========================================================================


def test_share_links_end_to_end_on_a_phone(app_env, tmp_path, monkeypatch, caplog):
    import flows
    from glossarion_mobile import runtime_bootstrap as rb
    from glossarion_mobile.diagnostics import fixtures
    from glossarion_mobile.diagnostics.e2e import ProcessGuards
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MODEL, FakeLLMServer
    from glossarion_mobile.services import share_links as sl
    from glossarion_mobile.ui.screens import cloud_sync as u10

    if not _has("flet_glossarion_native"):
        monkeypatch.syspath_prepend(str(EXTENSION_SRC))
    # Real-data isolation on top of the bootstrap's env contract (HOME, OUTPUT_DIRECTORY, Library, CONFIG_FILE)
    user = tmp_path / "user"
    for key, sub in (("USERPROFILE", "."), ("APPDATA", "AppData/Roaming"), ("LOCALAPPDATA", "AppData/Local")):
        folder = (user / sub).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(key, str(folder))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "OUTPUT_DIRECTORY", "GLOSSARION_LIBRARY_DIR",
                "GLOSSARION_DATA_DIR", "CONFIG_FILE"):
        assert os.environ.get(key) and _inside(os.environ[key], tmp_path), f"{key} is not inside the test's tmp dir"
    src_config = SRC_DIR / "config.json"
    src_config_md5 = _md5(src_config)
    caplog.set_level(logging.INFO)

    sl_helpers = _load("_glossarion_sl_helpers_u10a4", "test_share_links.py")
    tf = _load("_glossarion_tf_helpers_u10a4", "test_ui_foundations.py")
    picks = tmp_path / "picks"
    picks.mkdir()
    epub_pick = fixtures.build_tiny_epub(picks / "share_novel.epub", chapters=3)
    downloads = tmp_path / "Downloads" / "Glossarion"
    downloads.mkdir(parents=True)
    data_dir = Path(rb.get_paths().data)
    problems: list = []

    # every progress sheet repaint (U10Actions._push of the bar / line), recorded and passed through
    pushes: list = []
    real_push = u10.U10Actions._push

    def push_spy(*controls: Any) -> None:
        for control in controls:
            key = _key(control)
            if key == "share-progress-bar":
                pushes.append(("bar", control.value))
            elif key == "share-progress-text":
                pushes.append(("text", control.value))
        real_push(*controls)

    monkeypatch.setattr(u10.U10Actions, "_push", staticmethod(push_spy))

    guards = ProcessGuards()
    record = guards._record

    def record_with_origin(bucket: list, **event: Any) -> None:
        # the guard keeps the last 12 frames (often all stdlib); add the Glossarion frames that asked
        import traceback

        event["origin"] = [f"{os.path.relpath(f.filename, SRC_DIR)}:{f.lineno} {f.name}"
                           for f in traceback.extract_stack()[:-2] if _inside(f.filename, SRC_DIR)][-12:]
        record(bucket, **event)

    guards._record = record_with_origin
    guards._install_network_guard()

    gofile = sl_helpers.GofileFake()
    pixeldrain = sl_helpers.PixeldrainFake(keys=(PD_KEY,))
    send = sl_helpers.SendFake()

    async def scenario(server: Any) -> None:
        from host_tester import HostPicker, PyTester
        from ui_driver import UiDriver

        from glossarion_mobile.app import GlossarionApp

        phone = Phone(downloads)
        notes: list = []
        real_notify = GlossarionApp.notify

        def notify(self: Any, message: Any, action_label: Any = None, on_action: Any = None) -> Any:
            notes.append(str(message))
            return real_notify(self, message, action_label, on_action)

        # every snackbar (still shown: the spy calls through); patched on the class before the app starts, so
        # the contexts that keep ``app.notify`` (Settings, Library, chat) record too
        monkeypatch.setattr(GlossarionApp, "notify", notify)
        main_module = tf._load_main_module()
        conn, session = _TB._fake_session("android")
        _install_phone(conn, session, phone)
        session.apply_page_patch({"width": 412, "height": 860})
        page = session.page
        await main_module.main(page)
        await session.after_event(page)
        app = page.data
        picker = HostPicker({epub_pick.name: epub_pick})
        app.files._get_picker = lambda: picker

        async def back() -> None:
            views = list(page.views or [])
            if len(views) > 1:
                await session.dispatch_event(page._i, "view_pop", {"route": views[-1].route})

        tester = PyTester(session, page)
        d = UiDriver(tester, picker=picker, back=back, poll_ms=100, log=lambda *_a: None)
        dismiss_stop = asyncio.Event()

        async def client_dismisses() -> None:
            """What the Flutter client does once a dialog's ``open`` turns off: its route closes and the
            client sends ``dismiss`` (Flet then drops it from ``page._dialogs``). The plain fake client
            never does, so a sheet shown again under the same key would be diffed against the stale one."""
            sent: set = set()
            while not dismiss_stop.is_set():
                for dialog in list(getattr(getattr(page, "_dialogs", None), "controls", None) or []):
                    if not getattr(dialog, "open", False) and id(dialog) not in sent:
                        sent.add(id(dialog))
                        try:
                            await session.dispatch_event(dialog._i, "dismiss", None)
                        except Exception:  # pragma: no cover - a dialog already gone
                            pass
                await asyncio.sleep(0.05)

        dismisser = asyncio.ensure_future(client_dismisses())

        async def lifecycle(*states: str) -> None:
            for state in states:
                await session.dispatch_event(page._i, "app_lifecycle_state_change", {"state": state})
            await asyncio.sleep(0.1)

        async def note(text: str, timeout: float = 15.0) -> bool:
            return bool(await _until(lambda: any(text in n for n in notes), timeout))

        async def open_book_output() -> Any:
            """Result card › Open in Library -> the Book page's Output tab (its saved links loaded)."""
            await flows.go_home(d)
            card = await _until(lambda: _result_card(app), 30)
            assert card is not None, "the chat shows no Result card"
            await _tap_control(tester, card.action_buttons["library"])
            await d.wait(text="Output", timeout=30)
            await d.tap(text="Output")
            screen = await _until(lambda: getattr(app.shell, "top_screen", None) if type(
                getattr(app.shell, "top_screen", None)).__name__ == "BookPageScreen" else None, 30)
            assert screen is not None, "Open in Library did not open the Book page"
            assert await _until(lambda: not screen.output.stale and screen.output.identity, 30)
            return screen

        async def provider_sheet_from_book(screen: Any, file_index: int) -> dict:
            """Book page › Output › Share file via link -> Which file? -> the file -> the provider sheet."""
            await d.tap(key=_visible(tester, r"^out-share-link-\d+$")[0][0], timeout=30)
            await d.tap(key=f"share-file-{file_index}", timeout=30)
            await d.wait(key="share-settings", timeout=30)
            return {key[len("share-provider-"):]: control for key, control in _visible(tester, r"^share-provider-")}

        async def wait_link_sheet(timeout: float = 60.0) -> str:
            await d.wait(key="share-link-ready", timeout=timeout)
            sheet = _dialog(tester, "share-link-ready")
            return next(t for t in _texts_under(sheet) if t.startswith(("http://", "https://")))

        def svc_state() -> Any:
            return app.share_links.state

        stop_events: list = []
        events: list = []
        try:
            await flows.wait_home(d)
            assert app.key_status is not None and app.key_status.installed
            assert type(app.native.native).__name__ == "GlossarionNative" and not app.native.is_stub
            store = app.config_store
            store.set_many(flows.ui_config(server.url, FAKE_MODEL))
            store.flush()
            assert store.save_error is None, store.save_error
            shares = app.share_links
            assert shares is not None, "app.share_links was not installed"
            # the service's providers -> the local fakes (the only seam: the real provider classes, their URLs)
            gated_gofile = GatedProvider(gofile.provider())
            shares._providers = {"gofile": gated_gofile, "send": send.provider(), "pixeldrain": pixeldrain.provider()}
            unsub = shares.subscribe(lambda kind: events.append((kind, svc_state().phase, svc_state().provider,
                                                                 svc_state().sent, svc_state().total)))
            stop_events.append(unsub)

            # ---- the book: a chat translation that moves into the Library ----------------------------------
            await flows.chat_translate_and_migrate(d, epub=epub_pick.name, timeout=300)
            assert await _until(lambda: _result_card(app), 30), "no Result card after the chat run"

            def library_state() -> Optional[dict]:
                current = _result_card(app)
                bound = dict(getattr(current, "u10_state", None) or {})
                return bound if bound.get("in_library") and bound.get("workspace") else None

            state = await _until(library_state, 30)
            assert state, ("the Result card was never bound to the book's Library workspace "
                           f"(ChatFeature.bind_u10_card): {getattr(_result_card(app), 'u10_state', None)}")
            workspace = str(state["workspace"])
            epubs = [p for p, k in state["outputs"] if k == "epub"]
            assert len(epubs) == 1, state["outputs"]
            epub = epubs[0]
            assert _inside(epub, tmp_path) and "Attachments" not in Path(workspace).parts
            big_txt = Path(workspace) / "share_novel_translated.txt"
            line = b"The translated chapter text goes on and on for the share-link progress test.\n"
            big_txt.write_bytes((line * (BIG_TXT_BYTES // len(line) + 1))[:BIG_TXT_BYTES])
            config_keys_before = set(json.loads(Path(os.environ["CONFIG_FILE"]).read_text("utf-8")))

            # ---- defaults: every provider off, nothing sent --------------------------------------------------
            assert [s.id for s in shares.provider_states()] == ["transferit", "gofile", "send", "pixeldrain"]
            assert all(not s.enabled and not s.consented for s in shares.provider_states())
            assert await _until(lambda: _card_share_reason(_result_card(app)) == u10.SHARE_OFF_REASON, 15), \
                f"Result card share action: {_card_share_reason(_result_card(app))!r}"
            screen = await open_book_output()
            assert screen.output.share_reason() == u10.SHARE_OFF_REASON
            texts = _texts_under(screen.output.cloud_column)
            assert u10.SHARE_LABEL in texts and u10.SHARE_OFF_REASON in texts, texts
            share_button = _visible(tester, r"^out-share-link-\d+$")[0][1]
            assert type(share_button).__name__ == "Row" and share_button.controls[0].disabled, \
                "Share file via link is tappable while every service is off"
            # the output row's ⋮ sheet keeps the action, disabled with its reason
            sheet = screen.output.open_sheet(epub, "epub")
            assert sheet.item(u10.SHARE_LABEL).disabled_reason == u10.SHARE_OFF_REASON
            sheet.close()
            await asyncio.sleep(0.2)
            assert gofile.requests == [] and pixeldrain.requests == [] and send.uploads == []

            # ---- Settings › Cloud sync & sharing: each switch behind its consent sheet ------------------------
            await flows.open_settings(d)
            await d.tap(key="hub-settings.cloud", timeout=60, scroll=True)
            cloud = await _until(lambda: app.shell.top_screen if type(app.shell.top_screen).__name__ ==
                                 "CloudSyncScreen" and app.shell.top_screen.loaded and
                                 app.shell.top_screen.provider_switches else None, 30)
            assert cloud is not None, "Settings › Cloud sync & sharing did not load its providers"
            assert {pid: sw.value for pid, sw in cloud.provider_switches.items()} == {
                "transferit": False, "gofile": False, "send": False, "pixeldrain": False}

            async def switch_on(pid: str, *, accept: bool = True) -> list:
                switch = cloud.provider_switches[pid]
                await d.tap(key=_key(switch), timeout=30)
                await d.wait(key="share-consent", timeout=30)
                sheet = _dialog(tester, "share-consent")
                lines = _texts_under(sheet)
                confirm = (await d.find(key="consent-confirm"))
                assert tester.control(confirm.first).disabled, "Turn on is enabled before the checkbox"
                if not accept:
                    await d.tap(key="consent-cancel")
                else:
                    await d.tap(key="consent-rights")
                    assert not tester.control((await d.find(key="consent-confirm")).first).disabled
                    await d.tap(key="consent-confirm")
                await d.wait(key="share-consent", gone=True, timeout=30)
                await _until(lambda: shares.provider_state(pid).enabled is accept, 15)
                return lines

            lines = await switch_on("gofile", accept=False)
            await asyncio.sleep(0.5)
            assert not shares.provider_state("gofile").enabled and not shares.provider_state("gofile").consented
            assert await _until(lambda: cloud.provider_switches["gofile"].value is False, 10)
            joined = " ".join(lines)
            assert "Share file via link sends the file you pick from this phone to Gofile" in joined, lines
            assert "Not end-to-end encrypted: Gofile can read the file." in joined
            assert "IP address" in joined and "developers receive nothing" in joined
            lines = await switch_on("gofile")
            assert shares.provider_state("gofile").ready
            joined = " ".join(await switch_on("send"))
            assert "End-to-end encrypted" in joined and "Send (send.vis.ee) cannot read the file" in joined, joined
            joined = " ".join(await switch_on("transferit"))
            assert "sends nothing to transfer.it" in joined and "Not end-to-end encrypted" in joined, joined
            joined = " ".join(await switch_on("pixeldrain"))
            assert "your own pixeldrain account" in joined, joined
            assert shares.provider_state("pixeldrain").reason == \
                "Add your pixeldrain API key in Settings › Cloud sync & sharing"
            # the pixeldrain key: typed, saved encrypted, cleared from the screen, checked (one request)
            field_key = await _until(lambda: _key(cloud.key_fields.get("pixeldrain")), 15)
            await d.enter(PD_KEY, key=field_key, timeout=15)
            await d.tap(key=_visible(tester, r"^share-key-save-pixeldrain-\d+$")[0][0])
            assert await note("Key saved (encrypted)"), notes[-5:]
            assert await _until(lambda: shares.provider_state("pixeldrain").ready, 15)
            assert all(not (f.value or "") for f in cloud.key_fields.values()), "the key stayed on screen"
            check = await _until(lambda: _visible(tester, r"^share-key-check-pixeldrain-\d+$"), 15)
            await d.tap(key=check[0][0])
            assert await note("The key works"), notes[-5:]
            sidecar = json.loads((data_dir / sl.STORE_FILE).read_text("utf-8"))
            assert sidecar["secrets"]["pixeldrain_key"].startswith("ENC:")
            import api_key_encryption  # shared: the app's key (Keystore / Keychain on a phone)

            assert api_key_encryption.get_handler().decrypt_value(sidecar["secrets"]["pixeldrain_key"]) == PD_KEY
            assert cloud.send_options == {"expire": 259200, "downloads": 20}, cloud.send_options
            assert {pid for pid, e in sidecar["providers"].items() if e.get("enabled") and e.get("consent_version")} \
                == {"transferit", "gofile", "send", "pixeldrain"}

            # ---- Result card: Which file? -> Send (end-to-end encrypted) ------------------------------------
            await flows.go_home(d)
            assert await _until(lambda: _card_share_reason(_result_card(app)) is None, 20), \
                f"the Result card's share action stays off: {_card_share_reason(_result_card(app))!r}"
            await _tap_control(tester, _card_share_button(_result_card(app)))
            await d.wait(key="share-file-1", timeout=30)
            choices = [(_key(c), _texts_under(c)) for _k, c in _visible(tester, r"^share-file-\d+$")]
            assert [k for k, _t in choices] == ["share-file-0", "share-file-1"], choices
            assert any(Path(epub).name in " ".join(t) for _k, t in choices[:1]), choices
            await d.tap(key="share-file-0")
            await d.wait(key="share-settings", timeout=30)
            menu = _visible(tester, r"^share-provider-")
            assert [k for k, _c in menu] == ["share-provider-transferit", "share-provider-gofile", "share-provider-send",
                                             "share-provider-pixeldrain"], menu
            from glossarion_mobile.ui.components.reason_chip import ReasonChip

            for key, control in menu:
                assert not isinstance(getattr(control, "trailing", None), ReasonChip), f"{key} is off: {control.trailing}"
            assert "Open transfer.it…" in _texts_under(menu[0][1])
            await d.tap(key="share-provider-send")
            send_url = await wait_link_sheet()
            send_link_base, _, send_key = send_url.partition("#")
            assert re.match(rf"^{re.escape(send.public)}/download/[0-9a-f]+/$", send_link_base) and len(send_key) == 22
            metadata, plain = sl_helpers.ref_receive(send_url)
            assert plain == Path(epub).read_bytes(), "the Send link does not decrypt to the EPUB"
            assert metadata["name"] == Path(epub).name and metadata["type"] == "application/epub+zip"
            received = b"".join(m if isinstance(m, bytes) else m.encode() for m in send.raw)
            assert base64.urlsafe_b64decode(send_key + "==") not in received and send_key.encode() not in received
            assert Path(epub).name.encode("utf-8") not in received
            assert send.uploads[-1]["timeLimit"] == 259200 and send.uploads[-1]["dlimit"] == 20
            send_owner = next(iter(send.files.values()))["owner"]  # the delete handle (stored encrypted)
            await d.tap(key="link-copy")
            assert await _until(lambda: phone.clipboard == send_url, 10), "Copy did not put the link on the clipboard"
            await d.tap(key="link-share")
            assert await _until(lambda: any(a.get("text") == send_url for a in phone.calls_of("Share", "share_text")),
                                10), "Share did not open the share sheet with the link"
            await d.tap(key="link-done")
            await d.wait(key="share-link-ready", gone=True, timeout=15)
            # progress: preparing -> uploading -> finishing -> done, sizes up to the file's
            send_events = [e for e in events if e[0] == "upload" and e[2] == "send"]
            phases = [e[1] for e in send_events]
            assert phases[0] == "preparing" and phases[-1] == "done" and "uploading" in phases, phases
            assert [e[3] for e in send_events if e[1] in ("uploading", "finishing")][-1] == os.path.getsize(epub)
            # the card lists the link with Copy / Share / Delete
            assert await _until(lambda: _row_button(tester, "job-link", send_url, "copy") is not None, 20), \
                f"the Result card does not list the Send link: {_link_rows(tester, 'job-link')}"
            assert _row_labels(tester, "job-link", send_url) == ["Copy", "Share", "Delete"]
            phone.clipboard = None
            await d.tap(key=_row_button(tester, "job-link", send_url, "copy"))
            assert await _until(lambda: phone.clipboard == send_url, 10)

            # ---- Book page: the same link; Send again offers it -------------------------------------------
            screen = await open_book_output()
            assert await _until(lambda: _row_button(tester, "out-link", send_url, "copy") is not None, 20), \
                f"the Book page does not list the Result card's link: {_link_rows(tester, 'out-link')}"
            uploads_before = len(send.uploads)
            await provider_sheet_from_book(screen, 0)
            await d.tap(key="share-provider-send")
            await d.wait(key="share-link-ready", timeout=30)
            sheet_texts = _texts_under(_dialog(tester, "share-link-ready"))
            assert "You already have a link for this file" in sheet_texts and send_url in sheet_texts, sheet_texts
            assert (await d.count(key="link-again")) == 1
            await d.tap(key="link-done")
            assert len(send.uploads) == uploads_before, "the existing link was uploaded again without asking"

            # ---- Gofile: the 429 rule (one retry, only with a short Retry-After) --------------------------
            busy = (429, {"status": "error-rateLimit", "data": {}})

            async def gofile_attempt(script: list, file_index: int = 0) -> tuple:
                before = len([r for r in gofile.requests if r[0] == "POST"])
                gofile.script[:] = script
                mark = len(notes)
                await provider_sheet_from_book(screen, file_index)
                await d.tap(key="share-provider-gofile")
                ok = await _until(lambda: _dialog(tester, "share-link-ready") is not None or any(
                    n.startswith("Gofile: ") for n in notes[mark:]), 60)
                assert ok, f"the Gofile upload never ended: {svc_state()}"
                posts = len([r for r in gofile.requests if r[0] == "POST"]) - before
                failed = next((n for n in notes[mark:] if n.startswith("Gofile: ")), None)
                return posts, failed

            posts, failed = await gofile_attempt([(*busy, {"Retry-After": "0"}), (*busy, {"Retry-After": "0"})])
            assert posts == 2 and failed == "Gofile: Gofile is busy (too many requests). Try again later.", \
                (posts, failed)
            posts, failed = await gofile_attempt([(*busy, {})])
            assert posts == 1 and failed and "Try again later" in failed, (posts, failed)
            posts, failed = await gofile_attempt([(*busy, {"Retry-After": "3600"})])
            assert posts == 1 and failed == "Gofile: Gofile is busy (too many requests). Try again in about 60 min.", \
                (posts, failed)
            assert not [r for r in shares.links_blocking() if r.provider == "gofile"]
            posts, failed = await gofile_attempt([(*busy, {"Retry-After": "1"})])
            assert posts == 2 and failed is None, (posts, failed)
            gofile_url = await wait_link_sheet()
            assert re.match(r"^https://gofile\.io/d/[0-9a-f]+$", gofile_url), gofile_url
            await d.tap(key="link-done")
            uploaded = [f for f in gofile.folders.values() if f["content"] == Path(epub).read_bytes()]
            assert len(uploaded) == 1 and uploaded[0]["name"] == Path(epub).name
            first_token = uploaded[0]["token"]
            first_folder = next(k for k, f in gofile.folders.items() if f is uploaded[0])
            posts_auth = [r[2] for r in gofile.requests if r[0] == "POST"]
            assert all(a == "" for a in posts_auth), "a guest token was sent before Gofile gave one"
            assert await _until(lambda: _row_button(tester, "out-link", gofile_url, "share") is not None, 20)
            assert _row_labels(tester, "out-link", gofile_url) == ["Copy", "Share", "Delete"]
            await d.tap(key=_row_button(tester, "out-link", gofile_url, "share"))
            assert await _until(lambda: any(a.get("text") == gofile_url for a in phone.calls_of("Share", "share_text")),
                                10)
            await d.tap(key=_row_button(tester, "out-link", gofile_url, "copy"))
            assert await _until(lambda: phone.clipboard == gofile_url, 10)
            # Delete: the upload's folder (by its id, with the token that made it), then the record
            await d.tap(key=_row_button(tester, "out-link", gofile_url, "delete"))
            await d.wait(text="Delete the upload?", timeout=15)
            await d.tap(text="Delete")
            assert await note("Upload deleted"), notes[-5:]
            assert not [f for f in gofile.folders.values() if f["content"] == Path(epub).read_bytes()]
            method, route, auth, body = gofile.requests[-1]
            assert (method, route, auth) == ("DELETE", "/contents", f"Bearer {first_token}") and set(body) == {
                "contentsId"}, gofile.requests[-1]
            assert await _until(lambda: _row_button(tester, "out-link", gofile_url, "copy") is None, 15)
            # the guest account is reused (one per install), and kept encrypted
            await gofile_attempt([])
            gofile_url = await wait_link_sheet()
            await d.tap(key="link-done")
            assert gofile.requests[-1][0] == "POST" and gofile.requests[-1][2] == f"Bearer {first_token}", \
                "the second Gofile upload did not reuse the guest token"
            sidecar = json.loads((data_dir / sl.STORE_FILE).read_text("utf-8"))
            assert sidecar["secrets"]["gofile_token"].startswith("ENC:")

            # ---- pixeldrain (the user's key): progress in the sheet and the notification ---------------------
            pixeldrain.gate = threading.Event()
            pixeldrain.started.clear()
            pushes.clear()
            fgs_mark = len(phone.fgs_log)
            await provider_sheet_from_book(screen, 1)
            await d.tap(key="share-provider-pixeldrain")
            assert await _until(pixeldrain.started.is_set, 30), "the pixeldrain upload never reached the server"
            await d.wait(key="share-progress", timeout=15)
            assert phone.fgs is not None and phone.fgs["title"] == "Share file via link", phone.fgs
            assert phone.fgs["text"] == "pixeldrain · share_novel_translated.txt", phone.fgs
            await asyncio.sleep(1.5)  # the client is blocked mid-file: the next chunk reports progress
            mid = svc_state()
            assert mid.phase == "uploading" and 0 <= mid.sent < mid.total == BIG_TXT_BYTES, mid
            pixeldrain.gate.set()
            pd_url = await wait_link_sheet()
            await d.tap(key="link-done")
            file_id = pd_url.rsplit("/", 1)[1]
            assert pd_url == f"https://pixeldrain.com/u/{file_id}" and file_id in pixeldrain.files
            assert pixeldrain.files[file_id]["content"] == big_txt.read_bytes()
            assert pixeldrain.files[file_id]["key"] == PD_KEY
            bars = [v for kind, v in pushes if kind == "bar"]
            assert bars and bars[-1] == 1.0, bars
            if not any(isinstance(v, float) and 0.0 < v < 1.0 for v in bars):
                problems.append(f"the progress sheet showed no value between 0 and 100 % for a 12 MB upload: {bars}")
            assert any(kind == "text" and " of " in str(v) for kind, v in pushes), pushes
            updates = [row for row in phone.fgs_log[fgs_mark:] if row[0] == "update"]
            if not any("pixeldrain · " in str(row[2]) and "%" in str(row[2]) for row in updates):
                problems.append(f"the foreground notification never showed the upload's progress: {updates}")
            assert phone.fgs is None and phone.fgs_log[-1][0] == "stop", "the upload's foreground service stayed up"
            # the notification's Stop cancels an upload that holds the service alone
            pixeldrain.gate = threading.Event()
            pixeldrain.started.clear()
            files_before = dict(pixeldrain.files)
            await provider_sheet_from_book(screen, 1)
            await d.tap(key="share-provider-pixeldrain")
            await d.wait(key="link-again", timeout=30)  # "You already have a link for this file"
            await d.tap(key="link-again")
            assert await _until(pixeldrain.started.is_set, 30)
            await session.dispatch_event(app.native.native._i, "foreground", {"type": "button", "button_id": "stop"})
            assert await note("Upload cancelled", 30), notes[-5:]
            pixeldrain.gate.set()
            assert pixeldrain.files == files_before and phone.fgs is None
            assert not any((Path(shares.snapshot_root)).glob("*")), "an upload snapshot was left behind"

            # ---- transfer.it: the browser handoff (no request to transfer.it) ------------------------------
            await provider_sheet_from_book(screen, 0)
            await d.tap(key="share-provider-transferit")
            await d.wait(key="share-handoff", timeout=30)
            saved = phone.saves[-1]
            assert saved[0] == Path(epub).name and saved[2].read_bytes() == Path(epub).read_bytes()
            assert phone.calls_of("GlossarionNative", "save_to_downloads")[-1]["mime_type"] == "application/epub+zip"
            launches = [a.get("url") for a in phone.calls_of("UrlLauncher", "launch_url")]
            assert launches and launches[-1] == "https://transfer.it/start", launches
            handoff_texts = _texts_under(_dialog(tester, "share-handoff"))
            location = next(t for t in handoff_texts if t.startswith("The file: "))
            assert location == f"The file: Downloads › Glossarion › {Path(epub).name}", location
            assert any("Keep the page open" in t for t in handoff_texts), handoff_texts
            await d.tap(key="handoff-open")
            assert await _until(lambda: [a.get("url") for a in phone.calls_of("UrlLauncher", "launch_url")][-1:] == [
                "https://transfer.it/start"] and len(phone.calls_of("UrlLauncher", "launch_url")) == len(launches) + 1,
                10)
            await d.enter("https://example.com/t/AbCdEf123", key="handoff-link")
            await d.tap(key="handoff-save")
            error = tester.control((await d.find(key="handoff-error")).first)
            assert await _until(lambda: error.visible and "transfer.it/t/" in str(error.value), 10), error.value
            phone.clipboard = f"Here is the link: {TRANSFER_LINK}"  # what the user copied on transfer.it
            await d.tap(key="handoff-paste")
            assert await _until(lambda: getattr(_control(tester, "handoff-link"), "value", None) == phone.clipboard, 10)
            await d.tap(key="handoff-save")
            assert await note("transfer.it link saved on the book"), notes[-5:]
            assert await _until(lambda: _row_button(tester, "out-link", TRANSFER_LINK, "copy") is not None, 20)
            assert _row_labels(tester, "out-link", TRANSFER_LINK) == ["Copy", "Share", "Remove"]
            # a recompiled book handed off again: the file the hint names must be the new one
            fixtures.build_tiny_epub(Path(epub), chapters=4)
            await provider_sheet_from_book(screen, 0)
            await d.tap(key="share-provider-transferit")
            await d.wait(key="share-handoff", timeout=30)
            location = next(t for t in _texts_under(_dialog(tester, "share-handoff")) if t.startswith("The file: "))
            named = downloads / location.split(" › ")[-1]
            if not named.is_file() or named.read_bytes() != Path(epub).read_bytes():
                problems.append(
                    "transfer.it handoff after a recompile: the sheet tells the user to pick "
                    f"'{location[len('The file: '):]}', which is the OLD copy; the new one was saved as "
                    f"'{phone.saves[-1][2].name}' (FileBridge.save_to_downloads has no 'replace', so "
                    "TransferItHandoff._save never overwrites the entry; every handoff adds a copy to Downloads: "
                    f"{sorted(p.name for p in downloads.iterdir())})")
            await d.tap(key="handoff-close")
            await d.wait(key="share-handoff", gone=True, timeout=15)

            # ---- an upload that fails while the app is hidden: one notification, a route only --------------
            gated_gofile.armed = True
            gated_gofile.go.clear()
            gated_gofile.waiting.clear()
            mark_posted = len(phone.posted)
            await provider_sheet_from_book(screen, 1)
            await d.tap(key="share-provider-gofile")
            assert await _until(gated_gofile.waiting.is_set, 30)
            await d.tap(key="share-hide")
            await lifecycle(*LEAVE)
            assert app.jobs.background.app_visible is False
            gofile.script[:] = [(*busy, {})]
            gated_gofile.go.set()
            assert await _until(lambda: svc_state().phase == "failed", 30), svc_state()
            await asyncio.sleep(1.0)
            failure_notes = [p for p in phone.posted[mark_posted:] if "failed" in str(p.get("title") or "")]
            if not failure_notes:
                crashed = [r for r in caplog.records if r.getMessage() == "share-link listener failed"]
                cause = (f"{type(crashed[-1].exc_info[1]).__name__}: {crashed[-1].exc_info[1]}"
                         if crashed and crashed[-1].exc_info else "no listener error logged")
                problems.append(
                    "an upload that fails while the app is hidden posts no notification (GlossarionApp."
                    "_on_share_link_change calls shares.state(), but ShareLinkService.state is a property; the "
                    f"listener raised and ShareLinkService._emit swallowed it: {cause})")
            else:
                post = failure_notes[0]
                assert post["title"] == "Upload to Gofile failed" and post.get("channel_id") == "jobs.action", post
                assert "/library/book/" in str(post.get("payload")), post
                assert all(workspace not in str(v) and str(tmp_path) not in str(v) for v in post.values()), post
                assert len(failure_notes) == 1, failure_notes
            await lifecycle(*COME_BACK)

            # ---- back on the Result card: every link of the book; Delete from the card --------------------
            await flows.go_home(d)
            want = {send_url, gofile_url, pd_url, TRANSFER_LINK}
            assert await _until(lambda: want <= {u for _b, _i, t in _link_rows(tester, "job-link") for u in _urls_in(t)},
                                20), f"the Result card's links: {_link_rows(tester, 'job-link')}"
            assert _row_labels(tester, "job-link", TRANSFER_LINK) == ["Copy", "Share", "Remove"]
            assert _row_labels(tester, "job-link", pd_url) == ["Copy", "Share", "Delete"]
            await d.tap(key=_row_button(tester, "job-link", pd_url, "delete"))
            await d.wait(text="Delete the upload?", timeout=15)
            await d.tap(text="Delete")
            assert await note("Upload deleted"), notes[-5:]
            assert file_id not in pixeldrain.files
            await d.tap(key=_row_button(tester, "job-link", send_url, "delete"))
            await d.wait(text="Delete the upload?", timeout=15)
            await d.tap(text="Delete")
            assert await _until(lambda: not send.files, 15), "the Send upload was not deleted"
            await d.tap(key=_row_button(tester, "job-link", TRANSFER_LINK, "delete"))
            await d.wait(text="Remove the link?", timeout=15)
            await d.tap(text="Remove")
            assert await note("Link removed"), notes[-5:]
            assert await _until(lambda: {u for _b, _i, t in _link_rows(tester, "job-link") for u in _urls_in(t)}
                                == {gofile_url}, 20), _link_rows(tester, "job-link")
            remaining = [link.url for link in shares.links_blocking()]
            assert remaining == [gofile_url], remaining

            # ---- a service turned off again: listed off on the Result card, nothing uploads ---------------------
            await flows.open_settings(d)
            await d.tap(key="hub-settings.cloud", timeout=60, scroll=True)
            cloud = await _until(lambda: app.shell.top_screen if type(app.shell.top_screen).__name__ ==
                                 "CloudSyncScreen" and app.shell.top_screen.loaded and
                                 app.shell.top_screen.provider_switches else None, 30)
            assert cloud is not None and cloud.provider_switches["send"].value is True
            await d.tap(key=_key(cloud.provider_switches["send"]), timeout=30)
            assert await _until(lambda: not shares.provider_state("send").enabled, 15)
            assert _dialog(tester, "share-consent") is None, "turning a service off asked for consent"
            await flows.go_home(d)
            assert await _until(lambda: _card_share_reason(_result_card(app)) is None, 20)
            uploads_before = len(send.uploads)
            await _tap_control(tester, _card_share_button(_result_card(app)))
            await d.tap(key="share-file-0", timeout=30)
            await d.wait(key="share-settings", timeout=30)
            off = _control(tester, "share-provider-send")
            assert isinstance(off.trailing, ReasonChip) and off.trailing.reason == u10.SERVICE_OFF_REASON, off.trailing
            await d.tap(key="cancel")
            await d.wait(key="share-settings", gone=True, timeout=15)
            assert len(send.uploads) == uploads_before

            # ---- secrets, storage, network -----------------------------------------------------------------
            raw_sidecar = (data_dir / sl.STORE_FILE).read_text("utf-8")
            sidecar = json.loads(raw_sidecar)
            assert all(r["url"].startswith("ENC:") for r in sidecar["links"])
            assert all(str(r.get("delete") or "ENC:").startswith("ENC:") for r in sidecar["links"])
            secrets = {"pixeldrain key": PD_KEY, "gofile token": first_token, "send link": send_url,
                       "send key": send_key, "send owner token": send_owner, "gofile link": gofile_url,
                       "gofile code": gofile_url.rsplit("/", 1)[1], "gofile folder": first_folder,
                       "pixeldrain link": pd_url, "pixeldrain id": file_id, "transfer.it link": TRANSFER_LINK}
            leaks = []
            for root in {data_dir, Path(rb.get_paths().cache), Path(rb.get_paths().temp)}:
                for path in root.rglob("*"):
                    if path.is_file() and path.stat().st_size < 64 * 1024 * 1024:
                        blob = path.read_bytes().decode("utf-8", "replace")
                        leaks += [f"{name} in {path.relative_to(root)}" for name in _leaked(blob, secrets)]
            assert not leaks, leaks
            logs = caplog.text + "\n".join(str(getattr(x, "text", x)) for x in app.log_buffer.snapshot())
            assert not _leaked(logs, secrets), _leaked(logs, secrets)
            config_keys_after = set(json.loads(Path(os.environ["CONFIG_FILE"]).read_text("utf-8")))
            share_keys = [k for k in config_keys_after - config_keys_before
                          if any(w in k.lower() for w in ("share", "gofile", "pixeldrain", "send", "transfer", "link"))]
            assert not share_keys, f"config.json got share-link keys: {share_keys}"
            assert not list(Path(shares.snapshot_root).glob("*")) if os.path.isdir(shares.snapshot_root) else True
        finally:
            for unsub in stop_events:
                unsub()
            dismiss_stop.set()
            try:
                await asyncio.wait_for(dismisser, 5)
            except Exception:
                dismisser.cancel()
            try:
                app.jobs.close()
            finally:
                await tf._stop(app)

    try:
        with FakeLLMServer() as server:
            asyncio.run(scenario(server))
    finally:
        guards.uninstall()
        if pixeldrain.gate is not None:
            pixeldrain.gate.set()
        for fake in (gofile, pixeldrain, send):
            fake.close()
    # Nothing left the machine: no share service at all, and nothing else but the selected model's provider
    # catalog auto-poll (``model_options._fetch_provider_catalog``, desktop parity; refused like the rest).
    failures = list(problems)
    share_attempts = [(e.get("api"), e.get("target"), e.get("origin")) for e in guards.network
                      if any(h in str(e.get("target")).lower() for h in SHARE_HOSTS)]
    if share_attempts:
        failures.insert(0, f"a share service was contacted (the network guard refused it): {share_attempts}")
    others = [(e.get("api"), e.get("target"), (e.get("origin") or [])[-3:]) for e in guards.network
              if not any("_fetch_provider_catalog" in f for f in (e.get("origin") or []) + (e.get("stack") or []))
              and not any(h in str(e.get("target")).lower() for h in SHARE_HOSTS)]
    if others:
        failures.append(f"non-loopback network attempts: {others[:5]}")
    if _md5(src_config) != src_config_md5:
        failures.append("src/config.json changed")
    assert not failures, "U10 share-link acceptance:\n- " + "\n- ".join(failures)
