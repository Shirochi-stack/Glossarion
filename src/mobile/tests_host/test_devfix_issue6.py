"""Acceptance for the owner's device report #6 (2026-10-08): notifications.

What the owner saw on the U8 APK: no glossary-ready notification while the app was open, no way to
accept a generated glossary from the notification, no "always accept" option, the job notification
frozen once the app left the screen, the permission asked once and never again, and the job
notification gone for good after a swipe (Android 14+ lets users swipe a foreground service's
notification; the owner wants it back "like a VPN's").

The REAL app runs on a fake Flet session (``test_bootstrap._fake_session``) as an Android 14 phone:
the real ``flet_glossarion_native.GlossarionNative`` service and ``flet_permission_handler``, whose
platform calls are answered by ``AndroidDevice`` below (the Dart service, Kotlin plugin,
flutter_foreground_task and permission_handler as an Android 14 device answers them: no
POST_NOTIFICATIONS until the user grants it, a refused ``show_notification``, the foreground-service
notification with its Stop / Open buttons). Device events (notification taps and actions, the
foreground service's ``dismissed``, lifecycle) arrive through ``session.dispatch_event`` exactly like
the Flutter client sends them. The UI is driven with ``host_tester.PyTester`` + ``ui_driver.UiDriver``
(the device UI flows' driver). The jobs are real Balanced-glossary chat runs: JobService, HeadlessOwner
and the shared pipeline against the fake OpenAI server on 127.0.0.1 (``diagnostics.fake_llm_server``),
with the 12-chapter self-test EPUB attached in the chat; the glossary gate is the shared
``translation_pipeline`` Direct Text gate.

Runs (one app session):

1. default (nothing set): the first Start asks for the notification permission, the prompt gets no
   answer (an error) -> not remembered; the gate asks on the chat's approval card (on screen: no
   notification, no snackbar); ✓ Yes; the job finishes;
2. the next Start asks again -> granted; the gate arrives while the Library is on screen: snackbar
   "Glossary ready: review needed" (Review) + a ``jobs.action`` notification with Accept / Review;
   the notification's Accept answers the gate (through the chat's run controller) and the app cancels
   the notification; Done notification;
3. the gate arrives while the app is hidden (the UI pump parked): the notification with Accept, the
   foreground-service text says "waiting for your glossary decision"; a swipe on the job notification
   re-posts it at once (same service, Stop / Open kept) - also a second swipe; Accept; the app leaves
   again and the foreground-service text keeps counting chapters while hidden; after the job a swipe
   brings nothing back;
4. Settings › Notifications & background › "Always accept generated glossaries" on (Prefs, no
   config.json key): the next run never asks (no card, notification or snackbar) and logs the
   desktop's accepted line;
5. All chats off, Chat settings › This chat on (sidecar meta): never asks either;
6. All chats on, This chat off: asks again, on the chat the user looks at (its approval card is on
   screen: no notification, no snackbar); ■ No ends the run;
7. the owning chat is not on screen: the user opened it from the drawer, then tapped New chat while
   the glossary was generated: notification + snackbar;
8. the Notifications page reads the real state on every visit: On -> the user blocks notifications in
   the system settings -> "Blocked in system settings" + Open system settings -> On again.

Defects that do not stop the rest of the scenario are collected (``Harness.problems``) and fail the
test at the end, all of them listed.

Run from src/mobile with the mobile venv:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_issue6.py
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
TESTS_DIR = MOBILE_DIR / "tests"
SRC_DIR = MOBILE_DIR.parent
EXTENSION_SRC = MOBILE_DIR / "extensions" / "flet_glossarion_native" / "src"
SELFTEST_EPUB = APP_DIR / "assets" / "selftest" / "selftest_ko_12ch.epub"
for _path in (APP_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _tiktoken_assets() -> bool:
    folder = APP_DIR / "assets" / "tiktoken"
    return folder.is_dir() and any(p.is_file() and p.suffix == "" for p in folder.iterdir())


def _native_extension() -> bool:
    return _has("flet_glossarion_native") or (EXTENSION_SRC / "flet_glossarion_native" / "__init__.py").is_file()


pytestmark = [
    pytest.mark.skipif(not (_has("flet") and _has("msgpack") and _has("flet_permission_handler")),
                       reason="flet / msgpack / flet_permission_handler not installed"),
    pytest.mark.skipif(not _native_extension(), reason="flet_glossarion_native extension sources not present"),
    pytest.mark.skipif(not (_has("ebooklib") and _has("openai") and _has("tiktoken") and _has("bs4")),
                       reason="backend packages missing"),
    pytest.mark.skipif(not _tiktoken_assets(), reason="app/assets/tiktoken is generated by tools/prepare_assets.py"),
    pytest.mark.skipif(not SELFTEST_EPUB.is_file(), reason="app/assets/selftest is generated by tools/prepare_assets.py"),
]

_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_devfix6",
                                                  Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)
storage = _TB.storage
app_env = _TB.app_env

GATE = "direct_text_glossary_approval"
ACCEPTED_LINE = "✅ Direct Text: generated glossary accepted (Always accept is on)"
GLOSSARY_READY = "Glossary ready: review needed"
ALWAYS_ACCEPT = "Always accept generated glossaries"
LEAVE = ("inactive", "hide", "pause")  # Android: Home pressed
COME_BACK = ("show", "resume")


def _foundations():
    spec = importlib.util.spec_from_file_location("_glossarion_tf_helpers_devfix6",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else ""


# ==========================================================================
# The phone below the app
# ==========================================================================


class AndroidDevice:
    """An Android 14 phone as the app's platform calls see it.

    * ``GlossarionNative`` (native_service.dart + GlossarionNativePlugin.kt): ``get_platform_info``
      reports ``post_notifications_granted`` / ``notifications_enabled`` (areNotificationsEnabled is
      False on 13+ without POST_NOTIFICATIONS); ``show_notification`` returns False without the
      permission; ``cancel_notification``; ``start_job_service`` keeps the buttons;
      ``update_job_service`` (flutter_foreground_task ``updateService(title, text)``) returns False
      when no service runs, otherwise updates the text and posts the notification again with the
      service's buttons;
    * ``PermissionHandler`` (permission_handler): ``request`` shows the prompt and returns what the
      user did (``prompt_answers``: a status, or an exception for a prompt that never answered);
      ``get_status``; ``open_app_settings``.

    Everything else (SecureStorage, Wakelock, push_route, ...) answers None like the plain fake client.
    """

    def __init__(self) -> None:
        self.post_notifications = False
        self.app_notifications_on = True
        self.blocked = False
        self.prompt_answers: list = []
        self.calls: list = []  # (control type, method, args)
        self.posted: list = []  # every accepted show_notification (args)
        self.shade: dict = {}  # notification id -> args (what the notification shade shows)
        self.refused: list = []  # show_notification refused (no permission)
        self.cancelled: list = []  # cancel_notification ids (by the app)
        self.fgs = None  # the running foreground service {"title", "text", "buttons", "visible"}
        self.fgs_log: list = []  # (monotonic, method, text, app visible, dispatcher ticks)
        self.settings_opened = 0
        self.platform_reads = 0
        self.page = None
        self.dispatcher = None

    # ---- what the app asked ---------------------------------------------------------------------

    def answer(self, control: str, method: str, args) -> tuple:
        args = args if isinstance(args, dict) else {}
        self.calls.append((control, method, dict(args)))
        if control == "PermissionHandler":
            return self._permission(method, args)
        if control != "GlossarionNative":
            return None, None
        handler = getattr(self, "_n_" + method, None)
        return (handler(args), None) if handler is not None else (None, None)

    def requests(self, name: str = "notification") -> list:
        return [c for c in self.calls if c[0] == "PermissionHandler" and c[1] == "request"
                and _plain(c[2].get("permission")) == name]

    def notification_status(self) -> str:
        if self.post_notifications:
            return "granted"
        return "permanentlyDenied" if self.blocked else "denied"

    def _permission(self, method: str, args: dict) -> tuple:
        name = _plain(args.get("permission"))
        if method == "open_app_settings":
            self.settings_opened += 1
            return True, None
        if name != "notification":
            return "granted", None  # battery optimisation and the rest
        if method == "get_status":
            return self.notification_status(), None
        if method == "request":
            if self.post_notifications:
                return "granted", None
            if self.blocked:
                return "permanentlyDenied", None
            answer = self.prompt_answers.pop(0) if self.prompt_answers else "denied"
            if isinstance(answer, BaseException):
                return None, f"{type(answer).__name__}: {answer}"
            if answer == "granted":
                self.post_notifications = True
            elif answer == "permanentlyDenied":
                self.blocked = True
            return answer, None
        return None, None

    def _allowed(self) -> bool:
        return self.post_notifications and self.app_notifications_on

    def _visible(self) -> bool:
        return bool(getattr(self.page, "app_visible", True))

    def _ticks(self) -> int:
        return int(getattr(self.dispatcher, "ticks", -1))

    def _n_get_platform_info(self, args: dict) -> dict:
        self.platform_reads += 1
        return {"platform": "android", "sdk_int": 34, "notifications_enabled": self._allowed(),
                "post_notifications_granted": self.post_notifications, "fgs_types": ["dataSync"],
                "save_to_downloads": True}

    def _n_get_initial_shared(self, args: dict) -> list:
        return []

    def _n_init_notifications(self, args: dict) -> bool:
        return self._allowed()

    def _n_show_notification(self, args: dict) -> bool:
        if not self._allowed():
            self.refused.append(dict(args))
            return False
        self.posted.append(dict(args))
        self.shade[int(args["id"])] = dict(args)
        return True

    def _n_cancel_notification(self, args: dict) -> None:
        nid = int(args.get("id"))
        self.cancelled.append(nid)
        self.shade.pop(nid, None)

    def _n_get_launch_notification(self, args: dict) -> None:
        return None

    def _n_start_job_service(self, args: dict) -> bool:
        self.fgs = {"title": args.get("title"), "text": args.get("text"), "visible": True,
                    "buttons": [dict(b) for b in (args.get("buttons") or [])]}
        self.fgs_log.append((time.monotonic(), "start", args.get("text"), self._visible(), self._ticks()))
        return True

    def _n_update_job_service(self, args: dict) -> bool:
        if self.fgs is None:
            self.fgs_log.append((time.monotonic(), "update-refused", args.get("text"), self._visible(), self._ticks()))
            return False
        if args.get("title") is not None:
            self.fgs["title"] = args.get("title")
        if args.get("text") is not None:
            self.fgs["text"] = args.get("text")
        self.fgs["visible"] = True  # updateService re-posts the service notification (buttons kept)
        self.fgs_log.append((time.monotonic(), "update", args.get("text"), self._visible(), self._ticks()))
        return True

    def _n_stop_job_service(self, args: dict) -> bool:
        self.fgs = None
        self.fgs_log.append((time.monotonic(), "stop", None, self._visible(), self._ticks()))
        return True

    def _n_is_job_service_running(self, args: dict) -> bool:
        return self.fgs is not None

    # ---- what the user does on the phone ---------------------------------------------------------

    def action_notifications(self) -> list:
        return [a for a in self.posted if a.get("channel_id") == "jobs.action"]

    def updates_since(self, index: int) -> list:
        return [row for row in self.fgs_log[index:] if row[1] == "update"]


def _plain(value) -> str:
    return str(getattr(value, "value", value) or "")


def _install_device(conn, session, device: AndroidDevice) -> None:
    """Answer the app's invoke_method calls from ``device`` (the plain fake client answers None)."""
    from flet.messaging.protocol import MessageAction

    pending: dict = {}
    send = conn.send_message

    def send_message(message):
        if message.action == MessageAction.INVOKE_METHOD:
            pending[message.body.call_id] = message.body
        send(message)

    real_handle = type(session).handle_invoke_method_results

    def handle(control_id, call_id, result, error):
        body = pending.pop(call_id, None)
        if body is not None:
            control = session.index.get(control_id)
            try:
                result, error = device.answer(type(control).__name__, body.name, body.args)
            except Exception as exc:  # pragma: no cover - a broken fake must show up
                result, error = None, f"device: {exc!r}"
        real_handle(session, control_id, call_id, result, error)

    conn.send_message = send_message
    session.handle_invoke_method_results = handle


# ==========================================================================
# Helpers
# ==========================================================================


async def _until(predicate, timeout: float = 60.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            return value
        await asyncio.sleep(interval)


def _config(base_url: str, model: str) -> dict:
    """The desktop config the owner imports: the fake OpenAI endpoint and the welcome's "Balanced"
    glossary choice (the chat attachment run generates a glossary and asks before translating)."""
    import flows
    from glossarion_mobile.ui.screens.welcome_flow import welcome_glossary_updates

    values = flows.ui_config(base_url, model)
    values.update(welcome_glossary_updates("balanced"))
    return values


class Run:
    """One chat run from the UI: a new chat, ＋ › Files with the EPUB, Send, the Plan card's Start."""

    def __init__(self, harness: "Harness", name: str) -> None:
        self.h = harness
        self.name = name
        self.cid = None
        self.jid = None
        self.mark = 0
        self.notes_mark = 0
        self.device_mark = 0

    async def send(self) -> "Run":
        import flows

        d = self.h.driver
        await flows.go_home(d)
        await d.tap(tooltip="New chat")
        await d.tap(tooltip=flows.ATTACH_TOOLTIP)
        await d.pick_file(self.name, lambda: d.tap(key="attach-files"))
        await d.wait(contains=Path(self.name).stem, timeout=60)
        await d.tap(key="send-idle_ready", timeout=60)
        await d.wait(text="Ready to translate", timeout=60)
        self.cid = self.h.app.chat_view.cid
        return self

    async def start(self) -> "Run":
        from glossarion_mobile.services.background import PREF_BATTERY_PROMPT

        app, d = self.h.app, self.h.driver
        first_long_job = not app.prefs.get(PREF_BATTERY_PROMPT, False)
        self.mark = self.h.server.mark()
        self.notes_mark = len(self.h.notes)
        self.device_mark = len(self.h.device.posted)
        await d.tap(text="Start")
        if first_long_job:  # the one-time battery explanation; "Not now" starts the job without it
            await d.tap(text="Not now", timeout=30)
        runs = app.chat_feature.runs
        run = await _until(lambda: runs.run_for(self.cid) if getattr(runs.run_for(self.cid), "job_id", None) else None,
                           timeout=60)
        assert run is not None, f"{self.name}: the Start tap did not submit a job"
        self.jid = run.job_id
        return self

    def snapshot(self):
        return self.h.app.job_service.snapshot(self.jid)

    def question(self):
        return self.h.app.job_service.pending_question(self.jid)

    async def wait_question(self, timeout: float = 120.0) -> dict:
        question = await _until(self.question, timeout=timeout)
        assert question, f"{self.name}: the run never asked for glossary approval ({self._state()})"
        assert question["kind"] == GATE, question
        return question

    async def wait_done(self, timeout: float = 240.0):
        snap = await _until(lambda: (s := self.snapshot()) is not None and s.is_terminal and s, timeout=timeout)
        assert snap, f"{self.name}: the job did not end ({self._state()})"
        return snap

    def log_lines(self) -> list:
        service = self.h.app.job_service
        buffer = service.log_buffer(self.jid)
        lines = [line.text for line in buffer.snapshot()] if buffer is not None else []
        return lines or service.read_log_tail(self.jid)

    def notes(self) -> list:
        return [n[1] for n in self.h.notes[self.notes_mark:]]

    def action_posts(self) -> list:
        from glossarion_mobile.services.notifications import action_notification_id

        nid = action_notification_id(self.jid)
        return [a for a in self.h.device.posted[self.device_mark:] if int(a["id"]) == nid]

    def _state(self) -> str:
        snap = self.snapshot()
        return f"state={getattr(snap, 'state', None)} last={getattr(snap, 'last_line', '')!r}"


class Harness:
    def __init__(self, app, page, session, tester, driver, device, server, notes) -> None:
        self.app = app
        self.page = page
        self.session = session
        self.tester = tester
        self.driver = driver
        self.device = device
        self.server = server
        self.notes = notes
        self.problems: list = []  # owner-visible defects that do not block the rest of the scenario
        self.chat_answers: list = []  # (cid, accepted, answered) of ChatRuns.answer_glossary

    @property
    def native(self):
        return self.app.native.native

    async def lifecycle(self, *states: str) -> None:
        for state in states:
            await self.session.dispatch_event(self.page._i, "app_lifecycle_state_change", {"state": state})
        await asyncio.sleep(0.05)

    async def leave_app(self) -> None:
        await self.lifecycle(*LEAVE)

    async def return_to_app(self) -> None:
        await self.lifecycle(*COME_BACK)

    async def swipe_job_notification(self) -> None:
        """Android 14+: the user swipes the foreground-service notification away."""
        assert self.device.fgs is not None and self.device.fgs["visible"], "no job notification to swipe"
        self.device.fgs["visible"] = False
        await self.session.dispatch_event(self.native._i, "foreground", {"type": "dismissed"})

    async def press(self, notification: dict, action_id) -> None:
        """A tap on a ``jobs.action`` notification (``action_id`` None = its body). Android brings the
        app to the front (``PendingIntent.getActivity``); the plugin cancels it for an action button
        (and ``autoCancel`` for the body) and sends the ``notification`` event."""
        self.device.shade.pop(int(notification["id"]), None)
        await self.return_to_app()
        await self.session.dispatch_event(self.native._i, "notification", {
            "notification_id": int(notification["id"]), "action_id": action_id,
            "payload": notification.get("payload"), "launched_app": False})

    def config_keys(self) -> set:
        path = Path(self.app.paths.config_file)
        self.app.config_store.flush()
        return set(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else set()

    async def open_notifications_page(self):
        from glossarion_mobile.ui.screens.notifications import NotificationsScreen

        self.app.navigate_to("settings.notifications")
        await self.driver.wait(key="notif-status", timeout=30)
        screen = await _until(lambda: next((e for e in reversed(self._screens()) if isinstance(e, NotificationsScreen)),
                                           None), timeout=10)
        assert screen is not None, "the Notifications & background page did not open"
        await _until(lambda: screen.permission_state is not None, timeout=10)
        return screen

    def _screens(self) -> list:
        shell = self.app.shell
        out = []
        for entry in list(getattr(shell, "stack", None) or getattr(shell, "_stack", None) or []):
            out.append(getattr(entry, "screen", entry))
        current = getattr(shell, "current_screen", None)
        if current is not None:
            out.append(current)
        return out


# ==========================================================================
# The acceptance scenario
# ==========================================================================


def test_owner_report_6_notifications_on_a_phone(app_env, tmp_path, monkeypatch):
    import flows
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MODEL, FakeLLMServer

    if not _has("flet_glossarion_native"):
        monkeypatch.syspath_prepend(str(EXTENSION_SRC))
    # Real-data isolation on top of the bootstrap's env contract (HOME, OUTPUT_DIRECTORY, Library, CONFIG_FILE).
    for name in ("USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        monkeypatch.setenv(name, str(tmp_path / "home"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    src_config = SRC_DIR / "config.json"
    src_config_md5 = _md5(src_config)

    from glossarion_mobile import runtime_bootstrap as rb

    sandbox_config = str(rb.get_paths().config_file)
    app_paths = sys.modules.get("app_paths")
    if app_paths is not None and getattr(app_paths, "CONFIG_FILE", None) != sandbox_config:
        # imported before this test's bootstrap (another test file): the backend must not touch src/config.json
        old = app_paths.CONFIG_FILE
        for module in list(sys.modules.values()):
            try:
                if module is not None and vars(module).get("CONFIG_FILE") == old:
                    monkeypatch.setattr(module, "CONFIG_FILE", sandbox_config)
            except Exception:
                pass

    tf = _foundations()
    picks = tmp_path / "picks"
    picks.mkdir()
    names = [f"devfix6-run{i}.epub" for i in range(1, 8)]
    files = {}
    for name in names:
        shutil.copyfile(SELFTEST_EPUB, picks / name)
        files[name] = picks / name

    async def scenario(server):
        from host_tester import HostPicker, PyTester
        from ui_driver import UiDriver

        files[flows.CONFIG_NAME] = picks / flows.CONFIG_NAME
        files[flows.CONFIG_NAME].write_text(json.dumps(_config(server.url, FAKE_MODEL)), encoding="utf-8")
        device = AndroidDevice()
        device.prompt_answers = [RuntimeError("the first-launch dialog was dismissed without an answer"),  # U12
                                  RuntimeError("the permission prompt was dismissed without an answer"), "granted"]
        main_module = tf._load_main_module()
        conn, session = _TB._fake_session("android")
        _install_device(conn, session, device)
        session.apply_page_patch({"width": 412, "height": 860})
        page = session.page
        await main_module.main(page)
        await session.after_event(page)
        app = page.data
        device.page, device.dispatcher = page, app.dispatcher
        picker = HostPicker(files)
        app.files._get_picker = lambda: picker

        async def back():
            views = list(page.views or [])
            if len(views) > 1:
                await session.dispatch_event(page._i, "view_pop", {"route": views[-1].route})

        tester = PyTester(session, page)
        driver = UiDriver(tester, picker=picker, back=back, poll_ms=100, log=lambda *_a: None)
        notes: list = []
        real_notify = app.notify

        def notify(message, action_label=None, on_action=None):
            notes.append((time.monotonic(), message, action_label, on_action, page.app_visible))
            return real_notify(message, action_label, on_action)

        app.notify = notify  # the snackbars (still shown: the spy calls through)
        h = Harness(app, page, session, tester, driver, device, server, notes)
        runs = app.chat_feature.runs
        real_answer = runs.answer_glossary

        def answer_glossary(cid, accepted):  # ChatRuns.answer_glossary, recorded (still answers)
            result = real_answer(cid, accepted)
            h.chat_answers.append((str(cid), bool(accepted), bool(result)))
            return result

        runs.answer_glossary = answer_glossary
        try:
            await _scenario(h)
            assert not h.problems, "\n".join(h.problems)
        except BaseException:
            print("\n--- device calls (last 60) ---")
            for row in device.calls[-60:]:
                print(row)
            print("--- foreground service log ---")
            for row in device.fgs_log[-40:]:
                print(row)
            print("--- snackbars ---")
            for row in notes[-20:]:
                print(row[1:3], "visible" if row[4] else "hidden")
            print("--- screen ---")
            for row in tester.dump(150):
                print(row)
            raise
        finally:
            server.release(abort=True)
            try:
                app.job_service.request_stop(force=True, reason="test teardown")
                await asyncio.to_thread(app.job_service.wait_idle, 30)
            except Exception:
                pass
            for thread in list(getattr(app.chat_feature.runs, "finish_threads", []) or []):
                await asyncio.to_thread(thread.join, 30)
            app.jobs.close()
            await tf._stop(app)

    with FakeLLMServer() as server:
        asyncio.run(scenario(server))
    assert _md5(src_config) == src_config_md5, "src/config.json changed"


async def _scenario(h: Harness) -> None:
    import flows
    from glossarion_mobile.services.background import PREF_NOTIFICATION_ASKED
    from glossarion_mobile.services.notifications import action_notification_id
    from glossarion_mobile.ui.chat.direct_text_rules import AUTO_ACCEPT_GLOSSARY_PREF

    app, d, device, server = h.app, h.driver, h.device, h.server
    prefs = app.prefs
    await flows.wait_home(d)
    native = app.native.native
    assert type(native).__name__ == "GlossarionNative" and not app.native.is_stub
    # a notification tap that cold-started the app is asked for once the app is ready (C7, app.py)
    assert await _until(lambda: any(c[1] == "get_launch_notification" for c in device.calls), 30)
    await flows.import_desktop_config(d)
    assert app.config_store.get("openai_base_url") == server.url
    assert app.config_store.get("auto_glossary_mode") == "balanced"
    # default OFF: nothing stored, the chat's effective setting is off
    assert prefs.get(AUTO_ACCEPT_GLOSSARY_PREF, None) is None
    assert app.chat_view.settings().auto_accept_glossary is False

    # ---- run 1: the first Start asks for the permission; the prompt never answers -------------------------------
    # U12: the first launch already showed the dialog once (no answer from this fake prompt); runs count from here
    assert await _until(lambda: len(device.requests()) >= 1, 30), "no notification dialog on the first launch"
    launch = len(device.requests())
    run1 = await (await Run(h, "devfix6-run1.epub").send()).start()
    assert len(device.requests()) == launch + 1, device.requests()
    assert await _until(lambda: device.fgs is not None, 30), "no foreground service for the job"
    assert [b["id"] for b in device.fgs["buttons"]] == ["stop", "open"], device.fgs
    assert not prefs.get(PREF_NOTIFICATION_ASKED, False), "an unanswered prompt must be asked again"
    await run1.wait_question()
    assert run1.snapshot().spec.params.get("auto_accept_glossary") is False
    # the chat with its approval card is on screen: no notification, no snackbar
    await d.wait(text="✓ Yes", timeout=30)
    await asyncio.sleep(0.5)
    assert run1.action_posts() == [] and not any(GLOSSARY_READY in n for n in run1.notes())
    assert not [c for c in device.calls if c[1] == "show_notification" and c[2].get("channel_id") == "jobs.action"]
    await d.tap(text="✓ Yes")
    assert (run1.cid, True, True) in h.chat_answers, h.chat_answers
    snap = await run1.wait_done()
    assert snap.state.name == "DONE", run1._state()
    assert await _until(lambda: device.fgs is None, 30), "the foreground service outlived its job"
    # notifications are not allowed yet: the Done notification was refused and the app knows it
    assert await _until(lambda: any(r.get("channel_id") == "jobs.done" for r in device.refused), 30)
    assert app.jobs.notifications.last_result and app.jobs.notifications.last_result["ok"] is False

    # ---- run 2: Start asks again -> granted; the gate arrives while the Library is on screen ---------------------
    server.hold()
    run2 = await (await Run(h, "devfix6-run2.epub").send()).start()
    assert len(device.requests()) == launch + 2 and device.post_notifications
    assert prefs.get(PREF_NOTIFICATION_ASKED) is True
    assert await _until(lambda: server.parked >= 1, 60), "the run never reached the model"
    await flows.open_drawer(d)
    await d.tap(key="dest-library")
    await d.wait(key="lib-search", timeout=60)
    library_route = app.shell.current_route
    server.release()
    await run2.wait_question()
    nid2 = action_notification_id(run2.jid)
    note = await _until(lambda: device.shade.get(nid2), 30)
    assert note, f"no glossary notification while the Library was on screen: {device.action_notifications()}"
    assert note["title"] == GLOSSARY_READY and note["channel_id"] == "jobs.action"
    assert note["payload"] == f"glossarion://app/chat/{run2.cid}"
    assert [(a["id"], a["title"]) for a in note["actions"]] == [("accept", "Accept"), ("open", "Review")]
    snack = await _until(lambda: [n for n in h.notes[run2.notes_mark:] if n[1] == GLOSSARY_READY], 10)
    assert snack and snack[0][2] == "Review", h.notes[run2.notes_mark:]
    await d.wait(text=GLOSSARY_READY, timeout=10)  # the real SnackBar
    assert app.shell.current_route == library_route
    assert await _until(lambda: "waiting for your glossary decision" in str((device.fgs or {}).get("text")), 15), \
        device.fgs
    # the notification's Accept answers the gate, through the chat's run controller (the card's ✓ Yes path)
    await h.press(note, "accept")
    assert await _until(lambda: run2.question() is None, 30), "Accept did not answer the glossary gate"
    assert (run2.cid, True, True) in h.chat_answers, h.chat_answers
    assert await _until(lambda: nid2 in device.cancelled, 15), "the app did not cancel the glossary notification"
    assert await _until(lambda: "Glossary accepted · translating" in run2.notes(), 10), run2.notes()
    assert await _until(lambda: app.shell.current_route.startswith(f"/chat/{run2.cid}"), 10), app.shell.current_route
    snap = await run2.wait_done()
    assert snap.state.name == "DONE", run2._state()
    assert not any(ACCEPTED_LINE in line for line in run2.log_lines())  # the user accepted, not the toggle
    assert await _until(lambda: any(a.get("channel_id") == "jobs.done" and str(a.get("title", "")).startswith("Done:")
                                    for a in device.posted[run2.device_mark:]), 30)

    # ---- run 3: the gate arrives while the app is hidden ----------------------------------------------------------
    server.hold()
    run3 = await (await Run(h, "devfix6-run3.epub").send()).start()
    assert len(device.requests()) == launch + 2, "the permission is granted: no prompt"
    assert await _until(lambda: server.parked >= 1 and device.fgs is not None, 60)
    await h.leave_app()
    assert h.page.app_visible is False and app.jobs.background.app_visible is False
    # model answers that take a while, so the job's progress (ProgressWatcher, every 2 s) moves while hidden
    server.set_delay("translation", 2.0)
    hidden_at = len(device.fgs_log)
    notes_hidden = len(h.notes)
    server.release()
    await run3.wait_question()
    nid3 = action_notification_id(run3.jid)
    note = await _until(lambda: device.shade.get(nid3), 30)
    assert note and note["title"] == GLOSSARY_READY and [a["id"] for a in note["actions"]] == ["accept", "open"], \
        device.action_notifications()
    assert await _until(lambda: any("waiting for your glossary decision" in str(r[2])
                                    for r in device.updates_since(hidden_at)), 15), device.fgs_log[hidden_at:]
    assert all(not r[3] for r in device.updates_since(hidden_at)), "updates must come while hidden"
    assert not any(n[1] == GLOSSARY_READY for n in h.notes[notes_hidden:]), "no snackbar while hidden"
    assert note["payload"] == f"glossarion://app/chat/{run3.cid}"
    ticks = app.dispatcher.ticks
    # a swipe on the job notification (Android 14+): posted again at once, same service, Stop / Open kept;
    # again after a while, and again when the user swipes it right after it came back
    for swipe, pause in ((1, 1.2), (2, 0.2), (3, 0.0)):
        before = len(device.fgs_log)
        swiped_at = time.monotonic()
        await h.swipe_job_notification()
        back = await _until(lambda: device.fgs and device.fgs["visible"], 3)
        if not back and swipe == 3:
            # reported at the end, so the rest of the scenario still runs
            h.problems.append("a quick second swipe (about 0.2 s after the job notification came back) is never "
                              "re-posted: BackgroundExecution.repost_job_notification drops a dismissal within "
                              "REPOST_INTERVAL (1 s) and the ticker / jobs view skip an unchanged text, so the "
                              "notification stays gone while the text does not change (here: the whole wait for the "
                              f"glossary decision; device log after the swipe: {device.fgs_log[before:]})")
            break
        assert back, f"swipe {swipe}: the job notification did not come back within 3 s ({device.fgs_log[before:]})"
        assert [b["id"] for b in device.fgs["buttons"]] == ["stop", "open"]
        assert "waiting for your glossary decision" in device.fgs["text"]
        updates = [r for r in device.fgs_log[before:] if r[1] in ("start", "update")]
        assert updates[0][1] == "update", updates  # update_job_service on the running service, not a new one
        if swipe < 3:
            assert updates[0][0] - swiped_at < 0.5, f"swipe {swipe}: re-posted only after {updates[0][0] - swiped_at:.2f} s"
        await asyncio.sleep(pause)
    assert app.dispatcher.ticks == ticks, "the UI pump ran while the app was hidden"
    assert run3.question() is not None and run3.snapshot().state.name == "RUNNING"  # a swipe never stops the job
    # Accept from the notification (Android brings the app forward), then the user leaves again
    await h.press(note, "accept")
    assert await _until(lambda: run3.question() is None, 30), "Accept did not answer the glossary gate"
    assert await _until(lambda: nid3 in device.cancelled, 15)
    await h.leave_app()
    await asyncio.sleep(0.3)  # the pump finishes the tick it was in, then parks
    progress_at = len(device.fgs_log)
    ticks = app.dispatcher.ticks
    snap = await run3.wait_done()
    assert snap.state.name == "DONE", run3._state()
    texts = [r[2] for r in device.updates_since(progress_at)]
    counted = sorted({t.split(": ", 1)[1].split(" chapters")[0] for t in texts if " chapters" in str(t)})
    assert len(counted) >= 2, f"the job notification did not keep counting while hidden: {texts}"
    assert all(not r[3] for r in device.fgs_log[progress_at:]) and h.page.app_visible is False
    assert app.dispatcher.ticks == ticks, "the UI pump ran while the app was hidden"
    assert await _until(lambda: device.fgs is None, 30)
    # the job is over: a late swipe (the service is gone) brings nothing back
    after = len(device.fgs_log)
    await h.session.dispatch_event(h.native._i, "foreground", {"type": "dismissed"})
    await asyncio.sleep(0.3)
    assert device.fgs is None and device.fgs_log[after:] == []
    await h.return_to_app()
    server.set_delay("translation", 0.0)

    # ---- run 4: Settings › Notifications & background › Always accept (All chats, Prefs) ------------------------------
    screen = await h.open_notifications_page()
    assert screen.permission_state == "on" and screen.notification_status.value == "Notifications: On"
    assert screen.auto_accept.value is False  # default off
    keys = h.config_keys()
    await d.tap(text=ALWAYS_ACCEPT)
    assert prefs.get(AUTO_ACCEPT_GLOSSARY_PREF) is True
    assert h.config_keys() == keys, "the switch wrote config.json"
    run4 = await (await Run(h, "devfix6-run4.epub").send()).start()
    snap = await run4.wait_done()
    assert snap.state.name == "DONE", run4._state()
    assert snap.spec.params.get("auto_accept_glossary") is True
    assert ACCEPTED_LINE in run4.log_lines(), run4.log_lines()[-20:]
    assert server.count("glossary", since=run4.mark) >= 1, "the run never generated a glossary"
    assert run4.action_posts() == [] and not any(GLOSSARY_READY in n for n in run4.notes())
    assert not await d.exists(text="✓ Yes", timeout=0.5)

    # ---- run 5: All chats off, This chat on (the chat's sidecar meta) -----------------------------------------------
    screen = await h.open_notifications_page()
    await d.tap(text=ALWAYS_ACCEPT)
    assert prefs.get(AUTO_ACCEPT_GLOSSARY_PREF) is False
    run5 = Run(h, "devfix6-run5.epub")
    await flows.go_home(d)
    await d.tap(tooltip="New chat")
    cid5 = app.chat_view.cid
    keys = h.config_keys()
    sheet = app.chat_view.open_chat_settings("chat")
    await d.tap(text=ALWAYS_ACCEPT)
    assert app.chat_feature.chats.meta(cid5).get("auto_accept_glossary") is True
    assert prefs.get(AUTO_ACCEPT_GLOSSARY_PREF) is False and h.config_keys() == keys
    sheet.close()
    await d.tap(tooltip=flows.ATTACH_TOOLTIP)
    await d.pick_file(run5.name, lambda: d.tap(key="attach-files"))
    await d.wait(contains=Path(run5.name).stem, timeout=60)
    await d.tap(key="send-idle_ready", timeout=60)
    await d.wait(text="Ready to translate", timeout=60)
    run5.cid = cid5
    await run5.start()
    snap = await run5.wait_done()
    assert snap.state.name == "DONE" and snap.spec.params.get("auto_accept_glossary") is True, run5._state()
    assert ACCEPTED_LINE in run5.log_lines() and run5.action_posts() == []

    # ---- run 6: All chats on, This chat off: asks again ---------------------------------------------------------
    screen = await h.open_notifications_page()
    await d.tap(text=ALWAYS_ACCEPT)
    assert prefs.get(AUTO_ACCEPT_GLOSSARY_PREF) is True
    run6 = Run(h, "devfix6-run6.epub")
    await flows.go_home(d)
    await d.tap(tooltip="New chat")
    cid6 = app.chat_view.cid
    sheet = app.chat_view.open_chat_settings("chat")
    assert sheet.value_of("auto_accept_glossary") is True  # inherited from All chats
    await d.tap(text=ALWAYS_ACCEPT)
    assert app.chat_feature.chats.meta(cid6).get("auto_accept_glossary") is False
    sheet.close()
    await d.tap(tooltip=flows.ATTACH_TOOLTIP)
    await d.pick_file(run6.name, lambda: d.tap(key="attach-files"))
    await d.wait(contains=Path(run6.name).stem, timeout=60)
    await d.tap(key="send-idle_ready", timeout=60)
    await d.wait(text="Ready to translate", timeout=60)
    run6.cid = cid6
    await run6.start()
    await run6.wait_question()
    assert run6.snapshot().spec.params.get("auto_accept_glossary") is False
    # chat 6 with its approval card is what the user looks at: nothing else should pop up
    await d.wait(text="✓ Yes", timeout=30)
    await asyncio.sleep(0.5)
    if run6.action_posts() or any(GLOSSARY_READY in n for n in run6.notes()):
        h.problems.append(
            f"the owning chat {cid6} with its approval card is on screen, yet the app posted the "
            f"'{GLOSSARY_READY}' notification ({len(run6.action_posts())}) and snackbar "
            f"({sum(GLOSSARY_READY in n for n in run6.notes())}): JobsFeature.question_on_screen trusts "
            f"shell.current_route = {app.shell.current_route!r}, which is stale - an earlier notification tap "
            "navigated to that chat's route and 'New chat' (ChatView._on_new_chat) switches the chat home "
            "without changing the route")
    await d.tap(text="■ No", timeout=30)
    snap = await run6.wait_done()
    assert not any(ACCEPTED_LINE in line for line in run6.log_lines())

    # ---- run 7: the owning chat is NOT on screen (opened from the drawer, then New chat) -----------------------
    screen = await h.open_notifications_page()
    await d.tap(text=ALWAYS_ACCEPT)
    assert prefs.get(AUTO_ACCEPT_GLOSSARY_PREF) is False
    run7 = await Run(h, "devfix6-run7.epub").send()
    server.hold()
    await run7.start()
    assert await _until(lambda: server.parked >= 1, 60), "the run never reached the model"
    app._open_chat(run7.cid)  # the drawer's chat row: the chat's route
    assert await _until(lambda: app.shell.current_route == f"/chat/{run7.cid}", 10), app.shell.current_route
    await d.tap(tooltip="New chat")  # the user starts something else while the glossary is generated
    assert await _until(lambda: app.chat_view.cid not in (None, run7.cid), 10)
    other = app.chat_view.cid
    server.release()
    await run7.wait_question()
    await asyncio.sleep(0.8)
    posted = bool(run7.action_posts())
    snacked = any(GLOSSARY_READY in n for n in run7.notes())
    if not (posted and snacked):
        h.problems.append(
            f"chat {other} is on screen while chat {run7.cid}'s glossary is ready, yet the app posted "
            f"{'no notification' if not posted else 'the notification'} and {'no snackbar' if not snacked else 'a snackbar'}: "
            f"JobsFeature.question_on_screen sees shell.current_route = {app.shell.current_route!r} (stale after "
            "'New chat') and treats the owning chat as on screen, so the user is never told")
    assert app.chat_feature.runs.answer_glossary(run7.cid, False)  # ■ No on chat A's card ends the run
    await run7.wait_done()

    # ---- 8: the Notifications page reads the real state on every visit -----------------------------------------
    reads = device.platform_reads
    device.post_notifications, device.blocked = False, True  # the user blocked them in the system settings
    screen = await h.open_notifications_page()
    assert device.platform_reads > reads, "the page did not read the platform state again"
    assert screen.permission_state == "blocked", screen.results
    assert screen.notification_status.value.startswith("Notifications: Blocked in system settings")
    await d.tap(key="notif-open-settings", timeout=10)
    assert await _until(lambda: device.settings_opened == 1, 10)
    await h.leave_app()  # the system settings app is in front
    device.post_notifications, device.blocked = True, False  # ... and turned them on there
    reads = device.platform_reads
    await h.return_to_app()  # back on the same page: it reads the state again (lifecycle "resume")
    assert await _until(lambda: screen.permission_state == "on", 10), screen.results
    assert device.platform_reads > reads
    assert screen.notification_status.value == "Notifications: On"
    assert not await d.exists(key="notif-open-settings", timeout=0.3)
    await flows.go_home(d)
    reads = device.platform_reads
    screen = await h.open_notifications_page()
    assert device.platform_reads > reads and screen.permission_state == "on"
    assert screen.notification_status.value == "Notifications: On"
    assert not await d.exists(key="notif-open-settings", timeout=0.3)

    # ---- no config.json key; Prefs and the chat sidecar hold the setting -------------------------------------------
    config_keys = h.config_keys()
    assert not [k for k in config_keys if "accept" in k.lower()], sorted(config_keys)
    state = json.loads((Path(app.paths.data) / "mobile_state.json").read_text(encoding="utf-8"))
    assert state.get(AUTO_ACCEPT_GLOSSARY_PREF) is False  # the page switch's last write (Prefs)
    histories = list(Path(app.paths.data).rglob("direct_text_chats.json"))
    sidecars = list(Path(app.paths.data).rglob("direct_text_chats.mobile.json"))
    assert histories and sidecars, (histories, sidecars)
    for history in histories:  # the desktop v2 chat file never carries the mobile-only setting
        assert "auto_accept_glossary" not in history.read_text(encoding="utf-8")
    assert any("auto_accept_glossary" in s.read_text(encoding="utf-8") for s in sidecars)
