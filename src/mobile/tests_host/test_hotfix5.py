"""Hotfix 5: a tap outside a text field closes the keyboard, the composer's mini model picker (with the model's
Effort), the compact ＋ sheet and a working ＋ › Camera.

Owner (2026-10-27):

1. "Update it so tapping a blank area closes the keyboard."
2. "next to output mode. Add a mini model selector. It basically only shows models from the current model
   provider and behaves like our current claude model selector ..., where sliding up gracefully expands it,
   and a button at the bottom to redirect to the full model picker page"; "Include the thinking efforts
   associated with that model. Synced to the model picker settings menu. With drop down going from off to
   whatever that model supports. Fallback to openai compatible effort dropdown for unknown thinking routes
   and add the toggle ... 'pass thinking to unknown routes'. Keep this compact."
3. "the + button ... I only need it show the top part and not take up the entire screen ... we don't need
   the clipboard button ... So it's just attachment, LIbrary, Photos, Camera. Camera supports seems to
   require flet camera, so fix that".

Pinned here:

* ``keyboard_dismiss``: every ``ft.TextField`` gets an ``on_tap_outside`` that runs the field's own handler
  (``field_commit``'s commit) and then the extension's ``unfocus``; the composer keeps its keyboard while the
  slash-command list is open; on the real app (fake Android session) a ``tap_outside`` from the client sends
  ``unfocus`` to the extension, and the desktop stub installs nothing (Flutter's own mouse behaviour stays);
* ``thinking_effort``: per family the Off -> max list the backend accepts, the keys a pick writes, and the
  unknown-route fallback (GPT keys + ``pass_thinking_all_openai``);
* the composer's model chip and ``MiniModelSheet`` on the real app: the chat model's provider only, the
  extension's ``DraggableSheet`` sized for half the screen, an Effort pick written to the same global keys
  as the full sheet's "Thinking & effort", "All models" opening the full ModelSheet;
* the ＋ sheet: Files · Library · Photos · Camera (no Clipboard), the phone sheet opening on the tile row
  alone (Output mode / Tools / This chat under a swipe up), and Camera taking a photo through the extension's
  ``take_photo`` and attaching it like ＋ › Photos (errors in plain words, backing out attaches nothing);
* the contract between the layers, read from their sources (nothing compiles Dart here): the Dart cases
  ``unfocus`` / ``take_photo``, the ``GlossarionDraggableSheet`` widget and its property names, the
  ``image_picker`` dependency and the iOS camera usage string.

Real data stays untouched (``app_env`` under ``tmp_path``; USERPROFILE / APPDATA / LOCALAPPDATA too;
GLOSSARION_HTTP_LOG=0).

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_hotfix5.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
EXT_DIR = MOBILE_DIR / "extensions" / "flet_glossarion_native"
for entry in (str(APP_DIR), str(EXT_DIR / "src"), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")

FLUTTER_PKG = EXT_DIR / "src" / "flutter" / "flet_glossarion_native"
DART_SERVICE = FLUTTER_PKG / "lib" / "src" / "native_service.dart"
DART_EXTENSION = FLUTTER_PKG / "lib" / "src" / "extension.dart"
DART_SHEET = FLUTTER_PKG / "lib" / "src" / "draggable_sheet.dart"
PUBSPEC = FLUTTER_PKG / "pubspec.yaml"
PYPROJECT = MOBILE_DIR / "pyproject.toml"


def _text(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def _uf():
    spec = importlib.util.spec_from_file_location("_glossarion_uf_helpers_hotfix5",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


uf = _uf() if (_has("flet") and _has("msgpack")) else None
storage = uf.storage if uf is not None else None
app_env = uf.app_env if uf is not None else None


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    home = tmp_path / "winhome"
    for sub in ("", "AppData/Roaming", "AppData/Local"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    from glossarion_mobile.ui.components import keyboard_dismiss

    monkeypatch.setattr(keyboard_dismiss, "_unfocus", None)


# ---- 1. a tap outside a text field closes the keyboard ----------------------------------------------------


def test_every_text_field_drops_its_keyboard_on_a_tap_outside_after_its_own_handler():
    import flet as ft

    from glossarion_mobile.ui.components import keyboard_dismiss as kd
    from glossarion_mobile.ui.components.field_commit import commit_on_blur

    calls: list = []
    kd.install_tap_outside_dismiss(lambda: calls.append("unfocus"))
    plain = ft.TextField()
    plain.before_update()
    assert getattr(plain.on_tap_outside, kd._MARK, False)
    plain.on_tap_outside(None)
    assert calls == ["unfocus"]
    # a value field commits first (field_commit), then the keyboard goes
    field = commit_on_blur(ft.TextField(value="12"), lambda text: calls.append(f"commit {text}"))
    field.before_update()
    calls.clear()
    field.on_tap_outside(None)
    assert calls == ["commit 12", "unfocus"]
    # wrapped once, however often Flet updates it; a handler set later is wrapped again
    handler = field.on_tap_outside
    field.before_update()
    assert field.on_tap_outside is handler
    field.on_tap_outside = lambda e: calls.append("later")
    field.before_update()
    calls.clear()
    field.on_tap_outside(None)
    assert calls == ["later", "unfocus"]
    # a field that keeps its keyboard for now
    keep = {"on": True}
    kept = kd.keeps_keyboard(ft.TextField(), lambda: keep["on"])
    kept.before_update()
    calls.clear()
    kept.on_tap_outside(None)
    assert calls == []
    keep["on"] = False
    kept.on_tap_outside(None)
    assert calls == ["unfocus"]
    # installing again only swaps what unfocuses (one hook)
    hook = ft.TextField.before_update
    kd.install_tap_outside_dismiss(lambda: calls.append("other"))
    assert ft.TextField.before_update is hook


def test_the_composer_keeps_its_keyboard_while_the_slash_list_is_open():
    from glossarion_mobile.ui.chat.composer import Composer
    from glossarion_mobile.ui.components import keyboard_dismiss as kd

    calls: list = []
    kd.install_tap_outside_dismiss(lambda: calls.append("unfocus"))
    composer = Composer(on_slash=lambda text: None)
    composer.text_field.before_update()
    composer.text_field.on_tap_outside(None)
    assert calls == ["unfocus"]
    composer.handle_text("/")  # the command list opens; a tap on a command fills the field
    assert composer.slash.visible
    composer.text_field.on_tap_outside(None)
    assert calls == ["unfocus"]


def _device(conn, session, answers):
    """The fake client answers the extension's methods in ``answers`` like the device (name -> value);
    every other invoke keeps the fake's None."""
    from flet.messaging.protocol import MessageAction

    send = conn.send_message

    def send_message(message):
        body = message.body
        name = getattr(body, "name", None)
        if message.action == MessageAction.INVOKE_METHOD and name in answers:
            conn.messages.append(message)
            answer = answers[name]
            conn.loop.call_soon_threadsafe(session.handle_invoke_method_results, body.control_id, body.call_id,
                                           answer() if callable(answer) else answer, None)
            return
        send(message)

    conn.send_message = send_message


def _invoked(conn, name):
    from flet.messaging.protocol import MessageAction

    return [m.body for m in conn.messages if m.action == MessageAction.INVOKE_METHOD and m.body.name == name]


def test_phone_app_tap_outside_the_message_field_closes_the_keyboard(app_env):
    async def scenario():
        _m, conn, session, page, app = await uf._start("android")
        try:
            _device(conn, session, {"unfocus": True})
            composer = app.chat_view.composer
            field = composer.text_field
            from glossarion_mobile.ui.components import keyboard_dismiss as kd

            assert not app.native.is_stub and getattr(field.on_tap_outside, kd._MARK, False)
            await session.dispatch_event(field._i, "tap_outside", None)
            assert await uf._wait(lambda: len(_invoked(conn, "unfocus")) == 1), "the keyboard stayed up"
            composer.handle_text("/")  # the slash-command list is open: a tap on it keeps typing
            await session.dispatch_event(field._i, "tap_outside", None)
            await asyncio.sleep(0.2)
            assert len(_invoked(conn, "unfocus")) == 1
        finally:
            await uf._stop(app)

    asyncio.run(scenario())


def test_desktop_client_keeps_flutters_own_mouse_unfocus(app_env):
    async def scenario():
        import flet as ft

        original = ft.TextField.before_update
        _m, conn, session, page, app = await uf._start("windows")
        try:
            assert app.native.is_stub
            assert ft.TextField.before_update is original
            assert app.chat_view.composer.text_field.on_tap_outside is None
        finally:
            await uf._stop(app)

    asyncio.run(scenario())


# ---- 2. the model's Effort ------------------------------------------------------------------------------


DEFAULTS = {"enable_gpt_thinking": True, "gpt_effort": "medium", "enable_gemini_thinking": True,
            "thinking_level": "high", "thinking_budget": -1, "enable_anthropic_thinking": False,
            "anthropic_effort": "medium", "anthropic_thinking_budget": 10000, "anthropic_force_adaptive": False,
            "enable_deepseek_thinking": True, "deepseek_effort": "high", "pass_thinking_all_openai": False}


def _effort(model, **values):
    from glossarion_mobile.services.thinking_effort import effort_control

    config = dict(DEFAULTS, **values)
    return effort_control(model, config.get)


def _values(control):
    return [choice.value for choice in control.choices]


def test_effort_runs_from_off_to_what_each_model_supports():
    gpt = _effort("gpt-5.2")
    assert (gpt.family, gpt.unknown, gpt.value) == ("gpt", False, "medium")
    assert _values(gpt) == ["off", "low", "medium", "high", "xhigh", "max"]
    assert [c.label for c in gpt.choices][4] == "Extra high"
    assert gpt.choice("high").writes == (("enable_gpt_thinking", True), ("gpt_effort", "high"))
    assert gpt.choice("off").writes == (("enable_gpt_thinking", False),)
    assert _effort("gpt-5.2", enable_gpt_thinking=False).value == "off"
    assert _effort("gpt-5.2", gpt_effort="none").value == "off"
    # GPT-6 cannot turn reasoning off (the backend sends low for none)
    luna = _effort("authgpt/gpt-6-luna", enable_gpt_thinking=False)
    assert "off" not in _values(luna) and luna.value == "low"
    # Gemini 3+: levels, never off; Pro has no minimal
    pro = _effort("authgem/gemini-3-pro")
    assert (pro.family, _values(pro), pro.value) == ("gemini", ["low", "medium", "high"], "high")
    assert _values(_effort("gemini-3-flash")) == ["minimal", "low", "medium", "high"]
    assert _effort("gemini-3-flash", enable_gemini_thinking=False).value == "minimal"
    assert pro.choice("medium").writes == (("enable_gemini_thinking", True), ("thinking_level", "medium"))
    # Gemini 2.x: budgets; 2.5 Pro cannot turn it off (its lowest budget is 128)
    flash = _effort("gemini-2.5-flash", thinking_budget=8192)
    assert _values(flash) == ["off", "dynamic", "budget:1024", "budget:8192", "budget:24576"]
    assert flash.value == "budget:8192"
    assert flash.choice("dynamic").writes == (("enable_gemini_thinking", True), ("thinking_budget", -1))
    old_pro = _effort("gemini-2.5-pro", enable_gemini_thinking=False)
    assert "off" not in _values(old_pro) and old_pro.value == "budget:128"
    # Claude: adaptive effort (4.7+), budgets before; a stored budget of its own is listed
    opus = _effort("authcd/claude-opus-4-7", enable_anthropic_thinking=True, anthropic_effort="xhigh")
    assert (_values(opus), opus.value) == (["off", "low", "medium", "high", "xhigh", "max"], "xhigh")
    sonnet = _effort("claude-sonnet-4-5", enable_anthropic_thinking=True, anthropic_thinking_budget=12000)
    assert _values(sonnet) == ["off", "budget:4096", "budget:10000", "budget:12000", "budget:32000"]
    assert sonnet.value == "budget:12000"
    assert _values(_effort("claude-sonnet-4-5", anthropic_force_adaptive=True))[1] == "low"
    # DeepSeek / Chutes
    deepseek = _effort("deepseek/deepseek-v4", deepseek_effort="none")
    assert (_values(deepseek), deepseek.value) == (["off", "low", "high", "max"], "off")
    # OpenRouter follows the GPT controls (the backend's own family), never as an unknown route
    assert _effort("openrouter/anthropic/claude-sonnet-4.6").unknown is False
    # nothing to set: translation services, routes that cannot run here, no model
    assert _effort("deepl") is None and _effort("google-translate-free") is None and _effort("") is None


def test_unknown_routes_get_the_openai_compatible_effort_and_the_pass_toggle():
    from glossarion_mobile.services.thinking_effort import PASS_UNKNOWN_KEY, PASS_UNKNOWN_LABEL

    unknown = _effort("groq/llama-3.3-70b", pass_thinking_all_openai=True)
    assert (unknown.family, unknown.unknown, unknown.pass_unknown) == ("gpt", True, True)
    assert unknown.choice("low").writes == (("enable_gpt_thinking", True), ("gpt_effort", "low"))
    assert PASS_UNKNOWN_KEY == "pass_thinking_all_openai" and PASS_UNKNOWN_LABEL == "Pass thinking to unknown routes"
    import settings_schema_data

    assert PASS_UNKNOWN_KEY in settings_schema_data.SETTINGS  # the desktop Other Settings key


# ---- 2. the composer's model chip and the mini picker -----------------------------------------------------


def test_model_chip_shows_the_short_name_and_opens_the_picker():
    from glossarion_mobile.ui.chat.composer import Composer, model_chip_label

    opened: list = []
    composer = Composer(on_open_model=lambda: opened.append(True))
    assert not composer.model_chip.visible
    composer.set_model("authgpt/gpt-6-luna")
    assert composer.model_chip.visible and composer.model_text.value == "gpt-6-luna"
    assert composer.model_chip.tooltip == "Model: authgpt/gpt-6-luna"
    assert composer.model_text.overflow is not None and composer.model_chip.expand_loose
    composer.model_chip.on_click(None)
    assert opened == [True]
    assert model_chip_label("gpt-4o") == "gpt-4o" and model_chip_label("custom:lan/x") == "custom:lan/x"
    assert model_chip_label("openrouter/meta-llama/llama-3") == "meta-llama/llama-3"


def test_mini_sheet_keeps_to_the_chat_models_provider():
    from glossarion_mobile.ui.sheets.mini_model_sheet import MiniModelSheet, short_model_name
    from glossarion_mobile.ui.sheets.model_sheet import SheetEnv

    picked: list = []
    sheet = MiniModelSheet(current_model="authgpt/gpt-6-luna", env=SheetEnv(),
                           models=["gpt-4o", "authgpt/gpt-6-terra", "authgpt/gpt-6-luna", "deepseek/deepseek-v4"],
                           on_select=lambda field, value, chat: picked.append((field, value, chat)))
    keys = [row.key for row in sheet.row_controls]
    assert keys == ["model-mini-authgpt/gpt-6-terra", "model-mini-authgpt/gpt-6-luna"]
    luna = sheet.rows["authgpt/gpt-6-luna"]
    assert luna.title.value == "gpt-6-luna" and luna.trailing is not None  # ticked
    from glossarion_mobile.services import model_catalog as mc

    assert sheet.mini_subtitle.value == mc.provider_label("authgpt")  # the provider under the title
    assert not sheet.effort_dropdown.visible  # no settings store: no Effort
    assert short_model_name("authgpt2/gpt-6", "authgpt") == "authgpt2/gpt-6"
    sheet.rows["authgpt/gpt-6-terra"].on_click(None)
    assert picked == [("model", "authgpt/gpt-6-terra", False)] and sheet.closed


def _body(sheet):
    from glossarion_mobile.ui.components.sheet import sheet_body

    return sheet_body(sheet)


def test_phone_app_model_chip_opens_the_mini_picker_with_the_models_effort(app_env):
    async def scenario():
        _m, conn, session, page, app = await uf._start("android")
        try:
            from glossarion_mobile.ui.sheets.mini_model_sheet import MiniModelSheet
            from glossarion_mobile.ui.sheets.model_sheet import ModelSheet
            from flet_glossarion_native import DraggableSheet

            view = app.chat_view
            composer = view.composer
            model = app.state.chat_context.value.model
            assert composer.model_chip.visible and composer.model.startswith(model.split("/")[0])
            await session.dispatch_event(composer.model_chip._i, "click", None)
            assert await uf._wait(lambda: isinstance(view.model_sheet, MiniModelSheet) and view.model_sheet.dialog.open)
            mini = view.model_sheet
            body = _body(mini.dialog)
            assert isinstance(body, DraggableSheet) and mini.native
            assert 0.3 <= body.initial_size <= 0.6 and body.min_size < body.initial_size <= body.max_size <= 1.0
            assert body.header is mini.mini_header and body.footer is mini.mini_footer
            assert await uf._wait(lambda: len(mini.row_controls) > 1)
            assert all(mini._provider_of(m, []) == mini.provider for m in mini.rows)
            # the Effort writes the same global keys as the full sheet's "Thinking & effort"
            assert mini.effort_dropdown.visible and mini.effort is not None
            target = next(c.value for c in mini.effort.choices if c.value not in ("off", mini.effort.value))
            mini.effort_dropdown.value = target
            await session.dispatch_event(mini.effort_dropdown._i, "select", target)
            choice = mini.effort.choice(target)
            for key, value in choice.writes:
                assert await uf._wait(lambda k=key, v=value: app.config_store.get(k) == v), key
            assert mini.effort.value == target
            # "All models": the full sheet takes over
            await session.dispatch_event(mini.all_button._i, "click", None)
            assert await uf._wait(lambda: type(view.model_sheet) is ModelSheet and view.model_sheet.dialog.open)
            assert mini.closed and not mini.dialog.open
        finally:
            await uf._stop(app)

    asyncio.run(scenario())


def test_mini_picker_on_an_unknown_route_shows_the_pass_thinking_switch():
    from glossarion_mobile.services.thinking_effort import PASS_UNKNOWN_KEY
    from glossarion_mobile.ui.sheets.mini_model_sheet import MiniModelSheet
    from glossarion_mobile.ui.sheets.model_sheet import SheetEnv

    class Store:
        def __init__(self):
            self.data = {}

        def effective(self, key):
            return self.data.get(key, DEFAULTS.get(key))

        def get(self, key, default=None):
            return self.data.get(key, default)

    store = Store()
    written: list = []
    sheet = MiniModelSheet(current_model="groq/llama-3.3-70b", env=SheetEnv(store=store), models=["groq/llama-3.3-70b"])

    def write(values):
        written.append(values)
        store.data.update(values)

    sheet._write = write
    assert sheet.effort_dropdown.visible and sheet.pass_switch.visible and sheet.pass_switch.value is False
    assert sheet.pass_switch.label == "Pass thinking to unknown routes"
    sheet.set_pass_unknown(True)
    assert written == [{PASS_UNKNOWN_KEY: True}] and sheet.pass_switch.value is True
    assert sheet.set_effort("max") and written[-1] == {"enable_gpt_thinking": True, "gpt_effort": "max"}
    assert not sheet.set_effort("bogus")
    # a known family has no switch
    gpt = MiniModelSheet(current_model="gpt-5.2", env=SheetEnv(store=store), models=["gpt-5.2"])
    assert gpt.effort_dropdown.visible and not gpt.pass_switch.visible


# ---- 3. the compact ＋ sheet and ＋ › Camera -----------------------------------------------------------


def test_plus_sheet_is_files_library_photos_camera():
    from glossarion_mobile.ui.sheets.plus_sheet import CAMERA_OFF_REASON, PlusSheet, camera_error_text

    plus = PlusSheet(camera_reason=None)
    assert list(plus.tiles) == ["files", "library", "photos", "camera"]
    assert all(tile.on_click is not None and tile.expand for tile in plus.tiles.values())
    assert [t.content.controls[1].value for t in plus.tiles.values()] == ["Files", "Library", "Photos", "Camera"]
    off = PlusSheet()
    assert off.tiles["camera"].on_click is None and off.tiles["camera"].tooltip == CAMERA_OFF_REASON
    assert "Settings" in camera_error_text("camera_access_denied")
    assert camera_error_text("no_available_camera") == "This device has no camera to take a photo with."
    assert camera_error_text("weird", "boom") == "The camera could not open: boom"


def _jpeg(path: Path) -> Path:
    from PIL import Image

    Image.new("RGB", (8, 8), (200, 40, 40)).save(path, "JPEG")
    return path


def test_phone_app_plus_opens_on_the_tiles_and_camera_attaches_a_photo(app_env, tmp_path):
    pytest.importorskip("PIL")
    photo = _jpeg(tmp_path / "camera_shot.jpg")

    async def scenario():
        _m, conn, session, page, app = await uf._start("android")
        try:
            from flet_glossarion_native import DraggableSheet

            answers = {"take_photo": {"path": str(photo), "error": None}}
            _device(conn, session, answers)
            view = app.chat_view
            await session.dispatch_event(view.composer.plus_button._i, "click", None)
            assert await uf._wait(lambda: view.plus_sheet is not None and view.plus_sheet.dialog.open)
            plus = view.plus_sheet
            body = _body(plus.dialog)
            # opens on the tile row alone (about the top fifth of the screen); the rest under a swipe up
            assert isinstance(body, DraggableSheet) and body.header is plus.tiles_row
            assert body.initial_size <= 0.35 and body.max_size == 1.0
            assert plus.chat_tiles["chat_settings"] in body.controls and plus.tool_tiles["qa"] in body.controls
            camera = plus.tiles["camera"]
            assert camera.on_click is not None  # the phone app has a camera to open
            await session.dispatch_event(camera._i, "click", None)
            assert await uf._wait(lambda: len(_invoked(conn, "take_photo")) == 1)
            assert await uf._wait(lambda: (view.composer.attachment or {}).get("name", "").endswith(".jpg")), \
                "the photo was not attached"
            assert not plus.dialog.open
        finally:
            await uf._stop(app)

    asyncio.run(scenario())


def test_camera_errors_are_said_and_backing_out_attaches_nothing():
    from glossarion_mobile.ui.chat.chat_view import ChatView

    said: list = []
    attached: list = []

    def fake(result):
        async def take():
            return result

        async def run_io(fn, *args):
            return fn(*args)

        return types.SimpleNamespace(env=types.SimpleNamespace(take_photo=take, import_file=None, run_io=run_io),
                                     notify=said.append, attach_file=lambda p: attached.append(p) or True)

    assert asyncio.run(ChatView.capture_and_attach(fake({"path": None, "error": None}))) is None
    assert said == [] and attached == []
    asyncio.run(ChatView.capture_and_attach(fake({"path": None, "error": "camera_access_denied", "message": "x"})))
    assert said and "Settings" in said[-1] and attached == []
    assert asyncio.run(ChatView.capture_and_attach(fake({"path": "/tmp/p.jpg", "error": None}))) == "/tmp/p.jpg"
    assert attached == ["/tmp/p.jpg"]
    off = types.SimpleNamespace(env=types.SimpleNamespace(take_photo=None), notify=said.append)
    assert asyncio.run(ChatView.capture_and_attach(off)) is None and said[-1].startswith("The camera is only")


# ---- 4. logs open on their newest lines, ↓ jumps back ----------------------------------------------------


class _LogDispatcher:
    """Hands the backlog over at once (the pump's job) and runs on the 'loop thread'."""

    def __init__(self):
        self.dirty: list = []

    def subscribe_log(self, buffer, callback, *, backlog=0):
        lines = buffer.since(max(0, buffer.last_seq - backlog))[0] if hasattr(buffer, "since") else []
        callback(list(lines), 0)
        return lambda: None

    def on_loop_thread(self):
        return True

    def mark_dirty(self, control):
        self.dirty.append(control)


def _log_buffer(count):
    from glossarion_mobile.services.dispatcher import LogBuffer

    buffer = LogBuffer(5000, name="hotfix5")
    buffer.extend([f"line {i}" for i in range(count)])
    return buffer


def _scroll(console, pixels, maximum):
    console._on_scroll(types.SimpleNamespace(pixels=pixels, max_scroll_extent=maximum))


def test_a_log_opens_on_its_newest_lines_even_while_auto_scroll_is_paused(monkeypatch):
    from glossarion_mobile.ui.components import log_console as lc

    monkeypatch.setitem(lc.SESSION, "following", False)  # paused earlier in this app session
    console = lc.LogConsole(buffer=_log_buffer(30), dispatcher=_LogDispatcher(), backlog=500)
    # v9.15.7 hotfix 1: auto_scroll is always on (changing it rebuilt the scroll view at the log's top)
    assert not console.following and console.list_view.auto_scroll
    console.attach()  # the backlog arrives and shows: the client pins it after layout
    assert console.visible_text().endswith("line 29") and not console.pending
    from glossarion_mobile.services.dispatcher import LogLine

    console.on_lines([LogLine(31, 0.0, "line 30", "info")], 0)  # paused: a later line waits, the view stays
    assert [line.text for line in console.pending] == ["line 30"] and console.jump_holder.visible
    assert console.list_view.auto_scroll and console.visible_text().endswith("line 30")
    lc.SESSION["following"] = True
    following = lc.LogConsole(buffer=_log_buffer(5), dispatcher=_LogDispatcher())
    following.attach()
    assert following.following and following.list_view.auto_scroll
    assert following.list_view.auto_scroll_animation.duration == 0  # a jump, never the 1 s default animation


def test_scrolling_up_shows_the_jump_button_and_it_follows_again(monkeypatch):
    from glossarion_mobile.ui.components import log_console as lc

    monkeypatch.setitem(lc.SESSION, "following", True)
    console = lc.LogConsole(buffer=_log_buffer(80), dispatcher=_LogDispatcher(), header_follow=True)
    console.attach()
    jumps: list = []

    async def scroll_to(**kw):
        jumps.append(kw)

    monkeypatch.setattr(console, "_page_or_none", lambda: object())
    monkeypatch.setattr(lc.asyncio, "ensure_future", lambda coro: asyncio.new_event_loop().run_until_complete(coro))
    object.__setattr__(console.list_view, "scroll_to", scroll_to)
    for control in (console.jump_holder, console.follow_button, console.list_view):
        object.__setattr__(control, "update", lambda: None)
    assert console.jump_button.key == "log-jump-end" and not console.jump_holder.visible
    _scroll(console, 1000, 1000)
    assert console.at_end and not console.jump_holder.visible
    _scroll(console, 300, 1000)  # the reader scrolls up: this console stops following, ↓ shows
    assert not console.at_end and console.jump_holder.visible and not console.following
    assert lc.SESSION["following"] is True and console.list_view.auto_scroll  # not a session pause
    console.jump_button.on_click(None)  # ↓: the newest lines, following again
    assert jumps and jumps[-1]["offset"] == -1 and console.following and not console.jump_holder.visible
    # back at the end by hand after a scroll up: following again too
    _scroll(console, 300, 1000)
    assert not console.following
    _scroll(console, 990, 1000)
    assert console.following and not console.jump_holder.visible
    # a pause with the auto-scroll button stays a pause at the end
    console.follow_button.on_click(None)
    assert not console.following and lc.SESSION["following"] is False and console.list_view.auto_scroll
    _scroll(console, 500, 1000)
    _scroll(console, 1000, 1000)
    assert not console.following


# ---- 5. Share file via link defaults to Gofile; the Result card follows a compile; Share / Export ------------


class _Shares:
    """A share-link service with every provider off (``enabled``: the ones switched on)."""

    def __init__(self, enabled=()):
        self.enabled = set(enabled)

    def menu(self, path):
        states = [{"id": pid, "label": label, "enabled": pid in self.enabled, "consented": pid in self.enabled,
                   "handoff": pid == "transferit", "reason": None if pid in self.enabled else "Off · turn on in Settings"}
                  for pid, label in (("transferit", "transfer.it"), ("gofile", "Gofile"), ("send", "Send"),
                                     ("pixeldrain", "pixeldrain"))]
        return [(state, state["reason"]) for state in states]


def _share_actions(shares, calls):
    from glossarion_mobile.ui.screens import cloud_sync as u10

    actions = u10.U10Actions(shares=shares)

    async def start_provider(provider, path, identity):
        calls.append(("start", provider["id"]))

    async def provider_sheet(path, identity):
        calls.append(("sheet", None))

    actions.start_provider = start_provider
    actions.provider_sheet = provider_sheet
    return actions


def test_share_file_via_link_goes_to_gofile_while_every_service_is_off(tmp_path):
    from glossarion_mobile.ui.screens import cloud_sync as u10

    book = tmp_path / "Book.epub"
    book.write_bytes(b"epub")
    calls: list = []
    off = _share_actions(_Shares(), calls)
    # never "Turn on a service first": the tap goes on (consent sheet first: start_provider asks it)
    assert off.share_reason([{"id": "gofile", "enabled": False}]) is None
    assert off.share_reason([], has_outputs=False) == u10.NO_OUTPUT_REASON
    asyncio.run(off.share_file(str(book), "Book"))
    assert calls == [("start", u10.DEFAULT_SHARE_PROVIDER)] and u10.DEFAULT_SHARE_PROVIDER == "gofile"
    asyncio.run(off.share_link([(str(book), "epub")], "Book"))  # one compiled file: straight on
    assert calls[-1] == ("start", "gofile")
    # a service the user turned on: the services sheet, as before
    calls.clear()
    on = _share_actions(_Shares(enabled={"send"}), calls)
    asyncio.run(on.share_file(str(book), "Book"))
    assert calls == [("sheet", None)]


def test_result_cards_reread_their_files_when_a_job_ends():
    from glossarion_mobile.services.jobs import JobState
    from glossarion_mobile.ui.chat.integration import ChatFeature

    scheduled: list = []
    swept: list = []

    async def sweep(reason=""):
        swept.append(reason)

    feature = types.SimpleNamespace(_u10_cards={1: object()}, _u10_full=False, _post=lambda fn: fn(),
                                    _u10_schedule=lambda: scheduled.append(True), _active_job=lambda: object(),
                                    _spawn=lambda coro: coro.close(), sweep=sweep)
    assert ChatFeature._on_job_transition(feature, types.SimpleNamespace(state=JobState.RUNNING)) is None
    assert scheduled == []
    ChatFeature._on_job_transition(feature, types.SimpleNamespace(state=JobState.DONE))  # Compile ▾ ended
    assert scheduled == [True] and feature._u10_full is True


def test_share_export_offers_only_the_compiled_outputs(tmp_path):
    from glossarion_mobile.ui.chat.chat_ops import export_outputs, workspace_outputs

    book = tmp_path / "Novel"
    (book / "SDLXLIFF").mkdir(parents=True)
    (book / "Novel.epub").write_bytes(b"PK epub")
    for index in range(1, 4):
        (book / f"response_chapter{index:04d}.html.sdlxliff").write_text("<xliff/>", encoding="utf-8")
        (book / "SDLXLIFF" / f"response_chapter{index:04d}.html.sdlxliff").write_text("<xliff/>", encoding="utf-8")
    (book / "glossary.csv").write_text("type,raw_name,translated_name\n", encoding="utf-8")
    assert any(kind == "sdlxliff" for _p, kind in workspace_outputs(str(book)))  # what the sheet listed before
    assert [(Path(p).name, k) for p, k in export_outputs(str(book), str(tmp_path / "Novel.epub"))] == \
        [("Novel.epub", "epub")]
    # a book not compiled yet: nothing (Compile ▾ first), never its chapter SDLXLIFF
    (book / "Novel.epub").unlink()
    assert export_outputs(str(book), "Novel.epub") == []
    # an SDLXLIFF input's own results, a subtitle input's subtitles
    assert {k for _p, k in export_outputs(str(book), "Project.sdlxliff")} == {"sdlxliff"}
    subs = tmp_path / "Episode"
    subs.mkdir()
    (subs / "Episode_translated.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nHi\n", encoding="utf-8")
    assert [k for _p, k in export_outputs(str(subs), "Episode.srt")] == ["subtitle"]


def test_the_plan_cards_output_sheet_switches_modes():
    from glossarion_mobile.ui.chat.mode_options_sheet import ModeOptionsSheet
    from glossarion_mobile.ui.chat.output_modes import OUTPUT_MODES

    switched: list = []
    sheet = ModeOptionsSheet("text", on_switch=switched.append)
    assert sheet.mode_row is not None and list(sheet.mode_row.toggles) == [m.id for m in OUTPUT_MODES]
    assert sheet.scroll_body.controls[0] is sheet.mode_row  # the toggles above the options
    text_column = sheet.body_slot.content
    assert sheet.mode_row.tap("vision") == "selected"
    assert sheet.mode == "vision" and switched == ["vision"] and sheet.body_slot.content is not text_column
    assert sheet.mode_row.tap("vision") == "options"  # the active one: nothing switches
    assert switched == ["vision"]
    # the composer's "Options for <Mode>…" sheet keeps its single mode (no toggles there)
    assert ModeOptionsSheet("image").mode_row is None


def test_the_plan_card_output_chip_opens_the_switching_sheet():
    import inspect

    from glossarion_mobile.ui.chat.chat_view import ChatView

    source = inspect.getsource(ChatView._plan_controls)
    assert 'on_click=lambda e: self.open_plan_mode(plan), key="plan-output"' in source
    opened: list = []
    plans: list = []
    view = types.SimpleNamespace(cid="7", _update_plan=lambda cid, created, **kw: plans.append((cid, created, kw)),
                                 render_transcript=lambda: opened.append("render"),
                                 open_mode_options=lambda mode, on_switch=None: opened.append(mode) or on_switch)
    on_switch = ChatView.open_plan_mode(view, {"created": 42, "output_mode": "text"})
    assert opened == ["text"]
    on_switch("image")
    assert plans == [("7", 42, {"output_mode": "image"})] and opened[-1] == "render"


# ---- 6. the glossary sheets; the profile / language previews ------------------------------------------------


class _Page:
    width, height, web = 412.0, 860.0, False
    platform = types.SimpleNamespace(value="windows")

    def __init__(self):
        self.shown: list = []

    def show_dialog(self, dialog):
        self.shown.append(dialog)
        dialog.open = True


def _glossary_ctx(page):
    from glossarion_mobile.services.glossary import GlossaryService
    from glossarion_mobile.ui.glossary.common import GlossaryContext
    from glossarion_mobile.state.config_store import MobileConfigStore

    return GlossaryContext(service=GlossaryService(config=MobileConfigStore(None)), page=page, platform="android",
                           notify=lambda *a, **k: None, navigate=lambda *a, **k: None)


def test_the_glossary_mode_row_stands_out_and_the_mode_list_is_compact():
    import flet as ft

    from glossarion_mobile.ui.glossary.sheets import ModeSheet, PlanGlossarySheet

    page = _Page()
    ctx = _glossary_ctx(page)
    noop = lambda: None
    plan = PlanGlossarySheet(ctx, on_load_file=noop, on_clear=noop, on_review=noop, on_mode=noop,
                             effective="Glossary: No Glossary (auto)")
    tile = plan.mode_tile
    assert tile.bgcolor == ft.Colors.SECONDARY_CONTAINER and tile.title.weight == ft.FontWeight.W_600
    modes = ModeSheet(ctx).show(page)
    assert len(modes.group.content.controls) == 8
    assert modes.body.scroll is None  # only as tall as its eight modes: never the whole screen
    small = _Page()
    small.height = 420.0
    assert ModeSheet(_glossary_ctx(small)).show(small).body.scroll is not None  # a small phone still scrolls


def test_profile_and_language_chips_open_a_short_preview_with_custom_options_at_the_end():
    from glossarion_mobile.ui.sheets.mini_model_sheet import MiniModelSheet
    from glossarion_mobile.ui.sheets.model_sheet import SheetEnv

    picked: list = []
    languages = MiniModelSheet(tab="language", current_language="English", languages=("English", "Japanese", "Korean"),
                               env=SheetEnv(), on_select=lambda f, v, chat: picked.append((f, v, chat)))
    assert languages.mini_title.value == "Target language" and languages.mini_subtitle.value == "Now: English"
    assert languages.chat_switch.visible and languages.chat_switch in languages.mini_header.content.controls
    keys = [getattr(row, "key", None) for row in languages.row_controls]
    assert keys[-1] == "language-custom-row" and "language-add" not in keys  # custom at the end of the list
    assert {"language-Japanese", "language-Korean"} <= set(keys)
    assert not languages.effort_dropdown.visible
    languages.custom_language.value = "Klingon"
    languages.chat_switch.value = True
    assert languages.use_custom_language() == "Klingon"
    assert picked == [("language", "Klingon", True)] and languages.closed
    profiles = MiniModelSheet(tab="profile", current_profile="Universal", profiles=["Universal", "Korean_BeforeTranslation"],
                              env=SheetEnv())
    keys = [getattr(row, "key", None) for row in profiles.row_controls]
    assert profiles.mini_title.value == "Prompt profile" and "profile-Universal" in keys
    assert keys[-1] == "profile-extras"  # role, prefill and Edit / New / Manage at the end
    page = _Page()
    initial, maximum = profiles.sheet_sizes(page)
    assert 0.3 <= initial <= 0.6 and maximum <= 1.0


def test_apply_to_this_chat_only_keeps_a_profile_or_language_in_the_chat():
    from glossarion_mobile.ui.chat.chat_view import ChatView

    overrides: list = []
    writes: list = []
    env = types.SimpleNamespace(chats=types.SimpleNamespace(set_override=lambda cid, k, v: overrides.append((cid, k, v))),
                                store=None, prefs=None)
    view = types.SimpleNamespace(bound=True, cid="9", env=env, apply_settings_changed=lambda: writes.append("applied"))
    ChatView._on_model_sheet_select(view, "language", "Japanese", True)
    ChatView._on_model_sheet_select(view, "profile", "Universal", True)
    assert overrides == [("9", "target_language", "Japanese"), ("9", "profile", "Universal")]
    assert writes == ["applied", "applied"]


def test_the_plan_card_profile_and_language_chips_open_the_preview():
    import inspect

    from glossarion_mobile.ui.chat.chat_view import ChatView

    source = inspect.getsource(ChatView._setting_chips)
    assert 'self.open_mini_model_sheet("profile")' in source and 'self.open_mini_model_sheet("language")' in source


# ---- 7. the compiled EPUB the cards could not find; the background time limit --------------------------------


def test_a_moved_book_takes_its_cards_along_whatever_spelling_their_paths_use(tmp_path, monkeypatch):
    """The emulator repro: Android keeps app files under /data/user/0/<pkg>, a link to /data/data/<pkg>. The move
    resolved its source (realpath) and compared the cards' /data/user/0 paths as text: no card followed the book, so
    it showed no EPUB, "Compile an EPUB or PDF first" and "not in the Library yet"."""
    import os

    import direct_text_store

    real_root = tmp_path / "data" / "data" / "pkg"
    alias_root = tmp_path / "data" / "user" / "0" / "pkg"
    source = real_root / "Direct Text" / "Chat" / "Attachments" / "novel"
    source.mkdir(parents=True)
    target = tmp_path / "Output" / "novel"
    original = os.path.realpath

    def realpath(path, *args, **kwargs):  # the /data/user/0 -> /data/data link
        text = os.path.abspath(str(path))
        if text.startswith(str(alias_root)):
            text = str(real_root) + text[len(str(alias_root)):]
        return original(text, *args, **kwargs)

    monkeypatch.setattr(os.path, "realpath", realpath)
    store = direct_text_store.ChatStore(tmp_path / "history.json", output_root=tmp_path / "Output")
    aliased = alias_root / "Direct Text" / "Chat" / "Attachments" / "novel"
    session = {"messages": [("assistant", "Done", "", "", str(aliased), "", {}),
                            ("assistant", "Other", "", "", str(tmp_path / "elsewhere"), "", {}),
                            ("assistant", "Same spelling", "", "", str(source / "part"), "", {})]}
    store._relocate_session_attachment_paths(session, str(source), str(target))
    folders = [message[4] for message in session["messages"]]
    assert folders == [str(target), str(tmp_path / "elsewhere"), str(target / "part")]


def test_cards_an_earlier_build_did_not_move_still_find_their_book(tmp_path):
    from glossarion_mobile.ui.chat.chat_ops import moved_workspace

    gone = tmp_path / "Direct Text" / "Chat" / "Attachments" / "novel"
    book = tmp_path / "Output" / "novel"
    book.mkdir(parents=True)
    (book / "translation_progress.json").write_text("{}", encoding="utf-8")
    messages = [("user_file", "novel.epub", "x", 0, "", "user"), ("assistant", "Done", "", "", str(gone), "", {})]
    assert moved_workspace(messages, [1]) == ""
    assert moved_workspace(messages, [1], [str(tmp_path / "Output")]) == str(book)
    assert moved_workspace(messages, [1], [str(tmp_path / "Elsewhere")]) == ""


def _jobs_helpers():
    spec = importlib.util.spec_from_file_location("_glossarion_jobs_helpers_hotfix5",
                                                  Path(__file__).with_name("test_jobs.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_background_time_limit_lets_a_finishing_epub_build_go_on():
    from glossarion_mobile.services.background import job_finishing
    from glossarion_mobile.services.jobs import JobSpec, JobState, Progress

    tj = _jobs_helpers()
    background, native, _clock, jobs = tj._background("android")

    async def scenario():
        jobs.snap = tj._snap(state=JobState.RUNNING, progress=Progress(total=107, completed=107))  # the EPUB build
        await background.on_foreground_event({"type": "timeout", "is_timeout": True})
        assert jobs.stops == [], "the EPUB build was stopped"
        note = [c for c in native.calls if c[0] == "show_notification"][-1]
        assert note[1][1].startswith("Finishing")
        jobs.snap = tj._snap(state=JobState.RUNNING, progress=Progress(total=107, completed=40))  # still translating
        await background.on_foreground_event({"type": "timeout", "is_timeout": True})
        assert len(jobs.stops) == 1 and jobs.stops[0]["escalate"] is False
        assert [c for c in native.calls if c[0] == "show_notification"][-1][1][1] == "Paused (system limit): tap to resume"

    asyncio.run(scenario())
    compile_job = types.SimpleNamespace(spec=JobSpec("compile_epub", "Book", ("a",)), progress=Progress())
    assert job_finishing(compile_job)
    assert not job_finishing(types.SimpleNamespace(spec=JobSpec("translate", "Book", ("a",)), progress=Progress()))


# ---- the layers agree (sources) -------------------------------------------------------------------------


def test_dart_answers_the_new_methods_python_calls():
    import inspect

    import flet_glossarion_native.native as ext_native
    from glossarion_mobile.services import native as native_mod

    dart = _text(DART_SERVICE)
    for method in ("unfocus", "take_photo"):
        assert f"case '{method}':" in dart, method
        assert f'_call("{method}"' in inspect.getsource(getattr(ext_native.GlossarionNative, method))
        assert f'call("{method}"' in inspect.getsource(getattr(native_mod.NativeBridge, method))
        assert callable(getattr(native_mod.NativeStub, method))
    # unfocus drops only the field focused at the last touch, still focused, and not late
    assert "addGlobalRoute(_onPointer)" in dart and "removeGlobalRoute(_onPointer)" in dart
    assert "node.hasPrimaryFocus" in dart and "node.unfocus()" in dart
    # the photo comes from the phone's camera app (image_picker), its errors as data
    assert "picker.ImageSource.camera" in dart and "'error': e.code" in dart
    assert native_mod.METHOD_TIMEOUTS["take_photo"] >= 900
    assert asyncio.run(native_mod.NativeStub().take_photo())["error"] == "unavailable"


def test_dart_draggable_sheet_reads_the_python_controls_properties():
    import dataclasses

    from flet_glossarion_native import DraggableSheet, native_widgets_available

    assert 'case "GlossarionDraggableSheet":' in _text(DART_EXTENSION)
    dart = _text(DART_SHEET)
    read = set(re.findall(r'(?:getDouble|getBool|buildWidget|children)\("([a-z_]+)"', dart))
    own = {f.name for f in dataclasses.fields(DraggableSheet)} - {f.name for f in dataclasses.fields(__import__("flet").LayoutControl)}
    assert read == own == {"controls", "header", "footer", "initial_size", "min_size", "max_size", "snap"}
    assert "shouldCloseOnMinExtent: true" in dart and "AlwaysScrollableScrollPhysics" in dart
    sheet = DraggableSheet()
    assert type(sheet).__name__ == "DraggableSheet" and sheet._c == "GlossarionDraggableSheet"
    android = types.SimpleNamespace(platform=types.SimpleNamespace(value="android"), web=False)
    assert native_widgets_available(android)
    assert not native_widgets_available(types.SimpleNamespace(platform="windows", web=False))
    assert not native_widgets_available(types.SimpleNamespace(platform="android", web=True))


def test_build_has_image_picker_and_the_ios_camera_string():
    import tomllib

    assert re.search(r"^\s+image_picker: \^1\.", _text(PUBSPEC), re.MULTILINE)
    info = tomllib.loads(_text(PYPROJECT))["tool"]["flet"]["ios"]["info"]
    assert info["NSCameraUsageDescription"] and info["NSPhotoLibraryUsageDescription"]
    permissions = tomllib.loads(_text(PYPROJECT))["tool"]["flet"]["android"]["permission"]
    assert "android.permission.CAMERA" not in permissions  # the camera app takes the photo, not Glossarion
