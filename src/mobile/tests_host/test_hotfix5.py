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
