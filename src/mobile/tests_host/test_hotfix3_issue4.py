"""Mobile hotfix 3, owner issue 4 - acceptance test on the real app.

What the owner reported from the phone: "Why did glossary default to off after switching out of an on device
model instead of balanced glossary". Choosing a small on-device ``local/`` model sets NO GLOSSARY (owner
2026-10-19); choosing a cloud model again left the auto glossary at "Off" instead of the Balanced the Welcome
recommends (the previous / default mode).

The scenario replays it on the real objects: ``GlossarionApp`` on the fake Flet session as an Android phone
(``test_ui_foundations._start``), driven like the phone through ``tests/host_tester.PyTester`` taps (the event
Flutter sends for a tap, a typed search, a radio pick) - no setter is called behind the UI's back. The model is
picked the three ways the app offers for the global model:

* the minimal sheet: the drawer's model chip (``drawer-status``) opens ``model_sheet_min.ModelSheetMin`` (the
  U3 name ``ChatView.open_model_sheet`` builds) on the global model;
* Chat settings › All chats › Choose… (the chat settings model picker, the same sheet on the global model);
  This chat › Choose… is the control: it changes only that chat's own model, never the global glossary mode;
* the ModelSheet in field mode: the first-run Welcome's Model field (``ModelPicker``, a fresh install).

In every sheet the user types the model into "Search models" and taps its row. What the owner sees after
leaving the on-device model is read where the app shows it: the Plan / Batch glossary chip
(``ChatView._glossary_chip_text``), the Glossary mode selector (``GlossaryFeature.open_mode_sheet``: its
radio), the mode a run uses (``auto_glossary_mode_key``, the desktop ``_current_auto_glossary_mode``) and the
``AUTO_GLOSSARY_MODE`` of the environment a book job really gets (``env_preview.build_env_preview``: the
shared ``HeadlessOwner`` + ``run_env.build_translation_env``).

The rule checked (owner issue 4, mobile hotfix 3):

* fresh config with the Welcome's Balanced -> an on-device model (No glossary shown) -> a cloud model: Balanced
  and enabled again, the glossary settings exactly as before the on-device model;
* nothing saved (the Welcome skipped, the owner's state) or the Welcome's on-device first step -> Balanced and
  enabled, written like the selector writes it (the selector would change nothing more);
* a user-chosen Minimal before -> Minimal again; a mode picked while on the on-device model (Full, Off) stays;
* on-device to on-device keeps No glossary, cloud to cloud and the 9B model (it extracts its own glossary)
  never touch the mode; repeated round trips give the same result.

Real data stays untouched: the app runs on temp storage (FLET_APP_STORAGE_*, the bootstrap's HOME,
OUTPUT_DIRECTORY, GLOSSARION_LIBRARY_DIR, GLOSSARION_DATA_DIR, CONFIG_FILE, plus USERPROFILE, APPDATA,
LOCALAPPDATA under tmp; GLOSSARION_HTTP_LOG=0), the E2E network guard refuses every non-loopback connection,
and src/config.json is checked unchanged.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_hotfix3_issue4.py
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
TESTS_DIR = MOBILE_DIR / "tests"
for _entry in (str(APP_DIR), str(TESTS_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _backend_error() -> str:
    try:
        import prompt_profiles
        import run_env  # noqa: F401
        import settings_rules  # noqa: F401

        prompt_profiles.profile_state_from_config({})
    except Exception as exc:  # the flet-only venv lacks the backend dependencies
        return f"the shared backend is not importable here ({exc})"
    return ""


_BACKEND_ERROR = _backend_error()
pytestmark = [
    pytest.mark.skipif(bool(_BACKEND_ERROR), reason=_BACKEND_ERROR or "backend importable"),
    pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed"),
]


def _load(alias: str, file_name: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(file_name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_TB = _load("_glossarion_tb_helpers_hotfix3_issue4", "test_bootstrap.py")
storage = _TB.storage
app_env = _TB.app_env

#: the Light on-device preset (Qwen3.5 2B, the Welcome's ⭐ card): a small model -> NO GLOSSARY
LIGHT = "local/Qwen3.5-2B-UD-Q4_K_XL"
#: the Garbage preset (Qwen3.5 0.8B): another small on-device model
GARBAGE = "local/Qwen3.5-0.8B-UD-Q4_K_XL"
#: the Heavy preset (Qwen3.5 9B): extracts its own glossary, never changes the mode
HEAVY = "local/Qwen3.5-9B-Q4_K_M"
CLOUD = "gemini-2.5-flash"
OTHER_CLOUD = "deepseek-chat"
MODE = "auto_glossary_mode"
ENABLE = "enable_auto_glossary"
#: what the glossary-mode selector writes (``settings_rules.AUTO_GLOSSARY_MODE_KEYS``)
SELECTOR_KEYS = (MODE, ENABLE, "append_glossary", "append_glossary_auto_load", "fuzzy_auto_mapping")
CHIP_TITLES = {"balanced": "Balanced", "minimal": "Minimal", "full": "Full", "off": "Off",
               "no_glossary": "No Glossary"}
CONFIG_JSON = SRC_DIR / "config.json"


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else ""


def _inside(path, root) -> bool:
    try:
        a = os.path.normcase(os.path.abspath(str(path)))
        b = os.path.normcase(os.path.abspath(str(root)))
        return os.path.commonpath([a, b]) == b
    except ValueError:
        return False


@pytest.fixture
def iso(tmp_path, monkeypatch, request):
    """The app on temp storage, the Windows profile variables redirected too, the network refused."""
    user = tmp_path / "user"
    for key, sub in (("USERPROFILE", "."), ("APPDATA", "AppData/Roaming"), ("LOCALAPPDATA", "AppData/Local")):
        folder = (user / sub).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(key, str(folder))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    config_md5 = _md5(CONFIG_JSON)
    request.getfixturevalue("app_env")  # FLET_APP_STORAGE_* + bootstrap (HOME, OUTPUT_DIRECTORY, Library, config)
    for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "OUTPUT_DIRECTORY", "GLOSSARION_LIBRARY_DIR",
                "GLOSSARION_DATA_DIR", "CONFIG_FILE"):
        value = os.environ.get(key)
        assert value and _inside(value, tmp_path), f"{key}={value!r} is not inside the test's tmp dir"
    from glossarion_mobile.diagnostics.e2e import ProcessGuards

    guards = ProcessGuards()
    guards._install_network_guard()
    try:
        yield tmp_path
    finally:
        guards.uninstall()
        assert _md5(CONFIG_JSON) == config_md5, "src/config.json changed"


# ---- driving the phone ---------------------------------------------------------------------------------


class Phone:
    def __init__(self, app, session, page) -> None:
        from host_tester import PyTester
        from ui_driver import UiDriver

        self.app = app
        self.session = session
        self.page = page
        self.store = app.config_store
        self.view = app.chat_view
        self.tester = PyTester(session, page)

        async def back() -> None:
            views = list(page.views or [])
            if len(views) > 1:
                await session.dispatch_event(page._i, "view_pop", {"route": views[-1].route})

        self.driver = UiDriver(self.tester, back=back, poll_ms=50, log=lambda *_a: None)


def _foundations():
    return _load("_glossarion_tf_helpers_hotfix3_issue4", "test_ui_foundations.py")


async def _start(tf, *, config=None, first_run=False) -> Phone:
    if config is not None:  # what is saved on the phone before the app starts
        paths = _TB.rb.get_paths()
        (Path(paths.data) / "config.json").write_text(json.dumps(config), encoding="utf-8")
    _m, _conn, session, page, app = await tf._start("android", first_run=first_run)
    return Phone(app, session, page)


async def _stop(tf, phone: Phone) -> None:
    phone.app.jobs.close()
    await tf._stop(phone.app)


async def _until(predicate, timeout: float = 15.0, step: float = 0.03) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(step)
    return bool(predicate())


def _key(control) -> object:
    key = getattr(control, "key", None)
    return getattr(key, "value", key)


def _descendants(root):
    from host_tester import _children

    stack, seen = [root], set()
    while stack:
        control = stack.pop()
        if id(control) in seen:
            continue
        seen.add(id(control))
        yield control
        stack.extend(_children(control))


def _all_texts(root) -> list:
    from host_tester import _texts

    return [text for control in _descendants(root) for text in _texts(control)]


def _on_screen(phone: Phone, control) -> bool:
    """``control`` is in the visible tree of the top view / the open sheets (what a finger can reach)."""
    return any(c is control for c in phone.tester._walk())


def _shown(phone: Phone, dialog) -> bool:
    dialogs = getattr(getattr(phone.page, "_dialogs", None), "controls", None) or []
    return any(d is dialog for d in dialogs) and bool(getattr(dialog, "open", False))


async def _tap(phone: Phone, control) -> None:
    assert _on_screen(phone, control), f"{type(control).__name__} {_key(control)!r} is not on screen"
    await phone.tester.tap(phone.tester._register([control]))
    await asyncio.sleep(0.05)


async def _change(phone: Phone, control, value) -> None:
    """What Flutter sends when the user types into a field / picks a radio: the value patch, then ``change``."""
    assert _on_screen(phone, control), f"{type(control).__name__} is not on screen"
    phone.session.apply_patch(control._i, {"value": value})
    await phone.session.dispatch_event(control._i, "change", value)
    await asyncio.sleep(0.05)


async def _pick(phone: Phone, sheet, model: str) -> None:
    """In an open ModelSheet: type ``model`` into "Search models" and tap its row."""
    assert await _until(lambda: _shown(phone, sheet.dialog) and _on_screen(phone, sheet.search)), \
        "the ModelSheet did not open"
    await _change(phone, sheet.search, model)

    def row():
        for control in phone.tester._walk():
            key = _key(control)
            if key == f"model-search-{model}":
                return control
            if key == "model-free-text" and f"Use “{model}”" in _all_texts(control):
                return control
        return None

    # The catalog service may repaint the list at any moment (a provider poll the network guard refuses
    # ends at once): a row found just before a repaint is gone from the screen, so find it again and tap.
    for _attempt in range(5):
        assert await _until(lambda: row() is not None), f"no row for {model} under 'Search models'"
        target = row()
        if target is None or not _on_screen(phone, target):
            continue
        await _tap(phone, target)
        if await _until(lambda: not _shown(phone, sheet.dialog), 3.0):
            return
    raise AssertionError(f"the ModelSheet stayed open after tapping {model}")


async def _drawer_sheet(phone: Phone):
    """The drawer's model chip: the minimal sheet (``ModelSheetMin``) on the global model."""
    import flows
    from glossarion_mobile.ui.sheets.model_sheet_min import ModelSheetMin

    await flows.open_drawer(phone.driver)
    before = phone.view.model_sheet
    await phone.driver.tap(key="drawer-status")
    assert await _until(lambda: phone.view.model_sheet is not before), "the drawer's model chip opened no sheet"
    sheet = phone.view.model_sheet
    assert isinstance(sheet, ModelSheetMin)
    return sheet


async def _chat_settings(phone: Phone, scope: str):
    """⋯ › Chat settings, on ``scope`` ("chat": This chat; "global": All chats), the user's taps."""
    import flows

    await flows.go_home(phone.driver)
    await phone.driver.tap(key="chat-menu-chat_settings")
    assert await _until(lambda: phone.view.settings_sheet is not None and _shown(phone, phone.view.settings_sheet.dialog))
    settings = phone.view.settings_sheet
    if scope == "global":
        segment = next(s for s in settings.scope_button.segments if s.value == "global")
        await _tap(phone, segment)
        assert settings.scope == "global"
    assert settings.scope == scope
    return settings


async def _settings_sheet(phone: Phone, scope: str):
    """Chat settings › Choose… (the chat settings model picker): the minimal sheet on that scope's model."""
    from glossarion_mobile.ui.sheets.model_sheet_min import ModelSheetMin

    settings = await _chat_settings(phone, scope)
    choose = next(c for c in _descendants(settings.body)
                  if type(c).__name__ == "TextButton" and getattr(c, "content", None) == "Choose…")
    before = phone.view.model_sheet
    await _tap(phone, choose)
    assert await _until(lambda: phone.view.model_sheet is not before), "Choose… opened no sheet"
    sheet = phone.view.model_sheet
    assert isinstance(sheet, ModelSheetMin)
    return settings, sheet


async def _choose_model(phone: Phone, path: str, model: str) -> None:
    """The global model, picked the ``path`` way (the drawer chip / Chat settings › All chats › Choose…)."""
    if path == "drawer":
        sheet = await _drawer_sheet(phone)
        await _pick(phone, sheet, model)
    elif path == "chat-settings":
        settings, sheet = await _settings_sheet(phone, "global")
        await _pick(phone, sheet, model)
        assert await _until(lambda: settings.value_of("model") == model), "Chat settings › All chats shows another model"
        settings.close()
        assert await _until(lambda: not _shown(phone, settings.dialog))
    else:
        raise AssertionError(path)
    assert phone.store.get("model") == model, (path, phone.store.get("model"))


async def _mode_sheet(phone: Phone):
    sheet = phone.app.glossary_feature.open_mode_sheet()
    assert await _until(lambda: _shown(phone, sheet.dialog)), "the Glossary mode selector did not open"
    return sheet


async def _close_mode_sheet(phone: Phone, sheet) -> None:
    done = next(c for c in _descendants(sheet.dialog)
                if type(c).__name__ == "TextButton" and getattr(c, "content", None) == "Done")
    await _tap(phone, done)
    assert await _until(lambda: not _shown(phone, sheet.dialog)), "the Glossary mode selector stayed open"


async def _pick_mode(phone: Phone, mode: str) -> None:
    """The user picks ``mode`` in the Glossary mode selector (Glossary page / Plan glossary sheet)."""
    sheet = await _mode_sheet(phone)
    await _change(phone, sheet.group, mode)
    assert phone.store.get(MODE) == mode
    await _close_mode_sheet(phone, sheet)


async def _shown_mode(phone: Phone) -> str:
    """The mode the Glossary mode selector shows checked."""
    sheet = await _mode_sheet(phone)
    value = str(sheet.group.value)
    await _close_mode_sheet(phone, sheet)
    return value


def _selector_state(store) -> dict:
    return {key: store.get(key) for key in SELECTOR_KEYS if store.has(key)}


def _job_env_mode(store) -> tuple:
    """AUTO_GLOSSARY_MODE / ENABLE_AUTO_GLOSSARY of the environment a book job runs with."""
    from glossarion_mobile.ui.screens import env_preview as ep

    result = ep.build_env_preview(copy.deepcopy(store.snapshot()),
                                  input_path=ep.preview_input_path("epub", os.environ.get("GLOSSARION_DATA_DIR")))
    assert result.ok, result.error
    rows = {row.key: row.value for row in result.rows}
    return rows.get("AUTO_GLOSSARY_MODE"), rows.get("ENABLE_AUTO_GLOSSARY")


async def _sees(phone: Phone, mode: str, *, env: bool = False, saved: object = "same") -> None:
    """What the owner sees and what runs use: ``mode`` everywhere (``saved``: the saved value, None = unset)."""
    from glossarion_mobile.ui.chat.direct_text_rules import auto_glossary_mode_key

    store = phone.store
    assert store.get(MODE, None) == (mode if saved == "same" else saved), ("saved mode", store.get(MODE, None))
    assert auto_glossary_mode_key(store.get) == mode, "the mode a run extracts with"
    chip = phone.view._glossary_chip_text()
    assert chip.startswith(f"Glossary: {CHIP_TITLES[mode]} (auto)"), f"the Plan glossary chip says {chip!r}"
    assert await _shown_mode(phone) == mode, "the Glossary mode selector shows another mode"
    if env:
        assert _job_env_mode(store)[0] == mode, "AUTO_GLOSSARY_MODE of a book job"


async def _on_device(phone: Phone, model: str) -> None:
    assert phone.store.get("model") == model
    assert phone.store.get(MODE) == "no_glossary" and phone.store.get(ENABLE) is False
    await _sees(phone, "no_glossary")


async def _balanced_like_the_selector(phone: Phone) -> None:
    """Balanced and enabled, every selector key as the selector writes it (picking Balanced again changes
    nothing)."""
    import settings_rules

    store = phone.store
    assert store.get(MODE) == "balanced" and store.get(ENABLE) is True
    changed, _env = settings_rules.apply_change(copy.deepcopy(store.snapshot()), MODE, "balanced")
    assert {k: v for k, v in dict(changed).items() if k in SELECTOR_KEYS} == {}, changed
    await _sees(phone, "balanced", env=True)


def _forget_mode(store) -> None:
    """The owner's state: no glossary mode saved (the Welcome skipped, an imported config without one)."""
    for key in SELECTOR_KEYS:
        store.unset(key)


# ---- the scenarios ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["drawer", "chat-settings"])
def test_leaving_the_on_device_model_brings_the_glossary_mode_back(iso, path):
    """Fresh config with Balanced (what the Welcome's Get started writes) / nothing saved / a user-chosen
    Minimal / a mode picked while on-device, each round trip picked ``path``'s way."""
    from glossarion_mobile.ui.screens.welcome_flow import welcome_glossary_updates

    tf = _foundations()
    config = {"model": CLOUD, "glossary_mode_dialog_shown": True, **welcome_glossary_updates("balanced")}

    async def scenario():
        phone = await _start(tf, config=config)
        try:
            store = phone.store
            # 1. a fresh config with the Welcome's Balanced default
            await _sees(phone, "balanced")
            baseline = _selector_state(store)
            assert baseline == welcome_glossary_updates("balanced")
            for _round in range(2):
                await _choose_model(phone, path, LIGHT)
                await _on_device(phone, LIGHT)
                await _choose_model(phone, path, CLOUD)
                assert _selector_state(store) == baseline, "the glossary settings before the on-device model"
                assert store.get(ENABLE) is True
                await _sees(phone, "balanced", env=_round == 0)

            # 2. nothing saved before the on-device model (the owner's state): Balanced, not "Off"
            _forget_mode(store)
            await _sees(phone, "off", saved=None)  # what an unset mode shows and runs as (the bug's result)
            await _choose_model(phone, path, LIGHT)
            await _on_device(phone, LIGHT)
            await _choose_model(phone, path, GARBAGE)  # on-device to on-device: still No glossary
            await _on_device(phone, GARBAGE)
            await _choose_model(phone, path, CLOUD)
            await _balanced_like_the_selector(phone)
            await _choose_model(phone, path, OTHER_CLOUD)  # cloud to cloud: nothing changes
            await _sees(phone, "balanced")

            # 3. a user-chosen Minimal comes back as Minimal
            await _pick_mode(phone, "minimal")
            minimal = _selector_state(store)
            assert minimal[MODE] == "minimal" and minimal[ENABLE] is True
            for _round in range(2):
                await _choose_model(phone, path, LIGHT)
                await _on_device(phone, LIGHT)
                await _choose_model(phone, path, CLOUD)
                assert _selector_state(store) == minimal, "the user's Minimal did not come back"
                await _sees(phone, "minimal", env=_round == 0)
            await _choose_model(phone, path, HEAVY)  # the 9B model extracts its own glossary: no change
            assert _selector_state(store) == minimal
            await _choose_model(phone, path, CLOUD)
            assert _selector_state(store) == minimal

            # 4. a mode picked while on the on-device model stays (Full, and an explicit Off)
            for picked in ("full", "off"):
                await _choose_model(phone, path, LIGHT)
                await _on_device(phone, LIGHT)
                await _pick_mode(phone, picked)
                chosen = _selector_state(store)
                await _choose_model(phone, path, CLOUD)
                assert _selector_state(store) == chosen, f"the {picked} picked on the on-device model changed"
                await _sees(phone, picked, env=picked == "full")
        finally:
            await _stop(tf, phone)

    asyncio.run(scenario())


def test_this_chat_model_never_moves_the_global_glossary_mode(iso):
    """Chat settings › This chat › Choose… sets only that chat's model: the global glossary mode stays."""
    tf = _foundations()

    async def scenario():
        phone = await _start(tf, config={"model": CLOUD, "glossary_mode_dialog_shown": True})
        try:
            store = phone.store
            await _pick_mode(phone, "balanced")
            state = _selector_state(store)
            settings, sheet = await _settings_sheet(phone, "chat")
            await _pick(phone, sheet, LIGHT)
            assert (phone.view.env.chats.overrides(phone.view.cid) or {}).get("model") == LIGHT
            assert store.get("model") == CLOUD and _selector_state(store) == state
            settings.close()
            await _sees(phone, "balanced")
            # the drawer's global pick also moves the open chat (follow_open_chat): still no mode change
            await _choose_model(phone, "drawer", OTHER_CLOUD)
            assert (phone.view.env.chats.overrides(phone.view.cid) or {}).get("model") == OTHER_CLOUD
            assert _selector_state(store) == state
        finally:
            await _stop(tf, phone)

    asyncio.run(scenario())


@pytest.mark.parametrize("first", ["cloud", "on-device"])
def test_fresh_install_welcome_then_on_device_then_cloud(iso, first):
    """A fresh install: the Welcome's Model field (the ModelSheet in field mode) picks a cloud model (Get
    started writes Balanced) or the on-device model (its preselected NO GLOSSARY card); the drawer's model
    chip then goes on-device and back to a cloud model: Balanced and enabled."""
    from glossarion_mobile.ui.sheets.model_sheet import ModelSheet

    tf = _foundations()

    async def scenario():
        phone = await _start(tf, first_run=True)
        try:
            store = phone.store
            assert await _until(lambda: tf._routes(phone.page)[-1] == "/welcome"), "no Welcome on a fresh install"
            assert not store.has(MODE)
            screen = phone.app.chat_feature._welcome_screen
            picker = screen.model_picker
            await _tap(phone, picker.field)  # the Model field
            assert await _until(lambda: picker.sheet is not None)
            assert isinstance(picker.sheet, ModelSheet) and picker.sheet.field_mode
            await _pick(phone, picker.sheet, CLOUD if first == "cloud" else LIGHT)
            assert store.get("model") == (CLOUD if first == "cloud" else LIGHT)
            for _step in range(6):  # Next … Get started
                button = screen.next_button
                label = button.content
                await _tap(phone, button)
                if label == "Get started":
                    break
            assert await _until(lambda: tf._routes(phone.page)[-1] != "/welcome"), "the Welcome did not finish"
            if first == "cloud":
                await _sees(phone, "balanced")
                baseline = _selector_state(store)
                await _choose_model(phone, "drawer", LIGHT)
                await _on_device(phone, LIGHT)
                await _choose_model(phone, "drawer", CLOUD)
                assert _selector_state(store) == baseline and store.get(ENABLE) is True
                await _sees(phone, "balanced", env=True)
            else:
                await _on_device(phone, LIGHT)  # the Welcome's NO GLOSSARY card for the on-device model
                await _choose_model(phone, "drawer", CLOUD)
                await _balanced_like_the_selector(phone)
                await _choose_model(phone, "drawer", LIGHT)  # and once more
                await _on_device(phone, LIGHT)
                await _choose_model(phone, "drawer", CLOUD)
                await _balanced_like_the_selector(phone)
            saved = _selector_state(store)
            assert store.flush() or not store.dirty
        finally:
            await _stop(tf, phone)
        # the app opened again later (a returning user now): still the same mode, read from config.json
        phone = await _start(tf)
        try:
            assert tf._routes(phone.page)[-1] != "/welcome"
            assert phone.store is not store  # a new store, loaded from the saved config.json
            assert phone.store.get("model") == CLOUD and _selector_state(phone.store) == saved
            await _sees(phone, "balanced")
        finally:
            await _stop(tf, phone)

    asyncio.run(scenario())
