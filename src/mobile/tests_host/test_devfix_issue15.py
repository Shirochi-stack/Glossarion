"""Acceptance for the owner's device request #15 (2026-10-08): "we should be able to modify prompts in
the chat settings and add prompts".

On the U8 APK Chat settings › Model & prompt had only a "Prompt profile" dropdown (names only, and on a
fresh config only "Universal", #16): no way to read, edit or add a prompt without leaving the chat.
Owner decisions (2026-10-08): editing changes the SHARED profile (the desktop model); New = a copy of
the current profile; This chat leaves the global/desktop active profile alone; rename / delete stay
under Manage… (Settings › Profiles & prompts); chats whose profile was renamed follow the rename, a
deleted one falls back to the inherited profile (never a silent Universal).

The real app (``main.main``) runs on the fake Flet session as an Android phone (``test_ui_foundations``
``_start``); every step is a tap / text entry through ``host_tester.PyTester`` + ``ui_driver.UiDriver``,
the driver the device flows use; the model is the fake OpenAI server, set up through Settings › Import
from desktop. What the phone is sent is checked on the wire (the msgpack frames of the fake session), what
the model is asked on the fake server.

1. ``test_owner_issue15_edit_and_add_prompts_from_chat_settings`` (the owner's phone: a desktop config
   WITHOUT ``prompt_profiles`` / ``active_profile``):
   a. New chat A › ⋯ › Chat settings: Model & prompt shows the effective profile's prompt card
      (Universal, inherited) with Edit prompt · New profile… · Manage….
   b. Edit prompt opens the full-screen editor (``PromptEditorSheet``: placeholder chips, count, Reset to
      default for the built-in); a new text + Save closes it, Chat settings is still open and its card
      (and the phone) show the new text; config.json holds it in ``prompt_profiles`` (the shared
      profile); ``active_profile`` / ``text_extraction_method`` stay absent; the chat stays inherited.
   c. New profile… asks for a name (prefilled "Universal (copy)", "Starts as a copy of 'Universal'."), a
      taken name keeps the dialog open with the desktop message; Create makes the copy, assigns it to
      THIS chat (custom badge, header subtitle) and opens it in the editor; its own edit is saved too.
   d. The new profile is selectable: in the dropdown (away and back), the chat pickers (``env.profiles``),
      and a Send in chat A runs with its prompt; a Send in a new chat B runs the (edited) shared
      Universal: the global profile was never switched; chat B's header and Chat settings say Universal.
   e. Manage… closes Chat settings and opens Settings › Profiles & prompts, which lists the new profile
      and marks the edited built-in.
   f. config.json gained only the desktop profile keys (``prompt_profiles``, ``profile_name_autofill``,
      ``profile_mousewheel_locked``): no ``active_profile``, no new keys (UI_SPEC Appendix B).

2. ``test_owner_issue15_rename_and_delete_under_manage`` (same phone): chat A's own profile (made with
   New profile…) renamed and deleted under Chat settings › Manage…: chat A follows the rename and
   inherits after the delete; neither switches the global profile, so chat B (inheriting) never runs
   chat A's prompt.

3. ``test_owner_issue15_chat_edits_keep_the_desktop_active_profile`` (a desktop config whose active
   profile is ``Japanese_html2text`` with ``text_extraction_method`` ``enhanced``): This chat picks the
   non-active built-in ``Korean_BeautifulSoup``, edits it, puts it back with the editor's Reset to default
   (exactly the default: no "modified" mark) and copies it with New profile…; ``active_profile`` and
   ``text_extraction_method`` never change (the core alone would switch both on save / reset).

4. ``test_owner_issue15_prompts_from_chat_settings_on_a_tablet`` (1280 dp, Chat settings in the
   SidePanel): Edit prompt / New profile… work over the panel, All chats › New profile… makes the copy the
   active profile (desktop "+ New Profile"), and Manage… closes the panel and opens Settings › Profiles &
   prompts.

Real data stays out: ``app_env`` bootstraps the app into tmp storage (data, config, Library, output);
HOME / USERPROFILE / APPDATA / LOCALAPPDATA / GLOSSARION_LIBRARY_DIR / OUTPUT_DIRECTORY /
GLOSSARION_DATA_DIR point at tmp before that, GLOSSARION_HTTP_LOG=0, and src/config.json must be
byte-identical afterwards.

Run from src/mobile with the mobile venv:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_issue15.py
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
TESTS_DIR = MOBILE_DIR / "tests"
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


def _tiktoken_assets() -> bool:
    folder = APP_DIR / "assets" / "tiktoken"
    return folder.is_dir() and any(p.is_file() and not p.suffix for p in folder.iterdir())


pytestmark = [
    pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed"),
    pytest.mark.skipif(not (_has("ebooklib") and _has("openai") and _has("tiktoken") and _has("bs4")),
                       reason="backend packages missing"),
    # offline: the pipeline counts tokens with the encodings tools/prepare_assets.py ships
    pytest.mark.skipif(not _tiktoken_assets(), reason="app/assets/tiktoken is generated by tools/prepare_assets.py"),
]


def _load(name: str, filename: str):
    """A sibling test module as a helper module (one copy of the fake session and fixtures)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_TB = _load("_glossarion_devfix15_tb_helpers", "test_bootstrap.py")
storage = _TB.storage
app_env = _TB.app_env


def _foundations():
    return _load("_glossarion_devfix15_tf_helpers", "test_ui_foundations.py")


COMPOSER_HINT = "Message to translate…"
#: unique words in the prompts the test writes (no braces: ``{target_lang}`` is substituted at run time)
SHARED_MARK = "DEVFIX15-SHARED-EDIT"
CHAT_MARK = "DEVFIX15-CHAT-A-ONLY"
EDITED_UNIVERSAL = (f"You are {SHARED_MARK}, a careful translator into {{target_lang}}.\n"
                    "Keep every name exactly as the glossary writes it.")
CHAT_TEXT = f"{EDITED_UNIVERSAL}\nAlso write {CHAT_MARK} rules."
CHAT_PROFILE = "Chat A style"
RENAMED = "Chat A style v2"
TAKEN = "A profile with this name already exists. Choose another name."
#: the keys the desktop profile actions write (other_settings.save_profiles) besides active_profile /
#: text_extraction_method, which a This-chat edit must not write
PROFILE_WRITES = {"prompt_profiles", "profile_name_autofill", "profile_mousewheel_locked"}
#: test 3: the desktop's own active profile, and the non-active built-in this chat edits
DESKTOP_ACTIVE = "Japanese_html2text"
DESKTOP_EXTRACTION = "enhanced"
CHAT_BUILTIN = "Korean_BeautifulSoup"
BUILTIN_MARK = "DEVFIX15-BUILTIN-EDIT"


# ==========================================================================
# Isolation
# ==========================================================================


def _md5(path: Path):
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except OSError:
        return None


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Nothing reads or writes the developer's home, AppData, Library, output / data folders or
    src/config.json (autouse: runs before ``storage`` / the bootstrap env contract, which then points
    the app's own folders into its tmp storage)."""
    iso = tmp_path / "_iso"
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("APPDATA", "appdata"),
                      ("LOCALAPPDATA", "localappdata"), ("GLOSSARION_LIBRARY_DIR", "Library"),
                      ("OUTPUT_DIRECTORY", "Output"), ("GLOSSARION_DATA_DIR", "data")):
        folder = iso / sub
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    config = SRC_DIR / "config.json"
    before = _md5(config)
    yield iso
    assert _md5(config) == before, "src/config.json changed during the test"


# ==========================================================================
# Helpers
# ==========================================================================


class Wire:
    """Every msgpack frame the fake Flet client is sent (``_fake_session`` encodes each message with
    ``msgpack.packb`` like the socket transport): ``sent_since(mark, text)`` tells whether the phone
    was told about ``text`` after ``mark``."""

    def __init__(self, monkeypatch) -> None:
        import msgpack

        self._lock = threading.Lock()
        self.frames: list = []
        original = msgpack.packb

        def packb(obj, *args, **kwargs):
            data = original(obj, *args, **kwargs)
            with self._lock:
                self.frames.append(data)
            return data

        monkeypatch.setattr(msgpack, "packb", packb)

    def mark(self) -> int:
        with self._lock:
            return len(self.frames)

    def sent_since(self, mark: int, text: str) -> bool:
        needle = text.encode("utf-8")
        with self._lock:
            frames = list(self.frames[mark:])
        return any(needle in frame for frame in frames)


class Prompts:
    """The messages of every chat-completions request the fake server answers (its ``_reply_for``)."""

    def __init__(self, server) -> None:
        self._lock = threading.Lock()
        self.requests: list = []
        original = server._reply_for

        def reply_for(payload, record):
            messages = [m for m in (payload.get("messages") or []) if isinstance(m, dict)]
            with self._lock:
                self.requests.append((record.kind, messages))
            return original(payload, record)

        server._reply_for = reply_for

    def mark(self) -> int:
        with self._lock:
            return len(self.requests)

    def since(self, mark: int, kind: str = "translation") -> list:
        """``[full prompt text]`` of the ``kind`` requests after ``mark``."""
        from glossarion_mobile.diagnostics.fake_llm_server import _content_text

        with self._lock:
            items = list(self.requests[mark:])
        return ["\n".join(_content_text(m.get("content")) for m in messages)
                for got, messages in items if got == kind]


async def _until(predicate, timeout: float = 15.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while True:
        try:
            value = predicate()
        except Exception:
            value = None
        if value or time.monotonic() >= deadline:
            return value
        await asyncio.sleep(interval)


def _key(control):
    key = getattr(control, "key", None)
    return getattr(key, "value", key)


def _walk(control, out=None):
    from host_tester import _children

    out = [] if out is None else out
    out.append(control)
    for child in _children(control):
        _walk(child, out)
    return out


def _by_key(root, key):
    return next((c for c in _walk(root) if _key(c) == key), None)


def _texts(root) -> list:
    import flet as ft

    return [c.value for c in _walk(root) if isinstance(c, ft.Text) and isinstance(c.value, str)]


def _route(page) -> str:
    views = list(page.views or [])
    return str(views[-1].route) if views else ""


def _where(app, page) -> str:
    """The route on screen: the top View on a phone; on a tablet the main area's stack (one root View)."""
    shell = app.shell
    if getattr(shell, "tablet", False):
        return str(shell.stack[-1].route) if shell.stack else _route(page)
    return _route(page)


def _open_dialogs(page) -> list:
    return [d for d in (getattr(getattr(page, "_dialogs", None), "controls", None) or []) if getattr(d, "open", False)]


def _disk_config(app) -> dict:
    app.config_store.flush()
    path = Path(app.paths.config_file)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _profile_dropdown(sheet):
    import flet as ft

    row = _by_key(sheet.body, "setting-profile")
    assert row is not None, "Chat settings has no Prompt profile row"
    return next(c for c in _walk(row) if isinstance(c, ft.Dropdown))


async def _host_driver(tf, files: dict, *, platform: str = "android", width: int = 412):
    from host_tester import HostPicker, PyTester
    from ui_driver import UiDriver

    _m, conn, session, page, app = await tf._start(platform, width)
    picker = HostPicker(files)
    bridge = getattr(app, "files", None)
    if bridge is not None:
        bridge._get_picker = lambda: picker  # what the app's FilePicker service would answer

    async def back():
        views = list(page.views or [])
        if len(views) > 1:
            await session.dispatch_event(page._i, "view_pop", {"route": views[-1].route})

    tester = PyTester(session, page)
    driver = UiDriver(tester, picker=picker, back=back, poll_ms=100, log=lambda *_a: None)
    return app, page, session, tester, driver


async def _start_with_model(tf, server, picks: Path, *, width: int = 412, extra: dict = None):
    """The app with the fake model imported from a desktop config: no profile keys in it (the owner's
    phone), unless ``extra`` adds some (a desktop that chose a profile)."""
    import flows
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MODEL

    picks.mkdir(parents=True, exist_ok=True)
    config = flows.write_ui_config(picks / flows.CONFIG_NAME, server.url, FAKE_MODEL)
    data = json.loads(config.read_text(encoding="utf-8"))
    assert not {"prompt_profiles", "active_profile", "text_extraction_method"} & set(data)
    if extra:
        data.update(extra)
        config.write_text(json.dumps(data, indent=2), encoding="utf-8")
    app, page, session, tester, driver = await _host_driver(tf, {flows.CONFIG_NAME: config}, width=width)
    await flows.wait_home(driver)
    if not app.shell.tablet:
        await flows.import_desktop_config(driver)
        await flows.go_home(driver)
        return app, page, session, tester, driver
    # tablet: the sidebar (no drawer) has Settings in its pinned footer; the screens open in the main area
    await driver.tap(key="drawer-settings")
    await driver.wait(key="hub-settings.appearance", timeout=30)
    await driver.tap(key="hub-settings.import", timeout=30)
    await driver.pick_file(flows.CONFIG_NAME, lambda: driver.tap(key="import-pick-config"))
    await driver.wait(text=flows.CONFIG_NAME, timeout=30)
    await driver.tap(key="import-run", timeout=60)
    await driver.wait(contains="Imported ", timeout=60)
    root = page.views[0]
    for _ in range(6):  # the system back on the one root View pops the main area back to the chat
        if not app.shell.stack:
            break
        depth = len(app.shell.stack)
        await session.dispatch_event(root._i, "confirm_pop", None)
        await _until(lambda: len(app.shell.stack) < depth, 5)
    assert not app.shell.stack, [e.route for e in app.shell.stack]
    return app, page, session, tester, driver


async def _new_chat(app, driver) -> str:
    await driver.tap(tooltip="New chat")
    await driver.wait(key="transcript-empty", timeout=30)
    return app.chat_view.cid


async def _open_chat_settings(app, driver):
    """⋯ › Chat settings, as the owner opens it; returns the sheet once its prompt card is up."""
    view = app.chat_view
    previous = view.settings_sheet
    await driver.tap(tooltip="More")  # opens on the client; its items are in the tree already
    await driver.tap(key="chat-menu-chat_settings")
    sheet = await _until(lambda: view.settings_sheet if view.settings_sheet is not previous
                         and getattr(view.settings_sheet, "cid", None) == view.cid else None)
    assert sheet is not None, "⋯ › Chat settings opened nothing"
    await driver.wait(key="setting-profile-prompt", timeout=10)
    return sheet


async def _type_into(tester, field, text: str) -> None:
    """What typing into a field does on the phone: the value, then its ``change`` event."""
    await tester.enter_text(tester._register([field]), text)


async def _tap_control(tester, control) -> None:
    await tester.tap(tester._register([control]))
    await asyncio.sleep(0.15)


async def _tap_dialog_button(tester, page, label: str) -> None:
    """Tap the ``label`` button of the top-most open dialog (its title may carry the same word)."""
    import flet as ft

    dialogs = _open_dialogs(page)
    assert dialogs, f"no dialog is open for {label!r}"
    button = next((c for c in _walk(dialogs[-1]) if isinstance(c, (ft.FilledButton, ft.TextButton))
                   and getattr(c, "content", None) == label), None)
    assert button is not None, f"the open dialog has no {label!r} button: {_texts(dialogs[-1])}"
    await _tap_control(tester, button)


async def _select(session, dropdown, value: str) -> None:
    """Picking an option: Flutter sets the value, then sends ``select``."""
    dropdown.value = value
    await session.dispatch_event(dropdown._i, "select", value)
    await asyncio.sleep(0.15)


async def _open_editor(sheet, driver, previous=None):
    """Edit prompt on the card; the PromptEditorSheet once it is open."""
    await driver.tap(key="setting-profile-prompt-edit")
    editor = await _until(lambda: sheet.prompt_editor if sheet.prompt_editor is not None
                          and sheet.prompt_editor is not previous and sheet.prompt_editor.sheet.open else None)
    assert editor is not None, "Edit prompt opened no prompt editor"
    return editor


async def _save_editor(driver, editor) -> None:
    """The editor's Save (the top-most dialog's Save button); waits until it has closed."""
    assert editor.save_button in _walk(editor.header) and editor.save_button.content == "Save"
    await driver.tap(text="Save")  # the top-most dialog's match first: the editor's own button
    assert await _until(lambda: editor.closed and not editor.sheet.open, 10), \
        f"the prompt editor did not close after Save: {editor.error_text.value!r}"


async def _new_profile(sheet, driver, tester, name: str, previous_editor=None):
    """New profile… › ``name`` › Create; returns ``(dialog, field, editor)`` once the new profile is open
    in the editor."""
    await driver.tap(key="setting-profile-prompt-new")
    pair = await _until(lambda: sheet.new_profile_dialog
                        if sheet.new_profile_dialog is not None and sheet.new_profile_dialog[0].open else None)
    assert pair is not None, "New profile… opened no name dialog"
    dialog, name_field = pair
    if name is not None:
        await _type_into(tester, name_field, name)
    await driver.tap(text="Create")
    assert await _until(lambda: not dialog.open, 5), f"the name dialog stayed open: {name_field.error!r}"
    editor = await _until(lambda: sheet.prompt_editor if sheet.prompt_editor is not previous_editor
                          and sheet.prompt_editor is not None and sheet.prompt_editor.sheet.open else None)
    assert editor is not None, "the new profile did not open in the editor"
    return dialog, name_field, editor


async def _manage(app, page, driver, sheet):
    """Manage… on the card: Chat settings closes, Settings › Profiles & prompts opens."""
    from glossarion_mobile.ui.screens.profiles import ProfilesScreen

    await driver.tap(key="setting-profile-prompt-manage")
    assert await _until(lambda: sheet.dialog not in _open_dialogs(page), 5), "Manage… left Chat settings open"
    assert await _until(lambda: _where(app, page) == "/settings/profiles", 10), _where(app, page)
    screen = await _until(lambda: app.shell.top_screen if isinstance(app.shell.top_screen, ProfilesScreen)
                          else None, 10)
    assert screen is not None, f"Manage… opened {type(app.shell.top_screen).__name__}"
    return screen


def _chat_jobs(js, cid: str) -> list:
    view = js.view()
    items = [*view.history, *view.queue, *([view.active] if view.active is not None else [])]
    return [s for s in items if (getattr(s.spec, "origin", None) or {}).get("cid") == cid]


async def _send_and_wait(app, driver, cid: str, text: str, timeout: float = 120.0):
    """Send ``text`` in the open chat; the finished job of this send."""
    js = app.job_service
    before = {s.id for s in _chat_jobs(js, cid)}
    await driver.enter(text, text=COMPOSER_HINT)
    await driver.tap(key="send-idle_ready", timeout=30)
    # the first long job on Android explains battery optimisation once; "Not now" starts the job
    if await driver.exists(text="Keep translations running", timeout=3):
        await driver.tap(text="Not now")

    def finished():
        for snap in _chat_jobs(js, cid):
            if snap.id not in before and str(getattr(snap.state, "value", snap.state)) in (
                    "DONE", "FAILED", "CANCELLED", "INTERRUPTED"):
                return snap
        return None

    snap = await _until(finished, timeout, 0.2)
    assert snap is not None, f"the send in chat {cid} never finished: {[(s.id, s.state) for s in _chat_jobs(js, cid)]}"
    assert str(getattr(snap.state, "value", snap.state)) == "DONE", (snap.state, snap.error, snap.last_line)
    return snap


def _run(tf, scenario):
    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer

    with FakeLLMServer() as server:
        asyncio.run(scenario(server))


async def _finish(app, tester, server, tf):
    server.release()
    app.jobs.close()
    await tf._stop(app)


# ==========================================================================
# 1. Phone: edit, add, select, run, Manage…
# ==========================================================================


def test_owner_issue15_edit_and_add_prompts_from_chat_settings(tmp_path, app_env, monkeypatch):
    from glossarion_mobile.ui.screens.profiles import profile_id
    from glossarion_mobile.ui.screens.prompt_editor import PLACEHOLDERS, PromptEditorSheet

    wire = Wire(monkeypatch)
    tf = _foundations()

    async def scenario(server):
        prompts = Prompts(server)
        app, page, session, tester, driver = await _start_with_model(tf, server, tmp_path / "picks")
        try:
            view = app.chat_view
            store = app.config_store
            chats = app.chat_feature.chats
            imported = _disk_config(app)
            assert "prompt_profiles" not in imported and "active_profile" not in imported
            problems = []

            # ---- a. chat A › ⋯ › Chat settings: the prompt card ---------------------------------------
            cid_a = await _new_chat(app, driver)
            sheet = await _open_chat_settings(app, driver)
            assert sheet.dialog in _open_dialogs(page), "Chat settings is not open as a bottom sheet"
            assert sheet.scope == "chat" and sheet.value_of("profile") == "Universal"  # inherited
            card = _by_key(sheet.body, "setting-profile-prompt")
            listing = sheet.listing
            assert listing is not None and len(listing.names) >= 18, "the chat does not see the shared listing"
            default_universal = listing.texts["Universal"]
            card_texts = _texts(card)
            assert "System prompt" in card_texts and "Built-in" in card_texts, card_texts
            preview = _by_key(card, "setting-profile-prompt-preview")
            assert preview is not None and preview.value.splitlines()[0] == default_universal.strip().splitlines()[0]
            for suffix in ("edit", "new", "manage"):
                button = _by_key(card, f"setting-profile-prompt-{suffix}")
                assert button is not None and not button.disabled, f"{suffix} is missing or off"
            assert "Inherited from: All chats" in _texts(_by_key(sheet.body, "setting-profile"))

            # ---- b. Edit prompt: the full-screen editor, Save edits the shared profile ----------------
            editor = await _open_editor(sheet, driver)
            assert isinstance(editor, PromptEditorSheet), "Edit prompt opened no prompt editor"
            assert editor.sheet.fullscreen is True and editor.sheet in _open_dialogs(page)
            assert sheet.dialog in _open_dialogs(page)  # Chat settings stays under it
            assert editor.title == "Universal" and editor.field.value == default_universal
            assert editor.default == default_universal and not editor.reset_button.disabled  # built-in reset
            for placeholder in PLACEHOLDERS:  # the placeholder chips
                assert await driver.count(text=placeholder) >= 1, placeholder
            await _type_into(tester, editor.field, EDITED_UNIVERSAL)
            assert editor.counter.value.startswith(f"{len(EDITED_UNIVERSAL):,} chars")
            mark = wire.mark()
            await _save_editor(driver, editor)
            assert store.get("prompt_profiles")["Universal"] == EDITED_UNIVERSAL
            saved = _disk_config(app)
            assert saved["prompt_profiles"]["Universal"] == EDITED_UNIVERSAL  # persisted, the desktop key
            assert "active_profile" not in saved and "text_extraction_method" not in saved  # global untouched
            assert chats.own_overrides(cid_a).get("profile") is None  # the chat still inherits
            assert sheet.dialog in _open_dialogs(page), "Chat settings closed with the editor"
            preview = await _until(lambda: (lambda p: p if p is not None and SHARED_MARK in str(p.value) else None)(
                _by_key(sheet.body, "setting-profile-prompt-preview")), 5)
            assert preview is not None, "the prompt card still shows the old text"
            assert await _until(lambda: wire.sent_since(mark, SHARED_MARK), 5), \
                "the phone was never sent the edited prompt (the card did not update on the device)"
            assert sheet.listing.is_modified("Universal")

            # ---- c. New profile…: a copy of the current profile, assigned to this chat ------------------
            await driver.tap(key="setting-profile-prompt-new")
            pair = await _until(lambda: sheet.new_profile_dialog
                                if sheet.new_profile_dialog is not None and sheet.new_profile_dialog[0].open else None)
            assert pair is not None, "New profile… opened no name dialog"
            dialog, name_field = pair
            assert dialog in _open_dialogs(page) and name_field.value == "Universal (copy)"
            assert "Starts as a copy of 'Universal'." in _texts(dialog)
            # a taken name: the desktop message, the dialog stays
            await _type_into(tester, name_field, "Universal")
            await driver.tap(text="Create")
            assert name_field.error == TAKEN and dialog.open
            await _type_into(tester, name_field, CHAT_PROFILE)
            await driver.tap(text="Create")
            assert await _until(lambda: not dialog.open, 5), f"the name dialog stayed open: {name_field.error!r}"
            new_editor = await _until(lambda: sheet.prompt_editor if sheet.prompt_editor is not editor
                                      and sheet.prompt_editor is not None and sheet.prompt_editor.sheet.open else None)
            assert new_editor is not None, "the new profile did not open in the editor"
            assert new_editor.title == CHAT_PROFILE and new_editor.field.value == EDITED_UNIVERSAL  # the copy
            assert store.get("prompt_profiles")[CHAT_PROFILE] == EDITED_UNIVERSAL
            assert chats.own_overrides(cid_a).get("profile") == CHAT_PROFILE  # this chat's own profile
            await _type_into(tester, new_editor.field, CHAT_TEXT)
            await _save_editor(driver, new_editor)
            profiles = store.get("prompt_profiles")
            assert profiles[CHAT_PROFILE] == CHAT_TEXT and profiles["Universal"] == EDITED_UNIVERSAL
            saved = _disk_config(app)
            assert saved["prompt_profiles"][CHAT_PROFILE] == CHAT_TEXT
            assert "active_profile" not in saved and "text_extraction_method" not in saved  # still untouched
            assert sheet.dialog in _open_dialogs(page) and sheet.value_of("profile") == CHAT_PROFILE
            dropdown = _profile_dropdown(sheet)
            assert dropdown.value == CHAT_PROFILE and CHAT_PROFILE in [o.key for o in dropdown.options]
            assert "custom" in _texts(_by_key(sheet.body, "setting-profile"))
            assert CHAT_MARK in _by_key(sheet.body, "setting-profile-prompt-preview").value
            assert await _until(lambda: view.header.profile_span.text == CHAT_PROFILE, 5), \
                f"the chat header says {view.header.profile_span.text!r}"

            # ---- d. selectable: the dropdown, the chat pickers, and the runs -----------------------------
            await _select(session, _profile_dropdown(sheet), "Korean_html2text")
            assert chats.own_overrides(cid_a).get("profile") == "Korean_html2text"
            await _select(session, _profile_dropdown(sheet), CHAT_PROFILE)
            assert chats.own_overrides(cid_a).get("profile") == CHAT_PROFILE
            assert _profile_dropdown(sheet).value == CHAT_PROFILE
            assert CHAT_PROFILE in view._profiles()  # ModelSheet › Profile, /profile, Series defaults
            assert "active_profile" not in _disk_config(app)
            sheet.close()
            assert await _until(lambda: sheet.dialog not in _open_dialogs(page), 5)

            mark = prompts.mark()
            await _send_and_wait(app, driver, cid_a, "안녕하세요. 오늘은 날씨가 좋습니다.")
            sent = prompts.since(mark)
            assert sent, "chat A's send reached no translation request"
            assert all(CHAT_MARK in text and SHARED_MARK in text for text in sent), \
                f"chat A did not run with '{CHAT_PROFILE}': {[t[:200] for t in sent]}"

            cid_b = await _new_chat(app, driver)
            assert cid_b != cid_a and chats.own_overrides(cid_b).get("profile") is None
            mark = prompts.mark()
            await _send_and_wait(app, driver, cid_b, "반갑습니다. 내일 다시 만나요.")
            sent = prompts.since(mark)
            assert sent, "chat B's send reached no translation request"
            assert all(SHARED_MARK in text and CHAT_MARK not in text for text in sent), \
                f"chat B did not run the shared (edited) Universal: {[t[:200] for t in sent]}"
            sheet_b = await _open_chat_settings(app, driver)
            assert sheet_b.value_of("profile") == "Universal" and _profile_dropdown(sheet_b).value == "Universal"
            if view.header.profile_span.text != "Universal":
                problems.append(f"chat B's header subtitle names profile {view.header.profile_span.text!r}, "
                                "but chat B runs (and its Chat settings shows) 'Universal' "
                                "(ChatView.chat_context_for falls back to the previous chat's profile when "
                                "config.json has no active_profile)")

            # ---- e. Manage…: Settings › Profiles & prompts ---------------------------------------------
            screen = await _manage(app, page, driver, sheet_b)
            await driver.wait(key=f"profile-{profile_id(CHAT_PROFILE)}", timeout=10)
            universal_row = await driver.wait(key=f"profile-{profile_id('Universal')}", timeout=5)
            row = tester.control(universal_row.first)
            assert any(_tooltip == "Differs from the built-in default"
                       for _tooltip in (getattr(c, "tooltip", None) for c in _walk(row))), \
                "Settings › Profiles does not mark the edited Universal"
            assert screen.listing.texts[CHAT_PROFILE] == CHAT_TEXT
            assert view.cid == cid_b

            # ---- f. config.json: only the desktop profile keys were added --------------------------------
            final = _disk_config(app)
            added = set(final) - set(imported)
            assert added <= PROFILE_WRITES, f"new config keys: {sorted(added - PROFILE_WRITES)}"
            assert not problems, "; ".join(problems)
        except Exception:
            for row in tester.dump(300):
                print(row)
            raise
        finally:
            await _finish(app, tester, server, tf)

    _run(tf, scenario)


# ==========================================================================
# 2. Phone: rename / delete under Manage… follow the chat, never the global profile
# ==========================================================================


def test_owner_issue15_rename_and_delete_under_manage(tmp_path, app_env, monkeypatch):
    import flows

    from glossarion_mobile.ui.screens.profiles import ProfileDetailScreen, profile_id

    tf = _foundations()

    async def scenario(server):
        prompts = Prompts(server)
        app, page, session, tester, driver = await _start_with_model(tf, server, tmp_path / "picks")
        try:
            view = app.chat_view
            store = app.config_store
            chats = app.chat_feature.chats
            imported = _disk_config(app)
            problems = []

            # chat A gets its own profile through Chat settings › New profile…
            cid_a = await _new_chat(app, driver)
            sheet = await _open_chat_settings(app, driver)
            _dialog, _field, editor = await _new_profile(sheet, driver, tester, CHAT_PROFILE)
            await _type_into(tester, editor.field, CHAT_TEXT)
            await _save_editor(driver, editor)
            assert store.get("prompt_profiles")[CHAT_PROFILE] == CHAT_TEXT
            assert chats.own_overrides(cid_a).get("profile") == CHAT_PROFILE
            assert "active_profile" not in _disk_config(app)
            sheet.close()
            assert await _until(lambda: sheet.dialog not in _open_dialogs(page), 5)
            # a message in chat A (New chat on an empty chat stays on it), run with chat A's own profile
            mark = prompts.mark()
            await _send_and_wait(app, driver, cid_a, "안녕하세요. 오늘은 날씨가 좋습니다.")
            sent = prompts.since(mark)
            assert sent and all(CHAT_MARK in text for text in sent), \
                f"chat A did not run with '{CHAT_PROFILE}': {[t[:200] for t in sent]}"

            # chat B inherits All chats (Universal); its Chat settings › Manage… › chat A's profile › rename
            cid_b = await _new_chat(app, driver)
            assert cid_b != cid_a and chats.own_overrides(cid_b).get("profile") is None
            sheet_b = await _open_chat_settings(app, driver)
            await _manage(app, page, driver, sheet_b)
            await driver.tap(key=f"profile-{profile_id(CHAT_PROFILE)}", timeout=10)
            detail = await _until(lambda: app.shell.top_screen if isinstance(app.shell.top_screen, ProfileDetailScreen)
                                  and app.shell.top_screen.name == CHAT_PROFILE else None, 10)
            assert detail is not None, f"the profile row opened {type(app.shell.top_screen).__name__}"
            await _type_into(tester, detail.name_field, RENAMED)
            await driver.tap(text="Save")
            assert await _until(lambda: RENAMED in (store.get("prompt_profiles") or {})
                                and CHAT_PROFILE not in store.get("prompt_profiles"), 10), "the rename was not saved"
            assert await _until(lambda: chats.own_overrides(cid_a).get("profile") == RENAMED, 10), \
                f"chat A did not follow the rename: {chats.own_overrides(cid_a).get('profile')!r}"
            renamed = await _until(lambda: app.shell.top_screen if isinstance(app.shell.top_screen, ProfileDetailScreen)
                                   and app.shell.top_screen.name == RENAMED else None, 10)
            assert renamed is not None, "Save did not move to the renamed profile's page"
            after_rename = _disk_config(app)
            if "active_profile" in after_rename:  # chat A's profile was not the global one
                problems.append(
                    "renaming chat A's profile under Manage… (Settings › Profiles & prompts › profile page › Save) "
                    f"switched the GLOBAL/desktop active_profile to {after_rename['active_profile']!r} "
                    "(it was unset: every inheriting chat used Universal; ProfileDetailScreen.save calls "
                    "ProfileService.save without keep_active)")

            # chat B (no profile of its own) after that rename: still the shared Universal?
            await flows.go_home(driver)
            assert view.cid == cid_b
            mark = prompts.mark()
            await _send_and_wait(app, driver, cid_b, "그럼 이만 실례하겠습니다.")
            sent = prompts.since(mark)
            assert sent, "chat B's send reached no translation request"
            if any(CHAT_MARK in text for text in sent):
                problems.append("after that rename chat B (no profile of its own) RAN chat A's prompt: the chat-only "
                                "profile leaked into every inheriting chat (and the desktop)")

            # delete it (Chat settings › Manage… › its row › Delete): chat A falls back to its inherited profile
            sheet_b = await _open_chat_settings(app, driver)
            await _manage(app, page, driver, sheet_b)
            before_delete = _disk_config(app).get("active_profile")
            await driver.tap(key=f"profile-{profile_id(RENAMED)}", timeout=10)
            assert await _until(lambda: isinstance(app.shell.top_screen, ProfileDetailScreen)
                                and app.shell.top_screen.name == RENAMED, 10)
            await driver.tap(text="Delete")
            await _tap_dialog_button(tester, page, "Delete")  # "Are you sure you want to delete profile …?"
            assert await _until(lambda: RENAMED not in (store.get("prompt_profiles") or {}), 10), "the delete was not saved"
            assert await _until(lambda: chats.own_overrides(cid_a).get("profile") is None, 10), \
                f"chat A still points at the deleted profile: {chats.own_overrides(cid_a).get('profile')!r}"
            after_delete = _disk_config(app).get("active_profile")
            if before_delete is None and after_delete is not None:
                problems.append(f"deleting chat A's (non-active) profile under Manage… wrote active_profile="
                                f"{after_delete!r}")

            # config.json: only desktop profile keys; no global profile the owner never chose
            final = _disk_config(app)
            added = set(final) - set(imported)
            assert added <= PROFILE_WRITES | {"active_profile", "text_extraction_method"}, \
                f"new config keys: {sorted(added - PROFILE_WRITES)}"
            chosen = {key: final[key] for key in ("active_profile", "text_extraction_method") if key in final}
            if chosen:
                problems.append(f"at the end config.json has {chosen} although the owner never chose a global "
                                "profile" + (" (the delete fell back from the profile the rename had made active)"
                                             if before_delete is not None else ""))
            assert not problems, "; ".join(problems)
        except Exception:
            for row in tester.dump(300):
                print(row)
            raise
        finally:
            await _finish(app, tester, server, tf)

    _run(tf, scenario)


# ==========================================================================
# 3. Phone with a desktop-chosen profile: This-chat edits keep it
# ==========================================================================


def test_owner_issue15_chat_edits_keep_the_desktop_active_profile(tmp_path, app_env, monkeypatch):
    tf = _foundations()

    async def scenario(server):
        app, page, session, tester, driver = await _start_with_model(
            tf, server, tmp_path / "picks",
            extra={"active_profile": DESKTOP_ACTIVE, "text_extraction_method": DESKTOP_EXTRACTION})
        try:
            view = app.chat_view
            store = app.config_store
            chats = app.chat_feature.chats
            imported = _disk_config(app)
            assert imported.get("active_profile") == DESKTOP_ACTIVE
            assert imported.get("text_extraction_method") == DESKTOP_EXTRACTION

            def global_kept(step: str) -> None:
                disk = _disk_config(app)
                got = (disk.get("active_profile"), disk.get("text_extraction_method"))
                assert got == (DESKTOP_ACTIVE, DESKTOP_EXTRACTION), \
                    f"{step} changed the desktop's active profile / extraction method to {got}"

            cid = await _new_chat(app, driver)
            sheet = await _open_chat_settings(app, driver)
            assert sheet.value_of("profile") == DESKTOP_ACTIVE  # inherited from All chats
            assert view.header.profile_span.text == DESKTOP_ACTIVE
            listing = sheet.listing
            default = listing.defaults[CHAT_BUILTIN]
            assert listing.is_builtin(CHAT_BUILTIN) and not listing.is_modified(CHAT_BUILTIN)

            # This chat picks a non-active built-in
            await _select(session, _profile_dropdown(sheet), CHAT_BUILTIN)
            assert chats.own_overrides(cid).get("profile") == CHAT_BUILTIN
            global_kept("picking a profile for this chat")
            card_texts = _texts(_by_key(sheet.body, "setting-profile-prompt"))
            assert "Built-in" in card_texts, card_texts

            # Edit prompt › Save: the shared built-in changes, the desktop's choice does not
            editor = await _open_editor(sheet, driver)
            assert editor.title == CHAT_BUILTIN and editor.field.value == listing.texts[CHAT_BUILTIN]
            assert editor.default == default and not editor.reset_button.disabled
            edited = f"{default.strip()}\n{BUILTIN_MARK}: keep honorifics."
            await _type_into(tester, editor.field, edited)
            await _save_editor(driver, editor)
            assert store.get("prompt_profiles")[CHAT_BUILTIN] == edited
            assert sheet.listing.is_modified(CHAT_BUILTIN)
            global_kept("Edit prompt › Save from This chat")

            # Edit prompt › Reset to default › Save: exactly the default again (no "modified" mark)
            editor2 = await _open_editor(sheet, driver, previous=editor)
            assert editor2.field.value == edited
            await _tap_control(tester, editor2.reset_button)
            assert editor2.field.value == default, "Reset to default did not put the default text in the editor"
            await _save_editor(driver, editor2)
            assert not sheet.listing.is_modified(CHAT_BUILTIN), \
                "after Reset to default + Save the built-in still differs from its default"
            assert store.get("prompt_profiles")[CHAT_BUILTIN] == default
            preview = _by_key(sheet.body, "setting-profile-prompt-preview")
            assert BUILTIN_MARK not in preview.value
            global_kept("Reset to default › Save from This chat")

            # New profile…: a copy of the chat's profile, this chat's own; the desktop's choice stays
            dialog_copy = f"{CHAT_BUILTIN} (copy)"
            _dialog, name_field, new_editor = await _new_profile(sheet, driver, tester, None, previous_editor=editor2)
            assert name_field.value == dialog_copy
            assert f"Starts as a copy of '{CHAT_BUILTIN}'." in _texts(_dialog)
            assert new_editor.title == dialog_copy and new_editor.field.value.strip() == default.strip()
            await _save_editor(driver, new_editor)
            assert store.get("prompt_profiles")[dialog_copy].strip() == default.strip()
            assert chats.own_overrides(cid).get("profile") == dialog_copy
            assert await _until(lambda: view.header.profile_span.text == dialog_copy, 5), \
                f"the chat header says {view.header.profile_span.text!r}"
            global_kept("New profile… from This chat")

            # All chats still shows (and runs) the desktop's profile
            await driver.tap(text="All chats")
            assert sheet.scope == "global" and sheet.value_of("profile") == DESKTOP_ACTIVE
            assert _profile_dropdown(sheet).value == DESKTOP_ACTIVE

            final = _disk_config(app)
            added = set(final) - set(imported)
            assert added <= PROFILE_WRITES, f"new config keys: {sorted(added - PROFILE_WRITES)}"
            global_kept("the whole scenario")
        except Exception:
            for row in tester.dump(300):
                print(row)
            raise
        finally:
            await _finish(app, tester, server, tf)

    _run(tf, scenario)


# ==========================================================================
# 4. Tablet: the SidePanel
# ==========================================================================


def test_owner_issue15_prompts_from_chat_settings_on_a_tablet(tmp_path, app_env, monkeypatch):
    from glossarion_mobile.ui.components import surface
    from glossarion_mobile.ui.screens.prompt_editor import PromptEditorSheet

    tf = _foundations()

    async def scenario(server):
        app, page, session, tester, driver = await _start_with_model(tf, server, tmp_path / "picks", width=1280)
        try:
            assert surface.is_tablet(page)
            view = app.chat_view
            store = app.config_store
            chats = app.chat_feature.chats
            cid = await _new_chat(app, driver)
            sheet = await _open_chat_settings(app, driver)
            assert sheet.in_panel, "Chat settings is not in the tablet SidePanel"

            # Edit prompt over the panel
            editor = await _open_editor(sheet, driver)
            assert isinstance(editor, PromptEditorSheet) and editor.sheet in _open_dialogs(page)
            await _type_into(tester, editor.field, EDITED_UNIVERSAL)
            await _save_editor(driver, editor)
            assert store.get("prompt_profiles")["Universal"] == EDITED_UNIVERSAL
            assert sheet.in_panel and "active_profile" not in _disk_config(app)
            assert SHARED_MARK in _by_key(sheet.body, "setting-profile-prompt-preview").value

            # This chat › New profile…
            dialog, _field, new_editor = await _new_profile(sheet, driver, tester, CHAT_PROFILE, previous_editor=editor)
            assert new_editor.title == CHAT_PROFILE
            await _save_editor(driver, new_editor)  # unchanged copy: nothing more to write
            assert store.get("prompt_profiles")[CHAT_PROFILE] == EDITED_UNIVERSAL
            assert chats.own_overrides(cid).get("profile") == CHAT_PROFILE and "active_profile" not in _disk_config(app)

            # All chats › New profile…: the copy becomes the active profile (desktop "+ New Profile")
            await driver.tap(text="All chats")
            assert sheet.scope == "global" and sheet.value_of("profile") == "Universal"
            _dialog2, _field2, global_editor = await _new_profile(sheet, driver, tester, "Global style",
                                                                  previous_editor=new_editor)
            assert _dialog2 is not dialog and global_editor.title == "Global style"
            await _save_editor(driver, global_editor)
            assert store.get("active_profile") == "Global style"
            assert store.get("prompt_profiles")["Global style"] == EDITED_UNIVERSAL
            assert chats.own_overrides(cid).get("profile") == CHAT_PROFILE  # this chat keeps its own

            # Manage…: the panel closes, Settings › Profiles & prompts opens
            await driver.tap(key="setting-profile-prompt-manage")
            assert await _until(lambda: not sheet.in_panel, 5), "Manage… left Chat settings in the panel"
            assert await _until(lambda: _where(app, page) == "/settings/profiles", 10), _where(app, page)
            from glossarion_mobile.ui.screens.profiles import ProfilesScreen

            assert await _until(lambda: isinstance(app.shell.top_screen, ProfilesScreen), 10)
            assert view.cid == cid
        except Exception:
            for row in tester.dump(300):
                print(row)
            raise
        finally:
            await _finish(app, tester, server, tf)

    _run(tf, scenario)
