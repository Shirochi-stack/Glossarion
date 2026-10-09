"""Acceptance for the owner's device report #16 (2026-10-08, U8 APK): "why is the only prompt the universal
prompt. the rest are all missing from the mobile version".

On the phone the chat's prompt-profile pickers offered only "Universal": ``integration.profile_names``
built the built-ins on a bare SimpleNamespace, hit an AttributeError and fell back to ``["Universal"]`` on
every fresh config. Owner decisions (2026-10-08): the chat picker lists every desktop built-in (18) from the
shared ``prompt_profiles`` listing, translation profiles first and the task-specific ones under a
"Specialised" group; Settings › Profiles & prompts shows everything like the desktop; chats whose profile
was renamed follow the rename, deleted ones fall back to the inherited profile (no silent Universal).

The real app on a fake Flet session (the ``test_ui_flows`` / ``test_devfix_issue8`` pattern:
``host_tester.PyTester`` + ``ui_driver.UiDriver`` dispatch the events Flutter would send), the fake
OpenAI server (its requests are captured, so a run's prompt is checked, not just the stored setting) and
Settings › Import from desktop:

* fresh install, nothing imported: Chat settings (header ⋯ › Chat settings) lists all 18 built-ins,
  the 7 translation profiles, then a disabled "Specialised" heading, then the 11 task-specific ones, in
  This chat and All chats scope; the prompt card previews the effective profile; ``/profile`` opens the
  ModelSheet Profile tab with the same 18 rows (previews included, the Specialised heading once);
  ``/profile manga_jp`` picks Manga_JP; Settings › Profiles & prompts lists the 18 in desktop order; the
  listing itself writes nothing to config.json;
* an imported OLDER desktop config with custom profiles (no NanoBanana_Image / SDLXLIFF Editing v2 /
  Subtitle Translation yet): Settings › Profiles & prompts lists all 21 in the desktop start-up order
  (missing built-ins added, like the desktop), the chat picker and the ``/profile`` ModelSheet list the same
  21 (customs with the translation profiles, before "Specialised") and show the imported active profile;
  a chat's custom
  profile reaches the model; deleting it in Settings › Profiles & prompts moves the chat (and a series
  that used it) back to the inherited profile, with a notice, and the next run uses the inherited prompt,
  not Universal; deleting or renaming a profile that is NOT the active one leaves the global (All chats /
  desktop) active profile alone (the desktop can only delete / save the selected profile, so a row action
  there must not move everyone to the first profile); renaming a profile carries the chat along; a second
  desktop import that no longer has the chat's profile (deleted on the desktop) makes the chat inherit the
  newly imported active profile;
* a chat whose sidecar still names a deleted profile when the user presses Send (the re-check of the chats
  never ran: the app was killed first) shows "<name> (missing)" and runs the inherited profile, not the
  backend's silent first profile (Universal);
* Settings › Translation › Profile & System Prompt (the skeptic-checked diagnosis: the same symptom there,
  "Profile" shown as the text "Universal" next to a raw "Prompt profiles" JSON tile) links to Settings ›
  Profiles & prompts (all 21), its "Profile" tile never makes a name that is not a profile the active one
  (a run would then silently use Universal) and its raw JSON is no way around the profile editor (built-ins
  dropped for good).

Real data stays out: ``app_env`` bootstraps the app into tmp storage (HOME, OUTPUT_DIRECTORY,
GLOSSARION_LIBRARY_DIR, GLOSSARION_DATA_DIR, the config), USERPROFILE / APPDATA / LOCALAPPDATA point at tmp,
GLOSSARION_HTTP_LOG=0, and the repository's src/config.json is checked unchanged afterwards.

Run from src/mobile with the mobile venv:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_issue16.py
"""

from __future__ import annotations

import asyncio
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
TESTS_DIR = MOBILE_DIR / "tests"
SRC_DIR = MOBILE_DIR.parent
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
    return folder.is_dir() and any(p.is_file() and not p.suffix for p in folder.iterdir())


pytestmark = [
    pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed"),
    pytest.mark.skipif(not (_has("ebooklib") and _has("openai") and _has("tiktoken") and _has("bs4")),
                       reason="backend packages missing"),
    # offline: the pipeline counts tokens with the encodings tools/prepare_assets.py ships
    pytest.mark.skipif(not _tiktoken_assets(), reason="app/assets/tiktoken is generated by tools/prepare_assets.py"),
]

_TB_SPEC = importlib.util.spec_from_file_location("_glossarion_tb_helpers_devfix16",
                                                  Path(__file__).with_name("test_bootstrap.py"))
_TB = importlib.util.module_from_spec(_TB_SPEC)
_TB_SPEC.loader.exec_module(_TB)
storage = _TB.storage
app_env = _TB.app_env

#: The desktop's built-in prompt profiles in desktop order (owner_state.ConfigStateMixin._init_default_prompt_profiles).
BUILTINS = ["Universal", "Refinement", "Korean_BeautifulSoup", "Japanese_BeautifulSoup", "Chinese_BeautifulSoup",
            "Korean_html2text", "Japanese_html2text", "Chinese_html2text", "Manga_JP", "Manga_KR", "Manga_CN",
            "Glossary_Editor", "RPGMaker_GTool", "RPGMaker_GTool_Image", "NanoBanana_Image", "Original",
            "SDLXLIFF Editing v2", "Subtitle Translation"]
#: Owner decision: the chat picker's "Specialised" group (task-specific built-ins), after the translation ones.
SPECIALISED = ["Refinement", "Manga_JP", "Manga_KR", "Manga_CN", "Glossary_Editor", "RPGMaker_GTool",
               "RPGMaker_GTool_Image", "NanoBanana_Image", "Original", "SDLXLIFF Editing v2", "Subtitle Translation"]
TRANSLATION = [name for name in BUILTINS if name not in SPECIALISED]
#: Built-ins an older desktop config does not have yet (the desktop start-up adds them).
NEWER_BUILTINS = ("NanoBanana_Image", "SDLXLIFF Editing v2", "Subtitle Translation")

WUXIA, CASUAL, RENAME_ME, RENAMED = "Wuxia House Style", "Casual KR", "Rename Me", "Renamed KR"
#: Unique lines in the custom profiles' prompts: a run's request shows which profile it really used.
MARKERS = {WUXIA: "PROFILE-MARKER-WUXIA-7F3A", CASUAL: "PROFILE-MARKER-CASUAL-91BC",
           RENAME_ME: "PROFILE-MARKER-RENAME-2D4E"}
DESKTOP_V1 = "devfix16-desktop-config.json"
DESKTOP_V2 = "devfix16-desktop-config-v2.json"
COMPOSER_HINT = "Message to translate…"
INHERITED_NOTICE = "now use their inherited profile"
GROUP_PREFIX = "__group__:"


def _foundations():
    spec = importlib.util.spec_from_file_location("_glossarion_tf_helpers_devfix16",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _md5(path: Path) -> str:
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except OSError:
        return ""


@pytest.fixture
def isolated(app_env, tmp_path, monkeypatch):
    """``app_env`` (tmp FLET_APP_STORAGE_*, bootstrap: HOME / OUTPUT_DIRECTORY / Library / data / config in tmp)
    plus USERPROFILE / APPDATA / LOCALAPPDATA in tmp and no HTTP log. The repository's own src/config.json is
    never the app's config and is unchanged afterwards."""
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    for name in ("USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        folder = tmp_path / "_env" / name.lower()
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    repo_config = SRC_DIR / "config.json"
    before = _md5(repo_config)
    config_file = os.environ.get("CONFIG_FILE", "")
    assert config_file and Path(config_file).resolve().is_relative_to(tmp_path.resolve()), config_file
    for name in ("HOME", "OUTPUT_DIRECTORY", "GLOSSARION_LIBRARY_DIR", "GLOSSARION_DATA_DIR"):
        value = os.environ.get(name, "")
        assert value and Path(value).resolve().is_relative_to(tmp_path.resolve()), (name, value)
    picks = tmp_path / "picks"
    picks.mkdir()
    yield picks
    assert _md5(repo_config) == before, "the test changed the repository's src/config.json"


# ---- helpers ----------------------------------------------------------------------------------------------


async def _until(predicate, timeout: float = 30.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value or time.monotonic() >= deadline:
            return value
        await asyncio.sleep(interval)


async def _host_driver(tf, files: dict):
    from host_tester import HostPicker, PyTester
    from ui_driver import UiDriver

    _m, conn, session, page, app = await tf._start("android")
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
    return app, page, tester, driver


async def _tap_control(tester, control) -> None:
    """Tap one specific control (two on-screen texts can match, e.g. a dialog's title and its button)."""
    await tester.tap(tester._register([control]))


async def _select(tester, dropdown, value: str) -> None:
    """Pick a dropdown option the way the Flutter client reports it (value, then the select event)."""
    dropdown.value = value
    await tester._dispatch(dropdown, "select", value)


def _stored(app) -> dict:
    """config.json on disk (what the desktop would read after a sync); {} while nothing was ever saved."""
    store = app.config_store
    flush = getattr(store, "flush", None)
    if callable(flush):
        flush()
    path = store.path
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _profile_dropdown(tester, finder):
    row = tester.control(finder)
    dropdown = row.controls[0]
    assert type(dropdown).__name__ == "Dropdown", row
    return dropdown


def _option_rows(dropdown) -> list:
    """(key, text, disabled) of the dropdown's options, top to bottom."""
    return [(str(o.key), str(o.text), bool(getattr(o, "disabled", False))) for o in (dropdown.options or [])]


def _assert_grouped(options: list, names: list) -> None:
    """The chat picker's layout: the translation names, then ONE disabled "Specialised" heading, then the
    task-specific built-ins; each name once, in the listing's (desktop) order within its group."""
    keys = [key for key, _text, _off in options]
    assert keys.count(GROUP_PREFIX + "Specialised") == 1, keys
    head = keys.index(GROUP_PREFIX + "Specialised")
    heading = options[head]
    assert heading[1] == "Specialised" and heading[2] is True, heading  # a heading, never a value
    translation = [n for n in names if n not in SPECIALISED]
    specialised = [n for n in names if n in SPECIALISED]
    assert keys[:head] == translation, keys
    assert keys[head + 1:] == specialised, keys
    for key, text, disabled in options:
        if not key.startswith(GROUP_PREFIX):
            assert text == key and not disabled, (key, text, disabled)


def _key_of(control) -> str:
    key = getattr(control, "key", None)
    return str(getattr(key, "value", key) or "")


def _assert_sheet_grouped(rows: list, names: list) -> None:
    """The ModelSheet Profile tab: the translation names, ONE "Specialised" heading, the task-specific
    built-ins, then the extras row (prompt role, Assistant prefill, Edit / New / Manage)."""
    keys = [_key_of(r) for r in rows]
    assert keys.count("profile-group-specialised") == 1, keys
    head = keys.index("profile-group-specialised")
    translation = [n for n in names if n not in SPECIALISED]
    specialised = [n for n in names if n in SPECIALISED]
    assert keys[:head] == [f"profile-{n}" for n in translation], keys
    assert keys[head + 1:] == [f"profile-{n}" for n in specialised] + ["profile-extras"], keys


async def _open_chat_settings(app, driver, tester):
    """Header ⋯ › Chat settings; returns (sheet, profile dropdown)."""
    view = app.chat_view
    previous = view.settings_sheet
    await driver.tap(key="chat-menu-chat_settings", timeout=15)
    sheet = await _until(lambda: view.settings_sheet if view.settings_sheet is not previous
                         and view.settings_sheet.dialog.open else None, 10)
    assert sheet is not None, "Chat settings did not open"
    finder = await driver.wait(key="setting-profile", timeout=10)
    return sheet, _profile_dropdown(tester, finder)


async def _send_text(app, driver, server, captured: list, text: str) -> tuple:
    """Send ``text`` in the current chat; returns (finished job snapshot, the run's chat requests)."""
    js = app.job_service
    known = {s.id for s in js.view().history}
    mark = len(captured)
    await driver.enter(text, text=COMPOSER_HINT)
    await driver.tap(key="send-idle_ready", timeout=30)
    if await driver.exists(text="Keep translations running", timeout=2):
        await driver.tap(text="Not now")

    def finished():
        for snap in js.view().history:
            if snap.id not in known:
                return snap
        return None

    snap = await _until(finished, 120)
    assert snap is not None, "the chat run never finished"
    assert str(getattr(snap.state, "value", snap.state)) == "DONE", snap
    payloads = captured[mark:]
    assert payloads, "the run sent nothing to the model"
    return snap, payloads


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
    return str(content or "")


def _prompt_text(payloads: list) -> str:
    """Every message the run sent (system, user, assistant prefill)."""
    out = []
    for payload in payloads:
        for message in payload.get("messages") or ():
            out.append(_content_text(message.get("content")))
    return "\n".join(out)


def _signature_line(text: str, *, avoid: tuple = ()) -> str:
    """A long line of a built-in prompt without placeholders that no other given prompt contains."""
    for line in str(text).splitlines():
        line = line.strip()
        if len(line) >= 40 and "{" not in line and not any(line in other for other in avoid):
            return line
    raise AssertionError("no signature line in the prompt")


async def _run_command(driver, text: str) -> None:
    """Type a slash command and press Send: a complete command runs even while Send is blocked (fresh
    install, nobody signed in yet)."""
    await driver.enter(text, text=COMPOSER_HINT)
    index = await driver.wait_any({"key": "send-idle_ready"}, {"key": "send-blocked"}, timeout=10)
    await driver.tap(key=("send-idle_ready", "send-blocked")[index])
    await driver.pump(200)


async def _open_profiles_screen(app, driver):
    """Drawer › Settings › Profiles & prompts (the hub tile)."""
    import flows
    from glossarion_mobile.ui.screens.profiles import ProfilesScreen

    await flows.open_settings(driver)
    await driver.tap(key="hub-settings.profiles", timeout=flows.SCROLL_TIMEOUT, scroll=True)
    await driver.wait(key="profiles-prefill-link", timeout=30)
    screen = await _until(lambda: app.shell.top_screen if isinstance(app.shell.top_screen, ProfilesScreen) else None, 10)
    assert screen is not None, type(app.shell.top_screen)
    return screen


def _screen_names(screen) -> list:
    """The Profiles & prompts rows top to bottom, as profile names (rows are keyed by an opaque id)."""
    from glossarion_mobile.ui.screens.profiles import profile_id

    by_id = {f"profile-{profile_id(name)}": name for name in screen.listing.names}
    keys = [getattr(getattr(c, "key", None), "value", getattr(c, "key", None)) for c in screen.list_view.controls]
    assert keys[0] == "profiles-prefill-link", keys
    return [by_id.get(key, f"<unknown {key}>") for key in keys[1:]]


def _desktop_config(server_url: str, *, profiles: dict, active: str) -> dict:
    import flows
    from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MODEL

    config = flows.ui_config(server_url, FAKE_MODEL)
    config.update({"prompt_profiles": profiles, "active_profile": active})
    return config


# ---- fresh install ---------------------------------------------------------------------------------------


def test_owner_issue16_fresh_install_lists_every_builtin_profile(isolated):
    """Nothing imported, no profile ever touched: every chat picker and Settings › Profiles & prompts show
    the 18 desktop built-ins (the phone showed only "Universal")."""
    picks = isolated
    tf = _foundations()

    async def scenario():
        import flows
        import prompt_profiles

        app, page, tester, driver = await _host_driver(tf, {})
        try:
            await flows.wait_home(driver)
            store = app.config_store
            assert "prompt_profiles" not in store.keys() and "active_profile" not in store.keys()
            desktop = list(prompt_profiles.profile_state_from_config({}).prompt_profiles)
            assert desktop == BUILTINS  # the desktop start-up on a fresh config (the reference)
            defaults = prompt_profiles.profile_state_from_config({}).default_prompts
            view = app.chat_view
            chats = app.chat_feature.chats

            # ---- Chat settings › This chat: 7 translation profiles, "Specialised", 11 task-specific ones -----
            await driver.tap(tooltip="New chat")
            await driver.wait(key="transcript-empty", timeout=30)
            cid = view.cid
            sheet, dropdown = await _open_chat_settings(app, driver, tester)
            options = _option_rows(dropdown)
            assert [k for k, _t, _d in options if not k.startswith(GROUP_PREFIX)] != ["Universal"], (
                "owner #16: the chat picker still offers only Universal")
            _assert_grouped(options, BUILTINS)
            assert len([k for k, _t, _d in options if not k.startswith(GROUP_PREFIX)]) == 18
            for name in ("Manga_JP", "Korean_BeautifulSoup", "SDLXLIFF Editing v2", "Subtitle Translation"):
                await driver.wait(text=name, timeout=5)  # on screen in the open sheet
            # the profile the desktop start-up runs on a fresh config, shown (never written)
            assert dropdown.value == "Universal" and sheet.value_of("profile") == "Universal"
            preview = await driver.wait(key="setting-profile-prompt-preview", timeout=5)
            from glossarion_mobile.ui.screens.profiles import prompt_preview

            assert tester.control(preview).value == prompt_preview(defaults["Universal"])
            # picking a task-specific built-in for this chat: the chat's own override, config.json untouched
            await _select(tester, dropdown, "Korean_html2text")
            assert chats.own_overrides(cid).get("profile") == "Korean_html2text"
            assert "prompt_profiles" not in _stored(app) and "active_profile" not in _stored(app)
            sheet_dropdown = _profile_dropdown(tester, await driver.wait(key="setting-profile", timeout=5))
            assert sheet_dropdown.value == "Korean_html2text"
            _assert_grouped(_option_rows(sheet_dropdown), BUILTINS)  # still the full list after the rebuild
            preview = await driver.wait(key="setting-profile-prompt-preview", timeout=5)
            assert tester.control(preview).value == prompt_preview(defaults["Korean_html2text"])
            # ---- All chats: the same 18 ------------------------------------------------------------------------
            await driver.tap(text="All chats")
            assert sheet.scope == "global"
            global_dropdown = _profile_dropdown(tester, await driver.wait(key="setting-profile", timeout=5))
            _assert_grouped(_option_rows(global_dropdown), BUILTINS)
            assert global_dropdown.value == "Universal"
            sheet.close()
            assert await _until(lambda: not sheet.dialog.open, 5)

            # ---- /profile: the ModelSheet Profile tab, same 18 with previews from the shared listing -----------
            await _run_command(driver, "/profile")
            model_sheet = await _until(lambda: getattr(view, "model_sheet", None), 10)
            assert model_sheet is not None, "/profile did not open the ModelSheet"
            _assert_sheet_grouped(model_sheet.profile_rows(), BUILTINS)
            assert TRANSLATION == ["Universal", "Korean_BeautifulSoup", "Japanese_BeautifulSoup", "Chinese_BeautifulSoup",
                                   "Korean_html2text", "Japanese_html2text", "Chinese_html2text"]
            for name in ("Manga_JP", "Subtitle Translation", "Korean_BeautifulSoup"):
                assert model_sheet._profile_preview(name), f"no preview for the built-in {name} on a fresh config"
                await driver.wait(key=f"profile-{name}", timeout=5)
            model_sheet.close()
            assert "prompt_profiles" not in _stored(app)  # listing / previews wrote nothing
            # /profile <name> (case-insensitive) picks a built-in the old list never had
            await _run_command(driver, "/profile manga_jp")
            assert await _until(lambda: store.get("active_profile") == "Manga_JP", 10), store.get("active_profile")
            saved = _stored(app)
            assert saved.get("active_profile") == "Manga_JP"
            assert list(saved.get("prompt_profiles") or {}) == BUILTINS  # the desktop save_profiles key set
            assert saved["prompt_profiles"]["Manga_JP"] == defaults["Manga_JP"]

            # ---- Settings › Profiles & prompts: everything, desktop order ---------------------------------------
            screen = await _open_profiles_screen(app, driver)
            assert _screen_names(screen) == BUILTINS
            for name in ("Refinement", "Manga_KR", "NanoBanana_Image", "Subtitle Translation"):
                from glossarion_mobile.ui.screens.profiles import profile_id

                await driver.wait(key=f"profile-{profile_id(name)}", timeout=5)
        except Exception:
            for row in tester.dump(200):
                print(row)
            raise
        finally:
            await tf._stop(app)

    asyncio.run(scenario())


# ---- desktop import, custom profiles, rename / delete -----------------------------------------------------


def test_owner_issue16_desktop_profiles_custom_ones_and_no_silent_universal(isolated):
    picks = isolated
    tf = _foundations()

    async def scenario(server, captured):
        import flows
        import prompt_profiles
        from glossarion_mobile.ui.screens.profiles import ProfileService, chat_profile_order, profile_id

        defaults = prompt_profiles.profile_state_from_config({}).default_prompts
        universal_line = _signature_line(defaults["Universal"])
        html2text_line = _signature_line(defaults["Korean_html2text"], avoid=(defaults["Universal"],))
        # an older desktop: no NanoBanana_Image / SDLXLIFF Editing v2 / Subtitle Translation yet, three customs
        profiles = _old_desktop_profiles(defaults)
        v1 = _desktop_config(server.url, profiles=profiles, active=WUXIA)
        v2 = _desktop_config(server.url, profiles={n: t for n, t in profiles.items() if n not in (CASUAL, RENAME_ME)},
                             active="Korean_html2text")
        files = {DESKTOP_V1: picks / DESKTOP_V1, DESKTOP_V2: picks / DESKTOP_V2}
        files[DESKTOP_V1].write_text(json.dumps(v1, ensure_ascii=False, indent=2), encoding="utf-8")
        files[DESKTOP_V2].write_text(json.dumps(v2, ensure_ascii=False, indent=2), encoding="utf-8")
        expected = list(prompt_profiles.profile_state_from_config(v1).prompt_profiles)  # desktop start-up order
        assert len(expected) == 21 and set(BUILTINS) | set(MARKERS) == set(expected), expected

        app, page, tester, driver = await _host_driver(tf, files)
        try:
            await flows.wait_home(driver)
            await flows.import_desktop_config(driver, DESKTOP_V1)
            store = app.config_store
            assert store.get("active_profile") == WUXIA
            chats = app.chat_feature.chats
            view = app.chat_view
            #: defects found on the way; the scenario puts the user's state back and goes on, then fails at the end
            problems: list = []

            # ---- Settings › Profiles & prompts: everything the desktop shows, in its order ----------------------
            screen = await _open_profiles_screen(app, driver)
            assert _screen_names(screen) == expected
            for name in (*NEWER_BUILTINS, *MARKERS):
                await driver.wait(key=f"profile-{profile_id(name)}", timeout=5)
            await flows.go_home(driver)

            # ---- the chat picker: the same 21, customs with the translation profiles ----------------------------
            await driver.tap(tooltip="New chat")
            await driver.wait(key="transcript-empty", timeout=30)
            cid1 = view.cid
            # /profile: the ModelSheet Profile tab lists the same 21 (customs with the translation profiles, the
            # Specialised heading once), previews the customs and marks the imported active profile
            await _run_command(driver, "/profile")
            model_sheet = await _until(lambda: getattr(view, "model_sheet", None), 10)
            assert model_sheet is not None, "/profile did not open the ModelSheet"
            _assert_sheet_grouped(model_sheet.profile_rows(), expected)
            assert [n for n in expected if getattr(model_sheet.rows.get(n), "selected", False)] == [WUXIA]
            for name, marker in MARKERS.items():
                assert marker in (model_sheet._profile_preview(name) or ""), name
                await driver.wait(key=f"profile-{name}", timeout=5)
            model_sheet.close()
            sheet, dropdown = await _open_chat_settings(app, driver, tester)
            options = _option_rows(dropdown)
            _assert_grouped(options, expected)
            assert [k for k, _t, _d in options if not k.startswith(GROUP_PREFIX)] == chat_profile_order(expected)
            assert dropdown.value == WUXIA  # the imported active profile, inherited from All chats
            await _select(tester, dropdown, CASUAL)
            assert chats.own_overrides(cid1).get("profile") == CASUAL
            assert store.get("active_profile") == WUXIA  # This chat never switches the global profile
            sheet.close()
            snap, payloads = await _send_text(app, driver, server, captured, "안녕하세요. 오늘은 날씨가 맑아요.")
            assert (snap.spec.params.get("config_overrides") or {}).get("active_profile") == CASUAL
            prompt = _prompt_text(payloads)
            assert MARKERS[CASUAL] in prompt and universal_line not in prompt, prompt[:600]

            # a series that defaults to the same profile
            series = app.series.store.create("Murim Saga")
            series_sheet = app.series.open_defaults_sheet(series.id)
            assert series_sheet is not None
            series_dropdown = _profile_dropdown(tester, await driver.wait(key="setting-profile", timeout=10))
            _assert_grouped(_option_rows(series_dropdown), expected)
            await _select(tester, series_dropdown, CASUAL)
            assert app.series.store.defaults(series.id).get("profile") == CASUAL
            series_sheet.close()

            # ---- delete the chat's profile in Settings › Profiles & prompts -------------------------------------
            screen = await _open_profiles_screen(app, driver)
            await tester.long_press(await driver.wait(key=f"profile-{profile_id(CASUAL)}", timeout=10))
            await driver.tap(key="Delete", timeout=10)  # the row's action sheet
            await driver.wait(text=f"Are you sure you want to delete profile '{CASUAL}'?", timeout=10)
            dialog = next((d for d in reversed(page._dialogs.controls)
                           if getattr(d, "open", False) and type(d).__name__ == "AlertDialog"
                           and getattr(getattr(d, "title", None), "value", "") == "Delete"), None)
            assert dialog is not None, "no Delete confirmation"
            confirm = dialog.actions[-1]
            assert getattr(confirm, "content", None) == "Delete", dialog.actions
            await _tap_control(tester, confirm)
            assert await _until(lambda: CASUAL not in (store.get("prompt_profiles") or {}), 10)
            # the chat and the series go back to their inherited profile (never a stale name -> Universal)
            assert await _until(lambda: chats.own_overrides(cid1).get("profile") is None, 10), \
                chats.own_overrides(cid1)
            assert await _until(lambda: app.series.store.defaults(series.id).get("profile") is None, 10), \
                app.series.store.defaults(series.id)
            await driver.wait(contains=INHERITED_NOTICE, timeout=10)  # and the user is told
            assert CASUAL not in _screen_names(screen)
            await flows.go_home(driver)
            assert view.cid == cid1
            active = store.get("active_profile")
            sheet, dropdown = await _open_chat_settings(app, driver, tester)
            if active != WUXIA or dropdown.value != WUXIA:
                # the desktop can only delete the SELECTED (= active) profile; a row Delete of another profile
                # must not move All chats / the desktop to the first profile
                problems.append(
                    f"Settings › Profiles & prompts › Delete of '{CASUAL}' (NOT the active profile) silently switched "
                    f"the global All chats / desktop active profile {WUXIA!r} -> {active!r}: chat {cid1}, whose "
                    f"profile was deleted, now inherits {dropdown.value!r} instead of {WUXIA!r}, and every chat "
                    f"without its own profile runs {active!r} too (silent Universal fallback; ProfilesScreen."
                    "delete_or_reset calls ProfileService.delete_or_reset without keep_active, and the shared "
                    "delete_or_reset_profile then selects the first profile)")
                sheet.close()
                ProfileService(store).select(WUXIA)  # put the user's profile back; the rest checks inheritance
                sheet, dropdown = await _open_chat_settings(app, driver, tester)
            assert dropdown.value == WUXIA and CASUAL not in [k for k, _t, _d in _option_rows(dropdown)]
            assert not await driver.exists(contains="(missing)", timeout=0.5)
            await driver.wait(text="Inherited from: All chats", timeout=5)
            sheet.close()
            snap, payloads = await _send_text(app, driver, server, captured, "두 번째 메시지입니다.")
            assert not (snap.spec.params.get("config_overrides") or {}).get("active_profile")
            prompt = _prompt_text(payloads)
            assert MARKERS[WUXIA] in prompt, prompt[:600]
            assert MARKERS[CASUAL] not in prompt and universal_line not in prompt, prompt[:600]

            # ---- rename: the chat follows ----------------------------------------------------------------------
            await driver.tap(tooltip="New chat")
            await driver.wait(key="transcript-empty", timeout=30)
            cid2 = view.cid
            sheet, dropdown = await _open_chat_settings(app, driver, tester)
            await _select(tester, dropdown, RENAME_ME)
            assert chats.own_overrides(cid2).get("profile") == RENAME_ME
            sheet.close()
            await _open_profiles_screen(app, driver)
            await driver.tap(key=f"profile-{profile_id(RENAME_ME)}", timeout=10)
            await driver.wait(text="Profile name", timeout=10)
            await driver.enter(RENAMED, text="Profile name")
            await driver.tap(text="Save", timeout=5)
            assert await _until(lambda: RENAMED in (store.get("prompt_profiles") or {})
                                and RENAME_ME not in (store.get("prompt_profiles") or {}), 10)
            active = store.get("active_profile")
            if active != WUXIA:
                problems.append(
                    f"Settings › Profiles & prompts › '{RENAME_ME}' › Save under the new name '{RENAMED}' (NOT the "
                    f"active profile) silently switched the global All chats / desktop active profile {WUXIA!r} -> "
                    f"{active!r} (ProfileDetailScreen.save calls ProfileService.save without keep_active; only "
                    "'Use this profile' should switch it)")
                ProfileService(store).select(WUXIA)
            assert await _until(lambda: chats.own_overrides(cid2).get("profile") == RENAMED, 10), \
                chats.own_overrides(cid2)
            await flows.go_home(driver)
            assert view.cid == cid2
            sheet, dropdown = await _open_chat_settings(app, driver, tester)
            assert dropdown.value == RENAMED
            assert RENAMED in [k for k, _t, _d in _option_rows(dropdown)]
            sheet.close()
            snap, payloads = await _send_text(app, driver, server, captured, "세 번째 메시지입니다.")
            assert (snap.spec.params.get("config_overrides") or {}).get("active_profile") == RENAMED
            prompt = _prompt_text(payloads)
            assert MARKERS[RENAME_ME] in prompt and universal_line not in prompt, prompt[:600]

            # ---- deleted on the desktop: a second import without the chat's profile -----------------------------
            await flows.import_desktop_config(driver, DESKTOP_V2)
            assert await _until(lambda: store.get("active_profile") == "Korean_html2text", 10)
            assert RENAMED not in (store.get("prompt_profiles") or {})
            assert await _until(lambda: chats.own_overrides(cid2).get("profile") is None, 10), \
                chats.own_overrides(cid2)
            await flows.go_home(driver)
            sheet, dropdown = await _open_chat_settings(app, driver, tester)
            assert dropdown.value == "Korean_html2text"  # the newly imported All chats profile
            options = _option_rows(dropdown)
            expected_v2 = list(prompt_profiles.profile_state_from_config(_stored(app)).prompt_profiles)
            assert len(expected_v2) == 19 and WUXIA in expected_v2 and RENAMED not in expected_v2, expected_v2
            _assert_grouped(options, expected_v2)
            sheet.close()
            snap, payloads = await _send_text(app, driver, server, captured, "네 번째 메시지입니다.")
            prompt = _prompt_text(payloads)
            assert html2text_line in prompt, prompt[:600]
            assert universal_line not in prompt and MARKERS[RENAME_ME] not in prompt, prompt[:600]
            assert not problems, "\n".join(problems)
        except Exception:
            for row in tester.dump(200):
                print(row)
            raise
        finally:
            app.jobs.close()
            await tf._stop(app)

    _run_with_server(scenario)


def _run_with_server(scenario) -> None:
    """``scenario(server, captured)`` against the fake OpenAI server; ``captured`` collects every chat request
    payload the app sends (the server itself keeps only summaries)."""
    from glossarion_mobile.diagnostics.fake_llm_server import FakeLLMServer

    with FakeLLMServer() as server:
        captured: list = []
        original = server._classify

        def classify(payload):
            captured.append(payload)
            return original(payload)

        server._classify = classify
        asyncio.run(scenario(server, captured))


def _old_desktop_profiles(defaults: dict) -> dict:
    """An older desktop's prompt_profiles: the built-ins it had (no NanoBanana_Image / SDLXLIFF Editing v2 /
    Subtitle Translation yet) plus the three custom profiles of ``MARKERS``."""
    old = {name: text for name, text in defaults.items() if name not in NEWER_BUILTINS}
    customs = {name: f"You translate into {{target_lang}}.\n{marker}\nKeep the author's tone."
               for name, marker in MARKERS.items()}
    return {**old, **customs}


def test_owner_issue16_a_stale_chat_profile_at_send_runs_the_inherited_profile(isolated):
    """The chat's profile is gone but its sidecar still names it when the user presses Send: the app was killed
    after the delete reached config.json and before the posted re-check of the chats ran, and the next start's
    re-check runs only after the Library is up (``startup_sweep`` waits up to 30 s). Chat settings shows
    "<name> (missing)"; the run itself must use the inherited (All chats) profile, never the backend's
    silent first profile (Universal). The stale sidecar entry is written the way it was left behind."""
    picks = isolated
    tf = _foundations()

    async def scenario(server, captured):
        import flows
        import prompt_profiles
        from glossarion_mobile.ui.screens.profiles import ProfileService

        defaults = prompt_profiles.profile_state_from_config({}).default_prompts
        universal_line = _signature_line(defaults["Universal"])
        v1 = _desktop_config(server.url, profiles=_old_desktop_profiles(defaults), active=WUXIA)
        files = {DESKTOP_V1: picks / DESKTOP_V1}
        files[DESKTOP_V1].write_text(json.dumps(v1, ensure_ascii=False, indent=2), encoding="utf-8")
        app, page, tester, driver = await _host_driver(tf, files)
        try:
            await flows.wait_home(driver)
            await flows.import_desktop_config(driver, DESKTOP_V1)
            await flows.go_home(driver)
            store = app.config_store
            chats = app.chat_feature.chats
            view = app.chat_view
            await driver.tap(tooltip="New chat")
            await driver.wait(key="transcript-empty", timeout=30)
            cid = view.cid
            # the profile is deleted (global profile kept) and the chats re-checked ...
            ProfileService(store).delete_or_reset(CASUAL, keep_active=True)
            assert CASUAL not in (store.get("prompt_profiles") or {}) and store.get("active_profile") == WUXIA
            await driver.pump(300)
            # ... but this chat's sidecar entry predates that re-check (written as the killed app left it)
            chats.set_override(cid, "profile", CASUAL)
            view.apply_settings_changed()
            sheet, dropdown = await _open_chat_settings(app, driver, tester)
            assert (CASUAL, f"{CASUAL} (missing)", False) in _option_rows(dropdown)  # the sheet says so
            sheet.close()
            snap, payloads = await _send_text(app, driver, server, captured, "안녕하세요. 반갑습니다.")
            prompt = _prompt_text(payloads)
            used = (snap.spec.params.get("config_overrides") or {}).get("active_profile")
            assert MARKERS[WUXIA] in prompt and universal_line not in prompt, (
                f"a chat whose stored profile {CASUAL!r} no longer exists ran with config_overrides active_profile="
                f"{used!r}: the backend silently fell back to "
                f"{'Universal' if universal_line in prompt else 'another profile'} instead of the inherited "
                f"All chats profile {WUXIA!r} (no send-time guard in run_request.config_overrides / ChatView "
                "before job_params)")
        except Exception:
            for row in tester.dump(120):
                print(row)
            raise
        finally:
            app.jobs.close()
            await tf._stop(app)

    _run_with_server(scenario)


async def _end_run(app, run) -> None:
    """Teardown: force-stop the run's job if it still runs and wait for the chat's finish."""
    service = app.job_service
    job_id = getattr(run, "job_id", None)
    if job_id is None:
        return
    snap = service.snapshot(job_id)
    if snap is not None and not snap.is_terminal:
        service.request_stop(job_id, force=True, reason="test teardown")
    await asyncio.to_thread(service.wait_idle, 60)
    for thread in list(app.chat_feature.runs.finish_threads):
        await asyncio.to_thread(thread.join, 60)


@pytest.mark.parametrize("change", ["rename", "delete"])
def test_owner_issue16_resume_after_the_chats_profile_was_renamed_or_deleted(isolated, change):
    """DF2 verify: a book in a chat with its own profile (Casual KR) is stopped; the profile is renamed or deleted
    under Manage (``ProfileService``, ``keep_active`` like Settings › Profiles & prompts); the ended card's Resume.
    The resumed run follows the rename (the same prompt, under its new name) or, deleted, runs the inherited All
    chats profile (Wuxia House Style) - never the backend's silent first profile (Universal), which is what a
    Resume of the Send-time JobSpec params ran."""
    picks = isolated
    tf = _foundations()

    async def scenario(server, captured):
        import flows
        import prompt_profiles
        from glossarion_mobile.diagnostics import fixtures
        from glossarion_mobile.diagnostics.fake_llm_server import FAKE_MODEL
        from glossarion_mobile.ui.chat.cards import JobCard
        from glossarion_mobile.ui.chat.job_binding import progress_counts
        from glossarion_mobile.ui.screens.profiles import ProfileService

        server.set_delay("translation", 0.05)  # a run long enough to stop in the middle
        # After 3 chapter replies the fake model answers slowly, so the Stop lands with most of the book
        # left on any runner (Build Mobile 37977210280: a fast runner finished every chapter first and
        # 'the resumed run sent nothing'). Normal speed again before Resume.
        pace = {"replies": 0, "slow_after": 3}

        def pace_replies(record):
            if record.kind == "translation":
                pace["replies"] += 1
                if pace["slow_after"] is not None and pace["replies"] >= pace["slow_after"]:
                    server.set_delay("translation", 1.0)

        server.on_response.append(pace_replies)
        defaults = prompt_profiles.profile_state_from_config({}).default_prompts
        universal_line = _signature_line(defaults["Universal"])
        book = fixtures.build_tiny_epub(picks / "resume_book.epub", chapters=30)
        app, page, tester, driver = await _host_driver(tf, {book.name: book})
        runs = app.chat_feature.runs
        run = resumed = None
        try:
            await flows.wait_home(driver)
            store = app.config_store
            profiles = dict(store.get("prompt_profiles") or {})
            profiles.update(_old_desktop_profiles(defaults))
            store.set_many({**flows.ui_config(server.url, FAKE_MODEL), "prompt_profiles": profiles,
                            "active_profile": WUXIA})
            store.flush()
            await flows.go_home(driver)
            await driver.tap(tooltip="New chat")
            await driver.wait(key="transcript-empty", timeout=30)
            view = app.chat_view
            cid = view.cid
            chats = app.chat_feature.chats
            chats.set_override(cid, "profile", CASUAL)
            view.apply_settings_changed()
            await driver.tap(tooltip=flows.ATTACH_TOOLTIP)
            await driver.pick_file(book.name, lambda: driver.tap(key="attach-files"))
            await driver.wait(contains=book.stem, timeout=60)
            await driver.tap(key="send-idle_ready", timeout=60)
            await driver.wait(text="Ready to translate", timeout=60)
            await driver.tap(text="Start")
            if await driver.exists(text="Keep translations running", timeout=3):
                await driver.tap(text="Not now")
            run = await _until(lambda: runs.run_for(cid) if runs.run_for(cid) is not None
                               and runs.run_for(cid).job_id is not None else None, 60)
            assert run is not None, "Start did not submit the chat's job"

            def completed(r) -> int:
                return progress_counts(app.job_service.snapshot(r.job_id)).get("completed", 0)

            assert await _until(lambda: completed(run) >= 3, 120), "the first run translated nothing"
            first = _prompt_text(captured)
            assert MARKERS[CASUAL] in first and universal_line not in first
            await _tap_control(tester, view.live_job_card.action_buttons["stop"])
            assert await _until(lambda: not run.live, 120) and run.state == "stopped", run.state
            for thread in list(runs.finish_threads):
                await asyncio.to_thread(thread.join, 60)
            await driver.pump(300)
            pace["slow_after"] = None
            server.set_delay("translation", 0.05)

            # ---- Manage…: the chat's profile is renamed / deleted (the global profile stays) -------------------
            service = ProfileService(store)
            if change == "rename":
                service.save(CASUAL, RENAMED, profiles[CASUAL], keep_active=True)
                expected_marker, expected_name = MARKERS[CASUAL], RENAMED
            else:
                service.delete_or_reset(CASUAL, keep_active=True)
                expected_marker, expected_name = MARKERS[WUXIA], None
            assert CASUAL not in (store.get("prompt_profiles") or {}) and store.get("active_profile") == WUXIA
            await driver.pump(300)

            # ---- Resume on the ended card ----------------------------------------------------------------------
            mark = len(captured)
            ended = [getattr(s, "card", s) for s in view.transcript.messages]
            ended = [c for c in ended if isinstance(c, JobCard) and "resume" in c.action_buttons]
            assert len(ended) == 1, "no ended card with Resume"
            await _tap_control(tester, ended[0].action_buttons["resume"])
            resumed = await _until(lambda: runs.run_for(cid) if runs.run_for(cid) is not run
                                   and runs.run_for(cid) is not None and runs.run_for(cid).live else None, 30)
            assert resumed is not None, "Resume did not start a run"
            assert await _until(lambda: len(captured) - mark >= 3, 120), "the resumed run sent nothing"
            prompt = _prompt_text(captured[mark:])
            used = (resumed.params.get("config_overrides") or {}).get("active_profile")
            ran = ("Universal" if universal_line in prompt else "the expected profile" if expected_marker in prompt
                   else "another profile")
            assert expected_marker in prompt and universal_line not in prompt, (
                f"after the chat's profile {CASUAL!r} was {change}d, its Resume ran {ran} "
                f"(config_overrides active_profile={used!r})")
            assert used == expected_name, used
        except Exception:
            for row in tester.dump(120):
                print(row)
            raise
        finally:
            for r in (resumed, run):
                if r is not None:
                    await _end_run(app, r)
            app.jobs.close()
            await tf._stop(app)

    _run_with_server(scenario)


# ---- Settings › Translation › Profile & System Prompt ------------------------------------------------------


async def _open_prompt_section(app, driver):
    """Drawer › Settings › Profile & System Prompt (schema section ``main.prompt``, the Translation group):
    where a user who looks in Settings for the prompts lands. Returns the SectionPage."""
    import flows
    from glossarion_mobile.ui.settings.section_page import SectionPage

    await flows.open_settings(driver)
    await driver.tap(key="settings-section-main.prompt", timeout=flows.SCROLL_TIMEOUT, scroll=True)

    def shown():
        top = app.shell.top_screen
        return top if isinstance(top, SectionPage) and top.section_id == "main.prompt" and top.keys else None

    section = await _until(shown, 10)
    assert section is not None, type(app.shell.top_screen)
    await driver.wait(key="active_profile", timeout=10)
    return section


def test_owner_issue16_settings_profile_section_leads_to_every_profile_never_to_universal(isolated):
    """The skeptic-checked diagnosis found the same symptom in Settings › Translation › Profile & System Prompt:
    its "Profile" showed just "Universal" as text, next to a raw "Prompt profiles" JSON tile. With a desktop
    config imported (older desktop + custom profiles): the section links to Settings › Profiles & prompts,
    which lists all 21; its "Profile" tile cannot make a name that is not a profile the active one (the
    desktop combo only switches to an existing profile; a stored unknown name makes every run silently use
    the first profile, Universal); and its raw "Prompt profiles" JSON is not a way around the profile editor
    (a partial dict drops the built-ins the desktop start-up does not re-add: Manga_*, Glossary_Editor,
    Original). Defects are collected (the user's state is put back after each) and reported at the end."""
    picks = isolated
    tf = _foundations()

    async def scenario(server, captured):
        import flows
        import prompt_profiles
        from glossarion_mobile.ui.screens.profiles import ProfileService, ProfilesScreen

        defaults = prompt_profiles.profile_state_from_config({}).default_prompts
        universal_line = _signature_line(defaults["Universal"])
        v1 = _desktop_config(server.url, profiles=_old_desktop_profiles(defaults), active=WUXIA)
        files = {DESKTOP_V1: picks / DESKTOP_V1}
        files[DESKTOP_V1].write_text(json.dumps(v1, ensure_ascii=False, indent=2), encoding="utf-8")
        expected = list(prompt_profiles.profile_state_from_config(v1).prompt_profiles)
        assert len(expected) == 21, expected
        app, page, tester, driver = await _host_driver(tf, files)
        try:
            await flows.wait_home(driver)
            await flows.import_desktop_config(driver, DESKTOP_V1)
            store = app.config_store
            problems: list = []

            # ---- the section links to the full list -------------------------------------------------------------
            section = await _open_prompt_section(app, driver)
            await driver.tap(key="link-settings.profiles", timeout=10)
            screen = await _until(lambda: app.shell.top_screen if isinstance(app.shell.top_screen, ProfilesScreen)
                                  else None, 10)
            assert screen is not None, "the section's 'Profiles & prompts…' link opened nothing"
            assert _screen_names(screen) == expected

            # ---- "Profile": only a real profile becomes the active one ----------------------------------------
            section = await _open_prompt_section(app, driver)
            tile = section.tile("active_profile")
            assert tile.value() == WUXIA, tile.value()
            typed = WUXIA.lower()  # the profile's name with its case off, as a phone keyboard types it
            if tile.kind == "text" and tile.editable:
                await tester.enter_text(tester._register([tile.field]), typed)
                await tester._dispatch(tile.field, "submit", typed)  # the keyboard's Done
            elif tile.editable:  # a picker: whatever it offers must be a profile
                tile.apply(typed)
            else:
                assert tile.readonly_reason, "a read-only Profile tile says where profiles are chosen"
            await driver.pump(200)
            active = store.get("active_profile")
            if active not in ProfileService(store).listing().names:
                await flows.go_home(driver)
                await driver.tap(tooltip="New chat")
                await driver.wait(key="transcript-empty", timeout=30)
                snap, payloads = await _send_text(app, driver, server, captured, "안녕하세요. 좋은 아침입니다.")
                prompt = _prompt_text(payloads)
                ran = ("Universal" if universal_line in prompt else WUXIA if MARKERS[WUXIA] in prompt
                       else "another profile")
                problems.append(
                    f"Settings › Translation › Profile & System Prompt › 'Profile' is a free-text field ({tile.kind} "
                    f"tile, editable, no read-only reason, no list of profiles): typing {typed!r} there stored "
                    f"active_profile={active!r}, which is not a profile (setting_writes._select_profile keeps a name "
                    f"ProfileService.select rejects as the plain key; the desktop combo only ever switches to an "
                    f"existing profile), and the next chat run used {ran}: every chat without its own profile now "
                    "silently runs the first profile (owner_state._init_variables), with no notice")
                ProfileService(store).select(WUXIA)  # put the user's profile back
                assert store.get("active_profile") == WUXIA
                section = await _open_prompt_section(app, driver)

            # ---- "Prompt profiles": no raw JSON way around the profile editor -----------------------------------
            json_tile = section.tile("prompt_profiles")
            if json_tile.editable:
                await driver.wait(key="prompt_profiles", timeout=10)  # the tile on screen; its ListTile takes taps
                await _tap_control(tester, json_tile.list_tile)
                editor = await _until(lambda: json_tile.editor, 10)
                assert editor is not None, "the editable Prompt profiles tile opened no editor"
                kept = {n: t for n, t in (store.get("prompt_profiles") or {}).items() if n in ("Universal", WUXIA)}
                await tester.enter_text(tester._register([editor.field]), json.dumps(kept, ensure_ascii=False))
                await _tap_control(tester, editor.save_button)
                await driver.pump(300)
                after = list(ProfileService(store).listing().names)
                lost = [n for n in BUILTINS if n not in after]
                problems.append(
                    "Settings › Translation › Profile & System Prompt › 'Prompt profiles' is an editable raw JSON tile "
                    "(no read-only reason: the checked fix design marks prompt_profiles / active_profile in "
                    "settings_schema.READONLY_REASONS, like glossary_prompt_profiles, so the profile editor and its "
                    "desktop save_profiles semantics are the only way to change them)"
                    + (f"; saving it with only Universal and {WUXIA!r} left the built-ins {lost} missing from every "
                       "mobile profile list for good (the desktop start-up re-adds only its always-include built-ins; "
                       "Settings › Profiles & prompts never deletes a built-in, it resets it)" if lost else ""))
            assert not problems, "\n".join(problems)
        except Exception:
            for row in tester.dump(150):
                print(row)
            raise
        finally:
            app.jobs.close()
            await tf._stop(app)

    _run_with_server(scenario)
