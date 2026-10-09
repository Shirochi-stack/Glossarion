"""UI flows shared by the device tests (``test_ui_*.py``) and their host runs
(``tests_host/test_ui_flows.py``). Every step goes through ``UiDriver`` (keys, tooltips, labels).

Seed for the chat flow: a desktop-style ``config.json`` that points the default OpenAI route at
the fake model server (``glossarion_mobile.diagnostics.fake_llm_server``, run by the test process
on 127.0.0.1; a device reaches it through ``adb reverse``), imported with Settings › Import from
desktop, exactly as a user moves their desktop settings over.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "ATTACH_TOOLTIP", "CONFIG_NAME", "EPUB_NAME", "chat_translate_and_migrate", "dismiss_welcome", "go_home",
    "import_desktop_config", "library_book_chapters", "open_drawer", "open_settings", "POP_SETTLE", "run_selftest",
    "SCROLL_TIMEOUT", "smoke_navigation", "ui_config", "wait_home", "write_ui_config",
]

ATTACH_TOOLTIP = "Attach, output mode and tools"
CONFIG_NAME = "glossarion-ui-config.json"
EPUB_NAME = "glossarion-ui-selftest.epub"
#: start of the dc:title of the 12-chapter self-test EPUB (diagnostics/fixtures.py)
EPUB_TITLE = "글로사리온 자가진단"


def ui_config(base_url: str, model: str) -> dict:
    """The desktop config the chat flow imports: the fake OpenAI endpoint, a dummy key, the
    welcome's "no glossary" choice (no approval card), English output, no request spacing."""
    return {
        "model": model,
        "api_key": "sk-glossarion-ui-test-dummy",
        "use_custom_openai_endpoint": True,
        "openai_base_url": base_url,
        "output_language": "English",
        "delay": 0,
        "glossary_mode_dialog_shown": True,
        "auto_glossary_mode": "off",
        "enable_auto_glossary": False,
    }


def write_ui_config(path: Path, base_url: str, model: str) -> Path:
    path.write_text(json.dumps(ui_config(base_url, model), indent=2), encoding="utf-8")
    return path


# ---- navigation ---------------------------------------------------------------------------------

async def dismiss_welcome(d: Any, timeout: float = 10.0, *, first_run: bool = False) -> bool:
    """The Welcome flow covers the chat on a fresh install; "Skip" closes it. Returns whether it
    was skipped.

    ``first_run=True`` (the device tests: ``flutter test`` installs the app fresh for every test
    and conftest uninstalls a leftover copy first): tap "Skip" once it is up, then wait until it is
    gone; never settle for the chat home. The app mounts the home first and pushes the Welcome
    only after its feature installs, so a home seen early is about to be covered: in Build Mobile
    run 37800059580 the drawer then opened under the Welcome and 'dest-library' was never found.
    Without it (host runs, a returning user) return once either one is on screen."""
    if first_run:
        await d.tap(text="Skip", timeout=timeout)
        await d.wait(text="Skip", gone=True, timeout=30)
        return True
    index = await d.wait_any({"text": "Skip"}, {"tooltip": ATTACH_TOOLTIP}, timeout=timeout)
    if index == 0:
        await d.tap(text="Skip")
        await d.wait(text="Skip", gone=True, timeout=30)
        return True
    return False


async def wait_home(d: Any, timeout: float = 180.0) -> None:
    await d.wait(tooltip=ATTACH_TOOLTIP, timeout=timeout)


#: How long ``go_home`` waits for a Back to land before it presses another (a device only).
POP_SETTLE = 5.0


async def go_home(d: Any, max_steps: int = 6) -> None:
    """Back to the chat home, one screen at a time. A screen's own app-bar Back (Flutter's BackButton,
    tooltip "Back") is tapped when it has one, else the system Back is pressed, and the home is waited
    for before another Back. On a device a pop is a round trip (Flutter -> the app's on_view_pop -> the
    patch back): the old loop looked 300 ms after a system Back, pressed again, and that second Back
    reached the root route, where Android finishes the activity under the test (Build Mobile run
    37940790686: "Remote tester connection was closed"). An app-bar Back can never do that. The host
    tester handles the pop before ``back`` returns (``events_land_at_once``), so it does not wait."""
    from ui_driver import UiTimeout

    settle = 0.0 if getattr(getattr(d, "t", None), "events_land_at_once", False) else POP_SETTLE
    for step in range(max_steps):
        if await d.exists(tooltip="Open navigation", timeout=settle if step else 0.0):
            return
        if await d.count(tooltip="Back"):
            try:
                await d.tap(tooltip="Back", timeout=settle or None)
            except UiTimeout:
                pass  # the Back went away meanwhile: the previous pop landed
        else:
            await d.back()
    await d.wait(tooltip="Open navigation", timeout=10)


async def open_drawer(d: Any) -> None:
    await go_home(d)
    await d.tap(tooltip="Open navigation")
    await d.wait(key="dest-library", timeout=15)


async def open_settings(d: Any) -> None:
    """The Settings home (its page tiles are keyed ``hub-<route>``; lower groups need scrolling)."""
    await open_drawer(d)
    await d.tap(tooltip="Settings")
    await d.wait(key="hub-settings.appearance", timeout=30)


#: a device scrolls the Settings home's lower groups into view one swipe per poll (the Data group
#: is about a dozen swipes down a 320x640 emulator; a cold emulator takes 1-2 s per swipe)
SCROLL_TIMEOUT = 90.0


# ---- flows --------------------------------------------------------------------------------------

async def smoke_navigation(d: Any) -> None:
    """Chat home → drawer → Library (shelves) → Settings home → About › Updates page is listed."""
    await wait_home(d)
    await d.wait(key="mode-chip", timeout=30)
    await open_drawer(d)
    await d.tap(key="dest-library")
    await d.wait(key="lib-search", timeout=60)
    await open_settings(d)
    await d.wait(key="hub-settings.import", timeout=SCROLL_TIMEOUT, scroll=True)
    await d.wait(key="hub-settings.logs", timeout=SCROLL_TIMEOUT, scroll=True)
    await d.wait(key="hub-settings.updates", timeout=SCROLL_TIMEOUT, scroll=True)


async def run_selftest(d: Any, timeout: float = 600.0) -> str:
    """Settings › Logs & diagnostics › Run self-test → the result card says PASS."""
    await open_settings(d)
    await d.tap(key="hub-settings.logs", timeout=SCROLL_TIMEOUT, scroll=True)
    await d.tap(text="Run self-test", timeout=SCROLL_TIMEOUT, scroll=True)
    index = await d.wait_any({"contains": "PASS · "}, {"contains": "FAIL · "}, timeout=timeout)
    assert index == 0, "the self-test reported FAIL"
    return "PASS"


async def import_desktop_config(d: Any, name: str = CONFIG_NAME) -> None:
    await open_settings(d)
    await d.tap(key="hub-settings.import", timeout=SCROLL_TIMEOUT, scroll=True)
    await d.pick_file(name, lambda: d.tap(key="import-pick-config"))
    await d.wait(text=name, timeout=30)  # the chosen file's name
    await d.tap(key="import-run", timeout=60, scroll=True)
    await d.wait(contains="Imported ", timeout=60)


async def chat_translate_and_migrate(d: Any, epub: str = EPUB_NAME, timeout: float = 900.0) -> None:
    """A new chat, ＋ › Files with the EPUB, Send; the job card reaches Done; the book moves into the
    Library by itself ("Added to the Library"; there is no Migrate step)."""
    await go_home(d)
    await d.tap(tooltip="New chat")
    await d.tap(tooltip=ATTACH_TOOLTIP)
    await d.pick_file(epub, lambda: d.tap(key="attach-files"))
    await d.wait(contains=Path(epub).stem, timeout=60)  # the composer pill
    await d.tap(key="send-idle_ready", timeout=60)
    await d.wait(text="Ready to translate", timeout=60)  # the run plan card (model, glossary, output)
    await d.tap(text="Start")
    # First long job on Android: the battery-optimisation explanation (the system's own dialog
    # would follow "Continue"); "Not now" starts the job without it.
    if await d.exists(text="Keep translations running", timeout=15):
        await d.tap(text="Not now")
    index = await d.wait_any({"text": "Done"}, {"text": "Failed"}, {"text": "Stopped"}, timeout=timeout)
    assert index == 0, "the translation job did not finish"
    await d.wait(contains="Added to the Library", timeout=60)


async def library_book_chapters(d: Any, title: str = EPUB_TITLE) -> None:
    """Library → Completed shelf → the chat's book (moved in automatically) → Chapters: none
    untranslated or failed."""
    await open_drawer(d)
    await d.tap(key="dest-library")
    await d.wait(key="shelf", timeout=60)
    await d.tap(contains="Completed (", timeout=60)  # the shelf (a finished book)
    names = ({"contains": title}, {"contains": Path(EPUB_NAME).stem})
    index = await d.wait_any(*names, timeout=120)
    await d.tap(**names[index])
    await d.tap(text="Chapters", timeout=60)
    # the Chapters tab's progress chips: nothing left untranslated or failed; the first row done
    await d.wait(key="ch-stats", timeout=60)
    await d.wait(contains="Not Translated 0", timeout=60)
    await d.wait(contains="Failed 0", timeout=30)
    await d.wait(contains="Ch.001", timeout=30)
