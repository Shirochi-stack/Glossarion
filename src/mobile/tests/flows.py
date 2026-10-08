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
    "import_desktop_config", "library_book_chapters", "open_drawer", "open_settings", "run_selftest",
    "smoke_navigation", "ui_config", "wait_home", "write_ui_config",
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

async def dismiss_welcome(d: Any, timeout: float = 10.0) -> bool:
    """First run on a fresh install: the Welcome flow covers the chat; "Skip" closes it. Returns
    once either the Welcome flow or the chat home is on screen."""
    index = await d.wait_any({"text": "Skip"}, {"tooltip": ATTACH_TOOLTIP}, timeout=timeout)
    if index == 0:
        await d.tap(text="Skip")
        await d.wait(text="Skip", gone=True, timeout=30)
        return True
    return False


async def wait_home(d: Any, timeout: float = 180.0) -> None:
    await d.wait(tooltip=ATTACH_TOOLTIP, timeout=timeout)


async def go_home(d: Any, max_steps: int = 6) -> None:
    for _ in range(max_steps):
        if await d.count(tooltip="Open navigation"):
            return
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


# ---- flows --------------------------------------------------------------------------------------

async def smoke_navigation(d: Any) -> None:
    """Chat home → drawer → Library (shelves) → Settings home → About › Updates page is listed."""
    await wait_home(d)
    await d.wait(key="mode-chip", timeout=30)
    await open_drawer(d)
    await d.tap(key="dest-library")
    await d.wait(key="lib-search", timeout=60)
    await open_settings(d)
    await d.wait(key="hub-settings.import", timeout=30, scroll=True)
    await d.wait(key="hub-settings.logs", timeout=30, scroll=True)
    await d.wait(key="hub-settings.updates", timeout=30, scroll=True)


async def run_selftest(d: Any, timeout: float = 600.0) -> str:
    """Settings › Logs & diagnostics › Run self-test → the result card says PASS."""
    await open_settings(d)
    await d.tap(key="hub-settings.logs", timeout=30, scroll=True)
    await d.tap(text="Run self-test", timeout=30, scroll=True)
    index = await d.wait_any({"contains": "PASS · "}, {"contains": "FAIL · "}, timeout=timeout)
    assert index == 0, "the self-test reported FAIL"
    return "PASS"


async def import_desktop_config(d: Any, name: str = CONFIG_NAME) -> None:
    await open_settings(d)
    await d.tap(key="hub-settings.import", timeout=30, scroll=True)
    await d.pick_file(name, lambda: d.tap(key="import-pick-config"))
    await d.wait(text=name, timeout=30)  # the chosen file's name
    await d.tap(key="import-run", timeout=30, scroll=True)
    await d.wait(contains="Imported ", timeout=60)


async def chat_translate_and_migrate(d: Any, epub: str = EPUB_NAME, timeout: float = 900.0) -> None:
    """A new chat, ＋ › Files with the EPUB, Send; the job card reaches Done; Migrate."""
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
    await d.tap(text="Migrate", timeout=30, scroll=True)
    await d.wait(contains="migrated", timeout=60)


async def library_book_chapters(d: Any, title: str = EPUB_TITLE) -> None:
    """Library → Completed shelf → the migrated book → Chapters: none untranslated or failed."""
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
