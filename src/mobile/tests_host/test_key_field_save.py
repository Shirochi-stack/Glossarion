"""KeyField's Save button (floppy disk), where a Paste button was.

The phone's keyboard and long-press already paste. The key is stored as it is typed, but the config file is
written a moment later (the store's debounce); Save writes it now and says "Saved" only after that, or
"Not saved: <reason>". It never stores the empty field shown over a key this device cannot read, and it is
absent where the form around the field has its own Save (the KeyEditor).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import flet as ft  # noqa: E402

from glossarion_mobile.ui.screens.key_editor import KeyField, store_saver  # noqa: E402


def _store(path: Path):
    from glossarion_mobile.state.config_store import MobileConfigStore

    store = MobileConfigStore(str(path), debounce=30)  # the debounced write never lands inside a test
    store.load()
    return store


def _icon(button: ft.IconButton):
    """The icon a button shows (the Save button's sits in its animated switcher)."""
    icon = button.icon
    return icon.content.icon if isinstance(icon, ft.AnimatedSwitcher) else icon


def _shown(field: KeyField) -> list:
    return [_icon(control) for control in field.controls[0].controls
            if isinstance(control, ft.IconButton) and control.visible]


def test_the_save_button_answers_the_tap_with_a_tick_then_the_floppy_disk_again(monkeypatch):
    """Owner 2026-11-17: "The save button lacks any animated feedback": the icon turns into a tick (an error
    mark when the key was not saved) through the button's animated switcher, then back."""
    monkeypatch.setattr(KeyField, "SAVE_MARK_SECONDS", 0.05)
    problems = [None, "disk full", None]
    field = KeyField(value="sk-old-12345678", on_save=lambda value: problems.pop(0))
    assert isinstance(field.save_button.icon, ft.AnimatedSwitcher)
    assert field.save_mark.transition == ft.AnimatedSwitcherTransition.SCALE
    keys = []

    async def scenario():
        assert _icon(field.save_button) == ft.Icons.SAVE
        await field.save()
        keys.append(field.save_mark.content.key)
        assert _icon(field.save_button) == ft.Icons.CHECK_CIRCLE
        await asyncio.sleep(0.15)
        keys.append(field.save_mark.content.key)
        assert _icon(field.save_button) == ft.Icons.SAVE          # back by itself
        await field.save()                                        # a save that failed
        keys.append(field.save_mark.content.key)
        assert _icon(field.save_button) == ft.Icons.ERROR_OUTLINE
        assert field.save_mark.content.color == ft.Colors.ERROR
        await asyncio.sleep(0.03)
        await field.save()                                        # tapped again before the mark went back
        assert _icon(field.save_button) == ft.Icons.CHECK_CIRCLE
        await asyncio.sleep(0.03)                                 # the first tap's timer is over: not this one's
        assert _icon(field.save_button) == ft.Icons.CHECK_CIRCLE
        await asyncio.sleep(0.1)
        assert _icon(field.save_button) == ft.Icons.SAVE

    asyncio.run(scenario())
    assert len(set(keys)) == 3                                    # a new key each time: the switcher animates


def test_the_save_mark_fades_under_reduce_motion():
    from glossarion_mobile.ui import motion

    motion.set_reduce_motion(True)
    try:
        field = KeyField(value="sk-old-12345678", on_save=lambda value: None)
        asyncio.run(field.save())
        assert field.save_mark.transition == ft.AnimatedSwitcherTransition.FADE
    finally:
        motion.set_reduce_motion(False)


def test_the_row_has_save_and_copy_and_no_paste_button():
    field = KeyField(value="sk-old-12345678", on_change=lambda value: None, on_save=lambda value: None,
                     copy_text=lambda text: None, read_clipboard=lambda: "sk-from-clipboard")
    assert _shown(field) == [ft.Icons.SAVE, ft.Icons.CONTENT_COPY]
    assert field.save_button.tooltip == "Save" and not field.save_button.disabled
    assert not hasattr(field, "paste_button") and not hasattr(field, "paste")


def test_a_field_inside_a_form_with_its_own_save_has_no_save_button():
    field = KeyField(value="sk-old-12345678", on_change=lambda value: None, copy_text=lambda text: None)
    assert _shown(field) == [ft.Icons.CONTENT_COPY]
    assert asyncio.run(field.save()) is False and not field.message.visible


def test_save_writes_the_config_file_before_it_says_saved(tmp_path):
    path = tmp_path / "config.json"
    store = _store(path)

    def put(value):
        return store.set("api_key", value)

    field = KeyField(value="", on_change=put, on_save=store_saver(put, store))
    field.field.value = "  sk-new-87654321 "
    field._on_change()                                    # typing stores it; the file waits for the debounce
    assert store.get("api_key") == "sk-new-87654321" and store.save_count == 0
    assert asyncio.run(field.save()) is True
    assert store.save_count == 1
    again = _store(path)
    assert again.get("api_key") == "sk-new-87654321"      # a new start of the app reads it back
    assert field.message.visible and field.message.value == "Saved"
    assert asyncio.run(field.save()) is True              # nothing new to write is still "Saved"
    assert store.save_count == 1 and field.message.value == "Saved"
    field._on_change()                                    # the next edit takes the note away
    assert not field.message.visible
    field.field.value = ""
    field._on_change()
    assert asyncio.run(field.save()) is True and field.message.value == "Saved (no key)"
    for each in (store, again):
        each._saver.close()


def test_save_says_why_when_the_key_was_not_saved():
    class Store:
        save_error = None

        def flush(self):
            self.save_error = "Saving config.json failed: OSError: disk full"
            return False

    stored = []
    field = KeyField(value="sk-old-12345678", on_save=store_saver(stored.append, Store()))
    assert asyncio.run(field.save()) is False             # the write failed
    assert stored == ["sk-old-12345678"]
    assert field.message.value == "Not saved: Saving config.json failed: OSError: disk full"
    assert field.message.color == ft.Colors.ERROR

    def refuse(value):
        raise OSError("this phone cannot encrypt")

    field = KeyField(value="sk-old-12345678", on_save=store_saver(refuse, Store()))
    assert asyncio.run(field.save()) is False             # the store refused the key
    assert field.message.value == "Not saved: this phone cannot encrypt"

    field = KeyField(value="sk-old-12345678", on_save=store_saver(lambda value: "not editable", Store()))
    assert asyncio.run(field.save()) is False             # the caller's own reason
    assert field.message.value == "Not saved: not editable"


def test_save_does_not_erase_a_key_this_device_cannot_read():
    stored = []
    field = KeyField(value="ENC:gAAAAAB-not-readable-here", on_change=stored.append, on_save=stored.append)
    assert field.encrypted and field.field.value == ""
    assert asyncio.run(field.save()) is False
    assert stored == [] and field.message.value == "Not saved: enter the key again"
    field.field.value = "sk-typed-12345678"
    field._on_change()
    assert asyncio.run(field.save()) is True
    assert stored == ["sk-typed-12345678", "sk-typed-12345678"] and field.message.value == "Saved"


def test_save_after_a_test_replaces_the_test_result():
    field = KeyField(value="sk-old-12345678", on_save=lambda value: None, on_test=lambda value: {"ok": True})
    asyncio.run(field.test())
    assert field.status_chip.visible
    assert asyncio.run(field.save()) is True
    assert not field.status_chip.visible and field.result is None and field.message.value == "Saved"
    asyncio.run(field.test())                             # a test message is not shown in Save's colour
    assert field.message.color == ft.Colors.ON_SURFACE_VARIANT


def test_poe_setup_does_not_say_saved_for_an_empty_cookie(tmp_path):
    from glossarion_mobile.ui.sheets.model_sheet import PoeSetupSheet, SheetEnv

    store = _store(tmp_path / "config.json")
    store.set("api_key", "p-b:old%3D%3D")
    sheet = PoeSetupSheet(env=SheetEnv(store=store), model="poe/claude")
    assert _shown(sheet.field)[0] == ft.Icons.SAVE
    sheet.field.field.value = ""
    sheet.field._on_change()
    assert asyncio.run(sheet.field.save()) is False       # an empty field leaves the API key as it is
    assert sheet.field.message.value == "Not saved: enter the cookie value"
    assert store.get("api_key") == "p-b:old%3D%3D"
    sheet.field.field.value = "new%3D%3D"
    sheet.field._on_change()
    assert asyncio.run(sheet.field.save()) is True
    assert store.get("api_key") == "p-b:new%3D%3D" and store.save_count == 1
    store._saver.close()
