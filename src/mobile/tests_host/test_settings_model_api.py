"""Settings › Model & API (owner 2026-11-18): the model row sits above the API key and is the model menu's
searchable picker; the API key is edited in place with the shared KeyField (animated Save, Copy), no
pop-up editor. Synthetic schema and store; no network."""

import asyncio

import test_mobile_settings as MS  # noqa: E402  (path setup, the fake schema / store / session helpers)

from glossarion_mobile.ui.settings import model  # noqa: E402

needs_flet = MS.needs_flet
FakeSpec, FakeSection = MS.FakeSpec, MS.FakeSection

SPECS = [
    FakeSpec("api_key", "str", "", section="main.model", label="API Key", tooltip="Your provider key."),
    FakeSpec("authgem_project", "str", "", section="main.model", label="Authgem project"),
    FakeSpec("custom_model_list", "str", "", section="main.model", label="Custom model list"),
    FakeSpec("model", "str", "gpt-4o", section="main.model", label="Model", env=("MODEL",)),
    FakeSpec("model_manager_hide_unpolled_models", "bool", False, section="main.model", label="Hide unpolled models"),
]
SECTIONS = [FakeSection("main.model", "Model & API", tuple(s.key for s in SPECS), "main")]


def _schema():
    return MS.SchemaAccess(MS.make_fake_schema(SPECS, SECTIONS))


def test_lead_keys_put_the_model_above_the_api_key():
    specs, headings = model.lead_keys_first("main.model", SPECS, {"model_manager_hide_unpolled_models": "Models"})
    assert [s.key for s in specs] == ["model", "api_key", "authgem_project", "custom_model_list",
                                      "model_manager_hide_unpolled_models"]
    assert headings["model"] == headings["api_key"] == "Model & API key"
    assert headings["model_manager_hide_unpolled_models"] == "Models"  # the other headings are kept
    # a section without lead keys is returned as it is
    same, kept = model.lead_keys_first("other.response", SPECS, {"a": "b"})
    assert same == SPECS and kept == {"a": "b"}


def _page(tmp_path, data, **ctx_kw):
    from glossarion_mobile.ui.settings.section_page import SectionPage

    conn, session = MS._fake_session("android")
    page = session.page
    schema = _schema()
    store = MS._memory_store(tmp_path, data, schema)
    ctx = MS._ctx(store, schema, page, **ctx_kw)
    screen = SectionPage(MS._match("/settings/s/main.model"), ctx)
    page.views[0].controls.append(screen.get_body())
    page.update()
    screen.did_show()
    screen.test_keepalive = (conn, session)  # the page holds its session weakly
    return page, store, ctx, screen


@needs_flet
def test_page_builds_the_model_row_before_the_api_key_row(tmp_path):
    async def scenario():
        page, store, _ctx, screen = _page(tmp_path, {"api_key": "sk-abcdefghijklmnop1234"})
        assert screen.keys[:2] == ["model", "api_key"]
        assert screen.headings["model"] == screen.headings["api_key"] == "Model & API key"
        rendered = [getattr(c, "content", None) for c in screen.list_view.controls]
        titles = [c.value for c in rendered if hasattr(c, "value")]
        assert titles[0] == "Model & API key"  # the first heading of the page
        order = [screen.list_view.controls.index(screen.tiles[k].control) for k in ("model", "api_key")]
        assert order[0] < order[1]
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_api_key_row_edits_in_place_with_save_and_copy(tmp_path):
    from glossarion_mobile.ui.screens.key_editor import KeyField
    from glossarion_mobile.ui.settings.tiles import SecretTile

    async def scenario():
        copied, navigated = [], []
        page, store, ctx, screen = _page(tmp_path, {"api_key": "sk-abcdefghijklmnop1234"},
                                         copy_text=copied.append, navigated=navigated)
        tile = screen.tiles["api_key"]
        assert isinstance(tile, SecretTile) and isinstance(tile.key_field, KeyField)
        field = tile.key_field
        assert field.field.password and field.field.can_reveal_password and field.value == "sk-abcdefghijklmnop1234"
        assert field.save_button.visible and not field.copy_button.disabled
        assert tile.storage_note.value.startswith("Stored encrypted in config.json")
        # a tap opens nothing: no editor screen, no navigation, no dialog
        shown = []
        ctx.show_dialog = shown.append
        page.show_dialog = shown.append
        assert tile.activate() is None and tile._on_tap() is None and tile.editor is None
        assert navigated == [] and shown == []
        await field._on_copy()
        assert copied == ["sk-abcdefghijklmnop1234"]

        # Save stores the key through the tile's store write (the same one the old editor's Save used)
        writes = []
        real_set = store.set

        def spy(path, value, *a, **kw):
            writes.append((tuple(path) if isinstance(path, (list, tuple)) else (path,), value))
            return real_set(path, value, *a, **kw)

        store.set = spy
        field.field.value = "sk-new-key-0000000000"
        assert await field.save()
        assert (("api_key",), "sk-new-key-0000000000") in writes and store.get("api_key") == "sk-new-key-0000000000"
        assert field.message.value == "Saved" and field.save_mark.content.key == "key-save-saved"
        assert tile.value_text.value == "sk-…0000" and tile.modified_dot.visible

        # a key the schema refuses is not stored, and the line under the field says why
        real_coerce = ctx.schema.coerce

        def refuse(key, value):
            if key == "api_key":
                raise ValueError("Keys never contain spaces")
            return real_coerce(key, value)

        ctx.schema.coerce = refuse
        field.field.value = "sk bad key"
        assert not await field.save()
        assert field.message.value == "Not saved: Keys never contain spaces"
        assert field.save_mark.content.key == "key-save-failed" and store.get("api_key") == "sk-new-key-0000000000"
        ctx.schema.coerce = real_coerce

        # an emptied field removes the key; reset elsewhere puts the default back into the field
        field.set_value("")
        assert not store.has("api_key")
        store.set("api_key", "sk-from-elsewhere-1234")
        screen._on_config_change("api_key")
        assert field.value == "sk-from-elsewhere-1234"
        assert tile.reset() and not store.has("api_key") and field.value == ""
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_api_key_field_keeps_a_key_this_device_cannot_decrypt(tmp_path):
    async def scenario():
        page, store, _ctx, screen = _page(tmp_path, {"api_key": "ENC:not-decryptable"})
        field = screen.tiles["api_key"].key_field
        assert field.encrypted and field.field.value == ""
        assert not await field.save()  # Save over the empty field would erase it
        assert store.get("api_key") == "ENC:not-decryptable"
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_model_row_opens_the_searchable_picker_and_a_pick_writes_the_model(tmp_path, monkeypatch):
    from glossarion_mobile.ui.settings import tiles
    from glossarion_mobile.ui.sheets.model_sheet import ModelPicker, ModelSheet

    async def scenario():
        page, store, _ctx, screen = _page(tmp_path, {"model": "gpt-4o"})
        tile = screen.tiles["model"]
        assert isinstance(tile, tiles.ModelTile) and isinstance(tile.picker, ModelPicker)
        assert tile.picker.field.read_only and tile.picker.value == "gpt-4o"
        writes = []
        real = tiles.write_setting
        monkeypatch.setattr(tiles, "write_setting", lambda st, key, value: (writes.append((key, value)),
                                                                             real(st, key, value))[1])
        sheet = tile.activate()  # tapping the row opens the model menu's searchable sheet in field mode
        assert isinstance(sheet, ModelSheet) and sheet.field_mode and hasattr(sheet, "search")
        sheet.set_query("claude")
        assert sheet.query == "claude"
        sheet.select("model", "claude-sonnet-5-5")
        assert writes == [("model", "claude-sonnet-5-5")]  # the model menu's write (setting_writes)
        assert store.get("model") == "claude-sonnet-5-5" and tile.picker.field.value == "claude-sonnet-5-5"
        # a typed id that is not in the catalog is used as typed
        tile.picker.open().select("model", "my-provider/custom-model-7")
        assert store.get("model") == "my-provider/custom-model-7"
        # a change made elsewhere shows in the picker
        store.set("model", "gemini-3-pro")
        screen._on_config_change("model")
        assert tile.picker.value == tile.picker.field.value == "gemini-3-pro"
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_model_row_reason_chip_follows_the_model(tmp_path):
    """The picker's "Not available on mobile" chip is rebuilt when the model changes (a pick, or a change
    made elsewhere), not left from the value the page was built with."""
    from glossarion_mobile.ui.components.reason_chip import ReasonChip

    def chips(picker):
        return [c for c in picker.content.controls if isinstance(c, ReasonChip)]

    async def scenario():
        page, store, _ctx, screen = _page(tmp_path, {"model": "ocz/qwen3"})  # ocz/ is excluded on mobile
        picker = screen.tiles["model"].picker
        assert len(chips(picker)) == 1 and picker.reason is chips(picker)[0]
        picker.open().select("model", "gpt-4o")  # a pick of a supported model drops the chip
        assert store.get("model") == "gpt-4o" and chips(picker) == [] and picker.reason is None
        store.set("model", "ocz/llama3")  # an excluded model set elsewhere shows it
        screen._on_config_change("model")
        assert picker.value == "ocz/llama3" and len(chips(picker)) == 1
        store.set("model", "gemini-3-pro")
        screen._on_config_change("model")
        assert chips(picker) == []
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_search_hits_open_the_moved_rows(tmp_path):
    async def scenario():
        page, store, _ctx, screen = _page(tmp_path, {})
        for key in ("model", "api_key"):
            assert await screen.focus_key(key, highlight=False, settle=0)
        assert screen.jumps == ["model", "api_key"]
        store._saver.close()

    asyncio.run(scenario())

