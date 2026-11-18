"""Settings layout (owner 2026-11-18): Manga Translator in tabs (the desktop manga dialog's tabs), the
"Show debug buttons" switch directly in Settings › Data (no one-row Debug Controls page), and no
"Not available on mobile" rows anywhere in the settings lists or in Logs & diagnostics. The schema is the
shared generated one (code, no config.json); stores are temporary; no network."""

import asyncio
import types

import pytest

import test_mobile_settings as MS  # noqa: E402  (path setup, the fake session / store helpers)

needs_flet = MS.needs_flet
needs_schema = pytest.mark.skipif(not MS._importable("settings_schema"), reason="settings_schema is not importable")

MANGA = "manga.settings"
DEBUG_KEY = "show_debug_buttons"
NOT_ON_MOBILE = "Not available on mobile"


def _walk(control):
    """Every control under ``control`` (content / controls / title / subtitle / trailing / leading)."""
    stack, seen = [control], set()
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        for attr in ("content", "controls", "title", "subtitle", "trailing", "leading", "label"):
            child = getattr(node, attr, None)
            if isinstance(child, (list, tuple)):
                stack.extend(child)
            elif child is not None and not isinstance(child, (str, int, float)):
                stack.append(child)


def _texts(control) -> list:
    import flet as ft

    out = []
    for node in _walk(control):
        if isinstance(node, ft.Text) and node.value:
            out.append(str(node.value))
        reason = getattr(node, "reason", None)
        if isinstance(reason, str) and reason:
            out.append(reason)
    return out


def _keys(control) -> set:
    return {str(node.key) for node in _walk(control) if getattr(node, "key", None) is not None}


def _setup(tmp_path, data=None, *, with_page=False):
    """``with_page`` (inside a running loop only): a fake Flet session page, so pushes really diff."""
    session = conn = None
    if with_page:
        conn, session = MS._fake_session("android")
    schema = MS.SchemaAccess()
    store = MS._memory_store(tmp_path, data or {}, schema)
    navigated: list = []
    ctx = MS._ctx(store, schema, session.page if session is not None else None, navigated=navigated)
    ctx.test_keepalive = (conn, session)  # the page holds its session weakly
    return ctx.page, schema, store, ctx, navigated


def _manga_page(ctx, route=f"/settings/s/{MANGA}"):
    from glossarion_mobile.ui.settings.section_page import SectionPage

    screen = SectionPage(MS._match(route), ctx)
    screen.get_body()
    return screen


# ---- 1. Manga Translator tabs ------------------------------------------------------------------------


def test_manga_layout_puts_every_setting_in_exactly_one_tab():
    from glossarion_mobile.ui.settings.manga_layout import MANGA_TABS, manga_settings_layout

    specs = [types.SimpleNamespace(key=k, group=g) for k, g in (
        ("manga_settings.preprocessing.enabled", "Preprocessing"),
        ("manga_settings.ocr.provider", "OCR"),
        ("manga_settings.ocr.bubble_confidence", "OCR"),
        ("manga_settings.cloud_inference_steps", "Replicate API"),
        ("manga_bg_opacity", ""),  # the main panel: no dialog tab
        ("manga_full_page_context", ""),
        ("manga_settings.compression.enabled", "Preprocessing"),
        ("something_new", ""),
    )]
    ordered, headings, tabs = manga_settings_layout(specs)
    assert sorted(s.key for s in ordered) == sorted(s.key for s in specs)  # none lost, none twice
    labels = [label for label, _s, _e in tabs]
    assert labels == ["Preprocessing", "Replicate", "OCR", "Rendering", "General"]  # MANGA_TABS order, no empty tab
    assert [label for label, _d in MANGA_TABS][:6] == ["Preprocessing", "Inpainting", "Replicate", "Manual edit",
                                                      "OCR", "Advanced"]  # the desktop dialog's tab order
    assert all(len(label) <= 13 for label, _d in MANGA_TABS)  # phone-sized labels
    spans = {label: [s.key for s in ordered[start:end]] for label, start, end in tabs}
    assert spans["OCR"] == ["manga_settings.ocr.provider", "manga_settings.ocr.bubble_confidence"]
    assert headings["manga_settings.ocr.bubble_confidence"] == "Bubble detection"  # sub-headings inside a tab
    assert spans["General"] == ["manga_full_page_context", "something_new"]
    assert headings["manga_settings.cloud_inference_steps"] == ""  # one group in the tab: no heading


@needs_flet
@needs_schema
def test_manga_translator_page_has_the_desktop_tabs_and_lists_every_visible_setting(tmp_path):
    from glossarion_mobile.ui.settings.integration import IMPLEMENTED_ROUTES
    from glossarion_mobile.ui.settings.settings_home import SettingsHome
    from glossarion_mobile.ui.settings.tiles import listed_setting

    page, schema, store, ctx, _nav = _setup(tmp_path)
    screen = _manga_page(ctx)
    assert screen.tabs is not None and screen.tabs.key == "manga-settings-tabs"
    labels = [label for label, _s, _e in screen.tab_ranges]
    assert labels[:6] == ["Preprocessing", "Inpainting", "Replicate", "Manual edit", "OCR", "Advanced"]
    bar = screen.tabs.content.controls[0]
    assert bar.scrollable and [t.label for t in bar.tabs] == labels
    in_tabs = [k for _label, start, end in screen.tab_ranges for k in screen.keys[start:end]]
    assert len(in_tabs) == len(set(in_tabs)) == len(screen.keys)  # every setting in exactly one tab
    visible = [k for k in schema.section(MANGA).keys
               if schema.spec(k) is not None and listed_setting(ctx, schema.spec(k))]
    assert sorted(in_tabs) == sorted(visible) and all(end > start for _l, start, end in screen.tab_ranges)
    # the Settings home counts the same settings ("N settings")
    home = SettingsHome(MS._match("/settings"), ctx, implemented_routes=IMPLEMENTED_ROUTES)
    home.get_body()
    assert home.section_tiles[MANGA].subtitle.value.startswith(f"{len(in_tabs)} settings")
    # switching tabs shows that tab's settings and remembers it while the page is open
    asyncio.run(bar.on_click(types.SimpleNamespace(data=labels.index("OCR"))))
    _l, start, end = screen.tab_ranges[labels.index("OCR")]
    assert screen.tabs.selected_index == labels.index("OCR") and screen.window == (start, end)
    screen.refresh_all()
    assert screen.tabs.selected_index == labels.index("OCR")
    assert not store.keys()  # rendering writes nothing
    store._saver.close()


@needs_flet
@needs_schema
def test_manga_search_opens_the_page_on_the_tab_of_the_setting(tmp_path):
    from glossarion_mobile.ui.settings.search import find_settings

    page, schema, store, ctx, _nav = _setup(tmp_path)
    key = "manga_settings.ocr.bubble_confidence"
    hits = find_settings(ctx, "bubble confidence")
    hit = next(h for h in hits if h.key == key)
    assert hit.section_id == MANGA
    screen = _manga_page(ctx, f"/settings/s/{MANGA}#{key}")
    labels = [label for label, _s, _e in screen.tab_ranges]
    assert screen.tabs.selected_index == labels.index("OCR") and key in screen.visible_keys
    # a jump from another tab (the page's own search) switches to the setting's tab
    other = screen.keys[screen.tab_ranges[labels.index("Rendering")][1]]
    assert asyncio.run(screen.focus_key(other, highlight=False))
    assert screen.tabs.selected_index == labels.index("Rendering") and other in screen.visible_keys
    assert asyncio.run(screen.focus_key(key, highlight=False))
    assert screen.tabs.selected_index == labels.index("OCR") and key in screen.visible_keys
    store._saver.close()


# ---- 2. Show debug buttons directly in Settings › Data ---------------------------------------------------


@needs_flet
@needs_schema
def test_debug_switch_sits_in_the_data_section_with_no_page_of_its_own(tmp_path):
    from glossarion_mobile.ui.router import build_route
    from glossarion_mobile.ui.settings.integration import IMPLEMENTED_ROUTES
    from glossarion_mobile.ui.settings.search import find_settings
    from glossarion_mobile.ui.settings.settings_home import SettingsHome

    async def scenario():
        page, schema, store, ctx, navigated = _setup(tmp_path, with_page=True)
        home = SettingsHome(MS._match("/settings"), ctx, implemented_routes=IMPLEMENTED_ROUTES)
        page.views[0].controls.append(home.get_body())
        page.update()
        home.did_show()
        assert "other.debug" not in home.section_tiles  # no "Debug Controls · 1 setting" row
        cards = {card.key: card for card in home.group_cards()}
        data = cards["settings-group-Data"]
        tile = home.inline_tiles[DEBUG_KEY]
        assert tile.control in data.children and tile.label == "Show debug buttons"
        assert tile.info_button is not None and "settings-section-other.debug" not in _keys(data)
        texts = _texts(data)
        assert "Show debug buttons" in texts and not any("Debug Controls" in t for t in texts)
        # toggling it writes the same setting the page wrote
        tile.switch.value = True
        tile._on_switch(types.SimpleNamespace(control=tile.switch))
        assert store.get(DEBUG_KEY) is True
        home.refresh()  # a config change refreshes the switch in place
        assert tile.switch.value is True
        # search still finds it and opens the home on it
        hit = next(h for h in find_settings(ctx, "debug buttons") if h.key == DEBUG_KEY)
        home.search.field.value = "debug buttons"
        home.search.set_query("debug buttons")
        assert home.search.active
        assert home.open_hit(hit) == build_route("settings")
        assert not home.search.active and home.search.field.value == "" and data in home.content.controls
        await asyncio.sleep(0.1)
        assert tile.highlighted
        # other screens' searches (a section page's sheet, the chat's /settings) open the home, not a page
        assert ctx.open_setting("other.debug", DEBUG_KEY) == "/settings" and navigated[-1] == "/settings"
        home.dispose()
        store._saver.close()

    asyncio.run(scenario())


@needs_flet
@needs_schema
def test_diagnostics_shows_the_debug_switch_and_no_unavailable_rows(tmp_path):
    from glossarion_mobile.state.app_state import AppState
    from glossarion_mobile.ui.screens.diagnostics import DiagnosticsScreen

    class Runner:
        current_suite = None

        async def run(self, suite, *, source="button"):
            return {"suite": suite}

    page, schema, store, ctx, navigated = _setup(tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "data").mkdir()
    paths = types.SimpleNamespace(logs=str(tmp_path / "logs"), data=str(tmp_path / "data"))
    screen = DiagnosticsScreen(None, page=None, state=AppState(), dispatcher=None, runner=Runner(), paths=paths,
                               config_snapshot=store.snapshot, settings_ctx=ctx)
    body = screen.get_body()
    texts, keys = _texts(body), _keys(body)
    assert NOT_ON_MOBILE not in texts and "Process priority" not in texts and "CPU affinity" not in texts
    assert "diag-priority" not in keys and "diag-affinity" not in keys and "diag-debug-mode" not in keys
    tile = screen.debug_tile
    assert tile is not None and "Show debug buttons" in texts
    tile.switch.value = True
    tile._on_switch(types.SimpleNamespace(control=tile.switch))
    assert store.get(DEBUG_KEY) is True and navigated == []  # the switch itself, no page opened
    # without the settings context (older callers) the card simply has no switch
    bare = DiagnosticsScreen(None, page=None, state=AppState(), dispatcher=None, runner=Runner(), paths=paths)
    assert NOT_ON_MOBILE not in _texts(bare.get_body()) and bare.debug_tile is None
    store._saver.close()


# ---- 3. no "Not available on mobile" rows ----------------------------------------------------------------


@needs_flet
@needs_schema
def test_no_settings_row_shows_the_not_available_reason_and_counts_exclude_them(tmp_path):
    from glossarion_mobile.ui.settings.integration import IMPLEMENTED_ROUTES
    from glossarion_mobile.ui.settings.search import FILTERS, find_settings
    from glossarion_mobile.ui.settings.section_page import SectionPage
    from glossarion_mobile.ui.settings.settings_home import SettingsHome

    page, schema, store, ctx, _nav = _setup(tmp_path)
    unavailable = {str(spec.key) for spec in schema.all_specs() if not schema.availability(str(spec.key))[0]}
    assert unavailable  # the schema has desktop-only settings (Show CMD, Brush and eraser, Tor, ...)
    home = SettingsHome(MS._match("/settings"), ctx, implemented_routes=IMPLEMENTED_ROUTES)
    home.get_body()
    for section in schema.sections():
        screen = SectionPage(MS._match(f"/settings/s/{section.id}"), ctx, window_size=400)
        body = screen.get_body()
        for _title, start, _end in screen.tab_ranges:
            screen.render_window(start, push=False)
        assert not unavailable & set(screen.keys), section.id
        assert all(tile.available for tile in screen.tiles.values()), section.id
        assert NOT_ON_MOBILE not in _texts(body), section.id
        tile = home.section_tiles.get(section.id)
        if tile is not None:  # "N settings" counts only the rows the page shows
            count = len(screen.keys) - sum(1 for k in screen.keys if getattr(schema.spec(k), "virtual", ""))
            listed = len(home._listed_keys(section))
            assert not unavailable & set(home._listed_keys(section)) and listed >= count, section.id
    texts = [t for card in home.group_cards() for t in _texts(card)]
    assert NOT_ON_MOBILE not in texts
    # search and the filter chips never list them, even when stored and modified
    key = next(k for k in sorted(unavailable) if schema.spec(k) is not None and schema.spec(k).type == "bool"
               and len(schema.path_of(k)) == 1)
    store.set((key,), not bool(schema.effective_default(key)))
    assert all(h.key != key for h in find_settings(ctx, key))
    for fid, _label in FILTERS:
        assert all(h.key not in unavailable for h in find_settings(ctx, "", [fid])), fid
    assert store.get(key) is not None  # its stored value is kept untouched
    store._saver.close()


@needs_flet
@needs_schema
def test_tool_screens_and_manga_settings_skip_unavailable_tiles(tmp_path):
    from glossarion_mobile.ui.tools.common import schema_tiles
    from glossarion_mobile.ui.settings.manga_layout import group_manga_keys

    page, schema, store, ctx, _nav = _setup(tmp_path)
    tools_ctx = types.SimpleNamespace(settings=ctx)
    manga_unavailable = [k for k in schema.section(MANGA).keys if not schema.availability(k)[0]]
    assert manga_unavailable  # Brush and eraser, torch-only options, ...
    controls, tiles = schema_tiles(tools_ctx, [*manga_unavailable, "manga_bg_opacity"])
    assert list(tiles) == ["manga_bg_opacity"] and len(controls) == 1

    from glossarion_mobile.ui.tools.manga.settings import SettingsTab

    tab = SettingsTab.__new__(SettingsTab)
    tab.ctx = tools_ctx
    keys = tab._section_keys()
    assert keys and not set(manga_unavailable) & set(keys)
    assert sum(len(k) for _g, _t, k in group_manga_keys(keys)) == len(keys)
    store._saver.close()


@needs_flet
def test_appearance_has_no_not_on_mobile_row():
    from glossarion_mobile.ui.router import parse_route
    from glossarion_mobile.ui.screens.appearance import AppearanceScreen

    ctx = types.SimpleNamespace(page=None, prefs=None)
    texts = _texts(AppearanceScreen(parse_route("/settings/appearance"), ctx).get_body())
    assert "Text size 85%" in texts and "Auto DPI / GUI scale" not in texts and "Not on mobile" not in texts
