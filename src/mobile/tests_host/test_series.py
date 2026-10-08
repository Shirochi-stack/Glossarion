"""Host tests for the optional, mobile-only Series (UI_SPEC §2.15, Appendix B; milestone U9).

* ``state.series``: ``mobile_series.json`` round trip (Appendix B shape, atomic, unreadable file
  tolerated), route-safe ids, defaults limited to the per-chat override keys, linked books + cover;
* layering on the REAL chat adapter (shared ``direct_text_store.ChatStore`` binding): Global ->
  Series defaults -> per-chat overrides through ``ChatStoreAdapter.overrides``; the chat's own
  overrides (``own_overrides``) are untouched and win; ``series_id`` lives in the chat sidecar
  (``direct_text_chats.mobile.json``), never in the desktop v2 history; scratch chats cannot join;
* the UI: drawer Series section / search group, chat ⋯ "Move to Series…", the long-press sheet row,
  Chat settings "Inherited from: Series <name>" + the Series section + series scope, the series
  page cards, the Library "Add to Series" and Series filter;
* ``SeriesFeature`` wired into a fake app end to end (no backend behaviour is added).

Run from src/mobile with the 3.13 venv (Flet installed)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_series.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
for entry in (str(APP_DIR), str(SRC_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from glossarion_mobile.state import series as S  # noqa: E402
from glossarion_mobile.state.chat_index import ChatSummary  # noqa: E402
from glossarion_mobile.state.chat_store_adapter import OVERRIDE_KEYS, SIDECAR_NAME, ChatStoreAdapter, ChatStoreBinding  # noqa: E402

try:
    import flet as ft  # noqa: F401

    HAS_FLET = True
except Exception:  # pragma: no cover - CI without Flet runs the pure tests only
    HAS_FLET = False

needs_flet = pytest.mark.skipif(not HAS_FLET, reason="flet is not installed")

# The real app on a fake Flet session (test_ui_foundations' harness, shared rather than copied).
_UF_SPEC = importlib.util.spec_from_file_location("_glossarion_uf_helpers_series",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
_UF = importlib.util.module_from_spec(_UF_SPEC)
_UF_SPEC.loader.exec_module(_UF)
storage = _UF.storage
app_env = _UF.app_env


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Nothing reaches the real Library, output root, HOME or config."""
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(tmp_path / "home"))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(tmp_path / "Output"))
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(tmp_path / "Library"))
    monkeypatch.setenv("GLOSSARION_DIRECT_TEXT_HISTORY", str(tmp_path / "direct_text_chats.json"))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    yield
    try:
        from glossarion_mobile.ui.sheets import chat_settings

        chat_settings.SERIES_HOOKS.update({"inherited_label": None, "section": None})
    except Exception:
        pass
    try:
        from glossarion_mobile.ui.chat import series_feature

        series_feature._CURRENT["feature"] = None
    except Exception:
        pass


# --------------------------------------------------------------------------- store
def _store(tmp_path, ids=None):
    counter = iter(ids or [f"{n:012x}" for n in range(1, 100)])
    clock = iter(range(1000, 2000))
    return S.SeriesStore(str(tmp_path / S.SERIES_FILE), clock=lambda: float(next(clock)), new_id=lambda: next(counter))


def test_store_round_trip_matches_appendix_b(tmp_path):
    store = _store(tmp_path)
    assert store.load() is False and store.all() == []
    novel = store.create("  My   novel  ", color="teal", book_ids=["aaaaaaaaaaaa", "bbbbbbbbbbbb", "aaaaaaaaaaaa"],
                         defaults={"model": "gemini-2.5-flash", "bogus": 1, "profile": None})
    other = store.create("Another")
    assert novel.name == "My novel" and novel.book_ids == ("aaaaaaaaaaaa", "bbbbbbbbbbbb")
    assert novel.cover_bid == "aaaaaaaaaaaa"                     # the first linked book
    assert novel.defaults == {"model": "gemini-2.5-flash"}       # override keys only, None dropped
    assert other.color != novel.color                            # a free swatch
    data = json.loads((tmp_path / S.SERIES_FILE).read_text(encoding="utf-8"))
    assert data["version"] == 1 and [s["id"] for s in data["series"]] == [novel.id, other.id]
    assert set(data["series"][0]) == {"id", "name", "color", "cover_bid", "book_ids", "defaults", "created_at"}
    again = _store(tmp_path)
    assert again.load() is True
    assert [s.as_dict() for s in again.all()] == [s.as_dict() for s in store.all()]
    assert [s.name for s in again.all()] == ["Another", "My novel"]   # by name


def test_store_edits_defaults_books_and_delete(tmp_path):
    store = _store(tmp_path)
    item = store.create("Saga")
    events = []
    store.subscribe(lambda: events.append("x"))
    assert store.update(item.id, name="Saga II", color="amber", cover_bid="cccccccccccc")
    assert store.update(item.id, name="Saga II") is False          # nothing changed
    assert store.get(item.id).name == "Saga II" and store.get(item.id).color == "amber"
    assert store.set_default(item.id, "target_language", "Korean")
    assert store.set_default(item.id, "glossary_override_mode", "manual")
    with pytest.raises(KeyError):
        store.set_default(item.id, "api_key", "sk-x")            # never a config key
    assert store.defaults(item.id) == {"target_language": "Korean", "glossary_override_mode": "manual"}
    assert store.set_default(item.id, "target_language", None)
    assert store.defaults(item.id) == {"glossary_override_mode": "manual"}
    assert store.link_books(item.id, ["dddddddddddd", "eeeeeeeeeeee"]) == 2
    assert store.link_books(item.id, ["dddddddddddd"]) == 0
    assert [s.id for s in store.series_for_book("eeeeeeeeeeee")] == [item.id]
    store.update(item.id, cover_bid="dddddddddddd")
    assert store.unlink_book(item.id, "dddddddddddd")
    assert store.get(item.id).cover_bid == "eeeeeeeeeeee"          # the cover follows the remaining books
    assert store.reset_defaults(item.id) and store.defaults(item.id) == {}
    assert store.delete(item.id) and store.get(item.id) is None and not store.delete(item.id)
    assert len(events) >= 8


def test_store_ignores_unreadable_files_and_bad_rows(tmp_path):
    path = tmp_path / S.SERIES_FILE
    path.write_text("{not json", encoding="utf-8")
    store = S.SeriesStore(str(path))
    assert store.load() is False and store.all() == []
    path.write_text(json.dumps({"version": 1, "series": [
        {"id": "../etc", "name": "bad id"},
        {"id": "ok_1", "name": "x" * 500, "color": "neon", "book_ids": ["a", 3, "a"], "defaults": {"model": "m", "x": 1}},
        "junk",
    ]}), encoding="utf-8")
    assert store.load() is True
    (only,) = store.all()
    assert only.id == "ok_1" and len(only.name) == S.MAX_NAME_CHARS and only.color == "rose"
    assert only.book_ids == ("a",) and only.defaults == {"model": "m"}


def test_ids_are_route_safe(tmp_path):
    from glossarion_mobile.ui.router import build_route, parse_route

    store = S.SeriesStore(str(tmp_path / S.SERIES_FILE))
    item = store.create("Route")
    route = build_route("series", {"sid": item.id})
    assert route == f"/series/{item.id}" and parse_route(route).name == "series"
    assert set(S.DEFAULT_KEYS) == set(OVERRIDE_KEYS)
    assert len(S.SERIES_COLORS) == 8 and all(h.startswith("#") for _n, h in S.SERIES_COLORS)


# --------------------------------------------------------------------------- the real chat adapter
def _history(root: Path) -> Path:
    history = root / "direct_text_chats.json"
    sessions = []
    for cid, title in ((2, "Volume one"), (3, "Volume two"), (7, "Unrelated")):
        sessions.append({"id": cid, "title": title, "messages": [["user", f"hello {title}"]], "draft": "",
                         "attachment": None, "output_folder": "", "output_folder_name": "",
                         "next_output_index": 1, "expanded": []})
    history.write_text(json.dumps({"version": 2, "current_chat_id": 2, "sessions": sessions}), encoding="utf-8")
    return history


@pytest.fixture
def chats(tmp_path):
    if not (SRC_DIR / "direct_text_store.py").is_file():
        pytest.skip("src/direct_text_store.py not present")
    import direct_text_store

    history = _history(tmp_path)
    binding = ChatStoreBinding(direct_text_store.ChatStore(str(history), output_root=str(tmp_path / "Output")),
                               history_path=str(history))
    adapter = ChatStoreAdapter(binding, history_path=str(history), scratch_dir=str(tmp_path / "scratch"))
    assert adapter.load(), adapter.load_error
    yield adapter
    adapter.close()


def test_layering_global_series_chat_on_the_real_adapter(chats, tmp_path):
    store = _store(tmp_path)
    saga = store.create("Saga", defaults={"model": "series-model", "target_language": "Korean",
                                          "disable_thinking": True})
    chats.override_layers.append(lambda cid: store.defaults(S.chat_series_id(chats, cid, store)))
    chats.set_override("2", "target_language", "Japanese")      # the chat's own value
    assert chats.overrides("2") == {"target_language": "Japanese"}   # not in a series yet
    assert S.move_chat(chats, "2", saga.id, store) and S.move_chat(chats, "3", saga.id, store)
    assert S.move_chat(chats, "2", saga.id, store) is False       # already there
    assert chats.overrides("2") == {"model": "series-model", "target_language": "Japanese", "disable_thinking": True}
    assert chats.own_overrides("2") == {"target_language": "Japanese"}
    assert chats.overrides("3") == saga.defaults and chats.own_overrides("3") == {}
    assert chats.overrides("7") == {}
    chats.reset_overrides("2")                                   # resets the chat's own values only
    assert chats.overrides("2") == saga.defaults
    store.set_default(saga.id, "model", None)
    assert "model" not in chats.overrides("3")
    # the chat's series lives in the mobile sidecar, never in the desktop v2 history
    chats.flush()
    sidecar = json.loads((tmp_path / SIDECAR_NAME).read_text(encoding="utf-8"))
    assert sidecar["chats"]["2"]["series_id"] == saga.id
    history = json.loads((tmp_path / "direct_text_chats.json").read_text(encoding="utf-8"))
    assert "series_id" not in json.dumps(history)
    # a deleted series reads as "no series" (its defaults no longer apply)
    store.delete(saga.id)
    assert S.chat_series_id(chats, "2", store) is None and chats.overrides("3") == {}


def test_scratch_chats_never_join_and_members_rows_search(chats, tmp_path):
    store = _store(tmp_path)
    saga = store.create("Saga")
    scratch = chats.new_scratch()
    assert S.move_chat(chats, scratch, saga.id, store) is False
    assert S.move_chat(chats, "2", "nope", store) is False       # unknown series
    S.move_chat(chats, "2", saga.id, store)
    S.move_chat(chats, "3", saga.id, store)
    assert [c.cid for c in S.member_chats(chats, saga.id, store)] == sorted(
        ["2", "3"], key=lambda cid: -chats.get(cid).updated_at)
    (row,) = S.series_rows(store, chats)
    assert row.sid == saga.id and row.count == 2 and row.color.startswith("#")
    # series-scoped search: a series by name (all its chats), or the matching chats of a series
    assert [r.count for r in S.series_search(store, chats, "saga")] == [2]
    hit = S.series_search(store, chats, "volume two")
    assert [(r.sid, [c.cid for c in r.chats]) for r in hit] == [(saga.id, ["3"])]
    assert S.series_search(store, chats, "unrelated") == []
    assert S.move_chat(chats, "2", None, store) and S.chat_series_id(chats, "2", store) is None


# --------------------------------------------------------------------------- fakes for UI tests
class FakeChats:
    """The chat adapter surface the Series UI uses (meta / overrides / summaries)."""

    def __init__(self, titles=("Volume one", "Volume two", "Unrelated")):
        self.rows = {str(i + 2): ChatSummary(cid=str(i + 2), title=t, updated_at=100.0 + i) for i, t in enumerate(titles)}
        self.metas: dict = {}
        self.own: dict = {}
        self.override_layers: list = []
        self.listeners: list = []
        self.created = 0

    def all(self):
        return list(self.rows.values())

    def get(self, cid):
        return self.rows.get(str(cid))

    def pinned(self):
        return [c for c in self.rows.values() if c.pinned]

    def recents(self, now):
        return [("Today", sorted(self.rows.values(), key=lambda c: -c.updated_at))]

    def search(self, query):
        return [c for c in self.rows.values() if query.lower() in c.title.lower()]

    def meta(self, cid):
        return dict(self.metas.get(str(cid), {}))

    def set_meta(self, cid, key, value):
        entry = self.metas.setdefault(str(cid), {})
        if value is None:
            entry.pop(key, None)
        else:
            entry[key] = value
        for listener in list(self.listeners):
            listener()

    def own_overrides(self, cid):
        return dict(self.own.get(str(cid), {}))

    def overrides(self, cid):
        merged = {}
        for layer in self.override_layers:
            merged.update(layer(cid) or {})
        merged.update(self.own_overrides(cid))
        return merged

    def set_override(self, cid, key, value):
        entry = self.own.setdefault(str(cid), {})
        if value is None:
            entry.pop(key, None)
        else:
            entry[key] = value

    def reset_overrides(self, cid):
        self.own.pop(str(cid), None)

    def subscribe(self, callback):
        self.listeners.append(callback)
        return lambda: self.listeners.remove(callback)

    def new_chat(self):
        self.created += 1
        cid = str(100 + self.created)
        self.rows[cid] = ChatSummary(cid=cid, title="New chat", updated_at=500.0)
        return cid


class FakePage:
    def __init__(self, width=400):
        self.dialogs: list = []
        self.width = width
        self.height = 800

    def show_dialog(self, dialog):
        dialog.open = True
        self.dialogs.append(dialog)

    def update(self):
        return None


class FakeConfig:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set_many(self, updates):
        self.values.update(updates)


# --------------------------------------------------------------------------- Chat settings
@needs_flet
def test_chat_settings_series_scope_edits_the_series_defaults(tmp_path):
    from glossarion_mobile.ui.sheets.chat_settings import ChatSettingsSheet

    store = _store(tmp_path)
    saga = store.create("Saga")
    sheet = ChatSettingsSheet(cid=saga.id, config=FakeConfig({"model": "global-model"}),
                              chats=S.SeriesDefaultsChats(store, saga.id), profiles=["Universal"],
                              languages=["English", "Korean"], subject="series")
    assert sheet.scope_button.segments[0].label == "This series" and sheet.title == "Series defaults"
    keys = {getattr(c, "key", None) for tile in sheet.sections.values() for c in tile.controls}
    assert "setting-skip_plan" not in keys and "setting-text_scale" not in keys and "series" not in sheet.sections
    assert sheet.value_of("model") == "global-model" and not sheet.is_overridden("model")
    sheet.set_value("target_language", "Korean")
    sheet.set_value("disable_thinking", True)
    sheet.set_value("glossary_override_mode", "no_glossary")
    assert store.defaults(saga.id) == {"target_language": "Korean", "disable_thinking": True,
                                       "glossary_override_mode": "no_glossary"}
    assert sheet.is_overridden("target_language")
    footer = sheet.body.controls[-1]
    assert footer.content == "Reset series defaults"
    sheet.reset("disable_thinking")
    assert "disable_thinking" not in store.defaults(saga.id)
    sheet.reset()
    assert store.defaults(saga.id) == {}
    # All chats scope still edits the global config (shared with the desktop)
    sheet.scope = "global"
    sheet.set_value("target_language", "French")
    assert sheet.config.values["output_language"] == "French"


@needs_flet
def test_chat_settings_shows_the_series_layer_and_section(tmp_path):
    from glossarion_mobile.ui.chat.series_feature import SeriesFeature
    from glossarion_mobile.ui.sheets.chat_settings import ChatSettingsSheet

    store = _store(tmp_path)
    chats = FakeChats()
    app = _fake_app(chats)
    feature = SeriesFeature(app, store=store)
    feature.attach()
    saga = store.create("Saga", defaults={"target_language": "Korean", "disable_thinking": True})
    S.move_chat(chats, "2", saga.id, store)
    chats.set_override("2", "disable_thinking", False)
    sheet = ChatSettingsSheet(cid="2", config=FakeConfig({"output_language": "English"}), chats=chats,
                              languages=["English", "Korean"])
    assert sheet.value_of("target_language") == "Korean"                     # the series layer
    assert not sheet.is_overridden("target_language")
    assert sheet.inherited_from("target_language") == "Series Saga"
    assert sheet.inherited_from("model") == "All chats"
    assert sheet.is_overridden("disable_thinking") and sheet.value_of("disable_thinking") is False
    texts = [getattr(c, "value", "") for c in sheet._badge("target_language")]
    assert texts == ["Inherited from: Series Saga"]
    tile = sheet.sections["series"]
    labels = [b.content for b in tile.controls[-1].controls]
    assert labels == ["Move to Series…", "Series page ›"]
    feature.close()
    sheet.rebuild()
    assert "series" not in sheet.sections and sheet.inherited_from("target_language") == "All chats"


# --------------------------------------------------------------------------- header / drawer / long-press
@needs_flet
def test_header_move_to_series_item():
    from glossarion_mobile.ui.chat.header import MENU_ITEMS, ChatHeader

    order = [a for a, _l in MENU_ITEMS]
    assert order.index("text_size") < order.index("move_series") < order.index("export")   # UI_SPEC §2.1
    calls = []
    header = ChatHeader(on_menu_action=lambda a: calls.append(("menu", a)))
    item = header.menu_items["move_series"]
    assert item.visible is False                                  # no Series feature
    header.set_series_handler(lambda: calls.append("move"))
    assert item.visible is True
    header._menu("move_series")
    header._menu("export")
    assert calls == ["move", ("menu", "export")]
    header.set_scratch(True)
    assert item.visible is False                                  # scratch chats never join
    header.set_scratch(False)
    header.set_series_handler(None)
    assert item.visible is False


def _drawer(chats):
    from glossarion_mobile.state.app_state import AppState
    from glossarion_mobile.ui.shell.drawer import ChatDrawer

    state = AppState(chats=chats)
    return ChatDrawer(state=state, clock=lambda: 200.0)


@needs_flet
def test_drawer_series_section_and_search(tmp_path):
    import flet as ft

    store = _store(tmp_path)
    chats = FakeChats()
    drawer = _drawer(chats)
    assert "Series" not in drawer.section_titles and [g for g, _l, _m in drawer.search_groups()][-1] != "series"
    feature_calls = []
    saga = store.create("Saga")

    class Provider:
        def rows(self):
            return S.series_rows(store, chats)

        def search(self, query):
            return S.series_search(store, chats, query)

        def open_series(self, sid):
            feature_calls.append(("open", sid))

        def new_chat_in_series(self, sid):
            feature_calls.append(("new", sid))

    drawer.series = Provider()
    drawer.refresh()
    assert "Series" in drawer.section_titles                      # one series exists
    tiles = [c for c in drawer.body.controls if isinstance(c, ft.ExpansionTile)]
    assert len(tiles) == 1 and tiles[0].title.value == "Saga" and tiles[0].trailing.value == "0"
    S.move_chat(chats, "2", saga.id, store)
    drawer.refresh()
    tile = next(c for c in drawer.body.controls if isinstance(c, ft.ExpansionTile))
    assert tile.trailing.value == "1"
    assert [c.key for c in tile.controls][0] == "chat-2"
    tile.controls[-2].on_click(None)                              # ＋ New chat in series
    tile.controls[-1].on_click(None)                              # Series page ›
    assert feature_calls == [("new", saga.id), ("open", saga.id)]
    # the series' chats leave Recents
    recent_keys = [getattr(c, "key", "") for c in drawer.body.controls if isinstance(c, ft.ListTile)]
    assert "chat-2" not in recent_keys and "chat-3" in recent_keys
    # expanded state survives a refresh (per-build keys)
    drawer._series_toggled(saga.id, types.SimpleNamespace(data="true"))
    drawer.refresh()
    tile2 = next(c for c in drawer.body.controls if isinstance(c, ft.ExpansionTile))
    assert tile2.expanded is True and tile2.key != tile.key
    # series-scoped search group
    assert drawer.search_groups()[-1][0] == "series"
    drawer.search_group = "series"
    drawer.query = "volume one"
    drawer.refresh()
    hits = [c for c in drawer.body.controls if isinstance(c, ft.ExpansionTile)]
    assert len(hits) == 1 and hits[0].expanded is True
    drawer.query = "nothing like this"
    drawer.refresh()
    assert not [c for c in drawer.body.controls if isinstance(c, ft.ExpansionTile)]


@needs_flet
def test_long_press_sheet_lists_move_to_series():
    from glossarion_mobile.ui.chat.integration import ChatFeature

    page = FakePage()
    rows = []
    fake = types.SimpleNamespace(
        app=types.SimpleNamespace(chat_view=object(), shell=None, navigate_to=None),
        page=page,
        chats=types.SimpleNamespace(set_pinned=lambda *a: None, can_delete=lambda cid: True),
        chat_action_providers=[lambda chat: rows.append(chat.cid) or [_action("Move to Series…")]],
        _select_chat=lambda cid: None,
    )
    fake._extra_chat_actions = lambda chat: ChatFeature._extra_chat_actions(fake, chat)
    chat = ChatSummary(cid="2", title="Volume one", updated_at=1.0)
    sheet = ChatFeature.chat_actions(fake, chat)
    labels = [item.label for item in sheet.items]
    assert labels[:3] == ["Rename", "Pin", "Move to Series…"] and rows == ["2"]
    scratch = ChatSummary(cid="sabc12345", title="Scratch", updated_at=1.0, scratch=True)
    assert "Move to Series…" not in [i.label for i in ChatFeature.chat_actions(fake, scratch).items]


def _action(label):
    from glossarion_mobile.ui.components.action_sheet import ActionItem

    return ActionItem(label, lambda: None)


# --------------------------------------------------------------------------- the feature end to end
def _fake_app(chats, *, page=None, library=None, glossary=None):
    from glossarion_mobile.ui.chat.header import ChatHeader
    from glossarion_mobile.ui.chat.integration import ChatFeature

    navigations = []
    opened = []
    notes = []
    header = ChatHeader()
    chat_view = types.SimpleNamespace(header=header, bound=True, cid="2", last_manual_glossary=None,
                                      env=types.SimpleNamespace(store=FakeConfig({"model": "global-model"}),
                                                                languages=["English", "Korean"]),
                                      applied=[])
    chat_view.apply_settings_changed = lambda: chat_view.applied.append(True)
    chat_view._profiles = lambda: ["Universal"]
    chat_feature = types.SimpleNamespace(chats=chats, chat_action_providers=[])
    chat_feature._extra = lambda chat: ChatFeature._extra_chat_actions(chat_feature, chat)
    drawer = _drawer(chats) if HAS_FLET else None
    shell = types.SimpleNamespace(screen_factory=lambda match: ("fallback", match.name), tablet=False, top_screen=None)
    app = types.SimpleNamespace(
        page=page or FakePage(), dispatcher=None, paths=None, chat_feature=chat_feature, chat_view=chat_view,
        drawer=drawer, shell=shell, state=types.SimpleNamespace(current_chat=_Signal("2")),
        navigate_to=lambda name, params=None, query=None, reset=False: navigations.append((name, params, reset)),
        _open_chat=lambda cid: opened.append(cid), notify=lambda message, *a: notes.append(message),
        library=library, glossary=glossary,
    )
    app.navigations, app.opened, app.notes = navigations, opened, notes
    return app


class _Signal:
    def __init__(self, value):
        self.value = value
        self.subscribers = []

    def subscribe(self, callback):
        self.subscribers.append(callback)
        return lambda: self.subscribers.remove(callback)

    def set(self, value):
        self.value = value
        for callback in list(self.subscribers):
            callback(value)


class FakeLibrary:
    def __init__(self):
        self.books = {"aaaaaaaaaaaa": {"name": "Book A", "output_folder": ""},
                      "bbbbbbbbbbbb": {"name": "Book B", "output_folder": ""}}
        self.covers = {}
        self.snapshot = types.SimpleNamespace(views={})

    def book_for_bid(self, bid):
        return self.books.get(bid)

    def bid_for(self, book):
        return next(bid for bid, b in self.books.items() if b is book or b == book)

    def card_badge(self, book):
        return ("EPUB", "1 MB")


@needs_flet
def test_series_feature_end_to_end(tmp_path):
    from glossarion_mobile.ui.chat import series_feature
    from glossarion_mobile.ui.chat.series_page import SeriesScreen, defaults_summary
    from glossarion_mobile.ui.chat.series_sheets import SeriesEditorDialog, SeriesPickerSheet
    from glossarion_mobile.ui.router import parse_route

    chats = FakeChats()
    library = FakeLibrary()
    app = _fake_app(chats, library=library)
    feature = series_feature.SeriesFeature(app, store=_store(tmp_path))
    feature.attach()
    assert series_feature.current() is feature and app.series is feature
    assert app.drawer.series is feature and app.chat_view.header.menu_items["move_series"].visible
    assert feature.chat_actions in app.chat_feature.chat_action_providers
    assert len(chats.override_layers) == 1
    # ⋯ › Move to Series… › New series… -> the editor creates it and moves the chat
    picker = feature.move_current_chat()
    assert isinstance(picker, SeriesPickerSheet) and picker.remove_tile is None
    picker._new()
    editor = app.page.dialogs[-1]
    assert isinstance(feature.dialog, SeriesEditorDialog) and editor is feature.dialog.dialog
    feature.dialog.name_field.value = "Saga"
    feature.dialog.set_color("violet")
    feature.dialog._save()
    (saga,) = feature.store.all()
    assert saga.name == "Saga" and saga.color == "violet"
    assert S.chat_series_id(chats, "2", feature.store) == saga.id and app.notes[-1] == "Moved to Saga"
    # series defaults flow into the chat's effective overrides (Global -> Series -> chat)
    feature.store.set_default(saga.id, "model", "series-model")
    assert chats.overrides("2") == {"model": "series-model"} and app.chat_view.applied
    assert feature.inherited_label(chats, "2", "model") == "Series Saga"
    # the long-press row and "Remove from series"
    (row,) = feature.chat_actions(ChatSummary(cid="2", title="Volume one", updated_at=1.0))
    assert row.label == "Move to Series…"
    picker = feature.move_chat_sheet("2")
    assert picker.remove_tile is not None and picker.tiles[saga.id].selected
    picker._pick(None)
    assert S.chat_series_id(chats, "2", feature.store) is None and chats.overrides("2") == {}
    # ＋ New chat in series: a new chat, in the series, opened
    cid = feature.new_chat_in_series(saga.id)
    assert S.chat_series_id(chats, cid, feature.store) == saga.id and app.opened[-1] == cid
    # Library: Add to Series links books; the filter reads them back
    sheet = feature.add_books_sheet(["aaaaaaaaaaaa", "bbbbbbbbbbbb"])
    sheet._pick(saga.id)
    assert feature.store.get(saga.id).book_ids == ("aaaaaaaaaaaa", "bbbbbbbbbbbb")
    assert app.notes[-1] == "Added 2 books to Saga"
    assert feature.book_ids(saga.id) == frozenset({"aaaaaaaaaaaa", "bbbbbbbbbbbb"}) and feature.book_ids(None) is None
    assert feature.choices() == [(saga.id, "Saga", S.color_hex("violet"))]
    # the series page
    match = parse_route(f"/series/{saga.id}")
    screen = app.shell.screen_factory(match)
    assert isinstance(screen, SeriesScreen) and screen.title == "Saga"
    screen.get_body()
    keys = [getattr(c, "key", "") for c in screen.holder.controls]
    books_card = screen.holder.controls[3]
    assert len(books_card.content.controls) == 4                  # title, the 2 book rows, Open Library
    assert [k.rsplit("-", 1)[0] for k in keys] == ["series-header", "series-defaults", "series-glossary",
                                                   "series-books", "series-chats"]
    assert defaults_summary(feature.store.defaults(saga.id)) == [("Model", "series-model")]
    defaults = feature.open_defaults_sheet(saga.id)
    assert defaults.subject == "series" and app.page.dialogs[-1] is defaults.dialog
    defaults.set_value("output_mode", "vision")
    assert feature.store.defaults(saga.id)["output_mode"] == "vision"
    # an unknown series falls back to the "no longer available" state, other routes fall through
    missing = app.shell.screen_factory(parse_route("/series/abcdefabcdef"))
    missing.get_body()
    assert missing.holder.controls[0].title == "This series is no longer available"
    assert app.shell.screen_factory(parse_route("/library")) == ("fallback", "library")
    # delete: the chats stay and leave the series
    assert feature.delete_series(saga.id)
    assert feature.store.all() == [] and S.chat_series_id(chats, cid) is None
    screen.refresh()
    assert screen.holder.controls[0].title == "This series is no longer available"
    feature.close()
    assert series_feature.current() is None and chats.override_layers == [] and app.drawer.series is None


@needs_flet
def test_scratch_chat_cannot_move(tmp_path):
    from glossarion_mobile.ui.chat import series_feature

    chats = FakeChats()
    chats.rows["sdeadbeef00"] = ChatSummary(cid="sdeadbeef00", title="Scratch", updated_at=1.0, scratch=True)
    app = _fake_app(chats)
    feature = series_feature.SeriesFeature(app, store=_store(tmp_path))
    feature.attach()
    assert feature.move_chat_sheet("sdeadbeef00") is None and "scratch" in app.notes[-1]
    assert feature.chat_actions(chats.rows["sdeadbeef00"]) == []
    feature.close()


@needs_flet
def test_series_glossary_prefills_the_manual_glossary_sheet(tmp_path):
    from glossarion_mobile.ui.chat import series_feature

    chats = FakeChats()
    app = _fake_app(chats)
    feature = series_feature.SeriesFeature(app, store=_store(tmp_path))
    feature.attach()
    glossary = tmp_path / "saga_glossary.csv"
    glossary.write_text("type,raw_name,translated_name\ncharacter,김,Kim\n", encoding="utf-8")
    saga = feature.store.create("Saga")
    S.move_chat(chats, "2", saga.id, feature.store)
    feature.store.set_default(saga.id, "glossary_override_mode", "manual")
    feature.store.set_default(saga.id, "manual_glossary_path", str(glossary))
    last = app.chat_view.last_manual_glossary
    assert last is not None and last.path == str(glossary) and last.extension == ".csv"
    feature.clear_glossary(saga.id)
    assert feature.store.defaults(saga.id) == {}
    assert app.chat_view.last_manual_glossary is None  # Series page › Clear: the cleared file is not offered
    feature.close()


def _series_glossaries(feature, tmp_path, names=("A", "B")):
    """Series named "Series <name>", each in Force Manual Glossary with its own CSV: {name: (series, path)}."""
    made = {}
    for name in names:
        path = tmp_path / f"Series{name}_glossary.csv"
        path.write_text("type,raw_name,translated_name\ncharacter,김,Kim\n", encoding="utf-8")
        item = feature.store.create(f"Series {name}")
        feature.store.set_default(item.id, "glossary_override_mode", "manual")
        feature.store.set_default(item.id, "manual_glossary_path", str(path))
        made[name] = (item, str(path))
    return made


@needs_flet
def test_series_glossary_prefill_follows_the_chat_switch(tmp_path):
    """The prefill belongs to the chat shown: a chat of another series gets that series' file, a chat
    without a series glossary an empty sheet; a file picked in the sheet is never replaced."""
    from glossarion_mobile.ui.chat import series_feature
    from glossarion_mobile.ui.chat.direct_text_rules import ManualGlossarySource

    chats = FakeChats()
    app = _fake_app(chats)
    feature = series_feature.SeriesFeature(app, store=_store(tmp_path))
    feature.attach()
    made = _series_glossaries(feature, tmp_path)
    S.move_chat(chats, "2", made["A"][0].id, feature.store)
    S.move_chat(chats, "3", made["B"][0].id, feature.store)
    view = app.chat_view

    def switch(cid):
        view.cid = cid
        app.state.current_chat.set(cid)
        last = view.last_manual_glossary
        return last.path if last is not None else None

    assert switch("2") == made["A"][1]
    assert switch("3") == made["B"][1]  # Series B's chat: not Series A's file
    assert switch("4") is None          # no series: the sheet starts empty
    assert switch("2") == made["A"][1]
    mine = str(tmp_path / "mine.csv")
    view.last_manual_glossary = ManualGlossarySource("path", path=mine, extension=".csv")  # picked in the sheet
    assert switch("3") == mine and switch("4") == mine
    view.last_manual_glossary = None
    assert switch("3") == made["B"][1]
    feature.clear_glossary(made["B"][0].id)
    assert view.last_manual_glossary is None
    feature.close()


@needs_flet
def test_series_glossary_sheet_on_the_running_app(app_env, tmp_path):
    """On the real app: Send's "Provide Manual Glossary" sheet in a Series B chat opened after a Series A
    chat loads Series B's file, and in a chat outside any series no series file."""
    from glossarion_mobile.ui.chat.series_feature import SeriesFeature

    async def scenario():
        _m, conn, session, page, app = await _UF._start("android", width=412)
        installed = None
        try:
            assert await _UF._wait(lambda: app.chat_view.bound, timeout=30)
            feature = getattr(app, "series", None)
            if feature is None:
                feature = installed = await SeriesFeature.install(app)
            chats = feature.chats
            made = _series_glossaries(feature, tmp_path)
            first = str(app.state.current_chat.value)
            chats.record_user_turn(first, ("user", "first chat text"), "first chat text")
            feature._move(first, made["A"][0].id)
            second = str(chats.new_chat())
            chats.record_user_turn(second, ("user", "second chat text"), "second chat text")
            feature._move(second, made["B"][0].id)
            plain = str(chats.new_chat())
            view = app.chat_view

            def sheet_file():
                sheet = view.open_manual_glossary(lambda source: None)
                loaded = os.path.basename(sheet.source_path) if sheet.source_path else None
                sheet.close()
                return loaded

            app._open_chat(first)
            assert await _UF._wait(lambda: view.cid == first)
            assert sheet_file() == "SeriesA_glossary.csv"
            app._open_chat(second)
            assert await _UF._wait(lambda: view.cid == second)
            assert sheet_file() == "SeriesB_glossary.csv"
            app._open_chat(plain)
            assert await _UF._wait(lambda: view.cid == plain)
            assert sheet_file() is None
        finally:
            if installed is not None:
                installed.close()
            await _UF._stop(app)

    asyncio.run(scenario())


@needs_flet
def test_library_filter_sheet_series_chips():
    from glossarion_mobile.ui.library.filter_sheet import FilterSheet
    from glossarion_mobile.ui.library.models import FilterState

    changes = []
    empty = FilterSheet(FilterState(), on_change=lambda n, v: changes.append((n, v)))
    assert empty.series_chips == {}
    sheet = FilterSheet(FilterState(), series=[("s1", "Saga", "#6C63FF"), ("s2", "Other", "#3BA99C")],
                        series_selected="s2", on_change=lambda n, v: changes.append((n, v)))
    assert list(sheet.series_chips) == ["", "s1", "s2"] and sheet.series_chips["s2"].selected
    sheet._on_series("s1")
    assert changes[-1] == ("series", "s1") and sheet.series_chips["s1"].selected and not sheet.series_chips[""].selected
    sheet._on_series(None)
    assert changes[-1] == ("series", None) and sheet.series_chips[""].selected
    stale = FilterSheet(FilterState(), series=[("s1", "Saga", "#6C63FF")], series_selected="gone")
    assert stale.series_selected is None


def test_no_series_placeholders_left():
    """The U9 placeholders are wired (nothing says Series "arrives in U9" any more)."""
    app = APP_DIR / "glossarion_mobile"
    offenders = []
    for path in app.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "series" in line.lower() and ("arrive" in line.lower() and "U9" in line):
                offenders.append(f"{path.relative_to(app)}: {line.strip()}")
    assert not offenders, offenders


def test_series_modules_are_python_310_and_flet_free_where_pure():
    import ast

    for rel in ("state/series.py", "ui/chat/series_feature.py", "ui/chat/series_page.py", "ui/chat/series_sheets.py"):
        data = (APP_DIR / "glossarion_mobile" / rel).read_bytes()
        ast.parse(data.decode("utf-8"), feature_version=(3, 10))
        assert data.count(b"\r\n") in (0, data.count(b"\n"))
    tree = ast.parse((APP_DIR / "glossarion_mobile" / "state" / "series.py").read_text(encoding="utf-8"))
    imported = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(name.startswith("flet") for name in imported)


@needs_flet
def test_series_in_the_running_app(app_env):
    """SeriesFeature installed into the real app (fake Flet session): layering reaches the header,
    the drawer shows the section, the chat ⋯ menu the item, and /series/<sid> mounts its page."""
    from glossarion_mobile.ui.chat.series_feature import SeriesFeature
    from glossarion_mobile.ui.chat.series_page import SeriesScreen

    async def scenario():
        _m, conn, session, page, app = await _UF._start("android", width=412)
        feature = None
        try:
            feature = await SeriesFeature.install(app)
            assert app.series is feature and app.chat_view.header.menu_items["move_series"].visible
            cid = str(app.state.current_chat.value)
            saga = feature.store.create("Saga", defaults={"target_language": "Korean"})
            feature._move(cid, saga.id)
            assert await _UF._wait(lambda: app.state.chat_context.value.target_language == "Korean")
            assert app.state.chat_context.value.custom                  # the "custom" chip covers series defaults
            assert await _UF._wait(lambda: "Series" in app.drawer.section_titles)
            await app.navigate(f"/series/{saga.id}")
            assert isinstance(app.shell.top_screen, SeriesScreen) and app.shell.current_route == f"/series/{saga.id}"
            screen = app.shell.top_screen
            assert screen.holder.controls and "series-header" in screen.holder.controls[0].key
            feature.store.update(saga.id, name="Saga II")
            assert await _UF._wait(lambda: screen.title == "Saga II")
            sheet = feature.open_defaults_sheet(saga.id)
            assert sheet is not None and sheet.subject == "series"
            sheet.set_value("model", "series-model")
            assert await _UF._wait(lambda: app.state.chat_context.value.model == "series-model")
            sheet.close()
            assert feature.delete_series(saga.id)
            # the chat leaves the series: no inherited values, no Series section
            assert app.chat_feature.chats.overrides(cid) == {}
            assert await _UF._wait(lambda: "Series" not in app.drawer.section_titles)
        finally:
            if feature is not None:
                feature.close()
            await _UF._stop(app)

    asyncio.run(scenario())
