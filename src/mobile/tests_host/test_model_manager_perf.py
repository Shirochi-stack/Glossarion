"""Issue 19: the Model Manager with a desktop-sized model list (3,776 rows) stays responsive.

Run from src/mobile (Flet venv):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_model_manager_perf.py

The owner's desktop config carries ``custom_model_list`` = 3,760 ids (Settings › Import from desktop brings
it to the phone), so the manager opened with 3,776 rows: one 4.4 MB first patch of 53k controls, a 2-3 s
blocked loop on the host, and every keystroke / chip / swipe / poll re-sent or re-diffed every row.

The budgets here are deterministic: bytes and controls counted per message at send time on the fake Flet
session, render counts, mounted rows. Wall-clock values only appear as generous secondary bounds.
"""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import importlib.util
import json
import os
import threading
import time
import types
from pathlib import Path

_MK_SPEC = importlib.util.spec_from_file_location("_glossarion_mk_helpers_mmperf", Path(__file__).with_name("test_models_keys.py"))
MK = importlib.util.module_from_spec(_MK_SPEC)
_MK_SPEC.loader.exec_module(MK)

mc = MK.mc
parse_route = MK.parse_route
needs_flet = MK.needs_flet
_TB = MK._TB
_fake_session = MK._fake_session
app_env = MK.app_env
storage = MK.storage

# budgets (UI_SPEC §7.3): a first patch of the whole screen, and every later window-sized change
FIRST_PATCH_BYTES = 150_000
FIRST_PATCH_CONTROLS = 2_000
#: the real shell's route patch also carries the Settings home View below and the app bar
SHELL_FIRST_PATCH_BYTES = 200_000
SHELL_FIRST_PATCH_CONTROLS = 2_500
WINDOW_PATCH_BYTES = 150_000
SMALL_PATCH_BYTES = 5_000
#: secondary wall-clock bound on the loop's largest heartbeat gap (the issue-19 acceptance test's LOOP_GAP_S)
LOOP_GAP_S = 1.0

PREFIXES = ("", "openrouter/", "nanogpt/", "or/", "electronhub/", "groq/", "together/", "chutes/", "authgpt/",
            "deepseek/", "mistral/", "xai/", "fireworks/", "nvidia/")
EXCLUDED = ("autharena/", "antigravity/", "ocagy/", "ollamapull/")  # routes excluded on mobile (a ReasonChip)
CATALOG = ["gpt-6", "gpt-6-mini", "claude-opus-5-5", "gemini-3.5-flash", "authgpt/gpt-6-luna", "deepseek-v4",
           "grok-5", "mistral-large-3", "qwen3-max", "kimi-k3", "glm-5", "o5-mini", "gemini-3.5-pro",
           "claude-sonnet-5", "llama-5-70b", "command-r3"]


def desktop_ids(n: int = 3760) -> list:
    """The desktop-import case: ~35 % excluded routes, like the owner's list."""
    out = []
    for i in range(n):
        prefix = EXCLUDED[i % len(EXCLUDED)] if i % 20 < 7 else PREFIXES[i % len(PREFIXES)]
        out.append(f"{prefix}vendor{i % 37}/model-{i}-instruct")
    return out


class CountingOptions(MK.FakeOptions):
    """FakeOptions that counts catalog reads and can hold them (a slow phone start) or fail them."""

    def __init__(self, *a, gate=None, fail=None, **k):
        super().__init__(*a, **k)
        self.reads = 0
        self.gate = gate
        self.fail = fail

    def get_model_options(self):
        self.reads += 1
        if self.gate is not None:
            self.gate.wait(10)
        if self.fail is not None:
            raise self.fail
        return super().get_model_options()


def big_options(**kw) -> CountingOptions:
    ids = desktop_ids()
    return CountingOptions(catalog=CATALOG, polled={"openrouter": [m for i, m in enumerate(ids) if i % 3 == 0]
                                                    + ["gpt-6"]}, **kw)


# ---- wire measurements ------------------------------------------------------------------------------


def _instrument(conn) -> None:
    """Keep every message's encoded bytes at send time (a later re-encode would see mutated controls)."""
    import msgpack
    from flet.controls.base_control import BaseControl
    from flet.messaging.protocol import configure_encode_object_for_msgpack

    encode = configure_encode_object_for_msgpack(BaseControl)
    conn.raw_sent = []
    original = conn.send_message

    def send_message(message):
        conn.raw_sent.append(msgpack.packb([message.action, message.body], default=encode))
        return original(message)

    conn.send_message = send_message


def _stats(conn, start: int) -> dict:
    import msgpack

    total = controls = dismissibles = 0
    for raw in conn.raw_sent[start:]:
        total += len(raw)
        stack = [msgpack.unpackb(raw, strict_map_key=False)]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                if "_i" in cur and "_c" in cur:
                    controls += 1
                    dismissibles += cur.get("_c") == "Dismissible"
                stack.extend(cur.values())
            elif isinstance(cur, (list, tuple)):
                stack.extend(cur)
    return {"messages": len(conn.raw_sent) - start, "bytes": total, "controls": controls, "rows": dismissibles}


async def _settle(predicate, timeout: float = 3.0) -> bool:
    for _ in range(int(timeout / 0.02)):
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


def _keys(screen) -> list:
    return [getattr(c, "key", None) for c in screen.list_view.controls]


def _scroll_resets(conn) -> list:
    """(control id, offset) of every ``scroll_to(offset=…)`` the app asked the client for."""
    from flet.messaging.protocol import MessageAction

    return [(m.body.control_id, m.body.args.get("offset")) for m in conn.messages
            if m.action == MessageAction.INVOKE_METHOD and m.body.name == "scroll_to"]


async def _mount(tmp_path, *, ids=None, options=None, run_io=None, load=True, notes=None):
    """A standalone Model Manager on the fake android session (412 dp), mounted and shown."""
    from glossarion_mobile.ui.screens.model_manager import ModelManagerScreen

    conn, session = _fake_session("android")
    _instrument(conn)
    page = session.page
    store = MK._store(tmp_path, {"model": "gpt-6", "custom_model_list": desktop_ids() if ids is None else ids})
    options = options if options is not None else big_options()
    service = MK._service(store, options)
    if load:
        service.load_blocking()
    sink = notes if notes is not None else []
    screen = ModelManagerScreen(parse_route("/settings/models"), catalog=service, page=page,
                                notify=lambda m, a=None, cb=None: sink.append((m, a, cb)),
                                spawn=lambda c: asyncio.ensure_future(c), run_io=run_io)
    start = len(conn.raw_sent)
    page.views[0].controls.append(screen.get_body())
    page.update()
    first = _stats(conn, start)
    screen.did_show()
    await asyncio.sleep(0)
    return types.SimpleNamespace(conn=conn, session=session, page=page, store=store, options=options, service=service,
                                 screen=screen, first=first, notes=sink)


# ==========================================================================
# first paint, Show more, windows
# ==========================================================================


@needs_flet
def test_first_paint_of_3776_models_is_one_window(tmp_path):
    import flet as ft

    async def scenario():
        started = time.perf_counter()
        env = await _mount(tmp_path)
        elapsed = time.perf_counter() - started
        screen = env.screen
        try:
            assert len(env.service.snapshot.models) == 3776
            # before: 4.4 MB / 53k controls / every row mounted
            assert env.first["bytes"] <= FIRST_PATCH_BYTES and env.first["controls"] <= FIRST_PATCH_CONTROLS, env.first
            assert len(screen.list_view.controls) == env.first["rows"] == 100
            assert isinstance(screen.list_view, ft.ReorderableListView)
            assert screen.list_view is screen.rows.list_view  # the alias tests and the UI driver read
            assert _keys(screen)[:2] == ["mm-" + desktop_ids()[0], "mm-" + desktop_ids()[1]]
            assert screen.hint.value.startswith("3776 models · drag ☰ or long-press to move")
            more = screen.rows.more_button
            assert screen.rows.more_holder.visible and more.content == "Show 100 more (3,676 left)"
            assert screen.list_view.footer is screen.rows.more_holder
            assert elapsed < 5.0  # secondary (host: ~0.1 s; before: 2-3 s)
        finally:
            env.store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_show_more_appends_steps_then_pages_windows(tmp_path):
    async def scenario():
        env = await _mount(tmp_path)
        screen, conn, session = env.screen, env.conn, env.session
        try:
            more = screen.rows.more_button
            for mounted in (200, 300, 400, 500):
                start = len(conn.raw_sent)
                await session.dispatch_event(more._i, "click", None)
                stats = _stats(conn, start)
                assert len(screen.list_view.controls) == mounted
                assert stats["rows"] == 100 and stats["bytes"] <= WINDOW_PATCH_BYTES, stats
            assert more.content == "Show rows 501–1,000 (3,276 left)"
            assert _scroll_resets(conn) == []  # appending steps keeps the scroll position
            start = len(conn.raw_sent)
            await session.dispatch_event(more._i, "click", None)  # the next window
            stats = _stats(conn, start)
            assert screen.rows.window_start == 500 and len(screen.list_view.controls) == 100
            assert screen.rows.window_text.value == "Rows 501–1,000 of 3,776" and screen.rows.selector.visible
            assert stats["bytes"] <= WINDOW_PATCH_BYTES, stats
            assert _keys(screen)[0] == "mm-" + desktop_ids()[500]
            # the footer was tapped at the end of row 500: the new window opens at its first row (Flutter
            # keeps the offset and only clamps it, which would land next to row 600)
            assert await _settle(lambda: len(_scroll_resets(conn)) == 1)
            assert _scroll_resets(conn) == [(screen.list_view._i, 0)]
            # scrolling near the end appends the next step too (on_scroll, when the client reports it)
            screen.rows._on_scroll(types.SimpleNamespace(event_type="update", pixels=900.0, max_scroll_extent=1000.0))
            assert len(screen.list_view.controls) == 200
            # previous window (the page selector), at its first row again
            await session.dispatch_event(screen.rows.prev_button._i, "click", None)
            assert screen.rows.window_start == 0 and len(screen.list_view.controls) == 100
            assert await _settle(lambda: len(_scroll_resets(conn)) == 2)
        finally:
            env.store._saver.close()

    asyncio.run(scenario())


# ==========================================================================
# search
# ==========================================================================


@needs_flet
def test_search_is_debounced_off_the_loop_and_window_sized(tmp_path):
    calls: list = []

    async def run_io(fn, *args):
        calls.append(getattr(fn, "__name__", repr(fn)))
        return await asyncio.to_thread(fn, *args)

    async def scenario():
        env = await _mount(tmp_path, run_io=run_io)
        screen, conn, session = env.screen, env.conn, env.session
        try:
            first_rows = list(screen.list_view.controls)
            renders = screen.renders
            start = len(conn.raw_sent)
            for text in ("g", "gp", "gpt"):  # three quick keystrokes: one search
                session.apply_patch(screen.search._i, {"value": text})
                await session.dispatch_event(screen.search._i, "change", text)
                await asyncio.sleep(0.03)
            assert screen.renders == renders and screen.query == ""  # nothing yet (debounce)
            assert await _settle(lambda: screen.query == "gpt")
            stats = _stats(conn, start)
            assert calls.count("_visible_for") == 1 and screen.renders == renders + 1
            assert len(screen.list_view.controls) <= 100 and stats["bytes"] <= WINDOW_PATCH_BYTES, stats
            assert all("gpt" in str(k).casefold() for k in _keys(screen))
            assert not screen.reorderable and screen.hint.value.endswith("clear the search and filters to reorder")
            # clearing goes back to the first window; the rows are the cached ones (same objects)
            start = len(conn.raw_sent)
            session.apply_patch(screen.search._i, {"value": ""})
            await session.dispatch_event(screen.search._i, "change", "")
            assert await _settle(lambda: screen.query == "")
            stats = _stats(conn, start)
            assert stats["bytes"] <= WINDOW_PATCH_BYTES and len(screen.list_view.controls) == 100, stats
            assert screen.rows.window_start == 0 and screen.reorderable
            unchanged = sum(1 for a, b in zip(first_rows, screen.list_view.controls) if a is b)
            assert unchanged >= 90  # rows not touched by the search are reused, not rebuilt
        finally:
            env.store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_a_newer_search_wins_over_a_slower_one(tmp_path):
    release = threading.Event()

    async def slow_io(fn, *args):
        def run():
            release.wait(5)
            return fn(*args)

        return await asyncio.to_thread(run)

    async def scenario():
        env = await _mount(tmp_path, ids=desktop_ids(300), run_io=slow_io)
        screen = env.screen
        try:
            task = asyncio.ensure_future(screen.apply_search("gpt"))
            await asyncio.sleep(0.05)
            screen.set_query("claude")  # applied at once; the running "gpt" search is now stale
            release.set()
            await task
            assert screen.query == "claude"
            assert all("claude" in str(k).casefold() for k in _keys(screen))
        finally:
            release.set()
            env.store._saver.close()

    asyncio.run(scenario())


# ==========================================================================
# publishes: identical ones are dropped, the rest render once
# ==========================================================================


@needs_flet
def test_an_unchanged_publish_sends_nothing_and_keeps_the_rows(tmp_path):
    async def scenario():
        env = await _mount(tmp_path)
        screen, conn, service = env.screen, env.conn, env.service
        try:
            seen: list = []
            service.subscribe(seen.append)
            rows = list(screen.list_view.controls)
            renders, list_renders = screen.renders, screen.list_renders
            start = len(conn.raw_sent)
            before = service.snapshot
            assert service.load_blocking() is before  # nothing changed: the same snapshot, nobody notified
            await asyncio.sleep(0.05)
            assert seen == [] and screen.renders == renders and _stats(conn, start)["bytes"] == 0
            # a snapshot equal in content (another publisher): one render, no row rebuilt, nothing sent
            screen._on_snapshot(dataclasses.replace(before, updated_at=before.updated_at + 1))
            await asyncio.sleep(0.05)
            assert screen.list_renders == list_renders and _stats(conn, start)["bytes"] == 0
            assert all(a is b for a, b in zip(rows, screen.list_view.controls))
            # a status-only publish (a poll status) updates the header, not the rows
            service._publish(statuses={"openai": "online (3 models)"})
            await asyncio.sleep(0.05)
            stats = _stats(conn, start)
            assert screen.list_renders == list_renders and stats["rows"] == 0 and 0 < stats["bytes"] < SMALL_PATCH_BYTES
        finally:
            env.store._saver.close()

    asyncio.run(scenario())


def test_service_load_is_shared_and_identical_reloads_are_not_published(tmp_path):
    gate = threading.Event()
    options = big_options(gate=gate)
    store = MK._store(tmp_path, {"custom_model_list": ["mine"]})
    service = MK._service(store, options)
    seen: list = []
    service.subscribe(seen.append)

    async def scenario():
        first = asyncio.ensure_future(service.load())
        second = asyncio.ensure_future(service.load())  # a screen opened during a slow start
        await asyncio.sleep(0.05)
        gate.set()
        a, b = await asyncio.gather(first, second)
        assert a is b and options.reads == 1 and len(seen) == 1
        assert a.known_keys == frozenset(m.casefold() for m in CATALOG)
        # a reload after a config write the app made itself: same content, not published
        await service.load(fresh=True)
        assert options.reads == 2 and len(seen) == 1 and service.snapshot is a
        store.set("custom_model_list", ["mine", "yours"])
        snap = await service.load(fresh=True)
        assert len(seen) == 2 and snap.models[:2] == ("mine", "yours")

    try:
        asyncio.run(scenario())
    finally:
        store._saver.close()


# ==========================================================================
# edits
# ==========================================================================


@needs_flet
def test_swipe_remove_is_one_render_and_undo_puts_a_new_row_back(tmp_path):
    async def scenario():
        env = await _mount(tmp_path)
        screen, conn, session, store = env.screen, env.conn, env.session, env.store
        try:
            victim = screen.list_view.controls[4]
            model = desktop_ids()[4]
            renders, list_renders = screen.renders, screen.list_renders
            start = len(conn.raw_sent)
            await session.dispatch_event(victim._i, "dismiss", {"direction": "endToStart"})
            assert victim not in screen.list_view.controls  # left the tree in the event's own update
            await asyncio.sleep(0.1)
            stats = _stats(conn, start)
            assert screen.renders == renders + 1 and screen.list_renders <= list_renders + 1
            # the removal, the hint, and row 101 moving up to keep the step full (before: 3 full renders)
            assert stats["bytes"] < SMALL_PATCH_BYTES and stats["rows"] <= 1, stats
            assert model in store.get("model_manager_removed_models") and model not in env.service.snapshot.models
            assert len(screen.list_view.controls) == 100 and screen.hint.value.startswith("3775 models")
            assert model not in [m for m, _n in screen.rows.items]
            message, action, undo = env.notes[-1]
            assert action == "Undo" and message == f"Removed {model}"
            renders = screen.renders
            start = len(conn.raw_sent)
            undo()
            await asyncio.sleep(0.1)
            stats = _stats(conn, start)
            assert screen.renders == renders + 1 and stats["rows"] == 1 and stats["bytes"] < SMALL_PATCH_BYTES, stats
            back = screen.list_view.controls[4]
            assert back.key == "mm-" + model and back is not victim  # never re-mount a dismissed Dismissible
            assert model not in store.get("model_manager_removed_models")
        finally:
            env.store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_two_quick_swipes_both_persist(tmp_path):
    async def scenario():
        env = await _mount(tmp_path)
        screen, session, store = env.screen, env.session, env.store
        try:
            a, b = screen.list_view.controls[2], screen.list_view.controls[3]
            await session.dispatch_event(a._i, "dismiss", {"direction": "endToStart"})
            await session.dispatch_event(b._i, "dismiss", {"direction": "endToStart"})
            await asyncio.sleep(0.1)
            removed = store.get("model_manager_removed_models")
            ids = desktop_ids()
            assert ids[2] in removed and ids[3] in removed
            assert ids[2] not in env.service.snapshot.models and ids[3] not in env.service.snapshot.models
            assert a not in screen.list_view.controls and b not in screen.list_view.controls
        finally:
            env.store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_reorder_in_the_second_window_maps_to_absolute_indices(tmp_path):
    async def scenario():
        env = await _mount(tmp_path)
        screen, store = env.screen, env.store
        try:
            previous = list(env.service.snapshot.models)
            screen.rows.set_window(500)
            moved = screen.list_view.controls[3]
            # the client already moved row 3 to the top of the window (Flet sends the final index)
            screen._on_reorder(types.SimpleNamespace(old_index=3, new_index=0))
            await asyncio.sleep(0.05)
            expected = previous[:500] + [previous[503]] + previous[500:503] + previous[504:]
            assert list(env.service.snapshot.models) == expected
            assert store.get("custom_model_list")[:504] == expected[:504]
            assert screen.list_view.controls[0] is moved and screen.rows.window_start == 500
            # reordering is off while a search filters the list: nothing is saved
            screen.set_query("model-1")
            snapshot = env.service.snapshot
            screen._on_reorder(types.SimpleNamespace(old_index=1, new_index=0))
            assert env.service.snapshot is snapshot
        finally:
            env.store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_poll_updates_the_header_and_then_the_list_once(tmp_path):
    hold = threading.Event()

    class HeldOptions(CountingOptions):
        def refresh_provider_model_catalogs(self, **kwargs):
            hold.wait(5)
            return super().refresh_provider_model_catalogs(**kwargs)

    async def scenario():
        ids = desktop_ids()
        options = HeldOptions(catalog=CATALOG, polled={"openrouter": [m for i, m in enumerate(ids) if i % 3 == 0]})
        env = await _mount(tmp_path, options=options)
        screen, conn, session = env.screen, env.conn, env.session
        try:
            online = ids[:50] + ["gpt-7"]
            options.results = [options.ModelCatalogRefreshResult(online, {"openrouter": online},
                                                                 {"openrouter": "online (51 models)"}, None)]
            gaps: list = []

            async def heartbeat():
                loop = asyncio.get_running_loop()
                while True:
                    tick = loop.time()
                    await asyncio.sleep(0.01)
                    gaps.append(loop.time() - tick - 0.01)

            # a full GC of a long single-process run's heap stalls the loop for 0.5-1.2 s on its own
            # (measured in the CI-order run); collect it now and keep it out of the heartbeat window
            gc.collect()
            gc.freeze()
            beat = asyncio.ensure_future(heartbeat())
            list_renders = screen.list_renders
            start = len(conn.raw_sent)
            await session.dispatch_event(screen.poll_button._i, "click", None)
            assert await _settle(lambda: screen.poll_button.content == "⏳ Polling…")
            assert screen.poll_button.disabled and screen.poll_text.value.startswith("Contacting provider catalogs")
            await asyncio.sleep(0.1)
            during = _stats(conn, start)
            assert during["rows"] == 0 and 0 < during["bytes"] < SMALL_PATCH_BYTES, during  # header only
            hold.set()
            assert await _settle(lambda: not screen.polling and not env.service.snapshot.polling)
            await asyncio.sleep(0.1)
            assert screen.list_renders == list_renders + 1  # the list updated once, after the poll
            assert screen.poll_button.content == "🌐 Poll providers" and not screen.poll_button.disabled
            assert "gpt-7" in env.service.snapshot.models and _keys(screen)[0] == "mm-" + ids[0]
            assert _stats(conn, start)["bytes"] <= WINDOW_PATCH_BYTES
            beat.cancel()
            # secondary, generous for slow CI runners (host: 0.02 s; before the fix: up to 4 s per poll)
            assert max(gaps) < LOOP_GAP_S
        finally:
            gc.unfreeze()
            hold.set()
            env.store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_the_users_own_changes_update_the_list_while_a_poll_runs(tmp_path):
    """Poll providers holds only its own publishes (the header while it runs, the list once at the end).
    A chip, an Undo and a search the user makes meanwhile update the list at once, not when the poll ends
    (a real poll waits up to 8 s per provider on a phone network)."""
    hold = threading.Event()

    class HeldOptions(CountingOptions):
        def refresh_provider_model_catalogs(self, **kwargs):
            hold.wait(10)
            return super().refresh_provider_model_catalogs(**kwargs)

    async def scenario():
        ids = desktop_ids(300)
        options = HeldOptions(catalog=CATALOG, polled={"openrouter": [m for i, m in enumerate(ids) if i % 3 == 0]})
        env = await _mount(tmp_path, ids=ids, options=options)
        screen, session, service = env.screen, env.session, env.service
        try:
            tombstone = ids[7]
            await session.dispatch_event(screen.list_view.controls[7]._i, "dismiss", {"direction": "endToStart"})
            await asyncio.sleep(0.05)
            total = len(service.snapshot.models)
            online = ids[:50] + ["gpt-7"]
            options.results = [options.ModelCatalogRefreshResult(online, {"openrouter": online},
                                                                 {"openrouter": "online (51 models)"}, None)]
            await session.dispatch_event(screen.poll_button._i, "click", None)
            assert await _settle(lambda: screen.polling and screen.poll_button.content == "⏳ Polling…")
            # Removed: the tombstones show at once
            await session.dispatch_event(screen.removed_chip._i, "click", None)
            assert _keys(screen) == ["mm-" + tombstone] and screen.hint.value == "1 removed · swipe to restore"
            await session.dispatch_event(screen.removed_chip._i, "click", None)
            assert _keys(screen)[:2] == ["mm-" + ids[0], "mm-" + ids[1]]
            assert screen.hint.value.startswith(f"{total} models")
            # swipe + Undo: the row is back as soon as the edit's render runs
            model = ids[4]
            await session.dispatch_event(screen.list_view.controls[4]._i, "dismiss", {"direction": "endToStart"})
            await asyncio.sleep(0.05)
            assert "mm-" + model not in _keys(screen)
            _message, action, undo = env.notes[-1]
            assert action == "Undo"
            undo()
            await asyncio.sleep(0.05)
            assert model in service.snapshot.models and _keys(screen)[4] == "mm-" + model
            # a typed search: filtered after the debounce, still while polling
            session.apply_patch(screen.search._i, {"value": "model-7"})
            await session.dispatch_event(screen.search._i, "change", "model-7")
            assert await _settle(lambda: screen.query == "model-7")
            keys = _keys(screen)
            assert keys and all("model-7" in key for key in keys) and screen.hint.value.endswith("to reorder")
            assert screen.polling and service.snapshot.is_polling()
            # the poll's end: its result syncs the list once, the search stays applied
            list_renders = screen.list_renders
            hold.set()
            assert await _settle(lambda: not screen.polling and not service.snapshot.polling)
            await asyncio.sleep(0.1)
            assert screen.list_renders == list_renders + 1 and "gpt-7" in service.snapshot.models
            assert screen.query == "model-7" and all("model-7" in key for key in _keys(screen))
        finally:
            hold.set()
            env.store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_row_actions_move_a_model_anywhere_in_the_list(tmp_path):
    """Long-press › Move to top / up / down / bottom (the desktop ⇈ ↑ ↓ ⇊ buttons). A drag stays inside the
    mounted window; these reach any position of the whole list, e.g. a model in the second window to the
    top, saved like a drag (``move_model`` with indices in the whole list)."""
    async def scenario():
        env = await _mount(tmp_path)
        screen, conn, session, service, store, page = env.screen, env.conn, env.session, env.service, env.store, env.page
        try:
            order = list(service.snapshot.models)

            async def act(row_index: int, label: str) -> dict:
                row = screen.list_view.controls[row_index]
                await session.dispatch_event(row.content._i, "long_press", None)  # the row's ListTile
                sheet = screen.last_sheet
                assert sheet is not None and sheet.dialog in page._dialogs.controls
                start = len(conn.raw_sent)
                await session.dispatch_event(sheet.tiles[[i.label for i in sheet.items].index(label)]._i, "click", None)
                await asyncio.sleep(0.05)
                return _stats(conn, start)

            screen.rows.set_window(500)
            target = order[503]
            await session.dispatch_event(screen.list_view.controls[3].content._i, "long_press", None)
            sheet = screen.last_sheet
            assert sheet.title == target and sheet.subtitle == "Row 504 of 3,776"
            assert [i.label for i in sheet.items] == ["Move to top", "Move up", "Move down", "Move to bottom", "Remove"]
            assert all(i.disabled_reason is None for i in sheet.items)
            sheet.close()
            stats = await act(3, "Move to top")
            order = [target] + order[:503] + order[504:]
            assert list(service.snapshot.models) == order and store.get("custom_model_list")[:2] == order[:2]
            assert screen.rows.window_start == 500 and _keys(screen)[:4] == ["mm-" + m for m in order[500:504]]
            assert len(screen.list_view.controls) == 100 and env.notes[-1][0] == f"Moved {target} to the top"
            # one row joins the window (the one pushed down from row 500), the moved one leaves it
            assert stats["rows"] == 1 and stats["bytes"] < SMALL_PATCH_BYTES, stats
            # Move up across the window boundary: row 501 becomes row 500 of the first window
            first = order[500]
            stats = await act(0, "Move up")
            order[499], order[500] = order[500], order[499]
            assert list(service.snapshot.models) == order and _keys(screen)[0] == "mm-" + order[500]
            assert env.notes[-1][0] == f"Moved {first} to row 500"
            assert stats["rows"] == 1 and stats["bytes"] < SMALL_PATCH_BYTES, stats
            # Move down inside the mounted rows (no message), Move to bottom (the last row of the list)
            notes = len(env.notes)
            second = order[501]
            await act(1, "Move down")
            order[501], order[502] = order[502], order[501]
            assert list(service.snapshot.models) == order and _keys(screen)[1:3] == ["mm-" + order[501], "mm-" + second]
            assert len(env.notes) == notes
            await act(2, "Move to bottom")
            order = order[:502] + order[503:] + [second]
            assert list(service.snapshot.models) == order and store.get("custom_model_list")[-1] == second
            assert env.notes[-1][0] == f"Moved {second} to row 3,776"
            # at the top of the list: up / top are listed but unavailable, with the reason
            screen.rows.set_window(0)
            await session.dispatch_event(screen.list_view.controls[0].content._i, "long_press", None)
            reasons = [i.disabled_reason for i in screen.last_sheet.items]
            assert reasons == ["Already at the top.", "Already at the top.", None, None, None]
            screen.last_sheet.close()
            # while a search filters the list the moves are unavailable and save nothing
            screen.set_query("model-1")
            snapshot = service.snapshot
            await session.dispatch_event(screen.list_view.controls[1].content._i, "long_press", None)
            sheet = screen.last_sheet
            assert [i.disabled_reason for i in sheet.items][:4] == ["Clear the search and filters to reorder."] * 4
            await session.dispatch_event(sheet.tiles[0]._i, "click", None)  # explains why instead
            assert sheet.explained is sheet.items[0] and service.snapshot is snapshot
            # Removed: no row actions (swipe restores)
            screen.set_query("")
            assert screen.remove(order[10])
            screen.set_filter("removed")
            assert _keys(screen) == ["mm-" + order[10]] and screen.list_view.controls[0].content.on_long_press is None
        finally:
            env.store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_a_move_whose_save_fails_keeps_the_saved_order(tmp_path):
    """A drag the client already showed, then the save fails (no model catalog core, a config write error):
    the mounted rows keep the saved order and the moved row is re-sent so the client re-reads it."""
    async def scenario():
        ids = desktop_ids(300)
        env = await _mount(tmp_path, ids=ids)
        screen, conn, session, service = env.screen, env.conn, env.session, env.service
        core = service.core
        real_fn = core.fn
        try:
            saved = list(service.snapshot.models)
            core.fn = lambda *names: None if "save_model_order" in names else real_fn(*names)
            row5 = screen.list_view.controls[5]
            start = len(conn.raw_sent)
            await session.dispatch_event(screen.list_view._i, "reorder", {"old_index": 5, "new_index": 0})
            await asyncio.sleep(0.05)
            stats = _stats(conn, start)
            assert list(service.snapshot.models) == saved and env.notes[-1][0] == mc.CORE_MISSING
            assert _keys(screen) == ["mm-" + m for m in saved[:100]] and screen.list_view.controls[5] is row5
            # the drag mirrored, then moved back: two small Move patches the client applies (re-sending the
            # unchanged order would send nothing, Flet matches rows by key, and the moved row would stay)
            assert stats["messages"] == 2 and stats["rows"] == 0 and stats["bytes"] < SMALL_PATCH_BYTES, stats
            service._publish(statuses={"openai": "online (3 models)"})  # a later publish leaves it so
            await asyncio.sleep(0.05)
            assert _keys(screen) == ["mm-" + m for m in saved[:100]]
            # a row action whose save fails moves nothing
            rows_before = list(screen.list_view.controls)
            await session.dispatch_event(screen.list_view.controls[5].content._i, "long_press", None)
            await session.dispatch_event(screen.last_sheet.tiles[0]._i, "click", None)
            await asyncio.sleep(0.05)
            assert list(service.snapshot.models) == saved and env.notes[-1][0] == mc.CORE_MISSING
            assert all(a is b for a, b in zip(rows_before, screen.list_view.controls))
        finally:
            core.fn = real_fn
            env.store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_a_scroll_event_that_appends_nothing_repaints_nothing(tmp_path, monkeypatch):
    """The manager's list reports ``on_scroll`` every 120 ms while the user scrolls. Flet 1.0.3 auto-updates
    the Page after a handler that called no ``update()``: a whole-page diff (15-57 ms on the host with 100-500
    rows mounted) per event. An event that appends nothing now updates nothing."""
    async def scenario():
        env = await _mount(tmp_path)
        screen, conn, session, page = env.screen, env.conn, env.session, env.page
        try:
            updates: list = []
            original = type(page).update

            def counting(self, *controls):
                updates.append(len(controls))
                return original(self, *controls)

            monkeypatch.setattr(type(page), "update", counting)
            lv = screen.list_view
            event = {"event_type": "update", "pixels": 100.0, "max_scroll_extent": 99999.0, "min_scroll_extent": 0.0,
                     "viewport_dimension": 600.0}
            start = len(conn.raw_sent)
            for i in range(5):
                await session.dispatch_event(lv._i, "scroll", dict(event, pixels=100.0 + i))
            await session.dispatch_event(lv._i, "scroll", dict(event, event_type="overscroll", pixels=0.0,
                                                               overscroll=-4.0, velocity=0.0))
            assert updates == [] and len(conn.raw_sent) == start and len(lv.controls) == 100
            # near the end: the next step is appended with the list's own update (not the page's)
            await session.dispatch_event(lv._i, "scroll", dict(event, pixels=99700.0))
            stats = _stats(conn, start)
            assert len(lv.controls) == 200 and updates == [1] and stats["rows"] == 100, stats
        finally:
            env.store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_still_loading_retry_shares_the_stuck_load_and_offers_retry_again(tmp_path, monkeypatch):
    """"Still loading…" › Retry while the first catalog read is still stuck: the skeleton, then "Still
    loading…" with Retry again (never an endless skeleton), and no second read queued behind the first."""
    import flet as ft

    from glossarion_mobile.ui.screens import model_manager as mm

    monkeypatch.setattr(mm, "SLOW_LOAD_SECONDS", 0.2)

    def state(screen):
        return getattr(screen.state_box.content, "key", None) if screen.state_box.visible else None

    async def scenario():
        gate = threading.Event()
        options = big_options(gate=gate)
        env = await _mount(tmp_path, options=options, load=False)
        screen, session = env.screen, env.session
        try:
            assert await _settle(lambda: state(screen) == "mm-slow")
            retry = [c for c in screen.state_box.content.controls if isinstance(c, ft.TextButton)]
            assert len(retry) == 1
            await session.dispatch_event(retry[0]._i, "click", None)
            assert await _settle(lambda: state(screen) == "mm-skeleton", timeout=0.15)
            assert await _settle(lambda: state(screen) == "mm-slow", timeout=2)  # still stuck: Retry again
            assert options.reads == 1
            gate.set()
            assert await _settle(lambda: len(screen.list_view.controls) == 100 and state(screen) is None, timeout=5)
            await asyncio.sleep(0.2)
            assert options.reads == 1  # the Retry shared the read in flight
        finally:
            gate.set()
            env.store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_duplicate_ids_get_their_own_keys(tmp_path):
    async def scenario():
        env = await _mount(tmp_path, ids=["dup-model", "other", "dup-model"])
        try:
            keys = _keys(env.screen)
            assert keys[:3] == ["mm-dup-model", "mm-other", "mm-dup-model#1"] and len(set(keys)) == len(keys)
        finally:
            env.store._saver.close()

    asyncio.run(scenario())


# ==========================================================================
# loading / error states and the Custom filter
# ==========================================================================


@needs_flet
def test_a_failed_catalog_read_shows_retry(tmp_path):
    async def scenario():
        options = big_options(fail=RuntimeError("cache file is unreadable"))
        env = await _mount(tmp_path, options=options, load=False)
        screen = env.screen
        try:
            assert getattr(screen.state_box.content, "key", None) == "mm-skeleton"  # not loaded yet
            assert await _settle(lambda: getattr(screen.state_box.content, "key", None) == "mm-error")
            card = screen.state_box.content
            assert "cache file is unreadable" in card.text and card.on_retry is not None
            options.fail = None
            await card._retry()
            assert screen.state_box.visible is False and len(screen.list_view.controls) == 100
        finally:
            env.store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_a_raising_load_shows_retry_and_a_slow_one_says_so(tmp_path, monkeypatch):
    from glossarion_mobile.ui.screens import model_manager as mm

    monkeypatch.setattr(mm, "SLOW_LOAD_SECONDS", 0.05)

    async def scenario():
        gate = threading.Event()
        env = await _mount(tmp_path, options=big_options(gate=gate), load=False)
        screen = env.screen
        try:
            assert await _settle(lambda: getattr(screen.state_box.content, "key", None) == "mm-slow")
            gate.set()
            assert await _settle(lambda: len(screen.list_view.controls) == 100 and not screen.state_box.visible)
            # a load() that raises (outside the service's own error handling)
            real_load = env.service.load

            async def broken(**kw):
                raise RuntimeError("worker pool is gone")

            env.service.load = broken
            await screen._load()
            await asyncio.sleep(0.05)
            card = screen.state_box.content
            assert getattr(card, "key", None) == "mm-error" and "worker pool is gone" in card.text
            env.service.load = real_load
            await card._retry()
            assert not screen.state_box.visible
        finally:
            gate.set()
            env.store._saver.close()

    asyncio.run(scenario())


@needs_flet
def test_custom_filter_waits_for_the_catalog_and_reads_it_once(tmp_path):
    async def scenario():
        gate = threading.Event()
        options = big_options(gate=gate)
        env = await _mount(tmp_path, ids=["mine-1", "gpt-6", "mine-2"], options=options, load=False)
        screen = env.screen
        try:
            assert screen.builtin_keys is None and screen.custom_chip.disabled
            screen.set_filter("custom")  # never "every model is custom" before the catalog is read
            assert screen.list_view.controls == []
            gate.set()
            assert await _settle(lambda: env.service.snapshot.loaded)
            await asyncio.sleep(0.05)
            assert not screen.custom_chip.disabled
            assert _keys(screen) == ["mm-mine-1", "mm-mine-2"]
            assert options.reads == 1  # the snapshot's known_keys: no second catalog read
        finally:
            gate.set()
            env.store._saver.close()

    asyncio.run(scenario())


# ==========================================================================
# on the real app shell
# ==========================================================================


def _real_shell_setup(monkeypatch, options):
    from glossarion_mobile.ui.screens import keys as keys_module
    from glossarion_mobile.ui.screens import model_manager as mm

    monkeypatch.setattr(mm, "ModelCatalogService",
                        lambda s, **kw: mc.ModelCatalogService(s, options=options, is_mobile=False, **kw))
    real_backend = keys_module.KeyBackend
    monkeypatch.setattr(keys_module, "KeyBackend", lambda *a, **k: real_backend(MK._fake_key_service()))
    return mm


async def _start_app(conn_session):
    main_module = _TB._load_main_module()
    conn, session = conn_session
    page = session.page
    await main_module.main(page)
    await session.after_event(page)
    return page.data


@needs_flet
def test_real_shell_open_swipe_hidden_and_back(app_env, monkeypatch):
    """/settings/models on the real shell: a windowed first patch, a swipe that renders once (the feature's
    0.3 s config reload publishes nothing new), no renders while /settings/keys covers it, one on Back."""
    options = big_options()
    mm = _real_shell_setup(monkeypatch, options)
    from glossarion_mobile.ui.sheets import model_sheet as ms

    async def scenario():
        conn, session = _fake_session("android")
        _instrument(conn)
        page = session.page
        app = await _start_app((conn, session))
        try:
            feature = getattr(app, "models_keys", None) or await mm.ModelsKeysFeature.install(app)
            ids = desktop_ids()
            app.config_store.set("custom_model_list", ids)
            assert await _settle(lambda: len(feature.catalog.snapshot.models) == 3776, timeout=5)
            await asyncio.sleep(0.5)
            start = len(conn.raw_sent)
            await _TB._route(session, "/settings/models")
            first = _stats(conn, start)
            screen = app.shell.top_screen
            assert isinstance(screen, mm.ModelManagerScreen) and len(screen.list_view.controls) == 100
            assert first["bytes"] <= SHELL_FIRST_PATCH_BYTES and first["controls"] <= SHELL_FIRST_PATCH_CONTROLS, first
            assert first["rows"] == 100
            await asyncio.sleep(0.3)
            # swipe through the real Dismissible event
            renders = screen.renders
            start = len(conn.raw_sent)
            victim = screen.list_view.controls[4]
            await session.dispatch_event(victim._i, "dismiss", {"direction": "endToStart"})
            assert victim not in screen.list_view.controls
            await asyncio.sleep(1.0)  # past the feature's 0.3 s reload after the config write
            stats = _stats(conn, start)
            assert screen.renders == renders + 1 and stats["rows"] <= 1 and stats["bytes"] < SMALL_PATCH_BYTES, stats
            assert ids[4] in app.config_store.get("model_manager_removed_models")
            # covered by another screen: a catalog change does not paint it
            await _TB._route(session, "/settings/keys")
            assert any(e.screen is screen for e in app.shell.stack) and app.shell.top_screen is not screen
            renders = screen.renders
            app.config_store.set("custom_model_list", ["brand-new-model"] + ids)
            assert await _settle(lambda: feature.catalog.snapshot.models[0] == "brand-new-model")
            await asyncio.sleep(0.6)
            assert screen.renders == renders
            # Back: one render with the new data
            app.shell.pop()
            page.update()
            assert await _settle(lambda: screen.renders == renders + 1, timeout=2)
            await asyncio.sleep(0.4)
            assert screen.renders == renders + 1 and _keys(screen)[0] == "mm-brand-new-model"
        finally:
            ms.install_sheet_env(None)
            mc.set_default_service(None)
            await app.dispatcher.stop()
            await asyncio.sleep(0)

    asyncio.run(scenario())


@needs_flet
def test_real_shell_cold_open_reads_the_catalog_once_and_renders_the_list_once(app_env, monkeypatch):
    """Opened before the install's catalog load finished (a slow phone start): the skeleton, then one
    list render when that same load lands; the screen shares the load (one catalog read, not three)."""
    gate = threading.Event()
    options = big_options(gate=gate)
    mm = _real_shell_setup(monkeypatch, options)
    from glossarion_mobile.ui.sheets import model_sheet as ms

    with open(os.environ["CONFIG_FILE"], "w", encoding="utf-8") as fh:  # the desktop-import config, on disk
        json.dump({"model": "gpt-6", "custom_model_list": desktop_ids()}, fh)

    async def scenario():
        conn, session = _fake_session("android")
        _instrument(conn)
        app = await _start_app((conn, session))
        try:
            feature = getattr(app, "models_keys", None) or await mm.ModelsKeysFeature.install(app)
            assert not feature.catalog.snapshot.loaded  # the install's load waits on the slow catalog read
            await _TB._route(session, "/settings/models")
            screen = app.shell.top_screen
            assert isinstance(screen, mm.ModelManagerScreen)
            assert getattr(screen.state_box.content, "key", None) == "mm-skeleton" and not screen.list_view.controls
            list_renders = screen.list_renders
            gate.set()
            assert await _settle(lambda: len(screen.list_view.controls) == 100, timeout=5)
            await asyncio.sleep(0.6)
            assert screen.list_renders == list_renders + 1 and not screen.state_box.visible
            assert len(feature.catalog.snapshot.models) == 3776
            assert options.reads == 1  # before: the install, did_show and the Custom filter each read it
        finally:
            gate.set()
            ms.install_sheet_env(None)
            mc.set_default_service(None)
            await app.dispatcher.stop()
            await asyncio.sleep(0)

    asyncio.run(scenario())


# ==========================================================================
# WindowedList options (components)
# ==========================================================================


@needs_flet
def test_windowed_list_reorderable_options_and_single_row_edits():
    import flet as ft

    from glossarion_mobile.ui.components.windowed_list import WindowedList

    built: list = []

    def row(item, index):
        built.append(item)
        return ft.Container(content=ft.Text(item))

    rlv = ft.ReorderableListView(controls=[], show_default_drag_handles=False)
    windowed = WindowedList(build_row=row, key_of=lambda item: f"k-{item}", window_rows=6, step=3, key="t",
                            list_view=rlv, scroll_keys=False, show_more=True)
    assert windowed.list_view is rlv and rlv.on_scroll is not None and rlv.key == "t-list"
    assert rlv.footer is windowed.more_holder
    windowed.set_items([f"i{n}" for n in range(10)])
    assert [c.key for c in rlv.controls] == ["k-i0", "k-i1", "k-i2"]  # plain string keys, not ScrollKey
    assert windowed.more_button.content == "Show 3 more (7 left)" and windowed.more_holder.visible
    windowed.load_more()
    assert windowed.mounted_count == 6 and windowed.more_button.content == "Show rows 7–10 (4 left)"
    windowed.set_window(6)
    assert windowed.absolute(1) == 7 and windowed.mounted_count == 3
    windowed.set_window(0)
    windowed.load_more()
    ids = [id(c) for c in rlv.controls]
    # remove_key: one row out, the others keep their objects
    assert windowed.remove_key("k-i2") and "i2" not in windowed.items
    assert [c.key for c in rlv.controls] == ["k-i0", "k-i1", "k-i3", "k-i4", "k-i5"]
    assert [id(c) for c in rlv.controls] == ids[:2] + ids[3:]
    assert windowed.positions["k-i3"] == 2
    # insert_item: in place, mounted; a full window drops its last row
    built.clear()
    assert windowed.insert_item(2, "i2")
    assert built == ["i2"] and [c.key for c in rlv.controls][:4] == ["k-i0", "k-i1", "k-i2", "k-i3"]
    assert windowed.mounted_count == 6
    assert windowed.insert_item(0, "new") and windowed.mounted_count == 6 and rlv.controls[0].key == "k-new"
    assert windowed.items[:2] == ["new", "i0"] and windowed.positions["k-i0"] == 1
    # an insert past the mounted rows only joins ``items``
    assert not windowed.insert_item(9, "late") and windowed.items[9] == "late"
    # the default list is unchanged: a ListView with ScrollKey rows
    plain = WindowedList(build_row=row, key_of=lambda item: f"k-{item}")
    plain.set_items(["a"])
    assert isinstance(plain.list_view, ft.ListView) and isinstance(plain.list_view.controls[0].key, ft.ScrollKey)
    assert plain.more_button is None and not plain.quiet_scroll and not plain.reset_scroll
    plain.set_window(0)  # off a page: no scroll request
    assert plain.scroll_resets == 0
