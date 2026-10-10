"""Owner issue 19: "the model manager in the mobile version is too laggy, or heck it doesn't even load".

What the owner saw: tapping Settings > Model manager froze the app, or the screen never seemed to finish
loading. The checked diagnosis (wf_1d567362-825): every model became a full Dismissible + ListTile row,
so the real catalogs (884 static models, 2,559 with the owner's polled cache, 3,776 after importing the
desktop config) went out as one 1-4.5 MB first patch of 12k-53k controls while the asyncio loop that
drives every Flet event was blocked for 0.5-3 s on the host (several times that on a phone). Every later
change rebuilt and re-diffed every row: each search keystroke, each chip, 3 full renders per swipe, 4-6
per poll, renders while another screen covered the manager, a second catalog load when the screen was
opened during start-up, and a failed catalog read showed "No models match." with no way to retry.

This acceptance test drives the REAL app shell (``main.py`` on the fake Flet session as Android, 412 dp,
``/settings/models`` through the route-change event) with owner-sized catalogs built deterministically
here (no owner data, no network):

* ``static``: the built-in list only (``model_options`` static catalog, 884 rows today);
* ``polled``: a fresh catalog cache with the owner's per-provider sizes (nanogpt 1,046, openrouter 466,
  Arena 302, ...; NanoGPT absorbs the overlap the owner's real ids have with the built-in list: 2,559 rows);
* ``desktop``: the imported desktop config (``custom_model_list`` 3,760 ids with the owner's route mix,
  about a third of them routes excluded on mobile, 120 tombstones; 3,776 rows).

Every step goes through the real Flet events the client sends (route_change, TextField change, Chip
click, Dismissible dismiss, SnackBar action, ReorderableListView reorder, button clicks, view_pop):
open, search typing and clearing, the three chips on and off, swipe-remove + Undo, drag reorder,
"Show 100 more" and the next/previous window, Poll providers (a fake, offline catalog refresh), leaving
for the Multi-Key Manager and coming back, a cold open during a slow start-up, and a failed catalog load
(Retry shown and working).

Budgets are deterministic first: bytes / controls / Dismissible rows counted per message at send time,
render counts, mounted rows. Loop-block times (a 10 ms heartbeat's largest gap) and handler times are
generous secondary bounds. Each scenario collects every number before asserting, so the same file can
measure a pre-fix tree (set ``GLOSSARION_ISSUE19_REPORT=<file.jsonl>`` to append every step's numbers).

Run from src/mobile (mobile venv):
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_devfix_issue19.py
"""

from __future__ import annotations

import asyncio
import gc
import importlib.util
import json
import os
import random
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")

# The real-app helpers (fake Flet session, fixtures, _start / _wait / _stop) live in test_ui_foundations.py.
_UF_SPEC = importlib.util.spec_from_file_location("_glossarion_uf_helpers_issue19",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
uf = importlib.util.module_from_spec(_UF_SPEC)
_UF_SPEC.loader.exec_module(uf)
storage = uf.storage
app_env = uf.app_env
_TB = uf._TB

# ---- budgets ------------------------------------------------------------------------------------------

#: UI_SPEC §7.3 first-patch budget of a screen (the real shell's route patch also carries the app bar)
FIRST_PATCH_BYTES = 250_000
FIRST_PATCH_CONTROLS = 2_500
#: rows mounted on open and after a search / filter (one LIST_STEP)
STEP_ROWS = 100
WINDOW_ROWS = 500
#: any window-sized change (a search, a chip, Show more, the end of a poll, Back)
STEP_PATCH_BYTES = 150_000
#: a one-row change (swipe, Undo, reorder) and the header while polling
SMALL_PATCH_BYTES = 5_000
#: secondary wall-clock bounds, generous for slow CI runners (host after the fix: open 0.07-0.18 s, the largest
#: loop gap of any step 0.24 s, search results 0.45-0.55 s; before it 0.6-5.7 s per step at 884-3,776 rows)
HANDLER_S = 1.0
LOOP_GAP_S = 1.0
SEARCH_VISIBLE_S = 2.0

# ---- owner-sized catalogs (deterministic, synthetic ids) ------------------------------------------------

VENDORS = ("openai", "anthropic", "google", "deepseek-ai", "qwen", "meta-llama", "z-ai", "moonshotai", "x-ai",
           "mistralai", "nvidia", "microsoft")
BASES = ("gpt-6-mini", "claude-sonnet-5", "gemini-3.5-flash", "deepseek-v4", "qwen3.8-max", "llama-5-70b",
         "glm-5.3", "kimi-k3", "grok-5", "mistral-large-3", "nemotron-4", "phi-5", "gpt-5.5-codex")

#: (cache name, dropdown prefix, models, merged into the list): the owner's catalog cache on 2026-10-09;
#: the last four only keep seven-day poll markers (``last_successful``)
OWNER_CACHE = (
    ("nanogpt", "nan/", 1046, True), ("openrouter", "or/", 466, True), ("autharena", "autharena/", 302, True),
    ("gemini", "gemini-", 48, True), ("opencode", "oc/", 43, True), ("antigravity", "antigravity/", 34, True),
    ("authnd", "authnd/", 18, True), ("authcd", "authcd/", 13, True), ("ocagy", "ocagy/", 11, True),
    ("opencode-zen", "ocz/", 9, True), ("authgpt", "authgpt/", 8, True), ("authza", "authza/", 2, True),
    ("openai", "gpt-", 141, False), ("custom:lmstudio/", "lmstudio/", 3, False),
    ("ollamapull", "ollamapull/", 7, False), ("ollama", "ollama/", 4, False),
)
#: the owner's desktop custom_model_list by route (3,760 ids; the bare ids fill up the rest)
DESKTOP_MIX = (
    ("nan/", 805), ("autharena/", 704), ("or/", 410), ("autharena1/", 105), ("autharena0/", 105), ("lr/", 57),
    ("authnd/", 52), ("eh/", 50), ("oc/", 42), ("nd/", 42), ("ollamapull/", 42), ("ollama/", 42),
    ("lmstudio/", 38), ("antigravity/", 22), ("groq/", 14), ("za/", 13), ("authgpt/", 12), ("authza/", 12),
    ("vertex/", 11), ("ocz/", 9), ("chutes/", 8), ("ocagy/", 8), ("sam/", 7), ("authgrok/", 6), ("authcd/", 6),
    ("authgem-vertex/", 6), ("search/", 2), ("authgrok1/", 1),
)
OWNER_POLLED_ROWS = 2559  # the picker list with the owner's fresh cache
DESKTOP_LIST = 3760
DESKTOP_TOMBSTONES = 120
DESKTOP_NOT_SAVED = 16  # built-in models missing from the saved list: the merge appends them (3,776 rows)
NEW_POLLED = "or/acceptance/issue19-new-model"
NEW_TOP = "acceptance/issue19-added-on-desktop"
NEW_ADDED = "acceptance/issue19-added-with-the-fab"


def _synthetic(prefix: str, count: int, tag: str) -> list:
    out = []
    for i in range(count):
        base = BASES[(i * 5 + len(prefix)) % len(BASES)]
        if prefix.endswith("/"):
            out.append(f"{prefix}{VENDORS[i % len(VENDORS)]}/{base}-{tag}{i}")
        else:  # a bare family ("gemini-", "gpt-") or an unprefixed custom id
            out.append(f"{prefix}{base}-{tag}{i}")
    return out


def _static_models(mo) -> list:
    return list(mo._deduplicate_models(mo._get_static_model_options()))


def _owner_cache(mo) -> dict:
    """The owner's catalog cache, sized so the picker list has OWNER_POLLED_ROWS rows (the owner's real
    ids overlap the built-in list a little, synthetic ones do not: NanoGPT absorbs the difference)."""
    fetched = time.time() - 3600  # polled an hour ago: inside the 24 h list TTL and the 7-day markers

    def build(sizes: dict) -> dict:
        providers: dict = {}
        last: dict = {}
        for name, prefix, _count, merged in OWNER_CACHE:
            record: dict = {"fetched_at": fetched, "models": _synthetic(prefix, sizes[name], "p")}
            variant = mo._provider_catalog_variant(name)
            if variant:
                record["variant"] = variant
            last[name] = record
            if merged:
                providers[name] = dict(record)
        return {"version": mo._MODEL_CATALOG_CACHE_VERSION, "providers": providers, "last_successful": last,
                "attempts": {}, "attempt_variants": {}, "updated_at": fetched}

    sizes = {name: count for name, _prefix, count, _merged in OWNER_CACHE}
    cache = build(sizes)
    catalogs = {name: record["models"] for name, record in cache["providers"].items()}
    merged = mo._merge_dynamic_model_options(_static_models(mo), catalogs)
    sizes["nanogpt"] = max(1, sizes["nanogpt"] + OWNER_POLLED_ROWS - len(merged))
    return build(sizes)


def _desktop_lists(static: list) -> tuple:
    """(config keys of the imported desktop config, the custom ids in it)."""
    kept = static[DESKTOP_NOT_SAVED:]
    fill = DESKTOP_LIST - len(kept)
    mix = list(DESKTOP_MIX)
    total = sum(count for _prefix, count in mix)
    if total > fill:  # a much bigger built-in list one day: keep the route mix, scaled
        mix = [(prefix, max(1, count * fill // total)) for prefix, count in mix]
        total = sum(count for _prefix, count in mix)
    custom: list = []
    for prefix, count in mix + [("", fill - total)]:
        custom.extend(_synthetic(prefix, count, "d"))
    saved = kept + custom
    random.Random(19).shuffle(saved)  # the owner's list interleaves built-in and custom routes
    return {"custom_model_list": saved,
            "model_manager_removed_models": _synthetic("retired/", DESKTOP_TOMBSTONES, "r")}, custom


# ---- harness -------------------------------------------------------------------------------------------


class Polls:
    """``model_options.refresh_provider_model_catalogs`` offline: an automatic one-provider poll answers
    offline; the explicit "Poll providers" returns the current catalog plus one new OpenRouter model."""

    def __init__(self, mo) -> None:
        self.mo = mo
        self.calls: list = []
        self.hold = threading.Event()
        self.hold.set()

    def __call__(self, **kwargs: Any) -> Any:
        only = kwargs.get("only_provider")
        self.calls.append(only)
        self.hold.wait(15)
        result = self.mo.ModelCatalogRefreshResult
        if only:
            return result(models=[], provider_models={}, statuses={only: "offline (test)"}, requested_provider=only)
        catalog = list(self.mo.get_model_options())
        openrouter = [m for m in catalog if m.lower().startswith("or/")] + [NEW_POLLED]
        return result(models=catalog + [NEW_POLLED], provider_models={"openrouter": openrouter},
                      statuses={"openrouter": f"online ({len(openrouter)} models)", "nanogpt": "missing credential"},
                      requested_provider=None)


class Harness:
    """Wire bytes / controls / Dismissible rows per message at send time, a 10 ms loop heartbeat, and
    every ModelManagerScreen render."""

    def __init__(self, conn, session, page, app, renders: list) -> None:
        self.conn, self.session, self.page, self.app = conn, session, page, app
        self.renders = renders
        self.sent: list = []
        self.gaps: list = []
        self.metrics: dict = {}
        self.problems: list = []
        self._beat: Any = None
        self._instrument()

    def _instrument(self) -> None:
        import msgpack
        from flet.controls.base_control import BaseControl
        from flet.messaging.protocol import MessageAction, configure_encode_object_for_msgpack

        encode = configure_encode_object_for_msgpack(BaseControl)
        original = self.conn.send_message

        def send_message(message):
            raw = msgpack.packb([message.action, message.body], default=encode)
            controls = rows = 0
            stack = [msgpack.unpackb(raw, strict_map_key=False)]
            while stack:
                cur = stack.pop()
                if isinstance(cur, dict):
                    if "_i" in cur and "_c" in cur:
                        controls += 1
                        rows += cur.get("_c") == "Dismissible"
                    stack.extend(cur.values())
                elif isinstance(cur, (list, tuple)):
                    stack.extend(cur)
            self.sent.append({"bytes": len(raw), "controls": controls, "rows": rows,
                              "crashed": message.action == MessageAction.SESSION_CRASHED})
            return original(message)

        self.conn.send_message = send_message

    def start_heartbeat(self) -> None:
        async def beat() -> None:
            loop = asyncio.get_running_loop()
            while True:
                tick = loop.time()
                await asyncio.sleep(0.01)
                self.gaps.append(loop.time() - tick - 0.01)

        # a full GC of a long single-process run's heap (the whole tests_host suite) stalls the loop for
        # 0.5-1.2 s by itself: collect it before the steps and keep the old heap out of later collections
        gc.collect()
        gc.freeze()
        self._beat = asyncio.ensure_future(beat())

    def stop_heartbeat(self) -> None:
        if self._beat is not None:
            self._beat.cancel()
            self._beat = None
            gc.unfreeze()

    def wire(self, mark: int) -> dict:
        items = self.sent[mark:]
        return {"messages": len(items), "bytes": sum(i["bytes"] for i in items),
                "max_msg": max((i["bytes"] for i in items), default=0),
                "controls": sum(i["controls"] for i in items), "rows_sent": sum(i["rows"] for i in items)}

    def marks(self) -> tuple:
        return len(self.sent), len(self.gaps), len(self.renders)

    async def step(self, name: str, action: Callable[[], Awaitable[Any]], *, until: Optional[Callable[[], bool]] = None,
                   timeout: float = 6.0, settle: float = 0.8, screen: Any = None) -> dict:
        """Run one user action through its real event; record what it cost until things settle."""
        await asyncio.sleep(0.03)  # the heartbeat books any gap of the previous action before the marks
        wire_mark, gap_mark, render_mark = self.marks()
        list_mark = getattr(screen, "list_renders", None)
        started = time.perf_counter()
        await action()
        handler = time.perf_counter() - started
        visible = None
        if until is not None and await uf._wait(until, timeout):
            visible = time.perf_counter() - started
        await asyncio.sleep(settle)
        renders = self.renders[render_mark:]
        metrics = dict(self.wire(wire_mark))
        metrics.update({
            "handler_s": round(handler, 3),
            "visible_s": None if visible is None else round(visible, 3),
            "loop_gap_s": round(max(self.gaps[gap_mark:], default=0.0), 3),
            "renders": len(renders),
            "render_s": round(sum(seconds for seconds in renders), 3),
        })
        if screen is not None:
            now = getattr(screen, "list_renders", None)
            # before the fix every render rebuilt the whole list
            metrics["list_renders"] = (now - list_mark) if (now is not None and list_mark is not None) else len(renders)
            metrics["mounted"] = len(_rows(screen))
        if until is not None and visible is None:
            self.problems.append(f"{name}: the expected state never appeared within {timeout} s")
        self.metrics[name] = metrics
        return metrics

    def check(self, ok: bool, message: str) -> bool:
        if not ok:
            self.problems.append(message)
        return ok

    def budget(self, name: str, *, bytes_max: Optional[int] = None, controls_max: Optional[int] = None,
               rows_sent_max: Optional[int] = None, renders_max: Optional[int] = None,
               list_renders_max: Optional[int] = None, mounted_max: Optional[int] = None,
               handler_max: Optional[float] = None, gap_max: Optional[float] = LOOP_GAP_S) -> None:
        m = self.metrics[name]
        for key, limit in (("bytes", bytes_max), ("controls", controls_max), ("rows_sent", rows_sent_max),
                           ("renders", renders_max), ("list_renders", list_renders_max), ("mounted", mounted_max),
                           ("handler_s", handler_max), ("loop_gap_s", gap_max)):
            if limit is not None and m.get(key) is not None and m[key] > limit:
                self.problems.append(f"{name}: {key} {m[key]} > {limit}")


def _rows(screen) -> list:
    view = getattr(screen, "list_view", None)
    return [c for c in list(getattr(view, "controls", None) or []) if type(c).__name__ == "Dismissible"]


def _row_models(screen) -> list:
    out = []
    for control in _rows(screen):
        key = str(getattr(control, "key", "") or "")
        model = key[3:] if key.startswith("mm-") else key
        head, sep, tail = model.rpartition("#")
        out.append(head if sep and tail.isdigit() else model)
    return out


def _visible(root) -> list:
    out: list = []
    stack = [root]
    while stack:
        control = stack.pop()
        if control is None or getattr(control, "visible", True) is False:
            continue
        out.append(control)
        for attr in ("content", "controls", "leading", "trailing", "title", "subtitle", "footer"):
            child = getattr(control, attr, None)
            if isinstance(child, list):
                stack.extend(c for c in child if hasattr(c, "_i"))
            elif child is not None and hasattr(child, "_i"):
                stack.append(child)
    return out


def _texts(root) -> list:
    out = []
    for control in _visible(root):
        for attr in ("value", "content", "label"):  # Text, buttons, Semantics (the Skeleton's label)
            value = getattr(control, attr, None)
            if isinstance(value, str) and value:
                out.append(value)
    return out


def _retry_buttons(screen) -> list:
    return [c for c in _visible(screen.body) if "Button" in type(c).__name__ and getattr(c, "content", None) == "Retry"]


def _undo_bar(page) -> Any:
    bars = [c for c in list(page._dialogs.controls) if type(c).__name__ == "SnackBar"
            and str(getattr(c, "action", "") or "") == "Undo" and getattr(c, "open", False)]
    return bars[-1] if bars else None


def _digest(models) -> str:
    """The saved list's identity (a pre-fix and a fixed tree must end every edit with the same list)."""
    import hashlib

    return f"{len(models)}:" + hashlib.sha1("\n".join(map(str, models)).encode("utf-8")).hexdigest()[:12]


def _hint(screen) -> str:
    return str(getattr(getattr(screen, "hint", None), "value", "") or "")


def _report(kind: str, h: Harness, extra: Optional[dict] = None) -> None:
    record = {"kind": kind, "problems": list(h.problems), "metrics": h.metrics, **(extra or {})}
    path = os.environ.get("GLOSSARION_ISSUE19_REPORT")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    print(f"\n[issue19 {kind}] " + json.dumps(h.metrics, separators=(",", ":")))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No traffic beyond loopback (the fake client and the app's own loopback sockets)."""
    attempts: list = []
    loopback = ("127.0.0.1", "::1", "localhost")
    real_connect = socket.socket.connect
    real_create = socket.create_connection

    def connect(self, address):
        if isinstance(address, tuple) and address and address[0] not in loopback:
            attempts.append(str(address))
            raise OSError(f"network disabled in tests ({address})")
        return real_connect(self, address)

    def create_connection(address, *args, **kwargs):
        if address and address[0] not in loopback:
            attempts.append(str(address))
            raise OSError(f"network disabled in tests ({address})")
        return real_create(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    return attempts


def _prepare(kind: str, monkeypatch) -> dict:
    """Write the owner-sized config / catalog cache before the app starts; fake the catalog polls."""
    import model_options as mo

    from glossarion_mobile.ui.screens import model_manager as mm

    monkeypatch.setattr(mo, "_MODEL_CATALOG_MEMORY_CACHE", None)
    static = _static_models(mo)
    config: dict = {"model": "authgpt/gpt-6-luna"}
    expected_custom = 0
    removed = 0
    cache_path = os.environ["GLOSSARION_MODEL_CATALOG_CACHE"]
    if kind == "polled":
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(_owner_cache(mo), fh)
        expected = len(mo.get_model_options())
        monkeypatch.setattr(mo, "_MODEL_CATALOG_MEMORY_CACHE", None)
    elif kind == "desktop":
        lists, custom = _desktop_lists(static)
        config.update(lists)
        expected_custom = len(custom)
        removed = DESKTOP_TOMBSTONES
        expected = len(lists["custom_model_list"]) + DESKTOP_NOT_SAVED
    else:
        expected = len(static)
    os.makedirs(os.path.dirname(os.environ["CONFIG_FILE"]), exist_ok=True)
    with open(os.environ["CONFIG_FILE"], "w", encoding="utf-8") as fh:
        json.dump(config, fh)
    polls = Polls(mo)
    monkeypatch.setattr(mo, "refresh_provider_model_catalogs", polls)
    renders: list = []
    original = mm.ModelManagerScreen.render

    def render(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            renders.append(time.perf_counter() - started)

    monkeypatch.setattr(mm.ModelManagerScreen, "render", render)
    return {"mo": mo, "mm": mm, "static": static, "expected": expected, "custom": expected_custom,
            "removed": removed, "polls": polls, "renders": renders}


async def _start(prep: dict) -> Harness:
    _main, conn, session, page, app = await uf._start("android")
    h = Harness(conn, session, page, app, prep["renders"])
    h.start_heartbeat()
    feature = getattr(app, "models_keys", None)
    assert feature is not None, "ModelsKeysFeature was not installed"
    assert await uf._wait(lambda: feature.catalog.snapshot.loaded, 30), "the install's catalog load never finished"
    await asyncio.sleep(0.5)
    return h


async def _teardown(h: Optional[Harness]) -> None:
    from glossarion_mobile.services import model_catalog as mc
    from glossarion_mobile.ui.sheets import model_sheet as ms

    if h is None:
        return
    h.stop_heartbeat()
    ms.install_sheet_env(None)
    mc.set_default_service(None)
    await uf._stop(h.app)


# ---- the scenario ------------------------------------------------------------------------------------


async def _scenario(kind: str, prep: dict) -> Harness:
    h = await _start(prep)
    app, session, page = h.app, h.session, h.page
    catalog = app.models_keys.catalog
    store = app.config_store
    mm = prep["mm"]
    total = len(catalog.snapshot.models)
    h.metrics["models"] = total
    h.check(total == prep["expected"], f"catalog: {total} rows, expected {prep['expected']}")
    first_rows = min(STEP_ROWS, total)

    # 1. open: one window, one small first patch, the loop free again at once
    await h.step("open", lambda: _TB._route(session, "/settings/models"), settle=1.2)
    screen = app.shell.top_screen
    if not h.check(isinstance(screen, mm.ModelManagerScreen), f"open: top screen is {type(screen).__name__}"):
        return h
    m = h.metrics["open"]
    m["mounted"] = len(_rows(screen))
    h.check(m["mounted"] == first_rows, f"open: {m['mounted']} rows mounted, expected {first_rows}")
    h.check(_hint(screen).startswith(f"{total} models"), f"open: hint {_hint(screen)!r}")
    h.budget("open", bytes_max=FIRST_PATCH_BYTES, controls_max=FIRST_PATCH_CONTROLS, rows_sent_max=STEP_ROWS,
             handler_max=HANDLER_S)
    # routes excluded on mobile keep their "Not on mobile" reason on the row
    from glossarion_mobile.services import model_catalog as mc

    excluded = [bool(mc.excluded_route(model)) for model in _row_models(screen)]
    chips = [any(type(c).__name__ == "ReasonChip" for c in _visible(row)) for row in _rows(screen)]
    m["excluded_rows"] = sum(excluded)
    h.check(excluded == chips, "open: an excluded route lost its 'Not on mobile' chip (or another row got one)")

    # 2. search: three quick keystrokes, then clear
    async def type_query() -> None:
        for text in ("g", "gp", "gpt"):
            session.apply_patch(screen.search._i, {"value": text})
            await session.dispatch_event(screen.search._i, "change", text)
            await asyncio.sleep(0.08)

    def searched() -> bool:
        models = _row_models(screen)
        return bool(models) and all("gpt" in model.casefold() for model in models) and "shown" in _hint(screen)

    await h.step("search", type_query, until=searched, settle=0.6, screen=screen)
    h.check(h.metrics["search"]["visible_s"] is None or h.metrics["search"]["visible_s"] <= SEARCH_VISIBLE_S,
            f"search: results took {h.metrics['search']['visible_s']} s")
    h.budget("search", bytes_max=STEP_PATCH_BYTES, renders_max=1, mounted_max=STEP_ROWS)

    async def clear() -> None:
        session.apply_patch(screen.search._i, {"value": ""})
        await session.dispatch_event(screen.search._i, "change", "")

    await h.step("search_clear", clear, until=lambda: _hint(screen).startswith(f"{total} models"), settle=0.6,
                 screen=screen)
    h.check(h.metrics["search_clear"]["mounted"] == first_rows,
            f"search_clear: {h.metrics['search_clear']['mounted']} rows mounted")
    h.budget("search_clear", bytes_max=STEP_PATCH_BYTES, renders_max=1, mounted_max=STEP_ROWS)

    # 3. chips on and off (Polled only writes the config key: the 0.3 s reload after it must not repaint)
    polled_total = None  # the Polled only rows, read after the click
    for chip_name, attr, check in (
            ("chip_polled", "polled_chip", lambda: screen.polled_chip.selected is True),
            ("chip_custom", "custom_chip", lambda: screen.custom_chip.selected is True),
            ("chip_removed", "removed_chip", lambda: screen.removed_chip.selected is True)):
        chip = getattr(screen, attr)
        await h.step(f"{chip_name}_on", lambda c=chip: session.dispatch_event(c._i, "click", None), until=check,
                     settle=1.0, screen=screen)
        hint = _hint(screen)
        h.metrics[f"{chip_name}_on"]["hint"] = hint
        if chip_name == "chip_polled":
            polled_total = len(catalog.snapshot.visible_models())
            h.check(hint.startswith(f"{polled_total} shown"), f"{chip_name}_on: hint {hint!r}, polled {polled_total}")
        elif chip_name == "chip_custom":
            h.check(hint.startswith(f"{prep['custom']} shown"),
                    f"{chip_name}_on: hint {hint!r}, expected {prep['custom']} custom models (never every model)")
        else:
            h.check(hint.startswith(f"{prep['removed']} removed"), f"{chip_name}_on: hint {hint!r}")
        h.budget(f"{chip_name}_on", bytes_max=STEP_PATCH_BYTES, renders_max=1, mounted_max=STEP_ROWS)
        await h.step(f"{chip_name}_off", lambda c=chip: session.dispatch_event(c._i, "click", None),
                     until=lambda: _hint(screen).startswith(f"{total} models"), settle=1.0, screen=screen)
        h.budget(f"{chip_name}_off", bytes_max=STEP_PATCH_BYTES, renders_max=1, mounted_max=STEP_ROWS)
    h.metrics["polled_only_rows"] = polled_total

    # 4. swipe-remove a row, then Undo from the SnackBar
    victim = _rows(screen)[4]
    model = _row_models(screen)[4]
    before = list(catalog.snapshot.models)
    h.check(before[4] == model, f"swipe: row 4 shows {model!r}, the list has {before[4]!r}")
    gone: dict = {}

    async def dismiss() -> None:
        await session.dispatch_event(victim._i, "dismiss", {"direction": "endToStart"})
        gone["sync"] = victim not in screen.list_view.controls

    await h.step("swipe_remove", dismiss, until=lambda: model not in catalog.snapshot.models, settle=1.0,
                 screen=screen)
    h.check(gone.get("sync") is True, "swipe_remove: the dismissed row was still in the list after the event")
    h.check(model in (store.get("model_manager_removed_models") or []), "swipe_remove: no tombstone saved")
    h.check(model not in _row_models(screen), "swipe_remove: the row came back")
    h.check(_hint(screen).startswith(f"{total - 1} models"), f"swipe_remove: hint {_hint(screen)!r}")
    h.budget("swipe_remove", bytes_max=SMALL_PATCH_BYTES, rows_sent_max=1, renders_max=1)
    bar = _undo_bar(page)
    if h.check(bar is not None, "swipe_remove: no Undo SnackBar"):
        await h.step("undo", lambda: session.dispatch_event(bar._i, "action", None),
                     until=lambda: model in catalog.snapshot.models and model in _row_models(screen), settle=1.0,
                     screen=screen)
        h.check(list(catalog.snapshot.models)[:6] == before[:6], "undo: the model did not return to its place")
        h.check(_row_models(screen)[4:5] == [model], f"undo: row 4 is {_row_models(screen)[4:5]}")
        h.check(model not in (store.get("model_manager_removed_models") or []), "undo: the tombstone stayed")
        h.budget("undo", bytes_max=SMALL_PATCH_BYTES, rows_sent_max=1, renders_max=1)

    # 5. drag reorder in the first window (the client already moved row 3 to the top)
    order = list(catalog.snapshot.models)
    moved = order[3]
    await h.step("reorder", lambda: session.dispatch_event(screen.list_view._i, "reorder",
                                                           {"old_index": 3, "new_index": 0}),
                 until=lambda: list(catalog.snapshot.models)[:1] == [moved], settle=1.0, screen=screen)
    h.check(list(catalog.snapshot.models) == [moved] + order[:3] + order[4:], "reorder: saved order is wrong")
    h.check((store.get("custom_model_list") or [None])[0] == moved, "reorder: custom_model_list not saved")
    h.check(_row_models(screen)[:1] == [moved], f"reorder: row 0 is {_row_models(screen)[:1]}")
    h.budget("reorder", bytes_max=SMALL_PATCH_BYTES, rows_sent_max=1, renders_max=1)

    # 5b. the row's trash button, then Removed › swipe the first tombstone to restore it
    trashed = _row_models(screen)[2]
    trash = [c for c in _visible(_rows(screen)[2]) if type(c).__name__ == "IconButton"
             and getattr(c, "tooltip", None) == "Remove"]
    if h.check(len(trash) == 1, "remove_button: no Remove button on the row"):
        await h.step("remove_button", lambda: session.dispatch_event(trash[0]._i, "click", None),
                     until=lambda: trashed not in catalog.snapshot.models and trashed not in _row_models(screen),
                     settle=1.0, screen=screen)
        h.check(trashed in (store.get("model_manager_removed_models") or []), "remove_button: no tombstone saved")
        h.budget("remove_button", bytes_max=SMALL_PATCH_BYTES, rows_sent_max=1, renders_max=1)
    await h.step("removed_view", lambda: session.dispatch_event(screen.removed_chip._i, "click", None),
                 until=lambda: bool(_rows(screen)) and _hint(screen).endswith("swipe to restore"), settle=0.8,
                 screen=screen)
    h.budget("removed_view", bytes_max=STEP_PATCH_BYTES, renders_max=1, mounted_max=STEP_ROWS)
    tombstones = list(store.get("model_manager_removed_models") or [])
    if _rows(screen):
        back_model = _row_models(screen)[0]
        h.metrics["restored_model_is_trashed"] = back_model == trashed
        restore_row = _rows(screen)[0]
        await h.step("restore", lambda: session.dispatch_event(restore_row._i, "dismiss", {"direction": "endToStart"}),
                     until=lambda: back_model in catalog.snapshot.models and back_model not in _row_models(screen),
                     settle=1.0, screen=screen)
        h.check(back_model not in (store.get("model_manager_removed_models") or []), "restore: the tombstone stayed")
        h.check(_hint(screen).startswith(f"{len(tombstones) - 1} removed"), f"restore: hint {_hint(screen)!r}")
        h.budget("restore", bytes_max=SMALL_PATCH_BYTES, rows_sent_max=1, renders_max=1)
    await h.step("removed_view_off", lambda: session.dispatch_event(screen.removed_chip._i, "click", None),
                 until=lambda: _hint(screen).endswith("swipe to remove"), settle=0.8, screen=screen)
    h.budget("removed_view_off", bytes_max=STEP_PATCH_BYTES, renders_max=1, mounted_max=STEP_ROWS)

    # 5c. FAB "Add model": the dialog, then ➕ Add puts it at the top
    await session.dispatch_event(screen.add_fab._i, "click", None)
    dialog, field = getattr(screen, "last_dialog", None), getattr(screen, "add_field", None)
    if h.check(dialog is not None and field is not None and dialog in page._dialogs.controls, "add: no Add dialog"):
        add_button = dialog.actions[-1]
        session.apply_patch(field._i, {"value": NEW_ADDED})
        await h.step("add", lambda: session.dispatch_event(add_button._i, "click", None),
                     until=lambda: list(catalog.snapshot.models)[:1] == [NEW_ADDED]
                     and _row_models(screen)[:1] == [NEW_ADDED], settle=1.0, screen=screen)
        h.check((store.get("custom_model_list") or [None])[0] == NEW_ADDED, "add: custom_model_list not saved")
        h.budget("add", bytes_max=SMALL_PATCH_BYTES, rows_sent_max=1, renders_max=1)

    h.metrics["digest_after_edits"] = _digest(catalog.snapshot.models)
    total = len(catalog.snapshot.models)

    # 6. "Show 100 more", and on the desktop list the next window, a reorder there and back
    rows = getattr(screen, "rows", None)
    more = getattr(rows, "more_button", None)
    if h.check(more is not None, "show_more: the list has no 'Show 100 more' footer (every row is mounted)"):
        h.metrics["more_label"] = more.content
        h.check(more.content == f"Show 100 more ({total - 100:,} left)", f"show_more: label {more.content!r}")
        steps = (WINDOW_ROWS // STEP_ROWS - 1) if kind == "desktop" else 1
        for n in range(1, steps + 1):
            name = "show_more" if n == 1 else f"show_more_{n}"
            mounted = min((n + 1) * STEP_ROWS, total)
            await h.step(name, lambda: session.dispatch_event(more._i, "click", None),
                         until=lambda m=mounted: len(_rows(screen)) == m, settle=0.3, screen=screen)
            h.budget(name, bytes_max=STEP_PATCH_BYTES, rows_sent_max=STEP_ROWS, renders_max=0)
        if kind == "desktop":
            h.check(len(_rows(screen)) == WINDOW_ROWS, f"window: {len(_rows(screen))} rows mounted, never > 500")
            h.check(more.content == f"Show rows 501–1,000 ({total - WINDOW_ROWS:,} left)",
                    f"window: footer {more.content!r}")
            order = list(catalog.snapshot.models)
            await h.step("next_window", lambda: session.dispatch_event(more._i, "click", None),
                         until=lambda: rows.window_start == WINDOW_ROWS, settle=0.5, screen=screen)
            h.check(rows.window_text.value == f"Rows 501–1,000 of {total:,}" and rows.selector.visible,
                    f"next_window: selector {rows.window_text.value!r}")
            h.check(_row_models(screen)[:1] == [order[500]], "next_window: row 0 is not model 501")
            h.budget("next_window", bytes_max=STEP_PATCH_BYTES, rows_sent_max=STEP_ROWS)
            await h.step("reorder_window2", lambda: session.dispatch_event(screen.list_view._i, "reorder",
                                                                           {"old_index": 3, "new_index": 0}),
                         until=lambda: list(catalog.snapshot.models)[500:501] == [order[503]], settle=1.0,
                         screen=screen)
            h.check(list(catalog.snapshot.models) == order[:500] + [order[503]] + order[500:503] + order[504:],
                    "reorder_window2: window-relative indices were not mapped to the list (move_model(503, 500))")
            h.budget("reorder_window2", bytes_max=SMALL_PATCH_BYTES, rows_sent_max=1, renders_max=1)
            await h.step("previous_window", lambda: session.dispatch_event(rows.prev_button._i, "click", None),
                         until=lambda: rows.window_start == 0, settle=0.5, screen=screen)
            h.budget("previous_window", bytes_max=STEP_PATCH_BYTES, rows_sent_max=STEP_ROWS)
    elif kind == "desktop":  # a pre-fix tree mounts every row: the same drag (row 504 to 501) on the full list
        order = list(catalog.snapshot.models)
        await h.step("reorder_window2", lambda: session.dispatch_event(screen.list_view._i, "reorder",
                                                                       {"old_index": 503, "new_index": 500}),
                     until=lambda: list(catalog.snapshot.models)[500:501] == [order[503]], settle=1.0, screen=screen)
    h.metrics["digest_after_reorders"] = _digest(catalog.snapshot.models)

    # 7. Poll providers: only the header while it runs, the list once at the end
    polls = prep["polls"]
    polls.hold.clear()
    header: dict = {}

    async def poll() -> None:
        mark = len(h.sent)
        await session.dispatch_event(screen.poll_button._i, "click", None)
        await uf._wait(lambda: screen.poll_button.content == "⏳ Polling…", 3)
        await asyncio.sleep(0.3)
        header.update(h.wire(mark))
        header["polling_shown"] = screen.poll_button.content == "⏳ Polling…" and bool(screen.poll_button.disabled)
        polls.hold.set()

    await h.step("poll", poll, until=lambda: screen.poll_button.content == "🌐 Poll providers"
                 and not screen.poll_button.disabled and NEW_POLLED in catalog.snapshot.models, settle=1.0,
                 timeout=10, screen=screen)
    polls.hold.set()
    h.metrics["poll_header"] = header
    h.metrics["poll"]["poll_text"] = screen.poll_text.value
    h.metrics["poll"]["models_after"] = len(catalog.snapshot.models)
    h.metrics["digest_after_poll"] = _digest(catalog.snapshot.models)
    h.check(header.get("polling_shown") is True, "poll: '⏳ Polling…' never showed")
    h.check(header.get("rows_sent", 1) == 0 and header.get("bytes", SMALL_PATCH_BYTES) < SMALL_PATCH_BYTES,
            f"poll: the header update while polling re-sent rows {header}")
    h.check(h.metrics["poll"]["list_renders"] == 1, f"poll: list updated {h.metrics['poll']['list_renders']} times")
    h.check(None in polls.calls, "poll: the explicit poll never ran")
    # header-only renders while it runs are cheap (one per publish: polling on, statuses, polling off, the
    # saved list); the list itself is synced once
    h.budget("poll", bytes_max=STEP_PATCH_BYTES, mounted_max=WINDOW_ROWS)

    # 7b. the Custom prefixes tab and back: the model rows stay mounted, nothing is rebuilt
    async def tab(name: str) -> None:
        session.apply_patch(screen.tabs._i, {"selected": [name]})
        await session.dispatch_event(screen.tabs._i, "change", None)

    mounted_before = list(_rows(screen))
    await h.step("tab_prefixes", lambda: tab("prefixes"), until=lambda: screen.prefix_list.visible is True,
                 settle=0.5, screen=screen)
    h.budget("tab_prefixes", bytes_max=STEP_PATCH_BYTES, rows_sent_max=0, renders_max=1)
    await h.step("tab_models", lambda: tab("models"), until=lambda: screen.prefix_list.visible is False,
                 settle=0.5, screen=screen)
    h.check(all(a is b for a, b in zip(mounted_before, _rows(screen))) and len(_rows(screen)) == len(mounted_before),
            "tab_models: the model rows were rebuilt")
    h.budget("tab_models", bytes_max=STEP_PATCH_BYTES, rows_sent_max=0, renders_max=1)

    # 8. leave for the Multi-Key Manager (the manager stays below), change the list there, come back
    await h.step("to_keys", lambda: _TB._route(session, "/settings/keys"), settle=0.5)
    h.metrics["stack"] = [entry.route for entry in app.shell.stack]
    h.check(any(entry.screen is screen for entry in app.shell.stack) and app.shell.top_screen is not screen,
            f"to_keys: stack {h.metrics['stack']}")
    current = list(catalog.snapshot.models)

    async def edit_while_hidden() -> None:
        store.set("custom_model_list", [NEW_TOP] + current)

    await h.step("hidden_publish", edit_while_hidden,
                 until=lambda: list(catalog.snapshot.models)[:1] == [NEW_TOP], settle=0.8, screen=screen)
    h.check(h.metrics["hidden_publish"]["renders"] == 0,
            f"hidden_publish: {h.metrics['hidden_publish']['renders']} renders while covered")
    await h.step("back", lambda: session.dispatch_event(page._i, "view_pop", {"route": "/settings/keys"}),
                 until=lambda: app.shell.top_screen is screen and _row_models(screen)[:1] == [NEW_TOP], settle=0.8,
                 screen=screen)
    h.check(h.metrics["back"]["renders"] == 1, f"back: {h.metrics['back']['renders']} renders (expected 1)")
    h.budget("back", bytes_max=STEP_PATCH_BYTES, mounted_max=WINDOW_ROWS, handler_max=HANDLER_S)
    h.metrics["digest_final"] = _digest(catalog.snapshot.models)
    h.metrics["removed_final"] = _digest(store.get("model_manager_removed_models") or [])

    h.metrics["crashed"] = sum(1 for item in h.sent if item["crashed"])
    h.check(h.metrics["crashed"] == 0, "an event handler crashed the session")
    return h


@needs_flet
@pytest.mark.parametrize("kind", ["static", "polled", "desktop"])
def test_model_manager_stays_responsive_on_the_real_shell(kind, app_env, monkeypatch, offline):
    prep = _prepare(kind, monkeypatch)

    async def run() -> Harness:
        h = None
        try:
            h = await _scenario(kind, prep)
            return h
        finally:
            await _teardown(h)

    h = asyncio.run(run())
    _report(kind, h, {"network_attempts": list(offline)})
    assert not h.problems, "\n".join(h.problems) + "\n" + json.dumps(h.metrics, indent=1)


@needs_flet
def test_cold_open_during_start_up_reads_the_catalog_once_and_paints_the_list_once(app_env, monkeypatch, offline):
    """Opened while the install's catalog load is still running (a slow phone start): the screen shows
    its loading state, shares that load, and paints the list once when it lands."""
    prep = _prepare("desktop", monkeypatch)
    from glossarion_mobile.services import model_catalog as mc

    gate = threading.Event()
    loads: list = []
    original = mc.ModelCatalogService.load_blocking

    def load_blocking(self):
        loads.append(threading.current_thread().name)
        if threading.current_thread() is not threading.main_thread():
            gate.wait(20)
        return original(self)

    monkeypatch.setattr(mc.ModelCatalogService, "load_blocking", load_blocking)

    async def run() -> Harness:
        h = None
        try:
            _main, conn, session, page, app = await uf._start("android")
            h = Harness(conn, session, page, app, prep["renders"])
            h.start_heartbeat()
            catalog = app.models_keys.catalog
            h.check(not catalog.snapshot.loaded, "cold: the catalog was loaded before the gate opened")
            await h.step("cold_open", lambda: _TB._route(session, "/settings/models"), settle=0.5)
            screen = app.shell.top_screen
            if not h.check(isinstance(screen, prep["mm"].ModelManagerScreen), "cold: not on the Model manager"):
                return h
            texts = _texts(screen.body)
            h.metrics["cold_open"]["texts"] = [t for t in texts if "oad" in t or "match" in t][:4]
            h.metrics["cold_open"]["mounted"] = len(_rows(screen))
            h.check(any(t.startswith("Loading") for t in texts) and "No models match." not in texts,
                    f"cold: no loading state while the catalog loads ({h.metrics['cold_open']['texts']})")
            h.budget("cold_open", bytes_max=FIRST_PATCH_BYTES, controls_max=FIRST_PATCH_CONTROLS,
                     handler_max=HANDLER_S)

            async def release() -> None:
                gate.set()

            await h.step("cold_loaded", release, until=lambda: len(_rows(screen)) == STEP_ROWS, timeout=20,
                         settle=1.5, screen=screen)
            h.metrics["loads"] = len(loads)
            h.check(len(loads) == 1, f"cold: the catalog was loaded {len(loads)} times (expected 1, shared)")
            h.check(h.metrics["cold_loaded"]["list_renders"] == 1,
                    f"cold: the list was painted {h.metrics['cold_loaded']['list_renders']} times")
            h.check(len(catalog.snapshot.models) == prep["expected"], "cold: wrong row count after the load")
            h.budget("cold_loaded", bytes_max=FIRST_PATCH_BYTES, rows_sent_max=STEP_ROWS, renders_max=2,
                     mounted_max=STEP_ROWS)
            return h
        finally:
            gate.set()
            await _teardown(h)

    h = asyncio.run(run())
    _report("cold", h, {"network_attempts": list(offline)})
    assert not h.problems, "\n".join(h.problems) + "\n" + json.dumps(h.metrics, indent=1)


@needs_flet
def test_a_failed_catalog_load_shows_retry_and_retry_loads_the_list(app_env, monkeypatch, offline):
    prep = _prepare("desktop", monkeypatch)
    mo = prep["mo"]
    real = mo.get_model_options

    def unreadable():
        raise RuntimeError("model catalog cache is unreadable")

    monkeypatch.setattr(mo, "get_model_options", unreadable)

    async def run() -> Harness:
        h = None
        try:
            h = await _start(prep)
            app, session = h.app, h.session
            catalog = app.models_keys.catalog
            await h.step("failed_open", lambda: _TB._route(session, "/settings/models"), settle=1.0)
            screen = app.shell.top_screen
            if not h.check(isinstance(screen, prep["mm"].ModelManagerScreen), "failed: not on the Model manager"):
                return h
            texts = _texts(screen.body)
            h.metrics["failed_open"]["texts"] = [t for t in texts if "oad" in t or "match" in t][:4]
            h.check(any("model catalog cache is unreadable" in t for t in texts),
                    "failed: the load error is not shown")
            h.check(not any(t.startswith("Loading") or t.strip() == "No models match." for t in texts),
                    "failed: shows 'Loading…' / 'No models match.' instead of the error")
            buttons = _retry_buttons(screen)
            if h.check(len(buttons) == 1, f"failed: {len(buttons)} visible Retry buttons"):
                monkeypatch.setattr(mo, "get_model_options", real)
                await h.step("retry", lambda: session.dispatch_event(buttons[0]._i, "click", None),
                             until=lambda: len(_rows(screen)) == STEP_ROWS, timeout=10, settle=0.8, screen=screen)
                h.check(not _retry_buttons(screen), "retry: the Retry button is still shown")
                h.check(len(catalog.snapshot.models) == prep["expected"] and catalog.snapshot.error is None,
                        "retry: the list did not load")
                h.budget("retry", bytes_max=FIRST_PATCH_BYTES, rows_sent_max=STEP_ROWS, mounted_max=STEP_ROWS)
            return h
        finally:
            await _teardown(h)

    h = asyncio.run(run())
    _report("failed_load", h, {"network_attempts": list(offline)})
    assert not h.problems, "\n".join(h.problems) + "\n" + json.dumps(h.metrics, indent=1)
