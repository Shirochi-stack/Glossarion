"""Acceptance test, hotfix 3 owner issue 1: "The close menu button and slide to close is not working in some
edge cases, I have to backspace on my android".

What the owner saw on the phone, replayed on the REAL app (``main.main`` on the fake Android session with
the real ChatFeature, JobsFeature, Library, Reader, sheets and shell) against ``PhoneClient``: a model of the
Flet 1.0.3 Flutter client built ONLY from what Python actually sent over the wire. Every message the fake
transport encodes is decoded and applied to a client-side control tree exactly like ``Control.applyPatch``
(packages/flet/lib/src/models/control.dart, v1.0.3); nothing is read back from the Python objects. So a close
that Python decided but never sent, a button whose handler Python already forgot, or a sheet the client still
shows on another screen fails here like on the phone.

Client rules modelled (flet v1.0.3 Dart sources: controls/bottom_sheet.dart, alert_dialog.dart, view.dart,
page.dart, routing/router_delegate.dart, utils/misc.dart closeModalRoute, models/control.dart):

* Navigator pages are the page's Views keyed by route (``ValueKey(route)``); a View the client popped is
  hidden until Python drops it (``_pendingPoppedViewRoutes``). A page that leaves ``pages`` takes the pageless
  modal routes pushed above it along (they complete: ``open: false`` + ``dismiss`` go to Python).
* Dialogs (sheets, alert dialogs) are built only inside the View whose id is the LAST of the page's views
  (view.dart), keyed by control id, inside a ``ViewControl`` keyed by the View's id: another top View means
  new dialog States. A State keeps the modal route it pushed; ``open: true`` with ``_open`` unset pushes a
  route over the top page after the frame (``enableDrag`` = ``draggable`` read then); ``open: false`` closes
  that State's route only (``closeModalRoute``) - a State without one closes nothing. ``_open`` lives on the
  client Control (a freshly sent control dict starts without it).
* A modal route shows the ``content`` the sheet had when it opened; it is rebuilt with the current one only
  when the route's status changes (``ModalRoute.of``: a route pushed above it / popped).
* A completed route sends ``open: false`` to Python (only while its id is known to the client) and
  ``dismiss`` (only when the Control it opened with has ``on_dismiss``), a few frames later (exit animation).
* Android back (``SimpleRouterDelegate.popRoute`` -> ``Page._handleSystemPopRoute``): with more than one View
  and a top View that can pop and has no ``on_confirm_pop``, the top View is marked popped (``view_pop``) and
  leaves with its modal routes; otherwise ``Navigator.maybePop``: the topmost route - a modal route, or the top
  page (``can_pop=False`` -> ``confirm_pop`` to Python; ``confirm_pop(true)`` closes it). The root page alone
  leaves the app (``SystemNavigator.pop``).
* Taps reach only what the client shows on top (the top modal route's content, else the top page and the page
  overlay), and only enabled controls; a tapped id Python no longer knows does nothing.
* ``invoke_method`` on a control whose widget has no method listener yet waits until a frame built it
  (control.dart ``invokeMethod``; ScrollableControl, which column.dart wraps every Column in, registers its
  listener in initState). Modelled for the views guard's frame probe (``dialogs.FRAME_PROBE_METHOD``, a Column
  in the page overlay, which is built inside the last View only, like the dialogs); other calls are answered at
  once, as by the fake transport.

Frames: ``burst`` runs a frame after Python's whole burst (a fast phone); ``per_message`` runs a frame after
every message Python sends (a slow phone: a vsync between two messages of one handler, e.g. between a
sheet's close and the next screen that took 100 ms to build). Both are what a phone does.

Every sheet type is opened the way the owner opens it (the button / row / chip on screen), then closed with
"Close Menu ✕" and, after opening it again, with a swipe down; each time the client shows no modal route
anywhere (no ghost sheet under any screen), the app's own state no longer thinks it is open, and Android back
afterwards pops the screen (or leaves the app on the chat) instead of revealing a ghost sheet. The floating job
bar ("Translating · <book> · Ch n/m") is visible throughout (a held chat EPUB job, as in the owner's
screenshot). Two openers are called directly instead of tapped: Help (the drawer row's handler ``app._on_help``,
the drawer itself is not modelled) and the "sheet shown with a screen change" cases (one handler pass doing
``InfoSheet(...).show(page)`` and ``app.navigate_to(...)``, in both orders).

Test rule of the hotfix batch: RUNNING is driven with a hold gate (the job run waits for the test's release),
never with sleeps.

Real data stays untouched: HOME, USERPROFILE, APPDATA, LOCALAPPDATA, GLOSSARION_LIBRARY_DIR, OUTPUT_DIRECTORY,
GLOSSARION_DATA_DIR and CONFIG_FILE point into tmp, GLOSSARION_HTTP_LOG=0, and src/config.json must be
byte-identical afterwards.

Run from src/mobile with the mobile venv (Python 3.12, keep the shell PYTHONPATH)::

    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic \
        tests_host/test_hotfix3_issue1.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import importlib.util
import queue
import shutil
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Callable, Optional

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


pytestmark = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")


def _load(name: str, filename: str):
    """A sibling test module as a helper module (one copy of the fake session, backend and fixtures)."""
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


_TB = _load("_glossarion_hf3_issue1_tb_helpers", "test_bootstrap.py")
storage = _TB.storage
app_env = _TB.app_env


def _uf():
    return _load("_glossarion_hf3_issue1_uf_helpers", "test_ui_foundations.py")


def _tj():
    return _load("_glossarion_hf3_issue1_jobs_helpers", "test_jobs.py")


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
    """Nothing reads or writes the developer's Library, output folders, home, AppData or src/config.json
    (autouse: runs before ``storage`` / the bootstrap env contract)."""
    iso = tmp_path / "_iso"
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("APPDATA", "appdata"),
                      ("LOCALAPPDATA", "localappdata"), ("GLOSSARION_LIBRARY_DIR", "Library"),
                      ("OUTPUT_DIRECTORY", "Output"), ("GLOSSARION_DATA_DIR", "data")):
        folder = iso / sub
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    monkeypatch.setenv("CONFIG_FILE", str(iso / "config.json"))  # never src/config.json
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    # the desktop "app dir" (src/ on Windows: the developer's src/Glossary, Direct Text, ...) -> scratch; the
    # Glossaries screen lists <app dir>/Glossary
    app_dir = iso / "appdir"
    app_dir.mkdir(parents=True, exist_ok=True)
    try:
        import app_paths
    except Exception:
        app_paths = None
    if app_paths is not None:
        monkeypatch.setattr(app_paths, "_get_app_dir", lambda: str(app_dir))
    config = SRC_DIR / "config.json"
    before = _md5(config)
    yield iso
    assert _md5(config) == before, "src/config.json changed during the test"


@pytest.fixture(autouse=True)
def offline_catalogs(app_env, monkeypatch):
    """The app's provider-catalog auto-poll answers offline. The real one waits on the ChatGPT web page
    (``authchatgpt_auth.list_models`` through the WebView bridge), which the fake session never answers: its
    'gl-models-io' thread outlived the test and the storage fixture's join waited 15 s for it every time."""
    import model_options as mo

    def offline(**kwargs: Any) -> Any:
        only = kwargs.get("only_provider")
        if only:
            return mo.ModelCatalogRefreshResult(models=[], provider_models={}, statuses={only: "offline (test)"},
                                                requested_provider=only)
        return mo.ModelCatalogRefreshResult(models=list(mo.get_model_options()), provider_models={}, statuses={},
                                            requested_provider=None)

    monkeypatch.setattr(mo, "refresh_provider_model_catalogs", offline)
    yield


@pytest.fixture(autouse=True)
def _restore_flet_hooks():
    """The app patches flet classes for the whole process at start (``sheet.install_close_buttons``); undo
    it after each test so other test files see stock Flet."""
    try:
        import flet as ft
        from flet.controls.base_page import BasePage
    except Exception:
        yield
        return
    saved = [(ft.BottomSheet, "before_update", ft.BottomSheet.__dict__.get("before_update")),
             (BasePage, "show_dialog", BasePage.__dict__.get("show_dialog")),
             (BasePage, "_remove_dialog", BasePage.__dict__.get("_remove_dialog"))]
    yield
    for cls, name, value in saved:
        if value is None:
            if name in cls.__dict__:
                delattr(cls, name)
        else:
            setattr(cls, name, value)


# ==========================================================================
# The Flutter client, rebuilt from the wire
# ==========================================================================

PATCH_CONTROL = 2
#: dialogs the ScaffoldMessenger shows (no route)
ROUTE_FREE = ("SnackBar", "Banner")
#: handlers a tap fires, in Flet's order of preference
TAP_EVENTS = ("on_click", "on_tap", "on_change", "on_select")


class _Route:
    """A modal route on the client Navigator (a sheet / alert dialog the client pushed)."""

    def __init__(self, dialog: dict, drag: bool) -> None:
        self.dialog_id = dialog["_i"]
        self.dialog = dialog  # the client Control it opened with (the builder's captured ``control``)
        self.content = dialog.get("content")  # what the route shows until it rebuilds
        self.kind = dialog.get("_c")
        self.drag = drag
        self.done = False
        self.current = None  # last isCurrent the route was built with

    def __repr__(self) -> str:
        return f"<{self.kind} route {self.dialog_id}>"


class _Page:
    def __init__(self, key: str) -> None:
        self.key = key
        self.view: Optional[dict] = None
        self.modals: list = []


class _State:
    __slots__ = ("route",)

    def __init__(self) -> None:
        self.route: Optional[_Route] = None


class PhoneClient:
    """Flet 1.0.3's Flutter client for one fake session, from the decoded wire messages (module docstring)."""

    def __init__(self, *, schedule: str = "burst", dismiss_lag: int = 2) -> None:
        assert schedule in ("burst", "per_message")
        self.schedule = schedule
        self.dismiss_lag = dismiss_lag
        self.root: Optional[dict] = None
        self.index: dict = {}  # control id -> client control dict (the latest one, like controlsIndex)
        self.session: Any = None
        self.nav: list = []  # [_Page] = Navigator history (pages with their pageless modal routes)
        self.popped: set = set()  # _pendingPoppedViewRoutes
        self.sent_pop: set = set()  # _sentViewPopEventsForRoutes
        self.states: dict = {}  # (view id, dialog id) -> _State
        self.completing: list = []  # [route, frames left] -> open:false + dismiss to Python
        self.exited = False  # SystemNavigator.pop (the app left)
        self.unknown_patches: list = []
        self.dead_taps: list = []  # (what, id): a tap on an id Python no longer knows
        self.frames = 0
        self.history: list = []  # what the user did and saw (failure messages)
        self.built: set = set()  # overlay control ids a frame built inside the last View (method listeners)
        self.probe_calls: list = []  # (control id, call id) of frame probes waiting for their listener

    # ---- wire -------------------------------------------------------------------------------------

    def receive(self, data: Any) -> None:
        if isinstance(data, dict) and data.get("_c") == "Page":  # REGISTER_CLIENT: the whole page
            self.root = data
            self.index = {}
            self._index(data)
            return
        if not (isinstance(data, list) and len(data) == 2 and isinstance(data[1], dict)):
            return
        action, body = data
        if action == PATCH_CONTROL:
            self._patch(body.get("id"), body.get("patch") or [])
        if self.schedule == "per_message" and self.root is not None and self.session is not None:
            self.frame()

    def _index(self, value: Any) -> None:
        stack = [value]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                if "_c" in item and "_i" in item:
                    self.index[item["_i"]] = item
                stack.extend(v for v in item.values() if isinstance(v, (dict, list)))
            elif isinstance(item, list):
                stack.extend(v for v in item if isinstance(v, (dict, list)))

    def _patch(self, cid: Any, patch: list) -> None:
        target = self.index.get(cid)
        if target is None or len(patch) < 2:
            self.unknown_patches.append(cid)  # "Flet: dropped a patch for unknown control"
            return
        table: dict = {}

        def walk(node: list, path: list) -> None:
            table[node[0]] = path
            if len(node) > 1 and isinstance(node[1], dict):
                for key, child in node[1].items():
                    walk(child, path + [key])

        walk(patch[0], [])

        def resolve(i: int) -> Any:
            obj = target
            for part in table[i]:
                obj = obj[part]
            return obj

        for op in patch[1:]:
            kind = op[0]
            if kind == 0:  # replace
                obj = resolve(op[1])
                obj[op[2]] = op[3]
                self._index(op[3])
            elif kind == 1:  # add
                obj = resolve(op[1])
                if isinstance(obj, list):
                    obj.insert(op[2], op[3])
                else:
                    obj[op[2]] = op[3]
                self._index(op[3])
            elif kind == 2:  # remove
                obj = resolve(op[1])
                if isinstance(obj, list):
                    obj.pop(op[2])
                else:
                    obj.pop(op[2], None)
            elif kind == 3:  # move
                src = resolve(op[1])
                item = src.pop(op[2])
                dst = resolve(op[3])
                if isinstance(dst, list):
                    dst.insert(op[4], item)
                else:
                    dst[op[4]] = item
            else:
                raise AssertionError(f"unknown patch op {op!r}")

    # ---- client tree ------------------------------------------------------------------------------

    @staticmethod
    def vkey(view: dict) -> str:
        route = view.get("route")
        return route if route is not None else str(view["_i"])

    def views(self) -> list:
        return [v for v in (self.root or {}).get("views") or [] if isinstance(v, dict)]

    def dialogs(self) -> list:
        controls = ((self.root or {}).get("_dialogs") or {}).get("controls") or []
        return [d for d in controls if isinstance(d, dict) and d.get("_c") not in ROUTE_FREE]

    # ---- frames -----------------------------------------------------------------------------------

    def frame(self) -> None:
        views = self.views()
        keys = [self.vkey(v) for v in views]
        self.popped &= set(keys)
        self.sent_pop &= set(keys)
        effective = [v for v in views if self.vkey(v) not in self.popped]
        # page.dart: Navigator.pages keyed by route; a page that leaves takes its pageless routes along
        old = {page.key: page for page in self.nav}
        nav = []
        for view in effective:
            page = old.pop(self.vkey(view), None) or _Page(self.vkey(view))
            page.view = view
            nav.append(page)
        for page in old.values():
            for route in list(page.modals):
                self._complete(route)
        self.nav = nav
        # view.dart: the dialogs are built inside the last View only (Python's list, not the client's)
        last = views[-1] if views else None
        built_in = last["_i"] if last is not None and self.vkey(last) not in self.popped else None
        dialogs = self.dialogs()
        present = {(built_in, d["_i"]) for d in dialogs} if built_in is not None else set()
        for key in list(self.states):
            if key not in present:
                del self.states[key]  # element unmounted: its State and the route it kept are gone
        post: list = []
        if built_in is not None:
            for dialog in dialogs:
                state = self.states.setdefault((built_in, dialog["_i"]), _State())
                is_open, last_open = bool(dialog.get("open")), bool(dialog.get("_open"))
                if is_open and not last_open:
                    dialog["_open"] = True
                    post.append(("open", dialog, state))
                elif is_open != last_open and last_open:
                    dialog["_open"] = False
                    route = state.route
                    if dialog.get("_c") == "BottomSheet":
                        state.route = None
                    if route is not None:
                        post.append(("close", route, None))
        if built_in is not None:  # the overlay is built next to the dialogs, in the last View only
            overlay = ((self.root or {}).get("_overlay") or {}).get("controls") or []
            self.built |= {c["_i"] for c in overlay if isinstance(c, dict) and "_i" in c}
        for kind, item, state in post:  # addPostFrameCallback
            if kind == "open":
                drag = item.get("_c") == "BottomSheet" and bool(item.get("draggable"))
                route = _Route(item, drag)
                state.route = route
                if self.nav:
                    self.nav[-1].modals.append(route)
            else:
                self._complete(item)  # closeModalRoute: pop (current) or removeRoute
        self._rebuild_routes()
        self.frames += 1
        self._answer_probes()  # after the frame: the listener registered during its build

    def _answer_probes(self) -> None:
        """A frame probe answers once a frame built its control (``invokeMethod`` waits for the listener)."""
        waiting = []
        for cid, call_id in self.probe_calls:
            if cid in self.built:
                self.conn.loop.call_soon(self.session.handle_invoke_method_results, cid, call_id, None, None)
            else:
                waiting.append((cid, call_id))
        self.probe_calls = waiting

    def _rebuild_routes(self) -> None:
        """A modal route rebuilds its builder (the sheet's current content) when its isCurrent changes."""
        top = self.nav[-1].modals[-1] if self.nav and self.nav[-1].modals else None
        for route in self.modals():
            current = route is top
            if route.current is not None and route.current != current:
                route.content = route.dialog.get("content")
            route.current = current

    def _complete(self, route: _Route) -> None:
        if route.done:
            return
        route.done = True
        for page in self.nav:
            if route in page.modals:
                page.modals.remove(route)
        self.completing.append([route, self.dismiss_lag])

    async def _deliver(self) -> None:
        ready = []
        for item in list(self.completing):
            item[1] -= 1
            if item[1] <= 0:
                self.completing.remove(item)
                ready.append(item[0])
        for route in ready:
            cid = route.dialog_id
            node = self.index.get(cid)  # updateControl(id): the client Control known for that id now
            if node is not None:
                node["_open"] = False
                node["open"] = False
                self.session.apply_patch(cid, {"open": False})
            if route.dialog.get("on_dismiss"):  # triggerControlEvent on the Control it opened with
                await self.session.dispatch_event(cid, "dismiss", None)

    async def settle(self, rounds: int = 12) -> None:
        """Python's burst (spawned tasks too), then a frame and the dismissals due, ``rounds`` times."""
        for _ in range(rounds):
            for _ in range(6):
                await asyncio.sleep(0)
            self.frame()
            await self._deliver()

    async def until(self, predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            await self.settle(2)
            try:
                if predicate():
                    return True
            except Exception:
                pass
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.02)

    # ---- what the user sees -------------------------------------------------------------------------

    def modals(self) -> list:
        return [route for page in self.nav for route in page.modals]

    def top(self) -> tuple:
        if not self.nav:
            return ("none", None)
        page = self.nav[-1]
        if page.modals:
            return ("modal", page.modals[-1])
        return ("page", page)

    def top_route(self) -> str:
        return self.nav[-1].key if self.nav else ""

    def describe(self) -> str:
        """The client's Navigator (pages with their modal routes) and Python's view of each route's dialog."""
        parts = []
        for page in self.nav:
            routes = []
            for route in page.modals:
                control = self.session.index.get(route.dialog_id) if self.session is not None else None
                py_open = getattr(control, "open", None)
                in_dialogs = control is not None and control in self.session.page._dialogs.controls
                title = next((t for n, _c in self._walk(route.content) for t in self.texts_of(n)), "")[:40]
                routes.append(f"{route!r} {title!r} python={'gone' if control is None else type(control).__name__}"
                              f" open={py_open} in_dialogs={in_dialogs}")
            parts.append(f"{page.key}: [{'; '.join(routes)}]")
        return " | ".join(parts) + f" || last: {self.history[-5:]}"

    def showing(self, dialog: Any) -> bool:
        kind, what = self.top()
        return kind == "modal" and what.dialog_id == getattr(dialog, "_i", dialog)

    def surface(self) -> list:
        """The client trees the user can touch now."""
        kind, what = self.top()
        if kind == "modal":
            return [what.content] if isinstance(what.content, dict) else []
        if kind == "page":
            overlay = ((self.root or {}).get("_overlay") or {}).get("controls") or []
            return [what.view, *[c for c in overlay if isinstance(c, dict)]]
        return []

    @staticmethod
    def _walk(root: Any, trail: tuple = ()):
        """(control dict, ancestors) for every client control under ``root``; Tabs show their selected page."""
        stack = [(root, trail)]
        while stack:
            node, chain = stack.pop()
            if isinstance(node, list):
                stack.extend((item, chain) for item in reversed(node))
                continue
            if not isinstance(node, dict):
                continue
            if "_c" in node:
                if node.get("visible") is False:
                    continue
                yield node, chain
                chain = chain + (node,)
                if node["_c"] == "TabBarView":  # Flutter builds only the selected tab's page
                    tabs = next((c for c in reversed(chain) if c.get("_c") == "Tabs"), None)
                    index = int((tabs or {}).get("selected_index") or 0)
                    stack.append(((node.get("controls") or [])[index:index + 1], chain))
                    continue
            for key, value in node.items():
                # drawers and popup menu items are not on screen until opened (``menu`` opens a popup)
                if key.startswith("_") or key in ("drawer", "end_drawer", "items"):
                    continue
                if isinstance(value, (dict, list)):
                    stack.append((value, chain))

    def find(self, predicate: Callable[[dict], bool], roots: Optional[list] = None) -> list:
        out = []
        for root in self.surface() if roots is None else roots:
            for node, chain in self._walk(root):
                if predicate(node):
                    out.append((node, chain))
        return out

    @staticmethod
    def key_of(node: dict) -> Any:
        key = node.get("key")
        if isinstance(key, dict):
            return key.get("value")
        return key

    @staticmethod
    def tooltip_of(node: dict) -> Any:
        tip = node.get("tooltip")
        if isinstance(tip, dict):
            return tip.get("message")
        return tip

    @staticmethod
    def texts_of(node: dict) -> list:
        out = []
        for name in ("value", "text", "label", "content", "title", "semantics_label"):
            value = node.get(name)
            if isinstance(value, str):
                out.append(value)
        return out

    def find_one(self, *, key: Any = None, tooltip: Optional[str] = None, text: Optional[str] = None,
                 roots: Optional[list] = None) -> tuple:
        def match(node: dict) -> bool:
            if key is not None and self.key_of(node) != key:
                return False
            if tooltip is not None and self.tooltip_of(node) != tooltip:
                return False
            if text is not None and text not in self.texts_of(node):
                return False
            return True

        found = self.find(match, roots)
        assert found, f"nothing on screen matches key={key!r} tooltip={tooltip!r} text={text!r}; top {self.top()}"
        return found[0]

    # ---- what the user does -------------------------------------------------------------------------

    async def tap(self, *, key: Any = None, tooltip: Optional[str] = None, text: Optional[str] = None) -> None:
        """Tap what the client shows on top: the nearest tappable control (itself or an ancestor)."""
        node, chain = self.find_one(key=key, tooltip=tooltip, text=text)
        self.history.append(f"tap {key or tooltip or text!r} on {self.top()}")
        path = (*chain, node)
        tab = next((c for c in reversed(path) if c.get("_c") == "Tab"), None)
        if tab is not None:  # a tab of a TabBar: the Tabs control selects it and reports the index
            bar = next((c for c in reversed(path) if c.get("_c") == "TabBar"), None)
            tabs = next((c for c in reversed(path) if c.get("_c") == "Tabs"), None)
            if bar is not None and tabs is not None:
                index = next(i for i, t in enumerate(bar.get("tabs") or []) if t is tab)
                tabs["selected_index"] = index
                self.session.apply_patch(tabs["_i"], {"selected_index": index})
                if tabs.get("on_change"):
                    await self.session.dispatch_event(tabs["_i"], "change", index)
                return
        segment = next((c for c in reversed(path) if c.get("_c") == "Segment"), None)
        owner = next((c for c in reversed(path) if c.get("_c") == "SegmentedButton"), None)
        if segment is not None and owner is not None:  # a segment: its SegmentedButton reports the selection
            if any(c.get("disabled") for c in path):
                return
            owner["selected"] = [segment.get("value")]
            self.session.apply_patch(owner["_i"], {"selected": [segment.get("value")]})
            if owner.get("on_change"):
                await self.session.dispatch_event(owner["_i"], "change", [segment.get("value")])
            return
        for target in (node, *reversed(chain)):
            event = next((name for name in TAP_EVENTS if target.get(name)), None)
            if event is None:
                continue
            if any(c.get("disabled") for c in (*chain, node)):
                return  # a disabled control ignores the tap on the client
            cid = target["_i"]
            if self.session.index.get(cid) is None:
                self.dead_taps.append((key or tooltip or text, cid))
            name = event[3:]
            data = None
            if name == "change" and target.get("_c") in ("Switch", "Checkbox", "CupertinoSwitch"):
                target["value"] = not bool(target.get("value"))
                self.session.apply_patch(cid, {"value": target["value"]})
                data = str(target["value"]).lower()
            await self.session.dispatch_event(cid, name, data)
            return
        raise AssertionError(f"{key or tooltip or text!r} has no tap handler on the client")

    async def close_menu(self) -> None:
        """"Close Menu ✕" of the sheet on top."""
        kind, route = self.top()
        assert kind == "modal", f"no sheet on top: {self.top()}"
        await self.tap(key="sheet-close")

    async def swipe_down(self) -> None:
        kind, route = self.top()
        assert kind == "modal", f"no sheet on top: {self.top()}"
        assert route.drag, f"{route} does not slide closed (enableDrag was off when it opened)"
        self.history.append(f"swipe down {route}")
        self._complete(route)

    async def menu(self, item_text: str, *, key: Any = None, tooltip: Optional[str] = None) -> None:
        """A ``PopupMenuButton`` on screen: the client opens its menu, the item tapped closes it and fires the
        item's click (the menu route is gone before Python handles it)."""
        node, chain = self.find_one(key=key, tooltip=tooltip)
        button = next((c for c in (node, *reversed(chain)) if c.get("_c") == "PopupMenuButton"), None)
        assert button is not None, f"{key or tooltip!r} is not a popup menu button"
        item = next((i for i in button.get("items") or [] if isinstance(i, dict) and i.get("visible") is not False
                     and any(item_text in t for t in self.texts_of(i))), None)
        assert item is not None, f"{key or tooltip!r} has no item {item_text!r}"
        self.history.append(f"menu {key or tooltip!r} > {item_text!r} on {self.top()}")
        if item.get("disabled") or not item.get("on_click"):
            return
        if self.session.index.get(item["_i"]) is None:
            self.dead_taps.append((item_text, item["_i"]))
        await self.session.dispatch_event(item["_i"], "click", None)

    async def back(self) -> str:
        """Android back; returns what it did: 'view' (a View popped), 'modal', 'confirm', 'exit' or 'none'."""
        self.history.append(f"back on {self.top()}")
        views = self.views()
        if len(views) > 1:
            top = views[-1]
            if top.get("can_pop", True) and not top.get("on_confirm_pop"):
                self._mark_popped(self.vkey(top))
                await self._settle_event()
                return "view"
        # Navigator.maybePop
        kind, what = self.top()
        if kind == "modal":
            self._complete(what)
            return "modal"
        if kind != "page":
            return "none"
        view = what.view
        if view.get("can_pop", True):
            if len(self.nav) <= 1:
                self.exited = True  # the first route cannot pop: SystemNavigator.pop
                return "exit"
            self._mark_popped(what.key)
            await self._settle_event()
            return "view"
        if not view.get("on_confirm_pop"):
            return "none"
        sent = len(self.session_messages())
        await self.session.dispatch_event(view["_i"], "confirm_pop", None)
        should_pop = None
        for message in self.session_messages()[sent:]:
            body = getattr(message, "body", None)
            if getattr(body, "name", None) == "confirm_pop" and getattr(body, "control_id", None) == view["_i"]:
                should_pop = bool((body.args or {}).get("should_pop"))
        if should_pop:
            if len(self.nav) <= 1 or self.nav[0].view is view:
                self.exited = True
                return "exit"
            self._mark_popped(what.key)
            await self._settle_event()
            return "view"
        return "confirm"

    def session_messages(self) -> list:
        return self.conn.messages

    def _mark_popped(self, key: str) -> None:
        self.popped.add(key)
        self.frame()
        if key not in self.sent_pop:
            self.sent_pop.add(key)
            self._view_pops.append(key)

    async def _settle_event(self) -> None:
        pops, self._view_pops = self._view_pops, []
        for key in pops:
            await self.session.dispatch_event(self.session.page._i, "view_pop", {"route": key})

    def bind(self, session: Any, conn: Any) -> None:
        from flet.messaging.protocol import MessageAction
        from glossarion_mobile.ui.components.dialogs import FRAME_PROBE_METHOD

        self.session = session
        self.conn = conn
        self._view_pops: list = []
        # a client that reports every dismissed dialog (Flet's own transports do; components.dialogs)
        conn.confirms_dialog_dismiss = True
        send = conn.send_message

        def send_message(message: Any) -> None:
            body = getattr(message, "body", None)
            if message.action == MessageAction.INVOKE_METHOD and getattr(body, "name", None) == FRAME_PROBE_METHOD:
                conn.messages.append(message)
                self.probe_calls.append((body.control_id, body.call_id))
                if self.schedule == "per_message" and self.root is not None:
                    self.frame()  # answers it when an earlier frame built the probe
                else:
                    self._answer_probes()
                return
            send(message)

        conn.send_message = send_message


def _wire(monkeypatch, client: PhoneClient) -> None:
    """Feed every message the fake transport encodes (and the REGISTER_CLIENT page) to ``client``."""
    import msgpack

    original = msgpack.packb

    def packb(obj, *args, **kwargs):
        out = original(obj, *args, **kwargs)
        try:
            data = msgpack.unpackb(out, raw=False, strict_map_key=False)
        except Exception:
            return out
        client.receive(data)
        return out

    monkeypatch.setattr(msgpack, "packb", packb)


# ==========================================================================
# The app with a held chat EPUB job (the owner's floating job bar)
# ==========================================================================


class HeldRuns:
    """The ``run`` of the job kinds under test, replaced: each run announces itself with its own release gate
    and waits until the test releases it (or the job is stopped)."""

    def __init__(self, monkeypatch, kinds) -> None:
        from glossarion_mobile import job_kinds

        self.started: queue.Queue = queue.Queue()
        self.gates: list = []
        for kind in kinds:
            info = job_kinds.get_kind(kind)  # the real KindInfo: verb, icon, stop kind stay real
            monkeypatch.setitem(job_kinds._CACHE, kind, dataclasses.replace(info, run=self._run))

    def _run(self, ctx):
        release = threading.Event()
        self.gates.append(release)
        ctx.log("🚀 Starting translation")
        ctx.log("📖 Chapter 14/107: chapter0014.xhtml")
        self.started.put(release)
        while not release.wait(0.02):
            if ctx.stop_requested():
                break
        return None

    async def next(self, timeout: float = 15.0) -> threading.Event:
        return await asyncio.to_thread(self.started.get, True, timeout)

    def release_all(self) -> None:
        for gate in self.gates:
            gate.set()


class Phone(types.SimpleNamespace):
    pass


async def _start_phone(client: PhoneClient) -> Phone:
    uf = _uf()
    tj = _tj()
    main_module = uf._load_main_module()
    conn, session = uf._fake_session("android")  # the REGISTER_CLIENT page goes through the client
    client.bind(session, conn)
    session.apply_page_patch({"width": 412, "height": 860})
    page = session.page
    await main_module.main(page)
    await session.after_event(page)
    app = page.data
    if getattr(app, "jobs", None) is not None:
        app.jobs.background.permissions = tj.FakePermissions()  # no permission prompt on the fake session

        async def confirm(*_args, **_kwargs):
            return True

        app.jobs.background.confirm = confirm
    await client.until(lambda: app.state.engine_ready, 15)
    await client.settle()
    return Phone(uf=uf, conn=conn, session=session, page=page, app=app, client=client)


async def _stop_phone(phone: Phone, runs: Optional[HeldRuns] = None) -> None:
    app = phone.app
    if runs is not None:
        runs.release_all()
    service = getattr(app, "job_service", None)
    if service is not None:
        await asyncio.to_thread(service.wait_idle, 15)
    reader = getattr(app, "reader", None)
    if reader is not None and hasattr(reader, "detach"):
        try:
            reader.detach()
        except Exception:
            pass
    if getattr(app, "jobs", None) is not None:
        app.jobs.close()
    if getattr(app, "settings", None) is not None:
        app.settings.close()
    await phone.uf._stop(app)


def _fake_backend(monkeypatch, tmp_path) -> None:
    """The app's JobService runs on ``test_jobs.FakeBackend`` (no HeadlessOwner, no shared pipeline);
    JobsFeature, ChatFeature, the Library and the shell stay real."""
    from glossarion_mobile.services import jobs as jobs_service

    tj = _tj()
    monkeypatch.setattr(jobs_service, "JobBackend", lambda: tj.FakeBackend(str(tmp_path / "job-out")))


# ==========================================================================
# Checks
# ==========================================================================


def _in_dialogs(page: Any, dialog: Any) -> bool:
    return dialog in page._dialogs.controls


def assert_closed(phone: Phone, dialog: Any, *, owner_open: Optional[Callable[[], bool]] = None,
                  what: str = "") -> None:
    """The client shows no modal route anywhere (no ghost under any screen), Python dropped the dialog, and
    the owner's own state no longer thinks it is open."""
    client = phone.client
    ghosts = client.modals()
    assert ghosts == [], f"{what}: the client still shows {ghosts} (top {client.top()}); {client.history[-6:]}"
    assert not getattr(dialog, "open", False), f"{what}: Python still has it open"
    assert not _in_dialogs(phone.page, dialog), f"{what}: still in page._dialogs"
    if owner_open is not None:
        assert not owner_open(), f"{what}: the app's state still thinks it is open"
    assert not client.dead_taps, f"{what}: taps on controls Python no longer knows: {client.dead_taps}"
    assert not client.unknown_patches, \
        f"{what}: patches for controls the client does not have: {client.unknown_patches}"


async def open_and_close_twice(phone: Phone, opener: Callable[[], Any], dialog_of: Callable[[], Any], *,
                               what: str, owner_open: Optional[Callable[[], bool]] = None,
                               swipe: bool = True) -> list:
    """Open with ``opener`` (async: the user's tap), close with Close Menu ✕; open again, close with a swipe."""
    client = phone.client
    seen = []
    methods = ["close_menu", "swipe"] if swipe else ["close_menu", "close_menu"]
    for how in methods:
        await opener()
        assert await client.until(lambda: dialog_of() is not None and client.showing(dialog_of())), \
            f"{what}: did not open on top ({client.top()}); {client.history[-6:]}"
        dialog = dialog_of()
        seen.append(dialog)
        if owner_open is not None:
            assert owner_open(), f"{what}: the app does not know it is open"
        if how == "close_menu":
            await client.close_menu()
        else:
            await client.swipe_down()
        assert await client.until(lambda: not client.modals() and not _in_dialogs(phone.page, dialog)), \
            f"{what} ({how}): still showing {client.modals()} / in dialogs {_in_dialogs(phone.page, dialog)}"
        assert_closed(phone, dialog, owner_open=owner_open, what=f"{what} ({how})")
    return seen


# ==========================================================================
# Scenarios
# ==========================================================================


SCHEDULES = ["burst", "per_message"]
PLUS = "Attach, output mode and tools"


async def _held_job(phone: Phone, runs: HeldRuns, tmp_path: Path) -> tuple:
    """The owner's screenshot: a chat EPUB translation running (held at its first chapter), its floating job
    bar on screen. Returns (job id, chat id, release gate)."""
    from glossarion_mobile.diagnostics.fixtures import build_tiny_epub
    from glossarion_mobile.services.jobs import JOB_TARGET_PARAM, JobSpec, JobState

    app, client = phone.app, phone.client
    raw = tmp_path / "Inbox" / "[211151] Novel.epub"
    raw.parent.mkdir(parents=True, exist_ok=True)
    build_tiny_epub(raw, chapters=3)
    run_input = tmp_path / "runs" / "r1" / raw.name
    run_input.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(raw, run_input)
    chats = app.chat_feature.chats
    cid = chats.new_chat()
    chats.append_messages(cid, [("user_file", raw.name, str(raw), raw.stat().st_size, "", "user")])
    spec = JobSpec("direct_text", raw.name, (str(run_input),),
                   params={JOB_TARGET_PARAM: [str(raw)], "user_index": 0, "is_attachment": True},
                   origin={"type": "chat", "cid": cid, "label": f"Chat · {raw.name}"})
    jid = app.job_service.submit(spec)
    release = await runs.next()
    assert await client.until(lambda: app.job_service.snapshot(jid).state is JobState.RUNNING)
    return jid, cid, release


def _top_dialog(page: Any) -> Any:
    """The newest open route dialog Python has (what a screen showed by itself)."""
    return next((d for d in reversed(page._dialogs.controls)
                 if getattr(d, "open", False) and type(d).__name__ not in ROUTE_FREE), None)


def _bar_on_screen(client: PhoneClient) -> bool:
    """The floating job bar ("Translating · <book>") is part of what the client shows on top."""
    return bool(client.find(lambda n: n.get("_c") == "Text" and str(n.get("value") or "").startswith("Translating")))


async def _back_to_chat(phone: Phone) -> None:
    """Android back until the chat is on top (every back pops a screen: nothing else may be in the way)."""
    client = phone.client
    for _ in range(8):
        if not phone.app.shell.stack and len(client.nav) == 1:
            break
        did = await client.back()
        assert did == "view", f"back did {did!r} instead of popping a screen; top {client.top()}"
        assert await client.until(lambda: not client.modals()), f"a sheet came back: {client.modals()}"
    assert await client.until(lambda: not phone.app.shell.stack and len(client.nav) == 1 and not client.modals())


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_chat_sheets_close_with_close_menu_swipe_and_back(app_env, monkeypatch, tmp_path, schedule):
    """The chat's sheets with the floating job bar running: ＋, ＋ rows that swap it for another sheet (Chat
    settings, From Library), Chat settings rebuilt by a change while open, the model sheet over Chat settings
    (a sheet from a sheet), the header's model sheet (its list arrives after it opened) and Output mode ›
    Options…; Android back on the chat with a sheet open closes only the sheet."""
    runs = HeldRuns(monkeypatch, ("direct_text",))
    _fake_backend(monkeypatch, tmp_path)
    client = PhoneClient(schedule=schedule)
    _wire(monkeypatch, client)

    async def scenario():
        phone = await _start_phone(client)
        page, app = phone.page, phone.app
        try:
            view = app.chat_view
            await _held_job(phone, runs, tmp_path)
            plus_open = lambda: view.composer.plus_button.tooltip == "Close"  # ＋ turned into ×

            # ---- ＋ ---------------------------------------------------------------------------------------
            await open_and_close_twice(phone, lambda: client.tap(tooltip=PLUS),
                                       lambda: getattr(view.plus_sheet, "dialog", None), what="＋ sheet",
                                       owner_open=plus_open)
            # Android back with it open (the owner's way out): only the sheet closes
            await client.tap(tooltip=PLUS)
            assert await client.until(lambda: client.showing(view.plus_sheet.dialog))
            assert await client.back() == "modal"
            assert await client.until(lambda: not client.modals() and not _in_dialogs(page, view.plus_sheet.dialog))
            assert_closed(phone, view.plus_sheet.dialog, owner_open=plus_open, what="＋ sheet (back)")

            # ---- ＋ › This chat › Chat settings…: one sheet closes, the next opens in the same tap ---------
            await client.tap(tooltip=PLUS)
            assert await client.until(lambda: client.showing(view.plus_sheet.dialog))
            plus = view.plus_sheet.dialog
            await client.tap(key="this-chat-chat_settings")
            assert await client.until(lambda: view.settings_sheet is not None
                                      and client.showing(view.settings_sheet.dialog)
                                      and len(client.modals()) == 1), f"{client.modals()} {client.history[-4:]}"
            assert not _in_dialogs(page, plus) and not plus_open()
            settings = view.settings_sheet
            # a change rebuilds its rows while it is open, then Close Menu ✕
            switch, _chain = client.find_one(roots=client.surface(), key=None, tooltip=None, text=None) \
                if False else (None, None)
            switches = client.find(lambda n: n.get("_c") == "Switch" and n.get("on_change") and not n.get("disabled"))
            assert switches, "Chat settings shows no switch"
            node = switches[0][0]
            await client.session.dispatch_event(node["_i"], "change", "true" if not node.get("value") else "false")
            await client.settle()
            assert client.showing(settings.dialog)
            await client.close_menu()
            assert await client.until(lambda: not client.modals() and not _in_dialogs(page, settings.dialog))
            assert_closed(phone, settings.dialog, what="Chat settings (✕ after a change)")

            # ---- Chat settings (the composer's pill) › Model › Choose…: the model sheet over it -----------
            await client.tap(tooltip="Chat settings")
            assert await client.until(lambda: view.settings_sheet is not settings and view.settings_sheet is not None
                                      and client.showing(view.settings_sheet.dialog))
            settings = view.settings_sheet
            await client.tap(text="Choose…")
            assert await client.until(lambda: view.model_sheet is not None and client.showing(view.model_sheet.dialog))
            model_sheet = view.model_sheet
            assert [r.dialog_id for r in client.modals()] == [settings.dialog._i, model_sheet.dialog._i]
            await client.close_menu()  # the upper one only
            assert await client.until(lambda: client.showing(settings.dialog) and len(client.modals()) == 1
                                      and not _in_dialogs(page, model_sheet.dialog))
            assert model_sheet.closed
            await client.swipe_down()
            assert await client.until(lambda: not client.modals() and not _in_dialogs(page, settings.dialog))
            assert_closed(phone, settings.dialog, what="Chat settings under the model sheet")

            # ---- the header's model sheet: its list arrives after it opened (rebuilt while open) --------------
            def model_sheet_now():
                return getattr(view.model_sheet, "dialog", None)

            async def open_model():
                await client.tap(text=" ▾")
                await client.until(lambda: view.model_sheet is not None and view.model_sheet.snapshot.loaded, 15)

            await open_and_close_twice(phone, open_model, model_sheet_now, what="model sheet",
                                       owner_open=lambda: not view.model_sheet.closed)

            # ---- Output mode chip › Options… -----------------------------------------------------------------
            async def open_options():
                await client.menu("Options", key="mode-chip")

            await open_and_close_twice(phone, open_options, lambda: getattr(view.mode_sheet, "dialog", None),
                                       what="Output mode options")

            # ---- ＋ › From Library: the ＋ sheet makes way for the Library picker ------------------------------
            async def open_picker():
                await client.tap(tooltip=PLUS)
                assert await client.until(lambda: client.showing(view.plus_sheet.dialog))
                await client.tap(key="attach-library")

            await open_and_close_twice(phone, open_picker,
                                       lambda: getattr(getattr(view, "library_picker", None), "sheet", None),
                                       what="＋ › From Library picker")
            assert not plus_open()
            assert _bar_on_screen(client), "the floating job bar is gone from the chat"
            assert await client.back() == "exit"  # the chat alone: back leaves the app, no ghost sheet
        finally:
            await _stop_phone(phone, runs)

    asyncio.run(scenario())


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_sheet_rows_that_open_a_screen_leave_no_sheet_behind(app_env, monkeypatch, tmp_path, schedule):
    """A sheet row that closes its sheet and opens a screen in the same tap (＋ › Tools › Progress manager,
    whose screen opens its own book picker at once; Help › Logs & diagnostics): nothing is left under the new
    screen, the screen's own sheet is on it and closes with ✕, and Back pops screen after screen to the chat."""
    runs = HeldRuns(monkeypatch, ("direct_text",))
    _fake_backend(monkeypatch, tmp_path)
    client = PhoneClient(schedule=schedule)
    _wire(monkeypatch, client)

    async def scenario():
        phone = await _start_phone(client)
        page, app = phone.page, phone.app
        try:
            view = app.chat_view
            await _held_job(phone, runs, tmp_path)

            # ---- ＋ › Tools › Progress manager -------------------------------------------------------------
            await client.tap(tooltip=PLUS)
            assert await client.until(lambda: client.showing(view.plus_sheet.dialog))
            plus = view.plus_sheet.dialog
            await client.tap(key="tool-progress")
            # the Progress manager asks which book at once: its picker sheet opens on the new screen
            assert await client.until(lambda: app.shell.stack and client.top_route() == app.shell.current_route
                                      and not _in_dialogs(page, plus) and client.showing(_top_dialog(page))
                                      and len(client.modals()) == 1 and client.nav[-1].modals), \
                f"{client.describe()} {[e.route for e in app.shell.stack]}"
            picker = _top_dialog(page)
            await client.close_menu()
            assert await client.until(lambda: not client.modals() and not _in_dialogs(page, picker))
            assert_closed(phone, picker, what="Progress manager's book picker")
            assert _bar_on_screen(client), "the floating job bar is not on the pushed screen"
            await _back_to_chat(phone)

            # ---- Help › Logs & diagnostics -------------------------------------------------------------------
            help_sheet = app._on_help()  # the drawer's Help row
            assert await client.until(lambda: client.showing(help_sheet.dialog))
            await client.tap(text="Logs & diagnostics")
            assert await client.until(lambda: app.shell.stack and client.top_route() == app.shell.current_route
                                      and not client.modals() and not _in_dialogs(page, help_sheet.dialog)), \
                client.describe()
            await _back_to_chat(phone)
            assert await client.back() == "exit"
        finally:
            await _stop_phone(phone, runs)

    asyncio.run(scenario())


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_screen_opened_while_a_sheet_is_open_leaves_no_sheet_behind(app_env, monkeypatch, tmp_path, schedule):
    """The job notification's Open (or a link from outside) while the ＋ sheet is open: the sheet closes, the
    Job page is on top with nothing under it, the app knows the sheet is closed, Back returns to the chat."""
    runs = HeldRuns(monkeypatch, ("direct_text",))
    _fake_backend(monkeypatch, tmp_path)
    client = PhoneClient(schedule=schedule)
    _wire(monkeypatch, client)

    async def scenario():
        phone = await _start_phone(client)
        page, app = phone.page, phone.app
        try:
            view = app.chat_view
            jid, _cid, _release = await _held_job(phone, runs, tmp_path)
            await client.tap(tooltip=PLUS)
            assert await client.until(lambda: client.showing(view.plus_sheet.dialog))
            plus = view.plus_sheet.dialog
            app.navigate_to("jobs.detail", {"jid": str(jid)})
            assert await client.until(lambda: app.shell.stack and client.top_route() == app.shell.current_route
                                      and not client.modals() and not _in_dialogs(page, plus)), client.describe()
            assert view.composer.plus_button.tooltip != "Close"
            assert _bar_on_screen(client)
            await _back_to_chat(phone)
            assert await client.back() == "exit"
        finally:
            await _stop_phone(phone, runs)

    asyncio.run(scenario())


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("order", ["navigate_then_sheet", "sheet_then_navigate"])
def test_sheet_shown_with_a_screen_change(app_env, monkeypatch, tmp_path, schedule, order):
    """A sheet shown in the same handler as a navigation (``navigate_to`` and ``show`` in one go, either order):
    whatever the guard decides (move it to the new screen, or close it first), the client ends with no sheet
    under any screen, Close Menu ✕ closes a sheet it shows, Python keeps no dead dialog, and the screen change
    is not held up waiting for a dismiss that never comes."""
    runs = HeldRuns(monkeypatch, ("direct_text",))
    _fake_backend(monkeypatch, tmp_path)
    client = PhoneClient(schedule=schedule)
    _wire(monkeypatch, client)

    async def scenario():
        from glossarion_mobile.ui.components import dialogs
        from glossarion_mobile.ui.components.info_sheet import InfoSheet

        phone = await _start_phone(client)
        page, app = phone.page, phone.app
        try:
            jid, _cid, _release = await _held_job(phone, runs, tmp_path)
            info = InfoSheet(title="Job", body="Shown with the screen change.")
            started = time.monotonic()
            if order == "navigate_then_sheet":
                app.navigate_to("jobs.detail", {"jid": str(jid)})
                info.show(page)
            else:
                info.show(page)
                app.navigate_to("jobs.detail", {"jid": str(jid)})
            assert await client.until(lambda: app.shell.stack and client.top_route() == app.shell.current_route)
            took = time.monotonic() - started
            await client.settle()
            problems = []
            if took >= dialogs.VIEWS_WAIT * 2 / 3:  # the screen itself is up in a few 10 ms, also on CI
                problems.append(f"the screen change waited {took:.2f}s (VIEWS_WAIT {dialogs.VIEWS_WAIT}s) for a "
                                "dismiss that never came")
            stranded = [(p.key, p.modals) for p in client.nav[:-1] if p.modals]
            if stranded:
                problems.append(f"a sheet is stranded under the new screen: {client.describe()}")
                # what the owner then meets: back to the chat, the sheet is there, and Close Menu ✕ ...
                for _ in range(4):
                    if len(client.nav) == 1:
                        break
                    pages = len(client.views())
                    await client.back()
                    await client.until(lambda: len(client.views()) < pages, 4)
                if client.showing(info.dialog):
                    await client.close_menu()
                    await client.settle(20)
                    if client.showing(info.dialog):
                        problems.append("back on the chat the stranded sheet is on top and Close Menu ✕ does not "
                                        f"close it: {client.describe()}")
            elif client.showing(info.dialog):
                await client.close_menu()
            if not await client.until(lambda: not client.modals() and not _in_dialogs(page, info.dialog), 4):
                problems.append(f"not closed: client {client.describe()}; python: in _dialogs="
                                f"{_in_dialogs(page, info.dialog)} open={info.dialog.open}")
            assert not problems, f"({order}, {schedule}): " + " || ".join(problems)
            assert_closed(phone, info.dialog, what=f"sheet shown with a screen change ({order})")
            await _back_to_chat(phone)
            assert await client.back() == "exit"
        finally:
            await _stop_phone(phone, runs)

    asyncio.run(scenario())


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_job_page_sheets_with_the_floating_job_bar(app_env, monkeypatch, tmp_path, schedule):
    """The owner's screenshot: the floating job bar (Jobs › the job's row) opens the running job's Job page; the
    sheet there (the Library button's reason) closes with ✕ and with a swipe; Android back with it open closes it
    (Flet 1.0.3 takes the Job page along) without a sheet coming back on any screen."""
    runs = HeldRuns(monkeypatch, ("direct_text",))
    _fake_backend(monkeypatch, tmp_path)
    client = PhoneClient(schedule=schedule)
    _wire(monkeypatch, client)

    async def scenario():
        from glossarion_mobile.ui.screens.job_detail import JobDetailScreen

        phone = await _start_phone(client)
        page, app = phone.page, phone.app
        try:
            jid, _cid, _release = await _held_job(phone, runs, tmp_path)
            assert await client.until(lambda: _bar_on_screen(client)), "no floating job bar on the chat"

            async def open_job_page():
                node, _chain = client.find(lambda n: n.get("_c") == "Text"
                                           and str(n.get("value") or "").startswith("Translating"))[0]
                await client.tap(text=node["value"])  # the bar opens Jobs; its row opens the Job page
                assert await client.until(lambda: app.shell.stack and client.top_route() == app.shell.current_route
                                          and app.shell.current_route in ("/jobs", f"/jobs/{jid}")), client.describe()
                if app.shell.current_route == "/jobs":
                    assert await client.until(lambda: client.find(lambda n: "[211151] Novel.epub" in client.texts_of(n)))
                    await client.tap(text="[211151] Novel.epub")
                assert await client.until(lambda: isinstance(app.shell.top_screen, JobDetailScreen)
                                          and client.top_route() == app.shell.current_route), client.describe()
                detail = app.shell.top_screen
                assert await client.until(lambda: detail.library_reason.visible
                                          and client.find(lambda n: client.key_of(n) == "job-library-reason")), \
                    "the Job page shows no Library reason chip"
                return detail

            detail = await open_job_page()
            assert detail.job_id == jid and _bar_on_screen(client)

            async def open_reason():
                await client.tap(key="job-library-reason")

            await open_and_close_twice(phone, open_reason, lambda: _top_dialog(page), what="Job page reason sheet")

            # Android back with the sheet open (the owner's way out). Flet 1.0.3 pops a pushed View that does not
            # intercept back together with its sheet ('view'); a View that intercepts it closes the sheet only
            # ('modal'). Either way no sheet is left anywhere and the app agrees.
            await client.tap(key="job-library-reason")
            assert await client.until(lambda: _top_dialog(page) is not None and client.showing(_top_dialog(page)))
            sheet = _top_dialog(page)
            did = await client.back()
            assert did in ("view", "modal"), did
            if did == "view":
                assert await client.until(lambda: not app.shell.stack or not isinstance(app.shell.top_screen,
                                                                                         JobDetailScreen))
            assert await client.until(lambda: not client.modals() and not _in_dialogs(page, sheet)), client.describe()
            assert_closed(phone, sheet, what=f"Job page sheet closed by back ({did})")
            await _back_to_chat(phone)
            assert await client.back() == "exit"
        finally:
            await _stop_phone(phone, runs)

    asyncio.run(scenario())


def _devfix5():
    return _load("_glossarion_hf3_issue1_devfix5_helpers", "test_devfix_issue5.py")


def _library_cores() -> bool:
    return all(_has(m) for m in ("library_core", "reader_doc", "progress_core", "ebooklib"))


@pytest.mark.skipif(not _library_cores(), reason="the shared Library / Reader cores are not importable here")
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_library_reader_and_glossary_sheets(app_env, monkeypatch, tmp_path, schedule):
    """Library: Filter, ⋮ › Library settings, a book's actions (and its rows that open a screen or swap the sheet
    for another one); Reader: ⋯, Aa, search, ⋯ › Bookmarks, ⋯ › Open Book Details; Glossaries: a glossary's
    actions. Each closes with ✕ and with a swipe, nothing is left under any screen, and Back walks out screen by
    screen (the Reader asks before it pops)."""
    import library_core as lc

    from glossarion_mobile.ui.reader import reader_view

    monkeypatch.setattr(reader_view, "CHROME_AUTO_HIDE", 3600.0)  # the Reader's bars stay up for the taps
    runs = HeldRuns(monkeypatch, ("direct_text",))
    _fake_backend(monkeypatch, tmp_path)
    client = PhoneClient(schedule=schedule)
    _wire(monkeypatch, client)

    async def scenario():
        from glossarion_mobile.ui.library.book_page import BookPageScreen
        from glossarion_mobile.ui.library.home import LibraryScreen
        from glossarion_mobile.ui.reader.reader_view import ReaderScreen

        phone = await _start_phone(client)
        page, app = phone.page, phone.app
        d5 = _devfix5()
        try:
            assert await client.until(lambda: lc.current_library_env() is not None, 15)
            output, library = Path(app.paths.output), Path(app.paths.library)
            workspace = d5._epub_workspace(output, library, "Alpha", glossary=True)
            await _held_job(phone, runs, tmp_path)

            # ---- Library -------------------------------------------------------------------------------------
            app.navigate_to("library", reset=True)  # the drawer's Library
            assert await client.until(lambda: isinstance(app.shell.top_screen, LibraryScreen)
                                      and client.top_route() == app.shell.current_route, 15)
            home = app.shell.top_screen
            assert await client.until(lambda: home.snapshot.scanned_at, 30), "the Library never scanned"
            if not client.find(lambda n: "Alpha" in " ".join(client.texts_of(n))):
                await client.tap(text="Completed (1)")  # the finished book's shelf
            assert await client.until(lambda: client.find(lambda n: "Alpha" in " ".join(client.texts_of(n))), 30), \
                f"the Library never showed the book: {_screen_texts(client)[:60]} {app.paths.output} {app.paths.library}"

            await open_and_close_twice(phone, lambda: client.tap(key="lib-filter"), lambda: _top_dialog(page),
                                       what="Library filter sheet")
            await open_and_close_twice(phone, lambda: client.menu("Library settings", key="lib-menu"),
                                       lambda: _top_dialog(page), what="Library settings sheet")
            if not client.find(lambda n: client.tooltip_of(n) == "Book actions"):
                await client.tap(key="lib-view")  # the list view has the ⋮ Book actions button
            assert await client.until(lambda: client.find(lambda n: client.tooltip_of(n) == "Book actions"))
            await open_and_close_twice(phone, lambda: client.tap(tooltip="Book actions"), lambda: _top_dialog(page),
                                       what="Library book actions")
            # Book actions › Load for translation: the actions make way for the translate sheet
            await client.tap(tooltip="Book actions")
            assert await client.until(lambda: client.showing(_top_dialog(page)))
            actions = _top_dialog(page)
            await client.tap(text="\U0001f501 Load for translation")
            assert await client.until(lambda: _top_dialog(page) not in (None, actions)
                                      and client.showing(_top_dialog(page)) and len(client.modals()) == 1
                                      and not _in_dialogs(page, actions), 15), client.describe()
            translate = _top_dialog(page)
            await client.close_menu()
            assert await client.until(lambda: not client.modals() and not _in_dialogs(page, translate))
            assert_closed(phone, translate, what="translate sheet from the book actions")
            # Book actions › Files: the sheet closes and the Files screen opens; Back returns to the Library
            await client.tap(tooltip="Book actions")
            assert await client.until(lambda: client.showing(_top_dialog(page)))
            actions = _top_dialog(page)
            await client.tap(text="\U0001f4c1 Files")
            assert await client.until(lambda: app.shell.top_screen is not home
                                      and client.top_route() == app.shell.current_route
                                      and not client.modals() and not _in_dialogs(page, actions), 15), client.describe()
            assert await client.back() == "view"
            assert await client.until(lambda: app.shell.top_screen is home and not client.modals()), client.describe()
            # Android back with the filter sheet open on the Library
            await client.tap(key="lib-filter")
            assert await client.until(lambda: client.showing(_top_dialog(page)))
            sheet = _top_dialog(page)
            # (the Library asks before it pops: can_pop=False + confirm_pop, so back closes the sheet on top)
            assert await client.back() == "modal"
            assert await client.until(lambda: not client.modals() and not _in_dialogs(page, sheet)), client.describe()
            assert_closed(phone, sheet, what="Library filter sheet closed by back")
            assert app.shell.top_screen is home

            # ---- selection mode: long-press a card, the bar's More sheet; back closes it, then leaves selection --
            card = client.find(lambda n: n.get("on_long_press") and "Alpha" in " ".join(_texts_under(client, n)))
            assert card, "no long-pressable Alpha card"
            await client.session.dispatch_event(card[0][0]["_i"], "long_press", None)
            assert await client.until(lambda: home.selecting and client.find(lambda n: client.key_of(n) == "bulk-more"))
            await open_and_close_twice(phone, lambda: client.tap(key="bulk-more"), lambda: _top_dialog(page),
                                       what="Library selection More sheet")
            await client.tap(key="bulk-more")
            assert await client.until(lambda: client.showing(_top_dialog(page)))
            sheet = _top_dialog(page)
            assert await client.back() == "modal"
            assert await client.until(lambda: not client.modals() and not _in_dialogs(page, sheet)), client.describe()
            assert_closed(phone, sheet, what="Library selection More sheet closed by back")
            assert home.selecting
            assert await client.back() == "confirm"  # leaves selection mode, stays on the Library
            assert await client.until(lambda: not home.selecting and app.shell.top_screen is home
                                      and not client.modals())

            # ---- Reader --------------------------------------------------------------------------------------
            app.reader.open_book(path=str(workspace / "Alpha.epub"))
            assert await client.until(lambda: isinstance(app.shell.top_screen, ReaderScreen)
                                      and client.top_route() == app.shell.current_route, 15)
            reader = app.shell.top_screen
            assert await client.until(lambda: reader.state == "ready", 30), reader.state
            reader.chrome.set_visible(True)
            assert await client.until(lambda: client.find(lambda n: client.key_of(n) == "reader-more"), 10)
            for key, what in (("reader-more", "Reader ⋯ sheet"), ("reader-aa", "Reader Aa sheet"),
                              ("reader-search", "Reader search sheet")):
                await open_and_close_twice(phone, lambda k=key: client.tap(key=k), lambda: _top_dialog(page),
                                           what=what)
            # Android back with ⋯ open: the Reader asks before it pops, so only the sheet closes
            await client.tap(key="reader-more")
            assert await client.until(lambda: client.showing(_top_dialog(page)))
            more = _top_dialog(page)
            assert await client.back() == "modal"
            assert await client.until(lambda: not client.modals() and not _in_dialogs(page, more)), client.describe()
            assert_closed(phone, more, what="Reader ⋯ closed by back")
            assert app.shell.top_screen is reader and not reader.disposed
            # ⋯ › Bookmarks: the menu makes way for the bookmarks sheet
            await client.tap(key="reader-more")
            assert await client.until(lambda: client.showing(_top_dialog(page)))
            more = _top_dialog(page)
            await client.tap(key="Bookmarks")
            assert await client.until(lambda: _top_dialog(page) not in (None, more) and client.showing(_top_dialog(page))
                                      and len(client.modals()) == 1 and not _in_dialogs(page, more)), client.describe()
            marks = _top_dialog(page)
            await client.close_menu()
            assert await client.until(lambda: not client.modals() and not _in_dialogs(page, marks))
            assert_closed(phone, marks, what="Reader bookmarks sheet")
            # ⋯ › Open Book Details: the menu closes and the Book page opens over the Reader
            await client.tap(key="reader-more")
            assert await client.until(lambda: client.showing(_top_dialog(page)))
            more = _top_dialog(page)
            await client.tap(key="reader-more-details")
            assert await client.until(lambda: isinstance(app.shell.top_screen, BookPageScreen)
                                      and client.top_route() == app.shell.current_route
                                      and not client.modals() and not _in_dialogs(page, more), 20), client.describe()
            book_page = app.shell.top_screen
            # the Book page ⋮ › Translate…: a sheet on the screen a sheet opened
            await open_and_close_twice(phone, lambda: client.menu("Translate…", key="book-menu"),
                                       lambda: _top_dialog(page), what="Book page translate sheet")
            assert await client.back() == "view"
            assert await client.until(lambda: app.shell.top_screen is reader and not client.modals()), client.describe()
            # the Reader asks before it pops (can_pop=False + confirm_pop)
            did = await client.back()
            assert did in ("view", "confirm"), did
            if did == "confirm":
                did = await client.back()
            assert await client.until(lambda: app.shell.top_screen is not reader and not client.modals()), \
                f"{did} {client.describe()}"
            assert book_page.disposed if hasattr(book_page, "disposed") else True

            # ---- Glossaries: Extract glossary (and its Library-book picker swapped in from it) ---------------
            app.navigate_to("glossary", reset=True)
            assert await client.until(lambda: app.shell.stack and app.shell.current_route == "/glossary"
                                      and client.top_route() == "/glossary", 15)
            assert await client.until(lambda: client.find(lambda n: "Extract glossary" in client.texts_of(n)), 15), \
                f"the Glossaries screen has no Extract glossary: {_screen_texts(client)[:80]}"
            await open_and_close_twice(phone, lambda: client.tap(text="Extract glossary"), lambda: _top_dialog(page),
                                       what="Extract glossary sheet")
            await _back_to_chat(phone)
            assert await client.back() == "exit"
        finally:
            await _stop_phone(phone, runs)

    asyncio.run(scenario())


def _screen_texts(client: PhoneClient) -> list:
    return [t for root in client.surface() for t in _texts_under(client, root)]


def _texts_under(client: PhoneClient, node: dict) -> list:
    return [t for n, _c in client._walk(node) for t in client.texts_of(n)]
