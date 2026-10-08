"""``PyTester``: Flet's tester calls on the Python control tree of a running app (host only).

``tests_host/test_ui_flows.py`` runs the device UI flows (``flows.py``) against the real app on
a fake Flet session with this tester, so the keys, tooltips and texts the flows rely on, and
the screens they walk through, are checked without Flutter. It mirrors Flutter's finders:

* the tree is the open dialogs / sheets (top-most first), the top view (with its app bar and
  drawer) and the page overlay; invisible controls and their children are not in it;
* ``find_by_text`` matches a ``Text`` value or a string label (``content`` / ``title`` /
  ``label`` / ``text``) a control renders as text; ``find_by_text_containing`` a substring;
  ``find_by_tooltip`` a tooltip; ``find_by_key`` a key (string or ``ValueKey`` value);
* ``tap`` sends the event Flutter would: ``click`` (or ``tap`` / ``select``) to the nearest
  enabled control with a handler, a switch / checkbox toggles and sends ``change``, a tab sends
  its tab bar's ``change``; ``enter_text`` sets a field's value and sends ``change``.

A ``PopupMenuButton`` opens on the client, so tapping it does nothing here (its items are always
in the tree); the flows tap the button first anyway, as on a device.
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass
from typing import Any, Iterator, Optional

__all__ = ["HostFinder", "HostPicker", "PyTester"]

_SKIP_FIELDS = {"parent", "page", "data", "ref", "key", "tooltip", "badge"}
_TEXT_FIELDS = ("content", "title", "label", "text", "subtitle", "semantics_label", "hint_text")
_HANDLERS = (("on_click", "click"), ("on_tap", "tap"), ("on_select", "select"), ("on_long_press", "long_press"))


@dataclass
class HostFinder:
    id: int
    count: int
    index: int = 0

    @property
    def first(self) -> "HostFinder":
        if self.count == 0:
            raise ValueError("No controls found by this finder.")
        return HostFinder(self.id, 1, 0)

    @property
    def last(self) -> "HostFinder":
        if self.count == 0:
            raise ValueError("No controls found by this finder.")
        return HostFinder(self.id, 1, self.count - 1)

    def at(self, index: int) -> "HostFinder":
        if index < 0 or index >= self.count:
            raise IndexError("Index out of range.")
        return HostFinder(self.id, 1, index)


def _key_value(key: Any) -> Any:
    return getattr(key, "value", key)


def _tooltip(control: Any) -> Optional[str]:
    tip = getattr(control, "tooltip", None)
    if isinstance(tip, str):
        return tip
    message = getattr(tip, "message", None)
    return message if isinstance(message, str) else None


def _children(control: Any) -> Iterator[Any]:
    from flet.controls.base_control import BaseControl

    if not dataclasses.is_dataclass(control):
        return
    for field in dataclasses.fields(control):
        name = field.name
        if name.startswith("_") or name in _SKIP_FIELDS:
            continue
        try:
            value = getattr(control, name)
        except Exception:
            continue
        if isinstance(value, BaseControl):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, BaseControl):
                    yield item


def _texts(control: Any) -> list:
    out = []
    if type(control).__name__ == "Text":
        value = getattr(control, "value", None)
        if isinstance(value, str):
            out.append(value)
        return out
    for name in _TEXT_FIELDS:
        value = getattr(control, name, None)
        if isinstance(value, str) and value:
            out.append(value)
    return out


class PyTester:
    def __init__(self, session: Any, page: Any) -> None:
        self.session = session
        self.page = page
        self._finders: dict = {}
        self._next = 0
        self.events: list = []

    # ---- tree ---------------------------------------------------------------------------------

    def _roots(self) -> list:
        roots = []
        dialogs = getattr(getattr(self.page, "_dialogs", None), "controls", None) or []
        roots += [d for d in reversed(dialogs) if getattr(d, "open", False)]
        views = list(getattr(self.page, "views", None) or [])
        if views:
            roots.append(views[-1])
        roots += list(getattr(self.page, "overlay", None) or [])
        return roots

    def _walk(self) -> Iterator[Any]:
        seen: set = set()
        stack = list(reversed(self._roots()))
        while stack:
            control = stack.pop()
            if id(control) in seen or getattr(control, "visible", True) is False:
                continue
            seen.add(id(control))
            yield control
            children = list(_children(control))
            if type(control).__name__ == "TabBarView":
                # Flutter builds only the selected tab's page
                owner = getattr(control, "parent", None)
                while owner is not None and type(owner).__name__ != "Tabs":
                    owner = getattr(owner, "parent", None)
                index = int(getattr(owner, "selected_index", 0) or 0) if owner is not None else 0
                children = children[index:index + 1]
            stack.extend(reversed(children))

    def _register(self, matches: list) -> HostFinder:
        self._next += 1
        self._finders[self._next] = matches
        return HostFinder(self._next, len(matches))

    # ---- finders -------------------------------------------------------------------------------

    async def find_by_key(self, key: Any) -> HostFinder:
        want = _key_value(key)
        return self._register([c for c in self._walk() if _key_value(getattr(c, "key", None)) == want])

    async def find_by_text(self, text: str) -> HostFinder:
        return self._register([c for c in self._walk() if text in _texts(c)])

    async def find_by_text_containing(self, pattern: str) -> HostFinder:
        return self._register([c for c in self._walk() if any(pattern in t for t in _texts(c))])

    async def find_by_tooltip(self, value: str) -> HostFinder:
        return self._register([c for c in self._walk() if _tooltip(c) == value])

    async def find_by_icon(self, icon: Any) -> HostFinder:
        return self._register([c for c in self._walk() if getattr(c, "icon", None) == icon])

    def control(self, finder: HostFinder) -> Any:
        return self._finders[finder.id][finder.index]

    # ---- actions -------------------------------------------------------------------------------

    def _disabled(self, control: Any) -> bool:
        node = control
        while node is not None:
            if getattr(node, "disabled", False):
                return True
            node = getattr(node, "parent", None)
        return False

    async def _dispatch(self, control: Any, event: str, data: Any = None) -> None:
        self.events.append((type(control).__name__, getattr(control, "key", None), event))
        await self.session.dispatch_event(control._i, event, data)

    async def tap(self, finder: HostFinder) -> None:
        node = self.control(finder)
        while node is not None:
            kind = type(node).__name__
            if kind == "PopupMenuButton":
                return  # the menu opens on the client; its items are in the tree already
            if kind in ("Switch", "Checkbox", "CupertinoSwitch") and getattr(node, "on_change", None):
                if self._disabled(node):
                    return
                node.value = not bool(node.value)
                await self._dispatch(node, "change", str(node.value).lower())
                return
            if kind == "Tab":
                bar = getattr(node, "parent", None)
                tabs = getattr(bar, "tabs", None) or []
                owner = bar
                while owner is not None and type(owner).__name__ != "Tabs":
                    owner = getattr(owner, "parent", None)
                if owner is not None and node in tabs:
                    owner.selected_index = tabs.index(node)
                    await self._dispatch(owner, "change", owner.selected_index)
                return
            if kind == "Segment":
                owner = getattr(node, "parent", None)
                while owner is not None and type(owner).__name__ != "SegmentedButton":
                    owner = getattr(owner, "parent", None)
                if owner is not None and not self._disabled(owner):
                    owner.selected = [node.value]
                    if getattr(owner, "on_change", None) is not None:
                        await self._dispatch(owner, "change", list(owner.selected))
                return
            if kind == "ExpansionTile":
                node.expanded = not bool(getattr(node, "expanded", False))
                if getattr(node, "on_change", None):
                    await self._dispatch(node, "change", str(node.expanded).lower())
                return
            for attr, event in _HANDLERS:
                if getattr(node, attr, None) is not None:
                    if self._disabled(node):
                        return
                    await self._dispatch(node, event)
                    return
            node = getattr(node, "parent", None)

    async def long_press(self, finder: HostFinder) -> None:
        node = self.control(finder)
        while node is not None:
            if getattr(node, "on_long_press", None) is not None:
                await self._dispatch(node, "long_press")
                return
            node = getattr(node, "parent", None)

    async def enter_text(self, finder: HostFinder, text: str) -> None:
        field = self.control(finder)
        field.value = text
        if getattr(field, "on_change", None) is not None:
            await self._dispatch(field, "change", text)

    async def pump(self, duration: Any = None) -> None:
        """Flet's ``DurationValue``: an int is milliseconds (what ``UiDriver`` sends), or an
        ``ft.Duration``; no duration is a short tick."""
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            seconds = duration / 1000.0
        elif hasattr(duration, "in_milliseconds"):  # ft.Duration
            seconds = duration.in_milliseconds / 1000.0
        elif hasattr(duration, "total_seconds"):  # datetime.timedelta
            seconds = duration.total_seconds()
        else:
            seconds = 0.05
        await asyncio.sleep(max(0.01, seconds))

    async def pump_and_settle(self, duration: Any = None) -> None:
        await asyncio.sleep(0.2)

    async def take_screenshot(self, name: str) -> bytes:
        return b""

    async def mouse_hover(self, finder: HostFinder) -> None:
        return None

    async def teardown(self, timeout: Any = None) -> None:
        return None

    # ---- diagnostics ---------------------------------------------------------------------------

    def dump(self, limit: int = 400) -> list:
        """``[(type, key, tooltip, texts)]`` of the visible tree (debugging a flow)."""
        rows = []
        for control in self._walk():
            texts = _texts(control)
            key = getattr(control, "key", None)
            tip = _tooltip(control)
            if texts or key is not None or tip:
                rows.append((type(control).__name__, _key_value(key), tip, texts))
            if len(rows) >= limit:
                break
        return rows


class HostPicker:
    """The native file picker in host runs: the flow arms a file name, the app's picker call
    returns that file (as ``FilePicker.pick_files`` would: objects with ``path`` and ``name``)."""

    def __init__(self, files: dict) -> None:
        self.files = dict(files)  # name -> host path
        self.queue: asyncio.Queue = asyncio.Queue()
        self.picked: list = []

    async def arm(self, name: str) -> None:
        if name not in self.files:
            raise KeyError(name)
        self.queue.put_nowait(name)

    async def choose(self, name: str) -> None:
        return None

    async def pick_files(self, **kwargs: Any) -> list:
        import types

        name = await asyncio.wait_for(self.queue.get(), 30)
        self.picked.append((name, kwargs))
        return [types.SimpleNamespace(path=str(self.files[name]), name=name, size=None)]
