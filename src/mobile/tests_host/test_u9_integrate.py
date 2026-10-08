"""U9 Integrate: every milestone has shipped, so nothing in the real app is a placeholder any more.

* Every route in ``ui/router.py`` carries a milestone inside ``ui/screens/base.SHIPPED_MILESTONES``
  (U0-U9), and every milestone in that set is one the plan defines.
* The real app on a fake Flet session (every feature installed by ``app.py``): each static view or
  full-screen route, plus the Series and compose routes, builds a feature screen, never a
  ``PlaceholderScreen`` ("This screen arrives in U…"); the Settings home shows no "Arrives in"
  chip on any route tile and every Tools hub tile opens.
* No user-facing "arrives in U<n>" / "planned for U<n>" text is left in the app sources.

Run from src/mobile with the mobile venv:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_u9_integrate.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import re
import sys
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _load(name: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not _has("flet"), reason="flet not installed")
def test_every_route_milestone_has_shipped():
    from glossarion_mobile.ui.router import ROUTES
    from glossarion_mobile.ui.screens.base import SHIPPED_MILESTONES

    assert SHIPPED_MILESTONES == frozenset(f"U{n}" for n in range(10))
    outside = [(spec.name, spec.milestone) for spec in ROUTES if spec.milestone not in SHIPPED_MILESTONES]
    assert outside == []
    assert {spec.name for spec in ROUTES if spec.milestone == "U9"} == {"series", "settings.updates"}


#: user-facing promises of a later milestone (comments and docstrings may still name milestones)
_PLACEHOLDER = re.compile(r"(arrives?|planned|coming|lands?)\s+(in|for|with)\s+U\d", re.IGNORECASE)


def test_no_later_milestone_promises_in_the_app_strings():
    """String literals only: an f-string template such as ``f"Arrives in {spec.milestone}"`` (the
    generic fallback for a route nothing implements) names no milestone and is checked on the real
    app below instead."""
    import ast

    hits = []
    for path in sorted((APP_DIR / "glossarion_mobile").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", [])
                if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                    docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                if _PLACEHOLDER.search(node.value):
                    hits.append(f"{path.relative_to(APP_DIR)}:{node.lineno}: {node.value[:80]!r}")
    assert hits == []


_TB = _load("test_bootstrap.py", "_glossarion_tb_helpers_u9integ")
storage = _TB.storage
app_env = _TB.app_env

#: routes that need an id: a value the feature resolves (or reports as gone) without a placeholder
_PARAM_ROUTES = {
    "series": "/series/0123456789ab",
    "chat.compose": "/chat/1/compose",
}


@pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")
@pytest.mark.skipif(not (_has("requests") and _has("bs4")), reason="the real features drive the backend")
def test_real_app_has_no_placeholder_screens_or_milestone_chips(app_env):
    from glossarion_mobile.ui.router import FULLSCREEN, ROUTES, VIEW, build_route
    from glossarion_mobile.ui.screens.base import PlaceholderScreen
    from glossarion_mobile.ui.tools.hub import HUB_GROUPS, tile_available

    tf = _load("test_ui_foundations.py", "_glossarion_tf_helpers_u9integ")

    async def scenario():
        _m, _conn, _session, _page, app = await tf._start("android")
        try:
            await tf._wait(lambda: app.state.engine_ready, timeout=60)
            # every U9 feature installed by app.start (they are no-ops only where the platform lacks them)
            for attr in ("series", "updates_feature", "keyboard"):
                assert getattr(app, attr, None) is not None, attr
            assert hasattr(app, "webview_bridge")  # None on a host without flet-webview
            targets = [spec for spec in ROUTES
                       if spec.presentation in (VIEW, FULLSCREEN) and spec.is_static and spec.alias_of is None]
            built = {}
            for spec in targets:
                match = await app.navigate(build_route(spec.name))
                assert match is not None and match.name == spec.name, spec.name
                screen = app.shell.top_screen
                assert screen is not None and not isinstance(screen, PlaceholderScreen), spec.name
                built[spec.name] = type(screen).__name__
                await asyncio.sleep(0.05)
            for name, route in _PARAM_ROUTES.items():
                match = await app.navigate(route)
                assert match is not None and match.name == name, name
                screen = app.shell.top_screen
                assert screen is not None and not isinstance(screen, PlaceholderScreen), name
                built[name] = type(screen).__name__
            assert built["settings.updates"] == "UpdatesScreen" and built["series"] == "SeriesScreen"
            assert built["settings.notifications"] == "NotificationsScreen"

            # Settings home: every route tile opens (no "Arrives in U<n>" ReasonChip)
            await app.navigate("/settings")
            home = app.shell.top_screen
            chips = {name: getattr(tile.trailing, "reason", None) for name, tile in home.route_tiles.items()}
            assert chips and not {n: r for n, r in chips.items() if r}, chips
            # Tools hub: every tool opens
            await app.navigate("/tools")
            hub = app.shell.top_screen
            implemented = getattr(hub, "implemented", frozenset())
            closed = {t.route: tile_available(t, implemented) for _g, tiles in HUB_GROUPS for t in tiles}
            assert not {r: why for r, why in closed.items() if why}, closed
        finally:
            await tf._stop(app)

    asyncio.run(scenario())
