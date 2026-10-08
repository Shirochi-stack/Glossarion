"""Host tests for the log font size (owner's post-U8 device report #3: "Log fonts are too large,
reduce it to 8").

Every log surface renders through ``theme.log_text`` at ``tokens.LOG_STYLE`` (8/11, mono family):
LogConsole blocks (job detail Log, Diagnostics Live log, Manga Files log), the Diagnostics
log-file viewer and Check environment lines, and the Reader live Thinking/log pane.
``MONO_STYLE`` (code, editors, ErrorCard) stays 13.

Real data is never touched: every test runs with HOME / USERPROFILE / APPDATA /
GLOSSARION_LIBRARY_DIR / OUTPUT_DIRECTORY / GLOSSARION_DATA_DIR in pytest's tmp dir.

Run from src/mobile:
    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_log_text.py
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

MOBILE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = MOBILE_DIR / "app"
SRC_DIR = MOBILE_DIR.parent
UI_DIR = APP_DIR / "glossarion_mobile" / "ui"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from glossarion_mobile.ui import tokens  # noqa: E402  (pure data, no Flet)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet / msgpack not installed")
needs_live = pytest.mark.skipif(not _has("live_stream"), reason="shared live_stream module not importable")

LOG_HEIGHT = round(11 / 8, 4)  # 1.375: the token's line height, not bodyMedium's inherited 20/14


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """No test reads or writes the user's Library, output folders, home or data folder."""
    for name, sub in (("HOME", "home"), ("USERPROFILE", "home"), ("APPDATA", "appdata"),
                      ("GLOSSARION_LIBRARY_DIR", "lib"), ("OUTPUT_DIRECTORY", "out"),
                      ("GLOSSARION_DATA_DIR", "data")):
        folder = tmp_path / "_iso" / sub
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(folder))
    monkeypatch.setenv("GLOSSARION_HTTP_LOG", "0")
    yield tmp_path / "_iso"


def _assert_log_text(text, *, family: str = "monospace") -> None:
    assert text.size == 8 == tokens.LOG_STYLE.size
    assert text.style is not None and text.style.height == LOG_HEIGHT
    assert text.selectable is True
    assert text.font_family == family


# ==========================================================================
# Token (pure)
# ==========================================================================


def test_log_style_token_is_8_and_mono_stays_13():
    assert tokens.LOG_STYLE == tokens.TypeStyle(8, 11, 400)
    assert "LOG_STYLE" in tokens.__all__
    assert tokens.LOG_STYLE.size < tokens.TYPE_SCALE["label_small"].size  # the smallest log text in the app
    # code, editors and ErrorCard keep the general mono token
    assert tokens.MONO_STYLE == tokens.TypeStyle(13, 18, 400)


def test_log_surfaces_no_longer_size_logs_from_the_mono_token():
    """Static guard: the log surfaces size their text through ``log_text`` only."""
    for rel in ("components/log_console.py", "reader/live_panel.py"):
        source = (UI_DIR / rel).read_text(encoding="utf-8")
        assert "MONO_STYLE" not in source, rel
        assert "log_text(" in source, rel
    tree = ast.parse((UI_DIR / "screens" / "diagnostics.py").read_text(encoding="utf-8"))
    methods = {node.name: node for node in ast.walk(tree) if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))}
    for name in ("view_log", "check_environment"):
        calls = [n for n in ast.walk(methods[name]) if isinstance(n, ast.Call)]
        assert any(getattr(c.func, "id", None) == "log_text" for c in calls), name
        literal_sizes = [kw for c in calls for kw in c.keywords if kw.arg == "size"]
        assert not literal_sizes, name


# ==========================================================================
# theme.log_text (contract C8)
# ==========================================================================


@needs_flet
def test_log_text_helper():
    import flet as ft

    from glossarion_mobile.ui import theme

    plain = theme.log_text("x")
    assert plain.value == "x"
    _assert_log_text(plain)
    _assert_log_text(theme.log_text("x", family="Menlo"), family="Menlo")
    ios = types.SimpleNamespace(platform=ft.PagePlatform.IOS)
    _assert_log_text(theme.log_text("x", page=ios), family="Menlo")
    android = types.SimpleNamespace(platform=ft.PagePlatform.ANDROID)
    _assert_log_text(theme.log_text("x", page=android), family="monospace")
    # a fresh TextStyle per call: blocks never share one style instance
    first, second = theme.log_text("a"), theme.log_text("b")
    assert first.style is not second.style
    # extra Text kwargs pass through
    tinted = theme.log_text("", color=ft.Colors.ON_SURFACE_VARIANT, key="k")
    assert tinted.color == ft.Colors.ON_SURFACE_VARIANT and tinted.key == "k"
    assert "log_text" in theme.__all__


# ==========================================================================
# LogConsole (job detail Log, Diagnostics Live log, Manga Files log)
# ==========================================================================


@needs_flet
def test_log_console_blocks_use_the_log_size():
    from glossarion_mobile.services.dispatcher import LogLine
    from glossarion_mobile.ui.components.log_console import LogConsole

    console = LogConsole(block_lines=3, max_blocks=4, list_height=200)
    lines = [LogLine(i + 1, 0.0, f"line {i}", "error" if i % 2 else "info") for i in range(8)]
    console.on_lines(lines, 0)
    blocks = console.list_view.controls
    assert len(blocks) == 3 and blocks[0].value == "line 0\nline 1\nline 2"
    for block in blocks:
        _assert_log_text(block)  # 8 sp, not the old MONO_STYLE 13
    console.set_filter("errors")
    blocks = console.list_view.controls
    assert "\n".join(b.value for b in blocks) == "line 1\nline 3\nline 5\nline 7"
    for block in blocks:
        _assert_log_text(block)
    assert len({id(b.style) for b in blocks}) == len(blocks)
    # the gap notice is a log block too
    console.on_lines([], 3)
    _assert_log_text(console.list_view.controls[-1])
    # the empty placeholder keeps its body-small theme style
    console.set_filter("thinking")
    assert console.list_view.controls == [console.empty_text] and console.empty_text.size is None


@needs_flet
def test_log_console_on_a_page_serializes_and_follows_the_platform_family():
    """Mounted on a fake iOS session (full msgpack encoding): blocks take the page's mono family
    (Menlo) and size 8 + the pinned line height go over the wire."""
    from glossarion_mobile.services.dispatcher import LogLine
    from glossarion_mobile.ui.components.log_console import LogConsole

    spec = importlib.util.spec_from_file_location("_glossarion_tb_helpers_log_text",
                                                  Path(__file__).with_name("test_bootstrap.py"))
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)

    async def scenario():
        conn, session = tb._fake_session("ios")
        page = session.page
        console = LogConsole(list_height=200)
        page.views[0].controls.append(console)
        page.update()
        before = conn.bytes_sent
        console.on_lines([LogLine(1, 0.0, "🚀 Starting translation", "info")], 0)
        page.update()
        return conn, before, console

    conn, before, console = asyncio.run(scenario())
    block = console.list_view.controls[-1]
    assert block.value == "🚀 Starting translation"
    _assert_log_text(block, family="Menlo")
    assert conn.bytes_sent > before


# ==========================================================================
# Reader live Thinking / log pane
# ==========================================================================


@needs_flet
@needs_live
def test_live_panel_thinking_pane_uses_the_log_size():
    import flet as ft

    from glossarion_mobile.ui.reader import live as rl
    from glossarion_mobile.ui.reader.live_panel import LivePanel

    panel = LivePanel(chapter_file="ch3.xhtml", feed=rl.LiveFeed("ch3.xhtml"), clock=lambda: 100.0)
    _assert_log_text(panel.side_text)  # was MONO_STYLE.size - 1 = 12
    assert panel.side_text.color == ft.Colors.ON_SURFACE_VARIANT and panel.side_text.key == "live-side"
    ios = LivePanel(chapter_file="ch3.xhtml", feed=rl.LiveFeed("ch3.xhtml"), mono_family="Menlo", clock=lambda: 1.0)
    _assert_log_text(ios.side_text, family="Menlo")
    panel.add_lines(["🧠 [gpt] Thinking...", "    planning the chapter"])
    panel.render(force=True)
    assert "planning the chapter" in panel.side_text.value
    _assert_log_text(panel.side_text)


# ==========================================================================
# Diagnostics: log-file viewer and Check environment
# ==========================================================================


def _diagnostics(tmp_path, page=None):
    from glossarion_mobile.state.app_state import AppState
    from glossarion_mobile.ui.screens.diagnostics import DiagnosticsScreen

    class Runner:  # SelfTestRunner surface the screen uses
        current_suite = None

        async def run(self, suite, *, source="button"):
            return {"suite": suite}

    logs = tmp_path / "logs"
    data = tmp_path / "data"
    logs.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    return DiagnosticsScreen(None, page=page, state=AppState(), dispatcher=None, runner=Runner(),
                             paths=types.SimpleNamespace(logs=str(logs), data=str(data)),
                             config_snapshot=lambda: {})


def _texts(control) -> list:
    import flet as ft

    found, stack = [], [control]
    while stack:
        node = stack.pop()
        if isinstance(node, ft.Text):
            found.append(node)
        for attr in ("content", "controls"):
            child = getattr(node, attr, None)
            if isinstance(child, list):
                stack.extend(child)
            elif isinstance(child, ft.Control):
                stack.append(child)
    return found


@needs_flet
def test_diagnostics_log_viewer_uses_the_log_size(tmp_path):
    screen = _diagnostics(tmp_path)
    path = tmp_path / "logs" / "app.log"
    path.write_bytes("12:00 [INFO] started\n12:01 [ERROR] boom".encode("utf-8"))
    sheet = screen.view_log(str(path))
    assert sheet is not None
    body = [t for t in _texts(sheet) if t.value == "12:00 [INFO] started\n12:01 [ERROR] boom"]
    assert len(body) == 1
    _assert_log_text(body[0])  # was size 12
    empty = tmp_path / "logs" / "empty.log"
    empty.write_bytes(b"")
    shown = [t for t in _texts(screen.view_log(str(empty))) if t.value == "(empty)"]
    assert len(shown) == 1
    _assert_log_text(shown[0])


@needs_flet
def test_diagnostics_env_check_lines_use_the_log_size(tmp_path, monkeypatch):
    import flet as ft

    from glossarion_mobile.ui.screens import env_preview

    seen = []

    def fake_check(config):
        seen.append(config)
        return env_preview.EnvCheckResult(True, lines=["✅ [ENV_DEBUG] OK: 1", "[ENV_DEBUG] A=1", "❌ [ENV_DEBUG] B"])

    monkeypatch.setattr(env_preview, "run_env_check", fake_check)

    async def io(fn, *args):
        return fn(*args)

    shown = []
    ios_page = types.SimpleNamespace(platform=ft.PagePlatform.IOS, show_dialog=shown.append)
    for page, family in ((None, "monospace"), (ios_page, "Menlo")):
        screen = _diagnostics(tmp_path, page=page)
        screen.build_body()
        screen._io = io
        screen._push = lambda *controls: None
        result = asyncio.run(screen.check_environment())
        assert result is not None and seen[-1] == {}
        lines = screen.env_check_lines.controls
        assert [t.value for t in lines] == ["[ENV_DEBUG] A=1", "❌ [ENV_DEBUG] B"]
        for text in lines:
            _assert_log_text(text, family=family)  # was BODY_SMALL with a literal "monospace"
            assert text.theme_style is None
        # the Runtime card is not a log and keeps its 12 sp
        assert screen.runtime_text.size == 12 and screen.runtime_text.font_family == family
