"""Host tests for U9 polish (UI_SPEC §1.1-§1.2 tablet / wide, §6.3 motion, §7.3-§7.5).

* Reduce motion (``ui.motion``): shimmer → static tint, scale / rotation → cross-fade; the
  Appearance switch sets it.
* Effective text scale (``ui.text_scale``): the system font scale the invisible probe measures ×
  the Appearance scale drives the >= 160 % rules (composer pills → "Options (n)", the header's
  model-only subtitle and taller bar, the mode chip) on the real shell.
* Tablet SidePanel (``components.surface``): chat settings, Compare, the glossary term sheet and
  job detail open beside the content; ``close_dialog`` closes the panel copy; the chat column and
  the composer's output-mode style follow the narrower main area; navigation and phone widths close
  an unpinned panel.
* Wide master-detail: Settings (sections | section page), the Book page (Overview | tabs), Manga
  (Files | Editor · Settings), switching live with the size class.
* Components: Skeleton, ErrorCard, MasterDetail, PullToRefresh, WindowedList (moved to components).
* Accessibility audit over ``ui/**``: every icon button has a tooltip and a 48 dp target, no
  bare ``ft.Shimmer`` / constant ``animate_rotation`` outside ``ui.motion``; the JobStrip and the
  chat header grow with the text instead of clipping.
* Performance budgets measured on the host (printed with ``-s``): a 10k-entry glossary window,
  1k Library cards, 50k log lines through the LogConsole, a 5k-message transcript window, the UI
  import chain (cold start to shell without the backend), no polling while backgrounded.

Run from src/mobile with the mobile venv:
    .venv/Scripts/python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests_host/test_polish.py
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import os
import subprocess
import sys
import time
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

from glossarion_mobile.ui import responsive  # noqa: E402
from glossarion_mobile.ui import text_scale as ts  # noqa: E402


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


needs_flet = pytest.mark.skipif(not (_has("flet") and _has("msgpack")), reason="flet/msgpack not installed")

# The UI foundation helpers (fake Flet session, the real app on it) and the Library fixtures,
# shared rather than copied (one copy each).
_UF_SPEC = importlib.util.spec_from_file_location("_glossarion_uf_helpers_polish",
                                                  Path(__file__).with_name("test_ui_foundations.py"))
_UF = importlib.util.module_from_spec(_UF_SPEC)
_UF_SPEC.loader.exec_module(_UF)
_LU_SPEC = importlib.util.spec_from_file_location("_glossarion_lu_helpers_polish",
                                                  Path(__file__).with_name("test_library_ui.py"))
_LU = importlib.util.module_from_spec(_LU_SPEC)
_LU_SPEC.loader.exec_module(_LU)
storage = _UF.storage
app_env = _UF.app_env
real_env = _LU.real_env
_TB = _UF._TB

#: Generous host budgets (seconds); the measured times are printed for the report.
BUDGETS = {
    "glossary_10k_window": 3.0,
    "glossary_10k_jump": 3.0,
    "library_1k_cards": 6.0,
    "log_50k_lines": 6.0,
    "transcript_5k_window": 1.0,
    "ui_import": 8.0,
    "shell_ready": 3.0,
}


@pytest.fixture(autouse=True)
def _reset_motion_and_scale():
    """Process-wide switches (reduce motion, the measured system font scale): back to defaults."""
    from glossarion_mobile.ui import motion

    motion.set_reduce_motion(False)
    ts.set_os_scale(1.0)
    yield
    motion.set_reduce_motion(False)
    ts.set_os_scale(1.0)


def _report(name: str, seconds: float) -> None:
    print(f"\nPOLISH_BENCH {name} {seconds * 1000:.1f} ms (budget {BUDGETS[name] * 1000:.0f} ms)")
    assert seconds < BUDGETS[name], f"{name} took {seconds:.2f}s (budget {BUDGETS[name]}s)"


# ==========================================================================
# Pure rules: text scale, responsive, foreground
# ==========================================================================


def test_text_scale_probe_math_and_effective_scale():
    assert ts.scale_from_height(ts.PROBE_SP) == 1.0
    assert ts.scale_from_height(ts.PROBE_SP * 2) == 2.0
    assert ts.scale_from_height(0) is None and ts.scale_from_height("x") is None
    assert ts.scale_from_height(ts.PROBE_SP * 10) == 4.0  # clamped
    seen = []
    unsubscribe = ts.subscribe(seen.append)
    try:
        assert ts.set_os_scale(2.0) and ts.os_scale() == 2.0
        assert not ts.set_os_scale(2.02)  # measurement noise is ignored
        state = types.SimpleNamespace(text_scale=types.SimpleNamespace(value=1.3))
        assert ts.effective(state) == pytest.approx(2.6)
        assert ts.effective(app_scale=0.85) == pytest.approx(1.7)
        assert ts.effective(None) == 2.0
    finally:
        unsubscribe()
    assert seen == [2.0]


def test_layout_follows_the_side_panel_and_the_wide_class():
    wide = responsive.layout_for(1300)
    assert wide.wide and wide.chat_width == 860 and wide.output_row == "full" and not wide.side_panel
    panel = responsive.layout_for(1300, side_panel=True)
    assert panel.chat_width == 1300 - 320 - 380 and panel.side_panel and panel.size_class is wide.size_class
    assert panel.output_row == "icons"  # 600 dp column: the labelled toggles no longer fit
    tablet = responsive.layout_for(1000, side_panel=True)
    assert tablet.chat_width == 320 and tablet.output_row == "chip" and not tablet.wide
    phone = responsive.layout_for(412, side_panel=True)
    assert phone.chat_width == 412 and not phone.side_panel  # phones have no SidePanel
    # the >= 160 % rules take the effective scale (app x system)
    big = responsive.layout_for(1300, 1.3 * 1.6)
    assert big.compact_text and big.output_row == "chip" and big.text_scale == pytest.approx(2.08)


def test_poll_sleep_parks_while_the_app_is_hidden():
    from glossarion_mobile.ui import foreground

    class Page:
        def __init__(self):
            self.app_visible = False
            self.waits = 0

        async def wait_until_visible(self):
            self.waits += 1
            await asyncio.sleep(0)
            self.app_visible = True

    async def scenario():
        page = Page()
        assert not foreground.app_visible(page)
        await foreground.poll_sleep(page, 0)
        assert page.waits == 1 and foreground.app_visible(page)
        assert not await foreground.park_while_hidden(page)  # visible: no wait
        assert not await foreground.park_while_hidden(types.SimpleNamespace(app_visible=False))  # no API: no park
        assert not await foreground.park_while_hidden(None)

    asyncio.run(scenario())


def test_polling_loops_park_in_the_background():
    """Every UI polling loop sleeps through ``foreground`` (UI_SPEC §7.3: no polling while backgrounded)."""
    sources = {
        "glossary/editor.py": "poll_sleep(",
        "reader/reader_view.py": "poll_sleep(",
        "tools/sdlxliff.py": "poll_sleep(",
        "screens/keys.py": "poll_sleep(",
        "chat/chat_view.py": "park_while_hidden(",
    }
    for rel, call in sources.items():
        text = (UI_DIR / rel).read_text(encoding="utf-8")
        assert call in text, rel
    # no bare fixed-interval polling sleeps left in those loops
    editor = (UI_DIR / "glossary" / "editor.py").read_text(encoding="utf-8")
    assert "await asyncio.sleep(POLL_SECONDS)" not in editor
    sdl = (UI_DIR / "tools" / "sdlxliff.py").read_text(encoding="utf-8")
    assert "await asyncio.sleep(POLL_SECONDS)" not in sdl
    keys = (UI_DIR / "screens" / "keys.py").read_text(encoding="utf-8")
    assert "await asyncio.sleep(interval)" not in keys


# ==========================================================================
# Accessibility audit over ui/** (UI_SPEC §0 item 4, §5.0, §6.3, §7.5)
# ==========================================================================

_ICON_BUTTONS = {"ft.IconButton", "ft.FilledIconButton", "ft.FilledTonalIconButton", "ft.OutlinedIconButton"}
#: (file, icon) of icon buttons whose 48 dp target is the Container around them (it takes the tap too).
_TARGET_BY_CONTAINER = {("library/book_card.py", "PLAY_ARROW")}


def _icon_name(node):
    icon = _kw(node, "icon")
    value = getattr(icon, "value", None)
    return value.attr if isinstance(value, ast.Attribute) else None


def _calls():
    for path in sorted(UI_DIR.rglob("*.py")):
        if path.name == "spike.py":  # the U0 device-spike screen
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = str(path.relative_to(UI_DIR)).replace("\\", "/")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = f"{func.value.id}.{func.attr}" if isinstance(func, ast.Attribute) and isinstance(
                    func.value, ast.Name) else ""
                yield rel, node, name


def _kw(node, name):
    return next((k for k in node.keywords if k.arg == name), None)


def _splat(node):
    return any(k.arg is None for k in node.keywords)


def test_every_icon_button_has_a_tooltip_and_a_48dp_target():
    missing_tooltip, missing_target = [], []
    for rel, node, name in _calls():
        if name in _ICON_BUTTONS or name == "ft.PopupMenuButton":
            if _splat(node):
                continue
            if _kw(node, "tooltip") is None:
                missing_tooltip.append(f"{rel}:{node.lineno}")
            if name in _ICON_BUTTONS and _kw(node, "size_constraints") is None:
                if (rel, _icon_name(node)) not in _TARGET_BY_CONTAINER:
                    missing_target.append(f"{rel}:{node.lineno}")
    assert not missing_tooltip, f"icon buttons without a tooltip (spoken label): {missing_tooltip}"
    assert not missing_target, f"icon buttons without size_constraints=HIT_TARGET: {missing_target}"


def test_motion_goes_through_the_reduce_motion_helpers():
    offenders = []
    for rel, node, name in _calls():
        if rel == "motion.py":
            continue
        if name == "ft.Shimmer":
            offenders.append(f"{rel}:{node.lineno} ft.Shimmer (use motion.shimmer)")
        rotation = _kw(node, "animate_rotation")
        if rotation is not None and not isinstance(rotation.value, ast.Call):
            offenders.append(f"{rel}:{node.lineno} constant animate_rotation (use motion.rotation_animation)")
    assert not offenders, offenders


def test_no_fixed_height_around_text_in_the_shell_and_composer():
    """JobStrip and the custom badge were fixed-height text boxes (clipped at 200 %)."""
    strip = (UI_DIR / "shell" / "job_strip.py").read_text(encoding="utf-8")
    assert "self.height =" not in strip
    header = (UI_DIR / "chat" / "header.py").read_text(encoding="utf-8")
    assert "height=16" not in header


# ==========================================================================
# Components
# ==========================================================================


@needs_flet
def test_motion_helpers_and_the_animated_controls():
    import flet as ft

    from glossarion_mobile.ui import motion
    from glossarion_mobile.ui.chat.composer import Composer
    from glossarion_mobile.ui.chat.send_button import SendStopButton
    from glossarion_mobile.ui.chat.send_state import SendInputs
    from glossarion_mobile.ui.screens.appearance import apply_appearance

    calls = []
    unsubscribe = motion.subscribe(calls.append)
    try:
        assert isinstance(motion.shimmer(ft.Text("x")), ft.Shimmer)
        assert motion.switcher_transition(ft.AnimatedSwitcherTransition.SCALE) == ft.AnimatedSwitcherTransition.SCALE
        assert motion.rotation_animation(200) == 200
        page = types.SimpleNamespace(theme=None, dark_theme=None, theme_mode=None, update=lambda: None)
        apply_appearance(page, {"reduce_motion": True})
        assert motion.reduced() and calls == [True]
        assert page.theme.page_transitions.android == ft.PageTransitionTheme.NONE
        static = motion.shimmer(ft.Text("x"))
        assert not isinstance(static, ft.Shimmer) and static.opacity < 1
        assert motion.switcher_transition(ft.AnimatedSwitcherTransition.SCALE) == ft.AnimatedSwitcherTransition.FADE
        assert motion.rotation_animation(200) is None
        composer = Composer()
        composer.set_plus_open(True)
        assert composer.plus_button.animate_rotation is None  # turns at once
        assert composer.plus_button.rotate.angle == pytest.approx(3.14159265 / 4)
        button = SendStopButton()
        button.apply(SendInputs(has_content=True))
        assert button.switcher.transition == ft.AnimatedSwitcherTransition.FADE
        apply_appearance(page, {"reduce_motion": False})
        composer.set_plus_open(False)
        assert composer.plus_button.animate_rotation == 200
        button.apply(SendInputs(has_content=False))
        assert button.switcher.transition == ft.AnimatedSwitcherTransition.SCALE
    finally:
        unsubscribe()


@needs_flet
def test_skeleton_error_card_and_pull_to_refresh():
    import flet as ft

    from glossarion_mobile.ui import motion
    from glossarion_mobile.ui.components.error_card import ErrorCard, error_text
    from glossarion_mobile.ui.components.pull_to_refresh import PullToRefresh, is_top_pull
    from glossarion_mobile.ui.components.skeleton import Skeleton

    for kind in Skeleton.KINDS:
        skeleton = Skeleton(kind, count=3, label=f"Loading {kind}…", key=f"sk-{kind}")
        assert skeleton.control.label == f"Loading {kind}…" and skeleton.control.live_region
        assert isinstance(skeleton.control.content.content, ft.Shimmer)
    motion.set_reduce_motion(True)
    assert not isinstance(Skeleton("rows").control.content.content, ft.Shimmer)  # static tint
    motion.set_reduce_motion(False)

    assert error_text(ValueError("bad row 3")) == "ValueError: bad row 3" and error_text(KeyError()) == "KeyError"
    copied, retried = [], []

    async def retry():
        retried.append(True)

    card = ErrorCard(title="Could not list the glossaries", message=OSError("disk gone"), on_retry=retry,
                     on_copy=copied.append, fix=("Edit raw", lambda: None))
    assert card.icon.icon == ft.Icons.ERROR_OUTLINE and card.title_text.value.startswith("Could not")
    assert card.message_text.value == "OSError: disk gone" and card.message_text.selectable
    assert [b.content for b in card.buttons] == ["Retry", "Copy error", "Edit raw"]
    card._copy()
    assert copied == ["OSError: disk gone"]
    asyncio.run(card._retry())
    assert retried == [True] and card.state == "error" and not card.progress.visible
    long = ErrorCard(message="\n".join(f"line {i}" for i in range(20)))
    assert long.more_button.visible and long.message_text.max_lines == 6
    long._toggle()
    assert long.message_text.max_lines is None and long.more_button.content == "Show less"

    pulls = []

    async def refresh():
        pulls.append(True)
        await asyncio.sleep(0)

    async def scenario():
        pull = PullToRefresh(refresh)
        scroll = types.SimpleNamespace(event_type="update", pixels=0)
        top = types.SimpleNamespace(event_type="overscroll", overscroll=-12.0, pixels=0, min_scroll_extent=0)
        bottom = types.SimpleNamespace(event_type="overscroll", overscroll=8.0, pixels=900, min_scroll_extent=0)
        assert not pull.handle(scroll)  # an ordinary scroll is the caller's
        assert is_top_pull(top) and not is_top_pull(bottom)
        assert pull.handle(top) and pull.handle(top) and pull.handle(bottom)  # overscrolls are consumed
        assert pull.refreshing and pull.pulls == 1  # one refresh per pull, not one per notification
        await asyncio.sleep(0.01)
        assert pulls == [True] and not pull.refreshing and not pull.bar.visible

    asyncio.run(scenario())


@needs_flet
def test_master_detail_swaps_panes_and_releases_details():
    import flet as ft

    from glossarion_mobile.ui.components.master_detail import MasterDetail

    master = ft.Column([ft.Text("list")])
    closed = []
    md = MasterDetail(master, placeholder=ft.Text("pick one"), two_pane=False)
    assert md.control.content is master
    assert not md.show_detail("A", ft.Text("a"))  # single pane: the caller navigates
    assert md.set_two_pane(True) and not md.set_two_pane(True)
    row = md.control.content
    assert isinstance(row, ft.Row) and row.controls[0].content is master and row.controls[1] is md.detail_pane
    owner = object()
    assert md.show_detail("Section A", ft.Text("a"), actions=[ft.IconButton(icon=ft.Icons.SEARCH, tooltip="Search")],
                          owner=owner, on_close=lambda: closed.append("a"))
    assert md.shows(owner) and md.title_text.value == "Section A" and md.header.visible
    first_key = md.switcher.content.key
    md.show_detail("Section B", ft.Text("b"), on_close=lambda: closed.append("b"))
    assert closed == ["a"] and md.switcher.content.key != first_key  # a fresh wrapper per detail
    md.clear_detail()
    assert closed == ["a", "b"] and md.switcher.content is md.placeholder and not md.header.visible
    md.set_two_pane(False)
    assert md.control.content is master


@needs_flet
def test_windowed_list_lives_in_components_and_the_old_path_reexports_it():
    from glossarion_mobile.ui.components import windowed_list as canonical
    from glossarion_mobile.ui.glossary import windowed_list as legacy

    assert legacy.WindowedList is canonical.WindowedList and legacy.STEP == canonical.STEP
    assert legacy.WINDOW_ROWS == canonical.WINDOW_ROWS


@needs_flet
def test_side_panel_owners_and_close_dialog():
    import flet as ft

    from glossarion_mobile.ui.components import surface
    from glossarion_mobile.ui.components.dialogs import close_dialog
    from glossarion_mobile.ui.shell.side_panel import SidePanel

    changes, released = [], []
    panel = SidePanel(on_change=changes.append)

    class Host:
        size_class = responsive.SizeClass.TABLET

        def present(self, content, *, title, owner=None, on_close=None):
            panel.open(title, content, owner=owner, on_close=on_close)
            return True

        def hosts(self, owner):
            return panel.hosts(owner)

        def dismiss(self, owner):
            return panel.dismiss(owner)

    page = types.SimpleNamespace(width=1000, update=lambda: None)
    host = Host()
    surface.register(page, host)
    try:
        sheet = ft.BottomSheet(content=ft.Text("settings"))
        assert surface.is_tablet(page) and not surface.is_wide(page)
        assert surface.present_sheet(page, sheet, title="Chat settings", on_close=lambda: released.append(1))
        assert panel.is_open and panel.content is sheet.content and surface.hosts(page, sheet)
        assert changes == [True] and panel.title_text.value == "Chat settings"
        other = ft.BottomSheet(content=ft.Text("compare"))
        surface.present_sheet(page, other, title="Compare with original")
        assert released == [1] and changes == [True]  # replaced: the first owner was released, still open
        assert close_dialog(page, other) and not panel.is_open and changes == [True, False]
        assert not close_dialog(page, sheet)  # not open anywhere
        Host.size_class = responsive.SizeClass.PHONE
        assert not surface.present_sheet(page, sheet, title="Chat settings")  # phones: the caller shows it
    finally:
        Host.size_class = responsive.SizeClass.TABLET
        surface.unregister(page, host)
    assert surface.host_for(page) is None and surface.size_class(page) is responsive.SizeClass.TABLET


@needs_flet
def test_job_strip_grows_with_text_and_rate_limits_announcements():
    from glossarion_mobile.state.app_state import JobStripModel
    from glossarion_mobile.ui.shell.job_strip import JobStrip

    strip = JobStrip()
    assert strip.height is None  # 44 dp is the minimum, set by a spacer in the row
    spacer = strip.semantics.content.controls[0]
    assert spacer.height == 44 and spacer.width == 0
    strip.set_model(JobStripModel(title="Translating · Book.epub", subtitle="Ch 1/80"))
    assert strip.semantics.label == "Translating · Book.epub. Ch 1/80"
    strip.set_model(JobStripModel(title="Translating · Book.epub", subtitle="Ch 2/80"))
    assert strip.semantics.label.endswith("Ch 1/80") and strip.subtitle_text.value == "Ch 2/80"  # visible, not spoken
    strip.set_model(JobStripModel(title="Done · Book.epub", subtitle="", state="done"))
    assert strip.semantics.label == "Done · Book.epub."  # a state change is announced at once


@needs_flet
def test_composer_collapses_pills_and_the_header_grows_at_large_text():
    from glossarion_mobile.ui.chat.composer import Composer
    from glossarion_mobile.ui.chat.header import ChatHeader

    opened = []
    composer = Composer(on_pill=opened.append)
    composer.set_pills([("glossary", "Glossary: Off"), ("thinking", "Thinking off"), ("multipass", "Multipass on")])
    assert [c.key for c in composer.pills_row.controls] == ["pill-glossary", "pill-thinking", "pill-multipass"]
    composer.set_compact_text(True)
    (chip,) = composer.pills_row.controls
    assert chip.key == "pill-options" and chip.label.value == "Options (3)" and composer.text_field.max_lines == 4
    chip.on_click(None)
    assert opened == ["options"]
    composer.set_compact_text(False)
    assert len(composer.pills_row.controls) == 3
    composer.set_text_scale(1.3 * 1.5)
    assert composer.text_field.text_style.size == pytest.approx(15 * 1.95)

    header = ChatHeader()
    bar = header.build(tablet=False)
    assert bar.toolbar_height == 56
    header.set_compact_text(True)
    header.set_text_scale(2.0)
    assert bar.toolbar_height == 80 and header.subtitle_text.endswith("▾")  # model span only
    header.set_text_scale(1.0)
    assert bar.toolbar_height == 56
    tablet_bar = header.build(tablet=True)
    header.set_text_scale(1.8)
    assert tablet_bar.height == header.bar_height() > 56


# ==========================================================================
# The real shell: tablet SidePanel, text-scale probe, wide master-detail
# ==========================================================================


@needs_flet
def test_tablet_side_panel_hosts_chat_settings_compare_and_job_detail(app_env):
    import flet as ft

    from glossarion_mobile.ui.chat.media_cards import CompareSheet
    from glossarion_mobile.ui.components import surface
    from glossarion_mobile.ui.screens.jobs import open_job_in_panel

    async def scenario():
        _m, conn, session, page, app = await _UF._start("android", width=1000)
        try:
            shell = app.shell
            assert shell.tablet and surface.host_for(page) is shell
            await _UF._wait(lambda: app.chat_view.bound)
            assert app.chat_view.composer.output_row.style_name == "icons"  # 700 dp column
            sheet = app.chat_view.open_chat_settings()
            assert sheet is not None and sheet.in_panel and shell.side_panel.is_open
            assert not sheet.dialog.open  # not a bottom sheet on tablets
            assert shell.side_panel.content is sheet.dialog.content
            # the main area lost 380 dp: the chat column is 320 dp, so the mode control is the chip
            assert shell.chat_layout().chat_width == 320
            assert app.chat_view.composer.output_row.style_name == "chip"
            sheet.set_value("disable_thinking", True)  # edits re-render inside the panel
            assert sheet.value_of("disable_thinking") is True
            sheet.close()
            assert not shell.side_panel.is_open and app.chat_view.composer.output_row.style_name == "icons"
            # Android back closes an open panel before anything else (UI_SPEC §1.6 rule 1)
            app.chat_view.open_chat_settings()
            root = page.views[0]
            assert shell.side_panel.is_open and root.can_pop is False
            await shell._on_root_confirm_pop(types.SimpleNamespace(control=root))
            assert not shell.side_panel.is_open and root.can_pop is True and _UF._routes(page) == ["/"]
            compare = CompareSheet("one\n\ntwo", "one\n\nTWO")
            compare.show(page)
            assert shell.side_panel.hosts(compare.dialog) and shell.side_panel.title_text.value == "Compare with original"
            # job detail: a screen in the panel, disposed when replaced or closed
            assert open_job_in_panel(page, "nojob123")
            detail = shell.panel_screen
            assert detail is not None and shell.side_panel.hosts(detail)
            disposed = []
            detail.dispose = lambda d=detail: disposed.append(d)
            assert _UF._routes(page) == ["/"]  # nothing was pushed
            # navigating closes an unpinned panel (its screen disposes)
            await app.navigate("/jobs")
            assert not shell.side_panel.is_open and disposed == [detail] and shell.panel_screen is None
            # a pinned panel stays across navigation on wide screens
            await session.dispatch_event(page._i, "resize", {"width": 1300, "height": 800})
            assert shell.size_class is responsive.SizeClass.WIDE and shell.side_panel.pin_button.visible
            assert open_job_in_panel(page, "nojob456")
            shell.side_panel._toggle_pin()
            await app.navigate("/tools")
            assert shell.side_panel.is_open and shell.side_panel.pinned
            # phones have no SidePanel: going narrow closes it
            await session.dispatch_event(page._i, "resize", {"width": 412, "height": 860})
            assert not shell.tablet and not shell.side_panel.is_open
            assert not open_job_in_panel(page, "nojob789")
            assert isinstance(page.views[0].drawer, ft.NavigationDrawer)
        finally:
            await _UF._stop(app)

    asyncio.run(scenario())


@needs_flet
def test_system_font_scale_probe_drives_the_compact_rules(app_env):
    async def scenario():
        _m, conn, session, page, app = await _UF._start("android", width=412)
        try:
            shell = app.shell
            assert any(c is shell.text_probe.control for c in page.overlay)
            await _UF._wait(lambda: app.chat_view.bound)
            composer, header = app.chat_view.composer, app.chat_view.header
            assert not shell.layout.compact_text and header.wrapper.toolbar_height == 56
            # Android at 200 % font size: the probe line is twice as tall
            shell.text_probe._on_size(types.SimpleNamespace(width=40.0, height=ts.PROBE_SP * 2))
            assert ts.os_scale() == 2.0 and shell.layout.compact_text
            assert shell.layout.text_scale == pytest.approx(2.0)
            assert composer.text_field.max_lines == 4 and header.compact_text
            assert header.wrapper.toolbar_height == 80
            # Appearance 120 % on top of it
            app.state.text_scale.set(1.2)
            assert shell.layout.text_scale == pytest.approx(2.4) and header.wrapper.toolbar_height > 80
            shell.text_probe._on_size(types.SimpleNamespace(width=40.0, height=ts.PROBE_SP))
            app.state.text_scale.set(1.0)
            assert not shell.layout.compact_text and composer.text_field.max_lines == 6
            assert header.wrapper.toolbar_height == 56
        finally:
            await _UF._stop(app)

    asyncio.run(scenario())


@needs_flet
def test_chat_text_size_scales_the_transcript_and_composer(app_env):
    async def scenario():
        _m, conn, session, page, app = await _UF._start("android", width=412)
        try:
            await _UF._wait(lambda: app.chat_view.bound)
            view = app.chat_view
            assert view.scale_box.theme is None and view.scale_box.content is view.column
            sheet = view.open_chat_settings()
            sheet._set_text_scale(1.3)
            assert view.chat_text_scale == pytest.approx(1.3)
            body_large = view.scale_box.theme.text_theme.body_large
            assert body_large.size == pytest.approx(15 * 1.3) and view.scale_box.dark_theme is view.scale_box.theme
            assert view.composer.text_field.text_style.size == pytest.approx(15 * 1.3)
            app.state.text_scale.set(1.2)  # Appearance on top
            assert view.scale_box.theme.text_theme.body_large.size == pytest.approx(15 * 1.56)
            sheet._set_text_scale(1.0)
            assert view.scale_box.theme is None and view.composer.text_field.text_style.size == pytest.approx(18)
            sheet.close()
        finally:
            await _UF._stop(app)

    asyncio.run(scenario())


@needs_flet
def test_wide_settings_and_manga_are_master_detail(app_env):
    async def scenario():
        _m, conn, session, page, app = await _UF._start("android", width=1300)
        try:
            await app.navigate("/settings")
            home = app.shell.top_screen
            assert type(home).__name__ == "SettingsHome" and home.md is not None and home.md.two_pane
            if not home.section_tiles:
                pytest.skip("settings schema not available")
            section_id = next(iter(home.section_tiles))
            route = home.open_section(section_id)
            assert route == f"/settings/s/{section_id}" and _UF._routes(page) == ["/"]  # nothing pushed (tablet root)
            assert app.shell.current_route == "/settings"
            detail = home.detail_page
            assert type(detail).__name__ == "SectionPage" and home.md.shows(detail)
            assert home.section_tiles[section_id].selected
            await session.dispatch_event(page._i, "resize", {"width": 1000, "height": 800})
            # one pane again: the section page was released (disposed) and the list is the body
            assert not home.md.two_pane and home.md.detail is None and not home.md.shows(detail)
            assert not home.section_tiles[section_id].selected and home.md.control.content is home.md.master
            home.open_section(section_id)  # one pane: the section is a pushed screen again
            assert await _UF._wait(lambda: app.shell.current_route == f"/settings/s/{section_id}")
            await session.dispatch_event(page._i, "resize", {"width": 1300, "height": 800})
            await app.navigate("/tools/manga?tab=editor")
            manga = app.shell.top_screen
            if type(manga).__name__ != "MangaScreen":
                pytest.skip("manga screen not installed")
            assert manga.wide and manga.tab_names == ("editor", "settings") and manga.current_tab == "editor"
            files_pane = manga.layout_slot.content.controls[0]
            assert files_pane.content is manga.tab_bodies["files"]
            manga.select_tab("files")  # always on screen: the tabs stay
            assert manga.current_tab == "editor"
            manga.select_tab("settings")
            assert manga.current_tab == "settings"
            await session.dispatch_event(page._i, "resize", {"width": 412, "height": 860})
            assert not manga.wide and manga.tab_names == ("files", "settings", "editor")
            assert manga.current_tab == "settings" and manga.layout_slot.content is manga.tabs
            assert manga.tabs.key == "manga-tabs-2"  # a re-arranged subtree gets a fresh key
        finally:
            await _UF._stop(app)

    asyncio.run(scenario())


@needs_flet
def test_wide_book_page_keeps_the_overview_beside_the_tabs(real_env):
    if _LU._core("progress_core", "build_book_progress") is None:
        pytest.skip("progress core not importable")
    import flet as ft

    from glossarion_mobile.ui.library.book_page import BookPageScreen, WIDE_TABS
    from glossarion_mobile.ui.router import parse_route

    service = real_env["service"]

    async def scenario():
        conn, session = _TB._fake_session("android")
        session.apply_page_patch({"width": 1300, "height": 860})
        page = session.page
        await service.refresh()
        book = service.snapshot.in_progress[0]
        screen = BookPageScreen(parse_route(f"/library/book/{service.bid_for(book)}?tab=overview"),
                                _LU._ctx(page, service))
        screen.actions()
        _LU._mount(page, screen.get_body())
        assert screen.wide and screen.tab_names == WIDE_TABS and screen.current_tab == "chapters"
        row = screen.layout_slot.content
        assert isinstance(row, ft.Row) and row.controls[0].content is screen.tab_bodies["overview"]
        await screen.load()
        page.update()
        screen.set_tab("overview")  # its own pane: nothing to select
        assert screen.current_tab == "chapters"
        screen.set_tab("output")
        assert screen.current_tab == "output"
        screen.apply_size_class(responsive.SizeClass.PHONE)
        page.update()
        assert not screen.wide and screen.tab_names[0] == "overview" and screen.current_tab == "output"
        assert screen.layout_slot.content is screen.tabs and screen.tabs.key == "book-tabs-2"
        screen.set_tab("overview")
        assert screen.current_tab == "overview"
        screen.dispose()

    asyncio.run(scenario())


@needs_flet
def test_drawer_unified_search_through_feature_providers(app_env):
    """Drawer search (UI_SPEC §1.3 item 2): Books, Glossaries and Files come from the features."""
    from glossarion_mobile import runtime_bootstrap as rb
    from glossarion_mobile.ui.shell.drawer import SearchHit

    async def scenario():
        _m, conn, session, page, app = await _UF._start("android", width=412)
        try:
            drawer = app.drawer
            assert {"books", "glossaries", "files"} <= set(drawer.search_providers)
            # the real Files provider over the app's output root
            output = Path(rb.get_paths().output)
            (output / "Dragon Saga").mkdir(parents=True, exist_ok=True)
            (output / "Dragon Saga" / "notes.txt").write_text("x", encoding="utf-8")
            drawer.set_search_group("files")
            drawer.set_query("dragon")
            assert drawer.search_progress.visible  # the provider runs after the debounce
            assert await _UF._wait(lambda: ("files", "dragon") in drawer.search_hits)
            hits = drawer.search_hits[("files", "dragon")]
            assert hits and hits[0].title == "Dragon Saga" and hits[0].icon == "FOLDER"
            assert not drawer.search_progress.visible
            hits[0].open()
            assert await _UF._wait(lambda: app.shell.current_route.startswith("/tools/files/output/"))
            # a provider's rows open their surface
            opened = []
            drawer.register_search("books", lambda q: [SearchHit(title=f"{q} one", subtitle="In progress",
                                                                  open=lambda: opened.append(q), key="b1")])
            drawer.set_search_group("books")
            drawer.set_query("night")
            assert await _UF._wait(lambda: ("books", "night") in drawer.search_hits)
            row = next(c for c in drawer.body.controls if getattr(c, "key", None) == "search-books-b1")
            row.on_click(None)
            assert opened == ["night"]

            async def failing(query):
                raise OSError("disk gone")

            drawer.register_search("glossaries", failing)
            drawer.set_search_group("glossaries")
            assert await _UF._wait(lambda: ("glossaries", "night") in drawer.search_errors)
            assert any("Search failed: disk gone" in str(getattr(getattr(c, "content", None), "value", ""))
                       for c in drawer.body.controls)
            drawer.search_providers.pop("glossaries")
            drawer.set_search_group("glossaries")
            assert any("not available" in str(getattr(getattr(c, "content", None), "value", ""))
                       for c in drawer.body.controls)
            drawer.set_query("")
        finally:
            await _UF._stop(app)

    asyncio.run(scenario())


# ==========================================================================
# Loading / empty / error states (UI_SPEC §7.4)
# ==========================================================================


@needs_flet
def test_glossaries_home_skeleton_then_error_card_with_retry():
    import flet as ft

    from glossarion_mobile.ui.glossary.home import GlossariesScreen

    calls = {"n": 0}

    class Service:
        def mode_label(self):
            return "Balanced"

        def list_glossaries(self):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("Glossary folder unreadable")
            return []

        def count_entries(self, path):
            return None

    notes = []

    class Ctx:
        service = Service()
        feature = None
        page = None

        @staticmethod
        async def io(fn, *args):
            return fn(*args)

        @staticmethod
        def spawn(coro):
            return asyncio.ensure_future(coro)

        @staticmethod
        def say(text, *args):
            notes.append(text)

        @staticmethod
        def push(*controls):
            pass

    async def scenario():
        screen = GlossariesScreen(None, Ctx())
        screen.get_body()
        assert screen.list_holder.content.label == "Loading glossaries…"
        await screen.refresh()
        card = screen.list_holder.content
        assert type(card).__name__ == "ErrorCard" and "OSError: Glossary folder unreadable" in card.message_text.value
        await card._retry()
        assert screen.error is None and screen.list_holder.content.key == "gh-empty"
        assert notes and notes[0].startswith("Could not list the glossaries")
        assert isinstance(screen.list_holder.content, ft.Container)

    asyncio.run(scenario())


# ==========================================================================
# Performance budgets (UI_SPEC §7.3; design M8: 10k glossary, 1k Library, 50k log lines)
# ==========================================================================


@needs_flet
def test_budget_glossary_window_of_10k_entries():
    import flet as ft

    from glossarion_mobile.ui.components.windowed_list import STEP, WINDOW_ROWS, WindowedList

    def row(item, position):
        return ft.Container(content=ft.Column([ft.Text(item["raw"]), ft.Text(item["translated"])], spacing=2),
                            on_click=lambda e: None)

    items = [{"key": f"e{i}", "raw": f"이름{i}", "translated": f"Name {i}"} for i in range(10_000)]
    windowed = WindowedList(build_row=row, key_of=lambda item: item["key"])
    start = time.perf_counter()
    windowed.set_items(items)
    _report("glossary_10k_window", time.perf_counter() - start)
    assert windowed.mounted_count == STEP and windowed.windowed  # 150 rows, not 10k
    start = time.perf_counter()
    asyncio.run(windowed.jump_to("e9876", settle=0))
    _report("glossary_10k_jump", time.perf_counter() - start)
    assert windowed.window_start == 9000 and windowed.mounted_count <= WINDOW_ROWS


@needs_flet
def test_budget_1k_library_cards():
    from glossarion_mobile.ui.library.book_card import BookCard
    from glossarion_mobile.ui.library.models import build_card

    books = [_LU._book("in_progress", done=i % 40, total=40, name=f"Book {i}") for i in range(1000)]
    start = time.perf_counter()
    cards = [BookCard(build_card(book, key=f"k{i}", bid=f"{i:012x}"), card_w=120, cover_h=170)
             for i, book in enumerate(books)]
    _report("library_1k_cards", time.perf_counter() - start)
    assert len(cards) == 1000
    # the home grid appends ``epub_library_page_size`` cards per scroll step, lazily built
    home = (UI_DIR / "library" / "home.py").read_text(encoding="utf-8")
    assert "build_controls_on_demand=True" in home


@needs_flet
def test_budget_50k_log_lines_stay_capped():
    from glossarion_mobile.services.dispatcher import LogLine
    from glossarion_mobile.ui.components.log_console import LogConsole

    console = LogConsole()
    lines = [LogLine(seq=i, text=f"[{i:05d}] Translating chapter {i // 100}: request {i % 100}", kind="info",
                     ts=0.0) for i in range(50_000)]
    start = time.perf_counter()
    for index in range(0, len(lines), 400):  # the dispatcher pump hands over <= 400 lines per tick
        console.on_lines(lines[index:index + 400], 0)
    _report("log_50k_lines", time.perf_counter() - start)
    assert len(console.list_view.controls) <= console.max_blocks == 100
    assert len(console.lines) <= console.block_lines * console.max_blocks


def test_budget_transcript_window_of_5k_messages():
    from glossarion_mobile.ui.chat import transcript_model

    messages = []
    for i in range(2500):
        messages.append(["user", f"paragraph {i} " * 20])
        messages.append(["assistant", f"translated {i} " * 25])
    transcript_model.tail_window(messages[:10], 20)  # first call: the shared helpers load lazily
    start = time.perf_counter()
    window = transcript_model.tail_window(messages, 20, budget=120_000)
    _report("transcript_5k_window", time.perf_counter() - start)
    first, last = window[0], window[1]
    assert last == len(messages) and 0 < last - first <= 20


@needs_flet
def test_budget_cold_start_to_shell(app_env):
    """``main(page)`` to GLOSSARION_READY (the shell is on screen): the features and the backend warm
    import come after it (UI_SPEC §7.3 budget: < 3 s on a mid-range phone; the host is faster)."""
    async def scenario():
        started = time.time()
        _m, conn, session, page, app = await _UF._start("android", width=412)
        try:
            assert app.ready_at is not None
            _report("shell_ready", app.ready_at - started)
            assert app.shell is not None and page.views and page.views[0].route == "/"
        finally:
            await _UF._stop(app)

    asyncio.run(scenario())


def test_budget_ui_import_does_not_load_the_backend(tmp_path):
    """Cold start to shell: the app module and its UI import chain load without the translation
    backend (the warm import runs on a background thread after the shell is up, UI_SPEC §7.3)."""
    if not _has("flet"):
        pytest.skip("flet not installed")
    script = tmp_path / "probe_import.py"
    script.write_text(
        "import sys, time\n"
        f"sys.path.insert(0, {str(APP_DIR)!r})\n"
        "t = time.perf_counter()\n"
        "import glossarion_mobile.app\n"
        "print('SECONDS', time.perf_counter() - t)\n"
        "heavy = [m for m in ('unified_api_client', 'TransateKRtoEN', 'translation_pipeline', 'ebooklib',\n"
        "                     'tiktoken', 'openai', 'anthropic', 'google.genai', 'cv2', 'onnxruntime')\n"
        "         if m in sys.modules]\n"
        "print('HEAVY', ','.join(heavy))\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env.pop("GLOSSARION_BACKEND_DIR", None)
    env["GLOSSARION_LIBRARY_DIR"] = str(tmp_path / "lib")
    env["OUTPUT_DIRECTORY"] = str(tmp_path / "out")
    result = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True, timeout=120,
                            env=env, cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr[-2000:]
    out = dict(line.split(" ", 1) for line in result.stdout.splitlines() if " " in line)
    _report("ui_import", float(out["SECONDS"]))
    assert out.get("HEAVY", "").strip() == "", f"the UI import pulled in backend modules: {out.get('HEAVY')}"
