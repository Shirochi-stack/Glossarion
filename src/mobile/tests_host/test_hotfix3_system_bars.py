"""System-bar protection survives fold/unfold, navigation, and custom-sheet rebuilds.

The release APK is additionally checked on an Android emulator with its real
navigation/taskbar insets; these tests cover the shell and dialog lifecycles.
"""

import asyncio
import importlib.util
from pathlib import Path

import flet as ft
import pytest

_spec = importlib.util.spec_from_file_location(
    "_system_bars_helpers", Path(__file__).with_name("test_sidebar_footer.py"))
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)
uf = helpers.uf
app_env = helpers.app_env
storage = helpers.storage


def _protected(root, target):
    path = helpers._path(root, target)
    assert path, "the live control must still be mounted after a layout change"
    areas = [c for c in path if isinstance(c, ft.SafeArea)]
    assert any(c.avoid_intrusions_bottom for c in areas), "bottom action can enter the system bar"
    assert any(c.avoid_intrusions_left and c.avoid_intrusions_right for c in areas)
    # A keyboard already resizes the Scaffold body. Keeping viewPadding here
    # would leave an extra system-bar gap above the keyboard.
    assert all(not c.maintain_bottom_view_padding for c in areas)


def test_foldable_main_actions_stay_protected_across_routes_and_posture(app_env):
    async def scenario():
        _m, _conn, session, page, app = await uf._start("android", width=412)
        try:
            await uf._route(session, "/library")
            for width, height, nav, keyboard in (
                (412, 915, 48, 0), (1000, 900, 80, 0),
                (1300, 800, 80, 320), (1000, 900, 24, 0), (412, 915, 48, 0),
            ):
                await session.dispatch_event(page._i, "resize", {"width": width, "height": height})
                await helpers._media_change(session, helpers._media(24, nav, keyboard=keyboard))
                _protected(page.views[-1], app.shell.top_screen.get_body())
                if app.shell.tablet:
                    _protected(page.views[-1], app.shell.tablet_back_button)
            # Replacing the main pane for navigation must not replace its inset boundary.
            await session.dispatch_event(page._i, "resize", {"width": 1000, "height": 900})
            boundary = app.shell.main_area
            await uf._route(session, "/settings/logs")
            assert app.shell.main_area is boundary
            _protected(page.views[-1], app.shell.top_screen.get_body())
            await uf._route(session, "/")
            _protected(page.views[-1], app.chat_view.composer)
            assert app.shell.main_area is boundary
        finally:
            await uf._stop(app)

    asyncio.run(scenario())


@pytest.mark.parametrize("nested", [False, True])
def test_custom_sheet_bottom_actions_are_protected_when_content_is_replaced(nested):
    from glossarion_mobile.ui.components import sheet as sheets

    sheets.install_close_buttons()
    dialog = ft.BottomSheet(content=ft.Container(), open=True, show_drag_handle=True, scrollable=True)
    for title in ("Save preset", "Save edited preset"):
        action = ft.FilledButton(content=title)
        body = ft.Column([ft.Text("Custom editor"), action], tight=True)
        original = sheets.sheet_frame(body) if nested else body
        dialog.content = original
        dialog.before_update()
        wrapped = dialog.content
        dialog.before_update()
        assert dialog.content is wrapped
        assert sheets.sheet_body(dialog) is original
        _protected(dialog, action)
        assert wrapped.key == "sheet-system-safe" and not wrapped.avoid_intrusions_top
