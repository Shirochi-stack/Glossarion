"""Host-test isolation shared by every file in tests_host."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _restore_sheet_hook():
    """The app's start installs the "Close Menu" hook on ``ft.BottomSheet`` for the whole process
    (``ui.components.sheet.install_close_buttons``); a test that starts the app must not leave it on for
    the tests after it, which inspect a sheet's own content."""
    try:
        import flet as ft
    except Exception:  # a test environment without flet
        yield
        return
    original = ft.BottomSheet.before_update
    # hotfix 5: on the phone app the start also hooks every text field (``keyboard_dismiss``: a tap outside
    # closes the keyboard)
    original_field = ft.TextField.before_update
    yield
    ft.BottomSheet.before_update = original
    ft.TextField.before_update = original_field
