"""Host-test isolation shared by every file in tests_host."""

from __future__ import annotations

import sys

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
    # the phone app's start also finds the extension's ``TapAway`` (``ui.components.tap_away``)
    tap_away = sys.modules.get("glossarion_mobile.ui.components.tap_away")
    if tap_away is not None:
        tap_away._region_class = None
    # the app's child interpreters stay for its next job: a test's end is the end of that app
    interpreters = sys.modules.get("glossarion_mobile.services.interpreters")
    if interpreters is not None:
        interpreters.drop_kept_pool()
