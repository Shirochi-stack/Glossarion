"""Device UI test: launch, navigation and the self-test (``flet test android``; U9 UI layer).

Fresh install → the Welcome flow is skipped → chat home (composer, output-mode chip) → drawer →
Library → Settings (Logs & diagnostics, Updates listed) → Logs & diagnostics › Run self-test →
PASS. The same flow runs on the host in tests_host/test_ui_flows.py.
"""

import flows


async def test_launch_navigation_and_selftest(ui):
    await flows.dismiss_welcome(ui, timeout=90)
    await flows.smoke_navigation(ui)
    assert await flows.run_selftest(ui) == "PASS"
