"""Flutter test settings the device UI tests need in Flet's ``flet test`` driver (conftest applies them).

``flet test android`` renders ``<flutter app>/integration_test/app_test.dart`` from Flet's build
template (Flet 1.0.3: ``void main() => runFletDeviceTest(appMain: app.main);``) and then starts
pytest with ``FLET_TEST_FLUTTER_APP_DIR`` set; each ``flet_app`` fixture runs
``flutter test integration_test`` in that directory. ``conftest.pytest_configure`` rewrites the
driver before that, so two settings are in place when the test body runs:

* ``shouldPropagateDevicePointerEvents = true`` on the integration-test binding. By default it
  routes every real pointer event to ``WidgetTester.dispatchEvent``, which only prints "Some
  possible finders for the widgets at ..." and drops the event. ``adb shell input swipe`` is the
  only way these tests can scroll a lazily built list (Flet's RemoteTester has no drag), so the
  Settings home never scrolled down to "Import from desktop" (Build Mobile run 37800059580).
* ``WidgetController.hitTestWarningShouldBeFatal = true``: a tap on a target that is built but
  would not get the pointer (a list row still below the fold, a route mid-transition) fails with
  "would not receive pointer events" instead of tapping empty space; ``UiDriver`` scrolls a step
  and retries.

Both are set after ``runFletDeviceTest`` has created the binding and registered the test, and
before the test body runs, so Flutter's "shouldPropagateDevicePointerEvents was changed by the
test" check sees the same value before and after the body. Only ``flutter_test`` is imported (a
declared dev dependency of every ``flet test`` host).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

__all__ = ["DRIVER_LINE", "DriverPatchError", "MARK", "driver_path", "patch_device_driver", "patch_driver_text"]

#: marks a patched driver (the patch is applied once per provisioned test host)
MARK = "// glossarion-ui-tests: device driver settings"
#: the entry point of Flet 1.0.3's device driver template (integration_test/app_test.dart)
DRIVER_LINE = "void main() => runFletDeviceTest(appMain: app.main);"
IMPORT = "import 'package:flutter_test/flutter_test.dart' show LiveTestWidgetsFlutterBinding, WidgetController;"
MAIN = "\n".join((
    MARK,
    "void main() {",
    "  runFletDeviceTest(appMain: app.main);",
    "  LiveTestWidgetsFlutterBinding.instance.shouldPropagateDevicePointerEvents = true;",
    "  WidgetController.hitTestWarningShouldBeFatal = true;",
    "}",
))


class DriverPatchError(RuntimeError):
    """The ``flet test`` driver is missing or no longer looks like Flet 1.0.3's (revisit this patch
    after a Flet upgrade)."""


def driver_path(flutter_dir: os.PathLike) -> Path:
    return Path(flutter_dir) / "integration_test" / "app_test.dart"


def patch_driver_text(text: str) -> Optional[str]:
    """The patched driver source, or None when ``text`` is patched already."""
    if MARK in text:
        return None
    if text.count(DRIVER_LINE) != 1:
        raise DriverPatchError(f"the `flet test` driver has no single {DRIVER_LINE!r} (Flet template changed?)")
    lines = text.splitlines(keepends=True)
    first_import = next((i for i, line in enumerate(lines) if line.lstrip().startswith("import ")), None)
    if first_import is None:
        raise DriverPatchError("the `flet test` driver has no import directives (Flet template changed?)")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines.insert(first_import, IMPORT + newline)
    return "".join(lines).replace(DRIVER_LINE, MAIN.replace("\n", newline))


def patch_device_driver(flutter_dir: os.PathLike) -> bool:
    """Patch ``<flutter_dir>/integration_test/app_test.dart``: True when patched now, False when it
    already was. Raises ``DriverPatchError`` when the driver is missing or unexpected."""
    path = driver_path(flutter_dir)
    try:
        text = path.read_bytes().decode("utf-8")  # keep the file's own line endings
    except OSError as exc:
        raise DriverPatchError(f"the `flet test` driver was not generated: {path} ({exc})") from exc
    try:
        patched = patch_driver_text(text)
    except DriverPatchError as exc:
        raise DriverPatchError(f"{exc}: {path}") from None
    if patched is None:
        return False
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(patched, encoding="utf-8", newline="")
    os.replace(tmp, path)
    return True
