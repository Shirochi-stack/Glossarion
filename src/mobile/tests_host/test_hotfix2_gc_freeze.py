"""Hotfix 2: the startup heap is frozen (``gc.freeze``) once the backend warm import is done, on
real devices only, so gen-2 GC passes stop re-scanning the backend while lists scroll.

Run from src/mobile:  python -m pytest -p no:cacheprovider tests_host/test_hotfix2_gc_freeze.py -q
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from glossarion_mobile import runtime_bootstrap as rb  # noqa: E402


@pytest.fixture(autouse=True)
def _never_leave_the_heap_frozen(monkeypatch):
    monkeypatch.delenv("GLOSSARION_SIMULATE_PLATFORM", raising=False)
    gc.unfreeze()
    yield
    gc.unfreeze()  # a frozen heap would keep every earlier test's garbage for the rest of the run


def _paths(platform: str):
    return SimpleNamespace(is_device=platform in ("android", "ios"))


@pytest.mark.parametrize("platform", ["android", "ios"])
def test_device_freezes_the_heap_once(platform, monkeypatch):
    count = rb.freeze_startup_heap(_paths(platform))
    # frozen objects still die by refcount, so the count may only shrink after the freeze
    assert count > 0 and 0 < gc.get_freeze_count() <= count

    def no_second_pass(*args, **kwargs):
        raise AssertionError("the heap is collected and frozen once per process")

    monkeypatch.setattr(rb.gc, "collect", no_second_pass)
    monkeypatch.setattr(rb.gc, "freeze", no_second_pass)
    assert 0 < rb.freeze_startup_heap(_paths(platform)) <= count


def test_desktop_and_simulated_devices_are_left_alone(monkeypatch):
    assert rb.freeze_startup_heap(_paths("desktop")) == 0
    monkeypatch.setenv("GLOSSARION_SIMULATE_PLATFORM", "android")
    assert rb.freeze_startup_heap(_paths("android")) == 0
    assert gc.get_freeze_count() == 0


def test_no_paths_means_no_freeze(monkeypatch):
    monkeypatch.setattr(rb, "get_paths", lambda: None)
    assert rb.freeze_startup_heap() == 0
    assert gc.get_freeze_count() == 0


def test_backend_ready_freezes_with_the_app_paths(monkeypatch):
    app_module = pytest.importorskip("glossarion_mobile.app")
    calls = []
    monkeypatch.setattr(rb, "freeze_startup_heap", lambda paths=None: calls.append(paths) or 1)
    published = []
    fake = SimpleNamespace(state=SimpleNamespace(backend=SimpleNamespace(set=published.append)),
                           paths=_paths("android"))
    app_module.GlossarionApp._on_backend_ready(fake, {"ok": True})
    assert published == [{"ok": True}] and calls == [fake.paths]


def test_a_failing_freeze_never_breaks_backend_ready(monkeypatch):
    app_module = pytest.importorskip("glossarion_mobile.app")

    def boom(paths=None):
        raise RuntimeError("no gc")

    monkeypatch.setattr(rb, "freeze_startup_heap", boom)
    published = []
    fake = SimpleNamespace(state=SimpleNamespace(backend=SimpleNamespace(set=published.append)),
                           paths=_paths("android"))
    app_module.GlossarionApp._on_backend_ready(fake, {"ok": False, "failed": {"x": "y"}})
    assert published == [{"ok": False, "failed": {"x": "y"}}]
