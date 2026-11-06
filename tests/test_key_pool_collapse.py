"""Multi API Key Manager: folding a pool's key list is separate from switching the pool on/off.

Every dedicated pool section (Fallback, Glossary and the spec-driven ones such as Image Gen / Edit)
used to hide its key list through its "Enable <pool> Keys" checkbox, so using it as a collapse
control switched the pool off ("🔑 Image gen/edit key pool: disabled"). Each section now has its own
"▼ Hide keys" / "▶ Show keys" button, remembered per pool in ``multi_api_key_collapsed_pools``, and
the enable checkboxes take focus only from a click (a stray Space cannot toggle them).

Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_key_pool_collapse.py
"""
from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

POOLS = ("fallback", "glossary", "inpainter", "qa_scan")
TOGGLE_KEYS = {"fallback": "use_fallback_keys", "glossary": "use_glossary_keys",
               "inpainter": "use_inpainter_keys", "qa_scan": "use_qa_scan_keys"}
COLLAPSED = "multi_api_key_collapsed_pools"


def _qt(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    # the pool toggles export these; restore them after the test
    for name in ("USE_FALLBACK_KEYS", "USE_GLOSSARY_KEYS", "USE_INPAINTER_KEYS", "INPAINTER_API_KEYS",
                 "USE_QA_SCAN_KEYS", "QA_SCAN_API_KEYS"):
        monkeypatch.delenv(name, raising=False)
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    import multi_api_key_manager as mkm
    if not mkm.HAS_GUI:
        pytest.skip("key manager imported headless")
    return app, mkm


def _config():
    key = {"api_key": "sk-local-test-key-0001", "model": "gpt-image-1", "enabled": True}
    config = {toggle: True for toggle in TOGGLE_KEYS.values()}
    config.update({"fallback_keys": [dict(key)], "glossary_keys": [dict(key)],
                   "inpainter_keys": [dict(key)], "qa_scan_keys": [dict(key)]})
    return config


def _manager(mkm, config, preview_pool=None):
    saved, log = [], []
    gui = SimpleNamespace(config=config, save_config=lambda **_kwargs: saved.append(deepcopy(config)),
                          append_log=log.append)
    dialog = mkm.MultiAPIKeyDialog(None, gui, preview_pool=preview_pool)
    if preview_pool is None:
        for pool in POOLS:
            dialog._ensure_key_pool_section(pool)
    return dialog, saved, log


def _shown(widget):
    """Visible within its window (the dialog itself is never shown in these tests)."""
    while widget is not None and not widget.isWindow():
        if widget.isHidden():
            return False
        widget = widget.parentWidget()
    return True


@pytest.mark.parametrize("pool", POOLS)
def test_hiding_the_key_list_keeps_the_pool_enabled(monkeypatch, pool):
    app, mkm = _qt(monkeypatch)
    config = _config()
    dialog, saved, log = _manager(mkm, config)
    try:
        checkbox = dialog._pool_enable_checkbox(pool)
        button = dialog._pool_collapse_buttons[pool]
        tree = dialog._key_context_tree(pool)
        add_form = dialog._pool_add_form(pool)
        assert checkbox.focusPolicy() == mkm.Qt.ClickFocus
        assert checkbox.isChecked() and _shown(button) and button.text() == "▼ Hide keys"
        assert _shown(tree) and _shown(add_form)

        button.click()
        assert not _shown(tree) and not _shown(add_form)
        assert checkbox.isChecked() and config[TOGGLE_KEYS[pool]] is True
        assert dialog._pool_enabled(pool)
        assert button.text() == "▶ Show keys" and _shown(button)
        assert config[COLLAPSED] == {pool: True} and saved[-1][COLLAPSED] == {pool: True}
        assert saved[-1][TOGGLE_KEYS[pool]] is True
        assert not [line for line in log if "disabled" in line]

        button.click()
        assert _shown(tree) and _shown(add_form) and checkbox.isChecked()
        assert config[COLLAPSED] == {} and button.text() == "▼ Hide keys"
        app.processEvents()
    finally:
        dialog.deleteLater()


def test_enable_switch_only_enables_and_the_fold_is_remembered(monkeypatch):
    app, mkm = _qt(monkeypatch)
    config = _config()
    dialog, _saved, log = _manager(mkm, config)
    try:
        pool = "inpainter"
        checkbox = dialog._pool_enable_checkbox(pool)
        button = dialog._pool_collapse_buttons[pool]
        tree = dialog._key_context_tree(pool)
        button.click()
        # switching off: no list and no fold button; the fold is kept
        checkbox.setChecked(False)
        assert config["use_inpainter_keys"] is False and any("disabled" in line for line in log)
        assert not _shown(tree) and not _shown(button)
        assert config[COLLAPSED] == {pool: True}
        # switching on again does not unfold it
        checkbox.setChecked(True)
        assert config["use_inpainter_keys"] is True
        assert not _shown(tree) and _shown(button) and button.text() == "▶ Show keys"
        # the other pools were not touched
        for other in POOLS[:2] + POOLS[3:]:
            assert _shown(dialog._key_context_tree(other)) and dialog._pool_enable_checkbox(other).isChecked()
        app.processEvents()
    finally:
        dialog.deleteLater()

    # a new manager opens it folded (and still enabled)
    reopened, _saved, _log = _manager(mkm, config)
    try:
        assert reopened._pool_enable_checkbox("inpainter").isChecked()
        assert not _shown(reopened._key_context_tree("inpainter"))
        assert reopened._pool_collapse_buttons["inpainter"].text() == "▶ Show keys"
        assert _shown(reopened._key_context_tree("qa_scan"))
    finally:
        reopened.deleteLater()


def test_a_disabled_pool_looks_as_before(monkeypatch):
    app, mkm = _qt(monkeypatch)
    config = _config()
    config["use_qa_scan_keys"] = False
    dialog, _saved, _log = _manager(mkm, config)
    try:
        checkbox = dialog._pool_enable_checkbox("qa_scan")
        assert not checkbox.isChecked() and checkbox.focusPolicy() == mkm.Qt.ClickFocus
        assert not _shown(dialog._key_context_tree("qa_scan"))
        assert not _shown(dialog._pool_collapse_buttons["qa_scan"])
        checkbox.setChecked(True)
        assert config["use_qa_scan_keys"] is True and _shown(dialog._key_context_tree("qa_scan"))
        app.processEvents()
    finally:
        dialog.deleteLater()


def test_preview_pool_dialog_always_shows_its_list(monkeypatch):
    app, mkm = _qt(monkeypatch)
    config = _config()
    config[COLLAPSED] = {"inpainter": True, "fallback": True}
    for pool in ("inpainter", "fallback"):
        dialog, _saved, _log = _manager(mkm, config, preview_pool=pool)
        try:
            checkbox = dialog._pool_enable_checkbox(pool)
            assert checkbox.isChecked() and checkbox.focusPolicy() == mkm.Qt.ClickFocus
            assert dialog._pool_collapse_buttons == {}
            assert _shown(dialog._key_context_tree(pool)) and _shown(dialog._pool_add_form(pool))
            app.processEvents()
        finally:
            dialog.deleteLater()
    assert config[COLLAPSED] == {"inpainter": True, "fallback": True}


def test_schedules_tab_shows_a_folded_pool_add_form(monkeypatch):
    app, mkm = _qt(monkeypatch)
    config = _config()
    dialog, _saved, _log = _manager(mkm, config)
    try:
        dialog._pool_collapse_buttons["glossary"].click()
        dialog.manager_tabs.setCurrentIndex(1)
        assert dialog.select_schedule_pool("glossary")
        form = dialog._pool_add_form("glossary")
        assert getattr(dialog, "_borrowed_add_form", None) and dialog._borrowed_add_form[0] is form
        assert _shown(form) and not dialog.schedule_add_hint.isVisibleTo(dialog.schedule_add_hint.parentWidget())
        # the Schedules enable switch still drives the pool's own checkbox
        dialog.schedule_pool_enable_check.setChecked(False)
        assert not dialog._pool_enable_checkbox("glossary").isChecked() and config["use_glossary_keys"] is False
        dialog.schedule_pool_enable_check.setChecked(True)
        assert dialog._pool_enable_checkbox("glossary").isChecked() and config["use_glossary_keys"] is True
        # back on Key Pools the form returns into the folded section
        dialog.manager_tabs.setCurrentIndex(0)
        dialog._return_pool_add_form()
        assert form.parentWidget() is dialog._pool_section_bodies["glossary"]
        assert not _shown(form) and config[COLLAPSED] == {"glossary": True}
        app.processEvents()
    finally:
        dialog.deleteLater()
