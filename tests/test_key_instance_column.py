"""Multi API Key Manager: the multi-instance "Instance" column (src/multi_api_key_manager.py).

Every key of every pool (Translation ``multi_api_keys``, Fallback, Glossary and each dedicated
pool) carries an instance number (instance_plan, default 1). Each 🔑 Key Pools list shows it right
after the model and the ⏰ Schedules list leads the key's Parameters with it ("🔢2 · 📤…"); it is
edited like the other per-key value columns (double-click / the key menu / the Parameters menu's
"🔢 Instance…"). Translation keys are saved through the dialog's own save path
(``APIKeyEntry.to_dict``), the other pools' config rows at once; 1 is never written.

Runs offscreen. Run (repository root)::

    python -m pytest -p no:cacheprovider -W ignore -o console_output_style=classic tests/test_key_instance_column.py
"""
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:  # CI has no tests/conftest.py
    sys.path.insert(0, str(SRC))
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

MAIN_HEADERS = ["API Key", "Model", "Instance", "⌛", "Output Limit", "Temperature", "Delay (s)", "Status",
                "✅", "❌", "Requests"]
POOL_HEADERS = ["API Key", "Model", "Instance", "Output Limit", "Temperature", "Delay (s)", "Status",
                "✅", "❌", "Requests"]
SCHEDULE_HEADERS = ["#", "API Key", "Model", "Parameters", "Schedule", "Status"]
DICT_POOLS = ("fallback", "glossary", "metadata", "qa_scan", "tts")


def _qt(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    import multi_api_key_manager as mkm
    if not mkm.HAS_GUI:
        pytest.skip("key manager imported headless")
    return app, mkm


def _config():
    return {
        "use_multi_api_keys": True,
        "multi_api_keys": [
            {"api_key": "sk-inst-one-local-test", "model": "deepseek-chat", "enabled": True},
            {"api_key": "sk-inst-two-local-test", "model": "gemini-3-pro", "enabled": True, "instance": 2,
             "individual_output_token_limit": 4096},
            {"api_key": "sk-inst-three-local-test", "model": "gpt-5.6-sol", "enabled": True, "instance": "bad"},
        ],
        "use_fallback_keys": True,
        "fallback_keys": [
            {"api_key": "fb-one-local-test-key", "model": "gemini-3-pro", "enabled": True},
            {"api_key": "fb-two-local-test-key", "model": "deepseek-chat", "enabled": True, "instance": 2},
        ],
        "use_glossary_keys": True,
        "glossary_keys": [
            {"api_key": "gl-one-local-test-key", "model": "deepseek-chat", "enabled": True, "instance": 3},
            {"api_key": "gl-two-local-test-key", "model": "gemini-3-pro", "enabled": True},
        ],
        "use_metadata_keys": True,
        "metadata_keys": [
            {"api_key": "md-one-local-test-key", "model": "deepseek-chat", "enabled": True},
            {"api_key": "md-two-local-test-key", "model": "gemini-3-pro", "enabled": True, "instance": "2"},
        ],
        "qa_scan_keys": [{"api_key": "qa-one-local-test-key", "model": "gemini-3-pro", "instance": 2}],
        "tts_keys": [{"api_key": "tts-one-local-test-key", "model": "gpt-4o-mini-tts"}],
    }


def _manager(mkm, config):
    saved = []
    gui = SimpleNamespace(config=config, save_config=lambda **_kwargs: saved.append(deepcopy(config)))
    return mkm.MultiAPIKeyDialog(None, gui), saved


def _close(app, dialog):
    dialog.close()
    dialog.deleteLater()
    app.processEvents()


def _headers(tree):
    return [tree.headerItem().text(column) for column in range(tree.columnCount())]


def _column(tree, column):
    return [tree.topLevelItem(row).text(column) for row in range(tree.topLevelItemCount())]


class _FakeGetInt:
    """QInputDialog.getInt stand-in: records (value, minimum, maximum) and answers *answer*."""

    def __init__(self, answer, ok=True):
        self.answer, self.ok, self.calls = answer, ok, []

    def __call__(self, _parent, _title, _label, value=0, minimum=0, maximum=0, _step=1):
        self.calls.append((value, minimum, maximum))
        return self.answer, self.ok


def test_key_pools_list_shows_edits_and_saves_the_instance(monkeypatch):
    app, mkm = _qt(monkeypatch)
    config = _config()
    dialog, saved = _manager(mkm, config)
    tree = dialog.tree
    columns = dialog.MAIN_KEY_TREE_COLUMNS
    instance = columns["instance"]
    try:
        # the column sits right after Model; the other columns keep their order
        assert _headers(tree) == MAIN_HEADERS and instance == columns["model"] + 1
        assert MAIN_HEADERS[dialog._key_status_column("main")] == "Status"
        assert MAIN_HEADERS[columns["cooldown"]] == "⌛" and MAIN_HEADERS[columns["output_limit"]] == "Output Limit"
        assert "instance" in tree.headerItem().toolTip(instance).lower()
        # stored values (missing / invalid read as 1)
        assert [key.instance for key in dialog.key_pool.keys] == [1, 2, 1]
        assert _column(tree, instance) == ["1", "2", "1"]
        assert tree.topLevelItem(1).text(columns["output_limit"]) == "4096"
        assert tree.topLevelItem(0).text(columns["cooldown"]) == "60s"
        assert "Instance 2" in tree.topLevelItem(1).toolTip(instance)

        # the real edit path: double-click the column -> integer editor (>= 1)
        get_int = _FakeGetInt(3)
        monkeypatch.setattr(mkm.QInputDialog, "getInt", get_int)
        tree.itemDoubleClicked.emit(tree.topLevelItem(0), instance)
        app.processEvents()
        assert get_int.calls == [(1, 1, dialog.KEY_INSTANCE_MAX)]
        assert dialog.key_pool.keys[0].instance == 3 and tree.topLevelItem(0).text(instance) == "3"
        assert dialog.key_pool.keys[0].to_dict()["instance"] == 3
        # a cancelled editor changes nothing
        monkeypatch.setattr(mkm.QInputDialog, "getInt", _FakeGetInt(7, ok=False))
        tree.itemDoubleClicked.emit(tree.topLevelItem(1), instance)
        assert dialog.key_pool.keys[1].instance == 2
        # double-clicking another column does not open the instance editor
        get_int = _FakeGetInt(9)
        monkeypatch.setattr(mkm.QInputDialog, "getInt", get_int)
        monkeypatch.setattr(dialog, "_show_output_token_limit_dialog", lambda *_a, **_k: (0, False))
        tree.itemDoubleClicked.emit(tree.topLevelItem(1), columns["output_limit"])
        assert get_int.calls == [] and dialog.key_pool.keys[1].instance == 2

        # persisted through the dialog's save path; instance 1 stays out of the dict
        dialog._save_keys_to_config()
        assert [key.get("instance") for key in config["multi_api_keys"]] == [3, 2, None]
        assert saved and [key.get("instance") for key in saved[-1]["multi_api_keys"]] == [3, 2, None]
        assert "instance" not in config["fallback_keys"][0]

        # the key menu sets every selected key's instance
        from PySide6.QtCore import QPoint
        chosen = {}

        def fake_exec(menu, _position):
            chosen["texts"] = [action.text() for action in menu.actions()]
            next(action for action in menu.actions() if action.text().startswith("🔢 Set Instance")).trigger()

        dialog._exec_key_context_menu = fake_exec
        tree.clearSelection()
        tree.topLevelItem(0).setSelected(True)
        tree.topLevelItem(2).setSelected(True)
        tree.itemAt = lambda _pos: tree.topLevelItem(0)
        get_int = _FakeGetInt(4)
        monkeypatch.setattr(mkm.QInputDialog, "getInt", get_int)
        dialog._show_context_menu(QPoint(1, 1))
        app.processEvents()
        assert "🔢 Set Instance (2 selected)" in chosen["texts"]
        assert [call[1:] for call in get_int.calls] == [(1, dialog.KEY_INSTANCE_MAX)]
        assert [key.instance for key in dialog.key_pool.keys] == [4, 2, 4]
        assert _column(tree, instance) == ["4", "2", "4"]
        dialog._save_keys_to_config()
        assert [key.get("instance") for key in config["multi_api_keys"]] == [4, 2, 4]

        # a reload (APIKeyPool.load_from_list, the import / reopen path) keeps the numbers
        pool = mkm.APIKeyPool()
        pool.load_from_list(config["multi_api_keys"])
        assert [key.instance for key in pool.keys] == [4, 2, 4]
        assert [key.to_dict() for key in pool.keys] == config["multi_api_keys"]
    finally:
        _close(app, dialog)


def _instances(config, pool):
    import key_pool_service

    return [key.get("instance") for key in config[key_pool_service.POOL_SPECS[pool]["config_key"]]]


def _pool_tree(dialog, pool):
    dialog._ensure_key_pool_section(pool)
    return dialog._key_context_tree(pool)


def test_every_pool_list_shows_and_edits_the_instance(monkeypatch):
    """Fallback, Glossary and the dedicated pools: the same Instance column as the Translation list
    (after Model, header tooltip, cell tooltip), edited by double-click and by the key menu, stored
    on the config rows at once (1 is left out, like APIKeyEntry.to_dict); the columns after it
    (Status / ✅ / ❌ / Requests) moved along."""
    app, mkm = _qt(monkeypatch)
    config = _config()
    dialog, saved = _manager(mkm, config)
    columns = dialog.POOL_KEY_TREE_COLUMNS
    instance = columns["instance"]
    expected = {"fallback": ["1", "2"], "glossary": ["3", "1"], "metadata": ["1", "2"], "qa_scan": ["2"], "tts": ["1"]}
    try:
        assert dialog._pool_key_tree_columns("fallback") is columns and instance == columns["model"] + 1
        assert POOL_HEADERS[dialog._key_status_column("fallback")] == "Status"
        for pool in DICT_POOLS:
            tree = _pool_tree(dialog, pool)
            assert _headers(tree) == POOL_HEADERS, pool
            assert "instance" in tree.headerItem().toolTip(instance).lower(), pool
            assert _column(tree, instance) == expected[pool], pool
            assert _column(tree, columns["model"]) == [k["model"] for k in dialog._pool_key_list(pool)], pool
            assert tree.topLevelItem(0).text(columns["output_limit"]) == "global", pool
            assert tree.topLevelItem(0).text(dialog._key_status_column(pool)) in ("Enabled", "Disabled"), pool
            assert f"Instance {expected[pool][0]}" in tree.topLevelItem(0).toolTip(instance), pool

        # double-click on the column: the shared integer editor (>= 1); stored on the row at once
        for pool in DICT_POOLS:
            tree = _pool_tree(dialog, pool)
            get_int = _FakeGetInt(4)
            monkeypatch.setattr(mkm.QInputDialog, "getInt", get_int)
            tree.clearSelection()
            tree.topLevelItem(0).setSelected(True)
            saved.clear()
            tree.itemDoubleClicked.emit(tree.topLevelItem(0), instance)
            app.processEvents()
            assert get_int.calls == [(int(expected[pool][0]), 1, dialog.KEY_INSTANCE_MAX)], pool
            assert _instances(config, pool)[0] == 4 and tree.topLevelItem(0).text(instance) == "4", pool
            assert saved and _instances(saved[-1], pool)[0] == 4, pool
            # back to 1: the field is left out
            monkeypatch.setattr(mkm.QInputDialog, "getInt", _FakeGetInt(1))
            tree.itemDoubleClicked.emit(tree.topLevelItem(0), instance)
            app.processEvents()
            assert _instances(config, pool)[0] is None and tree.topLevelItem(0).text(instance) == "1", pool
            # cancelled: nothing changes
            monkeypatch.setattr(mkm.QInputDialog, "getInt", _FakeGetInt(9, ok=False))
            tree.itemDoubleClicked.emit(tree.topLevelItem(0), instance)
            assert _instances(config, pool)[0] is None, pool

        # the key menu: "🔢 Set Instance" for every selected key of the pool
        from PySide6.QtCore import QPoint
        chosen = {}

        def fake_exec(menu, _position):
            chosen["texts"] = [action.text() for action in menu.actions()]
            next(action for action in menu.actions() if action.text().startswith("🔢 Set Instance")).trigger()

        dialog._exec_key_context_menu = fake_exec
        for pool in ("fallback", "glossary", "metadata"):
            tree = _pool_tree(dialog, pool)
            tree.clearSelection()
            tree.topLevelItem(0).setSelected(True)
            tree.topLevelItem(1).setSelected(True)
            tree.itemAt = lambda _pos, t=tree: t.topLevelItem(0)
            monkeypatch.setattr(mkm.QInputDialog, "getInt", _FakeGetInt(5))
            dialog._pool_key_tree_actions(pool)["menu"](QPoint(1, 1))
            app.processEvents()
            assert "🔢 Set Instance (2 selected)" in chosen["texts"], pool
            assert _instances(config, pool) == [5, 5], pool
            assert _column(tree, instance) == ["5", "5"], pool
    finally:
        _close(app, dialog)


def test_pool_list_status_columns_follow_the_instance_column(monkeypatch):
    """The test result / timeout updates write the Status, ✅, ❌ and Requests cells, not Instance."""
    app, mkm = _qt(monkeypatch)
    config = _config()
    dialog, _saved = _manager(mkm, config)
    status = dialog.POOL_KEY_TREE_STATUS_COLUMN
    try:
        for pool, update, timeout in (
                ("fallback", dialog._update_fallback_test_result, dialog._update_fallback_timeout_status),
                ("glossary", dialog._update_glossary_test_result, dialog._update_glossary_timeout_status)):
            tree = _pool_tree(dialog, pool)
            update(1, True)
            item = tree.topLevelItem(1)
            assert item.text(status).startswith("✅") and item.text(status + 1) == "1", pool
            assert item.text(status + 3) == "1" and item.text(dialog.POOL_KEY_TREE_COLUMNS["instance"]) in ("1", "2")
            timeout(0)
            assert tree.topLevelItem(0).text(status).startswith("⏱️"), pool
        tree = _pool_tree(dialog, "metadata")
        dialog._dedicated_set_status_column("metadata", 0, "⏳ Testing...", mkm.Qt.blue)
        assert tree.topLevelItem(0).text(status) == "⏳ Testing..."
        assert tree.topLevelItem(0).text(dialog.POOL_KEY_TREE_COLUMNS["instance"]) == "1"
    finally:
        _close(app, dialog)


def test_schedules_parameters_lead_with_the_instance_in_every_pool(monkeypatch):
    """⏰ Schedules: no Instance column; every pool's Parameters start with "🔢N" and the Parameters
    double-click menu's "🔢 Instance…" runs the Key Pools list's own instance editor."""
    app, mkm = _qt(monkeypatch)
    config = _config()
    dialog, saved = _manager(mkm, config)

    def choose(label):
        def fake_exec(menu, _position):
            chosen.append([action.text() for action in menu.actions()])
            next(action for action in menu.actions() if action.text() == label).trigger()
        return fake_exec

    chosen = []
    try:
        dialog.manager_tabs.setCurrentIndex(1)
        app.processEvents()
        tree = dialog.schedule_key_tree
        column = dialog.SCHEDULE_PARAMETERS_COLUMN
        assert _headers(tree) == SCHEDULE_HEADERS == list(dialog.SCHEDULE_TREE_COLUMNS)
        assert "🔢" in tree.headerItem().toolTip(column)
        expected = {"main": ["1", "2", "1"], "fallback": ["1", "2"], "glossary": ["3", "1"], "metadata": ["1", "2"],
                    "qa_scan": ["2"], "tts": ["1"]}
        for pool, numbers in expected.items():
            assert dialog.select_schedule_pool(pool), pool
            app.processEvents()
            assert [text.split(" · ")[0] for text in _column(tree, column)] == [f"🔢{n}" for n in numbers], pool
            assert dialog._pool_has_instances(pool), pool
            assert tree.topLevelItem(0).toolTip(column).startswith(f"🔢 Instance {numbers[0]}\n"), pool
            assert dialog.KEY_INSTANCE_HELP in tree.topLevelItem(0).toolTip(column), pool
        # "🔢 Instance…" from the Parameters menu: the pool's own editor, shown in both tabs and saved
        dialog._exec_key_context_menu = choose("🔢 Instance…")
        for pool in ("main", "fallback", "metadata", "tts"):
            assert dialog.select_schedule_pool(pool)
            app.processEvents()
            monkeypatch.setattr(mkm.QInputDialog, "getInt", _FakeGetInt(6))
            tree.clearSelection()
            tree.itemDoubleClicked.emit(tree.topLevelItem(0), column)
            app.processEvents()
            assert chosen[-1][0] == "🔢 Instance…", pool
            assert tree.topLevelItem(0).text(column).startswith("🔢6 · "), pool
            pool_tree = dialog._key_context_tree(pool)
            assert pool_tree.topLevelItem(0).text(dialog._pool_key_tree_columns(pool)["instance"]) == "6", pool
            if pool == "main":
                assert dialog.key_pool.keys[0].instance == 6
            else:
                assert _instances(config, pool)[0] == 6 and _instances(saved[-1], pool)[0] == 6, pool
        # widths: Parameters fits its texts, Status as wide as its texts (the last column)
        assert tree.columnWidth(column) <= tree.fontMetrics().horizontalAdvance(
            dialog.SCHEDULE_PARAMETERS_MAX_WIDTH) + 24
        status_texts = _column(tree, dialog.SCHEDULE_STATUS_COLUMN)
        assert tree.columnWidth(dialog.SCHEDULE_STATUS_COLUMN) >= max(
            tree.fontMetrics().horizontalAdvance(text) for text in status_texts)
    finally:
        _close(app, dialog)


def test_export_and_import_keep_every_pools_instance(monkeypatch):
    app, mkm = _qt(monkeypatch)
    import key_pool_service

    config = _config()
    dialog, _saved = _manager(mkm, config)
    try:
        pools = dialog._collect_pools_for_export()
        assert [k.get("instance") for k in pools["main"]["keys"]] == [None, 2, None]
        assert [k.get("instance") for k in pools["fallback"]["keys"]] == [None, 2]
        assert [k.get("instance") for k in pools["metadata"]["keys"]] == [None, "2"]
        payload = key_pool_service.build_export_payload(pools)
        target = {}
        result = key_pool_service.import_pools(payload, target, apply=True)
        assert result["error"] is None
        assert [k.get("instance") for k in target["fallback_keys"]] == [None, 2]
        assert [k.get("instance") for k in target["qa_scan_keys"]] == [2]
        import instance_plan

        assert [instance_plan.key_instance(k) for k in target["metadata_keys"]] == [1, 2]
    finally:
        _close(app, dialog)


def test_the_manager_never_binds_a_narrowed_pool(monkeypatch):
    """While one instance of a multi-instance run narrows this process's keys, the dialog keeps
    its own pool: loading every key into the shared (narrowed) pool would hand the other
    instances' keys to the running instance."""
    app, mkm = _qt(monkeypatch)
    import instance_plan
    import unified_api_client

    narrowed = mkm.APIKeyPool("narrowed")
    narrowed.load_from_list([{"api_key": "sk-inst-one-local-test", "model": "deepseek-chat"}])
    monkeypatch.setattr(unified_api_client.UnifiedClient, "_api_key_pool", narrowed)
    monkeypatch.setenv(instance_plan.ENV_KEY_INSTANCE, "1")
    dialog, _saved = _manager(mkm, _config())
    try:
        assert dialog.key_pool is not narrowed and len(dialog.key_pool.keys) == 3
        assert [key.api_key for key in narrowed.keys] == ["sk-inst-one-local-test"]
    finally:
        _close(app, dialog)
    monkeypatch.delenv(instance_plan.ENV_KEY_INSTANCE)
    dialog, _saved = _manager(mkm, _config())
    try:
        assert dialog.key_pool is narrowed  # outside a run: the live pool, as before
    finally:
        _close(app, dialog)
