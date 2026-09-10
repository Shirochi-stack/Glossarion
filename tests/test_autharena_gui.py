import os
import sys
import threading
import time
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QComboBox, QPushButton, QWidget

import translator_gui


class ArenaHarness(QWidget):
    _iter_enabled_key_pool_models = translator_gui.TranslatorGUI._iter_enabled_key_pool_models
    _has_autharena_in_key_pools = translator_gui.TranslatorGUI._has_autharena_in_key_pools
    _autharena_pool_route_requested = translator_gui.TranslatorGUI._autharena_pool_route_requested
    _authgrok_pool_route_requested = translator_gui.TranslatorGUI._authgrok_pool_route_requested
    _refresh_autharena_login_visibility = translator_gui.TranslatorGUI._refresh_autharena_login_visibility
    _collect_auth_account_ids_from_pools = translator_gui.TranslatorGUI._collect_auth_account_ids_from_pools
    _refresh_auth_account_arrows = translator_gui.TranslatorGUI._refresh_auth_account_arrows
    _on_auth_acct_combo_changed = translator_gui.TranslatorGUI._on_auth_acct_combo_changed
    _get_autharena_account_id = translator_gui.TranslatorGUI._get_autharena_account_id
    _update_autharena_login_status = translator_gui.TranslatorGUI._update_autharena_login_status
    _add_autharena_account_slot = translator_gui.TranslatorGUI._add_autharena_account_slot
    _autharena_login_clicked = translator_gui.TranslatorGUI._autharena_login_clicked
    _autharena_login_finished = translator_gui.TranslatorGUI._autharena_login_finished
    _autharena_login_status_changed = translator_gui.TranslatorGUI._autharena_login_status_changed

    def __init__(self, model='autharena/model', config=None):
        super().__init__()
        self.config = config or {}
        self.model_var = model
        self.autharena_login_btn = QPushButton(self)
        self.autharena_acct_combo = QComboBox(self)
        self.autharena_acct_combo.currentIndexChanged.connect(
            lambda index: self._on_auth_acct_combo_changed('autharena', index)
        )
        self.logs = []
        self.catalog_requests = []
        self._autharena_login_status_changed()

    def append_log(self, message):
        assert QThread.currentThread() == QApplication.instance().thread()
        self.logs.append(message)

    def _schedule_current_provider_catalog_refresh(self, delay):
        self.catalog_requests.append(delay)


@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def arena(monkeypatch):
    signed_in = set()
    known = [0]
    adapter = SimpleNamespace(
        get_account_ids=lambda: list(known),
        get_account_status=lambda account: {'account_id': account, 'logged_in': account in signed_in},
        get_rotating_account_pool=lambda: pytest.fail('GUI must not advance the rotation cursor'),
    )
    monkeypatch.setitem(sys.modules, 'autharena', adapter)
    return adapter, signed_in, known


@pytest.mark.parametrize('model', ['autharena/model', 'autharena3/model', ' autharena0/model ', 'AUTHARENA/model'])
def test_arena_login_visible_for_main_default_numbered_and_pool(qapp, arena, model):
    gui = ArenaHarness(model)
    assert not gui.autharena_login_btn.isHidden()
    assert not gui.autharena_acct_combo.isHidden()
    assert gui.autharena_acct_combo.itemText(0) == 'Default'
    gui.close()


@pytest.mark.parametrize('pool,toggle', [
    ('multi_api_keys', 'use_multi_api_keys'),
    ('metadata_keys', 'use_metadata_keys'),
    ('qa_scan_keys', 'use_qa_scan_keys'),
    ('inpainter_keys', 'use_inpainter_keys'),
])
def test_arena_login_visibility_tracks_enabled_pool_and_rows(qapp, arena, pool, toggle):
    config = {toggle: True, pool: [{'model': 'autharena7/model', 'enabled': True}]}
    gui = ArenaHarness('gpt-other', config)
    assert not gui.autharena_login_btn.isHidden()
    assert 7 in gui._auth_account_ids['autharena']
    config[pool][0]['enabled'] = False
    gui._autharena_login_status_changed()
    assert gui.autharena_login_btn.isHidden()
    assert gui.autharena_acct_combo.isHidden()
    config[pool][0]['enabled'] = True
    config[toggle] = False
    gui._autharena_login_status_changed()
    assert gui.autharena_login_btn.isHidden()
    gui.close()


def test_arena_live_manager_hints_supply_visibility_and_physical_slots(qapp, arena):
    gui = ArenaHarness('gpt-other')
    assert gui.autharena_login_btn.isHidden()
    gui._multi_key_manager_autharena_pool_hint = True
    gui._multi_key_manager_autharena_account_ids = {4, 8}
    gui._autharena_login_status_changed()
    assert not gui.autharena_login_btn.isHidden()
    assert gui._auth_account_ids['autharena'] == [0, 4, 8]
    gui._multi_key_manager_autharena_pool_hint = False
    gui._autharena_login_status_changed()
    assert gui.autharena_login_btn.isHidden()
    assert gui.autharena_acct_combo.isHidden()
    gui.close()


def test_arena_account_selection_preserves_physical_slot_and_follows_new_model(qapp, arena):
    _adapter, _signed_in, known = arena
    known[:] = [0, 2, 5]
    gui = ArenaHarness('autharena3/model')
    assert gui._get_autharena_account_id() == 3
    gui.model_var = 'autharena/model'
    gui._autharena_login_status_changed()
    assert gui._get_autharena_account_id() == 0
    gui.model_var = 'autharena0/model'
    gui._autharena_login_status_changed()
    combo = gui.autharena_acct_combo
    combo.setCurrentIndex(combo.findData(5))
    known.insert(1, 1)
    gui._autharena_login_status_changed()
    assert gui._get_autharena_account_id() == 5
    assert combo.currentText() == '#5'
    assert 'rotates' in gui.autharena_login_btn.toolTip()
    gui.model_var = 'autharena2/model'
    gui._autharena_login_status_changed()
    assert gui._get_autharena_account_id() == 2
    gui.close()


def test_arena_pool_add_new_selects_unused_physical_slot(qapp, arena, monkeypatch):
    arena[2][:] = [0, 1, 3]
    queued = []
    monkeypatch.setattr(translator_gui, 'QTimer', SimpleNamespace(singleShot=lambda delay, callback: queued.append(callback)))
    gui = ArenaHarness('autharena0/model')
    gui.autharena_acct_combo.setCurrentIndex(
        gui.autharena_acct_combo.findData(translator_gui._AUTHARENA_ADD_ACCOUNT_SENTINEL)
    )
    assert gui._get_autharena_account_id() == 2
    assert gui.autharena_acct_combo.currentText() == '#2'
    assert gui._autharena_pending_account_ids == {2}
    assert len(queued) == 1
    gui.close()


@pytest.mark.parametrize('outcome', ['success', 'unverified', 'cancelled', 'exception', 'wrong-profile'])
def test_arena_login_worker_uses_captured_profile_and_only_gui_thread_updates(
    qapp, arena, monkeypatch, outcome
):
    adapter, signed_in, known = arena
    known[:] = [0, 2, 7]
    started = threading.Event()
    release = threading.Event()
    calls = []
    warnings = []
    monkeypatch.setattr(translator_gui.QMessageBox, 'warning', lambda *args: warnings.append(args))

    def login(*, account_id, timeout, log_fn):
        calls.append((account_id, timeout, threading.current_thread().name))
        log_fn('External browser status')
        started.set()
        assert release.wait(3)
        if outcome == 'exception':
            raise RuntimeError('Browser unavailable')
        if outcome == 'cancelled':
            return {'logged_in': True, 'cancelled': True}
        if outcome == 'unverified':
            return {'logged_in': False}
        if outcome == 'wrong-profile':
            return {'logged_in': True, 'account_id': 7}
        signed_in.add(account_id)
        return {'logged_in': True, 'account_id': account_id}

    adapter.login = login
    gui = ArenaHarness('autharena2/model')
    gui._autharena_login_clicked()
    assert started.wait(3)
    assert not gui.autharena_login_btn.isEnabled()
    assert not gui.autharena_acct_combo.isEnabled()
    gui._autharena_login_clicked()
    gui.model_var = 'autharena7/model'
    gui._autharena_login_status_changed()
    assert gui._get_autharena_account_id() == 7
    assert 'External browser status' not in gui.logs
    release.set()
    deadline = time.monotonic() + 4
    while gui._autharena_login_in_progress and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert not gui._autharena_login_in_progress
    assert calls == [(2, 180, 'autharena-login-2')]
    assert 'External browser status' in gui.logs
    assert gui.autharena_login_btn.isEnabled()
    assert gui.autharena_acct_combo.isEnabled()
    assert gui.autharena_login_btn.text() == '🔐 Arena #7 Login'
    assert any('Arena #2:' in line for line in gui.logs)
    assert any('Signed in and verified' in line for line in gui.logs) == (outcome == 'success')
    assert bool(gui.catalog_requests) == (outcome == 'success')
    assert bool(warnings) == (outcome in {'exception', 'unverified', 'wrong-profile'})
    gui.close()
