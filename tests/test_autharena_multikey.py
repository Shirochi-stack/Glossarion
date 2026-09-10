import os
import sys
import threading
import time
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
import shiboken6

QtCore = pytest.importorskip('PySide6.QtCore')
QtWidgets = pytest.importorskip('PySide6.QtWidgets')
import multi_api_key_manager as manager


@pytest.fixture(scope='module')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class Translator(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.config = {'fallback_keys': [{'model': 'autharena4/model'}]}
        self.logs = []
        self.notifications = []

    def append_log(self, message):
        self.logs.append((message, threading.get_ident()))

    def on_model_change(self):
        pass

    @QtCore.Slot()
    def _autharena_login_status_changed(self):
        self.notifications.append(threading.get_ident())


@pytest.fixture
def dialog(app, monkeypatch):
    api = SimpleNamespace(
        get_account_ids=lambda: [0, 2],
        get_account_status=lambda slot: {'logged_in': slot == 2},
        # Status redraw must not consume a round-robin rotation turn.
        get_rotating_account_pool=lambda: pytest.fail('UI consumed rotation'),
    )
    monkeypatch.setitem(sys.modules, 'autharena', api)
    widget = manager.MultiAPIKeyDialog.__new__(manager.MultiAPIKeyDialog)
    QtWidgets.QDialog.__init__(widget)
    widget.translator_gui = Translator()
    widget.key_pool = SimpleNamespace(keys=[])
    widget._model_search_combos = []
    yield widget, api
    worker = getattr(widget, '_autharena_login_thread', None)
    if worker:
        worker.join(timeout=2)
    if shiboken6.isValid(widget):
        widget.close()
    app.processEvents()


def field(dialog, model):
    combo = QtWidgets.QComboBox(dialog)
    combo.setEditable(True)
    combo.resize(430, 30)
    dialog._attach_model_autofill(combo, model_values=['autharena/model', 'authza/model'])
    combo.setCurrentText(model)
    return combo


def wait_for(app, condition):
    deadline = time.monotonic() + 3
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert condition()


@pytest.mark.parametrize('model,slot', [
    ('autharena/model', 0), ('AUTHARENA3/model', 3), ('autharena0/model', -1),
    ('autharena12/', 12), ('authgpt/model', None), ('autharenax/model', None),
])
def test_shared_editor_installs_visible_account_login(dialog, model, slot):
    widget, _ = dialog
    combo = field(widget, model)
    button = combo._autharena_login_button
    assert button.isHidden() == (slot is None)
    if slot is not None:
        assert button.property('autharenaAccountId') == slot
        assert button.text() == 'Arena Login'
        assert combo.lineEdit().textMargins().right() >= button.width() + 4


def test_editor_switching_keeps_other_auth_button_and_silent_reset(dialog):
    widget, _ = dialog
    combo = field(widget, 'autharena/model')
    combo.setCurrentText('authza/model')
    assert combo._autharena_login_button.isHidden()
    assert not combo._authza_login_button.isHidden()
    assert combo.lineEdit().textMargins().right() >= 28
    combo.setCurrentText('autharena0/model')
    assert combo._autharena_login_button.text() == 'Arena Login'
    assert 'Last verified signed in' in combo._autharena_login_button.toolTip()
    widget._set_combo_text_silently(combo, '')
    assert combo._autharena_login_button.isHidden()
    assert combo.lineEdit().textMargins().right() == 0


def test_pool_chooser_includes_default_existing_pending_and_configured_slots(dialog, monkeypatch):
    widget, _ = dialog
    field(widget, 'AUTHARENA7/model')
    field(widget, 'autharena0/model')
    assert widget._autharena_login_account_choices() == [0, 2, 4, 7]
    choices = []

    def choose(*args):
        choices.extend(args[3])
        return '4', True

    monkeypatch.setattr(manager.QInputDialog, 'getItem', choose)
    assert widget._choose_autharena_login_account(-1) == 4
    assert choices == ['0', '2', '4', '7', '+ New']
    monkeypatch.setattr(manager.QInputDialog, 'getItem', lambda *_: ('+ New', True))
    monkeypatch.setattr(manager.QInputDialog, 'getInt', lambda *_: (9, True))
    assert widget._choose_autharena_login_account(-1) == 9
    monkeypatch.setattr(manager.QInputDialog, 'getItem', lambda *_: ('', False))
    assert widget._choose_autharena_login_account(-1) is None


def test_manager_publishes_pending_arena_visibility_hints(dialog):
    widget, _ = dialog
    field(widget, 'autharena0/model')
    field(widget, 'AUTHARENA5/model')
    field(widget, 'autharena/model')
    widget._refresh_parent_model_requirements()
    assert widget.translator_gui._multi_key_manager_autharena_pool_hint
    assert widget.translator_gui._multi_key_manager_autharena_account_ids == {0, 5}
    assert widget._model_affects_parent_provider_controls('AUTHARENA0/model')


def test_login_runs_off_thread_and_notifies_on_gui_thread(dialog, app):
    widget, api = dialog
    combo = field(widget, 'AUTHARENA3/model')
    gui_thread = threading.get_ident()
    invoked = []
    release = threading.Event()

    def login(**kwargs):
        assert kwargs['timeout'] == 600
        invoked.append((kwargs['account_id'], threading.get_ident()))
        kwargs['log_fn']('Please sign in')
        assert release.wait(2)
        return {'logged_in': True, 'account_id': kwargs['account_id']}

    api.login = login
    combo._autharena_login_button.click()
    assert widget._autharena_login_busy
    assert not combo._autharena_login_button.isEnabled()
    assert combo._autharena_login_button.text() == 'Arena Login'
    assert 'Sign-in is in progress' in combo._autharena_login_button.toolTip()
    combo.setCurrentText('autharena8/model')  # Account belongs to the click, not current text.
    release.set()
    wait_for(app, lambda: bool(widget.translator_gui.notifications))
    assert invoked[0][0] == 3 and invoked[0][1] != gui_thread
    assert widget._autharena_login_result == {'logged_in': True, 'account_id': 3}
    assert all(thread == gui_thread for _, thread in widget.translator_gui.logs)
    assert widget.translator_gui.notifications == [gui_thread]
    assert not widget._autharena_login_busy
    assert combo._autharena_login_button.text() == 'Arena Login'


@pytest.mark.parametrize('outcome', ['exception', 'unverified'])
def test_login_errors_are_captured_without_worker_dialogs(dialog, app, monkeypatch, outcome):
    widget, api = dialog
    combo = field(widget, 'autharena/model')

    def login(**_kwargs):
        if outcome == 'exception':
            raise RuntimeError('Login window closed')
        return {'profile_saved': True}

    api.login = login
    monkeypatch.setattr(manager.QMessageBox, 'warning', lambda *_: pytest.fail('Unexpected modal warning'))
    combo._autharena_login_button.click()
    wait_for(app, lambda: not widget._autharena_login_busy)
    assert widget._autharena_login_error
    assert not widget.translator_gui.notifications
    assert combo._autharena_login_button.isEnabled()
    assert combo._autharena_login_button.text() == 'Arena Login'
    assert 'failed' in widget.translator_gui.logs[-1][0]


def test_pool_login_uses_selected_physical_slot(dialog, app, monkeypatch):
    widget, api = dialog
    combo = field(widget, 'autharena0/model')
    calls = []
    monkeypatch.setattr(manager.QInputDialog, 'getItem', lambda *_: ('0', True))
    api.login = lambda **kwargs: calls.append(kwargs['account_id']) or {'logged_in': True}
    combo._autharena_login_button.click()
    wait_for(app, lambda: not widget._autharena_login_busy)
    assert calls == [0]


def test_manager_can_close_while_login_is_pending(dialog, app):
    widget, api = dialog
    combo = field(widget, 'autharena2/model')
    release = threading.Event()

    def login(**_kwargs):
        assert release.wait(2)
        return {'logged_in': True}

    api.login = login
    combo._autharena_login_button.click()
    widget.close()
    assert widget._autharena_login_busy
    release.set()
    wait_for(app, lambda: not widget._autharena_login_busy)
    assert not widget.isVisible()


def test_destroyed_manager_does_not_raise_in_login_worker(dialog, app, monkeypatch):
    widget, api = dialog
    combo = field(widget, 'autharena2/model')
    release = threading.Event()
    failures = []
    monkeypatch.setattr(threading, 'excepthook', lambda args: failures.append(args.exc_value))

    def login(**kwargs):
        assert release.wait(2)
        kwargs['log_fn']('Login completed after manager closed')
        return {'logged_in': True}

    api.login = login
    combo._autharena_login_button.click()
    worker = widget._autharena_login_thread
    shiboken6.delete(widget)
    release.set()
    worker.join(timeout=2)
    app.processEvents()
    assert not worker.is_alive()
    assert failures == []
