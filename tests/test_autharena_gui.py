import os
import sys
import threading
import time
from collections import deque
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QPlainTextEdit, QPushButton, QTextBrowser, QWidget

import translator_gui
from streaming_log import encode_stream_fragment


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
    assert gui.autharena_login_btn.text() == 'Arena Login'
    assert not gui.autharena_acct_combo.isHidden()
    assert gui.autharena_acct_combo.itemText(0) == '0'
    gui.close()


class MainStreamHarness(QWidget):
    append_log = translator_gui.TranslatorGUI.append_log
    _append_gui_log_batch = translator_gui.TranslatorGUI._append_gui_log_batch
    _direct_log_message_is_suppressed = translator_gui.TranslatorGUI._direct_log_message_is_suppressed

    def __init__(self):
        super().__init__()
        self.log_text = QPlainTextEdit(self)
        self._extra_log_listeners = []
        self._input_output_run_active = False
        self.stop_requested = False

    def _schedule_log_autoscroll(self):
        pass


class DirectStreamHarness(translator_gui._InputOutputDialog):
    def __init__(self):
        QDialog.__init__(self)
        self._log_queue = deque()
        self._stream_phase_by_thread = {}
        self._listener_stream_phase_by_thread = {}
        self._listener_glossary_threads = set()
        self._request_segment_by_thread = {}
        self._active_request_segments = []
        self._active_request_next_number = 1
        self._run_source_is_attachment = False
        self._active = True
        self._streaming_text = False
        self._in_thinking = False
        self._streamed_content = ''
        self._thinking_stream_text = ''
        self._processing_text = ''
        self._thinking_token_count = 0
        self._generation_token_count = 0
        self._token_encoder_initialized = True
        self._token_encoder = None
        self.output = QTextBrowser(self)
        self.render_count = 0

    def _schedule_stream_render(self, immediate=False):
        # Use the normal content converter without constructing the entire
        # composer/history UI. Assert the queue asks for a live render.
        self.output.setHtml(self._markup_to_html(self._streamed_content))
        self.render_count += 1


@pytest.mark.parametrize('batched', [False, True])
def test_arena_main_log_joins_exact_fragments_before_completion(qapp, monkeypatch, batched):
    monkeypatch.setattr(translator_gui, '_persist_gui_log_message', lambda *_: None)
    gui = MainStreamHarness()
    gui.append_log('📡 AuthArena: Text streaming...')
    events = [encode_stream_fragment('content', value) for value in ('Hel', 'lo', ' ', 'world', '\n  Next')]
    if batched:
        gui._append_gui_log_batch(events[:2])
        assert gui.log_text.toPlainText().endswith('\nHello')
        gui._append_gui_log_batch(events[2:])
    else:
        for event in events:
            gui.append_log(event)
    assert gui.log_text.toPlainText() == '📡 AuthArena: Text streaming...\nHello world\n  Next'
    assert '[STREAM_FRAGMENT]' not in gui.log_text.toPlainText()
    gui.append_log('📡 AuthArena: Stream finished with an error')
    gui.append_log('📡 AuthArena: Text streaming...')
    gui.append_log(encode_stream_fragment('content', 'New response'))
    assert gui.log_text.toPlainText().endswith('📡 AuthArena: Text streaming...\nNew response')
    gui.close()


@pytest.mark.parametrize('end_banner', [
    '📡 AuthArena: Stream finished in 1s',
    '📡 AuthArena: Stream finished with an error',
])
def test_arena_direct_text_fragments_preserve_channels_and_close_on_terminal(qapp, end_banner):
    gui = DirectStreamHarness()
    for value in ('Con', 'sider', '\n  this'):
        assert gui._on_log_line(encode_stream_fragment('reasoning', value), 'Thread-A') == 'suppress-main-log'
    gui._drain_log_queue(final=True)
    assert gui._thinking_stream_text == 'Consider\n  this'
    assert gui._streamed_content == ''
    assert gui._active_request_segments[0]['phase'] == 'thinking'
    for value in ('Hel', 'lo', ' ', 'world'):
        gui._on_log_line(encode_stream_fragment('content', value), 'Thread-A')
    gui._drain_log_queue(final=True)
    segment = gui._active_request_segments[0]
    assert gui.output.toPlainText() == 'Hello world'
    assert segment['content'] == 'Hello world'
    assert segment['thinking'] == 'Consider\n  this'
    assert segment['phase'] == 'text' and not segment['complete']
    assert gui.render_count == 2  # Both phases are rendered before completion.
    gui._on_log_line(end_banner, 'Thread-A')
    gui._on_log_line('Saved text file: output.txt', 'Thread-A')
    gui._drain_log_queue(final=True)
    assert segment['phase'] == 'processing' and segment['complete']
    assert gui._listener_stream_phase_by_thread['Thread-A'] == 'processing'
    assert not gui._streaming_text and not gui._in_thinking
    assert segment['content'] == 'Hello world'
    gui.close()


def test_arena_fragment_status_words_and_thread_labels_remain_model_text(qapp, monkeypatch):
    monkeypatch.setattr(translator_gui, '_persist_gui_log_message', lambda *_: None)
    text = '\n    saved text file; operation cancelled [Thread-fake]\n'
    event = encode_stream_fragment('content', text)
    main = MainStreamHarness()
    main.stop_requested = True
    main.append_log(event)
    assert main.log_text.toPlainText() == text
    direct = DirectStreamHarness()
    direct._on_log_line(event, 'Thread-A')
    direct._on_log_line(encode_stream_fragment('content', 'Other'), 'Thread-B')
    direct._on_log_line(encode_stream_fragment('content', 'tail'), 'Thread-A')
    direct._drain_log_queue(final=True)
    assert direct._request_segment_for_thread('Thread-A')['content'] == text + 'tail'
    assert direct._request_segment_for_thread('Thread-B')['content'] == 'Other'
    assert 'Thread-fake' not in direct._request_segment_by_thread
    main.close()
    direct.close()


@pytest.mark.parametrize('direct_active', [False, True])
def test_arena_fragments_reach_direct_listener_and_only_suppress_active_main_copy(qapp, monkeypatch, direct_active):
    monkeypatch.setattr(translator_gui, '_persist_gui_log_message', lambda *_: None)
    main = MainStreamHarness()
    direct = DirectStreamHarness()
    main._extra_log_listeners = [direct._on_log_line]
    main._input_output_run_active = direct_active
    main.append_log(encode_stream_fragment('content', 'Partial'), source_thread='Thread-A')
    direct._drain_log_queue(final=True)
    assert direct.output.toPlainText() == 'Partial'
    assert main.log_text.toPlainText() == ('' if direct_active else 'Partial')
    main.close()
    direct.close()


def test_reader_forced_streaming_overrides_arena_log_hide_flag(monkeypatch):
    for key in translator_gui._InputOutputDialog._FORCED_STREAM_ENV_KEYS:
        monkeypatch.setenv(key, '0')
    translator_gui.TranslatorGUI._apply_forced_streaming_environment(SimpleNamespace())
    assert os.environ['AUTHARENA_LOG_STREAM_CHUNKS'] == '1'
    assert os.environ['ALLOW_AUTHGPT_BATCH_STREAM_LOGS'] == '1'
    assert os.environ['STREAM_THINKING_LOGS'] == '1'


@pytest.mark.parametrize('terminal', [
    '📡 AuthArena: Stream finished with an error',
    'Translation stopped by user',
])
def test_reader_renders_exact_arena_fragments_and_resets_after_stop(qapp, terminal):
    from epub_library import EpubReaderDialog

    class ReaderStreamHarness(QWidget):
        _on_live_log_line = EpubReaderDialog._on_live_log_line
        _classify_live_line = EpubReaderDialog._classify_live_line
        _drain_live_queue = EpubReaderDialog._drain_live_queue
        _render_live_content = EpubReaderDialog._render_live_content
        _wrap_live_html = EpubReaderDialog._wrap_live_html
        _LIVE_STATUS_CHARS = EpubReaderDialog._LIVE_STATUS_CHARS

        def __init__(self):
            super().__init__()
            self._live_log_queue = deque()
            self._live_in_thinking = False
            self._live_streaming_text = False
            self._live_content_buf = ''
            self._live_think_pending = ''
            self._live_log_pending = ''
            self._live_think_toggle = None
            self._live_follow_stream = False
            self._live_content_view = QTextBrowser(self)
            self._live_think_view = QPlainTextEdit(self)
            self._font_family = 'Arial'
            self._font_size = 12
            self._line_spacing = 1.5

        def _get_theme(self):
            return {'bg': '#ffffff', 'fg': '#111111'}

    reader = ReaderStreamHarness()
    for value in ('Con', 'sider', ' this'):
        reader._on_live_log_line(encode_stream_fragment('reasoning', value))
    reader._drain_live_queue()
    assert reader._live_think_view.toPlainText() == 'Consider this'
    for value in ('Hel', 'lo', ' world'):
        reader._on_live_log_line(encode_stream_fragment('content', value))
    reader._drain_live_queue()
    assert reader._live_content_view.toPlainText() == 'Hello world'
    assert reader._live_content_buf == 'Hello world'
    assert reader._live_streaming_text and not reader._live_in_thinking
    reader._on_live_log_line(terminal)
    reader._on_live_log_line('Ordinary pipeline output')
    reader._drain_live_queue()
    assert not reader._live_streaming_text and not reader._live_in_thinking
    assert reader._live_content_buf == 'Hello world'
    assert terminal in reader._live_think_view.toPlainText()
    assert '[STREAM_FRAGMENT]' not in reader._live_think_view.toPlainText()
    reader.close()


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
    assert combo.currentText() == '5'
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
    assert gui.autharena_acct_combo.currentText() == '2'
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
    assert gui.autharena_login_btn.text() == 'Arena Login'
    assert 'Sign-in is in progress' in gui.autharena_login_btn.toolTip()
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
    assert gui.autharena_login_btn.text() == 'Arena Login'
    assert 'Arena profile: 7.' in gui.autharena_login_btn.toolTip()
    assert any('Arena 2:' in line for line in gui.logs)
    assert any('Signed in and verified' in line for line in gui.logs) == (outcome == 'success')
    assert bool(gui.catalog_requests) == (outcome == 'success')
    assert bool(warnings) == (outcome in {'exception', 'unverified', 'wrong-profile'})
    if outcome == 'success':
        gui.model_var = 'autharena2/model'
        gui._autharena_login_status_changed()
        assert gui.autharena_login_btn.text() == 'Arena Login'
        assert 'Last verified signed in' in gui.autharena_login_btn.toolTip()
    gui.close()
