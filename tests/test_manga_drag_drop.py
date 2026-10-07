import os
import sys
import threading
from queue import Queue
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QMimeData, QUrl

import ImageRenderer
import manga_integration
import manga_translator
from manga_integration import (
    MangaTranslationTab,
    _manga_filename_without_skip_prefix,
    _translation_run_token_matches,
)
import manga_ocr_io
from local_inpainter import LocalInpainter
from ocr_manager import CustomAPIProvider, OCRManager, OCRProvider, OCRResult
from unified_api_client import UnifiedClient


class _FakeListWidget:
    def __init__(self):
        self.current_row = -1

    def currentRow(self):
        return self.current_row

    def setCurrentRow(self, row):
        self.current_row = row


class _DropHarness:
    def __init__(self):
        self.selected_files = []
        self.manga_selected_folder_roots = []
        self.file_listbox = _FakeListWidget()
        self.list_items = []
        self.logs = []

    def _can_add_manga_paths_from_single_source(self, _paths):
        return True

    def _add_manga_file_item(self, path):
        self.list_items.append(path)

    def _add_cbz_archive_images(self, _path, _image_extensions):
        return 0

    def _apply_manga_file_sort(self):
        MangaTranslationTab._apply_manga_file_sort(self)

    def _rebuild_manga_file_listbox(self):
        self.list_items = list(self.selected_files)
        if self.list_items and self.file_listbox.currentRow() < 0:
            self.file_listbox.setCurrentRow(0)

    def _update_manga_image_range_display(self):
        pass

    def _persist_selected_files(self):
        pass

    def _log(self, message, level):
        self.logs.append((message, level))


def test_manga_logger_keeps_backend_messages_after_stop(monkeypatch):
    import logging

    monkeypatch.setattr(MangaTranslationTab, '_persistent_log', [])
    updates = Queue()
    harness = SimpleNamespace(
        _should_suppress_debug_log=lambda *_args: False,
        _is_stop_requested=lambda: True,
        is_globally_cancelled=lambda: True,
        log_text=object(),
        update_queue=updates,
    )
    harness._log = lambda message, level='info': MangaTranslationTab._log(harness, message, level)
    handler = manga_integration._MangaGuiLogHandler(harness)
    record = logging.LogRecord(
        'unified_api_client', logging.INFO, __file__, 1,
        'Backend request completed after stop', (), None,
    )

    worker = threading.Thread(target=handler.emit, args=(record,))
    worker.start()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert updates.get_nowait() == ('log', 'Backend request completed after stop', 'info')
    assert MangaTranslationTab._persistent_log[-1] == ('Backend request completed after stop', 'info')


@pytest.mark.parametrize(
    'log_method',
    [manga_translator.MangaTranslator._log, LocalInpainter._log, OCRProvider._log],
)
def test_manga_backend_callbacks_keep_messages_after_stop(log_method):
    received = []
    harness = SimpleNamespace(
        _check_stop=lambda: True,
        is_globally_cancelled=lambda: True,
        concise_logs=False,
        log_callback=lambda message, level: received.append((message, level)),
    )

    log_method(harness, 'Backend diagnostic after stop', 'info')

    assert received == [('Backend diagnostic after stop', 'info')]


def test_skip_display_prefix_uses_emoji_and_accepts_legacy_marker():
    assert _manga_filename_without_skip_prefix('⏭️ page.png') == 'page.png'
    assert _manga_filename_without_skip_prefix('[SKIP] page.png') == 'page.png'
    assert _manga_filename_without_skip_prefix('page.png') == 'page.png'


def test_drop_payload_keeps_supported_local_paths_only(tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"image")
    ignored = tmp_path / "notes.txt"
    ignored.write_text("notes", encoding="utf-8")
    folder = tmp_path / "chapter"
    folder.mkdir()

    mime = QMimeData()
    mime.setUrls([
        QUrl.fromLocalFile(str(image)),
        QUrl.fromLocalFile(str(image)),
        QUrl.fromLocalFile(str(ignored)),
        QUrl.fromLocalFile(str(folder)),
        QUrl("https://example.com/page.png"),
    ])

    paths = MangaTranslationTab._manga_drop_local_paths(object(), mime)
    assert paths == [os.path.abspath(image), os.path.abspath(folder)]


def test_file_context_reorder_actions_move_the_clicked_entry(tmp_path):
    files = [
        os.path.abspath(tmp_path / 'one.png'),
        os.path.abspath(tmp_path / 'two.png'),
        os.path.abspath(tmp_path / 'three.png'),
        os.path.abspath(tmp_path / 'four.png'),
    ]
    rebuilds = []
    preview_updates = []
    persists = []
    manga_tab = SimpleNamespace(
        selected_files=list(files),
        image_preview_widget=object(),
        _skip_key_for_path=lambda path: os.path.normcase(os.path.abspath(path)),
        _rebuild_manga_file_listbox=lambda current_path=None: rebuilds.append(current_path),
        _update_manga_preview_image_list_for_range=lambda: preview_updates.append(True),
        _persist_selected_files=lambda: persists.append(True),
        _log=lambda *_args: None,
    )

    assert MangaTranslationTab._move_manga_file_entry(manga_tab, files[1], 'up')
    assert manga_tab.selected_files == [files[1], files[0], files[2], files[3]]
    assert MangaTranslationTab._move_manga_file_entry(manga_tab, files[1], 'bottom')
    assert manga_tab.selected_files == [files[0], files[2], files[3], files[1]]
    assert MangaTranslationTab._move_manga_file_entry(manga_tab, files[3], 'top')
    assert manga_tab.selected_files == [files[3], files[0], files[2], files[1]]
    assert MangaTranslationTab._move_manga_file_entry(manga_tab, files[3], 'down')
    assert manga_tab.selected_files == [files[0], files[3], files[2], files[1]]
    assert rebuilds == [files[1], files[1], files[3], files[3]]
    assert len(preview_updates) == 4
    assert len(persists) == 4


def test_ocr_drop_payload_keeps_unique_local_json_files_only(tmp_path):
    session = tmp_path / 'chapter_ocr_20260731_193045.json'
    session.write_text('{}', encoding='utf-8')
    ignored = tmp_path / 'notes.txt'
    ignored.write_text('notes', encoding='utf-8')
    mime = QMimeData()
    mime.setUrls([
        QUrl.fromLocalFile(str(session)),
        QUrl.fromLocalFile(str(session)),
        QUrl.fromLocalFile(str(ignored)),
        QUrl('https://example.com/session.json'),
    ])

    paths = MangaTranslationTab._ocr_drop_local_json_paths(object(), mime)

    assert paths == [os.path.abspath(session)]


def test_ocr_json_drop_routes_into_batch_import(tmp_path):
    session = os.path.abspath(tmp_path / 'chapter_ocr.json')

    class _Target:
        def isEnabled(self):
            return True

    class _DropEvent:
        def __init__(self):
            self.accepted = False

        def type(self):
            return QEvent.Drop

        def mimeData(self):
            return object()

        def acceptProposedAction(self):
            self.accepted = True

    target = _Target()
    dropped = []
    highlights = []
    manga_tab = SimpleNamespace(
        _ocr_import_drop_targets={target},
        _ocr_drop_local_json_paths=lambda _mime: [session],
        _set_ocr_import_drop_highlight=lambda active: highlights.append(active),
        _import_batch_ocr_path=lambda path: dropped.append(path),
    )
    event = _DropEvent()

    handled = MangaTranslationTab.eventFilter(manga_tab, target, event)

    assert handled is True
    assert event.accepted is True
    assert highlights == [False]
    assert dropped == [session]


def test_ocr_import_parsing_runs_off_the_gui_thread(tmp_path, monkeypatch):
    image = os.path.abspath(tmp_path / 'page.png')
    document = manga_ocr_io.create_document([], workflow='automatic')
    started = threading.Event()
    release = threading.Event()
    busy_states = []

    def _load(_path):
        started.set()
        release.wait(timeout=5)
        return document

    monkeypatch.setattr(manga_ocr_io, 'load_document', _load)
    manga_tab = SimpleNamespace(
        _ocr_import_generation=0,
        _set_ocr_import_busy=lambda busy: busy_states.append(busy),
        _finish_ocr_import_worker=lambda *_args: None,
        update_queue=Queue(),
    )

    MangaTranslationTab._start_ocr_import_worker(
        manga_tab,
        str(tmp_path / 'session.json'),
        [image],
    )

    assert started.wait(timeout=2)
    assert manga_tab._ocr_import_thread.is_alive()
    assert manga_tab.update_queue.empty()
    release.set()
    manga_tab._ocr_import_thread.join(timeout=5)
    assert busy_states == [True]
    assert manga_tab.update_queue.get_nowait()[0] == 'call_method'


def test_manual_ocr_export_serialization_runs_off_the_gui_thread(tmp_path, monkeypatch):
    image = os.path.abspath(tmp_path / 'page.png')
    destination = os.path.abspath(tmp_path / 'session.json')
    state = {
        'viewer_rectangles': [{'x': 1, 'y': 2, 'width': 30, 'height': 40}],
        'recognized_texts': [{'region_index': 0, 'text': 'source', 'bbox': [1, 2, 30, 40]}],
        'translated_texts': [{
            'original': {'region_index': 0, 'text': 'source'},
            'translation': 'translated',
            'bbox': [1, 2, 30, 40],
        }],
    }
    started = threading.Event()
    release = threading.Event()
    busy_states = []

    def _write(_path, _document):
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr(manga_ocr_io, 'write_document', _write)
    monkeypatch.setattr(
        manga_integration.QFileDialog,
        'getSaveFileName',
        lambda *_args: (destination, ''),
    )
    manga_tab = SimpleNamespace(
        image_preview_widget=SimpleNamespace(current_image_path=None),
        _manual_ocr_files=lambda: [image],
        _current_manga_source_dir=lambda: str(tmp_path),
        _manga_ocr_timestamped_export_filename=lambda: 'session.json',
        _manga_ocr_save_dialog_path=lambda filename: str(tmp_path / filename),
        _manual_editor_state_for_export=lambda _path: state,
        _set_manual_ocr_export_busy=lambda busy: busy_states.append(busy),
        _finish_manual_ocr_export=lambda *_args: None,
        _ocr_export_generation=0,
        update_queue=Queue(),
        dialog=object(),
    )

    MangaTranslationTab._export_manual_ocr_text(manga_tab)

    assert started.wait(timeout=2)
    assert manga_tab._ocr_export_thread.is_alive()
    assert manga_tab.update_queue.empty()
    release.set()
    manga_tab._ocr_export_thread.join(timeout=5)
    assert busy_states == [True]
    assert manga_tab.update_queue.get_nowait()[0] == 'call_method'


def test_batch_ocr_import_restores_translations_into_editor_state(tmp_path, monkeypatch):
    image = os.path.abspath(tmp_path / 'page.png')
    with open(image, 'wb') as handle:
        handle.write(b'image')
    page = manga_ocr_io.make_page(
        image,
        [{
            'text': 'source',
            'translated_text': 'translated',
            'bounding_box': [1, 2, 30, 40],
        }],
    )
    document = manga_ocr_io.create_document([page], workflow='automatic')

    class _StateManager:
        def __init__(self):
            self.states = {}
            self.flushed = False

        def get_state(self, path):
            return self.states.get(path, {})

        def set_state(self, path, state, save=True):
            self.states[path] = state

        def flush_async(self):
            self.flushed = True

    class _Button:
        def setText(self, text):
            self.text = text

        def setToolTip(self, text):
            self.tooltip = text

    state_manager = _StateManager()
    scheduled = []
    manga_tab = SimpleNamespace(
        _imported_ocr_document=None,
        _refresh_imported_ocr_page_map=lambda _files: {image: page},
        image_state_manager=state_manager,
        image_preview_widget=SimpleNamespace(current_image_path=image),
        batch_ocr_import_btn=_Button(),
        _start_imported_ocr_preview_refresh=lambda matches, **kwargs: scheduled.append((matches, kwargs)),
        _log=lambda *_args: None,
        dialog=object(),
    )
    monkeypatch.setattr(ImageRenderer, '_clear_cross_image_state', lambda *_args: None)
    monkeypatch.setattr(ImageRenderer, '_rehydrate_text_state_from_persisted', lambda *_args: None)
    monkeypatch.setattr(ImageRenderer, '_restore_image_state_overlays_only', lambda *_args: None)
    monkeypatch.setattr(manga_integration.QMessageBox, 'information', lambda *_args: None)

    imported = MangaTranslationTab._apply_imported_batch_ocr_document(
        manga_tab,
        str(tmp_path / 'session.json'),
        document,
        [image],
    )

    assert imported is True
    assert state_manager.flushed is True
    assert state_manager.states[image]['translated_texts'][0]['translation'] == 'translated'
    assert scheduled == [({image: page}, {'priority_path': image})]


def test_import_refresh_renders_every_nonvisible_translated_page(tmp_path, monkeypatch):
    current = os.path.abspath(tmp_path / '001.png')
    background = os.path.abspath(tmp_path / '002.png')
    untranslated = os.path.abspath(tmp_path / '003.png')
    output = os.path.abspath(tmp_path / '002_translated' / '002.png')
    translated_page = {'regions': [{'translated_text': 'translated'}]}
    matches = {
        current: translated_page,
        background: translated_page,
        untranslated: {'regions': [{'translated_text': None}]},
    }
    rendered = []

    def _render(_tab, image_path, refresh_preview=False):
        rendered.append((image_path, refresh_preview))
        return output

    monkeypatch.setattr(ImageRenderer, 'render_persisted_translation_state', _render)
    manga_tab = SimpleNamespace(
        _imported_ocr_preview_generation=0,
        _log=lambda *_args: None,
        update_queue=Queue(),
    )

    MangaTranslationTab._start_imported_ocr_preview_refresh(
        manga_tab,
        matches,
        exclude_path=current,
    )
    manga_tab._imported_ocr_preview_thread.join(timeout=5)

    assert rendered == [(background, False)]
    update = manga_tab.update_queue.get_nowait()
    assert update == ('preview_update', {
        'translated_path': output,
        'source_path': background,
        'switch_to_output': True,
    })


def test_imported_regions_keep_session_provenance_for_glossary_handoff(tmp_path):
    image = os.path.abspath(tmp_path / 'page.png')
    page = {
        'regions': [{
            'text': 'source',
            'translated_text': 'translated',
            'bounding_box': [1, 2, 30, 40],
        }],
    }
    normalized = os.path.normcase(os.path.abspath(image))
    manga_tab = SimpleNamespace(
        _imported_ocr_document={'pages': [page]},
        _imported_ocr_page_map={normalized: page},
    )

    regions = MangaTranslationTab._resolve_imported_ocr_regions(manga_tab, image)

    assert regions[0].translated_text == 'translated'
    assert regions[0]._imported_ocr_session is True


def test_folder_drop_recurses_and_skips_generated_output_folders(tmp_path):
    chapter = tmp_path / "chapter"
    nested = chapter / "nested"
    translated = chapter / "001_translated"
    ocr_dir = chapter / "OCR Text"
    nested.mkdir(parents=True)
    translated.mkdir()
    ocr_dir.mkdir()
    page_10 = chapter / "10.png"
    page_2 = chapter / "2.png"
    nested_page = nested / "3.webp"
    generated_page = translated / "1.png"
    ocr_artifact = ocr_dir / "preview.jpg"
    for path in (page_10, page_2, nested_page, generated_page, ocr_artifact):
        path.write_bytes(b"image")

    harness = _DropHarness()
    MangaTranslationTab._add_dropped_manga_paths(harness, [str(chapter)])

    assert harness.selected_files == [
        os.path.abspath(page_2),
        os.path.abspath(nested_page),
        os.path.abspath(page_10),
    ]
    assert harness.list_items == harness.selected_files
    assert harness.file_listbox.current_row == 0
    assert harness.manga_selected_folder_roots == [os.path.abspath(chapter)]


def test_translation_lifecycle_events_only_match_their_run():
    assert _translation_run_token_matches(current_token=2, event_token=2) is True
    assert _translation_run_token_matches(current_token=2, event_token=1) is False


def test_stale_completion_cannot_reset_a_new_translation_run():
    manga_tab = SimpleNamespace(
        _translation_start_token=2,
        is_running=True,
        _translation_startup_pending=True,
        _translation_start_cancel_requested=False,
    )

    reset = MangaTranslationTab._reset_ui_state(
        manga_tab,
        expected_start_token=1,
    )

    assert reset is False
    assert manga_tab.is_running is True
    assert manga_tab._translation_startup_pending is True


def test_manual_translate_uses_live_full_page_setting_not_stale_batch_snapshot():
    manga_tab = SimpleNamespace(
        full_page_context_value=True,
        _batch_full_page_context_enabled=False,
        main_gui=SimpleNamespace(config={'manga_full_page_context': False}),
    )

    assert ImageRenderer._manual_translate_full_page_context_enabled(manga_tab) is True


def test_missing_rendered_image_does_not_delete_imported_translation(tmp_path):
    image = tmp_path / 'page.png'
    image.write_bytes(b'image')

    class _StateManager:
        def __init__(self):
            self.state = {
                'rendered_image_path': str(tmp_path / 'missing-render.png'),
                'translated_texts': [
                    {
                        'original': {'region_index': 0, 'text': 'source'},
                        'translation': 'translated',
                        'bbox': [1, 2, 3, 4],
                    }
                ],
            }

        def get_state(self, _image_path):
            return self.state

        def set_state(self, _image_path, state, save=True):
            self.state = state

    manager = _StateManager()
    manga_tab = SimpleNamespace(image_state_manager=manager)

    ImageRenderer._validate_and_clean_stale_state(manga_tab, str(image))

    assert 'rendered_image_path' not in manager.state
    assert manager.state['translated_texts'][0]['translation'] == 'translated'


def test_manual_export_merges_live_translation_map(tmp_path):
    image = tmp_path / 'page.png'
    image.write_bytes(b'image')
    state = {
        'recognized_texts': [
            {'region_index': 0, 'text': 'source', 'bbox': [1, 2, 30, 40]}
        ],
        'viewer_rectangles': [
            {'x': 1, 'y': 2, 'width': 30, 'height': 40, 'shape': 'rect'}
        ],
    }

    class _StateManager:
        def get_state(self, _image_path):
            return state

    manga_tab = SimpleNamespace(
        image_state_manager=_StateManager(),
        image_preview_widget=SimpleNamespace(current_image_path=str(image)),
        _recognized_texts=state['recognized_texts'],
        _recognized_texts_image_path=str(image),
        _translation_data={
            0: {'original': 'source', 'translation': 'translated'}
        },
        _translation_data_image_path=str(image),
    )

    export_state = MangaTranslationTab._manual_editor_state_for_export(
        manga_tab,
        str(image),
    )
    regions = manga_ocr_io.canonical_regions_from_editor_state(export_state)

    assert regions[0]['text'] == 'source'
    assert regions[0]['translated_text'] == 'translated'

    page = manga_ocr_io.make_page(
        str(image),
        regions,
        editor_state=export_state,
    )
    document = manga_ocr_io.create_document([page], workflow='manual-editor')
    output = tmp_path / 'manual-session.json'
    manga_ocr_io.write_document(str(output), document)

    imported = manga_ocr_io.load_document(str(output))
    imported_state = manga_ocr_io.editor_state_from_page(imported['pages'][0])
    assert imported_state['recognized_texts'][0]['text'] == 'source'
    assert imported_state['translated_texts'][0]['translation'] == 'translated'


def test_manga_ocr_export_filename_includes_timestamp():
    manga_tab = SimpleNamespace(
        _manga_ocr_default_filename=lambda: 'chapter_ocr.json'
    )

    filename = MangaTranslationTab._manga_ocr_timestamped_export_filename(
        manga_tab,
        timestamp='20260731_193045',
    )

    assert filename == 'chapter_ocr_20260731_193045.json'


def test_manga_ocr_export_dialog_path_defaults_to_ocr_folder(tmp_path):
    ocr_folder = tmp_path / 'custom-output' / 'OCR Text'
    manga_tab = SimpleNamespace(
        _manga_ocr_output_dir=lambda: str(ocr_folder),
    )

    initial_path = MangaTranslationTab._manga_ocr_save_dialog_path(
        manga_tab,
        'chapter_ocr_20260731_193045.json',
    )

    assert initial_path == os.path.join(
        str(ocr_folder),
        'chapter_ocr_20260731_193045.json',
    )
    assert ocr_folder.is_dir()


def test_auto_ocr_folder_uses_the_epub_default_output_root(tmp_path, monkeypatch):
    app_output_root = tmp_path / 'app-output'
    monkeypatch.delenv('OUTPUT_DIRECTORY', raising=False)
    # U8: the method moved to manga_env (inherited by MangaTranslationTab); patch where it looks up
    monkeypatch.setattr(
        sys.modules[MangaTranslationTab._manga_ocr_output_dir.__module__],
        '_get_app_dir',
        lambda: str(app_output_root),
    )
    manga_tab = SimpleNamespace(main_gui=SimpleNamespace(config={}))

    output_dir = MangaTranslationTab._manga_ocr_output_dir(manga_tab)

    assert output_dir == os.path.join(str(app_output_root), 'OCR Text')


def test_auto_ocr_folder_respects_output_directory_override(tmp_path, monkeypatch):
    override_root = tmp_path / 'custom-output'
    monkeypatch.delenv('OUTPUT_DIRECTORY', raising=False)
    manga_tab = SimpleNamespace(
        main_gui=SimpleNamespace(config={'output_directory': str(override_root)})
    )

    output_dir = MangaTranslationTab._manga_ocr_output_dir(manga_tab)

    assert output_dir == os.path.join(str(override_root), 'OCR Text')


def test_ocr_import_dialog_defaults_to_auto_saved_ocr_folder(tmp_path, monkeypatch):
    ocr_folder = tmp_path / 'custom-output' / 'OCR Text'
    session = tmp_path / 'session.json'
    manga_ocr_io.write_document(
        str(session),
        manga_ocr_io.create_document([], workflow='automatic'),
    )
    captured = {}

    def _choose_file(_parent, _title, initial_dir, _filters):
        captured['initial_dir'] = initial_dir
        return str(session), ''

    monkeypatch.setattr(manga_integration.QFileDialog, 'getOpenFileName', _choose_file)
    manga_tab = SimpleNamespace(
        dialog=object(),
        _manga_ocr_output_dir=lambda: str(ocr_folder),
    )

    path = MangaTranslationTab._choose_ocr_document_path(
        manga_tab,
        'Import Manga OCR Text',
    )

    assert captured['initial_dir'] == str(ocr_folder)
    assert ocr_folder.is_dir()
    assert path == str(session)


def test_automatic_ocr_session_never_drops_saved_translation(tmp_path):
    image = tmp_path / 'page.png'
    image.write_bytes(b'image')
    output = tmp_path / 'chapter_ocr_20260731_193045.json'
    document = manga_ocr_io.create_document(
        [],
        workflow='automatic',
        source_root=str(tmp_path),
    )
    manga_tab = SimpleNamespace(
        _automatic_ocr_document=document,
        _automatic_ocr_export_path=str(output),
        _ocr_io_lock=threading.Lock(),
        selected_files=[str(image)],
        _current_manga_processing_files=lambda: [str(image)],
        _log=lambda *_args: None,
    )
    translated_region = {
        'rect_index': 0,
        'text': 'source',
        'translated_text': 'translated',
        'bounding_box': [1, 2, 30, 40],
    }
    ocr_only_region = {
        'rect_index': 0,
        'text': 'source',
        'translated_text': None,
        'bounding_box': [1, 2, 30, 40],
    }

    MangaTranslationTab._record_automatic_ocr_page(
        manga_tab,
        str(image),
        [translated_region],
    )
    MangaTranslationTab._record_automatic_ocr_page(
        manga_tab,
        str(image),
        [ocr_only_region],
    )

    saved = manga_ocr_io.load_document(str(output))
    assert saved['pages'][0]['regions'][0]['translated_text'] == 'translated'


def test_imported_translation_rerenders_and_reloads_preview(tmp_path, monkeypatch):
    image = tmp_path / 'page.png'
    image.write_bytes(b'image')
    rendered = tmp_path / 'page_translated' / 'page.png'
    state = {
        'translated_texts': [
            {
                'original': {'region_index': 0, 'text': 'source'},
                'translation': 'translated',
                'bbox': [1, 2, 30, 40],
            }
        ],
    }

    class _StateManager:
        def get_state(self, _image_path):
            return state

    class _Viewer:
        def __init__(self):
            self.loaded = []

        def load_image(self, path):
            self.loaded.append(path)

    output_viewer = _Viewer()
    preview_loads = []
    preview = SimpleNamespace(
        source_display_mode='original',
        cleaned_images_enabled=False,
        current_translated_path=None,
        output_viewer=output_viewer,
        load_image=lambda path, **kwargs: preview_loads.append((path, kwargs)),
    )
    logs = []
    manga_tab = SimpleNamespace(
        image_state_manager=_StateManager(),
        image_preview_widget=preview,
        main_gui=SimpleNamespace(config={}),
        _log=lambda message, level: logs.append((message, level)),
    )

    def _render_imported(_tab):
        rendered.parent.mkdir()
        rendered.write_bytes(b'rendered')
        state['rendered_image_path'] = str(rendered)

    monkeypatch.setattr(ImageRenderer, 'save_positions_and_rerender', _render_imported)

    refreshed = MangaTranslationTab._refresh_imported_manual_preview(
        manga_tab,
        str(image),
    )

    assert refreshed is True
    assert preview.source_display_mode == 'translated'
    assert preview.current_translated_path == str(rendered)
    assert output_viewer.loaded == [str(rendered)]
    assert preview_loads == [
        (
            str(image),
            {'preserve_rectangles': True, 'preserve_text_overlays': True},
        )
    ]


def test_completed_translation_switches_and_refreshes_both_previews(tmp_path):
    image = tmp_path / 'page.png'
    image.write_bytes(b'image')
    translated = tmp_path / 'page_translated' / 'page.png'
    translated.parent.mkdir()
    translated.write_bytes(b'translated')

    class _Viewer:
        def __init__(self):
            self.loaded = []

        def load_image(self, path):
            self.loaded.append(path)

    class _Toggle:
        def setText(self, text):
            self.text = text

        def setToolTip(self, text):
            self.tooltip = text

    class _StateManager:
        def __init__(self):
            self.updated = []

        def update_state(self, path, state):
            self.updated.append((path, state))

    output_viewer = _Viewer()
    preview_loads = []
    preview = SimpleNamespace(
        current_image_path=str(image),
        current_translated_path=None,
        source_display_mode='original',
        cleaned_images_enabled=False,
        cleaned_toggle_btn=_Toggle(),
        output_viewer=output_viewer,
        load_image=lambda path, **kwargs: preview_loads.append((path, kwargs)),
    )
    state_manager = _StateManager()
    manga_tab = SimpleNamespace(
        image_preview_widget=preview,
        image_state_manager=state_manager,
        _log=lambda *_args: None,
    )

    refreshed = MangaTranslationTab._apply_completed_translation_preview(
        manga_tab,
        {
            'translated_path': str(translated),
            'source_path': str(image),
            'switch_to_output': True,
        },
    )

    assert refreshed is True
    assert preview.source_display_mode == 'translated'
    assert preview.cleaned_images_enabled is True
    assert preview.current_translated_path == str(translated)
    assert output_viewer.loaded == [str(translated)]
    assert preview_loads == [(
        str(image),
        {'preserve_rectangles': True, 'preserve_text_overlays': True},
    )]
    assert state_manager.updated == [(
        str(image),
        {'rendered_image_path': str(translated)},
    )]


@pytest.mark.parametrize('context', ['manga', 'manga_ocr'])
def test_manga_api_failure_fallback_is_blank(context):
    client = object.__new__(UnifiedClient)
    assert client._handle_empty_result([], context, {'error': 'content_filter'}) == ''


@pytest.mark.parametrize(
    ('content', 'finish_reason', 'expected'),
    [
        ('[AI RESPONSE UNAVAILABLE]', 'error', []),
        ('[RATE LIMITED]', 'stop', []),
        ('partial OCR text', 'content_filter', []),
        ('こんにちは', 'stop', ['こんにちは']),
    ],
)
def test_custom_manga_ocr_does_not_keep_api_failures(
    monkeypatch, content, finish_reason, expected
):
    provider = CustomAPIProvider()
    provider.is_loaded = True
    provider.max_retries = 1
    provider.client = SimpleNamespace(send=lambda **_kwargs: (content, finish_reason))
    monkeypatch.setattr(provider, '_apply_manga_ocr_thinking_override', lambda: None)
    monkeypatch.setattr(provider, '_restore_thinking_override', lambda _value: None)

    results = provider.detect_text(np.zeros((60, 60, 3), dtype=np.uint8))
    assert [result.text for result in results] == expected


def test_custom_manga_ocr_no_text_marker_does_not_retry(monkeypatch):
    provider = CustomAPIProvider()
    provider.is_loaded = True
    provider.max_retries = 3
    calls = []
    provider.client = SimpleNamespace(send=lambda **kwargs: (calls.append(kwargs) or '[AI RESPONSE UNAVAILABLE]', 'stop'))
    monkeypatch.setattr(provider, '_apply_manga_ocr_thinking_override', lambda: None)
    monkeypatch.setattr(provider, '_restore_thinking_override', lambda _value: None)

    assert provider.detect_text(np.zeros((60, 60, 3), dtype=np.uint8)) == []
    assert len(calls) == 1


def test_ocr_manager_discards_no_text_marker_from_ai_provider():
    manager = object.__new__(OCRManager)
    manager.current_provider = 'Qwen2-VL'
    manager.providers = {
        'Qwen2-VL': SimpleNamespace(detect_text=lambda *_args, **_kwargs: [
            OCRResult(text='[AI RESPONSE UNAVAILABLE]', bbox=(0, 0, 5, 5), confidence=1.0),
            OCRResult(text='こんにちは', bbox=(5, 5, 10, 10), confidence=1.0),
        ])
    }

    results = manager.detect_text(np.zeros((20, 20, 3), dtype=np.uint8))
    assert [result.text for result in results] == ['こんにちは']


@pytest.mark.parametrize(
    ('content', 'expected'),
    [
        ('[AI RESPONSE UNAVAILABLE]', ''),
        ('[Translation Error: timeout]', ''),
        ('A real translation', 'A real translation'),
    ],
)
def test_manga_output_drops_error_placeholders(content, expected):
    assert ImageRenderer._manga_output_text(content) == expected


def test_automatic_manga_translation_leaves_failed_api_region_blank(monkeypatch):
    translator = object.__new__(manga_translator.MangaTranslator)
    translator.main_gui = SimpleNamespace(
        profile_var='Default',
        prompt_profiles={'Default': 'Translate the text'},
    )
    translator.client = object()
    translator.temperature = 0
    translator.max_tokens = 100
    translator.contextual_enabled = False
    translator.history_manager = None
    translator.visual_context_enabled = False
    translator.input_token_limit = None
    translator._log = lambda *_args, **_kwargs: None
    translator._check_stop = lambda: False
    translator._append_manga_glossary_to_system_prompt = lambda prompt, **_kwargs: prompt
    monkeypatch.setattr(
        manga_translator,
        'send_with_interrupt',
        lambda **_kwargs: ('[AI RESPONSE UNAVAILABLE]', 'error', None),
    )

    assert translator.translate_text('source text') == ''


@pytest.mark.parametrize('disable_performance_mode', [False, True])
def test_custom_image_edit_chunks_tall_page_without_cutting_text_boxes(disable_performance_mode):
    inpainter = object.__new__(LocalInpainter)
    inpainter.config = {
        'manga_settings': {'preprocessing': {'chunk_height': 2000, 'chunk_overlap': 100}},
        'manga_disable_inpaint_performance_mode': disable_performance_mode,
    }
    inpainter.current_method = 'custom-image-edit'
    inpainter.model_loaded = True
    inpainter._mp_enabled = False
    inpainter._check_stop = lambda: False
    inpainter._log = lambda *_args, **_kwargs: None
    inpainter._sync_inpainter_key_pool_from_config = lambda: None

    image = np.zeros((4500, 20, 3), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    boxes = [(100, 150), (1950, 2050), (3800, 3850)]
    for top, bottom in boxes:
        mask[top:bottom, 2:18] = 255

    ranges = inpainter._custom_image_edit_chunk_ranges(mask, 2000, 100)
    assert len(ranges) == 3
    for top, bottom in boxes:
        assert sum(start <= top and end >= bottom for start, end in ranges) == 1
    for (_, previous_end), (next_start, _) in zip(ranges, ranges[1:]):
        assert 0 <= previous_end - next_start <= 100
        assert not np.any(mask[next_start:previous_end])

    requests = []

    def fake_edit(crop, crop_mask, iterations=None, _pool_prepared=False):
        requests.append((crop.shape[0], int(np.count_nonzero(crop_mask))))
        edited = crop.copy()
        edited[crop_mask > 0] = (0, 0, 255)
        return edited

    inpainter._custom_image_edit_inpaint = fake_edit
    result = inpainter.inpaint(image, mask)

    assert len(requests) == 3
    assert all(masked_pixels > 0 for _, masked_pixels in requests)
    assert result.shape == image.shape
    assert np.all(result[mask > 0] == (0, 0, 255))
    assert np.all(result[mask == 0] == 0)

    requests.clear()
    blank_result = inpainter.inpaint(image, np.zeros_like(mask))
    assert np.array_equal(blank_result, image)
    assert not requests


def test_custom_image_edit_keeps_a_box_whole_when_crop_must_grow():
    mask = np.zeros((4300, 8), dtype=np.uint8)
    mask[1700:2400, 1:7] = 255
    mask[3500:3550, 1:7] = 255

    ranges = LocalInpainter._custom_image_edit_chunk_ranges(mask, 2000, 100)

    assert len(ranges) >= 2
    assert any(start <= 1700 and end >= 2400 for start, end in ranges)
    assert all(not np.any(mask[start:start + 1]) for start, _ in ranges[1:])


def test_custom_image_edit_logs_actual_request_count_for_sparse_page(capsys):
    inpainter = object.__new__(LocalInpainter)
    inpainter.config = {
        'manga_settings': {'preprocessing': {'chunk_height': 2000, 'chunk_overlap': 100}}
    }
    inpainter._check_stop = lambda: False
    inpainter._sync_inpainter_key_pool_from_config = lambda: None
    panel_messages = []
    inpainter.set_log_callback(lambda message, level: panel_messages.append(message))
    requests = []

    def fake_edit(crop, crop_mask, iterations=None, _pool_prepared=False):
        requests.append(crop.shape[0])
        return crop.copy()

    inpainter._custom_image_edit_inpaint = fake_edit
    image = np.zeros((4500, 20, 3), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[100:150, 2:18] = 255

    inpainter._custom_image_edit_inpaint_chunked(image, mask)

    assert len(requests) == 1
    assert any('1 crop request' in message for message in panel_messages)
    assert any('request 1/1' in message for message in panel_messages)
    console = capsys.readouterr().out
    assert '1 crop request' in console
    assert 'request 1/1' in console


@pytest.mark.parametrize(
    ('enabled', 'image_size', 'translation_size', 'expected_workers'),
    [
        (True, 2, 3, 2),
        (False, 1, 3, 3),
    ],
)
def test_custom_image_edit_crops_run_in_parallel_with_selected_limit(
    enabled, image_size, translation_size, expected_workers
):
    inpainter = object.__new__(LocalInpainter)
    inpainter.config = {
        'manga_settings': {'preprocessing': {'chunk_height': 2000, 'chunk_overlap': 100}},
        'manga_batch_image_requests_enabled': enabled,
        'manga_batch_image_requests_size': image_size,
        'batch_size': translation_size,
    }
    inpainter._check_stop = lambda: False
    inpainter._sync_inpainter_key_pool_from_config = lambda: None
    messages = []
    inpainter.set_log_callback(lambda message, level: messages.append(message))
    image = np.zeros((4500, 20, 3), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for top, bottom in [(100, 150), (1950, 2050), (3800, 3850)]:
        mask[top:bottom, 2:18] = 255

    barrier = threading.Barrier(expected_workers)
    lock = threading.Lock()
    active = 0
    peak = 0
    calls = 0

    def fake_edit(crop, crop_mask, iterations=None, _pool_prepared=False):
        nonlocal active, peak, calls
        assert _pool_prepared
        with lock:
            active += 1
            peak = max(peak, active)
            calls += 1
            call_number = calls
        if call_number <= expected_workers:
            barrier.wait(timeout=3)
        with lock:
            active -= 1
        edited = crop.copy()
        edited[crop_mask > 0] = (0, 0, 255)
        return edited

    inpainter._custom_image_edit_inpaint = fake_edit
    result = inpainter._custom_image_edit_inpaint_chunked(image, mask)

    assert peak == expected_workers
    assert np.all(result[mask > 0] == (0, 0, 255))
    assert any(f'concurrency: {expected_workers} simultaneous' in message for message in messages)
    assert sum('Custom image edit request ' in message for message in messages) == 3


def test_custom_image_edit_finishes_current_page_on_graceful_stop(monkeypatch):
    monkeypatch.setenv('GRACEFUL_STOP', '1')
    inpainter = object.__new__(LocalInpainter)
    inpainter.config = {'manga_settings': {'preprocessing': {'chunk_height': 100, 'chunk_overlap': 10}}}
    inpainter.current_method = 'custom-image-edit'
    inpainter.model_loaded = True
    inpainter._mp_enabled = False
    inpainter._check_stop = lambda: True
    inpainter._log = lambda *_args, **_kwargs: None
    inpainter._sync_inpainter_key_pool_from_config = lambda: None
    calls = []

    def fake_edit(crop, crop_mask, **_kwargs):
        calls.append(crop.shape[0])
        return crop.copy()

    inpainter._custom_image_edit_inpaint = fake_edit
    image = np.zeros((240, 8, 3), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[20:30, 1:7] = 255
    mask[170:180, 1:7] = 255

    result = inpainter.inpaint(image, mask)

    assert result is not None
    assert len(calls) == 2


def test_custom_image_edit_discards_result_and_skips_queued_crop_on_force_stop(monkeypatch):
    monkeypatch.setenv('GRACEFUL_STOP', '0')
    inpainter = object.__new__(LocalInpainter)
    inpainter.config = {
        'manga_settings': {'preprocessing': {'chunk_height': 100, 'chunk_overlap': 10}},
        'manga_batch_image_requests_enabled': True,
        'manga_batch_image_requests_size': 1,
    }
    inpainter._log = lambda *_args, **_kwargs: None
    inpainter._sync_inpainter_key_pool_from_config = lambda: None
    stopped = False
    calls = []
    inpainter._check_stop = lambda: stopped

    def fake_edit(crop, crop_mask, **_kwargs):
        nonlocal stopped
        calls.append(crop.shape[0])
        stopped = True
        return crop.copy()

    inpainter._custom_image_edit_inpaint = fake_edit
    image = np.zeros((240, 8, 3), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[20:30, 1:7] = 255
    mask[170:180, 1:7] = 255

    assert inpainter._custom_image_edit_inpaint_chunked(image, mask) is None
    assert len(calls) == 1


def test_custom_image_edit_force_stop_during_request_setup_never_posts(monkeypatch):
    import requests

    monkeypatch.setenv('GRACEFUL_STOP', '0')
    monkeypatch.setenv('USE_INPAINTER_KEYS', '0')
    inpainter = object.__new__(LocalInpainter)
    inpainter.config = {'custom_image_edit_model': 'nan/wan-2.6-image-edit'}
    inpainter._custom_image_edit_use_current_provider = False
    inpainter._custom_image_edit_endpoint = 'https://nano-gpt.com/v1'
    stopped = False
    inpainter._check_stop = lambda: stopped

    def log(message, *_args):
        nonlocal stopped
        if 'request starting' in message:
            stopped = True

    inpainter._log = log
    monkeypatch.setattr(requests, 'post', lambda *_args, **_kwargs: pytest.fail('request sent after force stop'))
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.full((16, 16), 255, dtype=np.uint8)

    assert inpainter._custom_image_edit_inpaint(image, mask, _pool_prepared=True) is None
    assert stopped


def test_custom_image_edit_accepts_fallback_image_url(monkeypatch):
    import requests
    import unified_api_client

    image_url = 'https://images.example.test/edited.png?signature=abc123'
    edited = np.full((16, 16, 3), 173, dtype=np.uint8)
    ok, encoded = cv2.imencode('.png', edited)
    assert ok
    downloads = []

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def send(self, _messages, **_kwargs):
            assert self._force_image_output_mode
            assert self._suppress_custom_image_edit_endpoint
            return image_url, 'stop'

        def _get_thread_local_client(self):
            return SimpleNamespace()

    def fake_get(url, **_kwargs):
        downloads.append(url)
        return SimpleNamespace(content=encoded.tobytes(), raise_for_status=lambda: None)

    monkeypatch.setenv('USE_INPAINTER_KEYS', '0')
    monkeypatch.setenv('CUSTOM_IMAGE_EDIT_MODEL', 'nan/wan-2.6-image-edit')
    monkeypatch.setenv('CUSTOM_IMAGE_EDIT_FULL_PAGE_OUTPUT', '100')
    monkeypatch.setattr(unified_api_client, 'UnifiedClient', FakeClient)
    monkeypatch.setattr(requests, 'get', fake_get)

    inpainter = object.__new__(LocalInpainter)
    inpainter.config = {'custom_image_edit_model': 'nan/wan-2.6-image-edit'}
    inpainter._custom_image_edit_use_current_provider = True
    inpainter._check_stop = lambda: False
    inpainter._log = lambda *_args: None
    inpainter._log_inpaint_diag = lambda *_args: None
    original = np.zeros_like(edited)
    mask = np.full(original.shape[:2], 255, dtype=np.uint8)

    result = inpainter._custom_image_edit_inpaint(original, mask, _pool_prepared=True)

    assert downloads == [image_url]
    assert np.array_equal(result, edited)


def test_fallback_client_preserves_image_edit_request_settings():
    source = UnifiedClient.__new__(UnifiedClient)
    source._thread_local = threading.local()
    source._force_image_output_mode = True
    source._forced_image_output_resolution = '2K'
    source._suppress_custom_image_edit_endpoint = True
    source._ignore_graceful_stop = True
    fallback = UnifiedClient.__new__(UnifiedClient)
    fallback._thread_local = threading.local()

    source._copy_retry_request_context_to_temp_client(fallback, context='Inpainter')

    assert fallback._force_image_output_mode is True
    assert fallback._forced_image_output_resolution == '2K'
    assert fallback._should_suppress_custom_image_edit_endpoint()
    assert fallback._ignore_graceful_stop is True


def test_clean_uses_detected_region_without_running_ocr(monkeypatch):
    image_path = 'C:/clean-test/page.png'
    monkeypatch.setenv('OUTPUT_DIRECTORY', 'C:/clean-test')
    monkeypatch.setattr(ImageRenderer, '_reset_cancellation_flags', lambda _self: None)
    monkeypatch.setattr(ImageRenderer, '_is_translation_cancelled', lambda _self: False)
    monkeypatch.setattr(
        ImageRenderer,
        '_run_ocr_on_regions',
        lambda *_args, **_kwargs: pytest.fail('Clean should not run OCR'),
    )
    monkeypatch.setattr(cv2, 'imread', lambda _path: np.full((32, 32, 3), 255, dtype=np.uint8))
    monkeypatch.setattr(ImageRenderer.os, 'makedirs', lambda *_args, **_kwargs: None)
    mask_pixels = []
    written_paths = []

    def fake_inpaint(image, mask, *_args):
        mask_pixels.append(int(np.count_nonzero(mask)))
        return image.copy()

    monkeypatch.setattr(cv2, 'inpaint', fake_inpaint)
    monkeypatch.setattr(cv2, 'imwrite', lambda path, _image: written_paths.append(path) or True)
    harness = SimpleNamespace(
        main_gui=SimpleNamespace(config={'manga_inpaint_method': 'none'}),
        image_preview_widget=SimpleNamespace(viewer=SimpleNamespace(rectangles=[])),
        update_queue=Queue(),
        _log=lambda *_args, **_kwargs: None,
    )

    ImageRenderer._run_clean_background(
        harness,
        image_path,
        [{'bbox': [8, 8, 12, 12], 'rect_index': 0, 'text': ''}],
    )

    assert len(mask_pixels) == 1 and mask_pixels[0] > 0
    assert len(written_paths) == 1
    assert written_paths[0].endswith('page_cleaned.png')


def test_custom_image_edit_receives_live_manga_crop_settings():
    preprocessing = {'chunk_height': 1200, 'chunk_overlap': 60}
    translator = object.__new__(manga_translator.MangaTranslator)
    translator.main_gui = SimpleNamespace(
        config={
            'manga_settings': {'preprocessing': preprocessing},
            'manga_batch_image_requests_enabled': False,
            'manga_batch_image_requests_size': 5,
            'batch_size': 2,
        },
        batch_size_var='7',
    )
    inpainter = SimpleNamespace(config={})

    translator._apply_custom_image_edit_request_config(inpainter)

    assert inpainter.config['manga_settings']['preprocessing'] == preprocessing
    assert inpainter.config['manga_batch_image_requests_enabled'] is False
    assert inpainter.config['manga_batch_image_requests_size'] == 5
    assert inpainter.config['batch_size'] == '7'


def test_legacy_ocr_prompt_migrates_once_without_replacing_custom_text():
    old = (
        "YOU ARE A TEXT EXTRACTION MACHINE. EXTRACT EXACTLY WHAT YOU SEE.\n\n"
        "Keep my custom instruction.\n"
        "8. IF YOU SEE NOTHING, OUTPUT NOTHING (empty response)\n"
        "If image is truly blank → Output: [empty/no response]"
    )
    migrated = MangaTranslationTab._migrate_legacy_manga_ocr_prompt(old)

    assert "Keep my custom instruction." in migrated
    assert "automatic text detector" in migrated
    assert "[AI RESPONSE UNAVAILABLE]" in migrated
    assert "IF YOU SEE NOTHING, OUTPUT NOTHING" not in migrated
    assert MangaTranslationTab._migrate_legacy_manga_ocr_prompt(migrated) == migrated
    assert MangaTranslationTab._migrate_legacy_manga_ocr_prompt("another custom prompt") == "another custom prompt"


def test_inpaint_mask_excludes_empty_and_unavailable_ocr_regions():
    config = {'manga_settings': {
        'auto_iterations': False,
        'mask_dilation': 0,
        'text_bubble_dilation_iterations': 0,
        'empty_bubble_dilation_iterations': 0,
        'free_text_dilation_iterations': 0,
    }}
    owner = SimpleNamespace(
        main_gui=SimpleNamespace(config=config),
        ocr_provider='custom-api',
        free_text_only_bg_opacity=False,
        _log=lambda *_args: None,
    )
    regions = [
        manga_translator.TextRegion(text=text, vertices=[], bounding_box=(x, 0, 10, 10), confidence=1,
                   region_type='text_block')
        for x, text in ((0, ''), (20, '[AI RESPONSE UNAVAILABLE]'),
                        (40, '[API RESPONSE UNAVAILABLE]'), (60, 'Hello'))
    ]
    for region in regions:
        region.bubble_type = 'text_bubble'

    image = np.zeros((20, 80, 3), dtype=np.uint8)
    mask = manga_translator.MangaTranslator.create_text_mask(owner, image, regions)

    assert not mask[:, :50].any()
    assert mask[:, 60:].any()
    assert np.array_equal(
        manga_translator.MangaTranslator.inpaint_regions(owner, image, np.zeros(image.shape[:2], dtype=np.uint8)),
        image,
    )


def test_custom_image_edit_only_selects_ocr_confirmed_regions(monkeypatch):
    regions = [
        {'bbox': [0, 0, 20, 20], 'bubble_type': 'text_bubble'},
        {'bbox': [30, 0, 20, 20], 'bubble_type': 'text_bubble'},
    ]
    monkeypatch.setattr(ImageRenderer, '_run_detection_sync', lambda *_args: regions)
    monkeypatch.setattr(ImageRenderer, '_get_detection_config', lambda *_args: {})
    monkeypatch.setattr(ImageRenderer, '_get_ocr_config', lambda *_args: {'provider': 'custom-api'})
    monkeypatch.setattr(ImageRenderer, '_run_ocr_on_regions', lambda *_args: [
        {'region_index': 0, 'bbox': regions[0]['bbox'], 'text': '[AI RESPONSE UNAVAILABLE]'},
        {'region_index': 1, 'bbox': regions[1]['bbox'], 'text': 'Hello'},
    ])
    owner = SimpleNamespace(
        _log=lambda *_args: None,
        update_queue=SimpleNamespace(put=lambda *_args: None),
        image_state_manager=SimpleNamespace(update_state=lambda *_args: None),
    )

    assert ImageRenderer._regions_for_custom_image_edit(
        owner, 'page.png', use_current_rectangles=False
    ) == [{**regions[1], 'rect_index': 1}]
