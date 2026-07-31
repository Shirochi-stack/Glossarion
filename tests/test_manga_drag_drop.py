import os
import threading
from queue import Queue
from types import SimpleNamespace

from PySide6.QtCore import QEvent, QMimeData, QUrl

import ImageRenderer
import manga_integration
from manga_integration import MangaTranslationTab, _translation_run_token_matches
import manga_ocr_io


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

    def _update_manga_image_range_display(self):
        pass

    def _persist_selected_files(self):
        pass

    def _log(self, message, level):
        self.logs.append((message, level))


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
        os.path.abspath(page_10),
        os.path.abspath(nested_page),
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
    monkeypatch.setattr(
        manga_integration,
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
