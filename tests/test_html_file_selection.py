from types import SimpleNamespace

import pytest

import translator_gui
from translator_gui import TranslatorGUI


def _selection_gui():
    selections = []
    logs = []
    return SimpleNamespace(
        selections=selections,
        logs=logs,
        append_log=logs.append,
        _handle_file_selection=selections.append,
    )


def _html_files(folder):
    paths = []
    for name in ('chapter.html', 'chapter.htm', 'chapter.XHTML'):
        path = folder / name
        path.write_text('<p>Original chapter</p>', encoding='utf-8')
        paths.append(str(path))
    return sorted(paths)


def test_file_picker_lists_html_formats_and_accepts_the_selected_files(tmp_path, monkeypatch):
    paths = _html_files(tmp_path)
    gui = _selection_gui()
    filters = []

    def choose_files(_parent, _title, _directory, file_filter):
        filters.append(file_filter)
        return paths, 'HTML files (*.html *.htm *.xhtml)'

    monkeypatch.setattr(translator_gui.QFileDialog, 'getOpenFileNames', choose_files)

    TranslatorGUI.browse_files(gui)

    assert gui.selections == [paths]
    supported = filters[0].split(';;')[0]
    assert all(pattern in supported.split() for pattern in ('*.html', '*.htm', '*.xhtml'))
    assert 'HTML files (*.html *.htm *.xhtml)' in filters[0].split(';;')


@pytest.mark.parametrize('deep_scan', [False, True])
def test_folder_picker_includes_html_and_honors_deep_scan(tmp_path, monkeypatch, deep_scan):
    paths = _html_files(tmp_path)
    nested = tmp_path / 'nested'
    nested.mkdir()
    nested_paths = _html_files(nested)
    (tmp_path / 'unrelated.bin').write_bytes(b'ignored')
    gui = _selection_gui()
    gui.deep_scan_var = deep_scan
    monkeypatch.setattr(translator_gui.QFileDialog, 'getExistingDirectory', lambda *_args: str(tmp_path))

    TranslatorGUI.browse_folder(gui)

    assert gui.selections == [sorted(paths + (nested_paths if deep_scan else []))]


def test_empty_folder_message_lists_html_formats(tmp_path, monkeypatch):
    gui = _selection_gui()
    warnings = []
    monkeypatch.setattr(translator_gui.QFileDialog, 'getExistingDirectory', lambda *_args: str(tmp_path))
    monkeypatch.setattr(translator_gui.QMessageBox, 'warning', lambda *args: warnings.append(args[-1]))

    TranslatorGUI.browse_folder(gui)

    assert not gui.selections
    assert 'HTML, HTM, XHTML' in warnings[0]


@pytest.mark.parametrize('drop_folder', [False, True])
def test_main_window_drop_accepts_html_files_and_html_folder_contents(tmp_path, drop_folder):
    paths = _html_files(tmp_path)
    (tmp_path / 'unrelated.bin').write_bytes(b'ignored')
    nested = tmp_path / 'nested'
    nested.mkdir()
    _html_files(nested)
    dropped = [str(tmp_path)] if drop_folder else paths
    urls = [SimpleNamespace(toLocalFile=lambda path=path: path) for path in dropped]
    accepted = []
    event = SimpleNamespace(
        mimeData=lambda: SimpleNamespace(urls=lambda: urls),
        acceptProposedAction=lambda: accepted.append(True),
    )
    gui = _selection_gui()

    TranslatorGUI.dropEvent(gui, event)

    assert gui.selections == [paths]
    assert accepted == [True]


def test_multiple_html_selection_displays_count_without_converting_sources(tmp_path):
    paths = _html_files(tmp_path)
    texts = []
    gui = SimpleNamespace(
        config={},
        entry_epub=SimpleNamespace(clear=lambda: None, setText=texts.append),
        _reset_parallel_epub_pair_for_input_change=lambda: None,
        _normalize_windows_input_filenames=lambda paths: paths,
        _rename_input_for_existing_workspace_collision=lambda path: path,
        _update_entry_epub_tooltip=lambda: None,
        _clear_automatic_glossary_for_non_epub_selection=lambda paths: None,
        append_log=lambda _message: None,
        save_config=lambda **_kwargs: None,
    )

    TranslatorGUI._handle_file_selection(gui, paths)

    assert gui.selected_files == paths
    assert gui.file_path == paths[0]
    assert gui.config['last_input_files'] == paths
    assert texts[-1] == '3 files selected (3 HTML)'
    assert not list(tmp_path.glob('*.epub'))


@pytest.mark.parametrize('extension', ['.html', '.htm', '.XHTML'])
def test_normalized_html_source_is_persisted_for_next_session(tmp_path, extension):
    original = str(tmp_path / f'chapter..{extension}')
    normalized = str(tmp_path / f'chapter{extension}')
    gui = SimpleNamespace(
        config={},
        _windows_supported_input_path=lambda _path: normalized,
        save_config=lambda **_kwargs: None,
    )

    assert TranslatorGUI._normalize_windows_input_filenames(gui, [original]) == [normalized]
    assert gui.config['last_input_files'] == [normalized]
    assert gui.config['last_epub_path'] == normalized
