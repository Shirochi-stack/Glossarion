import os
import zipfile
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from epub_package import find_epub_opf_member
from image_archive_epub import is_epub_zip
from translator_gui import TranslatorGUI


def _gui(tmp_path, monkeypatch):
    monkeypatch.delenv('OUTPUT_DIRECTORY', raising=False)
    monkeypatch.setenv('EPUB_PATH', '')
    logs = []
    updates = []
    saved = []
    subtitle_root = tmp_path / 'subtitle_work'
    subtitle_root.mkdir()
    gui = SimpleNamespace(
        config={}, selected_files=[], manual_glossary_map={},
        stop_requested=False, append_log=logs.append,
        logs=logs, updates=updates, saved=saved,
        subtitle_zip_temp_root=str(subtitle_root),
        input_files_updated_signal=SimpleNamespace(emit=updates.append),
        save_config=lambda **kwargs: saved.append(kwargs),
        _get_output_base_dir=lambda path: str(tmp_path / 'output'),
    )
    for name in (
        '_has_epub_conversion_inputs',
        '_convert_zip_input_to_epub_if_needed',
        '_extract_subtitle_zip_input_if_needed',
        '_resolve_zip_inputs_for_translation',
    ):
        setattr(gui, name, MethodType(getattr(TranslatorGUI, name), gui))
    return gui


def _html_zip(path):
    with zipfile.ZipFile(path, 'w') as archive:
        for number in (10, 2):
            archive.writestr(
                f'Book/chapter{number}.xhtml',
                '<html xmlns="http://www.w3.org/1999/xhtml"><head>'
                f'<title>Chapter {number}</title></head><body>'
                f'<p>Chapter {number} text.</p></body></html>',
            )


def test_html_zip_resolves_to_epub_and_updates_selection_and_glossary(tmp_path, monkeypatch):
    gui = _gui(tmp_path, monkeypatch)
    source = tmp_path / 'Book.zip'
    _html_zip(source)
    target = str(source.with_suffix('.epub'))
    glossary = str(tmp_path / 'terms.csv')
    gui.selected_files = [str(source)]
    gui.manual_glossary_map = {str(source): glossary}

    assert gui._resolve_zip_inputs_for_translation() == [target]

    assert is_epub_zip(target)
    assert gui.selected_files == [target]
    assert gui.file_path == target
    assert gui.selected_epub_files == [target]
    assert gui.config['last_input_files'] == [target]
    assert gui.config['last_epub_path'] == target
    assert gui.manual_glossary_map[target] == glossary
    assert gui.updates == [[target]]
    assert gui.saved == [{'show_message': False}]
    assert gui._zip_inputs_resolved_for_current_run is True
    assert any('Converted HTML chapter ZIP' in line for line in gui.logs)


def test_html_zip_honors_conversion_directory_and_reuses_fresh_output(tmp_path, monkeypatch):
    import html_archive_epub

    gui = _gui(tmp_path, monkeypatch)
    source = tmp_path / 'Book.zip'
    _html_zip(source)
    gui._direct_text_archive_conversion_dir = str(tmp_path / 'converted')
    target = str(tmp_path / 'converted' / 'Book.epub')
    assert gui._convert_zip_input_to_epub_if_needed(str(source)) == target
    assert not source.with_suffix('.epub').exists()

    def unexpected_conversion(*args, **kwargs):
        pytest.fail('Fresh generated HTML EPUB should be reused')

    monkeypatch.setattr(html_archive_epub, 'convert_html_archive_to_epub', unexpected_conversion)
    assert gui._convert_zip_input_to_epub_if_needed(str(source)) == target
    assert any('Using existing HTML chapter EPUB' in line for line in gui.logs)


def test_changed_html_zip_rebuilds_existing_epub(tmp_path, monkeypatch):
    gui = _gui(tmp_path, monkeypatch)
    source = tmp_path / 'Book.zip'
    _html_zip(source)
    target = gui._convert_zip_input_to_epub_if_needed(str(source))
    with zipfile.ZipFile(source, 'a') as archive:
        archive.writestr('Book/chapter20.html', '<html><body><p>New chapter</p></body></html>')
    older = os.path.getmtime(source) - 10
    os.utime(target, (older, older))

    assert gui._convert_zip_input_to_epub_if_needed(str(source)) == target

    with zipfile.ZipFile(target) as archive:
        assert b'New chapter' in archive.read('Book/chapter20.html')


def test_cancelled_html_conversion_keeps_input_and_leaves_no_epub(tmp_path, monkeypatch):
    gui = _gui(tmp_path, monkeypatch)
    source = tmp_path / 'Book.zip'
    _html_zip(source)
    gui.stop_requested = True

    assert gui._convert_zip_input_to_epub_if_needed(str(source)) == str(source)

    assert not source.with_suffix('.epub').exists()
    assert gui._zip_conversion_active is False
    assert any('ZIP conversion cancelled' in line for line in gui.logs)


def test_epub_and_image_zip_routes_remain_available(tmp_path, monkeypatch):
    gui = _gui(tmp_path, monkeypatch)
    images = tmp_path / 'Images.zip'
    with zipfile.ZipFile(images, 'w') as archive:
        archive.writestr('page01.svg', '<svg xmlns="http://www.w3.org/2000/svg"/>')
    target = gui._convert_zip_input_to_epub_if_needed(str(images))
    assert target == str(images.with_suffix('.epub'))
    assert is_epub_zip(target)
    with zipfile.ZipFile(target) as archive:
        assert b'urn:glossarion:image-archive:' in archive.read(find_epub_opf_member(archive))

    epub_zip = tmp_path / 'Renamed.zip'
    epub_zip.write_bytes(Path(target).read_bytes())
    copied = gui._convert_zip_input_to_epub_if_needed(str(epub_zip))
    with open(copied, 'rb') as stream:
        assert stream.read() == epub_zip.read_bytes()


def test_subtitle_zip_still_expands_into_subtitle_input(tmp_path, monkeypatch):
    gui = _gui(tmp_path, monkeypatch)
    source = tmp_path / 'Subtitles.zip'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('Episode.srt', '1\n00:00:01,000 --> 00:00:02,000\nHello\n')
    gui.selected_files = [str(source)]

    resolved = gui._resolve_zip_inputs_for_translation()

    assert len(resolved) == 1
    assert resolved[0].endswith('.srt')
    assert os.path.isfile(resolved[0])
    assert gui.config['last_input_files'] == [str(source)]
    assert not source.with_suffix('.epub').exists()


@pytest.mark.parametrize('extension', ['.zip', '.cbz', '.EPUB', '.html', '.xhtml', '.HTM'])
def test_archive_inputs_do_not_search_parent_folder_for_unrelated_opf(tmp_path, monkeypatch, extension):
    gui = _gui(tmp_path, monkeypatch)
    selected = [str(tmp_path / f'Book{extension}')]
    lookups = []
    monkeypatch.setattr('translator_gui.find_opf_path', lambda path: lookups.append(path))

    assert TranslatorGUI._get_opf_file_order(gui, selected) == selected
    assert lookups == []


@pytest.mark.parametrize('extension', ['.html', '.xhtml', '.HTM'])
def test_standalone_html_prepares_one_epub_and_keeps_original_selection_for_next_launch(tmp_path, monkeypatch, extension):
    gui = _gui(tmp_path, monkeypatch)
    source = tmp_path / f'Chapter{extension}'
    source.write_text('<html><body><p>한국어 chapter text.</p></body></html>', encoding='utf-8')
    original = source.read_bytes()
    gui.selected_files = [str(source)]
    glossary = str(tmp_path / 'terms.csv')
    gui.manual_glossary_map = {str(source): glossary}

    assert gui._has_epub_conversion_inputs()
    resolved = gui._resolve_zip_inputs_for_translation()

    assert resolved == [str(source) + '.epub']
    assert is_epub_zip(resolved[0])
    assert gui.manual_glossary_map[resolved[0]] == glossary
    assert gui.config['last_input_files'] == [str(source)]
    assert source.read_bytes() == original
    # The next run must inspect the HTML again even while the UI shows EPUB.
    assert gui._has_epub_conversion_inputs()


def test_standalone_html_reprepares_changed_source_and_assets_on_next_run(tmp_path, monkeypatch):
    gui = _gui(tmp_path, monkeypatch)
    source = tmp_path / 'Chapter.html'
    source.write_text('<p>Before<img src="image.svg"></p>', encoding='utf-8')
    asset = tmp_path / 'image.svg'
    asset.write_text('<svg xmlns="http://www.w3.org/2000/svg"><text>Before</text></svg>', encoding='utf-8')
    gui.selected_files = [str(source)]
    target = gui._resolve_zip_inputs_for_translation()[0]
    source.write_text('<p>After<img src="image.svg"></p>', encoding='utf-8')
    asset.write_text('<svg xmlns="http://www.w3.org/2000/svg"><text>After</text></svg>', encoding='utf-8')
    gui._zip_inputs_resolved_for_current_run = False

    assert gui._resolve_zip_inputs_for_translation() == [target]

    with zipfile.ZipFile(target) as archive:
        documents = [name for name in archive.namelist() if name.lower().endswith('.html')]
        assert len(documents) == 1
        assert b'After' in archive.read(documents[0])
        images = [name for name in archive.namelist() if name.endswith('image.svg')]
        assert len(images) == 1
        assert archive.read(images[0]) == asset.read_bytes()


def test_standalone_html_preserves_existing_epubs_and_honors_conversion_directory(tmp_path, monkeypatch):
    gui = _gui(tmp_path, monkeypatch)
    source = tmp_path / 'Book.html'
    source.write_text('<p>Chapter</p>', encoding='utf-8')
    original_epub = source.with_suffix('.epub')
    original_epub.write_bytes(b'Original book EPUB')
    conversion_dir = tmp_path / 'prepared'
    conversion_dir.mkdir()
    collision = conversion_dir / 'Book.html.epub'
    collision.write_bytes(b'Unrelated EPUB at wrapper path')
    gui._direct_text_archive_conversion_dir = str(conversion_dir)

    target = gui._convert_zip_input_to_epub_if_needed(str(source))

    assert target == str(conversion_dir / 'Book.html.1.epub')
    assert is_epub_zip(target)
    assert original_epub.read_bytes() == b'Original book EPUB'
    assert collision.read_bytes() == b'Unrelated EPUB at wrapper path'


def test_cancelled_standalone_html_preparation_keeps_source(tmp_path, monkeypatch):
    gui = _gui(tmp_path, monkeypatch)
    source = tmp_path / 'Book.xhtml'
    source.write_text('<p>Original</p>', encoding='utf-8')
    gui.stop_requested = True

    assert gui._convert_zip_input_to_epub_if_needed(str(source)) == str(source)
    assert not Path(str(source) + '.epub').exists()
    assert any('HTML conversion cancelled' in line for line in gui.logs)
