"""Interactive Images outside the step: the EPUB manifest and stylesheet link, the PDF image lookup, the
image reference map and the QA image counts leave Glossarion's images alone."""
import json
import os
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

import interactive_css_presets as presets  # noqa: E402
import interactive_images_core as core  # noqa: E402


def _png():
    from PIL import Image
    buffer = BytesIO()
    Image.new('RGB', (6, 6), (10, 120, 10)).save(buffer, format='PNG')
    return buffer.getvalue()


CHAPTER = ('<html><head><title>Ch 1</title></head><body><div class="glx-book" data-glx="book">\n<h1>Chapter 1</h1>\n'
           '<p>Mira walked to the gate.</p><figure class="glx-figure" data-glx="img"><img class="glx-illustration" '
           'src="Interactive_Images/Glossarion_ch001_1.png" alt=""/></figure>\n'
           '<p class="glx-dialogue" data-glx-cls="glx-dialogue">"Hello," said Tomas.</p>\n</div></body></html>')


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    out = tmp_path / 'book'
    (out / core.INTERACTIVE_FOLDER).mkdir(parents=True)
    (out / 'response_001_ch1.html').write_text(CHAPTER, encoding='utf-8')
    (out / 'translation_progress.json').write_text(json.dumps({'chapters': {'a': {
        'actual_num': 1, 'output_file': 'response_001_ch1.html', 'status': 'completed'}}}), encoding='utf-8')
    (out / core.INTERACTIVE_FOLDER / 'Glossarion_ch001_1.png').write_bytes(_png())
    (out / core.INTERACTIVE_FOLDER / 'Glossarion_ch009_1.png').write_bytes(_png())  # referenced by no chapter
    presets.write_book_stylesheet(str(out), presets.BUILTIN_PRESETS['Classic Book'])
    for name in ('EPUB_CSS_OVERRIDE_PATH', 'OUTPUT_DIRECTORY'):
        monkeypatch.delenv(name, raising=False)
    return out


def _compile(out, monkeypatch, *, legacy=False, attach=True, override=None):
    monkeypatch.setenv('EPUB_LAYOUT_MODE', 'epub2' if legacy else 'epub3')
    monkeypatch.setenv('LEGACY_EPUB_STRUCTURE', '1' if legacy else '0')
    monkeypatch.setenv('ATTACH_CSS_TO_CHAPTERS', '1' if attach else '0')
    if override:
        monkeypatch.setenv('EPUB_CSS_OVERRIDE_PATH', str(override))
    import epub_converter
    epub_converter.compile_epub(str(out), log_callback=lambda _m: None)
    epub = next(p for p in out.iterdir() if p.suffix == '.epub')
    with zipfile.ZipFile(epub) as zf:
        names = zf.namelist()
        opf = zf.read(next(n for n in names if n.endswith('.opf'))).decode('utf-8')
        chapter = zf.read(next(n for n in names if n.endswith('response_001_ch1.html'))).decode('utf-8')
    epub.unlink()
    return names, opf, chapter


@pytest.mark.parametrize("legacy", (False, True))
def test_epub_manifest_has_the_images_and_links_the_stylesheet_last(workspace, monkeypatch, tmp_path, legacy):
    override = tmp_path / 'mine.css'
    override.write_text('p { color: black; }', encoding='utf-8')
    names, opf, chapter = _compile(workspace, monkeypatch, legacy=legacy, override=override)
    assert any(n.endswith('Interactive_Images/Glossarion_ch001_1.png') for n in names)
    assert not any('Glossarion_ch009_1' in n for n in names)            # only referenced images
    assert not any('references' in n or n.endswith('.json') and 'Interactive' in n for n in names)
    assert 'href="Interactive_Images/Glossarion_ch001_1.png"' in opf
    assert 'glossarion_interactive.css' in opf
    prefix = '../' if legacy else ''
    assert f'src="{prefix}Interactive_Images/Glossarion_ch001_1.png"' in chapter
    links = [m for m in __import__('re').findall(r'<link[^>]*href="([^"]+)"', chapter)]
    assert links[-1].endswith('glossarion_interactive.css')
    assert any(link.endswith('mine.css') for link in links[:-1])
    assert 'class="glx-book"' in chapter
    # Load CSS's sync (old css/ files removed) never touched the interactive stylesheet
    assert (workspace / core.INTERACTIVE_FOLDER / presets.STYLESHEET_FILENAME).is_file()


def test_stylesheet_is_linked_without_attach_css(workspace, monkeypatch):
    _names, _opf, chapter = _compile(workspace, monkeypatch, attach=False)
    assert 'glossarion_interactive.css' in chapter


def test_pdf_lookup_includes_interactive_images(workspace):
    import epub_converter
    compiler = object.__new__(epub_converter.EPUBCompiler)
    compiler.output_dir = str(workspace)
    compiler.log = lambda _m: None
    assert compiler._interactive_image_files() == ['Glossarion_ch001_1.png']


def test_image_reference_map_never_binds_interactive_images():
    from bs4 import BeautifulSoup
    import image_reference_map

    soup = BeautifulSoup('<p><img src="images/a.png"/><img src="Interactive_Images/Glossarion_ch001_1.png"/>'
                         '<img src="../Interactive_Images/Glossarion_ch002_3.jpg"/></p>', 'html.parser')
    assert [url for url, _setter in image_reference_map._references(soup)] == ['images/a.png']


def test_qa_image_counts_ignore_interactive_images():
    from scan_html_folder import _html_preservation_profile, detect_missing_images

    info = {1: {'image_count': 1, 'image_srcs': ['images/a.png']}}
    translated = ('<p><img src="images/a.png"/></p><figure class="glx-figure" data-glx="img">'
                  '<img src="Interactive_Images/Glossarion_ch001_1.png" alt=""/></figure>')
    assert detect_missing_images(translated, 1, info) == (False, [])
    plain = '<p><img src="images/a.png"/></p>'
    assert _html_preservation_profile(translated)['image_srcs'] == _html_preservation_profile(plain)['image_srcs']
