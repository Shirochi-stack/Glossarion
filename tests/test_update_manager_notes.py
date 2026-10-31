import os
import threading
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6.QtCore import QBuffer, QIODevice, QTimer, QUrl
from PySide6.QtGui import QImage, QTextDocument
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGridLayout, QGroupBox, QMainWindow, QRadioButton

import update_manager


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def image_formats(browser):
    result = []
    block = browser.document().begin()
    while block.isValid():
        it = block.begin()
        while not it.atEnd():
            fragment = it.fragment()
            if fragment.isValid() and fragment.charFormat().isImageFormat():
                result.append(fragment.charFormat().toImageFormat())
            it += 1
        block = block.next()
    return result


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert predicate()


@pytest.fixture
def image_response(monkeypatch):
    image = QImage(1000, 500, QImage.Format_RGB32)
    image.fill(0xff4488aa)
    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    image.save(buffer, 'PNG')
    payload = bytes(buffer.data())
    gate = threading.Event()
    calls = []
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def raise_for_status(self):
            pass
        def iter_content(self, size):
            gate.wait(3)
            yield payload
    def get(url, **kwargs):
        calls.append((url, threading.get_ident()))
        return Response()
    monkeypatch.setattr(update_manager.requests, 'get', get)
    yield gate, calls
    gate.set()
    app = QApplication.instance()
    app._release_image_pool.waitForDone(4000)
    app.processEvents()


@pytest.mark.parametrize('pixel_ratio', [1.0, 2.0])
def test_html_and_markdown_images_load_without_blocking_and_fit_on_resize(app, image_response, pixel_ratio):
    gate, calls = image_response
    browser = update_manager.ReleaseNotesBrowser()
    browser.devicePixelRatioF = lambda: pixel_ratio
    browser.resize(450, 350)
    browser.show()
    url = 'https://github.com/user-attachments/assets/test-image'
    update_manager.UpdateManager.format_markdown_to_qt(None, browser,
        f'## Patch notes\n\n<img width="1000" height="500" alt="Screenshot" src="{url}" />\n\n'
        f'![Same screenshot]({url})\n\n- **Fix** with [details](https://github.com/example/release)')
    heartbeat = []
    QTimer.singleShot(0, lambda: heartbeat.append(True))
    wait_for(lambda: bool(heartbeat) and len(calls) == 1)
    assert calls[0][1] != threading.get_ident()
    assert '<img' not in browser.toPlainText()
    assert len(image_formats(browser)) == 2
    gate.set()
    wait_for(lambda: url in browser._images)
    assert not browser._images[url].isNull()
    for fmt in image_formats(browser):
        assert fmt.width() <= browser.viewport().width()
        assert fmt.height() == pytest.approx(fmt.width() / 2)
    browser.resize(300, 350)
    app.processEvents()
    assert all(fmt.width() <= browser.viewport().width() for fmt in image_formats(browser))
    browser.resize(1200, 600)
    app.processEvents()
    assert all(fmt.width() == min(560, 1000 / pixel_ratio) for fmt in image_formats(browser))
    assert all(fmt.height() <= 320 for fmt in image_formats(browser))
    assert all(fmt.isAnchor() and fmt.anchorHref() == url for fmt in image_formats(browser))
    preview = browser.document().resource(QTextDocument.ImageResource, QUrl(url))
    assert preview.devicePixelRatio() == pixel_ratio
    assert preview.width() <= browser._images[url].width()
    assert preview.width() == round(image_formats(browser)[0].width() * pixel_ratio)
    browser.close()
    browser.deleteLater()


def test_closing_notes_during_image_download_does_not_wait(app, image_response):
    gate, calls = image_response
    browser = update_manager.ReleaseNotesBrowser()
    browser.setMarkdown('![Image](https://github.com/user-attachments/assets/slow)')
    browser.show()
    wait_for(lambda: len(calls) == 1)
    browser.close()
    browser.deleteLater()
    QTest.qWait(20)
    assert not gate.is_set()
    gate.set()


def test_failed_image_keeps_patch_notes_readable(app, monkeypatch):
    calls = []
    def fail(url, **kwargs):
        calls.append(url)
        raise update_manager.requests.ConnectionError('offline')
    monkeypatch.setattr(update_manager.requests, 'get', fail)
    browser = update_manager.ReleaseNotesBrowser()
    browser.setMarkdown('## Notes\n\n![Screenshot](https://github.com/user-attachments/assets/offline)\n\nStill readable.')
    browser.show()
    wait_for(lambda: bool(browser._images))
    assert 'Still readable.' in browser.toPlainText()
    assert len(calls) == 1
    browser.close()
    browser.deleteLater()


def test_tag_like_text_stays_literal_and_keeps_the_notes_after_it(app):
    """v9.15.8: Qt's Markdown importer took <time> in config.corrupt-<time>.json for an open tag
    and dropped the text of every later bullet."""
    browser = update_manager.ReleaseNotesBrowser()
    browser.set_release_notes('### Fixes\n- kept as config.corrupt-<time>.json and restored\n  - a <br> and a <p> mention\n'
                              '- next bullet &amp; a < b\n\n### Packages\n- [Download](https://example.com/a) '
                              'or <https://example.com/auto>')
    plain = browser.toPlainText()
    assert 'kept as config.corrupt-<time>.json and restored' in plain
    assert 'a <br> and a <p> mention' in plain
    assert 'next bullet & a < b' in plain
    assert 'Packages' in plain
    assert 'https://example.com/a' in browser.toHtml() and 'https://example.com/auto' in browser.toHtml()
    browser.close()


def test_image_after_text_on_a_line_is_an_image_and_code_stays_code(app, image_response):
    url = 'https://github.com/user-attachments/assets/inline'
    browser = update_manager.ReleaseNotesBrowser()
    browser.set_release_notes(f'- A note.<img width="849" height="851" alt="Shot" src="{url}" /> And more.\n\n'
                              f'`<img src="{url}">` in code')
    plain = browser.toPlainText()
    assert len(image_formats(browser)) == 1 and image_formats(browser)[0].name() == url
    assert 'A note.' in plain and 'And more.' in plain
    assert f'<img src="{url}"> in code' in plain
    browser.close()


def test_update_manager_uses_two_columns_and_keeps_asset_selection(app, tmp_path):
    window = QMainWindow()
    window.config = {}
    manager = update_manager.UpdateManager(window, str(tmp_path))
    manager._detect_platform = lambda: 'windows'
    manager._detect_arch = lambda: 'x64'
    names = ['Glossarion.exe', 'L_Glossarion_Lite.exe', 'L_Glossarion_TurboLite.exe',
             'N_Glossarion_NoCuda.exe', 'L_Glossarion_MAC.dmg', 'L_Glossarion_MAC_Intel.dmg']
    release = {'tag_name': 'v9.10.8', 'body': '## Patch notes\n\n- A fix',
               'assets': [{'name': name, 'size': 60000000, 'browser_download_url': 'https://example.com/file'} for name in names]}
    manager.latest_release = release
    manager.all_releases = [release]
    manager.show_update_dialog()
    app.processEvents()
    dialog = manager._update_dialog
    group = dialog.findChild(QGroupBox, 'asset_group')
    layout = group.layout()
    assert isinstance(layout, QGridLayout)
    assert layout.rowCount() == 3
    assert layout.columnCount() == 2
    buttons = group.findChildren(QRadioButton)
    assert len(buttons) == 6
    assert manager.selected_asset['name'] == 'Glossarion.exe'
    buttons[1].click()
    assert manager.selected_asset['name'] == 'L_Glossarion_Lite.exe'
    assert all(not b.isEnabled() for b in buttons[-2:])
    assert all('Glossarion' in b.toolTip() for b in buttons)
    assert dialog.findChild(update_manager.ReleaseNotesBrowser).height() > group.height()
    dialog.close()
    window.deleteLater()



def test_release_heading_emoji_aliases_render_without_changing_authored_text(app):
    browser = update_manager.ReleaseNotesBrowser()
    notes = "## :tools: Fixes\n\n## :package: Packages\n\n- my exact release note\n\n## \U0001f6e0\ufe0f Existing emoji"
    browser.set_release_notes(notes)
    rendered = browser.toPlainText()
    assert '\U0001f6e0\ufe0f Fixes' in rendered
    assert '\U0001f4e6 Packages' in rendered
    assert 'my exact release note' in rendered
    assert ':tools:' not in rendered and ':package:' not in rendered
    assert '\U0001f6e0\ufe0f Existing emoji' in rendered
    browser.close()


def test_release_emoji_aliases_leave_code_escaped_text_and_urls_unchanged(app):
    browser = update_manager.ReleaseNotesBrowser()
    notes = ('## :tools: Fixes\n\n`:tools:` and ``:package:``\n\n'
             '```text\n:tools: :package:\n```\n\n'
             '~~~text\n:package:\n~~~\n\n'
             '[link](https://example.com/:package:)\n\n'
             r'\:tools: stays literal' + '\n\nhttps://example.com/:tools:')
    browser.set_release_notes(notes)
    plain = browser.toPlainText()
    assert '\U0001f6e0\ufe0f Fixes' in plain
    assert ':tools: and :package:' in plain
    assert ':tools: :package:' in plain
    assert ':tools: stays literal' in plain
    assert 'https://example.com/:tools:' in plain
    assert 'https://example.com/:package:' in browser.toHtml()
    browser.close()
