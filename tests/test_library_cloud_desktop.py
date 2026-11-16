"""Desktop Library cloud UI (library_cloud_ui + the epub_library / translator_gui wiring): Library
settings, Share, "Sync to Drive now" and the compile hooks. Everything runs against the local fake
Google / Gofile servers of test_cloud_drive_core; nothing talks to Google or a hosting site."""

import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

import pytest

pytest.importorskip("PySide6")

SRC = Path(__file__).resolve().parents[1] / "src"
HERE = Path(__file__).resolve().parent
for entry in (str(SRC), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QPoint  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QMenu  # noqa: E402

import drive_sync_core as dsc  # noqa: E402
import epub_library  # noqa: E402
import library_cloud as lc  # noqa: E402
import library_cloud_ui as ui  # noqa: E402
import library_core  # noqa: E402
import library_covers  # noqa: E402
import share_hosting as sh  # noqa: E402
from test_cloud_drive_core import FakeGofile, FakeGoogle, _browser  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Never the real Library, output root, config folder or encryption key file."""
    library = tmp_path / "_isolated" / "Library"
    output = tmp_path / "_isolated" / "Output"
    library.mkdir(parents=True)
    output.mkdir(parents=True)
    monkeypatch.setenv("GLOSSARION_LIBRARY_DIR", str(library))
    monkeypatch.setenv("OUTPUT_DIRECTORY", str(output))
    for module in (library_core, epub_library):
        monkeypatch.setattr(module, "_default_output_root", lambda: str(output))
    monkeypatch.setattr(library_covers, "_COVER_CACHE_DIR_OVERRIDE", str(tmp_path / "_isolated" / "covers"))
    import app_paths

    monkeypatch.setattr(app_paths, "CONFIG_FILE", str(tmp_path / "_isolated" / "config.json"))
    monkeypatch.setattr(lc, "_INSTANCE", None)
    yield
    current = lc._INSTANCE
    if current is not None:
        current.shutdown()
    lc.uninstall_compile_hook()


@pytest.fixture(autouse=True)
def delete_widgets_after_test(qapp):
    """The app opens these dialogs with WA_DeleteOnClose; the tests build them directly, so a closed
    dialog (and its combo popups and menus) lived to the end of the session and was destroyed after
    the QApplication. On Linux CI that crashed the interpreter at exit, after every test had passed."""
    before = list(qapp.topLevelWidgets())
    yield
    for widget in qapp.topLevelWidgets():
        if any(widget is known for known in before):
            continue
        widget.close()
        widget.deleteLater()
    for _ in range(3):
        qapp.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.fixture
def google():
    fake = FakeGoogle()
    yield fake
    fake.close()


@pytest.fixture
def box(monkeypatch):
    from cryptography.fernet import Fernet

    import api_key_encryption

    monkeypatch.setenv("GLOSSARION_API_KEY_FERNET", Fernet.generate_key().decode("ascii"))
    handler = api_key_encryption.APIKeyEncryption()
    assert handler.key_file is None
    return sh.SecretBox(handler=handler, device="computer")


def _cloud(google, tmp_path, box, *, providers=None, own_client=False):
    store = lc.LibraryCloudStore(str(tmp_path / "cfg" / "library_cloud.json"), secrets=box)
    endpoints = {"auth_url": google.base + "/auth", "token_url": google.base + "/token",
                 "revoke_url": google.base + "/revoke", "api_base": google.base + "/drive/v3",
                 "upload_base": google.base + "/upload/drive/v3"}
    if own_client:
        store.set_own_client(use_own=True, client_id="client-id", client_secret="client-secret")
    lines = []
    cloud = lc.LibraryCloud(store, endpoints=endpoints, providers=providers, log_line=lines.append,
                            on_change=ui._emit_changed)
    return cloud, store, lines


def _pump(app, until=None, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents(QEventLoop.AllEvents, 50)
        if until is not None and until():
            return True
        time.sleep(0.01)
    return until is None


def _book(tmp_path, compiled=True):
    folder = tmp_path / "_isolated" / "Output" / "The Book"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "translation_progress.json").write_text("{}", encoding="utf-8")
    book = {"path": str(folder), "output_folder": str(folder), "name": "The Book", "type": "in_progress"}
    if compiled:
        epub = folder / "The Book.epub"
        epub.write_bytes(b"compiled epub bytes")
        book["output_epub_path"] = str(epub)
    return book


# ---------------------------------------------------------------------------------------------------
# Library settings
# ---------------------------------------------------------------------------------------------------


def test_settings_start_signed_out_with_sign_in_disabled_and_save_the_drive_options(qapp, google, tmp_path, box):
    cloud, store, _ = _cloud(google, tmp_path, box)
    dialog = ui.LibrarySettingsDialog(None, cloud=cloud)
    try:
        # off by default; no client: Sign in is disabled with the one-line reason
        assert not dialog.drive_enabled.isChecked()
        assert dialog.mode.currentData() == "compiled" and dialog.folder.text() == "Glossarion"
        assert [dialog.mode.itemText(i) for i in range(dialog.mode.count())] == [dsc.MODE_LABELS[m]
                                                                                 for m in dsc.MODES]
        assert not dialog.sign_in_button.isEnabled()
        import google_drive_core as gdc

        assert dialog.sign_in_reason.text() == gdc.NO_DESKTOP_CLIENT
        assert dsc.FOLDER_NOTE in [w.text() for w in dialog.findChildren(type(dialog.sign_in_reason))]
        # the house dropdown: Halgakos arrow, wheel ignored
        assert "Halgakos.ico" in dialog.mode.styleSheet()
        # typing an own client enables Sign in before it is saved (as the text changes, no Save needed)
        dialog.own_client.setChecked(True)
        assert not dialog.sign_in_button.isEnabled()
        dialog.client_id.setText("my-client.apps.googleusercontent.com")
        assert not dialog.sign_in_button.isEnabled()
        dialog.client_secret.setText("my-secret-value")
        assert dialog.sign_in_button.isEnabled() and dialog.sign_in_reason.isHidden()
        dialog.drive_enabled.setChecked(True)
        dialog.folder.setText(" Books / Glossarion ")
        dialog.mode.setCurrentIndex(dialog.mode.findData("full_raw"))
        assert dialog.save()
        settings = store.drive_settings()
        assert (settings.enabled, settings.mode, settings.folder, settings.signed_in) == (
            True, "full_raw", "Books/Glossarion", False)
        assert store.own_client() == {"use_own": True, "client_id": "my-client.apps.googleusercontent.com",
                                      "has_secret": True}
        assert dialog.save_button.text() == "💾 Saved!"
        raw = Path(store.path).read_text("utf-8")
        assert "my-secret-value" not in raw
        assert dialog.client_secret.text() == "" and "Saved" in dialog.client_secret.placeholderText()
        assert google.requests == []
    finally:
        dialog.close()


def test_turning_a_site_on_asks_for_consent_first(qapp, google, tmp_path, box, monkeypatch):
    cloud, store, _ = _cloud(google, tmp_path, box)
    dialog = ui.LibrarySettingsDialog(None, cloud=cloud)
    shown = []

    def fake_exec(consent_dialog):
        shown.append((consent_dialog.windowTitle(), consent_dialog.confirm.isEnabled()))
        consent_dialog.checkbox.setChecked(True)
        assert consent_dialog.confirm.isEnabled()
        return answers.pop(0)

    answers = [QDialog.Rejected, QDialog.Accepted]
    monkeypatch.setattr(ui.ConsentDialog, "exec", fake_exec)
    try:
        box_ = dialog.provider_boxes["gofile"]
        box_.click()  # declined
        assert not store.provider_state("gofile").enabled and not box_.isChecked()
        box_.click()  # accepted
        state = store.provider_state("gofile")
        assert state.enabled and state.consented and box_.isChecked()
        assert dialog.provider_reasons["gofile"].text() == "Ready"
        assert shown == [("Share file via link · Gofile", False)] * 2
        # off and on again: no second consent
        box_.click()
        box_.click()
        assert store.provider_state("gofile").enabled and len(shown) == 2
        # pixeldrain needs the user's key
        answers.append(QDialog.Accepted)
        dialog.provider_boxes["pixeldrain"].click()
        assert dialog.provider_reasons["pixeldrain"].text() == "Add your pixeldrain API key"
        dialog.pixeldrain_key.setText("pd-key-value")
        assert dialog.save()
        assert store.pixeldrain_key() == "pd-key-value"
        assert dialog.provider_reasons["pixeldrain"].text() == "Ready"
        assert "pd-key-value" not in Path(store.path).read_text("utf-8")
    finally:
        dialog.close()


def test_the_consent_text_is_the_shared_computer_wording(qapp, google, tmp_path, box):
    cloud, _store, _ = _cloud(google, tmp_path, box)
    consent = cloud.consent_text("gofile")
    dialog = ui.ConsentDialog(consent)
    labels = " ".join(label.text() for label in dialog.findChildren(ui.QLabel))
    assert "from this computer to Gofile" in labels and "mobile data" not in labels
    assert dialog.checkbox.text() == consent.checkbox and not dialog.confirm.isEnabled()
    dialog.close()


def test_sign_in_and_sign_out_from_the_settings_dialog(qapp, google, tmp_path, box, monkeypatch):
    cloud, store, lines = _cloud(google, tmp_path, box, own_client=True)
    opened = []

    def fake_open(url):
        opened.append(url.toString())
        threading.Thread(target=_browser, args=(google, url.toString()), daemon=True).start()
        return True

    monkeypatch.setattr(ui.QDesktopServices, "openUrl", staticmethod(fake_open))
    dialog = ui.LibrarySettingsDialog(None, cloud=cloud)
    try:
        assert dialog.sign_in_button.isEnabled() and not dialog.sign_out_button.isVisible()
        dialog.sign_in_button.click()
        assert opened and opened[0].startswith(google.base + "/auth")
        assert _pump(qapp, lambda: store.drive_settings().signed_in, 20)
        assert _pump(qapp, lambda: dialog._sign_in is None, 10)
        assert "Connected" in dialog.drive_state.text()
        assert store.refresh_token() == "refresh-from-code"
        monkeypatch.setattr(ui.QMessageBox, "question", staticmethod(lambda *a, **k: ui.QMessageBox.Yes))
        dialog.sign_out_button.click()
        assert _pump(qapp, lambda: not store.drive_settings().signed_in and dialog._task is None, 15)
        assert google.revoked == ["refresh-from-code"]
        assert "Not connected" in dialog.drive_state.text()
    finally:
        dialog.close()


def test_closing_the_settings_cancels_a_pending_browser_sign_in(qapp, google, tmp_path, box, monkeypatch):
    cloud, store, _ = _cloud(google, tmp_path, box, own_client=True)
    monkeypatch.setattr(ui.QDesktopServices, "openUrl", staticmethod(lambda url: True))  # the user never signs in
    dialog = ui.LibrarySettingsDialog(None, cloud=cloud)
    dialog.sign_in_button.click()
    assert dialog.sign_in_button.text() == "Cancel sign-in"
    sign_in = dialog._sign_in
    dialog.close()
    assert _pump(qapp, lambda: sign_in.session.closed, 10)
    assert not store.drive_settings().signed_in


# ---------------------------------------------------------------------------------------------------
# Share
# ---------------------------------------------------------------------------------------------------


def test_share_uploads_the_compiled_file_in_the_background_and_shows_the_link(qapp, google, tmp_path, box):
    import share_hosting_gofile as gf

    gofile = FakeGofile()
    try:
        provider = gf.GofileProvider(api_base=gofile.base, upload_url=gofile.base + "/uploadfile")
        cloud, store, lines = _cloud(google, tmp_path, box, providers={"gofile": provider})
        store.set_provider_enabled("gofile", True, consent=True)
        book = _book(tmp_path)
        dialog = ui.ShareDialog(book, None, cloud=cloud)
        try:
            assert dialog.file_combo.count() == 1 and dialog.file_combo.currentData() == book["output_epub_path"]
            assert dialog.site_combo.currentData() == "gofile" and dialog.upload_button.isEnabled()
            assert not dialog.no_site_row.isVisibleTo(dialog) and not dialog.compile_row.isVisibleTo(dialog)
            dialog.upload_button.click()
            assert dialog._upload is not None and not dialog.upload_button.isVisibleTo(dialog)
            assert _pump(qapp, lambda: dialog._upload is None, 15)
            assert dialog.link.text() == "https://gofile.io/d/AbC123" and dialog.link_row.isVisibleTo(dialog)
            assert b"compiled epub bytes" in gofile.bodies[0][2]
            # recorded under the book, listed in the dialog, Copy puts it on the clipboard
            records = cloud.links_for(ui.book_key(book))
            assert records and records[0]["url"] == "https://gofile.io/d/AbC123"
            assert any("Gofile" in label.text() for label in dialog.findChildren(ui.QLabel))
            dialog.copy_button.click()
            assert QApplication.clipboard().text() == "https://gofile.io/d/AbC123"
            assert any("Share link ready" in line for line in lines)
        finally:
            dialog.close()
    finally:
        gofile.close()


def test_books_on_the_library_shelf_never_share_their_links(qapp, google, tmp_path, box, monkeypatch):
    """Organized / imported books without an output folder: links are keyed by the book's own file."""
    shelf = tmp_path / "Library" / "Translated"
    shelf.mkdir(parents=True)
    books = []
    for name in ("Alpha", "Beta"):
        path = shelf / f"{name}.epub"
        path.write_bytes(name.encode())
        books.append({"path": str(path), "name": name, "type": "translated", "in_library": True})
    monkeypatch.setattr(ui, "_resolve_book_output_folder", lambda book: "")
    monkeypatch.setattr(ui, "_resolve_book_translated_file", lambda book: book["path"])
    alpha, beta = (ui.book_key(b) for b in books)
    assert alpha != beta and alpha == lc.book_key_for(books[0]["path"])
    assert lc.book_key_for(str(shelf)) not in (alpha, beta)
    cloud, store, _ = _cloud(google, tmp_path, box)
    store.add_link(alpha, books[0]["path"], "gofile", sh.UploadResult("https://gofile.io/d/A1"))
    assert cloud.links_for(beta) == [] and len(cloud.links_for(alpha)) == 1


def test_share_without_a_site_points_to_library_settings(qapp, google, tmp_path, box):
    cloud, store, _ = _cloud(google, tmp_path, box)
    opened = []
    dialog = ui.ShareDialog(_book(tmp_path), None, cloud=cloud, open_settings=lambda: opened.append(True))
    try:
        assert dialog.site_combo.count() == 0 and dialog.no_site_row.isVisibleTo(dialog)
        assert not dialog.upload_button.isEnabled()
        dialog.settings_button.click()
        assert opened == [True]
        # a site turned on in Library settings shows up at once
        store.set_provider_enabled("send", True, consent=True)
        ui._emit_changed("share")
        assert _pump(qapp, lambda: dialog.site_combo.count() == 1, 5)
        assert dialog.upload_button.isEnabled() and not dialog.no_site_row.isVisibleTo(dialog)
        # Google Drive as a share site needs the sign-in
        store.set_provider_enabled("gdrive", True, consent=True)
        ui._emit_changed("share")
        assert _pump(qapp, lambda: dialog.site_combo.count() == 2, 5)
        dialog.site_combo.setCurrentIndex(dialog.site_combo.findData("gdrive"))
        assert not dialog.upload_button.isEnabled() and "Sign in" in dialog.site_reason.text()
    finally:
        dialog.close()
    assert google.requests == []


def test_share_offers_the_librarys_compile_when_nothing_is_compiled(qapp, google, tmp_path, box):
    cloud, store, _ = _cloud(google, tmp_path, box)
    store.set_provider_enabled("send", True, consent=True)
    book = _book(tmp_path, compiled=False)
    compiled = []

    def compile_cb():
        compiled.append(True)

    dialog = ui.ShareDialog(book, None, cloud=cloud, compile_cb=compile_cb)
    try:
        assert dialog.file_combo.count() == 0 and dialog.compile_row.isVisibleTo(dialog)
        assert dialog.compile_button.isVisibleTo(dialog) and dialog.compile_button.text().endswith("Compile EPUB")
        assert not dialog.upload_button.isEnabled()
        dialog.compile_button.click()
        assert compiled == [True] and "Compiling" in dialog.compile_note.text()
        (Path(book["path"]) / "The Book.epub").write_bytes(b"new")  # the compile finished
        assert _pump(qapp, lambda: dialog.file_combo.count() == 1, 10)
        assert dialog.upload_button.isEnabled() and not dialog.compile_row.isVisibleTo(dialog)
    finally:
        dialog.close()


def test_transferit_handoff_in_the_share_dialog(qapp, google, tmp_path, box, monkeypatch):
    cloud, store, _ = _cloud(google, tmp_path, box)
    store.set_provider_enabled("transferit", True, consent=True)
    opened = []
    monkeypatch.setattr(ui.QDesktopServices, "openUrl", staticmethod(lambda url: opened.append(url.toString())))
    book = _book(tmp_path)
    dialog = ui.ShareDialog(book, None, cloud=cloud)
    try:
        assert dialog.upload_button.text() == "Open transfer.it"
        dialog.upload_button.click()
        assert opened == ["https://transfer.it/start"] and dialog.handoff.isVisibleTo(dialog)
        assert "The Book.epub" in dialog.handoff_hint.text()
        dialog.paste.setText("https://transfer.it/t/AbCdEf12")
        dialog.save_link_button.click()
        assert dialog.link.text() == "https://transfer.it/t/AbCdEf12"
        assert cloud.links_for(ui.book_key(book))[0]["provider"] == "transferit"
    finally:
        dialog.close()


# ---------------------------------------------------------------------------------------------------
# epub_library wiring: header button, card menu, Book Details
# ---------------------------------------------------------------------------------------------------


def _library(qapp):
    dialog = epub_library.EpubLibraryDialog(config={"epub_library_tab": 0})
    dialog._initial_scan_started = True
    dialog._auto_refresh = lambda: None
    dialog.show()
    dialog._auto_refresh_timer.stop()
    dialog._hide_loading()
    qapp.processEvents()
    return dialog


def _menu_actions(library, book, monkeypatch):
    captured = {}

    class GrabMenu(QMenu):
        def exec(self, *args, **kwargs):  # noqa: A003 - QMenu.exec
            captured["menu"] = self
            captured["texts"] = [a.text() for a in self.actions()]

    monkeypatch.setattr(epub_library, "QMenu", GrabMenu)
    library._show_context_menu(book, QPoint(5, 5))
    return captured


def test_library_header_has_the_settings_button(qapp, google, tmp_path, box, monkeypatch):
    cloud, _store, _ = _cloud(google, tmp_path, box)
    monkeypatch.setattr(lc, "_INSTANCE", cloud)
    opened = []
    monkeypatch.setattr(ui, "open_library_settings", lambda parent=None: opened.append(parent))
    library = _library(qapp)
    try:
        button = library._library_settings_btn
        assert "Settings" in button.text() and "Google Drive" in button.toolTip()
        button.click()
        assert opened == [library]
    finally:
        library.close()


def test_card_menu_offers_share_and_sync_to_drive_only_when_sync_is_on(qapp, google, tmp_path, box,
                                                                       monkeypatch):
    cloud, store, _ = _cloud(google, tmp_path, box, own_client=True)
    monkeypatch.setattr(lc, "_INSTANCE", cloud)
    book = _book(tmp_path)
    library = _library(qapp)
    try:
        texts = _menu_actions(library, book, monkeypatch)["texts"]
        assert "\U0001f4e4  Share…" in texts and "☁️  Sync to Drive now" not in texts
        store.save_refresh_token("refresh-1")
        assert "☁️  Sync to Drive now" not in _menu_actions(library, book, monkeypatch)["texts"]
        store.set_drive_options(enabled=True)
        captured = _menu_actions(library, book, monkeypatch)
        assert "☁️  Sync to Drive now" in captured["texts"]
        # Share opens the Share dialog with the Library's own compile
        shared = []
        monkeypatch.setattr(ui, "open_share_dialog",
                            lambda parent, b, compile_cb=None: shared.append((parent, b, compile_cb)))
        action = [a for a in captured["menu"].actions() if a.text().startswith("\U0001f4e4")][0]
        action.trigger()
        assert shared and shared[0][0] is library and shared[0][1] is book and callable(shared[0][2])
        # Sync to Drive now queues the book (full mode: the whole folder) and uploads it
        store.set_drive_options(mode="full")
        monkeypatch.setattr(ui.QToolTip, "showText", staticmethod(lambda *a, **k: None))
        action = [a for a in captured["menu"].actions() if "Sync to Drive now" in a.text()][0]
        action.trigger()
        assert cloud.sync.worker.wait_idle(15)
        tree = google.tree()
        assert tree.get("Glossarion/The Book/The Book.epub") == b"compiled epub bytes"
        assert tree.get("Glossarion/The Book/translation_progress.json") == b"{}"
    finally:
        library.close()


def test_book_details_has_a_share_button(qapp, google, tmp_path, box, monkeypatch):
    cloud, _store, _ = _cloud(google, tmp_path, box)
    monkeypatch.setattr(lc, "_INSTANCE", cloud)
    monkeypatch.setattr(epub_library.BookDetailsDialog, "_start_loading", lambda self: None)
    shared = []
    monkeypatch.setattr(ui, "open_share_dialog",
                        lambda parent, b, compile_cb=None: shared.append((parent, b, compile_cb)))
    book = _book(tmp_path, compiled=False)
    details = epub_library.BookDetailsDialog(dict(book), config={})
    try:
        details._auto_refresh_timer.stop()
        assert "Share" in details._share_btn.text()
        details._share_btn.click()
        assert shared and shared[0][0] is details
        assert shared[0][2] == details._on_compile_epub_clicked
    finally:
        details.close()


# ---------------------------------------------------------------------------------------------------
# the compile hooks: the Library / main-window compile and the end of a translation run
# ---------------------------------------------------------------------------------------------------


class _Owner:
    def __init__(self):
        self.lines = []

    def append_log(self, line):
        self.lines.append(line)


def _compile_paths(monkeypatch):
    """(Library compile hook, translation-run hook) as the app calls them."""
    import translator_gui
    import TransateKRtoEN

    monkeypatch.setattr(translator_gui.QTimer, "singleShot", staticmethod(lambda *a, **k: None))  # no boxes
    owner = _Owner()

    def library_compile(path):
        translator_gui.TranslatorGUI._notify_compile_result(owner, "epub", path=path)

    def translation_run(path):
        TransateKRtoEN._report_compiled_output("epub", path)

    return owner, library_compile, translation_run


def test_both_compile_paths_upload_when_sync_is_on(qapp, google, tmp_path, box, monkeypatch):
    import translator_gui

    cloud, store, lines = _cloud(google, tmp_path, box, own_client=True)
    monkeypatch.setattr(lc, "_INSTANCE", cloud)
    owner, library_compile, translation_run = _compile_paths(monkeypatch)
    translator_gui.TranslatorGUI._install_library_cloud_hook(owner)  # what TranslatorGUI.__init__ does
    assert cloud.log_line == owner.append_log
    store.save_refresh_token("refresh-1")
    store.set_drive_options(enabled=True)
    book = _book(tmp_path)
    epub = book["output_epub_path"]
    library_compile(epub)
    assert cloud.sync.worker.wait_idle(15)
    assert google.tree() == {"Glossarion/": b"", "Glossarion/The Book.epub": b"compiled epub bytes"}
    Path(epub).write_bytes(b"recompiled at the end of a translation run")
    translation_run(epub)
    assert cloud.sync.worker.wait_idle(15)
    # the same Drive file, updated in place
    assert google.tree() == {"Glossarion/": b"",
                             "Glossarion/The Book.epub": b"recompiled at the end of a translation run"}
    assert store.last_sync()["updated"] == 1
    assert any("Drive sync done" in line for line in owner.lines)
    # a failed compile and a non-book file send nothing
    before = len(google.requests)
    translator_gui.TranslatorGUI._notify_compile_result(owner, "epub", error="boom")
    translation_run(str(Path(book["path"]) / "translation_progress.json"))
    assert cloud.sync.worker.wait_idle(5) and len(google.requests) == before


def test_sync_is_off_by_default_and_both_compile_paths_send_nothing(qapp, google, tmp_path, box, monkeypatch):
    import translator_gui

    cloud, store, lines = _cloud(google, tmp_path, box, own_client=True)
    monkeypatch.setattr(lc, "_INSTANCE", cloud)
    owner, library_compile, translation_run = _compile_paths(monkeypatch)
    translator_gui.TranslatorGUI._install_library_cloud_hook(owner)
    epub = _book(tmp_path)["output_epub_path"]
    library_compile(epub)
    translation_run(epub)
    store.save_refresh_token("refresh-1")  # signed in, switch still off
    library_compile(epub)
    translation_run(epub)
    assert cloud.sync.worker.wait_idle(5)
    assert google.requests == [] and owner.lines == [] and store.last_sync() is None
    assert not Path(store.path).exists() or '"enabled": false' in Path(store.path).read_text("utf-8")


def test_the_translation_engine_reports_each_end_of_run_compile():
    """Both EPUB builds (text and image mode) and the PDF output of a translation run report the file."""
    source = (SRC / "TransateKRtoEN.py").read_text(encoding="utf-8")
    assert source.count('_report_compiled_output("epub", compiled_epub)') == 2
    # the multi-section PDF and the single-section shortcut
    assert source.count('_report_compiled_output("pdf", combined_path)') == 2
    gui = (SRC / "translator_gui.py").read_text(encoding="utf-8")
    assert "self._install_library_cloud_hook()" in gui and "library_cloud.reset_instance()" in gui


def test_new_desktop_ui_module_follows_the_epub_library_spec_tiers():
    """library_cloud_ui is Qt + epub_library: in exactly the specs that ship epub_library."""
    import ast

    for spec_path in sorted(SRC.glob("translator*.spec")):
        source = spec_path.read_text(encoding="utf-8")

        def names(block):
            lines = source.splitlines()
            start = next(i for i, line in enumerate(lines) if line.startswith(f"{block} = ["))
            end = next(j for j in range(start + 1, len(lines)) if lines[j].rstrip() == "]")
            values = ast.literal_eval("\n".join(lines[start:end + 1]).split("=", 1)[1].strip())
            return [Path(v[0]).stem if isinstance(v, tuple) else v for v in values]

        for block in ("app_files", "app_modules"):
            listed = names(block)
            assert listed.count("library_cloud_ui") == listed.count("epub_library"), (spec_path.name, block)
    data = (SRC / "library_cloud_ui.py").read_bytes()
    ast.parse(data.decode("utf-8"), feature_version=(3, 10))
    assert data.count(b"\r\n") in (0, data.count(b"\n"))
