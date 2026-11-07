"""The Delete key deletes the selected box in the manga editor's preview (owner, 2026-11-08: "please
allow the keyboard shortcut del to delete text boxes here").

Del runs the 🗑 Delete Selected button's action (``viewer.delete_selected_rectangle``) while Manual
Editing is on; with it off, or while a box's text is being typed into, Del does nothing here.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("GLOSSARION_HTTP_LOG", "0")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor, QPixmap  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def preview(qapp, monkeypatch):
    import manga_image_preview

    widget = manga_image_preview.MangaImagePreviewWidget(main_gui=types.SimpleNamespace(config={}))
    widget.resize(900, 700)
    pixmap = QPixmap(400, 400)
    pixmap.fill(QColor(255, 255, 255))
    widget.viewer.setPhoto(pixmap)
    widget.show()
    qapp.processEvents()
    deleted = []
    monkeypatch.setattr(widget.viewer, "delete_selected_rectangle", lambda: deleted.append(True))
    yield widget, deleted
    widget.close()


def test_delete_key_deletes_the_selected_box_while_manual_editing(qapp, preview):
    widget, deleted = preview
    widget.manual_editing_enabled = True
    QTest.keyClick(widget.viewer, Qt.Key.Key_Delete)
    qapp.processEvents()
    assert deleted == [True]


def test_delete_key_does_nothing_with_manual_editing_off(qapp, preview):
    widget, deleted = preview
    widget.manual_editing_enabled = False
    QTest.keyClick(widget.viewer, Qt.Key.Key_Delete)
    qapp.processEvents()
    assert deleted == []


def test_delete_key_keeps_typing_in_a_box(qapp, preview):
    widget, deleted = preview
    widget.manual_editing_enabled = True
    text = QtWidgets.QGraphicsTextItem("abc")
    text.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
    widget.viewer.scene().addItem(text)
    widget.viewer.scene().setFocusItem(text)
    QTest.keyClick(widget.viewer, Qt.Key.Key_Delete)
    qapp.processEvents()
    assert deleted == []
