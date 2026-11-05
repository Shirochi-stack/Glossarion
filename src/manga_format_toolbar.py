# manga_format_toolbar.py
"""Word-style Format toolbar of the manga editor's image preview (per-box text style).

A row at the top of ``MangaImagePreviewWidget`` (shown while Enable Manual Editing is on): font,
size (Auto or a fixed px value), Bold / Italic / Underline, align left / center / right, text
colour and Reset to default. It acts on the selected box(es), shows the first selected box's
values and is disabled while no box is selected.

A change is stored per box in the page's editor state (``region_styles``, through
``manga_editor_core._update_region_styles``) and the page is re-rendered with the preview's own
Save & Update Overlay (``_on_save_overlay_clicked``: render only, on the cleaned image when there
is one). ``MangaTranslator.render_translated_text`` draws the style; a box without one renders
with the Rendering Settings as before.

The widget class itself stays as it is (tests/test_manga_editor_core.py pins it): the toolbar is
installed from outside by ``install_format_toolbar(preview, tab)``.
"""

import os

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import QColorDialog, QComboBox, QFrame, QHBoxLayout, QLabel, QSpinBox

import manga_editor_core

#: Debounce of the re-render after a change (a size typed digit by digit renders once).
RENDER_DELAY_MS = 350
DEFAULT_FONT_LABEL = "Default font"
#: Fixed size range (px); 0 is Auto (fit the box, the Rendering Settings sizing).
SIZE_MAX = 200
ALIGNMENTS = (("left", "Align left"), ("center", "Center"), ("right", "Align right"))

_COMBO_STYLE = """
    QComboBox, QSpinBox {
        background-color: #2d2d2d;
        color: white;
        border: 1px solid #3a3a3a;
        border-radius: 3px;
        padding: 2px 4px;
        min-height: 24px;
    }
    QComboBox:disabled, QSpinBox:disabled {
        background-color: #1a1a1a;
        color: #555555;
        border-color: #2a2a2a;
    }
"""


def _alignment_icon(align, size=18):
    """Word's alignment glyph: four lines flush left / centered / flush right."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setPen(QPen(QColor("white"), 2))
    for row, length in enumerate((14, 9, 14, 9)):
        y = 3 + row * 4
        if align == "left":
            x0 = 2
        elif align == "right":
            x0 = size - 2 - length
        else:
            x0 = (size - length) // 2
        painter.drawLine(x0, y, x0 + length, y)
    painter.end()
    return QIcon(pixmap)


def _color_icon(rgb, size=18):
    """The colour button's "A" over a bar of the box's text colour (empty bar: Rendering Settings)."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setPen(QPen(QColor("white"), 1))
    font = painter.font()
    font.setBold(True)
    font.setPixelSize(12)
    painter.setFont(font)
    painter.drawText(0, 0, size, size - 4, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, "A")
    if rgb:
        painter.fillRect(1, size - 4, size - 2, 4, QColor(*rgb))
    else:
        painter.setPen(QPen(QColor("#888888"), 1))
        painter.drawRect(1, size - 4, size - 3, 3)
    painter.end()
    return QIcon(pixmap)


class MangaFormatToolbar(QObject):
    """The Format row: ``frame`` is the widget, ``indexes()`` the selected boxes' region indexes."""

    def __init__(self, preview, tab):
        super().__init__(preview)
        self.preview = preview
        self.tab = tab
        self._updating = False
        self._pending = False
        self._tab_fonts = 0
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(RENDER_DELAY_MS)
        self._render_timer.timeout.connect(self._render)

        # The preview's own tool frame and tool buttons (same look as the Manual Tools rows)
        self.frame = preview._create_tool_frame("Format")
        layout = QHBoxLayout(self.frame)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)

        self.font_combo = QComboBox()
        self.font_combo.setToolTip("Font of the selected box (Default font: the Rendering Settings font)")
        self.font_combo.setStyleSheet(_COMBO_STYLE)
        self.font_combo.setMinimumWidth(140)
        self.font_combo.setMaxVisibleItems(20)
        self.font_combo.wheelEvent = lambda event: None
        self.font_combo.currentIndexChanged.connect(self._on_font_changed)
        layout.addWidget(self.font_combo, 1)

        self.size_spin = QSpinBox()
        self.size_spin.setRange(0, SIZE_MAX)
        self.size_spin.setSpecialValueText("Auto")
        self.size_spin.setSuffix(" px")
        self.size_spin.setToolTip("Font size of the selected box (Auto: fit the box like the Rendering Settings)")
        self.size_spin.setStyleSheet(_COMBO_STYLE)
        self.size_spin.setFixedWidth(76)
        self.size_spin.setKeyboardTracking(False)
        self.size_spin.wheelEvent = lambda event: None
        self.size_spin.valueChanged.connect(self._on_size_changed)
        layout.addWidget(self.size_spin)

        self.bold_btn = self._toggle("B", "Bold (Ctrl+B)", "bold")
        self.italic_btn = self._toggle("I", "Italic (Ctrl+I)", "italic")
        self.underline_btn = self._toggle("U", "Underline (Ctrl+U)", "underline")
        font = self.bold_btn.font()
        font.setBold(True)
        self.bold_btn.setFont(font)
        font = self.italic_btn.font()
        font.setItalic(True)
        self.italic_btn.setFont(font)
        font = self.underline_btn.font()
        font.setUnderline(True)
        self.underline_btn.setFont(font)
        for button in (self.bold_btn, self.italic_btn, self.underline_btn):
            layout.addWidget(button)
        layout.addWidget(self._separator())

        self.align_buttons = {}
        for align, tip in ALIGNMENTS:
            button = preview._create_tool_button("", tip)
            button.setIcon(_alignment_icon(align))
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, a=align: self._on_align(a))
            self.align_buttons[align] = button
            layout.addWidget(button)
        layout.addWidget(self._separator())

        self.color_btn = preview._create_tool_button("", "Text colour of the selected box")
        self.color_btn.setIcon(_color_icon(None))
        self.color_btn.clicked.connect(self._on_color)
        layout.addWidget(self.color_btn)
        self.reset_btn = preview._create_tool_button("⟲", "Reset to default (the Rendering Settings)")
        self.reset_btn.clicked.connect(self._on_reset)
        layout.addWidget(self.reset_btn)

        self.selection_label = QLabel("")
        self.selection_label.setStyleSheet("color: #9aa4b2; border: none; padding-left: 4px;")
        layout.addWidget(self.selection_label)

        self.controls = [self.font_combo, self.size_spin, self.bold_btn, self.italic_btn, self.underline_btn,
                         *self.align_buttons.values(), self.color_btn, self.reset_btn]
        self._shortcuts = []
        for keys, button in (("Ctrl+B", self.bold_btn), ("Ctrl+I", self.italic_btn), ("Ctrl+U", self.underline_btn)):
            shortcut = QShortcut(QKeySequence(keys), preview)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda b=button: b.isEnabled() and b.isVisible() and b.click())
            self._shortcuts.append(shortcut)

        self.reload_fonts()
        self._connect_preview()
        self.set_visible_for_editing(getattr(preview, 'manual_editing_enabled', False))
        self.refresh()

    # ---- building blocks -----------------------------------------------------------------------------

    def _toggle(self, text, tip, key):
        button = self.preview._create_tool_button(text, tip)
        button.setCheckable(True)
        button.clicked.connect(lambda checked, k=key: self._apply({k: bool(checked)}))
        return button

    @staticmethod
    def _separator():
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setStyleSheet("color: #3a3a3a; border: none; background-color: #3a3a3a;")
        line.setFixedWidth(1)
        return line

    def _connect_preview(self):
        viewer = self.preview.viewer
        try:
            viewer.scene().selectionChanged.connect(self.refresh)
        except Exception:
            pass
        for signal in ('rectangle_selected', 'rectangle_deleted', 'rectangle_created', 'image_loaded'):
            try:
                getattr(viewer, signal).connect(lambda *_args: self.refresh())
            except Exception:
                pass
        toggle = getattr(self.preview, 'manual_editing_toggle', None)
        if toggle is not None:
            toggle.stateChanged.connect(
                lambda state: self.set_visible_for_editing(state == Qt.CheckState.Checked.value))
        clear_btn = getattr(self.preview, 'clear_boxes_btn', None)
        if clear_btn is not None:
            # after the preview's own Clear Boxes handler (connected first): the boxes are gone
            clear_btn.clicked.connect(self._on_boxes_cleared)

    # ---- fonts ---------------------------------------------------------------------------------------

    def _tab_font_count(self):
        """How many entries the Rendering Settings font combo has (0: not built yet)."""
        combo = getattr(self.tab, 'font_combo', None)
        return combo.count() if combo is not None and getattr(self.tab, 'font_mapping', None) else 0

    def _font_choices(self):
        """(name, path) of the Rendering Settings font list (the tab's combo and its font mapping)."""
        choices = []
        combo = getattr(self.tab, 'font_combo', None)
        mapping = getattr(self.tab, 'font_mapping', None) or {}
        self._tab_fonts = self._tab_font_count()
        if self._tab_fonts:
            for i in range(combo.count()):
                name = combo.itemText(i)
                path = mapping.get(name)
                if path:
                    choices.append((name, path))
        if not choices:
            import manga_settings_defaults
            config = getattr(getattr(self.tab, 'main_gui', None), 'config', None) or {}
            choices = [(os.path.splitext(os.path.basename(p))[0], p)
                       for p in manga_settings_defaults.available_fonts(config)]
        return choices

    def reload_fonts(self):
        self._updating = True
        try:
            current = self.font_combo.currentData()
            self.font_combo.clear()
            self.font_combo.addItem(DEFAULT_FONT_LABEL, None)
            for name, path in self._font_choices():
                self.font_combo.addItem(name, path)
            self._select_font(current)
        finally:
            self._updating = False

    def _select_font(self, path):
        if not path:
            self.font_combo.setCurrentIndex(0)
            return
        wanted = os.path.normcase(os.path.abspath(path))
        for i in range(1, self.font_combo.count()):
            data = self.font_combo.itemData(i)
            if data and os.path.normcase(os.path.abspath(data)) == wanted:
                self.font_combo.setCurrentIndex(i)
                return
        self.font_combo.addItem(os.path.splitext(os.path.basename(path))[0], path)
        self.font_combo.setCurrentIndex(self.font_combo.count() - 1)

    # ---- selection -----------------------------------------------------------------------------------

    def image_path(self):
        return getattr(self.preview, 'current_image_path', None)

    def indexes(self):
        """Region indexes of the selected boxes (Qt selection, else the viewer's selected box)."""
        viewer = self.preview.viewer
        rectangles = list(getattr(viewer, 'rectangles', []) or [])
        selected = []
        try:
            selected = [item for item in viewer.scene().selectedItems() if item in rectangles]
        except Exception:
            selected = []
        if not selected and getattr(viewer, 'selected_rect', None) in rectangles:
            selected = [viewer.selected_rect]
        out = []
        for item in selected:
            index = getattr(item, 'region_index', None)
            if not isinstance(index, int):
                index = rectangles.index(item)
            if index not in out:
                out.append(index)
        return sorted(out)

    def styles(self):
        return manga_editor_core._region_text_styles(self.tab, self.image_path())

    def refresh(self, *_args):
        """Show the first selected box's style; disable everything while no box is selected."""
        if self._tab_font_count() != self._tab_fonts:
            self.reload_fonts()  # the Rendering Settings font list was built (or a font was added)
        indexes = self.indexes() if self.image_path() else []
        enabled = bool(indexes)
        for control in self.controls:
            control.setEnabled(enabled)
        style = self.styles().get(indexes[0], {}) if indexes else {}
        self._updating = True
        try:
            self._select_font(style.get('font'))
            self.size_spin.setValue(int(style.get('font_size') or 0))
            self.bold_btn.setChecked(bool(style.get('bold')))
            self.italic_btn.setChecked(bool(style.get('italic')))
            self.underline_btn.setChecked(bool(style.get('underline')))
            align = style.get('align') or 'center'
            for name, button in self.align_buttons.items():
                button.setChecked(enabled and name == align)
            self.color_btn.setIcon(_color_icon(style.get('color')))
        finally:
            self._updating = False
        if not indexes:
            self.selection_label.setText("Select a box to format it")
        elif len(indexes) == 1:
            self.selection_label.setText(f"Box {indexes[0] + 1}")
        else:
            self.selection_label.setText(f"{len(indexes)} boxes")

    def set_visible_for_editing(self, enabled):
        self.frame.setVisible(bool(enabled))
        if enabled:
            self.refresh()

    # ---- changes -------------------------------------------------------------------------------------

    def _apply(self, changes=None, *, reset=False):
        if self._updating:
            return None
        indexes = self.indexes()
        if not indexes:
            return None
        stored = manga_editor_core._update_region_styles(self.tab, self.image_path(), indexes, changes, reset=reset)
        self.refresh()
        self.schedule_render()
        return stored

    def _on_font_changed(self, _index):
        self._apply({'font': self.font_combo.currentData()})

    def _on_size_changed(self, value):
        self._apply({'font_size': int(value) or None})

    def _on_align(self, align):
        self._apply({'align': None if align == 'center' else align})

    def _on_color(self):
        indexes = self.indexes()
        if not indexes:
            return None
        current = self.styles().get(indexes[0], {}).get('color')
        if current:
            initial = QColor(*current)
        else:
            initial = QColor(getattr(self.tab, 'text_color_r_value', 0), getattr(self.tab, 'text_color_g_value', 0),
                             getattr(self.tab, 'text_color_b_value', 0))
        color = QColorDialog.getColor(initial, self.preview, "Choose Text Colour")
        if not color.isValid():
            return None
        return self._apply({'color': [color.red(), color.green(), color.blue()]})

    def _on_reset(self):
        return self._apply(reset=True)

    def _on_boxes_cleared(self):
        manga_editor_core._clear_region_styles(self.tab, self.image_path())
        self.refresh()

    # ---- render --------------------------------------------------------------------------------------

    def schedule_render(self):
        """Re-render the page once the changes settle (Save & Update Overlay: render only)."""
        self._pending = True
        self._render_timer.start()

    def _render(self):
        if not self._pending:
            return
        self._pending = False
        handler = getattr(self.preview, '_on_save_overlay_clicked', None)
        if callable(handler) and self.image_path():
            handler()


def install_format_toolbar(preview, tab):
    """Put the Format row at the top of the preview (under its title row) and return the toolbar."""
    toolbar = MangaFormatToolbar(preview, tab)
    layout = preview.layout()
    layout.insertWidget(1 if layout.count() > 1 else layout.count(), toolbar.frame)
    preview.format_toolbar = toolbar
    return toolbar
