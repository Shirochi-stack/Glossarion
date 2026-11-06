# manga_format_toolbar.py
"""Word-style Format toolbar of the manga editor's image preview (per-box text style).

A row at the top of ``MangaImagePreviewWidget`` (shown while Enable Manual Editing is on): font,
size (Auto or a fixed px value in 0.5 px steps), Bold / Italic / Underline, align left / center /
right, text colour and Reset to default. It acts on the selected box(es), shows the first selected
box's values and is disabled while no box is selected. A box on Auto shows the size it was last
rendered at (the page state's ``region_font_sizes``, "28 px (auto)"); stepping up / down from there
gives the box that size ± 0.5 px.

A change is stored per box in the page's editor state (``region_styles``, through
``manga_editor_core._update_region_styles``) and the page is re-rendered with the preview's own
Save & Update Overlay (``_on_save_overlay_clicked``: render only, on the cleaned image when there
is one). ``MangaTranslator.render_translated_text`` draws the style; a box without one renders
with the Rendering Settings as before.

The widget class itself stays as it is (tests/test_manga_editor_core.py pins it): the toolbar is
installed from outside by ``install_format_toolbar(preview, tab)``. A box's right-click menu gets the
same changes as a "Text format" submenu (``MangaFormatToolbar.add_context_menu``), with a
"−  28 px (auto)  +" row that stays open while the size is stepped.

``install_format_toolbar`` also keeps Enter from pressing a push button of the Manga Panel
Translator window (``guard_enter_key``): a QDialog's push buttons are auto-default, so Enter in a
spin box or line edit clicked "Add Files" and opened the file explorer. It also installs the
preview's Ctrl zoom (``manga_preview_zoom.install_preview_zoom``).
"""

import os

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QColorDialog, QComboBox, QDialog, QDoubleSpinBox, QFrame, QHBoxLayout,
                               QInputDialog, QLabel, QPushButton, QSizePolicy, QWidget, QWidgetAction)

import manga_editor_core
import manga_ocr_io

#: Debounce of the re-render after a change (a size typed digit by digit renders once).
RENDER_DELAY_MS = 350
DEFAULT_FONT_LABEL = "Default font"
#: Fixed size range (px); 0 is Auto (fit the box, the Rendering Settings sizing).
SIZE_MAX = 200
#: A fixed size goes up / down in half px (the spin box arrows, the menu's − / +).
SIZE_STEP = 0.5
SIZE_SUFFIX = " px"
#: The size box's suffix while it shows the size a box on Auto was rendered at.
AUTO_SIZE_SUFFIX = " px (auto)"
#: Disabled look of the native size box: the greyed text of the row's other disabled controls. A
#: colour-only rule keeps the native drawing (box-model rules break a spin box's arrows).
_SIZE_SPIN_STYLE = "QDoubleSpinBox:disabled { color: #555555; }"
ALIGNMENTS = (("left", "Align left"), ("center", "Center"), ("right", "Align right"))
SIZE_TOOLTIP = "Font size of the selected box in px (Auto: fit the box like the Rendering Settings)"

_COMBO_STYLE = """
    QComboBox {
        background-color: #2d2d2d;
        color: white;
        border: 1px solid #3a3a3a;
        border-radius: 3px;
        padding: 2px 4px;
        min-height: 24px;
    }
    QComboBox:disabled {
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


def size_text(size):
    """28 -> "28", 28.5 -> "28.5" (no trailing ".0")."""
    text = f"{float(size):.1f}"
    return text[:-2] if text.endswith(".0") else text


class SizeSpinBox(QDoubleSpinBox):
    """The size box: a native QDoubleSpinBox (no box-model style sheet, like Other Settings' GUI
    Font Scale) in 0.5 px steps that shows "28 px" / "28.5 px" (no trailing ".0"); 0 is "Auto"."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDecimals(1)
        self.setRange(0, SIZE_MAX)
        self.setSingleStep(SIZE_STEP)
        self.setSpecialValueText("Auto")
        self.setSuffix(SIZE_SUFFIX)

    def textFromValue(self, value):
        text = self.locale().toString(float(value), 'f', 1)
        zero = self.locale().decimalPoint() + "0"
        return text[:-len(zero)] if text.endswith(zero) else text


class _EnterGuard(QObject):
    """Enter / Return never presses a push button of ``dialog`` (the Manga Panel Translator window).

    A QDialog's push buttons are auto-default: the first one (Add Files) became the default button
    and Enter in a spin box or line edit clicked it, as did Enter on a button that had focus. Every
    push button of the window is made a plain button (now, once the window is built, when it is
    shown, and when a button made later gets focus), and an Enter the window itself receives (one a
    field did not take) is dropped. Enter still commits a spin box / line edit; Space still presses
    the focused button."""

    def __init__(self, dialog):
        super().__init__(dialog)
        self.dialog = dialog
        dialog.installEventFilter(self)
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._on_focus_changed)
        self.disarm()

    @staticmethod
    def _plain(button):
        if button.autoDefault() or button.isDefault():
            button.setDefault(False)
            button.setAutoDefault(False)

    def disarm(self):
        try:
            for button in self.dialog.findChildren(QPushButton):
                # only this window's buttons: a dialog parented to it (Manga Settings' default Save)
                # keeps its own default button
                if button.window() is self.dialog:
                    self._plain(button)
        except RuntimeError:
            pass  # the window is gone

    def _on_focus_changed(self, _old, new):
        try:
            if isinstance(new, QPushButton) and new.window() is self.dialog:
                self._plain(new)
        except RuntimeError:
            pass

    def eventFilter(self, obj, event):
        if obj is self.dialog:
            kind = event.type()
            if kind == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                event.accept()
                return True
            if kind == QEvent.Type.Show:
                self.disarm()
        return False


def guard_enter_key(dialog):
    """Keep Enter from pressing a push button of ``dialog`` (see ``_EnterGuard``); one guard per
    window. Returns the guard (None without a QDialog)."""
    if not isinstance(dialog, QDialog):
        return None
    guard = getattr(dialog, '_glossarion_enter_guard', None)
    if guard is None:
        guard = _EnterGuard(dialog)
        dialog._glossarion_enter_guard = guard
    else:
        guard.disarm()
    return guard


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
        self._pending = None  # the page whose re-render is waiting for the debounce
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

        self._auto_size = None  # the rendered size the size box shows for a box on Auto
        self.size_spin = SizeSpinBox()
        self.size_spin.setToolTip(SIZE_TOOLTIP)
        # The native spin box, like Other Settings' GUI Font Scale: no box-model style sheet (a
        # spin box rule without ::up-button / ::down-button rules draws broken arrows over the
        # text), only the disabled text colour of the row's other controls
        self.size_spin.setStyleSheet(_SIZE_SPIN_STYLE)
        self.size_spin.setSuffix(AUTO_SIZE_SUFFIX)  # wide enough for "200.5 px (auto)"
        self.size_spin.setFixedWidth(max(84, self.size_spin.sizeHint().width()))
        self.size_spin.setSuffix(SIZE_SUFFIX)
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
        self.color_btn.clicked.connect(lambda _checked=False: self._on_color())
        layout.addWidget(self.color_btn)
        self.reset_btn = preview._create_tool_button("⟲", "Reset to default (the Rendering Settings)")
        self.reset_btn.clicked.connect(self._on_reset)
        layout.addWidget(self.reset_btn)

        self.selection_label = QLabel("")
        self.selection_label.setStyleSheet("color: #9aa4b2; border: none; padding-left: 4px;")
        layout.addWidget(self.selection_label)

        self.controls = [self.font_combo, self.size_spin, self.bold_btn, self.italic_btn, self.underline_btn,
                         *self.align_buttons.values(), self.color_btn, self.reset_btn]
        # One height for the row: the font and size boxes take the tool buttons' height (the
        # native spin box was a third shorter than the buttons beside it)
        for box in (self.font_combo, self.size_spin):
            box.setSizePolicy(box.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Preferred)
            box.setMaximumHeight(self.bold_btn.maximumHeight())
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

    def rendered_sizes(self):
        """The size each box of the page was last rendered at ({} before a render)."""
        return manga_editor_core._region_font_sizes(self.tab, self.image_path())

    def box_size(self, index, styles=None, rendered=None):
        """(size, auto) of box ``index``: its own size (auto False), else the size it was rendered
        at on Auto (auto True; None before a render)."""
        style = (self.styles() if styles is None else styles).get(index, {})
        if style.get('font_size'):
            return style['font_size'], False
        size = (self.rendered_sizes() if rendered is None else rendered).get(index)
        if size is None:
            return None, True
        return manga_ocr_io.normalize_font_size(size), True  # shown (and stepped) in 0.5 px

    def refresh(self, *_args):
        """Show the first selected box's style; disable everything while no box is selected."""
        if self._tab_font_count() != self._tab_fonts:
            self.reload_fonts()  # the Rendering Settings font list was built (or a font was added)
        indexes = self.indexes() if self.image_path() else []
        enabled = bool(indexes)
        for control in self.controls:
            control.setEnabled(enabled)
        styles = self.styles() if indexes else {}
        style = styles.get(indexes[0], {}) if indexes else {}
        size, auto = self.box_size(indexes[0], styles) if indexes else (None, True)
        self._updating = True
        try:
            self._select_font(style.get('font'))
            self._show_size(size, auto)
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

    def _show_size(self, size, auto):
        """The size box: a box's own size ("28.5 px"), the size a box on Auto was rendered at
        ("28 px (auto)"), or "Auto" (no box, or not rendered yet)."""
        self._auto_size = size if auto and size else None
        self.size_spin.setSuffix(AUTO_SIZE_SUFFIX if self._auto_size else SIZE_SUFFIX)
        self.size_spin.setValue(float(size or 0))
        font = self.size_spin.font()
        font.setItalic(bool(self._auto_size))
        self.size_spin.setFont(font)
        if self._auto_size:
            self.size_spin.setToolTip(f"Auto: the box was rendered at {size_text(size)} px. The arrows (or a "
                                      f"typed size) give it a fixed size; Reset or 0 goes back to Auto.")
        else:
            self.size_spin.setToolTip(SIZE_TOOLTIP)

    def set_visible_for_editing(self, enabled):
        self.frame.setVisible(bool(enabled))
        if enabled:
            self.refresh()

    # ---- changes -------------------------------------------------------------------------------------

    def _blocked(self):
        """A batch is running (the viewer blocks box edits then): no style change, no re-render."""
        check = getattr(self.preview.viewer, '_is_processing_blocked', None)
        try:
            return bool(check()) if callable(check) else False
        except Exception:
            return False

    def _apply(self, changes=None, *, reset=False, indexes=None):
        if self._updating:
            return None
        if self._blocked():
            self.refresh()  # put the buttons back to the box's stored style
            return None
        indexes = self.indexes() if indexes is None else list(indexes)
        if not indexes:
            return None
        stored = manga_editor_core._update_region_styles(self.tab, self.image_path(), indexes, changes, reset=reset)
        self.refresh()
        self.schedule_render()
        return stored

    def _on_font_changed(self, _index):
        self._apply({'font': self.font_combo.currentData()})

    def _on_size_changed(self, value):
        # From "28 px (auto)" the arrows give 28.5 / 27.5; 0 ("Auto") clears the box's own size
        self._apply({'font_size': manga_ocr_io.normalize_font_size(value)})

    def step_size(self, indexes, delta, base=None):
        """Box sizes one step up / down (± 0.5 px) from box ``base``'s shown size (its own size, or the
        size it was rendered at on Auto). Returns the stored styles (None: no size to step from)."""
        indexes = list(indexes)
        if not indexes:
            return None
        size, _auto = self.box_size(indexes[0] if base is None else base)
        if size is None:
            return None
        new = min(SIZE_MAX, max(SIZE_STEP, float(size) + delta))
        return self._apply({'font_size': manga_ocr_io.normalize_font_size(new)}, indexes=indexes)

    def _on_align(self, align):
        self._apply({'align': None if align == 'center' else align})

    def _on_color(self, indexes=None):
        indexes = self.indexes() if indexes is None else list(indexes)
        if not indexes or self._blocked():
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
        return self._apply({'color': [color.red(), color.green(), color.blue()]}, indexes=indexes)

    def _on_reset(self):
        return self._apply(reset=True)

    def _on_size_dialog(self, indexes, base=None):
        size, _auto = self.box_size(indexes[0] if base is None else base)
        value, ok = QInputDialog.getDouble(self.preview, "Font Size", "Font size in px (0: Auto, fit the box):",
                                           float(size or 0), 0, SIZE_MAX, 1, Qt.WindowType.Dialog, SIZE_STEP)
        if ok:
            self._apply({'font_size': manga_ocr_io.normalize_font_size(value)}, indexes=indexes)

    def _size_label(self, index):
        size, auto = self.box_size(index)
        if size is None:
            return "Auto"
        return f"{size_text(size)}{AUTO_SIZE_SUFFIX if auto else SIZE_SUFFIX}"

    def _size_menu_row(self, menu, index, targets):
        """The menu's "Font size  −  28 px (auto)  +" row: − / + step the size by 0.5 px and the menu
        stays open (the label follows)."""
        row = QWidget(menu)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(4)
        title = QLabel("Font size")
        layout.addWidget(title)
        layout.addStretch(1)
        minus = self.preview._create_tool_button("A−", "Decrease font size (0.5 px)")
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumWidth(label.fontMetrics().horizontalAdvance("200.5 px (auto)") + 8)
        plus = self.preview._create_tool_button("A+", "Increase font size (0.5 px)")
        for widget in (minus, label, plus):
            layout.addWidget(widget)
        action = QWidgetAction(menu)
        action.setDefaultWidget(row)
        action.on_change = []  # the menu's other size entries follow a step

        def show():
            text = self._size_label(index)
            label.setText(text)
            action.setText(f"Font size: {text}")
            stepping = sub_enabled and self.box_size(index)[0] is not None
            minus.setEnabled(stepping)
            plus.setEnabled(stepping)
            for update in action.on_change:
                update()

        def step(delta):
            self.step_size(targets, delta, base=index)
            show()

        sub_enabled = bool(self.image_path()) and not self._blocked()
        minus.clicked.connect(lambda _checked=False: step(-SIZE_STEP))
        plus.clicked.connect(lambda _checked=False: step(SIZE_STEP))
        action.step = step  # (tests) the row's buttons
        action.minus_button, action.plus_button, action.size_label = minus, plus, label
        show()
        return action

    # ---- right-click menu ----------------------------------------------------------------------------

    def add_context_menu(self, menu, region_index):
        """Add the "Text format" submenu to a box's right-click menu (the same changes as the row).

        It acts on the selected boxes when the right-clicked box is one of them, else on that box."""
        selected = self.indexes() if self.image_path() else []
        targets = selected if region_index in selected else [region_index]
        style = self.styles().get(region_index, {})
        label = "🔤 Text format" if len(targets) == 1 else f"🔤 Text format ({len(targets)} boxes)"
        sub = menu.addMenu(label)
        sub.setEnabled(bool(self.image_path()) and not self._blocked())

        for key, text in (('bold', "Bold"), ('italic', "Italic"), ('underline', "Underline")):
            action = sub.addAction(text)
            action.setCheckable(True)
            action.setChecked(bool(style.get(key)))
            action.triggered.connect(lambda checked, k=key: self._apply({k: bool(checked)}, indexes=targets))
        sub.addSeparator()

        current_align = style.get('align') or 'center'
        for align, text in ALIGNMENTS:
            action = sub.addAction(_alignment_icon(align), text)
            action.setCheckable(True)
            action.setChecked(align == current_align)
            action.triggered.connect(lambda _checked=False, a=align: self._apply(
                {'align': None if a == 'center' else a}, indexes=targets))
        sub.addSeparator()

        size_row = self._size_menu_row(sub, region_index, targets)
        sub.addAction(size_row)
        size_action = sub.addAction("Font size…")
        size_action.triggered.connect(lambda: self._on_size_dialog(targets, base=region_index))
        auto_action = sub.addAction("Auto size (fit the box)")
        auto_action.setCheckable(True)
        auto_action.setChecked(not style.get('font_size'))
        auto_action.triggered.connect(lambda _checked=False: self._apply({'font_size': None}, indexes=targets))
        size_row.on_change.append(
            lambda: auto_action.setChecked(not self.styles().get(region_index, {}).get('font_size')))

        font_menu = sub.addMenu("Font")
        wanted = os.path.normcase(os.path.abspath(style['font'])) if style.get('font') else None
        for i in range(self.font_combo.count()):
            path = self.font_combo.itemData(i)
            action = font_menu.addAction(self.font_combo.itemText(i))
            action.setCheckable(True)
            action.setChecked(os.path.normcase(os.path.abspath(path)) == wanted if path else wanted is None)
            action.triggered.connect(lambda _checked=False, f=path: self._apply({'font': f}, indexes=targets))

        color_action = sub.addAction(_color_icon(style.get('color')), "Text colour…")
        color_action.triggered.connect(lambda: self._on_color(targets))
        sub.addSeparator()
        reset_action = sub.addAction("⟲ Reset to default")
        reset_action.triggered.connect(lambda: self._apply(reset=True, indexes=targets))
        return sub

    def _on_boxes_cleared(self):
        manga_editor_core._clear_region_styles(self.tab, self.image_path())
        self.refresh()

    # ---- render --------------------------------------------------------------------------------------

    def schedule_render(self):
        """Re-render the page once the changes settle (Save & Update Overlay: render only)."""
        page = self.image_path()
        if self._pending and self._pending != page:
            self._render_left_page(self._pending)  # a change on another page is still waiting
        self._pending = page
        self._render_timer.start()

    def _render(self):
        page, self._pending = self._pending, None
        if not page:
            return
        if page != self.image_path():
            # the page was left before its re-render: Save & Update Overlay would render the new one
            self._render_left_page(page)
            return
        handler = getattr(self.preview, '_on_save_overlay_clicked', None)
        if callable(handler):
            handler()

    def _render_left_page(self, page):
        """Render a page that is no longer shown from its saved state (render only, on the executor)."""
        try:
            import ImageRenderer
            render = ImageRenderer.render_persisted_translation_state
        except Exception as error:
            print(f"[FORMAT] Re-render of {os.path.basename(page)} skipped: {error}")
            return
        executor = getattr(getattr(self.preview, 'main_gui', None), 'executor', None)
        if executor is not None:
            executor.submit(render, self.tab, page, False)
        else:
            render(self.tab, page, refresh_preview=False)


def install_format_toolbar(preview, tab):
    """Put the Format row at the top of the preview (under its title row) and return the toolbar.

    Also guards the manga window's Enter key (``guard_enter_key``): now for the buttons built so
    far, and again once the window's build has finished. The preview's Ctrl zoom (Ctrl + / - / 0,
    Ctrl+wheel, pinch: ``manga_preview_zoom``) is installed here too, before the row."""
    try:
        from manga_preview_zoom import install_preview_zoom
        install_preview_zoom(preview, getattr(tab, 'dialog', None))
    except Exception as zoom_err:
        print(f"[ZOOM] Preview zoom keys unavailable: {zoom_err}")
    toolbar = MangaFormatToolbar(preview, tab)
    layout = preview.layout()
    layout.insertWidget(1 if layout.count() > 1 else layout.count(), toolbar.frame)
    preview.format_toolbar = toolbar
    try:
        # A render finished (the preview then reloads the page: the viewer's image_loaded, connected
        # by the toolbar): a box on Auto shows the size it was just rendered at
        # (the bound slot: a toolbar that is gone is disconnected, the main window outlives it)
        tab.main_gui.refresh_preview_signal.connect(toolbar.refresh)
    except Exception:
        pass
    guard = guard_enter_key(getattr(tab, 'dialog', None))
    if guard is not None:
        QTimer.singleShot(0, guard.disarm)
    return toolbar
