# review_dialog.py
"""
Review Dialog — A separate Qt dialog for generating EPUB reviews.
Launched from the "Generate Review" button in translator_gui.py.
"""

import os
import sys
import platform
import threading
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QPlainTextEdit, QTextEdit, QTextBrowser, QCheckBox, QApplication, QGroupBox, QSplitter,
    QComboBox, QWidget, QSpinBox
)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Slot, QEvent
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QGraphicsOpacityEffect

from review_generator import (
    DEFAULT_REVIEW_PROMPT,
    DEFAULT_FINAL_REVIEW_PROMPT,
    count_epub_tokens,
    generate_review,
    generate_chunked_review,
)


def _get_app_dir() -> str:
    """Return the application's base directory (Windows-safe)."""
    if platform.system() == 'Windows':
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()


class ReviewDialog(QDialog):
    """Dialog for generating an AI-powered review/summary of an EPUB."""

    def __init__(self, parent, translator_gui, file_path: str):
        super().__init__(parent)
        self.translator_gui = translator_gui
        self.file_path = file_path
        self._stop_requested = False
        self._force_stop = False
        self._review_thread = None
        self._counting = False
        self._token_cache = {}  # {file_path: token_count} — avoids recounting on switch
        self._raw_review_md = ''  # Raw markdown stored separately for saving

        self.setWindowTitle("Generate Review")
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

        # Sizing
        screen = QApplication.primaryScreen().availableGeometry()
        width = int(screen.width() * 0.48)
        height = int(screen.height() * 0.75)
        self.setMinimumSize(width, height)
        self.resize(width, height)

        # Icon
        try:
            base_dir = getattr(translator_gui, 'base_dir', _get_app_dir())
            icon_path = os.path.join(base_dir, 'Halgakos.ico')
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
        except Exception:
            pass

        self._build_ui()
        self._load_saved_prompt()
        self._load_existing_review()
        self._update_restore_btn_visibility()
        self._start_token_count()

    # ─── Styled checkbox (matches project pattern) ─────────────────

    def _create_styled_checkbox(self, text):
        """Create a checkbox with proper checkmark using text overlay"""
        checkbox = QCheckBox(text)
        checkbox.setStyleSheet("""
            QCheckBox {
                color: #e0e0e0;
                spacing: 8px;
                font-size: 12px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #5a5a5a;
                border-radius: 2px;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background-color: #4a7ba7;
                border-color: #4a7ba7;
            }
            QCheckBox::indicator:hover {
                border-color: #6a6a6a;
            }
        """)

        # Create checkmark overlay
        checkmark = QLabel("✓", checkbox)
        checkmark.setStyleSheet("""
            QLabel {
                color: white;
                background: transparent;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        checkmark.setAlignment(Qt.AlignCenter)
        checkmark.hide()
        checkmark.setAttribute(Qt.WA_TransparentForMouseEvents)

        def position_checkmark():
            try:
                if checkmark:
                    checkmark.setGeometry(2, 1, 16, 16)
            except RuntimeError:
                pass

        def update_checkmark():
            try:
                if checkbox and checkmark:
                    if checkbox.isChecked():
                        position_checkmark()
                        checkmark.show()
                    else:
                        checkmark.hide()
            except RuntimeError:
                pass

        checkbox.stateChanged.connect(update_checkmark)
        QTimer.singleShot(0, lambda: (position_checkmark(), update_checkmark()))

        return checkbox

    # ─── UI construction ─────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        # Header with EPUB navigator
        header_container = QHBoxLayout()
        header_container.setSpacing(6)

        # Gather all file paths from translator GUI (all review-supported formats)
        _REVIEW_EXTS = ('.epub', '.pdf', '.txt', '.html', '.htm', '.xhtml', '.md')
        self._all_epub_paths = []
        selected = getattr(self.translator_gui, 'selected_epub_files', None) or \
                   getattr(self.translator_gui, 'selected_files', None) or []
        for f in selected:
            f_str = str(f)
            if f_str.lower().endswith(_REVIEW_EXTS) and os.path.exists(f_str):
                self._all_epub_paths.append(f_str)
        # Ensure current file is in the list
        if self.file_path not in self._all_epub_paths:
            self._all_epub_paths.insert(0, self.file_path)

        nav_arrow_style = (
            "QPushButton { background-color: #3a3a3a; color: white; border: 1px solid #555; "
            "border-radius: 3px; padding: 4px 8px; font-size: 11pt; font-weight: bold; "
            "min-width: 26px; max-width: 26px; }"
            "QPushButton:hover { background-color: #505050; border-color: #5a9fd4; }"
            "QPushButton:disabled { color: #444; background-color: #252525; border-color: #333; }"
        )

        self._nav_prev_btn = QPushButton("◀")
        self._nav_prev_btn.setStyleSheet(nav_arrow_style)
        self._nav_prev_btn.setToolTip("Previous EPUB")
        header_container.addWidget(self._nav_prev_btn)

        from PySide6.QtWidgets import QComboBox
        self._epub_combo = QComboBox()
        self._epub_combo.setStyleSheet("""
            QComboBox {
                background-color: #2b2b2b; color: white; border: 1px solid #555;
                border-radius: 3px; padding: 4px 8px; font-size: 10pt;
            }
            QComboBox:hover { border-color: #5a9fd4; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox::down-arrow { image: none; border: none; }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b; color: white;
                selection-background-color: #5a9fd4;
                border: 1px solid #555;
            }
        """)
        for ep in self._all_epub_paths:
            self._epub_combo.addItem(f"📖 Review: {os.path.basename(ep)}", ep)
        # Set current index to match self.file_path
        try:
            idx = self._all_epub_paths.index(self.file_path)
            self._epub_combo.setCurrentIndex(idx)
        except ValueError:
            pass
        header_container.addWidget(self._epub_combo, 1)

        self._nav_next_btn = QPushButton("▶")
        self._nav_next_btn.setStyleSheet(nav_arrow_style)
        self._nav_next_btn.setToolTip("Next EPUB")
        header_container.addWidget(self._nav_next_btn)

        self._nav_counter = QLabel("")
        self._nav_counter.setStyleSheet("color: #5a9fd4; font-size: 9pt;")
        header_container.addWidget(self._nav_counter)

        # Reset to Default button (in the same header row)
        reset_prompt_btn = QPushButton("↺ Reset")
        reset_prompt_btn.setFixedHeight(26)
        reset_prompt_btn.setCursor(Qt.PointingHandCursor)
        reset_prompt_btn.setStyleSheet(
            "QPushButton { background-color: #2b3a4a; color: #5a9fd4; border: 1px solid #3d5a73; "
            "border-radius: 3px; padding: 2px 8px; font-size: 9pt; font-weight: 600; }"
            "QPushButton:hover { background-color: #3a4f66; color: #7ab8e8; border-color: #5a9fd4; }"
        )
        reset_prompt_btn.setToolTip("Reset system prompt to default")
        def _confirm_reset_prompt():
            from PySide6.QtWidgets import QMessageBox, QDialogButtonBox
            msg = QMessageBox(self)
            msg.setWindowTitle("Reset Prompt")
            msg.setText("Reset system prompt to default?")
            msg.setIcon(QMessageBox.Warning)
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setDefaultButton(QMessageBox.No)
            msg.setStyleSheet("""
                QPushButton {
                    min-width: 80px;
                    min-height: 30px;
                    padding: 6px 20px;
                    font-size: 10pt;
                }
                QDialogButtonBox {
                    qproperty-centerButtons: true;
                }
            """)
            if msg.exec() == QMessageBox.Yes:
                self.prompt_edit.setPlainText(DEFAULT_REVIEW_PROMPT)
        reset_prompt_btn.clicked.connect(_confirm_reset_prompt)
        header_container.addWidget(reset_prompt_btn)

        layout.addLayout(header_container)

        # Hide navigator if only 1 file
        show_nav = len(self._all_epub_paths) > 1
        self._nav_prev_btn.setVisible(show_nav)
        self._nav_next_btn.setVisible(show_nav)
        self._nav_counter.setVisible(show_nav)
        if not show_nav:
            self._epub_combo.setStyleSheet(
                "QComboBox { background-color: #2b2b2b; color: white; border: none; "
                "border-radius: 3px; padding: 4px 8px; font-size: 12pt; font-weight: bold; }"
                "QComboBox::drop-down { width: 0px; }"
                "QComboBox QAbstractItemView { background-color: #2b2b2b; color: white; "
                "selection-background-color: #5a9fd4; border: 1px solid #555; }"
            )

        # Wire up navigation
        def _on_epub_nav_prev():
            idx = self._epub_combo.currentIndex()
            if idx > 0:
                self._epub_combo.setCurrentIndex(idx - 1)

        def _on_epub_nav_next():
            idx = self._epub_combo.currentIndex()
            if idx < self._epub_combo.count() - 1:
                self._epub_combo.setCurrentIndex(idx + 1)

        def _on_epub_switched(index):
            if index < 0 or index >= len(self._all_epub_paths):
                return
            new_path = self._all_epub_paths[index]
            if new_path == self.file_path:
                return
            self.file_path = new_path
            self.setWindowTitle(f"Generate Review — {os.path.basename(new_path)}")
            # Reload review and token count
            self.log_field.clear()
            self._load_existing_review()
            self._update_restore_btn_visibility()
            self._start_token_count()
            # Update nav state
            self._nav_prev_btn.setEnabled(index > 0)
            self._nav_next_btn.setEnabled(index < len(self._all_epub_paths) - 1)
            self._nav_counter.setText(f"{index + 1} / {len(self._all_epub_paths)}")

        self._nav_prev_btn.clicked.connect(_on_epub_nav_prev)
        self._nav_next_btn.clicked.connect(_on_epub_nav_next)
        self._epub_combo.currentIndexChanged.connect(_on_epub_switched)

        # Initial counter state
        if show_nav:
            try:
                init_idx = self._all_epub_paths.index(self.file_path)
            except ValueError:
                init_idx = 0
            self._nav_prev_btn.setEnabled(init_idx > 0)
            self._nav_next_btn.setEnabled(init_idx < len(self._all_epub_paths) - 1)
            self._nav_counter.setText(f"{init_idx + 1} / {len(self._all_epub_paths)}")

        # ── 1. System Prompt ──
        prompt_group = QGroupBox("System Prompt")
        prompt_layout = QVBoxLayout(prompt_group)
        prompt_layout.setContentsMargins(8, 8, 8, 8)
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("Enter the system prompt for the review...")
        prompt_layout.addWidget(self.prompt_edit)

        # ── 2. Log / Summary field ──
        log_group = QGroupBox("Review Output")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(8, 8, 8, 8)
        self.log_field = QTextBrowser()
        self.log_field.setReadOnly(True)
        self.log_field.setOpenExternalLinks(True)
        self.log_field.setPlaceholderText("Generated review will appear here...")
        self.log_field.setContextMenuPolicy(Qt.CustomContextMenu)
        self.log_field.customContextMenuRequested.connect(self._show_log_context_menu)
        log_layout.addWidget(self.log_field)

        # ── Collapsible Font Settings ──
        self._font_settings_toggle = QPushButton("▶  Font Settings")
        self._font_settings_toggle.setStyleSheet(
            "QPushButton { background: transparent; color: #aaa; border: none; "
            "text-align: left; padding: 2px 8px; font-size: 9pt; }"
            "QPushButton:hover { color: #ddd; }"
        )
        self._font_settings_toggle.setFixedHeight(22)
        self._font_settings_toggle.setCursor(Qt.PointingHandCursor)
        self._font_settings_toggle.clicked.connect(self._toggle_font_settings)
        log_layout.addWidget(self._font_settings_toggle)



        self._review_font_combo = QComboBox()
        self._review_font_combo.setEditable(True)
        self._review_font_combo.setInsertPolicy(QComboBox.NoInsert)
        self._review_font_combo.setMinimumWidth(200)
        self._review_font_combo.setFocusPolicy(Qt.StrongFocus)
        self._review_font_combo.wheelEvent = lambda e: e.ignore()
        # Set dropdown arrow icon
        import os as _os
        _ico_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'Halgakos.ico').replace('\\', '/')
        self._review_font_combo.setStyleSheet(
            f'QComboBox::down-arrow {{ image: url({_ico_path}); width: 14px; height: 14px; }}'
        )
        # Populate with system fonts
        from PySide6.QtGui import QFontDatabase
        families = sorted(set(
            f for f in QFontDatabase.families()
            if QFontDatabase.isSmoothlyScalable(f)
        ))
        self._review_font_combo.addItem("Segoe UI")
        for fam in families:
            if fam != "Segoe UI":
                self._review_font_combo.addItem(fam)
        saved_font = self.translator_gui.config.get('review_font_family', 'Segoe UI')
        idx = self._review_font_combo.findText(saved_font)
        if idx >= 0:
            self._review_font_combo.setCurrentIndex(idx)
        self._review_font_combo.currentTextChanged.connect(self._on_review_font_changed)

        # Font size
        self._review_font_size_spin = QSpinBox()
        self._review_font_size_spin.setRange(5, 36)
        self._review_font_size_spin.setValue(int(self.translator_gui.config.get('review_font_size', 9)))
        self._review_font_size_spin.setSuffix("pt")
        self._review_font_size_spin.setFocusPolicy(Qt.StrongFocus)
        self._review_font_size_spin.wheelEvent = lambda e: e.ignore()
        self._review_font_size_spin.valueChanged.connect(self._on_review_font_changed)

        # Line height
        self._review_line_height_spin = QSpinBox()
        self._review_line_height_spin.setRange(20, 300)
        self._review_line_height_spin.setValue(int(self.translator_gui.config.get('review_line_height', 100)))
        self._review_line_height_spin.setSuffix("%")
        self._review_line_height_spin.setSingleStep(10)
        self._review_line_height_spin.setFocusPolicy(Qt.StrongFocus)
        self._review_line_height_spin.wheelEvent = lambda e: e.ignore()
        self._review_line_height_spin.valueChanged.connect(self._on_review_font_changed)

        # Color picker button
        self._review_font_color = self.translator_gui.config.get('review_font_color', '#e0e0e0')
        self._review_color_btn = QPushButton()
        self._review_color_btn.setFixedSize(24, 24)
        self._review_color_btn.setCursor(Qt.PointingHandCursor)
        self._review_color_btn.setStyleSheet(
            f"QPushButton {{ background-color: {self._review_font_color}; border: 1px solid #888; border-radius: 3px; }}"
            f"QPushButton:hover {{ border: 2px solid white; }}"
        )
        self._review_color_btn.clicked.connect(self._pick_review_font_color)

        # Header spacing
        self._review_header_spacing_spin = QSpinBox()
        self._review_header_spacing_spin.setRange(-4, 24)
        self._review_header_spacing_spin.setValue(int(self.translator_gui.config.get('review_header_spacing', 6)))
        self._review_header_spacing_spin.setSuffix("px")
        self._review_header_spacing_spin.setFocusPolicy(Qt.StrongFocus)
        self._review_header_spacing_spin.wheelEvent = lambda e: e.ignore()
        self._review_header_spacing_spin.valueChanged.connect(self._on_review_font_changed)

        # Paragraph spacing
        self._review_para_spacing_spin = QSpinBox()
        self._review_para_spacing_spin.setRange(0, 24)
        self._review_para_spacing_spin.setValue(int(self.translator_gui.config.get('review_spacing', 8)))
        self._review_para_spacing_spin.setSuffix("px")
        self._review_para_spacing_spin.setFocusPolicy(Qt.StrongFocus)
        self._review_para_spacing_spin.wheelEvent = lambda e: e.ignore()
        self._review_para_spacing_spin.valueChanged.connect(self._on_review_font_changed)

        # List item spacing
        self._review_list_gap_spin = QSpinBox()
        self._review_list_gap_spin.setRange(0, 24)
        self._review_list_gap_spin.setValue(int(self.translator_gui.config.get('review_list_gap', 10)))
        self._review_list_gap_spin.setSuffix("px")
        self._review_list_gap_spin.setFocusPolicy(Qt.StrongFocus)
        self._review_list_gap_spin.wheelEvent = lambda e: e.ignore()
        self._review_list_gap_spin.valueChanged.connect(self._on_review_font_changed)

        # Layout: two rows
        _s = "color: #aaa; font-size: 9pt;"
        font_row = QHBoxLayout()
        font_row.setContentsMargins(8, 0, 8, 0)
        font_row.setSpacing(6)
        for lbl, w in [("Font:", self._review_font_combo), ("Size:", self._review_font_size_spin),
                        ("Line:", self._review_line_height_spin), ("Color:", self._review_color_btn)]:
            l = QLabel(lbl)
            l.setStyleSheet(_s)
            font_row.addWidget(l)
            font_row.addWidget(w)
        font_row.addStretch()

        spacing_row = QHBoxLayout()
        spacing_row.setContentsMargins(8, 0, 8, 4)
        spacing_row.setSpacing(6)
        for lbl, w in [("Header Gap:", self._review_header_spacing_spin),
                        ("Paragraph Gap:", self._review_para_spacing_spin),
                        ("List Gap:", self._review_list_gap_spin)]:
            l = QLabel(lbl)
            l.setStyleSheet(_s)
            spacing_row.addWidget(l)
            spacing_row.addWidget(w)
        spacing_row.addStretch()

        self._reset_font_btn = QPushButton("↺ Reset")
        self._reset_font_btn.setFixedHeight(28)
        self._reset_font_btn.setCursor(Qt.PointingHandCursor)
        self._reset_font_btn.setStyleSheet(
            "QPushButton { color: #ccc; font-size: 10pt; padding: 4px 16px; "
            "border: 1px solid #555; border-radius: 3px; background: #383838; }"
            "QPushButton:hover { background: #4a4a4a; color: white; border-color: #888; }"
        )
        self._reset_font_btn.clicked.connect(self._reset_font_settings)
        spacing_row.addWidget(self._reset_font_btn)

        font_settings_layout = QVBoxLayout()
        font_settings_layout.setContentsMargins(0, 4, 0, 4)
        font_settings_layout.setSpacing(8)
        font_settings_layout.addLayout(font_row)
        font_settings_layout.addLayout(spacing_row)

        self._font_settings_widget = QWidget()
        self._font_settings_widget.setLayout(font_settings_layout)
        self._font_settings_widget.hide()
        log_layout.addWidget(self._font_settings_widget)

        # ── Splitter between prompt and output ──
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(prompt_group)
        splitter.addWidget(log_group)
        splitter.setStretchFactor(0, 7)  # prompt (~35%)
        splitter.setStretchFactor(1, 13)  # output (~65%)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet("""
            QSplitter::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 transparent, stop:0.3 #555, stop:0.5 #888,
                    stop:0.7 #555, stop:1 transparent);
                height: 6px;
                margin: 2px 40px;
                border-radius: 3px;
            }
            QSplitter::handle:vertical:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 transparent, stop:0.3 #4a7ba7, stop:0.5 #6ba3d6,
                    stop:0.7 #4a7ba7, stop:1 transparent);
            }
        """)
        layout.addWidget(splitter, stretch=1)

        # ── Controls row ──
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(12)

        # 3. Spoiler Mode
        self.spoiler_checkbox = self._create_styled_checkbox("Spoiler Mode")
        self.spoiler_checkbox.setToolTip(
            "When enabled, includes both the first and last chapters "
            "(50/50 split) for a comprehensive review.\n"
            "When disabled, only sends chapters from the beginning."
        )
        self.spoiler_checkbox.setChecked(False)
        controls_layout.addWidget(self.spoiler_checkbox)

        # Chunk Mode toggle
        self.chunk_mode_checkbox = self._create_styled_checkbox("Chunk Mode")
        self.chunk_mode_checkbox.setToolTip(
            "When enabled, splits the book into multiple chunks and reviews each\n"
            "separately, then sends a Final Review prompt to synthesize all chunk\n"
            "reviews into one comprehensive review.\n\n"
            "This allows reviewing the ENTIRE book regardless of token limits.\n"
            "Works with Spoiler Mode enabled (spoiler logic applies per-chunk)."
        )
        self.chunk_mode_checkbox.setChecked(False)
        controls_layout.addWidget(self.chunk_mode_checkbox)

        # Wrap Chunks toggle
        self.chunk_wrap_checkbox = self._create_styled_checkbox("Wrap Chunks")
        self.chunk_wrap_checkbox.setToolTip(
            "When enabled, each chunk review is wrapped with header/footer markers\n"
            "indicating the chapter range (e.g. === CHUNK REVIEW: ch1 → ch5 ===).\n"
            "Disable to send raw chunk reviews without markers."
        )
        self.chunk_wrap_checkbox.setChecked(True)
        controls_layout.addWidget(self.chunk_wrap_checkbox)

        # Final Review Prompt button
        self._final_prompt_btn = QPushButton("📝 Final Prompt")
        self._final_prompt_btn.setFixedHeight(26)
        self._final_prompt_btn.setCursor(Qt.PointingHandCursor)
        self._final_prompt_btn.setStyleSheet(
            "QPushButton { background-color: #2b3a4a; color: #5a9fd4; border: 1px solid #3d5a73; "
            "border-radius: 3px; padding: 2px 8px; font-size: 9pt; font-weight: 600; }"
            "QPushButton:hover { background-color: #3a4f66; color: #7ab8e8; border-color: #5a9fd4; }"
        )
        self._final_prompt_btn.setToolTip("Edit the Final Review synthesis prompt used in Chunk Mode")
        self._final_prompt_btn.clicked.connect(self._open_final_prompt_dialog)
        controls_layout.addWidget(self._final_prompt_btn)

        # Toggle visibility of chunk-related controls based on chunk mode
        def _on_chunk_mode_toggled(state):
            is_on = bool(state)
            self.chunk_wrap_checkbox.setVisible(is_on)
            self._final_prompt_btn.setVisible(is_on)
        self.chunk_mode_checkbox.stateChanged.connect(_on_chunk_mode_toggled)
        # Initial visibility
        self.chunk_wrap_checkbox.setVisible(False)
        self._final_prompt_btn.setVisible(False)

        # 5. Token count
        self.token_label = QLabel("⏳ Counting tokens...")
        self.token_label.setStyleSheet("color: #94a3b8; font-size: 10pt;")
        controls_layout.addWidget(self.token_label)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # ── Button row ──
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)

        # 4. Start Review (single EPUB)
        self.start_btn = QPushButton("🚀 Start Review")
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #4a7ba7; color: white; font-weight: bold; "
            "padding: 8px 16px; border-radius: 4px; font-size: 10pt; }"
            "QPushButton:disabled { background-color: #3a3a3a; color: #6a6a6a; border: 1px solid #4a4a4a; }"
        )
        self.start_btn.setMinimumWidth(140)
        self.start_btn.clicked.connect(self._on_start_review)
        button_layout.addWidget(self.start_btn)

        # Generate All button (visible only with multiple EPUBs)
        self.generate_all_btn = QPushButton("📚 Review all Files")
        self.generate_all_btn.setStyleSheet(
            "QPushButton { background-color: #6f42c1; color: white; font-weight: bold; "
            "padding: 8px 16px; border-radius: 4px; font-size: 10pt; }"
            "QPushButton:hover { background-color: #8552e0; }"
            "QPushButton:disabled { background-color: #3a3a3a; color: #6a6a6a; border: 1px solid #4a4a4a; }"
        )
        self.generate_all_btn.setMinimumWidth(140)
        self.generate_all_btn.setToolTip("Generate reviews for all selected EPUBs")
        self.generate_all_btn.clicked.connect(self._on_generate_all)
        button_layout.addWidget(self.generate_all_btn)
        if len(self._all_epub_paths) <= 1:
            self.generate_all_btn.hide()

        # Stop button with spinning Halgakos icon (hidden initially)
        self.stop_btn = QPushButton()
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #dc3545; color: white; font-weight: bold; "
            "padding: 8px 16px; border-radius: 4px; font-size: 10pt; border: none; }"
        )
        self.stop_btn.setMinimumWidth(150)
        self.stop_btn.clicked.connect(self._on_stop)

        # Build icon + text layout inside button
        stop_btn_layout = QHBoxLayout(self.stop_btn)
        stop_btn_layout.setContentsMargins(8, 4, 8, 4)
        stop_btn_layout.setSpacing(2)

        self._stop_icon_label = QLabel()
        self._stop_icon_label.setStyleSheet("background-color: transparent;")
        self._stop_icon_label.setFixedSize(32, 32)
        self._stop_icon_label.setAlignment(Qt.AlignCenter)

        self._stop_text_label = QLabel("Stop Review")
        self._stop_text_label.setStyleSheet("color: white; font-weight: bold; background-color: transparent; font-size: 10pt;")
        self._stop_text_label.setAlignment(Qt.AlignCenter)

        stop_btn_layout.addWidget(self._stop_icon_label)
        stop_btn_layout.addWidget(self._stop_text_label)

        # Load Halgakos.ico and build spinner frames
        self._stop_spinner_frames = []
        self._stop_spinner_idx = 0
        try:
            # Try multiple paths for base_dir
            base_dir = getattr(self.translator_gui, 'base_dir', None) or getattr(translator_gui, 'base_dir', None) or os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(base_dir, 'Halgakos.ico')
            print(f"[ReviewDialog] Stop icon path: {icon_path} exists={os.path.exists(icon_path)}")
            if os.path.exists(icon_path):
                from PySide6.QtGui import QPixmap, QTransform
                from PySide6.QtCore import QSize
                try:
                    dpr = self.devicePixelRatioF()
                except Exception:
                    dpr = 1.0
                logical_px = 16
                dev_px = int(logical_px * max(1.0, dpr))
                icon = QIcon(icon_path)
                pm = icon.pixmap(QSize(dev_px, dev_px))
                if pm.isNull():
                    pm = QPixmap(icon_path).scaled(dev_px, dev_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                try:
                    pm.setDevicePixelRatio(dpr)
                except Exception:
                    pass
                self._stop_icon_label.setPixmap(pm)
                self._stop_icon_base_pm = pm

                # Precompute rotation frames
                steps = 48
                for i in range(steps):
                    angle = (i * 360) / steps
                    rotated = pm.transformed(QTransform().rotate(angle), Qt.SmoothTransformation)
                    try:
                        rotated.setDevicePixelRatio(dpr)
                    except Exception:
                        pass
                    self._stop_spinner_frames.append(rotated)
        except Exception:
            pass

        # Spinner timer
        self._stop_spinner_timer = QTimer(self)
        self._stop_spinner_timer.setInterval(14)
        self._stop_spinner_timer.timeout.connect(self._advance_stop_spinner)

        self.stop_btn.hide()
        button_layout.addWidget(self.stop_btn)

        button_layout.addStretch()

        # 6. Save
        self.save_btn = QPushButton("💾 Save")
        self.save_btn.setStyleSheet(
            "QPushButton { background-color: #28a745; color: white; font-weight: bold; "
            "padding: 10px 24px; border-radius: 4px; font-size: 11pt; }"
            "QPushButton:disabled { background-color: #555; color: #888; }"
        )
        self.save_btn.setMinimumWidth(120)
        self.save_btn.clicked.connect(self._on_save)
        button_layout.addWidget(self.save_btn)

        # Delete Review
        self.delete_btn = QPushButton("🗑️ Delete")
        self.delete_btn.setStyleSheet(
            "QPushButton { background-color: #dc3545; color: white; font-weight: bold; "
            "padding: 10px 24px; border-radius: 4px; font-size: 11pt; }"
            "QPushButton:disabled { background-color: #555; color: #888; }"
        )
        self.delete_btn.setMinimumWidth(120)
        self.delete_btn.clicked.connect(self._on_delete)
        button_layout.addWidget(self.delete_btn)

        # Restore Review (hidden when no backups exist)
        self.restore_btn = QPushButton("↩️ Restore")
        self.restore_btn.setStyleSheet(
            "background-color: #6f42c1; color: white; font-weight: bold; "
            "padding: 10px 24px; border-radius: 4px; font-size: 11pt;"
        )
        self.restore_btn.setMinimumWidth(120)
        self.restore_btn.clicked.connect(self._on_restore)
        self.restore_btn.hide()
        button_layout.addWidget(self.restore_btn)

        # 7. Close
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            "padding: 10px 24px; border-radius: 4px; font-size: 11pt;"
        )
        close_btn.setMinimumWidth(120)
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    # ─── EPUB list sync ──────────────────────────────────────────────

    def refresh_epub_list(self):
        """Re-read selected files from translator_gui and update the dropdown."""
        _REVIEW_EXTS = ('.epub', '.pdf', '.txt', '.html', '.htm', '.xhtml', '.md')
        # Build authoritative list from translator_gui — do NOT keep stale file_path
        new_paths = []
        selected = getattr(self.translator_gui, 'selected_epub_files', None) or \
                   getattr(self.translator_gui, 'selected_files', None) or []
        for f in selected:
            f_str = str(f)
            if f_str.lower().endswith(_REVIEW_EXTS) and os.path.exists(f_str):
                new_paths.append(f_str)
        # If no files at all, fall back to current file_path if it still exists
        if not new_paths and os.path.exists(self.file_path):
            new_paths = [self.file_path]

        # Skip update if list hasn't changed
        if new_paths == self._all_epub_paths:
            return

        self._all_epub_paths = new_paths
        show_nav = len(new_paths) > 1

        # If current file_path is no longer in the list, switch to first
        if self.file_path not in new_paths and new_paths:
            self.file_path = new_paths[0]

        # Update combo
        self._epub_combo.blockSignals(True)
        self._epub_combo.clear()
        for ep in new_paths:
            self._epub_combo.addItem(f"📖 Review: {os.path.basename(ep)}", ep)
        try:
            idx = new_paths.index(self.file_path)
        except ValueError:
            idx = 0
        self._epub_combo.setCurrentIndex(idx)
        self._epub_combo.blockSignals(False)

        # Show/hide nav elements
        self._nav_prev_btn.setVisible(show_nav)
        self._nav_next_btn.setVisible(show_nav)
        self._nav_counter.setVisible(show_nav)
        if show_nav:
            self._nav_prev_btn.setEnabled(idx > 0)
            self._nav_next_btn.setEnabled(idx < len(new_paths) - 1)
            self._nav_counter.setText(f"{idx + 1} / {len(new_paths)}")
        # Show/hide Review All button
        if hasattr(self, 'generate_all_btn'):
            self.generate_all_btn.setVisible(show_nav)

        # Update combo style for single vs multi
        if not show_nav:
            self._epub_combo.setStyleSheet(
                "QComboBox { background-color: #2b2b2b; color: white; border: none; "
                "border-radius: 3px; padding: 4px 8px; font-size: 12pt; font-weight: bold; }"
                "QComboBox::drop-down { width: 0px; }"
                "QComboBox QAbstractItemView { background-color: #2b2b2b; color: white; "
                "selection-background-color: #5a9fd4; border: 1px solid #555; }"
            )
        else:
            self._epub_combo.setStyleSheet("""
                QComboBox {
                    background-color: #2b2b2b; color: white; border: 1px solid #555;
                    border-radius: 3px; padding: 4px 8px; font-size: 10pt;
                }
                QComboBox:hover { border-color: #5a9fd4; }
                QComboBox::drop-down { border: none; width: 20px; }
                QComboBox::down-arrow { image: none; border: none; }
                QComboBox QAbstractItemView {
                    background-color: #2b2b2b; color: white;
                    selection-background-color: #5a9fd4;
                    border: 1px solid #555;
                }
            """)

        # Update window title and reload for new file
        self.setWindowTitle(f"Generate Review — {os.path.basename(self.file_path)}")
        self.log_field.clear()
        self._load_existing_review()
        self._update_restore_btn_visibility()
        self._start_token_count()

    # ─── Config / persistence ────────────────────────────────────────

    def _load_saved_prompt(self):
        """Load previously saved review prompt, spoiler mode, and chunk mode settings from config."""
        config = getattr(self.translator_gui, 'config', {})
        saved = config.get('review_system_prompt', '')
        self.prompt_edit.setPlainText(saved if saved else DEFAULT_REVIEW_PROMPT)
        # Load spoiler mode
        spoiler = config.get('review_spoiler_mode', False)
        self.spoiler_checkbox.setChecked(bool(spoiler))
        # Load chunk mode settings
        chunk_mode = config.get('review_chunk_mode', False)
        self.chunk_mode_checkbox.setChecked(bool(chunk_mode))
        chunk_wrap = config.get('review_chunk_wrap', True)
        self.chunk_wrap_checkbox.setChecked(bool(chunk_wrap))
        # Load final review prompt (stored separately from the main system prompt)
        self._final_review_prompt = config.get('review_final_prompt', '') or DEFAULT_FINAL_REVIEW_PROMPT

    def _sync_settings_to_gui(self):
        """Sync current dialog values to translator_gui _var attributes for save_config."""
        gui = self.translator_gui
        gui.review_system_prompt_var = self.prompt_edit.toPlainText().strip()
        gui.review_spoiler_mode_var = self.spoiler_checkbox.isChecked()
        gui.review_chunk_mode_var = self.chunk_mode_checkbox.isChecked()
        gui.review_chunk_wrap_var = self.chunk_wrap_checkbox.isChecked()
        gui.review_final_prompt_var = getattr(self, '_final_review_prompt', DEFAULT_FINAL_REVIEW_PROMPT)

    def _save_prompt_to_config(self):
        """Persist the current system prompt, spoiler mode, and chunk mode settings to config."""
        try:
            self._sync_settings_to_gui()
            if hasattr(self.translator_gui, 'save_config'):
                self.translator_gui.save_config(show_message=False)
        except Exception:
            pass

    def _open_final_prompt_dialog(self):
        """Open a dialog to edit the Final Review synthesis prompt."""
        # Reuse existing dialog if open
        if hasattr(self, '_final_prompt_dialog') and self._final_prompt_dialog and self._final_prompt_dialog.isVisible():
            self._final_prompt_dialog.raise_()
            self._final_prompt_dialog.activateWindow()
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Final Review Prompt")
        dialog.setModal(False)
        dialog.setMinimumSize(600, 400)
        dialog.resize(700, 500)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        info_label = QLabel(
            "This prompt is sent as the system prompt for the final synthesis call.\n"
            "It receives all chunk reviews as user content and produces the combined review."
        )
        info_label.setStyleSheet("color: #94a3b8; font-size: 9pt;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        prompt_edit = QPlainTextEdit()
        prompt_edit.setPlainText(getattr(self, '_final_review_prompt', DEFAULT_FINAL_REVIEW_PROMPT))
        prompt_edit.setPlaceholderText("Enter the final review synthesis prompt...")
        layout.addWidget(prompt_edit, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        reset_btn = QPushButton("↺ Reset to Default")
        reset_btn.setCursor(Qt.PointingHandCursor)
        reset_btn.setStyleSheet(
            "QPushButton { background-color: #2b3a4a; color: #5a9fd4; border: 1px solid #3d5a73; "
            "border-radius: 3px; padding: 6px 16px; font-size: 9pt; }"
            "QPushButton:hover { background-color: #3a4f66; color: #7ab8e8; border-color: #5a9fd4; }"
        )
        def _confirm_reset_final_prompt():
            from PySide6.QtWidgets import QMessageBox, QDialogButtonBox
            msg = QMessageBox(dialog)
            msg.setWindowTitle("Reset Final Prompt")
            msg.setText("Reset final review prompt to default?")
            msg.setIcon(QMessageBox.Warning)
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setDefaultButton(QMessageBox.No)
            msg.setStyleSheet("""
                QPushButton {
                    min-width: 80px;
                    min-height: 30px;
                    padding: 6px 20px;
                    font-size: 10pt;
                }
                QDialogButtonBox {
                    qproperty-centerButtons: true;
                }
            """)
            if msg.exec() == QMessageBox.Yes:
                prompt_edit.setPlainText(DEFAULT_FINAL_REVIEW_PROMPT)
        reset_btn.clicked.connect(_confirm_reset_final_prompt)
        btn_row.addWidget(reset_btn)

        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("padding: 6px 20px; border-radius: 3px; font-size: 10pt;")
        cancel_btn.clicked.connect(dialog.close)
        btn_row.addWidget(cancel_btn)

        ok_btn = QPushButton("✅ Save")
        ok_btn.setStyleSheet(
            "QPushButton { background-color: #28a745; color: white; font-weight: bold; "
            "padding: 6px 20px; border-radius: 3px; font-size: 10pt; }"
            "QPushButton:hover { background-color: #2dbc4e; }"
        )
        def _save_and_close():
            self._final_review_prompt = prompt_edit.toPlainText().strip() or DEFAULT_FINAL_REVIEW_PROMPT
            self._save_prompt_to_config()
            try:
                import winsound
                winsound.MessageBeep(winsound.MB_OK)
            except Exception:
                pass
            dialog.close()
        ok_btn.clicked.connect(_save_and_close)
        btn_row.addWidget(ok_btn)

        layout.addLayout(btn_row)

        self._final_prompt_dialog = dialog
        dialog.show()

    def _load_existing_review(self):
        """If a review already exists, load it into the log field."""
        review_path = self._get_review_path()
        if review_path and os.path.exists(review_path):
            try:
                with open(review_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content.strip():
                    self._raw_review_md = content
                    self._last_rendered_html = self._md_to_html(content, **self._get_font_kwargs())
                    self.log_field.setHtml(self._last_rendered_html)
                    self._load_remote_images()
                    self.save_btn.setEnabled(True)
                    self.delete_btn.setEnabled(True)
                    # Apply font size on widget level
                    font_kwargs = self._get_font_kwargs()
                    f = self.log_field.font()
                    f.setPointSize(font_kwargs['font_size'])
                    self.log_field.document().setDefaultFont(f)
            except Exception:
                pass

    def _get_review_path(self) -> str:
        """Get the review output path for the current EPUB."""
        try:
            epub_base = os.path.splitext(os.path.basename(self.file_path))[0]
            override_dir = os.environ.get('OUTPUT_DIRECTORY') or \
                           getattr(self.translator_gui, 'config', {}).get('output_directory')
            if override_dir:
                output_dir = os.path.join(os.path.abspath(override_dir), epub_base)
            else:
                output_dir = os.path.join(_get_app_dir(), epub_base)
            return os.path.join(output_dir, "review", "review.md")
        except Exception:
            return None

    # ─── Token counting (background thread) ──────────────────────────

    def _start_token_count(self):
        """Count EPUB tokens in a background thread (cached per file)."""
        # Check cache first
        cached = self._token_cache.get(self.file_path)
        if cached is not None:
            self.token_label.setText(f"📊 File tokens: {cached:,}")
            self.token_label.setStyleSheet("color: #22c55e; font-size: 10pt;")
            self._counting = False
            return

        self._counting = True
        self._token_result = None  # Will be set by background thread
        self.token_label.setText("⏳ Counting tokens...")
        self.token_label.setStyleSheet("color: #f59e0b; font-size: 10pt;")

        file_path_for_count = self.file_path  # Capture for thread + closure

        def _count():
            try:
                total = count_epub_tokens(file_path_for_count)
                self._token_result = ('ok', total, file_path_for_count)
            except Exception as e:
                import traceback
                err_msg = f"{e}\n{traceback.format_exc()}"
                print(f"[ReviewDialog] Token counting error: {err_msg}")
                self._token_result = ('error', err_msg, file_path_for_count)

        t = threading.Thread(target=_count, daemon=True)
        t.start()

        # Poll for result on the main thread every 200ms
        self._token_poll_timer = QTimer(self)
        self._token_poll_timer.setInterval(200)
        def _check_result():
            result = self._token_result
            if result is None:
                return  # Still counting
            self._token_poll_timer.stop()
            self._counting = False
            counted_path = result[2]
            if result[0] == 'error':
                # Only update label if still on the same file
                if self.file_path == counted_path:
                    self.token_label.setText(f"⚠️ Token count failed: {result[1][:100]}")
                    self.token_label.setStyleSheet("color: #ef4444; font-size: 10pt;")
            elif result[1] >= 0:
                # Always cache the result
                self._token_cache[counted_path] = result[1]
                # Only update label if still on the same file
                if self.file_path == counted_path:
                    self.token_label.setText(f"📊 File tokens: {result[1]:,}")
                    self.token_label.setStyleSheet("color: #22c55e; font-size: 10pt;")
            else:
                if self.file_path == counted_path:
                    self.token_label.setText("⚠️ Could not count tokens")
                    self.token_label.setStyleSheet("color: #ef4444; font-size: 10pt;")
        self._token_poll_timer.timeout.connect(_check_result)
        self._token_poll_timer.start()

    # ─── Review generation ───────────────────────────────────────────

    def _on_start_review(self):
        """Start the review generation in a background thread."""
        if self._review_thread and self._review_thread.is_alive():
            return  # Already running

        # Warn only if a saved review file exists on disk
        review_path = self._get_review_path()
        if review_path and os.path.exists(review_path):
            from PySide6.QtWidgets import QMessageBox
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Overwrite Review?")
            msg.setText(
                "A review is already displayed in the output.\n"
                "Starting a new review will replace it.\n\n"
                "Continue?"
            )
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setDefaultButton(QMessageBox.No)
            msg.setStyleSheet("""
                QPushButton {
                    min-width: 80px;
                    min-height: 30px;
                    padding: 6px 20px;
                    font-size: 10pt;
                }
                QDialogButtonBox {
                    qproperty-centerButtons: true;
                }
            """)
            reply = msg.exec()
            if reply != QMessageBox.Yes:
                return

        self._stop_requested = False
        self._force_stop = False
        self._stop_click_times = []
        self._save_prompt_to_config()

        # Gather parameters from the GUI
        gui = self.translator_gui
        api_key = gui.api_key_entry.text().strip() if hasattr(gui, 'api_key_entry') else ''
        model = getattr(gui, 'model_var', os.getenv('MODEL', 'gemini-2.0-flash'))
        endpoint = os.environ.get('ENDPOINT', '') or gui.config.get('endpoint', '')
        temperature = float(gui.config.get('translation_temperature', 0.3))
        config = dict(gui.config)

        # Get input token limit
        try:
            token_limit = int(gui.token_limit_entry.text().replace(',', '').strip())
        except (ValueError, AttributeError):
            token_limit = 200000

        system_prompt = self.prompt_edit.toPlainText().strip()
        # Replace {target_lang} placeholder with the selected target language
        output_lang = getattr(gui, 'lang_var', 'English')
        system_prompt = system_prompt.replace('{target_lang}', output_lang)
        spoiler_mode = self.spoiler_checkbox.isChecked()

        # Determine output directory
        epub_base = os.path.splitext(os.path.basename(self.file_path))[0]
        override_dir = os.environ.get('OUTPUT_DIRECTORY') or config.get('output_directory')
        if override_dir:
            output_dir = os.path.join(os.path.abspath(override_dir), epub_base)
        else:
            output_dir = os.path.join(_get_app_dir(), epub_base)

        # UI state
        self.start_btn.hide()
        self.generate_all_btn.hide()
        self._stop_text_label.setText("Stop Review")
        self.stop_btn.show()
        self._stop_spinner_timer.start()
        self.log_field.clear()
        self._raw_review_md = ''
        self.save_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)
        self.restore_btn.hide()

        # Clear any lingering cancellation from a previous force stop
        try:
            from unified_api_client import UnifiedClient
            UnifiedClient.set_global_cancellation(False)
        except Exception:
            pass

        # ── Hijack translator_gui.append_log ──
        # Save original and replace with a wrapper that also writes to review dialog
        self._original_append_log = self.translator_gui.append_log
        self._review_log_active = True
        self._review_log_backlog = []

        def _hijacked_append_log(message):
            # During review: send ONLY to review dialog, buffer for later main GUI replay
            if self._review_log_active:
                self._review_queue.put(('log', str(message)))
                self._review_log_backlog.append(str(message))

        self.translator_gui.append_log = _hijacked_append_log

        # Message queue for thread → main-thread communication
        import queue
        self._review_queue = queue.Queue()

        # ── Hijack sys.stdout so print() calls from streaming/thinking code
        #    are routed through the review dialog queue instead of main GUI ──
        self._original_stdout = sys.stdout
        _review_queue_ref = self._review_queue
        _active_flag = self  # reference to self for checking _review_log_active

        class _ReviewStdoutWriter:
            def write(self, text):
                if text and text.strip() and getattr(_active_flag, '_review_log_active', False):
                    _review_queue_ref.put(('log', text.strip()))
            def flush(self):
                pass

        sys.stdout = _ReviewStdoutWriter()

        # ── Respect the streaming toggle from settings ──
        stream_on = bool(self.translator_gui.config.get('enable_streaming', False))
        os.environ['ENABLE_STREAMING'] = '1' if stream_on else '0'


        def _log(msg):
            # Send through the hijacked append_log (reaches both main GUI and review dialog)
            # Prefix each line with [Review], but skip separator-only and blank lines
            lines = str(msg).split('\n')
            prefixed = []
            for line in lines:
                stripped = line.strip()
                if not stripped or all(c in '─═' for c in stripped):
                    prefixed.append(line)
                else:
                    prefixed.append(f"[Review] {line}")
            full = '\n'.join(prefixed)
            try:
                self.translator_gui.append_log(full)
            except Exception:
                print(full)

        def _stop_check():
            return self._stop_requested

        chunk_mode = self.chunk_mode_checkbox.isChecked()
        wrap_chunks = self.chunk_wrap_checkbox.isChecked()
        final_review_prompt = getattr(self, '_final_review_prompt', DEFAULT_FINAL_REVIEW_PROMPT)
        # Replace {target_lang} in final prompt too
        final_review_prompt = final_review_prompt.replace('{target_lang}', output_lang)

        # Compute batch size for parallel chunk processing
        chunk_batch_size = 1
        if chunk_mode:
            batch_on = bool(getattr(gui, 'batch_translation_var', False))
            if batch_on:
                try:
                    chunk_batch_size = int(getattr(gui, 'batch_size_var', 1))
                    if chunk_batch_size < 1:
                        chunk_batch_size = 1
                except (ValueError, TypeError):
                    chunk_batch_size = 1

        def _run():
            try:
                if chunk_mode:
                    result = generate_chunked_review(
                        epub_path=self.file_path,
                        output_dir=output_dir,
                        api_key=api_key,
                        model=model,
                        endpoint=endpoint,
                        system_prompt=system_prompt,
                        final_review_prompt=final_review_prompt,
                        input_token_limit=token_limit,
                        spoiler_mode=spoiler_mode,
                        wrap_chunks=wrap_chunks,
                        temperature=temperature,
                        config=config,
                        batch_size=chunk_batch_size,
                        log_fn=_log,
                        stop_check_fn=_stop_check,
                    )
                else:
                    result = generate_review(
                        epub_path=self.file_path,
                        output_dir=output_dir,
                        api_key=api_key,
                        model=model,
                        endpoint=endpoint,
                        system_prompt=system_prompt,
                        input_token_limit=token_limit,
                        spoiler_mode=spoiler_mode,
                        temperature=temperature,
                        config=config,
                        log_fn=_log,
                        stop_check_fn=_stop_check,
                    )
                self._review_queue.put(('done', result))
            except Exception as e:
                self._review_queue.put(('error', str(e)))

        self._review_thread = threading.Thread(target=_run, daemon=True)
        self._review_thread.start()

        # Poll the queue on the main thread every 100ms
        self._review_poll_timer = QTimer(self)
        self._review_poll_timer.setInterval(100)
        def _drain_queue():
            while not self._review_queue.empty():
                try:
                    kind, data = self._review_queue.get_nowait()
                except Exception:
                    break
                if kind == 'log':
                    self._append_log(data)
                elif kind == 'done':
                    self._review_poll_timer.stop()
                    self._on_review_done(data)
                elif kind == 'error':
                    self._review_poll_timer.stop()
                    self._on_review_done(None, data)
        self._review_poll_timer.timeout.connect(_drain_queue)
        self._review_poll_timer.start()

    def _on_generate_all(self):
        """Generate reviews for all selected EPUBs (sequential or parallel)."""
        if self._review_thread and self._review_thread.is_alive():
            return  # Already running

        if len(self._all_epub_paths) <= 1:
            self._on_start_review()
            return

        # Confirm with user
        from PySide6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Generate All Reviews")
        msg.setText(
            f"Generate reviews for all {len(self._all_epub_paths)} EPUBs?\n\n"
            "This will process each EPUB and save the review automatically."
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)
        msg.setStyleSheet("""
            QPushButton {
                min-width: 80px;
                min-height: 30px;
                padding: 6px 20px;
                font-size: 10pt;
            }
            QDialogButtonBox {
                qproperty-centerButtons: true;
            }
        """)
        if msg.exec() != QMessageBox.Yes:
            return

        self._stop_requested = False
        self._force_stop = False
        self._save_prompt_to_config()

        # Gather parameters
        gui = self.translator_gui
        api_key = gui.api_key_entry.text().strip() if hasattr(gui, 'api_key_entry') else ''
        model = getattr(gui, 'model_var', os.getenv('MODEL', 'gemini-2.0-flash'))
        endpoint = os.environ.get('ENDPOINT', '') or gui.config.get('endpoint', '')
        temperature = float(gui.config.get('translation_temperature', 0.3))
        config = dict(gui.config)

        try:
            token_limit = int(gui.token_limit_entry.text().replace(',', '').strip())
        except (ValueError, AttributeError):
            token_limit = 200000

        system_prompt_template = self.prompt_edit.toPlainText().strip()
        output_lang = getattr(gui, 'lang_var', 'English')
        system_prompt = system_prompt_template.replace('{target_lang}', output_lang)
        spoiler_mode = self.spoiler_checkbox.isChecked()
        chunk_mode = self.chunk_mode_checkbox.isChecked()
        wrap_chunks = self.chunk_wrap_checkbox.isChecked()
        final_review_prompt = getattr(self, '_final_review_prompt', DEFAULT_FINAL_REVIEW_PROMPT)
        final_review_prompt = final_review_prompt.replace('{target_lang}', output_lang)

        # Respect batch mode toggle for parallelism
        # batch_translation_var is the on/off toggle; batch_size_var is the worker count
        batch_on = bool(getattr(gui, 'batch_translation_var', False))
        if batch_on:
            try:
                batch_size = int(getattr(gui, 'batch_size_var', 1))
                if batch_size < 1:
                    batch_size = 1
            except (ValueError, TypeError):
                batch_size = 1
        else:
            batch_size = 1  # Sequential when batch mode is OFF

        # UI state
        self.start_btn.hide()
        self.generate_all_btn.hide()
        self._stop_text_label.setText("Stop All")
        self.stop_btn.show()
        self._stop_spinner_timer.start()
        self.log_field.clear()
        self._raw_review_md = ''
        self.save_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)
        self.restore_btn.hide()
        # Disable nav during review
        self._nav_prev_btn.setEnabled(False)
        self._nav_next_btn.setEnabled(False)
        self._epub_combo.setEnabled(False)

        # Clear cancellation
        try:
            from unified_api_client import UnifiedClient
            UnifiedClient.set_global_cancellation(False)
        except Exception:
            pass

        # Bug fix #2: Hijack translator_gui.append_log (same as _on_start_review)
        self._original_append_log = self.translator_gui.append_log
        self._review_log_active = True
        self._review_log_backlog = []

        def _hijacked_append_log(message):
            if self._review_log_active:
                self._review_queue.put(('log', str(message)))
                self._review_log_backlog.append(str(message))

        self.translator_gui.append_log = _hijacked_append_log

        # Message queue for thread → main-thread
        import queue
        self._review_queue = queue.Queue()

        # Hijack sys.stdout so print() from streaming/thinking goes to review dialog
        self._original_stdout = sys.stdout
        _review_queue_ref = self._review_queue
        _active_flag = self

        class _ReviewStdoutWriter:
            def write(self, text):
                if text and text.strip() and getattr(_active_flag, '_review_log_active', False):
                    _review_queue_ref.put(('log', text.strip()))
            def flush(self):
                pass

        sys.stdout = _ReviewStdoutWriter()

        # Respect the streaming toggle from settings
        stream_on = bool(self.translator_gui.config.get('enable_streaming', False))
        os.environ['ENABLE_STREAMING'] = '1' if stream_on else '0'


        all_paths = list(self._all_epub_paths)
        total = len(all_paths)

        def _get_output_dir(epub_path):
            epub_base = os.path.splitext(os.path.basename(epub_path))[0]
            override_dir = os.environ.get('OUTPUT_DIRECTORY') or config.get('output_directory')
            if override_dir:
                return os.path.join(os.path.abspath(override_dir), epub_base)
            return os.path.join(_get_app_dir(), epub_base)

        def _stop_check():
            return self._stop_requested

        def _run_all():
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _generate_single(idx, epub_path):
                basename = os.path.basename(epub_path)
                self._review_queue.put(('log', f"\n{'═'*50}\n📖 [{idx+1}/{total}] {basename}\n{'═'*50}"))
                self._review_queue.put(('nav', idx))  # Navigate dropdown to this EPUB

                if self._stop_requested:
                    self._review_queue.put(('log', f"⏹️ Skipped (stopped): {basename}"))
                    return idx, None, None

                out_dir = _get_output_dir(epub_path)
                try:
                    _log_single = lambda msg, _bn=basename: self._review_queue.put(('log', f"[{_bn}] {msg}"))
                    if chunk_mode:
                        result = generate_chunked_review(
                            epub_path=epub_path,
                            output_dir=out_dir,
                            api_key=api_key,
                            model=model,
                            endpoint=endpoint,
                            system_prompt=system_prompt,
                            final_review_prompt=final_review_prompt,
                            input_token_limit=token_limit,
                            spoiler_mode=spoiler_mode,
                            wrap_chunks=wrap_chunks,
                            temperature=temperature,
                            config=config,
                            batch_size=batch_size,
                            log_fn=_log_single,
                            stop_check_fn=_stop_check,
                        )
                    else:
                        result = generate_review(
                            epub_path=epub_path,
                            output_dir=out_dir,
                            api_key=api_key,
                            model=model,
                            endpoint=endpoint,
                            system_prompt=system_prompt,
                            input_token_limit=token_limit,
                            spoiler_mode=spoiler_mode,
                            temperature=temperature,
                            config=config,
                            log_fn=_log_single,
                            stop_check_fn=_stop_check,
                        )
                    self._review_queue.put(('log', f"✅ Done: {basename}"))
                    return idx, result, None
                except Exception as e:
                    self._review_queue.put(('log', f"❌ Error [{basename}]: {e}"))
                    return idx, None, str(e)

            workers = min(batch_size, total)
            completed = 0
            errors = 0

            if workers <= 1:
                # Sequential
                for i, path in enumerate(all_paths):
                    if self._stop_requested:
                        break
                    idx, result, err = _generate_single(i, path)
                    completed += 1
                    if err:
                        errors += 1
            else:
                # Parallel
                self._review_queue.put(('log', f"🔄 Running {workers} reviews in parallel...\n"))
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(_generate_single, i, p): i for i, p in enumerate(all_paths)}
                    for future in as_completed(futures):
                        if self._stop_requested:
                            break
                        try:
                            idx, result, err = future.result()
                            completed += 1
                            if err:
                                errors += 1
                        except Exception as e:
                            completed += 1
                            errors += 1
                            self._review_queue.put(('log', f"❌ Future error: {e}"))

            self._review_queue.put(('log', f"\n{'═'*50}\n📊 Finished: {completed}/{total} reviews ({errors} errors)\n{'═'*50}"))
            self._review_queue.put(('all_done', None))

        self._review_thread = threading.Thread(target=_run_all, daemon=True)
        self._review_thread.start()

        # Poll queue
        self._review_poll_timer = QTimer(self)
        self._review_poll_timer.setInterval(100)
        def _drain_all_queue():
            while not self._review_queue.empty():
                try:
                    kind, data = self._review_queue.get_nowait()
                except Exception:
                    break
                if kind == 'log':
                    self._append_log(data)
                elif kind == 'nav':
                    # Navigate dropdown to this EPUB index
                    try:
                        if 0 <= data < len(self._all_epub_paths):
                            self._epub_combo.blockSignals(True)
                            self._epub_combo.setCurrentIndex(data)
                            self._epub_combo.blockSignals(False)
                    except Exception:
                        pass
                elif kind == 'all_done':
                    self._review_poll_timer.stop()

                    # Bug fix #2 (cleanup): Unhijack translator_gui.append_log
                    self._unhijack_log()

                    self.stop_btn.hide()
                    self._stop_spinner_timer.stop()
                    self.start_btn.show()
                    self.generate_all_btn.show()
                    self.start_btn.setEnabled(True)
                    self.generate_all_btn.setEnabled(True)
                    self.save_btn.setEnabled(True)

                    # Sync self.file_path and nav button states to current combo selection
                    try:
                        cur_idx = self._epub_combo.currentIndex()
                        n = len(self._all_epub_paths)
                        if 0 <= cur_idx < n:
                            self.file_path = self._all_epub_paths[cur_idx]
                            self._nav_prev_btn.setEnabled(cur_idx > 0)
                            self._nav_next_btn.setEnabled(cur_idx < n - 1)
                            self._nav_counter.setText(f"{cur_idx + 1} / {n}")
                    except Exception:
                        pass
                    # Re-enable combo
                    self._epub_combo.setEnabled(True)

                    # Load the current EPUB's review into the output
                    self._load_existing_review()
                    self._update_restore_btn_visibility()

                    # Update review indicator
                    try:
                        if hasattr(self.translator_gui, '_update_review_indicator'):
                            self.translator_gui._update_review_indicator()
                    except Exception:
                        pass
        self._review_poll_timer.timeout.connect(_drain_all_queue)
        self._review_poll_timer.start()

    def _safe_delayed_reset(self, btn, text, style, delay_ms=1500):
        """Reset a button's text/style after a delay, safely handling deleted widgets."""
        def _reset():
            try:
                btn.setText(text)
                btn.setStyleSheet(style)
            except RuntimeError:
                pass
        QTimer.singleShot(delay_ms, _reset)

    def _load_remote_images(self):
        """Fetch remote images and embed them as data URIs in the HTML."""
        import re
        import base64

        original_html = self._last_rendered_html if hasattr(self, '_last_rendered_html') else None
        if not original_html:
            print('[DEBUG] _load_remote_images: no _last_rendered_html')
            return

        urls = re.findall(r'<img[^>]+src="(https?://[^"]+)"', original_html)
        if not urls:
            return

        # De-duplicate
        unique_urls = list(dict.fromkeys(urls))

        def _fetch_image(url):
            """Try to fetch a single image, returns (data, content_type) or None."""
            headers = {
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/120.0.0.0 Safari/537.36'
                ),
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': url.rsplit('/', 1)[0] + '/',
                'Sec-Fetch-Dest': 'image',
                'Sec-Fetch-Mode': 'no-cors',
                'Sec-Fetch-Site': 'cross-site',
            }
            # Try requests library first (handles redirects/cookies better)
            try:
                import requests as req_lib
                resp = req_lib.get(url, headers=headers, timeout=10, allow_redirects=True)
                if resp.status_code == 200 and len(resp.content) > 1024:
                    ct = resp.headers.get('Content-Type', '')
                    if 'image' in ct or 'octet' in ct:
                        return resp.content, ct
            except Exception:
                pass
            # Fallback to urllib
            try:
                import urllib.request
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = resp.read()
                    ct = resp.headers.get('Content-Type', '')
                if data and len(data) > 1024 and ('image' in ct or 'octet' in ct):
                    return data, ct
            except Exception:
                pass
            return None

        def _fetch_all():
            replacements = {}  # url -> data_uri
            failed_urls = []
            for url in unique_urls:
                try:
                    result = _fetch_image(url)
                    if not result:
                        failed_urls.append(url)
                        continue
                    data, content_type = result
                    # Determine MIME type from content-type header or URL extension
                    url_lower = url.lower()
                    if 'gif' in content_type or url_lower.endswith('.gif'):
                        mime = 'image/gif'
                    elif 'png' in content_type or url_lower.endswith('.png'):
                        mime = 'image/png'
                    elif 'webp' in content_type or url_lower.endswith('.webp'):
                        mime = 'image/webp'
                    elif 'svg' in content_type or url_lower.endswith('.svg'):
                        mime = 'image/svg+xml'
                    else:
                        mime = 'image/jpeg'
                    b64 = base64.b64encode(data).decode('ascii')
                    replacements[url] = f"data:{mime};base64,{b64}"
                except Exception:
                    failed_urls.append(url)  # Track for fallback link
            if replacements or failed_urls:
                print(f'[DEBUG] _fetch_all done: {len(replacements)} ok, {len(failed_urls)} failed')
                self._pending_image_data = (replacements, failed_urls)
                # Post a custom event to trigger processing on the main thread
                app = QApplication.instance()
                if app:
                    app.postEvent(self, QEvent(QEvent.Type(QEvent.User + 1)))
            else:
                print('[DEBUG] _fetch_all: nothing to do')

        threading.Thread(target=_fetch_all, daemon=True).start()

    def event(self, e):
        """Handle custom events for cross-thread image loading."""
        if e.type() == QEvent.Type(QEvent.User + 1):
            data = getattr(self, '_pending_image_data', None)
            if data:
                self._pending_image_data = None
                self._apply_image_data_uris(data[0], data[1])
            return True
        return super().event(e)

    def _show_log_context_menu(self, pos):
        """Custom context menu for the review output with image actions."""
        from PySide6.QtWidgets import QMenu
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        import re

        menu = QMenu(self.log_field)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2b2b2b; color: white;
                border: 1px solid #555; padding: 4px;
            }
            QMenu::item { padding: 6px 24px; }
            QMenu::item:selected { background-color: #4a7ba7; }
            QMenu::item:disabled { color: #888; }
            QMenu::separator { height: 1px; background: #555; margin: 4px 8px; }
        """)

        # Detect image or link under cursor
        cursor = self.log_field.cursorForPosition(pos)
        char_fmt = cursor.charFormat()
        anchor = char_fmt.anchorHref() if char_fmt.isAnchor() else ''

        # Check if it's an image anchor (original URL, not data URI)
        image_url = ''
        if anchor and anchor.startswith('http'):
            image_url = anchor
        elif not anchor:
            # Check if cursor is on an image by looking at the block's HTML
            block = cursor.block()
            if block.isValid():
                doc = self.log_field.document()
                block_html = ''
                cursor_in_block = cursor.position() - block.position()
                # Check fragments in this block for images
                it = block.begin()
                while not it.atEnd():
                    frag = it.fragment()
                    if frag.isValid():
                        fmt = frag.charFormat()
                        if fmt.isImageFormat():
                            img_name = fmt.toImageFormat().name()
                            if img_name.startswith('http'):
                                image_url = img_name
                                break
                        if fmt.isAnchor() and fmt.anchorHref().startswith('http'):
                            image_url = fmt.anchorHref()
                            break
                    it += 1

        if image_url:
            # Show full URL as disabled label
            url_label = menu.addAction(f"🔗 {image_url}")
            url_label.setEnabled(False)
            menu.addSeparator()

            # Open in browser
            open_action = menu.addAction("🌐  Open Image in Browser")
            open_action.triggered.connect(
                lambda checked=False, u=image_url: QDesktopServices.openUrl(QUrl(u))
            )

            # Copy URL
            copy_url_action = menu.addAction("📋  Copy Image URL")
            copy_url_action.triggered.connect(
                lambda checked=False, u=image_url: QApplication.clipboard().setText(u)
            )
            menu.addSeparator()

        # Standard actions
        if self.log_field.textCursor().hasSelection():
            copy_action = menu.addAction("📄  Copy")
            copy_action.setShortcut("Ctrl+C")
            copy_action.triggered.connect(self.log_field.copy)

        select_all_action = menu.addAction("🔲  Select All")
        select_all_action.setShortcut("Ctrl+A")
        select_all_action.triggered.connect(self.log_field.selectAll)

        menu.exec(self.log_field.mapToGlobal(pos))

    def _apply_image_data_uris(self, replacements: dict, failed_urls: list = None):
        """Replace remote URLs with data URIs or fallback links, then re-render."""
        import re
        try:
            html = self._last_rendered_html
            for url, data_uri in replacements.items():
                # Only replace src="url" in <img> tags, keep href="url" in <a> intact
                html = html.replace(f'src="{url}"', f'src="{data_uri}"')
            # Replace failed images with clickable links showing full URL
            for url in (failed_urls or []):
                # Match the full <a><img/></a> wrapper and replace with a URL link
                escaped = re.escape(url)
                pattern = r'<a[^>]*href="' + escaped + r'"[^>]*>\s*<img[^>]*/>\s*</a>'
                link_html = (
                    f'<a href="{url}" style="color:#5a9fd4;text-decoration:underline;word-break:break-all;">'
                    f'{url}</a>'
                )
                html = re.sub(pattern, link_html, html)
            self._last_rendered_html = html
            self.log_field.setHtml(html)
            print(f'[DEBUG] _apply_image_data_uris: re-rendered with {len(replacements)} images, {len(failed_urls or [])} fallback links')
        except RuntimeError:
            pass  # Widget may have been destroyed

    def _get_font_kwargs(self):
        """Get font settings from config for _md_to_html."""
        return {
            'font_family': self.translator_gui.config.get('review_font_family', 'Segoe UI'),
            'font_size': int(self.translator_gui.config.get('review_font_size', 9)),
            'spacing': int(self.translator_gui.config.get('review_spacing', 8)),
            'header_spacing': int(self.translator_gui.config.get('review_header_spacing', 6)),
            'line_height': int(self.translator_gui.config.get('review_line_height', 100)),
            'font_color': self.translator_gui.config.get('review_font_color', '#e0e0e0'),
            'list_gap': int(self.translator_gui.config.get('review_list_gap', 10)),
        }

    @staticmethod
    def _md_to_html(md: str, font_family: str = 'Segoe UI', font_size: int = 9, spacing: int = 8,
                    header_spacing: int = 6, line_height: int = 100, font_color: str = '#e0e0e0',
                    list_gap: int = 10) -> str:
        """Convert basic markdown to HTML for display in QTextEdit."""
        import re

        def _inline(text):
            """Apply inline markdown formatting."""
            # Images: ![alt](url) — wrap in <a> so always clickable
            text = re.sub(
                r'!\[([^\]]*)\]\((https?://[^)]+)\)',
                r'<br/><a href="\2" style="text-decoration:none;">'
                r'<img src="\2" alt="\1" style="max-width:100%;margin:6px 0;border-radius:4px;"/>'
                r'</a><br/>',
                text
            )
            # Links: [text](url)  — must come after images to avoid matching ![...]
            text = re.sub(
                r'\[([^\]]+)\]\((https?://[^)]+)\)',
                r'<a href="\2" style="color:#5a9fd4;text-decoration:underline;">\1</a>',
                text
            )
            # Bold
            text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
            # Italic
            text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
            # Inline code
            text = re.sub(r'`([^`]+)`', r'<code style="background:#2a2a2a;padding:1px 4px;border-radius:3px;font-size:95%;">\1</code>', text)
            # Bare image URLs (common extensions) not already inside tags
            text = re.sub(
                r'(?<!["\'])(?<!=)(https?://\S+\.(?:png|jpg|jpeg|gif|webp|svg|bmp))(?!["\'])',
                r'<br/><a href="\1" style="text-decoration:none;">'
                r'<img src="\1" style="max-width:100%;margin:6px 0;border-radius:4px;"/>'
                r'</a><br/>',
                text,
                flags=re.IGNORECASE
            )
            return text

        def _is_table_row(s):
            """Check if a stripped line looks like a markdown table row."""
            return s.startswith('|') and s.endswith('|') and s.count('|') >= 3

        def _is_separator_row(s):
            """Check if a stripped line is a table separator (|---|---|)."""
            return bool(re.match(r'^\|[\s:]*-{2,}[\s:]*(\|[\s:]*-{2,}[\s:]*)+\|$', s))

        def _parse_cells(s):
            """Split a table row into cell contents."""
            # Strip leading/trailing |, then split on |
            inner = s.strip('|')
            return [c.strip() for c in inner.split('|')]

        def _parse_alignment(s):
            """Parse separator row to determine column alignments."""
            cells = _parse_cells(s)
            aligns = []
            for c in cells:
                c = c.strip()
                if c.startswith(':') and c.endswith(':'):
                    aligns.append('center')
                elif c.endswith(':'):
                    aligns.append('right')
                else:
                    aligns.append('left')
            return aligns

        def _render_table(table_lines):
            """Render collected table lines into an HTML table."""
            if len(table_lines) < 2:
                # Not enough lines for a table, render as plain text
                return ''.join(
                    f'<p style="margin-top:{spacing}px; margin-bottom:{spacing}px;">{_inline(l)}</p>\n'
                    for l in table_lines
                )

            header_line = table_lines[0].strip()
            sep_line = table_lines[1].strip() if len(table_lines) > 1 else ''

            has_header = _is_separator_row(sep_line)
            aligns = _parse_alignment(sep_line) if has_header else []

            table_style = (
                'border-collapse:collapse; margin:8px 0; width:100%;'
            )
            th_style = (
                'border:1px solid #555; padding:6px 10px; '
                f'background-color:#3a4f66; color:{font_color}; font-weight:bold;'
            )
            td_style_even = (
                'border:1px solid #444; padding:5px 10px; '
                f'background-color:#2b2b2b; color:{font_color};'
            )
            td_style_odd = (
                'border:1px solid #444; padding:5px 10px; '
                f'background-color:#333333; color:{font_color};'
            )

            html = f'<table style="{table_style}">\n'

            if has_header:
                cells = _parse_cells(header_line)
                html += '<tr>'
                for ci, cell in enumerate(cells):
                    align = aligns[ci] if ci < len(aligns) else 'left'
                    html += f'<th style="{th_style} text-align:{align};">{_inline(cell)}</th>'
                html += '</tr>\n'
                data_lines = table_lines[2:]
            else:
                data_lines = table_lines

            for ri, row_line in enumerate(data_lines):
                row_line = row_line.strip()
                if not _is_table_row(row_line):
                    continue
                if _is_separator_row(row_line):
                    continue
                cells = _parse_cells(row_line)
                style = td_style_odd if ri % 2 else td_style_even
                html += '<tr>'
                for ci, cell in enumerate(cells):
                    align = aligns[ci] if ci < len(aligns) else 'left'
                    html += f'<td style="{style} text-align:{align};">{_inline(cell)}</td>'
                html += '</tr>\n'

            html += '</table>'
            return html

        lines = md.split('\n')
        html_lines = []
        in_list = False
        table_buffer = []

        i = 0
        while i < len(lines):
            stripped = lines[i].strip()

            # ── Table detection ──
            if _is_table_row(stripped):
                table_buffer.append(lines[i])
                i += 1
                continue

            # If we had a table buffer, flush it
            if table_buffer:
                # Close list before table
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append(_render_table(table_buffer))
                table_buffer = []

            # Close list if we're not on a list item
            if in_list and not stripped.startswith(('- ', '* ', '• ')):
                html_lines.append('</ul>')
                in_list = False

            # Headers
            m = re.match(r'^(#{1,6})\s+(.*)', stripped)
            if m:
                level = len(m.group(1))
                sizes = {1: '18pt', 2: '15pt', 3: '13pt', 4: '12pt', 5: '11pt', 6: '10pt'}
                sz = sizes.get(level, '10pt')
                text = _inline(m.group(2))
                html_lines.append(
                    f'<p style="font-size:{sz}; font-weight:bold; margin-top:{header_spacing + 4}px; margin-bottom:{header_spacing}px;">{text}</p>'
                )
                i += 1
                continue

            # Unordered list items
            lm = re.match(r'^[-*•]\s+(.*)', stripped)
            if lm:
                if not in_list:
                    html_lines.append(f'<ul style="margin-top:{spacing}px; margin-bottom:{spacing}px;">')
                    in_list = True
                item_text = _inline(lm.group(1))
                html_lines.append(f'<li style="margin-bottom:{list_gap}px;">{item_text}</li>')
                i += 1
                continue

            # Horizontal rule
            if re.match(r'^[-*_]{3,}\s*$', stripped):
                html_lines.append('<hr/>')
                i += 1
                continue

            # Empty line — use spacing-controlled gap
            if not stripped:
                html_lines.append(f'<div style="margin:0; padding:0; line-height:{max(spacing, 1)}px; font-size:1px;">&nbsp;</div>')
                i += 1
                continue

            # Regular paragraph — apply inline formatting
            html_lines.append(f'<p style="margin-top:{spacing}px; margin-bottom:{spacing}px;">{_inline(stripped)}</p>')
            i += 1

        # Flush any remaining table buffer
        if table_buffer:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(_render_table(table_buffer))
            table_buffer = []

        if in_list:
            html_lines.append('</ul>')

        return (f'<div style="font-family: {font_family}, sans-serif; line-height: {line_height}%; color: {font_color};">\n'
                + '\n'.join(html_lines) + '\n</div>')

    def _append_log(self, msg: str):
        """Append a message to the log field (main thread)."""
        # Use cursor to insert plain text (QTextEdit has no appendPlainText)
        cursor = self.log_field.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        if self.log_field.toPlainText():
            cursor.insertText('\n')
        cursor.insertText(msg)
        self.log_field.setTextCursor(cursor)
        # Auto-scroll to bottom
        scrollbar = self.log_field.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _unhijack_log(self):
        """Restore the original translator_gui.append_log and sys.stdout, flush backlog."""
        self._review_log_active = False
        # Restore sys.stdout first
        if hasattr(self, '_original_stdout') and self._original_stdout is not None:
            try:
                sys.stdout = self._original_stdout
                self._original_stdout = None
            except Exception:
                pass
        if hasattr(self, '_original_append_log'):
            try:
                self.translator_gui.append_log = self._original_append_log
            except Exception:
                pass
            # Flush buffered messages to main GUI log
            for msg in getattr(self, '_review_log_backlog', []):
                try:
                    self._original_append_log(msg)
                except Exception:
                    pass
            self._review_log_backlog = []

    def _on_review_done(self, result: str = None, error: str = None):
        """Called on main thread when review generation finishes."""
        self._unhijack_log()
        self.stop_btn.hide()
        self._stop_spinner_timer.stop()
        self.start_btn.show()
        self.start_btn.setEnabled(True)
        if hasattr(self, 'generate_all_btn') and len(self._all_epub_paths) > 1:
            self.generate_all_btn.show()
            self.generate_all_btn.setEnabled(True)

        # Re-enable save (was disabled at review start)
        self.save_btn.setEnabled(True)

        # Re-enable delete (its handler already shows "No Review" if nothing to delete)
        self.delete_btn.setEnabled(True)
        self._update_restore_btn_visibility()

        # Re-enable nav (may have been disabled during review)
        try:
            self._epub_combo.setEnabled(True)
            cur_idx = self._epub_combo.currentIndex()
            n = len(self._all_epub_paths)
            if 0 <= cur_idx < n:
                self.file_path = self._all_epub_paths[cur_idx]
                self._nav_prev_btn.setEnabled(cur_idx > 0)
                self._nav_next_btn.setEnabled(cur_idx < n - 1)
                self._nav_counter.setText(f"{cur_idx + 1} / {n}")
        except Exception:
            pass

        if error:
            self._append_log(f"\n❌ Error: {error}")
        elif result:
            self.delete_btn.setEnabled(True)
            self._update_restore_btn_visibility()
            self._fade_to_text(result)

            # Update the review emoji in the main GUI
            try:
                if hasattr(self.translator_gui, '_update_review_indicator'):
                    self.translator_gui._update_review_indicator()
            except Exception:
                pass
        else:
            # Fade to existing review from file (if any)
            review_path = self._get_review_path()
            if review_path and os.path.exists(review_path):
                try:
                    with open(review_path, 'r', encoding='utf-8') as f:
                        existing = f.read().strip()
                    if existing:
                        self._fade_to_text(existing)
                        self.delete_btn.setEnabled(True)
                        self._update_restore_btn_visibility()
                        return
                except Exception:
                    pass
            self._load_existing_review()

    def _fade_to_text(self, text: str):
        """Fade out the log field, swap text, fade back in."""
        # Disable start button during transition
        self.start_btn.setEnabled(False)
        self.generate_all_btn.setEnabled(False)

        # SAFETY: Always re-enable buttons after max 3 seconds, even if animation fails
        def _safety_reenable():
            try:
                if not self.start_btn.isEnabled():
                    self.start_btn.setEnabled(True)
                if hasattr(self, 'generate_all_btn') and not self.generate_all_btn.isEnabled():
                    self.generate_all_btn.setEnabled(True)
            except RuntimeError:
                pass
        QTimer.singleShot(3000, _safety_reenable)

        opacity_effect = QGraphicsOpacityEffect(self.log_field)
        self.log_field.setGraphicsEffect(opacity_effect)
        opacity_effect.setOpacity(1.0)

        fade_out = QPropertyAnimation(opacity_effect, b"opacity", self)
        fade_out.setDuration(500)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        fade_out.setEasingCurve(QEasingCurve.InQuad)

        def _swap_and_fade_in():
            try:
                self.log_field.clear()
                self._raw_review_md = text
                self._last_rendered_html = self._md_to_html(text, **self._get_font_kwargs())
                self.log_field.setHtml(self._last_rendered_html)
                self._load_remote_images()
                fade_in = QPropertyAnimation(opacity_effect, b"opacity", self)
                fade_in.setDuration(2000)
                fade_in.setStartValue(0.0)
                fade_in.setEndValue(1.0)
                fade_in.setEasingCurve(QEasingCurve.OutQuad)
                # Remove effect after fade-in
                def _cleanup():
                    try:
                        self.log_field.setGraphicsEffect(None)
                    except RuntimeError:
                        pass
                fade_in.finished.connect(_cleanup)
                self._fade_in_anim = fade_in
                fade_in.start()
            except RuntimeError:
                pass
            # Re-enable start button ALWAYS (outside try/except)
            def _reenable_btns():
                try:
                    self.start_btn.setEnabled(True)
                    self.generate_all_btn.setEnabled(True)
                except RuntimeError:
                    pass
            QTimer.singleShot(200, _reenable_btns)

        fade_out.finished.connect(_swap_and_fade_in)
        self._fade_out_anim = fade_out
        fade_out.start()

    def _on_stop(self):
        """Stop the review generation — double-click to force stop."""
        import time
        current_time = time.time()

        if not hasattr(self, '_stop_click_times'):
            self._stop_click_times = []

        self._stop_click_times.append(current_time)
        # Remove clicks older than 1 second
        self._stop_click_times = [t for t in self._stop_click_times if current_time - t < 1.0]

        # Double-click OR single click with graceful stop disabled: force immediate stop
        graceful = getattr(self.translator_gui, 'graceful_stop_var', True)
        is_force = len(self._stop_click_times) >= 2 or not graceful

        if is_force:
            self._stop_requested = True
            self._force_stop = True
            self._stop_click_times = []
            self._stop_text_label.setText("Finishing...")
            # Keep spinner running (don't stop it)
            if graceful:
                try:
                    self.translator_gui.append_log("[Review] ⚡ Double-click detected — forcing immediate stop!")
                except Exception:
                    pass
            else:
                try:
                    self.translator_gui.append_log("[Review] ⚡ Forcing immediate stop!")
                except Exception:
                    pass

            # Hard cancel all in-flight HTTP requests
            try:
                from unified_api_client import UnifiedClient
                UnifiedClient.hard_cancel_all()
            except Exception:
                pass

            # Abandon the thread — it's a daemon, let it die on its own
            self._review_thread = None

            # Stop poll timer and drain any leftover queue messages
            if hasattr(self, '_review_poll_timer'):
                self._review_poll_timer.stop()
            if hasattr(self, '_review_queue'):
                while not self._review_queue.empty():
                    try:
                        self._review_queue.get_nowait()
                    except Exception:
                        break

            # Sync file_path to current combo selection (combo was updated by nav messages)
            try:
                cur_idx = self._epub_combo.currentIndex()
                n = len(self._all_epub_paths)
                if 0 <= cur_idx < n:
                    self.file_path = self._all_epub_paths[cur_idx]
            except Exception:
                pass
            self._epub_combo.setEnabled(True)

            # Reset UI immediately and load existing review for current EPUB
            self._on_review_done(None)
            return

        # First click (graceful mode): set flag and keep UI in "Finishing..." state
        self._stop_requested = True
        self._force_stop = False
        self._stop_text_label.setText("Finishing...")
        try:
            self.translator_gui.append_log("[Review] 🛑 Graceful stop — waiting for API response to complete... (double-click to force)")
        except Exception:
            pass
        # Poll timer keeps running — _on_review_done fires when thread finishes

    def _advance_stop_spinner(self):
        """Advance the spinning icon frame."""
        if not self._stop_spinner_frames:
            return
        self._stop_icon_label.setPixmap(self._stop_spinner_frames[self._stop_spinner_idx])
        self._stop_spinner_idx = (self._stop_spinner_idx + 1) % len(self._stop_spinner_frames)

    # ─── Font Settings ─────────────────────────────────────────────

    def _toggle_font_settings(self):
        """Toggle the font settings panel visibility."""
        visible = self._font_settings_widget.isVisible()
        self._font_settings_widget.setVisible(not visible)
        self._font_settings_toggle.setText(
            "▼  Font Settings" if not visible else "▶  Font Settings"
        )

    def _reset_font_settings(self):
        """Reset all font settings to defaults with sound and animation."""
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_OK)
        except Exception:
            pass

        self._review_font_combo.setCurrentText("Segoe UI")
        self._review_font_size_spin.setValue(9)
        self._review_line_height_spin.setValue(100)
        self._review_header_spacing_spin.setValue(6)
        self._review_para_spacing_spin.setValue(8)
        self._review_list_gap_spin.setValue(10)
        self._review_font_color = '#e0e0e0'
        self._review_color_btn.setStyleSheet(
            "QPushButton { background-color: #e0e0e0; border: 1px solid #888; border-radius: 3px; }"
            "QPushButton:hover { border: 2px solid white; }"
        )
        self._on_review_font_changed()

        # Brief green flash on the button
        self._reset_font_btn.setStyleSheet(
            "QPushButton { color: white; font-size: 9pt; padding: 2px 10px; "
            "border: 1px solid #4CAF50; border-radius: 3px; background: #4CAF50; }"
        )
        QTimer.singleShot(400, lambda: self._reset_font_btn.setStyleSheet(
            "QPushButton { color: #ccc; font-size: 9pt; padding: 2px 10px; "
            "border: 1px solid #555; border-radius: 3px; background: #383838; }"
            "QPushButton:hover { background: #4a4a4a; color: white; border-color: #888; }"
        ))

    def _on_review_font_changed(self, _=None):
        """Handle font family or size change — save and re-render."""
        font_family = self._review_font_combo.currentText()
        font_size = self._review_font_size_spin.value()
        spacing = self._review_para_spacing_spin.value()
        header_spacing = self._review_header_spacing_spin.value()
        line_height = self._review_line_height_spin.value()
        list_gap = self._review_list_gap_spin.value()
        font_color = self._review_font_color

        self.translator_gui.config['review_font_family'] = font_family
        self.translator_gui.config['review_font_size'] = font_size
        self.translator_gui.config['review_spacing'] = spacing
        self.translator_gui.config['review_header_spacing'] = header_spacing
        self.translator_gui.config['review_line_height'] = line_height
        self.translator_gui.config['review_font_color'] = font_color
        self.translator_gui.config['review_list_gap'] = list_gap
        try:
            self.translator_gui.save_config(show_message=False)
        except Exception:
            pass

        # Re-render the current review with the new font
        if self._raw_review_md:
            self._last_rendered_html = self._md_to_html(
                self._raw_review_md,
                font_family=font_family,
                font_size=font_size,
                spacing=spacing,
                header_spacing=header_spacing,
                line_height=line_height,
                font_color=font_color,
                list_gap=list_gap,
            )
            self.log_field.setHtml(self._last_rendered_html)
        # Apply font size on widget level so Ctrl+/- zoom still works
        f = self.log_field.font()
        f.setPointSize(font_size)
        self.log_field.document().setDefaultFont(f)

    def _pick_review_font_color(self):
        """Open color picker and update review font color."""
        from PySide6.QtWidgets import QColorDialog
        from PySide6.QtGui import QColor
        color = QColorDialog.getColor(
            QColor(self._review_font_color),
            self,
            "Review Font Color"
        )
        if color.isValid():
            self._review_font_color = color.name()
            self._review_color_btn.setStyleSheet(
                f"QPushButton {{ background-color: {self._review_font_color}; border: 1px solid #888; border-radius: 3px; }}"
                f"QPushButton:hover {{ border: 2px solid white; }}"
            )
            self._on_review_font_changed()

    # ─── Save ────────────────────────────────────────────────────────

    def _on_save(self):
        """Save config with animated button feedback."""
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_OK)
        except Exception:
            pass
        try:
            self._sync_settings_to_gui()
            if hasattr(self.translator_gui, 'save_config'):
                self.translator_gui.save_config(show_message=False)

            # Also save review text to file if present
            review_text = (self._raw_review_md or self.log_field.toPlainText()).strip()
            if review_text:
                review_path = self._get_review_path()
                if review_path:
                    os.makedirs(os.path.dirname(review_path), exist_ok=True)
                    with open(review_path, 'w', encoding='utf-8') as f:
                        f.write(review_text)

                    # Update indicator
                    try:
                        if hasattr(self.translator_gui, '_update_review_indicator'):
                            self.translator_gui._update_review_indicator()
                    except Exception:
                        pass

            # Animate button: flash green "Saved!"
            original_text = "💾 Save"
            original_style = (
                "QPushButton { background-color: #28a745; color: white; font-weight: bold; "
                "padding: 10px 24px; border-radius: 4px; font-size: 11pt; }"
                "QPushButton:disabled { background-color: #555; color: #888; }"
            )
            self.save_btn.setText("✅ Saved!")
            self.save_btn.setStyleSheet(
                "QPushButton { background-color: #1a8f3a; color: white; font-weight: bold; "
                "padding: 10px 24px; border-radius: 4px; font-size: 11pt; }"
                "QPushButton:disabled { background-color: #555; color: #888; }"
            )
            self._safe_delayed_reset(self.save_btn, original_text, original_style)

        except Exception as e:
            self.save_btn.setText("❌ Failed")
            self.save_btn.setStyleSheet(
                "QPushButton { background-color: #dc3545; color: white; font-weight: bold; "
                "padding: 10px 24px; border-radius: 4px; font-size: 11pt; }"
                "QPushButton:disabled { background-color: #555; color: #888; }"
            )
            self._safe_delayed_reset(self.save_btn, "💾 Save",
                "QPushButton { background-color: #28a745; color: white; font-weight: bold; "
                "padding: 10px 24px; border-radius: 4px; font-size: 11pt; }"
                "QPushButton:disabled { background-color: #555; color: #888; }",
                delay_ms=2000)

    # ─── Delete ──────────────────────────────────────────────────────

    def _on_delete(self):
        """Move the review file to a backups subfolder."""
        import shutil
        from datetime import datetime

        review_path = self._get_review_path()
        if not review_path or not os.path.exists(review_path):
            # Flash "No review found" on the button
            original_text = self.delete_btn.text()
            original_style = self.delete_btn.styleSheet()
            self.delete_btn.setText("📭 No Review")
            self.delete_btn.setStyleSheet(
                "background-color: #6c757d; color: white; font-weight: bold; "
                "padding: 10px 24px; border-radius: 4px; font-size: 11pt;"
            )
            self._safe_delayed_reset(self.delete_btn, original_text, original_style)
            return

        try:
            # Create backups subfolder next to review/
            review_dir = os.path.dirname(review_path)
            backups_dir = os.path.join(review_dir, "backups")
            os.makedirs(backups_dir, exist_ok=True)

            # Timestamped backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup_name = f"review_{timestamp}.md"
            backup_path = os.path.join(backups_dir, backup_name)

            shutil.move(review_path, backup_path)

            # Clear UI
            self.log_field.clear()
            self.delete_btn.setEnabled(False)

            # Ensure start button is usable after delete
            self.start_btn.show()
            self.start_btn.setEnabled(True)
            if hasattr(self, 'generate_all_btn') and len(self._all_epub_paths) > 1:
                self.generate_all_btn.show()
                self.generate_all_btn.setEnabled(True)

            # Update the review indicator in main GUI
            try:
                if hasattr(self.translator_gui, '_update_review_indicator'):
                    self.translator_gui._update_review_indicator()
            except Exception:
                pass

            # Show restore button now that a backup exists
            self._update_restore_btn_visibility()

            # Play sound + animate button
            try:
                import winsound
                winsound.MessageBeep(winsound.MB_OK)
            except Exception:
                pass

            original_text = "🗑️ Delete"
            original_style = (
                "QPushButton { background-color: #dc3545; color: white; font-weight: bold; "
                "padding: 10px 24px; border-radius: 4px; font-size: 11pt; }"
                "QPushButton:disabled { background-color: #555; color: #888; }"
            )
            self.delete_btn.setText("✅ Moved to backups")
            self.delete_btn.setStyleSheet(
                "background-color: #6c757d; color: white; font-weight: bold; "
                "padding: 10px 24px; border-radius: 4px; font-size: 11pt;"
            )
            self._safe_delayed_reset(self.delete_btn, original_text, original_style, delay_ms=1000)

        except Exception as e:
            self._append_log(f"❌ Failed to delete review: {e}")

    # ─── Restore ─────────────────────────────────────────────────────

    def _get_backups_dir(self) -> str:
        """Get the backups directory path."""
        review_path = self._get_review_path()
        if not review_path:
            return None
        return os.path.join(os.path.dirname(review_path), "backups")

    def _update_restore_btn_visibility(self):
        """Show/hide the restore button based on whether backups exist
        and the delete button is enabled (i.e. not during review generation)."""
        try:
            backups_dir = self._get_backups_dir()
            if backups_dir and os.path.isdir(backups_dir):
                backups = [f for f in os.listdir(backups_dir) if f.endswith('.md')]
                if backups:
                    self.restore_btn.show()
                    return
            self.restore_btn.hide()
        except Exception:
            self.restore_btn.hide()

    def _on_restore(self):
        """Restore the most recent backup to review.md."""
        import shutil
        from PySide6.QtWidgets import QMessageBox

        backups_dir = self._get_backups_dir()
        if not backups_dir or not os.path.isdir(backups_dir):
            return

        try:
            # Find the most recent backup
            backups = sorted(
                [f for f in os.listdir(backups_dir) if f.endswith('.md')],
                reverse=True
            )
            if not backups:
                return

            latest = os.path.join(backups_dir, backups[0])
            review_path = self._get_review_path()
            if not review_path:
                return

            # Warn if there's a current review.md that would be overwritten
            if os.path.exists(review_path):
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Restore Backup")
                msg.setText(
                    "Your current review will be overwritten.\n\n"
                    "Restoring will replace it with the previous backup:\n"
                    f"{backups[0]}\n\n"
                    "Are you sure you want to restore?"
                )
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg.setDefaultButton(QMessageBox.No)
                msg.setStyleSheet("""
                    QPushButton {
                        min-width: 80px;
                        min-height: 30px;
                        padding: 6px 20px;
                        font-size: 10pt;
                    }
                    QDialogButtonBox {
                        qproperty-centerButtons: true;
                    }
                """)
                if msg.exec() != QMessageBox.Yes:
                    return

            os.makedirs(os.path.dirname(review_path), exist_ok=True)
            shutil.copy2(latest, review_path)

            # Load restored content into UI
            with open(review_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self._raw_review_md = content
            self._last_rendered_html = self._md_to_html(content, **self._get_font_kwargs())
            self.log_field.setHtml(self._last_rendered_html)
            self._load_remote_images()
            self.delete_btn.setEnabled(True)

            # Update the review indicator in main GUI
            try:
                if hasattr(self.translator_gui, '_update_review_indicator'):
                    self.translator_gui._update_review_indicator()
            except Exception:
                pass

            # Play sound + animate
            try:
                import winsound
                winsound.MessageBeep(winsound.MB_OK)
            except Exception:
                pass

            original_text = self.restore_btn.text()
            original_style = self.restore_btn.styleSheet()
            self.restore_btn.setText("✅ Restored!")
            self.restore_btn.setStyleSheet(
                "background-color: #1a8f3a; color: white; font-weight: bold; "
                "padding: 10px 24px; border-radius: 4px; font-size: 11pt;"
            )
            self._safe_delayed_reset(self.restore_btn, original_text, original_style)

        except Exception as e:
            self._append_log(f"❌ Failed to restore: {e}")

    def closeEvent(self, event):
        """Hide instead of close to preserve state. Generation continues in background."""
        self._save_prompt_to_config()
        event.ignore()
        self.hide()

    def keyPressEvent(self, event):
        """F11 toggles fullscreen."""
        if event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            super().keyPressEvent(event)
