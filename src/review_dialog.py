# review_dialog.py
"""
Review Dialog — A separate Qt dialog for generating EPUB reviews.
Launched from the "Generate Review" button in translator_gui.py.
"""

import os
import threading
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QPlainTextEdit, QCheckBox, QApplication, QGroupBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QIcon

from review_generator import (
    DEFAULT_REVIEW_PROMPT,
    count_epub_tokens,
    generate_review,
)


class ReviewDialog(QDialog):
    """Dialog for generating an AI-powered review/summary of an EPUB."""

    def __init__(self, parent, translator_gui, file_path: str):
        super().__init__(parent)
        self.translator_gui = translator_gui
        self.file_path = file_path
        self._stop_requested = False
        self._review_thread = None
        self._counting = False

        self.setWindowTitle("Generate Review")
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        # Sizing
        screen = QApplication.primaryScreen().availableGeometry()
        width = int(screen.width() * 0.40)
        height = int(screen.height() * 0.55)
        self.setMinimumSize(width, height)
        self.resize(width, height)

        # Icon
        try:
            base_dir = getattr(translator_gui, 'base_dir', os.getcwd())
            icon_path = os.path.join(base_dir, 'Halgakos.ico')
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
        except Exception:
            pass

        self._build_ui()
        self._load_saved_prompt()
        self._load_existing_review()
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

        # Header
        header = QLabel(f"📖 Review: {os.path.basename(self.file_path)}")
        header_font = QFont()
        header_font.setPointSize(12)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setWordWrap(True)
        layout.addWidget(header)

        # ── 1. System Prompt ──
        prompt_group = QGroupBox("System Prompt")
        prompt_layout = QVBoxLayout(prompt_group)
        prompt_layout.setContentsMargins(8, 8, 8, 8)
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("Enter the system prompt for the review...")
        self.prompt_edit.setMaximumHeight(120)
        prompt_layout.addWidget(self.prompt_edit)
        layout.addWidget(prompt_group)

        # ── 2. Log / Summary field ──
        log_group = QGroupBox("Review Output")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(8, 8, 8, 8)
        self.log_field = QPlainTextEdit()
        self.log_field.setReadOnly(True)
        self.log_field.setPlaceholderText("Generated review will appear here...")
        log_layout.addWidget(self.log_field)
        layout.addWidget(log_group, stretch=1)

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

        # 5. Token count
        self.token_label = QLabel("⏳ Counting tokens...")
        self.token_label.setStyleSheet("color: #94a3b8; font-size: 10pt;")
        controls_layout.addWidget(self.token_label)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # ── Button row ──
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)

        # 4. Start Review
        self.start_btn = QPushButton("🚀 Start Review")
        self.start_btn.setStyleSheet(
            "background-color: #4a7ba7; color: white; font-weight: bold; "
            "padding: 8px 16px; border-radius: 4px; font-size: 10pt;"
        )
        self.start_btn.setMinimumWidth(140)
        self.start_btn.clicked.connect(self._on_start_review)
        button_layout.addWidget(self.start_btn)

        # Stop button (hidden initially)
        self.stop_btn = QPushButton("🛑 Stop")
        self.stop_btn.setStyleSheet(
            "background-color: #dc3545; color: white; font-weight: bold; "
            "padding: 8px 16px; border-radius: 4px; font-size: 10pt;"
        )
        self.stop_btn.setMinimumWidth(100)
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.hide()
        button_layout.addWidget(self.stop_btn)

        button_layout.addStretch()

        # 6. Save
        self.save_btn = QPushButton("💾 Save")
        self.save_btn.setStyleSheet(
            "background-color: #28a745; color: white; font-weight: bold; "
            "padding: 8px 16px; border-radius: 4px; font-size: 10pt;"
        )
        self.save_btn.setMinimumWidth(100)
        self.save_btn.clicked.connect(self._on_save)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)

        # 7. Close
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            "padding: 8px 16px; border-radius: 4px; font-size: 10pt;"
        )
        close_btn.setMinimumWidth(80)
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    # ─── Config / persistence ────────────────────────────────────────

    def _load_saved_prompt(self):
        """Load previously saved review prompt from config, or use default."""
        config = getattr(self.translator_gui, 'config', {})
        saved = config.get('review_system_prompt', '')
        self.prompt_edit.setPlainText(saved if saved else DEFAULT_REVIEW_PROMPT)

    def _save_prompt_to_config(self):
        """Persist the current system prompt to config."""
        try:
            config = getattr(self.translator_gui, 'config', {})
            config['review_system_prompt'] = self.prompt_edit.toPlainText()
            if hasattr(self.translator_gui, 'save_config'):
                self.translator_gui.save_config(show_message=False)
        except Exception:
            pass

    def _load_existing_review(self):
        """If a review already exists, load it into the log field."""
        review_path = self._get_review_path()
        if review_path and os.path.exists(review_path):
            try:
                with open(review_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content.strip():
                    self.log_field.setPlainText(content)
                    self.save_btn.setEnabled(True)
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
                output_dir = os.path.join(os.getcwd(), epub_base)
            return os.path.join(output_dir, "review", "review.md")
        except Exception:
            return None

    # ─── Token counting (background thread) ──────────────────────────

    def _start_token_count(self):
        """Count EPUB tokens in a background thread."""
        self._counting = True
        self.token_label.setText("⏳ Counting tokens...")
        self.token_label.setStyleSheet("color: #f59e0b; font-size: 10pt;")

        def _count():
            try:
                total = count_epub_tokens(self.file_path)
                QTimer.singleShot(0, lambda: self._on_token_count_done(total))
            except Exception as e:
                QTimer.singleShot(0, lambda: self._on_token_count_done(-1, str(e)))

        t = threading.Thread(target=_count, daemon=True)
        t.start()

    def _on_token_count_done(self, total: int, error: str = None):
        """Called on main thread when token counting finishes."""
        self._counting = False
        if error:
            self.token_label.setText(f"⚠️ Token count failed: {error}")
            self.token_label.setStyleSheet("color: #ef4444; font-size: 10pt;")
        elif total >= 0:
            self.token_label.setText(f"📊 File tokens: {total:,}")
            self.token_label.setStyleSheet("color: #22c55e; font-size: 10pt;")
        else:
            self.token_label.setText("⚠️ Could not count tokens")
            self.token_label.setStyleSheet("color: #ef4444; font-size: 10pt;")

    # ─── Review generation ───────────────────────────────────────────

    def _on_start_review(self):
        """Start the review generation in a background thread."""
        if self._review_thread and self._review_thread.is_alive():
            return  # Already running

        self._stop_requested = False
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
            token_limit = int(gui.token_limit_entry.text().strip())
        except (ValueError, AttributeError):
            token_limit = 200000

        system_prompt = self.prompt_edit.toPlainText().strip()
        spoiler_mode = self.spoiler_checkbox.isChecked()

        # Determine output directory
        epub_base = os.path.splitext(os.path.basename(self.file_path))[0]
        override_dir = os.environ.get('OUTPUT_DIRECTORY') or config.get('output_directory')
        if override_dir:
            output_dir = os.path.join(os.path.abspath(override_dir), epub_base)
        else:
            output_dir = os.path.join(os.getcwd(), epub_base)

        # UI state
        self.start_btn.hide()
        self.stop_btn.show()
        self.log_field.clear()
        self.save_btn.setEnabled(False)

        def _log(msg):
            QTimer.singleShot(0, lambda: self._append_log(msg))

        def _stop_check():
            return self._stop_requested

        def _run():
            try:
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
                QTimer.singleShot(0, lambda: self._on_review_done(result))
            except Exception as e:
                QTimer.singleShot(0, lambda: self._on_review_done(None, str(e)))

        self._review_thread = threading.Thread(target=_run, daemon=True)
        self._review_thread.start()

    def _append_log(self, msg: str):
        """Append a message to the log field (main thread)."""
        self.log_field.appendPlainText(msg)
        # Auto-scroll to bottom
        scrollbar = self.log_field.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_review_done(self, result: str = None, error: str = None):
        """Called on main thread when review generation finishes."""
        self.stop_btn.hide()
        self.start_btn.show()

        if error:
            self._append_log(f"\n❌ Error: {error}")
        elif result:
            # Clear log and show only the review
            self.log_field.clear()
            self.log_field.setPlainText(result)
            self.save_btn.setEnabled(True)
            self._append_log("")  # Scroll trigger

            # Update the review emoji in the main GUI
            try:
                if hasattr(self.translator_gui, '_update_review_indicator'):
                    self.translator_gui._update_review_indicator()
            except Exception:
                pass
        else:
            self._append_log("\n⚠️ No review generated.")

    def _on_stop(self):
        """Stop the review generation."""
        self._stop_requested = True
        self._append_log("🛑 Stopping...")

    # ─── Save ────────────────────────────────────────────────────────

    def _on_save(self):
        """Save the current review text to the review subfolder."""
        review_text = self.log_field.toPlainText().strip()
        if not review_text:
            return

        review_path = self._get_review_path()
        if not review_path:
            self._append_log("⚠️ Could not determine save path")
            return

        try:
            os.makedirs(os.path.dirname(review_path), exist_ok=True)
            with open(review_path, 'w', encoding='utf-8') as f:
                f.write(review_text)
            self._append_log(f"💾 Saved to: {review_path}")

            # Update indicator
            try:
                if hasattr(self.translator_gui, '_update_review_indicator'):
                    self.translator_gui._update_review_indicator()
            except Exception:
                pass
        except Exception as e:
            self._append_log(f"❌ Failed to save: {e}")

    def closeEvent(self, event):
        """Save prompt on close."""
        self._stop_requested = True
        self._save_prompt_to_config()
        super().closeEvent(event)
