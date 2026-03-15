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
        self._force_stop = False
        self._review_thread = None
        self._counting = False

        self.setWindowTitle("Generate Review")
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        # Sizing
        screen = QApplication.primaryScreen().availableGeometry()
        width = int(screen.width() * 0.40)
        height = int(screen.height() * 0.65)
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
        self.prompt_edit.setMaximumHeight(160)
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
            "QPushButton { background-color: #4a7ba7; color: white; font-weight: bold; "
            "padding: 8px 16px; border-radius: 4px; font-size: 10pt; }"
            "QPushButton:disabled { background-color: #3a3a3a; color: #6a6a6a; border: 1px solid #4a4a4a; }"
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
            "padding: 10px 24px; border-radius: 4px; font-size: 11pt;"
        )
        self.save_btn.setMinimumWidth(120)
        self.save_btn.clicked.connect(self._on_save)
        button_layout.addWidget(self.save_btn)

        # Delete Review
        self.delete_btn = QPushButton("🗑️ Delete")
        self.delete_btn.setStyleSheet(
            "background-color: #dc3545; color: white; font-weight: bold; "
            "padding: 10px 24px; border-radius: 4px; font-size: 11pt;"
        )
        self.delete_btn.setMinimumWidth(120)
        self.delete_btn.clicked.connect(self._on_delete)
        self.delete_btn.setEnabled(False)
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

    # ─── Config / persistence ────────────────────────────────────────

    def _load_saved_prompt(self):
        """Load previously saved review prompt and spoiler mode from config."""
        config = getattr(self.translator_gui, 'config', {})
        saved = config.get('review_system_prompt', '')
        self.prompt_edit.setPlainText(saved if saved else DEFAULT_REVIEW_PROMPT)
        # Load spoiler mode
        spoiler = config.get('review_spoiler_mode', False)
        self.spoiler_checkbox.setChecked(bool(spoiler))

    def _sync_settings_to_gui(self):
        """Sync current dialog values to translator_gui _var attributes for save_config."""
        gui = self.translator_gui
        gui.review_system_prompt_var = self.prompt_edit.toPlainText().strip()
        gui.review_spoiler_mode_var = self.spoiler_checkbox.isChecked()

    def _save_prompt_to_config(self):
        """Persist the current system prompt and spoiler mode to config."""
        try:
            self._sync_settings_to_gui()
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
                    self.delete_btn.setEnabled(True)
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
                QTimer.singleShot(0, lambda t=total: self._on_token_count_done(t))
            except Exception as e:
                err_msg = str(e)
                QTimer.singleShot(0, lambda msg=err_msg: self._on_token_count_done(-1, msg))

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
            output_dir = os.path.join(os.getcwd(), epub_base)

        # UI state
        self.start_btn.hide()
        self.stop_btn.setText("🛑 Stop")
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

    def _safe_delayed_reset(self, btn, text, style, delay_ms=1500):
        """Reset a button's text/style after a delay, safely handling deleted widgets."""
        def _reset():
            try:
                btn.setText(text)
                btn.setStyleSheet(style)
            except RuntimeError:
                pass
        QTimer.singleShot(delay_ms, _reset)

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
            self.delete_btn.setEnabled(True)
            self._update_restore_btn_visibility()
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
        """Stop the review generation — double-click to force stop."""
        import time
        current_time = time.time()

        if not hasattr(self, '_stop_click_times'):
            self._stop_click_times = []

        self._stop_click_times.append(current_time)
        # Remove clicks older than 1 second
        self._stop_click_times = [t for t in self._stop_click_times if current_time - t < 1.0]

        # Double-click: force immediate stop
        if len(self._stop_click_times) >= 2:
            self._stop_requested = True
            self._force_stop = True
            self._stop_click_times = []
            self.stop_btn.setText("⚡ Force Stopped")
            self._append_log("⚡ Double-click detected — forcing immediate stop!")

            # Kill the thread if still running
            if self._review_thread and self._review_thread.is_alive():
                # Thread is daemon so it will be cleaned up, but trigger UI reset
                QTimer.singleShot(500, lambda: self._on_review_done(None))
            return

        # First click: graceful stop
        graceful = getattr(self.translator_gui, 'graceful_stop_var', True)
        self._stop_requested = True
        self._force_stop = False

        if graceful:
            self.stop_btn.setText("🛑 Finishing...")
            self._append_log("🛑 Graceful stop — waiting for API response to complete... (double-click to force)")
        else:
            self._append_log("🛑 Stopping...")

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
            review_text = self.log_field.toPlainText().strip()
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
            original_text = self.save_btn.text()
            original_style = self.save_btn.styleSheet()
            self.save_btn.setText("✅ Saved!")
            self.save_btn.setStyleSheet(
                "background-color: #1a8f3a; color: white; font-weight: bold; "
                "padding: 10px 24px; border-radius: 4px; font-size: 11pt;"
            )
            self._safe_delayed_reset(self.save_btn, original_text, original_style)

        except Exception as e:
            self.save_btn.setText("❌ Failed")
            self.save_btn.setStyleSheet(
                "background-color: #dc3545; color: white; font-weight: bold; "
                "padding: 10px 24px; border-radius: 4px; font-size: 11pt;"
            )
            self._safe_delayed_reset(self.save_btn, "💾 Save",
                "background-color: #28a745; color: white; font-weight: bold; "
                "padding: 10px 24px; border-radius: 4px; font-size: 11pt;",
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
            self.delete_btn.setText("⚠️ No review found")
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"review_{timestamp}.md"
            backup_path = os.path.join(backups_dir, backup_name)

            shutil.move(review_path, backup_path)

            # Clear UI
            self.log_field.clear()
            self.delete_btn.setEnabled(False)

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

            original_text = self.delete_btn.text()
            original_style = self.delete_btn.styleSheet()
            self.delete_btn.setText("✅ Moved to backups")
            self.delete_btn.setStyleSheet(
                "background-color: #6c757d; color: white; font-weight: bold; "
                "padding: 10px 24px; border-radius: 4px; font-size: 11pt;"
            )
            self._safe_delayed_reset(self.delete_btn, original_text, original_style, delay_ms=2000)

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
        """Show/hide the restore button based on whether backups exist."""
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

            os.makedirs(os.path.dirname(review_path), exist_ok=True)
            shutil.copy2(latest, review_path)

            # Load restored content into UI
            with open(review_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.log_field.setPlainText(content)
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
        """Save prompt on close."""
        self._stop_requested = True
        self._save_prompt_to_config()
        super().closeEvent(event)
