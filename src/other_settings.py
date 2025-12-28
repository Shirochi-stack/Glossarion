"""Other Settings Dialog Methods for Glossarion

This module contains all the methods related to the "Other Settings" dialog.
These methods are dynamically injected into the TranslatorGUI class.
"""

# Standard library imports
import os
import json
import re
import sys

# PySide6 imports (fully migrated from Tkinter)
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QGroupBox,
    QMessageBox,
    QScrollArea,
    QWidget,
    QGridLayout,
    QFrame,
    QCheckBox,
    QComboBox,
    QLineEdit,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap

# Local imports - these will be available through the TranslatorGUI instance
# WindowManager and UIHelper removed - not needed in PySide6
from translator_gui import CONFIG_FILE
from ai_hunter_enhanced import AIHunterConfigGUI
# Bring in backup management methods
from config_backup import (
    _backup_config_file,
    _restore_config_from_backup,
    _create_manual_config_backup,
    _open_backup_folder,
    _manual_restore_config,
)


def setup_other_settings_methods(gui_instance):
    """Inject all other settings methods into the GUI instance"""
    import types
    import sys
    
    # Get this module
    current_module = sys.modules[__name__]
    
    # List of all method names to bind
    methods_to_bind = [
        # Core profile methods (needed at GUI init)
        'on_profile_select', 'save_profile', 'delete_profile', 'save_profiles',
        'import_profiles', 'export_profiles',
        # Other settings methods
        'configure_rolling_summary_prompts', 'toggle_thinking_budget', 
        'toggle_gpt_reasoning_controls', 'open_other_settings',
        'open_multi_api_key_manager', 'show_ai_hunter_settings',
        'delete_translated_headers_file', 'run_standalone_translate_headers', 'validate_epub_structure_gui',
        'show_header_help_dialog',
        'on_extraction_method_change', 'on_extraction_mode_change',
        # Toggle methods
        'toggle_extraction_workers', 'toggle_gemini_endpoint', 'toggle_ai_hunter',
        'toggle_custom_endpoint_ui', 'toggle_more_endpoints',
        '_toggle_multi_key_setting', '_toggle_http_tuning_controls',
        '_toggle_anti_duplicate_controls', 'toggle_image_translation_section',
        'toggle_anti_duplicate_section',
        # Provider autocomplete methods
        '_setup_provider_combobox_bindings', '_on_provider_combo_keyrelease',
        '_commit_provider_autocomplete', '_scroll_provider_list_to_value',
        '_validate_provider_selection',
        # Section creation methods
        '_create_context_management_section', '_create_response_handling_section',
        '_create_prompt_management_section', '_create_processing_options_section',
        '_create_image_translation_section', '_create_anti_duplicate_section',
        '_create_custom_api_endpoints_section', '_create_debug_controls_section',
        # Helper methods
        '_create_multi_key_row', '_create_manual_config_backup', '_manual_restore_config',
        '_open_backup_folder', '_backup_config_file', '_restore_config_from_backup',
        '_check_azure_endpoint', '_update_azure_api_version_env',
        '_reset_anti_duplicate_defaults', '_get_ai_hunter_status_text',
        'create_ai_hunter_section', 'test_api_connections',
        '_update_multi_key_status_label',
        # Prompt configuration dialogs
        'configure_translation_chunk_prompt', 'configure_image_chunk_prompt',
        'configure_image_compression',
        # Helper methods for styling
        '_create_styled_checkbox', '_disable_combobox_mousewheel', '_disable_spinbox_mousewheel',
        '_add_combobox_arrow'
    ]
    
    # Bind each method to the GUI instance
    for method_name in methods_to_bind:
        if hasattr(current_module, method_name):
            method = getattr(current_module, method_name)
            if callable(method):
                setattr(gui_instance, method_name, types.MethodType(method, gui_instance))


def _center_messagebox_buttons(msg_box):
    """Helper to center buttons in a QMessageBox"""
    from PySide6.QtWidgets import QDialogButtonBox
    button_box = msg_box.findChild(QDialogButtonBox)
    if button_box:
        button_box.setCenterButtons(True)

def _create_styled_checkbox(self, text):
    """Create a checkbox with proper checkmark using text overlay - from manga integration"""
    from PySide6.QtWidgets import QCheckBox, QLabel
    from PySide6.QtCore import Qt, QTimer
    
    checkbox = QCheckBox(text)
    # Don't set inline stylesheet - use the global stylesheet from container
    
    # Create checkmark overlay
    checkmark = QLabel("✓", checkbox)
    checkmark.setStyleSheet("""
        QLabel {
            color: white;
            background: transparent;
            font-weight: bold;
            font-size: 11px;
        }
    """)
    checkmark.setAlignment(Qt.AlignCenter)
    checkmark.hide()
    checkmark.setAttribute(Qt.WA_TransparentForMouseEvents)
    
    def position_checkmark():
        try:
            # Check if checkmark still exists and is valid
            if checkmark and not checkmark.isHidden() or True:  # Always try to set geometry
                checkmark.setGeometry(2, 1, 14, 14)
        except RuntimeError:
            # Widget was already deleted
            pass
    
    def update_checkmark():
        try:
            # Check if both widgets still exist
            if checkbox and checkmark:
                if checkbox.isChecked():
                    position_checkmark()
                    checkmark.show()
                else:
                    checkmark.hide()
        except RuntimeError:
            # Widget was already deleted
            pass
    
    checkbox.stateChanged.connect(update_checkmark)
    
    # Use try-except to handle case where widgets are deleted before timer fires
    def safe_init():
        try:
            position_checkmark()
            update_checkmark()
        except RuntimeError:
            pass
    
    QTimer.singleShot(0, safe_init)
    
    return checkbox

def _disable_combobox_mousewheel(self, combobox):
    """Disable mousewheel scrolling on a combobox (PySide6)"""
    combobox.wheelEvent = lambda event: None

def _disable_spinbox_mousewheel(self, spinbox):
    """Disable mousewheel scrolling on a spinbox (PySide6)"""
    spinbox.wheelEvent = lambda event: None

def _add_combobox_arrow(self, combobox):
    """Add a unicode arrow overlay to a combobox"""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QLabel
    from PySide6.QtCore import Qt
    
    arrow_label = QLabel("▼", combobox)
    arrow_label.setStyleSheet("""
        QLabel {
            color: white;
            background: transparent;
            font-size: 10pt;
            border: none;
        }
    """)
    arrow_label.setAlignment(Qt.AlignCenter)
    arrow_label.setAttribute(Qt.WA_TransparentForMouseEvents)
    
    def position_arrow():
        try:
            if arrow_label and combobox:
                width = combobox.width()
                height = combobox.height()
                arrow_label.setGeometry(width - 20, (height - 16) // 2, 20, 16)
        except RuntimeError:
            pass
    
    # Position arrow when combobox is resized
    original_resize = combobox.resizeEvent
    def new_resize(event):
        original_resize(event)
        position_arrow()
    
    combobox.resizeEvent = new_resize
    
    # Initial position
    QTimer.singleShot(0, position_arrow)


class HeaderTranslationHelpDialog(QDialog):
    """Dialog to display detailed information about header translation functionality"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Header Translation - Help")
        self.setup_ui()
        
        # Set icon if available
        icon_path = os.path.join(os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__)), "Halgakos.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
    
    def setup_ui(self):
        """Set up the dialog UI using ratios for sizing"""
        # Use ratios for dialog size based on screen resolution
        from PySide6.QtWidgets import QApplication
        screen = QApplication.primaryScreen().availableGeometry()
        width = int(screen.width() * 0.4)  # 40% of screen width
        height = int(screen.height() * 0.6)  # 60% of screen height
        self.resize(width, height)
        
        # Center the dialog
        self.move(
            screen.x() + (screen.width() - width) // 2,
            screen.y() + (screen.height() - height) // 2
        )
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            int(width * 0.03),  # 3% of dialog width
            int(height * 0.02),  # 2% of dialog height
            int(width * 0.03),  # 3% of dialog width
            int(height * 0.02)   # 2% of dialog height
        )
        layout.setSpacing(int(height * 0.02))  # 2% of dialog height
        
        # Title
        title_label = QLabel("Chapter Header Translation - Detailed Guide")
        title_label.setStyleSheet(f"""
            QLabel {{
                font-weight: bold;
                font-size: {int(height * 0.025)}pt;
                color: #6c7b7f;
                padding-bottom: {int(height * 0.015)}px;
            }}
        """)
        layout.addWidget(title_label)
        
        # Scrollable content area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(
            int(width * 0.02),  # 2% margins
            int(height * 0.01), 
            int(width * 0.02), 
            int(height * 0.01)
        )
        content_layout.setSpacing(int(height * 0.015))  # 1.5% spacing
        
        # Create sections with detailed explanations
        sections = [
            {
                "title": "🔄 Translation Modes",
                "content": [
                    "• OFF: Use existing headers from already translated chapters",
                    "• ON: Extract all headers → Translate in batch → Update files"
                ]
            },
            {
                "title": "⚙️ Options Explained",
                "content": [
                    "• Update headers in HTML files: Modifies the actual chapter files with translated headers",
                    "• Save translations to .txt: Creates backup files with translation mappings",
                    "• Headers per batch: Number of headers to translate simultaneously (affects API usage)"
                ]
            },
            {
                "title": "🚫 Ignore Options",
                "content": [
                    "• Ignore header: Skip h1/h2/h3 tags (prevents re-translation of visible headers)",
                    "• Ignore title: Skip <title> tag (prevents re-translation of document titles)"
                ]
            },
            {
                "title": "⚠️ Fallback System",
                "content": [
                    "• Use Sorted Fallback: If OPF-based matching fails, use sorted index matching",
                    "• WARNING: Less accurate - may mismatch chapters if file order differs from OPF spine",
                    "• Only use if you're experiencing matching issues with standard mode"
                ]
            },
            {
                "title": "📂 Standalone Mode",
                "content": [
                    "• Uses content.opf-based exact mapping for precise chapter matching",
                    "• Translates chapters with matching names (ignores 'response_' prefix and extensions)",
                    "• The regular translation logic uses this logic as well"
                ]
            },
            {
                "title": "🗑️ File Management",
                "content": [
                    "• Delete Header Files: Removes translated_headers.txt files for all selected EPUBs",
                    "• Use this to reset translation state or clean up after testing",
                    "• Safe operation - only removes translation cache files, not original content"
                ]
            },
            {
                "title": "💡 Best Practices",
                "content": [
                    "• Test with a small batch first to verify settings work correctly",
                    "• Enable 'Save translations to .txt' for backup and debugging",
                    "• Use 'Ignore header' if chapters already have translated visible titles",
                    "• Keep 'Headers per batch' moderate to be within your output token limit"
                ]
            }
        ]
        
        font_size = max(9, int(height * 0.018))  # Scale font with dialog size, minimum 9pt
        
        for section in sections:
            # Section title
            section_title = QLabel(section["title"])
            section_title.setStyleSheet(f"""
                QLabel {{
                    font-weight: bold;
                    font-size: {font_size + 1}pt;
                    color: #7f8c8d;
                    padding-top: {int(height * 0.01)}px;
                    padding-bottom: {int(height * 0.005)}px;
                }}
            """)
            content_layout.addWidget(section_title)
            
            # Section content
            section_text = "\n".join(section["content"])
            section_label = QLabel(section_text)
            section_label.setStyleSheet(f"""
                QLabel {{
                    font-size: {font_size}pt;
                    color: #95a5a6;
                    line-height: 1.4;
                    padding-left: {int(width * 0.02)}px;
                    padding-bottom: {int(height * 0.01)}px;
                }}
            """)
            section_label.setWordWrap(True)
            content_layout.addWidget(section_label)
        
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(int(height * 0.05))  # 5% of dialog height
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #3498db;
                color: white;
                padding: {int(height * 0.01)}px {int(width * 0.03)}px;
                border-radius: {int(height * 0.01)}px;
                font-weight: bold;
                font-size: {font_size}pt;
            }}
            QPushButton:hover {{
                background-color: #2980b9;
            }}
            QPushButton:pressed {{
                background-color: #21618c;
            }}
        """)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)


def show_header_help_dialog(self):
    """Show the header translation help dialog"""
    # Use the current GUI instance as parent, fallback to None if needed
    parent = getattr(self, '_other_settings_dialog', None) or self
    dialog = HeaderTranslationHelpDialog(parent)
    dialog.exec()

def configure_rolling_summary_prompts(self):
    """Configure rolling summary prompts (PySide6)"""
    from PySide6.QtGui import QIcon
    
    # Create a non-modal dialog
    from PySide6.QtWidgets import QApplication
    dialog = QDialog(None)
    dialog.setWindowTitle("Configure Memory System Prompts")
    # Use screen ratios for sizing
    screen = QApplication.primaryScreen().geometry()
    width = int(screen.width() * 0.42)  # 42% of screen width
    height = int(screen.height() * 0.75)  # 75% of screen height
    dialog.resize(width, height)
    
    # Set icon
    try:
        dialog.setWindowIcon(QIcon("halgakos.ico"))
    except Exception:
        pass

    # Keep a reference so it isn't garbage-collected
    self._rolling_summary_dialog = dialog

    layout = QVBoxLayout(dialog)

    # Title and description
    title_lbl = QLabel("Memory System Configuration")
    title_lbl.setStyleSheet("font-size: 16px; font-weight: bold;")
    layout.addWidget(title_lbl)

    desc_lbl = QLabel("Configure how the AI creates and maintains translation memory/context summaries.")
    desc_lbl.setStyleSheet("color: gray;")
    layout.addWidget(desc_lbl)

    # System Prompt group
    sys_group = QGroupBox("System Prompt (Role Definition)")
    sys_v = QVBoxLayout(sys_group)

    sys_help = QLabel("Defines the AI's role and behavior when creating summaries")
    sys_help.setStyleSheet("color: #1f6feb;")
    sys_v.addWidget(sys_help)

    self.summary_system_text = QTextEdit()
    self.summary_system_text.setAcceptRichText(False)
    self.summary_system_text.setPlainText(getattr(self, 'rolling_summary_system_prompt', ''))
    sys_v.addWidget(self.summary_system_text)

    layout.addWidget(sys_group)

    # User Prompt group
    user_group = QGroupBox("User Prompt Template")
    user_v = QVBoxLayout(user_group)

    user_help = QLabel("Template for summary requests. Use {translations} for content placeholder")
    user_help.setStyleSheet("color: #1f6feb;")
    user_v.addWidget(user_help)

    self.summary_user_text = QTextEdit()
    self.summary_user_text.setAcceptRichText(False)
    self.summary_user_text.setPlainText(getattr(self, 'rolling_summary_user_prompt', ''))
    user_v.addWidget(self.summary_user_text)

    layout.addWidget(user_group)

    # Buttons row
    btn_row = QHBoxLayout()

    def _save_prompts():
        self.rolling_summary_system_prompt = self.summary_system_text.toPlainText().strip()
        self.rolling_summary_user_prompt = self.summary_user_text.toPlainText().strip()

        self.config['rolling_summary_system_prompt'] = self.rolling_summary_system_prompt
        self.config['rolling_summary_user_prompt'] = self.rolling_summary_user_prompt

        os.environ['ROLLING_SUMMARY_SYSTEM_PROMPT'] = self.rolling_summary_system_prompt
        os.environ['ROLLING_SUMMARY_USER_PROMPT'] = self.rolling_summary_user_prompt

        QMessageBox.information(dialog, "Success", "Memory prompts saved!")
        dialog.close()

    def _reset_prompts():
        res = QMessageBox.question(
            dialog,
            "Reset Prompts",
            "Reset memory prompts to defaults?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if res == QMessageBox.Yes:
            self.summary_system_text.setPlainText(getattr(self, 'default_rolling_summary_system_prompt', ''))
            self.summary_user_text.setPlainText(getattr(self, 'default_rolling_summary_user_prompt', ''))

    save_btn = QPushButton("Save")
    save_btn.clicked.connect(_save_prompts)
    btn_row.addWidget(save_btn)

    reset_btn = QPushButton("Reset to Defaults")
    reset_btn.clicked.connect(_reset_prompts)
    btn_row.addWidget(reset_btn)

    cancel_btn = QPushButton("Cancel")
    cancel_btn.clicked.connect(dialog.close)
    btn_row.addWidget(cancel_btn)

    layout.addLayout(btn_row)

    # Show non-modally with smooth fade animation (no flash)
    try:
        from dialog_animations import show_dialog_with_fade
        show_dialog_with_fade(dialog, duration=220)
    except Exception:
        dialog.show()

def toggle_thinking_budget(self):
    """Enable/disable thinking budget entry and labels based on checkbox state (PySide6 version)"""
    try:
        enabled = bool(self.enable_gemini_thinking_var)
        
        if hasattr(self, 'thinking_budget_entry'):
            self.thinking_budget_entry.setEnabled(enabled)
            
        if hasattr(self, 'thinking_budget_label'):
            self.thinking_budget_label.setEnabled(enabled)
            color = "white" if enabled else "#808080"
            self.thinking_budget_label.setStyleSheet(f"color: {color};")
            
        if hasattr(self, 'thinking_tokens_label'):
            self.thinking_tokens_label.setEnabled(enabled)
            color = "white" if enabled else "#808080"
            self.thinking_tokens_label.setStyleSheet(f"color: {color};")
            
        if hasattr(self, 'thinking_level_combo'):
            self.thinking_level_combo.setEnabled(enabled)
            
        if hasattr(self, 'thinking_level_label'):
            self.thinking_level_label.setEnabled(enabled)
            color = "white" if enabled else "#808080"
            self.thinking_level_label.setStyleSheet(f"color: {color};")
            
        # Description label
        if hasattr(self, 'gemini_desc_label'):
            self.gemini_desc_label.setEnabled(enabled)
            color = "gray" if enabled else "#606060"
            self.gemini_desc_label.setStyleSheet(f"color: {color}; font-size: 10pt;")
    except Exception:
        pass

def toggle_gpt_reasoning_controls(self):
    """Enable/disable GPT reasoning controls and labels based on toggle state (PySide6 version)"""
    try:
        enabled = bool(self.enable_gpt_thinking_var)
        
        # Tokens entry and label
        if hasattr(self, 'gpt_reasoning_tokens_entry'):
            self.gpt_reasoning_tokens_entry.setEnabled(enabled)
        if hasattr(self, 'gpt_reasoning_tokens_label'):
            self.gpt_reasoning_tokens_label.setEnabled(enabled)
            color = "white" if enabled else "#808080"
            self.gpt_reasoning_tokens_label.setStyleSheet(f"color: {color};")
            
        # Effort combo and label
        if hasattr(self, 'gpt_effort_combo'):
            self.gpt_effort_combo.setEnabled(enabled)
        if hasattr(self, 'gpt_effort_label'):
            self.gpt_effort_label.setEnabled(enabled)
            color = "white" if enabled else "#808080"
            self.gpt_effort_label.setStyleSheet(f"color: {color};")
            
        # GPT tokens label
        if hasattr(self, 'gpt_tokens_label'):
            self.gpt_tokens_label.setEnabled(enabled)
            color = "white" if enabled else "#808080"
            self.gpt_tokens_label.setStyleSheet(f"color: {color};")
            
        # Description label
        if hasattr(self, 'gpt_desc_label'):
            self.gpt_desc_label.setEnabled(enabled)
            color = "gray" if enabled else "#606060"
            self.gpt_desc_label.setStyleSheet(f"color: {color}; font-size: 10pt;")
    except Exception:
        pass

def open_other_settings(self):
    """Open the Other Settings dialog (PySide6)"""
    from PySide6.QtGui import QIcon, QKeyEvent
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    
    # If dialog already exists, just show and focus it to preserve exact state
    try:
        if hasattr(self, "_other_settings_dialog") and self._other_settings_dialog is not None:
            # Show with fade animation
            try:
                from dialog_animations import show_dialog_with_fade
                show_dialog_with_fade(self._other_settings_dialog, duration=220)
            except Exception:
                self._other_settings_dialog.show()
            # Bring to front and focus
            try:
                self._other_settings_dialog.raise_()
                self._other_settings_dialog.activateWindow()
            except Exception:
                pass
            return
    except Exception:
        # If the old reference is invalid, recreate below
        self._other_settings_dialog = None
    
    # Create dialog with proper window attributes
    # Pass self as parent so it stays in front of main GUI but allows other dialogs on top
    dialog = QDialog(self)
    dialog.setWindowTitle("Other Settings")
    
    # Do not delete widgets on close; we'll hide instead to retain exact state
    dialog.setAttribute(Qt.WA_DeleteOnClose, False)
    
    # Set window flags
    dialog.setWindowFlags(
        Qt.WindowType.Window |
        Qt.WindowType.WindowSystemMenuHint |
        Qt.WindowType.WindowMinimizeButtonHint |
        Qt.WindowType.WindowMaximizeButtonHint |
        Qt.WindowType.WindowCloseButtonHint
    )
    
    # CRITICAL: Position dialog way off-screen during construction to prevent flash
    # We'll move it to proper position before showing
    dialog.move(-10000, -10000)
    
    # CRITICAL: Remove size constraints that prevent maximize
    dialog.setSizeGripEnabled(False)
    
    # Set icon with absolute path
    try:
        import sys
        base_dir = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base_dir, 'Halgakos.ico')
        dialog.setWindowIcon(QIcon(icon_path))
    except Exception:
        pass
    
    # Set initial size based on screen ratio
    try:
        from PySide6.QtWidgets import QApplication
        screen = QApplication.primaryScreen().availableGeometry()
        # Use 60% of screen width and 80% of screen height
        width = int(screen.width() * 0.48)
        height = int(screen.height() * 0.95)
        dialog.resize(width, height)
    except Exception:
        dialog.resize(950, 850)  # Fallback
    
    # Store original size for restoring after fullscreen
    original_geometry = None
    
    # Add F11 fullscreen toggle with stay on top for buttons visibility
    def toggle_fullscreen():
        nonlocal original_geometry
        if dialog.isFullScreen():
            dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)
            dialog.showNormal()
            # Restore original size
            if original_geometry:
                dialog.setGeometry(original_geometry)
        else:
            # Save current geometry before fullscreen
            original_geometry = dialog.geometry()
            dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
            dialog.showFullScreen()
    
    # Override key press event for F11
    original_keyPressEvent = dialog.keyPressEvent
    def custom_keyPressEvent(event: QKeyEvent):
        if event.key() == Qt.Key_F11:
            toggle_fullscreen()
        else:
            original_keyPressEvent(event)
    dialog.keyPressEvent = custom_keyPressEvent
    
    main_layout = QVBoxLayout(dialog)
    main_layout.setContentsMargins(5, 5, 5, 5)  # Set uniform margins
    main_layout.setSpacing(8)  # Set spacing between widgets

    # Set up icon path
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico')

    # Apply global stylesheet for blue checkboxes (from manga integration)
    # Back to regular string concatenation which was working before
    checkbox_radio_style = """
        QComboBox::down-arrow {
            image: url(""" + icon_path.replace('\\', '/') + """);
            width: 16px;
            height: 16px;
        }
        QCheckBox {
            color: white;
            spacing: 6px;
        }
        QCheckBox::indicator {
            width: 14px;
            height: 14px;
            border: 1px solid #5a9fd4;
            border-radius: 2px;
            background-color: #2d2d2d;
        }
        QCheckBox::indicator:checked {
            background-color: #5a9fd4;
            border-color: #5a9fd4;
        }
        QCheckBox::indicator:hover {
            border-color: #7bb3e0;
        }
        QCheckBox:disabled {
            color: #666666;
        }
        QCheckBox::indicator:disabled {
            background-color: #1a1a1a;
            border-color: #3a3a3a;
        }
        QRadioButton {
            color: white;
            spacing: 5px;
        }
        QRadioButton::indicator {
            width: 13px;
            height: 13px;
            border: 2px solid #5a9fd4;
            border-radius: 7px;
            background-color: #2d2d2d;
        }
        QRadioButton::indicator:checked {
            background-color: #5a9fd4;
            border: 2px solid #5a9fd4;
        }
        QRadioButton::indicator:hover {
            border-color: #7bb3e0;
        }
        QRadioButton:disabled {
            color: #666666;
        }
        QRadioButton::indicator:disabled {
            background-color: #1a1a1a;
            border-color: #3a3a3a;
        }
    """
    
    # Scrollable area with a 2-column grid
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    container = QWidget()
    container.setStyleSheet(checkbox_radio_style)  # Apply global stylesheet
    grid = QGridLayout(container)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)
    grid.setHorizontalSpacing(6)  # Compact horizontal spacing between columns
    grid.setVerticalSpacing(2)  # Minimal vertical spacing between sections
    grid.setContentsMargins(4, 4, 4, 4)  # Minimal grid margins

    # Build sections (converted sections will populate the Qt layout)
    self._create_context_management_section(container)
    self._create_response_handling_section(container)
    self._create_prompt_management_section(container)
    self._create_processing_options_section(container)
    self._create_image_translation_section(container)
    self._create_anti_duplicate_section(container)
    self._create_custom_api_endpoints_section(container)
    
    # Add debug controls section at the bottom
    self._create_debug_controls_section(container)

    scroll.setWidget(container)
    scroll.setWidgetResizable(True)
    
    main_layout.addWidget(scroll, 1)

    # Buttons row (Save and Close) - always visible at bottom
    btns = QHBoxLayout()
    btns.setContentsMargins(5, 10, 5, 10)  # Add padding around buttons

    def _save_and_close():
        try:
            # Mirror legacy behavior: persist some toggles on close
            if hasattr(self, 'retain_source_extension_var'):
                try:
                    self.config['retain_source_extension'] = self.retain_source_extension_var
                    os.environ['RETAIN_SOURCE_EXTENSION'] = '1' if self.retain_source_extension_var else '0'
                except Exception:
                    pass
            self.save_config(show_message=False)
            # CRITICAL: Reinitialize environment variables after saving
            # This ensures TRANSLATE_SPECIAL_FILES and other settings take effect immediately
            if hasattr(self, 'initialize_environment_variables'):
                self.initialize_environment_variables()
                self.append_log("✅ Settings saved and environment variables updated")
        except Exception as e:
            self.append_log(f"⚠️ Error saving settings: {e}")
        dialog.hide()

    save_btn = QPushButton("💾 Save Settings")
    save_btn.clicked.connect(_save_and_close)
    save_btn.setMinimumHeight(35)
    save_btn.setStyleSheet(
        "QPushButton { "
        "  background-color: #28a745; "
        "  color: white; "
        "  padding: 8px 20px; "
        "  font-size: 11pt; "
        "  font-weight: bold; "
        "  border-radius: 4px; "
        "} "
        "QPushButton:hover { background-color: #218838; }"
    )
    btns.addWidget(save_btn)

    close_btn = QPushButton("❌ Close")
    close_btn.clicked.connect(dialog.close)
    close_btn.setMinimumHeight(35)
    close_btn.setStyleSheet(
        "QPushButton { "
        "  background-color: #6c757d; "
        "  color: white; "
        "  padding: 8px 20px; "
        "  font-size: 11pt; "
        "  font-weight: bold; "
        "  border-radius: 4px; "
        "} "
        "QPushButton:hover { background-color: #5a6268; }"
    )
    btns.addWidget(close_btn)

    main_layout.addLayout(btns)

    # Intercept window close: hide instead of destroy to preserve state
    def _handle_close(event):
        try:
            event.ignore()
            dialog.hide()
        except Exception:
            # Best-effort: still hide on any error
            try:
                event.ignore()
            except Exception:
                pass
            dialog.hide()
    dialog.closeEvent = _handle_close

    # Store reference for later closing
    self._other_settings_dialog = dialog
    
    # Move dialog to center of screen (was off-screen during construction)
    try:
        screen = QApplication.primaryScreen().availableGeometry()
        dialog_x = screen.x() + (screen.width() - dialog.width()) // 2
        dialog_y = screen.y() + (screen.height() - dialog.height()) // 2
        dialog.move(dialog_x, dialog_y)
    except Exception:
        pass
    
    # Show with smooth fade animation (no flash of generic window)
    try:
        from dialog_animations import show_dialog_with_fade
        show_dialog_with_fade(dialog, duration=220)
    except Exception:
        dialog.show()
    
    # Auto-fit width to content after showing
    from PySide6.QtCore import QTimer
    def adjust_width():
        # Calculate width needed for both columns with spacing
        needed_width = container.sizeHint().width() + 40  # Add margins
        if needed_width > dialog.width():
            dialog.resize(needed_width, dialog.height())
    QTimer.singleShot(0, adjust_width)

def _create_context_management_section(self, parent):
    """Create context management section (PySide6)"""
    # Expect parent to have a QGridLayout
    grid = parent.layout() if hasattr(parent, 'layout') else None
    if not isinstance(grid, QGridLayout):
        # If no grid yet, create one so we can place the section
        grid = QGridLayout(parent)
        parent.setLayout(grid)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

    section_box = QGroupBox("Context Management & Memory")
    # No max width - let it expand in fullscreen
    section_v = QVBoxLayout(section_box)
    section_v.setContentsMargins(8, 8, 8, 8)  # Compact margins
    section_v.setSpacing(4)  # Compact spacing between widgets

    # Include previous source text toggle (controls whether source-side history is reused)
    include_source_cb = self._create_styled_checkbox("Include previous source text in history/memory (Not Recommended)")
    try:
        include_source_cb.setChecked(bool(self.include_source_in_history_var))
    except Exception:
        pass
    def _on_include_source_toggled(checked):
        try:
            self.include_source_in_history_var = bool(checked)
        except Exception:
            pass
    include_source_cb.toggled.connect(_on_include_source_toggled)
    section_v.addWidget(include_source_cb)

    # Rolling summary toggle
    rolling_cb = self._create_styled_checkbox("Use Rolling Summary (Memory)")
    try:
        rolling_cb.setChecked(bool(self.rolling_summary_var))
    except Exception:
        pass

    # Warning label (kept visually distinct, but on the same row)
    rolling_warn = QLabel("⚠ Do not use with contextual translation")
    rolling_warn.setStyleSheet(
        "color: #f59e0b; font-style: italic; font-size: 9pt;"
    )
    rolling_warn.setContentsMargins(0, 0, 0, 0)

    # Store references to controls that should be enabled/disabled
    rolling_controls = []
    
    def _on_rolling_toggled(checked):
        try:
            self.rolling_summary_var = bool(checked)
            # Enable/disable all rolling summary controls
            for control in rolling_controls:
                control.setEnabled(bool(checked))
        except Exception:
            pass
    rolling_cb.toggled.connect(_on_rolling_toggled)

    # Put checkbox + warning on the same line
    rolling_row = QWidget()
    rolling_row_l = QHBoxLayout(rolling_row)
    rolling_row_l.setContentsMargins(0, 0, 0, 0)
    rolling_row_l.setSpacing(8)
    rolling_row_l.addWidget(rolling_cb)
    rolling_row_l.addWidget(rolling_warn)
    rolling_row_l.addStretch(1)
    section_v.addWidget(rolling_row)

    # Description
    desc = QLabel("AI-powered memory system that maintains story context")
    desc.setStyleSheet("color: gray; font-size: 9pt;")
    desc.setContentsMargins(0, 0, 0, 8)
    section_v.addWidget(desc)

    # Settings container - 2 column grid layout
    settings_w = QWidget()
    settings_grid = QGridLayout(settings_w)
    settings_grid.setContentsMargins(0, 5, 0, 5)
    settings_grid.setHorizontalSpacing(0)  # No spacing between label and input
    settings_grid.setVerticalSpacing(6)

    # Row 0: Role and Mode
    role_lbl = QLabel("Role: ")
    settings_grid.addWidget(role_lbl, 0, 0, alignment=Qt.AlignRight)
    rolling_controls.append(role_lbl)
    role_combo = QComboBox()
    # Controls how the rolling-summary GENERATION request is built (summary API call).
    # NOTE: The rolling summary is always injected into translation as an assistant message.
    #
    # user   -> send user prompt only (configured summary user prompt + translated text)
    # system -> send system prompt + user message containing ONLY the translated text
    # both   -> send both system + user (legacy/current behavior)
    role_combo.addItems(["user", "system", "both"])
    role_combo.setFixedWidth(90)
    # Add custom styling with unicode arrow
    role_combo.setStyleSheet("""
        QComboBox::down-arrow {
            image: none;
            width: 12px;
            height: 12px;
            border: none;
        }
    """)
    self._add_combobox_arrow(role_combo)
    self._disable_combobox_mousewheel(role_combo)
    try:
        current_role = str(getattr(self, 'summary_role_var', 'system') or 'system').strip().lower()
        idx = role_combo.findText(current_role)
        if idx >= 0:
            role_combo.setCurrentIndex(idx)
        else:
            role_combo.setCurrentIndex(0)
    except Exception:
        pass
    def _on_role_changed(text):
        try:
            self.summary_role_var = str(text).strip().lower()
        except Exception:
            pass
    role_combo.currentTextChanged.connect(_on_role_changed)
    settings_grid.addWidget(role_combo, 0, 1, alignment=Qt.AlignLeft)
    rolling_controls.append(role_combo)
    
    mode_lbl = QLabel(" Mode: ")
    settings_grid.addWidget(mode_lbl, 0, 2, alignment=Qt.AlignRight)
    rolling_controls.append(mode_lbl)
    mode_combo = QComboBox()
    mode_combo.addItems(["append", "replace"])
    mode_combo.setFixedWidth(100)
    # Add custom styling with unicode arrow
    mode_combo.setStyleSheet("""
        QComboBox::down-arrow {
            image: none;
            width: 12px;
            height: 12px;
            border: none;
        }
    """)
    self._add_combobox_arrow(mode_combo)
    self._disable_combobox_mousewheel(mode_combo)
    try:
        mode_combo.setCurrentText(self.rolling_summary_mode_var)
    except Exception:
        pass
    def _on_mode_changed(text):
        try:
            self.rolling_summary_mode_var = text
        except Exception:
            pass
    mode_combo.currentTextChanged.connect(_on_mode_changed)
    settings_grid.addWidget(mode_combo, 0, 3, alignment=Qt.AlignLeft)
    rolling_controls.append(mode_combo)
    
    # Add Max Tokens field to the right of Mode
    max_tokens_lbl = QLabel(" Max tokens: ")
    settings_grid.addWidget(max_tokens_lbl, 0, 4, alignment=Qt.AlignRight)
    rolling_controls.append(max_tokens_lbl)
    max_tokens_edit = QLineEdit()
    max_tokens_edit.setFixedWidth(80)
    try:
        max_tokens_edit.setText(str(self.rolling_summary_max_tokens_var))
    except Exception:
        pass
    def _on_max_tokens_changed(text):
        try:
            self.rolling_summary_max_tokens_var = text
        except Exception:
            pass
    max_tokens_edit.textChanged.connect(_on_max_tokens_changed)
    settings_grid.addWidget(max_tokens_edit, 0, 5, alignment=Qt.AlignLeft)
    rolling_controls.append(max_tokens_edit)

    # Row 1: Summarize last and Retain
    summ_lbl = QLabel("Summarize last \n  N exchanges: ")
    settings_grid.addWidget(summ_lbl, 1, 0, alignment=Qt.AlignRight)
    rolling_controls.append(summ_lbl)
    exchanges_edit = QLineEdit()
    exchanges_edit.setFixedWidth(70)
    try:
        exchanges_edit.setText(str(self.rolling_summary_exchanges_var))
    except Exception:
        pass
    def _on_exchanges_changed(text):
        try:
            self.rolling_summary_exchanges_var = text
        except Exception:
            pass
    exchanges_edit.textChanged.connect(_on_exchanges_changed)
    settings_grid.addWidget(exchanges_edit, 1, 1, alignment=Qt.AlignLeft)
    rolling_controls.append(exchanges_edit)
    
    retain_lbl = QLabel(" Retain N entries: ")
    settings_grid.addWidget(retain_lbl, 1, 2, alignment=Qt.AlignRight)
    rolling_controls.append(retain_lbl)
    retain_edit = QLineEdit()
    retain_edit.setFixedWidth(70)
    try:
        retain_edit.setText(str(self.rolling_summary_max_entries_var))
    except Exception:
        pass
    def _on_retain_changed(text):
        try:
            self.rolling_summary_max_entries_var = text
        except Exception:
            pass
    retain_edit.textChanged.connect(_on_retain_changed)
    settings_grid.addWidget(retain_edit, 1, 3, alignment=Qt.AlignLeft)
    rolling_controls.append(retain_edit)

    section_v.addWidget(settings_w)
    
    # Configure prompts button
    cfg_btn = QPushButton("⚙️ Configure Memory Prompts")
    cfg_btn.setMinimumHeight(28)
    cfg_btn.setStyleSheet(
        "QPushButton { "
        "  background-color: #17a2b8; "
        "  color: white; "
        "  padding: 5px 12px; "
        "  font-size: 10pt; "
        "  font-weight: bold; "
        "  border-radius: 3px; "
        "} "
        "QPushButton:hover { background-color: #138496; } "
        "QPushButton:disabled { "
        "  background-color: #6c757d; "
        "  color: #adb5bd; "
        "}"
    )
    cfg_btn.clicked.connect(self.configure_rolling_summary_prompts)
    section_v.addWidget(cfg_btn)
    rolling_controls.append(cfg_btn)
    
    # Set initial enabled state based on checkbox
    initial_state = rolling_cb.isChecked()
    for control in rolling_controls:
        control.setEnabled(initial_state)

    # Separator
    sep1 = QFrame()
    sep1.setFrameShape(QFrame.HLine)
    sep1.setFrameShadow(QFrame.Sunken)
    section_v.addWidget(sep1)

    # Memory mode info
    info = QLabel(
        "💡 Memory Mode:\n• Append: Keeps adding summaries (longer context)\n• Replace: Only keeps latest summary (concise)"
    )
    info.setStyleSheet("color: #666;")
    section_v.addWidget(info)

    # Separator
    sep2 = QFrame()
    sep2.setFrameShape(QFrame.HLine)
    sep2.setFrameShadow(QFrame.Sunken)
    section_v.addWidget(sep2)

    # Application Updates
    section_v.addWidget(QLabel("Application Updates:"))
    updates_row = QWidget()
    updates_h = QHBoxLayout(updates_row)
    updates_h.setContentsMargins(0, 0, 0, 0)

    btn_check_updates = QPushButton("🔄 Check for Updates")
    btn_check_updates.clicked.connect(lambda: self.check_for_updates_manual())
    btn_check_updates.setStyleSheet(
        "QPushButton { background-color: #17a2b8; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold; } "
        "QPushButton:hover { background-color: #138496; }"
    )
    updates_h.addWidget(btn_check_updates)

    auto_cb = self._create_styled_checkbox("Check on startup")
    try:
        auto_cb.setChecked(bool(self.auto_update_check_var))
    except Exception:
        pass
    def _on_auto_update(checked):
        try:
            self.auto_update_check_var = bool(checked)
        except Exception:
            pass
    auto_cb.toggled.connect(_on_auto_update)
    updates_h.addSpacing(10)
    updates_h.addWidget(auto_cb)

    section_v.addWidget(updates_row)

    updates_desc = QLabel("Check GitHub for new Glossarion releases\nand download updates")
    updates_desc.setStyleSheet("color: gray;")
    section_v.addWidget(updates_desc)

    # Separator
    sep3 = QFrame()
    sep3.setFrameShape(QFrame.HLine)
    sep3.setFrameShadow(QFrame.Sunken)
    section_v.addWidget(sep3)

    # Config Backup Management
    section_v.addWidget(QLabel("Config Backup Management:"))

    backup_row = QWidget()
    backup_h = QHBoxLayout(backup_row)
    backup_h.setContentsMargins(0, 0, 0, 0)

    btn_backup = QPushButton("💾 Create Backup")
    btn_backup.clicked.connect(lambda: self._create_manual_config_backup())
    btn_backup.setStyleSheet(
        "QPushButton { background-color: #28a745; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold; } "
        "QPushButton:hover { background-color: #218838; }"
    )
    backup_h.addWidget(btn_backup)

    btn_restore = QPushButton("↶ Restore Backup")
    btn_restore.clicked.connect(lambda: self._manual_restore_config())
    btn_restore.setStyleSheet(
        "QPushButton { background-color: #ffc107; color: black; padding: 5px 10px; border-radius: 3px; font-weight: bold; } "
        "QPushButton:hover { background-color: #e0a800; }"
    )
    backup_h.addWidget(btn_restore)

    section_v.addWidget(backup_row)

    backup_desc = QLabel("Automatic backups are created before each config save.")
    backup_desc.setStyleSheet("color: gray;")
    section_v.addWidget(backup_desc)

    # Place the section at row 0, column 1 to match the original grid
    try:
        grid.addWidget(section_box, 0, 1)
    except Exception:
        # Fallback: just stack
        section_box.setParent(parent)

def _create_response_handling_section(self, parent):
    """Create response handling section with AI Hunter additions (PySide6)"""
    from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QComboBox, QLineEdit, QPushButton, QWidget, QRadioButton, QButtonGroup
    from PySide6.QtCore import Qt
    
    section_box = QGroupBox("Response Handling & Retry Logic")
    # No max width - let it expand in fullscreen
    section_v = QVBoxLayout(section_box)
    section_v.setContentsMargins(8, 8, 8, 8)  # Compact margins
    section_v.setSpacing(4)  # Compact spacing between widgets
    
    # GPT-5/OpenAI Reasoning Toggle
    gpt5_title = QLabel("GPT-5 Thinking (OpenRouter/OpenAI-style)")
    gpt5_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
    section_v.addWidget(gpt5_title)
    
    gpt_row1 = QWidget()
    gpt_h1 = QHBoxLayout(gpt_row1)
    gpt_h1.setContentsMargins(20, 5, 0, 0)
    
    gpt_enable_cb = self._create_styled_checkbox("Enable GPT / OR Thinking")
    try:
        gpt_enable_cb.setChecked(bool(self.enable_gpt_thinking_var))
    except Exception:
        pass
    def _on_gpt_thinking_toggle(checked):
        try:
            self.enable_gpt_thinking_var = bool(checked)
            self.toggle_gpt_reasoning_controls()
        except Exception:
            pass
    gpt_enable_cb.toggled.connect(_on_gpt_thinking_toggle)
    gpt_h1.addWidget(gpt_enable_cb)
    
    gpt_h1.addSpacing(20)
    self.gpt_effort_label = QLabel("Effort:")
    gpt_h1.addWidget(self.gpt_effort_label)
    self.gpt_effort_combo = QComboBox()
    # GPT thinking effort now supports "none" (disable) and "xhigh" in addition to low/medium/high
    self.gpt_effort_combo.addItems(["none", "low", "medium", "high", "xhigh"])
    self.gpt_effort_combo.setFixedWidth(100)
    # Add custom styling with unicode arrow
    self.gpt_effort_combo.setStyleSheet("""
        QComboBox::down-arrow {
            image: none;
            width: 12px;
            height: 12px;
            border: none;
        }
    """)
    self._add_combobox_arrow(self.gpt_effort_combo)
    self._disable_combobox_mousewheel(self.gpt_effort_combo)
    try:
        effort_val = self.gpt_effort_var
        idx = self.gpt_effort_combo.findText(effort_val)
        if idx >= 0:
            self.gpt_effort_combo.setCurrentIndex(idx)
    except Exception:
        pass
    def _on_effort_changed(text):
        try:
            self.gpt_effort_var = text
        except Exception:
            pass
    self.gpt_effort_combo.currentTextChanged.connect(_on_effort_changed)
    gpt_h1.addWidget(self.gpt_effort_combo)
    gpt_h1.addStretch()
    section_v.addWidget(gpt_row1)
    
    # Second row for OpenRouter-specific token budget
    gpt_row2 = QWidget()
    gpt_h2 = QHBoxLayout(gpt_row2)
    gpt_h2.setContentsMargins(40, 5, 0, 0)
    self.gpt_reasoning_tokens_label = QLabel("OR Thinking Tokens:")
    gpt_h2.addWidget(self.gpt_reasoning_tokens_label)
    self.gpt_reasoning_tokens_entry = QLineEdit()
    self.gpt_reasoning_tokens_entry.setFixedWidth(70)
    try:
        self.gpt_reasoning_tokens_entry.setText(str(self.gpt_reasoning_tokens_var))
    except Exception:
        pass
    def _on_gpt_tokens_changed(text):
        try:
            self.gpt_reasoning_tokens_var = text
        except Exception:
            pass
    self.gpt_reasoning_tokens_entry.textChanged.connect(_on_gpt_tokens_changed)
    gpt_h2.addWidget(self.gpt_reasoning_tokens_entry)
    self.gpt_tokens_label = QLabel("tokens")
    gpt_h2.addWidget(self.gpt_tokens_label)
    gpt_h2.addStretch()
    section_v.addWidget(gpt_row2)
    
    # Store reference to description label for enable/disable
    self.gpt_desc_label = QLabel("Controls GPT-5 and OpenRouter reasoning.\nProvide Tokens to force a max token budget for other models,\n GPT-5 uses Effort (none/low/medium/high/xhigh).")
    self.gpt_desc_label.setStyleSheet("color: gray; font-size: 10pt;")
    self.gpt_desc_label.setContentsMargins(20, 0, 0, 10)
    section_v.addWidget(self.gpt_desc_label)
    
    # Initialize enabled state for GPT controls
    self.toggle_gpt_reasoning_controls()
    
    # Gemini Thinking Mode
    gemini_title = QLabel("Gemini Thinking Mode")
    gemini_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
    section_v.addWidget(gemini_title)
    
    thinking_row = QWidget()
    thinking_h = QHBoxLayout(thinking_row)
    thinking_h.setContentsMargins(20, 5, 0, 0)
    
    gemini_thinking_cb = self._create_styled_checkbox("Enable Gemini Thinking")
    try:
        gemini_thinking_cb.setChecked(bool(self.enable_gemini_thinking_var))
    except Exception:
        pass
    def _on_gemini_thinking_toggle(checked):
        try:
            self.enable_gemini_thinking_var = bool(checked)
            self.toggle_thinking_budget()
        except Exception:
            pass
    gemini_thinking_cb.toggled.connect(_on_gemini_thinking_toggle)
    thinking_h.addWidget(gemini_thinking_cb)
    
    thinking_h.addSpacing(20)
    self.thinking_budget_label = QLabel("Budget:")
    thinking_h.addWidget(self.thinking_budget_label)
    self.thinking_budget_entry = QLineEdit()
    self.thinking_budget_entry.setFixedWidth(70)
    try:
        self.thinking_budget_entry.setText(str(self.thinking_budget_var))
    except Exception:
        pass
    def _on_budget_changed(text):
        try:
            self.thinking_budget_var = text
        except Exception:
            pass
    self.thinking_budget_entry.textChanged.connect(_on_budget_changed)
    thinking_h.addWidget(self.thinking_budget_entry)
    self.thinking_tokens_label = QLabel("tokens")
    thinking_h.addWidget(self.thinking_tokens_label)
    
    thinking_h.addSpacing(20)
    self.thinking_level_label = QLabel("Level (Gemini 3):")
    thinking_h.addWidget(self.thinking_level_label)
    self.thinking_level_combo = QComboBox()
    self.thinking_level_combo.addItems(["low", "high"])
    self.thinking_level_combo.setFixedWidth(80)
    self.thinking_level_combo.setStyleSheet("""
        QComboBox::down-arrow {
            image: none;
            width: 12px;
            height: 12px;
            border: none;
        }
    """)
    self._add_combobox_arrow(self.thinking_level_combo)
    self._disable_combobox_mousewheel(self.thinking_level_combo)
    try:
        level_val = self.thinking_level_var
        idx = self.thinking_level_combo.findText(level_val)
        if idx >= 0:
            self.thinking_level_combo.setCurrentIndex(idx)
    except Exception:
        pass
    def _on_level_changed(text):
        try:
            self.thinking_level_var = text
            os.environ["GEMINI_THINKING_LEVEL"] = text
        except Exception:
            pass
    self.thinking_level_combo.currentTextChanged.connect(_on_level_changed)
    thinking_h.addWidget(self.thinking_level_combo)
    
    thinking_h.addStretch()
    section_v.addWidget(thinking_row)
    
    # Store reference to description label for enable/disable
    self.gemini_desc_label = QLabel("Control Gemini's thinking process. 0 = disabled,\n512-24576 = limited thinking, -1 = dynamic (auto)")
    self.gemini_desc_label.setStyleSheet("color: gray; font-size: 10pt;")
    self.gemini_desc_label.setContentsMargins(20, 0, 0, 10)
    section_v.addWidget(self.gemini_desc_label)
    
    # Initialize enabled state for Gemini controls
    self.toggle_thinking_budget()

    # DeepSeek Thinking Mode
    deepseek_title = QLabel("DeepSeek Thinking Mode")
    deepseek_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
    section_v.addWidget(deepseek_title)

    deepseek_row = QWidget()
    deepseek_h = QHBoxLayout(deepseek_row)
    deepseek_h.setContentsMargins(20, 5, 0, 0)

    deepseek_cb = self._create_styled_checkbox("Enable DeepSeek Thinking")
    try:
        deepseek_cb.setChecked(bool(getattr(self, 'enable_deepseek_thinking_var', True)))
    except Exception:
        deepseek_cb.setChecked(True)

    def _on_deepseek_thinking_toggle(checked):
        try:
            self.enable_deepseek_thinking_var = bool(checked)
            os.environ['ENABLE_DEEPSEEK_THINKING'] = '1' if self.enable_deepseek_thinking_var else '0'
        except Exception:
            pass

    deepseek_cb.toggled.connect(_on_deepseek_thinking_toggle)
    deepseek_h.addWidget(deepseek_cb)
    deepseek_h.addStretch()
    section_v.addWidget(deepseek_row)

    deepseek_desc = QLabel("Adds extra_body={thinking:{type:enabled}} for DeepSeek OpenAI-compatible requests.\nEnables reasoning_content when supported.")
    deepseek_desc.setStyleSheet("color: gray; font-size: 10pt;")
    deepseek_desc.setContentsMargins(20, 0, 0, 10)
    section_v.addWidget(deepseek_desc)
    
    # Separator
    sep1 = QFrame()
    sep1.setFrameShape(QFrame.HLine)
    sep1.setFrameShadow(QFrame.Sunken)
    section_v.addWidget(sep1)
    
    # Parallel Extraction
    parallel_title = QLabel("Parallel Extraction")
    parallel_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
    section_v.addWidget(parallel_title)
    
    extraction_row = QWidget()
    extraction_h = QHBoxLayout(extraction_row)
    extraction_h.setContentsMargins(20, 5, 0, 0)
    
    parallel_cb = self._create_styled_checkbox("Enable Parallel Processing")
    try:
        parallel_cb.setChecked(bool(self.enable_parallel_extraction_var))
    except Exception:
        pass
    def _on_parallel_toggle(checked):
        try:
            self.enable_parallel_extraction_var = bool(checked)
            self.toggle_extraction_workers()
        except Exception:
            pass
    parallel_cb.toggled.connect(_on_parallel_toggle)
    extraction_h.addWidget(parallel_cb)
    
    extraction_h.addSpacing(20)
    self.workers_label = QLabel("Workers:")
    extraction_h.addWidget(self.workers_label)
    self.extraction_workers_entry = QLineEdit()
    self.extraction_workers_entry.setFixedWidth(50)
    try:
        self.extraction_workers_entry.setText(str(self.extraction_workers_var))
    except Exception:
        pass
    def _on_workers_changed(text):
        try:
            self.extraction_workers_var = text
        except Exception:
            pass
    self.extraction_workers_entry.textChanged.connect(_on_workers_changed)
    extraction_h.addWidget(self.extraction_workers_entry)
    self.threads_label = QLabel("threads")
    extraction_h.addWidget(self.threads_label)
    extraction_h.addStretch()
    section_v.addWidget(extraction_row)
    
    # Store reference to description label for enable/disable
    self.parallel_desc_label = QLabel("Speed up EPUB extraction using multiple threads.\nRecommended: 4-8 workers (set to 1 to disable)")
    self.parallel_desc_label.setStyleSheet("color: gray; font-size: 10pt;")
    self.parallel_desc_label.setContentsMargins(20, 0, 0, 10)
    section_v.addWidget(self.parallel_desc_label)
    
    # Initialize enabled state for Parallel Extraction controls
    self.toggle_extraction_workers()
    
    # GUI Yield Toggle
    gui_yield_row = QWidget()
    gui_yield_h = QHBoxLayout(gui_yield_row)
    gui_yield_h.setContentsMargins(20, 5, 0, 0)
    
    gui_yield_cb = self._create_styled_checkbox("Enable GUI Responsiveness Yield")
    try:
        gui_yield_cb.setChecked(bool(self.enable_gui_yield_var))
    except Exception:
        pass
    def _on_gui_yield_toggle(checked):
        try:
            self.enable_gui_yield_var = bool(checked)
        except Exception:
            pass
    gui_yield_cb.toggled.connect(_on_gui_yield_toggle)
    gui_yield_h.addWidget(gui_yield_cb)
    gui_yield_h.addStretch()
    section_v.addWidget(gui_yield_row)
    
    gui_yield_desc = QLabel("Adds small delays during extraction to keep GUI responsive.\n⚠️ Disable for maximum extraction speed (GUI may freeze temporarily)")
    gui_yield_desc.setStyleSheet("color: gray; font-size: 10pt;")
    gui_yield_desc.setContentsMargins(20, 5, 0, 10)
    section_v.addWidget(gui_yield_desc)
    
    # Separator
    sep2 = QFrame()
    sep2.setFrameShape(QFrame.HLine)
    sep2.setFrameShadow(QFrame.Sunken)
    section_v.addWidget(sep2)
    
    # Multi API Key Management Section
    multi_key_row = QWidget()
    multi_key_h = QHBoxLayout(multi_key_row)
    multi_key_h.setContentsMargins(0, 0, 0, 15)
    
    # Create status labels and store references for dynamic updates
    self.multi_key_status_label1 = QLabel("🔑 Multi-Key Mode:")
    self.multi_key_status_label1.setStyleSheet("font-weight: bold; font-size: 11pt;")
    multi_key_h.addWidget(self.multi_key_status_label1)
    
    self.multi_key_status_label2 = QLabel()
    multi_key_h.addWidget(self.multi_key_status_label2)
    
    # Update status initially
    self._update_multi_key_status_label()
    
    multi_key_h.addStretch()
    
    section_v.addWidget(multi_key_row)
    
    multi_key_desc = QLabel("Manage multiple API keys with automatic rotation and rate limit handling")
    multi_key_desc.setStyleSheet("color: gray; font-size: 10pt;")
    multi_key_desc.setContentsMargins(20, 0, 0, 5)
    section_v.addWidget(multi_key_desc)
    
    # Multi API Key Manager button (moved below description)
    btn_row = QWidget()
    btn_row_h = QHBoxLayout(btn_row)
    btn_row_h.setContentsMargins(20, 5, 0, 10)
    btn_multi_key = QPushButton("⚙️ Configure API Keys")
    btn_multi_key.setMinimumWidth(160)
    btn_multi_key.setMaximumWidth(200)
    btn_multi_key.setMinimumHeight(28)
    btn_multi_key.setStyleSheet(
        "QPushButton { "
        "  background-color: #17a2b8; "
        "  color: white; "
        "  padding: 5px 12px; "
        "  font-size: 10pt; "
        "  font-weight: bold; "
        "  border-radius: 3px; "
        "} "
        "QPushButton:hover { background-color: #138496; }"
    )
    btn_multi_key.clicked.connect(lambda: self.open_multi_api_key_manager())
    btn_row_h.addWidget(btn_multi_key)
    btn_row_h.addStretch()
    section_v.addWidget(btn_row)
    
    # Separator
    sep3 = QFrame()
    sep3.setFrameShape(QFrame.HLine)
    sep3.setFrameShadow(QFrame.Sunken)
    section_v.addWidget(sep3)
    
    # Retry Truncated
    retry_truncated_cb = self._create_styled_checkbox("Auto-retry Truncated Responses")
    try:
        retry_truncated_cb.setChecked(bool(self.retry_truncated_var))
    except Exception:
        pass
    def _on_retry_truncated_toggle(checked):
        try:
            self.retry_truncated_var = bool(checked)
        except Exception:
            pass
    retry_truncated_cb.toggled.connect(_on_retry_truncated_toggle)
    section_v.addWidget(retry_truncated_cb)
    
    retry_frame_w = QWidget()
    retry_frame_h = QHBoxLayout(retry_frame_w)
    retry_frame_h.setContentsMargins(20, 5, 0, 5)
    retry_frame_h.addWidget(QLabel("Token constraint:"))
    retry_tokens_edit = QLineEdit()
    retry_tokens_edit.setFixedWidth(80)
    try:
        retry_tokens_edit.setText(str(self.max_retry_tokens_var))
    except Exception:
        pass
    def _on_retry_tokens_changed(text):
        try:
            self.max_retry_tokens_var = text
        except Exception:
            pass
    retry_tokens_edit.textChanged.connect(_on_retry_tokens_changed)
    retry_frame_h.addWidget(retry_tokens_edit)
    retry_frame_h.addStretch()
    section_v.addWidget(retry_frame_w)
    
    retry_desc = QLabel("Retry when truncated. Acts as min/max constraint:\nbelow value = minimum, above value = maximum")
    retry_desc.setStyleSheet("color: gray; font-size: 10pt;")
    retry_desc.setContentsMargins(20, 0, 0, 10)
    section_v.addWidget(retry_desc)
    
    # Separator
    sep4 = QFrame()
    sep4.setFrameShape(QFrame.HLine)
    sep4.setFrameShadow(QFrame.Sunken)
    section_v.addWidget(sep4)
    
    # Preserve Original Text on Failure
    preserve_cb = self._create_styled_checkbox("Preserve Original Text on Failure")
    try:
        preserve_cb.setChecked(bool(self.preserve_original_text_var))
    except Exception:
        pass
    def _on_preserve_toggle(checked):
        try:
            self.preserve_original_text_var = bool(checked)
        except Exception:
            pass
    preserve_cb.toggled.connect(_on_preserve_toggle)
    section_v.addWidget(preserve_cb)
    
    preserve_desc = QLabel("Return original untranslated text when translation fails.\n⚠️ May mix source language into translated output")
    preserve_desc.setStyleSheet("color: gray; font-size: 10pt;")
    preserve_desc.setContentsMargins(20, 5, 0, 10)
    section_v.addWidget(preserve_desc)
    
    # Separator
    sep5 = QFrame()
    sep5.setFrameShape(QFrame.HLine)
    sep5.setFrameShadow(QFrame.Sunken)
    section_v.addWidget(sep5)
    
    # Compression Factor
    compression_title = QLabel("Translation Compression Factor")
    compression_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
    section_v.addWidget(compression_title)
    
    # Auto Compression Factor toggle
    auto_compression_cb = self._create_styled_checkbox("Auto Compression Factor")
    try:
        auto_compression_cb.setChecked(bool(self.config.get('auto_compression_factor', True)))
    except Exception:
        auto_compression_cb.setChecked(True)
    
    compression_w = QWidget()
    compression_h = QHBoxLayout(compression_w)
    compression_h.setContentsMargins(20, 5, 0, 0)
    compression_h.addWidget(QLabel("CJK→English compression:"))
    compression_edit = QLineEdit()
    compression_edit.setFixedWidth(60)
    try:
        compression_edit.setText(str(self.compression_factor_var))
    except Exception:
        pass
    
    def _update_compression_factor():
        """Update compression factor based on output token limit when auto is enabled"""
        try:
            if not auto_compression_cb.isChecked():
                return
            
            # Get current output token limit
            output_tokens = int(getattr(self, 'max_output_tokens', 65536))
            
            # Determine compression factor based on token limit
            if output_tokens < 16379:
                factor = 1.5
            elif output_tokens < 32769:
                factor = 2.0
            elif output_tokens < 65536:
                factor = 2.5
            else:  # 65536 or above
                factor = 3.0
            
            # Update the field and variable
            compression_edit.setText(str(factor))
            self.compression_factor_var = str(factor)
        except Exception as e:
            print(f"Error updating compression factor: {e}")
    
    # Store the update function as an instance method so it can be called from main GUI
    self._update_auto_compression_factor = _update_compression_factor
    
    def _on_compression_changed(text):
        try:
            self.compression_factor_var = text
        except Exception:
            pass
    
    def _on_auto_compression_toggle(checked):
        try:
            self.config['auto_compression_factor'] = bool(checked)
            # Enable/disable manual editing
            compression_edit.setEnabled(not checked)
            # Update factor when enabling auto
            if checked:
                _update_compression_factor()
        except Exception as e:
            print(f"Error toggling auto compression: {e}")
    
    auto_compression_cb.toggled.connect(_on_auto_compression_toggle)
    section_v.addWidget(auto_compression_cb)
    
    auto_compression_desc = QLabel("Automatically adjusts based on output token limit:\n<16379: 1.5 | <32769: 2.0 | <65536: 2.5 | ≥65536: 3.0")
    auto_compression_desc.setStyleSheet("color: gray; font-size: 10pt;")
    auto_compression_desc.setContentsMargins(20, 0, 0, 10)
    section_v.addWidget(auto_compression_desc)
    
    compression_edit.textChanged.connect(_on_compression_changed)
    compression_h.addWidget(compression_edit)
    compression_h.addWidget(QLabel("(1.0-5.0)"))
    compression_h.addStretch()
    section_v.addWidget(compression_w)
    
    # Apply initial state
    _on_auto_compression_toggle(auto_compression_cb.isChecked())
    
    compression_desc = QLabel("Ratio for chunk sizing based on output limits")
    compression_desc.setStyleSheet("color: gray; font-size: 10pt;")
    compression_desc.setContentsMargins(20, 0, 0, 10)
    section_v.addWidget(compression_desc)
    
    # Separator
    sep6 = QFrame()
    sep6.setFrameShape(QFrame.HLine)
    sep6.setFrameShadow(QFrame.Sunken)
    section_v.addWidget(sep6)
    
    # Retry Duplicate
    retry_duplicate_cb = self._create_styled_checkbox("Auto-retry Duplicate Content")
    try:
        retry_duplicate_cb.setChecked(bool(self.retry_duplicate_var))
    except Exception:
        pass
    def _on_retry_duplicate_toggle(checked):
        try:
            self.retry_duplicate_var = bool(checked)
            update_detection_visibility()
        except Exception:
            pass
    retry_duplicate_cb.toggled.connect(_on_retry_duplicate_toggle)
    section_v.addWidget(retry_duplicate_cb)
    
    duplicate_w = QWidget()
    duplicate_h = QHBoxLayout(duplicate_w)
    duplicate_h.setContentsMargins(20, 5, 0, 0)
    duplicate_h.addWidget(QLabel("Check last"))
    duplicate_edit = QLineEdit()
    duplicate_edit.setFixedWidth(40)
    try:
        duplicate_edit.setText(str(self.duplicate_lookback_var))
    except Exception:
        pass
    def _on_duplicate_lookback_changed(text):
        try:
            self.duplicate_lookback_var = text
        except Exception:
            pass
    duplicate_edit.textChanged.connect(_on_duplicate_lookback_changed)
    duplicate_h.addWidget(duplicate_edit)
    duplicate_h.addWidget(QLabel("chapters"))
    duplicate_h.addStretch()
    section_v.addWidget(duplicate_w)
    
    duplicate_desc = QLabel("Detects when AI returns same content\nfor different chapters")
    duplicate_desc.setStyleSheet("color: gray; font-size: 10pt;")
    duplicate_desc.setContentsMargins(20, 5, 0, 10)
    section_v.addWidget(duplicate_desc)
    
    # Container for detection-related options (to show/hide based on toggle)
    self.detection_options_container = QWidget()
    detection_options_v = QVBoxLayout(self.detection_options_container)
    detection_options_v.setContentsMargins(0, 0, 0, 0)
    
    # Update thinking budget entry state based on initial toggle state
    self.toggle_thinking_budget()
    
    # Function to show/hide detection options based on auto-retry toggle
    def update_detection_visibility():
        try:
            if self.retry_duplicate_var:
                self.detection_options_container.setVisible(True)
            else:
                self.detection_options_container.setVisible(False)
        except Exception:
            pass
    
    # Detection Method subsection (now inside the container)
    method_label = QLabel("Detection Method:")
    method_label.setStyleSheet("font-weight: bold; font-size: 10pt;")
    method_label.setContentsMargins(20, 10, 0, 5)
    detection_options_v.addWidget(method_label)
    
    methods = [
        ("basic", "Basic (Fast) - Original 85% threshold, 1000 chars"),
        ("ai-hunter", "AI Hunter - Multi-method semantic analysis"),
        ("cascading", "Cascading - Basic first, then AI Hunter")
    ]
    
    # Container for AI Hunter config (will be shown/hidden based on selection)
    self.ai_hunter_container = QWidget()
    ai_hunter_v = QVBoxLayout(self.ai_hunter_container)
    ai_hunter_v.setContentsMargins(0, 0, 0, 0)
    
    # Function to update AI Hunter visibility based on detection mode
    def update_ai_hunter_visibility(*args):
        """Update AI Hunter section visibility based on selection"""
        # Clear existing widgets
        while ai_hunter_v.count():
            child = ai_hunter_v.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Show AI Hunter config for both ai-hunter and cascading modes
        try:
            if self.duplicate_detection_mode_var in ['ai-hunter', 'cascading']:
                self.create_ai_hunter_section(self.ai_hunter_container)
        except Exception:
            pass
        
        # Update status if label exists
        if hasattr(self, 'ai_hunter_status_label'):
            try:
                self.ai_hunter_status_label.setText(self._get_ai_hunter_status_text())
            except Exception:
                pass
    
    # Create radio buttons (inside detection container)
    detection_button_group = QButtonGroup(self.detection_options_container)
    for value, text in methods:
        rb = QRadioButton(text)
        rb.setContentsMargins(40, 2, 0, 2)
        try:
            if self.duplicate_detection_mode_var == value:
                rb.setChecked(True)
        except Exception:
            pass
        def _make_rb_callback(val):
            def _cb(checked):
                if checked:
                    try:
                        self.duplicate_detection_mode_var = val
                        update_ai_hunter_visibility()
                    except Exception:
                        pass
            return _cb
        rb.toggled.connect(_make_rb_callback(value))
        detection_button_group.addButton(rb)
        detection_options_v.addWidget(rb)
    
    # Pack the AI Hunter container
    detection_options_v.addWidget(self.ai_hunter_container)
    
    section_v.addWidget(self.detection_options_container)
    
    # Initial visibility updates
    update_detection_visibility()
    update_ai_hunter_visibility()
    
    # Retry Slow
    retry_slow_cb = self._create_styled_checkbox("Auto-retry Slow Processing (API Timeouts)")
    retry_slow_cb.setContentsMargins(0, 15, 0, 0)
    try:
        retry_slow_cb.setChecked(bool(self.retry_timeout_var))
    except Exception:
        pass
    def _on_retry_slow_toggle(checked):
        try:
            self.retry_timeout_var = bool(checked)
            timeout_edit.setEnabled(bool(checked))
            if checked:
                timeout_edit.setStyleSheet("")  # default enabled style
            else:
                timeout_edit.setStyleSheet("color: #888; background-color: #1f1f1f;")
        except Exception:
            pass
    retry_slow_cb.toggled.connect(_on_retry_slow_toggle)
    section_v.addWidget(retry_slow_cb)
    
    timeout_w = QWidget()
    timeout_h = QHBoxLayout(timeout_w)
    timeout_h.setContentsMargins(20, 5, 0, 0)
    timeout_h.addWidget(QLabel("Timeout after"))
    timeout_edit = QLineEdit()
    timeout_edit.setFixedWidth(60)
    try:
        timeout_edit.setText(str(self.chunk_timeout_var))
    except Exception:
        pass
    def _on_timeout_changed(text):
        try:
            self.chunk_timeout_var = text
        except Exception:
            pass
    timeout_edit.textChanged.connect(_on_timeout_changed)
    timeout_h.addWidget(timeout_edit)
    timeout_h.addWidget(QLabel("seconds"))
    timeout_h.addStretch()
    section_v.addWidget(timeout_w)
    
    timeout_desc = QLabel("Adds API timeout logic to text/images chunks that take too long\nThis will also affect chapter extraction timeout")
    timeout_desc.setStyleSheet("color: gray; font-size: 10pt;")
    timeout_desc.setContentsMargins(20, 0, 0, 5)
    section_v.addWidget(timeout_desc)
    # Apply initial styling based on current toggle state
    try:
        timeout_edit.setEnabled(bool(self.retry_timeout_var))
        if bool(self.retry_timeout_var):
            timeout_edit.setStyleSheet("")
        else:
            timeout_edit.setStyleSheet("color: #888; background-color: #1f1f1f;")
    except Exception:
        pass
    
    # Separator
    sep7 = QFrame()
    sep7.setFrameShape(QFrame.HLine)
    sep7.setFrameShadow(QFrame.Sunken)
    section_v.addWidget(sep7)
    
    # HTTP Timeouts & Connection Pooling
    http_title = QLabel("HTTP Timeouts & Connection Pooling")
    http_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
    section_v.addWidget(http_title)
    
    http_main = QWidget()
    http_main_v = QVBoxLayout(http_main)
    http_main_v.setContentsMargins(20, 5, 0, 0)
    
    # Master toggle to enable/disable all HTTP tuning fields
    if not hasattr(self, 'enable_http_tuning_var'):
        self.enable_http_tuning_var = self.config.get('enable_http_tuning', False)
    
    self.http_tuning_checkbox = self._create_styled_checkbox("Enable HTTP timeout/pooling overrides")
    try:
        self.http_tuning_checkbox.setChecked(bool(self.enable_http_tuning_var))
    except Exception:
        pass
    def _on_http_tuning_toggle(checked):
        try:
            self.enable_http_tuning_var = bool(checked)
            if hasattr(self, '_toggle_http_tuning_controls'):
                self._toggle_http_tuning_controls()
        except Exception:
            pass
    self.http_tuning_checkbox.toggled.connect(_on_http_tuning_toggle)
    http_main_v.addWidget(self.http_tuning_checkbox)
    http_main_v.addSpacing(8)
    
    # 2 column grid layout for more compact display
    if not hasattr(self, 'connect_timeout_var'):
        self.connect_timeout_var = str(self.config.get('connect_timeout', os.environ.get('CONNECT_TIMEOUT', '10')))
    if not hasattr(self, 'read_timeout_var'):
        self.read_timeout_var = str(self.config.get('read_timeout', os.environ.get('READ_TIMEOUT', os.environ.get('CHUNK_TIMEOUT', '180'))))
    if not hasattr(self, 'http_pool_connections_var'):
        self.http_pool_connections_var = str(self.config.get('http_pool_connections', os.environ.get('HTTP_POOL_CONNECTIONS', '20')))
    if not hasattr(self, 'http_pool_maxsize_var'):
        self.http_pool_maxsize_var = str(self.config.get('http_pool_maxsize', os.environ.get('HTTP_POOL_MAXSIZE', '50')))
    
    # Create grid for 2-column layout
    http_grid = QWidget()
    http_grid_layout = QGridLayout(http_grid)
    http_grid_layout.setContentsMargins(0, 0, 0, 5)
    http_grid_layout.setHorizontalSpacing(0)  # No spacing between label and input
    http_grid_layout.setVerticalSpacing(5)
    
    # Row 0: Connect timeout and Read timeout
    self.connect_timeout_label = QLabel("Connect timeout (s): ")
    http_grid_layout.addWidget(self.connect_timeout_label, 0, 0, alignment=Qt.AlignRight)
    self.connect_timeout_entry = QLineEdit()
    self.connect_timeout_entry.setFixedWidth(70)
    try:
        self.connect_timeout_entry.setText(str(self.connect_timeout_var))
    except Exception:
        pass
    def _on_connect_timeout_changed(text):
        try:
            self.connect_timeout_var = text
        except Exception:
            pass
    self.connect_timeout_entry.textChanged.connect(_on_connect_timeout_changed)
    http_grid_layout.addWidget(self.connect_timeout_entry, 0, 1, alignment=Qt.AlignLeft)
    
    self.read_timeout_label = QLabel(" Read timeout (s): ")
    http_grid_layout.addWidget(self.read_timeout_label, 0, 2, alignment=Qt.AlignRight)
    self.read_timeout_entry = QLineEdit()
    self.read_timeout_entry.setFixedWidth(70)
    try:
        self.read_timeout_entry.setText(str(self.read_timeout_var))
    except Exception:
        pass
    def _on_read_timeout_changed(text):
        try:
            self.read_timeout_var = text
        except Exception:
            pass
    self.read_timeout_entry.textChanged.connect(_on_read_timeout_changed)
    http_grid_layout.addWidget(self.read_timeout_entry, 0, 3, alignment=Qt.AlignLeft)
    
    # Row 1: Pool connections and Pool max size
    self.http_pool_connections_label = QLabel("Pool connections: ")
    http_grid_layout.addWidget(self.http_pool_connections_label, 1, 0, alignment=Qt.AlignRight)
    self.http_pool_connections_entry = QLineEdit()
    self.http_pool_connections_entry.setFixedWidth(70)
    try:
        self.http_pool_connections_entry.setText(str(self.http_pool_connections_var))
    except Exception:
        pass
    def _on_pool_conn_changed(text):
        try:
            self.http_pool_connections_var = text
        except Exception:
            pass
    self.http_pool_connections_entry.textChanged.connect(_on_pool_conn_changed)
    http_grid_layout.addWidget(self.http_pool_connections_entry, 1, 1, alignment=Qt.AlignLeft)
    
    self.http_pool_maxsize_label = QLabel(" Pool max size: ")
    http_grid_layout.addWidget(self.http_pool_maxsize_label, 1, 2, alignment=Qt.AlignRight)
    self.http_pool_maxsize_entry = QLineEdit()
    self.http_pool_maxsize_entry.setFixedWidth(70)
    try:
        self.http_pool_maxsize_entry.setText(str(self.http_pool_maxsize_var))
    except Exception:
        pass
    def _on_pool_maxsize_changed(text):
        try:
            self.http_pool_maxsize_var = text
        except Exception:
            pass
    self.http_pool_maxsize_entry.textChanged.connect(_on_pool_maxsize_changed)
    http_grid_layout.addWidget(self.http_pool_maxsize_entry, 1, 3, alignment=Qt.AlignLeft)
    
    http_main_v.addWidget(http_grid)
    
    # Optional toggle: ignore server Retry-After header
    if not hasattr(self, 'ignore_retry_after_var'):
        self.ignore_retry_after_var = bool(self.config.get('ignore_retry_after', str(os.environ.get('IGNORE_RETRY_AFTER', '0')) == '1'))
    
    self.ignore_retry_after_checkbox = self._create_styled_checkbox("Ignore server Retry-After header (use local backoff)")
    try:
        self.ignore_retry_after_checkbox.setChecked(bool(self.ignore_retry_after_var))
    except Exception:
        pass
    def _on_ignore_retry_after_toggle(checked):
        try:
            self.ignore_retry_after_var = bool(checked)
        except Exception:
            pass
    self.ignore_retry_after_checkbox.toggled.connect(_on_ignore_retry_after_toggle)
    http_main_v.addSpacing(6)
    http_main_v.addWidget(self.ignore_retry_after_checkbox)
    
    section_v.addWidget(http_main)
    
    # Apply initial enable/disable state
    if hasattr(self, '_toggle_http_tuning_controls'):
        self._toggle_http_tuning_controls()
    
    http_desc = QLabel("Controls network behavior for connection establishment timeout, read timeout,\nHTTP connection pool sizes.")
    http_desc.setStyleSheet("color: gray; font-size: 10pt;")
    http_desc.setContentsMargins(20, 2, 0, 5)
    section_v.addWidget(http_desc)
    
    # Separator
    sep8 = QFrame()
    sep8.setFrameShape(QFrame.HLine)
    sep8.setFrameShadow(QFrame.Sunken)
    section_v.addWidget(sep8)
    
    # Max Retries Configuration
    retries_title = QLabel("API Request Retries")
    retries_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
    section_v.addWidget(retries_title)
    
    retries_w = QWidget()
    retries_h = QHBoxLayout(retries_w)
    retries_h.setContentsMargins(20, 5, 0, 0)
    
    # Create MAX_RETRIES variable if it doesn't exist
    if not hasattr(self, 'max_retries_var'):
        self.max_retries_var = str(self.config.get('max_retries', os.environ.get('MAX_RETRIES', '7')))
    
    retries_h.addWidget(QLabel("Maximum retry attempts:"))
    max_retries_edit = QLineEdit()
    max_retries_edit.setFixedWidth(40)
    try:
        max_retries_edit.setText(str(self.max_retries_var))
    except Exception:
        pass
    def _on_max_retries_changed(text):
        try:
            self.max_retries_var = text
        except Exception:
            pass
    max_retries_edit.textChanged.connect(_on_max_retries_changed)
    retries_h.addWidget(max_retries_edit)
    retries_h.addWidget(QLabel("(default: 7)"))
    retries_h.addStretch()
    section_v.addWidget(retries_w)
    
    retries_desc = QLabel("Number of times to retry failed API requests before giving up.\nApplies to all API providers (OpenAI, Gemini, Anthropic, etc.)")
    retries_desc.setStyleSheet("color: gray; font-size: 10pt;")
    retries_desc.setContentsMargins(20, 2, 0, 10)
    section_v.addWidget(retries_desc)
    
    # Indefinite Rate Limit Retry toggle
    indefinite_retry_cb = self._create_styled_checkbox("Indefinite Rate Limit Retry")
    indefinite_retry_cb.setContentsMargins(20, 0, 0, 0)
    try:
        indefinite_retry_cb.setChecked(bool(self.indefinite_rate_limit_retry_var))
    except Exception:
        pass
    def _on_indefinite_retry_toggle(checked):
        try:
            self.indefinite_rate_limit_retry_var = bool(checked)
        except Exception:
            pass
    indefinite_retry_cb.toggled.connect(_on_indefinite_retry_toggle)
    section_v.addWidget(indefinite_retry_cb)
    
    indefinite_desc = QLabel("When enabled, rate limit errors (429) will retry indefinitely with exponential backoff.\nWhen disabled, rate limits count against the maximum retry attempts above.")
    indefinite_desc.setStyleSheet("color: gray; font-size: 10pt;")
    indefinite_desc.setContentsMargins(40, 2, 0, 5)
    section_v.addWidget(indefinite_desc)
    
    # Add Halgakos icon under the description (HiDPI-aware 90x90, centered)
    import os
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico')
    if os.path.exists(icon_path):
        from PySide6.QtGui import QIcon, QPixmap
        from PySide6.QtCore import QSize
        icon_label = QLabel()
        icon_label.setStyleSheet("background-color: transparent;")
        try:
            dpr = self.devicePixelRatioF()
        except Exception:
            dpr = 1.0
        logical_px = 90
        dev_px = int(logical_px * max(1.0, dpr))
        icon = QIcon(icon_path)
        pm = icon.pixmap(QSize(dev_px, dev_px))
        if pm.isNull():
            raw = QPixmap(icon_path)
            img = raw.toImage().scaled(dev_px, dev_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pm = QPixmap.fromImage(img)
        try:
            pm.setDevicePixelRatio(dpr)
        except Exception:
            pass
        # Fit into logical size while preserving DPR
        pm_fitted = pm.scaled(int(logical_px * dpr), int(logical_px * dpr),
                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
        try:
            pm_fitted.setDevicePixelRatio(dpr)
        except Exception:
            pass
        icon_label.setPixmap(pm_fitted)
        icon_label.setFixedSize(logical_px, logical_px)
        icon_label.setAlignment(Qt.AlignCenter)
        section_v.addWidget(icon_label, alignment=Qt.AlignCenter)
    
    # Place the section at row 1, column 0 to match the original grid
    try:
        grid = parent.layout()
        if grid:
            grid.addWidget(section_box, 1, 0)
    except Exception:
        # Fallback: just stack
        section_box.setParent(parent)
    
            
def open_multi_api_key_manager(self):
    """Open the multi API key manager dialog"""
    from PySide6.QtWidgets import QMessageBox
    # Import here to avoid circular imports
    try:
        from multi_api_key_manager import MultiAPIKeyDialog
        
        # Use the static show_dialog method which handles PySide6 properly
        # This blocks until the dialog is closed
        MultiAPIKeyDialog.show_dialog(self.master, self)
        
        # Refresh the settings display if in settings dialog
        if hasattr(self, 'current_settings_dialog'):
            try:
                # Close and reopen settings to refresh (tkinter part)
                self.current_settings_dialog.destroy()
                self.show_settings()  # or open_other_settings()
            except Exception:
                pass
            
    except ImportError as e:
        QMessageBox.critical(None, "Error", f"Failed to load Multi API Key Manager: {str(e)}")
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Error opening Multi API Key Manager: {str(e)}")
        import traceback
        traceback.print_exc()

# DEPRECATED Tkinter function - not used in PySide6 version
# def _create_multi_key_row(self, parent):
#     """Create a compact multi-key configuration row"""
#     frame = tk.Frame(parent)
#     frame.pack(fill=tk.X, pady=5)
#     
#     # Status indicator
#     if self.config.get('use_multi_api_keys', False):
#         keys = self.config.get('multi_api_keys', [])
#         active = sum(1 for k in keys if k.get('enabled', True))
#         
#         # Checkbox to enable/disable
#         tb.Checkbutton(frame, text="Multi API Key Mode", 
#                       variable=self.use_multi_api_keys_var,
#                       bootstyle="round-toggle",
#                       command=self._toggle_multi_key_setting).pack(side=tk.LEFT)
#         
#         # Status
#         tk.Label(frame, text=f"({active}/{len(keys)} active)", 
#                 font=('TkDefaultFont', 10), fg='green').pack(side=tk.LEFT, padx=(5, 0))
#     else:
#         tb.Checkbutton(frame, text="Multi API Key Mode", 
#                       variable=self.use_multi_api_keys_var,
#                       bootstyle="round-toggle",
#                       command=self._toggle_multi_key_setting).pack(side=tk.LEFT)
#     
#     # Configure button
#     tb.Button(frame, text="Configure Keys...", 
#               command=self.open_multi_api_key_manager,
#               bootstyle="primary-outline").pack(side=tk.LEFT, padx=(20, 0))
#     
#     return frame
            
def _update_multi_key_status_label(self):
    """Update the multi-key mode status label dynamically"""
    try:
        if not hasattr(self, 'multi_key_status_label2'):
            return
        
        if self.config.get('use_multi_api_keys', False):
            multi_keys = self.config.get('multi_api_keys', [])
            active_keys = sum(1 for k in multi_keys if k.get('enabled', True))
            self.multi_key_status_label2.setText(f"ACTIVE ({active_keys}/{len(multi_keys)} keys)")
            self.multi_key_status_label2.setStyleSheet("font-weight: bold; font-size: 11pt; color: green;")
        else:
            self.multi_key_status_label2.setText("DISABLED")
            self.multi_key_status_label2.setStyleSheet("font-size: 11pt; color: gray;")
    except Exception:
        pass

def _toggle_multi_key_setting(self):
    """Toggle multi-key mode from settings dialog"""
    self.config['use_multi_api_keys'] = self.use_multi_api_keys_var
    # Don't save immediately, let the dialog's save button handle it

def toggle_image_translation_section(self):
    """Toggle visibility of image translation content with smooth fade"""
    try:
        if not hasattr(self, 'image_translation_content'):
            return
            
        enabled = bool(self.enable_image_translation_var)
        
        # Import animation components
        from PySide6.QtCore import QPropertyAnimation, QEasingCurve
        from PySide6.QtWidgets import QGraphicsOpacityEffect
        
        # Stop any existing animation
        if hasattr(self, '_image_section_animation') and self._image_section_animation:
            self._image_section_animation.stop()
        
        # Ensure widget is visible for animation
        self.image_translation_content.setVisible(True)
        
        # Create or get opacity effect
        if not hasattr(self.image_translation_content, '_opacity_effect'):
            effect = QGraphicsOpacityEffect()
            self.image_translation_content.setGraphicsEffect(effect)
            self.image_translation_content._opacity_effect = effect
        else:
            effect = self.image_translation_content._opacity_effect
        
        # Create opacity animation
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(150)  # Faster for no glitch
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        if enabled:
            # Fade in
            animation.setStartValue(0.0)
            animation.setEndValue(1.0)
        else:
            # Fade out
            animation.setStartValue(1.0)
            animation.setEndValue(0.0)
            # Hide after fade out
            animation.finished.connect(
                lambda: self.image_translation_content.setVisible(False) if not enabled else None
            )
        
        self._image_section_animation = animation
        animation.start()
        
    except Exception:
        # Fallback to simple show/hide if animation fails
        try:
            if hasattr(self, 'image_translation_content'):
                enabled = bool(self.enable_image_translation_var)
                self.image_translation_content.setVisible(enabled)
        except Exception:
            pass

def toggle_extraction_workers(self):
    """Enable/disable extraction workers entry and labels based on toggle (PySide6 version)"""
    try:
        enabled = bool(self.enable_parallel_extraction_var)
        
        # Workers entry
        if hasattr(self, 'extraction_workers_entry'):
            self.extraction_workers_entry.setEnabled(enabled)
            
        # Workers and threads labels
        if hasattr(self, 'workers_label'):
            self.workers_label.setEnabled(enabled)
            color = "white" if enabled else "#808080"
            self.workers_label.setStyleSheet(f"color: {color};")
            
        if hasattr(self, 'threads_label'):
            self.threads_label.setEnabled(enabled)
            color = "white" if enabled else "#808080"
            self.threads_label.setStyleSheet(f"color: {color};")
            
        # Description label
        if hasattr(self, 'parallel_desc_label'):
            self.parallel_desc_label.setEnabled(enabled)
            color = "gray" if enabled else "#606060"
            self.parallel_desc_label.setStyleSheet(f"color: {color}; font-size: 10pt;")
        
        if enabled:
            # Set environment variable
            os.environ["EXTRACTION_WORKERS"] = str(self.extraction_workers_var)
        else:
            # Set to 1 worker (sequential) when disabled
            os.environ["EXTRACTION_WORKERS"] = "1"
        
        # Ensure executor reflects current worker setting
        try:
            self._ensure_executor()
        except Exception:
            pass
    except Exception:
        pass

# DEPRECATED Tkinter function - not used in PySide6 version
# def _setup_provider_combobox_bindings(self):
#     """Setup bindings for OpenRouter provider combobox with autocomplete"""
#     try:
#         # Bind to key release events for live filtering and autofill
#         self.openrouter_provider_combo.bind('<KeyRelease>', self._on_provider_combo_keyrelease)
#         # Commit best match on Enter
#         self.openrouter_provider_combo.bind('<Return>', self._commit_provider_autocomplete)
#         # Also bind to FocusOut to validate selection
#         self.openrouter_provider_combo.bind('<FocusOut>', lambda e: self._validate_provider_selection())
#     except Exception:
#         pass  # Silently fail if combo doesn't exist

# DEPRECATED Tkinter function - not used in PySide6 version  
# def _on_provider_combo_keyrelease(self, event=None):
#     """Provider combobox type-to-search with autocomplete (reuses model dropdown logic)"""
#     try:
#         combo = self.openrouter_provider_combo
#         typed = combo.get()
#         prev = getattr(self, '_provider_prev_text', '')
#         keysym = (getattr(event, 'keysym', '') or '').lower()
# 
#         # Navigation/commit keys: don't interfere
#         if keysym in {'up', 'down', 'left', 'right', 'return', 'escape', 'tab'}:
#             return
# 
#         # Ensure we have the full source list
#         source = getattr(self, '_provider_all_values', [])
#         if not source:
#             return
# 
#         # Compute match set
#         first_match = None
#         if typed:
#             lowered = typed.lower()
#             # Prefix matches first
#             pref = [v for v in source if v.lower().startswith(lowered)]
#             # Contains matches second
#             cont = [v for v in source if lowered in v.lower() and v not in pref]
#             if pref:
#                 first_match = pref[0]
#             elif cont:
#                 first_match = cont[0]
# 
#         # Decide whether to autofill
#         grew = len(typed) > len(prev) and typed.startswith(prev)
#         is_deletion = keysym in {'backspace', 'delete'} or len(typed) < len(prev)
#         try:
#             at_end = combo.index(tk.INSERT) == len(typed)
#         except Exception:
#             at_end = True
#         try:
#             has_selection = combo.selection_present()
#         except Exception:
#             has_selection = False
# 
#         # Gentle autofill only when appending at the end
#         do_autofill_text = first_match is not None and grew and at_end and not has_selection and not is_deletion
# 
#         if do_autofill_text:
#             # Only complete if it's a true prefix match
#             if first_match.lower().startswith(typed.lower()) and first_match != typed:
#                 combo.set(first_match)
#                 try:
#                     combo.icursor(len(typed))
#                     combo.selection_range(len(typed), len(first_match))
#                 except Exception:
#                     pass
# 
#         # If we have a match and the dropdown is open, scroll/highlight it
#         if first_match:
#             self._scroll_provider_list_to_value(first_match)
# 
#         # Remember current text for next event
#         self._provider_prev_text = typed
#     except Exception:
#         pass  # Silently handle errors

def _commit_provider_autocomplete(self, event=None):
    """On Enter, commit to the best matching provider"""
    try:
        combo = self.openrouter_provider_combo
        typed = combo.get()
        source = getattr(self, '_provider_all_values', [])
        match = None
        if typed:
            lowered = typed.lower()
            pref = [v for v in source if v.lower().startswith(lowered)]
            cont = [v for v in source if lowered in v.lower()] if not pref else []
            match = pref[0] if pref else (cont[0] if cont else None)
        if match and match != typed:
            combo.set(match)
        # Move cursor to end and clear any selection
        try:
            combo.icursor('end')
            try:
                combo.selection_clear()
            except Exception:
                combo.selection_range(0, 0)
        except Exception:
            pass
        # Update prev text
        self._provider_prev_text = combo.get()
    except Exception:
        pass
    return "break"

def _scroll_provider_list_to_value(self, value: str):
    """If the provider combobox dropdown is open, scroll to and highlight the given value"""
    try:
        values = getattr(self, '_provider_all_values', [])
        if value not in values:
            return
        index = values.index(value)
        # Resolve the internal popdown listbox for this combobox
        popdown = self.openrouter_provider_combo.tk.eval(
            f'ttk::combobox::PopdownWindow {self.openrouter_provider_combo._w}'
        )
        listbox = f'{popdown}.f.l'
        tkobj = self.openrouter_provider_combo.tk
        # Scroll and highlight the item
        tkobj.call(listbox, 'see', index)
        tkobj.call(listbox, 'selection', 'clear', 0, 'end')
        tkobj.call(listbox, 'selection', 'set', index)
        tkobj.call(listbox, 'activate', index)
    except Exception:
        pass  # Dropdown may be closed or internals unavailable

def _validate_provider_selection(self):
    """Validate that the provider selection is from the list or default to Auto"""
    try:
        typed = self.openrouter_preferred_provider_var
        source = getattr(self, '_provider_all_values', [])
        if typed and typed not in source:
            # Find closest match or default to Auto
            lowered = typed.lower()
            matches = [v for v in source if lowered in v.lower()]
            if matches:
                self.openrouter_preferred_provider_var = matches[0]
            else:
                self.openrouter_preferred_provider_var = 'Auto'
    except Exception:
        pass
    
def create_ai_hunter_section(self, parent_frame):
    """Create the AI Hunter configuration section (Qt version)"""
    from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
    
    # Config row
    config_w = QWidget()
    config_h = QHBoxLayout(config_w)
    config_h.setContentsMargins(20, 10, 0, 5)
    
    # Status label
    self.ai_hunter_status_label = QLabel(self._get_ai_hunter_status_text())
    self.ai_hunter_status_label.setStyleSheet("font-size: 10pt;")
    config_h.addWidget(self.ai_hunter_status_label)
    
    config_h.addSpacing(10)
    
    # Configure button
    config_btn = QPushButton("⚙️ Configure AI Hunter")
    config_btn.clicked.connect(lambda: self.show_ai_hunter_settings())
    config_h.addWidget(config_btn)
    config_h.addStretch()
    
    # Add to parent layout (parent_frame should be a QVBoxLayout)
    if hasattr(parent_frame, 'layout'):
        # parent_frame is a QWidget, get its layout
        layout = parent_frame.layout()
        if layout:
            layout.addWidget(config_w)
    elif hasattr(parent_frame, 'addWidget'):
        # parent_frame is a QLayout
        parent_frame.addWidget(config_w)
    
    # Info text
    info_lbl = QLabel("AI Hunter uses multiple detection methods to identify duplicate content\nwith configurable thresholds and detection modes")
    info_lbl.setStyleSheet("color: gray; font-size: 10pt;")
    info_lbl.setContentsMargins(20, 0, 0, 10)
    if hasattr(parent_frame, 'layout'):
        # parent_frame is a QWidget, get its layout
        layout = parent_frame.layout()
        if layout:
            layout.addWidget(info_lbl)
    elif hasattr(parent_frame, 'addWidget'):
        # parent_frame is a QLayout
        parent_frame.addWidget(info_lbl)

def _get_ai_hunter_status_text(self):
    """Get status text for AI Hunter configuration"""
    ai_config = self.config.get('ai_hunter_config', {})
    
    # AI Hunter is shown when the detection mode is set to 'ai-hunter' or 'cascading'
    if self.duplicate_detection_mode_var not in ['ai-hunter', 'cascading']:
        return "AI Hunter: Not Selected"
    
    if not ai_config.get('enabled', True):
        return "AI Hunter: Disabled in Config"
    
    mode_text = {
        'single_method': 'Single Method',
        'multi_method': 'Multi-Method',
        'weighted_average': 'Weighted Average'
    }
    
    mode = mode_text.get(ai_config.get('detection_mode', 'multi_method'), 'Unknown')
    thresholds = ai_config.get('thresholds', {})
    
    if thresholds:
        avg_threshold = sum(thresholds.values()) / len(thresholds)
    else:
        avg_threshold = 85
    
    return f"AI Hunter: {mode} mode, Avg threshold: {int(avg_threshold)}%"

def show_ai_hunter_settings(self):
    """Open AI Hunter configuration window (PySide6)"""
    try:
        def on_config_saved():
            # Save the entire configuration (without showing message box)
            self.save_config(show_message=False)
            # Update status label if it still exists
            if hasattr(self, 'ai_hunter_status_label'):
                try:
                    if not self.ai_hunter_status_label.isHidden():
                        self.ai_hunter_status_label.setText(self._get_ai_hunter_status_text())
                except RuntimeError:
                    # Widget has been destroyed
                    pass
            if hasattr(self, 'ai_hunter_enabled_var'):
                self.ai_hunter_enabled_var = self.config.get('ai_hunter_config', {}).get('enabled', True)
        
        # Store reference to prevent garbage collection
        self._ai_hunter_gui = AIHunterConfigGUI(None, self.config, on_config_saved)
        self._ai_hunter_gui.show_ai_hunter_config()
    except Exception as e:
        print(f"Error opening AI Hunter settings: {e}")
        import traceback
        traceback.print_exc()
        from PySide6.QtWidgets import QMessageBox
        from PySide6.QtGui import QIcon
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Error")
        msg_box.setText(f"Failed to open AI Hunter settings:\n{str(e)}")
        msg_box.setIcon(QMessageBox.Critical)
        try:
            msg_box.setWindowIcon(QIcon("halgakos.ico"))
        except Exception:
            pass
        _center_messagebox_buttons(msg_box)
        msg_box.exec()

def configure_translation_chunk_prompt(self):
    """Configure the prompt template for translation chunks (PySide6)"""
    from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QGroupBox, QWidget, QMessageBox
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon
    
    dialog = QDialog(None)
    dialog.setWindowTitle("Configure Chunk Prompt")
    # Use screen ratios for sizing
    from PySide6.QtWidgets import QApplication
    screen = QApplication.primaryScreen().geometry()
    width = int(screen.width() * 0.36)  # 36% of screen width
    height = int(screen.height() * 0.56)  # 56% of screen height
    dialog.resize(width, height)
    
    # Set icon
    try:
        dialog.setWindowIcon(QIcon("halgakos.ico"))
    except Exception:
        pass
    
    main_layout = QVBoxLayout(dialog)
    main_layout.setContentsMargins(20, 20, 20, 20)
    
    # Title
    title = QLabel("Translation Chunk Prompt Template")
    title.setStyleSheet("font-size: 14pt; font-weight: bold;")
    main_layout.addWidget(title)
    
    desc = QLabel("Configure how chunks are presented to the AI when chapters are split.")
    desc.setStyleSheet("color: gray; font-size: 10pt;")
    main_layout.addWidget(desc)
    main_layout.addSpacing(10)
    
    # Instructions
    instructions_box = QGroupBox("Available Placeholders")
    instructions_v = QVBoxLayout(instructions_box)
    
    placeholders = [
        ("{chunk_idx}", "Current chunk number (1-based)"),
        ("{total_chunks}", "Total number of chunks"),
        ("{chunk_html}", "The actual HTML content to translate")
    ]
    
    for placeholder, description in placeholders:
        placeholder_lbl = QLabel(f"• <b>{placeholder}:</b> {description}")
        placeholder_lbl.setStyleSheet("font-family: Courier; font-size: 10pt;")
        instructions_v.addWidget(placeholder_lbl)
    
    main_layout.addWidget(instructions_box)
    
    # Prompt input
    prompt_box = QGroupBox("Chunk Prompt Template")
    prompt_v = QVBoxLayout(prompt_box)
    
    chunk_prompt_text = QTextEdit()
    chunk_prompt_text.setAcceptRichText(False)
    chunk_prompt_text.setPlainText(self.translation_chunk_prompt)
    prompt_v.addWidget(chunk_prompt_text)
    
    main_layout.addWidget(prompt_box)
    
    # Example
    example_box = QGroupBox("Example Output")
    example_v = QVBoxLayout(example_box)
    
    example_desc = QLabel("With chunk 2 of 5, the prompt would be:")
    example_v.addWidget(example_desc)
    
    example_label = QLabel()
    example_label.setStyleSheet("color: #5a9fd4; font-family: Courier; font-size: 9pt; font-style: italic;")
    example_label.setWordWrap(True)
    example_v.addWidget(example_label)
    
    def update_example():
        try:
            template = chunk_prompt_text.toPlainText()
            example = template.replace('{chunk_idx}', '2').replace('{total_chunks}', '5').replace('{chunk_html}', '<p>Chapter content here...</p>')
            display_text = example[:200] + "..." if len(example) > 200 else example
            example_label.setText(display_text)
        except Exception:
            example_label.setText("[Invalid template]")
    
    chunk_prompt_text.textChanged.connect(update_example)
    update_example()
    
    main_layout.addWidget(example_box)
    
    # Buttons
    button_layout = QHBoxLayout()
    
    def save_chunk_prompt():
        self.translation_chunk_prompt = chunk_prompt_text.toPlainText().strip()
        self.config['translation_chunk_prompt'] = self.translation_chunk_prompt
        QMessageBox.information(dialog, "Success", "Translation chunk prompt saved!")
        dialog.close()
    
    def reset_chunk_prompt():
        result = QMessageBox.question(dialog, "Reset Prompt", "Reset to default chunk prompt?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if result == QMessageBox.Yes:
            chunk_prompt_text.setPlainText(self.default_translation_chunk_prompt)
            update_example()
    
    save_btn = QPushButton("Save")
    save_btn.clicked.connect(save_chunk_prompt)
    button_layout.addWidget(save_btn)
    
    reset_btn = QPushButton("Reset to Default")
    reset_btn.clicked.connect(reset_chunk_prompt)
    button_layout.addWidget(reset_btn)
    
    cancel_btn = QPushButton("Cancel")
    cancel_btn.clicked.connect(dialog.close)
    button_layout.addWidget(cancel_btn)
    
    main_layout.addLayout(button_layout)
    
    dialog.show()

def configure_image_chunk_prompt(self):
    """Configure the prompt template for image chunks (PySide6)"""
    from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QGroupBox, QWidget, QMessageBox
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon
    
    dialog = QDialog(None)
    dialog.setWindowTitle("Configure Image Chunk Prompt")
    # Use screen ratios for sizing
    from PySide6.QtWidgets import QApplication
    screen = QApplication.primaryScreen().geometry()
    width = int(screen.width() * 0.36)  # 36% of screen width
    height = int(screen.height() * 0.56)  # 56% of screen height
    dialog.resize(width, height)
    
    # Set icon
    try:
        dialog.setWindowIcon(QIcon("halgakos.ico"))
    except Exception:
        pass
    
    main_layout = QVBoxLayout(dialog)
    main_layout.setContentsMargins(20, 20, 20, 20)
    
    # Title
    title = QLabel("Image Chunk Context Template")
    title.setStyleSheet("font-size: 14pt; font-weight: bold;")
    main_layout.addWidget(title)
    
    desc = QLabel("Configure the context provided when tall images are split into chunks.")
    desc.setStyleSheet("color: gray; font-size: 10pt;")
    main_layout.addWidget(desc)
    main_layout.addSpacing(10)
    
    # Instructions
    instructions_box = QGroupBox("Available Placeholders")
    instructions_v = QVBoxLayout(instructions_box)
    
    placeholders = [
        ("{chunk_idx}", "Current chunk number (1-based)"),
        ("{total_chunks}", "Total number of chunks"),
        ("{context}", "Additional context (e.g., chapter info)")
    ]
    
    for placeholder, description in placeholders:
        placeholder_lbl = QLabel(f"• <b>{placeholder}:</b> {description}")
        placeholder_lbl.setStyleSheet("font-family: Courier; font-size: 10pt;")
        instructions_v.addWidget(placeholder_lbl)
    
    main_layout.addWidget(instructions_box)
    
    # Prompt input
    prompt_box = QGroupBox("Image Chunk Prompt Template")
    prompt_v = QVBoxLayout(prompt_box)
    
    image_chunk_prompt_text = QTextEdit()
    image_chunk_prompt_text.setAcceptRichText(False)
    image_chunk_prompt_text.setPlainText(self.image_chunk_prompt)
    prompt_v.addWidget(image_chunk_prompt_text)
    
    main_layout.addWidget(prompt_box)
    
    # Example
    example_box = QGroupBox("Example Output")
    example_v = QVBoxLayout(example_box)
    
    example_desc = QLabel("With chunk 3 of 7 and chapter context, the prompt would be:")
    example_v.addWidget(example_desc)
    
    example_label = QLabel()
    example_label.setStyleSheet("color: #5a9fd4; font-family: Courier; font-size: 9pt; font-style: italic;")
    example_label.setWordWrap(True)
    example_v.addWidget(example_label)
    
    def update_image_example():
        try:
            template = image_chunk_prompt_text.toPlainText()
            example = template.replace('{chunk_idx}', '3').replace('{total_chunks}', '7').replace('{context}', 'Chapter 5: The Great Battle')
            example_label.setText(example)
        except Exception:
            example_label.setText("[Invalid template]")
    
    image_chunk_prompt_text.textChanged.connect(update_image_example)
    update_image_example()
    
    main_layout.addWidget(example_box)
    
    # Buttons
    button_layout = QHBoxLayout()
    
    def save_image_chunk_prompt():
        self.image_chunk_prompt = image_chunk_prompt_text.toPlainText().strip()
        self.config['image_chunk_prompt'] = self.image_chunk_prompt
        QMessageBox.information(dialog, "Success", "Image chunk prompt saved!")
        dialog.close()
    
    def reset_image_chunk_prompt():
        result = QMessageBox.question(dialog, "Reset Prompt", "Reset to default image chunk prompt?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if result == QMessageBox.Yes:
            image_chunk_prompt_text.setPlainText(self.default_image_chunk_prompt)
            update_image_example()
    
    save_btn = QPushButton("Save")
    save_btn.clicked.connect(save_image_chunk_prompt)
    button_layout.addWidget(save_btn)
    
    reset_btn = QPushButton("Reset to Default")
    reset_btn.clicked.connect(reset_image_chunk_prompt)
    button_layout.addWidget(reset_btn)
    
    cancel_btn = QPushButton("Cancel")
    cancel_btn.clicked.connect(dialog.close)
    button_layout.addWidget(cancel_btn)
    
    main_layout.addLayout(button_layout)
    
    dialog.show()

def configure_image_compression(self):
    """Open the image compression configuration dialog (PySide6)"""
    from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                                   QCheckBox, QLineEdit, QGroupBox, QRadioButton, QSlider, 
                                   QWidget, QScrollArea, QMessageBox, QFrame)
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon
    
    dialog = QDialog(None)
    dialog.setWindowTitle("Image Compression Settings")
    # Use screen ratios for sizing
    from PySide6.QtWidgets import QApplication
    screen = QApplication.primaryScreen().geometry()
    width = int(screen.width() * 0.34)  # 34% of screen width
    height = int(screen.height() * 0.65)  # 65% of screen height
    dialog.resize(width, height)
    
    # Set icon
    try:
        dialog.setWindowIcon(QIcon("halgakos.ico"))
    except Exception:
        pass
    
    # Apply global stylesheet for checkboxes and radio buttons
    checkbox_radio_style = """
        QCheckBox {
            color: white;
            spacing: 6px;
        }
        QCheckBox::indicator {
            width: 14px;
            height: 14px;
            border: 1px solid #5a9fd4;
            border-radius: 2px;
            background-color: #2d2d2d;
        }
        QCheckBox::indicator:checked {
            background-color: #5a9fd4;
            border-color: #5a9fd4;
        }
        QCheckBox::indicator:hover {
            border-color: #7bb3e0;
        }
        QCheckBox:disabled {
            color: #666666;
        }
        QCheckBox::indicator:disabled {
            background-color: #1a1a1a;
            border-color: #3a3a3a;
        }
        QRadioButton {
            color: white;
            spacing: 5px;
        }
        QRadioButton::indicator {
            width: 13px;
            height: 13px;
            border: 2px solid #5a9fd4;
            border-radius: 7px;
            background-color: #2d2d2d;
        }
        QRadioButton::indicator:checked {
            background-color: #5a9fd4;
            border: 2px solid #5a9fd4;
        }
        QRadioButton::indicator:hover {
            border-color: #7bb3e0;
        }
        QRadioButton:disabled {
            color: #666666;
        }
        QRadioButton::indicator:disabled {
            background-color: #1a1a1a;
            border-color: #3a3a3a;
        }
    """
    
    # Scrollable area
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    
    scroll_widget = QWidget()
    scroll_widget.setStyleSheet(checkbox_radio_style)  # Apply stylesheet to scroll widget
    main_layout = QVBoxLayout(scroll_widget)
    main_layout.setContentsMargins(20, 20, 20, 20)
    
    # Title
    title = QLabel("🗜️ Image Compression Settings")
    title.setStyleSheet("font-size: 14pt; font-weight: bold;")
    main_layout.addWidget(title)
    main_layout.addSpacing(15)
    
    # Enable compression toggle
    enable_cb = self._create_styled_checkbox("Enable Image Compression")
    enable_cb.setChecked(self.config.get('enable_image_compression', False))
    main_layout.addWidget(enable_cb)
    main_layout.addSpacing(20)
    
    # Container for all compression options
    compression_options = QWidget()
    options_layout = QVBoxLayout(compression_options)
    options_layout.setContentsMargins(0, 0, 0, 0)
    
    # Auto Compression Section
    auto_box = QGroupBox("Automatic Compression")
    auto_v = QVBoxLayout(auto_box)
    
    auto_compress_cb = self._create_styled_checkbox("Auto-compress to fit token limits")
    auto_compress_cb.setChecked(self.config.get('auto_compress_enabled', True))
    auto_v.addWidget(auto_compress_cb)
    
    token_w = QWidget()
    token_h = QHBoxLayout(token_w)
    token_h.setContentsMargins(0, 10, 0, 0)
    token_h.addWidget(QLabel("Target tokens per image:"))
    target_tokens_edit = QLineEdit(str(self.config.get('target_image_tokens', 1000)))
    target_tokens_edit.setFixedWidth(80)
    token_h.addWidget(target_tokens_edit)
    token_h.addWidget(QLabel("(Gemini uses ~258 tokens per image)"))
    token_h.addStretch()
    auto_v.addWidget(token_w)
    
    options_layout.addWidget(auto_box)
    
    # Format Selection
    format_box = QGroupBox("Output Format")
    format_v = QVBoxLayout(format_box)
    
    format_group = QWidget()
    format_buttons = []
    
    formats = [
        ("Auto (Best quality/size ratio)", "auto"),
        ("WebP (Best compression)", "webp"),
        ("JPEG (Wide compatibility)", "jpeg"),
        ("PNG (Lossless)", "png")
    ]
    
    current_format = self.config.get('image_compression_format', 'auto')
    for text, value in formats:
        rb = QRadioButton(text)
        rb.setProperty("format_value", value)
        if value == current_format:
            rb.setChecked(True)
        format_buttons.append(rb)
        format_v.addWidget(rb)
    
    options_layout.addWidget(format_box)
    
    # Quality Settings
    quality_box = QGroupBox("Quality Settings")
    quality_v = QVBoxLayout(quality_box)
    
    # WebP Quality
    webp_w = QWidget()
    webp_h = QHBoxLayout(webp_w)
    webp_h.addWidget(QLabel("WebP Quality:"))
    webp_slider = QSlider(Qt.Horizontal)
    webp_slider.setMinimum(1)
    webp_slider.setMaximum(100)
    webp_slider.setValue(self.config.get('webp_quality', 85))
    webp_slider.setFixedWidth(200)
    webp_h.addWidget(webp_slider)
    webp_label = QLabel(f"{webp_slider.value()}%")
    webp_slider.valueChanged.connect(lambda v: webp_label.setText(f"{v}%"))
    webp_h.addWidget(webp_label)
    webp_h.addStretch()
    quality_v.addWidget(webp_w)
    
    # JPEG Quality
    jpeg_w = QWidget()
    jpeg_h = QHBoxLayout(jpeg_w)
    jpeg_h.addWidget(QLabel("JPEG Quality:"))
    jpeg_slider = QSlider(Qt.Horizontal)
    jpeg_slider.setMinimum(1)
    jpeg_slider.setMaximum(100)
    jpeg_slider.setValue(self.config.get('jpeg_quality', 85))
    jpeg_slider.setFixedWidth(200)
    jpeg_h.addWidget(jpeg_slider)
    jpeg_label = QLabel(f"{jpeg_slider.value()}%")
    jpeg_slider.valueChanged.connect(lambda v: jpeg_label.setText(f"{v}%"))
    jpeg_h.addWidget(jpeg_label)
    jpeg_h.addStretch()
    quality_v.addWidget(jpeg_w)
    
    # PNG Compression
    png_w = QWidget()
    png_h = QHBoxLayout(png_w)
    png_h.addWidget(QLabel("PNG Compression:"))
    png_slider = QSlider(Qt.Horizontal)
    png_slider.setMinimum(0)
    png_slider.setMaximum(9)
    png_slider.setValue(self.config.get('png_compression', 6))
    png_slider.setFixedWidth(200)
    png_h.addWidget(png_slider)
    png_label = QLabel(f"Level {png_slider.value()}")
    png_slider.valueChanged.connect(lambda v: png_label.setText(f"Level {v}"))
    png_h.addWidget(png_label)
    png_h.addStretch()
    quality_v.addWidget(png_w)
    
    options_layout.addWidget(quality_box)
    
    # Resolution Limits
    resolution_box = QGroupBox("Resolution Limits")
    resolution_v = QVBoxLayout(resolution_box)
    
    max_dim_w = QWidget()
    max_dim_h = QHBoxLayout(max_dim_w)
    max_dim_h.addWidget(QLabel("Max dimension (px):"))
    max_dim_edit = QLineEdit(str(self.config.get('max_image_dimension', 2048)))
    max_dim_edit.setFixedWidth(80)
    max_dim_h.addWidget(max_dim_edit)
    max_dim_h.addWidget(QLabel("(Images larger than this will be resized)"))
    max_dim_h.addStretch()
    resolution_v.addWidget(max_dim_w)
    
    max_size_w = QWidget()
    max_size_h = QHBoxLayout(max_size_w)
    max_size_h.addWidget(QLabel("Max file size (MB):"))
    max_size_edit = QLineEdit(str(self.config.get('max_image_size_mb', 10)))
    max_size_edit.setFixedWidth(80)
    max_size_h.addWidget(max_size_edit)
    max_size_h.addWidget(QLabel("(Larger files will be compressed)"))
    max_size_h.addStretch()
    resolution_v.addWidget(max_size_w)
    
    options_layout.addWidget(resolution_box)
    
    # Advanced Options
    advanced_box = QGroupBox("Advanced Options")
    advanced_v = QVBoxLayout(advanced_box)
    
    preserve_transparency_cb = self._create_styled_checkbox("Preserve transparency (PNG/WebP only)")
    preserve_transparency_cb.setChecked(self.config.get('preserve_transparency', False))
    advanced_v.addWidget(preserve_transparency_cb)
    
    preserve_format_cb = self._create_styled_checkbox("Preserve original image format")
    preserve_format_cb.setChecked(self.config.get('preserve_original_format', False))
    advanced_v.addWidget(preserve_format_cb)
    
    optimize_ocr_cb = self._create_styled_checkbox("Optimize for OCR (maintain text clarity)")
    optimize_ocr_cb.setChecked(self.config.get('optimize_for_ocr', True))
    advanced_v.addWidget(optimize_ocr_cb)
    
    progressive_cb = self._create_styled_checkbox("Progressive encoding (JPEG)")
    progressive_cb.setChecked(self.config.get('progressive_encoding', True))
    advanced_v.addWidget(progressive_cb)
    
    save_compressed_cb = self._create_styled_checkbox("Save compressed images to disk")
    save_compressed_cb.setChecked(self.config.get('save_compressed_images', False))
    advanced_v.addWidget(save_compressed_cb)
    
    options_layout.addWidget(advanced_box)
    
    # Info
    info = QLabel("💡 Tips:\n• WebP offers the best compression with good quality\n• Use 'Auto' format for intelligent format selection\n• Higher quality = larger file size\n• OCR optimization maintains text readability")
    info.setStyleSheet("color: #666; font-size: 9pt;")
    options_layout.addWidget(info)
    
    main_layout.addWidget(compression_options)
    
    # Toggle function
    def toggle_options():
        enabled = enable_cb.isChecked()
        compression_options.setEnabled(enabled)
    
    enable_cb.toggled.connect(toggle_options)
    toggle_options()
    
    scroll.setWidget(scroll_widget)
    
    # Main dialog layout
    dialog_layout = QVBoxLayout(dialog)
    dialog_layout.addWidget(scroll)
    
    # Buttons
    button_layout = QHBoxLayout()
    
    def save_compression_settings():
        try:
            # Validate
            int(target_tokens_edit.text())
            int(max_dim_edit.text())
            float(max_size_edit.text())
        except ValueError:
            QMessageBox.critical(dialog, "Invalid Input", "Please enter valid numbers for numeric fields")
            return
        
        # Get selected format
        selected_format = 'auto'
        for rb in format_buttons:
            if rb.isChecked():
                selected_format = rb.property("format_value")
                break
        
        # Save all settings
        self.config['enable_image_compression'] = enable_cb.isChecked()
        self.config['auto_compress_enabled'] = auto_compress_cb.isChecked()
        self.config['target_image_tokens'] = int(target_tokens_edit.text())
        self.config['image_compression_format'] = selected_format
        self.config['webp_quality'] = webp_slider.value()
        self.config['jpeg_quality'] = jpeg_slider.value()
        self.config['png_compression'] = png_slider.value()
        self.config['max_image_dimension'] = int(max_dim_edit.text())
        self.config['max_image_size_mb'] = float(max_size_edit.text())
        self.config['preserve_transparency'] = preserve_transparency_cb.isChecked()
        self.config['preserve_original_format'] = preserve_format_cb.isChecked()
        self.config['optimize_for_ocr'] = optimize_ocr_cb.isChecked()
        self.config['progressive_encoding'] = progressive_cb.isChecked()
        self.config['save_compressed_images'] = save_compressed_cb.isChecked()
        
        self.append_log("✅ Image compression settings saved")
        dialog.close()
    
    save_btn = QPushButton("💾 Save Settings")
    save_btn.clicked.connect(save_compression_settings)
    save_btn.setMinimumHeight(35)
    save_btn.setStyleSheet(
        "QPushButton { "
        "  background-color: #28a745; "
        "  color: white; "
        "  padding: 8px 20px; "
        "  font-size: 11pt; "
        "  font-weight: bold; "
        "  border-radius: 4px; "
        "} "
        "QPushButton:hover { background-color: #218838; }"
    )
    button_layout.addWidget(save_btn)
    
    cancel_btn = QPushButton("❌ Cancel")
    cancel_btn.clicked.connect(dialog.close)
    cancel_btn.setMinimumHeight(35)
    cancel_btn.setStyleSheet(
        "QPushButton { "
        "  background-color: #6c757d; "
        "  color: white; "
        "  padding: 8px 20px; "
        "  font-size: 11pt; "
        "  font-weight: bold; "
        "  border-radius: 4px; "
        "} "
        "QPushButton:hover { background-color: #5a6268; }"
    )
    button_layout.addWidget(cancel_btn)
    
    dialog_layout.addLayout(button_layout)
    
    dialog.show()

def toggle_ai_hunter(self):
    """Toggle AI Hunter enabled state"""
    if 'ai_hunter_config' not in self.config:
        self.config['ai_hunter_config'] = {}
    
    self.config['ai_hunter_config']['enabled'] = self.ai_hunter_enabled_var
    self.save_config()
    # Note: ai_hunter_status_label is QLabel in PySide6, use setText instead of config
    if hasattr(self.ai_hunter_status_label, 'setText'):
        self.ai_hunter_status_label.setText(self._get_ai_hunter_status_text())

def _create_prompt_management_section(self, parent):
    """Create meta data section (formerly prompt management) - PySide6"""
    from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton, QWidget, QLineEdit
    from PySide6.QtCore import Qt
    
    section_box = QGroupBox("Meta Data")
    # No max width - let it expand in fullscreen
    section_v = QVBoxLayout(section_box)
    section_v.setContentsMargins(8, 8, 8, 8)  # Compact margins
    section_v.setSpacing(4)  # Compact spacing between widgets
    
    # Title frame with checkbox and buttons
    title_w = QWidget()
    title_h = QHBoxLayout(title_w)
    title_h.setContentsMargins(0, 10, 0, 10)
    
    translate_title_cb = self._create_styled_checkbox("Translate Book Title")
    try:
        translate_title_cb.setChecked(bool(self.translate_book_title_var))
    except Exception:
        pass
        
    # Toggle: include book title in glossary header/output
    glossary_title_cb = self._create_styled_checkbox("Include book title at top of glossary (during generation)")
    try:
        # Default to False if not present in config
        if not hasattr(self, 'include_book_title_glossary_var'):
            self.include_book_title_glossary_var = False
        glossary_title_cb.setChecked(bool(self.include_book_title_glossary_var))
    except Exception:
        glossary_title_cb.setChecked(False)
        
    glossary_title_cb.setToolTip(
        "Adds the book title row while generating the glossary (before deduplication).\n"
        "Uses translated metadata if available; skipped if metadata is missing."
    )

    # Toggle: auto-inject book title before dedup
    auto_inject_title_cb = self._create_styled_checkbox("Auto-inject book title (loaded glossaries only, bypasses dedup)")
    try:
        if not hasattr(self, 'auto_inject_book_title_var'):
            self.auto_inject_book_title_var = self.config.get('auto_inject_book_title', False)
        auto_inject_title_cb.setChecked(bool(self.auto_inject_book_title_var))
    except Exception:
        auto_inject_title_cb.setChecked(False)

    auto_inject_title_cb.setToolTip(
        "When loading an existing glossary file, inject the book title row after load\n"
        "(not part of dedup). Use only if your saved glossary lacks the title."
    )
    
    def _update_glossary_title_state(checked):
        """Update enabled state and styling of glossary toggles"""
        try:
            glossary_title_cb.setEnabled(checked)
            auto_inject_title_cb.setEnabled(checked)
            if checked:
                # Force white color for enabled state to ensure visibility
                glossary_title_cb.setStyleSheet("QCheckBox { color: white; }")
                auto_inject_title_cb.setStyleSheet("QCheckBox { color: white; }")
            else:
                # Disabled styling (grayed out) - do NOT change checked state
                glossary_title_cb.setStyleSheet("QCheckBox { color: #666666; }")
                auto_inject_title_cb.setStyleSheet("QCheckBox { color: #666666; }")
        except Exception:
            pass

    def _on_translate_title_toggle(checked):
        try:
            self.translate_book_title_var = bool(checked)
            _update_glossary_title_state(checked)
        except Exception:
            pass
            
    translate_title_cb.setToolTip(
        "Translate the book title and selected metadata fields\n"
        "using your current model/profile."
    )
    translate_title_cb.toggled.connect(_on_translate_title_toggle)
    title_h.addWidget(translate_title_cb)
    
    title_h.addSpacing(10)
    
    btn_configure_all = QPushButton("Configure All")
    btn_configure_all.setFixedWidth(120)
    btn_configure_all.clicked.connect(lambda: self.metadata_batch_ui.configure_translation_prompts())
    title_h.addWidget(btn_configure_all)
    
    title_h.addSpacing(5)
    
    btn_custom_metadata = QPushButton("Custom Metadata")
    btn_custom_metadata.setFixedWidth(150)
    btn_custom_metadata.clicked.connect(lambda: self.metadata_batch_ui.configure_metadata_fields())
    title_h.addWidget(btn_custom_metadata)
    
    title_h.addStretch()
    section_v.addWidget(title_w)
    
    title_desc = QLabel("When enabled: Book titles and selected metadata will be translated")
    title_desc.setStyleSheet("color: gray; font-size: 11pt;")
    title_desc.setContentsMargins(20, 0, 0, 10)
    section_v.addWidget(title_desc)
    
    def _on_glossary_title_toggle(checked):
        try:
            self.include_book_title_glossary_var = bool(checked)
            # Persist immediately in config so save_config captures it
            if hasattr(self, 'config'):
                self.config['include_book_title_glossary'] = bool(checked)
        except Exception:
            pass
    glossary_title_cb.toggled.connect(_on_glossary_title_toggle)
    section_v.addWidget(glossary_title_cb)

    def _on_auto_inject_toggle(checked):
        try:
            self.auto_inject_book_title_var = bool(checked)
            if hasattr(self, 'config'):
                self.config['auto_inject_book_title'] = bool(checked)
        except Exception:
            pass
    auto_inject_title_cb.toggled.connect(_on_auto_inject_toggle)
    section_v.addWidget(auto_inject_title_cb)
    
    # Initialize state based on current value
    _update_glossary_title_state(translate_title_cb.isChecked())
    
    # Separator
    sep1 = QFrame()
    sep1.setFrameShape(QFrame.HLine)
    sep1.setFrameShadow(QFrame.Sunken)
    section_v.addWidget(sep1)
    
    # Batch Header Translation Section
    header_title = QLabel("Chapter Header Translation:")
    header_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
    header_title.setContentsMargins(0, 5, 0, 5)
    section_v.addWidget(header_title)
    
    header_row1 = QWidget()
    header_h1 = QHBoxLayout(header_row1)
    header_h1.setContentsMargins(0, 5, 0, 10)
    
    # Master toggle for batch header translation
    batch_toggle_cb = self._create_styled_checkbox("Batch Translate Headers")
    try:
        batch_toggle_cb.setChecked(bool(self.batch_translate_headers_var))
    except Exception:
        pass
    batch_toggle_cb.setToolTip(
        "Translate chapter headers in batches instead of per file.\n"
        "Uses the settings below for batching and output."
    )
    
    header_h1.addWidget(batch_toggle_cb)
    header_h1.addSpacing(20)
    headers_per_batch_label = QLabel("Headers per batch:")
    header_h1.addWidget(headers_per_batch_label)
    
    batch_entry = QLineEdit()
    batch_entry.setFixedWidth(100)
    try:
        batch_entry.setText(str(self.headers_per_batch_var))
    except Exception:
        pass
    def _on_headers_per_batch_changed(text):
        try:
            self.headers_per_batch_var = text
        except Exception:
            pass
    batch_entry.textChanged.connect(_on_headers_per_batch_changed)
    header_h1.addWidget(batch_entry)
    
    # Add help button next to batch entry
    header_h1.addSpacing(5)
    help_btn = QPushButton("ℹ️")
    help_btn.setFixedSize(28, 28)
    help_btn.clicked.connect(lambda: self.show_header_help_dialog())
    help_btn.setStyleSheet(
        "QPushButton { "
        "background-color: transparent; "
        "border: none; "
        "font-size: 18px; "
        "padding: 0px; "
        "} "
        "QPushButton:hover { "
        "background-color: rgba(23, 162, 184, 0.2); "
        "border-radius: 14px; "
        "} "
        "QPushButton:pressed { "
        "background-color: rgba(23, 162, 184, 0.4); "
        "border-radius: 14px; "
        "}"
    )
    help_btn.setToolTip("Show detailed help for header translation options")
    header_h1.addWidget(help_btn)
    
    header_h1.addStretch()
    
    section_v.addWidget(header_row1)
    
    # Options for header translation
    update_row = QWidget()
    update_h = QHBoxLayout(update_row)
    update_h.setContentsMargins(20, 0, 0, 0)
    
    update_cb = self._create_styled_checkbox("Update headers in HTML files")
    try:
        update_cb.setChecked(bool(self.update_html_headers_var))
    except Exception:
        pass
    def _on_update_html_toggle(checked):
        try:
            self.update_html_headers_var = bool(checked)
        except Exception:
            pass
    update_cb.toggled.connect(_on_update_html_toggle)
    update_cb.setToolTip(
        "Write translated headers back into the HTML chapters.\n"
        "Disable if you only want preview/exports."
    )
    update_h.addWidget(update_cb)
    
    update_h.addSpacing(20)
    
    save_cb = self._create_styled_checkbox("Save translations to .txt")
    try:
        save_cb.setChecked(bool(self.save_header_translations_var))
    except Exception:
        pass
    def _on_save_translations_toggle(checked):
        try:
            self.save_header_translations_var = bool(checked)
        except Exception:
            pass
    save_cb.toggled.connect(_on_save_translations_toggle)
    save_cb.setToolTip(
        "Export header translation mappings to translated_headers.txt\n"
        "for auditing or reuse."
    )
    update_h.addWidget(save_cb)
    update_h.addStretch()
    
    section_v.addWidget(update_row)
    
    # Additional ignore header options
    ignore_row = QWidget()
    ignore_h = QHBoxLayout(ignore_row)
    ignore_h.setContentsMargins(20, 5, 0, 0)
    
    ignore_header_cb = self._create_styled_checkbox("Ignore header")
    try:
        ignore_header_cb.setChecked(bool(self.ignore_header_var))
    except Exception:
        pass
    def _on_ignore_header_toggle(checked):
        try:
            self.ignore_header_var = bool(checked)
        except Exception:
            pass
    ignore_header_cb.toggled.connect(_on_ignore_header_toggle)
    ignore_header_cb.setToolTip(
        "Skip translating visible header tags (h1/h2/h3) inside chapters.\n"
        "Useful if headers are already translated."
    )
    ignore_h.addWidget(ignore_header_cb)
    
    ignore_h.addSpacing(15)
    
    ignore_title_cb = self._create_styled_checkbox("Ignore title")
    try:
        ignore_title_cb.setChecked(bool(self.ignore_title_var))
    except Exception:
        pass
    def _on_ignore_title_toggle(checked):
        try:
            self.ignore_title_var = bool(checked)
        except Exception:
            pass
    ignore_title_cb.toggled.connect(_on_ignore_title_toggle)
    ignore_title_cb.setToolTip(
        "Skip translating the title tag in the HTML head.\n"
        "Keeps original document titles."
    )
    ignore_h.addWidget(ignore_title_cb)
    
    ignore_h.addStretch()
    section_v.addWidget(ignore_row)
    
    # Second ignore row for additional options
    ignore_row2 = QWidget()
    ignore_h2 = QHBoxLayout(ignore_row2)
    ignore_h2.setContentsMargins(20, 5, 0, 0)
    
    # Remove duplicate H1+P pairs
    if not hasattr(self, 'remove_duplicate_h1_p_var'):
        self.remove_duplicate_h1_p_var = self.config.get('remove_duplicate_h1_p', False)
    
    remove_dup_cb = self._create_styled_checkbox("Remove duplicate H1+P pairs")
    try:
        remove_dup_cb.setChecked(bool(self.remove_duplicate_h1_p_var))
    except Exception:
        pass
    def _on_remove_dup_toggle(checked):
        try:
            self.remove_duplicate_h1_p_var = bool(checked)
            self.config['remove_duplicate_h1_p'] = bool(checked)
        except Exception:
            pass
    remove_dup_cb.toggled.connect(_on_remove_dup_toggle)
    remove_dup_cb.setToolTip(
        "Remove paragraph tags that immediately follow H1 tags with identical text.\n"
        "Useful for novels that repeat chapter titles."
    )
    ignore_h2.addWidget(remove_dup_cb)
    
    ignore_h2.addSpacing(15)
    
    # Add fallback option with warning icon
    use_fallback_cb = self._create_styled_checkbox("⚠️ Use Sorted Fallback")
    try:
        use_fallback_cb.setChecked(bool(getattr(self, 'use_sorted_fallback_var', False)))
    except Exception:
        pass
    def _on_use_fallback_toggle(checked):
        try:
            self.use_sorted_fallback_var = bool(checked)
        except Exception:
            pass
    use_fallback_cb.toggled.connect(_on_use_fallback_toggle)
    use_fallback_cb.setToolTip(
        "If standalone OPF-based matching fails, fall back to sorted index matching.\n"
        "⚠️ Less accurate - may mismatch chapters if file order differs from OPF spine."
    )
    ignore_h2.addWidget(use_fallback_cb)
    ignore_h2.addStretch()
    
    section_v.addWidget(ignore_row2)
    
    # Buttons row (below ignore options)
    buttons_row = QWidget()
    buttons_h = QHBoxLayout(buttons_row)
    buttons_h.setContentsMargins(20, 5, 0, 0)
    
    translate_now_btn = QPushButton("Translate Headers Now")
    translate_now_btn.setFixedWidth(210)
    
    # Store reference for button transformation
    self.translate_headers_btn = translate_now_btn
    
    # Create a rotatable label for the icon
    from PySide6.QtCore import Property, QPropertyAnimation, QEasingCurve
    from PySide6.QtGui import QTransform
    
    class RotatableLabel(QLabel):
        def __init__(self, parent=None):
            super().__init__(parent)
            self._rotation = 0
            self._original_pixmap = None
        
        def set_rotation(self, angle):
            self._rotation = angle
            if self._original_pixmap:
                transform = QTransform()
                transform.rotate(angle)
                rotated = self._original_pixmap.transformed(transform, Qt.SmoothTransformation)
                self.setPixmap(rotated)
        
        def get_rotation(self):
            return self._rotation
        
        rotation = Property(float, get_rotation, set_rotation)
        
        def set_original_pixmap(self, pixmap):
            self._original_pixmap = pixmap
            self.setPixmap(pixmap)
    
    # Create icon with rotation support (HiDPI-aware, smaller than 36x36)
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Halgakos.ico")
    translate_icon_label = RotatableLabel()
    if os.path.exists(icon_path):
        try:
            from PySide6.QtGui import QIcon, QPixmap
            from PySide6.QtCore import QSize
            try:
                dpr = self.devicePixelRatioF()
            except Exception:
                dpr = 1.0
            logical_px = 12  # smaller than toolbar icons
            dev_px = int(logical_px * max(1.0, dpr))
            icon = QIcon(icon_path)
            pm = icon.pixmap(QSize(dev_px, dev_px))
            if pm.isNull():
                raw = QPixmap(icon_path)
                img = raw.toImage().scaled(dev_px, dev_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                pm = QPixmap.fromImage(img)
            try:
                pm.setDevicePixelRatio(dpr)
            except Exception:
                pass
            translate_icon_label.set_original_pixmap(pm)
        except Exception:
            pixmap = QPixmap(icon_path)
            scaled_pixmap = pixmap.scaled(16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            translate_icon_label.set_original_pixmap(scaled_pixmap)
    
    self.translate_headers_icon = translate_icon_label
    
    # Create rotation animations
    self.translate_icon_spin_animation = QPropertyAnimation(translate_icon_label, b"rotation")
    self.translate_icon_spin_animation.setDuration(900)  # 0.9 seconds per rotation
    self.translate_icon_spin_animation.setStartValue(0)
    self.translate_icon_spin_animation.setEndValue(360)
    self.translate_icon_spin_animation.setLoopCount(-1)  # Infinite loop
    self.translate_icon_spin_animation.setEasingCurve(QEasingCurve.Linear)
    
    # Create smooth stop animation
    self.translate_icon_stop_animation = QPropertyAnimation(translate_icon_label, b"rotation")
    self.translate_icon_stop_animation.setDuration(800)  # Deceleration time
    self.translate_icon_stop_animation.setEasingCurve(QEasingCurve.OutCubic)
    
    def _on_translate_toggle():
        from PySide6.QtWidgets import QMessageBox
        from PySide6.QtGui import QIcon
        
        # Check if currently running
        is_running = getattr(self, '_headers_translation_running', False)
        
        if is_running:
            # Stop the translation
            try:
                # Set local stop flag
                self._headers_stop_requested = True
                
                # Set stop flags on BatchHeaderTranslator if it exists
                if hasattr(self, '_batch_header_translator'):
                    self._batch_header_translator.set_stop_flag(True)
                    self.append_log("✅ Stop signal sent to batch header translator")
                
                # Set stop flags on unified_api_client (same as main translator GUI)
                try:
                    import unified_api_client
                    if hasattr(unified_api_client, 'set_stop_flag'):
                        unified_api_client.set_stop_flag(True)
                    # If there's a global client instance, stop it too
                    if hasattr(unified_api_client, 'global_stop_flag'):
                        unified_api_client.global_stop_flag = True
                    # Set the _cancelled flag on the UnifiedClient class itself
                    if hasattr(unified_api_client, 'UnifiedClient'):
                        unified_api_client.UnifiedClient._global_cancelled = True
                    self.append_log("✅ Stop signal sent to API client")
                except Exception as e:
                    self.append_log(f"⚠️ Could not set API client stop flags: {e}")
                
                # Also try to stop the API client instance if it exists
                if hasattr(self, 'api_client') and self.api_client:
                    try:
                        if hasattr(self.api_client, 'set_stop_flag'):
                            self.api_client.set_stop_flag(True)
                        if hasattr(self.api_client, '_cancelled'):
                            self.api_client._cancelled = True
                    except Exception:
                        pass
                
                # Update button to "Stopping..." state (gray)
                translate_now_btn.setText("⏹ Stopping...")
                translate_now_btn.setStyleSheet(
                    "QPushButton { background-color: #9e9e9e; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold; } "
                    "QPushButton:hover { background-color: #9e9e9e; } "
                    "QPushButton:disabled { background-color: #e0e0e0; color: #9e9e9e; }"
                )
                
                self.append_log("🛑 Stop requested — waiting for current operation to finish")
                
            except Exception as e:
                self.append_log(f"❌ Error stopping: {e}")
        else:
            # Start the translation
            # Get icon
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Halgakos.ico")
            icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
            
            # Show confirmation dialog
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setWindowTitle("Translate Headers")
            msg_box.setText("Start standalone header translation?")
            msg_box.setInformativeText(
                "This will translate chapter headers using content.opf-based exact matching."
            )
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.Yes)
            msg_box.setWindowIcon(icon)
            
            # Center the buttons
            from PySide6.QtWidgets import QDialogButtonBox
            button_box = msg_box.findChild(QDialogButtonBox)
            if button_box:
                button_box.setCenterButtons(True)
            
            result = msg_box.exec()
            
            if result == QMessageBox.Yes:
                # Don't close the dialog - keep it open so user can see spinning button and stop if needed
                # The button will transform to show stop state
                
                # Run translation in background thread
                self.run_standalone_translate_headers()
    
    translate_now_btn.clicked.connect(_on_translate_toggle)
    translate_now_btn.setStyleSheet(
        "QPushButton { background-color: #6c757d; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold; } "
        "QPushButton:hover { background-color: #28a745; } "
        "QPushButton:disabled { background-color: #e0e0e0; color: #9e9e9e; }"
    )
    # Set icon from the rotatable label
    if translate_icon_label._original_pixmap:
        translate_now_btn.setIcon(QIcon(translate_icon_label._original_pixmap))
    buttons_h.addWidget(translate_now_btn)
    
    buttons_h.addSpacing(10)
    
    delete_btn = QPushButton("🗑️Delete Header Files")
    delete_btn.setFixedWidth(210)
    delete_btn.clicked.connect(lambda: self.delete_translated_headers_file())
    delete_btn.setStyleSheet(
        "QPushButton { background-color: #6c757d; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold; } "
        "QPushButton:hover { background-color: #dc3545; } "
        "QPushButton:disabled { background-color: #e0e0e0; color: #9e9e9e; }"
    )
    # Set icon
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Halgakos.ico")
    if os.path.exists(icon_path):
        delete_btn.setIcon(QIcon(icon_path))
    buttons_h.addWidget(delete_btn)
    buttons_h.addStretch()
    
    section_v.addWidget(buttons_row)
    
    # Description for the buttons
    button_desc = QLabel(
        "Standalone mode: Translates chapter headers using exact content.opf mapping."
    )
    button_desc.setStyleSheet("color: gray; font-size: 10pt;")
    button_desc.setContentsMargins(20, 2, 0, 10)
    section_v.addWidget(button_desc)
    
    # Toggle function for enabling/disabling controls
    def _toggle_header_controls(checked):
        try:
            enabled = bool(self.batch_translate_headers_var)
            self.batch_translate_headers_var = bool(checked)
        except Exception:
            pass
        headers_per_batch_label.setEnabled(checked)
        batch_entry.setEnabled(checked)
        update_cb.setEnabled(checked)
        save_cb.setEnabled(checked)
        ignore_header_cb.setEnabled(checked)
        ignore_title_cb.setEnabled(checked)
        use_fallback_cb.setEnabled(checked)
        delete_btn.setEnabled(checked)
        translate_now_btn.setEnabled(checked)
    
    batch_toggle_cb.toggled.connect(_toggle_header_controls)
    
    # Initialize disabled state
    try:
        _toggle_header_controls(bool(self.batch_translate_headers_var))
    except Exception:
        _toggle_header_controls(False)
    
    
    # Separator
    sep2 = QFrame()
    sep2.setFrameShape(QFrame.HLine)
    sep2.setFrameShadow(QFrame.Sunken)
    section_v.addWidget(sep2)
    
    # EPUB Utilities
    epub_title = QLabel("EPUB Utilities:")
    epub_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
    epub_title.setContentsMargins(0, 5, 0, 5)
    section_v.addWidget(epub_title)
    
    btn_validate = QPushButton("🔍 Validate EPUB Structure")
    btn_validate.setFixedWidth(250)
    btn_validate.clicked.connect(lambda: self.validate_epub_structure_gui())
    btn_validate.setStyleSheet(
        "QPushButton { background-color: #6f42c1; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold; } "
        "QPushButton:hover { background-color: #5a32a3; }"
    )
    section_v.addWidget(btn_validate)
    
    validate_desc = QLabel("Check if all required EPUB files are present for compilation")
    validate_desc.setStyleSheet("color: gray; font-size: 10pt;")
    validate_desc.setContentsMargins(0, 0, 0, 5)
    section_v.addWidget(validate_desc)
    
    # NCX-only navigation toggle
    ncx_cb = self._create_styled_checkbox("Use NCX-only Navigation (Compatibility Mode)")
    try:
        ncx_cb.setChecked(bool(self.force_ncx_only_var))
    except Exception:
        pass
    def _on_ncx_toggle(checked):
        try:
            self.force_ncx_only_var = bool(checked)
        except Exception:
            pass
    ncx_cb.toggled.connect(_on_ncx_toggle)
    ncx_cb.setContentsMargins(0, 5, 0, 5)
    section_v.addWidget(ncx_cb)
    
    # CSS Attachment toggle + Load CSS button
    css_cb = self._create_styled_checkbox("Attach CSS to Chapters (May fix or cause styling issues)")
    try:
        css_cb.setChecked(bool(self.attach_css_to_chapters_var))
    except Exception:
        pass

    # Ensure we have a variable to store the override CSS path
    if not hasattr(self, 'epub_css_override_path_var'):
        self.epub_css_override_path_var = self.config.get('epub_css_override_path', '')

    def _on_css_toggle(checked):
        try:
            self.attach_css_to_chapters_var = bool(checked)
        except Exception:
            pass

    css_cb.toggled.connect(_on_css_toggle)

    from PySide6.QtWidgets import QFileDialog, QHBoxLayout

    css_row = QWidget()
    css_row_h = QHBoxLayout(css_row)
    css_row_h.setContentsMargins(0, 5, 0, 5)
    css_row_h.setSpacing(8)

    css_row_h.addWidget(css_cb)

    load_css_btn = QPushButton("Load CSS…")
    load_css_btn.setToolTip("Select a CSS file to use for all chapters (overrides original EPUB CSS)")
    load_css_btn.setMinimumWidth(90)
    load_css_btn.setStyleSheet(
        "QPushButton { background-color: #17a2b8; color: white; padding: 4px 10px; "
        "border-radius: 4px; font-weight: bold; } "
        "QPushButton:hover { background-color: #138496; }"
    )
    css_row_h.addWidget(load_css_btn)

    clear_css_btn = QPushButton("Clear")
    clear_css_btn.setToolTip("Remove CSS override and use original EPUB CSS again")
    clear_css_btn.setMinimumWidth(60)
    clear_css_btn.setStyleSheet(
        "QPushButton { background-color: #6c757d; color: white; padding: 4px 8px; "
        "border-radius: 4px; font-weight: bold; } "
        "QPushButton:hover { background-color: #5a6268; }"
    )
    css_row_h.addWidget(clear_css_btn)

    import os as _os

    css_status_label = QLabel()
    css_status_label.setStyleSheet("color: #28a745; font-size: 11pt; font-weight: bold;")
    css_status_label.hide()
    css_row_h.addWidget(css_status_label)

    css_path_label = QLabel()
    css_path_label.setStyleSheet("color: gray; font-size: 9pt;")
    if getattr(self, 'epub_css_override_path_var', ''):
        css_path_label.setText(_os.path.basename(self.epub_css_override_path_var))
        css_status_label.setText("✓")
        css_status_label.show()
    css_row_h.addWidget(css_path_label)

    css_row_h.addStretch()
    section_v.addWidget(css_row)

    def _on_load_css_clicked():
        try:
            start_dir = _os.path.dirname(self.epub_css_override_path_var) if getattr(self, 'epub_css_override_path_var', '') else _os.getcwd()
            file_name, _ = QFileDialog.getOpenFileName(parent, "Select CSS file", start_dir, "CSS Files (*.css);;All Files (*.*)")
            if file_name:
                self.epub_css_override_path_var = file_name
                css_path_label.setText(_os.path.basename(file_name))
                css_status_label.setText("✓")
                css_status_label.show()
                # If user explicitly loads CSS, ensure attachment is enabled
                if not css_cb.isChecked():
                    css_cb.setChecked(True)
        except Exception:
            pass

    def _on_clear_css_clicked():
        try:
            self.epub_css_override_path_var = ''
            css_path_label.setText('')
            css_status_label.hide()
        except Exception:
            pass

    load_css_btn.clicked.connect(_on_load_css_clicked)
    clear_css_btn.clicked.connect(_on_clear_css_clicked)
    
    # HTML serialization method toggle
    html_method_cb = self._create_styled_checkbox("Use HTML Method for EPUB (Better for preserving whitespaces)")
    try:
        html_method_cb.setChecked(bool(self.epub_use_html_method_var))
    except Exception:
        pass
    def _on_html_method_toggle(checked):
        try:
            self.epub_use_html_method_var = bool(checked)
        except Exception:
            pass
    html_method_cb.toggled.connect(_on_html_method_toggle)
    html_method_cb.setContentsMargins(0, 5, 0, 5)
    section_v.addWidget(html_method_cb)
    
    # Output file naming
    retain_cb = self._create_styled_checkbox("Retain source extension (no 'response_' prefix)")
    try:
        retain_cb.setChecked(bool(self.retain_source_extension_var))
    except Exception:
        pass
    def _on_retain_toggle(checked):
        try:
            self.retain_source_extension_var = bool(checked)
        except Exception:
            pass
    retain_cb.toggled.connect(_on_retain_toggle)
    retain_cb.setContentsMargins(0, 5, 0, 5)
    section_v.addWidget(retain_cb)
    
    # Place the section at row 0, column 0 to match the original grid
    try:
        grid = parent.layout()
        if grid:
            grid.addWidget(section_box, 0, 0)
    except Exception:
        # Fallback: just stack
        section_box.setParent(parent)

def _create_processing_options_section(self, parent):
    """Create processing options section - PySide6"""
    from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton, QWidget, QLineEdit, QComboBox, QRadioButton, QButtonGroup
    from PySide6.QtCore import Qt
    
    section_box = QGroupBox("Processing Options")
    # No max width - let it expand in fullscreen
    section_v = QVBoxLayout(section_box)
    section_v.setContentsMargins(8, 8, 8, 8)  # Compact margins
    section_v.setSpacing(4)  # Compact spacing between widgets
    
    # Create two-column layout
    columns_container = QWidget()
    columns_h = QHBoxLayout(columns_container)
    columns_h.setContentsMargins(0, 0, 0, 15)
    columns_h.setSpacing(20)
    
    # Left column - Checkboxes
    left_column = QWidget()
    left_v = QVBoxLayout(left_column)
    left_v.setContentsMargins(0, 0, 0, 0)
    
    # Emergency Paragraph Restoration
    emergency_cb = self._create_styled_checkbox("Emergency Paragraph Restoration")
    try:
        emergency_cb.setChecked(bool(self.emergency_restore_var))
    except Exception:
        pass
    def _on_emergency_toggle(checked):
        try:
            self.emergency_restore_var = bool(checked)
        except Exception:
            pass
    emergency_cb.toggled.connect(_on_emergency_toggle)
    emergency_cb.setContentsMargins(0, 2, 0, 0)
    left_v.addWidget(emergency_cb)
    
    emergency_desc = QLabel("Fixes AI responses that lose paragraph\nstructure (wall of text)")
    emergency_desc.setStyleSheet("color: gray; font-size: 10pt;")
    emergency_desc.setContentsMargins(20, 0, 0, 5)
    left_v.addWidget(emergency_desc)
    
    # Emergency Image Restoration (Add below Paragraph Restoration)
    img_restore_cb = self._create_styled_checkbox("Emergency Image Restoration")
    try:
        # Default to False if not present (disabled by default)
        if not hasattr(self, 'emergency_image_restore_var'):
            self.emergency_image_restore_var = False
        img_restore_cb.setChecked(bool(self.emergency_image_restore_var))
    except Exception:
        pass
    def _on_img_restore_toggle(checked):
        try:
            self.emergency_image_restore_var = bool(checked)
        except Exception:
            pass
    img_restore_cb.toggled.connect(_on_img_restore_toggle)
    img_restore_cb.setContentsMargins(0, 2, 0, 0)
    left_v.addWidget(img_restore_cb)
    
    img_restore_desc = QLabel("Restores &lt;img&gt; tags if missing in translation<br>(Matches source images to output)")
    img_restore_desc.setStyleSheet("color: gray; font-size: 10pt;")
    img_restore_desc.setContentsMargins(20, 0, 0, 5)
    img_restore_desc.setTextFormat(Qt.RichText)
    left_v.addWidget(img_restore_desc)
    
    # Enable Decimal Chapter Detection
    decimal_cb = self._create_styled_checkbox("Enable Decimal Chapter Detection (EPUBs)")
    try:
        decimal_cb.setChecked(bool(self.enable_decimal_chapters_var))
    except Exception:
        pass
    def _on_decimal_toggle(checked):
        try:
            self.enable_decimal_chapters_var = bool(checked)
        except Exception:
            pass
    decimal_cb.toggled.connect(_on_decimal_toggle)
    decimal_cb.setContentsMargins(0, 2, 0, 0)
    left_v.addWidget(decimal_cb)
    
    decimal_desc = QLabel("Detect chapters like 1.1, 1.2 in EPUB files\n(Text files always use decimal chapters when split)")
    decimal_desc.setStyleSheet("color: gray; font-size: 10pt;")
    decimal_desc.setContentsMargins(20, 0, 0, 5)
    left_v.addWidget(decimal_desc)
    
    left_v.addStretch()
    columns_h.addWidget(left_column)
    
    # Right column - Button and Reinforce field
    right_column = QWidget()
    right_v = QVBoxLayout(right_column)
    right_v.setContentsMargins(0, 0, 0, 0)
    
    # Translation Chunk Prompt button (renamed)
    btn_chunk_prompt = QPushButton("⚙️ Configure Chunk Prompt")
    btn_chunk_prompt.setFixedWidth(180)
    btn_chunk_prompt.clicked.connect(lambda: self.configure_translation_chunk_prompt())
    right_v.addWidget(btn_chunk_prompt)
    
    chunk_prompt_desc = QLabel("Split chapter context")
    chunk_prompt_desc.setStyleSheet("color: gray; font-size: 10pt;")
    chunk_prompt_desc.setContentsMargins(0, 0, 0, 15)
    right_v.addWidget(chunk_prompt_desc)
    
    # Message Reinforcement option
    reinforce_w = QWidget()
    reinforce_h = QHBoxLayout(reinforce_w)
    reinforce_h.setContentsMargins(0, 0, 0, 0)
    reinforce_h.addWidget(QLabel("Prompt Reinforcement:"))
    reinforce_edit = QLineEdit()
    reinforce_edit.setFixedWidth(60)
    try:
        reinforce_edit.setText(str(self.reinforcement_freq_var))
    except Exception:
        pass
    def _on_reinforce_changed(text):
        try:
            self.reinforcement_freq_var = text
        except Exception:
            pass
    reinforce_edit.textChanged.connect(_on_reinforce_changed)
    reinforce_h.addWidget(reinforce_edit)
    reinforce_h.addStretch()
    right_v.addWidget(reinforce_w)
    
    # Break Split Count option
    break_split_w = QWidget()
    break_split_h = QHBoxLayout(break_split_w)
    break_split_h.setContentsMargins(0, 5, 0, 0)
    break_split_h.addWidget(QLabel("Break Split Count:"))
    break_split_edit = QLineEdit()
    break_split_edit.setFixedWidth(60)
    try:
        break_split_edit.setText(str(self.break_split_count_var) if hasattr(self, 'break_split_count_var') and self.break_split_count_var else '')
    except Exception:
        pass
    def _on_break_split_changed(text):
        try:
            self.break_split_count_var = text
        except Exception:
            pass
    break_split_edit.textChanged.connect(_on_break_split_changed)
    break_split_h.addWidget(break_split_edit)
    break_split_h.addStretch()
    right_v.addWidget(break_split_w)
    
    break_split_desc = QLabel("Split chunks after N elements\n(Leave empty for token-only splitting)")
    break_split_desc.setStyleSheet("color: gray; font-size: 9pt;")
    break_split_desc.setContentsMargins(0, 0, 0, 5)
    right_v.addWidget(break_split_desc)
    
    right_v.addStretch()
    columns_h.addWidget(right_column)
    
    section_v.addWidget(columns_container)
    
    # === CHAPTER EXTRACTION SETTINGS ===
    extraction_box = QGroupBox("Chapter Extraction Settings")
    extraction_v = QVBoxLayout(extraction_box)
    
    # Initialize variables if not exists
    if not hasattr(self, 'text_extraction_method_var'):
        if self.config.get('extraction_mode') == 'enhanced':
            self.text_extraction_method_var = 'enhanced'
            self.file_filtering_level_var = self.config.get('enhanced_filtering', 'smart')
        else:
            self.text_extraction_method_var = 'standard'
            self.file_filtering_level_var = self.config.get('extraction_mode', 'smart')
    
    if not hasattr(self, 'enhanced_preserve_structure_var'):
        self.enhanced_preserve_structure_var = self.config.get('enhanced_preserve_structure', True)
    
    # Text Extraction Method
    method_title = QLabel("Text Extraction Method:")
    method_title.setStyleSheet("font-weight: bold; font-size: 10pt;")
    method_title.setContentsMargins(0, 0, 0, 5)
    extraction_v.addWidget(method_title)
    
    extraction_method_group = QButtonGroup(extraction_box)
    
    # Standard extraction
    standard_rb = QRadioButton("Standard (BeautifulSoup)")
    try:
        if self.text_extraction_method_var == "standard":
            standard_rb.setChecked(True)
    except Exception:
        pass
    def _on_standard_selected(checked):
        if checked:
            try:
                self.text_extraction_method_var = "standard"
                self.on_extraction_method_change()
            except Exception:
                pass
    standard_rb.toggled.connect(_on_standard_selected)
    extraction_method_group.addButton(standard_rb)
    standard_rb.setContentsMargins(0, 2, 0, 0)
    extraction_v.addWidget(standard_rb)
    
    standard_desc = QLabel("Traditional HTML parsing - fast and reliable")
    standard_desc.setStyleSheet("color: gray; font-size: 9pt;")
    standard_desc.setContentsMargins(20, 0, 0, 5)
    extraction_v.addWidget(standard_desc)
    
    # Enhanced extraction
    enhanced_rb = QRadioButton("🚀 Enhanced (html2text)")
    try:
        if self.text_extraction_method_var == "enhanced":
            enhanced_rb.setChecked(True)
    except Exception:
        pass
    def _on_enhanced_selected(checked):
        if checked:
            try:
                self.text_extraction_method_var = "enhanced"
                self.on_extraction_method_change()
            except Exception:
                pass
    enhanced_rb.toggled.connect(_on_enhanced_selected)
    extraction_method_group.addButton(enhanced_rb)
    enhanced_rb.setContentsMargins(0, 2, 0, 0)
    extraction_v.addWidget(enhanced_rb)
    
    enhanced_desc = QLabel("Superior Unicode handling, cleaner text extraction")
    enhanced_desc.setStyleSheet("color: darkgreen; font-size: 9pt;")
    enhanced_desc.setContentsMargins(20, 0, 0, 5)
    extraction_v.addWidget(enhanced_desc)
    
    # Enhanced options (shown when enhanced is selected)
    self.enhanced_options_frame = QWidget()
    enhanced_opts_v = QVBoxLayout(self.enhanced_options_frame)
    enhanced_opts_v.setContentsMargins(20, 5, 0, 0)
    
    preserve_cb = self._create_styled_checkbox("Preserve Markdown Structure")
    try:
        preserve_cb.setChecked(bool(self.enhanced_preserve_structure_var))
    except Exception:
        pass
    def _on_preserve_toggle(checked):
        try:
            self.enhanced_preserve_structure_var = bool(checked)
        except Exception:
            pass
    preserve_cb.toggled.connect(_on_preserve_toggle)
    preserve_cb.setContentsMargins(0, 2, 0, 0)
    enhanced_opts_v.addWidget(preserve_cb)
    
    preserve_desc = QLabel("Keep formatting (bold, headers, lists) for better AI context")
    preserve_desc.setStyleSheet("color: gray; font-size: 8pt;")
    preserve_desc.setContentsMargins(20, 0, 0, 3)
    enhanced_opts_v.addWidget(preserve_desc)
    
    extraction_v.addWidget(self.enhanced_options_frame)
    
    # Separator
    sep_extract1 = QFrame()
    sep_extract1.setFrameShape(QFrame.HLine)
    sep_extract1.setFrameShadow(QFrame.Sunken)
    extraction_v.addWidget(sep_extract1)
    
    # File Filtering Level
    filter_title = QLabel("File Filtering Level:")
    filter_title.setStyleSheet("font-weight: bold; font-size: 10pt;")
    filter_title.setContentsMargins(0, 0, 0, 5)
    extraction_v.addWidget(filter_title)
    
    filtering_group = QButtonGroup(extraction_box)
    
    # Smart filtering
    smart_rb = QRadioButton("Smart (Aggressive Filtering)")
    try:
        if self.file_filtering_level_var == "smart":
            smart_rb.setChecked(True)
    except Exception:
        pass
    def _on_smart_selected(checked):
        if checked:
            try:
                self.file_filtering_level_var = "smart"
            except Exception:
                pass
    smart_rb.toggled.connect(_on_smart_selected)
    filtering_group.addButton(smart_rb)
    smart_rb.setContentsMargins(0, 2, 0, 0)
    extraction_v.addWidget(smart_rb)
    
    smart_desc = QLabel("Skips navigation, TOC, copyright files\nBest for clean EPUBs with clear chapter structure")
    smart_desc.setStyleSheet("color: gray; font-size: 9pt;")
    smart_desc.setContentsMargins(20, 0, 0, 5)
    extraction_v.addWidget(smart_desc)
    
    # Comprehensive filtering
    comprehensive_rb = QRadioButton("Comprehensive (Moderate Filtering)")
    try:
        if self.file_filtering_level_var == "comprehensive":
            comprehensive_rb.setChecked(True)
    except Exception:
        pass
    def _on_comprehensive_selected(checked):
        if checked:
            try:
                self.file_filtering_level_var = "comprehensive"
            except Exception:
                pass
    comprehensive_rb.toggled.connect(_on_comprehensive_selected)
    filtering_group.addButton(comprehensive_rb)
    comprehensive_rb.setContentsMargins(0, 2, 0, 0)
    extraction_v.addWidget(comprehensive_rb)
    
    comprehensive_desc = QLabel("Only skips obvious navigation files\nGood when Smart mode misses chapters")
    comprehensive_desc.setStyleSheet("color: gray; font-size: 9pt;")
    comprehensive_desc.setContentsMargins(20, 0, 0, 5)
    extraction_v.addWidget(comprehensive_desc)
    
    # Full extraction
    full_rb = QRadioButton("Full (No Filtering)")
    try:
        if self.file_filtering_level_var == "full":
            full_rb.setChecked(True)
    except Exception:
        pass
    def _on_full_selected(checked):
        if checked:
            try:
                self.file_filtering_level_var = "full"
            except Exception:
                pass
    full_rb.toggled.connect(_on_full_selected)
    filtering_group.addButton(full_rb)
    full_rb.setContentsMargins(0, 2, 0, 0)
    extraction_v.addWidget(full_rb)
    
    full_desc = QLabel("Extracts ALL HTML/XHTML files\nUse when other modes skip important content")
    full_desc.setStyleSheet("color: gray; font-size: 9pt;")
    full_desc.setContentsMargins(20, 0, 0, 5)
    extraction_v.addWidget(full_desc)
    
    # Force BeautifulSoup for Traditional APIs
    if not hasattr(self, 'force_bs_for_traditional_var'):
        self.force_bs_for_traditional_var = self.config.get('force_bs_for_traditional', True)
    force_bs_cb = self._create_styled_checkbox("Force BeautifulSoup for DeepL / Google Translate / Google Free")
    try:
        force_bs_cb.setChecked(bool(self.force_bs_for_traditional_var))
    except Exception:
        pass
    def _on_force_bs_toggle(checked):
        try:
            self.force_bs_for_traditional_var = bool(checked)
        except Exception:
            pass
    force_bs_cb.toggled.connect(_on_force_bs_toggle)
    force_bs_cb.setContentsMargins(0, 0, 0, 5)
    extraction_v.addWidget(force_bs_cb)
    
    force_bs_desc = QLabel("Overrides HTML2Text.")
    force_bs_desc.setStyleSheet("color: gray; font-size: 8pt;")
    force_bs_desc.setContentsMargins(20, 0, 0, 5)
    extraction_v.addWidget(force_bs_desc)
    
    # Separator
    sep_extract2 = QFrame()
    sep_extract2.setFrameShape(QFrame.HLine)
    sep_extract2.setFrameShadow(QFrame.Sunken)
    extraction_v.addWidget(sep_extract2)
    
    # Disable Section Merging (renamed from Chapter Merging)
    if not hasattr(self, 'disable_chapter_merging_var'):
        self.disable_chapter_merging_var = self.config.get('disable_chapter_merging', True)
    
    disable_merging_cb = self._create_styled_checkbox("Disable Section Merging")
    try:
        disable_merging_cb.setChecked(bool(self.disable_chapter_merging_var))
    except Exception:
        pass
    def _on_disable_merging_toggle(checked):
        try:
            self.disable_chapter_merging_var = bool(checked)
        except Exception:
            pass
    disable_merging_cb.toggled.connect(_on_disable_merging_toggle)
    disable_merging_cb.setContentsMargins(0, 2, 0, 0)
    extraction_v.addWidget(disable_merging_cb)
    
    disable_merging_desc = QLabel("Disable automatic merging of Section/Chapter pairs.\nEach file will be treated as a separate section.")
    disable_merging_desc.setStyleSheet("color: gray; font-size: 9pt;")
    disable_merging_desc.setContentsMargins(20, 0, 0, 5)
    extraction_v.addWidget(disable_merging_desc)
    
    # Request Merging (combine multiple chapters into single API request)
    if not hasattr(self, 'request_merging_enabled_var'):
        self.request_merging_enabled_var = self.config.get('request_merging_enabled', False)
    if not hasattr(self, 'request_merge_count_var'):
        self.request_merge_count_var = str(self.config.get('request_merge_count', 3))
    
    request_merge_cb = self._create_styled_checkbox("Request Merging")
    try:
        request_merge_cb.setChecked(bool(self.request_merging_enabled_var))
    except Exception:
        pass
    
    # Container for merge count setting
    merge_count_widgets = []
    
    def _on_request_merge_toggle(checked):
        try:
            self.request_merging_enabled_var = bool(checked)
            # Enable/disable the merge count field
            for widget in merge_count_widgets:
                widget.setEnabled(checked)
        except Exception:
            pass
    request_merge_cb.toggled.connect(_on_request_merge_toggle)
    request_merge_cb.setContentsMargins(0, 2, 0, 0)
    extraction_v.addWidget(request_merge_cb)
    
    # Row for merge count setting
    merge_count_row = QWidget()
    merge_count_h = QHBoxLayout(merge_count_row)
    merge_count_h.setContentsMargins(20, 2, 0, 0)
    
    merge_count_label = QLabel("Chapters per request:")
    merge_count_h.addWidget(merge_count_label)
    merge_count_widgets.append(merge_count_label)
    
    merge_count_edit = QLineEdit()
    merge_count_edit.setFixedWidth(50)
    try:
        merge_count_edit.setText(str(self.request_merge_count_var))
    except Exception:
        merge_count_edit.setText("3")
    def _on_merge_count_changed(text):
        try:
            self.request_merge_count_var = text
        except Exception:
            pass
    merge_count_edit.textChanged.connect(_on_merge_count_changed)
    merge_count_h.addWidget(merge_count_edit)
    merge_count_widgets.append(merge_count_edit)
    
    merge_count_h.addWidget(QLabel("(default: 3)"))
    merge_count_h.addStretch()
    extraction_v.addWidget(merge_count_row)
    
    # Set initial enabled state
    for widget in merge_count_widgets:
        widget.setEnabled(bool(self.request_merging_enabled_var))
    
    request_merge_desc = QLabel("Combine multiple chapters into a single translation request.\nReduces API overhead and may improve context consistency.\n⚠️ EPUB and PDF files only - does NOT work with .txt, .csv, .json, or .md files.")
    request_merge_desc.setStyleSheet("color: gray; font-size: 9pt;")
    request_merge_desc.setContentsMargins(20, 0, 0, 5)
    extraction_v.addWidget(request_merge_desc)
    
    # Split the Merge (split merged output back into individual files by headers)
    if not hasattr(self, 'split_the_merge_var'):
        self.split_the_merge_var = self.config.get('split_the_merge', True)
    
    split_merge_cb = self._create_styled_checkbox("Split the Merge")
    try:
        split_merge_cb.setChecked(bool(self.split_the_merge_var))
    except Exception:
        pass
    
    # Track widgets that depend on request merging being enabled
    split_merge_widgets = [split_merge_cb]
    
    def _on_split_merge_toggle(checked):
        try:
            self.split_the_merge_var = bool(checked)
        except Exception:
            pass
    split_merge_cb.toggled.connect(_on_split_merge_toggle)
    split_merge_cb.setContentsMargins(20, 2, 0, 0)  # Indented to show it's a sub-option
    extraction_v.addWidget(split_merge_cb)
    
    split_merge_desc = QLabel("Split merged translation output back into separate files using invisible markers.\nEach chapter gets its own file named after the original content.opf entry.\nMarkers are automatically preserved during translation for reliable splitting.")
    split_merge_desc.setStyleSheet("color: gray; font-size: 9pt;")
    split_merge_desc.setContentsMargins(40, 0, 0, 5)
    extraction_v.addWidget(split_merge_desc)
    split_merge_widgets.append(split_merge_desc)
    
    # Disable Fallback (mark as qa_failed if split fails)
    if not hasattr(self, 'disable_merge_fallback_var'):
        self.disable_merge_fallback_var = self.config.get('disable_merge_fallback', True)
    
    disable_fallback_cb = self._create_styled_checkbox("Disable Fallback")
    try:
        disable_fallback_cb.setChecked(bool(self.disable_merge_fallback_var))
    except Exception:
        pass
    
    def _on_disable_fallback_toggle(checked):
        try:
            self.disable_merge_fallback_var = bool(checked)
        except Exception:
            pass
    disable_fallback_cb.toggled.connect(_on_disable_fallback_toggle)
    disable_fallback_cb.setContentsMargins(40, 2, 0, 0)  # Double indented to show it's a sub-sub-option
    extraction_v.addWidget(disable_fallback_cb)
    split_merge_widgets.append(disable_fallback_cb)
    
    disable_fallback_desc = QLabel("Mark merged chapters as qa_failed if split fails (no fallback to merged file).\nUseful when you want to manually review failed splits.")
    disable_fallback_desc.setStyleSheet("color: gray; font-size: 9pt;")
    disable_fallback_desc.setContentsMargins(60, 0, 0, 5)
    extraction_v.addWidget(disable_fallback_desc)
    split_merge_widgets.append(disable_fallback_desc)
    
    # NOTE: Split markers are now ALWAYS enabled (hardcoded)
    # They are required for split-the-merge to work, so no toggle is needed
    # The toggle has been removed from the UI
    
    # Set initial enabled state for split merge (depends on request merging)
    for widget in split_merge_widgets:
        widget.setEnabled(bool(self.request_merging_enabled_var))
    
    # Update the request merge toggle to also control split merge widgets
    original_on_request_merge_toggle = _on_request_merge_toggle
    def _on_request_merge_toggle_with_split(checked):
        original_on_request_merge_toggle(checked)
        # Enable/disable split merge option (but don't change its checked state)
        for widget in split_merge_widgets:
            widget.setEnabled(checked)
    request_merge_cb.toggled.disconnect(_on_request_merge_toggle)
    request_merge_cb.toggled.connect(_on_request_merge_toggle_with_split)
    
    section_v.addWidget(extraction_box)
    
    # === REMAINING OPTIONS ===
    # Disable Image Gallery
    gallery_cb = self._create_styled_checkbox("Disable Image Gallery in EPUB")
    try:
        gallery_cb.setChecked(bool(self.disable_epub_gallery_var))
    except Exception:
        pass
    def _on_gallery_toggle(checked):
        try:
            self.disable_epub_gallery_var = bool(checked)
        except Exception:
            pass
    gallery_cb.toggled.connect(_on_gallery_toggle)
    gallery_cb.setContentsMargins(0, 2, 0, 0)
    section_v.addWidget(gallery_cb)
    
    gallery_desc = QLabel("Skip creating image gallery page in EPUB")
    gallery_desc.setStyleSheet("color: gray; font-size: 10pt;")
    gallery_desc.setContentsMargins(20, 0, 0, 10)
    section_v.addWidget(gallery_desc)
    
    # Disable Automatic Cover Creation
    cover_cb = self._create_styled_checkbox("Disable Automatic Cover Creation")
    try:
        cover_cb.setChecked(bool(self.disable_automatic_cover_creation_var))
    except Exception:
        pass
    def _on_cover_toggle(checked):
        try:
            self.disable_automatic_cover_creation_var = bool(checked)
        except Exception:
            pass
    cover_cb.toggled.connect(_on_cover_toggle)
    cover_cb.setContentsMargins(0, 2, 0, 0)
    section_v.addWidget(cover_cb)
    
    cover_desc = QLabel("No auto-generated cover page is created.")
    cover_desc.setStyleSheet("color: gray; font-size: 10pt;")
    cover_desc.setContentsMargins(20, 0, 0, 10)
    section_v.addWidget(cover_desc)
    
    # Translate special files (cover, nav, toc, etc.)
    translate_special_cb = self._create_styled_checkbox("Translate Special Files (Skip Override)")
    try:
        translate_special_cb.setChecked(bool(self.translate_special_files_var))
    except Exception:
        pass
    def _on_translate_special_toggle(checked):
        try:
            old_value = self.translate_special_files_var
            self.translate_special_files_var = bool(checked)
            # Show helpful message if value changed
            if old_value != bool(checked):
                if checked:
                    self.append_log("✅ Special files override ENABLED - special files will be included in extraction")
                    self.append_log("🔄 If you already extracted an EPUB, re-translate to apply this setting")
                else:
                    self.append_log("❌ Special files override DISABLED - special files will be skipped (default behavior)")
        except Exception:
            pass
    translate_special_cb.toggled.connect(_on_translate_special_toggle)
    translate_special_cb.setContentsMargins(0, 2, 0, 0)
    section_v.addWidget(translate_special_cb)
    
    translate_special_desc = QLabel("Forces translation of special files (cover, nav, toc, message, etc.)\ninstead of skipping them during extraction and compilation.")
    translate_special_desc.setStyleSheet("color: gray; font-size: 10pt;")
    translate_special_desc.setContentsMargins(20, 0, 0, 10)
    section_v.addWidget(translate_special_desc)
    
    # === PDF OUTPUT SETTINGS ===
    # Separator
    pdf_sep = QFrame()
    pdf_sep.setFrameShape(QFrame.HLine)
    pdf_sep.setFrameShadow(QFrame.Sunken)
    section_v.addWidget(pdf_sep)
    
    # PDF Output Format section title
    pdf_title = QLabel("PDF Output Settings")
    pdf_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
    pdf_title.setContentsMargins(0, 5, 0, 5)
    section_v.addWidget(pdf_title)
    
    # Initialize PDF output format variable
    if not hasattr(self, 'pdf_output_format_var'):
        self.pdf_output_format_var = self.config.get('pdf_output_format', 'pdf')
    
    # PDF Output Format toggle
    pdf_format_row = QWidget()
    pdf_format_h = QHBoxLayout(pdf_format_row)
    pdf_format_h.setContentsMargins(20, 2, 0, 0)
    
    pdf_format_label = QLabel("Output format:")
    pdf_format_h.addWidget(pdf_format_label)
    
    pdf_format_combo = QComboBox()
    pdf_format_combo.addItems(["pdf", "epub"])
    pdf_format_combo.setFixedWidth(100)
    pdf_format_combo.setStyleSheet("""
        QComboBox::down-arrow {
            image: none;
            width: 12px;
            height: 12px;
            border: none;
        }
    """)
    self._add_combobox_arrow(pdf_format_combo)
    self._disable_combobox_mousewheel(pdf_format_combo)
    try:
        format_val = self.pdf_output_format_var
        idx = pdf_format_combo.findText(format_val)
        if idx >= 0:
            pdf_format_combo.setCurrentIndex(idx)
    except Exception:
        pass
    def _on_pdf_format_changed(text):
        try:
            self.pdf_output_format_var = text
        except Exception:
            pass
    pdf_format_combo.currentTextChanged.connect(_on_pdf_format_changed)
    pdf_format_h.addWidget(pdf_format_combo)
    pdf_format_h.addStretch()
    section_v.addWidget(pdf_format_row)
    
    pdf_format_desc = QLabel("Choose whether to output PDFs as .pdf or .epub files.\nPDF: Creates combined PDF with all translated pages\nEPUB: Compiles pages into EPUB format")
    pdf_format_desc.setStyleSheet("color: gray; font-size: 10pt;")
    pdf_format_desc.setContentsMargins(20, 0, 0, 10)
    section_v.addWidget(pdf_format_desc)
    
    # Initialize PDF render mode variable
    if not hasattr(self, 'pdf_render_mode_var'):
        self.pdf_render_mode_var = self.config.get('pdf_render_mode', 'xhtml')
    
    # PDF Render Mode toggle
    pdf_render_row = QWidget()
    pdf_render_h = QHBoxLayout(pdf_render_row)
    pdf_render_h.setContentsMargins(20, 2, 0, 0)
    
    pdf_render_label = QLabel("Render mode:")
    pdf_render_h.addWidget(pdf_render_label)
    
    pdf_render_combo = QComboBox()
    pdf_render_combo.addItems(["absolute", "semantic", "xhtml", "html"])
    pdf_render_combo.setFixedWidth(100)
    pdf_render_combo.setStyleSheet("""
        QComboBox::down-arrow {
            image: none;
            width: 12px;
            height: 12px;
            border: none;
        }
    """)
    self._add_combobox_arrow(pdf_render_combo)
    self._disable_combobox_mousewheel(pdf_render_combo)
    try:
        render_val = self.pdf_render_mode_var
        idx = pdf_render_combo.findText(render_val)
        if idx >= 0:
            pdf_render_combo.setCurrentIndex(idx)
    except Exception:
        pass
    def _on_pdf_render_changed(text):
        try:
            self.pdf_render_mode_var = text
        except Exception:
            pass
    pdf_render_combo.currentTextChanged.connect(_on_pdf_render_changed)
    pdf_render_h.addWidget(pdf_render_combo)
    pdf_render_h.addStretch()
    section_v.addWidget(pdf_render_row)
    
    pdf_render_desc = QLabel("PDF extraction mode:\n• absolute: Fixed positioning (perfect layout, smaller payloads)\n• semantic: Semantic HTML (better text flow, larger payloads)\n• xhtml/html: MuPDF native rendering (1:1 layout)")
    pdf_render_desc.setStyleSheet("color: gray; font-size: 10pt;")
    pdf_render_desc.setContentsMargins(20, 0, 0, 10)
    section_v.addWidget(pdf_render_desc)
    
    # Disable 0-based Chapter Detection
    zero_detect_cb = self._create_styled_checkbox("Disable 0-based Chapter Detection")
    try:
        zero_detect_cb.setChecked(bool(self.disable_zero_detection_var))
    except Exception:
        pass
    def _on_zero_detect_toggle(checked):
        try:
            self.disable_zero_detection_var = bool(checked)
        except Exception:
            pass
    zero_detect_cb.toggled.connect(_on_zero_detect_toggle)
    zero_detect_cb.setContentsMargins(0, 2, 0, 0)
    section_v.addWidget(zero_detect_cb)
    
    zero_detect_desc = QLabel("Always use chapter ranges as specified\n(don't force adjust to chapter 1)")
    zero_detect_desc.setStyleSheet("color: gray; font-size: 10pt;")
    zero_detect_desc.setContentsMargins(20, 0, 0, 10)
    section_v.addWidget(zero_detect_desc)
    
    # Use Header as Output Name
    header_output_cb = self._create_styled_checkbox("Use Header as Output Name")
    try:
        header_output_cb.setChecked(bool(self.use_header_as_output_var))
    except Exception:
        pass
    def _on_header_output_toggle(checked):
        try:
            self.use_header_as_output_var = bool(checked)
        except Exception:
            pass
    header_output_cb.toggled.connect(_on_header_output_toggle)
    header_output_cb.setContentsMargins(0, 2, 0, 0)
    section_v.addWidget(header_output_cb)
    
    header_output_desc = QLabel("Use chapter headers/titles as output filenames")
    header_output_desc.setStyleSheet("color: gray; font-size: 10pt;")
    header_output_desc.setContentsMargins(20, 0, 0, 10)
    section_v.addWidget(header_output_desc)
    
    # Separator
    sep_opts1 = QFrame()
    sep_opts1.setFrameShape(QFrame.HLine)
    sep_opts1.setFrameShadow(QFrame.Sunken)
    section_v.addWidget(sep_opts1)
    
    # Chapter Number Offset
    offset_w = QWidget()
    offset_h = QHBoxLayout(offset_w)
    offset_h.setContentsMargins(0, 5, 0, 0)
    offset_h.addWidget(QLabel("Chapter Number Offset:"))
    
    if not hasattr(self, 'chapter_number_offset_var'):
        self.chapter_number_offset_var = str(self.config.get('chapter_number_offset', '0'))
    
    offset_edit = QLineEdit()
    offset_edit.setFixedWidth(60)
    try:
        offset_edit.setText(str(self.chapter_number_offset_var))
    except Exception:
        pass
    def _on_offset_changed(text):
        try:
            self.chapter_number_offset_var = text
        except Exception:
            pass
    offset_edit.textChanged.connect(_on_offset_changed)
    offset_h.addWidget(offset_edit)
    offset_h.addWidget(QLabel("(+/- adjustment)"))
    offset_h.addStretch()
    section_v.addWidget(offset_w)
    
    offset_desc = QLabel("Adjust all chapter numbers by this amount.\nUseful for matching file numbers to actual chapters.")
    offset_desc.setStyleSheet("color: gray; font-size: 10pt;")
    offset_desc.setContentsMargins(20, 0, 0, 10)
    section_v.addWidget(offset_desc)
    
    # Separator
    sep_opts2 = QFrame()
    sep_opts2.setFrameShape(QFrame.HLine)
    sep_opts2.setFrameShadow(QFrame.Sunken)
    section_v.addWidget(sep_opts2)
    
    # Post-Translation Scanning Phase
    scan_w = QWidget()
    scan_h = QHBoxLayout(scan_w)
    scan_h.setContentsMargins(0, 10, 0, 0)
    
    scan_cb = self._create_styled_checkbox("Enable post-translation Scanning phase")
    try:
        scan_cb.setChecked(bool(self.scan_phase_enabled_var))
    except Exception:
        pass
    def _on_scan_toggle(checked):
        try:
            self.scan_phase_enabled_var = bool(checked)
        except Exception:
            pass
    scan_cb.toggled.connect(_on_scan_toggle)
    scan_h.addWidget(scan_cb)
    
    scan_h.addSpacing(15)
    scan_h.addWidget(QLabel("Mode:"))
    
    scan_combo = QComboBox()
    scan_combo.addItems(["quick-scan", "aggressive", "ai-hunter", "custom"])
    scan_combo.setFixedWidth(120)
    # Add custom styling with unicode arrow
    scan_combo.setStyleSheet("""
        QComboBox::down-arrow {
            image: none;
            width: 12px;
            height: 12px;
            border: none;
        }
    """)
    self._add_combobox_arrow(scan_combo)
    self._disable_combobox_mousewheel(scan_combo)
    try:
        mode_val = self.scan_phase_mode_var
        idx = scan_combo.findText(mode_val)
        if idx >= 0:
            scan_combo.setCurrentIndex(idx)
    except Exception:
        pass
    def _on_scan_mode_changed(text):
        try:
            self.scan_phase_mode_var = text
        except Exception:
            pass
    scan_combo.currentTextChanged.connect(_on_scan_mode_changed)
    scan_h.addWidget(scan_combo)
    scan_h.addStretch()
    section_v.addWidget(scan_w)
    
    scan_desc = QLabel("Automatically run QA Scanner after translation completes")
    scan_desc.setStyleSheet("color: gray; font-size: 10pt;")
    scan_desc.setContentsMargins(20, 0, 0, 10)
    section_v.addWidget(scan_desc)
    
    # Conservative Batching
    batch_cb = self._create_styled_checkbox("Use Conservative Batching")
    try:
        batch_cb.setChecked(bool(self.conservative_batching_var))
    except Exception:
        pass
    def _on_batch_toggle(checked):
        try:
            self.conservative_batching_var = bool(checked)
        except Exception:
            pass
    batch_cb.toggled.connect(_on_batch_toggle)
    batch_cb.setContentsMargins(0, 10, 0, 0)
    section_v.addWidget(batch_cb)
    
    batch_desc = QLabel("Groups chapters in batches of 3x batch size for memory management\nWhen disabled (default): Uses direct batch size for faster processing")
    batch_desc.setStyleSheet("color: gray; font-size: 10pt;")
    batch_desc.setContentsMargins(20, 0, 0, 10)
    section_v.addWidget(batch_desc)
    
    # Separator
    sep_opts3 = QFrame()
    sep_opts3.setFrameShape(QFrame.HLine)
    sep_opts3.setFrameShadow(QFrame.Sunken)
    section_v.addWidget(sep_opts3)
    
    # API Safety Settings
    safety_title = QLabel("API Safety Settings")
    safety_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
    safety_title.setContentsMargins(0, 5, 0, 5)
    section_v.addWidget(safety_title)
    
    if not hasattr(self, 'disable_gemini_safety_var'):
        self.disable_gemini_safety_var = self.config.get('disable_gemini_safety', True)
    
    safety_cb = self._create_styled_checkbox("Disable API Safety Filters (Gemini, Groq, Fireworks, etc.)")
    try:
        safety_cb.setChecked(bool(self.disable_gemini_safety_var))
    except Exception:
        pass
    def _on_safety_toggle(checked):
        try:
            self.disable_gemini_safety_var = bool(checked)
        except Exception:
            pass
    safety_cb.toggled.connect(_on_safety_toggle)
    safety_cb.setContentsMargins(0, 5, 0, 0)
    section_v.addWidget(safety_cb)
    
    safety_warning = QLabel("⚠️ Disables content safety filters for supported providers.\nGemini: Sets all harm categories to BLOCK_NONE.\nGroq/Fireworks: Disables moderation parameter.")
    safety_warning.setStyleSheet("color: #ff6b6b; font-size: 9pt;")
    safety_warning.setContentsMargins(20, 0, 0, 5)
    section_v.addWidget(safety_warning)
    
    safety_note = QLabel("Does NOT affect ElectronHub Gemini models (eh/gemini-*) or Together AI")
    safety_note.setStyleSheet("color: gray; font-size: 8pt;")
    safety_note.setContentsMargins(20, 0, 0, 8)
    section_v.addWidget(safety_note)
    
    # OpenRouter Transport Preference
    if not hasattr(self, 'openrouter_http_only_var'):
        self.openrouter_http_only_var = self.config.get('openrouter_use_http_only', False)
    
    http_only_cb = self._create_styled_checkbox("Use HTTP-only for OpenRouter (bypass SDK)")
    try:
        http_only_cb.setChecked(bool(self.openrouter_http_only_var))
    except Exception:
        pass
    def _on_http_only_toggle(checked):
        try:
            self.openrouter_http_only_var = bool(checked)
        except Exception:
            pass
    http_only_cb.toggled.connect(_on_http_only_toggle)
    http_only_cb.setContentsMargins(0, 8, 0, 0)
    section_v.addWidget(http_only_cb)
    
    http_only_desc = QLabel("Requests to OpenRouter use direct HTTP POST with explicit headers")
    http_only_desc.setStyleSheet("color: gray; font-size: 9pt;")
    http_only_desc.setContentsMargins(20, 0, 0, 5)
    section_v.addWidget(http_only_desc)
    
    # OpenRouter Disable Compression
    if not hasattr(self, 'openrouter_accept_identity_var'):
        self.openrouter_accept_identity_var = self.config.get('openrouter_accept_identity', False)
    
    accept_identity_cb = self._create_styled_checkbox("Disable compression for OpenRouter (Accept-Encoding)")
    try:
        accept_identity_cb.setChecked(bool(self.openrouter_accept_identity_var))
    except Exception:
        pass
    def _on_accept_identity_toggle(checked):
        try:
            self.openrouter_accept_identity_var = bool(checked)
        except Exception:
            pass
    accept_identity_cb.toggled.connect(_on_accept_identity_toggle)
    accept_identity_cb.setContentsMargins(0, 4, 0, 0)
    section_v.addWidget(accept_identity_cb)
    
    accept_identity_desc = QLabel("Sends Accept-Encoding: identity to request uncompressed responses.\nUse if proxies/CDNs cause corrupted or non-JSON compressed bodies.")
    accept_identity_desc.setStyleSheet("color: gray; font-size: 8pt;")
    accept_identity_desc.setContentsMargins(20, 0, 0, 8)
    section_v.addWidget(accept_identity_desc)
    
    # OpenRouter: Provider preference
    # Default to 'Auto' when missing or blank
    if not hasattr(self, 'openrouter_preferred_provider_var'):
        try:
            v = self.config.get('openrouter_preferred_provider', 'Auto')
            v = (v or '').strip() or 'Auto'
        except Exception:
            v = 'Auto'
        self.openrouter_preferred_provider_var = v
        # Keep config aligned so it won't come back blank next time
        try:
            self.config['openrouter_preferred_provider'] = v
        except Exception:
            pass
    
    provider_w = QWidget()
    provider_h = QHBoxLayout(provider_w)
    provider_h.setContentsMargins(0, 4, 0, 0)
    provider_h.addWidget(QLabel("Preferred OpenRouter Provider:"))
    
    # Comprehensive list of OpenRouter providers (alphabetically sorted, with Auto first)
    provider_options = [
        'Auto', 'AI21', 'AionLabs', 'Alibaba Cloud Int.', 'Amazon Bedrock', 'Anthropic',
        'AtlasCloud', 'Atoma', 'Avian.io', 'Azure', 'Baseten', 'Cerebras', 'Chutes',
        'Cloudflare', 'Cohere', 'CrofAI', 'Crusoe', 'DeepInfra', 'DeepSeek', 'Enfer',
        'Featherless', 'Fireworks', 'Friendli', 'GMICloud', 'Google AI Studio', 'Google Vertex',
        'Groq', 'Hyperbolic', 'Inception', 'inference.net', 'Infermatic', 'Inflection',
        'kluster.ai', 'Lambda', 'Lepton', 'Leschde', 'Liquid', 'Mancer (private)', 'Meta',
        'Minimax', 'Mistral', 'Moonshot AI', 'Morph', 'nCompass', 'Nebius AI Studio',
        'NextBit', 'Nineteen', 'NovitAI', 'NVIDIA', 'OpenAI', 'Open Inference', 'Parasail',
        'Perplexity', 'Phala', 'Relace', 'SambaNova', 'SiliconFlow', 'Stealth', 'Switchpoint',
        'Targon', 'Together', 'Ubicloud', 'Venice', 'Weights & Biases', 'xAI', 'Z.AI'
    ]
    
    # Create combobox with autocomplete support (editable)
    provider_combo = QComboBox()
    provider_combo.setEditable(True)
    provider_combo.addItems(provider_options)
    provider_combo.setFixedWidth(160)  # Reduced for more compact layout
    # Add custom styling with unicode arrow
    provider_combo.setStyleSheet("""
        QComboBox::down-arrow {
            image: none;
            width: 12px;
            height: 12px;
            border: none;
        }
    """)
    self._add_combobox_arrow(provider_combo)
    self._disable_combobox_mousewheel(provider_combo)
    try:
        idx = provider_combo.findText(self.openrouter_preferred_provider_var)
        if idx >= 0:
            provider_combo.setCurrentIndex(idx)
        else:
            provider_combo.setCurrentText(self.openrouter_preferred_provider_var)
    except Exception:
        pass
    def _on_provider_changed(text):
        try:
            self.openrouter_preferred_provider_var = text
        except Exception:
            pass
    provider_combo.currentTextChanged.connect(_on_provider_changed)
    provider_combo.lineEdit().textChanged.connect(_on_provider_changed)
    provider_h.addWidget(provider_combo)
    provider_h.addStretch()
    
    # Store reference for potential autocomplete logic (Tkinter specific, may not be needed)
    self.openrouter_provider_combo = provider_combo
    self._provider_all_values = provider_options
    self._provider_prev_text = self.openrouter_preferred_provider_var
    
    section_v.addWidget(provider_w)
    
    provider_desc = QLabel("Specify which upstream provider OpenRouter should prefer for your requests.\n'Auto' lets OpenRouter choose. Specific providers may have different availability.")
    provider_desc.setStyleSheet("color: gray; font-size: 8pt;")
    provider_desc.setContentsMargins(20, 0, 0, 8)
    section_v.addWidget(provider_desc)
    
    # Place the section at row 1, column 1 to match the original grid
    try:
        grid = parent.layout()
        if grid:
            grid.addWidget(section_box, 1, 1)
    except Exception:
        # Fallback: just stack
        section_box.setParent(parent)
    
    # Initial state - show/hide enhanced options
    self.on_extraction_method_change()

def on_extraction_method_change(self):
    """Handle extraction method changes and show/hide Enhanced options"""
    if hasattr(self, 'text_extraction_method_var') and hasattr(self, 'enhanced_options_frame'):
        try:
            # Qt version: use setVisible instead of pack/pack_forget
            if self.text_extraction_method_var == 'enhanced':
                self.enhanced_options_frame.setVisible(True)
            else:
                self.enhanced_options_frame.setVisible(False)
        except Exception:
            # Fallback for any errors during transition
            pass
            
def _enforce_image_output_dependency(self):
    """Enforce that image output mode is disabled when image translation is off,
    unless using the special gemini-3-pro-image-preview model."""
    try:
        # Check model exception
        model = str(getattr(self, 'model_var', '')).lower() 
        allow_without_translation = 'gemini-3-pro-image-preview' in model
        
        # Check if image translation is enabled
        image_translation_on = bool(getattr(self, 'enable_image_translation_var', False))
        
        # Determine if image output should be allowed
        allowed = image_translation_on or allow_without_translation
        
        if not allowed:
            # Force disable image output mode silently
            self.enable_image_output_mode_var = False
    except Exception:
        pass

def _create_image_translation_section(self, parent):
    """Create image translation section (PySide6)"""
    from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QWidget, QLineEdit, QGridLayout
    from PySide6.QtCore import Qt
    
    section_box = QGroupBox("Image Translation & Vision API")
    section_v = QVBoxLayout(section_box)
    section_v.setContentsMargins(8, 8, 8, 8)  # Compact margins
    section_v.setSpacing(4)  # Compact spacing between widgets
    
    # Create horizontal container for two columns inside content container
    columns_container = QWidget()
    section_h = QHBoxLayout(columns_container)
    section_h.setContentsMargins(0, 0, 0, 0)
    section_h.setSpacing(10)
    
    # Left column
    left_column = QWidget()
    left_v = QVBoxLayout(left_column)
    left_v.setContentsMargins(0, 0, 20, 0)
    
    # Enable Image Translation
    enable_cb = self._create_styled_checkbox("Enable Image Translation")
    try:
        enable_cb.setChecked(bool(self.enable_image_translation_var))
    except Exception:
        pass
    def _on_enable_image_toggle(checked):
        try:
            self.enable_image_translation_var = bool(checked)
            self.toggle_image_translation_section()
            # Enforce image output dependency when image translation changes
            if hasattr(self, '_enforce_image_output_dependency'):
                self._enforce_image_output_dependency()
        except Exception:
            pass
    enable_cb.toggled.connect(_on_enable_image_toggle)
    section_v.addWidget(enable_cb)
    
    enable_desc = QLabel("Extracts and translates text from images using vision models")
    enable_desc.setStyleSheet("color: gray; font-size: 10pt;")
    enable_desc.setContentsMargins(0, 0, 0, 10)
    section_v.addWidget(enable_desc)
    
    # Create container for all content below the main checkbox
    content_container = QWidget()
    content_layout = QVBoxLayout(content_container)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(4)
    
    # Store reference for fade animation (everything below main checkbox)
    self.image_translation_content = content_container
    
    # Process Long Images
    webnovel_cb = self._create_styled_checkbox("Process Long Images (Web Novel Style)")
    try:
        webnovel_cb.setChecked(bool(self.process_webnovel_images_var))
    except Exception:
        pass
    def _on_webnovel_toggle(checked):
        try:
            self.process_webnovel_images_var = bool(checked)
        except Exception:
            pass
    webnovel_cb.toggled.connect(_on_webnovel_toggle)
    left_v.addWidget(webnovel_cb)
    
    webnovel_desc = QLabel("Include tall images often used in web novels")
    webnovel_desc.setStyleSheet("color: gray; font-size: 10pt;")
    webnovel_desc.setContentsMargins(20, 0, 0, 10)
    left_v.addWidget(webnovel_desc)
    
    # Hide labels and remove OCR images
    hide_cb = self._create_styled_checkbox("Hide labels and remove OCR images")
    try:
        hide_cb.setChecked(bool(self.hide_image_translation_label_var))
    except Exception:
        pass
    def _on_hide_toggle(checked):
        try:
            self.hide_image_translation_label_var = bool(checked)
        except Exception:
            pass
    hide_cb.toggled.connect(_on_hide_toggle)
    left_v.addWidget(hide_cb)
    
    hide_desc = QLabel("Clean mode: removes image and shows only translated text")
    hide_desc.setStyleSheet("color: gray; font-size: 10pt;")
    hide_desc.setContentsMargins(20, 0, 0, 10)
    left_v.addWidget(hide_desc)
    
    # Enable Image Output Mode
    image_output_cb = self._create_styled_checkbox("Enable Image Output Mode")
    try:
        image_output_cb.setChecked(bool(self.enable_image_output_mode_var))
    except Exception:
        pass
    def _on_image_output_toggle(checked):
        try:
            self.enable_image_output_mode_var = bool(checked)
            # Save to config
            self.config['enable_image_output_mode'] = bool(checked)
        except Exception:
            pass
    image_output_cb.toggled.connect(_on_image_output_toggle)
    left_v.addWidget(image_output_cb)
    
    image_output_desc = QLabel("Request image output from vision models (e.g. gemini-3-pro-image-preview)")
    image_output_desc.setStyleSheet("color: gray; font-size: 10pt;")
    image_output_desc.setContentsMargins(20, 0, 0, 5)
    left_v.addWidget(image_output_desc)
    
    # Image Output Resolution dropdown
    resolution_w = QWidget()
    resolution_h = QHBoxLayout(resolution_w)
    resolution_h.setContentsMargins(20, 0, 0, 0)
    resolution_h.setSpacing(8)
    resolution_h.addWidget(QLabel("Output Resolution:"))
    
    resolution_combo = QComboBox()
    resolution_combo.addItems(["1K", "2K", "4K"])
    resolution_combo.setFixedWidth(80)
    resolution_combo.setStyleSheet("""
        QComboBox::down-arrow {
            image: none;
            width: 12px;
            height: 12px;
            border: none;
        }
    """)
    self._add_combobox_arrow(resolution_combo)
    self._disable_combobox_mousewheel(resolution_combo)
    
    # Initialize variable if not exists
    if not hasattr(self, 'image_output_resolution_var'):
        self.image_output_resolution_var = self.config.get('image_output_resolution', '1K')
    
    try:
        idx = resolution_combo.findText(self.image_output_resolution_var)
        if idx >= 0:
            resolution_combo.setCurrentIndex(idx)
    except Exception:
        pass
    
    def _on_resolution_changed(text):
        try:
            self.image_output_resolution_var = text
        except Exception:
            pass
    resolution_combo.currentTextChanged.connect(_on_resolution_changed)
    resolution_h.addWidget(resolution_combo)
    resolution_h.addStretch()
    
    left_v.addWidget(resolution_w)
    
    resolution_desc = QLabel("Higher resolution = better quality but slower generation")
    resolution_desc.setStyleSheet("color: gray; font-size: 10pt;")
    resolution_desc.setContentsMargins(40, 0, 0, 10)
    left_v.addWidget(resolution_desc)
    
    left_v.addSpacing(10)
    
    # Watermark Removal
    watermark_cb = self._create_styled_checkbox("Enable Watermark Removal")
    try:
        watermark_cb.setChecked(bool(self.enable_watermark_removal_var))
    except Exception:
        pass
    def _on_watermark_toggle(checked):
        try:
            self.enable_watermark_removal_var = bool(checked)
            _toggle_watermark_options()
        except Exception:
            pass
    watermark_cb.toggled.connect(_on_watermark_toggle)
    left_v.addWidget(watermark_cb)
    
    watermark_desc = QLabel("Advanced preprocessing to remove watermarks from images")
    watermark_desc.setStyleSheet("color: gray; font-size: 10pt;")
    watermark_desc.setContentsMargins(20, 0, 0, 10)
    left_v.addWidget(watermark_desc)
    
    # Save Cleaned Images
    self.save_cleaned_checkbox = self._create_styled_checkbox("Save Cleaned Images")
    try:
        self.save_cleaned_checkbox.setChecked(bool(self.save_cleaned_images_var))
    except Exception:
        pass
    def _on_save_cleaned_toggle(checked):
        try:
            self.save_cleaned_images_var = bool(checked)
        except Exception:
            pass
    self.save_cleaned_checkbox.toggled.connect(_on_save_cleaned_toggle)
    self.save_cleaned_checkbox.setContentsMargins(20, 0, 0, 0)
    left_v.addWidget(self.save_cleaned_checkbox)
    
    save_desc = QLabel("Keep watermark-removed images in translated_images/cleaned/")
    save_desc.setStyleSheet("color: gray; font-size: 10pt;")
    save_desc.setContentsMargins(40, 0, 0, 10)
    left_v.addWidget(save_desc)
    
    # Advanced Watermark Removal
    self.advanced_watermark_checkbox = self._create_styled_checkbox("Advanced Watermark Removal")
    try:
        self.advanced_watermark_checkbox.setChecked(bool(self.advanced_watermark_removal_var))
    except Exception:
        pass
    def _on_advanced_watermark_toggle(checked):
        try:
            self.advanced_watermark_removal_var = bool(checked)
        except Exception:
            pass
    self.advanced_watermark_checkbox.toggled.connect(_on_advanced_watermark_toggle)
    self.advanced_watermark_checkbox.setContentsMargins(20, 0, 0, 0)
    left_v.addWidget(self.advanced_watermark_checkbox)
    
    advanced_desc = QLabel("Use FFT-based pattern detection for stubborn watermarks")
    advanced_desc.setStyleSheet("color: gray; font-size: 10pt;")
    advanced_desc.setContentsMargins(40, 0, 0, 0)
    left_v.addWidget(advanced_desc)
    
    left_v.addStretch()
    section_h.addWidget(left_column)
    
    # Right column
    right_column = QWidget()
    right_v = QVBoxLayout(right_column)
    right_v.setContentsMargins(0, 0, 0, 0)
    
    # Settings grid
    settings_w = QWidget()
    settings_grid = QGridLayout(settings_w)
    settings_grid.setContentsMargins(0, 0, 0, 0)
    
    settings = [
        ("Min Image height (px):", self.webnovel_min_height_var, False),
        ("Max Images per chapter:", self.max_images_per_chapter_var, False),
        ("Chunk height:", self.image_chunk_height_var, False),
        ("Chunk overlap (%):", self.image_chunk_overlap_var, True)
    ]
    
    for row, (label, var, has_tip) in enumerate(settings):
        lbl = QLabel(label)
        settings_grid.addWidget(lbl, row, 0, Qt.AlignLeft)
        
        entry = QLineEdit()
        entry.setFixedWidth(80)
        try:
            entry.setText(str(var))
        except Exception:
            pass
        def _make_entry_callback(var_name):
            def _cb(text):
                try:
                    setattr(self, var_name, text)
                except Exception:
                    pass
            return _cb
        # Extract variable name from var reference
        var_name = None
        for name in ['webnovel_min_height_var', 'max_images_per_chapter_var', 'image_chunk_height_var', 'image_chunk_overlap_var']:
            if hasattr(self, name) and getattr(self, name) is var:
                var_name = name
                break
        if var_name:
            entry.textChanged.connect(_make_entry_callback(var_name))
        settings_grid.addWidget(entry, row, 1, Qt.AlignLeft)
    
    right_v.addWidget(settings_w)
    right_v.addSpacing(15)
    
    # Send tall image chunks in single API call
    single_api_cb = self._create_styled_checkbox("Send tall image chunks in single API call (NOT RECOMMENDED)")
    try:
        single_api_cb.setChecked(bool(self.single_api_image_chunks_var))
    except Exception:
        pass
    def _on_single_api_toggle(checked):
        try:
            self.single_api_image_chunks_var = bool(checked)
        except Exception:
            pass
    single_api_cb.toggled.connect(_on_single_api_toggle)
    right_v.addWidget(single_api_cb)
    
    single_api_desc = QLabel("All image chunks sent to 1 API call (Most AI models don't like this)")
    single_api_desc.setStyleSheet("color: gray; font-size: 10pt;")
    single_api_desc.setContentsMargins(20, 0, 0, 10)
    right_v.addWidget(single_api_desc)
    
    models_info = QLabel("💡 Supported models:\n• Gemini 1.5 Pro/Flash, 2.0 Flash\n• GPT-4V, GPT-4o, o4-mini")
    models_info.setStyleSheet("color: #666; font-size: 10pt;")
    models_info.setContentsMargins(0, 10, 0, 0)
    right_v.addWidget(models_info)
    
    # Configuration buttons section
    right_v.addSpacing(20)
    
    config_title = QLabel("Advanced Configuration:")
    config_title.setStyleSheet("font-weight: bold; font-size: 10pt;")
    right_v.addWidget(config_title)
    
    # Image chunk prompt button
    btn_image_chunk = QPushButton("⚙️ Configure Image Chunk Prompt")
    btn_image_chunk.setFixedWidth(250)
    btn_image_chunk.clicked.connect(lambda: self.configure_image_chunk_prompt())
    right_v.addWidget(btn_image_chunk)
    
    btn_image_chunk_desc = QLabel("Configure context for tall image chunks")
    btn_image_chunk_desc.setStyleSheet("color: gray; font-size: 9pt;")
    btn_image_chunk_desc.setContentsMargins(0, 0, 0, 5)
    right_v.addWidget(btn_image_chunk_desc)
    
    # Image compression button
    btn_compression = QPushButton("🗜️ Configure Image Compression")
    btn_compression.setFixedWidth(250)
    btn_compression.clicked.connect(lambda: self.configure_image_compression())
    right_v.addWidget(btn_compression)
    
    btn_compression_desc = QLabel("Optimize images for API token efficiency")
    btn_compression_desc.setStyleSheet("color: gray; font-size: 9pt;")
    btn_compression_desc.setContentsMargins(0, 0, 0, 5)
    right_v.addWidget(btn_compression_desc)
    
    right_v.addStretch()
    section_h.addWidget(right_column)
    
    # Add the columns container to the content container
    content_layout.addWidget(columns_container)
    
    # Add the content container (everything below main checkbox) to main section
    section_v.addWidget(content_container)
    
    # Dependency logic for watermark options
    def _toggle_watermark_options():
        try:
            enabled = bool(self.enable_watermark_removal_var)
            self.save_cleaned_checkbox.setEnabled(enabled)
            self.advanced_watermark_checkbox.setEnabled(enabled)
            if not enabled:
                self.save_cleaned_images_var = False
                self.advanced_watermark_removal_var = False
        except Exception:
            pass
    
    # Call once to set initial state
    _toggle_watermark_options()
    
    # Initialize image translation section visibility
    self.toggle_image_translation_section()
    
    # Place the section at row 2, spanning both columns
    try:
        grid = parent.layout()
        if grid:
            grid.addWidget(section_box, 2, 0, 1, 2)
    except Exception:
        # Fallback: just stack
        section_box.setParent(parent)
    
def on_extraction_mode_change(self):
    """Handle extraction mode changes and show/hide Enhanced options"""
    try:
        # Qt version: use setVisible instead of pack/pack_forget
        if self.extraction_mode_var == 'enhanced':
            if hasattr(self, 'enhanced_options_separator'):
                self.enhanced_options_separator.setVisible(True)
            if hasattr(self, 'enhanced_options_frame'):
                self.enhanced_options_frame.setVisible(True)
        else:
            if hasattr(self, 'enhanced_options_separator'):
                self.enhanced_options_separator.setVisible(False)
            if hasattr(self, 'enhanced_options_frame'):
                self.enhanced_options_frame.setVisible(False)
    except Exception:
        # Fallback for any errors during transition
        pass
            
def _create_anti_duplicate_section(self, parent):
    """Create comprehensive anti-duplicate parameter controls with tabs (PySide6)"""
    from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QWidget, QLineEdit, QTabWidget, QSlider, QPushButton
    from PySide6.QtCore import Qt
    
    section_box = QGroupBox("🎯 Anti-Duplicate Parameters")
    section_v = QVBoxLayout(section_box)
    
    # Description
    desc_label = QLabel("Configure parameters to reduce duplicate translations across all AI providers.")
    desc_label.setStyleSheet("color: gray; font-size: 9pt;")
    desc_label.setWordWrap(True)
    desc_label.setMaximumWidth(520)
    desc_label.setContentsMargins(0, 0, 0, 10)
    section_v.addWidget(desc_label)
    
    # Enable/Disable toggle
    self.enable_anti_duplicate_var = self.config.get('enable_anti_duplicate', False)
    enable_cb = self._create_styled_checkbox("Enable Anti-Duplicate Parameters")
    try:
        enable_cb.setChecked(bool(self.enable_anti_duplicate_var))
    except Exception:
        pass
    def _on_enable_anti_dup_toggle(checked):
        try:
            self.enable_anti_duplicate_var = bool(checked)
            self._toggle_anti_duplicate_controls()
        except Exception:
            pass
    enable_cb.toggled.connect(_on_enable_anti_dup_toggle)
    enable_cb.setContentsMargins(0, 0, 0, 10)
    section_v.addWidget(enable_cb)
    
    # Create container for all content below the main checkbox
    content_container = QWidget()
    content_layout = QVBoxLayout(content_container)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(4)
    
    # Store reference for slide animation
    self.anti_duplicate_content = content_container
    
    # Create tab widget for organized parameters
    self.anti_duplicate_notebook = QTabWidget()
    
    # Enhanced tab styling
    self.anti_duplicate_notebook.setStyleSheet("""
        QTabWidget::pane {
            border: 1px solid #555;
            background-color: #2d2d2d;
            border-top-left-radius: 0px;
            border-top-right-radius: 4px;
            border-bottom-left-radius: 4px;
            border-bottom-right-radius: 4px;
        }
        QTabWidget::tab-bar {
            left: 5px;
        }
        QTabBar::tab {
            background-color: #404040;
            color: #cccccc;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            min-width: 80px;
            font-weight: 500;
        }
        QTabBar::tab:selected {
            background-color: #2d2d2d;
            color: #ffffff;
            border-bottom: 2px solid #0078d4;
            font-weight: bold;
        }
        QTabBar::tab:hover:!selected {
            background-color: #4a4a4a;
            color: #ffffff;
        }
        QTabBar::tab:first {
            margin-left: 0;
        }
    """)
    
    content_layout.addWidget(self.anti_duplicate_notebook)
    
    # Tab 1: Core Parameters
    core_frame = QWidget()
    core_v = QVBoxLayout(core_frame)
    core_v.setContentsMargins(10, 10, 10, 10)
    self.anti_duplicate_notebook.addTab(core_frame, "Core Parameters")
    
    # Top-P (Nucleus Sampling)
    def _create_slider_row(parent_layout, label_text, var_holder, var_name, min_val, max_val, decimals=2, is_int=False):
        """Helper to create a slider row with label and value display
        var_holder: object that holds the variable (typically self)
        var_name: string name of the attribute to set
        """
        row_w = QWidget()
        row_h = QHBoxLayout(row_w)
        row_h.setContentsMargins(0, 5, 0, 5)
        
        lbl = QLabel(label_text)
        lbl.setFixedWidth(200)
        row_h.addWidget(lbl)
        
        slider = QSlider(Qt.Horizontal)
        if is_int:
            slider.setMinimum(int(min_val))
            slider.setMaximum(int(max_val))
            try:
                current_val = getattr(var_holder, var_name, min_val)
                slider.setValue(int(current_val))
            except Exception:
                pass
        else:
            # For double values, use int slider and scale
            steps = int((max_val - min_val) / (0.01 if decimals == 2 else 0.1))
            slider.setMinimum(0)
            slider.setMaximum(steps)
            try:
                current = float(getattr(var_holder, var_name, min_val))
                slider.setValue(int((current - min_val) / (max_val - min_val) * steps))
            except Exception:
                pass
        slider.setFixedWidth(200)
        # Disable mousewheel scrolling on slider
        slider.wheelEvent = lambda event: None
        row_h.addWidget(slider)
        
        value_lbl = QLabel("")
        value_lbl.setFixedWidth(80)
        row_h.addWidget(value_lbl)
        row_h.addStretch()
        
        def _on_slider_change(value):
            try:
                if is_int:
                    actual_value = value
                    setattr(var_holder, var_name, actual_value)
                    if actual_value == 0:
                        value_lbl.setText("OFF")
                    else:
                        value_lbl.setText(f"{actual_value}")
                else:
                    actual_value = min_val + (value / slider.maximum()) * (max_val - min_val)
                    setattr(var_holder, var_name, actual_value)
                    if decimals == 1:
                        value_lbl.setText(f"{actual_value:.1f}" if actual_value > 0 else "OFF")
                    else:
                        value_lbl.setText(f"{actual_value:.2f}")
            except Exception:
                pass
        
        slider.valueChanged.connect(_on_slider_change)
        # Trigger initial update
        _on_slider_change(slider.value())
        
        parent_layout.addWidget(row_w)
        return slider, value_lbl
    
    self.top_p_var = self.config.get('top_p', 1.0)
    top_p_slider, self.top_p_value_label = _create_slider_row(core_v, "Top-P (Nucleus Sampling):", self, 'top_p_var', 0.1, 1.0, decimals=2)
    
    # Top-K (Vocabulary Limit)
    self.top_k_var = self.config.get('top_k', 0)
    top_k_slider, self.top_k_value_label = _create_slider_row(core_v, "Top-K (Vocabulary Limit):", self, 'top_k_var', 0, 100, is_int=True)
    
    # Frequency Penalty
    self.frequency_penalty_var = self.config.get('frequency_penalty', 0.0)
    freq_slider, self.freq_penalty_value_label = _create_slider_row(core_v, "Frequency Penalty:", self, 'frequency_penalty_var', 0.0, 2.0, decimals=2)
    
    # Presence Penalty
    self.presence_penalty_var = self.config.get('presence_penalty', 0.0)
    pres_slider, self.pres_penalty_value_label = _create_slider_row(core_v, "Presence Penalty:", self, 'presence_penalty_var', 0.0, 2.0, decimals=2)
    
    core_v.addStretch()
    
    # Tab 2: Advanced Parameters
    advanced_frame = QWidget()
    advanced_v = QVBoxLayout(advanced_frame)
    advanced_v.setContentsMargins(10, 10, 10, 10)
    self.anti_duplicate_notebook.addTab(advanced_frame, "Advanced")
    
    # Repetition Penalty
    self.repetition_penalty_var = self.config.get('repetition_penalty', 1.0)
    rep_slider, self.rep_penalty_value_label = _create_slider_row(advanced_v, "Repetition Penalty:", self, 'repetition_penalty_var', 1.0, 2.0, decimals=2)
    
    # Candidate Count (Gemini)
    self.candidate_count_var = self.config.get('candidate_count', 1)
    candidate_slider, self.candidate_value_label = _create_slider_row(advanced_v, "Candidate Count (Gemini):", self, 'candidate_count_var', 1, 4, is_int=True)
    
    advanced_v.addStretch()
    
    # Tab 3: Stop Sequences
    stop_frame = QWidget()
    stop_v = QVBoxLayout(stop_frame)
    stop_v.setContentsMargins(10, 10, 10, 10)
    self.anti_duplicate_notebook.addTab(stop_frame, "Stop Sequences")
    
    # Custom Stop Sequences
    stop_row = QWidget()
    stop_h = QHBoxLayout(stop_row)
    stop_h.setContentsMargins(0, 5, 0, 5)
    stop_h.addWidget(QLabel("Custom Stop Sequences:"))
    
    self.custom_stop_sequences_var = self.config.get('custom_stop_sequences', '')
    stop_entry = QLineEdit()
    stop_entry.setFixedWidth(300)
    try:
        stop_entry.setText(str(self.custom_stop_sequences_var))
    except Exception:
        pass
    def _on_stop_seq_changed(text):
        try:
            self.custom_stop_sequences_var = text
        except Exception:
            pass
    stop_entry.textChanged.connect(_on_stop_seq_changed)
    stop_h.addWidget(stop_entry)
    
    stop_tip = QLabel("(comma-separated)")
    stop_tip.setStyleSheet("color: gray; font-size: 8pt;")
    stop_h.addWidget(stop_tip)
    stop_h.addStretch()
    stop_v.addWidget(stop_row)
    stop_v.addStretch()
    
    # Tab 4: Logit Bias (OpenAI)
    bias_frame = QWidget()
    bias_v = QVBoxLayout(bias_frame)
    bias_v.setContentsMargins(10, 10, 10, 10)
    self.anti_duplicate_notebook.addTab(bias_frame, "Logit Bias")
    
    # Logit Bias Enable
    self.logit_bias_enabled_var = self.config.get('logit_bias_enabled', False)
    bias_cb = self._create_styled_checkbox("Enable Logit Bias (OpenAI only)")
    try:
        bias_cb.setChecked(bool(self.logit_bias_enabled_var))
    except Exception:
        pass
    def _on_bias_enable_toggle(checked):
        try:
            self.logit_bias_enabled_var = bool(checked)
        except Exception:
            pass
    bias_cb.toggled.connect(_on_bias_enable_toggle)
    bias_cb.setContentsMargins(0, 0, 0, 5)
    bias_v.addWidget(bias_cb)
    
    # Logit Bias Strength
    self.logit_bias_strength_var = self.config.get('logit_bias_strength', -0.5)
    bias_slider, self.bias_strength_value_label = _create_slider_row(bias_v, "Bias Strength:", self, 'logit_bias_strength_var', -2.0, 2.0, decimals=1)
    
    # Preset bias targets
    preset_title = QLabel("Preset Bias Targets:")
    preset_title.setStyleSheet("font-weight: bold; font-size: 9pt;")
    preset_title.setContentsMargins(0, 10, 0, 5)
    bias_v.addWidget(preset_title)
    
    self.bias_common_words_var = self.config.get('bias_common_words', False)
    common_cb = self._create_styled_checkbox("Bias against common words (the, and, said)")
    try:
        common_cb.setChecked(bool(self.bias_common_words_var))
    except Exception:
        pass
    def _on_common_toggle(checked):
        try:
            self.bias_common_words_var = bool(checked)
        except Exception:
            pass
    common_cb.toggled.connect(_on_common_toggle)
    bias_v.addWidget(common_cb)
    
    self.bias_repetitive_phrases_var = self.config.get('bias_repetitive_phrases', False)
    phrases_cb = self._create_styled_checkbox("Bias against repetitive phrases")
    try:
        phrases_cb.setChecked(bool(self.bias_repetitive_phrases_var))
    except Exception:
        pass
    def _on_phrases_toggle(checked):
        try:
            self.bias_repetitive_phrases_var = bool(checked)
        except Exception:
            pass
    phrases_cb.toggled.connect(_on_phrases_toggle)
    bias_v.addWidget(phrases_cb)
    bias_v.addStretch()
    
    # Provider compatibility info
    compat_title = QLabel("Parameter Compatibility:")
    compat_title.setStyleSheet("font-weight: bold; font-size: 9pt;")
    compat_title.setContentsMargins(0, 15, 0, 0)
    content_layout.addWidget(compat_title)
    
    compat_text = QLabel("• Core: Most providers • Advanced: DeepSeek, Mistral, Groq • Logit Bias: OpenAI only")
    compat_text.setStyleSheet("color: gray; font-size: 8pt;")
    compat_text.setContentsMargins(0, 5, 0, 0)
    content_layout.addWidget(compat_text)
    
    # Reset button
    reset_row = QWidget()
    reset_h = QHBoxLayout(reset_row)
    reset_h.setContentsMargins(0, 10, 0, 0)
    
    reset_btn = QPushButton("🔄 Reset to Defaults")
    reset_btn.setFixedWidth(180)
    reset_btn.clicked.connect(lambda: self._reset_anti_duplicate_defaults())
    reset_h.addWidget(reset_btn)
    
    reset_desc = QLabel("Reset all anti-duplicate parameters to default values")
    reset_desc.setStyleSheet("color: gray; font-size: 8pt;")
    reset_h.addWidget(reset_desc)
    reset_h.addStretch()
    content_layout.addWidget(reset_row)
    
    # Add content container to main section
    section_v.addWidget(content_container)
    
    # Store all tab frames for enable/disable
    self.anti_duplicate_tabs = [core_frame, advanced_frame, stop_frame, bias_frame]
    
    # Place the section at row 6, spanning both columns
    try:
        grid = parent.layout()
        if grid:
            grid.addWidget(section_box, 6, 0, 1, 2)
    except Exception:
        # Fallback: just stack
        section_box.setParent(parent)
    
    # Initialize anti-duplicate section visibility
    self.toggle_anti_duplicate_section()

def toggle_anti_duplicate_section(self):
    """Toggle visibility of anti-duplicate content with smooth fade"""
    try:
        if not hasattr(self, 'anti_duplicate_content'):
            return
            
        enabled = bool(self.enable_anti_duplicate_var)
        
        # Import animation components
        from PySide6.QtCore import QPropertyAnimation, QEasingCurve
        from PySide6.QtWidgets import QGraphicsOpacityEffect
        
        # Stop any existing animation
        if hasattr(self, '_anti_duplicate_animation') and self._anti_duplicate_animation:
            self._anti_duplicate_animation.stop()
        
        # Ensure widget is visible for animation
        self.anti_duplicate_content.setVisible(True)
        
        # Create or get opacity effect
        if not hasattr(self.anti_duplicate_content, '_opacity_effect'):
            effect = QGraphicsOpacityEffect()
            self.anti_duplicate_content.setGraphicsEffect(effect)
            self.anti_duplicate_content._opacity_effect = effect
        else:
            effect = self.anti_duplicate_content._opacity_effect
        
        # Create opacity animation
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(150)  # Faster for no glitch
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        if enabled:
            # Fade in
            animation.setStartValue(0.0)
            animation.setEndValue(1.0)
        else:
            # Fade out
            animation.setStartValue(1.0)
            animation.setEndValue(0.0)
            # Hide after fade out
            animation.finished.connect(
                lambda: self.anti_duplicate_content.setVisible(False) if not enabled else None
            )
        
        self._anti_duplicate_animation = animation
        animation.start()
        
    except Exception:
        # Fallback to simple show/hide if animation fails
        try:
            if hasattr(self, 'anti_duplicate_content'):
                enabled = bool(self.enable_anti_duplicate_var)
                self.anti_duplicate_content.setVisible(enabled)
        except Exception:
            pass

def _toggle_anti_duplicate_controls(self):
    """Enable/disable anti-duplicate parameter controls (Qt version)"""
    # Call the slide animation function
    self.toggle_anti_duplicate_section()

def _toggle_http_tuning_controls(self):
    """Enable/disable the HTTP timeout/pooling controls as a group with proper styling (Qt version)"""
    try:
        enabled = bool(self.enable_http_tuning_var) if hasattr(self, 'enable_http_tuning_var') else False
    except Exception:
        enabled = False
    
    # Entry fields
    for attr in ['connect_timeout_entry', 'read_timeout_entry', 'http_pool_connections_entry', 'http_pool_maxsize_entry']:
        widget = getattr(self, attr, None)
        if widget is not None:
            try:
                widget.setEnabled(enabled)
            except Exception:
                pass
    
    # Labels with proper disabled styling
    label_attrs = [
        'connect_timeout_label', 'read_timeout_label', 
        'http_pool_connections_label', 'http_pool_maxsize_label'
    ]
    
    for attr in label_attrs:
        widget = getattr(self, attr, None)
        if widget is not None:
            try:
                widget.setEnabled(enabled)
                # Apply proper disabled state styling
                color = "white" if enabled else "#808080"
                widget.setStyleSheet(f"color: {color};")
            except Exception:
                pass
    
    # Retry-After checkbox
    if hasattr(self, 'ignore_retry_after_checkbox') and self.ignore_retry_after_checkbox is not None:
        try:
            self.ignore_retry_after_checkbox.setEnabled(enabled)
        except Exception:
            pass
                            
def _reset_anti_duplicate_defaults(self):
    """Reset all anti-duplicate parameters to their default values (Qt version)"""
    from PySide6.QtWidgets import QMessageBox
    
    # Ask for confirmation
    reply = QMessageBox.question(
        None,
        "Reset Anti-Duplicate Parameters",
        "Are you sure you want to reset all anti-duplicate parameters to their default values?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )
    if reply != QMessageBox.Yes:
        return
    
    # Reset all variables to defaults
    if hasattr(self, 'enable_anti_duplicate_var'):
        self.enable_anti_duplicate_var = False
    
    if hasattr(self, 'top_p_var'):
        self.top_p_var = 1.0  # Default = no effect
    
    if hasattr(self, 'top_k_var'):
        self.top_k_var = 0  # Default = disabled
    
    if hasattr(self, 'frequency_penalty_var'):
        self.frequency_penalty_var = 0.0  # Default = no penalty
    
    if hasattr(self, 'presence_penalty_var'):
        self.presence_penalty_var = 0.0  # Default = no penalty
    
    if hasattr(self, 'repetition_penalty_var'):
        self.repetition_penalty_var = 1.0  # Default = no penalty
    
    if hasattr(self, 'candidate_count_var'):
        self.candidate_count_var = 1  # Default = single response
    
    if hasattr(self, 'custom_stop_sequences_var'):
        self.custom_stop_sequences_var = ""  # Default = empty
    
    if hasattr(self, 'logit_bias_enabled_var'):
        self.logit_bias_enabled_var = False  # Default = disabled
    
    if hasattr(self, 'logit_bias_strength_var'):
        self.logit_bias_strength_var = -0.5  # Default strength
    
    if hasattr(self, 'bias_common_words_var'):
        self.bias_common_words_var = False  # Default = disabled
    
    if hasattr(self, 'bias_repetitive_phrases_var'):
        self.bias_repetitive_phrases_var = False  # Default = disabled
    
    # Update enable/disable state
    self._toggle_anti_duplicate_controls()
    
    # Show success message
    from PySide6.QtWidgets import QMessageBox
    QMessageBox.information(None, "Reset Complete", "All anti-duplicate parameters have been reset to their default values.")
    
    # Log the reset
    if hasattr(self, 'append_log'):
        self.append_log("🔄 Anti-duplicate parameters reset to defaults")

def _create_custom_api_endpoints_section(self, parent_frame):
    """Create the Custom API Endpoints section (PySide6)"""
    from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QWidget, QLineEdit, QPushButton, QComboBox
    from PySide6.QtCore import Qt
    
    section_box = QGroupBox("Custom API Endpoints")
    section_v = QVBoxLayout(section_box)
    section_v.setContentsMargins(8, 8, 8, 8)  # Compact margins
    section_v.setSpacing(4)  # Compact spacing between widgets
    
    # Checkbox to enable/disable custom endpoint
    enable_cb = self._create_styled_checkbox("Enable Custom OpenAI Endpoint")
    try:
        enable_cb.setChecked(bool(self.use_custom_openai_endpoint_var))
    except Exception:
        pass
    def _on_enable_custom_endpoint(checked):
        try:
            self.use_custom_openai_endpoint_var = bool(checked)
            self.toggle_custom_endpoint_ui(user_interaction=True)
        except Exception:
            pass
    enable_cb.toggled.connect(_on_enable_custom_endpoint)
    section_v.addWidget(enable_cb)
    
    # Main OpenAI Base URL
    openai_row = QWidget()
    openai_h = QHBoxLayout(openai_row)
    openai_h.setContentsMargins(0, 5, 0, 5)
    openai_h.addWidget(QLabel("Override API Endpoint:"))
    
    self.openai_base_url_var = self.config.get('openai_base_url', '')
    self.openai_base_url_entry = QLineEdit()
    try:
        self.openai_base_url_entry.setText(str(self.openai_base_url_var))
    except Exception:
        pass
    def _on_openai_url_changed(text):
        try:
            self.openai_base_url_var = text
            self._check_azure_endpoint()
        except Exception:
            pass
    self.openai_base_url_entry.textChanged.connect(_on_openai_url_changed)
    openai_h.addWidget(self.openai_base_url_entry)
    
    self.openai_clear_button = QPushButton("Clear")
    self.openai_clear_button.setFixedWidth(80)
    def _clear_openai_url():
        self.openai_base_url_var = ""
        self.openai_base_url_entry.setText("")
    self.openai_clear_button.clicked.connect(_clear_openai_url)
    openai_h.addWidget(self.openai_clear_button)
    section_v.addWidget(openai_row)
    
    # Help text
    help_lbl = QLabel("Enable checkbox to use custom endpoint. For Ollama: http://localhost:11434/v1")
    help_lbl.setStyleSheet("color: gray; font-size: 8pt;")
    help_lbl.setContentsMargins(0, 0, 0, 10)
    section_v.addWidget(help_lbl)
    
    # Azure version frame (initially hidden)
    self.azure_version_frame = QWidget()
    azure_h = QHBoxLayout(self.azure_version_frame)
    azure_h.setContentsMargins(0, 0, 0, 10)
    azure_h.addWidget(QLabel("Azure API Version:"))
    
    # Azure API version combo
    try:
        self.azure_api_version_var = self.config.get('azure_api_version', '2024-08-01-preview')
    except Exception:
        pass
    versions = [
        '2025-01-01-preview', '2024-12-01-preview', '2024-10-01-preview',
        '2024-08-01-preview', '2024-06-01', '2024-05-01-preview',
        '2024-04-01-preview', '2024-02-01', '2023-12-01-preview',
        '2023-10-01-preview', '2023-05-15'
    ]
    self.azure_version_combo = QComboBox()
    self.azure_version_combo.addItems(versions)
    self.azure_version_combo.setFixedWidth(200)
    # Add custom styling with unicode arrow
    self.azure_version_combo.setStyleSheet("""
        QComboBox::down-arrow {
            image: none;
            width: 12px;
            height: 12px;
            border: none;
        }
    """)
    self._add_combobox_arrow(self.azure_version_combo)
    self._disable_combobox_mousewheel(self.azure_version_combo)
    try:
        idx = self.azure_version_combo.findText(self.azure_api_version_var)
        if idx >= 0:
            self.azure_version_combo.setCurrentIndex(idx)
    except Exception:
        pass
    def _on_azure_version_changed(text):
        try:
            self.azure_api_version_var = text
            self._update_azure_api_version_env()
        except Exception:
            pass
    self.azure_version_combo.currentTextChanged.connect(_on_azure_version_changed)
    azure_h.addWidget(self.azure_version_combo)
    azure_h.addStretch()
    self.azure_version_frame.setVisible(False)  # Initially hidden
    section_v.addWidget(self.azure_version_frame)
    
    # Show More Fields button
    self.show_more_endpoints = False
    self.more_fields_button = QPushButton("▼ Show More Fields")
    self.more_fields_button.setStyleSheet("text-align: left; border: none; color: #0dcaf0;")
    self.more_fields_button.clicked.connect(lambda: self.toggle_more_endpoints())
    section_v.addWidget(self.more_fields_button)
    
    # Add spacing after Show More Fields button to prevent accidental clicks
    section_v.addSpacing(15)
    
    # Container for additional fields (initially hidden)
    self.additional_endpoints_frame = QWidget()
    additional_v = QVBoxLayout(self.additional_endpoints_frame)
    additional_v.setContentsMargins(0, 0, 0, 0)
    self.additional_endpoints_frame.setVisible(False)  # Initially hidden
    
    # Groq/Local Base URL
    groq_row = QWidget()
    groq_h = QHBoxLayout(groq_row)
    groq_h.setContentsMargins(0, 5, 0, 5)
    groq_h.addWidget(QLabel("Groq/Local Base URL:"))
    
    self.groq_base_url_var = self.config.get('groq_base_url', '')
    self.groq_base_url_entry = QLineEdit()
    try:
        self.groq_base_url_entry.setText(str(self.groq_base_url_var))
    except Exception:
        pass
    def _on_groq_url_changed(text):
        try:
            self.groq_base_url_var = text
        except Exception:
            pass
    self.groq_base_url_entry.textChanged.connect(_on_groq_url_changed)
    groq_h.addWidget(self.groq_base_url_entry)
    
    groq_clear_btn = QPushButton("Clear")
    groq_clear_btn.setFixedWidth(80)
    def _clear_groq_url():
        self.groq_base_url_var = ""
        self.groq_base_url_entry.setText("")
    groq_clear_btn.clicked.connect(_clear_groq_url)
    groq_h.addWidget(groq_clear_btn)
    additional_v.addWidget(groq_row)
    
    groq_help = QLabel("For vLLM: http://localhost:8000/v1 | For LM Studio: http://localhost:1234/v1")
    groq_help.setStyleSheet("color: gray; font-size: 8pt;")
    groq_help.setContentsMargins(0, 0, 0, 5)
    additional_v.addWidget(groq_help)
    
    # Fireworks Base URL
    fireworks_row = QWidget()
    fireworks_h = QHBoxLayout(fireworks_row)
    fireworks_h.setContentsMargins(0, 5, 0, 5)
    fireworks_h.addWidget(QLabel("Fireworks Base URL:"))
    
    self.fireworks_base_url_var = self.config.get('fireworks_base_url', '')
    self.fireworks_base_url_entry = QLineEdit()
    try:
        self.fireworks_base_url_entry.setText(str(self.fireworks_base_url_var))
    except Exception:
        pass
    def _on_fireworks_url_changed(text):
        try:
            self.fireworks_base_url_var = text
        except Exception:
            pass
    self.fireworks_base_url_entry.textChanged.connect(_on_fireworks_url_changed)
    fireworks_h.addWidget(self.fireworks_base_url_entry)
    
    fireworks_clear_btn = QPushButton("Clear")
    fireworks_clear_btn.setFixedWidth(80)
    def _clear_fireworks_url():
        self.fireworks_base_url_var = ""
        self.fireworks_base_url_entry.setText("")
    fireworks_clear_btn.clicked.connect(_clear_fireworks_url)
    fireworks_h.addWidget(fireworks_clear_btn)
    additional_v.addWidget(fireworks_row)
    
    # Info about multiple endpoints
    info_lbl = QLabel("💡 Advanced: Use multiple endpoints to run different local LLM servers simultaneously.\n• Use model prefix 'groq/' to route through Groq endpoint\n• Use model prefix 'fireworks/' to route through Fireworks endpoint\n• Most users only need the main OpenAI endpoint above")
    info_lbl.setStyleSheet("color: #0dcaf0; font-size: 8pt;")
    info_lbl.setWordWrap(True)
    info_lbl.setContentsMargins(0, 10, 0, 10)
    additional_v.addWidget(info_lbl)
    
    # Gemini OpenAI-Compatible Endpoint
    gemini_cb = self._create_styled_checkbox("Enable Gemini OpenAI-Compatible Endpoint")
    try:
        gemini_cb.setChecked(bool(self.use_gemini_openai_endpoint_var))
    except Exception:
        pass
    def _on_gemini_endpoint_toggle(checked):
        try:
            self.use_gemini_openai_endpoint_var = bool(checked)
            self.toggle_gemini_endpoint()
        except Exception:
            pass
    gemini_cb.toggled.connect(_on_gemini_endpoint_toggle)
    gemini_cb.setContentsMargins(0, 5, 0, 5)
    additional_v.addWidget(gemini_cb)
    
    # Gemini endpoint URL input
    gemini_row = QWidget()
    gemini_h = QHBoxLayout(gemini_row)
    gemini_h.setContentsMargins(0, 5, 0, 5)
    gemini_h.addWidget(QLabel("Gemini OpenAI Endpoint:"))
    
    self.gemini_endpoint_entry = QLineEdit()
    try:
        self.gemini_endpoint_entry.setText(str(self.gemini_openai_endpoint_var))
    except Exception:
        pass
    def _on_gemini_url_changed(text):
        try:
            self.gemini_openai_endpoint_var = text
        except Exception:
            pass
    self.gemini_endpoint_entry.textChanged.connect(_on_gemini_url_changed)
    gemini_h.addWidget(self.gemini_endpoint_entry)
    
    self.gemini_clear_button = QPushButton("Clear")
    self.gemini_clear_button.setFixedWidth(80)
    def _clear_gemini_url():
        self.gemini_openai_endpoint_var = ""
        self.gemini_endpoint_entry.setText("")
    self.gemini_clear_button.clicked.connect(_clear_gemini_url)
    gemini_h.addWidget(self.gemini_clear_button)
    additional_v.addWidget(gemini_row)
    
    gemini_help = QLabel("For Gemini rate limit optimization with proxy services (e.g., OpenRouter, LiteLLM)")
    gemini_help.setStyleSheet("color: gray; font-size: 8pt;")
    gemini_help.setContentsMargins(0, 0, 0, 5)
    additional_v.addWidget(gemini_help)
    
    # Add the additional endpoints frame to the main section
    section_v.addWidget(self.additional_endpoints_frame)
    
    # Test Connection button
    test_btn = QPushButton("Test Connection")
    test_btn.clicked.connect(lambda: self.test_api_connections())
    section_v.addWidget(test_btn)
    
    # Place the section at row 7, spanning both columns
    try:
        grid = parent_frame.layout()
        if grid:
            grid.addWidget(section_box, 7, 0, 1, 2)
    except Exception:
        # Fallback: just stack
        section_box.setParent(parent_frame)
    
    # Set initial states
    self.toggle_custom_endpoint_ui()
    self.toggle_gemini_endpoint()

def _check_azure_endpoint(self, *args):
    """Check if endpoint is Azure and update UI (Qt version)"""
    try:
        if not self.use_custom_openai_endpoint_var:
            if hasattr(self, 'azure_version_frame'):
                self.azure_version_frame.setVisible(False)
            return
            
        url = self.openai_base_url_var
        if '.azure.com' in url or '.cognitiveservices' in url:
            if hasattr(self, 'api_key_label'):
                try:
                    self.api_key_label.setText("Azure Key:")
                except Exception:
                    pass
            
            # Show Azure version frame
            if hasattr(self, 'azure_version_frame'):
                self.azure_version_frame.setVisible(True)
        else:
            if hasattr(self, 'api_key_label'):
                try:
                    self.api_key_label.setText("OpenAI/Gemini/... API Key:")
                except Exception:
                    pass
            
            # Hide Azure version frame
            if hasattr(self, 'azure_version_frame'):
                self.azure_version_frame.setVisible(False)
    except Exception:
        pass
            
def _update_azure_api_version_env(self, *args):
    """Update the AZURE_API_VERSION environment variable when the setting changes"""
    try:
        api_version = self.azure_api_version_var
        if api_version:
            os.environ['AZURE_API_VERSION'] = api_version
            #print(f"✅ Updated Azure API Version in environment: {api_version}")
    except Exception as e:
        print(f"❌ Error updating Azure API Version environment variable: {e}")

def toggle_gemini_endpoint(self):
    """Enable/disable Gemini endpoint entry based on toggle (Qt version)"""
    try:
        enabled = bool(self.use_gemini_openai_endpoint_var)
        if hasattr(self, 'gemini_endpoint_entry'):
            self.gemini_endpoint_entry.setEnabled(enabled)
        if hasattr(self, 'gemini_clear_button'):
            self.gemini_clear_button.setEnabled(enabled)
    except Exception:
        pass
    
def toggle_custom_endpoint_ui(self, user_interaction=False):
    """Enable/disable the OpenAI base URL entry and detect Azure (Qt version)"""
    try:
        enabled = bool(self.use_custom_openai_endpoint_var)
        
        if hasattr(self, 'openai_base_url_entry'):
            self.openai_base_url_entry.setEnabled(enabled)
        if hasattr(self, 'openai_clear_button'):
            self.openai_clear_button.setEnabled(enabled)
        
        if enabled:
            # Check if it's Azure
            url = self.openai_base_url_var
            if '.azure.com' in url or '.cognitiveservices' in url:
                if hasattr(self, 'api_key_label'):
                    try:
                        self.api_key_label.setText("Azure Key:")
                    except Exception:
                        pass
            else:
                if hasattr(self, 'api_key_label'):
                    try:
                        self.api_key_label.setText("OpenAI/Gemini/... API Key:")
                    except Exception:
                        pass
            # Only print when user actually interacts with the toggle
            if user_interaction:
                print("✅ Custom OpenAI endpoint enabled")
        else:
            if hasattr(self, 'api_key_label'):
                try:
                    self.api_key_label.setText("OpenAI/Gemini/... API Key:")
                except Exception:
                    pass
            # Only print when user actually interacts with the toggle
            if user_interaction:
                print("❌ Custom OpenAI endpoint disabled - using default OpenAI API")
    except Exception:
        pass

def toggle_more_endpoints(self):
    """Toggle visibility of additional endpoint fields (Qt version)"""
    try:
        self.show_more_endpoints = not self.show_more_endpoints
        
        if hasattr(self, 'additional_endpoints_frame'):
            self.additional_endpoints_frame.setVisible(self.show_more_endpoints)
        
        if hasattr(self, 'more_fields_button'):
            if self.show_more_endpoints:
                self.more_fields_button.setText("▲ Show Fewer Fields")
            else:
                self.more_fields_button.setText("▼ Show More Fields")
    except Exception:
        pass
             
def test_api_connections(self):
    """Test all configured API connections (Qt version)"""
    from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QMessageBox
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon
    
    # Show immediate feedback
    progress_dialog = QDialog(self.current_dialog if hasattr(self, 'current_dialog') else None)
    progress_dialog.setWindowTitle("Testing Connections...")
    # Use screen ratios for sizing
    from PySide6.QtWidgets import QApplication
    screen = QApplication.primaryScreen().geometry()
    width = int(screen.width() * 0.16)  # 16% of screen width
    height = int(screen.height() * 0.14)  # 14% of screen height
    progress_dialog.setFixedSize(width, height)
    
    # Set icon
    try:
        progress_dialog.setWindowIcon(QIcon("halgakos.ico"))
    except:
        pass
    
    # Center the dialog
    if progress_dialog.parent():
        parent_geo = progress_dialog.parent().geometry()
        x = parent_geo.x() + (parent_geo.width() - 300) // 2
        y = parent_geo.y() + (parent_geo.height() - 150) // 2
        progress_dialog.move(x, y)
    
    # Add progress message
    layout = QVBoxLayout(progress_dialog)
    progress_label = QLabel("Testing API connections...\nPlease wait...")
    progress_label.setAlignment(Qt.AlignCenter)
    progress_label.setStyleSheet("font-size: 10pt;")
    layout.addWidget(progress_label)
    
    # Show dialog non-modally so it's visible
    progress_dialog.show()
    progress_dialog.repaint()
    
    try:
        # Ensure we have the openai module
        import openai
    except ImportError:
        progress_dialog.close()
        QMessageBox.critical(None, "Error", "OpenAI library not installed")
        return
    
    # Get API key from the main GUI  
    api_key = ''
    if hasattr(self, 'api_key_entry'):
        if hasattr(self.api_key_entry, 'text'):  # PySide6 QLineEdit
            api_key = self.api_key_entry.text()
        elif hasattr(self.api_key_entry, 'get'):  # Tkinter Entry
            api_key = self.api_key_entry.get()
    if not api_key:
        api_key = self.config.get('api_key', '')
    if not api_key:
        api_key = "sk-dummy-key"  # For local models
    
    # Collect all configured endpoints
    endpoints_to_test = []
    
    # OpenAI endpoint - only test if checkbox is enabled
    if self.use_custom_openai_endpoint_var:
        openai_url = self.openai_base_url_var
        if openai_url:
            # Check if it's Azure
            if '.azure.com' in openai_url or '.cognitiveservices' in openai_url:
                # Azure endpoint
                deployment = self.model_var if hasattr(self, 'model_var') else "gpt-35-turbo"
                api_version = self.azure_api_version_var if hasattr(self, 'azure_api_version_var') else "2024-08-01-preview"
                
                # Format Azure URL
                if '/openai/deployments/' not in openai_url:
                    azure_url = f"{openai_url.rstrip('/')}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
                else:
                    azure_url = openai_url
                
                endpoints_to_test.append(("Azure OpenAI", azure_url, deployment, "azure"))
            else:
                # Regular custom endpoint
                endpoints_to_test.append(("OpenAI (Custom)", openai_url, self.model_var if hasattr(self, 'model_var') else "gpt-3.5-turbo"))
        else:
            # Use default OpenAI endpoint if checkbox is on but no custom URL provided
            endpoints_to_test.append(("OpenAI (Default)", "https://api.openai.com/v1", self.model_var if hasattr(self, 'model_var') else "gpt-3.5-turbo"))
    
    # Groq endpoint
    if hasattr(self, 'groq_base_url_var'):
        groq_url = self.groq_base_url_var
        if groq_url:
            # For Groq, we need a groq-prefixed model
            current_model = self.model_var if hasattr(self, 'model_var') else "llama-3-70b"
            groq_model = current_model if current_model.startswith('groq/') else current_model.replace('groq/', '')
            endpoints_to_test.append(("Groq/Local", groq_url, groq_model))
    
    # Fireworks endpoint
    if hasattr(self, 'fireworks_base_url_var'):
        fireworks_url = self.fireworks_base_url_var
        if fireworks_url:
            # For Fireworks, we need the accounts/ prefix
            current_model = self.model_var if hasattr(self, 'model_var') else "llama-v3-70b-instruct"
            fw_model = current_model if current_model.startswith('accounts/') else f"accounts/fireworks/models/{current_model.replace('fireworks/', '')}"
            endpoints_to_test.append(("Fireworks", fireworks_url, fw_model))
    
    # Gemini OpenAI-Compatible endpoint
    if hasattr(self, 'use_gemini_openai_endpoint_var') and self.use_gemini_openai_endpoint_var:
        gemini_url = self.gemini_openai_endpoint_var
        if gemini_url:
            # Ensure the endpoint ends with /openai/ for compatibility
            if not gemini_url.endswith('/openai/'):
                if gemini_url.endswith('/'):
                    gemini_url = gemini_url + 'openai/'
                else:
                    gemini_url = gemini_url + '/openai/'
            
            # For Gemini OpenAI-compatible endpoints, use the current model or a suitable default
            current_model = self.model_var if hasattr(self, 'model_var') else "gemini-2.0-flash-exp"
            # Remove any 'gemini/' prefix for the OpenAI-compatible endpoint
            gemini_model = current_model.replace('gemini/', '') if current_model.startswith('gemini/') else current_model
            endpoints_to_test.append(("Gemini (OpenAI-Compatible)", gemini_url, gemini_model))
    
    if not endpoints_to_test:
        QMessageBox.information(None, "Info", "No custom endpoints configured. Using default API endpoints.")
        return
    
    # Test each endpoint
    # Test each endpoint
    results = []
    for endpoint_info in endpoints_to_test:
        if len(endpoint_info) == 4 and endpoint_info[3] == "azure":
            # Azure endpoint
            name, base_url, model, endpoint_type = endpoint_info
            try:
                # Azure uses different headers
                import requests
                headers = {
                    "api-key": api_key,
                    "Content-Type": "application/json"
                }
                
                response = requests.post(
                    base_url,
                    headers=headers,
                    json={
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 5
                    },
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    results.append(f"✅ {name}: Connected successfully! (Deployment: {model})")
                else:
                    results.append(f"❌ {name}: {response.status_code} - {response.text[:100]}")
                    
            except Exception as e:
                error_msg = str(e)[:100]
                results.append(f"❌ {name}: {error_msg}")
        else:
            # Regular OpenAI-compatible endpoint
            name, base_url, model = endpoint_info[:3]
            try:
                # Create client for this endpoint
                test_client = openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=5.0  # Short timeout for testing
                )
                
                # Try a minimal completion
                response = test_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5
                )
                
                results.append(f"✅ {name}: Connected successfully! (Model: {model})")
            except Exception as e:
                error_msg = str(e)
                # Simplify common error messages
                if "404" in error_msg:
                    error_msg = "404 - Endpoint not found. Check URL and model name."
                elif "401" in error_msg or "403" in error_msg:
                    error_msg = "Authentication failed. Check API key."
                elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                    error_msg = f"Model '{model}' not found at this endpoint."
                
                results.append(f"❌ {name}: {error_msg}")
    
    # Show results
    result_message = "Connection Test Results:\n\n" + "\n\n".join(results)
    
    # Close progress dialog
    progress_dialog.close()
    
    # Determine if all succeeded
    all_success = all("✅" in r for r in results)
    
    if all_success:
        QMessageBox.information(None, "Success", result_message)
    else:
        QMessageBox.warning(None, "Test Results", result_message)
    

def run_standalone_translate_headers(self):
    """Run standalone header translation in a background thread"""
    from PySide6.QtWidgets import QMessageBox
    from PySide6.QtGui import QIcon
    import traceback
    import threading
    
    # Get icon path
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Halgakos.ico")
    icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
    
    try:
        # Get API key and model from GUI fields
        api_key = self.api_key_entry.text().strip() if hasattr(self, 'api_key_entry') else ""
        model = self.model_var.strip() if hasattr(self, 'model_var') else ""
        
        if not api_key:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Error")
            msg_box.setText("API key not configured.")
            msg_box.setInformativeText("Please enter your API key in the main window before using this feature.")
            msg_box.setWindowIcon(icon)
            _center_messagebox_buttons(msg_box)
            msg_box.exec()
            return
        
        if not model:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Error")
            msg_box.setText("Model not selected.")
            msg_box.setInformativeText("Please select a model in the main window before using this feature.")
            msg_box.setWindowIcon(icon)
            _center_messagebox_buttons(msg_box)
            msg_box.exec()
            return
        
        # Initialize stop flag and running state
        self._headers_stop_requested = False
        self._headers_translation_running = True
        
        # Transform button to stop mode (red)
        if hasattr(self, 'translate_headers_btn'):
            self.translate_headers_btn.setText("⏹ Stop Headers")
            self.translate_headers_btn.setStyleSheet(
                "QPushButton { background-color: #dc3545; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold; } "
                "QPushButton:hover { background-color: #c82333; } "
                "QPushButton:disabled { background-color: #e0e0e0; color: #9e9e9e; }"
            )
            
        # Start spinning animation on icon
        if hasattr(self, 'translate_icon_spin_animation') and hasattr(self, 'translate_headers_icon'):
            try:
                from PySide6.QtCore import QTimer, QPropertyAnimation
                
                # Start the continuous spin animation
                if self.translate_icon_spin_animation.state() != QPropertyAnimation.Running:
                    self.translate_icon_spin_animation.start()
                
                # Update button icon periodically and monitor for completion
                def update_and_monitor():
                    # Check if still running
                    if hasattr(self, '_headers_translation_running') and self._headers_translation_running:
                        # Update spinning icon
                        if hasattr(self, 'translate_headers_icon') and self.translate_headers_icon.pixmap():
                            self.translate_headers_btn.setIcon(QIcon(self.translate_headers_icon.pixmap()))
                        # Continue monitoring
                        QTimer.singleShot(50, update_and_monitor)
                    else:
                        # Translation finished - stop animation gracefully
                        if hasattr(self, 'translate_icon_spin_animation') and hasattr(self, 'translate_icon_stop_animation'):
                            if self.translate_icon_spin_animation.state() == QPropertyAnimation.Running:
                                # Stop the infinite spin
                                self.translate_icon_spin_animation.stop()
                                
                                # Get current rotation and smoothly decelerate to 0
                                if hasattr(self, 'translate_headers_icon'):
                                    current_rotation = self.translate_headers_icon.get_rotation()
                                    current_rotation = current_rotation % 360
                                    
                                    # Determine shortest path to 0
                                    if current_rotation > 180:
                                        target_rotation = 360
                                    else:
                                        target_rotation = 0
                                    
                                    self.translate_icon_stop_animation.setStartValue(current_rotation)
                                    self.translate_icon_stop_animation.setEndValue(target_rotation)
                                    self.translate_icon_stop_animation.start()
                        
                        # Wait for deceleration to finish before resetting button
                        def reset_button():
                            if hasattr(self, 'translate_headers_btn'):
                                self.translate_headers_btn.setText("Translate Headers Now")
                                self.translate_headers_btn.setStyleSheet(
                                    "QPushButton { background-color: #6c757d; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold; } "
                                    "QPushButton:hover { background-color: #28a745; } "
                                    "QPushButton:disabled { background-color: #e0e0e0; color: #9e9e9e; }"
                                )
                                # Reset icon to static position
                                if hasattr(self, 'translate_headers_icon'):
                                    self.translate_headers_icon.set_rotation(0)
                                    if self.translate_headers_icon._original_pixmap:
                                        self.translate_headers_btn.setIcon(QIcon(self.translate_headers_icon._original_pixmap))
                        
                        # Delay reset to allow deceleration animation to finish
                        QTimer.singleShot(900, reset_button)
                
                update_and_monitor()
            except Exception as e:
                pass
        
        # Log that translation is starting
        self.append_log("🌐 Starting standalone header translation in background...")
        
        # Define the thread function
        def translation_thread():
            try:
                # Use existing API client if available (it has multi-key support)
                # Otherwise create a new one with proper config
                original_client = getattr(self, 'api_client', None)
                
                if original_client:
                    # Use existing client - it already has multi-key mode configured
                    self.append_log("✅ Using existing API client (with multi-key support)")
                    api_client = original_client
                else:
                    # Initialize new API client with current settings
                    # Multi-key mode is configured via environment variables (not constructor params)
                    from unified_api_client import UnifiedClient
                    import json
                    
                    # Set environment variables for multi-key mode if configured
                    if hasattr(self, 'config'):
                        if self.config.get('use_multi_api_keys', False):
                            multi_keys = self.config.get('multi_api_keys', [])
                            os.environ['USE_MULTI_API_KEYS'] = '1'
                            os.environ['MULTI_API_KEYS'] = json.dumps(multi_keys)
                            os.environ['FORCE_KEY_ROTATION'] = '1' if self.config.get('force_key_rotation', True) else '0'
                            os.environ['ROTATION_FREQUENCY'] = str(self.config.get('rotation_frequency', 1))
                            self.append_log(f"🔑 Multi-key mode enabled ({len(multi_keys)} keys)")
                    
                    api_client = UnifiedClient(
                        model=model, 
                        api_key=api_key
                    )
                    self.append_log("✅ Created new API client")
                    
                    # Set it temporarily
                    self.api_client = api_client
                
                try:
                    # Import and run the translation GUI
                    from translate_headers_standalone import run_translate_headers_gui
                    run_translate_headers_gui(self)
                    
                    # After translation completes, run EPUB converter to rebuild the EPUB
                    # with the updated HTML files
                    self.append_log("\n📦 Rebuilding EPUB with translated headers...")
                    try:
                        from epub_converter import fallback_compile_epub
                        
                        # Find the output directory for the current EPUB
                        epub_path = self.get_current_epub_path() if hasattr(self, 'get_current_epub_path') else None
                        if not epub_path and hasattr(self, 'selected_files') and self.selected_files:
                            # Get first EPUB from selection
                            epub_files = [f for f in self.selected_files if f.lower().endswith('.epub')]
                            if epub_files:
                                epub_path = epub_files[0]
                        
                        if epub_path:
                            epub_base = os.path.splitext(os.path.basename(epub_path))[0]
                            current_dir = os.getcwd()
                            script_dir = os.path.dirname(os.path.abspath(__file__))
                            
                            # Find output directory (same logic as header translation)
                            candidates = [
                                os.path.join(current_dir, epub_base),
                                os.path.join(script_dir, epub_base),
                                os.path.join(current_dir, 'src', epub_base),
                            ]
                            
                            output_dir = None
                            for candidate in candidates:
                                if os.path.isdir(candidate):
                                    files = os.listdir(candidate)
                                    html_files = [f for f in files if f.lower().endswith(('.html', '.xhtml', '.htm'))]
                                    if html_files:
                                        output_dir = candidate
                                        break
                            
                            if output_dir:
                                # Set EPUB_PATH env var for the converter
                                os.environ['EPUB_PATH'] = epub_path
                                
                                self.append_log(f"📂 Output directory: {output_dir}")
                                fallback_compile_epub(output_dir, log_callback=self.append_log)
                                self.append_log("✅ EPUB rebuilt successfully with translated headers!")
                            else:
                                self.append_log("⚠️ Could not find output directory to rebuild EPUB")
                        else:
                            self.append_log("⚠️ No EPUB file selected - skipping EPUB rebuild")
                    except Exception as epub_error:
                        self.append_log(f"⚠️ Failed to rebuild EPUB: {epub_error}")
                        import traceback as tb
                        self.append_log(tb.format_exc())
                finally:
                    # Restore original client
                    if original_client is not None:
                        self.api_client = original_client
                    elif hasattr(self, 'api_client'):
                        delattr(self, 'api_client')
                
            except Exception as e:
                error_msg = f"Failed to run standalone header translation: {e}\n\n{traceback.format_exc()}"
                self.append_log(f"❌ {error_msg}")
            finally:
                # Reset button to initial state when thread completes
                # Just set flags - the monitoring timer will handle UI updates
                self._headers_translation_running = False
                self._headers_stop_requested = True  # Stop spinning animation
        
        # Start the thread
        thread = threading.Thread(target=translation_thread, daemon=True, name="HeaderTranslationThread")
        self._headers_thread = thread  # Store reference for stop button
        thread.start()
        
    except Exception as e:
        error_msg = f"Failed to start standalone header translation: {e}\n\n{traceback.format_exc()}"
        self.append_log(f"❌ {error_msg}")
        
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(f"Failed to start standalone header translation: {e}")
        msg_box.setDetailedText(traceback.format_exc())
        msg_box.setWindowIcon(icon)
        _center_messagebox_buttons(msg_box)
        msg_box.exec()

def delete_translated_headers_file(self):
    """Delete the translated_headers.txt file from the output directory for all selected EPUBs"""
    from PySide6.QtWidgets import QMessageBox
    
    # Get icon path
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Halgakos.ico")
    icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
    
    try:
        # Get all selected EPUB files using the same logic as QA scanner
        epub_files_to_process = []
        
        # First check if current selection actually contains EPUBs
        if hasattr(self, 'selected_files') and self.selected_files:
            current_epub_files = [f for f in self.selected_files if f.lower().endswith('.epub')]
            if current_epub_files:
                epub_files_to_process = current_epub_files
                self.append_log(f"📚 Found {len(epub_files_to_process)} EPUB files in current selection")
        
        # If no EPUBs in selection, try single EPUB methods
        if not epub_files_to_process:
            epub_path = self.get_current_epub_path()
            if not epub_path:
                entry_path = ''
                if hasattr(self, 'entry_epub'):
                    if hasattr(self.entry_epub, 'text'):  # PySide6 QLineEdit
                        entry_path = self.entry_epub.text().strip()
                    elif hasattr(self.entry_epub, 'get'):  # Tkinter Entry
                        entry_path = self.entry_epub.get().strip()
                if entry_path and entry_path != "No file selected" and os.path.exists(entry_path):
                    epub_path = entry_path
            
            if epub_path:
                epub_files_to_process = [epub_path]
        
        if not epub_files_to_process:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Error")
            msg_box.setText("No EPUB file(s) selected. Please select EPUB file(s) first.")
            msg_box.setWindowIcon(icon)
            _center_messagebox_buttons(msg_box)
            msg_box.exec()
            return
        
        # Process each EPUB file to find and delete translated_headers.txt
        files_found = []
        files_not_found = []
        files_deleted = []
        errors = []
        
        current_dir = os.getcwd()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # First pass: scan for files
        for epub_path in epub_files_to_process:
            try:
                epub_base = os.path.splitext(os.path.basename(epub_path))[0]
                self.append_log(f"🔍 Processing EPUB: {epub_base}")
                
                # Check the most common locations in order of priority (same as QA scanner)
                candidates = [
                    os.path.join(current_dir, epub_base),        # current working directory
                    os.path.join(script_dir, epub_base),         # src directory (where output typically goes)
                    os.path.join(current_dir, 'src', epub_base), # src subdirectory from current dir
                ]
                
                output_dir = None
                for candidate in candidates:
                    if os.path.isdir(candidate):
                        # Verify the folder actually contains HTML/XHTML files
                        try:
                            files = os.listdir(candidate)
                            html_files = [f for f in files if f.lower().endswith(('.html', '.xhtml', '.htm'))]
                            if html_files:
                                output_dir = candidate
                                break
                        except Exception:
                            continue
                
                if not output_dir:
                    self.append_log(f"  ⚠️ No output directory found for {epub_base}")
                    files_not_found.append((epub_base, "No output directory found"))
                    continue
                
                # Look for translated_headers.txt in the output directory
                headers_file = os.path.join(output_dir, "translated_headers.txt")
                
                if os.path.exists(headers_file):
                    files_found.append((epub_base, headers_file))
                    self.append_log(f"  ✓ Found translated_headers.txt in {os.path.basename(output_dir)}")
                else:
                    files_not_found.append((epub_base, "translated_headers.txt not found"))
                    self.append_log(f"  ⚠️ No translated_headers.txt in {os.path.basename(output_dir)}")
                    
            except Exception as e:
                epub_base = os.path.splitext(os.path.basename(epub_path))[0]
                errors.append((epub_base, str(e)))
                self.append_log(f"  ❌ Error processing {epub_base}: {e}")
        
        # Show summary and get user confirmation
        if not files_found and not files_not_found and not errors:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("No Files")
            msg_box.setText("No EPUB files were processed.")
            msg_box.setWindowIcon(icon)
            _center_messagebox_buttons(msg_box)
            msg_box.exec()
            return
        
        summary_text = f"Summary for {len(epub_files_to_process)} EPUB file(s):\n\n"
        
        if files_found:
            summary_text += f"✅ Files to delete ({len(files_found)}):\n"
            for epub_base, file_path in files_found:
                summary_text += f"  • {epub_base}\n"
            summary_text += "\n"
        
        if files_not_found:
            summary_text += f"⚠️ Files not found ({len(files_not_found)}):\n"
            for epub_base, reason in files_not_found:
                summary_text += f"  • {epub_base}: {reason}\n"
            summary_text += "\n"
        
        if errors:
            summary_text += f"❌ Errors ({len(errors)}):\n"
            for epub_base, error in errors:
                summary_text += f"  • {epub_base}: {error}\n"
            summary_text += "\n"
        
        if files_found:
            summary_text += "This will allow headers to be re-translated on the next run."
            
            # Confirm deletion
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setWindowTitle("Confirm Deletion")
            msg_box.setText(summary_text)
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.No)
            msg_box.setWindowIcon(icon)
            _center_messagebox_buttons(msg_box)
            result = msg_box.exec()
            
            if result == QMessageBox.Yes:
                # Delete the files
                for epub_base, headers_file in files_found:
                    try:
                        os.remove(headers_file)
                        files_deleted.append(epub_base)
                        self.append_log(f"✅ Deleted translated_headers.txt from {epub_base}")
                    except Exception as e:
                        errors.append((epub_base, f"Delete failed: {e}"))
                        self.append_log(f"❌ Failed to delete translated_headers.txt from {epub_base}: {e}")
                
                # Show final results
                if files_deleted:
                    success_msg = f"Successfully deleted {len(files_deleted)} file(s):\n"
                    success_msg += "\n".join([f"• {epub_base}" for epub_base in files_deleted])
                    if errors:
                        success_msg += f"\n\nErrors: {len(errors)} file(s) failed to delete."
                    msg_box = QMessageBox()
                    msg_box.setIcon(QMessageBox.Information)
                    msg_box.setWindowTitle("Success")
                    msg_box.setText(success_msg)
                    msg_box.setWindowIcon(icon)
                    _center_messagebox_buttons(msg_box)
                    msg_box.exec()
                else:
                    msg_box = QMessageBox()
                    msg_box.setIcon(QMessageBox.Critical)
                    msg_box.setWindowTitle("Error")
                    msg_box.setText("No files were successfully deleted.")
                    msg_box.setWindowIcon(icon)
                    _center_messagebox_buttons(msg_box)
                    msg_box.exec()
        else:
            # No files to delete
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("No Files to Delete")
            msg_box.setText(summary_text)
            msg_box.setWindowIcon(icon)
            _center_messagebox_buttons(msg_box)
            msg_box.exec()
        
    except Exception as e:
        self.append_log(f"❌ Error deleting translated_headers.txt: {e}")
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(f"Failed to delete file: {e}")
        msg_box.setWindowIcon(icon)
        _center_messagebox_buttons(msg_box)
        msg_box.exec()

def validate_epub_structure_gui(self):
    """GUI wrapper for EPUB structure validation"""
    from PySide6.QtWidgets import QMessageBox
    from PySide6.QtGui import QIcon
    import os
    
    # Get icon path
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "halgakos.ico")
    icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
    
    input_path = ''
    if hasattr(self, 'entry_epub'):
        if hasattr(self.entry_epub, 'text'):  # PySide6 QLineEdit
            input_path = self.entry_epub.text()
        elif hasattr(self.entry_epub, 'get'):  # Tkinter Entry
            input_path = self.entry_epub.get()
    if not input_path:
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText("Please select a file first.")
        msg_box.setWindowIcon(icon)
        _center_messagebox_buttons(msg_box)
        msg_box.exec()
        return
    
    if input_path.lower().endswith('.txt'):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Info")
        msg_box.setText("Structure validation is only available for EPUB files.")
        msg_box.setWindowIcon(icon)
        _center_messagebox_buttons(msg_box)
        msg_box.exec()
        return
    
    epub_base = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = epub_base
    
    if not os.path.exists(output_dir):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Info")
        msg_box.setText(f"No output directory found: {output_dir}")
        msg_box.setWindowIcon(icon)
        _center_messagebox_buttons(msg_box)
        msg_box.exec()
        return
    
    self.append_log("🔍 Validating EPUB structure...")
    
    try:
        from TransateKRtoEN import validate_epub_structure, check_epub_readiness
        
        structure_ok = validate_epub_structure(output_dir)
        readiness_ok = check_epub_readiness(output_dir)
        
        if structure_ok and readiness_ok:
            self.append_log("✅ EPUB validation PASSED - Ready for compilation!")
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("Validation Passed")
            msg_box.setText("✅ All EPUB structure files are present!\n\n"
                              "Your translation is ready for EPUB compilation.")
            msg_box.setWindowIcon(icon)
            _center_messagebox_buttons(msg_box)
            msg_box.exec()
        elif structure_ok:
            self.append_log("⚠️ EPUB structure OK, but some issues found")
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Validation Warning")
            msg_box.setText("⚠️ EPUB structure is mostly OK, but some issues were found.\n\n"
                                 "Check the log for details.")
            msg_box.setWindowIcon(icon)
            _center_messagebox_buttons(msg_box)
            msg_box.exec()
        else:
            self.append_log("❌ EPUB validation FAILED - Missing critical files")
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Validation Failed")
            msg_box.setText("❌ Missing critical EPUB files!\n\n"
                               "container.xml and/or OPF files are missing.\n"
                               "Try re-running the translation to extract them.")
            msg_box.setWindowIcon(icon)
            _center_messagebox_buttons(msg_box)
            msg_box.exec()
    
    except ImportError as e:
        self.append_log(f"❌ Could not import validation functions: {e}")
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText("Validation functions not available.")
        msg_box.setWindowIcon(icon)
        _center_messagebox_buttons(msg_box)
        msg_box.exec()
    except Exception as e:
        self.append_log(f"❌ Validation error: {e}")
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(f"Validation failed: {e}")
        msg_box.setWindowIcon(icon)
        _center_messagebox_buttons(msg_box)
        msg_box.exec()

def on_profile_select(self, event=None):
    """Load the selected profile's prompt into the text area."""
    # Get the current profile name from the combobox
    name = self.profile_menu.currentText() if hasattr(self, 'profile_menu') else self.profile_var
    
    # Skip if the name is empty or whitespace only
    if not name or not name.strip():
        return
    
    # Only update if the profile actually exists in prompt_profiles
    # This prevents switching to non-existent profiles while typing
    if name in self.prompt_profiles:
        # When switching profiles, revert any unsaved changes by loading from original content
        if not hasattr(self, '_original_profile_content'):
            self._original_profile_content = {}
        
        # If this profile hasn't been saved yet, store its original content
        if name not in self._original_profile_content:
            self._original_profile_content[name] = self.prompt_profiles.get(name, "")
        
        # Load the original (last saved) content, not the in-memory staged edits
        prompt = self._original_profile_content.get(name, "")
        current_text = self.prompt_text.toPlainText().strip()
        if current_text != prompt.strip():
            # PySide6: Clear and set QTextEdit content
            self.prompt_text.clear()
            self.prompt_text.setPlainText(prompt)
        
        # Also revert the in-memory profile to original content
        self.prompt_profiles[name] = prompt
        
        # Update profile_var to match only when profile exists
        self.profile_var = name
        self.config['active_profile'] = name
        
        # Set this as the active profile for autosave
        self._active_profile_for_autosave = name

def save_profile(self):
    """Save current prompt under selected profile and persist."""
    from PySide6.QtWidgets import QMessageBox
    
    # Get name from combobox or profile_var
    name = self.profile_menu.currentText().strip() if hasattr(self, 'profile_menu') else self.profile_var.strip()
    
    if not name:
        QMessageBox.critical(None, "Error", "Profile cannot be empty.")
        return
    
    # PySide6: Get text from QTextEdit
    content = self.prompt_text.toPlainText().strip()
    
    self.prompt_profiles[name] = content
    self.config['prompt_profiles'] = self.prompt_profiles
    self.config['active_profile'] = name
    
    # Update the original content to match the saved content
    if not hasattr(self, '_original_profile_content'):
        self._original_profile_content = {}
    self._original_profile_content[name] = content
    
    # Update combobox items only if the profile is new
    current_items = [self.profile_menu.itemText(i) for i in range(self.profile_menu.count())]
    if name not in current_items:
        # Only rebuild if it's a new profile
        self.profile_menu.addItem(name)
    
    # Ensure the current selection is set to the saved profile
    self.profile_menu.setCurrentText(name)
    
    # Show save confirmation with Halgakos icon
    from PySide6.QtGui import QIcon
    msg_box = QMessageBox()
    msg_box.setWindowTitle("Saved")
    msg_box.setText(f"Profile '{name}' saved.")
    msg_box.setIcon(QMessageBox.Information)
    msg_box.setStandardButtons(QMessageBox.Ok)
    try:
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Halgakos.ico")
        if os.path.exists(icon_path):
            msg_box.setWindowIcon(QIcon(icon_path))
    except:
        pass
    _center_messagebox_buttons(msg_box)
    msg_box.exec()
    self.save_profiles()

def delete_profile(self):
    """Delete the selected profile."""
    from PySide6.QtWidgets import QMessageBox
    
    # Get name from combobox or profile_var
    name = self.profile_menu.currentText() if hasattr(self, 'profile_menu') else self.profile_var
    
    if name not in self.prompt_profiles:
        QMessageBox.critical(None, "Error", f"Profile '{name}' not found.")
        return
    
    # Show delete confirmation with Halgakos icon
    from PySide6.QtGui import QIcon
    msg_box = QMessageBox()
    msg_box.setWindowTitle("Delete")
    msg_box.setText(f"Are you sure you want to delete language '{name}'?")
    msg_box.setIcon(QMessageBox.Question)
    msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    msg_box.setDefaultButton(QMessageBox.No)
    try:
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Halgakos.ico")
        if os.path.exists(icon_path):
            msg_box.setWindowIcon(QIcon(icon_path))
    except:
        pass
    _center_messagebox_buttons(msg_box)
    result = msg_box.exec()
    
    if result == QMessageBox.Yes:
        del self.prompt_profiles[name]
        self.config['prompt_profiles'] = self.prompt_profiles
        
        if self.prompt_profiles:
            new = next(iter(self.prompt_profiles))
            self.profile_var = new
            
            # Update combobox
            self.profile_menu.clear()
            self.profile_menu.addItems(list(self.prompt_profiles.keys()))
            self.profile_menu.setCurrentText(new)
            
            self.on_profile_select()
        else:
            self.profile_var = ""
            
            # Clear combobox and text
            self.profile_menu.clear()
            self.prompt_text.clear()
        
        self.save_profiles()

def save_profiles(self):
    """Persist only the prompt profiles and active profile."""
    from PySide6.QtWidgets import QMessageBox
    try:
        data = {}
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        data['prompt_profiles'] = self.prompt_profiles
        
        # Get current profile from combobox or profile_var
        data['active_profile'] = self.profile_menu.currentText() if hasattr(self, 'profile_menu') else self.profile_var
        
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Failed to save profiles: {e}")

def import_profiles(self):
    """Import profiles from a JSON file, merging into existing ones."""
    from PySide6.QtWidgets import QMessageBox, QFileDialog
    
    path, _ = QFileDialog.getOpenFileName(
        None, 
        "Import Profiles", 
        "", 
        "JSON files (*.json);;All files (*.*)"
    )
    
    if not path:
        return
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.prompt_profiles.update(data)
        self.config['prompt_profiles'] = self.prompt_profiles
        
        # Update combobox
        self.profile_menu.clear()
        self.profile_menu.addItems(list(self.prompt_profiles.keys()))
        
        QMessageBox.information(None, "Imported", f"Imported {len(data)} profiles.")
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Failed to import profiles: {e}")

def export_profiles(self):
    """Export all profiles to a JSON file."""
    from PySide6.QtWidgets import QMessageBox, QFileDialog
    
    path, _ = QFileDialog.getSaveFileName(
        None, 
        "Export Profiles", 
        "", 
        "JSON files (*.json);;All files (*.*)"
    )
    
    if not path:
        return
    
    # Add .json extension if not present
    if not path.endswith('.json'):
        path += '.json'
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.prompt_profiles, f, ensure_ascii=False, indent=2)
        QMessageBox.information(None, "Exported", f"Profiles exported to {path}.")
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Failed to export profiles: {e}")

def _create_debug_controls_section(self, parent_frame):
    """Create debug controls section at the bottom of Other Settings (PySide6)"""
    from PySide6.QtWidgets import (
        QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame
    )
    from PySide6.QtCore import Qt
    
    try:
        grid = parent_frame.layout()
        current_row = grid.rowCount()
        
        # Create a separator line above the debug section
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("QFrame { color: #404040; margin: 10px 0; }")
        
        # Add separator spanning both columns
        grid.addWidget(separator, current_row, 0, 1, 2)
        current_row += 1
        
        # Create debug controls group box
        debug_group = QGroupBox("🔧 Debug Controls")
        debug_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12pt;
                color: #dc3545;
                border: 2px solid #dc3545;
                border-radius: 8px;
                margin: 10px 5px;
                padding-top: 35px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 15px;
                top: 20px;
                padding: 0 8px;
                background-color: transparent;
                color: #dc3545;
                font-weight: bold;
            }
        """)
        
        debug_layout = QVBoxLayout(debug_group)
        debug_layout.setSpacing(10)
        debug_layout.setContentsMargins(15, 15, 15, 15)
        
        # Description label
        desc_label = QLabel(
            "Advanced debugging tools for troubleshooting environment variables and system configuration. "
            "Only enable debug mode when investigating issues."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #888; font-size: 9pt; font-weight: normal; margin-bottom: 10px;")
        debug_layout.addWidget(desc_label)
        
        # Button container
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        # Debug Mode Toggle Button
        def _toggle_debug_mode():
            try:
                current_debug_state = getattr(self, 'config', {}).get('show_debug_buttons', False)
                new_debug_state = not current_debug_state
                
                # Update config
                if not hasattr(self, 'config'):
                    self.config = {}
                self.config['show_debug_buttons'] = new_debug_state
                
                # Save config
                self.save_config(show_message=False)
                
                # Update button appearance
                if new_debug_state:
                    debug_toggle_btn.setText("🔍 Debug Mode: ON")
                    debug_toggle_btn.setStyleSheet(
                        "QPushButton { "
                        "  background-color: #dc3545; "
                        "  color: white; "
                        "  padding: 10px 20px; "
                        "  font-size: 11pt; "
                        "  font-weight: bold; "
                        "  border-radius: 6px; "
                        "  border: none; "
                        "} "
                        "QPushButton:hover { background-color: #c82333; }"
                        "QPushButton:pressed { background-color: #bd2130; }"
                    )
                    self.append_log("✅ [DEBUG MODE] Debug mode ENABLED - enhanced debugging active")
                    self.append_log("🔧 [DEBUG MODE] Debug features now available:")
                    self.append_log("   • Enhanced environment variable debugging in save functions")
                    self.append_log("   • Comprehensive variable verification")
                    self.append_log("   • Detailed before/after tracking")
                    
                    # Show debug action button
                    debug_action_btn.setVisible(True)
                    
                else:
                    debug_toggle_btn.setText("🔒 Debug Mode: OFF")
                    debug_toggle_btn.setStyleSheet(
                        "QPushButton { "
                        "  background-color: #6c757d; "
                        "  color: white; "
                        "  padding: 10px 20px; "
                        "  font-size: 11pt; "
                        "  font-weight: bold; "
                        "  border-radius: 6px; "
                        "  border: none; "
                        "} "
                        "QPushButton:hover { background-color: #5a6268; }"
                        "QPushButton:pressed { background-color: #545b62; }"
                    )
                    self.append_log("🔒 [DEBUG MODE] Debug mode DISABLED - standard logging only")
                    
                    # Hide debug action button
                    debug_action_btn.setVisible(False)
                
            except Exception as e:
                self.append_log(f"❌ [DEBUG MODE] Failed to toggle debug mode: {e}")
        
        def _run_debug_check():
            try:
                self.append_log("🔍 [DEBUG ACTION] Running comprehensive environment variable check...")
                # First initialize if needed
                init_success = self.initialize_environment_variables()
                # Then debug
                debug_success = self.debug_environment_variables(show_all=True)
                
                if init_success and debug_success:
                    self.append_log("✅ [DEBUG ACTION] Environment variables are properly configured")
                else:
                    self.append_log("❌ [DEBUG ACTION] Environment variable issues detected - check log for details")
                    
            except Exception as e:
                self.append_log(f"❌ [DEBUG ACTION] Debug check failed: {e}")
        
        # Create buttons
        current_debug_state = getattr(self, 'config', {}).get('show_debug_buttons', False)
        debug_toggle_btn = QPushButton("🔍 Debug Mode: ON" if current_debug_state else "🔒 Debug Mode: OFF")
        debug_toggle_btn.clicked.connect(_toggle_debug_mode)
        debug_toggle_btn.setMinimumHeight(45)
        debug_toggle_btn.setMinimumWidth(180)
        
        if current_debug_state:
            debug_toggle_btn.setStyleSheet(
                "QPushButton { "
                "  background-color: #dc3545; "
                "  color: white; "
                "  padding: 10px 20px; "
                "  font-size: 11pt; "
                "  font-weight: bold; "
                "  border-radius: 6px; "
                "  border: none; "
                "} "
                "QPushButton:hover { background-color: #c82333; }"
                "QPushButton:pressed { background-color: #bd2130; }"
            )
        else:
            debug_toggle_btn.setStyleSheet(
                "QPushButton { "
                "  background-color: #6c757d; "
                "  color: white; "
                "  padding: 10px 20px; "
                "  font-size: 11pt; "
                "  font-weight: bold; "
                "  border-radius: 6px; "
                "  border: none; "
                "} "
                "QPushButton:hover { background-color: #5a6268; }"
                "QPushButton:pressed { background-color: #545b62; }"
            )
        
        button_layout.addWidget(debug_toggle_btn)
        
        # Debug Action Button (only visible when debug mode is on)
        debug_action_btn = QPushButton("🔍 Check Environment Variables")
        debug_action_btn.clicked.connect(_run_debug_check)
        debug_action_btn.setMinimumHeight(45)
        debug_action_btn.setMinimumWidth(220)
        debug_action_btn.setStyleSheet(
            "QPushButton { "
            "  background-color: #17a2b8; "
            "  color: white; "
            "  padding: 10px 20px; "
            "  font-size: 11pt; "
            "  font-weight: bold; "
            "  border-radius: 6px; "
            "  border: none; "
            "} "
            "QPushButton:hover { background-color: #138496; }"
            "QPushButton:pressed { background-color: #117a8b; }"
        )
        
        # Only show action button if debug mode is currently enabled
        debug_action_btn.setVisible(current_debug_state)
        button_layout.addWidget(debug_action_btn)
        
        # Add stretch to center buttons
        button_layout.addStretch()
        
        debug_layout.addLayout(button_layout)
        
        # Store references for potential future use
        self._debug_toggle_btn = debug_toggle_btn
        self._debug_action_btn = debug_action_btn
        
        # Add debug group to the grid, spanning both columns at the bottom
        grid.addWidget(debug_group, current_row, 0, 1, 2)
        
    except Exception as e:
        print(f"Error creating debug controls section: {e}")
