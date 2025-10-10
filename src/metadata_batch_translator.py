"""
Metadata and Batch Header Translation Module
Handles custom metadata fields and batch chapter header translation
Complete implementation - no truncation
"""

import os
import json
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                               QCheckBox, QRadioButton, QButtonGroup, QWidget, QScrollArea,
                               QFrame, QGroupBox, QTabWidget, QTextEdit, QLineEdit, QComboBox,
                               QMessageBox, QApplication)
from PySide6.QtGui import QFont, QIcon
from PySide6.QtCore import Qt, QSize
from typing import Dict, List, Tuple, Optional, Any
import zipfile
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor


class MetadataBatchTranslatorUI:
    """UI handlers for metadata and batch translation features"""
    
    def __init__(self, parent_gui):
        """Initialize with reference to main TranslatorGUI"""
        self.gui = parent_gui
        # WindowManager and UIHelper removed - not needed in PySide6
        
        # Initialize default prompts if not in config
        self._initialize_default_prompts()
    
    def _create_styled_checkbox(self, text):
        """Create a checkbox with proper checkmark using text overlay"""
        from PySide6.QtCore import QTimer
        
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
    
    def _initialize_default_prompts(self):
        """Initialize all default prompts in config if not present"""
        # Batch header system prompt (NEW)
        if 'batch_header_system_prompt' not in self.gui.config:
            self.gui.config['batch_header_system_prompt'] = (
                "You are a professional translator specializing in novel chapter titles. "
                "Respond with only the translated JSON, nothing else. "
                "Maintain the original tone and style while making titles natural in the target language."
            )
        
        # Batch header user prompt (existing)
        if 'batch_header_prompt' not in self.gui.config:
            self.gui.config['batch_header_prompt'] = (
                "Translate these chapter titles to English.\n"
                "- For titles with parentheses containing Chinese/Japanese characters (like 終篇, 完結編, etc.), translate both the main title and the parenthetical text.\n"
                "- Common markers: 終篇/終章 = 'Final Chapter', 完結編 = 'Final Arc/Volume', 後編 = 'Part 2', 前編 = 'Part 1'.\n"
                "- Translate the meaning accurately - don't use overly dramatic words unless the original implies them.\n"
                "- Preserve the chapter number format exactly as shown.\n"
                "Return ONLY a JSON object with chapter numbers as keys.\n"
                "Format: {\"1\": \"translated title\", \"2\": \"translated title\"}"
            )
        
        # Metadata batch prompt
        if 'metadata_batch_prompt' not in self.gui.config:
            self.gui.config['metadata_batch_prompt'] = (
                "Translate the following metadata fields to English.\n"
                "Output ONLY a JSON object with the same field names as keys."
            )
        
        # Field-specific prompts
        if 'metadata_field_prompts' not in self.gui.config:
            self.gui.config['metadata_field_prompts'] = {
                'creator': "Romanize this author name. Do not output anything other than the romanized text.",
                'publisher': "Romanize this publisher name. Do not output anything other than the romanized text.",
                'subject': "Translate this book genre/subject to English. Do not output anything other than the translated text:",
                'description': "Translate this book description to English. Do not output anything other than the translated text:",
                'series': "Translate this series name to English. Do not output anything other than the translated text:",
                '_default': "Translate this text to English. Do not output anything other than the translated text:"
            }

            
    def configure_metadata_fields(self):
            """Configure which metadata fields to translate"""
            # Ensure QApplication instance exists
            import sys
            app = QApplication.instance()
            if not app:
                try:
                    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
                except:
                    pass
                app = QApplication(sys.argv)
            
            # Create dialog (use None as parent for top-level window)
            dialog = QDialog(None)
            dialog.setWindowTitle("Configure Metadata Translation")
            
            # Get screen dimensions and calculate size
            screen = app.primaryScreen().geometry()
            dialog_width = int(screen.width() * 0.28)  # Slightly increased from 0.25
            dialog_height = int(screen.height() * 0.50)
            dialog.resize(dialog_width, dialog_height)
            
            # Set window icon
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "halgakos.ico")
            if os.path.exists(icon_path):
                dialog.setWindowIcon(QIcon(icon_path))
            
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
                QCheckBox::indicator:hover {
                    border: 1px solid #7ab8e8;
                    background-color: #3d3d3d;
                }
                QCheckBox::indicator:checked {
                    background-color: #5a9fd4;
                    border: 1px solid #5a9fd4;
                }
                QCheckBox::indicator:checked:hover {
                    background-color: #7ab8e8;
                    border: 1px solid #7ab8e8;
                }
                QCheckBox::indicator:disabled {
                    border: 1px solid #555555;
                    background-color: #1e1e1e;
                }
                QRadioButton {
                    color: white;
                    spacing: 6px;
                }
                QRadioButton::indicator {
                    width: 14px;
                    height: 14px;
                    border: 1px solid #5a9fd4;
                    border-radius: 7px;
                    background-color: #2d2d2d;
                }
                QRadioButton::indicator:hover {
                    border: 1px solid #7ab8e8;
                    background-color: #3d3d3d;
                }
                QRadioButton::indicator:checked {
                    background-color: #5a9fd4;
                    border: 1px solid #5a9fd4;
                }
                QRadioButton::indicator:checked:hover {
                    background-color: #7ab8e8;
                    border: 1px solid #7ab8e8;
                }
                QRadioButton::indicator:disabled {
                    border: 1px solid #555555;
                    background-color: #1e1e1e;
                }
                QGroupBox {
                    color: white;
                    border: 1px solid #555555;
                    border-radius: 4px;
                    margin-top: 10px;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    left: 10px;
                    padding: 0 5px;
                    color: #5a9fd4;
                    font-weight: bold;
                }
            """
            dialog.setStyleSheet(checkbox_radio_style)
            
            # Main layout
            main_layout = QVBoxLayout()
            main_layout.setContentsMargins(20, 20, 20, 20)
            
            # Title
            title_label = QLabel("Select Metadata Fields to Translate")
            title_font = QFont()
            title_font.setPointSize(14)
            title_font.setBold(True)
            title_label.setFont(title_font)
            main_layout.addWidget(title_label)
            
            # Description
            desc_label = QLabel("These fields will be translated along with or separately from the book title:")
            desc_label.setStyleSheet("color: gray; font-size: 10pt;")
            desc_label.setWordWrap(True)
            desc_label.setContentsMargins(0, 10, 0, 20)
            main_layout.addWidget(desc_label)
            
            # Create scroll area for fields
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setFrameShape(QFrame.NoFrame)
            
            fields_container = QWidget()
            fields_layout = QVBoxLayout(fields_container)
            fields_layout.setContentsMargins(0, 0, 0, 0)
            
            # Load metadata fields from EPUB
            all_fields = self._detect_all_metadata_fields()
            
            # Standard fields
            standard_fields = {
                'title': ('Title', 'The book title'),
                'creator': ('Author/Creator', 'The author or creator'),
                'publisher': ('Publisher', 'The publishing company'),
                'subject': ('Subject/Genre', 'Subject categories or genres'),
                'description': ('Description', 'Book synopsis'),
                'series': ('Series Name', 'Name of the book series'),
                'language': ('Language', 'Original language'),
                'date': ('Publication Date', 'When published'),
                'rights': ('Rights', 'Copyright information')
            }
            
            field_vars = {}
            
            # Section for standard fields
            section_label = QLabel("Standard Metadata Fields:")
            section_font = QFont()
            section_font.setPointSize(12)
            section_font.setBold(True)
            section_label.setFont(section_font)
            section_label.setContentsMargins(0, 10, 0, 5)
            fields_layout.addWidget(section_label)
            
            # Get saved settings
            translate_fields = self.gui.config.get('translate_metadata_fields', {})
            
            for field, (label, description) in standard_fields.items():
                if field in all_fields:
                    frame = QWidget()
                    frame_layout = QHBoxLayout(frame)
                    frame_layout.setContentsMargins(0, 5, 0, 5)
                    
                    # Special handling for title field - show note instead of checkbox
                    if field == 'title':
                        # Show the title field info but with a note instead of checkbox
                        label_widget = QLabel(f"{label}:")
                        label_font = QFont()
                        label_font.setBold(True)
                        label_widget.setFont(label_font)
                        label_widget.setMinimumWidth(200)
                        frame_layout.addWidget(label_widget)
                        
                        # Show current value
                        current_value = str(all_fields[field])
                        if len(current_value) > 50:
                            current_value = current_value[:47] + "..."
                        value_label = QLabel(current_value)
                        value_label.setStyleSheet("color: gray; font-size: 9pt;")
                        frame_layout.addWidget(value_label)
                        frame_layout.addStretch()
                        fields_layout.addWidget(frame)
                        
                        # Add note explaining title is controlled elsewhere
                        note_label = QLabel("ℹ️ Title translation is controlled by the 'Translate Book Title' setting in the main interface")
                        note_label.setStyleSheet("color: #5dade2; font-size: 9pt;")  # Light blue/cyan for better contrast
                        note_label.setWordWrap(True)
                        note_label.setContentsMargins(25, 0, 0, 10)
                        fields_layout.addWidget(note_label)
                        continue  # Skip to next field
                    
                    # Normal handling for other fields
                    default_value = False  # All other fields default to False
                    checkbox = self._create_styled_checkbox(f"{label}:")
                    checkbox.setChecked(translate_fields.get(field, default_value))
                    checkbox.setMinimumWidth(200)
                    field_vars[field] = checkbox
                    frame_layout.addWidget(checkbox)
                    
                    # Show current value
                    current_value = str(all_fields[field])
                    if len(current_value) > 50:
                        current_value = current_value[:47] + "..."
                    value_label = QLabel(current_value)
                    value_label.setStyleSheet("color: gray; font-size: 9pt;")
                    frame_layout.addWidget(value_label)
                    frame_layout.addStretch()
                    fields_layout.addWidget(frame)
            
            # Custom fields section
            custom_fields = {k: v for k, v in all_fields.items() if k not in standard_fields}
            
            if custom_fields:
                custom_label = QLabel("Custom Metadata Fields:")
                custom_font = QFont()
                custom_font.setPointSize(12)
                custom_font.setBold(True)
                custom_label.setFont(custom_font)
                custom_label.setContentsMargins(0, 20, 0, 5)
                fields_layout.addWidget(custom_label)
                
                custom_desc = QLabel("(Non-standard fields found in your EPUB)")
                custom_desc.setStyleSheet("color: gray; font-size: 10pt;")
                custom_desc.setContentsMargins(0, 0, 0, 10)
                fields_layout.addWidget(custom_desc)
                
                for field, value in custom_fields.items():
                    frame = QWidget()
                    frame_layout = QHBoxLayout(frame)
                    frame_layout.setContentsMargins(0, 5, 0, 5)
                    
                    checkbox = self._create_styled_checkbox(f"{field}:")
                    checkbox.setChecked(translate_fields.get(field, False))
                    checkbox.setMinimumWidth(200)
                    field_vars[field] = checkbox
                    frame_layout.addWidget(checkbox)
                    
                    display_value = str(value)
                    if len(display_value) > 50:
                        display_value = display_value[:47] + "..."
                    value_label = QLabel(display_value)
                    value_label.setStyleSheet("color: gray; font-size: 9pt;")
                    frame_layout.addWidget(value_label)
                    frame_layout.addStretch()
                    fields_layout.addWidget(frame)
            
            # Add stretch to fields layout
            fields_layout.addStretch()
            scroll_area.setWidget(fields_container)
            main_layout.addWidget(scroll_area)
            
            # Translation mode
            mode_group = QGroupBox("Translation Mode")
            mode_layout = QVBoxLayout()
            mode_layout.setContentsMargins(10, 10, 10, 10)
            
            translation_mode_group = QButtonGroup()
            current_mode = self.gui.config.get('metadata_translation_mode', 'together')
            
            rb1 = QRadioButton("Translate together (single API call)")
            rb1.setChecked(current_mode == 'together')
            rb1.setProperty("value", "together")
            translation_mode_group.addButton(rb1)
            mode_layout.addWidget(rb1)
            
            rb2 = QRadioButton("Translate separately (parallel API calls)")
            rb2.setChecked(current_mode == 'parallel')
            rb2.setProperty("value", "parallel")
            translation_mode_group.addButton(rb2)
            mode_layout.addWidget(rb2)
            
            mode_group.setLayout(mode_layout)
            main_layout.addWidget(mode_group)
            
            # Buttons
            button_layout = QHBoxLayout()
            button_layout.setSpacing(10)
            button_layout.addStretch()  # Center buttons
            
            def save_metadata_config():
                # Update configuration
                self.gui.translate_metadata_fields = {}
                for field, checkbox in field_vars.items():
                    if checkbox.isChecked():
                        self.gui.translate_metadata_fields[field] = True
                
                # Get selected translation mode
                selected_mode = 'together'
                for button in translation_mode_group.buttons():
                    if button.isChecked():
                        selected_mode = button.property("value")
                        break
                
                self.gui.config['translate_metadata_fields'] = self.gui.translate_metadata_fields
                self.gui.config['metadata_translation_mode'] = selected_mode
                self.gui.save_config(show_message=False)
                dialog.accept()
            
            def reset_metadata_config():
                msg_box = QMessageBox(dialog)  # Set dialog as parent
                msg_box.setIcon(QMessageBox.Question)
                msg_box.setWindowTitle("Reset Metadata Fields")
                msg_box.setText("Are you sure you want to reset all metadata field selections?\n\nThis will uncheck all fields.")
                msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg_box.setDefaultButton(QMessageBox.No)
                msg_box.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)  # Force on top
                if os.path.exists(icon_path):
                    msg_box.setWindowIcon(QIcon(icon_path))
                
                if msg_box.exec() == QMessageBox.Yes:
                    for field, checkbox in field_vars.items():
                        # Since title is no longer in field_vars, all fields default to False
                        checkbox.setChecked(False)
            
            save_btn = QPushButton("💾 Save")
            save_btn.setMinimumWidth(120)
            save_btn.setMinimumHeight(35)
            save_btn.clicked.connect(save_metadata_config)
            save_btn.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    padding: 8px 20px;
                    font-size: 11pt;
                    font-weight: bold;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
            """)
            button_layout.addWidget(save_btn)
            
            reset_btn = QPushButton("↺ Reset")
            reset_btn.setMinimumWidth(120)
            reset_btn.setMinimumHeight(35)
            reset_btn.clicked.connect(reset_metadata_config)
            reset_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ffc107;
                    color: black;
                    padding: 8px 20px;
                    font-size: 11pt;
                    font-weight: bold;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #e0a800;
                }
            """)
            button_layout.addWidget(reset_btn)
            
            cancel_btn = QPushButton("❌ Cancel")
            cancel_btn.setMinimumWidth(120)
            cancel_btn.setMinimumHeight(35)
            cancel_btn.clicked.connect(dialog.reject)
            cancel_btn.setStyleSheet("""
                QPushButton {
                    background-color: #6c757d;
                    color: white;
                    padding: 8px 20px;
                    font-size: 11pt;
                    font-weight: bold;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #5a6268;
                }
            """)
            button_layout.addWidget(cancel_btn)
            button_layout.addStretch()  # Center buttons
            
            main_layout.addLayout(button_layout)
            dialog.setLayout(main_layout)
            
            # Run dialog in separate thread to avoid GIL conflicts
            import threading
            result = {'done': False}
            
            def run_dialog():
                dialog.exec()
                result['done'] = True
            
            thread = threading.Thread(target=run_dialog, daemon=True)
            thread.start()
            
            # Keep the dialog reference to prevent garbage collection
            self.gui._metadata_dialog = dialog
    
    def configure_translation_prompts(self):
        """Configure all translation prompts in one place"""
        # Ensure QApplication instance exists
        import sys
        app = QApplication.instance()
        if not app:
            try:
                QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
            except:
                pass
            app = QApplication(sys.argv)
        
        # Create dialog (use None as parent for top-level window)
        dialog = QDialog(None)
        dialog.setWindowTitle("Configure Translation Prompts")
        
        # Get screen dimensions and calculate size
        screen = app.primaryScreen().geometry()
        dialog_width = int(screen.width() * 0.3)  # Reduced from 70% to 35%
        dialog_height = int(screen.height() * 0.8)
        dialog.resize(dialog_width, dialog_height)
        
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
        dialog.setStyleSheet(checkbox_radio_style)
        
        # Set window icon
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "halgakos.ico")
        if os.path.exists(icon_path):
            dialog.setWindowIcon(QIcon(icon_path))
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("Configure All Translation Prompts")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel("Customize how different types of content are translated")
        desc_label.setStyleSheet("color: gray; font-size: 10pt;")
        desc_label.setContentsMargins(0, 10, 0, 20)
        main_layout.addWidget(desc_label)
        
        # Create tab widget for different prompt categories
        tab_widget = QTabWidget()
        
        # Style the tab widget to make tabs stand out
        tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #3daee9;
                border-radius: 4px;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                color: #e0e0e0;
                border: 1px solid #555555;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 20px;
                margin-right: 2px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #3daee9;
                color: white;
                border-color: #3daee9;
            }
            QTabBar::tab:hover:!selected {
                background-color: #505050;
                color: white;
            }
        """)
        
        # Tab 1: Book Title Prompts
        title_tab = QWidget()
        tab_widget.addTab(title_tab, "📖 Book Titles")
        self._create_title_prompts_tab(title_tab)
        
        # Tab 2: Chapter Header Prompts
        header_tab = QWidget()
        tab_widget.addTab(header_tab, "📑 Chapter Headers")
        self._create_header_prompts_tab(header_tab)
        
        # Tab 3: Metadata Field Prompts
        metadata_tab = QWidget()
        tab_widget.addTab(metadata_tab, "🏷️ Metadata Fields")
        self._create_metadata_prompts_tab(metadata_tab)
        
        # Tab 4: Advanced Prompts
        advanced_tab = QWidget()
        tab_widget.addTab(advanced_tab, "⚙️ Advanced")
        self._create_advanced_prompts_tab(advanced_tab)
        
        main_layout.addWidget(tab_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        def save_all_prompts():
            # Save all text widgets to config
            self._save_all_prompt_configs()
            self.gui.save_config(show_message=False)
            dialog.accept()
        
        def reset_all_prompts():
            msg_box = QMessageBox(dialog)  # Set dialog as parent
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setWindowTitle("Reset All Prompts")
            msg_box.setText("Are you sure you want to reset ALL prompts to their default values?\n\nThis cannot be undone.")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.No)
            msg_box.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)  # Force on top
            if os.path.exists(icon_path):
                msg_box.setWindowIcon(QIcon(icon_path))
            
            if msg_box.exec() == QMessageBox.Yes:
                self._reset_all_prompts_to_defaults()
                dialog.close()
                # Re-open dialog with defaults to refresh the UI
                import threading
                def reopen():
                    import time
                    time.sleep(0.1)  # Small delay to let dialog close cleanly
                    self.configure_translation_prompts()
                threading.Thread(target=reopen, daemon=True).start()
        
        save_btn = QPushButton("💾 Save All")
        save_btn.setMinimumWidth(150)
        save_btn.setMinimumHeight(35)
        save_btn.clicked.connect(save_all_prompts)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 8px 20px;
                font-size: 11pt;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        button_layout.addWidget(save_btn)
        
        reset_btn = QPushButton("↺ Reset All to Defaults")
        reset_btn.setMinimumWidth(200)
        reset_btn.setMinimumHeight(35)
        reset_btn.clicked.connect(reset_all_prompts)
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffc107;
                color: black;
                padding: 8px 20px;
                font-size: 11pt;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e0a800;
            }
        """)
        button_layout.addWidget(reset_btn)
        
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.setMinimumWidth(120)
        cancel_btn.setMinimumHeight(35)
        cancel_btn.clicked.connect(dialog.reject)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                padding: 8px 20px;
                font-size: 11pt;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        button_layout.addWidget(cancel_btn)
        
        # Center the buttons
        button_layout.insertStretch(0)
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
        dialog.setLayout(main_layout)
        
        # Run dialog in separate thread to avoid GIL conflicts
        import threading
        result = {'done': False}
        
        def run_dialog():
            dialog.exec()
            result['done'] = True
        
        thread = threading.Thread(target=run_dialog, daemon=True)
        thread.start()
        
        # Keep the dialog reference to prevent garbage collection
        self.gui._prompts_dialog = dialog
    
    def _create_title_prompts_tab(self, parent):
        """Create tab for book title prompts"""
        tab_layout = QVBoxLayout(parent)
        tab_layout.setContentsMargins(20, 20, 20, 20)
        
        # System prompt
        system_label = QLabel("System Prompt (AI Instructions)")
        system_font = QFont()
        system_font.setPointSize(12)
        system_font.setBold(True)
        system_label.setFont(system_font)
        tab_layout.addWidget(system_label)
        
        system_desc = QLabel("Defines how the AI should behave when translating titles:")
        system_desc.setStyleSheet("color: gray; font-size: 10pt;")
        system_desc.setContentsMargins(0, 5, 0, 10)
        tab_layout.addWidget(system_desc)
        
        self.title_system_text = QTextEdit()
        self.title_system_text.setMinimumHeight(100)
        self.title_system_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            QTextEdit:focus {
                border: 1px solid #5a9fd4;
            }
        """)
        self.title_system_text.setPlainText(self.gui.config.get('book_title_system_prompt', 
            "You are a translator. Respond with only the translated text, nothing else."))
        tab_layout.addWidget(self.title_system_text)
        
        # User prompt
        user_label = QLabel("User Prompt (Translation Request)")
        user_font = QFont()
        user_font.setPointSize(12)
        user_font.setBold(True)
        user_label.setFont(user_font)
        user_label.setContentsMargins(0, 15, 0, 0)
        tab_layout.addWidget(user_label)
        
        self.title_user_text = QTextEdit()
        self.title_user_text.setMinimumHeight(80)
        self.title_user_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                font-family: 'Consolas', 'Courier New', monospace;
            }
        """)
        self.title_user_text.setPlainText(self.gui.config.get('book_title_prompt',
            "Translate this book title to English while retaining any acronyms:"))
        tab_layout.addWidget(self.title_user_text)
        
        tab_layout.addStretch()
    
    def _create_header_prompts_tab(self, parent):
        """Create tab for chapter header prompts"""
        tab_layout = QVBoxLayout(parent)
        tab_layout.setContentsMargins(20, 20, 20, 20)
        
        # System prompt for batch headers
        system_label = QLabel("System Prompt (AI Instructions)")
        system_font = QFont()
        system_font.setPointSize(12)
        system_font.setBold(True)
        system_label.setFont(system_font)
        tab_layout.addWidget(system_label)
        
        system_desc = QLabel("Defines how the AI should behave when translating chapter headers:")
        system_desc.setStyleSheet("color: gray; font-size: 10pt;")
        system_desc.setContentsMargins(0, 5, 0, 10)
        tab_layout.addWidget(system_desc)
        
        self.header_batch_system_text = QTextEdit()
        self.header_batch_system_text.setMinimumHeight(100)
        self.header_batch_system_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                font-family: 'Consolas', 'Courier New', monospace;
            }
        """)
        self.header_batch_system_text.setPlainText(self.gui.config.get('batch_header_system_prompt',
            "You are a professional translator specializing in novel chapter titles. "
            "Respond with only the translated JSON, nothing else. "
            "Maintain the original tone and style while making titles natural in the target language."))
        tab_layout.addWidget(self.header_batch_system_text)
        
        # User prompt for batch headers
        user_label = QLabel("User Prompt (Translation Request)")
        user_font = QFont()
        user_font.setPointSize(12)
        user_font.setBold(True)
        user_label.setFont(user_font)
        user_label.setContentsMargins(0, 15, 0, 0)
        tab_layout.addWidget(user_label)
        
        user_desc = QLabel("Instructions for how to translate the chapter headers:")
        user_desc.setStyleSheet("color: gray; font-size: 10pt;")
        user_desc.setContentsMargins(0, 5, 0, 10)
        tab_layout.addWidget(user_desc)
        
        self.header_batch_text = QTextEdit()
        self.header_batch_text.setMinimumHeight(150)
        self.header_batch_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                font-family: 'Consolas', 'Courier New', monospace;
            }
        """)
        self.header_batch_text.setPlainText(self.gui.config.get('batch_header_prompt',
            "Translate these chapter titles to English.\n"
            "Return ONLY a JSON object with chapter numbers as keys.\n"
            "Format: {\"1\": \"translated title\", \"2\": \"translated title\"}"))
        tab_layout.addWidget(self.header_batch_text)
        
        var_label = QLabel("Variables available: {source_lang} - detected source language")
        var_label.setStyleSheet("color: blue; font-size: 10pt;")
        var_label.setContentsMargins(0, 10, 0, 0)
        tab_layout.addWidget(var_label)
        
        tab_layout.addStretch()
    
    def _create_metadata_prompts_tab(self, parent):
        """Create tab for metadata field prompts"""
        # Create scroll area for this tab
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        scroll_widget = QWidget()
        tab_layout = QVBoxLayout(scroll_widget)
        tab_layout.setContentsMargins(20, 20, 20, 20)
        
        # Batch prompt
        batch_label = QLabel("Batch Metadata Translation Prompt")
        batch_font = QFont()
        batch_font.setPointSize(12)
        batch_font.setBold(True)
        batch_label.setFont(batch_font)
        tab_layout.addWidget(batch_label)
        
        batch_desc = QLabel("Used when translating multiple metadata fields together:")
        batch_desc.setStyleSheet("color: gray; font-size: 10pt;")
        batch_desc.setContentsMargins(0, 5, 0, 10)
        tab_layout.addWidget(batch_desc)
        
        self.metadata_batch_text = QTextEdit()
        self.metadata_batch_text.setMinimumHeight(100)
        self.metadata_batch_text.setMaximumHeight(120)
        self.metadata_batch_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                font-family: 'Consolas', 'Courier New', monospace;
            }
        """)
        self.metadata_batch_text.setPlainText(self.gui.config.get('metadata_batch_prompt',
            "Translate the following metadata fields to English.\n"
            "Return ONLY a JSON object with the same field names as keys."))
        tab_layout.addWidget(self.metadata_batch_text)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setContentsMargins(0, 20, 0, 20)
        tab_layout.addWidget(separator)
        
        # Field-specific prompts
        field_label = QLabel("Field-Specific Prompts")
        field_font = QFont()
        field_font.setPointSize(12)
        field_font.setBold(True)
        field_label.setFont(field_font)
        tab_layout.addWidget(field_label)
        
        field_desc = QLabel("Customize prompts for each metadata field type:")
        field_desc.setStyleSheet("color: gray; font-size: 10pt;")
        field_desc.setContentsMargins(0, 5, 0, 10)
        tab_layout.addWidget(field_desc)
        
        field_prompts = self.gui.config.get('metadata_field_prompts', {})
        self.field_prompt_widgets = {}
        
        fields = [
            ('creator', 'Author/Creator'),
            ('publisher', 'Publisher'),
            ('subject', 'Subject/Genre'),
            ('description', 'Description'),
            ('series', 'Series Name'),
            ('_default', 'Default (Other Fields)')
        ]
        
        for field_key, field_label_text in fields:
            field_widget = QWidget()
            field_layout = QVBoxLayout(field_widget)
            field_layout.setContentsMargins(0, 10, 0, 0)
            
            label = QLabel(f"{field_label_text}:")
            label_font = QFont()
            label_font.setBold(True)
            label.setFont(label_font)
            field_layout.addWidget(label)
            
            text_widget = QTextEdit()
            text_widget.setMinimumHeight(60)
            text_widget.setMaximumHeight(80)
            text_widget.setStyleSheet("""
                QTextEdit {
                    background-color: #1e1e1e;
                    color: #e0e0e0;
                    border: 1px solid #555555;
                    border-radius: 3px;
                    padding: 5px;
                    font-family: 'Consolas', 'Courier New', monospace;
                }
            """)
            
            default_prompt = field_prompts.get(field_key, f"Translate this {field_label_text.lower()} to English:")
            text_widget.setPlainText(default_prompt)
            
            self.field_prompt_widgets[field_key] = text_widget
            field_layout.addWidget(text_widget)
            
            tab_layout.addWidget(field_widget)
        
        var_label = QLabel("Variables: {source_lang} - detected language, {field_value} - the text to translate")
        var_label.setStyleSheet("color: blue; font-size: 10pt;")
        var_label.setContentsMargins(0, 10, 0, 0)
        tab_layout.addWidget(var_label)
        
        tab_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        
        # Set scroll area as the tab's main widget
        main_layout = QVBoxLayout(parent)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)
            
    def _create_advanced_prompts_tab(self, parent):
        """Create tab for advanced prompt settings"""
        tab_layout = QVBoxLayout(parent)
        tab_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("Advanced Prompt Settings")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setContentsMargins(0, 0, 0, 10)
        tab_layout.addWidget(title_label)
        
        # Language detection behavior
        lang_group = QGroupBox("Language Detection")
        lang_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #3daee9;
            }
        """)
        lang_layout = QVBoxLayout()
        lang_layout.setContentsMargins(15, 15, 15, 15)
        
        lang_desc = QLabel("How to handle source language in prompts:")
        lang_desc.setContentsMargins(0, 0, 0, 10)
        lang_layout.addWidget(lang_desc)
        
        self.lang_behavior_group = QButtonGroup()
        current_behavior = self.gui.config.get('lang_prompt_behavior', 'auto')
        
        rb1 = QRadioButton("Auto-detect and include language (e.g., 'Translate this Korean text')")
        rb1.setChecked(current_behavior == 'auto')
        rb1.setProperty("value", "auto")
        self.lang_behavior_group.addButton(rb1)
        lang_layout.addWidget(rb1)
        
        rb2 = QRadioButton("Never include language (e.g., 'Translate this text')")
        rb2.setChecked(current_behavior == 'never')
        rb2.setProperty("value", "never")
        self.lang_behavior_group.addButton(rb2)
        lang_layout.addWidget(rb2)
        
        rb3 = QRadioButton("Always specify language:")
        rb3.setChecked(current_behavior == 'always')
        rb3.setProperty("value", "always")
        self.lang_behavior_group.addButton(rb3)
        lang_layout.addWidget(rb3)
        
        # Language entry
        lang_entry_widget = QWidget()
        lang_entry_layout = QHBoxLayout(lang_entry_widget)
        lang_entry_layout.setContentsMargins(20, 5, 0, 0)
        
        lang_entry_label = QLabel("Language to use:")
        lang_entry_layout.addWidget(lang_entry_label)
        
        self.forced_lang_entry = QLineEdit()
        self.forced_lang_entry.setText(self.gui.config.get('forced_source_lang', 'Korean'))
        self.forced_lang_entry.setFixedWidth(150)
        self.forced_lang_entry.setStyleSheet("""
            QLineEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
            }
            QLineEdit:focus {
                border: 1px solid #5a9fd4;
            }
        """)
        lang_entry_layout.addWidget(self.forced_lang_entry)
        lang_entry_layout.addStretch()
        
        lang_layout.addWidget(lang_entry_widget)
        lang_group.setLayout(lang_layout)
        tab_layout.addWidget(lang_group)
        
        # Output language
        output_group = QGroupBox("Output Language")
        output_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #3daee9;
            }
        """)
        output_layout = QVBoxLayout()
        output_layout.setContentsMargins(15, 15, 15, 15)
        
        output_desc = QLabel("Target language for translations:")
        output_desc.setContentsMargins(0, 0, 0, 10)
        output_layout.addWidget(output_desc)
        
        common_langs = ['English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 
                       'Russian', 'Japanese', 'Korean', 'Chinese (Simplified)', 'Chinese (Traditional)']
        
        lang_label = QLabel("Target language:")
        output_layout.addWidget(lang_label)
        
        self.output_lang_combo = QComboBox()
        self.output_lang_combo.addItems(common_langs)
        self.output_lang_combo.setEditable(True)
        self.output_lang_combo.setCurrentText(self.gui.config.get('output_language', 'English'))
        self.output_lang_combo.setFixedWidth(250)
        self.output_lang_combo.setStyleSheet("""
            QComboBox {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                padding-right: 30px;
            }
            QComboBox:focus {
                border: 1px solid #5a9fd4;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border-left: 1px solid #555555;
                background-color: transparent;
            }
            QComboBox::drop-down:hover {
                background-color: rgba(255, 255, 255, 0.1);
            }
            QComboBox::down-arrow {
                image: url(halgakos.ico);
                width: 16px;
                height: 16px;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #555555;
                selection-background-color: #3daee9;
                outline: none;
            }
            QComboBox QAbstractItemView::item {
                padding: 5px;
                border: none;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #3daee9;
                color: white;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #3daee9;
                color: white;
            }
        """)
        output_layout.addWidget(self.output_lang_combo)
        
        # Force the dropdown to be properly clickable
        from PySide6.QtCore import Qt
        
        # Get the dropdown list view and ensure it accepts mouse events
        dropdown_view = self.output_lang_combo.view()
        dropdown_view.setMouseTracking(True)
        dropdown_view.setAttribute(Qt.WA_Hover, True)
        
        # Ensure the combo box itself is properly configured
        self.output_lang_combo.setAttribute(Qt.WA_Hover, True)
        self.output_lang_combo.setFocusPolicy(Qt.StrongFocus)
        
        # Make sure it's enabled
        self.output_lang_combo.setEnabled(True)
        dropdown_view.setEnabled(True)
        
        # Force proper selection handling
        from PySide6.QtWidgets import QAbstractItemView
        dropdown_view.setSelectionMode(QAbstractItemView.SingleSelection)
        dropdown_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        
        # Connect a manual click handler to the view
        def on_view_clicked(index):
            if index.isValid():
                self.output_lang_combo.setCurrentIndex(index.row())
                self.output_lang_combo.hidePopup()
        
        dropdown_view.clicked.connect(on_view_clicked)
        
        # Debug: Print current state
        print(f"Dropdown enabled: {self.output_lang_combo.isEnabled()}")
        print(f"Dropdown view enabled: {dropdown_view.isEnabled()}")
        print(f"Parent group enabled: {output_group.isEnabled()}")
        print(f"Parent layout widget enabled: {output_group.parent().isEnabled() if output_group.parent() else 'No parent'}")
        
        # Add proper signal connections like the working model dropdown
        self.output_lang_combo.currentIndexChanged.connect(lambda: None)  # Basic signal connection
        
        # Disable mousewheel scrolling
        self.output_lang_combo.wheelEvent = lambda event: None
        
        output_note = QLabel("This will replace 'English' in all prompts with your chosen language")
        output_note.setStyleSheet("color: gray; font-size: 9pt;")
        output_note.setContentsMargins(0, 5, 0, 0)
        output_layout.addWidget(output_note)
        
        output_group.setLayout(output_layout)
        tab_layout.addWidget(output_group)
        
        tab_layout.addStretch()
    
    def _save_all_prompt_configs(self):
        """Save all prompt configurations"""
        # Book title prompts
        self.gui.config['book_title_system_prompt'] = self.title_system_text.toPlainText().strip()
        self.gui.config['book_title_prompt'] = self.title_user_text.toPlainText().strip()
        # Only set attribute if it exists on the gui object
        if hasattr(self.gui, 'book_title_prompt'):
            self.gui.book_title_prompt = self.gui.config['book_title_prompt']
        
        # Batch header prompts
        self.gui.config['batch_header_system_prompt'] = self.header_batch_system_text.toPlainText().strip()
        self.gui.config['batch_header_prompt'] = self.header_batch_text.toPlainText().strip()
        
        # Metadata prompts
        self.gui.config['metadata_batch_prompt'] = self.metadata_batch_text.toPlainText().strip()
        
        # Field-specific prompts
        field_prompts = {}
        for field_key, widget in self.field_prompt_widgets.items():
            field_prompts[field_key] = widget.toPlainText().strip()
        self.gui.config['metadata_field_prompts'] = field_prompts
        
        # Advanced settings - get selected radio button value
        lang_behavior = 'auto'
        for button in self.lang_behavior_group.buttons():
            if button.isChecked():
                lang_behavior = button.property("value")
                break
        
        self.gui.config['lang_prompt_behavior'] = lang_behavior
        self.gui.config['forced_source_lang'] = self.forced_lang_entry.text()
        self.gui.config['output_language'] = self.output_lang_combo.currentText()
    
    def _reset_all_prompts_to_defaults(self):
        """Reset all prompts to default values"""
        # Remove prompt-related keys from config
        prompt_keys = [
            'book_title_system_prompt', 'book_title_prompt',
            'batch_header_system_prompt',  # NEW
            'batch_header_prompt', 'metadata_batch_prompt',
            'metadata_field_prompts', 'lang_prompt_behavior',
            'forced_source_lang', 'output_language'
        ]
        
        for key in prompt_keys:
            if key in self.gui.config:
                del self.gui.config[key]
        
        # Re-initialize defaults
        self._initialize_default_prompts()
        self.gui.save_config(show_message=False)
    
    def _detect_all_metadata_fields(self) -> Dict[str, str]:
        """Detect ALL metadata fields in the current EPUB"""
        metadata_fields = {}
        
        # Try different possible attribute names for the file path
        epub_path = None
        
        # Common patterns for file path in translator GUIs
        path_attributes = [
            'entry_epub',      # Most common
            'file_entry',      
            'epub_entry',
            'input_entry',
            'file_path_entry',
            'epub_path',
            'file_path',
            'input_file'
        ]
        
        for attr in path_attributes:
            if hasattr(self.gui, attr):
                widget = getattr(self.gui, attr)
                # Try Qt widget methods first
                if hasattr(widget, 'text'):
                    epub_path = widget.text()
                    break
                # Then try tkinter methods
                elif hasattr(widget, 'get'):
                    epub_path = widget.get()
                    break
                # Handle if it's a string directly
                elif isinstance(widget, str):
                    epub_path = widget
                    break
        
        if not epub_path:
            # Try to get from config or recent files
            if hasattr(self.gui, 'config') and 'last_epub_path' in self.gui.config:
                epub_path = self.gui.config.get('last_epub_path', '')
        
        if not epub_path or not epub_path.endswith('.epub'):
            # Return empty dict if no EPUB loaded
            return metadata_fields
        
        try:
            with zipfile.ZipFile(epub_path, 'r') as zf:
                for name in zf.namelist():
                    if name.lower().endswith('.opf'):
                        opf_content = zf.read(name)
                        soup = BeautifulSoup(opf_content, 'xml')
                        
                        # Get Dublin Core elements
                        dc_elements = ['title', 'creator', 'subject', 'description', 
                                      'publisher', 'contributor', 'date', 'type', 
                                      'format', 'identifier', 'source', 'language', 
                                      'relation', 'coverage', 'rights']
                        
                        for element in dc_elements:
                            tag = soup.find(element)
                            if tag and tag.get_text(strip=True):
                                metadata_fields[element] = tag.get_text(strip=True)
                        
                        # Get ALL meta tags
                        meta_tags = soup.find_all('meta')
                        for meta in meta_tags:
                            name = meta.get('name') or meta.get('property', '')
                            content = meta.get('content', '')
                            
                            if name and content:
                                # Clean calibre: prefix
                                if name.startswith('calibre:'):
                                    name = name[8:]
                                
                                metadata_fields[name] = content
                        
                        break
                        
        except Exception as e:
            self.gui.append_log(f"Error reading EPUB metadata: {e}")
        
        return metadata_fields

class BatchHeaderTranslator:
    """Translate chapter headers in batches"""
    
    def __init__(self, client, config: dict = None):
        self.client = client
        self.config = config or {}
        self.stop_flag = False
        
        # Use the batch_header_system_prompt, with fallback to env var or default
        self.system_prompt = (
            self.config.get('batch_header_system_prompt') or  # CHANGED: Use correct config key
            os.getenv('BATCH_HEADER_SYSTEM_PROMPT') or  # CHANGED: Use specific env var
            "You are a professional translator specializing in novel chapter titles. "
            "Respond with only the translated JSON, nothing else. "
            "Maintain the original tone and style while making titles natural in the target language."
        )
        
        # Get default batch size from config or environment
        self.default_batch_size = int(os.getenv('HEADERS_PER_BATCH', 
                                      self.config.get('headers_per_batch', '350')))
        
    def set_stop_flag(self, flag: bool):
        self.stop_flag = flag
        
    def translate_and_save_headers(self,
                                  html_dir: str,
                                  headers_dict: Dict[int, str], 
                                  batch_size: int = None,  # Changed from hardcoded 500
                                  output_dir: str = None,
                                  update_html: bool = True,
                                  save_to_file: bool = True,
                                  current_titles: Dict[int, Dict[str, str]] = None) -> Dict[int, str]:
        """Translate headers with optional file output and HTML updates
        
        Args:
            html_dir: Directory containing HTML files
            headers_dict: Dict mapping chapter numbers to source titles
            batch_size: Number of titles to translate in one API call (uses config if not specified)
            output_dir: Directory for saving translation file
            update_html: Whether to update HTML files
            save_to_file: Whether to save translations to file
            current_titles: Dict mapping chapter numbers to {'title': str, 'filename': str}
        """
        # Use configured batch size if not explicitly provided
        if batch_size is None:
            batch_size = int(os.getenv('HEADERS_PER_BATCH', str(self.default_batch_size)))
            print(f"[DEBUG] Using headers_per_batch from GUI/env: {batch_size}")
        
        # Translate headers
        translated_headers = self.translate_headers_batch(
            headers_dict, batch_size
        )
        
        if not translated_headers:
            return {}
        
        # Save to file if requested
        if save_to_file:
            if output_dir is None:
                output_dir = html_dir
            translations_file = os.path.join(output_dir, "translated_headers.txt")
            self._save_translations_to_file(headers_dict, translated_headers, translations_file)
        
        # Update HTML files if requested
        if update_html:
            if current_titles:
                # Use exact replacement method
                self._update_html_headers_exact(html_dir, translated_headers, current_titles)
            else:
                # Fallback to pattern-based method
                self._update_html_headers(html_dir, translated_headers)
        
        return translated_headers
        
    def translate_headers_batch(self, headers_dict: Dict[int, str], batch_size: int = None) -> Dict[int, str]:
        """Translate headers in batches using configured prompts with parallel execution"""
        if not headers_dict:
            return {}
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        # Import tiktoken for token counting
        try:
            import tiktoken
            # Try to use model-specific encoding
            try:
                model_name = self.client.model if hasattr(self.client, 'model') else 'gpt-3.5-turbo'
                enc = tiktoken.encoding_for_model(model_name)
            except:
                # Fallback to cl100k_base encoding
                enc = tiktoken.get_encoding("cl100k_base")
            has_tiktoken = True
        except ImportError:
            has_tiktoken = False
            print("[DEBUG] tiktoken not available, using character-based estimation")
        
        def count_tokens(text: str) -> int:
            """Count tokens in text"""
            if has_tiktoken and enc:
                return len(enc.encode(text))
            else:
                # Fallback: estimate ~4 characters per token
                return max(1, len(text) // 4)
        
        # Get configured prompt template
        prompt_template = self.config.get('batch_header_prompt',
            "Translate these chapter titles to English.\n"
            "Return ONLY a JSON object with chapter numbers as keys.\n"
            "Format: {\"1\": \"translated title\", \"2\": \"translated title\"}")
        
        # Handle language in prompt
        source_lang = _get_source_language()
        lang_behavior = self.config.get('lang_prompt_behavior', 'auto')
        
        if lang_behavior == 'never':
            lang_str = ""
        elif lang_behavior == 'always':
            lang_str = self.config.get('forced_source_lang', 'Korean')
        else:  # auto
            lang_str = source_lang if source_lang else ""
        
        # Handle output language
        output_lang = self.config.get('output_language', 'English')
        
        # Replace variables in prompt
        prompt_template = prompt_template.replace('{source_lang}', lang_str)
        prompt_template = prompt_template.replace('English', output_lang)
        
        # Add the titles to translate
        user_prompt_template = prompt_template + "\n\nTitles to translate:\n"
        
        sorted_headers = sorted(headers_dict.items())
        all_translations = {}
        total_batches = (len(sorted_headers) + batch_size - 1) // batch_size
        
        # Get temperature and max_tokens from environment (passed by GUI) or config as fallback
        temperature = float(os.getenv('TRANSLATION_TEMPERATURE', self.config.get('temperature', 0.3)))
        max_tokens = int(os.getenv('MAX_OUTPUT_TOKENS', self.config.get('max_tokens', 12000)))
        
        print(f"[DEBUG] Using temperature: {temperature}, max_tokens: {max_tokens} (from GUI/env)")
        
        # Count system prompt tokens once
        system_tokens = count_tokens(self.system_prompt)
        print(f"[DEBUG] System prompt tokens: {system_tokens}")
        
        # Determine max workers from config or environment variable
        # Use extraction_workers setting, with fallback to default of 3
        configured_workers = int(os.getenv('EXTRACTION_WORKERS', self.config.get('extraction_workers', 3)))
        # Don't create more workers than batches, and cap at configured limit
        max_workers = min(configured_workers, total_batches)
        print(f"[DEBUG] Using ThreadPoolExecutor with {max_workers} workers for {total_batches} batches (configured: {configured_workers})")
        
        # Thread-safe lock for updating all_translations
        translations_lock = threading.Lock()
        
        def translate_batch(batch_num: int, batch_headers: Dict[int, str]):
            """Translate a single batch - thread-safe function"""
            if self.stop_flag:
                return None
            
            try:
                titles_json = json.dumps(batch_headers, ensure_ascii=False, indent=2)
                user_prompt = user_prompt_template + titles_json
                
                # Count tokens in the user prompt
                user_tokens = count_tokens(user_prompt)
                total_input_tokens = system_tokens + user_tokens
                
                # Debug output showing input tokens
                print(f"\n📚 Translating header batch {batch_num + 1}/{total_batches}")
                print(f"[DEBUG] Batch {batch_num + 1} input tokens:")
                print(f"  - User prompt: {user_tokens} tokens")
                print(f"  - Total input: {total_input_tokens} tokens (including system prompt)")
                print(f"  - Headers in batch: {len(batch_headers)}")
                
                # Show a sample of the headers being translated (first 3)
                sample_headers = list(batch_headers.items())[:3]
                if sample_headers:
                    print(f"[DEBUG] Sample headers being sent:")
                    for ch_num, title in sample_headers:
                        print(f"    Chapter {ch_num}: {title}")
                    if len(batch_headers) > 3:
                        print(f"    ... and {len(batch_headers) - 3} more")
                
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                # Pass temperature and max_tokens explicitly
                response = self.client.send(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    context='batch_header_translation'
                )
                
                # Extract content from response - handle both object and tuple formats
                response_content = None
                if hasattr(response, 'content'):
                    response_content = response.content
                elif isinstance(response, tuple):
                    # If it's a tuple, first element is usually the content
                    response_content = response[0] if response else ""
                else:
                    # Fallback: convert to string
                    response_content = str(response)
                
                if response_content:
                    translations = self._parse_json_response(response_content, batch_headers)
                    
                    # Count output tokens for debug
                    output_tokens = count_tokens(response_content)
                    print(f"[DEBUG] Batch {batch_num + 1} response tokens: {output_tokens}")
                    
                    for num, translated in translations.items():
                        if num in batch_headers:
                            print(f"  ✓ Ch{num}: {batch_headers[num]} → {translated}")
                    
                    return translations
                else:
                    print(f"  ⚠️ Batch {batch_num + 1}: Empty response from API")
                    return {}
                    
            except json.JSONDecodeError as e:
                print(f"  ❌ Batch {batch_num + 1}: Failed to parse JSON response: {e}")
                # Try to extract translations manually from the response
                if response_content:
                    return self._fallback_parse(response_content, batch_headers)
                return {}
            except Exception as e:
                print(f"  ❌ Batch {batch_num + 1}: Error: {e}")
                return {}
        
        # Create batches
        batch_tasks = []
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(sorted_headers))
            batch_headers = dict(sorted_headers[start_idx:end_idx])
            batch_tasks.append((batch_num, batch_headers))
        
        # Execute batches in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="HeaderBatch") as executor:
            # Submit all tasks
            future_to_batch = {executor.submit(translate_batch, batch_num, headers): batch_num 
                             for batch_num, headers in batch_tasks}
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_batch):
                if self.stop_flag:
                    print("\nTranslation interrupted by user")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                
                batch_num = future_to_batch[future]
                try:
                    translations = future.result()
                    if translations:
                        with translations_lock:
                            all_translations.update(translations)
                    completed += 1
                    print(f"\n[Progress] Completed {completed}/{total_batches} batches")
                except Exception as e:
                    print(f"\n❌ Batch {batch_num + 1} generated exception: {e}")
                    completed += 1
        
        print(f"\n✅ Translated {len(all_translations)} headers total (using parallel execution)")
        return all_translations
    
    def _parse_json_response(self, response: str, original_headers: Dict[int, str]) -> Dict[int, str]:
        """Parse JSON response from API"""
        try:
            response = response.strip()
            
            # Remove markdown blocks
            if response.startswith("```"):
                lines = response.split('\n')
                response_lines = []
                in_code_block = False
                
                for line in lines:
                    if line.strip().startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        response_lines.append(line)
                
                response = '\n'.join(response_lines)
            
            parsed = json.loads(response)
            
            result = {}
            for key, value in parsed.items():
                try:
                    chapter_num = int(key)
                    if chapter_num in original_headers:
                        result[chapter_num] = str(value).strip()
                except (ValueError, TypeError):
                    continue
                    
            return result
            
        except json.JSONDecodeError:
            return self._fallback_parse(response, original_headers)
        except Exception:
            return {}
    
    def _fallback_parse(self, response: str, original_headers: Dict[int, str]) -> Dict[int, str]:
        """Fallback parsing if JSON fails"""
        result = {}
        pattern = r'["\']?(\d+)["\']?\s*:\s*["\']([^"\']+)["\']'
        
        for match in re.finditer(pattern, response):
            try:
                num = int(match.group(1))
                title = match.group(2).strip()
                if num in original_headers and title:
                    result[num] = title
            except:
                continue
                
        return result
    
    def _save_translations_to_file(self, 
                                  original: Dict[int, str], 
                                  translated: Dict[int, str],
                                  output_path: str):
        """Save translations to text file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("Chapter Header Translations\n")
                f.write("=" * 50 + "\n\n")
                
                # Sort chapter numbers, ensuring chapter 0 comes first if present
                chapter_numbers = sorted(original.keys())
                
                # Summary info
                total_chapters = len(original)
                successfully_translated = len(translated)
                
                # Check if we have chapter 0
                has_chapter_zero = 0 in chapter_numbers
                if has_chapter_zero:
                    f.write(f"Note: This novel uses 0-based chapter numbering (starts with Chapter 0)\n")
                    f.write("-" * 50 + "\n\n")
                
                # Write each chapter's translation
                for num in chapter_numbers:
                    orig_title = original.get(num, "Unknown")
                    trans_title = translated.get(num, orig_title)
                    
                    f.write(f"Chapter {num}:\n")
                    f.write(f"  Original:   {orig_title}\n")
                    f.write(f"  Translated: {trans_title}\n")
                    
                    # Mark if translation failed for this chapter
                    if num not in translated:
                        f.write(f"  Status:     ⚠️ Using original (translation failed)\n")
                    
                    f.write("-" * 40 + "\n")
                
                # Summary at the end
                f.write(f"\nSummary:\n")
                f.write(f"Total chapters: {total_chapters}\n")
                f.write(f"Chapter range: {min(chapter_numbers)} to {max(chapter_numbers)}\n")
                f.write(f"Successfully translated: {successfully_translated}\n")
                
                if successfully_translated < total_chapters:
                    failed_chapters = [num for num in original if num not in translated]
                    f.write(f"Failed chapters: {', '.join(map(str, failed_chapters))}\n")
                
            print(f"✅ Saved translations to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving translations: {e}")
    
    def _check_html_has_header(self, html_path: str) -> tuple:
        """Check if HTML file has any header tags (h1-h6)
        
        Returns:
            tuple: (has_header, soup) where has_header is bool and soup is BeautifulSoup object
        """
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'lxml')
            
            # Check ONLY for header tags (h1-h6)
            # NOT checking title tag per user requirement
            header_tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
            has_header = bool(header_tags)
            
            if not has_header:
                print(f"📝 {os.path.basename(html_path)} has no header tags (h1-h6)")
            
            return has_header, soup
            
        except Exception as e:
            print(f"❌ Error checking HTML structure for {html_path}: {e}")
            return True, None  # Assume it has header to avoid accidental overwrites
    
    def _insert_header_into_html(self, soup, new_title: str, preferred_tag: str = 'h1') -> bool:
        """Insert a header into HTML that lacks one
        
        Args:
            soup: BeautifulSoup object of the HTML
            new_title: The translated title to insert
            preferred_tag: The header tag to use (h1, h2, h3, etc.)
            
        Returns:
            bool: True if successfully inserted, False otherwise
        """
        try:
            # Find or create body tag
            body = soup.find('body')
            
            # If no body tag exists, try to find the main content area
            if not body:
                # Sometimes the HTML might not have a proper body tag
                # Look for the html tag instead
                html_tag = soup.find('html')
                if html_tag:
                    # Create a body tag
                    body = soup.new_tag('body')
                    # Move all content from html to body
                    for child in list(html_tag.children):
                        if child.name != 'head':
                            body.append(child.extract())
                    html_tag.append(body)
                else:
                    # Last resort: treat the entire soup as the body
                    body = soup
            
            if body:
                # Create new header tag with the translated title
                header_tag = soup.new_tag(preferred_tag)
                header_tag.string = new_title
                
                # Add some styling to make it stand out (optional)
                # header_tag['style'] = 'text-align: center; margin: 1em 0;'
                
                # Find the best insertion point
                insertion_done = False
                
                # Strategy 1: Insert after any <head> or <meta> tags at the beginning
                first_content = None
                for child in body.children:
                    # Skip whitespace text nodes
                    if hasattr(child, 'name'):
                        if child.name and child.name not in ['script', 'style', 'link', 'meta']:
                            first_content = child
                            break
                    elif isinstance(child, str) and child.strip():
                        # Non-empty text node
                        first_content = child
                        break
                
                if first_content:
                    # Insert before the first content element
                    first_content.insert_before(header_tag)
                    insertion_done = True
                    print(f"✓ Inserted {preferred_tag} before first content element")
                else:
                    # Body appears to be empty or contains only scripts/styles
                    # Insert at the beginning of body
                    if len(list(body.children)) > 0:
                        # Body has some children (maybe scripts/styles)
                        # Insert at position 0
                        body.insert(0, header_tag)
                    else:
                        # Body is completely empty
                        body.append(header_tag)
                    insertion_done = True
                    print(f"✓ Inserted {preferred_tag} at beginning of body")
                
                # Also add a line break after the header for better formatting
                if insertion_done:
                    br_tag = soup.new_tag('br')
                    header_tag.insert_after(br_tag)
                    
                    # Optional: Also add a horizontal rule for visual separation
                    # hr_tag = soup.new_tag('hr')
                    # br_tag.insert_after(hr_tag)
                
                print(f"✓ Successfully added {preferred_tag} tag with: '{new_title}'")
                return True
                
            else:
                print(f"⚠️ Could not find or create body tag in HTML")
                
                # Fallback: Try to insert at the root level
                # Create the header tag
                header_tag = soup.new_tag(preferred_tag)
                header_tag.string = new_title
                
                # Find first element that's not a DOCTYPE or processing instruction
                first_element = None
                for item in soup.contents:
                    if hasattr(item, 'name') and item.name:
                        first_element = item
                        break
                
                if first_element:
                    first_element.insert_before(header_tag)
                    print(f"✓ Added {preferred_tag} tag at root level with: '{new_title}'")
                    return True
                else:
                    # Last resort: append to soup
                    soup.append(header_tag)
                    print(f"✓ Appended {preferred_tag} tag to document with: '{new_title}'")
                    return True
                    
        except Exception as e:
            print(f"❌ Error inserting header: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _update_html_headers_exact(self, html_dir: str, translated_headers: Dict[int, str], 
                                  current_titles: Dict[int, Dict[str, str]]):
        """Update HTML files by replacing exact current titles with translations
        Also handles HTML files without headers by adding them.
        """
        updated_count = 0
        added_count = 0
        
        for num, new_title in translated_headers.items():
            if num not in current_titles:
                print(f"⚠️ No HTML file mapping for chapter {num}")
                continue
                
            current_info = current_titles[num]
            current_title = current_info['title']
            html_file = current_info['filename']
            html_path = os.path.join(html_dir, html_file)
            
            try:
                # Check if HTML has a header
                has_header, soup = self._check_html_has_header(html_path)
                
                if not soup:
                    print(f"⚠️ Could not parse {html_file}")
                    continue
                
                if not has_header:
                    # HTML has no header, insert the translated one WITHOUT rewriting the rest of the file
                    print(f"📝 Adding header to {html_file}: '{new_title}'")
                    header_html = f"<h1>{new_title}</h1><br/>"
                    with open(html_path, 'r', encoding='utf-8') as rf:
                        original = rf.read()
                    import re as _re
                    new_content, n = _re.subn(r'(\<body\b[^>]*\>)', r"\1" + header_html, original, count=1, flags=_re.IGNORECASE)
                    if n == 0:
                        # Fallback: prepend at the very beginning if <body> not found
                        new_content = header_html + "\n" + original
                    with open(html_path, 'w', encoding='utf-8') as wf:
                        wf.write(new_content)
                    added_count += 1
                    print(f"✓ Added header to {html_file}")
                else:
                    # HTML has header, update it WITHOUT reparsing/rewriting the rest of the file
                    updated = False
                    import re as _re
                    with open(html_path, 'r', encoding='utf-8') as rf:
                        original = rf.read()
                    content = original
                    # Replace title text only if it exactly matches current_title
                    def _repl_title(m):
                        inner = m.group(2).strip()
                        return m.group(1) + new_title + m.group(3) if inner == current_title else m.group(0)
                    content2 = _re.sub(r'(<title[^>]*>)([\s\S]*?)(</title>)', _repl_title, content, count=1, flags=_re.IGNORECASE)
                    if content2 != content:
                        updated = True
                        content = content2
                    # Replace exact text inside first h1/h2/h3/p/div/span if it equals current_title
                    for tag in ['h1', 'h2', 'h3', 'p', 'div', 'span']:
                        pattern = rf'(<{tag}[^>]*>)\s*{_re.escape(current_title)}\s*(</{tag}>)'
                        content2, n = _re.subn(pattern, r'\1' + new_title + r'\2', content, count=1, flags=_re.IGNORECASE)
                        if n > 0:
                            updated = True
                            content = content2
                            break
                    # Update meta og:title content if it matches
                    def _repl_meta(m):
                        before, val, after = m.group(1), m.group(2), m.group(3)
                        return before + new_title + after if val.strip() == current_title else m.group(0)
                    content2 = _re.sub(r'(<meta\b[^>]*property=["\']og:title["\'][^>]*content=["\'])([^"\']*)(["\'][^>]*>)', _repl_meta, content, count=1, flags=_re.IGNORECASE)
                    if content2 != content:
                        updated = True
                        content = content2
                    if updated:
                        with open(html_path, 'w', encoding='utf-8') as wf:
                            wf.write(content)
                        updated_count += 1
                        print(f"✓ Updated {html_file}: '{current_title}' → '{new_title}'")
                    else:
                        print(f"⚠️ Could not find '{current_title}' in {html_file}")
                    
            except Exception as e:
                print(f"❌ Error processing {html_file}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n📝 Updated {updated_count} HTML files, added headers to {added_count} files")
    
    def _update_html_headers(self, html_dir: str, translated_headers: Dict[int, str]):
        """Fallback: Update HTML files with translated headers using pattern matching
        Also handles HTML files without headers by adding them.
        """
        updated_count = 0
        added_count = 0
        
        # Get all HTML files in directory (support all HTML extensions)
        html_extensions = ('.html', '.xhtml', '.htm')
        all_html_files = [f for f in os.listdir(html_dir) if f.lower().endswith(html_extensions)]
        
        for num, new_title in translated_headers.items():
            # Try multiple filename patterns
            possible_patterns = [
                f"response_{num}_",  # Standard pattern
                f"response_{num}.",  # With dot
                f"chapter_{num}_",   # Alternative pattern
                f"chapter{num}_",    # Without underscore
                f"{num}_",           # Just number
                f"{num}.",           # Number with dot
                f"ch{num}_",         # Abbreviated
                f"ch_{num}_",        # Abbreviated with underscore
            ]
            
            html_file = None
            for pattern in possible_patterns:
                matching_files = [f for f in all_html_files if f.startswith(pattern)]
                if matching_files:
                    html_file = matching_files[0]
                    break
            
            if not html_file:
                # Last resort: check if any file contains the chapter number
                for f in all_html_files:
                    if re.search(rf'\b{num}\b', f):
                        html_file = f
                        break
            
            if not html_file:
                print(f"⚠️ No HTML file found for chapter {num}")
                continue
                
            html_path = os.path.join(html_dir, html_file)
            
            try:
                # Check if HTML has a header
                has_header, soup = self._check_html_has_header(html_path)
                
                if not soup:
                    print(f"⚠️ Could not parse {html_file}")
                    continue
                
                if not has_header:
                    # HTML has no header, insert the translated one WITHOUT rewriting the rest of the file
                    print(f"📝 Adding header to {html_file}: '{new_title}'")
                    header_html = f"<h1>{new_title}</h1><br/>"
                    with open(html_path, 'r', encoding='utf-8') as rf:
                        original = rf.read()
                    import re as _re
                    new_content, n = _re.subn(r'(\<body\b[^>]*\>)', r"\1" + header_html, original, count=1, flags=_re.IGNORECASE)
                    if n == 0:
                        # Fallback: prepend at the top if <body> not found
                        new_content = header_html + "\n" + original
                    with open(html_path, 'w', encoding='utf-8') as wf:
                        wf.write(new_content)
                    added_count += 1
                    print(f"✓ Added header to {html_file}")
                else:
                    # HTML has header, update it WITHOUT reparsing/rewriting the rest of the file
                    updated = False
                    import re as _re
                    with open(html_path, 'r', encoding='utf-8') as rf:
                        original = rf.read()
                    content = original
                    # Replace title text (first <title>)
                    content2, n = _re.subn(r'(<title[^>]*>)[\s\S]*?(</title>)', r'\1' + new_title + r'\2', content, count=1, flags=_re.IGNORECASE)
                    if n > 0:
                        updated = True
                        content = content2
                    # Replace first h1/h2/h3
                    for tag in ['h1', 'h2', 'h3']:
                        content2, n = _re.subn(rf'(<{tag}[^>]*>)[\s\S]*?(</{tag}>)', r'\1' + new_title + r'\2', content, count=1, flags=_re.IGNORECASE)
                        if n > 0:
                            updated = True
                            content = content2
                            break
                    # Update or add meta og:title (set content=...)
                    # Replace existing
                    content2, n = _re.subn(r'(<meta\b[^>]*property=["\']og:title["\'][^>]*content=["\'])[^"\']*(["\'][^>]*>)', r'\1' + new_title + r'\2', content, count=1, flags=_re.IGNORECASE)
                    if n == 0:
                        # Try to add if missing: insert after <head>
                        content2, n2 = _re.subn(r'(<head\b[^>]*>)', r"\1\n<meta property=\"og:title\" content=\"" + new_title + "\"/>", content, count=1, flags=_re.IGNORECASE)
                        if n2 > 0:
                            updated = True
                            content = content2
                    else:
                        updated = True
                        content = content2
                    if updated:
                        with open(html_path, 'w', encoding='utf-8') as wf:
                            wf.write(content)
                        updated_count += 1
                        print(f"✓ Updated {html_file} with: {new_title}")
            except Exception as e:
                print(f"❌ Error processing chapter {num}: {e}")
        
        print(f"\n📝 Updated {updated_count} HTML files, added headers to {added_count} files")

class MetadataTranslator:
    """Translate EPUB metadata fields"""
    
    def __init__(self, client, config: dict = None):
        self.client = client
        self.config = config or {}
        self.system_prompt = os.getenv('BOOK_TITLE_SYSTEM_PROMPT',
            "You are a translator. Respond with only the translated text, nothing else.")
    
    def translate_metadata(self, 
                          metadata: Dict[str, Any],
                          fields_to_translate: Dict[str, bool],
                          mode: str = 'together') -> Dict[str, Any]:
        """Translate selected metadata fields"""
        if not any(fields_to_translate.values()):
            return metadata
            
        translated_metadata = metadata.copy()
        
        if mode == 'together':
            translated_fields = self._translate_fields_together(
                metadata, fields_to_translate
            )
            translated_metadata.update(translated_fields)
        else:
            translated_fields = self._translate_fields_parallel(
                metadata, fields_to_translate
            )
            translated_metadata.update(translated_fields)
            
        return translated_metadata
 
    def _is_already_english(self, text: str) -> bool:
        """Simple check if text is already in English"""
        if not text:
            return True
        
        # Check for CJK characters - if present, needs translation
        for char in text:
            if ('\u4e00' <= char <= '\u9fff' or  # Chinese
                '\u3040' <= char <= '\u309f' or  # Hiragana
                '\u30a0' <= char <= '\u30ff' or  # Katakana
                '\uac00' <= char <= '\ud7af'):   # Korean
                return False
        
        # If no CJK characters, assume it's already English
        return True
 
    def _translate_fields_together(self, 
                                  metadata: Dict[str, Any],
                                  fields_to_translate: Dict[str, bool]) -> Dict[str, Any]:
        """Translate all fields in one API call"""
        fields_to_send = {}
        
        for field, should_translate in fields_to_translate.items():
            if should_translate and field in metadata and metadata[field]:
                if not self._is_already_english(metadata[field]):
                    fields_to_send[field] = metadata[field]
                
        if not fields_to_send:
            return {}
        
        # Get configured prompt
        prompt_template = self.config.get('metadata_batch_prompt',
            "Translate the following metadata fields to English.\n"
            "Return ONLY a JSON object with the same field names as keys.")
        
        # Handle language behavior
        lang_behavior = self.config.get('lang_prompt_behavior', 'auto')
        source_lang = _get_source_language()
        
        if lang_behavior == 'never':
            lang_str = ""
        elif lang_behavior == 'always':
            lang_str = self.config.get('forced_source_lang', 'Korean')
        else:  # auto
            lang_str = source_lang if source_lang else ""
        
        # Handle output language
        output_lang = self.config.get('output_language', 'English')
        
        # Replace variables
        prompt_template = prompt_template.replace('{source_lang}', lang_str)
        prompt_template = prompt_template.replace('English', output_lang)
        
        user_prompt = prompt_template + f"\n\nFields to translate:\n{json.dumps(fields_to_send, ensure_ascii=False, indent=2)}"
        
        # Check if we're using a translation service (not AI)
        client_type = getattr(self.client, 'client_type', '')
        is_translation_service = client_type in ['deepl', 'google_translate']
        
        try:
            if is_translation_service:
                # For translation services, send only the field values without AI prompts
                print(f"🌐 Using translation service ({client_type}) - sending fields directly")
                # Convert fields to a simple text format
                field_text = "\n".join([f"{field}: {value}" for field, value in fields_to_send.items()])
                messages = [
                    {"role": "user", "content": field_text}
                ]
            else:
                # For AI services, use prompts as before
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            
            # Get temperature and max_tokens from environment or config
            temperature = float(os.getenv('TRANSLATION_TEMPERATURE', self.config.get('temperature', 0.3)))
            max_tokens = int(os.getenv('MAX_OUTPUT_TOKENS', self.config.get('max_tokens', 4096)))
            
            response = self.client.send(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                context='metadata_translation'
            )
            
            # Extract content from response - handle both object and tuple formats
            response_content = None
            if hasattr(response, 'content'):
                response_content = response.content
            elif isinstance(response, tuple):
                # If it's a tuple, first element is usually the content
                response_content = response[0] if response else ""
            else:
                # Fallback: convert to string
                response_content = str(response)
            
            if response_content:
                translated = self._parse_metadata_response(response_content, fields_to_send)
                
                for field, value in translated.items():
                    if field in metadata:
                        print(f"✔ Translated {field}: {metadata[field]} → {value}")
                        
                return translated
            else:
                print("⚠️ Empty response from API")
                return {}
            
        except Exception as e:
            print(f"Error translating metadata: {e}")
            return {}
    
    def _translate_fields_parallel(self,
                                  metadata: Dict[str, Any],
                                  fields_to_translate: Dict[str, bool]) -> Dict[str, Any]:
        """Translate fields in parallel"""
        fields_to_process = [
            (field, value) for field, should_translate in fields_to_translate.items()
            if should_translate and (value := metadata.get(field)) and not self._is_already_english(value)
        ]
        
        if not fields_to_process:
            return {}
            
        translated = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            
            for field, value in fields_to_process:
                future = executor.submit(self._translate_single_field, field, value)
                futures[future] = field
                
            for future in futures:
                field = futures[future]
                try:
                    result = future.result(timeout=30)
                    if result:
                        translated[field] = result
                        print(f"✓ Translated {field}: {metadata.get(field)} → {result}")
                except Exception as e:
                    print(f"❌ Error translating {field}: {e}")
                    
        return translated
    
    def _translate_single_field(self, field_name: str, field_value: str) -> Optional[str]:
        """Translate a single field using configured prompts"""
        if self._is_already_english(field_value):
            return field_value
        # Get field-specific prompts
        field_prompts = self.config.get('metadata_field_prompts', {})
        
        # Get the specific prompt or default
        prompt_template = field_prompts.get(field_name, 
                                           field_prompts.get('_default', 
                                                           "Translate this text to English:"))
        
        # Handle language behavior
        lang_behavior = self.config.get('lang_prompt_behavior', 'auto')
        source_lang = _get_source_language()
        
        if lang_behavior == 'never':
            lang_str = ""
        elif lang_behavior == 'always':
            lang_str = self.config.get('forced_source_lang', 'Korean')
        else:  # auto
            lang_str = source_lang if source_lang else ""
        
        # Handle output language
        output_lang = self.config.get('output_language', 'English')
        
        # Replace variables
        prompt = prompt_template.replace('{source_lang}', lang_str)
        prompt = prompt.replace('{field_value}', field_value)
        prompt = prompt.replace('English', output_lang)
        
        # Clean up double spaces
        prompt = ' '.join(prompt.split())
        
        # Check if we're using a translation service (not AI)
        client_type = getattr(self.client, 'client_type', '')
        is_translation_service = client_type in ['deepl', 'google_translate']
        
        try:
            if is_translation_service:
                # For translation services, send only the field value without AI prompts
                messages = [
                    {"role": "user", "content": field_value}
                ]
            else:
                # For AI services, use prompts as before
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"{prompt}\n\n{field_value}"}
                ]
            
            # Get temperature and max_tokens from environment or config
            temperature = float(os.getenv('TRANSLATION_TEMPERATURE', self.config.get('temperature', 0.3)))
            max_tokens = int(os.getenv('MAX_OUTPUT_TOKENS', self.config.get('max_tokens', 4096)))
            
            response = self.client.send(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                context='metadata_field_translation'
            )
            
            # Extract content from response - handle both object and tuple formats
            response_content = None
            if hasattr(response, 'content'):
                response_content = response.content
            elif isinstance(response, tuple):
                # If it's a tuple, first element is usually the content
                response_content = response[0] if response else ""
            else:
                # Fallback: convert to string
                response_content = str(response)
            
            if response_content:
                return response_content.strip()
            else:
                print(f"⚠️ Empty response when translating {field_name}")
                return None
            
        except Exception as e:
            print(f"Error translating {field_name}: {e}")
            return None
    
    def _parse_metadata_response(self, response: str, original_fields: Dict[str, str]) -> Dict[str, str]:
        """Parse metadata response"""
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.split("```")[0]
                
            parsed = json.loads(response.strip())
            
            result = {}
            for field, value in parsed.items():
                if field in original_fields and value:
                    result[field] = str(value).strip()
                    
            return result
            
        except Exception as e:
            print(f"Error parsing response: {e}")
            return {}


def _get_source_language() -> str:
    """
    Get source language from EPUB metadata or detect from content
    
    NOTE: This properly detects language from the actual content/metadata
    instead of inferring from profile names (which are for prompt presets)
    """
    # Try to get language from metadata first
    metadata_path = os.path.join(os.path.dirname(os.getenv('EPUB_PATH', '')), 'metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                # Check for language field
                lang = metadata.get('language', '').lower()
                if 'ko' in lang or 'korean' in lang:
                    return 'Korean'
                elif 'ja' in lang or 'japanese' in lang:
                    return 'Japanese'
                elif 'zh' in lang or 'chinese' in lang:
                    return 'Chinese'
        except:
            pass
    
    # If no metadata, return empty string so prompts are generic
    return ''


def enhance_epub_compiler(compiler_instance):
    """Enhance an EPUBCompiler instance with translation features"""
    # Get settings from environment
    translate_metadata_fields = {}
    try:
        fields_str = os.getenv('TRANSLATE_METADATA_FIELDS', '{}')
        translate_metadata_fields = json.loads(fields_str)
        if translate_metadata_fields:
            print(f"[DEBUG] Metadata fields to translate: {translate_metadata_fields}")
        else:
            print("[DEBUG] No metadata fields configured for translation")
    except Exception as e:
        print(f"[ERROR] Failed to parse TRANSLATE_METADATA_FIELDS: {e}")
        translate_metadata_fields = {}
    
    batch_translate = os.getenv('BATCH_TRANSLATE_HEADERS', '0') == '1'
    headers_per_batch = int(os.getenv('HEADERS_PER_BATCH', 350))
    update_html = os.getenv('UPDATE_HTML_HEADERS', '1') == '1'
    save_translations = os.getenv('SAVE_HEADER_TRANSLATIONS', '1') == '1'
    
    # Add settings to compiler
    compiler_instance.translate_metadata_fields = translate_metadata_fields
    compiler_instance.metadata_translation_mode = os.getenv('METADATA_TRANSLATION_MODE', 'together')
    compiler_instance.batch_translate_headers = batch_translate
    compiler_instance.headers_per_batch = headers_per_batch
    compiler_instance.update_html_headers = update_html
    compiler_instance.save_header_translations = save_translations
    
    # Log what we're setting
    print(f"[DEBUG] Compiler settings:")
    print(f"  - translate_metadata_fields: {compiler_instance.translate_metadata_fields}")
    print(f"  - metadata_translation_mode: {compiler_instance.metadata_translation_mode}")
    print(f"  - batch_translate_headers: {compiler_instance.batch_translate_headers}")
    
    # extraction method with mapping support
    compiler_instance._extract_source_headers_and_current_titles = lambda: extract_source_headers_and_current_titles(
        os.getenv('EPUB_PATH', ''), 
        compiler_instance.html_dir,
        compiler_instance.log
    )
    
    # Create translators if needed
    needs_translators = batch_translate or any(translate_metadata_fields.values())
    print(f"[DEBUG] Needs translators: {needs_translators} (batch={batch_translate}, metadata={any(translate_metadata_fields.values())})")
    
    if compiler_instance.api_client and needs_translators:
        # Try to get config from multiple locations
        config_paths = [
            os.path.join(compiler_instance.base_dir, '..', 'config.json'),
            os.path.join(os.path.dirname(compiler_instance.base_dir), 'config.json'),
            os.path.join(os.path.dirname(os.path.dirname(compiler_instance.base_dir)), 'config.json'),
            'config.json'  # Current directory as last resort
        ]
        
        config = {}
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    print(f"[DEBUG] Loaded config from: {config_path}")
                    break
                except Exception as e:
                    print(f"[WARNING] Failed to load config from {config_path}: {e}")
        
        # PRIORITY: Use GUI values from environment first, then config as fallback
        
        # Get temperature - GUI passes this via TRANSLATION_TEMPERATURE env var
        env_temp = os.getenv('TRANSLATION_TEMPERATURE')
        if env_temp:
            try:
                config['temperature'] = float(env_temp)
                print(f"[DEBUG] Using temperature from GUI (env): {config['temperature']}")
            except ValueError:
                print(f"[WARNING] Invalid temperature value: {env_temp}")
                config['temperature'] = 0.3
        elif 'translation_temperature' in config:
            config['temperature'] = config['translation_temperature']
            print(f"[DEBUG] Using temperature from config: {config['temperature']}")
        else:
            config['temperature'] = 0.3  # Last resort default
            print(f"[DEBUG] Using default temperature: {config['temperature']}")
            
        # Get max_tokens - GUI passes this via MAX_OUTPUT_TOKENS env var
        env_max_tokens = os.getenv('MAX_OUTPUT_TOKENS')
        if env_max_tokens and env_max_tokens.isdigit():
            config['max_tokens'] = int(env_max_tokens)
            print(f"[DEBUG] Using max_tokens from GUI (env): {config['max_tokens']}")
        elif 'max_output_tokens' in config:
            config['max_tokens'] = config['max_output_tokens']
            print(f"[DEBUG] Using max_tokens from config: {config['max_tokens']}")
        else:
            config['max_tokens'] = 4096  # Last resort default
            print(f"[DEBUG] Using default max_tokens: {config['max_tokens']}")
            
        # Set temperature and max_tokens on the client if possible
        if hasattr(compiler_instance.api_client, 'default_temperature'):
            compiler_instance.api_client.default_temperature = config['temperature']
        if hasattr(compiler_instance.api_client, 'default_max_tokens'):
            compiler_instance.api_client.default_max_tokens = config['max_tokens']
        
        # Get compression factor from environment or config
        compression_factor = float(os.getenv('COMPRESSION_FACTOR', '1.0'))
        if hasattr(compiler_instance.api_client, 'compression_factor'):
            compiler_instance.api_client.compression_factor = compression_factor
            print(f"[DEBUG] Set compression factor: {compression_factor}")
        
        try:
            # Create batch header translator if needed
            if batch_translate:
                compiler_instance.header_translator = BatchHeaderTranslator(
                    compiler_instance.api_client, config
                )
                print(f"[DEBUG] Created BatchHeaderTranslator")
            
            # Create metadata translator if needed
            if any(translate_metadata_fields.values()):
                compiler_instance.metadata_translator = MetadataTranslator(
                    compiler_instance.api_client, config
                )
                print(f"[DEBUG] Created MetadataTranslator for fields: {[k for k, v in translate_metadata_fields.items() if v]}")
                
                # Verify the translator was created
                if hasattr(compiler_instance, 'metadata_translator'):
                    print("[DEBUG] MetadataTranslator successfully attached to compiler")
                else:
                    print("[ERROR] MetadataTranslator not attached to compiler!")
            else:
                print("[DEBUG] No metadata fields selected for translation")
                
        except Exception as e:
            print(f"[ERROR] Failed to initialize translators: {e}")
            import traceback
            traceback.print_exc()
    else:
        if not compiler_instance.api_client:
            print("[WARNING] No API client available for translation")
        if not needs_translators:
            print("[DEBUG] No translation features requested")
    
    return compiler_instance
    
def extract_source_headers_and_current_titles(epub_path: str, html_dir: str, log_callback=None) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Extract source headers AND current titles from HTML files using STRICT OPF ordering
    
    Returns:
        Tuple of (source_headers, current_titles) where:
        - source_headers: Maps output chapter numbers to original language titles from source EPUB (in OPF order)
        - current_titles: Maps output chapter numbers to current titles in HTML files
    """
    from bs4 import BeautifulSoup
    import xml.etree.ElementTree as ET
    
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)
    
    log("📖 Extracting headers and mapping to output files...")
    
    # Step 1: Get HTML files that contain numbers (support all HTML extensions)
    html_extensions = ('.html', '.xhtml', '.htm')
    all_html_files = sorted([f for f in os.listdir(html_dir) 
                             if f.lower().endswith(html_extensions) 
                             and re.search(r'\d+', f)])  # Only files with numbers
    log(f"📁 Found {len(all_html_files)} HTML files with numbers in {html_dir}")
    
    # Step 2: Load translation_progress.json to understand the chapter mapping
    progress_file = os.path.join(html_dir, 'translation_progress.json')
    progress_data = {}
    chapter_to_file_map = {}
    has_chapter_zero = False
    uses_zero_based = False
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            uses_zero_based = progress_data.get('uses_zero_based', False)
            all_chapters = progress_data.get('chapters', {})
            
            log(f"📊 Scanning {len(all_chapters)} entries in translation_progress.json")
            log(f"📊 Novel uses {'0-based' if uses_zero_based else '1-based'} numbering")
            
            # Check if we have chapter 0
            for chapter_hash, chapter_info in all_chapters.items():
                if isinstance(chapter_info, dict):
                    actual_num = chapter_info.get('actual_num')
                    if actual_num == 0:
                        has_chapter_zero = True
                        log("  ✔ Found Chapter 0 in translation_progress.json")
                        break
            
            # Build complete mapping (only files with numbers)
            for chapter_hash, chapter_info in all_chapters.items():
                if isinstance(chapter_info, dict):
                    has_output = chapter_info.get('output_file')
                    is_completed = chapter_info.get('status') == 'completed' or has_output
                    
                    if is_completed and has_output:
                        actual_num = chapter_info.get('actual_num')
                        output_file = os.path.basename(chapter_info['output_file'])
                        
                        # Skip files without numbers (but allow retain extension naming patterns)
                        if not re.search(r'\d+', output_file):
                            continue
                        
                        if actual_num is not None and output_file in all_html_files:
                            chapter_to_file_map[actual_num] = output_file
                            
            log(f"📊 Found {len(chapter_to_file_map)} chapter mappings in translation_progress.json")
            
            if chapter_to_file_map:
                min_ch = min(chapter_to_file_map.keys())
                max_ch = max(chapter_to_file_map.keys())
                log(f"  Chapter range: {min_ch} to {max_ch}")
                        
        except Exception as e:
            log(f"⚠️ Could not load translation_progress.json: {e}")
    
    # Step 3: Extract current titles from HTML files
    current_titles = {}
    
    if chapter_to_file_map:
        for chapter_num, html_file in chapter_to_file_map.items():
            try:
                html_path = os.path.join(html_dir, html_file)
                with open(html_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                parser = 'lxml-xml' if html_path.lower().endswith('.xhtml') or content.lstrip().startswith('<?xml') else 'lxml'
                soup = BeautifulSoup(content, parser)
                
                current_title = None
                for tag_name in ['h1', 'h2', 'h3', 'title']:
                    tag = soup.find(tag_name)
                    if tag:
                        text = tag.get_text().strip()
                        if text:
                            current_title = text
                            break
                
                if not current_title:
                    current_title = f"Chapter {chapter_num}"
                
                current_titles[chapter_num] = {
                    'title': current_title,
                    'filename': html_file
                }
                    
            except Exception as e:
                log(f"⚠️ Error reading {html_file}: {e}")
    else:
        # No mapping - use file order (only files with numbers)
        log("⚠️ No translation_progress.json mapping found, using file order")
        start_num = 0 if has_chapter_zero else 1
        
        for idx, html_file in enumerate(all_html_files):
            chapter_num = idx + start_num
            try:
                html_path = os.path.join(html_dir, html_file)
                with open(html_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                soup = BeautifulSoup(content, 'html.parser')
                
                current_title = None
                for tag_name in ['h1', 'h2', 'h3', 'title']:
                    tag = soup.find(tag_name)
                    if tag:
                        text = tag.get_text().strip()
                        if text:
                            current_title = text
                            break
                
                if not current_title:
                    current_title = f"Chapter {chapter_num}"
                
                current_titles[chapter_num] = {
                    'title': current_title,
                    'filename': html_file
                }
                    
            except Exception as e:
                log(f"⚠️ Error reading {html_file}: {e}")
    
    log(f"📊 Found {len(current_titles)} current titles in HTML files")
    
    # Step 4: Extract headers from source EPUB using STRICT OPF ordering
    source_headers = {}
    
    if not os.path.exists(epub_path):
        log(f"⚠️ Source EPUB not found: {epub_path}")
        return source_headers, current_titles
    
    try:
        with zipfile.ZipFile(epub_path, 'r') as zf:
            # Find and parse OPF file
            opf_content = None
            opf_path = None
            
            for name in zf.namelist():
                if name.endswith('.opf'):
                    opf_path = name
                    opf_content = zf.read(name)
                    log(f"📋 Found OPF file: {name}")
                    break
            
            if not opf_content:
                try:
                    container = zf.read('META-INF/container.xml')
                    tree = ET.fromstring(container)
                    rootfile = tree.find('.//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile')
                    if rootfile is not None:
                        opf_path = rootfile.get('full-path')
                        if opf_path:
                            opf_content = zf.read(opf_path)
                            log(f"📋 Found OPF via container.xml: {opf_path}")
                except:
                    pass
            
            # Parse OPF to get spine order
            spine_order = []
            if opf_content:
                try:
                    root = ET.fromstring(opf_content)
                    
                    ns = {'opf': 'http://www.idpf.org/2007/opf'}
                    if root.tag.startswith('{'):
                        default_ns = root.tag[1:root.tag.index('}')]
                        ns = {'opf': default_ns}
                    
                    # Get manifest to map IDs to files
                    manifest = {}
                    opf_dir = os.path.dirname(opf_path) if opf_path else ''
                    
                    for item in root.findall('.//opf:manifest/opf:item', ns):
                        item_id = item.get('id')
                        href = item.get('href')
                        media_type = item.get('media-type', '')
                        
                        if item_id and href and ('html' in media_type.lower() or href.endswith(('.html', '.xhtml', '.htm'))):
                            if opf_dir:
                                full_path = os.path.join(opf_dir, href).replace('\\', '/')
                            else:
                                full_path = href
                            manifest[item_id] = full_path
                    
                    # Get spine order - only include files with numbers
                    spine = root.find('.//opf:spine', ns)
                    if spine is not None:
                        for itemref in spine.findall('opf:itemref', ns):
                            idref = itemref.get('idref')
                            if idref and idref in manifest:
                                file_path = manifest[idref]
                                # Only include files with numbers
                                if re.search(r'\d+', file_path):
                                    spine_order.append(file_path)
                    
                    log(f"📋 Found {len(spine_order)} content files with numbers in OPF spine order")
                    
                    # Show breakdown
                    notice_count = sum(1 for f in spine_order if 'notice' in f.lower())
                    chapter_count = sum(1 for f in spine_order if 'chapter' in f.lower() and 'notice' not in f.lower())
                    if notice_count > 0:
                        log(f"   • Notice/Copyright files: {notice_count}")
                    if chapter_count > 0:
                        log(f"   • Chapter files: {chapter_count}")
                    
                except Exception as e:
                    log(f"⚠️ Error parsing OPF: {e}")
                    spine_order = []
            
            # Use spine order if available, otherwise alphabetical (only files with numbers)
            if spine_order:
                epub_html_files = spine_order
                log("✅ Using STRICT OPF spine order for source headers")
            else:
                # Fallback: only files with numbers
                epub_html_files = sorted([f for f in zf.namelist() 
                                         if f.endswith(('.html', '.xhtml', '.htm')) 
                                         and not f.startswith('__MACOSX')
                                         and re.search(r'\d+', f)])
                log("⚠️ No OPF spine found, using alphabetical order (only files with numbers)")
            
            log(f"📚 Processing {len(epub_html_files)} content files from source EPUB")
            
            # Extract ALL titles from source EPUB files (in OPF order)
            source_titles_by_index = {}
            
            for idx, content_file in enumerate(epub_html_files):
                try:
                    html_content = zf.read(content_file).decode('utf-8', errors='ignore')
                    
                    if not html_content:
                        continue
                    
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    title = None
                    for tag_name in ['h1', 'h2', 'h3', 'title']:
                        tag = soup.find(tag_name)
                        if tag:
                            text = tag.get_text().strip()
                            if text:
                                title = text
                                break
                    
                    if not title:
                        p = soup.find('p')
                        if p:
                            text = p.get_text().strip()
                            if text and len(text) < 100:
                                title = text
                    
                    if title:
                        source_titles_by_index[idx] = title
                        if idx < 5:
                            log(f"  Source[{idx}] ({os.path.basename(content_file)}): {title}")
                    
                except Exception as e:
                    log(f"  ⚠️ Error reading source chapter {idx}: {e}")
                    continue
            
            log(f"📚 Extracted {len(source_titles_by_index)} titles from source EPUB")
            
            # NOW THE KEY FIX: Map source to output using OPF spine positions
            # The chapter_idx in content_hashes should match the OPF spine position
            source_to_output = {}
            
            # First try to use content_hashes if available
            if progress_data and 'content_hashes' in progress_data:
                content_hashes = progress_data.get('content_hashes', {})
                chapters_data = progress_data.get('chapters', {})
                
                log(f"  Checking {len(content_hashes)} content hash entries...")
                
                # The issue: chapter_idx might not match OPF order if files were processed alphabetically
                # We need to correct this by matching actual_num directly
                
                for chapter_hash, hash_info in content_hashes.items():
                    if not isinstance(hash_info, dict):
                        continue
                    
                    actual_num = hash_info.get('actual_num')
                    
                    # Check if this chapter is completed
                    chapter_info = chapters_data.get(chapter_hash, {})
                    has_output = chapter_info.get('output_file')
                    is_completed = chapter_info.get('status') == 'completed' or has_output
                    
                    # Skip if output file doesn't have a number or isn't an HTML file
                    if has_output:
                        output_file = os.path.basename(chapter_info['output_file'])
                        if not re.search(r'\d+', output_file) or not output_file.lower().endswith(html_extensions):
                            continue
                    
                    if is_completed and actual_num is not None:
                        # Map OPF spine index to actual_num
                        # The actual_num is what we need to match
                        # Find the spine index that corresponds to this chapter number
                        
                        # If OPF has notice files (0-13) and chapters (14+)
                        # And actual_num is 0-based, then:
                        # actual_num 0 = spine index 0 (first notice file)
                        # actual_num 14 = spine index 14 (first chapter file)
                        
                        # Direct mapping: actual_num IS the spine index
                        source_to_output[actual_num] = actual_num
            
            log(f"  Direct mapping: {len(source_to_output)} chapters mapped")
            
            # Apply the mapping to create source_headers
            for spine_idx, output_num in source_to_output.items():
                if spine_idx in source_titles_by_index and output_num in current_titles:
                    source_headers[output_num] = source_titles_by_index[spine_idx]
                    if len(source_headers) <= 5:
                        log(f"  Mapped: Spine[{spine_idx}] → Output Ch.{output_num}: {source_titles_by_index[spine_idx][:50]}...")
            
            # If we still have missing mappings, use direct index mapping
            missing_chapters = set(current_titles.keys()) - set(source_headers.keys())
            if missing_chapters:
                log(f"⚠️ Missing mappings for chapters: {sorted(missing_chapters)[:10]}...")
                
                for missing_ch in sorted(missing_chapters):
                    # Direct mapping: chapter number = spine index
                    if missing_ch in source_titles_by_index:
                        source_headers[missing_ch] = source_titles_by_index[missing_ch]
                        if len(missing_chapters) <= 10:
                            log(f"    Direct mapped: Ch.{missing_ch} → {source_titles_by_index[missing_ch][:50]}...")
            
            log(f"📊 Final result: {len(source_headers)} source headers mapped to output chapters")
            
            # Debug output
            if source_headers:
                log(f"📋 Sample mappings:")
                for ch_num in sorted(list(source_headers.keys()))[:5]:
                    current = current_titles.get(ch_num, {})
                    log(f"   Ch.{ch_num}: {source_headers[ch_num][:40]}... → {current.get('title', 'N/A')[:40]}...")
        
    except Exception as e:
        log(f"❌ Error extracting source headers: {e}")
        import traceback
        log(traceback.format_exc())
    
    return source_headers, current_titles
