"""
Glossary Manager GUI Module
Comprehensive glossary management for automatic and manual glossary extraction
"""

import os
import sys
import json
from PySide6.QtWidgets import (QDialog, QWidget, QLabel, QLineEdit, QPushButton, 
                                QCheckBox, QRadioButton, QTextEdit, QListWidget,
                                QTreeWidget, QTreeWidgetItem, QScrollArea, QTabWidget, QTabBar,
                                QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
                                QGroupBox, QSpinBox, QSlider, QMessageBox, QFileDialog,
                                QSizePolicy, QAbstractItemView, QButtonGroup, QApplication,
                                QComboBox, QMenu, QInputDialog)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QColor, QIcon, QKeySequence, QShortcut

# WindowManager and UIHelper removed - not needed in PySide6
# Qt handles window management and UI utilities automatically


class GlossaryManagerMixin:
    """Mixin class containing glossary management methods for TranslatorGUI"""
    
    @staticmethod
    def _disable_slider_mousewheel(slider):
        """Disable mousewheel scrolling on a slider to prevent accidental changes"""
        slider.wheelEvent = lambda event: None
    
    @staticmethod
    def _disable_spinbox_mousewheel(spinbox):
        """Disable mousewheel scrolling on a spinbox to prevent accidental changes"""
        spinbox.wheelEvent = lambda event: None
    
    @staticmethod
    def _disable_tabwidget_mousewheel(tabwidget):
        """Disable mousewheel scrolling on a tab widget to prevent accidental tab switching"""
        tabwidget.wheelEvent = lambda event: None
    
    @staticmethod
    def _disable_combobox_mousewheel(combobox):
        """Disable mousewheel scrolling on a combobox"""
        combobox.wheelEvent = lambda event: None
    
    @staticmethod
    def _add_combobox_arrow(combobox):
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
    
    def _create_styled_checkbox(self, text):
        """Create a checkbox; styling is handled by the dialog's global stylesheet."""
        from PySide6.QtWidgets import QCheckBox
        return QCheckBox(text)

    def glossary_manager(self):
        """Open comprehensive glossary management dialog"""
        # Create standalone PySide6 dialog (no Tkinter parent)
        # Note: self.master is a Tkinter window, so we use None as parent for PySide6
        dialog = QDialog(None)
        dialog.setWindowTitle("Glossary Manager")
        dialog.setFont(QFont("Segoe UI", 10))
        
        # Use screen ratios instead of fixed pixels
        self._screen = QApplication.primaryScreen().geometry()
        min_width = int(self._screen.width() * 0.45)   # 50% of screen width
        min_height = int(self._screen.height() * 0.9)  # 90% of screen height (leaves room for taskbar)
        dialog.setMinimumSize(min_width, min_height)
        
        # Set window icon
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico')
            if os.path.exists(icon_path):
                dialog.setWindowIcon(QIcon(icon_path))
        except Exception as e:
            print(f"Could not load window icon: {e}")
        
        # Store dialog reference for use in nested functions
        self.dialog = dialog
        
        # Apply simplified dark mode stylesheet
        global_stylesheet = """
            /* Global dark mode styling */
            QDialog {
                background-color: #2d2d2d;
                color: white;
            }
            QGroupBox {
                color: white;
                border: 1px solid #555;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: white;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: white;
                background-color: transparent;
                border: none;
            }
            /* Checkbox styling */
            QCheckBox {
                color: white;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #5a9fd4;
                border-radius: 2px;
                background-color: transparent;
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
                background-color: transparent;
                border-color: #3a3a3a;
            }
            /* Radio button styling */
            QRadioButton {
                color: white;
                spacing: 5px;
            }
            QRadioButton::indicator {
                width: 13px;
                height: 13px;
                border: 2px solid #5a9fd4;
                border-radius: 7px;
                background-color: transparent;
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
                background-color: transparent;
                border-color: #3a3a3a;
            }
            
            /* Input fields styling */
            QLineEdit, QTextEdit {
                background-color: transparent;
                color: white;
                border: 1px solid #4a5568;
                border-radius: 3px;
                padding: 4px;
            }
            QLineEdit:focus, QTextEdit:focus {
                border-color: #5a9fd4;
            }
            QLineEdit:disabled, QTextEdit:disabled {
                background-color: #1a1a1a;
                color: #666666;
                border: 1px solid #3a3a3a;
            }
            
            /* Slider styling */
            QSlider {
                background-color: transparent;
            }
            QSlider::groove:horizontal, QSlider::groove:vertical {
                background: transparent;
            }
            QSlider::add-page:horizontal, QSlider::sub-page:horizontal,
            QSlider::add-page:vertical, QSlider::sub-page:vertical {
                background: transparent;
            }
            QSlider::handle:horizontal, QSlider::handle:vertical {
                background: #5a9fd4;
                border: 2px solid #3b4f5e;
                width: 14px;
                height: 14px;
                border-radius: 7px;
            }
            
            /* ComboBox styling */
            QComboBox {
                background-color: transparent;
                color: white;
                border: 1px solid #4a5568;
                border-radius: 3px;
                padding: 4px;
            }
            QComboBox:disabled {
                background-color: #1a1a1a;
                color: #666666;
                border: 1px solid #3a3a3a;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                color: white;
                selection-background-color: #5a9fd4;
            }
            
            /* SpinBox styling */
            QSpinBox, QDoubleSpinBox {
                background-color: transparent;
                color: white;
                border: 1px solid #4a5568;
                border-radius: 3px;
                padding: 4px;
            }
            QSpinBox:disabled, QDoubleSpinBox:disabled {
                background-color: #1a1a1a;
                color: #666666;
                border: 1px solid #3a3a3a;
            }
            
            /* Slider styling */
            QSlider::groove:horizontal {
                background: transparent;
                height: 6px;
                border-radius: 3px;
                border: 1px solid #4a5568;
            }
            QSlider::groove:vertical {
                background: transparent;
                width: 6px;
                border-radius: 3px;
                border: 1px solid #4a5568;
            }
            QSlider::handle:horizontal {
                background: #5a9fd4;
                border: 2px solid #3b4f5e;
                width: 14px;
                height: 14px;
                margin: -5px 0;  /* center over 6px groove with 2px border */
                border-radius: 7px;
            }
            QSlider::handle:vertical {
                background: #5a9fd4;
                border: 2px solid #3b4f5e;
                width: 14px;
                height: 14px;
                margin: 0 -5px;  /* center for vertical groove */
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover, QSlider::handle:vertical:hover {
                background: #7bb3e0;
            }
            
            /* GroupBox styling */
            QGroupBox {
                color: white;
                border: 1px solid #4a5568;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 5px;
                color: #5a9fd4;
            }
            
            /* Label styling */
            QLabel {
                color: white;
            }
            QLabel:disabled {
                color: #666666;
            }
            
            /* ListWidget styling */
            QListWidget {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #4a5568;
                border-radius: 3px;
            }
            QListWidget {
                background-color: transparent;
                color: white;
                border: 1px solid #4a5568;
                border-radius: 3px;
            }
            QListWidget::item:selected {
                background-color: #5a9fd4;
            }
            QListWidget::item:hover {
                background-color: #3a3a3a;
            }
            
            /* TreeWidget styling */
            QTreeWidget {
                background-color: transparent;
                color: white;
                border: 1px solid #4a5568;
                border-radius: 3px;
                alternate-background-color: transparent;
            }
            QTreeWidget::item:selected {
                background-color: #5a9fd4;
            }
            QTreeWidget::item:hover {
                background-color: #3a3a3a;
            }
            QHeaderView::section {
                background-color: #252525;
                color: white;
                border: 1px solid #4a5568;
                padding: 4px;
            }
            
            /* TabWidget styling */
            QTabWidget::pane {
                border: 1px solid #4a5568;
                background-color: #1e1e1e;
            }
            
            /* ScrollArea styling */
            QScrollArea, QScrollArea > QWidget, QScrollArea > QWidget > QWidget {
                background: transparent;
            }
            QTabBar::tab {
                background-color: #252525;
                color: white;
                border: 1px solid #4a5568;
                padding: 8px 16px;
                margin-right: 2px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #5a9fd4;
                border-bottom: 2px solid #5a9fd4;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #3a3a3a;
            }
            
            /* ScrollBar styling */
            QScrollBar:vertical {
                background: transparent;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #4a5568;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #5a9fd4;
            }
            QScrollBar:horizontal {
                background: transparent;
                height: 12px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #4a5568;
                min-width: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #5a9fd4;
            }
        """
        dialog.setStyleSheet(global_stylesheet)
        
        # Main layout
        main_layout = QVBoxLayout(dialog)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Scrollable widget and layout
        scrollable_widget = QWidget()
        scrollable_layout = QVBoxLayout(scrollable_widget)
        scroll_area.setWidget(scrollable_widget)
        main_layout.addWidget(scroll_area)
        
        # Create notebook for tabs
        notebook = QTabWidget()
        # Prevent wheel from switching tabs, but keep wheel events working inside tab contents
        class NoWheelTabBar(QTabBar):
            def wheelEvent(self, event):
                event.ignore()
        notebook.setTabBar(NoWheelTabBar())
        scrollable_layout.addWidget(notebook)
        
        # Create and add tabs
        tabs = [
            ("Manual Glossary Extraction", self._setup_manual_glossary_tab),
            ("Automatic Glossary Generation", self._setup_auto_glossary_tab),
            ("Glossary Editor", self._setup_glossary_editor_tab)
        ]
        
        for tab_name, setup_method in tabs:
            tab_widget = QWidget()
            notebook.addTab(tab_widget, tab_name)
            setup_method(tab_widget)
        
        # Dialog Controls
        control_frame = QWidget()
        control_layout = QHBoxLayout(control_frame)
        main_layout.addWidget(control_frame)
        
        def save_glossary_settings():
            try:
                # Update prompts from text widgets to instance variables
                self.update_glossary_prompts()
                
                # Check if any types are enabled before saving
                # Note: save_config will update enabled status from checkboxes automatically
                enabled_types = []
                if hasattr(self, 'type_enabled_checks') and hasattr(self, 'custom_entry_types'):
                    # Check from UI checkboxes
                    for type_name, checkbox in self.type_enabled_checks.items():
                        if checkbox.isChecked():
                            enabled_types.append(type_name)
                elif hasattr(self, 'custom_entry_types'):
                    # Fallback: check from custom_entry_types dict
                    enabled_types = [t for t, cfg in self.custom_entry_types.items() if cfg.get('enabled', True)]
                
                # Only show warning if we have custom_entry_types and none are enabled
                if hasattr(self, 'custom_entry_types') and not enabled_types:
                    QMessageBox.warning(dialog, "Warning", "No entry types selected! The glossary extraction will not find any entries.")
                
                # CRITICAL: Update the main GUI's instance variables to match checkbox states
                # These vars are checked at runtime in _get_environment_variables() and translate_image()
                # Without updating them here, changes won't take effect until GUI restart
                checkbox_to_var_mapping = [
                    ('append_glossary_checkbox', 'append_glossary_var'),
                    ('enable_auto_glossary_checkbox', 'enable_auto_glossary_var'),
                    ('add_additional_glossary_checkbox', 'add_additional_glossary_var'),
                    ('compress_glossary_checkbox', 'compress_glossary_prompt_var'),
                    ('include_gender_context_checkbox', 'include_gender_context_var'),
                    ('include_description_checkbox', 'include_description_var'),
                    ('glossary_history_rolling_checkbox', 'glossary_history_rolling_var'),
                    ('strip_honorifics_checkbox', 'strip_honorifics_var'),
                    ('disable_honorifics_checkbox', 'disable_honorifics_var'),
                    ('use_legacy_csv_checkbox', 'use_legacy_csv_var'),
                ]
                
                # Handle inverted logic for disable_smart_filtering_checkbox
                if hasattr(self, 'disable_smart_filtering_checkbox'):
                    # Checkbox is "disable" so checked=True means use_smart_filter=False
                    use_smart_filter = not self.disable_smart_filtering_checkbox.isChecked()
                    self.config['glossary_use_smart_filter'] = use_smart_filter
                    setattr(self, 'glossary_use_smart_filter_var', use_smart_filter)
                    
                    # IMPORTANT: When smart filter is disabled, also disable frequency checking
                    # This ensures ALL AI-generated entries are kept, not just the pre-filtered text
                    # Without this, entries get filtered out during post-processing even though full text was sent
                    skip_frequency = not use_smart_filter  # If smart filter disabled, skip frequency checks
                    self.config['glossary_skip_frequency_check'] = skip_frequency
                    setattr(self, 'glossary_skip_frequency_check_var', skip_frequency)
                for checkbox_name, var_name in checkbox_to_var_mapping:
                    if hasattr(self, checkbox_name):
                        setattr(self, var_name, getattr(self, checkbox_name).isChecked())
                
                # Update glossary request merging checkbox
                if hasattr(self, 'glossary_request_merging_checkbox'):
                    glossary_request_merging = self.glossary_request_merging_checkbox.isChecked()
                    self.config['glossary_request_merging_enabled'] = glossary_request_merging
                    setattr(self, 'glossary_request_merging_enabled_var', glossary_request_merging)
                
                # Update text field variables from Targeted Extraction Settings
                text_field_to_var_mapping = [
                    ('glossary_min_frequency_entry', 'glossary_min_frequency', 'glossary_min_frequency_var'),
                    ('glossary_max_names_entry', 'glossary_max_names', 'glossary_max_names_var'),
                    ('glossary_max_titles_entry', 'glossary_max_titles', 'glossary_max_titles_var'),
                    ('glossary_context_window_entry', 'glossary_context_window', 'glossary_context_window_var'),
                    ('glossary_max_text_size_entry', 'glossary_max_text_size', 'glossary_max_text_size_var'),
                    ('glossary_max_sentences_entry', 'glossary_max_sentences', 'glossary_max_sentences_var'),
                    ('glossary_chapter_split_threshold_entry', 'glossary_chapter_split_threshold', 'glossary_chapter_split_threshold_var'),
                    ('glossary_request_merge_count_entry', 'glossary_request_merge_count', 'glossary_request_merge_count_var'),
                ]
                for field_name, config_key, var_name in text_field_to_var_mapping:
                    if hasattr(self, field_name):
                        try:
                            value = int(getattr(self, field_name).text())
                            self.config[config_key] = value
                            # Also update the instance variable (used by _get_environment_variables)
                            setattr(self, var_name, str(value))
                        except ValueError:
                            pass  # Keep existing value if invalid
                
                # Update glossary-specific float fields (compression factor)
                if hasattr(self, 'glossary_compression_factor_entry'):
                    try:
                        value = float(self.glossary_compression_factor_entry.text())
                        self.config['glossary_compression_factor'] = value
                        setattr(self, 'glossary_compression_factor_var', str(value))
                    except ValueError:
                        pass
                
                # Update glossary max output tokens
                if hasattr(self, 'glossary_output_token_limit_entry'):
                    try:
                        value = int(self.glossary_output_token_limit_entry.text())
                        self.config['glossary_max_output_tokens'] = value
                        setattr(self, 'glossary_max_output_tokens_var', str(value))
                    except ValueError:
                        pass
                
                # Update target language from combo box (check both auto and manual)
                if hasattr(self, 'glossary_target_language_combo'):
                    self.config['glossary_target_language'] = self.glossary_target_language_combo.currentText()
                elif hasattr(self, 'manual_target_language_combo'):
                    self.config['glossary_target_language'] = self.manual_target_language_combo.currentText()
                
                # Update duplicate detection algorithm from combo box
                if hasattr(self, 'duplicate_algo_combo'):
                    algo_index = self.duplicate_algo_combo.currentIndex()
                    algo_map = {0: 'auto', 1: 'strict', 2: 'balanced', 3: 'aggressive', 4: 'basic'}
                    self.glossary_duplicate_algorithm_var = algo_map.get(algo_index, 'auto')
                    self.config['glossary_duplicate_algorithm'] = self.glossary_duplicate_algorithm_var
                
                # Call main save_config - it will:
                # 1. Update custom_entry_types from checkboxes
                # 2. Read from all UI widgets and instance variables
                # 3. Write everything to config.json
                # 4. Set environment variables
                self.save_config(show_message=False)
                
                # CRITICAL: Reload config.json from disk, then reinitialize ALL environment variables
                # This ensures environment variables are fully synced with the saved config
                try:
                    import json
                    from api_key_encryption import decrypt_config
                    with open('config.json', 'r', encoding='utf-8') as f:
                        self.config = json.load(f)
                        self.config = decrypt_config(self.config)
                    # Now call save_config to set ALL environment variables from the reloaded config
                    self.save_config(show_message=False)
                    
                except Exception as e:
                    self.append_log(f"⚠️ Failed to reload config: {e}")
                
                self.append_log("✅ Glossary settings saved successfully")
                self.append_log("✅ Environment variables reinitialized")
                QMessageBox.information(dialog, "Success", "Glossary settings saved!")
                dialog.accept()
                
            except Exception as e:
                import traceback
                error_msg = f"Failed to save settings: {e}\n{traceback.format_exc()}"
                QMessageBox.critical(dialog, "Error", f"Failed to save settings: {e}")
                self.append_log(f"❌ Failed to save glossary settings: {e}")
                print(error_msg)
                
        # Add buttons
        save_button = QPushButton("Save All Settings")
        save_button.clicked.connect(save_glossary_settings)
        save_button.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 8px;")
        control_layout.addWidget(save_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        cancel_button.setStyleSheet("background-color: #6c757d; color: white; padding: 8px;")
        control_layout.addWidget(cancel_button)
        
        # Show dialog with fade animation
        try:
            from dialog_animations import exec_dialog_with_fade
            exec_dialog_with_fade(dialog, duration=250)
        except Exception:
            dialog.exec()

    def _setup_manual_glossary_tab(self, parent):
        """Setup manual glossary tab - simplified for new format"""
        # Create main layout for parent
        manual_layout = QVBoxLayout(parent)
        manual_layout.setContentsMargins(10, 10, 10, 10)
        
        # Type filtering section with custom types
        type_filter_frame = QGroupBox("Entry Type Configuration")
        type_filter_layout = QVBoxLayout(type_filter_frame)
        manual_layout.addWidget(type_filter_frame)
        
        # Always reload custom entry types from config to ensure latest saved state
        self.custom_entry_types = self.config.get('custom_entry_types', {
            'character': {'enabled': True, 'has_gender': True},
            'term': {'enabled': True, 'has_gender': False}
        })
        
        # Main container with grid for better control
        type_main_grid = QGridLayout()
        type_filter_layout.addLayout(type_main_grid)
        
        # Left side - type list with checkboxes
        type_list_widget = QWidget()
        type_list_layout = QVBoxLayout(type_list_widget)
        type_list_layout.setContentsMargins(0, 0, 15, 0)
        type_main_grid.addWidget(type_list_widget, 0, 0)
        type_main_grid.setColumnStretch(0, 3)
        type_main_grid.setColumnStretch(1, 2)
        
        label = QLabel("Active Entry Types:")
        # label.setStyleSheet("font-weight: bold;")
        type_list_layout.addWidget(label)
        
        # Scrollable frame for type checkboxes
        type_scroll_area = QScrollArea()
        type_scroll_area.setWidgetResizable(True)
        # Use screen ratio: ~16% of screen height
        scroll_height = int(self._screen.height() * 0.16)
        type_scroll_area.setMinimumHeight(scroll_height)
        type_scroll_area.setMaximumHeight(scroll_height)
        type_list_layout.addWidget(type_scroll_area)
        
        self.type_checkbox_widget = QWidget()
        self.type_checkbox_layout = QVBoxLayout(self.type_checkbox_widget)
        self.type_checkbox_layout.setContentsMargins(0, 0, 0, 0)
        type_scroll_area.setWidget(self.type_checkbox_widget)
        
        # Store checkbox variables
        self.type_enabled_checkboxes = {}
        
        def update_type_checkboxes():
            """Rebuild the checkbox list"""
            # Clear existing checkboxes
            while self.type_checkbox_layout.count():
                item = self.type_checkbox_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            # Sort types: built-in first, then custom alphabetically
            sorted_types = sorted(self.custom_entry_types.items(), 
                                key=lambda x: (x[0] not in ['character', 'term'], x[0]))
            
            # Create checkboxes for each type
            for type_name, type_config in sorted_types:
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 2, 0, 2)
                row_layout.setSpacing(6)
                
                # Checkbox
                cb = self._create_styled_checkbox(type_name)
                cb.setChecked(type_config.get('enabled', True))
                self.type_enabled_checkboxes[type_name] = cb
                row_layout.addWidget(cb)
                
                # Add gender indicator for types that support it
                if type_config.get('has_gender', False):
                    label = QLabel("(has gender field)")
                    # label.setStyleSheet("color: gray; font-size: 9pt;")
                    row_layout.addWidget(label)
                
                # Delete button for custom types (place right after the label/text)
                if type_name not in ['character', 'term']:
                    delete_btn = QPushButton("×")
                    delete_btn.setFixedWidth(24)
                    delete_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #dc3545;  /* red */
                            color: white;
                            font-weight: bold;
                            border: 1px solid #a71d2a;
                            border-radius: 4px;
                            padding: 0px 6px;
                            min-height: 18px;
                        }
                        QPushButton:hover { background-color: #c82333; }
                        QPushButton:pressed { background-color: #bd2130; }
                    """)
                    delete_btn.clicked.connect(lambda checked, t=type_name: remove_type(t))
                    row_layout.addWidget(delete_btn)
                
                # Push any remaining content to the far right
                row_layout.addStretch()
                
                self.type_checkbox_layout.addWidget(row_widget)
            
            self.type_checkbox_layout.addStretch()
        
        # Right side - controls for adding custom types
        type_control_widget = QWidget()
        type_control_layout = QVBoxLayout(type_control_widget)
        type_control_layout.setContentsMargins(0, 0, 0, 0)
        type_main_grid.addWidget(type_control_widget, 0, 1)
        
        label = QLabel("Add Custom Type:")
        # label.setStyleSheet("font-weight: bold;")
        type_control_layout.addWidget(label)
        
        # Entry for new type field
        QLabel("Type Field:").setParent(type_control_widget)
        type_control_layout.addWidget(QLabel("Type Field:"))
        new_type_entry = QLineEdit()
        type_control_layout.addWidget(new_type_entry)
        
        # Checkbox for gender field
        has_gender_checkbox = self._create_styled_checkbox("Include gender field")
        type_control_layout.addWidget(has_gender_checkbox)
        
        def add_custom_type():
            type_name = new_type_entry.text().strip().lower()
            if not type_name:
                QMessageBox.warning(parent, "Invalid Input", "Please enter a type name")
                return
            
            if type_name in self.custom_entry_types:
                QMessageBox.warning(parent, "Duplicate Type", f"Type '{type_name}' already exists")
                return
            
            # Add the new type
            self.custom_entry_types[type_name] = {
                'enabled': True,
                'has_gender': has_gender_checkbox.isChecked()
            }
            
            # Clear inputs
            new_type_entry.clear()
            has_gender_checkbox.setChecked(False)
            
            # Update display
            update_type_checkboxes()
            self.append_log(f"✅ Added custom type: {type_name}")
        
        def remove_type(type_name):
            if type_name in ['character', 'term']:
                QMessageBox.warning(parent, "Cannot Remove", "Built-in types cannot be removed")
                return
            
            reply = QMessageBox.question(parent, "Confirm Removal", f"Remove type '{type_name}'?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                del self.custom_entry_types[type_name]
                if type_name in self.type_enabled_checkboxes:
                    del self.type_enabled_checkboxes[type_name]
                update_type_checkboxes()
                self.append_log(f"🗑️ Removed custom type: {type_name}")
        
        add_type_button = QPushButton("Add Type")
        add_type_button.clicked.connect(add_custom_type)
        # add_type_button.setStyleSheet("background-color: #28a745; color: white; padding: 5px;")
        type_control_layout.addWidget(add_type_button)
        type_control_layout.addStretch()
        
        # Initialize checkboxes
        update_type_checkboxes()
        
        # Custom fields section
        custom_frame = QGroupBox("Custom Fields (Additional Columns)")
        custom_frame_layout = QVBoxLayout(custom_frame)
        manual_layout.addWidget(custom_frame)
        
        QLabel("Additional fields to extract (will be added as extra columns):").setParent(custom_frame)
        custom_frame_layout.addWidget(QLabel("Additional fields to extract (will be added as extra columns):"))
        
        self.custom_fields_listbox = QListWidget()
        # Use screen ratio: ~10% of screen height
        listbox_height = int(self._screen.height() * 0.10)
        self.custom_fields_listbox.setMaximumHeight(listbox_height)
        custom_frame_layout.addWidget(self.custom_fields_listbox)
        
        # Initialize custom_glossary_fields if not exists
        if not hasattr(self, 'custom_glossary_fields'):
            self.custom_glossary_fields = self.config.get('custom_glossary_fields', [])
        
        # Add "description" as default field if list is empty and user hasn't manually removed it
        description_removed_flag = self.config.get('custom_field_description_removed', False)
        if not self.custom_glossary_fields and not description_removed_flag:
            self.custom_glossary_fields = ['description']
            # Save to config so it persists
            self.config['custom_glossary_fields'] = self.custom_glossary_fields
            self.save_config(show_message=False)
        
        for field in self.custom_glossary_fields:
            self.custom_fields_listbox.addItem(field)
        
        custom_controls_widget = QWidget()
        custom_controls_layout = QHBoxLayout(custom_controls_widget)
        custom_controls_layout.setContentsMargins(0, 5, 0, 0)
        custom_frame_layout.addWidget(custom_controls_widget)
        
        self.custom_field_entry = QLineEdit()
        self.custom_field_entry.setPlaceholderText("Enter field name...")
        custom_controls_layout.addWidget(self.custom_field_entry)
        
        def add_custom_field():
            field = self.custom_field_entry.text().strip()
            if field and field not in self.custom_glossary_fields:
                self.custom_glossary_fields.append(field)
                self.custom_fields_listbox.addItem(field)
                self.custom_field_entry.clear()
                
                # If user manually adds "description" back, clear the removal flag
                if field.lower() == 'description':
                    self.config['custom_field_description_removed'] = False
                    self.save_config(show_message=False)
        
        def remove_custom_field():
            current_row = self.custom_fields_listbox.currentRow()
            if current_row >= 0:
                item = self.custom_fields_listbox.item(current_row)
                field = item.text()
                self.custom_glossary_fields.remove(field)
                self.custom_fields_listbox.takeItem(current_row)
                
                # If user manually removes "description", set flag to prevent re-adding
                if field.lower() == 'description':
                    self.config['custom_field_description_removed'] = True
                    self.save_config(show_message=False)
        
        # Use screen ratio for button widths: ~8% of screen width
        button_width = int(self._screen.width() * 0.08)
        
        add_field_btn = QPushButton("Add")
        add_field_btn.setFixedWidth(button_width)
        add_field_btn.clicked.connect(add_custom_field)
        custom_controls_layout.addWidget(add_field_btn)
        
        remove_field_btn = QPushButton("Remove")
        remove_field_btn.setFixedWidth(button_width)
        remove_field_btn.clicked.connect(remove_custom_field)
        custom_controls_layout.addWidget(remove_field_btn)
        
        # Duplicate Detection Settings
        duplicate_frame = QGroupBox("Duplicate Detection")
        duplicate_frame_layout = QVBoxLayout(duplicate_frame)
        manual_layout.addWidget(duplicate_frame)
        
        # Algorithm selection dropdown
        algo_label = QLabel("Detection Algorithm:")
        duplicate_frame_layout.addWidget(algo_label)
        
        algo_widget = QWidget()
        algo_layout = QHBoxLayout(algo_widget)
        algo_layout.setContentsMargins(0, 0, 0, 5)
        duplicate_frame_layout.addWidget(algo_widget)
        
        # Add icon before dropdown (HiDPI-aware 36x36 like Extract Glossary)
        algo_icon_label = QLabel()
        algo_icon_label.setStyleSheet("background-color: transparent;")
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico')
            if os.path.exists(icon_path):
                from PySide6.QtGui import QIcon, QPixmap
                from PySide6.QtCore import QSize
                icon = QIcon(icon_path)
                try:
                    dpr = self.devicePixelRatioF()
                except Exception:
                    dpr = 1.0
                logical_px = 16
                dev_px = int(logical_px * max(1.0, dpr))
                pm = icon.pixmap(QSize(dev_px, dev_px))
                if pm.isNull():
                    raw = QPixmap(icon_path)
                    img = raw.toImage().scaled(dev_px, dev_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    pm = QPixmap.fromImage(img)
                try:
                    pm.setDevicePixelRatio(dpr)
                except Exception:
                    pass
                algo_icon_label.setPixmap(pm)
                algo_icon_label.setFixedSize(36, 36)
                algo_icon_label.setAlignment(Qt.AlignCenter)
            else:
                algo_icon_label.setText("🎯")
                algo_icon_label.setStyleSheet("font-size: 18pt;")
        except Exception:
            algo_icon_label.setText("🎯")
            algo_icon_label.setStyleSheet("font-size: 18pt;")
        
        algo_layout.addWidget(algo_icon_label)
        
        self.duplicate_algo_combo = QComboBox()
        self.duplicate_algo_combo.addItems([
            "Auto - Uses all algorithms",
            "Strict - High precision, minimal merging",
            "Balanced - Token + Partial matching",
            "Aggressive - Maximum duplicate detection",
            "Basic Only - Simple Levenshtein distance"
        ])
        
        # Load saved setting or default to Balanced
        saved_algo = self.config.get('glossary_duplicate_algorithm', 'balanced')
        algo_index_map = {
            'auto': 0,
            'strict': 1,
            'balanced': 2,
            'aggressive': 3,
            'basic': 4
        }
        self.duplicate_algo_combo.setCurrentIndex(algo_index_map.get(saved_algo, 2))
        
        # Try to set custom dropdown icon using Halgakos.ico
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico')
            if os.path.exists(icon_path):
                # Create a small icon for the dropdown arrow
                icon = QIcon(icon_path)
                # Set the icon for each item (optional, makes it appear next to text)
                # self.duplicate_algo_combo.setItemIcon(0, icon)
                
                # Custom stylesheet for this combo box with icon-based dropdown
                combo_style = """
                    QComboBox {
                        padding-right: 28px;
                    }
                    QComboBox::drop-down {
                        subcontrol-origin: padding;
                        subcontrol-position: top right;
                        width: 24px;
                        border-left: 1px solid #4a5568;
                    }
                    QComboBox::down-arrow {
                        width: 16px;
                        height: 16px;
                        image: url(""" + icon_path.replace('\\', '/') + """);
                    }
                    QComboBox::down-arrow:on {
                        top: 1px;
                    }
                """
                self.duplicate_algo_combo.setStyleSheet(combo_style)
        except Exception as e:
            # If icon fails, just use default styling
            pass
        
        algo_layout.addWidget(self.duplicate_algo_combo)
        
        # Info button
        algo_info_btn = QPushButton("ℹ️ Info")
        algo_info_btn.setFixedWidth(60)
        
        def show_algorithm_info():
            msg_box = QMessageBox(parent)
            msg_box.setWindowTitle("Algorithm Information")
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText(
                "<b>Auto</b>: Uses all available algorithms (RapidFuzz, Jaro-Winkler, Token matching) and takes the best score.<br><br>"
                "<b>Strict</b>: Only matches very similar names (95%+ similarity). Keeps more entries, minimal merging. Good if you want to review duplicates manually.<br><br>"
                "<b>Balanced</b>: Uses token-based and partial matching. Handles word order (‘Park Ji-sung’ = ‘Ji-sung Park’) and substrings. Good middle ground.<br><br>"
                "<b>Aggressive</b>: Lower threshold (80%) with all algorithms. Catches romanization variants (‘Catherine’ = ‘Katherine’). May over-merge similar names.<br><br>"
                "<b>Basic Only</b>: Simple Levenshtein distance. Faster but less accurate. May miss variants like ‘Kim Sang-hyun’ vs ‘Kim Sanghyun’."
            )
            
            # Set size using screen ratios: 40% width, 50% height
            screen_width = self._screen.width()
            screen_height = self._screen.height()
            msg_box.setMinimumWidth(int(screen_width * 0.40))
            msg_box.setMinimumHeight(int(screen_height * 0.50))
            
            msg_box.exec()
        
        algo_info_btn.clicked.connect(show_algorithm_info)
        algo_layout.addWidget(algo_info_btn)
        algo_layout.addStretch()
        
        # Update description when algorithm changes
        def update_algo_description(index):
            descriptions = [
                "🎯 Auto mode uses multiple algorithms for best accuracy",
                "🔒 Strict mode: High precision, keeps more entries",
                "⚖️ Balanced mode: Handles word order and substrings",
                "🔥 Aggressive mode: Maximum duplicate detection (may over-merge)",
                "📄 Basic mode: Simple matching (faster, less accurate)"
            ]
            algo_desc.setText(descriptions[index])
        
        algo_desc = QLabel()
        algo_desc.setStyleSheet("color: gray; font-size: 9pt; margin-bottom: 15px;")
        duplicate_frame_layout.addWidget(algo_desc)
        
        # Set initial description based on saved algorithm
        update_algo_description(self.duplicate_algo_combo.currentIndex())
        
        self.duplicate_algo_combo.currentIndexChanged.connect(update_algo_description)
        
        # Honorifics filter toggle
        if not hasattr(self, 'disable_honorifics_checkbox'):
            self.disable_honorifics_checkbox = self._create_styled_checkbox("Disable honorifics filtering")
            self.disable_honorifics_checkbox.setChecked(self.config.get('glossary_disable_honorifics_filter', False))
        
        duplicate_frame_layout.addWidget(self.disable_honorifics_checkbox)
        
        honorifics_label = QLabel("When enabled, honorifics (님, さん, 先生, etc.) will NOT be removed from raw names")
        # honorifics_label.setStyleSheet("color: gray; font-size: 9pt; margin-left: 20px;")
        duplicate_frame_layout.addWidget(honorifics_label)
        
        # Fuzzy matching slider
        fuzzy_label = QLabel("Fuzzy Matching Threshold:")
        # fuzzy_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        duplicate_frame_layout.addWidget(fuzzy_label)

        desc_label = QLabel("Controls how similar names must be to be considered duplicates")
        # desc_label.setStyleSheet("color: gray; font-size: 9pt;")
        duplicate_frame_layout.addWidget(desc_label)

        # Slider widget
        slider_widget = QWidget()
        slider_layout = QHBoxLayout(slider_widget)
        slider_layout.setContentsMargins(0, 5, 0, 0)
        duplicate_frame_layout.addWidget(slider_widget)

        # Always reload fuzzy threshold value from config
        self.fuzzy_threshold_value = self.config.get('glossary_fuzzy_threshold', 0.90)

        # Slider (store as self.manual_fuzzy_slider for syncing)
        self.manual_fuzzy_slider = QSlider(Qt.Horizontal)
        self.manual_fuzzy_slider.setMinimum(50)  # 0.5 * 100
        self.manual_fuzzy_slider.setMaximum(100)  # 1.0 * 100
        self.manual_fuzzy_slider.setValue(int(self.fuzzy_threshold_value * 100))
        # Use screen ratio: ~30% of screen width
        slider_width = int(self._screen.width() * 0.30)
        self.manual_fuzzy_slider.setMinimumWidth(slider_width)
        self._disable_slider_mousewheel(self.manual_fuzzy_slider)  # Disable mouse wheel
        slider_layout.addWidget(self.manual_fuzzy_slider)

        # Value label
        self.manual_fuzzy_value_label = QLabel(f"{self.fuzzy_threshold_value:.2f}")
        slider_layout.addWidget(self.manual_fuzzy_value_label)

        # Description label
        self.manual_fuzzy_desc_label = QLabel("")
        # self.manual_fuzzy_desc_label.setStyleSheet("color: white; font-size: 9pt; margin-top: 5px;")
        duplicate_frame_layout.addWidget(self.manual_fuzzy_desc_label)

        # Token-efficient format toggle
        format_frame = QGroupBox("Output Format")
        format_frame_layout = QVBoxLayout(format_frame)
        manual_layout.addWidget(format_frame)

        # Initialize variable if not exists
        if not hasattr(self, 'use_legacy_csv_checkbox'):
            self.use_legacy_csv_checkbox = self._create_styled_checkbox("Use legacy CSV format")
            self.use_legacy_csv_checkbox.setChecked(self.config.get('glossary_use_legacy_csv', False))

        format_frame_layout.addWidget(self.use_legacy_csv_checkbox)

        label1 = QLabel("When disabled (default): Uses token-efficient format with sections (=== CHARACTERS ===)")
        # label1.setStyleSheet("color: gray; font-size: 9pt; margin-left: 20px;")
        format_frame_layout.addWidget(label1)

        label2 = QLabel("When enabled: Uses traditional CSV format with repeated type columns")
        # label2.setStyleSheet("color: gray; font-size: 9pt; margin-left: 20px;")
        format_frame_layout.addWidget(label2)
        
        # Update label when slider moves
        def update_manual_fuzzy_label(value):
            float_value = value / 100.0
            self.fuzzy_threshold_value = float_value
            self.manual_fuzzy_value_label.setText(f"{float_value:.2f}")
            
            # Show description
            if float_value >= 0.95:
                desc = "Exact match only (strict)"
            elif float_value >= 0.85:
                desc = "Very similar names (recommended)"
            elif float_value >= 0.75:
                desc = "Moderately similar names"
            elif float_value >= 0.65:
                desc = "Loosely similar names"
            else:
                desc = "Very loose matching (may over-merge)"
            
            self.manual_fuzzy_desc_label.setText(desc)
            
            # Sync with auto glossary slider and labels if they exist
            if hasattr(self, 'fuzzy_threshold_slider'):
                self.fuzzy_threshold_slider.blockSignals(True)
                self.fuzzy_threshold_slider.setValue(value)
                self.fuzzy_threshold_slider.blockSignals(False)
                
                # Update auto labels directly without triggering signals
                if hasattr(self, 'auto_fuzzy_value_label') and hasattr(self, 'auto_fuzzy_desc_label'):
                    self.auto_fuzzy_value_label.setText(f"{float_value:.2f}")
                    self.auto_fuzzy_desc_label.setText(desc)
        
        # Store update function for cross-tab syncing
        self.update_manual_fuzzy_label_func = update_manual_fuzzy_label
        
        # Connect slider to update function
        self.manual_fuzzy_slider.valueChanged.connect(update_manual_fuzzy_label)
        
        # Initialize description
        update_manual_fuzzy_label(self.manual_fuzzy_slider.value())
        
        # Target language dropdown (above prompt)
        language_frame = QGroupBox("Target Language")
        language_frame_layout = QVBoxLayout(language_frame)
        manual_layout.addWidget(language_frame)
        
        # Create language dropdown
        if not hasattr(self, 'manual_target_language_combo'):
            self.manual_target_language_combo = QComboBox()
            self.manual_target_language_combo.setMaximumWidth(200)
            self.manual_target_language_combo.setEditable(True)
            languages = [
                "English", "Spanish", "French", "German", "Italian", "Portuguese",
                "Russian", "Arabic", "Hindi", "Chinese (Simplified)",
                "Chinese (Traditional)", "Japanese", "Korean", "Turkish"
            ]
            self.manual_target_language_combo.addItems(languages)
            
            # Lock mousewheel scrolling on target language dropdown
            self._disable_combobox_mousewheel(self.manual_target_language_combo)
            
            # Use icon in dropdown arrow like auto glossary dropdown
            try:
                icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico')
                if os.path.exists(icon_path):
                    combo_style = """
                        QComboBox {
                            padding-right: 28px;
                        }
                        QComboBox::drop-down {
                            subcontrol-origin: padding;
                            subcontrol-position: top right;
                            width: 24px;
                            border-left: 1px solid #4a5568;
                        }
                        QComboBox::down-arrow {
                            width: 16px;
                            height: 16px;
                            image: url(""" + icon_path.replace('\\', '/') + """);
                        }
                        QComboBox::down-arrow:on {
                            top: 1px;
                        }
                    """
                    self.manual_target_language_combo.setStyleSheet(combo_style)
            except Exception:
                pass
        
        saved_language = self.config.get('glossary_target_language', self.config.get('output_language', 'English'))
        index = self.manual_target_language_combo.findText(saved_language)
        if index >= 0:
            self.manual_target_language_combo.setCurrentIndex(index)
        else:
            self.manual_target_language_combo.setCurrentText(saved_language)
        
        # Sync with auto glossary language dropdown and main GUI
        def sync_manual_to_auto(text):
            # Sync with auto glossary dropdown
            if hasattr(self, 'glossary_target_language_combo'):
                auto_index = self.glossary_target_language_combo.findText(text)
                if auto_index >= 0:
                    self.glossary_target_language_combo.blockSignals(True)
                    self.glossary_target_language_combo.setCurrentIndex(auto_index)
                    self.glossary_target_language_combo.blockSignals(False)
            
            # Sync with main GUI dropdown
            if hasattr(self, 'target_lang_combo'):
                main_index = self.target_lang_combo.findText(text)
                if main_index >= 0:
                    self.target_lang_combo.blockSignals(True)
                    self.target_lang_combo.setCurrentIndex(main_index)
                    self.target_lang_combo.blockSignals(False)
                else:
                    self.target_lang_combo.blockSignals(True)
                    self.target_lang_combo.setCurrentText(text)
                    self.target_lang_combo.blockSignals(False)
            
            # Update configs
            self.config['glossary_target_language'] = text
            self.config['output_language'] = text
            os.environ['OUTPUT_LANGUAGE'] = text
        
        self.manual_target_language_combo.currentTextChanged.connect(sync_manual_to_auto)
        
        language_frame_layout.addWidget(self.manual_target_language_combo)
        
        lang_desc = QLabel("Language for translated glossary entries (synced with Extraction Settings)")
        language_frame_layout.addWidget(lang_desc)
        
        # Prompt section
        prompt_frame = QGroupBox("Extraction Prompt")
        prompt_frame_layout = QVBoxLayout(prompt_frame)
        manual_layout.addWidget(prompt_frame)
        
        label1 = QLabel("Use {fields} for field list and {language} for target language")
        # label1.setStyleSheet("color: white; font-size: 9pt;")
        prompt_frame_layout.addWidget(label1)
        
        label2 = QLabel("Placeholders will be replaced with actual values during extraction")
        # label2.setStyleSheet("color: gray; font-size: 9pt;")
        prompt_frame_layout.addWidget(label2)
        
        self.manual_prompt_text = QTextEdit()
        # Use screen ratio: ~25% of screen height
        prompt_height = int(self._screen.height() * 0.25)
        self.manual_prompt_text.setMinimumHeight(prompt_height)
        self.manual_prompt_text.setLineWrapMode(QTextEdit.WidgetWidth)
        prompt_frame_layout.addWidget(self.manual_prompt_text)

        # If the user clears the prompt and leaves the field, restore the default.
        # (Avoids persisting an empty template and doesn't interfere with copy/paste while typing.)
        _orig_focus_out = self.manual_prompt_text.focusOutEvent
        def _manual_prompt_focus_out(event):
            try:
                if not self.manual_prompt_text.toPlainText().strip():
                    default_manual = getattr(self, 'default_manual_glossary_prompt', None)
                    if default_manual:
                        self.manual_prompt_text.setPlainText(default_manual)
            except Exception:
                pass
            return _orig_focus_out(event)
        self.manual_prompt_text.focusOutEvent = _manual_prompt_focus_out
        
        # Always reload prompt from config to ensure fresh state
        # Treat empty strings as missing so users always get a usable default.
        default_manual_prompt = """Extract character names and important terms from the following text.

Output format:
{fields}

Rules:
- Output ONLY CSV lines in the exact format shown above
- No headers, no extra text, no JSON
- One entry per line
- Leave gender empty for terms (just end with comma)
- Do not add generic pronoun only entries (Example: I, you, he, she, etc.) and common nouns (father, mother, etc.)
- For all fields except 'raw_name', use {language} translation
    """
        # Keep a copy for later (e.g., when saving and the field was cleared)
        self.default_manual_glossary_prompt = default_manual_prompt

        manual_prompt_from_config = self.config.get('manual_glossary_prompt', default_manual_prompt)
        if not manual_prompt_from_config or not manual_prompt_from_config.strip():
            self.manual_glossary_prompt = default_manual_prompt
        else:
            self.manual_glossary_prompt = manual_prompt_from_config

        self.manual_prompt_text.setPlainText(self.manual_glossary_prompt)
        
        prompt_controls_widget = QWidget()
        prompt_controls_layout = QHBoxLayout(prompt_controls_widget)
        prompt_controls_layout.setContentsMargins(0, 10, 0, 0)
        manual_layout.addWidget(prompt_controls_widget)
        
        def reset_manual_prompt():
            reply = QMessageBox.question(parent, "Reset Prompt", "Reset manual glossary prompt to default?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.manual_prompt_text.setPlainText(getattr(self, 'default_manual_glossary_prompt', self.manual_prompt_text.toPlainText()))
        
        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(reset_manual_prompt)
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #b8860b;  /* dark yellow */
                color: black;
                padding: 5px;
                border: 1px solid #8a6a08;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #9a6d07; }
            QPushButton:pressed { background-color: #8a6106; }
        """)
        prompt_controls_layout.addWidget(reset_btn)
        prompt_controls_layout.addStretch()
        
        # Settings
        settings_frame = QGroupBox("Extraction Settings")
        settings_frame_layout = QVBoxLayout(settings_frame)
        manual_layout.addWidget(settings_frame)
        
        settings_grid = QGridLayout()
        settings_grid.setContentsMargins(2, 4, 6, 6)
        settings_grid.setHorizontalSpacing(8)
        settings_grid.setVerticalSpacing(6)
        settings_frame_layout.addLayout(settings_grid)
        
        # Compact label+field pair helper for manual Extraction Settings
        def _m_pair(label_text, field_widget, label_width=120):
            cont = QWidget()
            h = QHBoxLayout(cont)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            lbl = QLabel(label_text)
            lbl.setFixedWidth(label_width)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            h.addWidget(lbl)
            h.addWidget(field_widget)
            h.addStretch()
            return cont
        
        # Add icon to third column
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico')
        if os.path.exists(icon_path):
            from PySide6.QtGui import QPixmap, QIcon
            icon_label = QLabel()
            icon_label.setMinimumSize(180, 180)
            icon = QIcon(icon_path)
            pixmap = icon.pixmap(140, 140)  # Smaller icon with padding
            icon_label.setPixmap(pixmap)
            icon_label.setAlignment(Qt.AlignCenter)
            settings_grid.addWidget(icon_label, 0, 2, 4, 1)  # Span 4 rows
        
        # Row 0: Temperature and Context Limit
        self.manual_temp_entry = QLineEdit(str(self.config.get('manual_glossary_temperature', 0.1)))
        self.manual_temp_entry.setFixedWidth(80)
        settings_grid.addWidget(_m_pair("Temperature:", self.manual_temp_entry), 0, 0)
        
        self.manual_context_entry = QLineEdit(str(self.config.get('manual_context_limit', 2)))
        self.manual_context_entry.setFixedWidth(80)
        settings_grid.addWidget(_m_pair("Context Limit:", self.manual_context_entry), 0, 1)
        
        # Row 1: Compression Factor and Rolling window checkbox
        self.glossary_compression_factor_entry = QLineEdit(str(self.config.get('glossary_compression_factor', 1.0)))
        self.glossary_compression_factor_entry.setFixedWidth(80)
        settings_grid.addWidget(_m_pair("Compression Factor:", self.glossary_compression_factor_entry), 1, 0)
        
        if not hasattr(self, 'glossary_history_rolling_checkbox'):
            self.glossary_history_rolling_checkbox = self._create_styled_checkbox("Keep recent context instead of reset")
        self.glossary_history_rolling_checkbox.setChecked(self.config.get('glossary_history_rolling', False))
        settings_grid.addWidget(self.glossary_history_rolling_checkbox, 1, 1)
        
        # Row 2: Output Token Limit and Request Merging checkbox
        self.glossary_output_token_limit_entry = QLineEdit(str(self.config.get('glossary_max_output_tokens', 65536)))
        self.glossary_output_token_limit_entry.setFixedWidth(80)
        settings_grid.addWidget(_m_pair("Output Token Limit:", self.glossary_output_token_limit_entry), 2, 0)
        
        if not hasattr(self, 'glossary_request_merging_checkbox'):
            self.glossary_request_merging_checkbox = self._create_styled_checkbox("Glossary Request Merging")
        self.glossary_request_merging_checkbox.setChecked(self.config.get('glossary_request_merging_enabled', False))
        settings_grid.addWidget(self.glossary_request_merging_checkbox, 2, 1)
        
        # Row 3: Empty and Merge Count
        self.glossary_request_merge_count_entry = QLineEdit(str(self.config.get('glossary_request_merge_count', 10)))
        self.glossary_request_merge_count_entry.setFixedWidth(80)
        settings_grid.addWidget(_m_pair("Merge Count:", self.glossary_request_merge_count_entry), 3, 1)

    def update_glossary_prompts(self):
        """Update glossary prompts from text widgets if they exist"""
        try:
            debug_enabled = getattr(self, 'config', {}).get('show_debug_buttons', False)
            
            if hasattr(self, 'manual_prompt_text'):
                manual_text = self.manual_prompt_text.toPlainText()
                self.manual_glossary_prompt = manual_text.strip()

                # If the prompt was cleared, restore the default so we never persist an empty template.
                if not self.manual_glossary_prompt:
                    default_manual = getattr(self, 'default_manual_glossary_prompt', None)
                    if default_manual:
                        self.manual_glossary_prompt = default_manual.strip()
                        # Update the UI to reflect the restored default
                        try:
                            self.manual_prompt_text.blockSignals(True)
                            self.manual_prompt_text.setPlainText(default_manual)
                        finally:
                            self.manual_prompt_text.blockSignals(False)

                if debug_enabled:
                    print(f"🔍 [UPDATE] manual_glossary_prompt: {len(self.manual_glossary_prompt)} chars")
            
            if hasattr(self, 'auto_prompt_text'):
                self.unified_auto_glossary_prompt = self.auto_prompt_text.toPlainText().strip()
                if debug_enabled:
                    print(f"🔍 [UPDATE] unified_auto_glossary_prompt: {len(self.unified_auto_glossary_prompt)} chars")
            
            if hasattr(self, 'append_prompt_text'):
                old_value = getattr(self, 'append_glossary_prompt', '<NOT SET>')
                self.append_glossary_prompt = self.append_prompt_text.toPlainText().strip()
                if debug_enabled:
                    print(f"🔍 [UPDATE] append_glossary_prompt: OLD='{old_value[:50]}...' NEW='{self.append_glossary_prompt[:50]}...' ({len(self.append_glossary_prompt)} chars)")
                else:
                    # Always print this one since it's the problematic field
                    #print(f"Updated append_glossary_prompt from UI: '{self.append_glossary_prompt[:80]}...' ({len(self.append_glossary_prompt)} chars)")
                    pass
            
        except Exception as e:
            print(f"❌ Error updating glossary prompts: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_additional_glossary(self):
        """Load an additional glossary file (CSV/TXT/JSON/PDF) to append to auto-generated glossary"""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        
        # Open file dialog to select glossary file
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Additional Glossary File",
            "",
            "Glossary Files (*.csv *.txt *.json *.pdf *.md);;CSV Files (*.csv);;Text Files (*.txt);;JSON Files (*.json);;PDF Files (*.pdf);;Markdown Files (*.md);;All Files (*.*)"
        )
        
        if not file_path:
            return  # User cancelled
        
        # Validate file exists
        if not os.path.exists(file_path):
            QMessageBox.warning(None, "File Not Found", f"Selected file does not exist:\n{file_path}")
            return
        
        # Load and validate the file content
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            content_preview = ""
            
            if file_ext == '.csv':
                # Read CSV and validate format
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        content_preview = f"CSV file with {len(lines)} lines\nFirst line: {lines[0][:100]}"
            
            elif file_ext == '.txt':
                # Read text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    content_preview = f"Text file ({len(content)} chars)\nFirst 100 chars: {content[:100]}"
            
            elif file_ext == '.md':
                # Read markdown file (treat same as text)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    content_preview = f"Markdown file ({len(content)} chars)\nFirst 100 chars: {content[:100]}"
            
            elif file_ext == '.json':
                # Read and validate JSON
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    content_preview = f"JSON file with {len(data)} entries" if isinstance(data, (list, dict)) else "JSON file"
            
            elif file_ext == '.pdf':
                # Just validate PDF exists (parsing will happen during glossary generation)
                file_size = os.path.getsize(file_path)
                content_preview = f"PDF file ({file_size} bytes)"
            
            else:
                QMessageBox.warning(None, "Unsupported Format", f"Unsupported file format: {file_ext}\nSupported formats: .csv, .txt, .json, .pdf, .md")
                return
            
            # Save to config
            self.config['additional_glossary_path'] = file_path
            self.config['add_additional_glossary'] = True  # Auto-enable the checkbox
            self.save_config()
            
            # Update checkbox if it exists
            if hasattr(self, 'add_additional_glossary_checkbox'):
                self.add_additional_glossary_checkbox.setChecked(True)
            
            # Update label if it exists
            if hasattr(self, 'additional_glossary_label'):
                self.additional_glossary_label.setText(f"(Current: {os.path.basename(file_path)})")
            
            # Show success message with icon
            msg_box = QMessageBox(None)
            msg_box.setWindowTitle("Additional Glossary Loaded")
            msg_box.setIcon(QMessageBox.Information)
            # Use the actual file extension in the message
            target_filename = f"glossary_extension{file_ext}"
            msg_box.setText(f"Successfully loaded additional glossary:\n\n{os.path.basename(file_path)}\n\n{content_preview}\n\nThis will be copied as '{target_filename}' alongside the main glossary and sent to the API.")
            
            # Set window icon
            try:
                icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico')
                if os.path.exists(icon_path):
                    msg_box.setWindowIcon(QIcon(icon_path))
            except Exception:
                pass  # If icon fails to load, continue without it
            
            msg_box.exec()
            
        except Exception as e:
            QMessageBox.critical(
                None,
                "Error Loading File",
                f"Failed to load additional glossary:\n\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
            
    def _setup_auto_glossary_tab(self, parent):
        """Setup automatic glossary tab with fully configurable prompts"""
        # Create main layout for parent
        auto_layout = QVBoxLayout(parent)
        auto_layout.setContentsMargins(10, 10, 10, 10)
        
        # Master toggle
        master_toggle_widget = QWidget()
        master_toggle_layout = QHBoxLayout(master_toggle_widget)
        master_toggle_layout.setContentsMargins(0, 0, 0, 15)
        auto_layout.addWidget(master_toggle_widget)
        
        if not hasattr(self, 'enable_auto_glossary_checkbox'):
            self.enable_auto_glossary_checkbox = self._create_styled_checkbox("Enable Automatic Glossary Generation")
            self.enable_auto_glossary_checkbox.setChecked(self.config.get('enable_auto_glossary', False))
        master_toggle_layout.addWidget(self.enable_auto_glossary_checkbox)
        
        label = QLabel("(Automatic extraction and translation of character names/Terms)")
        # label.setStyleSheet("color: gray; font-size: 9pt;")
        master_toggle_layout.addWidget(label)
        master_toggle_layout.addStretch()
        
        # Append glossary toggle
        append_widget = QWidget()
        append_layout = QHBoxLayout(append_widget)
        append_layout.setContentsMargins(0, 0, 0, 15)
        auto_layout.addWidget(append_widget)
        
        if not hasattr(self, 'append_glossary_checkbox'):
            self.append_glossary_checkbox = self._create_styled_checkbox("Append Glossary to System Prompt")
            self.append_glossary_checkbox.setChecked(self.config.get('append_glossary', False))
        append_layout.addWidget(self.append_glossary_checkbox)
        
        label2 = QLabel("(Applies to ALL glossaries - manual and automatic)")
        # label2.setStyleSheet("color: white; font-size: 10pt; font-style: italic;")
        append_layout.addWidget(label2)
        append_layout.addStretch()
        
        # Add additional glossary toggle (below append glossary)
        additional_glossary_widget = QWidget()
        additional_glossary_layout = QHBoxLayout(additional_glossary_widget)
        additional_glossary_layout.setContentsMargins(0, 0, 0, 15)
        auto_layout.addWidget(additional_glossary_widget)
        
        if not hasattr(self, 'add_additional_glossary_checkbox'):
            self.add_additional_glossary_checkbox = self._create_styled_checkbox("Add Additional Glossary")
            self.add_additional_glossary_checkbox.setChecked(self.config.get('add_additional_glossary', False))
        additional_glossary_layout.addWidget(self.add_additional_glossary_checkbox)
        
        # Load additional glossary button
        load_additional_btn = QPushButton("Load Additional Glossary")
        load_additional_btn.setStyleSheet("""
            QPushButton {
                background-color: #5a9fd4;
                color: white;
                padding: 5px 15px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #7ab8e8; }
            QPushButton:pressed { background-color: #4a8fc4; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """)
        load_additional_btn.clicked.connect(self._load_additional_glossary)
        additional_glossary_layout.addWidget(load_additional_btn)
        
        # Show current additional glossary path if exists
        additional_glossary_path = self.config.get('additional_glossary_path', '')
        if additional_glossary_path:
            label_additional = QLabel(f"(Current: {os.path.basename(additional_glossary_path)})")
        else:
            label_additional = QLabel("(Sends additional glossary file alongside main glossary to API)")
        additional_glossary_layout.addWidget(label_additional)
        self.additional_glossary_label = label_additional  # Store reference to update later
        additional_glossary_layout.addStretch()
        
        # Compress glossary toggle
        compress_widget = QWidget()
        compress_layout = QHBoxLayout(compress_widget)
        compress_layout.setContentsMargins(0, 0, 0, 15)
        auto_layout.addWidget(compress_widget)
        
        if not hasattr(self, 'compress_glossary_checkbox'):
            self.compress_glossary_checkbox = self._create_styled_checkbox("Compress Glossary Prompt")
            self.compress_glossary_checkbox.setChecked(self.config.get('compress_glossary_prompt', True))
        compress_layout.addWidget(self.compress_glossary_checkbox)
        
        label3 = QLabel("(Excludes glossary entries that don't appear in source text before sending to API)")
        # label3.setStyleSheet("color: white; font-size: 10pt; font-style: italic;")
        compress_layout.addWidget(label3)
        compress_layout.addStretch()
        
        # Include gender context toggle (below compress glossary)
        gender_context_widget = QWidget()
        gender_context_layout = QHBoxLayout(gender_context_widget)
        gender_context_layout.setContentsMargins(0, 0, 0, 15)
        auto_layout.addWidget(gender_context_widget)
        
        if not hasattr(self, 'include_gender_context_checkbox'):
            self.include_gender_context_checkbox = self._create_styled_checkbox("Include Gender Context (More Expensive)")
            self.include_gender_context_checkbox.setChecked(self.config.get('include_gender_context', False))
        gender_context_layout.addWidget(self.include_gender_context_checkbox)
        
        label4 = QLabel("(Expands text snippets to include surrounding sentences for better gender detection)")
        gender_context_layout.addWidget(label4)
        gender_context_layout.addStretch()
        
        # Include description column toggle (below gender context)
        description_widget = QWidget()
        description_layout = QHBoxLayout(description_widget)
        description_layout.setContentsMargins(0, 0, 0, 15)
        auto_layout.addWidget(description_widget)
        
        if not hasattr(self, 'include_description_checkbox'):
            self.include_description_checkbox = self._create_styled_checkbox("Include Description Column")
            self.include_description_checkbox.setChecked(self.config.get('include_description', False))
        description_layout.addWidget(self.include_description_checkbox)
        
        label5 = QLabel("(Adds a description/context field for each glossary entry)")
        description_layout.addWidget(label5)
        description_layout.addStretch()
        
        # Function to update description checkbox state based on gender context
        def update_description_state():
            gender_enabled = self.include_gender_context_checkbox.isChecked()
            self.include_description_checkbox.setEnabled(gender_enabled)
            label5.setEnabled(gender_enabled)
            if not gender_enabled:
                # Uncheck if disabled
                self.include_description_checkbox.setChecked(False)
        
        # Connect gender context checkbox to update description state
        self.include_gender_context_checkbox.stateChanged.connect(update_description_state)
        # Initialize state
        update_description_state()
        
        # Disable smart filtering toggle (below description)
        disable_filtering_widget = QWidget()
        disable_filtering_layout = QHBoxLayout(disable_filtering_widget)
        disable_filtering_layout.setContentsMargins(0, 0, 0, 15)
        auto_layout.addWidget(disable_filtering_widget)
        
        if not hasattr(self, 'disable_smart_filtering_checkbox'):
            self.disable_smart_filtering_checkbox = self._create_styled_checkbox("Disable Smart Filtering (Send Full Text)")
            # Invert the logic: checkbox is "disable" so checked=True means use_smart_filter=False
            self.disable_smart_filtering_checkbox.setChecked(not self.config.get('glossary_use_smart_filter', True))
        disable_filtering_layout.addWidget(self.disable_smart_filtering_checkbox)
        
        label6 = QLabel("(Disables all text filtering and sends the entire novel to the API - very expensive!)")
        disable_filtering_layout.addWidget(label6)
        disable_filtering_layout.addStretch()
        
        # Custom append prompt section
        append_prompt_frame = QGroupBox("Glossary Append Format")
        append_prompt_layout = QVBoxLayout(append_prompt_frame)
        append_prompt_layout.setContentsMargins(10, 10, 10, 10)  # Tighter margins
        append_prompt_layout.setSpacing(5)  # Reduced spacing
        auto_layout.addWidget(append_prompt_frame)
        
        self.append_prompt_text = QTextEdit()
        self.append_prompt_text.setFixedHeight(60)
        self.append_prompt_text.setLineWrapMode(QTextEdit.WidgetWidth)
        append_prompt_layout.addWidget(self.append_prompt_text)
        
        # Always reload append prompt from config to ensure fresh state
        # Treat empty string as missing to ensure users get the default
        default_append_prompt = "- Follow this reference glossary for consistent translation (Do not output any raw entries):\n"
        append_prompt_from_config = self.config.get('append_glossary_prompt', default_append_prompt)
        if not append_prompt_from_config or not append_prompt_from_config.strip():
            self.append_glossary_prompt = default_append_prompt
        else:
            self.append_glossary_prompt = append_prompt_from_config
        
        self.append_prompt_text.setPlainText(self.append_glossary_prompt)
        
        append_prompt_controls_widget = QWidget()
        append_prompt_controls_layout = QHBoxLayout(append_prompt_controls_widget)
        append_prompt_controls_layout.setContentsMargins(0, 5, 0, 0)
        append_prompt_layout.addWidget(append_prompt_controls_widget)
        
        def reset_append_prompt():
            reply = QMessageBox.question(parent, "Reset Prompt", "Reset to default glossary append format?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.append_prompt_text.setPlainText("- Follow this reference glossary for consistent translation (Do not output any raw entries):\n")
        
        reset_append_btn = QPushButton("Reset to Default")
        reset_append_btn.clicked.connect(reset_append_prompt)
        reset_append_btn.setStyleSheet("""
            QPushButton {
                background-color: #b8860b;
                color: black;
                padding: 5px;
                border: 1px solid #8a6a08;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #9a6d07; }
            QPushButton:pressed { background-color: #8a6106; }
        """)
        append_prompt_controls_layout.addWidget(reset_append_btn)
        append_prompt_controls_layout.addStretch()
        
        # Create notebook for tabs
        notebook = QTabWidget()
        # Prevent wheel from switching tabs, but allow wheel scrolling inside sections
        class NoWheelTabBar(QTabBar):
            def wheelEvent(self, event):
                event.ignore()
        notebook.setTabBar(NoWheelTabBar())
        auto_layout.addWidget(notebook)
        
        # Add stretch to eliminate the massive empty space
        auto_layout.addStretch(1)
        
        # Tab 1: Extraction Settings
        extraction_tab = QWidget()
        extraction_tab_layout = QVBoxLayout(extraction_tab)
        extraction_tab_layout.setContentsMargins(10, 10, 10, 10)
        notebook.addTab(extraction_tab, "Extraction Settings")
        
        # Extraction settings
        settings_label_frame = QGroupBox("Targeted Extraction Settings")
        settings_label_layout = QVBoxLayout(settings_label_frame)
        settings_label_layout.setContentsMargins(6, 6, 6, 6)
        extraction_tab_layout.addWidget(settings_label_frame)
        
        extraction_grid = QGridLayout()
        # Tighten spacing between labels and controls inside Targeted Extraction Settings
        extraction_grid.setContentsMargins(2, 4, 6, 6)
        extraction_grid.setHorizontalSpacing(8)
        extraction_grid.setVerticalSpacing(6)
        # Set column stretch factors to minimize gap between left and right columns
        extraction_grid.setColumnStretch(0, 0)
        extraction_grid.setColumnStretch(1, 1)
        extraction_grid.setColumnStretch(2, 0)
        extraction_grid.setColumnStretch(3, 1)
        settings_label_layout.addLayout(extraction_grid)
        
        # Initialize entry widgets with config values
        if not hasattr(self, 'glossary_min_frequency_entry'):
            self.glossary_min_frequency_entry = QLineEdit()
            self.glossary_min_frequency_entry.setFixedWidth(80)
        self.glossary_min_frequency_entry.setText(str(self.config.get('glossary_min_frequency', 2)))
        
        if not hasattr(self, 'glossary_max_names_entry'):
            self.glossary_max_names_entry = QLineEdit()
            self.glossary_max_names_entry.setFixedWidth(80)
        self.glossary_max_names_entry.setText(str(self.config.get('glossary_max_names', 100)))
        
        if not hasattr(self, 'glossary_max_titles_entry'):
            self.glossary_max_titles_entry = QLineEdit()
            self.glossary_max_titles_entry.setFixedWidth(80)
        self.glossary_max_titles_entry.setText(str(self.config.get('glossary_max_titles', 50)))
        
        if not hasattr(self, 'glossary_context_window_entry'):
            self.glossary_context_window_entry = QLineEdit()
            self.glossary_context_window_entry.setFixedWidth(80)
        self.glossary_context_window_entry.setText(str(self.config.get('glossary_context_window', 2)))
        
        if not hasattr(self, 'glossary_max_text_size_entry'):
            self.glossary_max_text_size_entry = QLineEdit()
            self.glossary_max_text_size_entry.setFixedWidth(80)
        self.glossary_max_text_size_entry.setText(str(self.config.get('glossary_max_text_size', 0)))
        
        if not hasattr(self, 'glossary_chapter_split_threshold_entry'):
            self.glossary_chapter_split_threshold_entry = QLineEdit()
            self.glossary_chapter_split_threshold_entry.setFixedWidth(80)
        self.glossary_chapter_split_threshold_entry.setText(str(self.config.get('glossary_chapter_split_threshold', 0)))
        
        if not hasattr(self, 'glossary_max_sentences_entry'):
            self.glossary_max_sentences_entry = QLineEdit()
            self.glossary_max_sentences_entry.setFixedWidth(80)
        self.glossary_max_sentences_entry.setText(str(self.config.get('glossary_max_sentences', 200)))
        
        # Helper: compact label+field pair in one cell
        def _pair(label_text, field_widget, label_width=180):
            cont = QWidget()
            h = QHBoxLayout(cont)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            lbl = QLabel(label_text)
            lbl.setFixedWidth(label_width)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            h.addWidget(lbl)
            h.addWidget(field_widget)
            h.addStretch()
            return cont
        
        # Row 1 (left/right pairs)
        extraction_grid.addWidget(_pair("Min frequency:", self.glossary_min_frequency_entry), 0, 0, 1, 2)
        extraction_grid.addWidget(_pair("Max names:", self.glossary_max_names_entry), 0, 2, 1, 2)
        
        # Row 2
        extraction_grid.addWidget(_pair("Max titles:", self.glossary_max_titles_entry), 1, 0, 1, 2)
        extraction_grid.addWidget(_pair("Context window size:", self.glossary_context_window_entry), 1, 2, 1, 2)
        
        # Row 3 - Max text size and target language
        extraction_grid.addWidget(_pair("Max text size:", self.glossary_max_text_size_entry), 2, 0, 1, 2)
        
        # Target language dropdown
        if not hasattr(self, 'glossary_target_language_combo'):
            self.glossary_target_language_combo = QComboBox()
            self.glossary_target_language_combo.setMaximumWidth(120)
            self.glossary_target_language_combo.setEditable(True)
            languages = [
                "English", "Spanish", "French", "German", "Italian", "Portuguese",
                "Russian", "Arabic", "Hindi", "Chinese (Simplified)",
                "Chinese (Traditional)", "Japanese", "Korean", "Turkish"
            ]
            self.glossary_target_language_combo.addItems(languages)
            
            # Lock mousewheel scrolling on target language dropdown
            self._disable_combobox_mousewheel(self.glossary_target_language_combo)
            
            # Use icon in dropdown arrow like duplicate algorithm dropdown
            try:
                icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico')
                if os.path.exists(icon_path):
                    combo_style = """
                        QComboBox {
                            padding-right: 28px;
                        }
                        QComboBox::drop-down {
                            subcontrol-origin: padding;
                            subcontrol-position: top right;
                            width: 24px;
                            border-left: 1px solid #4a5568;
                        }
                        QComboBox::down-arrow {
                            width: 16px;
                            height: 16px;
                            image: url(""" + icon_path.replace('\\', '/') + """);
                        }
                        QComboBox::down-arrow:on {
                            top: 1px;
                        }
                    """
                    self.glossary_target_language_combo.setStyleSheet(combo_style)
            except Exception:
                pass
        
        saved_language = self.config.get('glossary_target_language', self.config.get('output_language', 'English'))
        index = self.glossary_target_language_combo.findText(saved_language)
        if index >= 0:
            self.glossary_target_language_combo.setCurrentIndex(index)
        else:
            self.glossary_target_language_combo.setCurrentText(saved_language)
        
        # Sync with manual glossary language dropdown and main GUI
        def sync_auto_to_manual(text):
            # Sync with manual glossary dropdown
            if hasattr(self, 'manual_target_language_combo'):
                manual_index = self.manual_target_language_combo.findText(text)
                if manual_index >= 0:
                    self.manual_target_language_combo.blockSignals(True)
                    self.manual_target_language_combo.setCurrentIndex(manual_index)
                    self.manual_target_language_combo.blockSignals(False)
            
            # Sync with main GUI dropdown
            if hasattr(self, 'target_lang_combo'):
                main_index = self.target_lang_combo.findText(text)
                if main_index >= 0:
                    self.target_lang_combo.blockSignals(True)
                    self.target_lang_combo.setCurrentIndex(main_index)
                    self.target_lang_combo.blockSignals(False)
                else:
                    self.target_lang_combo.blockSignals(True)
                    self.target_lang_combo.setCurrentText(text)
                    self.target_lang_combo.blockSignals(False)
            
            # Update configs
            self.config['glossary_target_language'] = text
            self.config['output_language'] = text
            os.environ['OUTPUT_LANGUAGE'] = text
        
        self.glossary_target_language_combo.currentTextChanged.connect(sync_auto_to_manual)
        
        extraction_grid.addWidget(_pair("Target language:", self.glossary_target_language_combo), 2, 2, 1, 2)
        
        # Row 4 - Max sentences and chapter split threshold
        # Max sentences for glossary (with inline hint)
        ms_cont = QWidget()
        ms_layout = QHBoxLayout(ms_cont)
        ms_layout.setContentsMargins(0, 0, 0, 0)
        ms_layout.setSpacing(6)
        ms_label = QLabel("Max sentences:")
        ms_label.setFixedWidth(180)
        ms_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        ms_layout.addWidget(ms_label)
        ms_layout.addWidget(self.glossary_max_sentences_entry)
        hint = QLabel("(Limit for AI processing)")
        hint.setStyleSheet("color: gray;")
        ms_layout.addWidget(hint)
        ms_layout.addStretch()
        extraction_grid.addWidget(ms_cont, 3, 0, 1, 2)
        
        extraction_grid.addWidget(_pair("Chapter split threshold:", self.glossary_chapter_split_threshold_entry), 3, 2, 1, 2)
        
        # Row 5 - Filter mode
        extraction_grid.addWidget(QLabel("Filter mode:"), 4, 0)
        filter_widget = QWidget()
        filter_layout = QHBoxLayout(filter_widget)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(8)
        extraction_grid.addWidget(filter_widget, 4, 1, 1, 3)
        
        if not hasattr(self, 'glossary_filter_mode_buttons'):
            self.glossary_filter_mode_buttons = {}
            filter_mode_value = self.config.get('glossary_filter_mode', 'all')
            
        radio1 = QRadioButton("All names & terms")
        radio1.setChecked(self.config.get('glossary_filter_mode', 'all') == 'all')
        self.glossary_filter_mode_buttons['all'] = radio1
        filter_layout.addWidget(radio1)
        
        radio2 = QRadioButton("Names with honorifics only")
        radio2.setChecked(self.config.get('glossary_filter_mode', 'all') == 'only_with_honorifics')
        self.glossary_filter_mode_buttons['only_with_honorifics'] = radio2
        filter_layout.addWidget(radio2)
        
        radio3 = QRadioButton("Names without honorifics & terms")
        radio3.setChecked(self.config.get('glossary_filter_mode', 'all') == 'only_without_honorifics')
        self.glossary_filter_mode_buttons['only_without_honorifics'] = radio3
        filter_layout.addWidget(radio3)
        filter_layout.addStretch()

        # Row 6 - Strip honorifics
        extraction_grid.addWidget(QLabel("Strip honorifics:"), 5, 0)
        if not hasattr(self, 'strip_honorifics_checkbox'):
            self.strip_honorifics_checkbox = self._create_styled_checkbox("Remove honorifics from extracted names")
        # Always reload from config
        self.strip_honorifics_checkbox.setChecked(self.config.get('strip_honorifics', True))
        extraction_grid.addWidget(self.strip_honorifics_checkbox, 5, 1, 1, 3)
        
        # Row 7 - Fuzzy matching threshold (reuse existing value)
        extraction_grid.addWidget(QLabel("Fuzzy threshold:"), 6, 0)
        
        auto_fuzzy_widget = QWidget()
        auto_fuzzy_layout = QHBoxLayout(auto_fuzzy_widget)
        auto_fuzzy_layout.setContentsMargins(0, 0, 0, 0)
        auto_fuzzy_layout.setSpacing(8)
        extraction_grid.addWidget(auto_fuzzy_widget, 6, 1, 1, 3)
        
        # Always reload fuzzy threshold value from config
        try:
            self.fuzzy_threshold_value = float(self.config.get('glossary_fuzzy_threshold', 0.90))
        except Exception:
            self.fuzzy_threshold_value = 0.90
        
        # Create slider and expose on self for save handler
        self.fuzzy_threshold_slider = QSlider(Qt.Horizontal)
        self.fuzzy_threshold_slider.setMinimum(50)
        self.fuzzy_threshold_slider.setMaximum(100)
        self.fuzzy_threshold_slider.setValue(int(self.fuzzy_threshold_value * 100))
        self.fuzzy_threshold_slider.setMinimumWidth(250)
        self._disable_slider_mousewheel(self.fuzzy_threshold_slider)  # Disable mouse wheel
        auto_fuzzy_layout.addWidget(self.fuzzy_threshold_slider)
        
        self.auto_fuzzy_value_label = QLabel(f"{self.fuzzy_threshold_value:.2f}")
        auto_fuzzy_layout.addWidget(self.auto_fuzzy_value_label)
        
        self.auto_fuzzy_desc_label = QLabel("")
        self.auto_fuzzy_desc_label.setStyleSheet("color: gray; font-size: 9pt;")
        auto_fuzzy_layout.addWidget(self.auto_fuzzy_desc_label)
        auto_fuzzy_layout.addStretch()
        
        # Update function for auto fuzzy slider
        def update_auto_fuzzy_label(value):
            float_value = value / 100.0
            self.fuzzy_threshold_value = float_value
            self.auto_fuzzy_value_label.setText(f"{float_value:.2f}")
            
            if float_value >= 0.95:
                desc = "Exact match only (strict)"
            elif float_value >= 0.85:
                desc = "Very similar names (recommended)"
            elif float_value >= 0.75:
                desc = "Moderately similar names"
            elif float_value >= 0.65:
                desc = "Loosely similar names"
            else:
                desc = "Very loose matching (may over-merge)"
            
            self.auto_fuzzy_desc_label.setText(desc)
            
            # Sync with manual glossary slider and labels if they exist
            if hasattr(self, 'manual_fuzzy_slider'):
                self.manual_fuzzy_slider.blockSignals(True)
                self.manual_fuzzy_slider.setValue(value)
                self.manual_fuzzy_slider.blockSignals(False)
                
                # Update manual labels directly without triggering signals
                if hasattr(self, 'manual_fuzzy_value_label') and hasattr(self, 'manual_fuzzy_desc_label'):
                    self.manual_fuzzy_value_label.setText(f"{float_value:.2f}")
                    self.manual_fuzzy_desc_label.setText(desc)
        
        # Store update function for cross-tab syncing
        self.update_auto_fuzzy_label_func = update_auto_fuzzy_label
        
        self.fuzzy_threshold_slider.valueChanged.connect(update_auto_fuzzy_label)
        update_auto_fuzzy_label(self.fuzzy_threshold_slider.value())
        
        # Row 8 - Reset to Defaults button
        reset_extraction_btn = QPushButton("Reset to Defaults")
        reset_extraction_btn.setStyleSheet("""
            QPushButton {
                background-color: #b8860b;
                color: black;
                padding: 5px 10px;
                border: 1px solid #8a6a08;
                border-radius: 4px;
                font-size: 9pt;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #9a6d07; }
            QPushButton:pressed { background-color: #8a6106; }
        """)
        
        def reset_extraction_settings():
            reply = QMessageBox.question(parent, "Reset Settings", 
                                         "Reset all extraction settings to defaults?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                # Reset all fields to defaults
                self.glossary_min_frequency_entry.setText("2")
                self.glossary_max_names_entry.setText("100")
                self.glossary_max_titles_entry.setText("50")
                self.glossary_context_window_entry.setText("2")
                self.glossary_max_text_size_entry.setText("0")
                self.glossary_chapter_split_threshold_entry.setText("0")
                self.glossary_max_sentences_entry.setText("200")
                self.glossary_target_language_combo.setCurrentText("English")
                
                # Reset filter mode to 'all'
                if 'all' in self.glossary_filter_mode_buttons:
                    self.glossary_filter_mode_buttons['all'].setChecked(True)
                
                # Reset strip honorifics to True
                if hasattr(self, 'strip_honorifics_checkbox'):
                    self.strip_honorifics_checkbox.setChecked(True)
                
                # Reset fuzzy threshold to 0.90
                if hasattr(self, 'fuzzy_threshold_slider'):
                    self.fuzzy_threshold_slider.setValue(90)
        
        reset_extraction_btn.clicked.connect(reset_extraction_settings)
        extraction_grid.addWidget(reset_extraction_btn, 7, 0, 1, 2)
        
        # Help text
        help_widget = QWidget()
        help_layout = QVBoxLayout(help_widget)
        help_layout.setContentsMargins(10, 10, 0, 0)
        extraction_tab_layout.addWidget(help_widget)
        
        help_title = QLabel("💡 Settings Guide:")
        help_title.setStyleSheet("font-size: 12pt; font-weight: bold;")
        help_layout.addWidget(help_title)
        
        help_texts = [
            "• Min frequency: How many times a name must appear (lower = more terms)",
            "• Max names/titles: Limits to prevent huge glossaries",
            "• Context window size: Number of sentences before/after for gender detection (default: 2)",
            "• Max text size: Characters to analyze (0 = entire text, 50000 = first 50k chars)",
            "• Chapter split: Split large texts into chunks (0 = no splitting, 100000 = split at 100k chars)",
            "• Max sentences: Maximum sentences to send to AI (default 200, increase for more context)",
            "• Filter mode:",
            "  - All names & terms: Extract character names (with/without honorifics) + titles/terms",
            "  - Names with honorifics only: ONLY character names with honorifics (no titles/terms)",
            "  - Names without honorifics & terms: Character names without honorifics + titles/terms",
            "• Strip honorifics: Remove suffixes from extracted names (e.g., '김' instead of '김님')",
            "• Fuzzy threshold: How similar terms must be to match (0.9 = 90% match, 1.0 = exact match)"
        ]
        for txt in help_texts:
            label = QLabel(txt)
            label.setStyleSheet("color: gray; font-size: 10pt; margin-left: 20px;")
            help_layout.addWidget(label)
        
        # Tab 2: Glossary Prompt (unified system + extraction)
        glossary_prompt_tab = QWidget()
        glossary_prompt_tab_layout = QVBoxLayout(glossary_prompt_tab)
        glossary_prompt_tab_layout.setContentsMargins(10, 10, 10, 10)
        notebook.addTab(glossary_prompt_tab, "Glossary Prompt")
        
        # Unified glossary prompt section
        glossary_prompt_frame = QGroupBox("Glossary Extraction Prompt")
        glossary_prompt_frame_layout = QVBoxLayout(glossary_prompt_frame)
        glossary_prompt_tab_layout.addWidget(glossary_prompt_frame)
        
        desc_label = QLabel("This prompt guides the AI to extract character names, terms, and titles from the text:")
        glossary_prompt_frame_layout.addWidget(desc_label)
        
        placeholder_label = QLabel("Available placeholders: {language}, {min_frequency}, {max_names}, {max_titles}")
        placeholder_label.setStyleSheet("color: #5a9fd4; font-size: 9pt; font-style: italic;")
        glossary_prompt_frame_layout.addWidget(placeholder_label)
        
        self.auto_prompt_text = QTextEdit()
        self.auto_prompt_text.setMinimumHeight(250)
        self.auto_prompt_text.setLineWrapMode(QTextEdit.WidgetWidth)
        glossary_prompt_frame_layout.addWidget(self.auto_prompt_text)
        
        # Default unified prompt (combines system + extraction instructions)
        default_unified_prompt = """You are a novel glossary extraction assistant.

You must strictly return ONLY CSV format with 3-5 columns in this exact order: type,raw_name,translated_name,gender,description.
For character entries, determine gender from context, leave empty if context is insufficient.
For non-character entries, leave gender empty.
The description column is optional and can contain brief context (role, location, significance).
Only include terms that actually appear in the text.
Do not use quotes around values unless they contain commas.

CRITICAL EXTRACTION RULES:
- Extract ONLY: Character names, Location names, Ability/Skill names, Item names, Organization names, Titles/Ranks
- Do NOT extract sentences, dialogue, actions, questions, or statements as glossary entries
- The raw_name and translated_name must be SHORT NOUNS ONLY (1-5 words max)
- REJECT entries that contain verbs or end with punctuation (?, !, .)
- REJECT entries starting with: "How", "What", "Why", "I", "He", "She", "They", "That's", "So", "Therefore", "Still", "But". (The description column is excluded from this restriction)
- Do NOT output any entries that are rejected by the above rules; skip them entirely
- If unsure whether something is a proper noun/name, skip it
- The description column must contain detailed context/explanation

Critical Requirement: The translated name column should be in {language}.

For example:
character,ᫀ뢔사난,Kim Sang-hyu,male
character,ᫀ간편헤,Gale Hardest  
character,ᫀ이히리ᄐ 나애,Dihirit Ade,female

Focus on identifying:
1. Character names with their honorifics
2. Important titles and ranks
3. Frequently mentioned terms (min frequency: {min_frequency})

Extract up to {max_names} character names and {max_titles} titles.
Prioritize names that appear with honorifics or in important contexts."""
        
        # Load from config or use default
        # Note: Ignoring old 'auto_glossary_prompt' key to force update to new prompt
        # Also treat empty strings as missing to ensure users get the new default
        self.unified_auto_glossary_prompt = self.config.get('unified_auto_glossary_prompt', default_unified_prompt)
        if not self.unified_auto_glossary_prompt or not self.unified_auto_glossary_prompt.strip():
            self.unified_auto_glossary_prompt = default_unified_prompt
        self.auto_prompt_text.setPlainText(self.unified_auto_glossary_prompt)
        
        glossary_prompt_controls_widget = QWidget()
        glossary_prompt_controls_layout = QHBoxLayout(glossary_prompt_controls_widget)
        glossary_prompt_controls_layout.setContentsMargins(0, 0, 0, 0)
        glossary_prompt_tab_layout.addWidget(glossary_prompt_controls_widget)
        
        def reset_glossary_prompt():
            reply = QMessageBox.question(parent, "Reset Prompt", "Reset glossary prompt to default?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.auto_prompt_text.setPlainText(default_unified_prompt)
        
        reset_glossary_btn = QPushButton("Reset to Default")
        reset_glossary_btn.clicked.connect(reset_glossary_prompt)
        reset_glossary_btn.setStyleSheet("""
            QPushButton {
                background-color: #b8860b;
                color: black;
                padding: 5px;
                border: 1px solid #8a6a08;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #9a6d07; }
            QPushButton:pressed { background-color: #8a6106; }
        """)
        glossary_prompt_controls_layout.addWidget(reset_glossary_btn)
        glossary_prompt_controls_layout.addStretch()
        
        # Format Instructions removed - now hardcoded to just append {text_sample}
        
        # Update states function with proper error handling - converted to use signals
        def update_auto_glossary_state(checked=None):
            enabled = self.enable_auto_glossary_checkbox.isChecked()
            
            # Enable/disable the entire Targeted Extraction Settings group box
            settings_label_frame.setEnabled(enabled)
            
            # Enable/disable all extraction grid widgets (for thorough coverage)
            for i in range(extraction_grid.count()):
                item = extraction_grid.itemAt(i)
                if item:
                    widget = item.widget()
                    if widget:
                        widget.setEnabled(enabled)
                        # Also enable/disable all children within compound widgets
                        for child in widget.findChildren(QWidget):
                            child.setEnabled(enabled)
            
            # Enable/disable text widgets
            self.auto_prompt_text.setEnabled(enabled)
        
        def update_append_prompt_state(checked=None):
            enabled = self.append_glossary_checkbox.isChecked()
            self.append_prompt_text.setEnabled(enabled)
        
        # Initialize states
        update_auto_glossary_state()
        update_append_prompt_state()
        
        # Connect signals
        self.enable_auto_glossary_checkbox.stateChanged.connect(update_auto_glossary_state)
        self.append_glossary_checkbox.stateChanged.connect(update_append_prompt_state)

    def _setup_glossary_editor_tab(self, parent):
        """Set up the glossary editor/trimmer tab"""
        # Create main layout
        editor_layout = QVBoxLayout(parent)
        editor_layout.setContentsMargins(10, 10, 10, 10)

        file_widget = QWidget()
        file_layout = QHBoxLayout(file_widget)
        file_layout.setContentsMargins(0, 0, 0, 10)
        editor_layout.addWidget(file_widget)

        file_layout.addWidget(QLabel("Glossary File:"))
        self.editor_file_entry = QLineEdit()
        self.editor_file_entry.setReadOnly(True)
        file_layout.addWidget(self.editor_file_entry)

        stats_widget = QWidget()
        stats_layout = QHBoxLayout(stats_widget)
        stats_layout.setContentsMargins(0, 0, 0, 5)
        editor_layout.addWidget(stats_widget)
        
        self.stats_label = QLabel("No glossary loaded")
        self.stats_label.setStyleSheet("font-size: 10pt; font-style: italic;")
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch()

        content_frame = QGroupBox("Glossary Entries")
        content_frame_layout = QVBoxLayout(content_frame)
        editor_layout.addWidget(content_frame)

        # Create tree widget
        self.glossary_tree = QTreeWidget()
        self.glossary_tree.setColumnCount(1)
        self.glossary_tree.setHeaderLabels(["#"])
        self.glossary_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.glossary_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        content_frame_layout.addWidget(self.glossary_tree)

        self.glossary_tree.itemDoubleClicked.connect(lambda item, col: self._on_tree_double_click(item, col))
        self.glossary_tree.customContextMenuRequested.connect(lambda pos: None)  # will be rebound after helpers are defined

        self.current_glossary_data = None
        self.current_glossary_format = None
        self.glossary_column_fields = []

        # Editor functions
        def load_glossary_for_editing():
           path = self.editor_file_entry.text()
           if not path or not os.path.exists(path):
               QMessageBox.critical(parent, "Error", "Please select a valid glossary file")
               return
           
           try:
               # Helpers for token-efficient format (sectioned, bullet-style CSV text)
               def parse_token_efficient_glossary(lines):
                   entries = []
                   sections = []
                   current_section = None
                   gender_keywords = {'male', 'female', 'unknown'}
                   header_columns = ['translated_name', 'raw_name', 'gender', 'description']
                   # Default extra columns: pattern manager + custom fields (used if header omits them)
                   default_extra_columns = []
                   try:
                       import PatternManager as _pm
                       pf = getattr(_pm, 'PATTERN_ADDITIONAL_FIELDS', [])
                       if isinstance(pf, (list, tuple)):
                           default_extra_columns.extend(pf)
                   except Exception:
                       pass
                   default_extra_columns.extend(self.config.get('custom_glossary_fields', []))
                   extra_columns = list(default_extra_columns)
                   # Map section -> type (from custom entry types only, plus simple plurals)
                   custom_types = getattr(self, 'custom_entry_types', {}) or {
                       'character': {'enabled': True, 'has_gender': True},
                       'term': {'enabled': True, 'has_gender': False},
                   }

                   type_map = {}
                   for t in custom_types.keys():
                       type_map[t.lower()] = t
                       # naive plural
                       if not t.lower().endswith('s'):
                           type_map[f"{t.lower()}s"] = t

                   for raw_line in lines:
                       line = raw_line.strip()
                       if not line:
                           continue
                       if line.lower().startswith('glossary columns:'):
                           # Parse header columns
                           cols_text = line.split(':', 1)[1]
                           header_columns = [c.strip() for c in cols_text.split(',') if c.strip()]
                           if len(header_columns) < 4:
                               header_columns = ['translated_name', 'raw_name', 'gender', 'description']
                           extra_columns = header_columns[4:] or list(default_extra_columns)
                           continue
                       if line.startswith('===') and line.endswith('==='):
                           section_name = line.strip('=').strip()
                           current_section = section_name
                           sections.append(section_name)
                           continue
                       if not line.startswith('* '):
                           continue

                       # Pattern: * translated (raw) [gender]: description
                       import re
                       m = re.match(r'^\*\s+(.*?)\s*(?:\((.*?)\))?\s*(?:\[(.*?)\])?\s*(?::\s*(.*))?$', line)
                       if not m:
                           continue
                       translated = (m.group(1) or '').strip()
                       raw_name = (m.group(2) or '').strip()
                       bracket = (m.group(3) or '').strip()
                       desc = (m.group(4) or '').strip()

                       # Split out extra column values encoded as " | key: val"
                       extra_values = {}
                       if desc and ' | ' in desc:
                           parts = desc.split(' | ')
                           desc = parts[0].strip()
                           for part in parts[1:]:
                               if ':' in part:
                                   k, v = part.split(':', 1)
                                   extra_values[k.strip()] = v.strip()

                       gender = ''
                       if bracket:
                           if bracket.lower() in gender_keywords:
                               gender = bracket
                           else:
                               # treat bracket content as description fragment
                               desc = f"{bracket}: {desc}".strip(': ').strip() if desc else bracket

                       entry = {
                           'type': type_map.get((current_section or 'term').lower(), 'term'),
                           'raw_name': raw_name,
                           'translated_name': translated,
                           'gender': gender,
                       }
                       if desc:
                           entry['description'] = desc
                       if current_section:
                           entry['_section'] = current_section
                       # Apply any extra columns from header
                       for col in extra_columns:
                           if col in extra_values:
                               entry[col] = extra_values[col]
                       entries.append(entry)

                   return entries, sections

               # Prepare accumulator for field discovery
               all_fields = set()

               # Try CSV first
               if path.endswith('.csv'):
                   # Peek to detect token-efficient format
                   with open(path, 'r', encoding='utf-8') as f:
                       lines = f.readlines()

                   token_style = False
                   for l in lines:
                       lstrip = l.lstrip()
                       if lstrip.startswith('===') or lstrip.startswith('* '):
                           token_style = True
                           break
                   if not token_style and lines and lines[0].lower().startswith('glossary columns:'):
                       token_style = True

                   if token_style:
                       entries, sections = parse_token_efficient_glossary(lines)
                       self.current_glossary_data = entries
                       self.current_glossary_format = 'token_csv'
                       self.current_glossary_sections = sections
                       for e in entries:
                           all_fields.update(e.keys())
                   else:
                       import csv
                       entries = []
                       with open(path, 'r', encoding='utf-8') as f:
                           reader = csv.reader(f)
                           for row in reader:
                               if len(row) >= 3:
                                   entry = {
                                       'type': row[0],
                                       'raw_name': row[1],
                                       'translated_name': row[2]
                                   }
                                   if row[0] == 'character' and len(row) > 3:
                                       entry['gender'] = row[3]
                                   # include any extra columns
                                   if len(row) > 4:
                                       entry['description'] = row[4]
                                   entries.append(entry)
                       self.current_glossary_data = entries
                       self.current_glossary_format = 'list'
                       for e in entries:
                           all_fields.update(e.keys())
               else:
                   # JSON format
                   with open(path, 'r', encoding='utf-8') as f:
                       data = json.load(f)
                   
                   entries = []
                   
                   if isinstance(data, dict):
                       if 'entries' in data:
                           self.current_glossary_data = data
                           self.current_glossary_format = 'dict'
                           for original, translated in data['entries'].items():
                               entry = {'original': original, 'translated': translated}
                               entries.append(entry)
                               all_fields.update(entry.keys())
                       else:
                           self.current_glossary_data = {'entries': data}
                           self.current_glossary_format = 'dict'
                           for original, translated in data.items():
                               entry = {'original': original, 'translated': translated}
                               entries.append(entry)
                               all_fields.update(entry.keys())
                   
                   elif isinstance(data, list):
                       self.current_glossary_data = data
                       self.current_glossary_format = 'list'
                       for item in data:
                           all_fields.update(item.keys())
                           entries.append(item)
               
               # Set up columns based on new format
               if self.current_glossary_format in ['list', 'token_csv'] and entries and 'type' in entries[0]:
                   # New simple format
                   column_fields = []
                   # Show section if present
                   if any('_section' in e for e in entries):
                       column_fields.append('_section')
                   column_fields.extend(['type', 'raw_name', 'translated_name', 'gender'])

                   # Include description/custom fields
                   for entry in entries:
                       for field in entry.keys():
                           if field.startswith('_'):
                               continue
                           if field not in column_fields:
                               column_fields.append(field)
                   
                   # Check for any custom fields
                   for entry in entries:
                       for field in entry.keys():
                           if field.startswith('_'):
                               continue
                           if field not in column_fields:
                               column_fields.append(field)
               else:
                   # Old format compatibility
                   standard_fields = ['original_name', 'name', 'original', 'translated', 'gender', 
                                    'title', 'group_affiliation', 'traits', 'how_they_refer_to_others', 
                                    'locations']
                   
                   column_fields = []
                   for field in standard_fields:
                       if field in all_fields:
                           column_fields.append(field)
                   
                   custom_fields = sorted(all_fields - set(standard_fields))
                   column_fields.extend(custom_fields)
               
               self.glossary_tree.clear()
               self.glossary_column_fields = list(column_fields)
               self.glossary_tree.setColumnCount(len(column_fields) + 1)  # +1 for index column
               
               headers = ['#'] + [field.replace('_', ' ').title() for field in column_fields]
               self.glossary_tree.setHeaderLabels(headers)
               
               self.glossary_tree.setColumnWidth(0, 80)
               
               for idx, field in enumerate(column_fields, start=1):
                   if field in ['raw_name', 'translated_name', 'original_name', 'name', 'original', 'translated']:
                       width = 150
                   elif field in ['traits', 'locations', 'how_they_refer_to_others']:
                       width = 200
                   else:
                       width = 100
                   
                   self.glossary_tree.setColumnWidth(idx, width)
               
               for idx, entry in enumerate(entries):
                   values = [str(idx + 1)]
                   for field in column_fields:
                       value = entry.get(field, '')
                       if isinstance(value, list):
                           value = ', '.join(str(v) for v in value)
                       elif isinstance(value, dict):
                           value = ', '.join(f"{k}: {v}" for k, v in value.items())
                       elif value is None:
                           value = ''
                       values.append(str(value))
                   
                   item = QTreeWidgetItem(values)
                   if self.current_glossary_format == 'dict':
                       item.setData(0, Qt.UserRole, entry.get('original', ''))
                   else:
                       item.setData(0, Qt.UserRole, idx)
                   self.glossary_tree.addTopLevelItem(item)
               
               # Update stats
               stats = []
               stats.append(f"Total entries: {len(entries)}")
               
               if self.current_glossary_format in ['list', 'token_csv'] and entries and 'type' in entries[0]:
                   # New format stats
                   characters = sum(1 for e in entries if e.get('type') == 'character')
                   terms = sum(1 for e in entries if e.get('type') == 'term')
                   stats.append(f"Characters: {characters}, Terms: {terms}")
               elif self.current_glossary_format == 'list':
                   # Old format stats
                   chars = sum(1 for e in entries if 'original_name' in e or 'name' in e)
                   locs = sum(1 for e in entries if 'locations' in e and e['locations'])
                   stats.append(f"Characters: {chars}, Locations: {locs}")
               
               self.stats_label.setText(" | ".join(stats))
               self.append_log(f"✅ Loaded {len(entries)} entries from glossary")
               self._last_find_text = ""
               self._last_find_pos = -1
               
           except Exception as e:
               QMessageBox.critical(parent, "Error", f"Failed to load glossary: {e}")
               self.append_log(f"❌ Failed to load glossary: {e}")
       
        def browse_glossary():
           path, _ = QFileDialog.getOpenFileName(
               parent,
               "Select glossary file",
               "",
               "Glossary files (*.json *.csv);;JSON files (*.json);;CSV files (*.csv)"
           )
           if path:
               self.editor_file_entry.setText(path)
               load_glossary_for_editing()

        # Common save helper
        def save_current_glossary():
           path = self.editor_file_entry.text()
           if not path or not self.current_glossary_data:
               return False
           try:
               if path.endswith('.csv'):
                   if getattr(self, 'current_glossary_format', '') == 'token_csv':
                       def save_token_csv(entries, path_out):
                           sections = getattr(self, 'current_glossary_sections', []) or []
                           if not sections:
                               sections = ['CHARACTERS', 'TITLES', 'ORGANIZATIONS', 'LOCATIONS', 'ITEMS', 'ABILITYS']

                           grouped = {sec: [] for sec in sections}
                           default_map = {'character': 'CHARACTERS', 'term': 'TITLES'}
                           for entry in entries:
                               sec = entry.get('_section')
                               if not sec:
                                   sec = default_map.get(entry.get('type', 'term'), 'TITLES')
                               if sec not in grouped:
                                   grouped[sec] = []
                                   sections.append(sec)
                               grouped[sec].append(entry)

                           # Build header columns: standard + pattern-manager fields + custom/additional fields
                           standard_cols = ['translated_name', 'raw_name', 'gender', 'description']
                           pattern_fields = []
                           try:
                               import PatternManager as _pm
                               pf = getattr(_pm, 'PATTERN_ADDITIONAL_FIELDS', [])
                               if isinstance(pf, (list, tuple)):
                                   pattern_fields = list(pf)
                           except Exception:
                               pattern_fields = []

                           custom_fields = self.config.get('custom_glossary_fields', [])
                           # include any fields present in data that are not internal/standard
                           data_fields = []
                           for e in entries:
                               for k in e.keys():
                                   if k.startswith('_') or k in ['type'] + standard_cols:
                                       continue
                                   if k not in custom_fields and k not in pattern_fields and k not in data_fields:
                                       data_fields.append(k)
                           header_cols = standard_cols + pattern_fields + custom_fields + data_fields

                           lines = [f"Glossary Columns: {', '.join(header_cols)}", ""]
                           for sec in sections:
                               sec_entries = grouped.get(sec, [])
                               if not sec_entries:
                                   continue
                               lines.append(f"=== {sec} ===")
                               for e in sec_entries:
                                   translated = e.get('translated_name', '')
                                   raw_name = e.get('raw_name', '')
                                   gender = e.get('gender', '')
                                   desc = e.get('description', '')

                                   line = f"* {translated}"
                                   if raw_name:
                                       line += f" ({raw_name})"
                                   if gender:
                                       line += f" [{gender}]"
                                   extra_tail = []
                                   for col in header_cols[4:]:
                                       val = e.get(col, '')
                                       if val:
                                           extra_tail.append(f"{col}: {val}")
                                   if desc:
                                       line += f": {desc}"
                                   if extra_tail:
                                       tail_str = " | ".join(extra_tail)
                                       line += f" | {tail_str}" if desc else f": {tail_str}"
                                   lines.append(line)
                               lines.append("")

                           with open(path_out, 'w', encoding='utf-8', newline='') as f:
                               f.write("\n".join(lines).rstrip() + "\n")

                       save_token_csv(self.current_glossary_data, path)
                   else:
                       import csv
                       standard_fields = ['type', 'raw_name', 'translated_name', 'gender']
                       extra_fields = []
                       for entry in self.current_glossary_data:
                           for k in entry.keys():
                               if k.startswith('_') or k in standard_fields:
                                   continue
                               if k not in extra_fields:
                                   extra_fields.append(k)
                       with open(path, 'w', encoding='utf-8', newline='') as f:
                           writer = csv.writer(f)
                           for entry in self.current_glossary_data:
                               row = [
                                   entry.get('type', ''),
                                   entry.get('raw_name', ''),
                                   entry.get('translated_name', ''),
                                   entry.get('gender', '')
                               ]
                               for field in extra_fields:
                                   row.append(entry.get(field, ''))
                               writer.writerow(row)
               else:
                   with open(path, 'w', encoding='utf-8') as f:
                       json.dump(self.current_glossary_data, f, ensure_ascii=False, indent=2)
               return True
           except Exception as e:
               QMessageBox.critical(parent, "Error", f"Failed to save: {e}")
               return False

        def clean_empty_fields():
            if not self.current_glossary_data:
                QMessageBox.critical(parent, "Error", "No glossary loaded")
                return
            
            if self.current_glossary_format in ['list', 'token_csv']:
                # Check if there are any empty fields
                empty_fields_found = False
                fields_cleaned = {}
                
                # Count empty fields first
                for entry in self.current_glossary_data:
                    for field in list(entry.keys()):
                        value = entry[field]
                        if value is None or value == "" or (isinstance(value, list) and len(value) == 0) or (isinstance(value, dict) and len(value) == 0):
                            empty_fields_found = True
                            fields_cleaned[field] = fields_cleaned.get(field, 0) + 1
                
                # If no empty fields found, show message and return
                if not empty_fields_found:
                    QMessageBox.information(parent, "Info", "No empty fields found in glossary")
                    return
                
                # Only create backup if there are fields to clean
                if not self.create_glossary_backup("before_clean"):
                    return
                
                # Now actually clean the fields
                total_cleaned = 0
                for entry in self.current_glossary_data:
                    for field in list(entry.keys()):
                        value = entry[field]
                        if value is None or value == "" or (isinstance(value, list) and len(value) == 0) or (isinstance(value, dict) and len(value) == 0):
                            entry.pop(field)
                            total_cleaned += 1
                
                if save_current_glossary():
                    load_glossary_for_editing()
                    
                    # Provide detailed feedback
                    msg = f"Cleaned {total_cleaned} empty fields\n\n"
                    msg += "Fields cleaned:\n"
                    for field, count in sorted(fields_cleaned.items(), key=lambda x: x[1], reverse=True):
                        msg += f"• {field}: {count} entries\n"
                    
                    QMessageBox.information(parent, "Success", msg)
        
        def delete_selected_entries():
            selected = self.glossary_tree.selectedItems()
            if not selected:
                QMessageBox.warning(parent, "No Selection", "Please select entries to delete")
                return
            
            count = len(selected)
            reply = QMessageBox.question(parent, "Confirm Delete", f"Delete {count} selected entries?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                # automatic backup
                if not self.create_glossary_backup(f"before_delete_{count}"):
                    return
                    
                indices_to_delete = []
                for item in selected:
                   idx = int(item.text(0)) - 1  # First column is index
                   indices_to_delete.append(idx)

                indices_to_delete.sort(reverse=True)

                if self.current_glossary_format in ['list', 'token_csv']:
                   for idx in indices_to_delete:
                       if 0 <= idx < len(self.current_glossary_data):
                           del self.current_glossary_data[idx]

                elif self.current_glossary_format == 'dict':
                   entries_list = list(self.current_glossary_data.get('entries', {}).items())
                   for idx in indices_to_delete:
                       if 0 <= idx < len(entries_list):
                           key = entries_list[idx][0]
                           self.current_glossary_data['entries'].pop(key, None)

                if save_current_glossary():
                   load_glossary_for_editing()
                   QMessageBox.information(parent, "Success", f"Deleted {len(indices_to_delete)} entries")
                
        def remove_duplicates():
            if not self.current_glossary_data:
                QMessageBox.critical(parent, "Error", "No glossary loaded")
                return
            
            if self.current_glossary_format in ['list', 'token_csv']:
                # Import the skip function from the updated script
                try:
                    from extract_glossary_from_epub import skip_duplicate_entries, remove_honorifics
                    
                    # Set environment variable for honorifics toggle
                    os.environ['GLOSSARY_DISABLE_HONORIFICS_FILTER'] = '1' if self.config.get('glossary_disable_honorifics_filter', False) else '0'
                    
                    original_count = len(self.current_glossary_data)
                    self.current_glossary_data = skip_duplicate_entries(self.current_glossary_data)
                    duplicates_removed = original_count - len(self.current_glossary_data)
                    
                    if duplicates_removed > 0:
                        if self.config.get('glossary_auto_backup', False):
                            self.create_glossary_backup(f"before_remove_{duplicates_removed}_dupes")
                        
                        if save_current_glossary():
                            load_glossary_for_editing()
                            QMessageBox.information(parent, "Success", f"Removed {duplicates_removed} duplicate entries")
                            self.append_log(f"🗑️ Removed {duplicates_removed} duplicates based on raw_name")
                    else:
                        QMessageBox.information(parent, "Info", "No duplicates found")
                        
                except ImportError:
                    # Fallback implementation
                    seen_raw_names = set()
                    unique_entries = []
                    duplicates = 0
                    
                    for entry in self.current_glossary_data:
                        raw_name = entry.get('raw_name', '').lower().strip()
                        if raw_name and raw_name not in seen_raw_names:
                            seen_raw_names.add(raw_name)
                            unique_entries.append(entry)
                        elif raw_name:
                            duplicates += 1
                    
                    if duplicates > 0:
                        self.current_glossary_data = unique_entries
                        if save_current_glossary():
                            load_glossary_for_editing()
                            QMessageBox.information(parent, "Success", f"Removed {duplicates} duplicate entries")
                    else:
                        QMessageBox.information(parent, "Info", "No duplicates found")

        # dialog function for configuring duplicate detection mode
        def duplicate_detection_settings():
            """Show info about duplicate detection (simplified for new format)"""
            QMessageBox.information(
                parent,
                "Duplicate Detection", 
                "Duplicate detection is based on the raw_name field.\n\n"
                "• Entries with identical raw_name values are considered duplicates\n"
                "• The first occurrence is kept, later ones are removed\n"
                "• Honorifics filtering can be toggled in the Manual Glossary tab\n\n"
                "When honorifics filtering is enabled, names are compared after removing honorifics."
            )

        def backup_settings_dialog():
            """Show dialog for configuring automatic backup settings"""
            # Create dialog
            backup_dialog = QDialog(parent)
            backup_dialog.setWindowTitle("Automatic Backup Settings")
            # Use screen ratios for sizing
            screen = QApplication.primaryScreen().geometry()
            width = int(screen.width() * 0.26)  # 26% of screen width
            height = int(screen.height() * 0.39)  # 39% of screen height
            backup_dialog.setMinimumSize(width, height)
            
            # Main layout
            main_layout = QVBoxLayout(backup_dialog)
            main_layout.setContentsMargins(20, 20, 20, 20)
            
            # Title
            title_label = QLabel("Automatic Backup Settings")
            title_label.setStyleSheet("font-size: 22pt; font-weight: bold;")
            main_layout.addWidget(title_label)
            main_layout.addSpacing(20)
            
            # Backup toggle
            backup_checkbox = self._create_styled_checkbox("Enable automatic backups before modifications")
            backup_checkbox.setChecked(self.config.get('glossary_auto_backup', True))
            main_layout.addWidget(backup_checkbox)
            main_layout.addSpacing(5)
            
            # Settings frame (indented)
            settings_widget = QWidget()
            settings_layout = QVBoxLayout(settings_widget)
            settings_layout.setContentsMargins(20, 10, 0, 0)
            main_layout.addWidget(settings_widget)
            
            # Max backups setting
            max_backups_widget = QWidget()
            max_backups_layout = QHBoxLayout(max_backups_widget)
            max_backups_layout.setContentsMargins(0, 5, 0, 5)
            settings_layout.addWidget(max_backups_widget)
            
            max_backups_layout.addWidget(QLabel("Maximum backups to keep:"))
            max_backups_spinbox = QSpinBox()
            max_backups_spinbox.setRange(0, 999)
            max_backups_spinbox.setValue(self.config.get('glossary_max_backups', 50))
            max_backups_spinbox.setFixedWidth(80)
            self._disable_spinbox_mousewheel(max_backups_spinbox)  # Disable mouse wheel
            max_backups_layout.addWidget(max_backups_spinbox)
            
            unlimited_label = QLabel("(0 = unlimited)")
            unlimited_label.setStyleSheet("color: gray; font-size: 9pt;")
            max_backups_layout.addWidget(unlimited_label)
            max_backups_layout.addStretch()
            
            # Backup naming pattern info
            settings_layout.addSpacing(15)
            
            pattern_label = QLabel("Backup naming pattern:")
            pattern_label.setStyleSheet("font-weight: bold;")
            settings_layout.addWidget(pattern_label)
            
            pattern_text = QLabel("[original_name]_[operation]_[YYYYMMDD_HHMMSS].json")
            pattern_text.setStyleSheet("color: #666; font-style: italic; font-size: 9pt; margin-left: 10px;")
            settings_layout.addWidget(pattern_text)
            
            # Example
            example_text = "Example: my_glossary_before_delete_5_20240115_143052.json"
            example_label = QLabel(example_text)
            example_label.setStyleSheet("color: gray; font-size: 8pt; margin-left: 10px; margin-top: 2px;")
            settings_layout.addWidget(example_label)
            
            # Separator
            main_layout.addSpacing(20)
            separator = QFrame()
            separator.setFrameShape(QFrame.HLine)
            separator.setFrameShadow(QFrame.Sunken)
            main_layout.addWidget(separator)
            main_layout.addSpacing(15)
            
            # Backup location info
            location_label = QLabel("📁 Backup Location:")
            location_label.setStyleSheet("font-weight: bold;")
            main_layout.addWidget(location_label)
            
            if self.editor_file_entry.text():
                glossary_dir = os.path.dirname(self.editor_file_entry.text())
                backup_path = "Backups"
                full_path = os.path.join(glossary_dir, "Backups")
                
                path_label = QLabel(f"{backup_path}/")
                path_label.setStyleSheet("color: #7bb3e0; font-size: 9pt; margin-left: 10px;")
                main_layout.addWidget(path_label)
                
                # Check if backup folder exists and show count
                if os.path.exists(full_path):
                    backup_count = len([f for f in os.listdir(full_path) if f.endswith('.json')])
                    count_label = QLabel(f"Currently contains {backup_count} backup(s)")
                    count_label.setStyleSheet("color: gray; font-size: 8pt; margin-left: 10px;")
                    main_layout.addWidget(count_label)
            else:
                backup_label = QLabel("Backups")
                backup_label.setStyleSheet("color: gray; font-size: 9pt; margin-left: 10px;")
                main_layout.addWidget(backup_label)
            
            def toggle_settings_state(checked):
                max_backups_spinbox.setEnabled(backup_checkbox.isChecked())
            
            backup_checkbox.stateChanged.connect(toggle_settings_state)
            toggle_settings_state(backup_checkbox.isChecked())  # Set initial state
            
            # Buttons
            main_layout.addSpacing(25)
            
            button_widget = QWidget()
            button_layout = QHBoxLayout(button_widget)
            button_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.addWidget(button_widget)
            
            button_layout.addStretch()
            
            def save_settings():
                # Save backup settings
                self.config['glossary_auto_backup'] = backup_checkbox.isChecked()
                self.config['glossary_max_backups'] = max_backups_spinbox.value()
                
                # Save to config file
                CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')
                with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                
                status = "enabled" if backup_checkbox.isChecked() else "disabled"
                if backup_checkbox.isChecked():
                    limit = max_backups_spinbox.value()
                    limit_text = "unlimited" if limit == 0 else f"max {limit}"
                    msg = f"Automatic backups {status} ({limit_text})"
                else:
                    msg = f"Automatic backups {status}"
                    
                QMessageBox.information(backup_dialog, "Success", msg)
                backup_dialog.accept()
            
            def create_manual_backup():
                """Create a manual backup right now"""
                if not self.current_glossary_data:
                    QMessageBox.critical(backup_dialog, "Error", "No glossary loaded")
                    return
                    
                if self.create_glossary_backup("manual"):
                    QMessageBox.information(backup_dialog, "Success", "Manual backup created successfully!")
            
            save_btn = QPushButton("Save Settings")
            save_btn.setFixedWidth(120)
            save_btn.clicked.connect(save_settings)
            save_btn.setStyleSheet("background-color: #28a745; color: white; padding: 8px;")
            button_layout.addWidget(save_btn)
            
            backup_now_btn = QPushButton("Backup Now")
            backup_now_btn.setFixedWidth(120)
            backup_now_btn.clicked.connect(create_manual_backup)
            backup_now_btn.setStyleSheet("background-color: #17a2b8; color: white; padding: 8px;")
            button_layout.addWidget(backup_now_btn)
            
            cancel_btn = QPushButton("Cancel")
            cancel_btn.setFixedWidth(120)
            cancel_btn.clicked.connect(backup_dialog.reject)
            cancel_btn.setStyleSheet("background-color: #6c757d; color: white; padding: 8px;")
            button_layout.addWidget(cancel_btn)
            
            button_layout.addStretch()
            
            # Show dialog with fade animation
            try:
                from dialog_animations import exec_dialog_with_fade
                exec_dialog_with_fade(backup_dialog, duration=250)
            except Exception:
                backup_dialog.exec()
    
        def smart_trim_dialog():
            if not self.current_glossary_data:
                QMessageBox.critical(parent, "Error", "No glossary loaded")
                return
            
            # Create dialog
            trim_dialog = QDialog(parent)
            trim_dialog.setWindowTitle("Smart Trim Glossary")
            # Use screen ratios for sizing
            screen = QApplication.primaryScreen().geometry()
            width = int(screen.width() * 0.31)  # 31% of screen width
            height = int(screen.height() * 0.49)  # 49% of screen height
            trim_dialog.setMinimumSize(width, height)
            
            main_layout = QVBoxLayout(trim_dialog)
            main_layout.setContentsMargins(20, 20, 20, 20)
            
            # Title and description
            title = QLabel("Smart Glossary Trimming")
            title.setStyleSheet("font-size: 14pt; font-weight: bold;")
            main_layout.addWidget(title)
            
            desc = QLabel("Limit the number of entries in your glossary")
            desc.setStyleSheet("color: gray; font-size: 10pt;")
            desc.setWordWrap(True)
            main_layout.addWidget(desc)
            main_layout.addSpacing(15)
            
            # Display current glossary stats
            stats_group = QGroupBox("Current Glossary Statistics")
            stats_layout = QVBoxLayout(stats_group)
            main_layout.addWidget(stats_group)
            
            entry_count = len(self.current_glossary_data) if self.current_glossary_format in ['list', 'token_csv'] else len(self.current_glossary_data.get('entries', {}))
            stats_layout.addWidget(QLabel(f"Total entries: {entry_count}"))
            
            # For new format, show type breakdown
            if self.current_glossary_format in ['list', 'token_csv'] and self.current_glossary_data and 'type' in self.current_glossary_data[0]:
                characters = sum(1 for e in self.current_glossary_data if e.get('type') == 'character')
                terms = sum(1 for e in self.current_glossary_data if e.get('type') == 'term')
                stats_layout.addWidget(QLabel(f"Characters: {characters}, Terms: {terms}"))
            
            main_layout.addSpacing(15)
            
            # Entry limit section
            limit_group = QGroupBox("Entry Limit")
            limit_layout = QVBoxLayout(limit_group)
            main_layout.addWidget(limit_group)
            
            limit_desc = QLabel("Keep only the first N entries to reduce glossary size")
            limit_desc.setStyleSheet("color: gray; font-size: 9pt;")
            limit_desc.setWordWrap(True)
            limit_layout.addWidget(limit_desc)
            
            top_widget = QWidget()
            top_layout = QHBoxLayout(top_widget)
            top_layout.setContentsMargins(0, 5, 0, 0)
            limit_layout.addWidget(top_widget)
            
            top_layout.addWidget(QLabel("Keep first"))
            top_entry = QLineEdit(str(min(100, entry_count)))
            top_entry.setFixedWidth(80)
            top_layout.addWidget(top_entry)
            top_layout.addWidget(QLabel(f"entries (out of {entry_count})"))
            top_layout.addStretch()
            
            main_layout.addSpacing(15)
            
            # Preview section
            preview_group = QGroupBox("Preview")
            preview_layout = QVBoxLayout(preview_group)
            main_layout.addWidget(preview_group)
            
            preview_label = QLabel("Click 'Preview Changes' to see the effect")
            preview_label.setStyleSheet("color: gray; font-size: 10pt;")
            preview_layout.addWidget(preview_label)
            
            def preview_changes():
                try:
                    top_n = int(top_entry.text())
                    entries_to_remove = max(0, entry_count - top_n)
                    
                    preview_text = f"Preview of changes:\n"
                    preview_text += f"• Entries: {entry_count} → {top_n} ({entries_to_remove} removed)\n"
                    
                    preview_label.setText(preview_text)
                    preview_label.setStyleSheet("color: #7bb3e0; font-size: 10pt;")
                    
                except ValueError:
                    preview_label.setText("Please enter a valid number")
                    preview_label.setStyleSheet("color: red; font-size: 10pt;")
            
            preview_btn = QPushButton("Preview Changes")
            preview_btn.clicked.connect(preview_changes)
            preview_btn.setStyleSheet("background-color: #17a2b8; color: white; padding: 5px;")
            preview_layout.addWidget(preview_btn)
            
            main_layout.addSpacing(10)
            
            # Action buttons
            button_widget = QWidget()
            button_layout = QHBoxLayout(button_widget)
            main_layout.addWidget(button_widget)
            
            def apply_smart_trim():
                try:
                    top_n = int(top_entry.text())
                    
                    # Calculate how many entries will be removed
                    entries_to_remove = len(self.current_glossary_data) - top_n
                    if entries_to_remove > 0:
                        if not self.create_glossary_backup(f"before_trim_{entries_to_remove}"):
                            return
                    
                    if self.current_glossary_format in ['list', 'token_csv']:
                        # Keep only top N entries
                        if top_n < len(self.current_glossary_data):
                            self.current_glossary_data = self.current_glossary_data[:top_n]
                    
                    elif self.current_glossary_format == 'dict':
                        # For dict format, only support entry limit
                        entries = list(self.current_glossary_data['entries'].items())
                        if top_n < len(entries):
                            self.current_glossary_data['entries'] = dict(entries[:top_n])
                    
                    if save_current_glossary():
                        load_glossary_for_editing()
                        
                        QMessageBox.information(trim_dialog, "Success", f"Trimmed glossary to {top_n} entries")
                        trim_dialog.accept()
                        
                except ValueError:
                    QMessageBox.critical(trim_dialog, "Error", "Please enter valid numbers")

            button_layout.addStretch()
            
            apply_btn = QPushButton("Apply Trim")
            apply_btn.setFixedWidth(120)
            apply_btn.clicked.connect(apply_smart_trim)
            apply_btn.setStyleSheet("background-color: #28a745; color: white; padding: 8px;")
            button_layout.addWidget(apply_btn)
            
            cancel_btn = QPushButton("Cancel")
            cancel_btn.setFixedWidth(120)
            cancel_btn.clicked.connect(trim_dialog.reject)
            cancel_btn.setStyleSheet("background-color: #6c757d; color: white; padding: 8px;")
            button_layout.addWidget(cancel_btn)
            
            button_layout.addStretch()

            # Info section at bottom
            main_layout.addSpacing(20)
            tip_label = QLabel("💡 Tip: Entries are kept in their original order")
            tip_label.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
            main_layout.addWidget(tip_label)

            # Show dialog with fade animation
            try:
                from dialog_animations import exec_dialog_with_fade
                exec_dialog_with_fade(trim_dialog, duration=250)
            except Exception:
                trim_dialog.exec()
       
        def filter_entries_dialog():
            if not self.current_glossary_data:
                QMessageBox.critical(self.dialog, "Error", "No glossary loaded")
                return
            
            # Create dialog with scroll area
            filter_dialog = QDialog(self.dialog)
            filter_dialog.setWindowTitle("Filter Entries")
            filter_dialog.setMinimumWidth(600)
            
            main_layout = QVBoxLayout(filter_dialog)
            
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            main_layout.addWidget(scroll_area)
            
            main_frame = QWidget()
            content_layout = QVBoxLayout(main_frame)
            scroll_area.setWidget(main_frame)
            
            # Title and description
            title_label = QLabel("Filter Glossary Entries")
            title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
            content_layout.addWidget(title_label)
            content_layout.addSpacing(5)
            
            desc_label = QLabel("Filter entries by type or content")
            desc_label.setStyleSheet("color: gray; font-size: 10pt;")
            desc_label.setWordWrap(True)
            content_layout.addWidget(desc_label)
            content_layout.addSpacing(15)
            
            # Current stats
            entry_count = len(self.current_glossary_data) if self.current_glossary_format in ['list', 'token_csv'] else len(self.current_glossary_data.get('entries', {}))
            
            stats_group = QGroupBox("Current Status")
            stats_layout = QVBoxLayout(stats_group)
            stats_label = QLabel(f"Total entries: {entry_count}")
            stats_label.setStyleSheet("font-size: 10pt;")
            stats_layout.addWidget(stats_label)
            content_layout.addWidget(stats_group)
            content_layout.addSpacing(15)
            
            # Check if new format
            is_new_format = (self.current_glossary_format in ['list', 'token_csv'] and 
                           self.current_glossary_data and 
                           'type' in self.current_glossary_data[0])
            
            # Filter conditions
            conditions_group = QGroupBox("Filter Conditions")
            conditions_layout = QVBoxLayout(conditions_group)
            content_layout.addWidget(conditions_group)
            content_layout.addSpacing(15)
            
            # Type filter for new format
            type_checks = {}
            if is_new_format:
                type_group = QGroupBox("Entry Type")
                type_layout = QVBoxLayout(type_group)
                conditions_layout.addWidget(type_group)
                conditions_layout.addSpacing(10)
                
                char_check = self._create_styled_checkbox("Keep characters")
                char_check.setChecked(True)
                type_checks['character'] = char_check
                type_layout.addWidget(char_check)
                
                term_check = self._create_styled_checkbox("Keep terms/locations")
                term_check.setChecked(True)
                type_checks['term'] = term_check
                type_layout.addWidget(term_check)
            
            # Text content filter
            text_filter_group = QGroupBox("Text Content Filter")
            text_filter_layout = QVBoxLayout(text_filter_group)
            conditions_layout.addWidget(text_filter_group)
            conditions_layout.addSpacing(10)
            
            text_hint_label = QLabel("Keep entries containing text (case-insensitive):")
            text_hint_label.setStyleSheet("color: gray; font-size: 9pt;")
            text_filter_layout.addWidget(text_hint_label)
            text_filter_layout.addSpacing(5)
            
            search_entry = QLineEdit()
            text_filter_layout.addWidget(search_entry)
            
            # Gender filter for new format
            gender_value = "all"
            gender_buttons = {}
            if is_new_format:
                gender_group = QGroupBox("Gender Filter (Characters Only)")
                gender_layout = QVBoxLayout(gender_group)
                conditions_layout.addWidget(gender_group)
                conditions_layout.addSpacing(10)
                
                gender_button_group = QButtonGroup(filter_dialog)
                
                all_radio = QRadioButton("All genders")
                all_radio.setChecked(True)
                gender_buttons['all'] = all_radio
                gender_button_group.addButton(all_radio, 0)
                gender_layout.addWidget(all_radio)
                
                male_radio = QRadioButton("Male only")
                gender_buttons['Male'] = male_radio
                gender_button_group.addButton(male_radio, 1)
                gender_layout.addWidget(male_radio)
                
                female_radio = QRadioButton("Female only")
                gender_buttons['Female'] = female_radio
                gender_button_group.addButton(female_radio, 2)
                gender_layout.addWidget(female_radio)
                
                unknown_radio = QRadioButton("Unknown only")
                gender_buttons['Unknown'] = unknown_radio
                gender_button_group.addButton(unknown_radio, 3)
                gender_layout.addWidget(unknown_radio)
                
                def update_gender_value():
                    nonlocal gender_value
                    for val, btn in gender_buttons.items():
                        if btn.isChecked():
                            gender_value = val
                            break
                
                gender_button_group.buttonClicked.connect(update_gender_value)
            
            # Preview section
            preview_group = QGroupBox("Preview")
            preview_layout = QVBoxLayout(preview_group)
            content_layout.addWidget(preview_group)
            content_layout.addSpacing(15)
            
            preview_label = QLabel("Click 'Preview Filter' to see how many entries match")
            preview_label.setStyleSheet("color: gray; font-size: 10pt;")
            preview_layout.addWidget(preview_label)
            
            def check_entry_matches(entry):
                """Check if an entry matches the filter conditions"""
                # Type filter
                if is_new_format and entry.get('type'):
                    type_check = type_checks.get(entry['type'])
                    if type_check and not type_check.isChecked():
                        return False
                
                # Text filter
                search_text = search_entry.text().strip().lower()
                if search_text:
                    # Search in all text fields
                    entry_text = ' '.join(str(v) for v in entry.values() if isinstance(v, str)).lower()
                    if search_text not in entry_text:
                        return False
                
                # Gender filter
                if is_new_format and gender_value != "all":
                    if entry.get('type') == 'character' and entry.get('gender') != gender_value:
                        return False
                
                return True
            
            def preview_filter():
                """Preview the filter results"""
                nonlocal gender_value
                # Update gender value first
                for val, btn in gender_buttons.items():
                    if btn.isChecked():
                        gender_value = val
                        break
                
                matching = 0
                
                if self.current_glossary_format in ['list', 'token_csv']:
                    for entry in self.current_glossary_data:
                        if check_entry_matches(entry):
                            matching += 1
                else:
                    for key, entry in self.current_glossary_data.get('entries', {}).items():
                        if check_entry_matches(entry):
                            matching += 1
                
                removed = entry_count - matching
                preview_label.setText(f"Filter matches: {matching} entries ({removed} will be removed)")
                preview_label.setStyleSheet(f"color: {'#5a9fd4' if matching > 0 else 'red'}; font-size: 10pt; font-style: italic;")
            
            preview_btn = QPushButton("Preview Filter")
            preview_btn.clicked.connect(preview_filter)
            preview_btn.setStyleSheet("background-color: #0dcaf0; color: white; padding: 8px;")
            preview_layout.addWidget(preview_btn)
            
            # Action buttons
            content_layout.addSpacing(10)
            button_layout = QHBoxLayout()
            content_layout.addLayout(button_layout)
            content_layout.addSpacing(20)
            
            def apply_filter():
                nonlocal gender_value
                # Update gender value first
                for val, btn in gender_buttons.items():
                    if btn.isChecked():
                        gender_value = val
                        break
                
                if self.current_glossary_format in ['list', 'token_csv']:
                    filtered = []
                    for entry in self.current_glossary_data:
                        if check_entry_matches(entry):
                            filtered.append(entry)
                    
                    removed = len(self.current_glossary_data) - len(filtered)
                    
                    if removed > 0:
                        if not self.create_glossary_backup(f"before_filter_remove_{removed}"):
                            return
                    
                    self.current_glossary_data[:] = filtered
                    
                    if save_current_glossary():
                        load_glossary_for_editing()
                        QMessageBox.information(filter_dialog, "Success", 
                            f"Filter applied!\n\nKept: {len(filtered)} entries\nRemoved: {removed} entries")
                        filter_dialog.accept()
            
            button_layout.addStretch()
            
            apply_btn = QPushButton("Apply Filter")
            apply_btn.setFixedWidth(120)
            apply_btn.clicked.connect(apply_filter)
            apply_btn.setStyleSheet("background-color: #198754; color: white; padding: 8px;")
            button_layout.addWidget(apply_btn)
            
            cancel_btn = QPushButton("Cancel")
            cancel_btn.setFixedWidth(120)
            cancel_btn.clicked.connect(filter_dialog.reject)
            cancel_btn.setStyleSheet("background-color: #6c757d; color: white; padding: 8px;")
            button_layout.addWidget(cancel_btn)
            
            button_layout.addStretch()
            
            # Show dialog with fade animation
            try:
                from dialog_animations import exec_dialog_with_fade
                exec_dialog_with_fade(filter_dialog, duration=250)
            except Exception:
                filter_dialog.exec()
    
        def export_selection():
           selected = self.glossary_tree.selectedItems()
           if not selected:
               QMessageBox.warning(self.dialog, "Warning", "No entries selected")
               return
           
           path, _ = QFileDialog.getSaveFileName(
               self.dialog,
               "Export Selected Entries",
               "",
               "JSON files (*.json);;CSV files (*.csv)"
           )
           
           if not path:
               return
           
           try:
               if self.current_glossary_format == 'list':
                   exported = []
                   for item in selected:
                       idx = int(item.text(0)) - 1
                       if 0 <= idx < len(self.current_glossary_data):
                           exported.append(self.current_glossary_data[idx])
                   
                   if path.endswith('.csv'):
                       # Export as CSV
                       import csv
                       with open(path, 'w', encoding='utf-8', newline='') as f:
                           writer = csv.writer(f)
                           for entry in exported:
                               if entry.get('type') == 'character':
                                   writer.writerow([entry.get('type', ''), entry.get('raw_name', ''), 
                                                  entry.get('translated_name', ''), entry.get('gender', '')])
                               else:
                                   writer.writerow([entry.get('type', ''), entry.get('raw_name', ''), 
                                                  entry.get('translated_name', ''), ''])
                   else:
                       # Export as JSON
                       with open(path, 'w', encoding='utf-8') as f:
                           json.dump(exported, f, ensure_ascii=False, indent=2)
               
               else:
                   exported = {}
                   entries_list = list(self.current_glossary_data.get('entries', {}).items())
                   for item in selected:
                       idx = int(item.text(0)) - 1
                       if 0 <= idx < len(entries_list):
                           key, value = entries_list[idx]
                           exported[key] = value
                   
                   with open(path, 'w', encoding='utf-8') as f:
                       json.dump(exported, f, ensure_ascii=False, indent=2)
               
               QMessageBox.information(self.dialog, "Success", f"Exported {len(selected)} entries to {os.path.basename(path)}")
               
           except Exception as e:
               QMessageBox.critical(self.dialog, "Error", f"Failed to export: {e}")
       
        def save_edited_glossary():
           if save_current_glossary():
               QMessageBox.information(self.dialog, "Success", "Glossary saved successfully")
               self.append_log(f"✅ Saved glossary to: {self.editor_file_entry.text()}")
       
        def save_as_glossary():
           if not self.current_glossary_data:
               QMessageBox.critical(self.dialog, "Error", "No glossary loaded")
               return
           
           path, _ = QFileDialog.getSaveFileName(
               self.dialog,
               "Save Glossary As",
               "",
               "JSON files (*.json);;CSV files (*.csv)"
           )
           
           if not path:
               return
           
           try:
               if path.endswith('.csv'):
                   # Save as CSV
                   import csv
                   with open(path, 'w', encoding='utf-8', newline='') as f:
                       writer = csv.writer(f)
                       if self.current_glossary_format == 'list':
                           for entry in self.current_glossary_data:
                               if entry.get('type') == 'character':
                                   writer.writerow([entry.get('type', ''), entry.get('raw_name', ''), 
                                                  entry.get('translated_name', ''), entry.get('gender', '')])
                               else:
                                   writer.writerow([entry.get('type', ''), entry.get('raw_name', ''), 
                                                  entry.get('translated_name', ''), ''])
               else:
                   # Save as JSON
                   with open(path, 'w', encoding='utf-8') as f:
                       json.dump(self.current_glossary_data, f, ensure_ascii=False, indent=2)
               
               self.editor_file_entry.setText(path)
               QMessageBox.information(self.dialog, "Success", f"Glossary saved to {os.path.basename(path)}")
               self.append_log(f"✅ Saved glossary as: {path}")
               
           except Exception as e:
               QMessageBox.critical(self.dialog, "Error", f"Failed to save: {e}")

        # Automatically load the currently auto-loaded glossary (CSV preferred) when opening the tab
        def auto_select_current_glossary():
            try:
                auto_path = getattr(self, 'auto_loaded_glossary_path', None)
                manual_path = getattr(self, 'manual_glossary_path', None)

                # Prefer the auto-loaded glossary if it exists and is a CSV
                if auto_path and os.path.exists(auto_path):
                    self.editor_file_entry.setText(auto_path)
                    load_glossary_for_editing()
                    return

                # Fallback to any currently loaded manual glossary
                if manual_path and os.path.exists(manual_path):
                    self.editor_file_entry.setText(manual_path)
                    load_glossary_for_editing()
            except Exception as e:
                # Fail silently but log for debugging
                try:
                    self.append_log(f"⚠️ Failed to auto-select glossary for editor: {e}")
                except Exception:
                    pass

        auto_select_current_glossary()
       
        # Quick toolbar above the entry list
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(0, 0, 0, 4)
        for text, handler, color in [
            ("Save", save_edited_glossary, "#15803d"),
            ("Save As...", save_as_glossary, "#0f766e"),
            ("Delete Selected", delete_selected_entries, "#991b1b")
        ]:
            btn = QPushButton(text)
            btn.setFixedWidth(115)
            btn.clicked.connect(handler)
            btn.setStyleSheet(f"background-color: {color}; color: white; padding: 6px; font-weight: bold;")
            toolbar_layout.addWidget(btn)
        toolbar_layout.addStretch()
        content_frame_layout.insertWidget(0, toolbar_widget)

        def show_tree_context_menu(pos):
            menu = QMenu(self.glossary_tree)
            menu.addAction("Save Changes", save_edited_glossary)
            menu.addAction("Save As...", save_as_glossary)
            menu.addSeparator()
            menu.addAction("Delete Selected", delete_selected_entries)
            menu.addAction("Reload", load_glossary_for_editing)
            menu.exec(self.glossary_tree.viewport().mapToGlobal(pos))

        try:
            self.glossary_tree.customContextMenuRequested.disconnect()
        except Exception:
            pass
        self.glossary_tree.customContextMenuRequested.connect(show_tree_context_menu)
       
        def find_in_tree():
            import re

            dialog = QDialog(self.dialog)
            dialog.setWindowTitle("Find / Replace")
            dialog_layout = QVBoxLayout(dialog)

            grid = QGridLayout()
            dialog_layout.addLayout(grid)

            find_edit = QLineEdit(getattr(self, "_last_find_text", ""))
            replace_edit = QLineEdit(getattr(self, "_last_replace_text", ""))

            grid.addWidget(QLabel("Find:"), 0, 0)
            grid.addWidget(find_edit, 0, 1)
            grid.addWidget(QLabel("Replace with:"), 1, 0)
            grid.addWidget(replace_edit, 1, 1)

            status_label = QLabel("")
            status_label.setStyleSheet("color: #9ca3af;")
            dialog_layout.addWidget(status_label)

            def find_next():
                text = find_edit.text()
                if not text:
                    return
                total = self.glossary_tree.topLevelItemCount()
                if total == 0:
                    return
                if text != getattr(self, "_last_find_text", ""):
                    self._last_find_pos = -1
                text_lower = text.lower()
                start = (getattr(self, "_last_find_pos", -1) + 1) % total
                for offset in range(total):
                    idx = (start + offset) % total
                    item = self.glossary_tree.topLevelItem(idx)
                    cols = [item.text(c) for c in range(item.columnCount())]
                    if any(text_lower in c.lower() for c in cols):
                        self.glossary_tree.setCurrentItem(item)
                        self.glossary_tree.scrollToItem(item)
                        self._last_find_text = text
                        self._last_find_pos = idx
                        status_label.setText(f"Found at row {idx + 1}")
                        return
                status_label.setText("No matches found.")

            def replace_in_item(item):
                """Replace occurrences in a single tree item and keep data in sync."""
                text = find_edit.text()
                repl = replace_edit.text()
                if not text or item is None:
                    return 0

                pattern = re.compile(re.escape(text), re.IGNORECASE)
                replacements = 0

                for col_idx in range(1, item.columnCount()):
                    before = item.text(col_idx)
                    after, count = pattern.subn(repl, before)
                    if count == 0:
                        continue

                    item.setText(col_idx, after)
                    col_key = self.glossary_column_fields[col_idx - 1] if self.glossary_column_fields else None

                    if self.current_glossary_format in ['list', 'token_csv']:
                        try:
                            row_idx = int(item.text(0)) - 1
                        except Exception:
                            row_idx = -1
                        if 0 <= row_idx < len(self.current_glossary_data) and col_key:
                            entry = self.current_glossary_data[row_idx]
                            if col_key == '_section':
                                entry['_section'] = after
                            elif after:
                                entry[col_key] = after
                            else:
                                entry.pop(col_key, None)

                    elif self.current_glossary_format == 'dict':
                        key = item.data(0, Qt.UserRole)
                        entries = self.current_glossary_data.get('entries', {})
                        if col_key == 'original':
                            value = entries.pop(key, None)
                            new_key = after if after else key
                            entries[new_key] = value
                            item.setData(0, Qt.UserRole, new_key)
                        elif col_key == 'translated' and key in entries:
                            entries[key] = after

                    replacements += count

                return replacements

            def replace_current():
                count = replace_in_item(self.glossary_tree.currentItem())
                status_label.setText(f"Replaced {count} occurrence(s) in current row" if count else "No matches in current row.")

            def replace_all():
                total_items = self.glossary_tree.topLevelItemCount()
                if total_items == 0 or not find_edit.text():
                    return
                total_repl = 0
                for idx in range(total_items):
                    item = self.glossary_tree.topLevelItem(idx)
                    total_repl += replace_in_item(item)
                self._last_find_text = find_edit.text()
                self._last_replace_text = replace_edit.text()
                status_label.setText(f"Replaced {total_repl} occurrence(s) across all entries.")

            buttons = QHBoxLayout()
            dialog_layout.addLayout(buttons)

            find_btn = QPushButton("Find Next")
            find_btn.clicked.connect(find_next)
            buttons.addWidget(find_btn)

            replace_btn = QPushButton("Replace")
            replace_btn.clicked.connect(replace_current)
            buttons.addWidget(replace_btn)

            replace_all_btn = QPushButton("Replace All")
            replace_all_btn.clicked.connect(replace_all)
            buttons.addWidget(replace_all_btn)

            buttons.addStretch()

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            buttons.addWidget(close_btn)

            find_edit.returnPressed.connect(find_next)
            dialog.exec()
       
        # Buttons
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(135)
        browse_btn.clicked.connect(browse_glossary)
        browse_btn.setStyleSheet("background-color: #495057; color: white; padding: 8px; font-weight: bold;")
        file_layout.addWidget(browse_btn)

        # Advanced editing toggle and button grid placed above the tree
        advanced_toggle_widget = QWidget()
        advanced_toggle_layout = QHBoxLayout(advanced_toggle_widget)
        advanced_toggle_layout.setContentsMargins(0, 4, 0, 4)
        advanced_toggle_layout.addStretch()
        advanced_checkbox = self._create_styled_checkbox("Advanced editing")
        advanced_checkbox.setChecked(False)
        advanced_toggle_layout.addWidget(advanced_checkbox)
        content_frame_layout.insertWidget(0, advanced_toggle_widget)

        advanced_tools_widget = QWidget()
        advanced_tools_layout = QGridLayout(advanced_tools_widget)
        advanced_tools_layout.setContentsMargins(0, 0, 0, 4)
        advanced_tools_layout.setHorizontalSpacing(10)
        advanced_tools_layout.setVerticalSpacing(8)

        def add_adv_btn(row, col, text, handler, color, width=150):
            btn = QPushButton(text)
            btn.setFixedWidth(width)
            btn.clicked.connect(handler)
            btn.setStyleSheet(f"background-color: {color}; color: white; padding: 8px; font-weight: bold;")
            advanced_tools_layout.addWidget(btn, row, col)

        add_adv_btn(0, 0, "Reload", load_glossary_for_editing, "#0891b2")
        add_adv_btn(0, 1, "Clean Empty Fields", clean_empty_fields, "#b45309")
        add_adv_btn(0, 2, "Remove Duplicates", remove_duplicates, "#b45309")
        add_adv_btn(0, 3, "Backup Settings", backup_settings_dialog, "#15803d")

        add_adv_btn(1, 0, "Trim Entries", smart_trim_dialog, "#1e40af")
        add_adv_btn(1, 1, "Filter Entries", filter_entries_dialog, "#1e40af")
        add_adv_btn(1, 2, "Convert Format", lambda: self.convert_glossary_format(load_glossary_for_editing), "#0891b2")
        add_adv_btn(1, 3, "Export Selection", export_selection, "#4b5563")
        add_adv_btn(1, 4, "About Format", duplicate_detection_settings, "#0891b2")

        advanced_tools_widget.setVisible(False)
        content_frame_layout.insertWidget(2, advanced_tools_widget)

        def toggle_advanced(state):
            advanced_tools_widget.setVisible(bool(state))
        advanced_checkbox.stateChanged.connect(toggle_advanced)

        # Keyboard shortcuts
        QShortcut(QKeySequence.Save, self.dialog, activated=save_edited_glossary)
        QShortcut(QKeySequence.Delete, self.dialog, activated=delete_selected_entries)
        QShortcut(QKeySequence.Find, self.dialog, activated=find_in_tree)

    def _on_tree_double_click(self, item, column_idx):
       """Handle double-click on treeview item for inline editing"""
       if not item or column_idx <= 0:
           return
       
       if not self.glossary_column_fields or column_idx - 1 >= len(self.glossary_column_fields):
           return
       
       col_key = self.glossary_column_fields[column_idx - 1]
       current_value = item.text(column_idx)
       
       edit_dialog = QDialog(self.dialog)
       edit_dialog.setWindowTitle(f"Edit {col_key.replace('_', ' ').title()}")
       edit_dialog.setMinimumWidth(400)
       edit_dialog.setMinimumHeight(150)
       
       dialog_layout = QVBoxLayout(edit_dialog)
       dialog_layout.setContentsMargins(20, 20, 20, 20)
       
       label = QLabel(f"Edit {col_key.replace('_', ' ').title()}:")
       dialog_layout.addWidget(label)
       
       entry = QLineEdit(current_value)
       dialog_layout.addWidget(entry)
       dialog_layout.addSpacing(5)
       entry.setFocus()
       entry.selectAll()
       
       def save_edit():
           new_value = entry.text()
           item.setText(column_idx, new_value)
           
           row_idx = int(item.text(0)) - 1
           
           if self.current_glossary_format in ['list', 'token_csv']:
               if 0 <= row_idx < len(self.current_glossary_data):
                   data_entry = self.current_glossary_data[row_idx]
                   if new_value:
                       data_entry[col_key] = new_value
                   else:
                       data_entry.pop(col_key, None)
           
           elif self.current_glossary_format == 'dict':
               key = item.data(0, Qt.UserRole)
               entries = self.current_glossary_data.get('entries', {})
               if key in entries:
                   if col_key == 'original':
                       value = entries.pop(key)
                       new_key = new_value or key
                       entries[new_key] = value
                       item.setData(0, Qt.UserRole, new_key)
                   elif col_key == 'translated':
                       entries[key] = new_value
           
           edit_dialog.accept()
       
       dialog_layout.addSpacing(10)
       button_layout = QHBoxLayout()
       dialog_layout.addLayout(button_layout)
       
       save_btn = QPushButton("Save")
       save_btn.setFixedWidth(80)
       save_btn.clicked.connect(save_edit)
       save_btn.setStyleSheet("background-color: #198754; color: white; padding: 8px;")
       button_layout.addWidget(save_btn)
       
       cancel_btn = QPushButton("Cancel")
       cancel_btn.setFixedWidth(80)
       cancel_btn.clicked.connect(edit_dialog.reject)
       cancel_btn.setStyleSheet("background-color: #6c757d; color: white; padding: 8px;")
       button_layout.addWidget(cancel_btn)
       
       entry.returnPressed.connect(save_edit)
       
       try:
           from dialog_animations import exec_dialog_with_fade
           exec_dialog_with_fade(edit_dialog, duration=250)
       except Exception:
           edit_dialog.exec()

    def convert_glossary_format(self, reload_callback):
        """Export glossary to CSV format"""
        if not self.current_glossary_data:
            QMessageBox.critical(self.dialog, "Error", "No glossary loaded")
            return
        
        # Create backup before conversion
        if not self.create_glossary_backup("before_export"):
            return
        
        # Get current file path
        current_path = self.editor_file_entry.text()
        default_csv_path = current_path.replace('.json', '.csv')
        
        # Ask user for CSV save location
        csv_path, _ = QFileDialog.getSaveFileName(
            self.dialog,
            "Export Glossary to CSV",
            default_csv_path,
            "CSV files (*.csv);;All files (*.*)"
        )
        
        if not csv_path:
            return
        
        try:
            import csv
            
            # Get custom types for gender info
            custom_types = self.config.get('custom_entry_types', {
                'character': {'enabled': True, 'has_gender': True},
                'term': {'enabled': True, 'has_gender': False}
            })
            
            # Get custom fields
            custom_fields = self.config.get('custom_glossary_fields', [])
            
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                
                # Build header row
                header = ['type', 'raw_name', 'translated_name', 'gender']
                if custom_fields:
                    header.extend(custom_fields)
                
                # Write header row
                writer.writerow(header)
                
                # Process based on format
                if isinstance(self.current_glossary_data, list) and self.current_glossary_data:
                    if 'type' in self.current_glossary_data[0]:
                        # New format - direct export
                        for entry in self.current_glossary_data:
                            entry_type = entry.get('type', 'term')
                            type_config = custom_types.get(entry_type, {})
                            
                            row = [
                                entry_type,
                                entry.get('raw_name', ''),
                                entry.get('translated_name', '')
                            ]
                            
                            # Add gender
                            if type_config.get('has_gender', False):
                                row.append(entry.get('gender', ''))
                            else:
                                row.append('')
                            
                            # Add custom field values
                            for field in custom_fields:
                                row.append(entry.get(field, ''))
                            
                            writer.writerow(row)
                    else:
                        # Old format - convert then export
                        for entry in self.current_glossary_data:
                            # Determine type
                            is_location = False
                            if 'locations' in entry and entry['locations']:
                                is_location = True
                            elif 'title' in entry and any(term in str(entry.get('title', '')).lower() 
                                                         for term in ['location', 'place', 'city', 'region']):
                                is_location = True
                            
                            entry_type = 'term' if is_location else 'character'
                            type_config = custom_types.get(entry_type, {})
                            
                            row = [
                                entry_type,
                                entry.get('original_name', entry.get('original', '')),
                                entry.get('name', entry.get('translated', ''))
                            ]
                            
                            # Add gender
                            if type_config.get('has_gender', False):
                                row.append(entry.get('gender', 'Unknown'))
                            else:
                                row.append('')
                            
                            # Add empty custom fields
                            for field in custom_fields:
                                row.append('')
                            
                            writer.writerow(row)
            
            QMessageBox.information(self.dialog, "Success", f"Glossary exported to CSV:\n{csv_path}")
            self.append_log(f"✅ Exported glossary to: {csv_path}")
            
        except Exception as e:
            QMessageBox.critical(self.dialog, "Export Error", f"Failed to export CSV: {e}")
            self.append_log(f"❌ CSV export failed: {e}")
