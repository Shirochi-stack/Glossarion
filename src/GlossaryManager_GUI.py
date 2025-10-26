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
                                QComboBox)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QColor, QIcon

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
        
        # Use screen ratios instead of fixed pixels
        self._screen = QApplication.primaryScreen().geometry()
        min_width = int(self._screen.width() * 0.42)   # 50% of screen width
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
                    ('compress_glossary_checkbox', 'compress_glossary_prompt_var'),
                    ('include_gender_context_checkbox', 'include_gender_context_var'),
                    ('glossary_history_rolling_checkbox', 'glossary_history_rolling_var'),
                    ('strip_honorifics_checkbox', 'strip_honorifics_var'),
                    ('disable_honorifics_checkbox', 'disable_honorifics_var'),
                    ('use_legacy_csv_checkbox', 'use_legacy_csv_var'),
                ]
                for checkbox_name, var_name in checkbox_to_var_mapping:
                    if hasattr(self, checkbox_name):
                        setattr(self, var_name, getattr(self, checkbox_name).isChecked())
                
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
        
        def remove_custom_field():
            current_row = self.custom_fields_listbox.currentRow()
            if current_row >= 0:
                item = self.custom_fields_listbox.item(current_row)
                field = item.text()
                self.custom_glossary_fields.remove(field)
                self.custom_fields_listbox.takeItem(current_row)
        
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
        
        # Add icon before dropdown
        algo_icon_label = QLabel()
        try:
            # Try to load Halgakos.ico as icon
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico')
            if os.path.exists(icon_path):
                from PySide6.QtGui import QPixmap
                pixmap = QPixmap(icon_path)
                if not pixmap.isNull():
                    # Scale to 24x24 for nice display
                    scaled_pixmap = pixmap.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    algo_icon_label.setPixmap(scaled_pixmap)
                else:
                    # Fallback to emoji if icon doesn't load
                    algo_icon_label.setText("🎯")
                    algo_icon_label.setStyleSheet("font-size: 18pt;")
            else:
                # Icon file not found, use emoji
                algo_icon_label.setText("🎯")
                algo_icon_label.setStyleSheet("font-size: 18pt;")
        except Exception as e:
            # Any error, fallback to emoji
            algo_icon_label.setText("🎯")
            algo_icon_label.setStyleSheet("font-size: 18pt;")
        
        algo_layout.addWidget(algo_icon_label)
        
        self.duplicate_algo_combo = QComboBox()
        self.duplicate_algo_combo.addItems([
            "Auto (Recommended) - Uses all algorithms",
            "Strict - High precision, minimal merging",
            "Balanced - Token + Partial matching",
            "Aggressive - Maximum duplicate detection",
            "Basic Only - Simple Levenshtein distance"
        ])
        
        # Load saved setting or default to Auto
        saved_algo = self.config.get('glossary_duplicate_algorithm', 'auto')
        algo_index_map = {
            'auto': 0,
            'strict': 1,
            'balanced': 2,
            'aggressive': 3,
            'basic': 4
        }
        self.duplicate_algo_combo.setCurrentIndex(algo_index_map.get(saved_algo, 0))
        
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
                "<b>Auto (Recommended)</b>: Uses all available algorithms (RapidFuzz, Jaro-Winkler, Token matching) and takes the best score. Best for most cases.<br><br>"
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
        
        algo_desc = QLabel("🎯 Auto mode uses multiple algorithms for best accuracy")
        algo_desc.setStyleSheet("color: gray; font-size: 9pt; margin-bottom: 15px;")
        duplicate_frame_layout.addWidget(algo_desc)
        
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
        
        # Prompt section
        prompt_frame = QGroupBox("Extraction Prompt")
        prompt_frame_layout = QVBoxLayout(prompt_frame)
        manual_layout.addWidget(prompt_frame)
        
        label1 = QLabel("Use {fields} for field list and {chapter_text} for content placeholder")
        # label1.setStyleSheet("color: white; font-size: 9pt;")
        prompt_frame_layout.addWidget(label1)
        
        label2 = QLabel("The {fields} placeholder will be replaced with the format specification")
        # label2.setStyleSheet("color: gray; font-size: 9pt;")
        prompt_frame_layout.addWidget(label2)
        
        self.manual_prompt_text = QTextEdit()
        # Use screen ratio: ~25% of screen height
        prompt_height = int(self._screen.height() * 0.25)
        self.manual_prompt_text.setMinimumHeight(prompt_height)
        self.manual_prompt_text.setLineWrapMode(QTextEdit.WidgetWidth)
        prompt_frame_layout.addWidget(self.manual_prompt_text)
        
        # Always reload prompt from config to ensure fresh state
        default_manual_prompt = """Extract character names and important terms from the following text.

Output format:
{fields}

Rules:
- Output ONLY CSV lines in the exact format shown above
- No headers, no extra text, no JSON
- One entry per line
- Leave gender empty for terms (just end with comma)
    """
        self.manual_glossary_prompt = self.config.get('manual_glossary_prompt', default_manual_prompt)
        
        self.manual_prompt_text.setPlainText(self.manual_glossary_prompt)
        
        prompt_controls_widget = QWidget()
        prompt_controls_layout = QHBoxLayout(prompt_controls_widget)
        prompt_controls_layout.setContentsMargins(0, 10, 0, 0)
        manual_layout.addWidget(prompt_controls_widget)
        
        def reset_manual_prompt():
            reply = QMessageBox.question(parent, "Reset Prompt", "Reset manual glossary prompt to default?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                default_prompt = """Extract character names and important terms from the following text.

    Output format:
    {fields}

    Rules:
    - Output ONLY CSV lines in the exact format shown above
    - No headers, no extra text, no JSON
    - One entry per line
    - Leave gender empty for terms (just end with comma)
    """
                self.manual_prompt_text.setPlainText(default_prompt)
        
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
        
        # Temperature and Context Limit
        self.manual_temp_entry = QLineEdit(str(self.config.get('manual_glossary_temperature', 0.1)))
        self.manual_temp_entry.setFixedWidth(80)
        self.manual_context_entry = QLineEdit(str(self.config.get('manual_context_limit', 2)))
        self.manual_context_entry.setFixedWidth(80)
        settings_grid.addWidget(_m_pair("Temperature:", self.manual_temp_entry), 0, 0, 1, 2)
        settings_grid.addWidget(_m_pair("Context Limit:", self.manual_context_entry), 0, 2, 1, 2)
        
        # Rolling window checkbox + description
        if not hasattr(self, 'glossary_history_rolling_checkbox'):
            self.glossary_history_rolling_checkbox = self._create_styled_checkbox("Keep recent context instead of reset")
        # Always reload from config
        self.glossary_history_rolling_checkbox.setChecked(self.config.get('glossary_history_rolling', False))
        settings_grid.addWidget(self.glossary_history_rolling_checkbox, 1, 0, 1, 4)
        
        rolling_label = QLabel("When context limit is reached, keep recent chapters instead of clearing all history")
        # rolling_label.setStyleSheet("color: gray; font-size: 10pt; margin-left: 20px;")
        settings_grid.addWidget(rolling_label, 2, 0, 1, 4)

    def update_glossary_prompts(self):
        """Update glossary prompts from text widgets if they exist"""
        try:
            debug_enabled = getattr(self, 'config', {}).get('show_debug_buttons', False)
            
            if hasattr(self, 'manual_prompt_text'):
                self.manual_glossary_prompt = self.manual_prompt_text.toPlainText().strip()
                if debug_enabled:
                    print(f"🔍 [UPDATE] manual_glossary_prompt: {len(self.manual_glossary_prompt)} chars")
            
            if hasattr(self, 'auto_prompt_text'):
                self.auto_glossary_prompt = self.auto_prompt_text.toPlainText().strip()
                if debug_enabled:
                    print(f"🔍 [UPDATE] auto_glossary_prompt: {len(self.auto_glossary_prompt)} chars")
            
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
        default_append_prompt = "- Follow this reference glossary for consistent translation (Do not output any raw entries):\n"
        self.append_glossary_prompt = self.config.get('append_glossary_prompt', default_append_prompt)
        
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
        
        if not hasattr(self, 'glossary_batch_size_entry'):
            self.glossary_batch_size_entry = QLineEdit()
            self.glossary_batch_size_entry.setFixedWidth(80)
        self.glossary_batch_size_entry.setText(str(self.config.get('glossary_batch_size', 10)))
        
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
        extraction_grid.addWidget(_pair("Translation batch:", self.glossary_batch_size_entry), 1, 2, 1, 2)
        
        # Row 3 - Max text size and target language
        extraction_grid.addWidget(_pair("Max text size:", self.glossary_max_text_size_entry), 2, 0, 1, 2)
        
        # Target language dropdown
        if not hasattr(self, 'glossary_target_language_combo'):
            self.glossary_target_language_combo = QComboBox()
            self.glossary_target_language_combo.setMaximumWidth(100)
            languages = [
                "English", "Spanish", "French", "German", "Italian", "Portuguese",
                "Russian", "Arabic", "Hindi", "Chinese (Simplified)",
                "Chinese (Traditional)", "Japanese", "Korean"
            ]
            self.glossary_target_language_combo.addItems(languages)
            
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
        
        saved_language = self.config.get('glossary_target_language', 'English')
        index = self.glossary_target_language_combo.findText(saved_language)
        if index >= 0:
            self.glossary_target_language_combo.setCurrentIndex(index)
        
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
            "• Translation batch: Terms per API call (larger = faster but may reduce quality)",
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
        default_unified_prompt = """You are a glossary extraction assistant for Korean / Japanese / Chinese novels.

Return ONLY CSV format with exactly 4 columns: type,raw_name,translated_name,gender.
For character entries, determine gender from context, leave empty if context is insufficient.
For non-character entries, leave gender empty.
Only include terms that actually appear in the text.
Do not use quotes around values unless they contain commas.

Critical Requirement: The translated name column should be in {language}.

For example:
character,김상현,Kim Sang-hyu,male
character,갈편제,Gale Hardest  
character,디히릿 아데,Dihirit Ade,female

Focus on identifying:
1. Character names with their honorifics
2. Important titles and ranks
3. Frequently mentioned terms (min frequency: {min_frequency})

Extract up to {max_names} character names and {max_titles} titles.
Prioritize names that appear with honorifics or in important contexts."""
        
        # Load from config or use default
        self.auto_glossary_prompt = self.config.get('auto_glossary_prompt', default_unified_prompt)
        self.auto_prompt_text.setPlainText(self.auto_glossary_prompt)
        
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
        content_frame_layout.addWidget(self.glossary_tree)

        self.glossary_tree.itemDoubleClicked.connect(lambda item, col: self._on_tree_double_click(item, col))

        self.current_glossary_data = None
        self.current_glossary_format = None

        # Editor functions
        def load_glossary_for_editing():
           path = self.editor_file_entry.text()
           if not path or not os.path.exists(path):
               QMessageBox.critical(parent, "Error", "Please select a valid glossary file")
               return
           
           try:
               # Try CSV first
               if path.endswith('.csv'):
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
                               entries.append(entry)
                   self.current_glossary_data = entries
                   self.current_glossary_format = 'list'
               else:
                   # JSON format
                   with open(path, 'r', encoding='utf-8') as f:
                       data = json.load(f)
                   
                   entries = []
                   all_fields = set()
                   
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
               if self.current_glossary_format == 'list' and entries and 'type' in entries[0]:
                   # New simple format
                   column_fields = ['type', 'raw_name', 'translated_name', 'gender']
                   
                   # Check for any custom fields
                   for entry in entries:
                       for field in entry.keys():
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
               self.glossary_tree.setColumnCount(len(column_fields) + 1)  # +1 for index column
               
               headers = ['#'] + [field.replace('_', ' ').title() for field in column_fields]
               self.glossary_tree.setHeaderLabels(headers)
               
               self.glossary_tree.setColumnWidth(0, 40)
               
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
                   self.glossary_tree.addTopLevelItem(item)
               
               # Update stats
               stats = []
               stats.append(f"Total entries: {len(entries)}")
               
               if self.current_glossary_format == 'list' and entries and 'type' in entries[0]:
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
                   # Save as CSV
                   import csv
                   with open(path, 'w', encoding='utf-8', newline='') as f:
                       writer = csv.writer(f)
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
               return True
           except Exception as e:
               QMessageBox.critical(parent, "Error", f"Failed to save: {e}")
               return False
       
        def clean_empty_fields():
            if not self.current_glossary_data:
                QMessageBox.critical(parent, "Error", "No glossary loaded")
                return
            
            if self.current_glossary_format == 'list':
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

                if self.current_glossary_format == 'list':
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
            
            if self.current_glossary_format == 'list':
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
            
            entry_count = len(self.current_glossary_data) if self.current_glossary_format == 'list' else len(self.current_glossary_data.get('entries', {}))
            stats_layout.addWidget(QLabel(f"Total entries: {entry_count}"))
            
            # For new format, show type breakdown
            if self.current_glossary_format == 'list' and self.current_glossary_data and 'type' in self.current_glossary_data[0]:
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
                    
                    if self.current_glossary_format == 'list':
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
            entry_count = len(self.current_glossary_data) if self.current_glossary_format == 'list' else len(self.current_glossary_data.get('entries', {}))
            
            stats_group = QGroupBox("Current Status")
            stats_layout = QVBoxLayout(stats_group)
            stats_label = QLabel(f"Total entries: {entry_count}")
            stats_label.setStyleSheet("font-size: 10pt;")
            stats_layout.addWidget(stats_label)
            content_layout.addWidget(stats_group)
            content_layout.addSpacing(15)
            
            # Check if new format
            is_new_format = (self.current_glossary_format == 'list' and 
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
                
                if self.current_glossary_format == 'list':
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
                
                if self.current_glossary_format == 'list':
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
       
        # Buttons
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(135)
        browse_btn.clicked.connect(browse_glossary)
        browse_btn.setStyleSheet("background-color: #495057; color: white; padding: 8px; font-weight: bold;")
        file_layout.addWidget(browse_btn)
        
        # Editor control buttons
        editor_layout.addSpacing(10)
        
        # Row 1
        row1_layout = QHBoxLayout()
        editor_layout.addLayout(row1_layout)
        editor_layout.addSpacing(2)
       
        buttons_row1 = [
           ("Reload", load_glossary_for_editing, "#0891b2"),
           ("Delete Selected", delete_selected_entries, "#991b1b"),
           ("Clean Empty Fields", clean_empty_fields, "#b45309"),
           ("Remove Duplicates", remove_duplicates, "#b45309"),
           ("Backup Settings", backup_settings_dialog, "#15803d")
        ]
       
        for text, cmd, color in buttons_row1:
            btn = QPushButton(text)
            btn.setFixedWidth(135)
            btn.clicked.connect(cmd)
            btn.setStyleSheet(f"background-color: {color}; color: white; padding: 8px; font-weight: bold;")
            row1_layout.addWidget(btn)
       
        # Row 2
        row2_layout = QHBoxLayout()
        editor_layout.addLayout(row2_layout)
        editor_layout.addSpacing(2)

        buttons_row2 = [
           ("Trim Entries", smart_trim_dialog, "#1e40af"),
           ("Filter Entries", filter_entries_dialog, "#1e40af"),
           ("Convert Format", lambda: self.convert_glossary_format(load_glossary_for_editing), "#0891b2"),
           ("Export Selection", export_selection, "#4b5563"),
           ("About Format", duplicate_detection_settings, "#0891b2")
        ]

        for text, cmd, color in buttons_row2:
            btn = QPushButton(text)
            btn.setFixedWidth(135)
            btn.clicked.connect(cmd)
            btn.setStyleSheet(f"background-color: {color}; color: white; padding: 8px; font-weight: bold;")
            row2_layout.addWidget(btn)

        # Row 3
        row3_layout = QHBoxLayout()
        editor_layout.addLayout(row3_layout)
        editor_layout.addSpacing(2)

        save_btn = QPushButton("Save Changes")
        save_btn.setFixedWidth(165)
        save_btn.clicked.connect(save_edited_glossary)
        save_btn.setStyleSheet("background-color: #15803d; color: white; padding: 8px; font-weight: bold;")
        row3_layout.addWidget(save_btn)
        
        save_as_btn = QPushButton("Save As...")
        save_as_btn.setFixedWidth(165)
        save_as_btn.clicked.connect(save_as_glossary)
        save_as_btn.setStyleSheet("background-color: #15803d; color: white; padding: 8px; font-weight: bold; border: 1px solid #15803d;")
        row3_layout.addWidget(save_as_btn)

    def _on_tree_double_click(self, item, column_idx):
       """Handle double-click on treeview item for inline editing"""
       if not item or column_idx < 0:
           return
       
       # Get column count
       column_count = self.glossary_tree.columnCount()
       if column_idx >= column_count:
           return
       
       # Get column name from header
       col_name = self.glossary_tree.headerItem().text(column_idx)
       current_value = item.text(column_idx)
       
       edit_dialog = QDialog(self.dialog)
       edit_dialog.setWindowTitle(f"Edit {col_name.replace('_', ' ').title()}")
       edit_dialog.setMinimumWidth(400)
       edit_dialog.setMinimumHeight(150)
       
       dialog_layout = QVBoxLayout(edit_dialog)
       dialog_layout.setContentsMargins(20, 20, 20, 20)
       
       label = QLabel(f"Edit {col_name.replace('_', ' ').title()}:")
       dialog_layout.addWidget(label)
       
       # Simple entry for new format fields
       entry = QLineEdit(current_value)
       dialog_layout.addWidget(entry)
       dialog_layout.addSpacing(5)
       entry.setFocus()
       entry.selectAll()
       
       def save_edit():
           new_value = entry.text()
           
           # Update tree item
           item.setText(column_idx, new_value)
           
           # Update data
           row_idx = int(item.text(0)) - 1
           
           if self.current_glossary_format == 'list':
               if 0 <= row_idx < len(self.current_glossary_data):
                   data_entry = self.current_glossary_data[row_idx]
                   
                   if new_value:
                       data_entry[col_name] = new_value
                   else:
                       data_entry.pop(col_name, None)
           
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
       
       # Connect Enter/Escape shortcuts
       entry.returnPressed.connect(save_edit)
       
       # Show edit dialog with fade animation
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
