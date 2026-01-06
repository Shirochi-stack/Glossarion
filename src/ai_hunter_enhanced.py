# ai_hunter_enhanced.py
# Combined AI Hunter configuration GUI and detection logic

import json
import os
import re
import unicodedata
from difflib import SequenceMatcher
from collections import Counter

# PySide6 imports - optional for non-GUI usage
try:
    from PySide6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QCheckBox, QTabWidget, QWidget, QScrollArea, QFrame,
        QSlider, QSpinBox, QDoubleSpinBox, QRadioButton, QComboBox,
        QGroupBox, QMessageBox
    )
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon
    HAS_GUI = True
except ImportError:
    HAS_GUI = False
    # Dummy classes for non-GUI usage
    QDialog = object
    QWidget = object

class AIHunterConfigGUI:
    """GUI for configuring AI Hunter detection parameters"""
    def __init__(self, parent, config_dict, callback=None):
        """
        Initialize with reference to main config dictionary
        
        Args:
            parent: Parent window
            config_dict: Reference to main translator config dictionary
            callback: Function to call after saving
        """
        self.parent = parent
        self.config = config_dict  # Reference to main config
        self.callback = callback
        self.window = None
        
        # Default AI Hunter settings structure
        self.default_ai_hunter = {
            'enabled': True,
            'ai_hunter_max_workers': 1,
            'retry_attempts': 6,
            'disable_temperature_change': False,
            'sample_size': 3000,
            'thresholds': {
                'exact': 90,
                'text': 35,
                'semantic': 85,
                'structural': 85,
                'character': 90,
                'pattern': 80
            },
            'weights': {
                'exact': 1.5,
                'text': 1.2,
                'semantic': 1.0,
                'structural': 1.0,
                'character': 0.8,
                'pattern': 0.8
            },
            'detection_mode': 'weighted_average',
            'multi_method_requirements': {
                'methods_required': 3,
                'min_methods': ['semantic', 'structural']
            },
            'preprocessing': {
                'remove_html_spacing': True,
                'normalize_unicode': True,
                'ignore_case': True,
                'remove_extra_whitespace': True
            },
            'edge_filters': {
                'min_text_length': 500,
                'max_length_ratio': 1.3,
                'min_length_ratio': 0.7
            },
            'language_detection': {
                'enabled': False,
                'target_language': 'english',
                'threshold_characters': 500,
                'languages': {
                    'english': ['en'],
                    'japanese': ['ja', 'jp'],
                    'korean': ['ko', 'kr'],
                    'chinese': ['zh', 'zh-cn', 'zh-tw'],
                    'spanish': ['es'],
                    'french': ['fr'],
                    'german': ['de'],
                    'russian': ['ru'],
                    'arabic': ['ar'],
                    'hindi': ['hi'],
                    'portuguese': ['pt'],
                    'italian': ['it'],
                    'dutch': ['nl'],
                    'thai': ['th'],
                    'vietnamese': ['vi'],
                    'turkish': ['tr'],
                    'polish': ['pl'],
                    'swedish': ['sv'],
                    'danish': ['da'],
                    'norwegian': ['no'],
                    'finnish': ['fi']
                }
            }
        }
        
        # Initialize AI Hunter config in main config if not present
        if 'ai_hunter_config' not in self.config:
            self.config['ai_hunter_config'] = self.default_ai_hunter.copy()
        else:
            # Merge with defaults to ensure all keys exist
            self.config['ai_hunter_config'] = self._merge_configs(
                self.default_ai_hunter, 
                self.config['ai_hunter_config']
            )
    
    def _merge_configs(self, default, existing):
        """Recursively merge existing config with defaults"""
        result = default.copy()
        for key, value in existing.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def get_ai_config(self):
        """Get AI Hunter configuration from main config"""
        return self.config.get('ai_hunter_config', self.default_ai_hunter)
    
    def _disable_mousewheel(self, widget):
        """Disable mousewheel scrolling on a widget (PySide6)"""
        widget.wheelEvent = lambda event: None
    
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
    
    def show_ai_hunter_config(self):
        """Display the AI Hunter configuration window (PySide6)"""
        try:
            if self.window and not self.window.isHidden():
                self.window.raise_()
                self.window.activateWindow()
                return
        except RuntimeError:
            # Window was deleted
            self.window = None
        
        # Create dialog
        dialog = QDialog(None)
        dialog.setWindowTitle("AI Hunter Configuration")
        
        # Use screen ratios for sizing (more reliable across different displays)
        from PySide6.QtWidgets import QApplication
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.47)  # 47% of screen width
        height = int(screen.height() * 0.69)  # 69% of screen height
        dialog.resize(width, height)
        
        # Set icon
        try:
            dialog.setWindowIcon(QIcon("halgakos.ico"))
        except Exception:
            pass
        
        self.window = dialog
        
        # Apply global stylesheet for checkboxes, radio buttons, and tabs
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
            QTabWidget::pane {
                border: 1px solid #5a9fd4;
                background-color: #2d2d2d;
                border-radius: 3px;
            }
            QTabBar::tab {
                background-color: #1a1a1a;
                color: #aaaaaa;
                padding: 8px 16px;
                margin-right: 2px;
                border: 1px solid #3a3a3a;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 100px;
            }
            QTabBar::tab:selected {
                background-color: #5a9fd4;
                color: white;
                font-weight: bold;
                border: 1px solid #5a9fd4;
                border-bottom: none;
            }
            QTabBar::tab:hover {
                background-color: #3a3a3a;
                color: white;
            }
            QTabBar::tab:selected:hover {
                background-color: #7bb3e0;
            }
        """
        
        main_layout = QVBoxLayout(dialog)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create tab widget
        tabs = QTabWidget()
        tabs.setStyleSheet(checkbox_radio_style)
        main_layout.addWidget(tabs)
        
        # Tab 1: Detection Thresholds
        self.create_thresholds_tab(tabs)
        
        # Tab 2: Detection Mode
        self.create_mode_tab(tabs)
        
        # Tab 3: Preprocessing
        self.create_preprocessing_tab(tabs)
        
        # Tab 4: Advanced Settings
        self.create_advanced_tab(tabs)
        
        # Buttons at the bottom
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(5, 10, 5, 10)
        
        reset_btn = QPushButton("⚠️ Reset to Defaults")
        reset_btn.clicked.connect(self.reset_defaults)
        reset_btn.setMinimumHeight(35)
        reset_btn.setStyleSheet(
            "QPushButton { "
            "  background-color: #ffc107; "
            "  color: black; "
            "  padding: 8px 20px; "
            "  font-size: 11pt; "
            "  font-weight: bold; "
            "  border-radius: 4px; "
            "} "
            "QPushButton:hover { background-color: #e0a800; }"
        )
        button_layout.addWidget(reset_btn)
        button_layout.addStretch()
        
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self.apply_ai_hunter_settings)
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
        
        main_layout.addLayout(button_layout)
        
        dialog.show()
    
    def create_thresholds_tab(self, tabs):
        """Create the thresholds configuration tab (PySide6)"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        frame = QWidget()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Detection Method Thresholds")
        title.setStyleSheet("font-size: 12pt; font-weight: bold;")
        layout.addWidget(title)
        
        desc = QLabel("Higher values = fewer false positives (more strict)\n"
                     "Lower values = more false positives (more sensitive)")
        desc.setStyleSheet("color: gray; font-size: 10pt;")
        layout.addWidget(desc)
        layout.addSpacing(10)
        
        # Threshold controls
        self.threshold_vars = {}
        self.threshold_labels = {}
        
        descriptions = {
            'exact': 'Exact Text Match - Direct character-by-character comparison',
            'text': 'Smart Text Similarity - Intelligent text comparison with sampling',
            'semantic': 'Semantic Analysis - Character names, dialogue patterns, numbers',
            'structural': 'Structural Patterns - Paragraph structure, dialogue distribution',
            'character': 'Character Overlap - Common character names between chapters',
            'pattern': 'Pattern Analysis - Narrative flow and structure patterns'
        }
        
        ai_config = self.get_ai_config()
        
        for method, desc in descriptions.items():
            method_frame = QWidget()
            method_layout = QVBoxLayout(method_frame)
            method_layout.setContentsMargins(0, 10, 0, 10)
            
            # Method name and description
            label_widget = QWidget()
            label_layout = QHBoxLayout(label_widget)
            label_layout.setContentsMargins(0, 0, 0, 0)
            
            method_label = QLabel(f"{method.title()}:")
            method_label.setStyleSheet("font-weight: bold; font-size: 10pt;")
            label_layout.addWidget(method_label)
            
            desc_label = QLabel(f" {desc}")
            desc_label.setStyleSheet("color: gray; font-size: 9pt;")
            label_layout.addWidget(desc_label)
            label_layout.addStretch()
            
            method_layout.addWidget(label_widget)
            
            # Slider and value
            slider_widget = QWidget()
            slider_layout = QHBoxLayout(slider_widget)
            slider_layout.setContentsMargins(20, 5, 0, 0)
            
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(10)
            slider.setMaximum(100)
            slider.setValue(ai_config['thresholds'][method])
            slider.setFixedWidth(400)
            self._disable_mousewheel(slider)
            self.threshold_vars[method] = slider
            slider_layout.addWidget(slider)
            
            value_label = QLabel(f"{slider.value()}%")
            value_label.setFixedWidth(50)
            self.threshold_labels[method] = value_label
            slider_layout.addWidget(value_label)
            
            # Connect slider to label update
            slider.valueChanged.connect(
                lambda val, lbl=value_label: lbl.setText(f"{val}%")
            )
            
            slider_layout.addStretch()
            method_layout.addWidget(slider_widget)
            
            layout.addWidget(method_frame)
        
        # Weight configuration
        layout.addSpacing(20)
        weight_title = QLabel("Method Weights (for weighted average mode)")
        weight_title.setStyleSheet("font-size: 11pt; font-weight: bold;")
        layout.addWidget(weight_title)
        layout.addSpacing(10)
        
        self.weight_vars = {}
        
        for method in descriptions.keys():
            w_widget = QWidget()
            w_layout = QHBoxLayout(w_widget)
            w_layout.setContentsMargins(0, 5, 0, 5)
            
            w_label = QLabel(f"{method.title()} weight:")
            w_label.setFixedWidth(150)
            w_layout.addWidget(w_label)
            
            w_spinbox = QDoubleSpinBox()
            w_spinbox.setMinimum(0.1)
            w_spinbox.setMaximum(2.0)
            w_spinbox.setSingleStep(0.1)
            w_spinbox.setValue(ai_config['weights'][method])
            w_spinbox.setFixedWidth(80)
            self._disable_mousewheel(w_spinbox)
            self.weight_vars[method] = w_spinbox
            w_layout.addWidget(w_spinbox)
            
            w_layout.addStretch()
            layout.addWidget(w_widget)
        
        layout.addStretch()
        scroll.setWidget(frame)
        tabs.addTab(scroll, "Detection Thresholds")
    
    def create_mode_tab(self, tabs):
        """Create the detection mode configuration tab (PySide6)"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        frame = QWidget()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        title = QLabel("Detection Mode Configuration")
        title.setStyleSheet("font-size: 12pt; font-weight: bold;")
        layout.addWidget(title)
        layout.addSpacing(10)
        
        # Detection mode selection
        mode_box = QGroupBox("Detection Mode")
        mode_layout = QVBoxLayout(mode_box)
        mode_layout.setSpacing(10)
        
        ai_config = self.get_ai_config()
        self.mode_buttons = {}
        
        modes = [
            ('single_method', 'Single Method', 
             'Flag as duplicate if ANY method exceeds its threshold\n(Most sensitive, most false positives)'),
            ('multi_method', 'Multi-Method Agreement', 
             'Require multiple methods to agree before flagging\n(Balanced approach)'),
            ('weighted_average', 'Weighted Average', 
             'Calculate weighted average of all methods\n(Most nuanced, least false positives)')
        ]
        
        for value, text, desc in modes:
            rb_widget = QWidget()
            rb_layout = QVBoxLayout(rb_widget)
            rb_layout.setContentsMargins(0, 10, 0, 10)
            
            rb = QRadioButton(text)
            if value == ai_config['detection_mode']:
                rb.setChecked(True)
            self.mode_buttons[value] = rb
            rb_layout.addWidget(rb)
            
            desc_label = QLabel(desc)
            desc_label.setStyleSheet("color: gray; font-size: 9pt;")
            desc_label.setContentsMargins(25, 0, 0, 0)
            rb_layout.addWidget(desc_label)
            
            mode_layout.addWidget(rb_widget)
        
        layout.addWidget(mode_box)
        
        # Multi-method configuration
        multi_box = QGroupBox("Multi-Method Settings")
        multi_layout = QVBoxLayout(multi_box)
        
        req_label = QLabel("Number of methods required to agree:")
        req_label.setStyleSheet("font-size: 10pt;")
        multi_layout.addWidget(req_label)
        
        self.methods_required_spinbox = QSpinBox()
        self.methods_required_spinbox.setMinimum(1)
        self.methods_required_spinbox.setMaximum(6)
        self.methods_required_spinbox.setValue(
            ai_config['multi_method_requirements']['methods_required'])
        self.methods_required_spinbox.setFixedWidth(80)
        self._disable_mousewheel(self.methods_required_spinbox)
        multi_layout.addWidget(self.methods_required_spinbox)
        multi_layout.addSpacing(10)
        
        min_label = QLabel("Required methods (at least one must be included):")
        min_label.setStyleSheet("font-size: 10pt;")
        multi_layout.addWidget(min_label)
        multi_layout.addSpacing(5)
        
        self.required_method_checkboxes = {}
        for method in ['exact', 'text', 'semantic', 'structural', 'character', 'pattern']:
            cb = self._create_styled_checkbox(method.title())
            cb.setChecked(method in ai_config['multi_method_requirements']['min_methods'])
            cb.setContentsMargins(20, 0, 0, 0)
            self.required_method_checkboxes[method] = cb
            multi_layout.addWidget(cb)
        
        layout.addWidget(multi_box)
        layout.addStretch()
        
        scroll.setWidget(frame)
        tabs.addTab(scroll, "Detection Mode")
    
    def create_preprocessing_tab(self, tabs):
        """Create the preprocessing configuration tab (PySide6)"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        frame = QWidget()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        title = QLabel("Text Preprocessing Options")
        title.setStyleSheet("font-size: 12pt; font-weight: bold;")
        layout.addWidget(title)
        
        desc = QLabel("Configure how text is processed before comparison")
        desc.setStyleSheet("color: gray; font-size: 10pt;")
        layout.addWidget(desc)
        layout.addSpacing(10)
        
        # Preprocessing options
        self.prep_checkboxes = {}
        ai_config = self.get_ai_config()
        
        options = [
            ('remove_html_spacing', 'Remove HTML with spacing', 
             'Replace HTML tags with spaces instead of removing completely'),
            ('normalize_unicode', 'Normalize Unicode', 
             'Normalize unicode characters (recommended)'),
            ('ignore_case', 'Case-insensitive comparison', 
             'Ignore character case when comparing'),
            ('remove_extra_whitespace', 'Remove extra whitespace', 
             'Collapse multiple spaces/newlines into single spaces')
        ]
        
        for key, text, desc_text in options:
            opt_widget = QWidget()
            opt_layout = QVBoxLayout(opt_widget)
            opt_layout.setContentsMargins(0, 10, 0, 10)
            
            cb = self._create_styled_checkbox(text)
            cb.setChecked(ai_config['preprocessing'][key])
            self.prep_checkboxes[key] = cb
            opt_layout.addWidget(cb)
            
            desc_label = QLabel(desc_text)
            desc_label.setStyleSheet("color: gray; font-size: 9pt;")
            desc_label.setContentsMargins(25, 0, 0, 0)
            opt_layout.addWidget(desc_label)
            
            layout.addWidget(opt_widget)
        
        layout.addStretch()
        scroll.setWidget(frame)
        tabs.addTab(scroll, "Preprocessing")
    
    def create_advanced_tab(self, tabs):
        """Create the advanced settings tab (PySide6)"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        frame = QWidget()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        title = QLabel("Advanced Settings")
        title.setStyleSheet("font-size: 12pt; font-weight: bold;")
        layout.addWidget(title)
        layout.addSpacing(10)
        
        ai_config = self.get_ai_config()
        
        # General settings
        general_box = QGroupBox("General")
        general_layout = QVBoxLayout(general_box)
        
        # Sample size
        ss_widget = QWidget()
        ss_layout = QHBoxLayout(ss_widget)
        ss_layout.setContentsMargins(0, 0, 0, 0)
        
        ss_label = QLabel("Sample size:")
        ss_label.setFixedWidth(150)
        ss_layout.addWidget(ss_label)
        
        self.sample_size_spinbox = QSpinBox()
        self.sample_size_spinbox.setMinimum(1000)
        self.sample_size_spinbox.setMaximum(10000)
        self.sample_size_spinbox.setSingleStep(500)
        self.sample_size_spinbox.setValue(ai_config['sample_size'])
        self.sample_size_spinbox.setFixedWidth(100)
        self._disable_mousewheel(self.sample_size_spinbox)
        ss_layout.addWidget(self.sample_size_spinbox)
        
        ss_unit = QLabel("characters")
        ss_unit.setStyleSheet("color: gray; font-size: 9pt;")
        ss_layout.addWidget(ss_unit)
        ss_layout.addStretch()
        general_layout.addWidget(ss_widget)
        
        # AI Hunter Behavior Settings
        behavior_label = QLabel("AI Hunter Behavior")
        behavior_label.setStyleSheet("font-size: 10pt; font-weight: bold;")
        general_layout.addWidget(behavior_label)
        general_layout.addSpacing(5)
        
        # Retry Attempts
        retry_widget = QWidget()
        retry_layout = QHBoxLayout(retry_widget)
        retry_layout.setContentsMargins(0, 0, 0, 0)
        
        retry_label = QLabel("Retry attempts:")
        retry_label.setFixedWidth(150)
        retry_layout.addWidget(retry_label)
        
        self.retry_attempts_spinbox = QSpinBox()
        self.retry_attempts_spinbox.setMinimum(1)
        self.retry_attempts_spinbox.setMaximum(10)
        self.retry_attempts_spinbox.setValue(ai_config.get('retry_attempts', 3))
        self.retry_attempts_spinbox.setFixedWidth(100)
        self._disable_mousewheel(self.retry_attempts_spinbox)
        retry_layout.addWidget(self.retry_attempts_spinbox)
        
        retry_unit = QLabel("attempts")
        retry_unit.setStyleSheet("color: gray; font-size: 9pt;")
        retry_layout.addWidget(retry_unit)
        retry_layout.addStretch()
        general_layout.addWidget(retry_widget)
        
        # Temperature Change Toggle
        temp_widget = QWidget()
        temp_layout = QVBoxLayout(temp_widget)
        temp_layout.setContentsMargins(0, 10, 0, 0)
        
        self.disable_temp_change_checkbox = self._create_styled_checkbox("Disable temperature change behavior")
        self.disable_temp_change_checkbox.setChecked(ai_config.get('disable_temperature_change', False))
        temp_layout.addWidget(self.disable_temp_change_checkbox)
        
        temp_desc = QLabel("Prevents AI Hunter from modifying temperature settings during retries")
        temp_desc.setStyleSheet("color: gray; font-size: 9pt;")
        temp_desc.setContentsMargins(25, 0, 0, 0)
        temp_layout.addWidget(temp_desc)
        general_layout.addWidget(temp_widget)
        
        layout.addWidget(general_box)
        
        # Edge filters
        edge_box = QGroupBox("Edge Case Filters")
        edge_layout = QVBoxLayout(edge_box)
        
        # Min text length
        min_widget = QWidget()
        min_layout = QHBoxLayout(min_widget)
        min_layout.setContentsMargins(0, 0, 0, 0)
        
        min_label = QLabel("Minimum text length:")
        min_label.setFixedWidth(150)
        min_layout.addWidget(min_label)
        
        self.min_length_spinbox = QSpinBox()
        self.min_length_spinbox.setMinimum(100)
        self.min_length_spinbox.setMaximum(2000)
        self.min_length_spinbox.setSingleStep(100)
        self.min_length_spinbox.setValue(ai_config['edge_filters']['min_text_length'])
        self.min_length_spinbox.setFixedWidth(100)
        self._disable_mousewheel(self.min_length_spinbox)
        min_layout.addWidget(self.min_length_spinbox)
        
        min_unit = QLabel("characters")
        min_unit.setStyleSheet("color: gray; font-size: 9pt;")
        min_layout.addWidget(min_unit)
        min_layout.addStretch()
        edge_layout.addWidget(min_widget)
        
        # Length ratios
        ratio_title = QLabel("Length ratio limits:")
        edge_layout.addWidget(ratio_title)
        edge_layout.addSpacing(5)
        
        ratio_widget = QWidget()
        ratio_layout = QHBoxLayout(ratio_widget)
        ratio_layout.setContentsMargins(20, 0, 0, 0)
        
        min_ratio_label = QLabel("Min ratio:")
        min_ratio_label.setFixedWidth(80)
        ratio_layout.addWidget(min_ratio_label)
        
        self.min_ratio_spinbox = QDoubleSpinBox()
        self.min_ratio_spinbox.setMinimum(0.5)
        self.min_ratio_spinbox.setMaximum(0.9)
        self.min_ratio_spinbox.setSingleStep(0.1)
        self.min_ratio_spinbox.setValue(ai_config['edge_filters']['min_length_ratio'])
        self.min_ratio_spinbox.setFixedWidth(80)
        self._disable_mousewheel(self.min_ratio_spinbox)
        ratio_layout.addWidget(self.min_ratio_spinbox)
        
        max_ratio_label = QLabel("Max ratio:")
        max_ratio_label.setFixedWidth(80)
        ratio_layout.addWidget(max_ratio_label)
        
        self.max_ratio_spinbox = QDoubleSpinBox()
        self.max_ratio_spinbox.setMinimum(1.1)
        self.max_ratio_spinbox.setMaximum(2.0)
        self.max_ratio_spinbox.setSingleStep(0.1)
        self.max_ratio_spinbox.setValue(ai_config['edge_filters']['max_length_ratio'])
        self.max_ratio_spinbox.setFixedWidth(80)
        self._disable_mousewheel(self.max_ratio_spinbox)
        ratio_layout.addWidget(self.max_ratio_spinbox)
        
        ratio_layout.addStretch()
        edge_layout.addWidget(ratio_widget)
        
        ratio_desc = QLabel("Chapters with vastly different lengths won't be compared")
        ratio_desc.setStyleSheet("color: gray; font-size: 9pt;")
        ratio_desc.setContentsMargins(20, 5, 0, 0)
        edge_layout.addWidget(ratio_desc)
        
        layout.addWidget(edge_box)
        
        # Language Detection
        lang_box = QGroupBox("Non-Target Language Detection")
        lang_layout = QVBoxLayout(lang_box)
        
        # Enable toggle
        enable_widget = QWidget()
        enable_layout = QVBoxLayout(enable_widget)
        enable_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lang_enabled_checkbox = self._create_styled_checkbox("Enable non-target language detection")
        self.lang_enabled_checkbox.setChecked(ai_config['language_detection']['enabled'])
        enable_layout.addWidget(self.lang_enabled_checkbox)
        
        enable_desc = QLabel("Trigger retranslation when too much non-target language is detected")
        enable_desc.setStyleSheet("color: gray; font-size: 9pt;")
        enable_desc.setContentsMargins(25, 0, 0, 0)
        enable_layout.addWidget(enable_desc)
        lang_layout.addWidget(enable_widget)
        lang_layout.addSpacing(10)
        
        # Target language selection
        target_widget = QWidget()
        target_layout = QHBoxLayout(target_widget)
        target_layout.setContentsMargins(0, 0, 0, 0)
        
        target_label = QLabel("Target language:")
        target_label.setFixedWidth(150)
        target_layout.addWidget(target_label)
        
        lang_options = list(ai_config['language_detection']['languages'].keys())
        self.target_lang_combo = QComboBox()
        self.target_lang_combo.addItems(lang_options)
        
        # Prioritize main config's target language for sync
        main_target_lang = self.config.get('glossary_target_language') or self.config.get('output_language')
        if main_target_lang:
             # Find closest match
             if main_target_lang in lang_options:
                 self.target_lang_combo.setCurrentText(main_target_lang)
             else:
                 # Try case-insensitive
                 found = False
                 for opt in lang_options:
                     if opt.lower() == main_target_lang.lower():
                         self.target_lang_combo.setCurrentText(opt)
                         found = True
                         break
                 if not found:
                     self.target_lang_combo.setCurrentText(ai_config['language_detection']['target_language'])
        else:
             self.target_lang_combo.setCurrentText(ai_config['language_detection']['target_language'])
             
        self.target_lang_combo.setFixedWidth(150)
        self._disable_mousewheel(self.target_lang_combo)
        target_layout.addWidget(self.target_lang_combo)
        
        target_desc = QLabel("Language that should be in the translation")
        target_desc.setStyleSheet("color: gray; font-size: 9pt;")
        target_layout.addWidget(target_desc)
        target_layout.addStretch()
        lang_layout.addWidget(target_widget)
        
        # Threshold setting
        thresh_widget = QWidget()
        thresh_layout = QHBoxLayout(thresh_widget)
        thresh_layout.setContentsMargins(0, 5, 0, 0)
        
        thresh_label = QLabel("Character threshold:")
        thresh_label.setFixedWidth(150)
        thresh_layout.addWidget(thresh_label)
        
        self.lang_threshold_spinbox = QSpinBox()
        self.lang_threshold_spinbox.setMinimum(100)
        self.lang_threshold_spinbox.setMaximum(2000)
        self.lang_threshold_spinbox.setSingleStep(50)
        self.lang_threshold_spinbox.setValue(ai_config['language_detection']['threshold_characters'])
        self.lang_threshold_spinbox.setFixedWidth(100)
        self._disable_mousewheel(self.lang_threshold_spinbox)
        thresh_layout.addWidget(self.lang_threshold_spinbox)
        
        thresh_desc = QLabel("non-target language characters to trigger retranslation")
        thresh_desc.setStyleSheet("color: gray; font-size: 9pt;")
        thresh_layout.addWidget(thresh_desc)
        thresh_layout.addStretch()
        lang_layout.addWidget(thresh_widget)
        
        layout.addWidget(lang_box)
        layout.addStretch()
        
        scroll.setWidget(frame)
        tabs.addTab(scroll, "Advanced")
    
    def apply_ai_hunter_settings(self):
        """Apply AI Hunter settings to the main config (PySide6)"""
        ai_config = self.get_ai_config()
        
        # Update from GUI variables
        for method, slider in self.threshold_vars.items():
            ai_config['thresholds'][method] = slider.value()
        
        for method, spinbox in self.weight_vars.items():
            ai_config['weights'][method] = spinbox.value()
        
        # Get selected detection mode
        for mode_value, radio_btn in self.mode_buttons.items():
            if radio_btn.isChecked():
                ai_config['detection_mode'] = mode_value
                break
        
        ai_config['multi_method_requirements']['methods_required'] = self.methods_required_spinbox.value()
        
        min_methods = [method for method, cb in self.required_method_checkboxes.items() if cb.isChecked()]
        ai_config['multi_method_requirements']['min_methods'] = min_methods
        
        for key, cb in self.prep_checkboxes.items():
            ai_config['preprocessing'][key] = cb.isChecked()
        
        ai_config['sample_size'] = self.sample_size_spinbox.value()
        
        ai_config['edge_filters']['min_text_length'] = self.min_length_spinbox.value()
        ai_config['edge_filters']['min_length_ratio'] = self.min_ratio_spinbox.value()
        ai_config['edge_filters']['max_length_ratio'] = self.max_ratio_spinbox.value()
        
        # Language detection settings
        ai_config['language_detection']['enabled'] = self.lang_enabled_checkbox.isChecked()
        new_target_lang = self.target_lang_combo.currentText()
        ai_config['language_detection']['target_language'] = new_target_lang
        ai_config['language_detection']['threshold_characters'] = self.lang_threshold_spinbox.value()
        
        # Sync back to main config
        self.config['output_language'] = new_target_lang
        self.config['glossary_target_language'] = new_target_lang
        # Also update environment variable immediately
        os.environ['OUTPUT_LANGUAGE'] = new_target_lang
        os.environ['GLOSSARY_TARGET_LANGUAGE'] = new_target_lang
        
        # Update retry attempts and temperature change settings
        ai_config['retry_attempts'] = self.retry_attempts_spinbox.value()
        ai_config['disable_temperature_change'] = self.disable_temp_change_checkbox.isChecked()
        
        # Update main config
        self.config['ai_hunter_config'] = ai_config
        
        # Call callback if provided (this should trigger main save_configuration)
        # The callback (save_config) will show its own success message
        if self.callback:
            self.callback()
        
        self.window.close()
    
    def reset_defaults(self):
        """Reset all values to defaults (PySide6)"""
        msg_box = QMessageBox(self.window)
        msg_box.setWindowTitle("Reset to Defaults")
        msg_box.setText("Are you sure you want to reset all settings to defaults?")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
        msg_box.setIcon(QMessageBox.Question)
        try:
            msg_box.setWindowIcon(QIcon("halgakos.ico"))
        except Exception:
            pass
        reply = msg_box.exec()
        
        if reply == QMessageBox.Yes:
            self.config['ai_hunter_config'] = self.default_ai_hunter.copy()
            self.window.close()
            self.show_ai_hunter_config()  # Reopen with default values


class ImprovedAIHunterDetection:
    """Improved AI Hunter detection methods for TranslateKRtoEN"""
    
    def __init__(self, main_config):
        """
        Initialize with reference to main config
        
        Args:
            main_config: Reference to main translator config dictionary
        """
        self.main_config = main_config
        
        # Default AI Hunter settings
        self.default_ai_hunter = {
            'enabled': True,
            'lookback_chapters': 5,
            'retry_attempts': 3,
            'disable_temperature_change': False,
            'sample_size': 3000,
            'thresholds': {
                'exact': 90,
                'text': 85,
                'semantic': 85,
                'structural': 85,
                'character': 80,
                'pattern': 80
            },
            'weights': {
                'exact': 1.5,
                'text': 1.2,
                'semantic': 1.0,
                'structural': 1.0,
                'character': 0.8,
                'pattern': 0.8
            },
            'detection_mode': 'multi_method',
            'multi_method_requirements': {
                'methods_required': 2,
                'min_methods': ['semantic', 'structural']
            },
            'preprocessing': {
                'remove_html_spacing': True,
                'normalize_unicode': True,
                'ignore_case': True,
                'remove_extra_whitespace': True
            },
            'edge_filters': {
                'min_text_length': 500,
                'max_length_ratio': 1.3,
                'min_length_ratio': 0.7
            },
            'language_detection': {
                'enabled': False,
                'target_language': 'english',
                'threshold_characters': 500,
                'languages': {
                    'english': ['en'],
                    'japanese': ['ja', 'jp'],
                    'korean': ['ko', 'kr'],
                    'chinese': ['zh', 'zh-cn', 'zh-tw'],
                    'spanish': ['es'],
                    'french': ['fr'],
                    'german': ['de'],
                    'russian': ['ru'],
                    'arabic': ['ar'],
                    'hindi': ['hi'],
                    'portuguese': ['pt'],
                    'italian': ['it'],
                    'dutch': ['nl'],
                    'thai': ['th'],
                    'vietnamese': ['vi'],
                    'turkish': ['tr'],
                    'polish': ['pl'],
                    'swedish': ['sv'],
                    'danish': ['da'],
                    'norwegian': ['no'],
                    'finnish': ['fi']
                }
            }
        }
    
    def get_ai_config(self):
        """Get AI Hunter configuration from main config"""
        return self.main_config.get('ai_hunter_config', self.default_ai_hunter)

    def detect_duplicate_ai_hunter_enhanced(self, result, idx, prog, out, current_chapter_num=None):
        """Enhanced AI Hunter duplicate detection with configurable parameters"""
        try:
            print(f"\n    ========== AI HUNTER DEBUG START ==========")
            print(f"    📍 Current chapter index: {idx}")
            if current_chapter_num:
                print(f"    📖 Current chapter number: {current_chapter_num}")
            
            # Get configuration
            config = self.get_ai_config()
            
            if not config.get('enabled', True):
                print(f"    ⚠️ AI Hunter is disabled")
                print(f"    ========== AI HUNTER DEBUG END ==========\n")
                return False, 0
            
            # Preprocess text
            result_clean = self._preprocess_text(result, config['preprocessing'])
            print(f"    📄 Text length after preprocessing: {len(result_clean)} chars")
            
            # Check for non-target language detection
            if config['language_detection']['enabled']:
                non_target_detected, non_target_count = self._check_non_target_language(
                    result_clean, config['language_detection']
                )
                if non_target_detected:
                    print(f"\n    🌐 NON-TARGET LANGUAGE DETECTED!")
                    print(f"       Non-target characters found: {non_target_count}")
                    print(f"       Threshold: {config['language_detection']['threshold_characters']}")
                    print(f"       Target language: {config['language_detection']['target_language']}")
                    print(f"    ========== AI HUNTER DEBUG END ==========\n")
                    return True, 100  # High confidence for language detection
            
            # Check edge cases
            if len(result_clean) < config['edge_filters']['min_text_length']:
                print(f"    ⚠️ Text too short ({len(result_clean)} < {config['edge_filters']['min_text_length']})")
                print(f"    ========== AI HUNTER DEBUG END ==========\n")
                return False, 0
            
            # Extract features
            print(f"    🔬 Extracting text features...")
            result_features = self._extract_text_features(result_clean)
            
            # Get lookback from main config, then fall back to env var if not found
            lookback = self.main_config.get('duplicate_lookback_chapters', 
                                           int(os.getenv('DUPLICATE_LOOKBACK_CHAPTERS', '5')))
            
            # Log configuration
            print(f"\n    🔧 Configuration:")
            print(f"       Detection mode: {config['detection_mode']}")
            print(f"       Lookback chapters: {lookback}")
            print(f"       Sample size: {config['sample_size']}")
            
            # FIX: Get all completed chapters sorted by actual chapter number
            completed_chapters = []
            for chapter_key, chapter_info in prog["chapters"].items():
                if chapter_info.get("status") == "completed" and chapter_info.get("output_file"):
                    # Handle both numeric and hash-based chapter keys
                    try:
                        # Get actual_num from progress (this is the real chapter number)
                        chapter_num = chapter_info.get("actual_num")
                        if chapter_num is None:
                            # Try chapter_num as fallback
                            chapter_num = chapter_info.get("chapter_num")
                        if chapter_num is None:
                            # Skip chapters without valid numbers
                            print(f"       ⚠️ No chapter number found for key {chapter_key}, skipping")
                            continue

                        completed_chapters.append({
                            'key': chapter_key,
                            'num': chapter_num,
                            'file': chapter_info.get("output_file"),
                            'ai_features': chapter_info.get("ai_features")
                        })
                    except Exception as e:
                        print(f"       ⚠️ Error processing chapter {chapter_key}: {e}")
                        continue
            
            # Sort by actual chapter number
            completed_chapters.sort(key=lambda x: x['num'])
            
            # If no current chapter number provided, try to infer it
            if current_chapter_num is None:
                # The current chapter should be passed in, but if not, we need to find it
                # Since we're using content hash keys, we can't use idx directly
                print(f"    ⚠️ No current chapter number provided")
                print(f"    📊 Current index: {idx}")
                
                # The current chapter number should have been passed from the wrapper
                # If it wasn't, we have a problem
                print(f"    ❌ ERROR: Current chapter number not provided to AI Hunter!")
                print(f"    ❌ This indicates the wrapper function is not passing the chapter number correctly")
                
                # Emergency: just use a high number so we don't compare against anything
                current_chapter_num = 999999
                print(f"    ⚠️ Using index-based chapter number: {current_chapter_num}")
            
            print(f"\n    📚 Found {len(completed_chapters)} completed chapters in progress")
            if completed_chapters:
                chapter_nums = [ch['num'] for ch in completed_chapters]
                print(f"    📊 Chapter numbers in progress: {sorted(chapter_nums)[:10]}{'...' if len(chapter_nums) > 10 else ''}")
            print(f"    🎯 Current chapter number: {current_chapter_num}")
            print(f"    🔍 Will check against last {lookback} chapters before chapter {current_chapter_num}")
            
            # Check previous chapters
            all_similarities = []
            highest_similarity = 0.0
            detected_method = None
            detected_chapter = None
            
            # FIX: Look at chapters by actual number, not index
            chapters_checked = 0
            for completed_chapter in reversed(completed_chapters):
                # Only check chapters that come before the current one
                if completed_chapter['num'] >= current_chapter_num:
                    continue
                    
                # Only check up to lookback number of chapters
                if chapters_checked >= lookback:
                    break
                    
                chapters_checked += 1
                
                print(f"\n    📝 Checking against chapter {completed_chapter['num']}...")
                
                # Get previous chapter features
                prev_features = completed_chapter.get('ai_features')
                prev_clean = None
                
                # Try to get cached features first
                if prev_features:
                    print(f"       ✅ Using cached features")
                else:
                    # Read and extract features
                    prev_path = os.path.join(out, completed_chapter['file'])
                    
                    if os.path.exists(prev_path):
                        try:
                            with open(prev_path, 'r', encoding='utf-8') as f:
                                prev_content = f.read()
                                prev_clean = self._preprocess_text(prev_content, config['preprocessing'])
                                
                                # Check length ratio
                                len_ratio = len(result_clean) / max(1, len(prev_clean))
                                if (len_ratio < config['edge_filters']['min_length_ratio'] or 
                                    len_ratio > config['edge_filters']['max_length_ratio']):
                                    print(f"       ⚠️ Length ratio out of bounds: {len_ratio:.2f}")
                                    continue
                                
                                prev_features = self._extract_text_features(prev_clean)
                                print(f"       📄 Extracted features from file")
                        except Exception as e:
                            print(f"       ❌ Failed to read file: {e}")
                            continue
                    else:
                        print(f"       ❌ File not found: {prev_path}")
                        continue
                
                # Calculate similarities
                print(f"       🔍 Calculating similarities...")
                similarities = self._calculate_all_similarities(
                    result_clean, result_features, 
                    prev_clean, prev_features, config
                )
                
                # Store for reporting
                all_similarities.append({
                    'chapter': completed_chapter['num'],
                    'similarities': similarities
                })
                
                # Log similarity scores
                for method, score in similarities.items():
                    if score > 0:
                        print(f"          {method}: {int(score*100)}%")
                
                # Check if duplicate based on configured mode
                is_duplicate, confidence, methods_triggered = self._evaluate_duplicate(
                    similarities, config
                )
                
                if is_duplicate:
                    print(f"\n    🚨 DUPLICATE DETECTED!")
                    print(f"       Detection mode: {config['detection_mode']}")
                    print(f"       Confidence: {int(confidence*100)}%")
                    print(f"       Triggered methods: {', '.join(methods_triggered)}")
                    print(f"       Match with: Chapter {completed_chapter['num']}")
                    print(f"    ========== AI HUNTER DEBUG END ==========\n")
                    return True, int(confidence * 100)
                
                # Track highest for reporting
                for method, sim in similarities.items():
                    if sim > highest_similarity:
                        highest_similarity = sim
                        detected_method = method
                        detected_chapter = completed_chapter['num']
            
            # No duplicate found
            print(f"\n    ✅ No duplicate found")
            if detected_method:
                print(f"       Highest similarity: {int(highest_similarity*100)}% via {detected_method}")
                print(f"       Closest match: Chapter {detected_chapter}")
            
            # Show top 3 closest matches
            if all_similarities:
                print(f"\n    📊 Top 3 closest matches:")
                sorted_chapters = sorted(all_similarities, 
                                       key=lambda x: self._get_chapter_score(x['similarities'], config), 
                                       reverse=True)[:3]
                for i, chapter_data in enumerate(sorted_chapters, 1):
                    score = self._get_chapter_score(chapter_data['similarities'], config)
                    print(f"       {i}. Chapter {chapter_data['chapter']}: {int(score*100)}%")
            
            print(f"    ========== AI HUNTER DEBUG END ==========\n")
            return False, 0
            
        except Exception as e:
            print(f"    ❌ AI Hunter detection failed with error: {e}")
            import traceback
            print(f"    {traceback.format_exc()}")
            print(f"    ========== AI HUNTER DEBUG END ==========\n")
            return False, 0
    
    def _preprocess_text(self, text, prep_config):
        """Preprocess text according to configuration"""
        # Remove HTML
        if prep_config.get('remove_html_spacing', True):
            text = re.sub(r'<[^>]+>', ' ', text)
        else:
            text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize unicode
        if prep_config.get('normalize_unicode', True):
            text = unicodedata.normalize('NFKD', text)
        
        # Remove extra whitespace
        if prep_config.get('remove_extra_whitespace', True):
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
        
        text = text.strip()
        
        # Convert to lowercase if case-insensitive
        if prep_config.get('ignore_case', True):
            text = text.lower()
        
        return text
    
    def _calculate_all_similarities(self, result_clean, result_features, 
                                   prev_clean, prev_features, config):
        """Calculate all similarity metrics"""
        similarities = {}
        
        # Method 1: Exact content match
        if prev_clean is not None:
            sample_size = min(config['sample_size'], len(result_clean), len(prev_clean))
            exact_sim = self._calculate_exact_similarity(
                result_clean[:sample_size], 
                prev_clean[:sample_size]
            )
            similarities['exact'] = exact_sim
            
            # Method 2: Smart text similarity
            text_sim = self._calculate_smart_similarity(
                result_clean, prev_clean, config['sample_size']
            )
            similarities['text'] = text_sim
        else:
            similarities['exact'] = 0.0
            similarities['text'] = 0.0
        
        # Method 3: Semantic fingerprint
        semantic_sim = self._calculate_semantic_similarity(
            result_features.get('semantic', {}), 
            prev_features.get('semantic', {})
        )
        similarities['semantic'] = semantic_sim
        
        # Method 4: Structural signature
        structural_sim = self._calculate_structural_similarity(
            result_features.get('structural', {}), 
            prev_features.get('structural', {})
        )
        similarities['structural'] = structural_sim
        
        # Method 5: Character analysis
        char_sim = self._calculate_character_similarity(
            result_features.get('characters', []), 
            prev_features.get('characters', [])
        )
        similarities['character'] = char_sim
        
        # Method 6: Pattern analysis
        pattern_sim = self._calculate_pattern_similarity(
            result_features.get('patterns', {}), 
            prev_features.get('patterns', {})
        )
        similarities['pattern'] = pattern_sim
        
        return similarities
    
    def _evaluate_duplicate(self, similarities, config):
        """Evaluate if similarities indicate a duplicate based on detection mode"""
        mode = config['detection_mode']
        thresholds = {k: v/100.0 for k, v in config['thresholds'].items()}
        
        if mode == 'single_method':
            # Any method exceeding threshold
            for method, sim in similarities.items():
                if sim >= thresholds.get(method, 0.85):
                    return True, sim, [method]
            return False, 0, []
        
        elif mode == 'multi_method':
            # Multiple methods must agree
            triggered_methods = []
            for method, sim in similarities.items():
                if sim >= thresholds.get(method, 0.85):
                    triggered_methods.append(method)
            
            # Check if enough methods triggered
            required = config.get('multi_method_requirements', {}).get('methods_required', 2)
            min_methods = config.get('multi_method_requirements', {}).get('min_methods', [])
            
            if len(triggered_methods) >= required:
                # Check if at least one required method is included
                if not min_methods or any(m in triggered_methods for m in min_methods):
                    # Calculate average confidence of triggered methods
                    confidence = sum(similarities[m] for m in triggered_methods) / len(triggered_methods)
                    return True, confidence, triggered_methods
            
            return False, 0, []
        
        elif mode == 'weighted_average':
            # Calculate weighted average
            weights = config.get('weights', {})
            total_weight = sum(weights.get(m, 1.0) for m in similarities)
            weighted_sum = sum(similarities[m] * weights.get(m, 1.0) for m in similarities)
            weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0
            
            # Check if weighted average exceeds average threshold
            avg_threshold = sum(thresholds.values()) / len(thresholds) if thresholds else 0.85
            
            if weighted_avg >= avg_threshold:
                # Find which methods contributed most
                triggered = [m for m, sim in similarities.items() 
                           if sim >= thresholds.get(m, 0.85)]
                return True, weighted_avg, triggered
            
            return False, 0, []
        
        return False, 0, []
    
    def _get_chapter_score(self, similarities, config):
        """Calculate overall score for a chapter comparison"""
        if config['detection_mode'] == 'weighted_average':
            weights = config.get('weights', {})
            total_weight = sum(weights.get(m, 1.0) for m in similarities)
            return sum(similarities.get(m, 0) * weights.get(m, 1.0) for m in similarities) / total_weight if total_weight > 0 else 0
        else:
            return max(similarities.values()) if similarities else 0
    
    def _extract_text_features(self, text):
        """Extract multiple features from text for AI Hunter analysis"""
        features = {
            'semantic': {},
            'structural': {},
            'characters': [],
            'patterns': {}
        }
        
        # Semantic fingerprint
        lines = text.split('\n')
        
        # Character extraction (names that appear 3+ times)
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        word_freq = Counter(words)
        features['characters'] = [name for name, count in word_freq.items() 
                                 if count >= 3 and name not in {
                                     'The', 'A', 'An', 'In', 'On', 'At', 'To', 
                                     'From', 'With', 'By', 'For', 'Of', 'As', 
                                     'But', 'And', 'Or', 'He', 'She', 'It', 
                                     'They', 'We', 'You', 'What', 'When', 'Where',
                                     'Who', 'Why', 'How', 'That', 'This', 'These'
                                 }]
        
        # Dialogue patterns
        dialogue_patterns = re.findall(r'"([^"]+)"', text)
        features['semantic']['dialogue_count'] = len(dialogue_patterns)
        features['semantic']['dialogue_lengths'] = [len(d) for d in dialogue_patterns[:10]]
        
        # Speaker patterns
        speaker_patterns = re.findall(r'(\w+)\s+(?:said|asked|replied|shouted|whispered)', text.lower())
        features['semantic']['speakers'] = list(set(speaker_patterns[:20]))
        
        # Number extraction
        numbers = re.findall(r'\b\d+\b', text)
        features['patterns']['numbers'] = numbers[:20]
        
        # Structural signature
        para_lengths = []
        dialogue_count = 0
        for para in text.split('\n\n'):
            if para.strip():
                para_lengths.append(len(para))
                if '"' in para:
                    dialogue_count += 1
        
        features['structural']['para_count'] = len(para_lengths)
        features['structural']['avg_para_length'] = sum(para_lengths) / max(1, len(para_lengths))
        features['structural']['dialogue_ratio'] = dialogue_count / max(1, len(para_lengths))
        
        # Create structural pattern string
        pattern = []
        for para in text.split('\n\n')[:20]:  # First 20 paragraphs
            if para.strip():
                if '"' in para:
                    pattern.append('D')  # Dialogue
                elif len(para) > 300:
                    pattern.append('L')  # Long
                elif len(para) < 100:
                    pattern.append('S')  # Short
                else:
                    pattern.append('M')  # Medium
        features['structural']['pattern'] = ''.join(pattern)
        
        # Action density
        action_verbs = len(re.findall(r'\b\w+ed\b', text))
        features['semantic']['action_density'] = action_verbs / max(1, len(text.split()))
        
        # Text length
        features['semantic']['text_length'] = len(text)
        
        return features
    
    def _calculate_exact_similarity(self, text1, text2):
        """Calculate exact text similarity"""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _calculate_smart_similarity(self, text1, text2, sample_size):
        """Smart similarity with configurable sample size"""
        if len(text1) > sample_size * 3 and len(text2) > sample_size * 3:
            # Use multiple samples
            samples1 = [
                text1[:sample_size],
                text1[len(text1)//2 - sample_size//2:len(text1)//2 + sample_size//2],
                text1[-sample_size:]
            ]
            samples2 = [
                text2[:sample_size],
                text2[len(text2)//2 - sample_size//2:len(text2)//2 + sample_size//2],
                text2[-sample_size:]
            ]
            similarities = [SequenceMatcher(None, s1, s2).ratio() 
                           for s1, s2 in zip(samples1, samples2)]
            return sum(similarities) / len(similarities)
        else:
            # Use full text up to sample size
            return SequenceMatcher(None, text1[:sample_size], text2[:sample_size]).ratio()
    
    def _calculate_semantic_similarity(self, sem1, sem2):
        """Calculate semantic fingerprint similarity"""
        score = 0.0
        weights = 0.0
        
        # Compare dialogue counts
        if 'dialogue_count' in sem1 and 'dialogue_count' in sem2:
            weights += 0.3
            if sem1['dialogue_count'] > 0 or sem2['dialogue_count'] > 0:
                ratio = min(sem1['dialogue_count'], sem2['dialogue_count']) / \
                       max(1, max(sem1['dialogue_count'], sem2['dialogue_count']))
                score += ratio * 0.3
        
        # Compare speakers
        if 'speakers' in sem1 and 'speakers' in sem2:
            weights += 0.4
            if sem1['speakers'] and sem2['speakers']:
                overlap = len(set(sem1['speakers']) & set(sem2['speakers']))
                total = len(set(sem1['speakers']) | set(sem2['speakers']))
                score += (overlap / max(1, total)) * 0.4
            elif not sem1['speakers'] and not sem2['speakers']:
                score += 0.4  # Both have no speakers
        
        # Compare dialogue lengths pattern
        if 'dialogue_lengths' in sem1 and 'dialogue_lengths' in sem2:
            weights += 0.2
            if sem1['dialogue_lengths'] and sem2['dialogue_lengths']:
                len1 = sem1['dialogue_lengths'][:10]
                len2 = sem2['dialogue_lengths'][:10]
                if len1 and len2:
                    avg1 = sum(len1) / len(len1)
                    avg2 = sum(len2) / len(len2)
                    ratio = min(avg1, avg2) / max(1, max(avg1, avg2))
                    score += ratio * 0.2
            elif not sem1['dialogue_lengths'] and not sem2['dialogue_lengths']:
                score += 0.2  # Both have no dialogue
        
        # Action density
        if 'action_density' in sem1 and 'action_density' in sem2:
            weights += 0.1
            act_sim = 1 - abs(sem1['action_density'] - sem2['action_density'])
            score += act_sim * 0.1
        
        return score / max(0.1, weights)
    
    def _calculate_structural_similarity(self, struct1, struct2):
        """Calculate structural signature similarity"""
        score = 0.0
        
        # Compare paragraph patterns
        if 'pattern' in struct1 and 'pattern' in struct2:
            pattern_sim = SequenceMatcher(None, struct1['pattern'], struct2['pattern']).ratio()
            score += pattern_sim * 0.5
        
        # Compare paragraph statistics
        if all(k in struct1 for k in ['para_count', 'avg_para_length', 'dialogue_ratio']) and \
           all(k in struct2 for k in ['para_count', 'avg_para_length', 'dialogue_ratio']):
            
            # Paragraph count ratio
            para_ratio = min(struct1['para_count'], struct2['para_count']) / \
                        max(1, max(struct1['para_count'], struct2['para_count']))
            score += para_ratio * 0.2
            
            # Average length ratio
            avg_ratio = min(struct1['avg_para_length'], struct2['avg_para_length']) / \
                       max(1, max(struct1['avg_para_length'], struct2['avg_para_length']))
            score += avg_ratio * 0.15
            
            # Dialogue ratio similarity
            dialogue_diff = abs(struct1['dialogue_ratio'] - struct2['dialogue_ratio'])
            score += (1 - min(1, dialogue_diff)) * 0.15
        
        return score
    
    def _calculate_character_similarity(self, chars1, chars2):
        """Calculate character overlap similarity"""
        if not chars1 or not chars2:
            return 0.0
        
        # Convert to sets
        set1 = set(chars1)
        set2 = set(chars2)
        
        # If no overlap at all, return 0
        intersection = set1 & set2
        if not intersection:
            return 0.0
        
        # Calculate Jaccard index (intersection over union)
        union = set1 | set2
        jaccard = len(intersection) / len(union)
        
        # Also consider the proportion of matching characters relative to each set
        # This prevents small overlaps from scoring too high
        overlap1 = len(intersection) / len(set1)
        overlap2 = len(intersection) / len(set2)
        
        # Take the minimum overlap to be more conservative
        min_overlap = min(overlap1, overlap2)
        
        # Combine jaccard and overlap scores
        # Jaccard penalizes when sets are very different sizes
        # Min overlap ensures both texts share a significant portion of characters
        score = (jaccard + min_overlap) / 2
        
        return score
    
    def _calculate_pattern_similarity(self, pat1, pat2):
        """Calculate pattern similarity (numbers, etc.)"""
        score = 0.0
        
        # Number overlap
        if 'numbers' in pat1 and 'numbers' in pat2:
            nums1 = set(pat1['numbers'])
            nums2 = set(pat2['numbers'])
            
            if nums1 or nums2:
                overlap = len(nums1 & nums2)
                total = len(nums1 | nums2)
                score = overlap / max(1, total)
            else:
                score = 1.0  # Both have no numbers
        
        return score
    
    def _check_non_target_language(self, text, lang_config):
        """Check if text contains too much non-target language"""
        target_language = lang_config['target_language'].lower()
        threshold = lang_config['threshold_characters']
        
        # Character ranges for different languages
        language_ranges = {
            'english': [  # Latin script + basic symbols
                (0x0000, 0x007F),  # Basic Latin
                (0x0080, 0x00FF),  # Latin-1 Supplement
                (0x0100, 0x017F),  # Latin Extended-A
                (0x0180, 0x024F),  # Latin Extended-B
                (0x2000, 0x206F),  # General Punctuation
                (0x20A0, 0x20CF),  # Currency Symbols
                (0xFF00, 0xFFEF),  # Halfwidth and Fullwidth Forms
            ],
            'japanese': [
                (0x3040, 0x309F),  # Hiragana
                (0x30A0, 0x30FF),  # Katakana
                (0x4E00, 0x9FAF),  # CJK Unified Ideographs
                (0x3400, 0x4DBF),  # CJK Extension A
                (0xFF66, 0xFF9F),  # Halfwidth Katakana
            ],
            'korean': [
                (0xAC00, 0xD7AF),  # Hangul Syllables
                (0x1100, 0x11FF),  # Hangul Jamo
                (0x3130, 0x318F),  # Hangul Compatibility Jamo
                (0xA960, 0xA97F),  # Hangul Jamo Extended-A
                (0xD7B0, 0xD7FF),  # Hangul Jamo Extended-B
            ],
            'chinese': [
                (0x4E00, 0x9FAF),  # CJK Unified Ideographs
                (0x3400, 0x4DBF),  # CJK Extension A
                (0x20000, 0x2A6DF), # CJK Extension B
                (0x2A700, 0x2B73F), # CJK Extension C
                (0x2B740, 0x2B81F), # CJK Extension D
                (0x3000, 0x303F),  # CJK Symbols and Punctuation
            ],
            'arabic': [
                (0x0600, 0x06FF),  # Arabic
                (0x0750, 0x077F),  # Arabic Supplement
                (0x08A0, 0x08FF),  # Arabic Extended-A
                (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
                (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
            ],
            'russian': [
                (0x0400, 0x04FF),  # Cyrillic
                (0x0500, 0x052F),  # Cyrillic Supplement
                (0x2DE0, 0x2DFF),  # Cyrillic Extended-A
                (0xA640, 0xA69F),  # Cyrillic Extended-B
            ],
            'thai': [
                (0x0E00, 0x0E7F),  # Thai
            ],
            'hindi': [
                (0x0900, 0x097F),  # Devanagari
                (0xA8E0, 0xA8FF),  # Devanagari Extended
            ],
            'spanish': [  # Same as English (Latin script)
                (0x0000, 0x007F),  # Basic Latin
                (0x0080, 0x00FF),  # Latin-1 Supplement
                (0x0100, 0x017F),  # Latin Extended-A
                (0x0180, 0x024F),  # Latin Extended-B
            ],
            'french': [  # Same as English (Latin script)
                (0x0000, 0x007F),  # Basic Latin
                (0x0080, 0x00FF),  # Latin-1 Supplement
                (0x0100, 0x017F),  # Latin Extended-A
                (0x0180, 0x024F),  # Latin Extended-B
            ],
            'german': [  # Same as English (Latin script)
                (0x0000, 0x007F),  # Basic Latin
                (0x0080, 0x00FF),  # Latin-1 Supplement
                (0x0100, 0x017F),  # Latin Extended-A
                (0x0180, 0x024F),  # Latin Extended-B
            ],
            'portuguese': [  # Same as English (Latin script)
                (0x0000, 0x007F),  # Basic Latin
                (0x0080, 0x00FF),  # Latin-1 Supplement
                (0x0100, 0x017F),  # Latin Extended-A
                (0x0180, 0x024F),  # Latin Extended-B
            ],
            'italian': [  # Same as English (Latin script)
                (0x0000, 0x007F),  # Basic Latin
                (0x0080, 0x00FF),  # Latin-1 Supplement
                (0x0100, 0x017F),  # Latin Extended-A
                (0x0180, 0x024F),  # Latin Extended-B
            ],
            'dutch': [  # Same as English (Latin script)
                (0x0000, 0x007F),  # Basic Latin
                (0x0080, 0x00FF),  # Latin-1 Supplement
                (0x0100, 0x017F),  # Latin Extended-A
                (0x0180, 0x024F),  # Latin Extended-B
            ],
            'vietnamese': [
                (0x0000, 0x007F),  # Basic Latin
                (0x0080, 0x00FF),  # Latin-1 Supplement
                (0x0100, 0x017F),  # Latin Extended-A
                (0x0180, 0x024F),  # Latin Extended-B
                (0x1EA0, 0x1EFF),  # Latin Extended Additional (Vietnamese)
            ],
            'turkish': [
                (0x0000, 0x007F),  # Basic Latin
                (0x0080, 0x00FF),  # Latin-1 Supplement
                (0x0100, 0x017F),  # Latin Extended-A
                (0x0180, 0x024F),  # Latin Extended-B
            ],
            'polish': [
                (0x0000, 0x007F),  # Basic Latin
                (0x0080, 0x00FF),  # Latin-1 Supplement
                (0x0100, 0x017F),  # Latin Extended-A
                (0x0180, 0x024F),  # Latin Extended-B
            ],
            'swedish': [  # Same as English (Latin script)
                (0x0000, 0x007F),  # Basic Latin
                (0x0080, 0x00FF),  # Latin-1 Supplement
                (0x0100, 0x017F),  # Latin Extended-A
                (0x0180, 0x024F),  # Latin Extended-B
            ],
            'danish': [  # Same as English (Latin script)
                (0x0000, 0x007F),  # Basic Latin
                (0x0080, 0x00FF),  # Latin-1 Supplement
                (0x0100, 0x017F),  # Latin Extended-A
                (0x0180, 0x024F),  # Latin Extended-B
            ],
            'norwegian': [  # Same as English (Latin script)
                (0x0000, 0x007F),  # Basic Latin
                (0x0080, 0x00FF),  # Latin-1 Supplement
                (0x0100, 0x017F),  # Latin Extended-A
                (0x0180, 0x024F),  # Latin Extended-B
            ],
            'finnish': [  # Same as English (Latin script)
                (0x0000, 0x007F),  # Basic Latin
                (0x0080, 0x00FF),  # Latin-1 Supplement
                (0x0100, 0x017F),  # Latin Extended-A
                (0x0180, 0x024F),  # Latin Extended-B
            ],
        }
        
        # Get target language ranges
        target_ranges = language_ranges.get(target_language, language_ranges['english'])
        
        # Count characters that are NOT in target language ranges
        non_target_count = 0
        total_letters = 0
        
        for char in text:
            # Skip whitespace, punctuation, and numbers for counting
            if char.isspace() or char.isdigit():
                continue
                
            # Count as letter character
            total_letters += 1
            
            # Check if character is in any target language range
            char_code = ord(char)
            is_target_char = any(start <= char_code <= end for start, end in target_ranges)
            
            if not is_target_char:
                non_target_count += 1
        
        # Debug logging
        if non_target_count > 0:
            print(f"       🌐 Language detection: {non_target_count}/{total_letters} non-target chars ({target_language})")
        
        # Return True if non-target character count exceeds threshold
        return non_target_count >= threshold, non_target_count
