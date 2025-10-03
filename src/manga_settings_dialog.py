# manga_settings_dialog.py
"""
Enhanced settings dialog for manga translation with all settings visible
Properly integrated with TranslatorGUI's WindowManager and UIHelper
"""

import os
import json
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, 
                                QLabel, QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox,
                                QSlider, QComboBox, QLineEdit, QGroupBox, QTabWidget,
                                QWidget, QScrollArea, QFrame, QRadioButton, QButtonGroup,
                                QMessageBox, QFileDialog, QSizePolicy)
from PySide6.QtCore import Qt, Signal, QTimer, QEvent, QObject
from PySide6.QtGui import QFont, QIcon
from typing import Dict, Any, Optional, Callable
from bubble_detector import BubbleDetector
import logging
import time
import copy

# Use the same logging infrastructure initialized by translator_gui
logger = logging.getLogger(__name__)

class MangaSettingsDialog(QDialog):
    """Settings dialog for manga translation"""
    
    def __init__(self, parent, main_gui, config: Dict[str, Any], callback: Optional[Callable] = None):
        """Initialize settings dialog
        
        Args:
            parent: Parent window (should be QWidget or None)
            main_gui: Reference to TranslatorGUI instance
            config: Configuration dictionary
            callback: Function to call after saving
        """
        # Ensure parent is a QWidget or None for proper PySide6 initialization
        if parent is not None and not hasattr(parent, 'windowTitle'):
            # If parent is not a QWidget, use None
            parent = None
        super().__init__(parent)
        self.parent = parent
        self.main_gui = main_gui
        self.config = config
        self.callback = callback
        
        # Enhanced default settings structure with all options
        self.default_settings = {
            'preprocessing': {
                'enabled': False,
                'auto_detect_quality': True,
                'contrast_threshold': 0.4,
                'sharpness_threshold': 0.3,
                'noise_threshold': 20,
                'enhancement_strength': 1.5,
                'denoise_strength': 10,
                'max_image_dimension': 2000,
                'max_image_pixels': 2000000,
                'chunk_height': 2000,
                'chunk_overlap': 100,
                # Inpainting tiling
                'inpaint_tiling_enabled': False,  # Off by default
                'inpaint_tile_size': 512,  # Default tile size
                'inpaint_tile_overlap': 64  # Overlap to avoid seams
            },
            'compression': {
                'enabled': False,
                'format': 'jpeg',
                'jpeg_quality': 85,
                'png_compress_level': 6,
                'webp_quality': 85
            },
'ocr': {
                'language_hints': ['ja', 'ko', 'zh'],
                'confidence_threshold': 0.7,
                'merge_nearby_threshold': 20,
                'azure_merge_multiplier': 3.0,
                'text_detection_mode': 'document',
                'enable_rotation_correction': True,
                'bubble_detection_enabled': True,
                'roi_locality_enabled': False,
                'bubble_model_path': '',
                'bubble_confidence': 0.5,
                'detector_type': 'rtdetr_onnx',
                'rtdetr_confidence': 0.3,
                'detect_empty_bubbles': True,
                'detect_text_bubbles': True,
                'detect_free_text': True,
                'rtdetr_model_url': '',
                'azure_reading_order': 'natural',
                'azure_model_version': 'latest',
                'azure_max_wait': 60,
                'azure_poll_interval': 0.5,
                'min_text_length': 0,
                'exclude_english_text': False,
                'english_exclude_threshold': 0.7,
                'english_exclude_min_chars': 4,
                'english_exclude_short_tokens': False
            },
            'advanced': {
                'format_detection': True,
                'webtoon_mode': 'auto',
                'debug_mode': False,
                'save_intermediate': False,
                'parallel_processing': True,
                'max_workers': 2,
                'parallel_panel_translation': False,
                'panel_max_workers': 2,
                'use_singleton_models': False,
                'auto_cleanup_models': False,
                'unload_models_after_translation': False,
                'auto_convert_to_onnx': False,  # Disabled by default
                'auto_convert_to_onnx_background': True,
                'quantize_models': False,
                'onnx_quantize': False,
                'torch_precision': 'fp16',
                # HD strategy defaults (mirrors comic-translate)
                'hd_strategy': 'resize',                # 'original' | 'resize' | 'crop'
                'hd_strategy_resize_limit': 1536,       # long-edge cap for resize
                'hd_strategy_crop_margin': 16,          # pixels padding around cropped ROIs
                'hd_strategy_crop_trigger_size': 1024,  # only crop if long edge exceeds this
                # RAM cap defaults
                'ram_cap_enabled': False,
                'ram_cap_mb': 4096,
                'ram_cap_mode': 'soft',
                'ram_gate_timeout_sec': 15.0,
                'ram_min_floor_over_baseline_mb': 256
                },
            'inpainting': {
                'batch_size': 10,
                'enable_cache': True,
                'method': 'local',
                'local_method': 'anime'
            },
            'font_sizing': {
            'algorithm': 'smart',  # 'smart', 'conservative', 'aggressive'
            'prefer_larger': True,  # Prefer larger readable text
            'max_lines': 10,  # Maximum lines before forcing smaller
            'line_spacing': 1.3,  # Line height multiplier
            'bubble_size_factor': True  # Scale font based on bubble size
            },
            
            # Mask dilation settings with new iteration controls
            'mask_dilation': 0,
            'use_all_iterations': True,  # Master control - use same for all by default
            'all_iterations': 2,  # Value when using same for all
            'text_bubble_dilation_iterations': 2,  # Text-filled speech bubbles
            'empty_bubble_dilation_iterations': 3,  # Empty speech bubbles
            'free_text_dilation_iterations': 0,  # Free text (0 for clean B&W)
            'bubble_dilation_iterations': 2,  # Legacy support
            'dilation_iterations': 2,  # Legacy support
            
            # Cloud inpainting settings
            'cloud_inpaint_model': 'ideogram-v2',
            'cloud_custom_version': '',
            'cloud_inpaint_prompt': 'clean background, smooth surface',
            'cloud_negative_prompt': 'text, writing, letters',
            'cloud_inference_steps': 20,
            'cloud_timeout': 60
        }
        
        # Merge with existing config
        self.settings = self._merge_settings(config.get('manga_settings', {}))
        
        # Show dialog
        self.show_dialog()
            
    def _disable_spinbox_scroll(self, widget):
        """Disable mouse wheel scrolling on a spinbox, combobox, or slider (PySide6 version)"""
        # Install event filter to block wheel events
        class WheelEventFilter(QObject):
            def eventFilter(self, obj, event):
                if event.type() == QEvent.Wheel:
                    event.ignore()
                    return True
                return False
        
        filter_obj = WheelEventFilter(widget)  # Parent it to the widget
        widget.installEventFilter(filter_obj)
        # Store the filter so it doesn't get garbage collected
        if not hasattr(widget, '_wheel_filter'):
            widget._wheel_filter = filter_obj

    def _disable_all_spinbox_scrolling(self, parent):
        """Recursively find and disable scrolling on all spinboxes, comboboxes, and sliders (PySide6 version)"""
        # Check if the parent itself is a spinbox, combobox, or slider
        if isinstance(parent, (QSpinBox, QDoubleSpinBox, QComboBox, QSlider)):
            self._disable_spinbox_scroll(parent)
        
        # Check all children recursively
        if hasattr(parent, 'children'):
            for child in parent.children():
                if isinstance(child, QWidget):
                    self._disable_all_spinbox_scrolling(child)
            
    def _create_font_size_controls(self, parent_layout):
        """Create improved font size controls with presets"""
        
        # Font size frame
        font_frame = QWidget()
        font_layout = QHBoxLayout(font_frame)
        font_layout.setContentsMargins(0, 0, 0, 0)
        parent_layout.addWidget(font_frame)
        
        label = QLabel("Font Size:")
        label.setMinimumWidth(150)
        font_layout.addWidget(label)
        
        # Font size mode selection
        mode_frame = QWidget()
        mode_layout = QHBoxLayout(mode_frame)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        font_layout.addWidget(mode_frame)
        
        # Radio buttons for mode - using QButtonGroup
        self.font_size_mode_group = QButtonGroup()
        self.font_size_mode = 'auto'  # Store mode as string attribute
        
        modes = [
            ("Auto", "auto", "Automatically fit text to bubble size"),
            ("Fixed", "fixed", "Use a specific font size"),
            ("Scale", "scale", "Scale auto size by percentage")
        ]
        
        for text, value, tooltip in modes:
            rb = QRadioButton(text)
            rb.setChecked(value == 'auto')
            rb.setToolTip(tooltip)
            rb.toggled.connect(lambda checked, v=value: self._on_font_mode_change(v) if checked else None)
            mode_layout.addWidget(rb)
            self.font_size_mode_group.addButton(rb)
        
        # Controls frame (changes based on mode)
        self.font_controls_frame = QWidget()
        self.font_controls_layout = QVBoxLayout(self.font_controls_frame)
        self.font_controls_layout.setContentsMargins(20, 5, 0, 5)
        parent_layout.addWidget(self.font_controls_frame)
        
        # Fixed size controls
        self.fixed_size_frame = QWidget()
        fixed_layout = QHBoxLayout(self.fixed_size_frame)
        fixed_layout.setContentsMargins(0, 0, 0, 0)
        fixed_layout.addWidget(QLabel("Size:"))
        
        self.fixed_font_size_spin = QSpinBox()
        self.fixed_font_size_spin.setRange(8, 72)
        self.fixed_font_size_spin.setValue(16)
        self.fixed_font_size_spin.valueChanged.connect(self._save_rendering_settings)
        fixed_layout.addWidget(self.fixed_font_size_spin)
        
        # Quick presets for fixed size
        fixed_layout.addWidget(QLabel("Presets:"))
        
        presets = [
            ("Small", 12),
            ("Medium", 16),
            ("Large", 20),
            ("XL", 24)
        ]
        
        for text, size in presets:
            btn = QPushButton(text)
            btn.setMaximumWidth(60)
            btn.clicked.connect(lambda checked, s=size: self._set_fixed_size(s))
            fixed_layout.addWidget(btn)
        
        fixed_layout.addStretch()
        
        # Scale controls
        self.scale_frame = QWidget()
        scale_layout = QHBoxLayout(self.scale_frame)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_layout.addWidget(QLabel("Scale:"))
        
        # QSlider uses integers, so we'll use 50-200 to represent 0.5-2.0
        self.font_scale_slider = QSlider(Qt.Horizontal)
        self.font_scale_slider.setRange(50, 200)
        self.font_scale_slider.setValue(100)
        self.font_scale_slider.setMinimumWidth(200)
        self.font_scale_slider.valueChanged.connect(self._update_scale_label)
        scale_layout.addWidget(self.font_scale_slider)
        
        self.scale_label = QLabel("100%")
        self.scale_label.setMinimumWidth(50)
        scale_layout.addWidget(self.scale_label)
        
        # Quick scale presets
        scale_layout.addWidget(QLabel("Quick:"))
        
        scale_presets = [
            ("75%", 0.75),
            ("100%", 1.0),
            ("125%", 1.25),
            ("150%", 1.5)
        ]
        
        for text, scale in scale_presets:
            btn = QPushButton(text)
            btn.setMaximumWidth(50)
            btn.clicked.connect(lambda checked, s=scale: self._set_scale(s))
            scale_layout.addWidget(btn)
        
        scale_layout.addStretch()
        
        # Auto size settings
        self.auto_frame = QWidget()
        auto_layout = QVBoxLayout(self.auto_frame)
        auto_layout.setContentsMargins(0, 0, 0, 0)
        
        # Min/Max size constraints for auto mode
        constraints_frame = QWidget()
        constraints_layout = QHBoxLayout(constraints_frame)
        constraints_layout.setContentsMargins(0, 0, 0, 0)
        auto_layout.addWidget(constraints_frame)
        
        constraints_layout.addWidget(QLabel("Size Range:"))
        
        constraints_layout.addWidget(QLabel("Min:"))
        self.min_font_size_spin = QSpinBox()
        self.min_font_size_spin.setRange(6, 20)
        self.min_font_size_spin.setValue(10)
        self.min_font_size_spin.valueChanged.connect(self._save_rendering_settings)
        constraints_layout.addWidget(self.min_font_size_spin)
        
        constraints_layout.addWidget(QLabel("Max:"))
        self.max_font_size_spin = QSpinBox()
        self.max_font_size_spin.setRange(16, 48)
        self.max_font_size_spin.setValue(28)
        self.max_font_size_spin.valueChanged.connect(self._save_rendering_settings)
        constraints_layout.addWidget(self.max_font_size_spin)
        
        constraints_layout.addStretch()
        
        # Auto fit quality
        quality_frame = QWidget()
        quality_layout = QHBoxLayout(quality_frame)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        auto_layout.addWidget(quality_frame)
        
        quality_layout.addWidget(QLabel("Fit Style:"))
        
        self.auto_fit_style_group = QButtonGroup()
        self.auto_fit_style = 'balanced'  # Store as string attribute
        
        fit_styles = [
            ("Compact", "compact", "Fit more text, smaller size"),
            ("Balanced", "balanced", "Balance readability and fit"),
            ("Readable", "readable", "Prefer larger, more readable text")
        ]
        
        for text, value, tooltip in fit_styles:
            rb = QRadioButton(text)
            rb.setChecked(value == 'balanced')
            rb.setToolTip(tooltip)
            rb.toggled.connect(lambda checked, v=value: self._on_fit_style_change(v) if checked else None)
            quality_layout.addWidget(rb)
            self.auto_fit_style_group.addButton(rb)
        
        quality_layout.addStretch()
        
        # Initialize the correct frame
        self._on_font_mode_change('auto')

    def _on_font_mode_change(self, mode):
        """Show/hide appropriate font controls based on mode"""
        # Update the stored mode
        self.font_size_mode = mode
        
        # Remove all frames from layout
        self.font_controls_layout.removeWidget(self.fixed_size_frame)
        self.font_controls_layout.removeWidget(self.scale_frame)
        self.font_controls_layout.removeWidget(self.auto_frame)
        self.fixed_size_frame.hide()
        self.scale_frame.hide()
        self.auto_frame.hide()
        
        # Show the appropriate frame
        if mode == 'fixed':
            self.font_controls_layout.addWidget(self.fixed_size_frame)
            self.fixed_size_frame.show()
        elif mode == 'scale':
            self.font_controls_layout.addWidget(self.scale_frame)
            self.scale_frame.show()
        else:  # auto
            self.font_controls_layout.addWidget(self.auto_frame)
            self.auto_frame.show()
        
        self._save_rendering_settings()

    def _set_fixed_size(self, size):
        """Set fixed font size from preset"""
        self.fixed_font_size_spin.setValue(size)
        self._save_rendering_settings()

    def _set_scale(self, scale):
        """Set font scale from preset"""
        # Scale is 0.5-2.0, slider uses 50-200
        self.font_scale_slider.setValue(int(scale * 100))
        self._update_scale_label()
        self._save_rendering_settings()

    def _update_scale_label(self):
        """Update the scale percentage label"""
        # Get value from slider (50-200) and convert to percentage
        scale_value = self.font_scale_slider.value()
        self.scale_label.setText(f"{scale_value}%")
        self._save_rendering_settings()
    
    def _on_fit_style_change(self, style):
        """Handle fit style change"""
        self.auto_fit_style = style
        self._save_rendering_settings()

    def _create_tooltip(self, widget, text):
        """Create a tooltip for a widget - PySide6 version"""
        # In PySide6, tooltips are much simpler - just set the toolTip property
        widget.setToolTip(text)
    
    def _merge_settings(self, existing: Dict) -> Dict:
        """Merge existing settings with defaults"""
        result = self.default_settings.copy()
        
        def deep_merge(base: Dict, update: Dict) -> Dict:
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        return deep_merge(result, existing)
    
    def show_dialog(self):
        """Display the settings dialog using PySide6"""
        # Set initialization flag to prevent auto-saves during setup
        self._initializing = True
        
        # Set dialog properties
        self.setWindowTitle("Manga Translation Settings")
        self.setModal(True)
        
        # Set the halgakos.ico icon
        try:
            icon_path = os.path.join(os.path.dirname(__file__), 'Halgakos.ico')
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
        except Exception:
            pass  # Fail silently if icon can't be loaded
            
        # Apply overall dark theme styling
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: white;
                font-family: Arial;
            }
            QGroupBox {
                font-family: Arial;
                font-size: 10pt;
                font-weight: bold;
                color: white;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                color: white;
                font-family: Arial;
                font-size: 9pt;
            }
            QLineEdit {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 3px;
                font-family: Arial;
                font-size: 9pt;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 3px;
                font-family: Arial;
                font-size: 9pt;
            }
            QComboBox {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 3px 5px;
                padding-right: 25px;
                font-family: Arial;
                font-size: 9pt;
                min-height: 20px;
            }
            QComboBox:hover {
                border: 1px solid #7bb3e0;
            }
            QComboBox:focus {
                border: 1px solid #5a9fd4;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 20px;
                border-left: 1px solid #555;
                background-color: #3c3c3c;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            QComboBox::drop-down:hover {
                background-color: #4a4a4a;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #aaa;
                width: 0;
                height: 0;
                margin-right: 5px;
            }
            QComboBox::down-arrow:hover {
                border-top: 5px solid #fff;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                color: white;
                selection-background-color: #5a9fd4;
                selection-color: white;
                border: 1px solid #555;
                outline: none;
            }
            QPushButton {
                font-family: Arial;
                font-size: 9pt;
                padding: 5px 15px;
                border-radius: 3px;
                border: none;
            }
            QSlider::groove:horizontal {
                border: 1px solid #555;
                height: 6px;
                background: #2d2d2d;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #5a9fd4;
                border: 1px solid #5a9fd4;
                width: 18px;
                border-radius: 9px;
                margin: -6px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #7bb3e0;
                border: 1px solid #7bb3e0;
            }
            QRadioButton {
                color: white;
                spacing: 6px;
                font-family: Arial;
                font-size: 9pt;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #5a9fd4;
                border-radius: 8px;
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
            QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
                background-color: #1a1a1a;
                color: #666666;
                border: 1px solid #3a3a3a;
            }
            QLabel:disabled {
                color: #666666;
            }
            QScrollArea {
                background-color: #1e1e1e;
                border: none;
            }
            QWidget {
                background-color: #1e1e1e;
                color: white;
            }
        """)
        
        # Calculate size based on screen dimensions
        screen = self.parent.screen() if self.parent else self.screen()
        screen_size = screen.availableGeometry()
        dialog_width = min(800, int(screen_size.width() * 0.5))
        dialog_height = min(900, int(screen_size.height() * 0.85))
        self.resize(dialog_width, dialog_height)
        
        # Center the dialog
        self.move(
            screen_size.center().x() - dialog_width // 2,
            screen_size.center().y() - dialog_height // 2
        )
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create scroll area for the content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create content widget that will go inside scroll area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create tab widget with enhanced styling
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                color: #cccccc;
                border: 1px solid #555;
                border-bottom: none;
                padding: 8px 16px;
                margin-right: 2px;
                font-family: Arial;
                font-size: 10pt;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #5a9fd4;
                color: white;
                border-color: #7bb3e0;
                margin-bottom: -1px;
            }
            QTabBar::tab:hover:!selected {
                background-color: #4a4a4a;
                color: white;
                border-color: #7bb3e0;
            }
            QTabBar::tab:first {
                margin-left: 0;
            }
        """)
        content_layout.addWidget(self.tab_widget)
        
        # Create all tabs
        self._create_preprocessing_tab()
        self._create_ocr_tab()
        self._create_inpainting_tab()
        self._create_advanced_tab()
        self._create_cloud_api_tab()
        # NOTE: Font Sizing tab removed; controls are now in Manga Integration UI
        
        # Set content widget in scroll area
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
        
        # Create button frame at bottom
        button_frame = QWidget()
        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(0, 5, 0, 0)
        
        # Reset button on left
        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self._reset_defaults)
        reset_button.setMinimumWidth(120)
        reset_button.setMinimumHeight(32)
        reset_button.setStyleSheet("""
            QPushButton {
                background-color: #ffc107;
                color: #1a1a1a;
                font-weight: bold;
                font-size: 10pt;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
            }
            QPushButton:hover {
                background-color: #ffcd38;
            }
            QPushButton:pressed {
                background-color: #e0a800;
            }
        """)
        button_layout.addWidget(reset_button)
        
        button_layout.addStretch()  # Push other buttons to the right
        
        # Cancel and Save buttons on right
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self._cancel)
        cancel_button.setMinimumWidth(90)
        cancel_button.setMinimumHeight(32)
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                font-weight: bold;
                font-size: 10pt;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
            }
            QPushButton:hover {
                background-color: #7d8a96;
            }
            QPushButton:pressed {
                background-color: #5a6268;
            }
        """)
        button_layout.addWidget(cancel_button)
        
        save_button = QPushButton("Save")
        save_button.clicked.connect(self._save_settings)
        save_button.setDefault(True)  # Make it the default button
        save_button.setMinimumWidth(90)
        save_button.setMinimumHeight(32)
        save_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                font-size: 10pt;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
            }
            QPushButton:hover {
                background-color: #34c759;
            }
            QPushButton:pressed {
                background-color: #218838;
            }
        """)
        button_layout.addWidget(save_button)
        
        main_layout.addWidget(button_frame)
        
        # Clear initialization flag after setup is complete
        self._initializing = False
        
        # Initialize preprocessing state
        self._toggle_preprocessing()
        
        # Initialize tiling controls state (must be after widgets are created)
        try:
            self._toggle_tiling_controls()
        except Exception as e:
            print(f"Warning: Failed to initialize tiling controls: {e}")
        
        # Initialize iteration controls state
        try:
            self._toggle_iteration_controls()
        except Exception:
            pass
        
        # Disable mouse wheel scrolling on all spinboxes and comboboxes
        self._disable_all_spinbox_scrolling(self)
        
        # Show the dialog
        self.show()
    
    def _create_preprocessing_tab(self):
        """Create preprocessing settings tab with all options"""
        # Create tab widget and add to tab widget
        tab_widget = QWidget()
        self.tab_widget.addTab(tab_widget, "Preprocessing")
        
        # Main scrollable content
        main_layout = QVBoxLayout(tab_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(6)
        
        # Enable preprocessing group
        enable_group = QGroupBox("Image Preprocessing")
        main_layout.addWidget(enable_group)
        enable_layout = QVBoxLayout(enable_group)
        enable_layout.setContentsMargins(8, 8, 8, 6)
        enable_layout.setSpacing(4)
        
        self.preprocess_enabled = QCheckBox("Enable Image Preprocessing")
        self.preprocess_enabled.setChecked(self.settings['preprocessing']['enabled'])
        self.preprocess_enabled.toggled.connect(self._toggle_preprocessing)
        enable_layout.addWidget(self.preprocess_enabled)
        
        # Store all preprocessing controls for enable/disable
        self.preprocessing_controls = []
        
        # Auto quality detection
        self.auto_detect = QCheckBox("Auto-detect image quality issues")
        self.auto_detect.setChecked(self.settings['preprocessing']['auto_detect_quality'])
        enable_layout.addWidget(self.auto_detect)
        self.preprocessing_controls.append(self.auto_detect)
        
        # Quality thresholds section
        threshold_group = QGroupBox("Image Enhancement")
        main_layout.addWidget(threshold_group)
        threshold_layout = QVBoxLayout(threshold_group)
        threshold_layout.setContentsMargins(8, 8, 8, 6)
        threshold_layout.setSpacing(4)
        self.preprocessing_controls.append(threshold_group)
        
        # Contrast threshold
        contrast_frame = QWidget()
        contrast_layout = QHBoxLayout(contrast_frame)
        contrast_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.addWidget(contrast_frame)
        
        contrast_label = QLabel("Contrast Adjustment:")
        contrast_label.setMinimumWidth(150)
        contrast_layout.addWidget(contrast_label)
        self.preprocessing_controls.append(contrast_label)
        
        self.contrast_threshold = QDoubleSpinBox()
        self.contrast_threshold.setRange(0.0, 1.0)
        self.contrast_threshold.setSingleStep(0.01)
        self.contrast_threshold.setDecimals(2)
        self.contrast_threshold.setValue(self.settings['preprocessing']['contrast_threshold'])
        contrast_layout.addWidget(self.contrast_threshold)
        self.preprocessing_controls.append(self.contrast_threshold)
        contrast_layout.addStretch()
        
        # Sharpness threshold
        sharpness_frame = QWidget()
        sharpness_layout = QHBoxLayout(sharpness_frame)
        sharpness_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.addWidget(sharpness_frame)
        
        sharpness_label = QLabel("Sharpness Enhancement:")
        sharpness_label.setMinimumWidth(150)
        sharpness_layout.addWidget(sharpness_label)
        self.preprocessing_controls.append(sharpness_label)
        
        self.sharpness_threshold = QDoubleSpinBox()
        self.sharpness_threshold.setRange(0.0, 1.0)
        self.sharpness_threshold.setSingleStep(0.01)
        self.sharpness_threshold.setDecimals(2)
        self.sharpness_threshold.setValue(self.settings['preprocessing']['sharpness_threshold'])
        sharpness_layout.addWidget(self.sharpness_threshold)
        self.preprocessing_controls.append(self.sharpness_threshold)
        sharpness_layout.addStretch()
        
        # Enhancement strength
        enhance_frame = QWidget()
        enhance_layout = QHBoxLayout(enhance_frame)
        enhance_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.addWidget(enhance_frame)
        
        enhance_label = QLabel("Overall Enhancement:")
        enhance_label.setMinimumWidth(150)
        enhance_layout.addWidget(enhance_label)
        self.preprocessing_controls.append(enhance_label)
        
        self.enhancement_strength = QDoubleSpinBox()
        self.enhancement_strength.setRange(0.0, 3.0)
        self.enhancement_strength.setSingleStep(0.01)
        self.enhancement_strength.setDecimals(2)
        self.enhancement_strength.setValue(self.settings['preprocessing']['enhancement_strength'])
        enhance_layout.addWidget(self.enhancement_strength)
        self.preprocessing_controls.append(self.enhancement_strength)
        enhance_layout.addStretch()
        
        # Noise reduction section
        noise_group = QGroupBox("Noise Reduction")
        main_layout.addWidget(noise_group)
        noise_layout = QVBoxLayout(noise_group)
        noise_layout.setContentsMargins(8, 8, 8, 6)
        noise_layout.setSpacing(4)
        self.preprocessing_controls.append(noise_group)
        
        # Noise threshold
        noise_threshold_frame = QWidget()
        noise_threshold_layout = QHBoxLayout(noise_threshold_frame)
        noise_threshold_layout.setContentsMargins(0, 0, 0, 0)
        noise_layout.addWidget(noise_threshold_frame)
        
        noise_label = QLabel("Noise Threshold:")
        noise_label.setMinimumWidth(150)
        noise_threshold_layout.addWidget(noise_label)
        self.preprocessing_controls.append(noise_label)
        
        self.noise_threshold = QSpinBox()
        self.noise_threshold.setRange(0, 50)
        self.noise_threshold.setValue(self.settings['preprocessing']['noise_threshold'])
        noise_threshold_layout.addWidget(self.noise_threshold)
        self.preprocessing_controls.append(self.noise_threshold)
        noise_threshold_layout.addStretch()
        
        # Denoise strength
        denoise_frame = QWidget()
        denoise_layout = QHBoxLayout(denoise_frame)
        denoise_layout.setContentsMargins(0, 0, 0, 0)
        noise_layout.addWidget(denoise_frame)
        
        denoise_label = QLabel("Denoise Strength:")
        denoise_label.setMinimumWidth(150)
        denoise_layout.addWidget(denoise_label)
        self.preprocessing_controls.append(denoise_label)
        
        self.denoise_strength = QSpinBox()
        self.denoise_strength.setRange(0, 30)
        self.denoise_strength.setValue(self.settings['preprocessing']['denoise_strength'])
        denoise_layout.addWidget(self.denoise_strength)
        self.preprocessing_controls.append(self.denoise_strength)
        denoise_layout.addStretch()
        
        # Size limits section
        size_group = QGroupBox("Image Size Limits")
        main_layout.addWidget(size_group)
        size_layout = QVBoxLayout(size_group)
        size_layout.setContentsMargins(8, 8, 8, 6)
        size_layout.setSpacing(4)
        self.preprocessing_controls.append(size_group)
        
        # Max dimension
        dimension_frame = QWidget()
        dimension_layout = QHBoxLayout(dimension_frame)
        dimension_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.addWidget(dimension_frame)
        
        dimension_label = QLabel("Max Dimension:")
        dimension_label.setMinimumWidth(150)
        dimension_layout.addWidget(dimension_label)
        self.preprocessing_controls.append(dimension_label)
        
        self.dimension_spinbox = QSpinBox()
        self.dimension_spinbox.setRange(500, 4000)
        self.dimension_spinbox.setSingleStep(100)
        self.dimension_spinbox.setValue(self.settings['preprocessing']['max_image_dimension'])
        dimension_layout.addWidget(self.dimension_spinbox)
        self.preprocessing_controls.append(self.dimension_spinbox)
        
        dimension_layout.addWidget(QLabel("pixels"))
        dimension_layout.addStretch()
        
        # Max pixels
        pixels_frame = QWidget()
        pixels_layout = QHBoxLayout(pixels_frame)
        pixels_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.addWidget(pixels_frame)
        
        pixels_label = QLabel("Max Total Pixels:")
        pixels_label.setMinimumWidth(150)
        pixels_layout.addWidget(pixels_label)
        self.preprocessing_controls.append(pixels_label)
        
        self.pixels_spinbox = QSpinBox()
        self.pixels_spinbox.setRange(1000000, 10000000)
        self.pixels_spinbox.setSingleStep(100000)
        self.pixels_spinbox.setValue(self.settings['preprocessing']['max_image_pixels'])
        pixels_layout.addWidget(self.pixels_spinbox)
        self.preprocessing_controls.append(self.pixels_spinbox)
        
        pixels_layout.addWidget(QLabel("pixels"))
        pixels_layout.addStretch()
        
        # Compression section
        compression_group = QGroupBox("Image Compression (applies to OCR uploads)")
        main_layout.addWidget(compression_group)
        compression_layout = QVBoxLayout(compression_group)
        compression_layout.setContentsMargins(8, 8, 8, 6)
        compression_layout.setSpacing(4)
        # Do NOT add compression controls to preprocessing_controls; keep independent of preprocessing toggle
        
        # Enable compression toggle
        self.compression_enabled = QCheckBox("Enable compression for OCR uploads")
        self.compression_enabled.setChecked(self.settings.get('compression', {}).get('enabled', False))
        self.compression_enabled.toggled.connect(self._toggle_compression_enabled)
        compression_layout.addWidget(self.compression_enabled)
        
        # Format selection
        format_frame = QWidget()
        format_layout = QHBoxLayout(format_frame)
        format_layout.setContentsMargins(0, 0, 0, 0)
        compression_layout.addWidget(format_frame)
        
        self.format_label = QLabel("Format:")
        self.format_label.setMinimumWidth(150)
        format_layout.addWidget(self.format_label)
        
        self.compression_format_combo = QComboBox()
        self.compression_format_combo.addItems(['jpeg', 'png', 'webp'])
        self.compression_format_combo.setCurrentText(self.settings.get('compression', {}).get('format', 'jpeg'))
        self.compression_format_combo.currentTextChanged.connect(self._toggle_compression_format)
        format_layout.addWidget(self.compression_format_combo)
        format_layout.addStretch()
        
        # JPEG quality
        self.jpeg_frame = QWidget()
        jpeg_layout = QHBoxLayout(self.jpeg_frame)
        jpeg_layout.setContentsMargins(0, 0, 0, 0)
        compression_layout.addWidget(self.jpeg_frame)
        
        self.jpeg_label = QLabel("JPEG Quality:")
        self.jpeg_label.setMinimumWidth(150)
        jpeg_layout.addWidget(self.jpeg_label)
        
        self.jpeg_quality_spin = QSpinBox()
        self.jpeg_quality_spin.setRange(1, 95)
        self.jpeg_quality_spin.setValue(self.settings.get('compression', {}).get('jpeg_quality', 85))
        jpeg_layout.addWidget(self.jpeg_quality_spin)
        
        self.jpeg_help = QLabel("(higher = better quality, larger size)")
        self.jpeg_help.setStyleSheet("color: gray; font-size: 9pt;")
        jpeg_layout.addWidget(self.jpeg_help)
        jpeg_layout.addStretch()
        
        # PNG compression level
        self.png_frame = QWidget()
        png_layout = QHBoxLayout(self.png_frame)
        png_layout.setContentsMargins(0, 0, 0, 0)
        compression_layout.addWidget(self.png_frame)
        
        self.png_label = QLabel("PNG Compression:")
        self.png_label.setMinimumWidth(150)
        png_layout.addWidget(self.png_label)
        
        self.png_level_spin = QSpinBox()
        self.png_level_spin.setRange(0, 9)
        self.png_level_spin.setValue(self.settings.get('compression', {}).get('png_compress_level', 6))
        png_layout.addWidget(self.png_level_spin)
        
        self.png_help = QLabel("(0 = fastest, 9 = smallest)")
        self.png_help.setStyleSheet("color: gray; font-size: 9pt;")
        png_layout.addWidget(self.png_help)
        png_layout.addStretch()
        
        # WEBP quality
        self.webp_frame = QWidget()
        webp_layout = QHBoxLayout(self.webp_frame)
        webp_layout.setContentsMargins(0, 0, 0, 0)
        compression_layout.addWidget(self.webp_frame)
        
        self.webp_label = QLabel("WEBP Quality:")
        self.webp_label.setMinimumWidth(150)
        webp_layout.addWidget(self.webp_label)
        
        self.webp_quality_spin = QSpinBox()
        self.webp_quality_spin.setRange(1, 100)
        self.webp_quality_spin.setValue(self.settings.get('compression', {}).get('webp_quality', 85))
        webp_layout.addWidget(self.webp_quality_spin)
        
        self.webp_help = QLabel("(higher = better quality, larger size)")
        self.webp_help.setStyleSheet("color: gray; font-size: 9pt;")
        webp_layout.addWidget(self.webp_help)
        webp_layout.addStretch()
        
        # Initialize format-specific visibility and enabled state
        self._toggle_compression_format()
        self._toggle_compression_enabled()
        
        # HD Strategy (Inpainting acceleration) - Independent of preprocessing toggle
        hd_group = QGroupBox("Inpainting HD Strategy")
        main_layout.addWidget(hd_group)
        hd_layout = QVBoxLayout(hd_group)
        hd_layout.setContentsMargins(8, 8, 8, 6)
        hd_layout.setSpacing(4)
        # Do NOT add to preprocessing_controls - HD Strategy should be independent
        
        # Chunk settings for large images - Independent of preprocessing toggle
        chunk_group = QGroupBox("Large Image Processing")
        main_layout.addWidget(chunk_group)
        chunk_layout = QVBoxLayout(chunk_group)
        chunk_layout.setContentsMargins(8, 8, 8, 6)
        chunk_layout.setSpacing(4)
        # Do NOT add to preprocessing_controls - Large Image Processing should be independent
        
        # Strategy selector
        strat_frame = QWidget()
        strat_layout = QHBoxLayout(strat_frame)
        strat_layout.setContentsMargins(0, 0, 0, 0)
        hd_layout.addWidget(strat_frame)
        
        strat_label = QLabel("Strategy:")
        strat_label.setMinimumWidth(150)
        strat_layout.addWidget(strat_label)
        
        self.hd_strategy_combo = QComboBox()
        self.hd_strategy_combo.addItems(['original', 'resize', 'crop'])
        self.hd_strategy_combo.setCurrentText(self.settings.get('advanced', {}).get('hd_strategy', 'resize'))
        self.hd_strategy_combo.currentTextChanged.connect(self._on_hd_strategy_change)
        strat_layout.addWidget(self.hd_strategy_combo)
        
        strat_help = QLabel("(original = legacy full-image; resize/crop = faster)")
        strat_help.setStyleSheet("color: gray; font-size: 9pt;")
        strat_layout.addWidget(strat_help)
        strat_layout.addStretch()
        
        # Resize limit row
        self.hd_resize_frame = QWidget()
        resize_layout = QHBoxLayout(self.hd_resize_frame)
        resize_layout.setContentsMargins(0, 0, 0, 0)
        hd_layout.addWidget(self.hd_resize_frame)
        
        resize_label = QLabel("Resize limit (long edge):")
        resize_label.setMinimumWidth(150)
        resize_layout.addWidget(resize_label)
        
        self.hd_resize_limit_spin = QSpinBox()
        self.hd_resize_limit_spin.setRange(512, 4096)
        self.hd_resize_limit_spin.setSingleStep(64)
        self.hd_resize_limit_spin.setValue(int(self.settings.get('advanced', {}).get('hd_strategy_resize_limit', 1536)))
        resize_layout.addWidget(self.hd_resize_limit_spin)
        
        resize_layout.addWidget(QLabel("px"))
        resize_layout.addStretch()
        
        # Crop params rows
        self.hd_crop_margin_frame = QWidget()
        margin_layout = QHBoxLayout(self.hd_crop_margin_frame)
        margin_layout.setContentsMargins(0, 0, 0, 0)
        hd_layout.addWidget(self.hd_crop_margin_frame)
        
        margin_label = QLabel("Crop margin:")
        margin_label.setMinimumWidth(150)
        margin_layout.addWidget(margin_label)
        
        self.hd_crop_margin_spin = QSpinBox()
        self.hd_crop_margin_spin.setRange(0, 256)
        self.hd_crop_margin_spin.setSingleStep(2)
        self.hd_crop_margin_spin.setValue(int(self.settings.get('advanced', {}).get('hd_strategy_crop_margin', 16)))
        margin_layout.addWidget(self.hd_crop_margin_spin)
        
        margin_layout.addWidget(QLabel("px"))
        margin_layout.addStretch()
        
        self.hd_crop_trigger_frame = QWidget()
        trigger_layout = QHBoxLayout(self.hd_crop_trigger_frame)
        trigger_layout.setContentsMargins(0, 0, 0, 0)
        hd_layout.addWidget(self.hd_crop_trigger_frame)
        
        trigger_label = QLabel("Crop trigger size:")
        trigger_label.setMinimumWidth(150)
        trigger_layout.addWidget(trigger_label)
        
        self.hd_crop_trigger_spin = QSpinBox()
        self.hd_crop_trigger_spin.setRange(256, 4096)
        self.hd_crop_trigger_spin.setSingleStep(64)
        self.hd_crop_trigger_spin.setValue(int(self.settings.get('advanced', {}).get('hd_strategy_crop_trigger_size', 1024)))
        trigger_layout.addWidget(self.hd_crop_trigger_spin)
        
        trigger_help = QLabel("px (apply crop only if long edge > trigger)")
        trigger_layout.addWidget(trigger_help)
        trigger_layout.addStretch()
        
        # Initialize strategy-specific visibility
        self._on_hd_strategy_change()
        
        # Clarifying note about precedence with tiling
        note_label = QLabel(
            "Note: HD Strategy (resize/crop) takes precedence over Inpainting Tiling when it triggers.\n"
            "Set strategy to 'original' if you want tiling to control large-image behavior."
        )
        note_label.setStyleSheet("color: gray; font-size: 9pt;")
        note_label.setWordWrap(True)
        hd_layout.addWidget(note_label)
        
        # Chunk height
        chunk_height_frame = QWidget()
        chunk_height_layout = QHBoxLayout(chunk_height_frame)
        chunk_height_layout.setContentsMargins(0, 0, 0, 0)
        chunk_layout.addWidget(chunk_height_frame)
        
        chunk_height_label = QLabel("Chunk Height:")
        chunk_height_label.setMinimumWidth(150)
        chunk_height_layout.addWidget(chunk_height_label)
        # Do NOT add to preprocessing_controls - chunk settings should be independent
        
        self.chunk_height_spinbox = QSpinBox()
        self.chunk_height_spinbox.setRange(500, 2000)
        self.chunk_height_spinbox.setSingleStep(100)
        self.chunk_height_spinbox.setValue(self.settings['preprocessing']['chunk_height'])
        chunk_height_layout.addWidget(self.chunk_height_spinbox)
        # Do NOT add to preprocessing_controls - chunk settings should be independent
        
        chunk_height_layout.addWidget(QLabel("pixels"))
        chunk_height_layout.addStretch()
        
        # Chunk overlap
        chunk_overlap_frame = QWidget()
        chunk_overlap_layout = QHBoxLayout(chunk_overlap_frame)
        chunk_overlap_layout.setContentsMargins(0, 0, 0, 0)
        chunk_layout.addWidget(chunk_overlap_frame)
        
        chunk_overlap_label = QLabel("Chunk Overlap:")
        chunk_overlap_label.setMinimumWidth(150)
        chunk_overlap_layout.addWidget(chunk_overlap_label)
        # Do NOT add to preprocessing_controls - chunk settings should be independent
        
        self.chunk_overlap_spinbox = QSpinBox()
        self.chunk_overlap_spinbox.setRange(0, 200)
        self.chunk_overlap_spinbox.setSingleStep(10)
        self.chunk_overlap_spinbox.setValue(self.settings['preprocessing']['chunk_overlap'])
        chunk_overlap_layout.addWidget(self.chunk_overlap_spinbox)
        # Do NOT add to preprocessing_controls - chunk settings should be independent
        
        chunk_overlap_layout.addWidget(QLabel("pixels"))
        chunk_overlap_layout.addStretch()

        # Inpainting Tiling section
        tiling_group = QGroupBox("Inpainting Tiling")
        main_layout.addWidget(tiling_group)
        tiling_layout = QVBoxLayout(tiling_group)
        tiling_layout.setContentsMargins(8, 8, 8, 6)
        tiling_layout.setSpacing(4)
        # Do NOT add to preprocessing_controls - tiling should be independent

        # Enable tiling
        # Prefer values from legacy 'tiling' section if present, otherwise use 'preprocessing'
        tiling_enabled_value = self.settings['preprocessing'].get('inpaint_tiling_enabled', False)
        if 'tiling' in self.settings and isinstance(self.settings['tiling'], dict) and 'enabled' in self.settings['tiling']:
            tiling_enabled_value = self.settings['tiling']['enabled']
            
        self.inpaint_tiling_enabled = QCheckBox("Enable automatic tiling for inpainting (processes large images in tiles)")
        self.inpaint_tiling_enabled.setChecked(tiling_enabled_value)
        self.inpaint_tiling_enabled.toggled.connect(self._toggle_tiling_controls)
        tiling_layout.addWidget(self.inpaint_tiling_enabled)

        # Tile size
        tile_size_frame = QWidget()
        tile_size_layout = QHBoxLayout(tile_size_frame)
        tile_size_layout.setContentsMargins(0, 0, 0, 0)
        tiling_layout.addWidget(tile_size_frame)
        
        self.tile_size_label = QLabel("Tile Size:")
        self.tile_size_label.setMinimumWidth(150)
        tile_size_layout.addWidget(self.tile_size_label)

        tile_size_value = self.settings['preprocessing'].get('inpaint_tile_size', 512)
        if 'tiling' in self.settings and isinstance(self.settings['tiling'], dict) and 'tile_size' in self.settings['tiling']:
            tile_size_value = self.settings['tiling']['tile_size']
            
        self.tile_size_spinbox = QSpinBox()
        self.tile_size_spinbox.setRange(256, 2048)
        self.tile_size_spinbox.setSingleStep(128)
        self.tile_size_spinbox.setValue(tile_size_value)
        tile_size_layout.addWidget(self.tile_size_spinbox)

        self.tile_size_unit_label = QLabel("pixels")
        tile_size_layout.addWidget(self.tile_size_unit_label)
        tile_size_layout.addStretch()

        # Tile overlap
        tile_overlap_frame = QWidget()
        tile_overlap_layout = QHBoxLayout(tile_overlap_frame)
        tile_overlap_layout.setContentsMargins(0, 0, 0, 0)
        tiling_layout.addWidget(tile_overlap_frame)
        
        self.tile_overlap_label = QLabel("Tile Overlap:")
        self.tile_overlap_label.setMinimumWidth(150)
        tile_overlap_layout.addWidget(self.tile_overlap_label)

        tile_overlap_value = self.settings['preprocessing'].get('inpaint_tile_overlap', 64)
        if 'tiling' in self.settings and isinstance(self.settings['tiling'], dict) and 'tile_overlap' in self.settings['tiling']:
            tile_overlap_value = self.settings['tiling']['tile_overlap']
            
        self.tile_overlap_spinbox = QSpinBox()
        self.tile_overlap_spinbox.setRange(0, 256)
        self.tile_overlap_spinbox.setSingleStep(16)
        self.tile_overlap_spinbox.setValue(tile_overlap_value)
        tile_overlap_layout.addWidget(self.tile_overlap_spinbox)

        self.tile_overlap_unit_label = QLabel("pixels")
        tile_overlap_layout.addWidget(self.tile_overlap_unit_label)
        tile_overlap_layout.addStretch()
        
        # Don't initialize here - will be done after dialog is shown
        
        # ELIMINATE ALL EMPTY SPACE - Add stretch at the end
        main_layout.addStretch()

    def _create_inpainting_tab(self):
        """Create inpainting settings tab with comprehensive per-text-type dilation controls"""
        # Create tab widget and add to tab widget
        tab_widget = QWidget()
        self.tab_widget.addTab(tab_widget, "Inpainting")
        
        # Main scrollable content
        main_layout = QVBoxLayout(tab_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(6)
        
        # General Mask Settings (applies to all inpainting methods)
        mask_group = QGroupBox("Mask Settings")
        main_layout.addWidget(mask_group)
        mask_layout = QVBoxLayout(mask_group)
        mask_layout.setContentsMargins(8, 8, 8, 6)
        mask_layout.setSpacing(4)
        
        # Auto toggle (affects both mask dilation and iterations)
        if not hasattr(self, 'auto_iterations_enabled'):
            self.auto_iterations_enabled = QCheckBox("Auto (affects mask dilation and iterations)")
            self.auto_iterations_enabled.setChecked(self.settings.get('auto_iterations', True))
            self.auto_iterations_enabled.toggled.connect(self._toggle_iteration_controls)
        mask_layout.addWidget(self.auto_iterations_enabled)

        # Mask Dilation frame (affected by auto setting)
        mask_dilation_group = QGroupBox("Mask Dilation")
        mask_layout.addWidget(mask_dilation_group)
        mask_dilation_layout = QVBoxLayout(mask_dilation_group)
        mask_dilation_layout.setContentsMargins(8, 8, 8, 6)
        mask_dilation_layout.setSpacing(4)
        
        # Note about dilation importance
        note_label = QLabel(
            "Mask dilation is critical for avoiding white spots in final images.\n"
            "Adjust per text type for optimal results."
        )
        note_label.setStyleSheet("color: gray; font-style: italic;")
        note_label.setWordWrap(True)
        mask_dilation_layout.addWidget(note_label)
        
        # Keep all three dilation controls in a list for easy access
        if not hasattr(self, 'mask_dilation_controls'):
            self.mask_dilation_controls = []
        
        # Mask dilation size
        dilation_frame = QWidget()
        dilation_layout = QHBoxLayout(dilation_frame)
        dilation_layout.setContentsMargins(0, 0, 0, 0)
        mask_dilation_layout.addWidget(dilation_frame)
        
        self.dilation_label = QLabel("Mask Dilation:")
        self.dilation_label.setMinimumWidth(150)
        dilation_layout.addWidget(self.dilation_label)
        
        self.mask_dilation_spinbox = QSpinBox()
        self.mask_dilation_spinbox.setRange(0, 50)
        self.mask_dilation_spinbox.setSingleStep(5)
        self.mask_dilation_spinbox.setValue(self.settings.get('mask_dilation', 15))
        dilation_layout.addWidget(self.mask_dilation_spinbox)
        
        self.dilation_unit_label = QLabel("pixels (expand mask beyond text)")
        dilation_layout.addWidget(self.dilation_unit_label)
        dilation_layout.addStretch()
        
        # Per-Text-Type Iterations - EXPANDED SECTION
        iterations_group = QGroupBox("Dilation Iterations Control")
        iterations_layout = QVBoxLayout(iterations_group)
        mask_dilation_layout.addWidget(iterations_group)
        
        # All Iterations Master Control (NEW)
        all_iter_widget = QWidget()
        all_iter_layout = QHBoxLayout(all_iter_widget)
        all_iter_layout.setContentsMargins(0, 0, 0, 0)
        iterations_layout.addWidget(all_iter_widget)
        
        # Auto-iterations toggle (secondary control reflects the same setting)
        if not hasattr(self, 'auto_iterations_enabled'):
            self.auto_iterations_enabled = self.settings.get('auto_iterations', True)
        self.auto_iter_secondary_checkbox = QCheckBox("Auto (set by image: B&W vs Color)")
        self.auto_iter_secondary_checkbox.setChecked(self.auto_iterations_enabled)
        self.auto_iter_secondary_checkbox.stateChanged.connect(self._toggle_iteration_controls)
        all_iter_layout.addWidget(self.auto_iter_secondary_checkbox)
        
        all_iter_layout.addSpacing(10)
        
        # Checkbox to enable/disable uniform iterations
        self.use_all_iterations_checkbox = QCheckBox("Use Same For All:")
        self.use_all_iterations_checkbox.setChecked(self.settings.get('use_all_iterations', True))
        self.use_all_iterations_checkbox.stateChanged.connect(self._toggle_iteration_controls)
        all_iter_layout.addWidget(self.use_all_iterations_checkbox)
        
        all_iter_layout.addSpacing(10)
        
        self.all_iterations_spinbox = QSpinBox()
        self.all_iterations_spinbox.setRange(0, 5)
        self.all_iterations_spinbox.setValue(self.settings.get('all_iterations', 2))
        self.all_iterations_spinbox.setEnabled(self.use_all_iterations_checkbox.isChecked())
        all_iter_layout.addWidget(self.all_iterations_spinbox)
        
        all_iter_label = QLabel("iterations (applies to all text types)")
        all_iter_layout.addWidget(all_iter_label)
        all_iter_layout.addStretch()
        
        # Separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setFrameShadow(QFrame.Shadow.Sunken)
        iterations_layout.addWidget(separator1)
        
        # Individual Controls Label
        individual_label = QLabel("Individual Text Type Controls:")
        individual_label_font = QFont('Arial', 9)
        individual_label_font.setBold(True)
        individual_label.setFont(individual_label_font)
        iterations_layout.addWidget(individual_label)
        
        # Text Bubble iterations (modified from original bubble iterations)
        text_bubble_iter_widget = QWidget()
        text_bubble_iter_layout = QHBoxLayout(text_bubble_iter_widget)
        text_bubble_iter_layout.setContentsMargins(0, 0, 0, 0)
        iterations_layout.addWidget(text_bubble_iter_widget)
        
        self.text_bubble_label = QLabel("Text Bubbles:")
        self.text_bubble_label.setMinimumWidth(120)
        text_bubble_iter_layout.addWidget(self.text_bubble_label)
        
        self.text_bubble_iter_spinbox = QSpinBox()
        self.text_bubble_iter_spinbox.setRange(0, 5)
        self.text_bubble_iter_spinbox.setValue(self.settings.get('text_bubble_dilation_iterations', 
                                                                  self.settings.get('bubble_dilation_iterations', 2)))
        text_bubble_iter_layout.addWidget(self.text_bubble_iter_spinbox)
        
        text_bubble_desc = QLabel("iterations (speech/dialogue bubbles)")
        text_bubble_iter_layout.addWidget(text_bubble_desc)
        text_bubble_iter_layout.addStretch()
        
        # Empty Bubble iterations (NEW)
        empty_bubble_iter_widget = QWidget()
        empty_bubble_iter_layout = QHBoxLayout(empty_bubble_iter_widget)
        empty_bubble_iter_layout.setContentsMargins(0, 0, 0, 0)
        iterations_layout.addWidget(empty_bubble_iter_widget)
        
        self.empty_bubble_label = QLabel("Empty Bubbles:")
        self.empty_bubble_label.setMinimumWidth(120)
        empty_bubble_iter_layout.addWidget(self.empty_bubble_label)
        
        self.empty_bubble_iter_spinbox = QSpinBox()
        self.empty_bubble_iter_spinbox.setRange(0, 5)
        self.empty_bubble_iter_spinbox.setValue(self.settings.get('empty_bubble_dilation_iterations', 3))
        empty_bubble_iter_layout.addWidget(self.empty_bubble_iter_spinbox)
        
        empty_bubble_desc = QLabel("iterations (empty speech bubbles)")
        empty_bubble_iter_layout.addWidget(empty_bubble_desc)
        empty_bubble_iter_layout.addStretch()
        
        # Free text iterations
        free_text_iter_widget = QWidget()
        free_text_iter_layout = QHBoxLayout(free_text_iter_widget)
        free_text_iter_layout.setContentsMargins(0, 0, 0, 0)
        iterations_layout.addWidget(free_text_iter_widget)
        
        self.free_text_label = QLabel("Free Text:")
        self.free_text_label.setMinimumWidth(120)
        free_text_iter_layout.addWidget(self.free_text_label)
        
        self.free_text_iter_spinbox = QSpinBox()
        self.free_text_iter_spinbox.setRange(0, 5)
        self.free_text_iter_spinbox.setValue(self.settings.get('free_text_dilation_iterations', 0))
        free_text_iter_layout.addWidget(self.free_text_iter_spinbox)
        
        free_text_desc = QLabel("iterations (0 = perfect for B&W panels)")
        free_text_iter_layout.addWidget(free_text_desc)
        free_text_iter_layout.addStretch()
        
        # Store individual control widgets for enable/disable
        self.individual_iteration_controls = [
            (self.text_bubble_label, self.text_bubble_iter_spinbox),
            (self.empty_bubble_label, self.empty_bubble_iter_spinbox),
            (self.free_text_label, self.free_text_iter_spinbox)
        ]
        
        # Apply initial state
        self._toggle_iteration_controls()
        
        # Quick presets - UPDATED VERSION
        preset_widget = QWidget()
        preset_layout = QHBoxLayout(preset_widget)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        mask_dilation_layout.addWidget(preset_widget)
        
        preset_label = QLabel("Quick Presets:")
        preset_layout.addWidget(preset_label)
        preset_layout.addSpacing(10)
        
        bw_manga_btn = QPushButton("B&W Manga")
        bw_manga_btn.clicked.connect(lambda: self._set_mask_preset(15, False, 2, 2, 3, 0))
        preset_layout.addWidget(bw_manga_btn)
        
        colored_btn = QPushButton("Colored")
        colored_btn.clicked.connect(lambda: self._set_mask_preset(15, False, 2, 2, 3, 3))
        preset_layout.addWidget(colored_btn)
        
        uniform_btn = QPushButton("Uniform")
        uniform_btn.clicked.connect(lambda: self._set_mask_preset(0, True, 2, 2, 2, 0))
        preset_layout.addWidget(uniform_btn)
        
        preset_layout.addStretch()
        
        # Help text - UPDATED
        help_text = QLabel(
            "💡 B&W Manga: Optimized for black & white panels with clean bubbles\n"
            "💡 Colored: For colored manga with complex backgrounds\n"
            "💡 Aggressive: For difficult text removal cases\n"
            "💡 Uniform: Good for Manga-OCR\n"
            "ℹ️ Empty bubbles often need more iterations than text bubbles\n"
            "ℹ️ Set Free Text to 0 for crisp B&W panels without bleeding"
        )
        help_text_font = QFont('Arial', 9)
        help_text.setFont(help_text_font)
        help_text.setStyleSheet("color: gray;")
        help_text.setWordWrap(True)
        mask_dilation_layout.addWidget(help_text)
        
        content_layout.addStretch()
        
        # Note about method selection
        info_widget = QWidget()
        info_layout = QHBoxLayout(info_widget)
        info_layout.setContentsMargins(20, 0, 20, 0)
        content_layout.addWidget(info_widget)
        
        info_label = QLabel(
            "ℹ️ Note: Inpainting method (Cloud/Local) and model selection are configured\n"
            "     in the Manga tab when you select images for translation."
        )
        info_font = QFont('Arial', 10)
        info_label.setFont(info_font)
        info_label.setStyleSheet("color: #4a9eff;")
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        info_layout.addStretch()

    def _toggle_iteration_controls(self):
        """Enable/disable iteration controls based on Auto and 'Use Same For All' toggles"""
        # Get auto checkbox state
        auto_on = False
        if hasattr(self, 'auto_iterations_checkbox'):
            auto_on = self.auto_iterations_checkbox.isChecked()
        elif hasattr(self, 'auto_iter_secondary_checkbox'):
            auto_on = self.auto_iter_secondary_checkbox.isChecked()
        
        # Get use_all checkbox state
        use_all = False
        if hasattr(self, 'use_all_iterations_checkbox'):
            use_all = self.use_all_iterations_checkbox.isChecked()
        
        # Also update the auto_iterations_enabled attribute
        self.auto_iterations_enabled = auto_on
        
        if auto_on:
            # Disable everything when auto is on
            try:
                self.all_iterations_spinbox.setEnabled(False)
            except Exception:
                pass
            try:
                if hasattr(self, 'use_all_iterations_checkbox'):
                    self.use_all_iterations_checkbox.setEnabled(False)
            except Exception:
                pass
            try:
                if hasattr(self, 'mask_dilation_spinbox'):
                    self.mask_dilation_spinbox.setEnabled(False)
            except Exception:
                pass
            for label, spinbox in getattr(self, 'individual_iteration_controls', []):
                try:
                    spinbox.setEnabled(False)
                    label.setStyleSheet("color: gray;")
                except Exception:
                    pass
            return
        
        # Auto off -> respect 'use all'
        try:
            self.all_iterations_spinbox.setEnabled(use_all)
        except Exception:
            pass
        try:
            if hasattr(self, 'use_all_iterations_checkbox'):
                self.use_all_iterations_checkbox.setEnabled(True)
        except Exception:
            pass
        try:
            if hasattr(self, 'mask_dilation_spinbox'):
                self.mask_dilation_spinbox.setEnabled(True)
        except Exception:
            pass
        for label, spinbox in getattr(self, 'individual_iteration_controls', []):
            enabled = not use_all
            try:
                spinbox.setEnabled(enabled)
                label.setStyleSheet("color: gray;" if use_all else "")
            except Exception:
                pass

    def _set_mask_preset(self, dilation, use_all, all_iter, text_bubble_iter, empty_bubble_iter, free_text_iter):
        """Set mask dilation preset values with comprehensive iteration controls"""
        self.mask_dilation_spinbox.setValue(dilation)
        self.use_all_iterations_checkbox.setChecked(use_all)
        self.all_iterations_spinbox.setValue(all_iter)
        self.text_bubble_iter_spinbox.setValue(text_bubble_iter)
        self.empty_bubble_iter_spinbox.setValue(empty_bubble_iter)
        self.free_text_iter_spinbox.setValue(free_text_iter)
        self._toggle_iteration_controls()
    
    def _create_cloud_api_tab(self, parent):
            """Create cloud API settings tab"""
            # Create scroll area for content
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setFrameShape(QFrame.Shape.NoFrame)
            
            content_widget = QWidget()
            content_layout = QVBoxLayout(content_widget)
            content_layout.setSpacing(10)
            content_layout.setContentsMargins(20, 20, 20, 20)
            
            scroll_area.setWidget(content_widget)
            
            # Add scroll area to parent layout
            parent_layout = QVBoxLayout(parent)
            parent_layout.setContentsMargins(0, 0, 0, 0)
            parent_layout.addWidget(scroll_area)
            
            # API Model Selection
            model_group = QGroupBox("Inpainting Model")
            model_layout = QVBoxLayout(model_group)
            content_layout.addWidget(model_group)
            
            model_desc = QLabel("Select the Replicate model to use for inpainting:")
            model_layout.addWidget(model_desc)
            model_layout.addSpacing(10)
            
            # Model options - use button group for radio buttons
            self.cloud_model_button_group = QButtonGroup()
            self.cloud_model_selected = self.settings.get('cloud_inpaint_model', 'ideogram-v2')
            
            models = [
                ('ideogram-v2', 'Ideogram V2 (Best quality, with prompts)', 'ideogram-ai/ideogram-v2'),
                ('sd-inpainting', 'Stable Diffusion Inpainting (Classic, fast)', 'stability-ai/stable-diffusion-inpainting'),
                ('flux-inpainting', 'FLUX Dev Inpainting (High quality)', 'zsxkib/flux-dev-inpainting'),
                ('custom', 'Custom Model (Enter model identifier)', '')
            ]
            
            for value, text, model_id in models:
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                model_layout.addWidget(row_widget)
                
                rb = QRadioButton(text)
                rb.setChecked(value == self.cloud_model_selected)
                rb.toggled.connect(lambda checked, v=value: self._on_cloud_model_change(v) if checked else None)
                self.cloud_model_button_group.addButton(rb)
                row_layout.addWidget(rb)
                
                if model_id:
                    model_id_label = QLabel(f"({model_id})")
                    model_id_font = QFont('Arial', 8)
                    model_id_label.setFont(model_id_font)
                    model_id_label.setStyleSheet("color: gray;")
                    row_layout.addWidget(model_id_label)
                
                row_layout.addStretch()
            
            # Custom version ID (now model identifier)
            self.custom_version_widget = QWidget()
            custom_version_layout = QVBoxLayout(self.custom_version_widget)
            custom_version_layout.setContentsMargins(0, 10, 0, 0)
            model_layout.addWidget(self.custom_version_widget)
            
            custom_id_row = QWidget()
            custom_id_layout = QHBoxLayout(custom_id_row)
            custom_id_layout.setContentsMargins(0, 0, 0, 0)
            custom_version_layout.addWidget(custom_id_row)
            
            custom_id_label = QLabel("Model ID:")
            custom_id_label.setMinimumWidth(120)
            custom_id_layout.addWidget(custom_id_label)
            
            self.custom_version_entry = QLineEdit()
            self.custom_version_entry.setText(self.settings.get('cloud_custom_version', ''))
            custom_id_layout.addWidget(self.custom_version_entry)
            
            # Add helper text for custom model
            helper_text = QLabel("Format: owner/model-name (e.g. stability-ai/stable-diffusion-inpainting)")
            helper_font = QFont('Arial', 8)
            helper_text.setFont(helper_font)
            helper_text.setStyleSheet("color: gray;")
            helper_text.setContentsMargins(120, 0, 0, 0)
            custom_version_layout.addWidget(helper_text)
            
            # Initially hide custom version entry
            if self.cloud_model_selected != 'custom':
                self.custom_version_widget.setVisible(False)
            
            # Performance Settings
            perf_group = QGroupBox("Performance Settings")
            perf_layout = QVBoxLayout(perf_group)
            content_layout.addWidget(perf_group)
    
            # Timeout
            timeout_widget = QWidget()
            timeout_layout = QHBoxLayout(timeout_widget)
            timeout_layout.setContentsMargins(0, 0, 0, 0)
            perf_layout.addWidget(timeout_widget)
            
            timeout_label = QLabel("API Timeout:")
            timeout_label.setMinimumWidth(120)
            timeout_layout.addWidget(timeout_label)
            
            self.cloud_timeout_spinbox = QSpinBox()
            self.cloud_timeout_spinbox.setRange(30, 300)
            self.cloud_timeout_spinbox.setValue(self.settings.get('cloud_timeout', 60))
            timeout_layout.addWidget(self.cloud_timeout_spinbox)
            
            timeout_unit = QLabel("seconds")
            timeout_unit_font = QFont('Arial', 9)
            timeout_unit.setFont(timeout_unit_font)
            timeout_layout.addWidget(timeout_unit)
            timeout_layout.addStretch()
            
            # Help text
            help_text = QLabel(
                "💡 Tips:\n"
                "• Ideogram V2 is currently the best quality option\n"
                "• SD inpainting is fast and supports prompts\n"
                "• FLUX inpainting offers high quality results\n"
                "• Find more models at replicate.com/collections/inpainting"
            )
            help_font = QFont('Arial', 9)
            help_text.setFont(help_font)
            help_text.setStyleSheet("color: gray;")
            help_text.setWordWrap(True)
            content_layout.addWidget(help_text)
            
            # Prompt Settings (for all models except custom)
            self.prompt_group = QGroupBox("Prompt Settings")
            prompt_layout = QVBoxLayout(self.prompt_group)
            content_layout.addWidget(self.prompt_group)
            
            # Positive prompt
            prompt_label = QLabel("Inpainting Prompt:")
            prompt_layout.addWidget(prompt_label)
            
            self.cloud_prompt_entry = QLineEdit()
            self.cloud_prompt_entry.setText(self.settings.get('cloud_inpaint_prompt', 'clean background, smooth surface'))
            prompt_layout.addWidget(self.cloud_prompt_entry)
            
            # Add note about prompts
            prompt_tip = QLabel("Tip: Describe what you want in the inpainted area (e.g., 'white wall', 'wooden floor')")
            prompt_tip_font = QFont('Arial', 8)
            prompt_tip.setFont(prompt_tip_font)
            prompt_tip.setStyleSheet("color: gray;")
            prompt_tip.setWordWrap(True)
            prompt_tip.setContentsMargins(0, 2, 0, 10)
            prompt_layout.addWidget(prompt_tip)
            
            # Negative prompt (mainly for SD)
            self.negative_prompt_label = QLabel("Negative Prompt (SD only):")
            prompt_layout.addWidget(self.negative_prompt_label)
            
            self.negative_entry = QLineEdit()
            self.negative_entry.setText(self.settings.get('cloud_negative_prompt', 'text, writing, letters'))
            prompt_layout.addWidget(self.negative_entry)
            
            # Inference steps (for SD)
            self.steps_widget = QWidget()
            steps_layout = QHBoxLayout(self.steps_widget)
            steps_layout.setContentsMargins(0, 10, 0, 5)
            prompt_layout.addWidget(self.steps_widget)
            
            self.steps_label = QLabel("Inference Steps (SD only):")
            self.steps_label.setMinimumWidth(180)
            steps_layout.addWidget(self.steps_label)
            
            self.steps_spinbox = QSpinBox()
            self.steps_spinbox.setRange(10, 50)
            self.steps_spinbox.setValue(self.settings.get('cloud_inference_steps', 20))
            steps_layout.addWidget(self.steps_spinbox)
            
            steps_desc = QLabel("(Higher = better quality, slower)")
            steps_desc_font = QFont('Arial', 9)
            steps_desc.setFont(steps_desc_font)
            steps_desc.setStyleSheet("color: gray;")
            steps_layout.addWidget(steps_desc)
            steps_layout.addStretch()
            
            # Add stretch at end
            content_layout.addStretch()
            
            # Initially hide prompt frame if not using appropriate model
            if self.cloud_model_selected == 'custom':
                self.prompt_group.setVisible(False)
            
            # Show/hide SD-specific options based on model
            self._on_cloud_model_change(self.cloud_model_selected)
    
    def _on_cloud_model_change(self, model):
        """Handle cloud model selection change"""
        # Store the selected model
        self.cloud_model_selected = model
        
        # Show/hide custom version entry
        if model == 'custom':
            self.custom_version_widget.setVisible(True)
            # DON'T HIDE THE PROMPT FRAME FOR CUSTOM MODELS
            self.prompt_group.setVisible(True)
        else:
            self.custom_version_widget.setVisible(False)
            self.prompt_group.setVisible(True)
        
        # Show/hide SD-specific options
        if model == 'sd-inpainting':
            # Show negative prompt and steps
            self.negative_prompt_label.setVisible(True)
            self.negative_entry.setVisible(True)
            self.steps_widget.setVisible(True)
        else:
            # Hide SD-specific options
            self.negative_prompt_label.setVisible(False)
            self.negative_entry.setVisible(False)
            self.steps_widget.setVisible(False)
        
    def _toggle_preprocessing(self):
        """Enable/disable preprocessing controls based on main toggle"""
        enabled = self.preprocess_enabled.get()
        
        # Widgets that must remain enabled regardless of toggle (widgets only, not Tk variables)
        always_on = []
        for name in [
            'tiling_frame',
            'tile_size_spinbox', 'tile_overlap_spinbox',
            'chunk_frame', 'chunk_height_spinbox', 'chunk_overlap_spinbox',
            'chunk_height_label', 'chunk_overlap_label',
            'chunk_height_unit_label', 'chunk_overlap_unit_label',
            # Compression controls should always be active (separate from preprocessing)
            'compression_frame', 'compression_toggle', 'compression_format_combo', 'jpeg_quality_spin', 'png_level_spin', 'webp_quality_spin'
        ]:
            if hasattr(self, name):
                always_on.append(getattr(self, name))
        
        for control in self.preprocessing_controls:
            try:
                if control in always_on:
                    # Ensure enabled
                    if isinstance(control, (tk.Scale, tb.Spinbox, tb.Checkbutton)):
                        control.config(state='normal')
                    elif isinstance(control, tk.LabelFrame):
                        control.config(fg='white')
                        self._toggle_frame_children(control, True)
                    elif isinstance(control, tk.Label):
                        control.config(fg='white')
                    elif isinstance(control, tk.Frame):
                        self._toggle_frame_children(control, True)
                    continue
            except Exception:
                pass
            
            # Normal enable/disable logic for other controls
            if isinstance(control, (tk.Scale, tb.Spinbox, tb.Checkbutton)):
                control.config(state='normal' if enabled else 'disabled')
            elif isinstance(control, tk.LabelFrame):
                control.config(fg='white' if enabled else 'gray')
            elif isinstance(control, tk.Label):
                control.config(fg='white' if enabled else 'gray')
            elif isinstance(control, tk.Frame):
                self._toggle_frame_children(control, enabled)
        
        # Final enforcement for always-on widgets (in case they were not in list)
        try:
            if hasattr(self, 'chunk_height_spinbox'):
                self.chunk_height_spinbox.config(state='normal')
            if hasattr(self, 'chunk_overlap_spinbox'):
                self.chunk_overlap_spinbox.config(state='normal')
            if hasattr(self, 'chunk_height_label'):
                self.chunk_height_label.config(fg='white')
            if hasattr(self, 'chunk_overlap_label'):
                self.chunk_overlap_label.config(fg='white')
            if hasattr(self, 'chunk_height_unit_label'):
                self.chunk_height_unit_label.config(fg='white')
            if hasattr(self, 'chunk_overlap_unit_label'):
                self.chunk_overlap_unit_label.config(fg='white')
        except Exception:
            pass
        # Ensure tiling fields respect their own toggle regardless of preprocessing state
        try:
            if hasattr(self, '_toggle_tiling_controls'):
                self._toggle_tiling_controls()
        except Exception:
            pass
    def _toggle_frame_children(self, frame, enabled):
        """Recursively enable/disable all children of a frame"""
        for child in frame.winfo_children():
            if isinstance(child, (tk.Scale, tb.Spinbox, tb.Checkbutton, ttk.Combobox)):
                try:
                    child.config(state='readonly' if (enabled and isinstance(child, ttk.Combobox)) else ('normal' if enabled else 'disabled'))
                except Exception:
                    child.config(state='normal' if enabled else 'disabled')
            elif isinstance(child, tk.Label):
                child.config(fg='white' if enabled else 'gray')
            elif isinstance(child, tk.Frame):
                self._toggle_frame_children(child, enabled)

    def _toggle_roi_locality_controls(self):
        """Show/hide ROI locality controls based on toggle."""
        try:
            enabled = self.roi_locality_var.get()
        except Exception:
            enabled = False
        # Rows to manage
        rows = [
            getattr(self, 'roi_pad_row', None),
            getattr(self, 'roi_min_row', None),
            getattr(self, 'roi_area_row', None),
            getattr(self, 'roi_max_row', None)
        ]
        for row in rows:
            try:
                if row is None: continue
                if enabled:
                    # Only pack if not already managed
                    row.pack(fill='x', pady=5)
                else:
                    row.pack_forget()
            except Exception:
                pass

    def _toggle_tiling_controls(self):
        """Enable/disable tiling size/overlap fields based on tiling toggle."""
        try:
            enabled = bool(self.inpaint_tiling_enabled.get())
        except Exception:
            enabled = False
        state = 'normal' if enabled else 'disabled'
        try:
            self.tile_size_spinbox.config(state=state)
        except Exception:
            pass
        try:
            self.tile_overlap_spinbox.config(state=state)
        except Exception:
            pass

    def _toggle_compression_format(self):
        """Show only the controls relevant to the selected format (hide others)."""
        fmt = getattr(self, 'compression_format_var', tk.StringVar(value='jpeg')).get()
        try:
            # Hide all rows first
            for row in [getattr(self, 'jpeg_row', None), getattr(self, 'png_row', None), getattr(self, 'webp_row', None)]:
                try:
                    if row is not None:
                        row.pack_forget()
                except Exception:
                    pass
            # Show the selected one
            if fmt == 'jpeg':
                if hasattr(self, 'jpeg_row') and self.jpeg_row is not None:
                    self.jpeg_row.pack(fill='x', pady=5)
            elif fmt == 'png':
                if hasattr(self, 'png_row') and self.png_row is not None:
                    self.png_row.pack(fill='x', pady=5)
            else:  # webp
                if hasattr(self, 'webp_row') and self.webp_row is not None:
                    self.webp_row.pack(fill='x', pady=5)
        except Exception:
            pass
    
    def _toggle_ocr_batching_controls(self):
        """Show/hide OCR batching rows based on enable toggle."""
        try:
            enabled = bool(self.ocr_batch_enabled_var.get())
        except Exception:
            enabled = False
        try:
            if hasattr(self, 'ocr_bs_row') and self.ocr_bs_row:
                (self.ocr_bs_row.pack if enabled else self.ocr_bs_row.pack_forget)()
        except Exception:
            pass
        try:
            if hasattr(self, 'ocr_cc_row') and self.ocr_cc_row:
                (self.ocr_cc_row.pack if enabled else self.ocr_cc_row.pack_forget)()
        except Exception:
            pass

    def _create_ocr_tab(self, notebook):
        """Create OCR settings tab with all options"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="OCR Settings")
        
        # Main content
        content_frame = tk.Frame(frame)
        content_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Language hints
        lang_frame = tk.LabelFrame(content_frame, text="Language Detection", padx=15, pady=10)
        lang_frame.pack(fill='x', padx=20, pady=20)
        
        tk.Label(
            lang_frame,
            text="Select languages to prioritize during OCR:",
            font=('Arial', 10)
        ).pack(anchor='w', pady=(0, 10))
        
        # Language checkboxes
        self.lang_vars = {}
        languages = [
            ('ja', 'Japanese'),
            ('ko', 'Korean'),
            ('zh', 'Chinese (Simplified)'),
            ('zh-TW', 'Chinese (Traditional)'),
            ('en', 'English')
        ]
        
        lang_grid = tk.Frame(lang_frame)
        lang_grid.pack(fill='x')
        
        for i, (code, name) in enumerate(languages):
            var = tk.BooleanVar(value=code in self.settings['ocr']['language_hints'])
            self.lang_vars[code] = var
            tb.Checkbutton(
                lang_grid,
                text=name,
                variable=var,
                bootstyle="round-toggle"
            ).grid(row=i//2, column=i%2, sticky='w', padx=10, pady=5)
        
        # OCR parameters
        ocr_frame = tk.LabelFrame(content_frame, text="OCR Parameters", padx=15, pady=10)
        ocr_frame.pack(fill='x', padx=20)
        
        # Confidence threshold
        conf_frame = tk.Frame(ocr_frame)
        conf_frame.pack(fill='x', pady=5)
        tk.Label(conf_frame, text="Confidence Threshold:", width=20, anchor='w').pack(side='left')
        self.confidence_threshold = tk.DoubleVar(value=self.settings['ocr']['confidence_threshold'])
        conf_scale = tk.Scale(
            conf_frame,
            from_=0.0, to=1.0,
            resolution=0.01,
            orient='horizontal',
            variable=self.confidence_threshold,
            length=250
        )
        conf_scale.pack(side='left', padx=10)
        tk.Label(conf_frame, textvariable=self.confidence_threshold, width=5).pack(side='left')
        
        # Detection mode
        mode_frame = tk.Frame(ocr_frame)
        mode_frame.pack(fill='x', pady=5)
        tk.Label(mode_frame, text="Detection Mode:", width=20, anchor='w').pack(side='left')
        self.detection_mode = tk.StringVar(value=self.settings['ocr']['text_detection_mode'])
        mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.detection_mode,
            values=['document', 'text'],
            state='readonly',
            width=15
        )
        mode_combo.pack(side='left', padx=10)
        
        tk.Label(
            mode_frame, 
            text="(document = better for manga, text = simple layouts)",
            font=('Arial', 9),
            fg='gray'
        ).pack(side='left', padx=5)
        
        # Text merging settings
        merge_frame = tk.LabelFrame(content_frame, text="Text Region Merging", padx=15, pady=10)
        merge_frame.pack(fill='x', padx=20, pady=(10, 0))

        
        # Merge nearby threshold
        nearby_frame = tk.Frame(merge_frame)
        nearby_frame.pack(fill='x', pady=5)
        tk.Label(nearby_frame, text="Merge Distance:", width=20, anchor='w').pack(side='left')
        self.merge_nearby_threshold = tk.IntVar(value=self.settings['ocr']['merge_nearby_threshold'])
        nearby_spinbox = tb.Spinbox(
            nearby_frame,
            from_=0,
            to=200,
            textvariable=self.merge_nearby_threshold,
            increment=10,
            width=10
        )
        nearby_spinbox.pack(side='left', padx=10)
        tk.Label(nearby_frame, text="pixels").pack(side='left')

        # Text Filtering Setting
        filter_frame = tk.LabelFrame(content_frame, text="Text Filtering", padx=15, pady=10)
        filter_frame.pack(fill='x', padx=20, pady=(10, 0))
        
        # Minimum text length
        min_length_frame = tk.Frame(filter_frame)
        min_length_frame.pack(fill='x', pady=5)
        
        tk.Label(min_length_frame, text="Min Text Length:", width=20, anchor='w').pack(side='left')
        self.min_text_length_var = tk.IntVar(
            value=self.settings['ocr'].get('min_text_length', 0)
        )
        min_length_spinbox = tb.Spinbox(
            min_length_frame,
            from_=1,
            to=10,
            textvariable=self.min_text_length_var,
            increment=1,
            width=10
        )
        min_length_spinbox.pack(side='left', padx=10)
        tk.Label(min_length_frame, text="characters").pack(side='left')
        
        tk.Label(
            min_length_frame,
            text="(skip text shorter than this)",
            font=('Arial', 9),
            fg='gray'
        ).pack(side='left', padx=10)
        
        # Exclude English text checkbox
        exclude_english_frame = tk.Frame(filter_frame)
        exclude_english_frame.pack(fill='x', pady=(5, 0))
        
        self.exclude_english_var = tk.BooleanVar(
            value=self.settings['ocr'].get('exclude_english_text', False)
        )
        
        tb.Checkbutton(
            exclude_english_frame,
            text="Exclude primarily English text (tunable threshold)",
            variable=self.exclude_english_var,
            bootstyle="round-toggle"
        ).pack(anchor='w')
        
        # Threshold slider
        english_threshold_frame = tk.Frame(filter_frame)
        english_threshold_frame.pack(fill='x', pady=5)
        tk.Label(english_threshold_frame, text="English Exclude Threshold:", width=28, anchor='w').pack(side='left')
        self.english_exclude_threshold = tk.DoubleVar(
            value=self.settings['ocr'].get('english_exclude_threshold', 0.7)
        )
        threshold_scale = tk.Scale(
            english_threshold_frame,
            from_=0.6, to=0.99,
            resolution=0.01,
            orient='horizontal',
            variable=self.english_exclude_threshold,
            length=250,
            command=lambda v: self.english_threshold_label.config(text=f"{float(v)*100:.0f}%")
        )
        threshold_scale.pack(side='left', padx=10)
        self.english_threshold_label = tk.Label(english_threshold_frame, text=f"{int(self.english_exclude_threshold.get()*100)}%", width=5)
        self.english_threshold_label.pack(side='left')
        
        # Minimum character count
        min_chars_frame = tk.Frame(filter_frame)
        min_chars_frame.pack(fill='x', pady=5)
        tk.Label(min_chars_frame, text="Min chars to exclude as English:", width=28, anchor='w').pack(side='left')
        self.english_exclude_min_chars = tk.IntVar(
            value=self.settings['ocr'].get('english_exclude_min_chars', 4)
        )
        min_chars_spinbox = tb.Spinbox(
            min_chars_frame,
            from_=1,
            to=10,
            textvariable=self.english_exclude_min_chars,
            increment=1,
            width=10
        )
        min_chars_spinbox.pack(side='left', padx=10)
        tk.Label(min_chars_frame, text="characters").pack(side='left')
        
        # Legacy aggressive short-token filter
        exclude_short_frame = tk.Frame(filter_frame)
        exclude_short_frame.pack(fill='x', pady=(5, 0))
        self.english_exclude_short_tokens = tk.BooleanVar(
            value=self.settings['ocr'].get('english_exclude_short_tokens', False)
        )
        tb.Checkbutton(
            exclude_short_frame,
            text="Aggressively drop very short ASCII tokens (legacy)",
            variable=self.english_exclude_short_tokens,
            bootstyle="round-toggle"
        ).pack(anchor='w')
        
        # Help text
        tk.Label(
            filter_frame,
            text="💡 Text filtering helps skip:\n"
                 "   • UI elements and watermarks\n"
                 "   • Page numbers and copyright text\n"
                 "   • Single characters or symbols\n"
                 "   • Non-target language text",
            font=('Arial', 9),
            fg='gray',
            justify='left'
        ).pack(anchor='w', pady=(10, 0))
        
        # Azure-specific OCR settings (existing code continues here)
        azure_ocr_frame = tk.LabelFrame(content_frame, text="Azure OCR Settings", padx=15, pady=10)

        # Azure-specific OCR settings
        azure_ocr_frame = tk.LabelFrame(content_frame, text="Azure OCR Settings", padx=15, pady=10)
        azure_ocr_frame.pack(fill='x', padx=20, pady=(10, 0))

        # Azure merge multiplier
        merge_mult_frame = tk.Frame(azure_ocr_frame)
        merge_mult_frame.pack(fill='x', pady=5)
        tk.Label(merge_mult_frame, text="Merge Multiplier:", width=20, anchor='w').pack(side='left')

        self.azure_merge_multiplier = tk.DoubleVar(
            value=self.settings['ocr'].get('azure_merge_multiplier', 2.0)
        )
        azure_scale = tk.Scale(
            merge_mult_frame,
            from_=1.0,
            to=5.0,
            resolution=0.01,
            orient='horizontal',
            variable=self.azure_merge_multiplier,
            length=200,
            command=lambda v: self._update_azure_label()
        )
        azure_scale.pack(side='left', padx=10)

        self.azure_label = tk.Label(merge_mult_frame, text="2.0x", width=5)
        self.azure_label.pack(side='left')
        self._update_azure_label()

        tk.Label(
            merge_mult_frame,
            text="(multiplies merge distance for Azure lines)",
            font=('Arial', 9),
            fg='gray'
        ).pack(side='left', padx=5)

        # Reading order
        reading_order_frame = tk.Frame(azure_ocr_frame)
        reading_order_frame.pack(fill='x', pady=5)
        tk.Label(reading_order_frame, text="Reading Order:", width=20, anchor='w').pack(side='left')

        self.azure_reading_order = tk.StringVar(
            value=self.settings['ocr'].get('azure_reading_order', 'natural')
        )
        order_combo = ttk.Combobox(
            reading_order_frame,
            textvariable=self.azure_reading_order,
            values=['basic', 'natural'],
            state='readonly',
            width=15
        )
        order_combo.pack(side='left', padx=10)

        tk.Label(
            reading_order_frame,
            text="(natural = better for complex layouts)",
            font=('Arial', 9),
            fg='gray'
        ).pack(side='left', padx=5)

        # Model version
        model_version_frame = tk.Frame(azure_ocr_frame)
        model_version_frame.pack(fill='x', pady=5)
        tk.Label(model_version_frame, text="Model Version:", width=20, anchor='w').pack(side='left')

        self.azure_model_version = tk.StringVar(
            value=self.settings['ocr'].get('azure_model_version', 'latest')
        )
        version_combo = ttk.Combobox(
            model_version_frame,
            textvariable=self.azure_model_version,
            values=['latest', '2022-04-30', '2022-01-30', '2021-09-30'],
            width=15
        )
        version_combo.pack(side='left', padx=10)

        tk.Label(
            model_version_frame,
            text="(use 'latest' for newest features)",
            font=('Arial', 9),
            fg='gray'
        ).pack(side='left', padx=5)

        # Timeout settings
        timeout_frame = tk.Frame(azure_ocr_frame)
        timeout_frame.pack(fill='x', pady=5)

        tk.Label(timeout_frame, text="Max Wait Time:", width=20, anchor='w').pack(side='left')

        self.azure_max_wait = tk.IntVar(
            value=self.settings['ocr'].get('azure_max_wait', 60)
        )
        wait_spinbox = tb.Spinbox(
            timeout_frame,
            from_=10,
            to=120,
            textvariable=self.azure_max_wait,
            increment=5,
            width=10
        )
        wait_spinbox.pack(side='left', padx=10)
        tk.Label(timeout_frame, text="seconds").pack(side='left')

        # Poll interval
        poll_frame = tk.Frame(azure_ocr_frame)
        poll_frame.pack(fill='x', pady=5)

        tk.Label(poll_frame, text="Poll Interval:", width=20, anchor='w').pack(side='left')

        self.azure_poll_interval = tk.DoubleVar(
            value=self.settings['ocr'].get('azure_poll_interval', 0.5)
        )
        poll_scale = tk.Scale(
            poll_frame,
            from_=0.0,
            to=2.0,
            resolution=0.01,
            orient='horizontal',
            variable=self.azure_poll_interval,
            length=200
        )
        poll_scale.pack(side='left', padx=10)

        tk.Label(poll_frame, textvariable=self.azure_poll_interval, width=5).pack(side='left')
        tk.Label(poll_frame, text="sec").pack(side='left')

        # Help text
        tk.Label(
            azure_ocr_frame,
            text="💡 Azure Read API auto-detects language well\n"
                 "💡 Natural reading order works better for manga panels",
            font=('Arial', 9),
            fg='gray',
            justify='left'
        ).pack(anchor='w', pady=(10, 0))
        
        # Rotation correction
        rotation_frame = tk.Frame(merge_frame)
        rotation_frame.pack(fill='x', pady=5)
        self.enable_rotation = tk.BooleanVar(value=self.settings['ocr']['enable_rotation_correction'])
        tb.Checkbutton(
            rotation_frame,
            text="Enable automatic rotation correction for tilted text",
            variable=self.enable_rotation,
            bootstyle="round-toggle"
        ).pack(anchor='w')

        # OCR batching and locality settings
        ocr_batch_frame = tk.LabelFrame(content_frame, text="OCR Batching & Concurrency", padx=15, pady=10)
        ocr_batch_frame.pack(fill='x', padx=20, pady=(10, 0))

        # Enable OCR batching
        self.ocr_batch_enabled_var = tk.BooleanVar(value=self.settings['ocr'].get('ocr_batch_enabled', True))
        tb.Checkbutton(
            ocr_batch_frame,
            text="Enable OCR batching (independent of translation batching)",
            variable=self.ocr_batch_enabled_var,
            command=lambda: self._toggle_ocr_batching_controls(),
            bootstyle="round-toggle"
        ).pack(anchor='w')
        
        # OCR batch size
        ocr_bs_row = tk.Frame(ocr_batch_frame)
        self.ocr_bs_row = ocr_bs_row
        ocr_bs_row.pack(fill='x', pady=5)
        tk.Label(ocr_bs_row, text="OCR Batch Size:", width=20, anchor='w').pack(side='left')
        self.ocr_batch_size_var = tk.IntVar(value=int(self.settings['ocr'].get('ocr_batch_size', 8)))
        self.ocr_batch_size_spin = tb.Spinbox(
            ocr_bs_row,
            from_=1,
            to=32,
            textvariable=self.ocr_batch_size_var,
            width=10
        )
        self.ocr_batch_size_spin.pack(side='left', padx=10)
        tk.Label(ocr_bs_row, text="(Google: items/request; Azure: drives concurrency)", font=('Arial', 9), fg='gray').pack(side='left')

        # OCR Max Concurrency
        ocr_cc_row = tk.Frame(ocr_batch_frame)
        self.ocr_cc_row = ocr_cc_row
        ocr_cc_row.pack(fill='x', pady=5)
        tk.Label(ocr_cc_row, text="OCR Max Concurrency:", width=20, anchor='w').pack(side='left')
        self.ocr_max_conc_var = tk.IntVar(value=int(self.settings['ocr'].get('ocr_max_concurrency', 2)))
        self.ocr_max_conc_spin = tb.Spinbox(
            ocr_cc_row,
            from_=1,
            to=8,
            textvariable=self.ocr_max_conc_var,
            width=10
        )
        self.ocr_max_conc_spin.pack(side='left', padx=10)
        tk.Label(ocr_cc_row, text="(Google: concurrent requests; Azure: workers, capped at 4)", font=('Arial', 9), fg='gray').pack(side='left')
        
        # Apply initial visibility for OCR batching controls
        try:
            self._toggle_ocr_batching_controls()
        except Exception:
            pass

        # ROI sizing
        roi_frame_local = tk.LabelFrame(content_frame, text="ROI Locality Controls", padx=15, pady=10)
        roi_frame_local.pack(fill='x', padx=20, pady=(10, 0))

        # ROI locality toggle (now inside this section)
        self.roi_locality_var = tk.BooleanVar(value=self.settings['ocr'].get('roi_locality_enabled', False))
        tb.Checkbutton(
            roi_frame_local,
            text="Enable ROI-based OCR locality and batching (uses bubble detection)",
            variable=self.roi_locality_var,
            command=self._toggle_roi_locality_controls,
            bootstyle="round-toggle"
        ).pack(anchor='w', pady=(0,5))

        # ROI padding ratio
        roi_pad_row = tk.Frame(roi_frame_local)
        roi_pad_row.pack(fill='x', pady=5)
        self.roi_pad_row = roi_pad_row
        tk.Label(roi_pad_row, text="ROI Padding Ratio:", width=20, anchor='w').pack(side='left')
        self.roi_padding_ratio_var = tk.DoubleVar(value=float(self.settings['ocr'].get('roi_padding_ratio', 0.08)))
        roi_pad_scale = tk.Scale(
            roi_pad_row,
            from_=0.0,
            to=0.30,
            resolution=0.01,
            orient='horizontal',
            variable=self.roi_padding_ratio_var,
            length=200
        )
        roi_pad_scale.pack(side='left', padx=10)
        tk.Label(roi_pad_row, textvariable=self.roi_padding_ratio_var, width=5).pack(side='left')

        # ROI min side / area
        roi_min_row = tk.Frame(roi_frame_local)
        roi_min_row.pack(fill='x', pady=5)
        self.roi_min_row = roi_min_row
        tk.Label(roi_min_row, text="Min ROI Side:", width=20, anchor='w').pack(side='left')
        self.roi_min_side_var = tk.IntVar(value=int(self.settings['ocr'].get('roi_min_side_px', 12)))
        self.roi_min_side_spin = tb.Spinbox(
            roi_min_row,
            from_=1,
            to=64,
            textvariable=self.roi_min_side_var,
            width=10
        )
        self.roi_min_side_spin.pack(side='left', padx=10)
        tk.Label(roi_min_row, text="px").pack(side='left')

        roi_area_row = tk.Frame(roi_frame_local)
        roi_area_row.pack(fill='x', pady=5)
        self.roi_area_row = roi_area_row
        tk.Label(roi_area_row, text="Min ROI Area:", width=20, anchor='w').pack(side='left')
        self.roi_min_area_var = tk.IntVar(value=int(self.settings['ocr'].get('roi_min_area_px', 100)))
        self.roi_min_area_spin = tb.Spinbox(
            roi_area_row,
            from_=1,
            to=5000,
            textvariable=self.roi_min_area_var,
            width=10
        )
        self.roi_min_area_spin.pack(side='left', padx=10)
        tk.Label(roi_area_row, text="px^2").pack(side='left')

        # ROI max side (0 disables)
        roi_max_row = tk.Frame(roi_frame_local)
        roi_max_row.pack(fill='x', pady=5)
        self.roi_max_row = roi_max_row
        tk.Label(roi_max_row, text="ROI Max Side (0=off):", width=20, anchor='w').pack(side='left')
        self.roi_max_side_var = tk.IntVar(value=int(self.settings['ocr'].get('roi_max_side', 0)))
        self.roi_max_side_spin = tb.Spinbox(
            roi_max_row,
            from_=0,
            to=2048,
            textvariable=self.roi_max_side_var,
            width=10
        )
        self.roi_max_side_spin.pack(side='left', padx=10)

        # Apply initial visibility based on toggle
        self._toggle_roi_locality_controls()

        # AI Bubble Detection Settings
        bubble_frame = tk.LabelFrame(content_frame, text="AI Bubble Detection", padx=15, pady=10)
        bubble_frame.pack(fill='x', padx=20, pady=(10, 0))

        # Enable bubble detection
        self.bubble_detection_enabled = tk.BooleanVar(
            value=self.settings['ocr'].get('bubble_detection_enabled', False)
        )

        bubble_enable_cb = tb.Checkbutton(
            bubble_frame,
            text="Enable AI-powered bubble detection (overrides traditional merging)",
            variable=self.bubble_detection_enabled,
            bootstyle="round-toggle",
            command=self._toggle_bubble_controls
        )
        bubble_enable_cb.pack(anchor='w')


        # Detector type dropdown - PUT THIS DIRECTLY IN bubble_frame
        detector_type_frame = tk.Frame(bubble_frame)
        detector_type_frame.pack(fill='x', pady=(10, 0))

        tk.Label(detector_type_frame, text="Detector:", width=15, anchor='w').pack(side='left')

        # Model mapping
        self.detector_models = {
            'RTEDR_onnx': 'ogkalu/comic-text-and-bubble-detector',
            'RT-DETR': 'ogkalu/comic-text-and-bubble-detector',
            'YOLOv8 Speech': 'ogkalu/comic-speech-bubble-detector-yolov8m',
            'YOLOv8 Text': 'ogkalu/comic-text-segmenter-yolov8m',
            'YOLOv8 Manga': 'ogkalu/manga-text-detector-yolov8s',
            'Custom Model': ''
        }

        # Get saved detector type (default to ONNX backend)
        saved_type = self.settings['ocr'].get('detector_type', 'rtdetr_onnx')
        if saved_type == 'rtdetr_onnx':
            initial_selection = 'RTEDR_onnx'
        elif saved_type == 'rtdetr':
            initial_selection = 'RT-DETR'
        elif saved_type == 'yolo':
            initial_selection = 'YOLOv8 Speech'
        elif saved_type == 'custom':
            initial_selection = 'Custom Model'
        else:
            initial_selection = 'RTEDR_onnx'

        self.detector_type = tk.StringVar(value=initial_selection)

        detector_combo = ttk.Combobox(
            detector_type_frame,
            textvariable=self.detector_type,
            values=list(self.detector_models.keys()),
            state='readonly',
            width=20
        )
        detector_combo.pack(side='left', padx=(10, 0))
        detector_combo.bind('<<ComboboxSelected>>', lambda e: self._on_detector_type_changed())

        # NOW create the settings frame
        self.yolo_settings_frame = tk.LabelFrame(bubble_frame, text="Model Settings", padx=10, pady=5)
        self.rtdetr_settings_frame = self.yolo_settings_frame  # Alias

        # NOW you can create model_frame inside yolo_settings_frame
        model_frame = tk.Frame(self.yolo_settings_frame)
        model_frame.pack(fill='x', pady=(5, 0))

        tk.Label(model_frame, text="Model:", width=12, anchor='w').pack(side='left')

        self.bubble_model_path = tk.StringVar(
            value=self.settings['ocr'].get('bubble_model_path', '')
        )
        self.rtdetr_model_url = self.bubble_model_path  # Alias

        # Style the entry to match GUI theme
        self.bubble_model_entry = tk.Entry(
            model_frame,
            textvariable=self.bubble_model_path,
            width=35,
            state='readonly',
            bg='#2b2b2b',  # Dark background
            fg='#ffffff',  # White text
            insertbackground='#ffffff',  # White cursor
            readonlybackground='#1e1e1e',  # Even darker when readonly
            relief='flat',
            bd=1
        )
        self.bubble_model_entry.pack(side='left', padx=(0, 10))
        self.rtdetr_url_entry = self.bubble_model_entry  # Alias
        
        # Store for compatibility
        self.detector_radio_widgets = [detector_combo]

        # Settings frames
        self.yolo_settings_frame = tk.LabelFrame(bubble_frame, text="Model Settings", padx=10, pady=5)
        self.rtdetr_settings_frame = self.yolo_settings_frame  # Alias

        # Model path/URL
        model_frame = tk.Frame(self.yolo_settings_frame)
        model_frame.pack(fill='x', pady=(5, 0))

        tk.Label(model_frame, text="Model:", width=12, anchor='w').pack(side='left')

        self.bubble_model_path = tk.StringVar(
            value=self.settings['ocr'].get('bubble_model_path', '')
        )
        self.rtdetr_model_url = self.bubble_model_path  # Alias

        self.bubble_model_entry = tk.Entry(
            model_frame,
            textvariable=self.bubble_model_path,
            width=35,
            state='readonly'
        )
        self.bubble_model_entry.pack(side='left', padx=(0, 10))
        self.rtdetr_url_entry = self.bubble_model_entry  # Alias

        self.bubble_browse_btn = tb.Button(
            model_frame,
            text="Browse",
            command=self._browse_bubble_model,
            bootstyle="primary"
        )
        self.bubble_browse_btn.pack(side='left')

        self.bubble_clear_btn = tb.Button(
            model_frame,
            text="Clear",
            command=self._clear_bubble_model,
            bootstyle="secondary"
        )
        self.bubble_clear_btn.pack(side='left', padx=(5, 0))

        # Download and Load buttons
        button_frame = tk.Frame(self.yolo_settings_frame)
        button_frame.pack(fill='x', pady=(10, 0))

        tk.Label(button_frame, text="Actions:", width=12, anchor='w').pack(side='left')

        self.rtdetr_download_btn = tb.Button(
            button_frame,
            text="Download",
            command=self._download_rtdetr_model,
            bootstyle="success"
        )
        self.rtdetr_download_btn.pack(side='left', padx=(0, 5))

        self.rtdetr_load_btn = tb.Button(
            button_frame,
            text="Load Model",
            command=self._load_rtdetr_model,
            bootstyle="primary"
        )
        self.rtdetr_load_btn.pack(side='left')

        self.rtdetr_status_label = tk.Label(
            button_frame,
            text="",
            font=('Arial', 9)
        )
        self.rtdetr_status_label.pack(side='left', padx=(15, 0))

        # RT-DETR Detection classes
        rtdetr_classes_frame = tk.Frame(self.yolo_settings_frame)
        rtdetr_classes_frame.pack(fill='x', pady=(10, 0))

        tk.Label(rtdetr_classes_frame, text="Detect:", width=12, anchor='w').pack(side='left')

        self.detect_empty_bubbles = tk.BooleanVar(
            value=self.settings['ocr'].get('detect_empty_bubbles', True)
        )
        empty_cb = tk.Checkbutton(
            rtdetr_classes_frame,
            text="Empty Bubbles",
            variable=self.detect_empty_bubbles
        )
        empty_cb.pack(side='left', padx=(0, 10))

        self.detect_text_bubbles = tk.BooleanVar(
            value=self.settings['ocr'].get('detect_text_bubbles', True)
        )
        text_cb = tk.Checkbutton(
            rtdetr_classes_frame,
            text="Text Bubbles",
            variable=self.detect_text_bubbles
        )
        text_cb.pack(side='left', padx=(0, 10))

        self.detect_free_text = tk.BooleanVar(
            value=self.settings['ocr'].get('detect_free_text', True)
        )
        free_cb = tk.Checkbutton(
            rtdetr_classes_frame,
            text="Free Text",
            variable=self.detect_free_text
        )
        free_cb.pack(side='left')
        
        self.rtdetr_classes_frame = rtdetr_classes_frame

        # Confidence
        conf_frame = tk.Frame(self.yolo_settings_frame)
        conf_frame.pack(fill='x', pady=(10, 0))

        tk.Label(conf_frame, text="Confidence:", width=12, anchor='w').pack(side='left')

        detector_label = self.detector_type.get()
        default_conf = 0.3 if ('RT-DETR' in detector_label or 'RTEDR_onnx' in detector_label or 'onnx' in detector_label.lower()) else 0.5
        
        self.bubble_confidence = tk.DoubleVar(
            value=self.settings['ocr'].get('bubble_confidence', default_conf)
        )
        self.rtdetr_confidence = self.bubble_confidence

        self.bubble_conf_scale = tk.Scale(
            conf_frame,
            from_=0.0,
            to=0.99,
            resolution=0.01,
            orient='horizontal',
            variable=self.bubble_confidence,
            length=200,
            command=lambda v: self.bubble_conf_label.config(text=f"{float(v):.2f}")
        )
        self.bubble_conf_scale.pack(side='left', padx=(0, 10))
        self.rtdetr_conf_scale = self.bubble_conf_scale

        self.bubble_conf_label = tk.Label(conf_frame, text=f"{self.bubble_confidence.get():.2f}", width=5)
        self.bubble_conf_label.pack(side='left')
        self.rtdetr_conf_label = self.bubble_conf_label

        # Status label
        # YOLO-specific: Max detections (only visible for YOLO)
        self.yolo_maxdet_row = tk.Frame(self.yolo_settings_frame)
        self.yolo_maxdet_row.pack_forget()
        tk.Label(self.yolo_maxdet_row, text="Max detections:", width=12, anchor='w').pack(side='left')
        self.bubble_max_det_yolo_var = tk.IntVar(
            value=self.settings['ocr'].get('bubble_max_detections_yolo', 100)
        )
        tb.Spinbox(
            self.yolo_maxdet_row,
            from_=1,
            to=2000,
            textvariable=self.bubble_max_det_yolo_var,
            width=10
        ).pack(side='left', padx=(0,10))

        self.bubble_status_label = tk.Label(
            bubble_frame,
            text="",
            font=('Arial', 9)
        )
        self.bubble_status_label.pack(anchor='w', pady=(10, 0))

        # Store controls
        self.bubble_controls = [
            detector_combo,
            self.bubble_model_entry,
            self.bubble_browse_btn,
            self.bubble_clear_btn,
            self.bubble_conf_scale,
            self.rtdetr_download_btn,
            self.rtdetr_load_btn
        ]

        self.rtdetr_controls = [
            self.rtdetr_url_entry,
            self.rtdetr_load_btn,
            self.rtdetr_download_btn,
            self.rtdetr_conf_scale,
            empty_cb,
            text_cb,
            free_cb
        ]

        self.yolo_controls = [
            self.bubble_model_entry,
            self.bubble_browse_btn,
            self.bubble_clear_btn,
            self.bubble_conf_scale,
            self.yolo_maxdet_row
        ]

        # Initialize control states
        self._toggle_bubble_controls()

        # Only call detector change after everything is initialized
        if self.bubble_detection_enabled.get():
            try:
                self._on_detector_type_changed()
                self._update_bubble_status()
            except AttributeError:
                # Frames not yet created, skip initialization
                pass

        # Check status after dialog ready
        self.dialog.after(500, self._check_rtdetr_status)
    
    def _on_detector_type_changed(self):
        """Handle detector type change"""
        if not hasattr(self, 'bubble_detection_enabled'):
            return
            
        if not self.bubble_detection_enabled.get():
            self.yolo_settings_frame.pack_forget()
            return
        
        detector = self.detector_type.get()
        
        # Handle different detector types
        if detector == 'Custom Model':
            # Custom model - enable manual entry
            self.bubble_model_path.set(self.settings['ocr'].get('custom_model_path', ''))
            self.bubble_model_entry.config(
                state='normal',
                bg='#2b2b2b',
                readonlybackground='#2b2b2b'
            )
            # Show browse/clear buttons for custom
            self.bubble_browse_btn.pack(side='left')
            self.bubble_clear_btn.pack(side='left', padx=(5, 0))
            # Hide download button
            self.rtdetr_download_btn.pack_forget()
        elif detector in self.detector_models:
            # HuggingFace model
            url = self.detector_models[detector]
            self.bubble_model_path.set(url)
            # Make entry read-only for HuggingFace models
            self.bubble_model_entry.config(
                state='readonly',
                readonlybackground='#1e1e1e'
            )
            # Hide browse/clear buttons for HuggingFace models
            self.bubble_browse_btn.pack_forget()
            self.bubble_clear_btn.pack_forget()
            # Show download button
            self.rtdetr_download_btn.pack(side='left', padx=(0, 5))
        
        # Show/hide RT-DETR specific controls
        if 'RT-DETR' in detector or 'RTEDR_onnx' in detector:
            self.rtdetr_classes_frame.pack(fill='x', pady=(10, 0), after=self.rtdetr_load_btn.master)
            # Hide YOLO-only max det row
            self.yolo_maxdet_row.pack_forget()
        else:
            self.rtdetr_classes_frame.pack_forget()
            # Show YOLO-only max det row for YOLO models
            if 'YOLO' in detector or 'Yolo' in detector or 'yolo' in detector or detector == 'Custom Model':
                self.yolo_maxdet_row.pack(fill='x', pady=(6,0))
            else:
                self.yolo_maxdet_row.pack_forget()
        
        # Always show settings frame
        self.yolo_settings_frame.pack(fill='x', pady=(10, 0))
        
        # Update status
        self._update_bubble_status()

    def _download_rtdetr_model(self):
        """Download selected model"""
        try:
            detector = self.detector_type.get()
            model_url = self.bubble_model_path.get()
            
            self.rtdetr_status_label.config(text="Downloading...", fg='orange')
            self.dialog.update_idletasks()
            
            if 'RTEDR_onnx' in detector:
                from bubble_detector import BubbleDetector
                bd = BubbleDetector()
                if bd.load_rtdetr_onnx_model(model_id=model_url):
                    self.rtdetr_status_label.config(text="✅ Downloaded", fg='green')
                    messagebox.showinfo("Success", f"RTEDR_onnx model downloaded successfully!")
                else:
                    self.rtdetr_status_label.config(text="❌ Failed", fg='red')
                    messagebox.showerror("Error", f"Failed to download RTEDR_onnx model")
            elif 'RT-DETR' in detector:
                # RT-DETR handling (works fine)
                from bubble_detector import BubbleDetector
                bd = BubbleDetector()
                
                if bd.load_rtdetr_model(model_id=model_url):
                    self.rtdetr_status_label.config(text="✅ Downloaded", fg='green')
                    messagebox.showinfo("Success", f"RT-DETR model downloaded successfully!")
                else:
                    self.rtdetr_status_label.config(text="❌ Failed", fg='red')
                    messagebox.showerror("Error", f"Failed to download RT-DETR model")
            else:
                # FIX FOR YOLO: Download to a simpler local path
                from huggingface_hub import hf_hub_download
                import os
                
                # Create models directory
                models_dir = "models"
                os.makedirs(models_dir, exist_ok=True)
                
                # Define simple local filenames
                filename_map = {
                    'ogkalu/comic-speech-bubble-detector-yolov8m': 'comic-speech-bubble-detector.pt',
                    'ogkalu/comic-text-segmenter-yolov8m': 'comic-text-segmenter.pt',
                    'ogkalu/manga-text-detector-yolov8s': 'manga-text-detector.pt'
                }
                
                filename = filename_map.get(model_url, 'model.pt')
                
                # Download to cache first
                cached_path = hf_hub_download(repo_id=model_url, filename=filename)
                
                # Copy to local models directory with simple path
                import shutil
                local_path = os.path.join(models_dir, filename)
                shutil.copy2(cached_path, local_path)
                
                # Set the simple local path instead of the cache path
                self.bubble_model_path.set(local_path)
                self.rtdetr_status_label.config(text="✅ Downloaded", fg='green')
                messagebox.showinfo("Success", f"Model downloaded to:\n{local_path}")
        
        except ImportError:
            self.rtdetr_status_label.config(text="❌ Missing deps", fg='red')
            messagebox.showerror("Error", "Install: pip install huggingface-hub transformers")
        except Exception as e:
            self.rtdetr_status_label.config(text="❌ Error", fg='red')
            messagebox.showerror("Error", f"Download failed: {e}")

    def _check_rtdetr_status(self):
        """Check if model is already loaded"""
        try:
            from bubble_detector import BubbleDetector
            
            if hasattr(self.main_gui, 'manga_tab') and hasattr(self.main_gui.manga_tab, 'translator'):
                translator = self.main_gui.manga_tab.translator
                if hasattr(translator, 'bubble_detector') and translator.bubble_detector:
                    if getattr(translator.bubble_detector, 'rtdetr_onnx_loaded', False):
                        self.rtdetr_status_label.config(text="✅ Loaded", fg='green')
                        return True
                    if getattr(translator.bubble_detector, 'rtdetr_loaded', False):
                        self.rtdetr_status_label.config(text="✅ Loaded", fg='green')
                        return True
                    elif getattr(translator.bubble_detector, 'model_loaded', False):
                        self.rtdetr_status_label.config(text="✅ Loaded", fg='green')
                        return True
            
            self.rtdetr_status_label.config(text="Not loaded", fg='gray')
            return False
            
        except ImportError:
            self.rtdetr_status_label.config(text="❌ Missing deps", fg='red')
            return False
        except Exception:
            self.rtdetr_status_label.config(text="Not loaded", fg='gray')
            return False

    def _load_rtdetr_model(self):
        """Load selected model"""
        try:
            from bubble_detector import BubbleDetector
            
            self.rtdetr_status_label.config(text="Loading...", fg='orange')
            self.dialog.update_idletasks()
            
            bd = BubbleDetector()
            detector = self.detector_type.get()
            model_path = self.bubble_model_path.get()
            
            if 'RTEDR_onnx' in detector:
                # RT-DETR (ONNX) uses repo id directly
                if bd.load_rtdetr_onnx_model(model_id=model_path):
                    self.rtdetr_status_label.config(text="✅ Ready", fg='green')
                    messagebox.showinfo("Success", f"RTEDR_onnx model loaded successfully!")
                else:
                    self.rtdetr_status_label.config(text="❌ Failed", fg='red')
            elif 'RT-DETR' in detector:
                # RT-DETR uses model_id directly
                if bd.load_rtdetr_model(model_id=model_path):
                    self.rtdetr_status_label.config(text="✅ Ready", fg='green')
                    messagebox.showinfo("Success", f"RT-DETR model loaded successfully!")
                else:
                    self.rtdetr_status_label.config(text="❌ Failed", fg='red')
            else:
                # YOLOv8 - CHECK LOCAL MODELS FOLDER FIRST
                if model_path.startswith('ogkalu/'):
                    # It's a HuggingFace ID - check if already downloaded
                    filename_map = {
                        'ogkalu/comic-speech-bubble-detector-yolov8m': 'comic-speech-bubble-detector.pt',
                        'ogkalu/comic-text-segmenter-yolov8m': 'comic-text-segmenter.pt',
                        'ogkalu/manga-text-detector-yolov8s': 'manga-text-detector.pt'
                    }
                    
                    filename = filename_map.get(model_path, 'model.pt')
                    local_path = os.path.join('models', filename)
                    
                    # Check if it exists locally
                    if os.path.exists(local_path):
                        # Use the local file
                        model_path = local_path
                        self.bubble_model_path.set(local_path)  # Update the field
                    else:
                        # Not downloaded yet
                        messagebox.showwarning("Download Required", 
                            f"Model not found locally.\nPlease download it first using the Download button.")
                        self.rtdetr_status_label.config(text="❌ Not downloaded", fg='orange')
                        return
                
                # Now model_path should be a local file
                if not os.path.exists(model_path):
                    messagebox.showerror("Error", f"Model file not found: {model_path}")
                    self.rtdetr_status_label.config(text="❌ File not found", fg='red')
                    return
                
                # Load the YOLOv8 model from local file
                if bd.load_model(model_path):
                    self.rtdetr_status_label.config(text="✅ Ready", fg='green')
                    messagebox.showinfo("Success", f"YOLOv8 model loaded successfully!")
                    
                    # Auto-convert to ONNX if enabled
                    if os.environ.get('AUTO_CONVERT_TO_ONNX', 'true').lower() == 'true':
                        onnx_path = model_path.replace('.pt', '.onnx')
                        if not os.path.exists(onnx_path):
                            if bd.convert_to_onnx(model_path, onnx_path):
                                logger.info(f"✅ Converted to ONNX: {onnx_path}")
                else:
                    self.rtdetr_status_label.config(text="❌ Failed", fg='red')
                
        except ImportError:
            self.rtdetr_status_label.config(text="❌ Missing deps", fg='red')
            messagebox.showerror("Error", "Install transformers: pip install transformers")
        except Exception as e:
            self.rtdetr_status_label.config(text="❌ Error", fg='red')
            messagebox.showerror("Error", f"Failed to load: {e}")
        
    def _toggle_bubble_controls(self):
        """Enable/disable bubble detection controls"""
        enabled = self.bubble_detection_enabled.get()
        
        if enabled:
            # Enable controls
            for widget in self.bubble_controls:
                try:
                    widget.config(state='normal')
                except:
                    pass
            
            # Show/hide frames based on detector type
            self._on_detector_type_changed()
        else:
            # Disable controls
            for widget in self.bubble_controls:
                try:
                    widget.config(state='disabled')
                except:
                    pass
            
            # Hide frames
            self.yolo_settings_frame.pack_forget()
            self.bubble_status_label.config(text="")

    def _browse_bubble_model(self):
        """Browse for model file"""
        from tkinter import filedialog
        
        path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[
                ("Model files", "*.pt;*.pth;*.bin;*.safetensors"),
                ("All files", "*.*")
            ]
        )
        
        if path:
            self.bubble_model_path.set(path)
            self._update_bubble_status()

    def _clear_bubble_model(self):
        """Clear selected model"""
        self.bubble_model_path.set("")
        self._update_bubble_status()

    def _update_bubble_status(self):
        """Update bubble model status label"""
        if not self.bubble_detection_enabled.get():
            self.bubble_status_label.config(text="")
            return
        
        detector = self.detector_type.get()
        model_path = self.bubble_model_path.get()
        
        if not model_path:
            self.bubble_status_label.config(text="⚠️ No model selected", fg='orange')
            return
        
        if model_path.startswith("ogkalu/"):
            self.bubble_status_label.config(text=f"📥 {detector} ready to download", fg='blue')
        elif os.path.exists(model_path):
            self.bubble_status_label.config(text="✅ Model file ready", fg='green')
        else:
            self.bubble_status_label.config(text="❌ Model file not found", fg='red')

    def _update_azure_label(self):
        """Update Azure multiplier label"""
        value = self.azure_merge_multiplier.get()
        self.azure_label.config(text=f"{value:.1f}x")

    def _set_azure_multiplier(self, value):
        """Set Azure multiplier from preset"""
        self.azure_merge_multiplier.set(value)
        self._update_azure_label()
    
    def _create_advanced_tab(self, notebook):
        """Create advanced settings tab with all options"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Advanced")
        
        # Main content
        content_frame = tk.Frame(frame)
        content_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Format detection
        detect_frame = tk.LabelFrame(content_frame, text="Format Detection", padx=15, pady=10)
        detect_frame.pack(fill='x', padx=20, pady=20)
        
        self.format_detection = tk.IntVar(value=1 if self.settings['advanced']['format_detection'] else 0)
        tb.Checkbutton(
            detect_frame,
            text="Enable automatic manga format detection (reading direction)",
            variable=self.format_detection,
            bootstyle="round-toggle"
        ).pack(anchor='w')
        
        # Webtoon mode
        webtoon_frame = tk.Frame(detect_frame)
        webtoon_frame.pack(fill='x', pady=(10, 0))
        tk.Label(webtoon_frame, text="Webtoon Mode:", width=20, anchor='w').pack(side='left')
        self.webtoon_mode = tk.StringVar(value=self.settings['advanced']['webtoon_mode'])
        webtoon_combo = ttk.Combobox(
            webtoon_frame,
            textvariable=self.webtoon_mode,
            values=['auto', 'enabled', 'disabled'],
            state='readonly',
            width=15
        )
        webtoon_combo.pack(side='left', padx=10)
        
        # Debug settings
        debug_frame = tk.LabelFrame(content_frame, text="Debug Options", padx=15, pady=10)
        debug_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        self.debug_mode = tk.IntVar(value=1 if self.settings['advanced']['debug_mode'] else 0)
        tb.Checkbutton(
            debug_frame,
            text="Enable debug mode (verbose logging)",
            variable=self.debug_mode,
            bootstyle="round-toggle"
        ).pack(anchor='w')
        
        # New: Concise pipeline logs (reduce noise)
        self.concise_logs_var = tk.BooleanVar(value=bool(self.settings.get('advanced', {}).get('concise_logs', True)))
        def _save_concise():
            try:
                self.settings.setdefault('advanced', {})['concise_logs'] = bool(self.concise_logs_var.get())
                if hasattr(self, 'config'):
                    self.config['manga_settings'] = self.settings
                if hasattr(self.main_gui, 'save_config'):
                    self.main_gui.save_config(show_message=False)
            except Exception:
                pass
        tb.Checkbutton(
            debug_frame,
            text="Concise pipeline logs (reduce noise)",
            variable=self.concise_logs_var,
            command=_save_concise,
            bootstyle="round-toggle"
        ).pack(anchor='w', pady=(5, 0))
        
        self.save_intermediate = tk.IntVar(value=1 if self.settings['advanced']['save_intermediate'] else 0)
        tb.Checkbutton(
            debug_frame,
            text="Save intermediate images (preprocessed, detection overlays)",
            variable=self.save_intermediate,
            bootstyle="round-toggle"
        ).pack(anchor='w', pady=(5, 0))
        
        # Performance settings
        perf_frame = tk.LabelFrame(content_frame, text="Performance", padx=15, pady=10)
        # Defer packing until after memory_frame so this section appears below it
        
        # New: Parallel rendering (per-region overlays)
        self.render_parallel_var = tk.BooleanVar(
            value=self.settings.get('advanced', {}).get('render_parallel', True)
        )
        tb.Checkbutton(
            perf_frame,
            text="Enable parallel rendering (per-region overlays)",
            variable=self.render_parallel_var,
            bootstyle="round-toggle"
        ).pack(anchor='w')
        
        self.parallel_processing = tk.IntVar(value=1 if self.settings['advanced']['parallel_processing'] else 0)
        parallel_cb = tb.Checkbutton(
            perf_frame,
            text="Enable parallel processing (experimental)",
            variable=self.parallel_processing,
            bootstyle="round-toggle",
            command=self._toggle_workers
        )
        parallel_cb.pack(anchor='w')
        
        
        # Max workers
        workers_frame = tk.Frame(perf_frame)
        workers_frame.pack(fill='x', pady=(10, 0))
        self.workers_label = tk.Label(workers_frame, text="Max Workers:", width=20, anchor='w')
        self.workers_label.pack(side='left')
        
        self.max_workers = tk.IntVar(value=self.settings['advanced']['max_workers'])
        self.workers_spinbox = tb.Spinbox(
            workers_frame,
            from_=1,
            to=8,
            textvariable=self.max_workers,
            increment=1,
            width=10
        )
        self.workers_spinbox.pack(side='left', padx=10)
        
        tk.Label(workers_frame, text="(threads for parallel processing)").pack(side='left')
        
        # Initialize workers state
        self._toggle_workers()
        
        # Memory management section
        memory_frame = tk.LabelFrame(content_frame, text="Memory Management", padx=15, pady=10)
        memory_frame.pack(fill='x', padx=20, pady=(10, 0))
        
        # Now pack performance BELOW memory management
        perf_frame.pack(fill='x', padx=20)
        
        # Singleton mode for model instances
        self.use_singleton_models = tk.BooleanVar(
            value=self.settings.get('advanced', {}).get('use_singleton_models', True)
        )
        
        def _toggle_singleton_mode():
            """Disable LOCAL parallel processing options when singleton mode is enabled.
            Note: This does NOT affect parallel API calls (batch translation).
            """
            # Update settings immediately to avoid background preloads
            try:
                if 'advanced' not in self.settings:
                    self.settings['advanced'] = {}
                if self.use_singleton_models.get():
                    # Turn off local parallelism and panel preloads
                    self.settings['advanced']['parallel_processing'] = False
                    self.settings['advanced']['parallel_panel_translation'] = False
                    self.settings['advanced']['preload_local_inpainting_for_panels'] = False
                # Persist to config if available
                if hasattr(self, 'config'):
                    self.config['manga_settings'] = self.settings
                if hasattr(self.main_gui, 'save_config'):
                    self.main_gui.save_config(show_message=False)
            except Exception:
                pass
            
            if self.use_singleton_models.get():
                # Disable LOCAL parallel processing toggles (but NOT API batch translation)
                self.parallel_processing.set(0)
                self.parallel_panel_var.set(False)
                # Disable the UI elements for LOCAL parallel processing
                parallel_cb.config(state='disabled')
                panel_cb.config(state='disabled')
                # Also disable the spinboxes
                self.workers_spinbox.config(state='disabled')
                panel_workers_spinbox.config(state='disabled')
                panel_stagger_spinbox.config(state='disabled')
            else:
                # Re-enable the UI elements
                parallel_cb.config(state='normal')
                panel_cb.config(state='normal')
                # Re-enable spinboxes based on their toggle states
                self._toggle_workers()
                _toggle_panel_controls()
        
        singleton_cb = tb.Checkbutton(
            memory_frame,
            text="Use single model instances (saves RAM, only affects local models)",
            variable=self.use_singleton_models,
            bootstyle="round-toggle",
            command=_toggle_singleton_mode
        )
        singleton_cb.pack(anchor='w')
        
        singleton_note = tk.Label(
            memory_frame,
            text="When enabled: One bubble detector & one inpainter shared across all images.\n"
                 "When disabled: Each thread/image can have its own models (uses more RAM).\n"
                 "✅ Batch API translation remains fully functional with singleton mode enabled.",
            font=('Arial', 9),
            fg='gray',
            justify='left'
        )
        singleton_note.pack(anchor='w', pady=(2, 10), padx=(20, 0))
        
        self.auto_cleanup_models = tk.BooleanVar(
            value=self.settings.get('advanced', {}).get('auto_cleanup_models', False)
        )
        cleanup_cb = tb.Checkbutton(
            memory_frame,
            text="Automatically cleanup models after translation to free RAM",
            variable=self.auto_cleanup_models,
            bootstyle="round-toggle"
        )
        cleanup_cb.pack(anchor='w')
        
        # Unload models after translation (disabled by default)
        self.unload_models_var = tk.BooleanVar(
            value=self.settings.get('advanced', {}).get('unload_models_after_translation', False)
        )
        unload_cb = tb.Checkbutton(
            memory_frame,
            text="Unload models after translation (reset translator instance)",
            variable=self.unload_models_var,
            bootstyle="round-toggle"
        )
        unload_cb.pack(anchor='w', pady=(4,0))
        
        # Add a note about parallel processing
        note_label = tk.Label(
            memory_frame,
            text="Note: When parallel panel translation is enabled, cleanup happens after ALL panels complete.",
            font=('Arial', 9),
            fg='gray',
            wraplength=450
        )
        note_label.pack(anchor='w', pady=(5, 0), padx=(20, 0))

        # Panel-level parallel translation
        panel_frame = tk.LabelFrame(content_frame, text="Parallel Panel Translation", padx=15, pady=10)
        panel_frame.pack(fill='x', padx=20, pady=(10, 0))

        # New: Preload local inpainting for panels (default ON)
        preload_row = tk.Frame(panel_frame)
        preload_row.pack(fill='x', pady=5)
        self.preload_local_panels_var = tk.BooleanVar(
            value=self.settings.get('advanced', {}).get('preload_local_inpainting_for_panels', True)
        )
        tb.Checkbutton(
            preload_row,
            text="Preload local inpainting instances for panel-parallel runs",
            variable=self.preload_local_panels_var,
            bootstyle="round-toggle"
        ).pack(anchor='w')

        self.parallel_panel_var = tk.BooleanVar(
            value=self.settings.get('advanced', {}).get('parallel_panel_translation', False)
        )
        
        def _toggle_panel_controls():
            """Enable/disable panel spinboxes based on panel parallel toggle"""
            if self.parallel_panel_var.get() and not self.use_singleton_models.get():
                panel_workers_spinbox.config(state='normal')
                panel_stagger_spinbox.config(state='normal')
            else:
                panel_workers_spinbox.config(state='disabled')
                panel_stagger_spinbox.config(state='disabled')
        
        panel_cb = tb.Checkbutton(
            panel_frame,
            text="Enable parallel panel translation (process multiple images concurrently)",
            variable=self.parallel_panel_var,
            bootstyle="round-toggle",
            command=_toggle_panel_controls
        )
        panel_cb.pack(anchor='w')
        
        # Inpainting Performance (moved from Inpainting tab)
        inpaint_perf = tk.LabelFrame(perf_frame, text="Inpainting Performance", padx=15, pady=10)
        inpaint_perf.pack(fill='x', padx=0, pady=(10,0))
        inpaint_bs_row = tk.Frame(inpaint_perf)
        inpaint_bs_row.pack(fill='x', pady=5)
        tk.Label(inpaint_bs_row, text="Batch Size:", width=20, anchor='w').pack(side='left')
        self.inpaint_batch_size = getattr(self, 'inpaint_batch_size', tk.IntVar(value=self.settings.get('inpainting', {}).get('batch_size', 10)))
        tb.Spinbox(
            inpaint_bs_row,
            from_=1,
            to=32,
            textvariable=self.inpaint_batch_size,
            width=10
        ).pack(side='left', padx=10)
        tk.Label(inpaint_bs_row, text="(process multiple regions at once)", font=('Arial',9), fg='gray').pack(side='left')
        
        cache_row = tk.Frame(inpaint_perf)
        cache_row.pack(fill='x', pady=5)
        self.enable_cache_var = getattr(self, 'enable_cache_var', tk.BooleanVar(value=self.settings.get('inpainting', {}).get('enable_cache', True)))
        tb.Checkbutton(
            cache_row,
            text="Enable inpainting cache (speeds up repeated processing)",
            variable=self.enable_cache_var,
            bootstyle="round-toggle"
        ).pack(anchor='w')

        panels_row = tk.Frame(panel_frame)
        panels_row.pack(fill='x', pady=5)
        tk.Label(panels_row, text="Max concurrent panels:", width=20, anchor='w').pack(side='left')
        self.panel_max_workers_var = tk.IntVar(
            value=self.settings.get('advanced', {}).get('panel_max_workers', 2)
        )
        panel_workers_spinbox = tb.Spinbox(
            panels_row,
            from_=1,
            to=12,
            textvariable=self.panel_max_workers_var,
            width=10
        )
        panel_workers_spinbox.pack(side='left', padx=10)
        
        # Panel start stagger (ms)
        stagger_row = tk.Frame(panel_frame)
        stagger_row.pack(fill='x', pady=5)
        tk.Label(stagger_row, text="Panel start stagger:", width=20, anchor='w').pack(side='left')
        self.panel_stagger_ms_var = tk.IntVar(
            value=self.settings.get('advanced', {}).get('panel_start_stagger_ms', 30)
        )
        panel_stagger_spinbox = tb.Spinbox(
            stagger_row,
            from_=0,
            to=1000,
            textvariable=self.panel_stagger_ms_var,
            width=10
        )
        panel_stagger_spinbox.pack(side='left', padx=10)
        tk.Label(stagger_row, text="ms").pack(side='left')
        
        # Initialize control states
        _toggle_panel_controls()  # Initialize panel spinbox states
        _toggle_singleton_mode()  # Initialize singleton mode state (may override above)

        # ONNX conversion settings
        onnx_frame = tk.LabelFrame(content_frame, text="ONNX Conversion", padx=15, pady=10)
        onnx_frame.pack(fill='x', padx=20, pady=(10, 0))
        
        self.auto_convert_onnx_var = tk.BooleanVar(value=self.settings['advanced'].get('auto_convert_to_onnx', False))
        self.auto_convert_onnx_bg_var = tk.BooleanVar(value=self.settings['advanced'].get('auto_convert_to_onnx_background', True))
        
        def _toggle_onnx_controls():
            # If auto-convert is off, background toggle should be disabled
            state = 'normal' if self.auto_convert_onnx_var.get() else 'disabled'
            try:
                bg_cb.config(state=state)
            except Exception:
                pass
        
        auto_cb = tb.Checkbutton(
            onnx_frame,
            text="Auto-convert local models to ONNX for faster inference (recommended)",
            variable=self.auto_convert_onnx_var,
            bootstyle="round-toggle",
            command=_toggle_onnx_controls
        )
        auto_cb.pack(anchor='w')
        
        bg_cb = tb.Checkbutton(
            onnx_frame,
            text="Convert in background (non-blocking; switches to ONNX when ready)",
            variable=self.auto_convert_onnx_bg_var,
            bootstyle="round-toggle"
        )
        bg_cb.pack(anchor='w', pady=(5, 0))
        
        _toggle_onnx_controls()
        
        # Model memory optimization (quantization)
        quant_frame = tk.LabelFrame(content_frame, text="Model Memory Optimization", padx=15, pady=10)
        quant_frame.pack(fill='x', padx=20, pady=(10, 0))
        
        self.quantize_models_var = tk.BooleanVar(value=self.settings['advanced'].get('quantize_models', False))
        tb.Checkbutton(
            quant_frame,
            text="Reduce RAM with quantized models (global switch)",
            variable=self.quantize_models_var,
            bootstyle="round-toggle"
        ).pack(anchor='w')
        
        # ONNX quantize sub-toggle
        onnx_row = tk.Frame(quant_frame)
        onnx_row.pack(fill='x', pady=(6, 0))
        self.onnx_quantize_var = tk.BooleanVar(value=self.settings['advanced'].get('onnx_quantize', False))
        tb.Checkbutton(
            onnx_row,
            text="Quantize ONNX models to INT8 (dynamic)",
            variable=self.onnx_quantize_var,
            bootstyle="round-toggle"
        ).pack(side='left')
        tk.Label(onnx_row, text="(lower RAM/CPU; slight accuracy trade-off)", font=('Arial', 9), fg='gray').pack(side='left', padx=8)
        
        # Torch precision dropdown
        precision_row = tk.Frame(quant_frame)
        precision_row.pack(fill='x', pady=(6, 0))
        tk.Label(precision_row, text="Torch precision:", width=20, anchor='w').pack(side='left')
        self.torch_precision_var = tk.StringVar(value=self.settings['advanced'].get('torch_precision', 'fp16'))
        ttk.Combobox(
            precision_row,
            textvariable=self.torch_precision_var,
            values=['fp16', 'fp32', 'auto'],
            state='readonly',
            width=10
        ).pack(side='left', padx=10)
        tk.Label(precision_row, text="(fp16 only, since fp32 is currently bugged)", font=('Arial', 9), fg='gray').pack(side='left')
        
        # Aggressive memory cleanup
        cleanup_frame = tk.LabelFrame(content_frame, text="Memory & Cleanup", padx=15, pady=10)
        cleanup_frame.pack(fill='x', padx=20, pady=(10, 0))
        
        self.force_deep_cleanup_var = tk.BooleanVar(value=self.settings.get('advanced', {}).get('force_deep_cleanup_each_image', False))
        tb.Checkbutton(
            cleanup_frame,
            text="Force deep model cleanup after every image (slowest, lowest RAM)",
            variable=self.force_deep_cleanup_var,
            bootstyle="round-toggle"
        ).pack(anchor='w')
        tk.Label(cleanup_frame, text="Also clears shared caches at batch end.", font=('Arial', 9), fg='gray').pack(anchor='w', padx=(0,0), pady=(2,0))
        
        # RAM cap controls
        ramcap_frame = tk.Frame(cleanup_frame)
        ramcap_frame.pack(fill='x', pady=(10, 0))
        self.ram_cap_enabled_var = tk.BooleanVar(value=self.settings.get('advanced', {}).get('ram_cap_enabled', False))
        tb.Checkbutton(
            ramcap_frame,
            text="Enable RAM cap",
            variable=self.ram_cap_enabled_var,
            bootstyle="round-toggle"
        ).pack(anchor='w')
        
        # RAM cap value
        ramcap_value_row = tk.Frame(cleanup_frame)
        ramcap_value_row.pack(fill='x', pady=5)
        tk.Label(ramcap_value_row, text="Max RAM (MB):", width=20, anchor='w').pack(side='left')
        self.ram_cap_mb_var = tk.IntVar(value=int(self.settings.get('advanced', {}).get('ram_cap_mb', 0) or 0))
        tb.Spinbox(
            ramcap_value_row,
            from_=512,
            to=131072,
            textvariable=self.ram_cap_mb_var,
            width=12
        ).pack(side='left', padx=10)
        tk.Label(ramcap_value_row, text="(0 = disabled)", font=('Arial', 9), fg='gray').pack(side='left')
        
        # RAM cap mode
        ramcap_mode_row = tk.Frame(cleanup_frame)
        ramcap_mode_row.pack(fill='x', pady=(5, 0))
        tk.Label(ramcap_mode_row, text="Cap mode:", width=20, anchor='w').pack(side='left')
        self.ram_cap_mode_var = tk.StringVar(value=self.settings.get('advanced', {}).get('ram_cap_mode', 'soft'))
        ttk.Combobox(
            ramcap_mode_row,
            textvariable=self.ram_cap_mode_var,
            values=['soft', 'hard (Windows only)'],
            state='readonly',
            width=20
        ).pack(side='left', padx=10)
        tk.Label(ramcap_mode_row, text="Soft = clean/trim, Hard = OS-enforced (may OOM)", font=('Arial', 9), fg='gray').pack(side='left')
        
        # Advanced RAM gate tuning
        gate_row = tk.Frame(cleanup_frame)
        gate_row.pack(fill='x', pady=(5, 0))
        tk.Label(gate_row, text="Gate timeout (sec):", width=20, anchor='w').pack(side='left')
        self.ram_gate_timeout_var = tk.DoubleVar(value=float(self.settings.get('advanced', {}).get('ram_gate_timeout_sec', 10.0)))
        tb.Spinbox(
            gate_row,
            from_=2.0,
            to=60.0,
            increment=0.5,
            textvariable=self.ram_gate_timeout_var,
            width=12
        ).pack(side='left', padx=10)
        
        floor_row = tk.Frame(cleanup_frame)
        floor_row.pack(fill='x', pady=(5, 0))
        tk.Label(floor_row, text="Gate floor over baseline (MB):", width=25, anchor='w').pack(side='left')
        self.ram_gate_floor_var = tk.IntVar(value=int(self.settings.get('advanced', {}).get('ram_min_floor_over_baseline_mb', 128)))
        tb.Spinbox(
            floor_row,
            from_=64,
            to=2048,
            textvariable=self.ram_gate_floor_var,
            width=12
        ).pack(side='left', padx=10)

    def _toggle_workers(self):
        """Enable/disable worker settings based on parallel processing toggle"""
        enabled = bool(self.parallel_processing.get())
        self.workers_spinbox.config(state='normal' if enabled else 'disabled')
        self.workers_label.config(fg='white' if enabled else 'gray')

    def _apply_defaults_to_controls(self):
        """Apply default values to all visible Tk variables/controls across tabs without rebuilding the dialog."""
        try:
            # Use current in-memory settings (which we set to defaults above)
            s = self.settings if isinstance(getattr(self, 'settings', None), dict) else self.default_settings
            pre = s.get('preprocessing', {})
            comp = s.get('compression', {})
            ocr = s.get('ocr', {})
            adv = s.get('advanced', {})
            inp = s.get('inpainting', {})
            font = s.get('font_sizing', {})

            # Preprocessing
            if hasattr(self, 'preprocess_enabled'): self.preprocess_enabled.set(bool(pre.get('enabled', False)))
            if hasattr(self, 'auto_detect'): self.auto_detect.set(bool(pre.get('auto_detect_quality', True)))
            if hasattr(self, 'contrast_threshold'): self.contrast_threshold.set(float(pre.get('contrast_threshold', 0.4)))
            if hasattr(self, 'sharpness_threshold'): self.sharpness_threshold.set(float(pre.get('sharpness_threshold', 0.3)))
            if hasattr(self, 'enhancement_strength'): self.enhancement_strength.set(float(pre.get('enhancement_strength', 1.5)))
            if hasattr(self, 'noise_threshold'): self.noise_threshold.set(int(pre.get('noise_threshold', 20)))
            if hasattr(self, 'denoise_strength'): self.denoise_strength.set(int(pre.get('denoise_strength', 10)))
            if hasattr(self, 'max_dimension'): self.max_dimension.set(int(pre.get('max_image_dimension', 2000)))
            if hasattr(self, 'max_pixels'): self.max_pixels.set(int(pre.get('max_image_pixels', 2000000)))
            if hasattr(self, 'chunk_height'): self.chunk_height.set(int(pre.get('chunk_height', 1000)))
            if hasattr(self, 'chunk_overlap'): self.chunk_overlap.set(int(pre.get('chunk_overlap', 100)))
            # Compression
            if hasattr(self, 'compression_enabled_var'): self.compression_enabled_var.set(bool(comp.get('enabled', False)))
            if hasattr(self, 'compression_format_var'): self.compression_format_var.set(str(comp.get('format', 'jpeg')))
            if hasattr(self, 'jpeg_quality_var'): self.jpeg_quality_var.set(int(comp.get('jpeg_quality', 85)))
            if hasattr(self, 'png_level_var'): self.png_level_var.set(int(comp.get('png_compress_level', 6)))
            if hasattr(self, 'webp_quality_var'): self.webp_quality_var.set(int(comp.get('webp_quality', 85)))
            # Tiling
            if hasattr(self, 'inpaint_tiling_enabled'): self.inpaint_tiling_enabled.set(bool(pre.get('inpaint_tiling_enabled', False)))
            if hasattr(self, 'inpaint_tile_size'): self.inpaint_tile_size.set(int(pre.get('inpaint_tile_size', 512)))
            if hasattr(self, 'inpaint_tile_overlap'): self.inpaint_tile_overlap.set(int(pre.get('inpaint_tile_overlap', 64)))

            # OCR basic
            if hasattr(self, 'confidence_threshold'): self.confidence_threshold.set(float(ocr.get('confidence_threshold', 0.7)))
            if hasattr(self, 'detection_mode'): self.detection_mode.set(str(ocr.get('text_detection_mode', 'document')))
            if hasattr(self, 'merge_nearby_threshold'): self.merge_nearby_threshold.set(int(ocr.get('merge_nearby_threshold', 20)))
            if hasattr(self, 'enable_rotation'): self.enable_rotation.set(bool(ocr.get('enable_rotation_correction', True)))

            # Language checkboxes
            try:
                if hasattr(self, 'lang_vars') and isinstance(self.lang_vars, dict):
                    langs = set(ocr.get('language_hints', ['ja', 'ko', 'zh']))
                    for code, var in self.lang_vars.items():
                        var.set(code in langs)
            except Exception:
                pass

            # OCR batching/locality
            if hasattr(self, 'ocr_batch_enabled_var'): self.ocr_batch_enabled_var.set(bool(ocr.get('ocr_batch_enabled', True)))
            if hasattr(self, 'ocr_batch_size_var'): self.ocr_batch_size_var.set(int(ocr.get('ocr_batch_size', 8)))
            if hasattr(self, 'ocr_max_conc_var'): self.ocr_max_conc_var.set(int(ocr.get('ocr_max_concurrency', 2)))
            if hasattr(self, 'roi_locality_var'): self.roi_locality_var.set(bool(ocr.get('roi_locality_enabled', False)))
            if hasattr(self, 'roi_padding_ratio_var'): self.roi_padding_ratio_var.set(float(ocr.get('roi_padding_ratio', 0.08)))
            if hasattr(self, 'roi_min_side_var'): self.roi_min_side_var.set(int(ocr.get('roi_min_side_px', 12)))
            if hasattr(self, 'roi_min_area_var'): self.roi_min_area_var.set(int(ocr.get('roi_min_area_px', 100)))
            if hasattr(self, 'roi_max_side_var'): self.roi_max_side_var.set(int(ocr.get('roi_max_side', 0)))

            # English filters
            if hasattr(self, 'exclude_english_var'): self.exclude_english_var.set(bool(ocr.get('exclude_english_text', False)))
            if hasattr(self, 'english_exclude_threshold'): self.english_exclude_threshold.set(float(ocr.get('english_exclude_threshold', 0.7)))
            if hasattr(self, 'english_exclude_min_chars'): self.english_exclude_min_chars.set(int(ocr.get('english_exclude_min_chars', 4)))
            if hasattr(self, 'english_exclude_short_tokens'): self.english_exclude_short_tokens.set(bool(ocr.get('english_exclude_short_tokens', False)))

            # Azure
            if hasattr(self, 'azure_merge_multiplier'): self.azure_merge_multiplier.set(float(ocr.get('azure_merge_multiplier', 3.0)))
            if hasattr(self, 'azure_reading_order'): self.azure_reading_order.set(str(ocr.get('azure_reading_order', 'natural')))
            if hasattr(self, 'azure_model_version'): self.azure_model_version.set(str(ocr.get('azure_model_version', 'latest')))
            if hasattr(self, 'azure_max_wait'): self.azure_max_wait.set(int(ocr.get('azure_max_wait', 60)))
            if hasattr(self, 'azure_poll_interval'): self.azure_poll_interval.set(float(ocr.get('azure_poll_interval', 0.5)))
            try:
                self._update_azure_label()
            except Exception:
                pass

            # Bubble detector
            if hasattr(self, 'bubble_detection_enabled'): self.bubble_detection_enabled.set(bool(ocr.get('bubble_detection_enabled', False)))
            # Detector type mapping to UI labels
            if hasattr(self, 'detector_type'):
                dt = str(ocr.get('detector_type', 'rtdetr_onnx'))
                if dt == 'rtdetr_onnx': self.detector_type.set('RTEDR_onnx')
                elif dt == 'rtdetr': self.detector_type.set('RT-DETR')
                elif dt == 'yolo': self.detector_type.set('YOLOv8 Speech')
                elif dt == 'custom': self.detector_type.set('Custom Model')
                else: self.detector_type.set('RTEDR_onnx')
            if hasattr(self, 'bubble_model_path'): self.bubble_model_path.set(str(ocr.get('bubble_model_path', '')))
            if hasattr(self, 'bubble_confidence'): self.bubble_confidence.set(float(ocr.get('bubble_confidence', 0.5)))
            if hasattr(self, 'detect_empty_bubbles'): self.detect_empty_bubbles.set(bool(ocr.get('detect_empty_bubbles', True)))
            if hasattr(self, 'detect_text_bubbles'): self.detect_text_bubbles.set(bool(ocr.get('detect_text_bubbles', True)))
            if hasattr(self, 'detect_free_text'): self.detect_free_text.set(bool(ocr.get('detect_free_text', True)))
            if hasattr(self, 'bubble_max_det_yolo_var'): self.bubble_max_det_yolo_var.set(int(ocr.get('bubble_max_detections_yolo', 100)))

            # Inpainting
            if hasattr(self, 'inpaint_batch_size'): self.inpaint_batch_size.set(int(inp.get('batch_size', 1)))
            if hasattr(self, 'enable_cache_var'): self.enable_cache_var.set(bool(inp.get('enable_cache', True)))
            if hasattr(self, 'mask_dilation_var'): self.mask_dilation_var.set(int(s.get('mask_dilation', 0)))
            if hasattr(self, 'use_all_iterations_var'): self.use_all_iterations_var.set(bool(s.get('use_all_iterations', True)))
            if hasattr(self, 'all_iterations_var'): self.all_iterations_var.set(int(s.get('all_iterations', 2)))
            if hasattr(self, 'text_bubble_iterations_var'): self.text_bubble_iterations_var.set(int(s.get('text_bubble_dilation_iterations', 2)))
            if hasattr(self, 'empty_bubble_iterations_var'): self.empty_bubble_iterations_var.set(int(s.get('empty_bubble_dilation_iterations', 3)))
            if hasattr(self, 'free_text_iterations_var'): self.free_text_iterations_var.set(int(s.get('free_text_dilation_iterations', 0)))

            # Advanced
            if hasattr(self, 'format_detection'): self.format_detection.set(1 if adv.get('format_detection', True) else 0)
            if hasattr(self, 'webtoon_mode'): self.webtoon_mode.set(str(adv.get('webtoon_mode', 'auto')))
            if hasattr(self, 'debug_mode'): self.debug_mode.set(1 if adv.get('debug_mode', False) else 0)
            if hasattr(self, 'save_intermediate'): self.save_intermediate.set(1 if adv.get('save_intermediate', False) else 0)
            if hasattr(self, 'parallel_processing'): self.parallel_processing.set(1 if adv.get('parallel_processing', False) else 0)
            if hasattr(self, 'max_workers'): self.max_workers.set(int(adv.get('max_workers', 4)))
            if hasattr(self, 'use_singleton_models'): self.use_singleton_models.set(bool(adv.get('use_singleton_models', True)))
            if hasattr(self, 'auto_cleanup_models'): self.auto_cleanup_models.set(bool(adv.get('auto_cleanup_models', False)))
            if hasattr(self, 'unload_models_var'): self.unload_models_var.set(bool(adv.get('unload_models_after_translation', False)))
            if hasattr(self, 'parallel_panel_var'): self.parallel_panel_var.set(bool(adv.get('parallel_panel_translation', False)))
            if hasattr(self, 'panel_max_workers_var'): self.panel_max_workers_var.set(int(adv.get('panel_max_workers', 2)))
            if hasattr(self, 'panel_stagger_ms_var'): self.panel_stagger_ms_var.set(int(adv.get('panel_start_stagger_ms', 30)))
            # New: preload local inpainting for parallel panels (default True)
            if hasattr(self, 'preload_local_panels_var'): self.preload_local_panels_var.set(bool(adv.get('preload_local_inpainting_for_panels', True)))
            if hasattr(self, 'auto_convert_onnx_var'): self.auto_convert_onnx_var.set(bool(adv.get('auto_convert_to_onnx', False)))
            if hasattr(self, 'auto_convert_onnx_bg_var'): self.auto_convert_onnx_bg_var.set(bool(adv.get('auto_convert_to_onnx_background', True)))
            if hasattr(self, 'quantize_models_var'): self.quantize_models_var.set(bool(adv.get('quantize_models', False)))
            if hasattr(self, 'onnx_quantize_var'): self.onnx_quantize_var.set(bool(adv.get('onnx_quantize', False)))
            if hasattr(self, 'torch_precision_var'): self.torch_precision_var.set(str(adv.get('torch_precision', 'auto')))

            # Font sizing tab
            if hasattr(self, 'font_algorithm_var'): self.font_algorithm_var.set(str(font.get('algorithm', 'smart')))
            if hasattr(self, 'min_font_size_var'): self.min_font_size_var.set(int(font.get('min_size', 10)))
            if hasattr(self, 'max_font_size_var'): self.max_font_size_var.set(int(font.get('max_size', 40)))
            if hasattr(self, 'min_readable_var'): self.min_readable_var.set(int(font.get('min_readable', 14)))
            if hasattr(self, 'prefer_larger_var'): self.prefer_larger_var.set(bool(font.get('prefer_larger', True)))
            if hasattr(self, 'bubble_size_factor_var'): self.bubble_size_factor_var.set(bool(font.get('bubble_size_factor', True)))
            if hasattr(self, 'line_spacing_var'): self.line_spacing_var.set(float(font.get('line_spacing', 1.3)))
            if hasattr(self, 'max_lines_var'): self.max_lines_var.set(int(font.get('max_lines', 10)))
            try:
                if hasattr(self, '_on_font_mode_change'):
                    self._on_font_mode_change()
            except Exception:
                pass

            # Rendering controls (if present in this dialog)
            if hasattr(self, 'font_size_mode_var'): self.font_size_mode_var.set(str(s.get('rendering', {}).get('font_size_mode', 'auto')))
            if hasattr(self, 'fixed_font_size_var'): self.fixed_font_size_var.set(int(s.get('rendering', {}).get('fixed_font_size', 16)))
            if hasattr(self, 'font_scale_var'): self.font_scale_var.set(float(s.get('rendering', {}).get('font_scale', 1.0)))
            if hasattr(self, 'auto_fit_style_var'): self.auto_fit_style_var.set(str(s.get('rendering', {}).get('auto_fit_style', 'balanced')))

            # Cloud API tab
            if hasattr(self, 'cloud_model_var'): self.cloud_model_var.set(str(s.get('cloud_inpaint_model', 'ideogram-v2')))
            if hasattr(self, 'custom_version_var'): self.custom_version_var.set(str(s.get('cloud_custom_version', '')))
            if hasattr(self, 'cloud_prompt_var'): self.cloud_prompt_var.set(str(s.get('cloud_inpaint_prompt', 'clean background, smooth surface')))
            if hasattr(self, 'cloud_negative_prompt_var'): self.cloud_negative_prompt_var.set(str(s.get('cloud_negative_prompt', 'text, writing, letters')))
            if hasattr(self, 'cloud_steps_var'): self.cloud_steps_var.set(int(s.get('cloud_inference_steps', 20)))
            if hasattr(self, 'cloud_timeout_var'): self.cloud_timeout_var.set(int(s.get('cloud_timeout', 60)))

            # Trigger dependent UI updates
            try:
                self._toggle_preprocessing()
            except Exception:
                pass
            try:
                if hasattr(self, '_on_cloud_model_change'):
                    self._on_cloud_model_change()
            except Exception:
                pass
            try:
                self._toggle_iteration_controls()
            except Exception:
                pass
            try:
                self._toggle_roi_locality_controls()
            except Exception:
                pass
            try:
                self._toggle_workers()
            except Exception:
                pass
            
            # Build/attach advanced control for local inpainting preload if not present
            try:
                if not hasattr(self, 'preload_local_panels_var') and hasattr(self, '_create_advanced_tab_ui'):
                    # If there is a helper to build advanced UI, we rely on it. Otherwise, attach to existing advanced frame if available.
                    pass
            except Exception:
                pass
            try:
                if hasattr(self, 'compression_format_combo'):
                    self._toggle_compression_format()
            except Exception:
                pass
            try:
                if hasattr(self, 'detector_type'):
                    self._on_detector_type_changed()
            except Exception:
                pass
            try:
                self.dialog.update_idletasks()
            except Exception:
                pass
        except Exception:
            # Best-effort application only
            pass


    def _set_font_preset(self, preset: str):
        """Apply font sizing preset"""
        if preset == 'small':
            # For manga with small bubbles
            self.font_algorithm_var.set('conservative')
            self.min_font_size_var.set(8)
            self.max_font_size_var.set(24)
            self.min_readable_var.set(12)
            self.prefer_larger_var.set(False)
            self.bubble_size_factor_var.set(True)
            self.line_spacing_var.set(1.2)
            self.max_lines_var.set(8)
        elif preset == 'balanced':
            # Default balanced settings
            self.font_algorithm_var.set('smart')
            self.min_font_size_var.set(10)
            self.max_font_size_var.set(40)
            self.min_readable_var.set(14)
            self.prefer_larger_var.set(True)
            self.bubble_size_factor_var.set(True)
            self.line_spacing_var.set(1.3)
            self.max_lines_var.set(10)
        elif preset == 'large':
            # For maximum readability
            self.font_algorithm_var.set('aggressive')
            self.min_font_size_var.set(14)
            self.max_font_size_var.set(50)
            self.min_readable_var.set(16)
            self.prefer_larger_var.set(True)
            self.bubble_size_factor_var.set(False)
            self.line_spacing_var.set(1.4)
            self.max_lines_var.set(12)
    

    def _save_rendering_settings(self, *args):
        """Auto-save font and rendering settings when controls change"""
        # Don't save during initialization
        if hasattr(self, '_initializing') and self._initializing:
            return
        
        try:
            # Ensure rendering section exists in settings
            if 'rendering' not in self.settings:
                self.settings['rendering'] = {}
            
            # Save font size controls if they exist
            if hasattr(self, 'font_size_mode_var'):
                self.settings['rendering']['font_size_mode'] = self.font_size_mode_var.get()
                self.settings['rendering']['fixed_font_size'] = self.fixed_font_size_var.get()
                self.settings['rendering']['font_scale'] = self.font_scale_var.get()
                self.settings['rendering']['auto_fit_style'] = self.auto_fit_style_var.get()
                
                # Save min/max for auto mode
                if hasattr(self, 'min_font_size_var'):
                    self.settings['rendering']['auto_min_size'] = self.min_font_size_var.get()
                if hasattr(self, 'max_font_size_var'):
                    self.settings['rendering']['auto_max_size'] = self.max_font_size_var.get()
            
            # Update config
            self.config['manga_settings'] = self.settings

            # Mirror only auto max to top-level config for backward compatibility; keep min nested
            try:
                auto_max = self.settings.get('rendering', {}).get('auto_max_size', None)
                if auto_max is not None:
                    self.config['manga_max_font_size'] = int(auto_max)
            except Exception:
                pass
            
            # Save to file immediately
            if hasattr(self.main_gui, 'save_config'):
                self.main_gui.save_config()
                print(f"Auto-saved rendering settings")
                time.sleep(0.1)  # Brief pause for stability
                print("💤 Auto-save pausing briefly for stability")
            
        except Exception as e:
            print(f"Error auto-saving rendering settings: {e}")

    def _save_settings(self):
        """Save all settings including expanded iteration controls"""
        try:
            # Collect all preprocessing settings
            self.settings['preprocessing']['enabled'] = self.preprocess_enabled.get()
            self.settings['preprocessing']['auto_detect_quality'] = self.auto_detect.get()
            self.settings['preprocessing']['contrast_threshold'] = self.contrast_threshold.get()
            self.settings['preprocessing']['sharpness_threshold'] = self.sharpness_threshold.get()
            self.settings['preprocessing']['enhancement_strength'] = self.enhancement_strength.get()
            self.settings['preprocessing']['noise_threshold'] = self.noise_threshold.get()
            self.settings['preprocessing']['denoise_strength'] = self.denoise_strength.get()
            self.settings['preprocessing']['max_image_dimension'] = self.max_dimension.get()
            self.settings['preprocessing']['max_image_pixels'] = self.max_pixels.get()
            self.settings['preprocessing']['chunk_height'] = self.chunk_height.get()
            self.settings['preprocessing']['chunk_overlap'] = self.chunk_overlap.get()
            # Compression (saved separately from preprocessing)
            if 'compression' not in self.settings:
                self.settings['compression'] = {}
            self.settings['compression']['enabled'] = bool(self.compression_enabled_var.get()) if hasattr(self, 'compression_enabled_var') else False
            self.settings['compression']['format'] = str(self.compression_format_var.get()) if hasattr(self, 'compression_format_var') else 'jpeg'
            self.settings['compression']['jpeg_quality'] = int(self.jpeg_quality_var.get()) if hasattr(self, 'jpeg_quality_var') else 85
            self.settings['compression']['png_compress_level'] = int(self.png_level_var.get()) if hasattr(self, 'png_level_var') else 6
            self.settings['compression']['webp_quality'] = int(self.webp_quality_var.get()) if hasattr(self, 'webp_quality_var') else 85
            # TILING SETTINGS - save under preprocessing (primary) and mirror under 'tiling' for backward compatibility
            self.settings['preprocessing']['inpaint_tiling_enabled'] = self.inpaint_tiling_enabled.get()
            self.settings['preprocessing']['inpaint_tile_size'] = self.inpaint_tile_size.get()
            self.settings['preprocessing']['inpaint_tile_overlap'] = self.inpaint_tile_overlap.get()
            # Back-compat mirror
            self.settings['tiling'] = {
                'enabled': self.inpaint_tiling_enabled.get(),
                'tile_size': self.inpaint_tile_size.get(),
                'tile_overlap': self.inpaint_tile_overlap.get()
            }
            
            # OCR settings
            self.settings['ocr']['language_hints'] = [code for code, var in self.lang_vars.items() if var.get()]
            self.settings['ocr']['confidence_threshold'] = self.confidence_threshold.get()
            self.settings['ocr']['text_detection_mode'] = self.detection_mode.get()
            self.settings['ocr']['merge_nearby_threshold'] = self.merge_nearby_threshold.get()
            self.settings['ocr']['enable_rotation_correction'] = self.enable_rotation.get()
            self.settings['ocr']['azure_merge_multiplier'] = self.azure_merge_multiplier.get()
            self.settings['ocr']['azure_reading_order'] = self.azure_reading_order.get()
            self.settings['ocr']['azure_model_version'] = self.azure_model_version.get()
            self.settings['ocr']['azure_max_wait'] = self.azure_max_wait.get()
            self.settings['ocr']['azure_poll_interval'] = self.azure_poll_interval.get()
            self.settings['ocr']['min_text_length'] = self.min_text_length_var.get()
            self.settings['ocr']['exclude_english_text'] = self.exclude_english_var.get()
            self.settings['ocr']['roi_locality_enabled'] = bool(self.roi_locality_var.get()) if hasattr(self, 'roi_locality_var') else True
            # OCR batching & locality
            self.settings['ocr']['ocr_batch_enabled'] = bool(self.ocr_batch_enabled_var.get()) if hasattr(self, 'ocr_batch_enabled_var') else True
            self.settings['ocr']['ocr_batch_size'] = int(self.ocr_batch_size_var.get()) if hasattr(self, 'ocr_batch_size_var') else 8
            self.settings['ocr']['ocr_max_concurrency'] = int(self.ocr_max_conc_var.get()) if hasattr(self, 'ocr_max_conc_var') else 2
            self.settings['ocr']['roi_padding_ratio'] = float(self.roi_padding_ratio_var.get()) if hasattr(self, 'roi_padding_ratio_var') else 0.08
            self.settings['ocr']['roi_min_side_px'] = int(self.roi_min_side_var.get()) if hasattr(self, 'roi_min_side_var') else 12
            self.settings['ocr']['roi_min_area_px'] = int(self.roi_min_area_var.get()) if hasattr(self, 'roi_min_area_var') else 100
            self.settings['ocr']['roi_max_side'] = int(self.roi_max_side_var.get()) if hasattr(self, 'roi_max_side_var') else 0
            self.settings['ocr']['english_exclude_threshold'] = self.english_exclude_threshold.get()
            self.settings['ocr']['english_exclude_min_chars'] = self.english_exclude_min_chars.get()
            self.settings['ocr']['english_exclude_short_tokens'] = self.english_exclude_short_tokens.get()
            
            # Bubble detection settings
            self.settings['ocr']['bubble_detection_enabled'] = self.bubble_detection_enabled.get()
            self.settings['ocr']['bubble_model_path'] = self.bubble_model_path.get()
            self.settings['ocr']['bubble_confidence'] = self.bubble_confidence.get()
            self.settings['ocr']['rtdetr_confidence'] = self.bubble_confidence.get()
            self.settings['ocr']['detect_empty_bubbles'] = self.detect_empty_bubbles.get()
            self.settings['ocr']['detect_text_bubbles'] = self.detect_text_bubbles.get()
            self.settings['ocr']['detect_free_text'] = self.detect_free_text.get()
            self.settings['ocr']['rtdetr_model_url'] = self.bubble_model_path.get()
            self.settings['ocr']['bubble_max_detections_yolo'] = int(self.bubble_max_det_yolo_var.get())
            
            # Save the detector type properly
            if hasattr(self, 'detector_type'):
                detector_display = self.detector_type.get()
                if 'RTEDR_onnx' in detector_display or 'ONNX' in detector_display.upper():
                    self.settings['ocr']['detector_type'] = 'rtdetr_onnx'
                elif 'RT-DETR' in detector_display:
                    self.settings['ocr']['detector_type'] = 'rtdetr'
                elif 'YOLOv8' in detector_display:
                    self.settings['ocr']['detector_type'] = 'yolo'
                elif detector_display == 'Custom Model':
                    self.settings['ocr']['detector_type'] = 'custom'
                    self.settings['ocr']['custom_model_path'] = self.bubble_model_path.get()
                else:
                    self.settings['ocr']['detector_type'] = 'rtdetr_onnx'
            
            # Inpainting settings
            if hasattr(self, 'inpaint_batch_size'):
                if 'inpainting' not in self.settings:
                    self.settings['inpainting'] = {}
                self.settings['inpainting']['batch_size'] = self.inpaint_batch_size.get()
                self.settings['inpainting']['enable_cache'] = self.enable_cache_var.get()
                
                # Save all dilation settings
                self.settings['mask_dilation'] = self.mask_dilation_var.get()
                self.settings['use_all_iterations'] = self.use_all_iterations_var.get()
                self.settings['all_iterations'] = self.all_iterations_var.get()
                self.settings['text_bubble_dilation_iterations'] = self.text_bubble_iterations_var.get()
                self.settings['empty_bubble_dilation_iterations'] = self.empty_bubble_iterations_var.get()
                self.settings['free_text_dilation_iterations'] = self.free_text_iterations_var.get()
                self.settings['auto_iterations'] = self.auto_iterations_var.get()
                
                # Legacy support
                self.settings['bubble_dilation_iterations'] = self.text_bubble_iterations_var.get()
                self.settings['dilation_iterations'] = self.text_bubble_iterations_var.get()
            
            # Advanced settings
            self.settings['advanced']['format_detection'] = bool(self.format_detection.get())
            self.settings['advanced']['webtoon_mode'] = self.webtoon_mode.get()
            self.settings['advanced']['debug_mode'] = bool(self.debug_mode.get())
            self.settings['advanced']['save_intermediate'] = bool(self.save_intermediate.get())
            self.settings['advanced']['parallel_processing'] = bool(self.parallel_processing.get())
            self.settings['advanced']['max_workers'] = self.max_workers.get()
            
            # Save HD strategy settings
            try:
                self.settings['advanced']['hd_strategy'] = str(self.hd_strategy_var.get())
                self.settings['advanced']['hd_strategy_resize_limit'] = int(self.hd_resize_limit_var.get())
                self.settings['advanced']['hd_strategy_crop_margin'] = int(self.hd_crop_margin_var.get())
                self.settings['advanced']['hd_strategy_crop_trigger_size'] = int(self.hd_crop_trigger_var.get())
                # Also reflect into environment for immediate effect in this session
                os.environ['HD_STRATEGY'] = self.settings['advanced']['hd_strategy']
                os.environ['HD_RESIZE_LIMIT'] = str(self.settings['advanced']['hd_strategy_resize_limit'])
                os.environ['HD_CROP_MARGIN'] = str(self.settings['advanced']['hd_strategy_crop_margin'])
                os.environ['HD_CROP_TRIGGER'] = str(self.settings['advanced']['hd_strategy_crop_trigger_size'])
            except Exception:
                pass
            
            # Save parallel rendering toggle
            if hasattr(self, 'render_parallel_var'):
                self.settings['advanced']['render_parallel'] = bool(self.render_parallel_var.get())
            # Panel-level parallel translation settings
            self.settings['advanced']['parallel_panel_translation'] = bool(self.parallel_panel_var.get())
            self.settings['advanced']['panel_max_workers'] = int(self.panel_max_workers_var.get())
            self.settings['advanced']['panel_start_stagger_ms'] = int(self.panel_stagger_ms_var.get())
            # New: preload local inpainting for panels
            if hasattr(self, 'preload_local_panels_var'):
                self.settings['advanced']['preload_local_inpainting_for_panels'] = bool(self.preload_local_panels_var.get())
            
            # Memory management settings
            self.settings['advanced']['use_singleton_models'] = bool(self.use_singleton_models.get())
            self.settings['advanced']['auto_cleanup_models'] = bool(self.auto_cleanup_models.get())
            self.settings['advanced']['unload_models_after_translation'] = bool(getattr(self, 'unload_models_var', tk.BooleanVar(value=False)).get())
            
            # ONNX auto-convert settings (persist and apply to environment)
            if hasattr(self, 'auto_convert_onnx_var'):
                self.settings['advanced']['auto_convert_to_onnx'] = bool(self.auto_convert_onnx_var.get())
                os.environ['AUTO_CONVERT_TO_ONNX'] = 'true' if self.auto_convert_onnx_var.get() else 'false'
            if hasattr(self, 'auto_convert_onnx_bg_var'):
                self.settings['advanced']['auto_convert_to_onnx_background'] = bool(self.auto_convert_onnx_bg_var.get())
                os.environ['AUTO_CONVERT_TO_ONNX_BACKGROUND'] = 'true' if self.auto_convert_onnx_bg_var.get() else 'false'
            
            # Quantization toggles and precision
            if hasattr(self, 'quantize_models_var'):
                self.settings['advanced']['quantize_models'] = bool(self.quantize_models_var.get())
                os.environ['MODEL_QUANTIZE'] = 'true' if self.quantize_models_var.get() else 'false'
            if hasattr(self, 'onnx_quantize_var'):
                self.settings['advanced']['onnx_quantize'] = bool(self.onnx_quantize_var.get())
                os.environ['ONNX_QUANTIZE'] = 'true' if self.onnx_quantize_var.get() else 'false'
            if hasattr(self, 'torch_precision_var'):
                self.settings['advanced']['torch_precision'] = str(self.torch_precision_var.get())
                os.environ['TORCH_PRECISION'] = self.settings['advanced']['torch_precision']
            
            # Memory cleanup toggle
            if hasattr(self, 'force_deep_cleanup_var'):
                if 'advanced' not in self.settings:
                    self.settings['advanced'] = {}
                self.settings['advanced']['force_deep_cleanup_each_image'] = bool(self.force_deep_cleanup_var.get())
            # RAM cap settings
            if hasattr(self, 'ram_cap_enabled_var'):
                self.settings['advanced']['ram_cap_enabled'] = bool(self.ram_cap_enabled_var.get())
            if hasattr(self, 'ram_cap_mb_var'):
                try:
                    self.settings['advanced']['ram_cap_mb'] = int(self.ram_cap_mb_var.get())
                except Exception:
                    self.settings['advanced']['ram_cap_mb'] = 0
            if hasattr(self, 'ram_cap_mode_var'):
                mode = self.ram_cap_mode_var.get()
                if mode not in ['soft', 'hard (Windows only)']:
                    mode = 'soft'
                # Normalize to 'soft' or 'hard'
                self.settings['advanced']['ram_cap_mode'] = 'hard' if mode.startswith('hard') else 'soft'
            if hasattr(self, 'ram_gate_timeout_var'):
                try:
                    self.settings['advanced']['ram_gate_timeout_sec'] = float(self.ram_gate_timeout_var.get())
                except Exception:
                    self.settings['advanced']['ram_gate_timeout_sec'] = 10.0
            if hasattr(self, 'ram_gate_floor_var'):
                try:
                    self.settings['advanced']['ram_min_floor_over_baseline_mb'] = int(self.ram_gate_floor_var.get())
                except Exception:
                    self.settings['advanced']['ram_min_floor_over_baseline_mb'] = 128
            
            # Cloud API settings
            if hasattr(self, 'cloud_model_var'):
                self.settings['cloud_inpaint_model'] = self.cloud_model_var.get()
                self.settings['cloud_custom_version'] = self.custom_version_var.get()
                self.settings['cloud_inpaint_prompt'] = self.cloud_prompt_var.get()
                self.settings['cloud_negative_prompt'] = self.cloud_negative_prompt_var.get()
                self.settings['cloud_inference_steps'] = self.cloud_steps_var.get()
                self.settings['cloud_timeout'] = self.cloud_timeout_var.get()
            
            # Font sizing settings from Font Sizing tab
            if hasattr(self, 'font_algorithm_var'):
                if 'font_sizing' not in self.settings:
                    self.settings['font_sizing'] = {}
                self.settings['font_sizing']['algorithm'] = self.font_algorithm_var.get()
                self.settings['font_sizing']['min_size'] = self.min_font_size_var.get()
                self.settings['font_sizing']['max_size'] = self.max_font_size_var.get()
                self.settings['font_sizing']['min_readable'] = self.min_readable_var.get()
                self.settings['font_sizing']['prefer_larger'] = self.prefer_larger_var.get()
                self.settings['font_sizing']['bubble_size_factor'] = self.bubble_size_factor_var.get()
                self.settings['font_sizing']['line_spacing'] = self.line_spacing_var.get()
                self.settings['font_sizing']['max_lines'] = self.max_lines_var.get()
            
            # SAVE FONT SIZE CONTROLS FROM RENDERING (if they exist)
            if hasattr(self, 'font_size_mode_var'):
                if 'rendering' not in self.settings:
                    self.settings['rendering'] = {}
                
                self.settings['rendering']['font_size_mode'] = self.font_size_mode_var.get()
                self.settings['rendering']['fixed_font_size'] = self.fixed_font_size_var.get()
                self.settings['rendering']['font_scale'] = self.font_scale_var.get()
                self.settings['rendering']['auto_min_size'] = self.min_font_size_var.get() if hasattr(self, 'min_font_size_var') else 10
                self.settings['rendering']['auto_max_size'] = self.max_font_size_var.get() if hasattr(self, 'max_font_size_var') else 28
                self.settings['rendering']['auto_fit_style'] = self.auto_fit_style_var.get()
            
            # Clear bubble detector cache to force reload with new settings
            if hasattr(self.main_gui, 'manga_tab') and hasattr(self.main_gui.manga_tab, 'translator'):
                if hasattr(self.main_gui.manga_tab.translator, 'bubble_detector'):
                    self.main_gui.manga_tab.translator.bubble_detector = None
            
            # Save to config
            self.config['manga_settings'] = self.settings
            
            # Save to file - using the correct method name
            try:
                if hasattr(self.main_gui, 'save_config'):
                    self.main_gui.save_config()
                    print("Settings saved successfully via save_config")
                    time.sleep(0.1)  # Brief pause for stability
                    print("💤 Main settings save pausing briefly for stability")
                elif hasattr(self.main_gui, 'save_configuration'):
                    self.main_gui.save_configuration()
                    print("Settings saved successfully via save_configuration")
                else:
                    print("Warning: No save method found on main_gui")
                    # Try direct save as fallback
                    if hasattr(self.main_gui, 'config_file'):
                        import json
                        with open(self.main_gui.config_file, 'w') as f:
                            json.dump(self.config, f, indent=2)
                        print("Settings saved directly to config file")
            except Exception as e:
                print(f"Error saving configuration: {e}")
                from tkinter import messagebox
                messagebox.showerror("Save Error", f"Failed to save settings: {e}")
            
            # Call callback if provided
            if self.callback:
                try:
                    self.callback(self.settings)
                except Exception as e:
                    print(f"Error in callback: {e}")
            
            # Close dialog with cleanup
            try:
                if hasattr(self.dialog, '_cleanup_scrolling'):
                    self.dialog._cleanup_scrolling()
                self.dialog.destroy()
            except Exception as e:
                print(f"Error closing dialog: {e}")
                self.dialog.destroy()
                
        except Exception as e:
            print(f"Critical error in _save_settings: {e}")
            from tkinter import messagebox
            messagebox.showerror("Save Error", f"Failed to save settings: {e}")

    def _reset_defaults(self):
        """Reset by removing manga_settings from config and reinitializing the dialog."""
        from tkinter import messagebox
        if not messagebox.askyesno("Reset Settings", "Reset all manga settings to defaults?\nThis will remove custom manga settings from config.json."):
            return
        # Remove manga_settings key to force defaults
        try:
            if isinstance(self.config, dict) and 'manga_settings' in self.config:
                del self.config['manga_settings']
        except Exception:
            pass
        # Persist changes
        try:
            if hasattr(self.main_gui, 'save_config'):
                self.main_gui.save_config()
            elif hasattr(self.main_gui, 'save_configuration'):
                self.main_gui.save_configuration()
            elif hasattr(self.main_gui, 'config_file') and isinstance(self.main_gui.config_file, str):
                with open(self.main_gui.config_file, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception:
            try:
                if hasattr(self.main_gui, 'CONFIG_FILE') and isinstance(self.main_gui.CONFIG_FILE, str):
                    with open(self.main_gui.CONFIG_FILE, 'w', encoding='utf-8') as f:
                        import json
                        json.dump(self.config, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        # Close and reopen dialog so defaults apply
        try:
            if hasattr(self.dialog, '_cleanup_scrolling'):
                self.dialog._cleanup_scrolling()
        except Exception:
            pass
        try:
            self.dialog.destroy()
        except Exception:
            pass
        try:
            MangaSettingsDialog(parent=self.parent, main_gui=self.main_gui, config=self.config, callback=self.callback)
        except Exception:
            try:
                messagebox.showinfo("Reset", "Settings reset. Please reopen the dialog.")
            except Exception:
                pass

    def _cancel(self):
        """Cancel without saving"""
        if hasattr(self.dialog, '_cleanup_scrolling'):
            self.dialog._cleanup_scrolling()
        self.dialog.destroy()

