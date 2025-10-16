# manga_settings_dialog.py
"""
Enhanced settings dialog for manga translation with all settings visible
"""

import os
import sys
import json
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, 
                                QLabel, QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox,
                                QSlider, QComboBox, QLineEdit, QGroupBox, QTabWidget,
                                QWidget, QScrollArea, QFrame, QRadioButton, QButtonGroup,
                                QMessageBox, QFileDialog, QSizePolicy, QApplication,
                                QGraphicsOpacityEffect)
from PySide6.QtCore import Qt, Signal, QTimer, QEvent, QObject, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QIcon, QPixmap, QImage, QPainter, QColor
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
        
        # Make dialog non-modal so it doesn't block the manga integration GUI
        self.setModal(False)
        
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
                'confidence_threshold': 0.0,  # DEFAULT 0.0 (accept all, like comic-translate) to avoid missing text
                'cloud_ocr_confidence': 0.0,  # Explicit default for cloud OCR (Azure/Google)
                'min_region_size': 50,  # Minimum dimension for cloud OCR regions (0 = disabled)
                'merge_nearby_threshold': 20,
                'azure_merge_multiplier': 3.0,
                'text_detection_mode': 'document',
                'enable_rotation_correction': True,
                'bubble_detection_enabled': True,
                'roi_locality_enabled': False,
                'bubble_model_path': '',
                'bubble_confidence': 0.3,
                'detector_type': 'rtdetr_onnx',
                'rtdetr_confidence': 0.3,
                'detect_empty_bubbles': True,
                'detect_text_bubbles': True,
                'detect_free_text': True,
                'rtdetr_model_url': '',
                'use_rtdetr_for_ocr_regions': True,  # On by default for best accuracy
                'enable_fallback_ocr': False,  # Disabled by default - fallback OCR for empty RT-DETR blocks
                # Toggles for RT-DETR behavior customization
                'skip_rtdetr_merging': False,    # Do not merge overlapping RT-DETR regions (manual mode behavior)
                'preserve_empty_blocks': False,  # Keep empty RT-DETR blocks even if OCR found no text
                # Azure settings removed - new API is synchronous, no polling/version settings needed
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
            'dilation_kernel_size': 5,  # Kernel size for dilation operations
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
            
    def _show_styled_messagebox(self, icon, title, text, buttons=QMessageBox.Ok):
        """Show a styled message box with proper button styling"""
        msg_box = QMessageBox(self)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.setStandardButtons(buttons)
        
        # Style the message box buttons
        for button in msg_box.buttons():
            if msg_box.buttonRole(button) == QMessageBox.AcceptRole:
                # OK, Yes, Save, etc. - use green/blue
                button.setStyleSheet("""
                    QPushButton {
                        background-color: #28a745;
                        color: white;
                        font-weight: bold;
                        border: none;
                        border-radius: 4px;
                        padding: 6px 20px;
                        min-width: 80px;
                        font-size: 10pt;
                    }
                    QPushButton:hover {
                        background-color: #34c759;
                    }
                    QPushButton:pressed {
                        background-color: #218838;
                    }
                """)
            elif msg_box.buttonRole(button) == QMessageBox.RejectRole:
                # No, Cancel, etc. - use gray
                button.setStyleSheet("""
                    QPushButton {
                        background-color: #6c757d;
                        color: white;
                        font-weight: bold;
                        border: none;
                        border-radius: 4px;
                        padding: 6px 20px;
                        min-width: 80px;
                        font-size: 10pt;
                    }
                    QPushButton:hover {
                        background-color: #7d8a96;
                    }
                    QPushButton:pressed {
                        background-color: #5a6268;
                    }
                """)
        
        return msg_box.exec()
    
    def _create_styled_checkbox(self, text):
        """Create a checkbox with proper checkmark using text overlay (same as manga_integration.py)"""
        checkbox = QCheckBox(text)
        checkbox.setStyleSheet("""
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
        """)
        
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
        checkmark.setAttribute(Qt.WA_TransparentForMouseEvents)  # Make checkmark click-through
        
        # Position checkmark properly after widget is shown
        def position_checkmark():
            # Position over the checkbox indicator
            checkmark.setGeometry(2, 1, 14, 14)
        
        # Show/hide checkmark based on checked state
        def update_checkmark():
            if checkbox.isChecked():
                position_checkmark()
                checkmark.show()
            else:
                checkmark.hide()
        
        checkbox.stateChanged.connect(update_checkmark)
        # Delay initial positioning to ensure widget is properly rendered
        QTimer.singleShot(0, lambda: (position_checkmark(), update_checkmark()))
        
        return checkbox
    
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
        # Dialog is already non-modal from __init__, don't override it
        
        # Set the halgakos.ico icon
        try:
            icon_path = os.path.join(os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico')
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
        except Exception:
            pass  # Fail silently if icon can't be loaded
            
        # Apply overall dark theme styling
        # Set up icon path
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico')
        
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
                margin-top: 3px;
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
            QComboBox::drop-down:disabled {
                background-color: #252525;
                border-left: 1px solid #3a3a3a;
            }
            QComboBox::down-arrow {
                image: url(""" + icon_path.replace('\\', '/') + """);
                width: 16px;
                height: 16px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                color: white;
                selection-background-color: #5a9fd4;
                selection-color: white;
                border: 1px solid #555;
                outline: none;
            }
            QComboBox QAbstractItemView::item {
                padding: 4px;
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
        # Use availableGeometry to exclude taskbar and other system UI
        app = QApplication.instance()
        if app:
            screen = app.primaryScreen().availableGeometry()
        else:
            screen = self.parent.screen().availableGeometry() if self.parent else self.screen().availableGeometry()
        
        dialog_width = min(800, int(screen.width() * 0.5))
        dialog_height = int(screen.height() * 0.90)  # Use 90% of available height for more screen space with safety margin
        self.resize(dialog_width, dialog_height)
        
        # Center the dialog within available screen space
        dialog_x = screen.x() + (screen.width() - dialog_width) // 2
        dialog_y = screen.y() + (screen.height() - dialog_height) // 2
        self.move(dialog_x, dialog_y)
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create scroll area for the content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Store reference for auto-scrolling in OCR tab
        self.main_scroll_area = scroll_area
        
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
        
        self.preprocess_enabled = self._create_styled_checkbox("Enable Image Preprocessing")
        self.preprocess_enabled.setChecked(self.settings['preprocessing']['enabled'])
        self.preprocess_enabled.toggled.connect(self._toggle_preprocessing)
        enable_layout.addWidget(self.preprocess_enabled)
        
        # Store all preprocessing controls for enable/disable
        self.preprocessing_controls = []
        
        # Auto quality detection
        self.auto_detect = self._create_styled_checkbox("Auto-detect image quality issues")
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
        self.compression_enabled = self._create_styled_checkbox("Enable compression for OCR uploads")
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
        loaded_value = int(self.settings.get('advanced', {}).get('hd_strategy_resize_limit', 1536))
        print(f"[RESIZE_LIMIT_DEBUG] Loading hd_strategy_resize_limit from settings: {loaded_value}")
        self.hd_resize_limit_spin.setValue(loaded_value)
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
            
        self.inpaint_tiling_enabled = self._create_styled_checkbox("Enable automatic tiling for inpainting (processes large images in tiles)")
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
        
        # Auto toggle (affects both mask dilation AND iterations)
        self.auto_iterations_checkbox = self._create_styled_checkbox("Auto Iterations (automatically set values based on OCR provider and B&W vs Color)")
        self.auto_iterations_checkbox.setChecked(self.settings.get('auto_iterations', True))
        self.auto_iterations_checkbox.toggled.connect(self._toggle_iteration_controls)
        self.auto_iterations_checkbox.toggled.connect(self._on_primary_auto_toggle)  # Sync with "Use Same For All"
        mask_layout.addWidget(self.auto_iterations_checkbox)

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
        
        # Kernel size
        kernel_frame = QWidget()
        kernel_layout = QHBoxLayout(kernel_frame)
        kernel_layout.setContentsMargins(0, 0, 0, 0)
        mask_dilation_layout.addWidget(kernel_frame)
        
        self.kernel_size_label = QLabel("Kernel Size:")
        self.kernel_size_label.setMinimumWidth(150)
        kernel_layout.addWidget(self.kernel_size_label)
        
        self.kernel_size_spinbox = QSpinBox()
        self.kernel_size_spinbox.setRange(0, 15)  # Allow 0 to disable dilation
        self.kernel_size_spinbox.setSingleStep(2)  # Only odd numbers (except 0)
        kernel_value = self.settings.get('dilation_kernel_size', 5)
        print(f"[KERNEL_DEBUG] Loading kernel_size from settings: {kernel_value}")
        self.kernel_size_spinbox.setValue(kernel_value)
        kernel_layout.addWidget(self.kernel_size_spinbox)
        
        self.kernel_size_unit_label = QLabel("pixels (0=disable, or odd number for kernel size)")
        kernel_layout.addWidget(self.kernel_size_unit_label)
        kernel_layout.addStretch()
        
        # Per-Text-Type Iterations - EXPANDED SECTION
        iterations_group = QGroupBox("Dilation Iterations Control")
        iterations_layout = QVBoxLayout(iterations_group)
        mask_dilation_layout.addWidget(iterations_group)
        
        # All Iterations Master Control (NEW)
        all_iter_widget = QWidget()
        all_iter_layout = QHBoxLayout(all_iter_widget)
        all_iter_layout.setContentsMargins(0, 0, 0, 0)
        iterations_layout.addWidget(all_iter_widget)
        
        # Checkbox to enable/disable uniform iterations
        self.use_all_iterations_checkbox = self._create_styled_checkbox("Use Same For All:")
        self.use_all_iterations_checkbox.setChecked(self.settings.get('use_all_iterations', True))
        self.use_all_iterations_checkbox.toggled.connect(self._toggle_iteration_controls)
        all_iter_layout.addWidget(self.use_all_iterations_checkbox)
        
        all_iter_layout.addSpacing(10)
        
        self.all_iterations_spinbox = QSpinBox()
        self.all_iterations_spinbox.setRange(0, 5)
        self.all_iterations_spinbox.setValue(self.settings.get('all_iterations', 2))
        self.all_iterations_spinbox.setEnabled(self.use_all_iterations_checkbox.isChecked())
        all_iter_layout.addWidget(self.all_iterations_spinbox)
        
        self.all_iter_label = QLabel("iterations (applies to all text types)")
        all_iter_layout.addWidget(self.all_iter_label)
        all_iter_layout.addStretch()
        
        # Separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setFrameShadow(QFrame.Shadow.Sunken)
        iterations_layout.addWidget(separator1)
        
        # Individual Controls Label
        self.individual_controls_header_label = QLabel("Individual Text Type Controls:")
        individual_label_font = QFont('Arial', 9)
        individual_label_font.setBold(True)
        self.individual_controls_header_label.setFont(individual_label_font)
        iterations_layout.addWidget(self.individual_controls_header_label)
        
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
        
        self.text_bubble_desc = QLabel("iterations (speech/dialogue bubbles)")
        text_bubble_iter_layout.addWidget(self.text_bubble_desc)
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
        
        self.empty_bubble_desc = QLabel("iterations (empty speech bubbles)")
        empty_bubble_iter_layout.addWidget(self.empty_bubble_desc)
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
        
        self.free_text_desc = QLabel("iterations (0 = perfect for B&W panels)")
        free_text_iter_layout.addWidget(self.free_text_desc)
        free_text_iter_layout.addStretch()
        
        # Store individual control widgets for enable/disable (includes descriptive labels)
        self.individual_iteration_controls = [
            (self.text_bubble_label, self.text_bubble_iter_spinbox, self.text_bubble_desc),
            (self.empty_bubble_label, self.empty_bubble_iter_spinbox, self.empty_bubble_desc),
            (self.free_text_label, self.free_text_iter_spinbox, self.free_text_desc)
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
        bw_manga_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a7ca5;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a8cb5;
            }
            QPushButton:pressed {
                background-color: #2a6c95;
            }
        """)
        bw_manga_btn.clicked.connect(lambda: self._set_mask_preset(15, False, 2, 2, 3, 0))
        preset_layout.addWidget(bw_manga_btn)
        
        colored_btn = QPushButton("Colored")
        colored_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a7ca5;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a8cb5;
            }
            QPushButton:pressed {
                background-color: #2a6c95;
            }
        """)
        colored_btn.clicked.connect(lambda: self._set_mask_preset(15, False, 2, 2, 3, 3))
        preset_layout.addWidget(colored_btn)
        
        uniform_btn = QPushButton("Uniform")
        uniform_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a7ca5;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a8cb5;
            }
            QPushButton:pressed {
                background-color: #2a6c95;
            }
        """)
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
        
        main_layout.addStretch()

    def _toggle_iteration_controls(self):
        """Enable/disable iteration controls based on Auto and 'Use Same For All' toggles"""
        # Get auto checkbox state
        auto_on = False
        if hasattr(self, 'auto_iterations_checkbox'):
            auto_on = self.auto_iterations_checkbox.isChecked()
        
        # Get use_all checkbox state
        use_all = False
        if hasattr(self, 'use_all_iterations_checkbox'):
            use_all = self.use_all_iterations_checkbox.isChecked()
        
        # Also update the auto_iterations_enabled attribute
        self.auto_iterations_enabled = auto_on
        
        if auto_on:
            # Disable ALL mask dilation and iteration controls when auto is on
            # Mask dilation controls
            try:
                if hasattr(self, 'mask_dilation_spinbox'):
                    self.mask_dilation_spinbox.setEnabled(False)
            except Exception:
                pass
            try:
                if hasattr(self, 'dilation_label'):
                    self.dilation_label.setEnabled(False)
            except Exception:
                pass
            try:
                if hasattr(self, 'dilation_unit_label'):
                    self.dilation_unit_label.setEnabled(False)
            except Exception:
                pass
            # Kernel size controls
            try:
                if hasattr(self, 'kernel_size_spinbox'):
                    self.kernel_size_spinbox.setEnabled(False)
            except Exception:
                pass
            try:
                if hasattr(self, 'kernel_size_label'):
                    self.kernel_size_label.setEnabled(False)
            except Exception:
                pass
            try:
                if hasattr(self, 'kernel_size_unit_label'):
                    self.kernel_size_unit_label.setEnabled(False)
            except Exception:
                pass
            # Iteration controls
            try:
                self.all_iterations_spinbox.setEnabled(False)
            except Exception:
                pass
            try:
                if hasattr(self, 'all_iter_label'):
                    self.all_iter_label.setEnabled(False)
            except Exception:
                pass
            try:
                if hasattr(self, 'use_all_iterations_checkbox'):
                    self.use_all_iterations_checkbox.setEnabled(False)
            except Exception:
                pass
            try:
                if hasattr(self, 'individual_controls_header_label'):
                    self.individual_controls_header_label.setEnabled(False)
            except Exception:
                pass
            # Disable individual controls and their description labels
            for control_tuple in getattr(self, 'individual_iteration_controls', []):
                try:
                    if len(control_tuple) == 3:
                        label, spinbox, desc_label = control_tuple
                        spinbox.setEnabled(False)
                        label.setEnabled(False)
                        desc_label.setEnabled(False)
                    elif len(control_tuple) == 2:
                        label, spinbox = control_tuple
                        spinbox.setEnabled(False)
                        label.setEnabled(False)
                except Exception:
                    pass
            return
        
        # Auto off -> enable mask dilation (always) and "Use Same For All" (respect its state)
        # Mask dilation is always enabled when auto is off
        try:
            if hasattr(self, 'mask_dilation_spinbox'):
                self.mask_dilation_spinbox.setEnabled(True)
        except Exception:
            pass
        try:
            if hasattr(self, 'dilation_label'):
                self.dilation_label.setEnabled(True)
        except Exception:
            pass
        try:
            if hasattr(self, 'dilation_unit_label'):
                self.dilation_unit_label.setEnabled(True)
        except Exception:
            pass
        # Kernel size is always enabled when auto is off
        try:
            if hasattr(self, 'kernel_size_spinbox'):
                self.kernel_size_spinbox.setEnabled(True)
        except Exception:
            pass
        try:
            if hasattr(self, 'kernel_size_label'):
                self.kernel_size_label.setEnabled(True)
        except Exception:
            pass
        try:
            if hasattr(self, 'kernel_size_unit_label'):
                self.kernel_size_unit_label.setEnabled(True)
        except Exception:
            pass
        
        try:
            if hasattr(self, 'use_all_iterations_checkbox'):
                self.use_all_iterations_checkbox.setEnabled(True)
        except Exception:
            pass
        
        try:
            self.all_iterations_spinbox.setEnabled(use_all)
        except Exception:
            pass
        try:
            if hasattr(self, 'all_iter_label'):
                self.all_iter_label.setEnabled(use_all)
        except Exception:
            pass
        try:
            if hasattr(self, 'individual_controls_header_label'):
                self.individual_controls_header_label.setEnabled(not use_all)
        except Exception:
            pass
        
        # Individual controls respect the "Use Same For All" state
        for control_tuple in getattr(self, 'individual_iteration_controls', []):
            enabled = not use_all
            try:
                if len(control_tuple) == 3:
                    label, spinbox, desc_label = control_tuple
                    spinbox.setEnabled(enabled)
                    label.setEnabled(enabled)
                    desc_label.setEnabled(enabled)
                elif len(control_tuple) == 2:
                    label, spinbox = control_tuple
                    spinbox.setEnabled(enabled)
                    label.setEnabled(enabled)
            except Exception:
                pass
    
    def _on_primary_auto_toggle(self, checked):
        """When primary Auto toggle changes, disable/enable 'Use Same For All' checkbox"""
        if hasattr(self, 'use_all_iterations_checkbox'):
            self.use_all_iterations_checkbox.setEnabled(not checked)

    def _set_mask_preset(self, dilation, use_all, all_iter, text_bubble_iter, empty_bubble_iter, free_text_iter):
        """Set mask dilation preset values with comprehensive iteration controls"""
        self.mask_dilation_spinbox.setValue(dilation)
        self.use_all_iterations_checkbox.setChecked(use_all)
        self.all_iterations_spinbox.setValue(all_iter)
        self.text_bubble_iter_spinbox.setValue(text_bubble_iter)
        self.empty_bubble_iter_spinbox.setValue(empty_bubble_iter)
        self.free_text_iter_spinbox.setValue(free_text_iter)
        self._toggle_iteration_controls()
    
    def _create_cloud_api_tab(self):
            """Create cloud API settings tab"""
            # Create tab widget and add to tab widget
            tab_widget = QWidget()
            self.tab_widget.addTab(tab_widget, "Cloud API")
            
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
            parent_layout = QVBoxLayout(tab_widget)
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
        enabled = self.preprocess_enabled.isChecked()
        
        # Process each control in preprocessing_controls list
        for control in self.preprocessing_controls:
            try:
                if isinstance(control, QGroupBox):
                    # Enable/disable entire group box children
                    self._toggle_frame_children(control, enabled)
                elif isinstance(control, (QSlider, QSpinBox, QCheckBox, QDoubleSpinBox, QComboBox, QLabel)):
                    # Just use setEnabled() - the global stylesheet handles the visual state
                    control.setEnabled(enabled)
            except Exception as e:
                pass
        
        # Ensure tiling fields respect their own toggle regardless of preprocessing state
        try:
            if hasattr(self, '_toggle_tiling_controls'):
                self._toggle_tiling_controls()
        except Exception:
            pass
    
    def _toggle_frame_children(self, widget, enabled):
        """Recursively enable/disable all children of a widget"""
        # Handle all controls including labels - just use setEnabled()
        for child in widget.findChildren(QWidget):
            if isinstance(child, (QSlider, QSpinBox, QCheckBox, QDoubleSpinBox, QComboBox, QLineEdit, QLabel)):
                try:
                    child.setEnabled(enabled)
                except Exception:
                    pass

    def _toggle_roi_locality_controls(self):
        """Show/hide ROI locality controls based on toggle."""
        try:
            enabled = self.roi_locality_checkbox.isChecked()
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
                row.setVisible(enabled)
            except Exception:
                pass

    def _toggle_tiling_controls(self):
        """Enable/disable tiling size/overlap fields based on tiling toggle."""
        try:
            enabled = bool(self.inpaint_tiling_enabled.isChecked())
        except Exception:
            enabled = False
        
        # Enable/disable tiling widgets and their labels
        widgets_to_toggle = [
            ('tile_size_spinbox', 'tile_size_label', 'tile_size_unit_label'),
            ('tile_overlap_spinbox', 'tile_overlap_label', 'tile_overlap_unit_label')
        ]
        
        for widget_names in widgets_to_toggle:
            for widget_name in widget_names:
                try:
                    widget = getattr(self, widget_name, None)
                    if widget is not None:
                        # Just use setEnabled() for everything - stylesheet handles visuals
                        widget.setEnabled(enabled)
                except Exception:
                    pass

    def _on_hd_strategy_change(self):
        """Show/hide HD strategy controls based on selected strategy."""
        try:
            strategy = self.hd_strategy_combo.currentText()
        except Exception:
            strategy = 'original'
        
        # Show/hide resize limit based on strategy
        if hasattr(self, 'hd_resize_frame'):
            self.hd_resize_frame.setVisible(strategy == 'resize')
        
        # Show/hide crop params based on strategy
        if hasattr(self, 'hd_crop_margin_frame'):
            self.hd_crop_margin_frame.setVisible(strategy == 'crop')
        if hasattr(self, 'hd_crop_trigger_frame'):
            self.hd_crop_trigger_frame.setVisible(strategy == 'crop')
    
    def _toggle_compression_enabled(self):
        """Enable/disable compression controls based on compression toggle."""
        try:
            enabled = bool(self.compression_enabled.isChecked())
        except Exception:
            enabled = False
        
        # Enable/disable all compression format controls
        compression_widgets = [
            getattr(self, 'format_label', None),
            getattr(self, 'compression_format_combo', None),
            getattr(self, 'jpeg_frame', None),
            getattr(self, 'jpeg_label', None),
            getattr(self, 'jpeg_quality_spin', None),
            getattr(self, 'jpeg_help', None),
            getattr(self, 'png_frame', None),
            getattr(self, 'png_label', None),
            getattr(self, 'png_level_spin', None),
            getattr(self, 'png_help', None),
            getattr(self, 'webp_frame', None),
            getattr(self, 'webp_label', None),
            getattr(self, 'webp_quality_spin', None),
            getattr(self, 'webp_help', None),
        ]
        
        for widget in compression_widgets:
            try:
                if widget is not None:
                    widget.setEnabled(enabled)
            except Exception:
                pass
    
    def _toggle_compression_format(self):
        """Show only the controls relevant to the selected format (hide others)."""
        fmt = self.compression_format_combo.currentText().lower() if hasattr(self, 'compression_format_combo') else 'jpeg'
        try:
            # Hide all rows first
            for row in [getattr(self, 'jpeg_frame', None), getattr(self, 'png_frame', None), getattr(self, 'webp_frame', None)]:
                try:
                    if row is not None:
                        row.setVisible(False)
                except Exception:
                    pass
            # Show the selected one
            if fmt == 'jpeg':
                if hasattr(self, 'jpeg_frame') and self.jpeg_frame is not None:
                    self.jpeg_frame.setVisible(True)
            elif fmt == 'png':
                if hasattr(self, 'png_frame') and self.png_frame is not None:
                    self.png_frame.setVisible(True)
            else:  # webp
                if hasattr(self, 'webp_frame') and self.webp_frame is not None:
                    self.webp_frame.setVisible(True)
        except Exception:
            pass
    
    def _toggle_ocr_batching_controls(self):
        """Show/hide OCR batching rows based on enable toggle."""
        try:
            enabled = bool(self.ocr_batch_enabled_checkbox.isChecked())
        except Exception:
            enabled = False
        try:
            if hasattr(self, 'ocr_bs_row') and self.ocr_bs_row:
                self.ocr_bs_row.setVisible(enabled)
        except Exception:
            pass
        try:
            if hasattr(self, 'ocr_cc_row') and self.ocr_cc_row:
                self.ocr_cc_row.setVisible(enabled)
        except Exception:
            pass

    def _create_ocr_tab(self):
        """Create OCR settings tab with all options"""
        # Create tab widget and add to tab widget
        tab_widget = QWidget()
        self.tab_widget.addTab(tab_widget, "OCR")
        
        # Create content layout directly (no duplicate scroll area)
        content_layout = QVBoxLayout(tab_widget)
        content_layout.setSpacing(10)
        content_layout.setContentsMargins(20, 20, 20, 20)
        
        # Language hints
        lang_group = QGroupBox("Language Detection")
        lang_layout = QVBoxLayout(lang_group)
        content_layout.addWidget(lang_group)
        
        lang_desc = QLabel("Select languages to prioritize during OCR:")
        lang_desc_font = QFont('Arial', 10)
        lang_desc.setFont(lang_desc_font)
        lang_layout.addWidget(lang_desc)
        lang_layout.addSpacing(10)
        
        # Language checkboxes
        self.lang_checkboxes = {}
        languages = [
            ('ja', 'Japanese'),
            ('ko', 'Korean'),
            ('zh', 'Chinese (Simplified)'),
            ('zh-TW', 'Chinese (Traditional)'),
            ('en', 'English')
        ]
        
        lang_grid_widget = QWidget()
        lang_grid_layout = QGridLayout(lang_grid_widget)
        lang_layout.addWidget(lang_grid_widget)
        
        for i, (code, name) in enumerate(languages):
            checkbox = self._create_styled_checkbox(name)
            checkbox.setChecked(code in self.settings['ocr']['language_hints'])
            self.lang_checkboxes[code] = checkbox
            lang_grid_layout.addWidget(checkbox, i//2, i%2)
        
        # OCR parameters
        ocr_group = QGroupBox("OCR Parameters")
        ocr_layout = QVBoxLayout(ocr_group)
        content_layout.addWidget(ocr_group)
        
        # Cloud OCR Confidence threshold (for Google/Azure only)
        conf_widget = QWidget()
        conf_layout = QHBoxLayout(conf_widget)
        conf_layout.setContentsMargins(0, 0, 0, 0)
        ocr_layout.addWidget(conf_widget)
        
        conf_label = QLabel("☁️ Cloud OCR Confidence:")
        conf_label.setMinimumWidth(180)
        conf_label.setToolTip("Applies to Google Cloud Vision and Azure OCR only.\nLocal OCR (RapidOCR, PaddleOCR, etc.) uses RT-DETR confidence only (comic-translate approach).")
        conf_layout.addWidget(conf_label)
        
        # Get cloud OCR confidence (fallback to old setting for migration)
        # DEFAULT TO 0.0 (accept all, like comic-translate) to avoid missing text
        cloud_conf = self.settings['ocr'].get('cloud_ocr_confidence', self.settings['ocr'].get('confidence_threshold', 0.0))
        
        self.confidence_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_threshold_slider.setRange(0, 100)
        self.confidence_threshold_slider.setValue(int(cloud_conf * 100))
        self.confidence_threshold_slider.setMinimumWidth(250)
        self.confidence_threshold_slider.setToolTip("0 = accept all (recommended, like comic-translate)\nHigher values filter low-confidence cloud OCR results")
        conf_layout.addWidget(self.confidence_threshold_slider)
        
        self.confidence_threshold_label = QLabel(f"{cloud_conf:.2f}")
        self.confidence_threshold_label.setMinimumWidth(50)
        self.confidence_threshold_slider.valueChanged.connect(
            lambda v: self.confidence_threshold_label.setText(f"{v/100:.2f}")
        )
        conf_layout.addWidget(self.confidence_threshold_label)
        conf_layout.addStretch()
        
        # Add info label below slider with icon
        conf_info_widget = QWidget()
        conf_info_layout = QHBoxLayout(conf_info_widget)
        conf_info_layout.setContentsMargins(10, 0, 0, 0)
        conf_info_layout.setSpacing(5)
        ocr_layout.addWidget(conf_info_widget)
        
        # Add icon
        icon_label = QLabel()
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Halgakos.ico')
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path)
            # Scale icon to 16x16 to match text size
            scaled_pixmap = pixmap.scaled(16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon_label.setPixmap(scaled_pixmap)
        else:
            # Fallback to emoji if icon not found
            icon_label.setText("ℹ️")
        icon_label.setFixedSize(16, 16)
        conf_info_layout.addWidget(icon_label)
        
        # Add text label
        conf_info = QLabel("Local OCR providers don't use this")
        conf_info_font = QFont('Arial', 9)
        conf_info.setFont(conf_info_font)
        conf_info.setStyleSheet("color: gray; font-style: italic;")
        conf_info.setWordWrap(False)
        conf_info_layout.addWidget(conf_info)
        conf_info_layout.addStretch()
        
        # Detection mode
        mode_widget = QWidget()
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        ocr_layout.addWidget(mode_widget)
        
        mode_label = QLabel("Detection Mode:")
        mode_label.setMinimumWidth(180)
        mode_layout.addWidget(mode_label)
        
        self.detection_mode_combo = QComboBox()
        self.detection_mode_combo.addItems(['document', 'text'])
        self.detection_mode_combo.setCurrentText(self.settings['ocr']['text_detection_mode'])
        mode_layout.addWidget(self.detection_mode_combo)
        
        mode_desc = QLabel("(document = better for manga, text = simple layouts)")
        mode_desc_font = QFont('Arial', 9)
        mode_desc.setFont(mode_desc_font)
        mode_desc.setStyleSheet("color: gray;")
        mode_layout.addWidget(mode_desc)
        mode_layout.addStretch()
        
        # Minimum region size for cloud OCR (Google/Azure)
        min_region_widget = QWidget()
        min_region_layout = QHBoxLayout(min_region_widget)
        min_region_layout.setContentsMargins(0, 0, 0, 0)
        ocr_layout.addWidget(min_region_widget)
        
        min_region_label = QLabel("☁️ Min Region Size:")
        min_region_label.setMinimumWidth(180)
        min_region_label.setToolTip(
            "Minimum dimension for cloud OCR regions (Google/Azure).\n"
            "Regions smaller than this will be resized before OCR.\n\n"
            "• 50px = Default (safer, ensures good OCR quality)\n"
            "• 32px = Smaller minimum (like some implementations)\n"
            "• 0px = Disabled (send regions as-is, may fail on very small text)"
        )
        min_region_layout.addWidget(min_region_label)
        
        self.min_region_size_spinbox = QSpinBox()
        self.min_region_size_spinbox.setRange(0, 100)
        self.min_region_size_spinbox.setSingleStep(1)
        self.min_region_size_spinbox.setValue(self.settings['ocr'].get('min_region_size', 50))
        self.min_region_size_spinbox.setToolTip("0 = disabled, 32-50 = recommended range")
        min_region_layout.addWidget(self.min_region_size_spinbox)
        
        min_region_unit = QLabel("pixels")
        min_region_layout.addWidget(min_region_unit)
        
        min_region_desc = QLabel("(0 = no resize, comic-translate style)")
        min_region_desc_font = QFont('Arial', 9)
        min_region_desc.setFont(min_region_desc_font)
        min_region_desc.setStyleSheet("color: gray;")
        min_region_layout.addWidget(min_region_desc)
        min_region_layout.addStretch()
        
        # OCR Max Retries
        ocr_retry_widget = QWidget()
        ocr_retry_layout = QHBoxLayout(ocr_retry_widget)
        ocr_retry_layout.setContentsMargins(0, 0, 0, 0)
        ocr_layout.addWidget(ocr_retry_widget)
        
        ocr_retry_label = QLabel("OCR Max Retries:")
        ocr_retry_label.setMinimumWidth(180)
        ocr_retry_label.setToolTip(
            "Number of retries for failed OCR attempts per region.\n\n"
            "• 0 = Disabled (default, fastest - try once only)\n"
            "• 1-2 = Conservative (retry on genuine API errors)\n"
            "• 3-5 = Aggressive (for unreliable connections)\n\n"
            "Note: Empty regions often mean there's truly no text,\n"
            "so retrying doesn't help and just slows things down."
        )
        ocr_retry_layout.addWidget(ocr_retry_label)
        
        self.ocr_max_retries_spinbox = QSpinBox()
        self.ocr_max_retries_spinbox.setRange(0, 5)
        self.ocr_max_retries_spinbox.setSingleStep(1)
        self.ocr_max_retries_spinbox.setValue(self.settings['ocr'].get('ocr_max_retries', 0))
        self.ocr_max_retries_spinbox.setToolTip("0 = disabled (fastest), 5 = maximum")
        ocr_retry_layout.addWidget(self.ocr_max_retries_spinbox)
        
        ocr_retry_unit = QLabel("retries")
        ocr_retry_layout.addWidget(ocr_retry_unit)
        
        ocr_retry_desc = QLabel("(0 = disabled, recommended for speed)")
        ocr_retry_desc_font = QFont('Arial', 9)
        ocr_retry_desc.setFont(ocr_retry_desc_font)
        ocr_retry_desc.setStyleSheet("color: gray;")
        ocr_retry_layout.addWidget(ocr_retry_desc)
        ocr_retry_layout.addStretch()
        
        # Text merging settings
        merge_group = QGroupBox("Text Region Merging")
        merge_layout = QVBoxLayout(merge_group)
        content_layout.addWidget(merge_group)
        
        # Merge nearby threshold
        nearby_widget = QWidget()
        nearby_layout = QHBoxLayout(nearby_widget)
        nearby_layout.setContentsMargins(0, 0, 0, 0)
        merge_layout.addWidget(nearby_widget)
        
        nearby_label = QLabel("Merge Distance:")
        nearby_label.setMinimumWidth(180)
        nearby_layout.addWidget(nearby_label)
        
        self.merge_nearby_threshold_spinbox = QSpinBox()
        self.merge_nearby_threshold_spinbox.setRange(0, 200)
        self.merge_nearby_threshold_spinbox.setSingleStep(10)
        self.merge_nearby_threshold_spinbox.setValue(self.settings['ocr']['merge_nearby_threshold'])
        nearby_layout.addWidget(self.merge_nearby_threshold_spinbox)
        
        nearby_unit = QLabel("pixels")
        nearby_layout.addWidget(nearby_unit)
        nearby_layout.addStretch()

        # Text Filtering Setting
        filter_group = QGroupBox("Text Filtering")
        filter_layout = QVBoxLayout(filter_group)
        content_layout.addWidget(filter_group)
        
        # Minimum text length
        min_length_widget = QWidget()
        min_length_layout = QHBoxLayout(min_length_widget)
        min_length_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.addWidget(min_length_widget)
        
        min_length_label = QLabel("Min Text Length:")
        min_length_label.setMinimumWidth(180)
        min_length_layout.addWidget(min_length_label)
        
        self.min_text_length_spinbox = QSpinBox()
        self.min_text_length_spinbox.setRange(0, 10)  # Allow 0 (no minimum)
        self.min_text_length_spinbox.setValue(self.settings['ocr'].get('min_text_length', 0))
        min_length_layout.addWidget(self.min_text_length_spinbox)
        
        min_length_unit = QLabel("characters")
        min_length_layout.addWidget(min_length_unit)
        
        min_length_desc = QLabel("(skip text shorter than this)")
        min_length_desc_font = QFont('Arial', 9)
        min_length_desc.setFont(min_length_desc_font)
        min_length_desc.setStyleSheet("color: gray;")
        min_length_layout.addWidget(min_length_desc)
        min_length_layout.addStretch()
        
        # Exclude English text checkbox
        self.exclude_english_checkbox = self._create_styled_checkbox("Exclude primarily English text (tunable threshold)")
        self.exclude_english_checkbox.setChecked(self.settings['ocr'].get('exclude_english_text', False))
        filter_layout.addWidget(self.exclude_english_checkbox)
        
        # Threshold slider
        english_threshold_widget = QWidget()
        english_threshold_layout = QHBoxLayout(english_threshold_widget)
        english_threshold_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.addWidget(english_threshold_widget)
        
        threshold_label = QLabel("English Exclude Threshold:")
        threshold_label.setMinimumWidth(240)
        english_threshold_layout.addWidget(threshold_label)
        
        self.english_exclude_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.english_exclude_threshold_slider.setRange(60, 99)
        self.english_exclude_threshold_slider.setValue(int(self.settings['ocr'].get('english_exclude_threshold', 0.7) * 100))
        self.english_exclude_threshold_slider.setMinimumWidth(250)
        english_threshold_layout.addWidget(self.english_exclude_threshold_slider)
        
        self.english_threshold_label = QLabel(f"{int(self.settings['ocr'].get('english_exclude_threshold', 0.7)*100)}%")
        self.english_threshold_label.setMinimumWidth(50)
        self.english_exclude_threshold_slider.valueChanged.connect(
            lambda v: self.english_threshold_label.setText(f"{v}%")
        )
        english_threshold_layout.addWidget(self.english_threshold_label)
        english_threshold_layout.addStretch()
        
        # Minimum character count
        min_chars_widget = QWidget()
        min_chars_layout = QHBoxLayout(min_chars_widget)
        min_chars_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.addWidget(min_chars_widget)
        
        min_chars_label = QLabel("Min chars to exclude as English:")
        min_chars_label.setMinimumWidth(240)
        min_chars_layout.addWidget(min_chars_label)
        
        self.english_exclude_min_chars_spinbox = QSpinBox()
        self.english_exclude_min_chars_spinbox.setRange(1, 10)
        self.english_exclude_min_chars_spinbox.setValue(self.settings['ocr'].get('english_exclude_min_chars', 4))
        min_chars_layout.addWidget(self.english_exclude_min_chars_spinbox)
        
        min_chars_unit = QLabel("characters")
        min_chars_layout.addWidget(min_chars_unit)
        min_chars_layout.addStretch()
        
        # Legacy aggressive short-token filter
        self.english_exclude_short_tokens_checkbox = self._create_styled_checkbox("Aggressively drop very short ASCII tokens (legacy)")
        self.english_exclude_short_tokens_checkbox.setChecked(self.settings['ocr'].get('english_exclude_short_tokens', False))
        filter_layout.addWidget(self.english_exclude_short_tokens_checkbox)
        
        # Help text
        filter_help = QLabel(
            "💡 Text filtering helps skip:\n"
            "   • UI elements and watermarks\n"
            "   • Page numbers and copyright text\n"
            "   • Single characters or symbols\n"
            "   • Non-target language text"
        )
        filter_help_font = QFont('Arial', 9)
        filter_help.setFont(filter_help_font)
        filter_help.setStyleSheet("color: gray;")
        filter_help.setWordWrap(True)
        filter_help.setContentsMargins(0, 10, 0, 0)
        filter_layout.addWidget(filter_help)

        # Azure-specific OCR settings (simplified - new API is synchronous)
        azure_ocr_group = QGroupBox("Azure OCR Settings")
        azure_ocr_layout = QVBoxLayout(azure_ocr_group)
        content_layout.addWidget(azure_ocr_group)

        # Azure merge multiplier (kept for backward compatibility)
        merge_mult_widget = QWidget()
        merge_mult_layout = QHBoxLayout(merge_mult_widget)
        merge_mult_layout.setContentsMargins(0, 0, 0, 0)
        azure_ocr_layout.addWidget(merge_mult_widget)
        
        merge_mult_label = QLabel("Merge Multiplier:")
        merge_mult_label.setMinimumWidth(180)
        merge_mult_layout.addWidget(merge_mult_label)
        
        self.azure_merge_multiplier_slider = QSlider(Qt.Orientation.Horizontal)
        self.azure_merge_multiplier_slider.setRange(100, 500)
        self.azure_merge_multiplier_slider.setValue(int(self.settings['ocr'].get('azure_merge_multiplier', 2.0) * 100))
        self.azure_merge_multiplier_slider.setMinimumWidth(200)
        merge_mult_layout.addWidget(self.azure_merge_multiplier_slider)

        self.azure_label = QLabel(f"{self.settings['ocr'].get('azure_merge_multiplier', 2.0):.2f}x")
        self.azure_label.setMinimumWidth(50)
        self.azure_merge_multiplier_slider.valueChanged.connect(
            lambda v: self.azure_label.setText(f"{v/100:.2f}x")
        )
        merge_mult_layout.addWidget(self.azure_label)

        merge_mult_desc = QLabel("(multiplies merge distance for Azure lines)")
        merge_mult_desc_font = QFont('Arial', 9)
        merge_mult_desc.setFont(merge_mult_desc_font)
        merge_mult_desc.setStyleSheet("color: gray;")
        merge_mult_layout.addWidget(merge_mult_desc)
        merge_mult_layout.addStretch()

        # Help text
        azure_help = QLabel(
            "💡 Azure uses new Image Analysis API (synchronous, no polling)\n"
            "💡 Language auto-detection works well for manga"
        )
        azure_help_font = QFont('Arial', 9)
        azure_help.setFont(azure_help_font)
        azure_help.setStyleSheet("color: gray;")
        azure_help.setWordWrap(True)
        azure_help.setContentsMargins(0, 10, 0, 0)
        azure_ocr_layout.addWidget(azure_help)
        
        # Rotation correction
        self.enable_rotation_checkbox = self._create_styled_checkbox("Enable automatic rotation correction for tilted text")
        self.enable_rotation_checkbox.setChecked(self.settings['ocr']['enable_rotation_correction'])
        merge_layout.addWidget(self.enable_rotation_checkbox)

        # OCR batching and locality settings
        ocr_batch_group = QGroupBox("OCR Batching & Concurrency")
        ocr_batch_layout = QVBoxLayout(ocr_batch_group)
        content_layout.addWidget(ocr_batch_group)

        # Enable OCR batching
        self.ocr_batch_enabled_checkbox = self._create_styled_checkbox("Enable OCR batching (independent of translation batching)")
        self.ocr_batch_enabled_checkbox.setChecked(self.settings['ocr'].get('ocr_batch_enabled', True))
        self.ocr_batch_enabled_checkbox.stateChanged.connect(self._toggle_ocr_batching_controls)
        ocr_batch_layout.addWidget(self.ocr_batch_enabled_checkbox)
        
        # OCR batch size
        ocr_bs_widget = QWidget()
        ocr_bs_layout = QHBoxLayout(ocr_bs_widget)
        ocr_bs_layout.setContentsMargins(0, 0, 0, 0)
        ocr_batch_layout.addWidget(ocr_bs_widget)
        self.ocr_bs_row = ocr_bs_widget
        
        ocr_bs_label = QLabel("OCR Batch Size:")
        ocr_bs_label.setMinimumWidth(180)
        ocr_bs_layout.addWidget(ocr_bs_label)
        
        self.ocr_batch_size_spinbox = QSpinBox()
        self.ocr_batch_size_spinbox.setRange(1, 32)
        self.ocr_batch_size_spinbox.setValue(int(self.settings['ocr'].get('ocr_batch_size', 8)))
        ocr_bs_layout.addWidget(self.ocr_batch_size_spinbox)
        
        ocr_bs_desc = QLabel("(Google: items/request; Azure: drives concurrency)")
        ocr_bs_desc_font = QFont('Arial', 9)
        ocr_bs_desc.setFont(ocr_bs_desc_font)
        ocr_bs_desc.setStyleSheet("color: gray;")
        ocr_bs_layout.addWidget(ocr_bs_desc)
        ocr_bs_layout.addStretch()

        # OCR Max Concurrency
        ocr_cc_widget = QWidget()
        ocr_cc_layout = QHBoxLayout(ocr_cc_widget)
        ocr_cc_layout.setContentsMargins(0, 0, 0, 0)
        ocr_batch_layout.addWidget(ocr_cc_widget)
        self.ocr_cc_row = ocr_cc_widget
        
        ocr_cc_label = QLabel("OCR Max Concurrency:")
        ocr_cc_label.setMinimumWidth(180)
        ocr_cc_layout.addWidget(ocr_cc_label)
        
        self.ocr_max_conc_spinbox = QSpinBox()
        self.ocr_max_conc_spinbox.setRange(1, 8)
        self.ocr_max_conc_spinbox.setValue(int(self.settings['ocr'].get('ocr_max_concurrency', 2)))
        ocr_cc_layout.addWidget(self.ocr_max_conc_spinbox)
        
        ocr_cc_desc = QLabel("(Google: concurrent requests; Azure: workers, capped at 4)")
        ocr_cc_desc_font = QFont('Arial', 9)
        ocr_cc_desc.setFont(ocr_cc_desc_font)
        ocr_cc_desc.setStyleSheet("color: gray;")
        ocr_cc_layout.addWidget(ocr_cc_desc)
        ocr_cc_layout.addStretch()
        
        # Apply initial visibility for OCR batching controls
        try:
            self._toggle_ocr_batching_controls()
        except Exception:
            pass

        # ROI sizing
        roi_group = QGroupBox("ROI Locality Controls")
        roi_layout = QVBoxLayout(roi_group)
        content_layout.addWidget(roi_group)

        # ROI locality toggle (now inside this section)
        self.roi_locality_checkbox = self._create_styled_checkbox("Enable ROI-based OCR locality and batching (uses bubble detection)")
        self.roi_locality_checkbox.setChecked(self.settings['ocr'].get('roi_locality_enabled', False))
        self.roi_locality_checkbox.stateChanged.connect(self._toggle_roi_locality_controls)
        roi_layout.addWidget(self.roi_locality_checkbox)

        # ROI padding ratio
        roi_pad_widget = QWidget()
        roi_pad_layout = QHBoxLayout(roi_pad_widget)
        roi_pad_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.addWidget(roi_pad_widget)
        self.roi_pad_row = roi_pad_widget
        
        roi_pad_label = QLabel("ROI Padding Ratio:")
        roi_pad_label.setMinimumWidth(180)
        roi_pad_layout.addWidget(roi_pad_label)
        
        self.roi_padding_ratio_slider = QSlider(Qt.Orientation.Horizontal)
        self.roi_padding_ratio_slider.setRange(0, 30)
        self.roi_padding_ratio_slider.setValue(int(float(self.settings['ocr'].get('roi_padding_ratio', 0.08)) * 100))
        self.roi_padding_ratio_slider.setMinimumWidth(200)
        roi_pad_layout.addWidget(self.roi_padding_ratio_slider)
        
        self.roi_padding_ratio_label = QLabel(f"{float(self.settings['ocr'].get('roi_padding_ratio', 0.08)):.2f}")
        self.roi_padding_ratio_label.setMinimumWidth(50)
        self.roi_padding_ratio_slider.valueChanged.connect(
            lambda v: self.roi_padding_ratio_label.setText(f"{v/100:.2f}")
        )
        roi_pad_layout.addWidget(self.roi_padding_ratio_label)
        roi_pad_layout.addStretch()

        # ROI min side / area
        roi_min_widget = QWidget()
        roi_min_layout = QHBoxLayout(roi_min_widget)
        roi_min_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.addWidget(roi_min_widget)
        self.roi_min_row = roi_min_widget
        
        roi_min_label = QLabel("Min ROI Side:")
        roi_min_label.setMinimumWidth(180)
        roi_min_layout.addWidget(roi_min_label)
        
        self.roi_min_side_spinbox = QSpinBox()
        self.roi_min_side_spinbox.setRange(1, 64)
        self.roi_min_side_spinbox.setValue(int(self.settings['ocr'].get('roi_min_side_px', 12)))
        roi_min_layout.addWidget(self.roi_min_side_spinbox)
        
        roi_min_unit = QLabel("px")
        roi_min_layout.addWidget(roi_min_unit)
        roi_min_layout.addStretch()

        roi_area_widget = QWidget()
        roi_area_layout = QHBoxLayout(roi_area_widget)
        roi_area_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.addWidget(roi_area_widget)
        self.roi_area_row = roi_area_widget
        
        roi_area_label = QLabel("Min ROI Area:")
        roi_area_label.setMinimumWidth(180)
        roi_area_layout.addWidget(roi_area_label)
        
        self.roi_min_area_spinbox = QSpinBox()
        self.roi_min_area_spinbox.setRange(1, 5000)
        self.roi_min_area_spinbox.setValue(int(self.settings['ocr'].get('roi_min_area_px', 100)))
        roi_area_layout.addWidget(self.roi_min_area_spinbox)
        
        roi_area_unit = QLabel("px^2")
        roi_area_layout.addWidget(roi_area_unit)
        roi_area_layout.addStretch()

        # ROI max side (0 disables)
        roi_max_widget = QWidget()
        roi_max_layout = QHBoxLayout(roi_max_widget)
        roi_max_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.addWidget(roi_max_widget)
        self.roi_max_row = roi_max_widget
        
        roi_max_label = QLabel("ROI Max Side (0=off):")
        roi_max_label.setMinimumWidth(180)
        roi_max_layout.addWidget(roi_max_label)
        
        self.roi_max_side_spinbox = QSpinBox()
        self.roi_max_side_spinbox.setRange(0, 2048)
        self.roi_max_side_spinbox.setValue(int(self.settings['ocr'].get('roi_max_side', 0)))
        roi_max_layout.addWidget(self.roi_max_side_spinbox)
        roi_max_layout.addStretch()

        # Apply initial visibility based on toggle
        self._toggle_roi_locality_controls()

        # AI Bubble Detection Settings
        bubble_group = QGroupBox("AI Bubble Detection")
        bubble_layout = QVBoxLayout(bubble_group)
        content_layout.addWidget(bubble_group)

        # Enable bubble detection
        self.bubble_detection_enabled_checkbox = self._create_styled_checkbox("Enable AI-powered bubble detection (overrides traditional merging)")
        # IMPORTANT: Default to True for optimal text detection (especially for Chinese/Japanese text)
        self.bubble_detection_enabled_checkbox.setChecked(self.settings['ocr'].get('bubble_detection_enabled', True))
        self.bubble_detection_enabled_checkbox.stateChanged.connect(self._toggle_bubble_controls)
        bubble_layout.addWidget(self.bubble_detection_enabled_checkbox)
        
        # Use RT-DETR for text region detection (not just bubble detection)
        self.use_rtdetr_for_ocr_checkbox = self._create_styled_checkbox("Use RT-DETR to guide OCR (Google/Azure only - others already do this)")
        self.use_rtdetr_for_ocr_checkbox.setChecked(self.settings['ocr'].get('use_rtdetr_for_ocr_regions', True))  # Default: True for best accuracy
        self.use_rtdetr_for_ocr_checkbox.setToolTip(
            "When enabled, RT-DETR first detects all text regions (text bubbles + free text), \n"
            "then your OCR provider reads each region separately.\n\n"
            "🎯 Applies to: Google Cloud Vision, Azure Computer Vision\n"
            "✓ Already enabled: Qwen2-VL, Custom API, EasyOCR, PaddleOCR, DocTR, manga-ocr\n\n"
            "Benefits:\n"
            "• More accurate text detection (trained specifically for manga/comics)\n"
            "• Better separation of overlapping text\n"
            "• Improved handling of different text types (bubbles vs. free text)\n"
            "• Focused OCR on actual text regions (faster, more accurate)\n\n"
            "Note: Requires bubble detection to be enabled and uses the selected detector above."
        )
        bubble_layout.addWidget(self.use_rtdetr_for_ocr_checkbox)
        
        # Enable fallback OCR for empty RT-DETR blocks (Azure/Azure Document Intelligence only)
        self.enable_fallback_ocr_checkbox = self._create_styled_checkbox("Enable fallback OCR for empty blocks (Azure only)")
        self.enable_fallback_ocr_checkbox.setChecked(self.settings['ocr'].get('enable_fallback_ocr', False))  # Default: False (disabled)
        self.enable_fallback_ocr_checkbox.setToolTip(
            "When enabled, if RT-DETR detects text bubbles but Azure OCR finds no text,\n"
            "the system will re-run OCR on those specific regions with padding and upscaling.\n\n"
            "⚠️ WARNING: This adds extra API calls and may increase costs.\n\n"
            "🎯 Only applies to: Azure Computer Vision, Azure Document Intelligence\n"
            "   (when using RT-DETR guidance)\n\n"
            "Use cases:\n"
            "• Small or faint text that full-image OCR misses\n"
            "• Text at unusual angles or perspectives\n"
            "• Low-quality scans where upscaling helps\n\n"
            "Note: Disabled by default to reduce API costs. Enable only if experiencing\n"
            "missing text in detected bubbles."
        )
        bubble_layout.addWidget(self.enable_fallback_ocr_checkbox)
        
        # Skip RT-DETR region merging (disabled by default)
        self.skip_rtdetr_merging_checkbox = self._create_styled_checkbox("Skip RT-DETR region merging (preserve all detected regions)")
        self.skip_rtdetr_merging_checkbox.setChecked(self.settings['ocr'].get('skip_rtdetr_merging', False))
        self.skip_rtdetr_merging_checkbox.setToolTip(
            "When enabled, overlapping RT-DETR regions will NOT be merged.\n\n"
            "Use this to preserve granular free-text regions (e.g., separate SFX parts)\n"
            "and mimic manual edit mode behavior."
        )
        bubble_layout.addWidget(self.skip_rtdetr_merging_checkbox)
        
        # Preserve empty RT-DETR blocks (disabled by default)
        self.preserve_empty_blocks_checkbox = self._create_styled_checkbox("Preserve empty RT-DETR blocks (no OCR text)")
        self.preserve_empty_blocks_checkbox.setChecked(self.settings['ocr'].get('preserve_empty_blocks', False))
        self.preserve_empty_blocks_checkbox.setToolTip(
            "When enabled, regions detected by RT-DETR that have no matched OCR lines\n"
            "will still be kept. Useful for debugging missed text and SFX."
        )
        bubble_layout.addWidget(self.preserve_empty_blocks_checkbox)

        # Detector type dropdown
        detector_type_widget = QWidget()
        detector_type_layout = QHBoxLayout(detector_type_widget)
        detector_type_layout.setContentsMargins(0, 10, 0, 0)
        bubble_layout.addWidget(detector_type_widget)

        detector_type_label = QLabel("Detector:")
        detector_type_label.setMinimumWidth(120)
        detector_type_layout.addWidget(detector_type_label)

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

        self.detector_type_combo = QComboBox()
        self.detector_type_combo.addItems(list(self.detector_models.keys()))
        self.detector_type_combo.setCurrentText(initial_selection)
        self.detector_type_combo.currentTextChanged.connect(self._on_detector_type_changed)
        detector_type_layout.addWidget(self.detector_type_combo)
        detector_type_layout.addStretch()

        # NOW create the settings frame
        self.yolo_settings_group = QGroupBox("Model Settings")
        yolo_settings_layout = QVBoxLayout(self.yolo_settings_group)
        bubble_layout.addWidget(self.yolo_settings_group)
        self.rtdetr_settings_frame = self.yolo_settings_group  # Alias for compatibility

        # Model path/URL row
        model_widget = QWidget()
        model_layout = QHBoxLayout(model_widget)
        model_layout.setContentsMargins(0, 5, 0, 0)
        yolo_settings_layout.addWidget(model_widget)

        model_label = QLabel("Model:")
        model_label.setMinimumWidth(100)
        model_layout.addWidget(model_label)

        self.bubble_model_entry = QLineEdit()
        self.bubble_model_entry.setText(self.settings['ocr'].get('bubble_model_path', ''))
        self.bubble_model_entry.setReadOnly(True)
        self.bubble_model_entry.setStyleSheet(
            "QLineEdit { background-color: #1e1e1e; color: #ffffff; border: 1px solid #3a3a3a; }"
        )
        model_layout.addWidget(self.bubble_model_entry)
        self.rtdetr_url_entry = self.bubble_model_entry  # Alias
        
        # Store for compatibility
        self.detector_radio_widgets = [self.detector_type_combo]

        # Browse and Clear buttons (initially hidden for HuggingFace models)
        self.bubble_browse_btn = QPushButton("Browse")
        self.bubble_browse_btn.clicked.connect(self._browse_bubble_model)
        self.bubble_browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #5a9fd4;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 3px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #7bb3e0;
            }
            QPushButton:pressed {
                background-color: #4a8fc4;
            }
        """)
        model_layout.addWidget(self.bubble_browse_btn)

        self.bubble_clear_btn = QPushButton("Clear")
        self.bubble_clear_btn.clicked.connect(self._clear_bubble_model)
        self.bubble_clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 3px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #7d8a96;
            }
            QPushButton:pressed {
                background-color: #5a6268;
            }
        """)
        model_layout.addWidget(self.bubble_clear_btn)
        model_layout.addStretch()

        # Download and Load buttons
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 10, 0, 0)
        yolo_settings_layout.addWidget(button_widget)

        button_label = QLabel("Actions:")
        button_label.setMinimumWidth(100)
        button_layout.addWidget(button_label)

        self.rtdetr_download_btn = QPushButton("Download")
        self.rtdetr_download_btn.clicked.connect(self._download_rtdetr_model)
        self.rtdetr_download_btn.setStyleSheet("""
            QPushButton {
                background-color: #5a9fd4;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 3px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #7bb3e0;
            }
            QPushButton:pressed {
                background-color: #4a8fc4;
            }
        """)
        button_layout.addWidget(self.rtdetr_download_btn)

        self.rtdetr_load_btn = QPushButton("Load Model")
        self.rtdetr_load_btn.clicked.connect(self._load_rtdetr_model)
        self.rtdetr_load_btn.setStyleSheet("""
            QPushButton {
                background-color: #5a9fd4;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 3px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #7bb3e0;
            }
            QPushButton:pressed {
                background-color: #4a8fc4;
            }
        """)
        button_layout.addWidget(self.rtdetr_load_btn)

        self.rtdetr_status_label = QLabel("")
        rtdetr_status_font = QFont('Arial', 9)
        self.rtdetr_status_label.setFont(rtdetr_status_font)
        button_layout.addWidget(self.rtdetr_status_label)
        button_layout.addStretch()

        # RT-DETR Detection classes
        rtdetr_classes_widget = QWidget()
        rtdetr_classes_layout = QHBoxLayout(rtdetr_classes_widget)
        rtdetr_classes_layout.setContentsMargins(0, 10, 0, 0)
        yolo_settings_layout.addWidget(rtdetr_classes_widget)
        self.rtdetr_classes_frame = rtdetr_classes_widget

        classes_label = QLabel("Detect:")
        classes_label.setMinimumWidth(100)
        rtdetr_classes_layout.addWidget(classes_label)

        self.detect_empty_bubbles_checkbox = self._create_styled_checkbox("Empty Bubbles")
        self.detect_empty_bubbles_checkbox.setChecked(self.settings['ocr'].get('detect_empty_bubbles', True))
        rtdetr_classes_layout.addWidget(self.detect_empty_bubbles_checkbox)

        self.detect_text_bubbles_checkbox = self._create_styled_checkbox("Text Bubbles")
        self.detect_text_bubbles_checkbox.setChecked(self.settings['ocr'].get('detect_text_bubbles', True))
        rtdetr_classes_layout.addWidget(self.detect_text_bubbles_checkbox)

        self.detect_free_text_checkbox = self._create_styled_checkbox("Free Text")
        self.detect_free_text_checkbox.setChecked(self.settings['ocr'].get('detect_free_text', True))
        rtdetr_classes_layout.addWidget(self.detect_free_text_checkbox)
        rtdetr_classes_layout.addStretch()

        # Confidence
        conf_widget = QWidget()
        conf_layout = QHBoxLayout(conf_widget)
        conf_layout.setContentsMargins(0, 10, 0, 0)
        yolo_settings_layout.addWidget(conf_widget)

        conf_label = QLabel("Confidence:")
        conf_label.setMinimumWidth(100)
        conf_layout.addWidget(conf_label)

        detector_label = self.detector_type_combo.currentText()
        default_conf = 0.3 if ('RT-DETR' in detector_label or 'RTEDR_onnx' in detector_label or 'onnx' in detector_label.lower()) else 0.5
        
        self.bubble_conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.bubble_conf_slider.setRange(0, 99)
        self.bubble_conf_slider.setValue(int(self.settings['ocr'].get('bubble_confidence', default_conf) * 100))
        self.bubble_conf_slider.setMinimumWidth(200)
        conf_layout.addWidget(self.bubble_conf_slider)
        self.rtdetr_conf_scale = self.bubble_conf_slider  # Alias

        self.bubble_conf_label = QLabel(f"{self.settings['ocr'].get('bubble_confidence', default_conf):.2f}")
        self.bubble_conf_label.setMinimumWidth(50)
        self.bubble_conf_slider.valueChanged.connect(
            lambda v: self.bubble_conf_label.setText(f"{v/100:.2f}")
        )
        conf_layout.addWidget(self.bubble_conf_label)
        self.rtdetr_conf_label = self.bubble_conf_label  # Alias
        conf_layout.addStretch()

        # YOLO-specific: Max detections (only visible for YOLO)
        self.yolo_maxdet_widget = QWidget()
        yolo_maxdet_layout = QHBoxLayout(self.yolo_maxdet_widget)
        yolo_maxdet_layout.setContentsMargins(0, 6, 0, 0)
        yolo_settings_layout.addWidget(self.yolo_maxdet_widget)
        self.yolo_maxdet_row = self.yolo_maxdet_widget  # Alias
        self.yolo_maxdet_widget.setVisible(False)  # Hidden initially
        
        maxdet_label = QLabel("Max detections:")
        maxdet_label.setMinimumWidth(100)
        yolo_maxdet_layout.addWidget(maxdet_label)
        
        self.bubble_max_det_yolo_spinbox = QSpinBox()
        self.bubble_max_det_yolo_spinbox.setRange(1, 2000)
        self.bubble_max_det_yolo_spinbox.setValue(self.settings['ocr'].get('bubble_max_detections_yolo', 100))
        yolo_maxdet_layout.addWidget(self.bubble_max_det_yolo_spinbox)
        yolo_maxdet_layout.addStretch()

        # Status label at the bottom of bubble group
        self.bubble_status_label = QLabel("")
        bubble_status_font = QFont('Arial', 9)
        self.bubble_status_label.setFont(bubble_status_font)
        bubble_status_label_container = QWidget()
        bubble_status_label_layout = QVBoxLayout(bubble_status_label_container)
        bubble_status_label_layout.setContentsMargins(0, 10, 0, 0)
        bubble_status_label_layout.addWidget(self.bubble_status_label)
        bubble_layout.addWidget(bubble_status_label_container)

        # Store controls for enable/disable
        self.bubble_controls = [
            self.use_rtdetr_for_ocr_checkbox,
            self.enable_fallback_ocr_checkbox,
            self.skip_rtdetr_merging_checkbox,
            self.preserve_empty_blocks_checkbox,
            self.detector_type_combo,
            self.bubble_model_entry,
            self.bubble_browse_btn,
            self.bubble_clear_btn,
            self.bubble_conf_slider,
            self.rtdetr_download_btn,
            self.rtdetr_load_btn
        ]

        self.rtdetr_controls = [
            self.bubble_model_entry,
            self.rtdetr_load_btn,
            self.rtdetr_download_btn,
            self.bubble_conf_slider,
            self.detect_empty_bubbles_checkbox,
            self.detect_text_bubbles_checkbox,
            self.detect_free_text_checkbox
        ]

        self.yolo_controls = [
            self.bubble_model_entry,
            self.bubble_browse_btn,
            self.bubble_clear_btn,
            self.bubble_conf_slider,
            self.yolo_maxdet_widget
        ]

        # Add stretch to end of OCR tab content
        content_layout.addStretch()

        # Set initialization flag to prevent auto-scroll during setup
        self._initializing_bubble_controls = True
        
        # Initialize control states
        self._toggle_bubble_controls()

        # Only call detector change after everything is initialized
        if self.bubble_detection_enabled_checkbox.isChecked():
            try:
                self._on_detector_type_changed()
                self._update_bubble_status()
            except AttributeError:
                # Frames not yet created, skip initialization
                pass

        # Clear initialization flag
        self._initializing_bubble_controls = False
        
        # Check status after dialog ready
        QTimer.singleShot(500, self._check_rtdetr_status)
    
    def _on_detector_type_changed(self, detector=None):
        """Handle detector type change"""
        if not hasattr(self, 'bubble_detection_enabled_checkbox'):
            return
            
        if not self.bubble_detection_enabled_checkbox.isChecked():
            self.yolo_settings_group.setVisible(False)
            return
        
        if detector is None:
            detector = self.detector_type_combo.currentText()
        
        # Handle different detector types
        if detector == 'Custom Model':
            # Custom model - enable manual entry
            self.bubble_model_entry.setText(self.settings['ocr'].get('custom_model_path', ''))
            self.bubble_model_entry.setReadOnly(False)
            self.bubble_model_entry.setStyleSheet(
                "QLineEdit { background-color: #2b2b2b; color: #ffffff; border: 1px solid #3a3a3a; }"
            )
            # Show browse/clear buttons for custom
            self.bubble_browse_btn.setVisible(True)
            self.bubble_clear_btn.setVisible(True)
            # Hide download button
            self.rtdetr_download_btn.setVisible(False)
        elif detector in self.detector_models:
            # HuggingFace model
            url = self.detector_models[detector]
            self.bubble_model_entry.setText(url)
            # Make entry read-only for HuggingFace models
            self.bubble_model_entry.setReadOnly(True)
            self.bubble_model_entry.setStyleSheet(
                "QLineEdit { background-color: #1e1e1e; color: #ffffff; border: 1px solid #3a3a3a; }"
            )
            # Hide browse/clear buttons for HuggingFace models
            self.bubble_browse_btn.setVisible(False)
            self.bubble_clear_btn.setVisible(False)
            # Show download button
            self.rtdetr_download_btn.setVisible(True)
        
        # Show/hide RT-DETR specific controls
        is_rtdetr = 'RT-DETR' in detector or 'RTEDR_onnx' in detector
        
        if is_rtdetr:
            self.rtdetr_classes_frame.setVisible(True)
            # Hide YOLO-only max det row
            self.yolo_maxdet_widget.setVisible(False)
        else:
            self.rtdetr_classes_frame.setVisible(False)
            # Show YOLO-only max det row for YOLO models
            if 'YOLO' in detector or 'Yolo' in detector or 'yolo' in detector or detector == 'Custom Model':
                self.yolo_maxdet_widget.setVisible(True)
            else:
                self.yolo_maxdet_widget.setVisible(False)
        
        # Show/hide RT-DETR concurrency control in Performance section (Advanced tab)
        # Only update if the widget has been created (Advanced tab may not be loaded yet)
        if hasattr(self, 'rtdetr_conc_frame'):
            self.rtdetr_conc_frame.setVisible(is_rtdetr)
        
        # Always show settings frame
        self.yolo_settings_group.setVisible(True)
        
        # Update status
        self._update_bubble_status()

    def _download_rtdetr_model(self):
        """Download selected model"""
        try:
            detector = self.detector_type_combo.currentText()
            model_url = self.bubble_model_entry.text()
            
            self.rtdetr_status_label.setText("Downloading...")
            self.rtdetr_status_label.setStyleSheet("color: orange;")
            QApplication.processEvents()
            
            if 'RTEDR_onnx' in detector:
                from bubble_detector import BubbleDetector
                bd = BubbleDetector()
                if bd.load_rtdetr_onnx_model(model_id=model_url):
                    self.rtdetr_status_label.setText("✅ Downloaded")
                    self.rtdetr_status_label.setStyleSheet("color: green;")
                    self._show_styled_messagebox(QMessageBox.Information, "Success", "RTEDR_onnx model downloaded successfully!")
                else:
                    self.rtdetr_status_label.setText("❌ Failed")
                    self.rtdetr_status_label.setStyleSheet("color: red;")
                    self._show_styled_messagebox(QMessageBox.Critical, "Error", "Failed to download RTEDR_onnx model")
            elif 'RT-DETR' in detector:
                # RT-DETR handling (works fine)
                from bubble_detector import BubbleDetector
                bd = BubbleDetector()
                
                if bd.load_rtdetr_model(model_id=model_url):
                    self.rtdetr_status_label.setText("✅ Downloaded")
                    self.rtdetr_status_label.setStyleSheet("color: green;")
                    self._show_styled_messagebox(QMessageBox.Information, "Success", "RT-DETR model downloaded successfully!")
                else:
                    self.rtdetr_status_label.setText("❌ Failed")
                    self.rtdetr_status_label.setStyleSheet("color: red;")
                    self._show_styled_messagebox(QMessageBox.Critical, "Error", "Failed to download RT-DETR model")
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
                self.bubble_model_entry.setText(local_path)
                self.rtdetr_status_label.setText("✅ Downloaded")
                self.rtdetr_status_label.setStyleSheet("color: green;")
                self._show_styled_messagebox(QMessageBox.Information, "Success", f"Model downloaded to:\n{local_path}")
        
        except ImportError:
            self.rtdetr_status_label.setText("❌ Missing deps")
            self.rtdetr_status_label.setStyleSheet("color: red;")
            self._show_styled_messagebox(QMessageBox.Critical, "Error", "Install: pip install huggingface-hub transformers")
        except Exception as e:
            self.rtdetr_status_label.setText("❌ Error")
            self.rtdetr_status_label.setStyleSheet("color: red;")
            self._show_styled_messagebox(QMessageBox.Critical, "Error", f"Download failed: {e}")

    def _check_rtdetr_status(self):
        """Check if model is already loaded"""
        try:
            from bubble_detector import BubbleDetector
            
            if hasattr(self.main_gui, 'manga_tab') and hasattr(self.main_gui.manga_tab, 'translator'):
                translator = self.main_gui.manga_tab.translator
                if hasattr(translator, 'bubble_detector') and translator.bubble_detector:
                    if getattr(translator.bubble_detector, 'rtdetr_onnx_loaded', False):
                        self.rtdetr_status_label.setText("✅ Loaded")
                        self.rtdetr_status_label.setStyleSheet("color: green;")
                        return True
                    if getattr(translator.bubble_detector, 'rtdetr_loaded', False):
                        self.rtdetr_status_label.setText("✅ Loaded")
                        self.rtdetr_status_label.setStyleSheet("color: green;")
                        return True
                    elif getattr(translator.bubble_detector, 'model_loaded', False):
                        self.rtdetr_status_label.setText("✅ Loaded")
                        self.rtdetr_status_label.setStyleSheet("color: green;")
                        return True
            
            self.rtdetr_status_label.setText("Not loaded")
            self.rtdetr_status_label.setStyleSheet("color: gray;")
            return False
            
        except ImportError:
            self.rtdetr_status_label.setText("❌ Missing deps")
            self.rtdetr_status_label.setStyleSheet("color: red;")
            return False
        except Exception:
            self.rtdetr_status_label.setText("Not loaded")
            self.rtdetr_status_label.setStyleSheet("color: gray;")
            return False

    def _load_rtdetr_model(self):
        """Load selected model"""
        try:
            from bubble_detector import BubbleDetector
            from PySide6.QtWidgets import QApplication
            
            self.rtdetr_status_label.setText("Loading...")
            self.rtdetr_status_label.setStyleSheet("color: orange;")
            QApplication.processEvents()
            
            bd = BubbleDetector()
            detector = self.detector_type_combo.currentText()
            model_path = self.bubble_model_entry.text()
            
            if 'RTEDR_onnx' in detector:
                # RT-DETR (ONNX) uses repo id directly
                if bd.load_rtdetr_onnx_model(model_id=model_path):
                    self.rtdetr_status_label.setText("✅ Ready")
                    self.rtdetr_status_label.setStyleSheet("color: green;")
                    self._show_styled_messagebox(QMessageBox.Information, "Success", "RTEDR_onnx model loaded successfully!")
                else:
                    self.rtdetr_status_label.setText("❌ Failed")
                    self.rtdetr_status_label.setStyleSheet("color: red;")
            elif 'RT-DETR' in detector:
                # RT-DETR uses model_id directly
                if bd.load_rtdetr_model(model_id=model_path):
                    self.rtdetr_status_label.setText("✅ Ready")
                    self.rtdetr_status_label.setStyleSheet("color: green;")
                    self._show_styled_messagebox(QMessageBox.Information, "Success", "RT-DETR model loaded successfully!")
                else:
                    self.rtdetr_status_label.setText("❌ Failed")
                    self.rtdetr_status_label.setStyleSheet("color: red;")
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
                        self.bubble_model_entry.setText(local_path)  # Update the field
                    else:
                        # Not downloaded yet
                        self._show_styled_messagebox(QMessageBox.Warning, "Download Required", 
                            "Model not found locally.\nPlease download it first using the Download button.")
                        self.rtdetr_status_label.setText("❌ Not downloaded")
                        self.rtdetr_status_label.setStyleSheet("color: orange;")
                        return
                
                # Now model_path should be a local file
                if not os.path.exists(model_path):
                    self._show_styled_messagebox(QMessageBox.Critical, "Error", f"Model file not found: {model_path}")
                    self.rtdetr_status_label.setText("❌ File not found")
                    self.rtdetr_status_label.setStyleSheet("color: red;")
                    return
                
                # Load the YOLOv8 model from local file
                if bd.load_model(model_path):
                    self.rtdetr_status_label.setText("✅ Ready")
                    self.rtdetr_status_label.setStyleSheet("color: green;")
                    self._show_styled_messagebox(QMessageBox.Information, "Success", "YOLOv8 model loaded successfully!")
                    
                    # Auto-convert to ONNX if enabled
                    if os.environ.get('AUTO_CONVERT_TO_ONNX', 'true').lower() == 'true':
                        onnx_path = model_path.replace('.pt', '.onnx')
                        if not os.path.exists(onnx_path):
                            if bd.convert_to_onnx(model_path, onnx_path):
                                logger.info(f"✅ Converted to ONNX: {onnx_path}")
                else:
                    self.rtdetr_status_label.setText("❌ Failed")
                    self.rtdetr_status_label.setStyleSheet("color: red;")
                
        except ImportError:
            self.rtdetr_status_label.setText("❌ Missing deps")
            self.rtdetr_status_label.setStyleSheet("color: red;")
            self._show_styled_messagebox(QMessageBox.Critical, "Error", "Install transformers: pip install transformers")
        except Exception as e:
            self.rtdetr_status_label.setText("❌ Error")
            self.rtdetr_status_label.setStyleSheet("color: red;")
            self._show_styled_messagebox(QMessageBox.Critical, "Error", f"Failed to load: {e}")
        
    def _toggle_bubble_controls(self):
        """Enable/disable bubble detection controls with fade animation"""
        enabled = self.bubble_detection_enabled_checkbox.isChecked()
        
        if enabled:
            # Fade in and enable controls
            self._fade_widget(self.use_rtdetr_for_ocr_checkbox, fade_in=True)
            self._fade_widget(self.enable_fallback_ocr_checkbox, fade_in=True)
            self._fade_widget(self.skip_rtdetr_merging_checkbox, fade_in=True)
            self._fade_widget(self.preserve_empty_blocks_checkbox, fade_in=True)
            self._fade_widget(self.yolo_settings_group, fade_in=True)
            
            # Enable controls after starting fade
            for widget in self.bubble_controls:
                try:
                    widget.setEnabled(True)
                except:
                    pass
            
            # Show/hide frames based on detector type
            self._on_detector_type_changed()
            
            # Auto-scroll to show the expanded bubble detection settings (but not during initialization)
            if not getattr(self, '_initializing_bubble_controls', False):
                QTimer.singleShot(100, self._scroll_to_bubble_settings)
        else:
            # Fade out and disable controls
            self._fade_widget(self.use_rtdetr_for_ocr_checkbox, fade_in=False)
            self._fade_widget(self.enable_fallback_ocr_checkbox, fade_in=False)
            self._fade_widget(self.skip_rtdetr_merging_checkbox, fade_in=False)
            self._fade_widget(self.preserve_empty_blocks_checkbox, fade_in=False)
            self._fade_widget(self.yolo_settings_group, fade_in=False, 
                            on_finished=lambda: self._finish_bubble_disable())
            
            # Disable controls immediately
            for widget in self.bubble_controls:
                try:
                    widget.setEnabled(False)
                except:
                    pass
    
    def _scroll_to_bubble_settings(self):
        """Smoothly scroll DOWN to show expanded bubble detection settings"""
        if not hasattr(self, 'main_scroll_area'):
            return
        
        try:
            # Get current scroll position
            scrollbar = self.main_scroll_area.verticalScrollBar()
            current_value = scrollbar.value()
            
            # Scroll down by a reasonable amount to reveal the expanded content
            # Add 200 pixels to current position to scroll down
            scroll_increment = 200
            target_value = min(current_value + scroll_increment, scrollbar.maximum())
            
            # Only scroll if we're not already at the bottom
            if current_value < scrollbar.maximum():
                # Smooth scroll animation
                self._scroll_animation = QPropertyAnimation(scrollbar, b"value")
                self._scroll_animation.setDuration(300)
                self._scroll_animation.setStartValue(current_value)
                self._scroll_animation.setEndValue(target_value)
                self._scroll_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
                self._scroll_animation.start()
        except Exception as e:
            # Fallback to instant scroll if animation fails
            pass
    
    def _finish_bubble_disable(self):
        """Complete the bubble disable process after fade out"""
        self.yolo_settings_group.setVisible(False)
        self.bubble_status_label.setText("")
    
    def _fade_widget(self, widget, fade_in=True, duration=200, on_finished=None):
        """Fade a widget in or out with animation
        
        Args:
            widget: The widget to fade
            fade_in: True to fade in, False to fade out
            duration: Animation duration in milliseconds
            on_finished: Optional callback when animation finishes
        """
        # Ensure widget is visible for fade-in
        if fade_in:
            widget.setVisible(True)
        
        # Create opacity effect
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        
        # Create animation
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(duration)
        animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        def cleanup():
            """Remove graphics effect and call optional callback"""
            widget.setGraphicsEffect(None)  # Remove effect to restore normal rendering
            if fade_in:
                widget.setVisible(True)  # Ensure visible after fade in
            if on_finished:
                on_finished()
        
        if fade_in:
            animation.setStartValue(0.0)
            animation.setEndValue(1.0)
            animation.finished.connect(cleanup)
        else:
            animation.setStartValue(1.0)
            animation.setEndValue(0.0)
            animation.finished.connect(cleanup)
        
        # Store animation reference to prevent garbage collection
        if not hasattr(self, '_animations'):
            self._animations = []
        self._animations.append(animation)
        
        animation.start()

    def _browse_bubble_model(self):
        """Browse for model file"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "Model files (*.pt *.pth *.bin *.safetensors);;All files (*.*)"
        )
        
        if path:
            self.bubble_model_entry.setText(path)
            self._update_bubble_status()

    def _clear_bubble_model(self):
        """Clear selected model"""
        self.bubble_model_entry.setText("")
        self._update_bubble_status()

    def _update_bubble_status(self):
        """Update bubble model status label"""
        if not self.bubble_detection_enabled_checkbox.isChecked():
            self.bubble_status_label.setText("")
            return
        
        detector = self.detector_type_combo.currentText()
        model_path = self.bubble_model_entry.text()
        
        if not model_path:
            self.bubble_status_label.setText("⚠️ No model selected")
            self.bubble_status_label.setStyleSheet("color: orange;")
            return
        
        if model_path.startswith("ogkalu/"):
            self.bubble_status_label.setText(f"📥 {detector} ready to download")
            self.bubble_status_label.setStyleSheet("color: #5dade2;")  # Light cyan for better contrast
        elif os.path.exists(model_path):
            self.bubble_status_label.setText("✅ Model file ready")
            self.bubble_status_label.setStyleSheet("color: green;")
        else:
            self.bubble_status_label.setText("❌ Model file not found")
            self.bubble_status_label.setStyleSheet("color: red;")

    def _update_azure_label(self):
        """Update Azure multiplier label"""
        # This method is deprecated - Azure multiplier UI was removed
        pass

    def _set_azure_multiplier(self, value):
        """Set Azure multiplier from preset"""
        # This method is deprecated - Azure multiplier UI was removed
        pass
    
    def _create_advanced_tab(self):
        """Create advanced settings tab with all options"""
        # Create tab widget and add to tab widget
        tab_widget = QWidget()
        self.tab_widget.addTab(tab_widget, "Advanced")
        
        # Main scrollable content
        main_layout = QVBoxLayout(tab_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(6)
        
        # Format detection
        detect_group = QGroupBox("Format Detection")
        main_layout.addWidget(detect_group)
        detect_layout = QVBoxLayout(detect_group)
        detect_layout.setContentsMargins(8, 8, 8, 6)
        detect_layout.setSpacing(4)
        
        self.format_detection_checkbox = self._create_styled_checkbox("Enable automatic manga format detection (reading direction)")
        self.format_detection_checkbox.setChecked(self.settings['advanced']['format_detection'])
        detect_layout.addWidget(self.format_detection_checkbox)
        
        # Webtoon mode
        webtoon_frame = QWidget()
        webtoon_layout = QHBoxLayout(webtoon_frame)
        webtoon_layout.setContentsMargins(0, 0, 0, 0)
        detect_layout.addWidget(webtoon_frame)
        
        webtoon_label = QLabel("Webtoon Mode:")
        webtoon_label.setMinimumWidth(150)
        webtoon_layout.addWidget(webtoon_label)
        
        self.webtoon_mode_combo = QComboBox()
        self.webtoon_mode_combo.addItems(['auto', 'enabled', 'disabled'])
        self.webtoon_mode_combo.setCurrentText(self.settings['advanced']['webtoon_mode'])
        webtoon_layout.addWidget(self.webtoon_mode_combo)
        webtoon_layout.addStretch()
        
        # Debug settings
        debug_group = QGroupBox("Debug Options")
        main_layout.addWidget(debug_group)
        debug_layout = QVBoxLayout(debug_group)
        debug_layout.setContentsMargins(8, 8, 8, 6)
        debug_layout.setSpacing(4)
        
        self.debug_mode_checkbox = self._create_styled_checkbox("Enable debug mode (verbose logging)")
        self.debug_mode_checkbox.setChecked(self.settings['advanced']['debug_mode'])
        debug_layout.addWidget(self.debug_mode_checkbox)
        
        # New: Concise pipeline logs (reduce noise)
        self.concise_logs_checkbox = self._create_styled_checkbox("Concise pipeline logs (reduce noise)")
        self.concise_logs_checkbox.setChecked(bool(self.settings.get('advanced', {}).get('concise_logs', True)))
        def _save_concise():
            try:
                if 'advanced' not in self.settings:
                    self.settings['advanced'] = {}
                self.settings['advanced']['concise_logs'] = bool(self.concise_logs_checkbox.isChecked())
                if hasattr(self, 'config'):
                    self.config['manga_settings'] = self.settings
                if hasattr(self.main_gui, 'save_config'):
                    self.main_gui.save_config(show_message=False)
            except Exception:
                pass
        self.concise_logs_checkbox.toggled.connect(_save_concise)
        debug_layout.addWidget(self.concise_logs_checkbox)
        
        # Add mutual exclusion logic for debug mode and concise logs
        def _on_debug_mode_changed(checked):
            """Handle debug mode checkbox changes"""
            if checked:
                # Disable and uncheck concise logs when debug mode is enabled
                self.concise_logs_checkbox.blockSignals(True)
                self.concise_logs_checkbox.setChecked(False)
                self.concise_logs_checkbox.setEnabled(False)
                # Update styling for disabled state
                self.concise_logs_checkbox.setStyleSheet("""
                    QCheckBox {
                        color: #666666;
                        spacing: 6px;
                    }
                    QCheckBox::indicator {
                        width: 14px;
                        height: 14px;
                        border: 1px solid #3a3a3a;
                        border-radius: 2px;
                        background-color: #1a1a1a;
                    }
                    QCheckBox::indicator:checked {
                        background-color: #3a3a3a;
                        border-color: #3a3a3a;
                    }
                """)
                self.concise_logs_checkbox.blockSignals(False)
                # Save the unchecked state for concise logs
                try:
                    if 'advanced' not in self.settings:
                        self.settings['advanced'] = {}
                    self.settings['advanced']['concise_logs'] = False
                    if hasattr(self, 'config'):
                        self.config['manga_settings'] = self.settings
                    if hasattr(self.main_gui, 'save_config'):
                        self.main_gui.save_config(show_message=False)
                except Exception:
                    pass
            else:
                # Re-enable concise logs when debug mode is disabled
                self.concise_logs_checkbox.setEnabled(True)
                # Restore normal styling
                self.concise_logs_checkbox.setStyleSheet("""
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
                """)
        
        def _on_concise_logs_changed(checked):
            """Handle concise logs checkbox changes"""
            if checked:
                # Disable and uncheck debug mode when concise logs is enabled
                self.debug_mode_checkbox.blockSignals(True)
                self.debug_mode_checkbox.setChecked(False)
                self.debug_mode_checkbox.setEnabled(False)
                # Update styling for disabled state
                self.debug_mode_checkbox.setStyleSheet("""
                    QCheckBox {
                        color: #666666;
                        spacing: 6px;
                    }
                    QCheckBox::indicator {
                        width: 14px;
                        height: 14px;
                        border: 1px solid #3a3a3a;
                        border-radius: 2px;
                        background-color: #1a1a1a;
                    }
                    QCheckBox::indicator:checked {
                        background-color: #3a3a3a;
                        border-color: #3a3a3a;
                    }
                """)
                self.debug_mode_checkbox.blockSignals(False)
                # Save the unchecked state for debug mode
                try:
                    if 'advanced' not in self.settings:
                        self.settings['advanced'] = {}
                    self.settings['advanced']['debug_mode'] = False
                    if hasattr(self, 'config'):
                        self.config['manga_settings'] = self.settings
                    if hasattr(self.main_gui, 'save_config'):
                        self.main_gui.save_config(show_message=False)
                except Exception:
                    pass
            else:
                # Re-enable debug mode when concise logs is disabled
                self.debug_mode_checkbox.setEnabled(True)
                # Restore normal styling
                self.debug_mode_checkbox.setStyleSheet("""
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
                """)
        
        # Connect the mutual exclusion handlers
        self.debug_mode_checkbox.toggled.connect(_on_debug_mode_changed)
        self.concise_logs_checkbox.toggled.connect(_on_concise_logs_changed)
        
        # Initialize the state based on current values (do this AFTER connecting handlers)
        # Block signals temporarily to avoid triggering handlers during initialization
        self.debug_mode_checkbox.blockSignals(True)
        self.concise_logs_checkbox.blockSignals(True)
        
        if self.debug_mode_checkbox.isChecked():
            _on_debug_mode_changed(True)
        elif self.concise_logs_checkbox.isChecked():
            _on_concise_logs_changed(True)
        
        # Re-enable signals
        self.debug_mode_checkbox.blockSignals(False)
        self.concise_logs_checkbox.blockSignals(False)
        
        self.save_intermediate_checkbox = self._create_styled_checkbox("Save intermediate images (preprocessed, detection overlays)")
        self.save_intermediate_checkbox.setChecked(self.settings['advanced']['save_intermediate'])
        debug_layout.addWidget(self.save_intermediate_checkbox)
        
        # Performance settings  
        perf_group = QGroupBox("Performance")
        main_layout.addWidget(perf_group)
        perf_layout = QVBoxLayout(perf_group)
        perf_layout.setContentsMargins(8, 8, 8, 6)
        perf_layout.setSpacing(4)
        
        # New: Parallel rendering (per-region overlays)
        self.render_parallel_checkbox = self._create_styled_checkbox("Enable parallel rendering (per-region overlays)")
        self.render_parallel_checkbox.setChecked(self.settings.get('advanced', {}).get('render_parallel', True))
        perf_layout.addWidget(self.render_parallel_checkbox)
        
        self.parallel_processing_checkbox = self._create_styled_checkbox("Enable parallel processing (experimental)")
        self.parallel_processing_checkbox.setChecked(self.settings['advanced']['parallel_processing'])
        self.parallel_processing_checkbox.toggled.connect(self._toggle_workers)
        perf_layout.addWidget(self.parallel_processing_checkbox)
        
        # Max workers
        workers_frame = QWidget()
        workers_layout = QHBoxLayout(workers_frame)
        workers_layout.setContentsMargins(0, 0, 0, 0)
        perf_layout.addWidget(workers_frame)
        
        self.workers_label = QLabel("Max Workers:")
        self.workers_label.setMinimumWidth(150)
        workers_layout.addWidget(self.workers_label)
        
        self.max_workers_spinbox = QSpinBox()
        self.max_workers_spinbox.setRange(1, 999)
        self.max_workers_spinbox.setValue(self.settings['advanced']['max_workers'])
        workers_layout.addWidget(self.max_workers_spinbox)
        
        self.workers_desc_label = QLabel("(threads for parallel processing)")
        workers_layout.addWidget(self.workers_desc_label)
        workers_layout.addStretch()
        
        # Initialize workers state
        self._toggle_workers()
        
        # Memory management section
        memory_group = QGroupBox("Memory Management")
        main_layout.addWidget(memory_group)
        memory_layout = QVBoxLayout(memory_group)
        memory_layout.setContentsMargins(8, 8, 8, 6)
        memory_layout.setSpacing(4)
        
        # Singleton mode checkbox - will connect handler later after panel widgets created
        self.use_singleton_models_checkbox = self._create_styled_checkbox("Use single model instances (saves RAM, only affects local models)")
        self.use_singleton_models_checkbox.setChecked(self.settings.get('advanced', {}).get('use_singleton_models', True))
        self.use_singleton_models_checkbox.toggled.connect(self._toggle_singleton_controls)
        memory_layout.addWidget(self.use_singleton_models_checkbox)
        
        # Singleton note
        singleton_note = QLabel(
            "When enabled: One bubble detector & one inpainter shared across all images.\n"
            "When disabled: Each thread/image can have its own models (uses more RAM).\n"
            "✅ Batch API translation remains fully functional with singleton mode enabled."
        )
        singleton_note_font = QFont('Arial', 9)
        singleton_note.setFont(singleton_note_font)
        singleton_note.setStyleSheet("color: gray;")
        singleton_note.setWordWrap(True)
        memory_layout.addWidget(singleton_note)
        
        self.auto_cleanup_models_checkbox = self._create_styled_checkbox("Automatically cleanup models after translation to free RAM")
        self.auto_cleanup_models_checkbox.setChecked(self.settings.get('advanced', {}).get('auto_cleanup_models', False))
        memory_layout.addWidget(self.auto_cleanup_models_checkbox)
        
        # Unload models after translation (disabled by default)
        self.unload_models_checkbox = self._create_styled_checkbox("Unload models after translation (reset translator instance)")
        self.unload_models_checkbox.setChecked(self.settings.get('advanced', {}).get('unload_models_after_translation', False))
        memory_layout.addWidget(self.unload_models_checkbox)
        
        # Add a note about parallel processing
        note_label = QLabel("Note: When parallel panel translation is enabled, cleanup happens after ALL panels complete.")
        note_font = QFont('Arial', 9)
        note_label.setFont(note_font)
        note_label.setStyleSheet("color: gray;")
        note_label.setWordWrap(True)
        memory_layout.addWidget(note_label)

        # Panel-level parallel translation
        panel_group = QGroupBox("Parallel Panel Translation")
        main_layout.addWidget(panel_group)
        panel_layout = QVBoxLayout(panel_group)
        panel_layout.setContentsMargins(8, 8, 8, 6)
        panel_layout.setSpacing(4)

        # New: Preload local inpainting for panels (default ON)
        self.preload_local_panels_checkbox = self._create_styled_checkbox("Preload local inpainting instances for panel-parallel runs")
        self.preload_local_panels_checkbox.setChecked(self.settings.get('advanced', {}).get('preload_local_inpainting_for_panels', True))
        panel_layout.addWidget(self.preload_local_panels_checkbox)

        self.parallel_panel_checkbox = self._create_styled_checkbox("Enable parallel panel translation (process multiple images concurrently)")
        self.parallel_panel_checkbox.setChecked(self.settings.get('advanced', {}).get('parallel_panel_translation', False))
        self.parallel_panel_checkbox.toggled.connect(self._toggle_panel_controls)
        panel_layout.addWidget(self.parallel_panel_checkbox)
        
        # Local LLM Performance (add to performance group)
        inpaint_perf_group = QGroupBox("Local LLM Performance")
        perf_layout.addWidget(inpaint_perf_group)
        inpaint_perf_layout = QVBoxLayout(inpaint_perf_group)
        inpaint_perf_layout.setContentsMargins(8, 8, 8, 6)
        inpaint_perf_layout.setSpacing(4)
        
        # RT-DETR Concurrency (for memory optimization)
        rtdetr_conc_widget = QWidget()
        rtdetr_conc_layout = QHBoxLayout(rtdetr_conc_widget)
        rtdetr_conc_layout.setContentsMargins(0, 0, 0, 0)
        inpaint_perf_layout.addWidget(rtdetr_conc_widget)
        self.rtdetr_conc_frame = rtdetr_conc_widget
        
        rtdetr_conc_label = QLabel("RT-DETR Concurrency:")
        rtdetr_conc_label.setMinimumWidth(150)
        rtdetr_conc_layout.addWidget(rtdetr_conc_label)
        
        self.rtdetr_max_concurrency_spinbox = QSpinBox()
        self.rtdetr_max_concurrency_spinbox.setRange(1, 999)
        self.rtdetr_max_concurrency_spinbox.setValue(self.settings['ocr'].get('rtdetr_max_concurrency', 12))
        self.rtdetr_max_concurrency_spinbox.setToolTip("Maximum concurrent RT-DETR region OCR calls (rate limiting handled via delays)")
        rtdetr_conc_layout.addWidget(self.rtdetr_max_concurrency_spinbox)
        
        rtdetr_conc_desc = QLabel("parallel OCR calls (lower = less RAM)")
        rtdetr_conc_desc_font = QFont('Arial', 9)
        rtdetr_conc_desc.setFont(rtdetr_conc_desc_font)
        rtdetr_conc_desc.setStyleSheet("color: gray;")
        rtdetr_conc_layout.addWidget(rtdetr_conc_desc)
        rtdetr_conc_layout.addStretch()
        
        # Initially hide RT-DETR concurrency control until we check detector type
        self.rtdetr_conc_frame.setVisible(False)
        
        # Inpainting Concurrency
        inpaint_bs_frame = QWidget()
        inpaint_bs_layout = QHBoxLayout(inpaint_bs_frame)
        inpaint_bs_layout.setContentsMargins(0, 0, 0, 0)
        inpaint_perf_layout.addWidget(inpaint_bs_frame)
        
        inpaint_bs_label = QLabel("Inpainting Concurrency:")
        inpaint_bs_label.setMinimumWidth(150)
        inpaint_bs_layout.addWidget(inpaint_bs_label)
        
        self.inpaint_batch_size_spinbox = QSpinBox()
        self.inpaint_batch_size_spinbox.setRange(1, 32)
        self.inpaint_batch_size_spinbox.setValue(self.settings.get('inpainting', {}).get('batch_size', 10))
        inpaint_bs_layout.addWidget(self.inpaint_batch_size_spinbox)
        
        inpaint_bs_help = QLabel("(process multiple regions at once)")
        inpaint_bs_help_font = QFont('Arial', 9)
        inpaint_bs_help.setFont(inpaint_bs_help_font)
        inpaint_bs_help.setStyleSheet("color: gray;")
        inpaint_bs_layout.addWidget(inpaint_bs_help)
        inpaint_bs_layout.addStretch()
        
        self.enable_cache_checkbox = self._create_styled_checkbox("Enable inpainting cache (speeds up repeated processing)")
        self.enable_cache_checkbox.setChecked(self.settings.get('inpainting', {}).get('enable_cache', True))
        inpaint_perf_layout.addWidget(self.enable_cache_checkbox)

        # Max concurrent panels
        panels_frame = QWidget()
        panels_layout = QHBoxLayout(panels_frame)
        panels_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(panels_frame)
        
        self.panels_label = QLabel("Max concurrent panels:")
        self.panels_label.setMinimumWidth(150)
        panels_layout.addWidget(self.panels_label)
        
        self.panel_max_workers_spinbox = QSpinBox()
        self.panel_max_workers_spinbox.setRange(1, 999)
        self.panel_max_workers_spinbox.setValue(self.settings.get('advanced', {}).get('panel_max_workers', 2))
        panels_layout.addWidget(self.panel_max_workers_spinbox)
        panels_layout.addStretch()
        
        # Panel start stagger (ms)
        stagger_frame = QWidget()
        stagger_layout = QHBoxLayout(stagger_frame)
        stagger_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(stagger_frame)
        
        self.stagger_label = QLabel("Panel start stagger:")
        self.stagger_label.setMinimumWidth(150)
        stagger_layout.addWidget(self.stagger_label)
        
        self.panel_stagger_ms_spinbox = QSpinBox()
        self.panel_stagger_ms_spinbox.setRange(0, 1000)
        self.panel_stagger_ms_spinbox.setValue(self.settings.get('advanced', {}).get('panel_start_stagger_ms', 30))
        stagger_layout.addWidget(self.panel_stagger_ms_spinbox)
        
        self.stagger_unit_label = QLabel("ms")
        stagger_layout.addWidget(self.stagger_unit_label)
        stagger_layout.addStretch()
        
        # Initialize panel controls state
        self._toggle_panel_controls()
        self._toggle_singleton_controls()

        # ONNX conversion settings
        onnx_group = QGroupBox("ONNX Conversion")
        main_layout.addWidget(onnx_group)
        onnx_layout = QVBoxLayout(onnx_group)
        onnx_layout.setContentsMargins(8, 8, 8, 6)
        onnx_layout.setSpacing(4)
        
        self.auto_convert_onnx_checkbox = self._create_styled_checkbox("Auto-convert local models to ONNX for faster inference (recommended)")
        self.auto_convert_onnx_checkbox.setChecked(self.settings['advanced'].get('auto_convert_to_onnx', False))
        onnx_layout.addWidget(self.auto_convert_onnx_checkbox)
        
        self.auto_convert_onnx_bg_checkbox = self._create_styled_checkbox("Convert in background (non-blocking; switches to ONNX when ready)")
        self.auto_convert_onnx_bg_checkbox.setChecked(self.settings['advanced'].get('auto_convert_to_onnx_background', True))
        onnx_layout.addWidget(self.auto_convert_onnx_bg_checkbox)
        
        # Connect toggle handler
        def _toggle_onnx_controls():
            self.auto_convert_onnx_bg_checkbox.setEnabled(self.auto_convert_onnx_checkbox.isChecked())
        self.auto_convert_onnx_checkbox.toggled.connect(_toggle_onnx_controls)
        _toggle_onnx_controls()
        
        # Model memory optimization (quantization)
        quant_group = QGroupBox("Model Memory Optimization")
        main_layout.addWidget(quant_group)
        quant_layout = QVBoxLayout(quant_group)
        quant_layout.setContentsMargins(8, 8, 8, 6)
        quant_layout.setSpacing(4)
        
        self.quantize_models_checkbox = self._create_styled_checkbox("Reduce RAM with quantized models (global switch)")
        self.quantize_models_checkbox.setChecked(self.settings['advanced'].get('quantize_models', False))
        quant_layout.addWidget(self.quantize_models_checkbox)
        
        # ONNX quantize sub-toggle
        onnx_quant_frame = QWidget()
        onnx_quant_layout = QHBoxLayout(onnx_quant_frame)
        onnx_quant_layout.setContentsMargins(0, 0, 0, 0)
        quant_layout.addWidget(onnx_quant_frame)
        
        self.onnx_quantize_checkbox = self._create_styled_checkbox("Quantize ONNX models to INT8 (dynamic)")
        self.onnx_quantize_checkbox.setChecked(self.settings['advanced'].get('onnx_quantize', False))
        onnx_quant_layout.addWidget(self.onnx_quantize_checkbox)
        
        onnx_quant_help = QLabel("(lower RAM/CPU; slight accuracy trade-off)")
        onnx_quant_help_font = QFont('Arial', 9)
        onnx_quant_help.setFont(onnx_quant_help_font)
        onnx_quant_help.setStyleSheet("color: gray;")
        onnx_quant_layout.addWidget(onnx_quant_help)
        onnx_quant_layout.addStretch()
        
        # Torch precision dropdown
        precision_frame = QWidget()
        precision_layout = QHBoxLayout(precision_frame)
        precision_layout.setContentsMargins(0, 0, 0, 0)
        quant_layout.addWidget(precision_frame)
        
        precision_label = QLabel("Torch precision:")
        precision_label.setMinimumWidth(150)
        precision_layout.addWidget(precision_label)
        
        self.torch_precision_combo = QComboBox()
        self.torch_precision_combo.addItems(['fp16', 'fp32', 'auto'])
        self.torch_precision_combo.setCurrentText(self.settings['advanced'].get('torch_precision', 'fp16'))
        precision_layout.addWidget(self.torch_precision_combo)
        
        precision_help = QLabel("(fp16 only, since fp32 is currently bugged)")
        precision_help_font = QFont('Arial', 9)
        precision_help.setFont(precision_help_font)
        precision_help.setStyleSheet("color: gray;")
        precision_layout.addWidget(precision_help)
        precision_layout.addStretch()
        
        # Aggressive memory cleanup
        cleanup_group = QGroupBox("Memory & Cleanup")
        main_layout.addWidget(cleanup_group)
        cleanup_layout = QVBoxLayout(cleanup_group)
        cleanup_layout.setContentsMargins(8, 8, 8, 6)
        cleanup_layout.setSpacing(4)
        
        self.force_deep_cleanup_checkbox = self._create_styled_checkbox("Force deep model cleanup after every image (slowest, lowest RAM)")
        self.force_deep_cleanup_checkbox.setChecked(self.settings.get('advanced', {}).get('force_deep_cleanup_each_image', False))
        cleanup_layout.addWidget(self.force_deep_cleanup_checkbox)
        
        cleanup_help = QLabel("Also clears shared caches at batch end.")
        cleanup_help_font = QFont('Arial', 9)
        cleanup_help.setFont(cleanup_help_font)
        cleanup_help.setStyleSheet("color: gray;")
        cleanup_layout.addWidget(cleanup_help)
        
        # RAM cap controls
        self.ram_cap_enabled_checkbox = self._create_styled_checkbox("Enable RAM cap")
        self.ram_cap_enabled_checkbox.setChecked(self.settings.get('advanced', {}).get('ram_cap_enabled', False))
        cleanup_layout.addWidget(self.ram_cap_enabled_checkbox)
        
        # RAM cap value
        ramcap_value_frame = QWidget()
        ramcap_value_layout = QHBoxLayout(ramcap_value_frame)
        ramcap_value_layout.setContentsMargins(0, 0, 0, 0)
        cleanup_layout.addWidget(ramcap_value_frame)
        
        ramcap_value_label = QLabel("Max RAM (MB):")
        ramcap_value_label.setMinimumWidth(150)
        ramcap_value_layout.addWidget(ramcap_value_label)
        
        self.ram_cap_mb_spinbox = QSpinBox()
        self.ram_cap_mb_spinbox.setRange(512, 131072)
        ram_cap_mb_value = int(self.settings.get('advanced', {}).get('ram_cap_mb', 4096))
        self.ram_cap_mb_spinbox.setValue(ram_cap_mb_value)
        ramcap_value_layout.addWidget(self.ram_cap_mb_spinbox)
        
        ramcap_value_help = QLabel("(0 = disabled)")
        ramcap_value_help_font = QFont('Arial', 9)
        ramcap_value_help.setFont(ramcap_value_help_font)
        ramcap_value_help.setStyleSheet("color: gray;")
        ramcap_value_layout.addWidget(ramcap_value_help)
        ramcap_value_layout.addStretch()
        
        # RAM cap mode
        ramcap_mode_frame = QWidget()
        ramcap_mode_layout = QHBoxLayout(ramcap_mode_frame)
        ramcap_mode_layout.setContentsMargins(0, 0, 0, 0)
        cleanup_layout.addWidget(ramcap_mode_frame)
        
        ramcap_mode_label = QLabel("Cap mode:")
        ramcap_mode_label.setMinimumWidth(150)
        ramcap_mode_layout.addWidget(ramcap_mode_label)
        
        self.ram_cap_mode_combo = QComboBox()
        self.ram_cap_mode_combo.addItems(['soft', 'hard (Windows only)'])
        ram_cap_mode_value = self.settings.get('advanced', {}).get('ram_cap_mode', 'soft')
        # Handle both 'hard' and 'hard (Windows only)' formats
        if ram_cap_mode_value == 'hard':
            self.ram_cap_mode_combo.setCurrentText('hard (Windows only)')
        else:
            self.ram_cap_mode_combo.setCurrentText('soft')
        ramcap_mode_layout.addWidget(self.ram_cap_mode_combo)
        
        ramcap_mode_help = QLabel("Soft = clean/trim, Hard = OS-enforced (may OOM)")
        ramcap_mode_help_font = QFont('Arial', 9)
        ramcap_mode_help.setFont(ramcap_mode_help_font)
        ramcap_mode_help.setStyleSheet("color: gray;")
        ramcap_mode_layout.addWidget(ramcap_mode_help)
        ramcap_mode_layout.addStretch()
        
        # Advanced RAM gate tuning
        gate_frame = QWidget()
        gate_layout = QHBoxLayout(gate_frame)
        gate_layout.setContentsMargins(0, 0, 0, 0)
        cleanup_layout.addWidget(gate_frame)
        
        gate_label = QLabel("Gate timeout (sec):")
        gate_label.setMinimumWidth(150)
        gate_layout.addWidget(gate_label)
        
        self.ram_gate_timeout_spinbox = QDoubleSpinBox()
        self.ram_gate_timeout_spinbox.setRange(2.0, 60.0)
        self.ram_gate_timeout_spinbox.setSingleStep(0.5)
        ram_gate_timeout_value = float(self.settings.get('advanced', {}).get('ram_gate_timeout_sec', 15.0))
        self.ram_gate_timeout_spinbox.setValue(ram_gate_timeout_value)
        gate_layout.addWidget(self.ram_gate_timeout_spinbox)
        gate_layout.addStretch()
        
        # Gate floor
        floor_frame = QWidget()
        floor_layout = QHBoxLayout(floor_frame)
        floor_layout.setContentsMargins(0, 0, 0, 0)
        cleanup_layout.addWidget(floor_frame)
        
        floor_label = QLabel("Gate floor over baseline (MB):")
        floor_label.setMinimumWidth(180)
        floor_layout.addWidget(floor_label)
        
        self.ram_gate_floor_spinbox = QSpinBox()
        self.ram_gate_floor_spinbox.setRange(64, 2048)
        ram_gate_floor_value = int(self.settings.get('advanced', {}).get('ram_min_floor_over_baseline_mb', 256))
        self.ram_gate_floor_spinbox.setValue(ram_gate_floor_value)
        floor_layout.addWidget(self.ram_gate_floor_spinbox)
        floor_layout.addStretch()
        
        # Update RT-DETR concurrency control visibility based on current detector type
        # This is called after the Advanced tab is fully created to sync with OCR tab state
        QTimer.singleShot(0, self._sync_rtdetr_concurrency_visibility)
        
        # Add stretch at the end to push all content to the top and prevent sections from stretching
        main_layout.addStretch()

    def _sync_rtdetr_concurrency_visibility(self):
        """Sync RT-DETR concurrency control visibility with detector type selection"""
        if hasattr(self, 'detector_type_combo') and hasattr(self, 'rtdetr_conc_frame'):
            detector = self.detector_type_combo.currentText()
            is_rtdetr = 'RT-DETR' in detector or 'RTEDR_onnx' in detector
            self.rtdetr_conc_frame.setVisible(is_rtdetr)

    def _toggle_workers(self):
        """Enable/disable worker settings based on parallel processing toggle"""
        if hasattr(self, 'parallel_processing_checkbox'):
            enabled = bool(self.parallel_processing_checkbox.isChecked())
            if hasattr(self, 'max_workers_spinbox'):
                self.max_workers_spinbox.setEnabled(enabled)
            if hasattr(self, 'workers_label'):
                self.workers_label.setEnabled(enabled)
                self.workers_label.setStyleSheet("color: white;" if enabled else "color: gray;")
            if hasattr(self, 'workers_desc_label'):
                self.workers_desc_label.setEnabled(enabled)
                self.workers_desc_label.setStyleSheet("color: white;" if enabled else "color: gray;")
    
    def _toggle_singleton_controls(self):
        """Enable/disable parallel panel translation based on singleton toggle."""
        # When singleton mode is ENABLED, parallel panel translation should be DISABLED
        try:
            singleton_enabled = bool(self.use_singleton_models_checkbox.isChecked())
        except Exception:
            singleton_enabled = True  # Default
        
        # Disable parallel panel checkbox when singleton is enabled
        if hasattr(self, 'parallel_panel_checkbox'):
            self.parallel_panel_checkbox.setEnabled(not singleton_enabled)
            if singleton_enabled:
                # Also gray out the label when disabled
                pass  # The checkbox itself shows as disabled
    
    def _toggle_panel_controls(self):
        """Enable/disable panel control fields based on parallel panel toggle."""
        try:
            enabled = bool(self.parallel_panel_checkbox.isChecked())
        except Exception:
            enabled = False
        
        # Enable/disable panel control widgets and their labels
        panel_widgets = [
            ('panel_max_workers_spinbox', 'panels_label'),
            ('panel_stagger_ms_spinbox', 'stagger_label', 'stagger_unit_label'),
            ('preload_local_panels_checkbox',)  # Add preload checkbox
        ]
        
        for widget_names in panel_widgets:
            for widget_name in widget_names:
                try:
                    widget = getattr(self, widget_name, None)
                    if widget is not None:
                        # Just use setEnabled() - stylesheet handles visuals
                        widget.setEnabled(enabled)
                except Exception:
                    pass

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

            # Azure (simplified - only merge multiplier remains)
            if hasattr(self, 'azure_merge_multiplier'): self.azure_merge_multiplier.set(float(ocr.get('azure_merge_multiplier', 3.0)))
            try:
                if hasattr(self, '_update_azure_label'):
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
            
            # CRITICAL: Directly update main_gui.config to ensure settings propagate immediately
            if hasattr(self.main_gui, 'config'):
                self.main_gui.config['manga_settings'] = self.settings

            # Mirror only auto max to top-level config for backward compatibility; keep min nested
            try:
                auto_max = self.settings.get('rendering', {}).get('auto_max_size', None)
                if auto_max is not None:
                    self.config['manga_max_font_size'] = int(auto_max)
                    # Also update main_gui.config for immediate effect
                    if hasattr(self.main_gui, 'config'):
                        self.main_gui.config['manga_max_font_size'] = int(auto_max)
            except Exception:
                pass
            
            # Save to file immediately
            if hasattr(self.main_gui, 'save_config'):
                self.main_gui.save_config()
                print(f"✅ Auto-saved rendering settings to main_gui.config")
                time.sleep(0.1)  # Brief pause for stability
                print("💤 Auto-save pausing briefly for stability")
            
        except Exception as e:
            print(f"Error auto-saving rendering settings: {e}")

    def _save_settings(self):
        """Save all settings including expanded iteration controls"""
        try:
            # Collect all preprocessing settings
            self.settings['preprocessing']['enabled'] = self.preprocess_enabled.isChecked()
            self.settings['preprocessing']['auto_detect_quality'] = self.auto_detect.isChecked()
            self.settings['preprocessing']['contrast_threshold'] = self.contrast_threshold.value()
            self.settings['preprocessing']['sharpness_threshold'] = self.sharpness_threshold.value()
            self.settings['preprocessing']['enhancement_strength'] = self.enhancement_strength.value()
            self.settings['preprocessing']['noise_threshold'] = self.noise_threshold.value()
            self.settings['preprocessing']['denoise_strength'] = self.denoise_strength.value()
            self.settings['preprocessing']['max_image_dimension'] = self.dimension_spinbox.value()
            self.settings['preprocessing']['max_image_pixels'] = self.pixels_spinbox.value()
            self.settings['preprocessing']['chunk_height'] = self.chunk_height_spinbox.value()
            self.settings['preprocessing']['chunk_overlap'] = self.chunk_overlap_spinbox.value()
            
            # Compression (saved separately from preprocessing)
            if 'compression' not in self.settings:
                self.settings['compression'] = {}
            self.settings['compression']['enabled'] = bool(self.compression_enabled.isChecked())
            self.settings['compression']['format'] = str(self.compression_format_combo.currentText())
            self.settings['compression']['jpeg_quality'] = int(self.jpeg_quality_spin.value())
            self.settings['compression']['png_compress_level'] = int(self.png_level_spin.value())
            self.settings['compression']['webp_quality'] = int(self.webp_quality_spin.value())
            
            # TILING SETTINGS - save under preprocessing (primary) and mirror under 'tiling' for backward compatibility
            self.settings['preprocessing']['inpaint_tiling_enabled'] = self.inpaint_tiling_enabled.isChecked()
            self.settings['preprocessing']['inpaint_tile_size'] = self.tile_size_spinbox.value()
            self.settings['preprocessing']['inpaint_tile_overlap'] = self.tile_overlap_spinbox.value()
            # Back-compat mirror
            self.settings['tiling'] = {
                'enabled': self.inpaint_tiling_enabled.isChecked(),
                'tile_size': self.tile_size_spinbox.value(),
                'tile_overlap': self.tile_overlap_spinbox.value()
            }
            
            # OCR settings
            self.settings['ocr']['language_hints'] = [code for code, checkbox in self.lang_checkboxes.items() if checkbox.isChecked()]
            # Save as cloud_ocr_confidence (applies to Google/Azure only)
            self.settings['ocr']['cloud_ocr_confidence'] = self.confidence_threshold_slider.value() / 100.0
            # Keep old setting for backward compatibility
            self.settings['ocr']['confidence_threshold'] = self.settings['ocr']['cloud_ocr_confidence']
            self.settings['ocr']['text_detection_mode'] = self.detection_mode_combo.currentText()
            self.settings['ocr']['min_region_size'] = self.min_region_size_spinbox.value()
            self.settings['ocr']['merge_nearby_threshold'] = self.merge_nearby_threshold_spinbox.value()
            self.settings['ocr']['enable_rotation_correction'] = self.enable_rotation_checkbox.isChecked()
            # Azure settings - only merge multiplier remains (new API is synchronous)
            self.settings['ocr']['azure_merge_multiplier'] = self.azure_merge_multiplier_slider.value() / 100.0
            self.settings['ocr']['min_text_length'] = self.min_text_length_spinbox.value()
            self.settings['ocr']['exclude_english_text'] = self.exclude_english_checkbox.isChecked()
            
            # OCR batching & locality
            self.settings['ocr']['ocr_batch_enabled'] = bool(self.ocr_batch_enabled_checkbox.isChecked())
            self.settings['ocr']['ocr_batch_size'] = int(self.ocr_batch_size_spinbox.value())
            self.settings['ocr']['ocr_max_concurrency'] = int(self.ocr_max_conc_spinbox.value())
            self.settings['ocr']['ocr_max_retries'] = int(self.ocr_max_retries_spinbox.value())
            self.settings['ocr']['roi_locality_enabled'] = bool(self.roi_locality_checkbox.isChecked())
            self.settings['ocr']['roi_padding_ratio'] = float(self.roi_padding_ratio_slider.value() / 100.0)
            self.settings['ocr']['roi_min_side_px'] = int(self.roi_min_side_spinbox.value())
            self.settings['ocr']['roi_min_area_px'] = int(self.roi_min_area_spinbox.value())
            self.settings['ocr']['roi_max_side'] = int(self.roi_max_side_spinbox.value())
            self.settings['ocr']['english_exclude_threshold'] = self.english_exclude_threshold_slider.value() / 100.0
            self.settings['ocr']['english_exclude_min_chars'] = self.english_exclude_min_chars_spinbox.value()
            self.settings['ocr']['english_exclude_short_tokens'] = self.english_exclude_short_tokens_checkbox.isChecked()
            
            # Bubble detection settings
            self.settings['ocr']['bubble_detection_enabled'] = self.bubble_detection_enabled_checkbox.isChecked()
            self.settings['ocr']['use_rtdetr_for_ocr_regions'] = self.use_rtdetr_for_ocr_checkbox.isChecked()  # NEW: RT-DETR for OCR guidance
            self.settings['ocr']['enable_fallback_ocr'] = self.enable_fallback_ocr_checkbox.isChecked()  # NEW: Fallback OCR for empty blocks
            # New toggles
            self.settings['ocr']['skip_rtdetr_merging'] = self.skip_rtdetr_merging_checkbox.isChecked()
            self.settings['ocr']['preserve_empty_blocks'] = self.preserve_empty_blocks_checkbox.isChecked()
            self.settings['ocr']['bubble_model_path'] = self.bubble_model_entry.text()
            self.settings['ocr']['bubble_confidence'] = self.bubble_conf_slider.value() / 100.0
            self.settings['ocr']['rtdetr_confidence'] = self.bubble_conf_slider.value() / 100.0
            self.settings['ocr']['detect_empty_bubbles'] = self.detect_empty_bubbles_checkbox.isChecked()
            self.settings['ocr']['detect_text_bubbles'] = self.detect_text_bubbles_checkbox.isChecked()
            self.settings['ocr']['detect_free_text'] = self.detect_free_text_checkbox.isChecked()
            self.settings['ocr']['rtdetr_model_url'] = self.bubble_model_entry.text()
            self.settings['ocr']['bubble_max_detections_yolo'] = int(self.bubble_max_det_yolo_spinbox.value())
            self.settings['ocr']['rtdetr_max_concurrency'] = int(self.rtdetr_max_concurrency_spinbox.value())
            
            # Save the detector type properly
            detector_display = self.detector_type_combo.currentText()
            if 'RTEDR_onnx' in detector_display or 'ONNX' in detector_display.upper():
                self.settings['ocr']['detector_type'] = 'rtdetr_onnx'
            elif 'RT-DETR' in detector_display:
                self.settings['ocr']['detector_type'] = 'rtdetr'
            elif 'YOLOv8' in detector_display:
                self.settings['ocr']['detector_type'] = 'yolo'
            elif detector_display == 'Custom Model':
                self.settings['ocr']['detector_type'] = 'custom'
                self.settings['ocr']['custom_model_path'] = self.bubble_model_entry.text()
            else:
                self.settings['ocr']['detector_type'] = 'rtdetr_onnx'
            
            # Inpainting settings
            if 'inpainting' not in self.settings:
                self.settings['inpainting'] = {}
            self.settings['inpainting']['batch_size'] = self.inpaint_batch_size_spinbox.value()
            self.settings['inpainting']['enable_cache'] = self.enable_cache_checkbox.isChecked()
            
            # Save all dilation settings
            self.settings['mask_dilation'] = self.mask_dilation_spinbox.value()
            kernel_value = self.kernel_size_spinbox.value()
            print(f"[KERNEL_DEBUG] Saving kernel_size to settings: {kernel_value}")
            self.settings['dilation_kernel_size'] = kernel_value
            self.settings['use_all_iterations'] = self.use_all_iterations_checkbox.isChecked()
            self.settings['all_iterations'] = self.all_iterations_spinbox.value()
            self.settings['text_bubble_dilation_iterations'] = self.text_bubble_iter_spinbox.value()
            self.settings['empty_bubble_dilation_iterations'] = self.empty_bubble_iter_spinbox.value()
            self.settings['free_text_dilation_iterations'] = self.free_text_iter_spinbox.value()
            self.settings['auto_iterations'] = self.auto_iterations_checkbox.isChecked()
            
            # Legacy support
            self.settings['bubble_dilation_iterations'] = self.text_bubble_iter_spinbox.value()
            self.settings['dilation_iterations'] = self.text_bubble_iter_spinbox.value()
            
            # Advanced settings
            self.settings['advanced']['format_detection'] = bool(self.format_detection_checkbox.isChecked())
            self.settings['advanced']['webtoon_mode'] = self.webtoon_mode_combo.currentText()
            self.settings['advanced']['debug_mode'] = bool(self.debug_mode_checkbox.isChecked())
            self.settings['advanced']['concise_logs'] = bool(self.concise_logs_checkbox.isChecked())
            self.settings['advanced']['save_intermediate'] = bool(self.save_intermediate_checkbox.isChecked())
            self.settings['advanced']['parallel_processing'] = bool(self.parallel_processing_checkbox.isChecked())
            self.settings['advanced']['max_workers'] = self.max_workers_spinbox.value()
            
            # Save HD strategy settings
            self.settings['advanced']['hd_strategy'] = str(self.hd_strategy_combo.currentText())
            resize_limit_value = int(self.hd_resize_limit_spin.value())
            print(f"[RESIZE_LIMIT_DEBUG] Saving hd_strategy_resize_limit: {resize_limit_value}")
            self.settings['advanced']['hd_strategy_resize_limit'] = resize_limit_value
            self.settings['advanced']['hd_strategy_crop_margin'] = int(self.hd_crop_margin_spin.value())
            self.settings['advanced']['hd_strategy_crop_trigger_size'] = int(self.hd_crop_trigger_spin.value())
            # Also reflect into environment for immediate effect in this session
            os.environ['HD_STRATEGY'] = self.settings['advanced']['hd_strategy']
            os.environ['HD_RESIZE_LIMIT'] = str(self.settings['advanced']['hd_strategy_resize_limit'])
            os.environ['HD_CROP_MARGIN'] = str(self.settings['advanced']['hd_strategy_crop_margin'])
            os.environ['HD_CROP_TRIGGER'] = str(self.settings['advanced']['hd_strategy_crop_trigger_size'])
            
            # Save parallel rendering toggle
            if hasattr(self, 'render_parallel_checkbox'):
                self.settings['advanced']['render_parallel'] = bool(self.render_parallel_checkbox.isChecked())
                
            # Panel-level parallel translation settings
            self.settings['advanced']['parallel_panel_translation'] = bool(self.parallel_panel_checkbox.isChecked())
            self.settings['advanced']['panel_max_workers'] = int(self.panel_max_workers_spinbox.value())
            self.settings['advanced']['panel_start_stagger_ms'] = int(self.panel_stagger_ms_spinbox.value())
            # New: preload local inpainting for panels
            if hasattr(self, 'preload_local_panels_checkbox'):
                self.settings['advanced']['preload_local_inpainting_for_panels'] = bool(self.preload_local_panels_checkbox.isChecked())
            
            # Memory management settings
            self.settings['advanced']['use_singleton_models'] = bool(self.use_singleton_models_checkbox.isChecked())
            self.settings['advanced']['auto_cleanup_models'] = bool(self.auto_cleanup_models_checkbox.isChecked())
            self.settings['advanced']['unload_models_after_translation'] = bool(self.unload_models_checkbox.isChecked() if hasattr(self, 'unload_models_checkbox') else False)
            
            # ONNX auto-convert settings (persist and apply to environment)
            if hasattr(self, 'auto_convert_onnx_checkbox'):
                self.settings['advanced']['auto_convert_to_onnx'] = bool(self.auto_convert_onnx_checkbox.isChecked())
                os.environ['AUTO_CONVERT_TO_ONNX'] = 'true' if self.auto_convert_onnx_checkbox.isChecked() else 'false'
            if hasattr(self, 'auto_convert_onnx_bg_checkbox'):
                self.settings['advanced']['auto_convert_to_onnx_background'] = bool(self.auto_convert_onnx_bg_checkbox.isChecked())
                os.environ['AUTO_CONVERT_TO_ONNX_BACKGROUND'] = 'true' if self.auto_convert_onnx_bg_checkbox.isChecked() else 'false'
            
            # Quantization toggles and precision
            if hasattr(self, 'quantize_models_checkbox'):
                self.settings['advanced']['quantize_models'] = bool(self.quantize_models_checkbox.isChecked())
                os.environ['MODEL_QUANTIZE'] = 'true' if self.quantize_models_checkbox.isChecked() else 'false'
            if hasattr(self, 'onnx_quantize_checkbox'):
                self.settings['advanced']['onnx_quantize'] = bool(self.onnx_quantize_checkbox.isChecked())
                os.environ['ONNX_QUANTIZE'] = 'true' if self.onnx_quantize_checkbox.isChecked() else 'false'
            if hasattr(self, 'torch_precision_combo'):
                self.settings['advanced']['torch_precision'] = str(self.torch_precision_combo.currentText())
                os.environ['TORCH_PRECISION'] = self.settings['advanced']['torch_precision']
            
            # Memory cleanup toggle
            if hasattr(self, 'force_deep_cleanup_checkbox'):
                if 'advanced' not in self.settings:
                    self.settings['advanced'] = {}
                self.settings['advanced']['force_deep_cleanup_each_image'] = bool(self.force_deep_cleanup_checkbox.isChecked())
                
            # RAM cap settings
            if hasattr(self, 'ram_cap_enabled_checkbox'):
                self.settings['advanced']['ram_cap_enabled'] = bool(self.ram_cap_enabled_checkbox.isChecked())
            if hasattr(self, 'ram_cap_mb_spinbox'):
                ram_cap_mb = int(self.ram_cap_mb_spinbox.value())
                self.settings['advanced']['ram_cap_mb'] = ram_cap_mb
            if hasattr(self, 'ram_cap_mode_combo'):
                mode = self.ram_cap_mode_combo.currentText()
                ram_cap_mode = 'hard' if mode.startswith('hard') else 'soft'
                self.settings['advanced']['ram_cap_mode'] = ram_cap_mode
            if hasattr(self, 'ram_gate_timeout_spinbox'):
                ram_gate_timeout = float(self.ram_gate_timeout_spinbox.value())
                self.settings['advanced']['ram_gate_timeout_sec'] = ram_gate_timeout
            if hasattr(self, 'ram_gate_floor_spinbox'):
                ram_gate_floor = int(self.ram_gate_floor_spinbox.value())
                self.settings['advanced']['ram_min_floor_over_baseline_mb'] = ram_gate_floor
            
            # Cloud API settings
            if hasattr(self, 'cloud_model_selected'):
                self.settings['cloud_inpaint_model'] = self.cloud_model_selected
                self.settings['cloud_custom_version'] = self.custom_version_entry.text()
                self.settings['cloud_inpaint_prompt'] = self.cloud_prompt_entry.text()
                self.settings['cloud_negative_prompt'] = self.negative_entry.text()
                self.settings['cloud_inference_steps'] = self.steps_spinbox.value()
                self.settings['cloud_timeout'] = self.cloud_timeout_spinbox.value()
            
            # Clear bubble detector cache to force reload with new settings
            if hasattr(self.main_gui, 'manga_tab') and hasattr(self.main_gui.manga_tab, 'translator'):
                if hasattr(self.main_gui.manga_tab.translator, 'bubble_detector'):
                    self.main_gui.manga_tab.translator.bubble_detector = None
            
            # Save to config - CRITICAL: Update both local and main_gui config
            self.config['manga_settings'] = self.settings
            
            # CRITICAL: Directly update main_gui.config to ensure settings propagate immediately
            # This is essential because many settings require the updated config to take effect
            # without requiring a GUI restart
            if hasattr(self.main_gui, 'config'):
                self.main_gui.config['manga_settings'] = self.settings
                
                # Log key settings that were updated
                ocr_settings = self.settings.get('ocr', {})
                advanced_settings = self.settings.get('advanced', {})
                preprocessing_settings = self.settings.get('preprocessing', {})
                
                print(f"✅ Updated main_gui.config with manga_settings:")
                print(f"   OCR: confidence={ocr_settings.get('cloud_ocr_confidence', 'N/A')}, "
                      f"min_text_length={ocr_settings.get('min_text_length', 'N/A')}, "
                      f"bubble_detection={ocr_settings.get('bubble_detection_enabled', 'N/A')}, "
                      f"use_rtdetr={ocr_settings.get('use_rtdetr_for_ocr_regions', 'N/A')}, "
                      f"enable_fallback_ocr={ocr_settings.get('enable_fallback_ocr', 'N/A')}")
                print(f"   Advanced: debug_mode={advanced_settings.get('debug_mode', 'N/A')}, "
                      f"parallel_processing={advanced_settings.get('parallel_processing', 'N/A')}, "
                      f"max_workers={advanced_settings.get('max_workers', 'N/A')}")
                print(f"   Preprocessing: enabled={preprocessing_settings.get('enabled', 'N/A')}, "
                      f"tiling={preprocessing_settings.get('inpaint_tiling_enabled', 'N/A')}")
            
            # Save to file - using the correct method name
            try:
                if hasattr(self.main_gui, 'save_config'):
                    self.main_gui.save_config(show_message=False)
                    print("Settings saved successfully")
                elif hasattr(self.main_gui, 'save_configuration'):
                    self.main_gui.save_configuration()
                    print("Settings saved successfully")
                else:
                    # Try direct save as fallback
                    if hasattr(self.main_gui, 'config_file'):
                        with open(self.main_gui.config_file, 'w') as f:
                            json.dump(self.config, f, indent=2)
                        print("Settings saved directly to config file")
            except Exception as e:
                print(f"Error saving configuration: {e}")
                QMessageBox.critical(self, "Save Error", f"Failed to save settings: {e}")
                return
            
            # Refresh manga integration status if available (fixes status label not updating after bubble detection changes)
            if hasattr(self.main_gui, 'manga_tab') and hasattr(self.main_gui.manga_tab, '_check_provider_status'):
                try:
                    self.main_gui.manga_tab._check_provider_status()
                except Exception as e:
                    print(f"Error refreshing manga provider status: {e}")
            
            # Call callback if provided
            if self.callback:
                try:
                    self.callback(self.settings)
                except Exception as e:
                    print(f"Error in callback: {e}")
            
            # Close dialog
            self.accept()
                
        except Exception as e:
            import traceback
            print(f"Critical error in _save_settings: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Save Error", f"Failed to save settings: {e}")

    def _reset_defaults(self):
        """Reset by removing manga_settings from config and reinitializing the dialog."""
        reply = QMessageBox.question(self, "Reset Settings", 
                                      "Reset all manga settings to defaults?\nThis will remove custom manga settings from config.json.",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        # Remove manga_settings key to force defaults
        try:
            if isinstance(self.config, dict) and 'manga_settings' in self.config:
                del self.config['manga_settings']
        except Exception:
            pass
        # Persist changes WITHOUT showing message
        try:
            if hasattr(self.main_gui, 'save_config'):
                self.main_gui.save_config(show_message=False)
            elif hasattr(self.main_gui, 'save_configuration'):
                self.main_gui.save_configuration()
            elif hasattr(self.main_gui, 'config_file') and isinstance(self.main_gui.config_file, str):
                with open(self.main_gui.config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception:
            try:
                if hasattr(self.main_gui, 'CONFIG_FILE') and isinstance(self.main_gui.CONFIG_FILE, str):
                    with open(self.main_gui.CONFIG_FILE, 'w', encoding='utf-8') as f:
                        json.dump(self.config, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        # Close and reopen dialog so defaults apply
        self.close()
        try:
            MangaSettingsDialog(parent=self.parent, main_gui=self.main_gui, config=self.config, callback=self.callback)
        except Exception:
            pass  # Don't show any message

    def _cancel(self):
        """Cancel without saving"""
        self.reject()
