# manga_settings_dialog.py
"""
Enhanced settings dialog for manga translation with all settings visible
Properly integrated with TranslatorGUI's WindowManager and UIHelper
"""

import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as tb
from typing import Dict, Any, Optional, Callable
from bubble_detector import BubbleDetector
import logging
import time
import copy

# Use the same logging infrastructure initialized by translator_gui
logger = logging.getLogger(__name__)

class MangaSettingsDialog:
    """Settings dialog for manga translation"""
    
    def __init__(self, parent, main_gui, config: Dict[str, Any], callback: Optional[Callable] = None):
        """Initialize settings dialog
        
        Args:
            parent: Parent window
            main_gui: Reference to TranslatorGUI instance
            config: Configuration dictionary
            callback: Function to call after saving
        """
        self.parent = parent
        self.main_gui = main_gui
        self.config = config
        self.callback = callback
        self.dialog = None
        
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
        """Disable mouse wheel scrolling on a spinbox or combobox"""
        def dummy_scroll(event):
            # Return "break" to prevent the default scroll behavior
            return "break"
        
        # Bind mouse wheel events to the dummy handler
        widget.bind("<MouseWheel>", dummy_scroll)  # Windows
        widget.bind("<Button-4>", dummy_scroll)    # Linux scroll up
        widget.bind("<Button-5>", dummy_scroll)    # Linux scroll down

    def _disable_all_spinbox_scrolling(self, parent):
        """Recursively find and disable scrolling on all spinboxes and comboboxes"""
        for widget in parent.winfo_children():
            # Check if it's a Spinbox (both ttk and tk versions)
            if isinstance(widget, (tb.Spinbox, tk.Spinbox, ttk.Spinbox)):
                self._disable_spinbox_scroll(widget)
            # Check if it's a Combobox (ttk and ttkbootstrap versions)
            elif isinstance(widget, (ttk.Combobox, tb.Combobox)):
                self._disable_spinbox_scroll(widget)
            # Recursively check frames and other containers
            elif hasattr(widget, 'winfo_children'):
                self._disable_all_spinbox_scrolling(widget)
            
    def _create_font_size_controls(self, parent_frame):
        """Create improved font size controls with presets"""
        
        # Font size frame
        font_frame = tk.Frame(parent_frame)
        font_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(font_frame, text="Font Size:", width=20, anchor='w').pack(side=tk.LEFT)
        
        # Font size mode selection
        mode_frame = tk.Frame(font_frame)
        mode_frame.pack(side=tk.LEFT, padx=10)
        
        # Radio buttons for mode
        self.font_size_mode_var = tk.StringVar(value='auto')
        
        modes = [
            ("Auto", "auto", "Automatically fit text to bubble size"),
            ("Fixed", "fixed", "Use a specific font size"),
            ("Scale", "scale", "Scale auto size by percentage")
        ]
        
        for text, value, tooltip in modes:
            rb = ttk.Radiobutton(
                mode_frame,
                text=text,
                variable=self.font_size_mode_var,
                value=value,
                command=self._on_font_mode_change
            )
            rb.pack(side=tk.LEFT, padx=5)
            
            # Add tooltip
            self._create_tooltip(rb, tooltip)
        
        # Controls frame (changes based on mode)
        self.font_controls_frame = tk.Frame(parent_frame)
        self.font_controls_frame.pack(fill=tk.X, pady=5, padx=(20, 0))
        
        # Fixed size controls
        self.fixed_size_frame = tk.Frame(self.font_controls_frame)
        tk.Label(self.fixed_size_frame, text="Size:").pack(side=tk.LEFT)
        
        self.fixed_font_size_var = tk.IntVar(value=16)
        fixed_spin = tb.Spinbox(
            self.fixed_size_frame,
            from_=8,
            to=72,
            textvariable=self.fixed_font_size_var,
            width=10,
            command=self._save_rendering_settings
        )
        fixed_spin.pack(side=tk.LEFT, padx=5)
        
        # Quick presets for fixed size
        tk.Label(self.fixed_size_frame, text="Presets:").pack(side=tk.LEFT, padx=(10, 5))
        
        presets = [
            ("Small", 12),
            ("Medium", 16),
            ("Large", 20),
            ("XL", 24)
        ]
        
        for text, size in presets:
            ttk.Button(
                self.fixed_size_frame,
                text=text,
                command=lambda s=size: self._set_fixed_size(s),
                width=6
            ).pack(side=tk.LEFT, padx=2)
        
        # Scale controls
        self.scale_frame = tk.Frame(self.font_controls_frame)
        tk.Label(self.scale_frame, text="Scale:").pack(side=tk.LEFT)
        
        self.font_scale_var = tk.DoubleVar(value=1.0)
        scale_slider = tk.Scale(
            self.scale_frame,
            from_=0.5,
            to=2.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.font_scale_var,
            length=200,
            command=lambda v: self._update_scale_label()
        )
        scale_slider.pack(side=tk.LEFT, padx=5)
        
        self.scale_label = tk.Label(self.scale_frame, text="100%", width=5)
        self.scale_label.pack(side=tk.LEFT)
        
        # Quick scale presets
        tk.Label(self.scale_frame, text="Quick:").pack(side=tk.LEFT, padx=(10, 5))
        
        scale_presets = [
            ("75%", 0.75),
            ("100%", 1.0),
            ("125%", 1.25),
            ("150%", 1.5)
        ]
        
        for text, scale in scale_presets:
            ttk.Button(
                self.scale_frame,
                text=text,
                command=lambda s=scale: self._set_scale(s),
                width=5
            ).pack(side=tk.LEFT, padx=2)
        
        # Auto size settings
        self.auto_frame = tk.Frame(self.font_controls_frame)
        
        # Min/Max size constraints for auto mode
        constraints_frame = tk.Frame(self.auto_frame)
        constraints_frame.pack(fill=tk.X)
        
        tk.Label(constraints_frame, text="Size Range:").pack(side=tk.LEFT)
        
        tk.Label(constraints_frame, text="Min:").pack(side=tk.LEFT, padx=(10, 2))
        self.min_font_size_var = tk.IntVar(value=10)
        tb.Spinbox(
            constraints_frame,
            from_=6,
            to=20,
            textvariable=self.min_font_size_var,
            width=8,
            command=self._save_rendering_settings
        ).pack(side=tk.LEFT)
        
        tk.Label(constraints_frame, text="Max:").pack(side=tk.LEFT, padx=(10, 2))
        self.max_font_size_var = tk.IntVar(value=28)
        tb.Spinbox(
            constraints_frame,
            from_=16,
            to=48,
            textvariable=self.max_font_size_var,
            width=8,
            command=self._save_rendering_settings
        ).pack(side=tk.LEFT)
        
        # Auto fit quality
        quality_frame = tk.Frame(self.auto_frame)
        quality_frame.pack(fill=tk.X, pady=(5, 0))
        
        tk.Label(quality_frame, text="Fit Style:").pack(side=tk.LEFT)
        
        self.auto_fit_style_var = tk.StringVar(value='balanced')
        
        fit_styles = [
            ("Compact", "compact", "Fit more text, smaller size"),
            ("Balanced", "balanced", "Balance readability and fit"),
            ("Readable", "readable", "Prefer larger, more readable text")
        ]
        
        for text, value, tooltip in fit_styles:
            rb = ttk.Radiobutton(
                quality_frame,
                text=text,
                variable=self.auto_fit_style_var,
                value=value,
                command=self._save_rendering_settings
            )
            rb.pack(side=tk.LEFT, padx=5)
            self._create_tooltip(rb, tooltip)
        
        # Initialize the correct frame
        self._on_font_mode_change()

    def _on_font_mode_change(self):
        """Show/hide appropriate font controls based on mode"""
        # Hide all frames
        for frame in [self.fixed_size_frame, self.scale_frame, self.auto_frame]:
            frame.pack_forget()
        
        # Show the appropriate frame
        mode = self.font_size_mode_var.get()
        if mode == 'fixed':
            self.fixed_size_frame.pack(fill=tk.X)
        elif mode == 'scale':
            self.scale_frame.pack(fill=tk.X)
        else:  # auto
            self.auto_frame.pack(fill=tk.X)
        
        self._save_rendering_settings()

    def _set_fixed_size(self, size):
        """Set fixed font size from preset"""
        self.fixed_font_size_var.set(size)
        self._save_rendering_settings()

    def _set_scale(self, scale):
        """Set font scale from preset"""
        self.font_scale_var.set(scale)
        self._update_scale_label()
        self._save_rendering_settings()

    def _update_scale_label(self):
        """Update the scale percentage label"""
        scale = self.font_scale_var.get()
        self.scale_label.config(text=f"{int(scale * 100)}%")
        self._save_rendering_settings()

    def _create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = tk.Label(
                tooltip,
                text=text,
                background="#ffffe0",
                relief=tk.SOLID,
                borderwidth=1,
                font=('Arial', 9)
            )
            label.pack()
            
            widget.tooltip = tooltip
        
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
        
        widget.bind('<Enter>', on_enter)
        widget.bind('<Leave>', on_leave)
    
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
        """Display the settings dialog using WindowManager"""
        # Set initialization flag to prevent auto-saves during setup
        self._initializing = True
        
        # Use WindowManager to create scrollable dialog
        if self.main_gui.wm._force_safe_ratios:
            max_width_ratio = 0.5
            max_height_ratio = 0.85
        else:
            max_width_ratio = 0.5
            max_height_ratio = 1.05
            
        self.dialog, scrollable_frame, canvas = self.main_gui.wm.setup_scrollable(
            self.parent,
            "Manga Translation Settings",
            width=None,
            height=None,
            max_width_ratio=max_width_ratio,
            max_height_ratio=max_height_ratio 
        )
        
        # Store canvas reference for potential cleanup
        self.canvas = canvas
        
        # Create main content frame (that will scroll)
        content_frame = tk.Frame(scrollable_frame)
        content_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create notebook for tabs inside the content frame
        notebook = ttk.Notebook(content_frame)
        notebook.pack(fill='both', expand=True)
        
        # Create all tabs
        self._create_preprocessing_tab(notebook)
        self._create_ocr_tab(notebook)
        self._create_inpainting_tab(notebook)
        self._create_advanced_tab(notebook)
        # NOTE: Font Sizing tab removed; controls are now in Manga Integration UI
        
        # Cloud API tab
        self.cloud_tab = ttk.Frame(notebook)
        notebook.add(self.cloud_tab, text="Cloud API")
        self._create_cloud_api_tab(self.cloud_tab)
        
        # DISABLE SCROLL WHEEL ON ALL SPINBOXES
        self.dialog.after(10, lambda: self._disable_all_spinbox_scrolling(self.dialog))
        
        # Clear initialization flag after setup is complete
        self._initializing = False
        
        # Create fixed button frame at bottom of dialog (not inside scrollable content)
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(fill='x', padx=10, pady=(5, 10), side='bottom')
        
        # Buttons
        tb.Button(
            button_frame,
            text="Save",
            command=self._save_settings,
            bootstyle="success"
        ).pack(side='right', padx=(5, 0))
        
        tb.Button(
            button_frame,
            text="Cancel",
            command=self._cancel,
            bootstyle="secondary"
        ).pack(side='right', padx=(5, 0))
        
        tb.Button(
            button_frame,
            text="Reset to Defaults",
            command=self._reset_defaults,
            bootstyle="warning"
        ).pack(side='left')
        
        # Initialize preprocessing state
        self._toggle_preprocessing()
        
        # Handle window close protocol
        self.dialog.protocol("WM_DELETE_WINDOW", self._cancel)
    
    def _create_preprocessing_tab(self, notebook):
        """Create preprocessing settings tab with all options"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Preprocessing")
        
        # Main scrollable content
        content_frame = tk.Frame(frame)
        content_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Enable preprocessing with command
        enable_frame = tk.Frame(content_frame)
        enable_frame.pack(fill='x', padx=20, pady=(20, 10))
        
        self.preprocess_enabled = tk.BooleanVar(value=self.settings['preprocessing']['enabled'])
        tb.Checkbutton(
            enable_frame,
            text="Enable Image Preprocessing",
            variable=self.preprocess_enabled,
            bootstyle="round-toggle",
            command=self._toggle_preprocessing 
        ).pack(anchor='w')
        
        # Store all preprocessing controls for enable/disable
        self.preprocessing_controls = []
        
        # Auto quality detection
        self.auto_detect = tk.BooleanVar(value=self.settings['preprocessing']['auto_detect_quality'])
        auto_cb = tb.Checkbutton(
            enable_frame,
            text="Auto-detect image quality issues",
            variable=self.auto_detect,
            bootstyle="round-toggle"
        )
        auto_cb.pack(anchor='w', pady=(10, 0))
        self.preprocessing_controls.append(auto_cb)
        
        # Quality thresholds section
        threshold_frame = tk.LabelFrame(content_frame, text="Image Enhancement", padx=15, pady=10)
        threshold_frame.pack(fill='x', padx=20, pady=(10, 0))
        self.preprocessing_controls.append(threshold_frame)
        
        # Contrast threshold
        contrast_frame = tk.Frame(threshold_frame)
        contrast_frame.pack(fill='x', pady=5)
        contrast_label = tk.Label(contrast_frame, text="Contrast Adjustment:", width=20, anchor='w')
        contrast_label.pack(side='left')
        self.preprocessing_controls.append(contrast_label)
        
        self.contrast_threshold = tk.DoubleVar(value=self.settings['preprocessing']['contrast_threshold'])
        contrast_scale = tk.Scale(
            contrast_frame,
            from_=0.0, to=1.0,
            resolution=0.01,
            orient='horizontal',
            variable=self.contrast_threshold,
            length=250
        )
        contrast_scale.pack(side='left', padx=10)
        self.preprocessing_controls.append(contrast_scale)
        
        contrast_value = tk.Label(contrast_frame, textvariable=self.contrast_threshold, width=5)
        contrast_value.pack(side='left')
        self.preprocessing_controls.append(contrast_value)
        
        # Sharpness threshold
        sharpness_frame = tk.Frame(threshold_frame)
        sharpness_frame.pack(fill='x', pady=5)
        sharpness_label = tk.Label(sharpness_frame, text="Sharpness Enhancement:", width=20, anchor='w')
        sharpness_label.pack(side='left')
        self.preprocessing_controls.append(sharpness_label)
        
        self.sharpness_threshold = tk.DoubleVar(value=self.settings['preprocessing']['sharpness_threshold'])
        sharpness_scale = tk.Scale(
            sharpness_frame,
            from_=0.0, to=1.0,
            resolution=0.01,
            orient='horizontal',
            variable=self.sharpness_threshold,
            length=250
        )
        sharpness_scale.pack(side='left', padx=10)
        self.preprocessing_controls.append(sharpness_scale)
        
        sharpness_value = tk.Label(sharpness_frame, textvariable=self.sharpness_threshold, width=5)
        sharpness_value.pack(side='left')
        self.preprocessing_controls.append(sharpness_value)
        
        # Enhancement strength
        enhance_frame = tk.Frame(threshold_frame)
        enhance_frame.pack(fill='x', pady=5)
        enhance_label = tk.Label(enhance_frame, text="Overall Enhancement:", width=20, anchor='w')
        enhance_label.pack(side='left')
        self.preprocessing_controls.append(enhance_label)
        
        self.enhancement_strength = tk.DoubleVar(value=self.settings['preprocessing']['enhancement_strength'])
        enhance_scale = tk.Scale(
            enhance_frame,
            from_=0.0, to=3.0,
            resolution=0.01,
            orient='horizontal',
            variable=self.enhancement_strength,
            length=250
        )
        enhance_scale.pack(side='left', padx=10)
        self.preprocessing_controls.append(enhance_scale)
        
        enhance_value = tk.Label(enhance_frame, textvariable=self.enhancement_strength, width=5)
        enhance_value.pack(side='left')
        self.preprocessing_controls.append(enhance_value)
        
        # Noise reduction section
        noise_frame = tk.LabelFrame(content_frame, text="Noise Reduction", padx=15, pady=10)
        noise_frame.pack(fill='x', padx=20, pady=(10, 0))
        self.preprocessing_controls.append(noise_frame)
        
        # Noise threshold
        noise_threshold_frame = tk.Frame(noise_frame)
        noise_threshold_frame.pack(fill='x', pady=5)
        noise_label = tk.Label(noise_threshold_frame, text="Noise Threshold:", width=20, anchor='w')
        noise_label.pack(side='left')
        self.preprocessing_controls.append(noise_label)
        
        self.noise_threshold = tk.IntVar(value=self.settings['preprocessing']['noise_threshold'])
        noise_scale = tk.Scale(
            noise_threshold_frame,
            from_=0, to=50,
            orient='horizontal',
            variable=self.noise_threshold,
            length=250
        )
        noise_scale.pack(side='left', padx=10)
        self.preprocessing_controls.append(noise_scale)
        
        noise_value = tk.Label(noise_threshold_frame, textvariable=self.noise_threshold, width=5)
        noise_value.pack(side='left')
        self.preprocessing_controls.append(noise_value)
        
        # Denoise strength
        denoise_frame = tk.Frame(noise_frame)
        denoise_frame.pack(fill='x', pady=5)
        denoise_label = tk.Label(denoise_frame, text="Denoise Strength:", width=20, anchor='w')
        denoise_label.pack(side='left')
        self.preprocessing_controls.append(denoise_label)
        
        self.denoise_strength = tk.IntVar(value=self.settings['preprocessing']['denoise_strength'])
        denoise_scale = tk.Scale(
            denoise_frame,
            from_=0, to=30,
            orient='horizontal',
            variable=self.denoise_strength,
            length=250
        )
        denoise_scale.pack(side='left', padx=10)
        self.preprocessing_controls.append(denoise_scale)
        
        denoise_value = tk.Label(denoise_frame, textvariable=self.denoise_strength, width=5)
        denoise_value.pack(side='left')
        self.preprocessing_controls.append(denoise_value)
        
        # Size limits section
        size_frame = tk.LabelFrame(content_frame, text="Image Size Limits", padx=15, pady=10)
        size_frame.pack(fill='x', padx=20, pady=(10, 0))
        self.preprocessing_controls.append(size_frame)
        
        # Max dimension
        dimension_frame = tk.Frame(size_frame)
        dimension_frame.pack(fill='x', pady=5)
        dimension_label = tk.Label(dimension_frame, text="Max Dimension:", width=20, anchor='w')
        dimension_label.pack(side='left')
        self.preprocessing_controls.append(dimension_label)
        
        self.max_dimension = tk.IntVar(value=self.settings['preprocessing']['max_image_dimension'])
        self.dimension_spinbox = tb.Spinbox(
            dimension_frame,
            from_=500,
            to=4000,
            textvariable=self.max_dimension,
            increment=100,
            width=10
        )
        self.dimension_spinbox.pack(side='left', padx=10)
        self.preprocessing_controls.append(self.dimension_spinbox)
        
        tk.Label(dimension_frame, text="pixels").pack(side='left')
        
        # Max pixels
        pixels_frame = tk.Frame(size_frame)
        pixels_frame.pack(fill='x', pady=5)
        pixels_label = tk.Label(pixels_frame, text="Max Total Pixels:", width=20, anchor='w')
        pixels_label.pack(side='left')
        self.preprocessing_controls.append(pixels_label)
        
        self.max_pixels = tk.IntVar(value=self.settings['preprocessing']['max_image_pixels'])
        self.pixels_spinbox = tb.Spinbox(
            pixels_frame,
            from_=1000000,
            to=10000000,
            textvariable=self.max_pixels,
            increment=100000,
            width=10
        )
        self.pixels_spinbox.pack(side='left', padx=10)
        self.preprocessing_controls.append(self.pixels_spinbox)
        
        tk.Label(pixels_frame, text="pixels").pack(side='left')
        
        # Compression section
        compression_frame = tk.LabelFrame(content_frame, text="Image Compression (applies to OCR uploads)", padx=15, pady=10)
        compression_frame.pack(fill='x', padx=20, pady=(10, 0))
        # Do NOT add compression controls to preprocessing_controls; keep independent of preprocessing toggle
        
        # Enable compression toggle
        self.compression_enabled_var = tk.BooleanVar(value=self.settings.get('compression', {}).get('enabled', False))
        compression_toggle = tb.Checkbutton(
            compression_frame,
            text="Enable compression for OCR uploads",
            variable=self.compression_enabled_var,
            bootstyle="round-toggle",
        )
        compression_toggle.pack(anchor='w')
        self.compression_toggle = compression_toggle
        
        # Hook toggle to enable/disable compression fields
        def _toggle_compression_enabled():
            enabled = bool(self.compression_enabled_var.get())
            state = 'normal' if enabled else 'disabled'
            try:
                self.compression_format_combo.config(state='readonly' if enabled else 'disabled')
            except Exception:
                pass
            for w in [getattr(self, 'jpeg_quality_spin', None), getattr(self, 'png_level_spin', None), getattr(self, 'webp_quality_spin', None)]:
                try:
                    if w is not None:
                        w.config(state=state)
                except Exception:
                    pass
        compression_toggle.config(command=_toggle_compression_enabled)
        
        # Format selection
        format_row = tk.Frame(compression_frame)
        format_row.pack(fill='x', pady=5)
        tk.Label(format_row, text="Format:", width=20, anchor='w').pack(side='left')
        self.compression_format_var = tk.StringVar(value=self.settings.get('compression', {}).get('format', 'jpeg'))
        self.compression_format_combo = ttk.Combobox(
            format_row,
            textvariable=self.compression_format_var,
            values=['jpeg', 'png', 'webp'],
            state='readonly',
            width=10
        )
        self.compression_format_combo.pack(side='left', padx=10)
        
        # JPEG quality
        self.jpeg_row = tk.Frame(compression_frame)
        self.jpeg_row.pack(fill='x', pady=5)
        tk.Label(self.jpeg_row, text="JPEG Quality:", width=20, anchor='w').pack(side='left')
        self.jpeg_quality_var = tk.IntVar(value=self.settings.get('compression', {}).get('jpeg_quality', 85))
        self.jpeg_quality_spin = tb.Spinbox(
            self.jpeg_row,
            from_=1,
            to=95,
            textvariable=self.jpeg_quality_var,
            width=10
        )
        self.jpeg_quality_spin.pack(side='left', padx=10)
        tk.Label(self.jpeg_row, text="(higher = better quality, larger size)", font=('Arial', 9), fg='gray').pack(side='left')
        
        # PNG compression level
        self.png_row = tk.Frame(compression_frame)
        self.png_row.pack(fill='x', pady=5)
        tk.Label(self.png_row, text="PNG Compression:", width=20, anchor='w').pack(side='left')
        self.png_level_var = tk.IntVar(value=self.settings.get('compression', {}).get('png_compress_level', 6))
        self.png_level_spin = tb.Spinbox(
            self.png_row,
            from_=0,
            to=9,
            textvariable=self.png_level_var,
            width=10
        )
        self.png_level_spin.pack(side='left', padx=10)
        tk.Label(self.png_row, text="(0 = fastest, 9 = smallest)", font=('Arial', 9), fg='gray').pack(side='left')
        
        # WEBP quality
        self.webp_row = tk.Frame(compression_frame)
        self.webp_row.pack(fill='x', pady=5)
        tk.Label(self.webp_row, text="WEBP Quality:", width=20, anchor='w').pack(side='left')
        self.webp_quality_var = tk.IntVar(value=self.settings.get('compression', {}).get('webp_quality', 85))
        self.webp_quality_spin = tb.Spinbox(
            self.webp_row,
            from_=1,
            to=100,
            textvariable=self.webp_quality_var,
            width=10
        )
        self.webp_quality_spin.pack(side='left', padx=10)
        tk.Label(self.webp_row, text="(higher = better quality, larger size)", font=('Arial', 9), fg='gray').pack(side='left')
        
        # Hook to toggle visibility based on format
        self.compression_format_combo.bind('<<ComboboxSelected>>', lambda e: self._toggle_compression_format())
        self._toggle_compression_format()
        # Apply enabled/disabled state for compression fields initially
        try:
            _toggle_compression_enabled()
        except Exception:
            pass
        
        # Chunk settings for large images (moved above compression)
        chunk_frame = tk.LabelFrame(content_frame, text="Large Image Processing", padx=15, pady=10)
        chunk_frame.pack(fill='x', padx=20, pady=(10, 0), before=compression_frame)
        self.preprocessing_controls.append(chunk_frame)
        
        # HD Strategy (Inpainting acceleration)
        hd_frame = tk.LabelFrame(chunk_frame, text="Inpainting HD Strategy", padx=10, pady=8)
        hd_frame.pack(fill='x', pady=(5, 10))
        
        # Strategy selector
        strat_row = tk.Frame(hd_frame)
        strat_row.pack(fill='x', pady=4)
        tk.Label(strat_row, text="Strategy:", width=20, anchor='w').pack(side='left')
        self.hd_strategy_var = tk.StringVar(value=self.settings.get('advanced', {}).get('hd_strategy', 'resize'))
        self.hd_strategy_combo = ttk.Combobox(
            strat_row,
            textvariable=self.hd_strategy_var,
            values=['original', 'resize', 'crop'],
            state='readonly',
            width=12
        )
        self.hd_strategy_combo.pack(side='left', padx=10)
        tk.Label(strat_row, text="(original = legacy full-image; resize/crop = faster)", font=('Arial', 9), fg='gray').pack(side='left')
        
        # Resize limit row
        self.hd_resize_row = tk.Frame(hd_frame)
        self.hd_resize_row.pack(fill='x', pady=4)
        tk.Label(self.hd_resize_row, text="Resize limit (long edge):", width=20, anchor='w').pack(side='left')
        self.hd_resize_limit_var = tk.IntVar(value=int(self.settings.get('advanced', {}).get('hd_strategy_resize_limit', 1536)))
        self.hd_resize_limit_spin = tb.Spinbox(
            self.hd_resize_row,
            from_=512,
            to=4096,
            textvariable=self.hd_resize_limit_var,
            increment=64,
            width=10
        )
        self.hd_resize_limit_spin.pack(side='left', padx=10)
        tk.Label(self.hd_resize_row, text="px").pack(side='left')
        
        # Crop params rows
        self.hd_crop_margin_row = tk.Frame(hd_frame)
        self.hd_crop_margin_row.pack(fill='x', pady=4)
        tk.Label(self.hd_crop_margin_row, text="Crop margin:", width=20, anchor='w').pack(side='left')
        self.hd_crop_margin_var = tk.IntVar(value=int(self.settings.get('advanced', {}).get('hd_strategy_crop_margin', 16)))
        self.hd_crop_margin_spin = tb.Spinbox(
            self.hd_crop_margin_row,
            from_=0,
            to=256,
            textvariable=self.hd_crop_margin_var,
            increment=2,
            width=10
        )
        self.hd_crop_margin_spin.pack(side='left', padx=10)
        tk.Label(self.hd_crop_margin_row, text="px").pack(side='left')
        
        self.hd_crop_trigger_row = tk.Frame(hd_frame)
        self.hd_crop_trigger_row.pack(fill='x', pady=4)
        tk.Label(self.hd_crop_trigger_row, text="Crop trigger size:", width=20, anchor='w').pack(side='left')
        self.hd_crop_trigger_var = tk.IntVar(value=int(self.settings.get('advanced', {}).get('hd_strategy_crop_trigger_size', 1024)))
        self.hd_crop_trigger_spin = tb.Spinbox(
            self.hd_crop_trigger_row,
            from_=256,
            to=4096,
            textvariable=self.hd_crop_trigger_var,
            increment=64,
            width=10
        )
        self.hd_crop_trigger_spin.pack(side='left', padx=10)
        tk.Label(self.hd_crop_trigger_row, text="px (apply crop only if long edge > trigger)").pack(side='left')
        
        # Toggle rows based on current selection
        def _on_hd_strategy_change(*_):
            strat = self.hd_strategy_var.get()
            try:
                if strat == 'resize':
                    self.hd_resize_row.pack(fill='x', pady=4)
                    self.hd_crop_margin_row.pack_forget()
                    self.hd_crop_trigger_row.pack_forget()
                elif strat == 'crop':
                    self.hd_resize_row.pack_forget()
                    self.hd_crop_margin_row.pack(fill='x', pady=4)
                    self.hd_crop_trigger_row.pack(fill='x', pady=4)
                else:  # original
                    self.hd_resize_row.pack_forget()
                    self.hd_crop_margin_row.pack_forget()
                    self.hd_crop_trigger_row.pack_forget()
            except Exception:
                pass
        
        self.hd_strategy_combo.bind('<<ComboboxSelected>>', _on_hd_strategy_change)
        _on_hd_strategy_change()
        
        # Clarifying note about precedence with tiling
        try:
            tk.Label(
                hd_frame,
                text="Note: HD Strategy (resize/crop) takes precedence over Inpainting Tiling when it triggers.\nSet strategy to 'original' if you want tiling to control large-image behavior.",
                font=('Arial', 9),
                fg='gray',
                justify='left'
            ).pack(anchor='w', pady=(2, 2))
        except Exception:
            pass
        
        # Chunk height
        self.chunk_frame = chunk_frame
        chunk_height_frame = tk.Frame(chunk_frame)
        chunk_height_frame.pack(fill='x', pady=5)
        self.chunk_height_label = tk.Label(chunk_height_frame, text="Chunk Height:", width=20, anchor='w')
        self.chunk_height_label.pack(side='left')
        self.preprocessing_controls.append(self.chunk_height_label)
        
        self.chunk_height = tk.IntVar(value=self.settings['preprocessing']['chunk_height'])
        self.chunk_height_spinbox = tb.Spinbox(
            chunk_height_frame,
            from_=500,
            to=2000,
            textvariable=self.chunk_height,
            increment=100,
            width=10
        )
        self.chunk_height_spinbox.pack(side='left', padx=10)
        self.preprocessing_controls.append(self.chunk_height_spinbox)
        
        self.chunk_height_unit_label = tk.Label(chunk_height_frame, text="pixels")
        self.chunk_height_unit_label.pack(side='left')
        self.preprocessing_controls.append(self.chunk_height_unit_label)
        
        # Chunk overlap
        chunk_overlap_frame = tk.Frame(chunk_frame)
        chunk_overlap_frame.pack(fill='x', pady=5)
        self.chunk_overlap_label = tk.Label(chunk_overlap_frame, text="Chunk Overlap:", width=20, anchor='w')
        self.chunk_overlap_label.pack(side='left')
        self.preprocessing_controls.append(self.chunk_overlap_label)
        
        self.chunk_overlap = tk.IntVar(value=self.settings['preprocessing']['chunk_overlap'])
        self.chunk_overlap_spinbox = tb.Spinbox(
            chunk_overlap_frame,
            from_=0,
            to=200,
            textvariable=self.chunk_overlap,
            increment=10,
            width=10
        )
        self.chunk_overlap_spinbox.pack(side='left', padx=10)
        self.preprocessing_controls.append(self.chunk_overlap_spinbox)
        
        self.chunk_overlap_unit_label = tk.Label(chunk_overlap_frame, text="pixels")
        self.chunk_overlap_unit_label.pack(side='left')
        self.preprocessing_controls.append(self.chunk_overlap_unit_label)

        # Inpainting Tiling section (add after the "Large Image Processing" section)
        self.tiling_frame = tk.LabelFrame(content_frame, text="Inpainting Tiling", padx=15, pady=10)
        self.tiling_frame.pack(fill='x', padx=20, pady=(10, 0))
        tiling_frame = self.tiling_frame
        self.preprocessing_controls.append(self.tiling_frame)

        # Enable tiling
        # Prefer values from legacy 'tiling' section if present, otherwise use 'preprocessing'
        tiling_enabled_value = self.settings['preprocessing'].get('inpaint_tiling_enabled', False)
        if 'tiling' in self.settings and isinstance(self.settings['tiling'], dict) and 'enabled' in self.settings['tiling']:
            tiling_enabled_value = self.settings['tiling']['enabled']
        self.inpaint_tiling_enabled = tk.BooleanVar(value=tiling_enabled_value)
        tiling_enable_cb = tb.Checkbutton(
            tiling_frame,
            text="Enable automatic tiling for inpainting (processes large images in tiles)",
            variable=self.inpaint_tiling_enabled,
            command=lambda: self._toggle_tiling_controls(),
            bootstyle="round-toggle"
        )
        tiling_enable_cb.pack(anchor='w', pady=(5, 10))

        # Tile size
        tile_size_frame = tk.Frame(tiling_frame)
        tile_size_frame.pack(fill='x', pady=5)
        tile_size_label = tk.Label(tile_size_frame, text="Tile Size:", width=20, anchor='w')
        tile_size_label.pack(side='left')

        tile_size_value = self.settings['preprocessing'].get('inpaint_tile_size', 512)
        if 'tiling' in self.settings and isinstance(self.settings['tiling'], dict) and 'tile_size' in self.settings['tiling']:
            tile_size_value = self.settings['tiling']['tile_size']
        self.inpaint_tile_size = tk.IntVar(value=tile_size_value)
        self.tile_size_spinbox = tb.Spinbox(
            tile_size_frame,
            from_=256,
            to=2048,
            textvariable=self.inpaint_tile_size,
            increment=128,
            width=10
        )
        self.tile_size_spinbox.pack(side='left', padx=10)

        tk.Label(tile_size_frame, text="pixels").pack(side='left')
        # Initial tiling fields state
        try:
            self._toggle_tiling_controls()
        except Exception:
            pass

        # Tile overlap
        tile_overlap_frame = tk.Frame(tiling_frame)
        tile_overlap_frame.pack(fill='x', pady=5)
        tile_overlap_label = tk.Label(tile_overlap_frame, text="Tile Overlap:", width=20, anchor='w')
        tile_overlap_label.pack(side='left')

        tile_overlap_value = self.settings['preprocessing'].get('inpaint_tile_overlap', 64)
        if 'tiling' in self.settings and isinstance(self.settings['tiling'], dict) and 'tile_overlap' in self.settings['tiling']:
            tile_overlap_value = self.settings['tiling']['tile_overlap']
        self.inpaint_tile_overlap = tk.IntVar(value=tile_overlap_value)
        self.tile_overlap_spinbox = tb.Spinbox(
            tile_overlap_frame,
            from_=0,
            to=256,
            textvariable=self.inpaint_tile_overlap,
            increment=16,
            width=10
        )
        self.tile_overlap_spinbox.pack(side='left', padx=10)

        tk.Label(tile_overlap_frame, text="pixels").pack(side='left')

    def _create_inpainting_tab(self, notebook):
        """Create inpainting settings tab with comprehensive per-text-type dilation controls"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Inpainting")
        
        content_frame = tk.Frame(frame)
        content_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # General Mask Settings (applies to all inpainting methods)
        mask_frame = tk.LabelFrame(content_frame, text="Mask Settings", padx=15, pady=10)
        mask_frame.pack(fill='x', padx=20, pady=(20, 10))
        
        # Auto toggle (affects both mask dilation and iterations)
        auto_global_frame = tk.Frame(mask_frame)
        auto_global_frame.pack(fill='x', pady=(0, 5))
        if not hasattr(self, 'auto_iterations_var'):
            self.auto_iterations_var = tk.BooleanVar(value=self.settings.get('auto_iterations', True))
        tb.Checkbutton(
            auto_global_frame,
            text="Auto (affects mask dilation and iterations)",
            variable=self.auto_iterations_var,
            command=self._toggle_iteration_controls,
            bootstyle="round-toggle"
        ).pack(anchor='w')

        # Mask dilation size
        dilation_frame = tk.Frame(mask_frame)
        dilation_frame.pack(fill='x', pady=5)
        
        tk.Label(dilation_frame, text="Mask Dilation:", width=15, anchor='w').pack(side='left')
        self.mask_dilation_var = tk.IntVar(value=self.settings.get('mask_dilation', 15))
        self.mask_dilation_spinbox = tb.Spinbox(
            dilation_frame,
            from_=0,
            to=50,
            textvariable=self.mask_dilation_var,
            increment=5,
            width=10
        )
        self.mask_dilation_spinbox.pack(side='left', padx=10)
        tk.Label(dilation_frame, text="pixels (expand mask beyond text)").pack(side='left')
        
        # Per-Text-Type Iterations - EXPANDED SECTION
        iterations_label_frame = tk.LabelFrame(mask_frame, text="Dilation Iterations Control", padx=10, pady=5)
        iterations_label_frame.pack(fill='x', pady=(10, 5))
        
        # All Iterations Master Control (NEW)
        all_iter_frame = tk.Frame(iterations_label_frame)
        all_iter_frame.pack(fill='x', pady=5)
        
        # Auto-iterations toggle (secondary control reflects the same setting)
        if not hasattr(self, 'auto_iterations_var'):
            self.auto_iterations_var = tk.BooleanVar(value=self.settings.get('auto_iterations', True))
        auto_iter_checkbox = tb.Checkbutton(
            all_iter_frame,
            text="Auto (set by image: B&W vs Color)",
            variable=self.auto_iterations_var,
            command=self._toggle_iteration_controls,
            bootstyle="round-toggle"
        )
        auto_iter_checkbox.pack(side='left', padx=(0, 10))
        
        # Checkbox to enable/disable uniform iterations
        self.use_all_iterations_var = tk.BooleanVar(value=self.settings.get('use_all_iterations', True))
        all_iter_checkbox = tb.Checkbutton(
            all_iter_frame,
            text="Use Same For All:",
            variable=self.use_all_iterations_var,
            command=self._toggle_iteration_controls,
            bootstyle="round-toggle"
        )
        all_iter_checkbox.pack(side='left', padx=(0, 10))
        self.use_all_iterations_checkbox = all_iter_checkbox
        
        self.all_iterations_var = tk.IntVar(value=self.settings.get('all_iterations', 2))
        self.all_iterations_spinbox = tb.Spinbox(
            all_iter_frame,
            from_=0,
            to=5,
            textvariable=self.all_iterations_var,
            width=10,
            state='disabled' if not self.use_all_iterations_var.get() else 'normal'
        )
        self.all_iterations_spinbox.pack(side='left', padx=10)
        tk.Label(all_iter_frame, text="iterations (applies to all text types)").pack(side='left')
        
        # Separator
        ttk.Separator(iterations_label_frame, orient='horizontal').pack(fill='x', pady=(10, 5))
        
        # Individual Controls Label
        tk.Label(
            iterations_label_frame, 
            text="Individual Text Type Controls:",
            font=('Arial', 9, 'bold')
        ).pack(anchor='w', pady=(5, 5))
        
        # Text Bubble iterations (modified from original bubble iterations)
        text_bubble_iter_frame = tk.Frame(iterations_label_frame)
        text_bubble_iter_frame.pack(fill='x', pady=5)
        
        text_bubble_label = tk.Label(text_bubble_iter_frame, text="Text Bubbles:", width=15, anchor='w')
        text_bubble_label.pack(side='left')
        self.text_bubble_iterations_var = tk.IntVar(value=self.settings.get('text_bubble_dilation_iterations', 
                                                                            self.settings.get('bubble_dilation_iterations', 2)))
        self.text_bubble_iter_spinbox = tb.Spinbox(
            text_bubble_iter_frame,
            from_=0,
            to=5,
            textvariable=self.text_bubble_iterations_var,
            width=10
        )
        self.text_bubble_iter_spinbox.pack(side='left', padx=10)
        tk.Label(text_bubble_iter_frame, text="iterations (speech/dialogue bubbles)").pack(side='left')
        
        # Empty Bubble iterations (NEW)
        empty_bubble_iter_frame = tk.Frame(iterations_label_frame)
        empty_bubble_iter_frame.pack(fill='x', pady=5)
        
        empty_bubble_label = tk.Label(empty_bubble_iter_frame, text="Empty Bubbles:", width=15, anchor='w')
        empty_bubble_label.pack(side='left')
        self.empty_bubble_iterations_var = tk.IntVar(value=self.settings.get('empty_bubble_dilation_iterations', 3))
        self.empty_bubble_iter_spinbox = tb.Spinbox(
            empty_bubble_iter_frame,
            from_=0,
            to=5,
            textvariable=self.empty_bubble_iterations_var,
            width=10
        )
        self.empty_bubble_iter_spinbox.pack(side='left', padx=10)
        tk.Label(empty_bubble_iter_frame, text="iterations (empty speech bubbles)").pack(side='left')
        
        # Free text iterations
        free_text_iter_frame = tk.Frame(iterations_label_frame)
        free_text_iter_frame.pack(fill='x', pady=5)
        
        free_text_label = tk.Label(free_text_iter_frame, text="Free Text:", width=15, anchor='w')
        free_text_label.pack(side='left')
        self.free_text_iterations_var = tk.IntVar(value=self.settings.get('free_text_dilation_iterations', 0))
        self.free_text_iter_spinbox = tb.Spinbox(
            free_text_iter_frame,
            from_=0,
            to=5,
            textvariable=self.free_text_iterations_var,
            width=10
        )
        self.free_text_iter_spinbox.pack(side='left', padx=10)
        tk.Label(free_text_iter_frame, text="iterations (0 = perfect for B&W panels)").pack(side='left')
        
        # Store individual control widgets for enable/disable
        self.individual_iteration_controls = [
            (text_bubble_label, self.text_bubble_iter_spinbox),
            (empty_bubble_label, self.empty_bubble_iter_spinbox),
            (free_text_label, self.free_text_iter_spinbox)
        ]
        
        # Apply initial state
        self._toggle_iteration_controls()
        
        # Legacy iterations (backwards compatibility)
        self.bubble_iterations_var = self.text_bubble_iterations_var  # Link to text bubble for legacy
        self.dilation_iterations_var = self.text_bubble_iterations_var  # Legacy support
        
        # Quick presets - UPDATED VERSION
        preset_frame = tk.Frame(mask_frame)
        preset_frame.pack(fill='x', pady=(10, 5))
        
        tk.Label(preset_frame, text="Quick Presets:").pack(side='left', padx=(0, 10))
        
        tb.Button(
            preset_frame,
            text="B&W Manga",
            command=lambda: self._set_mask_preset(15, False, 2, 2, 3, 0),
            bootstyle="secondary",
            width=12
        ).pack(side='left', padx=2)
        
        tb.Button(
            preset_frame,
            text="Colored",
            command=lambda: self._set_mask_preset(15, False, 2, 2, 3, 3),
            bootstyle="secondary",
            width=12
        ).pack(side='left', padx=2)
        
        tb.Button(
            preset_frame,
            text="Uniform",
            command=lambda: self._set_mask_preset(0, True, 2, 2, 2, 0),
            bootstyle="secondary",
            width=12
        ).pack(side='left', padx=2)
        
        # Help text - UPDATED
        tk.Label(
            mask_frame,
            text="💡 B&W Manga: Optimized for black & white panels with clean bubbles\n"
                 "💡 Colored: For colored manga with complex backgrounds\n"
                 "💡 Aggressive: For difficult text removal cases\n"
                 "💡 Uniform: Good for Manga-OCR\n"
                 "ℹ️ Empty bubbles often need more iterations than text bubbles\n"
                 "ℹ️ Set Free Text to 0 for crisp B&W panels without bleeding",
            font=('Arial', 9),
            fg='gray',
            justify='left'
        ).pack(anchor='w', pady=(10, 0))
        
        # Note about method selection
        info_frame = tk.Frame(content_frame)
        info_frame.pack(fill='x', padx=20, pady=(20, 0))
        
        tk.Label(
            info_frame,
            text="ℹ️ Note: Inpainting method (Cloud/Local) and model selection are configured\n"
                 "     in the Manga tab when you select images for translation.",
            font=('Arial', 10),
            fg='#4a9eff',
            justify='left'
        ).pack(anchor='w')

    def _toggle_iteration_controls(self):
        """Enable/disable iteration controls based on Auto and 'Use Same For All' toggles"""
        auto_on = getattr(self, 'auto_iterations_var', tk.BooleanVar(value=True)).get()
        use_all = self.use_all_iterations_var.get()
        
        if auto_on:
            # Disable everything when auto is on
            try:
                self.all_iterations_spinbox.config(state='disabled')
            except Exception:
                pass
            try:
                if hasattr(self, 'use_all_iterations_checkbox'):
                    self.use_all_iterations_checkbox.config(state='disabled')
            except Exception:
                pass
            try:
                if hasattr(self, 'mask_dilation_spinbox'):
                    self.mask_dilation_spinbox.config(state='disabled')
            except Exception:
                pass
            for label, spinbox in getattr(self, 'individual_iteration_controls', []):
                try:
                    spinbox.config(state='disabled')
                    label.config(fg='gray')
                except Exception:
                    pass
            return
        
        # Auto off -> respect 'use all'
        try:
            self.all_iterations_spinbox.config(state='normal' if use_all else 'disabled')
        except Exception:
            pass
        try:
            if hasattr(self, 'use_all_iterations_checkbox'):
                self.use_all_iterations_checkbox.config(state='normal')
        except Exception:
            pass
        try:
            if hasattr(self, 'mask_dilation_spinbox'):
                self.mask_dilation_spinbox.config(state='normal')
        except Exception:
            pass
        for label, spinbox in getattr(self, 'individual_iteration_controls', []):
            state = 'disabled' if use_all else 'normal'
            try:
                spinbox.config(state=state)
                label.config(fg='gray' if use_all else 'white')
            except Exception:
                pass

    def _set_mask_preset(self, dilation, use_all, all_iter, text_bubble_iter, empty_bubble_iter, free_text_iter):
        """Set mask dilation preset values with comprehensive iteration controls"""
        self.mask_dilation_var.set(dilation)
        self.use_all_iterations_var.set(use_all)
        self.all_iterations_var.set(all_iter)
        self.text_bubble_iterations_var.set(text_bubble_iter)
        self.empty_bubble_iterations_var.set(empty_bubble_iter)
        self.free_text_iterations_var.set(free_text_iter)
        self._toggle_iteration_controls()
    
    def _create_cloud_api_tab(self, parent):
            """Create cloud API settings tab"""
            # NO CANVAS - JUST USE PARENT DIRECTLY
            frame = parent
            
            # API Model Selection
            model_frame = tk.LabelFrame(frame, text="Inpainting Model", padx=15, pady=10)
            model_frame.pack(fill='x', padx=20, pady=(20, 0))
            
            tk.Label(model_frame, text="Select the Replicate model to use for inpainting:").pack(anchor='w', pady=(0, 10))
            
            # Model options
            self.cloud_model_var = tk.StringVar(value=self.settings.get('cloud_inpaint_model', 'ideogram-v2'))
            
            models = [
                ('ideogram-v2', 'Ideogram V2 (Best quality, with prompts)', 'ideogram-ai/ideogram-v2'),
                ('sd-inpainting', 'Stable Diffusion Inpainting (Classic, fast)', 'stability-ai/stable-diffusion-inpainting'),
                ('flux-inpainting', 'FLUX Dev Inpainting (High quality)', 'zsxkib/flux-dev-inpainting'),
                ('custom', 'Custom Model (Enter model identifier)', '')
            ]
            
            for value, text, model_id in models:
                row_frame = tk.Frame(model_frame)
                row_frame.pack(fill='x', pady=2)
                
                rb = tb.Radiobutton(
                    row_frame,
                    text=text,
                    variable=self.cloud_model_var,
                    value=value,
                    command=self._on_cloud_model_change
                )
                rb.pack(side='left')
                
                if model_id:
                    tk.Label(row_frame, text=f"({model_id})", font=('Arial', 8), fg='gray').pack(side='left', padx=(10, 0))
            
            # Custom version ID (now model identifier)
            self.custom_version_frame = tk.Frame(model_frame)
            self.custom_version_frame.pack(fill='x', pady=(10, 0))
            
            tk.Label(self.custom_version_frame, text="Model ID:", width=15, anchor='w').pack(side='left')
            self.custom_version_var = tk.StringVar(value=self.settings.get('cloud_custom_version', ''))
            self.custom_version_entry = tk.Entry(self.custom_version_frame, textvariable=self.custom_version_var, width=50)
            self.custom_version_entry.pack(side='left', padx=10)
            
            # Add helper text for custom model
            helper_text = tk.Label(
                self.custom_version_frame, 
                text="Format: owner/model-name (e.g. stability-ai/stable-diffusion-inpainting)",
                font=('Arial', 8), 
                fg='gray'
            )
            helper_text.pack(anchor='w', padx=(70, 0), pady=(2, 0))
            
            # Initially hide custom version entry
            if self.cloud_model_var.get() != 'custom':
                self.custom_version_frame.pack_forget()
            
            # Performance Settings
            perf_frame = tk.LabelFrame(frame, text="Performance Settings", padx=15, pady=10)
            perf_frame.pack(fill='x', padx=20, pady=(20, 0))
    
            # Timeout
            timeout_frame = tk.Frame(perf_frame)
            timeout_frame.pack(fill='x', pady=5)
            
            tk.Label(timeout_frame, text="API Timeout:", width=15, anchor='w').pack(side='left')
            self.cloud_timeout_var = tk.IntVar(value=self.settings.get('cloud_timeout', 60))
            timeout_spinbox = tb.Spinbox(
                timeout_frame,
                from_=30,
                to=300,
                textvariable=self.cloud_timeout_var,
                width=10
            )
            timeout_spinbox.pack(side='left', padx=10)
            tk.Label(timeout_frame, text="seconds", font=('Arial', 9)).pack(side='left')
            
            # Help text
            help_frame = tk.Frame(frame)
            help_frame.pack(fill='x', padx=20, pady=20)
            
            help_text = tk.Label(
                help_frame,
                text="💡 Tips:\n"
                     "• Ideogram V2 is currently the best quality option\n"
                     "• SD inpainting is fast and supports prompts\n"
                     "• FLUX inpainting offers high quality results\n"
                     "• Find more models at replicate.com/collections/inpainting",
                font=('Arial', 9),
                fg='gray',
                justify='left'
            )
            help_text.pack(anchor='w')
            
            # Prompt Settings (for all models except custom)
            self.prompt_frame = tk.LabelFrame(frame, text="Prompt Settings", padx=15, pady=10)
            self.prompt_frame.pack(fill='x', padx=20, pady=(0, 20))
            
            # Positive prompt
            tk.Label(self.prompt_frame, text="Inpainting Prompt:").pack(anchor='w', pady=(0, 5))
            self.cloud_prompt_var = tk.StringVar(value=self.settings.get('cloud_inpaint_prompt', 'clean background, smooth surface'))
            prompt_entry = tk.Entry(self.prompt_frame, textvariable=self.cloud_prompt_var, width=60)
            prompt_entry.pack(fill='x', padx=(20, 20))
            
            # Add note about prompts
            tk.Label(
                self.prompt_frame, 
                text="Tip: Describe what you want in the inpainted area (e.g., 'white wall', 'wooden floor')",
                font=('Arial', 8), 
                fg='gray'
            ).pack(anchor='w', padx=(20, 0), pady=(2, 10))
            
            # Negative prompt (mainly for SD)
            self.negative_prompt_label = tk.Label(self.prompt_frame, text="Negative Prompt (SD only):")
            self.negative_prompt_label.pack(anchor='w', pady=(0, 5))
            self.cloud_negative_prompt_var = tk.StringVar(value=self.settings.get('cloud_negative_prompt', 'text, writing, letters'))
            self.negative_entry = tk.Entry(self.prompt_frame, textvariable=self.cloud_negative_prompt_var, width=60)
            self.negative_entry.pack(fill='x', padx=(20, 20))
            
            # Inference steps (for SD)
            self.steps_frame = tk.Frame(self.prompt_frame)
            self.steps_frame.pack(fill='x', pady=(10, 5))
            
            self.steps_label = tk.Label(self.steps_frame, text="Inference Steps (SD only):", width=20, anchor='w')
            self.steps_label.pack(side='left', padx=(20, 0))
            self.cloud_steps_var = tk.IntVar(value=self.settings.get('cloud_inference_steps', 20))
            self.steps_spinbox = tb.Spinbox(
                self.steps_frame,
                from_=10,
                to=50,
                textvariable=self.cloud_steps_var,
                width=10
            )
            self.steps_spinbox.pack(side='left', padx=10)
            tk.Label(self.steps_frame, text="(Higher = better quality, slower)", font=('Arial', 9), fg='gray').pack(side='left')
            
            # Initially hide prompt frame if not using appropriate model
            if self.cloud_model_var.get() == 'custom':
                self.prompt_frame.pack_forget()
            
            # Show/hide SD-specific options based on model
            self._on_cloud_model_change()
    
    def _on_cloud_model_change(self):
        """Handle cloud model selection change"""
        model = self.cloud_model_var.get()
        
        # Show/hide custom version entry
        if model == 'custom':
            self.custom_version_frame.pack(fill='x', pady=(10, 0))
            # DON'T HIDE THE PROMPT FRAME FOR CUSTOM MODELS
            self.prompt_frame.pack(fill='x', padx=20, pady=(20, 0))
        else:
            self.custom_version_frame.pack_forget()
            self.prompt_frame.pack(fill='x', padx=20, pady=(20, 0))
        
        # Show/hide SD-specific options
        if model == 'sd-inpainting':
            # Show negative prompt and steps
            self.negative_prompt_label.pack(anchor='w', pady=(10, 5))
            self.negative_entry.pack(fill='x', padx=(20, 0))
            self.steps_frame.pack(fill='x', pady=(10, 0))
        else:
            # Hide SD-specific options
            self.negative_prompt_label.pack_forget()
            self.negative_entry.pack_forget()
            self.steps_frame.pack_forget()
        
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

