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
                'chunk_height': 1000,
                'chunk_overlap': 100
            },
            'ocr': {
                'language_hints': ['ja', 'ko', 'zh'],
                'confidence_threshold': 0.7,
                'merge_nearby_threshold': 20,
                'azure_merge_multiplier': 3.0,
                'text_detection_mode': 'document',
                'enable_rotation_correction': True,
                'bubble_detection_enabled': False,
                'bubble_model_path': '',
                'bubble_confidence': 0.5,
                'detector_type': 'rtdetr',
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
                'exclude_english_text': False
            },
            'advanced': {
                'format_detection': True,
                'webtoon_mode': 'auto',
                'debug_mode': False,
                'save_intermediate': False,
                'parallel_processing': False,
                'max_workers': 4
            },
            'inpainting': {
                'batch_size': 1,
                'enable_cache': True
            },
            'font_sizing': {
            'algorithm': 'smart',  # 'smart', 'conservative', 'aggressive'
            'prefer_larger': True,  # Prefer larger readable text
            'max_lines': 10,  # Maximum lines before forcing smaller
            'line_spacing': 1.3,  # Line height multiplier
            'bubble_size_factor': True  # Scale font based on bubble size
            },
            
            # Mask dilation settings with new iteration controls
            'mask_dilation': 15,
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
        
        # Create notebook for tabs
        notebook = ttk.Notebook(scrollable_frame)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create all tabs
        self._create_preprocessing_tab(notebook)
        self._create_ocr_tab(notebook)
        self._create_inpainting_tab(notebook)
        self._create_advanced_tab(notebook)
        self._create_font_sizing_tab(notebook)
        
        # Cloud API tab
        self.cloud_tab = ttk.Frame(notebook)
        notebook.add(self.cloud_tab, text="Cloud API")
        self._create_cloud_api_tab(self.cloud_tab)
        
        # DISABLE SCROLL WHEEL ON ALL SPINBOXES
        self.dialog.after(10, lambda: self._disable_all_spinbox_scrolling(self.dialog))       
        
        # Button frame at bottom (inside scrollable frame for proper scrolling)
        button_frame = tk.Frame(scrollable_frame)
        button_frame.pack(fill='x', padx=10, pady=(10, 20))
        
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
        
        # Chunk settings for large images
        chunk_frame = tk.LabelFrame(content_frame, text="Large Image Processing", padx=15, pady=10)
        chunk_frame.pack(fill='x', padx=20, pady=(10, 0))
        self.preprocessing_controls.append(chunk_frame)
        
        # Chunk height
        chunk_height_frame = tk.Frame(chunk_frame)
        chunk_height_frame.pack(fill='x', pady=5)
        chunk_height_label = tk.Label(chunk_height_frame, text="Chunk Height:", width=20, anchor='w')
        chunk_height_label.pack(side='left')
        self.preprocessing_controls.append(chunk_height_label)
        
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
        
        tk.Label(chunk_height_frame, text="pixels").pack(side='left')
        
        # Chunk overlap
        chunk_overlap_frame = tk.Frame(chunk_frame)
        chunk_overlap_frame.pack(fill='x', pady=5)
        chunk_overlap_label = tk.Label(chunk_overlap_frame, text="Chunk Overlap:", width=20, anchor='w')
        chunk_overlap_label.pack(side='left')
        self.preprocessing_controls.append(chunk_overlap_label)
        
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
        
        tk.Label(chunk_overlap_frame, text="pixels").pack(side='left')

    def _create_inpainting_tab(self, notebook):
        """Create inpainting settings tab with comprehensive per-text-type dilation controls"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Inpainting")
        
        content_frame = tk.Frame(frame)
        content_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # General Mask Settings (applies to all inpainting methods)
        mask_frame = tk.LabelFrame(content_frame, text="Mask Settings", padx=15, pady=10)
        mask_frame.pack(fill='x', padx=20, pady=(20, 10))
        
        # Mask dilation size
        dilation_frame = tk.Frame(mask_frame)
        dilation_frame.pack(fill='x', pady=5)
        
        tk.Label(dilation_frame, text="Mask Dilation:", width=15, anchor='w').pack(side='left')
        self.mask_dilation_var = tk.IntVar(value=self.settings.get('mask_dilation', 15))
        dilation_spinbox = tb.Spinbox(
            dilation_frame,
            from_=0,
            to=50,
            textvariable=self.mask_dilation_var,
            increment=5,
            width=10
        )
        dilation_spinbox.pack(side='left', padx=10)
        tk.Label(dilation_frame, text="pixels (expand mask beyond text)").pack(side='left')
        
        # Per-Text-Type Iterations - EXPANDED SECTION
        iterations_label_frame = tk.LabelFrame(mask_frame, text="Dilation Iterations Control", padx=10, pady=5)
        iterations_label_frame.pack(fill='x', pady=(10, 5))
        
        # All Iterations Master Control (NEW)
        all_iter_frame = tk.Frame(iterations_label_frame)
        all_iter_frame.pack(fill='x', pady=5)
        
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
        
        # Performance settings  
        perf_frame = tk.LabelFrame(content_frame, text="Performance", padx=15, pady=10)
        perf_frame.pack(fill='x', padx=20, pady=(10, 0))
        
        # Batch size for processing
        batch_frame = tk.Frame(perf_frame)
        batch_frame.pack(fill='x', pady=5)
        
        tk.Label(batch_frame, text="Batch Size:", width=15, anchor='w').pack(side='left')
        
        self.inpaint_batch_size = tk.IntVar(
            value=self.settings.get('inpainting', {}).get('batch_size', 1)
        )
        
        tb.Spinbox(
            batch_frame,
            from_=1,
            to=10,
            textvariable=self.inpaint_batch_size,
            width=10
        ).pack(side='left', padx=10)
        
        tk.Label(
            batch_frame,
            text="(Process multiple regions at once)",
            font=('Arial', 9),
            fg='gray'
        ).pack(side='left')
        
        # Cache settings
        cache_frame = tk.Frame(perf_frame)
        cache_frame.pack(fill='x', pady=5)
        
        self.enable_cache_var = tk.BooleanVar(
            value=self.settings.get('inpainting', {}).get('enable_cache', True)
        )
        
        tb.Checkbutton(
            cache_frame,
            text="Enable inpainting cache (speeds up repeated processing)",
            variable=self.enable_cache_var,
            bootstyle="round-toggle"
        ).pack(anchor='w')
        
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
        """Enable/disable individual iteration controls based on 'use all' checkbox"""
        use_all = self.use_all_iterations_var.get()
        
        # Enable/disable the all iterations spinbox
        self.all_iterations_spinbox.config(state='normal' if use_all else 'disabled')
        
        # Enable/disable individual controls
        for label, spinbox in self.individual_iteration_controls:
            state = 'disabled' if use_all else 'normal'
            spinbox.config(state=state)
            # Gray out labels when disabled
            label.config(fg='gray' if use_all else 'white')

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
        
        for control in self.preprocessing_controls:
            if isinstance(control, (tk.Scale, tb.Spinbox, tb.Checkbutton)):
                control.config(state='normal' if enabled else 'disabled')
            elif isinstance(control, tk.LabelFrame):
                # For LabelFrame, change the foreground color
                control.config(fg='white' if enabled else 'gray')
            elif isinstance(control, tk.Label):
                # For labels, change the foreground color
                control.config(fg='white' if enabled else 'gray')
            elif isinstance(control, tk.Frame):
                # For frames, recursively disable children
                self._toggle_frame_children(control, enabled)

    def _toggle_frame_children(self, frame, enabled):
        """Recursively enable/disable all children of a frame"""
        for child in frame.winfo_children():
            if isinstance(child, (tk.Scale, tb.Spinbox, tb.Checkbutton)):
                child.config(state='normal' if enabled else 'disabled')
            elif isinstance(child, tk.Label):
                child.config(fg='white' if enabled else 'gray')
            elif isinstance(child, tk.Frame):
                self._toggle_frame_children(child, enabled)
    
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
            text="Exclude primarily English text (>70% English characters)",
            variable=self.exclude_english_var,
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
            'RT-DETR': 'ogkalu/comic-text-and-bubble-detector',
            'YOLOv8 Speech': 'ogkalu/comic-speech-bubble-detector-yolov8m',
            'YOLOv8 Text': 'ogkalu/comic-text-segmenter-yolov8m',
            'YOLOv8 Manga': 'ogkalu/manga-text-detector-yolov8s',
            'Custom Model': ''
        }

        # Get saved detector type
        saved_type = self.settings['ocr'].get('detector_type', 'rtdetr')
        if saved_type == 'rtdetr':
            initial_selection = 'RT-DETR'
        elif saved_type == 'yolo':
            initial_selection = 'YOLOv8 Speech'
        elif saved_type == 'custom':
            initial_selection = 'Custom Model'
        else:
            initial_selection = 'RT-DETR'

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

        default_conf = 0.3 if 'RT-DETR' in self.detector_type.get() else 0.5
        
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
            self.bubble_conf_scale
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
        if 'RT-DETR' in detector:
            self.rtdetr_classes_frame.pack(fill='x', pady=(10, 0), after=self.rtdetr_load_btn.master)
        else:
            self.rtdetr_classes_frame.pack_forget()
        
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
            
            if 'RT-DETR' in detector:
                from bubble_detector import BubbleDetector
                bd = BubbleDetector()
                
                if bd.load_rtdetr_model(model_id=model_url):
                    self.rtdetr_status_label.config(text="✅ Downloaded", fg='green')
                    messagebox.showinfo("Success", f"RT-DETR model downloaded successfully!")
                else:
                    self.rtdetr_status_label.config(text="❌ Failed", fg='red')
                    messagebox.showerror("Error", f"Failed to download RT-DETR model")
            else:
                # Download YOLOv8 model
                from huggingface_hub import hf_hub_download
                
                filename_map = {
                    'ogkalu/comic-speech-bubble-detector-yolov8m': 'comic-speech-bubble-detector.pt',
                    'ogkalu/comic-text-segmenter-yolov8m': 'comic-text-segmenter.pt',
                    'ogkalu/manga-text-detector-yolov8s': 'manga-text-detector.pt'
                }
                
                filename = filename_map.get(model_url, 'model.pt')
                local_path = hf_hub_download(repo_id=model_url, filename=filename)
                
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
                    if translator.bubble_detector.rtdetr_loaded:
                        self.rtdetr_status_label.config(text="✅ Loaded", fg='green')
                        return True
                    elif translator.bubble_detector.model_loaded:
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
            
            if 'RT-DETR' in detector:
                if bd.load_rtdetr_model(model_id=model_path):
                    self.rtdetr_status_label.config(text="✅ Ready", fg='green')
                    messagebox.showinfo("Success", f"RT-DETR model loaded successfully!")
                else:
                    self.rtdetr_status_label.config(text="❌ Failed", fg='red')
            else:
                # Load YOLOv8 model
                if bd.load_model(model_path):
                    self.rtdetr_status_label.config(text="✅ Ready", fg='green')
                    messagebox.showinfo("Success", f"YOLOv8 model loaded successfully!")
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
        
        self.save_intermediate = tk.IntVar(value=1 if self.settings['advanced']['save_intermediate'] else 0)
        tb.Checkbutton(
            debug_frame,
            text="Save intermediate images (preprocessed, detection overlays)",
            variable=self.save_intermediate,
            bootstyle="round-toggle"
        ).pack(anchor='w', pady=(5, 0))
        
        # Performance settings
        perf_frame = tk.LabelFrame(content_frame, text="Performance", padx=15, pady=10)
        perf_frame.pack(fill='x', padx=20)
        
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
    
    def _toggle_workers(self):
        """Enable/disable worker settings based on parallel processing toggle"""
        enabled = bool(self.parallel_processing.get())
        self.workers_spinbox.config(state='normal' if enabled else 'disabled')
        self.workers_label.config(fg='white' if enabled else 'gray')

    def _create_font_sizing_tab(self, notebook):
        """Create font sizing settings tab with improved controls"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Font Sizing")
        
        # Main content
        content_frame = tk.Frame(frame)
        content_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Algorithm selection
        algo_frame = tk.LabelFrame(content_frame, text="Sizing Algorithm", padx=15, pady=10)
        algo_frame.pack(fill='x', padx=20, pady=(20, 10))
        
        tk.Label(algo_frame, text="Select font sizing behavior:").pack(anchor='w', pady=(0, 10))
        
        self.font_algorithm_var = tk.StringVar(value=self.settings.get('font_sizing', {}).get('algorithm', 'smart'))
        
        algorithms = [
            ('smart', 'Smart', 'Adapts to bubble size intelligently'),
            ('conservative', 'Conservative', 'Prefers smaller, safer sizes'),
            ('aggressive', 'Aggressive', 'Maximizes text size when possible')
        ]
        
        for value, text, tooltip in algorithms:
            rb = ttk.Radiobutton(
                algo_frame,
                text=text,
                variable=self.font_algorithm_var,
                value=value
            )
            rb.pack(anchor='w', padx=20, pady=2)
            # Add tooltip
            self._create_tooltip(rb, tooltip)
        
        # Size limits
        limits_frame = tk.LabelFrame(content_frame, text="Size Limits", padx=15, pady=10)
        limits_frame.pack(fill='x', padx=20, pady=(10, 0))
        
        # Minimum font size
        min_frame = tk.Frame(limits_frame)
        min_frame.pack(fill='x', pady=5)
        
        tk.Label(min_frame, text="Minimum Font Size:", width=20, anchor='w').pack(side='left')
        self.min_font_size_var = tk.IntVar(value=self.settings.get('font_sizing', {}).get('min_size', 10))
        min_spinbox = tb.Spinbox(
            min_frame,
            from_=8,
            to=20,
            textvariable=self.min_font_size_var,
            width=10
        )
        min_spinbox.pack(side='left', padx=10)
        tk.Label(min_frame, text="pixels").pack(side='left')
        
        # Maximum font size
        max_frame = tk.Frame(limits_frame)
        max_frame.pack(fill='x', pady=5)
        
        tk.Label(max_frame, text="Maximum Font Size:", width=20, anchor='w').pack(side='left')
        self.max_font_size_var = tk.IntVar(value=self.settings.get('font_sizing', {}).get('max_size', 40))
        max_spinbox = tb.Spinbox(
            max_frame,
            from_=20,
            to=60,
            textvariable=self.max_font_size_var,
            width=10
        )
        max_spinbox.pack(side='left', padx=10)
        tk.Label(max_frame, text="pixels").pack(side='left')
        
        # Readable minimum
        readable_frame = tk.Frame(limits_frame)
        readable_frame.pack(fill='x', pady=5)
        
        tk.Label(readable_frame, text="Minimum Readable:", width=20, anchor='w').pack(side='left')
        self.min_readable_var = tk.IntVar(value=self.settings.get('font_sizing', {}).get('min_readable', 14))
        readable_spinbox = tb.Spinbox(
            readable_frame,
            from_=10,
            to=20,
            textvariable=self.min_readable_var,
            width=10
        )
        readable_spinbox.pack(side='left', padx=10)
        tk.Label(readable_frame, text="pixels (won't go below this)").pack(side='left')
        
        # Behavior settings
        behavior_frame = tk.LabelFrame(content_frame, text="Sizing Behavior", padx=15, pady=10)
        behavior_frame.pack(fill='x', padx=20, pady=(10, 0))
        
        # Prefer larger text
        self.prefer_larger_var = tk.BooleanVar(
            value=self.settings.get('font_sizing', {}).get('prefer_larger', True)
        )
        tb.Checkbutton(
            behavior_frame,
            text="Prefer larger text when possible",
            variable=self.prefer_larger_var,
            bootstyle="round-toggle"
        ).pack(anchor='w', pady=2)
        
        # Bubble size factor
        self.bubble_size_factor_var = tk.BooleanVar(
            value=self.settings.get('font_sizing', {}).get('bubble_size_factor', True)
        )
        tb.Checkbutton(
            behavior_frame,
            text="Scale font based on bubble size (smaller bubbles = smaller text)",
            variable=self.bubble_size_factor_var,
            bootstyle="round-toggle"
        ).pack(anchor='w', pady=2)
        
        # Line spacing
        spacing_frame = tk.Frame(behavior_frame)
        spacing_frame.pack(fill='x', pady=(10, 5))
        
        tk.Label(spacing_frame, text="Line Spacing:", width=20, anchor='w').pack(side='left')
        self.line_spacing_var = tk.DoubleVar(
            value=self.settings.get('font_sizing', {}).get('line_spacing', 1.3)
        )
        spacing_scale = tk.Scale(
            spacing_frame,
            from_=1.0,
            to=2.0,
            resolution=0.01,
            orient='horizontal',
            variable=self.line_spacing_var,
            length=200
        )
        spacing_scale.pack(side='left', padx=10)
        tk.Label(spacing_frame, textvariable=self.line_spacing_var, width=5).pack(side='left')
        
        # Maximum lines
        max_lines_frame = tk.Frame(behavior_frame)
        max_lines_frame.pack(fill='x', pady=5)
        
        tk.Label(max_lines_frame, text="Max Lines Per Bubble:", width=20, anchor='w').pack(side='left')
        self.max_lines_var = tk.IntVar(
            value=self.settings.get('font_sizing', {}).get('max_lines', 10)
        )
        lines_spinbox = tb.Spinbox(
            max_lines_frame,
            from_=5,
            to=20,
            textvariable=self.max_lines_var,
            width=10
        )
        lines_spinbox.pack(side='left', padx=10)
        tk.Label(max_lines_frame, text="lines").pack(side='left')
        
        # Quick presets
        preset_frame = tk.LabelFrame(content_frame, text="Quick Presets", padx=15, pady=10)
        preset_frame.pack(fill='x', padx=20, pady=(10, 0))
        
        preset_buttons = tk.Frame(preset_frame)
        preset_buttons.pack(fill='x', pady=5)
        
        tb.Button(
            preset_buttons,
            text="Small Bubbles",
            command=lambda: self._set_font_preset('small'),
            bootstyle="secondary",
            width=15
        ).pack(side='left', padx=5)
        
        tb.Button(
            preset_buttons,
            text="Balanced",
            command=lambda: self._set_font_preset('balanced'),
            bootstyle="secondary",
            width=15
        ).pack(side='left', padx=5)
        
        tb.Button(
            preset_buttons,
            text="Large Text",
            command=lambda: self._set_font_preset('large'),
            bootstyle="secondary",
            width=15
        ).pack(side='left', padx=5)
        
        # Help text
        help_frame = tk.Frame(content_frame)
        help_frame.pack(fill='x', padx=20, pady=(20, 0))
        
        help_text = tk.Label(
            help_frame,
            text="💡 Tips:\n"
                 "• Smart algorithm adapts to different bubble sizes automatically\n"
                 "• Conservative is best for dense text or small bubbles\n"
                 "• Aggressive maximizes readability but may overflow bubbles\n"
                 "• Bubble size factor prevents huge text in large bubbles",
            font=('Arial', 9),
            fg='gray',
            justify='left'
        )
        help_text.pack(anchor='w')

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
    
    def _save_settings(self):
        """Save all settings including expanded iteration controls"""
        # Collect all settings
        self.settings['preprocessing']['enabled'] = self.preprocess_enabled.get()
        self.settings['preprocessing']['auto_detect_quality'] = self.auto_detect.get()
        self.settings['preprocessing']['contrast_threshold'] = self.contrast_threshold.get()
        self.settings['preprocessing']['sharpness_threshold'] = self.sharpness_threshold.get()
        self.settings['preprocessing']['enhancement_strength'] = self.enhancement_strength.get()
        self.settings['preprocessing']['max_image_dimension'] = self.max_dimension.get()
        self.settings['preprocessing']['max_image_pixels'] = self.max_pixels.get()
        self.settings['preprocessing']['chunk_height'] = self.chunk_height.get()
        self.settings['preprocessing']['chunk_overlap'] = self.chunk_overlap.get()
        
        # OCR settings
        self.settings['ocr']['language_hints'] = [code for code, var in self.lang_vars.items() if var.get()]
        self.settings['ocr']['confidence_threshold'] = self.confidence_threshold.get()
        self.settings['ocr']['text_detection_mode'] = self.detection_mode.get()
        self.settings['ocr']['merge_nearby_threshold'] = self.merge_nearby_threshold.get()
        self.settings['ocr']['enable_rotation_correction'] = self.enable_rotation.get()
        self.settings['ocr']['azure_merge_multiplier'] = self.azure_merge_multiplier.get()
        self.settings['ocr']['bubble_detection_enabled'] = self.bubble_detection_enabled.get()
        self.settings['ocr']['bubble_model_path'] = self.bubble_model_path.get()
        self.settings['ocr']['bubble_confidence'] = self.bubble_confidence.get()
        self.settings['ocr']['detector_type'] = self.detector_type.get()
        self.settings['ocr']['rtdetr_confidence'] = self.rtdetr_confidence.get()
        self.settings['ocr']['detect_empty_bubbles'] = self.detect_empty_bubbles.get()
        self.settings['ocr']['detect_text_bubbles'] = self.detect_text_bubbles.get()
        self.settings['ocr']['detect_free_text'] = self.detect_free_text.get()
        self.settings['ocr']['rtdetr_model_url'] = self.rtdetr_model_url.get()
        self.settings['ocr']['azure_reading_order'] = self.azure_reading_order.get()
        self.settings['ocr']['azure_model_version'] = self.azure_model_version.get()
        self.settings['ocr']['azure_max_wait'] = self.azure_max_wait.get()
        self.settings['ocr']['azure_poll_interval'] = self.azure_poll_interval.get()
        self.settings['ocr']['min_text_length'] = self.min_text_length_var.get()
        self.settings['ocr']['exclude_english_text'] = self.exclude_english_var.get()
        
        # Save the detector type as the backend expects - NO AUTO MODE
        if hasattr(self, 'detector_type'):
            detector_display = self.detector_type.get()
            if 'RT-DETR' in detector_display:
                self.settings['ocr']['detector_type'] = 'rtdetr'
            elif 'YOLOv8' in detector_display:
                self.settings['ocr']['detector_type'] = 'yolo'
            elif detector_display == 'Custom Model':
                self.settings['ocr']['detector_type'] = 'custom'
                # Save the custom model path separately
                self.settings['ocr']['custom_model_path'] = self.bubble_model_path.get()
            # NO else clause - if nothing matches, don't change the setting
        
        # Inpainting settings - EXPANDED SECTION
        if hasattr(self, 'inpaint_batch_size'):
            if 'inpainting' not in self.settings:
                self.settings['inpainting'] = {}
            self.settings['inpainting']['batch_size'] = self.inpaint_batch_size.get()
            self.settings['inpainting']['enable_cache'] = self.enable_cache_var.get()
            
            # Save all dilation settings - EXPANDED
            self.settings['mask_dilation'] = self.mask_dilation_var.get()
            
            # Save master control settings
            self.settings['use_all_iterations'] = self.use_all_iterations_var.get()
            self.settings['all_iterations'] = self.all_iterations_var.get()
            
            # Save individual iteration settings
            self.settings['text_bubble_dilation_iterations'] = self.text_bubble_iterations_var.get()
            self.settings['empty_bubble_dilation_iterations'] = self.empty_bubble_iterations_var.get()
            self.settings['free_text_dilation_iterations'] = self.free_text_iterations_var.get()
            
            # Legacy support - map old names to new ones
            self.settings['bubble_dilation_iterations'] = self.text_bubble_iterations_var.get()
            self.settings['dilation_iterations'] = self.text_bubble_iterations_var.get()
        
        # Advanced settings
        self.settings['advanced']['format_detection'] = bool(self.format_detection.get())
        self.settings['advanced']['webtoon_mode'] = self.webtoon_mode.get()
        self.settings['advanced']['debug_mode'] = bool(self.debug_mode.get())
        self.settings['advanced']['save_intermediate'] = bool(self.save_intermediate.get())
        self.settings['advanced']['parallel_processing'] = bool(self.parallel_processing.get())
        self.settings['advanced']['max_workers'] = self.max_workers.get()
        
        # Cloud API settings (only save if the tab was created)
        if hasattr(self, 'cloud_model_var'):
            self.settings['cloud_inpaint_model'] = self.cloud_model_var.get()
            self.settings['cloud_custom_version'] = self.custom_version_var.get()
            self.settings['cloud_inpaint_prompt'] = self.cloud_prompt_var.get()
            self.settings['cloud_negative_prompt'] = self.cloud_negative_prompt_var.get()
            self.settings['cloud_inference_steps'] = self.cloud_steps_var.get()
            self.settings['cloud_timeout'] = self.cloud_timeout_var.get()
            
        # Font sizing settings
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
            
        # Clear bubble detector cache to force reload with new settings
        if hasattr(self.main_gui, 'manga_tab') and hasattr(self.main_gui.manga_tab, 'translator'):
            if hasattr(self.main_gui.manga_tab.translator, 'bubble_detector'):
                self.main_gui.manga_tab.translator.bubble_detector = None
                
        # Save to config
        self.config['manga_settings'] = self.settings
        
        # Save to file
        if hasattr(self.main_gui, 'save_configuration'):
            self.main_gui.save_configuration()
        
        # Call callback if provided
        if self.callback:
            self.callback(self.settings)
        
        # Close dialog with cleanup
        if hasattr(self.dialog, '_cleanup_scrolling'):
            self.dialog._cleanup_scrolling()
        self.dialog.destroy()

    def _cancel(self):
        """Cancel without saving"""
        if hasattr(self.dialog, '_cleanup_scrolling'):
            self.dialog._cleanup_scrolling()
        self.dialog.destroy()

    def _reset_defaults(self):
        """Reset all settings to defaults"""
        if messagebox.askyesno("Reset Settings", "Reset all settings to defaults?"):
            self.settings = self.default_settings.copy()
            # Close and recreate dialog
            if hasattr(self.dialog, '_cleanup_scrolling'):
                self.dialog._cleanup_scrolling()
            self.dialog.destroy()
            self.show_dialog()
