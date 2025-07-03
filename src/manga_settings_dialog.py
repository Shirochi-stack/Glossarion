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
                'enabled': True,
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
                'confidence_threshold': 0.8,
                'merge_nearby_threshold': 50,
                'text_detection_mode': 'document',
                'enable_rotation_correction': True
            },
            'advanced': {
                'format_detection': True,
                'webtoon_mode': 'auto',
                'debug_mode': False,
                'save_intermediate': False,
                'parallel_processing': False,
                'max_workers': 4
            }
        }
        
        # Merge with existing config
        self.settings = self._merge_settings(config.get('manga_settings', {}))
        
        # Show dialog
        self.show_dialog()
    
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
        self.dialog, scrollable_frame, canvas = self.main_gui.wm.setup_scrollable(
            self.parent,
            "Manga Translation Settings",
            width=None,
            height=None,
            max_width_ratio=0.5,
            max_height_ratio=0.9
        )
        
        # Store canvas reference for potential cleanup
        self.canvas = canvas
        
        # Create notebook for tabs
        notebook = ttk.Notebook(scrollable_frame)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create all tabs
        self._create_preprocessing_tab(notebook)
        self._create_ocr_tab(notebook)
        self._create_advanced_tab(notebook)
        
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
            from_=0.1, to=1.0,
            resolution=0.1,
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
            from_=0.1, to=1.0,
            resolution=0.1,
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
            from_=0.1, to=3.0,
            resolution=0.1,
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
            from_=0.1, to=1.0,
            resolution=0.05,
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
        
        # Add tooltips for detection modes
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
        
        self.format_detection = tk.BooleanVar(value=self.settings['advanced']['format_detection'])
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
        
        self.debug_mode = tk.BooleanVar(value=self.settings['advanced']['debug_mode'])
        tb.Checkbutton(
            debug_frame,
            text="Enable debug mode (verbose logging)",
            variable=self.debug_mode,
            bootstyle="round-toggle"
        ).pack(anchor='w')
        
        self.save_intermediate = tk.BooleanVar(value=self.settings['advanced']['save_intermediate'])
        tb.Checkbutton(
            debug_frame,
            text="Save intermediate images (preprocessed, detection overlays)",
            variable=self.save_intermediate,
            bootstyle="round-toggle"
        ).pack(anchor='w', pady=(5, 0))
        
        # Performance settings
        perf_frame = tk.LabelFrame(content_frame, text="Performance", padx=15, pady=10)
        perf_frame.pack(fill='x', padx=20)
        
        self.parallel_processing = tk.BooleanVar(value=self.settings['advanced']['parallel_processing'])
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
        enabled = self.parallel_processing.get()
        self.workers_spinbox.config(state='normal' if enabled else 'disabled')
        self.workers_label.config(fg='white' if enabled else 'gray')
    
    def _save_settings(self):
        """Save all settings"""
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
        
        # Advanced settings
        self.settings['advanced']['format_detection'] = self.format_detection.get()
        self.settings['advanced']['webtoon_mode'] = self.webtoon_mode.get()
        self.settings['advanced']['debug_mode'] = self.debug_mode.get()
        self.settings['advanced']['save_intermediate'] = self.save_intermediate.get()
        self.settings['advanced']['parallel_processing'] = self.parallel_processing.get()
        self.settings['advanced']['max_workers'] = self.max_workers.get()
        
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