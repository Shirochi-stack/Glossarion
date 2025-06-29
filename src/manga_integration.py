# manga_integration.py
"""
Enhanced GUI Integration module for Manga Translation with text visibility controls
Integrates with TranslatorGUI using WindowManager and existing infrastructure
"""

import os
import json
import threading
import time
import traceback
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk
import ttkbootstrap as tb
from typing import List, Dict, Optional, Any
from queue import Queue
from manga_translator import MangaTranslator, GOOGLE_CLOUD_VISION_AVAILABLE

# Try to import UnifiedClient for API initialization
try:
    from unified_api_client import UnifiedClient
except ImportError:
    UnifiedClient = None

class MangaTranslationTab:
    """GUI interface for manga translation integrated with TranslatorGUI"""
    
    def __init__(self, parent_frame: tk.Frame, main_gui, dialog, canvas):
        """Initialize manga translation interface
        
        Args:
            parent_frame: The scrollable frame from WindowManager
            main_gui: Reference to TranslatorGUI instance
            dialog: The dialog window
            canvas: The canvas for scrolling
        """
        self.parent_frame = parent_frame
        self.main_gui = main_gui
        self.dialog = dialog
        self.canvas = canvas
        
        # Translation state
        self.translator = None
        self.is_running = False
        self.stop_flag = threading.Event()
        self.translation_thread = None
        self.selected_files = []
        self.current_file_index = 0
        
        # Progress tracking
        self.total_files = 0
        self.completed_files = 0
        self.failed_files = 0
        
        # Queue for thread-safe GUI updates
        self.update_queue = Queue()
        
        # Flag to prevent saving during initialization
        self._initializing = True
        
        # Load saved text rendering settings from config
        self._load_rendering_settings()
        
        # Build interface
        self._build_interface()
        
        # Now that everything is initialized, allow saving
        self._initializing = False
        
        # Start update loop
        self._process_updates()
    
    def _build_interface(self):
        """Build the enhanced manga translation interface"""
        # Title
        title_frame = tk.Frame(self.parent_frame)
        title_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        title_label = tk.Label(
            title_frame,
            text="🎌 Manga Translation",
            font=('Arial', 16, 'bold')
        )
        title_label.pack(side=tk.LEFT)
        
        # Requirements check
        has_api_key = bool(self.main_gui.api_key_entry.get().strip())
        has_vision = os.path.exists(self.main_gui.config.get('google_vision_credentials', ''))
        
        status_text = "✅ Ready" if (has_api_key and has_vision) else "❌ Setup Required"
        status_color = "green" if (has_api_key and has_vision) else "red"
        
        status_label = tk.Label(
            title_frame,
            text=status_text,
            font=('Arial', 12),
            fg=status_color
        )
        status_label.pack(side=tk.RIGHT)
        
        # Store reference for updates
        self.status_label = status_label
        
        # Add instructions and Google Cloud setup
        if not (has_api_key and has_vision):
            req_frame = tk.Frame(self.parent_frame)
            req_frame.pack(fill=tk.X, padx=20, pady=5)
            
            req_text = []
            if not has_api_key:
                req_text.append("• API Key not configured")
            if not has_vision:
                req_text.append("• Google Cloud Vision credentials not set")
            
            tk.Label(
                req_frame,
                text="\n".join(req_text),
                font=('Arial', 10),
                fg='red',
                justify=tk.LEFT
            ).pack(anchor=tk.W)
            
            # Add Google Cloud credentials selector
            if not has_vision:
                cred_frame = tk.Frame(req_frame)
                cred_frame.pack(fill=tk.X, pady=(10, 0))
                
                tk.Label(
                    cred_frame,
                    text="Set Google Cloud Vision credentials:",
                    font=('Arial', 10)
                ).pack(side=tk.LEFT)
                
                tb.Button(
                    cred_frame,
                    text="Browse JSON File",
                    command=self._browse_google_credentials,
                    bootstyle="primary"
                ).pack(side=tk.LEFT, padx=10)
        
        # File selection frame
        file_frame = tk.LabelFrame(
            self.parent_frame,
            text="Select Manga Images",
            font=('Arial', 12, 'bold'),
            padx=15,
            pady=10
        )
        file_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # File listbox with scrollbar
        list_frame = tk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            height=8,
            selectmode=tk.EXTENDED
        )
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.file_listbox.yview)
        
        # File buttons
        file_btn_frame = tk.Frame(file_frame)
        file_btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        tb.Button(
            file_btn_frame,
            text="Add Files",
            command=self._add_files,
            bootstyle="primary"
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        tb.Button(
            file_btn_frame,
            text="Add Folder",
            command=self._add_folder,
            bootstyle="primary"
        ).pack(side=tk.LEFT, padx=5)
        
        tb.Button(
            file_btn_frame,
            text="Remove Selected",
            command=self._remove_selected,
            bootstyle="danger"
        ).pack(side=tk.LEFT, padx=5)
        
        tb.Button(
            file_btn_frame,
            text="Clear All",
            command=self._clear_all,
            bootstyle="warning"
        ).pack(side=tk.LEFT, padx=5)
        
        # Settings frame
        settings_frame = tk.LabelFrame(
            self.parent_frame,
            text="Translation Settings",
            font=('Arial', 12, 'bold'),
            padx=15,
            pady=10
        )
        settings_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # API Settings - Hybrid approach
        api_frame = tk.Frame(settings_frame)
        api_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(api_frame, text="OCR: Google Cloud Vision | Translation: API Key", 
                font=('Arial', 10, 'italic'), fg='gray').pack(side=tk.LEFT)
        
        # Show current model
        current_model = 'Unknown'
        if hasattr(self.main_gui, 'model_var'):
            current_model = self.main_gui.model_var.get()
        elif hasattr(self.main_gui, 'model_combo'):
            current_model = self.main_gui.model_combo.get()
        elif hasattr(self.main_gui, 'config'):
            current_model = self.main_gui.config.get('model', 'Unknown')
        
        tk.Label(api_frame, text=f"Model: {current_model}", 
                font=('Arial', 10, 'italic'), fg='gray').pack(side=tk.RIGHT)
        
        # Google Cloud Credentials section
        creds_frame = tk.Frame(settings_frame)
        creds_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(creds_frame, text="Google Cloud Credentials:", width=20, anchor='w').pack(side=tk.LEFT)
        
        # Show current credentials file
        google_creds_path = self.main_gui.config.get('google_vision_credentials', '') or self.main_gui.config.get('google_cloud_credentials', '')
        creds_display = os.path.basename(google_creds_path) if google_creds_path else "Not Set"
        
        self.creds_label = tk.Label(creds_frame, text=creds_display, 
                                   font=('Arial', 9), fg='green' if google_creds_path else 'red')
        self.creds_label.pack(side=tk.LEFT, padx=10)
        
        tb.Button(
            creds_frame,
            text="Browse",
            command=self._browse_google_credentials_permanent,
            bootstyle="primary"
        ).pack(side=tk.LEFT)
        
        # Text Rendering Settings Frame
        render_frame = tk.LabelFrame(
            self.parent_frame,
            text="Text Visibility Settings",
            font=('Arial', 12, 'bold'),
            padx=15,
            pady=10
        )
        render_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Background opacity slider
        opacity_frame = tk.Frame(render_frame)
        opacity_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(opacity_frame, text="Background Opacity:", width=20, anchor='w').pack(side=tk.LEFT)
        
        opacity_scale = tk.Scale(
            opacity_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self.bg_opacity_var,
            command=self._update_opacity_label,
            length=200
        )
        opacity_scale.pack(side=tk.LEFT, padx=10)
        
        self.opacity_label = tk.Label(opacity_frame, text="100%", width=5)
        self.opacity_label.pack(side=tk.LEFT)
        
        # Initialize the label with the loaded value
        self._update_opacity_label(self.bg_opacity_var.get())
 
        # Skip inpainting toggle
        tb.Checkbutton(render_frame, text="Skip Inpainting (Preserve Original Art)", 
                      variable=self.skip_inpainting_var,
                      bootstyle="round-toggle").pack(anchor='w', pady=5)
        
        tk.Label(render_frame, text="Keep original manga art under translated text",
                font=('TkDefaultFont', 9), fg='gray').pack(anchor='w', padx=20, pady=(0, 10))
                
        # Background style selection
        style_frame = tk.Frame(render_frame)
        style_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(style_frame, text="Background Style:", width=20, anchor='w').pack(side=tk.LEFT)
        
        styles = [('box', 'Box'), ('circle', 'Circle'), ('wrap', 'Wrap Text')]
        for value, text in styles:
            tb.Radiobutton(
                style_frame,
                text=text,
                variable=self.bg_style_var,
                value=value,
                bootstyle="primary",
                command=self._save_rendering_settings
            ).pack(side=tk.LEFT, padx=10)
        
        # Background size reduction
        reduction_frame = tk.Frame(render_frame)
        reduction_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(reduction_frame, text="Background Size:", width=20, anchor='w').pack(side=tk.LEFT)
        
        reduction_scale = tk.Scale(
            reduction_frame,
            from_=0.5,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.bg_reduction_var,
            command=self._update_reduction_label,
            length=200
        )
        reduction_scale.pack(side=tk.LEFT, padx=10)
        
        self.reduction_label = tk.Label(reduction_frame, text="100%", width=5)
        self.reduction_label.pack(side=tk.LEFT)
        
        # Initialize the label with the loaded value
        self._update_reduction_label(self.bg_reduction_var.get())
        
        # Font settings
        font_frame = tk.Frame(render_frame)
        font_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(font_frame, text="Font Size:", width=20, anchor='w').pack(side=tk.LEFT)
        
        font_size_spinbox = tb.Spinbox(
            font_frame,
            from_=0,
            to=72,
            textvariable=self.font_size_var,
            width=10,
            command=self._save_rendering_settings
        )
        font_size_spinbox.pack(side=tk.LEFT, padx=10)
        # Also bind to save on manual entry
        font_size_spinbox.bind('<Return>', lambda e: self._save_rendering_settings())
        font_size_spinbox.bind('<FocusOut>', lambda e: self._save_rendering_settings())
        
        tk.Label(font_frame, text="(0 = Auto)", font=('Arial', 9), fg='gray').pack(side=tk.LEFT)
        
        # Font style selection
        font_style_frame = tk.Frame(render_frame)
        font_style_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(font_style_frame, text="Font Style:", width=20, anchor='w').pack(side=tk.LEFT)
        
        # Font style will be set from loaded config in _load_rendering_settings
        self.font_combo = ttk.Combobox(
            font_style_frame,
            textvariable=self.font_style_var,
            values=self._get_available_fonts(),
            width=30,
            state='readonly'
        )
        self.font_combo.pack(side=tk.LEFT, padx=10)
        self.font_combo.bind('<<ComboboxSelected>>', self._on_font_selected)
        
        # Font color selection
        color_frame = tk.Frame(render_frame)
        color_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(color_frame, text="Font Color:", width=20, anchor='w').pack(side=tk.LEFT)
        
        # RGB sliders in a sub-frame
        rgb_frame = tk.Frame(color_frame)
        rgb_frame.pack(side=tk.LEFT, padx=10)
        
        # Red
        r_frame = tk.Frame(rgb_frame)
        r_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(r_frame, text="R:", width=2).pack(side=tk.LEFT)
        tk.Scale(r_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.text_color_r,
                length=80, command=self._update_color_preview).pack(side=tk.LEFT)
        
        # Green
        g_frame = tk.Frame(rgb_frame)
        g_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(g_frame, text="G:", width=2).pack(side=tk.LEFT)
        tk.Scale(g_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.text_color_g,
                length=80, command=self._update_color_preview).pack(side=tk.LEFT)
        
        # Blue
        b_frame = tk.Frame(rgb_frame)
        b_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(b_frame, text="B:", width=2).pack(side=tk.LEFT)
        tk.Scale(b_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.text_color_b,
                length=80, command=self._update_color_preview).pack(side=tk.LEFT)
        
        # Color preview
        self.color_preview = tk.Canvas(color_frame, width=40, height=30, 
                                     highlightthickness=1, highlightbackground="gray")
        self.color_preview.pack(side=tk.LEFT, padx=10)
        self._update_color_preview(None)  # Initialize with loaded colors
        
        # Shadow settings frame
        shadow_frame = tk.LabelFrame(render_frame, text="Text Shadow", padx=10, pady=5)
        shadow_frame.pack(fill=tk.X, pady=10)
        
        # Shadow enabled checkbox
        tb.Checkbutton(
            shadow_frame,
            text="Enable Shadow",
            variable=self.shadow_enabled_var,
            bootstyle="round-toggle",
            command=self._toggle_shadow_controls
        ).pack(anchor='w')
        
        # Shadow controls container
        self.shadow_controls = tk.Frame(shadow_frame)
        self.shadow_controls.pack(fill=tk.X, pady=5)
        
        # Shadow color
        shadow_color_frame = tk.Frame(self.shadow_controls)
        shadow_color_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(shadow_color_frame, text="Shadow Color:", width=15, anchor='w').pack(side=tk.LEFT)
        
        # Shadow RGB sliders
        shadow_rgb_frame = tk.Frame(shadow_color_frame)
        shadow_rgb_frame.pack(side=tk.LEFT, padx=10)
        
        # Shadow Red
        sr_frame = tk.Frame(shadow_rgb_frame)
        sr_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(sr_frame, text="R:", width=2).pack(side=tk.LEFT)
        tk.Scale(sr_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.shadow_color_r,
                length=60, command=self._update_shadow_preview).pack(side=tk.LEFT)
        
        # Shadow Green
        sg_frame = tk.Frame(shadow_rgb_frame)
        sg_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(sg_frame, text="G:", width=2).pack(side=tk.LEFT)
        tk.Scale(sg_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.shadow_color_g,
                length=60, command=self._update_shadow_preview).pack(side=tk.LEFT)
        
        # Shadow Blue
        sb_frame = tk.Frame(shadow_rgb_frame)
        sb_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(sb_frame, text="B:", width=2).pack(side=tk.LEFT)
        tk.Scale(sb_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.shadow_color_b,
                length=60, command=self._update_shadow_preview).pack(side=tk.LEFT)
        
        # Shadow color preview
        self.shadow_preview = tk.Canvas(shadow_color_frame, width=30, height=25, 
                                      highlightthickness=1, highlightbackground="gray")
        self.shadow_preview.pack(side=tk.LEFT, padx=10)
        self._update_shadow_preview(None)  # Initialize with loaded colors
        
        # Shadow offset
        offset_frame = tk.Frame(self.shadow_controls)
        offset_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(offset_frame, text="Shadow Offset:", width=15, anchor='w').pack(side=tk.LEFT)
        
        # X offset
        x_frame = tk.Frame(offset_frame)
        x_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(x_frame, text="X:", width=2).pack(side=tk.LEFT)
        x_spinbox = tb.Spinbox(x_frame, from_=-10, to=10, textvariable=self.shadow_offset_x_var,
                  width=5, command=self._save_rendering_settings)
        x_spinbox.pack(side=tk.LEFT)
        x_spinbox.bind('<Return>', lambda e: self._save_rendering_settings())
        x_spinbox.bind('<FocusOut>', lambda e: self._save_rendering_settings())
        
        # Y offset
        y_frame = tk.Frame(offset_frame)
        y_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(y_frame, text="Y:", width=2).pack(side=tk.LEFT)
        y_spinbox = tb.Spinbox(y_frame, from_=-10, to=10, textvariable=self.shadow_offset_y_var,
                  width=5, command=self._save_rendering_settings)
        y_spinbox.pack(side=tk.LEFT)
        y_spinbox.bind('<Return>', lambda e: self._save_rendering_settings())
        y_spinbox.bind('<FocusOut>', lambda e: self._save_rendering_settings())
        
        # Shadow blur
        blur_frame = tk.Frame(self.shadow_controls)
        blur_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(blur_frame, text="Shadow Blur:", width=15, anchor='w').pack(side=tk.LEFT)
        tk.Scale(blur_frame, from_=0, to=10, orient=tk.HORIZONTAL, variable=self.shadow_blur_var,
                length=150, command=lambda v: self._save_rendering_settings()).pack(side=tk.LEFT, padx=10)
        tk.Label(blur_frame, text="(0=sharp, 10=blurry)", font=('Arial', 9), fg='gray').pack(side=tk.LEFT)
        
        # Initially disable shadow controls
        self._toggle_shadow_controls()
        
        # Output settings
        output_frame = tk.Frame(settings_frame)
        output_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.create_subfolder_var = tk.BooleanVar(value=self.main_gui.config.get('manga_create_subfolder', True))
        tb.Checkbutton(
            output_frame,
            text="Create 'translated' subfolder for output",
            variable=self.create_subfolder_var,
            bootstyle="round-toggle",
            command=self._save_rendering_settings
        ).pack(side=tk.LEFT)
        
        # Control buttons
        control_frame = tk.Frame(self.parent_frame)
        control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Check if ready
        has_api_key = bool(self.main_gui.api_key_entry.get().strip()) if hasattr(self.main_gui, 'api_key_entry') else False
        has_vision = os.path.exists(self.main_gui.config.get('google_vision_credentials', ''))
        is_ready = has_api_key and has_vision
        
        self.start_button = tb.Button(
            control_frame,
            text="Start Translation",
            command=self._start_translation,
            bootstyle="success",
            state=tk.NORMAL if is_ready else tk.DISABLED
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add tooltip to show why button is disabled
        if not is_ready:
            reasons = []
            if not has_api_key:
                reasons.append("API key not configured")
            if not has_vision:
                reasons.append("Google Vision credentials not set")
            tooltip_text = "Cannot start: " + ", ".join(reasons)
            # You can add a tooltip here if you have a tooltip library
        
        self.stop_button = tb.Button(
            control_frame,
            text="Stop",
            command=self._stop_translation,
            bootstyle="danger",
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT)
        
        # Progress frame
        progress_frame = tk.LabelFrame(
            self.parent_frame,
            text="Progress",
            font=('Arial', 12, 'bold'),
            padx=15,
            pady=10
        )
        progress_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Overall progress
        self.progress_label = tk.Label(
            progress_frame,
            text="Ready to start",
            font=('Arial', 11)
        )
        self.progress_label.pack(anchor=tk.W)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=400
        )
        self.progress_bar.pack(fill=tk.X, pady=(5, 10))
        
        # Current file status
        self.current_file_label = tk.Label(
            progress_frame,
            text="",
            font=('Arial', 10),
            fg='gray'
        )
        self.current_file_label.pack(anchor=tk.W)
        
        # Log frame
        log_frame = tk.LabelFrame(
            self.parent_frame,
            text="Translation Log",
            font=('Arial', 12, 'bold'),
            padx=15,
            pady=10
        )
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
        
        # Log text with scrollbar
        log_scroll_frame = tk.Frame(log_frame)
        log_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        log_scrollbar = tk.Scrollbar(log_scroll_frame)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(
            log_scroll_frame,
            height=15,
            wrap=tk.WORD,
            yscrollcommand=log_scrollbar.set
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.config(command=self.log_text.yview)
        
        # Configure text tags for colored output
        self.log_text.tag_config('info', foreground='white')
        self.log_text.tag_config('success', foreground='green')
        self.log_text.tag_config('warning', foreground='orange')
        self.log_text.tag_config('error', foreground='red')
    
    def _update_color_preview(self, event):
        """Update the font color preview"""
        r = self.text_color_r.get()
        g = self.text_color_g.get()
        b = self.text_color_b.get()
        color = f'#{r:02x}{g:02x}{b:02x}'
        self.color_preview.configure(bg=color)
        # Auto-save on change
        if event is not None:  # Only save on user interaction, not initial load
            self._save_rendering_settings()
    
    def _update_shadow_preview(self, event):
        """Update the shadow color preview"""
        r = self.shadow_color_r.get()
        g = self.shadow_color_g.get()
        b = self.shadow_color_b.get()
        color = f'#{r:02x}{g:02x}{b:02x}'
        self.shadow_preview.configure(bg=color)
        # Auto-save on change
        if event is not None:  # Only save on user interaction, not initial load
            self._save_rendering_settings()
    
    def _toggle_shadow_controls(self):
        """Enable/disable shadow controls based on checkbox"""
        if self.shadow_enabled_var.get():
            for widget in self.shadow_controls.winfo_children():
                self._enable_widget_tree(widget)
        else:
            for widget in self.shadow_controls.winfo_children():
                self._disable_widget_tree(widget)
        # Auto-save on change (but not during initialization)
        if not getattr(self, '_initializing', False):
            self._save_rendering_settings()
    
    def _enable_widget_tree(self, widget):
        """Recursively enable a widget and its children"""
        try:
            widget.configure(state=tk.NORMAL)
        except:
            pass
        for child in widget.winfo_children():
            self._enable_widget_tree(child)
    
    def _disable_widget_tree(self, widget):
        """Recursively disable a widget and its children"""
        try:
            widget.configure(state=tk.DISABLED)
        except:
            pass
        for child in widget.winfo_children():
            self._disable_widget_tree(child)
    
    def _load_rendering_settings(self):
        """Load text rendering settings from config"""
        config = self.main_gui.config
        
        # Initialize with defaults
        self.bg_opacity_var = tk.IntVar(value=config.get('manga_bg_opacity', 255))
        self.bg_style_var = tk.StringVar(value=config.get('manga_bg_style', 'box'))
        self.bg_reduction_var = tk.DoubleVar(value=config.get('manga_bg_reduction', 1.0))
        self.font_size_var = tk.IntVar(value=config.get('manga_font_size', 0))
        self.selected_font_path = config.get('manga_font_path', None)
        self.skip_inpainting_var = tk.BooleanVar(value=config.get('manga_skip_inpainting', False))
        
        # Font color settings
        manga_text_color = config.get('manga_text_color', [0, 0, 0])
        self.text_color_r = tk.IntVar(value=manga_text_color[0])
        self.text_color_g = tk.IntVar(value=manga_text_color[1])
        self.text_color_b = tk.IntVar(value=manga_text_color[2])
        
        # Shadow settings
        self.shadow_enabled_var = tk.BooleanVar(value=config.get('manga_shadow_enabled', False))
        manga_shadow_color = config.get('manga_shadow_color', [128, 128, 128])
        self.shadow_color_r = tk.IntVar(value=manga_shadow_color[0])
        self.shadow_color_g = tk.IntVar(value=manga_shadow_color[1])
        self.shadow_color_b = tk.IntVar(value=manga_shadow_color[2])
        self.shadow_offset_x_var = tk.IntVar(value=config.get('manga_shadow_offset_x', 2))
        self.shadow_offset_y_var = tk.IntVar(value=config.get('manga_shadow_offset_y', 2))
        self.shadow_blur_var = tk.IntVar(value=config.get('manga_shadow_blur', 0))
        
        # Also set font style var if we have a saved font
        if self.selected_font_path:
            # Reverse lookup the font name from path
            font_map = {
                "C:/Windows/Fonts/arial.ttf": "Arial",
                "C:/Windows/Fonts/calibri.ttf": "Calibri",
                "C:/Windows/Fonts/comic.ttf": "Comic Sans MS",
                "C:/Windows/Fonts/tahoma.ttf": "Tahoma",
                "C:/Windows/Fonts/times.ttf": "Times New Roman",
                "C:/Windows/Fonts/verdana.ttf": "Verdana",
                "C:/Windows/Fonts/georgia.ttf": "Georgia",
                "C:/Windows/Fonts/impact.ttf": "Impact",
                "C:/Windows/Fonts/trebuc.ttf": "Trebuchet MS",
                "C:/Windows/Fonts/cour.ttf": "Courier New"
            }
            font_name = next((name for path, name in font_map.items() if path == self.selected_font_path), "Default")
        else:
            font_name = "Default"
        
        self.font_style_var = tk.StringVar(value=font_name)
    
    def _save_rendering_settings(self):
        """Save text rendering settings to config"""
        # Don't save during initialization
        if hasattr(self, '_initializing') and self._initializing:
            return
            
        # Update Manga GUI config
        self.main_gui.config['manga_bg_opacity'] = self.bg_opacity_var.get()
        self.main_gui.config['manga_bg_style'] = self.bg_style_var.get()
        self.main_gui.config['manga_bg_reduction'] = self.bg_reduction_var.get()
        self.main_gui.config['manga_font_size'] = self.font_size_var.get()
        self.main_gui.config['manga_font_path'] = self.selected_font_path
        self.main_gui.config['manga_skip_inpainting'] = self.skip_inpainting_var.get()

        
        # Save font color as list
        self.main_gui.config['manga_text_color'] = [
            self.text_color_r.get(),
            self.text_color_g.get(),
            self.text_color_b.get()
        ]
        
        # Save shadow settings
        self.main_gui.config['manga_shadow_enabled'] = self.shadow_enabled_var.get()
        self.main_gui.config['manga_shadow_color'] = [
            self.shadow_color_r.get(),
            self.shadow_color_g.get(),
            self.shadow_color_b.get()
        ]
        self.main_gui.config['manga_shadow_offset_x'] = self.shadow_offset_x_var.get()
        self.main_gui.config['manga_shadow_offset_y'] = self.shadow_offset_y_var.get()
        self.main_gui.config['manga_shadow_blur'] = self.shadow_blur_var.get()
        
        # Save existing create_subfolder setting if it exists
        if hasattr(self, 'create_subfolder_var'):
            self.main_gui.config['manga_create_subfolder'] = self.create_subfolder_var.get()
        
        # Call main GUI's save_configuration to persist to file
        if hasattr(self.main_gui, 'save_configuration'):
            self.main_gui.save_configuration()
    
    def _browse_google_credentials_permanent(self):
        """Browse and set Google Cloud Vision credentials from the permanent button"""
        file_path = filedialog.askopenfilename(
            title="Select Google Cloud Service Account JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            # Save to config with both keys for compatibility
            self.main_gui.config['google_vision_credentials'] = file_path
            self.main_gui.config['google_cloud_credentials'] = file_path
            
            # Save configuration
            if hasattr(self.main_gui, 'save_configuration'):
                self.main_gui.save_configuration()
            
            # Update button state immediately
            self.start_button.config(state=tk.NORMAL)
            
            # Update credentials display
            self.creds_label.config(text=os.path.basename(file_path), fg='green')
            
            # Update status if we have a reference
            if hasattr(self, 'status_label'):
                self.status_label.config(text="✅ Ready", fg="green")
            
            messagebox.showinfo("Success", "Google Cloud credentials set successfully!")
    
    def _browse_google_credentials(self):
        """Browse and set Google Cloud Vision credentials"""
        file_path = filedialog.askopenfilename(
            title="Select Google Cloud Service Account JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            # Save to config with both keys for compatibility
            self.main_gui.config['google_vision_credentials'] = file_path
            self.main_gui.config['google_cloud_credentials'] = file_path
            
            # Save configuration
            if hasattr(self.main_gui, 'save_configuration'):
                self.main_gui.save_configuration()
            
            # Update button state immediately
            self.start_button.config(state=tk.NORMAL)
            
            # Update status display
            self._update_status_display()
            
            messagebox.showinfo("Success", "Google Cloud credentials set successfully!")
    
    def _update_status_display(self):
        """Update the status display after credentials change"""
        # This would update the status label if we had a reference to it
        # For now, we'll just ensure the button is enabled
        google_creds_path = self.main_gui.config.get('google_vision_credentials', '') or self.main_gui.config.get('google_cloud_credentials', '')
        has_vision = os.path.exists(google_creds_path) if google_creds_path else False
        
        if has_vision:
            self.start_button.config(state=tk.NORMAL)
    
    def _get_available_fonts(self):
        """Get list of available fonts"""
        fonts = ["Default"]
        
        # Windows fonts including CJK support
        windows_fonts = [
            ("Arial", "C:/Windows/Fonts/arial.ttf"),
            ("Calibri", "C:/Windows/Fonts/calibri.ttf"),
            ("Comic Sans MS", "C:/Windows/Fonts/comic.ttf"),
            ("MS Gothic", "C:/Windows/Fonts/msgothic.ttc"),      # Japanese
            ("MS Mincho", "C:/Windows/Fonts/msmincho.ttc"),      # Japanese
            ("Meiryo", "C:/Windows/Fonts/meiryo.ttc"),           # Japanese
            ("Yu Gothic", "C:/Windows/Fonts/yugothic.ttc"),      # Japanese
            ("Malgun Gothic", "C:/Windows/Fonts/malgun.ttf"),    # Korean
            ("SimSun", "C:/Windows/Fonts/simsun.ttc"),           # Chinese
            ("Microsoft YaHei", "C:/Windows/Fonts/msyh.ttc"),    # Chinese
            ("Tahoma", "C:/Windows/Fonts/tahoma.ttf"),
            ("Times New Roman", "C:/Windows/Fonts/times.ttf"),
            ("Verdana", "C:/Windows/Fonts/verdana.ttf"),
            ("Georgia", "C:/Windows/Fonts/georgia.ttf"),
            ("Impact", "C:/Windows/Fonts/impact.ttf"),
            ("Trebuchet MS", "C:/Windows/Fonts/trebuc.ttf"),
            ("Courier New", "C:/Windows/Fonts/cour.ttf")
        ]
        
        for name, path in windows_fonts:
            if os.path.exists(path):
                fonts.append(name)
        
        return fonts
    
    def _on_font_selected(self, event):
        """Handle font selection"""
        selected = self.font_style_var.get()
        
        if selected == "Default":
            self.selected_font_path = None
        else:
            # Map font names to paths
            font_map = {
                "Arial": "C:/Windows/Fonts/arial.ttf",
                "Calibri": "C:/Windows/Fonts/calibri.ttf",
                "Comic Sans MS": "C:/Windows/Fonts/comic.ttf",
                "MS Gothic": "C:/Windows/Fonts/msgothic.ttc",
                "MS Mincho": "C:/Windows/Fonts/msmincho.ttc",
                "Meiryo": "C:/Windows/Fonts/meiryo.ttc",
                "Yu Gothic": "C:/Windows/Fonts/yugothic.ttc",
                "Malgun Gothic": "C:/Windows/Fonts/malgun.ttf",
                "SimSun": "C:/Windows/Fonts/simsun.ttc",
                "Microsoft YaHei": "C:/Windows/Fonts/msyh.ttc",
                "Tahoma": "C:/Windows/Fonts/tahoma.ttf",
                "Times New Roman": "C:/Windows/Fonts/times.ttf",
                "Verdana": "C:/Windows/Fonts/verdana.ttf",
                "Georgia": "C:/Windows/Fonts/georgia.ttf",
                "Impact": "C:/Windows/Fonts/impact.ttf",
                "Trebuchet MS": "C:/Windows/Fonts/trebuc.ttf",
                "Courier New": "C:/Windows/Fonts/cour.ttf"
            }
            
            self.selected_font_path = font_map.get(selected, None)
        
        # Auto-save on change
        self._save_rendering_settings()
    
    def _update_opacity_label(self, value):
        """Update opacity percentage label"""
        percentage = int((float(value) / 255) * 100)
        self.opacity_label.config(text=f"{percentage}%")
        # Auto-save on change
        self._save_rendering_settings()
    
    def _update_reduction_label(self, value):
        """Update size reduction percentage label"""
        percentage = int(float(value) * 100)
        self.reduction_label.config(text=f"{percentage}%")
        # Auto-save on change
        self._save_rendering_settings()
    
    def _add_files(self):
        """Add image files to the list"""
        files = filedialog.askopenfilenames(
            title="Select Manga Images",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )
        
        for file in files:
            if file not in self.selected_files:
                self.selected_files.append(file)
                self.file_listbox.insert(tk.END, os.path.basename(file))
    
    def _add_folder(self):
        """Add all images from a folder"""
        folder = filedialog.askdirectory(title="Select Folder with Manga Images")
        if not folder:
            return
        
        # Find all image files in folder
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        
        for filename in sorted(os.listdir(folder)):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                filepath = os.path.join(folder, filename)
                if filepath not in self.selected_files:
                    self.selected_files.append(filepath)
                    self.file_listbox.insert(tk.END, filename)
    
    def _remove_selected(self):
        """Remove selected files from the list"""
        selected_indices = list(self.file_listbox.curselection())
        
        # Remove in reverse order to maintain indices
        for index in reversed(selected_indices):
            self.file_listbox.delete(index)
            del self.selected_files[index]
    
    def _clear_all(self):
        """Clear all files from the list"""
        self.file_listbox.delete(0, tk.END)
        self.selected_files.clear()
    
    def _log(self, message: str, level: str = 'info'):
        """Thread-safe logging to GUI"""
        self.update_queue.put(('log', message, level))
    
    def _update_progress(self, current: int, total: int, status: str):
        """Thread-safe progress update"""
        self.update_queue.put(('progress', current, total, status))
    
    def _update_current_file(self, filename: str):
        """Thread-safe current file update"""
        self.update_queue.put(('current_file', filename))
    
    def _process_updates(self):
        """Process queued GUI updates"""
        try:
            while True:
                update = self.update_queue.get_nowait()
                
                if update[0] == 'log':
                    _, message, level = update
                    self.log_text.insert(tk.END, message + '\n', level)
                    self.log_text.see(tk.END)
                    
                elif update[0] == 'progress':
                    _, current, total, status = update
                    if total > 0:
                        percentage = (current / total) * 100
                        self.progress_bar['value'] = percentage
                    self.progress_label.config(text=status)
                    
                elif update[0] == 'current_file':
                    _, filename = update
                    self.current_file_label.config(text=f"Current: {filename}")
                    
        except:
            pass
        
        # Schedule next update
        self.parent_frame.after(100, self._process_updates)
    
    def _start_translation(self):
        """Start the translation process"""
        if not self.selected_files:
            messagebox.showwarning("No Files", "Please select manga images to translate.")
            return
        
        # Check both possible config keys for backward compatibility
        google_creds = self.main_gui.config.get('google_vision_credentials', '') or self.main_gui.config.get('google_cloud_credentials', '')
        
        if not google_creds or not os.path.exists(google_creds):
            messagebox.showerror("Error", "Google Cloud Vision credentials not found.\nPlease set up credentials in the main settings.")
            return
        
        # Initialize API client if needed
        if not hasattr(self.main_gui, 'client') or not self.main_gui.client:
            # Try to create the client from saved config
            api_key = None
            model = 'gemini-1.5-flash'  # default
            
            # Try to get API key from various sources
            if hasattr(self.main_gui, 'api_key_entry') and self.main_gui.api_key_entry.get().strip():
                api_key = self.main_gui.api_key_entry.get().strip()
            elif hasattr(self.main_gui, 'config') and self.main_gui.config.get('api_key'):
                api_key = self.main_gui.config.get('api_key')
            
            # Try to get model
            if hasattr(self.main_gui, 'model_var'):
                model = self.main_gui.model_var.get()
            elif hasattr(self.main_gui, 'config') and self.main_gui.config.get('model'):
                model = self.main_gui.config.get('model')
            
            if not api_key:
                messagebox.showerror("Error", "API key not found.\nPlease configure your API key in the main settings.")
                return
            
            # Create the unified client
            try:
                from unified_api_client import UnifiedClient
                self.main_gui.client = UnifiedClient(model=model, api_key=api_key)
                self._log(f"Created API client with model: {model}", "info")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create API client:\n{str(e)}")
                return
        
        # Initialize translator if needed
        if not self.translator:
            try:
                self.translator = MangaTranslator(
                    google_creds,
                    self.main_gui.client,
                    self.main_gui,
                    log_callback=self._log
                )
                
                # Apply text rendering settings
                self._apply_rendering_settings()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to initialize translator:\n{str(e)}")
                self._log(f"Initialization error: {str(e)}", "error")
                self._log(traceback.format_exc(), "error")
                return
        else:
            # Update rendering settings
            self._apply_rendering_settings()
        
        # Clear log
        self.log_text.delete('1.0', tk.END)
        
        # Reset progress
        self.total_files = len(self.selected_files)
        self.completed_files = 0
        self.failed_files = 0
        self.current_file_index = 0
        
        # Update UI state
        self.is_running = True
        self.stop_flag.clear()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Disable file modification during translation
        for widget in [self.file_listbox]:
            widget.config(state=tk.DISABLED)
        
        # Log start message
        self._log(f"Starting translation of {self.total_files} files...", "info")
        self._log(f"Using Google Vision credentials: {os.path.basename(google_creds)}", "info")
        self._log(f"Using API model: {self.main_gui.client.model if hasattr(self.main_gui.client, 'model') else 'unknown'}", "info")
        
        # Start translation thread
        self.translation_thread = threading.Thread(
            target=self._translation_worker,
            daemon=True
        )
        self.translation_thread.start()
    
    def _apply_rendering_settings(self):
        """Apply current rendering settings to translator"""
        if self.translator:
            # Get text color as tuple
            text_color = (
                self.text_color_r.get(),
                self.text_color_g.get(),
                self.text_color_b.get()
            )
            
            # Get shadow color as tuple
            shadow_color = (
                self.shadow_color_r.get(),
                self.shadow_color_g.get(),
                self.shadow_color_b.get()
            )
            
            self.translator.update_text_rendering_settings(
                bg_opacity=self.bg_opacity_var.get(),
                bg_style=self.bg_style_var.get(),
                bg_reduction=self.bg_reduction_var.get(),
                font_style=self.selected_font_path,
                font_size=self.font_size_var.get() if self.font_size_var.get() > 0 else None,
                text_color=text_color,
                shadow_enabled=self.shadow_enabled_var.get(),
                shadow_color=shadow_color,
                shadow_offset_x=self.shadow_offset_x_var.get(),
                shadow_offset_y=self.shadow_offset_y_var.get(),
                shadow_blur=self.shadow_blur_var.get()
            )
            if hasattr(self, 'skip_inpainting_var'):
                self.translator.skip_inpainting = self.skip_inpainting_var.get()
            
            self._log(f"Applied rendering settings:", "info")
            self._log(f"  Background: {self.bg_style_var.get()} @ {int(self.bg_opacity_var.get()/255*100)}% opacity", "info")
            self._log(f"  Font: {os.path.basename(self.selected_font_path) if self.selected_font_path else 'Default'}", "info")
            self._log(f"  Text Color: RGB({text_color[0]}, {text_color[1]}, {text_color[2]})", "info")
            self._log(f"  Shadow: {'Enabled' if self.shadow_enabled_var.get() else 'Disabled'}", "info")
    
    def _translation_worker(self):
        """Worker thread for translation"""
        try:
            for index, filepath in enumerate(self.selected_files):
                if self.stop_flag.is_set():
                    self._log("\n⏹️ Translation stopped by user", "warning")
                    break
                
                self.current_file_index = index
                filename = os.path.basename(filepath)
                
                self._update_current_file(filename)
                self._update_progress(
                    index,
                    self.total_files,
                    f"Processing {index + 1}/{self.total_files}: {filename}"
                )
                
                # Determine output path
                if self.create_subfolder_var.get():
                    output_dir = os.path.join(os.path.dirname(filepath), 'translated')
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, filename)
                else:
                    base, ext = os.path.splitext(filepath)
                    output_path = f"{base}_translated{ext}"
                
                # Process the image
                result = self.translator.process_image(filepath, output_path)
                
                if result['success']:
                    self.completed_files += 1
                    self._log(f"✅ Successfully translated: {filename}", "success")
                else:
                    self.failed_files += 1
                    errors = '\n'.join(result['errors'])
                    self._log(f"❌ Failed to translate {filename}:\n{errors}", "error")
            
            # Final summary
            self._log(f"\n{'='*60}", "info")
            self._log(f"📊 Translation Summary:", "info")
            self._log(f"   Total files: {self.total_files}", "info")
            self._log(f"   ✅ Successful: {self.completed_files}", "success")
            self._log(f"   ❌ Failed: {self.failed_files}", "error" if self.failed_files > 0 else "info")
            self._log(f"{'='*60}\n", "info")
            
            self._update_progress(
                self.total_files,
                self.total_files,
                f"Complete! {self.completed_files} successful, {self.failed_files} failed"
            )
            
        except Exception as e:
            self._log(f"\n❌ Translation error: {str(e)}", "error")
            self._log(traceback.format_exc(), "error")
        
        finally:
            # Reset UI state
            self.parent_frame.after(0, self._reset_ui_state)
    
    def _stop_translation(self):
        """Stop the translation process"""
        if self.is_running:
            self.stop_flag.set()
            self.stop_button.config(state=tk.DISABLED)
            self._log("\n⏸️ Stopping translation...", "warning")
    
    def _reset_ui_state(self):
        """Reset UI to ready state"""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        # Re-enable file modification
        for widget in [self.file_listbox]:
            widget.config(state=tk.NORMAL)
