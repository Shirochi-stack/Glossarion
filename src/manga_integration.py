# manga_integration.py
"""
Enhanced GUI Integration module for Manga Translation with text visibility controls
Integrates with TranslatorGUI using WindowManager and existing infrastructure
Now includes full page context mode with customizable prompt
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
from manga_settings_dialog import MangaSettingsDialog


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
        self.font_mapping = {}  # Initialize font mapping dictionary

        
        # Progress tracking
        self.total_files = 0
        self.completed_files = 0
        self.failed_files = 0
        
        # Queue for thread-safe GUI updates
        self.update_queue = Queue()
        
        # Flag to prevent saving during initialization
        self._initializing = True
        
        # IMPORTANT: Load settings BEFORE building interface
        # This ensures all variables are initialized before they're used in the GUI
        self._load_rendering_settings()
        
        # Initialize the full page context prompt
        self.full_page_context_prompt = (
            "You will receive multiple text segments from a manga page. "
            "Translate each segment considering the context of all segments together. "
            "Maintain consistency in character names, tone, and style across all translations.\n\n"
            "IMPORTANT: Return your response as a valid JSON object where each key is the EXACT original text "
            "(without the [0], [1] index prefixes) and each value is the translation.\n"
            "Make sure to properly escape any special characters in the JSON:\n"
            "- Use \\n for newlines\n"
            "- Use \\\" for quotes\n"
            "- Use \\\\ for backslashes\n\n"
            "Example:\n"
            '{\n'
            '  こんにちは: Hello,\n'
            '  ありがとう: Thank you\n'
            '}\n\n'
            'Do NOT include the [0], [1], etc. prefixes in the JSON keys.'
        )
        
        # Build interface AFTER loading settings
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
        
        # Separator for context settings
        ttk.Separator(settings_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 10))
        
        # Context and Full Page Mode Settings
        context_frame = tk.LabelFrame(
            settings_frame,
            text="🔄 Context & Translation Mode",
            font=('Arial', 11, 'bold'),
            padx=10,
            pady=10
        )
        context_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Show current contextual settings from main GUI
        context_info = tk.Frame(context_frame)
        context_info.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(
            context_info,
            text="Main GUI Context Settings:",
            font=('Arial', 10, 'bold')
        ).pack(anchor=tk.W)
        
        # Display current settings
        settings_frame_display = tk.Frame(context_info)
        settings_frame_display.pack(fill=tk.X, padx=(20, 0))
        
        # Contextual enabled status
        contextual_status = "Enabled" if self.main_gui.contextual_var.get() else "Disabled"
        self.contextual_status_label = tk.Label(
            settings_frame_display,
            text=f"• Contextual Translation: {contextual_status}",
            font=('Arial', 10)
        )
        self.contextual_status_label.pack(anchor=tk.W)
        
        # History limit
        history_limit = self.main_gui.trans_history.get() if hasattr(self.main_gui, 'trans_history') else "3"
        self.history_limit_label = tk.Label(
            settings_frame_display,
            text=f"• Translation History Limit: {history_limit} exchanges",
            font=('Arial', 10)
        )
        self.history_limit_label.pack(anchor=tk.W)
        
        # Rolling history status
        rolling_status = "Enabled (Rolling Window)" if self.main_gui.translation_history_rolling_var.get() else "Disabled (Reset on Limit)"
        self.rolling_status_label = tk.Label(
            settings_frame_display,
            text=f"• Rolling History: {rolling_status}",
            font=('Arial', 10)
        )
        self.rolling_status_label.pack(anchor=tk.W)
        
        # Separator
        ttk.Separator(context_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 10))
        
        # Full Page Context Translation Settings
        full_page_frame = tk.Frame(context_frame)
        full_page_frame.pack(fill=tk.X)
        
        tk.Label(
            full_page_frame,
            text="Full Page Context Mode (Manga-specific):",
            font=('Arial', 10, 'bold')
        ).pack(anchor=tk.W, pady=(0, 5))
        
        # Enable/disable toggle
        self.full_page_context_var = tk.BooleanVar(
            value=self.main_gui.config.get('manga_full_page_context', True)
        )
        
        toggle_frame = tk.Frame(full_page_frame)
        toggle_frame.pack(fill=tk.X, padx=(20, 0))
        
        self.context_checkbox = tb.Checkbutton(
            toggle_frame,
            text="Enable Full Page Context Translation (Recommended)",
            variable=self.full_page_context_var,
            command=self._on_context_toggle,
            bootstyle="round-toggle"
        )
        self.context_checkbox.pack(side=tk.LEFT)
        
        # Edit prompt button
        tb.Button(
            toggle_frame,
            text="Edit Prompt",
            command=self._edit_context_prompt,
            bootstyle="secondary"
        ).pack(side=tk.LEFT, padx=(10, 0))
        
        # Help text
        help_text = tk.Label(
            full_page_frame,
            text="Full page context sends all text regions from the page together in a single request.\n"
                 "This allows the AI to see all text at once for more contextually accurate translations,\n"
                 "especially useful for maintaining character name consistency and understanding\n"
                 "conversation flow across multiple speech bubbles.",
            font=('Arial', 10),
            fg='gray',
            justify=tk.LEFT
        )
        help_text.pack(anchor=tk.W, padx=(20, 0), pady=(5, 0))
        
        # Pros and cons
        pros_cons_frame = tk.Frame(full_page_frame)
        pros_cons_frame.pack(fill=tk.X, padx=(20, 0), pady=(5, 0))
        
        pros_label = tk.Label(
            pros_cons_frame,
            text="✅ Better context awareness, consistent translations\n"
                 "❌ Single API call failure affects all text, may use more tokens",
            font=('Arial', 8),
            fg='gray',
            justify=tk.LEFT
        )
        pros_label.pack(anchor=tk.W)
        
        # Refresh button to update from main GUI
        tb.Button(
            context_frame,
            text="↻ Refresh from Main GUI",
            command=self._refresh_context_settings,
            bootstyle="secondary"
        ).pack(pady=(10, 0))
        
        # Text Rendering Settings Frame
        render_frame = tk.LabelFrame(
            self.parent_frame,
            text="Text Visibility Settings",
            font=('Arial', 12, 'bold'),
            padx=15,
            pady=10
        )
        render_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Advanced Settings button at the top of render_frame
        advanced_button_frame = tk.Frame(render_frame)
        advanced_button_frame.pack(fill=tk.X, pady=(0, 10))

        tb.Button(
            advanced_button_frame,
            text="⚙️ Advanced Settings",
            command=self._open_advanced_settings,
            bootstyle="info"
        ).pack(side=tk.RIGHT)

        tk.Label(
            advanced_button_frame,
            text="Configure OCR, preprocessing, and performance options",
            font=('Arial', 9),
            fg='gray'
        ).pack(side=tk.LEFT)
        
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

        # Background style selection
        style_frame = tk.Frame(render_frame)
        style_frame.pack(fill=tk.X, pady=5)

        tk.Label(style_frame, text="Background Style:", width=20, anchor='w').pack(side=tk.LEFT)

        # Radio buttons for background style
        style_selection_frame = tk.Frame(style_frame)
        style_selection_frame.pack(side=tk.LEFT, padx=10)

        tb.Radiobutton(
            style_selection_frame,
            text="Box",
            variable=self.bg_style_var,
            value="box",
            command=self._save_rendering_settings,
            bootstyle="primary"
        ).pack(side=tk.LEFT, padx=(0, 10))

        tb.Radiobutton(
            style_selection_frame,
            text="Circle",
            variable=self.bg_style_var,
            value="circle",
            command=self._save_rendering_settings,
            bootstyle="primary"
        ).pack(side=tk.LEFT, padx=(0, 10))

        tb.Radiobutton(
            style_selection_frame,
            text="Wrap",
            variable=self.bg_style_var,
            value="wrap",
            command=self._save_rendering_settings,
            bootstyle="primary"
        ).pack(side=tk.LEFT)

        # Add tooltips or descriptions
        style_help = tk.Label(
            style_frame,
            text="(Box: rounded rectangle, Circle: ellipse, Wrap: per-line)",
            font=('Arial', 9),
            fg='gray'
        )
        style_help.pack(side=tk.LEFT, padx=(10, 0))
 
        # Skip inpainting toggle - store as instance variable
        self.skip_inpainting_checkbox = tb.Checkbutton(
            render_frame, 
            text="Skip Inpainter (Recommended)", 
            variable=self.skip_inpainting_var,
            bootstyle="round-toggle",
            command=self._toggle_inpaint_visibility
        )
        self.skip_inpainting_checkbox.pack(anchor='w', pady=5)

        # Inpaint quality selection (only visible when inpainting is enabled)
        self.inpaint_quality_frame = tk.Frame(render_frame)
        self.inpaint_quality_frame.pack(fill=tk.X, pady=5)

        tk.Label(self.inpaint_quality_frame, text="Inpaint Quality:", width=20, anchor='w').pack(side=tk.LEFT)

        quality_options = [('high', 'High Quality'), ('fast', 'Fast')]
        for value, text in quality_options:
            tb.Radiobutton(
                self.inpaint_quality_frame,
                text=text,
                variable=self.inpaint_quality_var,
                value=value,
                bootstyle="primary",
                command=self._save_rendering_settings
            ).pack(side=tk.LEFT, padx=10)

        # Cloud inpainting API configuration
        api_loader_frame = tk.Frame(self.inpaint_quality_frame)
        api_loader_frame.pack(fill=tk.X, pady=(10, 0))

        # Check if API key exists
        saved_api_key = self.main_gui.config.get('replicate_api_key', '')
        if saved_api_key:
            status_text = "✅ Cloud inpainting configured"
            status_color = 'green'
        else:
            status_text = "❌ Inpainting API not configured"
            status_color = 'red'

        self.inpaint_api_status_label = tk.Label(
            api_loader_frame, 
            text=status_text,
            font=('Arial', 9),
            fg=status_color
        )
        self.inpaint_api_status_label.pack(side=tk.LEFT)

        tb.Button(
            api_loader_frame,
            text="Configure API Key",
            command=self._configure_inpaint_api,
            bootstyle="info"
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Add a clear button if API is configured
        if saved_api_key:
            tb.Button(
                api_loader_frame,
                text="Clear",
                command=self._clear_inpaint_api,
                bootstyle="secondary"
            ).pack(side=tk.LEFT, padx=(5, 0))

        # Set initial visibility based on current setting
        self._toggle_inpaint_quality_visibility()
        
        # Background size reduction
        reduction_frame = tk.Frame(render_frame)
        reduction_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(reduction_frame, text="Background Size:", width=20, anchor='w').pack(side=tk.LEFT)
        
        reduction_scale = tk.Scale(
            reduction_frame,
            from_=0.5,
            to=2.0,
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
        
        # Font size selection with mode toggle
        font_frame = tk.Frame(render_frame)
        font_frame.pack(fill=tk.X, pady=5)
        
        # Mode selection frame
        mode_frame = tk.Frame(font_frame)
        mode_frame.pack(fill=tk.X)

        tk.Label(mode_frame, text="Font Size Mode:", width=20, anchor='w').pack(side=tk.LEFT)

        # Radio buttons for mode selection
        mode_selection_frame = tk.Frame(mode_frame)
        mode_selection_frame.pack(side=tk.LEFT, padx=10)

        tb.Radiobutton(
            mode_selection_frame,
            text="Fixed Size",
            variable=self.font_size_mode_var,
            value="fixed",
            command=self._toggle_font_size_mode,
            bootstyle="primary"
        ).pack(side=tk.LEFT, padx=(0, 10))

        tb.Radiobutton(
            mode_selection_frame,
            text="Dynamic Multiplier",
            variable=self.font_size_mode_var,
            value="multiplier",
            command=self._toggle_font_size_mode,
            bootstyle="primary"
        ).pack(side=tk.LEFT)

        # Fixed font size frame
        self.fixed_size_frame = tk.Frame(font_frame)
        # Don't pack yet - let _toggle_font_size_mode handle it

        tk.Label(self.fixed_size_frame, text="Font Size:", width=20, anchor='w').pack(side=tk.LEFT)

        font_size_spinbox = tb.Spinbox(
            self.fixed_size_frame,
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

        tk.Label(self.fixed_size_frame, text="(0 = Auto)", font=('Arial', 9), fg='gray').pack(side=tk.LEFT)

        # Dynamic multiplier frame
        self.multiplier_frame = tk.Frame(font_frame)
        # Don't pack yet - let _toggle_font_size_mode handle it

        tk.Label(self.multiplier_frame, text="Size Multiplier:", width=20, anchor='w').pack(side=tk.LEFT)

        multiplier_scale = tk.Scale(
            self.multiplier_frame,
            from_=0.5,
            to=2.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.font_size_multiplier_var,
            command=self._update_multiplier_label,
            length=200
        )
        multiplier_scale.pack(side=tk.LEFT, padx=10)

        self.multiplier_label = tk.Label(self.multiplier_frame, text="1.0x", width=5)
        self.multiplier_label.pack(side=tk.LEFT)

        tk.Label(self.multiplier_frame, text="(Scales with panel size)", font=('Arial', 9), fg='gray').pack(side=tk.LEFT, padx=5)

        # Constraint checkbox frame (only visible in multiplier mode)
        self.constraint_frame = tk.Frame(font_frame)
        # Don't pack yet - let _toggle_font_size_mode handle it
        
        self.constrain_checkbox = tb.Checkbutton(
            self.constraint_frame,
            text="Constrain text to bubble boundaries",
            variable=self.constrain_to_bubble_var,
            command=self._save_rendering_settings,
            bootstyle="primary"
        )
        self.constrain_checkbox.pack(side=tk.LEFT, padx=(20, 0))

        tk.Label(
            self.constraint_frame, 
            text="(Unchecked allows text to exceed bubbles)", 
            font=('Arial', 9), 
            fg='gray'
        ).pack(side=tk.LEFT, padx=5)

        # Initialize visibility AFTER all frames are created
        self._toggle_font_size_mode()

        # Minimum font size setting (for auto mode)
        min_size_frame = tk.Frame(render_frame)
        min_size_frame.pack(fill=tk.X, pady=5)

        tk.Label(min_size_frame, text="Minimum Font Size:", width=20, anchor='w').pack(side=tk.LEFT)

        self.min_readable_size_var = tk.IntVar(value=self.main_gui.config.get('manga_min_readable_size', 16))

        min_size_spinbox = ttk.Spinbox(
            min_size_frame,
            from_=10,
            to=24,
            textvariable=self.min_readable_size_var,
            width=10,
            command=self._save_rendering_settings
        )
        min_size_spinbox.pack(side=tk.LEFT, padx=10)

        tk.Label(
            min_size_frame, 
            text="(Auto mode won't go below this)", 
            font=('Arial', 9), 
            fg='gray'
        ).pack(side=tk.LEFT, padx=5)
    
        # Maximum font size setting
        max_size_frame = tk.Frame(render_frame)
        max_size_frame.pack(fill=tk.X, pady=5)

        tk.Label(max_size_frame, text="Maximum Font Size:", width=20, anchor='w').pack(side=tk.LEFT)

        self.max_font_size_var = tk.IntVar(value=self.main_gui.config.get('manga_max_font_size', 24))

        max_size_spinbox = ttk.Spinbox(
            max_size_frame,
            from_=20,
            to=100,
            textvariable=self.max_font_size_var,
            width=10,
            command=self._save_rendering_settings
        )
        max_size_spinbox.pack(side=tk.LEFT, padx=10)

        tk.Label(
            max_size_frame, 
            text="(Limits maximum text size)", 
            font=('Arial', 9), 
            fg='gray'
        ).pack(side=tk.LEFT, padx=5)

        # Text wrapping mode
        wrap_frame = tk.Frame(render_frame)
        wrap_frame.pack(fill=tk.X, pady=5)

        self.strict_text_wrapping_var = tk.BooleanVar(value=self.main_gui.config.get('manga_strict_text_wrapping', False))

        self.strict_wrap_checkbox = tb.Checkbutton(
            wrap_frame,
            text="Strict text wrapping (force text to fit within bubbles)",
            variable=self.strict_text_wrapping_var,
            command=self._save_rendering_settings,
            bootstyle="primary"
        )
        self.strict_wrap_checkbox.pack(side=tk.LEFT)

        tk.Label(
            wrap_frame, 
            text="(Break words with hyphens if needed)", 
            font=('Arial', 9), 
            fg='gray'
        ).pack(side=tk.LEFT, padx=5)
    
        # Update multiplier label with loaded value
        self._update_multiplier_label(self.font_size_multiplier_var.get())
        
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
        
        # Color button and preview
        color_button_frame = tk.Frame(color_frame)
        color_button_frame.pack(side=tk.LEFT, padx=10)
        
        # Color preview
        self.color_preview = tk.Canvas(color_button_frame, width=40, height=30, 
                                     highlightthickness=1, highlightbackground="gray")
        self.color_preview.pack(side=tk.LEFT, padx=(0, 10))
        
        # RGB display label
        r, g, b = self.text_color_r.get(), self.text_color_g.get(), self.text_color_b.get()
        self.rgb_label = tk.Label(color_button_frame, text=f"RGB({r},{g},{b})", width=12)
        self.rgb_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Color picker button
        def pick_font_color():
            from tkinter import colorchooser
            # Get current color
            current_color = (self.text_color_r.get(), self.text_color_g.get(), self.text_color_b.get())
            color_hex = f'#{current_color[0]:02x}{current_color[1]:02x}{current_color[2]:02x}'
            
            # Open color dialog
            color = colorchooser.askcolor(initialcolor=color_hex, parent=self.dialog, 
                                         title="Choose Font Color")
            if color[0]:  # If a color was chosen (not cancelled)
                # Update RGB values
                self.text_color_r.set(int(color[0][0]))
                self.text_color_g.set(int(color[0][1]))
                self.text_color_b.set(int(color[0][2]))
                # Update display
                self.rgb_label.config(text=f"RGB({int(color[0][0])},{int(color[0][1])},{int(color[0][2])})")
                self._update_color_preview(None)
        
        tb.Button(
            color_button_frame,
            text="Choose Color",
            command=pick_font_color,
            bootstyle="info"
        ).pack(side=tk.LEFT)
        
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
        
        # Shadow color button and preview
        shadow_button_frame = tk.Frame(shadow_color_frame)
        shadow_button_frame.pack(side=tk.LEFT, padx=10)
        
        # Shadow color preview
        self.shadow_preview = tk.Canvas(shadow_button_frame, width=30, height=25, 
                                      highlightthickness=1, highlightbackground="gray")
        self.shadow_preview.pack(side=tk.LEFT, padx=(0, 10))
        
        # Shadow RGB display label
        sr, sg, sb = self.shadow_color_r.get(), self.shadow_color_g.get(), self.shadow_color_b.get()
        self.shadow_rgb_label = tk.Label(shadow_button_frame, text=f"RGB({sr},{sg},{sb})", width=15)
        self.shadow_rgb_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Shadow color picker button
        def pick_shadow_color():
            from tkinter import colorchooser
            # Get current color
            current_color = (self.shadow_color_r.get(), self.shadow_color_g.get(), self.shadow_color_b.get())
            color_hex = f'#{current_color[0]:02x}{current_color[1]:02x}{current_color[2]:02x}'
            
            # Open color dialog
            color = colorchooser.askcolor(initialcolor=color_hex, parent=self.dialog, 
                                         title="Choose Shadow Color")
            if color[0]:  # If a color was chosen (not cancelled)
                # Update RGB values
                self.shadow_color_r.set(int(color[0][0]))
                self.shadow_color_g.set(int(color[0][1]))
                self.shadow_color_b.set(int(color[0][2]))
                # Update display
                self.shadow_rgb_label.config(text=f"RGB({int(color[0][0])},{int(color[0][1])},{int(color[0][2])})")
                self._update_shadow_preview(None)
        
        tb.Button(
            shadow_button_frame,
            text="Choose Color",
            command=pick_shadow_color,
            bootstyle="info",
            width=12
        ).pack(side=tk.LEFT)
        
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
        
    def _open_advanced_settings(self):
        """Open the manga advanced settings dialog"""
        try:
            def on_settings_saved(settings):
                """Callback when settings are saved"""
                # Update config with new settings
                self.main_gui.config['manga_settings'] = settings
                
                # Reload settings in translator if it exists
                if self.translator:
                    self._log("📋 Reloading settings in translator...", "info")
                    # The translator will pick up new settings on next operation
                
                self._log("✅ Advanced settings saved and applied", "success")
            
            # Open the settings dialog
            MangaSettingsDialog(
                parent=self.dialog,
                main_gui=self.main_gui,
                config=self.main_gui.config,
                callback=on_settings_saved
            )
            
        except Exception as e:
            self._log(f"❌ Error opening settings dialog: {str(e)}", "error")
            messagebox.showerror("Error", f"Failed to open settings dialog:\n{str(e)}")
        
    def _toggle_font_size_mode(self):
        """Toggle between fixed size and multiplier modes"""
        mode = self.font_size_mode_var.get()
        
        # Check if frames exist before trying to pack/unpack them
        if hasattr(self, 'fixed_size_frame') and hasattr(self, 'multiplier_frame'):
            if mode == "fixed":
                self.fixed_size_frame.pack(fill=tk.X, pady=(5, 0))
                self.multiplier_frame.pack_forget()
                # Hide constraint frame if it exists
                if hasattr(self, 'constraint_frame'):
                    self.constraint_frame.pack_forget()
            else:
                self.fixed_size_frame.pack_forget()
                self.multiplier_frame.pack(fill=tk.X, pady=(5, 0))
                # Show constraint frame if it exists
                if hasattr(self, 'constraint_frame'):
                    self.constraint_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Only save if we're not initializing
        if not hasattr(self, '_initializing') or not self._initializing:
            self._save_rendering_settings()

    def _update_multiplier_label(self, value):
        """Update multiplier label"""
        self.multiplier_label.config(text=f"{float(value):.1f}x")
        # Auto-save on change
        self._save_rendering_settings()
    
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
        self.bg_opacity_var = tk.IntVar(value=config.get('manga_bg_opacity', 130))
        self.bg_style_var = tk.StringVar(value=config.get('manga_bg_style', 'circle'))
        self.bg_reduction_var = tk.DoubleVar(value=config.get('manga_bg_reduction', 1.0))
        self.font_size_var = tk.IntVar(value=config.get('manga_font_size', 0))
        self.selected_font_path = config.get('manga_font_path', None)
        self.skip_inpainting_var = tk.BooleanVar(value=config.get('manga_skip_inpainting', True))
        self.inpaint_quality_var = tk.StringVar(value=config.get('manga_inpaint_quality', 'high'))
        self.inpaint_dilation_var = tk.IntVar(value=config.get('manga_inpaint_dilation', 15))
        self.inpaint_passes_var = tk.IntVar(value=config.get('manga_inpaint_passes', 2))
        self.font_size_mode_var = tk.StringVar(value=config.get('manga_font_size_mode', 'fixed'))
        self.font_size_multiplier_var = tk.DoubleVar(value=config.get('manga_font_size_multiplier', 1.0))
        self.constrain_to_bubble_var = tk.BooleanVar(value=config.get('manga_constrain_to_bubble', True))
        self.max_font_size_var = tk.IntVar(value=config.get('manga_max_font_size', 24))
        self.strict_text_wrapping_var = tk.BooleanVar(value=config.get('manga_strict_text_wrapping', False))
        
        # Font color settings
        manga_text_color = config.get('manga_text_color', [102, 0, 0])
        self.text_color_r = tk.IntVar(value=manga_text_color[0])
        self.text_color_g = tk.IntVar(value=manga_text_color[1])
        self.text_color_b = tk.IntVar(value=manga_text_color[2])
        
        # Shadow settings
        self.shadow_enabled_var = tk.BooleanVar(value=config.get('manga_shadow_enabled', True))
        manga_shadow_color = config.get('manga_shadow_color', [204, 128, 128])
        self.shadow_color_r = tk.IntVar(value=manga_shadow_color[0])
        self.shadow_color_g = tk.IntVar(value=manga_shadow_color[1])
        self.shadow_color_b = tk.IntVar(value=manga_shadow_color[2])
        self.shadow_offset_x_var = tk.IntVar(value=config.get('manga_shadow_offset_x', 2))
        self.shadow_offset_y_var = tk.IntVar(value=config.get('manga_shadow_offset_y', 2))
        self.shadow_blur_var = tk.IntVar(value=config.get('manga_shadow_blur', 0))
        
        # Initialize font_style_var with saved value or default
        saved_font_style = config.get('manga_font_style', 'Default')
        self.font_style_var = tk.StringVar(value=saved_font_style)
        
        # Full page context settings
        self.full_page_context_var = tk.BooleanVar(value=config.get('manga_full_page_context', False))
        self.full_page_context_prompt = config.get('manga_full_page_context_prompt', 
            "You will receive multiple text segments from a manga page. "
            "Translate each segment considering the context of all segments together. "
            "Maintain consistency in character names, tone, and style across all segments."
        )
        
        # Output settings
        self.create_subfolder_var = tk.BooleanVar(value=config.get('manga_create_subfolder', True))
 
    def _set_min_size(self, size):
        """Set minimum font size from preset"""
        self.min_readable_size_var.set(size)
        self._save_rendering_settings()
    
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
        self.main_gui.config['manga_inpaint_quality'] = self.inpaint_quality_var.get()
        self.main_gui.config['manga_inpaint_dilation'] = self.inpaint_dilation_var.get()
        self.main_gui.config['manga_inpaint_passes'] = self.inpaint_passes_var.get()
        self.main_gui.config['manga_font_size_mode'] = self.font_size_mode_var.get()
        self.main_gui.config['manga_font_size_multiplier'] = self.font_size_multiplier_var.get()
        self.main_gui.config['manga_font_style'] = self.font_style_var.get()
        self.main_gui.config['manga_constrain_to_bubble'] = self.constrain_to_bubble_var.get()
        self.main_gui.config['manga_min_readable_size'] = self.min_readable_size_var.get()
        self.main_gui.config['manga_max_font_size'] = self.max_font_size_var.get()
        self.main_gui.config['manga_strict_text_wrapping'] = self.strict_text_wrapping_var.get()
        
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
        
        # Save output settings
        if hasattr(self, 'create_subfolder_var'):
            self.main_gui.config['manga_create_subfolder'] = self.create_subfolder_var.get()
        
        # Save full page context settings
        self.main_gui.config['manga_full_page_context'] = self.full_page_context_var.get()
        self.main_gui.config['manga_full_page_context_prompt'] = self.full_page_context_prompt
        
        # Call main GUI's save_config to persist to file
        if hasattr(self.main_gui, 'save_config'):
            self.main_gui.save_config(show_message=False)
    
    def _on_context_toggle(self):
        """Handle full page context toggle"""
        enabled = self.full_page_context_var.get()
        self._save_rendering_settings()
    
    def _edit_context_prompt(self):
            """Open dialog to edit full page context prompt"""
            # Store parent canvas for scroll restoration
            parent_canvas = self.canvas
            
            # Use WindowManager to create scrollable dialog
            dialog, scrollable_frame, canvas = self.main_gui.wm.setup_scrollable(
                self.dialog,  # parent window
                "Edit Full Page Context Prompt",
                width=700,
                height=500,
                max_width_ratio=0.7,
                max_height_ratio=0.8
            )
            
            # Instructions
            instructions = tk.Label(
                scrollable_frame,
                text="Edit the prompt used for full page context translation.\n"
                     "This will be appended to the main translation system prompt.",
                font=('Arial', 10),
                justify=tk.LEFT
            )
            instructions.pack(padx=20, pady=(20, 10))
            
            # Text editor with UIHelper for undo/redo support
            text_editor = self.main_gui.ui.setup_scrollable_text(
                scrollable_frame,
                wrap=tk.WORD,
                height=15
            )
            text_editor.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
            
            # Insert current prompt
            text_editor.insert(1.0, self.full_page_context_prompt)
            
            # Button frame
            button_frame = tk.Frame(scrollable_frame)
            button_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
            
            def close_dialog():
                """Properly close dialog and restore parent scrolling"""
                try:
                    # Clean up this dialog's scrolling
                    if hasattr(dialog, '_cleanup_scrolling') and callable(dialog._cleanup_scrolling):
                        dialog._cleanup_scrolling()
                except:
                    pass
                
                # Destroy the dialog
                dialog.destroy()
            
            def save_prompt():
                self.full_page_context_prompt = text_editor.get(1.0, tk.END).strip()
                self._save_rendering_settings()
                self._log("✅ Updated full page context prompt", "success")
                close_dialog()
            
            def reset_prompt():
                default_prompt = (
                    "You will receive multiple text segments from a manga page. "
                    "Translate each segment considering the context of all segments together. "
                    "Maintain consistency in character names, tone, and style across all translations.\n\n"
                    "IMPORTANT: Return your response as a JSON object where each key is the EXACT original text "
                    "(without the [0], [1] index prefixes) and each value is the translation. Example:\n"
                    '{\n'
                    '  こんにちは: Hello,\n'
                    '  ありがとう": Thank you\n'
                    '}\n\n'
                    'Do NOT include the [0], [1], etc. prefixes in the JSON keys.'
                )
                text_editor.delete(1.0, tk.END)
                text_editor.insert(1.0, default_prompt)
            
            # Buttons
            tb.Button(
                button_frame,
                text="Save",
                command=save_prompt,
                bootstyle="primary"
            ).pack(side=tk.LEFT, padx=(0, 10))
            
            tb.Button(
                button_frame,
                text="Reset to Default",
                command=reset_prompt,
                bootstyle="secondary"
            ).pack(side=tk.LEFT, padx=(0, 10))
            
            tb.Button(
                button_frame,
                text="Cancel",
                command=close_dialog,
                bootstyle="secondary"
            ).pack(side=tk.LEFT)
            
            # Auto-resize dialog to fit content
            self.main_gui.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.7, max_height_ratio=0.6)
            
            # Handle window close
            dialog.protocol("WM_DELETE_WINDOW", close_dialog)
    
    def _refresh_context_settings(self):
        """Refresh context settings from main GUI"""
        # Update labels
        contextual_status = "Enabled" if self.main_gui.contextual_var.get() else "Disabled"
        self.contextual_status_label.config(text=f"• Contextual Translation: {contextual_status}")
        
        history_limit = self.main_gui.trans_history.get() if hasattr(self.main_gui, 'trans_history') else "3"
        self.history_limit_label.config(text=f"• Translation History Limit: {history_limit} exchanges")
        
        rolling_status = "Enabled (Rolling Window)" if self.main_gui.translation_history_rolling_var.get() else "Disabled (Reset on Limit)"
        self.rolling_status_label.config(text=f"• Rolling History: {rolling_status}")
        
        self._log("✅ Refreshed context settings from main GUI", "success")
    
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
            if hasattr(self.main_gui, 'save_config'):
                self.main_gui.save_config(show_message=False)

            
            # Update button state immediately
            self.start_button.config(state=tk.NORMAL)
            
            # Update credentials display
            self.creds_label.config(text=os.path.basename(file_path), fg='green')
            
            # Update status if we have a reference
            if hasattr(self, 'status_label'):
                self.status_label.config(text="✅ Ready", fg="green")
            
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
        """Get list of available fonts from system and custom directories"""
        fonts = ["Default"]  # Default option
        
        # Reset font mapping
        self.font_mapping = {}
        
        # Windows system fonts with paths
        windows_fonts = [
            # Basic fonts
            ("Arial", "C:/Windows/Fonts/arial.ttf"),
            ("Calibri", "C:/Windows/Fonts/calibri.ttf"),
            ("Comic Sans MS", "C:/Windows/Fonts/comic.ttf"),
            ("Tahoma", "C:/Windows/Fonts/tahoma.ttf"),
            ("Times New Roman", "C:/Windows/Fonts/times.ttf"),
            ("Verdana", "C:/Windows/Fonts/verdana.ttf"),
            ("Georgia", "C:/Windows/Fonts/georgia.ttf"),
            ("Impact", "C:/Windows/Fonts/impact.ttf"),
            ("Trebuchet MS", "C:/Windows/Fonts/trebuc.ttf"),
            ("Courier New", "C:/Windows/Fonts/cour.ttf"),
            
            # Japanese fonts
            ("MS Gothic", "C:/Windows/Fonts/msgothic.ttc"),
            ("MS Mincho", "C:/Windows/Fonts/msmincho.ttc"),
            ("Meiryo", "C:/Windows/Fonts/meiryo.ttc"),
            ("Yu Gothic", "C:/Windows/Fonts/yugothic.ttc"),
            ("Yu Mincho", "C:/Windows/Fonts/yumin.ttc"),
            
            # Korean fonts
            ("Malgun Gothic", "C:/Windows/Fonts/malgun.ttf"),
            ("Gulim", "C:/Windows/Fonts/gulim.ttc"),
            ("Dotum", "C:/Windows/Fonts/dotum.ttc"),
            ("Batang", "C:/Windows/Fonts/batang.ttc"),
            
            # Chinese fonts
            ("SimSun", "C:/Windows/Fonts/simsun.ttc"),
            ("SimHei", "C:/Windows/Fonts/simhei.ttf"),
            ("Microsoft YaHei", "C:/Windows/Fonts/msyh.ttc"),
            ("Microsoft JhengHei", "C:/Windows/Fonts/msjh.ttc"),
            ("KaiTi", "C:/Windows/Fonts/simkai.ttf"),
            ("FangSong", "C:/Windows/Fonts/simfang.ttf"),
        ]
        
        # Check which fonts exist and add to mapping
        for font_name, font_path in windows_fonts:
            if os.path.exists(font_path):
                fonts.append(font_name)
                self.font_mapping[font_name] = font_path
        
        # Check for custom fonts directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fonts_dir = os.path.join(script_dir, "fonts")
        
        if os.path.exists(fonts_dir):
            for root, dirs, files in os.walk(fonts_dir):
                for font_file in files:
                    if font_file.endswith(('.ttf', '.ttc', '.otf')):
                        font_path = os.path.join(root, font_file)
                        font_name = os.path.splitext(font_file)[0]
                        # Add category from folder
                        category = os.path.basename(root)
                        if category != "fonts":
                            font_name = f"{font_name} ({category})"
                        fonts.append(font_name)
                        self.font_mapping[font_name] = font_path
        
        # Load previously saved custom fonts
        if 'custom_fonts' in self.main_gui.config:
            for custom_font in self.main_gui.config['custom_fonts']:
                if os.path.exists(custom_font['path']):
                    # Check if this font is already in the list
                    if custom_font['name'] not in fonts:
                        fonts.append(custom_font['name'])
                        self.font_mapping[custom_font['name']] = custom_font['path']
        
        # Add custom fonts option at the end
        fonts.append("Browse Custom Font...")
        
        return fonts
    
    def _on_font_selected(self, event):
        """Handle font selection"""
        selected = self.font_style_var.get()
        
        if selected == "Default":
            self.selected_font_path = None
        elif selected == "Browse Custom Font...":
            # Open file dialog to select custom font
            font_path = filedialog.askopenfilename(
                title="Select Font File",
                filetypes=[
                    ("Font files", "*.ttf *.ttc *.otf"),
                    ("TrueType fonts", "*.ttf"),
                    ("TrueType collections", "*.ttc"),
                    ("OpenType fonts", "*.otf"),
                    ("All files", "*.*")
                ]
            )
            
            if font_path:
                # Add to combo box
                font_name = os.path.basename(font_path)
                current_values = list(self.font_combo['values'])
                
                # Insert before "Browse Custom Font..." option
                if font_name not in [n for n in self.font_mapping.keys()]:
                    current_values.insert(-1, font_name)
                    self.font_combo['values'] = current_values
                    self.font_combo.set(font_name)
                    
                    # Update font mapping
                    self.font_mapping[font_name] = font_path
                    self.selected_font_path = font_path
                    
                    # Save custom font to config
                    if 'custom_fonts' not in self.main_gui.config:
                        self.main_gui.config['custom_fonts'] = []
                    
                    custom_font_entry = {'name': font_name, 'path': font_path}
                    # Check if this exact entry already exists
                    font_exists = False
                    for existing_font in self.main_gui.config['custom_fonts']:
                        if existing_font['path'] == font_path:
                            font_exists = True
                            break
                    
                    if not font_exists:
                        self.main_gui.config['custom_fonts'].append(custom_font_entry)
                        # Save config immediately to persist custom fonts
                        if hasattr(self.main_gui, 'save_config'):
                            self.main_gui.save_config(show_message=False)
                else:
                    # Font already exists, just select it
                    self.font_combo.set(font_name)
                    self.selected_font_path = self.font_mapping[font_name]
            else:
                # User cancelled, revert to previous selection
                if hasattr(self, 'previous_font_selection'):
                    self.font_combo.set(self.previous_font_selection)
                else:
                    self.font_combo.set("Default")
                return
        else:
            # Check if it's in the font mapping
            if selected in self.font_mapping:
                self.selected_font_path = self.font_mapping[selected]
            else:
                # This shouldn't happen, but just in case
                self.selected_font_path = None
        
        # Store current selection for next time
        self.previous_font_selection = selected
        
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
        
    def _toggle_inpaint_quality_visibility(self):
        """Show/hide inpaint quality options based on skip_inpainting setting"""
        if hasattr(self, 'inpaint_quality_frame'):
            if self.skip_inpainting_var.get():
                # Hide quality options when inpainting is skipped
                self.inpaint_quality_frame.pack_forget()
            else:
                # Show quality options when inpainting is enabled
                self.inpaint_quality_frame.pack(fill=tk.X, pady=5, after=self.skip_inpainting_checkbox)

    def _toggle_inpaint_visibility(self):
        """Toggle visibility of all inpaint-related options"""
        self._toggle_inpaint_quality_visibility()
        self._toggle_inpaint_controls_visibility()

    def _toggle_inpaint_controls_visibility(self):
            """Toggle visibility of inpaint controls (mask expansion and passes) based on skip inpainting setting"""
            # Just return if the frame doesn't exist - prevents AttributeError
            if not hasattr(self, 'inpaint_controls_frame'):
                return
                
            if self.skip_inpainting_var.get():
                self.inpaint_controls_frame.pack_forget()
            else:
                # Pack it back in the right position
                self.inpaint_controls_frame.pack(fill=tk.X, pady=5, after=self.inpaint_quality_frame)

    def _configure_inpaint_api(self):
        """Configure cloud inpainting API"""
        # Show instructions
        result = messagebox.askyesno(
            "Configure Cloud Inpainting",
            "Cloud inpainting uses Replicate API for questionable results.\n\n"
            "1. Go to replicate.com and sign up (free tier available?)\n"
            "2. Get your API token from Account Settings\n"
            "3. Enter it here\n\n"
            "Pricing: ~$0.0023 per image?\n"
            "Free tier: ~100 images per month?\n\n"
            "Would you like to proceed?"
        )
        
        if not result:
            return
        
        # Open Replicate page
        import webbrowser
        webbrowser.open("https://replicate.com/account/api-tokens")
        
        # Get API key from user using WindowManager
        dialog = self.main_gui.wm.create_simple_dialog(
            self.main_gui.master,
            "Replicate API Key",
            width=400,
            height=200,
            hide_initially=True  # Hide initially so we can position it
        )
        
        # Force the height by overriding after creation
        dialog.update_idletasks()  # Process pending geometry
        dialog.minsize(400, 200)   # Set minimum size
        dialog.maxsize(720, 250)   # Set maximum size to lock it
        
        # Get cursor position
        cursor_x = self.main_gui.master.winfo_pointerx()
        cursor_y = self.main_gui.master.winfo_pointery()
        
        # Offset the dialog slightly so it doesn't appear directly under cursor
        # This prevents the cursor from immediately being over a button
        offset_x = 10
        offset_y = 10
        
        # Ensure dialog doesn't go off-screen
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        
        # Adjust position if it would go off-screen
        if cursor_x + 400 + offset_x > screen_width:
            cursor_x = screen_width - 400 - offset_x
        if cursor_y + 150 + offset_y > screen_height:
            cursor_y = screen_height - 150 - offset_y
        
        # Set position and show
        dialog.geometry(f"400x150+{cursor_x + offset_x}+{cursor_y + offset_y}")
        dialog.deiconify()  # Show the dialog
        
        # Variables
        api_key_var = tk.StringVar()
        result = {'key': None}
        
        # Content
        frame = tk.Frame(dialog, padx=20, pady=20)
        frame.pack(fill='both', expand=True)
        
        tk.Label(frame, text="Enter your Replicate API key:").pack(anchor='w', pady=(0, 10))
        
        # Entry with show/hide
        entry_frame = tk.Frame(frame)
        entry_frame.pack(fill='x')
        
        entry = tk.Entry(entry_frame, textvariable=api_key_var, show='*', width=35)
        entry.pack(side='left', fill='x', expand=True)
        
        # Toggle show/hide
        def toggle_show():
            current = entry.cget('show')
            entry.config(show='' if current else '*')
            show_btn.config(text='Hide' if current else 'Show')
        
        show_btn = tb.Button(entry_frame, text="Show", command=toggle_show, width=8)
        show_btn.pack(side='left', padx=(10, 0))
        
        # Buttons
        btn_frame = tk.Frame(frame)
        btn_frame.pack(fill='x', pady=(20, 0))
        
        def on_ok():
            result['key'] = api_key_var.get().strip()
            dialog.destroy()
        
        tb.Button(btn_frame, text="OK", command=on_ok, bootstyle="success").pack(side='right', padx=(5, 0))
        tb.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side='right')
        
        # Focus and bindings
        entry.focus_set()
        dialog.bind('<Return>', lambda e: on_ok())
        dialog.bind('<Escape>', lambda e: dialog.destroy())
        
        # Wait for dialog
        dialog.wait_window()
        
        api_key = result['key']
        
        if api_key:
            try:
                # Save the API key
                self.main_gui.config['replicate_api_key'] = api_key
                self.main_gui.save_config(show_message=False)
                
                # Update UI
                self.inpaint_api_status_label.config(
                    text="✅ Cloud inpainting configured",
                    fg='green'
                )
                
                # Add clear button if it doesn't exist
                clear_button_exists = False
                for widget in self.inpaint_api_status_label.master.winfo_children():
                    if isinstance(widget, tb.Button) and widget.cget('text') == 'Clear':
                        clear_button_exists = True
                        break
                
                if not clear_button_exists:
                    tb.Button(
                        self.inpaint_api_status_label.master,
                        text="Clear",
                        command=self._clear_inpaint_api,
                        bootstyle="secondary"
                    ).pack(side=tk.LEFT, padx=(5, 0))
                
                # Set flag on translator
                if self.translator:
                    self.translator.use_cloud_inpainting = True
                    self.translator.replicate_api_key = api_key
                    
                self._log("✅ Cloud inpainting API configured", "success")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save API key:\n{str(e)}")

    def _clear_inpaint_api(self):
        """Clear the inpainting API configuration"""
        self.main_gui.config['replicate_api_key'] = ''
        self.main_gui.save_config(show_message=False)
        
        self.inpaint_api_status_label.config(
            text="❌ Inpainting API not configured", 
            fg='red'
        )
        
        if hasattr(self, 'translator') and self.translator:
            self.translator.use_cloud_inpainting = False
            self.translator.replicate_api_key = None
            
        self._log("🗑️ Cleared inpainting API configuration", "info")
        
        # Find and destroy the clear button
        for widget in self.inpaint_api_status_label.master.winfo_children():
            if isinstance(widget, tb.Button) and widget.cget('text') == 'Clear':
                widget.destroy()
                break 
            
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
            
            # Get current API key and model
            api_key = None
            model = 'gemini-1.5-flash'  # default
            
            # Try to get API key from various sources
            if hasattr(self.main_gui, 'api_key_entry') and self.main_gui.api_key_entry.get().strip():
                api_key = self.main_gui.api_key_entry.get().strip()
            elif hasattr(self.main_gui, 'config') and self.main_gui.config.get('api_key'):
                api_key = self.main_gui.config.get('api_key')
            
            # Try to get model - ALWAYS get the current selection from GUI
            if hasattr(self.main_gui, 'model_var'):
                model = self.main_gui.model_var.get()
            elif hasattr(self.main_gui, 'config') and self.main_gui.config.get('model'):
                model = self.main_gui.config.get('model')
            
            if not api_key:
                messagebox.showerror("Error", "API key not found.\nPlease configure your API key in the main settings.")
                return
            
            # Check if we need to create or update the client
            needs_new_client = False
            
            if not hasattr(self.main_gui, 'client') or not self.main_gui.client:
                needs_new_client = True
                self._log(f"Creating new API client with model: {model}", "info")
            elif hasattr(self.main_gui.client, 'model') and self.main_gui.client.model != model:
                needs_new_client = True
                self._log(f"Model changed from {self.main_gui.client.model} to {model}, creating new client", "info")
            
            if needs_new_client:
                # Create the unified client with the current model
                try:
                    from unified_api_client import UnifiedClient
                    self.main_gui.client = UnifiedClient(model=model, api_key=api_key)
                    self._log(f"Created API client with model: {model}", "info")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to create API client:\n{str(e)}")
                    return
                    
            # Reset the translator's history manager for new batch
            if hasattr(self, 'translator') and self.translator and hasattr(self.translator, 'reset_history_manager'):
                self.translator.reset_history_manager()
    
            # Initialize translator if needed
            if not self.translator:
                try:
                    self.translator = MangaTranslator(
                        google_creds,
                        self.main_gui.client,
                        self.main_gui,
                        log_callback=self._log
                    )
                    # Set cloud inpainting if configured
                    saved_api_key = self.main_gui.config.get('replicate_api_key', '')
                    if saved_api_key:
                        self.translator.use_cloud_inpainting = True
                        self.translator.replicate_api_key = saved_api_key                    
                    # Apply text rendering settings
                    self._apply_rendering_settings()
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to initialize translator:\n{str(e)}")
                    self._log(f"Initialization error: {str(e)}", "error")
                    self._log(traceback.format_exc(), "error")
                    return
            else:
                # Update the translator with the new client if model changed
                if needs_new_client and hasattr(self.translator, 'client'):
                    self.translator.client = self.main_gui.client
                    self._log(f"Updated translator with new API client", "info")
                
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
            self._log(f"Contextual: {'Enabled' if self.main_gui.contextual_var.get() else 'Disabled'}", "info")
            self._log(f"History limit: {self.main_gui.trans_history.get()} exchanges", "info")
            self._log(f"Rolling history: {'Enabled' if self.main_gui.translation_history_rolling_var.get() else 'Disabled'}", "info")
            self._log(f"  Full Page Context: {'Enabled' if self.full_page_context_var.get() else 'Disabled'}", "info")
            
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
            
            # Determine font size value based on mode
            if self.font_size_mode_var.get() == 'multiplier':
                # Pass negative value to indicate multiplier mode
                font_size = -self.font_size_multiplier_var.get()
            else:
                # Fixed mode - use the font size value directly
                font_size = self.font_size_var.get() if self.font_size_var.get() > 0 else None
            
            self.translator.update_text_rendering_settings(
                bg_opacity=self.bg_opacity_var.get(),
                bg_style=self.bg_style_var.get(),
                bg_reduction=self.bg_reduction_var.get(),
                font_style=self.selected_font_path,
                font_size=font_size,
                text_color=text_color,
                shadow_enabled=self.shadow_enabled_var.get(),
                shadow_color=shadow_color,
                shadow_offset_x=self.shadow_offset_x_var.get(),
                shadow_offset_y=self.shadow_offset_y_var.get(),
                shadow_blur=self.shadow_blur_var.get()
            )
            
            # Update font mode and multiplier explicitly
            self.translator.font_size_mode = self.font_size_mode_var.get()
            self.translator.font_size_multiplier = self.font_size_multiplier_var.get()
            self.translator.min_readable_size = self.min_readable_size_var.get()
            self.translator.min_readable_size = self.min_readable_size_var.get()
            self.translator.max_font_size_limit = self.max_font_size_var.get()
            self.translator.strict_text_wrapping = self.strict_text_wrapping_var.get()
            
            # Update constrain to bubble setting
            if hasattr(self, 'constrain_to_bubble_var'):
                self.translator.constrain_to_bubble = self.constrain_to_bubble_var.get()
            
            if hasattr(self, 'skip_inpainting_var'):
                self.translator.skip_inpainting = self.skip_inpainting_var.get()
            
            # Set full page context mode
            self.translator.set_full_page_context(
                enabled=self.full_page_context_var.get(),
                custom_prompt=self.full_page_context_prompt
            )
            
            # Update logging to include new settings
            self._log(f"Applied rendering settings:", "info")
            self._log(f"  Background: {self.bg_style_var.get()} @ {int(self.bg_opacity_var.get()/255*100)}% opacity", "info")
            self._log(f"  Font: {os.path.basename(self.selected_font_path) if self.selected_font_path else 'Default'}", "info")
            self._log(f"  Minimum Font Size: {self.min_readable_size_var.get()}pt", "info")
            self._log(f"  Maximum Font Size: {self.max_font_size_var.get()}pt", "info")
            self._log(f"  Strict Text Wrapping: {'Enabled (force fit)' if self.strict_text_wrapping_var.get() else 'Disabled (allow overflow)'}", "info")
            
            # Log font size mode
            if self.font_size_mode_var.get() == 'multiplier':
                self._log(f"  Font Size: Dynamic multiplier ({self.font_size_multiplier_var.get():.1f}x)", "info")
                if hasattr(self, 'constrain_to_bubble_var'):
                    constraint_status = "constrained" if self.constrain_to_bubble_var.get() else "unconstrained"
                    self._log(f"  Text Constraint: {constraint_status}", "info")
            else:
                size_text = f"{self.font_size_var.get()}pt" if self.font_size_var.get() > 0 else "Auto"
                self._log(f"  Font Size: Fixed ({size_text})", "info")
            
            self._log(f"  Text Color: RGB({text_color[0]}, {text_color[1]}, {text_color[2]})", "info")
            self._log(f"  Shadow: {'Enabled' if self.shadow_enabled_var.get() else 'Disabled'}", "info")
            self._log(f"  Full Page Context: {'Enabled' if self.full_page_context_var.get() else 'Disabled'}", "info")
            
            # Apply cloud inpainting settings
            saved_api_key = self.main_gui.config.get('replicate_api_key', '')
            if saved_api_key:
                self.translator.use_cloud_inpainting = True
                self.translator.replicate_api_key = saved_api_key
                self._log(f"  Cloud Inpainting: Enabled", "info")
            else:
                self.translator.use_cloud_inpainting = False
                self.translator.replicate_api_key = None
    
    def _translation_worker(self):
        """Worker thread for translation"""
        try:
            self.translator.set_stop_flag(self.stop_flag)
            
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
                
                try:
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
                    
                    # Check if translation was interrupted
                    if result.get('interrupted', False):
                        self._log(f"⏸️ Translation of {filename} was interrupted", "warning")
                        self.failed_files += 1
                        if self.stop_flag.is_set():
                            break
                    elif result['success']:
                        self.completed_files += 1
                        self._log(f"✅ Successfully translated: {filename}", "success")
                    else:
                        self.failed_files += 1
                        errors = '\n'.join(result['errors'])
                        self._log(f"❌ Failed to translate {filename}:\n{errors}", "error")
                        
                        # Check for specific error types in the error messages
                        errors_lower = errors.lower()
                        if '429' in errors or 'rate limit' in errors_lower:
                            self._log(f"⚠️ RATE LIMIT DETECTED - Please wait before continuing", "error")
                            self._log(f"   The API provider is limiting your requests", "error")
                            self._log(f"   Consider increasing delay between requests in settings", "error")
                            
                            # Optionally pause for a bit
                            self._log(f"   Pausing for 60 seconds...", "warning")
                            for sec in range(60):
                                if self.stop_flag.is_set():
                                    break
                                time.sleep(1)
                                if sec % 10 == 0:
                                    self._log(f"   Waiting... {60-sec} seconds remaining", "warning")
                    
                except Exception as e:
                    self.failed_files += 1
                    error_str = str(e)
                    error_type = type(e).__name__
                    
                    self._log(f"❌ Error processing {filename}:", "error")
                    self._log(f"   Error type: {error_type}", "error")
                    self._log(f"   Details: {error_str}", "error")
                    
                    # Check for specific API errors
                    if "429" in error_str or "rate limit" in error_str.lower():
                        self._log(f"⚠️ RATE LIMIT ERROR (429) - API is throttling requests", "error")
                        self._log(f"   Please wait before continuing or reduce request frequency", "error")
                        self._log(f"   Consider increasing the API delay in settings", "error")
                        
                        # Pause for rate limit
                        self._log(f"   Pausing for 60 seconds...", "warning")
                        for sec in range(60):
                            if self.stop_flag.is_set():
                                break
                            time.sleep(1)
                            if sec % 10 == 0:
                                self._log(f"   Waiting... {60-sec} seconds remaining", "warning")
                        
                    elif "401" in error_str or "unauthorized" in error_str.lower():
                        self._log(f"❌ AUTHENTICATION ERROR (401) - Check your API key", "error")
                        self._log(f"   The API key appears to be invalid or expired", "error")
                        
                    elif "403" in error_str or "forbidden" in error_str.lower():
                        self._log(f"❌ FORBIDDEN ERROR (403) - Access denied", "error")
                        self._log(f"   Check your API subscription and permissions", "error")
                        
                    elif "timeout" in error_str.lower():
                        self._log(f"⏱️ TIMEOUT ERROR - Request took too long", "error")
                        self._log(f"   Consider increasing timeout settings", "error")
                        
                    else:
                        # Generic error with full traceback
                        self._log(f"   Full traceback:", "error")
                        self._log(traceback.format_exc(), "error")
            
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
