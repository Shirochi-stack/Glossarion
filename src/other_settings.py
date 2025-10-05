"""Other Settings Dialog Methods for Glossarion

This module contains all the methods related to the "Other Settings" dialog.
These methods are dynamically injected into the TranslatorGUI class.
"""

# Standard library imports
import os
import json
import re
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Third-party imports
import ttkbootstrap as tb

# Local imports - these will be available through the TranslatorGUI instance
# Import UIHelper and CONFIG_FILE from translator_gui for use in the methods
from translator_gui import UIHelper, CONFIG_FILE
from ai_hunter_enhanced import AIHunterConfigGUI


def setup_other_settings_methods(gui_instance):
    """Inject all other settings methods into the GUI instance"""
    import types
    import sys
    
    # Get this module
    current_module = sys.modules[__name__]
    
    # List of all method names to bind
    methods_to_bind = [
        # Core profile methods (needed at GUI init)
        'on_profile_select', 'save_profile', 'delete_profile', 'save_profiles',
        'import_profiles', 'export_profiles',
        # Other settings methods
        'configure_rolling_summary_prompts', 'toggle_thinking_budget', 
        'toggle_gpt_reasoning_controls', 'open_other_settings',
        'open_multi_api_key_manager', 'show_ai_hunter_settings',
        'delete_translated_headers_file', 'validate_epub_structure_gui',
        'on_extraction_method_change', 'on_extraction_mode_change',
        # Toggle methods
        'toggle_extraction_workers', 'toggle_gemini_endpoint', 'toggle_ai_hunter',
        'toggle_custom_endpoint_ui', 'toggle_more_endpoints',
        '_toggle_multi_key_setting', '_toggle_http_tuning_controls',
        '_toggle_anti_duplicate_controls',
        # Section creation methods
        '_create_context_management_section', '_create_response_handling_section',
        '_create_prompt_management_section', '_create_processing_options_section',
        '_create_image_translation_section', '_create_anti_duplicate_section',
        '_create_custom_api_endpoints_section', '_create_settings_buttons',
        # Helper methods
        '_create_multi_key_row', '_create_manual_config_backup', '_manual_restore_config',
        '_check_azure_endpoint', '_update_azure_api_version_env',
        '_reset_anti_duplicate_defaults', '_get_ai_hunter_status_text',
        'create_ai_hunter_section', 'test_api_connections'
    ]
    
    # Bind each method to the GUI instance
    for method_name in methods_to_bind:
        if hasattr(current_module, method_name):
            method = getattr(current_module, method_name)
            if callable(method):
                setattr(gui_instance, method_name, types.MethodType(method, gui_instance))


def configure_rolling_summary_prompts(self):
   """Configure rolling summary prompts"""
   dialog = self.wm.create_simple_dialog(
       self.master,
       "Configure Memory System Prompts",
       width=800,
       height=1050
   )
   
   main_frame = tk.Frame(dialog, padx=20, pady=20)
   main_frame.pack(fill=tk.BOTH, expand=True)
   
   tk.Label(main_frame, text="Memory System Configuration", 
           font=('TkDefaultFont', 14, 'bold')).pack(anchor=tk.W, pady=(0, 5))
   
   tk.Label(main_frame, text="Configure how the AI creates and maintains translation memory/context summaries.",
           font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, pady=(0, 15))
   
   system_frame = tk.LabelFrame(main_frame, text="System Prompt (Role Definition)", padx=10, pady=10)
   system_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
   
   tk.Label(system_frame, text="Defines the AI's role and behavior when creating summaries",
           font=('TkDefaultFont', 9), fg='blue').pack(anchor=tk.W, pady=(0, 5))
   
   self.summary_system_text = self.ui.setup_scrollable_text(
       system_frame, height=5, wrap=tk.WORD
   )
   self.summary_system_text.pack(fill=tk.BOTH, expand=True)
   self.summary_system_text.insert('1.0', self.rolling_summary_system_prompt)
   
   user_frame = tk.LabelFrame(main_frame, text="User Prompt Template", padx=10, pady=10)
   user_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
   
   tk.Label(user_frame, text="Template for summary requests. Use {translations} for content placeholder",
           font=('TkDefaultFont', 9), fg='blue').pack(anchor=tk.W, pady=(0, 5))
   
   self.summary_user_text = self.ui.setup_scrollable_text(
       user_frame, height=12, wrap=tk.WORD
   )
   self.summary_user_text.pack(fill=tk.BOTH, expand=True)
   self.summary_user_text.insert('1.0', self.rolling_summary_user_prompt)
   
   button_frame = tk.Frame(main_frame)
   button_frame.pack(fill=tk.X, pady=(10, 0))
   
   def save_prompts():
       self.rolling_summary_system_prompt = self.summary_system_text.get('1.0', tk.END).strip()
       self.rolling_summary_user_prompt = self.summary_user_text.get('1.0', tk.END).strip()
       
       self.config['rolling_summary_system_prompt'] = self.rolling_summary_system_prompt
       self.config['rolling_summary_user_prompt'] = self.rolling_summary_user_prompt
       
       os.environ['ROLLING_SUMMARY_SYSTEM_PROMPT'] = self.rolling_summary_system_prompt
       os.environ['ROLLING_SUMMARY_USER_PROMPT'] = self.rolling_summary_user_prompt
       
       messagebox.showinfo("Success", "Memory prompts saved!")
       dialog.destroy()
   
   def reset_prompts():
       if messagebox.askyesno("Reset Prompts", "Reset memory prompts to defaults?"):
           self.summary_system_text.delete('1.0', tk.END)
           self.summary_system_text.insert('1.0', self.default_rolling_summary_system_prompt)
           self.summary_user_text.delete('1.0', tk.END)
           self.summary_user_text.insert('1.0', self.default_rolling_summary_user_prompt)
   
   tb.Button(button_frame, text="Save", command=save_prompts, 
            bootstyle="success", width=15).pack(side=tk.LEFT, padx=5)
   tb.Button(button_frame, text="Reset to Defaults", command=reset_prompts, 
            bootstyle="warning", width=15).pack(side=tk.LEFT, padx=5)
   tb.Button(button_frame, text="Cancel", command=dialog.destroy, 
            bootstyle="secondary", width=15).pack(side=tk.LEFT, padx=5)
   
   dialog.deiconify()

def toggle_thinking_budget(self):
    """Enable/disable thinking budget entry based on checkbox state"""
    if hasattr(self, 'thinking_budget_entry'):
        if self.enable_gemini_thinking_var.get():
            self.thinking_budget_entry.config(state='normal')
        else:
            self.thinking_budget_entry.config(state='disabled')

def toggle_gpt_reasoning_controls(self):
    """Enable/disable GPT reasoning controls based on toggle state"""
    enabled = self.enable_gpt_thinking_var.get()
    # Tokens entry
    if hasattr(self, 'gpt_reasoning_tokens_entry'):
        self.gpt_reasoning_tokens_entry.config(state='normal' if enabled else 'disabled')
    # Effort combo
    if hasattr(self, 'gpt_effort_combo'):
        try:
            self.gpt_effort_combo.config(state='readonly' if enabled else 'disabled')
        except Exception:
            # Fallback for ttk on some platforms
            self.gpt_effort_combo.configure(state='readonly' if enabled else 'disabled')

def open_other_settings(self):
    """Open the Other Settings dialog"""
    dialog, scrollable_frame, canvas = self.wm.setup_scrollable(
        self.master,
        "Other Settings",
        width=0,
        height=None,
        max_width_ratio=0.7,
        max_height_ratio=0.8
    )
    
    scrollable_frame.grid_columnconfigure(0, weight=1, uniform="column")
    scrollable_frame.grid_columnconfigure(1, weight=1, uniform="column")
    
    # Section 1: Context Management
    self._create_context_management_section(scrollable_frame)
    
    # Section 2: Response Handling
    self._create_response_handling_section(scrollable_frame)
    
    # Section 3: Prompt Management
    self._create_prompt_management_section(scrollable_frame)
    
    # Section 4: Processing Options
    self._create_processing_options_section(scrollable_frame)
    
    # Section 5: Image Translation
    self._create_image_translation_section(scrollable_frame)
    
    # Section 6: Anti-Duplicate Parameters
    self._create_anti_duplicate_section(scrollable_frame)
    
    # Section 7: Custom API Endpoints (NEW)
    self._create_custom_api_endpoints_section(scrollable_frame)
    
    # Save & Close buttons
    self._create_settings_buttons(scrollable_frame, dialog, canvas)
    
    # Persist toggle change on dialog close
    def _persist_settings():
        self.config['retain_source_extension'] = self.retain_source_extension_var.get()
        os.environ['RETAIN_SOURCE_EXTENSION'] = '1' if self.retain_source_extension_var.get() else '0'
        # Save without user-facing message when closing Other Settings
        self.save_config(show_message=False)
        dialog._cleanup_scrolling()
        dialog.destroy()
    dialog.protocol("WM_DELETE_WINDOW", _persist_settings)
    
    # Auto-resize and show
    self.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.78, max_height_ratio=1.82)

def _create_context_management_section(self, parent):
    """Create context management section"""
    section_frame = tk.LabelFrame(parent, text="Context Management & Memory", padx=10, pady=10)
    section_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=(10, 5))
       
    content_frame = tk.Frame(section_frame)
    content_frame.pack(anchor=tk.NW, fill=tk.BOTH, expand=True)

    tb.Checkbutton(content_frame, text="Use Rolling Summary (Memory)", 
                 variable=self.rolling_summary_var,
                 bootstyle="round-toggle").pack(anchor=tk.W)

    tk.Label(content_frame, text="AI-powered memory system that maintains story context",
           font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))

    settings_frame = tk.Frame(content_frame)
    settings_frame.pack(anchor=tk.W, padx=20, fill=tk.X, pady=(5, 10))

    row1 = tk.Frame(settings_frame)
    row1.pack(fill=tk.X, pady=(0, 10))

    tk.Label(row1, text="Role:").pack(side=tk.LEFT, padx=(0, 5))
    role_combo = ttk.Combobox(row1, textvariable=self.summary_role_var,
               values=["user", "system"], state="readonly", width=10)
    role_combo.pack(side=tk.LEFT, padx=(0, 30))
    # Prevent accidental changes from mouse wheel while scrolling
    UIHelper.disable_spinbox_mousewheel(role_combo)

    tk.Label(row1, text="Mode:").pack(side=tk.LEFT, padx=(0, 5))
    mode_combo = ttk.Combobox(row1, textvariable=self.rolling_summary_mode_var,
               values=["append", "replace"], state="readonly", width=10)
    mode_combo.pack(side=tk.LEFT, padx=(0, 10))
    # Prevent accidental changes from mouse wheel while scrolling
    UIHelper.disable_spinbox_mousewheel(mode_combo)

    row2 = tk.Frame(settings_frame)
    row2.pack(fill=tk.X, pady=(0, 10))

    tk.Label(row2, text="Summarize last").pack(side=tk.LEFT, padx=(0, 5))
    tb.Entry(row2, width=5, textvariable=self.rolling_summary_exchanges_var).pack(side=tk.LEFT, padx=(0, 5))
    tk.Label(row2, text="exchanges").pack(side=tk.LEFT)

    # Spacer
    tk.Label(row2, text="   ").pack(side=tk.LEFT)
    # New controls: Retain last N summaries (append mode)
    tk.Label(row2, text="Retain").pack(side=tk.LEFT, padx=(10, 5))
    tb.Entry(row2, width=5, textvariable=self.rolling_summary_max_entries_var).pack(side=tk.LEFT, padx=(0, 5))
    tk.Label(row2, text="entries").pack(side=tk.LEFT)

    tb.Button(content_frame, text="⚙️ Configure Memory Prompts", 
            command=self.configure_rolling_summary_prompts,
            bootstyle="info-outline", width=30).pack(anchor=tk.W, padx=20, pady=(10, 10))

    ttk.Separator(section_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 10))
    
    tk.Label(section_frame, text="💡 Memory Mode:\n"
           "• Append: Keeps adding summaries (longer context)\n"
           "• Replace: Only keeps latest summary (concise)",
           font=('TkDefaultFont', 11), fg='#666', justify=tk.LEFT).pack(anchor=tk.W, padx=5, pady=(0, 5))       

    ttk.Separator(section_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 10))
           
    
    tk.Label(section_frame, text="Application Updates:", font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W, pady=(5, 5))
    
    # Create a frame for update-related controls
    update_frame = tk.Frame(section_frame)
    update_frame.pack(anchor=tk.W, fill=tk.X)

    tb.Button(update_frame, text="🔄 Check for Updates", 
             command=lambda: self.check_for_updates_manual(), 
             bootstyle="info-outline",
             width=25).pack(side=tk.LEFT, pady=2)

    # Add auto-update checkbox
    tb.Checkbutton(update_frame, text="Check on startup", 
                 variable=self.auto_update_check_var, 
                 bootstyle="round-toggle").pack(side=tk.LEFT, padx=(10, 0))

    tk.Label(section_frame, text="Check GitHub for new Glossarion releases\nand download updates",
            font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 5))
                 
    ttk.Separator(section_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 10))
    
    tk.Label(section_frame, text="Config Backup Management:", font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W, pady=(5, 5))
    
    # Create a frame for backup-related controls
    backup_frame = tk.Frame(section_frame)
    backup_frame.pack(anchor=tk.W, fill=tk.X)

    tb.Button(backup_frame, text="💾 Create Backup", 
             command=lambda: self._create_manual_config_backup(), 
             bootstyle="success-outline",
             width=20).pack(side=tk.LEFT, pady=2, padx=(0, 10))
             
    tb.Button(backup_frame, text="↶ Restore Backup", 
             command=lambda: self._manual_restore_config(), 
             bootstyle="warning-outline",
             width=20).pack(side=tk.LEFT, pady=2)

    tk.Label(section_frame, text="Automatic backups are created before each config save.",
           font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=5, pady=(5, 0))

def _create_response_handling_section(self, parent):
    """Create response handling section with AI Hunter additions"""
    section_frame = tk.LabelFrame(parent, text="Response Handling & Retry Logic", padx=10, pady=10)
    section_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=5)
    
    # GPT-5/OpenAI Reasoning Toggle (NEW)
    tk.Label(section_frame, text="GPT-5 Thinking (OpenRouter/OpenAI-style)", 
            font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W)

    gpt_frame = tk.Frame(section_frame)
    gpt_frame.pack(anchor=tk.W, padx=20, pady=(5, 0))

    tb.Checkbutton(gpt_frame, text="Enable GPT / OR Thinking", 
                  variable=self.enable_gpt_thinking_var,
                  bootstyle="round-toggle",
                  command=self.toggle_gpt_reasoning_controls).pack(side=tk.LEFT)

    tk.Label(gpt_frame, text="Effort:").pack(side=tk.LEFT, padx=(20, 5))
    self.gpt_effort_combo = ttk.Combobox(gpt_frame, textvariable=self.gpt_effort_var,
                                         values=["low", "medium", "high"], state="readonly", width=8)
    self.gpt_effort_combo.pack(side=tk.LEFT, padx=5)
    UIHelper.disable_spinbox_mousewheel(self.gpt_effort_combo)

    # Second row for OpenRouter-specific token budget
    gpt_row2 = tk.Frame(section_frame)
    gpt_row2.pack(anchor=tk.W, padx=40, pady=(5, 0))
    tk.Label(gpt_row2, text="OR Thinking Tokens:").pack(side=tk.LEFT)
    self.gpt_reasoning_tokens_entry = tb.Entry(gpt_row2, width=8, textvariable=self.gpt_reasoning_tokens_var)
    self.gpt_reasoning_tokens_entry.pack(side=tk.LEFT, padx=5)
    tk.Label(gpt_row2, text="tokens").pack(side=tk.LEFT)

    # Initialize enabled state for GPT controls
    self.toggle_gpt_reasoning_controls()

    tk.Label(section_frame, text="Controls GPT-5 and OpenRouter reasoning. \nProvide Tokens to force a max token budget for other models; GPT-5 only uses Effort (low/medium/high).",
           font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))

    # Add Thinking Tokens Toggle with Budget Control (NEW)
    tk.Label(section_frame, text="Gemini Thinking Mode", 
            font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W)
    
    thinking_frame = tk.Frame(section_frame)
    thinking_frame.pack(anchor=tk.W, padx=20, pady=(5, 0))
    
    tb.Checkbutton(thinking_frame, text="Enable Gemini Thinking", 
                  variable=self.enable_gemini_thinking_var,
                  bootstyle="round-toggle",
                  command=self.toggle_thinking_budget).pack(side=tk.LEFT)
    
    tk.Label(thinking_frame, text="Budget:").pack(side=tk.LEFT, padx=(20, 5))
    self.thinking_budget_entry = tb.Entry(thinking_frame, width=8, textvariable=self.thinking_budget_var)
    self.thinking_budget_entry.pack(side=tk.LEFT, padx=5)
    tk.Label(thinking_frame, text="tokens").pack(side=tk.LEFT)
    
    tk.Label(section_frame, text="Control Gemini's thinking process. 0 = disabled,\n512-24576 = limited thinking, -1 = dynamic (auto)",
           font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
    
    # Add separator after thinking toggle
    ttk.Separator(section_frame, orient='horizontal').pack(fill='x', pady=10)
    
    # ADD EXTRACTION WORKERS CONFIGURATION HERE
    tk.Label(section_frame, text="Parallel Extraction", 
            font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W)
    
    extraction_frame = tk.Frame(section_frame)
    extraction_frame.pack(anchor=tk.W, padx=20, pady=(5, 0))
    
    tb.Checkbutton(extraction_frame, text="Enable Parallel Processing", 
                  variable=self.enable_parallel_extraction_var,
                  bootstyle="round-toggle",
                  command=self.toggle_extraction_workers).pack(side=tk.LEFT)
    
    tk.Label(extraction_frame, text="Workers:").pack(side=tk.LEFT, padx=(20, 5))
    self.extraction_workers_entry = tb.Entry(extraction_frame, width=6, textvariable=self.extraction_workers_var)
    self.extraction_workers_entry.pack(side=tk.LEFT, padx=5)
    tk.Label(extraction_frame, text="threads").pack(side=tk.LEFT)
    
    tk.Label(section_frame, text="Speed up EPUB extraction using multiple threads.\nRecommended: 4-8 workers (set to 1 to disable)",
           font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
    
    # Add separator after extraction workers
    ttk.Separator(section_frame, orient='horizontal').pack(fill='x', pady=10)
    
    # Multi API Key Management Section
    multi_key_frame = tk.Frame(section_frame)
    multi_key_frame.pack(anchor=tk.W, fill=tk.X, pady=(0, 15))
    
    # Multi-key indicator and button in same row
    multi_key_row = tk.Frame(multi_key_frame)
    multi_key_row.pack(fill=tk.X)
    
    # Show status if multi-key is enabled
    if self.config.get('use_multi_api_keys', False):
        multi_keys = self.config.get('multi_api_keys', [])
        active_keys = sum(1 for k in multi_keys if k.get('enabled', True))
        
        status_frame = tk.Frame(multi_key_row)
        status_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(status_frame, text="🔑 Multi-Key Mode:", 
                font=('TkDefaultFont', 11, 'bold')).pack(side=tk.LEFT)
        
        tk.Label(status_frame, text=f"ACTIVE ({active_keys}/{len(multi_keys)} keys)", 
                font=('TkDefaultFont', 11, 'bold'), fg='green').pack(side=tk.LEFT, padx=(5, 0))
    else:
        tk.Label(multi_key_row, text="🔑 Multi-Key Mode: DISABLED", 
                font=('TkDefaultFont', 11), fg='gray').pack(side=tk.LEFT)
    
    # Multi API Key Manager button
    tb.Button(multi_key_row, text="Configure API Keys", 
              command=self.open_multi_api_key_manager,
              bootstyle="primary-outline",
              width=20).pack(side=tk.RIGHT)
    
    tk.Label(section_frame, text="Manage multiple API keys with automatic rotation and rate limit handling",
             font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
    
    # Add separator after Multi API Key section
    ttk.Separator(section_frame, orient='horizontal').pack(fill='x', pady=10)
 
    # Retry Truncated
    tb.Checkbutton(section_frame, text="Auto-retry Truncated Responses", 
                      variable=self.retry_truncated_var,
                      bootstyle="round-toggle").pack(anchor=tk.W)
    retry_frame = tk.Frame(section_frame)
    retry_frame.pack(anchor=tk.W, padx=20, pady=(5, 5))
    tk.Label(retry_frame, text="Token constraint:").pack(side=tk.LEFT)
    tb.Entry(retry_frame, width=8, textvariable=self.max_retry_tokens_var).pack(side=tk.LEFT, padx=5)
    tk.Label(section_frame, text="Retry when truncated. Acts as min/max constraint:\nbelow value = minimum, above value = maximum",
               font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
    # Compression Factor
    # Add separator line for clarity
    ttk.Separator(section_frame, orient='horizontal').pack(fill='x', pady=10)
    
    # Compression Factor
    tk.Label(section_frame, text="Translation Compression Factor", 
                font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W)
    
    compression_frame = tk.Frame(section_frame)
    compression_frame.pack(anchor=tk.W, padx=20, pady=(5, 0))
    tk.Label(compression_frame, text="CJK→English compression:").pack(side=tk.LEFT)
    tb.Entry(compression_frame, width=6, textvariable=self.compression_factor_var).pack(side=tk.LEFT, padx=5)
    tk.Label(compression_frame, text="(0.7-1.0)", font=('TkDefaultFont', 11)).pack(side=tk.LEFT)
    
    # TODO: Implement configure_translation_chunk_prompt method
    # tb.Button(compression_frame, text=" Chunk Prompt", 
    #              command=self.configure_translation_chunk_prompt,
    #              bootstyle="info-outline", width=15).pack(side=tk.LEFT, padx=(15, 0))
    tk.Label(section_frame, text="Ratio for chunk sizing based on output limits\n",
               font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
   
    # Add separator after compression factor
    ttk.Separator(section_frame, orient='horizontal').pack(fill='x', pady=10)
    
    # Retry Duplicate
    tb.Checkbutton(section_frame, text="Auto-retry Duplicate Content", 
                     variable=self.retry_duplicate_var,
                     bootstyle="round-toggle").pack(anchor=tk.W)
    duplicate_frame = tk.Frame(section_frame)
    duplicate_frame.pack(anchor=tk.W, padx=20, pady=(5, 0))
    tk.Label(duplicate_frame, text="Check last").pack(side=tk.LEFT)
    tb.Entry(duplicate_frame, width=4, textvariable=self.duplicate_lookback_var).pack(side=tk.LEFT, padx=3)
    tk.Label(duplicate_frame, text="chapters").pack(side=tk.LEFT)
    tk.Label(section_frame, text="Detects when AI returns same content\nfor different chapters",
               font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(5, 10))
    # Container for detection-related options (to show/hide based on toggle)
    self.detection_options_container = tk.Frame(section_frame)
    
    # Update thinking budget entry state based on initial toggle state
    self.toggle_thinking_budget()

    # Function to show/hide detection options based on auto-retry toggle
    def update_detection_visibility():
        try:
            # Check if widgets still exist before manipulating them
            if (hasattr(self, 'detection_options_container') and 
                self.detection_options_container.winfo_exists() and
                duplicate_frame.winfo_exists()):
                
                if self.retry_duplicate_var.get():
                    self.detection_options_container.pack(fill='x', after=duplicate_frame)
                else:
                    self.detection_options_container.pack_forget()
        except tk.TclError:
            # Widget has been destroyed, ignore
            pass

    # Add trace to update visibility when toggle changes
    self.retry_duplicate_var.trace('w', lambda *args: update_detection_visibility())

    # Detection Method subsection (now inside the container)
    method_label = tk.Label(self.detection_options_container, text="Detection Method:", 
                           font=('TkDefaultFont', 10, 'bold'))
    method_label.pack(anchor=tk.W, padx=20, pady=(10, 5))

    methods = [
       ("basic", "Basic (Fast) - Original 85% threshold, 1000 chars"),
       ("ai-hunter", "AI Hunter - Multi-method semantic analysis"),
       ("cascading", "Cascading - Basic first, then AI Hunter")
    ]

    # Container for AI Hunter config (will be shown/hidden based on selection)
    self.ai_hunter_container = tk.Frame(self.detection_options_container)

    # Function to update AI Hunter visibility based on detection mode
    def update_ai_hunter_visibility(*args):
        """Update AI Hunter section visibility based on selection"""
        # Clear existing widgets
        for widget in self.ai_hunter_container.winfo_children():
            widget.destroy()
        
        # Show AI Hunter config for both ai-hunter and cascading modes
        if self.duplicate_detection_mode_var.get() in ['ai-hunter', 'cascading']:
            self.create_ai_hunter_section(self.ai_hunter_container)
        
        # Update status if label exists and hasn't been destroyed
        if hasattr(self, 'ai_hunter_status_label'):
            try:
                # Check if the widget still exists before updating
                self.ai_hunter_status_label.winfo_exists()
                self.ai_hunter_status_label.config(text=self._get_ai_hunter_status_text())
            except tk.TclError:
                # Widget has been destroyed, remove the reference
                delattr(self, 'ai_hunter_status_label')

    # Create radio buttons (inside detection container) - ONLY ONCE
    for value, text in methods:
       rb = tb.Radiobutton(self.detection_options_container, text=text, 
                          variable=self.duplicate_detection_mode_var, 
                          value=value, bootstyle="primary")
       rb.pack(anchor=tk.W, padx=40, pady=2)

    # Pack the AI Hunter container
    self.ai_hunter_container.pack(fill='x')

    # Add trace to detection mode variable - ONLY ONCE
    self.duplicate_detection_mode_var.trace('w', update_ai_hunter_visibility)

    # Initial visibility updates
    update_detection_visibility()
    update_ai_hunter_visibility()
    
    # Retry Slow
    tb.Checkbutton(section_frame, text="Auto-retry Slow Chunks", 
                  variable=self.retry_timeout_var,
                  bootstyle="round-toggle").pack(anchor=tk.W, pady=(15, 0))

    timeout_frame = tk.Frame(section_frame)
    timeout_frame.pack(anchor=tk.W, padx=20, pady=(5, 0))
    tk.Label(timeout_frame, text="Timeout after").pack(side=tk.LEFT)
    tb.Entry(timeout_frame, width=6, textvariable=self.chunk_timeout_var).pack(side=tk.LEFT, padx=5)
    tk.Label(timeout_frame, text="seconds").pack(side=tk.LEFT)

    tk.Label(section_frame, text="Retry chunks/images that take too long\n(reduces tokens for faster response)",
            font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 5))

    # Separator
    ttk.Separator(section_frame, orient='horizontal').pack(fill='x', pady=10)

    # HTTP Timeouts & Connection Pooling
    title_http = tk.Label(section_frame, text="HTTP Timeouts & Connection Pooling", 
                          font=('TkDefaultFont', 11, 'bold'))
    title_http.pack(anchor=tk.W)

    http_frame = tk.Frame(section_frame)
    http_frame.pack(anchor=tk.W, padx=20, pady=(5, 0), fill=tk.X)

    # Master toggle to enable/disable all HTTP tuning fields (disabled by default)
    if not hasattr(self, 'enable_http_tuning_var'):
        self.enable_http_tuning_var = tk.BooleanVar(value=self.config.get('enable_http_tuning', False))
    self.http_tuning_checkbox = tb.Checkbutton(
        http_frame,
        text="Enable HTTP timeout/pooling overrides",
        variable=self.enable_http_tuning_var,
        command=getattr(self, '_toggle_http_tuning_controls', None) or (lambda: None),
        bootstyle="round-toggle"
    )
    self.http_tuning_checkbox.pack(anchor=tk.W, pady=(0, 6))

    # Build a compact grid so fields align nicely
    http_grid = tk.Frame(http_frame)
    http_grid.pack(anchor=tk.W, fill=tk.X)

    if not hasattr(self, 'connect_timeout_var'):
        self.connect_timeout_var = tk.StringVar(value=str(self.config.get('connect_timeout', os.environ.get('CONNECT_TIMEOUT', '10'))))
    if not hasattr(self, 'read_timeout_var'):
        # Default to READ_TIMEOUT, fallback to CHUNK_TIMEOUT if provided, else 180
        self.read_timeout_var = tk.StringVar(value=str(self.config.get('read_timeout', os.environ.get('READ_TIMEOUT', os.environ.get('CHUNK_TIMEOUT', '180')))))
    if not hasattr(self, 'http_pool_connections_var'):
        self.http_pool_connections_var = tk.StringVar(value=str(self.config.get('http_pool_connections', os.environ.get('HTTP_POOL_CONNECTIONS', '20'))))
    if not hasattr(self, 'http_pool_maxsize_var'):
        self.http_pool_maxsize_var = tk.StringVar(value=str(self.config.get('http_pool_maxsize', os.environ.get('HTTP_POOL_MAXSIZE', '50'))))

    # Layout columns
    http_grid.grid_columnconfigure(0, weight=0)
    http_grid.grid_columnconfigure(1, weight=0)
    http_grid.grid_columnconfigure(2, weight=1)  # spacer
    http_grid.grid_columnconfigure(3, weight=0)
    http_grid.grid_columnconfigure(4, weight=0)

    # Optional toggle: ignore server Retry-After header
    if not hasattr(self, 'ignore_retry_after_var'):
        self.ignore_retry_after_var = tk.BooleanVar(value=bool(self.config.get('ignore_retry_after', str(os.environ.get('IGNORE_RETRY_AFTER', '0')) == '1')))
    self.ignore_retry_after_checkbox = tb.Checkbutton(
        http_frame,
        text="Ignore server Retry-After header (use local backoff)", 
        variable=self.ignore_retry_after_var,
        bootstyle="round-toggle"
    )
    self.ignore_retry_after_checkbox.pack(anchor=tk.W, pady=(6, 0))

    # Row 0: Timeouts
    tk.Label(http_grid, text="Connect timeout (s):").grid(row=0, column=0, sticky='w', padx=(0, 6), pady=2)
    self.connect_timeout_entry = tb.Entry(http_grid, width=6, textvariable=self.connect_timeout_var)
    self.connect_timeout_entry.grid(row=0, column=1, sticky='w', pady=2)
    tk.Label(http_grid, text="Read timeout (s):").grid(row=0, column=3, sticky='w', padx=(12, 6), pady=2)
    self.read_timeout_entry = tb.Entry(http_grid, width=6, textvariable=self.read_timeout_var)
    self.read_timeout_entry.grid(row=0, column=4, sticky='w', pady=2)

    # Row 1: Pool sizes
    tk.Label(http_grid, text="Pool connections:").grid(row=1, column=0, sticky='w', padx=(0, 6), pady=2)
    self.http_pool_connections_entry = tb.Entry(http_grid, width=6, textvariable=self.http_pool_connections_var)
    self.http_pool_connections_entry.grid(row=1, column=1, sticky='w', pady=2)
    tk.Label(http_grid, text="Pool max size:").grid(row=1, column=3, sticky='w', padx=(12, 6), pady=2)
    self.http_pool_maxsize_entry = tb.Entry(http_grid, width=6, textvariable=self.http_pool_maxsize_var)
    self.http_pool_maxsize_entry.grid(row=1, column=4, sticky='w', pady=2)

    # Apply initial enable/disable state
    if hasattr(self, '_toggle_http_tuning_controls'):
        self._toggle_http_tuning_controls()

    tk.Label(section_frame, text="Controls network behavior to reduce 500/503s: connection establishment timeout, read timeout,\nHTTP connection pool sizes.",
            font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(2, 5))
    
    # Separator
    ttk.Separator(section_frame, orient='horizontal').pack(fill='x', pady=10)
    
    # Max Retries Configuration
    title_retries = tk.Label(section_frame, text="API Request Retries", 
                            font=('TkDefaultFont', 11, 'bold'))
    title_retries.pack(anchor=tk.W)
    
    retries_frame = tk.Frame(section_frame)
    retries_frame.pack(anchor=tk.W, padx=20, pady=(5, 0))
    
    # Create MAX_RETRIES variable if it doesn't exist
    if not hasattr(self, 'max_retries_var'):
        self.max_retries_var = tk.StringVar(value=str(self.config.get('max_retries', os.environ.get('MAX_RETRIES', '7'))))
    
    tk.Label(retries_frame, text="Maximum retry attempts:").pack(side=tk.LEFT)
    tb.Entry(retries_frame, width=4, textvariable=self.max_retries_var).pack(side=tk.LEFT, padx=5)
    tk.Label(retries_frame, text="(default: 7)").pack(side=tk.LEFT)
    
    tk.Label(section_frame, text="Number of times to retry failed API requests before giving up.\nApplies to all API providers (OpenAI, Gemini, Anthropic, etc.)",
            font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(2, 10))
    
    # Enable/disable combobox based on toggle
    def _toggle_scan_mode_state(*args):
        try:
            if self.scan_phase_enabled_var.get():
                scan_mode_combo.config(state="readonly")
            else:
                scan_mode_combo.config(state="disabled")
        except Exception:
            pass
    _toggle_scan_mode_state()
    self.scan_phase_enabled_var.trace('w', lambda *a: _toggle_scan_mode_state())
    
    # Indefinite Rate Limit Retry toggle
    tb.Checkbutton(section_frame, text="Indefinite Rate Limit Retry", 
                  variable=self.indefinite_rate_limit_retry_var,
                  bootstyle="round-toggle").pack(anchor=tk.W, padx=20)
    
    tk.Label(section_frame, text="When enabled, rate limit errors (429) will retry indefinitely with exponential backoff.\nWhen disabled, rate limits count against the maximum retry attempts above.",
            font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=40, pady=(2, 5))
    
            
def toggle_gemini_endpoint(self):
    """Enable/disable Gemini endpoint entry based on toggle"""
    if self.use_gemini_openai_endpoint_var.get():
        self.gemini_endpoint_entry.config(state='normal')
    else:
        self.gemini_endpoint_entry.config(state='disabled')

def open_multi_api_key_manager(self):
    """Open the multi API key manager dialog"""
    # Import here to avoid circular imports
    try:
        from multi_api_key_manager import MultiAPIKeyDialog
        
        # Create and show dialog
        dialog = MultiAPIKeyDialog(self.master, self)
        
        # Wait for dialog to close
        self.master.wait_window(dialog.dialog)
        
        # Refresh the settings display if in settings dialog
        if hasattr(self, 'current_settings_dialog'):
            # Close and reopen settings to refresh
            self.current_settings_dialog.destroy()
            self.show_settings()  # or open_other_settings()
            
    except ImportError as e:
        messagebox.showerror("Error", f"Failed to load Multi API Key Manager: {str(e)}")
    except Exception as e:
        messagebox.showerror("Error", f"Error opening Multi API Key Manager: {str(e)}")
        import traceback
        traceback.print_exc()

def _create_multi_key_row(self, parent):
    """Create a compact multi-key configuration row"""
    frame = tk.Frame(parent)
    frame.pack(fill=tk.X, pady=5)
    
    # Status indicator
    if self.config.get('use_multi_api_keys', False):
        keys = self.config.get('multi_api_keys', [])
        active = sum(1 for k in keys if k.get('enabled', True))
        
        # Checkbox to enable/disable
        tb.Checkbutton(frame, text="Multi API Key Mode", 
                      variable=self.use_multi_api_keys_var,
                      bootstyle="round-toggle",
                      command=self._toggle_multi_key_setting).pack(side=tk.LEFT)
        
        # Status
        tk.Label(frame, text=f"({active}/{len(keys)} active)", 
                font=('TkDefaultFont', 10), fg='green').pack(side=tk.LEFT, padx=(5, 0))
    else:
        tb.Checkbutton(frame, text="Multi API Key Mode", 
                      variable=self.use_multi_api_keys_var,
                      bootstyle="round-toggle",
                      command=self._toggle_multi_key_setting).pack(side=tk.LEFT)
    
    # Configure button
    tb.Button(frame, text="Configure Keys...", 
              command=self.open_multi_api_key_manager,
              bootstyle="primary-outline").pack(side=tk.LEFT, padx=(20, 0))
    
    return frame
            
def _toggle_multi_key_setting(self):
    """Toggle multi-key mode from settings dialog"""
    self.config['use_multi_api_keys'] = self.use_multi_api_keys_var.get()
    # Don't save immediately, let the dialog's save button handle it

def toggle_extraction_workers(self):
    """Enable/disable extraction workers entry based on toggle"""
    if self.enable_parallel_extraction_var.get():
        self.extraction_workers_entry.config(state='normal')
        # Set environment variable
        os.environ["EXTRACTION_WORKERS"] = str(self.extraction_workers_var.get())
    else:
        self.extraction_workers_entry.config(state='disabled')
        # Set to 1 worker (sequential) when disabled
        os.environ["EXTRACTION_WORKERS"] = "1"
    
    # Ensure executor reflects current worker setting
    try:
        self._ensure_executor()
    except Exception:
        pass
    
def create_ai_hunter_section(self, parent_frame):
    """Create the AI Hunter configuration section - without redundant toggle"""
    # AI Hunter Configuration
    config_frame = tk.Frame(parent_frame)
    config_frame.pack(anchor=tk.W, padx=20, pady=(10, 5))
    
    # Status label
    ai_config = self.config.get('ai_hunter_config', {})
    self.ai_hunter_status_label = tk.Label(
        config_frame, 
        text=self._get_ai_hunter_status_text(),
        font=('TkDefaultFont', 10)
    )
    self.ai_hunter_status_label.pack(side=tk.LEFT)
    
    # Configure button
    tb.Button(
        config_frame, 
        text="Configure AI Hunter", 
        command=self.show_ai_hunter_settings,
        bootstyle="info"
    ).pack(side=tk.LEFT, padx=(10, 0))
    
    # Info text
    tk.Label(
        parent_frame,  # Use parent_frame instead of section_frame
        text="AI Hunter uses multiple detection methods to identify duplicate content\n"
             "with configurable thresholds and detection modes",
        font=('TkDefaultFont', 10), 
        fg='gray', 
        justify=tk.LEFT
    ).pack(anchor=tk.W, padx=20, pady=(0, 10))

def _get_ai_hunter_status_text(self):
    """Get status text for AI Hunter configuration"""
    ai_config = self.config.get('ai_hunter_config', {})
    
    # AI Hunter is shown when the detection mode is set to 'ai-hunter' or 'cascading'
    if self.duplicate_detection_mode_var.get() not in ['ai-hunter', 'cascading']:
        return "AI Hunter: Not Selected"
    
    if not ai_config.get('enabled', True):
        return "AI Hunter: Disabled in Config"
    
    mode_text = {
        'single_method': 'Single Method',
        'multi_method': 'Multi-Method',
        'weighted_average': 'Weighted Average'
    }
    
    mode = mode_text.get(ai_config.get('detection_mode', 'multi_method'), 'Unknown')
    thresholds = ai_config.get('thresholds', {})
    
    if thresholds:
        avg_threshold = sum(thresholds.values()) / len(thresholds)
    else:
        avg_threshold = 85
    
    return f"AI Hunter: {mode} mode, Avg threshold: {int(avg_threshold)}%"

def show_ai_hunter_settings(self):
    """Open AI Hunter configuration window"""
    def on_config_saved():
        # Save the entire configuration
        self.save_config()
        # Update status label if it still exists
        if hasattr(self, 'ai_hunter_status_label'):
            try:
                self.ai_hunter_status_label.winfo_exists()
                self.ai_hunter_status_label.config(text=self._get_ai_hunter_status_text())
            except tk.TclError:
                # Widget has been destroyed
                pass
        if hasattr(self, 'ai_hunter_enabled_var'):
            self.ai_hunter_enabled_var.set(self.config.get('ai_hunter_config', {}).get('enabled', True))
    
    gui = AIHunterConfigGUI(self.master, self.config, on_config_saved)
    gui.show_ai_hunter_config()

def toggle_ai_hunter(self):
    """Toggle AI Hunter enabled state"""
    if 'ai_hunter_config' not in self.config:
        self.config['ai_hunter_config'] = {}
    
    self.config['ai_hunter_config']['enabled'] = self.ai_hunter_enabled_var.get()
    self.save_config()
    self.ai_hunter_status_label.config(text=self._get_ai_hunter_status_text())

def _create_prompt_management_section(self, parent):
    """Create meta data section (formerly prompt management)"""
    section_frame = tk.LabelFrame(parent, text="Meta Data", padx=10, pady=10)
    section_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=(10, 5))
    
    title_frame = tk.Frame(section_frame)
    title_frame.pack(anchor=tk.W, pady=(10, 10))
    
    tb.Checkbutton(title_frame, text="Translate Book Title", 
                  variable=self.translate_book_title_var,
                  bootstyle="round-toggle").pack(side=tk.LEFT)
    
    # CHANGED: New button text and command
    tb.Button(title_frame, text="Configure All", 
             command=self.metadata_batch_ui.configure_translation_prompts,
             bootstyle="info-outline", width=12).pack(side=tk.LEFT, padx=(10, 5))
    
    # NEW: Custom Metadata Fields button
    tb.Button(title_frame, text="Custom Metadata", 
             command=self.metadata_batch_ui.configure_metadata_fields,
             bootstyle="info-outline", width=15).pack(side=tk.LEFT, padx=(5, 0))
    
    tk.Label(section_frame, text="When enabled: Book titles and selected metadata will be translated",
                font=('TkDefaultFont', 11), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
    
    # NEW: Batch Header Translation Section
    ttk.Separator(section_frame, orient='horizontal').pack(fill=tk.X, pady=(5, 10))
    
    tk.Label(section_frame, text="Chapter Header Translation:", 
            font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W, pady=(5, 5))
    
    header_frame = tk.Frame(section_frame)
    header_frame.pack(anchor=tk.W, fill=tk.X, pady=(5, 10))
    
    # Master toggle for batch header translation
    def _toggle_header_controls():
        enabled = bool(self.batch_translate_headers_var.get())
        new_state = tk.NORMAL if enabled else tk.DISABLED
        update_cb.configure(state=new_state)
        save_cb.configure(state=new_state)
        ignore_header_cb.configure(state=new_state)
        ignore_title_cb.configure(state=new_state)
        delete_btn.configure(state=new_state)
    
    batch_toggle = tb.Checkbutton(header_frame, text="Batch Translate Headers", 
                  variable=self.batch_translate_headers_var,
                  bootstyle="round-toggle",
                  command=_toggle_header_controls)
    batch_toggle.pack(side=tk.LEFT)
    
    tk.Label(header_frame, text="Headers per batch:").pack(side=tk.LEFT, padx=(20, 5))
    
    batch_entry = tk.Entry(header_frame, textvariable=self.headers_per_batch_var, width=10)
    batch_entry.pack(side=tk.LEFT)
    
    # Options for header translation
    update_frame = tk.Frame(section_frame)
    update_frame.pack(anchor=tk.W, fill=tk.X, padx=20)
    
    update_cb = tb.Checkbutton(update_frame, text="Update headers in HTML files", 
                  variable=self.update_html_headers_var,
                  bootstyle="round-toggle")
    update_cb.pack(side=tk.LEFT)
    
    save_cb = tb.Checkbutton(update_frame, text="Save translations to .txt", 
                  variable=self.save_header_translations_var,
                  bootstyle="round-toggle")
    save_cb.pack(side=tk.LEFT, padx=(20, 0))
    
    # Additional ignore header option
    ignore_frame = tk.Frame(section_frame)
    ignore_frame.pack(anchor=tk.W, fill=tk.X, padx=20, pady=(5, 0))
    
    ignore_header_cb = tb.Checkbutton(ignore_frame, text="Ignore header", 
                  variable=self.ignore_header_var,
                  bootstyle="round-toggle")
    ignore_header_cb.pack(side=tk.LEFT)
    
    ignore_title_cb = tb.Checkbutton(ignore_frame, text="Ignore title", 
                  variable=self.ignore_title_var,
                  bootstyle="round-toggle")
    ignore_title_cb.pack(side=tk.LEFT, padx=(15, 0))
    
    # Delete translated_headers.txt button
    delete_btn = tb.Button(ignore_frame, text="🗑️Delete Header Files", 
             command=self.delete_translated_headers_file,
             bootstyle="danger-outline", width=21)
    delete_btn.pack(side=tk.LEFT, padx=(20, 0))
    
    # Initialize disabled state when batch headers is OFF
    _toggle_header_controls()
    
    tk.Label(section_frame, 
            text="• OFF: Use existing headers from translated chapters\n"
                 "• ON: Extract all headers → Translate in batch → Update files\n"
                 "• Ignore header: Skip h1/h2/h3 tags (prevents re-translation of visible headers)\n"
                 "• Ignore title: Skip <title> tag (prevents re-translation of document titles)\n"
                 "• Delete button: Removes translated_headers.txt files for all selected EPUBs",
            font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(5, 10))
    
    # EPUB Validation (keep existing)
    ttk.Separator(section_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 10))
    
    tk.Label(section_frame, text="EPUB Utilities:", font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W, pady=(5, 5))
    
    tb.Button(section_frame, text="🔍 Validate EPUB Structure", 
             command=self.validate_epub_structure_gui, 
             bootstyle="success-outline",
             width=25).pack(anchor=tk.W, pady=2)
    
    tk.Label(section_frame, text="Check if all required EPUB files are present for compilation",
            font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 5))
    
    # NCX-only navigation toggle
    tb.Checkbutton(section_frame, text="Use NCX-only Navigation (Compatibility Mode)", 
                  variable=self.force_ncx_only_var,
                  bootstyle="round-toggle").pack(anchor=tk.W, pady=(5, 5))
                  
    # CSS Attachment toggle - NEW!
    tb.Checkbutton(section_frame, text="Attach CSS to Chapters (Fixes styling issues)", 
                  variable=self.attach_css_to_chapters_var,
          bootstyle="round-toggle").pack(anchor=tk.W, pady=(5, 5))      
    
    # Output file naming
    tb.Checkbutton(section_frame, text="Retain source extension (no 'response_' prefix)", 
                  variable=self.retain_source_extension_var,
                  bootstyle="round-toggle").pack(anchor=tk.W, pady=(5, 5))

def _create_processing_options_section(self, parent):
    """Create processing options section"""
    section_frame = tk.LabelFrame(parent, text="Processing Options", padx=10, pady=10)
    section_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=5)
    
    # Reinforce messages option
    reinforce_frame = tk.Frame(section_frame)
    reinforce_frame.pack(anchor=tk.W, pady=(0, 10))
    tk.Label(reinforce_frame, text="Reinforce every").pack(side=tk.LEFT)
    tb.Entry(reinforce_frame, width=6, textvariable=self.reinforcement_freq_var).pack(side=tk.LEFT, padx=5)
    tk.Label(reinforce_frame, text="messages").pack(side=tk.LEFT)
    
    tb.Checkbutton(section_frame, text="Emergency Paragraph Restoration", 
                  variable=self.emergency_restore_var,
                  bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
    
    tk.Label(section_frame, text="Fixes AI responses that lose paragraph\nstructure (wall of text)",
            font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 5))
            
    tb.Checkbutton(section_frame, text="Enable Decimal Chapter Detection (EPUBs)", 
          variable=self.enable_decimal_chapters_var,
          bootstyle="round-toggle").pack(anchor=tk.W, pady=2)

    tk.Label(section_frame, text="Detect chapters like 1.1, 1.2 in EPUB files\n(Text files always use decimal chapters when split)",
            font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
    
    # === CHAPTER EXTRACTION SETTINGS ===
    # Main extraction frame
    extraction_frame = tk.LabelFrame(section_frame, text="Chapter Extraction Settings", padx=10, pady=5)
    extraction_frame.pack(fill=tk.X, pady=(0, 10))
    
    # Initialize variables if not exists
    if not hasattr(self, 'text_extraction_method_var'):
        # Check if using old enhanced mode
        if self.config.get('extraction_mode') == 'enhanced':
            self.text_extraction_method_var = tk.StringVar(value='enhanced')
            # Set filtering from enhanced_filtering or default to smart
            self.file_filtering_level_var = tk.StringVar(
                value=self.config.get('enhanced_filtering', 'smart')
            )
        else:
            self.text_extraction_method_var = tk.StringVar(value='standard')
            self.file_filtering_level_var = tk.StringVar(
                value=self.config.get('extraction_mode', 'smart')
            )
    
    if not hasattr(self, 'enhanced_preserve_structure_var'):
        self.enhanced_preserve_structure_var = tk.BooleanVar(
            value=self.config.get('enhanced_preserve_structure', True)
        )
    
    # --- Text Extraction Method Section ---
    method_frame = tk.Frame(extraction_frame)
    method_frame.pack(fill=tk.X, pady=(0, 15))
    
    tk.Label(method_frame, text="Text Extraction Method:", 
            font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
    
    # Standard extraction
    tb.Radiobutton(method_frame, text="Standard (BeautifulSoup)", 
                  variable=self.text_extraction_method_var, value="standard",
                  bootstyle="round-toggle",
                  command=self.on_extraction_method_change).pack(anchor=tk.W, pady=2)
    
    tk.Label(method_frame, text="Traditional HTML parsing - fast and reliable",
            font=('TkDefaultFont', 9), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 5))
    
    # Enhanced extraction
    tb.Radiobutton(method_frame, text="🚀 Enhanced (html2text)", 
                  variable=self.text_extraction_method_var, value="enhanced",
                  bootstyle="success-round-toggle",
                  command=self.on_extraction_method_change).pack(anchor=tk.W, pady=2)

    tk.Label(method_frame, text="Superior Unicode handling, cleaner text extraction",
            font=('TkDefaultFont', 9), fg='dark green', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 5))
    
    # Enhanced options (shown when enhanced is selected)
    self.enhanced_options_frame = tk.Frame(method_frame)
    self.enhanced_options_frame.pack(fill=tk.X, padx=20, pady=(5, 0))
    
    # Structure preservation
    tb.Checkbutton(self.enhanced_options_frame, text="Preserve Markdown Structure", 
                  variable=self.enhanced_preserve_structure_var,
                  bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
    
    tk.Label(self.enhanced_options_frame, text="Keep formatting (bold, headers, lists) for better AI context",
            font=('TkDefaultFont', 8), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 3))
    
    # Requirements note
    requirements_frame = tk.Frame(self.enhanced_options_frame)
    requirements_frame.pack(anchor=tk.W, pady=(5, 0))
    
    # Separator
    ttk.Separator(method_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 10))
    
    # --- File Filtering Level Section ---
    filtering_frame = tk.Frame(extraction_frame)
    filtering_frame.pack(fill=tk.X, pady=(0, 10))
    
    tk.Label(filtering_frame, text="File Filtering Level:", 
            font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
    
    # Smart filtering
    tb.Radiobutton(filtering_frame, text="Smart (Aggressive Filtering)", 
                  variable=self.file_filtering_level_var, value="smart",
                  bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
    
    tk.Label(filtering_frame, text="Skips navigation, TOC, copyright files\nBest for clean EPUBs with clear chapter structure",
            font=('TkDefaultFont', 9), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 5))
    
    # Comprehensive filtering
    tb.Radiobutton(filtering_frame, text="Comprehensive (Moderate Filtering)", 
                  variable=self.file_filtering_level_var, value="comprehensive",
                  bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
    
    tk.Label(filtering_frame, text="Only skips obvious navigation files\nGood when Smart mode misses chapters",
            font=('TkDefaultFont', 9), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 5))
    
    # Full extraction
    tb.Radiobutton(filtering_frame, text="Full (No Filtering)", 
                  variable=self.file_filtering_level_var, value="full",
                  bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
    
    tk.Label(filtering_frame, text="Extracts ALL HTML/XHTML files\nUse when other modes skip important content",
            font=('TkDefaultFont', 9), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 5))
    
    # NEW: Force BeautifulSoup for Traditional APIs toggle
    if not hasattr(self, 'force_bs_for_traditional_var'):
        self.force_bs_for_traditional_var = tk.BooleanVar(
            value=self.config.get('force_bs_for_traditional', True)
        )
    tb.Checkbutton(extraction_frame, text="Force BeautifulSoup for DeepL / Google Translate / Google Free",
                  variable=self.force_bs_for_traditional_var,
                  bootstyle="round-toggle").pack(anchor=tk.W, pady=(0, 5))
    tk.Label(extraction_frame, text="When enabled, DeepL/Google Translate/Google Free always use BeautifulSoup extraction even if Enhanced is selected.",
             font=('TkDefaultFont', 8), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 5))
    
    # Chapter merging option
    ttk.Separator(extraction_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 10))
    
    # Initialize disable_chapter_merging_var if not exists
    if not hasattr(self, 'disable_chapter_merging_var'):
        self.disable_chapter_merging_var = tk.BooleanVar(
            value=self.config.get('disable_chapter_merging', False)
        )
    
    tb.Checkbutton(extraction_frame, text="Disable Chapter Merging", 
                  variable=self.disable_chapter_merging_var,
                  bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
    
    tk.Label(extraction_frame, text="Disable automatic merging of Section/Chapter pairs.\nEach file will be treated as a separate chapter.",
            font=('TkDefaultFont', 9), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 5))
    
    # === REMAINING OPTIONS ===
    tb.Checkbutton(section_frame, text="Disable Image Gallery in EPUB", 
                  variable=self.disable_epub_gallery_var,
                  bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
    
    tk.Label(section_frame, text="Skip creating image gallery page in EPUB",
            font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))

    # New: Disable Automatic Cover Creation
    tb.Checkbutton(section_frame, text="Disable Automatic Cover Creation", 
                  variable=self.disable_automatic_cover_creation_var,
                  bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
    
    tk.Label(section_frame, text="When enabled: no auto-generated cover page is created.",
            font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))

    # New: Translate cover.html (Skip Override)
    tb.Checkbutton(section_frame, text="Translate cover.html (Skip Override)", 
                  variable=self.translate_cover_html_var,
                  bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
    
    tk.Label(section_frame, text="When enabled: existing cover.html/cover.xhtml will be included and translated (not skipped).",
            font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
    
    tb.Checkbutton(section_frame, text="Disable 0-based Chapter Detection", 
                  variable=self.disable_zero_detection_var,
                  bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
    
    tk.Label(section_frame, text="Always use chapter ranges as specified\n(don't force adjust to chapter 1)",
            font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
            
    tb.Checkbutton(section_frame, text="Use Header as Output Name", 
                  variable=self.use_header_as_output_var,
                  bootstyle="round-toggle").pack(anchor=tk.W, pady=2)

    tk.Label(section_frame, text="Use chapter headers/titles as output filenames",
            font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
            
    # Chapter number offset
    ttk.Separator(section_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 10))
    
    offset_frame = tk.Frame(section_frame)
    offset_frame.pack(anchor=tk.W, pady=5)
    
    tk.Label(offset_frame, text="Chapter Number Offset:").pack(side=tk.LEFT)
    
    # Create variable if not exists
    if not hasattr(self, 'chapter_number_offset_var'):
        self.chapter_number_offset_var = tk.StringVar(
            value=str(self.config.get('chapter_number_offset', '0'))
        )
    
    tb.Entry(offset_frame, width=6, textvariable=self.chapter_number_offset_var).pack(side=tk.LEFT, padx=5)
    
    tk.Label(offset_frame, text="(+/- adjustment)").pack(side=tk.LEFT)
    
    tk.Label(section_frame, text="Adjust all chapter numbers by this amount.\nUseful for matching file numbers to actual chapters.",
            font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))    
            
    # Add separator before API safety settings
    ttk.Separator(section_frame, orient='horizontal').pack(fill=tk.X, pady=(15, 10))

    # Post-Translation Scanning Phase
    scan_phase_frame = tk.Frame(section_frame)
    scan_phase_frame.pack(anchor=tk.W, fill=tk.X, pady=(10, 0))
    
    tb.Checkbutton(scan_phase_frame, text="Enable post-translation Scanning phase",
                  variable=self.scan_phase_enabled_var,
                  bootstyle="round-toggle").pack(side=tk.LEFT)
    
    # Mode selector
    tk.Label(scan_phase_frame, text="Mode:").pack(side=tk.LEFT, padx=(15, 5))
    scan_modes = ["quick-scan", "aggressive", "ai-hunter", "custom"]
    scan_mode_combo = ttk.Combobox(scan_phase_frame, textvariable=self.scan_phase_mode_var, values=scan_modes, state="readonly", width=12)
    scan_mode_combo.pack(side=tk.LEFT)
    # Prevent accidental changes from mouse wheel while scrolling
    UIHelper.disable_spinbox_mousewheel(scan_mode_combo)
    
    tk.Label(section_frame, text="Automatically run QA Scanner after translation completes",
           font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
    
    # Conservative Batching Toggle
    tb.Checkbutton(section_frame, text="Use Conservative Batching",
                  variable=self.conservative_batching_var,
                  bootstyle="round-toggle").pack(anchor=tk.W, pady=(10, 0))
    
    tk.Label(section_frame, text="When enabled: Groups chapters in batches of 3x batch size for memory management\nWhen disabled (default): Uses direct batch size for faster processing",
           font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
    
    ttk.Separator(section_frame, orient='horizontal').pack(fill=tk.X, pady=(15, 10))
    
    # API Safety Settings subsection
    tk.Label(section_frame, text="API Safety Settings", 
             font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W, pady=(5, 5))
    
    # Create the Gemini safety checkbox
    if not hasattr(self, 'disable_gemini_safety_var'):
        self.disable_gemini_safety_var = tk.BooleanVar(
            value=self.config.get('disable_gemini_safety', False)
        )
    
    tb.Checkbutton(
        section_frame,
        text="Disable API Safety Filters (Gemini, Groq, Fireworks, etc.)",
        variable=self.disable_gemini_safety_var,
        bootstyle="round-toggle"
    ).pack(anchor=tk.W, pady=(5, 0))
    
    # Add warning text
    warning_text = ("⚠️ Disables content safety filters for supported providers.\n"
                   "Gemini: Sets all harm categories to BLOCK_NONE.\n"
                   "Groq/Fireworks: Disables moderation parameter.\n")
    tk.Label(
        section_frame,
        text=warning_text,
        font=('TkDefaultFont', 9),
        fg='#ff6b6b',
        justify=tk.LEFT
    ).pack(anchor=tk.W, padx=(20, 0), pady=(0, 5))
    
    # Add note about affected models
    tk.Label(
        section_frame,
        text="Does NOT affect ElectronHub Gemini models (eh/gemini-*) or Together AI",
        font=('TkDefaultFont', 8),
        fg='gray',
        justify=tk.LEFT
    ).pack(anchor=tk.W, padx=(20, 0), pady=(0, 8))

    # New: OpenRouter Transport Preference
    # Toggle to force HTTP-only path for OpenRouter (SDK bypass)
    if not hasattr(self, 'openrouter_http_only_var'):
        self.openrouter_http_only_var = tk.BooleanVar(
            value=self.config.get('openrouter_use_http_only', False)
        )
    
    tb.Checkbutton(
        section_frame,
        text="Use HTTP-only for OpenRouter (bypass SDK)",
        variable=self.openrouter_http_only_var,
        bootstyle="round-toggle"
    ).pack(anchor=tk.W, pady=(8, 0))
    
    tk.Label(
        section_frame,
        text="When enabled, requests to OpenRouter use direct HTTP POST with explicit headers (Accept, Referer, X-Title).",
        font=('TkDefaultFont', 9),
        fg='gray',
        justify=tk.LEFT
    ).pack(anchor=tk.W, padx=(20, 0), pady=(0, 5))

    # OpenRouter: Disable compression (Accept-Encoding: identity)
    if not hasattr(self, 'openrouter_accept_identity_var'):
        self.openrouter_accept_identity_var = tk.BooleanVar(
            value=self.config.get('openrouter_accept_identity', False)
        )
    tb.Checkbutton(
        section_frame,
        text="Disable compression for OpenRouter (Accept-Encoding)",
        variable=self.openrouter_accept_identity_var,
        bootstyle="round-toggle"
    ).pack(anchor=tk.W, pady=(4, 0))
    tk.Label(
        section_frame,
        text="Sends Accept-Encoding: identity to request uncompressed responses.\n"
             "Use if proxies/CDNs cause corrupted or non-JSON compressed bodies.",
        font=('TkDefaultFont', 8),
        fg='gray',
        justify=tk.LEFT
    ).pack(anchor=tk.W, padx=(20, 0), pady=(0, 8))

    # OpenRouter: Provider preference
    provider_frame = tk.Frame(section_frame)
    provider_frame.pack(anchor=tk.W, fill=tk.X, pady=(4, 0))
    
    tk.Label(provider_frame, text="Preferred OpenRouter Provider:").pack(side=tk.LEFT)
    
    if not hasattr(self, 'openrouter_preferred_provider_var'):
        self.openrouter_preferred_provider_var = tk.StringVar(
            value=self.config.get('openrouter_preferred_provider', 'Auto')
        )
    
    # List of common OpenRouter providers
    provider_options = ['Auto', 'DeepInfra', 'OpenInference', 'Together', 'Fireworks', 'Lepton', 'Mancer']
    provider_combo = ttk.Combobox(
        provider_frame,
        textvariable=self.openrouter_preferred_provider_var,
        values=provider_options,
        state="readonly",
        width=15
    )
    provider_combo.pack(side=tk.LEFT, padx=(10, 0))
    # Prevent accidental changes from mouse wheel while scrolling
    UIHelper.disable_spinbox_mousewheel(provider_combo)
    
    tk.Label(
        section_frame,
        text="Specify which upstream provider OpenRouter should prefer for your requests.\n"
             "'Auto' lets OpenRouter choose. Specific providers may have different availability.",
        font=('TkDefaultFont', 8),
        fg='gray',
        justify=tk.LEFT
    ).pack(anchor=tk.W, padx=(20, 0), pady=(0, 8))
    
    # Initial state - show/hide enhanced options
    self.on_extraction_method_change()

def on_extraction_method_change(self):
    """Handle extraction method changes and show/hide Enhanced options"""
    if hasattr(self, 'text_extraction_method_var') and hasattr(self, 'enhanced_options_frame'):
        if self.text_extraction_method_var.get() == 'enhanced':
            self.enhanced_options_frame.pack(fill=tk.X, padx=20, pady=(5, 0))
        else:
            self.enhanced_options_frame.pack_forget()
            
def _create_image_translation_section(self, parent):
    """Create image translation section"""
    section_frame = tk.LabelFrame(parent, text="Image Translation", padx=10, pady=8)
    section_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=(5, 10))
    
    left_column = tk.Frame(section_frame)
    left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
    
    right_column = tk.Frame(section_frame)
    right_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Left column
    enable_frame = tk.Frame(left_column)
    enable_frame.pack(fill=tk.X, pady=(0, 10))
    
    tb.Checkbutton(enable_frame, text="Enable Image Translation", 
                  variable=self.enable_image_translation_var,
                  bootstyle="round-toggle").pack(anchor=tk.W)
    
    tk.Label(left_column, text="Extracts and translates text from images using vision models",
            font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, pady=(0, 10))
    
    tb.Checkbutton(left_column, text="Process Long Images (Web Novel Style)", 
                  variable=self.process_webnovel_images_var,
                  bootstyle="round-toggle").pack(anchor=tk.W)
    
    tk.Label(left_column, text="Include tall images often used in web novels",
            font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, padx=20, pady=(0, 10))
    
    tb.Checkbutton(left_column, text="Hide labels and remove OCR images", 
                  variable=self.hide_image_translation_label_var,
                  bootstyle="round-toggle").pack(anchor=tk.W)
    
    tk.Label(left_column, text="Clean mode: removes image and shows only translated text",
            font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, padx=20, pady=(0, 10))

    # Add some spacing
    tk.Frame(left_column, height=10).pack()

    # Watermark removal toggle
    tb.Checkbutton(left_column, text="Enable Watermark Removal", 
                  variable=self.enable_watermark_removal_var,
                  bootstyle="round-toggle").pack(anchor=tk.W)

    tk.Label(left_column, text="Advanced preprocessing to remove watermarks from images",
            font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, padx=20, pady=(0, 10))

    # Save cleaned images toggle - create with reference
    self.save_cleaned_checkbox = tb.Checkbutton(left_column, text="Save Cleaned Images", 
                                               variable=self.save_cleaned_images_var,
                                               bootstyle="round-toggle")
    self.save_cleaned_checkbox.pack(anchor=tk.W, padx=(20, 0))

    tk.Label(left_column, text="Keep watermark-removed images in translated_images/cleaned/",
            font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, padx=40, pady=(0, 10))

    # Advanced watermark removal toggle - create with reference
    self.advanced_watermark_checkbox = tb.Checkbutton(left_column, text="Advanced Watermark Removal", 
                                                     variable=self.advanced_watermark_removal_var,
                                                     bootstyle="round-toggle")
    self.advanced_watermark_checkbox.pack(anchor=tk.W, padx=(20, 0))

    tk.Label(left_column, text="Use FFT-based pattern detection for stubborn watermarks",
            font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, padx=40)
    
    # Right column
    settings_frame = tk.Frame(right_column)
    settings_frame.pack(fill=tk.X)

    settings_frame.grid_columnconfigure(1, minsize=80)

    settings = [
        ("Min Image height (px):", self.webnovel_min_height_var),
        ("Max Images per chapter:", self.max_images_per_chapter_var),
        ("Chunk height:", self.image_chunk_height_var),
        ("Chunk overlap (%):", self.image_chunk_overlap_var)  # Add this new setting
    ]

    for row, (label, var) in enumerate(settings):
        tk.Label(settings_frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
        entry = tb.Entry(settings_frame, width=10, textvariable=var)
        entry.grid(row=row, column=1, sticky=tk.W, pady=3)
        
        # Add tooltip for the overlap setting
        if "overlap" in label.lower():
            tk.Label(settings_frame, text="1-10% recommended", 
                    font=('TkDefaultFont', 8), fg='gray').grid(row=row, column=2, sticky=tk.W, padx=(5, 0))

    # Buttons for prompts and compression
    # TODO: Implement configure_image_chunk_prompt method
    # tb.Button(settings_frame, text="Image Chunk Prompt", 
    #          command=self.configure_image_chunk_prompt,
    #          bootstyle="info-outline", width=20).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
    
    # Add Image Compression button
    # TODO: Implement configure_image_compression method
    # tb.Button(settings_frame, text="🗜️ Image Compression", 
    #          command=self.configure_image_compression,
    #          bootstyle="info-outline", width=25).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

    # Add the toggle here in the right column with some spacing
    tk.Frame(right_column, height=15).pack()  # Add some spacing

    tb.Checkbutton(right_column, text="Send tall image chunks in single API call (NOT RECOMMENDED)", 
                  variable=self.single_api_image_chunks_var,
                  bootstyle="round-toggle").pack(anchor=tk.W)

    tk.Label(right_column, text="All image chunks sent to 1 API call (Most AI models don't like this)",
            font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, padx=20, pady=(0, 10))

    tk.Label(right_column, text="💡 Supported models:\n"
            "• Gemini 1.5 Pro/Flash, 2.0 Flash\n"
            "• GPT-4V, GPT-4o, o4-mini",
            font=('TkDefaultFont', 10), fg='#666', justify=tk.LEFT).pack(anchor=tk.W, pady=(10, 0))
        
            
    # Set up the dependency logic
    def toggle_watermark_options(*args):
        if self.enable_watermark_removal_var.get():
            # Enable both sub-options
            self.save_cleaned_checkbox.config(state=tk.NORMAL)
            self.advanced_watermark_checkbox.config(state=tk.NORMAL)
        else:
            # Disable both sub-options and turn them off
            self.save_cleaned_checkbox.config(state=tk.DISABLED)
            self.advanced_watermark_checkbox.config(state=tk.DISABLED)
            self.save_cleaned_images_var.set(False)
            self.advanced_watermark_removal_var.set(False)

    # Bind the trace to the watermark removal variable
    self.enable_watermark_removal_var.trace('w', toggle_watermark_options)
    
    # Call once to set initial state
    toggle_watermark_options()
    
def on_extraction_mode_change(self):
    """Handle extraction mode changes and show/hide Enhanced options"""
    if self.extraction_mode_var.get() == 'enhanced':
        # Show enhanced options
        if hasattr(self, 'enhanced_options_separator'):
            self.enhanced_options_separator.pack(fill=tk.X, pady=(5, 5))
        if hasattr(self, 'enhanced_options_frame'):
            self.enhanced_options_frame.pack(fill=tk.X, padx=20)
    else:
        # Hide enhanced options
        if hasattr(self, 'enhanced_options_separator'):
            self.enhanced_options_separator.pack_forget()
        if hasattr(self, 'enhanced_options_frame'):
            self.enhanced_options_frame.pack_forget()
            
def _create_anti_duplicate_section(self, parent):
    """Create comprehensive anti-duplicate parameter controls with tabs"""
    # Anti-Duplicate Parameters section
    ad_frame = tk.LabelFrame(parent, text="🎯 Anti-Duplicate Parameters", padx=15, pady=10)
    ad_frame.grid(row=6, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 15))
    
    # Description
    desc_label = tk.Label(ad_frame, 
        text="Configure parameters to reduce duplicate translations across all AI providers.",
        font=('TkDefaultFont', 9), fg='gray', wraplength=520)
    desc_label.pack(anchor=tk.W, pady=(0, 10))
    
    # Enable/Disable toggle
    self.enable_anti_duplicate_var = tk.BooleanVar(value=self.config.get('enable_anti_duplicate', False))
    enable_cb = tb.Checkbutton(ad_frame, text="Enable Anti-Duplicate Parameters", 
                              variable=self.enable_anti_duplicate_var,
                              command=self._toggle_anti_duplicate_controls)
    enable_cb.pack(anchor=tk.W, pady=(0, 10))
    
    # Create notebook for organized parameters
    self.anti_duplicate_notebook = ttk.Notebook(ad_frame)
    self.anti_duplicate_notebook.pack(fill=tk.BOTH, expand=True, pady=5)
    
    # Tab 1: Core Parameters
    core_frame = tk.Frame(self.anti_duplicate_notebook)
    self.anti_duplicate_notebook.add(core_frame, text="Core Parameters")
    
    # Top-P (Nucleus Sampling)
    top_p_frame = tk.Frame(core_frame)
    top_p_frame.pack(fill=tk.X, pady=5)
    
    tk.Label(top_p_frame, text="Top-P (Nucleus Sampling):", width=25, anchor='w').pack(side=tk.LEFT)
    self.top_p_var = tk.DoubleVar(value=self.config.get('top_p', 1.0))
    top_p_scale = tk.Scale(top_p_frame, from_=0.1, to=1.0, resolution=0.01, 
                          orient=tk.HORIZONTAL, variable=self.top_p_var, length=200)
    top_p_scale.pack(side=tk.LEFT, padx=5)
    self.top_p_value_label = tk.Label(top_p_frame, text="", width=8)
    self.top_p_value_label.pack(side=tk.LEFT, padx=5)
    
    def update_top_p_label(*args):
        val = self.top_p_var.get()
        self.top_p_value_label.config(text=f"{val:.2f}")
    self.top_p_var.trace('w', update_top_p_label)
    update_top_p_label()
    
    # Top-K (Vocabulary Limit)
    top_k_frame = tk.Frame(core_frame)
    top_k_frame.pack(fill=tk.X, pady=5)
    
    tk.Label(top_k_frame, text="Top-K (Vocabulary Limit):", width=25, anchor='w').pack(side=tk.LEFT)
    self.top_k_var = tk.IntVar(value=self.config.get('top_k', 0))
    top_k_scale = tk.Scale(top_k_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                          variable=self.top_k_var, length=200)
    top_k_scale.pack(side=tk.LEFT, padx=5)
    self.top_k_value_label = tk.Label(top_k_frame, text="", width=8)
    self.top_k_value_label.pack(side=tk.LEFT, padx=5)
    
    def update_top_k_label(*args):
        val = self.top_k_var.get()
        self.top_k_value_label.config(text=f"{val}" if val > 0 else "OFF")
    self.top_k_var.trace('w', update_top_k_label)
    update_top_k_label()
    
    # Frequency Penalty
    freq_penalty_frame = tk.Frame(core_frame)
    freq_penalty_frame.pack(fill=tk.X, pady=5)
    
    tk.Label(freq_penalty_frame, text="Frequency Penalty:", width=25, anchor='w').pack(side=tk.LEFT)
    self.frequency_penalty_var = tk.DoubleVar(value=self.config.get('frequency_penalty', 0.0))
    freq_scale = tk.Scale(freq_penalty_frame, from_=0.0, to=2.0, resolution=0.1, 
                         orient=tk.HORIZONTAL, variable=self.frequency_penalty_var, length=200)
    freq_scale.pack(side=tk.LEFT, padx=5)
    self.freq_penalty_value_label = tk.Label(freq_penalty_frame, text="", width=8)
    self.freq_penalty_value_label.pack(side=tk.LEFT, padx=5)
    
    def update_freq_label(*args):
        val = self.frequency_penalty_var.get()
        self.freq_penalty_value_label.config(text=f"{val:.1f}" if val > 0 else "OFF")
    self.frequency_penalty_var.trace('w', update_freq_label)
    update_freq_label()
    
    # Presence Penalty
    pres_penalty_frame = tk.Frame(core_frame)
    pres_penalty_frame.pack(fill=tk.X, pady=5)
    
    tk.Label(pres_penalty_frame, text="Presence Penalty:", width=25, anchor='w').pack(side=tk.LEFT)
    self.presence_penalty_var = tk.DoubleVar(value=self.config.get('presence_penalty', 0.0))
    pres_scale = tk.Scale(pres_penalty_frame, from_=0.0, to=2.0, resolution=0.1, 
                         orient=tk.HORIZONTAL, variable=self.presence_penalty_var, length=200)
    pres_scale.pack(side=tk.LEFT, padx=5)
    self.pres_penalty_value_label = tk.Label(pres_penalty_frame, text="", width=8)
    self.pres_penalty_value_label.pack(side=tk.LEFT, padx=5)
    
    def update_pres_label(*args):
        val = self.presence_penalty_var.get()
        self.pres_penalty_value_label.config(text=f"{val:.1f}" if val > 0 else "OFF")
    self.presence_penalty_var.trace('w', update_pres_label)
    update_pres_label()
    
    # Tab 2: Advanced Parameters
    advanced_frame = tk.Frame(self.anti_duplicate_notebook)
    self.anti_duplicate_notebook.add(advanced_frame, text="Advanced")
    
    # Repetition Penalty
    rep_penalty_frame = tk.Frame(advanced_frame)
    rep_penalty_frame.pack(fill=tk.X, pady=5)
    
    tk.Label(rep_penalty_frame, text="Repetition Penalty:", width=25, anchor='w').pack(side=tk.LEFT)
    self.repetition_penalty_var = tk.DoubleVar(value=self.config.get('repetition_penalty', 1.0))
    rep_scale = tk.Scale(rep_penalty_frame, from_=1.0, to=2.0, resolution=0.05, 
                        orient=tk.HORIZONTAL, variable=self.repetition_penalty_var, length=200)
    rep_scale.pack(side=tk.LEFT, padx=5)
    self.rep_penalty_value_label = tk.Label(rep_penalty_frame, text="", width=8)
    self.rep_penalty_value_label.pack(side=tk.LEFT, padx=5)
    
    def update_rep_label(*args):
        val = self.repetition_penalty_var.get()
        self.rep_penalty_value_label.config(text=f"{val:.2f}" if val > 1.0 else "OFF")
    self.repetition_penalty_var.trace('w', update_rep_label)
    update_rep_label()
    
    # Candidate Count (Gemini)
    candidate_frame = tk.Frame(advanced_frame)
    candidate_frame.pack(fill=tk.X, pady=5)
    
    tk.Label(candidate_frame, text="Candidate Count (Gemini):", width=25, anchor='w').pack(side=tk.LEFT)
    self.candidate_count_var = tk.IntVar(value=self.config.get('candidate_count', 1))
    candidate_scale = tk.Scale(candidate_frame, from_=1, to=4, orient=tk.HORIZONTAL, 
                              variable=self.candidate_count_var, length=200)
    candidate_scale.pack(side=tk.LEFT, padx=5)
    self.candidate_value_label = tk.Label(candidate_frame, text="", width=8)
    self.candidate_value_label.pack(side=tk.LEFT, padx=5)
    
    def update_candidate_label(*args):
        val = self.candidate_count_var.get()
        self.candidate_value_label.config(text=f"{val}")
    self.candidate_count_var.trace('w', update_candidate_label)
    update_candidate_label()
    
    # Tab 3: Stop Sequences
    stop_frame = tk.Frame(self.anti_duplicate_notebook)
    self.anti_duplicate_notebook.add(stop_frame, text="Stop Sequences")
    
    # Custom Stop Sequences
    stop_seq_frame = tk.Frame(stop_frame)
    stop_seq_frame.pack(fill=tk.X, pady=5)
    
    tk.Label(stop_seq_frame, text="Custom Stop Sequences:", width=25, anchor='w').pack(side=tk.LEFT)
    self.custom_stop_sequences_var = tk.StringVar(value=self.config.get('custom_stop_sequences', ''))
    stop_entry = tb.Entry(stop_seq_frame, textvariable=self.custom_stop_sequences_var, width=30)
    stop_entry.pack(side=tk.LEFT, padx=5)
    tk.Label(stop_seq_frame, text="(comma-separated)", font=('TkDefaultFont', 8), fg='gray').pack(side=tk.LEFT)
    
    # Tab 4: Logit Bias (OpenAI)
    bias_frame = tk.Frame(self.anti_duplicate_notebook)
    self.anti_duplicate_notebook.add(bias_frame, text="Logit Bias")
    
    # Logit Bias Enable
    self.logit_bias_enabled_var = tk.BooleanVar(value=self.config.get('logit_bias_enabled', False))
    bias_cb = tb.Checkbutton(bias_frame, text="Enable Logit Bias (OpenAI only)", 
                            variable=self.logit_bias_enabled_var)
    bias_cb.pack(anchor=tk.W, pady=5)
    
    # Logit Bias Strength
    bias_strength_frame = tk.Frame(bias_frame)
    bias_strength_frame.pack(fill=tk.X, pady=5)
    
    tk.Label(bias_strength_frame, text="Bias Strength:", width=25, anchor='w').pack(side=tk.LEFT)
    self.logit_bias_strength_var = tk.DoubleVar(value=self.config.get('logit_bias_strength', -0.5))
    bias_scale = tk.Scale(bias_strength_frame, from_=-2.0, to=2.0, resolution=0.1, 
                         orient=tk.HORIZONTAL, variable=self.logit_bias_strength_var, length=200)
    bias_scale.pack(side=tk.LEFT, padx=5)
    self.bias_strength_value_label = tk.Label(bias_strength_frame, text="", width=8)
    self.bias_strength_value_label.pack(side=tk.LEFT, padx=5)
    
    def update_bias_strength_label(*args):
        val = self.logit_bias_strength_var.get()
        self.bias_strength_value_label.config(text=f"{val:.1f}")
    self.logit_bias_strength_var.trace('w', update_bias_strength_label)
    update_bias_strength_label()
    
    # Preset bias targets
    preset_frame = tk.Frame(bias_frame)
    preset_frame.pack(fill=tk.X, pady=5)
    
    tk.Label(preset_frame, text="Preset Bias Targets:", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W)
    
    self.bias_common_words_var = tk.BooleanVar(value=self.config.get('bias_common_words', False))
    tb.Checkbutton(preset_frame, text="Bias against common words (the, and, said)", 
                  variable=self.bias_common_words_var).pack(anchor=tk.W)
    
    self.bias_repetitive_phrases_var = tk.BooleanVar(value=self.config.get('bias_repetitive_phrases', False))
    tb.Checkbutton(preset_frame, text="Bias against repetitive phrases", 
                  variable=self.bias_repetitive_phrases_var).pack(anchor=tk.W)
    
    # Provider compatibility info
    compat_frame = tk.Frame(ad_frame)
    compat_frame.pack(fill=tk.X, pady=(15, 0))

    tk.Label(compat_frame, text="Parameter Compatibility:", 
            font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W)

    compat_text = tk.Label(compat_frame, 
        text="• Core: Most providers • Advanced: DeepSeek, Mistral, Groq • Logit Bias: OpenAI only",
        font=('TkDefaultFont', 8), fg='gray', justify=tk.LEFT)
    compat_text.pack(anchor=tk.W, pady=(5, 0))

    # Reset button
    reset_frame = tk.Frame(ad_frame)
    reset_frame.pack(fill=tk.X, pady=(10, 0))

    tb.Button(reset_frame, text="🔄 Reset to Defaults", 
             command=self._reset_anti_duplicate_defaults,
             bootstyle="secondary", width=20).pack(side=tk.LEFT)

    tk.Label(reset_frame, text="Reset all anti-duplicate parameters to default values", 
            font=('TkDefaultFont', 8), fg='gray').pack(side=tk.LEFT, padx=(10, 0))

    # Store all tab frames for enable/disable
    self.anti_duplicate_tabs = [core_frame, advanced_frame, stop_frame, bias_frame]

    # Initial state
    self._toggle_anti_duplicate_controls()

def _toggle_anti_duplicate_controls(self):
    """Enable/disable anti-duplicate parameter controls"""
    state = tk.NORMAL if self.enable_anti_duplicate_var.get() else tk.DISABLED
    
    # Disable/enable the notebook itself
    if hasattr(self, 'anti_duplicate_notebook'):
        try:
            self.anti_duplicate_notebook.config(state=state)
        except tk.TclError:
            pass
    
    # Disable/enable all controls in tabs
    if hasattr(self, 'anti_duplicate_tabs'):
        for tab_frame in self.anti_duplicate_tabs:
            for widget in tab_frame.winfo_children():
                for child in widget.winfo_children():
                    if hasattr(child, 'config'):
                        try:
                            child.config(state=state)
                        except tk.TclError:
                            pass

def _toggle_http_tuning_controls(self):
    """Enable/disable the HTTP timeout/pooling controls as a group"""
    enabled = bool(self.enable_http_tuning_var.get()) if hasattr(self, 'enable_http_tuning_var') else False
    state = 'normal' if enabled else 'disabled'
    # Entries
    for attr in ['connect_timeout_entry', 'read_timeout_entry', 'http_pool_connections_entry', 'http_pool_maxsize_entry']:
        widget = getattr(self, attr, None)
        if widget is not None:
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass
    # Retry-After checkbox
    if hasattr(self, 'ignore_retry_after_checkbox') and self.ignore_retry_after_checkbox is not None:
        try:
            self.ignore_retry_after_checkbox.configure(state=state)
        except tk.TclError:
            pass
                            
def _reset_anti_duplicate_defaults(self):
    """Reset all anti-duplicate parameters to their default values"""
    import tkinter.messagebox as messagebox
    
    # Ask for confirmation
    if not messagebox.askyesno("Reset Anti-Duplicate Parameters", 
                              "Are you sure you want to reset all anti-duplicate parameters to their default values?"):
        return
    
    # Reset all variables to defaults
    if hasattr(self, 'enable_anti_duplicate_var'):
        self.enable_anti_duplicate_var.set(False)
    
    if hasattr(self, 'top_p_var'):
        self.top_p_var.set(1.0)  # Default = no effect
    
    if hasattr(self, 'top_k_var'):
        self.top_k_var.set(0)  # Default = disabled
    
    if hasattr(self, 'frequency_penalty_var'):
        self.frequency_penalty_var.set(0.0)  # Default = no penalty
    
    if hasattr(self, 'presence_penalty_var'):
        self.presence_penalty_var.set(0.0)  # Default = no penalty
    
    if hasattr(self, 'repetition_penalty_var'):
        self.repetition_penalty_var.set(1.0)  # Default = no penalty
    
    if hasattr(self, 'candidate_count_var'):
        self.candidate_count_var.set(1)  # Default = single response
    
    if hasattr(self, 'custom_stop_sequences_var'):
        self.custom_stop_sequences_var.set("")  # Default = empty
    
    if hasattr(self, 'logit_bias_enabled_var'):
        self.logit_bias_enabled_var.set(False)  # Default = disabled
    
    if hasattr(self, 'logit_bias_strength_var'):
        self.logit_bias_strength_var.set(-0.5)  # Default strength
    
    if hasattr(self, 'bias_common_words_var'):
        self.bias_common_words_var.set(False)  # Default = disabled
    
    if hasattr(self, 'bias_repetitive_phrases_var'):
        self.bias_repetitive_phrases_var.set(False)  # Default = disabled
    
    # Update enable/disable state
    self._toggle_anti_duplicate_controls()
    
    # Show success message
    messagebox.showinfo("Reset Complete", "All anti-duplicate parameters have been reset to their default values.")
    
    # Log the reset
    if hasattr(self, 'append_log'):
        self.append_log("🔄 Anti-duplicate parameters reset to defaults")        

def _create_custom_api_endpoints_section(self, parent_frame):
    """Create the Custom API Endpoints section"""
    # Custom API Endpoints Section
    endpoints_frame = tb.LabelFrame(parent_frame, text="Custom API Endpoints", padding=10)
    endpoints_frame.grid(row=7, column=0, columnspan=2, sticky=tk.NSEW, padx=5, pady=5)
    
    # Checkbox to enable/disable custom endpoint (MOVED TO TOP)
    custom_endpoint_checkbox_frame = tb.Frame(endpoints_frame)
    custom_endpoint_checkbox_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

    self.use_custom_endpoint_checkbox = tb.Checkbutton(
        custom_endpoint_checkbox_frame,
        text="Enable Custom OpenAI Endpoint",
        variable=self.use_custom_openai_endpoint_var,
        command=self.toggle_custom_endpoint_ui,
        bootstyle="primary"
    )
    self.use_custom_endpoint_checkbox.pack(side=tk.LEFT)

    # Main OpenAI Base URL
    openai_url_frame = tb.Frame(endpoints_frame)
    openai_url_frame.pack(fill=tk.X, padx=5, pady=5)

    tb.Label(openai_url_frame, text="Override API Endpoint:").pack(side=tk.LEFT, padx=(0, 5))
    self.openai_base_url_var = tk.StringVar(value=self.config.get('openai_base_url', ''))
    self.openai_base_url_entry = tb.Entry(openai_url_frame, textvariable=self.openai_base_url_var, width=50)
    self.openai_base_url_var.trace('w', self._check_azure_endpoint)
    self.openai_base_url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    
    # Clear button
    self.openai_clear_button = tb.Button(openai_url_frame, text="Clear", 
             command=lambda: self.openai_base_url_var.set(""),
             bootstyle="secondary", width=8)
    self.openai_clear_button.pack(side=tk.LEFT)
    
    # Set initial state based on checkbox
    if not self.use_custom_openai_endpoint_var.get():
        self.openai_base_url_entry.configure(state='disabled')
        self.openai_clear_button.configure(state='disabled')

    # Help text for main field
    help_text = tb.Label(endpoints_frame, 
                        text="Enable checkbox to use custom endpoint. For Ollama: http://localhost:11434/v1",
                        font=('TkDefaultFont', 8), foreground='gray')
    help_text.pack(anchor=tk.W, padx=5, pady=(0, 10))
    
    # ADD AZURE VERSION FRAME HERE (initially hidden):
    self.azure_version_frame = tb.Frame(endpoints_frame)
    # Don't pack it yet - it will be shown/hidden dynamically
    
    tb.Label(self.azure_version_frame, text="Azure API Version:").pack(side=tk.LEFT, padx=(5, 5))
    
    # Update the existing azure_api_version_var with current config and add trace
    self.azure_api_version_var.set(self.config.get('azure_api_version', '2024-08-01-preview'))
    # Add trace to update env var immediately when changed
    self.azure_api_version_var.trace('w', self._update_azure_api_version_env)
    versions = [
        '2025-01-01-preview',  # Latest preview
        '2024-12-01-preview',
        '2024-10-01-preview', 
        '2024-08-01-preview',  # Current default
        '2024-06-01',         # Stable release
        '2024-05-01-preview',
        '2024-04-01-preview',
        '2024-02-01',         # Older stable
        '2023-12-01-preview',
        '2023-10-01-preview',
        '2023-05-15'          # Legacy
    ]
    self.azure_version_combo = ttk.Combobox(
        self.azure_version_frame, 
        textvariable=self.azure_api_version_var,
        values=versions,
        width=20,
        state='normal'
    )
    self.azure_version_combo.pack(side=tk.LEFT, padx=(0, 5))

    # Show More Fields button
    self.show_more_endpoints = False
    self.more_fields_button = tb.Button(endpoints_frame, 
                                       text="▼ Show More Fields", 
                                       command=self.toggle_more_endpoints,
                                       bootstyle="link")
    self.more_fields_button.pack(anchor=tk.W, padx=5, pady=5)

    # Container for additional fields (initially hidden)
    self.additional_endpoints_frame = tb.Frame(endpoints_frame)
    # Don't pack it initially - it's hidden

    # Inside the additional_endpoints_frame:
    # Groq/Local Base URL
    groq_url_frame = tb.Frame(self.additional_endpoints_frame)
    groq_url_frame.pack(fill=tk.X, padx=5, pady=5)

    tb.Label(groq_url_frame, text="Groq/Local Base URL:").pack(side=tk.LEFT, padx=(0, 5))
    self.groq_base_url_var = tk.StringVar(value=self.config.get('groq_base_url', ''))
    self.groq_base_url_entry = tb.Entry(groq_url_frame, textvariable=self.groq_base_url_var, width=50)
    self.groq_base_url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    tb.Button(groq_url_frame, text="Clear", 
             command=lambda: self.groq_base_url_var.set(""),
             bootstyle="secondary", width=8).pack(side=tk.LEFT)

    groq_help = tb.Label(self.additional_endpoints_frame, 
                        text="For vLLM: http://localhost:8000/v1 | For LM Studio: http://localhost:1234/v1",
                        font=('TkDefaultFont', 8), foreground='gray')
    groq_help.pack(anchor=tk.W, padx=5, pady=(0, 5))

    # Fireworks Base URL
    fireworks_url_frame = tb.Frame(self.additional_endpoints_frame)
    fireworks_url_frame.pack(fill=tk.X, padx=5, pady=5)

    tb.Label(fireworks_url_frame, text="Fireworks Base URL:").pack(side=tk.LEFT, padx=(0, 5))
    self.fireworks_base_url_var = tk.StringVar(value=self.config.get('fireworks_base_url', ''))
    self.fireworks_base_url_entry = tb.Entry(fireworks_url_frame, textvariable=self.fireworks_base_url_var, width=50)
    self.fireworks_base_url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    tb.Button(fireworks_url_frame, text="Clear", 
             command=lambda: self.fireworks_base_url_var.set(""),
             bootstyle="secondary", width=8).pack(side=tk.LEFT)

    # Info about multiple endpoints
    info_frame = tb.Frame(self.additional_endpoints_frame)
    info_frame.pack(fill=tk.X, padx=5, pady=10)

    info_text = """💡 Advanced: Use multiple endpoints to run different local LLM servers simultaneously.
    • Use model prefix 'groq/' to route through Groq endpoint
    • Use model prefix 'fireworks/' to route through Fireworks endpoint
    • Most users only need the main OpenAI endpoint above"""

    tb.Label(info_frame, text=info_text, 
            font=('TkDefaultFont', 8), foreground='#0dcaf0',  # Light blue color
            wraplength=600, justify=tk.LEFT).pack(anchor=tk.W)

    # Test Connection button (always visible)
    test_button = tb.Button(endpoints_frame, text="Test Connection", 
                           command=self.test_api_connections,
                           bootstyle="info")
    test_button.pack(pady=10)

    # Gemini OpenAI-Compatible Endpoint (inside additional_endpoints_frame)
    gemini_frame = tb.Frame(self.additional_endpoints_frame)
    gemini_frame.pack(fill=tk.X, padx=5, pady=5)

    # Checkbox for enabling Gemini endpoint
    self.gemini_checkbox = tb.Checkbutton(
        gemini_frame,
        text="Enable Gemini OpenAI-Compatible Endpoint",
        variable=self.use_gemini_openai_endpoint_var,
        command=self.toggle_gemini_endpoint,  # Add the command
        bootstyle="primary"
    )
    self.gemini_checkbox.pack(anchor=tk.W, pady=(5, 5))

    # Gemini endpoint URL input
    gemini_url_frame = tb.Frame(self.additional_endpoints_frame)
    gemini_url_frame.pack(fill=tk.X, padx=5, pady=5)

    tb.Label(gemini_url_frame, text="Gemini OpenAI Endpoint:").pack(side=tk.LEFT, padx=(0, 5))
    self.gemini_endpoint_entry = tb.Entry(gemini_url_frame, textvariable=self.gemini_openai_endpoint_var, width=50)
    self.gemini_endpoint_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    self.gemini_clear_button = tb.Button(gemini_url_frame, text="Clear", 
             command=lambda: self.gemini_openai_endpoint_var.set(""),
             bootstyle="secondary", width=8)
    self.gemini_clear_button.pack(side=tk.LEFT)

    # Help text
    gemini_help = tb.Label(self.additional_endpoints_frame, 
                          text="For Gemini rate limit optimization with proxy services (e.g., OpenRouter, LiteLLM)",
                          font=('TkDefaultFont', 8), foreground='gray')
    gemini_help.pack(anchor=tk.W, padx=5, pady=(0, 5))

    # Set initial state based on checkbox
    if not self.use_gemini_openai_endpoint_var.get():
        self.gemini_endpoint_entry.configure(state='disabled')
        self.gemini_clear_button.configure(state='disabled')

def _check_azure_endpoint(self, *args):
    """Check if endpoint is Azure and update UI"""
    if not self.use_custom_openai_endpoint_var.get():
        if hasattr(self, 'azure_version_frame'):
            self.azure_version_frame.pack_forget()
        return
        
    url = self.openai_base_url_var.get()
    if '.azure.com' in url or '.cognitiveservices' in url:
        self.api_key_label.config(text="Azure Key:")
        
        # Show Azure version frame in settings dialog
        if hasattr(self, 'azure_version_frame'):
            self.azure_version_frame.pack(before=self.more_fields_button, pady=(0, 10))
    else:
        self.api_key_label.config(text="OpenAI/Gemini/... API Key:")
        
        # Hide Azure version frame
        if hasattr(self, 'azure_version_frame'):
            self.azure_version_frame.pack_forget()
            
def _update_azure_api_version_env(self, *args):
    """Update the AZURE_API_VERSION environment variable when the setting changes"""
    try:
        api_version = self.azure_api_version_var.get()
        if api_version:
            os.environ['AZURE_API_VERSION'] = api_version
            #print(f"✅ Updated Azure API Version in environment: {api_version}")
    except Exception as e:
        print(f"❌ Error updating Azure API Version environment variable: {e}")

def toggle_gemini_endpoint(self):
    """Enable/disable Gemini endpoint entry based on toggle"""
    if self.use_gemini_openai_endpoint_var.get():
        self.gemini_endpoint_entry.configure(state='normal')
        self.gemini_clear_button.configure(state='normal')
    else:
        self.gemini_endpoint_entry.configure(state='disabled')
        self.gemini_clear_button.configure(state='disabled')
    
def toggle_custom_endpoint_ui(self):
    """Enable/disable the OpenAI base URL entry and detect Azure"""
    if self.use_custom_openai_endpoint_var.get():
        self.openai_base_url_entry.configure(state='normal')
        self.openai_clear_button.configure(state='normal')
        
        # Check if it's Azure
        url = self.openai_base_url_var.get()
        if '.azure.com' in url or '.cognitiveservices' in url:
            self.api_key_label.config(text="Azure Key:")
        else:
            self.api_key_label.config(text="OpenAI/Gemini/... API Key:")
            
        print("✅ Custom OpenAI endpoint enabled")
    else:
        self.openai_base_url_entry.configure(state='disabled')
        self.openai_clear_button.configure(state='disabled')
        self.api_key_label.config(text="OpenAI/Gemini/... API Key:")
        print("❌ Custom OpenAI endpoint disabled - using default OpenAI API")

def toggle_more_endpoints(self):
    """Toggle visibility of additional endpoint fields"""
    self.show_more_endpoints = not self.show_more_endpoints
    
    if self.show_more_endpoints:
        self.additional_endpoints_frame.pack(fill=tk.BOTH, expand=True, after=self.more_fields_button)
        self.more_fields_button.configure(text="▲ Show Fewer Fields")
    else:
        self.additional_endpoints_frame.pack_forget()
        self.more_fields_button.configure(text="▼ Show More Fields")
    
    # Update dialog scrolling if needed
    if hasattr(self, 'current_dialog') and self.current_dialog:
        self.current_dialog.update_idletasks()
        self.current_dialog.canvas.configure(scrollregion=self.current_dialog.canvas.bbox("all"))
             
def test_api_connections(self):
    """Test all configured API connections"""
    # Show immediate feedback
    progress_dialog = tk.Toplevel(self.current_dialog if hasattr(self, 'current_dialog') else self.master)
    progress_dialog.title("Testing Connections...")
    
    # Set icon
    try:
        progress_dialog.iconbitmap("halgakos.ico")
    except:
        pass  # Icon setting failed, continue without icon
    
    # Center the dialog
    progress_dialog.update_idletasks()
    width = 300
    height = 150
    x = (progress_dialog.winfo_screenwidth() // 2) - (width // 2)
    y = (progress_dialog.winfo_screenheight() // 2) - (height // 2)
    progress_dialog.geometry(f"{width}x{height}+{x}+{y}")
    
    # Add progress message
    progress_label = tb.Label(progress_dialog, text="Testing API connections...\nPlease wait...", 
                             font=('TkDefaultFont', 10))
    progress_label.pack(pady=50)
    
    # Force update to show dialog immediately
    progress_dialog.update()
    
    try:
        # Ensure we have the openai module
        import openai
    except ImportError:
        progress_dialog.destroy()
        messagebox.showerror("Error", "OpenAI library not installed")
        return
    
    # Get API key from the main GUI
    api_key = self.api_key_entry.get() if hasattr(self, 'api_key_entry') else self.config.get('api_key', '')
    if not api_key:
        api_key = "sk-dummy-key"  # For local models
    
    # Collect all configured endpoints
    endpoints_to_test = []
    
    # OpenAI endpoint - only test if checkbox is enabled
    if self.use_custom_openai_endpoint_var.get():
        openai_url = self.openai_base_url_var.get()
        if openai_url:
            # Check if it's Azure
            if '.azure.com' in openai_url or '.cognitiveservices' in openai_url:
                # Azure endpoint
                deployment = self.model_var.get() if hasattr(self, 'model_var') else "gpt-35-turbo"
                api_version = self.azure_api_version_var.get() if hasattr(self, 'azure_api_version_var') else "2024-08-01-preview"
                
                # Format Azure URL
                if '/openai/deployments/' not in openai_url:
                    azure_url = f"{openai_url.rstrip('/')}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
                else:
                    azure_url = openai_url
                
                endpoints_to_test.append(("Azure OpenAI", azure_url, deployment, "azure"))
            else:
                # Regular custom endpoint
                endpoints_to_test.append(("OpenAI (Custom)", openai_url, self.model_var.get() if hasattr(self, 'model_var') else "gpt-3.5-turbo"))
        else:
            # Use default OpenAI endpoint if checkbox is on but no custom URL provided
            endpoints_to_test.append(("OpenAI (Default)", "https://api.openai.com/v1", self.model_var.get() if hasattr(self, 'model_var') else "gpt-3.5-turbo"))
    
    # Groq endpoint
    if hasattr(self, 'groq_base_url_var'):
        groq_url = self.groq_base_url_var.get()
        if groq_url:
            # For Groq, we need a groq-prefixed model
            current_model = self.model_var.get() if hasattr(self, 'model_var') else "llama-3-70b"
            groq_model = current_model if current_model.startswith('groq/') else current_model.replace('groq/', '')
            endpoints_to_test.append(("Groq/Local", groq_url, groq_model))
    
    # Fireworks endpoint
    if hasattr(self, 'fireworks_base_url_var'):
        fireworks_url = self.fireworks_base_url_var.get()
        if fireworks_url:
            # For Fireworks, we need the accounts/ prefix
            current_model = self.model_var.get() if hasattr(self, 'model_var') else "llama-v3-70b-instruct"
            fw_model = current_model if current_model.startswith('accounts/') else f"accounts/fireworks/models/{current_model.replace('fireworks/', '')}"
            endpoints_to_test.append(("Fireworks", fireworks_url, fw_model))
    
    # Gemini OpenAI-Compatible endpoint
    if hasattr(self, 'use_gemini_openai_endpoint_var') and self.use_gemini_openai_endpoint_var.get():
        gemini_url = self.gemini_openai_endpoint_var.get()
        if gemini_url:
            # Ensure the endpoint ends with /openai/ for compatibility
            if not gemini_url.endswith('/openai/'):
                if gemini_url.endswith('/'):
                    gemini_url = gemini_url + 'openai/'
                else:
                    gemini_url = gemini_url + '/openai/'
            
            # For Gemini OpenAI-compatible endpoints, use the current model or a suitable default
            current_model = self.model_var.get() if hasattr(self, 'model_var') else "gemini-2.0-flash-exp"
            # Remove any 'gemini/' prefix for the OpenAI-compatible endpoint
            gemini_model = current_model.replace('gemini/', '') if current_model.startswith('gemini/') else current_model
            endpoints_to_test.append(("Gemini (OpenAI-Compatible)", gemini_url, gemini_model))
    
    if not endpoints_to_test:
        messagebox.showinfo("Info", "No custom endpoints configured. Using default API endpoints.")
        return
    
    # Test each endpoint
    # Test each endpoint
    results = []
    for endpoint_info in endpoints_to_test:
        if len(endpoint_info) == 4 and endpoint_info[3] == "azure":
            # Azure endpoint
            name, base_url, model, endpoint_type = endpoint_info
            try:
                # Azure uses different headers
                import requests
                headers = {
                    "api-key": api_key,
                    "Content-Type": "application/json"
                }
                
                response = requests.post(
                    base_url,
                    headers=headers,
                    json={
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 5
                    },
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    results.append(f"✅ {name}: Connected successfully! (Deployment: {model})")
                else:
                    results.append(f"❌ {name}: {response.status_code} - {response.text[:100]}")
                    
            except Exception as e:
                error_msg = str(e)[:100]
                results.append(f"❌ {name}: {error_msg}")
        else:
            # Regular OpenAI-compatible endpoint
            name, base_url, model = endpoint_info[:3]
            try:
                # Create client for this endpoint
                test_client = openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=5.0  # Short timeout for testing
                )
                
                # Try a minimal completion
                response = test_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5
                )
                
                results.append(f"✅ {name}: Connected successfully! (Model: {model})")
            except Exception as e:
                error_msg = str(e)
                # Simplify common error messages
                if "404" in error_msg:
                    error_msg = "404 - Endpoint not found. Check URL and model name."
                elif "401" in error_msg or "403" in error_msg:
                    error_msg = "Authentication failed. Check API key."
                elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                    error_msg = f"Model '{model}' not found at this endpoint."
                
                results.append(f"❌ {name}: {error_msg}")
    
    # Show results
    result_message = "Connection Test Results:\n\n" + "\n\n".join(results)
    
    # Close progress dialog
    progress_dialog.destroy()
    
    # Determine if all succeeded
    all_success = all("✅" in r for r in results)
    
    if all_success:
        messagebox.showinfo("Success", result_message)
    else:
        messagebox.showwarning("Test Results", result_message)
    
def _create_settings_buttons(self, parent, dialog, canvas):
    """Create save and close buttons for settings dialog"""
    button_frame = tk.Frame(parent)
    button_frame.grid(row=3, column=0, columnspan=2, pady=(10, 10))
    
    button_container = tk.Frame(button_frame)
    button_container.pack(expand=True)

    def save_and_close():
        try:
            # Use the main save_config method which handles all config AND environment variables
            # This ensures consistency and reduces duplicate code
            self.save_config(show_message=False)
            
            # Log success and close dialog
            self.append_log("✅ Other Settings saved successfully")
            dialog.destroy()
            
        except Exception as e:
            print(f"❌ Failed to save Other Settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    tb.Button(button_container, text="💾 Save Settings", command=save_and_close, 
             bootstyle="success", width=20).pack(side=tk.LEFT, padx=5)

    tb.Button(button_container, text="❌ Cancel", command=lambda: [dialog._cleanup_scrolling(), dialog.destroy()], 
             bootstyle="secondary", width=20).pack(side=tk.LEFT, padx=5)       

def delete_translated_headers_file(self):
    """Delete the translated_headers.txt file from the output directory for all selected EPUBs"""
    try:
        # Get all selected EPUB files using the same logic as QA scanner
        epub_files_to_process = []
        
        # First check if current selection actually contains EPUBs
        if hasattr(self, 'selected_files') and self.selected_files:
            current_epub_files = [f for f in self.selected_files if f.lower().endswith('.epub')]
            if current_epub_files:
                epub_files_to_process = current_epub_files
                self.append_log(f"📚 Found {len(epub_files_to_process)} EPUB files in current selection")
        
        # If no EPUBs in selection, try single EPUB methods
        if not epub_files_to_process:
            epub_path = self.get_current_epub_path()
            if not epub_path:
                entry_path = self.entry_epub.get().strip()
                if entry_path and entry_path != "No file selected" and os.path.exists(entry_path):
                    epub_path = entry_path
            
            if epub_path:
                epub_files_to_process = [epub_path]
        
        if not epub_files_to_process:
            messagebox.showerror("Error", "No EPUB file(s) selected. Please select EPUB file(s) first.")
            return
        
        # Process each EPUB file to find and delete translated_headers.txt
        files_found = []
        files_not_found = []
        files_deleted = []
        errors = []
        
        current_dir = os.getcwd()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # First pass: scan for files
        for epub_path in epub_files_to_process:
            try:
                epub_base = os.path.splitext(os.path.basename(epub_path))[0]
                self.append_log(f"🔍 Processing EPUB: {epub_base}")
                
                # Check the most common locations in order of priority (same as QA scanner)
                candidates = [
                    os.path.join(current_dir, epub_base),        # current working directory
                    os.path.join(script_dir, epub_base),         # src directory (where output typically goes)
                    os.path.join(current_dir, 'src', epub_base), # src subdirectory from current dir
                ]
                
                output_dir = None
                for candidate in candidates:
                    if os.path.isdir(candidate):
                        # Verify the folder actually contains HTML/XHTML files
                        try:
                            files = os.listdir(candidate)
                            html_files = [f for f in files if f.lower().endswith(('.html', '.xhtml', '.htm'))]
                            if html_files:
                                output_dir = candidate
                                break
                        except Exception:
                            continue
                
                if not output_dir:
                    self.append_log(f"  ⚠️ No output directory found for {epub_base}")
                    files_not_found.append((epub_base, "No output directory found"))
                    continue
                
                # Look for translated_headers.txt in the output directory
                headers_file = os.path.join(output_dir, "translated_headers.txt")
                
                if os.path.exists(headers_file):
                    files_found.append((epub_base, headers_file))
                    self.append_log(f"  ✓ Found translated_headers.txt in {os.path.basename(output_dir)}")
                else:
                    files_not_found.append((epub_base, "translated_headers.txt not found"))
                    self.append_log(f"  ⚠️ No translated_headers.txt in {os.path.basename(output_dir)}")
                    
            except Exception as e:
                epub_base = os.path.splitext(os.path.basename(epub_path))[0]
                errors.append((epub_base, str(e)))
                self.append_log(f"  ❌ Error processing {epub_base}: {e}")
        
        # Show summary and get user confirmation
        if not files_found and not files_not_found and not errors:
            messagebox.showinfo("No Files", "No EPUB files were processed.")
            return
        
        summary_text = f"Summary for {len(epub_files_to_process)} EPUB file(s):\n\n"
        
        if files_found:
            summary_text += f"✅ Files to delete ({len(files_found)}):\n"
            for epub_base, file_path in files_found:
                summary_text += f"  • {epub_base}\n"
            summary_text += "\n"
        
        if files_not_found:
            summary_text += f"⚠️ Files not found ({len(files_not_found)}):\n"
            for epub_base, reason in files_not_found:
                summary_text += f"  • {epub_base}: {reason}\n"
            summary_text += "\n"
        
        if errors:
            summary_text += f"❌ Errors ({len(errors)}):\n"
            for epub_base, error in errors:
                summary_text += f"  • {epub_base}: {error}\n"
            summary_text += "\n"
        
        if files_found:
            summary_text += "This will allow headers to be re-translated on the next run."
            
            # Confirm deletion
            result = messagebox.askyesno("Confirm Deletion", summary_text)
            
            if result:
                # Delete the files
                for epub_base, headers_file in files_found:
                    try:
                        os.remove(headers_file)
                        files_deleted.append(epub_base)
                        self.append_log(f"✅ Deleted translated_headers.txt from {epub_base}")
                    except Exception as e:
                        errors.append((epub_base, f"Delete failed: {e}"))
                        self.append_log(f"❌ Failed to delete translated_headers.txt from {epub_base}: {e}")
                
                # Show final results
                if files_deleted:
                    success_msg = f"Successfully deleted {len(files_deleted)} file(s):\n"
                    success_msg += "\n".join([f"• {epub_base}" for epub_base in files_deleted])
                    if errors:
                        success_msg += f"\n\nErrors: {len(errors)} file(s) failed to delete."
                    messagebox.showinfo("Success", success_msg)
                else:
                    messagebox.showerror("Error", "No files were successfully deleted.")
        else:
            # No files to delete
            messagebox.showinfo("No Files to Delete", summary_text)
        
    except Exception as e:
        self.append_log(f"❌ Error deleting translated_headers.txt: {e}")
        messagebox.showerror("Error", f"Failed to delete file: {e}")

def validate_epub_structure_gui(self):
    """GUI wrapper for EPUB structure validation"""
    input_path = self.entry_epub.get()
    if not input_path:
        messagebox.showerror("Error", "Please select a file first.")
        return
    
    if input_path.lower().endswith('.txt'):
        messagebox.showinfo("Info", "Structure validation is only available for EPUB files.")
        return
    
    epub_base = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = epub_base
    
    if not os.path.exists(output_dir):
        messagebox.showinfo("Info", f"No output directory found: {output_dir}")
        return
    
    self.append_log("🔍 Validating EPUB structure...")
    
    try:
        from TransateKRtoEN import validate_epub_structure, check_epub_readiness
        
        structure_ok = validate_epub_structure(output_dir)
        readiness_ok = check_epub_readiness(output_dir)
        
        if structure_ok and readiness_ok:
            self.append_log("✅ EPUB validation PASSED - Ready for compilation!")
            messagebox.showinfo("Validation Passed", 
                              "✅ All EPUB structure files are present!\n\n"
                              "Your translation is ready for EPUB compilation.")
        elif structure_ok:
            self.append_log("⚠️ EPUB structure OK, but some issues found")
            messagebox.showwarning("Validation Warning", 
                                 "⚠️ EPUB structure is mostly OK, but some issues were found.\n\n"
                                 "Check the log for details.")
        else:
            self.append_log("❌ EPUB validation FAILED - Missing critical files")
            messagebox.showerror("Validation Failed", 
                               "❌ Missing critical EPUB files!\n\n"
                               "container.xml and/or OPF files are missing.\n"
                               "Try re-running the translation to extract them.")
    
    except ImportError as e:
        self.append_log(f"❌ Could not import validation functions: {e}")
        messagebox.showerror("Error", "Validation functions not available.")
    except Exception as e:
        self.append_log(f"❌ Validation error: {e}")
        messagebox.showerror("Error", f"Validation failed: {e}")

def on_profile_select(self, event=None):
    """Load the selected profile's prompt into the text area."""
    name = self.profile_var.get()
    prompt = self.prompt_profiles.get(name, "")
    self.prompt_text.delete("1.0", tk.END)
    self.prompt_text.insert("1.0", prompt)
    self.config['active_profile'] = name

def save_profile(self):
    """Save current prompt under selected profile and persist."""
    name = self.profile_var.get().strip()
    if not name:
        messagebox.showerror("Error", "Profile cannot be empty.")
        return
    content = self.prompt_text.get('1.0', tk.END).strip()
    self.prompt_profiles[name] = content
    self.config['prompt_profiles'] = self.prompt_profiles
    self.config['active_profile'] = name
    self.profile_menu['values'] = list(self.prompt_profiles.keys())
    messagebox.showinfo("Saved", f"Profile '{name}' saved.")
    self.save_profiles()

def delete_profile(self):
    """Delete the selected profile."""
    name = self.profile_var.get()
    if name not in self.prompt_profiles:
        messagebox.showerror("Error", f"Profile '{name}' not found.")
        return
    if messagebox.askyesno("Delete", f"Are you sure you want to delete language '{name}'?"):
        del self.prompt_profiles[name]
        self.config['prompt_profiles'] = self.prompt_profiles
        if self.prompt_profiles:
            new = next(iter(self.prompt_profiles))
            self.profile_var.set(new)
            self.on_profile_select()
        else:
            self.profile_var.set("")
            self.prompt_text.delete('1.0', tk.END)
        self.profile_menu['values'] = list(self.prompt_profiles.keys())
        self.save_profiles()

def save_profiles(self):
    """Persist only the prompt profiles and active profile."""
    try:
        data = {}
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        data['prompt_profiles'] = self.prompt_profiles
        data['active_profile'] = self.profile_var.get()
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save profiles: {e}")

def import_profiles(self):
    """Import profiles from a JSON file, merging into existing ones."""
    path = filedialog.askopenfilename(title="Import Profiles", filetypes=[("JSON files","*.json")])
    if not path:
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.prompt_profiles.update(data)
        self.config['prompt_profiles'] = self.prompt_profiles
        self.profile_menu['values'] = list(self.prompt_profiles.keys())
        messagebox.showinfo("Imported", f"Imported {len(data)} profiles.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to import profiles: {e}")

def export_profiles(self):
    """Export all profiles to a JSON file."""
    path = filedialog.asksaveasfilename(title="Export Profiles", defaultextension=".json", 
                                      filetypes=[("JSON files","*.json")])
    if not path:
        return
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.prompt_profiles, f, ensure_ascii=False, indent=2)
        messagebox.showinfo("Exported", f"Profiles exported to {path}.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to export profiles: {e}")