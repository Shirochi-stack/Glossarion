"""
Glossary Manager GUI Module
Comprehensive glossary management for automatic and manual glossary extraction
"""

import os
import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as tb

# Import from translator_gui if available
try:
    from translator_gui import WindowManager, UIHelper
except ImportError:
    WindowManager = None
    UIHelper = None


class GlossaryManagerMixin:
    """Mixin class containing glossary management methods for TranslatorGUI"""

    def glossary_manager(self):
        """Open comprehensive glossary management dialog"""
        # Create scrollable dialog (stays hidden)
        dialog, scrollable_frame, canvas = self.wm.setup_scrollable(
            self.master, 
            "Glossary Manager",
            width=0,  # Will be auto-sized
            height=None,
            max_width_ratio=0.9,
            max_height_ratio=0.85
        )
        
        # Create notebook for tabs
        notebook = ttk.Notebook(scrollable_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create and add tabs
        tabs = [
            ("Manual Glossary Extraction", self._setup_manual_glossary_tab),
            ("Automatic Glossary Generation", self._setup_auto_glossary_tab),
            ("Glossary Editor", self._setup_glossary_editor_tab)
        ]
        
        for tab_name, setup_method in tabs:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=tab_name)
            setup_method(frame)
        
        # Dialog Controls
        control_frame = tk.Frame(dialog)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def save_glossary_settings():
            try:
                # Update prompts from text widgets
                self.update_glossary_prompts()
                
                # Save custom fields
                self.config['custom_glossary_fields'] = self.custom_glossary_fields
                
                # Update enabled status from checkboxes
                if hasattr(self, 'type_enabled_vars'):
                    for type_name, var in self.type_enabled_vars.items():
                        if type_name in self.custom_entry_types:
                            self.custom_entry_types[type_name]['enabled'] = var.get()
                
                # Save custom entry types
                self.config['custom_entry_types'] = self.custom_entry_types
                
                # Save all glossary-related settings
                self.config['enable_auto_glossary'] = self.enable_auto_glossary_var.get()
                self.config['append_glossary'] = self.append_glossary_var.get()
                self.config['glossary_min_frequency'] = int(self.glossary_min_frequency_var.get())
                self.config['glossary_max_names'] = int(self.glossary_max_names_var.get())
                self.config['glossary_max_titles'] = int(self.glossary_max_titles_var.get())
                self.config['glossary_batch_size'] = int(self.glossary_batch_size_var.get())
                self.config['glossary_format_instructions'] = getattr(self, 'glossary_format_instructions', '')
                self.config['glossary_max_text_size'] = self.glossary_max_text_size_var.get()
                self.config['glossary_max_sentences'] = int(self.glossary_max_sentences_var.get())

                
                # Honorifics and other settings
                if hasattr(self, 'strip_honorifics_var'):
                    self.config['strip_honorifics'] = self.strip_honorifics_var.get()
                if hasattr(self, 'disable_honorifics_var'):
                    self.config['glossary_disable_honorifics_filter'] = self.disable_honorifics_var.get()
                
                # Save format preference
                if hasattr(self, 'use_legacy_csv_var'):
                    self.config['glossary_use_legacy_csv'] = self.use_legacy_csv_var.get()
                    
                # Temperature and context limit
                try:
                    self.config['manual_glossary_temperature'] = float(self.manual_temp_var.get())
                    self.config['manual_context_limit'] = int(self.manual_context_var.get())
                except ValueError:
                    messagebox.showwarning("Invalid Input", 
                        "Please enter valid numbers for temperature and context limit")
                    return
                
                # Fuzzy matching threshold
                self.config['glossary_fuzzy_threshold'] = self.fuzzy_threshold_var.get()
                
                # Save prompts
                self.config['manual_glossary_prompt'] = self.manual_glossary_prompt
                self.config['auto_glossary_prompt'] = self.auto_glossary_prompt
                self.config['append_glossary_prompt'] = self.append_glossary_prompt
                self.config['glossary_translation_prompt'] = getattr(self, 'glossary_translation_prompt', '')
                
                # Update environment variables for immediate use
                os.environ['GLOSSARY_SYSTEM_PROMPT'] = self.manual_glossary_prompt
                os.environ['AUTO_GLOSSARY_PROMPT'] = self.auto_glossary_prompt
                os.environ['GLOSSARY_DISABLE_HONORIFICS_FILTER'] = '1' if self.disable_honorifics_var.get() else '0'
                os.environ['GLOSSARY_STRIP_HONORIFICS'] = '1' if self.strip_honorifics_var.get() else '0'
                os.environ['GLOSSARY_FUZZY_THRESHOLD'] = str(self.fuzzy_threshold_var.get())
                os.environ['GLOSSARY_TRANSLATION_PROMPT'] = getattr(self, 'glossary_translation_prompt', '')
                os.environ['GLOSSARY_FORMAT_INSTRUCTIONS'] = getattr(self, 'glossary_format_instructions', '')
                os.environ['GLOSSARY_USE_LEGACY_CSV'] = '1' if self.use_legacy_csv_var.get() else '0'
                os.environ['GLOSSARY_MAX_SENTENCES'] = str(self.glossary_max_sentences_var.get())
                
                # Set custom entry types and fields as environment variables
                os.environ['GLOSSARY_CUSTOM_ENTRY_TYPES'] = json.dumps(self.custom_entry_types)
                if self.custom_glossary_fields:
                    os.environ['GLOSSARY_CUSTOM_FIELDS'] = json.dumps(self.custom_glossary_fields)
                
                # Save config using the main save_config method to ensure encryption
                self.save_config(show_message=False)
                
                self.append_log("✅ Glossary settings saved successfully")
                
                # Check if any types are enabled
                enabled_types = [t for t, cfg in self.custom_entry_types.items() if cfg.get('enabled', True)]
                if not enabled_types:
                    messagebox.showwarning("Warning", "No entry types selected! The glossary extraction will not find any entries.")
                else:
                    self.append_log(f"📑 Enabled types: {', '.join(enabled_types)}")
                
                messagebox.showinfo("Success", "Glossary settings saved!")
                dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save settings: {e}")
                self.append_log(f"❌ Failed to save glossary settings: {e}")
                
        # Create button container
        button_container = tk.Frame(control_frame)
        button_container.pack(expand=True)
        
        # Add buttons
        tb.Button(
            button_container, 
            text="Save All Settings", 
            command=save_glossary_settings, 
            bootstyle="success", 
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        tb.Button(
            button_container, 
            text="Cancel", 
            command=lambda: [dialog._cleanup_scrolling(), dialog.destroy()], 
            bootstyle="secondary", 
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        # Auto-resize and show
        self.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.9, max_height_ratio=1.5)
        
        dialog.protocol("WM_DELETE_WINDOW", 
                       lambda: [dialog._cleanup_scrolling(), dialog.destroy()])

    def _setup_manual_glossary_tab(self, parent):
        """Setup manual glossary tab - simplified for new format"""
        manual_container = tk.Frame(parent)
        manual_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Type filtering section with custom types
        type_filter_frame = tk.LabelFrame(manual_container, text="Entry Type Configuration", padx=10, pady=10)
        type_filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Initialize custom entry types if not exists
        if not hasattr(self, 'custom_entry_types'):
            # Default types with their enabled status
            self.custom_entry_types = self.config.get('custom_entry_types', {
                'character': {'enabled': True, 'has_gender': True},
                'term': {'enabled': True, 'has_gender': False}
            })
        
        # Main container with grid for better control
        type_main_container = tk.Frame(type_filter_frame)
        type_main_container.pack(fill=tk.X)
        type_main_container.grid_columnconfigure(0, weight=3)  # Left side gets 3/5 of space
        type_main_container.grid_columnconfigure(1, weight=2)  # Right side gets 2/5 of space
        
        # Left side - type list with checkboxes
        type_list_frame = tk.Frame(type_main_container)
        type_list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        
        tk.Label(type_list_frame, text="Active Entry Types:",
                font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W)
        
        # Scrollable frame for type checkboxes
        type_scroll_frame = tk.Frame(type_list_frame)
        type_scroll_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        type_canvas = tk.Canvas(type_scroll_frame, height=150)
        type_scrollbar = ttk.Scrollbar(type_scroll_frame, orient="vertical", command=type_canvas.yview)
        self.type_checkbox_frame = tk.Frame(type_canvas)
        
        type_canvas.configure(yscrollcommand=type_scrollbar.set)
        type_canvas_window = type_canvas.create_window((0, 0), window=self.type_checkbox_frame, anchor="nw")
        
        type_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        type_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Store checkbox variables
        self.type_enabled_vars = {}
        
        def update_type_checkboxes():
            """Rebuild the checkbox list"""
            # Clear existing checkboxes
            for widget in self.type_checkbox_frame.winfo_children():
                widget.destroy()
            
            # Sort types: built-in first, then custom alphabetically
            sorted_types = sorted(self.custom_entry_types.items(), 
                                key=lambda x: (x[0] not in ['character', 'term'], x[0]))
            
            # Create checkboxes for each type
            for type_name, type_config in sorted_types:
                var = tk.BooleanVar(value=type_config.get('enabled', True))
                self.type_enabled_vars[type_name] = var
                
                frame = tk.Frame(self.type_checkbox_frame)
                frame.pack(fill=tk.X, pady=2)
                
                # Checkbox
                cb = tb.Checkbutton(frame, text=type_name, variable=var,
                                  bootstyle="round-toggle")
                cb.pack(side=tk.LEFT)
                
                # Add gender indicator for types that support it
                if type_config.get('has_gender', False):
                    tk.Label(frame, text="(has gender field)", 
                            font=('TkDefaultFont', 9), fg='gray').pack(side=tk.LEFT, padx=(10, 0))
                
                # Delete button for custom types
                if type_name not in ['character', 'term']:
                    tb.Button(frame, text="×", command=lambda t=type_name: remove_type(t),
                             bootstyle="danger", width=3).pack(side=tk.RIGHT, padx=(5, 0))
            
            # Update canvas scroll region
            self.type_checkbox_frame.update_idletasks()
            type_canvas.configure(scrollregion=type_canvas.bbox("all"))
        
        # Right side - controls for adding custom types
        type_control_frame = tk.Frame(type_main_container)
        type_control_frame.grid(row=0, column=1, sticky="nsew")
        
        tk.Label(type_control_frame, text="Add Custom Type:",
                font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W)
        
        # Entry for new type field
        new_type_frame = tk.Frame(type_control_frame)
        new_type_frame.pack(fill=tk.X, pady=(5, 0))
        
        tk.Label(new_type_frame, text="Type Field:").pack(anchor=tk.W)
        new_type_entry = tb.Entry(new_type_frame)
        new_type_entry.pack(fill=tk.X, pady=(2, 0))
        
        # Checkbox for gender field
        has_gender_var = tk.BooleanVar(value=False)
        tb.Checkbutton(new_type_frame, text="Include gender field", 
                      variable=has_gender_var).pack(anchor=tk.W, pady=(5, 0))
        
        def add_custom_type():
            type_name = new_type_entry.get().strip().lower()
            if not type_name:
                messagebox.showwarning("Invalid Input", "Please enter a type name")
                return
            
            if type_name in self.custom_entry_types:
                messagebox.showwarning("Duplicate Type", f"Type '{type_name}' already exists")
                return
            
            # Add the new type
            self.custom_entry_types[type_name] = {
                'enabled': True,
                'has_gender': has_gender_var.get()
            }
            
            # Clear inputs
            new_type_entry.delete(0, tk.END)
            has_gender_var.set(False)
            
            # Update display
            update_type_checkboxes()
            self.append_log(f"✅ Added custom type: {type_name}")
        
        def remove_type(type_name):
            if type_name in ['character', 'term']:
                messagebox.showwarning("Cannot Remove", "Built-in types cannot be removed")
                return
            
            if messagebox.askyesno("Confirm Removal", f"Remove type '{type_name}'?"):
                del self.custom_entry_types[type_name]
                if type_name in self.type_enabled_vars:
                    del self.type_enabled_vars[type_name]
                update_type_checkboxes()
                self.append_log(f"🗑️ Removed custom type: {type_name}")
        
        tb.Button(new_type_frame, text="Add Type", command=add_custom_type,
                 bootstyle="success").pack(fill=tk.X, pady=(10, 0))
        
        # Initialize checkboxes
        update_type_checkboxes()
        
        # Custom fields section
        custom_frame = tk.LabelFrame(manual_container, text="Custom Fields (Additional Columns)", padx=10, pady=10)
        custom_frame.pack(fill=tk.X, pady=(0, 10))
        
        custom_list_frame = tk.Frame(custom_frame)
        custom_list_frame.pack(fill=tk.X)
        
        tk.Label(custom_list_frame, text="Additional fields to extract (will be added as extra columns):").pack(anchor=tk.W)
        
        custom_scroll = ttk.Scrollbar(custom_list_frame)
        custom_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.custom_fields_listbox = tk.Listbox(custom_list_frame, height=4, 
                                              yscrollcommand=custom_scroll.set)
        self.custom_fields_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        custom_scroll.config(command=self.custom_fields_listbox.yview)
        
        # Initialize custom_glossary_fields if not exists
        if not hasattr(self, 'custom_glossary_fields'):
            self.custom_glossary_fields = self.config.get('custom_glossary_fields', [])
        
        for field in self.custom_glossary_fields:
            self.custom_fields_listbox.insert(tk.END, field)
        
        custom_controls = tk.Frame(custom_frame)
        custom_controls.pack(fill=tk.X, pady=(5, 0))
        
        self.custom_field_entry = tb.Entry(custom_controls, width=30)
        self.custom_field_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        def add_custom_field():
            field = self.custom_field_entry.get().strip()
            if field and field not in self.custom_glossary_fields:
                self.custom_glossary_fields.append(field)
                self.custom_fields_listbox.insert(tk.END, field)
                self.custom_field_entry.delete(0, tk.END)
        
        def remove_custom_field():
            selection = self.custom_fields_listbox.curselection()
            if selection:
                idx = selection[0]
                field = self.custom_fields_listbox.get(idx)
                self.custom_glossary_fields.remove(field)
                self.custom_fields_listbox.delete(idx)
        
        tb.Button(custom_controls, text="Add", command=add_custom_field, width=10).pack(side=tk.LEFT, padx=2)
        tb.Button(custom_controls, text="Remove", command=remove_custom_field, width=10).pack(side=tk.LEFT, padx=2)
        
        # Duplicate Detection Settings
        duplicate_frame = tk.LabelFrame(manual_container, text="Duplicate Detection", padx=10, pady=10)
        duplicate_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Honorifics filter toggle
        if not hasattr(self, 'disable_honorifics_var'):
            self.disable_honorifics_var = tk.BooleanVar(value=self.config.get('glossary_disable_honorifics_filter', False))
        
        tb.Checkbutton(duplicate_frame, text="Disable honorifics filtering", 
                      variable=self.disable_honorifics_var,
                      bootstyle="round-toggle").pack(anchor=tk.W)
        
        tk.Label(duplicate_frame, text="When enabled, honorifics (님, さん, 先生, etc.) will NOT be removed from raw names",
                font=('TkDefaultFont', 9), fg='gray').pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        # Fuzzy matching slider
        fuzzy_frame = tk.Frame(duplicate_frame)
        fuzzy_frame.pack(fill=tk.X, pady=(10, 0))

        tk.Label(fuzzy_frame, text="Fuzzy Matching Threshold:",
                font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W)

        tk.Label(fuzzy_frame, text="Controls how similar names must be to be considered duplicates",
                font=('TkDefaultFont', 9), fg='gray').pack(anchor=tk.W, pady=(0, 5))

        # Slider frame
        slider_frame = tk.Frame(fuzzy_frame)
        slider_frame.pack(fill=tk.X, pady=(5, 0))

        # Initialize fuzzy threshold variable
        if not hasattr(self, 'fuzzy_threshold_var'):
            self.fuzzy_threshold_var = tk.DoubleVar(value=self.config.get('glossary_fuzzy_threshold', 0.90))

        # Slider
        fuzzy_slider = tb.Scale(
            slider_frame,
            from_=0.5,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.fuzzy_threshold_var,
            style="info.Horizontal.TScale",
            length=300
        )
        fuzzy_slider.pack(side=tk.LEFT, padx=(0, 10))

        # Value label
        self.fuzzy_value_label = tk.Label(slider_frame, text=f"{self.fuzzy_threshold_var.get():.2f}")
        self.fuzzy_value_label.pack(side=tk.LEFT)

        # Description label - CREATE THIS FIRST
        fuzzy_desc_label = tk.Label(fuzzy_frame, text="", font=('TkDefaultFont', 9), fg='blue')
        fuzzy_desc_label.pack(anchor=tk.W, pady=(5, 0))

        # Token-efficient format toggle
        format_frame = tk.LabelFrame(manual_container, text="Output Format", padx=10, pady=10)
        format_frame.pack(fill=tk.X, pady=(0, 10))

        # Initialize variable if not exists
        if not hasattr(self, 'use_legacy_csv_var'):
            self.use_legacy_csv_var = tk.BooleanVar(value=self.config.get('glossary_use_legacy_csv', False))

        tb.Checkbutton(format_frame, text="Use legacy CSV format", 
                      variable=self.use_legacy_csv_var,
                      bootstyle="round-toggle").pack(anchor=tk.W)

        tk.Label(format_frame, text="When disabled (default): Uses token-efficient format with sections (=== CHARACTERS ===)",
                font=('TkDefaultFont', 9), fg='gray').pack(anchor=tk.W, padx=20, pady=(0, 5))

        tk.Label(format_frame, text="When enabled: Uses traditional CSV format with repeated type columns",
                font=('TkDefaultFont', 9), fg='gray').pack(anchor=tk.W, padx=20)
        
        # Update label when slider moves - DEFINE AFTER CREATING THE LABEL
        def update_fuzzy_label(*args):
            try:
                # Check if widgets still exist before updating
                if not fuzzy_desc_label.winfo_exists():
                    return
                if not self.fuzzy_value_label.winfo_exists():
                    return
                    
                value = self.fuzzy_threshold_var.get()
                self.fuzzy_value_label.config(text=f"{value:.2f}")
                
                # Show description
                if value >= 0.95:
                    desc = "Exact match only (strict)"
                elif value >= 0.85:
                    desc = "Very similar names (recommended)"
                elif value >= 0.75:
                    desc = "Moderately similar names"
                elif value >= 0.65:
                    desc = "Loosely similar names"
                else:
                    desc = "Very loose matching (may over-merge)"
                
                fuzzy_desc_label.config(text=desc)
            except tk.TclError:
                # Widget was destroyed, ignore
                pass
            except Exception as e:
                # Catch any other unexpected errors
                print(f"Error updating fuzzy label: {e}")
                pass

        # Remove any existing trace before adding a new one
        if hasattr(self, 'manual_fuzzy_trace_id'):
            try:
                self.fuzzy_threshold_var.trace_remove('write', self.manual_fuzzy_trace_id)
            except:
                pass
        
        # Set up the trace AFTER creating the label and store the trace ID
        self.manual_fuzzy_trace_id = self.fuzzy_threshold_var.trace('w', update_fuzzy_label)
        
        # Initialize description by calling the function
        try:
            update_fuzzy_label()
        except:
            # If initialization fails, just continue
            pass
        
        # Prompt section (continues as before)
        prompt_frame = tk.LabelFrame(manual_container, text="Extraction Prompt", padx=10, pady=10)
        prompt_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(prompt_frame, text="Use {fields} for field list and {chapter_text} for content placeholder",
                font=('TkDefaultFont', 9), fg='blue').pack(anchor=tk.W, pady=(0, 5))
        
        tk.Label(prompt_frame, text="The {fields} placeholder will be replaced with the format specification",
                font=('TkDefaultFont', 9), fg='gray').pack(anchor=tk.W, pady=(0, 5))
        
        self.manual_prompt_text = self.ui.setup_scrollable_text(
            prompt_frame, height=13, wrap=tk.WORD
        )
        self.manual_prompt_text.pack(fill=tk.BOTH, expand=True)
        
        # Set default prompt if not already set
        if not hasattr(self, 'manual_glossary_prompt') or not self.manual_glossary_prompt:
            self.manual_glossary_prompt = """Extract character names and important terms from the following text.

Output format:
{fields}

Rules:
- Output ONLY CSV lines in the exact format shown above
- No headers, no extra text, no JSON
- One entry per line
- Leave gender empty for terms (just end with comma)
    """
        
        self.manual_prompt_text.insert('1.0', self.manual_glossary_prompt)
        self.manual_prompt_text.edit_reset()
        
        prompt_controls = tk.Frame(manual_container)
        prompt_controls.pack(fill=tk.X, pady=(10, 0))
        
        def reset_manual_prompt():
            if messagebox.askyesno("Reset Prompt", "Reset manual glossary prompt to default?"):
                self.manual_prompt_text.delete('1.0', tk.END)
                default_prompt = """Extract character names and important terms from the following text.

    Output format:
    {fields}

    Rules:
    - Output ONLY CSV lines in the exact format shown above
    - No headers, no extra text, no JSON
    - One entry per line
    - Leave gender empty for terms (just end with comma)
    """
                self.manual_prompt_text.insert('1.0', default_prompt)
        
        tb.Button(prompt_controls, text="Reset to Default", command=reset_manual_prompt, 
                bootstyle="warning").pack(side=tk.LEFT, padx=5)
        
        # Settings
        settings_frame = tk.LabelFrame(manual_container, text="Extraction Settings", padx=10, pady=10)
        settings_frame.pack(fill=tk.X, pady=(10, 0))
        
        settings_grid = tk.Frame(settings_frame)
        settings_grid.pack()
        
        tk.Label(settings_grid, text="Temperature:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.manual_temp_var = tk.StringVar(value=str(self.config.get('manual_glossary_temperature', 0.1)))
        tb.Entry(settings_grid, textvariable=self.manual_temp_var, width=10).grid(row=0, column=1, padx=5)
        
        tk.Label(settings_grid, text="Context Limit:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.manual_context_var = tk.StringVar(value=str(self.config.get('manual_context_limit', 2)))
        tb.Entry(settings_grid, textvariable=self.manual_context_var, width=10).grid(row=0, column=3, padx=5)
        
        tk.Label(settings_grid, text="Rolling Window:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=(10, 0))
        tb.Checkbutton(settings_grid, text="Keep recent context instead of reset", 
                      variable=self.glossary_history_rolling_var,
                      bootstyle="round-toggle").grid(row=1, column=1, columnspan=3, sticky=tk.W, padx=5, pady=(10, 0))
        
        tk.Label(settings_grid, text="When context limit is reached, keep recent chapters instead of clearing all history",
                font=('TkDefaultFont', 11), fg='gray').grid(row=2, column=0, columnspan=4, sticky=tk.W, padx=20, pady=(0, 5))

    def update_glossary_prompts(self):
        """Update glossary prompts from text widgets if they exist"""
        try:
            if hasattr(self, 'manual_prompt_text'):
                self.manual_glossary_prompt = self.manual_prompt_text.get('1.0', tk.END).strip()
            
            if hasattr(self, 'auto_prompt_text'):
                self.auto_glossary_prompt = self.auto_prompt_text.get('1.0', tk.END).strip()
            
            if hasattr(self, 'append_prompt_text'):
                self.append_glossary_prompt = self.append_prompt_text.get('1.0', tk.END).strip()
            
            if hasattr(self, 'translation_prompt_text'):
                self.glossary_translation_prompt = self.translation_prompt_text.get('1.0', tk.END).strip()

            if hasattr(self, 'format_instructions_text'):
                self.glossary_format_instructions = self.format_instructions_text.get('1.0', tk.END).strip()
                
        except Exception as e:
            print(f"Error updating glossary prompts: {e}")
            
    def _setup_auto_glossary_tab(self, parent):
        """Setup automatic glossary tab with fully configurable prompts"""
        auto_container = tk.Frame(parent)
        auto_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Master toggle
        master_toggle_frame = tk.Frame(auto_container)
        master_toggle_frame.pack(fill=tk.X, pady=(0, 15))
        
        tb.Checkbutton(master_toggle_frame, text="Enable Automatic Glossary Generation", 
                      variable=self.enable_auto_glossary_var,
                      bootstyle="round-toggle").pack(side=tk.LEFT)
        
        tk.Label(master_toggle_frame, text="(Automatic extraction and translation of character names/Terms)",
                font=('TkDefaultFont', 9), fg='gray').pack(side=tk.LEFT, padx=(10, 0))
        
        # Append glossary toggle
        append_frame = tk.Frame(auto_container)
        append_frame.pack(fill=tk.X, pady=(0, 15))
        
        tb.Checkbutton(append_frame, text="Append Glossary to System Prompt", 
                      variable=self.append_glossary_var,
                      bootstyle="round-toggle").pack(side=tk.LEFT)
        
        tk.Label(append_frame, text="(Applies to ALL glossaries - manual and automatic)",
                font=('TkDefaultFont', 10, 'italic'), fg='blue').pack(side=tk.LEFT, padx=(10, 0))
        
        # Custom append prompt section
        append_prompt_frame = tk.LabelFrame(auto_container, text="Glossary Append Format", padx=10, pady=10)
        append_prompt_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(append_prompt_frame, text="This text will be added before the glossary entries:",
                font=('TkDefaultFont', 10)).pack(anchor=tk.W, pady=(0, 5))
        
        self.append_prompt_text = self.ui.setup_scrollable_text(
            append_prompt_frame, height=2, wrap=tk.WORD
        )
        self.append_prompt_text.pack(fill=tk.X)
        
        # Set default append prompt if not already set
        if not hasattr(self, 'append_glossary_prompt') or not self.append_glossary_prompt:
            self.append_glossary_prompt = "- Follow this reference glossary for consistent translation (Do not output any raw entries):\n"
        
        self.append_prompt_text.insert('1.0', self.append_glossary_prompt)
        self.append_prompt_text.edit_reset()
        
        append_prompt_controls = tk.Frame(append_prompt_frame)
        append_prompt_controls.pack(fill=tk.X, pady=(5, 0))
        
        def reset_append_prompt():
            if messagebox.askyesno("Reset Prompt", "Reset to default glossary append format?"):
                self.append_prompt_text.delete('1.0', tk.END)
                self.append_prompt_text.insert('1.0', "- Follow this reference glossary for consistent translation (Do not output any raw entries):\n")
        
        tb.Button(append_prompt_controls, text="Reset to Default", command=reset_append_prompt, 
                 bootstyle="warning").pack(side=tk.LEFT, padx=5)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(auto_container)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Extraction Settings
        extraction_tab = tk.Frame(notebook)
        notebook.add(extraction_tab, text="Extraction Settings")
        
        # Extraction settings
        settings_label_frame = tk.LabelFrame(extraction_tab, text="Targeted Extraction Settings", padx=10, pady=10)
        settings_label_frame.pack(fill=tk.X, padx=10, pady=10)
        
        extraction_grid = tk.Frame(settings_label_frame)
        extraction_grid.pack(fill=tk.X)
        
        # Row 1
        tk.Label(extraction_grid, text="Min frequency:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        tb.Entry(extraction_grid, textvariable=self.glossary_min_frequency_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        tk.Label(extraction_grid, text="Max names:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        tb.Entry(extraction_grid, textvariable=self.glossary_max_names_var, width=10).grid(row=0, column=3, sticky=tk.W)
        
        # Row 2
        tk.Label(extraction_grid, text="Max titles:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        tb.Entry(extraction_grid, textvariable=self.glossary_max_titles_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=(0, 20), pady=(5, 0))
        
        tk.Label(extraction_grid, text="Translation batch:").grid(row=1, column=2, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        tb.Entry(extraction_grid, textvariable=self.glossary_batch_size_var, width=10).grid(row=1, column=3, sticky=tk.W, pady=(5, 0))
        
        # Row 3 - Max text size and chapter split
        tk.Label(extraction_grid, text="Max text size:").grid(row=3, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        tb.Entry(extraction_grid, textvariable=self.glossary_max_text_size_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=(0, 20), pady=(5, 0))

        tk.Label(extraction_grid, text="Chapter split threshold:").grid(row=3, column=2, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        tb.Entry(extraction_grid, textvariable=self.glossary_chapter_split_threshold_var, width=10).grid(row=3, column=3, sticky=tk.W, pady=(5, 0))
        
        # Row 4 - Max sentences for glossary
        tk.Label(extraction_grid, text="Max sentences:").grid(row=4, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        tb.Entry(extraction_grid, textvariable=self.glossary_max_sentences_var, width=10).grid(row=4, column=1, sticky=tk.W, padx=(0, 20), pady=(5, 0))
        
        tk.Label(extraction_grid, text="(Limit for AI processing)", font=('TkDefaultFont', 9), fg='gray').grid(row=4, column=2, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # Row 5 - Filter mode
        tk.Label(extraction_grid, text="Filter mode:").grid(row=5, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        filter_frame = tk.Frame(extraction_grid)
        filter_frame.grid(row=5, column=1, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        tb.Radiobutton(filter_frame, text="All names & terms", variable=self.glossary_filter_mode_var, 
                      value="all", bootstyle="info").pack(side=tk.LEFT, padx=(0, 10))
        tb.Radiobutton(filter_frame, text="Names with honorifics only", variable=self.glossary_filter_mode_var, 
                      value="only_with_honorifics", bootstyle="info").pack(side=tk.LEFT, padx=(0, 10))
        tb.Radiobutton(filter_frame, text="Names without honorifics & terms", variable=self.glossary_filter_mode_var, 
                      value="only_without_honorifics", bootstyle="info").pack(side=tk.LEFT)

        # Row 6 - Strip honorifics
        tk.Label(extraction_grid, text="Strip honorifics:").grid(row=6, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        tb.Checkbutton(extraction_grid, text="Remove honorifics from extracted names", 
                      variable=self.strip_honorifics_var,
                      bootstyle="round-toggle").grid(row=6, column=1, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Row 7 - Fuzzy matching threshold (reuse existing variable)
        tk.Label(extraction_grid, text="Fuzzy threshold:").grid(row=7, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        
        fuzzy_frame = tk.Frame(extraction_grid)
        fuzzy_frame.grid(row=7, column=1, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Reuse the existing fuzzy_threshold_var that's already initialized elsewhere
        fuzzy_slider = tb.Scale(
            fuzzy_frame,
            from_=0.5,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.fuzzy_threshold_var,
            length=200,
            bootstyle="info"
        )
        fuzzy_slider.pack(side=tk.LEFT, padx=(0, 10))
        
        fuzzy_value_label = tk.Label(fuzzy_frame, text=f"{self.fuzzy_threshold_var.get():.2f}")
        fuzzy_value_label.pack(side=tk.LEFT, padx=(0, 10))
        
        fuzzy_desc_label = tk.Label(fuzzy_frame, text="", font=('TkDefaultFont', 9), fg='gray')
        fuzzy_desc_label.pack(side=tk.LEFT)
        
        # Reuse the exact same update function logic
        def update_fuzzy_label(*args):
            try:
                # Check if widgets still exist before updating
                if not fuzzy_desc_label.winfo_exists():
                    return
                if not fuzzy_value_label.winfo_exists():
                    return
                    
                value = self.fuzzy_threshold_var.get()
                fuzzy_value_label.config(text=f"{value:.2f}")
                
                # Show description
                if value >= 0.95:
                    desc = "Exact match only (strict)"
                elif value >= 0.85:
                    desc = "Very similar names (recommended)"
                elif value >= 0.75:
                    desc = "Moderately similar names"
                elif value >= 0.65:
                    desc = "Loosely similar names"
                else:
                    desc = "Very loose matching (may over-merge)"
                
                fuzzy_desc_label.config(text=desc)
            except tk.TclError:
                # Widget was destroyed, ignore
                pass
            except Exception as e:
                # Catch any other unexpected errors
                print(f"Error updating auto fuzzy label: {e}")
                pass
        
        # Remove any existing auto trace before adding a new one
        if hasattr(self, 'auto_fuzzy_trace_id'):
            try:
                self.fuzzy_threshold_var.trace_remove('write', self.auto_fuzzy_trace_id)
            except:
                pass
        
        # Set up the trace AFTER creating the label and store the trace ID
        self.auto_fuzzy_trace_id = self.fuzzy_threshold_var.trace('w', update_fuzzy_label)
        
        # Initialize description by calling the function
        try:
            update_fuzzy_label()
        except:
            # If initialization fails, just continue
            pass
                
        # Initialize the variable if not exists
        if not hasattr(self, 'strip_honorifics_var'):
            self.strip_honorifics_var = tk.BooleanVar(value=True)
        
        # Help text
        help_frame = tk.Frame(extraction_tab)
        help_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        
        tk.Label(help_frame, text="💡 Settings Guide:", font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W)
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
            tk.Label(help_frame, text=txt, font=('TkDefaultFont', 11), fg='gray').pack(anchor=tk.W, padx=20)
        
        # Tab 2: Extraction Prompt
        extraction_prompt_tab = tk.Frame(notebook)
        notebook.add(extraction_prompt_tab, text="Extraction Prompt")
        
        # Auto prompt section
        auto_prompt_frame = tk.LabelFrame(extraction_prompt_tab, text="Extraction Template (System Prompt)", padx=10, pady=10)
        auto_prompt_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(auto_prompt_frame, text="Available placeholders: {language}, {min_frequency}, {max_names}, {max_titles}",
                font=('TkDefaultFont', 9), fg='blue').pack(anchor=tk.W, pady=(0, 5))
        
        self.auto_prompt_text = self.ui.setup_scrollable_text(
            auto_prompt_frame, height=12, wrap=tk.WORD
        )
        self.auto_prompt_text.pack(fill=tk.BOTH, expand=True)
        
        # Set default extraction prompt if not set
        if not hasattr(self, 'auto_glossary_prompt') or not self.auto_glossary_prompt:
            self.auto_glossary_prompt = self.default_auto_glossary_prompt
        
        self.auto_prompt_text.insert('1.0', self.auto_glossary_prompt)
        self.auto_prompt_text.edit_reset()
        
        auto_prompt_controls = tk.Frame(extraction_prompt_tab)
        auto_prompt_controls.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        def reset_auto_prompt():
            if messagebox.askyesno("Reset Prompt", "Reset automatic glossary prompt to default?"):
                self.auto_prompt_text.delete('1.0', tk.END)
                self.auto_prompt_text.insert('1.0', self.default_auto_glossary_prompt)
        
        tb.Button(auto_prompt_controls, text="Reset to Default", command=reset_auto_prompt, 
                 bootstyle="warning").pack(side=tk.LEFT, padx=5)
        
        # Tab 3: Format Instructions - NEW TAB
        format_tab = tk.Frame(notebook)
        notebook.add(format_tab, text="Format Instructions")
        
        # Format instructions section
        format_prompt_frame = tk.LabelFrame(format_tab, text="Output Format Instructions (User Prompt)", padx=10, pady=10)
        format_prompt_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(format_prompt_frame, text="These instructions are added to your extraction prompt to specify the output format:",
                font=('TkDefaultFont', 10)).pack(anchor=tk.W, pady=(0, 5))
        
        tk.Label(format_prompt_frame, text="Available placeholders: {text_sample}",
                font=('TkDefaultFont', 9), fg='blue').pack(anchor=tk.W, pady=(0, 5))
        
        # Initialize format instructions variable and text widget
        if not hasattr(self, 'glossary_format_instructions'):
            self.glossary_format_instructions = """
Return the results in EXACT CSV format with this header:
type,raw_name,translated_name

For example:
character,김상현,Kim Sang-hyu
character,갈편제,Gale Hardest  
character,디히릿 아데,Dihirit Ade

Only include terms that actually appear in the text.
Do not use quotes around values unless they contain commas.

Text to analyze:
{text_sample}"""
        
        self.format_instructions_text = self.ui.setup_scrollable_text(
            format_prompt_frame, height=12, wrap=tk.WORD
        )
        self.format_instructions_text.pack(fill=tk.BOTH, expand=True)
        self.format_instructions_text.insert('1.0', self.glossary_format_instructions)
        self.format_instructions_text.edit_reset()
        
        format_prompt_controls = tk.Frame(format_tab)
        format_prompt_controls.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        def reset_format_instructions():
            if messagebox.askyesno("Reset Prompt", "Reset format instructions to default?"):
                default_format_instructions = """
Return the results in EXACT CSV format with this header:
type,raw_name,translated_name

For example:
character,김상현,Kim Sang-hyu
character,갈편제,Gale Hardest  
character,디히릿 아데,Dihirit Ade

Only include terms that actually appear in the text.
Do not use quotes around values unless they contain commas.

Text to analyze:
{text_sample}"""
                self.format_instructions_text.delete('1.0', tk.END)
                self.format_instructions_text.insert('1.0', default_format_instructions)
        
        tb.Button(format_prompt_controls, text="Reset to Default", command=reset_format_instructions, 
                 bootstyle="warning").pack(side=tk.LEFT, padx=5)
        
        # Tab 4: Translation Prompt (moved from Tab 3)
        translation_prompt_tab = tk.Frame(notebook)
        notebook.add(translation_prompt_tab, text="Translation Prompt")
        
        # Translation prompt section
        trans_prompt_frame = tk.LabelFrame(translation_prompt_tab, text="Glossary Translation Template (User Prompt)", padx=10, pady=10)
        trans_prompt_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(trans_prompt_frame, text="This prompt is used to translate extracted terms to English:",
                font=('TkDefaultFont', 10)).pack(anchor=tk.W, pady=(0, 5))
        
        tk.Label(trans_prompt_frame, text="Available placeholders: {language}, {terms_list}, {batch_size}",
                font=('TkDefaultFont', 9), fg='blue').pack(anchor=tk.W, pady=(0, 5))
        
        # Initialize translation prompt variable and text widget
        if not hasattr(self, 'glossary_translation_prompt'):
            self.glossary_translation_prompt = """
You are translating {language} character names and important terms to English.
For character names, provide English transliterations or keep as romanized.
Keep honorifics/suffixes only if they are integral to the name.
Respond with the same numbered format.

Terms to translate:
{terms_list}

Provide translations in the same numbered format."""
        
        self.translation_prompt_text = self.ui.setup_scrollable_text(
            trans_prompt_frame, height=12, wrap=tk.WORD
        )
        self.translation_prompt_text.pack(fill=tk.BOTH, expand=True)
        self.translation_prompt_text.insert('1.0', self.glossary_translation_prompt)
        self.translation_prompt_text.edit_reset()
        
        trans_prompt_controls = tk.Frame(translation_prompt_tab)
        trans_prompt_controls.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        def reset_trans_prompt():
            if messagebox.askyesno("Reset Prompt", "Reset translation prompt to default?"):
                default_trans_prompt = """
You are translating {language} character names and important terms to English.
For character names, provide English transliterations or keep as romanized.
Keep honorifics/suffixes only if they are integral to the name.
Respond with the same numbered format.

Terms to translate:
{terms_list}

Provide translations in the same numbered format."""
                self.translation_prompt_text.delete('1.0', tk.END)
                self.translation_prompt_text.insert('1.0', default_trans_prompt)
        
        tb.Button(trans_prompt_controls, text="Reset to Default", command=reset_trans_prompt, 
                 bootstyle="warning").pack(side=tk.LEFT, padx=5)
        
        # Update states function with proper error handling
        def update_auto_glossary_state():
            try:
                if not extraction_grid.winfo_exists():
                    return
                state = tk.NORMAL if self.enable_auto_glossary_var.get() else tk.DISABLED
                for widget in extraction_grid.winfo_children():
                    if isinstance(widget, (tb.Entry, ttk.Entry, tb.Checkbutton, ttk.Checkbutton)):
                        widget.config(state=state)
                    # Handle frames that contain radio buttons or scales
                    elif isinstance(widget, tk.Frame):
                        for child in widget.winfo_children():
                            if isinstance(child, (tb.Radiobutton, ttk.Radiobutton, tb.Scale, ttk.Scale)):
                                child.config(state=state)
                if self.auto_prompt_text.winfo_exists():
                    self.auto_prompt_text.config(state=state)
                if hasattr(self, 'format_instructions_text') and self.format_instructions_text.winfo_exists():
                    self.format_instructions_text.config(state=state)
                if hasattr(self, 'translation_prompt_text') and self.translation_prompt_text.winfo_exists():
                    self.translation_prompt_text.config(state=state)
                for widget in auto_prompt_controls.winfo_children():
                    if isinstance(widget, (tb.Button, ttk.Button)) and widget.winfo_exists():
                        widget.config(state=state)
                for widget in format_prompt_controls.winfo_children():
                    if isinstance(widget, (tb.Button, ttk.Button)) and widget.winfo_exists():
                        widget.config(state=state)
                for widget in trans_prompt_controls.winfo_children():
                    if isinstance(widget, (tb.Button, ttk.Button)) and widget.winfo_exists():
                        widget.config(state=state)
            except tk.TclError:
                # Widget was destroyed, ignore
                pass
        
        def update_append_prompt_state():
            try:
                if not self.append_prompt_text.winfo_exists():
                    return
                state = tk.NORMAL if self.append_glossary_var.get() else tk.DISABLED
                self.append_prompt_text.config(state=state)
                for widget in append_prompt_controls.winfo_children():
                    if isinstance(widget, (tb.Button, ttk.Button)) and widget.winfo_exists():
                        widget.config(state=state)
            except tk.TclError:
                # Widget was destroyed, ignore
                pass
        
        # Initialize states
        update_auto_glossary_state()
        update_append_prompt_state()
        
        # Add traces
        self.enable_auto_glossary_var.trace('w', lambda *args: update_auto_glossary_state())
        self.append_glossary_var.trace('w', lambda *args: update_append_prompt_state())

    def _setup_glossary_editor_tab(self, parent):
        """Set up the glossary editor/trimmer tab"""
        container = tk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        file_frame = tk.Frame(container)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(file_frame, text="Glossary File:").pack(side=tk.LEFT, padx=(0, 5))
        self.editor_file_var = tk.StringVar()
        tb.Entry(file_frame, textvariable=self.editor_file_var, state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        stats_frame = tk.Frame(container)
        stats_frame.pack(fill=tk.X, pady=(0, 5))
        self.stats_label = tk.Label(stats_frame, text="No glossary loaded", font=('TkDefaultFont', 10, 'italic'))
        self.stats_label.pack(side=tk.LEFT)

        content_frame = tk.LabelFrame(container, text="Glossary Entries", padx=10, pady=10)
        content_frame.pack(fill=tk.BOTH, expand=True)

        tree_frame = tk.Frame(content_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")

        self.glossary_tree = ttk.Treeview(tree_frame, show='tree headings',
                                        yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.config(command=self.glossary_tree.yview)
        hsb.config(command=self.glossary_tree.xview)

        self.glossary_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        self.glossary_tree.bind('<Double-Button-1>', self._on_tree_double_click)

        self.current_glossary_data = None
        self.current_glossary_format = None

        # Editor functions
        def load_glossary_for_editing():
           path = self.editor_file_var.get()
           if not path or not os.path.exists(path):
               messagebox.showerror("Error", "Please select a valid glossary file")
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
               
               self.glossary_tree.delete(*self.glossary_tree.get_children())
               self.glossary_tree['columns'] = column_fields
               
               self.glossary_tree.heading('#0', text='#')
               self.glossary_tree.column('#0', width=40, stretch=False)
               
               for field in column_fields:
                   display_name = field.replace('_', ' ').title()
                   self.glossary_tree.heading(field, text=display_name)
                   
                   if field in ['raw_name', 'translated_name', 'original_name', 'name', 'original', 'translated']:
                       width = 150
                   elif field in ['traits', 'locations', 'how_they_refer_to_others']:
                       width = 200
                   else:
                       width = 100
                   
                   self.glossary_tree.column(field, width=width)
               
               for idx, entry in enumerate(entries):
                   values = []
                   for field in column_fields:
                       value = entry.get(field, '')
                       if isinstance(value, list):
                           value = ', '.join(str(v) for v in value)
                       elif isinstance(value, dict):
                           value = ', '.join(f"{k}: {v}" for k, v in value.items())
                       elif value is None:
                           value = ''
                       values.append(value)
                   
                   self.glossary_tree.insert('', 'end', text=str(idx + 1), values=values)
               
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
               
               self.stats_label.config(text=" | ".join(stats))
               self.append_log(f"✅ Loaded {len(entries)} entries from glossary")
               
           except Exception as e:
               messagebox.showerror("Error", f"Failed to load glossary: {e}")
               self.append_log(f"❌ Failed to load glossary: {e}")
       
        def browse_glossary():
           path = filedialog.askopenfilename(
               title="Select glossary file",
               filetypes=[("Glossary files", "*.json *.csv"), ("JSON files", "*.json"), ("CSV files", "*.csv")]
           )
           if path:
               self.editor_file_var.set(path)
               load_glossary_for_editing()
       
        # Common save helper
        def save_current_glossary():
           path = self.editor_file_var.get()
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
               messagebox.showerror("Error", f"Failed to save: {e}")
               return False
       
        def clean_empty_fields():
            if not self.current_glossary_data:
                messagebox.showerror("Error", "No glossary loaded")
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
                    messagebox.showinfo("Info", "No empty fields found in glossary")
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
                    
                    messagebox.showinfo("Success", msg)
        
        def delete_selected_entries():
            selected = self.glossary_tree.selection()
            if not selected:
                messagebox.showwarning("No Selection", "Please select entries to delete")
                return
            
            count = len(selected)
            if messagebox.askyesno("Confirm Delete", f"Delete {count} selected entries?"):
                # automatic backup
                if not self.create_glossary_backup(f"before_delete_{count}"):
                    return
                    
                indices_to_delete = []
                for item in selected:
                   idx = int(self.glossary_tree.item(item)['text']) - 1
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
                   messagebox.showinfo("Success", f"Deleted {len(indices_to_delete)} entries")
                
        def remove_duplicates():
            if not self.current_glossary_data:
                messagebox.showerror("Error", "No glossary loaded")
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
                            messagebox.showinfo("Success", f"Removed {duplicates_removed} duplicate entries")
                            self.append_log(f"🗑️ Removed {duplicates_removed} duplicates based on raw_name")
                    else:
                        messagebox.showinfo("Info", "No duplicates found")
                        
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
                            messagebox.showinfo("Success", f"Removed {duplicates} duplicate entries")
                    else:
                        messagebox.showinfo("Info", "No duplicates found")

        # dialog function for configuring duplicate detection mode
        def duplicate_detection_settings():
            """Show info about duplicate detection (simplified for new format)"""
            messagebox.showinfo(
                "Duplicate Detection", 
                "Duplicate detection is based on the raw_name field.\n\n"
                "• Entries with identical raw_name values are considered duplicates\n"
                "• The first occurrence is kept, later ones are removed\n"
                "• Honorifics filtering can be toggled in the Manual Glossary tab\n\n"
                "When honorifics filtering is enabled, names are compared after removing honorifics."
            )

        def backup_settings_dialog():
            """Show dialog for configuring automatic backup settings"""
            # Use setup_scrollable with custom ratios
            dialog, scrollable_frame, canvas = self.wm.setup_scrollable(
                self.master,
                "Automatic Backup Settings",
                width=500,
                height=None,
                max_width_ratio=0.45,
                max_height_ratio=0.51
            )
            
            # Main frame
            main_frame = ttk.Frame(scrollable_frame, padding="20")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Title
            ttk.Label(main_frame, text="Automatic Backup Settings", 
                      font=('TkDefaultFont', 22, 'bold')).pack(pady=(0, 20))
            
            # Backup toggle
            backup_var = tk.BooleanVar(value=self.config.get('glossary_auto_backup', True))
            backup_frame = ttk.Frame(main_frame)
            backup_frame.pack(fill=tk.X, pady=5)
            
            backup_check = ttk.Checkbutton(backup_frame, 
                                           text="Enable automatic backups before modifications",
                                           variable=backup_var)
            backup_check.pack(anchor=tk.W)
            
            # Settings frame (indented)
            settings_frame = ttk.Frame(main_frame)
            settings_frame.pack(fill=tk.X, pady=(10, 0), padx=(20, 0))
            
            # Max backups setting
            max_backups_frame = ttk.Frame(settings_frame)
            max_backups_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(max_backups_frame, text="Maximum backups to keep:").pack(side=tk.LEFT, padx=(0, 10))
            max_backups_var = tk.IntVar(value=self.config.get('glossary_max_backups', 50))
            max_backups_spin = ttk.Spinbox(max_backups_frame, from_=0, to=999, 
                                           textvariable=max_backups_var, width=10)
            max_backups_spin.pack(side=tk.LEFT)
            ttk.Label(max_backups_frame, text="(0 = unlimited)", 
                      font=('TkDefaultFont', 9), 
                      foreground='gray').pack(side=tk.LEFT, padx=(10, 0))
            
            # Backup naming pattern info
            pattern_frame = ttk.Frame(settings_frame)
            pattern_frame.pack(fill=tk.X, pady=(15, 5))
            
            ttk.Label(pattern_frame, text="Backup naming pattern:", 
                      font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W)
            ttk.Label(pattern_frame, 
                      text="[original_name]_[operation]_[YYYYMMDD_HHMMSS].json",
                      font=('TkDefaultFont', 9, 'italic'),
                      foreground='#666').pack(anchor=tk.W, padx=(10, 0))
            
            # Example
            example_text = "Example: my_glossary_before_delete_5_20240115_143052.json"
            ttk.Label(pattern_frame, text=example_text,
                      font=('TkDefaultFont', 8),
                      foreground='gray').pack(anchor=tk.W, padx=(10, 0), pady=(2, 0))
            
            # Separator
            ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=(20, 15))
            
            # Backup location info
            location_frame = ttk.Frame(main_frame)
            location_frame.pack(fill=tk.X)
            
            ttk.Label(location_frame, text="📁 Backup Location:", 
                      font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W)
            
            if self.editor_file_var.get():
                glossary_dir = os.path.dirname(self.editor_file_var.get())
                backup_path = "Backups"
                full_path = os.path.join(glossary_dir, "Backups")
                
                path_label = ttk.Label(location_frame, 
                                      text=f"{backup_path}/",
                                      font=('TkDefaultFont', 9),
                                      foreground='#0066cc')
                path_label.pack(anchor=tk.W, padx=(10, 0))
                
                # Check if backup folder exists and show count
                if os.path.exists(full_path):
                    backup_count = len([f for f in os.listdir(full_path) if f.endswith('.json')])
                    ttk.Label(location_frame, 
                             text=f"Currently contains {backup_count} backup(s)",
                             font=('TkDefaultFont', 8),
                             foreground='gray').pack(anchor=tk.W, padx=(10, 0))
            else:
                ttk.Label(location_frame, 
                         text="Backups",
                         font=('TkDefaultFont', 9),
                         foreground='gray').pack(anchor=tk.W, padx=(10, 0))
            
            def toggle_settings_state(*args):
                state = tk.NORMAL if backup_var.get() else tk.DISABLED
                max_backups_spin.config(state=state)
            
            backup_var.trace('w', toggle_settings_state)
            toggle_settings_state()  # Set initial state
            
            # Buttons
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=(25, 0))
            
            # Inner frame for centering buttons
            button_inner_frame = ttk.Frame(button_frame)
            button_inner_frame.pack(anchor=tk.CENTER)
            
            def save_settings():
                # Save backup settings
                self.config['glossary_auto_backup'] = backup_var.get()
                self.config['glossary_max_backups'] = max_backups_var.get()
                
                # Save to config file
                with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                
                status = "enabled" if backup_var.get() else "disabled"
                if backup_var.get():
                    limit = max_backups_var.get()
                    limit_text = "unlimited" if limit == 0 else f"max {limit}"
                    msg = f"Automatic backups {status} ({limit_text})"
                else:
                    msg = f"Automatic backups {status}"
                    
                messagebox.showinfo("Success", msg)
                dialog.destroy()
            
            def create_manual_backup():
                """Create a manual backup right now"""
                if not self.current_glossary_data:
                    messagebox.showerror("Error", "No glossary loaded")
                    return
                    
                if self.create_glossary_backup("manual"):
                    messagebox.showinfo("Success", "Manual backup created successfully!")
            
            tb.Button(button_inner_frame, text="Save Settings", command=save_settings, 
                      bootstyle="success", width=15).pack(side=tk.LEFT, padx=5)
            tb.Button(button_inner_frame, text="Backup Now", command=create_manual_backup,
                      bootstyle="info", width=15).pack(side=tk.LEFT, padx=5)
            tb.Button(button_inner_frame, text="Cancel", command=dialog.destroy,
                      bootstyle="secondary", width=15).pack(side=tk.LEFT, padx=5)
            
            # Auto-resize and show
            self.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.45, max_height_ratio=0.41)
    
        def smart_trim_dialog():
            if not self.current_glossary_data:
                messagebox.showerror("Error", "No glossary loaded")
                return
            
            # Use WindowManager's setup_scrollable for unified scrolling
            dialog, scrollable_frame, canvas = self.wm.setup_scrollable(
                self.master,
                "Smart Trim Glossary",
                width=600,
                height=None,
                max_width_ratio=0.9,
                max_height_ratio=0.85
            )
            
            main_frame = scrollable_frame
            
            # Title and description
            tk.Label(main_frame, text="Smart Glossary Trimming", 
                    font=('TkDefaultFont', 14, 'bold')).pack(pady=(20, 5))
            
            tk.Label(main_frame, text="Limit the number of entries in your glossary",
                    font=('TkDefaultFont', 10), fg='gray', wraplength=550).pack(pady=(0, 15))
            
            # Display current glossary stats
            stats_frame = tk.LabelFrame(main_frame, text="Current Glossary Statistics", padx=15, pady=10)
            stats_frame.pack(fill=tk.X, pady=(0, 15), padx=20)
            
            entry_count = len(self.current_glossary_data) if self.current_glossary_format == 'list' else len(self.current_glossary_data.get('entries', {}))
            tk.Label(stats_frame, text=f"Total entries: {entry_count}", font=('TkDefaultFont', 10)).pack(anchor=tk.W)
            
            # For new format, show type breakdown
            if self.current_glossary_format == 'list' and self.current_glossary_data and 'type' in self.current_glossary_data[0]:
                characters = sum(1 for e in self.current_glossary_data if e.get('type') == 'character')
                terms = sum(1 for e in self.current_glossary_data if e.get('type') == 'term')
                tk.Label(stats_frame, text=f"Characters: {characters}, Terms: {terms}", font=('TkDefaultFont', 10)).pack(anchor=tk.W)
            
            # Entry limit section
            limit_frame = tk.LabelFrame(main_frame, text="Entry Limit", padx=15, pady=10)
            limit_frame.pack(fill=tk.X, pady=(0, 15), padx=20)
            
            tk.Label(limit_frame, text="Keep only the first N entries to reduce glossary size",
                    font=('TkDefaultFont', 9), fg='gray', wraplength=520).pack(anchor=tk.W, pady=(0, 10))
            
            top_frame = tk.Frame(limit_frame)
            top_frame.pack(fill=tk.X, pady=5)
            tk.Label(top_frame, text="Keep first").pack(side=tk.LEFT)
            top_var = tk.StringVar(value=str(min(100, entry_count)))
            tb.Entry(top_frame, textvariable=top_var, width=10).pack(side=tk.LEFT, padx=5)
            tk.Label(top_frame, text=f"entries (out of {entry_count})").pack(side=tk.LEFT)
            
            # Preview section
            preview_frame = tk.LabelFrame(main_frame, text="Preview", padx=15, pady=10)
            preview_frame.pack(fill=tk.X, pady=(0, 15), padx=20)
            
            preview_label = tk.Label(preview_frame, text="Click 'Preview Changes' to see the effect",
                                   font=('TkDefaultFont', 10), fg='gray')
            preview_label.pack(pady=5)
            
            def preview_changes():
                try:
                    top_n = int(top_var.get())
                    entries_to_remove = max(0, entry_count - top_n)
                    
                    preview_text = f"Preview of changes:\n"
                    preview_text += f"• Entries: {entry_count} → {top_n} ({entries_to_remove} removed)\n"
                    
                    preview_label.config(text=preview_text, fg='blue')
                    
                except ValueError:
                    preview_label.config(text="Please enter a valid number", fg='red')
            
            tb.Button(preview_frame, text="Preview Changes", command=preview_changes,
                     bootstyle="info").pack()
            
            # Action buttons
            button_frame = tk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=(10, 20), padx=20)
            
            def apply_smart_trim():
                try:
                    top_n = int(top_var.get())
                    
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
                        
                        messagebox.showinfo("Success", f"Trimmed glossary to {top_n} entries")
                        dialog.destroy()
                        
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid numbers")

            # Create inner frame for buttons
            button_inner_frame = tk.Frame(button_frame)
            button_inner_frame.pack()

            tb.Button(button_inner_frame, text="Apply Trim", command=apply_smart_trim,
                 bootstyle="success", width=15).pack(side=tk.LEFT, padx=5)
            tb.Button(button_inner_frame, text="Cancel", command=dialog.destroy,
                 bootstyle="secondary", width=15).pack(side=tk.LEFT, padx=5)

            # Info section at bottom
            info_frame = tk.Frame(main_frame)
            info_frame.pack(fill=tk.X, pady=(0, 20), padx=20)

            tk.Label(info_frame, text="💡 Tip: Entries are kept in their original order",
                font=('TkDefaultFont', 9, 'italic'), fg='#666').pack()

            # Auto-resize the dialog to fit content
            self.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.9, max_height_ratio=1.2)
       
        def filter_entries_dialog():
            if not self.current_glossary_data:
                messagebox.showerror("Error", "No glossary loaded")
                return
            
            # Use WindowManager's setup_scrollable for unified scrolling
            dialog, scrollable_frame, canvas = self.wm.setup_scrollable(
                self.master,
                "Filter Entries",
                width=600,
                height=None,
                max_width_ratio=0.9,
                max_height_ratio=0.85
            )
            
            main_frame = scrollable_frame
            
            # Title and description
            tk.Label(main_frame, text="Filter Glossary Entries", 
                    font=('TkDefaultFont', 14, 'bold')).pack(pady=(20, 5))
            
            tk.Label(main_frame, text="Filter entries by type or content",
                    font=('TkDefaultFont', 10), fg='gray', wraplength=550).pack(pady=(0, 15))
            
            # Current stats
            entry_count = len(self.current_glossary_data) if self.current_glossary_format == 'list' else len(self.current_glossary_data.get('entries', {}))
            
            stats_frame = tk.LabelFrame(main_frame, text="Current Status", padx=15, pady=10)
            stats_frame.pack(fill=tk.X, pady=(0, 15), padx=20)
            tk.Label(stats_frame, text=f"Total entries: {entry_count}", font=('TkDefaultFont', 10)).pack(anchor=tk.W)
            
            # Check if new format
            is_new_format = (self.current_glossary_format == 'list' and 
                           self.current_glossary_data and 
                           'type' in self.current_glossary_data[0])
            
            # Filter conditions
            conditions_frame = tk.LabelFrame(main_frame, text="Filter Conditions", padx=15, pady=10)
            conditions_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15), padx=20)
            
            # Type filter for new format
            type_vars = {}
            if is_new_format:
                type_frame = tk.LabelFrame(conditions_frame, text="Entry Type", padx=10, pady=10)
                type_frame.pack(fill=tk.X, pady=(0, 10))
                
                type_vars['character'] = tk.BooleanVar(value=True)
                type_vars['term'] = tk.BooleanVar(value=True)
                
                tb.Checkbutton(type_frame, text="Keep characters", variable=type_vars['character']).pack(anchor=tk.W)
                tb.Checkbutton(type_frame, text="Keep terms/locations", variable=type_vars['term']).pack(anchor=tk.W)
            
            # Text content filter
            text_filter_frame = tk.LabelFrame(conditions_frame, text="Text Content Filter", padx=10, pady=10)
            text_filter_frame.pack(fill=tk.X, pady=(0, 10))
            
            tk.Label(text_filter_frame, text="Keep entries containing text (case-insensitive):",
                    font=('TkDefaultFont', 9), fg='gray').pack(anchor=tk.W, pady=(0, 5))
            
            search_var = tk.StringVar()
            tb.Entry(text_filter_frame, textvariable=search_var, width=40).pack(fill=tk.X, pady=5)
            
            # Gender filter for new format
            gender_var = tk.StringVar(value="all")
            if is_new_format:
                gender_frame = tk.LabelFrame(conditions_frame, text="Gender Filter (Characters Only)", padx=10, pady=10)
                gender_frame.pack(fill=tk.X, pady=(0, 10))
                
                tk.Radiobutton(gender_frame, text="All genders", variable=gender_var, value="all").pack(anchor=tk.W)
                tk.Radiobutton(gender_frame, text="Male only", variable=gender_var, value="Male").pack(anchor=tk.W)
                tk.Radiobutton(gender_frame, text="Female only", variable=gender_var, value="Female").pack(anchor=tk.W)
                tk.Radiobutton(gender_frame, text="Unknown only", variable=gender_var, value="Unknown").pack(anchor=tk.W)
            
            # Preview section
            preview_frame = tk.LabelFrame(main_frame, text="Preview", padx=15, pady=10)
            preview_frame.pack(fill=tk.X, pady=(0, 15), padx=20)
            
            preview_label = tk.Label(preview_frame, text="Click 'Preview Filter' to see how many entries match",
                                   font=('TkDefaultFont', 10), fg='gray')
            preview_label.pack(pady=5)
            
            def check_entry_matches(entry):
                """Check if an entry matches the filter conditions"""
                # Type filter
                if is_new_format and entry.get('type'):
                    if not type_vars.get(entry['type'], tk.BooleanVar(value=True)).get():
                        return False
                
                # Text filter
                search_text = search_var.get().strip().lower()
                if search_text:
                    # Search in all text fields
                    entry_text = ' '.join(str(v) for v in entry.values() if isinstance(v, str)).lower()
                    if search_text not in entry_text:
                        return False
                
                # Gender filter
                if is_new_format and gender_var.get() != "all":
                    if entry.get('type') == 'character' and entry.get('gender') != gender_var.get():
                        return False
                
                return True
            
            def preview_filter():
                """Preview the filter results"""
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
                preview_label.config(
                    text=f"Filter matches: {matching} entries ({removed} will be removed)",
                    fg='blue' if matching > 0 else 'red'
                )
            
            tb.Button(preview_frame, text="Preview Filter", command=preview_filter,
                     bootstyle="info").pack()
            
            # Action buttons
            button_frame = tk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=(10, 20), padx=20)
            
            def apply_filter():
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
                        messagebox.showinfo("Success", 
                            f"Filter applied!\n\nKept: {len(filtered)} entries\nRemoved: {removed} entries")
                        dialog.destroy()
            
            # Create inner frame for buttons
            button_inner_frame = tk.Frame(button_frame)
            button_inner_frame.pack()

            tb.Button(button_inner_frame, text="Apply Filter", command=apply_filter,
                     bootstyle="success", width=15).pack(side=tk.LEFT, padx=5)
            tb.Button(button_inner_frame, text="Cancel", command=dialog.destroy,
                     bootstyle="secondary", width=15).pack(side=tk.LEFT, padx=5)
            
            # Auto-resize the dialog to fit content
            self.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.9, max_height_ratio=1.49)
    
        def export_selection():
           selected = self.glossary_tree.selection()
           if not selected:
               messagebox.showwarning("Warning", "No entries selected")
               return
           
           path = filedialog.asksaveasfilename(
               title="Export Selected Entries",
               defaultextension=".json",
               filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv")]
           )
           
           if not path:
               return
           
           try:
               if self.current_glossary_format == 'list':
                   exported = []
                   for item in selected:
                       idx = int(self.glossary_tree.item(item)['text']) - 1
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
                       idx = int(self.glossary_tree.item(item)['text']) - 1
                       if 0 <= idx < len(entries_list):
                           key, value = entries_list[idx]
                           exported[key] = value
                   
                   with open(path, 'w', encoding='utf-8') as f:
                       json.dump(exported, f, ensure_ascii=False, indent=2)
               
               messagebox.showinfo("Success", f"Exported {len(selected)} entries to {os.path.basename(path)}")
               
           except Exception as e:
               messagebox.showerror("Error", f"Failed to export: {e}")
       
        def save_edited_glossary():
           if save_current_glossary():
               messagebox.showinfo("Success", "Glossary saved successfully")
               self.append_log(f"✅ Saved glossary to: {self.editor_file_var.get()}")
       
        def save_as_glossary():
           if not self.current_glossary_data:
               messagebox.showerror("Error", "No glossary loaded")
               return
           
           path = filedialog.asksaveasfilename(
               title="Save Glossary As",
               defaultextension=".json",
               filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv")]
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
               
               self.editor_file_var.set(path)
               messagebox.showinfo("Success", f"Glossary saved to {os.path.basename(path)}")
               self.append_log(f"✅ Saved glossary as: {path}")
               
           except Exception as e:
               messagebox.showerror("Error", f"Failed to save: {e}")
       
        # Buttons
        tb.Button(file_frame, text="Browse", command=browse_glossary, width=15).pack(side=tk.LEFT)
        
       
        editor_controls = tk.Frame(container)
        editor_controls.pack(fill=tk.X, pady=(10, 0))
       
        # Row 1
        row1 = tk.Frame(editor_controls)
        row1.pack(fill=tk.X, pady=2)
       
        buttons_row1 = [
           ("Reload", load_glossary_for_editing, "info"),
           ("Delete Selected", delete_selected_entries, "danger"),
           ("Clean Empty Fields", clean_empty_fields, "warning"),
           ("Remove Duplicates", remove_duplicates, "warning"),
           ("Backup Settings", backup_settings_dialog, "success")
        ]
       
        for text, cmd, style in buttons_row1:
           tb.Button(row1, text=text, command=cmd, bootstyle=style, width=15).pack(side=tk.LEFT, padx=2)
       
        # Row 2
        row2 = tk.Frame(editor_controls)
        row2.pack(fill=tk.X, pady=2)

        buttons_row2 = [
           ("Trim Entries", smart_trim_dialog, "primary"),
           ("Filter Entries", filter_entries_dialog, "primary"),
           ("Convert Format", lambda: self.convert_glossary_format(load_glossary_for_editing), "info"),
           ("Export Selection", export_selection, "secondary"),
           ("About Format", duplicate_detection_settings, "info")
        ]

        for text, cmd, style in buttons_row2:
           tb.Button(row2, text=text, command=cmd, bootstyle=style, width=15).pack(side=tk.LEFT, padx=2)

        # Row 3
        row3 = tk.Frame(editor_controls)
        row3.pack(fill=tk.X, pady=2)

        tb.Button(row3, text="Save Changes", command=save_edited_glossary,
                bootstyle="success", width=20).pack(side=tk.LEFT, padx=2)
        tb.Button(row3, text="Save As...", command=save_as_glossary,
                bootstyle="success-outline", width=20).pack(side=tk.LEFT, padx=2)

    def _on_tree_double_click(self, event):
       """Handle double-click on treeview item for inline editing"""
       region = self.glossary_tree.identify_region(event.x, event.y)
       if region != 'cell':
           return
       
       item = self.glossary_tree.identify_row(event.y)
       column = self.glossary_tree.identify_column(event.x)
       
       if not item or column == '#0':
           return
       
       col_idx = int(column.replace('#', '')) - 1
       columns = self.glossary_tree['columns']
       if col_idx >= len(columns):
           return
       
       col_name = columns[col_idx]
       values = self.glossary_tree.item(item)['values']
       current_value = values[col_idx] if col_idx < len(values) else ''
       
       dialog = self.wm.create_simple_dialog(
           self.master,
           f"Edit {col_name.replace('_', ' ').title()}",
           width=400,
           height=150
       )
       
       frame = tk.Frame(dialog, padx=20, pady=20)
       frame.pack(fill=tk.BOTH, expand=True)
       
       tk.Label(frame, text=f"Edit {col_name.replace('_', ' ').title()}:").pack(anchor=tk.W)
       
       # Simple entry for new format fields
       var = tk.StringVar(value=current_value)
       entry = tb.Entry(frame, textvariable=var, width=50)
       entry.pack(fill=tk.X, pady=5)
       entry.focus()
       entry.select_range(0, tk.END)
       
       def save_edit():
           new_value = var.get()
           
           new_values = list(values)
           new_values[col_idx] = new_value
           self.glossary_tree.item(item, values=new_values)
           
           row_idx = int(self.glossary_tree.item(item)['text']) - 1
           
           if self.current_glossary_format == 'list':
               if 0 <= row_idx < len(self.current_glossary_data):
                   entry = self.current_glossary_data[row_idx]
                   
                   if new_value:
                       entry[col_name] = new_value
                   else:
                       entry.pop(col_name, None)
           
           dialog.destroy()
       
       button_frame = tk.Frame(frame)
       button_frame.pack(fill=tk.X, pady=(10, 0))
       
       tb.Button(button_frame, text="Save", command=save_edit,
                bootstyle="success", width=10).pack(side=tk.LEFT, padx=5)
       tb.Button(button_frame, text="Cancel", command=dialog.destroy,
                bootstyle="secondary", width=10).pack(side=tk.LEFT, padx=5)
       
       dialog.bind('<Return>', lambda e: save_edit())
       dialog.bind('<Escape>', lambda e: dialog.destroy())
       
       dialog.deiconify()

    def convert_glossary_format(self, reload_callback):
        """Export glossary to CSV format"""
        if not self.current_glossary_data:
            messagebox.showerror("Error", "No glossary loaded")
            return
        
        # Create backup before conversion
        if not self.create_glossary_backup("before_export"):
            return
        
        # Get current file path
        current_path = self.editor_file_var.get()
        default_csv_path = current_path.replace('.json', '.csv')
        
        # Ask user for CSV save location
        from tkinter import filedialog
        csv_path = filedialog.asksaveasfilename(
            title="Export Glossary to CSV",
            defaultextension=".csv",
            initialfile=os.path.basename(default_csv_path),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
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
            
            messagebox.showinfo("Success", f"Glossary exported to CSV:\n{csv_path}")
            self.append_log(f"✅ Exported glossary to: {csv_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export CSV: {e}")
            self.append_log(f"❌ CSV export failed: {e}")
