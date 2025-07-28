"""
Metadata and Batch Header Translation Module
Handles custom metadata fields and batch chapter header translation
Complete implementation - no truncation
"""

import os
import json
import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as tb
from typing import Dict, List, Tuple, Optional, Any
import zipfile
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor


class MetadataBatchTranslatorUI:
    """UI handlers for metadata and batch translation features"""
    
    def __init__(self, parent_gui):
        """Initialize with reference to main TranslatorGUI"""
        self.gui = parent_gui
        self.wm = parent_gui.wm
        self.ui = parent_gui.ui
        
        # Initialize default prompts if not in config
        self._initialize_default_prompts()
    
    def _initialize_default_prompts(self):
        """Initialize all default prompts in config if not present"""
        # Batch header prompt
        if 'batch_header_prompt' not in self.gui.config:
            self.gui.config['batch_header_prompt'] = (
                "Translate these chapter titles to English.\n"
                "- For titles with parentheses containing Chinese/Japanese characters (like 終篇, 完結編, etc.), translate both the main title and the parenthetical text.\n"
                "- Common markers: 終篇/終章 = 'Final Chapter', 完結編 = 'Final Arc/Volume', 後編 = 'Part 2', 前編 = 'Part 1'.\n"
                "- Translate the meaning accurately - don't use overly dramatic words unless the original implies them.\n"
                "- Preserve the chapter number format exactly as shown.\n"
                "Return ONLY a JSON object with chapter numbers as keys.\n"
                "Format: {\"1\": \"translated title\", \"2\": \"translated title\"}"
            )
        
        # Metadata batch prompt
        if 'metadata_batch_prompt' not in self.gui.config:
            self.gui.config['metadata_batch_prompt'] = (
                "Translate the following metadata fields to English.\n"
                "Output ONLY a JSON object with the same field names as keys."
            )
        
        # Field-specific prompts
        if 'metadata_field_prompts' not in self.gui.config:
            self.gui.config['metadata_field_prompts'] = {
                'creator': "Translate this author name to English (romanize if needed). Do not output anything other than the translated text:",
                'publisher': "Translate this publisher name to English. Do not output anything other than the translated text:",
                'subject': "Translate this book genre/subject to English. Do not output anything other than the translated text:",
                'description': "Translate this book description to English. Do not output anything other than the translated text:",
                'series': "Translate this series name to English. Do not output anything other than the translated text:",
                '_default': "Translate this text to English. Do not output anything other than the translated text:"
            }
            
    def configure_metadata_fields(self):
            """Configure which metadata fields to translate"""
            # Use scrollable dialog with proper ratios
            dialog, scrollable_frame, canvas = self.wm.setup_scrollable(
                self.gui.master,
                "Configure Metadata Translation",
                width=950,
                height=None,
                max_width_ratio=0.9,
                max_height_ratio=0.7
            )
            
            # Main content
            tk.Label(scrollable_frame, text="Select Metadata Fields to Translate", 
                    font=('TkDefaultFont', 14, 'bold')).pack(pady=(20, 10))
            
            tk.Label(scrollable_frame, text="These fields will be translated along with or separately from the book title:",
                    font=('TkDefaultFont', 10), fg='gray').pack(pady=(0, 20), padx=20)
            
            # Create content frame for fields
            fields_container = tk.Frame(scrollable_frame)
            fields_container.pack(fill=tk.BOTH, expand=True, padx=20)
            
            # Load metadata fields from EPUB
            all_fields = self._detect_all_metadata_fields()
            
            # Standard fields
            standard_fields = {
                'title': ('Title', 'The book title'),
                'creator': ('Author/Creator', 'The author or creator'),
                'publisher': ('Publisher', 'The publishing company'),
                'subject': ('Subject/Genre', 'Subject categories or genres'),
                'description': ('Description', 'Book synopsis'),
                'series': ('Series Name', 'Name of the book series'),
                'language': ('Language', 'Original language'),
                'date': ('Publication Date', 'When published'),
                'rights': ('Rights', 'Copyright information')
            }
            
            field_vars = {}
            
            # Section for standard fields
            tk.Label(fields_container, text="Standard Metadata Fields:", 
                    font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, pady=(10, 5))
            
            # Get saved settings
            translate_fields = self.gui.config.get('translate_metadata_fields', {})
            
            for field, (label, description) in standard_fields.items():
                if field in all_fields:
                    frame = tk.Frame(fields_container)
                    frame.pack(fill=tk.X, pady=5)
                    
                    # Special handling for title field - show note instead of checkbox
                    if field == 'title':
                        # Show the title field info but with a note instead of checkbox
                        tk.Label(frame, text=f"{label}:", width=25, anchor='w',
                                font=('TkDefaultFont', 10, 'bold')).pack(side=tk.LEFT)
                        
                        # Show current value
                        current_value = str(all_fields[field])
                        if len(current_value) > 50:
                            current_value = current_value[:47] + "..."
                        tk.Label(frame, text=current_value, font=('TkDefaultFont', 9), 
                                fg='gray').pack(side=tk.LEFT, padx=(10, 0))
                        
                        # Add note explaining title is controlled elsewhere
                        note_frame = tk.Frame(fields_container)
                        note_frame.pack(fill=tk.X, pady=(0, 10))
                        tk.Label(note_frame, 
                                text="ℹ️ Title translation is controlled by the 'Translate Book Title' setting in the main interface",
                                font=('TkDefaultFont', 9), fg='blue', wraplength=600).pack(anchor=tk.W, padx=(25, 0))
                        continue  # Skip to next field
                    
                    # Normal handling for other fields
                    default_value = False  # All other fields default to False
                    var = tk.BooleanVar(value=translate_fields.get(field, default_value))
                    field_vars[field] = var
                    
                    cb = tb.Checkbutton(frame, text=f"{label}:", variable=var,
                                       bootstyle="round-toggle", width=25)
                    cb.pack(side=tk.LEFT)
                    
                    # Show current value
                    current_value = str(all_fields[field])
                    if len(current_value) > 50:
                        current_value = current_value[:47] + "..."
                    tk.Label(frame, text=current_value, font=('TkDefaultFont', 9), 
                            fg='gray').pack(side=tk.LEFT, padx=(10, 0))
            
            # Custom fields section
            custom_fields = {k: v for k, v in all_fields.items() if k not in standard_fields}
            
            if custom_fields:
                tk.Label(fields_container, text="Custom Metadata Fields:", 
                        font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, pady=(20, 5))
                
                tk.Label(fields_container, text="(Non-standard fields found in your EPUB)", 
                        font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, pady=(0, 10))
                
                for field, value in custom_fields.items():
                    frame = tk.Frame(fields_container)
                    frame.pack(fill=tk.X, pady=5)
                    
                    var = tk.BooleanVar(value=translate_fields.get(field, False))
                    field_vars[field] = var
                    
                    cb = tb.Checkbutton(frame, text=f"{field}:", variable=var,
                                       bootstyle="round-toggle", width=25)
                    cb.pack(side=tk.LEFT)
                    
                    display_value = str(value)
                    if len(display_value) > 50:
                        display_value = display_value[:47] + "..."
                    tk.Label(frame, text=display_value, font=('TkDefaultFont', 9), 
                            fg='gray').pack(side=tk.LEFT, padx=(10, 0))
            
            # Translation mode
            mode_frame = tk.LabelFrame(scrollable_frame, text="Translation Mode", padx=10, pady=10)
            mode_frame.pack(fill=tk.X, pady=(20, 10), padx=20)
            
            translation_mode_var = tk.StringVar(value=self.gui.config.get('metadata_translation_mode', 'together'))
            
            rb1 = tk.Radiobutton(mode_frame, text="Translate together (single API call)",
                                variable=translation_mode_var, value='together')
            rb1.pack(anchor=tk.W, pady=5)
            
            rb2 = tk.Radiobutton(mode_frame, text="Translate separately (parallel API calls)",
                                variable=translation_mode_var, value='parallel')
            rb2.pack(anchor=tk.W, pady=5)
            
            # Buttons
            button_frame = tk.Frame(scrollable_frame)
            button_frame.pack(fill=tk.X, pady=(20, 20), padx=20)
            
            def save_metadata_config():
                # Update configuration
                self.gui.translate_metadata_fields = {}
                for field, var in field_vars.items():
                    if var.get():
                        self.gui.translate_metadata_fields[field] = True
                
                self.gui.config['translate_metadata_fields'] = self.gui.translate_metadata_fields
                self.gui.config['metadata_translation_mode'] = translation_mode_var.get()
                self.gui.save_config()
                
                messagebox.showinfo("Success", 
                                   f"Saved {len(self.gui.translate_metadata_fields)} fields for translation!")
                dialog.destroy()
            
            def reset_metadata_config():
                if messagebox.askyesno("Reset Settings", "Reset all metadata fields to their defaults?"):
                    for field, var in field_vars.items():
                        # Since title is no longer in field_vars, all fields default to False
                        var.set(False)
            
            tb.Button(button_frame, text="Save", command=save_metadata_config,
                     bootstyle="success", width=20).pack(side=tk.LEFT, padx=(0, 10))
            
            tb.Button(button_frame, text="Reset", command=reset_metadata_config,
                     bootstyle="warning-outline", width=20).pack(side=tk.LEFT, padx=(0, 10))
            
            tb.Button(button_frame, text="Cancel", command=dialog.destroy,
                     bootstyle="secondary-outline", width=20).pack(side=tk.LEFT)
            
            # Auto-resize dialog
            self.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.9, max_height_ratio=0.7)
            
            # Handle window close
            dialog.protocol("WM_DELETE_WINDOW", lambda: [
                dialog._cleanup_scrolling() if hasattr(dialog, '_cleanup_scrolling') else None,
                dialog.destroy()
            ])
    
    def configure_translation_prompts(self):
        """Configure all translation prompts in one place"""
        dialog, scrollable_frame, canvas = self.wm.setup_scrollable(
            self.gui.master,
            "Configure Translation Prompts",
            width=1000,
            height=None,
            max_width_ratio=0.9,
            max_height_ratio=1.3
        )
        
        # Title
        tk.Label(scrollable_frame, text="Configure All Translation Prompts", 
                font=('TkDefaultFont', 14, 'bold')).pack(pady=(20, 10))
        
        tk.Label(scrollable_frame, text="Customize how different types of content are translated",
                font=('TkDefaultFont', 10), fg='gray').pack(pady=(0, 20))
        
        # Create notebook for different prompt categories
        notebook = ttk.Notebook(scrollable_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Tab 1: Book Title Prompts
        title_frame = ttk.Frame(notebook)
        notebook.add(title_frame, text="Book Titles")
        self._create_title_prompts_tab(title_frame)
        
        # Tab 2: Chapter Header Prompts
        header_frame = ttk.Frame(notebook)
        notebook.add(header_frame, text="Chapter Headers")
        self._create_header_prompts_tab(header_frame)
        
        # Tab 3: Metadata Field Prompts
        metadata_frame = ttk.Frame(notebook)
        notebook.add(metadata_frame, text="Metadata Fields")
        self._create_metadata_prompts_tab(metadata_frame)
        
        # Tab 4: Advanced Prompts
        advanced_frame = ttk.Frame(notebook)
        notebook.add(advanced_frame, text="Advanced")
        self._create_advanced_prompts_tab(advanced_frame)
        
        # Buttons
        button_frame = tk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=(20, 20), padx=20)
        
        def save_all_prompts():
            # Save all text widgets to config
            self._save_all_prompt_configs()
            self.gui.save_config()
            #messagebox.showinfo("Success", "All prompts saved!")
            dialog.destroy()
        
        def reset_all_prompts():
            if messagebox.askyesno("Reset Prompts", "Reset ALL prompts to defaults?"):
                self._reset_all_prompts_to_defaults()
                messagebox.showinfo("Success", "All prompts reset to defaults!")
                dialog.destroy()
                # Re-open dialog with defaults
                self.configure_translation_prompts()
        
        tb.Button(button_frame, text="Save All", command=save_all_prompts,
                 bootstyle="success", width=20).pack(side=tk.LEFT, padx=(0, 10))
        
        tb.Button(button_frame, text="Reset All to Defaults", command=reset_all_prompts,
                 bootstyle="warning-outline", width=25).pack(side=tk.LEFT, padx=(0, 10))
        
        tb.Button(button_frame, text="Cancel", command=dialog.destroy,
                 bootstyle="secondary-outline", width=20).pack(side=tk.LEFT)
        
        # Auto-resize
        self.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.9, max_height_ratio=1.3)
        
        # Handle close
        dialog.protocol("WM_DELETE_WINDOW", lambda: [
            dialog._cleanup_scrolling() if hasattr(dialog, '_cleanup_scrolling') else None,
            dialog.destroy()
        ])
    
    def _create_title_prompts_tab(self, parent):
        """Create tab for book title prompts"""
        # System prompt
        tk.Label(parent, text="System Prompt (AI Instructions)", 
                font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, padx=20, pady=(20, 5))
        
        tk.Label(parent, text="Defines how the AI should behave when translating titles:",
                font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, padx=20, pady=(0, 10))
        
        self.title_system_text = self.ui.setup_scrollable_text(parent, height=4, wrap=tk.WORD)
        self.title_system_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
        self.title_system_text.insert('1.0', self.gui.config.get('book_title_system_prompt', 
            "You are a translator. Respond with only the translated text, nothing else."))
        
        # User prompt
        tk.Label(parent, text="User Prompt (Translation Request)", 
                font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, padx=20, pady=(10, 5))
        
        self.title_user_text = self.ui.setup_scrollable_text(parent, height=3, wrap=tk.WORD)
        self.title_user_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        self.title_user_text.insert('1.0', self.gui.config.get('book_title_prompt',
            "Translate this book title to English while retaining any acronyms:"))
    
    def _create_header_prompts_tab(self, parent):
        """Create tab for chapter header prompts"""
        tk.Label(parent, text="Batch Chapter Header Translation Prompt", 
                font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, padx=20, pady=(20, 5))
        
        tk.Label(parent, text="Used when translating multiple chapter headers at once:",
                font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, padx=20, pady=(0, 10))
        
        self.header_batch_text = self.ui.setup_scrollable_text(parent, height=6, wrap=tk.WORD)
        self.header_batch_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        self.header_batch_text.insert('1.0', self.gui.config.get('batch_header_prompt',
            "Translate these chapter titles to English.\n"
            "Return ONLY a JSON object with chapter numbers as keys.\n"
            "Format: {\"1\": \"translated title\", \"2\": \"translated title\"}"))
        
        tk.Label(parent, text="Variables available: {source_lang} - detected source language",
                font=('TkDefaultFont', 10), fg='blue').pack(anchor=tk.W, padx=20)
    
    def _create_metadata_prompts_tab(self, parent):
        """Create tab for metadata field prompts"""
        # Batch prompt
        tk.Label(parent, text="Batch Metadata Translation Prompt", 
                font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, padx=20, pady=(20, 5))
        
        tk.Label(parent, text="Used when translating multiple metadata fields together:",
                font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, padx=20, pady=(0, 10))
        
        self.metadata_batch_text = self.ui.setup_scrollable_text(parent, height=4, wrap=tk.WORD)
        self.metadata_batch_text.pack(fill=tk.X, padx=20, pady=(0, 20))
        self.metadata_batch_text.insert('1.0', self.gui.config.get('metadata_batch_prompt',
            "Translate the following metadata fields to English.\n"
            "Return ONLY a JSON object with the same field names as keys."))
        
        # Field-specific prompts
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(parent, text="Field-Specific Prompts", 
                font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        tk.Label(parent, text="Customize prompts for each metadata field type:",
                font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, padx=20, pady=(0, 10))
        
        # NO NESTED SCROLLING - just put fields directly in parent
        # The main dialog already handles scrolling
        field_prompts = self.gui.config.get('metadata_field_prompts', {})
        self.field_prompt_widgets = {}
        
        fields = [
            ('creator', 'Author/Creator'),
            ('publisher', 'Publisher'),
            ('subject', 'Subject/Genre'),
            ('description', 'Description'),
            ('series', 'Series Name'),
            ('_default', 'Default (Other Fields)')
        ]
        
        for field_key, field_label in fields:
            frame = tk.Frame(parent)
            frame.pack(fill=tk.X, pady=10, padx=20)
            
            tk.Label(frame, text=f"{field_label}:", width=20, anchor='w',
                    font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W)
            
            text_widget = tk.Text(frame, height=2, wrap=tk.WORD)
            text_widget.pack(fill=tk.X, pady=(5, 0))
            
            default_prompt = field_prompts.get(field_key, f"Translate this {field_label.lower()} to English:")
            text_widget.insert('1.0', default_prompt)
            
            self.field_prompt_widgets[field_key] = text_widget
        
        tk.Label(parent, text="Variables: {source_lang} - detected language, {field_value} - the text to translate",
                font=('TkDefaultFont', 10), fg='blue').pack(anchor=tk.W, padx=20, pady=(10, 0))
            
    def _create_advanced_prompts_tab(self, parent):
        """Create tab for advanced prompt settings"""
        tk.Label(parent, text="Advanced Prompt Settings", 
                font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, padx=20, pady=(20, 10))
        
        # Language detection behavior
        lang_frame = tk.LabelFrame(parent, text="Language Detection", padx=15, pady=10)
        lang_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(lang_frame, text="How to handle source language in prompts:",
                font=('TkDefaultFont', 10)).pack(anchor=tk.W, pady=(0, 10))
        
        self.lang_behavior_var = tk.StringVar(value=self.gui.config.get('lang_prompt_behavior', 'auto'))
        
        rb1 = tk.Radiobutton(lang_frame, text="Auto-detect and include language (e.g., 'Translate this Korean text')",
                            variable=self.lang_behavior_var, value='auto')
        rb1.pack(anchor=tk.W, pady=2)
        
        rb2 = tk.Radiobutton(lang_frame, text="Never include language (e.g., 'Translate this text')",
                            variable=self.lang_behavior_var, value='never')
        rb2.pack(anchor=tk.W, pady=2)
        
        rb3 = tk.Radiobutton(lang_frame, text="Always specify language:",
                            variable=self.lang_behavior_var, value='always')
        rb3.pack(anchor=tk.W, pady=2)
        
        lang_entry_frame = tk.Frame(lang_frame)
        lang_entry_frame.pack(anchor=tk.W, padx=20, pady=5)
        
        tk.Label(lang_entry_frame, text="Language to use:").pack(side=tk.LEFT)
        self.forced_lang_var = tk.StringVar(value=self.gui.config.get('forced_source_lang', 'Korean'))
        tk.Entry(lang_entry_frame, textvariable=self.forced_lang_var, width=20).pack(side=tk.LEFT, padx=(10, 0))
        
        # Output language
        output_frame = tk.LabelFrame(parent, text="Output Language", padx=15, pady=10)
        output_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(output_frame, text="Target language for translations:",
                font=('TkDefaultFont', 10)).pack(anchor=tk.W, pady=(0, 10))
        
        self.output_lang_var = tk.StringVar(value=self.gui.config.get('output_language', 'English'))
        
        common_langs = ['English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 
                       'Russian', 'Japanese', 'Korean', 'Chinese (Simplified)', 'Chinese (Traditional)']
        
        tk.Label(output_frame, text="Target language:").pack(anchor=tk.W)
        output_combo = tb.Combobox(output_frame, textvariable=self.output_lang_var, 
                                  values=common_langs, state="normal", width=30)
        output_combo.pack(anchor=tk.W, pady=5)
        
        tk.Label(output_frame, text="This will replace 'English' in all prompts with your chosen language",
                font=('TkDefaultFont', 9), fg='gray').pack(anchor=tk.W, pady=(5, 0))
    
    def _save_all_prompt_configs(self):
        """Save all prompt configurations"""
        # Book title prompts
        self.gui.config['book_title_system_prompt'] = self.title_system_text.get('1.0', tk.END).strip()
        self.gui.config['book_title_prompt'] = self.title_user_text.get('1.0', tk.END).strip()
        self.gui.book_title_prompt = self.gui.config['book_title_prompt']
        
        # Batch prompts
        self.gui.config['batch_header_prompt'] = self.header_batch_text.get('1.0', tk.END).strip()
        self.gui.config['metadata_batch_prompt'] = self.metadata_batch_text.get('1.0', tk.END).strip()
        
        # Field-specific prompts
        field_prompts = {}
        for field_key, widget in self.field_prompt_widgets.items():
            field_prompts[field_key] = widget.get('1.0', tk.END).strip()
        self.gui.config['metadata_field_prompts'] = field_prompts
        
        # Advanced settings
        self.gui.config['lang_prompt_behavior'] = self.lang_behavior_var.get()
        self.gui.config['forced_source_lang'] = self.forced_lang_var.get()
        self.gui.config['output_language'] = self.output_lang_var.get()
    
    def _reset_all_prompts_to_defaults(self):
        """Reset all prompts to default values"""
        # Remove prompt-related keys from config
        prompt_keys = [
            'book_title_system_prompt', 'book_title_prompt',
            'batch_header_prompt', 'metadata_batch_prompt',
            'metadata_field_prompts', 'lang_prompt_behavior',
            'forced_source_lang', 'output_language'
        ]
        
        for key in prompt_keys:
            if key in self.gui.config:
                del self.gui.config[key]
        
        # Re-initialize defaults
        self._initialize_default_prompts()
        self.gui.save_config()
    
    def _detect_all_metadata_fields(self) -> Dict[str, str]:
        """Detect ALL metadata fields in the current EPUB"""
        metadata_fields = {}
        
        # Try different possible attribute names for the file path
        epub_path = None
        
        # Common patterns for file path in translator GUIs
        path_attributes = [
            'entry_epub',      # Most common
            'file_entry',      
            'epub_entry',
            'input_entry',
            'file_path_entry',
            'epub_path',
            'file_path',
            'input_file'
        ]
        
        for attr in path_attributes:
            if hasattr(self.gui, attr):
                widget = getattr(self.gui, attr)
                if hasattr(widget, 'get'):
                    epub_path = widget.get()
                    break
                elif isinstance(widget, str):
                    epub_path = widget
                    break
        
        if not epub_path:
            # Try to get from config or recent files
            if hasattr(self.gui, 'config') and 'last_epub_path' in self.gui.config:
                epub_path = self.gui.config.get('last_epub_path', '')
        
        if not epub_path or not epub_path.endswith('.epub'):
            # Return empty dict if no EPUB loaded
            return metadata_fields
        
        try:
            with zipfile.ZipFile(epub_path, 'r') as zf:
                for name in zf.namelist():
                    if name.lower().endswith('.opf'):
                        opf_content = zf.read(name)
                        soup = BeautifulSoup(opf_content, 'xml')
                        
                        # Get Dublin Core elements
                        dc_elements = ['title', 'creator', 'subject', 'description', 
                                      'publisher', 'contributor', 'date', 'type', 
                                      'format', 'identifier', 'source', 'language', 
                                      'relation', 'coverage', 'rights']
                        
                        for element in dc_elements:
                            tag = soup.find(element)
                            if tag and tag.get_text(strip=True):
                                metadata_fields[element] = tag.get_text(strip=True)
                        
                        # Get ALL meta tags
                        meta_tags = soup.find_all('meta')
                        for meta in meta_tags:
                            name = meta.get('name') or meta.get('property', '')
                            content = meta.get('content', '')
                            
                            if name and content:
                                # Clean calibre: prefix
                                if name.startswith('calibre:'):
                                    name = name[8:]
                                
                                metadata_fields[name] = content
                        
                        break
                        
        except Exception as e:
            self.gui.append_log(f"Error reading EPUB metadata: {e}")
        
        return metadata_fields

class BatchHeaderTranslator:
    """Translate chapter headers in batches"""
    
    def __init__(self, client, config: dict = None):
        self.client = client
        self.config = config or {}
        self.stop_flag = False
        self.system_prompt = (
            self.config.get('book_title_system_prompt') or
            os.getenv('BOOK_TITLE_SYSTEM_PROMPT') or
            "You are a translator. Respond with only the translated text, nothing else."
        )
        
    def set_stop_flag(self, flag: bool):
        self.stop_flag = flag
        
    def translate_and_save_headers(self,
                                  html_dir: str,
                                  headers_dict: Dict[int, str], 
                                  batch_size: int = 500,
                                  output_dir: str = None,
                                  update_html: bool = True,
                                  save_to_file: bool = True,
                                  current_titles: Dict[int, Dict[str, str]] = None) -> Dict[int, str]:
        """Translate headers with optional file output and HTML updates
        
        Args:
            html_dir: Directory containing HTML files
            headers_dict: Dict mapping chapter numbers to source titles
            batch_size: Number of titles to translate in one API call
            output_dir: Directory for saving translation file
            update_html: Whether to update HTML files
            save_to_file: Whether to save translations to file
            current_titles: Dict mapping chapter numbers to {'title': str, 'filename': str}
        """
        # Translate headers
        translated_headers = self.translate_headers_batch(
            headers_dict, batch_size
        )
        
        if not translated_headers:
            return {}
        
        # Save to file if requested
        if save_to_file:
            if output_dir is None:
                output_dir = html_dir
            translations_file = os.path.join(output_dir, "translated_headers.txt")
            self._save_translations_to_file(headers_dict, translated_headers, translations_file)
        
        # Update HTML files if requested
        if update_html:
            if current_titles:
                # Use exact replacement method
                self._update_html_headers_exact(html_dir, translated_headers, current_titles)
            else:
                # Fallback to pattern-based method
                self._update_html_headers(html_dir, translated_headers)
        
        return translated_headers
        
    def translate_headers_batch(self, headers_dict: Dict[int, str], batch_size: int = 500) -> Dict[int, str]:
        """Translate headers in batches using configured prompts"""
        if not headers_dict:
            return {}
        
        # Import tiktoken for token counting
        try:
            import tiktoken
            # Try to use model-specific encoding
            try:
                model_name = self.client.model if hasattr(self.client, 'model') else 'gpt-3.5-turbo'
                enc = tiktoken.encoding_for_model(model_name)
            except:
                # Fallback to cl100k_base encoding
                enc = tiktoken.get_encoding("cl100k_base")
            has_tiktoken = True
        except ImportError:
            has_tiktoken = False
            print("[DEBUG] tiktoken not available, using character-based estimation")
        
        def count_tokens(text: str) -> int:
            """Count tokens in text"""
            if has_tiktoken and enc:
                return len(enc.encode(text))
            else:
                # Fallback: estimate ~4 characters per token
                return max(1, len(text) // 4)
        
        # Get configured prompt template
        prompt_template = self.config.get('batch_header_prompt',
            "Translate these chapter titles to English.\n"
            "Return ONLY a JSON object with chapter numbers as keys.\n"
            "Format: {\"1\": \"translated title\", \"2\": \"translated title\"}")
        
        # Handle language in prompt
        source_lang = _get_source_language()
        lang_behavior = self.config.get('lang_prompt_behavior', 'auto')
        
        if lang_behavior == 'never':
            lang_str = ""
        elif lang_behavior == 'always':
            lang_str = self.config.get('forced_source_lang', 'Korean')
        else:  # auto
            lang_str = source_lang if source_lang else ""
        
        # Handle output language
        output_lang = self.config.get('output_language', 'English')
        
        # Replace variables in prompt
        prompt_template = prompt_template.replace('{source_lang}', lang_str)
        prompt_template = prompt_template.replace('English', output_lang)
        
        # Add the titles to translate
        user_prompt_template = prompt_template + "\n\nTitles to translate:\n"
        
        sorted_headers = sorted(headers_dict.items())
        all_translations = {}
        total_batches = (len(sorted_headers) + batch_size - 1) // batch_size
        
        # Get temperature and max_tokens from environment (passed by GUI) or config as fallback
        temperature = float(os.getenv('TRANSLATION_TEMPERATURE', self.config.get('temperature', 0.3)))
        max_tokens = int(os.getenv('MAX_OUTPUT_TOKENS', self.config.get('max_tokens', 4096)))
        
        print(f"[DEBUG] Using temperature: {temperature}, max_tokens: {max_tokens} (from GUI/env)")
        
        # Count system prompt tokens once
        system_tokens = count_tokens(self.system_prompt)
        print(f"[DEBUG] System prompt tokens: {system_tokens}")
        
        for batch_num in range(total_batches):
            if self.stop_flag:
                print("Translation interrupted by user")
                break
                
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(sorted_headers))
            batch_headers = dict(sorted_headers[start_idx:end_idx])
            
            print(f"\n📚 Translating header batch {batch_num + 1}/{total_batches}")
            
            try:
                titles_json = json.dumps(batch_headers, ensure_ascii=False, indent=2)
                user_prompt = user_prompt_template + titles_json
                
                # Count tokens in the user prompt
                user_tokens = count_tokens(user_prompt)
                total_input_tokens = system_tokens + user_tokens
                
                # Debug output showing input tokens
                print(f"[DEBUG] Batch {batch_num + 1} input tokens:")
                print(f"  - User prompt: {user_tokens} tokens")
                print(f"  - Total input: {total_input_tokens} tokens (including system prompt)")
                print(f"  - Headers in batch: {len(batch_headers)}")
                
                # Show a sample of the headers being translated (first 3)
                sample_headers = list(batch_headers.items())[:3]
                if sample_headers:
                    print(f"[DEBUG] Sample headers being sent:")
                    for ch_num, title in sample_headers:
                        print(f"    Chapter {ch_num}: {title}")
                    if len(batch_headers) > 3:
                        print(f"    ... and {len(batch_headers) - 3} more")
                
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                # Pass temperature and max_tokens explicitly
                response = self.client.send(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    context='batch_header_translation'
                )
                
                # Extract content from response - handle both object and tuple formats
                response_content = None
                if hasattr(response, 'content'):
                    response_content = response.content
                elif isinstance(response, tuple):
                    # If it's a tuple, first element is usually the content
                    response_content = response[0] if response else ""
                else:
                    # Fallback: convert to string
                    response_content = str(response)
                
                if response_content:
                    translations = self._parse_json_response(response_content, batch_headers)
                    all_translations.update(translations)
                    
                    # Count output tokens for debug
                    output_tokens = count_tokens(response_content)
                    print(f"[DEBUG] Response tokens: {output_tokens}")
                    
                    for num, translated in translations.items():
                        if num in batch_headers:
                            print(f"  ✓ Ch{num}: {batch_headers[num]} → {translated}")
                else:
                    print(f"  ⚠️ Empty response from API")
                    
            except json.JSONDecodeError as e:
                print(f"  ❌ Failed to parse JSON response: {e}")
                # Try to extract translations manually from the response
                if response_content:
                    translations = self._fallback_parse(response_content, batch_headers)
                    all_translations.update(translations)
            except Exception as e:
                print(f"  ❌ Error in batch {batch_num + 1}: {e}")
                continue
        
        print(f"\n✅ Translated {len(all_translations)} headers total")
        return all_translations
    
    def _parse_json_response(self, response: str, original_headers: Dict[int, str]) -> Dict[int, str]:
        """Parse JSON response from API"""
        try:
            response = response.strip()
            
            # Remove markdown blocks
            if response.startswith("```"):
                lines = response.split('\n')
                response_lines = []
                in_code_block = False
                
                for line in lines:
                    if line.strip().startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        response_lines.append(line)
                
                response = '\n'.join(response_lines)
            
            parsed = json.loads(response)
            
            result = {}
            for key, value in parsed.items():
                try:
                    chapter_num = int(key)
                    if chapter_num in original_headers:
                        result[chapter_num] = str(value).strip()
                except (ValueError, TypeError):
                    continue
                    
            return result
            
        except json.JSONDecodeError:
            return self._fallback_parse(response, original_headers)
        except Exception:
            return {}
    
    def _fallback_parse(self, response: str, original_headers: Dict[int, str]) -> Dict[int, str]:
        """Fallback parsing if JSON fails"""
        result = {}
        pattern = r'["\']?(\d+)["\']?\s*:\s*["\']([^"\']+)["\']'
        
        for match in re.finditer(pattern, response):
            try:
                num = int(match.group(1))
                title = match.group(2).strip()
                if num in original_headers and title:
                    result[num] = title
            except:
                continue
                
        return result
    
    def _save_translations_to_file(self, 
                                  original: Dict[int, str], 
                                  translated: Dict[int, str],
                                  output_path: str):
        """Save translations to text file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("Chapter Header Translations\n")
                f.write("=" * 50 + "\n\n")
                
                # Sort chapter numbers, ensuring chapter 0 comes first if present
                chapter_numbers = sorted(original.keys())
                
                # Summary info
                total_chapters = len(original)
                successfully_translated = len(translated)
                
                # Check if we have chapter 0
                has_chapter_zero = 0 in chapter_numbers
                if has_chapter_zero:
                    f.write(f"Note: This novel uses 0-based chapter numbering (starts with Chapter 0)\n")
                    f.write("-" * 50 + "\n\n")
                
                # Write each chapter's translation
                for num in chapter_numbers:
                    orig_title = original.get(num, "Unknown")
                    trans_title = translated.get(num, orig_title)
                    
                    f.write(f"Chapter {num}:\n")
                    f.write(f"  Original:   {orig_title}\n")
                    f.write(f"  Translated: {trans_title}\n")
                    
                    # Mark if translation failed for this chapter
                    if num not in translated:
                        f.write(f"  Status:     ⚠️ Using original (translation failed)\n")
                    
                    f.write("-" * 40 + "\n")
                
                # Summary at the end
                f.write(f"\nSummary:\n")
                f.write(f"Total chapters: {total_chapters}\n")
                f.write(f"Chapter range: {min(chapter_numbers)} to {max(chapter_numbers)}\n")
                f.write(f"Successfully translated: {successfully_translated}\n")
                
                if successfully_translated < total_chapters:
                    failed_chapters = [num for num in original if num not in translated]
                    f.write(f"Failed chapters: {', '.join(map(str, failed_chapters))}\n")
                
            print(f"✅ Saved translations to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving translations: {e}")
    
    def _update_html_headers_exact(self, html_dir: str, translated_headers: Dict[int, str], 
                                  current_titles: Dict[int, Dict[str, str]]):
        """Update HTML files by replacing exact current titles with translations"""
        updated_count = 0
        
        for num, new_title in translated_headers.items():
            if num not in current_titles:
                print(f"⚠️ No HTML file mapping for chapter {num}")
                continue
                
            current_info = current_titles[num]
            current_title = current_info['title']
            html_file = current_info['filename']
            html_path = os.path.join(html_dir, html_file)
            
            try:
                with open(html_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                soup = BeautifulSoup(content, 'html.parser')
                updated = False
                
                # Update title tag
                title_tag = soup.find('title')
                if title_tag and title_tag.get_text().strip() == current_title:
                    title_tag.string = new_title
                    updated = True
                
                # Update ALL elements that contain the exact current title
                # This includes h1, h2, h3, p, etc.
                for element in soup.find_all(text=current_title):
                    if element.parent.name not in ['script', 'style']:
                        element.replace_with(new_title)
                        updated = True
                
                # Also check elements where the text might have extra whitespace
                for tag in ['h1', 'h2', 'h3', 'p', 'div', 'span']:
                    for element in soup.find_all(tag):
                        if element.get_text().strip() == current_title:
                            element.clear()
                            element.string = new_title
                            updated = True
                
                # Update meta og:title if it matches
                meta_title = soup.find('meta', {'property': 'og:title'})
                if meta_title and meta_title.get('content', '').strip() == current_title:
                    meta_title['content'] = new_title
                    updated = True
                
                if updated:
                    # Write back with proper encoding
                    html_str = str(soup)
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(html_str)
                    updated_count += 1
                    print(f"✓ Updated {html_file}: '{current_title}' → '{new_title}'")
                else:
                    print(f"⚠️ Could not find '{current_title}' in {html_file}")
                    
            except Exception as e:
                print(f"❌ Error updating {html_file}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n📝 Updated {updated_count} HTML files")
    
    def _update_html_headers(self, html_dir: str, translated_headers: Dict[int, str]):
        """Fallback: Update HTML files with translated headers using pattern matching"""
        updated_count = 0
        
        # Get all HTML files in directory
        all_html_files = [f for f in os.listdir(html_dir) if f.endswith('.html')]
        
        for num, new_title in translated_headers.items():
            # Try multiple filename patterns
            possible_patterns = [
                f"response_{num}_",  # Standard pattern
                f"response_{num}.",  # With dot
                f"chapter_{num}_",   # Alternative pattern
                f"chapter{num}_",    # Without underscore
                f"{num}_",           # Just number
                f"{num}.",           # Number with dot
                f"ch{num}_",         # Abbreviated
                f"ch_{num}_",        # Abbreviated with underscore
            ]
            
            html_file = None
            for pattern in possible_patterns:
                matching_files = [f for f in all_html_files if f.startswith(pattern)]
                if matching_files:
                    html_file = matching_files[0]
                    break
            
            if not html_file:
                # Last resort: check if any file contains the chapter number
                for f in all_html_files:
                    if re.search(rf'\b{num}\b', f):
                        html_file = f
                        break
            
            if not html_file:
                print(f"⚠️ No HTML file found for chapter {num}")
                continue
                
            html_path = os.path.join(html_dir, html_file)
            
            try:
                with open(html_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                soup = BeautifulSoup(content, 'html.parser')
                updated = False
                
                # Update title tag
                title_tag = soup.find('title')
                if title_tag:
                    title_tag.string = new_title
                    updated = True
                
                # Update first h1, h2, or h3 tag
                for tag in ['h1', 'h2', 'h3']:
                    header = soup.find(tag)
                    if header:
                        header.clear()
                        header.string = new_title
                        updated = True
                        break
                
                # Update meta og:title
                meta_title = soup.find('meta', {'property': 'og:title'})
                if meta_title:
                    meta_title['content'] = new_title
                    updated = True
                
                if updated:
                    # Ensure proper encoding
                    html_str = str(soup)
                    # Preserve Unicode characters
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(html_str)
                    updated_count += 1
                    print(f"✓ Updated {html_file} with: {new_title}")
                    
            except Exception as e:
                print(f"❌ Error updating chapter {num}: {e}")
        
        print(f"\n📝 Updated {updated_count} HTML files")

class MetadataTranslator:
    """Translate EPUB metadata fields"""
    
    def __init__(self, client, config: dict = None):
        self.client = client
        self.config = config or {}
        self.system_prompt = os.getenv('BOOK_TITLE_SYSTEM_PROMPT',
            "You are a translator. Respond with only the translated text, nothing else.")
    
    def translate_metadata(self, 
                          metadata: Dict[str, Any],
                          fields_to_translate: Dict[str, bool],
                          mode: str = 'together') -> Dict[str, Any]:
        """Translate selected metadata fields"""
        if not any(fields_to_translate.values()):
            return metadata
            
        translated_metadata = metadata.copy()
        
        if mode == 'together':
            translated_fields = self._translate_fields_together(
                metadata, fields_to_translate
            )
            translated_metadata.update(translated_fields)
        else:
            translated_fields = self._translate_fields_parallel(
                metadata, fields_to_translate
            )
            translated_metadata.update(translated_fields)
            
        return translated_metadata
    
    def _translate_fields_together(self, 
                                  metadata: Dict[str, Any],
                                  fields_to_translate: Dict[str, bool]) -> Dict[str, Any]:
        """Translate all fields in one API call"""
        fields_to_send = {}
        
        for field, should_translate in fields_to_translate.items():
            if should_translate and field in metadata and metadata[field]:
                fields_to_send[field] = metadata[field]
                
        if not fields_to_send:
            return {}
        
        # Get configured prompt
        prompt_template = self.config.get('metadata_batch_prompt',
            "Translate the following metadata fields to English.\n"
            "Return ONLY a JSON object with the same field names as keys.")
        
        # Handle language behavior
        lang_behavior = self.config.get('lang_prompt_behavior', 'auto')
        source_lang = _get_source_language()
        
        if lang_behavior == 'never':
            lang_str = ""
        elif lang_behavior == 'always':
            lang_str = self.config.get('forced_source_lang', 'Korean')
        else:  # auto
            lang_str = source_lang if source_lang else ""
        
        # Handle output language
        output_lang = self.config.get('output_language', 'English')
        
        # Replace variables
        prompt_template = prompt_template.replace('{source_lang}', lang_str)
        prompt_template = prompt_template.replace('English', output_lang)
        
        user_prompt = prompt_template + f"\n\nFields to translate:\n{json.dumps(fields_to_send, ensure_ascii=False, indent=2)}"
        
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.client.send(messages)
            translated = self._parse_metadata_response(response.content, fields_to_send)
            
            for field, value in translated.items():
                if field in metadata:
                    print(f"✓ Translated {field}: {metadata[field]} → {value}")
                    
            return translated
            
        except Exception as e:
            print(f"Error translating metadata: {e}")
            return {}
    
    def _translate_fields_parallel(self,
                                  metadata: Dict[str, Any],
                                  fields_to_translate: Dict[str, bool]) -> Dict[str, Any]:
        """Translate fields in parallel"""
        fields_to_process = [
            (field, value) for field, should_translate in fields_to_translate.items()
            if should_translate and (value := metadata.get(field))
        ]
        
        if not fields_to_process:
            return {}
            
        translated = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            
            for field, value in fields_to_process:
                future = executor.submit(self._translate_single_field, field, value)
                futures[future] = field
                
            for future in futures:
                field = futures[future]
                try:
                    result = future.result(timeout=30)
                    if result:
                        translated[field] = result
                        print(f"✓ Translated {field}: {metadata.get(field)} → {result}")
                except Exception as e:
                    print(f"❌ Error translating {field}: {e}")
                    
        return translated
    
    def _translate_single_field(self, field_name: str, field_value: str) -> Optional[str]:
        """Translate a single field using configured prompts"""
        # Get field-specific prompts
        field_prompts = self.config.get('metadata_field_prompts', {})
        
        # Get the specific prompt or default
        prompt_template = field_prompts.get(field_name, 
                                           field_prompts.get('_default', 
                                                           "Translate this text to English:"))
        
        # Handle language behavior
        lang_behavior = self.config.get('lang_prompt_behavior', 'auto')
        source_lang = _get_source_language()
        
        if lang_behavior == 'never':
            lang_str = ""
        elif lang_behavior == 'always':
            lang_str = self.config.get('forced_source_lang', 'Korean')
        else:  # auto
            lang_str = source_lang if source_lang else ""
        
        # Handle output language
        output_lang = self.config.get('output_language', 'English')
        
        # Replace variables
        prompt = prompt_template.replace('{source_lang}', lang_str)
        prompt = prompt.replace('{field_value}', field_value)
        prompt = prompt.replace('English', output_lang)
        
        # Clean up double spaces
        prompt = ' '.join(prompt.split())
        
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"{prompt}\n\n{field_value}"}
            ]
            
            response = self.client.send(messages)
            return response.content.strip()
            
        except Exception as e:
            print(f"Error translating {field_name}: {e}")
            return None
    
    def _parse_metadata_response(self, response: str, original_fields: Dict[str, str]) -> Dict[str, str]:
        """Parse metadata response"""
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.split("```")[0]
                
            parsed = json.loads(response.strip())
            
            result = {}
            for field, value in parsed.items():
                if field in original_fields and value:
                    result[field] = str(value).strip()
                    
            return result
            
        except Exception as e:
            print(f"Error parsing response: {e}")
            return {}


def _get_source_language() -> str:
    """
    Get source language from EPUB metadata or detect from content
    
    NOTE: This properly detects language from the actual content/metadata
    instead of inferring from profile names (which are for prompt presets)
    """
    # Try to get language from metadata first
    metadata_path = os.path.join(os.path.dirname(os.getenv('EPUB_PATH', '')), 'metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                # Check for language field
                lang = metadata.get('language', '').lower()
                if 'ko' in lang or 'korean' in lang:
                    return 'Korean'
                elif 'ja' in lang or 'japanese' in lang:
                    return 'Japanese'
                elif 'zh' in lang or 'chinese' in lang:
                    return 'Chinese'
        except:
            pass
    
    # If no metadata, return empty string so prompts are generic
    return ''


def enhance_epub_compiler(compiler_instance):
    """Enhance an EPUBCompiler instance with translation features"""
    # Get settings from environment
    translate_metadata_fields = {}
    try:
        fields_str = os.getenv('TRANSLATE_METADATA_FIELDS', '{}')
        translate_metadata_fields = json.loads(fields_str)
        if translate_metadata_fields:
            print(f"[DEBUG] Metadata fields to translate: {translate_metadata_fields}")
        else:
            print("[DEBUG] No metadata fields configured for translation")
    except Exception as e:
        print(f"[ERROR] Failed to parse TRANSLATE_METADATA_FIELDS: {e}")
        translate_metadata_fields = {}
    
    batch_translate = os.getenv('BATCH_TRANSLATE_HEADERS', '0') == '1'
    headers_per_batch = int(os.getenv('HEADERS_PER_BATCH', '400'))
    update_html = os.getenv('UPDATE_HTML_HEADERS', '1') == '1'
    save_translations = os.getenv('SAVE_HEADER_TRANSLATIONS', '1') == '1'
    
    # Add settings to compiler
    compiler_instance.translate_metadata_fields = translate_metadata_fields
    compiler_instance.metadata_translation_mode = os.getenv('METADATA_TRANSLATION_MODE', 'together')
    compiler_instance.batch_translate_headers = batch_translate
    compiler_instance.headers_per_batch = headers_per_batch
    compiler_instance.update_html_headers = update_html
    compiler_instance.save_header_translations = save_translations
    
    # Log what we're setting
    print(f"[DEBUG] Compiler settings:")
    print(f"  - translate_metadata_fields: {compiler_instance.translate_metadata_fields}")
    print(f"  - metadata_translation_mode: {compiler_instance.metadata_translation_mode}")
    print(f"  - batch_translate_headers: {compiler_instance.batch_translate_headers}")
    
    # extraction method with mapping support
    compiler_instance._extract_source_headers_and_current_titles = lambda: extract_source_headers_and_current_titles(
        os.getenv('EPUB_PATH', ''), 
        compiler_instance.html_dir,
        compiler_instance.log
    )
    
    # Create translators if needed
    needs_translators = batch_translate or any(translate_metadata_fields.values())
    print(f"[DEBUG] Needs translators: {needs_translators} (batch={batch_translate}, metadata={any(translate_metadata_fields.values())})")
    
    if compiler_instance.api_client and needs_translators:
        # Try to get config from multiple locations
        config_paths = [
            os.path.join(compiler_instance.base_dir, '..', 'config.json'),
            os.path.join(os.path.dirname(compiler_instance.base_dir), 'config.json'),
            os.path.join(os.path.dirname(os.path.dirname(compiler_instance.base_dir)), 'config.json'),
            'config.json'  # Current directory as last resort
        ]
        
        config = {}
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    print(f"[DEBUG] Loaded config from: {config_path}")
                    break
                except Exception as e:
                    print(f"[WARNING] Failed to load config from {config_path}: {e}")
        
        # PRIORITY: Use GUI values from environment first, then config as fallback
        
        # Get temperature - GUI passes this via TRANSLATION_TEMPERATURE env var
        env_temp = os.getenv('TRANSLATION_TEMPERATURE')
        if env_temp:
            try:
                config['temperature'] = float(env_temp)
                print(f"[DEBUG] Using temperature from GUI (env): {config['temperature']}")
            except ValueError:
                print(f"[WARNING] Invalid temperature value: {env_temp}")
                config['temperature'] = 0.3
        elif 'translation_temperature' in config:
            config['temperature'] = config['translation_temperature']
            print(f"[DEBUG] Using temperature from config: {config['temperature']}")
        else:
            config['temperature'] = 0.3  # Last resort default
            print(f"[DEBUG] Using default temperature: {config['temperature']}")
            
        # Get max_tokens - GUI passes this via MAX_OUTPUT_TOKENS env var
        env_max_tokens = os.getenv('MAX_OUTPUT_TOKENS')
        if env_max_tokens and env_max_tokens.isdigit():
            config['max_tokens'] = int(env_max_tokens)
            print(f"[DEBUG] Using max_tokens from GUI (env): {config['max_tokens']}")
        elif 'max_output_tokens' in config:
            config['max_tokens'] = config['max_output_tokens']
            print(f"[DEBUG] Using max_tokens from config: {config['max_tokens']}")
        else:
            config['max_tokens'] = 4096  # Last resort default
            print(f"[DEBUG] Using default max_tokens: {config['max_tokens']}")
            
        # Set temperature and max_tokens on the client if possible
        if hasattr(compiler_instance.api_client, 'default_temperature'):
            compiler_instance.api_client.default_temperature = config['temperature']
        if hasattr(compiler_instance.api_client, 'default_max_tokens'):
            compiler_instance.api_client.default_max_tokens = config['max_tokens']
        
        # Get compression factor from environment or config
        compression_factor = float(os.getenv('COMPRESSION_FACTOR', '1.0'))
        if hasattr(compiler_instance.api_client, 'compression_factor'):
            compiler_instance.api_client.compression_factor = compression_factor
            print(f"[DEBUG] Set compression factor: {compression_factor}")
        
        try:
            # Create batch header translator if needed
            if batch_translate:
                compiler_instance.header_translator = BatchHeaderTranslator(
                    compiler_instance.api_client, config
                )
                print(f"[DEBUG] Created BatchHeaderTranslator")
            
            # Create metadata translator if needed
            if any(translate_metadata_fields.values()):
                compiler_instance.metadata_translator = MetadataTranslator(
                    compiler_instance.api_client, config
                )
                print(f"[DEBUG] Created MetadataTranslator for fields: {[k for k, v in translate_metadata_fields.items() if v]}")
                
                # Verify the translator was created
                if hasattr(compiler_instance, 'metadata_translator'):
                    print("[DEBUG] MetadataTranslator successfully attached to compiler")
                else:
                    print("[ERROR] MetadataTranslator not attached to compiler!")
            else:
                print("[DEBUG] No metadata fields selected for translation")
                
        except Exception as e:
            print(f"[ERROR] Failed to initialize translators: {e}")
            import traceback
            traceback.print_exc()
    else:
        if not compiler_instance.api_client:
            print("[WARNING] No API client available for translation")
        if not needs_translators:
            print("[DEBUG] No translation features requested")
    
    return compiler_instance
    
def extract_source_headers_and_current_titles(epub_path: str, html_dir: str, log_callback=None) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Extract source headers AND current titles from HTML files
    
    Returns:
        Tuple of (source_headers, current_titles) where:
        - source_headers: Maps chapter numbers to original language titles from source EPUB
        - current_titles: Maps chapter numbers to current titles in HTML files
    """
    from bs4 import BeautifulSoup
    import json
    import os
    import zipfile
    
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)
    
    log("📖 Extracting headers and mapping to output files...")
    
    # Step 1: Get ALL HTML files and sort them
    all_html_files = sorted([f for f in os.listdir(html_dir) if f.endswith('.html')])
    log(f"📁 Found {len(all_html_files)} HTML files in {html_dir}")
    
    # Step 2: Load translation_progress.json to understand the chapter mapping
    progress_file = os.path.join(html_dir, 'translation_progress.json')
    progress_data = {}
    chapter_to_file_map = {}
    has_chapter_zero = False
    uses_zero_based = False  # Track if the novel uses 0-based numbering
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            # Check if novel uses 0-based numbering
            uses_zero_based = progress_data.get('uses_zero_based', False)
            
            # Scan ALL entries in progress.json - using hash keys
            all_chapters = progress_data.get('chapters', {})
            log(f"📊 Scanning {len(all_chapters)} entries in translation_progress.json")
            log(f"📊 Novel uses {'0-based' if uses_zero_based else '1-based'} numbering")
            
            # First pass: check if we have chapter 0
            for chapter_hash, chapter_info in all_chapters.items():
                if isinstance(chapter_info, dict):  # Make sure it's a valid chapter entry
                    actual_num = chapter_info.get('actual_num')
                    if actual_num == 0:
                        has_chapter_zero = True
                        log("  ✓ Found Chapter 0 in translation_progress.json")
                        break
            
            # Second pass: build complete mapping
            for chapter_hash, chapter_info in all_chapters.items():
                if isinstance(chapter_info, dict):  # Make sure it's a valid chapter entry
                    # Check if either has 'completed' status OR just has output_file (for compatibility)
                    has_output = chapter_info.get('output_file')
                    is_completed = chapter_info.get('status') == 'completed' or has_output
                    
                    if is_completed and has_output:
                        actual_num = chapter_info.get('actual_num')
                        output_file = os.path.basename(chapter_info['output_file'])
                        
                        # Important: Check for None explicitly to include chapter 0
                        if actual_num is not None and output_file in all_html_files:
                            chapter_to_file_map[actual_num] = output_file
                        elif actual_num is None:
                            log(f"  ⚠️ Skipping entry {chapter_hash[:8]}...: missing actual_num")
                        elif output_file not in all_html_files:
                            log(f"  ⚠️ Skipping chapter {actual_num}: file '{output_file}' not found in directory")
                            
            log(f"📊 Found {len(chapter_to_file_map)} chapter mappings in translation_progress.json")
            
            # Show chapter number range
            if chapter_to_file_map:
                min_ch = min(chapter_to_file_map.keys())
                max_ch = max(chapter_to_file_map.keys())
                log(f"  Chapter range: {min_ch} to {max_ch}")
                        
        except Exception as e:
            log(f"⚠️ Could not load translation_progress.json: {e}")
    
    # Step 3: Extract current titles from HTML files
    current_titles = {}
    
    # If we have a mapping from progress.json, use it
    if chapter_to_file_map:
        for chapter_num, html_file in chapter_to_file_map.items():
            try:
                html_path = os.path.join(html_dir, html_file)
                with open(html_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                soup = BeautifulSoup(content, 'html.parser')
                
                # Get title from the HTML
                current_title = None
                
                # Try h1, h2, h3, title in order
                for tag_name in ['h1', 'h2', 'h3', 'title']:
                    tag = soup.find(tag_name)
                    if tag:
                        text = tag.get_text().strip()
                        if text:
                            current_title = text
                            break
                
                if not current_title:
                    current_title = f"Chapter {chapter_num}"
                
                current_titles[chapter_num] = {
                    'title': current_title,
                    'filename': html_file
                }
                log(f"  Ch.{chapter_num} ({html_file}): {current_title}")
                    
            except Exception as e:
                log(f"⚠️ Error reading {html_file}: {e}")
    else:
        # No translation_progress.json mapping - use file order
        log("⚠️ No translation_progress.json mapping found, using file order")
        # Start from 0 if we detected chapter 0, otherwise from 1
        start_num = 0 if has_chapter_zero else 1
        
        for idx, html_file in enumerate(all_html_files):
            chapter_num = idx + start_num
            try:
                html_path = os.path.join(html_dir, html_file)
                with open(html_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                soup = BeautifulSoup(content, 'html.parser')
                
                # Get title
                current_title = None
                for tag_name in ['h1', 'h2', 'h3', 'title']:
                    tag = soup.find(tag_name)
                    if tag:
                        text = tag.get_text().strip()
                        if text:
                            current_title = text
                            break
                
                if not current_title:
                    current_title = f"Chapter {chapter_num}"
                
                current_titles[chapter_num] = {
                    'title': current_title,
                    'filename': html_file
                }
                log(f"  Ch.{chapter_num} ({html_file}): {current_title}")
                    
            except Exception as e:
                log(f"⚠️ Error reading {html_file}: {e}")
    
    log(f"📊 Found {len(current_titles)} current titles in HTML files")
    
    # Step 4: Extract headers from source EPUB
    source_headers = {}
    
    if not os.path.exists(epub_path):
        log(f"⚠️ Source EPUB not found: {epub_path}")
        return source_headers, current_titles
    
    try:
        with zipfile.ZipFile(epub_path, 'r') as zf:
            # Get all content files
            epub_html_files = sorted([f for f in zf.namelist() 
                                     if f.endswith(('.html', '.xhtml', '.htm')) 
                                     and not f.startswith('__MACOSX')
                                     and not any(skip in f.lower() for skip in ['nav.', 'toc.', 'contents.', 'copyright.', 'cover.'])])
            
            log(f"📚 Found {len(epub_html_files)} content files in source EPUB")
            
            # FIRST: Extract ALL titles from source EPUB files
            source_titles_by_index = {}
            
            for idx, content_file in enumerate(epub_html_files):
                try:
                    html_content = zf.read(content_file).decode('utf-8', errors='ignore')
                    
                    if not html_content:
                        continue
                    
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Extract title
                    title = None
                    
                    for tag_name in ['h1', 'h2', 'h3', 'title']:
                        tag = soup.find(tag_name)
                        if tag:
                            text = tag.get_text().strip()
                            if text:
                                title = text
                                break
                    
                    # If no title found, check first paragraph
                    if not title:
                        p = soup.find('p')
                        if p:
                            text = p.get_text().strip()
                            if text and len(text) < 100:
                                title = text
                    
                    if title:
                        source_titles_by_index[idx] = title
                        log(f"  Extracted from source idx {idx}: {title}")
                    
                except Exception as e:
                    log(f"  ⚠️ Error reading source chapter {idx}: {e}")
                    continue
            
            log(f"📚 Extracted {len(source_titles_by_index)} titles from source EPUB")
            
            # SECOND: Try to build mapping using content_hashes
            source_to_output = {}
            
            if progress_data and 'content_hashes' in progress_data:
                content_hashes = progress_data.get('content_hashes', {})
                chapters_data = progress_data.get('chapters', {})
                
                log(f"  Building mapping from {len(content_hashes)} content hash entries...")
                
                for chapter_hash, hash_info in content_hashes.items():
                    if not isinstance(hash_info, dict):
                        continue
                    
                    # Get chapter_idx from content_hashes section
                    chapter_idx = hash_info.get('chapter_idx')
                    actual_num = hash_info.get('actual_num')
                    
                    # Check if this chapter is completed in the chapters section
                    chapter_info = chapters_data.get(chapter_hash, {})
                    has_output = chapter_info.get('output_file')
                    is_completed = chapter_info.get('status') == 'completed' or has_output
                    
                    if is_completed and chapter_idx is not None and actual_num is not None:
                        source_to_output[chapter_idx] = actual_num
            
            log(f"  Mapping from content_hashes: {len(source_to_output)} chapters mapped")
            
            # THIRD: Fill in any missing mappings with direct mapping
            # This ensures we don't lose chapters like 64
            expected_chapters = set(current_titles.keys())
            mapped_chapters = set(source_to_output.values())
            missing_chapters = expected_chapters - mapped_chapters
            
            if missing_chapters:
                log(f"⚠️ Missing mappings for chapters: {sorted(missing_chapters)}")
                log(f"  Will attempt direct index mapping for missing chapters")
                
                # For missing chapters, try direct index mapping
                for missing_ch in sorted(missing_chapters):
                    # Try to find source index for this chapter
                    # First, check if the chapter number matches a source index
                    if has_chapter_zero or uses_zero_based:
                        # For 0-based: chapter N maps to source index N
                        source_idx = missing_ch
                    else:
                        # For 1-based: chapter N maps to source index N-1
                        source_idx = missing_ch - 1
                    
                    # Check if this source index exists and isn't already mapped
                    if source_idx in source_titles_by_index and source_idx not in source_to_output:
                        source_to_output[source_idx] = missing_ch
                        log(f"    Mapped missing chapter: source idx {source_idx} → chapter {missing_ch}")
            
            # FOURTH: Apply the mapping to create source_headers
            for source_idx, output_num in source_to_output.items():
                if source_idx in source_titles_by_index:
                    source_headers[output_num] = source_titles_by_index[source_idx]
                    log(f"  Final mapping: Source Ch.{source_idx} → Output Ch.{output_num}: {source_titles_by_index[source_idx]}")
            
            log(f"📊 Final result: {len(source_headers)} source headers mapped to output chapters")
            
            # Debug: Show what we have vs what we expect
            missing_final = []
            for chapter_num in current_titles:
                if chapter_num not in source_headers:
                    missing_final.append(chapter_num)
            
            if missing_final:
                log(f"❌ Still missing source headers for chapters: {sorted(missing_final)}")
                # Try to debug why
                for m in missing_final[:3]:  # Show first 3
                    log(f"  Chapter {m}: current title = {current_titles[m]['title']}")
        
    except Exception as e:
        log(f"❌ Error extracting source headers: {e}")
        import traceback
        log(traceback.format_exc())
    
    return source_headers, current_titles
