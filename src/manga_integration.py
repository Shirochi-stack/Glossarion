# manga_integration.py
"""
GUI Integration module for Manga Translation
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
        
        # Build interface
        self._build_interface()
        
        # Start update loop
        self._process_updates()
    
    def _build_interface(self):
        """Build the manga translation interface"""
        # Title
        title_frame = tk.Frame(self.parent_frame)
        title_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        title_label = tk.Label(
            title_frame,
            text="Manga Panel Translator",
            font=('Arial', 24, 'bold')
        )
        title_label.pack(side=tk.LEFT)
        
        # Status checks
        has_api_key = False
        if hasattr(self.main_gui, 'api_key_entry'):
            api_key = self.main_gui.api_key_entry.get()
            has_api_key = bool(api_key and api_key.strip())
        elif hasattr(self.main_gui, 'client') and hasattr(self.main_gui.client, 'api_key'):
            has_api_key = bool(self.main_gui.client.api_key)
        
        has_vision = GOOGLE_CLOUD_VISION_AVAILABLE
        
        if has_api_key and has_vision:
            status_text = "✅ Ready (Google Cloud Vision + API Key)"
            status_color = "green"
        elif not has_vision:
            status_text = "❌ Google Cloud Vision not installed"
            status_color = "red"
        elif not has_api_key:
            status_text = "❌ No API key set"
            status_color = "red"
        else:
            status_text = "❌ Missing requirements"
            status_color = "red"
        
        status_label = tk.Label(
            title_frame,
            text=status_text,
            font=('Arial', 12),
            fg=status_color
        )
        status_label.pack(side=tk.RIGHT)
        
        # File selection frame
        file_frame = tk.LabelFrame(
            self.parent_frame,
            text="File Selection",
            font=('Arial', 12, 'bold'),
            padx=15,
            pady=10
        )
        file_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # File list
        list_frame = tk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for file list
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_listbox = tk.Listbox(
            list_frame,
            height=8,
            selectmode=tk.EXTENDED,
            yscrollcommand=scrollbar.set
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
            current_model = self.main_gui.config.get('model', 'gemini-1.5-flash')
            
        tk.Label(api_frame, text=f"Translation Model: {current_model}", 
                font=('Arial', 10), fg='green').pack(side=tk.LEFT, padx=(20, 0))
        
        # Google Cloud credentials
        cred_frame = tk.Frame(settings_frame)
        cred_frame.pack(fill=tk.X, pady=(10, 10))
        
        tk.Label(cred_frame, text="Google Cloud Credentials (for OCR):").pack(side=tk.LEFT)
        
        self.cred_path_var = tk.StringVar(value=self.main_gui.config.get('google_cloud_credentials', ''))
        cred_entry = tk.Entry(cred_frame, textvariable=self.cred_path_var, width=50)
        cred_entry.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)
        
        tb.Button(
            cred_frame,
            text="Browse",
            command=self._browse_credentials,
            bootstyle="secondary"
        ).pack(side=tk.LEFT)
        
        # Language settings
        lang_frame = tk.Frame(settings_frame)
        lang_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(lang_frame, text="Source Language:").pack(side=tk.LEFT)
        self.source_lang_var = tk.StringVar(value=self.main_gui.config.get('manga_source_lang', 'auto'))
        source_combo = ttk.Combobox(
            lang_frame,
            textvariable=self.source_lang_var,
            values=['auto', 'ja', 'ko', 'zh', 'zh-TW'],
            width=15,
            state='readonly'
        )
        source_combo.pack(side=tk.LEFT, padx=(10, 20))
        
        tk.Label(lang_frame, text="Target Language:").pack(side=tk.LEFT)
        self.target_lang_var = tk.StringVar(value=self.main_gui.config.get('manga_target_lang', 'en'))
        target_combo = ttk.Combobox(
            lang_frame,
            textvariable=self.target_lang_var,
            values=['en', 'es', 'fr', 'de', 'pt', 'ru', 'ar', 'hi'],
            width=15,
            state='readonly'
        )
        target_combo.pack(side=tk.LEFT, padx=10)
        
        # Output settings
        output_frame = tk.Frame(settings_frame)
        output_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.create_subfolder_var = tk.BooleanVar(value=self.main_gui.config.get('manga_create_subfolder', True))
        tb.Checkbutton(
            output_frame,
            text="Create 'translated' subfolder for output",
            variable=self.create_subfolder_var,
            bootstyle="round-toggle"
        ).pack(side=tk.LEFT)
        
        # Control buttons
        control_frame = tk.Frame(self.parent_frame)
        control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Check if ready
        is_ready = has_api_key and has_vision
        
        self.start_button = tb.Button(
            control_frame,
            text="Start Translation",
            command=self._start_translation,
            bootstyle="success",
            state=tk.NORMAL if is_ready else tk.DISABLED
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
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
        
        # Current file progress
        self.file_progress_label = tk.Label(
            progress_frame,
            text="",
            font=('Arial', 10)
        )
        self.file_progress_label.pack(anchor=tk.W)
        
        # Statistics
        stats_frame = tk.Frame(progress_frame)
        stats_frame.pack(fill=tk.X)
        
        self.stats_label = tk.Label(
            stats_frame,
            text="Completed: 0 | Failed: 0 | Remaining: 0",
            font=('Arial', 10)
        )
        self.stats_label.pack(side=tk.LEFT)
        
        # Log area
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
        
        log_scrollbar = ttk.Scrollbar(log_scroll_frame)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(
            log_scroll_frame,
            height=12,
            wrap=tk.WORD,
            yscrollcommand=log_scrollbar.set
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.config(command=self.log_text.yview)
        
        # Configure text tags for colored output
        self.log_text.tag_config("info", foreground="white")
        self.log_text.tag_config("success", foreground="green")
        self.log_text.tag_config("error", foreground="red")
        self.log_text.tag_config("warning", foreground="orange")
    
    def _add_files(self):
        """Add image files to the processing list"""
        files = filedialog.askopenfilenames(
            title="Select Manga Images",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.webp *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        for file in files:
            if file not in self.selected_files:
                self.selected_files.append(file)
                self.file_listbox.insert(tk.END, os.path.basename(file))
    
    def _add_folder(self):
        """Add all images from a folder"""
        folder = filedialog.askdirectory(title="Select Manga Folder")
        if not folder:
            return
        
        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        
        for root, dirs, files in os.walk(folder):
            for file in sorted(files):
                if os.path.splitext(file)[1].lower() in image_extensions:
                    full_path = os.path.join(root, file)
                    if full_path not in self.selected_files:
                        self.selected_files.append(full_path)
                        rel_path = os.path.relpath(full_path, folder)
                        self.file_listbox.insert(tk.END, rel_path)
    
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
    
    def _browse_credentials(self):
        """Browse for Google Cloud credentials file"""
        file = filedialog.askopenfilename(
            title="Select Google Cloud Credentials JSON",
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        
        if file:
            self.cred_path_var.set(file)
            # Save to config
            self.main_gui.config['google_cloud_credentials'] = file
    
    def _start_translation(self):
        """Start the translation process"""
        if not self.selected_files:
            messagebox.showwarning("No Files", "Please select files to translate.")
            return
        
        if not self.main_gui.api_key_entry.get():
            messagebox.showwarning("No API Key", "Please set your API key in the main translator.")
            return
            
        if not self.cred_path_var.get():
            messagebox.showwarning("No Credentials", "Please select Google Cloud credentials file for OCR.")
            return
        
        # Initialize translator with both Google Cloud Vision and API Key
        try:
            # Create UnifiedClient instance with current settings
            from unified_api_client import UnifiedClient
            
            api_key = self.main_gui.api_key_entry.get()
            model = self.main_gui.model_var.get() if hasattr(self.main_gui, 'model_var') else 'gemini-1.5-flash'
            
            # Set environment variable for safety filters if needed
            if hasattr(self.main_gui, 'disable_gemini_safety_var') and self.main_gui.disable_gemini_safety_var.get():
                os.environ['DISABLE_GEMINI_SAFETY'] = 'true'
            else:
                os.environ['DISABLE_GEMINI_SAFETY'] = 'false'
            
            unified_client = UnifiedClient(model=model, api_key=api_key)
            
            self.translator = MangaTranslator(
                google_credentials_path=self.cred_path_var.get(),
                unified_client=unified_client,
                main_gui=self.main_gui,
                log_callback=self._log
            )
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize translator: {str(e)}")
            return
        
        # Update UI state
        self.is_running = True
        self.stop_flag.clear()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Reset counters
        self.total_files = len(self.selected_files)
        self.completed_files = 0
        self.failed_files = 0
        self.current_file_index = 0
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        
        # Save settings
        self.main_gui.config['manga_source_lang'] = self.source_lang_var.get()
        self.main_gui.config['manga_target_lang'] = self.target_lang_var.get()
        self.main_gui.config['manga_create_subfolder'] = self.create_subfolder_var.get()
        
        # Start translation thread
        self.translation_thread = threading.Thread(target=self._translation_worker, daemon=True)
        self.translation_thread.start()
    
    def _stop_translation(self):
        """Stop the translation process"""
        self.stop_flag.set()
        self.stop_button.config(state=tk.DISABLED)
        self._log("Stopping translation...", "warning")
    
    def _translation_worker(self):
        """Worker thread for translation"""
        try:
            # Process each file
            for i, file_path in enumerate(self.selected_files):
                if self.stop_flag.is_set():
                    break
                
                self.current_file_index = i
                
                # Update progress
                self._queue_update('progress', {
                    'file_index': i + 1,
                    'total': self.total_files,
                    'filename': os.path.basename(file_path)
                })
                
                # Determine output path
                output_path = self._get_output_path(file_path)
                
                # Log start
                self._log(f"Processing: {os.path.basename(file_path)}", "info")
                
                # Process image
                try:
                    result = self.translator.process_single_image(
                        file_path,
                        output_path
                    )
                    
                    if result['success']:
                        self.completed_files += 1
                        self._log(f"✅ Completed: {os.path.basename(file_path)}", "success")
                        self._log(f"   Detected {len(result['regions'])} text regions", "info")
                        self._log(f"   Output: {result['output_path']}", "info")
                    else:
                        self.failed_files += 1
                        errors = '\n'.join(result['errors'])
                        self._log(f"❌ Failed: {os.path.basename(file_path)}", "error")
                        self._log(f"   Errors: {errors}", "error")
                        
                except Exception as e:
                    self.failed_files += 1
                    self._log(f"❌ Error processing {os.path.basename(file_path)}: {str(e)}", "error")
                    self._log(traceback.format_exc(), "error")
                
                # Update statistics
                self._queue_update('stats', {
                    'completed': self.completed_files,
                    'failed': self.failed_files,
                    'remaining': self.total_files - self.completed_files - self.failed_files
                })
                
                # API delay is handled within the translator itself
            
            # Translation complete
            self._log(f"\n{'='*50}", "info")
            self._log(f"Translation complete!", "success")
            self._log(f"Total: {self.total_files} | Completed: {self.completed_files} | Failed: {self.failed_files}", "info")
            
        except Exception as e:
            self._log(f"Translation error: {str(e)}", "error")
            self._log(traceback.format_exc(), "error")
        
        finally:
            # Reset UI state
            self._queue_update('complete', {})
    
    def _get_output_path(self, input_path: str) -> str:
        """Generate output path based on settings"""
        dir_path = os.path.dirname(input_path)
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        
        if self.create_subfolder_var.get():
            output_dir = os.path.join(dir_path, 'translated')
            os.makedirs(output_dir, exist_ok=True)
            return os.path.join(output_dir, f"{name}_translated{ext}")
        else:
            return os.path.join(dir_path, f"{name}_translated{ext}")
    
    def _log(self, message: str, level: str = "info"):
        """Add message to log (thread-safe)"""
        self._queue_update('log', {'message': message, 'level': level})
    
    def _queue_update(self, update_type: str, data: Dict[str, Any]):
        """Queue a GUI update from worker thread"""
        self.update_queue.put((update_type, data))
    
    def _process_updates(self):
        """Process queued GUI updates"""
        try:
            while not self.update_queue.empty():
                update_type, data = self.update_queue.get_nowait()
                
                if update_type == 'progress':
                    # Update progress bar and label
                    progress = (data['file_index'] / data['total']) * 100
                    self.progress_bar['value'] = progress
                    self.progress_label.config(
                        text=f"Processing file {data['file_index']} of {data['total']}"
                    )
                    self.file_progress_label.config(
                        text=f"Current: {data['filename']}"
                    )
                
                elif update_type == 'stats':
                    # Update statistics
                    self.stats_label.config(
                        text=f"Completed: {data['completed']} | Failed: {data['failed']} | Remaining: {data['remaining']}"
                    )
                
                elif update_type == 'log':
                    # Add to log
                    self.log_text.insert(tk.END, data['message'] + '\n', data['level'])
                    self.log_text.see(tk.END)
                
                elif update_type == 'complete':
                    # Reset UI after completion
                    self.is_running = False
                    self.start_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                    self.progress_label.config(text="Translation complete")
                    self.file_progress_label.config(text="")
        
        except:
            pass
        
        # Schedule next update
        if self.dialog.winfo_exists():
            self.dialog.after(100, self._process_updates)