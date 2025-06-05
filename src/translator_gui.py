import io
import os
import sys
import json
import threading
import math
import ttkbootstrap as tb
import tkinter as tk
import tkinter as ttk
from tkinter import filedialog, messagebox, scrolledtext
from ttkbootstrap.constants import *
import logging
import shutil
from tkinter import scrolledtext
from PIL import Image, ImageTk
from tkinter import simpledialog
from tkinter import ttk

# CRITICAL: Import all modules at the top level for PyInstaller
# This ensures they're bundled into the executable
try:
    from TransateKRtoEN import main as translation_main, set_stop_flag as translation_stop_flag, is_stop_requested as translation_stop_check
except ImportError:
    translation_main = None
    translation_stop_flag = None
    translation_stop_check = None
    print("Warning: Could not import TransateKRtoEN module")

try:
    from extract_glossary_from_epub import main as glossary_main, set_stop_flag as glossary_stop_flag, is_stop_requested as glossary_stop_check
except ImportError:
    glossary_main = None
    glossary_stop_flag = None
    glossary_stop_check = None
    print("Warning: Could not import extract_glossary_from_epub module")

try:
    from epub_converter import fallback_compile_epub
except ImportError:
    fallback_compile_epub = None
    print("Warning: Could not import epub_converter module")

try:
    from scan_html_folder import scan_html_folder
except ImportError:
    scan_html_folder = None
    print("Warning: Could not import scan_html_folder module")

CONFIG_FILE = "config.json"
BASE_WIDTH, BASE_HEIGHT = 1550, 1000

class TranslatorGUI:
    def __init__(self, master):
        self.master = master
        self.max_output_tokens = 8192  # default fallback
        self.proc = None
        self.glossary_proc = None       
        master.title("Glossarion v1.6.5")
        master.geometry(f"{BASE_WIDTH}x{BASE_HEIGHT}")
        master.minsize(1550, 1000)
        master.bind('<F11>', self.toggle_fullscreen)
        master.bind('<Escape>', lambda e: master.attributes('-fullscreen', False))
        self.payloads_dir = os.path.join(os.getcwd(), "Payloads")        
        
        # Add stop flags for threading
        self.stop_requested = False
        self.translation_thread = None
        self.glossary_thread = None
        self.qa_thread = None
        
        # Warn on close
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Base directory for resources
        self.base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        ico_path = os.path.join(self.base_dir, 'Halgakos.ico')

        # Load and set window icon
        if os.path.isfile(ico_path):
            try:
                master.iconbitmap(ico_path)
            except Exception:
                pass

        # Load embedded icon image for display
        try:
            self.logo_img = ImageTk.PhotoImage(Image.open(ico_path)) if os.path.isfile(ico_path) else None
        except Exception as e:
            logging.error(f"Failed to load logo: {e}")
            self.logo_img = None
        if self.logo_img:
            master.iconphoto(False, self.logo_img)
            
        # Load config FIRST before setting up variables
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
                print(f"[DEBUG] Loaded config: {self.config.keys()}")  # Debug logging
        except Exception as e:
            print(f"[DEBUG] Config load failed: {e}")  # Debug logging
            self.config = {}
        
        # Set max_output_tokens from config
        self.max_output_tokens = self.config.get('max_output_tokens', self.max_output_tokens)
        
        # ─── restore rolling-summary state from config.json ───
        self.rolling_summary_var = tk.BooleanVar(
            value=self.config.get('use_rolling_summary', False)
        )
        self.summary_role_var = tk.StringVar(
            value=self.config.get('summary_role', 'user')
        )
        
        # ─── NEW: Add variables for new toggles ───
        self.disable_system_prompt_var = tk.BooleanVar(
            value=self.config.get('disable_system_prompt', False)
        )
        self.disable_auto_glossary_var = tk.BooleanVar(
            value=self.config.get('disable_auto_glossary', False)
        )
        self.disable_auto_glossary_var = tk.BooleanVar(
            value=self.config.get('disable_auto_glossary', False)
        )
        # Append Glossary:
        self.append_glossary_var = tk.BooleanVar(
            value=self.config.get('append_glossary', True)  # Default to True
        )   
        
        # Add after the other variable initializations
        self.reinforcement_freq_var = tk.StringVar(
            value=str(self.config.get('reinforcement_frequency', '10'))
        )
        
        # |Reset failed chapters
        self.reset_failed_chapters_var = tk.BooleanVar(
            value=self.config.get('reset_failed_chapters', True)  # Default to True
        )
        
        self.retry_truncated_var = tk.BooleanVar(
            value=self.config.get('retry_truncated', True)  # Default to True
        )
        self.max_retry_tokens_var = tk.StringVar(
            value=str(self.config.get('max_retry_tokens', 16384))  # Default max
        )        
        
        # Add after the other toggle variables
        self.retry_duplicate_var = tk.BooleanVar(
            value=self.config.get('retry_duplicate_bodies', True)  # Default to True
        )
        self.duplicate_lookback_var = tk.StringVar(
            value=str(self.config.get('duplicate_lookback_chapters', '5'))  # Check last 5 chapters
        )     
        
        # Default prompts
        self.default_prompts = {
            "korean": "You are a professional Korean to English novel translator, you must strictly output only English/HTML text while following these rules:\n- Use a context rich and natural translation style.\n- Retain honorifics, and suffixes like -nim, -ssi.\n- Preserve original intent, and speech tone.\n- retain onomatopoeia in Romaji.",
            "japanese": "You are a professional Japanese to English novel translator, you must strictly output only English/HTML text while following these rules:\n- Use a context rich and natural translation style.\n- Retain honorifics, and suffixes like -san, -sama, -chan, -kun.\n- Preserve original intent, and speech tone.\n- retain onomatopoeia in Romaji.",
            "chinese": "You are a professional Chinese to English novel translator, you must strictly output only English/HTML text while following these rules:\n- Use a context rich and natural translation style.\n- Preserve original intent, and speech tone.\n- retain onomatopoeia in Romaji."
        }

        # Profiles - FIXED: Load from config properly
        self.prompt_profiles = self.config.get('prompt_profiles', self.default_prompts.copy())
        active = self.config.get('active_profile', next(iter(self.prompt_profiles)))
        self.profile_var = tk.StringVar(value=active)
        self.lang_var = self.profile_var

        # Initialize GUI components
        self._setup_gui()
        
        # Check module availability and update UI accordingly
        self._check_modules()

    def _check_modules(self):
        """Check which modules are available and disable buttons if needed"""
        if translation_main is None:
            self.run_button.config(state='disabled')
            self.append_log("⚠️ Translation module not available")
        
        if glossary_main is None and hasattr(self, 'glossary_button'):
            self.glossary_button.config(state='disabled')
            self.append_log("⚠️ Glossary extraction module not available")
        
        if fallback_compile_epub is None:
            # Find and disable EPUB converter button
            for child in self.frame.winfo_children():
                if isinstance(child, tb.Frame):
                    for btn in child.winfo_children():
                        if isinstance(btn, tb.Button) and btn.cget('text') == 'EPUB Converter':
                            btn.config(state='disabled')
            self.append_log("⚠️ EPUB converter module not available")
        
        if scan_html_folder is None and hasattr(self, 'qa_button'):
            self.qa_button.config(state='disabled')
            self.append_log("⚠️ QA scanner module not available")

    def _setup_gui(self):
        """Initialize all GUI components"""
        # Main frame
        self.frame = tb.Frame(self.master, padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Grid config
        self.frame.grid_columnconfigure(0, weight=0)
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_columnconfigure(2, weight=0)
        self.frame.grid_columnconfigure(3, weight=1)
        self.frame.grid_columnconfigure(4, weight=0)
        for r in range(12):
            self.frame.grid_rowconfigure(r, weight=0)
        self.frame.grid_rowconfigure(9, weight=1, minsize=200)
        self.frame.grid_rowconfigure(10, weight=1, minsize=150)

        # EPUB File
        tb.Label(self.frame, text="EPUB File:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.entry_epub = tb.Entry(self.frame, width=50)
        self.entry_epub.grid(row=0, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        tb.Button(self.frame, text="Browse", command=self.browse_file, width=12).grid(row=0, column=4, sticky=tk.EW, padx=5, pady=5)

        # Model - FIXED: Load from config properly
        tb.Label(self.frame, text="Model:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        default_model = self.config.get('model', 'gemini-1.5-flash')  # Changed default
        print(f"[DEBUG] Setting model to: {default_model}")  # Debug logging
        self.model_var = tk.StringVar(value=default_model)
        tb.Combobox(self.frame, textvariable=self.model_var,
                    values=["gpt-4o","gpt-4o-mini","gpt-4-turbo","gpt-4.1-nano","gpt-4.1-mini","gpt-4.1","gpt-3.5-turbo","gemini-1.5-pro","gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-exp","deepseek-chat","claude-3-5-sonnet-20241022","claude-3-7-sonnet-20250219"], state="normal").grid(
            row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)

        # Language
        tb.Label(self.frame, text="Language:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.profile_menu = tb.Combobox(self.frame, textvariable=self.profile_var,
                                        values=list(self.prompt_profiles.keys()), state="normal")
        self.profile_menu.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        self.profile_menu.bind("<<ComboboxSelected>>", self.on_profile_select)
        self.profile_menu.bind("<Return>", self.on_profile_select)
        tb.Button(self.frame, text="Save Language", command=self.save_profile,
                  width=14).grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        tb.Button(self.frame, text="Delete Language", command=self.delete_profile,
                  width=14).grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)

        # Contextual
        self.contextual_var = tk.BooleanVar(value=self.config.get('contextual',True))
        tb.Checkbutton(self.frame, text="Contextual Translation",
                       variable=self.contextual_var).grid(row=3, column=0, columnspan=2,
                                                          sticky=tk.W, padx=5, pady=5)

        # API delay
        tb.Label(self.frame, text="API call delay (s):").grid(row=4, column=0,
                                                              sticky=tk.W, padx=5, pady=5)
        self.delay_entry = tb.Entry(self.frame, width=8)
        self.delay_entry.insert(0,str(self.config.get('delay',2)))
        self.delay_entry.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Chapter Range field
        tb.Label(self.frame, text="Chapter range (e.g., 5-10):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.chapter_range_entry = tb.Entry(self.frame, width=12)
        self.chapter_range_entry.insert(0, self.config.get('chapter_range', ''))
        self.chapter_range_entry.grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Token limit controls
        tb.Label(self.frame, text="Input Token limit:").grid(row=6, column=0,sticky=tk.W, padx=5, pady=5)
        self.token_limit_entry = tb.Entry(self.frame, width=8)
        self.token_limit_entry.insert(0, str(self.config.get('token_limit', 1000000)))
        self.token_limit_entry.grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.toggle_token_btn = tb.Button(
            self.frame,
            text="Disable Input Token Limit",
            command=self.toggle_token_limit,
            bootstyle="danger-outline",
            width=21
        )
        self.toggle_token_btn.grid(row=7, column=1, sticky=tk.W, padx=5, pady=5)

        # Translation settings
        tb.Label(self.frame, text="Temperature:").grid(row=4, column=2, sticky=tk.W, padx=5, pady=5)
        self.trans_temp = tb.Entry(self.frame, width=6)
        self.trans_temp.insert(0,str(self.config.get('translation_temperature',0.3)))
        self.trans_temp.grid(row=4, column=3, sticky=tk.W, padx=5, pady=5)
        tb.Label(self.frame, text="Transl. Hist. Limit:").grid(row=5, column=2, sticky=tk.W, padx=5, pady=5)
        self.trans_history = tb.Entry(self.frame, width=6)
        self.trans_history.insert(0,str(self.config.get('translation_history_limit',3)))
        self.trans_history.grid(row=5, column=3, sticky=tk.W, padx=5, pady=5)

        # Glossary settings
        tb.Label(self.frame, text="Glossary Temp:").grid(row=6, column=2, sticky=tk.W, padx=5, pady=5)
        self.glossary_temp = tb.Entry(self.frame, width=6)
        self.glossary_temp.insert(0,str(self.config.get('glossary_temperature',0.3)))
        self.glossary_temp.grid(row=6, column=3, sticky=tk.W, padx=5, pady=5)
        tb.Label(self.frame, text="Glossary Hist. Limit:").grid(row=7, column=2, sticky=tk.W, padx=5, pady=5)
        self.glossary_history = tb.Entry(self.frame, width=6)
        self.glossary_history.insert(0,str(self.config.get('glossary_history_limit',3)))
        self.glossary_history.grid(row=7, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Trim controls (hidden but present for compatibility)
        self.title_trim = tb.Entry(self.frame, width=6)
        self.title_trim.insert(0, str(self.config.get('title_trim_count', 1)))
        self.group_trim = tb.Entry(self.frame, width=6)
        self.group_trim.insert(0, str(self.config.get('group_affiliation_trim_count', 1)))
        self.traits_trim = tb.Entry(self.frame, width=6)
        self.traits_trim.insert(0, str(self.config.get('traits_trim_count', 1)))
        self.refer_trim = tb.Entry(self.frame, width=6)
        self.refer_trim.insert(0, str(self.config.get('refer_trim_count', 1)))
        self.loc_trim = tb.Entry(self.frame, width=6)
        self.loc_trim.insert(0, str(self.config.get('locations_trim_count', 1)))

        # Emergency restore
        self.emergency_restore_var = tk.BooleanVar(
        value=self.config.get('emergency_paragraph_restore', True)  # Default to enabled
)
        # API Key - FIXED: Load from config properly  
        tb.Label(self.frame, text="OpenAI / Gemini API Key:").grid(row=8, column=0, sticky=tk.W, padx=5, pady=5)
        self.api_key_entry = tb.Entry(self.frame, show='*')
        self.api_key_entry.grid(row=8, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        initial_key = self.config.get('api_key', '')
        if initial_key:
            self.api_key_entry.insert(0, initial_key)
            print(f"[DEBUG] Loaded API key: {initial_key[:10]}...")  # Debug logging
        tb.Button(self.frame, text="Show", command=self.toggle_api_visibility,width=12).grid(row=8, column=4, sticky=tk.EW, padx=5, pady=5)  
        
        # Other Settings button
        tb.Button(
            self.frame,
            text="⚙️  Other Setting",
            command=self.open_other_settings,
            bootstyle="info-outline",
            width=15
        ).grid(row=7, column=4, sticky=tk.EW, padx=5, pady=5)
        
        # Remove AI Artificats checkbox
        self.REMOVE_AI_ARTIFACTS_var = tk.BooleanVar(value=self.config.get('REMOVE_AI_ARTIFACTS', False))
        tb.Checkbutton(
            self.frame,
            text="Remove AI Artifacts",
            variable=self.REMOVE_AI_ARTIFACTS_var,
            bootstyle="round-toggle"
        ).grid(row=7, column=0, columnspan=5, sticky=tk.W, padx=5, pady=(0,5))
        
        # System Prompt
        tb.Label(self.frame, text="System Prompt:").grid(row=9, column=0, sticky=tk.NW, padx=5, pady=5)
        self.prompt_text = tk.Text(
            self.frame,
            height=5,
            width=60,
            wrap='word',
            undo=True,
            autoseparators=True,
            maxundo=-1
        )
        self.prompt_text.bind('<Control-z>', lambda e: self.prompt_text.edit_undo())
        self.prompt_text.bind('<Control-y>', lambda e: self.prompt_text.edit_redo())
        self.prompt_text.grid(row=9, column=1, columnspan=3, sticky=tk.NSEW, padx=5, pady=5)
        
        # Output Token Limit button
        self.output_btn = tb.Button(
            self.frame,
            text=f"Output Token Limit: {self.max_output_tokens}",
            command=self.prompt_custom_token_limit,
            bootstyle="info",
            width=22
        )
        self.output_btn.grid(row=9, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Run Translation button
        self.run_button = tb.Button(self.frame, text="Run Translation",
                                    command=self.run_translation_thread,
                                    bootstyle="success", width=14)
        self.run_button.grid(row=9, column=4, sticky=tk.N+tk.S+tk.EW, padx=5, pady=5)
        self.master.update_idletasks()
        self.run_base_w = self.run_button.winfo_width()
        self.run_base_h = self.run_button.winfo_height()
        self.master.bind('<Configure>', self.on_resize)

        # Log area
        self.log_text = scrolledtext.ScrolledText(self.frame, wrap=tk.WORD,
                                                  state=tk.DISABLED)
        self.log_text.grid(row=10, column=0, columnspan=5, sticky=tk.NSEW, padx=5, pady=5)

        # Bottom toolbar
        self._make_bottom_toolbar()

        self.token_limit_disabled = False

        # Initial prompt
        self.on_profile_select()

        print("[DEBUG] GUI setup completed with config values loaded")  # Debug logging
    def force_retranslation(self):
        """Force retranslation of specific chapters"""
        epub_path = self.entry_epub.get()
        if not epub_path or not os.path.isfile(epub_path):
            messagebox.showerror("Error", "Please select a valid EPUB file first.")
            return
        
        # Get the output directory
        epub_base = os.path.splitext(os.path.basename(epub_path))[0]
        output_dir = epub_base
        
        if not os.path.exists(output_dir):
            messagebox.showinfo("Info", "No translation output found for this EPUB.")
            return
        
        # Load progress file
        progress_file = os.path.join(output_dir, "translation_progress.json")
        if not os.path.exists(progress_file):
            messagebox.showinfo("Info", "No progress tracking found.")
            return
        
        with open(progress_file, 'r', encoding='utf-8') as f:
            prog = json.load(f)
        
        # Create dialog to select chapters
        dialog = tk.Toplevel(self.master)
        dialog.title("Force Retranslation")
        dialog.geometry("660x600")
        
        # Instructions
        tk.Label(dialog, text="Select chapters to retranslate:", font=('Arial', 12)).pack(pady=10)
        
        # Create frame with scrollbar
        frame = tk.Frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Listbox for chapters
        listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Populate with chapters
        chapter_keys = []
        for chapter_key, chapter_info in sorted(prog.get("chapters", {}).items(), 
                                               key=lambda x: int(x[0])):
            chapter_num = chapter_info.get("chapter_num", "?")
            status = chapter_info.get("status", "unknown")
            output_file = chapter_info.get("output_file", "")
            
            # Check if file exists
            file_exists = "✓" if output_file and os.path.exists(os.path.join(output_dir, output_file)) else "✗"
            
            display_text = f"Chapter {chapter_num} - {status} - File: {file_exists}"
            listbox.insert(tk.END, display_text)
            chapter_keys.append(chapter_key)
        
        def retranslate_selected():
            selected_indices = listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("Warning", "No chapters selected.")
                return
            
            count = 0
            for idx in selected_indices:
                chapter_key = chapter_keys[idx]
                
                # Mark chapter for retranslation
                if chapter_key in prog["chapters"]:
                    chapter_info = prog["chapters"][chapter_key]
                    
                    # Clear the chapter data to force retranslation
                    del prog["chapters"][chapter_key]
                    
                    # Remove from content hashes
                    content_hash = chapter_info.get("content_hash")
                    if content_hash and content_hash in prog.get("content_hashes", {}):
                        if prog["content_hashes"][content_hash].get("chapter_idx") == int(chapter_key):
                            del prog["content_hashes"][content_hash]
                    
                    # Remove chunk data
                    if chapter_key in prog.get("chapter_chunks", {}):
                        del prog["chapter_chunks"][chapter_key]
                    
                    # Delete the output file if it exists
                    output_file = chapter_info.get("output_file")
                    if output_file:
                        output_path = os.path.join(output_dir, output_file)
                        if os.path.exists(output_path):
                            os.remove(output_path)
                            self.append_log(f"🗑️ Deleted: {output_file}")
                    
                    count += 1
            
            # Save updated progress
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(prog, f, ensure_ascii=False, indent=2)
            
            self.append_log(f"🔄 Marked {count} chapters for retranslation")
            messagebox.showinfo("Success", f"Marked {count} chapters for retranslation.\nRun translation to process them.")
            dialog.destroy()
        
        # Buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Select All", 
                  command=lambda: listbox.select_set(0, tk.END)).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Clear Selection", 
                  command=lambda: listbox.select_clear(0, tk.END)).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Retranslate Selected", 
                  command=retranslate_selected, bg="#ff6b6b", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", 
                  command=dialog.destroy).pack(side=tk.LEFT, padx=5)
                  
    def _make_bottom_toolbar(self):
        """Create the bottom toolbar with all action buttons"""
        btn_frame = tb.Frame(self.frame)
        btn_frame.grid(row=11, column=0, columnspan=5, sticky=tk.EW, pady=5)
        
        # QA Scan button
        self.qa_button = tb.Button(btn_frame, text="QA Scan", command=self.run_qa_scan, bootstyle="warning")
        self.qa_button.grid(row=0, column=99, sticky=tk.EW, padx=5)

        toolbar_items = [
            ("EPUB Converter",      self.epub_converter,               "info"),
            ("Extract Glossary",    self.run_glossary_extraction_thread, "warning"),
            ("Trim Glossary",       self.trim_glossary,               "secondary"),
            ("Retranslate",   self.force_retranslation,         "warning"),  # NEW
            ("Save Config",         self.save_config,                 "secondary"),
            ("Load Glossary",       self.load_glossary,               "secondary"),
            ("Import Profiles",     self.import_profiles,             "secondary"),
            ("Export Profiles",     self.export_profiles,             "secondary"),
        ]
        for idx, (lbl, cmd, style) in enumerate(toolbar_items):
            btn_frame.columnconfigure(idx, weight=1)
            btn = tb.Button(btn_frame, text=lbl, command=cmd, bootstyle=style)
            btn.grid(row=0, column=idx, sticky=tk.EW, padx=2)
            if lbl == "Extract Glossary":
                self.glossary_button = btn

        self.frame.grid_rowconfigure(12, weight=0)

    # === DIRECT FUNCTION CALLS ===
    
    def run_translation_thread(self):
        """Start translation in a separate thread"""
        if translation_main is None:
            self.append_log("❌ Translation module is not available")
            messagebox.showerror("Module Error", "Translation module is not available. Please ensure all files are present.")
            return
            
        # Check if glossary extraction is running
        if hasattr(self, 'glossary_thread') and self.glossary_thread and self.glossary_thread.is_alive():
            self.append_log("⚠️ Cannot run translation while glossary extraction is in progress.")
            messagebox.showwarning("Process Running", "Please wait for glossary extraction to complete before starting translation.")
            return
            
        if self.translation_thread and self.translation_thread.is_alive():
            # Stop existing translation
            self.stop_translation()
            return
            
        self.stop_requested = False
        if translation_stop_flag:
            translation_stop_flag(False)
        self.translation_thread = threading.Thread(target=self.run_translation_direct, daemon=True)
        self.translation_thread.start()
        # Update button immediately after starting thread
        self.master.after(100, self.update_run_button)

    def run_translation_direct(self):
        """Run translation directly without subprocess"""
        try:
            # Validate inputs
            epub_path = self.entry_epub.get()
            if not epub_path or not os.path.isfile(epub_path):
                self.append_log("❌ Error: Please select a valid EPUB file.")
                return

            api_key = self.api_key_entry.get()
            if not api_key:
                self.append_log("❌ Error: Please enter your API key.")
                return

            # Set up environment variables that the translation script expects
            old_argv = sys.argv
            old_env = dict(os.environ)
            
            try:
                # Debug logging
                self.append_log(f"🔧 Setting up environment variables...")
                self.append_log(f"📖 EPUB: {os.path.basename(epub_path)}")
                self.append_log(f"🤖 Model: {self.model_var.get()}")
                self.append_log(f"🔑 API Key: {api_key[:10]}...")
                self.append_log(f"📤 Output Token Limit: {self.max_output_tokens}")
                
                # ─── NEW: Log the state of new toggles ───
                if self.disable_system_prompt_var.get():
                    self.append_log("⚠️ Hardcoded prompts disabled")
                if self.disable_auto_glossary_var.get():
                    self.append_log("⚠️ Automatic glossary disabled")
                    
                # Log glossary status
                if self.append_glossary_var.get():
                    self.append_log("✅ Glossary will be appended to prompts")
                else:
                    self.append_log("⚠️ Glossary appending is disabled")
                
                # Set environment variables - FIXED: Use multiple API key variables
                os.environ.update({
                    'EPUB_PATH': epub_path,
                    'MODEL': self.model_var.get(),
                    'CONTEXTUAL': '1' if self.contextual_var.get() else '0',
                    'SEND_INTERVAL_SECONDS': str(self.delay_entry.get()),
                    'MAX_OUTPUT_TOKENS': str(self.max_output_tokens),
                    'API_KEY': api_key,                    # Primary
                    'OPENAI_API_KEY': api_key,             # OpenAI
                    'OPENAI_OR_Gemini_API_KEY': api_key,   # Fallback name
                    'GEMINI_API_KEY': api_key,             # Gemini
                    'SYSTEM_PROMPT': self.prompt_text.get("1.0", "end").strip(),
                    'REMOVE_AI_ARTIFACTS': "1" if self.REMOVE_AI_ARTIFACTS_var.get() else "0",
                    'USE_ROLLING_SUMMARY': "1" if self.config.get('use_rolling_summary') else "0",
                    'SUMMARY_ROLE': self.config.get('summary_role', 'user'),
                    'TRANSLATION_LANG': self.lang_var.get().lower(),
                    'TRANSLATION_TEMPERATURE': str(self.trans_temp.get()),
                    'TRANSLATION_HISTORY_LIMIT': str(self.trans_history.get()),
                    'EPUB_OUTPUT_DIR': os.getcwd(),
                    # ─── NEW: Add environment variables for new toggles ───
                    'DISABLE_SYSTEM_PROMPT': "1" if self.disable_system_prompt_var.get() else "0",
                    'DISABLE_AUTO_GLOSSARY': "1" if self.disable_auto_glossary_var.get() else "0",
                    'APPEND_GLOSSARY': "1" if self.append_glossary_var.get() else "0",
                    'EMERGENCY_PARAGRAPH_RESTORE': "1" if self.emergency_restore_var.get() else "0",
                    'REINFORCEMENT_FREQUENCY': self.reinforcement_freq_var.get(),
                    'RESET_FAILED_CHAPTERS': "1" if self.reset_failed_chapters_var.get() else "0",
                    'RETRY_TRUNCATED': "1" if self.retry_truncated_var.get() else "0",
                    'MAX_RETRY_TOKENS': self.max_retry_tokens_var.get()
                })
                
                # Set chapter range if specified
                chap_range = self.chapter_range_entry.get().strip()
                if chap_range:
                    os.environ['CHAPTER_RANGE'] = chap_range
                    self.append_log(f"📊 Chapter Range: {chap_range}")
                
                # Debug what state we're in
                self.append_log(f"[DEBUG] token_limit_disabled = {self.token_limit_disabled}")
                self.append_log(f"[DEBUG] token_limit_entry value = '{self.token_limit_entry.get()}'")
                
                # Set token limit based on UI state
                if self.token_limit_disabled:
                    # Token limit is disabled - set empty string
                    os.environ['MAX_INPUT_TOKENS'] = ''
                    self.append_log("🎯 Input Token Limit: Unlimited (disabled)")
                else:
                    # Token limit is enabled - get value from entry
                    token_val = self.token_limit_entry.get().strip()
                    if token_val and token_val.isdigit():
                        os.environ['MAX_INPUT_TOKENS'] = token_val
                        self.append_log(f"🎯 Input Token Limit: {token_val}")
                    else:
                        # Invalid or empty input, use default
                        default_limit = '1000000'
                        os.environ['MAX_INPUT_TOKENS'] = default_limit
                        self.append_log(f"🎯 Input Token Limit: {default_limit} (default)")
                
                # Debug log to verify
                self.append_log(f"[DEBUG] MAX_INPUT_TOKENS env var = '{os.environ.get('MAX_INPUT_TOKENS', 'NOT SET')}'")

                    
                # Set manual glossary if loaded
                if hasattr(self, 'manual_glossary_path'):
                    os.environ['MANUAL_GLOSSARY'] = self.manual_glossary_path
                    self.append_log(f"📑 Manual Glossary: {os.path.basename(self.manual_glossary_path)}")

                # Set sys.argv for the translation script
                sys.argv = ['TransateKRtoEN.py', epub_path]
                
                self.append_log("🚀 Starting translation...")
                
                # Print environment check right before calling
                self.append_log(f"[DEBUG] Right before translation_main: MAX_INPUT_TOKENS = '{os.environ.get('MAX_INPUT_TOKENS', 'NOT SET')}'")
                
                # Ensure Payloads directory exists in current working directory
                os.makedirs("Payloads", exist_ok=True)
                self.append_log(f"[DEBUG] Created Payloads directory in: {os.getcwd()}")

                # Call the translation main function directly with callbacks
                translation_main(
                    log_callback=self.append_log,
                    stop_callback=lambda: self.stop_requested
                )
                
                if not self.stop_requested:
                    self.append_log("✅ Translation completed successfully!")
                
            except Exception as e:
                self.append_log(f"❌ Translation error: {e}")
                import traceback
                self.append_log(f"❌ Full error: {traceback.format_exc()}")
                
            finally:
                # Restore environment and argv
                sys.argv = old_argv
                os.environ.clear()
                os.environ.update(old_env)
                
        except Exception as e:
            self.append_log(f"❌ Translation setup error: {e}")
            
        finally:
            self.stop_requested = False
            if translation_stop_flag:
                translation_stop_flag(False)
            # Clear the thread reference to fix double-click issue
            self.translation_thread = None
            self.master.after(0, self.update_run_button)

    def run_glossary_extraction_thread(self):
        """Start glossary extraction in a separate thread"""
        if glossary_main is None:
            self.append_log("❌ Glossary extraction module is not available")
            messagebox.showerror("Module Error", "Glossary extraction module is not available.")
            return
            
        # Check if translation is running
        if hasattr(self, 'translation_thread') and self.translation_thread and self.translation_thread.is_alive():
            self.append_log("⚠️ Cannot run glossary extraction while translation is in progress.")
            messagebox.showwarning("Process Running", "Please wait for translation to complete before extracting glossary.")
            return
            
        if self.glossary_thread and self.glossary_thread.is_alive():
            # Stop existing glossary extraction
            self.stop_glossary_extraction()
            return
            
        self.stop_requested = False
        if glossary_stop_flag:
            glossary_stop_flag(False)
        self.glossary_thread = threading.Thread(target=self.run_glossary_extraction_direct, daemon=True)
        self.glossary_thread.start()
        # Update buttons immediately after starting thread
        self.master.after(100, self.update_run_button)

    def run_glossary_extraction_direct(self):
        """Run glossary extraction directly without subprocess"""
        try:
            epub_path = self.entry_epub.get()
            if not epub_path or not os.path.isfile(epub_path):
                self.append_log("❌ Error: Please select a valid EPUB file for glossary extraction.")
                return

            # Check for API key
            api_key = self.api_key_entry.get()
            if not api_key:
                self.append_log("❌ Error: Please enter your API key.")
                return

            # Save current sys.argv and environment
            old_argv = sys.argv
            old_env = dict(os.environ)
            
            try:
                # Set up environment for glossary extraction
                env_updates = {
                    'GLOSSARY_TEMPERATURE': str(self.glossary_temp.get()),
                    'GLOSSARY_CONTEXT_LIMIT': str(self.glossary_history.get()),
                    'MODEL': self.model_var.get(),
                    'OPENAI_API_KEY': self.api_key_entry.get(),
                    'OPENAI_OR_Gemini_API_KEY': self.api_key_entry.get(),
                    'API_KEY': self.api_key_entry.get()
                }
                
                # Use the same token limit logic as translation
                # The complete section should look like:
                if self.token_limit_disabled:
                    os.environ['MAX_INPUT_TOKENS'] = ''  # NOT GLOSSARY_TOKEN_LIMIT
                    self.append_log("🎯 Input Token Limit: Unlimited (disabled)")
                else:
                    token_val = self.token_limit_entry.get().strip()
                    if token_val and token_val.isdigit():
                        os.environ['MAX_INPUT_TOKENS'] = token_val  # NOT GLOSSARY_TOKEN_LIMIT
                        self.append_log(f"🎯 Input Token Limit: {token_val}")
                    else:
                        os.environ['MAX_INPUT_TOKENS'] = '1000000'  # NOT GLOSSARY_TOKEN_LIMIT
                        self.append_log(f"🎯 Input Token Limit: 1000000 (default)")
                
                self.append_log(f"[DEBUG] After setting env, MAX_INPUT_TOKENS = {os.environ.get('MAX_INPUT_TOKENS', 'NOT SET')}")
                
                # Set up argv for glossary extraction
                epub_base = os.path.splitext(os.path.basename(epub_path))[0]
                output_path = f"{epub_base}_glossary.json"
                
                sys.argv = [
                    'extract_glossary_from_epub.py',
                    '--epub', epub_path,
                    '--output', output_path,
                    '--config', CONFIG_FILE  # Use the main config.json
                ]
                
                self.append_log("🚀 Starting glossary extraction...")
                self.append_log(f"📤 Output Token Limit: {self.max_output_tokens}") 
                os.environ['MAX_OUTPUT_TOKENS'] = str(self.max_output_tokens)

                
                # Call glossary extraction directly with callbacks
                glossary_main(
                    log_callback=self.append_log,
                    stop_callback=lambda: self.stop_requested
                )
                
                if not self.stop_requested:
                    self.append_log("✅ Glossary extraction completed successfully!")
                    
            finally:
                # Restore environment and argv
                sys.argv = old_argv
                os.environ.clear()
                os.environ.update(old_env)
                
        except Exception as e:
            self.append_log(f"❌ Glossary extraction error: {e}")
            
        finally:
            self.stop_requested = False
            if glossary_stop_flag:
                glossary_stop_flag(False)
            # Clear the thread reference to fix double-click issue
            self.glossary_thread = None
            self.master.after(0, self.update_run_button)
                    
    def toggle_token_limit(self):
        """Toggle whether the token-limit entry is active or not."""
        if not self.token_limit_disabled:
            # disable it
            self.token_limit_entry.config(state=tk.DISABLED)
            self.toggle_token_btn.config(text="Enable Input Token Limit", bootstyle="success-outline")
            self.append_log("⚠️ Input token limit disabled - both translation and glossary extraction will process chapters of any size.")
            self.token_limit_disabled = True
        else:
            # re-enable it
            self.token_limit_entry.config(state=tk.NORMAL)
            if not self.token_limit_entry.get().strip():
                self.token_limit_entry.insert(0, str(self.config.get('token_limit', 1000000)))
            self.toggle_token_btn.config(text="Disable Input Token Limit", bootstyle="danger-outline")
            self.append_log(f"✅ Input token limit enabled: {self.token_limit_entry.get()} tokens (applies to both translation and glossary extraction)")
            self.token_limit_disabled = False
            
    def update_run_button(self):
        """Switch Run↔Stop depending on whether a process is active."""
        translation_running = (
            hasattr(self, 'translation_thread') and 
            self.translation_thread and 
            self.translation_thread.is_alive()
        )
        glossary_running = (
            hasattr(self, 'glossary_thread') and 
            self.glossary_thread and 
            self.glossary_thread.is_alive()
        )
        qa_running = (
            hasattr(self, 'qa_thread') and 
            self.qa_thread and 
            self.qa_thread.is_alive()
        )

        # Update translation button
        if translation_running:
            self.run_button.config(
                text="Stop Translation",
                command=self.stop_translation,
                bootstyle="danger",
                state=tk.NORMAL
            )
        else:
            self.run_button.config(
                text="Run Translation",
                command=self.run_translation_thread,
                bootstyle="success",
                state=tk.NORMAL if translation_main and not (glossary_running or qa_running) else tk.DISABLED
            )
            
        # Update glossary button if it exists
        if hasattr(self, 'glossary_button'):
            if glossary_running:
                self.glossary_button.config(
                    text="Stop Glossary",
                    command=self.stop_glossary_extraction,
                    bootstyle="danger",
                    state=tk.NORMAL
                )
            else:
                self.glossary_button.config(
                    text="Extract Glossary",
                    command=self.run_glossary_extraction_thread,
                    bootstyle="warning",
                    state=tk.NORMAL if glossary_main and not (translation_running or qa_running) else tk.DISABLED
                )
                
        # Disable other buttons when any process is running
        if hasattr(self, 'qa_button'):
            self.qa_button.config(state=tk.NORMAL if not (translation_running or glossary_running) else tk.DISABLED)

    def stop_translation(self):
        """Stop translation only"""
        self.stop_requested = True
        if translation_stop_flag:
            translation_stop_flag(True)
            
        # Also try to set the module-level stop flag directly
        try:
            import TransateKRtoEN
            if hasattr(TransateKRtoEN, 'set_stop_flag'):
                TransateKRtoEN.set_stop_flag(True)
        except:
            pass
            
        self.append_log("❌ Translation stop requested.")
        self.append_log("⏳ Please wait... stopping after current operation completes.")
        self.update_run_button()

    def stop_glossary_extraction(self):
        """Stop glossary extraction specifically"""
        self.stop_requested = True
        if glossary_stop_flag:
            glossary_stop_flag(True)
            
        # Also try to set the module-level stop flag directly
        try:
            import extract_glossary_from_epub
            if hasattr(extract_glossary_from_epub, 'set_stop_flag'):
                extract_glossary_from_epub.set_stop_flag(True)
        except:
            pass
            
        self.append_log("❌ Glossary extraction stop requested.")
        self.append_log("⏳ Please wait... stopping after current API call completes.")
        self.update_run_button()

    def epub_converter(self):
        """Run EPUB converter directly without subprocess"""
        if fallback_compile_epub is None:
            self.append_log("❌ EPUB converter module is not available")
            messagebox.showerror("Module Error", "EPUB converter module is not available.")
            return

        folder = filedialog.askdirectory(title="Select translation output folder")
        if not folder:
            return

        try:
            self.append_log("📦 Running EPUB Converter...")
            
            # Call the EPUB converter function directly with callback
            fallback_compile_epub(folder, log_callback=self.append_log)
            
            # Update to use the new filename format
            base_name = os.path.basename(folder)
            out_file = os.path.join(folder, f"{base_name}.epub")
            
            # Check if the file was actually created
            if os.path.exists(out_file):
                messagebox.showinfo("EPUB Compilation Success", f"Created: {out_file}")
            else:
                self.append_log("⚠️ EPUB file was not created. Check the logs for details.")
            
        except Exception as e:
            error_str = str(e)
            self.append_log(f"❌ EPUB Converter error: {error_str}")
            
            # Don't show popup for "Document is empty" errors
            if "Document is empty" not in error_str:
                messagebox.showerror("EPUB Converter Failed", f"Error: {error_str}")
            else:
                # Just log it, no popup
                self.append_log("📋 Check the log above for details about what went wrong.")

    def run_qa_scan(self):
        """Run QA scan directly without subprocess"""
        if scan_html_folder is None:
            self.append_log("❌ QA scanner module is not available")
            messagebox.showerror("Module Error", "QA scanner module is not available.")
            return

        if hasattr(self, 'qa_thread') and self.qa_thread and self.qa_thread.is_alive():
            self.stop_requested = True
            self.append_log("⛔ QA scan stop requested.")
            return
            
        folder_path = filedialog.askdirectory(title="Select Folder with HTML Files")
        if not folder_path:
            self.append_log("⚠️ QA scan canceled.")
            return

        self.append_log(f"🔍 Starting QA scan for folder: {folder_path}")
        self.stop_requested = False

        def run_scan():
            # Update buttons when scan starts
            self.master.after(0, self.update_run_button)
            self.qa_button.config(text="Stop Scan", command=self.stop_qa_scan, bootstyle="danger")
            
            try:
                scan_html_folder(folder_path, log=self.append_log, stop_flag=lambda: self.stop_requested)
                self.append_log("✅ QA scan completed successfully.")
            except Exception as e:
                self.append_log(f"❌ QA scan error: {e}")
            finally:
                # Clear thread reference and update buttons when done
                self.qa_thread = None
                self.master.after(0, self.update_run_button)
                self.master.after(0, lambda: self.qa_button.config(
                    text="QA Scan", 
                    command=self.run_qa_scan, 
                    bootstyle="warning",
                    state=tk.NORMAL if scan_html_folder else tk.DISABLED
                ))

        self.qa_thread = threading.Thread(target=run_scan, daemon=True)
        self.qa_thread.start()

    def stop_qa_scan(self):
        """Stop QA scan"""
        self.stop_requested = True
        self.append_log("⛔ QA scan stop requested.")

    def on_close(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to exit?"):
            self.stop_requested = True
            self.master.destroy()
            sys.exit(0)

    def append_log(self, message):
        def _append():
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)
        
        if threading.current_thread() is threading.main_thread():
            _append()
        else:
            self.master.after(0, _append)

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("EPUB files","*.epub")])
        if path:
            self.entry_epub.delete(0, tk.END)
            self.entry_epub.insert(0, path)

    def toggle_fullscreen(self, event=None):
        is_full = self.master.attributes('-fullscreen')
        self.master.attributes('-fullscreen', not is_full)

    def toggle_api_visibility(self):
        show = self.api_key_entry.cget('show')
        self.api_key_entry.config(show='' if show == '*' else '*')

    def prompt_custom_token_limit(self):
        val = simpledialog.askinteger(
            "Set Max Output Token Limit",
            "Enter max output tokens for API output (e.g., 2048, 4196, 8192):",
            minvalue=1,
            maxvalue=200000
        )
        if val:
            self.max_output_tokens = val
            self.output_btn.config(text=f"Output Token Limit: {val}")
            self.append_log(f"✅ Output token limit set to {val}")

    def open_other_settings(self):
        top = tk.Toplevel(self.master)
        top.title("Other Settings")
        top.geometry("730x1050")  # Fixed width, reasonable height
        
        # Create a canvas and scrollbar for scrolling
        canvas = tk.Canvas(top)
        scrollbar = ttk.Scrollbar(top, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # --- CONTENT STARTS HERE ---
        
        # Section 1: Rolling Summary
        section1_frame = tk.LabelFrame(scrollable_frame, text="Context Management", padx=10, pady=10)
        section1_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        tb.Checkbutton(section1_frame, text="Use Rolling Summary", 
                       variable=self.rolling_summary_var,
                       bootstyle="round-toggle").pack(anchor=tk.W)
        
        summary_frame = tk.Frame(section1_frame)
        summary_frame.pack(anchor=tk.W, padx=20, pady=(5, 0))
        tk.Label(summary_frame, text="Summary Role:").pack(side=tk.LEFT)
        ttk.Combobox(summary_frame, textvariable=self.summary_role_var,
                     values=["user", "system"], state="readonly", width=10).pack(side=tk.LEFT, padx=5)
        
        # Section 2: Response Handling, after the truncated response checkbox:
        section2_frame = tk.LabelFrame(scrollable_frame, text="Response Handling", padx=10, pady=10)
        section2_frame.pack(fill="x", padx=10, pady=5)
        
        tb.Checkbutton(section2_frame, text="Auto-retry Truncated Responses", 
                       variable=self.retry_truncated_var,
                       bootstyle="round-toggle").pack(anchor=tk.W)
        
        retry_frame = tk.Frame(section2_frame)
        retry_frame.pack(anchor=tk.W, padx=20, pady=(5, 0))
        tk.Label(retry_frame, text="Max retry tokens:").pack(side=tk.LEFT)
        tb.Entry(retry_frame, width=8, textvariable=self.max_retry_tokens_var).pack(side=tk.LEFT, padx=5)
        
        tk.Label(section2_frame, 
                 text="Automatically retry when API response is cut off",
                 font=('TkDefaultFont', 9), fg='gray').pack(anchor=tk.W, padx=20)
        
        
        tb.Checkbutton(section2_frame, text="Auto-retry Duplicate Body Content", 
                       variable=self.retry_duplicate_var,
                       bootstyle="round-toggle").pack(anchor=tk.W, pady=(10, 0))

        duplicate_frame = tk.Frame(section2_frame)
        duplicate_frame.pack(anchor=tk.W, padx=20, pady=(5, 0))
        tk.Label(duplicate_frame, text="Check last").pack(side=tk.LEFT)
        tb.Entry(duplicate_frame, width=4, textvariable=self.duplicate_lookback_var).pack(side=tk.LEFT, padx=5)
        tk.Label(duplicate_frame, text="chapters for duplicates").pack(side=tk.LEFT)

        tk.Label(section2_frame, 
                 text="Detects when AI returns same content for different chapters",
                 font=('TkDefaultFont', 9), fg='gray').pack(anchor=tk.W, padx=20)
                 
        # Section 3: Prompt Management
        section3_frame = tk.LabelFrame(scrollable_frame, text="Prompt Management", padx=10, pady=10)
        section3_frame.pack(fill="x", padx=10, pady=5)
        
        # Reinforcement frequency
        reinforce_frame = tk.Frame(section3_frame)
        reinforce_frame.pack(anchor=tk.W, pady=(0, 10))
        tk.Label(reinforce_frame, text="Reinforce every").pack(side=tk.LEFT)
        tb.Entry(reinforce_frame, width=6, textvariable=self.reinforcement_freq_var).pack(side=tk.LEFT, padx=5)
        tk.Label(reinforce_frame, text="messages (0 = disabled)").pack(side=tk.LEFT)
        
        tk.Label(section3_frame, 
                 text="Periodically reminds the AI of your system prompt",
                 font=('TkDefaultFont', 9), fg='gray').pack(anchor=tk.W, pady=(0, 10))
        
        tb.Checkbutton(section3_frame, text="Disable Hardcoded System Prompts", 
                       variable=self.disable_system_prompt_var,
                       bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
        
        # Section 4: Glossary Settings
        section4_frame = tk.LabelFrame(scrollable_frame, text="Glossary Settings", padx=10, pady=10)
        section4_frame.pack(fill="x", padx=10, pady=5)
        
        tb.Checkbutton(section4_frame, text="Disable Automatic Glossary Generation", 
                       variable=self.disable_auto_glossary_var,
                       bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
        
        tb.Checkbutton(section4_frame, text="Append Glossary to System Prompt", 
                       variable=self.append_glossary_var,
                       bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
        
        # Section 5: Processing Options
        section5_frame = tk.LabelFrame(scrollable_frame, text="Processing Options", padx=10, pady=10)
        section5_frame.pack(fill="x", padx=10, pady=5)
        
        tb.Checkbutton(section5_frame, text="Emergency Paragraph Restoration", 
                       variable=self.emergency_restore_var,
                       bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
        
        tb.Checkbutton(section5_frame, text="Reset Failed Chapters on Start", 
                       variable=self.reset_failed_chapters_var,
                       bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
        
        tk.Label(section5_frame, 
                 text="Automatically retry failed chapters on each run",
                 font=('TkDefaultFont', 9), fg='gray').pack(anchor=tk.W, padx=20)
        
        # Save and close function
        def save_and_close():
            self.config['use_rolling_summary'] = self.rolling_summary_var.get()
            self.config['summary_role'] = self.summary_role_var.get()
            self.config['disable_system_prompt'] = self.disable_system_prompt_var.get()
            self.config['disable_auto_glossary'] = self.disable_auto_glossary_var.get()
            self.config['append_glossary'] = self.append_glossary_var.get()
            self.config['emergency_paragraph_restore'] = self.emergency_restore_var.get()
            self.config['reinforcement_frequency'] = int(self.reinforcement_freq_var.get())
            self.config['reset_failed_chapters'] = self.reset_failed_chapters_var.get()
            self.config['retry_truncated'] = self.retry_truncated_var.get()
            self.config['max_retry_tokens'] = int(self.max_retry_tokens_var.get())
            self.config['retry_duplicate_bodies'] = self.retry_duplicate_var.get()
            self.config['duplicate_lookback_chapters'] = int(self.duplicate_lookback_var.get())
            
            # Set environment variables
            os.environ["USE_ROLLING_SUMMARY"] = "1" if self.rolling_summary_var.get() else "0"
            os.environ["SUMMARY_ROLE"] = self.summary_role_var.get()
            os.environ["APPEND_GLOSSARY"] = "1" if self.append_glossary_var.get() else "0"
            os.environ["EMERGENCY_PARAGRAPH_RESTORE"] = "1" if self.emergency_restore_var.get() else "0"
            os.environ["REINFORCEMENT_FREQUENCY"] = self.reinforcement_freq_var.get()
            os.environ["RESET_FAILED_CHAPTERS"] = "1" if self.reset_failed_chapters_var.get() else "0"
            os.environ["RETRY_TRUNCATED"] = "1" if self.retry_truncated_var.get() else "0"
            os.environ["MAX_RETRY_TOKENS"] = self.max_retry_tokens_var.get()
            os.environ["RETRY_DUPLICATE_BODIES"] = "1" if self.retry_duplicate_var.get() else "0"
            os.environ["DUPLICATE_LOOKBACK_CHAPTERS"] = self.duplicate_lookback_var.get()
            
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            top.destroy()
        
        # Save button at the bottom
        tb.Button(scrollable_frame, text="Save", command=save_and_close, 
                  bootstyle="success").pack(pady=20)
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            # Check if canvas still exists before trying to scroll
            try:
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except tk.TclError:
                # Canvas was destroyed, unbind the event
                canvas.unbind_all("<MouseWheel>")

        # Bind mouse wheel to canvas (use bind instead of bind_all)
        canvas.bind("<MouseWheel>", _on_mousewheel)
        # Also bind to the scrollable frame
        scrollable_frame.bind("<MouseWheel>", _on_mousewheel)

        # Make sure canvas has focus to receive wheel events
        canvas.focus_set()

        # Unbind when window is closed
        def on_close():
            try:
                canvas.unbind("<MouseWheel>")
                scrollable_frame.unbind("<MouseWheel>")
            except:
                pass
            top.destroy()

        top.protocol("WM_DELETE_WINDOW", on_close)
        
        # Bind mouse wheel to canvas
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Unbind when window is closed
        def on_close():
            canvas.unbind_all("<MouseWheel>")
            top.destroy()
        
        top.protocol("WM_DELETE_WINDOW", on_close)


    def on_profile_select(self, event=None):
        """Load the selected profile's prompt into the text area."""
        name = self.profile_var.get()
        prompt = self.prompt_profiles.get(name, "")
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert("1.0", prompt)

    def save_profile(self):
        """Save current prompt under selected profile and persist."""
        name = self.profile_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Language name cannot be empty.")
            return
        content = self.prompt_text.get('1.0', tk.END).strip()
        self.prompt_profiles[name] = content
        self.config['prompt_profiles'] = self.prompt_profiles
        self.config['active_profile'] = name
        self.profile_menu['values'] = list(self.prompt_profiles.keys())
        messagebox.showinfo("Saved", f"Language '{name}' saved.")
        self.save_profiles()

    def delete_profile(self):
        """Delete the selected language/profile."""
        name = self.profile_var.get()
        if name not in self.prompt_profiles:
            messagebox.showerror("Error", f"Language '{name}' not found.")
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

    def load_glossary(self):
        """Let the user pick a glossary.json and remember its path."""
        path = filedialog.askopenfilename(
            title="Select glossary.json",
            filetypes=[("JSON files", "*.json")]
        )
        if not path:
            return
        self.manual_glossary_path = path
        self.append_log(f"📑 Loaded manual glossary: {path}")

    def trim_glossary(self):
        path = filedialog.askopenfilename(
            title="Select glossary.json to trim",
            filetypes=[("JSON files","*.json")]
        )
        if not path:
            return

        with open(path, 'r', encoding='utf-8') as f:
            glossary = json.load(f)

        dlg = tk.Toplevel(self.master)
        dlg.title("Glossary Trimmer")
        dlg.geometry("420x480")
        dlg.transient(self.master)
        dlg.grab_set()

        labels = [
            "Entries (appearance order):",
            "Traits Trim Count:",
            "Title Keep (0=remove):",
            "GroupAffil Trim Count:",
            "Ref-To-Others Trim Count:",
            "Locations Trim Count:"
        ]
        defaults = [
            "100",
            self.traits_trim.get(),
            self.title_trim.get(),
            self.group_trim.get(),
            self.refer_trim.get(),
            self.loc_trim.get()
        ]
        entries = []
        for i,(lab,defval) in enumerate(zip(labels,defaults)):
            tb.Label(dlg, text=lab).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            e = tb.Entry(dlg, width=6)
            e.insert(0, defval)
            e.grid(row=i, column=1, padx=5, pady=2)
            entries.append(e)

        def aggregate_locations():
            all_locs = []
            for char in glossary:
                locs = char.get('locations', [])
                if isinstance(locs, list):
                    all_locs.extend(locs)
                char.pop('locations', None)

            seen = set()
            unique_locs = []
            for loc in all_locs:
                if loc not in seen:
                    seen.add(loc)
                    unique_locs.append(loc)

            glossary[:] = [entry for entry in glossary if entry.get('original_name') != "📍 Location Summary"]
            glossary.append({
                "original_name": "📍 Location Summary",
                "name": "Location Summary",
                "locations": unique_locs
            })

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(glossary, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Aggregated", f"{len(unique_locs)} unique locations added to glossary.")
            dlg.lift()

        def apply_trim():
            top_limit, traits_lim, title_lim, group_lim, refer_lim, loc_lim = (
                int(e.get()) for e in entries
            )
            trimmed = glossary[:top_limit]
            for char in trimmed:
                if title_lim <= 0:
                    char.pop('title', None)
                if traits_lim <= 0:
                    char.pop('traits', None)
                else:
                    t = char.get('traits', [])
                    char['traits'] = t[:-traits_lim] if len(t)>traits_lim else []
                if group_lim <= 0:
                    char.pop('group_affiliation', None)
                else:
                    g = char.get('group_affiliation', [])
                    char['group_affiliation'] = g[:-group_lim] if len(g)>group_lim else []
                if refer_lim <= 0:
                    char.pop('how_they_refer_to_others', None)
                else:
                    items = list(char.get('how_they_refer_to_others',{}).items())
                    keep = items[:-refer_lim] if len(items)>refer_lim else []
                    char['how_they_refer_to_others'] = dict(keep)
                if loc_lim <= 0:
                    char.pop('locations', None)
                else:
                    l = char.get('locations', [])
                    char['locations'] = l[:-loc_lim] if len(l)>loc_lim else []

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(trimmed, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Trimmed", f"Glossary written with {top_limit} entries.")
            dlg.destroy()

        def delete_empty_fields():
            for char in glossary:
                for key in list(char.keys()):
                    val = char[key]
                    if val in (None, [], {}, ""):
                        char.pop(key, None)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(glossary, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Deleted", "Empty fields removed.")
            dlg.lift()

        tb.Button(dlg, text="Apply", command=apply_trim, bootstyle="success") \
          .grid(row=len(labels), column=0, columnspan=2, pady=10)
        tb.Button(dlg, text="➕ Aggregate Unique Locations",
                  command=aggregate_locations, bootstyle="info") \
          .grid(row=len(labels)+1, column=0, columnspan=2, pady=5)
        tb.Button(
            dlg,
            text="Delete Empty Fields",
            command=delete_empty_fields,
            bootstyle="warning"
        ).grid(row=len(labels)+2, column=0, columnspan=2, pady=5)
        dlg.wait_window()

    def save_config(self):
        """Persist all settings to config.json."""
        try:
            self.config['model'] = self.model_var.get()
            self.config['active_profile'] = self.profile_var.get()
            self.config['prompt_profiles'] = self.prompt_profiles
            self.config['contextual'] = self.contextual_var.get()
            self.config['delay'] = int(self.delay_entry.get())
            self.config['translation_temperature'] = float(self.trans_temp.get())
            self.config['translation_history_limit'] = int(self.trans_history.get())
            self.config['glossary_temperature'] = float(self.glossary_temp.get())
            self.config['glossary_history_limit'] = int(self.glossary_history.get())
            self.config['api_key'] = self.api_key_entry.get()
            self.config['REMOVE_AI_ARTIFACTS'] = self.REMOVE_AI_ARTIFACTS_var.get()
            self.config['chapter_range'] = self.chapter_range_entry.get().strip()
            self.config['use_rolling_summary'] = self.rolling_summary_var.get()
            self.config['summary_role'] = self.summary_role_var.get()
            self.config['max_output_tokens'] = self.max_output_tokens
            # ─── NEW: Save new toggle states ───
            self.config['disable_system_prompt'] = self.disable_system_prompt_var.get()
            self.config['disable_auto_glossary'] = self.disable_auto_glossary_var.get()
            self.config['append_glossary'] = self.append_glossary_var.get()
            self.config['emergency_paragraph_restore'] = self.emergency_restore_var.get()
            self.config['reinforcement_frequency'] = int(self.reinforcement_freq_var.get())
            self.config['reset_failed_chapters'] = self.reset_failed_chapters_var.get()
            self.config['retry_duplicate_bodies'] = self.retry_duplicate_var.get()
            self.config['duplicate_lookback_chapters'] = int(self.duplicate_lookback_var.get())

            
            _tl = self.token_limit_entry.get().strip()
            if _tl.isdigit():
                self.config['token_limit'] = int(_tl)
            else:
                self.config['token_limit'] = None

            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Saved", "Configuration saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {e}")

    def on_resize(self, event):
        if event.widget is self.master:
            sx = event.width / BASE_WIDTH
            sy = event.height / BASE_HEIGHT
            s = min(sx, sy)
            new_w = int(self.run_base_w * s)
            new_h = int(self.run_base_h * s)
            ipadx = max(0, (new_w - self.run_base_w)//2)
            ipady = max(0, (new_h - self.run_base_h)//2)
            self.run_button.grid_configure(ipadx=ipadx, ipady=ipady)

    def log_debug(self, message):
        self.append_log(f"[DEBUG] {message}")


if __name__ == "__main__":
    root = tb.Window(themename="darkly")
    app = TranslatorGUI(root)
    root.mainloop()
