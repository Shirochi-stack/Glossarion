# Standard Library - Core
import io
import json
import logging
import math
import os
import shutil
import sys
import threading
import time

# Standard Library - GUI Framework
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk

# Third-Party - UI Theme Framework (back to normal imports)
import ttkbootstrap as tb
from ttkbootstrap.constants import *

# Splash Screen Manager
from splash_utils import SplashManager

if getattr(sys, 'frozen', False):
    try:
        import multiprocessing
        multiprocessing.freeze_support()
    except:
        pass
        
# =============================================================================
# DEFERRED HEAVY MODULES - Only translation modules need lazy loading
# =============================================================================

# Translation modules (loaded by _lazy_load_modules in TranslatorGUI)
translation_main = None
translation_stop_flag = None
translation_stop_check = None
glossary_main = None
glossary_stop_flag = None
glossary_stop_check = None
fallback_compile_epub = None
scan_html_folder = None

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_FILE = "config.json"
BASE_WIDTH, BASE_HEIGHT = 1550, 1000

# =============================================================================
# ICON LOADING UTILITY (for main window after startup)
# =============================================================================

def load_application_icon(window, base_dir):
    """Load application icon with fallback handling"""
    ico_path = os.path.join(base_dir, 'Halgakos.ico')
    
    # Set window icon (Windows)
    if os.path.isfile(ico_path):
        try:
            window.iconbitmap(ico_path)
        except Exception as e:
            logging.warning(f"Could not set window icon: {e}")
    
    # Set taskbar icon (cross-platform)
    try:
        from PIL import Image, ImageTk
        if os.path.isfile(ico_path):
            icon_image = Image.open(ico_path)
            if icon_image.mode != 'RGBA':
                icon_image = icon_image.convert('RGBA')
            icon_photo = ImageTk.PhotoImage(icon_image)
            window.iconphoto(False, icon_photo)
            return icon_photo  # Keep reference to prevent garbage collection
    except (ImportError, Exception) as e:
        logging.warning(f"Could not load icon image: {e}")
    
    return None

class TranslatorGUI:
    def __init__(self, master):
        self.master = master
        self.max_output_tokens = 8192  # default fallback
        self.proc = None
        self.glossary_proc = None       
        master.title("Glossarion v1.7.1")
        master.geometry(f"{BASE_WIDTH}x{BASE_HEIGHT}")
        master.minsize(1550, 1000)
        master.bind('<F11>', self.toggle_fullscreen)
        master.bind('<Escape>', lambda e: master.attributes('-fullscreen', False))
        self.payloads_dir = os.path.join(os.getcwd(), "Payloads")        
        
        # Module loading state
        self._modules_loaded = False
        self._modules_loading = False
        
        # Add stop flags for threading
        self.stop_requested = False
        self.translation_thread = None
        self.glossary_thread = None
        self.qa_thread = None
        self.epub_thread = None
        
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

        # Load embedded icon image for display (lazy load PIL)
        self.logo_img = None
        try:
            # Delay PIL import
            from PIL import Image, ImageTk
            self.logo_img = ImageTk.PhotoImage(Image.open(ico_path)) if os.path.isfile(ico_path) else None
            if self.logo_img:
                master.iconphoto(False, self.logo_img)
        except Exception as e:
            logging.error(f"Failed to load logo: {e}")
            
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
        
        # Load token limit disabled state from config
        self.token_limit_disabled = self.config.get('token_limit_disabled', False)
        
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
        
        self.disable_glossary_translation_var = tk.BooleanVar(
            value=self.config.get('disable_glossary_translation', False)  # Default to False (translation enabled)
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
        self.glossary_min_frequency_var = tk.StringVar(
            value=str(self.config.get('glossary_min_frequency', 2))  # Changed default to 2
        )
        self.glossary_max_names_var = tk.StringVar(
            value=str(self.config.get('glossary_max_names', 50))
        )
        self.glossary_max_titles_var = tk.StringVar(
            value=str(self.config.get('glossary_max_titles', 30))  # NEW
        )
        self.glossary_batch_size_var = tk.StringVar(
            value=str(self.config.get('glossary_batch_size', 50))
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


    def _lazy_load_modules(self, splash_callback=None):
        """Load heavy modules only when needed - optimized version"""
        if self._modules_loaded:
            return True
            
        if self._modules_loading:
            # Wait for loading to complete
            while self._modules_loading and not self._modules_loaded:
                time.sleep(0.1)
            return self._modules_loaded
                
        self._modules_loading = True
        
        if splash_callback:
            splash_callback("Loading translation modules...")
        
        global translation_main, translation_stop_flag, translation_stop_check
        global glossary_main, glossary_stop_flag, glossary_stop_check  
        global fallback_compile_epub, scan_html_folder
        
        success_count = 0
        total_modules = 4
        
        # Load modules with better error handling and progress feedback
        modules_to_load = [
            ('TransateKRtoEN', 'translation engine'),
            ('extract_glossary_from_epub', 'glossary extractor'),
            ('epub_converter', 'EPUB converter'),
            ('scan_html_folder', 'QA scanner')
        ]
        
        for module_name, display_name in modules_to_load:
            try:
                if splash_callback:
                    splash_callback(f"Loading {display_name}...")
                
                if module_name == 'TransateKRtoEN':
                    from TransateKRtoEN import main as translation_main, set_stop_flag as translation_stop_flag, is_stop_requested as translation_stop_check
                    success_count += 1
                    
                elif module_name == 'extract_glossary_from_epub':
                    from extract_glossary_from_epub import main as glossary_main, set_stop_flag as glossary_stop_flag, is_stop_requested as glossary_stop_check
                    success_count += 1
                    
                elif module_name == 'epub_converter':
                    from epub_converter import fallback_compile_epub
                    success_count += 1
                    
                elif module_name == 'scan_html_folder':
                    from scan_html_folder import scan_html_folder
                    success_count += 1
                    
            except ImportError as e:
                print(f"Warning: Could not import {module_name} module: {e}")
                # Set appropriate globals to None
                if module_name == 'TransateKRtoEN':
                    translation_main = translation_stop_flag = translation_stop_check = None
                elif module_name == 'extract_glossary_from_epub':
                    glossary_main = glossary_stop_flag = glossary_stop_check = None
                elif module_name == 'epub_converter':
                    fallback_compile_epub = None
                elif module_name == 'scan_html_folder':
                    scan_html_folder = None
                    
            except Exception as e:
                print(f"Error loading {module_name}: {e}")
        
        self._modules_loaded = True
        self._modules_loading = False
        
        if splash_callback:
            splash_callback(f"Loaded {success_count}/{total_modules} modules successfully")
        
        # Update UI state after loading (schedule on main thread)
        if hasattr(self, 'master'):
            self.master.after(0, self._check_modules)
        
        # Log success
        if hasattr(self, 'append_log'):
            self.append_log(f"✅ Loaded {success_count}/{total_modules} modules successfully")
        
        return True

    def _check_modules(self):
        """Check which modules are available and disable buttons if needed"""
        if not self._modules_loaded:
            return
            
        if translation_main is None and hasattr(self, 'run_button'):
            self.run_button.config(state='disabled')
            self.append_log("⚠️ Translation module not available")
        
        if glossary_main is None and hasattr(self, 'glossary_button'):
            self.glossary_button.config(state='disabled')
            self.append_log("⚠️ Glossary extraction module not available")
        
        if fallback_compile_epub is None and hasattr(self, 'epub_button'):
            self.epub_button.config(state='disabled')
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

        # Apply the loaded token limit state
        if self.token_limit_disabled:
            self.token_limit_entry.config(state=tk.DISABLED)
            self.toggle_token_btn.config(text="Enable Input Token Limit", bootstyle="success-outline")
        else:
            self.token_limit_entry.config(state=tk.NORMAL)
            self.toggle_token_btn.config(text="Disable Input Token Limit", bootstyle="danger-outline")

        # Initial prompt
        self.on_profile_select()

        print("[DEBUG] GUI setup completed with config values loaded")  # Debug logging
        
        # Add initial log message
        self.append_log("🚀 Glossarion v1.7.1 - Ready to use!")
        self.append_log("💡 Click any function button to load modules automatically")

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
            ("Retranslate",         self.force_retranslation,         "warning"),
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
            elif lbl == "EPUB Converter":
                self.epub_button = btn

        self.frame.grid_rowconfigure(12, weight=0)

    # === DIRECT FUNCTION CALLS ===
    
    def run_translation_thread(self):
        """Start translation in a separate thread"""
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
        
        # FIXED: Start thread immediately without loading modules on main thread
        self.translation_thread = threading.Thread(target=self.run_translation_direct, daemon=True)
        self.translation_thread.start()
        
        # Update button immediately after starting thread
        self.master.after(100, self.update_run_button)

    def run_translation_direct(self):
        """Run translation directly without subprocess"""
        try:
            # FIXED: Load modules in background thread to prevent GUI freezing
            self.append_log("🔄 Loading translation modules...")
            if not self._lazy_load_modules():
                self.append_log("❌ Failed to load translation modules")
                return
                
            if translation_main is None:
                self.append_log("❌ Translation module is not available")
                messagebox.showerror("Module Error", "Translation module is not available. Please ensure all files are present.")
                return
                
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
                    
                # ─── NEW: Log glossary extraction settings ───
                self.append_log(f"📑 Targeted Glossary Settings:")
                self.append_log(f"   • Min frequency: {self.glossary_min_frequency_var.get()} occurrences")
                self.append_log(f"   • Max character names: {self.glossary_max_names_var.get()}")
                self.append_log(f"   • Max titles/ranks: {self.glossary_max_titles_var.get()}")
                self.append_log(f"   • Translation batch size: {self.glossary_batch_size_var.get()}")
                
                # Log glossary translation status
                if self.disable_glossary_translation_var.get():
                    self.append_log("⚠️ Glossary translation disabled - terms will remain in original language")
                else:
                    self.append_log(f"✅ Glossary translation enabled with {self.glossary_batch_size_var.get()} terms per batch")                    
                
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
                    'MAX_RETRY_TOKENS': self.max_retry_tokens_var.get(),
                    'RETRY_DUPLICATE_BODIES': "1" if self.retry_duplicate_var.get() else "0",
                    'DUPLICATE_LOOKBACK_CHAPTERS': self.duplicate_lookback_var.get(),
                    'GLOSSARY_MIN_FREQUENCY': self.glossary_min_frequency_var.get(),
                    'GLOSSARY_MAX_NAMES': self.glossary_max_names_var.get(),
                    'GLOSSARY_MAX_TITLES': self.glossary_max_titles_var.get(),
                    'GLOSSARY_BATCH_SIZE': self.glossary_batch_size_var.get()
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
        # Load modules first
        if not self._lazy_load_modules():
            self.append_log("❌ Failed to load glossary modules")
            return
            
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

    def epub_converter(self):
        """Start EPUB converter in a separate thread"""
        # Load modules first
        if not self._lazy_load_modules():
            self.append_log("❌ Failed to load EPUB converter modules")
            return
            
        if fallback_compile_epub is None:
            self.append_log("❌ EPUB converter module is not available")
            messagebox.showerror("Module Error", "EPUB converter module is not available.")
            return

        # Check if other processes are running
        if hasattr(self, 'translation_thread') and self.translation_thread and self.translation_thread.is_alive():
            self.append_log("⚠️ Cannot run EPUB converter while translation is in progress.")
            messagebox.showwarning("Process Running", "Please wait for translation to complete before converting EPUB.")
            return
            
        if hasattr(self, 'glossary_thread') and self.glossary_thread and self.glossary_thread.is_alive():
            self.append_log("⚠️ Cannot run EPUB converter while glossary extraction is in progress.")
            messagebox.showwarning("Process Running", "Please wait for glossary extraction to complete before converting EPUB.")
            return

        if hasattr(self, 'epub_thread') and self.epub_thread and self.epub_thread.is_alive():
            # Stop existing EPUB conversion
            self.stop_epub_converter()
            return

        folder = filedialog.askdirectory(title="Select translation output folder")
        if not folder:
            return

        self.epub_folder = folder  # Store for the thread
        self.stop_requested = False
        self.epub_thread = threading.Thread(target=self.run_epub_converter_direct, daemon=True)
        self.epub_thread.start()
        # Update buttons immediately after starting thread
        self.master.after(100, self.update_run_button)

    def run_epub_converter_direct(self):
        """Run EPUB converter directly without blocking GUI"""
        try:
            folder = self.epub_folder
            self.append_log("📦 Starting EPUB Converter...")
            
            # Call the EPUB converter function directly with callbacks
            fallback_compile_epub(
                folder, 
                log_callback=self.append_log
            )
            
            if not self.stop_requested:
                # Update to use the new filename format
                base_name = os.path.basename(folder)
                out_file = os.path.join(folder, f"{base_name}.epub")
                
                # Check if the file was actually created
                if os.path.exists(out_file):
                    self.append_log("✅ EPUB Converter completed successfully!")
                    # Show success message on main thread
                    self.master.after(0, lambda: messagebox.showinfo("EPUB Compilation Success", f"Created: {out_file}"))
                else:
                    self.append_log("⚠️ EPUB file was not created. Check the logs for details.")
            
        except Exception as e:
            error_str = str(e)
            self.append_log(f"❌ EPUB Converter error: {error_str}")
            
            # Show error message on main thread (only for non-empty errors)
            if "Document is empty" not in error_str:
                self.master.after(0, lambda: messagebox.showerror("EPUB Converter Failed", f"Error: {error_str}"))
            else:
                self.append_log("📋 Check the log above for details about what went wrong.")
                
        finally:
            # CRITICAL: Clear thread reference and update buttons
            self.epub_thread = None
            self.stop_requested = False
            
            # Schedule button update on main thread
            self.master.after(0, self.update_run_button)
            
            # Also explicitly update the EPUB button (extra safety)
            if hasattr(self, 'epub_button'):
                self.master.after(0, lambda: self.epub_button.config(
                    text="EPUB Converter",
                    command=self.epub_converter,
                    bootstyle="info",
                    state=tk.NORMAL if fallback_compile_epub else tk.DISABLED
                ))

    def run_qa_scan(self):
        """Run QA scan directly without subprocess"""
        # Load modules first
        if not self._lazy_load_modules():
            self.append_log("❌ Failed to load QA scanner modules")
            return
            
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
        epub_running = (
            hasattr(self, 'epub_thread') and 
            self.epub_thread and 
            self.epub_thread.is_alive()
        )

        any_process_running = translation_running or glossary_running or qa_running or epub_running

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
                state=tk.NORMAL if translation_main and not any_process_running else tk.DISABLED
            )
            
        # Update glossary button
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
                    state=tk.NORMAL if glossary_main and not any_process_running else tk.DISABLED
                )
        
        # Update EPUB button
        if hasattr(self, 'epub_button'):
            if epub_running:
                self.epub_button.config(
                    text="Stop EPUB",
                    command=self.stop_epub_converter,
                    bootstyle="danger",
                    state=tk.NORMAL
                )
            else:
                self.epub_button.config(
                    text="EPUB Converter",
                    command=self.epub_converter,
                    bootstyle="info",
                    state=tk.NORMAL if fallback_compile_epub and not any_process_running else tk.DISABLED
                )
                
        # Update QA button
        if hasattr(self, 'qa_button'):
            self.qa_button.config(
                state=tk.NORMAL if scan_html_folder and not any_process_running else tk.DISABLED
            )

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

    def stop_epub_converter(self):
        """Stop EPUB converter"""
        self.stop_requested = True
        self.append_log("❌ EPUB converter stop requested.")
        self.append_log("⏳ Please wait... stopping after current operation completes.")
        self.update_run_button()

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
        """Open the Other Settings dialog with all advanced options in a grid layout"""
        top = tk.Toplevel(self.master)
        top.title("Other Settings")
        top.geometry("735x920")  # Made taller to accommodate new controls
        top.transient(self.master)
        top.grab_set()
        
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
        
        # Configure grid columns for the scrollable frame (2 columns layout)
        scrollable_frame.grid_columnconfigure(0, weight=1, uniform="column")
        scrollable_frame.grid_columnconfigure(1, weight=1, uniform="column")
        
        # =================================================================
        # SECTION 1: CONTEXT MANAGEMENT (Top Left)
        # =================================================================
        section1_frame = tk.LabelFrame(scrollable_frame, text="Context Management", padx=10, pady=10)
        section1_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=(10, 5))
        
        # Rolling Summary
        tb.Checkbutton(section1_frame, text="Use Rolling Summary", 
                       variable=self.rolling_summary_var,
                       bootstyle="round-toggle").pack(anchor=tk.W)
        
        tk.Label(section1_frame, 
                 text="Generates context summaries to maintain\ncontinuity when history is cleared",
                 font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        # Summary Role
        summary_frame = tk.Frame(section1_frame)
        summary_frame.pack(anchor=tk.W, padx=20, pady=(0, 10))
        tk.Label(summary_frame, text="Summary Role:").pack(side=tk.LEFT)
        ttk.Combobox(summary_frame, textvariable=self.summary_role_var,
                     values=["user", "system"], state="readonly", width=10).pack(side=tk.LEFT, padx=5)
        
        # =================================================================
        # SECTION 2: RESPONSE HANDLING (Top Right)
        # =================================================================
        section2_frame = tk.LabelFrame(scrollable_frame, text="Response Handling & Retry Logic", padx=10, pady=10)
        section2_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=(10, 5))
        
        # Retry Truncated Responses
        tb.Checkbutton(section2_frame, text="Auto-retry Truncated Responses", 
                       variable=self.retry_truncated_var,
                       bootstyle="round-toggle").pack(anchor=tk.W)
        
        retry_frame = tk.Frame(section2_frame)
        retry_frame.pack(anchor=tk.W, padx=20, pady=(5, 5))
        tk.Label(retry_frame, text="Max retry tokens:").pack(side=tk.LEFT)
        tb.Entry(retry_frame, width=8, textvariable=self.max_retry_tokens_var).pack(side=tk.LEFT, padx=5)
        
        tk.Label(section2_frame, 
                 text="Automatically retry when API response\nis cut off due to token limits",
                 font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
        
        # Retry Duplicate Content
        tb.Checkbutton(section2_frame, text="Auto-retry Duplicate Content", 
                       variable=self.retry_duplicate_var,
                       bootstyle="round-toggle").pack(anchor=tk.W)

        duplicate_frame = tk.Frame(section2_frame)
        duplicate_frame.pack(anchor=tk.W, padx=20, pady=(5, 0))
        tk.Label(duplicate_frame, text="Check last").pack(side=tk.LEFT)
        tb.Entry(duplicate_frame, width=4, textvariable=self.duplicate_lookback_var).pack(side=tk.LEFT, padx=3)
        tk.Label(duplicate_frame, text="chapters").pack(side=tk.LEFT)

        tk.Label(section2_frame, 
                 text="Detects when AI returns same content\nfor different chapters",
                 font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(5, 5))
        
        # =================================================================
        # SECTION 3: PROMPT MANAGEMENT (Middle Left)
        # =================================================================
        section3_frame = tk.LabelFrame(scrollable_frame, text="Prompt Management", padx=10, pady=10)
        section3_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=5)
        
        # Reinforcement Frequency
        reinforce_frame = tk.Frame(section3_frame)
        reinforce_frame.pack(anchor=tk.W, pady=(0, 5))
        tk.Label(reinforce_frame, text="Reinforce every").pack(side=tk.LEFT)
        tb.Entry(reinforce_frame, width=6, textvariable=self.reinforcement_freq_var).pack(side=tk.LEFT, padx=5)
        tk.Label(reinforce_frame, text="messages").pack(side=tk.LEFT)
        
        tk.Label(section3_frame, 
                 text="Periodically reminds the AI of your\nsystem prompt (0 = disabled)",
                 font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 10))
        
        # Disable System Prompts
        tb.Checkbutton(section3_frame, text="Disable Hardcoded Prompts", 
                       variable=self.disable_system_prompt_var,
                       bootstyle="round-toggle").pack(anchor=tk.W)
        
        tk.Label(section3_frame, 
                 text="Uses only your custom system prompt,\nignoring built-in language prompts",
                 font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        # =================================================================
        # SECTION 4: GLOSSARY SETTINGS (Middle Right) - EXPANDED
        # =================================================================
        section4_frame = tk.LabelFrame(scrollable_frame, text="Glossary Settings", padx=10, pady=10)
        section4_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=5)
        
        # Disable Auto Glossary
        tb.Checkbutton(section4_frame, text="Disable Auto Glossary Generation", 
                       variable=self.disable_auto_glossary_var,
                       bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
        
        # Disable Glossary Translation
        tb.Checkbutton(section4_frame, text="Disable Glossary Translation", 
                       variable=self.disable_glossary_translation_var,
                       bootstyle="round-toggle").pack(anchor=tk.W, pady=2)

        tk.Label(section4_frame, 
                 text="Keeps foreign terms in original language\nwithout English translations",
                 font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        # Append Glossary
        tb.Checkbutton(section4_frame, text="Append Glossary to Prompt", 
                       variable=self.append_glossary_var,
                       bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
        
        tk.Label(section4_frame, 
                 text="Automatically includes glossary terms\nin translation prompts",
                 font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 8))
        
        # =================================================================
        # SECTION 5: AUTOMATIC GLOSSARY EXTRACTION CONTROLS (NEW - Bottom Left)
        # =================================================================
        section5_frame = tk.LabelFrame(scrollable_frame, text="Targeted Automatic Glossary Extraction", padx=10, pady=10)
        section5_frame.grid(row=2, column=0, sticky="nsew", padx=(10, 5), pady=5)

        # Configure grid for consistent alignment
        section5_frame.grid_columnconfigure(0, weight=0)  # Label column
        section5_frame.grid_columnconfigure(1, weight=1)  # Entry column

        # Add description
        tk.Label(section5_frame, 
                 text="Extracts only character names with honorifics and titles",
                 font=('TkDefaultFont', 10, 'italic'), fg='gray', justify=tk.LEFT).grid(
                 row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))

        # Min Frequency
        tk.Label(section5_frame, text="Min frequency:").grid(row=1, column=0, sticky=tk.W, pady=2)
        tb.Entry(section5_frame, width=8, textvariable=self.glossary_min_frequency_var).grid(row=1, column=1, sticky=tk.W, padx=(5,0), pady=2)

        tk.Label(section5_frame, 
                 text="Minimum appearances required (lower = more terms)",
                 font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))

        # Max Names
        tk.Label(section5_frame, text="Max names:").grid(row=3, column=0, sticky=tk.W, pady=2)
        tb.Entry(section5_frame, width=8, textvariable=self.glossary_max_names_var).grid(row=3, column=1, sticky=tk.W, padx=(5,0), pady=2)

        tk.Label(section5_frame, 
                 text="Maximum character names to extract",
                 font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))

        # Max Titles
        tk.Label(section5_frame, text="Max titles:").grid(row=5, column=0, sticky=tk.W, pady=2)
        tb.Entry(section5_frame, width=8, textvariable=self.glossary_max_titles_var).grid(row=5, column=1, sticky=tk.W, padx=(5,0), pady=2)

        tk.Label(section5_frame, 
                 text="Maximum titles/ranks to extract",
                 font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))

        # Batch Size
        tk.Label(section5_frame, text="Translation batch:").grid(row=7, column=0, sticky=tk.W, pady=2)
        tb.Entry(section5_frame, width=8, textvariable=self.glossary_batch_size_var).grid(row=7, column=1, sticky=tk.W, padx=(5,0), pady=2)

        tk.Label(section5_frame, 
                 text="Terms per API call for translation",
                 font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        # =================================================================
        # SECTION 6: PROCESSING OPTIONS (Bottom Right)
        # =================================================================
        section6_frame = tk.LabelFrame(scrollable_frame, text="Processing Options", padx=10, pady=10)
        section6_frame.grid(row=2, column=1, sticky="nsew", padx=(5, 10), pady=5)
        
        # Emergency Paragraph Restoration
        tb.Checkbutton(section6_frame, text="Emergency Paragraph Restoration", 
                       variable=self.emergency_restore_var,
                       bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
        
        tk.Label(section6_frame, 
                 text="Fixes AI responses that lose paragraph\nstructure (wall of text)",
                 font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        # Reset Failed Chapters
        tb.Checkbutton(section6_frame, text="Reset Failed Chapters on Start", 
                       variable=self.reset_failed_chapters_var,
                       bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
        
        tk.Label(section6_frame, 
                 text="Automatically retry failed/deleted chapters\non each translation run",
                 font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
        
        # EPUB Utilities
        tk.Label(section6_frame, text="EPUB Utilities:", font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W, pady=(5, 5))
        
        # Validation Button
        tb.Button(section6_frame, text="🔍 Validate EPUB Structure", 
                  command=self.validate_epub_structure_gui, 
                  bootstyle="success-outline",
                  width=25).pack(anchor=tk.W, pady=2)
        
        tk.Label(section6_frame, 
                 text="Check if all required EPUB files are\npresent for compilation",
                 font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 5))
        
        # =================================================================
        # SAVE & CLOSE FUNCTIONALITY (Bottom spanning both columns)
        # =================================================================
        def save_and_close():
            """Save all settings and close the dialog"""
            try:
                # Context Management
                self.config['use_rolling_summary'] = self.rolling_summary_var.get()
                self.config['summary_role'] = self.summary_role_var.get()
                
                # Response Handling
                self.config['retry_truncated'] = self.retry_truncated_var.get()
                self.config['max_retry_tokens'] = int(self.max_retry_tokens_var.get())
                self.config['retry_duplicate_bodies'] = self.retry_duplicate_var.get()
                self.config['duplicate_lookback_chapters'] = int(self.duplicate_lookback_var.get())
                
                # Prompt Management
                self.config['reinforcement_frequency'] = int(self.reinforcement_freq_var.get())
                self.config['disable_system_prompt'] = self.disable_system_prompt_var.get()
                
                # Glossary Settings
                self.config['disable_auto_glossary'] = self.disable_auto_glossary_var.get()
                self.config['disable_glossary_translation'] = self.disable_glossary_translation_var.get()
                self.config['append_glossary'] = self.append_glossary_var.get()
                
                # NEW: Glossary Extraction Controls
                self.config['glossary_min_frequency'] = int(self.glossary_min_frequency_var.get())
                self.config['glossary_max_names'] = int(self.glossary_max_names_var.get())
                self.config['glossary_max_titles'] = int(self.glossary_max_titles_var.get()) 
                self.config['glossary_batch_size'] = int(self.glossary_batch_size_var.get())
                
                # Processing Options
                self.config['emergency_paragraph_restore'] = self.emergency_restore_var.get()
                self.config['reset_failed_chapters'] = self.reset_failed_chapters_var.get()
                
                # Set environment variables for immediate effect
                os.environ["USE_ROLLING_SUMMARY"] = "1" if self.rolling_summary_var.get() else "0"
                os.environ["SUMMARY_ROLE"] = self.summary_role_var.get()
                os.environ["RETRY_TRUNCATED"] = "1" if self.retry_truncated_var.get() else "0"
                os.environ["MAX_RETRY_TOKENS"] = self.max_retry_tokens_var.get()
                os.environ["RETRY_DUPLICATE_BODIES"] = "1" if self.retry_duplicate_var.get() else "0"
                os.environ["DUPLICATE_LOOKBACK_CHAPTERS"] = self.duplicate_lookback_var.get()
                os.environ["REINFORCEMENT_FREQUENCY"] = self.reinforcement_freq_var.get()
                os.environ["DISABLE_SYSTEM_PROMPT"] = "1" if self.disable_system_prompt_var.get() else "0"
                os.environ["DISABLE_AUTO_GLOSSARY"] = "1" if self.disable_auto_glossary_var.get() else "0"
                os.environ["DISABLE_GLOSSARY_TRANSLATION"] = "1" if self.disable_glossary_translation_var.get() else "0"
                os.environ["APPEND_GLOSSARY"] = "1" if self.append_glossary_var.get() else "0"
                os.environ["EMERGENCY_PARAGRAPH_RESTORE"] = "1" if self.emergency_restore_var.get() else "0"
                os.environ["RESET_FAILED_CHAPTERS"] = "1" if self.reset_failed_chapters_var.get() else "0"
                
                # NEW: Glossary extraction environment variables
                os.environ["GLOSSARY_MIN_FREQUENCY"] = self.glossary_min_frequency_var.get()
                os.environ["GLOSSARY_MAX_NAMES"] = self.glossary_max_names_var.get()
                os.environ["GLOSSARY_MAX_TITLES"] = self.glossary_max_titles_var.get()
                os.environ["GLOSSARY_BATCH_SIZE"] = self.glossary_batch_size_var.get()
                
                # Save to config file
                with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                
                self.append_log("✅ Other Settings saved successfully")
                top.destroy()
                
            except ValueError as e:
                messagebox.showerror("Invalid Input", f"Please enter valid numbers for all numeric fields.\nError: {e}")
            except Exception as e:
                print(f"❌ Failed to save Other Settings: {e}")
                messagebox.showerror("Error", f"Failed to save settings: {e}")
        
        # Save button frame spanning both columns
        button_frame = tk.Frame(scrollable_frame)
        button_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=20)
        
        # Center the buttons
        button_container = tk.Frame(button_frame)
        button_container.pack(expand=True)
        
        tb.Button(button_container, text="💾 Save Settings", command=save_and_close, 
                  bootstyle="success", width=20).pack(side=tk.LEFT, padx=5)
        
        tb.Button(button_container, text="❌ Cancel", command=top.destroy, 
                  bootstyle="secondary", width=15).pack(side=tk.LEFT, padx=5)
        
        # =================================================================
        # MOUSE WHEEL SCROLLING SUPPORT
        # =================================================================
        def _on_mousewheel(event):
            """Handle mouse wheel scrolling"""
            try:
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except tk.TclError:
                # Canvas was destroyed, ignore
                pass

        # Bind mouse wheel events
        canvas.bind("<MouseWheel>", _on_mousewheel)
        scrollable_frame.bind("<MouseWheel>", _on_mousewheel)
        canvas.focus_set()

        # Cleanup on close
        def on_close():
            """Cleanup when dialog is closed"""
            try:
                canvas.unbind("<MouseWheel>")
                scrollable_frame.unbind("<MouseWheel>")
            except:
                pass
            top.destroy()

        top.protocol("WM_DELETE_WINDOW", on_close)



    # Keep the validation function as-is:
    def validate_epub_structure_gui(self):
        """GUI wrapper for EPUB structure validation"""
        epub_path = self.entry_epub.get()
        if not epub_path:
            messagebox.showerror("Error", "Please select an EPUB file first.")
            return
        
        # Get output directory
        epub_base = os.path.splitext(os.path.basename(epub_path))[0]
        output_dir = epub_base
        
        if not os.path.exists(output_dir):
            messagebox.showinfo("Info", f"No output directory found: {output_dir}")
            return
        
        self.append_log("🔍 Validating EPUB structure...")
        
        # Import the validation functions
        try:
            from TransateKRtoEN import validate_epub_structure, check_epub_readiness
            
            # Run validation
            structure_ok = validate_epub_structure(output_dir)
            readiness_ok = check_epub_readiness(output_dir)
            
            # Show results
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
            self.config['disable_system_prompt'] = self.disable_system_prompt_var.get()
            self.config['disable_auto_glossary'] = self.disable_auto_glossary_var.get()
            self.config['append_glossary'] = self.append_glossary_var.get()
            self.config['emergency_paragraph_restore'] = self.emergency_restore_var.get()
            self.config['reinforcement_frequency'] = int(self.reinforcement_freq_var.get())
            self.config['reset_failed_chapters'] = self.reset_failed_chapters_var.get()
            self.config['retry_duplicate_bodies'] = self.retry_duplicate_var.get()
            self.config['duplicate_lookback_chapters'] = int(self.duplicate_lookback_var.get())
            self.config['token_limit_disabled'] = self.token_limit_disabled
            self.config['disable_glossary_translation'] = self.disable_glossary_translation_var.get()
            self.config['glossary_min_frequency'] = int(self.glossary_min_frequency_var.get())
            self.config['glossary_max_names'] = int(self.glossary_max_names_var.get())
            self.config['glossary_max_titles'] = int(self.glossary_max_titles_var.get())
            self.config['glossary_batch_size'] = int(self.glossary_batch_size_var.get())

            
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
    import time  # Add this import
    
    print("🚀 Starting Glossarion v1.7.1...")
    
    # Initialize splash screen (main thread only)
    splash_manager = None
    try:
        from splash_utils import SplashManager
        splash_manager = SplashManager()
        splash_started = splash_manager.start_splash()
        
        if splash_started:
            splash_manager.update_status("Loading theme framework...")
            time.sleep(0.5)  # Give user time to see the status
        
    except Exception as e:
        print(f"⚠️ Splash screen failed: {e}")
        splash_manager = None
    
    try:
        # Import heavy modules
        if splash_manager:
            splash_manager.update_status("Loading UI framework...")
            time.sleep(0.3)
        
        import ttkbootstrap as tb
        from ttkbootstrap.constants import *
        
        if splash_manager:
            splash_manager.update_status("Creating main window...")
            time.sleep(0.3)
        
        # Close splash before creating main window
        if splash_manager:
            splash_manager.update_status("Ready!")
            time.sleep(0.5)
            splash_manager.close_splash()
        
        # Now create main window (on same thread)
        root = tb.Window(themename="darkly")
        
        # Initialize the app
        app = TranslatorGUI(root)
        
        print("✅ Ready to use!")
        
        # Start main loop
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        if splash_manager:
            splash_manager.close_splash()
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Ensure splash is closed
        if splash_manager:
            try:
                splash_manager.close_splash()
            except:
                pass
