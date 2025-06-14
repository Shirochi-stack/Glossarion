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
        master.title("Glossarion v2.1.0")
        master.geometry(f"{BASE_WIDTH}x{BASE_HEIGHT}")
        master.minsize(1600, 1000)
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
        
        self.default_manual_glossary_prompt = """Output exactly a JSON array of objects and nothing else.
        You are a glossary extractor for Korean, Japanese, or Chinese novels.
        - Extract character information (e.g., name, traits), locations (countries, regions, cities), and translate them into English (romanization or equivalent).
        - Romanize all untranslated honorifics and suffixes (e.g., 님 to '-nim', さん to '-san').
        - all output must be in english, unless specified otherwise
        For each character, provide JSON fields:
        {fields}
        Sort by appearance order; respond with a JSON array only.

        Text:
        {chapter_text}"""

        self.default_auto_glossary_prompt = """You are extracting a targeted glossary from a Korean/Japanese/Chinese novel.
        Focus on identifying:
        1. Character names with their honorifics/suffixes
        2. Important titles and ranks
        3. Frequently mentioned terms (min frequency: {min_frequency})

        Extract up to {max_names} character names and {max_titles} titles.
        Prioritize names that appear with honorifics or in important contexts.
        Return the glossary in a simple key-value format."""

        # In __init__ method, after default_auto_glossary_prompt
        self.default_rolling_summary_system_prompt = """You are a context summarization assistant. Create concise, informative summaries that preserve key story elements for translation continuity."""

        self.default_rolling_summary_user_prompt = """Analyze the recent translation exchanges and create a structured summary for context continuity.

        Focus on extracting and preserving:
        1. **Character Information**: Names (with original forms), relationships, roles, and important character developments
        2. **Plot Points**: Key events, conflicts, and story progression
        3. **Locations**: Important places and settings
        4. **Terminology**: Special terms, abilities, items, or concepts (with original forms)
        5. **Tone & Style**: Writing style, mood, and any notable patterns
        6. **Unresolved Elements**: Questions, mysteries, or ongoing situations

        Format the summary clearly with sections. Be concise but comprehensive.

        Recent translations to summarize:
        {translations}"""

        # Load saved prompts from config
        self.rolling_summary_system_prompt = self.config.get('rolling_summary_system_prompt', self.default_rolling_summary_system_prompt)
        self.rolling_summary_user_prompt = self.config.get('rolling_summary_user_prompt', self.default_rolling_summary_user_prompt)
        
        # Load saved prompts from config
        self.manual_glossary_prompt = self.config.get('manual_glossary_prompt', self.default_manual_glossary_prompt)
        self.auto_glossary_prompt = self.config.get('auto_glossary_prompt', self.default_auto_glossary_prompt)
        
        

        # Add custom glossary fields configuration
        self.custom_glossary_fields = self.config.get('custom_glossary_fields', [])
        
        # Load token limit disabled state from config
        self.token_limit_disabled = self.config.get('token_limit_disabled', False)
        
        # ─── restore rolling-summary state from config.json ───
        self.rolling_summary_var = tk.BooleanVar(
            value=self.config.get('use_rolling_summary', False)
        )
        self.summary_role_var = tk.StringVar(
            value=self.config.get('summary_role', 'user')
        )
        
         # ADD THESE NEW LINES HERE:
        self.rolling_summary_exchanges_var = tk.StringVar(
            value=str(self.config.get('rolling_summary_exchanges', '5'))  # How many exchanges to summarize
        )
        self.rolling_summary_mode_var = tk.StringVar(
            value=self.config.get('rolling_summary_mode', 'append')  # append or replace
        )
        
        # ─── NEW: Add variables for new toggles ───
        self.disable_system_prompt_var = tk.BooleanVar(
            value=self.config.get('disable_system_prompt', False)
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
        # ─── IMAGE TRANSLATION SETTINGS ───
        self.enable_image_translation_var = tk.BooleanVar(
            value=self.config.get('enable_image_translation', False)  # Default OFF
        )
        
        # Web novel image settings
        self.process_webnovel_images_var = tk.BooleanVar(
            value=self.config.get('process_webnovel_images', True)  # Default ON
        )
        
        self.webnovel_min_height_var = tk.StringVar(
            value=str(self.config.get('webnovel_min_height', '1000'))
        )
        
        self.image_max_tokens_var = tk.StringVar(
            value=str(self.config.get('image_max_tokens', '16384'))
        )
        
        self.max_images_per_chapter_var = tk.StringVar(
            value=str(self.config.get('max_images_per_chapter', '10'))
        )
        self.comprehensive_extraction_var = tk.BooleanVar(
            value=self.config.get('comprehensive_extraction', False)  # Default to False (smart mode)
        )
        self.image_chunk_height_var = tk.StringVar(
            value=str(self.config.get('image_chunk_height', '2000'))
        )        
        self.hide_image_translation_label_var = tk.BooleanVar(
            value=self.config.get('hide_image_translation_label', True)
        )
        self.chunk_timeout_var = tk.StringVar(
            value=str(self.config.get('chunk_timeout', '300'))  # 5 minutes default
        )
        self.retry_timeout_var = tk.BooleanVar(
            value=self.config.get('retry_timeout', False)
        )
        self.batch_translation_var = tk.BooleanVar(
            value=self.config.get('batch_translation', False)  # Default to False
        )
        self.batch_size_var = tk.StringVar(
            value=str(self.config.get('batch_size', '3'))  # Default to 3
)
        # Default prompts
        self.default_prompts = {
            "korean": "You are a professional Korean to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n- Use an easy to read and grammatically accurate comedy translation style.\n- Retain honorifics, and suffixes like -nim, -ssi.\n- Preserve original intent, and speech tone.\n- retain onomatopoeia in Romaji.",
            "japanese": "You are a professional Japanese to English novel translator, you must strictly output only English text and HTML tags text while following these rules:\n- Use an easy to read and grammatically accurate comedy translation style.\n- Retain honorifics, and suffixes like -san, -sama, -chan, -kun.\n- Preserve original intent, and speech tone.\n- retain onomatopoeia in Romaji.",
            "chinese": "You are a professional Chinese to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n- Use an easy to read and grammatically accurate comedy translation style.\n- Preserve original intent, and speech tone.\n- retain onomatopoeia in Romaji."
        }

        # Profiles - FIXED: Load from config properly
        self.prompt_profiles = self.config.get('prompt_profiles', self.default_prompts.copy())
        active = self.config.get('active_profile', next(iter(self.prompt_profiles)))
        self.profile_var = tk.StringVar(value=active)
        self.lang_var = self.profile_var

        # Initialize GUI components
        self._setup_gui()
        
    def _setup_text_undo_redo(self, text_widget):
        """Set up undo/redo bindings for a text widget with error handling"""
        def handle_undo(event):
            try:
                text_widget.edit_undo()
            except tk.TclError:
                pass  # Nothing to undo
            return "break"
            
        def handle_redo(event):
            try:
                text_widget.edit_redo()
            except tk.TclError:
                pass  # Nothing to redo
            return "break"
        
        # Windows/Linux bindings
        text_widget.bind('<Control-z>', handle_undo)
        text_widget.bind('<Control-y>', handle_redo)
        
        # macOS bindings
        text_widget.bind('<Command-z>', handle_undo)
        text_widget.bind('<Command-Shift-z>', handle_redo)

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
            
            
    def _auto_resize_dialog(self, dialog, canvas, max_width_ratio=0.9, max_height_ratio=0.95):
        """Auto-resize dialog WIDTH ONLY - preserves existing height"""
        # Force all widgets to calculate their sizes
        dialog.update()
        canvas.update()
        
        # Get the current geometry to preserve height
        current_geometry = dialog.geometry()
        current_height = int(current_geometry.split('x')[1].split('+')[0])
        
        # Get the frame inside the canvas
        scrollable_frame = None
        for child in canvas.winfo_children():
            if isinstance(child, ttk.Frame):
                scrollable_frame = child
                break
        
        if not scrollable_frame:
            return
        
        # Force the frame to calculate its natural size
        scrollable_frame.update_idletasks()
        
        # Calculate WIDTH based on content
        window_width = scrollable_frame.winfo_reqwidth()+ 20
        
        # Get screen dimensions
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        
        # Calculate width limit
        max_width = int(screen_width * max_width_ratio)
        
        # Apply width limit
        final_width = min(window_width, max_width)
        
        # KEEP THE EXISTING HEIGHT
        final_height = current_height
        
        # Set size and center
        x = (screen_width - final_width) // 2
        y = max(20, (screen_height - final_height) // 2)
        dialog.geometry(f"{final_width}x{final_height}+{x}+{y}")
        
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
                    values=["gpt-4o","gpt-4o-mini","gpt-4-turbo","gpt-4.1-nano","gpt-4.1-mini","gpt-4.1","gpt-3.5-turbo","o4-mini","gemini-1.5-pro","gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-exp","gemini-2.5-flash-preview-05-20","gemini-2.5-pro-preview-06-05","deepseek-chat","claude-3-5-sonnet-20241022","claude-3-7-sonnet-20250219"], state="normal").grid(
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

        #Batch Translation
        tb.Checkbutton(self.frame, text="Batch Translation", 
                       variable=self.batch_translation_var,
                       bootstyle="round-toggle").grid(row=6, column=2, sticky=tk.W, padx=5, pady=5)

        self.batch_size_entry = tb.Entry(self.frame, width=6, textvariable=self.batch_size_var)
        self.batch_size_entry.grid(row=6, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Update the toggle state on checkbox change
        def toggle_batch_entry():
            if self.batch_translation_var.get():
                self.batch_size_entry.config(state=tk.NORMAL)
            else:
                self.batch_size_entry.config(state=tk.DISABLED)

        # Set initial state first
        toggle_batch_entry()

        # Then bind the toggle function
        self.batch_translation_var.trace('w', lambda *args: toggle_batch_entry())
                
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
        self._setup_text_undo_redo(self.prompt_text)
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

        # Log area - Fixed scrolling and copying issue
        self.log_text = scrolledtext.ScrolledText(self.frame, wrap=tk.WORD)
        self.log_text.grid(row=10, column=0, columnspan=5, sticky=tk.NSEW, padx=5, pady=5)

        # Make log read-only but allow selection and copying
        self.log_text.bind("<Key>", self._block_editing)

        # Enable right-click context menu for copying
        self.log_text.bind("<Button-3>", self._show_context_menu)  # Right-click on Windows/Linux
        if sys.platform == "darwin":  # macOS
            self.log_text.bind("<Button-2>", self._show_context_menu)
            
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
        self.append_log("🚀 Glossarion v2.1.0 - Ready to use!")
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
    


    def _setup_dialog_scrolling(self, dialog_window, canvas):
        """Setup mouse wheel scrolling for any dialog with a canvas
        
        Args:
            dialog_window: The toplevel window
            canvas: The canvas widget to scroll
        """
        def _on_mousewheel(event):
            try:
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except:
                pass

        def _on_mousewheel_linux(event, direction):
            """Handle mouse wheel scrolling on Linux"""
            try:
                if canvas.winfo_exists():
                    canvas.yview_scroll(direction, "units")
            except tk.TclError:
                pass

        # Create event handler references for cleanup
        wheel_handler = lambda e: _on_mousewheel(e)
        wheel_up_handler = lambda e: _on_mousewheel_linux(e, -1)
        wheel_down_handler = lambda e: _on_mousewheel_linux(e, 1)
        
        # Bind mouse wheel events
        dialog_window.bind_all("<MouseWheel>", wheel_handler)
        dialog_window.bind_all("<Button-4>", wheel_up_handler)  # Linux
        dialog_window.bind_all("<Button-5>", wheel_down_handler)  # Linux
        
        # Clean up bindings when window is destroyed
        def cleanup_bindings():
            try:
                dialog_window.unbind_all("<MouseWheel>")
                dialog_window.unbind_all("<Button-4>")
                dialog_window.unbind_all("<Button-5>")
            except:
                pass
        
        # Return the cleanup function so it can be used in buttons/close handlers
        return cleanup_bindings
    
    def glossary_manager(self):
        """Open comprehensive glossary management dialog"""
        manager = tk.Toplevel(self.master)
        manager.title("Glossary Manager")
        manager.geometry("1200x1470")
        manager.transient(self.master)
        
        
        # Main container
        main_container = tk.Frame(manager)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas and scrollbar for scrollable content
        canvas = tk.Canvas(main_container, bg='white')
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
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
        
        # Set up mouse wheel scrolling
        cleanup_scrolling = self._setup_dialog_scrolling(manager, canvas)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(scrollable_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Manual Glossary Settings
        manual_frame = ttk.Frame(notebook)
        notebook.add(manual_frame, text="Manual Glossary Extraction")
        
        # Tab 2: Automatic Glossary Settings
        auto_frame = ttk.Frame(notebook)
        notebook.add(auto_frame, text="Automatic Glossary Generation")
        
        # Tab 3: Glossary Editor/Trimmer
        editor_frame = ttk.Frame(notebook)
        notebook.add(editor_frame, text="Glossary Editor")
        
        # ===== MANUAL GLOSSARY TAB =====
        manual_container = tk.Frame(manual_frame)
        manual_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Field selection section
        fields_frame = tk.LabelFrame(manual_container, text="Extraction Fields", padx=10, pady=10)
        fields_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create field variables if not exists
        if not hasattr(self, 'manual_field_vars'):
            self.manual_field_vars = {
                'original_name': tk.BooleanVar(value=self.config.get('manual_extract_original_name', True)), 
                'name': tk.BooleanVar(value=self.config.get('manual_extract_name', True)),
                'gender': tk.BooleanVar(value=self.config.get('manual_extract_gender', True)),
                'title': tk.BooleanVar(value=self.config.get('manual_extract_title', True)),
                'group_affiliation': tk.BooleanVar(value=self.config.get('manual_extract_group', True)),
                'traits': tk.BooleanVar(value=self.config.get('manual_extract_traits', True)),
                'how_they_refer_to_others': tk.BooleanVar(value=self.config.get('manual_extract_refer', True)),
                'locations': tk.BooleanVar(value=self.config.get('manual_extract_locations', True))
            }
        
        # Field descriptions
        field_info = {
            'original_name': "Original name in source language",
            'name': "English/romanized name translation",
            'gender': "Character gender",
            'title': "Title or rank (with romanized suffix)",
            'group_affiliation': "Organization/group membership",
            'traits': "Character traits and descriptions",
            'how_they_refer_to_others': "How they address other characters",
            'locations': "Place names mentioned"
        }
        
        # Create checkboxes in grid
        fields_grid = tk.Frame(fields_frame)
        fields_grid.pack(fill=tk.X)

        row = 0
        for field, var in self.manual_field_vars.items():
            cb = tb.Checkbutton(fields_grid, text=field.replace('_', ' ').title(), 
                                variable=var, bootstyle="round-toggle")
            cb.grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            
            desc = tk.Label(fields_grid, text=field_info[field], 
                           font=('TkDefaultFont', 9), fg='gray')
            desc.grid(row=row, column=1, sticky=tk.W, padx=20, pady=2)
            
            row += 1
        
        # Custom fields section
        custom_frame = tk.LabelFrame(manual_container, text="Custom Fields", padx=10, pady=10)
        custom_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Custom fields list
        custom_list_frame = tk.Frame(custom_frame)
        custom_list_frame.pack(fill=tk.X)
        
        tk.Label(custom_list_frame, text="Additional fields to extract:").pack(anchor=tk.W)
        
        # Scrollable frame for custom fields
        custom_scroll = ttk.Scrollbar(custom_list_frame)
        custom_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.custom_fields_listbox = tk.Listbox(custom_list_frame, height=5, 
                                               yscrollcommand=custom_scroll.set)
        self.custom_fields_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        custom_scroll.config(command=self.custom_fields_listbox.yview)
        
        # Populate with existing custom fields
        for field in self.custom_glossary_fields:
            self.custom_fields_listbox.insert(tk.END, field)
        
        # Custom field controls
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
        
        tb.Button(custom_controls, text="Add", command=add_custom_field, 
                  width=10).pack(side=tk.LEFT, padx=2)
        tb.Button(custom_controls, text="Remove", command=remove_custom_field, 
                  width=10).pack(side=tk.LEFT, padx=2)
        
        # System prompt section
        prompt_frame = tk.LabelFrame(manual_container, text="Extraction Prompt Template", 
                                    padx=10, pady=10)
        prompt_frame.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        tk.Label(prompt_frame, 
                 text="Use {fields} for field list and {chapter_text} for content placeholder",
                 font=('TkDefaultFont', 9), fg='blue').pack(anchor=tk.W, pady=(0, 5))
                 
        
        # manual prompt text area
        self.manual_prompt_text = scrolledtext.ScrolledText(prompt_frame, height=12, wrap=tk.WORD)

        self.manual_prompt_text = scrolledtext.ScrolledText(
            prompt_frame, 
            height=12, 
            wrap=tk.WORD,
            undo=True,
            autoseparators=True,
            maxundo=-1
        )
        self.manual_prompt_text.pack(fill=tk.BOTH, expand=True)
        self.manual_prompt_text.insert('1.0', self.manual_glossary_prompt)

        # Add undo/redo bindings
        self.manual_prompt_text.edit_reset()  # Reset undo stack after initial insert
        self._setup_text_undo_redo(self.manual_prompt_text)
        
        # Prompt controls
        prompt_controls = tk.Frame(manual_container)
        prompt_controls.pack(fill=tk.X, pady=(10, 0))

        
        def reset_manual_prompt():
            if messagebox.askyesno("Reset Prompt", 
                                  "Reset manual glossary prompt to default?"):
                self.manual_prompt_text.delete('1.0', tk.END)
                self.manual_prompt_text.insert('1.0', self.default_manual_glossary_prompt)
        
        tb.Button(prompt_controls, text="Reset to Default", 
                  command=reset_manual_prompt, bootstyle="warning").pack(side=tk.LEFT, padx=5)
        
        # Settings controls
        settings_frame = tk.LabelFrame(manual_container, text="Extraction Settings", padx=10, pady=10)
        settings_frame.pack(fill=tk.X, pady=(10, 0))
        
        settings_grid = tk.Frame(settings_frame)
        settings_grid.pack()
        
        tk.Label(settings_grid, text="Temperature:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.manual_temp_var = tk.StringVar(value=str(self.config.get('manual_glossary_temperature', 0.3)))
        tb.Entry(settings_grid, textvariable=self.manual_temp_var, width=10).grid(row=0, column=1, padx=5)
        
        tk.Label(settings_grid, text="Context Limit:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.manual_context_var = tk.StringVar(value=str(self.config.get('manual_context_limit', 3)))
        tb.Entry(settings_grid, textvariable=self.manual_context_var, width=10).grid(row=0, column=3, padx=5)
        
        # ===== AUTOMATIC GLOSSARY TAB =====
        auto_container = tk.Frame(auto_frame)
        auto_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Automatic glossary settings
        auto_settings_frame = tk.LabelFrame(auto_container, text="Extraction Settings", 
                                           padx=10, pady=10)
        auto_settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        settings_info = tk.Label(auto_settings_frame, 
                                text="These settings control the automatic glossary extraction during translation",
                                font=('TkDefaultFont', 9, 'italic'), fg='gray')
        settings_info.pack(anchor=tk.W, pady=(0, 10))
        
        # Settings are already in main GUI, just show current values
        current_settings = tk.Frame(auto_settings_frame)
        current_settings.pack(fill=tk.X)
        
        tk.Label(current_settings, text=f"Current Settings:", font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W)
        tk.Label(current_settings, text=f"• Min Frequency: {self.glossary_min_frequency_var.get()}").pack(anchor=tk.W, padx=20)
        tk.Label(current_settings, text=f"• Max Names: {self.glossary_max_names_var.get()}").pack(anchor=tk.W, padx=20)
        tk.Label(current_settings, text=f"• Max Titles: {self.glossary_max_titles_var.get()}").pack(anchor=tk.W, padx=20)
        tk.Label(current_settings, text=f"• Translation Batch Size: {self.glossary_batch_size_var.get()}").pack(anchor=tk.W, padx=20)
        
        tk.Label(current_settings, 
                 text="(Adjust these in 'Other Settings' from the main window)",
                 font=('TkDefaultFont', 9), fg='blue').pack(anchor=tk.W, pady=(5, 0))
        
        # Automatic prompt section
        auto_prompt_frame = tk.LabelFrame(auto_container, text="Extraction Prompt Template", 
                                         padx=10, pady=10)
        auto_prompt_frame.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        tk.Label(auto_prompt_frame, 
                 text="Available placeholders: {language}, {min_frequency}, {max_names}, {max_titles}",
                 font=('TkDefaultFont', 9), fg='blue').pack(anchor=tk.W, pady=(0, 5))
        
        # Automatic Prompt text area
        self.auto_prompt_text = scrolledtext.ScrolledText(auto_prompt_frame, height=12, wrap=tk.WORD)

        self.auto_prompt_text = scrolledtext.ScrolledText(
            auto_prompt_frame, 
            height=12, 
            wrap=tk.WORD,
            undo=True,
            autoseparators=True,
            maxundo=-1
        )
        self.auto_prompt_text.pack(fill=tk.BOTH, expand=True)
        self.auto_prompt_text.insert('1.0', self.auto_glossary_prompt)

        # Add undo/redo bindings
        self.auto_prompt_text.edit_reset()  # Reset undo stack after initial insert
        self._setup_text_undo_redo(self.auto_prompt_text)
        
        # Prompt controls
        auto_prompt_controls = tk.Frame(auto_container)
        auto_prompt_controls.pack(fill=tk.X, pady=(10, 0))
                
        def reset_auto_prompt():
            if messagebox.askyesno("Reset Prompt", 
                                  "Reset automatic glossary prompt to default?"):
                self.auto_prompt_text.delete('1.0', tk.END)
                self.auto_prompt_text.insert('1.0', self.default_auto_glossary_prompt)
        
        tb.Button(auto_prompt_controls, text="Reset to Default", 
                  command=reset_auto_prompt, bootstyle="warning").pack(side=tk.LEFT, padx=5)
        
        # ===== GLOSSARY EDITOR TAB =====
        self._setup_glossary_editor_tab(editor_frame)
        
        # ===== DIALOG CONTROLS =====
        control_frame = tk.Frame(manager)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def save_glossary_settings():
            """Save all glossary settings"""
            try:
                # Save manual field selections
                for field, var in self.manual_field_vars.items():
                    # Save all fields including original_name
                    self.config[f'manual_extract_{field}'] = var.get()
                
                # Save custom fields
                self.config['custom_glossary_fields'] = self.custom_glossary_fields
                
                # Save prompts
                self.manual_glossary_prompt = self.manual_prompt_text.get('1.0', tk.END).strip()
                self.auto_glossary_prompt = self.auto_prompt_text.get('1.0', tk.END).strip()
                self.config['manual_glossary_prompt'] = self.manual_glossary_prompt
                self.config['auto_glossary_prompt'] = self.auto_glossary_prompt
                
                
                # Save manual settings
                try:
                    self.config['manual_glossary_temperature'] = float(self.manual_temp_var.get())
                    self.config['manual_context_limit'] = int(self.manual_context_var.get())
                except ValueError:
                    messagebox.showwarning("Invalid Input", "Please enter valid numbers for temperature and context limit")
                    return
                
                # Update environment variables for immediate effect
                os.environ['GLOSSARY_SYSTEM_PROMPT'] = self.manual_glossary_prompt
                os.environ['AUTO_GLOSSARY_PROMPT'] = self.auto_glossary_prompt
                
                # Build fields string for manual glossary
                enabled_fields = []
                for field, var in self.manual_field_vars.items():
                    if var.get():
                        os.environ[f'GLOSSARY_EXTRACT_{field.upper()}'] = '1'
                        enabled_fields.append(field)
                    else:
                        os.environ[f'GLOSSARY_EXTRACT_{field.upper()}'] = '0'
                
                # Set custom fields
                if self.custom_glossary_fields:
                    os.environ['GLOSSARY_CUSTOM_FIELDS'] = json.dumps(self.custom_glossary_fields)
                
                # Save config to file
                with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                
                self.append_log("✅ Glossary settings saved successfully")
                messagebox.showinfo("Success", "Glossary settings saved!")
                manager.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save settings: {e}")
                self.append_log(f"❌ Failed to save glossary settings: {e}")
        
        def cancel():
            manager.destroy()
        
        # Create a centered button container
        button_container = tk.Frame(control_frame)
        button_container.pack(expand=True)  # This centers the container

        tb.Button(button_container, text="Save All Settings", command=save_glossary_settings, 
                  bootstyle="success", width=20).pack(side=tk.LEFT, padx=5)
        tb.Button(button_container, text="Cancel", command=lambda: [cleanup_scrolling(), manager.destroy()], 
                  bootstyle="secondary", width=20).pack(side=tk.LEFT, padx=5)
        
        # Auto-resize and center the dialog (up to 85% of screen height)
        self._auto_resize_dialog(manager, canvas, max_width_ratio=0.8, max_height_ratio=0.85)


    def _setup_glossary_editor_tab(self, parent):
        """Set up the glossary editor/trimmer tab"""
        container = tk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # File selection
        file_frame = tk.Frame(container)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(file_frame, text="Glossary File:").pack(side=tk.LEFT, padx=(0, 5))
        self.editor_file_var = tk.StringVar()
        tb.Entry(file_frame, textvariable=self.editor_file_var, state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        def browse_glossary():
            path = filedialog.askopenfilename(
                title="Select glossary.json",
                filetypes=[("JSON files", "*.json")]
            )
            if path:
                self.editor_file_var.set(path)
                load_glossary_for_editing()
        
        tb.Button(file_frame, text="Browse", command=browse_glossary, width=15).pack(side=tk.LEFT)
        
        # Glossary content area
        content_frame = tk.LabelFrame(container, text="Glossary Entries", padx=10, pady=10)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for glossary entries
        tree_frame = tk.Frame(content_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        
        # Create treeview
        self.glossary_tree = ttk.Treeview(tree_frame, 
                                         columns=('original', 'translated', 'type', 'frequency'),
                                         show='tree headings',
                                         yscrollcommand=vsb.set,
                                         xscrollcommand=hsb.set)
        
        vsb.config(command=self.glossary_tree.yview)
        hsb.config(command=self.glossary_tree.xview)
        
        # Configure columns
        self.glossary_tree.heading('#0', text='Index')
        self.glossary_tree.heading('original', text='Original')
        self.glossary_tree.heading('translated', text='Translation')
        self.glossary_tree.heading('type', text='Type')
        self.glossary_tree.heading('frequency', text='Count')
        
        self.glossary_tree.column('#0', width=60)
        self.glossary_tree.column('original', width=200)
        self.glossary_tree.column('translated', width=200)
        self.glossary_tree.column('type', width=100)
        self.glossary_tree.column('frequency', width=80)
        
        # Pack treeview and scrollbars
        self.glossary_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Editor controls
        editor_controls = tk.Frame(container)
        editor_controls.pack(fill=tk.X, pady=(10, 0))
        
        # Keep the current glossary data in memory
        self.current_glossary_data = None
        
        def load_glossary_for_editing():
            """Load glossary file into the tree view"""
            path = self.editor_file_var.get()
            if not path or not os.path.exists(path):
                messagebox.showerror("Error", "Please select a valid glossary file")
                return
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Clear existing items
                for item in self.glossary_tree.get_children():
                    self.glossary_tree.delete(item)
                
                # Handle different glossary formats
                entries = []
                
                if isinstance(data, dict):
                    if 'entries' in data:
                        # New format with metadata
                        self.current_glossary_data = data
                        entries = data['entries'].items()
                    else:
                        # Simple dict format
                        self.current_glossary_data = {'entries': data}
                        entries = data.items()
                elif isinstance(data, list):
                    # Manual glossary format
                    self.current_glossary_data = data
                    for idx, char in enumerate(data):
                        original = char.get('original_name', '')
                        translated = char.get('name', original)
                        char_type = 'character'
                        freq = idx + 1
                        
                        if original:
                            item = self.glossary_tree.insert('', 'end', text=str(idx + 1),
                                                           values=(original, translated, char_type, freq))
                    return
                
                # Load entries for dict format
                for idx, (original, translated) in enumerate(entries):
                    # Detect type
                    if any(h in original for h in ['님', '씨', 'さん', 'ちゃん', '-san', '-nim']):
                        entry_type = 'name'
                    elif any(t in original.lower() for t in ['king', 'queen', '왕', '여왕', '王']):
                        entry_type = 'title'
                    else:
                        entry_type = 'term'
                    
                    item = self.glossary_tree.insert('', 'end', text=str(idx + 1),
                                                   values=(original, translated, entry_type, ''))
                
                self.append_log(f"✅ Loaded {len(entries)} entries from glossary")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load glossary: {e}")
        
        def apply_trimming():
            """Apply the classic trimming logic"""
            if not self.current_glossary_data:
                messagebox.showerror("Error", "No glossary loaded")
                return
            
            # Create trimming dialog
            trim_dialog = tk.Toplevel(self.master)
            trim_dialog.title("Trim Glossary")
            trim_dialog.geometry("400x500")
            trim_dialog.transient(self.master)
            
            # Get values from the hidden trim entries in main GUI
            labels = [
                "Entries to keep (by frequency):",
                "Traits Trim Count:",
                "Title Keep (0=remove):",
                "Group Affiliation Trim Count:",
                "Refer-to-Others Trim Count:",
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
            for i, (lab, defval) in enumerate(zip(labels, defaults)):
                tb.Label(trim_dialog, text=lab).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
                e = tb.Entry(trim_dialog, width=10)
                e.insert(0, defval)
                e.grid(row=i, column=1, padx=5, pady=2)
                entries.append(e)
            
            def execute_trim():
                try:
                    # Get values
                    top_limit = int(entries[0].get())
                    traits_lim = int(entries[1].get())
                    title_lim = int(entries[2].get())
                    group_lim = int(entries[3].get())
                    refer_lim = int(entries[4].get())
                    loc_lim = int(entries[5].get())
                    
                    # Apply trimming based on format
                    if isinstance(self.current_glossary_data, list):
                        # Manual glossary format
                        trimmed = self.current_glossary_data[:top_limit]
                        for char in trimmed:
                            if title_lim <= 0:
                                char.pop('title', None)
                            if traits_lim <= 0:
                                char.pop('traits', None)
                            else:
                                t = char.get('traits', [])
                                if isinstance(t, list) and len(t) > traits_lim:
                                    char['traits'] = t[:-traits_lim]
                            # Apply other trims...
                        
                        self.current_glossary_data = trimmed
                    else:
                        # Dict format - just keep top N entries
                        if 'entries' in self.current_glossary_data:
                            entries_list = list(self.current_glossary_data['entries'].items())
                            self.current_glossary_data['entries'] = dict(entries_list[:top_limit])
                    
                    # Save and reload
                    path = self.editor_file_var.get()
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(self.current_glossary_data, f, ensure_ascii=False, indent=2)
                    
                    load_glossary_for_editing()
                    messagebox.showinfo("Success", f"Trimmed glossary to {top_limit} entries")
                    trim_dialog.destroy()
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Trimming failed: {e}")
            
            tb.Button(trim_dialog, text="Apply Trim", command=execute_trim, 
                     bootstyle="warning").grid(row=len(labels), column=0, columnspan=2, pady=10)
        
        def save_edited_glossary():
            """Save the current glossary"""
            if not self.current_glossary_data:
                messagebox.showerror("Error", "No glossary loaded")
                return
            
            path = self.editor_file_var.get()
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_glossary_data, f, ensure_ascii=False, indent=2)
                
                messagebox.showinfo("Success", "Glossary saved successfully")
                self.append_log(f"✅ Saved glossary to: {path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")
        
        # Control buttons
        tb.Button(editor_controls, text="Reload", command=load_glossary_for_editing,
                  bootstyle="info", width=15).pack(side=tk.LEFT, padx=5)
        tb.Button(editor_controls, text="Apply Classic Trim", command=apply_trimming,
                  bootstyle="warning", width=15).pack(side=tk.LEFT, padx=5)
        tb.Button(editor_controls, text="Aggregate Locations", 
                  command=lambda: self._aggregate_locations(load_glossary_for_editing),
                  bootstyle="info", width=15).pack(side=tk.LEFT, padx=5)
        tb.Button(editor_controls, text="Save Changes", command=save_edited_glossary,
                  bootstyle="success", width=15).pack(side=tk.LEFT, padx=5)

    def _aggregate_locations(self, reload_callback):
        """Aggregate all location entries into a single entry"""
        if not self.current_glossary_data:
            messagebox.showerror("Error", "No glossary loaded")
            return
        
        if isinstance(self.current_glossary_data, list):
            # Manual glossary format
            all_locs = []
            for char in self.current_glossary_data:
                locs = char.get('locations', [])
                if isinstance(locs, list):
                    all_locs.extend(locs)
                char.pop('locations', None)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_locs = []
            for loc in all_locs:
                if loc not in seen:
                    seen.add(loc)
                    unique_locs.append(loc)
            
            # Remove existing location summary
            self.current_glossary_data = [
                entry for entry in self.current_glossary_data 
                if entry.get('original_name') != "📍 Location Summary"
            ]
            
            # Add aggregated entry
            self.current_glossary_data.append({
                "original_name": "📍 Location Summary",
                "name": "Location Summary",
                "locations": unique_locs
            })
            
            # Save
            path = self.editor_file_var.get()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.current_glossary_data, f, ensure_ascii=False, indent=2)
            
            messagebox.showinfo("Success", f"Aggregated {len(unique_locs)} unique locations")
            reload_callback()
        else:
            messagebox.showinfo("Info", "Location aggregation only works with manual glossary format")
        
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
            ("Glossary Manager",    self.glossary_manager,            "secondary"),
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

                # Log batch translation status
                if self.batch_translation_var.get():
                    self.append_log(f"📦 Batch translation ENABLED - processing {self.batch_size_var.get()} chapters per API call")
                    self.append_log("   💡 This can improve speed but may reduce per-chapter customization")
                else:
                    self.append_log("📄 Standard translation mode - processing one chapter at a time")
                                
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
                    'ROLLING_SUMMARY_EXCHANGES': self.rolling_summary_exchanges_var.get(),
                    'ROLLING_SUMMARY_MODE': self.rolling_summary_mode_var.get(),
                    'ROLLING_SUMMARY_SYSTEM_PROMPT': self.rolling_summary_system_prompt,
                    'ROLLING_SUMMARY_USER_PROMPT': self.rolling_summary_user_prompt,
                    'PROFILE_NAME': self.lang_var.get().lower(),
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
                    'GLOSSARY_BATCH_SIZE': self.glossary_batch_size_var.get(),
                    # Image translation settings
                    'ENABLE_IMAGE_TRANSLATION': "1" if self.enable_image_translation_var.get() else "0",
                    'PROCESS_WEBNOVEL_IMAGES': "1" if self.process_webnovel_images_var.get() else "0",
                    'WEBNOVEL_MIN_HEIGHT': self.webnovel_min_height_var.get(),
                    'IMAGE_MAX_TOKENS': self.image_max_tokens_var.get(),
                    'MAX_IMAGES_PER_CHAPTER': self.max_images_per_chapter_var.get(),
                    'IMAGE_API_DELAY': '1.0',  # Delay between image API calls
                    'SAVE_IMAGE_TRANSLATIONS': '1',  # Save individual translations
                    'IMAGE_CHUNK_HEIGHT': self.image_chunk_height_var.get(),
                    'HIDE_IMAGE_TRANSLATION_LABEL': "1" if self.hide_image_translation_label_var.get() else "0",
                    'RETRY_TIMEOUT': "1" if self.retry_timeout_var.get() else "0",
                    'CHUNK_TIMEOUT': self.chunk_timeout_var.get(),
                    'BATCH_TRANSLATION': "1" if self.batch_translation_var.get() else "0",
                    'BATCH_SIZE': self.batch_size_var.get()

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
                
                # Log image translation status
                if self.enable_image_translation_var.get():
                    self.append_log("🖼️ Image translation ENABLED")
                    if self.model_var.get().lower() in ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-exp', 'gpt-4-turbo', 'gpt-4o']:
                        self.append_log(f"   ✅ Using vision-capable model: {self.model_var.get()}")
                        self.append_log(f"   • Max images per chapter: {self.max_images_per_chapter_var.get()}")
                        if self.process_webnovel_images_var.get():
                            self.append_log(f"   • Web novel images: Enabled (min height: {self.webnovel_min_height_var.get()}px)")
                    else:
                        self.append_log(f"   ⚠️ Model {self.model_var.get()} does not support vision")
                        self.append_log("   ⚠️ Image translation will be skipped")
                else:
                    self.append_log("🖼️ Image translation disabled")

                    
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
                    'GLOSSARY_TEMPERATURE': str(self.config.get('manual_glossary_temperature', 0.3)),
                    'GLOSSARY_CONTEXT_LIMIT': str(self.config.get('manual_context_limit', 3)),
                    'MODEL': self.model_var.get(),
                    'OPENAI_API_KEY': self.api_key_entry.get(),
                    'OPENAI_OR_Gemini_API_KEY': self.api_key_entry.get(),
                    'API_KEY': self.api_key_entry.get(),
                    'MAX_OUTPUT_TOKENS': str(self.max_output_tokens),
                    'GLOSSARY_SYSTEM_PROMPT': self.manual_glossary_prompt,
                    
                    # FIELD-SPECIFIC SETTINGS (ADD THESE):
                    'GLOSSARY_EXTRACT_ORIGINAL_NAME': '1' if self.config.get('manual_extract_original_name', True) else '0',
                    'GLOSSARY_EXTRACT_NAME': '1' if self.config.get('manual_extract_name', True) else '0',
                    'GLOSSARY_EXTRACT_GENDER': '1' if self.config.get('manual_extract_gender', True) else '0',
                    'GLOSSARY_EXTRACT_TITLE': '1' if self.config.get('manual_extract_title', True) else '0',
                    'GLOSSARY_EXTRACT_GROUP_AFFILIATION': '1' if self.config.get('manual_extract_group_affiliation', True) else '0',
                    'GLOSSARY_EXTRACT_TRAITS': '1' if self.config.get('manual_extract_traits', True) else '0',
                    'GLOSSARY_EXTRACT_HOW_THEY_REFER_TO_OTHERS': '1' if self.config.get('manual_extract_how_they_refer_to_others', True) else '0',
                    'GLOSSARY_EXTRACT_LOCATIONS': '1' if self.config.get('manual_extract_locations', True) else '0',
                }

                # Also add custom fields if any
                if self.custom_glossary_fields:
                    env_updates['GLOSSARY_CUSTOM_FIELDS'] = json.dumps(self.custom_glossary_fields)

                os.environ.update(env_updates)
                
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
                self.append_log("✅ EPUB Converter completed successfully!")
                
                # Look for the actual EPUB file that was created
                epub_files = [f for f in os.listdir(folder) if f.endswith('.epub')]
                if epub_files:
                    # Use the most recently created EPUB file
                    epub_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True)
                    out_file = os.path.join(folder, epub_files[0])
                    
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
        """Append message to log with special formatting for memory"""
        def _append():
            # Get current scroll position
            at_bottom = self.log_text.yview()[1] >= 0.98
            
            # Check if this is a memory-related message
            is_memory = any(keyword in message for keyword in ['[MEMORY]', '📝', 'rolling summary', 'memory'])
            
            # Add the message
            if is_memory:
                # Add with special formatting
                self.log_text.insert(tk.END, message + "\n", "memory")
                # Configure the tag if not already done
                if "memory" not in self.log_text.tag_names():
                    self.log_text.tag_config("memory", foreground="#4CAF50", font=('TkDefaultFont', 10, 'italic'))
            else:
                self.log_text.insert(tk.END, message + "\n")
            
            # Auto-scroll only if we were at the bottom
            if at_bottom:
                self.log_text.see(tk.END)
        
        if threading.current_thread() is threading.main_thread():
            _append()
        else:
            self.master.after(0, _append)
         
    def update_status_line(self, message, progress_percent=None):
        """Update a status line in the log (overwrites the last line if it's a status)"""
        def _update():
            # Get current content
            content = self.log_text.get("1.0", "end-1c")
            lines = content.split('\n')
            
            # Check if last line is a status line (starts with specific markers)
            status_markers = ['⏳', '📊', '✅', '❌', '🔄']
            is_status_line = False
            
            if lines and any(lines[-1].strip().startswith(marker) for marker in status_markers):
                is_status_line = True
            
            # Build new status message
            if progress_percent is not None:
                # Create a mini progress bar
                bar_width = 10
                filled = int(bar_width * progress_percent / 100)
                bar = "▓" * filled + "░" * (bar_width - filled)
                status_msg = f"⏳ {message} [{bar}] {progress_percent:.1f}%"
            else:
                status_msg = f"📊 {message}"
            
            # Update or append
            if is_status_line and lines[-1].strip().startswith(('⏳', '📊')):
                # Delete last line and replace
                start_pos = f"{len(lines)}.0"
                self.log_text.delete(f"{start_pos} linestart", "end")
                if len(lines) > 1:
                    self.log_text.insert("end", "\n" + status_msg)
                else:
                    self.log_text.insert("end", status_msg)
            else:
                # Just append
                if content and not content.endswith('\n'):
                    self.log_text.insert("end", "\n" + status_msg)
                else:
                    self.log_text.insert("end", status_msg + "\n")
            
            # Auto-scroll
            self.log_text.see("end")
        
        if threading.current_thread() is threading.main_thread():
            _update()
        else:
            self.master.after(0, _update)
        
    def append_chunk_progress(self, chunk_num, total_chunks, chunk_type="text", chapter_info="", 
                             overall_current=None, overall_total=None, extra_info=None):
        """Append chunk progress with enhanced visual indicator for all chunks"""
        progress_bar_width = 20
        
        # Calculate overall progress
        overall_progress = 0
        if overall_current is not None and overall_total is not None and overall_total > 0:
            overall_progress = overall_current / overall_total
        
        # Create overall progress bar
        overall_filled = int(progress_bar_width * overall_progress)
        overall_bar = "█" * overall_filled + "░" * (progress_bar_width - overall_filled)
        
        # For single chunks, show enhanced formatting
        if total_chunks == 1:
            icon = "📄" if chunk_type == "text" else "🖼️"
            
            # Create a more informative message
            msg_parts = [f"{icon} {chapter_info}"]
            
            # Add size info if provided
            if extra_info:
                msg_parts.append(f"[{extra_info}]")
            
            # Add overall progress
            if overall_current is not None and overall_total is not None:
                # Show both numeric and visual progress
                msg_parts.append(f"\n    Progress: [{overall_bar}] {overall_current}/{overall_total} ({overall_progress*100:.1f}%)")
                
                # Add ETA if we can calculate it
                if hasattr(self, '_chunk_start_times'):
                    if overall_current > 1:
                        # Calculate average time per chunk
                        elapsed = time.time() - self._translation_start_time
                        avg_time = elapsed / (overall_current - 1)
                        remaining = overall_total - overall_current + 1
                        eta_seconds = remaining * avg_time
                        
                        if eta_seconds < 60:
                            eta_str = f"{int(eta_seconds)}s"
                        elif eta_seconds < 3600:
                            eta_str = f"{int(eta_seconds/60)}m {int(eta_seconds%60)}s"
                        else:
                            hours = int(eta_seconds / 3600)
                            minutes = int((eta_seconds % 3600) / 60)
                            eta_str = f"{hours}h {minutes}m"
                        
                        msg_parts.append(f" - ETA: {eta_str}")
                else:
                    # Initialize timing
                    self._translation_start_time = time.time()
                    self._chunk_start_times = {}
            
            msg = " ".join(msg_parts)
            
        else:
            # Multi-chunk display with enhanced formatting
            chunk_progress = chunk_num / total_chunks if total_chunks > 0 else 0
            chunk_filled = int(progress_bar_width * chunk_progress)
            chunk_bar = "█" * chunk_filled + "░" * (progress_bar_width - chunk_filled)
            
            icon = "📄" if chunk_type == "text" else "🖼️"
            
            # Build the progress message with both chunk and overall progress
            msg_parts = [f"{icon} {chapter_info}"]
            msg_parts.append(f"\n    Chunk: [{chunk_bar}] {chunk_num}/{total_chunks} ({chunk_progress*100:.1f}%)")
            
            # Add overall progress on a new line
            if overall_current is not None and overall_total is not None:
                msg_parts.append(f"\n    Overall: [{overall_bar}] {overall_current}/{overall_total} ({overall_progress*100:.1f}%)")
            
            msg = "".join(msg_parts)
        
        # Track timing for current chunk
        if hasattr(self, '_chunk_start_times'):
            self._chunk_start_times[f"{chapter_info}_{chunk_num}"] = time.time()
        
        self.append_log(msg)



    def _block_editing(self, event):
        """Block editing in log text but allow selection and copying"""
        # Allow Ctrl+C for copy
        if event.state & 0x4 and event.keysym.lower() == 'c':  # Ctrl+C
            return None
        # Allow Ctrl+A for select all
        if event.state & 0x4 and event.keysym.lower() == 'a':  # Ctrl+A
            self.log_text.tag_add(tk.SEL, "1.0", tk.END)
            self.log_text.mark_set(tk.INSERT, "1.0")
            self.log_text.see(tk.INSERT)
            return "break"
        # Allow navigation keys
        if event.keysym in ['Left', 'Right', 'Up', 'Down', 'Home', 'End', 'Prior', 'Next']:
            return None
        # Allow selection with Shift
        if event.state & 0x1:  # Shift key
            return None
        # Block everything else
        return "break"

    def _show_context_menu(self, event):
        """Show context menu for log text"""
        try:
            # Create context menu
            context_menu = tk.Menu(self.master, tearoff=0)
            
            # Check if there's selected text
            try:
                self.log_text.selection_get()
                context_menu.add_command(label="Copy", command=self.copy_selection)
            except tk.TclError:
                # No selection
                context_menu.add_command(label="Copy", state="disabled")
            
            context_menu.add_separator()
            context_menu.add_command(label="Select All", command=self.select_all_log)
            
            # Show the menu
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            # Make sure to release the grab
            context_menu.grab_release()

    def copy_selection(self):
        """Copy selected text from log to clipboard"""
        try:
            text = self.log_text.selection_get()
            self.master.clipboard_clear()
            self.master.clipboard_append(text)
        except tk.TclError:
            # No selection
            pass

    def select_all_log(self):
        """Select all text in the log"""
        self.log_text.tag_add(tk.SEL, "1.0", tk.END)
        self.log_text.mark_set(tk.INSERT, "1.0")
        self.log_text.see(tk.INSERT)


    
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

    def configure_rolling_summary_prompts(self):
        """Configure rolling summary prompts"""
        dialog = tk.Toplevel(self.master)
        dialog.title("Configure Memory System Prompts")
        dialog.geometry("800x1050")
        dialog.transient(self.master)
        
        # Main container with padding
        main_frame = tk.Frame(dialog, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title and description
        tk.Label(main_frame, text="Memory System Configuration", 
                 font=('TkDefaultFont', 14, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        tk.Label(main_frame, 
                 text="Configure how the AI creates and maintains translation memory/context summaries.",
                 font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, pady=(0, 15))
        
        # System Prompt Section
        system_frame = tk.LabelFrame(main_frame, text="System Prompt (Role Definition)", padx=10, pady=10)
        system_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        tk.Label(system_frame, 
                 text="Defines the AI's role and behavior when creating summaries",
                 font=('TkDefaultFont', 9), fg='blue').pack(anchor=tk.W, pady=(0, 5))
        
        self.summary_system_text = scrolledtext.ScrolledText(
            system_frame, height=5, wrap=tk.WORD,
            undo=True, autoseparators=True, maxundo=-1
        )
        self.summary_system_text.pack(fill=tk.BOTH, expand=True)
        self.summary_system_text.insert('1.0', self.rolling_summary_system_prompt)
        self._setup_text_undo_redo(self.summary_system_text)
        
        # User Prompt Section
        user_frame = tk.LabelFrame(main_frame, text="User Prompt Template", padx=10, pady=10)
        user_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        tk.Label(user_frame, 
                 text="Template for summary requests. Use {translations} for content placeholder",
                 font=('TkDefaultFont', 9), fg='blue').pack(anchor=tk.W, pady=(0, 5))
        
        self.summary_user_text = scrolledtext.ScrolledText(
            user_frame, height=12, wrap=tk.WORD,
            undo=True, autoseparators=True, maxundo=-1
        )
        self.summary_user_text.pack(fill=tk.BOTH, expand=True)
        self.summary_user_text.insert('1.0', self.rolling_summary_user_prompt)
        self._setup_text_undo_redo(self.summary_user_text)
        
        # Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        def save_prompts():
            self.rolling_summary_system_prompt = self.summary_system_text.get('1.0', tk.END).strip()
            self.rolling_summary_user_prompt = self.summary_user_text.get('1.0', tk.END).strip()
            
            # Save to config
            self.config['rolling_summary_system_prompt'] = self.rolling_summary_system_prompt
            self.config['rolling_summary_user_prompt'] = self.rolling_summary_user_prompt
            
            # Update environment variables
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

    def open_other_settings(self):
        """Open the Other Settings dialog with all advanced options in a grid layout"""
        top = tk.Toplevel(self.master)
        top.title("Other Settings")
        top.geometry("900x1460")
        top.transient(self.master)
        #top.grab_set()
        
        # Store reference to prevent garbage collection issues
        self._settings_window = top
            
        # Main container
        main_container = tk.Frame(top)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Scrollable content area
        content_area = tk.Frame(main_container)
        content_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(content_area, bg='white')
        scrollbar = ttk.Scrollbar(content_area, orient="vertical", command=canvas.yview)
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
        
        # Configure grid columns for the scrollable frame
        scrollable_frame.grid_columnconfigure(0, weight=1, uniform="column")
        scrollable_frame.grid_columnconfigure(1, weight=1, uniform="column")
        
        # =================================================================
        # SECTION 1: CONTEXT MANAGEMENT (Top Left) - ENHANCED VERSION
        # =================================================================
        section1_frame = tk.LabelFrame(scrollable_frame, text="Context Management & Memory", padx=10, pady=10)
        section1_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=(10, 5))

        # Create inner frame to control content placement
        content_frame = tk.Frame(section1_frame)
        content_frame.pack(anchor=tk.NW, fill=tk.BOTH, expand=True)

        # Rolling Summary Enable
        tb.Checkbutton(content_frame, text="Use Rolling Summary (Memory)", 
                       variable=self.rolling_summary_var,
                       bootstyle="round-toggle").pack(anchor=tk.W)

        tk.Label(content_frame, 
                 text="AI-powered memory system that maintains story context",
                 font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))  # Increased bottom padding

        # Summary Settings Frame
        settings_frame = tk.Frame(content_frame)
        settings_frame.pack(anchor=tk.W, padx=20, fill=tk.X, pady=(5, 10))  # Increased padding

        # Row 1: Role and Mode - WITH PROPER SPACING
        row1 = tk.Frame(settings_frame)
        row1.pack(fill=tk.X, pady=(0, 10))  # Added bottom padding

        tk.Label(row1, text="Role:").pack(side=tk.LEFT, padx=(0, 5))  # Added right padding
        ttk.Combobox(row1, textvariable=self.summary_role_var,
                     values=["user", "system"], state="readonly", width=10).pack(side=tk.LEFT, padx=(0, 30))  # Added significant right padding

        tk.Label(row1, text="Mode:").pack(side=tk.LEFT, padx=(0, 5))  # Added right padding
        ttk.Combobox(row1, textvariable=self.rolling_summary_mode_var,
                     values=["append", "replace"], state="readonly", width=10).pack(side=tk.LEFT, padx=(0, 10))

        # Row 2: Exchanges to summarize - WITH BETTER SPACING
        row2 = tk.Frame(settings_frame)
        row2.pack(fill=tk.X, pady=(0, 10))  # Added bottom padding

        tk.Label(row2, text="Summarize last").pack(side=tk.LEFT, padx=(0, 5))
        tb.Entry(row2, width=5, textvariable=self.rolling_summary_exchanges_var).pack(side=tk.LEFT, padx=(0, 5))
        tk.Label(row2, text="exchanges").pack(side=tk.LEFT)

        # Configure Prompts Button with more spacing
        tb.Button(content_frame, text="⚙️ Configure Memory Prompts", 
                  command=self.configure_rolling_summary_prompts,
                  bootstyle="info-outline", width=30).pack(anchor=tk.W, padx=20, pady=(10, 10))  # Increased vertical padding

        # Add a separator line
        ttk.Separator(section1_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 10))  # Increased padding

        # Help text
        tk.Label(section1_frame, 
                 text="💡 Memory Mode:\n"
                 "• Append: Keeps adding summaries (longer context)\n"
                 "• Replace: Only keeps latest summary (concise)",
                 font=('TkDefaultFont', 11), fg='#666', justify=tk.LEFT).pack(anchor=tk.W, padx=5, pady=(0, 5))
        
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
                     
        # Retry Slow Chunks
        tb.Checkbutton(section2_frame, text="Auto-retry Slow Chunks", 
                       variable=self.retry_timeout_var,
                       bootstyle="round-toggle").pack(anchor=tk.W, pady=(10, 0))

        timeout_frame = tk.Frame(section2_frame)
        timeout_frame.pack(anchor=tk.W, padx=20, pady=(5, 0))
        tk.Label(timeout_frame, text="Timeout after").pack(side=tk.LEFT)
        tb.Entry(timeout_frame, width=6, textvariable=self.chunk_timeout_var).pack(side=tk.LEFT, padx=5)
        tk.Label(timeout_frame, text="seconds").pack(side=tk.LEFT)

        tk.Label(section2_frame, 
                 text="Retry chunks/images that take too long\n(reduces tokens for faster response)",
                 font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 5))
        
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
        # SECTION 4: GLOSSARY SETTINGS (Middle Right)
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
        # SECTION 5: AUTOMATIC GLOSSARY EXTRACTION CONTROLS (Bottom Left)
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

        # Comprehensive Chapter Extraction
        tb.Checkbutton(section6_frame, text="Comprehensive Chapter Extraction", 
                       variable=self.comprehensive_extraction_var,
                       bootstyle="round-toggle").pack(anchor=tk.W, pady=2)

        tk.Label(section6_frame, 
                 text="Extract ALL files (disable smart filtering)",
                 font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
        
        # =================================================================
        # SECTION 7: IMAGE TRANSLATION
        # =================================================================
        section7_frame = tk.LabelFrame(scrollable_frame, text="Image Translation", padx=10, pady=8)
        section7_frame.grid(row=3, column=0, sticky="nsew", padx=(10, 5), pady=5)
        
        # Enable checkbox with description on same line
        enable_frame = tk.Frame(section7_frame)
        enable_frame.pack(fill=tk.X, pady=(0, 5))
        
        tb.Checkbutton(enable_frame, text="Enable Image Translation", 
                       variable=self.enable_image_translation_var,
                       bootstyle="round-toggle").pack(side=tk.LEFT)
        
        # Web novel option on same line
        tb.Checkbutton(enable_frame, text="Include Long Images", 
                       variable=self.process_webnovel_images_var,
                       bootstyle="round-toggle").pack(side=tk.LEFT, padx=(20, 0))
        
        # Compact grid for numeric settings
        grid_frame = tk.Frame(section7_frame)
        grid_frame.pack(fill=tk.X, pady=5)

        # Configure columns for alignment
        grid_frame.columnconfigure(1, minsize=60)
        grid_frame.columnconfigure(3, minsize=60)

        # Row 1: Min height and Image Output Token Limit
        tk.Label(grid_frame, text="Min height:", font=('TkDefaultFont', 9)).grid(row=0, column=0, sticky=tk.W)
        tb.Entry(grid_frame, width=7, textvariable=self.webnovel_min_height_var).grid(row=0, column=1, padx=(2, 5))

        tk.Label(grid_frame, text="Image Output Token Limit:", font=('TkDefaultFont', 9)).grid(row=0, column=2, sticky=tk.W)
        tb.Entry(grid_frame, width=7, textvariable=self.image_max_tokens_var).grid(row=0, column=3, padx=2)

        # Row 2: Max per chapter and Chunk height
        tk.Label(grid_frame, text="Max/chapter:", font=('TkDefaultFont', 9)).grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        tb.Entry(grid_frame, width=7, textvariable=self.max_images_per_chapter_var).grid(row=1, column=1, padx=(2, 5), pady=(5, 0))

        tk.Label(grid_frame, text="Chunk height:", font=('TkDefaultFont', 9)).grid(row=1, column=2, sticky=tk.W, pady=(5, 0))
        tb.Entry(grid_frame, width=7, textvariable=self.image_chunk_height_var).grid(row=1, column=3, padx=2, pady=(5, 0))

        # Help text
        tk.Label(section7_frame, 
                 text="Vision models: Gemini 1.5-pro/flash, GPT-4V/4o",
                 font=('TkDefaultFont', 9), fg='gray').pack(anchor=tk.W, pady=(2, 0))

        tk.Label(section7_frame, 
                 text="Chunk height: Pixels per chunk for tall images",
                 font=('TkDefaultFont', 9), fg='gray').pack(anchor=tk.W, pady=(2, 0))        

        tb.Checkbutton(section7_frame, text="Hide labels and remove OCR images", 
                       variable=self.hide_image_translation_label_var,
                       bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
        
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
                self.config['retry_timeout'] = self.retry_timeout_var.get()
                self.config['chunk_timeout'] = int(self.chunk_timeout_var.get())
                
                # Prompt Management
                self.config['reinforcement_frequency'] = int(self.reinforcement_freq_var.get())
                self.config['disable_system_prompt'] = self.disable_system_prompt_var.get()
                
                # Glossary Settings
                self.config['disable_auto_glossary'] = self.disable_auto_glossary_var.get()
                self.config['disable_glossary_translation'] = self.disable_glossary_translation_var.get()
                self.config['append_glossary'] = self.append_glossary_var.get()
                
                # Glossary Extraction Controls
                self.config['glossary_min_frequency'] = int(self.glossary_min_frequency_var.get())
                self.config['glossary_max_names'] = int(self.glossary_max_names_var.get())
                self.config['glossary_max_titles'] = int(self.glossary_max_titles_var.get()) 
                self.config['glossary_batch_size'] = int(self.glossary_batch_size_var.get())
                
                # Processing Options
                self.config['emergency_paragraph_restore'] = self.emergency_restore_var.get()
                self.config['reset_failed_chapters'] = self.reset_failed_chapters_var.get()
                self.config['comprehensive_extraction'] = self.comprehensive_extraction_var.get()
                
                # Image Translation Settings
                self.config['enable_image_translation'] = self.enable_image_translation_var.get()
                self.config['process_webnovel_images'] = self.process_webnovel_images_var.get()
                self.config['webnovel_min_height'] = int(self.webnovel_min_height_var.get())
                self.config['image_max_tokens'] = int(self.image_max_tokens_var.get())
                self.config['max_images_per_chapter'] = int(self.max_images_per_chapter_var.get())
                self.config['image_chunk_height'] = int(self.image_chunk_height_var.get())
                self.config['hide_image_translation_label'] = self.hide_image_translation_label_var.get()
                self.config['use_rolling_summary'] = self.rolling_summary_var.get()
                self.config['summary_role'] = self.summary_role_var.get()
                self.config['rolling_summary_exchanges'] = int(self.rolling_summary_exchanges_var.get())
                self.config['rolling_summary_mode'] = self.rolling_summary_mode_var.get()
                
                # Set environment variables for immediate effect
                os.environ.update({
                    "USE_ROLLING_SUMMARY": "1" if self.rolling_summary_var.get() else "0",
                    "SUMMARY_ROLE": self.summary_role_var.get(),
                    "ROLLING_SUMMARY_EXCHANGES": self.rolling_summary_exchanges_var.get(),
                    "ROLLING_SUMMARY_MODE": self.rolling_summary_mode_var.get(),
                    "ROLLING_SUMMARY_SYSTEM_PROMPT": self.rolling_summary_system_prompt,
                    "ROLLING_SUMMARY_USER_PROMPT": self.rolling_summary_user_prompt,
                    "SUMMARY_ROLE": self.summary_role_var.get(),
                    "RETRY_TRUNCATED": "1" if self.retry_truncated_var.get() else "0",
                    "MAX_RETRY_TOKENS": self.max_retry_tokens_var.get(),
                    "RETRY_DUPLICATE_BODIES": "1" if self.retry_duplicate_var.get() else "0",
                    "DUPLICATE_LOOKBACK_CHAPTERS": self.duplicate_lookback_var.get(),
                    "RETRY_TIMEOUT": "1" if self.retry_timeout_var.get() else "0",
                    "CHUNK_TIMEOUT": self.chunk_timeout_var.get(),
                    "REINFORCEMENT_FREQUENCY": self.reinforcement_freq_var.get(),
                    "DISABLE_SYSTEM_PROMPT": "1" if self.disable_system_prompt_var.get() else "0",
                    "DISABLE_AUTO_GLOSSARY": "1" if self.disable_auto_glossary_var.get() else "0",
                    "DISABLE_GLOSSARY_TRANSLATION": "1" if self.disable_glossary_translation_var.get() else "0",
                    "APPEND_GLOSSARY": "1" if self.append_glossary_var.get() else "0",
                    "EMERGENCY_PARAGRAPH_RESTORE": "1" if self.emergency_restore_var.get() else "0",
                    "RESET_FAILED_CHAPTERS": "1" if self.reset_failed_chapters_var.get() else "0",
                    "COMPREHENSIVE_EXTRACTION": "1" if self.comprehensive_extraction_var.get() else "0",
                    "ENABLE_IMAGE_TRANSLATION": "1" if self.enable_image_translation_var.get() else "0",
                    "PROCESS_WEBNOVEL_IMAGES": "1" if self.process_webnovel_images_var.get() else "0",
                    "WEBNOVEL_MIN_HEIGHT": self.webnovel_min_height_var.get(),
                    "IMAGE_MAX_TOKENS": self.image_max_tokens_var.get(),
                    "MAX_IMAGES_PER_CHAPTER": self.max_images_per_chapter_var.get(),
                    "IMAGE_CHUNK_HEIGHT": self.image_chunk_height_var.get(),
                    "HIDE_IMAGE_TRANSLATION_LABEL": "1" if self.hide_image_translation_label_var.get() else "0",
                    "GLOSSARY_MIN_FREQUENCY": self.glossary_min_frequency_var.get(),
                    "GLOSSARY_MAX_NAMES": self.glossary_max_names_var.get(),
                    "GLOSSARY_MAX_TITLES": self.glossary_max_titles_var.get(),
                    "GLOSSARY_BATCH_SIZE": self.glossary_batch_size_var.get()

                    
                })
                
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
                
        # Auto-resize and center the dialog (up to 80% of screen height)
        self._auto_resize_dialog(top, canvas, max_width_ratio=0.7, max_height_ratio=0.8)
        # =================================================================
        # SAVE & CLOSE BUTTONS (Right side of row 3, next to Image Translation)
        # =================================================================
        
        # Save button frame in the empty space next to Image Translation
        button_frame = tk.LabelFrame(scrollable_frame, text="Actions", padx=10, pady=10)
        button_frame.grid(row=3, column=1, sticky="nsew", padx=(5, 10), pady=5)
        
        # Center buttons vertically
        button_container = tk.Frame(button_frame)
        button_container.pack(expand=True, fill='both')
        
        tb.Button(button_container, text="💾 Save Settings", command=save_and_close, 
                  bootstyle="success", width=20).pack(pady=5)
        
        tb.Button(button_container, text="❌ Cancel", command=lambda: [cleanup_bindings(), top.destroy()], 
                  bootstyle="secondary", width=20).pack(pady=5)
                        
        # =================================================================
        # MOUSE WHEEL SCROLLING SUPPORT
        # ================================================================
        
        # Replace the existing mouse wheel scrolling section with:
        cleanup_bindings = self._setup_dialog_scrolling(top, canvas)

        # Update the existing protocol and button handlers to use cleanup_bindings
        top.protocol("WM_DELETE_WINDOW", lambda: [cleanup_bindings(), top.destroy()])
        



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
            self.config['enable_image_translation'] = self.enable_image_translation_var.get()
            self.config['process_webnovel_images'] = self.process_webnovel_images_var.get()
            self.config['webnovel_min_height'] = int(self.webnovel_min_height_var.get())
            self.config['image_max_tokens'] = int(self.image_max_tokens_var.get())
            self.config['max_images_per_chapter'] = int(self.max_images_per_chapter_var.get())
            self.config['batch_translation'] = self.batch_translation_var.get()
            self.config['batch_size'] = int(self.batch_size_var.get())

            
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



    def log_debug(self, message):
        self.append_log(f"[DEBUG] {message}")

if __name__ == "__main__":
    import time  # Add this import
    
    print("🚀 Starting Glossarion v2.1.0...")
    
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
