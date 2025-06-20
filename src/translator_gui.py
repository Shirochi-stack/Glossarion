# Standard Library
import io, json, logging, math, os, shutil, sys, threading, time, re
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk

# Third-Party
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from splash_utils import SplashManager

if getattr(sys, 'frozen', False):
    try:
        import multiprocessing
        multiprocessing.freeze_support()
    except: pass

# Deferred modules
translation_main = translation_stop_flag = translation_stop_check = None
glossary_main = glossary_stop_flag = glossary_stop_check = None
fallback_compile_epub = scan_html_folder = None

# Constants
CONFIG_FILE = "config.json"
BASE_WIDTH, BASE_HEIGHT = 1550, 1000

def load_application_icon(window, base_dir):
    """Load application icon with fallback handling"""
    ico_path = os.path.join(base_dir, 'Halgakos.ico')
    if os.path.isfile(ico_path):
        try:
            window.iconbitmap(ico_path)
        except Exception as e:
            logging.warning(f"Could not set window icon: {e}")
    try:
        from PIL import Image, ImageTk
        if os.path.isfile(ico_path):
            icon_image = Image.open(ico_path)
            if icon_image.mode != 'RGBA':
                icon_image = icon_image.convert('RGBA')
            icon_photo = ImageTk.PhotoImage(icon_image)
            window.iconphoto(False, icon_photo)
            return icon_photo
    except (ImportError, Exception) as e:
        logging.warning(f"Could not load icon image: {e}")
    return None

class TranslatorGUI:
    def __init__(self, master):
        master.configure(bg='#2b2b2b')
        self.master = master

        self.max_output_tokens = 8192
        self.proc = self.glossary_proc = None
        master.title("Glossarion v2.7.2 — The AI Hunter Unleashed!!")
        master.geometry(f"{BASE_WIDTH}x{BASE_HEIGHT}")
        master.minsize(1600, 1000)
        master.bind('<F11>', self.toggle_fullscreen)
        master.bind('<Escape>', lambda e: master.attributes('-fullscreen', False))
        self.payloads_dir = os.path.join(os.getcwd(), "Payloads")
        
        self._modules_loaded = self._modules_loading = False
        self.stop_requested = False
        self.translation_thread = self.glossary_thread = self.qa_thread = self.epub_thread = None
        
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        self.base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        
        # Load icon
        ico_path = os.path.join(self.base_dir, 'Halgakos.ico')
        if os.path.isfile(ico_path):
            try: master.iconbitmap(ico_path)
            except: pass
        
        self.logo_img = None
        try:
            from PIL import Image, ImageTk
            self.logo_img = ImageTk.PhotoImage(Image.open(ico_path)) if os.path.isfile(ico_path) else None
            if self.logo_img: master.iconphoto(False, self.logo_img)
        except Exception as e:
            logging.error(f"Failed to load logo: {e}")
        
        # Load config
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except: self.config = {}
        
        self.max_output_tokens = self.config.get('max_output_tokens', self.max_output_tokens)
        
        # Default prompts
        self.default_prompts = {
            "korean": "You are a professional Korean to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n- Use an easy to read and grammatically accurate comedy translation style.\n- Retain honorifics like -nim, -ssi.\n- Preserve original intent, and speech tone.\n- retain onomatopoeia in Romaji.\n- Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title> ,<h1>, <h2>, <p>, <br>, <div>, etc.",
            "japanese": "You are a professional Japanese to English novel translator, you must strictly output only English text and HTML tags text while following these rules:\n- Use an easy to read and grammatically accurate comedy translation style.\n- Retain honorifics like -san, -sama, -chan, -kun.\n- Preserve original intent, and speech tone.\n- retain onomatopoeia in Romaji.\n- retain onomatopoeia in Romaji.\n- Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title> ,<h1>, <h2>, <p>, <br>, <div>, etc.",
            "chinese": "You are a professional Chinese to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n- Use an easy to read and grammatically accurate comedy translation style.\n- Preserve original intent, and speech tone.\n- retain onomatopoeia in Romaji.\n- Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title> ,<h1>, <h2>, <p>, <br>, <div>, etc.",
            "korean_OCR": "You are a professional Korean to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n- Use an easy to read and grammatically accurate comedy translation style.\n- Retain honorifics like -nim, -ssi.\n- Preserve original intent, and speech tone.\n- retain onomatopoeia in Romaji.\n- Add HTML tags for proper formatting as expected of a novel.",
            "japanese_OCR": "You are a professional Japanese to English novel translator, you must strictly output only English text and HTML tags text while following these rules:\n- Use an easy to read and grammatically accurate comedy translation style.\n- Retain honorifics like -san, -sama, -chan, -kun.\n- Preserve original intent, and speech tone.\n- retain onomatopoeia in Romaji.\n- Add HTML tags for proper formatting as expected of a novel.",
            "chinese_OCR": "You are a professional Chinese to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n- Use an easy to read and grammatically accurate comedy translation style.\n- Preserve original intent, and speech tone.\n- retain onomatopoeia in Romaji.\n- Add HTML tags for proper formatting as expected of a novel.",
            "Original": "Return text and html tags exactly as they appear on the source."
        }
        
        # Define default prompts as class attributes
        self._init_default_prompts()
        self._init_variables()
        self._setup_gui()
    
    def _init_default_prompts(self):
        """Initialize all default prompt templates"""
        self.default_manual_glossary_prompt = """Output exactly a JSON array of objects and nothing else.
        You are a glossary extractor for Korean, Japanese, or Chinese novels.
        - Extract character information (e.g., name, traits), locations (countries, regions, cities), and translate them into English (romanization or equivalent).
        - Romanize all untranslated honorifics (e.g., 님 to '-nim', さん to '-san').
        - all output must be in english, unless specified otherwise
        For each character, provide JSON fields:
        {fields}
        Sort by appearance order; respond with a JSON array only.

        Text:
        {chapter_text}"""
        
        self.default_auto_glossary_prompt = """You are extracting a targeted glossary from a {language} novel.
        Focus on identifying:
        1. Character names with their honorifics
        2. Important titles and ranks
        3. Frequently mentioned terms (min frequency: {min_frequency})

        Extract up to {max_names} character names and {max_titles} titles.
        Prioritize names that appear with honorifics or in important contexts.
        Return the glossary in a simple key-value format."""
        
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
    
    def _init_variables(self):
        """Initialize all configuration variables"""
        # Load saved prompts
        self.manual_glossary_prompt = self.config.get('manual_glossary_prompt', self.default_manual_glossary_prompt)
        self.auto_glossary_prompt = self.config.get('auto_glossary_prompt', self.default_auto_glossary_prompt)
        self.rolling_summary_system_prompt = self.config.get('rolling_summary_system_prompt', self.default_rolling_summary_system_prompt)
        self.rolling_summary_user_prompt = self.config.get('rolling_summary_user_prompt', self.default_rolling_summary_user_prompt)
        
        self.custom_glossary_fields = self.config.get('custom_glossary_fields', [])
        self.token_limit_disabled = self.config.get('token_limit_disabled', False)
        
        # Create all config variables with helper
        def create_var(var_type, key, default):
            return var_type(value=self.config.get(key, default))
        
        # Boolean variables
        bool_vars = [
            ('rolling_summary_var', 'use_rolling_summary', False),
            ('translation_history_rolling_var', 'translation_history_rolling', False),
            ('glossary_history_rolling_var', 'glossary_history_rolling', False),
            ('translate_book_title_var', 'translate_book_title', True),
            ('enable_auto_glossary_var', 'enable_auto_glossary', False),
            ('append_glossary_var', 'append_glossary', False),
            ('reset_failed_chapters_var', 'reset_failed_chapters', True),
            ('retry_truncated_var', 'retry_truncated', True),
            ('retry_duplicate_var', 'retry_duplicate_bodies', True),
            ('enable_image_translation_var', 'enable_image_translation', False),
            ('process_webnovel_images_var', 'process_webnovel_images', True),
            ('comprehensive_extraction_var', 'comprehensive_extraction', False),
            ('hide_image_translation_label_var', 'hide_image_translation_label', True),
            ('retry_timeout_var', 'retry_timeout', True),
            ('batch_translation_var', 'batch_translation', False),
            ('disable_epub_gallery_var', 'disable_epub_gallery', False),
            ('disable_zero_detection_var', 'disable_zero_detection', False),
            ('emergency_restore_var', 'emergency_paragraph_restore', True),
            ('contextual_var', 'contextual', True),
            ('REMOVE_AI_ARTIFACTS_var', 'REMOVE_AI_ARTIFACTS', False)
        ]
        
        for var_name, key, default in bool_vars:
            setattr(self, var_name, create_var(tk.BooleanVar, key, default))
        
        # String variables
        str_vars = [
            ('summary_role_var', 'summary_role', 'user'),
            ('rolling_summary_exchanges_var', 'rolling_summary_exchanges', '5'),
            ('rolling_summary_mode_var', 'rolling_summary_mode', 'append'),
            ('reinforcement_freq_var', 'reinforcement_frequency', '10'),
            ('max_retry_tokens_var', 'max_retry_tokens', '16384'),
            ('duplicate_lookback_var', 'duplicate_lookback_chapters', '5'),
            ('glossary_min_frequency_var', 'glossary_min_frequency', '2'),
            ('glossary_max_names_var', 'glossary_max_names', '50'),
            ('glossary_max_titles_var', 'glossary_max_titles', '30'),
            ('glossary_batch_size_var', 'glossary_batch_size', '50'),
            ('webnovel_min_height_var', 'webnovel_min_height', '1000'),
            ('image_max_tokens_var', 'image_max_tokens', '16384'),
            ('max_images_per_chapter_var', 'max_images_per_chapter', '1'),
            ('image_chunk_height_var', 'image_chunk_height', '1500'),
            ('chunk_timeout_var', 'chunk_timeout', '900'),
            ('batch_size_var', 'batch_size', '3')
        ]
        
        for var_name, key, default in str_vars:
            setattr(self, var_name, create_var(tk.StringVar, key, str(default)))
        
        self.book_title_prompt = self.config.get('book_title_prompt', 
            "Translate this book title to English while retaining any acronyms:")
        
        # Profiles
        self.prompt_profiles = self.config.get('prompt_profiles', self.default_prompts.copy())
        active = self.config.get('active_profile', next(iter(self.prompt_profiles)))
        self.profile_var = tk.StringVar(value=active)
        self.lang_var = self.profile_var

    def _setup_gui(self):
        """Initialize all GUI components"""
        self.frame = tb.Frame(self.master, padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid
        for i in range(5):
            self.frame.grid_columnconfigure(i, weight=1 if i in [1, 3] else 0)
        for r in range(12):
            self.frame.grid_rowconfigure(r, weight=1 if r in [9, 10] else 0, minsize=200 if r == 9 else 150 if r == 10 else 0)
        
        # Create UI elements using helper methods
        self._create_file_section()
        self._create_model_section()
        self._create_language_section()
        self._create_settings_section()
        self._create_api_section()
        self._create_prompt_section()
        self._create_log_section()
        self._make_bottom_toolbar()
        
        # Apply token limit state
        if self.token_limit_disabled:
            self.token_limit_entry.config(state=tk.DISABLED)
            self.toggle_token_btn.config(text="Enable Input Token Limit", bootstyle="success-outline")
        
        self.on_profile_select()
        self.append_log("🚀 Glossarion v2.7.2 - Ready to use!")
        self.append_log("💡 Click any function button to load modules automatically")
    
    def _create_file_section(self):
        """Create file selection section"""
        tb.Label(self.frame, text="Input File:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.entry_epub = tb.Entry(self.frame, width=50)
        self.entry_epub.grid(row=0, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        tb.Button(self.frame, text="Browse", command=self.browse_file, width=12).grid(row=0, column=4, sticky=tk.EW, padx=5, pady=5)
    
    def _create_model_section(self):
        """Create model selection section"""
        tb.Label(self.frame, text="Model:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        default_model = self.config.get('model', 'gemini-2.0-flash')
        self.model_var = tk.StringVar(value=default_model)
        models = ["gpt-4o","gpt-4o-mini","gpt-4-turbo","gpt-4.1-nano","gpt-4.1-mini","gpt-4.1",
                  "gpt-3.5-turbo","o4-mini","gemini-1.5-pro","gemini-1.5-flash", "gemini-2.0-flash",
                  "gemini-2.0-flash-exp","gemini-2.5-flash-preview-05-20","gemini-2.5-pro-preview-06-05",
                  "deepseek-chat","claude-3-5-sonnet-20241022","claude-3-7-sonnet-20250219"]
        tb.Combobox(self.frame, textvariable=self.model_var, values=models, state="normal").grid(
            row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
    
    def _create_language_section(self):
        """Create language/profile section"""
        tb.Label(self.frame, text="Language:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.profile_menu = tb.Combobox(self.frame, textvariable=self.profile_var,
                                       values=list(self.prompt_profiles.keys()), state="normal")
        self.profile_menu.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        self.profile_menu.bind("<<ComboboxSelected>>", self.on_profile_select)
        self.profile_menu.bind("<Return>", self.on_profile_select)
        tb.Button(self.frame, text="Save Language", command=self.save_profile, width=14).grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        tb.Button(self.frame, text="Delete Language", command=self.delete_profile, width=14).grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)
    
    def _create_settings_section(self):
        """Create all settings controls"""
        # Contextual
        tb.Checkbutton(self.frame, text="Contextual Translation", variable=self.contextual_var).grid(
            row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # API delay
        tb.Label(self.frame, text="API call delay (s):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.delay_entry = tb.Entry(self.frame, width=8)
        self.delay_entry.insert(0, str(self.config.get('delay', 2)))
        self.delay_entry.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Chapter Range
        tb.Label(self.frame, text="Chapter range (e.g., 5-10):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.chapter_range_entry = tb.Entry(self.frame, width=12)
        self.chapter_range_entry.insert(0, self.config.get('chapter_range', ''))
        self.chapter_range_entry.grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Token limit
        tb.Label(self.frame, text="Input Token limit:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.token_limit_entry = tb.Entry(self.frame, width=8)
        self.token_limit_entry.insert(0, str(self.config.get('token_limit', 50000)))
        self.token_limit_entry.grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.toggle_token_btn = tb.Button(self.frame, text="Disable Input Token Limit",
                                         command=self.toggle_token_limit, bootstyle="danger-outline", width=21)
        self.toggle_token_btn.grid(row=7, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Translation settings (right side)
        tb.Label(self.frame, text="Temperature:").grid(row=4, column=2, sticky=tk.W, padx=5, pady=5)
        self.trans_temp = tb.Entry(self.frame, width=6)
        self.trans_temp.insert(0, str(self.config.get('translation_temperature', 0.3)))
        self.trans_temp.grid(row=4, column=3, sticky=tk.W, padx=5, pady=5)
        
        tb.Label(self.frame, text="Transl. Hist. Limit:").grid(row=5, column=2, sticky=tk.W, padx=5, pady=5)
        self.trans_history = tb.Entry(self.frame, width=6)
        self.trans_history.insert(0, str(self.config.get('translation_history_limit', 3)))
        self.trans_history.grid(row=5, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Batch Translation
        tb.Checkbutton(self.frame, text="Batch Translation", variable=self.batch_translation_var,
                      bootstyle="round-toggle").grid(row=6, column=2, sticky=tk.W, padx=5, pady=5)
        self.batch_size_entry = tb.Entry(self.frame, width=6, textvariable=self.batch_size_var)
        self.batch_size_entry.grid(row=6, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Set batch entry state
        self.batch_size_entry.config(state=tk.NORMAL if self.batch_translation_var.get() else tk.DISABLED)
        self.batch_translation_var.trace('w', lambda *args: self.batch_size_entry.config(
            state=tk.NORMAL if self.batch_translation_var.get() else tk.DISABLED))
        
        # Rolling History
        tb.Checkbutton(self.frame, text="Rolling History Window", variable=self.translation_history_rolling_var,
                      bootstyle="round-toggle").grid(row=7, column=2, sticky=tk.W, padx=5, pady=5)
        tk.Label(self.frame, text="(Keep recent history instead of purging)",
                font=('TkDefaultFont', 11), fg='gray').grid(row=7, column=3, sticky=tk.W, padx=5, pady=5)
                
        #detection mode
        self.duplicate_detection_mode_var = tk.StringVar(value=self.config.get('duplicate_detection_mode', 'basic'))
        self.ai_hunter_threshold_var = tk.StringVar(value=str(self.config.get('ai_hunter_threshold', 75)))


        

        
        # Hidden entries for compatibility
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
    
    def _create_api_section(self):
        """Create API key section"""
        tb.Label(self.frame, text="OpenAI/Gemini/... API Key:").grid(row=8, column=0, sticky=tk.W, padx=5, pady=5)
        self.api_key_entry = tb.Entry(self.frame, show='*')
        self.api_key_entry.grid(row=8, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        initial_key = self.config.get('api_key', '')
        if initial_key:
            self.api_key_entry.insert(0, initial_key)
        tb.Button(self.frame, text="Show", command=self.toggle_api_visibility, width=12).grid(row=8, column=4, sticky=tk.EW, padx=5, pady=5)
        
        # Other Settings button
        tb.Button(self.frame, text="⚙️  Other Setting", command=self.open_other_settings,
                 bootstyle="info-outline", width=15).grid(row=7, column=4, sticky=tk.EW, padx=5, pady=5)
        
        # Remove AI Artifacts
        tb.Checkbutton(self.frame, text="Remove AI Artifacts", variable=self.REMOVE_AI_ARTIFACTS_var,
                      bootstyle="round-toggle").grid(row=7, column=0, columnspan=5, sticky=tk.W, padx=5, pady=(0,5))
    
    def _create_prompt_section(self):
        """Create system prompt section"""
        tb.Label(self.frame, text="System Prompt:").grid(row=9, column=0, sticky=tk.NW, padx=5, pady=5)
        self.prompt_text = tk.Text(self.frame, height=5, width=60, wrap='word', undo=True, autoseparators=True, maxundo=-1)
        self._setup_text_undo_redo(self.prompt_text)
        self.prompt_text.grid(row=9, column=1, columnspan=3, sticky=tk.NSEW, padx=5, pady=5)
        
        # Output Token Limit button
        self.output_btn = tb.Button(self.frame, text=f"Output Token Limit: {self.max_output_tokens}",
                                   command=self.prompt_custom_token_limit, bootstyle="info", width=22)
        self.output_btn.grid(row=9, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Run Translation button
        self.run_button = tb.Button(self.frame, text="Run Translation", command=self.run_translation_thread,
                                   bootstyle="success", width=14)
        self.run_button.grid(row=9, column=4, sticky=tk.N+tk.S+tk.EW, padx=5, pady=5)
        self.master.update_idletasks()
        self.run_base_w = self.run_button.winfo_width()
        self.run_base_h = self.run_button.winfo_height()
        self.master.bind('<Configure>', self.on_resize)
    
    def _create_log_section(self):
        """Create log text area"""
        self.log_text = scrolledtext.ScrolledText(self.frame, wrap=tk.WORD)
        self.log_text.grid(row=10, column=0, columnspan=5, sticky=tk.NSEW, padx=5, pady=5)
        self.log_text.bind("<Key>", self._block_editing)
        self.log_text.bind("<Button-3>", self._show_context_menu)
        if sys.platform == "darwin":
            self.log_text.bind("<Button-2>", self._show_context_menu)

    def _lazy_load_modules(self, splash_callback=None):
            """Load heavy modules only when needed - Enhanced with thread safety, retry logic, and progress tracking"""
            # Quick return if already loaded (unchanged for compatibility)
            if self._modules_loaded:
                return True
                
            # Enhanced thread safety with timeout protection
            if self._modules_loading:
                timeout_start = time.time()
                timeout_duration = 30.0  # 30 second timeout to prevent infinite waiting
                
                while self._modules_loading and not self._modules_loaded:
                    # Check for timeout to prevent infinite loops
                    if time.time() - timeout_start > timeout_duration:
                        self.append_log("⚠️ Module loading timeout - resetting loading state")
                        self._modules_loading = False
                        break
                    time.sleep(0.1)
                return self._modules_loaded
            
            # Set loading flag with enhanced error handling
            self._modules_loading = True
            loading_start_time = time.time()
            
            try:
                if splash_callback:
                    splash_callback("Loading translation modules...")
                
                # Global module imports (unchanged for compatibility)
                global translation_main, translation_stop_flag, translation_stop_check
                global glossary_main, glossary_stop_flag, glossary_stop_check
                global fallback_compile_epub, scan_html_folder
                
                # Enhanced module configuration with validation and retry info
                modules = [
                    {
                        'name': 'TransateKRtoEN',
                        'display_name': 'translation engine',
                        'imports': ['main', 'set_stop_flag', 'is_stop_requested'],
                        'global_vars': ['translation_main', 'translation_stop_flag', 'translation_stop_check'],
                        'critical': True,
                        'retry_count': 0,
                        'max_retries': 2
                    },
                    {
                        'name': 'extract_glossary_from_epub',
                        'display_name': 'glossary extractor', 
                        'imports': ['main', 'set_stop_flag', 'is_stop_requested'],
                        'global_vars': ['glossary_main', 'glossary_stop_flag', 'glossary_stop_check'],
                        'critical': True,
                        'retry_count': 0,
                        'max_retries': 2
                    },
                    {
                        'name': 'epub_converter',
                        'display_name': 'EPUB converter',
                        'imports': ['fallback_compile_epub'],
                        'global_vars': ['fallback_compile_epub'],
                        'critical': False,
                        'retry_count': 0,
                        'max_retries': 1
                    },
                    {
                        'name': 'scan_html_folder', 
                        'display_name': 'QA scanner',
                        'imports': ['scan_html_folder'],
                        'global_vars': ['scan_html_folder'],
                        'critical': False,
                        'retry_count': 0,
                        'max_retries': 1
                    }
                ]
                
                success_count = 0
                total_modules = len(modules)
                failed_modules = []
                
                # Enhanced module loading with progress tracking and retry logic
                for i, module_info in enumerate(modules):
                    module_name = module_info['name']
                    display_name = module_info['display_name']
                    max_retries = module_info['max_retries']
                    
                    # Progress callback with detailed information
                    if splash_callback:
                        progress_percent = int((i / total_modules) * 100)
                        splash_callback(f"Loading {display_name}... ({progress_percent}%)")
                    
                    # Retry logic for robust loading
                    loaded_successfully = False
                    
                    for retry_attempt in range(max_retries + 1):
                        try:
                            if retry_attempt > 0:
                                # Add small delay between retries
                                time.sleep(0.2)
                                if splash_callback:
                                    splash_callback(f"Retrying {display_name}... (attempt {retry_attempt + 1})")
                            
                            # Enhanced import logic with specific error handling
                            if module_name == 'TransateKRtoEN':
                                # Validate the module before importing critical functions
                                import TransateKRtoEN
                                # Verify the module has required functions
                                if hasattr(TransateKRtoEN, 'main') and hasattr(TransateKRtoEN, 'set_stop_flag'):
                                    from TransateKRtoEN import main as translation_main, set_stop_flag as translation_stop_flag, is_stop_requested as translation_stop_check
                                else:
                                    raise ImportError("TransateKRtoEN module missing required functions")
                                    
                            elif module_name == 'extract_glossary_from_epub':
                                # Validate the module before importing critical functions  
                                import extract_glossary_from_epub
                                if hasattr(extract_glossary_from_epub, 'main') and hasattr(extract_glossary_from_epub, 'set_stop_flag'):
                                    from extract_glossary_from_epub import main as glossary_main, set_stop_flag as glossary_stop_flag, is_stop_requested as glossary_stop_check
                                else:
                                    raise ImportError("extract_glossary_from_epub module missing required functions")
                                    
                            elif module_name == 'epub_converter':
                                # Validate the module before importing
                                import epub_converter
                                if hasattr(epub_converter, 'fallback_compile_epub'):
                                    from epub_converter import fallback_compile_epub
                                else:
                                    raise ImportError("epub_converter module missing fallback_compile_epub function")
                                    
                            elif module_name == 'scan_html_folder':
                                # Validate the module before importing
                                import scan_html_folder
                                if hasattr(scan_html_folder, 'scan_html_folder'):
                                    from scan_html_folder import scan_html_folder
                                else:
                                    raise ImportError("scan_html_folder module missing scan_html_folder function")
                            
                            # If we reach here, import was successful
                            loaded_successfully = True
                            success_count += 1
                            break
                            
                        except ImportError as e:
                            module_info['retry_count'] = retry_attempt + 1
                            error_msg = str(e)
                            
                            # Log retry attempts
                            if retry_attempt < max_retries:
                                if hasattr(self, 'append_log'):
                                    self.append_log(f"⚠️ Failed to load {display_name} (attempt {retry_attempt + 1}): {error_msg}")
                            else:
                                # Final failure
                                print(f"Warning: Could not import {module_name} after {max_retries + 1} attempts: {error_msg}")
                                failed_modules.append({
                                    'name': module_name,
                                    'display_name': display_name,
                                    'error': error_msg,
                                    'critical': module_info['critical']
                                })
                                break
                        
                        except Exception as e:
                            # Handle unexpected errors
                            error_msg = f"Unexpected error: {str(e)}"
                            print(f"Warning: Unexpected error loading {module_name}: {error_msg}")
                            failed_modules.append({
                                'name': module_name,
                                'display_name': display_name, 
                                'error': error_msg,
                                'critical': module_info['critical']
                            })
                            break
                    
                    # Enhanced progress feedback
                    if loaded_successfully and splash_callback:
                        progress_percent = int(((i + 1) / total_modules) * 100)
                        splash_callback(f"✅ {display_name} loaded ({progress_percent}%)")
                
                # Calculate loading time for performance monitoring
                loading_time = time.time() - loading_start_time
                
                # Enhanced success/failure reporting
                if splash_callback:
                    if success_count == total_modules:
                        splash_callback(f"Loaded {success_count}/{total_modules} modules successfully in {loading_time:.1f}s")
                    else:
                        splash_callback(f"Loaded {success_count}/{total_modules} modules ({len(failed_modules)} failed)")
                
                # Enhanced logging with module status details
                if hasattr(self, 'append_log'):
                    if success_count == total_modules:
                        self.append_log(f"✅ Loaded {success_count}/{total_modules} modules successfully in {loading_time:.1f}s")
                    else:
                        self.append_log(f"⚠️ Loaded {success_count}/{total_modules} modules successfully ({len(failed_modules)} failed)")
                        
                        # Report critical failures
                        critical_failures = [f for f in failed_modules if f['critical']]
                        if critical_failures:
                            for failure in critical_failures:
                                self.append_log(f"❌ Critical module failed: {failure['display_name']} - {failure['error']}")
                        
                        # Report non-critical failures
                        non_critical_failures = [f for f in failed_modules if not f['critical']]
                        if non_critical_failures:
                            for failure in non_critical_failures:
                                self.append_log(f"⚠️ Optional module failed: {failure['display_name']} - {failure['error']}")
                
                # Final module state update with enhanced error checking
                self._modules_loaded = True
                self._modules_loading = False
                
                # Enhanced module availability checking with better integration
                if hasattr(self, 'master'):
                    self.master.after(0, self._check_modules)
                
                # Return success status - maintain compatibility by returning True if any modules loaded
                # But also check for critical module failures
                critical_failures = [f for f in failed_modules if f['critical']]
                if critical_failures and success_count == 0:
                    # Complete failure case
                    if hasattr(self, 'append_log'):
                        self.append_log("❌ Critical module loading failed - some functionality may be unavailable")
                    return False
                
                return True
                
            except Exception as unexpected_error:
                # Enhanced error recovery for unexpected failures
                error_msg = f"Unexpected error during module loading: {str(unexpected_error)}"
                print(f"Critical error: {error_msg}")
                
                if hasattr(self, 'append_log'):
                    self.append_log(f"❌ Module loading failed: {error_msg}")
                
                # Reset states for retry possibility
                self._modules_loaded = False
                self._modules_loading = False
                
                if splash_callback:
                    splash_callback(f"Module loading failed: {str(unexpected_error)}")
                
                return False
                
            finally:
                # Enhanced cleanup - ensure loading flag is always reset
                if self._modules_loading:
                    self._modules_loading = False

    def _check_modules(self):
        """Check which modules are available and disable buttons if needed"""
        if not self._modules_loaded:
            return
        
        button_checks = [
            (translation_main, 'run_button', "Translation"),
            (glossary_main, 'glossary_button', "Glossary extraction"),
            (fallback_compile_epub, 'epub_button', "EPUB converter"),
            (scan_html_folder, 'qa_button', "QA scanner")
        ]
        
        for module, button_attr, name in button_checks:
            if module is None and hasattr(self, button_attr):
                getattr(self, button_attr).config(state='disabled')
                self.append_log(f"⚠️ {name} module not available")

    def _setup_text_undo_redo(self, text_widget):
        """Set up undo/redo bindings for a text widget"""
        def handle_undo(event):
            try: text_widget.edit_undo()
            except tk.TclError: pass
            return "break"
        
        def handle_redo(event):
            try: text_widget.edit_redo()
            except tk.TclError: pass
            return "break"
        
        text_widget.bind('<Control-z>', handle_undo)
        text_widget.bind('<Control-y>', handle_redo)
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
        dialog.update()
        canvas.update()
        
        current_geometry = dialog.geometry()
        current_height = int(current_geometry.split('x')[1].split('+')[0])
        
        scrollable_frame = None
        for child in canvas.winfo_children():
            if isinstance(child, ttk.Frame):
                scrollable_frame = child
                break
        
        if not scrollable_frame:
            return
        
        scrollable_frame.update_idletasks()
        window_width = scrollable_frame.winfo_reqwidth() + 20
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        
        max_width = int(screen_width * max_width_ratio)
        final_width = min(window_width, max_width)
        final_height = current_height
        
        x = (screen_width - final_width) // 2
        y = max(20, (screen_height - final_height) // 2)
        dialog.geometry(f"{final_width}x{final_height}+{x}+{y}")

    def _setup_dialog_scrolling(self, dialog_window, canvas):
        """Setup mouse wheel scrolling for dialogs"""
        def _on_mousewheel(event):
            try: canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except: pass
        
        def _on_mousewheel_linux(event, direction):
            try:
                if canvas.winfo_exists():
                    canvas.yview_scroll(direction, "units")
            except tk.TclError: pass
        
        wheel_handler = lambda e: _on_mousewheel(e)
        wheel_up = lambda e: _on_mousewheel_linux(e, -1)
        wheel_down = lambda e: _on_mousewheel_linux(e, 1)
        
        dialog_window.bind_all("<MouseWheel>", wheel_handler)
        dialog_window.bind_all("<Button-4>", wheel_up)
        dialog_window.bind_all("<Button-5>", wheel_down)
        
        def cleanup_bindings():
            try:
                dialog_window.unbind_all("<MouseWheel>")
                dialog_window.unbind_all("<Button-4>")
                dialog_window.unbind_all("<Button-5>")
            except: pass
        
        return cleanup_bindings

    def configure_title_prompt(self):
        """Configure the book title translation prompt"""
        dialog = tk.Toplevel(self.master)
        dialog.title("Configure Book Title Translation")
        dialog.geometry("950x700")
        dialog.transient(self.master)
        load_application_icon(dialog, self.base_dir)
        
        main_frame = tk.Frame(dialog, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="Book Title Translation Prompt", 
                font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        tk.Label(main_frame, text="This prompt will be used when translating book titles.\n"
                "The book title will be appended after this prompt.",
                font=('TkDefaultFont', 11), fg='gray').pack(anchor=tk.W, pady=(0, 10))
        
        self.title_prompt_text = scrolledtext.ScrolledText(main_frame, height=8, wrap=tk.WORD,
                                                          undo=True, autoseparators=True, maxundo=-1)
        self.title_prompt_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.title_prompt_text.insert('1.0', self.book_title_prompt)
        self._setup_text_undo_redo(self.title_prompt_text)
        
        lang_frame = tk.Frame(main_frame)
        lang_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(lang_frame, text="💡 Tip: Modify the prompt above to translate to other languages",
                font=('TkDefaultFont', 10), fg='blue').pack(anchor=tk.W)
        
        example_frame = tk.LabelFrame(main_frame, text="Example Prompts", padx=10, pady=10)
        example_frame.pack(fill=tk.X, pady=(10, 0))
        
        examples = [
            ("Spanish", "Traduce este título de libro al español manteniendo los acrónimos:"),
            ("French", "Traduisez ce titre de livre en français en conservant les acronymes:"),
            ("German", "Übersetzen Sie diesen Buchtitel ins Deutsche und behalten Sie Akronyme bei:"),
            ("Keep Original", "Return the title exactly as provided without any translation:")
        ]
        
        for lang, prompt in examples:
            btn = tb.Button(example_frame, text=f"Use {lang}", 
                           command=lambda p=prompt: self.title_prompt_text.replace('1.0', tk.END, p),
                           bootstyle="secondary-outline", width=15)
            btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        def save_title_prompt():
            self.book_title_prompt = self.title_prompt_text.get('1.0', tk.END).strip()
            self.config['book_title_prompt'] = self.book_title_prompt
            messagebox.showinfo("Success", "Book title prompt saved!")
            dialog.destroy()
        
        def reset_title_prompt():
            if messagebox.askyesno("Reset Prompt", "Reset to default English translation prompt?"):
                default_prompt = "Translate this book title to English while retaining any acronyms:"
                self.title_prompt_text.delete('1.0', tk.END)
                self.title_prompt_text.insert('1.0', default_prompt)
        
        tb.Button(button_frame, text="Save", command=save_title_prompt, 
                 bootstyle="success", width=15).pack(side=tk.LEFT, padx=5)
        tb.Button(button_frame, text="Reset to Default", command=reset_title_prompt, 
                 bootstyle="warning", width=15).pack(side=tk.LEFT, padx=5)
        tb.Button(button_frame, text="Cancel", command=dialog.destroy, 
                 bootstyle="secondary", width=15).pack(side=tk.LEFT, padx=5)

    def force_retranslation(self):
        """Force retranslation of specific chapters"""
        input_path = self.entry_epub.get()
        if not input_path or not os.path.isfile(input_path):
            messagebox.showerror("Error", "Please select a valid EPUB or text file first.")
            return
        
        epub_base = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = epub_base
        
        if not os.path.exists(output_dir):
            messagebox.showinfo("Info", "No translation output found for this EPUB.")
            return
        
        progress_file = os.path.join(output_dir, "translation_progress.json")
        if not os.path.exists(progress_file):
            messagebox.showinfo("Info", "No progress tracking found.")
            return
        
        with open(progress_file, 'r', encoding='utf-8') as f:
            prog = json.load(f)
        
        chapters_info_file = os.path.join(output_dir, "chapters_info.json")
        chapters_info_map = {}
        if os.path.exists(chapters_info_file):
            try:
                with open(chapters_info_file, 'r', encoding='utf-8') as f:
                    chapters_info = json.load(f)
                    for ch_info in chapters_info:
                        if 'num' in ch_info:
                            chapters_info_map[ch_info['num']] = ch_info
            except: pass
        
        dialog = tk.Toplevel(self.master)
        dialog.title("Force Retranslation")
        dialog.geometry("660x600")
        load_application_icon(dialog, self.base_dir)
        
        tk.Label(dialog, text="Select chapters to retranslate:", font=('Arial', 12)).pack(pady=10)
        
        frame = tk.Frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Process chapters
        all_extracted_nums = []
        for chapter_key, chapter_info in prog.get("chapters", {}).items():
            output_file = chapter_info.get("output_file", "")
            if output_file:
                patterns = [r'(\d{4})[_\.]', r'(\d{3,4})[_\.]', r'No(\d+)Chapter',
                           r'response_(\d+)[_\.]', r'chapter[_\s]*(\d+)', r'_(\d+)_']
                for pattern in patterns:
                    match = re.search(pattern, output_file, re.IGNORECASE)
                    if match:
                        num = int(match.group(1))
                        all_extracted_nums.append(num)
                        break
        
        # Detect numbering system
        uses_zero_based = False
        for chapter_key, chapter_info in prog.get("chapters", {}).items():
            if chapter_info.get("status") == "completed":
                output_file = chapter_info.get("output_file", "")
                stored_chapter_num = chapter_info.get("chapter_num", 0)
                if output_file:
                    match = re.search(r'response_(\d+)', output_file)
                    if match:
                        file_num = int(match.group(1))
                        if file_num == stored_chapter_num - 1:
                            uses_zero_based = True
                            break
                        elif file_num == stored_chapter_num:
                            uses_zero_based = False
                            break
        
        if not uses_zero_based:
            try:
                for file in os.listdir(output_dir):
                    if re.search(r'_0+[_\.]', file):
                        uses_zero_based = True
                        break
            except: pass
        
        chapter_keys = []
        chapters_with_nums = []
        
        for chapter_key, chapter_info in prog.get("chapters", {}).items():
            output_file = chapter_info.get("output_file", "")
            stored_num = chapter_info.get("chapter_num", 0)
            actual_num = stored_num
            
            sources_to_try = []
            if output_file:
                sources_to_try.append(("output_file", output_file))
            if "display_name" in chapter_info:
                sources_to_try.append(("display_name", chapter_info["display_name"]))
            if "file_basename" in chapter_info:
                sources_to_try.append(("file_basename", chapter_info["file_basename"]))
            
            patterns = [r'(\d{4})[_\.]', r'(\d{3,4})[_\.]', r'No(\d+)Chapter',
                       r'response_(\d+)[_\.]', r'chapter[_\s]*(\d+)', r'ch[_\s]*(\d+)',
                       r'_(\d+)_', r'(\d{3,4})[^\d]']
            
            found = False
            for source_name, source in sources_to_try:
                if not source: continue
                for pattern in patterns:
                    match = re.search(pattern, source, re.IGNORECASE)
                    if match:
                        extracted_num = int(match.group(1))
                        if chapter_info.get("status") == "in_progress":
                            actual_num = chapter_info.get("chapter_idx", stored_num)
                            chapters_with_nums.append((chapter_key, chapter_info, actual_num))
                            continue
                        else:
                            actual_num = extracted_num + 1 if uses_zero_based else extracted_num
                        found = True
                        break
                if found: break
            
            if not found and chapter_info.get("status") == "in_progress":
                final_num = chapter_info.get("actual_num", 0)
                if final_num == 0:
                    chapter_idx = chapter_info.get("chapter_idx")
                    final_num = chapter_idx if chapter_idx is not None else stored_num
                chapters_with_nums.append((chapter_key, chapter_info, final_num))
                continue
            
            chapters_with_nums.append((chapter_key, chapter_info, actual_num))
        
        # Remove duplicates
        seen_chapter_indices = {}
        final_chapters = []
        for chapter_key, chapter_info, actual_num in chapters_with_nums:
            chapter_idx = chapter_info.get("chapter_idx", actual_num)
            if chapter_idx not in seen_chapter_indices:
                seen_chapter_indices[chapter_idx] = (chapter_key, chapter_info, actual_num)
                final_chapters.append((chapter_key, chapter_info, actual_num))
        
        chapters_with_nums = sorted(final_chapters, key=lambda x: x[2])
        
        # Populate listbox
        for chapter_key, chapter_info, actual_num in chapters_with_nums:
            status = chapter_info.get("status", "unknown")
            output_file = chapter_info.get("output_file", "")
            file_exists = "✓" if output_file and os.path.exists(os.path.join(output_dir, output_file)) else "✗"
            display_text = f"Chapter {actual_num} - {status} - File: {file_exists}"
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
                if chapter_key in prog["chapters"]:
                    chapter_info = prog["chapters"][chapter_key]
                    del prog["chapters"][chapter_key]
                    
                    content_hash = chapter_info.get("content_hash")
                    if content_hash and content_hash in prog.get("content_hashes", {}):
                        stored_hash_info = prog["content_hashes"].get(content_hash, {})
                        if stored_hash_info.get("chapter_idx") == chapter_info.get("chapter_idx"):
                            del prog["content_hashes"][content_hash]
                    
                    if chapter_key in prog.get("chapter_chunks", {}):
                        del prog["chapter_chunks"][chapter_key]
                    
                    output_file = chapter_info.get("output_file", "")
                    if output_file:
                        output_path = os.path.join(output_dir, output_file)
                        if os.path.exists(output_path):
                            os.remove(output_path)
                            self.append_log(f"🗑️ Deleted: {output_file}")
                    count += 1
            
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(prog, f, ensure_ascii=False, indent=2)
            
            self.append_log(f"🔄 Marked {count} chapters for retranslation")
            messagebox.showinfo("Success", f"Marked {count} chapters for retranslation.\nRun translation to process them.")
            dialog.destroy()
        
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

    def glossary_manager(self):
        """Open comprehensive glossary management dialog"""
        manager = tk.Toplevel(self.master)
        manager.title("Glossary Manager")
        
        screen_width = manager.winfo_screenwidth()
        screen_height = manager.winfo_screenheight()
        
        width, height = 0, 1550
        x = (screen_width - width) // 2
        y = max(20, (screen_height - height) // 2)
        
        manager.geometry(f"{width}x{height}+{x}+{y}")
        manager.withdraw()
        manager.transient(self.master)
        load_application_icon(manager, self.base_dir)
        
        main_container = tk.Frame(manager)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(main_container, bg='white')
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        cleanup_scrolling = self._setup_dialog_scrolling(manager, canvas)
        
        notebook = ttk.Notebook(scrollable_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        manual_frame = ttk.Frame(notebook)
        notebook.add(manual_frame, text="Manual Glossary Extraction")
        
        auto_frame = ttk.Frame(notebook)
        notebook.add(auto_frame, text="Automatic Glossary Generation")
        
        editor_frame = ttk.Frame(notebook)
        notebook.add(editor_frame, text="Glossary Editor")
        
        # Manual Glossary Tab
        self._setup_manual_glossary_tab(manual_frame)
        
        # Automatic Glossary Tab
        self._setup_auto_glossary_tab(auto_frame)
        
        # Editor Tab
        self._setup_glossary_editor_tab(editor_frame)
        
        # Dialog Controls
        control_frame = tk.Frame(manager)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def save_glossary_settings():
            try:
                for field, var in self.manual_field_vars.items():
                    self.config[f'manual_extract_{field}'] = var.get()
                
                self.config['custom_glossary_fields'] = self.custom_glossary_fields
                
                self.manual_glossary_prompt = self.manual_prompt_text.get('1.0', tk.END).strip()
                self.auto_glossary_prompt = self.auto_prompt_text.get('1.0', tk.END).strip()
                self.config['manual_glossary_prompt'] = self.manual_glossary_prompt
                self.config['enable_auto_glossary'] = self.enable_auto_glossary_var.get()
                self.config['append_glossary'] = self.append_glossary_var.get()
                self.config['auto_glossary_prompt'] = self.auto_glossary_prompt
                
                try:
                    self.config['manual_glossary_temperature'] = float(self.manual_temp_var.get())
                    self.config['manual_context_limit'] = int(self.manual_context_var.get())
                except ValueError:
                    messagebox.showwarning("Invalid Input", "Please enter valid numbers for temperature and context limit")
                    return
                
                os.environ['GLOSSARY_SYSTEM_PROMPT'] = self.manual_glossary_prompt
                os.environ['AUTO_GLOSSARY_PROMPT'] = self.auto_glossary_prompt
                
                enabled_fields = []
                for field, var in self.manual_field_vars.items():
                    if var.get():
                        os.environ[f'GLOSSARY_EXTRACT_{field.upper()}'] = '1'
                        enabled_fields.append(field)
                    else:
                        os.environ[f'GLOSSARY_EXTRACT_{field.upper()}'] = '0'
                
                if self.custom_glossary_fields:
                    os.environ['GLOSSARY_CUSTOM_FIELDS'] = json.dumps(self.custom_glossary_fields)
                
                with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                
                self.append_log("✅ Glossary settings saved successfully")
                messagebox.showinfo("Success", "Glossary settings saved!")
                manager.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save settings: {e}")
                self.append_log(f"❌ Failed to save glossary settings: {e}")
        
        button_container = tk.Frame(control_frame)
        button_container.pack(expand=True)
        
        tb.Button(button_container, text="Save All Settings", command=save_glossary_settings, 
                 bootstyle="success", width=20).pack(side=tk.LEFT, padx=5)
        tb.Button(button_container, text="Cancel", command=lambda: [cleanup_scrolling(), manager.destroy()], 
                 bootstyle="secondary", width=20).pack(side=tk.LEFT, padx=5)
        
        self._auto_resize_dialog(manager, canvas, max_width_ratio=0.8, max_height_ratio=0.85)
        manager.deiconify()

    def _setup_manual_glossary_tab(self, parent):
        """Setup manual glossary tab"""
        manual_container = tk.Frame(parent)
        manual_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        fields_frame = tk.LabelFrame(manual_container, text="Extraction Fields", padx=10, pady=10)
        fields_frame.pack(fill=tk.X, pady=(0, 10))
        
        if not hasattr(self, 'manual_field_vars'):
            self.manual_field_vars = {
                'original_name': tk.BooleanVar(value=self.config.get('manual_extract_original_name', True)),
                'name': tk.BooleanVar(value=self.config.get('manual_extract_name', True)),
                'gender': tk.BooleanVar(value=self.config.get('manual_extract_gender', True)),
                'title': tk.BooleanVar(value=self.config.get('manual_extract_title', True)),
                'group_affiliation': tk.BooleanVar(value=self.config.get('manual_extract_group_affiliation', True)),
                'traits': tk.BooleanVar(value=self.config.get('manual_extract_traits', True)),
                'how_they_refer_to_others': tk.BooleanVar(value=self.config.get('manual_extract_how_they_refer_to_others', True)),
                'locations': tk.BooleanVar(value=self.config.get('manual_extract_locations', True))
            }
        
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
        
        fields_grid = tk.Frame(fields_frame)
        fields_grid.pack(fill=tk.X)
        
        for row, (field, var) in enumerate(self.manual_field_vars.items()):
            cb = tb.Checkbutton(fields_grid, text=field.replace('_', ' ').title(), 
                               variable=var, bootstyle="round-toggle")
            cb.grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            
            desc = tk.Label(fields_grid, text=field_info[field], 
                          font=('TkDefaultFont', 9), fg='gray')
            desc.grid(row=row, column=1, sticky=tk.W, padx=20, pady=2)
        
        # Custom fields
        custom_frame = tk.LabelFrame(manual_container, text="Custom Fields", padx=10, pady=10)
        custom_frame.pack(fill=tk.X, pady=(0, 10))
        
        custom_list_frame = tk.Frame(custom_frame)
        custom_list_frame.pack(fill=tk.X)
        
        tk.Label(custom_list_frame, text="Additional fields to extract:").pack(anchor=tk.W)
        
        custom_scroll = ttk.Scrollbar(custom_list_frame)
        custom_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.custom_fields_listbox = tk.Listbox(custom_list_frame, height=5, 
                                               yscrollcommand=custom_scroll.set)
        self.custom_fields_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        custom_scroll.config(command=self.custom_fields_listbox.yview)
        
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
        
        # Prompt section
        prompt_frame = tk.LabelFrame(manual_container, text="Extraction Prompt Template", padx=10, pady=10)
        prompt_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(prompt_frame, text="Use {fields} for field list and {chapter_text} for content placeholder",
                font=('TkDefaultFont', 9), fg='blue').pack(anchor=tk.W, pady=(0, 5))
        
        self.manual_prompt_text = scrolledtext.ScrolledText(prompt_frame, height=12, wrap=tk.WORD,
                                                           undo=True, autoseparators=True, maxundo=-1)
        self.manual_prompt_text.pack(fill=tk.BOTH, expand=True)
        self.manual_prompt_text.insert('1.0', self.manual_glossary_prompt)
        self.manual_prompt_text.edit_reset()
        self._setup_text_undo_redo(self.manual_prompt_text)
        
        prompt_controls = tk.Frame(manual_container)
        prompt_controls.pack(fill=tk.X, pady=(10, 0))
        
        def reset_manual_prompt():
            if messagebox.askyesno("Reset Prompt", "Reset manual glossary prompt to default?"):
                self.manual_prompt_text.delete('1.0', tk.END)
                self.manual_prompt_text.insert('1.0', self.default_manual_glossary_prompt)
        
        tb.Button(prompt_controls, text="Reset to Default", command=reset_manual_prompt, 
                 bootstyle="warning").pack(side=tk.LEFT, padx=5)
        
        # Settings
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
        
        tk.Label(settings_grid, text="Rolling Window:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=(10, 0))
        tb.Checkbutton(settings_grid, text="Keep recent context instead of reset", 
                      variable=self.glossary_history_rolling_var,
                      bootstyle="round-toggle").grid(row=1, column=1, columnspan=3, sticky=tk.W, padx=5, pady=(10, 0))
        
        tk.Label(settings_grid, text="When context limit is reached, keep recent chapters instead of clearing all history",
                font=('TkDefaultFont', 11), fg='gray').grid(row=2, column=0, columnspan=4, sticky=tk.W, padx=20, pady=(0, 5))

    def _setup_auto_glossary_tab(self, parent):
        """Setup automatic glossary tab"""
        auto_container = tk.Frame(parent)
        auto_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        master_toggle_frame = tk.Frame(auto_container)
        master_toggle_frame.pack(fill=tk.X, pady=(0, 15))
        
        tb.Checkbutton(master_toggle_frame, text="Enable Automatic Glossary Generation", 
                      variable=self.enable_auto_glossary_var,
                      bootstyle="round-toggle").pack(side=tk.LEFT)
        
        tk.Label(master_toggle_frame, text="(Automatically extracts and translates character names/terms during translation)",
                font=('TkDefaultFont', 10), fg='gray').pack(side=tk.LEFT, padx=(10, 0))
        
        append_frame = tk.Frame(auto_container)
        append_frame.pack(fill=tk.X, pady=(0, 15))
        
        tb.Checkbutton(append_frame, text="Append Glossary to System Prompt", 
                      variable=self.append_glossary_var,
                      bootstyle="round-toggle").pack(side=tk.LEFT)
        
        tk.Label(append_frame, text="(Applies to ALL glossaries - manual and automatic)",
                font=('TkDefaultFont', 10, 'italic'), fg='blue').pack(side=tk.LEFT, padx=(10, 0))
        
        tk.Label(auto_container, 
                text="When enabled: Glossary entries are automatically added to your system prompt\n"
                "When disabled: Glossary is loaded but not injected into prompts\n"
                "This affects both translation and image processing",
                font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 15))
        
        settings_container = tk.Frame(auto_container)
        settings_container.pack(fill=tk.BOTH, expand=True)
        
        extraction_frame = tk.LabelFrame(settings_container, text="Targeted Extraction Settings", padx=10, pady=10)
        extraction_frame.pack(fill=tk.X, pady=(0, 10))
        
        extraction_grid = tk.Frame(extraction_frame)
        extraction_grid.pack(fill=tk.X)
        for i in range(4):
            extraction_grid.grid_columnconfigure(i, weight=1 if i % 2 else 0)
        
        settings = [
            ("Min frequency:", self.glossary_min_frequency_var, 0, 0),
            ("Max names:", self.glossary_max_names_var, 0, 2),
            ("Max titles:", self.glossary_max_titles_var, 1, 0),
            ("Translation batch:", self.glossary_batch_size_var, 1, 2)
        ]
        
        for label, var, row, col in settings:
            tk.Label(extraction_grid, text=label).grid(row=row, column=col, sticky=tk.W, padx=5, pady=5)
            tb.Entry(extraction_grid, textvariable=var, width=8).grid(row=row, column=col+1, sticky=tk.W, padx=5, pady=5)
        
        help_frame = tk.Frame(extraction_frame)
        help_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(help_frame, text="💡 Settings Guide:", font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W)
        help_texts = [
            "• Min frequency: How many times a name must appear (lower = more terms)",
            "• Max names/titles: Limits to prevent huge glossaries",
            "• Translation batch: Terms per API call (larger = faster but may reduce quality)"
        ]
        for txt in help_texts:
            tk.Label(help_frame, text=txt, font=('TkDefaultFont', 11), fg='gray').pack(anchor=tk.W, padx=20)
        
        auto_prompt_frame = tk.LabelFrame(settings_container, text="Extraction Prompt Template", padx=10, pady=10)
        auto_prompt_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(auto_prompt_frame, text="Available placeholders: {language}, {min_frequency}, {max_names}, {max_titles}",
                font=('TkDefaultFont', 9), fg='blue').pack(anchor=tk.W, pady=(0, 5))
        
        self.auto_prompt_text = scrolledtext.ScrolledText(auto_prompt_frame, height=12, wrap=tk.WORD,
                                                         undo=True, autoseparators=True, maxundo=-1)
        self.auto_prompt_text.pack(fill=tk.BOTH, expand=True)
        self.auto_prompt_text.insert('1.0', self.auto_glossary_prompt)
        self.auto_prompt_text.edit_reset()
        self._setup_text_undo_redo(self.auto_prompt_text)
        
        auto_prompt_controls = tk.Frame(settings_container)
        auto_prompt_controls.pack(fill=tk.X, pady=(10, 0))
        
        def reset_auto_prompt():
            if messagebox.askyesno("Reset Prompt", "Reset automatic glossary prompt to default?"):
                self.auto_prompt_text.delete('1.0', tk.END)
                self.auto_prompt_text.insert('1.0', self.default_auto_glossary_prompt)
        
        tb.Button(auto_prompt_controls, text="Reset to Default", command=reset_auto_prompt, 
                 bootstyle="warning").pack(side=tk.LEFT, padx=5)
        
        def update_auto_glossary_state():
            state = tk.NORMAL if self.enable_auto_glossary_var.get() else tk.DISABLED
            for widget in extraction_grid.winfo_children():
                if isinstance(widget, (tb.Entry, ttk.Entry)):
                    widget.config(state=state)
            self.auto_prompt_text.config(state=state)
            for widget in auto_prompt_controls.winfo_children():
                if isinstance(widget, (tb.Button, ttk.Button)):
                    widget.config(state=state)
        
        update_auto_glossary_state()
        self.enable_auto_glossary_var.trace('w', lambda *args: update_auto_glossary_state())

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
                    
                    if field in ['original_name', 'name', 'original', 'translated']:
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
                
                stats = []
                stats.append(f"Total entries: {len(entries)}")
                if self.current_glossary_format == 'list':
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
                title="Select glossary.json",
                filetypes=[("JSON files", "*.json")]
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
            
            count = 0
            if self.current_glossary_format == 'list':
                for entry in self.current_glossary_data:
                    fields_to_remove = []
                    for field, value in entry.items():
                        if value is None or value == '' or (isinstance(value, list) and not value) or (isinstance(value, dict) and not value):
                            fields_to_remove.append(field)
                    for field in fields_to_remove:
                        entry.pop(field)
                        count += 1
            
            elif self.current_glossary_format == 'dict':
                messagebox.showinfo("Info", "Empty field cleaning is only available for manual glossary format")
                return
            
            if count > 0 and save_current_glossary():
                load_glossary_for_editing()
                messagebox.showinfo("Success", f"Removed {count} empty fields and saved")
                self.append_log(f"✅ Cleaned {count} empty fields from glossary")
        
        def delete_selected_entries():
            selected = self.glossary_tree.selection()
            if not selected:
                messagebox.showwarning("Warning", "No entries selected")
                return
            
            if messagebox.askyesno("Confirm Delete", f"Delete {len(selected)} selected entries?"):
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
                seen = {}
                unique_entries = []
                duplicates = 0
                
                for entry in self.current_glossary_data:
                    key = entry.get('original_name') or entry.get('name')
                    if key and key not in seen:
                        seen[key] = True
                        unique_entries.append(entry)
                    else:
                        duplicates += 1
                
                self.current_glossary_data[:] = unique_entries
                
                if duplicates > 0 and save_current_glossary():
                    load_glossary_for_editing()
                    messagebox.showinfo("Success", f"Removed {duplicates} duplicate entries")
                else:
                    messagebox.showinfo("Info", "No duplicates found")
        
        def smart_trim_dialog():
            if not self.current_glossary_data:
                messagebox.showerror("Error", "No glossary loaded")
                return
            
            dialog = tk.Toplevel(self.master)
            dialog.title("Smart Trim Glossary")
            dialog.geometry("500x700")
            dialog.transient(self.master)
            load_application_icon(dialog, self.base_dir)
            
            main_frame = tk.Frame(dialog, padx=20, pady=20)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            tk.Label(main_frame, text="Smart Glossary Trimming", 
                    font=('TkDefaultFont', 12, 'bold')).pack(pady=(0, 10))
            
            options_frame = tk.LabelFrame(main_frame, text="Trimming Options", padx=10, pady=10)
            options_frame.pack(fill=tk.X, pady=(0, 10))
            
            top_frame = tk.Frame(options_frame)
            top_frame.pack(fill=tk.X, pady=5)
            tk.Label(top_frame, text="Keep top").pack(side=tk.LEFT)
            top_var = tk.StringVar(value="100")
            tb.Entry(top_frame, textvariable=top_var, width=10).pack(side=tk.LEFT, padx=5)
            tk.Label(top_frame, text="entries").pack(side=tk.LEFT)
            
            tk.Label(options_frame, text="Field-specific limits:", 
                    font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))
            
            field_vars = {}
            fields_to_limit = [
                ('traits', "Max traits per character:", "5"),
                ('locations', "Max locations per entry:", "10"),
                ('group_affiliation', "Max groups per character:", "3")
            ]
            
            for field, label, default in fields_to_limit:
                frame = tk.Frame(options_frame)
                frame.pack(fill=tk.X, pady=2)
                tk.Label(frame, text=label).pack(side=tk.LEFT)
                var = tk.StringVar(value=default)
                tb.Entry(frame, textvariable=var, width=10).pack(side=tk.LEFT, padx=5)
                field_vars[field] = var
            
            tk.Label(options_frame, text="Remove fields:", 
                    font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))
            
            remove_vars = {}
            fields_to_remove = ['title', 'how_they_refer_to_others', 'gender']
            
            for field in fields_to_remove:
                var = tk.BooleanVar(value=False)
                tb.Checkbutton(options_frame, text=f"Remove {field.replace('_', ' ')}", 
                             variable=var).pack(anchor=tk.W, padx=20)
                remove_vars[field] = var
            
            def apply_smart_trim():
                try:
                    top_n = int(top_var.get())
                    
                    if self.current_glossary_format == 'list':
                        if top_n < len(self.current_glossary_data):
                            self.current_glossary_data = self.current_glossary_data[:top_n]
                        
                        for entry in self.current_glossary_data:
                            for field, var in field_vars.items():
                                if field in entry and isinstance(entry[field], list):
                                    limit = int(var.get())
                                    if len(entry[field]) > limit:
                                        entry[field] = entry[field][:limit]
                            
                            for field, var in remove_vars.items():
                                if var.get() and field in entry:
                                    entry.pop(field)
                    
                    elif self.current_glossary_format == 'dict':
                        entries = list(self.current_glossary_data['entries'].items())
                        if top_n < len(entries):
                            self.current_glossary_data['entries'] = dict(entries[:top_n])
                    
                    if save_current_glossary():
                        load_glossary_for_editing()
                        messagebox.showinfo("Success", "Smart trim applied successfully")
                        dialog.destroy()
                        
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid numbers")
            
            button_frame = tk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=(10, 0))
            
            tb.Button(button_frame, text="Apply Trim", command=apply_smart_trim,
                     bootstyle="primary", width=15).pack(side=tk.LEFT, padx=5)
            tb.Button(button_frame, text="Cancel", command=dialog.destroy,
                     bootstyle="secondary", width=15).pack(side=tk.LEFT, padx=5)
        
        def filter_entries_dialog():
            if not self.current_glossary_data:
                messagebox.showerror("Error", "No glossary loaded")
                return
            
            dialog = tk.Toplevel(self.master)
            dialog.title("Filter Entries")
            dialog.geometry("400x300")
            dialog.transient(self.master)
            load_application_icon(dialog, self.base_dir)
            
            main_frame = tk.Frame(dialog, padx=20, pady=20)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            tk.Label(main_frame, text="Filter Glossary Entries", 
                    font=('TkDefaultFont', 12, 'bold')).pack(pady=(0, 10))
            
            filter_frame = tk.LabelFrame(main_frame, text="Keep only entries that:", padx=10, pady=10)
            filter_frame.pack(fill=tk.X, pady=(0, 10))
            
            filter_vars = {
                'name': (tk.BooleanVar(value=True), "Have name/original_name"),
                'translation': (tk.BooleanVar(value=False), "Have English translation"),
                'traits': (tk.BooleanVar(value=False), "Have traits"),
                'locations': (tk.BooleanVar(value=False), "Have locations")
            }
            
            for key, (var, label) in filter_vars.items():
                tb.Checkbutton(filter_frame, text=label, variable=var).pack(anchor=tk.W)
            
            def apply_filter():
                if self.current_glossary_format == 'list':
                    filtered = []
                    for entry in self.current_glossary_data:
                        keep = True
                        
                        if filter_vars['name'][0].get():
                            if not (entry.get('name') or entry.get('original_name')):
                                keep = False
                        
                        if filter_vars['translation'][0].get():
                            if not entry.get('name'):
                                keep = False
                        
                        if filter_vars['traits'][0].get():
                            if not entry.get('traits'):
                                keep = False
                        
                        if filter_vars['locations'][0].get():
                            if not entry.get('locations'):
                                keep = False
                        
                        if keep:
                            filtered.append(entry)
                    
                    removed = len(self.current_glossary_data) - len(filtered)
                    self.current_glossary_data[:] = filtered
                    
                    if save_current_glossary():
                        load_glossary_for_editing()
                        messagebox.showinfo("Success", f"Filtered out {removed} entries")
                        dialog.destroy()
                else:
                    messagebox.showinfo("Info", "Filtering is only available for manual glossary format")
                    dialog.destroy()
            
            button_frame = tk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=(10, 0))
            
            tb.Button(button_frame, text="Apply Filter", command=apply_filter,
                     bootstyle="primary", width=15).pack(side=tk.LEFT, padx=5)
            tb.Button(button_frame, text="Cancel", command=dialog.destroy,
                     bootstyle="secondary", width=15).pack(side=tk.LEFT, padx=5)
        
        def export_selection():
            selected = self.glossary_tree.selection()
            if not selected:
                messagebox.showwarning("Warning", "No entries selected")
                return
            
            path = filedialog.asksaveasfilename(
                title="Export Selected Entries",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")]
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
                filetypes=[("JSON files", "*.json")]
            )
            
            if not path:
                return
            
            try:
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
            ("Remove Duplicates", remove_duplicates, "warning")
        ]
        
        for text, cmd, style in buttons_row1:
            tb.Button(row1, text=text, command=cmd, bootstyle=style, width=15).pack(side=tk.LEFT, padx=2)
        
        # Row 2
        row2 = tk.Frame(editor_controls)
        row2.pack(fill=tk.X, pady=2)
        
        buttons_row2 = [
            ("Smart Trim", smart_trim_dialog, "primary"),
            ("Filter Entries", filter_entries_dialog, "primary"),
            ("Aggregate Locations", lambda: self._aggregate_locations(load_glossary_for_editing), "info"),
            ("Export Selection", export_selection, "secondary")
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
        
        dialog = tk.Toplevel(self.master)
        dialog.title(f"Edit {col_name.replace('_', ' ').title()}")
        dialog.geometry("400x150")
        dialog.transient(self.master)
        load_application_icon(dialog, self.base_dir)
        
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        frame = tk.Frame(dialog, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(frame, text=f"Edit {col_name.replace('_', ' ').title()}:").pack(anchor=tk.W)
        
        if col_name in ['traits', 'locations', 'group_affiliation'] or ',' in str(current_value):
            text_widget = tk.Text(frame, height=4, width=50)
            text_widget.pack(fill=tk.BOTH, expand=True, pady=5)
            text_widget.insert('1.0', current_value)
            
            def get_value():
                return text_widget.get('1.0', tk.END).strip()
        else:
            var = tk.StringVar(value=current_value)
            entry = tb.Entry(frame, textvariable=var, width=50)
            entry.pack(fill=tk.X, pady=5)
            entry.focus()
            entry.select_range(0, tk.END)
            
            def get_value():
                return var.get()
        
        def save_edit():
            new_value = get_value()
            
            new_values = list(values)
            new_values[col_idx] = new_value
            self.glossary_tree.item(item, values=new_values)
            
            row_idx = int(self.glossary_tree.item(item)['text']) - 1
            
            if self.current_glossary_format == 'list':
                if 0 <= row_idx < len(self.current_glossary_data):
                    entry = self.current_glossary_data[row_idx]
                    
                    if col_name in ['traits', 'locations', 'group_affiliation']:
                        if new_value:
                            entry[col_name] = [v.strip() for v in new_value.split(',') if v.strip()]
                        else:
                            entry.pop(col_name, None)
                    else:
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

    def _aggregate_locations(self, reload_callback):
        """Aggregate all location entries into a single entry"""
        if not self.current_glossary_data:
            messagebox.showerror("Error", "No glossary loaded")
            return
        
        if isinstance(self.current_glossary_data, list):
            all_locs = []
            for char in self.current_glossary_data:
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
            
            self.current_glossary_data = [
                entry for entry in self.current_glossary_data 
                if entry.get('original_name') != "📍 Location Summary"
            ]
            
            self.current_glossary_data.append({
                "original_name": "📍 Location Summary",
                "name": "Location Summary",
                "locations": unique_locs
            })
            
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
        
        self.qa_button = tb.Button(btn_frame, text="QA Scan", command=self.run_qa_scan, bootstyle="warning")
        self.qa_button.grid(row=0, column=99, sticky=tk.EW, padx=5)
        
        toolbar_items = [
            ("EPUB Converter", self.epub_converter, "info"),
            ("Extract Glossary", self.run_glossary_extraction_thread, "warning"),
            ("Glossary Manager", self.glossary_manager, "secondary"),
            ("Retranslate", self.force_retranslation, "warning"),
            ("Save Config", self.save_config, "secondary"),
            ("Load Glossary", self.load_glossary, "secondary"),
            ("Import Profiles", self.import_profiles, "secondary"),
            ("Export Profiles", self.export_profiles, "secondary"),
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

    # Thread management methods
    def run_translation_thread(self):
        """Start translation in a separate thread"""
        if hasattr(self, 'glossary_thread') and self.glossary_thread and self.glossary_thread.is_alive():
            self.append_log("⚠️ Cannot run translation while glossary extraction is in progress.")
            messagebox.showwarning("Process Running", "Please wait for glossary extraction to complete before starting translation.")
            return
        
        if self.translation_thread and self.translation_thread.is_alive():
            self.stop_translation()
            return
        
        self.stop_requested = False
        if translation_stop_flag:
            translation_stop_flag(False)
        
        self.translation_thread = threading.Thread(target=self.run_translation_direct, daemon=True)
        self.translation_thread.start()
        self.master.after(100, self.update_run_button)

    def run_translation_direct(self):
        """Run translation directly without subprocess"""
        try:
            self.append_log("🔄 Loading translation modules...")
            if not self._lazy_load_modules():
                self.append_log("❌ Failed to load translation modules")
                return
            
            if translation_main is None:
                self.append_log("❌ Translation module is not available")
                messagebox.showerror("Module Error", "Translation module is not available. Please ensure all files are present.")
                return
            
            epub_path = self.entry_epub.get()
            if not epub_path or not os.path.isfile(epub_path):
                self.append_log("❌ Error: Please select a valid EPUB file.")
                return
            
            api_key = self.api_key_entry.get()
            if not api_key:
                self.append_log("❌ Error: Please enter your API key.")
                return
            
            old_argv = sys.argv
            old_env = dict(os.environ)
            
            try:
                self.append_log(f"🔧 Setting up environment variables...")
                self.append_log(f"📖 EPUB: {os.path.basename(epub_path)}")
                self.append_log(f"🤖 Model: {self.model_var.get()}")
                self.append_log(f"🔑 API Key: {api_key[:10]}...")
                self.append_log(f"📤 Output Token Limit: {self.max_output_tokens}")
                
                # Log key settings
                if self.enable_auto_glossary_var.get():
                    self.append_log("✅ Automatic glossary generation ENABLED")
                    self.append_log(f"📑 Targeted Glossary Settings:")
                    self.append_log(f"   • Min frequency: {self.glossary_min_frequency_var.get()} occurrences")
                    self.append_log(f"   • Max character names: {self.glossary_max_names_var.get()}")
                    self.append_log(f"   • Max titles/ranks: {self.glossary_max_titles_var.get()}")
                    self.append_log(f"   • Translation batch size: {self.glossary_batch_size_var.get()}")
                else:
                    self.append_log("⚠️ Automatic glossary generation DISABLED")
                
                if self.batch_translation_var.get():
                    self.append_log(f"📦 Batch translation ENABLED - processing {self.batch_size_var.get()} chapters per API call")
                    self.append_log("   💡 This can improve speed but may reduce per-chapter customization")
                else:
                    self.append_log("📄 Standard translation mode - processing one chapter at a time")
                
                # Set environment variables
                env_vars = self._get_environment_variables(epub_path, api_key)
                os.environ.update(env_vars)
                
                chap_range = self.chapter_range_entry.get().strip()
                if chap_range:
                    os.environ['CHAPTER_RANGE'] = chap_range
                    self.append_log(f"📊 Chapter Range: {chap_range}")
                
                # Handle token limit
                if self.token_limit_disabled:
                    os.environ['MAX_INPUT_TOKENS'] = ''
                    self.append_log("🎯 Input Token Limit: Unlimited (disabled)")
                else:
                    token_val = self.token_limit_entry.get().strip()
                    if token_val and token_val.isdigit():
                        os.environ['MAX_INPUT_TOKENS'] = token_val
                        self.append_log(f"🎯 Input Token Limit: {token_val}")
                    else:
                        default_limit = '1000000'
                        os.environ['MAX_INPUT_TOKENS'] = default_limit
                        self.append_log(f"🎯 Input Token Limit: {default_limit} (default)")
                
                # Log image translation status
                if self.enable_image_translation_var.get():
                    self.append_log("🖼️ Image translation ENABLED")
                    vision_models = ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-2.0-flash', 
                                   'gemini-2.0-flash-exp', 'gpt-4-turbo', 'gpt-4o']
                    if self.model_var.get().lower() in vision_models:
                        self.append_log(f"   ✅ Using vision-capable model: {self.model_var.get()}")
                        self.append_log(f"   • Max images per chapter: {self.max_images_per_chapter_var.get()}")
                        if self.process_webnovel_images_var.get():
                            self.append_log(f"   • Web novel images: Enabled (min height: {self.webnovel_min_height_var.get()}px)")
                    else:
                        self.append_log(f"   ⚠️ Model {self.model_var.get()} does not support vision")
                        self.append_log("   ⚠️ Image translation will be skipped")
                else:
                    self.append_log("🖼️ Image translation disabled")
                
                if hasattr(self, 'manual_glossary_path'):
                    os.environ['MANUAL_GLOSSARY'] = self.manual_glossary_path
                    self.append_log(f"📑 Manual Glossary: {os.path.basename(self.manual_glossary_path)}")
                
                sys.argv = ['TransateKRtoEN.py', epub_path]
                
                self.append_log("🚀 Starting translation...")
                
                os.makedirs("Payloads", exist_ok=True)
                
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
                sys.argv = old_argv
                os.environ.clear()
                os.environ.update(old_env)
        
        except Exception as e:
            self.append_log(f"❌ Translation setup error: {e}")
        
        finally:
            self.stop_requested = False
            if translation_stop_flag:
                translation_stop_flag(False)
            self.translation_thread = None
            self.master.after(0, self.update_run_button)

    def _get_environment_variables(self, epub_path, api_key):
        """Get all environment variables for translation/glossary"""
        return {
            'EPUB_PATH': epub_path,
            'MODEL': self.model_var.get(),
            'CONTEXTUAL': '1' if self.contextual_var.get() else '0',
            'SEND_INTERVAL_SECONDS': str(self.delay_entry.get()),
            'MAX_OUTPUT_TOKENS': str(self.max_output_tokens),
            'API_KEY': api_key,
            'OPENAI_API_KEY': api_key,
            'OPENAI_OR_Gemini_API_KEY': api_key,
            'GEMINI_API_KEY': api_key,
            'SYSTEM_PROMPT': self.prompt_text.get("1.0", "end").strip(),
            'TRANSLATE_BOOK_TITLE': "1" if self.translate_book_title_var.get() else "0",
            'BOOK_TITLE_PROMPT': self.book_title_prompt,
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
            'DISABLE_AUTO_GLOSSARY': "0" if self.enable_auto_glossary_var.get() else "1",
            'DISABLE_GLOSSARY_TRANSLATION': "0" if self.enable_auto_glossary_var.get() else "1",
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
            'ENABLE_IMAGE_TRANSLATION': "1" if self.enable_image_translation_var.get() else "0",
            'PROCESS_WEBNOVEL_IMAGES': "1" if self.process_webnovel_images_var.get() else "0",
            'WEBNOVEL_MIN_HEIGHT': self.webnovel_min_height_var.get(),
            'IMAGE_MAX_TOKENS': self.image_max_tokens_var.get(),
            'MAX_IMAGES_PER_CHAPTER': self.max_images_per_chapter_var.get(),
            'IMAGE_API_DELAY': '1.0',
            'SAVE_IMAGE_TRANSLATIONS': '1',
            'IMAGE_CHUNK_HEIGHT': self.image_chunk_height_var.get(),
            'HIDE_IMAGE_TRANSLATION_LABEL': "1" if self.hide_image_translation_label_var.get() else "0",
            'RETRY_TIMEOUT': "1" if self.retry_timeout_var.get() else "0",
            'CHUNK_TIMEOUT': self.chunk_timeout_var.get(),
            'BATCH_TRANSLATION': "1" if self.batch_translation_var.get() else "0",
            'BATCH_SIZE': self.batch_size_var.get(),
            'DISABLE_ZERO_DETECTION': "1" if self.disable_zero_detection_var.get() else "0",
            'TRANSLATION_HISTORY_ROLLING': "1" if self.translation_history_rolling_var.get() else "0",
            'COMPREHENSIVE_EXTRACTION': "1" if self.comprehensive_extraction_var.get() else "0",
            'DISABLE_EPUB_GALLERY': "1" if self.disable_epub_gallery_var.get() else "0",
            'DUPLICATE_DETECTION_MODE': self.duplicate_detection_mode_var.get(),
            'DUPLICATE_THRESHOLD_MODE': self.duplicate_threshold_mode_var.get()
        }

    def run_glossary_extraction_thread(self):
        """Start glossary extraction in a separate thread"""
        if not self._lazy_load_modules():
            self.append_log("❌ Failed to load glossary modules")
            return
        
        if glossary_main is None:
            self.append_log("❌ Glossary extraction module is not available")
            messagebox.showerror("Module Error", "Glossary extraction module is not available.")
            return
        
        if hasattr(self, 'translation_thread') and self.translation_thread and self.translation_thread.is_alive():
            self.append_log("⚠️ Cannot run glossary extraction while translation is in progress.")
            messagebox.showwarning("Process Running", "Please wait for translation to complete before extracting glossary.")
            return
        
        if self.glossary_thread and self.glossary_thread.is_alive():
            self.stop_glossary_extraction()
            return
        
        self.stop_requested = False
        if glossary_stop_flag:
            glossary_stop_flag(False)
        self.glossary_thread = threading.Thread(target=self.run_glossary_extraction_direct, daemon=True)
        self.glossary_thread.start()
        self.master.after(100, self.update_run_button)

    def run_glossary_extraction_direct(self):
        """Run glossary extraction directly without subprocess"""
        try:
            input_path = self.entry_epub.get()
            if not input_path or not os.path.isfile(input_path):
                self.append_log("❌ Error: Please select a valid EPUB or text file for glossary extraction.")
                return
            
            api_key = self.api_key_entry.get()
            if not api_key:
                self.append_log("❌ Error: Please enter your API key.")
                return
            
            old_argv = sys.argv
            old_env = dict(os.environ)
            
            try:
                env_updates = {
                    'GLOSSARY_TEMPERATURE': str(self.config.get('manual_glossary_temperature', 0.3)),
                    'GLOSSARY_CONTEXT_LIMIT': str(self.config.get('manual_context_limit', 3)),
                    'MODEL': self.model_var.get(),
                    'OPENAI_API_KEY': self.api_key_entry.get(),
                    'OPENAI_OR_Gemini_API_KEY': self.api_key_entry.get(),
                    'API_KEY': self.api_key_entry.get(),
                    'MAX_OUTPUT_TOKENS': str(self.max_output_tokens),
                    'GLOSSARY_SYSTEM_PROMPT': self.manual_glossary_prompt,
                    'CHAPTER_RANGE': self.chapter_range_entry.get().strip(),
                    'GLOSSARY_EXTRACT_ORIGINAL_NAME': '1' if self.config.get('manual_extract_original_name', True) else '0',
                    'GLOSSARY_EXTRACT_NAME': '1' if self.config.get('manual_extract_name', True) else '0',
                    'GLOSSARY_EXTRACT_GENDER': '1' if self.config.get('manual_extract_gender', True) else '0',
                    'GLOSSARY_EXTRACT_TITLE': '1' if self.config.get('manual_extract_title', True) else '0',
                    'GLOSSARY_EXTRACT_GROUP_AFFILIATION': '1' if self.config.get('manual_extract_group_affiliation', True) else '0',
                    'GLOSSARY_EXTRACT_TRAITS': '1' if self.config.get('manual_extract_traits', True) else '0',
                    'GLOSSARY_EXTRACT_HOW_THEY_REFER_TO_OTHERS': '1' if self.config.get('manual_extract_how_they_refer_to_others', True) else '0',
                    'GLOSSARY_EXTRACT_LOCATIONS': '1' if self.config.get('manual_extract_locations', True) else '0',
                    'GLOSSARY_HISTORY_ROLLING': "1" if self.glossary_history_rolling_var.get() else "0"
                }
                
                if self.custom_glossary_fields:
                    env_updates['GLOSSARY_CUSTOM_FIELDS'] = json.dumps(self.custom_glossary_fields)
                
                os.environ.update(env_updates)
                
                chap_range = self.chapter_range_entry.get().strip()
                if chap_range:
                    self.append_log(f"📊 Chapter Range: {chap_range} (glossary extraction will only process these chapters)")
                
                if self.token_limit_disabled:
                    os.environ['MAX_INPUT_TOKENS'] = ''
                    self.append_log("🎯 Input Token Limit: Unlimited (disabled)")
                else:
                    token_val = self.token_limit_entry.get().strip()
                    if token_val and token_val.isdigit():
                        os.environ['MAX_INPUT_TOKENS'] = token_val
                        self.append_log(f"🎯 Input Token Limit: {token_val}")
                    else:
                        os.environ['MAX_INPUT_TOKENS'] = '50000'
                        self.append_log(f"🎯 Input Token Limit: 50000 (default)")
                
                epub_base = os.path.splitext(os.path.basename(input_path))[0]
                output_path = f"{epub_base}_glossary.json"
                
                sys.argv = [
                    'extract_glossary_from_epub.py',
                    '--epub', input_path,
                    '--output', output_path,
                    '--config', CONFIG_FILE
                ]
                
                self.append_log("🚀 Starting glossary extraction...")
                self.append_log(f"📤 Output Token Limit: {self.max_output_tokens}")
                os.environ['MAX_OUTPUT_TOKENS'] = str(self.max_output_tokens)
                
                glossary_main(
                    log_callback=self.append_log,
                    stop_callback=lambda: self.stop_requested
                )
                
                if not self.stop_requested:
                    self.append_log("✅ Glossary extraction completed successfully!")
                
            finally:
                sys.argv = old_argv
                os.environ.clear()
                os.environ.update(old_env)
        
        except Exception as e:
            self.append_log(f"❌ Glossary extraction error: {e}")
        
        finally:
            self.stop_requested = False
            if glossary_stop_flag:
                glossary_stop_flag(False)
            self.glossary_thread = None
            self.master.after(0, self.update_run_button)

    def epub_converter(self):
        """Start EPUB converter in a separate thread"""
        if not self._lazy_load_modules():
            self.append_log("❌ Failed to load EPUB converter modules")
            return
        
        if fallback_compile_epub is None:
            self.append_log("❌ EPUB converter module is not available")
            messagebox.showerror("Module Error", "EPUB converter module is not available.")
            return
        
        if hasattr(self, 'translation_thread') and self.translation_thread and self.translation_thread.is_alive():
            self.append_log("⚠️ Cannot run EPUB converter while translation is in progress.")
            messagebox.showwarning("Process Running", "Please wait for translation to complete before converting EPUB.")
            return
        
        if hasattr(self, 'glossary_thread') and self.glossary_thread and self.glossary_thread.is_alive():
            self.append_log("⚠️ Cannot run EPUB converter while glossary extraction is in progress.")
            messagebox.showwarning("Process Running", "Please wait for glossary extraction to complete before converting EPUB.")
            return
        
        if hasattr(self, 'epub_thread') and self.epub_thread and self.epub_thread.is_alive():
            self.stop_epub_converter()
            return
        
        folder = filedialog.askdirectory(title="Select translation output folder")
        if not folder:
            return
        
        self.epub_folder = folder
        self.stop_requested = False
        self.epub_thread = threading.Thread(target=self.run_epub_converter_direct, daemon=True)
        self.epub_thread.start()
        self.master.after(100, self.update_run_button)

    def run_epub_converter_direct(self):
        """Run EPUB converter directly without blocking GUI"""
        try:
            folder = self.epub_folder
            self.append_log("📦 Starting EPUB Converter...")
            os.environ['DISABLE_EPUB_GALLERY'] = "1" if self.disable_epub_gallery_var.get() else "0"
            
            fallback_compile_epub(folder, log_callback=self.append_log)
            
            if not self.stop_requested:
                self.append_log("✅ EPUB Converter completed successfully!")
                
                epub_files = [f for f in os.listdir(folder) if f.endswith('.epub')]
                if epub_files:
                    epub_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True)
                    out_file = os.path.join(folder, epub_files[0])
                    self.master.after(0, lambda: messagebox.showinfo("EPUB Compilation Success", f"Created: {out_file}"))
                else:
                    self.append_log("⚠️ EPUB file was not created. Check the logs for details.")
            
        except Exception as e:
            error_str = str(e)
            self.append_log(f"❌ EPUB Converter error: {error_str}")
            
            if "Document is empty" not in error_str:
                self.master.after(0, lambda: messagebox.showerror("EPUB Converter Failed", f"Error: {error_str}"))
            else:
                self.append_log("📋 Check the log above for details about what went wrong.")
        
        finally:
            self.epub_thread = None
            self.stop_requested = False
            self.master.after(0, self.update_run_button)
            
            if hasattr(self, 'epub_button'):
                self.master.after(0, lambda: self.epub_button.config(
                    text="EPUB Converter",
                    command=self.epub_converter,
                    bootstyle="info",
                    state=tk.NORMAL if fallback_compile_epub else tk.DISABLED
                ))

    def run_qa_scan(self):
        """Run QA scan with mode selection"""
        # Create a small loading window with icon
        loading_window = tk.Toplevel(self.master)
        loading_window.title("Loading QA Scanner")
        loading_window.geometry("300x120")
        loading_window.transient(self.master)
        loading_window.resizable(False, False)
        
        # Load and set icon
        load_application_icon(loading_window, self.base_dir)
        
        # Center the loading window
        loading_window.update_idletasks()
        x = (loading_window.winfo_screenwidth() // 2) - (loading_window.winfo_width() // 2)
        y = (loading_window.winfo_screenheight() // 2) - (loading_window.winfo_height() // 2)
        loading_window.geometry(f"+{x}+{y}")
        
        # Create content frame
        content_frame = tk.Frame(loading_window, padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Try to add icon image if available
        status_label = None
        try:
            from PIL import Image, ImageTk
            ico_path = os.path.join(self.base_dir, 'Halgakos.ico')
            if os.path.isfile(ico_path):
                # Load icon at small size
                icon_image = Image.open(ico_path)
                icon_image = icon_image.resize((32, 32), Image.Resampling.LANCZOS)
                icon_photo = ImageTk.PhotoImage(icon_image)
                
                # Create horizontal layout
                icon_label = tk.Label(content_frame, image=icon_photo)
                icon_label.image = icon_photo  # Keep reference
                icon_label.pack(side=tk.LEFT, padx=(0, 10))
                
                # Text on the right
                text_frame = tk.Frame(content_frame)
                text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                tk.Label(text_frame, text="Initializing QA Scanner...", 
                        font=('TkDefaultFont', 11)).pack(anchor=tk.W)
                status_label = tk.Label(text_frame, text="Loading modules...", 
                                      font=('TkDefaultFont', 9), fg='gray')
                status_label.pack(anchor=tk.W, pady=(5, 0))
            else:
                # Fallback without icon
                tk.Label(content_frame, text="Initializing QA Scanner...", 
                        font=('TkDefaultFont', 11)).pack()
                status_label = tk.Label(content_frame, text="Loading modules...", 
                                      font=('TkDefaultFont', 9), fg='gray')
                status_label.pack(pady=(10, 0))
        except ImportError:
            # No PIL, simple text only
            tk.Label(content_frame, text="Initializing QA Scanner...", 
                    font=('TkDefaultFont', 11)).pack()
            status_label = tk.Label(content_frame, text="Loading modules...", 
                                  font=('TkDefaultFont', 9), fg='gray')
            status_label.pack(pady=(10, 0))
        
        # Disable main window interaction
        loading_window.grab_set()
        self.master.update_idletasks()
        
        try:
            # Update status
            if status_label:
                status_label.config(text="Loading translation modules...")
            loading_window.update_idletasks()
            
            if not self._lazy_load_modules():
                loading_window.destroy()
                self.append_log("❌ Failed to load QA scanner modules")
                return
            
            if status_label:
                status_label.config(text="Preparing scanner...")
            loading_window.update_idletasks()
            
            if scan_html_folder is None:
                loading_window.destroy()
                self.append_log("❌ QA scanner module is not available")
                messagebox.showerror("Module Error", "QA scanner module is not available.")
                return
            
            if hasattr(self, 'qa_thread') and self.qa_thread and self.qa_thread.is_alive():
                loading_window.destroy()
                self.stop_requested = True
                self.append_log("⛔ QA scan stop requested.")
                return
            
            # Close loading window
            loading_window.destroy()
            self.append_log("✅ QA scanner initialized successfully")
            
        except Exception as e:
            loading_window.destroy()
            self.append_log(f"❌ Error initializing QA scanner: {e}")
            return
        
        # ALWAYS show mode selection dialog
        mode_dialog = tk.Toplevel(self.master)
        mode_dialog.title("Select QA Scanner Mode")
        mode_dialog.geometry("1920x820")
        mode_dialog.withdraw()  # Hide initially for smooth opening
        mode_dialog.transient(self.master)
        load_application_icon(mode_dialog, self.base_dir)
        
        # Variables
        selected_mode_value = None
        
        # Main container
        main_container = tk.Frame(mode_dialog)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Content with padding
        main_frame = tk.Frame(main_container, padx=40, pady=35)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title with subtitle
        title_frame = tk.Frame(main_frame)
        title_frame.pack(pady=(0, 30))
        
        tk.Label(title_frame, text="Select Detection Mode", 
                 font=('Arial', 36, 'bold'), fg='#f0f0f0').pack()
        tk.Label(title_frame, text="Choose how sensitive the duplicate detection should be",
                 font=('Arial', 20), fg='#d0d0d0').pack(pady=(8, 0))
        
        # Mode cards container
        modes_container = tk.Frame(main_frame)
        modes_container.pack(fill=tk.BOTH, expand=True)
                
        mode_data = [
            {
                "value": "ai-hunter",
                "emoji": "🤖",
                "title": "AI HUNTER",
                "subtitle": "30% threshold",
                "features": [
                    "✓ Catches AI retranslations",
                    "✓ Different translation styles",
                    "⚠ MANY false positives",
                    "✓ Same chapter, different words",
                    "✓ Detects paraphrasing",
                    "✓ Ultimate duplicate finder"
                ],
                "bg_color": "#2a1a3e",  # Dark purple
                "hover_color": "#6a4c93",  # Medium purple
                "border_color": "#8b5cf6",
                "accent_color": "#a78bfa",
                "text_color": "#f0f0f0",
                "feature_color": "#e0e0e0",
                "recommendation": "EXTREME"
            },
            {
                "value": "aggressive",
                "emoji": "🔴",
                "title": "AGGRESSIVE",
                "subtitle": "75% threshold",
                "features": [
                    "✓ Catches more duplicates",
                    "✓ Best for initial cleanup", 
                    "⚠ May have false positives",
                    "✓ Detects partial matches",
                    "✓ Finds similar content patterns",
                    "✓ Aggressive fuzzy matching"
                ],
                "bg_color": "#4a1515",
                "hover_color": "#dc143c",
                "border_color": "#ff0000",
                "accent_color": "#ff5555",
                "text_color": "#f0f0f0",
                "feature_color": "#e0e0e0",
                "recommendation": "BEST"
            },
            {
                "value": "standard", 
                "emoji": "🟡",
                "title": "STANDARD",
                "subtitle": "85% threshold",
                "features": [
                    "✓ Balanced detection",
                    "✓ Good for most cases",
                    "✓ Recommended approach",
                    "✓ Reliable accuracy",
                    "✓ Minimal false alarms",
                    "✓ Production ready"
                ],
                "bg_color": "#4a4015",
                "hover_color": "#d4a017",
                "border_color": "#ffaa00",
                "accent_color": "#ffdd44",
                "text_color": "#f0f0f0",
                "feature_color": "#e0e0e0",
                "recommendation": "RECOMMENDED"
            },
            {
                "value": "strict",
                "emoji": "🟢", 
                "title": "STRICT",
                "subtitle": "95% threshold",
                "features": [
                    "✓ Only identical matches",
                    "✓ Minimal false positives",
                    "✓ Precision focused",
                    "✓ Conservative approach",
                    "✓ High confidence results",
                    "✓ Final QA checks"
                ],
                "bg_color": "#1a3a1c",
                "hover_color": "#228b22",
                "border_color": "#2e7d32",
                "accent_color": "#66bb6a",
                "text_color": "#f0f0f0",
                "feature_color": "#e0e0e0",
                "recommendation": "FASTEST"
            }

        ]
        
        def select_mode(mode_value):
            nonlocal selected_mode_value
            selected_mode_value = mode_value
            mode_dialog.destroy()
        
        for i, mode in enumerate(mode_data):
            # Card frame
            card = tk.Frame(modes_container, bg=mode["bg_color"], 
                           relief=tk.RAISED, bd=3)
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0 if i == 0 else 25, 0))
            
            # Make entire card clickable
            def make_card_click_handler(value):
                return lambda e: select_mode(value)
            
            click_handler = make_card_click_handler(mode["value"])
            
            # Card content
            content_frame = tk.Frame(card, padx=45, pady=40, bg=mode["bg_color"])
            content_frame.pack(fill=tk.BOTH, expand=True)
            
            # Emoji at top
            emoji_label = tk.Label(content_frame, text=mode["emoji"], 
                                  font=('Arial', 64), 
                                  bg=mode["bg_color"])
            emoji_label.pack()
            
            # Title
            title_label = tk.Label(content_frame, text=mode["title"], 
                                  font=('Arial', 32, 'bold'),
                                  fg=mode["text_color"], bg=mode["bg_color"])
            title_label.pack(pady=(12, 0))
            
            # Subtitle
            tk.Label(content_frame, text=mode["subtitle"], 
                    font=('Arial', 20), 
                    fg=mode["feature_color"], bg=mode["bg_color"]).pack(pady=(2, 0))
            
            # Recommendation badge
            if mode["recommendation"]:
                rec_frame = tk.Frame(content_frame, bg=mode["bg_color"])
                rec_frame.pack(pady=(10, 0))
                rec_label = tk.Label(rec_frame, text=f"★ {mode['recommendation']} ★", 
                                   font=('Arial', 18, 'bold'), 
                                   fg=mode["accent_color"], bg=mode["bg_color"])
                rec_label.pack()
            else:
                # Empty space to maintain alignment
                tk.Label(content_frame, text=" ", font=('Arial', 18), 
                        bg=mode["bg_color"]).pack(pady=(10, 0))
            
            # Features list
            features_frame = tk.Frame(content_frame, bg=mode["bg_color"])
            features_frame.pack(pady=(20, 10), fill=tk.BOTH, expand=True)
            
            for feature in mode["features"]:
                feature_label = tk.Label(features_frame, text=feature, 
                                       font=('Arial', 18), 
                                       fg=mode["feature_color"], bg=mode["bg_color"],
                                       justify=tk.LEFT)
                feature_label.pack(anchor=tk.W, pady=4)
            
            # Create closure for each card's hover effects
            def create_hover_handlers(card, content_frame, mode, all_widgets):
                hover_state = {'active': False}
                original_bg = mode["bg_color"]
                hover_bg = mode["hover_color"]
                
                def on_enter(e):
                    if not hover_state['active']:
                        hover_state['active'] = True
                        card.config(bg=hover_bg)
                        content_frame.config(bg=hover_bg)
                        # Update all widgets that were captured
                        for widget in all_widgets:
                            try:
                                widget.config(bg=hover_bg)
                            except:
                                pass
                
                def on_leave(e):
                    if hover_state['active']:
                        hover_state['active'] = False
                        card.config(bg=original_bg)
                        content_frame.config(bg=original_bg)
                        # Restore all widgets that were captured
                        for widget in all_widgets:
                            try:
                                widget.config(bg=original_bg)
                            except:
                                pass
                
                return on_enter, on_leave

            # Collect ALL widgets that need background color changes
            all_widgets = []
            all_widgets.append(emoji_label)
            all_widgets.append(title_label)
            all_widgets.append(content_frame)
            all_widgets.extend([child for child in content_frame.winfo_children() if isinstance(child, (tk.Label, tk.Frame))])
            all_widgets.append(features_frame)
            all_widgets.extend([child for child in features_frame.winfo_children() if isinstance(child, tk.Label)])
            if mode["recommendation"]:
                all_widgets.append(rec_frame)
                all_widgets.append(rec_label)

            # Get handlers for this specific card with ALL widgets captured
            on_enter, on_leave = create_hover_handlers(card, content_frame, mode, all_widgets)

            # Bind events to all interactive elements
            interactive_widgets = [card, content_frame, emoji_label, title_label, features_frame] + list(features_frame.winfo_children())
            for widget in interactive_widgets:
                widget.bind("<Enter>", on_enter)
                widget.bind("<Leave>", on_leave)
                widget.bind("<Button-1>", click_handler)
                if hasattr(widget, 'config'):
                    widget.config(cursor='hand2')
            
            # Make features clickable too
            for child in features_frame.winfo_children():
                child.bind("<Enter>", on_enter)
                child.bind("<Leave>", on_leave)
                child.bind("<Button-1>", click_handler)
                child.config(cursor='hand2')
        
        # Make dialog modal
        mode_dialog.grab_set()
        
        # Handle window close (X button)
        def on_close():
            nonlocal selected_mode_value
            selected_mode_value = None
            mode_dialog.destroy()
        
        mode_dialog.protocol("WM_DELETE_WINDOW", on_close)
        
        # Center the dialog and show smoothly
        mode_dialog.update_idletasks()
        screen_width = mode_dialog.winfo_screenwidth()
        screen_height = mode_dialog.winfo_screenheight()
        window_width = mode_dialog.winfo_width()
        window_height = mode_dialog.winfo_height()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        mode_dialog.geometry(f"+{x}+{y}")
        
        # Smooth opening effect
        mode_dialog.deiconify()
        mode_dialog.lift()
        mode_dialog.focus_force()
        
        mode_dialog.wait_window()
        
        # Check if user selected a mode
        if selected_mode_value is None:
            self.append_log("⚠️ QA scan canceled.")
            return
        
        # Now get the folder
        folder_path = filedialog.askdirectory(title="Select Folder with HTML Files")
        if not folder_path:
            self.append_log("⚠️ QA scan canceled.")
            return
        
        mode = selected_mode_value
        self.append_log(f"🔍 Starting QA scan in {mode.upper()} mode for folder: {folder_path}")
        self.stop_requested = False
        
        def run_scan():
            self.master.after(0, self.update_run_button)
            self.qa_button.config(text="Stop Scan", command=self.stop_qa_scan, bootstyle="danger")
            
            try:
                # Call scan_html_folder with the mode parameter
                scan_html_folder(folder_path, log=self.append_log, stop_flag=lambda: self.stop_requested, mode=mode)
                self.append_log("✅ QA scan completed successfully.")
            except Exception as e:
                self.append_log(f"❌ QA scan error: {e}")
            finally:
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
            self.token_limit_entry.config(state=tk.DISABLED)
            self.toggle_token_btn.config(text="Enable Input Token Limit", bootstyle="success-outline")
            self.append_log("⚠️ Input token limit disabled - both translation and glossary extraction will process chapters of any size.")
            self.token_limit_disabled = True
        else:
            self.token_limit_entry.config(state=tk.NORMAL)
            if not self.token_limit_entry.get().strip():
                self.token_limit_entry.insert(0, str(self.config.get('token_limit', 1000000)))
            self.toggle_token_btn.config(text="Disable Input Token Limit", bootstyle="danger-outline")
            self.append_log(f"✅ Input token limit enabled: {self.token_limit_entry.get()} tokens (applies to both translation and glossary extraction)")
            self.token_limit_disabled = False

    def update_run_button(self):
        """Switch Run↔Stop depending on whether a process is active."""
        translation_running = hasattr(self, 'translation_thread') and self.translation_thread and self.translation_thread.is_alive()
        glossary_running = hasattr(self, 'glossary_thread') and self.glossary_thread and self.glossary_thread.is_alive()
        qa_running = hasattr(self, 'qa_thread') and self.qa_thread and self.qa_thread.is_alive()
        epub_running = hasattr(self, 'epub_thread') and self.epub_thread and self.epub_thread.is_alive()
        
        any_process_running = translation_running or glossary_running or qa_running or epub_running
        
        # Translation button
        if translation_running:
            self.run_button.config(text="Stop Translation", command=self.stop_translation,
                                 bootstyle="danger", state=tk.NORMAL)
        else:
            self.run_button.config(text="Run Translation", command=self.run_translation_thread,
                                 bootstyle="success", state=tk.NORMAL if translation_main and not any_process_running else tk.DISABLED)
        
        # Glossary button
        if hasattr(self, 'glossary_button'):
            if glossary_running:
                self.glossary_button.config(text="Stop Glossary", command=self.stop_glossary_extraction,
                                          bootstyle="danger", state=tk.NORMAL)
            else:
                self.glossary_button.config(text="Extract Glossary", command=self.run_glossary_extraction_thread,
                                          bootstyle="warning", state=tk.NORMAL if glossary_main and not any_process_running else tk.DISABLED)
        
        # EPUB button
        if hasattr(self, 'epub_button'):
            if epub_running:
                self.epub_button.config(text="Stop EPUB", command=self.stop_epub_converter,
                                      bootstyle="danger", state=tk.NORMAL)
            else:
                self.epub_button.config(text="EPUB Converter", command=self.epub_converter,
                                      bootstyle="info", state=tk.NORMAL if fallback_compile_epub and not any_process_running else tk.DISABLED)
        
        # QA button
        if hasattr(self, 'qa_button'):
            self.qa_button.config(state=tk.NORMAL if scan_html_folder and not any_process_running else tk.DISABLED)

    def stop_translation(self):
        """Stop translation while preserving loaded file"""
        current_file = self.entry_epub.get() if hasattr(self, 'entry_epub') else None
        
        self.stop_requested = True
        if translation_stop_flag:
            translation_stop_flag(True)
        
        try:
            import TransateKRtoEN
            if hasattr(TransateKRtoEN, 'set_stop_flag'):
                TransateKRtoEN.set_stop_flag(True)
        except: pass
        
        self.append_log("❌ Translation stop requested.")
        self.append_log("⏳ Please wait... stopping after current operation completes.")
        self.update_run_button()
        
        if current_file and hasattr(self, 'entry_epub'):
            self.master.after(100, lambda: self.preserve_file_path(current_file))

    def preserve_file_path(self, file_path):
        """Helper to ensure file path stays in the entry field"""
        if hasattr(self, 'entry_epub') and file_path:
            current = self.entry_epub.get()
            if not current or current != file_path:
                self.entry_epub.delete(0, tk.END)
                self.entry_epub.insert(0, file_path)

    def stop_glossary_extraction(self):
        """Stop glossary extraction specifically"""
        self.stop_requested = True
        if glossary_stop_flag:
            glossary_stop_flag(True)
        
        try:
            import extract_glossary_from_epub
            if hasattr(extract_glossary_from_epub, 'set_stop_flag'):
                extract_glossary_from_epub.set_stop_flag(True)
        except: pass
        
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
            at_bottom = self.log_text.yview()[1] >= 0.98
            is_memory = any(keyword in message for keyword in ['[MEMORY]', '📝', 'rolling summary', 'memory'])
            
            if is_memory:
                self.log_text.insert(tk.END, message + "\n", "memory")
                if "memory" not in self.log_text.tag_names():
                    self.log_text.tag_config("memory", foreground="#4CAF50", font=('TkDefaultFont', 10, 'italic'))
            else:
                self.log_text.insert(tk.END, message + "\n")
            
            if at_bottom:
                self.log_text.see(tk.END)
        
        if threading.current_thread() is threading.main_thread():
            _append()
        else:
            self.master.after(0, _append)

    def update_status_line(self, message, progress_percent=None):
        """Update a status line in the log"""
        def _update():
            content = self.log_text.get("1.0", "end-1c")
            lines = content.split('\n')
            
            status_markers = ['⏳', '📊', '✅', '❌', '🔄']
            is_status_line = False
            
            if lines and any(lines[-1].strip().startswith(marker) for marker in status_markers):
                is_status_line = True
            
            if progress_percent is not None:
                bar_width = 10
                filled = int(bar_width * progress_percent / 100)
                bar = "▓" * filled + "░" * (bar_width - filled)
                status_msg = f"⏳ {message} [{bar}] {progress_percent:.1f}%"
            else:
                status_msg = f"📊 {message}"
            
            if is_status_line and lines[-1].strip().startswith(('⏳', '📊')):
                start_pos = f"{len(lines)}.0"
                self.log_text.delete(f"{start_pos} linestart", "end")
                if len(lines) > 1:
                    self.log_text.insert("end", "\n" + status_msg)
                else:
                    self.log_text.insert("end", status_msg)
            else:
                if content and not content.endswith('\n'):
                    self.log_text.insert("end", "\n" + status_msg)
                else:
                    self.log_text.insert("end", status_msg + "\n")
            
            self.log_text.see("end")
        
        if threading.current_thread() is threading.main_thread():
            _update()
        else:
            self.master.after(0, _update)

    def append_chunk_progress(self, chunk_num, total_chunks, chunk_type="text", chapter_info="", 
                            overall_current=None, overall_total=None, extra_info=None):
        """Append chunk progress with enhanced visual indicator"""
        progress_bar_width = 20
        
        overall_progress = 0
        if overall_current is not None and overall_total is not None and overall_total > 0:
            overall_progress = overall_current / overall_total
        
        overall_filled = int(progress_bar_width * overall_progress)
        overall_bar = "█" * overall_filled + "░" * (progress_bar_width - overall_filled)
        
        if total_chunks == 1:
            icon = "📄" if chunk_type == "text" else "🖼️"
            msg_parts = [f"{icon} {chapter_info}"]
            
            if extra_info:
                msg_parts.append(f"[{extra_info}]")
            
            if overall_current is not None and overall_total is not None:
                msg_parts.append(f"\n    Progress: [{overall_bar}] {overall_current}/{overall_total} ({overall_progress*100:.1f}%)")
                
                if hasattr(self, '_chunk_start_times'):
                    if overall_current > 1:
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
                    self._translation_start_time = time.time()
                    self._chunk_start_times = {}
            
            msg = " ".join(msg_parts)
        else:
            chunk_progress = chunk_num / total_chunks if total_chunks > 0 else 0
            chunk_filled = int(progress_bar_width * chunk_progress)
            chunk_bar = "█" * chunk_filled + "░" * (progress_bar_width - chunk_filled)
            
            icon = "📄" if chunk_type == "text" else "🖼️"
            
            msg_parts = [f"{icon} {chapter_info}"]
            msg_parts.append(f"\n    Chunk: [{chunk_bar}] {chunk_num}/{total_chunks} ({chunk_progress*100:.1f}%)")
            
            if overall_current is not None and overall_total is not None:
                msg_parts.append(f"\n    Overall: [{overall_bar}] {overall_current}/{overall_total} ({overall_progress*100:.1f}%)")
            
            msg = "".join(msg_parts)
        
        if hasattr(self, '_chunk_start_times'):
            self._chunk_start_times[f"{chapter_info}_{chunk_num}"] = time.time()
        
        self.append_log(msg)

    def _block_editing(self, event):
        """Block editing in log text but allow selection and copying"""
        if event.state & 0x4 and event.keysym.lower() == 'c':
            return None
        if event.state & 0x4 and event.keysym.lower() == 'a':
            self.log_text.tag_add(tk.SEL, "1.0", tk.END)
            self.log_text.mark_set(tk.INSERT, "1.0")
            self.log_text.see(tk.INSERT)
            return "break"
        if event.keysym in ['Left', 'Right', 'Up', 'Down', 'Home', 'End', 'Prior', 'Next']:
            return None
        if event.state & 0x1:
            return None
        return "break"

    def _show_context_menu(self, event):
        """Show context menu for log text"""
        try:
            context_menu = tk.Menu(self.master, tearoff=0)
            
            try:
                self.log_text.selection_get()
                context_menu.add_command(label="Copy", command=self.copy_selection)
            except tk.TclError:
                context_menu.add_command(label="Copy", state="disabled")
            
            context_menu.add_separator()
            context_menu.add_command(label="Select All", command=self.select_all_log)
            
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()

    def copy_selection(self):
        """Copy selected text from log to clipboard"""
        try:
            text = self.log_text.selection_get()
            self.master.clipboard_clear()
            self.master.clipboard_append(text)
        except tk.TclError:
            pass

    def select_all_log(self):
        """Select all text in the log"""
        self.log_text.tag_add(tk.SEL, "1.0", tk.END)
        self.log_text.mark_set(tk.INSERT, "1.0")
        self.log_text.see(tk.INSERT)

    def auto_load_glossary_for_file(self, file_path):
        """Automatically load glossary if it exists in the output folder"""
        if not file_path or not os.path.isfile(file_path):
            return
        
        if not file_path.lower().endswith('.epub'):
            return
        
        file_base = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = file_base
        
        glossary_candidates = [
            os.path.join(output_dir, "glossary.json"),
            os.path.join(output_dir, f"{file_base}_glossary.json"),
            os.path.join(output_dir, "Glossary", f"{file_base}_glossary.json")
        ]
        
        for glossary_path in glossary_candidates:
            if os.path.exists(glossary_path):
                try:
                    with open(glossary_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if data:
                        self.manual_glossary_path = glossary_path
                        self.append_log(f"📑 Auto-loaded glossary: {os.path.basename(glossary_path)}")
                        return True
                except Exception:
                    continue
        
        return False

    def browse_file(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Supported files", "*.epub;*.txt"),
                ("EPUB files", "*.epub"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.entry_epub.delete(0, tk.END)
            self.entry_epub.insert(0, path)
            
            if path.lower().endswith('.epub'):
                self.auto_load_glossary_for_file(path)

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
        load_application_icon(dialog, self.base_dir)
        
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
        
        self.summary_system_text = scrolledtext.ScrolledText(system_frame, height=5, wrap=tk.WORD,
                                                           undo=True, autoseparators=True, maxundo=-1)
        self.summary_system_text.pack(fill=tk.BOTH, expand=True)
        self.summary_system_text.insert('1.0', self.rolling_summary_system_prompt)
        self._setup_text_undo_redo(self.summary_system_text)
        
        user_frame = tk.LabelFrame(main_frame, text="User Prompt Template", padx=10, pady=10)
        user_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        tk.Label(user_frame, text="Template for summary requests. Use {translations} for content placeholder",
                font=('TkDefaultFont', 9), fg='blue').pack(anchor=tk.W, pady=(0, 5))
        
        self.summary_user_text = scrolledtext.ScrolledText(user_frame, height=12, wrap=tk.WORD,
                                                          undo=True, autoseparators=True, maxundo=-1)
        self.summary_user_text.pack(fill=tk.BOTH, expand=True)
        self.summary_user_text.insert('1.0', self.rolling_summary_user_prompt)
        self._setup_text_undo_redo(self.summary_user_text)
        
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

    def open_other_settings(self):
        """Open the Other Settings dialog"""
        top = tk.Toplevel(self.master)
        top.title("Other Settings")
        
        screen_width = top.winfo_screenwidth()
        screen_height = top.winfo_screenheight()
        
        initial_width = 0
        initial_height = 1460
        x = (screen_width - initial_width) // 2
        y = max(20, (screen_height - initial_height) // 2)
        
        top.geometry(f"{initial_width}x{initial_height}+{x}+{y}")
        top.withdraw()
        top.transient(self.master)
        load_application_icon(top, self.base_dir)
        
        self._settings_window = top
        
        main_container = tk.Frame(top)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        content_area = tk.Frame(main_container)
        content_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        canvas = tk.Canvas(content_area, bg='white')
        scrollbar = ttk.Scrollbar(content_area, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
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
        
        # Save & Close buttons
        self._create_settings_buttons(scrollable_frame, top, canvas)
        
        # Show window
        top.after(50, lambda: [top.update_idletasks(), top.deiconify()])
        
        cleanup_bindings = self._setup_dialog_scrolling(top, canvas)
        top.protocol("WM_DELETE_WINDOW", lambda: [cleanup_bindings(), top.destroy()])
        
        self._auto_resize_dialog(top, canvas, max_width_ratio=0.7, max_height_ratio=0.8)

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
        ttk.Combobox(row1, textvariable=self.summary_role_var,
                    values=["user", "system"], state="readonly", width=10).pack(side=tk.LEFT, padx=(0, 30))
        
        tk.Label(row1, text="Mode:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Combobox(row1, textvariable=self.rolling_summary_mode_var,
                    values=["append", "replace"], state="readonly", width=10).pack(side=tk.LEFT, padx=(0, 10))
        
        row2 = tk.Frame(settings_frame)
        row2.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(row2, text="Summarize last").pack(side=tk.LEFT, padx=(0, 5))
        tb.Entry(row2, width=5, textvariable=self.rolling_summary_exchanges_var).pack(side=tk.LEFT, padx=(0, 5))
        tk.Label(row2, text="exchanges").pack(side=tk.LEFT)
        
        tb.Button(content_frame, text="⚙️ Configure Memory Prompts", 
                 command=self.configure_rolling_summary_prompts,
                 bootstyle="info-outline", width=30).pack(anchor=tk.W, padx=20, pady=(10, 10))
        
        ttk.Separator(section_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 10))
        
        tk.Label(section_frame, text="💡 Memory Mode:\n"
                "• Append: Keeps adding summaries (longer context)\n"
                "• Replace: Only keeps latest summary (concise)",
                font=('TkDefaultFont', 11), fg='#666', justify=tk.LEFT).pack(anchor=tk.W, padx=5, pady=(0, 5))

    def _create_response_handling_section(self, parent):
        """Create response handling section with AI Hunter additions"""
        section_frame = tk.LabelFrame(parent, text="Response Handling & Retry Logic", padx=10, pady=10)
        section_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=5)  # Fixed: row=1, column=0
        
        # Retry Truncated
        tb.Checkbutton(section_frame, text="Auto-retry Truncated Responses", 
                      variable=self.retry_truncated_var,
                      bootstyle="round-toggle").pack(anchor=tk.W)
        
        retry_frame = tk.Frame(section_frame)
        retry_frame.pack(anchor=tk.W, padx=20, pady=(5, 5))
        tk.Label(retry_frame, text="Max retry tokens:").pack(side=tk.LEFT)
        tb.Entry(retry_frame, width=8, textvariable=self.max_retry_tokens_var).pack(side=tk.LEFT, padx=5)
        
        tk.Label(section_frame, text="Automatically retry when API response\nis cut off due to token limits",
                font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
        
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
        
        # NEW: Detection Method subsection
        method_label = tk.Label(section_frame, text="Detection Method:", font=('TkDefaultFont', 10, 'bold'))
        method_label.pack(anchor=tk.W, padx=20, pady=(10, 5))
        
        methods = [
            ("basic", "Basic (Fast) - Original 85% threshold, 1000 chars"),
            ("ai-hunter", "AI Hunter - Multi-method semantic analysis"),
            ("cascading", "Cascading - Basic first, then AI Hunter")
        ]
        
        for value, text in methods:
            rb = tb.Radiobutton(section_frame, text=text, variable=self.duplicate_detection_mode_var, 
                               value=value, bootstyle="primary")
            rb.pack(anchor=tk.W, padx=40, pady=2)
        
        # NEW: AI Hunter Custom Threshold
        threshold_frame = tk.Frame(section_frame)
        threshold_frame.pack(anchor=tk.W, padx=20, pady=(10, 5))
        
        tk.Label(threshold_frame, text="AI Hunter Threshold:", font=('TkDefaultFont', 10, 'bold')).pack(side=tk.LEFT)
        
        # Create custom threshold entry
        self.ai_hunter_threshold_var = tk.StringVar(value=str(self.config.get('ai_hunter_threshold', 75)))
        tb.Entry(threshold_frame, width=6, textvariable=self.ai_hunter_threshold_var).pack(side=tk.LEFT, padx=(10, 5))
        tk.Label(threshold_frame, text="%").pack(side=tk.LEFT)
        
        tk.Label(section_frame, text="Custom similarity threshold for AI Hunter detection\n"
                "Lower = more sensitive (more duplicates found)\n"
                "Higher = less sensitive (fewer false positives)\n"
                "Recommended: 40-90%",
                font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack
        
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

    def _create_prompt_management_section(self, parent):
        """Create meta data section (formerly prompt management)"""
        section_frame = tk.LabelFrame(parent, text="Meta Data", padx=10, pady=10)
        section_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=(10, 5))
        
        
        title_frame = tk.Frame(section_frame)
        title_frame.pack(anchor=tk.W, pady=(10, 10))
        
        tb.Checkbutton(title_frame, text="Translate Book Title", 
                      variable=self.translate_book_title_var,
                      bootstyle="round-toggle").pack(side=tk.LEFT)
        
        tb.Button(title_frame, text="Configure Title Prompt", 
                 command=self.configure_title_prompt,
                 bootstyle="info-outline", width=20).pack(side=tk.LEFT, padx=(10, 0))
        
        tk.Label(section_frame, text="When enabled: Book titles will be translated to English\n"
                    "When disabled: Book titles remain in original language",
                    font=('TkDefaultFont', 11), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
            
        # EPUB Validation (moved from Processing Options)
        ttk.Separator(section_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 10))
        
        tk.Label(section_frame, text="EPUB Utilities:", font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W, pady=(5, 5))
        
        tb.Button(section_frame, text="🔍 Validate EPUB Structure", 
                 command=self.validate_epub_structure_gui, 
                 bootstyle="success-outline",
                 width=25).pack(anchor=tk.W, pady=2)
        
        tk.Label(section_frame, text="Check if all required EPUB files are\npresent for compilation",
                font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 5))

    def _create_processing_options_section(self, parent):
        """Create processing options section"""
        section_frame = tk.LabelFrame(parent, text="Processing Options", padx=10, pady=10)
        section_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=5)
        
        # Reinforce messages option (moved from Meta Data)
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
        
        tb.Checkbutton(section_frame, text="Reset Failed Chapters on Start", 
                      variable=self.reset_failed_chapters_var,
                      bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
        
        tk.Label(section_frame, text="Automatically retry failed/deleted chapters\non each translation run",
                font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
        
        
        tb.Checkbutton(section_frame, text="Comprehensive Chapter Extraction", 
                      variable=self.comprehensive_extraction_var,
                      bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
        
        tk.Label(section_frame, text="Extract ALL files (disable smart filtering)",
                font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
        
        tb.Checkbutton(section_frame, text="Disable Image Gallery in EPUB", 
                      variable=self.disable_epub_gallery_var,
                      bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
        
        tk.Label(section_frame, text="Skip creating image gallery page in EPUB",
                font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
        
        tb.Checkbutton(section_frame, text="Disable 0-based Chapter Detection", 
                      variable=self.disable_zero_detection_var,
                      bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
        
        tk.Label(section_frame, text="Always use chapter ranges as specified\n(don't adjust for 0-based novels)",
                font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))

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
                font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, padx=20)
        
        # Right column
        settings_frame = tk.Frame(right_column)
        settings_frame.pack(fill=tk.X)
        
        settings_frame.grid_columnconfigure(1, minsize=80)
        
        settings = [
            ("Min Image height (px):", self.webnovel_min_height_var),
            ("Max Images per chapter:", self.max_images_per_chapter_var),
            ("Output token Limit:", self.image_max_tokens_var),
            ("Chunk height:", self.image_chunk_height_var)
        ]
        
        for row, (label, var) in enumerate(settings):
            tk.Label(settings_frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
            tb.Entry(settings_frame, width=10, textvariable=var).grid(row=row, column=1, sticky=tk.W, pady=3)
        
        tk.Label(right_column, text="💡 Supported models:\n"
                "• Gemini 1.5 Pro/Flash, 2.0 Flash\n"
                "• GPT-4V, GPT-4o, o4-mini",
                font=('TkDefaultFont', 10), fg='#666', justify=tk.LEFT).pack(anchor=tk.W, pady=(10, 0))

    def _create_settings_buttons(self, parent, dialog, canvas):
        """Create save and close buttons for settings dialog"""
        button_frame = tk.Frame(parent)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(10, 10))
        
        button_container = tk.Frame(button_frame)
        button_container.pack(expand=True)
        
        def save_and_close():
            try:
                def safe_int(value, default):
                    try: return int(value)
                    except (ValueError, TypeError): return default
                
                # Save all settings
                self.config.update({
                    'use_rolling_summary': self.rolling_summary_var.get(),
                    'summary_role': self.summary_role_var.get(),
                    'rolling_summary_exchanges': safe_int(self.rolling_summary_exchanges_var.get(), 5),
                    'rolling_summary_mode': self.rolling_summary_mode_var.get(),
                    'retry_truncated': self.retry_truncated_var.get(),
                    'max_retry_tokens': safe_int(self.max_retry_tokens_var.get(), 16384),
                    'retry_duplicate_bodies': self.retry_duplicate_var.get(),
                    'duplicate_lookback_chapters': safe_int(self.duplicate_lookback_var.get(), 5),
                    'retry_timeout': self.retry_timeout_var.get(),
                    'chunk_timeout': safe_int(self.chunk_timeout_var.get(), 900),
                    'reinforcement_frequency': safe_int(self.reinforcement_freq_var.get(), 10),
                    'translate_book_title': self.translate_book_title_var.get(),
                    'book_title_prompt': getattr(self, 'book_title_prompt', 
                        "Translate this book title to English while retaining any acronyms:"),
                    'emergency_paragraph_restore': self.emergency_restore_var.get(),
                    'reset_failed_chapters': self.reset_failed_chapters_var.get(),
                    'comprehensive_extraction': self.comprehensive_extraction_var.get(),
                    'disable_epub_gallery': self.disable_epub_gallery_var.get(),
                    'disable_zero_detection': self.disable_zero_detection_var.get(),
                    'enable_image_translation': self.enable_image_translation_var.get(),
                    'process_webnovel_images': self.process_webnovel_images_var.get(),
                    'hide_image_translation_label': self.hide_image_translation_label_var.get(),
                    'duplicate_detection_mode': self.duplicate_detection_mode_var.get(),
                    'ai_hunter_threshold': safe_int(self.ai_hunter_threshold_var.get(), 75)
                })
                
                # Validate numeric fields
                numeric_fields = [
                    ('webnovel_min_height', self.webnovel_min_height_var, 1000),
                    ('image_max_tokens', self.image_max_tokens_var, 16384),
                    ('max_images_per_chapter', self.max_images_per_chapter_var, 1),
                    ('image_chunk_height', self.image_chunk_height_var, 1500)
                ]
                
                for field_name, var, default in numeric_fields:
                    value = var.get().strip()
                    if value and not value.isdigit():
                        messagebox.showerror("Invalid Input", 
                            f"Please enter a valid number for {field_name.replace('_', ' ').title()}")
                        return
                
                for field_name, var, default in numeric_fields:
                    self.config[field_name] = safe_int(var.get(), default)
                
                # Update environment variables
                env_updates = {
                    "USE_ROLLING_SUMMARY": "1" if self.rolling_summary_var.get() else "0",
                    "SUMMARY_ROLE": self.summary_role_var.get(),
                    "ROLLING_SUMMARY_EXCHANGES": str(self.config['rolling_summary_exchanges']),
                    "ROLLING_SUMMARY_MODE": self.rolling_summary_mode_var.get(),
                    "ROLLING_SUMMARY_SYSTEM_PROMPT": self.rolling_summary_system_prompt,
                    "ROLLING_SUMMARY_USER_PROMPT": self.rolling_summary_user_prompt,
                    "RETRY_TRUNCATED": "1" if self.retry_truncated_var.get() else "0",
                    "MAX_RETRY_TOKENS": str(self.config['max_retry_tokens']),
                    "RETRY_DUPLICATE_BODIES": "1" if self.retry_duplicate_var.get() else "0",
                    "DUPLICATE_LOOKBACK_CHAPTERS": str(self.config['duplicate_lookback_chapters']),
                    "RETRY_TIMEOUT": "1" if self.retry_timeout_var.get() else "0",
                    "CHUNK_TIMEOUT": str(self.config['chunk_timeout']),
                    "REINFORCEMENT_FREQUENCY": str(self.config['reinforcement_frequency']),
                    "TRANSLATE_BOOK_TITLE": "1" if self.translate_book_title_var.get() else "0",
                    "BOOK_TITLE_PROMPT": self.book_title_prompt,
                    "EMERGENCY_PARAGRAPH_RESTORE": "1" if self.emergency_restore_var.get() else "0",
                    "RESET_FAILED_CHAPTERS": "1" if self.reset_failed_chapters_var.get() else "0",
                    "COMPREHENSIVE_EXTRACTION": "1" if self.comprehensive_extraction_var.get() else "0",
                    "ENABLE_IMAGE_TRANSLATION": "1" if self.enable_image_translation_var.get() else "0",
                    "PROCESS_WEBNOVEL_IMAGES": "1" if self.process_webnovel_images_var.get() else "0",
                    "WEBNOVEL_MIN_HEIGHT": str(self.config['webnovel_min_height']),
                    "IMAGE_MAX_TOKENS": str(self.config['image_max_tokens']),
                    "MAX_IMAGES_PER_CHAPTER": str(self.config['max_images_per_chapter']),
                    "IMAGE_CHUNK_HEIGHT": str(self.config['image_chunk_height']),
                    "HIDE_IMAGE_TRANSLATION_LABEL": "1" if self.hide_image_translation_label_var.get() else "0",
                    "DISABLE_EPUB_GALLERY": "1" if self.disable_epub_gallery_var.get() else "0",
                    "DISABLE_ZERO_DETECTION": "1" if self.disable_zero_detection_var.get() else "0",
                    "DUPLICATE_DETECTION_MODE": self.duplicate_detection_mode_var.get(),
                    "AI_HUNTER_THRESHOLD": self.ai_hunter_threshold_var.get()
                }
                os.environ.update(env_updates)
                
                with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                
                self.append_log("✅ Other Settings saved successfully")
                dialog.destroy()
                
            except Exception as e:
                print(f"❌ Failed to save Other Settings: {e}")
                messagebox.showerror("Error", f"Failed to save settings: {e}")
        
        cleanup_bindings = self._setup_dialog_scrolling(dialog, canvas)
        
        tb.Button(button_container, text="💾 Save Settings", command=save_and_close, 
                 bootstyle="success", width=20).pack(side=tk.LEFT, padx=5)
        
        tb.Button(button_container, text="❌ Cancel", command=lambda: [cleanup_bindings(), dialog.destroy()], 
                 bootstyle="secondary", width=20).pack(side=tk.LEFT, padx=5)

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
        
        self.append_glossary_var.set(True)
        self.append_log("✅ Automatically enabled 'Append Glossary to System Prompt'")

    def save_config(self):
        """Persist all settings to config.json."""
        try:
            def safe_int(value, default):
                try: return int(value)
                except (ValueError, TypeError): return default
            
            def safe_float(value, default):
                try: return float(value)
                except (ValueError, TypeError): return default
            
            # Basic settings
            self.config['model'] = self.model_var.get()
            self.config['active_profile'] = self.profile_var.get()
            self.config['prompt_profiles'] = self.prompt_profiles
            self.config['contextual'] = self.contextual_var.get()
            
            # Validate numeric fields
            delay_val = self.delay_entry.get().strip()
            if delay_val and not delay_val.replace('.', '', 1).isdigit():
                messagebox.showerror("Invalid Input", "Please enter a valid number for API call delay")
                return
            self.config['delay'] = safe_int(delay_val, 2)
            
            trans_temp_val = self.trans_temp.get().strip()
            if trans_temp_val:
                try: float(trans_temp_val)
                except ValueError:
                    messagebox.showerror("Invalid Input", "Please enter a valid number for Temperature")
                    return
            self.config['translation_temperature'] = safe_float(trans_temp_val, 0.3)
            
            trans_history_val = self.trans_history.get().strip()
            if trans_history_val and not trans_history_val.isdigit():
                messagebox.showerror("Invalid Input", "Please enter a valid number for Translation History Limit")
                return
            self.config['translation_history_limit'] = safe_int(trans_history_val, 3)
            
            # Save all other settings
            self.config['api_key'] = self.api_key_entry.get()
            self.config['REMOVE_AI_ARTIFACTS'] = self.REMOVE_AI_ARTIFACTS_var.get()
            self.config['chapter_range'] = self.chapter_range_entry.get().strip()
            self.config['use_rolling_summary'] = self.rolling_summary_var.get()
            self.config['summary_role'] = self.summary_role_var.get()
            self.config['max_output_tokens'] = self.max_output_tokens
            self.config['translate_book_title'] = self.translate_book_title_var.get()
            self.config['book_title_prompt'] = self.book_title_prompt
            self.config['append_glossary'] = self.append_glossary_var.get()
            self.config['emergency_paragraph_restore'] = self.emergency_restore_var.get()
            self.config['reinforcement_frequency'] = safe_int(self.reinforcement_freq_var.get(), 10)
            self.config['reset_failed_chapters'] = self.reset_failed_chapters_var.get()
            self.config['retry_duplicate_bodies'] = self.retry_duplicate_var.get()
            self.config['duplicate_lookback_chapters'] = safe_int(self.duplicate_lookback_var.get(), 5)
            self.config['token_limit_disabled'] = self.token_limit_disabled
            self.config['glossary_min_frequency'] = safe_int(self.glossary_min_frequency_var.get(), 2)
            self.config['glossary_max_names'] = safe_int(self.glossary_max_names_var.get(), 50)
            self.config['glossary_max_titles'] = safe_int(self.glossary_max_titles_var.get(), 30)
            self.config['glossary_batch_size'] = safe_int(self.glossary_batch_size_var.get(), 50)
            self.config['enable_image_translation'] = self.enable_image_translation_var.get()
            self.config['process_webnovel_images'] = self.process_webnovel_images_var.get()
            self.config['webnovel_min_height'] = safe_int(self.webnovel_min_height_var.get(), 1000)
            self.config['image_max_tokens'] = safe_int(self.image_max_tokens_var.get(), 16384)
            self.config['max_images_per_chapter'] = safe_int(self.max_images_per_chapter_var.get(), 1)
            self.config['batch_translation'] = self.batch_translation_var.get()
            self.config['batch_size'] = safe_int(self.batch_size_var.get(), 3)
            self.config['translation_history_rolling'] = self.translation_history_rolling_var.get()
            self.config['glossary_history_rolling'] = self.glossary_history_rolling_var.get()
            self.config['disable_epub_gallery'] = self.disable_epub_gallery_var.get()
            self.config['enable_auto_glossary'] = self.enable_auto_glossary_var.get()
            self.config['duplicate_detection_mode'] = self.duplicate_detection_mode_var.get()
            self.config['ai_hunter_threshold'] = safe_int(self.ai_hunter_threshold_var.get(), 75)

            
            _tl = self.token_limit_entry.get().strip()
            if _tl.isdigit():
                self.config['token_limit'] = int(_tl)
            else:
                self.config['token_limit'] = None
            
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Saved", "Configuration saved.")
            self.append_log("✅ Configuration saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config: {e}")
            self.append_log(f"❌ Failed to save configuration: {e}")

    def log_debug(self, message):
        self.append_log(f"[DEBUG] {message}")

if __name__ == "__main__":
    import time
    
    print("🚀 Starting Glossarion v2.7.2...")
    
    # Initialize splash screen
    splash_manager = None
    try:
        from splash_utils import SplashManager
        splash_manager = SplashManager()
        splash_started = splash_manager.start_splash()
        
        if splash_started:
            splash_manager.update_status("Loading theme framework...")
            time.sleep(0.1)  # Reduced from 0.15 to 0.1
    except Exception as e:
        print(f"⚠️ Splash screen failed: {e}")
        splash_manager = None
    
    try:
        if splash_manager:
            splash_manager.update_status("Loading UI framework...")
            time.sleep(0.08)  # Reduced from 0.1 to 0.08
        
        # Import ttkbootstrap while splash is visible
        import ttkbootstrap as tb
        from ttkbootstrap.constants import *
        
        # REAL module loading during splash screen with gradual progression
        if splash_manager:
            # Create a custom callback function for splash updates
            def splash_callback(message):
                if splash_manager and splash_manager.splash_window:
                    splash_manager.update_status(message)
                    splash_manager.splash_window.update()
                    time.sleep(0.09)  # Reduced from 0.15 to 0.1 - faster but still visible
            
            # Actually load modules during splash with real feedback
            splash_callback("Loading translation modules...")
            
            # Import and test each module for real
            translation_main = translation_stop_flag = translation_stop_check = None
            glossary_main = glossary_stop_flag = glossary_stop_check = None
            fallback_compile_epub = scan_html_folder = None
            
            modules_loaded = 0
            total_modules = 4
            
            # Load TranslateKRtoEN
            splash_callback("Loading translation engine...")
            try:
                splash_callback("Validating translation engine...")
                import TransateKRtoEN
                if hasattr(TransateKRtoEN, 'main') and hasattr(TransateKRtoEN, 'set_stop_flag'):
                    from TransateKRtoEN import main as translation_main, set_stop_flag as translation_stop_flag, is_stop_requested as translation_stop_check
                    modules_loaded += 1
                    splash_callback("✅ translation engine loaded")
                else:
                    splash_callback("⚠️ translation engine incomplete")
            except Exception as e:
                splash_callback("❌ translation engine failed")
                print(f"Warning: Could not import TransateKRtoEN: {e}")
            
            # Load extract_glossary_from_epub
            splash_callback("Loading glossary extractor...")
            try:
                splash_callback("Validating glossary extractor...")
                import extract_glossary_from_epub
                if hasattr(extract_glossary_from_epub, 'main') and hasattr(extract_glossary_from_epub, 'set_stop_flag'):
                    from extract_glossary_from_epub import main as glossary_main, set_stop_flag as glossary_stop_flag, is_stop_requested as glossary_stop_check
                    modules_loaded += 1
                    splash_callback("✅ glossary extractor loaded")
                else:
                    splash_callback("⚠️ glossary extractor incomplete")
            except Exception as e:
                splash_callback("❌ glossary extractor failed")
                print(f"Warning: Could not import extract_glossary_from_epub: {e}")
            
            # Load epub_converter
            splash_callback("Loading EPUB converter...")
            try:
                import epub_converter
                if hasattr(epub_converter, 'fallback_compile_epub'):
                    from epub_converter import fallback_compile_epub
                    modules_loaded += 1
                    splash_callback("✅ EPUB converter loaded")
                else:
                    splash_callback("⚠️ EPUB converter incomplete")
            except Exception as e:
                splash_callback("❌ EPUB converter failed")
                print(f"Warning: Could not import epub_converter: {e}")
            
            # Load scan_html_folder
            splash_callback("Loading QA scanner...")
            try:
                import scan_html_folder
                if hasattr(scan_html_folder, 'scan_html_folder'):
                    from scan_html_folder import scan_html_folder
                    modules_loaded += 1
                    splash_callback("✅ QA scanner loaded")
                else:
                    splash_callback("⚠️ QA scanner incomplete")
            except Exception as e:
                splash_callback("❌ QA scanner failed")
                print(f"Warning: Could not import scan_html_folder: {e}")
            
            # Final status with pause for visibility
            splash_callback("Finalizing module initialization...")
            if modules_loaded == total_modules:
                splash_callback("✅ All modules loaded successfully")
            else:
                splash_callback(f"⚠️ {modules_loaded}/{total_modules} modules loaded")
            
            # Store loaded modules globally for GUI access
            import translator_gui
            translator_gui.translation_main = translation_main
            translator_gui.translation_stop_flag = translation_stop_flag  
            translator_gui.translation_stop_check = translation_stop_check
            translator_gui.glossary_main = glossary_main
            translator_gui.glossary_stop_flag = glossary_stop_flag
            translator_gui.glossary_stop_check = glossary_stop_check
            translator_gui.fallback_compile_epub = fallback_compile_epub
            translator_gui.scan_html_folder = scan_html_folder
        
        if splash_manager:
            splash_manager.update_status("Creating main window...")
            time.sleep(0.07)  # Reduced from 0.1 to 0.08
            
            # Extra pause to show "Ready!" before closing
            splash_manager.update_status("Ready!")
            time.sleep(0.1)  # Reduced from 0.2 to 0.15
            splash_manager.close_splash()
        
        # Create main window (modules already loaded)
        root = tb.Window(themename="darkly")
        
        # CRITICAL: Hide window immediately to prevent white flash
        root.withdraw()
        
        # Initialize the app (modules already available)  
        app = TranslatorGUI(root)
        
        # Mark modules as already loaded to skip lazy loading
        app._modules_loaded = True
        app._modules_loading = False
        
        # CRITICAL: Let all widgets and theme fully initialize
        root.update_idletasks()
        
        # CRITICAL: Now show the window after everything is ready
        root.deiconify()
        
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
        if splash_manager:
            try:
                splash_manager.close_splash()
            except:
                pass
