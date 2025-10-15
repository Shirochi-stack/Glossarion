#translator_gui.py
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

# Standard Library
import io, json, logging, math, os, shutil, sys, threading, time, re, concurrent.futures, signal
from logging.handlers import RotatingFileHandler
import atexit
import faulthandler
import platform

# PySide6 imports (replacing Tkinter)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
                                QTextEdit, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
                                QMenuBar, QMenu, QMessageBox, QFileDialog, QDialog,
                                QScrollArea, QTabWidget, QCheckBox, QComboBox, QSpinBox,
                                QSizePolicy, QSplitter, QProgressBar, QStyle, QToolButton, QGraphicsOpacityEffect)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QThread, QSize, QEvent, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QFont, QColor, QIcon, QTextCursor, QKeySequence, QAction, QTextCharFormat, QTransform

from ai_hunter_enhanced import AIHunterConfigGUI, ImprovedAIHunterDetection
import traceback
from splash_utils import SplashManager
from api_key_encryption import encrypt_config, decrypt_config
from metadata_batch_translator import MetadataBatchTranslatorUI
from model_options import get_model_options

# Support worker-mode dispatch in frozen builds to avoid requiring Python interpreter
# This allows spawning the same .exe with a special flag to run helper tasks.
if '--run-chapter-extraction' in sys.argv:
    try:
        # Ensure UTF-8 I/O in worker mode
        os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
        # Remove the flag so worker's argv aligns: argv[1]=epub, argv[2]=out, argv[3]=mode
        try:
            _flag_idx = sys.argv.index('--run-chapter-extraction')
            sys.argv = [sys.argv[0]] + sys.argv[_flag_idx + 1:]
        except ValueError:
            # Shouldn't happen, but continue with current argv
            pass
        from chapter_extraction_worker import main as _ce_main
        _ce_main()
    except Exception as _e:
        try:
            print(f"[ERROR] Worker failed: {_e}")
        except Exception:
            pass
    finally:
        # Make sure we exit without initializing the GUI when in worker mode
        sys.exit(0)

# The frozen check can stay here for other purposes
if getattr(sys, 'frozen', False):
    # Any other frozen-specific setup
    pass
    
# Manga translation support (optional)
try:
    from manga_integration import MangaTranslationTab
    MANGA_SUPPORT = True
except ImportError:
    MANGA_SUPPORT = False
    print("Manga translation modules not found.")

# Async processing support (lazy loaded)
ASYNC_SUPPORT = False
try:
    # Check if module exists without importing
    import importlib.util
    spec = importlib.util.find_spec('async_api_processor')
    if spec is not None:
        ASYNC_SUPPORT = True
except ImportError:
    pass
    
# Deferred modules
translation_main = translation_stop_flag = translation_stop_check = None
glossary_main = glossary_stop_flag = glossary_stop_check = None
fallback_compile_epub = scan_html_folder = None

CONFIG_FILE = "config.json"
BASE_WIDTH, BASE_HEIGHT = 1920, 1080

# --- Robust file logging and crash tracing setup ---
_FAULT_LOG_FH = None

def _setup_file_logging():
    """Initialize rotating file logging and crash tracing (faulthandler).
    Ensures logs directory is writable in both source and PyInstaller one-file builds.
    """
    global _FAULT_LOG_FH
    
    def _can_write(dir_path: str) -> bool:
        try:
            os.makedirs(dir_path, exist_ok=True)
            test_file = os.path.join(dir_path, ".write_test")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(test_file)
            return True
        except Exception:
            return False
    
    def _resolve_logs_dir() -> str:
        # 1) Explicit override
        env_dir = os.environ.get("GLOSSARION_LOG_DIR")
        if env_dir and _can_write(os.path.expanduser(env_dir)):
            return os.path.expanduser(env_dir)
        
        # 2) Next to the executable for frozen builds
        try:
            if getattr(sys, 'frozen', False) and hasattr(sys, 'executable'):
                exe_dir = os.path.dirname(sys.executable)
                candidate = os.path.join(exe_dir, "logs")
                if _can_write(candidate):
                    return candidate
        except Exception:
            pass
        
        # 3) User-local app data (always writable)
        try:
            base = os.environ.get('LOCALAPPDATA') or os.environ.get('APPDATA') or os.path.expanduser('~')
            candidate = os.path.join(base, 'Glossarion', 'logs')
            if _can_write(candidate):
                return candidate
        except Exception:
            pass
        
        # 4) Development: alongside source file
        try:
            base_dir = os.path.abspath(os.path.dirname(__file__))
            candidate = os.path.join(base_dir, "logs")
            if _can_write(candidate):
                return candidate
        except Exception:
            pass
        
        # 5) Last resort: current working directory
        fallback = os.path.join(os.getcwd(), "logs")
        os.makedirs(fallback, exist_ok=True)
        return fallback
    
    try:
        logs_dir = _resolve_logs_dir()
        # Export for helper modules (e.g., memory_usage_reporter)
        os.environ["GLOSSARION_LOG_DIR"] = logs_dir

        # Rotating log handler - use custom Windows-safe version
        log_file = os.path.join(logs_dir, "run.log")
        
        # On Windows, use a safer handler that doesn't fail on rotation errors
        if platform.system() == 'Windows':
            import glob
            
            class WindowsSafeRotatingFileHandler(RotatingFileHandler):
                """Windows-safe rotating file handler that gracefully handles file locking."""
                
                def doRollover(self):
                    """Override doRollover to handle Windows permission errors gracefully."""
                    if self.stream:
                        self.stream.close()
                        self.stream = None
                    
                    try:
                        # Try standard rotation
                        # Rotate existing backup files
                        for i in range(self.backupCount - 1, 0, -1):
                            sfn = self.rotation_filename("%s.%d" % (self.baseFilename, i))
                            dfn = self.rotation_filename("%s.%d" % (self.baseFilename, i + 1))
                            if os.path.exists(sfn):
                                if os.path.exists(dfn):
                                    try:
                                        os.remove(dfn)
                                    except (OSError, PermissionError):
                                        pass  # Ignore if locked
                                try:
                                    os.rename(sfn, dfn)
                                except (OSError, PermissionError):
                                    pass  # Ignore if locked
                        
                        # Rotate current file
                        dfn = self.rotation_filename(self.baseFilename + ".1")
                        if os.path.exists(dfn):
                            try:
                                os.remove(dfn)
                            except (OSError, PermissionError):
                                pass  # Ignore if locked
                        
                        try:
                            if os.path.exists(self.baseFilename):
                                os.rename(self.baseFilename, dfn)
                        except (OSError, PermissionError):
                            # If we can't rotate, just truncate the current file
                            try:
                                with open(self.baseFilename, 'w') as f:
                                    f.write('')
                            except Exception:
                                pass
                    except Exception:
                        # If anything fails, continue logging to the current file
                        pass
                    
                    # Open new log file
                    if not self.stream:
                        self.stream = self._open()
            
            handler = WindowsSafeRotatingFileHandler(
                log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8", delay=False
            )
        else:
            handler = RotatingFileHandler(
                log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(process)d:%(threadName)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        
        # Close and remove any existing RotatingFileHandler to avoid file locks
        handlers_to_remove = []
        for h in root_logger.handlers:
            if isinstance(h, RotatingFileHandler):
                handlers_to_remove.append(h)
        
        for h in handlers_to_remove:
            try:
                h.close()
            except Exception:
                pass
            try:
                root_logger.removeHandler(h)
            except Exception:
                pass
        
        # Add the new handler
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

        # Suppress verbose Azure SDK HTTP logging
        logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)
        logging.getLogger('azure').setLevel(logging.WARNING)

        # Capture warnings via logging
        logging.captureWarnings(True)

        # Enable faulthandler to capture hard crashes (e.g., native OOM)
        crash_file = os.path.join(logs_dir, "crash.log")
        # Keep the file handle open for the lifetime of the process
        _FAULT_LOG_FH = open(crash_file, "a", encoding="utf-8")
        try:
            faulthandler.enable(file=_FAULT_LOG_FH, all_threads=True)
        except Exception:
            # Best-effort: continue even if faulthandler cannot be enabled
            pass

        # Ensure the crash log handle is closed on exit
        @atexit.register
        def _close_fault_log():
            try:
                if _FAULT_LOG_FH and not _FAULT_LOG_FH.closed:
                    _FAULT_LOG_FH.flush()
                    _FAULT_LOG_FH.close()
            except Exception:
                pass
        
        # Add aggressive cleanup for GIL issues
        @atexit.register 
        def _emergency_thread_cleanup():
            """Emergency cleanup to prevent GIL issues on shutdown"""
            try:
                # Force garbage collection
                import gc
                gc.collect()
                
                # Try to stop any remaining daemon threads
                import threading
                for thread in threading.enumerate():
                    if thread != threading.current_thread() and thread.daemon:
                        try:
                            # Don't wait, just mark for cleanup
                            pass
                        except Exception:
                            pass
            except Exception:
                pass

        # Log uncaught exceptions as critical errors
        _original_excepthook = sys.excepthook
        
        # Use a mutable container to avoid global scope issues
        _excepthook_state = {'in_progress': False}
        
        def _log_excepthook(exc_type, exc_value, exc_tb):
            # Prevent recursion if the excepthook itself raises an exception
            if _excepthook_state['in_progress']:
                # Fall back to original excepthook to avoid infinite recursion
                try:
                    _original_excepthook(exc_type, exc_value, exc_tb)
                except Exception:
                    pass
                return
            
            _excepthook_state['in_progress'] = True
            try:
                logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
            except Exception:
                # If logging fails, try to use the original excepthook
                try:
                    _original_excepthook(exc_type, exc_value, exc_tb)
                except Exception:
                    pass
            finally:
                _excepthook_state['in_progress'] = False
        
        sys.excepthook = _log_excepthook

        logging.getLogger(__name__).info("File logging initialized at %s", log_file)
    except Exception as e:
        # Fallback to basicConfig if anything goes wrong
        try:
            logging.basicConfig(level=logging.INFO)
        except Exception:
            pass
        try:
            print(f"Logging setup failed: {e}")
        except Exception:
            pass

# Initialize logging at import time to catch early failures
_setup_file_logging()

# Start a lightweight background memory usage logger so we can track RAM over time
# TEMPORARILY DISABLED to fix GIL issue
# try:
#     from memory_usage_reporter import start_global_memory_logger
#     start_global_memory_logger()
# except Exception as _e:
#     try:
#         logging.getLogger(__name__).warning("Memory usage logger failed to start: %s", _e)
#     except Exception:
#         pass

# Apply a safety patch for tqdm to avoid shutdown-time AttributeError without disabling tqdm
try:
    from tqdm_safety import apply_tqdm_safety_patch
    apply_tqdm_safety_patch()
except Exception as _e:
    try:
        logging.getLogger(__name__).debug("tqdm safety patch failed to apply: %s", _e)
    except Exception:
        pass

def is_traditional_translation_api(model: str) -> bool:
    """Check if the model is a traditional translation API"""
    return model in ['deepl', 'google-translate', 'google-translate-free'] or model.startswith('deepl/') or model.startswith('google-translate/')
    
def check_epub_folder_match(epub_name, folder_name, custom_suffixes=''):
    """
    Check if EPUB name and folder name likely refer to the same content
    Uses strict matching to avoid false positives with similar numbered titles
    """
    # Normalize names for comparison
    epub_norm = normalize_name_for_comparison(epub_name)
    folder_norm = normalize_name_for_comparison(folder_name)
    
    # Direct match
    if epub_norm == folder_norm:
        return True
    
    # Check if folder has common output suffixes that should be ignored
    output_suffixes = ['_output', '_translated', '_trans', '_en', '_english', '_done', '_complete', '_final']
    if custom_suffixes:
        custom_list = [s.strip() for s in custom_suffixes.split(',') if s.strip()]
        output_suffixes.extend(custom_list)
    
    for suffix in output_suffixes:
        if folder_norm.endswith(suffix):
            folder_base = folder_norm[:-len(suffix)]
            if folder_base == epub_norm:
                return True
        if epub_norm.endswith(suffix):
            epub_base = epub_norm[:-len(suffix)]
            if epub_base == folder_norm:
                return True
    
    # Check for exact match with version numbers removed
    version_pattern = r'[\s_-]v\d+$'
    epub_no_version = re.sub(version_pattern, '', epub_norm)
    folder_no_version = re.sub(version_pattern, '', folder_norm)
    
    if epub_no_version == folder_no_version and (epub_no_version != epub_norm or folder_no_version != folder_norm):
        return True
    
    # STRICT NUMBER CHECK - all numbers must match exactly
    epub_numbers = re.findall(r'\d+', epub_name)
    folder_numbers = re.findall(r'\d+', folder_name)
    
    if epub_numbers != folder_numbers:
        return False
    
    # If we get here, numbers match, so check if the text parts are similar enough
    epub_text_only = re.sub(r'\d+', '', epub_norm).strip()
    folder_text_only = re.sub(r'\d+', '', folder_norm).strip()
    
    if epub_numbers and folder_numbers:
        return epub_text_only == folder_text_only
    
    return False

def normalize_name_for_comparison(name):
    """Normalize a filename for comparison - preserving number positions"""
    name = name.lower()
    name = re.sub(r'\.(epub|txt|html?)$', '', name)
    name = re.sub(r'[-_\s]+', ' ', name)
    name = re.sub(r'\[(?![^\]]*\d)[^\]]*\]', '', name)
    name = re.sub(r'\((?![^)]*\d)[^)]*\)', '', name)
    name = re.sub(r'[^\w\s\-]', ' ', name)
    name = ' '.join(name.split())
    return name.strip()
        
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

# UIHelper class removed - not needed in PySide6
# PySide6 has built-in undo/redo support

# WindowManager class removed - not needed in PySide6
# Qt handles window management automatically

   
# Import QA Scanner, Retranslation, and Glossary Manager mixins
from QA_Scanner_GUI import QAScannerMixin
from Retranslation_GUI import RetranslationMixin
from GlossaryManager_GUI import GlossaryManagerMixin

class TranslatorGUI(QAScannerMixin, RetranslationMixin, GlossaryManagerMixin, QMainWindow):
    # Qt Signal for thread-safe logging
    log_signal = Signal(str)
    # Qt Signal for notifying when threads complete
    thread_complete_signal = Signal()
    # Qt Signal for triggering QA scan from background thread
    trigger_qa_scan_signal = Signal()
    
    def __init__(self, parent=None):
        # Initialize QMainWindow
        super().__init__(parent)
        
        # Connect the log signal to append_log_direct
        self.log_signal.connect(self.append_log_direct)
        # Connect thread complete signal to update buttons
        self.thread_complete_signal.connect(self.update_run_button)
        # Connect QA scan trigger signal
        self.trigger_qa_scan_signal.connect(self._trigger_qa_scan_on_main_thread)
        
        # Store master reference for compatibility (will be self now)
        self.master = self
        self.base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        
        # Set window properties with comprehensive dark theme styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: white;
            }
            QLabel {
                color: white;
                background-color: transparent;
            }
            QLineEdit, QTextEdit {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #4a5568;
                border-radius: 3px;
                padding: 4px;
            }
            QLineEdit:focus, QTextEdit:focus {
                border-color: #5a9fd4;
            }
            QLineEdit:disabled, QTextEdit:disabled {
                background-color: #1a1a1a;
                color: #666666;
                border: 1px solid #3a3a3a;
            }
            QPushButton {
                background-color: #3d3d3d;
                color: white;
                border: 1px solid #4a5568;
                border-radius: 3px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
                border-color: #5a9fd4;
            }
            QPushButton:pressed {
                background-color: #2d2d2d;
            }
            QPushButton:disabled {
                background-color: #1a1a1a;
                color: #666666;
                border: 1px solid #3a3a3a;
            }
            QPushButton::menu-indicator {
                width: 0;
                height: 0;
                border: none;
                background: none;
                margin-right: 8px;
            }
            QComboBox {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #4a5568;
                border-radius: 3px;
                padding: 4px;
                padding-right: 25px;
            }
            QComboBox:disabled {
                background-color: #1a1a1a;
                color: #666666;
                border: 1px solid #3a3a3a;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #4a5568;
            }
            QComboBox::down-arrow {
                width: 16px;
                height: 16px;
                background: transparent;
                border: none;
            }
            QComboBox::down-arrow:on {
                top: 1px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                color: white;
                selection-background-color: #5a9fd4;
            }
            QCheckBox {
                color: white;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #5a9fd4;
                border-radius: 2px;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background-color: #5a9fd4;
                border-color: #5a9fd4;
            }
            QCheckBox::indicator:hover {
                border-color: #7bb3e0;
            }
            QCheckBox:disabled {
                color: #666666;
            }
            QCheckBox::indicator:disabled {
                background-color: #1a1a1a;
                border-color: #3a3a3a;
            }
            QMenu {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #4a5568;
            }
            QMenu::item:selected {
                background-color: #5a9fd4;
            }
        """)
        
        self.max_output_tokens = 8192
        self.proc = self.glossary_proc = None
        __version__ = "6.1.0"
        self.__version__ = __version__
        self.setWindowTitle(f"Glossarion v{__version__}")
        
        # Track fullscreen state
        self.is_fullscreen = False
        
        # Get screen dimensions
        screen = QApplication.primaryScreen()
        rect = screen.availableGeometry()
        
        # Use more reasonable window size ratios
        # 70% width and 85% height are more standard and fit better on single monitors
        width_ratio = 0.70   # 70% of screen width
        height_ratio = 0.88  # 88% of available height
        
        window_width = int(rect.width() * width_ratio)
        window_height = int(rect.height() * height_ratio)
        
        print(f"[DEBUG] Calculated window size: {window_width}x{window_height}")
        print(f"[DEBUG] Screen dimensions: {rect.width()}x{rect.height()}")
        print(f"[DEBUG] Width ratio: {width_ratio}, Height ratio: {height_ratio}")
        
        # Apply size
        self.resize(window_width, window_height)
        
        # Set minimum size (40% of screen as minimum)
        min_width = int(rect.width() * 0.40)
        min_height = int(rect.height() * 0.40)
        self.setMinimumSize(min_width, min_height)
        
        # Center window on screen
        self.move(rect.center() - self.rect().center())
        
        self.payloads_dir = os.path.join(os.getcwd(), "Payloads")
        
        # Auto-scroll control: delay forcing scroll on new runs
        self._autoscroll_delay_until = 0.0  # epoch seconds
        self._user_scrolled_up = False  # Track if user manually scrolled up
        
        self._modules_loaded = self._modules_loading = False
        self.stop_requested = False
        self.translation_thread = self.glossary_thread = self.qa_thread = self.epub_thread = None
        self.qa_thread = None
        # Futures for executor-based tasks
        self.translation_future = self.glossary_future = self.qa_future = self.epub_future = None
        # Shared executor for background tasks
        self.executor = None
        self._executor_workers = None
        
        # Glossary tracking
        self.manual_glossary_path = None
        self.auto_loaded_glossary_path = None
        self.auto_loaded_glossary_for_file = None
        self.manual_glossary_manually_loaded = False
        
        # Load icon
        ico_path = os.path.join(self.base_dir, 'Halgakos.ico')
        if os.path.isfile(ico_path):
            try:
                self.setWindowIcon(QIcon(ico_path))
            except:
                pass
        
        self.logo_img = None
        
        # Load config
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
                # Decrypt API keys
                self.config = decrypt_config(self.config)
        except: 
            self.config = {}
            
        # Ensure default values exist
        if 'auto_update_check' not in self.config:
            self.config['auto_update_check'] = True
            # Save the default config immediately so it exists
            try:
                with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Warning: Could not save config.json: {e}")

        # After loading config, check for Google Cloud credentials
        if self.config.get('google_cloud_credentials'):
            creds_path = self.config['google_cloud_credentials']
            if os.path.exists(creds_path):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path
                # Log will be added after GUI is created
            
        if 'force_ncx_only' not in self.config:
            self.config['force_ncx_only'] = True

        # Initialize OpenRouter transport/compression toggles early so they're available
        # before the settings UI creates these variables. This prevents attribute errors
        # when features (like glossary extraction) access them at startup.
        try:
            self.openrouter_http_only_var = self.config.get('openrouter_use_http_only', False)
        except Exception:
            self.openrouter_http_only_var = False
        
        try:
            self.openrouter_accept_identity_var = self.config.get('openrouter_accept_identity', False)
        except Exception:
            self.openrouter_accept_identity_var = False
            
        # Initialize retain_source_extension env var on startup
        try:
            os.environ['RETAIN_SOURCE_EXTENSION'] = '1' if self.config.get('retain_source_extension', False) else '0'
        except Exception:
            pass
        
        # Force safe ratios is not needed in PySide6
        # Window sizing is handled by Qt's layout system
    
        # Initialize auto-update check and other variables (converted from Tkinter to Python vars)
        self.auto_update_check_var = self.config.get('auto_update_check', True)
        self.force_ncx_only_var = self.config.get('force_ncx_only', True)
        self.single_api_image_chunks_var = False
        self.enable_gemini_thinking_var = self.config.get('enable_gemini_thinking', True)
        self.thinking_budget_var = str(self.config.get('thinking_budget', '-1'))
        # NEW: GPT/OpenRouter reasoning controls
        self.enable_gpt_thinking_var = self.config.get('enable_gpt_thinking', True)
        self.gpt_reasoning_tokens_var = str(self.config.get('gpt_reasoning_tokens', '2000'))
        self.gpt_effort_var = self.config.get('gpt_effort', 'medium')
        self.thread_delay_var = str(self.config.get('thread_submission_delay', 0.5))
        self.remove_ai_artifacts = os.getenv("REMOVE_AI_ARTIFACTS", "0") == "1"
        print(f"   🎨 Remove AI Artifacts: {'ENABLED' if self.remove_ai_artifacts else 'DISABLED'}")
        self.disable_chapter_merging_var = self.config.get('disable_chapter_merging', False)
        self.selected_files = []
        self.current_file_index = 0
        self.use_gemini_openai_endpoint_var = self.config.get('use_gemini_openai_endpoint', False)
        self.gemini_openai_endpoint_var = self.config.get('gemini_openai_endpoint', '')
        self.azure_api_version_var = self.config.get('azure_api_version', '2025-01-01-preview')
        # Set initial Azure API version environment variable
        azure_version = self.config.get('azure_api_version', '2025-01-01-preview')
        os.environ['AZURE_API_VERSION'] = azure_version
        print(f"🔧 Initial Azure API Version set: {azure_version}")
        self.use_fallback_keys_var = self.config.get('use_fallback_keys', False)

        # Initialize fuzzy threshold variable
        if not hasattr(self, 'fuzzy_threshold_var'):
            self.fuzzy_threshold_var = self.config.get('glossary_fuzzy_threshold', 0.90)
        self.use_legacy_csv_var = self.config.get('glossary_use_legacy_csv', False)

        
        # Initialize the variables with default values
        self.enable_parallel_extraction_var = self.config.get('enable_parallel_extraction', True)
        self.extraction_workers_var = self.config.get('extraction_workers', 2)
        # GUI yield toggle - disabled by default for maximum speed
        self.enable_gui_yield_var = self.config.get('enable_gui_yield', True)

        # Set initial environment variable and ensure executor
        if self.enable_parallel_extraction_var:
            # Set workers for glossary extraction optimization
            workers = self.extraction_workers_var
            os.environ["EXTRACTION_WORKERS"] = str(workers)
            # Also enable glossary parallel processing explicitly
            os.environ["GLOSSARY_PARALLEL_ENABLED"] = "1"
            print(f"✅ Parallel extraction enabled with {workers} workers")
        else:
            os.environ["EXTRACTION_WORKERS"] = "1"
            os.environ["GLOSSARY_PARALLEL_ENABLED"] = "0"
        
        # Set GUI yield environment variable (disabled by default for maximum speed)
        os.environ['ENABLE_GUI_YIELD'] = '1' if self.enable_gui_yield_var else '0'
        print(f"⚡ GUI yield: {'ENABLED (responsive)' if self.enable_gui_yield_var else 'DISABLED (maximum speed)'}")
        
        # Initialize the executor based on current settings
        try:
            self._ensure_executor()
        except Exception:
            pass


        # Initialize compression-related variables
        self.enable_image_compression_var = self.config.get('enable_image_compression', False)
        self.auto_compress_enabled_var = self.config.get('auto_compress_enabled', True)
        self.target_image_tokens_var = str(self.config.get('target_image_tokens', 1000))
        self.image_format_var = self.config.get('image_compression_format', 'auto')
        self.webp_quality_var = self.config.get('webp_quality', 85)
        self.jpeg_quality_var = self.config.get('jpeg_quality', 85)
        self.png_compression_var = self.config.get('png_compression', 6)
        self.max_image_dimension_var = str(self.config.get('max_image_dimension', 2048))
        self.max_image_size_mb_var = str(self.config.get('max_image_size_mb', 10))
        self.preserve_transparency_var = self.config.get('preserve_transparency', False)
        self.preserve_original_format_var = self.config.get('preserve_original_format', False)
        self.optimize_for_ocr_var = self.config.get('optimize_for_ocr', True)
        self.progressive_encoding_var = self.config.get('progressive_encoding', True)
        self.save_compressed_images_var = self.config.get('save_compressed_images', False)
        self.image_chunk_overlap_var = str(self.config.get('image_chunk_overlap', '1'))

        # Glossary-related variables (existing)
        self.append_glossary_var = self.config.get('append_glossary', False)
        self.glossary_min_frequency_var = str(self.config.get('glossary_min_frequency', 2))
        self.glossary_max_names_var = str(self.config.get('glossary_max_names', 50))
        self.glossary_max_titles_var = str(self.config.get('glossary_max_titles', 30))
        self.glossary_batch_size_var = str(self.config.get('glossary_batch_size', 50))
        self.glossary_max_text_size_var = str(self.config.get('glossary_max_text_size', 50000))
        self.glossary_chapter_split_threshold_var = self.config.get('glossary_chapter_split_threshold', '8192')
        self.glossary_max_sentences_var = str(self.config.get('glossary_max_sentences', 200))
        self.glossary_filter_mode_var = self.config.get('glossary_filter_mode', 'all')

        
        # NEW: Additional glossary settings
        self.strip_honorifics_var = self.config.get('strip_honorifics', True)
        self.disable_honorifics_var = self.config.get('glossary_disable_honorifics_filter', False)
        self.manual_temp_var = str(self.config.get('manual_glossary_temperature', 0.3))
        self.manual_context_var = str(self.config.get('manual_context_limit', 5))
        
        # Custom glossary fields and entry types
        self.custom_glossary_fields = self.config.get('custom_glossary_fields', [])
        self.custom_entry_types = self.config.get('custom_entry_types', {
            'character': {'enabled': True, 'has_gender': True},
            'term': {'enabled': True, 'has_gender': False}
        })
        
        # Glossary prompts
        self.manual_glossary_prompt = self.config.get('manual_glossary_prompt', 
            """Extract character names and important terms from the text.
Format each entry as: type,raw_name,translated_name,gender
For terms use: term,raw_name,translated_name,""")
        
        self.auto_glossary_prompt = self.config.get('auto_glossary_prompt', 
            """Extract all character names and important terms from the text.
Focus on:
1. Character names (maximum {max_names})
2. Important titles and positions (maximum {max_titles})
3. Terms that appear at least {min_frequency} times

Return as JSON: {"term": "translation", ...}""")
        
        self.append_glossary_prompt = self.config.get('append_glossary_prompt', 
           '- Follow this reference glossary for consistent translation (Do not output any raw entries):\n')
        
        self.glossary_translation_prompt = self.config.get('glossary_translation_prompt', 
            """
You are translating {language} character names and important terms to English.
For character names, provide English transliterations or keep as romanized.
Keep honorifics/suffixes only if they are integral to the name.
Respond with the same numbered format.

Terms to translate:
{terms_list}

Provide translations in the same numbered format.""")
        self.glossary_format_instructions = self.config.get('glossary_format_instructions', 
            """
Return the results in EXACT CSV format with this header:
type,raw_name,translated_name

For example:
character,김상현,Kim Sang-hyu
character,갈편제,Gale Hardest  
character,디히릿 아데,Dihirit Ade

Only include terms that actually appear in the text.
Do not use quotes around values unless they contain commas.

Text to analyze:
{text_sample}""")  
        
        # Initialize custom API endpoint variables
        self.openai_base_url_var = self.config.get('openai_base_url', '')
        self.groq_base_url_var = self.config.get('groq_base_url', '')
        self.fireworks_base_url_var = self.config.get('fireworks_base_url', '')
        self.use_custom_openai_endpoint_var = self.config.get('use_custom_openai_endpoint', False)
        
        # Initialize metadata/batch variables the same way
        self.translate_metadata_fields = self.config.get('translate_metadata_fields', {})
        # Initialize metadata translation UI and prompts
        try:
            from metadata_batch_translator import MetadataBatchTranslatorUI
            self.metadata_ui = MetadataBatchTranslatorUI(self)
            # This ensures default prompts are in config
        except ImportError:
            print("Metadata translation UI not available")
        self.batch_translate_headers_var = self.config.get('batch_translate_headers', False)
        self.headers_per_batch_var = self.config.get('headers_per_batch', '400')
        self.update_html_headers_var = self.config.get('update_html_headers', True)
        self.save_header_translations_var = self.config.get('save_header_translations', True)
        self.ignore_header_var = self.config.get('ignore_header', False)
        self.ignore_title_var = self.config.get('ignore_title', False)
        self.use_sorted_fallback_var = self.config.get('use_sorted_fallback', False)  # Disabled by default
        self.attach_css_to_chapters_var = self.config.get('attach_css_to_chapters', False)
        
        # Retain exact source extension and disable 'response_' prefix
        self.retain_source_extension_var = self.config.get('retain_source_extension', False)
        
        # Initialize extraction settings (from Other Settings)
        self.force_bs_for_traditional_var = self.config.get('force_bs_for_traditional', False)
        
        # Initialize HTTP/Network tuning variables (from Other Settings)
        self.enable_http_tuning_var = self.config.get('enable_http_tuning', False)
        self.connect_timeout_var = str(self.config.get('connect_timeout', 10))
        self.read_timeout_var = str(self.config.get('read_timeout', 180))
        self.http_pool_connections_var = str(self.config.get('http_pool_connections', 20))
        self.http_pool_maxsize_var = str(self.config.get('http_pool_maxsize', 50))
        self.ignore_retry_after_var = self.config.get('ignore_retry_after', False)
        self.max_retries_var = str(self.config.get('max_retries', 7))
        
        # Initialize anti-duplicate parameters (from Other Settings)
        self.enable_anti_duplicate_var = self.config.get('enable_anti_duplicate', False)
        self.top_p_var = self.config.get('top_p', 1.0)
        self.top_k_var = self.config.get('top_k', 0)
        self.frequency_penalty_var = self.config.get('frequency_penalty', 0.0)
        self.presence_penalty_var = self.config.get('presence_penalty', 0.0)
        self.repetition_penalty_var = self.config.get('repetition_penalty', 1.0)
        self.candidate_count_var = self.config.get('candidate_count', 1)
        self.custom_stop_sequences_var = self.config.get('custom_stop_sequences', '')
        self.logit_bias_enabled_var = self.config.get('logit_bias_enabled', False)
        self.logit_bias_strength_var = self.config.get('logit_bias_strength', 1.0)
        self.bias_common_words_var = self.config.get('bias_common_words', False)
        self.bias_repetitive_phrases_var = self.config.get('bias_repetitive_phrases', False)

        
        self.max_output_tokens = self.config.get('max_output_tokens', self.max_output_tokens)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(500, lambda: self.on_model_change() if hasattr(self, 'model_var') else None)
        
        
        # Async processing settings
        self.async_wait_for_completion_var = self.config.get('async_wait_for_completion', False)
        self.async_poll_interval_var = self.config.get('async_poll_interval', 60)
        
         # Enhanced filtering level
        if not hasattr(self, 'enhanced_filtering_var'):
            self.enhanced_filtering_var = self.config.get('enhanced_filtering', 'smart')
        
        # Preserve structure toggle
        if not hasattr(self, 'enhanced_preserve_structure_var'):
            self.enhanced_preserve_structure_var = self.config.get('enhanced_preserve_structure', True)
             
        # Initialize update manager AFTER config is loaded
        try:
            from update_manager import UpdateManager
            self.update_manager = UpdateManager(self, self.base_dir)
            
            # Check for updates on startup if enabled
            auto_check_enabled = self.config.get('auto_update_check', True)
            print(f"[DEBUG] Auto-update check enabled: {auto_check_enabled}")
            
            if auto_check_enabled:
                print("[DEBUG] Scheduling update check for 5 seconds from now...")
                from PySide6.QtCore import QTimer
                QTimer.singleShot(5000, self._check_updates_on_startup)
            else:
                print("[DEBUG] Auto-update check is disabled")
        except ImportError as e:
            self.update_manager = None
            print(f"[DEBUG] Update manager not available: {e}")
        
        try:
            from metadata_batch_translator import MetadataBatchTranslatorUI
            self.metadata_ui = MetadataBatchTranslatorUI(self)
            # This ensures default prompts are in config
        except ImportError:
            print("Metadata translation UI not available")
        
        # Default prompts
        self.default_translation_chunk_prompt = "[This is part {chunk_idx}/{total_chunks}]. You must maintain the narrative flow with the previous chunks while translating it and following all system prompt guidelines previously mentioned.\n{chunk_html}"
        self.default_image_chunk_prompt = "This is part {chunk_idx} of {total_chunks} of a longer image. You must maintain the narrative flow with the previous chunks while translating it and following all system prompt guidelines previously mentioned. {context}"
        self.default_prompts = {

            "korean": (
                "You are a professional Korean to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Retain Korean honorifics and respectful speech markers in romanized form, including but not limited to: -nim, -ssi, -yang, -gun, -isiyeo, -hasoseo. For archaic/classical Korean honorific forms (like 이시여/isiyeo, 하소서/hasoseo), preserve them as-is rather than converting to modern equivalents.\n"
                "- Always localize Korean terminology to proper English equivalents instead of literal translations (examples: 마왕 = Demon King; 마술 = magic).\n"
                "- When translating Korean's pronoun-dropping style, insert pronouns in English only where needed for clarity: prioritize original pronouns as implied or according to the glossary, and only use they/them as a last resort, use I/me for first-person narration, and maintain natural English flow without overusing pronouns just because they're omitted in Korean.\n"
                "- All Korean profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Romaji.\n"
                "- Keep original Korean quotation marks (" ", ' ', 「」, 『』) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 생 means 'life/living', 활 means 'active', 관 means 'hall/building' - together 생활관 means Dormitory.\n"
                "- Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title>, <h1>, <h2>, <p>, <br>, <div>, etc.\n"
            ),
            "japanese": (
                "You are a professional Japanese to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Retain Japanese honorifics and respectful speech markers in romanized form, including but not limited to: -san, -sama, -chan, -kun, -dono, -sensei, -senpai, -kouhai. For archaic/classical Japanese honorific forms, preserve them as-is rather than converting to modern equivalents.\n"
                "- Always localize Japanese terminology to proper English equivalents instead of literal translations (examples: 魔王 = Demon King; 魔術 = magic).\n"
                "- When translating Japanese's pronoun-dropping style, insert pronouns in English only where needed for clarity: prioritize original pronouns as implied or according to the glossary, and only use they/them as a last resort, use I/me for first-person narration while reflecting the Japanese pronoun's nuance (私/僕/俺/etc.) through speech patterns rather than the pronoun itself, and maintain natural English flow without overusing pronouns just because they're omitted in Japanese.\n"
                "- All Japanese profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Romaji.\n"
                "- Keep original Japanese quotation marks (「」 and 『』) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 生 means 'life/living', 活 means 'active', 館 means 'hall/building' - together 生活館 means Dormitory.\n"
                "- Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title>, <h1>, <h2>, <p>, <br>, <div>, etc.\n"
            ),
            "chinese": (
                "You are a professional Chinese to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Retain Chinese titles and respectful forms of address in romanized form, including but not limited to: laoban, laoshi, shifu, xiaojie, xiansheng, taitai, daren, qianbei. For archaic/classical Chinese respectful forms, preserve them as-is rather than converting to modern equivalents.\n"
                "- Always localize Chinese terminology to proper English equivalents instead of literal translations (examples: 魔王 = Demon King; 法术 = magic).\n"
                "- When translating Chinese's flexible pronoun usage, insert pronouns in English only where needed for clarity: prioritize original pronouns as implied or according to the glossary, and only use they/them as a last resort, use I/me for first-person narration while reflecting the pronoun's nuance (我/吾/咱/人家/etc.) through speech patterns and formality level rather than the pronoun itself, and since Chinese pronouns don't indicate gender in speech (他/她/它 all sound like 'tā'), rely on context or glossary rather than assuming gender.\n"
                "- All Chinese profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Romaji.\n"
                "- Keep original Chinese quotation marks (「」 for dialogue, 《》 for titles) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 生 means 'life/living', 活 means 'active', 館 means 'hall/building' - together 生活館 means Dormitory.\n"
                "- Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title>, <h1>, <h2>, <p>, <br>, <div>, etc.\n"
            ),
            "korean_OCR": (
                "You are a professional Korean to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Retain Korean honorifics and respectful speech markers in romanized form, including but not limited to: -nim, -ssi, -yang, -gun, -isiyeo, -hasoseo. For archaic/classical Korean honorific forms (like 이시여/isiyeo, 하소서/hasoseo), preserve them as-is rather than converting to modern equivalents.\n"
                "- Always localize Korean terminology to proper English equivalents instead of literal translations (examples: 마왕 = Demon King; 마술 = magic).\n"
                "- When translating Korean's pronoun-dropping style, insert pronouns in English only where needed for clarity: prioritize original pronouns as implied or according to the glossary, and only use they/them as a last resort, use I/me for first-person narration, and maintain natural English flow without overusing pronouns just because they're omitted in Korean.\n"
                "- All Korean profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Romaji.\n"
                "- Keep original Korean quotation marks (" ", ' ', 「」, 『』) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 생 means 'life/living', 활 means 'active', 관 means 'hall/building' - together 생활관 means Dormitory.\n"
                "- Add HTML tags for proper formatting as expected of a novel.\n"
                "- Wrap every paragraph in <p> tags; do not insert any literal tabs or spaces.\n"
            ),
            "japanese_OCR": (
                "You are a professional Japanese to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Retain Japanese honorifics and respectful speech markers in romanized form, including but not limited to: -san, -sama, -chan, -kun, -dono, -sensei, -senpai, -kouhai. For archaic/classical Japanese honorific forms, preserve them as-is rather than converting to modern equivalents.\n"
                "- Always localize Japanese terminology to proper English equivalents instead of literal translations (examples: 魔王 = Demon King; 魔術 = magic).\n"
                "- When translating Japanese's pronoun-dropping style, insert pronouns in English only where needed for clarity: prioritize original pronouns as implied or according to the glossary, and only use they/them as a last resort, use I/me for first-person narration while reflecting the Japanese pronoun's nuance (私/僕/俺/etc.) through speech patterns rather than the pronoun itself, and maintain natural English flow without overusing pronouns just because they're omitted in Japanese.\n"
                "- All Japanese profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Romaji.\n"
                "- Keep original Japanese quotation marks (「」 and 『』) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 生 means 'life/living', 活 means 'active', 館 means 'hall/building' - together 生活館 means Dormitory.\n"
                "- Add HTML tags for proper formatting as expected of a novel.\n"
                "- Wrap every paragraph in <p> tags; do not insert any literal tabs or spaces.\n"
            ),
            "chinese_OCR": (
                "You are a professional Chinese to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Retain Chinese titles and respectful forms of address in romanized form, including but not limited to: laoban, laoshi, shifu, xiaojie, xiansheng, taitai, daren, qianbei. For archaic/classical Chinese respectful forms, preserve them as-is rather than converting to modern equivalents.\n"
                "- Always localize Chinese terminology to proper English equivalents instead of literal translations (examples: 魔王 = Demon King; 法术 = magic).\n"
                "- When translating Chinese's flexible pronoun usage, insert pronouns in English only where needed for clarity: prioritize original pronouns as implied or according to the glossary, and only use they/them as a last resort, use I/me for first-person narration while reflecting the pronoun's nuance (我/吾/咱/人家/etc.) through speech patterns and formality level rather than the pronoun itself, and since Chinese pronouns don't indicate gender in speech (他/她/它 all sound like 'tā'), rely on context or glossary rather than assuming gender.\n"
                "- All Chinese profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Romaji.\n"
                "- Keep original Chinese quotation marks (「」 for dialogue, 《》 for titles) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 生 means 'life/living', 活 means 'active', 館 means 'hall/building' - together 生活館 means Dormitory.\n"
                "- Add HTML tags for proper formatting as expected of a novel.\n"
                "- Wrap every paragraph in <p> tags; do not insert any literal tabs or spaces.\n"
            ),
            "korean_TXT": (
                "You are a professional Korean to English novel translator, you must strictly output only English text while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Retain Korean honorifics and respectful speech markers in romanized form, including but not limited to: -nim, -ssi, -yang, -gun, -isiyeo, -hasoseo. For archaic/classical Korean honorific forms (like 이시여/isiyeo, 하소서/hasoseo), preserve them as-is rather than converting to modern equivalents.\n"
                "- Always localize Korean terminology to proper English equivalents instead of literal translations (examples: 마왕 = Demon King; 마술 = magic).\n"
                "- When translating Korean's pronoun-dropping style, insert pronouns in English only where needed for clarity: prioritize original pronouns as implied or according to the glossary, and only use they/them as a last resort, use I/me for first-person narration, and maintain natural English flow without overusing pronouns just because they're omitted in Korean.\n"
                "- All Korean profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Romaji.\n"
                "- Keep original Korean quotation marks (" ", ' ', 「」, 『』) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 생 means 'life/living', 활 means 'active', 관 means 'hall/building' - together 생활관 means Dormitory.\n"
                "- Use line breaks for proper formatting as expected of a novel.\n"
            ),
            "japanese_TXT": (
                "You are a professional Japanese to English novel translator, you must strictly output only English text while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Retain Japanese honorifics and respectful speech markers in romanized form, including but not limited to: -san, -sama, -chan, -kun, -dono, -sensei, -senpai, -kouhai. For archaic/classical Japanese honorific forms, preserve them as-is rather than converting to modern equivalents.\n"
                "- Always localize Japanese terminology to proper English equivalents instead of literal translations (examples: 魔王 = Demon King; 魔術 = magic).\n"
                "- When translating Japanese's pronoun-dropping style, insert pronouns in English only where needed for clarity: prioritize original pronouns as implied or according to the glossary, and only use they/them as a last resort, use I/me for first-person narration while reflecting the Japanese pronoun's nuance (私/僕/俺/etc.) through speech patterns rather than the pronoun itself, and maintain natural English flow without overusing pronouns just because they're omitted in Japanese.\n"
                "- All Japanese profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Romaji.\n"
                "- Keep original Japanese quotation marks (「」 and 『』) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 生 means 'life/living', 活 means 'active', 館 means 'hall/building' - together 生活館 means Dormitory.\n"
                "- Use line breaks for proper formatting as expected of a novel.\n"
            ),
            "chinese_TXT": (
                "You are a professional Chinese to English novel translator, you must strictly output only English text while following these rules:\n"
                "- Use a natural, comedy-friendly English translation style that captures both humor and readability without losing any original meaning.\n"
                "- Include 100% of the source text - every word, phrase, and sentence must be fully translated without exception.\n"
                "- Retain Chinese titles and respectful forms of address in romanized form, including but not limited to: laoban, laoshi, shifu, xiaojie, xiansheng, taitai, daren, qianbei. For archaic/classical Chinese respectful forms, preserve them as-is rather than converting to modern equivalents.\n"
                "- Always localize Chinese terminology to proper English equivalents instead of literal translations (examples: 魔王 = Demon King; 法术 = magic).\n"
                "- When translating Chinese's flexible pronoun usage, insert pronouns in English only where needed for clarity: prioritize original pronouns as implied or according to the glossary, and only use they/them as a last resort, use I/me for first-person narration while reflecting the pronoun's nuance (我/吾/咱/人家/etc.) through speech patterns and formality level rather than the pronoun itself, and since Chinese pronouns don't indicate gender in speech (他/她/它 all sound like 'tā'), rely on context or glossary rather than assuming gender.\n"
                "- All Chinese profanity must be translated to English profanity.\n"
                "- Preserve original intent, and speech tone.\n"
                "- Retain onomatopoeia in Romaji.\n"
                "- Keep original Chinese quotation marks (「」 for dialogue, 《》 for titles) as-is without converting to English quotes.\n"
                "- Every Korean/Chinese/Japanese character must be converted to its English meaning. Examples: The character 生 means 'life/living', 活 means 'active', 館 means 'hall/building' - together 生活館 means Dormitory.\n"
                "- Use line breaks for proper formatting as expected of a novel.\n"
            ),
            "Manga_JP": (
                "You are a professional Japanese to English Manga translator.\n"
                "You have both the image of the Manga panel and the extracted text to work with.\n"
                "Output only English text while following these rules: \n\n"

                "VISUAL CONTEXT:\n"
                "- Analyze the character’s facial expressions and body language in the image.\n"
                "- Consider the scene’s mood and atmosphere.\n"
                "- Note any action or movement depicted.\n"
                "- Use visual cues to determine the appropriate tone and emotion.\n"
                "- USE THE IMAGE to inform your translation choices. The image is not decorative - it contains essential context for accurate translation.\n\n"

                "DIALOGUE REQUIREMENTS:\n"
                "- Match the translation tone to the character's expression.\n"
                "- If a character looks angry, use appropriately intense language.\n"
                "- If a character looks shy or embarrassed, reflect that in the translation.\n"
                "- Keep speech patterns consistent with the character's appearance and demeanor.\n"
                "- Retain honorifics and onomatopoeia in Romaji.\n"
                "- Keep original Japanese quotation marks (「」, 『』) as-is without converting to English quotes.\n\n"

                "IMPORTANT: Use both the visual context and text to create the most accurate and natural-sounding translation.\n"
            ), 
            "Manga_KR": (
                "You are a professional Korean to English Manhwa translator.\n"
                "You have both the image of the Manhwa panel and the extracted text to work with.\n"
                "Output only English text while following these rules: \n\n"

                "VISUAL CONTEXT:\n"
                "- Analyze the character’s facial expressions and body language in the image.\n"
                "- Consider the scene’s mood and atmosphere.\n"
                "- Note any action or movement depicted.\n"
                "- Use visual cues to determine the appropriate tone and emotion.\n"
                "- USE THE IMAGE to inform your translation choices. The image is not decorative - it contains essential context for accurate translation.\n\n"

                "DIALOGUE REQUIREMENTS:\n"
                "- Match the translation tone to the character's expression.\n"
                "- If a character looks angry, use appropriately intense language.\n"
                "- If a character looks shy or embarrassed, reflect that in the translation.\n"
                "- Keep speech patterns consistent with the character's appearance and demeanor.\n"
                "- Retain honorifics and onomatopoeia in Romaji.\n"
                "- Keep original Korean quotation marks (" ", ' ', 「」, 『』) as-is without converting to English quotes.\n\n"

                "IMPORTANT: Use both the visual context and text to create the most accurate and natural-sounding translation.\n"
            ), 
            "Manga_CN": (
                "You are a professional Chinese to English Manga translator.\n"
                "You have both the image of the Manga panel and the extracted text to work with.\n"
                "Output only English text while following these rules: \n\n"

                "VISUAL CONTEXT:\n"
                "- Analyze the character’s facial expressions and body language in the image.\n"
                "- Consider the scene’s mood and atmosphere.\n"
                "- Note any action or movement depicted.\n"
                "- Use visual cues to determine the appropriate tone and emotion.\n"
                "- USE THE IMAGE to inform your translation choices. The image is not decorative - it contains essential context for accurate translation.\n"

                "DIALOGUE REQUIREMENTS:\n"
                "- Match the translation tone to the character's expression.\n"
                "- If a character looks angry, use appropriately intense language.\n"
                "- If a character looks shy or embarrassed, reflect that in the translation.\n"
                "- Keep speech patterns consistent with the character's appearance and demeanor.\n"
                "- Retain honorifics and onomatopoeia in Romaji.\n"
                "- Keep original Chinese quotation marks (「」, 『』) as-is without converting to English quotes.\n\n"

                "IMPORTANT: Use both the visual context and text to create the most accurate and natural-sounding translation.\n"
            ),   
            "Glossary_Editor": (
                "I have a messy character glossary from a Korean web novel that needs to be cleaned up and restructured. Please Output only JSON entries while creating a clean JSON glossary with the following requirements:\n"
                "1. Merge duplicate character entries - Some characters appear multiple times (e.g., Noah, Ichinose family members).\n"
                "2. Separate mixed character data - Some entries incorrectly combine multiple characters' information.\n"
                "3. Use 'Korean = English' format - Replace all parentheses with equals signs (e.g., '이로한 = Lee Rohan' instead of '이로한 (Lee Rohan)').\n"
                "4. Merge original_name fields - Combine original Korean names with English names in the name field.\n"
                "5. Remove empty fields - Don't include empty arrays or objects.\n"
                "6. Fix gender inconsistencies - Correct based on context from aliases.\n"

            ),
            "Original": "Return everything exactly as seen on the source."
        }

        self._init_default_prompts()
        self._init_variables()
        
        # Bind other settings methods early so they're available during GUI setup
        from other_settings import setup_other_settings_methods
        setup_other_settings_methods(self)
        
        self._setup_gui()
        
        # Initialize all environment variables after GUI setup but before first use
        self.initialize_environment_variables()

        # Attach logging handlers to forward client logs into the GUI
        try:
            self._attach_gui_logging_handlers()
        except Exception as _e_attach:
            # Do not fail GUI initialization due to logging handler issues
            try:
                self.append_log(f"⚠️ Failed to attach GUI logging handlers: {_e_attach}")
            except Exception:
                pass
        
        self.metadata_batch_ui = MetadataBatchTranslatorUI(self)
        
        try:
            needs_encryption = False
            if 'api_key' in self.config and self.config['api_key']:
                if not self.config['api_key'].startswith('ENC:'):
                    needs_encryption = True
            if 'replicate_api_key' in self.config and self.config['replicate_api_key']:
                if not self.config['replicate_api_key'].startswith('ENC:'):
                    needs_encryption = True
            
            if needs_encryption:
                # Auto-migrate to encrypted format
                print("Auto-encrypting API keys...")
                self.save_config(show_message=False)
                print("API keys encrypted successfully!")
        except Exception as e:
            print(f"Auto-encryption check failed: {e}")
    
    def closeEvent(self, event):
        """Handle window close event properly"""
        try:
            print("[CLOSE] Window closing...")
            
            # Stop any background operations first
            self.stop_all_operations()
            
            print("[CLOSE] Background operations stopped, accepting close event...")
            
            # Accept the close event - this will naturally end the Qt event loop
            event.accept()
            
            # Quit the application - this will cause app.exec() to return
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                print("[CLOSE] Calling app.quit()...")
                app.quit()
            
            print("[CLOSE] Close event handling completed")
            
            # DO NOT call sys.exit() from here - let the main function handle it
            
        except Exception as e:
            print(f"[CLOSE] Error during close: {e}")
            # Even on error, just accept the event and let main function handle exit
            try:
                event.accept()
                from PySide6.QtWidgets import QApplication
                app = QApplication.instance()
                if app:
                    app.quit()
            except:
                pass
    
    def stop_all_operations(self):
        """Stop all background operations and threads"""
        try:
            print("[CLEANUP] Stopping all background operations...")
            
            # Stop any translation operations
            if hasattr(self, '_translation_thread') and self._translation_thread:
                try:
                    print("[CLEANUP] Terminating translation thread...")
                    self._translation_thread.terminate()
                    self._translation_thread.wait(1000)  # Wait up to 1 second
                except:
                    pass
            
            # Stop glossary operations
            if hasattr(self, '_glossary_thread') and self._glossary_thread:
                try:
                    print("[CLEANUP] Terminating glossary thread...")
                    self._glossary_thread.terminate()
                    self._glossary_thread.wait(1000)  # Wait up to 1 second
                except:
                    pass
            
            # Stop executor if it exists
            if hasattr(self, 'executor') and self.executor:
                try:
                    print("[CLEANUP] Shutting down thread pool executor...")
                    self.executor.shutdown(wait=False)
                except:
                    pass
            
            # Set any stop flags that might exist
            if hasattr(self, 'stop_flag'):
                self.stop_flag.set()
            
            # Close any open dialogs
            if hasattr(self, '_manga_dialog') and self._manga_dialog:
                try:
                    print("[CLEANUP] Closing manga dialog...")
                    self._manga_dialog.close()
                except:
                    pass
            
            print("[CLEANUP] Background operations stopped")
            
        except Exception as e:
            print(f"[CLEANUP] Error stopping operations: {e}")
        
    def _check_updates_on_startup(self):
        """Check for updates on startup with debug logging (async)"""
        print("[DEBUG] Running startup update check...")
        if self.update_manager:
            try:
                self.update_manager.check_for_updates_async(silent=True)
                print(f"[DEBUG] Update check dispatched asynchronously")
            except Exception as e:
                print(f"[DEBUG] Update check failed to dispatch: {e}")
        else:
            print("[DEBUG] Update manager is None")
        
    def check_for_updates_manual(self):
        """Manually check for updates from the Other Settings dialog"""
        if hasattr(self, 'update_manager') and self.update_manager:
            # Use the consolidated update checking system instead of separate loading dialog
            self.update_manager.check_for_updates_manual()
        else:
            QMessageBox.critical(self, "Update Check", 
                               "Update manager is not available.\n"
                               "Please check the GitHub releases page manually:\n"
                               "https://github.com/Shirochi-stack/Glossarion/releases")

# Removed old _show_update_loading_and_check method to prevent conflicts
# Now using the consolidated update checking system in UpdateManager
                               
    # Logging handler to forward client logs into GUI
    class GuiLogHandler(logging.Handler):
        def __init__(self, outer, level=logging.INFO):
            super().__init__(level)
            self.outer = outer
            self.outer_id = id(outer)  # Store ID to identify handlers from same instance
        def emit(self, record):
            try:
                # Use the raw message without logger name/level prefixes
                msg = record.getMessage()
                self.outer.append_log(msg)
            except Exception:
                # Never raise from logging path
                pass

    def _attach_gui_logging_handlers(self):
        """Attach logging handlers so library/client logs appear in the GUI log.
        Safe to call multiple times (won't duplicate handlers).
        """
        try:
            # Build handler
            handler = TranslatorGUI.GuiLogHandler(self, level=logging.INFO)
            fmt = logging.Formatter('%(message)s')
            handler.setFormatter(fmt)
            
            gui_id = id(self)
            
            # Target relevant loggers
            target_loggers = [
                'unified_api_client',
                'httpx',
                'requests.packages.urllib3',
                'openai'
            ]
            for name in target_loggers:
                lg = logging.getLogger(name)
                # Remove any existing handlers from THIS SAME gui instance (any GuiLogHandler type)
                lg.handlers = [h for h in lg.handlers 
                              if not (hasattr(h, 'outer_id') and h.outer_id == gui_id)]
                # Now add the new handler
                lg.addHandler(handler)
                # Ensure at least INFO level to see retry/backoff notices
                if lg.level > logging.INFO or lg.level == logging.NOTSET:
                    lg.setLevel(logging.INFO)
        except Exception as e:
            try:
                self.append_log(f"⚠️ Failed to attach GUI log handlers: {e}")
            except Exception:
                pass

    def create_glossary_backup(self, operation_name="manual"):
        """Create a backup of the current glossary if auto-backup is enabled"""
        # For manual backups, always proceed. For automatic backups, check the setting.
        if operation_name != "manual" and not self.config.get('glossary_auto_backup', True):
            return True
        
        if not self.current_glossary_data or not self.editor_file_var:
            return True
        
        try:
            # Get the original glossary file path
            original_path = self.editor_file_var
            original_dir = os.path.dirname(original_path)
            original_name = os.path.basename(original_path)
            
            # Create backup directory
            backup_dir = os.path.join(original_dir, "Backups")
            
            # Create directory if it doesn't exist
            try:
                os.makedirs(backup_dir, exist_ok=True)
            except Exception as e:
                self.append_log(f"⚠️ Failed to create backup directory: {str(e)}")
                return False
            
            # Generate timestamp-based backup filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_name = f"{os.path.splitext(original_name)[0]}_{operation_name}_{timestamp}.json"
            backup_path = os.path.join(backup_dir, backup_name)
            
            # Try to save backup
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(self.current_glossary_data, f, ensure_ascii=False, indent=2)
            
            self.append_log(f"💾 Backup created: {backup_name}")
            
            # Optional: Clean old backups if more than limit
            max_backups = self.config.get('glossary_max_backups', 50)
            if max_backups > 0:
                self._clean_old_backups(backup_dir, original_name, max_backups)
            
            return True
            
        except Exception as e:
            # Log the actual error
            self.append_log(f"⚠️ Backup failed: {str(e)}")
            # Ask user if they want to continue anyway
            reply = QMessageBox.question(self, "Backup Failed", 
                                      f"Failed to create backup: {str(e)}\n\nContinue anyway?",
                                      QMessageBox.Yes | QMessageBox.No)
            return reply == QMessageBox.Yes

    def get_current_epub_path(self):
        """Get the currently selected EPUB path from various sources"""
        epub_path = None
        
        # Try different sources in order of preference
        sources = [
            # Direct selection
            lambda: getattr(self, 'selected_epub_path', None),
            # From config
            lambda: self.config.get('last_epub_path', None) if hasattr(self, 'config') else None,
            # From file path variable (if it exists)
            lambda: self.epub_file_path if hasattr(self, 'epub_file_path') and self.epub_file_path else None,
            # From current translation
            lambda: getattr(self, 'current_epub_path', None),
        ]
        
        for source in sources:
            try:
                path = source()
                if path and os.path.exists(path):
                    epub_path = path
                    print(f"[DEBUG] Found EPUB path from source: {path}")  # Debug line
                    break
            except Exception as e:
                print(f"[DEBUG] Error checking source: {e}")  # Debug line
                continue
        
        if not epub_path:
            print("[DEBUG] No EPUB path found from any source")  # Debug line
        
        return epub_path
    
    def _clean_old_backups(self, backup_dir, original_name, max_backups):
        """Remove old backups exceeding the limit"""
        try:
            # Find all backups for this glossary
            prefix = os.path.splitext(original_name)[0]
            backups = []
            
            for file in os.listdir(backup_dir):
                if file.startswith(prefix) and file.endswith('.json'):
                    file_path = os.path.join(backup_dir, file)
                    backups.append((file_path, os.path.getmtime(file_path)))
            
            # Sort by modification time (oldest first)
            backups.sort(key=lambda x: x[1])
            
            # Remove oldest backups if exceeding limit
            while len(backups) > max_backups:
                old_backup = backups.pop(0)
                os.remove(old_backup[0])
                self.append_log(f"🗑️ Removed old backup: {os.path.basename(old_backup[0])}")
                
        except Exception as e:
            self.append_log(f"⚠️ Error cleaning old backups: {str(e)}")
        
    def open_manga_translator(self):
        """Open manga translator in a new window"""
        if not MANGA_SUPPORT:
            QMessageBox.warning(self, "Not Available", "Manga translation modules not found.")
            return
        
        # Always open directly - model preloading will be handled inside the manga tab
        self._open_manga_translator_direct()
    
    def _open_manga_translator_direct(self):
        """Open manga translator directly without loading screen"""
        # Import PySide6 components for the manga translator
        try:
            from PySide6.QtWidgets import QApplication, QDialog, QWidget, QVBoxLayout, QScrollArea
            from PySide6.QtCore import Qt
        except ImportError:
            QMessageBox.critical(self, "Missing Dependency", 
                               "PySide6 is required for manga translation. Please install it:\npip install PySide6")
            return

        # If dialog already exists, just show and focus it to preserve exact state
        try:
            if hasattr(self, "_manga_dialog") and self._manga_dialog is not None:
                # Show with fade-in animation
                try:
                    from dialog_animations import show_dialog_with_fade
                    show_dialog_with_fade(self._manga_dialog, duration=250)
                except Exception:
                    self._manga_dialog.show()
                # Bring to front and focus
                try:
                    self._manga_dialog.raise_()
                    self._manga_dialog.activateWindow()
                except Exception:
                    pass
                return
        except Exception:
            # If the old reference is invalid, recreate below
            self._manga_dialog = None
        
        # Create or get QApplication instance
        app = QApplication.instance()
        if not app:
            # Set DPI awareness before creating QApplication
            try:
                QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
            except:
                pass
            app = QApplication(sys.argv)
        
        # Create PySide6 dialog with standard window controls
        dialog = QDialog()
        dialog.setWindowTitle("🎌 Manga Panel Translator")
        
        # Enable maximize button and standard window controls (minimize, maximize, close)
        dialog.setWindowFlags(
            Qt.Window |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )
        # Do not delete widgets on close; we'll hide instead to retain exact state
        dialog.setAttribute(Qt.WA_DeleteOnClose, False)
        
        # Set icon if available
        try:
            icon_path = os.path.join(self.base_dir, 'Halgakos.ico')
            if os.path.exists(icon_path):
                from PySide6.QtGui import QIcon
                dialog.setWindowIcon(QIcon(icon_path))
        except Exception:
            pass
        
        # Size and position the dialog
        # Use availableGeometry to exclude taskbar and other system UI
        screen = app.primaryScreen().availableGeometry()
        dialog_width = min(1400, int(screen.width() * 0.95))  # Increased from 900 to 1400 for 2-column layout
        dialog_height = int(screen.height() * 0.90)  # Use 90% of available screen height for safety margin
        dialog.resize(dialog_width, dialog_height)
        
        # Center the dialog within available screen space
        dialog_x = screen.x() + (screen.width() - dialog_width) // 2
        dialog_y = screen.y() + (screen.height() - dialog_height) // 2
        dialog.move(dialog_x, dialog_y)
        
        # Create scrollable content area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create main content widget
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        
        # Set dialog layout
        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(0, 0, 0, 0)
        dialog_layout.addWidget(scroll_area)
        
        # Initialize the manga translator interface with PySide6 widget
        self.manga_translator = MangaTranslationTab(content_widget, self, dialog, scroll_area)
        
        # Intercept window close: hide instead of destroy to preserve state
        def _handle_close(event):
            try:
                event.ignore()
                dialog.hide()
            except Exception:
                # Best-effort: still hide on any error
                try:
                    event.ignore()
                except Exception:
                    pass
                dialog.hide()
        dialog.closeEvent = _handle_close
        
        # Keep reference to prevent garbage collection and allow reuse
        self._manga_dialog = dialog
        
        # Show the dialog with smooth fade-in animation
        try:
            from dialog_animations import show_dialog_with_fade
            show_dialog_with_fade(dialog, duration=250)
        except Exception:
            # Fallback to normal show if animation fails
            dialog.show()

        
    def _init_default_prompts(self):
        """Initialize all default prompt templates"""
        self.default_manual_glossary_prompt = """Extract character names and important terms from the following text.

Output format:
{fields}

Rules:
- Output ONLY CSV lines in the exact format shown above
- No headers, no extra text, no JSON
- One entry per line
- Leave gender empty for terms (just end with comma)
"""
        
        self.default_auto_glossary_prompt = """You are extracting a targeted glossary from a {language} novel.
Focus on identifying:
1. Character names with their honorifics
2. Important titles and ranks
3. Frequently mentioned terms (min frequency: {min_frequency})

Extract up to {max_names} character names and {max_titles} titles.
Prioritize names that appear with honorifics or in important contexts.
Return the glossary in a simple key-value format.
        """
        
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
{translations}
        """
    
    def _init_variables(self):
        """Initialize all configuration variables"""
        # Load saved prompts
        self.manual_glossary_prompt = self.config.get('manual_glossary_prompt', self.default_manual_glossary_prompt)
        self.auto_glossary_prompt = self.config.get('auto_glossary_prompt', self.default_auto_glossary_prompt)
        self.rolling_summary_system_prompt = self.config.get('rolling_summary_system_prompt', self.default_rolling_summary_system_prompt)
        self.rolling_summary_user_prompt = self.config.get('rolling_summary_user_prompt', self.default_rolling_summary_user_prompt)
        self.append_glossary_prompt = self.config.get('append_glossary_prompt', "- Follow this reference glossary for consistent translation (Do not output any raw entries):\n")
        self.translation_chunk_prompt = self.config.get('translation_chunk_prompt', self.default_translation_chunk_prompt)
        self.image_chunk_prompt = self.config.get('image_chunk_prompt', self.default_image_chunk_prompt)
        
        self.custom_glossary_fields = self.config.get('custom_glossary_fields', [])
        self.token_limit_disabled = self.config.get('token_limit_disabled', False)
        self.api_key_visible = False  # Default to hidden
        
        if 'glossary_duplicate_key_mode' not in self.config:
            self.config['glossary_duplicate_key_mode'] = 'fuzzy'
        # Initialize fuzzy threshold variable
        if not hasattr(self, 'fuzzy_threshold_var'):
            self.fuzzy_threshold_var = self.config.get('glossary_fuzzy_threshold', 0.90)
        
        # Create all config variables with helper
        def create_var(var_type, key, default):
            # For PySide6 conversion: just return the value directly
            return self.config.get(key, default)
                
        # Boolean variables
        bool_vars = [
            ('rolling_summary_var', 'use_rolling_summary', False),
            ('translation_history_rolling_var', 'translation_history_rolling', False),
            ('glossary_history_rolling_var', 'glossary_history_rolling', False),
            ('translate_book_title_var', 'translate_book_title', True),
            ('enable_auto_glossary_var', 'enable_auto_glossary', False),
            ('append_glossary_var', 'append_glossary', False),
            ('retry_truncated_var', 'retry_truncated', False),
            ('retry_duplicate_var', 'retry_duplicate_bodies', False),
            ('preserve_original_text_var', 'preserve_original_text_on_failure', False),
            # NEW: QA scanning helpers
            ('qa_auto_search_output_var', 'qa_auto_search_output', True),
            ('scan_phase_enabled_var', 'scan_phase_enabled', False),
            ('indefinite_rate_limit_retry_var', 'indefinite_rate_limit_retry', True),
            # Keep existing variables intact
            ('enable_image_translation_var', 'enable_image_translation', False),
            ('process_webnovel_images_var', 'process_webnovel_images', True),
            # REMOVED: ('comprehensive_extraction_var', 'comprehensive_extraction', False),
            ('hide_image_translation_label_var', 'hide_image_translation_label', True),
            ('retry_timeout_var', 'retry_timeout', True),
            ('batch_translation_var', 'batch_translation', False),
            ('conservative_batching_var', 'conservative_batching', True),
            ('disable_epub_gallery_var', 'disable_epub_gallery', False),
            # NEW: Disable automatic cover creation (affects extraction and EPUB cover page)
            ('disable_automatic_cover_creation_var', 'disable_automatic_cover_creation', False),
            ('disable_zero_detection_var', 'disable_zero_detection', True),
            ('use_header_as_output_var', 'use_header_as_output', False),
            ('emergency_restore_var', 'emergency_paragraph_restore', False),
            ('contextual_var', 'contextual', False),
            ('REMOVE_AI_ARTIFACTS_var', 'REMOVE_AI_ARTIFACTS', False),
            ('enable_watermark_removal_var', 'enable_watermark_removal', True),
            ('save_cleaned_images_var', 'save_cleaned_images', False),
            ('advanced_watermark_removal_var', 'advanced_watermark_removal', False),
            ('enable_decimal_chapters_var', 'enable_decimal_chapters', False),
            ('disable_gemini_safety_var', 'disable_gemini_safety', False),
            ('single_api_image_chunks_var', 'single_api_image_chunks', False),

        ]
        
        for var_name, key, default in bool_vars:
            setattr(self, var_name, create_var(bool, key, default))
        
        # Translate special files - with backward compatibility for old translate_cover_html setting
        self.translate_special_files_var = self.config.get('translate_special_files', 
                                                           self.config.get('translate_cover_html', False))
        
        # String variables
        str_vars = [
            ('summary_role_var', 'summary_role', 'user'),
            ('rolling_summary_exchanges_var', 'rolling_summary_exchanges', '5'),
            ('rolling_summary_mode_var', 'rolling_summary_mode', 'append'),
            # New: how many summaries to retain in append mode
            ('rolling_summary_max_entries_var', 'rolling_summary_max_entries', '5'),
            ('reinforcement_freq_var', 'reinforcement_frequency', '10'),
            ('max_retry_tokens_var', 'max_retry_tokens', '16384'),
            ('duplicate_lookback_var', 'duplicate_lookback_chapters', '5'),
            ('glossary_min_frequency_var', 'glossary_min_frequency', '2'),
            ('glossary_max_names_var', 'glossary_max_names', '50'),
            ('glossary_max_titles_var', 'glossary_max_titles', '30'),
            ('glossary_batch_size_var', 'glossary_batch_size', '50'),
            ('webnovel_min_height_var', 'webnovel_min_height', '1000'),
            ('max_images_per_chapter_var', 'max_images_per_chapter', '1'),
            ('image_chunk_height_var', 'image_chunk_height', '1500'),
            ('chunk_timeout_var', 'chunk_timeout', '900'),
            ('batch_size_var', 'batch_size', '3'),
            ('chapter_number_offset_var', 'chapter_number_offset', '0'),
            ('compression_factor_var', 'compression_factor', '1.0'),
            # NEW: scanning phase mode (quick-scan/aggressive/ai-hunter/custom)
            ('scan_phase_mode_var', 'scan_phase_mode', 'quick-scan') 
        ]
        
        for var_name, key, default in str_vars:
            setattr(self, var_name, create_var(str, key, str(default)))
        
        # NEW: Initialize extraction mode variable
        self.extraction_mode_var = self.config.get('extraction_mode', 'smart')
        
        self.book_title_prompt = self.config.get('book_title_prompt', 
            "Translate this book title to English while retaining any acronyms:")
        # Initialize book title system prompt
        if 'book_title_system_prompt' not in self.config:
            self.config['book_title_system_prompt'] = "You are a translator. Respond with only the translated text, nothing else. Do not add any explanation or additional content."
        
        # Profiles
        self.prompt_profiles = self.config.get('prompt_profiles', self.default_prompts.copy())
        active = self.config.get('active_profile', next(iter(self.prompt_profiles)))
        self.profile_var = active
        self.lang_var = self.profile_var
        
        # Detection mode
        self.duplicate_detection_mode_var = self.config.get('duplicate_detection_mode', 'basic')

    def _setup_gui(self):
        """Initialize all GUI components"""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout (QGridLayout for precise control)
        self.frame = QGridLayout(central_widget)
        self.frame.setContentsMargins(10, 10, 10, 10)
        self.frame.setVerticalSpacing(8)
        self.frame.setHorizontalSpacing(5)
        
        # Configure grid column stretches
        for i in range(5):
            self.frame.setColumnStretch(i, 1 if i in [1, 3] else 0)
        
        # Configure grid row stretches and minimum heights
        # Make the log row (row 10) greedily consume extra space on window resize
        for r in range(12):
            # Only the log row should stretch significantly
            self.frame.setRowStretch(r, 10 if r == 10 else (0 if r != 9 else 1))
            if r == 9:
                # Keep a modest minimum for the prompt row but do not let it grow too much
                self.frame.setRowMinimumHeight(r, 180)
            elif r == 10:
                # Log row minimum; will expand aggressively due to stretch factor
                self.frame.setRowMinimumHeight(r, 200)
            elif r == 11:
                # Toolbar row - ensure it stays visible with minimum height
                self.frame.setRowMinimumHeight(r, 55)  # Ensure toolbar is always visible
                # Keep stretch at 0 to prevent it from expanding
        
        # Store row stretch defaults for fullscreen toggle
        self._default_row_stretches = {r: (10 if r == 10 else (1 if r == 9 else 0)) for r in range(12)}
        
        # Create UI elements using helper methods
        self.create_file_section()
        self._create_model_section()
        self._create_profile_section()
        self._create_settings_section()
        self._create_api_section()
        self._create_prompt_section()
        self._create_log_section()
        
        # Add bottom toolbar to layout
        bottom_toolbar = self._make_bottom_toolbar()
        self.frame.addWidget(bottom_toolbar, 11, 0, 1, 5)  # Span all 5 columns at row 11
        
        # Apply token limit state
        if self.token_limit_disabled:
            self.token_limit_entry.setEnabled(False)
            self.toggle_token_btn.setText("Enable Input Token Limit")
            self.toggle_token_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")  # success-outline
        
        self.on_profile_select()
        self.append_log("🚀 Glossarion v6.1.0 - Ready to use!")
        self.append_log("💡 Click any function button to load modules automatically")
        
        # Restore last selected input files if available
        try:
            last_files = self.config.get('last_input_files', []) if hasattr(self, 'config') else []
            if isinstance(last_files, list) and last_files:
                existing = [p for p in last_files if isinstance(p, str) and os.path.exists(p)]
                if existing:
                    # Populate the entry and internal state using shared handler
                    self._handle_file_selection(existing)
                    self.append_log(f"📁 Restored last selection: {len(existing)} file(s)")
        except Exception:
            pass
    
    def _add_combobox_arrow(self, combobox):
        """Add a unicode arrow overlay to a combobox"""
        from PySide6.QtCore import QTimer
        
        arrow_label = QLabel("▼", combobox)
        arrow_label.setStyleSheet("""
            QLabel {
                color: white;
                background: transparent;
                font-size: 10pt;
                border: none;
            }
        """)
        arrow_label.setAlignment(Qt.AlignCenter)
        arrow_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        def position_arrow():
            try:
                if arrow_label and combobox:
                    width = combobox.width()
                    height = combobox.height()
                    arrow_label.setGeometry(width - 20, (height - 16) // 2, 20, 16)
            except RuntimeError:
                pass
        
        # Position arrow when combobox is resized
        original_resize = combobox.resizeEvent
        def new_resize(event):
            original_resize(event)
            position_arrow()
        combobox.resizeEvent = new_resize
        
        # Initial position
        QTimer.singleShot(0, position_arrow)
    
    def _create_styled_checkbox(self, text):
        """Create a checkbox with proper checkmark using text overlay - from manga integration"""
        from PySide6.QtWidgets import QCheckBox, QLabel
        from PySide6.QtCore import Qt, QTimer
        
        checkbox = QCheckBox(text)
        checkbox.setStyleSheet("""
            QCheckBox {
                color: white;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #5a9fd4;
                border-radius: 2px;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background-color: #5a9fd4;
                border-color: #5a9fd4;
            }
            QCheckBox::indicator:hover {
                border-color: #7bb3e0;
            }
            QCheckBox:disabled {
                color: #666666;
            }
            QCheckBox::indicator:disabled {
                background-color: #1a1a1a;
                border-color: #3a3a3a;
            }
        """)
        
        # Create checkmark overlay
        checkmark = QLabel("✓", checkbox)
        checkmark.setStyleSheet("""
            QLabel {
                color: white;
                background: transparent;
                font-weight: bold;
                font-size: 11px;
            }
        """)
        checkmark.setAlignment(Qt.AlignCenter)
        checkmark.hide()
        checkmark.setAttribute(Qt.WA_TransparentForMouseEvents)  # Make checkmark click-through
        
        # Position checkmark properly after widget is shown
        def position_checkmark():
            try:
                # Check if checkmark still exists and is valid
                if checkmark and not checkmark.isHidden() or True:  # Always try to set geometry
                    checkmark.setGeometry(2, 1, 14, 14)
            except RuntimeError:
                # Widget was already deleted
                pass
        
        # Show/hide checkmark based on checked state
        def update_checkmark():
            try:
                # Check if both widgets still exist
                if checkbox and checkmark:
                    if checkbox.isChecked():
                        position_checkmark()
                        checkmark.show()
                    else:
                        checkmark.hide()
            except RuntimeError:
                # Widget was already deleted
                pass
        
        checkbox.stateChanged.connect(update_checkmark)
        # Delay initial positioning to ensure widget is properly rendered
        QTimer.singleShot(0, lambda: (position_checkmark(), update_checkmark()))
        
        return checkbox
    

    def create_file_section(self):
        """Create file selection section with multi-file support"""
        # Initialize file selection variables
        self.selected_files = []
        self.current_file_index = 0
        
        # File label
        file_label = QLabel("Input File(s):")
        self.frame.addWidget(file_label, 0, 0, Qt.AlignLeft)
        
        # File entry
        self.entry_epub = QLineEdit()
        self.entry_epub.setText("No file selected")
        self.entry_epub.setMinimumWidth(400)
        self.frame.addWidget(self.entry_epub, 0, 1, 1, 3)  # row, col, rowspan, colspan
        
        # Create browse menu button with dropdown
        self.btn_browse_menu = QPushButton("Browse ▼")
        self.btn_browse_menu.setMinimumWidth(100)
        self.btn_browse_menu.setStyleSheet("background-color: #007bff; color: white; font-weight: bold;")  # primary
        
        # Create browse menu
        self.browse_menu = QMenu(self)
        self.browse_menu.addAction("📄 Select Files", self.browse_files)
        self.browse_menu.addAction("📁 Select Folder", self.browse_folder)
        self.browse_menu.addSeparator()
        self.browse_menu.addAction("🗑️ Clear Selection", self.clear_file_selection)
        
        # Attach menu to button
        self.btn_browse_menu.setMenu(self.browse_menu)
        self.frame.addWidget(self.btn_browse_menu, 0, 4)
        
        # File selection status label (shows file count and details)
        self.file_status_label = QLabel("")
        self.file_status_label.setStyleSheet("color: #17a2b8; font-size: 9pt;")
        self.frame.addWidget(self.file_status_label, 1, 1, 1, 3, Qt.AlignLeft)
        
        # Google Cloud Credentials button
        self.gcloud_button = QPushButton("GCloud Creds")
        self.gcloud_button.clicked.connect(self.select_google_credentials)
        self.gcloud_button.setMinimumWidth(100)
        self.gcloud_button.setEnabled(False)
        # Store both enabled and disabled styles
        self.gcloud_button_enabled_style = "background-color: #007bff; color: white; font-weight: bold;"  # primary blue
        self.gcloud_button_disabled_style = "background-color: #3a3a3a; color: #888888; font-weight: bold;"  # dark gray
        self.gcloud_button.setStyleSheet(self.gcloud_button_disabled_style)
        self.frame.addWidget(self.gcloud_button, 2, 4)
        
        # Vertex AI Location text entry
        self.vertex_location_var = self.config.get('vertex_ai_location', 'us-east5')
        self.vertex_location_entry = QLineEdit(self.vertex_location_var)
        self.vertex_location_entry.setMinimumWidth(100)
        self.frame.addWidget(self.vertex_location_entry, 3, 4)
        
        # Hide by default
        self.vertex_location_entry.hide()
        
        # Status label for credentials - positioned BELOW the vertex location entry (row 4, column 4)
        self.gcloud_status_label = QLabel("")
        self.gcloud_status_label.setStyleSheet("color: #6c757d; font-size: 9pt;")
        self.gcloud_status_label.setWordWrap(True)  # Allow text wrapping
        self.gcloud_status_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.frame.addWidget(self.gcloud_status_label, 4, 4, Qt.AlignLeft)
        
        # Optional: Add checkbox for enhanced functionality
        options_frame = QWidget()
        options_layout = QHBoxLayout(options_frame)
        options_layout.setContentsMargins(0, 0, 0, 0)
        
        # Deep scan option for folders
        self.deep_scan_var = self.config.get('deep_scan', False)
        self.deep_scan_check = self._create_styled_checkbox("include subfolders")
        self.deep_scan_check.setChecked(self.deep_scan_var)
        self.deep_scan_check.stateChanged.connect(self._on_deep_scan_changed)
        options_layout.addWidget(self.deep_scan_check)
        options_layout.addStretch()
        
        self.frame.addWidget(options_frame, 1, 4)
    
    def _on_deep_scan_changed(self, state):
        """Handle deep scan checkbox state change"""
        self.deep_scan_var = (state == Qt.Checked)

    def select_google_credentials(self):
        """Select Google Cloud credentials JSON file"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Google Cloud Credentials JSON",
            "",
            "JSON files (*.json);;All files (*.*)"
        )
        
        if filename:
            try:
                # Validate it's a valid Google Cloud credentials file
                with open(filename, 'r') as f:
                    creds_data = json.load(f)
                    if 'type' in creds_data and 'project_id' in creds_data:
                        # Save to config
                        self.config['google_cloud_credentials'] = filename
                        self.save_config()
                        
                        # Update UI
                        self.gcloud_status_label.setText(
                            f"✓ Credentials: {os.path.basename(filename)} (Project: {creds_data.get('project_id', 'Unknown')})"
                        )
                        self.gcloud_status_label.setStyleSheet("color: green; font-size: 9pt;")
                        
                        # Set environment variable for child processes
                        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = filename
                        
                        self.append_log(f"Google Cloud credentials loaded: {os.path.basename(filename)}")
                    else:
                        QMessageBox.critical(
                            self,
                            "Error", 
                            "Invalid Google Cloud credentials file. Please select a valid service account JSON file."
                        )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load credentials: {str(e)}")

    def on_model_change(self, index=None):
        """Handle model selection change from dropdown or manual input"""
        # Get the current model value (from dropdown or manually typed)
        model = self.model_var
        
        # Show Google Cloud Credentials button for Vertex AI models AND Google Translate (paid)
        needs_google_creds = False
        
        if '@' in model or model.startswith('vertex/') or model.startswith('vertex_ai/'):
            needs_google_creds = True
            self.vertex_location_entry.show()  # Show location selector for Vertex
        elif model.lower() == 'google-translate':  # Exact match for paid Google Translate (not google-translate-free)
            needs_google_creds = True
            self.vertex_location_entry.hide()  # Hide location selector for Google Translate
        
        if needs_google_creds:
            self.gcloud_button.setEnabled(True)
            self.gcloud_button.setStyleSheet(self.gcloud_button_enabled_style)  # Apply enabled style
            
            # Check if credentials are already loaded
            if self.config.get('google_cloud_credentials'):
                creds_path = self.config['google_cloud_credentials']
                if os.path.exists(creds_path):
                    try:
                        with open(creds_path, 'r') as f:
                            creds_data = json.load(f)
                            project_id = creds_data.get('project_id', 'Unknown')
                            
                            # Different status messages for different services
                            if model == 'google-translate':
                                status_text = f"✓ Google Translate ready\n(Project: {project_id})"
                            else:
                                status_text = f"✓ Credentials: {os.path.basename(creds_path)} (Project: {project_id})"
                            
                            self.gcloud_status_label.setText(status_text)
                            self.gcloud_status_label.setStyleSheet("color: green; font-size: 9pt;")
                    except:
                        self.gcloud_status_label.setText("⚠ Error reading credentials")
                        self.gcloud_status_label.setStyleSheet("color: red; font-size: 9pt;")
                else:
                    self.gcloud_status_label.setText("⚠ Credentials file not found")
                    self.gcloud_status_label.setStyleSheet("color: red; font-size: 9pt;")
            else:
                # Different prompts for different services
                if model == 'google-translate':
                    warning_text = "⚠ Google Cloud credentials needed for Translate API"
                else:
                    warning_text = "⚠ No Google Cloud credentials selected"
                
                self.gcloud_status_label.setText(warning_text)
                self.gcloud_status_label.setStyleSheet("color: orange; font-size: 9pt;")
        else:
            # Not a Google service, hide everything
            self.gcloud_button.setEnabled(False)
            self.gcloud_button.setStyleSheet(self.gcloud_button_disabled_style)  # Apply disabled style
            self.vertex_location_entry.hide()
            self.gcloud_status_label.setText("")

    # PySide6 helper method for model text changes
    def _on_model_text_changed(self, text):
        """Handle model combobox text changes"""
        # Update the model_var to the current text
        self.model_var = text
        
        # Re-check model requirements (GCloud, POE, etc.)
        self.on_model_change()
        
        # Check for POE model
        if hasattr(self, '_check_poe_model'):
            self._check_poe_model()
    
    # Also add this to bind manual typing events to the combobox
    def setup_model_combobox_bindings(self):
        """Setup bindings for manual model input in combobox with autocomplete"""
        # PySide6: QComboBox already has built-in autocomplete
        # We can set the completer for better UX
        from PySide6.QtWidgets import QCompleter
        from PySide6.QtCore import Qt
        
        completer = QCompleter(self._model_all_values)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self.model_combo.setCompleter(completer)
        
    # Note: These Tkinter-specific methods are replaced by PySide6's QCompleter
    # which provides built-in autocomplete functionality
    def _create_model_section(self):
        """Create model selection section"""
        # Model label
        model_label = QLabel("Model:")
        self.frame.addWidget(model_label, 1, 0, Qt.AlignLeft)
        
        # Get default model and model list
        default_model = self.config.get('model', 'gemini-2.0-flash')
        self.model_var = default_model
        models = get_model_options()
        self._model_all_values = models
        
        # Create editable combobox
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(models)
        self.model_combo.setCurrentText(default_model)
        self.model_combo.setMaximumWidth(450)
        # Add custom styling with unicode arrow
        self.model_combo.setStyleSheet("""
            QComboBox::down-arrow {
                image: none;
                width: 12px;
                height: 12px;
                border: none;
            }
        """)
        # Add unicode arrow using a label overlay
        self._add_combobox_arrow(self.model_combo)
        self.frame.addWidget(self.model_combo, 1, 1, 1, 2)  # row, col, rowspan, colspan
        
        # Track previous text to make autocomplete less aggressive
        self._model_prev_text = default_model
        
        # Connect signals
        self.model_combo.currentIndexChanged.connect(self.on_model_change)
        self.model_combo.editTextChanged.connect(self._on_model_text_changed)
        
        # Setup autocomplete bindings
        self.setup_model_combobox_bindings()
        
        # Initial check
        self.on_model_change()
    
    def _create_profile_section(self):
        """Create profile/profile section"""
        # Profile label
        profile_label = QLabel("Profile:")
        self.frame.addWidget(profile_label, 2, 0, Qt.AlignLeft)
        
        # Profile combobox
        self.profile_menu = QComboBox()
        self.profile_menu.setEditable(True)
        self.profile_menu.addItems(list(self.prompt_profiles.keys()))
        self.profile_menu.setCurrentText(self.profile_var)
        self.profile_menu.setMaximumWidth(380)
        # Add custom styling with unicode arrow
        self.profile_menu.setStyleSheet("""
            QComboBox::down-arrow {
                image: none;
                width: 12px;
                height: 12px;
                border: none;
            }
        """)
        # Add unicode arrow using a label overlay
        self._add_combobox_arrow(self.profile_menu)
        self.frame.addWidget(self.profile_menu, 2, 1)
        
        # Connect signals for profile selection
        self.profile_menu.currentIndexChanged.connect(lambda: self.on_profile_select())
        self.profile_menu.lineEdit().returnPressed.connect(lambda: self.on_profile_select())
        
        # Create a horizontal layout for profile buttons to keep them close together
        profile_buttons_widget = QWidget()
        profile_buttons_layout = QHBoxLayout(profile_buttons_widget)
        profile_buttons_layout.setContentsMargins(0, 0, 0, 0)
        profile_buttons_layout.setSpacing(10)
        
        # Save Profile button
        save_profile_btn = QPushButton("Save Profile")
        save_profile_btn.clicked.connect(self.save_profile)
        save_profile_btn.setFixedWidth(95)
        profile_buttons_layout.addWidget(save_profile_btn)
        
        # Delete Profile button
        delete_profile_btn = QPushButton("Delete Profile")
        delete_profile_btn.clicked.connect(self.delete_profile)
        delete_profile_btn.setFixedWidth(95)
        profile_buttons_layout.addWidget(delete_profile_btn)
        
        profile_buttons_layout.addStretch()
        
        # Add the buttons widget spanning columns 2-3
        self.frame.addWidget(profile_buttons_widget, 2, 2, 1, 2)
    
    def _create_settings_section(self):
        """Create all settings controls"""
        # Threading delay (with extra spacing at top)
        thread_delay_label = QLabel("Threading delay (s):")
        self.frame.addWidget(thread_delay_label, 3, 0, Qt.AlignLeft)
        
        self.thread_delay_entry = QLineEdit()
        self.thread_delay_entry.setText(str(self.thread_delay_var))
        self.thread_delay_entry.setMaximumWidth(80)
        self.frame.addWidget(self.thread_delay_entry, 3, 1, Qt.AlignLeft)

        # API delay (left side)
        api_delay_label = QLabel("API call delay (s):")
        self.frame.addWidget(api_delay_label, 4, 0, Qt.AlignLeft)
        
        self.delay_entry = QLineEdit()
        self.delay_entry.setText(str(self.config.get('delay', 2)))
        self.delay_entry.setMaximumWidth(80)
        self.frame.addWidget(self.delay_entry, 4, 1, Qt.AlignLeft)

        # Optional help text (spanning both columns)
        help_label = QLabel("(0 = simultaneous)")
        help_label.setStyleSheet("color: gray; font-size: 8pt;")
        self.frame.addWidget(help_label, 3, 2, Qt.AlignLeft)
        
        # Chapter Range
        chapter_range_label = QLabel("Chapter range (e.g., 5-10):")
        self.frame.addWidget(chapter_range_label, 5, 0, Qt.AlignLeft)
        
        self.chapter_range_entry = QLineEdit()
        self.chapter_range_entry.setText(self.config.get('chapter_range', ''))
        self.chapter_range_entry.setMaximumWidth(120)
        self.frame.addWidget(self.chapter_range_entry, 5, 1, Qt.AlignLeft)
        
        # Token limit
        token_limit_label = QLabel("Input Token limit:")
        self.frame.addWidget(token_limit_label, 6, 0, Qt.AlignLeft)
        
        self.token_limit_entry = QLineEdit()
        self.token_limit_entry.setText(str(self.config.get('token_limit', 200000)))
        self.token_limit_entry.setMaximumWidth(80)
        self.frame.addWidget(self.token_limit_entry, 6, 1, Qt.AlignLeft)
        
        self.toggle_token_btn = QPushButton("Disable Input Token Limit")
        self.toggle_token_btn.clicked.connect(self.toggle_token_limit)
        self.toggle_token_btn.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold;")
        self.toggle_token_btn.setMinimumWidth(150)
        self.frame.addWidget(self.toggle_token_btn, 7, 1, Qt.AlignLeft)
        
        # Contextual Translation (right side, row 3) - with extra padding on top
        contextual_container = QWidget()
        contextual_layout = QHBoxLayout(contextual_container)
        contextual_layout.setContentsMargins(0, 0, 0, 0)
        contextual_layout.setSpacing(8)
        
        self.contextual_checkbox = self._create_styled_checkbox("Contextual Translation")
        self.contextual_checkbox.setChecked(self.contextual_var)
        self.contextual_checkbox.stateChanged.connect(self._on_contextual_toggle)
        contextual_layout.addWidget(self.contextual_checkbox)
        
        contextual_warning = QLabel("⚠️ May result in duplicate outputs")
        contextual_warning.setStyleSheet("color: #ff9800; font-size: 9pt; font-style: italic;")
        contextual_layout.addWidget(contextual_warning)
        contextual_layout.addStretch()
        
        self.frame.addWidget(contextual_container, 3, 2, 1, 2, Qt.AlignLeft)
        
        # Translation History Limit (row 4)
        self.trans_history_label = QLabel("Translation History Limit:")
        self.frame.addWidget(self.trans_history_label, 4, 2, Qt.AlignLeft)
        
        self.trans_history = QLineEdit()
        self.trans_history.setText(str(self.config.get('translation_history_limit', 2)))
        self.trans_history.setMaximumWidth(60)
        self.frame.addWidget(self.trans_history, 4, 3, Qt.AlignLeft)
        
        # Rolling History (row 5)
        self.rolling_checkbox = self._create_styled_checkbox("Rolling History Window")
        self.rolling_checkbox.setChecked(self.translation_history_rolling_var)
        self.frame.addWidget(self.rolling_checkbox, 5, 2, Qt.AlignLeft)
        
        self.rolling_history_desc = QLabel("(Keep recent history instead of purging)")
        self.rolling_history_desc.setStyleSheet("color: gray; font-size: 9pt;")
        self.frame.addWidget(self.rolling_history_desc, 5, 3, Qt.AlignLeft)
        
        # Temperature (row 6)
        temp_label = QLabel("Temperature:")
        self.frame.addWidget(temp_label, 6, 2, Qt.AlignLeft)
        
        self.trans_temp = QLineEdit()
        self.trans_temp.setText(str(self.config.get('translation_temperature', 0.3)))
        self.trans_temp.setMaximumWidth(60)
        self.frame.addWidget(self.trans_temp, 6, 3, Qt.AlignLeft)
        
        # Batch Translation (row 7) with spinning icon
        batch_container = QWidget()
        batch_layout = QHBoxLayout(batch_container)
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_layout.setSpacing(8)
        
        self.batch_checkbox = self._create_styled_checkbox("Batch Translation")
        self.batch_checkbox.setChecked(self.batch_translation_var)
        self.batch_checkbox.stateChanged.connect(self._on_batch_toggle)
        
        # Add spinning icon next to batch checkbox using spinning helper
        from spinning import create_icon_label, animate_icon
        base_dir = getattr(self, 'base_dir', None)
        self.batch_icon = create_icon_label(size=20, base_dir=base_dir)
        self.batch_checkbox.toggled.connect(lambda: animate_icon(self.batch_icon))
        
        batch_layout.addWidget(self.batch_icon)
        batch_layout.addWidget(self.batch_checkbox)
        batch_layout.addStretch()
        
        self.frame.addWidget(batch_container, 7, 2, Qt.AlignLeft)
        
        self.batch_size_entry = QLineEdit()
        self.batch_size_entry.setText(str(self.batch_size_var))
        self.batch_size_entry.setMaximumWidth(60)
        self.frame.addWidget(self.batch_size_entry, 7, 3, Qt.AlignLeft)
        
        # Set batch entry initial state
        self.batch_size_entry.setEnabled(self.batch_translation_var)
        
        # Hidden entries for compatibility (not displayed, just storing values)
        self.title_trim = QLineEdit()
        self.title_trim.setText(str(self.config.get('title_trim_count', 1)))
        self.title_trim.setMaximumWidth(60)
        self.title_trim.hide()
        
        self.group_trim = QLineEdit()
        self.group_trim.setText(str(self.config.get('group_affiliation_trim_count', 1)))
        self.group_trim.setMaximumWidth(60)
        self.group_trim.hide()
        
        self.traits_trim = QLineEdit()
        self.traits_trim.setText(str(self.config.get('traits_trim_count', 1)))
        self.traits_trim.setMaximumWidth(60)
        self.traits_trim.hide()
        
        self.refer_trim = QLineEdit()
        self.refer_trim.setText(str(self.config.get('refer_trim_count', 1)))
        self.refer_trim.setMaximumWidth(60)
        self.refer_trim.hide()
        
        self.loc_trim = QLineEdit()
        self.loc_trim.setText(str(self.config.get('locations_trim_count', 1)))
        self.loc_trim.setMaximumWidth(60)
        self.loc_trim.hide()
        
        # Set initial state based on contextual translation
        self._on_contextual_toggle()

    def _on_contextual_toggle(self, state=None):
        """Handle contextual translation toggle - enable/disable related controls"""
        # If called with a state (Qt.CheckState), update the var
        if state is not None:
            self.contextual_var = (state == Qt.Checked)
        
        # Always get the current checkbox state to be sure
        if hasattr(self, 'contextual_checkbox'):
            is_contextual = self.contextual_checkbox.isChecked()
            self.contextual_var = is_contextual
        else:
            is_contextual = self.contextual_var
        
        # Enable/disable translation history limit entry and update label color
        if hasattr(self, 'trans_history'):
            self.trans_history.setEnabled(is_contextual)
        if hasattr(self, 'trans_history_label'):
            label_color = 'white' if is_contextual else 'gray'
            self.trans_history_label.setStyleSheet(f"color: {label_color};")
        
        # Enable/disable rolling history checkbox and update description color
        if hasattr(self, 'rolling_checkbox'):
            self.rolling_checkbox.setEnabled(is_contextual)
        if hasattr(self, 'rolling_history_desc'):
            desc_color = 'gray' if is_contextual else '#404040'
            self.rolling_history_desc.setStyleSheet(f"color: {desc_color}; font-size: 9pt;")
    
    def _on_batch_toggle(self, state=None):
        """Handle batch translation toggle - enable/disable batch size entry"""
        # If called with a state (Qt.CheckState), update the var
        if state is not None:
            self.batch_translation_var = (state == Qt.Checked)
        
        # Always get the current checkbox state to be sure
        if hasattr(self, 'batch_checkbox'):
            is_batch = self.batch_checkbox.isChecked()
            self.batch_translation_var = is_batch
        else:
            is_batch = self.batch_translation_var
        
        # Enable/disable batch size entry based on checkbox state
        if hasattr(self, 'batch_size_entry'):
            self.batch_size_entry.setEnabled(is_batch)
    
    def _on_remove_artifacts_toggle(self, state=None):
        """Handle Remove AI Artifacts toggle"""
        # If called with a state (Qt.CheckState), update the var
        if state is not None:
            self.REMOVE_AI_ARTIFACTS_var = (state == Qt.Checked)
    
    def _auto_save_system_prompt(self):
        """Auto-save system prompt to current profile as user types"""
        try:
            # Get current profile name
            if not hasattr(self, 'profile_menu'):
                return
            
            name = self.profile_menu.currentText().strip()
            if not name:
                return
            
            # Get current text from prompt_text
            content = self.prompt_text.toPlainText().strip()
            
            # Update the profile in memory
            self.prompt_profiles[name] = content
            
            # Update config
            self.config['prompt_profiles'] = self.prompt_profiles
            self.config['active_profile'] = name
            
            # Note: We don't call save_profiles() here to avoid constant disk I/O
            # The profile will be saved when the config is saved (on exit or manual save)
        except Exception as e:
            # Silently fail to avoid disrupting user's typing
            pass
    
    def _create_api_section(self):
        """Create API key section"""
        # API Key Label (row 8)
        self.api_key_label = QLabel("OpenAI/Gemini/... API Key:")
        self.frame.addWidget(self.api_key_label, 8, 0, Qt.AlignLeft)
        
        # API Key Entry (row 8, spans 3 columns)
        self.api_key_entry = QLineEdit()
        self.api_key_entry.setEchoMode(QLineEdit.Password)  # Show '*' instead of text
        initial_key = self.config.get('api_key', '')
        if initial_key:
            self.api_key_entry.setText(initial_key)
        self.frame.addWidget(self.api_key_entry, 8, 1, 1, 3)  # row, col, rowspan, colspan
        
        # Show/Hide API Key button (row 8)
        self.show_api_btn = QPushButton("Show")
        self.show_api_btn.clicked.connect(self.toggle_api_visibility)
        self.show_api_btn.setMinimumWidth(100)
        self.frame.addWidget(self.show_api_btn, 8, 4)
        
        # Other Settings button (row 7, column 4)
        other_settings_btn = QPushButton("⚙️  Other Setting")
        other_settings_btn.clicked.connect(self.open_other_settings)
        other_settings_btn.setStyleSheet("background-color: #17a2b8; color: white; font-weight: bold; font-size: 11pt; padding-top: 8px; padding-bottom: 12px;")  # info-outline
        other_settings_btn.setMinimumWidth(120)
        self.frame.addWidget(other_settings_btn, 7, 4)
        
        # Remove AI Artifacts checkbox (row 7, spans all columns)
        self.remove_artifacts_checkbox = self._create_styled_checkbox("Remove AI Artifacts")
        self.remove_artifacts_checkbox.setChecked(self.REMOVE_AI_ARTIFACTS_var)
        self.remove_artifacts_checkbox.stateChanged.connect(self._on_remove_artifacts_toggle)
        self.frame.addWidget(self.remove_artifacts_checkbox, 7, 0, 1, 5, Qt.AlignLeft)
    
    def _create_prompt_section(self):
        """Create system prompt section"""
        # System Prompt Label (row 9, column 0)
        prompt_label = QLabel("System Prompt:")
        self.frame.addWidget(prompt_label, 9, 0, Qt.AlignTop | Qt.AlignLeft)
        
        # System Prompt Text Edit (row 9, spans 3 columns)
        self.prompt_text = QTextEdit()
        self.prompt_text.setMinimumHeight(100)
        # Cap its vertical growth so the log gets most of the extra space
        self.prompt_text.setMaximumHeight(220)
        self.prompt_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.prompt_text.setAcceptRichText(False)  # Plain text only
        
        # Auto-save system prompt as user types
        self.prompt_text.textChanged.connect(self._auto_save_system_prompt)
        
        self.frame.addWidget(self.prompt_text, 9, 1, 1, 3)  # row, col, rowspan, colspan
        
        # Output Token Limit button (row 9, column 0 - below label)
        self.output_btn = QPushButton(f"Output Token Limit: {self.max_output_tokens}")
        self.output_btn.clicked.connect(self.prompt_custom_token_limit)
        self.output_btn.setStyleSheet("background-color: #17a2b8; color: white; font-weight: bold; padding: 8px 6px;")  # info
        self.output_btn.setMinimumWidth(180)
        # Place below the label in a vertical layout
        output_container = QWidget()
        output_layout = QVBoxLayout(output_container)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(prompt_label)
        output_layout.addWidget(self.output_btn)
        output_layout.addStretch()
        self.frame.addWidget(output_container, 9, 0, Qt.AlignTop)
        
        # Run Translation button (row 9, column 4) - Fill the space
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QIcon, QPixmap
        
        # Create a custom button widget with icon above text
        self.run_button = QPushButton()
        self.run_button.clicked.connect(self.run_translation_thread)
        
        # Create a container widget for custom layout
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setContentsMargins(0, 8, 0, 0)
        button_layout.setSpacing(2)  # Minimal spacing between icon and text
        
        # Icon label with rotation support - wrapped in its own container
        # This allows the icon to rotate independently without affecting the label
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Halgakos.ico")
        
        # Create a custom label that supports rotation
        class RotatableLabel(QLabel):
            def __init__(self, parent=None):
                super().__init__(parent)
                self._rotation = 0
                self._original_pixmap = None
            
            def set_rotation(self, angle):
                self._rotation = angle
                if self._original_pixmap:
                    # Create transformation
                    transform = QTransform()
                    transform.rotate(angle)
                    # Apply rotation to pixmap
                    rotated = self._original_pixmap.transformed(transform, Qt.SmoothTransformation)
                    self.setPixmap(rotated)
            
            def get_rotation(self):
                return self._rotation
            
            # Define rotation as a Qt Property for animation
            rotation = Property(float, get_rotation, set_rotation)
            
            def set_original_pixmap(self, pixmap):
                self._original_pixmap = pixmap
                self.setPixmap(pixmap)
        
        # Create icon container to isolate rotation effect
        icon_container = QWidget()
        icon_container.setFixedSize(90, 90)  # Fixed size to prevent layout shift during rotation
        icon_container.setStyleSheet("background-color: transparent;")  # Transparent background
        icon_layout = QVBoxLayout(icon_container)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        icon_layout.setAlignment(Qt.AlignCenter)
        
        self.run_button_icon = RotatableLabel(icon_container)
        self.run_button_icon.setStyleSheet("background-color: transparent;")  # Transparent background for icon label
        if os.path.exists(icon_path):
            # Load the icon at the highest available resolution
            from PySide6.QtGui import QImage
            icon = QIcon(icon_path)
            # Get the largest available size from the icon
            available_sizes = icon.availableSizes()
            if available_sizes:
                largest_size = max(available_sizes, key=lambda s: s.width() * s.height())
                pixmap = icon.pixmap(largest_size)
            else:
                pixmap = QPixmap(icon_path)
            
            # Scale with high-quality transformation
            scaled_pixmap = pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.run_button_icon.set_original_pixmap(scaled_pixmap)
        self.run_button_icon.setAlignment(Qt.AlignCenter)
        icon_layout.addWidget(self.run_button_icon)
        
        # Add icon container to button layout
        button_layout.addWidget(icon_container, alignment=Qt.AlignCenter)
        
        # Create rotation animation (but don't start it yet)
        self.icon_spin_animation = QPropertyAnimation(self.run_button_icon, b"rotation")
        self.icon_spin_animation.setDuration(900)  # 0.9 seconds per rotation (faster)
        self.icon_spin_animation.setStartValue(0)
        self.icon_spin_animation.setEndValue(360)
        self.icon_spin_animation.setLoopCount(-1)  # Infinite loop
        self.icon_spin_animation.setEasingCurve(QEasingCurve.Linear)
        
        # Create a smooth stop animation for graceful deceleration
        self.icon_stop_animation = QPropertyAnimation(self.run_button_icon, b"rotation")
        self.icon_stop_animation.setDuration(800)  # Deceleration time
        self.icon_stop_animation.setEasingCurve(QEasingCurve.OutCubic)  # Smooth deceleration
        
        # Text label - separate from icon, won't be affected by rotation
        self.run_button_text = QLabel("Run Translation")
        self.run_button_text.setAlignment(Qt.AlignCenter)
        self.run_button_text.setStyleSheet("color: white; font-size: 14pt; font-weight: bold;")
        button_layout.addWidget(self.run_button_text)
        button_layout.addStretch()
        
        # Set the container as the button's layout
        self.run_button.setLayout(button_layout)
        
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                border: none;
            }
            QPushButton:disabled {
                background-color: #555555;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        self.run_button.setMinimumWidth(160)
        self.run_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frame.addWidget(self.run_button, 9, 4)
    
    def _create_log_section(self):
        """Create log text area"""
        # Log Text Edit (row 10, spans all 5 columns)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)  # Make it read-only
        self.log_text.setMinimumHeight(200)  # Reduced from 300 to ensure toolbar visibility on low-res screens
        # Make sure it greedily expands vertically and horizontally but respects minimum/maximum constraints
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.log_text.setAcceptRichText(False)  # Plain text only
        self.frame.addWidget(self.log_text, 10, 0, 1, 5)  # row, col, rowspan, colspan
        
        # Setup context menu
        self.log_text.setContextMenuPolicy(Qt.CustomContextMenu)
        self.log_text.customContextMenuRequested.connect(self._show_context_menu)
        
        # Connect scrollbar to detect manual scrolling
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.valueChanged.connect(self._on_log_scroll)

    def _check_poe_model(self, *args):
        """Automatically show POE helper when POE model is selected"""
        model = self.model_var.lower()  # model_var is now a string, not a StringVar
        
        # Check if POE model is selected
        if model.startswith('poe/'):
            current_key = self.api_key_entry.text().strip()
            
            # Only show helper if no valid POE cookie is set
            if not current_key.startswith('p-b:'):
                # Use a flag to prevent showing multiple times in same session
                if not getattr(self, '_poe_helper_shown', False):
                    self._poe_helper_shown = True
                    # Use QTimer instead of after
                    QTimer.singleShot(100, self._show_poe_setup_dialog)
        else:
            # Reset flag when switching away from POE
            self._poe_helper_shown = False

    def _show_poe_setup_dialog(self):
        """Show POE cookie setup dialog"""
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("POE Authentication Required")
        # Use screen ratios for sizing
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.34)  # 34% of screen width
        height = int(screen.height() * 0.44)  # 44% of screen height
        dialog.setMinimumSize(width, height)
        
        # Main layout
        main_layout = QVBoxLayout(dialog)
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Content widget
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        # Header
        header_label = QLabel("POE Cookie Authentication")
        header_font = QFont()
        header_font.setPointSize(12)
        header_font.setBold(True)
        header_label.setFont(header_font)
        content_layout.addWidget(header_label)
        
        # Important notice
        notice_label1 = QLabel("⚠️ POE uses HttpOnly cookies that cannot be accessed by JavaScript")
        notice_label1.setStyleSheet("color: red; font-weight: bold;")
        content_layout.addWidget(notice_label1)
        
        notice_label2 = QLabel("You must manually copy the cookie from Developer Tools")
        notice_label2.setStyleSheet("color: gray;")
        content_layout.addWidget(notice_label2)
        
        content_layout.addSpacing(10)
        
        # Instructions
        self._create_poe_manual_instructions(content_layout)
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
        
        # Button
        def close_dialog():
            dialog.accept()
            # Check if user added a cookie
            current_key = self.api_key_entry.text().strip()
            model = self.model_var.lower()
            if model.startswith('poe/') and not current_key.startswith('p-b:'):
                self.append_log("⚠️ POE models require cookie authentication. Please add your p-b cookie to the API key field.")
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(close_dialog)
        close_btn.setStyleSheet("background-color: #6c757d; color: white;")
        main_layout.addWidget(close_btn)
        
        dialog.exec()

    def _create_poe_manual_instructions(self, parent_layout):
        """Create manual instructions for getting POE cookie"""
        # Group box for instructions
        group_box = QLabel("How to Get Your POE Cookie")
        group_box.setStyleSheet("font-weight: bold; font-size: 11pt;")
        parent_layout.addWidget(group_box)
        
        # Step-by-step with visual formatting
        steps = [
            ("1.", "Go to poe.com and LOG IN to your account", None),
            ("2.", "Press F12 to open Developer Tools", None),
            ("3.", "Navigate to:", None),
            ("", "• Chrome/Edge: Application → Cookies → https://poe.com", "indent"),
            ("", "• Firefox: Storage → Cookies → https://poe.com", "indent"),
            ("", "• Safari: Storage → Cookies → poe.com", "indent"),
            ("4.", "Find the cookie named 'p-b'", None),
            ("5.", "Double-click its Value to select it", None),
            ("6.", "Copy the value (Ctrl+C or right-click → Copy)", None),
            ("7.", "In Glossarion's API key field, type: p-b:", None),
            ("8.", "Paste the cookie value after p-b:", None)
        ]
        
        for num, text, style in steps:
            step_widget = QWidget()
            step_layout = QHBoxLayout(step_widget)
            step_layout.setContentsMargins(20 if style != "indent" else 40, 2, 0, 2)
            
            if num:
                num_label = QLabel(num)
                num_label.setStyleSheet("font-weight: bold;")
                num_label.setFixedWidth(30)
                step_layout.addWidget(num_label)
            
            text_label = QLabel(text)
            step_layout.addWidget(text_label)
            step_layout.addStretch()
            
            parent_layout.addWidget(step_widget)
        
        parent_layout.addSpacing(10)
        
        # Example
        example_label = QLabel("Example API Key Format")
        example_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        parent_layout.addWidget(example_label)
        
        example_entry = QLineEdit()
        example_entry.setText("p-b:RyP5ORQXFO8qXbiTBKD2vA%3D%3D")
        example_entry.setReadOnly(True)
        example_entry.setFont(QFont("Consolas", 11))
        parent_layout.addWidget(example_entry)
        
        parent_layout.addSpacing(10)
        
        # Additional info
        info_text = """Note: The cookie value is usually a long string ending with %3D%3D
If you see multiple p-b cookies, use the one with the longest value."""
        
        info_label = QLabel(info_text)
        info_label.setStyleSheet("color: gray;")
        info_label.setWordWrap(True)
        parent_layout.addWidget(info_label)

    def open_async_processing(self):
        """Open the async processing dialog"""
        # Check if translation is running
        if hasattr(self, 'translation_thread') and self.translation_thread and self.translation_thread.is_alive():
            self.append_log("⚠️ Cannot open async processing while translation is in progress.")
            QMessageBox.warning(self, "Process Running", "Please wait for the current translation to complete.")
            return
        
        # Check if glossary extraction is running
        if hasattr(self, 'glossary_thread') and self.glossary_thread and self.glossary_thread.is_alive():
            self.append_log("⚠️ Cannot open async processing while glossary extraction is in progress.")
            QMessageBox.warning(self, "Process Running", "Please wait for glossary extraction to complete.")
            return
        
        # Check if file is selected
        if not hasattr(self, 'file_path') or not self.file_path:
            self.append_log("⚠️ Please select a file before opening async processing.")
            QMessageBox.warning(self, "No File Selected", "Please select an EPUB or TXT file first.")
            return
        
        try:
            # Lazy import the async processor
            if not hasattr(self, '_async_processor_imported'):
                self.append_log("Loading async processing module...")
                from async_api_processor import show_async_processing_dialog
                self._async_processor_imported = True
                self._show_async_processing_dialog = show_async_processing_dialog
            
            # Show the dialog
            self.append_log("Opening async processing dialog...")
            self._show_async_processing_dialog(self, self)
            
        except ImportError as e:
            self.append_log(f"❌ Failed to load async processing module: {e}")
            QMessageBox.critical(
                self,
                "Module Not Found", 
                "The async processing module could not be loaded.\n"
                "Please ensure async_api_processor.py is in the same directory."
            )
        except Exception as e:
            self.append_log(f"❌ Error opening async processing: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open async processing: {str(e)}")

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
            
            # Initialize global variables to None FIRST to avoid NameError
            global translation_main, translation_stop_flag, translation_stop_check
            global glossary_main, glossary_stop_flag, glossary_stop_check
            global fallback_compile_epub, scan_html_folder
            
            # Set all to None initially in case imports fail
            translation_main = None
            translation_stop_flag = None
            translation_stop_check = None
            glossary_main = None
            glossary_stop_flag = None
            glossary_stop_check = None
            fallback_compile_epub = None
            scan_html_folder = None
            
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
                                translation_main = TransateKRtoEN.main
                                translation_stop_flag = TransateKRtoEN.set_stop_flag
                                translation_stop_check = TransateKRtoEN.is_stop_requested if hasattr(TransateKRtoEN, 'is_stop_requested') else None
                            else:
                                raise ImportError("TransateKRtoEN module missing required functions")
                                
                        elif module_name == 'extract_glossary_from_epub':
                            # Validate the module before importing critical functions  
                            import extract_glossary_from_epub
                            if hasattr(extract_glossary_from_epub, 'main') and hasattr(extract_glossary_from_epub, 'set_stop_flag'):
                                glossary_main = extract_glossary_from_epub.main
                                glossary_stop_flag = extract_glossary_from_epub.set_stop_flag
                                glossary_stop_check = extract_glossary_from_epub.is_stop_requested if hasattr(extract_glossary_from_epub, 'is_stop_requested') else None
                            else:
                                raise ImportError("extract_glossary_from_epub module missing required functions")
                                
                        elif module_name == 'epub_converter':
                            # Validate the module before importing
                            import epub_converter
                            if hasattr(epub_converter, 'fallback_compile_epub'):
                                fallback_compile_epub = epub_converter.fallback_compile_epub
                            else:
                                raise ImportError("epub_converter module missing fallback_compile_epub function")
                                
                        elif module_name == 'scan_html_folder':
                            # Validate the module before importing
                            import scan_html_folder as scan_module
                            if hasattr(scan_module, 'scan_html_folder'):
                                scan_html_folder = scan_module.scan_html_folder
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
            
            # Store references to imported modules in instance variables for later use
            self._translation_main = translation_main
            self._translation_stop_flag = translation_stop_flag
            self._translation_stop_check = translation_stop_check
            self._glossary_main = glossary_main
            self._glossary_stop_flag = glossary_stop_flag
            self._glossary_stop_check = glossary_stop_check
            self._fallback_compile_epub = fallback_compile_epub
            self._scan_html_folder = scan_html_folder
            
            # Final module state update with enhanced error checking
            self._modules_loaded = True
            self._modules_loading = False
            
            # Enhanced module availability checking with better integration
            if hasattr(self, 'master'):
                QTimer.singleShot(0, self._check_modules)
            
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
        
        # Use the stored instance variables instead of globals
        button_checks = [
            (self._translation_main if hasattr(self, '_translation_main') else None, 'button_run', "Translation"),
            (self._glossary_main if hasattr(self, '_glossary_main') else None, 'glossary_button', "Glossary extraction"),
            (self._fallback_compile_epub if hasattr(self, '_fallback_compile_epub') else None, 'epub_button', "EPUB converter"),
            (self._scan_html_folder if hasattr(self, '_scan_html_folder') else None, 'qa_button', "QA scanner")
        ]
        
        for module, button_attr, name in button_checks:
            if module is None and hasattr(self, button_attr):
                button = getattr(self, button_attr, None)
                if button:
                    button.config(state='disabled')
                    self.append_log(f"⚠️ {name} module not available")


    def detect_novel_numbering_unified(self, output_dir, progress_data):
        """
        Use the backend's detect_novel_numbering function for consistent detection
        """
        try:
            # Try to load the backend detection function
            if not self._lazy_load_modules():
                # Fallback to current GUI logic if modules not loaded
                return self._detect_novel_numbering_gui_fallback(output_dir, progress_data)
            
            # Import the detection function from backend
            from TransateKRtoEN import detect_novel_numbering
            
            # Build a chapters list from progress data to pass to backend function
            chapters = []
            for chapter_key, chapter_info in progress_data.get("chapters", {}).items():
                # Get the output file, handling None values
                output_file = chapter_info.get('output_file', '')
                
                chapter_dict = {
                    'original_basename': chapter_info.get('original_basename', ''),
                    'filename': output_file or '',  # Ensure it's never None
                    'num': chapter_info.get('chapter_num', 0)
                }
                
                # Only add the output file path if it exists and is not empty
                if output_file and output_file.strip():
                    chapter_dict['filename'] = os.path.join(output_dir, output_file)
                else:
                    # If no output file, try to discover a file based on original basename or chapter number
                    retain = os.getenv('RETAIN_SOURCE_EXTENSION', '0') == '1' or self.config.get('retain_source_extension', False)
                    allowed_exts = ('.html', '.xhtml', '.htm')
                    discovered = None
                    
                    if chapter_dict['original_basename']:
                        base = chapter_dict['original_basename']
                        # Scan output_dir for either response_{base}.* or {base}.*
                        try:
                            for f in os.listdir(output_dir):
                                f_low = f.lower()
                                if f_low.endswith(allowed_exts):
                                    name_no_ext = os.path.splitext(f)[0]
                                    if name_no_ext.startswith('response_'):
                                        candidate_base = name_no_ext[9:]
                                    else:
                                        candidate_base = name_no_ext
                                    if candidate_base == base:
                                        discovered = f
                                        break
                        except Exception:
                            pass
                        
                        if not discovered:
                            # Fall back to expected naming per mode
                            if retain:
                                # Default to original basename with .html
                                discovered = f"{base}.html"
                            else:
                                discovered = f"response_{base}.html"
                    else:
                        # Last resort: use chapter number pattern
                        chapter_num = chapter_info.get('actual_num', chapter_info.get('chapter_num', 0))
                        num_str = f"{int(chapter_num):04d}" if isinstance(chapter_num, (int, float)) else str(chapter_num)
                        try:
                            for f in os.listdir(output_dir):
                                f_low = f.lower()
                                if f_low.endswith(allowed_exts):
                                    name_no_ext = os.path.splitext(f)[0]
                                    # Remove optional response_ prefix
                                    core = name_no_ext[9:] if name_no_ext.startswith('response_') else name_no_ext
                                    if core.startswith(num_str):
                                        discovered = f
                                        break
                        except Exception:
                            pass
                        
                        if not discovered:
                            if retain:
                                discovered = f"{num_str}.html"
                            else:
                                discovered = f"response_{num_str}.html"
                    
                    chapter_dict['filename'] = os.path.join(output_dir, discovered)
                
                chapters.append(chapter_dict)
            
            # Use the backend detection logic
            uses_zero_based = detect_novel_numbering(chapters)
            
            print(f"[GUI] Unified detection result: {'0-based' if uses_zero_based else '1-based'}")
            return uses_zero_based
            
        except Exception as e:
            print(f"[GUI] Error in unified detection: {e}")
            # Fallback to GUI logic on error
            return self._detect_novel_numbering_gui_fallback(output_dir, progress_data)

    def _detect_novel_numbering_gui_fallback(self, output_dir, progress_data):
        """
        Fallback detection logic (current GUI implementation)
        """
        uses_zero_based = False
        
        for chapter_key, chapter_info in progress_data.get("chapters", {}).items():
            if chapter_info.get("status") == "completed":
                output_file = chapter_info.get("output_file", "")
                stored_chapter_num = chapter_info.get("chapter_num", 0)
                if output_file:
                    # Allow filenames with or without 'response_' prefix
                    match = re.search(r'(?:^response_)?(\d+)', output_file)
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
        
        return uses_zero_based
      
    def _make_bottom_toolbar(self):
        """Create the bottom toolbar with all action buttons"""
        from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSizePolicy
        from PySide6.QtCore import Qt
        
        btn_frame = QWidget()
        btn_frame.setMinimumHeight(50)  # Increased for taller buttons
        btn_frame.setMaximumHeight(60)  # Increased for taller buttons
        # Ensure toolbar never overlaps with log by using MinimumExpanding policy
        btn_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setContentsMargins(0, 5, 0, 5)
        btn_layout.setSpacing(2)
        
        # QA Scan button with mini icon
        from PySide6.QtGui import QPixmap, QIcon
        self.qa_button = QPushButton()
        self.qa_button.clicked.connect(self.run_qa_scan)
        self.qa_button.setMinimumWidth(120)  # Wider button
        self.qa_button.setMinimumHeight(40)  # Increased button height
        self.qa_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Expand horizontally to fill space
        
        # Create horizontal layout for button content
        qa_btn_widget = QWidget()
        qa_btn_layout = QHBoxLayout(qa_btn_widget)
        qa_btn_layout.setContentsMargins(0, 0, 0, 0)
        qa_btn_layout.setSpacing(3)  # Minimal spacing between icon and text
        qa_btn_layout.setAlignment(Qt.AlignCenter)  # Center the contents
        
        # Mini icon for QA button
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Halgakos.ico")
        
        # Reuse the RotatableLabel class (already defined in _create_prompt_section)
        class RotatableLabel(QLabel):
            def __init__(self, parent=None):
                super().__init__(parent)
                self._rotation = 0
                self._original_pixmap = None
            
            def set_rotation(self, angle):
                self._rotation = angle
                if self._original_pixmap:
                    transform = QTransform()
                    transform.rotate(angle)
                    rotated = self._original_pixmap.transformed(transform, Qt.SmoothTransformation)
                    self.setPixmap(rotated)
            
            def get_rotation(self):
                return self._rotation
            
            rotation = Property(float, get_rotation, set_rotation)
            
            def set_original_pixmap(self, pixmap):
                self._original_pixmap = pixmap
                self.setPixmap(pixmap)
        
        self.qa_button_icon = RotatableLabel()
        self.qa_button_icon.setStyleSheet("background-color: transparent;")
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            available_sizes = icon.availableSizes()
            if available_sizes:
                largest_size = max(available_sizes, key=lambda s: s.width() * s.height())
                pixmap = icon.pixmap(largest_size)
            else:
                pixmap = QPixmap(icon_path)
            scaled_pixmap = pixmap.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.qa_button_icon.set_original_pixmap(scaled_pixmap)
        self.qa_button_icon.setFixedSize(36, 36)  # Larger container to prevent clipping during rotation
        self.qa_button_icon.setAlignment(Qt.AlignCenter)
        
        # Create animations for QA button icon
        self.qa_icon_spin_animation = QPropertyAnimation(self.qa_button_icon, b"rotation")
        self.qa_icon_spin_animation.setDuration(900)
        self.qa_icon_spin_animation.setStartValue(0)
        self.qa_icon_spin_animation.setEndValue(360)
        self.qa_icon_spin_animation.setLoopCount(-1)
        self.qa_icon_spin_animation.setEasingCurve(QEasingCurve.Linear)
        
        self.qa_icon_stop_animation = QPropertyAnimation(self.qa_button_icon, b"rotation")
        self.qa_icon_stop_animation.setDuration(800)
        self.qa_icon_stop_animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # Button text label
        self.qa_text_label = QLabel("QA Scan")  # Store as instance variable
        self.qa_text_label.setStyleSheet("color: white; font-weight: bold; background-color: transparent;")
        self.qa_text_label.setAlignment(Qt.AlignCenter)
        
        qa_btn_layout.addWidget(self.qa_button_icon)
        qa_btn_layout.addWidget(self.qa_text_label)
        self.qa_button.setLayout(qa_btn_layout)
        
        self.qa_button.setStyleSheet("""
            QPushButton {
                background-color: #e67e22;
                color: white;
                padding: 6px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
        """)
        
        # Define toolbar items with button styles - modern color scheme
        style_colors = {
            "info": "#3498db",      # Modern blue
            "warning": "#e67e22",   # Modern orange (replacing yellow)
            "secondary": "#95a5a6", # Modern gray
            "primary": "#9b59b6",   # Modern purple
            "success": "#27ae60",   # Modern green
            "glossary": "#f39c12"   # Yellow/gold for glossary
        }
        
        toolbar_items = [
            ("EPUB Converter", self.epub_converter, "info"),
            ("Extract Glossary", self.run_glossary_extraction_thread, "warning"),
            ("Glossary Manager", self.glossary_manager, "glossary"),
        ]
        
        # Add Manga Translator if available
        if MANGA_SUPPORT:
            toolbar_items.append(("Manga Translator", self.open_manga_translator, "primary"))
         
        # Async Processing 
        toolbar_items.append(("Async Translation", self.open_async_processing, "success"))
        
        toolbar_items.extend([
            ("Retranslate", self.force_retranslation, "warning"),
            ("Save Config", self.save_config, "secondary"),
            ("Load Glossary", self.load_glossary, "glossary"),
            ("Import Profiles", self.import_profiles, "secondary"),
            ("Export Profiles", self.export_profiles, "secondary"),
        ])
        
        # Create buttons
        for idx, (lbl, cmd, style) in enumerate(toolbar_items):
            # Special handling for Extract Glossary - don't set text yet, we'll add it with icon
            if lbl == "Extract Glossary":
                btn = QPushButton()
            else:
                btn = QPushButton(lbl)
            
            # Special-case Save Config for inline feedback
            if lbl == "Save Config":
                self.save_config_button = btn
                btn.clicked.connect(self._on_save_config_clicked)
                btn.setToolTip("Save all settings to config.json")
            elif lbl != "Extract Glossary":  # Don't connect yet for Extract Glossary
                btn.clicked.connect(cmd)
            
            btn.setMinimumHeight(40)  # Increased button height for all toolbar buttons
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Expand horizontally to fill space
            
            # Set maximum width for Import/Export Profile buttons to make them narrower
            if lbl in ["Import Profiles", "Export Profiles"]:
                btn.setMaximumWidth(130)  # Constrain width for these buttons
            
            color = style_colors.get(style, "#95a5a6")
            btn.setStyleSheet(f"background-color: {color}; color: white; padding: 6px; font-weight: bold;")
            btn_layout.addWidget(btn)
            
            if lbl == "Extract Glossary":
                # Create Extract Glossary button with mini icon
                glossary_btn_widget = QWidget()
                glossary_btn_layout = QHBoxLayout(glossary_btn_widget)
                glossary_btn_layout.setContentsMargins(0, 0, 0, 0)
                glossary_btn_layout.setSpacing(3)  # Minimal spacing between icon and text
                glossary_btn_layout.setAlignment(Qt.AlignCenter)  # Center the contents
                
                # Mini icon for Glossary button
                icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Halgakos.ico")
                
                # Reuse RotatableLabel class
                class RotatableLabel(QLabel):
                    def __init__(self, parent=None):
                        super().__init__(parent)
                        self._rotation = 0
                        self._original_pixmap = None
                    
                    def set_rotation(self, angle):
                        self._rotation = angle
                        if self._original_pixmap:
                            transform = QTransform()
                            transform.rotate(angle)
                            rotated = self._original_pixmap.transformed(transform, Qt.SmoothTransformation)
                            self.setPixmap(rotated)
                    
                    def get_rotation(self):
                        return self._rotation
                    
                    rotation = Property(float, get_rotation, set_rotation)
                    
                    def set_original_pixmap(self, pixmap):
                        self._original_pixmap = pixmap
                        self.setPixmap(pixmap)
                
                self.glossary_button_icon = RotatableLabel()
                self.glossary_button_icon.setStyleSheet("background-color: transparent;")
                if os.path.exists(icon_path):
                    from PySide6.QtGui import QIcon, QPixmap
                    icon = QIcon(icon_path)
                    available_sizes = icon.availableSizes()
                    if available_sizes:
                        largest_size = max(available_sizes, key=lambda s: s.width() * s.height())
                        pixmap = icon.pixmap(largest_size)
                    else:
                        pixmap = QPixmap(icon_path)
                    scaled_pixmap = pixmap.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.glossary_button_icon.set_original_pixmap(scaled_pixmap)
                self.glossary_button_icon.setFixedSize(36, 36)  # Larger container to prevent clipping during rotation
                self.glossary_button_icon.setAlignment(Qt.AlignCenter)
                
                # Create animations for Glossary button icon
                self.glossary_icon_spin_animation = QPropertyAnimation(self.glossary_button_icon, b"rotation")
                self.glossary_icon_spin_animation.setDuration(900)
                self.glossary_icon_spin_animation.setStartValue(0)
                self.glossary_icon_spin_animation.setEndValue(360)
                self.glossary_icon_spin_animation.setLoopCount(-1)
                self.glossary_icon_spin_animation.setEasingCurve(QEasingCurve.Linear)
                
                self.glossary_icon_stop_animation = QPropertyAnimation(self.glossary_button_icon, b"rotation")
                self.glossary_icon_stop_animation.setDuration(800)
                self.glossary_icon_stop_animation.setEasingCurve(QEasingCurve.OutCubic)
                
                # Button text label
                self.glossary_text_label = QLabel("Extract Glossary")  # Store as instance variable
                self.glossary_text_label.setStyleSheet("color: white; font-weight: bold; background-color: transparent;")
                self.glossary_text_label.setAlignment(Qt.AlignCenter)
                
                glossary_btn_layout.addWidget(self.glossary_button_icon)
                glossary_btn_layout.addWidget(self.glossary_text_label)
                btn.setLayout(glossary_btn_layout)
                
                # Now connect the command after layout is set
                btn.clicked.connect(cmd)
                
                self.glossary_button = btn
                # Add disabled state styling for Extract Glossary button
                btn.setMinimumWidth(150)  # Wider button
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {color};
                        color: white;
                        padding: 6px;
                        font-weight: bold;
                    }}
                    QPushButton:disabled {{
                        background-color: #555555;
                        color: #888888;
                    }}
                """)
            elif lbl == "EPUB Converter":
                self.epub_button = btn
            elif lbl == "Async Processing (50% Off)":
                self.async_button = btn
        
        # Add QA button at the end
        btn_layout.addWidget(self.qa_button)
        
        # Add toolbar to main layout
        # Note: This will need to be integrated into the main GUI layout in _setup_gui
        return btn_frame

 
    def _on_save_config_clicked(self):
        """Provide inline feedback when saving config from the toolbar button."""
        try:
            from PySide6.QtWidgets import QApplication
            from PySide6.QtCore import QTimer
            btn = getattr(self, 'save_config_button', None)
            original_text = None
            original_style = None
            if btn is not None:
                original_text = btn.text()
                original_style = btn.styleSheet()
                btn.setEnabled(False)
                btn.setText("Saving…")
                # Subtle highlight during save
                btn.setStyleSheet("background-color: #17a2b8; color: white; font-weight: bold;")
                QApplication.processEvents()
            # Log feedback
            try:
                self.append_log("💾 Saving configuration…")
            except Exception:
                pass
            # Perform save (keep message box behavior as-is)
            self.save_config(show_message=True)
            if btn is not None:
                btn.setText("Saved ✓")
                btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
                # Restore after a short delay
                def _restore():
                    try:
                        btn.setText(original_text or "Save Config")
                        if original_style:
                            btn.setStyleSheet(original_style)
                        btn.setEnabled(True)
                    except Exception:
                        pass
                QTimer.singleShot(900, _restore)
            try:
                self.append_log("✅ Configuration saved")
            except Exception:
                pass
        except Exception as e:
            # Best-effort restore on failure
            try:
                if getattr(self, 'save_config_button', None):
                    self.save_config_button.setText("Save Failed")
                    self.save_config_button.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold;")
                    def _restore_fail():
                        try:
                            self.save_config_button.setText("Save Config")
                            # Do not assume original style on failure
                            self.save_config_button.setEnabled(True)
                        except Exception:
                            pass
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(1500, _restore_fail)
            except Exception:
                pass
            # Surface error minimally in log
            try:
                self.append_log(f"❌ Save failed: {e}")
            except Exception:
                pass
    
    def keyPressEvent(self, event):
        """Handle key press events for shortcuts"""
        if event.key() == Qt.Key_F11:
            self.toggle_fullscreen()
        else:
            super().keyPressEvent(event)
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode and expand log area"""
        if not self.is_fullscreen:
            # Enter fullscreen
            self.showFullScreen()
            self.is_fullscreen = True
            
            # Make log area expand more in fullscreen
            # Set all rows to zero stretch except log (row 10)
            for r in range(12):
                if r == 10:
                    self.frame.setRowStretch(r, 5)  # Log gets all the stretch
                else:
                    self.frame.setRowStretch(r, 0)  # All other rows including toolbar (11) don't stretch
            
            self.append_log("🖥️ Fullscreen mode enabled (Press F11 to exit)")
        else:
            # Exit fullscreen
            self.showNormal()
            self.is_fullscreen = False
            
            # Restore default stretches for all rows
            for r in range(12):
                self.frame.setRowStretch(r, self._default_row_stretches.get(r, 0))
            
            self.append_log("🖥️ Fullscreen mode disabled")
    
    def _get_opf_file_order(self, file_list):
        """
        Sort files based on OPF spine order if available.
        Uses STRICT OPF ordering - includes ALL files from spine without filtering.
        This ensures notice files, copyright pages, etc. are processed in the correct order.
        Returns sorted file list based on OPF, or original list if no OPF found.
        """
        try:
            import xml.etree.ElementTree as ET
            import zipfile
            import re
            
            # First, check if we have content.opf in the current directory
            opf_file = None
            if file_list:
                current_dir = os.path.dirname(file_list[0]) if file_list else os.getcwd()
                possible_opf = os.path.join(current_dir, 'content.opf')
                if os.path.exists(possible_opf):
                    opf_file = possible_opf
                    self.append_log(f"📋 Found content.opf in directory")
            
            # If no OPF, check if any of the files is an OPF
            if not opf_file:
                for file_path in file_list:
                    if file_path.lower().endswith('.opf'):
                        opf_file = file_path
                        self.append_log(f"📋 Found OPF file: {os.path.basename(opf_file)}")
                        break
            
            # If no OPF, try to extract from EPUB
            if not opf_file:
                epub_files = [f for f in file_list if f.lower().endswith('.epub')]
                if epub_files:
                    epub_path = epub_files[0]
                    try:
                        with zipfile.ZipFile(epub_path, 'r') as zf:
                            for name in zf.namelist():
                                if name.endswith('.opf'):
                                    opf_content = zf.read(name)
                                    temp_opf = os.path.join(os.path.dirname(epub_path), 'temp_content.opf')
                                    with open(temp_opf, 'wb') as f:
                                        f.write(opf_content)
                                    opf_file = temp_opf
                                    self.append_log(f"📋 Extracted OPF from EPUB: {os.path.basename(epub_path)}")
                                    break
                    except Exception as e:
                        self.append_log(f"⚠️ Could not extract OPF from EPUB: {e}")
            
            if not opf_file:
                self.append_log(f"ℹ️ No OPF file found, using default file order")
                return file_list
            
            # Parse the OPF file
            try:
                tree = ET.parse(opf_file)
                root = tree.getroot()
                
                # Handle namespaces
                ns = {'opf': 'http://www.idpf.org/2007/opf'}
                if root.tag.startswith('{'):
                    default_ns = root.tag[1:root.tag.index('}')]
                    ns = {'opf': default_ns}
                
                # Get manifest to map IDs to files
                manifest = {}
                for item in root.findall('.//opf:manifest/opf:item', ns):
                    item_id = item.get('id')
                    href = item.get('href')
                    
                    if item_id and href:
                        filename = os.path.basename(href)
                        manifest[item_id] = filename
                        # Store multiple variations for matching
                        name_without_ext = os.path.splitext(filename)[0]
                        manifest[item_id + '_noext'] = name_without_ext
                        # Also store with response_ prefix for matching
                        manifest[item_id + '_response'] = f"response_{filename}"
                        manifest[item_id + '_response_noext'] = f"response_{name_without_ext}"
                
                # Get spine order - include ALL files first for correct indexing
                spine_order_full = []
                spine = root.find('.//opf:spine', ns)
                if spine is not None:
                    for itemref in spine.findall('opf:itemref', ns):
                        idref = itemref.get('idref')
                        if idref and idref in manifest:
                            spine_order_full.append(manifest[idref])
                
                # Now filter out cover and nav/toc files for processing (unless override is enabled)
                translate_special = os.environ.get('TRANSLATE_SPECIAL_FILES', '0') == '1'
                # Backward compatibility
                translate_special = translate_special or (os.environ.get('TRANSLATE_COVER_HTML', '0') == '1')
                
                spine_order = []
                for item in spine_order_full:
                    # Skip navigation and cover files unless override is enabled
                    if translate_special or not any(skip in item.lower() for skip in ['nav.', 'toc.', 'cover.']):
                        spine_order.append(item)
                
                self.append_log(f"📋 Found {len(spine_order_full)} items in OPF spine ({len(spine_order)} after filtering)")
                
                # Count file types
                notice_count = sum(1 for f in spine_order if 'notice' in f.lower())
                chapter_count = sum(1 for f in spine_order if 'chapter' in f.lower() and 'notice' not in f.lower())
                skipped_count = len(spine_order_full) - len(spine_order)
                
                if skipped_count > 0:
                    self.append_log(f"   • Skipped files (cover/nav/toc): {skipped_count}")
                if notice_count > 0:
                    self.append_log(f"   • Notice/Copyright files: {notice_count}")
                if chapter_count > 0:
                    self.append_log(f"   • Chapter files: {chapter_count}")
                
                # Show first few spine entries
                if spine_order:
                    self.append_log(f"   📖 Spine order preview:")
                    for i, entry in enumerate(spine_order[:5]):
                        self.append_log(f"      [{i}]: {entry}")
                    if len(spine_order) > 5:
                        self.append_log(f"      ... and {len(spine_order) - 5} more")
                
                # Map input files to spine positions
                ordered_files = []
                unordered_files = []
                
                for file_path in file_list:
                    basename = os.path.basename(file_path)
                    basename_noext = os.path.splitext(basename)[0]
                    
                    # Try to find this file in the spine
                    found_position = None
                    matched_spine_file = None
                    
                    # Direct exact match
                    if basename in spine_order:
                        found_position = spine_order.index(basename)
                        matched_spine_file = basename
                    # Match without extension
                    elif basename_noext in spine_order:
                        found_position = spine_order.index(basename_noext)
                        matched_spine_file = basename_noext
                    else:
                        # Try pattern matching for response_ files
                        for idx, spine_item in enumerate(spine_order):
                            spine_noext = os.path.splitext(spine_item)[0]
                            
                            # Check if this is a response_ file matching spine item
                            if basename.startswith('response_'):
                                # Remove response_ prefix and try to match
                                clean_name = basename[9:]  # Remove 'response_'
                                clean_noext = os.path.splitext(clean_name)[0]
                                
                                if clean_name == spine_item or clean_noext == spine_noext:
                                    found_position = idx
                                    matched_spine_file = spine_item
                                    break
                                
                                # Try matching by chapter number
                                spine_num = re.search(r'(\d+)', spine_item)
                                file_num = re.search(r'(\d+)', clean_name)
                                if spine_num and file_num and spine_num.group(1) == file_num.group(1):
                                    # Check if both are notice or both are chapter files
                                    both_notice = 'notice' in spine_item.lower() and 'notice' in clean_name.lower()
                                    both_chapter = 'chapter' in spine_item.lower() and 'chapter' in clean_name.lower()
                                    if both_notice or both_chapter:
                                        found_position = idx
                                        matched_spine_file = spine_item
                                        break
                            else:
                                # For non-response files, check if spine item is contained
                                if spine_noext in basename_noext:
                                    found_position = idx
                                    matched_spine_file = spine_item
                                    break
                                
                                # Number-based matching
                                spine_num = re.search(r'(\d+)', spine_item)
                                file_num = re.search(r'(\d+)', basename)
                                if spine_num and file_num and spine_num.group(1) == file_num.group(1):
                                    # Check file type match
                                    both_notice = 'notice' in spine_item.lower() and 'notice' in basename.lower()
                                    both_chapter = 'chapter' in spine_item.lower() and 'chapter' in basename.lower()
                                    if both_notice or both_chapter:
                                        found_position = idx
                                        matched_spine_file = spine_item
                                        break
                    
                    if found_position is not None:
                        ordered_files.append((found_position, file_path))
                        self.append_log(f"  ✓ Matched: {basename} → spine[{found_position}]: {matched_spine_file}")
                    else:
                        unordered_files.append(file_path)
                        self.append_log(f"  ⚠️ Not in spine: {basename}")
                
                # Sort by spine position
                ordered_files.sort(key=lambda x: x[0])
                final_order = [f for _, f in ordered_files]
                
                # Add unmapped files at the end
                if unordered_files:
                    self.append_log(f"📋 Adding {len(unordered_files)} unmapped files at the end")
                    final_order.extend(sorted(unordered_files))
                
                # Clean up temp OPF if created
                if opf_file and 'temp_content.opf' in opf_file and os.path.exists(opf_file):
                    try:
                        os.remove(opf_file)
                    except:
                        pass
                
                self.append_log(f"✅ Files sorted using STRICT OPF spine order")
                self.append_log(f"   • Total files: {len(final_order)}")
                self.append_log(f"   • Following exact spine sequence from OPF")
                
                return final_order if final_order else file_list
                
            except Exception as e:
                self.append_log(f"⚠️ Error parsing OPF file: {e}")
                if opf_file and 'temp_content.opf' in opf_file and os.path.exists(opf_file):
                    try:
                        os.remove(opf_file)
                    except:
                        pass
                return file_list
                
        except Exception as e:
            self.append_log(f"⚠️ Error in OPF sorting: {e}")
            return file_list
 
    def run_translation_thread(self):
        """Start translation in a background worker (ThreadPoolExecutor)"""
        # Prevent overlap with glossary extraction
        if (hasattr(self, 'glossary_thread') and self.glossary_thread and self.glossary_thread.is_alive()) or \
           (hasattr(self, 'glossary_future') and self.glossary_future and not self.glossary_future.done()):
            self.append_log("⚠️ Cannot run translation while glossary extraction is in progress.")
            QMessageBox.warning(self, "Process Running", "Please wait for glossary extraction to complete before starting translation.")
            return
        
        if self.translation_thread and self.translation_thread.is_alive():
            self.stop_translation()
            return
        
        # Check if files are selected
        if not hasattr(self, 'selected_files') or not self.selected_files:
            file_path = self.entry_epub.text().strip()
            if not file_path or file_path.startswith("No file selected") or "files selected" in file_path:
                QMessageBox.critical(self, "Error", "Please select file(s) to translate.")
                return
            self.selected_files = [file_path]
        
        # Reset stop flags
        self.stop_requested = False
        if translation_stop_flag:
            translation_stop_flag(False)
        
        # Also reset the module's internal stop flag
        try:
            if hasattr(self, '_main_module') and self._main_module:
                if hasattr(self._main_module, 'set_stop_flag'):
                    self._main_module.set_stop_flag(False)
        except:
            pass
        
        # Update button immediately to show translation is starting
        if hasattr(self, 'button_run'):
            self.button_run.config(text="⏹ Stop", state="normal")
        
        # Delay auto-scroll so first log is readable (set to 0 for immediate scrolling)
        self._start_autoscroll_delay(0)
        # Show immediate feedback that translation is starting
        self.append_log("🚀 Initializing translation process...")
        # Force immediate scroll to bottom so user sees the latest output right away
        try:
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        except Exception:
            pass
        # Reset stop notice dedupe flag at start of a run
        self._stop_notice_shown = False
        
        # SIMPLIFIED: Skip the wrapper and call run_translation_direct in a thread
        def simple_thread_target():
            try:
                self.append_log("🟢 Thread started successfully!")
                
                # Load modules if needed
                if not self._modules_loaded:
                    self.append_log("📬 Loading translation modules...")
                    if not self._lazy_load_modules():
                        self.append_log("❌ Failed to load modules")
                        return
                    self.append_log("✅ Modules loaded")
                
                # Check for large EPUBs and set optimization parameters
                epub_files = [f for f in self.selected_files if f.lower().endswith('.epub')]
                
                for epub_path in epub_files:
                    try:
                        import zipfile
                        with zipfile.ZipFile(epub_path, 'r') as zf:
                            # Quick count without reading content
                            html_files = [f for f in zf.namelist() if f.lower().endswith(('.html', '.xhtml', '.htm'))]
                            file_count = len(html_files)
                            
                            if file_count > 50:
                                self.append_log(f"📚 Large EPUB detected: {file_count} chapters")
                                
                                # Get user-configured worker count
                                if hasattr(self, 'config') and 'extraction_workers' in self.config:
                                    max_workers = self.config.get('extraction_workers', 2)
                                else:
                                    # Fallback to environment variable or default
                                    max_workers = int(os.environ.get('EXTRACTION_WORKERS', '2'))
                                
                                # Set extraction parameters
                                os.environ['EXTRACTION_WORKERS'] = str(max_workers)
                                os.environ['EXTRACTION_PROGRESS_CALLBACK'] = 'enabled'
                                
                                # Set progress interval based on file count
                                if file_count > 500:
                                    progress_interval = 50
                                    os.environ['EXTRACTION_BATCH_SIZE'] = '100'
                                    self.append_log(f"⚡ Using {max_workers} workers with batch size 100")
                                elif file_count > 200:
                                    progress_interval = 25
                                    os.environ['EXTRACTION_BATCH_SIZE'] = '50'
                                    self.append_log(f"⚡ Using {max_workers} workers with batch size 50")
                                elif file_count > 100:
                                    progress_interval = 20
                                    os.environ['EXTRACTION_BATCH_SIZE'] = '25'
                                    self.append_log(f"⚡ Using {max_workers} workers with batch size 25")
                                else:
                                    progress_interval = 10
                                    os.environ['EXTRACTION_BATCH_SIZE'] = '20'
                                    self.append_log(f"⚡ Using {max_workers} workers with batch size 20")
                                
                                os.environ['EXTRACTION_PROGRESS_INTERVAL'] = str(progress_interval)
                                
                                # Enable performance flags for large files
                                os.environ['FAST_EXTRACTION'] = '1'
                                os.environ['PARALLEL_PARSE'] = '1'
                                
                    except Exception as e:
                        # If we can't check, just continue
                        pass
                
                # Set essential environment variables from current config before translation
                os.environ['BATCH_TRANSLATE_HEADERS'] = '1' if self.config.get('batch_translate_headers', False) else '0'
                os.environ['IGNORE_HEADER'] = '1' if self.config.get('ignore_header', False) else '0'
                os.environ['IGNORE_TITLE'] = '1' if self.config.get('ignore_title', False) else '0'
                os.environ['USE_SORTED_FALLBACK'] = '1' if self.config.get('use_sorted_fallback', False) else '0'
                
                # Call the direct function
                self.append_log("🚀 Starting translation...")
                translation_completed = self.run_translation_direct()
                
                # Post-translation scanning phase
                # If scanning phase toggle is enabled, launch scanner after translation
                # BUT only if translation completed successfully (not stopped by user)
                try:
                    if (hasattr(self, 'scan_phase_enabled_var') and self.scan_phase_enabled_var and 
                        translation_completed and not self.stop_requested):
                        mode = self.scan_phase_mode_var if hasattr(self, 'scan_phase_mode_var') else 'quick-scan'
                        self.append_log(f"🧪 Scanning phase enabled — launching QA Scanner in {mode} mode...")
                        # Emit signal to trigger QA scan on main thread
                        self.trigger_qa_scan_signal.emit()
                except Exception as e:
                    self.append_log(f"⚠️ Could not launch post-translation scan: {e}")
                    import traceback
                    self.append_log(traceback.format_exc())
                
            except Exception as e:
                self.append_log(f"❌ Error in thread: {e}")
                import traceback
                self.append_log(traceback.format_exc())
            finally:
                # Clean up environment variables
                env_vars = [
                    'EXTRACTION_WORKERS', 'EXTRACTION_BATCH_SIZE',
                    'EXTRACTION_PROGRESS_CALLBACK', 'EXTRACTION_PROGRESS_INTERVAL',
                    'FAST_EXTRACTION', 'PARALLEL_PARSE'
                ]
                for var in env_vars:
                    if var in os.environ:
                        del os.environ[var]
                
                self.translation_thread = None
                # Emit signal to update button (thread-safe)
                self.thread_complete_signal.emit()
        
        thread_name = f"TranslationThread_{int(time.time())}"
        self.translation_thread = threading.Thread(
            target=simple_thread_target,
            name=thread_name,
            daemon=True
        )
        self.translation_thread.start()
        
        # Update button IMMEDIATELY after starting thread (synchronous)
        self.update_run_button()
        
        self.append_log("🟡 Thread started...")

    def run_translation_direct(self):
        """Run translation directly - handles multiple files and different file types"""
        try:
            # Re-attach GUI logging handlers to reclaim logs from standalone header translation
            try:
                self._attach_gui_logging_handlers()
            except Exception:
                pass
            
            # Restore print hijack if it was captured by manga translator
            # This ensures main GUI logs go to main GUI, not manga GUI
            try:
                import builtins
                # Check if print was hijacked by manga translator
                if hasattr(builtins, '_manga_log_callbacks') and builtins._manga_log_callbacks:
                    # Restore original print for main GUI
                    if hasattr(builtins, 'print') and hasattr(builtins.print, '__name__'):
                        if builtins.print.__name__ == 'manga_print':
                            # Print is hijacked, restore it
                            from manga_translator import MangaTranslator
                            if hasattr(MangaTranslator, '_original_print_backup'):
                                builtins.print = MangaTranslator._original_print_backup
                                # Also restore in unified_api_client
                                try:
                                    import sys
                                    import unified_api_client
                                    uc_module = sys.modules.get('unified_api_client')
                                    if uc_module:
                                        uc_module.__dict__['print'] = MangaTranslator._original_print_backup
                                except Exception:
                                    pass
            except Exception:
                pass
            
            # Check stop at the very beginning
            if self.stop_requested:
                return False
            
            # DON'T CALL _lazy_load_modules HERE!
            # Modules are already loaded in the wrapper
            # Just verify they're loaded
            if not self._modules_loaded:
                self.append_log("❌ Translation modules not loaded")
                return False

            # Check stop after verification
            if self.stop_requested:
                return False

            # SET GLOSSARY IN ENVIRONMENT
            if hasattr(self, 'manual_glossary_path') and self.manual_glossary_path:
                os.environ['MANUAL_GLOSSARY'] = self.manual_glossary_path
                self.append_log(f"📑 Set glossary in environment: {os.path.basename(self.manual_glossary_path)}")
            else:
                # Clear any previous glossary from environment
                if 'MANUAL_GLOSSARY' in os.environ:
                    del os.environ['MANUAL_GLOSSARY']
                self.append_log(f"ℹ️ No glossary loaded")

            # ========== NEW: APPLY OPF-BASED SORTING ==========
            # Sort files based on OPF order if available
            original_file_count = len(self.selected_files)
            self.selected_files = self._get_opf_file_order(self.selected_files)
            self.append_log(f"📚 Processing {original_file_count} files in reading order")
            # ====================================================

            # Process each file
            total_files = len(self.selected_files)
            successful = 0
            failed = 0
            
            # Check if we're processing multiple images - if so, create a combined output folder
            image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
            image_files = [f for f in self.selected_files if os.path.splitext(f)[1].lower() in image_extensions]
            
            combined_image_output_dir = None
            if len(image_files) > 1:
                # Check stop before creating directories
                if self.stop_requested:
                    return False
                    
                # Get the common parent directory name or use timestamp
                parent_dir = os.path.dirname(self.selected_files[0])
                folder_name = os.path.basename(parent_dir) if parent_dir else f"translated_images_{int(time.time())}"
                combined_image_output_dir = folder_name
                os.makedirs(combined_image_output_dir, exist_ok=True)
                
                # Create images subdirectory for originals
                images_dir = os.path.join(combined_image_output_dir, "images")
                os.makedirs(images_dir, exist_ok=True)
                
                self.append_log(f"📁 Created combined output directory: {combined_image_output_dir}")
            
            for i, file_path in enumerate(self.selected_files):
                if self.stop_requested:
                    # Suppress per-file stop spam; summary will be shown later
                    break
                
                self.current_file_index = i
                
                # Log progress for multiple files
                if total_files > 1:
                    self.append_log(f"\n{'='*60}")
                    self.append_log(f"📄 Processing file {i+1}/{total_files}: {os.path.basename(file_path)}")
                    progress_percent = ((i + 1) / total_files) * 100
                    self.append_log(f"📊 Overall progress: {progress_percent:.1f}%")
                    self.append_log(f"{'='*60}")
                
                if not os.path.exists(file_path):
                    self.append_log(f"❌ File not found: {file_path}")
                    failed += 1
                    continue
                
                # Determine file type and process accordingly
                ext = os.path.splitext(file_path)[1].lower()
                
                try:
                    if ext in image_extensions:
                        # Process as image with combined output directory if applicable
                        if self._process_image_file(file_path, combined_image_output_dir):
                            successful += 1
                        else:
                            failed += 1
                    elif ext in {'.epub', '.txt'}:
                        # Process as EPUB/TXT
                        if self._process_text_file(file_path):
                            successful += 1
                        else:
                            failed += 1
                    else:
                        self.append_log(f"⚠️ Unsupported file type: {ext}")
                        failed += 1
                        
                except Exception as e:
                    self.append_log(f"❌ Error processing {os.path.basename(file_path)}: {str(e)}")
                    import traceback
                    self.append_log(f"❌ Full error: {traceback.format_exc()}")
                    failed += 1
            
            # Check stop before final summary
            if self.stop_requested:
                self.append_log(f"\n⏹️ Translation stopped - processed {successful} of {total_files} files")
                return False
                
            # Final summary
            if total_files > 1:
                self.append_log(f"\n{'='*60}")
                self.append_log(f"📊 Translation Summary:")
                self.append_log(f"   ✅ Successful: {successful} files")
                if failed > 0:
                    self.append_log(f"   ❌ Failed: {failed} files")
                self.append_log(f"   📁 Total: {total_files} files")
                
                if combined_image_output_dir and successful > 0:
                    self.append_log(f"\n💡 Tip: You can now compile the HTML files in '{combined_image_output_dir}' into an EPUB")
                    
                    # Check for cover image
                    cover_found = False
                    for img_name in ['cover.png', 'cover.jpg', 'cover.jpeg', 'cover.webp']:
                        if os.path.exists(os.path.join(combined_image_output_dir, "images", img_name)):
                            self.append_log(f"   📖 Found cover image: {img_name}")
                            cover_found = True
                            break
                    
                    if not cover_found:
                        # Use first image as cover
                        images_in_dir = os.listdir(os.path.join(combined_image_output_dir, "images"))
                        if images_in_dir:
                            self.append_log(f"   📖 First image will be used as cover: {images_in_dir[0]}")
                
                self.append_log(f"{'='*60}")
            
            # Only return True if at least one file succeeded
            # This prevents QA scanner from running when all files failed
            if successful == 0:
                return False
            
            return True  # Translation completed successfully
            
        except Exception as e:
            self.append_log(f"❌ Translation setup error: {e}")
            import traceback
            self.append_log(f"❌ Full error: {traceback.format_exc()}")
            return False
        
        finally:
            self.stop_requested = False
            if translation_stop_flag:
                translation_stop_flag(False)
                
            # Also reset the module's internal stop flag
            try:
                if hasattr(self, '_main_module') and self._main_module:
                    if hasattr(self._main_module, 'set_stop_flag'):
                        self._main_module.set_stop_flag(False)
            except:
                pass
                
            self.translation_thread = None
            self.current_file_index = 0
            # Emit signal to update button (thread-safe)
            self.thread_complete_signal.emit()

    def _process_image_file(self, image_path, combined_output_dir=None):
        """Process a single image file using the direct image translation API with progress tracking"""
        try:
            import time
            import shutil
            import hashlib
            import os
            import json
            
            # Determine output directory early for progress tracking
            image_name = os.path.basename(image_path)
            base_name = os.path.splitext(image_name)[0]
            
            if combined_output_dir:
                output_dir = combined_output_dir
            else:
                output_dir = base_name
            
            # Initialize progress manager if not already done
            if not hasattr(self, 'image_progress_manager'):
                # Use the determined output directory
                os.makedirs(output_dir, exist_ok=True)
                
                # Import or define a simplified ImageProgressManager
                class ImageProgressManager:
                    def __init__(self, output_dir=None):
                        self.output_dir = output_dir
                        if output_dir:
                            self.PROGRESS_FILE = os.path.join(output_dir, "translation_progress.json")
                            self.prog = self._init_or_load()
                        else:
                            self.PROGRESS_FILE = None
                            self.prog = {"images": {}, "content_hashes": {}, "version": "1.0"}
                    
                    def set_output_dir(self, output_dir):
                        """Set or update the output directory and load progress"""
                        self.output_dir = output_dir
                        self.PROGRESS_FILE = os.path.join(output_dir, "translation_progress.json")
                        self.prog = self._init_or_load()
                    
                    def _init_or_load(self):
                        """Initialize or load progress tracking"""
                        if os.path.exists(self.PROGRESS_FILE):
                            try:
                                with open(self.PROGRESS_FILE, "r", encoding="utf-8") as pf:
                                    return json.load(pf)
                            except Exception as e:
                                if hasattr(self, 'append_log'):
                                    self.append_log(f"⚠️ Creating new progress file due to error: {e}")
                                return {"images": {}, "content_hashes": {}, "version": "1.0"}
                        else:
                            return {"images": {}, "content_hashes": {}, "version": "1.0"}
                    
                    def save(self):
                        """Save progress to file atomically"""
                        if not self.PROGRESS_FILE:
                            return
                        try:
                            # Ensure directory exists
                            os.makedirs(os.path.dirname(self.PROGRESS_FILE), exist_ok=True)
                            
                            temp_file = self.PROGRESS_FILE + '.tmp'
                            with open(temp_file, "w", encoding="utf-8") as pf:
                                json.dump(self.prog, pf, ensure_ascii=False, indent=2)
                            
                            if os.path.exists(self.PROGRESS_FILE):
                                os.remove(self.PROGRESS_FILE)
                            os.rename(temp_file, self.PROGRESS_FILE)
                        except Exception as e:
                            if hasattr(self, 'append_log'):
                                self.append_log(f"⚠️ Failed to save progress: {e}")
                            else:
                                print(f"⚠️ Failed to save progress: {e}")
                    
                    def get_content_hash(self, file_path):
                        """Generate content hash for a file"""
                        hasher = hashlib.sha256()
                        with open(file_path, 'rb') as f:
                            # Read in chunks to handle large files
                            for chunk in iter(lambda: f.read(4096), b""):
                                hasher.update(chunk)
                        return hasher.hexdigest()
                    
                    def check_image_status(self, image_path, content_hash):
                        """Check if an image needs translation"""
                        image_name = os.path.basename(image_path)
                        
                        # NEW: Check for skip markers created by "Mark as Skipped" button
                        skip_key = f"skip_{image_name}"
                        if skip_key in self.prog:
                            skip_info = self.prog[skip_key]
                            if skip_info.get('status') == 'skipped':
                                return False, f"Image marked as skipped", None
                        
                        # NEW: Check if image already exists in images folder (marked as skipped)
                        if self.output_dir:
                            images_dir = os.path.join(self.output_dir, "images")
                            dest_image_path = os.path.join(images_dir, image_name)
                            
                            if os.path.exists(dest_image_path):
                                return False, f"Image in skipped folder", None
                        
                        # Check if image has already been processed
                        if content_hash in self.prog["images"]:
                            image_info = self.prog["images"][content_hash]
                            status = image_info.get("status")
                            output_file = image_info.get("output_file")
                            
                            if status == "completed" and output_file:
                                # Check if output file exists
                                if output_file and os.path.exists(output_file):
                                    return False, f"Image already translated: {output_file}", output_file
                                else:
                                    # Output file missing, mark for retranslation
                                    image_info["status"] = "file_deleted"
                                    image_info["deletion_detected"] = time.time()
                                    self.save()
                                    return True, None, None
                            
                            elif status == "skipped_cover":
                                return False, "Cover image - skipped", None
                            
                            elif status == "error":
                                # Previous error, retry
                                return True, None, None
                        
                        return True, None, None
                    
                    def update(self, image_path, content_hash, output_file=None, status="in_progress", error=None):
                        """Update progress for an image"""
                        image_name = os.path.basename(image_path)
                        
                        image_info = {
                            "name": image_name,
                            "path": image_path,
                            "content_hash": content_hash,
                            "status": status,
                            "last_updated": time.time()
                        }
                        
                        if output_file:
                            image_info["output_file"] = output_file
                        
                        if error:
                            image_info["error"] = str(error)
                        
                        self.prog["images"][content_hash] = image_info
                        
                        # Update content hash index for duplicates
                        if status == "completed" and output_file:
                            self.prog["content_hashes"][content_hash] = {
                                "original_name": image_name,
                                "output_file": output_file
                            }
                        
                        self.save()
                
                # Initialize the progress manager
                self.image_progress_manager = ImageProgressManager(output_dir)
                # Add append_log reference for the progress manager
                self.image_progress_manager.append_log = self.append_log
                self.append_log(f"📊 Progress tracking in: {os.path.join(output_dir, 'translation_progress.json')}")
            
            # Check for stop request early
            if self.stop_requested:
                self.append_log("⏹️ Image translation cancelled by user")
                return False
            
            # Get content hash for the image
            try:
                content_hash = self.image_progress_manager.get_content_hash(image_path)
            except Exception as e:
                self.append_log(f"⚠️ Could not generate content hash: {e}")
                # Fallback to using file path as identifier
                content_hash = hashlib.sha256(image_path.encode()).hexdigest()
            
            # Check if image needs translation
            needs_translation, skip_reason, existing_output = self.image_progress_manager.check_image_status(
                image_path, content_hash
            )
            
            if not needs_translation:
                self.append_log(f"⏭️ {skip_reason}")
                
                # NEW: If image is marked as skipped but not in images folder yet, copy it there
                if "marked as skipped" in skip_reason and combined_output_dir:
                    images_dir = os.path.join(combined_output_dir, "images")
                    os.makedirs(images_dir, exist_ok=True)
                    dest_image = os.path.join(images_dir, image_name)
                    if not os.path.exists(dest_image):
                        shutil.copy2(image_path, dest_image)
                        self.append_log(f"📁 Copied skipped image to: {dest_image}")
                
                return True
            
            # Update progress to "in_progress"
            self.image_progress_manager.update(image_path, content_hash, status="in_progress")
            
            # Check if image translation is enabled
            if not hasattr(self, 'enable_image_translation_var') or not self.enable_image_translation_var:
                self.append_log(f"⚠️ Image translation not enabled. Enable it in settings to translate images.")
                return False
            
            # Check for cover images
            if 'cover' in image_name.lower():
                self.append_log(f"⏭️ Skipping cover image: {image_name}")
                
                # Update progress for cover
                self.image_progress_manager.update(image_path, content_hash, status="skipped_cover")
                
                # Copy cover image to images folder if using combined output
                if combined_output_dir:
                    images_dir = os.path.join(combined_output_dir, "images")
                    os.makedirs(images_dir, exist_ok=True)
                    dest_image = os.path.join(images_dir, image_name)
                    if not os.path.exists(dest_image):
                        shutil.copy2(image_path, dest_image)
                        self.append_log(f"📁 Copied cover to: {dest_image}")
                
                return True  # Return True to indicate successful skip (not an error)
            
            # Check for stop before processing
            if self.stop_requested:
                self.append_log("⏹️ Image translation cancelled before processing")
                self.image_progress_manager.update(image_path, content_hash, status="cancelled")
                return False
            
            # Get the file index for numbering
            file_index = getattr(self, 'current_file_index', 0) + 1
            
            # Get API key and model
            api_key = self.api_key_entry.text().strip()
            model = self.model_var.strip()
            
            if not api_key:
                self.append_log("❌ Error: Please enter your API key.")
                self.image_progress_manager.update(image_path, content_hash, status="error", error="No API key")
                return False
            
            if not model:
                self.append_log("❌ Error: Please select a model.")
                self.image_progress_manager.update(image_path, content_hash, status="error", error="No model selected")
                return False
            
            self.append_log(f"🖼️ Processing image: {os.path.basename(image_path)}")
            self.append_log(f"🤖 Using model: {model}")
            
            # Check if it's a vision-capable model
            vision_models = [
                'claude-opus-4-20250514', 'claude-sonnet-4-20250514',
                'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-5-mini','gpt-5','gpt-5-nano',
                'gpt-4-vision-preview',
                'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-exp',
                'gemini-2.5-pro', 'gemini-2.5-flash',
                'llama-3.2-11b-vision', 'llama-3.2-90b-vision',
                'eh/gemini-2.5-flash', 'eh/gemini-1.5-flash', 'eh/gpt-4o'  # ElectronHub variants
            ]
            
            model_lower = model.lower()
            if not any(vm in model_lower for vm in [m.lower() for m in vision_models]):
                self.append_log(f"⚠️ Model '{model}' may not support vision. Trying anyway...")
            
            # Check for stop before API initialization
            if self.stop_requested:
                self.append_log("⏹️ Image translation cancelled before API initialization")
                self.image_progress_manager.update(image_path, content_hash, status="cancelled")
                return False
            
            # Initialize API client with output_dir to enable multi-key mode from environment
            try:
                from unified_api_client import UnifiedClient
                # Pass output_dir to enable environment-based multi-key initialization
                client = UnifiedClient(model=model, api_key=api_key, output_dir=output_dir)
                
                # Set stop flag if the client supports it
                if hasattr(client, 'set_stop_flag'):
                    client.set_stop_flag(self.stop_requested)
                elif hasattr(client, 'stop_flag'):
                    client.stop_flag = self.stop_requested
                    
            except Exception as e:
                self.append_log(f"❌ Failed to initialize API client: {str(e)}")
                self.image_progress_manager.update(image_path, content_hash, status="error", error=f"API client init failed: {e}")
                return False
            
            # Read the image
            try:
                # Get image name for payload naming
                base_name = os.path.splitext(image_name)[0]
                
                with open(image_path, 'rb') as img_file:
                    image_data = img_file.read()
                
                # Convert to base64
                import base64
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                
                # Check image size
                size_mb = len(image_data) / (1024 * 1024)
                self.append_log(f"📊 Image size: {size_mb:.2f} MB")
                
            except Exception as e:
                self.append_log(f"❌ Failed to read image: {str(e)}")
                self.image_progress_manager.update(image_path, content_hash, status="error", error=f"Failed to read image: {e}")
                return False
            
            # Get system prompt from configuration
            profile_name = self.config.get('active_profile', 'korean')
            prompt_profiles = self.config.get('prompt_profiles', {})
            
            # Get the main translation prompt
            system_prompt = ""
            if isinstance(prompt_profiles, dict) and profile_name in prompt_profiles:
                profile_data = prompt_profiles[profile_name]
                if isinstance(profile_data, str):
                    # Old format: prompt_profiles[profile_name] = "prompt text"
                    system_prompt = profile_data
                elif isinstance(profile_data, dict):
                    # New format: prompt_profiles[profile_name] = {"prompt": "...", "book_title_prompt": "..."}
                    system_prompt = profile_data.get('prompt', '')
            else:
                # Fallback to check if prompt is stored directly in config
                system_prompt = self.config.get(profile_name, '')
            
            if not system_prompt:
                # Last fallback - empty string
                system_prompt = ""

            # Check if we should append glossary to the prompt
            append_glossary = self.config.get('append_glossary', True)  # Default to True
            if hasattr(self, 'append_glossary_var'):
                append_glossary = self.append_glossary_var
            
            # Check if automatic glossary is enabled
            enable_auto_glossary = self.config.get('enable_auto_glossary', False)
            if hasattr(self, 'enable_auto_glossary_var'):
                enable_auto_glossary = self.enable_auto_glossary_var
            
            if append_glossary:
                # Check for manual glossary
                manual_glossary_path = os.getenv('MANUAL_GLOSSARY')
                if not manual_glossary_path and hasattr(self, 'manual_glossary_path'):
                    manual_glossary_path = self.manual_glossary_path
                
                # If automatic glossary is enabled and no manual glossary exists, defer appending
                if enable_auto_glossary and (not manual_glossary_path or not os.path.exists(manual_glossary_path)):
                    self.append_log(f"📑 Automatic glossary enabled - glossary will be appended after generation")
                    # Set a flag to indicate deferred glossary appending
                    os.environ['DEFER_GLOSSARY_APPEND'] = '1'
                    # Store the append prompt for later use
                    glossary_prompt = self.config.get('append_glossary_prompt', 
                        "- Follow this reference glossary for consistent translation (Do not output any raw entries):\n")
                    os.environ['GLOSSARY_APPEND_PROMPT'] = glossary_prompt
                else:
                    # Original behavior - append manual glossary immediately
                    if manual_glossary_path and os.path.exists(manual_glossary_path):
                        try:
                            self.append_log(f"📑 Loading glossary for system prompt: {os.path.basename(manual_glossary_path)}")
                            
                            # Copy to output as the same extension, and prefer CSV naming
                            ext = os.path.splitext(manual_glossary_path)[1].lower()
                            out_name = "glossary.csv" if ext == ".csv" else "glossary.json"
                            output_glossary_path = os.path.join(output_dir, out_name)
                            try:
                                import shutil as _shutil
                                _shutil.copy(manual_glossary_path, output_glossary_path)
                                self.append_log(f"💾 Saved glossary to output folder for auto-loading: {out_name}")
                            except Exception as copy_err:
                                self.append_log(f"⚠️ Could not copy glossary into output: {copy_err}")
                            
                            # Append to prompt
                            if ext == ".csv":
                                with open(manual_glossary_path, 'r', encoding='utf-8') as f:
                                    csv_text = f.read()
                                if system_prompt:
                                    system_prompt += "\n\n"
                                glossary_prompt = self.config.get('append_glossary_prompt', 
                                    "- Follow this reference glossary for consistent translation (Do not output any raw entries):\n")
                                system_prompt += f"{glossary_prompt}\n{csv_text}"
                                self.append_log(f"✅ Appended CSV glossary to system prompt")
                            else:
                                with open(manual_glossary_path, 'r', encoding='utf-8') as f:
                                    glossary_data = json.load(f)
                                
                                formatted_entries = {}
                                if isinstance(glossary_data, list):
                                    for char in glossary_data:
                                        if not isinstance(char, dict):
                                            continue
                                        original = char.get('original_name', '')
                                        translated = char.get('name', original)
                                        if original and translated:
                                            formatted_entries[original] = translated
                                        title = char.get('title')
                                        if title and original:
                                            formatted_entries[f"{original} ({title})"] = f"{translated} ({title})"
                                        refer_map = char.get('how_they_refer_to_others', {})
                                        if isinstance(refer_map, dict):
                                            for other_name, reference in refer_map.items():
                                                if other_name and reference:
                                                    formatted_entries[f"{original} → {other_name}"] = f"{translated} → {reference}"
                                elif isinstance(glossary_data, dict):
                                    if "entries" in glossary_data and isinstance(glossary_data["entries"], dict):
                                        formatted_entries = glossary_data["entries"]
                                    else:
                                        formatted_entries = {k: v for k, v in glossary_data.items() if k != "metadata"}
                                if formatted_entries:
                                    glossary_block = json.dumps(formatted_entries, ensure_ascii=False, indent=2)
                                    if system_prompt:
                                        system_prompt += "\n\n"
                                    glossary_prompt = self.config.get('append_glossary_prompt', 
                                        "- Follow this reference glossary for consistent translation (Do not output any raw entries):\n")
                                    system_prompt += f"{glossary_prompt}\n{glossary_block}"
                                    self.append_log(f"✅ Added {len(formatted_entries)} glossary entries to system prompt")
                                else:
                                    self.append_log(f"⚠️ Glossary file has no valid entries")
                                
                        except Exception as e:
                            self.append_log(f"⚠️ Failed to append glossary to prompt: {str(e)}")
                    else:
                        self.append_log(f"ℹ️ No glossary file found to append to prompt")
            else:
                self.append_log(f"ℹ️ Glossary appending disabled in settings")
                # Clear any deferred append flag
                if 'DEFER_GLOSSARY_APPEND' in os.environ:
                    del os.environ['DEFER_GLOSSARY_APPEND']
            
            # Get temperature and max tokens from GUI
            temperature = float(self.temperature_entry.text()) if hasattr(self, 'temperature_entry') else 0.3
            max_tokens = int(self.max_output_tokens_var) if hasattr(self, 'max_output_tokens_var') else 8192
            
            # Build messages for vision API
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            self.append_log(f"🌐 Sending image to vision API...")
            self.append_log(f"   System prompt length: {len(system_prompt)} chars")
            self.append_log(f"   Temperature: {temperature}")
            self.append_log(f"   Max tokens: {max_tokens}")          
            
            # Debug: Show first 200 chars of system prompt
            if system_prompt:
                preview = system_prompt[:] if len(system_prompt) > 200 else system_prompt
                self.append_log(f"   System prompt: {preview}")
            
            # Check stop before making API call
            if self.stop_requested:
                self.append_log("⏹️ Image translation cancelled before API call")
                self.image_progress_manager.update(image_path, content_hash, status="cancelled")
                return False
            
            # Make the API call
            try:
                # Create Payloads directory for API response tracking
                payloads_dir = "Payloads"
                os.makedirs(payloads_dir, exist_ok=True)
                
                # Create timestamp for unique filename
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                payload_file = os.path.join(payloads_dir, f"image_api_{timestamp}_{base_name}.json")
                
                # Save the request payload
                request_payload = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": model,
                    "image_file": image_name,
                    "image_size_mb": size_mb,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "messages": messages,
                    "image_base64": image_base64  # Full payload without truncation
                }
                
                with open(payload_file, 'w', encoding='utf-8') as f:
                    json.dump(request_payload, f, ensure_ascii=False, indent=2)
                
                self.append_log(f"📝 Saved request payload: {payload_file}")
                
                # Call the vision API with interrupt support
                # Check if the client supports a stop_callback parameter
                # Import the send_with_interrupt function from TransateKRtoEN
                try:
                    from TransateKRtoEN import send_with_interrupt
                except ImportError:
                    self.append_log("⚠️ send_with_interrupt not available, using direct call")
                    send_with_interrupt = None
                
                # Call the vision API with interrupt support
                if send_with_interrupt:
                    # For image calls, we need a wrapper since send_with_interrupt expects client.send()
                    # Create a temporary wrapper client that handles image calls
                    class ImageClientWrapper:
                        def __init__(self, real_client, image_data):
                            self.real_client = real_client
                            self.image_data = image_data
                        
                        def send(self, messages, temperature, max_tokens):
                            return self.real_client.send_image(messages, self.image_data, temperature=temperature, max_tokens=max_tokens)
                        
                        def __getattr__(self, name):
                            return getattr(self.real_client, name)
                    
                    # Create wrapped client
                    wrapped_client = ImageClientWrapper(client, image_base64)
                    
                    # Use send_with_interrupt
                    response = send_with_interrupt(
                        messages,
                        wrapped_client,
                        temperature,
                        max_tokens,
                        lambda: self.stop_requested,
                        chunk_timeout=self.config.get('chunk_timeout', 300)  # 5 min default
                    )
                else:
                    # Fallback to direct call
                    response = client.send_image(
                        messages,
                        image_base64,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                
                # Check if stopped after API call
                if self.stop_requested:
                    self.append_log("⏹️ Image translation stopped after API call")
                    self.image_progress_manager.update(image_path, content_hash, status="cancelled")
                    return False
                
                # Extract content and finish reason from response
                response_content = None
                finish_reason = None
                
                if hasattr(response, 'content'):
                    response_content = response.content
                    finish_reason = response.finish_reason if hasattr(response, 'finish_reason') else 'unknown'
                elif isinstance(response, tuple) and len(response) >= 2:
                    # Handle tuple response (content, finish_reason)
                    response_content, finish_reason = response
                elif isinstance(response, str):
                    # Handle direct string response
                    response_content = response
                    finish_reason = 'complete'
                else:
                    self.append_log(f"❌ Unexpected response type: {type(response)}")
                    self.append_log(f"   Response: {response}")
                
                # Save the response payload
                response_payload = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "response_content": response_content,
                    "finish_reason": finish_reason,
                    "content_length": len(response_content) if response_content else 0
                }
                
                response_file = os.path.join(payloads_dir, f"image_api_response_{timestamp}_{base_name}.json")
                with open(response_file, 'w', encoding='utf-8') as f:
                    json.dump(response_payload, f, ensure_ascii=False, indent=2)
                
                self.append_log(f"📝 Saved response payload: {response_file}")
                
                # Check if we got valid content
                if not response_content or response_content.strip() == "[IMAGE TRANSLATION FAILED]":
                    self.append_log(f"❌ Image translation failed - no text extracted from image")
                    self.append_log(f"   This may mean:")
                    self.append_log(f"   - The image doesn't contain readable text")
                    self.append_log(f"   - The model couldn't process the image")
                    self.append_log(f"   - The image format is not supported")
                    
                    # Try to get more info about the failure
                    if hasattr(response, 'error_details'):
                        self.append_log(f"   Error details: {response.error_details}")
                    
                    self.image_progress_manager.update(image_path, content_hash, status="error", error="No text extracted")
                    return False
                
                if response_content:
                    self.append_log(f"✅ Received translation from API")
                    
                    # We already have output_dir defined at the top
                    # Copy original image to the output directory if not using combined output
                    if not combined_output_dir and not os.path.exists(os.path.join(output_dir, image_name)):
                        shutil.copy2(image_path, os.path.join(output_dir, image_name))
                    
                    # Get book title prompt for translating the filename
                    book_title_prompt = self.config.get('book_title_prompt', '')
                    book_title_system_prompt = self.config.get('book_title_system_prompt', '')
                    
                    # If no book title prompt in main config, check in profile
                    if not book_title_prompt and isinstance(prompt_profiles, dict) and profile_name in prompt_profiles:
                        profile_data = prompt_profiles[profile_name]
                        if isinstance(profile_data, dict):
                            book_title_prompt = profile_data.get('book_title_prompt', '')
                            # Also check for system prompt in profile
                            if 'book_title_system_prompt' in profile_data:
                                book_title_system_prompt = profile_data['book_title_system_prompt']
                    
                    # If still no book title prompt, use the main system prompt
                    if not book_title_prompt:
                        book_title_prompt = system_prompt
                    
                    # If no book title system prompt configured, use the main system prompt
                    if not book_title_system_prompt:
                        book_title_system_prompt = system_prompt
                    
                    # Translate the image filename/title
                    self.append_log(f"📝 Translating image title...")
                    title_messages = [
                        {"role": "system", "content": book_title_system_prompt},
                        {"role": "user", "content": f"{book_title_prompt}\n\n{base_name}" if book_title_prompt != system_prompt else base_name}
                    ]
                    
                    try:
                        # Check for stop before title translation
                        if self.stop_requested:
                            self.append_log("⏹️ Image translation cancelled before title translation")
                            self.image_progress_manager.update(image_path, content_hash, status="cancelled")
                            return False
                        
                        title_response = client.send(
                            title_messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        
                        # Extract title translation
                        if hasattr(title_response, 'content'):
                            translated_title = title_response.content.strip() if title_response.content else base_name
                        else:
                            # Handle tuple response
                            title_content, *_ = title_response
                            translated_title = title_content.strip() if title_content else base_name
                    except Exception as e:
                        self.append_log(f"⚠️ Title translation failed: {str(e)}")
                        translated_title = base_name  # Fallback to original if translation fails
                    
                    # Create clean HTML content with just the translated title and content
                    html_content = f'''<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <title>{translated_title}</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                line-height: 1.6; 
                margin: 40px;
                max-width: 800px;
            }}
            h1 {{
                color: #333;
                border-bottom: 2px solid #0066cc;
                padding-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>{translated_title}</h1>
        {response_content}
    </body>
    </html>'''
                    
                    # Save HTML file with proper numbering
                    html_file = os.path.join(output_dir, f"response_{file_index:03d}_{base_name}.html")
                    with open(html_file, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    
                    # Copy original image to the output directory (for reference, not displayed)
                    if not combined_output_dir:
                        shutil.copy2(image_path, os.path.join(output_dir, image_name))
                    
                    # Update progress to completed
                    self.image_progress_manager.update(image_path, content_hash, output_file=html_file, status="completed")
                    
                    # Show preview
                    if response_content and response_content.strip():
                        preview = response_content[:200] + "..." if len(response_content) > 200 else response_content
                        self.append_log(f"📝 Translation preview:")
                        self.append_log(f"{preview}")
                    else:
                        self.append_log(f"⚠️ Translation appears to be empty")
                    
                    self.append_log(f"✅ Translation saved to: {html_file}")
                    self.append_log(f"📁 Output directory: {output_dir}")
                    
                    return True
                else:
                    self.append_log(f"❌ No translation received from API")
                    if finish_reason:
                        self.append_log(f"   Finish reason: {finish_reason}")
                    self.image_progress_manager.update(image_path, content_hash, status="error", error="No response from API")
                    return False
                    
            except Exception as e:
                # Check if this was a stop/interrupt exception
                if "stop" in str(e).lower() or "interrupt" in str(e).lower() or self.stop_requested:
                    self.append_log("⏹️ Image translation interrupted")
                    self.image_progress_manager.update(image_path, content_hash, status="cancelled")
                    return False
                else:
                    self.append_log(f"❌ API call failed: {str(e)}")
                    import traceback
                    self.append_log(f"❌ Full error: {traceback.format_exc()}")
                    self.image_progress_manager.update(image_path, content_hash, status="error", error=f"API call failed: {e}")
                    return False
            
        except Exception as e:
            self.append_log(f"❌ Error processing image: {str(e)}")
            import traceback
            self.append_log(f"❌ Full error: {traceback.format_exc()}")
            return False
        
    def _process_text_file(self, file_path):
        """Process EPUB or TXT file (existing translation logic)"""
        try:
            if translation_main is None:
                self.append_log("❌ Translation module is not available")
                return False

            api_key = self.api_key_entry.text()
            model = self.model_var
            
            # Validate API key and model (same as original)
            if '@' in model or model.startswith('vertex/'):
                google_creds = self.config.get('google_cloud_credentials')
                if not google_creds or not os.path.exists(google_creds):
                    self.append_log("❌ Error: Google Cloud credentials required for Vertex AI models.")
                    return False
                
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_creds
                self.append_log(f"🔑 Using Google Cloud credentials: {os.path.basename(google_creds)}")
                
                if not api_key:
                    try:
                        with open(google_creds, 'r') as f:
                            creds_data = json.load(f)
                            api_key = creds_data.get('project_id', 'vertex-ai-project')
                            self.append_log(f"🔑 Using project ID as API key: {api_key}")
                    except:
                        api_key = 'vertex-ai-project'
            elif not api_key:
                self.append_log("❌ Error: Please enter your API key.")
                return False

            # Determine output directory and save source EPUB path
            if file_path.lower().endswith('.epub'):
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_dir = base_name  # This is how your code determines the output dir
                
                # Save source EPUB path for EPUB converter
                source_epub_file = os.path.join(output_dir, 'source_epub.txt')
                try:
                    os.makedirs(output_dir, exist_ok=True)  # Ensure output dir exists
                    with open(source_epub_file, 'w', encoding='utf-8') as f:
                        f.write(file_path)
                    self.append_log(f"📚 Saved source EPUB reference for chapter ordering")
                except Exception as e:
                    self.append_log(f"⚠️ Could not save source EPUB reference: {e}")
                
                # Set EPUB_PATH in environment for immediate use
                os.environ['EPUB_PATH'] = file_path
                
            old_argv = sys.argv
            old_env = dict(os.environ)
            

            try:
                # Set up environment (same as original)
                self.append_log(f"🔧 Setting up environment variables...")
                self.append_log(f"📖 File: {os.path.basename(file_path)}")
                self.append_log(f"🤖 Model: {self.model_var}")
                
                # Get the system prompt and log first 100 characters
                system_prompt = self.prompt_text.toPlainText().strip()
                prompt_preview = system_prompt[:] if len(system_prompt) > 100 else system_prompt
                self.append_log(f"📝 System prompt: {prompt_preview}")
                self.append_log(f"📏 System prompt length: {len(system_prompt)} characters")
                
                # Check if glossary info is in the system prompt
                if "glossary" in system_prompt.lower() or "character entry" in system_prompt.lower():
                    self.append_log(f"📚 ✅ Glossary appears to be included in system prompt")
                else:
                    self.append_log(f"📚 ⚠️ No glossary detected in system prompt")
                
                # Log glossary status
                if hasattr(self, 'manual_glossary_path') and self.manual_glossary_path:
                    self.append_log(f"📑 Manual glossary loaded: {os.path.basename(self.manual_glossary_path)}")
                else:
                    self.append_log(f"📑 No manual glossary loaded")
                
                # IMPORTANT: Set IS_TEXT_FILE_TRANSLATION flag for text files
                if file_path.lower().endswith('.txt'):
                    os.environ['IS_TEXT_FILE_TRANSLATION'] = '1'
                    self.append_log("📄 Processing as text file")
                
                # Set environment variables
                env_vars = self._get_environment_variables(file_path, api_key)
                
                # Enable async chapter extraction for EPUBs to prevent GUI freezing
                if file_path.lower().endswith('.epub'):
                    env_vars['USE_ASYNC_CHAPTER_EXTRACTION'] = '1'
                    self.append_log("🚀 Using async chapter extraction (subprocess mode)")
                
                os.environ.update(env_vars)
                
                # Handle chapter range
                chap_range = self.chapter_range_entry.text().strip()
                if chap_range:
                    os.environ['CHAPTER_RANGE'] = chap_range
                    self.append_log(f"📊 Chapter Range: {chap_range}")
                
                # Set other environment variables (token limits, etc.)
                if hasattr(self, 'token_limit_disabled') and self.token_limit_disabled:
                    os.environ['MAX_INPUT_TOKENS'] = ''
                else:
                    token_val = self.token_limit_entry.text().strip()
                    if token_val and token_val.isdigit():
                        os.environ['MAX_INPUT_TOKENS'] = token_val
                    else:
                        os.environ['MAX_INPUT_TOKENS'] = '1000000'
                
                # Validate glossary path
                if hasattr(self, 'manual_glossary_path') and self.manual_glossary_path:
                    if (hasattr(self, 'auto_loaded_glossary_path') and 
                        self.manual_glossary_path == self.auto_loaded_glossary_path):
                        if (hasattr(self, 'auto_loaded_glossary_for_file') and 
                            hasattr(self, 'file_path') and 
                            self.file_path == self.auto_loaded_glossary_for_file):
                            os.environ['MANUAL_GLOSSARY'] = self.manual_glossary_path
                            self.append_log(f"📑 Using auto-loaded glossary: {os.path.basename(self.manual_glossary_path)}")
                    else:
                        os.environ['MANUAL_GLOSSARY'] = self.manual_glossary_path
                        self.append_log(f"📑 Using manual glossary: {os.path.basename(self.manual_glossary_path)}")
                
                # Set sys.argv to match what TransateKRtoEN.py expects
                sys.argv = ['TransateKRtoEN.py', file_path]
                
                self.append_log("🚀 Starting translation...")
                
                # Ensure Payloads directory exists
                os.makedirs("Payloads", exist_ok=True)
                
                # Run translation
                translation_main(
                    log_callback=self.append_log,
                    stop_callback=lambda: self.stop_requested
                )
                
                if not self.stop_requested:
                    self.append_log("✅ Translation completed successfully!")
                    return True
                else:
                    return False
                    
            except ValueError as e:
                # ValueError is used for user-facing errors like invalid chapter range
                # These already have clear error messages, so no need for traceback
                error_msg = str(e)
                if "Chapter range" not in error_msg:
                    # If it's not a chapter range error, show the message
                    self.append_log(f"❌ Translation error: {e}")
                # Don't show traceback for user-friendly errors
                return False
                
            except Exception as e:
                # For other exceptions, show full details
                self.append_log(f"❌ Translation error: {e}")
                if hasattr(self, 'append_log_with_api_error_detection'):
                    self.append_log_with_api_error_detection(str(e))
                import traceback
                self.append_log(f"❌ Full error: {traceback.format_exc()}")
                return False
            
            finally:
                sys.argv = old_argv
                os.environ.clear()
                os.environ.update(old_env)
                
        except Exception as e:
            self.append_log(f"❌ Error in text file processing: {str(e)}")
            return False

    def _get_environment_variables(self, epub_path, api_key):
        """Get all environment variables for translation/glossary"""

        # Get Google Cloud project ID if using Vertex AI
        google_cloud_project = ''
        model = self.model_var
        if '@' in model or model.startswith('vertex/'):
            google_creds = self.config.get('google_cloud_credentials')
            if google_creds and os.path.exists(google_creds):
                try:
                    with open(google_creds, 'r') as f:
                        creds_data = json.load(f)
                        google_cloud_project = creds_data.get('project_id', '')
                except:
                    pass
                    
        # Handle extraction mode - check which variables exist
        if hasattr(self, 'text_extraction_method_var'):
            # New cleaner UI variables
            extraction_method = self.text_extraction_method_var
            filtering_level = self.file_filtering_level_var
            
            if extraction_method == 'enhanced':
                extraction_mode = 'enhanced'
                enhanced_filtering = filtering_level
            else:
                extraction_mode = filtering_level
                enhanced_filtering = 'smart'  # default
        else:
            # Old UI variables
            extraction_mode = self.extraction_mode_var
            if extraction_mode == 'enhanced':
                enhanced_filtering = getattr(self, 'enhanced_filtering_var', 'smart')
            else:
                enhanced_filtering = 'smart'
                    
        # Ensure multi-key env toggles are set early for the main translation path as well
        try:
            if self.config.get('use_multi_api_keys', False):
                os.environ['USE_MULTI_KEYS'] = '1'
            else:
                os.environ['USE_MULTI_KEYS'] = '0'
            if self.config.get('use_fallback_keys', False):
                os.environ['USE_FALLBACK_KEYS'] = '1'
            else:
                os.environ['USE_FALLBACK_KEYS'] = '0'
        except Exception:
            pass

        return {
            'EPUB_PATH': epub_path,
            'MODEL': self.model_var,
            'CONTEXTUAL': '1' if self.contextual_var else '0',
            'SEND_INTERVAL_SECONDS': str(self.delay_entry.text()),
            'THREAD_SUBMISSION_DELAY_SECONDS': self.thread_delay_entry.text().strip() or '0.5',
            'MAX_OUTPUT_TOKENS': str(self.max_output_tokens),
            'API_KEY': api_key,
            'OPENAI_API_KEY': api_key,
            'OPENAI_OR_Gemini_API_KEY': api_key,
            'GEMINI_API_KEY': api_key,
            'SYSTEM_PROMPT': self.prompt_text.toPlainText().strip(),
            'TRANSLATE_BOOK_TITLE': "1" if self.translate_book_title_var else "0",
            'BOOK_TITLE_PROMPT': self.book_title_prompt,
            'BOOK_TITLE_SYSTEM_PROMPT': self.config.get('book_title_system_prompt', 
                "You are a translator. Respond with only the translated text, nothing else. Do not add any explanation or additional content."),
            'REMOVE_AI_ARTIFACTS': "1" if self.REMOVE_AI_ARTIFACTS_var else "0",
            'USE_ROLLING_SUMMARY': "1" if (hasattr(self, 'rolling_summary_var') and self.rolling_summary_var) else ("1" if self.config.get('use_rolling_summary') else "0"),
            'SUMMARY_ROLE': self.config.get('summary_role', 'user'),
            'ROLLING_SUMMARY_EXCHANGES': str(self.rolling_summary_exchanges_var),
            'ROLLING_SUMMARY_MODE': str(self.rolling_summary_mode_var),
            'ROLLING_SUMMARY_SYSTEM_PROMPT': str(self.rolling_summary_system_prompt),
            'ROLLING_SUMMARY_USER_PROMPT': str(self.rolling_summary_user_prompt),
            'ROLLING_SUMMARY_MAX_ENTRIES': str(self.rolling_summary_max_entries_var),
            'PROFILE_NAME': self.lang_var.lower(),
            'TRANSLATION_TEMPERATURE': str(self.trans_temp.text()),
            'TRANSLATION_HISTORY_LIMIT': str(self.trans_history.text()),
            'EPUB_OUTPUT_DIR': os.getcwd(),
            'APPEND_GLOSSARY': "1" if self.append_glossary_var else "0",
            'APPEND_GLOSSARY_PROMPT': self.append_glossary_prompt,
            'EMERGENCY_PARAGRAPH_RESTORE': "1" if self.emergency_restore_var else "0",
            'REINFORCEMENT_FREQUENCY': str(self.reinforcement_freq_var),
            'RETRY_TRUNCATED': "1" if self.retry_truncated_var else "0",
            'MAX_RETRY_TOKENS': str(self.max_retry_tokens_var),
            'RETRY_DUPLICATE_BODIES': "1" if self.retry_duplicate_var else "0",
            'PRESERVE_ORIGINAL_TEXT_ON_FAILURE': "1" if self.preserve_original_text_var else "0",
            'DUPLICATE_LOOKBACK_CHAPTERS': str(self.duplicate_lookback_var),
            'GLOSSARY_MIN_FREQUENCY': str(self.glossary_min_frequency_var),
            'GLOSSARY_MAX_NAMES': str(self.glossary_max_names_var),
            'GLOSSARY_MAX_TITLES': str(self.glossary_max_titles_var),
            'GLOSSARY_BATCH_SIZE': str(self.glossary_batch_size_var),
            'GLOSSARY_STRIP_HONORIFICS': "1" if self.strip_honorifics_var else "0",
            'GLOSSARY_CHAPTER_SPLIT_THRESHOLD': str(self.glossary_chapter_split_threshold_var),
            'GLOSSARY_FILTER_MODE': self.glossary_filter_mode_var,
            'ENABLE_AUTO_GLOSSARY': "1" if self.enable_auto_glossary_var else "0",
            'AUTO_GLOSSARY_PROMPT': self.auto_glossary_prompt if hasattr(self, 'auto_glossary_prompt') else '',
            'APPEND_GLOSSARY_PROMPT': self.append_glossary_prompt if hasattr(self, 'append_glossary_prompt') else '',
            'GLOSSARY_TRANSLATION_PROMPT': self.glossary_translation_prompt if hasattr(self, 'glossary_translation_prompt') else '',
            'GLOSSARY_FORMAT_INSTRUCTIONS': self.glossary_format_instructions if hasattr(self, 'glossary_format_instructions') else '',
            'GLOSSARY_USE_LEGACY_CSV': '1' if self.use_legacy_csv_var else '0',
            'ENABLE_IMAGE_TRANSLATION': "1" if self.enable_image_translation_var else "0",
            'PROCESS_WEBNOVEL_IMAGES': "1" if self.process_webnovel_images_var else "0",
            'WEBNOVEL_MIN_HEIGHT': str(self.webnovel_min_height_var),
            'MAX_IMAGES_PER_CHAPTER': str(self.max_images_per_chapter_var),
            'IMAGE_API_DELAY': '1.0',
            'SAVE_IMAGE_TRANSLATIONS': '1',
            'IMAGE_CHUNK_HEIGHT': str(self.image_chunk_height_var),
            'HIDE_IMAGE_TRANSLATION_LABEL': "1" if self.hide_image_translation_label_var else "0",
            'RETRY_TIMEOUT': "1" if self.retry_timeout_var else "0",
            'CHUNK_TIMEOUT': str(self.chunk_timeout_var),
            # New network/HTTP controls
            'ENABLE_HTTP_TUNING': '1' if self.config.get('enable_http_tuning', False) else '0',
            'CONNECT_TIMEOUT': str(self.config.get('connect_timeout', os.environ.get('CONNECT_TIMEOUT', '10'))),
            'READ_TIMEOUT': str(self.config.get('read_timeout', os.environ.get('READ_TIMEOUT', os.environ.get('CHUNK_TIMEOUT', '180')))),
            'HTTP_POOL_CONNECTIONS': str(self.config.get('http_pool_connections', os.environ.get('HTTP_POOL_CONNECTIONS', '20'))),
            'HTTP_POOL_MAXSIZE': str(self.config.get('http_pool_maxsize', os.environ.get('HTTP_POOL_MAXSIZE', '50'))),
            'IGNORE_RETRY_AFTER': '1' if (hasattr(self, 'ignore_retry_after_var') and self.ignore_retry_after_var) else '0',
            'MAX_RETRIES': str(self.config.get('max_retries', os.environ.get('MAX_RETRIES', '7'))),
            'INDEFINITE_RATE_LIMIT_RETRY': '1' if self.config.get('indefinite_rate_limit_retry', False) else '0',
            # Scanning/QA settings
            'SCAN_PHASE_ENABLED': '1' if self.config.get('scan_phase_enabled', False) else '0',
            'SCAN_PHASE_MODE': self.config.get('scan_phase_mode', 'quick-scan'),
            'QA_AUTO_SEARCH_OUTPUT': '1' if self.config.get('qa_auto_search_output', True) else '0',
            'BATCH_TRANSLATION': "1" if self.batch_translation_var else "0",
            'BATCH_SIZE': str(self.batch_size_var),
            'CONSERVATIVE_BATCHING': "1" if self.conservative_batching_var else "0",
            'DISABLE_ZERO_DETECTION': "1" if self.disable_zero_detection_var else "0",
            'TRANSLATION_HISTORY_ROLLING': "1" if self.translation_history_rolling_var else "0",
            'USE_GEMINI_OPENAI_ENDPOINT': '1' if self.use_gemini_openai_endpoint_var else '0',
            'GEMINI_OPENAI_ENDPOINT': self.gemini_openai_endpoint_var if self.gemini_openai_endpoint_var else '',
            "ATTACH_CSS_TO_CHAPTERS": "1" if self.attach_css_to_chapters_var else "0",
            'GLOSSARY_FUZZY_THRESHOLD': str(self.config.get('glossary_fuzzy_threshold', 0.90)),
            'GLOSSARY_MAX_TEXT_SIZE': str(self.config.get('glossary_max_text_size', 50000)),
            'GLOSSARY_MAX_SENTENCES': str(self.config.get('glossary_max_sentences', 200)),
            'USE_FALLBACK_KEYS': '1' if self.config.get('use_fallback_keys', False) else '0',
            'FALLBACK_KEYS': json.dumps(self.config.get('fallback_keys', [])),

            # Extraction settings
            "EXTRACTION_MODE": extraction_mode,
            "ENHANCED_FILTERING": enhanced_filtering,
            "ENHANCED_PRESERVE_STRUCTURE": "1" if getattr(self, 'enhanced_preserve_structure_var', True) else "0",
            'FORCE_BS_FOR_TRADITIONAL': '1' if getattr(self, 'force_bs_for_traditional_var', False) else '0',
            
            # For new UI
            "TEXT_EXTRACTION_METHOD": extraction_method if hasattr(self, 'text_extraction_method_var') else ('enhanced' if extraction_mode == 'enhanced' else 'standard'),
            "FILE_FILTERING_LEVEL": filtering_level if hasattr(self, 'file_filtering_level_var') else extraction_mode,
            'DISABLE_CHAPTER_MERGING': '1' if self.disable_chapter_merging_var else '0',
            'DISABLE_EPUB_GALLERY': "1" if self.disable_epub_gallery_var else "0",
            'DISABLE_AUTOMATIC_COVER_CREATION': "1" if getattr(self, 'disable_automatic_cover_creation_var', False) else "0",
            'TRANSLATE_COVER_HTML': "1" if getattr(self, 'translate_cover_html_var', False) else "0",
            'DUPLICATE_DETECTION_MODE': str(self.duplicate_detection_mode_var),
            'CHAPTER_NUMBER_OFFSET': str(self.chapter_number_offset_var), 
            'USE_HEADER_AS_OUTPUT': "1" if self.use_header_as_output_var else "0",
            'ENABLE_DECIMAL_CHAPTERS': "1" if self.enable_decimal_chapters_var else "0",
            'ENABLE_WATERMARK_REMOVAL': "1" if self.enable_watermark_removal_var else "0",
            'ADVANCED_WATERMARK_REMOVAL': "1" if self.advanced_watermark_removal_var else "0",
            'SAVE_CLEANED_IMAGES': "1" if self.save_cleaned_images_var else "0",
            'COMPRESSION_FACTOR': str(self.compression_factor_var),
            'DISABLE_GEMINI_SAFETY': str(self.config.get('disable_gemini_safety', False)).lower(),
            'GLOSSARY_DUPLICATE_KEY_MODE': self.config.get('glossary_duplicate_key_mode', 'auto'),
            'GLOSSARY_DUPLICATE_CUSTOM_FIELD': self.config.get('glossary_duplicate_custom_field', ''),
            'MANUAL_GLOSSARY': self.manual_glossary_path if hasattr(self, 'manual_glossary_path') and self.manual_glossary_path else '',
            'FORCE_NCX_ONLY': '1' if self.force_ncx_only_var else '0',
            'SINGLE_API_IMAGE_CHUNKS': "1" if self.single_api_image_chunks_var else "0",
            'ENABLE_GEMINI_THINKING': "1" if self.enable_gemini_thinking_var else "0",
            'THINKING_BUDGET': self.thinking_budget_var if self.enable_gemini_thinking_var else '0',
            # GPT/OpenRouter reasoning
            'ENABLE_GPT_THINKING': "1" if self.enable_gpt_thinking_var else "0",
            'GPT_REASONING_TOKENS': self.gpt_reasoning_tokens_var if self.enable_gpt_thinking_var else '',
            'GPT_EFFORT': self.gpt_effort_var,
            'OPENROUTER_EXCLUDE': '1',
            'OPENROUTER_PREFERRED_PROVIDER': self.config.get('openrouter_preferred_provider', 'Auto'),
            # Custom API endpoints
            'OPENAI_CUSTOM_BASE_URL': self.openai_base_url_var if self.openai_base_url_var else '',
            'GROQ_API_URL': self.groq_base_url_var if self.groq_base_url_var else '',
            'FIREWORKS_API_URL': self.fireworks_base_url_var if hasattr(self, 'fireworks_base_url_var') and self.fireworks_base_url_var else '',
            'USE_CUSTOM_OPENAI_ENDPOINT': '1' if self.use_custom_openai_endpoint_var else '0',

            # Image compression settings
            'ENABLE_IMAGE_COMPRESSION': "1" if self.config.get('enable_image_compression', False) else "0",
            'AUTO_COMPRESS_ENABLED': "1" if self.config.get('auto_compress_enabled', True) else "0",
            'TARGET_IMAGE_TOKENS': str(self.config.get('target_image_tokens', 1000)),
            'IMAGE_COMPRESSION_FORMAT': self.config.get('image_compression_format', 'auto'),
            'WEBP_QUALITY': str(self.config.get('webp_quality', 85)),
            'JPEG_QUALITY': str(self.config.get('jpeg_quality', 85)),
            'PNG_COMPRESSION': str(self.config.get('png_compression', 6)),
            'MAX_IMAGE_DIMENSION': str(self.config.get('max_image_dimension', 2048)),
            'MAX_IMAGE_SIZE_MB': str(self.config.get('max_image_size_mb', 10)),
            'PRESERVE_TRANSPARENCY': "1" if self.config.get('preserve_transparency', False) else "0",
            'PRESERVE_ORIGINAL_FORMAT': "1" if self.config.get('preserve_original_format', False) else "0", 
            'OPTIMIZE_FOR_OCR': "1" if self.config.get('optimize_for_ocr', True) else "0",
            'PROGRESSIVE_ENCODING': "1" if self.config.get('progressive_encoding', True) else "0",
            'SAVE_COMPRESSED_IMAGES': "1" if self.config.get('save_compressed_images', False) else "0",
            'IMAGE_CHUNK_OVERLAP_PERCENT': str(self.image_chunk_overlap_var),


            # Metadata and batch header translation settings
            'TRANSLATE_METADATA_FIELDS': json.dumps(self.translate_metadata_fields),
            'METADATA_TRANSLATION_MODE': self.config.get('metadata_translation_mode', 'together'),
            'BATCH_TRANSLATE_HEADERS': "1" if self.batch_translate_headers_var else "0",
            'HEADERS_PER_BATCH': str(self.headers_per_batch_var),
            'UPDATE_HTML_HEADERS': "1" if self.update_html_headers_var else "0",
            'SAVE_HEADER_TRANSLATIONS': "1" if self.save_header_translations_var else "0",
            'METADATA_FIELD_PROMPTS': json.dumps(self.config.get('metadata_field_prompts', {})),
            'LANG_PROMPT_BEHAVIOR': self.config.get('lang_prompt_behavior', 'auto'),
            'FORCED_SOURCE_LANG': self.config.get('forced_source_lang', 'Korean'),
            'OUTPUT_LANGUAGE': self.config.get('output_language', 'English'),
            'METADATA_BATCH_PROMPT': self.config.get('metadata_batch_prompt', ''),
            
            # AI Hunter configuration
            'AI_HUNTER_CONFIG': json.dumps(self.config.get('ai_hunter_config', {})),

            # Anti-duplicate parameters
            'ENABLE_ANTI_DUPLICATE': '1' if hasattr(self, 'enable_anti_duplicate_var') and self.enable_anti_duplicate_var else '0',
            'TOP_P': str(self.top_p_var) if hasattr(self, 'top_p_var') else '1.0',
            'TOP_K': str(self.top_k_var) if hasattr(self, 'top_k_var') else '0',
            'FREQUENCY_PENALTY': str(self.frequency_penalty_var) if hasattr(self, 'frequency_penalty_var') else '0.0',
            'PRESENCE_PENALTY': str(self.presence_penalty_var) if hasattr(self, 'presence_penalty_var') else '0.0',
            'REPETITION_PENALTY': str(self.repetition_penalty_var) if hasattr(self, 'repetition_penalty_var') else '1.0',
            'CANDIDATE_COUNT': str(self.candidate_count_var) if hasattr(self, 'candidate_count_var') else '1',
            'CUSTOM_STOP_SEQUENCES': self.custom_stop_sequences_var if hasattr(self, 'custom_stop_sequences_var') else '',
            'LOGIT_BIAS_ENABLED': '1' if hasattr(self, 'logit_bias_enabled_var') and self.logit_bias_enabled_var else '0',
            'LOGIT_BIAS_STRENGTH': str(self.logit_bias_strength_var) if hasattr(self, 'logit_bias_strength_var') else '-0.5',
            'BIAS_COMMON_WORDS': '1' if hasattr(self, 'bias_common_words_var') and self.bias_common_words_var else '0',
            'BIAS_REPETITIVE_PHRASES': '1' if hasattr(self, 'bias_repetitive_phrases_var') and self.bias_repetitive_phrases_var else '0',
            'GOOGLE_APPLICATION_CREDENTIALS': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''),
            'GOOGLE_CLOUD_PROJECT': google_cloud_project,  # Now properly set from credentials
            'VERTEX_AI_LOCATION': self.vertex_location_var if hasattr(self, 'vertex_location_var') and isinstance(self.vertex_location_var, str) else (self.vertex_location_var.text() if hasattr(self, 'vertex_location_var') and hasattr(self.vertex_location_var, 'text') else 'us-east5'),
            'IS_AZURE_ENDPOINT': '1' if (self.use_custom_openai_endpoint_var and 
                                  ('.azure.com' in self.openai_base_url_var or 
                                   '.cognitiveservices' in self.openai_base_url_var)) else '0',
            'AZURE_API_VERSION': str(self.config.get('azure_api_version', '2024-08-01-preview')),
            
           # Multi API Key support
            'USE_MULTI_API_KEYS': "1" if self.config.get('use_multi_api_keys', False) else "0",
            'MULTI_API_KEYS': json.dumps(self.config.get('multi_api_keys', [])) if self.config.get('use_multi_api_keys', False) else '[]',
            'FORCE_KEY_ROTATION': '1' if self.config.get('force_key_rotation', True) else '0',
            'ROTATION_FREQUENCY': str(self.config.get('rotation_frequency', 1)),
           
       }
        print(f"[DEBUG] DISABLE_CHAPTER_MERGING = '{os.getenv('DISABLE_CHAPTER_MERGING', '0')}'")
        
    def run_glossary_extraction_thread(self):
        """Start glossary extraction in a background worker (ThreadPoolExecutor)"""
        if ((hasattr(self, 'translation_thread') and self.translation_thread and self.translation_thread.is_alive()) or
            (hasattr(self, 'translation_future') and self.translation_future and not self.translation_future.done())):
            self.append_log("⚠️ Cannot run glossary extraction while translation is in progress.")
            QMessageBox.warning(self, "Process Running", "Please wait for translation to complete before extracting glossary.")
            return
        
        if self.glossary_thread and self.glossary_thread.is_alive():
            self.stop_glossary_extraction()
            return
        
        # Check if files are selected
        if not hasattr(self, 'selected_files') or not self.selected_files:
            # Try to get file from entry field (backward compatibility)
            file_path = self.entry_epub.text().strip()
            if not file_path or file_path.startswith("No file selected") or "files selected" in file_path:
                QMessageBox.critical(self, "Error", "Please select file(s) to extract glossary from.")
                return
            self.selected_files = [file_path]
        
        # Reset stop flags
        self.stop_requested = False
        if glossary_stop_flag:
            glossary_stop_flag(False)
        
        # IMPORTANT: Also reset the module's internal stop flag
        try:
            import extract_glossary_from_epub
            extract_glossary_from_epub.set_stop_flag(False)
        except:
            pass
        
        # Use shared executor
        self._ensure_executor()
        if self.executor:
            self.glossary_future = self.executor.submit(self.run_glossary_extraction_direct)
            # Add callback to clean up when done (button update handled by thread_complete_signal in finally block)
            def _glossary_done_callback(f):
                try:
                    self.glossary_future = None
                except Exception:
                    pass
            try:
                self.glossary_future.add_done_callback(_glossary_done_callback)
            except Exception:
                pass
        else:
            thread_name = f"GlossaryThread_{int(time.time())}"
            self.glossary_thread = threading.Thread(target=self.run_glossary_extraction_direct, name=thread_name, daemon=True)
            self.glossary_thread.start()
        
        # Delay auto-scroll so first log is readable (set to 0 for immediate scrolling)
        self._start_autoscroll_delay(0)
        # Update button IMMEDIATELY after thread starts (synchronous)
        self.update_run_button()
        # Force immediate scroll to bottom so user sees the latest output right away
        try:
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        except Exception:
            pass

    def run_glossary_extraction_direct(self):
        """Run glossary extraction directly - handles multiple files and different file types"""
        try:
            # Re-attach GUI logging handlers FIRST to reclaim logs from standalone header translation
            try:
                self._attach_gui_logging_handlers()
            except Exception:
                pass
            
            # Restore print hijack if it was captured by manga translator
            # This ensures main GUI logs go to main GUI, not manga GUI
            try:
                import builtins
                # Check if print was hijacked by manga translator
                if hasattr(builtins, '_manga_log_callbacks') and builtins._manga_log_callbacks:
                    # Restore original print for main GUI
                    if hasattr(builtins, 'print') and hasattr(builtins.print, '__name__'):
                        if builtins.print.__name__ == 'manga_print':
                            # Print is hijacked, restore it
                            from manga_translator import MangaTranslator
                            if hasattr(MangaTranslator, '_original_print_backup'):
                                builtins.print = MangaTranslator._original_print_backup
                                # Also restore in unified_api_client
                                try:
                                    import sys
                                    import unified_api_client
                                    uc_module = sys.modules.get('unified_api_client')
                                    if uc_module:
                                        uc_module.__dict__['print'] = MangaTranslator._original_print_backup
                                except Exception:
                                    pass
            except Exception:
                pass
            
            self.append_log("🔄 Loading glossary modules...")
            if not self._lazy_load_modules():
                self.append_log("❌ Failed to load glossary modules")
                return
            
            if glossary_main is None:
                self.append_log("❌ Glossary extraction module is not available")
                return

            # Create Glossary folder
            os.makedirs("Glossary", exist_ok=True)
            
            # ========== NEW: APPLY OPF-BASED SORTING ==========
            # Sort files based on OPF order if available
            original_file_count = len(self.selected_files)
            self.selected_files = self._get_opf_file_order(self.selected_files)
            self.append_log(f"📚 Processing {original_file_count} files in reading order for glossary extraction")
            # ====================================================
            
            # Group files by type and folder
            image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
            
            # Separate images and text files
            image_files = []
            text_files = []
            
            for file_path in self.selected_files:
                ext = os.path.splitext(file_path)[1].lower()
                if ext in image_extensions:
                    image_files.append(file_path)
                elif ext in {'.epub', '.txt'}:
                    text_files.append(file_path)
                else:
                    self.append_log(f"⚠️ Skipping unsupported file type: {ext}")
            
            # Group images by folder
            image_groups = {}
            for img_path in image_files:
                folder = os.path.dirname(img_path)
                if folder not in image_groups:
                    image_groups[folder] = []
                image_groups[folder].append(img_path)
            
            total_groups = len(image_groups) + len(text_files)
            current_group = 0
            successful = 0
            failed = 0
            
            # Process image groups (each folder gets one combined glossary)
            for folder, images in image_groups.items():
                if self.stop_requested:
                    break
                
                current_group += 1
                folder_name = os.path.basename(folder) if folder else "images"
                
                self.append_log(f"\n{'='*60}")
                self.append_log(f"📁 Processing image folder ({current_group}/{total_groups}): {folder_name}")
                self.append_log(f"   Found {len(images)} images")
                self.append_log(f"{'='*60}")
                
                # Process all images in this folder and extract glossary
                if self._process_image_folder_for_glossary(folder_name, images):
                    successful += 1
                else:
                    failed += 1
            
            # Process text files individually
            for text_file in text_files:
                if self.stop_requested:
                    break
                
                current_group += 1
                
                self.append_log(f"\n{'='*60}")
                self.append_log(f"📄 Processing file ({current_group}/{total_groups}): {os.path.basename(text_file)}")
                self.append_log(f"{'='*60}")
                
                if self._extract_glossary_from_text_file(text_file):
                    successful += 1
                else:
                    failed += 1
            
            # Final summary
            self.append_log(f"\n{'='*60}")
            self.append_log(f"📊 Glossary Extraction Summary:")
            self.append_log(f"   ✅ Successful: {successful} glossaries")
            if failed > 0:
                self.append_log(f"   ❌ Failed: {failed} glossaries")
            self.append_log(f"   📁 Total: {total_groups} glossaries")
            self.append_log(f"   📂 All glossaries saved in: Glossary/")
            self.append_log(f"{'='*60}")
            
        except Exception as e:
            self.append_log(f"❌ Glossary extraction setup error: {e}")
            import traceback
            self.append_log(f"❌ Full error: {traceback.format_exc()}")
        
        finally:
            self.stop_requested = False
            if glossary_stop_flag:
                glossary_stop_flag(False)
            
            # IMPORTANT: Also reset the module's internal stop flag
            try:
                import extract_glossary_from_epub
                extract_glossary_from_epub.set_stop_flag(False)
            except:
                pass
                
            self.glossary_thread = None
            if hasattr(self, 'glossary_future'):
                try:
                    self.glossary_future = None
                except Exception:
                    pass
            self.current_file_index = 0
            # Emit signal to update button (thread-safe)
            self.thread_complete_signal.emit()

    def _process_image_folder_for_glossary(self, folder_name, image_files):
        """Process all images from a folder and create a combined glossary with new format"""
        try:
            import hashlib
            from unified_api_client import UnifiedClient, UnifiedClientError
            
            # Initialize folder-specific progress manager for images
            self.glossary_progress_manager = self._init_image_glossary_progress_manager(folder_name)
            
            all_glossary_entries = []
            processed = 0
            skipped = 0
            
            # Get API key and model
            api_key = self.api_key_entry.text().strip()
            model = self.model_var
            
            if not api_key or not model:
                self.append_log("❌ Error: API key and model required")
                return False
            
            if not self.manual_glossary_prompt:
                self.append_log("❌ Error: No glossary prompt configured")
                return False
            
            # Initialize API client
            try:
                client = UnifiedClient(model=model, api_key=api_key)
            except Exception as e:
                self.append_log(f"❌ Failed to initialize API client: {str(e)}")
                return False
            
            # Get temperature and other settings from glossary config
            temperature = float(self.config.get('manual_glossary_temperature', 0.1))
            max_tokens = int(self.max_output_tokens) if hasattr(self, 'max_output_tokens') else 8192
            api_delay = float(self.delay_entry.text()) if hasattr(self, 'delay_entry') else 2.0
            
            self.append_log(f"🔧 Glossary extraction settings:")
            self.append_log(f"   Temperature: {temperature}")
            self.append_log(f"   Max tokens: {max_tokens}")
            self.append_log(f"   API delay: {api_delay}s")
            format_parts = ["type", "raw_name", "translated_name", "gender"]
            custom_fields_json = self.config.get('manual_custom_fields', '[]')
            try:
                custom_fields = json.loads(custom_fields_json) if isinstance(custom_fields_json, str) else custom_fields_json
                if custom_fields:
                    format_parts.extend(custom_fields)
            except:
                custom_fields = []
            self.append_log(f"   Format: Simple ({', '.join(format_parts)})")
            
            # Check honorifics filter toggle
            honorifics_disabled = self.config.get('glossary_disable_honorifics_filter', False)
            if honorifics_disabled:
                self.append_log(f"   Honorifics Filter: ❌ DISABLED")
            else:
                self.append_log(f"   Honorifics Filter: ✅ ENABLED")
            
            # Track timing for ETA calculation
            start_time = time.time()
            total_entries_extracted = 0
            
            # Set up thread-safe payload directory
            thread_name = threading.current_thread().name
            thread_id = threading.current_thread().ident
            thread_dir = os.path.join("Payloads", "glossary", f"{thread_name}_{thread_id}")
            os.makedirs(thread_dir, exist_ok=True)
            
            # Process each image
            for i, image_path in enumerate(image_files):
                if self.stop_requested:
                    self.append_log("⏹️ Glossary extraction stopped by user")
                    break
                
                image_name = os.path.basename(image_path)
                self.append_log(f"\n   🖼️ Processing image {i+1}/{len(image_files)}: {image_name}")
                
                # Check progress tracking for this image
                try:
                    content_hash = self.glossary_progress_manager.get_content_hash(image_path)
                except Exception as e:
                    content_hash = hashlib.sha256(image_path.encode()).hexdigest()
                
                # Check if already processed
                needs_extraction, skip_reason, _ = self.glossary_progress_manager.check_image_status(image_path, content_hash)
                
                if not needs_extraction:
                    self.append_log(f"      ⏭️ {skip_reason}")
                    # Try to load previous results if available
                    existing_data = self.glossary_progress_manager.get_cached_result(content_hash)
                    if existing_data:
                        all_glossary_entries.extend(existing_data)
                    continue
                
                # Skip cover images
                if 'cover' in image_name.lower():
                    self.append_log(f"      ⏭️ Skipping cover image")
                    self.glossary_progress_manager.update(image_path, content_hash, status="skipped_cover")
                    skipped += 1
                    continue
                
                # Update progress to in-progress
                self.glossary_progress_manager.update(image_path, content_hash, status="in_progress")
                
                try:
                    # Read image
                    with open(image_path, 'rb') as img_file:
                        image_data = img_file.read()
                    
                    import base64
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    size_mb = len(image_data) / (1024 * 1024)
                    base_name = os.path.splitext(image_name)[0]
                    self.append_log(f"      📊 Image size: {size_mb:.2f} MB")
                    
                    # Build prompt for new format
                    custom_fields_json = self.config.get('manual_custom_fields', '[]')
                    try:
                        custom_fields = json.loads(custom_fields_json) if isinstance(custom_fields_json, str) else custom_fields_json
                    except:
                        custom_fields = []
                    
                    # Build honorifics instruction based on toggle
                    honorifics_instruction = ""
                    if not honorifics_disabled:
                        honorifics_instruction = "- Do NOT include honorifics (님, 씨, さん, 様, etc.) in raw_name\n"
                    
                    if self.manual_glossary_prompt:
                        prompt = self.manual_glossary_prompt
                        
                        # Build fields description
                        fields_str = """- type: "character" for people/beings or "term" for locations/objects/concepts
- raw_name: name in the original language/script  
- translated_name: English/romanized translation
- gender: (for characters only) Male/Female/Unknown"""
                        
                        if custom_fields:
                            for field in custom_fields:
                                fields_str += f"\n- {field}: custom field"
                        
                        # Replace placeholders
                        prompt = prompt.replace('{fields}', fields_str)
                        prompt = prompt.replace('{chapter_text}', '')
                        prompt = prompt.replace('{{fields}}', fields_str)
                        prompt = prompt.replace('{{chapter_text}}', '')
                        prompt = prompt.replace('{text}', '')
                        prompt = prompt.replace('{{text}}', '')
                    else:
                        # Default prompt
                        fields_str = """For each entity, provide JSON with these fields:
- type: "character" for people/beings or "term" for locations/objects/concepts
- raw_name: name in the original language/script
- translated_name: English/romanized translation
- gender: (for characters only) Male/Female/Unknown"""
                        
                        if custom_fields:
                            fields_str += "\nAdditional custom fields:"
                            for field in custom_fields:
                                fields_str += f"\n- {field}"
                        
                        prompt = f"""Extract all characters and important terms from this image.

{fields_str}

Important rules:
{honorifics_instruction}- Romanize names appropriately
- Output ONLY a JSON array"""
                    
                    messages = [{"role": "user", "content": prompt}]
                    
                    # Save request payload in thread-safe location
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    payload_file = os.path.join(thread_dir, f"image_{timestamp}_{base_name}_request.json")
                    
                    request_payload = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "model": model,
                        "image_file": image_name,
                        "image_size_mb": size_mb,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "messages": messages,
                        "processed_prompt": prompt,
                        "honorifics_filter_enabled": not honorifics_disabled
                    }
                    
                    with open(payload_file, 'w', encoding='utf-8') as f:
                        json.dump(request_payload, f, ensure_ascii=False, indent=2)
                    
                    self.append_log(f"      📝 Saved request: {os.path.basename(payload_file)}")
                    self.append_log(f"      🌐 Extracting glossary from image...")
                    
                    # API call with interrupt support
                    response = self._call_api_with_interrupt(
                        client, messages, image_base64, temperature, max_tokens
                    )
                    
                    # Check if stopped after API call
                    if self.stop_requested:
                        self.append_log("⏹️ Glossary extraction stopped after API call")
                        self.glossary_progress_manager.update(image_path, content_hash, status="cancelled")
                        return False
                    
                    # Get response content
                    glossary_json = None
                    if isinstance(response, (list, tuple)) and len(response) >= 2:
                        glossary_json = response[0]
                    elif hasattr(response, 'content'):
                        glossary_json = response.content
                    elif isinstance(response, str):
                        glossary_json = response
                    else:
                        glossary_json = str(response)
                    
                    if glossary_json and glossary_json.strip():
                        # Save response in thread-safe location
                        response_file = os.path.join(thread_dir, f"image_{timestamp}_{base_name}_response.json")
                        response_payload = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "response_content": glossary_json,
                            "content_length": len(glossary_json)
                        }
                        with open(response_file, 'w', encoding='utf-8') as f:
                            json.dump(response_payload, f, ensure_ascii=False, indent=2)
                        
                        self.append_log(f"      📝 Saved response: {os.path.basename(response_file)}")
                        
                        # Parse the JSON response
                        try:
                            # Clean up the response
                            glossary_json = glossary_json.strip()
                            if glossary_json.startswith('```'):
                                glossary_json = glossary_json.split('```')[1]
                                if glossary_json.startswith('json'):
                                    glossary_json = glossary_json[4:]
                                glossary_json = glossary_json.strip()
                                if glossary_json.endswith('```'):
                                    glossary_json = glossary_json[:-3].strip()
                            
                            # Parse JSON
                            glossary_data = json.loads(glossary_json)
                            
                            # Process entries
                            entries_for_this_image = []
                            if isinstance(glossary_data, list):
                                for entry in glossary_data:
                                    # Validate entry format
                                    if isinstance(entry, dict) and 'type' in entry and 'raw_name' in entry:
                                        # Clean raw_name
                                        entry['raw_name'] = entry['raw_name'].strip()
                                        
                                        # Ensure required fields
                                        if 'translated_name' not in entry:
                                            entry['translated_name'] = entry.get('name', entry['raw_name'])
                                        
                                        # Add gender for characters if missing
                                        if entry['type'] == 'character' and 'gender' not in entry:
                                            entry['gender'] = 'Unknown'
                                        
                                        entries_for_this_image.append(entry)
                                        all_glossary_entries.append(entry)
                            
                            # Show progress
                            elapsed = time.time() - start_time
                            valid_count = len(entries_for_this_image)
                            
                            for j, entry in enumerate(entries_for_this_image):
                                total_entries_extracted += 1
                                
                                # Calculate ETA
                                if total_entries_extracted == 1:
                                    eta = 0.0
                                else:
                                    avg_time = elapsed / total_entries_extracted
                                    remaining_images = len(image_files) - (i + 1)
                                    estimated_remaining_entries = remaining_images * 3
                                    eta = avg_time * estimated_remaining_entries
                                
                                # Get entry name
                                entry_name = f"{entry['raw_name']} ({entry['translated_name']})"
                                
                                # Print progress
                                progress_msg = f'[Image {i+1}/{len(image_files)}] [{j+1}/{valid_count}] ({elapsed:.1f}s elapsed, ETA {eta:.1f}s) → {entry["type"]}: {entry_name}'
                                print(progress_msg)
                                self.append_log(progress_msg)
                            
                            self.append_log(f"      ✅ Extracted {valid_count} entries")
                            
                            # Update progress with extracted data
                            self.glossary_progress_manager.update(
                                image_path, 
                                content_hash, 
                                status="completed",
                                extracted_data=entries_for_this_image
                            )
                            
                            processed += 1
                            
                            # Save intermediate progress with skip logic
                            if all_glossary_entries:
                                self._save_intermediate_glossary_with_skip(folder_name, all_glossary_entries)
                            
                        except json.JSONDecodeError as e:
                            self.append_log(f"      ❌ Failed to parse JSON: {e}")
                            self.append_log(f"      Response preview: {glossary_json[:200]}...")
                            self.glossary_progress_manager.update(image_path, content_hash, status="error", error=str(e))
                            skipped += 1
                    else:
                        self.append_log(f"      ⚠️ No glossary data in response")
                        self.glossary_progress_manager.update(image_path, content_hash, status="error", error="No data")
                        skipped += 1
                    
                    # Add delay between API calls
                    if i < len(image_files) - 1 and not self.stop_requested:
                        self.append_log(f"      ⏱️ Waiting {api_delay}s before next image...")
                        elapsed = 0
                        while elapsed < api_delay and not self.stop_requested:
                            time.sleep(0.1)
                            elapsed += 0.1
                            
                except Exception as e:
                    self.append_log(f"      ❌ Failed to process: {str(e)}")
                    self.glossary_progress_manager.update(image_path, content_hash, status="error", error=str(e))
                    skipped += 1
            
            if not all_glossary_entries:
                self.append_log(f"❌ No glossary entries extracted from any images")
                return False
            
            self.append_log(f"\n📝 Extracted {len(all_glossary_entries)} total entries from {processed} images")
            
            # Save the final glossary with skip logic
            output_file = os.path.join("Glossary", f"{folder_name}_glossary.json")
            
            try:
                # Apply skip logic for duplicates
                self.append_log(f"📊 Applying skip logic for duplicate raw names...")
                
                # Import or define the skip function
                try:
                    from extract_glossary_from_epub import skip_duplicate_entries, remove_honorifics
                    # Set environment variable for honorifics toggle
                    import os
                    os.environ['GLOSSARY_DISABLE_HONORIFICS_FILTER'] = '1' if honorifics_disabled else '0'
                    final_entries = skip_duplicate_entries(all_glossary_entries)
                except:
                    # Fallback implementation
                    def remove_honorifics_local(name):
                        if not name or honorifics_disabled:
                            return name.strip()
                        
                        # Modern honorifics
                        korean_honorifics = ['님', '씨', '군', '양', '선생님', '사장님', '과장님', '대리님', '주임님', '이사님']
                        japanese_honorifics = ['さん', 'さま', '様', 'くん', '君', 'ちゃん', 'せんせい', '先生']
                        chinese_honorifics = ['先生', '女士', '小姐', '老师', '师傅', '大人']
                        
                        # Archaic honorifics
                        korean_archaic = ['공', '옹', '어른', '나리', '나으리', '대감', '영감', '마님', '마마']
                        japanese_archaic = ['どの', '殿', 'みこと', '命', '尊', 'ひめ', '姫']
                        chinese_archaic = ['公', '侯', '伯', '子', '男', '王', '君', '卿', '大夫']
                        
                        all_honorifics = (korean_honorifics + japanese_honorifics + chinese_honorifics + 
                                        korean_archaic + japanese_archaic + chinese_archaic)
                        
                        name_cleaned = name.strip()
                        sorted_honorifics = sorted(all_honorifics, key=len, reverse=True)
                        
                        for honorific in sorted_honorifics:
                            if name_cleaned.endswith(honorific):
                                name_cleaned = name_cleaned[:-len(honorific)].strip()
                                break
                        
                        return name_cleaned
                    
                    seen_raw_names = set()
                    final_entries = []
                    skipped = 0
                    
                    for entry in all_glossary_entries:
                        raw_name = entry.get('raw_name', '')
                        if not raw_name:
                            continue
                        
                        cleaned_name = remove_honorifics_local(raw_name)
                        
                        if cleaned_name.lower() in seen_raw_names:
                            skipped += 1
                            self.append_log(f"   ⏭️ Skipping duplicate: {raw_name}")
                            continue
                        
                        seen_raw_names.add(cleaned_name.lower())
                        final_entries.append(entry)
                    
                    self.append_log(f"✅ Kept {len(final_entries)} unique entries (skipped {skipped} duplicates)")
                
                # Save final glossary
                os.makedirs("Glossary", exist_ok=True)
                
                self.append_log(f"💾 Writing glossary to: {output_file}")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(final_entries, f, ensure_ascii=False, indent=2)
                
                # Also save as CSV for compatibility
                csv_file = output_file.replace('.json', '.csv')
                with open(csv_file, 'w', encoding='utf-8', newline='') as f:
                    import csv
                    writer = csv.writer(f)
                    # Write header
                    header = ['type', 'raw_name', 'translated_name', 'gender']
                    if custom_fields:
                        header.extend(custom_fields)
                    writer.writerow(header)
                    
                    for entry in final_entries:
                        row = [
                            entry.get('type', ''),
                            entry.get('raw_name', ''),
                            entry.get('translated_name', ''),
                            entry.get('gender', '') if entry.get('type') == 'character' else ''
                        ]
                        # Add custom field values
                        if custom_fields:
                            for field in custom_fields:
                                row.append(entry.get(field, ''))
                        writer.writerow(row)
                
                self.append_log(f"💾 Also saved as CSV: {os.path.basename(csv_file)}")
                
                # Verify files were created
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    self.append_log(f"✅ Glossary saved successfully ({file_size} bytes)")
                    
                    # Show sample of what was saved
                    if final_entries:
                        self.append_log(f"\n📋 Sample entries:")
                        for entry in final_entries[:5]:
                            self.append_log(f"   - [{entry['type']}] {entry['raw_name']} → {entry['translated_name']}")
                else:
                    self.append_log(f"❌ File was not created!")
                    return False
                
                return True
                
            except Exception as e:
                self.append_log(f"❌ Failed to save glossary: {e}")
                import traceback
                self.append_log(f"Full error: {traceback.format_exc()}")
                return False
                
        except Exception as e:
            self.append_log(f"❌ Error processing image folder: {str(e)}")
            import traceback
            self.append_log(f"❌ Full error: {traceback.format_exc()}")
            return False

    def _init_image_glossary_progress_manager(self, folder_name):
        """Initialize a folder-specific progress manager for image glossary extraction"""
        import hashlib
        
        class ImageGlossaryProgressManager:
            def __init__(self, folder_name):
                self.PROGRESS_FILE = os.path.join("Glossary", f"{folder_name}_glossary_progress.json")
                self.prog = self._init_or_load()
            
            def _init_or_load(self):
                """Initialize or load progress tracking"""
                if os.path.exists(self.PROGRESS_FILE):
                    try:
                        with open(self.PROGRESS_FILE, "r", encoding="utf-8") as pf:
                            return json.load(pf)
                    except Exception as e:
                        return {"images": {}, "content_hashes": {}, "extracted_data": {}, "version": "1.0"}
                else:
                    return {"images": {}, "content_hashes": {}, "extracted_data": {}, "version": "1.0"}
            
            def save(self):
                """Save progress to file atomically"""
                try:
                    os.makedirs(os.path.dirname(self.PROGRESS_FILE), exist_ok=True)
                    temp_file = self.PROGRESS_FILE + '.tmp'
                    with open(temp_file, "w", encoding="utf-8") as pf:
                        json.dump(self.prog, pf, ensure_ascii=False, indent=2)
                    
                    if os.path.exists(self.PROGRESS_FILE):
                        os.remove(self.PROGRESS_FILE)
                    os.rename(temp_file, self.PROGRESS_FILE)
                except Exception as e:
                    pass
            
            def get_content_hash(self, file_path):
                """Generate content hash for a file"""
                hasher = hashlib.sha256()
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
                return hasher.hexdigest()
            
            def check_image_status(self, image_path, content_hash):
                """Check if an image needs glossary extraction"""
                image_name = os.path.basename(image_path)
                
                # Check for skip markers
                skip_key = f"skip_{image_name}"
                if skip_key in self.prog:
                    skip_info = self.prog[skip_key]
                    if skip_info.get('status') == 'skipped':
                        return False, f"Image marked as skipped", None
                
                # Check if image has already been processed
                if content_hash in self.prog["images"]:
                    image_info = self.prog["images"][content_hash]
                    status = image_info.get("status")
                    
                    if status == "completed":
                        return False, f"Already processed", None
                    elif status == "skipped_cover":
                        return False, "Cover image - skipped", None
                    elif status == "error":
                        # Previous error, retry
                        return True, None, None
                
                return True, None, None
            
            def get_cached_result(self, content_hash):
                """Get cached extraction result for a content hash"""
                if content_hash in self.prog.get("extracted_data", {}):
                    return self.prog["extracted_data"][content_hash]
                return None
            
            def update(self, image_path, content_hash, status="in_progress", error=None, extracted_data=None):
                """Update progress for an image"""
                image_name = os.path.basename(image_path)
                
                image_info = {
                    "name": image_name,
                    "path": image_path,
                    "content_hash": content_hash,
                    "status": status,
                    "last_updated": time.time()
                }
                
                if error:
                    image_info["error"] = str(error)
                
                self.prog["images"][content_hash] = image_info
                
                # Store extracted data separately for reuse
                if extracted_data and status == "completed":
                    if "extracted_data" not in self.prog:
                        self.prog["extracted_data"] = {}
                    self.prog["extracted_data"][content_hash] = extracted_data
                
                self.save()
        
        # Create and return the progress manager
        progress_manager = ImageGlossaryProgressManager(folder_name)
        self.append_log(f"📊 Progress tracking in: Glossary/{folder_name}_glossary_progress.json")
        return progress_manager

    def _save_intermediate_glossary_with_skip(self, folder_name, entries):
        """Save intermediate glossary results with skip logic"""
        try:
            output_file = os.path.join("Glossary", f"{folder_name}_glossary.json")
            
            # Apply skip logic
            try:
                from extract_glossary_from_epub import skip_duplicate_entries
                unique_entries = skip_duplicate_entries(entries)
            except:
                # Fallback
                seen = set()
                unique_entries = []
                for entry in entries:
                    key = entry.get('raw_name', '').lower().strip()
                    if key and key not in seen:
                        seen.add(key)
                        unique_entries.append(entry)
            
            # Write the file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(unique_entries, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.append_log(f"      ⚠️ Could not save intermediate glossary: {e}")

    def _call_api_with_interrupt(self, client, messages, image_base64, temperature, max_tokens):
        """Make API call with interrupt support and thread safety"""
        import threading
        import queue
        from unified_api_client import UnifiedClientError
        
        result_queue = queue.Queue()
        
        def api_call():
            try:
                result = client.send_image(messages, image_base64, temperature=temperature, max_tokens=max_tokens)
                result_queue.put(('success', result))
            except Exception as e:
                result_queue.put(('error', e))
        
        api_thread = threading.Thread(target=api_call)
        api_thread.daemon = True
        api_thread.start()
        
        # Check for stop every 0.5 seconds
        while api_thread.is_alive():
            if self.stop_requested:
                # Cancel the operation
                if hasattr(client, 'cancel_current_operation'):
                    client.cancel_current_operation()
                raise UnifiedClientError("Glossary extraction stopped by user")
            
            try:
                status, result = result_queue.get(timeout=0.5)
                if status == 'error':
                    raise result
                return result
            except queue.Empty:
                continue
        
        # Thread finished, get final result
        try:
            status, result = result_queue.get(timeout=1.0)
            if status == 'error':
                raise result
            return result
        except queue.Empty:
            raise UnifiedClientError("API call completed but no result received")

    def _extract_glossary_from_text_file(self, file_path):
        """Extract glossary from EPUB or TXT file using existing glossary extraction"""
            # Skip glossary extraction for traditional APIs
        try:
            api_key = self.api_key_entry.text()
            model = self.model_var
            if is_traditional_translation_api(model):
               self.append_log("ℹ️ Skipping automatic glossary extraction (not supported by Google Translate / DeepL translation APIs)")
               return {}
            
            # Validate Vertex AI credentials if needed
            elif '@' in model or model.startswith('vertex/'):
                google_creds = self.config.get('google_cloud_credentials')
                if not google_creds or not os.path.exists(google_creds):
                    self.append_log("❌ Error: Google Cloud credentials required for Vertex AI models.")
                    return False
                
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_creds
                self.append_log(f"🔑 Using Google Cloud credentials: {os.path.basename(google_creds)}")
                
                if not api_key:
                    try:
                        with open(google_creds, 'r') as f:
                            creds_data = json.load(f)
                            api_key = creds_data.get('project_id', 'vertex-ai-project')
                            self.append_log(f"🔑 Using project ID as API key: {api_key}")
                    except:
                        api_key = 'vertex-ai-project'
            elif not api_key:
                self.append_log("❌ Error: Please enter your API key.")
                return False
            
            old_argv = sys.argv
            old_env = dict(os.environ)
            
            # Output file - do NOT prepend Glossary/ because extract_glossary_from_epub.py handles that
            epub_base = os.path.splitext(os.path.basename(file_path))[0]
            output_path = f"{epub_base}_glossary.json"
            
            try:
                # Set up environment variables
                env_updates = {
                    'GLOSSARY_TEMPERATURE': str(self.config.get('manual_glossary_temperature', 0.1)),
                    'GLOSSARY_CONTEXT_LIMIT': str(self.config.get('manual_context_limit', 2)),
                    'MODEL': self.model_var,
                    'OPENAI_API_KEY': api_key,
                    'OPENAI_OR_Gemini_API_KEY': api_key,
                    'API_KEY': api_key,
                    'MAX_OUTPUT_TOKENS': str(self.max_output_tokens),
                    'BATCH_TRANSLATION': "1" if self.batch_translation_var else "0",
                    'BATCH_SIZE': str(self.batch_size_var),
                    'GLOSSARY_SYSTEM_PROMPT': self.manual_glossary_prompt,
                    'CHAPTER_RANGE': self.chapter_range_entry.text().strip(),
                    'GLOSSARY_DISABLE_HONORIFICS_FILTER': '1' if self.config.get('glossary_disable_honorifics_filter', False) else '0',
                    'GLOSSARY_HISTORY_ROLLING': "1" if self.glossary_history_rolling_var else "0",
                    'DISABLE_GEMINI_SAFETY': str(self.config.get('disable_gemini_safety', False)).lower(),
                    'OPENROUTER_USE_HTTP_ONLY': '1' if self.openrouter_http_only_var else '0',
                    'GLOSSARY_DUPLICATE_KEY_MODE': 'skip',  # Always use skip mode for new format
                    'SEND_INTERVAL_SECONDS': str(self.delay_entry.text()),
                    'THREAD_SUBMISSION_DELAY_SECONDS': self.thread_delay_entry.text().strip() or '0.5',
                    'CONTEXTUAL': '1' if self.contextual_var else '0',
                    'GOOGLE_APPLICATION_CREDENTIALS': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''),
                    
                    # NEW GLOSSARY ADDITIONS
                    'GLOSSARY_MIN_FREQUENCY': str(self.glossary_min_frequency_var),
                    'GLOSSARY_MAX_NAMES': str(self.glossary_max_names_var),
                    'GLOSSARY_MAX_TITLES': str(self.glossary_max_titles_var),
                    'GLOSSARY_BATCH_SIZE': str(self.glossary_batch_size_var),
                    'ENABLE_AUTO_GLOSSARY': "1" if self.enable_auto_glossary_var else "0",
                    'APPEND_GLOSSARY': "1" if self.append_glossary_var else "0",
                    'GLOSSARY_STRIP_HONORIFICS': '1' if hasattr(self, 'strip_honorifics_var') and self.strip_honorifics_var else '1',
                    'AUTO_GLOSSARY_PROMPT': getattr(self, 'auto_glossary_prompt', ''),
                    'APPEND_GLOSSARY_PROMPT': getattr(self, 'append_glossary_prompt', '- Follow this reference glossary for consistent translation (Do not output any raw entries):\n'),
                    'GLOSSARY_TRANSLATION_PROMPT': getattr(self, 'glossary_translation_prompt', ''),
                    'GLOSSARY_CUSTOM_ENTRY_TYPES': json.dumps(getattr(self, 'custom_entry_types', {})),
                    'GLOSSARY_CUSTOM_FIELDS': json.dumps(getattr(self, 'custom_glossary_fields', [])),
                    'GLOSSARY_FUZZY_THRESHOLD': str(self.config.get('glossary_fuzzy_threshold', 0.90)),
                    'MANUAL_GLOSSARY': self.manual_glossary_path if hasattr(self, 'manual_glossary_path') and self.manual_glossary_path else '',
                    'GLOSSARY_FORMAT_INSTRUCTIONS': self.glossary_format_instructions if hasattr(self, 'glossary_format_instructions') else '',
                    'GLOSSARY_MAX_SENTENCES': str(self.config.get('glossary_max_sentences', 200)),
                    'GLOSSARY_MAX_TEXT_SIZE': str(self.config.get('glossary_max_text_size', 50000)),
                    'GLOSSARY_FILTER_MODE': self.config.get('glossary_filter_mode', 'all'),
                }
                
                # Add project ID for Vertex AI
                if '@' in model or model.startswith('vertex/'):
                    google_creds = self.config.get('google_cloud_credentials')
                    if google_creds and os.path.exists(google_creds):
                        try:
                            with open(google_creds, 'r') as f:
                                creds_data = json.load(f)
                                env_updates['GOOGLE_CLOUD_PROJECT'] = creds_data.get('project_id', '')
                                env_updates['VERTEX_AI_LOCATION'] = 'us-central1'
                        except:
                            pass
                
                if self.custom_glossary_fields:
                    env_updates['GLOSSARY_CUSTOM_FIELDS'] = json.dumps(self.custom_glossary_fields)
                
                # Propagate multi-key toggles so retry logic can engage
                # Both must be enabled for main-then-fallback retry
                try:
                    if self.config.get('use_multi_api_keys', False):
                        os.environ['USE_MULTI_KEYS'] = '1'
                    else:
                        os.environ['USE_MULTI_KEYS'] = '0'
                    if self.config.get('use_fallback_keys', False):
                        os.environ['USE_FALLBACK_KEYS'] = '1'
                    else:
                        os.environ['USE_FALLBACK_KEYS'] = '0'
                except Exception:
                    # Keep going even if we can't set env for some reason
                    pass

                os.environ.update(env_updates)
                
                chap_range = self.chapter_range_entry.text().strip()
                if chap_range:
                    self.append_log(f"📊 Chapter Range: {chap_range} (glossary extraction will only process these chapters)")
                
                if self.token_limit_disabled:
                    os.environ['MAX_INPUT_TOKENS'] = ''
                    self.append_log("🎯 Input Token Limit: Unlimited (disabled)")
                else:
                    token_val = self.token_limit_entry.text().strip()
                    if token_val and token_val.isdigit():
                        os.environ['MAX_INPUT_TOKENS'] = token_val
                        self.append_log(f"🎯 Input Token Limit: {token_val}")
                    else:
                        os.environ['MAX_INPUT_TOKENS'] = '50000'
                        self.append_log(f"🎯 Input Token Limit: 50000 (default)")
                
                sys.argv = [
                    'extract_glossary_from_epub.py',
                    '--epub', file_path,
                    '--output', output_path,
                    '--config', CONFIG_FILE
                ]
                
                self.append_log(f"🚀 Extracting glossary from: {os.path.basename(file_path)}")
                self.append_log(f"📤 Output Token Limit: {self.max_output_tokens}")
                format_parts = ["type", "raw_name", "translated_name", "gender"]
                custom_fields_json = self.config.get('manual_custom_fields', '[]')
                try:
                    custom_fields = json.loads(custom_fields_json) if isinstance(custom_fields_json, str) else custom_fields_json
                    if custom_fields:
                        format_parts.extend(custom_fields)
                except:
                    custom_fields = []
                self.append_log(f"   Format: Simple ({', '.join(format_parts)})")
                
                # Check honorifics filter
                if self.config.get('glossary_disable_honorifics_filter', False):
                    self.append_log(f"📑 Honorifics Filter: ❌ DISABLED")
                else:
                    self.append_log(f"📑 Honorifics Filter: ✅ ENABLED")
                
                os.environ['MAX_OUTPUT_TOKENS'] = str(self.max_output_tokens)
                
                # Enhanced stop callback that checks both flags
                def enhanced_stop_callback():
                    # Check GUI stop flag
                    if self.stop_requested:
                        return True
                        
                    # Also check if the glossary extraction module has its own stop flag
                    try:
                        import extract_glossary_from_epub
                        if hasattr(extract_glossary_from_epub, 'is_stop_requested') and extract_glossary_from_epub.is_stop_requested():
                            return True
                    except:
                        pass
                        
                    return False

                try:
                    # Import traceback for better error info
                    import traceback
                    
                    # Run glossary extraction with enhanced stop callback
                    glossary_main(
                        log_callback=self.append_log,
                        stop_callback=enhanced_stop_callback
                    )
                except Exception as e:
                    # Get the full traceback
                    tb_lines = traceback.format_exc()
                    self.append_log(f"❌ FULL ERROR TRACEBACK:\n{tb_lines}")
                    self.append_log(f"❌ Error extracting glossary from {os.path.basename(file_path)}: {e}")
                    return False
                
                # Check if stopped
                if self.stop_requested:
                    self.append_log("⏹️ Glossary extraction was stopped")
                    return False
                
                # Check if output file exists
                if not self.stop_requested and os.path.exists(output_path):
                    self.append_log(f"✅ Glossary saved to: {output_path}")
                    return True
                else:
                    # Check if it was saved in Glossary folder by the script
                    glossary_path = os.path.join("Glossary", output_path)
                    if os.path.exists(glossary_path):
                        self.append_log(f"✅ Glossary saved to: {glossary_path}")
                        return True
                    return False
                
            finally:
                sys.argv = old_argv
                os.environ.clear()
                os.environ.update(old_env)
                
        except Exception as e:
            self.append_log(f"❌ Error extracting glossary from {os.path.basename(file_path)}: {e}")
            return False
        
    def epub_converter(self):
       """Start EPUB converter in a separate thread"""
       if not self._lazy_load_modules():
           self.append_log("❌ Failed to load EPUB converter modules")
           return
       
       if fallback_compile_epub is None:
           self.append_log("❌ EPUB converter module is not available")
           QMessageBox.critical(self, "Module Error", "EPUB converter module is not available.")
           return
       
       if hasattr(self, 'translation_thread') and self.translation_thread and self.translation_thread.is_alive():
           self.append_log("⚠️ Cannot run EPUB converter while translation is in progress.")
           QMessageBox.warning(self, "Process Running", "Please wait for translation to complete before converting EPUB.")
           return
       
       if hasattr(self, 'glossary_thread') and self.glossary_thread and self.glossary_thread.is_alive():
           self.append_log("⚠️ Cannot run EPUB converter while glossary extraction is in progress.")
           QMessageBox.warning(self, "Process Running", "Please wait for glossary extraction to complete before converting EPUB.")
           return
       
       if hasattr(self, 'epub_thread') and self.epub_thread and self.epub_thread.is_alive():
           self.stop_epub_converter()
           return
       
       folder = QFileDialog.getExistingDirectory(self, "Select translation output folder")
       if not folder:
           return
       
       self.epub_folder = folder
       self.stop_requested = False
       # Run via shared executor
       self._ensure_executor()
       if self.executor:
           self.epub_future = self.executor.submit(self.run_epub_converter_direct)
           # Ensure button state is refreshed when the future completes (button update handled by thread_complete_signal in finally block)
           def _epub_done_callback(f):
               try:
                   self.epub_future = None
               except Exception:
                   pass
           try:
               self.epub_future.add_done_callback(_epub_done_callback)
           except Exception:
               pass
       else:
           self.epub_thread = threading.Thread(target=self.run_epub_converter_direct, daemon=True)
           self.epub_thread.start()
       
       # Update button IMMEDIATELY after starting thread (synchronous)
       self.update_run_button()
 
    def run_epub_converter_direct(self):
        """Run EPUB converter directly without blocking GUI"""
        try:
            folder = self.epub_folder
            self.append_log("📦 Starting EPUB Converter...")
            
            # Set environment variables for EPUB converter
            os.environ['DISABLE_EPUB_GALLERY'] = "1" if self.disable_epub_gallery_var else "0"
            os.environ['DISABLE_AUTOMATIC_COVER_CREATION'] = "1" if getattr(self, 'disable_automatic_cover_creation_var', False) else "0"
            os.environ['TRANSLATE_COVER_HTML'] = "1" if getattr(self, 'translate_cover_html_var', False) else "0"

            source_epub_file = os.path.join(folder, 'source_epub.txt')
            if os.path.exists(source_epub_file):
                try:
                    with open(source_epub_file, 'r', encoding='utf-8') as f:
                        source_epub_path = f.read().strip()
                        
                    if source_epub_path and os.path.exists(source_epub_path):
                        os.environ['EPUB_PATH'] = source_epub_path
                        self.append_log(f"✅ Using source EPUB for proper chapter ordering: {os.path.basename(source_epub_path)}")
                    else:
                        self.append_log(f"⚠️ Source EPUB file not found: {source_epub_path}")
                except Exception as e:
                    self.append_log(f"⚠️ Could not read source EPUB reference: {e}")
            else:
                self.append_log("ℹ️ No source EPUB reference found - using filename-based ordering")
            
            # Set API credentials and model
            api_key = self.api_key_entry.text()
            if api_key:
                os.environ['API_KEY'] = api_key
                os.environ['OPENAI_API_KEY'] = api_key
                os.environ['OPENAI_OR_Gemini_API_KEY'] = api_key
            
            model = self.model_var
            if model:
                os.environ['MODEL'] = model
            
            # Set translation parameters from GUI
            os.environ['TRANSLATION_TEMPERATURE'] = str(self.trans_temp.text())
            os.environ['MAX_OUTPUT_TOKENS'] = str(self.max_output_tokens)
            
            # Set batch translation settings
            os.environ['BATCH_TRANSLATE_HEADERS'] = "1" if self.batch_translate_headers_var else "0"
            os.environ['HEADERS_PER_BATCH'] = str(self.headers_per_batch_var)
            os.environ['UPDATE_HTML_HEADERS'] = "1" if self.update_html_headers_var else "0"
            os.environ['SAVE_HEADER_TRANSLATIONS'] = "1" if self.save_header_translations_var else "0"
            
            # Set metadata translation settings
            os.environ['TRANSLATE_METADATA_FIELDS'] = json.dumps(self.translate_metadata_fields)
            os.environ['METADATA_TRANSLATION_MODE'] = self.config.get('metadata_translation_mode', 'together')
            print(f"[DEBUG] METADATA_FIELD_PROMPTS from env: {os.getenv('METADATA_FIELD_PROMPTS', 'NOT SET')[:100]}...")

            # Debug: Log what we're setting
            self.append_log(f"[DEBUG] Setting TRANSLATE_METADATA_FIELDS: {self.translate_metadata_fields}")
            self.append_log(f"[DEBUG] Enabled fields: {[k for k, v in self.translate_metadata_fields.items() if v]}")
            
            # Set book title translation settings
            os.environ['TRANSLATE_BOOK_TITLE'] = "1" if self.translate_book_title_var else "0"
            os.environ['BOOK_TITLE_PROMPT'] = self.book_title_prompt
            os.environ['BOOK_TITLE_SYSTEM_PROMPT'] = self.config.get('book_title_system_prompt', 
                "You are a translator. Respond with only the translated text, nothing else.")
            
            # Set prompts
            os.environ['SYSTEM_PROMPT'] = self.prompt_text.toPlainText().strip()
            
            fallback_compile_epub(folder, log_callback=self.append_log)
            
            if not self.stop_requested:
                self.append_log("✅ EPUB Converter completed successfully!")
                
                epub_files = [f for f in os.listdir(folder) if f.endswith('.epub')]
                if epub_files:
                    epub_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True)
                    out_file = os.path.join(folder, epub_files[0])
                    QTimer.singleShot(0, lambda: QMessageBox.information(self, "EPUB Compilation Success", f"Created: {out_file}"))
                else:
                    self.append_log("⚠️ EPUB file was not created. Check the logs for details.")
            
        except Exception as e:
            error_str = str(e)
            self.append_log(f"❌ EPUB Converter error: {error_str}")
            
            if "Document is empty" not in error_str:
                QTimer.singleShot(0, lambda: QMessageBox.critical(self, "EPUB Converter Failed", f"Error: {error_str}"))
            else:
                self.append_log("📋 Check the log above for details about what went wrong.")
        
        finally:
            # Always reset the thread and update button state when done
            self.epub_thread = None
            # Clear any future handle so update_run_button won't consider it running
            if hasattr(self, 'epub_future'):
                try:
                    # Don't cancel; just drop the reference. Future is already done here.
                    self.epub_future = None
                except Exception:
                    pass
            self.stop_requested = False
            # Emit signal to update button (thread-safe)
            self.thread_complete_signal.emit()
        
    def toggle_token_limit(self):
       """Toggle whether the token-limit entry is active or not."""
       if not self.token_limit_disabled:
           self.token_limit_entry.setEnabled(False)
           self.toggle_token_btn.setText("Enable Input Token Limit")
           self.toggle_token_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
           self.append_log("⚠️ Input token limit disabled - both translation and glossary extraction will process chapters of any size.")
           self.token_limit_disabled = True
       else:
           self.token_limit_entry.setEnabled(True)
           if not self.token_limit_entry.text().strip():
               self.token_limit_entry.setText(str(self.config.get('token_limit', 1000000)))
           self.toggle_token_btn.setText("Disable Input Token Limit")
           self.toggle_token_btn.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold;")
           self.append_log(f"✅ Input token limit enabled: {self.token_limit_entry.text()} tokens (applies to both translation and glossary extraction)")
           self.token_limit_disabled = False

    def update_run_button(self):
       """Switch Run↔Stop depending on whether a process is active."""
       translation_running = (
           (hasattr(self, 'translation_thread') and self.translation_thread and self.translation_thread.is_alive()) or
           (hasattr(self, 'translation_future') and self.translation_future and not self.translation_future.done())
       )
       glossary_running = (
           (hasattr(self, 'glossary_thread') and self.glossary_thread and self.glossary_thread.is_alive()) or
           (hasattr(self, 'glossary_future') and self.glossary_future and not self.glossary_future.done())
       )
       qa_running = (
           (hasattr(self, 'qa_thread') and self.qa_thread and self.qa_thread.is_alive()) or
           (hasattr(self, 'qa_future') and self.qa_future and not self.qa_future.done())
       )
       epub_running = (
           (hasattr(self, 'epub_thread') and self.epub_thread and self.epub_thread.is_alive()) or
           (hasattr(self, 'epub_future') and self.epub_future and not self.epub_future.done())
       )
       
       any_process_running = translation_running or glossary_running or qa_running or epub_running
       
       # Translation button
       try:
           self.run_button.clicked.disconnect()
       except:
           pass
       
       if translation_running:
           if hasattr(self, 'run_button_text'):
               self.run_button_text.setText("Stop Translation")
           self.run_button.setStyleSheet("""
               QPushButton {
                   background-color: #dc3545;
                   border: none;
               }
               QPushButton:hover {
                   background-color: #c82333;
               }
           """)
           self.run_button.clicked.connect(self.stop_translation)
           self.run_button.setEnabled(True)
           # Start spinning animation
           if hasattr(self, 'icon_spin_animation') and hasattr(self, 'run_button_icon'):
               if self.icon_spin_animation.state() != QPropertyAnimation.Running:
                   self.icon_spin_animation.start()
       else:
           if hasattr(self, 'run_button_text'):
               self.run_button_text.setText("Run Translation")
           self.run_button.setStyleSheet("""
               QPushButton {
                   background-color: #28a745;
                   border: none;
               }
               QPushButton:disabled {
                   background-color: #555555;
               }
               QPushButton:hover {
                   background-color: #218838;
               }
           """)
           self.run_button.clicked.connect(self.run_translation_thread)
           self.run_button.setEnabled(translation_main and not any_process_running)
           # Stop spinning animation gracefully with deceleration
           if hasattr(self, 'icon_spin_animation') and hasattr(self, 'run_button_icon') and hasattr(self, 'icon_stop_animation'):
               if self.icon_spin_animation.state() == QPropertyAnimation.Running:
                   # Stop the infinite spin animation
                   self.icon_spin_animation.stop()
                   
                   # Get current rotation angle
                   current_rotation = self.run_button_icon.get_rotation()
                   
                   # Calculate the shortest path to 0 degrees
                   # Normalize to 0-360 range
                   current_rotation = current_rotation % 360
                   
                   # Determine if we should go forward or backward to reach 0
                   if current_rotation > 180:
                       # Go forward (e.g., 270 -> 360 -> 0)
                       target_rotation = 360
                   else:
                       # Go backward (e.g., 90 -> 0)
                       target_rotation = 0
                   
                   # Set up smooth deceleration animation
                   self.icon_stop_animation.setStartValue(current_rotation)
                   self.icon_stop_animation.setEndValue(target_rotation)
                   self.icon_stop_animation.start()
               elif self.icon_stop_animation.state() != QPropertyAnimation.Running:
                   # If no animation is running, just reset to 0
                   self.run_button_icon.set_rotation(0)
       
       # Glossary button
       if hasattr(self, 'glossary_button'):
           try:
               self.glossary_button.clicked.disconnect()
           except:
               pass
           
           if glossary_running:
               # Update text label instead of button text
               if hasattr(self, 'glossary_text_label'):
                   self.glossary_text_label.setText("Stop Glossary")
               self.glossary_button.setStyleSheet("""
                   QPushButton {
                       background-color: #dc3545;
                       color: white;
                       padding: 6px;
                   }
               """)
               self.glossary_button.clicked.connect(self.stop_glossary_extraction)
               self.glossary_button.setEnabled(True)
               # Start spinning animation for glossary icon
               if hasattr(self, 'glossary_icon_spin_animation') and hasattr(self, 'glossary_button_icon'):
                   if self.glossary_icon_spin_animation.state() != QPropertyAnimation.Running:
                       self.glossary_icon_spin_animation.start()
           else:
               # Update text label instead of button text
               if hasattr(self, 'glossary_text_label'):
                   self.glossary_text_label.setText("Extract Glossary")
               self.glossary_button.setStyleSheet("""
                   QPushButton {
                       background-color: #e67e22;
                       color: white;
                       padding: 6px;
                       font-weight: bold;
                   }
                   QPushButton:disabled {
                       background-color: #555555;
                       color: #888888;
                   }
               """)
               self.glossary_button.clicked.connect(self.run_glossary_extraction_thread)
               self.glossary_button.setEnabled(glossary_main and not any_process_running)
               # Stop spinning animation gracefully for glossary icon
               if hasattr(self, 'glossary_icon_spin_animation') and hasattr(self, 'glossary_button_icon') and hasattr(self, 'glossary_icon_stop_animation'):
                   if self.glossary_icon_spin_animation.state() == QPropertyAnimation.Running:
                       self.glossary_icon_spin_animation.stop()
                       current_rotation = self.glossary_button_icon.get_rotation()
                       current_rotation = current_rotation % 360
                       if current_rotation > 180:
                           target_rotation = 360
                       else:
                           target_rotation = 0
                       self.glossary_icon_stop_animation.setStartValue(current_rotation)
                       self.glossary_icon_stop_animation.setEndValue(target_rotation)
                       self.glossary_icon_stop_animation.start()
                   elif self.glossary_icon_stop_animation.state() != QPropertyAnimation.Running:
                       self.glossary_button_icon.set_rotation(0)
    
       # EPUB button
       if hasattr(self, 'epub_button'):
           try:
               self.epub_button.clicked.disconnect()
           except:
               pass
           
           if epub_running:
               self.epub_button.setText("Stop EPUB")
               self.epub_button.setStyleSheet("background-color: #dc3545; color: white; padding: 6px;")  # red
               self.epub_button.clicked.connect(self.stop_epub_converter)
               self.epub_button.setEnabled(True)
           else:
               self.epub_button.setText("EPUB Converter")
               self.epub_button.setStyleSheet("background-color: #17a2b8; color: white; padding: 6px;")  # info blue
               self.epub_button.clicked.connect(self.epub_converter)
               self.epub_button.setEnabled(fallback_compile_epub and not any_process_running)
       
       # QA button
       if hasattr(self, 'qa_button'):
           try:
               self.qa_button.clicked.disconnect()
           except:
               pass
           
           if qa_running:
               # Update text label instead of button text
               if hasattr(self, 'qa_text_label'):
                   self.qa_text_label.setText("Stop Scan")
               self.qa_button.setStyleSheet("""
                   QPushButton {
                       background-color: #dc3545;
                       color: white;
                       padding: 6px;
                   }
               """)
               self.qa_button.clicked.connect(self.stop_qa_scan)
               self.qa_button.setEnabled(True)
               # Start spinning animation for QA icon
               if hasattr(self, 'qa_icon_spin_animation') and hasattr(self, 'qa_button_icon'):
                   if self.qa_icon_spin_animation.state() != QPropertyAnimation.Running:
                       self.qa_icon_spin_animation.start()
           else:
               # Update text label instead of button text
               if hasattr(self, 'qa_text_label'):
                   self.qa_text_label.setText("QA Scan")
               self.qa_button.setStyleSheet("""
                   QPushButton {
                       background-color: #e67e22;
                       color: white;
                       padding: 6px;
                       font-weight: bold;
                   }
                   QPushButton:disabled {
                       background-color: #555555;
                       color: #888888;
                   }
               """)
               self.qa_button.clicked.connect(self.run_qa_scan)
               self.qa_button.setEnabled(scan_html_folder and not any_process_running)
               # Stop spinning animation gracefully for QA icon
               if hasattr(self, 'qa_icon_spin_animation') and hasattr(self, 'qa_button_icon') and hasattr(self, 'qa_icon_stop_animation'):
                   if self.qa_icon_spin_animation.state() == QPropertyAnimation.Running:
                       self.qa_icon_spin_animation.stop()
                       current_rotation = self.qa_button_icon.get_rotation()
                       current_rotation = current_rotation % 360
                       if current_rotation > 180:
                           target_rotation = 360
                       else:
                           target_rotation = 0
                       self.qa_icon_stop_animation.setStartValue(current_rotation)
                       self.qa_icon_stop_animation.setEndValue(target_rotation)
                       self.qa_icon_stop_animation.start()
                   elif self.qa_icon_stop_animation.state() != QPropertyAnimation.Running:
                       self.qa_button_icon.set_rotation(0)

    def stop_translation(self):
        """Stop translation while preserving loaded file"""
        current_file = self.entry_epub.text() if hasattr(self, 'entry_epub') else None
        
        # Disable button immediately to prevent multiple clicks
        if hasattr(self, 'run_button'):
            self.run_button.setEnabled(False)
            if hasattr(self, 'run_button_text'):
                self.run_button_text.setText("Stopping...")
            self.run_button.setStyleSheet("QPushButton { background-color: #6c757d; border: none; }")
        
        # Set environment variable to suppress multi-key logging
        os.environ['TRANSLATION_CANCELLED'] = '1'
        
        self.stop_requested = True
        
        # Use the imported translation_stop_flag function from TransateKRtoEN
        # This was imported during lazy loading as: translation_stop_flag = TransateKRtoEN.set_stop_flag
        if 'translation_stop_flag' in globals() and translation_stop_flag:
            translation_stop_flag(True)
        
        # Also try to call it directly on the module if imported
        try:
            import TransateKRtoEN
            if hasattr(TransateKRtoEN, 'set_stop_flag'):
                TransateKRtoEN.set_stop_flag(True)
        except: 
            pass
        
        try:
            import unified_api_client
            if hasattr(unified_api_client, 'set_stop_flag'):
                unified_api_client.set_stop_flag(True)
            # If there's a global client instance, stop it too
            if hasattr(unified_api_client, 'global_stop_flag'):
                unified_api_client.global_stop_flag = True
            
            # Set the _cancelled flag on the UnifiedClient class itself
            if hasattr(unified_api_client, 'UnifiedClient'):
                unified_api_client.UnifiedClient._global_cancelled = True
                
        except Exception as e:
            print(f"Error setting stop flags: {e}")
        
        # Save and encrypt config when stopping
        try:
            self.save_config(show_message=False)
        except:
            pass
        
        # Single concise stop message
        self.append_log("🛑 Stop requested — waiting for current operation to finish")
        # Don't call update_run_button() here - keep the "Stopping..." state until thread finishes
        
        if current_file and hasattr(self, 'entry_epub'):
            QTimer.singleShot(100, lambda: self.preserve_file_path(current_file))

    def preserve_file_path(self, file_path):
       """Helper to ensure file path stays in the entry field"""
       if hasattr(self, 'entry_epub') and file_path:
           current = self.entry_epub.text()
           if not current or current != file_path:
               self.entry_epub.setText(file_path)
    
    def _trigger_qa_scan_on_main_thread(self):
        """Handler called on main thread to trigger QA scan"""
        try:
            mode = self.scan_phase_mode_var if hasattr(self, 'scan_phase_mode_var') else 'quick-scan'
            # Call run_qa_scan directly on the main thread with correct parameters
            self.run_qa_scan(mode_override=mode, non_interactive=True)
        except Exception as e:
            self.append_log(f"❌ Failed to start QA scan: {e}")
            import traceback
            self.append_log(traceback.format_exc())

    def stop_glossary_extraction(self):
       """Stop glossary extraction specifically"""
       # Disable button immediately to prevent multiple clicks
       if hasattr(self, 'glossary_button'):
           self.glossary_button.setEnabled(False)
           # Update text label instead of button text
           if hasattr(self, 'glossary_text_label'):
               self.glossary_text_label.setText("Stopping...")
           self.glossary_button.setStyleSheet("background-color: #6c757d; color: white; padding: 6px;")
       
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
       # Don't call update_run_button() here - keep the "Stopping..." state until thread finishes


    def stop_epub_converter(self):
        """Stop EPUB converter"""
        # Disable button immediately to prevent multiple clicks
        if hasattr(self, 'epub_button'):
            self.epub_button.setEnabled(False)
            self.epub_button.setText("Stopping...")
            self.epub_button.setStyleSheet("background-color: #6c757d; color: white; padding: 6px;")
        
        self.stop_requested = True
        self.append_log("❌ EPUB converter stop requested.")
        self.append_log("⏳ Please wait... stopping after current operation completes.")
        # Don't call update_run_button() here - keep the "Stopping..." state until thread finishes

    def stop_qa_scan(self):
        # Disable button immediately to prevent multiple clicks
        if hasattr(self, 'qa_button'):
            self.qa_button.setEnabled(False)
            # Update text label instead of button text
            if hasattr(self, 'qa_text_label'):
                self.qa_text_label.setText("Stopping...")
            self.qa_button.setStyleSheet("background-color: #6c757d; color: white; padding: 6px;")
        
        self.stop_requested = True
        try:
            from scan_html_folder import stop_scan
            if stop_scan():
                self.append_log("✅ Stop scan signal sent successfully")
        except Exception as e:
            self.append_log(f"❌ Failed to stop scan: {e}")
        self.append_log("⛔ QA scan stop requested.")
        self.append_log("⏳ Please wait... stopping after current operation completes.")
        # Don't call update_run_button() here - keep the "Stopping..." state until thread finishes
       

    def on_close(self):
        reply = QMessageBox.question(self, "Quit", "Are you sure you want to exit?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.stop_requested = True
            
            # Save and encrypt config before closing
            try:
                self.save_config(show_message=False)
            except:
                pass  # Don't prevent closing if save fails
            
            # Shutdown the executor to stop accepting new tasks
            try:
                if getattr(self, 'executor', None):
                    self.executor.shutdown(wait=False)
            except Exception:
                pass
            
            self.close()
            sys.exit(0)

    def _on_log_scroll(self, value):
        """Detect when user manually scrolls up in the log"""
        try:
            scrollbar = self.log_text.verticalScrollBar()
            # If user scrolled up (not at bottom), mark it
            at_bottom = value >= scrollbar.maximum() - 10
            
            # Only mark as user scrolled if we were previously at bottom and now we're not
            # This prevents false positives when content is added and scrollbar max changes
            was_at_bottom = getattr(self, '_was_at_bottom', True)
            
            if not at_bottom and was_at_bottom:
                # User intentionally scrolled up from the bottom
                self._user_scrolled_up = True
            elif at_bottom:
                # User scrolled back to bottom, resume auto-scroll
                self._user_scrolled_up = False
            
            # Track current state for next comparison
            self._was_at_bottom = at_bottom
        except Exception:
            pass
    
    def _start_autoscroll_delay(self, ms=0):
        try:
            import time as _time
            self._autoscroll_delay_until = _time.time() + (ms / 1000.0)
            # Reset manual scroll flag when starting new operation
            self._user_scrolled_up = False
        except Exception:
            self._autoscroll_delay_until = 0.0
    
    def append_log_direct(self, message):
        """Direct append - MUST be called from main thread only"""
        try:
            if not hasattr(self, 'log_text') or not self.log_text:
                return
            
            try:
                _ = self.log_text.document()
            except RuntimeError:
                return
            
            # Use textCursor for more compact logging (no extra spacing)
            cursor = self.log_text.textCursor()
            cursor.movePosition(QTextCursor.End)
            
            # Add newline if not first message
            if not cursor.atStart():
                cursor.insertText("\n")
            
            cursor.insertText(message)
            
            # Scroll to bottom (respect delay and manual scrolling)
            try:
                import time as _time
                # Only auto-scroll if delay passed AND user hasn't scrolled up
                if (_time.time() >= getattr(self, '_autoscroll_delay_until', 0) and 
                    not getattr(self, '_user_scrolled_up', False)):
                    scrollbar = self.log_text.verticalScrollBar()
                    scrollbar.setValue(scrollbar.maximum())
            except Exception:
                pass
        except Exception as e:
            pass  # Silent failure
    
    def append_log(self, message):
       """Append message to log with safety checks (fallback to print if GUI is gone).
       Also suppresses repeated stop/cancel notices once a stop has been requested.
       """
       def _append():
           try:
               # Stop-notice suppression: if user has requested stop, allow only one concise notice
               try:
                   if getattr(self, 'stop_requested', False):
                       msg_low = str(message).lower()
                       stop_keys = ['stop requested', 'stopped by user', 'operation cancelled', 'cancelled', 'stopping after current']
                       if any(k in msg_low for k in stop_keys):
                           if getattr(self, '_stop_notice_shown', False):
                               return
                           else:
                               self._stop_notice_shown = True
               except Exception:
                   pass
               # Bail out if the widget no longer exists
               if not hasattr(self, 'log_text'):
                   print(message)
                   return
               try:
                   # Check if widget still exists and is visible
                   # QTextEdit may not be visible initially but should still accept text
                   if not self.log_text:
                       exists = False
                   else:
                       # Try to access the widget - if it throws RuntimeError, it's been deleted
                       try:
                           _ = self.log_text.document()
                           exists = True
                       except RuntimeError:
                           exists = False
               except Exception:
                   exists = False
               if not exists:
                   print(message)
                   return
               
               at_bottom = False
               try:
                   # Get scrollbar and check if at bottom
                   scrollbar = self.log_text.verticalScrollBar()
                   at_bottom = scrollbar.value() >= scrollbar.maximum() - 10
               except Exception:
                   at_bottom = True  # Default to scrolling to bottom
               
               is_memory = any(keyword in message for keyword in ['[MEMORY]', '📝', 'rolling summary', 'memory'])
               
               if is_memory:
                   # Apply green color formatting for memory messages
                   cursor = self.log_text.textCursor()
                   cursor.movePosition(QTextCursor.End)
                   
                   # Set format for memory text
                   format = QTextCharFormat()
                   format.setForeground(QColor("#4CAF50"))
                   font = QFont()
                   font.setItalic(True)
                   format.setFont(font)
                   
                   cursor.insertText(message + "\n", format)
               else:
                   # Regular text append
                   self.log_text.append(message)
               
               # Try to scroll to bottom to ensure visibility, but respect delayed auto-scroll window and manual scrolling
               try:
                   import time as _time
                   # Only auto-scroll if delay passed AND user hasn't scrolled up
                   if (_time.time() >= getattr(self, '_autoscroll_delay_until', 0) and 
                       not getattr(self, '_user_scrolled_up', False)):
                       scrollbar = self.log_text.verticalScrollBar()
                       if at_bottom or True:
                           scrollbar.setValue(scrollbar.maximum())
                   # Force immediate update of the widget
                   self.log_text.update()
                   self.log_text.repaint()
               except Exception:
                   pass
           except Exception as e:
               # As a last resort, print to stdout to avoid crashing callbacks
               try:
                   print(f"{message} [append_log error: {e}]")
               except Exception:
                   pass
       
       if threading.current_thread() is threading.main_thread():
           _append()
       else:
           # Use Qt Signal for thread-safe logging
           try:
               self.log_signal.emit(message)
           except Exception:
               pass  # Silent failure

    def update_status_line(self, message, progress_percent=None):
       """Update a status line in the log safely (fallback to print)."""
       def _update():
           try:
               if not hasattr(self, 'log_text'):
                   print(message)
                   return
               try:
                   if not self.log_text.isVisible() and not self.log_text.parent():
                       print(message)
                       return
               except:
                   print(message)
                   return
                   
               content = self.log_text.toPlainText()
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
                   # Remove the last line and replace with new status
                   cursor = self.log_text.textCursor()
                   cursor.movePosition(QTextCursor.End)
                   cursor.select(QTextCursor.LineUnderCursor)
                   cursor.removeSelectedText()
                   cursor.deletePreviousChar()  # Remove the newline
                   
                   if len(lines) > 1:
                       self.log_text.append(status_msg)
                   else:
                       cursor.insertText(status_msg)
               else:
                   if content and not content.endswith('\n'):
                       self.log_text.append(status_msg)
                   else:
                       self.log_text.append(status_msg)
               
               # Scroll to bottom
               scrollbar = self.log_text.verticalScrollBar()
               scrollbar.setValue(scrollbar.maximum())
           except Exception:
               try:
                   print(message)
               except Exception:
                   pass
       
       if threading.current_thread() is threading.main_thread():
           _update()
       else:
           try:
               QTimer.singleShot(0, _update)
           except Exception:
               try:
                   print(message)
               except Exception:
                   pass

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

    def _show_context_menu(self, pos):
       """Show context menu for log text"""
       context_menu = QMenu(self)
       
       # Check if there's selected text
       cursor = self.log_text.textCursor()
       has_selection = cursor.hasSelection()
       
       copy_action = context_menu.addAction("Copy")
       copy_action.triggered.connect(self.copy_selection)
       copy_action.setEnabled(has_selection)
       
       context_menu.addSeparator()
       
       select_all_action = context_menu.addAction("Select All")
       select_all_action.triggered.connect(self.select_all_log)
       
       # Show the context menu at the cursor position
       context_menu.exec(self.log_text.mapToGlobal(pos))

    def copy_selection(self):
       """Copy selected text from log to clipboard"""
       cursor = self.log_text.textCursor()
       if cursor.hasSelection():
           selected_text = cursor.selectedText()
           clipboard = QApplication.clipboard()
           clipboard.setText(selected_text)

    def select_all_log(self):
       """Select all text in the log"""
       self.log_text.selectAll()

    def auto_load_glossary_for_file(self, file_path):
        """Automatically load glossary if it exists in the output folder"""
        
        # CHECK FOR EPUB FIRST - before any clearing logic!
        if not file_path or not os.path.isfile(file_path):
            return
        
        if not file_path.lower().endswith('.epub'):
            return  # Exit early for non-EPUB files - don't touch glossaries!
        
        # Clear previous auto-loaded glossary if switching EPUB files
        if file_path != self.auto_loaded_glossary_for_file:
            # Only clear if the current glossary was auto-loaded AND not manually loaded
            if (self.auto_loaded_glossary_path and 
                self.manual_glossary_path == self.auto_loaded_glossary_path and
                not getattr(self, 'manual_glossary_manually_loaded', False)):  # Check manual flag
                self.manual_glossary_path = None
                self.append_log("📑 Cleared auto-loaded glossary from previous novel")
            
            self.auto_loaded_glossary_path = None
            self.auto_loaded_glossary_for_file = None
        
        # Don't override manually loaded glossaries
        if getattr(self, 'manual_glossary_manually_loaded', False) and self.manual_glossary_path:
            self.append_log(f"📑 Keeping manually loaded glossary: {os.path.basename(self.manual_glossary_path)}")
            return
        
        file_base = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = file_base
        
        # Prefer CSV over JSON when both exist
        glossary_candidates = [
            os.path.join(output_dir, "glossary.csv"),
            os.path.join(output_dir, f"{file_base}_glossary.csv"),
            os.path.join(output_dir, "Glossary", f"{file_base}_glossary.csv"),
            os.path.join(output_dir, "glossary.json"),
            os.path.join(output_dir, f"{file_base}_glossary.json"),
            os.path.join(output_dir, "Glossary", f"{file_base}_glossary.json")
        ]
        for glossary_path in glossary_candidates:
            if os.path.exists(glossary_path):
                ext = os.path.splitext(glossary_path)[1].lower()
                try:
                    if ext == '.csv':
                        # Accept CSV without parsing
                        self.manual_glossary_path = glossary_path
                        self.auto_loaded_glossary_path = glossary_path
                        self.auto_loaded_glossary_for_file = file_path
                        self.manual_glossary_manually_loaded = False  # This is auto-loaded
                        self.append_log(f"📑 Auto-loaded glossary (CSV) for {file_base}: {os.path.basename(glossary_path)}")
                        break
                    else:
                        with open(glossary_path, 'r', encoding='utf-8') as f:
                            glossary_data = json.load(f)
                        self.manual_glossary_path = glossary_path
                        self.auto_loaded_glossary_path = glossary_path
                        self.auto_loaded_glossary_for_file = file_path
                        self.manual_glossary_manually_loaded = False  # This is auto-loaded
                        self.append_log(f"📑 Auto-loaded glossary (JSON) for {file_base}: {os.path.basename(glossary_path)}")
                        break
                except Exception:
                    # If JSON parsing fails, try next candidate
                    continue
                    continue
        
        return False

    # File Selection Methods
    def browse_files(self):
        """Select one or more files - automatically handles single/multiple selection"""
        file_filter = (
            "Supported files (*.epub *.cbz *.txt *.json *.png *.jpg *.jpeg *.gif *.bmp *.webp);;"
            "EPUB/CBZ (*.epub *.cbz);;"
            "EPUB files (*.epub);;"
            "Comic Book Zip (*.cbz);;"
            "Text files (*.txt *.json);;"
            "Image files (*.png *.jpg *.jpeg *.gif *.bmp *.webp);;"
            "PNG files (*.png);;"
            "JPEG files (*.jpg *.jpeg);;"
            "GIF files (*.gif);;"
            "BMP files (*.bmp);;"
            "WebP files (*.webp);;"
            "All files (*.*)"
        )
        
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select File(s) - Hold Ctrl/Shift to select multiple",
            "",
            file_filter
        )
        if paths:
            self._handle_file_selection(paths)

    def browse_folder(self):
        """Select an entire folder of files"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder Containing Files to Translate"
        )
        if folder_path:
            # Find all supported files in the folder
            supported_extensions = {'.epub', '.cbz', '.txt', '.json', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
            files = []
            
            # Recursively find files if deep scan is enabled
            if hasattr(self, 'deep_scan_var') and self.deep_scan_var:
                for root, dirs, filenames in os.walk(folder_path):
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        if os.path.splitext(filename)[1].lower() in supported_extensions:
                            files.append(file_path)
            else:
                # Just scan the immediate folder
                for filename in sorted(os.listdir(folder_path)):
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        ext = os.path.splitext(filename)[1].lower()
                        if ext in supported_extensions:
                            files.append(file_path)
            
            if files:
                self._handle_file_selection(sorted(files))
                self.append_log(f"📁 Found {len(files)} supported files in: {os.path.basename(folder_path)}")
            else:
                QMessageBox.warning(self, "No Files Found", 
                                     f"No supported files found in:\n{folder_path}\n\nSupported formats: EPUB, TXT, PNG, JPG, JPEG, GIF, BMP, WebP")

    def clear_file_selection(self):
        """Clear all selected files"""
        self.entry_epub.clear()
        self.entry_epub.setText("No file selected")
        self.selected_files = []
        self.file_path = None
        self.current_file_index = 0
        
        # Clear EPUB tracking
        if hasattr(self, 'selected_epub_path'):
            self.selected_epub_path = None
        if hasattr(self, 'selected_epub_files'):
            self.selected_epub_files = []
        
        # Persist clear state
        try:
            self.config['last_input_files'] = []
            self.config['last_epub_path'] = None
            self.save_config(show_message=False)
        except Exception:
            pass
        self.append_log("🗑️ Cleared file selection")


    def _handle_file_selection(self, paths):
        """Common handler for file selection"""
        if not paths:
            return
        
        # Initialize JSON conversion tracking if not exists
        if not hasattr(self, 'json_conversions'):
            self.json_conversions = {}  # Maps converted .txt paths to original .json paths
        
        # Process JSON files first - convert them to TXT
        processed_paths = []
        
        for path in paths:
            lower = path.lower()
            if lower.endswith('.json'):
                # Convert JSON to TXT
                txt_path = self._convert_json_to_txt(path)
                if txt_path:
                    processed_paths.append(txt_path)
                    # Track the conversion for potential reverse conversion later
                    self.json_conversions[txt_path] = path
                    self.append_log(f"📄 Converted JSON to TXT: {os.path.basename(path)}")
                else:
                    self.append_log(f"❌ Failed to convert JSON: {os.path.basename(path)}")
            elif lower.endswith('.cbz'):
                # Extract images from CBZ (ZIP) to a temp folder and add them
                try:
                    import zipfile, tempfile, shutil
                    temp_root = getattr(self, 'cbz_temp_root', None)
                    if not temp_root:
                        temp_root = tempfile.mkdtemp(prefix='glossarion_cbz_')
                        self.cbz_temp_root = temp_root
                    base = os.path.splitext(os.path.basename(path))[0]
                    extract_dir = os.path.join(temp_root, base)
                    os.makedirs(extract_dir, exist_ok=True)
                    with zipfile.ZipFile(path, 'r') as zf:
                        members = [m for m in zf.namelist() if m.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'))]
                        # Preserve order by natural sort
                        members.sort()
                        for m in members:
                            target_path = os.path.join(extract_dir, os.path.basename(m))
                            if not os.path.exists(target_path):
                                with zf.open(m) as src, open(target_path, 'wb') as dst:
                                    shutil.copyfileobj(src, dst)
                            processed_paths.append(target_path)
                    self.append_log(f"📦 Extracted {len([p for p in processed_paths if p.startswith(extract_dir)])} images from {os.path.basename(path)}")
                except Exception as e:
                    self.append_log(f"❌ Failed to read CBZ: {os.path.basename(path)} - {e}")
            else:
                # Non-JSON/CBZ files pass through unchanged
                processed_paths.append(path)
        
        # Store the list of selected files (using processed paths)
        self.selected_files = processed_paths
        self.current_file_index = 0
        
        # Persist last selection to config for next session
        try:
            self.config['last_input_files'] = processed_paths
            self.save_config(show_message=False)
        except Exception:
            pass
        
        # Update the entry field
        self.entry_epub.clear()
        
        # Define image extensions
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        
        if len(processed_paths) == 1:
            # Single file - display full path
            # Check if this was a JSON conversion
            if processed_paths[0] in self.json_conversions:
                # Show original JSON filename in parentheses
                original_json = self.json_conversions[processed_paths[0]]
                display_path = f"{processed_paths[0]} (from {os.path.basename(original_json)})"
                self.entry_epub.setText(display_path)
            else:
                self.entry_epub.setText(processed_paths[0])
            self.file_path = processed_paths[0]  # For backward compatibility
        else:
            # Multiple files - display count and summary
            # Group by type (count original types, not processed)
            images = [p for p in processed_paths if os.path.splitext(p)[1].lower() in image_extensions]
            epubs = [p for p in processed_paths if p.lower().endswith('.epub')]
            txts = [p for p in processed_paths if p.lower().endswith('.txt') and p not in self.json_conversions]
            jsons = [p for p in self.json_conversions.values()]  # Count original JSON files
            converted_txts = [p for p in processed_paths if p in self.json_conversions]
            
            summary_parts = []
            if epubs:
                summary_parts.append(f"{len(epubs)} EPUB")
            if txts:
                summary_parts.append(f"{len(txts)} TXT")
            if jsons:
                summary_parts.append(f"{len(jsons)} JSON")
            if images:
                summary_parts.append(f"{len(images)} images")
            
            display_text = f"{len(paths)} files selected ({', '.join(summary_parts)})"
            self.entry_epub.setText(display_text)
            self.file_path = processed_paths[0]  # Set first file as primary
        
        # Check if these are image files
        image_files = [p for p in processed_paths if os.path.splitext(p)[1].lower() in image_extensions]
        
        if image_files:
            # Enable image translation if not already enabled
            if hasattr(self, 'enable_image_translation_var') and not self.enable_image_translation_var:
                self.enable_image_translation_var = True
                self.append_log(f"🖼️ Detected {len(image_files)} image file(s) - automatically enabled image translation")
            
            # Clear glossary for image files
            if hasattr(self, 'auto_loaded_glossary_path'):
                #self.manual_glossary_path = None
                self.auto_loaded_glossary_path = None
                self.auto_loaded_glossary_for_file = None
                self.append_log("📑 Cleared glossary settings (image files selected)")
        else:
            # Handle EPUB/TXT files
            epub_files = [p for p in processed_paths if p.lower().endswith('.epub')]
            
            if len(epub_files) == 1:
                # Single EPUB - auto-load glossary
                self.auto_load_glossary_for_file(epub_files[0])
                # Persist EPUB path for QA defaults
                try:
                    self.selected_epub_path = epub_files[0]
                    self.selected_epub_files = [epub_files[0]]  # Track single EPUB in list format
                    self.config['last_epub_path'] = epub_files[0]
                    os.environ['EPUB_PATH'] = epub_files[0]
                    self.save_config(show_message=False)
                except Exception:
                    pass
            elif len(epub_files) > 1:
                # Multiple EPUBs - clear glossary but update EPUB path tracking
                if hasattr(self, 'auto_loaded_glossary_path'):
                    self.manual_glossary_path = None
                    self.auto_loaded_glossary_path = None
                    self.auto_loaded_glossary_for_file = None
                    self.append_log("📁 Multiple files selected - glossary auto-loading disabled")
                
                # For multiple EPUBs, set the selected_epub_path to the first one
                # but track all EPUBs for word count analysis
                try:
                    self.selected_epub_path = epub_files[0]  # Use first EPUB as primary
                    self.selected_epub_files = epub_files  # Track all EPUBs
                    self.config['last_epub_path'] = epub_files[0]
                    os.environ['EPUB_PATH'] = epub_files[0]
                    self.save_config(show_message=False)
                    
                    # Log that multiple EPUBs are selected
                    self.append_log(f"📖 {len(epub_files)} EPUB files selected - using '{os.path.basename(epub_files[0])}' as primary for word count analysis")
                except Exception:
                    pass

    def _convert_json_to_txt(self, json_path):
        """Convert a JSON file to TXT format for translation."""
        try:
            # Read JSON file
            with open(json_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                self.append_log(f"⚠️ JSON parsing error: {str(e)}")
                self.append_log("🔧 Attempting to fix JSON...")
                fixed_content = self._comprehensive_json_fix(content)
                data = json.loads(fixed_content)
                self.append_log("✅ JSON fixed successfully")
            
            # Create output file
            base_dir = os.path.dirname(json_path)
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            txt_path = os.path.join(base_dir, f"{base_name}_json_temp.txt")
            
            # CHECK IF THIS IS A GLOSSARY - PUT EVERYTHING IN ONE CHAPTER
            filename_lower = os.path.basename(json_path).lower()
            is_glossary = any(term in filename_lower for term in ['glossary', 'dictionary', 'terms', 'characters', 'names'])
            
            # Also check structure
            if not is_glossary and isinstance(data, dict):
                # If it's a flat dictionary with many short entries, it's probably a glossary
                if len(data) > 20:  # More than 20 entries
                    values = list(data.values())[:10]  # Check first 10
                    if all(isinstance(v, str) and len(v) < 500 for v in values):
                        is_glossary = True
                        self.append_log("📚 Detected glossary structure (many short entries)")
                        self.append_log(f"🔍 Found {len(data)} dictionary entries with avg length < 500 chars")
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                # Add metadata header
                f.write(f"[JSON_SOURCE: {os.path.basename(json_path)}]\n")
                f.write(f"[JSON_STRUCTURE_TYPE: {type(data).__name__}]\n")
                f.write(f"[JSON_CONVERSION_VERSION: 1.0]\n")
                if is_glossary:
                    f.write("[GLOSSARY_MODE: SINGLE_CHUNK]\n")
                f.write("\n")
                
                if is_glossary:
                    # PUT ENTIRE GLOSSARY IN ONE CHAPTER
                    self.append_log(f"📚 Glossary mode: Creating single chapter for {len(data)} entries")
                    self.append_log("🚫 CHUNK SPLITTING DISABLED for glossary file")
                    self.append_log(f"📝 All {len(data)} entries will be processed in ONE API call")
                    f.write("=== Chapter 1: Full Glossary ===\n\n")
                    
                    if isinstance(data, dict):
                        for key, value in data.items():
                            f.write(f"{key}: {value}\n\n")
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, str):
                                f.write(f"{item}\n\n")
                            else:
                                f.write(f"{json.dumps(item, ensure_ascii=False, indent=2)}\n\n")
                    else:
                        f.write(json.dumps(data, ensure_ascii=False, indent=2))
                
                else:
                    # NORMAL PROCESSING - SEPARATE CHAPTERS
                    if isinstance(data, dict):
                        for idx, (key, value) in enumerate(data.items(), 1):
                            f.write(f"\n=== Chapter {idx}: {key} ===\n\n")
                            
                            if isinstance(value, str):
                                f.write(value)
                            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                                for item in value:
                                    f.write(f"{item}\n\n")
                            else:
                                f.write(json.dumps(value, ensure_ascii=False, indent=2))
                            
                            f.write("\n\n")
                    
                    elif isinstance(data, list):
                        for idx, item in enumerate(data, 1):
                            f.write(f"\n=== Chapter {idx} ===\n\n")
                            
                            if isinstance(item, str):
                                f.write(item)
                            else:
                                f.write(json.dumps(item, ensure_ascii=False, indent=2))
                            
                            f.write("\n\n")
                    
                    else:
                        f.write("=== Content ===\n\n")
                        if isinstance(data, str):
                            f.write(data)
                        else:
                            f.write(json.dumps(data, ensure_ascii=False, indent=2))
            
            return txt_path
            
        except Exception as e:
            self.append_log(f"❌ Error converting JSON: {str(e)}")
            import traceback
            self.append_log(f"Debug: {traceback.format_exc()}")
            return None

    def convert_translated_to_json(self, translated_txt_path):
        """Convert translated TXT back to JSON format if it was originally JSON."""
        
        # Check if this was a JSON conversion
        original_json_path = None
        for txt_path, json_path in self.json_conversions.items():
            # Check if this is the translated version of a converted file
            if translated_txt_path.replace("_translated", "_json_temp") == txt_path:
                original_json_path = json_path
                break
            # Also check direct match
            if txt_path.replace("_json_temp", "_translated") == translated_txt_path:
                original_json_path = json_path
                break
        
        if not original_json_path:
            return None
        
        try:
            # Read original JSON structure
            with open(original_json_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            # Read translated content
            with open(translated_txt_path, 'r', encoding='utf-8') as f:
                translated_content = f.read()
            
            # Remove metadata headers
            lines = translated_content.split('\n')
            content_start = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('[JSON_'):
                    content_start = i
                    break
            translated_content = '\n'.join(lines[content_start:])
            
            # Parse chapters from translated content
            import re
            chapter_pattern = r'=== Chapter \d+(?:: ([^=]+))? ==='
            chapters = re.split(chapter_pattern, translated_content)
            
            # Clean up chapters
            cleaned_chapters = []
            for i, chapter in enumerate(chapters):
                if chapter and chapter.strip() and not chapter.startswith('==='):
                    cleaned_chapters.append(chapter.strip())
            
            # Rebuild JSON structure with translated content
            if isinstance(original_data, dict):
                result = {}
                keys = list(original_data.keys())
                
                # Match chapters to original keys
                for i, key in enumerate(keys):
                    if i < len(cleaned_chapters):
                        result[key] = cleaned_chapters[i]
                    else:
                        # Preserve original if no translation found
                        result[key] = original_data[key]
            
            elif isinstance(original_data, list):
                result = []
                
                for i, item in enumerate(original_data):
                    if i < len(cleaned_chapters):
                        if isinstance(item, dict) and 'content' in item:
                            # Preserve structure for dictionary items
                            new_item = item.copy()
                            new_item['content'] = cleaned_chapters[i]
                            result.append(new_item)
                        else:
                            # Direct replacement
                            result.append(cleaned_chapters[i])
                    else:
                        # Preserve original if no translation found
                        result.append(item)
            
            else:
                # Single value
                result = cleaned_chapters[0] if cleaned_chapters else original_data
            
            # Save as JSON
            output_json_path = translated_txt_path.replace('.txt', '.json')
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            self.append_log(f"✅ Converted back to JSON: {os.path.basename(output_json_path)}")
            return output_json_path
            
        except Exception as e:
            self.append_log(f"❌ Error converting back to JSON: {str(e)}")
            import traceback
            self.append_log(f"Debug: {traceback.format_exc()}")
            return None

    def toggle_api_visibility(self):
        if self.api_key_entry.echoMode() == QLineEdit.Password:
            self.api_key_entry.setEchoMode(QLineEdit.Normal)
            self.show_api_btn.setText("Hide")
            self.api_key_visible = True
        else:
            self.api_key_entry.setEchoMode(QLineEdit.Password)
            self.show_api_btn.setText("Show")
            self.api_key_visible = False
    
    def prompt_custom_token_limit(self):
       from PySide6.QtWidgets import QInputDialog
       val, ok = QInputDialog.getInt(
           self,
           "Set Max Output Token Limit",
           "Enter max output tokens for API output (e.g., 16384, 32768, 65536):",
           value=self.max_output_tokens,
           minValue=1,
           maxValue=2000000
       )
       if ok and val:
           self.max_output_tokens = val
           self.output_btn.setText(f"Output Token Limit: {val}")
           self.append_log(f"✅ Output token limit set to {val}")

    # Note: open_other_settings method is bound from other_settings.py during __init__
    # No need to define it here - it's injected dynamically
            
    def __setattr__(self, name, value):
        """Debug method to track when manual_glossary_path gets cleared"""
        if name == 'manual_glossary_path':
            import traceback
            if value is None and hasattr(self, 'manual_glossary_path') and self.manual_glossary_path is not None:
                if hasattr(self, 'append_log'):
                    self.append_log(f"[DEBUG] CLEARING manual_glossary_path from {self.manual_glossary_path} to None")
                    self.append_log(f"[DEBUG] Stack trace: {''.join(traceback.format_stack()[-3:-1])}")
                else:
                    print(f"[DEBUG] CLEARING manual_glossary_path from {getattr(self, 'manual_glossary_path', 'unknown')} to None")
                    print(f"[DEBUG] Stack trace: {''.join(traceback.format_stack()[-3:-1])}")
        super().__setattr__(name, value)

    def load_glossary(self):
        """Let the user pick a glossary file (JSON or CSV) and remember its path."""
        import json
        import shutil
        
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select glossary file",
            "",
            "Supported files (*.json *.csv *.txt);;JSON files (*.json);;CSV files (*.csv);;Text files (*.txt);;All files (*.*)"
        )
        if not path:
            return
        
        # Determine file type
        file_extension = os.path.splitext(path)[1].lower()
        
        if file_extension == '.csv':
            # Handle CSV file - just pass it through as-is
            # The translation system will handle the CSV file format
            pass
                
        elif file_extension == '.txt':
            # Handle TXT file - just pass it through as-is
            # The translation system will handle the text file format
            pass
                
        elif file_extension == '.json':
            # Original JSON handling code
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Store original content for comparison
                original_content = content
                
                # Try normal JSON load first
                try:
                    json.loads(content)
                except json.JSONDecodeError as e:
                    self.append_log(f"⚠️ JSON error detected: {str(e)}")
                    self.append_log("🔧 Attempting comprehensive auto-fix...")
                    
                    # Apply comprehensive auto-fixes
                    fixed_content = self._comprehensive_json_fix(content)
                    
                    # Try to parse the fixed content
                    try:
                        json.loads(fixed_content)
                        
                        # If successful, ask user if they want to save the fixed version
                        response = QMessageBox.question(
                            self,
                            "JSON Auto-Fix Successful",
                            f"The JSON file had errors that were automatically fixed.\n\n"
                            f"Original error: {str(e)}\n\n"
                            f"Do you want to save the fixed version?\n"
                            f"(A backup of the original will be created)",
                            QMessageBox.Yes | QMessageBox.No
                        )
                        response = (response == QMessageBox.Yes)
                        
                        if response:
                            # Save the fixed version
                            backup_path = path.replace('.json', '_backup.json')
                            shutil.copy2(path, backup_path)
                            
                            with open(path, 'w', encoding='utf-8') as f:
                                f.write(fixed_content)
                            
                            self.append_log(f"✅ Auto-fixed JSON and saved. Backup created: {os.path.basename(backup_path)}")
                            content = fixed_content
                        else:
                            self.append_log("⚠️ Using original JSON with errors (may cause issues)")
                        
                    except json.JSONDecodeError as e2:
                        # Auto-fix failed, show error and options
                        self.append_log(f"❌ Auto-fix failed: {str(e2)}")
                        
                        # Build detailed error message
                        error_details = self._analyze_json_errors(content, fixed_content, e, e2)
                        
                        msgbox = QMessageBox(self)
                        msgbox.setWindowTitle("JSON Fix Failed")
                        msgbox.setText(
                            f"The JSON file has errors that couldn't be automatically fixed.\n\n"
                            f"Original error: {str(e)}\n"
                            f"After auto-fix attempt: {str(e2)}\n\n"
                            f"{error_details}\n\n"
                            f"Options:\n"
                            f"• YES: Open the file in your default editor to fix manually\n"
                            f"• NO: Try to use the file anyway (may fail)\n"
                            f"• CANCEL: Cancel loading this glossary"
                        )
                        msgbox.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
                        response = msgbox.exec()
                        
                        # Convert QMessageBox response to True/False/None
                        if response == QMessageBox.Yes:
                            response = True
                        elif response == QMessageBox.No:
                            response = False
                        else:
                            response = None
                        
                        if response is True:  # YES - open in editor
                            try:
                                # Open file in default editor
                                import subprocess
                                import sys
                                
                                if sys.platform.startswith('win'):
                                    os.startfile(path)
                                elif sys.platform.startswith('darwin'):
                                    subprocess.run(['open', path])
                                else:  # linux
                                    subprocess.run(['xdg-open', path])
                                
                                QMessageBox.information(
                                    self,
                                    "Manual Edit",
                                    "Please fix the JSON errors in your editor and save the file.\n"
                                    "Then click OK to retry loading the glossary."
                                )
                                
                                # Recursively call load_glossary to retry
                                self.load_glossary()
                                return
                                
                            except Exception as editor_error:
                                QMessageBox.critical(
                                    self,
                                    "Error",
                                    f"Failed to open file in editor: {str(editor_error)}\n\n"
                                    f"Please manually edit the file:\n{path}"
                                )
                                return
                        
                        elif response is False:  # NO - try to use anyway
                            self.append_log("⚠️ Attempting to use JSON with errors (may cause issues)")
                            # Continue with the original content
                            
                        else:  # CANCEL
                            self.append_log("❌ Glossary loading cancelled")
                            return
            
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read glossary file: {str(e)}")
                return
        
        else:
            QMessageBox.critical(
                self,
                "Error", 
                f"Unsupported file type: {file_extension}\n"
                "Please select a JSON, CSV, or TXT file."
            )
            return
        
        # Clear auto-loaded tracking when manually loading
        self.auto_loaded_glossary_path = None
        self.auto_loaded_glossary_for_file = None
        
        self.manual_glossary_path = path
        self.manual_glossary_manually_loaded = True
        self.append_log(f"📑 Loaded manual glossary: {path}")
        
        # Save the file extension for later reference
        self.manual_glossary_file_extension = file_extension
        
        self.append_glossary_var = True
        self.append_log("✅ Automatically enabled 'Append Glossary to System Prompt'")

    def _comprehensive_json_fix(self, content):
        """Apply comprehensive JSON fixes."""
        import re
        
        # Store original for comparison
        fixed = content
        
        # 1. Remove BOM if present
        if fixed.startswith('\ufeff'):
            fixed = fixed[1:]
        
        # 2. Fix common Unicode issues first
        replacements = {
            '"': '"',  # Left smart quote
            '"': '"',  # Right smart quote
            ''': "'",  # Left smart apostrophe
            ''': "'",  # Right smart apostrophe
            '–': '-',  # En dash
            '—': '-',  # Em dash
            '…': '...',  # Ellipsis
            '\u200b': '',  # Zero-width space
            '\u00a0': ' ',  # Non-breaking space
        }
        for old, new in replacements.items():
            fixed = fixed.replace(old, new)
        
        # 3. Fix trailing commas in objects and arrays
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        # 4. Fix multiple commas
        fixed = re.sub(r',\s*,+', ',', fixed)
        
        # 5. Fix missing commas between array/object elements
        # Between closing and opening braces/brackets
        fixed = re.sub(r'}\s*{', '},{', fixed)
        fixed = re.sub(r']\s*\[', '],[', fixed)
        fixed = re.sub(r'}\s*\[', '},[', fixed)
        fixed = re.sub(r']\s*{', '],{', fixed)
        
        # Between string values (but not inside strings)
        # This is tricky, so we'll be conservative
        fixed = re.sub(r'"\s+"(?=[^:]*":)', '","', fixed)
        
        # 6. Fix unquoted keys (simple cases)
        # Match unquoted keys that are followed by a colon
        fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)
        
        # 7. Fix single quotes to double quotes for keys and simple string values
        # Keys
        fixed = re.sub(r"([{,]\s*)'([^']+)'(\s*:)", r'\1"\2"\3', fixed)
        # Simple string values (be conservative)
        fixed = re.sub(r"(:\s*)'([^'\"]*)'(\s*[,}])", r'\1"\2"\3', fixed)
        
        # 8. Fix common escape issues
        # Replace single backslashes with double backslashes (except for valid escapes)
        # This is complex, so we'll only fix obvious cases
        fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', fixed)
        
        # 9. Ensure proper brackets/braces balance
        # Count opening and closing brackets
        open_braces = fixed.count('{')
        close_braces = fixed.count('}')
        open_brackets = fixed.count('[')
        close_brackets = fixed.count(']')
        
        # Add missing closing braces/brackets at the end
        if open_braces > close_braces:
            fixed += '}' * (open_braces - close_braces)
        if open_brackets > close_brackets:
            fixed += ']' * (open_brackets - close_brackets)
        
        # 10. Remove trailing comma before EOF
        fixed = re.sub(r',\s*$', '', fixed.strip())
        
        # 11. Fix unescaped newlines in strings (conservative approach)
        # This is very tricky to do with regex without a proper parser
        # We'll skip this for safety
        
        # 12. Remove comments (JSON doesn't support comments)
        # Remove // style comments
        fixed = re.sub(r'//.*$', '', fixed, flags=re.MULTILINE)
        # Remove /* */ style comments
        fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
        
        return fixed

    def _analyze_json_errors(self, original, fixed, original_error, fixed_error):
        """Analyze JSON errors and provide helpful information."""
        analysis = []
        
        # Check for common issues
        if '{' in original and original.count('{') != original.count('}'):
            analysis.append(f"• Mismatched braces: {original.count('{')} opening, {original.count('}')} closing")
        
        if '[' in original and original.count('[') != original.count(']'):
            analysis.append(f"• Mismatched brackets: {original.count('[')} opening, {original.count(']')} closing")
        
        if original.count('"') % 2 != 0:
            analysis.append("• Odd number of quotes (possible unclosed string)")
        
        # Check for BOM
        if original.startswith('\ufeff'):
            analysis.append("• File starts with BOM (Byte Order Mark)")
        
        # Check for common problematic patterns
        if re.search(r'[''""…]', original):
            analysis.append("• Contains smart quotes or special Unicode characters")
        
        if re.search(r':\s*[a-zA-Z_][a-zA-Z0-9_]*\s*[,}]', original):
            analysis.append("• Possible unquoted string values")
        
        if re.search(r'[{,]\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:', original):
            analysis.append("• Possible unquoted keys")
        
        if '//' in original or '/*' in original:
            analysis.append("• Contains comments (not valid in JSON)")
        
        # Try to find the approximate error location
        if hasattr(original_error, 'lineno'):
            lines = original.split('\n')
            if 0 < original_error.lineno <= len(lines):
                error_line = lines[original_error.lineno - 1]
                analysis.append(f"\nError near line {original_error.lineno}:")
                analysis.append(f"  {error_line.strip()}")
        
        return "\n".join(analysis) if analysis else "Unable to determine specific issues."

    def save_config(self, show_message=True):
        """Persist all settings to config.json."""
        try:
            # Add comprehensive environment variable debugging for all saves (only when debug mode is enabled)
            debug_enabled = getattr(self, 'config', {}).get('show_debug_buttons', False)
            if show_message and debug_enabled:
                self.append_log("🔍 [SAVE_CONFIG] Starting comprehensive config save with environment variable debugging...")

            # Create backup of existing config before saving
            self._backup_config_file()

            # Helper functions for safe type conversion
            def safe_int(value, default):
                try: return int(value)
                except (ValueError, TypeError): return default
            
            def safe_float(value, default):
                try: return float(value)
                except (ValueError, TypeError): return default

            # --- 1. Input Validation ---
            # Validate numeric fields before saving (skip if called silently)
            if show_message:
                validation_map = [
                    (self.delay_entry, "API call delay", lambda v: v.replace('.', '', 1).isdigit() or v == ""),
                    (self.thread_delay_entry, "Threading Delay", lambda v: v.replace('.', '', 1).isdigit()),
                    (self.trans_temp, "Temperature", lambda v: v == "" or v.replace('.', '', 1).replace('-', '', 1).isdigit()),
                    (self.trans_history, "Translation History Limit", lambda v: v.isdigit() or v == ""),
                ]
                from PySide6.QtWidgets import QMessageBox
                for source, name, is_valid_func in validation_map:
                    try:
                        value = source.text().strip() if hasattr(source, 'text') else source.strip()
                        if not is_valid_func(value):
                            QMessageBox.critical(None, "Invalid Input", f"Please enter a valid number for {name}")
                            return
                    except (AttributeError, ValueError):
                        QMessageBox.critical(None, "Invalid Input", f"Please enter a valid number for {name}")
                        return

            # --- 2. Data-Driven Configuration Mapping ---
            # Helper to get value from a source (widget or variable)
            def _get_value(source_attr):
                if not hasattr(self, source_attr):
                    return None
                
                attr = getattr(self, source_attr)
                if hasattr(attr, 'isChecked'): return attr.isChecked()
                if hasattr(attr, 'toPlainText'): return attr.toPlainText().strip()
                if hasattr(attr, 'text'): return attr.text().strip()
                if hasattr(attr, 'currentIndex'): return attr.currentIndex()
                return attr

            # Central mapping of configuration settings
            # format: (config_key, [source_attributes_in_priority_order], default_value, type_converter_func)
            settings_map = [
                # Basic settings
                ('model', ['model_var'], None, str),
                ('active_profile', ['profile_var'], None, str),
                ('prompt_profiles', ['prompt_profiles'], {}, dict),
                ('contextual', ['contextual_var'], None, bool),
                ('api_key', ['api_key_entry'], '', str),
                ('chapter_range', ['chapter_range_entry'], '', str),
                
                # Numeric settings
                ('delay', ['delay_entry'], 2.0, lambda v: safe_float(v, 2.0)),
                ('thread_submission_delay', ['thread_delay_entry'], 0.5, lambda v: safe_float(v, 0.5)),
                ('translation_temperature', ['trans_temp'], 0.3, lambda v: safe_float(v, 0.3)),
                ('translation_history_limit', ['trans_history'], 2, lambda v: safe_int(v, 2)),
                ('reinforcement_frequency', ['reinforcement_freq_var'], 10, lambda v: safe_int(v, 10)),
                ('duplicate_lookback_chapters', ['duplicate_lookback_var'], 5, lambda v: safe_int(v, 5)),

                # Boolean toggles - prioritize checkboxes over vars
                ('REMOVE_AI_ARTIFACTS', ['remove_artifacts_checkbox', 'REMOVE_AI_ARTIFACTS_var'], False, bool),
                ('attach_css_to_chapters', ['attach_css_to_chapters_var'], False, bool),
                ('use_rolling_summary', ['rolling_summary_var'], False, bool),
                ('translate_book_title', ['translate_book_title_var'], False, bool),
                ('emergency_paragraph_restore', ['emergency_restore_var'], False, bool),
                ('retry_duplicate_bodies', ['retry_duplicate_var'], False, bool),
                ('token_limit_disabled', ['token_limit_disabled'], False, bool),
                ('conservative_batching', ['conservative_batching_var'], False, bool),
                ('translation_history_rolling', ['rolling_checkbox', 'translation_history_rolling_var'], False, bool),
                ('disable_epub_gallery', ['disable_epub_gallery_var'], False, bool),
                ('disable_automatic_cover_creation', ['disable_automatic_cover_creation_var'], False, bool),
                ('duplicate_detection_mode', ['duplicate_detection_mode_var'], 'off', str),
                ('use_header_as_output', ['use_header_as_output_var'], False, bool),
                ('enable_decimal_chapters', ['enable_decimal_chapters_var'], False, bool),
                ('force_ncx_only', ['force_ncx_only_var'], False, bool),
                ('batch_translate_headers', ['batch_translate_headers_var'], False, bool),
                ('update_html_headers', ['update_html_headers_var'], False, bool),
                ('save_header_translations', ['save_header_translations_var'], False, bool),
                ('use_sorted_fallback', ['use_sorted_fallback_var'], False, bool),
                ('single_api_image_chunks', ['single_api_image_chunks_var'], False, bool),
                ('use_custom_openai_endpoint', ['use_custom_openai_endpoint_var'], False, bool),
                ('disable_chapter_merging', ['disable_chapter_merging_var'], False, bool),
                ('use_gemini_openai_endpoint', ['use_gemini_openai_endpoint_var'], False, bool),
                ('use_fallback_keys', ['use_fallback_keys_var'], False, bool),
                ('auto_update_check', ['auto_update_check_var'], True, bool),
                ('ignore_header', ['ignore_header_var'], False, bool),
                ('ignore_title', ['ignore_title_var'], False, bool),
                ('scan_phase_enabled', ['scan_phase_enabled_var'], False, bool),

                # Prompts and text fields
                ('summary_role', ['summary_role_var'], '', str),
                ('book_title_prompt', ['book_title_prompt'], '', str),
                ('translation_chunk_prompt', ['translation_chunk_prompt'], '', str),
                ('image_chunk_prompt', ['image_chunk_prompt'], '', str),
                ('vertex_ai_location', ['vertex_location_var'], '', str),
                ('openai_base_url', ['openai_base_url_var'], '', str),
                ('groq_base_url', ['groq_base_url_var'], '', str),
                ('fireworks_base_url', ['fireworks_base_url_var'], '', str),
                ('gemini_openai_endpoint', ['gemini_openai_endpoint_var'], '', str),

                # Image settings
                ('enable_image_translation', ['enable_image_translation_var'], False, bool),
                ('process_webnovel_images', ['process_webnovel_images_var'], False, bool),
                ('webnovel_min_height', ['webnovel_min_height_var'], 1000, lambda v: safe_int(v, 1000)),
                ('max_images_per_chapter', ['max_images_per_chapter_var'], 1, lambda v: safe_int(v, 1)),
                ('enable_watermark_removal', ['enable_watermark_removal_var'], False, bool),
                ('save_cleaned_images', ['save_cleaned_images_var'], False, bool),
                ('advanced_watermark_removal', ['advanced_watermark_removal_var'], False, bool),
                ('compression_factor', ['compression_factor_var'], 1.0, float),
                ('image_chunk_overlap', ['image_chunk_overlap_var'], 1.0, lambda v: safe_float(v, 1.0)),

                # Batching
                ('batch_translation', ['batch_checkbox', 'batch_translation_var'], False, bool),
                ('batch_size', ['batch_size_entry', 'batch_size_var'], 3, lambda v: safe_int(v, 3)),
                ('headers_per_batch', ['headers_per_batch_var'], 10, int),

                # Gemini/GPT Thinking
                ('enable_gemini_thinking', ['enable_gemini_thinking_var'], False, bool),
                ('thinking_budget', ['thinking_budget_var'], 0, lambda v: int(v) if str(v).lstrip('-').isdigit() else 0),
                ('enable_gpt_thinking', ['enable_gpt_thinking_var'], False, bool),
                ('gpt_reasoning_tokens', ['gpt_reasoning_tokens_var'], 0, lambda v: int(v) if str(v).lstrip('-').isdigit() else 0),
                ('gpt_effort', ['gpt_effort_var'], 'auto', str),
                
                # Chapter processing
                ('chapter_number_offset', ['chapter_number_offset_var'], 0, lambda v: safe_int(v, 0)),
                ('max_output_tokens', ['max_output_tokens'], 8192, int),

                # Glossary Settings
                ('append_glossary', ['append_glossary_checkbox', 'append_glossary_var'], False, bool),
                ('glossary_min_frequency', ['glossary_min_frequency_entry', 'glossary_min_frequency_var'], 2, lambda v: safe_int(v, 2)),
                ('glossary_max_names', ['glossary_max_names_entry', 'glossary_max_names_var'], 50, lambda v: safe_int(v, 50)),
                ('glossary_max_titles', ['glossary_max_titles_entry', 'glossary_max_titles_var'], 30, lambda v: safe_int(v, 30)),
                ('glossary_batch_size', ['glossary_batch_size_entry', 'glossary_batch_size_var'], 50, lambda v: safe_int(v, 50)),
                ('glossary_max_text_size', ['glossary_max_text_size_entry', 'glossary_max_text_size_var'], 50000, lambda v: safe_int(v, 50000)),
                ('glossary_chapter_split_threshold', ['glossary_chapter_split_threshold_entry', 'glossary_chapter_split_threshold_var'], 8192, lambda v: safe_int(v, 8192)),
                ('glossary_max_sentences', ['glossary_max_sentences_entry', 'glossary_max_sentences_var'], 200, lambda v: safe_int(v, 200)),
                ('strip_honorifics', ['strip_honorifics_checkbox', 'strip_honorifics_var'], False, bool),
                ('glossary_disable_honorifics_filter', ['disable_honorifics_checkbox', 'disable_honorifics_var'], False, bool),
                ('manual_glossary_temperature', ['manual_temp_entry', 'manual_temp_var'], 0.3, lambda v: safe_float(v, 0.3)),
                ('manual_context_limit', ['manual_context_entry', 'manual_context_var'], 5, lambda v: safe_int(v, 5)),
                ('glossary_history_rolling', ['glossary_history_rolling_checkbox', 'glossary_history_rolling_var'], False, bool),
                ('enable_auto_glossary', ['enable_auto_glossary_checkbox', 'enable_auto_glossary_var'], False, bool),
                ('glossary_use_legacy_csv', ['use_legacy_csv_checkbox', 'use_legacy_csv_var'], False, bool),
                ('glossary_filter_mode', ['glossary_filter_mode_var'], 'strict', str),
                ('scan_phase_mode', ['scan_phase_mode_var'], 'translate', str),

                # Extraction settings - NOTE: these are only created in Other Settings dialog
                ('enable_parallel_extraction', ['enable_parallel_extraction_var'], False, bool),
                ('extraction_workers', ['extraction_workers_var'], 1, int),
                ('text_extraction_method', ['text_extraction_method_var'], 'standard', str),
                ('file_filtering_level', ['file_filtering_level_var'], 'smart', str),
                ('enhanced_preserve_structure', ['enhanced_preserve_structure_var'], True, bool),
                ('enhanced_filtering', ['enhanced_filtering_var'], 'smart', str), # Backwards compatibility
                ('force_bs_for_traditional', ['force_bs_for_traditional_var'], False, bool),  # Updated by other_settings.py
                
                # HTTP/Network tuning - prioritize entry widgets over vars
                ('chunk_timeout', ['chunk_timeout_var'], 900, lambda v: safe_int(v, 900)),
                ('enable_http_tuning', ['http_tuning_checkbox', 'enable_http_tuning_var'], False, bool),
                ('connect_timeout', ['connect_timeout_entry', 'connect_timeout_var'], 10.0, lambda v: safe_float(v, 10.0)),
                ('read_timeout', ['read_timeout_entry', 'read_timeout_var'], 180.0, lambda v: safe_float(v, 180.0)),
                ('http_pool_connections', ['http_pool_connections_entry', 'http_pool_connections_var'], 20, lambda v: safe_int(v, 20)),
                ('http_pool_maxsize', ['http_pool_maxsize_entry', 'http_pool_maxsize_var'], 50, lambda v: safe_int(v, 50)),
                ('ignore_retry_after', ['ignore_retry_after_checkbox', 'ignore_retry_after_var'], False, bool),
                ('max_retries', ['max_retries_var'], 7, lambda v: safe_int(v, 7)),
                ('indefinite_rate_limit_retry', ['indefinite_rate_limit_retry_var'], False, bool),

                # Retry settings
                ('retry_truncated', ['retry_truncated_var'], False, bool),
                ('max_retry_tokens', ['max_retry_tokens_var'], 16384, lambda v: safe_int(v, 16384)),
                ('retry_timeout', ['retry_timeout_var'], False, bool),
                ('preserve_original_text_on_failure', ['preserve_original_text_var'], False, bool),
                
                # Rolling summary
                ('rolling_summary_exchanges', ['rolling_summary_exchanges_var'], 5, lambda v: safe_int(v, 5)),
                ('rolling_summary_mode', ['rolling_summary_mode_var'], 'chapter', str),
                ('rolling_summary_max_entries', ['rolling_summary_max_entries_var'], 10, lambda v: safe_int(v, 10)),

                # QA/Scanning
                ('qa_auto_search_output', ['qa_auto_search_output_var'], False, bool),
                ('disable_zero_detection', ['disable_zero_detection_var'], False, bool),
                ('disable_gemini_safety', ['disable_gemini_safety_var'], False, bool),
                
                # Anti-duplicate parameters - all vars updated by other_settings.py callbacks
                ('enable_anti_duplicate', ['enable_anti_duplicate_var'], False, bool),
                ('top_p', ['top_p_var'], 1.0, float),
                ('top_k', ['top_k_var'], 50, int),
                ('frequency_penalty', ['frequency_penalty_var'], 0.0, float),
                ('presence_penalty', ['presence_penalty_var'], 0.0, float),
                ('repetition_penalty', ['repetition_penalty_var'], 1.0, float),
                ('candidate_count', ['candidate_count_var'], 1, int),
                ('custom_stop_sequences', ['custom_stop_sequences_var'], '', str),
                ('logit_bias_enabled', ['logit_bias_enabled_var'], False, bool),
                ('logit_bias_strength', ['logit_bias_strength_var'], 1.0, float),
                ('bias_common_words', ['bias_common_words_var'], False, bool),
                ('bias_repetitive_phrases', ['bias_repetitive_phrases_var'], False, bool),

                # OpenRouter
                ('openrouter_use_http_only', ['openrouter_http_only_var'], False, bool),
                ('openrouter_accept_identity', ['openrouter_accept_identity_var'], False, bool),
                ('openrouter_preferred_provider', ['openrouter_preferred_provider_var', ('config', 'openrouter_preferred_provider')], '', str),

                # Environment-backed settings
                ('retain_source_extension', ['retain_source_extension_var'], False, bool),
                ('enable_gui_yield', ['enable_gui_yield_var'], True, bool),
                
                # File selection settings
                ('deep_scan', ['deep_scan_check', 'deep_scan_var'], False, bool),
                
                # Async processing settings
                ('async_wait_for_completion', ['async_wait_for_completion_var'], False, bool),
                ('async_poll_interval', ['async_poll_interval_var'], 60, lambda v: safe_int(v, 60)),
            ]
            
            # Process the settings map to populate self.config
            for key, sources, default, converter in settings_map:
                final_value = None
                found = False
                for source in sources:
                    if isinstance(source, tuple): # Handle special config source
                        val = self.config.get(source[1])
                    else:
                        val = _get_value(source)

                    if val is not None:
                        final_value = val
                        found = True
                        break
                
                if found:
                    converted_value = converter(final_value) if converter else final_value
                    self.config[key] = converted_value
                elif default is not None:
                    self.config[key] = default

            # --- 3. Handle Special Cases and Complex Logic ---
            
            # Fuzzy matching threshold with range validation
            # Check slider first (created in glossary settings dialog), then fallback to var
            if hasattr(self, 'fuzzy_threshold_slider'):
                fuzzy_val = self.fuzzy_threshold_slider.value() / 100.0
                self.config['glossary_fuzzy_threshold'] = fuzzy_val if 0.5 <= fuzzy_val <= 1.0 else 0.90
            elif hasattr(self, 'fuzzy_threshold_value'):
                fuzzy_val = self.fuzzy_threshold_value
                self.config['glossary_fuzzy_threshold'] = fuzzy_val if 0.5 <= fuzzy_val <= 1.0 else 0.90
            elif hasattr(self, 'fuzzy_threshold_var'):
                fuzzy_val = self.fuzzy_threshold_var
                self.config['glossary_fuzzy_threshold'] = fuzzy_val if 0.5 <= fuzzy_val <= 1.0 else 0.90

            # Glossary filter mode from radio buttons
            if hasattr(self, 'glossary_filter_mode_buttons'):
                for mode_key, radio_button in self.glossary_filter_mode_buttons.items():
                    if radio_button.isChecked():
                        self.config['glossary_filter_mode'] = mode_key
                        break
            
            # Duplicate algorithm from combo box
            if hasattr(self, 'duplicate_algo_combo'):
                algo_reverse_map = {0: 'auto', 1: 'strict', 2: 'balanced', 3: 'aggressive', 4: 'basic'}
                self.config['glossary_duplicate_algorithm'] = algo_reverse_map.get(self.duplicate_algo_combo.currentIndex(), 'auto')
            elif hasattr(self, 'glossary_duplicate_algorithm_var'):
                self.config['glossary_duplicate_algorithm'] = self.glossary_duplicate_algorithm_var

            # Custom glossary data structures
            if hasattr(self, 'custom_glossary_fields'):
                self.config['custom_glossary_fields'] = self.custom_glossary_fields
            # Update enabled status from checkboxes (try both possible attribute names)
            if hasattr(self, 'type_enabled_checks'):
                for type_name, checkbox in self.type_enabled_checks.items():
                    if type_name in self.custom_entry_types:
                        self.custom_entry_types[type_name]['enabled'] = checkbox.isChecked()
            elif hasattr(self, 'type_enabled_checkboxes'):
                for type_name, checkbox in self.type_enabled_checkboxes.items():
                    if type_name in self.custom_entry_types:
                        self.custom_entry_types[type_name]['enabled'] = checkbox.isChecked()
            if hasattr(self, 'custom_entry_types'):
                self.config['custom_entry_types'] = self.custom_entry_types

            # Backward compatibility for translate_special_files
            if hasattr(self, 'translate_special_files_var'):
                self.config['translate_special_files'] = self.translate_special_files_var
                self.config['translate_cover_html'] = self.translate_special_files_var
            elif hasattr(self, 'translate_cover_html_var'):
                self.config['translate_cover_html'] = self.translate_cover_html_var
                self.config['translate_special_files'] = self.translate_cover_html_var

            # Backward compatibility for extraction_mode
            if hasattr(self, 'text_extraction_method_var') and hasattr(self, 'file_filtering_level_var'):
                if self.text_extraction_method_var == 'enhanced':
                    self.config['extraction_mode'] = 'enhanced'
                    self.config['enhanced_filtering'] = self.file_filtering_level_var
                else:
                    self.config['extraction_mode'] = self.file_filtering_level_var
            elif hasattr(self, 'extraction_mode_var'):
                self.config['extraction_mode'] = self.extraction_mode_var

            # Token limit
            _tl = self.token_limit_entry.text().strip()
            self.config['token_limit'] = int(_tl) if _tl.isdigit() else None

            # Update last update check time
            if hasattr(self, 'update_manager') and self.update_manager:
                self.config['last_update_check_time'] = self.update_manager._last_check_time

            # Save prompts from text widgets
            prompt_widgets = {
                'auto_glossary_prompt': 'auto_prompt_text',
                'append_glossary_prompt': 'append_prompt_text',
                'glossary_translation_prompt': 'translation_prompt_text',
                'glossary_format_instructions': 'format_instructions_text',
            }
            for key, widget_name in prompt_widgets.items():
                if hasattr(self, widget_name):
                    try:
                        self.config[key] = getattr(self, widget_name).toPlainText().strip()
                    except Exception:
                        pass

            # Set defaults for settings that might not exist yet
            self.config.setdefault('glossary_auto_backup', True)
            self.config.setdefault('glossary_max_backups', 50)
            default_qa_settings = {'foreign_char_threshold': 10, 'excluded_characters': '', 'target_language': 'english', 'check_encoding_issues': False, 'check_repetition': True, 'check_translation_artifacts': False, 'check_glossary_leakage': True, 'min_file_length': 0, 'report_format': 'detailed', 'auto_save_report': True, 'check_word_count_ratio': False, 'check_multiple_headers': True, 'warn_name_mismatch': False, 'check_missing_html_tag': True, 'check_paragraph_structure': True, 'check_invalid_nesting': False, 'paragraph_threshold': 0.3, 'cache_enabled': True, 'cache_auto_size': False, 'cache_show_stats': False}
            self.config.setdefault('qa_scanner_settings', default_qa_settings)
            self.config.setdefault('ai_hunter_config', {}).setdefault('ai_hunter_max_workers', 1)
            # Image compression defaults
            compression_defaults = {'enable_image_compression': False, 'auto_compress_enabled': True, 'target_image_tokens': 1000, 'image_compression_format': 'auto', 'webp_quality': 85, 'jpeg_quality': 85, 'png_compression': 6, 'max_image_dimension': 2048, 'max_image_size_mb': 10, 'preserve_transparency': False, 'preserve_original_format': False, 'optimize_for_ocr': True, 'progressive_encoding': True, 'save_compressed_images': False}
            for key, val in compression_defaults.items():
                self.config.setdefault(key, val)
            
            # --- 4. Update Environment Variables ---
            def _update_env(key, new_val, is_bool=False):
                val_to_set = str(new_val)
                if is_bool:
                    val_to_set = '1' if new_val else '0'
                
                old_val = os.environ.get(key, '<NOT SET>')
                os.environ[key] = val_to_set
                if show_message and debug_enabled and old_val != val_to_set:
                    self.append_log(f"🔍 [DEBUG] ENV {key}: '{old_val}' → '{val_to_set}'")
                return key
            
            env_vars_set = []
            # Standard env vars
            env_vars_set.append(_update_env('OPENROUTER_USE_HTTP_ONLY', self.config.get('openrouter_use_http_only'), is_bool=True))
            env_vars_set.append(_update_env('OPENROUTER_ACCEPT_IDENTITY', self.config.get('openrouter_accept_identity'), is_bool=True))
            env_vars_set.append(_update_env('OPENROUTER_PREFERRED_PROVIDER', self.config.get('openrouter_preferred_provider', '')))
            env_vars_set.append(_update_env('RETAIN_SOURCE_EXTENSION', self.config.get('retain_source_extension'), is_bool=True))
            env_vars_set.append(_update_env('ENABLE_GUI_YIELD', self.config.get('enable_gui_yield'), is_bool=True))

            # Extraction workers env var
            new_workers = str(self.config['extraction_workers']) if self.config['enable_parallel_extraction'] else "1"
            env_vars_set.append(_update_env('EXTRACTION_WORKERS', new_workers))

            # Wire debug payload saving to GUI debug mode
            os.environ['DEBUG_SAVE_REQUEST_PAYLOADS_VERBOSE'] = '1' if debug_enabled else '0'
            os.environ['SHOW_DEBUG_BUTTONS'] = '1' if debug_enabled else '0'
            os.environ['DEBUG_SAVE_REQUEST_PAYLOADS'] = '1'

            # Glossary-related environment variables
            if show_message and debug_enabled: self.append_log("🔍 [DEBUG] Setting glossary environment variables...")
            try:
                # Normalize and align glossary prompts
                prompt_keys = ['manual_glossary_prompt', 'append_glossary_prompt', 'auto_glossary_prompt', 'glossary_translation_prompt', 'glossary_format_instructions']
                for key in prompt_keys:
                    self.config[key] = self.config.get(key, '') or ''

                glossary_env_mappings = [
                    ('GLOSSARY_SYSTEM_PROMPT', self.config.get('manual_glossary_prompt', '')),
                    ('AUTO_GLOSSARY_PROMPT', self.config.get('auto_glossary_prompt', '')),
                    ('APPEND_GLOSSARY_PROMPT', self.config.get('append_glossary_prompt', '')),
                    ('GLOSSARY_TRANSLATION_PROMPT', self.config.get('glossary_translation_prompt', '')),
                    ('GLOSSARY_FORMAT_INSTRUCTIONS', self.config.get('glossary_format_instructions', '')),
                    ('GLOSSARY_DISABLE_HONORIFICS_FILTER', '1' if self.config.get('glossary_disable_honorifics_filter') else '0'),
                    ('GLOSSARY_STRIP_HONORIFICS', '1' if self.config.get('strip_honorifics') else '0'),
                    ('GLOSSARY_FUZZY_THRESHOLD', str(self.config.get('glossary_fuzzy_threshold', 0.90))),
                    ('GLOSSARY_USE_LEGACY_CSV', '1' if self.config.get('glossary_use_legacy_csv') else '0'),
                    ('GLOSSARY_MAX_SENTENCES', str(self.config.get('glossary_max_sentences', 200))),
                    # Add missing environment variables that GlossaryManager.py reads
                    ('GLOSSARY_MIN_FREQUENCY', str(self.config.get('glossary_min_frequency', 2))),
                    ('GLOSSARY_MAX_NAMES', str(self.config.get('glossary_max_names', 50))),
                    ('GLOSSARY_MAX_TITLES', str(self.config.get('glossary_max_titles', 30))),
                    ('GLOSSARY_BATCH_SIZE', str(self.config.get('glossary_batch_size', 50))),
                    ('GLOSSARY_MAX_TEXT_SIZE', str(self.config.get('glossary_max_text_size', 50000))),
                    ('GLOSSARY_CHAPTER_SPLIT_THRESHOLD', str(self.config.get('glossary_chapter_split_threshold', 8192))),
                    ('GLOSSARY_FILTER_MODE', self.config.get('glossary_filter_mode', 'strict')),
                ]
                for env_key, env_value in glossary_env_mappings:
                    if env_key:  # Skip None entries
                        env_vars_set.append(_update_env(env_key, env_value))
                
                # JSON environment variables
                custom_types_json = json.dumps(self.config.get('custom_entry_types', {}))
                env_vars_set.append(_update_env('GLOSSARY_CUSTOM_ENTRY_TYPES', custom_types_json))
                custom_fields_json = json.dumps(self.config.get('custom_glossary_fields', []))
                env_vars_set.append(_update_env('GLOSSARY_CUSTOM_FIELDS', custom_fields_json))
            except Exception as e:
                if show_message and debug_enabled: self.append_log(f"❌ [DEBUG] Glossary environment variable setup failed: {e}")

            if show_message and debug_enabled:
                self.append_log(f"🔍 [DEBUG] Set {len(env_vars_set)} environment variables.")

            # --- 5. Final Write to File ---
            google_creds_path = self.config.get('google_cloud_credentials')
            encrypted_config = encrypt_config(self.config)
            if google_creds_path:
                encrypted_config['google_cloud_credentials'] = google_creds_path
            
            json.dumps(encrypted_config, ensure_ascii=False, indent=2) # Validation check
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(encrypted_config, f, ensure_ascii=False, indent=2)

            # --- 6. Post-Save Verification and Messaging ---
            if show_message and debug_enabled:
                self.append_log("🔍 [SAVE_CONFIG] Verifying environment variables after config save...")
                critical_vars_to_check = [
                    ('OPENROUTER_USE_HTTP_ONLY', '1' if self.config.get('openrouter_use_http_only') else '0'),
                    ('OPENROUTER_ACCEPT_IDENTITY', '1' if self.config.get('openrouter_accept_identity') else '0'),
                    ('OPENROUTER_PREFERRED_PROVIDER', self.config.get('openrouter_preferred_provider', '')),
                    ('EXTRACTION_WORKERS', str(self.config.get('extraction_workers')) if self.config.get('enable_parallel_extraction') else '1'),
                    ('ENABLE_GUI_YIELD', '1' if self.config.get('enable_gui_yield') else '0'),
                    ('RETAIN_SOURCE_EXTENSION', '1' if self.config.get('retain_source_extension') else '0'),
                ]
                total_issues = 0
                for env_key, expected_str in critical_vars_to_check:
                    actual_value = os.environ.get(env_key, '<NOT SET>')
                    if actual_value != expected_str:
                        self.append_log(f"⚠️ [SAVE_CONFIG] {env_key}: expected '{expected_str}', got '{actual_value}'")
                        total_issues += 1
                    else:
                        self.append_log(f"✅ [SAVE_CONFIG] {env_key}: '{actual_value}' (correct)")
                
                if total_issues > 0:
                    self.append_log(f"❌ [SAVE_CONFIG] {total_issues} environment variable issues found!")
                else:
                    self.append_log("✅ [SAVE_CONFIG] All environment variables appear to be properly set!")

            if show_message:
                from PySide6.QtWidgets import QMessageBox
                from PySide6.QtGui import QIcon
                msg_box = QMessageBox(QMessageBox.Information, "Saved", "Configuration saved.", QMessageBox.Ok)
                try:
                    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "halgakos.ico")
                    if os.path.exists(icon_path):
                        msg_box.setWindowIcon(QIcon(icon_path))
                except Exception: pass
                msg_box.exec()
                
        except Exception as e:
            if show_message:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(None, "Error", f"Failed to save configuration: {e}")
            else:
                print(f"Warning: Config save failed (silent): {e}")
            self._restore_config_from_backup()
        
    def debug_environment_variables(self, show_all=False):
        """Debug and verify all critical environment variables are set correctly.

        Args:
            show_all (bool): If True, shows all environment variables. If False, only shows critical ones.
        """
        # Check if debug mode is enabled
        debug_mode = self.config.get('show_debug_buttons', False)
        
        if debug_mode:
            self.append_log("🔍 [ENV_DEBUG] Starting comprehensive environment variable check...")
        
        # Critical environment variables that should always be set
        critical_env_vars = {
            # Glossary-related
            'GLOSSARY_SYSTEM_PROMPT': 'Manual glossary extraction prompt',
            'AUTO_GLOSSARY_PROMPT': 'Auto glossary generation prompt',
            'APPEND_GLOSSARY_PROMPT': 'Append glossary prompt',
            'GLOSSARY_CUSTOM_ENTRY_TYPES': 'Custom entry types configuration (JSON)',
            'GLOSSARY_CUSTOM_FIELDS': 'Custom glossary fields (JSON)',
            'GLOSSARY_TRANSLATION_PROMPT': 'Glossary translation prompt',
            'GLOSSARY_FORMAT_INSTRUCTIONS': 'Glossary formatting instructions',
            'GLOSSARY_DISABLE_HONORIFICS_FILTER': 'Honorifics filter disable flag',
            'GLOSSARY_STRIP_HONORIFICS': 'Strip honorifics flag',
            'GLOSSARY_FUZZY_THRESHOLD': 'Fuzzy matching threshold',
            'GLOSSARY_USE_LEGACY_CSV': 'Legacy CSV format flag',
            'GLOSSARY_MAX_SENTENCES': 'Maximum sentences for glossary processing',
            
            # OpenRouter settings
            'OPENROUTER_USE_HTTP_ONLY': 'OpenRouter HTTP-only transport',
            'OPENROUTER_ACCEPT_IDENTITY': 'OpenRouter identity encoding',
            'OPENROUTER_PREFERRED_PROVIDER': 'OpenRouter preferred provider',
            
            # General application settings
            'EXTRACTION_WORKERS': 'Number of extraction worker threads',
            'ENABLE_GUI_YIELD': 'GUI yield during processing',
            'RETAIN_SOURCE_EXTENSION': 'Retain source file extension',
            'GLOSSARY_PARALLEL_ENABLED': 'Glossary parallel processing enabled',
            
            # Debug/Logging
            'DEBUG_SAVE_REQUEST_PAYLOADS': 'Save API request payloads',
            'DEBUG_SAVE_REQUEST_PAYLOADS_VERBOSE': 'Verbose payload logging',
            'SHOW_DEBUG_BUTTONS': 'Show debug buttons in UI',
            
            # QA Scanner settings
            'QA_FOREIGN_CHAR_THRESHOLD': 'Foreign character detection threshold',
            'QA_TARGET_LANGUAGE': 'Target language for QA checks',
            'QA_CHECK_ENCODING': 'Check for encoding issues',
            'QA_CHECK_REPETITION': 'Check for repetitive text',
            'QA_CHECK_ARTIFACTS': 'Check for translation artifacts',
            'QA_CHECK_GLOSSARY_LEAKAGE': 'Check for glossary leakage',
            'QA_MIN_FILE_LENGTH': 'Minimum file length for QA',
            'QA_REPORT_FORMAT': 'QA report format',
            'QA_AUTO_SAVE_REPORT': 'Auto-save QA reports',
            'QA_CACHE_ENABLED': 'QA cache enabled',
            'QA_PARAGRAPH_THRESHOLD': 'Paragraph structure threshold',
            'AI_HUNTER_MAX_WORKERS': 'AI Hunter maximum workers',
        }
        
        # Optional/Informational environment variables
        optional_env_vars = {
            # Post-translation scanning phase
            'SCAN_PHASE_ENABLED': 'Enable post-translation scanning phase',
            'SCAN_PHASE_MODE': 'Scanning mode (quick-scan/aggressive/ai-hunter/custom)',
            
            # AI Model settings
            'ENABLE_GEMINI_THINKING': 'Enable Gemini thinking mode',
            'THINKING_BUDGET': 'Gemini thinking budget',
            'ENABLE_GPT_THINKING': 'Enable GPT-4o reasoning',
            'GPT_REASONING_TOKENS': 'GPT reasoning effort tokens',
            'GPT_EFFORT': 'GPT reasoning effort level',
            
            # API Endpoints
            'OPENAI_CUSTOM_BASE_URL': 'Custom OpenAI API base URL',
            'GROQ_API_URL': 'Groq API endpoint',
            'FIREWORKS_API_URL': 'Fireworks API endpoint',
            'USE_CUSTOM_OPENAI_ENDPOINT': 'Use custom OpenAI endpoint',
            'USE_GEMINI_OPENAI_ENDPOINT': 'Use Gemini OpenAI-compatible endpoint',
            'GEMINI_OPENAI_ENDPOINT': 'Gemini OpenAI endpoint URL',
            
            # Image Compression
            'ENABLE_IMAGE_COMPRESSION': 'Enable image compression',
            'AUTO_COMPRESS_ENABLED': 'Auto compress images',
            'TARGET_IMAGE_TOKENS': 'Target image token count',
            'IMAGE_COMPRESSION_FORMAT': 'Image compression format',
            'WEBP_QUALITY': 'WebP quality',
            'JPEG_QUALITY': 'JPEG quality',
            'PNG_COMPRESSION': 'PNG compression level',
            'MAX_IMAGE_DIMENSION': 'Max image dimension',
            'MAX_IMAGE_SIZE_MB': 'Max image size MB',
            'PRESERVE_TRANSPARENCY': 'Preserve image transparency',
            'OPTIMIZE_FOR_OCR': 'Optimize images for OCR',
            'PROGRESSIVE_ENCODING': 'Progressive image encoding',
            'SAVE_COMPRESSED_IMAGES': 'Save compressed images',
            'IMAGE_CHUNK_OVERLAP_PERCENT': 'Image chunk overlap percentage',
            
            # Metadata and Headers
            'TRANSLATE_METADATA_FIELDS': 'Metadata fields to translate (JSON)',
            'METADATA_TRANSLATION_MODE': 'Metadata translation mode',
            'BATCH_TRANSLATE_HEADERS': 'Batch translate headers',
            'HEADERS_PER_BATCH': 'Headers per batch',
            'UPDATE_HTML_HEADERS': 'Update HTML headers',
            'SAVE_HEADER_TRANSLATIONS': 'Save header translations',
            'IGNORE_HEADER': 'Ignore header metadata',
            'IGNORE_TITLE': 'Ignore title metadata',
            
            # Extraction
            'TEXT_EXTRACTION_METHOD': 'Text extraction method',
            'FILE_FILTERING_LEVEL': 'File filtering level',
            'EXTRACTION_MODE': 'Extraction mode',
            'ENHANCED_FILTERING': 'Enhanced filtering level',
            
            # Anti-Duplicate
            'ENABLE_ANTI_DUPLICATE': 'Enable anti-duplicate measures',
            'TOP_P': 'Top-P sampling parameter',
            'TOP_K': 'Top-K sampling parameter',
            'FREQUENCY_PENALTY': 'Frequency penalty',
            'PRESENCE_PENALTY': 'Presence penalty',
            'REPETITION_PENALTY': 'Repetition penalty',
            'CANDIDATE_COUNT': 'Candidate count',
            'CUSTOM_STOP_SEQUENCES': 'Custom stop sequences',
            'LOGIT_BIAS_ENABLED': 'Logit bias enabled',
            'LOGIT_BIAS_STRENGTH': 'Logit bias strength',
            'BIAS_COMMON_WORDS': 'Bias against common words',
            'BIAS_REPETITIVE_PHRASES': 'Bias against repetitive phrases',
            
            # Azure
            'AZURE_API_VERSION': 'Azure API version',
            
            # Fallback Keys
            'USE_FALLBACK_KEYS': 'Use fallback API keys',
            'FALLBACK_KEYS': 'Fallback API keys (JSON)',
            
            # Manga Integration and Manga Settings Dialog variables
            'MANGA_FULL_PAGE_CONTEXT': 'Enable full page context translation',
            'MANGA_VISUAL_CONTEXT_ENABLED': 'Include page image in requests',
            'MANGA_CREATE_SUBFOLDER': "Create 'translated' subfolder for output",
            'MANGA_BG_OPACITY': 'Background opacity (0-255)',
            'MANGA_BG_STYLE': 'Background style (box/circle/wrap)',
            'MANGA_BG_REDUCTION': 'Background reduction factor',
            'MANGA_FONT_SIZE': 'Fixed font size (0=auto)',
            'MANGA_FONT_STYLE': 'Font style name',
            'MANGA_FONT_PATH': 'Selected font path',
            'MANGA_FONT_SIZE_MODE': 'Font size mode (fixed/multiplier)',
            'MANGA_FONT_SIZE_MULTIPLIER': 'Font size multiplier (for multiplier mode)',
            'MANGA_MAX_FONT_SIZE': 'Maximum font size',
            'MANGA_AUTO_MIN_SIZE': 'Automatic minimum readable font size',
            'MANGA_FREE_TEXT_ONLY_BG_OPACITY': 'Apply BG opacity only to free text',
            'MANGA_FORCE_CAPS_LOCK': 'Force caps lock',
            'MANGA_STRICT_TEXT_WRAPPING': 'Strict text wrapping (force fit)',
            'MANGA_CONSTRAIN_TO_BUBBLE': 'Constrain text to bubble bounds',
            'MANGA_TEXT_COLOR': 'Text color RGB (R,G,B)',
            'MANGA_SHADOW_ENABLED': 'Shadow enabled',
            'MANGA_SHADOW_COLOR': 'Shadow color RGB (R,G,B)',
            'MANGA_SHADOW_OFFSET_X': 'Shadow offset X',
            'MANGA_SHADOW_OFFSET_Y': 'Shadow offset Y',
            'MANGA_SHADOW_BLUR': 'Shadow blur radius',
            'MANGA_SKIP_INPAINTING': 'Skip inpainting',
            'MANGA_INPAINT_QUALITY': 'Inpainting quality preset',
            'MANGA_INPAINT_DILATION': 'Inpainting dilation (px)',
            'MANGA_INPAINT_PASSES': 'Inpainting passes',
            'MANGA_INPAINT_METHOD': 'Inpainting method (local/cloud/hybrid/skip)',
            'MANGA_LOCAL_INPAINT_METHOD': 'Local inpainting model type',
            'MANGA_FONT_ALGORITHM': 'Font sizing algorithm preset',
            'MANGA_PREFER_LARGER': 'Prefer larger font sizing',
            'MANGA_BUBBLE_SIZE_FACTOR': 'Use bubble size factor for sizing',
            'MANGA_LINE_SPACING': 'Line spacing multiplier',
            'MANGA_MAX_LINES': 'Maximum lines per bubble',
            'MANGA_QWEN2VL_MODEL_SIZE': 'Qwen2-VL model size selection',
            'MANGA_RAPIDOCR_USE_RECOGNITION': 'RapidOCR: use recognition step',
            'MANGA_RAPIDOCR_LANGUAGE': 'RapidOCR detection language',
            'MANGA_RAPIDOCR_DETECTION_MODE': 'RapidOCR detection mode',
            'MANGA_FULL_PAGE_CONTEXT_PROMPT_LEN': 'Length of full page context prompt',
            'MANGA_OCR_PROMPT_LEN': 'Length of OCR system prompt',
            # Manga Advanced Settings (Memory Management)
            'MANGA_AUTO_CLEANUP_MODELS': 'Auto cleanup models after translation',
            'MANGA_UNLOAD_MODELS_AFTER_TRANSLATION': 'Unload models after translation (reset instance)',
            'MANGA_USE_SINGLETON_MODELS': 'Use singleton model instances',
            'MANGA_PARALLEL_PROCESSING': 'Enable parallel processing',
            'MANGA_MAX_WORKERS': 'Maximum worker threads',
            'MANGA_PARALLEL_PANEL_TRANSLATION': 'Enable parallel panel translation',
            'MANGA_PANEL_MAX_WORKERS': 'Maximum concurrent panels',
            'MANGA_DEBUG_MODE': 'Manga debug mode',
            'MANGA_SAVE_INTERMEDIATE': 'Save intermediate debug images',
            'MANGA_CONCISE_LOGS': 'Concise pipeline logs (suppress verbose steps)',
            'MANGA_SKIP_INPAINTING': 'Skip inpainting step (show detected bubbles only)',
        }
        
        # Check critical variables
        missing_critical = []
        empty_critical = []
        set_critical = []
        
        for var_name, description in critical_env_vars.items():
            value = os.environ.get(var_name)
            
            if value is None:
                missing_critical.append(var_name)
                if debug_mode:
                    self.append_log(f"❌ [ENV_DEBUG] CRITICAL MISSING: {var_name} - {description}")
            elif not value.strip():
                empty_critical.append(var_name)
                if debug_mode:
                    self.append_log(f"⚠️ [ENV_DEBUG] CRITICAL EMPTY: {var_name} - {description}")
            else:
                set_critical.append(var_name)
                if debug_mode:
                    value_preview = str(value)[:100] + ('...' if len(str(value)) > 100 else '')
                    self.append_log(f"✅ [ENV_DEBUG] {var_name}: {value_preview}")
        
        # Treat previous 'optional' as critical as well
        for var_name, description in optional_env_vars.items():
            value = os.environ.get(var_name)
            if value is None:
                missing_critical.append(var_name)
                if debug_mode:
                    self.append_log(f"❌ [ENV_DEBUG] CRITICAL MISSING: {var_name} - {description}")
            elif not str(value).strip():
                empty_critical.append(var_name)
                if debug_mode:
                    self.append_log(f"⚠️ [ENV_DEBUG] CRITICAL EMPTY: {var_name} - {description}")
            else:
                set_critical.append(var_name)
                if debug_mode:
                    value_preview = str(value)[:100] + ('...' if len(str(value)) > 100 else '')
                    self.append_log(f"✅ [ENV_DEBUG] {var_name}: {value_preview}")
        
        # Summary (now includes all former optional variables)
        total_critical = len(critical_env_vars) + len(optional_env_vars)
        if debug_mode:
            self.append_log(f"🔍 [ENV_DEBUG] Summary: {len(set_critical)}/{total_critical} critical vars set")
        
        if missing_critical and debug_mode:
            self.append_log(f"❌ [ENV_DEBUG] {len(missing_critical)} MISSING: {', '.join(missing_critical)}")
            
        if empty_critical and debug_mode:
            self.append_log(f"⚠️ [ENV_DEBUG] {len(empty_critical)} EMPTY: {', '.join(empty_critical)}")
            
        # Check for initialization issues
        if missing_critical or empty_critical:
            if debug_mode:
                self.append_log("❌ [ENV_DEBUG] RECOMMENDATION: Some variables are not initialized!")
                self.append_log("🔧 [ENV_DEBUG] Try calling self.initialize_environment_variables() on startup")
            return False
        else:
            if debug_mode:
                self.append_log("✅ [ENV_DEBUG] All critical environment variables are properly set")
            return True
    
    def initialize_environment_variables(self):
        """Initialize all environment variables from config on startup.
        Call this method during application initialization to ensure all environment variables are set.
        """
        # Check if debug mode is enabled
        debug_mode = self.config.get('show_debug_buttons', False)
        
        if debug_mode:
            self.append_log("🚀 [INIT] Initializing all environment variables from config...")
        
        # Wire verbose payload saving to GUI debug mode
        try:
            os.environ['DEBUG_SAVE_REQUEST_PAYLOADS_VERBOSE'] = '1' if debug_mode else '0'
            # Also reflect debug mode for the client
            os.environ['SHOW_DEBUG_BUTTONS'] = '1' if debug_mode else '0'
            # Ensure capture itself is enabled
            os.environ['DEBUG_SAVE_REQUEST_PAYLOADS'] = '1'
            if debug_mode:
                self.append_log("🔍 [INIT] Verbose payload logging enabled (DEBUG_SAVE_REQUEST_PAYLOADS_VERBOSE=1)")
                self.append_log("🔍 [INIT] Definitive payload capture enabled (DEBUG_SAVE_REQUEST_PAYLOADS=1)")
        except Exception:
            pass
        
        try:
            # Initialize glossary-related environment variables
            env_mappings = [
                ('GLOSSARY_SYSTEM_PROMPT', self.config.get('manual_glossary_prompt', getattr(self, 'manual_glossary_prompt', ''))),
                ('AUTO_GLOSSARY_PROMPT', self.config.get('auto_glossary_prompt', getattr(self, 'auto_glossary_prompt', ''))),
                ('GLOSSARY_DISABLE_HONORIFICS_FILTER', '1' if self.config.get('glossary_disable_honorifics_filter', False) else '0'),
                ('GLOSSARY_STRIP_HONORIFICS', '1' if self.config.get('strip_honorifics', False) else '0'),
                ('GLOSSARY_FUZZY_THRESHOLD', str(self.config.get('glossary_fuzzy_threshold', 0.90))),
                ('GLOSSARY_TRANSLATION_PROMPT', self.config.get('glossary_translation_prompt', '')),
                ('GLOSSARY_FORMAT_INSTRUCTIONS', self.config.get('glossary_format_instructions', '')),
                ('GLOSSARY_USE_LEGACY_CSV', '1' if self.config.get('glossary_use_legacy_csv', False) else '0'),
                ('GLOSSARY_MAX_SENTENCES', str(self.config.get('glossary_max_sentences', 10))),
                
                # OpenRouter settings
                ('OPENROUTER_USE_HTTP_ONLY', '1' if self.config.get('openrouter_use_http_only', False) else '0'),
                ('OPENROUTER_ACCEPT_IDENTITY', '1' if self.config.get('openrouter_accept_identity', False) else '0'),
                ('OPENROUTER_PREFERRED_PROVIDER', self.config.get('openrouter_preferred_provider', '')),
                
                # General settings
                ('EXTRACTION_WORKERS', str(self.config.get('extraction_workers', 1)) if self.config.get('enable_parallel_extraction', False) else '1'),
                ('ENABLE_GUI_YIELD', '1' if self.config.get('enable_gui_yield', True) else '0'),
                ('RETAIN_SOURCE_EXTENSION', '1' if self.config.get('retain_source_extension', False) else '0'),
            ]
            
            # Add QA Scanner environment variables
            qa_settings = self.config.get('qa_scanner_settings', {})
            ai_hunter_config = self.config.get('ai_hunter_config', {})
            qa_env_mappings = [
                ('QA_FOREIGN_CHAR_THRESHOLD', str(qa_settings.get('foreign_char_threshold', 10))),
                ('QA_TARGET_LANGUAGE', qa_settings.get('target_language', 'english')),
                ('QA_CHECK_ENCODING', '1' if qa_settings.get('check_encoding_issues', False) else '0'),
                ('QA_CHECK_REPETITION', '1' if qa_settings.get('check_repetition', True) else '0'),
                ('QA_CHECK_ARTIFACTS', '1' if qa_settings.get('check_translation_artifacts', False) else '0'),
                ('QA_CHECK_GLOSSARY_LEAKAGE', '1' if qa_settings.get('check_glossary_leakage', True) else '0'),
                ('QA_MIN_FILE_LENGTH', str(qa_settings.get('min_file_length', 0))),
                ('QA_REPORT_FORMAT', qa_settings.get('report_format', 'detailed')),
                ('QA_AUTO_SAVE_REPORT', '1' if qa_settings.get('auto_save_report', True) else '0'),
                ('QA_CACHE_ENABLED', '1' if qa_settings.get('cache_enabled', True) else '0'),
                ('QA_PARAGRAPH_THRESHOLD', str(qa_settings.get('paragraph_threshold', 0.3))),
                ('AI_HUNTER_MAX_WORKERS', str(ai_hunter_config.get('ai_hunter_max_workers', 1))),
            ]
            
            # Add Manga Integration and Manga Settings Dialog environment variables
            ms = self.config.get('manga_settings', {}) if isinstance(self.config.get('manga_settings', {}), dict) else {}
            inpaint = ms.get('inpainting', {}) if isinstance(ms.get('inpainting', {}), dict) else {}
            rendering = ms.get('rendering', {}) if isinstance(ms.get('rendering', {}), dict) else {}
            font_cfg = ms.get('font_sizing', {}) if isinstance(ms.get('font_sizing', {}), dict) else {}

            # Convenience getters with fallbacks to top-level keys used by MangaIntegration
            def _rgb_list_to_str(lst, default):
                try:
                    if isinstance(lst, (list, tuple)) and len(lst) == 3:
                        return f"{int(lst[0])},{int(lst[1])},{int(lst[2])}"
                except Exception:
                    pass
                return default

            manga_env_mappings = [
                ('MANGA_FULL_PAGE_CONTEXT', '1' if self.config.get('manga_full_page_context', False) else '0'),
                ('MANGA_VISUAL_CONTEXT_ENABLED', '1' if self.config.get('manga_visual_context_enabled', True) else '0'),
                ('MANGA_CREATE_SUBFOLDER', '1' if self.config.get('manga_create_subfolder', True) else '0'),
                ('MANGA_BG_OPACITY', str(self.config.get('manga_bg_opacity', 130))),
                ('MANGA_BG_STYLE', str(self.config.get('manga_bg_style', 'circle'))),
                ('MANGA_BG_REDUCTION', str(self.config.get('manga_bg_reduction', 1.0))),
                ('MANGA_FONT_SIZE', str(self.config.get('manga_font_size', 0))),
                ('MANGA_FONT_STYLE', str(self.config.get('manga_font_style', 'Default'))),
                ('MANGA_FONT_PATH', str(self.config.get('manga_font_path', ''))),
                ('MANGA_FONT_SIZE_MODE', str(self.config.get('manga_font_size_mode', 'fixed'))),
                ('MANGA_FONT_SIZE_MULTIPLIER', str(self.config.get('manga_font_size_multiplier', 1.0))),
                ('MANGA_MAX_FONT_SIZE', str(self.config.get('manga_max_font_size', rendering.get('auto_max_size', font_cfg.get('max_size', 48))))),
                ('MANGA_AUTO_MIN_SIZE', str(rendering.get('auto_min_size', font_cfg.get('min_size', 10)))),
                ('MANGA_FREE_TEXT_ONLY_BG_OPACITY', '1' if self.config.get('manga_free_text_only_bg_opacity', True) else '0'),
                ('MANGA_FORCE_CAPS_LOCK', '1' if self.config.get('manga_force_caps_lock', True) else '0'),
                ('MANGA_STRICT_TEXT_WRAPPING', '1' if self.config.get('manga_strict_text_wrapping', True) else '0'),
                ('MANGA_CONSTRAIN_TO_BUBBLE', '1' if self.config.get('manga_constrain_to_bubble', True) else '0'),
                ('MANGA_TEXT_COLOR', _rgb_list_to_str(self.config.get('manga_text_color', [102,0,0]), '102,0,0')),
                ('MANGA_SHADOW_ENABLED', '1' if self.config.get('manga_shadow_enabled', True) else '0'),
                ('MANGA_SHADOW_COLOR', _rgb_list_to_str(self.config.get('manga_shadow_color', [204,128,128]), '204,128,128')),
                ('MANGA_SHADOW_OFFSET_X', str(self.config.get('manga_shadow_offset_x', 2))),
                ('MANGA_SHADOW_OFFSET_Y', str(self.config.get('manga_shadow_offset_y', 2))),
                ('MANGA_SHADOW_BLUR', str(self.config.get('manga_shadow_blur', 0))),
                ('MANGA_SKIP_INPAINTING', '1' if self.config.get('manga_skip_inpainting', False) else '0'),
                ('MANGA_INPAINT_QUALITY', str(self.config.get('manga_inpaint_quality', 'high'))),
                ('MANGA_INPAINT_DILATION', str(self.config.get('manga_inpaint_dilation', 15))),
                ('MANGA_INPAINT_PASSES', str(self.config.get('manga_inpaint_passes', 2))),
                ('MANGA_INPAINT_METHOD', str(inpaint.get('method', 'local'))),
                ('MANGA_LOCAL_INPAINT_METHOD', str(inpaint.get('local_method', 'anime_onnx'))),
                ('MANGA_FONT_ALGORITHM', str(font_cfg.get('algorithm', 'smart'))),
                ('MANGA_PREFER_LARGER', '1' if font_cfg.get('prefer_larger', True) else '0'),
                ('MANGA_BUBBLE_SIZE_FACTOR', '1' if font_cfg.get('bubble_size_factor', True) else '0'),
                ('MANGA_LINE_SPACING', str(font_cfg.get('line_spacing', 1.3))),
                ('MANGA_MAX_LINES', str(font_cfg.get('max_lines', 10))),
                ('MANGA_QWEN2VL_MODEL_SIZE', str(self.config.get('qwen2vl_model_size', '1'))),
                ('MANGA_RAPIDOCR_USE_RECOGNITION', '1' if self.config.get('rapidocr_use_recognition', True) else '0'),
                ('MANGA_RAPIDOCR_LANGUAGE', str(self.config.get('rapidocr_language', 'auto'))),
                ('MANGA_RAPIDOCR_DETECTION_MODE', str(self.config.get('rapidocr_detection_mode', 'document'))),
                # Prompt lengths for quick sanity without leaking content
                ('MANGA_FULL_PAGE_CONTEXT_PROMPT_LEN', str(len(self.config.get('manga_full_page_context_prompt', '') or ''))),
                ('MANGA_OCR_PROMPT_LEN', str(len(self.config.get('manga_ocr_prompt', '') or ''))),
            ]
            
            # Add Manga Advanced Settings (Memory Management)
            manga_adv = ms.get('advanced', {}) if isinstance(ms.get('advanced', {}), dict) else {}
            manga_advanced_env_mappings = [
                ('MANGA_AUTO_CLEANUP_MODELS', '1' if manga_adv.get('auto_cleanup_models', False) else '0'),
                ('MANGA_UNLOAD_MODELS_AFTER_TRANSLATION', '1' if manga_adv.get('unload_models_after_translation', False) else '0'),
                ('MANGA_USE_SINGLETON_MODELS', '1' if manga_adv.get('use_singleton_models', True) else '0'),
                ('MANGA_PARALLEL_PROCESSING', '1' if manga_adv.get('parallel_processing', False) else '0'),
                ('MANGA_MAX_WORKERS', str(manga_adv.get('max_workers', 4))),
                ('MANGA_PARALLEL_PANEL_TRANSLATION', '1' if manga_adv.get('parallel_panel_translation', False) else '0'),
                ('MANGA_PANEL_MAX_WORKERS', str(manga_adv.get('panel_max_workers', 2))),
                ('MANGA_DEBUG_MODE', '1' if manga_adv.get('debug_mode', False) else '0'),
                ('MANGA_SAVE_INTERMEDIATE', '1' if manga_adv.get('save_intermediate', False) else '0'),
                ('MANGA_CONCISE_LOGS', '1' if manga_adv.get('concise_logs', True) else '0'),
                # Note: MANGA_SKIP_INPAINTING is set from manga_skip_inpainting (line 8951) - don't duplicate here
            ]

            # Combine all environment variable mappings
            env_mappings.extend(qa_env_mappings)
            env_mappings.extend(manga_env_mappings)
            env_mappings.extend(manga_advanced_env_mappings)

            # Add additional environment variables converted from legacy Tkinter to PySide6 attributes
            try:
                import json as _json
            except Exception:
                _json = json
            extra_env_mappings = [
                # Rolling summary
                ('USE_ROLLING_SUMMARY', '1' if getattr(self, 'rolling_summary_var', False) else '0'),
                ('SUMMARY_ROLE', getattr(self, 'summary_role_var', 'user')),
                ('ROLLING_SUMMARY_EXCHANGES', str(getattr(self, 'rolling_summary_exchanges_var', '5'))),
                ('ROLLING_SUMMARY_MODE', getattr(self, 'rolling_summary_mode_var', 'append')),
                ('ROLLING_SUMMARY_SYSTEM_PROMPT', getattr(self, 'rolling_summary_system_prompt', getattr(self, 'default_rolling_summary_system_prompt', ''))),
                ('ROLLING_SUMMARY_USER_PROMPT', getattr(self, 'rolling_summary_user_prompt', getattr(self, 'default_rolling_summary_user_prompt', ''))),
                ('ROLLING_SUMMARY_MAX_ENTRIES', str(getattr(self, 'rolling_summary_max_entries_var', '10'))),

                # Retry/network controls
                ('RETRY_TRUNCATED', '1' if getattr(self, 'retry_truncated_var', False) else '0'),
                ('MAX_RETRY_TOKENS', str(getattr(self, 'max_retry_tokens_var', '16384'))),
                ('RETRY_DUPLICATE_BODIES', '1' if getattr(self, 'retry_duplicate_var', False) else '0'),
                ('DUPLICATE_LOOKBACK_CHAPTERS', str(getattr(self, 'duplicate_lookback_var', '5'))),
                ('RETRY_TIMEOUT', '1' if getattr(self, 'retry_timeout_var', True) else '0'),
                ('CHUNK_TIMEOUT', str(getattr(self, 'chunk_timeout_var', '900'))),
                ('ENABLE_HTTP_TUNING', '1' if self.config.get('enable_http_tuning', False) else '0'),
                ('CONNECT_TIMEOUT', str(getattr(self, 'connect_timeout_var', '10'))),
                ('READ_TIMEOUT', str(getattr(self, 'read_timeout_var', '180'))),
                ('HTTP_POOL_CONNECTIONS', str(getattr(self, 'http_pool_connections_var', '20'))),
                ('HTTP_POOL_MAXSIZE', str(getattr(self, 'http_pool_maxsize_var', '50'))),
                ('IGNORE_RETRY_AFTER', '1' if self.config.get('ignore_retry_after', False) else '0'),
                ('MAX_RETRIES', str(getattr(self, 'max_retries_var', '7'))),

                # QA/meta preferences
                ('QA_AUTO_SEARCH_OUTPUT', '1' if getattr(self, 'qa_auto_search_output_var', True) else '0'),
                ('INDEFINITE_RATE_LIMIT_RETRY', '1' if getattr(self, 'indefinite_rate_limit_retry_var', True) else '0'),
                ('REINFORCEMENT_FREQUENCY', str(getattr(self, 'reinforcement_freq_var', '10'))),
                # Post-translation scanning phase
                ('SCAN_PHASE_ENABLED', '1' if getattr(self, 'scan_phase_enabled_var', False) else '0'),
                ('SCAN_PHASE_MODE', getattr(self, 'scan_phase_mode_var', 'quick-scan')),

                # Book title handling
                ('TRANSLATE_BOOK_TITLE', '1' if getattr(self, 'translate_book_title_var', True) else '0'),
                ('BOOK_TITLE_PROMPT', getattr(self, 'book_title_prompt', '')),

                # Safety/merge toggles
                ('EMERGENCY_PARAGRAPH_RESTORE', '1' if getattr(self, 'emergency_restore_var', False) else '0'),
                ('DISABLE_CHAPTER_MERGING', '1' if getattr(self, 'disable_chapter_merging_var', False) else '0'),

                # Image translation controls
                ('ENABLE_IMAGE_TRANSLATION', '1' if getattr(self, 'enable_image_translation_var', False) else '0'),
                ('PROCESS_WEBNOVEL_IMAGES', '1' if getattr(self, 'process_webnovel_images_var', True) else '0'),
                ('WEBNOVEL_MIN_HEIGHT', str(getattr(self, 'webnovel_min_height_var', '1000'))),
                ('MAX_IMAGES_PER_CHAPTER', str(getattr(self, 'max_images_per_chapter_var', '1'))),
                ('IMAGE_CHUNK_HEIGHT', str(getattr(self, 'image_chunk_height_var', '1500'))),
                ('HIDE_IMAGE_TRANSLATION_LABEL', '1' if getattr(self, 'hide_image_translation_label_var', True) else '0'),
                ('DISABLE_EPUB_GALLERY', '1' if getattr(self, 'disable_epub_gallery_var', False) else '0'),
                ('DISABLE_AUTOMATIC_COVER_CREATION', '1' if getattr(self, 'disable_automatic_cover_creation_var', False) else '0'),
                # New: Translate special files (cover, nav, toc, message, etc.)
                ('TRANSLATE_SPECIAL_FILES', '1' if getattr(self, 'translate_special_files_var', False) else '0'),
                # Backward compatibility: Also set the old TRANSLATE_COVER_HTML for any legacy code
                ('TRANSLATE_COVER_HTML', '1' if getattr(self, 'translate_special_files_var', False) else '0'),
                ('DISABLE_ZERO_DETECTION', '1' if getattr(self, 'disable_zero_detection_var', True) else '0'),
                ('DUPLICATE_DETECTION_MODE', getattr(self, 'duplicate_detection_mode_var', 'basic')),
                ('ENABLE_DECIMAL_CHAPTERS', '1' if getattr(self, 'enable_decimal_chapters_var', False) else '0'),

                # Watermark/image toggles
                ('ENABLE_WATERMARK_REMOVAL', '1' if getattr(self, 'enable_watermark_removal_var', True) else '0'),
                ('SAVE_CLEANED_IMAGES', '1' if getattr(self, 'save_cleaned_images_var', False) else '0'),

                # Prompts
                ('TRANSLATION_CHUNK_PROMPT', str(getattr(self, 'translation_chunk_prompt', ''))),
                ('IMAGE_CHUNK_PROMPT', str(getattr(self, 'image_chunk_prompt', ''))),

                # Safety flags
                ('DISABLE_GEMINI_SAFETY', str(self.config.get('disable_gemini_safety', False)).lower()),

                # OpenRouter (duplicates are okay; ensures presence)
                ('OPENROUTER_USE_HTTP_ONLY', '1' if getattr(self, 'openrouter_http_only_var', False) else '0'),
                ('OPENROUTER_ACCEPT_IDENTITY', '1' if getattr(self, 'openrouter_accept_identity_var', False) else '0'),

                # Misc toggles
                ('auto_update_check', str(getattr(self, 'auto_update_check_var', True))),
                ('FORCE_NCX_ONLY', '1' if getattr(self, 'force_ncx_only_var', True) else '0'),
                ('SINGLE_API_IMAGE_CHUNKS', '1' if getattr(self, 'single_api_image_chunks_var', False) else '0'),

                # Thinking features
                ('ENABLE_GEMINI_THINKING', '1' if getattr(self, 'enable_gemini_thinking_var', True) else '0'),
                ('THINKING_BUDGET', str(getattr(self, 'thinking_budget_var', '-1')) if getattr(self, 'enable_gemini_thinking_var', True) else '0'),
                ('ENABLE_GPT_THINKING', '1' if getattr(self, 'enable_gpt_thinking_var', True) else '0'),
                ('GPT_REASONING_TOKENS', str(getattr(self, 'gpt_reasoning_tokens_var', '2000')) if getattr(self, 'enable_gpt_thinking_var', True) else ''),
                ('GPT_EFFORT', getattr(self, 'gpt_effort_var', 'medium')),

                # Custom API endpoints
                ('OPENAI_CUSTOM_BASE_URL', getattr(self, 'openai_base_url_var', '')),
                ('GROQ_API_URL', getattr(self, 'groq_base_url_var', '')),
                ('FIREWORKS_API_URL', getattr(self, 'fireworks_base_url_var', '')),
                ('USE_CUSTOM_OPENAI_ENDPOINT', '1' if getattr(self, 'use_custom_openai_endpoint_var', False) else '0'),
                ('USE_GEMINI_OPENAI_ENDPOINT', '1' if getattr(self, 'use_gemini_openai_endpoint_var', False) else '0'),
                ('GEMINI_OPENAI_ENDPOINT', getattr(self, 'gemini_openai_endpoint_var', '')),

                # Image compression settings
                ('ENABLE_IMAGE_COMPRESSION', '1' if getattr(self, 'enable_image_compression_var', False) else '0'),
                ('AUTO_COMPRESS_ENABLED', '1' if getattr(self, 'auto_compress_enabled_var', True) else '0'),
                ('TARGET_IMAGE_TOKENS', str(getattr(self, 'target_image_tokens_var', '1000'))),
                ('IMAGE_COMPRESSION_FORMAT', getattr(self, 'image_format_var', 'auto')),
                ('WEBP_QUALITY', str(getattr(self, 'webp_quality_var', 85))),
                ('JPEG_QUALITY', str(getattr(self, 'jpeg_quality_var', 85))),
                ('PNG_COMPRESSION', str(getattr(self, 'png_compression_var', 6))),
                ('MAX_IMAGE_DIMENSION', str(getattr(self, 'max_image_dimension_var', '2048'))),
                ('MAX_IMAGE_SIZE_MB', str(getattr(self, 'max_image_size_mb_var', '10'))),
                ('PRESERVE_TRANSPARENCY', '1' if getattr(self, 'preserve_transparency_var', False) else '0'),
                ('OPTIMIZE_FOR_OCR', '1' if getattr(self, 'optimize_for_ocr_var', True) else '0'),
                ('PROGRESSIVE_ENCODING', '1' if getattr(self, 'progressive_encoding_var', True) else '0'),
                ('SAVE_COMPRESSED_IMAGES', '1' if getattr(self, 'save_compressed_images_var', False) else '0'),
                ('USE_FALLBACK_KEYS', '1' if getattr(self, 'use_fallback_keys_var', False) else '0'),
                ('FALLBACK_KEYS', _json.dumps(self.config.get('fallback_keys', []))),
                ('IMAGE_CHUNK_OVERLAP_PERCENT', str(getattr(self, 'image_chunk_overlap_var', '1'))),

                # Metadata and batch header settings
                ('TRANSLATE_METADATA_FIELDS', _json.dumps(getattr(self, 'translate_metadata_fields', {}))),
                ('METADATA_TRANSLATION_MODE', self.config.get('metadata_translation_mode', 'together')),
                ('BATCH_TRANSLATE_HEADERS', '1' if getattr(self, 'batch_translate_headers_var', False) else '0'),
                ('HEADERS_PER_BATCH', str(getattr(self, 'headers_per_batch_var', '400'))),
                ('UPDATE_HTML_HEADERS', '1' if getattr(self, 'update_html_headers_var', True) else '0'),
                ('SAVE_HEADER_TRANSLATIONS', '1' if getattr(self, 'save_header_translations_var', True) else '0'),
                ('IGNORE_HEADER', '1' if getattr(self, 'ignore_header_var', False) else '0'),
                ('IGNORE_TITLE', '1' if getattr(self, 'ignore_title_var', False) else '0'),

                # Extraction mode
                ('TEXT_EXTRACTION_METHOD', getattr(self, 'text_extraction_method_var', 'standard') if hasattr(self, 'text_extraction_method_var') else 'standard'),
                ('FILE_FILTERING_LEVEL', getattr(self, 'file_filtering_level_var', 'smart') if hasattr(self, 'file_filtering_level_var') else 'smart'),
                ('EXTRACTION_MODE', getattr(self, 'extraction_mode_var', 'smart')),
                ('ENHANCED_FILTERING', getattr(self, 'enhanced_filtering_var', 'smart')),

                # Anti-duplicate
                ('ENABLE_ANTI_DUPLICATE', '1' if getattr(self, 'enable_anti_duplicate_var', False) else '0'),
                ('TOP_P', str(getattr(self, 'top_p_var', '1.0'))),
                ('TOP_K', str(getattr(self, 'top_k_var', '0'))),
                ('FREQUENCY_PENALTY', str(getattr(self, 'frequency_penalty_var', '0.0'))),
                ('PRESENCE_PENALTY', str(getattr(self, 'presence_penalty_var', '0.0'))),
                ('REPETITION_PENALTY', str(getattr(self, 'repetition_penalty_var', '1.0'))),
                ('CANDIDATE_COUNT', str(getattr(self, 'candidate_count_var', '1'))),
                ('CUSTOM_STOP_SEQUENCES', getattr(self, 'custom_stop_sequences_var', '')),
                ('LOGIT_BIAS_ENABLED', '1' if getattr(self, 'logit_bias_enabled_var', False) else '0'),
                ('LOGIT_BIAS_STRENGTH', str(getattr(self, 'logit_bias_strength_var', '-0.5'))),
                ('BIAS_COMMON_WORDS', '1' if getattr(self, 'bias_common_words_var', False) else '0'),
                ('BIAS_REPETITIVE_PHRASES', '1' if getattr(self, 'bias_repetitive_phrases_var', False) else '0'),

                # Azure API version
                ('AZURE_API_VERSION', self.config.get('azure_api_version', '2025-01-01-preview')),
            ]

            env_mappings.extend(extra_env_mappings)
            
            initialized_count = 0
            for env_key, env_value in env_mappings:
                try:
                    old_value = os.environ.get(env_key, '<NOT SET>')
                    os.environ[env_key] = str(env_value) if env_value is not None else ''
                    new_value = os.environ[env_key]
                    
                    if old_value != new_value and debug_mode:
                        self.append_log(f"🔍 [INIT] ENV {env_key}: '{old_value}' → '{new_value[:50]}{'...' if len(str(new_value)) > 50 else ''}'")
                    
                    initialized_count += 1
                except Exception as e:
                    if debug_mode:
                        self.append_log(f"❌ [INIT] Failed to initialize {env_key}: {e}")
            
            # JSON environment variables
            try:
                # Prefer in-memory types, then config, then sensible defaults
                custom_entry_types = getattr(self, 'custom_entry_types', None)
                if not custom_entry_types:
                    custom_entry_types = self.config.get('custom_entry_types')
                if not custom_entry_types:
                    custom_entry_types = {
                        'character': {'enabled': True, 'has_gender': True},
                        'term': {'enabled': True, 'has_gender': False}
                    }
                custom_types_json = json.dumps(custom_entry_types)
                os.environ['GLOSSARY_CUSTOM_ENTRY_TYPES'] = custom_types_json
                if debug_mode:
                    self.append_log(f"🔍 [INIT] ENV GLOSSARY_CUSTOM_ENTRY_TYPES: {len(custom_types_json)} chars")
                initialized_count += 1
            except Exception as e:
                if debug_mode:
                    self.append_log(f"❌ [INIT] Failed to initialize GLOSSARY_CUSTOM_ENTRY_TYPES: {e}")
            
            try:
                custom_glossary_fields = self.config.get('custom_glossary_fields', [])
                if custom_glossary_fields:
                    custom_fields_json = json.dumps(custom_glossary_fields)
                    os.environ['GLOSSARY_CUSTOM_FIELDS'] = custom_fields_json
                    if debug_mode:
                        self.append_log(f"🔍 [INIT] ENV GLOSSARY_CUSTOM_FIELDS: {len(custom_fields_json)} chars")
                    initialized_count += 1
            except Exception as e:
                if debug_mode:
                    self.append_log(f"❌ [INIT] Failed to initialize GLOSSARY_CUSTOM_FIELDS: {e}")
                
            if debug_mode:
                self.append_log(f"✅ [INIT] Successfully initialized {initialized_count} environment variables")
            
            # Verify initialization (optional - don't fail if debug method doesn't exist)
            try:
                return self.debug_environment_variables(show_all=False)
            except AttributeError:
                # Method doesn't exist (e.g., in test mocks), return True since variables were set
                if debug_mode:
                    self.append_log("✅ [INIT] Environment variables initialized successfully (debug verification skipped)")
                return True
            
        except Exception as e:
            if debug_mode:
                self.append_log(f"❌ [INIT] Environment variable initialization failed: {e}")
                import traceback
                self.append_log(f"❌ [INIT] Traceback: {traceback.format_exc()}")
            return False
    
    def _ensure_executor(self):
        """Ensure a ThreadPoolExecutor exists and matches configured worker count.
        Also updates EXTRACTION_WORKERS environment variable.
        """
        try:
            workers = 1
            try:
                workers = int(self.extraction_workers_var) if self.enable_parallel_extraction_var else 1
            except Exception:
                workers = 1
            if workers < 1:
                workers = 1
            os.environ["EXTRACTION_WORKERS"] = str(workers)
            
            # If executor exists with same worker count, keep it
            if getattr(self, 'executor', None) and getattr(self, '_executor_workers', None) == workers:
                return
            
            # If executor exists but tasks are running, don't recreate to avoid disruption
            active = any([
                getattr(self, 'translation_future', None) and not self.translation_future.done(),
                getattr(self, 'glossary_future', None) and not self.glossary_future.done(),
                getattr(self, 'epub_future', None) and not self.epub_future.done(),
                getattr(self, 'qa_future', None) and not self.qa_future.done(),
            ])
            if getattr(self, 'executor', None) and active:
                self._executor_workers = workers  # Remember desired workers for later
                return
            
            # Safe to (re)create
            if getattr(self, 'executor', None):
                try:
                    self.executor.shutdown(wait=False)
                except Exception:
                    pass
                self.executor = None
            
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="Glossarion")
            self._executor_workers = workers
        except Exception as e:
            try:
                print(f"Executor setup failed: {e}")
            except Exception:
                pass
    
    def log_debug(self, message):
        self.append_log(f"[DEBUG] {message}")
    
    def _sync_gui_to_config(self):
        """Sync current GUI widget values using comprehensive environment variables method
        The _get_environment_variables method already reads all GUI widgets comprehensively
        """
        try:
            # Get current API key for the environment variables method
            api_key = ''
            if hasattr(self, 'api_key_entry') and self.api_key_entry:
                if hasattr(self.api_key_entry, 'text'):  # PySide6
                    api_key = self.api_key_entry.text().strip()
            
            # Call the comprehensive environment variables method which reads ALL GUI widgets
            # This method already handles model_var, contextual_var, batch_translation_var, etc.
            # The manga integration can read these values directly from the updated widget variables
            env_vars = self._get_environment_variables('', api_key)
            
            print(f"[GUI_SYNC] ✅ GUI sync completed - _get_environment_variables read all widgets")
            print(f"[GUI_SYNC] Model from GUI: {env_vars.get('MODEL', 'Unknown')}")
            print(f"[GUI_SYNC] API key present: {bool(api_key)}")
            print(f"[GUI_SYNC] Contextual: {env_vars.get('CONTEXTUAL') == '1'}")
            print(f"[GUI_SYNC] Batch translation: {env_vars.get('BATCH_TRANSLATION') == '1'}")
            print(f"[GUI_SYNC] Multi-key mode: {self.config.get('use_multi_api_keys', False)}")
            
        except Exception as e:
            print(f"[GUI_SYNC] ❌ Error syncing GUI: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    import time
    import sys
    
    # Add global exception handler to catch any unhandled exceptions
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        print(f"[GLOBAL_EXCEPTION] Unhandled exception: {exc_type.__name__}: {exc_value}")
        import traceback
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print("[GLOBAL_EXCEPTION] Forcing exit due to unhandled exception")
        import os
        os._exit(1)  # Exit with code 1 for exceptions
    
    sys.excepthook = global_exception_handler
    
    # Ensure console encoding can handle emojis/Unicode in frozen exe environments
    try:
        import io, sys as _sys
        if hasattr(_sys.stdout, 'reconfigure'):
            try:
                _sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
                _sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
            except Exception:
                pass
        else:
            try:
                _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='ignore')
                _sys.stderr = io.TextIOWrapper(_sys.stderr.buffer, encoding='utf-8', errors='ignore')
            except Exception:
                pass
    except Exception:
        pass
    
    print("🚀 Starting Glossarion v6.1.0...")
    
    # Initialize splash screen
    splash_manager = None
    try:
        from splash_utils import SplashManager
        splash_manager = SplashManager()
        splash_started = splash_manager.start_splash()
        
        if splash_started:
            splash_manager.update_status("Loading theme framework...")
            time.sleep(0.1)
    except Exception as e:
        print(f"⚠️ Splash screen failed: {e}")
        splash_manager = None
    
    try:
        if splash_manager:
            splash_manager.update_status("Loading UI framework...")
            time.sleep(0.08)
        
        # PySide6 is already imported at the top of the file
        
        # REAL module loading during splash screen with gradual progression
        if splash_manager:
            # Check if debug mode is enabled in config
            debug_mode_enabled = False
            try:
                import json
                config_path = "config.json"
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        debug_mode_enabled = config.get('show_debug_buttons', False)
            except Exception:
                pass
            
            # Validate all Python scripts first (only if debug mode is enabled)
            if debug_mode_enabled:
                try:
                    success_count, total_count, failed_scripts = splash_manager.validate_all_scripts()
                    if failed_scripts:
                        print(f"\n⚠️ WARNING: {len(failed_scripts)} script(s) have compilation errors!")
                        print("The application will continue but some features may not work.\n")
                except Exception as e:
                    print(f"⚠️ Script validation failed: {e}")
            else:
                # Skip validation, jump straight to 25%
                splash_manager.set_progress(25)
            
            # Create a custom callback function for splash updates
            def splash_callback(message):
                if splash_manager and splash_manager.splash_window:
                    splash_manager.update_status(message)
                    splash_manager.splash_window.update()
                    time.sleep(0.09)
            
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
            time.sleep(0.07)
            
            # Extra pause to show "Ready!" before closing
            splash_manager.update_status("Ready!")
            time.sleep(0.1)
            splash_manager.close_splash()
        
        # Create main window (modules already loaded)
        from PySide6.QtWidgets import QApplication
        import sys
        
        # Check if QApplication already exists
        qapp = QApplication.instance()
        if not qapp:
            qapp = QApplication(sys.argv)
        
        # Initialize the app (modules already available)  
        main_window = TranslatorGUI()
        
        # Mark modules as already loaded to skip lazy loading
        main_window._modules_loaded = True
        main_window._modules_loading = False
        
        # Show the window (ensure not minimized)
        try:
            from PySide6.QtCore import Qt, QTimer
            # Clear any minimized state potentially inherited
            main_window.setWindowState(main_window.windowState() & ~Qt.WindowMinimized)
            main_window.showNormal()
            main_window.raise_()
            main_window.activateWindow()
            # Re-assert focus shortly after show to avoid race with splash/OS focus
            QTimer.singleShot(150, lambda: (main_window.raise_(), main_window.activateWindow()))
        except Exception:
            main_window.show()
        
        print("✅ Ready to use!")
        
        # Note: closeEvent is now handled by the TranslatorGUI.closeEvent method
        # No need to override it here
        
        # Add simple signal handlers for clean shutdown
        import signal
        def signal_handler(signum, frame):
            """Handle system signals gracefully"""
            print(f"[SIGNAL] Received signal {signum}, forcing exit")
            import os
            os._exit(0)
        
        # Register signal handlers (Windows-safe)
        try:
            if hasattr(signal, 'SIGINT'):
                signal.signal(signal.SIGINT, signal_handler)
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, signal_handler)
        except Exception as e:
            print(f"[SIGNAL] Could not register signal handlers: {e}")
        
        # Start main loop with debugging and proper cleanup
        exit_code = 0
        try:
            print("[MAIN] Starting Qt event loop...")
            exit_code = qapp.exec()
            print(f"[MAIN] Qt event loop ended naturally with code: {exit_code}")
        except Exception as e:
            print(f"[MAIN] Main loop error: {e}")
            import traceback
            traceback.print_exc()
            exit_code = 1
        except KeyboardInterrupt:
            print("[MAIN] Keyboard interrupt received")
            main_window.stop_all_operations()
            exit_code = 0
        
        # Ensure proper Qt cleanup
        try:
            print("[MAIN] Performing final Qt cleanup...")
            main_window.stop_all_operations()
            qapp.processEvents()  # Process any remaining events
            print("[MAIN] Qt cleanup completed")
        except Exception as e:
            print(f"[MAIN] Error during Qt cleanup: {e}")
        
        print(f"[MAIN] Main function completed with exit code {exit_code}, calling sys.exit({exit_code})...")
        # Exit with the code returned by Qt event loop
        import sys
        try:
            sys.exit(exit_code)
        except SystemExit as e:
            print(f"[MAIN] SystemExit raised with code: {e.code}")
            raise  # Re-raise to actually exit
        except Exception as e:
            print(f"[MAIN] Unexpected error during sys.exit: {e}")
            import os
            os._exit(exit_code)
        
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        if splash_manager:
            try:
                splash_manager.close_splash()
            except:
                pass
        import traceback
        traceback.print_exc()
        print("[MAIN] Exception occurred, exiting with code 1...")
        import sys
        sys.exit(1)
    
    finally:
        if splash_manager and not getattr(splash_manager, '_already_closed', False):
            try:
                print("[MAIN] Closing splash screen in finally block...")
                splash_manager.close_splash()
                splash_manager._already_closed = True
                print("[MAIN] Splash screen closed in finally")
            except Exception as e:
                print(f"[MAIN] Error closing splash: {e}")
        elif splash_manager:
            print("[MAIN] Splash screen already closed, skipping")
        print("[MAIN] Finally block executed")
