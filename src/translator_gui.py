#translator_gui.py
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

# Standard Library
import io, json, logging, math, os, shutil, sys, threading, time, re, concurrent.futures, signal
from logging.handlers import RotatingFileHandler
import atexit
import faulthandler
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk
from ai_hunter_enhanced import AIHunterConfigGUI, ImprovedAIHunterDetection
import traceback
# Third-Party
import ttkbootstrap as tb
from ttkbootstrap.constants import *
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

        # Rotating log handler
        log_file = os.path.join(logs_dir, "run.log")
        handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(process)d:%(threadName)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        # Avoid duplicate handlers on reload
        if not any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
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
        def _log_excepthook(exc_type, exc_value, exc_tb):
            try:
                logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
            except Exception:
                pass
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

class UIHelper:
    """Consolidated UI utility functions"""
    
    @staticmethod
    def setup_text_undo_redo(text_widget):
        """Set up undo/redo bindings for a text widget"""
        # NUCLEAR OPTION: Disable built-in undo completely
        try:
            text_widget.config(undo=False)
        except:
            pass
        
        # Remove ALL possible z-related bindings
        all_z_bindings = [
            'z', 'Z', '<z>', '<Z>', '<Key-z>', '<Key-Z>', 
            '<Alt-z>', '<Alt-Z>', '<Meta-z>', '<Meta-Z>', 
            '<Mod1-z>', '<Mod1-Z>', '<<Undo>>', '<<Redo>>',
            '<Control-Key-z>', '<Control-Key-Z>'
        ]
        
        for seq in all_z_bindings:
            try:
                text_widget.unbind(seq)
                text_widget.unbind_all(seq)  
                text_widget.unbind_class('Text', seq)
            except:
                pass
        
        # Create our own undo/redo stack with better management
        class UndoRedoManager:
            def __init__(self):
                self.undo_stack = []
                self.redo_stack = []
                self.is_undoing = False
                self.is_redoing = False
                self.last_action_was_undo = False
                
            def save_state(self):
                """Save current state to undo stack"""
                if self.is_undoing or self.is_redoing:
                    return
                    
                try:
                    content = text_widget.get(1.0, tk.END)
                    # Only save if content changed
                    if not self.undo_stack or self.undo_stack[-1] != content:
                        self.undo_stack.append(content)
                        if len(self.undo_stack) > 100:
                            self.undo_stack.pop(0)
                        # Only clear redo stack if this is a new edit (not from undo)
                        if not self.last_action_was_undo:
                            self.redo_stack.clear()
                        self.last_action_was_undo = False
                except:
                    pass
            
            def undo(self):
                """Perform undo"""
                #print(f"[DEBUG] Undo called. Stack size: {len(self.undo_stack)}, Redo stack: {len(self.redo_stack)}")
                if len(self.undo_stack) > 1:
                    self.is_undoing = True
                    self.last_action_was_undo = True
                    try:
                        # Save cursor position
                        cursor_pos = text_widget.index(tk.INSERT)
                        
                        # Move current state to redo stack
                        current = self.undo_stack.pop()
                        self.redo_stack.append(current)
                        
                        # Restore previous state
                        previous = self.undo_stack[-1]
                        text_widget.delete(1.0, tk.END)
                        text_widget.insert(1.0, previous.rstrip('\n'))
                        
                        # Restore cursor position
                        try:
                            text_widget.mark_set(tk.INSERT, cursor_pos)
                            text_widget.see(tk.INSERT)
                        except:
                            text_widget.mark_set(tk.INSERT, "1.0")
                            
                        #print(f"[DEBUG] Undo complete. New redo stack size: {len(self.redo_stack)}")
                    finally:
                        self.is_undoing = False
                return "break"
            
            def redo(self):
                """Perform redo"""
                print(f"[DEBUG] Redo called. Redo stack size: {len(self.redo_stack)}")
                if self.redo_stack:
                    self.is_redoing = True
                    try:
                        # Save cursor position
                        cursor_pos = text_widget.index(tk.INSERT)
                        
                        # Get next state
                        next_state = self.redo_stack.pop()
                        
                        # Add to undo stack
                        self.undo_stack.append(next_state)
                        
                        # Restore state
                        text_widget.delete(1.0, tk.END)
                        text_widget.insert(1.0, next_state.rstrip('\n'))
                        
                        # Restore cursor position
                        try:
                            text_widget.mark_set(tk.INSERT, cursor_pos)
                            text_widget.see(tk.INSERT)
                        except:
                            text_widget.mark_set(tk.INSERT, "end-1c")
                            
                        print(f"[DEBUG] Redo complete. Remaining redo stack: {len(self.redo_stack)}")
                    finally:
                        self.is_redoing = False
                        self.last_action_was_undo = True
                return "break"
        
        # Create manager instance
        manager = UndoRedoManager()
        
        # CRITICAL: Override ALL key handling to intercept 'z'
        def handle_key_press(event):
            """Intercept ALL key presses"""
            # Check for 'z' or 'Z'
            if event.keysym.lower() == 'z':
                # Check if Control is pressed
                if event.state & 0x4:  # Control key is pressed
                    # This is Control+Z - let it pass to our undo handler
                    return None  # Let it pass through to our Control+Z binding
                else:
                    # Just 'z' without Control - insert it manually
                    if event.char in ['z', 'Z']:
                        try:
                            text_widget.insert(tk.INSERT, event.char)
                        except:
                            pass
                        return "break"
            
            # Check for Control+Y (redo)  
            if event.keysym.lower() == 'y' and (event.state & 0x4):
                return None  # Let it pass through to our Control+Y binding
            
            # All other keys pass through
            return None
        
        # Bind with highest priority
        text_widget.bind('<Key>', handle_key_press, add=False)
        
        # Bind undo/redo commands
        text_widget.bind('<Control-z>', lambda e: manager.undo())
        text_widget.bind('<Control-Z>', lambda e: manager.undo())
        text_widget.bind('<Control-y>', lambda e: manager.redo())
        text_widget.bind('<Control-Y>', lambda e: manager.redo())
        text_widget.bind('<Control-Shift-z>', lambda e: manager.redo())
        text_widget.bind('<Control-Shift-Z>', lambda e: manager.redo())
        
        # macOS bindings
        text_widget.bind('<Command-z>', lambda e: manager.undo())
        text_widget.bind('<Command-Z>', lambda e: manager.undo())
        text_widget.bind('<Command-Shift-z>', lambda e: manager.redo())
        
        # Track changes more efficiently
        save_timer = [None]
        
        def schedule_save():
            """Schedule a save operation with debouncing"""
            # Cancel any pending save
            if save_timer[0]:
                text_widget.after_cancel(save_timer[0])
            # Schedule new save
            save_timer[0] = text_widget.after(200, manager.save_state)
        
        def on_text_modified(event=None):
            """Handle text modifications"""
            # Don't save during undo/redo or for modifier keys
            if event and event.keysym in ['Control_L', 'Control_R', 'Alt_L', 'Alt_R', 
                                         'Shift_L', 'Shift_R', 'Left', 'Right', 'Up', 'Down',
                                         'Home', 'End', 'Prior', 'Next']:
                return
            
            if not manager.is_undoing and not manager.is_redoing:
                schedule_save()
        
        # More efficient change tracking
        text_widget.bind('<KeyRelease>', on_text_modified)
        text_widget.bind('<<Paste>>', lambda e: text_widget.after(10, manager.save_state))
        text_widget.bind('<<Cut>>', lambda e: text_widget.after(10, manager.save_state))
        
        # Save initial state
        def initialize():
            """Initialize with current content"""
            try:
                content = text_widget.get(1.0, tk.END)
                manager.undo_stack.append(content)
                #print(f"[DEBUG] Initial state saved. Content length: {len(content)}")
            except:
                pass
        
        text_widget.after(50, initialize)
    
    @staticmethod
    def setup_dialog_scrolling(dialog_window, canvas):
        """Setup mouse wheel scrolling for dialogs"""
        def on_mousewheel(event):
            try: 
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except: 
                pass
        
        def on_mousewheel_linux(event, direction):
            try:
                if canvas.winfo_exists():
                    canvas.yview_scroll(direction, "units")
            except tk.TclError: 
                pass
        
        # Bind events TO THE CANVAS AND DIALOG, NOT GLOBALLY
        dialog_window.bind("<MouseWheel>", on_mousewheel)
        dialog_window.bind("<Button-4>", lambda e: on_mousewheel_linux(e, -1))
        dialog_window.bind("<Button-5>", lambda e: on_mousewheel_linux(e, 1))
        
        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Button-4>", lambda e: on_mousewheel_linux(e, -1))
        canvas.bind("<Button-5>", lambda e: on_mousewheel_linux(e, 1))
        
        # Return cleanup function
        def cleanup_bindings():
            try:
                dialog_window.unbind("<MouseWheel>")
                dialog_window.unbind("<Button-4>")
                dialog_window.unbind("<Button-5>")
                canvas.unbind("<MouseWheel>")
                canvas.unbind("<Button-4>")
                canvas.unbind("<Button-5>")
            except: 
                pass
        
        return cleanup_bindings
    
    @staticmethod
    def create_button_resize_handler(button, base_width, base_height, 
                                   master_window, reference_width, reference_height):
        """Create a resize handler for dynamic button scaling"""
        def on_resize(event):
            if event.widget is master_window:
                sx = event.width / reference_width
                sy = event.height / reference_height
                s = min(sx, sy)
                new_w = int(base_width * s)
                new_h = int(base_height * s)
                ipadx = max(0, (new_w - base_width) // 2)
                ipady = max(0, (new_h - base_height) // 2)
                button.grid_configure(ipadx=ipadx, ipady=ipady)
        
        return on_resize
    
    @staticmethod
    def setup_scrollable_text(parent, **text_kwargs):
        """Create a scrolled text widget with undo/redo support"""
        # Remove undo=True from kwargs if present, as we'll handle it ourselves
        text_kwargs.pop('undo', None)
        text_kwargs.pop('autoseparators', None)
        text_kwargs.pop('maxundo', None)
        
        # Create ScrolledText without built-in undo
        text_widget = scrolledtext.ScrolledText(parent, **text_kwargs)
        
        # Apply our custom undo/redo setup
        UIHelper.setup_text_undo_redo(text_widget)
        
        # Extra protection for ScrolledText widgets
        UIHelper._fix_scrolledtext_z_key(text_widget)
        
        return text_widget
    
    @staticmethod
    def _fix_scrolledtext_z_key(scrolled_widget):
        """Apply additional fixes specifically for ScrolledText widgets"""
        # ScrolledText stores the actual Text widget in different ways depending on version
        # Try to find the actual text widget
        text_widget = None
        
        # Method 1: Direct attribute
        if hasattr(scrolled_widget, 'text'):
            text_widget = scrolled_widget.text
        # Method 2: It might be the widget itself
        elif hasattr(scrolled_widget, 'insert') and hasattr(scrolled_widget, 'delete'):
            text_widget = scrolled_widget
        # Method 3: Look in children
        else:
            for child in scrolled_widget.winfo_children():
                if isinstance(child, tk.Text):
                    text_widget = child
                    break
        
        if not text_widget:
            # If we can't find the text widget, work with scrolled_widget directly
            text_widget = scrolled_widget
        
        # Remove ALL 'z' related bindings at all levels
        for widget in [text_widget, scrolled_widget]:
            for seq in ['z', 'Z', '<z>', '<Z>', '<Key-z>', '<Key-Z>', 
                       '<<Undo>>', '<<Redo>>', '<Alt-z>', '<Alt-Z>',
                       '<Meta-z>', '<Meta-Z>', '<Mod1-z>', '<Mod1-Z>']:
                try:
                    widget.unbind(seq)
                    widget.unbind_all(seq)
                except:
                    pass
        
        # Override the 'z' key completely
        def intercept_z(event):
            if event.char in ['z', 'Z']:
                if not (event.state & 0x4):  # No Control key
                    text_widget.insert(tk.INSERT, event.char)
                    return "break"
            return None
        
        # Bind with high priority to both widgets
        text_widget.bind('<KeyPress>', intercept_z, add=False)
        text_widget.bind('z', lambda e: intercept_z(e))
        text_widget.bind('Z', lambda e: intercept_z(e))
    
    @staticmethod
    def block_text_editing(text_widget):
        """Make a text widget read-only but allow selection and copying"""
        def block_editing(event):
            # Allow copy
            if event.state & 0x4 and event.keysym.lower() == 'c':
                return None
            # Allow select all
            if event.state & 0x4 and event.keysym.lower() == 'a':
                text_widget.tag_add(tk.SEL, "1.0", tk.END)
                text_widget.mark_set(tk.INSERT, "1.0")
                text_widget.see(tk.INSERT)
                return "break"
            # Allow navigation
            if event.keysym in ['Left', 'Right', 'Up', 'Down', 'Home', 'End', 'Prior', 'Next']:
                return None
            # Allow shift selection
            if event.state & 0x1:
                return None
            return "break"
        
        text_widget.bind("<Key>", block_editing)
    
    @staticmethod
    def disable_spinbox_mousewheel(spinbox):
        """Disable mousewheel scrolling on a spinbox to prevent accidental value changes"""
        def block_wheel(event):
            return "break"
        
        spinbox.bind("<MouseWheel>", block_wheel)  # Windows
        spinbox.bind("<Button-4>", block_wheel)    # Linux scroll up
        spinbox.bind("<Button-5>", block_wheel)    # Linux scroll down
        
class WindowManager:
    """Unified window geometry and dialog management - FULLY REFACTORED V2"""
    
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.ui = UIHelper()
        self._stored_geometries = {}
        self._pending_operations = {}
        self._dpi_scale = None
        self._topmost_protection_active = {}
        self._force_safe_ratios = False
        self._primary_monitor_width = None  # Cache the detected width

    def toggle_safe_ratios(self):
        """Toggle forcing 1080p Windows ratios"""
        self._force_safe_ratios = not self._force_safe_ratios
        return self._force_safe_ratios
    
    def get_dpi_scale(self, window):
        """Get and cache DPI scaling factor"""
        if self._dpi_scale is None:
            try:
                self._dpi_scale = window.tk.call('tk', 'scaling') / 1.333
            except:
                self._dpi_scale = 1.0
        return self._dpi_scale
    
    def responsive_size(self, window, base_width, base_height, 
                       scale_factor=None, center=True, use_full_height=True):
        """Size window responsively based on primary monitor"""
        
        # Auto-detect primary monitor
        primary_width = self.detect_primary_monitor_width(window)
        screen_height = window.winfo_screenheight()
        
        if use_full_height:
            width = min(int(base_width * 1.2), int(primary_width * 0.98))
            height = int(screen_height * 0.98)
        else:
            width = base_width
            height = base_height
            
            if width > primary_width * 0.9:
                width = int(primary_width * 0.85)
            if height > screen_height * 0.9:
                height = int(screen_height * 0.85)
        
        if center:
            x = (primary_width - width) // 2
            y = (screen_height - height) // 2
            geometry_str = f"{width}x{height}+{x}+{y}"
        else:
            geometry_str = f"{width}x{height}"
        
        window.geometry(geometry_str)
        window.attributes('-topmost', False)
        
        return width, height

    def setup_window(self, window, width=None, height=None, 
                    center=True, icon=True, hide_initially=False,
                    max_width_ratio=0.98, max_height_ratio=0.98,
                    min_width=400, min_height=300):
        """Universal window setup with auto-detected primary monitor"""
        
        if hide_initially:
            window.withdraw()
        
        window.attributes('-topmost', False)
        
        if icon:
            window.after_idle(lambda: load_application_icon(window, self.base_dir))
        
        primary_width = self.detect_primary_monitor_width(window)
        screen_height = window.winfo_screenheight()
        dpi_scale = self.get_dpi_scale(window)
        
        if width is None:
            width = min_width
        else:
            width = int(width / dpi_scale)
            
        if height is None:
            height = int(screen_height * max_height_ratio)
        else:
            height = int(height / dpi_scale)
        
        max_width = int(primary_width * max_width_ratio)  # Use primary width
        max_height = int(screen_height * max_height_ratio)
        
        final_width = max(min_width, min(width, max_width))
        final_height = max(min_height, min(height, max_height))
        
        if center:
            x = max(0, (primary_width - final_width) // 2)  # Center on primary
            y = 5
            geometry_str = f"{final_width}x{final_height}+{x}+{y}"
        else:
            geometry_str = f"{final_width}x{final_height}"
        
        window.geometry(geometry_str)
        
        if hide_initially:
            window.after(10, window.deiconify)
        
        return final_width, final_height
    
    def get_monitor_from_coord(self, x, y):
        """Get monitor info for coordinates (for multi-monitor support)"""
        # This is a simplified version - returns primary monitor info
        # For true multi-monitor, you'd need to use win32api or other libraries
        monitors = []
        
        # Try to detect if window is on secondary monitor
        # This is a heuristic - if x > screen_width, likely on second monitor
        primary_width = self.root.winfo_screenwidth() if hasattr(self, 'root') else 1920
        
        if x > primary_width:
            # Likely on second monitor
            return {'x': primary_width, 'width': primary_width, 'height': 1080}
        else:
            # Primary monitor
            return {'x': 0, 'width': primary_width, 'height': 1080}
    
    def _fix_maximize_behavior(self, window):
        """Fix the standard Windows maximize button for multi-monitor"""
        # Store original window protocol
        original_state_change = None
        
        def on_window_state_change(event):
            """Intercept maximize from title bar button"""
            if event.widget == window:
                try:
                    state = window.state()
                    if state == 'zoomed':
                        # Window was just maximized - fix it
                        window.after(10, lambda: self._proper_maximize(window))
                except:
                    pass
        
        # Bind to window state changes to intercept maximize
        window.bind('<Configure>', on_window_state_change, add='+')
    
    def _proper_maximize(self, window):
        """Properly maximize window to current monitor only"""
        try:
            # Get current position
            x = window.winfo_x()
            screen_width = window.winfo_screenwidth()
            screen_height = window.winfo_screenheight()
            
            # Check if on secondary monitor
            if x > screen_width or x < -screen_width/2:
                # Likely on a secondary monitor
                # Force back to primary monitor for now
                window.state('normal')
                window.geometry(f"{screen_width-100}x{screen_height-100}+50+50")
                window.state('zoomed')
            
            # The zoomed state should now respect monitor boundaries
            
        except Exception as e:
            print(f"Error in proper maximize: {e}")
    
    def auto_resize_dialog(self, dialog, canvas=None, max_width_ratio=0.9, max_height_ratio=0.95):
        """Auto-resize dialog based on content"""
        
        # Override ratios if 1080p mode is on
        if self._force_safe_ratios:
            max_height_ratio = min(max_height_ratio, 0.85)  # Force 85% max
            max_width_ratio = min(max_width_ratio, 0.85)
        
        was_hidden = not dialog.winfo_viewable()
        
        def perform_resize():
            try:
                screen_width = dialog.winfo_screenwidth()
                screen_height = dialog.winfo_screenheight()
                dpi_scale = self.get_dpi_scale(dialog)
                
                final_height = int(screen_height * max_height_ratio)
                
                if canvas and canvas.winfo_exists():
                    scrollable_frame = None
                    for child in canvas.winfo_children():
                        if isinstance(child, ttk.Frame):
                            scrollable_frame = child
                            break
                    
                    if scrollable_frame and scrollable_frame.winfo_exists():
                        content_width = scrollable_frame.winfo_reqwidth()
                        # Add 5% more space to content width, plus scrollbar space
                        window_width = int(content_width * 1.15) + 120
                    else:
                        window_width = int(dialog.winfo_reqwidth() * 1.15)
                else:
                    window_width = int(dialog.winfo_reqwidth() * 1.15)
                
                window_width = int(window_width / dpi_scale)
                
                max_width = int(screen_width * max_width_ratio)
                final_width = min(window_width, max_width)
                final_width = max(final_width, 600)
                
                x = (screen_width - final_width) // 2
                y = max(20, (screen_height - final_height) // 2)
                
                dialog.geometry(f"{final_width}x{final_height}+{x}+{y}")
                
                if was_hidden and dialog.winfo_exists():
                    dialog.deiconify()
                
                return final_width, final_height
                
            except tk.TclError:
                return None, None
        
        dialog.after(20, perform_resize)
        return None, None
    
    def setup_scrollable(self, parent_window, title, width=None, height=None,
                        modal=True, resizable=True, max_width_ratio=0.9, 
                        max_height_ratio=0.95, **kwargs):
        """Create a scrollable dialog with proper setup"""
        
        dialog = tk.Toplevel(parent_window)
        dialog.title(title)
        dialog.withdraw()
        
        # Ensure not topmost
        dialog.attributes('-topmost', False)
        
        if not resizable:
            dialog.resizable(False, False)
        
        if modal:
            dialog.transient(parent_window)
            # Don't grab - it blocks other windows
        
        dialog.after_idle(lambda: load_application_icon(dialog, self.base_dir))
        
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        dpi_scale = self.get_dpi_scale(dialog)
        
        if height is None:
            height = int(screen_height * max_height_ratio)
        else:
            height = int(height / dpi_scale)
            
        if width is None or width == 0:
            width = int(screen_width * 0.8)
        else:
            width = int(width / dpi_scale)
        
        width = min(width, int(screen_width * max_width_ratio))
        height = min(height, int(screen_height * max_height_ratio))
        
        x = (screen_width - width) // 2
        y = max(20, (screen_height - height) // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        main_container = tk.Frame(dialog)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(main_container, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        def configure_scroll_region(event=None):
            if canvas.winfo_exists():
                canvas.configure(scrollregion=canvas.bbox("all"))
                canvas_width = canvas.winfo_width()
                if canvas_width > 1:
                    canvas.itemconfig(canvas_window, width=canvas_width)
        
        scrollable_frame.bind("<Configure>", configure_scroll_region)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        cleanup_scrolling = self.ui.setup_dialog_scrolling(dialog, canvas)
        
        dialog._cleanup_scrolling = cleanup_scrolling
        dialog._canvas = canvas
        dialog._scrollable_frame = scrollable_frame
        dialog._kwargs = kwargs
        
        dialog.after(50, dialog.deiconify)
        
        return dialog, scrollable_frame, canvas
    
    def create_simple_dialog(self, parent, title, width=None, height=None, 
                           modal=True, hide_initially=True):
        """Create a simple non-scrollable dialog"""
        
        dialog = tk.Toplevel(parent)
        dialog.title(title)
        
        # Ensure not topmost
        dialog.attributes('-topmost', False)
        
        if modal:
            dialog.transient(parent)
            # Don't grab - it blocks other windows
        
        dpi_scale = self.get_dpi_scale(dialog)
        
        adjusted_width = None
        adjusted_height = None
        
        if width is not None:
            adjusted_width = int(width / dpi_scale)
        
        if height is not None:
            adjusted_height = int(height / dpi_scale)
        else:
            screen_height = dialog.winfo_screenheight()
            adjusted_height = int(screen_height * 0.98)
        
        final_width, final_height = self.setup_window(
            dialog, 
            width=adjusted_width, 
            height=adjusted_height,
            hide_initially=hide_initially,
            max_width_ratio=0.98, 
            max_height_ratio=0.98
        )
        
        return dialog
    
    def setup_maximize_support(self, window):
        """Setup F11 to maximize window - simple working version"""
        
        def toggle_maximize(event=None):
            """F11 toggles maximize"""
            current = window.state()
            if current == 'zoomed':
                window.state('normal')
            else:
                window.state('zoomed')
            return "break"
        
        # Bind F11
        window.bind('<F11>', toggle_maximize)
        
        # Bind Escape to exit maximize only
        window.bind('<Escape>', lambda e: window.state('normal') if window.state() == 'zoomed' else None)
        
        return toggle_maximize
    
    def setup_fullscreen_support(self, window):
        """Legacy method - just calls setup_maximize_support"""
        return self.setup_maximize_support(window)
    
    def _setup_maximize_fix(self, window):
        """Setup for Windows title bar maximize button"""
        # For now, just let Windows handle maximize naturally
        # Most modern Windows versions handle multi-monitor maximize correctly
        pass
    
    def _fix_multi_monitor_maximize(self, window):
        """No longer needed - Windows handles maximize correctly"""
        pass
    
    def store_geometry(self, window, key):
        """Store window geometry for later restoration"""
        if window.winfo_exists():
            self._stored_geometries[key] = window.geometry()
    
    def restore_geometry(self, window, key, delay=100):
        """Restore previously stored geometry"""
        if key in self._stored_geometries:
            geometry = self._stored_geometries[key]
            window.after(delay, lambda: window.geometry(geometry) if window.winfo_exists() else None)
    
    def toggle_window_maximize(self, window):
        """Toggle maximize state for any window (multi-monitor safe)"""
        try:
            current_state = window.state()
            
            if current_state == 'zoomed':
                # Restore to normal
                window.state('normal')
            else:
                # Get current monitor
                x = window.winfo_x()
                screen_width = window.winfo_screenwidth()
                
                # Ensure window is fully on one monitor before maximizing
                if x >= screen_width:
                    # On second monitor
                    window.geometry(f"+{screen_width}+0")
                elif x + window.winfo_width() > screen_width:
                    # Spanning monitors - move to primary
                    window.geometry(f"+0+0")
                
                # Maximize to current monitor
                window.state('zoomed')
                
        except Exception as e:
            print(f"Error toggling maximize: {e}")
            # Fallback method
            self._manual_maximize(window)
    
    def _manual_maximize(self, window):
        """Manual maximize implementation as fallback"""
        if not hasattr(window, '_maximize_normal_geometry'):
            window._maximize_normal_geometry = None
        
        if window._maximize_normal_geometry:
            # Restore
            window.geometry(window._maximize_normal_geometry)
            window._maximize_normal_geometry = None
        else:
            # Store current
            window._maximize_normal_geometry = window.geometry()
            
            # Get dimensions
            x = window.winfo_x()
            screen_width = window.winfo_screenwidth()
            screen_height = window.winfo_screenheight()
            
            # Determine monitor
            if x >= screen_width:
                new_x = screen_width
            else:
                new_x = 0
            
            # Leave space for taskbar
            taskbar_height = 40
            usable_height = screen_height - taskbar_height
            
            window.geometry(f"{screen_width}x{usable_height}+{new_x}+0")
            
    def detect_primary_monitor_width(self, reference_window):
        """Auto-detect primary monitor width"""
        if self._primary_monitor_width is not None:
            return self._primary_monitor_width
        
        try:
            # Create a hidden test window at origin (0,0) - should be on primary monitor
            test = tk.Toplevel(reference_window)
            test.withdraw()
            test.overrideredirect(True)  # No window decorations
            
            # Position at origin
            test.geometry("100x100+0+0")
            test.update_idletasks()
            
            # Now maximize it to get the monitor's dimensions
            test.state('zoomed')
            test.update_idletasks()
            
            # Get the maximized width - this is the primary monitor width
            primary_width = test.winfo_width()
            primary_height = test.winfo_height()
            
            test.destroy()
            
            # Get total desktop width for comparison
            total_width = reference_window.winfo_screenwidth()
            screen_height = reference_window.winfo_screenheight()
            
            print(f"[DEBUG] Maximized test window: {primary_width}x{primary_height}")
            print(f"[DEBUG] Total desktop: {total_width}x{screen_height}")
            
            # If the maximized width equals total width, check for dual monitors
            if primary_width >= total_width * 0.95:
                # Maximized window = total desktop width, need to detect if dual monitor
                aspect = total_width / screen_height
                print(f"[DEBUG] Aspect ratio: {aspect:.2f}")
                
                # For dual monitors detection:
                # - Two 1920x1080 monitors = 3840x1080 (aspect 3.56)
                # - Two 2560x1440 monitors = 5120x1440 (aspect 3.56)
                # - Two 1280x1440 monitors = 2560x1440 (aspect 1.78)
                # Single ultrawide:
                # - 3440x1440 = aspect 2.39
                # - 2560x1080 = aspect 2.37
                
                # If width is exactly double a common resolution, it's dual monitors
                if total_width == 3840 and screen_height == 1080:
                    # Two 1920x1080 monitors
                    primary_width = 1920
                    print(f"[DEBUG] Detected dual 1920x1080 monitors: {primary_width}")
                elif total_width == 2560 and screen_height == 1440:
                    # Two 1280x1440 monitors OR could be single 1440p
                    # Check if this is likely dual by seeing if half width makes sense
                    primary_width = 1280
                    print(f"[DEBUG] Detected dual 1280x1440 monitors: {primary_width}")
                elif total_width == 5120 and screen_height == 1440:
                    # Two 2560x1440 monitors
                    primary_width = 2560
                    print(f"[DEBUG] Detected dual 2560x1440 monitors: {primary_width}")
                elif aspect > 3.0:
                    # Likely dual monitor based on aspect ratio
                    primary_width = total_width // 2
                    print(f"[DEBUG] Detected dual monitors by aspect ratio: {primary_width}")
                else:
                    # Single ultrawide or normal monitor
                    print(f"[DEBUG] Single monitor detected: {primary_width}")
            else:
                print(f"[DEBUG] Primary monitor width detected: {primary_width}")
            
            self._primary_monitor_width = primary_width
            print(f"✅ Final primary monitor width: {primary_width}")
            return primary_width
            
        except Exception as e:
            print(f"⚠️ Error detecting monitor: {e}")
            # Fallback to common resolutions based on height
            height = reference_window.winfo_screenheight()
            if height >= 2160:
                return 3840  # 4K
            elif height >= 1440:
                return 2560  # 1440p
            elif height >= 1080:
                return 1920  # 1080p
            else:
                return 1366  # 720p

    def center_window(self, window):
        """Center a window on primary screen with auto-detection and taskbar awareness"""
        def do_center():
            if window.winfo_exists():
                window.update_idletasks()
                width = window.winfo_width()
                height = window.winfo_height()
                screen_height = window.winfo_screenheight()
                
                # Auto-detect primary monitor width
                primary_width = self.detect_primary_monitor_width(window)
                
                # Windows taskbar is typically 40-50px at the bottom
                taskbar_height = 50
                usable_height = screen_height - taskbar_height
                
                # Center horizontally on primary monitor (which starts at x=0)
                # If window is wider than primary monitor, center it anyway
                # (it will extend into the second monitor, which is fine)
                x = (primary_width - width) // 2
                
                # Allow negative x if window is wider - this centers it on primary monitor
                # even if it extends into second monitor
                
                # Position vertically - lower on screen
                y = 50
                
                print(f"[DEBUG] Positioning window at: {x}, {y} (size: {width}x{height})")
                print(f"[DEBUG] Primary monitor width: {primary_width}, Screen height: {screen_height}")
                
                window.geometry(f"+{x}+{y}")
        
        # Execute immediately (no after_idle delay)
        do_center()
    
class TranslatorGUI:
    def __init__(self, master):        
        # Initialization
        master.configure(bg='#2b2b2b')
        self.master = master
        self.base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        self.wm = WindowManager(self.base_dir)
        self.ui = UIHelper()
        master.attributes('-topmost', False)
        master.lift()
        self.max_output_tokens = 8192
        self.proc = self.glossary_proc = None
        __version__ = "5.0.5"
        self.__version__ = __version__  # Store as instance variable
        master.title(f"Glossarion v{__version__}")
        
        # Get screen dimensions - need to detect primary monitor width first
        screen_height = master.winfo_screenheight()
        
        # Detect primary monitor width (not combined width of all monitors)
        primary_width = self.wm.detect_primary_monitor_width(master)
        
        # Set window size - making it wider as requested
        # 95% was 1216px, +30% = ~1580px, which is 1.234x the primary monitor
        # This will span slightly into the second monitor but centered on primary
        width_ratio = 1.23  # 123% of primary monitor width (30% wider than before)
        # Account for Windows taskbar (typically 40-50px)
        taskbar_height = 50
        usable_height = screen_height - taskbar_height
        height_ratio = 0.92  # 92% of usable height (slightly reduced)
        
        window_width = int(primary_width * width_ratio)
        window_height = int(usable_height * height_ratio)
        
        print(f"[DEBUG] Calculated window size: {window_width}x{window_height}")
        print(f"[DEBUG] Primary width: {primary_width}, Usable height: {usable_height}")
        print(f"[DEBUG] Width ratio: {width_ratio}, Height ratio: {height_ratio}")
        
        # Apply size
        master.geometry(f"{window_width}x{window_height}")
        
        # Set minimum size as ratio too
        min_width = int(primary_width * 0.5)  # 50% minimum of primary monitor
        min_height = int(usable_height * 0.5)  # 50% minimum
        master.minsize(min_width, min_height)
        
        self.wm.center_window(master)
        
        # Setup fullscreen support
        self.wm.setup_fullscreen_support(master)
        
        self.payloads_dir = os.path.join(os.getcwd(), "Payloads")
        
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
        
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        
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
            self.openrouter_http_only_var = tk.BooleanVar(
                value=self.config.get('openrouter_use_http_only', False)
            )
        except Exception:
            self.openrouter_http_only_var = tk.BooleanVar(value=False)
        
        try:
            self.openrouter_accept_identity_var = tk.BooleanVar(
                value=self.config.get('openrouter_accept_identity', False)
            )
        except Exception:
            self.openrouter_accept_identity_var = tk.BooleanVar(value=False)
            
        # Initialize retain_source_extension env var on startup
        try:
            os.environ['RETAIN_SOURCE_EXTENSION'] = '1' if self.config.get('retain_source_extension', False) else '0'
        except Exception:
            pass
        
        if self.config.get('force_safe_ratios', False):
            self.wm._force_safe_ratios = True
            # Update button after GUI is created
            self.master.after(500, lambda: (
                self.safe_ratios_btn.config(text="📐 1080p: ON", bootstyle="success") 
                if hasattr(self, 'safe_ratios_btn') else None
            ))
    
        # Initialize auto-update check and other variables
        self.auto_update_check_var = tk.BooleanVar(value=self.config.get('auto_update_check', True))
        self.force_ncx_only_var = tk.BooleanVar(value=self.config.get('force_ncx_only', True))
        self.single_api_image_chunks_var = tk.BooleanVar(value=False)
        self.enable_gemini_thinking_var = tk.BooleanVar(value=self.config.get('enable_gemini_thinking', True))
        self.thinking_budget_var = tk.StringVar(value=str(self.config.get('thinking_budget', '-1')))
        # NEW: GPT/OpenRouter reasoning controls
        self.enable_gpt_thinking_var = tk.BooleanVar(value=self.config.get('enable_gpt_thinking', True))
        self.gpt_reasoning_tokens_var = tk.StringVar(value=str(self.config.get('gpt_reasoning_tokens', '2000')))
        self.gpt_effort_var = tk.StringVar(value=self.config.get('gpt_effort', 'medium'))
        self.thread_delay_var = tk.StringVar(value=str(self.config.get('thread_submission_delay', 0.5)))
        self.remove_ai_artifacts = os.getenv("REMOVE_AI_ARTIFACTS", "0") == "1"
        print(f"   🎨 Remove AI Artifacts: {'ENABLED' if self.remove_ai_artifacts else 'DISABLED'}")
        self.disable_chapter_merging_var = tk.BooleanVar(value=self.config.get('disable_chapter_merging', False))
        self.selected_files = []
        self.current_file_index = 0
        self.use_gemini_openai_endpoint_var = tk.BooleanVar(value=self.config.get('use_gemini_openai_endpoint', False))
        self.gemini_openai_endpoint_var = tk.StringVar(value=self.config.get('gemini_openai_endpoint', ''))
        self.azure_api_version_var = tk.StringVar(value=self.config.get('azure_api_version', '2025-01-01-preview'))
        # Set initial Azure API version environment variable
        azure_version = self.config.get('azure_api_version', '2025-01-01-preview')
        os.environ['AZURE_API_VERSION'] = azure_version
        print(f"🔧 Initial Azure API Version set: {azure_version}")
        self.use_fallback_keys_var = tk.BooleanVar(value=self.config.get('use_fallback_keys', False))

        # Initialize fuzzy threshold variable
        if not hasattr(self, 'fuzzy_threshold_var'):
            self.fuzzy_threshold_var = tk.DoubleVar(value=self.config.get('glossary_fuzzy_threshold', 0.90))
        self.use_legacy_csv_var = tk.BooleanVar(value=self.config.get('glossary_use_legacy_csv', False))

        
        # Initialize the variables with default values
        self.enable_parallel_extraction_var = tk.BooleanVar(value=self.config.get('enable_parallel_extraction', True))
        self.extraction_workers_var = tk.IntVar(value=self.config.get('extraction_workers', 2))
        # GUI yield toggle - disabled by default for maximum speed
        self.enable_gui_yield_var = tk.BooleanVar(value=self.config.get('enable_gui_yield', True))

        # Set initial environment variable and ensure executor
        if self.enable_parallel_extraction_var.get():
            # Set workers for glossary extraction optimization
            workers = self.extraction_workers_var.get()
            os.environ["EXTRACTION_WORKERS"] = str(workers)
            # Also enable glossary parallel processing explicitly
            os.environ["GLOSSARY_PARALLEL_ENABLED"] = "1"
            print(f"✅ Parallel extraction enabled with {workers} workers")
        else:
            os.environ["EXTRACTION_WORKERS"] = "1"
            os.environ["GLOSSARY_PARALLEL_ENABLED"] = "0"
        
        # Set GUI yield environment variable (disabled by default for maximum speed)
        os.environ['ENABLE_GUI_YIELD'] = '1' if self.enable_gui_yield_var.get() else '0'
        print(f"⚡ GUI yield: {'ENABLED (responsive)' if self.enable_gui_yield_var.get() else 'DISABLED (maximum speed)'}")
        
        # Initialize the executor based on current settings
        try:
            self._ensure_executor()
        except Exception:
            pass


        # Initialize compression-related variables
        self.enable_image_compression_var = tk.BooleanVar(value=self.config.get('enable_image_compression', False))
        self.auto_compress_enabled_var = tk.BooleanVar(value=self.config.get('auto_compress_enabled', True))
        self.target_image_tokens_var = tk.StringVar(value=str(self.config.get('target_image_tokens', 1000)))
        self.image_format_var = tk.StringVar(value=self.config.get('image_compression_format', 'auto'))
        self.webp_quality_var = tk.IntVar(value=self.config.get('webp_quality', 85))
        self.jpeg_quality_var = tk.IntVar(value=self.config.get('jpeg_quality', 85))
        self.png_compression_var = tk.IntVar(value=self.config.get('png_compression', 6))
        self.max_image_dimension_var = tk.StringVar(value=str(self.config.get('max_image_dimension', 2048)))
        self.max_image_size_mb_var = tk.StringVar(value=str(self.config.get('max_image_size_mb', 10)))
        self.preserve_transparency_var = tk.BooleanVar(value=self.config.get('preserve_transparency', False)) 
        self.preserve_original_format_var = tk.BooleanVar(value=self.config.get('preserve_original_format', False)) 
        self.optimize_for_ocr_var = tk.BooleanVar(value=self.config.get('optimize_for_ocr', True))
        self.progressive_encoding_var = tk.BooleanVar(value=self.config.get('progressive_encoding', True))
        self.save_compressed_images_var = tk.BooleanVar(value=self.config.get('save_compressed_images', False))
        self.image_chunk_overlap_var = tk.StringVar(value=str(self.config.get('image_chunk_overlap', '1')))

        # Glossary-related variables (existing)
        self.append_glossary_var = tk.BooleanVar(value=self.config.get('append_glossary', False))
        self.glossary_min_frequency_var = tk.StringVar(value=str(self.config.get('glossary_min_frequency', 2)))
        self.glossary_max_names_var = tk.StringVar(value=str(self.config.get('glossary_max_names', 50)))
        self.glossary_max_titles_var = tk.StringVar(value=str(self.config.get('glossary_max_titles', 30)))
        self.glossary_batch_size_var = tk.StringVar(value=str(self.config.get('glossary_batch_size', 50)))
        self.glossary_max_text_size_var = tk.StringVar(value=str(self.config.get('glossary_max_text_size', 50000)))
        self.glossary_chapter_split_threshold_var = tk.StringVar(value=self.config.get('glossary_chapter_split_threshold', '8192'))
        self.glossary_max_sentences_var = tk.StringVar(value=str(self.config.get('glossary_max_sentences', 200)))
        self.glossary_filter_mode_var = tk.StringVar(value=self.config.get('glossary_filter_mode', 'all'))

        
        # NEW: Additional glossary settings
        self.strip_honorifics_var = tk.BooleanVar(value=self.config.get('strip_honorifics', True))
        self.disable_honorifics_var = tk.BooleanVar(value=self.config.get('glossary_disable_honorifics_filter', False))
        self.manual_temp_var = tk.StringVar(value=str(self.config.get('manual_glossary_temperature', 0.3)))
        self.manual_context_var = tk.StringVar(value=str(self.config.get('manual_context_limit', 5)))
        
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
        self.openai_base_url_var = tk.StringVar(value=self.config.get('openai_base_url', ''))
        self.groq_base_url_var = tk.StringVar(value=self.config.get('groq_base_url', ''))
        self.fireworks_base_url_var = tk.StringVar(value=self.config.get('fireworks_base_url', ''))
        self.use_custom_openai_endpoint_var = tk.BooleanVar(value=self.config.get('use_custom_openai_endpoint', False))
        
        # Initialize metadata/batch variables the same way
        self.translate_metadata_fields = self.config.get('translate_metadata_fields', {})
        # Initialize metadata translation UI and prompts
        try:
            from metadata_batch_translator import MetadataBatchTranslatorUI
            self.metadata_ui = MetadataBatchTranslatorUI(self)
            # This ensures default prompts are in config
        except ImportError:
            print("Metadata translation UI not available")
        self.batch_translate_headers_var = tk.BooleanVar(value=self.config.get('batch_translate_headers', False))
        self.headers_per_batch_var = tk.StringVar(value=self.config.get('headers_per_batch', '400'))
        self.update_html_headers_var = tk.BooleanVar(value=self.config.get('update_html_headers', True))
        self.save_header_translations_var = tk.BooleanVar(value=self.config.get('save_header_translations', True))
        self.ignore_header_var = tk.BooleanVar(value=self.config.get('ignore_header', False))
        self.ignore_title_var = tk.BooleanVar(value=self.config.get('ignore_title', False))
        self.attach_css_to_chapters_var = tk.BooleanVar(value=self.config.get('attach_css_to_chapters', False)) 
        
        # Retain exact source extension and disable 'response_' prefix
        self.retain_source_extension_var = tk.BooleanVar(value=self.config.get('retain_source_extension', False))

        
        self.max_output_tokens = self.config.get('max_output_tokens', self.max_output_tokens)
        self.master.after(500, lambda: self.on_model_change() if hasattr(self, 'model_var') else None)
        
        
        # Async processing settings
        self.async_wait_for_completion_var = tk.BooleanVar(value=False)
        self.async_poll_interval_var = tk.IntVar(value=60)
        
         # Enhanced filtering level
        if not hasattr(self, 'enhanced_filtering_var'):
            self.enhanced_filtering_var = tk.StringVar(
                value=self.config.get('enhanced_filtering', 'smart')
            )
        
        # Preserve structure toggle
        if not hasattr(self, 'enhanced_preserve_structure_var'):
            self.enhanced_preserve_structure_var = tk.BooleanVar(
                value=self.config.get('enhanced_preserve_structure', True)
            )
             
        # Initialize update manager AFTER config is loaded
        try:
            from update_manager import UpdateManager
            self.update_manager = UpdateManager(self, self.base_dir)
            
            # Check for updates on startup if enabled
            auto_check_enabled = self.config.get('auto_update_check', True)
            print(f"[DEBUG] Auto-update check enabled: {auto_check_enabled}")
            
            if auto_check_enabled:
                print("[DEBUG] Scheduling update check for 5 seconds from now...")
                self.master.after(5000, self._check_updates_on_startup)
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
        """Manually check for updates from the Other Settings dialog with loading animation"""
        if hasattr(self, 'update_manager') and self.update_manager:
            self._show_update_loading_and_check()
        else:
            messagebox.showerror("Update Check", 
                               "Update manager is not available.\n"
                               "Please check the GitHub releases page manually:\n"
                               "https://github.com/Shirochi-stack/Glossarion/releases")

    def _show_update_loading_and_check(self):
        """Show animated loading dialog while checking for updates"""
        import tkinter as tk
        import tkinter.ttk as ttk
        from PIL import Image, ImageTk
        import threading
        import os
        
        # Create loading dialog
        loading_dialog = tk.Toplevel(self.master)
        loading_dialog.title("Checking for Updates")
        loading_dialog.geometry("300x150")
        loading_dialog.resizable(False, False)
        loading_dialog.transient(self.master)
        loading_dialog.grab_set()
        
        # Set the proper application icon for the dialog
        try:
            # Use the same icon loading method as the main application
            load_application_icon(loading_dialog, self.base_dir)
        except Exception as e:
            print(f"Could not load icon for loading dialog: {e}")
        
        # Position dialog at mouse cursor
        try:
            mouse_x = self.master.winfo_pointerx()
            mouse_y = self.master.winfo_pointery()
            # Offset slightly so dialog doesn't cover cursor
            loading_dialog.geometry("+%d+%d" % (mouse_x + 10, mouse_y + 10))
        except:
            # Fallback to center of main window if mouse position fails
            loading_dialog.geometry("+%d+%d" % (
                self.master.winfo_rootx() + 50,
                self.master.winfo_rooty() + 50
            ))
        
        # Create main frame
        main_frame = ttk.Frame(loading_dialog, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # Try to load and resize the icon (same approach as main GUI)
        icon_label = None
        try:
            ico_path = os.path.join(self.base_dir, 'Halgakos.ico')
            if os.path.isfile(ico_path):
                # Load and resize image
                original_image = Image.open(ico_path)
                # Resize to 48x48 for loading animation
                resized_image = original_image.resize((48, 48), Image.Resampling.LANCZOS)
                self.loading_icon = ImageTk.PhotoImage(resized_image)
                
                icon_label = ttk.Label(main_frame, image=self.loading_icon)
                icon_label.pack(pady=(0, 10))
        except Exception as e:
            print(f"Could not load loading icon: {e}")
        
        # Add loading text
        loading_text = ttk.Label(main_frame, text="Checking for updates...", 
                                font=('TkDefaultFont', 11))
        loading_text.pack()
        
        # Add progress bar
        progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
        progress_bar.pack(pady=(10, 10), fill='x')
        progress_bar.start(10)  # Start animation
        
        # Animation state
        self.loading_animation_active = True
        self.loading_rotation = 0
        
        def animate_icon():
            """Animate the loading icon by rotating it"""
            if not self.loading_animation_active or not icon_label:
                return
                
            try:
                if hasattr(self, 'loading_icon'):
                    # Simple text-based animation instead of rotation
                    dots = "." * ((self.loading_rotation // 10) % 4)
                    loading_text.config(text=f"Checking for updates{dots}")
                    self.loading_rotation += 1
                    
                    # Schedule next animation frame
                    loading_dialog.after(100, animate_icon)
            except:
                pass  # Dialog might have been destroyed
        
        # Start icon animation if we have an icon
        if icon_label:
            animate_icon()
        
        def check_updates_thread():
            """Run update check in background thread"""
            try:
                # Perform the actual update check
                self.update_manager.check_for_updates(silent=False, force_show=True)
            except Exception as e:
                # Schedule error display on main thread
                loading_dialog.after(0, lambda: self._show_update_error(str(e)))
            finally:
                # Schedule cleanup on main thread
                loading_dialog.after(0, cleanup_loading)
        
        def cleanup_loading():
            """Clean up the loading dialog"""
            try:
                self.loading_animation_active = False
                progress_bar.stop()
                loading_dialog.grab_release()
                loading_dialog.destroy()
            except:
                pass  # Dialog might already be destroyed
        
        def _show_update_error(error_msg):
            """Show update check error"""
            cleanup_loading()
            messagebox.showerror("Update Check Failed", 
                               f"Failed to check for updates:\n{error_msg}")
        
        # Start the update check in a separate thread
        update_thread = threading.Thread(target=check_updates_thread, daemon=True)
        update_thread.start()
        
        # Handle dialog close
        def on_dialog_close():
            self.loading_animation_active = False
            cleanup_loading()
        
        loading_dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)
                               
    def append_log_with_api_error_detection(self, message):
        """Enhanced log appending that detects and highlights API errors"""
        # First append the regular log message
        self.append_log(message)
        
        # Check for API error patterns
        message_lower = message.lower()
        
        if "429" in message or "rate limit" in message_lower:
            # Rate limit error detected
            self.append_log("⚠️ RATE LIMIT ERROR DETECTED (HTTP 429)")
            self.append_log("   The API is throttling your requests.")
            self.append_log("   Please wait before continuing or increase the delay between requests.")
            self.append_log("   You can increase 'Delay between API calls' in settings.")
            
        elif "401" in message or "unauthorized" in message_lower:
            # Authentication error
            self.append_log("❌ AUTHENTICATION ERROR (HTTP 401)")
            self.append_log("   Your API key is invalid or missing.")
            self.append_log("   Please check your API key in the settings.")
            
        elif "403" in message or "forbidden" in message_lower:
            # Forbidden error
            self.append_log("❌ ACCESS FORBIDDEN ERROR (HTTP 403)")
            self.append_log("   You don't have permission to access this API.")
            self.append_log("   Please check your API subscription and permissions.")
            
        elif "400" in message or "bad request" in message_lower:
            # Bad request error
            self.append_log("❌ BAD REQUEST ERROR (HTTP 400)")
            self.append_log("   The API request was malformed or invalid.")
            self.append_log("   This might be due to unsupported model settings.")
            
        elif "timeout" in message_lower:
            # Timeout error
            self.append_log("⏱️ TIMEOUT ERROR")
            self.append_log("   The API request took too long to respond.")
            self.append_log("   Consider increasing timeout settings or retrying.")

    
    def create_glossary_backup(self, operation_name="manual"):
        """Create a backup of the current glossary if auto-backup is enabled"""
        # For manual backups, always proceed. For automatic backups, check the setting.
        if operation_name != "manual" and not self.config.get('glossary_auto_backup', True):
            return True
        
        if not self.current_glossary_data or not self.editor_file_var.get():
            return True
        
        try:
            # Get the original glossary file path
            original_path = self.editor_file_var.get()
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
            return messagebox.askyesno("Backup Failed", 
                                      f"Failed to create backup: {str(e)}\n\nContinue anyway?")

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
            lambda: self.epub_file_path.get() if hasattr(self, 'epub_file_path') and self.epub_file_path.get() else None,
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
            messagebox.showwarning("Not Available", "Manga translation modules not found.")
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
            messagebox.showerror("Missing Dependency", 
                               "PySide6 is required for manga translation. Please install it:\npip install PySide6")
            return
        
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
        
        # Handle window close
        def on_close():
            try:
                if self.manga_translator:
                    # Stop any running translations
                    if hasattr(self.manga_translator, 'stop_translation'):
                        self.manga_translator.stop_translation()
                    self.manga_translator = None
                dialog.close()
            except Exception as e:
                print(f"Error closing manga translator: {e}")
        
        dialog.finished.connect(on_close)
        
        # Show the dialog
        dialog.show()
        
        # Keep reference to prevent garbage collection
        self._manga_dialog = dialog
      
        
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
            self.fuzzy_threshold_var = tk.DoubleVar(value=self.config.get('glossary_fuzzy_threshold', 0.90))        
        
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
            # NEW: Translate cover.html (Skip Override)
            ('translate_cover_html_var', 'translate_cover_html', False),
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
            setattr(self, var_name, create_var(tk.BooleanVar, key, default))
        
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
            setattr(self, var_name, create_var(tk.StringVar, key, str(default)))
        
        # NEW: Initialize extraction mode variable
        self.extraction_mode_var = tk.StringVar(
            value=self.config.get('extraction_mode', 'smart')
        )
        
        self.book_title_prompt = self.config.get('book_title_prompt', 
            "Translate this book title to English while retaining any acronyms:")
        # Initialize book title system prompt
        if 'book_title_system_prompt' not in self.config:
            self.config['book_title_system_prompt'] = "You are a translator. Respond with only the translated text, nothing else. Do not add any explanation or additional content."
        
        # Profiles
        self.prompt_profiles = self.config.get('prompt_profiles', self.default_prompts.copy())
        active = self.config.get('active_profile', next(iter(self.prompt_profiles)))
        self.profile_var = tk.StringVar(value=active)
        self.lang_var = self.profile_var
        
        # Detection mode
        self.duplicate_detection_mode_var = tk.StringVar(value=self.config.get('duplicate_detection_mode', 'basic'))

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
        self.create_file_section()
        self._create_model_section()
        self._create_profile_section()
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
        self.append_log("🚀 Glossarion v5.0.5 - Ready to use!")
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
    
    def create_file_section(self):
        """Create file selection section with multi-file support"""
        # Initialize file selection variables
        self.selected_files = []
        self.current_file_index = 0
        
        # File label
        tb.Label(self.frame, text="Input File(s):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        # File entry
        self.entry_epub = tb.Entry(self.frame, width=50)
        self.entry_epub.grid(row=0, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        self.entry_epub.insert(0, "No file selected")
        
        # Create browse menu
        self.browse_menu = tk.Menu(self.master, tearoff=0, font=('Arial', 12))
        self.browse_menu.add_command(label="📄 Select Files", command=self.browse_files)
        self.browse_menu.add_command(label="📁 Select Folder", command=self.browse_folder)
        self.browse_menu.add_separator()
        self.browse_menu.add_command(label="🗑️ Clear Selection", command=self.clear_file_selection)
        
        # Create browse menu button
        self.btn_browse_menu = tb.Menubutton(
            self.frame,
            text="Browse ▼",
            menu=self.browse_menu,
            width=12,
            bootstyle="primary"
        )
        self.btn_browse_menu.grid(row=0, column=4, sticky=tk.EW, padx=5, pady=5)
        
        # File selection status label (shows file count and details)
        self.file_status_label = tb.Label(
            self.frame,
            text="",
            font=('Arial', 9),
            bootstyle="info"
        )
        self.file_status_label.grid(row=1, column=1, columnspan=3, sticky=tk.W, padx=5, pady=(0, 5))
        
        # Google Cloud Credentials button
        self.gcloud_button = tb.Button(
            self.frame, 
            text="GCloud Creds", 
            command=self.select_google_credentials, 
            width=12,
            state=tk.DISABLED,
            bootstyle="secondary"
        )
        self.gcloud_button.grid(row=2, column=4, sticky=tk.EW, padx=5, pady=5)
        
        # Vertex AI Location text entry
        self.vertex_location_var = tk.StringVar(value=self.config.get('vertex_ai_location', 'us-east5'))
        self.vertex_location_entry = tb.Entry(
            self.frame,
            textvariable=self.vertex_location_var,
            width=12
        )
        self.vertex_location_entry.grid(row=3, column=4, sticky=tk.EW, padx=5, pady=5)
        
        # Hide by default
        self.vertex_location_entry.grid_remove()
        
        # Status label for credentials
        self.gcloud_status_label = tb.Label(
            self.frame,
            text="",
            font=('Arial', 9),
            bootstyle="secondary"
        )
        self.gcloud_status_label.grid(row=2, column=1, columnspan=3, sticky=tk.W, padx=5, pady=(0, 5))
        
        # Optional: Add checkbox for enhanced functionality
        options_frame = tb.Frame(self.frame)
        options_frame.grid(row=1, column=4, columnspan=1, sticky=tk.EW, padx=5, pady=5)
        
        # Deep scan option for folders
        self.deep_scan_var = tk.BooleanVar(value=False)
        self.deep_scan_check = tb.Checkbutton(
            options_frame,
            text="include subfolders",
            variable=self.deep_scan_var,
            bootstyle="round-toggle"
        )
        self.deep_scan_check.pack(side='left')

    def select_google_credentials(self):
        """Select Google Cloud credentials JSON file"""
        filename = filedialog.askopenfilename(
            title="Select Google Cloud Credentials JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
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
                        self.gcloud_status_label.config(
                            text=f"✓ Credentials: {os.path.basename(filename)} (Project: {creds_data.get('project_id', 'Unknown')})",
                            foreground='green'
                        )
                        
                        # Set environment variable for child processes
                        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = filename
                        
                        self.append_log(f"Google Cloud credentials loaded: {os.path.basename(filename)}")
                    else:
                        messagebox.showerror(
                            "Error", 
                            "Invalid Google Cloud credentials file. Please select a valid service account JSON file."
                        )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load credentials: {str(e)}")

    def on_model_change(self, event=None):
        """Handle model selection change from dropdown or manual input"""
        # Get the current model value (from dropdown or manually typed)
        model = self.model_var.get()
        
        # Show Google Cloud Credentials button for Vertex AI models AND Google Translate
        needs_google_creds = False
        
        if '@' in model or model.startswith('vertex/') or model.startswith('vertex_ai/'):
            needs_google_creds = True
            self.vertex_location_entry.grid()  # Show location selector for Vertex
        elif model == 'google-translate':
            needs_google_creds = True
            self.vertex_location_entry.grid_remove()  # Hide location selector for Google Translate
        
        if needs_google_creds:
            self.gcloud_button.config(state=tk.NORMAL)
            
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
                                status_text = f"✓ Google Translate ready (Project: {project_id})"
                            else:
                                status_text = f"✓ Credentials: {os.path.basename(creds_path)} (Project: {project_id})"
                            
                            self.gcloud_status_label.config(
                                text=status_text,
                                foreground='green'
                            )
                    except:
                        self.gcloud_status_label.config(
                            text="⚠ Error reading credentials",
                            foreground='red'
                        )
                else:
                    self.gcloud_status_label.config(
                        text="⚠ Credentials file not found",
                        foreground='red'
                    )
            else:
                # Different prompts for different services
                if model == 'google-translate':
                    warning_text = "⚠ Google Cloud credentials needed for Translate API"
                else:
                    warning_text = "⚠ No Google Cloud credentials selected"
                
                self.gcloud_status_label.config(
                    text=warning_text,
                    foreground='orange'
                )
        else:
            # Not a Google service, hide everything
            self.gcloud_button.config(state=tk.DISABLED)
            self.vertex_location_entry.grid_remove()
            self.gcloud_status_label.config(text="")

    # Also add this to bind manual typing events to the combobox
    def setup_model_combobox_bindings(self):
        """Setup bindings for manual model input in combobox with autocomplete"""
        # Bind to key release events for live filtering and autofill
        self.model_combo.bind('<KeyRelease>', self._on_model_combo_keyrelease)
        # Commit best match on Enter
        self.model_combo.bind('<Return>', self._commit_model_autocomplete)
        # Also bind to FocusOut to catch when user clicks away after typing
        self.model_combo.bind('<FocusOut>', self.on_model_change)
        # Keep the existing binding for dropdown selection
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
    def _on_model_combo_keyrelease(self, event=None):
        """Combobox type-to-search without filtering values.
        - Keeps the full model list intact; does not replace the Combobox values.
        - Finds the best match and, if the dropdown is open, scrolls/highlights to it.
        - Does NOT auto-fill text on deletion or mid-string edits (and by default avoids autofill).
        - Calls on_model_change only when the entry text actually changes.
        """
        try:
            combo = self.model_combo
            typed = combo.get()
            prev = getattr(self, '_model_prev_text', '')
            keysym = (getattr(event, 'keysym', '') or '').lower()

            # Navigation/commit keys: don't interfere; Combobox handles selection events
            if keysym in {'up', 'down', 'left', 'right', 'return', 'escape', 'tab'}:
                return

            # Ensure we have the full source list
            if not hasattr(self, '_model_all_values') or not self._model_all_values:
                try:
                    self._model_all_values = list(combo['values'])
                except Exception:
                    self._model_all_values = []

            source = self._model_all_values

            # Compute match set without altering combobox values
            first_match = None
            if typed:
                lowered = typed.lower()
                pref = [v for v in source if v.lower().startswith(lowered)]
                cont = [v for v in source if lowered in v.lower()] if not pref else []
                if pref:
                    first_match = pref[0]
                elif cont:
                    first_match = cont[0]

            # Decide whether to perform any autofill: default to no text autofill
            grew = len(typed) > len(prev) and typed.startswith(prev)
            is_deletion = keysym in {'backspace', 'delete'} or len(typed) < len(prev)
            try:
                at_end = combo.index(tk.INSERT) == len(typed)
            except Exception:
                at_end = True
            try:
                has_selection = combo.selection_present()
            except Exception:
                has_selection = False

            # Gentle autofill only when appending at the end (not on delete or mid-string edits)
            do_autofill_text = first_match is not None and grew and at_end and not has_selection and not is_deletion

            if do_autofill_text:
                # Only complete if it's a true prefix match to avoid surprising jumps
                if first_match.lower().startswith(typed.lower()) and first_match != typed:
                    combo.set(first_match)
                    try:
                        combo.icursor(len(typed))
                        combo.selection_range(len(typed), len(first_match))
                    except Exception:
                        pass

            # If we have a match and the dropdown is open, scroll/highlight it (values intact)
            if first_match:
                self._scroll_model_list_to_value(first_match)

            # Remember current text for next event
            self._model_prev_text = typed

            # Only trigger change logic when the text actually changed
            if typed != prev:
                self.on_model_change()
        except Exception as e:
            try:
                logging.debug(f"Model combobox autocomplete error: {e}")
            except Exception:
                pass

    def _commit_model_autocomplete(self, event=None):
        """On Enter, commit to the best matching model (prefix preferred, then contains)."""
        try:
            combo = self.model_combo
            typed = combo.get()
            source = getattr(self, '_model_all_values', []) or list(combo['values'])
            match = None
            if typed:
                lowered = typed.lower()
                pref = [v for v in source if v.lower().startswith(lowered)]
                cont = [v for v in source if lowered in v.lower()] if not pref else []
                match = pref[0] if pref else (cont[0] if cont else None)
            if match and match != typed:
                combo.set(match)
            # Move cursor to end and clear any selection
            try:
                combo.icursor('end')
                try:
                    combo.selection_clear()
                except Exception:
                    combo.selection_range(0, 0)
            except Exception:
                pass
            # Update prev text and trigger change
            self._model_prev_text = combo.get()
            self.on_model_change()
        except Exception as e:
            try:
                logging.debug(f"Model combobox enter-commit error: {e}")
            except Exception:
                pass
        return "break"

    def _ensure_model_dropdown_open(self):
        """Open the combobox dropdown if it isn't already visible."""
        try:
            tkobj = self.model_combo.tk
            popdown = tkobj.eval(f'ttk::combobox::PopdownWindow {self.model_combo._w}')
            viewable = int(tkobj.eval(f'winfo viewable {popdown}'))
            if not viewable:
                # Prefer internal Post proc
                tkobj.eval(f'ttk::combobox::Post {self.model_combo._w}')
        except Exception:
            # Fallback: try keyboard event to open
            try:
                self.model_combo.event_generate('<Down>')
            except Exception:
                pass

    def _scroll_model_list_to_value(self, value: str):
        """If the combobox dropdown is open, scroll to and highlight the given value.
        Uses Tk internals for ttk::combobox to access the popdown listbox.
        Safe no-op if anything fails.
        """
        try:
            values = getattr(self, '_model_all_values', []) or list(self.model_combo['values'])
            if value not in values:
                return
            index = values.index(value)
            # Resolve the internal popdown listbox for this combobox
            popdown = self.model_combo.tk.eval(f'ttk::combobox::PopdownWindow {self.model_combo._w}')
            listbox = f'{popdown}.f.l'
            tkobj = self.model_combo.tk
            # Scroll and highlight the item
            tkobj.call(listbox, 'see', index)
            tkobj.call(listbox, 'selection', 'clear', 0, 'end')
            tkobj.call(listbox, 'selection', 'set', index)
            tkobj.call(listbox, 'activate', index)
        except Exception:
            # Dropdown may be closed or internals unavailable; ignore
            pass
    def _create_model_section(self):
        """Create model selection section"""
        tb.Label(self.frame, text="Model:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        default_model = self.config.get('model', 'gemini-2.0-flash')
        self.model_var = tk.StringVar(value=default_model)
        models = get_model_options()
        self._model_all_values = models
        self.model_combo = tb.Combobox(self.frame, textvariable=self.model_var, values=models, state="normal")
        self.model_combo.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        
        # Track previous text to make autocomplete less aggressive
        self._model_prev_text = self.model_var.get()
        
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        self.setup_model_combobox_bindings()
        self.model_var.trace('w', self._check_poe_model)
        self.on_model_change()
    
    def _create_profile_section(self):
        """Create profile/profile section"""
        tb.Label(self.frame, text="Profile:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.profile_menu = tb.Combobox(self.frame, textvariable=self.profile_var,
                                       values=list(self.prompt_profiles.keys()), state="normal")
        self.profile_menu.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        self.profile_menu.bind("<<ComboboxSelected>>", self.on_profile_select)
        self.profile_menu.bind("<Return>", self.on_profile_select)
        tb.Button(self.frame, text="Save Profile", command=self.save_profile, width=14).grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        tb.Button(self.frame, text="Delete Profile", command=self.delete_profile, width=14).grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)
    
    def _create_settings_section(self):
        """Create all settings controls"""
        # Threading delay (with extra spacing at top)
        tb.Label(self.frame, text="Threading delay (s):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=(15, 5))  # (top, bottom)
        self.thread_delay_entry = tb.Entry(self.frame, textvariable=self.thread_delay_var, width=8)
        self.thread_delay_entry.grid(row=3, column=1, sticky=tk.W, padx=5, pady=(15, 5))  # Match the label padding

        # API delay (left side)
        tb.Label(self.frame, text="API call delay (s):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.delay_entry = tb.Entry(self.frame, width=8)
        self.delay_entry.insert(0, str(self.config.get('delay', 2)))
        self.delay_entry.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)

        # Optional help text (spanning both columns)
        tb.Label(self.frame, text="(0 = simultaneous)", 
                 font=('TkDefaultFont', 8), foreground='gray').grid(row=3, column=2, sticky=tk.W, padx=5, pady=(15, 5))
        
        # Chapter Range
        tb.Label(self.frame, text="Chapter range (e.g., 5-10):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.chapter_range_entry = tb.Entry(self.frame, width=12)
        self.chapter_range_entry.insert(0, self.config.get('chapter_range', ''))
        self.chapter_range_entry.grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Token limit
        tb.Label(self.frame, text="Input Token limit:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.token_limit_entry = tb.Entry(self.frame, width=8)
        self.token_limit_entry.insert(0, str(self.config.get('token_limit', 200000)))
        self.token_limit_entry.grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.toggle_token_btn = tb.Button(self.frame, text="Disable Input Token Limit",
                                         command=self.toggle_token_limit, bootstyle="danger-outline", width=21)
        self.toggle_token_btn.grid(row=7, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Contextual Translation (right side, row 3) - with extra padding on top
        tb.Checkbutton(self.frame, text="Contextual Translation", variable=self.contextual_var,
                      command=self._on_contextual_toggle).grid(
            row=3, column=2, columnspan=2, sticky=tk.W, padx=5, pady=(25, 5))  # Added extra top padding
        
        # Translation History Limit (row 4)
        self.trans_history_label = tb.Label(self.frame, text="Translation History Limit:")
        self.trans_history_label.grid(row=4, column=2, sticky=tk.W, padx=5, pady=5)
        self.trans_history = tb.Entry(self.frame, width=6)
        self.trans_history.insert(0, str(self.config.get('translation_history_limit', 2)))
        self.trans_history.grid(row=4, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Rolling History (row 5)
        self.rolling_checkbox = tb.Checkbutton(self.frame, text="Rolling History Window", variable=self.translation_history_rolling_var,
                      bootstyle="round-toggle")
        self.rolling_checkbox.grid(row=5, column=2, sticky=tk.W, padx=5, pady=5)
        self.rolling_history_desc = tk.Label(self.frame, text="(Keep recent history instead of purging)",
                font=('TkDefaultFont', 11), fg='gray')
        self.rolling_history_desc.grid(row=5, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Temperature (row 6)
        tb.Label(self.frame, text="Temperature:").grid(row=6, column=2, sticky=tk.W, padx=5, pady=5)
        self.trans_temp = tb.Entry(self.frame, width=6)
        self.trans_temp.insert(0, str(self.config.get('translation_temperature', 0.3)))
        self.trans_temp.grid(row=6, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Batch Translation (row 7)
        self.batch_checkbox = tb.Checkbutton(self.frame, text="Batch Translation", variable=self.batch_translation_var,
                      bootstyle="round-toggle")
        self.batch_checkbox.grid(row=7, column=2, sticky=tk.W, padx=5, pady=5)
        self.batch_size_entry = tb.Entry(self.frame, width=6, textvariable=self.batch_size_var)
        self.batch_size_entry.grid(row=7, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Set batch entry state
        self.batch_size_entry.config(state=tk.NORMAL if self.batch_translation_var.get() else tk.DISABLED)
        self.batch_translation_var.trace('w', lambda *args: self.batch_size_entry.config(
            state=tk.NORMAL if self.batch_translation_var.get() else tk.DISABLED))
        
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
        
        # Set initial state based on contextual translation
        self._on_contextual_toggle()

    def _on_contextual_toggle(self):
        """Handle contextual translation toggle - enable/disable related controls"""
        is_contextual = self.contextual_var.get()
        
        # Disable controls when contextual is ON, enable when OFF
        state = tk.NORMAL if is_contextual else tk.DISABLED
        
        # Disable/enable translation history limit entry and gray out label
        self.trans_history.config(state=state)
        self.trans_history_label.config(foreground='white' if is_contextual else 'gray')
        
        # Disable/enable rolling history checkbox and gray out description
        self.rolling_checkbox.config(state=state)
        self.rolling_history_desc.config(foreground='gray' if is_contextual else '#404040')
    
    def _create_api_section(self):
        """Create API key section"""
        self.api_key_label = tb.Label(self.frame, text="OpenAI/Gemini/... API Key:")
        self.api_key_label.grid(row=8, column=0, sticky=tk.W, padx=5, pady=5)
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
        """Create system prompt section with UIHelper"""
        tb.Label(self.frame, text="System Prompt:").grid(row=9, column=0, sticky=tk.NW, padx=5, pady=5)
        
        # Use UIHelper to create text widget with undo/redo
        self.prompt_text = self.ui.setup_scrollable_text(
            self.frame, 
            height=5, 
            width=60, 
            wrap='word'
        )
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
        
        # Setup resize handler
        self._resize_handler = self.ui.create_button_resize_handler(
            self.run_button, 
            self.run_base_w, 
            self.run_base_h,
            self.master,
            BASE_WIDTH,
            BASE_HEIGHT
        )
    
    def _create_log_section(self):
        """Create log text area with UIHelper"""
        self.log_text = scrolledtext.ScrolledText(self.frame, wrap=tk.WORD)
        self.log_text.grid(row=10, column=0, columnspan=5, sticky=tk.NSEW, padx=5, pady=5)
        
        # Use UIHelper to block editing
        self.ui.block_text_editing(self.log_text)
        
        # Setup context menu
        self.log_text.bind("<Button-3>", self._show_context_menu)
        if sys.platform == "darwin":
            self.log_text.bind("<Button-2>", self._show_context_menu)

    def _check_poe_model(self, *args):
        """Automatically show POE helper when POE model is selected"""
        model = self.model_var.get().lower()
        
        # Check if POE model is selected
        if model.startswith('poe/'):
            current_key = self.api_key_entry.get().strip()
            
            # Only show helper if no valid POE cookie is set
            if not current_key.startswith('p-b:'):
                # Use a flag to prevent showing multiple times in same session
                if not getattr(self, '_poe_helper_shown', False):
                    self._poe_helper_shown = True
                    # Change self.root to self.master
                    self.master.after(100, self._show_poe_setup_dialog)
        else:
            # Reset flag when switching away from POE
            self._poe_helper_shown = False

    def _show_poe_setup_dialog(self):
        """Show POE cookie setup dialog"""
        # Create dialog using WindowManager
        dialog, scrollable_frame, canvas = self.wm.setup_scrollable(
            self.master,
            "POE Authentication Required",
            width=650,
            height=450,
            max_width_ratio=0.8,
            max_height_ratio=0.85
        )
        
        # Header
        header_frame = tk.Frame(scrollable_frame)
        header_frame.pack(fill='x', padx=20, pady=(20, 10))
        
        tk.Label(header_frame, text="POE Cookie Authentication",
                font=('TkDefaultFont', 12, 'bold')).pack()
        
        # Important notice
        notice_frame = tk.Frame(scrollable_frame)
        notice_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        tk.Label(notice_frame, 
                text="⚠️ POE uses HttpOnly cookies that cannot be accessed by JavaScript",
                foreground='red', font=('TkDefaultFont', 10, 'bold')).pack()
        
        tk.Label(notice_frame,
                text="You must manually copy the cookie from Developer Tools",
                foreground='gray').pack()
        
        # Instructions
        self._create_poe_manual_instructions(scrollable_frame)
        
        # Button
        button_frame = tk.Frame(scrollable_frame)
        button_frame.pack(fill='x', padx=20, pady=(10, 20))
        
        def close_dialog():
            dialog.destroy()
            # Check if user added a cookie
            current_key = self.api_key_entry.get().strip()
            if model := self.model_var.get().lower():
                if model.startswith('poe/') and not current_key.startswith('p-b:'):
                    self.append_log("⚠️ POE models require cookie authentication. Please add your p-b cookie to the API key field.")
        
        tb.Button(button_frame, text="Close", command=close_dialog,
                 bootstyle="secondary").pack()
        
        # Auto-resize and show
        self.wm.auto_resize_dialog(dialog, canvas)

    def _create_poe_manual_instructions(self, parent):
        """Create manual instructions for getting POE cookie"""
        frame = tk.LabelFrame(parent, text="How to Get Your POE Cookie")
        frame.pack(fill='both', expand=True, padx=20, pady=10)
        
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
            step_frame = tk.Frame(frame)
            step_frame.pack(anchor='w', padx=20, pady=2)
            
            if style == "indent":
                tk.Label(step_frame, text="    ").pack(side='left')
            
            if num:
                tk.Label(step_frame, text=num, font=('TkDefaultFont', 10, 'bold'),
                        width=3).pack(side='left')
            
            tk.Label(step_frame, text=text).pack(side='left')
        
        # Example
        example_frame = tk.LabelFrame(parent, text="Example API Key Format")
        example_frame.pack(fill='x', padx=20, pady=(10, 0))
        
        example_entry = tk.Entry(example_frame, font=('Consolas', 11))
        example_entry.pack(padx=10, pady=10, fill='x')
        example_entry.insert(0, "p-b:RyP5ORQXFO8qXbiTBKD2vA%3D%3D")
        example_entry.config(state='readonly')
        
        # Additional info
        info_frame = tk.Frame(parent)
        info_frame.pack(fill='x', padx=20, pady=(10, 0))
        
        info_text = """Note: The cookie value is usually a long string ending with %3D%3D
    If you see multiple p-b cookies, use the one with the longest value."""
        
        tk.Label(info_frame, text=info_text, foreground='gray',
                justify='left').pack(anchor='w')

    def open_async_processing(self):
        """Open the async processing dialog"""
        # Check if translation is running
        if hasattr(self, 'translation_thread') and self.translation_thread and self.translation_thread.is_alive():
            self.append_log("⚠️ Cannot open async processing while translation is in progress.")
            messagebox.showwarning("Process Running", "Please wait for the current translation to complete.")
            return
        
        # Check if glossary extraction is running
        if hasattr(self, 'glossary_thread') and self.glossary_thread and self.glossary_thread.is_alive():
            self.append_log("⚠️ Cannot open async processing while glossary extraction is in progress.")
            messagebox.showwarning("Process Running", "Please wait for glossary extraction to complete.")
            return
        
        # Check if file is selected
        if not hasattr(self, 'file_path') or not self.file_path:
            self.append_log("⚠️ Please select a file before opening async processing.")
            messagebox.showwarning("No File Selected", "Please select an EPUB or TXT file first.")
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
            self._show_async_processing_dialog(self.master, self)
            
        except ImportError as e:
            self.append_log(f"❌ Failed to load async processing module: {e}")
            messagebox.showerror(
                "Module Not Found", 
                "The async processing module could not be loaded.\n"
                "Please ensure async_api_processor.py is in the same directory."
            )
        except Exception as e:
            self.append_log(f"❌ Error opening async processing: {e}")
            messagebox.showerror("Error", f"Failed to open async processing: {str(e)}")

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

    def configure_title_prompt(self):
        """Configure the book title translation prompt"""
        dialog = self.wm.create_simple_dialog(
            self.master,
            "Configure Book Title Translation",
            width=950,
            height=850  # Increased height for two prompts
        )
        
        main_frame = tk.Frame(dialog, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # System Prompt Section
        tk.Label(main_frame, text="System Prompt (AI Instructions)", 
                font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        tk.Label(main_frame, text="This defines how the AI should behave when translating titles:",
                font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, pady=(0, 10))
        
        self.title_system_prompt_text = self.ui.setup_scrollable_text(
            main_frame, height=4, wrap=tk.WORD
        )
        self.title_system_prompt_text.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        self.title_system_prompt_text.insert('1.0', self.config.get('book_title_system_prompt', 
            "You are a translator. Respond with only the translated text, nothing else. Do not add any explanation or additional content."))
        
        # User Prompt Section
        tk.Label(main_frame, text="User Prompt (Translation Request)", 
                font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        
        tk.Label(main_frame, text="This prompt will be used when translating book titles.\n"
                "The book title will be appended after this prompt.",
                font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, pady=(0, 10))
        
        self.title_prompt_text = self.ui.setup_scrollable_text(
            main_frame, height=6, wrap=tk.WORD
        )
        self.title_prompt_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.title_prompt_text.insert('1.0', self.book_title_prompt)
        
        lang_frame = tk.Frame(main_frame)
        lang_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(lang_frame, text="💡 Tip: Modify the prompts above to translate to other languages",
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
            
            # Save the system prompt too
            self.config['book_title_system_prompt'] = self.title_system_prompt_text.get('1.0', tk.END).strip()
            
            #messagebox.showinfo("Success", "Book title prompts saved!")
            dialog.destroy()
        
        def reset_title_prompt():
            if messagebox.askyesno("Reset Prompts", "Reset both prompts to defaults?"):
                # Reset system prompt
                default_system = "You are a translator. Respond with only the translated text, nothing else. Do not add any explanation or additional content."
                self.title_system_prompt_text.delete('1.0', tk.END)
                self.title_system_prompt_text.insert('1.0', default_system)
                
                # Reset user prompt
                default_prompt = "Translate this book title to English while retaining any acronyms:"
                self.title_prompt_text.delete('1.0', tk.END)
                self.title_prompt_text.insert('1.0', default_prompt)
        
        tb.Button(button_frame, text="Save", command=save_title_prompt, 
                 bootstyle="success", width=15).pack(side=tk.LEFT, padx=5)
        tb.Button(button_frame, text="Reset to Default", command=reset_title_prompt, 
                 bootstyle="warning", width=15).pack(side=tk.LEFT, padx=5)
        tb.Button(button_frame, text="Cancel", command=dialog.destroy, 
                 bootstyle="secondary", width=15).pack(side=tk.LEFT, padx=5)
        
        dialog.deiconify()

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
    
    def force_retranslation(self):
        """Force retranslation of specific chapters or images with improved display"""
        
        # Check for multiple file selection first
        if hasattr(self, 'selected_files') and len(self.selected_files) > 1:
            self._force_retranslation_multiple_files()
            return
        
        # Check if it's a folder selection (for images)
        if hasattr(self, 'selected_files') and len(self.selected_files) > 0:
            # Check if the first selected file is actually a folder
            first_item = self.selected_files[0]
            if os.path.isdir(first_item):
                self._force_retranslation_images_folder(first_item)
                return
        
        # Original logic for single files
        input_path = self.entry_epub.get()
        if not input_path or not os.path.isfile(input_path):
            messagebox.showerror("Error", "Please select a valid EPUB, text file, or image folder first.")
            return
        
        # Check if it's an image file
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
        if input_path.lower().endswith(image_extensions):
            self._force_retranslation_single_image(input_path)
            return
        
        # For EPUB/text files, use the shared logic
        self._force_retranslation_epub_or_text(input_path)


    def _force_retranslation_epub_or_text(self, file_path, parent_dialog=None, tab_frame=None):
        """
        Shared logic for force retranslation of EPUB/text files with OPF support
        Can be used standalone or embedded in a tab
        
        Args:
            file_path: Path to the EPUB/text file
            parent_dialog: If provided, won't create its own dialog
            tab_frame: If provided, will render into this frame instead of creating dialog
        
        Returns:
            dict: Contains all the UI elements and data for external access
        """
        
        epub_base = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = epub_base
        
        if not os.path.exists(output_dir):
            if not parent_dialog:
                messagebox.showinfo("Info", "No translation output found for this file.")
            return None
        
        progress_file = os.path.join(output_dir, "translation_progress.json")
        if not os.path.exists(progress_file):
            if not parent_dialog:
                messagebox.showinfo("Info", "No progress tracking found.")
            return None
        
        with open(progress_file, 'r', encoding='utf-8') as f:
            prog = json.load(f)
        
        # =====================================================
        # PARSE CONTENT.OPF FOR CHAPTER MANIFEST
        # =====================================================
        
        spine_chapters = []
        opf_chapter_order = {}
        is_epub = file_path.lower().endswith('.epub')
        
        if is_epub and os.path.exists(file_path):
            try:
                import xml.etree.ElementTree as ET
                import zipfile
                
                with zipfile.ZipFile(file_path, 'r') as zf:
                    # Find content.opf file
                    opf_path = None
                    opf_content = None
                    
                    # First try to find via container.xml
                    try:
                        container_content = zf.read('META-INF/container.xml')
                        container_root = ET.fromstring(container_content)
                        rootfile = container_root.find('.//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile')
                        if rootfile is not None:
                            opf_path = rootfile.get('full-path')
                    except:
                        pass
                    
                    # Fallback: search for content.opf
                    if not opf_path:
                        for name in zf.namelist():
                            if name.endswith('content.opf'):
                                opf_path = name
                                break
                    
                    if opf_path:
                        opf_content = zf.read(opf_path)
                        
                        # Parse OPF
                        root = ET.fromstring(opf_content)
                        
                        # Handle namespaces
                        ns = {'opf': 'http://www.idpf.org/2007/opf'}
                        if root.tag.startswith('{'):
                            default_ns = root.tag[1:root.tag.index('}')]
                            ns = {'opf': default_ns}
                        
                        # Get manifest - all chapter files
                        manifest_chapters = {}
                        
                        for item in root.findall('.//opf:manifest/opf:item', ns):
                            item_id = item.get('id')
                            href = item.get('href')
                            media_type = item.get('media-type', '')
                            
                            if item_id and href and ('html' in media_type.lower() or href.endswith(('.html', '.xhtml', '.htm'))):
                                filename = os.path.basename(href)
                                
                                # Skip navigation, toc, and cover files
                                if not any(skip in filename.lower() for skip in ['nav.', 'toc.', 'cover.']):
                                    manifest_chapters[item_id] = {
                                        'filename': filename,
                                        'href': href,
                                        'media_type': media_type
                                    }
                        
                        # Get spine order - the reading order
                        spine = root.find('.//opf:spine', ns)
                        
                        if spine is not None:
                            for itemref in spine.findall('opf:itemref', ns):
                                idref = itemref.get('idref')
                                if idref and idref in manifest_chapters:
                                    chapter_info = manifest_chapters[idref]
                                    filename = chapter_info['filename']
                                    
                                    # Skip navigation, toc, and cover files
                                    if not any(skip in filename.lower() for skip in ['nav.', 'toc.', 'cover.']):
                                        # Extract chapter number from filename
                                        import re
                                        matches = re.findall(r'(\d+)', filename)
                                        if matches:
                                            file_chapter_num = int(matches[-1])
                                        else:
                                            file_chapter_num = len(spine_chapters)
                                        
                                        spine_chapters.append({
                                            'id': idref,
                                            'filename': filename,
                                            'position': len(spine_chapters),
                                            'file_chapter_num': file_chapter_num,
                                            'status': 'unknown',  # Will be updated
                                            'output_file': None    # Will be updated
                                        })
                                        
                                        # Store the order for later use
                                        opf_chapter_order[filename] = len(spine_chapters) - 1
                                        
                                        # Also store without extension for matching
                                        filename_noext = os.path.splitext(filename)[0]
                                        opf_chapter_order[filename_noext] = len(spine_chapters) - 1
                        
            except Exception as e:
                print(f"Warning: Could not parse OPF: {e}")
        
        # =====================================================
        # MATCH OPF CHAPTERS WITH TRANSLATION PROGRESS
        # =====================================================
        
        # Build a map of original basenames to progress entries
        basename_to_progress = {}
        for chapter_key, chapter_info in prog.get("chapters", {}).items():
            original_basename = chapter_info.get("original_basename", "")
            if original_basename:
                if original_basename not in basename_to_progress:
                    basename_to_progress[original_basename] = []
                basename_to_progress[original_basename].append((chapter_key, chapter_info))
        
        # Also build a map of response files
        response_file_to_progress = {}
        for chapter_key, chapter_info in prog.get("chapters", {}).items():
            output_file = chapter_info.get("output_file", "")
            if output_file:
                if output_file not in response_file_to_progress:
                    response_file_to_progress[output_file] = []
                response_file_to_progress[output_file].append((chapter_key, chapter_info))
        
        # Update spine chapters with translation status
        for spine_ch in spine_chapters:
            filename = spine_ch['filename']
            chapter_num = spine_ch['file_chapter_num']
            
            # Find the actual response file that exists
            base_name = os.path.splitext(filename)[0]
            expected_response = None
            
            # Handle .htm.html -> .html conversion
            stripped_base_name = base_name
            if base_name.endswith('.htm'):
                stripped_base_name = base_name[:-4]  # Remove .htm suffix

            # Look for translated file matching base name, with or without 'response_' and with allowed extensions
            allowed_exts = ('.html', '.xhtml', '.htm')
            for file in os.listdir(output_dir):
                f_low = file.lower()
                if f_low.endswith(allowed_exts):
                    name_no_ext = os.path.splitext(file)[0]
                    core = name_no_ext[9:] if name_no_ext.startswith('response_') else name_no_ext
                    # Accept matches for:
                    # - OPF filename without last extension (base_name)
                    # - Stripped base for .htm cases
                    # - OPF filename as-is (e.g., 'chapter_02.htm') when the output file is 'chapter_02.htm.xhtml'
                    if core == base_name or core == stripped_base_name or core == filename:
                        expected_response = file
                        break

            # Fallback - per mode, prefer OPF filename when retain mode is on
            if not expected_response:
                retain = os.getenv('RETAIN_SOURCE_EXTENSION', '0') == '1' or self.config.get('retain_source_extension', False)
                if retain:
                    expected_response = filename
                else:
                    expected_response = f"response_{stripped_base_name}.html"
            
            response_path = os.path.join(output_dir, expected_response)
            
            # Check various ways to find the translation progress info
            matched_info = None
            
            # Method 1: Check by original basename
            if filename in basename_to_progress:
                entries = basename_to_progress[filename]
                if entries:
                    _, chapter_info = entries[0]
                    matched_info = chapter_info
            
            # Method 2: Check by response file (with corrected extension)
            if not matched_info and expected_response in response_file_to_progress:
                entries = response_file_to_progress[expected_response]
                if entries:
                    _, chapter_info = entries[0]
                    matched_info = chapter_info
            
            # Method 3: Search through all progress entries for matching output file
            if not matched_info:
                for chapter_key, chapter_info in prog.get("chapters", {}).items():
                    if chapter_info.get('output_file') == expected_response:
                        matched_info = chapter_info
                        break
            
            # Method 4: CRUCIAL - Match by chapter number (actual_num vs file_chapter_num)
            if not matched_info:
                for chapter_key, chapter_info in prog.get("chapters", {}).items():
                    actual_num = chapter_info.get('actual_num')
                    # Also check 'chapter_num' as fallback
                    if actual_num is None:
                        actual_num = chapter_info.get('chapter_num')
                    
                    if actual_num is not None and actual_num == chapter_num:
                        matched_info = chapter_info
                        break
            
            # Determine if translation file exists
            file_exists = os.path.exists(response_path)
            
            # Set status and output file based on findings
            if matched_info:
                # We found progress tracking info - use its status
                spine_ch['status'] = matched_info.get('status', 'unknown')
                spine_ch['output_file'] = matched_info.get('output_file', expected_response)
                spine_ch['progress_entry'] = matched_info
                
                # Handle null output_file (common for failed/in_progress chapters)
                if not spine_ch['output_file']:
                    spine_ch['output_file'] = expected_response
                
                # Keep original extension (html/xhtml/htm) as written on disk
                
                # Verify file actually exists for completed status
                if spine_ch['status'] == 'completed':
                    output_path = os.path.join(output_dir, spine_ch['output_file'])
                    if not os.path.exists(output_path):
                        spine_ch['status'] = 'file_missing'
            
            elif file_exists:
                # File exists but no progress tracking - mark as completed
                spine_ch['status'] = 'completed'
                spine_ch['output_file'] = expected_response
            
            else:
                # No file and no progress tracking - not translated
                spine_ch['status'] = 'not_translated'
                spine_ch['output_file'] = expected_response
        
        # =====================================================
        # BUILD DISPLAY INFO
        # =====================================================
        
        chapter_display_info = []
        
        if spine_chapters:
            # Use OPF order
            for spine_ch in spine_chapters:
                display_info = {
                    'key': spine_ch.get('filename', ''),
                    'num': spine_ch['file_chapter_num'],
                    'info': spine_ch.get('progress_entry', {}),
                    'output_file': spine_ch['output_file'],
                    'status': spine_ch['status'],
                    'duplicate_count': 1,
                    'entries': [],
                    'opf_position': spine_ch['position'],
                    'original_filename': spine_ch['filename']
                }
                chapter_display_info.append(display_info)
        else:
            # Fallback to original logic if no OPF
            files_to_entries = {}
            for chapter_key, chapter_info in prog.get("chapters", {}).items():
                output_file = chapter_info.get("output_file", "")
                if output_file:
                    if output_file not in files_to_entries:
                        files_to_entries[output_file] = []
                    files_to_entries[output_file].append((chapter_key, chapter_info))
            
            for output_file, entries in files_to_entries.items():
                chapter_key, chapter_info = entries[0]
                
                # Extract chapter number
                import re
                matches = re.findall(r'(\d+)', output_file)
                if matches:
                    chapter_num = int(matches[-1])
                else:
                    chapter_num = 999999
                
                # Override with stored values if available
                if 'actual_num' in chapter_info and chapter_info['actual_num'] is not None:
                    chapter_num = chapter_info['actual_num']
                elif 'chapter_num' in chapter_info and chapter_info['chapter_num'] is not None:
                    chapter_num = chapter_info['chapter_num']
                
                status = chapter_info.get("status", "unknown")
                if status == "completed_empty":
                    status = "completed"
                
                # Check file existence
                if status == "completed":
                    output_path = os.path.join(output_dir, output_file)
                    if not os.path.exists(output_path):
                        status = "file_missing"
                
                chapter_display_info.append({
                    'key': chapter_key,
                    'num': chapter_num,
                    'info': chapter_info,
                    'output_file': output_file,
                    'status': status,
                    'duplicate_count': len(entries),
                    'entries': entries
                })
            
            # Sort by chapter number
            chapter_display_info.sort(key=lambda x: x['num'] if x['num'] is not None else 999999)
        
        # =====================================================
        # CREATE UI
        # =====================================================
        
        # If no parent dialog or tab frame, create standalone dialog
        if not parent_dialog and not tab_frame:
            dialog = self.wm.create_simple_dialog(
                self.master,
                "Force Retranslation - OPF Based" if spine_chapters else "Force Retranslation",
                width=1000,
                height=700
            )
            container = dialog
        else:
            container = tab_frame or parent_dialog
            dialog = parent_dialog
        
        # Title
        title_text = "Chapters from content.opf (in reading order):" if spine_chapters else "Select chapters to retranslate:"
        tk.Label(container, text=title_text, 
                font=('Arial', 12 if not tab_frame else 11, 'bold')).pack(pady=5)
        
        # Statistics if OPF is available
        if spine_chapters:
            stats_frame = tk.Frame(container)
            stats_frame.pack(pady=5)
            
            total_chapters = len(spine_chapters)
            completed = sum(1 for ch in spine_chapters if ch['status'] == 'completed')
            missing = sum(1 for ch in spine_chapters if ch['status'] == 'not_translated')
            failed = sum(1 for ch in spine_chapters if ch['status'] in ['failed', 'qa_failed'])
            file_missing = sum(1 for ch in spine_chapters if ch['status'] == 'file_missing')
            
            tk.Label(stats_frame, text=f"Total: {total_chapters} | ", font=('Arial', 10)).pack(side=tk.LEFT)
            tk.Label(stats_frame, text=f"✅ Completed: {completed} | ", font=('Arial', 10), fg='green').pack(side=tk.LEFT)
            tk.Label(stats_frame, text=f"❌ Missing: {missing} | ", font=('Arial', 10), fg='red').pack(side=tk.LEFT)
            tk.Label(stats_frame, text=f"⚠️ Failed: {failed} | ", font=('Arial', 10), fg='orange').pack(side=tk.LEFT)
            tk.Label(stats_frame, text=f"📁 File Missing: {file_missing}", font=('Arial', 10), fg='purple').pack(side=tk.LEFT)
        
        # Main frame for listbox
        main_frame = tk.Frame(container)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10 if not tab_frame else 5, pady=5)
        
        # Create scrollbars and listbox
        h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(
            main_frame, 
            selectmode=tk.EXTENDED, 
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set,
            width=120,
            font=('Courier', 10)  # Fixed-width font for better alignment
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        v_scrollbar.config(command=listbox.yview)
        h_scrollbar.config(command=listbox.xview)
        
        # Populate listbox
        status_icons = {
            'completed': '✅',
            'failed': '❌',
            'qa_failed': '❌',
            'file_missing': '⚠️',
            'in_progress': '🔄',
            'not_translated': '❌',
            'unknown': '❓'
        }
        
        status_labels = {
            'completed': 'Completed',
            'failed': 'Failed',
            'qa_failed': 'QA Failed',
            'file_missing': 'File Missing',
            'in_progress': 'In Progress',
            'not_translated': 'Not Translated',
            'unknown': 'Unknown'
        }
        
        for info in chapter_display_info:
            chapter_num = info['num']
            status = info['status']
            output_file = info['output_file']
            icon = status_icons.get(status, '❓')
            status_label = status_labels.get(status, status)
            
            # Format display with OPF info if available
            if 'opf_position' in info:
                # OPF-based display
                original_file = info.get('original_filename', '')
                opf_pos = info['opf_position'] + 1  # 1-based for display
                
                # Format: [OPF Position] Chapter Number | Status | Original File -> Response File
                if isinstance(chapter_num, float) and chapter_num.is_integer():
                    display = f"[{opf_pos:03d}] Ch.{int(chapter_num):03d} | {icon} {status_label:15s} | {original_file:30s} -> {output_file}"
                else:
                    display = f"[{opf_pos:03d}] Ch.{chapter_num:03d} | {icon} {status_label:15s} | {original_file:30s} -> {output_file}"
            else:
                # Original format
                if isinstance(chapter_num, float) and chapter_num.is_integer():
                    display = f"Chapter {int(chapter_num):03d} | {icon} {status_label:15s} | {output_file}"
                elif isinstance(chapter_num, float):
                    display = f"Chapter {chapter_num:06.1f} | {icon} {status_label:15s} | {output_file}"
                else:
                    display = f"Chapter {chapter_num:03d} | {icon} {status_label:15s} | {output_file}"
            
            if info.get('duplicate_count', 1) > 1:
                display += f" | ({info['duplicate_count']} entries)"
            
            listbox.insert(tk.END, display)
            
            # Color code based on status
            if status == 'completed':
                listbox.itemconfig(tk.END, fg='green')
            elif status in ['failed', 'qa_failed', 'not_translated']:
                listbox.itemconfig(tk.END, fg='red')
            elif status == 'file_missing':
                listbox.itemconfig(tk.END, fg='purple')
            elif status == 'in_progress':
                listbox.itemconfig(tk.END, fg='orange')
        
        # Selection count label
        selection_count_label = tk.Label(container, text="Selected: 0", 
                                       font=('Arial', 10 if not tab_frame else 9))
        selection_count_label.pack(pady=(5, 10) if not tab_frame else 2)
        
        def update_selection_count(*args):
            count = len(listbox.curselection())
            selection_count_label.config(text=f"Selected: {count}")
        
        listbox.bind('<<ListboxSelect>>', update_selection_count)
        
        # Return data structure for external access
        result = {
            'file_path': file_path,
            'output_dir': output_dir,
            'progress_file': progress_file,
            'prog': prog,
            'spine_chapters': spine_chapters,
            'opf_chapter_order': opf_chapter_order,
            'chapter_display_info': chapter_display_info,
            'listbox': listbox,
            'selection_count_label': selection_count_label,
            'dialog': dialog,
            'container': container
        }
        
        # If standalone (no parent), add buttons
        if not parent_dialog or tab_frame:
            self._add_retranslation_buttons_opf(result)
        
        return result


    def _add_retranslation_buttons_opf(self, data, button_frame=None):
        """Add the standard button set for retranslation dialogs with OPF support"""
        
        if not button_frame:
            button_frame = tk.Frame(data['container'])
            button_frame.pack(pady=10)
        
        # Configure column weights
        for i in range(5):
            button_frame.columnconfigure(i, weight=1)
        
        # Helper functions that work with the data dict
        def select_all():
            data['listbox'].select_set(0, tk.END)
            data['selection_count_label'].config(text=f"Selected: {data['listbox'].size()}")
        
        def clear_selection():
            data['listbox'].select_clear(0, tk.END)
            data['selection_count_label'].config(text="Selected: 0")
        
        def select_status(status_to_select):
            data['listbox'].select_clear(0, tk.END)
            for idx, info in enumerate(data['chapter_display_info']):
                if status_to_select == 'failed':
                    if info['status'] in ['failed', 'qa_failed']:
                        data['listbox'].select_set(idx)
                elif status_to_select == 'missing':
                    if info['status'] in ['not_translated', 'file_missing']:
                        data['listbox'].select_set(idx)
                else:
                    if info['status'] == status_to_select:
                        data['listbox'].select_set(idx)
            count = len(data['listbox'].curselection())
            data['selection_count_label'].config(text=f"Selected: {count}")
        
        def remove_qa_failed_mark():
            selected = data['listbox'].curselection()
            if not selected:
                messagebox.showwarning("No Selection", "Please select at least one chapter.")
                return
            
            selected_chapters = [data['chapter_display_info'][i] for i in selected]
            qa_failed_chapters = [ch for ch in selected_chapters if ch['status'] == 'qa_failed']
            
            if not qa_failed_chapters:
                messagebox.showwarning("No QA Failed Chapters", 
                                     "None of the selected chapters have 'qa_failed' status.")
                return
            
            count = len(qa_failed_chapters)
            if not messagebox.askyesno("Confirm Remove QA Failed Mark", 
                                      f"Remove QA failed mark from {count} chapters?"):
                return
            
            # Remove marks
            cleared_count = 0
            for info in qa_failed_chapters:
                # Find the actual numeric key in progress by matching output_file
                target_output_file = info['output_file']
                chapter_key = None
                
                # Search through all chapters to find the one with matching output_file
                for key, ch_info in data['prog']["chapters"].items():
                    if ch_info.get('output_file') == target_output_file:
                        chapter_key = key
                        break
                
                # Update the chapter status if we found the key
                if chapter_key and chapter_key in data['prog']["chapters"]:
                    print(f"Updating chapter key {chapter_key} (output file: {target_output_file})")
                    data['prog']["chapters"][chapter_key]["status"] = "completed"
                    
                    # Remove all QA-related fields
                    fields_to_remove = ["qa_issues", "qa_timestamp", "qa_issues_found", "duplicate_confidence"]
                    for field in fields_to_remove:
                        if field in data['prog']["chapters"][chapter_key]:
                            del data['prog']["chapters"][chapter_key][field]
                    
                    cleared_count += 1
                else:
                    print(f"WARNING: Could not find chapter key for output file: {target_output_file}")
            
            # Save the updated progress
            with open(data['progress_file'], 'w', encoding='utf-8') as f:
                json.dump(data['prog'], f, ensure_ascii=False, indent=2)
            
            messagebox.showinfo("Success", f"Removed QA failed mark from {cleared_count} chapters.")
            if data.get('dialog'):
                data['dialog'].destroy()
        
        def retranslate_selected():
            selected = data['listbox'].curselection()
            if not selected:
                messagebox.showwarning("No Selection", "Please select at least one chapter.")
                return
            
            selected_chapters = [data['chapter_display_info'][i] for i in selected]
            
            # Count different types
            missing_count = sum(1 for ch in selected_chapters if ch['status'] == 'not_translated')
            existing_count = sum(1 for ch in selected_chapters if ch['status'] != 'not_translated')
            
            count = len(selected)
            if count > 10:
                if missing_count > 0 and existing_count > 0:
                    confirm_msg = f"This will:\n• Mark {missing_count} missing chapters for translation\n• Delete and retranslate {existing_count} existing chapters\n\nTotal: {count} chapters\n\nContinue?"
                elif missing_count > 0:
                    confirm_msg = f"This will mark {missing_count} missing chapters for translation.\n\nContinue?"
                else:
                    confirm_msg = f"This will delete {existing_count} translated chapters and mark them for retranslation.\n\nContinue?"
            else:
                chapters = [f"Ch.{ch['num']}" for ch in selected_chapters]
                confirm_msg = f"This will process:\n\n{', '.join(chapters)}\n\n"
                if missing_count > 0:
                    confirm_msg += f"• {missing_count} missing chapters will be marked for translation\n"
                if existing_count > 0:
                    confirm_msg += f"• {existing_count} existing chapters will be deleted and retranslated\n"
                confirm_msg += "\nContinue?"
            
            if not messagebox.askyesno("Confirm Retranslation", confirm_msg):
                return
            
            # Process chapters - DELETE FILES AND UPDATE PROGRESS
            deleted_count = 0
            marked_count = 0
            status_reset_count = 0
            progress_updated = False

            for ch_info in selected_chapters:
                output_file = ch_info['output_file']
                
                if ch_info['status'] != 'not_translated':
                    # Delete existing file
                    if output_file:
                        output_path = os.path.join(data['output_dir'], output_file)
                        try:
                            if os.path.exists(output_path):
                                os.remove(output_path)
                                deleted_count += 1
                                print(f"Deleted: {output_path}")
                        except Exception as e:
                            print(f"Failed to delete {output_path}: {e}")
                    
                    # Reset status for any completed or qa_failed chapters
                    if ch_info['status'] in ['completed', 'qa_failed']:
                        target_output_file = ch_info['output_file']
                        chapter_key = None
                        
                        # Search through all chapters to find the one with matching output_file
                        for key, ch_data in data['prog']["chapters"].items():
                            if ch_data.get('output_file') == target_output_file:
                                chapter_key = key
                                break
                        
                        # Update the chapter status if we found the key
                        if chapter_key and chapter_key in data['prog']["chapters"]:
                            old_status = ch_info['status']
                            print(f"Resetting {old_status} status to pending for chapter key {chapter_key} (output file: {target_output_file})")
                            
                            # Reset status to pending for retranslation
                            data['prog']["chapters"][chapter_key]["status"] = "pending"
                            
                            # Remove completion-related fields if they exist
                            fields_to_remove = []
                            if old_status == 'qa_failed':
                                # Remove QA-related fields for qa_failed chapters
                                fields_to_remove = ["qa_issues", "qa_timestamp", "qa_issues_found", "duplicate_confidence"]
                            elif old_status == 'completed':
                                # Remove completion-related fields if any exist for completed chapters
                                fields_to_remove = ["completion_timestamp", "final_word_count", "translation_quality_score"]
                            
                            for field in fields_to_remove:
                                if field in data['prog']["chapters"][chapter_key]:
                                    del data['prog']["chapters"][chapter_key][field]
                            
                            status_reset_count += 1
                            progress_updated = True
                        else:
                            print(f"WARNING: Could not find chapter key for {old_status} output file: {target_output_file}")
                else:
                    # Just marking for translation (no file to delete)
                    marked_count += 1
            
            # Save the updated progress if we made changes
            if progress_updated:
                try:
                    with open(data['progress_file'], 'w', encoding='utf-8') as f:
                        json.dump(data['prog'], f, ensure_ascii=False, indent=2)
                    print(f"Updated progress tracking file - reset {status_reset_count} chapter statuses to pending")
                except Exception as e:
                    print(f"Failed to update progress file: {e}")
            
            # Build success message
            success_parts = []
            if deleted_count > 0:
                success_parts.append(f"Deleted {deleted_count} files")
            if marked_count > 0:
                success_parts.append(f"marked {marked_count} missing chapters for translation")
            if status_reset_count > 0:
                success_parts.append(f"reset {status_reset_count} chapter statuses to pending")
            
            if success_parts:
                success_msg = "Successfully " + ", ".join(success_parts) + "."
                if deleted_count > 0 or marked_count > 0:
                    success_msg += f"\n\nTotal {len(selected)} chapters ready for translation."
                messagebox.showinfo("Success", success_msg)
            else:
                messagebox.showinfo("Info", "No changes made.")
            
            if data.get('dialog'):
                data['dialog'].destroy()
        
        # Add buttons - First row
        tb.Button(button_frame, text="Select All", command=select_all, 
                  bootstyle="info").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        tb.Button(button_frame, text="Clear", command=clear_selection, 
                  bootstyle="secondary").grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tb.Button(button_frame, text="Select Completed", command=lambda: select_status('completed'), 
                  bootstyle="success").grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        tb.Button(button_frame, text="Select Missing", command=lambda: select_status('missing'), 
                  bootstyle="danger").grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        tb.Button(button_frame, text="Select Failed", command=lambda: select_status('failed'), 
                  bootstyle="warning").grid(row=0, column=4, padx=5, pady=5, sticky="ew")
        
        # Second row
        tb.Button(button_frame, text="Retranslate Selected", command=retranslate_selected, 
                  bootstyle="warning").grid(row=1, column=0, columnspan=2, padx=5, pady=10, sticky="ew")
        tb.Button(button_frame, text="Remove QA Failed Mark", command=remove_qa_failed_mark, 
                  bootstyle="success").grid(row=1, column=2, columnspan=1, padx=5, pady=10, sticky="ew")
        tb.Button(button_frame, text="Cancel", command=lambda: data['dialog'].destroy() if data.get('dialog') else None, 
                  bootstyle="secondary").grid(row=1, column=3, columnspan=2, padx=5, pady=10, sticky="ew")


    def _force_retranslation_multiple_files(self):
        """Handle force retranslation when multiple files are selected - now uses shared logic"""
        
        # First, check if all selected files are images from the same folder
        # This handles the case where folder selection results in individual file selections
        if len(self.selected_files) > 1:
            all_images = True
            parent_dirs = set()
            
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
            
            for file_path in self.selected_files:
                if os.path.isfile(file_path) and file_path.lower().endswith(image_extensions):
                    parent_dirs.add(os.path.dirname(file_path))
                else:
                    all_images = False
                    break
            
            # If all files are images from the same directory, treat it as a folder selection
            if all_images and len(parent_dirs) == 1:
                folder_path = parent_dirs.pop()
                print(f"[DEBUG] Detected {len(self.selected_files)} images from same folder: {folder_path}")
                print(f"[DEBUG] Treating as folder selection")
                self._force_retranslation_images_folder(folder_path)
                return
        
        # Otherwise, continue with normal categorization
        epub_files = []
        text_files = []
        image_files = []
        folders = []
        
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
        
        for file_path in self.selected_files:
            if os.path.isdir(file_path):
                folders.append(file_path)
            elif file_path.lower().endswith('.epub'):
                epub_files.append(file_path)
            elif file_path.lower().endswith('.txt'):
                text_files.append(file_path)
            elif file_path.lower().endswith(image_extensions):
                image_files.append(file_path)
        
        # Build summary
        summary_parts = []
        if epub_files:
            summary_parts.append(f"{len(epub_files)} EPUB file(s)")
        if text_files:
            summary_parts.append(f"{len(text_files)} text file(s)")
        if image_files:
            summary_parts.append(f"{len(image_files)} image file(s)")
        if folders:
            summary_parts.append(f"{len(folders)} folder(s)")
        
        if not summary_parts:
            messagebox.showinfo("Info", "No valid files selected.")
            return
        
        # Create main dialog
        dialog = self.wm.create_simple_dialog(
            self.master,
            "Force Retranslation - Multiple Files",
            width=950,
            height=700
        )
        
        # Summary label
        tk.Label(dialog, text=f"Selected: {', '.join(summary_parts)}", 
                font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Create notebook
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Track all tab data
        tab_data = []
        tabs_created = False
        
        # Create tabs for EPUB/text files using shared logic
        for file_path in epub_files + text_files:
            file_base = os.path.splitext(os.path.basename(file_path))[0]
            
            # Quick check if output exists
            if not os.path.exists(file_base):
                continue
            
            # Create tab
            tab_frame = tk.Frame(notebook)
            tab_name = file_base[:20] + "..." if len(file_base) > 20 else file_base
            notebook.add(tab_frame, text=tab_name)
            tabs_created = True
            
            # Use shared logic to populate the tab
            tab_result = self._force_retranslation_epub_or_text(
                file_path, 
                parent_dialog=dialog, 
                tab_frame=tab_frame
            )
            
            if tab_result:
                tab_data.append(tab_result)
        
        # Create tabs for image folders (keeping existing logic for now)
        for folder_path in folders:
            folder_result = self._create_image_folder_tab(
                folder_path, 
                notebook, 
                dialog
            )
            if folder_result:
                tab_data.append(folder_result)
                tabs_created = True
        
        # If only individual image files selected and no tabs created yet
        if image_files and not tabs_created:
            # Create a single tab for all individual images
            image_tab_result = self._create_individual_images_tab(
                image_files,
                notebook,
                dialog
            )
            if image_tab_result:
                tab_data.append(image_tab_result)
                tabs_created = True
        
        # If no tabs were created, show error
        if not tabs_created:
            messagebox.showinfo("Info", 
                "No translation output found for any of the selected files.\n\n"
                "Make sure the output folders exist in your script directory.")
            dialog.destroy()
            return
        
        # Add unified button bar that works across all tabs
        self._add_multi_file_buttons(dialog, notebook, tab_data)

    def _add_multi_file_buttons(self, dialog, notebook, tab_data):
        """Add a simple cancel button at the bottom of the dialog"""
        button_frame = tk.Frame(dialog)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        tb.Button(button_frame, text="Close All", command=dialog.destroy, 
                  bootstyle="secondary").pack(side=tk.RIGHT, padx=5)
              
    def _create_individual_images_tab(self, image_files, notebook, parent_dialog):
        """Create a tab for individual image files"""
        # Create tab
        tab_frame = tk.Frame(notebook)
        notebook.add(tab_frame, text="Individual Images")
        
        # Instructions
        tk.Label(tab_frame, text=f"Selected {len(image_files)} individual image(s):", 
                 font=('Arial', 11)).pack(pady=5)
        
        # Main frame
        main_frame = tk.Frame(tab_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbars and listbox
        h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(
            main_frame,
            selectmode=tk.EXTENDED,
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set,
            width=100
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        v_scrollbar.config(command=listbox.yview)
        h_scrollbar.config(command=listbox.xview)
        
        # File info
        file_info = []
        script_dir = os.getcwd()
        
        # Check each image for translations
        for img_path in sorted(image_files):
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            
            # Look for translations in various possible locations
            found_translations = []
            
            # Check in script directory with base name
            possible_dirs = [
                os.path.join(script_dir, base_name),
                os.path.join(script_dir, f"{base_name}_translated"),
                base_name,
                f"{base_name}_translated"
            ]
            
            for output_dir in possible_dirs:
                if os.path.exists(output_dir) and os.path.isdir(output_dir):
                    # Look for HTML files
                    for file in os.listdir(output_dir):
                        if file.lower().endswith(('.html', '.xhtml', '.htm')) and base_name in file:
                            found_translations.append((output_dir, file))
            
            if found_translations:
                for output_dir, html_file in found_translations:
                    display = f"📄 {img_name} → {html_file} | ✅ Translated"
                    listbox.insert(tk.END, display)
                    
                    file_info.append({
                        'type': 'translated',
                        'source_image': img_path,
                        'output_dir': output_dir,
                        'file': html_file,
                        'path': os.path.join(output_dir, html_file)
                    })
            else:
                display = f"🖼️ {img_name} | ❌ No translation found"
                listbox.insert(tk.END, display)
        
        # Selection count
        selection_count_label = tk.Label(tab_frame, text="Selected: 0", font=('Arial', 9))
        selection_count_label.pack(pady=2)
        
        def update_selection_count(*args):
            count = len(listbox.curselection())
            selection_count_label.config(text=f"Selected: {count}")
        
        listbox.bind('<<ListboxSelect>>', update_selection_count)
        
        return {
            'type': 'individual_images',
            'listbox': listbox,
            'file_info': file_info,
            'selection_count_label': selection_count_label
        }


    def _create_image_folder_tab(self, folder_path, notebook, parent_dialog):
        """Create a tab for image folder retranslation"""
        folder_name = os.path.basename(folder_path)
        output_dir = f"{folder_name}_translated"
        
        if not os.path.exists(output_dir):
            return None
        
        # Create tab
        tab_frame = tk.Frame(notebook)
        tab_name = "📁 " + (folder_name[:17] + "..." if len(folder_name) > 17 else folder_name)
        notebook.add(tab_frame, text=tab_name)
        
        # Instructions
        tk.Label(tab_frame, text="Select images to retranslate:", font=('Arial', 11)).pack(pady=5)
        
        # Main frame
        main_frame = tk.Frame(tab_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbars and listbox
        h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(
            main_frame,
            selectmode=tk.EXTENDED,
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set,
            width=100
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        v_scrollbar.config(command=listbox.yview)
        h_scrollbar.config(command=listbox.xview)
        
        # Find files
        file_info = []
        
        # Add HTML files
        for file in os.listdir(output_dir):
            if file.startswith('response_'):
                # Allow response_{index}_{name}.html and compound extensions like .html.xhtml
                match = re.match(r'^response_(\d+)_([^\.]*)\.(?:html?|xhtml|htm)(?:\.xhtml)?$', file, re.IGNORECASE)
                if match:
                    index = match.group(1)
                    base_name = match.group(2)
                    display = f"📄 Image {index} | {base_name} | ✅ Translated"
                else:
                    display = f"📄 {file} | ✅ Translated"
                
                listbox.insert(tk.END, display)
                file_info.append({
                    'type': 'translated',
                    'file': file,
                    'path': os.path.join(output_dir, file)
                })
        
        # Add cover images
        images_dir = os.path.join(output_dir, "images")
        if os.path.exists(images_dir):
            for file in sorted(os.listdir(images_dir)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                    display = f"🖼️ Cover | {file} | ⏭️ Skipped"
                    listbox.insert(tk.END, display)
                    file_info.append({
                        'type': 'cover',
                        'file': file,
                        'path': os.path.join(images_dir, file)
                    })
        
        # Selection count
        selection_count_label = tk.Label(tab_frame, text="Selected: 0", font=('Arial', 9))
        selection_count_label.pack(pady=2)
        
        def update_selection_count(*args):
            count = len(listbox.curselection())
            selection_count_label.config(text=f"Selected: {count}")
        
        listbox.bind('<<ListboxSelect>>', update_selection_count)
        
        return {
            'type': 'image_folder',
            'folder_path': folder_path,
            'output_dir': output_dir,
            'listbox': listbox,
            'file_info': file_info,
            'selection_count_label': selection_count_label
        }


    def _force_retranslation_images_folder(self, folder_path):
        """Handle force retranslation for image folders"""
        folder_name = os.path.basename(folder_path)
        
        # Look for output folder in the SCRIPT'S directory, not relative to the selected folder
        script_dir = os.getcwd()  # Current working directory where the script is running
        
        # Check multiple possible output folder patterns IN THE SCRIPT DIRECTORY
        possible_output_dirs = [
            os.path.join(script_dir, folder_name),  # Script dir + folder name
            os.path.join(script_dir, f"{folder_name}_translated"),  # Script dir + folder_translated
            folder_name,  # Just the folder name in current directory
            f"{folder_name}_translated",  # folder_translated in current directory
        ]
        
        output_dir = None
        for possible_dir in possible_output_dirs:
            print(f"Checking: {possible_dir}")
            if os.path.exists(possible_dir):
                # Check if it has translation_progress.json or HTML files
                if os.path.exists(os.path.join(possible_dir, "translation_progress.json")):
                    output_dir = possible_dir
                    print(f"Found output directory with progress tracker: {output_dir}")
                    break
                # Check if it has any HTML files
                elif os.path.isdir(possible_dir):
                    try:
                        files = os.listdir(possible_dir)
                        if any(f.lower().endswith(('.html', '.xhtml', '.htm')) for f in files):
                            output_dir = possible_dir
                            print(f"Found output directory with HTML files: {output_dir}")
                            break
                    except:
                        pass
        
        if not output_dir:
            messagebox.showinfo("Info", 
                f"No translation output found for '{folder_name}'.\n\n"
                f"Selected folder: {folder_path}\n"
                f"Script directory: {script_dir}\n\n"
                f"Checked locations:\n" + "\n".join(f"- {d}" for d in possible_output_dirs))
            return
        
        print(f"Using output directory: {output_dir}")
        
        # Check for progress tracking file
        progress_file = os.path.join(output_dir, "translation_progress.json")
        has_progress_tracking = os.path.exists(progress_file)
        
        print(f"Progress tracking: {has_progress_tracking} at {progress_file}")
        
        # Find all HTML files in the output directory
        html_files = []
        image_files = []
        progress_data = None
        
        if has_progress_tracking:
            # Load progress data for image translations
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    print(f"Loaded progress data with {len(progress_data)} entries")
                    
                # Extract files from progress data
                # The structure appears to use hash keys at the root level
                for key, value in progress_data.items():
                    if isinstance(value, dict) and 'output_file' in value:
                        output_file = value['output_file']
                        # Handle both forward and backslashes in paths
                        output_file = output_file.replace('\\', '/')
                        if '/' in output_file:
                            output_file = os.path.basename(output_file)
                        html_files.append(output_file)
                        print(f"Found tracked file: {output_file}")
            except Exception as e:
                print(f"Error loading progress file: {e}")
                import traceback
                traceback.print_exc()
                has_progress_tracking = False
        
        # Also scan directory for any HTML files not in progress
        try:
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path) and file.endswith('.html') and file not in html_files:
                    html_files.append(file)
                    print(f"Found untracked HTML file: {file}")
        except Exception as e:
            print(f"Error scanning directory: {e}")
        
        # Check for images subdirectory (cover images)
        images_dir = os.path.join(output_dir, "images")
        if os.path.exists(images_dir):
            try:
                for file in os.listdir(images_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                        image_files.append(file)
            except Exception as e:
                print(f"Error scanning images directory: {e}")
        
        print(f"Total files found: {len(html_files)} HTML, {len(image_files)} images")
        
        if not html_files and not image_files:
            messagebox.showinfo("Info", 
                f"No translated files found in: {output_dir}\n\n"
                f"Progress tracking: {'Yes' if has_progress_tracking else 'No'}")
            return
        
        # Create dialog
        dialog = self.wm.create_simple_dialog(
            self.master,
            "Force Retranslation - Images",
            width=800,
            height=600
        )
        
        # Add instructions with more detail
        instruction_text = f"Output folder: {output_dir}\n"
        instruction_text += f"Found {len(html_files)} translated images and {len(image_files)} cover images"
        if has_progress_tracking:
            instruction_text += " (with progress tracking)"
        tk.Label(dialog, text=instruction_text, font=('Arial', 11), justify=tk.LEFT).pack(pady=10)
        
        # Create main frame for listbox and scrollbars
        main_frame = tk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create scrollbars
        h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create listbox
        listbox = tk.Listbox(
            main_frame, 
            selectmode=tk.EXTENDED, 
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set,
            width=100
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbars
        v_scrollbar.config(command=listbox.yview)
        h_scrollbar.config(command=listbox.xview)
        
        # Keep track of file info
        file_info = []
        
        # Add translated HTML files
        for html_file in sorted(set(html_files)):  # Use set to avoid duplicates
            # Extract original image name from HTML filename
            # Expected format: response_001_imagename.html
            match = re.match(r'response_(\d+)_(.+)\.html', html_file)
            if match:
                index = match.group(1)
                base_name = match.group(2)
                display = f"📄 Image {index} | {base_name} | ✅ Translated"
            else:
                display = f"📄 {html_file} | ✅ Translated"
            
            listbox.insert(tk.END, display)
            
            # Find the hash key for this file if progress tracking exists
            hash_key = None
            if progress_data:
                for key, value in progress_data.items():
                    if isinstance(value, dict) and 'output_file' in value:
                        if html_file in value['output_file']:
                            hash_key = key
                            break
            
            file_info.append({
                'type': 'translated',
                'file': html_file,
                'path': os.path.join(output_dir, html_file),
                'hash_key': hash_key,
                'output_dir': output_dir  # Store for later use
            })
        
        # Add cover images
        for img_file in sorted(image_files):
            display = f"🖼️ Cover | {img_file} | ⏭️ Skipped (cover)"
            listbox.insert(tk.END, display)
            file_info.append({
                'type': 'cover',
                'file': img_file,
                'path': os.path.join(images_dir, img_file),
                'hash_key': None,
                'output_dir': output_dir
            })
        
        # Selection count label
        selection_count_label = tk.Label(dialog, text="Selected: 0", font=('Arial', 10))
        selection_count_label.pack(pady=(5, 10))
        
        def update_selection_count(*args):
            count = len(listbox.curselection())
            selection_count_label.config(text=f"Selected: {count}")
        
        listbox.bind('<<ListboxSelect>>', update_selection_count)
        
        # Button frame
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        
        # Configure grid columns
        for i in range(4):
            button_frame.columnconfigure(i, weight=1)
        
        def select_all():
            listbox.select_set(0, tk.END)
            update_selection_count()
        
        def clear_selection():
            listbox.select_clear(0, tk.END)
            update_selection_count()
        
        def select_translated():
            listbox.select_clear(0, tk.END)
            for idx, info in enumerate(file_info):
                if info['type'] == 'translated':
                    listbox.select_set(idx)
            update_selection_count()
        
        def mark_as_skipped():
            """Move selected images to the images folder to be skipped"""
            selected = listbox.curselection()
            if not selected:
                messagebox.showwarning("No Selection", "Please select at least one image to mark as skipped.")
                return
            
            # Get all selected items
            selected_items = [(i, file_info[i]) for i in selected]
            
            # Filter out items already in images folder (covers)
            items_to_move = [(i, item) for i, item in selected_items if item['type'] != 'cover']
            
            if not items_to_move:
                messagebox.showinfo("Info", "Selected items are already in the images folder (skipped).")
                return
            
            count = len(items_to_move)
            if not messagebox.askyesno("Confirm Mark as Skipped", 
                                      f"Move {count} translated image(s) to the images folder?\n\n"
                                      "This will:\n"
                                      "• Delete the translated HTML files\n"
                                      "• Copy source images to the images folder\n"
                                      "• Skip these images in future translations"):
                return
            
            # Create images directory if it doesn't exist
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            moved_count = 0
            failed_count = 0
            
            for idx, item in items_to_move:
                try:
                    # Extract the original image name from the HTML filename
                    # Expected format: response_001_imagename.html (also accept compound extensions)
                    html_file = item['file']
                    match = re.match(r'^response_\d+_([^\.]*)\.(?:html?|xhtml|htm)(?:\.xhtml)?$', html_file, re.IGNORECASE)
                    
                    if match:
                        base_name = match.group(1)
                        # Try to find the original image with common extensions
                        original_found = False
                        
                        for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                            # Check in the parent folder (where source images are)
                            possible_source = os.path.join(folder_path, base_name + ext)
                            if os.path.exists(possible_source):
                                # Copy to images folder
                                dest_path = os.path.join(images_dir, base_name + ext)
                                if not os.path.exists(dest_path):
                                    import shutil
                                    shutil.copy2(possible_source, dest_path)
                                    print(f"Copied {base_name + ext} to images folder")
                                original_found = True
                                break
                        
                        if not original_found:
                            print(f"Warning: Could not find original image for {html_file}")
                    
                    # Delete the HTML translation file
                    if os.path.exists(item['path']):
                        os.remove(item['path'])
                        print(f"Deleted translation: {item['path']}")
                        
                        # Remove from progress tracking if applicable
                        if progress_data and item.get('hash_key') and item['hash_key'] in progress_data:
                            del progress_data[item['hash_key']]
                    
                    # Update the listbox display
                    display = f"🖼️ Skipped | {base_name if match else item['file']} | ⏭️ Moved to images folder"
                    listbox.delete(idx)
                    listbox.insert(idx, display)
                    
                    # Update file_info
                    file_info[idx] = {
                        'type': 'cover',  # Treat as cover type since it's in images folder
                        'file': base_name + ext if match and original_found else item['file'],
                        'path': os.path.join(images_dir, base_name + ext if match and original_found else item['file']),
                        'hash_key': None,
                        'output_dir': output_dir
                    }
                    
                    moved_count += 1
                    
                except Exception as e:
                    print(f"Failed to process {item['file']}: {e}")
                    failed_count += 1
            
            # Save updated progress if modified
            if progress_data:
                try:
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump(progress_data, f, ensure_ascii=False, indent=2)
                    print(f"Updated progress tracking file")
                except Exception as e:
                    print(f"Failed to update progress file: {e}")
            
            # Update selection count
            update_selection_count()
            
            # Show result
            if failed_count > 0:
                messagebox.showwarning("Partial Success", 
                    f"Moved {moved_count} image(s) to be skipped.\n"
                    f"Failed to process {failed_count} item(s).")
            else:
                messagebox.showinfo("Success", 
                    f"Moved {moved_count} image(s) to the images folder.\n"
                    "They will be skipped in future translations.")
        
        def retranslate_selected():
            selected = listbox.curselection()
            if not selected:
                messagebox.showwarning("No Selection", "Please select at least one file.")
                return
            
            # Count types
            translated_count = sum(1 for i in selected if file_info[i]['type'] == 'translated')
            cover_count = sum(1 for i in selected if file_info[i]['type'] == 'cover')
            
            # Build confirmation message
            msg_parts = []
            if translated_count > 0:
                msg_parts.append(f"{translated_count} translated image(s)")
            if cover_count > 0:
                msg_parts.append(f"{cover_count} cover image(s)")
            
            confirm_msg = f"This will delete {' and '.join(msg_parts)}.\n\nContinue?"
            
            if not messagebox.askyesno("Confirm Deletion", confirm_msg):
                return
            
            # Delete selected files
            deleted_count = 0
            progress_updated = False
            
            for idx in selected:
                info = file_info[idx]
                try:
                    if os.path.exists(info['path']):
                        os.remove(info['path'])
                        deleted_count += 1
                        print(f"Deleted: {info['path']}")
                        
                        # Remove from progress tracking if applicable
                        if progress_data and info['hash_key'] and info['hash_key'] in progress_data:
                            del progress_data[info['hash_key']]
                            progress_updated = True
                            
                except Exception as e:
                    print(f"Failed to delete {info['path']}: {e}")
            
            # Save updated progress if modified
            if progress_updated and progress_data:
                try:
                    with open(progress_file, 'w', encoding='utf-8') as f:
                        json.dump(progress_data, f, ensure_ascii=False, indent=2)
                    print(f"Updated progress tracking file")
                except Exception as e:
                    print(f"Failed to update progress file: {e}")
            
            messagebox.showinfo("Success", 
                f"Deleted {deleted_count} file(s).\n\n"
                "They will be retranslated on the next run.")
            
            dialog.destroy()
        
        # Add buttons in grid layout (similar to EPUB/text retranslation)
        # Row 0: Selection buttons
        tb.Button(button_frame, text="Select All", command=select_all, 
                  bootstyle="info").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        tb.Button(button_frame, text="Clear Selection", command=clear_selection, 
                  bootstyle="secondary").grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tb.Button(button_frame, text="Select Translated", command=select_translated, 
                  bootstyle="success").grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        tb.Button(button_frame, text="Mark as Skipped", command=mark_as_skipped, 
                  bootstyle="warning").grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        
        # Row 1: Action buttons
        tb.Button(button_frame, text="Delete Selected", command=retranslate_selected, 
                  bootstyle="danger").grid(row=1, column=0, columnspan=2, padx=5, pady=10, sticky="ew")
        tb.Button(button_frame, text="Cancel", command=dialog.destroy, 
                  bootstyle="secondary").grid(row=1, column=2, columnspan=2, padx=5, pady=10, sticky="ew")
        
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
        ]
        
        # Add Manga Translator if available
        if MANGA_SUPPORT:
            toolbar_items.append(("Manga Translator", self.open_manga_translator, "primary"))
         
        # Async Processing 
        toolbar_items.append(("Async Translation", self.open_async_processing, "success"))
        
        toolbar_items.extend([
            ("Retranslate", self.force_retranslation, "warning"),
            ("Save Config", self.save_config, "secondary"),
            ("Load Glossary", self.load_glossary, "secondary"),
            ("Import Profiles", self.import_profiles, "secondary"),
            ("Export Profiles", self.export_profiles, "secondary"),
            ("📐 1080p: OFF", self.toggle_safe_ratios, "secondary"), 
        ])
        
        for idx, (lbl, cmd, style) in enumerate(toolbar_items):
            btn_frame.columnconfigure(idx, weight=1)
            btn = tb.Button(btn_frame, text=lbl, command=cmd, bootstyle=style)
            btn.grid(row=0, column=idx, sticky=tk.EW, padx=2)
            if lbl == "Extract Glossary":
                self.glossary_button = btn
            elif lbl == "EPUB Converter":
                self.epub_button = btn
            elif "1080p" in lbl:
                self.safe_ratios_btn = btn
            elif lbl == "Async Processing (50% Off)":
                self.async_button = btn
        
        self.frame.grid_rowconfigure(12, weight=0)

    def toggle_safe_ratios(self):
        """Toggle 1080p Windows ratios mode"""
        is_safe = self.wm.toggle_safe_ratios()
        
        if is_safe:
            self.safe_ratios_btn.config(
                text="📐 1080p: ON",
                bootstyle="success"
            )
            self.append_log("✅ 1080p Windows ratios enabled - all dialogs will fit on screen")
        else:
            self.safe_ratios_btn.config(
                text="📐 1080p: OFF",
                bootstyle="secondary"
            )
            self.append_log("❌ 1080p Windows ratios disabled - using default sizes")
        
        # Save preference
        self.config['force_safe_ratios'] = is_safe
        self.save_config()
 
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
                
                # Now filter out cover and nav/toc files for processing
                spine_order = []
                for item in spine_order_full:
                    # Skip navigation and cover files
                    if not any(skip in item.lower() for skip in ['nav.', 'toc.', 'cover.']):
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
            messagebox.showwarning("Process Running", "Please wait for glossary extraction to complete before starting translation.")
            return
        
        if self.translation_thread and self.translation_thread.is_alive():
            self.stop_translation()
            return
        
        # Check if files are selected
        if not hasattr(self, 'selected_files') or not self.selected_files:
            file_path = self.entry_epub.get().strip()
            if not file_path or file_path.startswith("No file selected") or "files selected" in file_path:
                messagebox.showerror("Error", "Please select file(s) to translate.")
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
        
        # Show immediate feedback that translation is starting
        self.append_log("🚀 Initializing translation process...")
        
        # Start worker immediately - no heavy operations here
        # IMPORTANT: Do NOT call _ensure_executor() here as it may be slow
        # Just start the thread directly
        thread_name = f"TranslationThread_{int(time.time())}"
        self.translation_thread = threading.Thread(
            target=self.run_translation_wrapper,
            name=thread_name,
            daemon=True
        )
        self.translation_thread.start()
        
        # Schedule button update check
        self.master.after(100, self.update_run_button)

    def run_translation_wrapper(self):
        """Wrapper that handles ALL initialization in background thread"""
        try:
            # Ensure executor is available (do this in background thread)
            if not hasattr(self, 'executor') or self.executor is None:
                try:
                    self._ensure_executor()
                except Exception as e:
                    self.append_log(f"⚠️ Could not initialize executor: {e}")
            
            # Load modules in background thread (not main thread!)
            if not self._modules_loaded:
                self.append_log("📦 Loading translation modules (this may take a moment)...")
                
                # Create a progress callback that uses append_log
                def module_progress(msg):
                    self.append_log(f"   {msg}")
                
                # Load modules with progress feedback
                if not self._lazy_load_modules(splash_callback=module_progress):
                    self.append_log("❌ Failed to load required modules")
                    return
                
                self.append_log("✅ Translation modules loaded successfully")
            
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
                            
                            # For very large files, enable aggressive optimization
                            #if file_count > 300:
                            #    os.environ['SKIP_VALIDATION'] = '1'
                            #    os.environ['LAZY_LOAD_CONTENT'] = '1'
                            #    self.append_log("🚀 Enabled aggressive optimization for very large file")
                            
                except Exception as e:
                    # If we can't check, just continue
                    pass
            
            # Set essential environment variables from current config before translation
            os.environ['BATCH_TRANSLATE_HEADERS'] = '1' if self.config.get('batch_translate_headers', False) else '0'
            os.environ['IGNORE_HEADER'] = '1' if self.config.get('ignore_header', False) else '0'
            os.environ['IGNORE_TITLE'] = '1' if self.config.get('ignore_title', False) else '0'
            
            # Now run the actual translation
            translation_completed = self.run_translation_direct()
            
            # If scanning phase toggle is enabled, launch scanner after translation
            # BUT only if translation completed successfully (not stopped by user)
            try:
                if (getattr(self, 'scan_phase_enabled_var', None) and self.scan_phase_enabled_var.get() and 
                    translation_completed and not self.stop_requested):
                    mode = self.scan_phase_mode_var.get() if hasattr(self, 'scan_phase_mode_var') else 'quick-scan'
                    self.append_log(f"🧪 Scanning phase enabled — launching QA Scanner in {mode} mode (post-translation)...")
                    # Non-interactive: skip dialogs and use auto-search
                    self.master.after(0, lambda: self.run_qa_scan(mode_override=mode, non_interactive=True))
            except Exception:
                pass
            
        except Exception as e:
            self.append_log(f"❌ Translation error: {e}")
            import traceback
            self.append_log(f"❌ Full error: {traceback.format_exc()}")
        finally:
            # Clean up environment variables
            env_vars = [
                'EXTRACTION_WORKERS', 'EXTRACTION_BATCH_SIZE',
                'EXTRACTION_PROGRESS_CALLBACK', 'EXTRACTION_PROGRESS_INTERVAL',
                'FAST_EXTRACTION', 'PARALLEL_PARSE', 'SKIP_VALIDATION',
                'LAZY_LOAD_CONTENT'
            ]
            for var in env_vars:
                if var in os.environ:
                    del os.environ[var]
            
            # Update button state on main thread
            self.master.after(0, self.update_run_button)

    def run_translation_direct(self):
        """Run translation directly - handles multiple files and different file types"""
        try:
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
                    self.append_log(f"⏹️ Translation stopped by user at file {i+1}/{total_files}")
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
            self.master.after(0, self.update_run_button)

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
            if not hasattr(self, 'enable_image_translation_var') or not self.enable_image_translation_var.get():
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
            api_key = self.api_key_entry.get().strip()
            model = self.model_var.get().strip()
            
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
            
            # Initialize API client
            try:
                from unified_api_client import UnifiedClient
                client = UnifiedClient(model=model, api_key=api_key)
                
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
                append_glossary = self.append_glossary_var.get()
            
            # Check if automatic glossary is enabled
            enable_auto_glossary = self.config.get('enable_auto_glossary', False)
            if hasattr(self, 'enable_auto_glossary_var'):
                enable_auto_glossary = self.enable_auto_glossary_var.get()
            
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
            temperature = float(self.temperature_entry.get()) if hasattr(self, 'temperature_entry') else 0.3
            max_tokens = int(self.max_output_tokens_var.get()) if hasattr(self, 'max_output_tokens_var') else 8192
            
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
                preview = system_prompt[:200] + "..." if len(system_prompt) > 200 else system_prompt
                self.append_log(f"   System prompt preview: {preview}")
            
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

            api_key = self.api_key_entry.get()
            model = self.model_var.get()
            
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
                self.append_log(f"🤖 Model: {self.model_var.get()}")
                
                # Get the system prompt and log first 100 characters
                system_prompt = self.prompt_text.get("1.0", "end").strip()
                prompt_preview = system_prompt[:200] + "..." if len(system_prompt) > 100 else system_prompt
                self.append_log(f"📝 System prompt preview: {prompt_preview}")
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
                chap_range = self.chapter_range_entry.get().strip()
                if chap_range:
                    os.environ['CHAPTER_RANGE'] = chap_range
                    self.append_log(f"📊 Chapter Range: {chap_range}")
                
                # Set other environment variables (token limits, etc.)
                if hasattr(self, 'token_limit_disabled') and self.token_limit_disabled:
                    os.environ['MAX_INPUT_TOKENS'] = ''
                else:
                    token_val = self.token_limit_entry.get().strip()
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
                    
            except Exception as e:
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
        model = self.model_var.get()
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
            extraction_method = self.text_extraction_method_var.get()
            filtering_level = self.file_filtering_level_var.get()
            
            if extraction_method == 'enhanced':
                extraction_mode = 'enhanced'
                enhanced_filtering = filtering_level
            else:
                extraction_mode = filtering_level
                enhanced_filtering = 'smart'  # default
        else:
            # Old UI variables
            extraction_mode = self.extraction_mode_var.get()
            if extraction_mode == 'enhanced':
                enhanced_filtering = getattr(self, 'enhanced_filtering_var', tk.StringVar(value='smart')).get()
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
            'MODEL': self.model_var.get(),
            'CONTEXTUAL': '1' if self.contextual_var.get() else '0',
            'SEND_INTERVAL_SECONDS': str(self.delay_entry.get()),
            'THREAD_SUBMISSION_DELAY_SECONDS': self.thread_delay_var.get().strip() or '0.5',
            'MAX_OUTPUT_TOKENS': str(self.max_output_tokens),
            'API_KEY': api_key,
            'OPENAI_API_KEY': api_key,
            'OPENAI_OR_Gemini_API_KEY': api_key,
            'GEMINI_API_KEY': api_key,
            'SYSTEM_PROMPT': self.prompt_text.get("1.0", "end").strip(),
            'TRANSLATE_BOOK_TITLE': "1" if self.translate_book_title_var.get() else "0",
            'BOOK_TITLE_PROMPT': self.book_title_prompt,
            'BOOK_TITLE_SYSTEM_PROMPT': self.config.get('book_title_system_prompt', 
                "You are a translator. Respond with only the translated text, nothing else. Do not add any explanation or additional content."),
            'REMOVE_AI_ARTIFACTS': "1" if self.REMOVE_AI_ARTIFACTS_var.get() else "0",
            'USE_ROLLING_SUMMARY': "1" if (hasattr(self, 'rolling_summary_var') and self.rolling_summary_var.get()) else ("1" if self.config.get('use_rolling_summary') else "0"),
            'SUMMARY_ROLE': self.config.get('summary_role', 'user'),
            'ROLLING_SUMMARY_EXCHANGES': self.rolling_summary_exchanges_var.get(),
            'ROLLING_SUMMARY_MODE': self.rolling_summary_mode_var.get(),
            'ROLLING_SUMMARY_SYSTEM_PROMPT': self.rolling_summary_system_prompt,
            'ROLLING_SUMMARY_USER_PROMPT': self.rolling_summary_user_prompt,
            'ROLLING_SUMMARY_MAX_ENTRIES': self.rolling_summary_max_entries_var.get(),
            'PROFILE_NAME': self.lang_var.get().lower(),
            'TRANSLATION_TEMPERATURE': str(self.trans_temp.get()),
            'TRANSLATION_HISTORY_LIMIT': str(self.trans_history.get()),
            'EPUB_OUTPUT_DIR': os.getcwd(),
            'APPEND_GLOSSARY': "1" if self.append_glossary_var.get() else "0",
            'APPEND_GLOSSARY_PROMPT': self.append_glossary_prompt,
            'EMERGENCY_PARAGRAPH_RESTORE': "1" if self.emergency_restore_var.get() else "0",
            'REINFORCEMENT_FREQUENCY': self.reinforcement_freq_var.get(),
            'RETRY_TRUNCATED': "1" if self.retry_truncated_var.get() else "0",
            'MAX_RETRY_TOKENS': self.max_retry_tokens_var.get(),
            'RETRY_DUPLICATE_BODIES': "1" if self.retry_duplicate_var.get() else "0",
            'PRESERVE_ORIGINAL_TEXT_ON_FAILURE': "1" if self.preserve_original_text_var.get() else "0",
            'DUPLICATE_LOOKBACK_CHAPTERS': self.duplicate_lookback_var.get(),
            'GLOSSARY_MIN_FREQUENCY': self.glossary_min_frequency_var.get(),
            'GLOSSARY_MAX_NAMES': self.glossary_max_names_var.get(),
            'GLOSSARY_MAX_TITLES': self.glossary_max_titles_var.get(),
            'GLOSSARY_BATCH_SIZE': self.glossary_batch_size_var.get(),
            'GLOSSARY_STRIP_HONORIFICS': "1" if self.strip_honorifics_var.get() else "0",
            'GLOSSARY_CHAPTER_SPLIT_THRESHOLD': self.glossary_chapter_split_threshold_var.get(),
            'GLOSSARY_FILTER_MODE': self.glossary_filter_mode_var.get(),
            'ENABLE_AUTO_GLOSSARY': "1" if self.enable_auto_glossary_var.get() else "0",
            'AUTO_GLOSSARY_PROMPT': self.auto_glossary_prompt if hasattr(self, 'auto_glossary_prompt') else '',
            'APPEND_GLOSSARY_PROMPT': self.append_glossary_prompt if hasattr(self, 'append_glossary_prompt') else '',
            'GLOSSARY_TRANSLATION_PROMPT': self.glossary_translation_prompt if hasattr(self, 'glossary_translation_prompt') else '',
            'GLOSSARY_FORMAT_INSTRUCTIONS': self.glossary_format_instructions if hasattr(self, 'glossary_format_instructions') else '',
            'GLOSSARY_USE_LEGACY_CSV': '1' if self.use_legacy_csv_var.get() else '0',
            'ENABLE_IMAGE_TRANSLATION': "1" if self.enable_image_translation_var.get() else "0",
            'PROCESS_WEBNOVEL_IMAGES': "1" if self.process_webnovel_images_var.get() else "0",
            'WEBNOVEL_MIN_HEIGHT': self.webnovel_min_height_var.get(),
            'MAX_IMAGES_PER_CHAPTER': self.max_images_per_chapter_var.get(),
            'IMAGE_API_DELAY': '1.0',
            'SAVE_IMAGE_TRANSLATIONS': '1',
            'IMAGE_CHUNK_HEIGHT': self.image_chunk_height_var.get(),
            'HIDE_IMAGE_TRANSLATION_LABEL': "1" if self.hide_image_translation_label_var.get() else "0",
            'RETRY_TIMEOUT': "1" if self.retry_timeout_var.get() else "0",
            'CHUNK_TIMEOUT': self.chunk_timeout_var.get(),
            # New network/HTTP controls
            'ENABLE_HTTP_TUNING': '1' if self.config.get('enable_http_tuning', False) else '0',
            'CONNECT_TIMEOUT': str(self.config.get('connect_timeout', os.environ.get('CONNECT_TIMEOUT', '10'))),
            'READ_TIMEOUT': str(self.config.get('read_timeout', os.environ.get('READ_TIMEOUT', os.environ.get('CHUNK_TIMEOUT', '180')))),
            'HTTP_POOL_CONNECTIONS': str(self.config.get('http_pool_connections', os.environ.get('HTTP_POOL_CONNECTIONS', '20'))),
            'HTTP_POOL_MAXSIZE': str(self.config.get('http_pool_maxsize', os.environ.get('HTTP_POOL_MAXSIZE', '50'))),
            'IGNORE_RETRY_AFTER': '1' if (hasattr(self, 'ignore_retry_after_var') and self.ignore_retry_after_var.get()) else '0',
            'MAX_RETRIES': str(self.config.get('max_retries', os.environ.get('MAX_RETRIES', '7'))),
            'INDEFINITE_RATE_LIMIT_RETRY': '1' if self.config.get('indefinite_rate_limit_retry', False) else '0',
            # Scanning/QA settings
            'SCAN_PHASE_ENABLED': '1' if self.config.get('scan_phase_enabled', False) else '0',
            'SCAN_PHASE_MODE': self.config.get('scan_phase_mode', 'quick-scan'),
            'QA_AUTO_SEARCH_OUTPUT': '1' if self.config.get('qa_auto_search_output', True) else '0',
            'BATCH_TRANSLATION': "1" if self.batch_translation_var.get() else "0",
            'BATCH_SIZE': self.batch_size_var.get(),
            'CONSERVATIVE_BATCHING': "1" if self.conservative_batching_var.get() else "0",
            'DISABLE_ZERO_DETECTION': "1" if self.disable_zero_detection_var.get() else "0",
            'TRANSLATION_HISTORY_ROLLING': "1" if self.translation_history_rolling_var.get() else "0",
            'USE_GEMINI_OPENAI_ENDPOINT': '1' if self.use_gemini_openai_endpoint_var.get() else '0',
            'GEMINI_OPENAI_ENDPOINT': self.gemini_openai_endpoint_var.get() if self.gemini_openai_endpoint_var.get() else '',
            "ATTACH_CSS_TO_CHAPTERS": "1" if self.attach_css_to_chapters_var.get() else "0",
            'GLOSSARY_FUZZY_THRESHOLD': str(self.config.get('glossary_fuzzy_threshold', 0.90)),
            'GLOSSARY_MAX_TEXT_SIZE': self.glossary_max_text_size_var.get(),
            'GLOSSARY_MAX_SENTENCES': self.glossary_max_sentences_var.get(),
            'USE_FALLBACK_KEYS': '1' if self.config.get('use_fallback_keys', False) else '0',
            'FALLBACK_KEYS': json.dumps(self.config.get('fallback_keys', [])),

            # Extraction settings
            "EXTRACTION_MODE": extraction_mode,
            "ENHANCED_FILTERING": enhanced_filtering,
            "ENHANCED_PRESERVE_STRUCTURE": "1" if getattr(self, 'enhanced_preserve_structure_var', tk.BooleanVar(value=True)).get() else "0",
            'FORCE_BS_FOR_TRADITIONAL': '1' if getattr(self, 'force_bs_for_traditional_var', tk.BooleanVar(value=False)).get() else '0',
            
            # For new UI
            "TEXT_EXTRACTION_METHOD": extraction_method if hasattr(self, 'text_extraction_method_var') else ('enhanced' if extraction_mode == 'enhanced' else 'standard'),
            "FILE_FILTERING_LEVEL": filtering_level if hasattr(self, 'file_filtering_level_var') else extraction_mode,
            'DISABLE_CHAPTER_MERGING': '1' if self.disable_chapter_merging_var.get() else '0',
            'DISABLE_EPUB_GALLERY': "1" if self.disable_epub_gallery_var.get() else "0",
            'DISABLE_AUTOMATIC_COVER_CREATION': "1" if getattr(self, 'disable_automatic_cover_creation_var', tk.BooleanVar(value=False)).get() else "0",
            'TRANSLATE_COVER_HTML': "1" if getattr(self, 'translate_cover_html_var', tk.BooleanVar(value=False)).get() else "0",
            'DUPLICATE_DETECTION_MODE': self.duplicate_detection_mode_var.get(),
            'CHAPTER_NUMBER_OFFSET': str(self.chapter_number_offset_var.get()), 
            'USE_HEADER_AS_OUTPUT': "1" if self.use_header_as_output_var.get() else "0",
            'ENABLE_DECIMAL_CHAPTERS': "1" if self.enable_decimal_chapters_var.get() else "0",
            'ENABLE_WATERMARK_REMOVAL': "1" if self.enable_watermark_removal_var.get() else "0",
            'ADVANCED_WATERMARK_REMOVAL': "1" if self.advanced_watermark_removal_var.get() else "0",
            'SAVE_CLEANED_IMAGES': "1" if self.save_cleaned_images_var.get() else "0",
            'COMPRESSION_FACTOR': self.compression_factor_var.get(),
            'DISABLE_GEMINI_SAFETY': str(self.config.get('disable_gemini_safety', False)).lower(),
            'GLOSSARY_DUPLICATE_KEY_MODE': self.config.get('glossary_duplicate_key_mode', 'auto'),
            'GLOSSARY_DUPLICATE_CUSTOM_FIELD': self.config.get('glossary_duplicate_custom_field', ''),
            'MANUAL_GLOSSARY': self.manual_glossary_path if hasattr(self, 'manual_glossary_path') and self.manual_glossary_path else '',
            'FORCE_NCX_ONLY': '1' if self.force_ncx_only_var.get() else '0',
            'SINGLE_API_IMAGE_CHUNKS': "1" if self.single_api_image_chunks_var.get() else "0",
            'ENABLE_GEMINI_THINKING': "1" if self.enable_gemini_thinking_var.get() else "0",
            'THINKING_BUDGET': self.thinking_budget_var.get() if self.enable_gemini_thinking_var.get() else '0',
            # GPT/OpenRouter reasoning
            'ENABLE_GPT_THINKING': "1" if self.enable_gpt_thinking_var.get() else "0",
            'GPT_REASONING_TOKENS': self.gpt_reasoning_tokens_var.get() if self.enable_gpt_thinking_var.get() else '',
            'GPT_EFFORT': self.gpt_effort_var.get(),
            'OPENROUTER_EXCLUDE': '1',
            'OPENROUTER_PREFERRED_PROVIDER': self.config.get('openrouter_preferred_provider', 'Auto'),
            # Custom API endpoints
            'OPENAI_CUSTOM_BASE_URL': self.openai_base_url_var.get() if self.openai_base_url_var.get() else '',
            'GROQ_API_URL': self.groq_base_url_var.get() if self.groq_base_url_var.get() else '',
            'FIREWORKS_API_URL': self.fireworks_base_url_var.get() if hasattr(self, 'fireworks_base_url_var') and self.fireworks_base_url_var.get() else '',
            'USE_CUSTOM_OPENAI_ENDPOINT': '1' if self.use_custom_openai_endpoint_var.get() else '0',

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
            'IMAGE_CHUNK_OVERLAP_PERCENT': self.image_chunk_overlap_var.get(),


            # Metadata and batch header translation settings
            'TRANSLATE_METADATA_FIELDS': json.dumps(self.translate_metadata_fields),
            'METADATA_TRANSLATION_MODE': self.config.get('metadata_translation_mode', 'together'),
            'BATCH_TRANSLATE_HEADERS': "1" if self.batch_translate_headers_var.get() else "0",
            'HEADERS_PER_BATCH': self.headers_per_batch_var.get(),
            'UPDATE_HTML_HEADERS': "1" if self.update_html_headers_var.get() else "0",
            'SAVE_HEADER_TRANSLATIONS': "1" if self.save_header_translations_var.get() else "0",
            'METADATA_FIELD_PROMPTS': json.dumps(self.config.get('metadata_field_prompts', {})),
            'LANG_PROMPT_BEHAVIOR': self.config.get('lang_prompt_behavior', 'auto'),
            'FORCED_SOURCE_LANG': self.config.get('forced_source_lang', 'Korean'),
            'OUTPUT_LANGUAGE': self.config.get('output_language', 'English'),
            'METADATA_BATCH_PROMPT': self.config.get('metadata_batch_prompt', ''),
            
            # AI Hunter configuration
            'AI_HUNTER_CONFIG': json.dumps(self.config.get('ai_hunter_config', {})),

            # Anti-duplicate parameters
            'ENABLE_ANTI_DUPLICATE': '1' if hasattr(self, 'enable_anti_duplicate_var') and self.enable_anti_duplicate_var.get() else '0',
            'TOP_P': str(self.top_p_var.get()) if hasattr(self, 'top_p_var') else '1.0',
            'TOP_K': str(self.top_k_var.get()) if hasattr(self, 'top_k_var') else '0',
            'FREQUENCY_PENALTY': str(self.frequency_penalty_var.get()) if hasattr(self, 'frequency_penalty_var') else '0.0',
            'PRESENCE_PENALTY': str(self.presence_penalty_var.get()) if hasattr(self, 'presence_penalty_var') else '0.0',
            'REPETITION_PENALTY': str(self.repetition_penalty_var.get()) if hasattr(self, 'repetition_penalty_var') else '1.0',
            'CANDIDATE_COUNT': str(self.candidate_count_var.get()) if hasattr(self, 'candidate_count_var') else '1',
            'CUSTOM_STOP_SEQUENCES': self.custom_stop_sequences_var.get() if hasattr(self, 'custom_stop_sequences_var') else '',
            'LOGIT_BIAS_ENABLED': '1' if hasattr(self, 'logit_bias_enabled_var') and self.logit_bias_enabled_var.get() else '0',
            'LOGIT_BIAS_STRENGTH': str(self.logit_bias_strength_var.get()) if hasattr(self, 'logit_bias_strength_var') else '-0.5',
            'BIAS_COMMON_WORDS': '1' if hasattr(self, 'bias_common_words_var') and self.bias_common_words_var.get() else '0',
            'BIAS_REPETITIVE_PHRASES': '1' if hasattr(self, 'bias_repetitive_phrases_var') and self.bias_repetitive_phrases_var.get() else '0',
            'GOOGLE_APPLICATION_CREDENTIALS': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''),
            'GOOGLE_CLOUD_PROJECT': google_cloud_project,  # Now properly set from credentials
            'VERTEX_AI_LOCATION': self.vertex_location_var.get() if hasattr(self, 'vertex_location_var') else 'us-east5',
            'IS_AZURE_ENDPOINT': '1' if (self.use_custom_openai_endpoint_var.get() and 
                                  ('.azure.com' in self.openai_base_url_var.get() or 
                                   '.cognitiveservices' in self.openai_base_url_var.get())) else '0',
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
            messagebox.showwarning("Process Running", "Please wait for translation to complete before extracting glossary.")
            return
        
        if self.glossary_thread and self.glossary_thread.is_alive():
            self.stop_glossary_extraction()
            return
        
        # Check if files are selected
        if not hasattr(self, 'selected_files') or not self.selected_files:
            # Try to get file from entry field (backward compatibility)
            file_path = self.entry_epub.get().strip()
            if not file_path or file_path.startswith("No file selected") or "files selected" in file_path:
                messagebox.showerror("Error", "Please select file(s) to extract glossary from.")
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
        else:
            thread_name = f"GlossaryThread_{int(time.time())}"
            self.glossary_thread = threading.Thread(target=self.run_glossary_extraction_direct, name=thread_name, daemon=True)
            self.glossary_thread.start()
        self.master.after(100, self.update_run_button)

    def run_glossary_extraction_direct(self):
        """Run glossary extraction directly - handles multiple files and different file types"""
        try:
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
            self.current_file_index = 0
            self.master.after(0, self.update_run_button)

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
            api_key = self.api_key_entry.get().strip()
            model = self.model_var.get().strip()
            
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
            max_tokens = int(self.max_output_tokens_var.get()) if hasattr(self, 'max_output_tokens_var') else 8192
            api_delay = float(self.delay_entry.get()) if hasattr(self, 'delay_entry') else 2.0
            
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
            api_key = self.api_key_entry.get()
            model = self.model_var.get()
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
                    'MODEL': self.model_var.get(),
                    'OPENAI_API_KEY': api_key,
                    'OPENAI_OR_Gemini_API_KEY': api_key,
                    'API_KEY': api_key,
                    'MAX_OUTPUT_TOKENS': str(self.max_output_tokens),
                    'BATCH_TRANSLATION': "1" if self.batch_translation_var.get() else "0",
                    'BATCH_SIZE': str(self.batch_size_var.get()),
                    'GLOSSARY_SYSTEM_PROMPT': self.manual_glossary_prompt,
                    'CHAPTER_RANGE': self.chapter_range_entry.get().strip(),
                    'GLOSSARY_DISABLE_HONORIFICS_FILTER': '1' if self.config.get('glossary_disable_honorifics_filter', False) else '0',
                    'GLOSSARY_HISTORY_ROLLING': "1" if self.glossary_history_rolling_var.get() else "0",
                    'DISABLE_GEMINI_SAFETY': str(self.config.get('disable_gemini_safety', False)).lower(),
                    'OPENROUTER_USE_HTTP_ONLY': '1' if self.openrouter_http_only_var.get() else '0',
                    'GLOSSARY_DUPLICATE_KEY_MODE': 'skip',  # Always use skip mode for new format
                    'SEND_INTERVAL_SECONDS': str(self.delay_entry.get()),
                    'THREAD_SUBMISSION_DELAY_SECONDS': self.thread_delay_var.get().strip() or '0.5',
                    'CONTEXTUAL': '1' if self.contextual_var.get() else '0',
                    'GOOGLE_APPLICATION_CREDENTIALS': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''),
                    
                    # NEW GLOSSARY ADDITIONS
                    'GLOSSARY_MIN_FREQUENCY': str(self.glossary_min_frequency_var.get()),
                    'GLOSSARY_MAX_NAMES': str(self.glossary_max_names_var.get()),
                    'GLOSSARY_MAX_TITLES': str(self.glossary_max_titles_var.get()),
                    'GLOSSARY_BATCH_SIZE': str(self.glossary_batch_size_var.get()),
                    'ENABLE_AUTO_GLOSSARY': "1" if self.enable_auto_glossary_var.get() else "0",
                    'APPEND_GLOSSARY': "1" if self.append_glossary_var.get() else "0",
                    'GLOSSARY_STRIP_HONORIFICS': '1' if hasattr(self, 'strip_honorifics_var') and self.strip_honorifics_var.get() else '1',
                    'AUTO_GLOSSARY_PROMPT': getattr(self, 'auto_glossary_prompt', ''),
                    'APPEND_GLOSSARY_PROMPT': getattr(self, 'append_glossary_prompt', '- Follow this reference glossary for consistent translation (Do not output any raw entries):\n'),
                    'GLOSSARY_TRANSLATION_PROMPT': getattr(self, 'glossary_translation_prompt', ''),
                    'GLOSSARY_CUSTOM_ENTRY_TYPES': json.dumps(getattr(self, 'custom_entry_types', {})),
                    'GLOSSARY_CUSTOM_FIELDS': json.dumps(getattr(self, 'custom_glossary_fields', [])),
                    'GLOSSARY_FUZZY_THRESHOLD': str(self.config.get('glossary_fuzzy_threshold', 0.90)),
                    'MANUAL_GLOSSARY': self.manual_glossary_path if hasattr(self, 'manual_glossary_path') and self.manual_glossary_path else '',
                    'GLOSSARY_FORMAT_INSTRUCTIONS': self.glossary_format_instructions if hasattr(self, 'glossary_format_instructions') else '',
                    

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
       # Run via shared executor
       self._ensure_executor()
       if self.executor:
           self.epub_future = self.executor.submit(self.run_epub_converter_direct)
           # Ensure button state is refreshed when the future completes
           def _epub_done_callback(f):
               try:
                   self.master.after(0, lambda: (setattr(self, 'epub_future', None), self.update_run_button()))
               except Exception:
                   pass
           try:
               self.epub_future.add_done_callback(_epub_done_callback)
           except Exception:
               pass
       else:
           self.epub_thread = threading.Thread(target=self.run_epub_converter_direct, daemon=True)
           self.epub_thread.start()
       self.master.after(100, self.update_run_button)
 
    def run_epub_converter_direct(self):
        """Run EPUB converter directly without blocking GUI"""
        try:
            folder = self.epub_folder
            self.append_log("📦 Starting EPUB Converter...")
            
            # Set environment variables for EPUB converter
            os.environ['DISABLE_EPUB_GALLERY'] = "1" if self.disable_epub_gallery_var.get() else "0"
            os.environ['DISABLE_AUTOMATIC_COVER_CREATION'] = "1" if getattr(self, 'disable_automatic_cover_creation_var', tk.BooleanVar(value=False)).get() else "0"
            os.environ['TRANSLATE_COVER_HTML'] = "1" if getattr(self, 'translate_cover_html_var', tk.BooleanVar(value=False)).get() else "0"

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
            api_key = self.api_key_entry.get()
            if api_key:
                os.environ['API_KEY'] = api_key
                os.environ['OPENAI_API_KEY'] = api_key
                os.environ['OPENAI_OR_Gemini_API_KEY'] = api_key
            
            model = self.model_var.get()
            if model:
                os.environ['MODEL'] = model
            
            # Set translation parameters from GUI
            os.environ['TRANSLATION_TEMPERATURE'] = str(self.trans_temp.get())
            os.environ['MAX_OUTPUT_TOKENS'] = str(self.max_output_tokens)
            
            # Set batch translation settings
            os.environ['BATCH_TRANSLATE_HEADERS'] = "1" if self.batch_translate_headers_var.get() else "0"
            os.environ['HEADERS_PER_BATCH'] = str(self.headers_per_batch_var.get())
            os.environ['UPDATE_HTML_HEADERS'] = "1" if self.update_html_headers_var.get() else "0"
            os.environ['SAVE_HEADER_TRANSLATIONS'] = "1" if self.save_header_translations_var.get() else "0"
            
            # Set metadata translation settings
            os.environ['TRANSLATE_METADATA_FIELDS'] = json.dumps(self.translate_metadata_fields)
            os.environ['METADATA_TRANSLATION_MODE'] = self.config.get('metadata_translation_mode', 'together')
            print(f"[DEBUG] METADATA_FIELD_PROMPTS from env: {os.getenv('METADATA_FIELD_PROMPTS', 'NOT SET')[:100]}...")

            # Debug: Log what we're setting
            self.append_log(f"[DEBUG] Setting TRANSLATE_METADATA_FIELDS: {self.translate_metadata_fields}")
            self.append_log(f"[DEBUG] Enabled fields: {[k for k, v in self.translate_metadata_fields.items() if v]}")
            
            # Set book title translation settings
            os.environ['TRANSLATE_BOOK_TITLE'] = "1" if self.translate_book_title_var.get() else "0"
            os.environ['BOOK_TITLE_PROMPT'] = self.book_title_prompt
            os.environ['BOOK_TITLE_SYSTEM_PROMPT'] = self.config.get('book_title_system_prompt', 
                "You are a translator. Respond with only the translated text, nothing else.")
            
            # Set prompts
            os.environ['SYSTEM_PROMPT'] = self.prompt_text.get("1.0", "end").strip()
            
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
            # Schedule GUI update on main thread
            self.master.after(0, self.update_run_button)

                
    def run_qa_scan(self, mode_override=None, non_interactive=False, preselected_files=None):
            """Run QA scan with mode selection and settings"""
            # Create a small loading window with icon
            loading_window = self.wm.create_simple_dialog(
                self.master,
                "Loading QA Scanner",
                width=300,
                height=120,
                modal=True,
                hide_initially=False
            )
            
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
            
            # Load QA scanner settings from config
            qa_settings = self.config.get('qa_scanner_settings', {
                'foreign_char_threshold': 10,
                'excluded_characters': '',
                'check_encoding_issues': False,
                'check_repetition': True,
'check_translation_artifacts': False,
                'min_file_length': 0,
                'report_format': 'detailed',
                'auto_save_report': True,
                'check_missing_html_tag': True,
                'check_invalid_nesting': False,
                'check_word_count_ratio': False,
                'check_multiple_headers': True,
                'warn_name_mismatch': True,
                'cache_enabled': True,
                'cache_auto_size': False,
                'cache_show_stats': False,
                'cache_normalize_text': 10000,
                'cache_similarity_ratio': 20000,
                'cache_content_hashes': 5000,
                'cache_semantic_fingerprint': 2000,
                'cache_structural_signature': 2000,
                'cache_translation_artifacts': 1000             
            })
            # Debug: Print current settings
            print(f"[DEBUG] QA Settings: {qa_settings}")
            print(f"[DEBUG] Word count check enabled: {qa_settings.get('check_word_count_ratio', False)}")
            
            # Optionally skip mode dialog if a mode override was provided (e.g., scanning phase)
            selected_mode_value = mode_override if mode_override else None
            if selected_mode_value is None:
                # Show mode selection dialog with settings - calculate proportional sizing
                screen_width = self.master.winfo_screenwidth()
                screen_height = self.master.winfo_screenheight()
                dialog_width = int(screen_width * 0.98)  # 98% of screen width
                dialog_height = int(screen_height * 0.80)  # 80% of screen height
                
                mode_dialog = self.wm.create_simple_dialog(
                    self.master,
                    "Select QA Scanner Mode",
                    width=dialog_width,  # Proportional width for 4-card layout
                    height=dialog_height,  # Proportional height
                    hide_initially=True
                )
            
            if selected_mode_value is None:
                # Set minimum size to prevent dialog from being too small
                mode_dialog.minsize(1200, 600)
                
                # Variables
                # selected_mode_value already set above
                
                # Main container with constrained expansion
                main_container = tk.Frame(mode_dialog)
                main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)  # Add padding
                
                # Content with padding
                main_frame = tk.Frame(main_container, padx=30, pady=20)  # Reduced padding
                main_frame.pack(fill=tk.X)  # Only fill horizontally, don't expand
                
                # Title with subtitle
                title_frame = tk.Frame(main_frame)
                title_frame.pack(pady=(0, 15))  # Further reduced
                
                tk.Label(title_frame, text="Select Detection Mode", 
                         font=('Arial', 28, 'bold'), fg='#f0f0f0').pack()  # Further reduced
                tk.Label(title_frame, text="Choose how sensitive the duplicate detection should be",
                         font=('Arial', 16), fg='#d0d0d0').pack(pady=(3, 0))  # Further reduced
                
                # Mode cards container - don't expand vertically to leave room for buttons
                modes_container = tk.Frame(main_frame)
                modes_container.pack(fill=tk.X, pady=(0, 10))  # Reduced bottom padding
                        
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
                    "recommendation": "⚡ Best for finding ALL similar content"
                },
                {
                    "value": "aggressive",
                    "emoji": "🔥",
                    "title": "AGGRESSIVE",
                    "subtitle": "75% threshold",
                    "features": [
                        "✓ Catches most duplicates",
                        "✓ Good for similar chapters",
                        "⚠ Some false positives",
                        "✓ Finds edited duplicates",
                        "✓ Moderate detection",
                        "✓ Balanced approach"
                    ],
                    "bg_color": "#3a1f1f",  # Dark red
                    "hover_color": "#8b3a3a",  # Medium red
                    "border_color": "#dc2626",
                    "accent_color": "#ef4444",
                    "recommendation": None
                },
                {
                    "value": "quick-scan",
                    "emoji": "⚡",
                    "title": "QUICK SCAN",
                    "subtitle": "85% threshold, Speed optimized",
                    "features": [
                        "✓ 3-5x faster scanning",
                        "✓ Checks consecutive chapters only",
                        "✓ Simplified analysis",
                        "✓ Skips AI Hunter",
                        "✓ Good for large libraries",
                        "✓ Minimal resource usage"
                    ],
                    "bg_color": "#1f2937",  # Dark gray
                    "hover_color": "#374151",  # Medium gray
                    "border_color": "#059669",
                    "accent_color": "#10b981",
                    "recommendation": "✅ Recommended for quick checks & large folders"
                },
                {
                    "value": "custom",
                    "emoji": "⚙️",
                    "title": "CUSTOM",
                    "subtitle": "Configurable",
                    "features": [
                        "✓ Fully customizable",
                        "✓ Set your own thresholds",
                        "✓ Advanced controls",
                        "✓ Fine-tune detection",
                        "✓ Expert mode",
                        "✓ Maximum flexibility"
                    ],
                    "bg_color": "#1e3a5f",  # Dark blue
                    "hover_color": "#2c5aa0",  # Medium blue
                    "border_color": "#3b82f6",
                    "accent_color": "#60a5fa",
                    "recommendation": None
                }
            ]
            
            # Restore original single-row layout (four cards across)
            if selected_mode_value is None:
                # Make each column share space evenly
                for col in range(len(mode_data)):
                    modes_container.columnconfigure(col, weight=1)
                # Keep row height stable
                modes_container.rowconfigure(0, weight=0)
                
                for idx, mi in enumerate(mode_data):
                    # Main card frame with initial background
                    card = tk.Frame(
                        modes_container,
                        bg=mi["bg_color"],
                        highlightbackground=mi["border_color"],
                        highlightthickness=2,
                        relief='flat'
                    )
                    card.grid(row=0, column=idx, padx=10, pady=5, sticky='nsew')
                    
                    # Content frame
                    content_frame = tk.Frame(card, bg=mi["bg_color"], cursor='hand2')
                    content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
                    
                    # Emoji
                    emoji_label = tk.Label(content_frame, text=mi["emoji"], font=('Arial', 48), bg=mi["bg_color"]) 
                    emoji_label.pack(pady=(0, 5))
                    
                    # Title
                    title_label = tk.Label(content_frame, text=mi["title"], font=('Arial', 24, 'bold'), fg='white', bg=mi["bg_color"]) 
                    title_label.pack()
                    
                    # Subtitle
                    tk.Label(content_frame, text=mi["subtitle"], font=('Arial', 14), fg=mi["accent_color"], bg=mi["bg_color"]).pack(pady=(3, 10))
                    
                    # Features
                    features_frame = tk.Frame(content_frame, bg=mi["bg_color"]) 
                    features_frame.pack(fill=tk.X)
                    for feature in mi["features"]:
                        tk.Label(features_frame, text=feature, font=('Arial', 11), fg='#e0e0e0', bg=mi["bg_color"], justify=tk.LEFT).pack(anchor=tk.W, pady=1)
                    
                    # Recommendation badge if present
                    rec_frame = None
                    rec_label = None
                    if mi["recommendation"]:
                        rec_frame = tk.Frame(content_frame, bg=mi["accent_color"]) 
                        rec_frame.pack(pady=(10, 0), fill=tk.X)
                        rec_label = tk.Label(rec_frame, text=mi["recommendation"], font=('Arial', 11, 'bold'), fg='white', bg=mi["accent_color"], padx=8, pady=4)
                        rec_label.pack()
                    
                    # Click handler
                    def make_click_handler(mode_value):
                        def handler(event=None):
                            nonlocal selected_mode_value
                            selected_mode_value = mode_value
                            mode_dialog.destroy()
                        return handler
                    click_handler = make_click_handler(mi["value"]) 
                    
                    # Hover effects for this card only
                    def create_hover_handlers(md, widgets):
                        def on_enter(event=None):
                            for w in widgets:
                                try:
                                    w.config(bg=md["hover_color"])
                                except Exception:
                                    pass
                        def on_leave(event=None):
                            for w in widgets:
                                try:
                                    w.config(bg=md["bg_color"])
                                except Exception:
                                    pass
                        return on_enter, on_leave
                    
                    all_widgets = [content_frame, emoji_label, title_label, features_frame]
                    all_widgets += [child for child in features_frame.winfo_children() if isinstance(child, tk.Label)]
                    if rec_frame is not None:
                        all_widgets += [rec_frame, rec_label]
                    on_enter, on_leave = create_hover_handlers(mi, all_widgets)
                    
                    for widget in [card, content_frame, emoji_label, title_label, features_frame] + list(features_frame.winfo_children()):
                        widget.bind("<Enter>", on_enter)
                        widget.bind("<Leave>", on_leave)
                        widget.bind("<Button-1>", click_handler)
                        try:
                            widget.config(cursor='hand2')
                        except Exception:
                            pass
            
            if selected_mode_value is None:
                # Add separator line before buttons
                separator = tk.Frame(main_frame, height=1, bg='#cccccc')  # Thinner separator
                separator.pack(fill=tk.X, pady=(10, 0))
                
                # Add settings button at the bottom
                button_frame = tk.Frame(main_frame)
                button_frame.pack(fill=tk.X, pady=(10, 5))  # Reduced padding
                
                # Create inner frame for centering buttons
                button_inner = tk.Frame(button_frame)
                button_inner.pack()
                
                def show_qa_settings():
                    """Show QA Scanner settings dialog"""
                    self.show_qa_scanner_settings(mode_dialog, qa_settings)
                
                # Auto-search checkbox - moved to left side of Scanner Settings
                if not hasattr(self, 'qa_auto_search_output_var'):
                    self.qa_auto_search_output_var = tk.BooleanVar(value=self.config.get('qa_auto_search_output', True))
                tb.Checkbutton(
                    button_inner,
                    text="Auto-search output",  # Renamed from "Auto-search output folder"
                    variable=self.qa_auto_search_output_var,
                    bootstyle="round-toggle"
                ).pack(side=tk.LEFT, padx=10)
                
                settings_btn = tb.Button(
                    button_inner,
                    text="⚙️  Scanner Settings",  # Added extra space
                    command=show_qa_settings,
                    bootstyle="info-outline",  # Changed to be more visible
                    width=18,  # Slightly smaller
                    padding=(8, 10)  # Reduced padding
                )
                settings_btn.pack(side=tk.LEFT, padx=10)
                
                cancel_btn = tb.Button(
                    button_inner,
                    text="Cancel",
                    command=lambda: mode_dialog.destroy(),
                    bootstyle="danger",  # Changed from outline to solid
                    width=12,  # Smaller
                    padding=(8, 10)  # Reduced padding
                )
                cancel_btn.pack(side=tk.LEFT, padx=10)
                
                # Handle window close (X button)
                def on_close():
                    nonlocal selected_mode_value
                    selected_mode_value = None
                    mode_dialog.destroy()
                
                mode_dialog.protocol("WM_DELETE_WINDOW", on_close)
                
                # Show dialog
                mode_dialog.deiconify()
                mode_dialog.update_idletasks()  # Force geometry update
                mode_dialog.wait_window()
                
                # Check if user selected a mode
                if selected_mode_value is None:
                    self.append_log("⚠️ QA scan canceled.")
                    return

            # End of optional mode dialog
            
            # Show custom settings dialog if custom mode is selected

            # Show custom settings dialog if custom mode is selected
            if selected_mode_value == "custom":
                # Use WindowManager's setup_scrollable for proper scrolling support
                dialog, scrollable_frame, canvas = self.wm.setup_scrollable(
                    self.master,
                    "Custom Mode Settings",
                    width=800,
                    height=650,
                    max_width_ratio=0.9,
                    max_height_ratio=0.85
                )
                
                # Variables for custom settings
                custom_settings = {
                    'similarity': tk.IntVar(value=85),
                    'semantic': tk.IntVar(value=80),
                    'structural': tk.IntVar(value=90),
                    'word_overlap': tk.IntVar(value=75),
                    'minhash_threshold': tk.IntVar(value=80),
                    'consecutive_chapters': tk.IntVar(value=2),
                    'check_all_pairs': tk.BooleanVar(value=False),
                    'sample_size': tk.IntVar(value=3000),
                    'min_text_length': tk.IntVar(value=500)
                }
                
                # Title using consistent styling
                title_label = tk.Label(scrollable_frame, text="Configure Custom Detection Settings", 
                                      font=('Arial', 20, 'bold'))
                title_label.pack(pady=(0, 20))
                
                # Detection Thresholds Section using ttkbootstrap
                threshold_frame = tb.LabelFrame(scrollable_frame, text="Detection Thresholds (%)", 
                                                padding=25, bootstyle="secondary")
                threshold_frame.pack(fill='x', padx=20, pady=(0, 25))
                
                threshold_descriptions = {
                    'similarity': ('Text Similarity', 'Character-by-character comparison'),
                    'semantic': ('Semantic Analysis', 'Meaning and context matching'),
                    'structural': ('Structural Patterns', 'Document structure similarity'),
                    'word_overlap': ('Word Overlap', 'Common words between texts'),
                    'minhash_threshold': ('MinHash Similarity', 'Fast approximate matching')
                }
                
                # Create percentage labels dictionary to store references
                percentage_labels = {}
                
                for setting_key, (label_text, description) in threshold_descriptions.items():
                    # Container for each threshold
                    row_frame = tk.Frame(threshold_frame)
                    row_frame.pack(fill='x', pady=8)
                    
                    # Left side - labels
                    label_container = tk.Frame(row_frame)
                    label_container.pack(side='left', fill='x', expand=True)
                    
                    main_label = tk.Label(label_container, text=f"{label_text} - {description}:",
                                         font=('TkDefaultFont', 11))
                    main_label.pack(anchor='w')
                    
                    # Right side - slider and percentage
                    slider_container = tk.Frame(row_frame)
                    slider_container.pack(side='right', padx=(20, 0))
                    
                    # Percentage label (shows current value)
                    percentage_label = tk.Label(slider_container, text=f"{custom_settings[setting_key].get()}%",
                                               font=('TkDefaultFont', 12, 'bold'), width=5, anchor='e')
                    percentage_label.pack(side='right', padx=(10, 0))
                    percentage_labels[setting_key] = percentage_label
                    
                    # Create slider
                    slider = tb.Scale(slider_container, 
                                     from_=10, to=100,
                                     variable=custom_settings[setting_key],
                                     bootstyle="info",
                                     length=300,
                                     orient='horizontal')
                    slider.pack(side='right')
                    
                    # Update percentage label when slider moves
                    def create_update_function(key, label):
                        def update_percentage(*args):
                            value = custom_settings[key].get()
                            label.config(text=f"{value}%")
                        return update_percentage
                    
                    # Bind the update function
                    update_func = create_update_function(setting_key, percentage_label)
                    custom_settings[setting_key].trace('w', update_func)
                
                # Processing Options Section
                options_frame = tb.LabelFrame(scrollable_frame, text="Processing Options", 
                                              padding=20, bootstyle="secondary")
                options_frame.pack(fill='x', padx=20, pady=15)
                
                # Consecutive chapters option with spinbox
                consec_frame = tk.Frame(options_frame)
                consec_frame.pack(fill='x', pady=5)
                
                tk.Label(consec_frame, text="Consecutive chapters to check:", 
                         font=('TkDefaultFont', 11)).pack(side='left')
                
                tb.Spinbox(consec_frame, from_=1, to=10, 
                           textvariable=custom_settings['consecutive_chapters'],
                           width=10, bootstyle="info").pack(side='left', padx=(10, 0))
                
                # Sample size option
                sample_frame = tk.Frame(options_frame)
                sample_frame.pack(fill='x', pady=5)
                
                tk.Label(sample_frame, text="Sample size for comparison (characters):", 
                         font=('TkDefaultFont', 11)).pack(side='left')
                
                # Sample size spinbox with larger range
                sample_spinbox = tb.Spinbox(sample_frame, from_=1000, to=10000, increment=500,
                                            textvariable=custom_settings['sample_size'],
                                            width=10, bootstyle="info")
                sample_spinbox.pack(side='left', padx=(10, 0))
                
                # Minimum text length option
                min_length_frame = tk.Frame(options_frame)
                min_length_frame.pack(fill='x', pady=5)
                
                tk.Label(min_length_frame, text="Minimum text length to process (characters):", 
                         font=('TkDefaultFont', 11)).pack(side='left')
                
                # Minimum length spinbox
                min_length_spinbox = tb.Spinbox(min_length_frame, from_=100, to=5000, increment=100,
                                                textvariable=custom_settings['min_text_length'],
                                                width=10, bootstyle="info")
                min_length_spinbox.pack(side='left', padx=(10, 0))
                
                # Check all file pairs option
                tb.Checkbutton(options_frame, text="Check all file pairs (slower but more thorough)",
                               variable=custom_settings['check_all_pairs'],
                               bootstyle="primary").pack(anchor='w', pady=8)
                
                # Create button frame at bottom (inside scrollable_frame)
                button_frame = tk.Frame(scrollable_frame)
                button_frame.pack(fill='x', pady=(30, 20))
                
                # Center buttons using inner frame
                button_inner = tk.Frame(button_frame)
                button_inner.pack()
                
                # Flag to track if settings were saved
                settings_saved = False
                
                def save_custom_settings():
                    """Save custom settings and close dialog"""
                    nonlocal settings_saved
                    qa_settings['custom_mode_settings'] = {
                        'thresholds': {
                            'similarity': custom_settings['similarity'].get() / 100,
                            'semantic': custom_settings['semantic'].get() / 100,
                            'structural': custom_settings['structural'].get() / 100,
                            'word_overlap': custom_settings['word_overlap'].get() / 100,
                            'minhash_threshold': custom_settings['minhash_threshold'].get() / 100
                        },
                        'consecutive_chapters': custom_settings['consecutive_chapters'].get(),
                        'check_all_pairs': custom_settings['check_all_pairs'].get(),
                        'sample_size': custom_settings['sample_size'].get(),
                        'min_text_length': custom_settings['min_text_length'].get()
                    }
                    settings_saved = True
                    self.append_log("✅ Custom detection settings saved")
                    dialog._cleanup_scrolling()  # Clean up scrolling bindings
                    dialog.destroy()
                
                def reset_to_defaults():
                    """Reset all values to default settings"""
                    if messagebox.askyesno("Reset to Defaults", 
                                           "Reset all values to default settings?",
                                           parent=dialog):
                        custom_settings['similarity'].set(85)
                        custom_settings['semantic'].set(80)
                        custom_settings['structural'].set(90)
                        custom_settings['word_overlap'].set(75)
                        custom_settings['minhash_threshold'].set(80)
                        custom_settings['consecutive_chapters'].set(2)
                        custom_settings['check_all_pairs'].set(False)
                        custom_settings['sample_size'].set(3000)
                        custom_settings['min_text_length'].set(500)
                        self.append_log("ℹ️ Settings reset to defaults")
                
                def cancel_settings():
                    """Cancel without saving"""
                    nonlocal settings_saved
                    if not settings_saved:
                        # Check if any settings were changed
                        defaults = {
                            'similarity': 85,
                            'semantic': 80,
                            'structural': 90,
                            'word_overlap': 75,
                            'minhash_threshold': 80,
                            'consecutive_chapters': 2,
                            'check_all_pairs': False,
                            'sample_size': 3000,
                            'min_text_length': 500
                        }
                        
                        changed = False
                        for key, default_val in defaults.items():
                            if custom_settings[key].get() != default_val:
                                changed = True
                                break
                        
                        if changed:
                            if messagebox.askyesno("Unsaved Changes", 
                                                  "You have unsaved changes. Are you sure you want to cancel?",
                                                  parent=dialog):
                                dialog._cleanup_scrolling()
                                dialog.destroy()
                        else:
                            dialog._cleanup_scrolling()
                            dialog.destroy()
                    else:
                        dialog._cleanup_scrolling()
                        dialog.destroy()
                
                # Use ttkbootstrap buttons with better styling
                tb.Button(button_inner, text="Cancel", 
                         command=cancel_settings,
                         bootstyle="secondary", width=15).pack(side='left', padx=5)
                
                tb.Button(button_inner, text="Reset Defaults", 
                         command=reset_to_defaults,
                         bootstyle="warning", width=15).pack(side='left', padx=5)
                
                tb.Button(button_inner, text="Start Scan", 
                         command=save_custom_settings,
                         bootstyle="success", width=15).pack(side='left', padx=5)
                
                # Use WindowManager's auto-resize
                self.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.9, max_height_ratio=0.72)
                
                # Handle window close properly - treat as cancel
                dialog.protocol("WM_DELETE_WINDOW", cancel_settings)
                
                # Wait for dialog to close
                dialog.wait_window()
                
                # If user cancelled at this dialog, cancel the whole scan
                if not settings_saved:
                    self.append_log("⚠️ QA scan canceled - no custom settings were saved.")
                    return
            # Check if word count cross-reference is enabled but no EPUB is selected
            check_word_count = qa_settings.get('check_word_count_ratio', False)
            epub_files_to_scan = []
            primary_epub_path = None
            
            # ALWAYS populate epub_files_to_scan for auto-search, regardless of word count checking
            # First check if current selection actually contains EPUBs
            current_epub_files = []
            if hasattr(self, 'selected_files') and self.selected_files:
                current_epub_files = [f for f in self.selected_files if f.lower().endswith('.epub')]
                print(f"[DEBUG] Current selection contains {len(current_epub_files)} EPUB files")
            
            if current_epub_files:
                # Use EPUBs from current selection
                epub_files_to_scan = current_epub_files
                print(f"[DEBUG] Using {len(epub_files_to_scan)} EPUB files from current selection")
            else:
                # No EPUBs in current selection - check if we have stored EPUBs
                primary_epub_path = self.get_current_epub_path()
                print(f"[DEBUG] get_current_epub_path returned: {primary_epub_path}")
                
                if primary_epub_path:
                    epub_files_to_scan = [primary_epub_path]
                    print(f"[DEBUG] Using stored EPUB file for auto-search")
            
            # Now handle word count specific logic if enabled
            if check_word_count:
                print("[DEBUG] Word count check is enabled, validating EPUB availability...")
                
                # Check if we have EPUBs for word count analysis
                if not epub_files_to_scan:
                    # No EPUBs available for word count analysis
                    result = messagebox.askyesnocancel(
                        "No Source EPUB Selected",
                        "Word count cross-reference is enabled but no source EPUB file is selected.\n\n" +
                        "Would you like to:\n" +
                        "• YES - Continue scan without word count analysis\n" +
                        "• NO - Select an EPUB file now\n" +
                        "• CANCEL - Cancel the scan",
                        icon='warning'
                    )
                    
                    if result is None:  # Cancel
                        self.append_log("⚠️ QA scan canceled.")
                        return
                    elif result is False:  # No - Select EPUB now
                        epub_path = filedialog.askopenfilename(
                            title="Select Source EPUB File",
                            filetypes=[("EPUB files", "*.epub"), ("All files", "*.*")]
                        )
                        
                        if not epub_path:
                            retry = messagebox.askyesno(
                                "No File Selected",
                                "No EPUB file was selected.\n\n" +
                                "Do you want to continue the scan without word count analysis?",
                                icon='question'
                            )
                            
                            if not retry:
                                self.append_log("⚠️ QA scan canceled.")
                                return
                            else:
                                qa_settings = qa_settings.copy()
                                qa_settings['check_word_count_ratio'] = False
                                self.append_log("ℹ️ Proceeding without word count analysis.")
                                epub_files_to_scan = []
                        else:
                            self.selected_epub_path = epub_path
                            self.config['last_epub_path'] = epub_path
                            self.save_config(show_message=False)
                            self.append_log(f"✅ Selected EPUB: {os.path.basename(epub_path)}")
                            epub_files_to_scan = [epub_path]
                    else:  # Yes - Continue without word count
                        qa_settings = qa_settings.copy()
                        qa_settings['check_word_count_ratio'] = False
                        self.append_log("ℹ️ Proceeding without word count analysis.")
                        epub_files_to_scan = []
            # Persist latest auto-search preference
            try:
                self.config['qa_auto_search_output'] = bool(self.qa_auto_search_output_var.get())
                self.save_config(show_message=False)
            except Exception:
                pass
            
            # Try to auto-detect output folders based on EPUB files
            folders_to_scan = []
            auto_search_enabled = self.config.get('qa_auto_search_output', True)
            try:
                if hasattr(self, 'qa_auto_search_output_var'):
                    auto_search_enabled = bool(self.qa_auto_search_output_var.get())
            except Exception:
                pass
            
            # Debug output for scanning phase
            if non_interactive:
                self.append_log(f"📝 Debug: auto_search_enabled = {auto_search_enabled}")
                self.append_log(f"📝 Debug: epub_files_to_scan = {len(epub_files_to_scan)} files")
                self.append_log(f"📝 Debug: Will run folder detection = {auto_search_enabled and epub_files_to_scan}")
            
            if auto_search_enabled and epub_files_to_scan:
                # Process each EPUB file to find its corresponding output folder
                self.append_log(f"🔍 DEBUG: Auto-search running with {len(epub_files_to_scan)} EPUB files")
                for epub_path in epub_files_to_scan:
                    self.append_log(f"🔍 DEBUG: Processing EPUB: {epub_path}")
                    try:
                        epub_base = os.path.splitext(os.path.basename(epub_path))[0]
                        current_dir = os.getcwd()
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        
                        self.append_log(f"🔍 DEBUG: EPUB base name: '{epub_base}'")
                        self.append_log(f"🔍 DEBUG: Current dir: {current_dir}")
                        self.append_log(f"🔍 DEBUG: Script dir: {script_dir}")
                        
                        # Check the most common locations in order of priority
                        candidates = [
                            os.path.join(current_dir, epub_base),        # current working directory
                            os.path.join(script_dir, epub_base),         # src directory (where output typically goes)
                            os.path.join(current_dir, 'src', epub_base), # src subdirectory from current dir
                        ]
                        
                        folder_found = None
                        for i, candidate in enumerate(candidates):
                            exists = os.path.isdir(candidate)
                            self.append_log(f"  [{epub_base}] Checking candidate {i+1}: {candidate} - {'EXISTS' if exists else 'NOT FOUND'}")
                            
                            if exists:
                                # Verify the folder actually contains HTML/XHTML files
                                try:
                                    files = os.listdir(candidate)
                                    html_files = [f for f in files if f.lower().endswith(('.html', '.xhtml', '.htm'))]
                                    if html_files:
                                        folder_found = candidate
                                        self.append_log(f"📁 Auto-selected output folder for {epub_base}: {folder_found}")
                                        self.append_log(f"   Found {len(html_files)} HTML/XHTML files to scan")
                                        break
                                    else:
                                        self.append_log(f"  [{epub_base}] Folder exists but contains no HTML/XHTML files: {candidate}")
                                except Exception as e:
                                    self.append_log(f"  [{epub_base}] Error checking files in {candidate}: {e}")
                        
                        if folder_found:
                            folders_to_scan.append(folder_found)
                            self.append_log(f"🔍 DEBUG: Added to folders_to_scan: {folder_found}")
                        else:
                            self.append_log(f"  ⚠️ No output folder found for {epub_base}")
                                
                    except Exception as e:
                        self.append_log(f"  ❌ Error processing {epub_base}: {e}")
                
                self.append_log(f"🔍 DEBUG: Final folders_to_scan: {folders_to_scan}")
            
            # Fallback behavior - if no folders found through auto-detection
            if not folders_to_scan:
                if auto_search_enabled:
                    # Auto-search failed, offer manual selection as fallback
                    self.append_log("⚠️ Auto-search enabled but no matching output folder found")
                    self.append_log("📁 Falling back to manual folder selection...")
                    
                    selected_folder = filedialog.askdirectory(title="Auto-search failed - Select Output Folder to Scan")
                    if not selected_folder:
                        self.append_log("⚠️ QA scan canceled - no folder selected.")
                        return
                    
                    # Verify the selected folder contains scannable files
                    try:
                        files = os.listdir(selected_folder)
                        html_files = [f for f in files if f.lower().endswith(('.html', '.xhtml', '.htm'))]
                        if html_files:
                            folders_to_scan.append(selected_folder)
                            self.append_log(f"✓ Manual selection: {os.path.basename(selected_folder)} ({len(html_files)} HTML/XHTML files)")
                        else:
                            self.append_log(f"❌ Selected folder contains no HTML/XHTML files: {selected_folder}")
                            return
                    except Exception as e:
                        self.append_log(f"❌ Error checking selected folder: {e}")
                        return
                if non_interactive:
                    # Add debug info for scanning phase
                    if epub_files_to_scan:
                        self.append_log(f"⚠️ Scanning phase: No matching output folders found for {len(epub_files_to_scan)} EPUB file(s)")
                        for epub_path in epub_files_to_scan:
                            epub_base = os.path.splitext(os.path.basename(epub_path))[0]
                            current_dir = os.getcwd()
                            expected_folder = os.path.join(current_dir, epub_base)
                            self.append_log(f"  [{epub_base}] Expected: {expected_folder}")
                            self.append_log(f"  [{epub_base}] Exists: {os.path.isdir(expected_folder)}")
                        
                        # List actual folders in current directory for debugging
                        try:
                            current_dir = os.getcwd()
                            actual_folders = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d)) and not d.startswith('.')]
                            if actual_folders:
                                self.append_log(f"  Available folders: {', '.join(actual_folders[:10])}{'...' if len(actual_folders) > 10 else ''}")
                        except Exception:
                            pass
                    else:
                        self.append_log("⚠️ Scanning phase: No EPUB files available for folder detection")
                    
                    self.append_log("⚠️ Skipping scan")
                    return
                
                # Clean single folder selection - no messageboxes, no harassment
                self.append_log("📁 Select folder to scan...")
                
                folders_to_scan = []
                
                # Simply select one folder - clean and simple
                selected_folder = filedialog.askdirectory(title="Select Folder with HTML Files")
                if not selected_folder:
                    self.append_log("⚠️ QA scan canceled - no folder selected.")
                    return
                
                folders_to_scan.append(selected_folder)
                self.append_log(f"  ✓ Selected folder: {os.path.basename(selected_folder)}")
                self.append_log(f"📁 Single folder scan mode - scanning: {os.path.basename(folders_to_scan[0])}")

            mode = selected_mode_value
            
            # Initialize epub_path for use in run_scan() function
            # This ensures epub_path is always defined even when manually selecting folders
            epub_path = None
            if epub_files_to_scan:
                epub_path = epub_files_to_scan[0]  # Use first EPUB if multiple
                self.append_log(f"📚 Using EPUB from scan list: {os.path.basename(epub_path)}")
            elif hasattr(self, 'selected_epub_path') and self.selected_epub_path:
                epub_path = self.selected_epub_path
                self.append_log(f"📚 Using stored EPUB: {os.path.basename(epub_path)}")
            elif primary_epub_path:
                epub_path = primary_epub_path
                self.append_log(f"📚 Using primary EPUB: {os.path.basename(epub_path)}")
            else:
                self.append_log("ℹ️ No EPUB file configured (word count analysis will be disabled if needed)")
            
            # Initialize global selected_files that applies to single-folder scans
            global_selected_files = None
            if len(folders_to_scan) == 1 and preselected_files:
                global_selected_files = list(preselected_files)
            elif len(folders_to_scan) == 1 and (not non_interactive) and (not auto_search_enabled):
                # Scan all files in the folder - no messageboxes asking about specific files
                # User can set up file preselection if they need specific files
                pass
            
            # Log bulk scan start
            if len(folders_to_scan) == 1:
                self.append_log(f"🔍 Starting QA scan in {mode.upper()} mode for folder: {folders_to_scan[0]}")
            else:
                self.append_log(f"🔍 Starting bulk QA scan in {mode.upper()} mode for {len(folders_to_scan)} folders")
            
            self.stop_requested = False
 
            # Extract cache configuration from qa_settings
            cache_config = {
                'enabled': qa_settings.get('cache_enabled', True),
                'auto_size': qa_settings.get('cache_auto_size', False),
                'show_stats': qa_settings.get('cache_show_stats', False),
                'sizes': {}
            }
            
            # Get individual cache sizes
            for cache_name in ['normalize_text', 'similarity_ratio', 'content_hashes', 
                              'semantic_fingerprint', 'structural_signature', 'translation_artifacts']:
                size = qa_settings.get(f'cache_{cache_name}', None)
                if size is not None:
                    # Convert -1 to None for unlimited
                    cache_config['sizes'][cache_name] = None if size == -1 else size
            
            # Create custom settings that includes cache config
            custom_settings = {
                'qa_settings': qa_settings,
                'cache_config': cache_config,
                'log_cache_stats': qa_settings.get('cache_show_stats', False)
            }
     
            def run_scan():
                # Update UI on the main thread
                self.master.after(0, self.update_run_button)
                self.master.after(0, lambda: self.qa_button.config(text="Stop Scan", command=self.stop_qa_scan, bootstyle="danger"))
                
                try:
                    # Extract cache configuration from qa_settings
                    cache_config = {
                        'enabled': qa_settings.get('cache_enabled', True),
                        'auto_size': qa_settings.get('cache_auto_size', False),
                        'show_stats': qa_settings.get('cache_show_stats', False),
                        'sizes': {}
                    }
                    
                    # Get individual cache sizes
                    for cache_name in ['normalize_text', 'similarity_ratio', 'content_hashes', 
                                      'semantic_fingerprint', 'structural_signature', 'translation_artifacts']:
                        size = qa_settings.get(f'cache_{cache_name}', None)
                        if size is not None:
                            # Convert -1 to None for unlimited
                            cache_config['sizes'][cache_name] = None if size == -1 else size
                    
                    # Configure the cache BEFORE calling scan_html_folder
                    from scan_html_folder import configure_qa_cache
                    configure_qa_cache(cache_config)
                    
                    # Loop through all selected folders for bulk scanning
                    successful_scans = 0
                    failed_scans = 0
                    
                    for i, current_folder in enumerate(folders_to_scan):
                        if self.stop_requested:
                            self.append_log(f"⚠️ Bulk scan stopped by user at folder {i+1}/{len(folders_to_scan)}")
                            break
                        
                        folder_name = os.path.basename(current_folder)
                        if len(folders_to_scan) > 1:
                            self.append_log(f"\n📁 [{i+1}/{len(folders_to_scan)}] Scanning folder: {folder_name}")
                        
                        # Determine the correct EPUB path for this specific folder
                        current_epub_path = epub_path
                        current_qa_settings = qa_settings.copy()
                        
                        # For bulk scanning, try to find a matching EPUB for each folder
                        if len(folders_to_scan) > 1 and current_qa_settings.get('check_word_count_ratio', False):
                            # Try to find EPUB file matching this specific folder
                            folder_basename = os.path.basename(current_folder.rstrip('/\\'))
                            self.append_log(f"  🔍 Searching for EPUB matching folder: {folder_basename}")
                            
                            # Look for EPUB in various locations
                            folder_parent = os.path.dirname(current_folder)
                            
                            # Simple exact matching first, with minimal suffix handling
                            base_name = folder_basename
                            
                            # Only handle the most common output suffixes
                            common_suffixes = ['_output', '_translated', '_en']
                            for suffix in common_suffixes:
                                if base_name.endswith(suffix):
                                    base_name = base_name[:-len(suffix)]
                                    break
                            
                            # Simple EPUB search - focus on exact matching
                            search_names = [folder_basename]  # Start with exact folder name
                            if base_name != folder_basename:  # Add base name only if different
                                search_names.append(base_name)
                            
                            potential_epub_paths = [
                                # Most common locations in order of priority
                                os.path.join(folder_parent, f"{folder_basename}.epub"),  # Same directory as output folder
                                os.path.join(folder_parent, f"{base_name}.epub"),        # Same directory with base name
                                os.path.join(current_folder, f"{folder_basename}.epub"), # Inside the output folder
                                os.path.join(current_folder, f"{base_name}.epub"),       # Inside with base name
                            ]
                            
                            # Find the first existing EPUB
                            folder_epub_path = None
                            for potential_path in potential_epub_paths:
                                if os.path.isfile(potential_path):
                                    folder_epub_path = potential_path
                                    if len(folders_to_scan) > 1:
                                        self.append_log(f"      Found matching EPUB: {os.path.basename(potential_path)}")
                                    break
                            
                            if folder_epub_path:
                                current_epub_path = folder_epub_path
                                if len(folders_to_scan) > 1:  # Only log for bulk scans
                                    self.append_log(f"  📖 Using EPUB: {os.path.basename(current_epub_path)}")
                            else:
                                # NO FALLBACK TO GLOBAL EPUB FOR BULK SCANS - This prevents wrong EPUB usage!
                                if len(folders_to_scan) > 1:
                                    self.append_log(f"  ⚠️ No matching EPUB found for folder '{folder_name}' - disabling word count analysis")
                                    expected_names = ', '.join([f"{name}.epub" for name in search_names])
                                    self.append_log(f"      Expected EPUB names: {expected_names}")
                                    current_epub_path = None
                                elif current_epub_path:  # Single folder scan can use global EPUB
                                    self.append_log(f"  📖 Using global EPUB: {os.path.basename(current_epub_path)} (no folder-specific EPUB found)")
                                else:
                                    current_epub_path = None
                                
                                # Disable word count analysis when no matching EPUB is found
                                if not current_epub_path:
                                    current_qa_settings = current_qa_settings.copy()
                                    current_qa_settings['check_word_count_ratio'] = False
                        
                        # Check for EPUB/folder name mismatch
                        if current_epub_path and current_qa_settings.get('check_word_count_ratio', False) and current_qa_settings.get('warn_name_mismatch', True):
                            epub_name = os.path.splitext(os.path.basename(current_epub_path))[0]
                            folder_name_for_check = os.path.basename(current_folder.rstrip('/\\'))
                            
                            if not check_epub_folder_match(epub_name, folder_name_for_check, current_qa_settings.get('custom_output_suffixes', '')):
                                if len(folders_to_scan) == 1:
                                    # Interactive dialog for single folder scans
                                    result = messagebox.askyesnocancel(
                                        "EPUB/Folder Name Mismatch",
                                        f"The source EPUB and output folder names don't match:\n\n" +
                                        f"📖 EPUB: {epub_name}\n" +
                                        f"📁 Folder: {folder_name_for_check}\n\n" +
                                        "This might mean you're comparing the wrong files.\n" +
                                        "Would you like to:\n" +
                                        "• YES - Continue anyway (I'm sure these match)\n" +
                                        "• NO - Select a different EPUB file\n" +
                                        "• CANCEL - Cancel the scan",
                                        icon='warning'
                                    )
                                    
                                    if result is None:  # Cancel
                                        self.append_log("⚠️ QA scan canceled due to EPUB/folder mismatch.")
                                        return
                                    elif result is False:  # No - select different EPUB
                                        new_epub_path = filedialog.askopenfilename(
                                            title="Select Different Source EPUB File",
                                            filetypes=[("EPUB files", "*.epub"), ("All files", "*.*")]
                                        )
                                        
                                        if new_epub_path:
                                            current_epub_path = new_epub_path
                                            self.selected_epub_path = new_epub_path
                                            self.config['last_epub_path'] = new_epub_path
                                            self.save_config(show_message=False)
                                            self.append_log(f"✅ Updated EPUB: {os.path.basename(new_epub_path)}")
                                        else:
                                            proceed = messagebox.askyesno(
                                                "No File Selected",
                                                "No EPUB file was selected.\n\n" +
                                                "Continue scan without word count analysis?",
                                                icon='question'
                                            )
                                            if not proceed:
                                                self.append_log("⚠️ QA scan canceled.")
                                                return
                                            else:
                                                current_qa_settings = current_qa_settings.copy()
                                                current_qa_settings['check_word_count_ratio'] = False
                                                current_epub_path = None
                                                self.append_log("ℹ️ Proceeding without word count analysis.")
                                    # If YES, just continue with warning
                                else:
                                    # For bulk scans, just warn and continue
                                    self.append_log(f"  ⚠️ Warning: EPUB/folder name mismatch - {epub_name} vs {folder_name_for_check}")
                        
                        try:
                            # Determine selected_files for this folder
                            current_selected_files = None
                            if global_selected_files and len(folders_to_scan) == 1:
                                current_selected_files = global_selected_files
                            
                            # Pass the QA settings to scan_html_folder
                            scan_html_folder(
                                current_folder, 
                                log=self.append_log, 
                                stop_flag=lambda: self.stop_requested, 
                                mode=mode,
                                qa_settings=current_qa_settings,
                                epub_path=current_epub_path,
                                selected_files=current_selected_files
                            )
                            
                            successful_scans += 1
                            if len(folders_to_scan) > 1:
                                self.append_log(f"✅ Folder '{folder_name}' scan completed successfully")
                        
                        except Exception as folder_error:
                            failed_scans += 1
                            self.append_log(f"❌ Folder '{folder_name}' scan failed: {folder_error}")
                            if len(folders_to_scan) == 1:
                                # Re-raise for single folder scans
                                raise
                    
                    # Final summary for bulk scans
                    if len(folders_to_scan) > 1:
                        self.append_log(f"\n📋 Bulk scan summary: {successful_scans} successful, {failed_scans} failed")
                    
                    # If show_stats is enabled, log cache statistics
                    if qa_settings.get('cache_show_stats', False):
                        from scan_html_folder import get_cache_info
                        cache_stats = get_cache_info()
                        self.append_log("\n📊 Cache Performance Statistics:")
                        for name, info in cache_stats.items():
                            if info:  # Check if info exists
                                hit_rate = info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0
                                self.append_log(f"  {name}: {info.hits} hits, {info.misses} misses ({hit_rate:.1%} hit rate)")
                    
                    if len(folders_to_scan) == 1:
                        self.append_log("✅ QA scan completed successfully.")
                    else:
                        self.append_log("✅ Bulk QA scan completed.")
        
                except Exception as e:
                    self.append_log(f"❌ QA scan error: {e}")
                    self.append_log(f"Traceback: {traceback.format_exc()}")
                finally:
                    # Clear thread/future refs so buttons re-enable
                    self.qa_thread = None
                    if hasattr(self, 'qa_future'):
                        try:
                            self.qa_future = None
                        except Exception:
                            pass
                    self.master.after(0, self.update_run_button)
                    self.master.after(0, lambda: self.qa_button.config(
                        text="QA Scan", 
                        command=self.run_qa_scan, 
                        bootstyle="warning",
                        state=tk.NORMAL if scan_html_folder else tk.DISABLED
                    ))
            
            # Run via shared executor
            self._ensure_executor()
            if self.executor:
                self.qa_future = self.executor.submit(run_scan)
                # Ensure UI is refreshed when QA work completes
                def _qa_done_callback(f):
                    try:
                        self.master.after(0, lambda: (setattr(self, 'qa_future', None), self.update_run_button()))
                    except Exception:
                        pass
                try:
                    self.qa_future.add_done_callback(_qa_done_callback)
                except Exception:
                    pass
            else:
                self.qa_thread = threading.Thread(target=run_scan, daemon=True)
                self.qa_thread.start()

    def show_qa_scanner_settings(self, parent_dialog, qa_settings):
            """Show QA Scanner settings dialog using WindowManager properly"""
            # Use setup_scrollable from WindowManager - NOT create_scrollable_dialog
            dialog, scrollable_frame, canvas = self.wm.setup_scrollable(
                parent_dialog,
                "QA Scanner Settings",
                width=800,
                height=None,  # Let WindowManager calculate optimal height
                modal=True,
                resizable=True,
                max_width_ratio=0.9,
                max_height_ratio=0.9
            )
            
            # Main settings frame
            main_frame = tk.Frame(scrollable_frame, padx=30, pady=20)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Title
            title_label = tk.Label(
                main_frame,
                text="QA Scanner Settings",
                font=('Arial', 24, 'bold')
            )
            title_label.pack(pady=(0, 20))
            
            # Foreign Character Settings Section
            foreign_section = tk.LabelFrame(
                main_frame,
                text="Foreign Character Detection",
                font=('Arial', 12, 'bold'),
                padx=20,
                pady=15
            )
            foreign_section.pack(fill=tk.X, pady=(0, 20))
            
            # Threshold setting
            threshold_frame = tk.Frame(foreign_section)
            threshold_frame.pack(fill=tk.X, pady=(0, 10))
            
            tk.Label(
                threshold_frame,
                text="Minimum foreign characters to flag:",
                font=('Arial', 10)
            ).pack(side=tk.LEFT)
            
            threshold_var = tk.IntVar(value=qa_settings.get('foreign_char_threshold', 10))
            threshold_spinbox = tb.Spinbox(
                threshold_frame,
                from_=0,
                to=1000,
                textvariable=threshold_var,
                width=10,
                bootstyle="primary"
            )
            threshold_spinbox.pack(side=tk.LEFT, padx=(10, 0))
            
            # Disable mousewheel scrolling on spinbox
            UIHelper.disable_spinbox_mousewheel(threshold_spinbox)
            
            tk.Label(
                threshold_frame,
                text="(0 = always flag, higher = more tolerant)",
                font=('Arial', 9),
                fg='gray'
            ).pack(side=tk.LEFT, padx=(10, 0))
            
            # Excluded characters - using UIHelper for scrollable text
            excluded_frame = tk.Frame(foreign_section)
            excluded_frame.pack(fill=tk.X, pady=(10, 0))
            
            tk.Label(
                excluded_frame,
                text="Additional characters to exclude from detection:",
                font=('Arial', 10)
            ).pack(anchor=tk.W)
            
            # Use regular Text widget with manual scroll setup instead of ScrolledText
            excluded_text_frame = tk.Frame(excluded_frame)
            excluded_text_frame.pack(fill=tk.X, pady=(5, 0))
            
            excluded_text = tk.Text(
                excluded_text_frame,
                height=7,
                width=60,
                font=('Consolas', 10),
                wrap=tk.WORD,
                undo=True
            )
            excluded_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Add scrollbar manually
            excluded_scrollbar = ttk.Scrollbar(excluded_text_frame, orient="vertical", command=excluded_text.yview)
            excluded_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            excluded_text.configure(yscrollcommand=excluded_scrollbar.set)
            
            # Setup undo/redo for the text widget
            UIHelper.setup_text_undo_redo(excluded_text)
            
            excluded_text.insert(1.0, qa_settings.get('excluded_characters', ''))
            
            tk.Label(
                excluded_frame,
                text="Enter characters separated by spaces (e.g., ™ © ® • …)",
                font=('Arial', 9),
                fg='gray'
            ).pack(anchor=tk.W)
            
            # Detection Options Section
            detection_section = tk.LabelFrame(
                main_frame,
                text="Detection Options",
                font=('Arial', 12, 'bold'),
                padx=20,
                pady=15
            )
            detection_section.pack(fill=tk.X, pady=(0, 20))
            
            # Checkboxes for detection options
            check_encoding_var = tk.BooleanVar(value=qa_settings.get('check_encoding_issues', False))
            check_repetition_var = tk.BooleanVar(value=qa_settings.get('check_repetition', True))
            check_artifacts_var = tk.BooleanVar(value=qa_settings.get('check_translation_artifacts', False))
            check_glossary_var = tk.BooleanVar(value=qa_settings.get('check_glossary_leakage', True))
            
            tb.Checkbutton(
                detection_section,
                text="Check for encoding issues (�, □, ◇)",
                variable=check_encoding_var,
                bootstyle="primary"
            ).pack(anchor=tk.W, pady=2)
            
            tb.Checkbutton(
                detection_section,
                text="Check for excessive repetition",
                variable=check_repetition_var,
                bootstyle="primary"
            ).pack(anchor=tk.W, pady=2)
            
            tb.Checkbutton(
                detection_section,
                text="Check for translation artifacts (MTL notes, watermarks)",
                variable=check_artifacts_var,
                bootstyle="primary"
            ).pack(anchor=tk.W, pady=2)
            tb.Checkbutton(
                detection_section,
                text="Check for glossary leakage (raw glossary entries in translation)",
                variable=check_glossary_var,
                bootstyle="primary"
            ).pack(anchor=tk.W, pady=2)
            
            # File Processing Section
            file_section = tk.LabelFrame(
                main_frame,
                text="File Processing",
                font=('Arial', 12, 'bold'),
                padx=20,
                pady=15
            )
            file_section.pack(fill=tk.X, pady=(0, 20))
            
            # Minimum file length
            min_length_frame = tk.Frame(file_section)
            min_length_frame.pack(fill=tk.X, pady=(0, 10))
            
            tk.Label(
                min_length_frame,
                text="Minimum file length (characters):",
                font=('Arial', 10)
            ).pack(side=tk.LEFT)
            
            min_length_var = tk.IntVar(value=qa_settings.get('min_file_length', 0))
            min_length_spinbox = tb.Spinbox(
                min_length_frame,
                from_=0,
                to=10000,
                textvariable=min_length_var,
                width=10,
                bootstyle="primary"
            )
            min_length_spinbox.pack(side=tk.LEFT, padx=(10, 0))
            
            # Disable mousewheel scrolling on spinbox
            UIHelper.disable_spinbox_mousewheel(min_length_spinbox)

            # Add a separator
            separator = ttk.Separator(main_frame, orient='horizontal')
            separator.pack(fill=tk.X, pady=15)
            
            # Word Count Cross-Reference Section
            wordcount_section = tk.LabelFrame(
                main_frame,
                text="Word Count Analysis",
                font=('Arial', 12, 'bold'),
                padx=20,
                pady=15
            )
            wordcount_section.pack(fill=tk.X, pady=(0, 20))
            
            check_word_count_var = tk.BooleanVar(value=qa_settings.get('check_word_count_ratio', False))
            tb.Checkbutton(
                wordcount_section,
                text="Cross-reference word counts with original EPUB",
                variable=check_word_count_var,
                bootstyle="primary"
            ).pack(anchor=tk.W, pady=(0, 5))
            
            tk.Label(
                wordcount_section,
                text="Compares word counts between original and translated files to detect missing content.\n" +
                     "Accounts for typical expansion ratios when translating from CJK to English.",
                wraplength=700,
                justify=tk.LEFT,
                fg='gray'
            ).pack(anchor=tk.W, padx=(20, 0))
     
            # Show current EPUB status and allow selection
            epub_frame = tk.Frame(wordcount_section)
            epub_frame.pack(anchor=tk.W, pady=(10, 5))

            # Get EPUBs from actual current selection (not stored config)
            current_epub_files = []
            if hasattr(self, 'selected_files') and self.selected_files:
                current_epub_files = [f for f in self.selected_files if f.lower().endswith('.epub')]
            
            if len(current_epub_files) > 1:
                # Multiple EPUBs in current selection
                primary_epub = os.path.basename(current_epub_files[0])
                status_text = f"📖 {len(current_epub_files)} EPUB files selected (Primary: {primary_epub})"
                status_color = 'green'
            elif len(current_epub_files) == 1:
                # Single EPUB in current selection
                status_text = f"📖 Current EPUB: {os.path.basename(current_epub_files[0])}"
                status_color = 'green'
            else:
                # No EPUB files in current selection
                status_text = "📖 No EPUB in current selection"
                status_color = 'orange'

            status_label = tk.Label(
                epub_frame,
                text=status_text,
                fg=status_color,
                font=('Arial', 10)
            )
            status_label.pack(side=tk.LEFT)

            def select_epub_for_qa():
                epub_path = filedialog.askopenfilename(
                    title="Select Source EPUB File",
                    filetypes=[("EPUB files", "*.epub"), ("All files", "*.*")],
                    parent=dialog
                )
                if epub_path:
                    self.selected_epub_path = epub_path
                    self.config['last_epub_path'] = epub_path
                    self.save_config(show_message=False)
                    
                    # Clear multiple EPUB tracking when manually selecting a single EPUB
                    if hasattr(self, 'selected_epub_files'):
                        self.selected_epub_files = [epub_path]
                    
                    status_label.config(
                        text=f"📖 Current EPUB: {os.path.basename(epub_path)}",
                        fg='green'
                    )
                    self.append_log(f"✅ Selected EPUB for QA: {os.path.basename(epub_path)}")

            tk.Button(
                epub_frame,
                text="Select EPUB",
                command=select_epub_for_qa,
                font=('Arial', 9)
            ).pack(side=tk.LEFT, padx=(10, 0))

            # Add option to disable mismatch warning
            warn_mismatch_var = tk.BooleanVar(value=qa_settings.get('warn_name_mismatch', True))
            tb.Checkbutton(
                wordcount_section,
                text="Warn when EPUB and folder names don't match",
                variable=warn_mismatch_var,
                bootstyle="primary"
            ).pack(anchor=tk.W, pady=(10, 5))

            # Additional Checks Section
            additional_section = tk.LabelFrame(
                main_frame,
                text="Additional Checks",
                font=('Arial', 12, 'bold'),
                padx=20,
                pady=15
            )
            additional_section.pack(fill=tk.X, pady=(20, 0))

            # Multiple headers check
            check_multiple_headers_var = tk.BooleanVar(value=qa_settings.get('check_multiple_headers', True))
            tb.Checkbutton(
                additional_section,
                text="Detect files with 2 or more headers (h1-h6 tags)",
                variable=check_multiple_headers_var,
                bootstyle="primary"
            ).pack(anchor=tk.W, pady=(5, 5))

            tk.Label(
                additional_section,
                text="Identifies files that may have been incorrectly split or merged.\n" +
                     "Useful for detecting chapters that contain multiple sections.",
                wraplength=700,
                justify=tk.LEFT,
                fg='gray'
            ).pack(anchor=tk.W, padx=(20, 0))

            # Missing HTML tag check
            html_tag_frame = tk.Frame(additional_section)
            html_tag_frame.pack(fill=tk.X, pady=(10, 5))

            check_missing_html_tag_var = tk.BooleanVar(value=qa_settings.get('check_missing_html_tag', True))
            check_missing_html_tag_check = tb.Checkbutton(
                html_tag_frame,
                text="Flag HTML files with missing <html> tag",
                variable=check_missing_html_tag_var,
                bootstyle="primary"
            )
            check_missing_html_tag_check.pack(side=tk.LEFT)

            tk.Label(
                html_tag_frame,
                text="(Checks if HTML files have proper structure)",
                font=('Arial', 9),
                foreground='gray'
            ).pack(side=tk.LEFT, padx=(10, 0))

            # Invalid nesting check (separate toggle)
            check_invalid_nesting_var = tk.BooleanVar(value=qa_settings.get('check_invalid_nesting', False))
            tb.Checkbutton(
                additional_section,
                text="Check for invalid tag nesting",
                variable=check_invalid_nesting_var,
                bootstyle="primary"
            ).pack(anchor=tk.W, pady=(5, 5))

            # NEW: Paragraph Structure Check
            paragraph_section_frame = tk.Frame(additional_section)
            paragraph_section_frame.pack(fill=tk.X, pady=(15, 5))
            
            # Separator line
            ttk.Separator(paragraph_section_frame, orient='horizontal').pack(fill=tk.X, pady=(0, 10))
            
            # Checkbox for paragraph structure check
            check_paragraph_structure_var = tk.BooleanVar(value=qa_settings.get('check_paragraph_structure', True))
            paragraph_check = tb.Checkbutton(
                paragraph_section_frame,
                text="Check for insufficient paragraph tags",
                variable=check_paragraph_structure_var,
                bootstyle="primary"
            )
            paragraph_check.pack(anchor=tk.W)
            
            # Threshold setting frame
            threshold_container = tk.Frame(paragraph_section_frame)
            threshold_container.pack(fill=tk.X, pady=(10, 5), padx=(20, 0))
            
            tk.Label(
                threshold_container,
                text="Minimum text in <p> tags:",
                font=('Arial', 10)
            ).pack(side=tk.LEFT)
            
            # Get current threshold value (default 30%)
            current_threshold = int(qa_settings.get('paragraph_threshold', 0.3) * 100)
            paragraph_threshold_var = tk.IntVar(value=current_threshold)
            
            # Spinbox for threshold
            paragraph_threshold_spinbox = tb.Spinbox(
                threshold_container,
                from_=0,
                to=100,
                textvariable=paragraph_threshold_var,
                width=8,
                bootstyle="primary"
            )
            paragraph_threshold_spinbox.pack(side=tk.LEFT, padx=(10, 5))
            
            # Disable mousewheel scrolling on the spinbox
            UIHelper.disable_spinbox_mousewheel(paragraph_threshold_spinbox)
            
            tk.Label(
                threshold_container,
                text="%",
                font=('Arial', 10)
            ).pack(side=tk.LEFT)
            
            # Threshold value label
            threshold_value_label = tk.Label(
                threshold_container,
                text=f"(currently {current_threshold}%)",
                font=('Arial', 9),
                fg='gray'
            )
            threshold_value_label.pack(side=tk.LEFT, padx=(10, 0))
            
            # Update label when spinbox changes
            def update_threshold_label(*args):
                try:
                    value = paragraph_threshold_var.get()
                    threshold_value_label.config(text=f"(currently {value}%)")
                except (tk.TclError, ValueError):
                    # Handle empty or invalid input
                    threshold_value_label.config(text="(currently --%)")
            paragraph_threshold_var.trace('w', update_threshold_label)
            
            # Description
            tk.Label(
                paragraph_section_frame,
                text="Detects HTML files where text content is not properly wrapped in paragraph tags.\n" +
                     "Files with less than the specified percentage of text in <p> tags will be flagged.\n" +
                     "Also checks for large blocks of unwrapped text directly in the body element.",
                wraplength=700,
                justify=tk.LEFT,
                fg='gray'
            ).pack(anchor=tk.W, padx=(20, 0), pady=(5, 0))
            
            # Enable/disable threshold setting based on checkbox
            def toggle_paragraph_threshold(*args):
                if check_paragraph_structure_var.get():
                    paragraph_threshold_spinbox.config(state='normal')
                else:
                    paragraph_threshold_spinbox.config(state='disabled')
            
            check_paragraph_structure_var.trace('w', toggle_paragraph_threshold)
            toggle_paragraph_threshold()  # Set initial state

            # Report Settings Section
            report_section = tk.LabelFrame(
                main_frame,
                text="Report Settings",
                font=('Arial', 12, 'bold'),
                padx=20,
                pady=15
            )
            report_section.pack(fill=tk.X, pady=(0, 20))

            # Cache Settings Section
            cache_section = tk.LabelFrame(
                main_frame,
                text="Performance Cache Settings",
                font=('Arial', 12, 'bold'),
                padx=20,
                pady=15
            )
            cache_section.pack(fill=tk.X, pady=(0, 20))
            
            # Enable cache checkbox
            cache_enabled_var = tk.BooleanVar(value=qa_settings.get('cache_enabled', True))
            cache_checkbox = tb.Checkbutton(
                cache_section,
                text="Enable performance cache (speeds up duplicate detection)",
                variable=cache_enabled_var,
                bootstyle="primary"
            )
            cache_checkbox.pack(anchor=tk.W, pady=(0, 10))
            
            # Cache size settings frame
            cache_sizes_frame = tk.Frame(cache_section)
            cache_sizes_frame.pack(fill=tk.X, padx=(20, 0))
            
            # Description
            tk.Label(
                cache_sizes_frame,
                text="Cache sizes (0 = disabled, -1 = unlimited):",
                font=('Arial', 10)
            ).pack(anchor=tk.W, pady=(0, 5))
            
            # Cache size variables
            cache_vars = {}
            cache_defaults = {
                'normalize_text': 10000,
                'similarity_ratio': 20000,
                'content_hashes': 5000,
                'semantic_fingerprint': 2000,
                'structural_signature': 2000,
                'translation_artifacts': 1000
            }
            
            # Create input fields for each cache type
            for cache_name, default_value in cache_defaults.items():
                row_frame = tk.Frame(cache_sizes_frame)
                row_frame.pack(fill=tk.X, pady=2)
                
                # Label
                label_text = cache_name.replace('_', ' ').title() + ":"
                tk.Label(
                    row_frame,
                    text=label_text,
                    width=25,
                    anchor='w',
                    font=('Arial', 9)
                ).pack(side=tk.LEFT)
                
                # Get current value
                current_value = qa_settings.get(f'cache_{cache_name}', default_value)
                cache_var = tk.IntVar(value=current_value)
                cache_vars[cache_name] = cache_var
                
                # Spinbox
                spinbox = tb.Spinbox(
                    row_frame,
                    from_=-1,
                    to=50000,
                    textvariable=cache_var,
                    width=10,
                    bootstyle="primary"
                )
                spinbox.pack(side=tk.LEFT, padx=(0, 10))
                
                # Disable mousewheel scrolling
                UIHelper.disable_spinbox_mousewheel(spinbox)
                
                # Quick preset buttons
                button_frame = tk.Frame(row_frame)
                button_frame.pack(side=tk.LEFT)
                
                tk.Button(
                    button_frame,
                    text="Off",
                    width=4,
                    font=('Arial', 8),
                    command=lambda v=cache_var: v.set(0)
                ).pack(side=tk.LEFT, padx=1)
                
                tk.Button(
                    button_frame,
                    text="Small",
                    width=5,
                    font=('Arial', 8),
                    command=lambda v=cache_var: v.set(1000)
                ).pack(side=tk.LEFT, padx=1)
                
                tk.Button(
                    button_frame,
                    text="Medium",
                    width=7,
                    font=('Arial', 8),
                    command=lambda v=cache_var, d=default_value: v.set(d)
                ).pack(side=tk.LEFT, padx=1)
                
                tk.Button(
                    button_frame,
                    text="Large",
                    width=5,
                    font=('Arial', 8),
                    command=lambda v=cache_var, d=default_value: v.set(d * 2)
                ).pack(side=tk.LEFT, padx=1)
                
                tk.Button(
                    button_frame,
                    text="Max",
                    width=4,
                    font=('Arial', 8),
                    command=lambda v=cache_var: v.set(-1)
                ).pack(side=tk.LEFT, padx=1)
            
            # Enable/disable cache size controls based on checkbox
            def toggle_cache_controls(*args):
                state = 'normal' if cache_enabled_var.get() else 'disabled'
                for widget in cache_sizes_frame.winfo_children():
                    if isinstance(widget, tk.Frame):
                        for child in widget.winfo_children():
                            if isinstance(child, (tb.Spinbox, tk.Button)):
                                child.config(state=state)
            
            cache_enabled_var.trace('w', toggle_cache_controls)
            toggle_cache_controls()  # Set initial state
            
            # Auto-size cache option
            auto_size_frame = tk.Frame(cache_section)
            auto_size_frame.pack(fill=tk.X, pady=(10, 5))
            
            auto_size_var = tk.BooleanVar(value=qa_settings.get('cache_auto_size', False))
            auto_size_check = tb.Checkbutton(
                auto_size_frame,
                text="Auto-size caches based on available RAM",
                variable=auto_size_var,
                bootstyle="primary"
            )
            auto_size_check.pack(side=tk.LEFT)
            
            tk.Label(
                auto_size_frame,
                text="(overrides manual settings)",
                font=('Arial', 9),
                fg='gray'
            ).pack(side=tk.LEFT, padx=(10, 0))
            
            # Cache statistics display
            stats_frame = tk.Frame(cache_section)
            stats_frame.pack(fill=tk.X, pady=(10, 0))
            
            show_stats_var = tk.BooleanVar(value=qa_settings.get('cache_show_stats', False))
            tb.Checkbutton(
                stats_frame,
                text="Show cache hit/miss statistics after scan",
                variable=show_stats_var,
                bootstyle="primary"
            ).pack(anchor=tk.W)
            
            # Info about cache
            tk.Label(
                cache_section,
                text="Larger cache sizes use more memory but improve performance for:\n" +
                     "• Large datasets (100+ files)\n" +
                     "• AI Hunter mode (all file pairs compared)\n" +
                     "• Repeated scans of the same folder",
                wraplength=700,
                justify=tk.LEFT,
                fg='gray',
                font=('Arial', 9)
            ).pack(anchor=tk.W, padx=(20, 0), pady=(10, 0))

            # AI Hunter Performance Section
            ai_hunter_section = tk.LabelFrame(
                main_frame,
                text="AI Hunter Performance Settings",
                font=('Arial', 12, 'bold'),
                padx=20,
                pady=15
            )
            ai_hunter_section.pack(fill=tk.X, pady=(0, 20))

            # Description
            tk.Label(
                ai_hunter_section,
                text="AI Hunter mode performs exhaustive duplicate detection by comparing every file pair.\n" +
                     "Parallel processing can significantly speed up this process on multi-core systems.",
                wraplength=700,
                justify=tk.LEFT,
                fg='gray',
                font=('Arial', 9)
            ).pack(anchor=tk.W, pady=(0, 10))

            # Parallel workers setting
            workers_frame = tk.Frame(ai_hunter_section)
            workers_frame.pack(fill=tk.X, pady=(0, 10))

            tk.Label(
                workers_frame,
                text="Maximum parallel workers:",
                font=('Arial', 10)
            ).pack(side=tk.LEFT)

            # Get current value from AI Hunter config
            ai_hunter_config = self.config.get('ai_hunter_config', {})
            current_max_workers = ai_hunter_config.get('ai_hunter_max_workers', 1)

            ai_hunter_workers_var = tk.IntVar(value=current_max_workers)
            workers_spinbox = tb.Spinbox(
                workers_frame,
                from_=0,
                to=64,
                textvariable=ai_hunter_workers_var,
                width=10,
                bootstyle="primary"
            )
            workers_spinbox.pack(side=tk.LEFT, padx=(10, 0))

            # Disable mousewheel scrolling on spinbox
            UIHelper.disable_spinbox_mousewheel(workers_spinbox)

            # CPU count display
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            cpu_label = tk.Label(
                workers_frame,
                text=f"(0 = use all {cpu_count} cores)",
                font=('Arial', 9),
                fg='gray'
            )
            cpu_label.pack(side=tk.LEFT, padx=(10, 0))

            # Quick preset buttons
            preset_frame = tk.Frame(ai_hunter_section)
            preset_frame.pack(fill=tk.X)

            tk.Label(
                preset_frame,
                text="Quick presets:",
                font=('Arial', 9)
            ).pack(side=tk.LEFT, padx=(0, 10))

            tk.Button(
                preset_frame,
                text=f"All cores ({cpu_count})",
                font=('Arial', 9),
                command=lambda: ai_hunter_workers_var.set(0)
            ).pack(side=tk.LEFT, padx=2)

            tk.Button(
                preset_frame,
                text="Half cores",
                font=('Arial', 9),
                command=lambda: ai_hunter_workers_var.set(max(1, cpu_count // 2))
            ).pack(side=tk.LEFT, padx=2)

            tk.Button(
                preset_frame,
                text="4 cores",
                font=('Arial', 9),
                command=lambda: ai_hunter_workers_var.set(4)
            ).pack(side=tk.LEFT, padx=2)

            tk.Button(
                preset_frame,
                text="8 cores",
                font=('Arial', 9),
                command=lambda: ai_hunter_workers_var.set(8)
            ).pack(side=tk.LEFT, padx=2)

            tk.Button(
                preset_frame,
                text="Single thread",
                font=('Arial', 9),
                command=lambda: ai_hunter_workers_var.set(1)
            ).pack(side=tk.LEFT, padx=2)

            # Performance tips
            tips_text = "Performance Tips:\n" + \
                        f"• Your system has {cpu_count} CPU cores available\n" + \
                        "• Using all cores provides maximum speed but may slow other applications\n" + \
                        "• 4-8 cores usually provides good balance of speed and system responsiveness\n" + \
                        "• Single thread (1) disables parallel processing for debugging"

            tk.Label(
                ai_hunter_section,
                text=tips_text,
                wraplength=700,
                justify=tk.LEFT,
                fg='gray',
                font=('Arial', 9)
            ).pack(anchor=tk.W, padx=(20, 0), pady=(10, 0))

            # Report format
            format_frame = tk.Frame(report_section)
            format_frame.pack(fill=tk.X, pady=(0, 10))

            tk.Label(
                format_frame,
                text="Report format:",
                font=('Arial', 10)
            ).pack(side=tk.LEFT)

            format_var = tk.StringVar(value=qa_settings.get('report_format', 'detailed'))
            format_options = [
                ("Summary only", "summary"),
                ("Detailed (recommended)", "detailed"),
                ("Verbose (all data)", "verbose")
            ]

            for idx, (text, value) in enumerate(format_options):
                rb = tb.Radiobutton(
                    format_frame,
                    text=text,
                    variable=format_var,
                    value=value,
                    bootstyle="primary"
                )
                rb.pack(side=tk.LEFT, padx=(10 if idx == 0 else 5, 0))

            # Auto-save report
            auto_save_var = tk.BooleanVar(value=qa_settings.get('auto_save_report', True))
            tb.Checkbutton(
                report_section,
                text="Automatically save report after scan",
                variable=auto_save_var,
                bootstyle="primary"
            ).pack(anchor=tk.W)

            # Buttons
            button_frame = tk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=(20, 0))
            button_inner = tk.Frame(button_frame)
            button_inner.pack()
            
            def save_settings():
                """Save QA scanner settings"""
                try:
                    qa_settings['foreign_char_threshold'] = threshold_var.get()
                    qa_settings['excluded_characters'] = excluded_text.get(1.0, tk.END).strip()
                    qa_settings['check_encoding_issues'] = check_encoding_var.get()
                    qa_settings['check_repetition'] = check_repetition_var.get()
                    qa_settings['check_translation_artifacts'] = check_artifacts_var.get()
                    qa_settings['check_glossary_leakage'] = check_glossary_var.get()
                    qa_settings['min_file_length'] = min_length_var.get()
                    qa_settings['report_format'] = format_var.get()
                    qa_settings['auto_save_report'] = auto_save_var.get()
                    qa_settings['check_word_count_ratio'] = check_word_count_var.get()
                    qa_settings['check_multiple_headers'] = check_multiple_headers_var.get()
                    qa_settings['warn_name_mismatch'] = warn_mismatch_var.get()
                    qa_settings['check_missing_html_tag'] = check_missing_html_tag_var.get()
                    qa_settings['check_paragraph_structure'] = check_paragraph_structure_var.get()
                    qa_settings['check_invalid_nesting'] = check_invalid_nesting_var.get()
                    
                    # Save cache settings
                    qa_settings['cache_enabled'] = cache_enabled_var.get()
                    qa_settings['cache_auto_size'] = auto_size_var.get()
                    qa_settings['cache_show_stats'] = show_stats_var.get()
                    
                    # Save individual cache sizes
                    for cache_name, cache_var in cache_vars.items():
                        qa_settings[f'cache_{cache_name}'] = cache_var.get()

                    if 'ai_hunter_config' not in self.config:
                        self.config['ai_hunter_config'] = {}
                    self.config['ai_hunter_config']['ai_hunter_max_workers'] = ai_hunter_workers_var.get()
        
                    # Validate and save paragraph threshold
                    try:
                        threshold_value = paragraph_threshold_var.get()
                        if 0 <= threshold_value <= 100:
                            qa_settings['paragraph_threshold'] = threshold_value / 100.0  # Convert to decimal
                        else:
                            raise ValueError("Threshold must be between 0 and 100")
                    except (tk.TclError, ValueError) as e:
                        # Default to 30% if invalid
                        qa_settings['paragraph_threshold'] = 0.3
                        self.append_log("⚠️ Invalid paragraph threshold, using default 30%")

                    
                    # Save to main config
                    self.config['qa_scanner_settings'] = qa_settings
                    
                    # Call save_config with show_message=False to avoid the error
                    self.save_config(show_message=False)
                    
                    self.append_log("✅ QA Scanner settings saved")
                    dialog._cleanup_scrolling()  # Clean up scrolling bindings
                    dialog.destroy()
                    
                except Exception as e:
                    self.append_log(f"❌ Error saving QA settings: {str(e)}")
                    messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
            
            def reset_defaults():
                """Reset to default settings"""
                result = messagebox.askyesno(
                    "Reset to Defaults", 
                    "Are you sure you want to reset all settings to defaults?",
                    parent=dialog
                )
                if result:
                    threshold_var.set(10)
                    excluded_text.delete(1.0, tk.END)
                    check_encoding_var.set(False)
                    check_repetition_var.set(True)
                    check_artifacts_var.set(False)

                    check_glossary_var.set(True)
                    min_length_var.set(0)
                    format_var.set('detailed')
                    auto_save_var.set(True)
                    check_word_count_var.set(False)
                    check_multiple_headers_var.set(True)
                    warn_mismatch_var.set(False)
                    check_missing_html_tag_var.set(True)
                    check_paragraph_structure_var.set(True)
                    check_invalid_nesting_var.set(False)
                    paragraph_threshold_var.set(30)  # 30% default
                    paragraph_threshold_var.set(30)  # 30% default
                    
                    # Reset cache settings
                    cache_enabled_var.set(True)
                    auto_size_var.set(False)
                    show_stats_var.set(False)
                    
                    # Reset cache sizes to defaults
                    for cache_name, default_value in cache_defaults.items():
                        cache_vars[cache_name].set(default_value)
                        
                    ai_hunter_workers_var.set(1)
            
            # Create buttons using ttkbootstrap styles
            save_btn = tb.Button(
                button_inner,
                text="Save Settings",
                command=save_settings,
                bootstyle="success",
                width=15
            )
            save_btn.pack(side=tk.LEFT, padx=5)
            
            reset_btn = tb.Button(
                button_inner,
                text="Reset Defaults",
                command=reset_defaults,
                bootstyle="warning",
                width=15
            )
            reset_btn.pack(side=tk.RIGHT, padx=(5, 0))
            
            cancel_btn = tb.Button(
                button_inner,
                text="Cancel",
                command=lambda: [dialog._cleanup_scrolling(), dialog.destroy()],
                bootstyle="secondary",
                width=15
            )
            cancel_btn.pack(side=tk.RIGHT)
            
            # Use WindowManager's auto_resize_dialog to properly size the window
            self.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.9, max_height_ratio=0.85)
            
            # Handle window close - setup_scrollable adds _cleanup_scrolling method
            dialog.protocol("WM_DELETE_WINDOW", lambda: [dialog._cleanup_scrolling(), dialog.destroy()])
        
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
       if qa_running:
           self.qa_button.config(text="Stop Scan", command=self.stop_qa_scan, 
                                 bootstyle="danger", state=tk.NORMAL)
       else:
           self.qa_button.config(text="QA Scan", command=self.run_qa_scan, 
                                 bootstyle="warning", state=tk.NORMAL if scan_html_folder and not any_process_running else tk.DISABLED)   

    def stop_translation(self):
        """Stop translation while preserving loaded file"""
        current_file = self.entry_epub.get() if hasattr(self, 'entry_epub') else None
        
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
       
       # Important: Reset the thread/future references so button updates properly
       if hasattr(self, 'glossary_thread'):
           self.glossary_thread = None
       if hasattr(self, 'glossary_future'):
           self.glossary_future = None
       
       self.append_log("❌ Glossary extraction stop requested.")
       self.append_log("⏳ Please wait... stopping after current API call completes.")
       self.update_run_button()


    def stop_epub_converter(self):
        """Stop EPUB converter"""
        self.stop_requested = True
        self.append_log("❌ EPUB converter stop requested.")
        self.append_log("⏳ Please wait... stopping after current operation completes.")
        
        # Important: Reset the thread reference so button updates properly
        if hasattr(self, 'epub_thread'):
            self.epub_thread = None
        
        self.update_run_button()

    def stop_qa_scan(self):
        self.stop_requested = True
        try:
            from scan_html_folder import stop_scan
            if stop_scan():
                self.append_log("✅ Stop scan signal sent successfully")
        except Exception as e:
            self.append_log(f"❌ Failed to stop scan: {e}")
        self.append_log("⛔ QA scan stop requested.")
       

    def on_close(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to exit?"):
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
            
            self.master.destroy()
            sys.exit(0)

    def append_log(self, message):
       """Append message to log with safety checks (fallback to print if GUI is gone)."""
       def _append():
           try:
               # Bail out if the widget no longer exists
               if not hasattr(self, 'log_text'):
                   print(message)
                   return
               try:
                   exists = bool(self.log_text.winfo_exists())
               except Exception:
                   exists = False
               if not exists:
                   print(message)
                   return
               
               at_bottom = False
               try:
                   at_bottom = self.log_text.yview()[1] >= 0.98
               except Exception:
                   at_bottom = False
               
               is_memory = any(keyword in message for keyword in ['[MEMORY]', '📝', 'rolling summary', 'memory'])
               
               if is_memory:
                   self.log_text.insert(tk.END, message + "\n", "memory")
                   if "memory" not in self.log_text.tag_names():
                       self.log_text.tag_config("memory", foreground="#4CAF50", font=('TkDefaultFont', 10, 'italic'))
               else:
                   self.log_text.insert(tk.END, message + "\n")
               
               if at_bottom:
                   self.log_text.see(tk.END)
           except Exception:
               # As a last resort, print to stdout to avoid crashing callbacks
               try:
                   print(message)
               except Exception:
                   pass
       
       if threading.current_thread() is threading.main_thread():
           _append()
       else:
           try:
               self.master.after(0, _append)
           except Exception:
               # If the master window is gone, just print
               try:
                   print(message)
               except Exception:
                   pass

    def update_status_line(self, message, progress_percent=None):
       """Update a status line in the log safely (fallback to print)."""
       def _update():
           try:
               if not hasattr(self, 'log_text') or not self.log_text.winfo_exists():
                   print(message)
                   return
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
           except Exception:
               try:
                   print(message)
               except Exception:
                   pass
       
       if threading.current_thread() is threading.main_thread():
           _update()
       else:
           try:
               self.master.after(0, _update)
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
        paths = filedialog.askopenfilenames(
            title="Select File(s) - Hold Ctrl/Shift to select multiple",
            filetypes=[
                ("Supported files", "*.epub;*.cbz;*.txt;*.json;*.png;*.jpg;*.jpeg;*.gif;*.bmp;*.webp"),
                ("EPUB/CBZ", "*.epub;*.cbz"),
                ("EPUB files", "*.epub"),
                ("Comic Book Zip", "*.cbz"),
                ("Text files", "*.txt;*.json"),
                ("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp;*.webp"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg;*.jpeg"),
                ("GIF files", "*.gif"),
                ("BMP files", "*.bmp"),
                ("WebP files", "*.webp"),
                ("All files", "*.*")
            ]
        )
        if paths:
            self._handle_file_selection(list(paths))

    def browse_folder(self):
        """Select an entire folder of files"""
        folder_path = filedialog.askdirectory(
            title="Select Folder Containing Files to Translate"
        )
        if folder_path:
            # Find all supported files in the folder
            supported_extensions = {'.epub', '.cbz', '.txt', '.json', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
            files = []
            
            # Recursively find files if deep scan is enabled
            if hasattr(self, 'deep_scan_var') and self.deep_scan_var.get():
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
                messagebox.showwarning("No Files Found", 
                                     f"No supported files found in:\n{folder_path}\n\nSupported formats: EPUB, TXT, PNG, JPG, JPEG, GIF, BMP, WebP")

    def clear_file_selection(self):
        """Clear all selected files"""
        self.entry_epub.delete(0, tk.END)
        self.entry_epub.insert(0, "No file selected")
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
        self.entry_epub.delete(0, tk.END)
        
        # Define image extensions
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        
        if len(processed_paths) == 1:
            # Single file - display full path
            # Check if this was a JSON conversion
            if processed_paths[0] in self.json_conversions:
                # Show original JSON filename in parentheses
                original_json = self.json_conversions[processed_paths[0]]
                display_path = f"{processed_paths[0]} (from {os.path.basename(original_json)})"
                self.entry_epub.insert(0, display_path)
            else:
                self.entry_epub.insert(0, processed_paths[0])
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
            self.entry_epub.insert(0, display_text)
            self.file_path = processed_paths[0]  # Set first file as primary
        
        # Check if these are image files
        image_files = [p for p in processed_paths if os.path.splitext(p)[1].lower() in image_extensions]
        
        if image_files:
            # Enable image translation if not already enabled
            if hasattr(self, 'enable_image_translation_var') and not self.enable_image_translation_var.get():
                self.enable_image_translation_var.set(True)
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
        show = self.api_key_entry.cget('show')
        self.api_key_entry.config(show='' if show == '*' else '*')
        # Track the visibility state
        self.api_key_visible = (show == '*')  # Will be True when showing, False when hiding
    
    def prompt_custom_token_limit(self):
       val = simpledialog.askinteger(
           "Set Max Output Token Limit",
           "Enter max output tokens for API output (e.g., 16384, 32768, 65536):",
           minvalue=1,
           maxvalue=2000000
       )
       if val:
           self.max_output_tokens = val
           self.output_btn.config(text=f"Output Token Limit: {val}")
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
        from tkinter import filedialog, messagebox
        
        path = filedialog.askopenfilename(
            title="Select glossary file",
            filetypes=[
                ("Supported files", "*.json;*.csv;*.txt"),
                ("JSON files", "*.json"),
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
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
                        response = messagebox.askyesno(
                            "JSON Auto-Fix Successful",
                            f"The JSON file had errors that were automatically fixed.\n\n"
                            f"Original error: {str(e)}\n\n"
                            f"Do you want to save the fixed version?\n"
                            f"(A backup of the original will be created)"
                        )
                        
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
                        
                        response = messagebox.askyesnocancel(
                            "JSON Fix Failed",
                            f"The JSON file has errors that couldn't be automatically fixed.\n\n"
                            f"Original error: {str(e)}\n"
                            f"After auto-fix attempt: {str(e2)}\n\n"
                            f"{error_details}\n\n"
                            f"Options:\n"
                            f"• YES: Open the file in your default editor to fix manually\n"
                            f"• NO: Try to use the file anyway (may fail)\n"
                            f"• CANCEL: Cancel loading this glossary"
                        )
                        
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
                                
                                messagebox.showinfo(
                                    "Manual Edit",
                                    "Please fix the JSON errors in your editor and save the file.\n"
                                    "Then click OK to retry loading the glossary."
                                )
                                
                                # Recursively call load_glossary to retry
                                self.load_glossary()
                                return
                                
                            except Exception as editor_error:
                                messagebox.showerror(
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
                messagebox.showerror("Error", f"Failed to read glossary file: {str(e)}")
                return
        
        else:
            messagebox.showerror(
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
        
        self.append_glossary_var.set(True)
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
            # Create backup of existing config before saving
            self._backup_config_file()
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
            
            # Validate numeric fields (skip validation if called from manga integration with show_message=False)
            if show_message:
                delay_val = self.delay_entry.get().strip()
                if delay_val and not delay_val.replace('.', '', 1).isdigit():
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.critical(None, "Invalid Input", "Please enter a valid number for API call delay")
                    return
            self.config['delay'] = safe_float(self.delay_entry.get().strip(), 2)

            if show_message:
                thread_delay_val = self.thread_delay_var.get().strip()
                if not thread_delay_val.replace('.', '', 1).isdigit():
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.critical(None, "Invalid Input", "Please enter a valid number for Threading Delay")
                    return
            self.config['thread_submission_delay'] = safe_float(self.thread_delay_var.get().strip(), 0.5)
            
            if show_message:
                trans_temp_val = self.trans_temp.get().strip()
                if trans_temp_val:
                    try: float(trans_temp_val)
                    except ValueError:
                        from PySide6.QtWidgets import QMessageBox
                        QMessageBox.critical(None, "Invalid Input", "Please enter a valid number for Temperature")
                        return
            self.config['translation_temperature'] = safe_float(self.trans_temp.get().strip(), 0.3)
            
            if show_message:
                trans_history_val = self.trans_history.get().strip()
                if trans_history_val and not trans_history_val.isdigit():
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.critical(None, "Invalid Input", "Please enter a valid number for Translation History Limit")
                    return
            self.config['translation_history_limit'] = safe_int(self.trans_history.get().strip(), 2)
            
            # Add fuzzy matching threshold
            if hasattr(self, 'fuzzy_threshold_var'):
                fuzzy_val = self.fuzzy_threshold_var.get()
                if 0.5 <= fuzzy_val <= 1.0:
                    self.config['glossary_fuzzy_threshold'] = fuzzy_val
                else:
                    self.config['glossary_fuzzy_threshold'] = 0.90  # default

            # Add glossary format preference
            if hasattr(self, 'use_legacy_csv_var'):
                self.config['glossary_use_legacy_csv'] = self.use_legacy_csv_var.get()
    
             # Add after saving translation_prompt_text:
            if hasattr(self, 'format_instructions_text'):
                try:
                    self.config['glossary_format_instructions'] = self.format_instructions_text.get('1.0', tk.END).strip()
                except:
                    pass 
 
            if hasattr(self, 'azure_api_version_var'):
                self.config['azure_api_version'] = self.azure_api_version_var.get()
    
            # Save all other settings
            self.config['api_key'] = self.api_key_entry.get()
            self.config['REMOVE_AI_ARTIFACTS'] = self.REMOVE_AI_ARTIFACTS_var.get()
            self.config['attach_css_to_chapters'] = self.attach_css_to_chapters_var.get()
            self.config['chapter_range'] = self.chapter_range_entry.get().strip()
            self.config['use_rolling_summary'] = self.rolling_summary_var.get()
            self.config['summary_role'] = self.summary_role_var.get()
            self.config['max_output_tokens'] = self.max_output_tokens
            self.config['translate_book_title'] = self.translate_book_title_var.get()
            self.config['book_title_prompt'] = self.book_title_prompt
            self.config['append_glossary'] = self.append_glossary_var.get()
            self.config['emergency_paragraph_restore'] = self.emergency_restore_var.get()
            self.config['reinforcement_frequency'] = safe_int(self.reinforcement_freq_var.get(), 10)
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
            self.config['max_images_per_chapter'] = safe_int(self.max_images_per_chapter_var.get(), 1)
            self.config['batch_translation'] = self.batch_translation_var.get()
            self.config['batch_size'] = safe_int(self.batch_size_var.get(), 3)
            self.config['conservative_batching'] = self.conservative_batching_var.get()
            self.config['translation_history_rolling'] = self.translation_history_rolling_var.get()

            # OpenRouter transport/compression toggles (ensure persisted even when dialog not open)
            if hasattr(self, 'openrouter_http_only_var'):
                self.config['openrouter_use_http_only'] = bool(self.openrouter_http_only_var.get())
                os.environ['OPENROUTER_USE_HTTP_ONLY'] = '1' if self.openrouter_http_only_var.get() else '0'
            if hasattr(self, 'openrouter_accept_identity_var'):
                self.config['openrouter_accept_identity'] = bool(self.openrouter_accept_identity_var.get())
                os.environ['OPENROUTER_ACCEPT_IDENTITY'] = '1' if self.openrouter_accept_identity_var.get() else '0'
            if hasattr(self, 'openrouter_preferred_provider_var'):
                self.config['openrouter_preferred_provider'] = self.openrouter_preferred_provider_var.get()
                os.environ['OPENROUTER_PREFERRED_PROVIDER'] = self.openrouter_preferred_provider_var.get()
            self.config['glossary_history_rolling'] = self.glossary_history_rolling_var.get()
            self.config['disable_epub_gallery'] = self.disable_epub_gallery_var.get()
            self.config['disable_automatic_cover_creation'] = self.disable_automatic_cover_creation_var.get()
            self.config['translate_cover_html'] = self.translate_cover_html_var.get()
            self.config['enable_auto_glossary'] = self.enable_auto_glossary_var.get()
            self.config['duplicate_detection_mode'] = self.duplicate_detection_mode_var.get()
            self.config['chapter_number_offset'] = safe_int(self.chapter_number_offset_var.get(), 0)
            self.config['use_header_as_output'] = self.use_header_as_output_var.get()
            self.config['enable_decimal_chapters'] = self.enable_decimal_chapters_var.get()
            self.config['enable_watermark_removal'] = self.enable_watermark_removal_var.get()
            self.config['save_cleaned_images'] = self.save_cleaned_images_var.get()
            self.config['advanced_watermark_removal'] = self.advanced_watermark_removal_var.get()
            self.config['compression_factor'] = self.compression_factor_var.get()
            self.config['translation_chunk_prompt'] = self.translation_chunk_prompt
            self.config['image_chunk_prompt'] = self.image_chunk_prompt
            self.config['force_ncx_only'] = self.force_ncx_only_var.get()
            self.config['vertex_ai_location'] = self.vertex_location_var.get()
            self.config['batch_translate_headers'] = self.batch_translate_headers_var.get()
            self.config['headers_per_batch'] = self.headers_per_batch_var.get()
            self.config['update_html_headers'] = self.update_html_headers_var.get() 
            self.config['save_header_translations'] = self.save_header_translations_var.get()
            self.config['single_api_image_chunks'] = self.single_api_image_chunks_var.get()
            self.config['enable_gemini_thinking'] = self.enable_gemini_thinking_var.get()
            self.config['thinking_budget'] = int(self.thinking_budget_var.get()) if self.thinking_budget_var.get().lstrip('-').isdigit() else 0
            self.config['enable_gpt_thinking'] = self.enable_gpt_thinking_var.get()
            self.config['gpt_reasoning_tokens'] = int(self.gpt_reasoning_tokens_var.get()) if self.gpt_reasoning_tokens_var.get().lstrip('-').isdigit() else 0
            self.config['gpt_effort'] = self.gpt_effort_var.get()
            self.config['openai_base_url'] = self.openai_base_url_var.get()
            self.config['groq_base_url'] = self.groq_base_url_var.get()  # This was missing!
            self.config['fireworks_base_url'] = self.fireworks_base_url_var.get()
            self.config['use_custom_openai_endpoint'] = self.use_custom_openai_endpoint_var.get()
            
            # Save additional important missing settings
            if hasattr(self, 'retain_source_extension_var'):
                self.config['retain_source_extension'] = self.retain_source_extension_var.get()
                # Update environment variable
                os.environ['RETAIN_SOURCE_EXTENSION'] = '1' if self.retain_source_extension_var.get() else '0'
            
            if hasattr(self, 'use_fallback_keys_var'):
                self.config['use_fallback_keys'] = self.use_fallback_keys_var.get()
            
            if hasattr(self, 'auto_update_check_var'):
                self.config['auto_update_check'] = self.auto_update_check_var.get()
                
            # Preserve last update check time if it exists
            if hasattr(self, 'update_manager') and self.update_manager:
                self.config['last_update_check_time'] = self.update_manager._last_check_time
                
            # Save window manager safe ratios setting
            if hasattr(self, 'wm') and hasattr(self.wm, '_force_safe_ratios'):
                self.config['force_safe_ratios'] = self.wm._force_safe_ratios
                
            # Save metadata-related ignore settings
            if hasattr(self, 'ignore_header_var'):
                self.config['ignore_header'] = self.ignore_header_var.get()
                
            if hasattr(self, 'ignore_title_var'):
                self.config['ignore_title'] = self.ignore_title_var.get()
            self.config['disable_chapter_merging'] = self.disable_chapter_merging_var.get()
            self.config['use_gemini_openai_endpoint'] = self.use_gemini_openai_endpoint_var.get()
            self.config['gemini_openai_endpoint'] = self.gemini_openai_endpoint_var.get()
            # Save extraction worker settings
            self.config['enable_parallel_extraction'] = self.enable_parallel_extraction_var.get()
            self.config['extraction_workers'] = self.extraction_workers_var.get()
            # Save GUI yield setting and set environment variable
            if hasattr(self, 'enable_gui_yield_var'):
                self.config['enable_gui_yield'] = self.enable_gui_yield_var.get()
                os.environ['ENABLE_GUI_YIELD'] = '1' if self.enable_gui_yield_var.get() else '0'
            self.config['glossary_max_text_size'] = self.glossary_max_text_size_var.get()
            self.config['glossary_chapter_split_threshold'] = self.glossary_chapter_split_threshold_var.get()
            self.config['glossary_filter_mode'] = self.glossary_filter_mode_var.get()
            self.config['image_chunk_overlap'] = safe_float(self.image_chunk_overlap_var.get(), 1.0)

            # Save HTTP/Network tuning settings (from Other Settings)
            if hasattr(self, 'chunk_timeout_var'):
                self.config['chunk_timeout'] = safe_int(self.chunk_timeout_var.get(), 900)
            if hasattr(self, 'enable_http_tuning_var'):
                self.config['enable_http_tuning'] = self.enable_http_tuning_var.get()
            if hasattr(self, 'connect_timeout_var'):
                self.config['connect_timeout'] = safe_float(self.connect_timeout_var.get(), 10.0)
            if hasattr(self, 'read_timeout_var'):
                self.config['read_timeout'] = safe_float(self.read_timeout_var.get(), 180.0)
            if hasattr(self, 'http_pool_connections_var'):
                self.config['http_pool_connections'] = safe_int(self.http_pool_connections_var.get(), 20)
            if hasattr(self, 'http_pool_maxsize_var'):
                self.config['http_pool_maxsize'] = safe_int(self.http_pool_maxsize_var.get(), 50)
            if hasattr(self, 'ignore_retry_after_var'):
                self.config['ignore_retry_after'] = self.ignore_retry_after_var.get()
            if hasattr(self, 'max_retries_var'):
                self.config['max_retries'] = safe_int(self.max_retries_var.get(), 7)
            if hasattr(self, 'indefinite_rate_limit_retry_var'):
                self.config['indefinite_rate_limit_retry'] = self.indefinite_rate_limit_retry_var.get()
            
            # Save retry settings (from Other Settings)
            if hasattr(self, 'retry_truncated_var'):
                self.config['retry_truncated'] = self.retry_truncated_var.get()
            if hasattr(self, 'max_retry_tokens_var'):
                self.config['max_retry_tokens'] = safe_int(self.max_retry_tokens_var.get(), 16384)
            if hasattr(self, 'retry_timeout_var'):
                self.config['retry_timeout'] = self.retry_timeout_var.get()
            if hasattr(self, 'preserve_original_text_var'):
                self.config['preserve_original_text_on_failure'] = self.preserve_original_text_var.get()
            
            # Save rolling summary settings (from Other Settings)
            if hasattr(self, 'rolling_summary_exchanges_var'):
                self.config['rolling_summary_exchanges'] = safe_int(self.rolling_summary_exchanges_var.get(), 5)
            if hasattr(self, 'rolling_summary_mode_var'):
                self.config['rolling_summary_mode'] = self.rolling_summary_mode_var.get()
            if hasattr(self, 'rolling_summary_max_entries_var'):
                self.config['rolling_summary_max_entries'] = safe_int(self.rolling_summary_max_entries_var.get(), 10)
            
            # Save QA/scanning settings (from Other Settings)
            if hasattr(self, 'qa_auto_search_output_var'):
                self.config['qa_auto_search_output'] = self.qa_auto_search_output_var.get()
            if hasattr(self, 'disable_zero_detection_var'):
                self.config['disable_zero_detection'] = self.disable_zero_detection_var.get()
            if hasattr(self, 'disable_gemini_safety_var'):
                self.config['disable_gemini_safety'] = self.disable_gemini_safety_var.get()

            # NEW: Save strip honorifics setting
            self.config['strip_honorifics'] = self.strip_honorifics_var.get()
            
            # Save glossary backup settings
            if hasattr(self, 'config') and 'glossary_auto_backup' in self.config:
                # These might be set from the glossary backup dialog
                pass  # Already in config, don't overwrite
            else:
                # Set defaults if not already set
                self.config.setdefault('glossary_auto_backup', True)
                self.config.setdefault('glossary_max_backups', 50)
                
            # Save QA Scanner settings if they exist
            if hasattr(self, 'config') and 'qa_scanner_settings' in self.config:
                # QA scanner settings already exist in config, keep them
                pass
            else:
                # Initialize default QA scanner settings if not present
                default_qa_settings = {
                    'foreign_char_threshold': 10,
                    'excluded_characters': '',
                    'check_encoding_issues': False,
                    'check_repetition': True,
'check_translation_artifacts': False,
                    'check_glossary_leakage': True,
                    'min_file_length': 0,
                    'report_format': 'detailed',
                    'auto_save_report': True,
                    'check_word_count_ratio': False,
                    'check_multiple_headers': True,
                    'warn_name_mismatch': False,
                    'check_missing_html_tag': True,
                    'check_paragraph_structure': True,
                    'check_invalid_nesting': False,
                    'paragraph_threshold': 0.3,
                    'cache_enabled': True,
                    'cache_auto_size': False,
                    'cache_show_stats': False
                }
                self.config.setdefault('qa_scanner_settings', default_qa_settings)
            
            # Save AI Hunter config settings if they exist
            if 'ai_hunter_config' not in self.config:
                self.config['ai_hunter_config'] = {}
            # Ensure ai_hunter_max_workers has a default value
            self.config['ai_hunter_config'].setdefault('ai_hunter_max_workers', 1)
            
            # NEW: Save prompts from text widgets if they exist
            if hasattr(self, 'auto_prompt_text'):
                try:
                    self.config['auto_glossary_prompt'] = self.auto_prompt_text.get('1.0', tk.END).strip()
                except:
                    pass
            
            if hasattr(self, 'append_prompt_text'):
                try:
                    self.config['append_glossary_prompt'] = self.append_prompt_text.get('1.0', tk.END).strip()
                except:
                    pass
            
            if hasattr(self, 'translation_prompt_text'):
                try:
                    self.config['glossary_translation_prompt'] = self.translation_prompt_text.get('1.0', tk.END).strip()
                except:
                    pass
                    
            # Update environment variable when saving
            if self.enable_parallel_extraction_var.get():
                os.environ["EXTRACTION_WORKERS"] = str(self.extraction_workers_var.get())
            else:
                os.environ["EXTRACTION_WORKERS"] = "1"
                
            # Chapter Extraction Settings - Save all extraction-related settings
            # These are the critical settings shown in the screenshot
            
            # Save Text Extraction Method (Standard/Enhanced)
            if hasattr(self, 'text_extraction_method_var'):
                self.config['text_extraction_method'] = self.text_extraction_method_var.get()
            
            # Save File Filtering Level (Smart/Comprehensive/Full)
            if hasattr(self, 'file_filtering_level_var'):
                self.config['file_filtering_level'] = self.file_filtering_level_var.get()
            
            # Save Preserve Markdown Structure setting
            if hasattr(self, 'enhanced_preserve_structure_var'):
                self.config['enhanced_preserve_structure'] = self.enhanced_preserve_structure_var.get()
            
            # Save Enhanced Filtering setting (for backwards compatibility)
            if hasattr(self, 'enhanced_filtering_var'):
                self.config['enhanced_filtering'] = self.enhanced_filtering_var.get()
            
            # Save force BeautifulSoup for traditional APIs
            if hasattr(self, 'force_bs_for_traditional_var'):
                self.config['force_bs_for_traditional'] = self.force_bs_for_traditional_var.get()
            
            # Update extraction_mode for backwards compatibility with older versions
            if hasattr(self, 'text_extraction_method_var') and hasattr(self, 'file_filtering_level_var'):
                if self.text_extraction_method_var.get() == 'enhanced':
                    self.config['extraction_mode'] = 'enhanced'
                    # When enhanced mode is selected, the filtering level applies to enhanced mode
                    self.config['enhanced_filtering'] = self.file_filtering_level_var.get()
                else:
                    # When standard mode is selected, use the filtering level directly
                    self.config['extraction_mode'] = self.file_filtering_level_var.get()
            elif hasattr(self, 'extraction_mode_var'):
                # Fallback for older UI
                self.config['extraction_mode'] = self.extraction_mode_var.get()

            # Save image compression settings if they exist
            # These are saved from the compression dialog, but we ensure defaults here
            if 'enable_image_compression' not in self.config:
                self.config['enable_image_compression'] = False
            if 'auto_compress_enabled' not in self.config:
                self.config['auto_compress_enabled'] = True
            if 'target_image_tokens' not in self.config:
                self.config['target_image_tokens'] = 1000
            if 'image_compression_format' not in self.config:
                self.config['image_compression_format'] = 'auto'
            if 'webp_quality' not in self.config:
                self.config['webp_quality'] = 85
            if 'jpeg_quality' not in self.config:
                self.config['jpeg_quality'] = 85
            if 'png_compression' not in self.config:
                self.config['png_compression'] = 6
            if 'max_image_dimension' not in self.config:
                self.config['max_image_dimension'] = 2048
            if 'max_image_size_mb' not in self.config:
                self.config['max_image_size_mb'] = 10
            if 'preserve_transparency' not in self.config:
                self.config['preserve_transparency'] = False  
            if 'preserve_original_format' not in self.config:
                self.config['preserve_original_format'] = False 
            if 'optimize_for_ocr' not in self.config:
                self.config['optimize_for_ocr'] = True
            if 'progressive_encoding' not in self.config:
                self.config['progressive_encoding'] = True
            if 'save_compressed_images' not in self.config:
                self.config['save_compressed_images'] = False
        
            
            # Add anti-duplicate parameters
            if hasattr(self, 'enable_anti_duplicate_var'):
                self.config['enable_anti_duplicate'] = self.enable_anti_duplicate_var.get()
                self.config['top_p'] = self.top_p_var.get()
                self.config['top_k'] = self.top_k_var.get()
                self.config['frequency_penalty'] = self.frequency_penalty_var.get()
                self.config['presence_penalty'] = self.presence_penalty_var.get()
                self.config['repetition_penalty'] = self.repetition_penalty_var.get()
                self.config['candidate_count'] = self.candidate_count_var.get()  
                self.config['custom_stop_sequences'] = self.custom_stop_sequences_var.get()
                self.config['logit_bias_enabled'] = self.logit_bias_enabled_var.get()
                self.config['logit_bias_strength'] = self.logit_bias_strength_var.get()
                self.config['bias_common_words'] = self.bias_common_words_var.get()
                self.config['bias_repetitive_phrases'] = self.bias_repetitive_phrases_var.get()
            
            # Save scanning phase settings
            if hasattr(self, 'scan_phase_enabled_var'):
                self.config['scan_phase_enabled'] = self.scan_phase_enabled_var.get()
            if hasattr(self, 'scan_phase_mode_var'):
                self.config['scan_phase_mode'] = self.scan_phase_mode_var.get()

            _tl = self.token_limit_entry.get().strip()
            if _tl.isdigit():
                self.config['token_limit'] = int(_tl)
            else:
                self.config['token_limit'] = None
            
            # Store Google Cloud credentials path BEFORE encryption
            # This should NOT be encrypted since it's just a file path
            google_creds_path = self.config.get('google_cloud_credentials')
            
            # Encrypt the config
            encrypted_config = encrypt_config(self.config)
            
            # Re-add the Google Cloud credentials path after encryption
            # This ensures the path is stored unencrypted for easy access
            if google_creds_path:
                encrypted_config['google_cloud_credentials'] = google_creds_path

            # Validate config can be serialized to JSON before writing
            try:
                json_test = json.dumps(encrypted_config, ensure_ascii=False, indent=2)
            except Exception as e:
                raise Exception(f"Config validation failed - invalid JSON: {e}")
            
            # Write to file
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(encrypted_config, f, ensure_ascii=False, indent=2)
            
            # Only show message if requested
            if show_message:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(None, "Saved", "Configuration saved.")
                
        except Exception as e:
            # Always show error messages regardless of show_message
            if show_message:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(None, "Error", f"Failed to save configuration: {e}")
            else:
                # Silent fail when called from manga integration auto-save
                print(f"Warning: Config save failed (silent): {e}")
            # Try to restore from backup if save failed
            self._restore_config_from_backup()

    def _backup_config_file(self):
        """Create backup of the existing config file before saving."""
        try:
            # Skip if config file doesn't exist yet
            if not os.path.exists(CONFIG_FILE):
                return
                
            # Get base directory that works in both development and frozen environments
            base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            
            # Resolve config file path for backup directory
            if os.path.isabs(CONFIG_FILE):
                config_dir = os.path.dirname(CONFIG_FILE)
            else:
                config_dir = os.path.dirname(os.path.abspath(CONFIG_FILE))
            
            # Create backup directory
            backup_dir = os.path.join(config_dir, "config_backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Create timestamped backup name
            backup_name = f"config_{time.strftime('%Y%m%d_%H%M%S')}.json.bak"
            backup_path = os.path.join(backup_dir, backup_name)
            
            # Copy the file
            shutil.copy2(CONFIG_FILE, backup_path)
            
            # Maintain only the last 10 backups
            backups = [os.path.join(backup_dir, f) for f in os.listdir(backup_dir) 
                       if f.startswith("config_") and f.endswith(".json.bak")]
            backups.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Remove oldest backups if more than 10
            for old_backup in backups[10:]:
                try:
                    os.remove(old_backup)
                except Exception:
                    pass  # Ignore errors when cleaning old backups
        
        except Exception as e:
            # Silent exception - don't interrupt normal operation if backup fails
            print(f"Warning: Could not create config backup: {e}")
    
    def _restore_config_from_backup(self):
        """Attempt to restore config from the most recent backup."""
        try:
            # Locate backups directory
            if os.path.isabs(CONFIG_FILE):
                config_dir = os.path.dirname(CONFIG_FILE)
            else:
                config_dir = os.path.dirname(os.path.abspath(CONFIG_FILE))
            backup_dir = os.path.join(config_dir, "config_backups")
            
            if not os.path.exists(backup_dir):
                return
            
            # Find most recent backup
            backups = [os.path.join(backup_dir, f) for f in os.listdir(backup_dir) 
                      if f.startswith("config_") and f.endswith(".json.bak")]
            
            if not backups:
                return
                
            backups.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_backup = backups[0]
            
            # Copy backup to config file
            shutil.copy2(latest_backup, CONFIG_FILE)
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(None, "Config Restored", 
                              f"Configuration was restored from backup: {os.path.basename(latest_backup)}")
            
            # Reload config
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                    self.config = decrypt_config(self.config)
            except Exception as e:
                QMessageBox.critical(None, "Error", f"Failed to reload configuration: {e}")
                
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "Restore Failed", f"Could not restore config from backup: {e}")
            
    def _create_manual_config_backup(self):
        """Create a manual config backup."""
        try:
            # Force create backup even if config file doesn't exist
            self._backup_config_file()
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(None, "Backup Created", "Configuration backup created successfully!")
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "Backup Failed", f"Failed to create backup: {e}")
    
    def _open_backup_folder(self):
        """Open the config backups folder in file explorer."""
        try:
            if os.path.isabs(CONFIG_FILE):
                config_dir = os.path.dirname(CONFIG_FILE)
            else:
                config_dir = os.path.dirname(os.path.abspath(CONFIG_FILE))
            backup_dir = os.path.join(config_dir, "config_backups")
            
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir, exist_ok=True)
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(None, "Backup Folder", f"Created backup folder: {backup_dir}")
            
            # Open folder in explorer (cross-platform)
            import subprocess
            import platform
            
            if platform.system() == "Windows":
                os.startfile(backup_dir)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", backup_dir])
            else:  # Linux
                subprocess.run(["xdg-open", backup_dir])
                
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "Error", f"Could not open backup folder: {e}")
    
    def _manual_restore_config(self):
        """Show dialog to manually select and restore a config backup."""
        try:
            if os.path.isabs(CONFIG_FILE):
                config_dir = os.path.dirname(CONFIG_FILE)
            else:
                config_dir = os.path.dirname(os.path.abspath(CONFIG_FILE))
            backup_dir = os.path.join(config_dir, "config_backups")
            
            if not os.path.exists(backup_dir):
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(None, "No Backups", "No backup folder found. No backups have been created yet.")
                return
            
            # Get list of available backups
            backups = [f for f in os.listdir(backup_dir) 
                      if f.startswith("config_") and f.endswith(".json.bak")]
            
            if not backups:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(None, "No Backups", "No config backups found.")
                return
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: os.path.getmtime(os.path.join(backup_dir, x)), reverse=True)
            
            # Use WindowManager to create scrollable dialog
            dialog, scrollable_frame, canvas = self.wm.setup_scrollable(
                self.master,
                "Config Backup Manager",
                width=0,
                height=None,
                max_width_ratio=0.6,
                max_height_ratio=0.8
            )
            
            # Main content
            header_frame = tk.Frame(scrollable_frame)
            header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
            
            tk.Label(header_frame, text="Configuration Backup Manager", 
                    font=('TkDefaultFont', 14, 'bold')).pack(anchor=tk.W)
            
            tk.Label(header_frame, 
                    text="Select a backup to restore or manage your configuration backups.",
                    font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, pady=(5, 0))
            
            # Info section
            info_frame = tk.LabelFrame(scrollable_frame, text="Backup Information", padx=10, pady=10)
            info_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
            
            info_text = f"📁 Backup Location: {backup_dir}\n📊 Total Backups: {len(backups)}"
            tk.Label(info_frame, text=info_text, font=('TkDefaultFont', 10), 
                    fg='#333', justify=tk.LEFT).pack(anchor=tk.W)
            
            # Backup list section
            list_frame = tk.LabelFrame(scrollable_frame, text="Available Backups (Newest First)", padx=10, pady=10)
            list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
            
            # Create treeview for better display
            columns = ('timestamp', 'filename', 'size')
            tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=8)
            
            # Define headings
            tree.heading('timestamp', text='Date & Time')
            tree.heading('filename', text='Backup File')
            tree.heading('size', text='Size')
            
            # Configure column widths
            tree.column('timestamp', width=150, anchor='center')
            tree.column('filename', width=200)
            tree.column('size', width=80, anchor='center')
            
            # Add scrollbars for treeview
            v_scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=tree.yview)
            h_scrollbar = ttk.Scrollbar(list_frame, orient='horizontal', command=tree.xview)
            tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            
            # Pack treeview and scrollbars
            tree.pack(side='left', fill='both', expand=True)
            v_scrollbar.pack(side='right', fill='y')
            h_scrollbar.pack(side='bottom', fill='x')
            
            # Populate treeview with backup information
            backup_items = []
            for backup in backups:
                backup_path = os.path.join(backup_dir, backup)
                
                # Extract timestamp from filename
                try:
                    timestamp_part = backup.replace("config_", "").replace(".json.bak", "")
                    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", 
                                                  time.strptime(timestamp_part, "%Y%m%d_%H%M%S"))
                except:
                    formatted_time = "Unknown"
                
                # Get file size
                try:
                    size_bytes = os.path.getsize(backup_path)
                    if size_bytes < 1024:
                        size_str = f"{size_bytes} B"
                    elif size_bytes < 1024 * 1024:
                        size_str = f"{size_bytes // 1024} KB"
                    else:
                        size_str = f"{size_bytes // (1024 * 1024)} MB"
                except:
                    size_str = "Unknown"
                
                # Insert into treeview
                item_id = tree.insert('', 'end', values=(formatted_time, backup, size_str))
                backup_items.append((item_id, backup, formatted_time))
            
            # Select first item by default
            if backup_items:
                tree.selection_set(backup_items[0][0])
                tree.focus(backup_items[0][0])
            
            # Action buttons frame
            button_frame = tk.LabelFrame(scrollable_frame, text="Actions", padx=10, pady=10)
            button_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
            
            # Create button layout
            button_row1 = tk.Frame(button_frame)
            button_row1.pack(fill=tk.X, pady=(0, 5))
            
            button_row2 = tk.Frame(button_frame)
            button_row2.pack(fill=tk.X)
            
            def get_selected_backup():
                """Get currently selected backup from treeview"""
                selection = tree.selection()
                if not selection:
                    return None
                    
                selected_item = selection[0]
                for item_id, backup_filename, formatted_time in backup_items:
                    if item_id == selected_item:
                        return backup_filename, formatted_time
                return None
            
            def restore_selected():
                selected = get_selected_backup()
                if not selected:
                    messagebox.showwarning("No Selection", "Please select a backup to restore.")
                    return
                
                selected_backup, formatted_time = selected
                backup_path = os.path.join(backup_dir, selected_backup)
                
                # Confirm restore
                if messagebox.askyesno("Confirm Restore", 
                                     f"This will replace your current configuration with the backup from:\n\n"
                                     f"{formatted_time}\n{selected_backup}\n\n"
                                     f"A backup of your current config will be created first.\n\n"
                                     f"Are you sure you want to continue?"):
                    
                    try:
                        # Create backup of current config before restore
                        self._backup_config_file()
                        
                        # Copy backup to config file
                        shutil.copy2(backup_path, CONFIG_FILE)
                        
                        messagebox.showinfo("Restore Complete", 
                                          f"Configuration restored from: {selected_backup}\n\n"
                                          f"Please restart the application for changes to take effect.")
                        dialog._cleanup_scrolling()
                        dialog.destroy()
                        
                    except Exception as e:
                        messagebox.showerror("Restore Failed", f"Failed to restore backup: {e}")
            
            def delete_selected():
                selected = get_selected_backup()
                if not selected:
                    messagebox.showwarning("No Selection", "Please select a backup to delete.")
                    return
                
                selected_backup, formatted_time = selected
                
                if messagebox.askyesno("Confirm Delete", 
                                     f"Delete backup from {formatted_time}?\n\n{selected_backup}\n\n"
                                     f"This action cannot be undone."):
                    try:
                        os.remove(os.path.join(backup_dir, selected_backup))
                        
                        # Remove from treeview
                        selection = tree.selection()
                        if selection:
                            tree.delete(selection[0])
                        
                        # Update backup items list
                        backup_items[:] = [(item_id, backup, time_str) 
                                         for item_id, backup, time_str in backup_items 
                                         if backup != selected_backup]
                        
                        messagebox.showinfo("Deleted", "Backup deleted successfully.")
                    except Exception as e:
                        messagebox.showerror("Delete Failed", f"Failed to delete backup: {e}")
            
            def create_new_backup():
                """Create a new manual backup"""
                try:
                    self._backup_config_file()
                    messagebox.showinfo("Backup Created", "New configuration backup created successfully!")
                    # Refresh the dialog
                    dialog._cleanup_scrolling()
                    dialog.destroy()
                    self._manual_restore_config()  # Reopen with updated list
                except Exception as e:
                    messagebox.showerror("Backup Failed", f"Failed to create backup: {e}")
            
            def open_backup_folder():
                """Open backup folder in file explorer"""
                self._open_backup_folder()
            
            # Primary action buttons (Row 1)
            tb.Button(button_row1, text="✅ Restore Selected", 
                     command=restore_selected, bootstyle="success", 
                     width=20).pack(side=tk.LEFT, padx=(0, 10))
                     
            tb.Button(button_row1, text="💾 Create New Backup", 
                     command=create_new_backup, bootstyle="primary-outline", 
                     width=20).pack(side=tk.LEFT, padx=(0, 10))
                     
            tb.Button(button_row1, text="📁 Open Folder", 
                     command=open_backup_folder, bootstyle="info-outline", 
                     width=20).pack(side=tk.LEFT)
            
            # Secondary action buttons (Row 2)
            tb.Button(button_row2, text="🗑️ Delete Selected", 
                     command=delete_selected, bootstyle="danger-outline", 
                     width=20).pack(side=tk.LEFT, padx=(0, 10))
                     
            tb.Button(button_row2, text="❌ Close", 
                     command=lambda: [dialog._cleanup_scrolling(), dialog.destroy()], 
                     bootstyle="secondary", 
                     width=20).pack(side=tk.RIGHT)
            
            # Auto-resize and show dialog
            self.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.7, max_height_ratio=0.9)
            
            # Handle window close
            dialog.protocol("WM_DELETE_WINDOW", lambda: [dialog._cleanup_scrolling(), dialog.destroy()])
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open backup restore dialog: {e}")
            
    def _ensure_executor(self):
        """Ensure a ThreadPoolExecutor exists and matches configured worker count.
        Also updates EXTRACTION_WORKERS environment variable.
        """
        try:
            workers = 1
            try:
                workers = int(self.extraction_workers_var.get()) if self.enable_parallel_extraction_var.get() else 1
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

if __name__ == "__main__":
    import time
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
    
    print("🚀 Starting Glossarion v5.0.5...")
    
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
        
        # Add cleanup handler for graceful shutdown
        def on_closing():
            """Handle application shutdown gracefully to avoid GIL issues"""
            try:
                # Stop any background threads before destroying GUI
                if hasattr(app, 'stop_all_operations'):
                    app.stop_all_operations()
                
                # Give threads a moment to stop
                import time
                time.sleep(0.1)
                
                # Destroy window
                root.quit()
                root.destroy()
            except Exception:
                # Force exit if cleanup fails
                import os
                os._exit(0)
        
        # Set the window close handler
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Add signal handlers for clean shutdown
        def signal_handler(signum, frame):
            """Handle system signals gracefully"""
            print(f"Received signal {signum}, shutting down gracefully...")
            try:
                on_closing()
            except Exception:
                os._exit(1)
        
        # Register signal handlers (Windows-safe)
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
        
        # Start main loop with error handling
        try:
            root.mainloop()
        except Exception as e:
            print(f"Main loop error: {e}")
        finally:
            # Ensure cleanup even if mainloop fails
            try:
                on_closing()
            except Exception:
                pass
        
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