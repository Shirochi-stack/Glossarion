# Standard Library
import io, json, logging, math, os, shutil, sys, threading, time, re
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk
from ai_hunter_enhanced import AIHunterConfigGUI, ImprovedAIHunterDetection
import traceback
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

class UIHelper:
    """Consolidated UI utility functions"""
    
    @staticmethod
    def setup_text_undo_redo(text_widget):
        """Set up undo/redo bindings for a text widget"""
        def handle_undo(event):
            try: 
                text_widget.edit_undo()
            except tk.TclError: 
                pass
            return "break"
        
        def handle_redo(event):
            try: 
                text_widget.edit_redo()
            except tk.TclError: 
                pass
            return "break"
        
        # Windows/Linux bindings
        text_widget.bind('<Control-z>', handle_undo)
        text_widget.bind('<Control-y>', handle_redo)
        # macOS bindings
        text_widget.bind('<Command-z>', handle_undo)
        text_widget.bind('<Command-Shift-z>', handle_redo)
    
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
        
        # Bind events
        wheel_handler = lambda e: on_mousewheel(e)
        wheel_up = lambda e: on_mousewheel_linux(e, -1)
        wheel_down = lambda e: on_mousewheel_linux(e, 1)
        
        dialog_window.bind_all("<MouseWheel>", wheel_handler)
        dialog_window.bind_all("<Button-4>", wheel_up)
        dialog_window.bind_all("<Button-5>", wheel_down)
        
        # Return cleanup function
        def cleanup_bindings():
            try:
                dialog_window.unbind_all("<MouseWheel>")
                dialog_window.unbind_all("<Button-4>")
                dialog_window.unbind_all("<Button-5>")
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
        text_widget = scrolledtext.ScrolledText(parent, 
                                               undo=True, 
                                               autoseparators=True, 
                                               maxundo=-1,
                                               **text_kwargs)
        UIHelper.setup_text_undo_redo(text_widget)
        return text_widget
    
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

class WindowManager:
    """Unified window geometry and dialog management - FULLY REFACTORED V2"""
    
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.ui = UIHelper()
        self._stored_geometries = {}
        self._pending_operations = {}
        self._dpi_scale = None
        self._topmost_protection_active = {}
    
    def get_dpi_scale(self, window):
        """Get and cache DPI scaling factor"""
        if self._dpi_scale is None:
            try:
                self._dpi_scale = window.tk.call('tk', 'scaling') / 1.333
            except:
                self._dpi_scale = 1.0
        return self._dpi_scale
    
    def setup_window(self, window, width=None, height=None, 
                    center=True, icon=True, hide_initially=False,
                    max_width_ratio=0.98, max_height_ratio=0.98,
                    min_width=400, min_height=300):
        """Universal window setup with proper deferral and DPI awareness"""
        
        if hide_initially:
            window.withdraw()
        
        # Always ensure not topmost
        window.attributes('-topmost', False)
        
        if icon:
            window.after_idle(lambda: load_application_icon(window, self.base_dir))
        
        screen_width = window.winfo_screenwidth()
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
        
        max_width = int(screen_width * max_width_ratio)
        max_height = int(screen_height * max_height_ratio)
        
        final_width = max(min_width, min(width, max_width))
        final_height = max(min_height, min(height, max_height))
        
        if center:
            x = max(0, (screen_width - final_width) // 2)
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
    
    def responsive_size(self, window, base_width, base_height, 
                       scale_factor=None, center=True, use_full_height=True):
        """Size window responsively based on screen size - FIXED"""
        
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        
        if use_full_height:
            # Maximize to 98% of screen
            width = min(int(base_width * 1.2), int(screen_width * 0.98))
            height = int(screen_height * 0.98)
        else:
            # Use base dimensions directly
            width = base_width
            height = base_height
            
            # Only scale down if window doesn't fit on screen
            if width > screen_width * 0.9:
                width = int(screen_width * 0.85)
            if height > screen_height * 0.9:
                height = int(screen_height * 0.85)
        
        if center:
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2
            geometry_str = f"{width}x{height}+{x}+{y}"
        else:
            geometry_str = f"{width}x{height}"
        
        window.geometry(geometry_str)
        
        # Ensure not topmost
        window.attributes('-topmost', False)
        
        return width, height
    
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
                        window_width = content_width + 120
                    else:
                        window_width = dialog.winfo_reqwidth()
                else:
                    window_width = dialog.winfo_reqwidth()
                
                window_width = int(window_width / dpi_scale)
                
                max_width = int(screen_width * max_width_ratio)
                final_width = min(window_width, max_width)
                final_width = max(final_width, 400)
                
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
            
    def center_window(self, window):
        """Center a window on screen"""
        def do_center():
            if window.winfo_exists():
                window.update_idletasks()
                width = window.winfo_width()
                height = window.winfo_height()
                screen_width = window.winfo_screenwidth()
                screen_height = window.winfo_screenheight()
                
                x = (screen_width - width) // 2
                y = (screen_height - height) // 2
                
                window.geometry(f"+{x}+{y}")
        
        window.after_idle(do_center)
class TranslatorGUI:
    def __init__(self, master):
        master.configure(bg='#2b2b2b')
        self.master = master
        
        # Initialize managers
        self.base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        self.wm = WindowManager(self.base_dir)
        self.ui = UIHelper()
        master.attributes('-topmost', False)
        master.lift()
        
        self.max_output_tokens = 8192
        self.proc = self.glossary_proc = None
        master.title("Glossarion v2.9.4")
        
        self.wm.responsive_size(master, BASE_WIDTH, BASE_HEIGHT)
        master.minsize(1600, 1000)
        self.wm.center_window(master)
        
        
        # Setup fullscreen support
        self.wm.setup_fullscreen_support(master)
        
        self.payloads_dir = os.path.join(os.getcwd(), "Payloads")
        
        self._modules_loaded = self._modules_loading = False
        self.stop_requested = False
        self.translation_thread = self.glossary_thread = self.qa_thread = self.epub_thread = None
        self.qa_thread = None
        # Glossary tracking
        self.manual_glossary_path = None
        self.auto_loaded_glossary_path = None
        self.auto_loaded_glossary_for_file = None
        
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
        except: self.config = {}
        
        self.max_output_tokens = self.config.get('max_output_tokens', self.max_output_tokens)
        
        # Default prompts
        self.default_translation_chunk_prompt = "[PART {chunk_idx}/{total_chunks}]\n{chunk_html}"
        self.default_image_chunk_prompt = "This is part {chunk_idx} of {total_chunks} of a longer image.  Ensure that all formatting (indentation, line breaks, spacing, paragraph structure, markup) exactly matches the style used in the previous chunks. {context}"
        self.default_prompts = {
            "korean": "You are a professional Korean to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                          "- Use an easy to read and grammatically accurate comedy translation style.\n"
                          "- Retain honorifics like -nim, -ssi.\n"
                          "- Preserve original intent, and speech tone.\n"
                          "- Retain onomatopoeia in Romaji.\n"
                          "- Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title>, <h1>, <h2>, <p>, <br>, <div>, etc.",
            "japanese": "You are a professional Japanese to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                          "- Use an easy to read and grammatically accurate comedy translation style.\n"
                          "- Retain honorifics like -san, -sama, -chan, -kun.\n"
                          "- Preserve original intent, and speech tone.\n"
                          "- Retain onomatopoeia in Romaji.\n"
                          "- Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title>, <h1>, <h2>, <p>, <br>, <div>, etc.",
            "chinese": "You are a professional Chinese to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                         "- Use an easy to read and grammatically accurate comedy translation style.\n"
                         "- Preserve original intent, and speech tone.\n"
                         "- Retain onomatopoeia in Romaji.\n"
                         "- Preserve ALL HTML tags exactly as they appear in the source, including <head>, <title>, <h1>, <h2>, <p>, <br>, <div>, etc.",
            "korean_OCR": "You are a professional Korean to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                          "- Use an easy to read and grammatically accurate comedy translation style.\n"
                          "- Retain honorifics like -nim, -ssi.\n"
                          "- Preserve original intent, and speech tone.\n"
                          "- Retain onomatopoeia in Romaji.\n"
                          "- Add HTML tags for proper formatting as expected of a novel.",
            "japanese_OCR": "You are a professional Japanese to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                            "- Use an easy to read and grammatically accurate comedy translation style.\n"
                            "- Retain honorifics like -san, -sama, -chan, -kun.\n"
                            "- Preserve original intent, and speech tone.\n"
                            "- Retain onomatopoeia in Romaji.\n"
                            "- Add HTML tags for proper formatting as expected of a novel.",
            "chinese_OCR": "You are a professional Chinese to English novel translator, you must strictly output only English text and HTML tags while following these rules:\n"
                           "- Use an easy to read and grammatically accurate comedy translation style.\n"
                           "- Preserve original intent, and speech tone.\n"
                           "- Retain onomatopoeia in Romaji.\n"
                           "- Add HTML tags for proper formatting as expected of a novel.",

            # TXT-only entries
            "korean_TXT": "You are a professional Korean to English novel translator, you must strictly output only English text while following these rules:\n"
                           "- Use an easy to read and grammatically accurate comedy translation style.\n"
                           "- Retain honorifics like -nim, -ssi.\n"
                           "- Preserve original intent, and speech tone.\n"
                           "- Retain onomatopoeia in Romaji.\n"
                           "- Use line breaks for proper formatting as expected of a novel.",
            "japanese_TXT": "You are a professional Japanese to English novel translator, you must strictly output only English text while following these rules:\n"
                             "- Use an easy to read and grammatically accurate comedy translation style.\n"
                             "- Retain honorifics like -san, -sama, -chan, -kun.\n"
                             "- Preserve original intent, and speech tone.\n"
                             "- Retain onomatopoeia in Romaji.\n"
                             "- Use line breaks for proper formatting as expected of a novel.",
            "chinese_TXT": "You are a professional Chinese to English novel translator, you must strictly output only English text while following these rules:\n"
                            "- Use an easy to read and grammatically accurate comedy translation style.\n"
                            "- Preserve original intent, and speech tone.\n"
                            "- Retain onomatopoeia in Romaji.\n"
                            "- Use line breaks for proper formatting as expected of a novel.",

            "Original": "Return everything exactly as seen on the source."
        }

        
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
        self.append_glossary_prompt = self.config.get('append_glossary_prompt', "Character/Term Glossary (use these translations consistently):")
        self.translation_chunk_prompt = self.config.get('translation_chunk_prompt', self.default_translation_chunk_prompt)
        self.image_chunk_prompt = self.config.get('image_chunk_prompt', self.default_image_chunk_prompt)
        
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
            ('disable_zero_detection_var', 'disable_zero_detection', True),
            ('use_header_as_output_var', 'use_header_as_output', False),
            ('emergency_restore_var', 'emergency_paragraph_restore', False),
            ('contextual_var', 'contextual', True),
            ('REMOVE_AI_ARTIFACTS_var', 'REMOVE_AI_ARTIFACTS', False),
            ('enable_watermark_removal_var', 'enable_watermark_removal', True),
            ('save_cleaned_images_var', 'save_cleaned_images', False),
            ('advanced_watermark_removal_var', 'advanced_watermark_removal', False),
            ('enable_decimal_chapters_var', 'enable_decimal_chapters', False)
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
            ('max_images_per_chapter_var', 'max_images_per_chapter', '1'),
            ('image_chunk_height_var', 'image_chunk_height', '1500'),
            ('chunk_timeout_var', 'chunk_timeout', '900'),
            ('batch_size_var', 'batch_size', '3'),
            ('chapter_number_offset_var', 'chapter_number_offset', '0'),
            ('compression_factor_var', 'compression_factor', '1.0') 
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
        self.append_log("🚀 Glossarion v2.9.4 - Ready to use!")
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
        models = [
            # OpenAI Models
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1",
            "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k",
            "o1-preview", "o1-mini", "o4-mini",
            
            # Google Gemini Models
            "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash",
            "gemini-2.5-flash", "gemini-2.5-pro", "gemini-pro", "gemini-pro-vision",
            
            # Anthropic Claude Models
            "claude-opus-4-20250514", "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219",
            "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
            "claude-2.1", "claude-2", "claude-instant-1.2",
            
            # DeepSeek Models
            "deepseek-chat", "deepseek-coder", "deepseek-coder-33b-instruct",
            
            # Mistral Models
            "mistral-large", "mistral-medium", "mistral-small", "mistral-tiny",
            "mixtral-8x7b-instruct", "mixtral-8x22b", "codestral-latest",
            
            # Meta Llama Models (via Together/other providers)
            "llama-2-7b-chat", "llama-2-13b-chat", "llama-2-70b-chat",
            "llama-3-8b-instruct", "llama-3-70b-instruct", "codellama-34b-instruct",
            
            # Yi Models
            "yi-34b-chat", "yi-34b-chat-200k", "yi-6b-chat",
            
            # Qwen Models
            "qwen-72b-chat", "qwen-14b-chat", "qwen-7b-chat", "qwen-plus", "qwen-turbo",
            
            # Cohere Models
            "command", "command-light", "command-nightly", "command-r", "command-r-plus",
            
            # AI21 Models
            "j2-ultra", "j2-mid", "j2-light", "jamba-instruct",
            
            # Perplexity Models
            "perplexity-70b-online", "perplexity-7b-online", "pplx-70b-online", "pplx-7b-online",
            
            # Groq Models (usually with suffix)
            "llama-3-70b-groq", "llama-3-8b-groq", "mixtral-8x7b-groq",
            
            # Chinese Models
            "glm-4", "glm-3-turbo", "chatglm-6b", "chatglm2-6b", "chatglm3-6b",
            "baichuan-13b-chat", "baichuan2-13b-chat",
            "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k",
            
            # Other Models
            "falcon-40b-instruct", "falcon-7b-instruct",
            "phi-2", "phi-3-mini", "phi-3-small", "phi-3-medium",
            "orca-2-13b", "orca-2-7b",
            "vicuna-13b", "vicuna-7b",
            "alpaca-7b",
            "wizardlm-70b", "wizardlm-13b",
            "openchat-3.5",
            
            # For ElectronHub, prefix with 'eh/'
            "eh/gpt-4", "eh/gpt-3.5-turbo", "eh/claude-3-opus", "eh/claude-3-sonnet",
            "eh/llama-2-70b-chat", "eh/yi-34b-chat-200k", "eh/mistral-large",
            "eh/gemini-pro", "eh/deepseek-coder-33b",
        ]
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

    def configure_title_prompt(self):
        """Configure the book title translation prompt"""
        dialog = self.wm.create_simple_dialog(
            self.master,
            "Configure Book Title Translation",
            width=950,
            height=700
        )
        
        main_frame = tk.Frame(dialog, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="Book Title Translation Prompt", 
                font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        tk.Label(main_frame, text="This prompt will be used when translating book titles.\n"
                "The book title will be appended after this prompt.",
                font=('TkDefaultFont', 11), fg='gray').pack(anchor=tk.W, pady=(0, 10))
        
        self.title_prompt_text = self.ui.setup_scrollable_text(
            main_frame, height=8, wrap=tk.WORD
        )
        self.title_prompt_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.title_prompt_text.insert('1.0', self.book_title_prompt)
        
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
                    # If no output file, try to construct from original basename
                    if chapter_dict['original_basename']:
                        chapter_dict['filename'] = os.path.join(output_dir, f"response_{chapter_dict['original_basename']}.html")
                    else:
                        # Last resort: use chapter number
                        chapter_num = chapter_info.get('actual_num', chapter_info.get('chapter_num', 0))
                        chapter_dict['filename'] = os.path.join(output_dir, f"response_{chapter_num:04d}.html")
                
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
        
        return uses_zero_based
    
    def force_retranslation(self):
        """Force retranslation of specific chapters with improved display"""
        input_path = self.entry_epub.get()
        if not input_path or not os.path.isfile(input_path):
            messagebox.showerror("Error", "Please select a valid EPUB or text file first.")
            return
        
        epub_base = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = epub_base  # KEEP THIS AS IS - it's relative to current working directory
        
        if not os.path.exists(output_dir):
            messagebox.showinfo("Info", "No translation output found for this EPUB.")
            return
        
        progress_file = os.path.join(output_dir, "translation_progress.json")
        if not os.path.exists(progress_file):
            messagebox.showinfo("Info", "No progress tracking found.")
            return
        
        with open(progress_file, 'r', encoding='utf-8') as f:
            prog = json.load(f)
            
            # Check if there are any chapters
            if not prog.get("chapters"):
                messagebox.showinfo("Info", "No chapters found in progress tracking.")
                return
            
            # Group all entries by their output filename to handle duplicates
            files_to_entries = {}
            no_file_entries = []
            
            for chapter_key, chapter_info in prog.get("chapters", {}).items():
                output_file = chapter_info.get("output_file", "")
                
                if output_file:
                    if output_file not in files_to_entries:
                        files_to_entries[output_file] = []
                    files_to_entries[output_file].append((chapter_key, chapter_info))
                else:
                    no_file_entries.append((chapter_key, chapter_info))
            
            #print(f"Found {len(files_to_entries)} unique files from {len(prog.get('chapters', {}))} total entries")
        
            # Load chapters info if available
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
            
            # Create dialog using WindowManager
            dialog = self.wm.create_simple_dialog(
                self.master,
                "Force Retranslation",
                width=900,
                height=600  # Back to original height since no toggle
            )
            
            # Add instructions label
            instruction_text = "Select chapters to retranslate (scroll horizontally if needed):"
            tk.Label(dialog, text=instruction_text, font=('Arial', 12)).pack(pady=10)
            
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
                selectmode=tk.MULTIPLE, 
                yscrollcommand=v_scrollbar.set,
                xscrollcommand=h_scrollbar.set,
                width=100
            )
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Configure scrollbars
            v_scrollbar.config(command=listbox.yview)
            h_scrollbar.config(command=listbox.xview)
            
            # Build display info - ONE entry per unique file
            chapter_display_info = []
            
            for output_file, entries in files_to_entries.items():
                # Use the first entry for each file (they're duplicates anyway)
                chapter_key, chapter_info = entries[0]
                
                # Extract chapter number from filename using existing patterns
                chapter_num = 0
                
                # Import the pattern from TransateKRtoEN at the top of the method if not already imported
                from TransateKRtoEN import extract_chapter_number_from_filename
                
                # Use the existing function that handles all the patterns
                chapter_num, _ = extract_chapter_number_from_filename(output_file)
                
                # Determine chapter number with proper None handling
                chapter_num = 0
                if 'actual_num' in chapter_info and chapter_info['actual_num'] is not None:
                    chapter_num = chapter_info['actual_num']
                elif 'chapter_num' in chapter_info and chapter_info['chapter_num'] is not None:
                    chapter_num = chapter_info['chapter_num']
                else:
                    # Fallback: try to extract from output filename
                    if output_file:
                        match = re.search(r'response_(\d+)', output_file)
                        if match:
                            chapter_num = int(match.group(1))
                        else:
                            # Last resort: use a sequential number based on position
                            chapter_num = len(chapter_display_info) + 1
                    else:
                        chapter_num = len(chapter_display_info) + 1
                
                # Get status and validate
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
                    'duplicate_count': len(entries)  # Track how many duplicates
                })
            
            # Sort by chapter number with None handling
            chapter_display_info.sort(key=lambda x: x['num'] if x['num'] is not None else 999999)
            
            # Populate listbox
            for info in chapter_display_info:
                chapter_num = info['num']
                status = info['status']
                output_file = info['output_file']
                
                # Status indicators
                status_icons = {
                    'completed': '✅',
                    'failed': '❌',
                    'qa_failed': '❌',
                    'file_missing': '⚠️',
                    'in_progress': '🔄',
                    'unknown': '❓'
                }
                
                icon = status_icons.get(status, '❓')
                
                # Handle both integer and decimal chapter numbers
                if isinstance(chapter_num, float) and chapter_num.is_integer():
                    display = f"Chapter {int(chapter_num):03d} | {icon} {status} | {output_file}"
                elif isinstance(chapter_num, float):
                    display = f"Chapter {chapter_num:06.1f} | {icon} {status} | {output_file}"
                else:
                    display = f"Chapter {chapter_num:03d} | {icon} {status} | {output_file}"
                
                # Add duplicate count if more than 1
                if info['duplicate_count'] > 1:
                    display += f" | ({info['duplicate_count']} entries)"
                
                listbox.insert(tk.END, display)
            
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
            
            def select_all():
                listbox.select_set(0, tk.END)
                update_selection_count()
            
            def clear_selection():
                listbox.select_clear(0, tk.END)
                update_selection_count()
            
            def select_status(status_to_select):
                listbox.select_clear(0, tk.END)
                for idx, info in enumerate(chapter_display_info):
                    # Check for both 'failed' and 'qa_failed' statuses
                    if status_to_select == 'failed':
                        if info['status'] in ['failed', 'qa_failed']:
                            listbox.select_set(idx)
                    else:
                        if info['status'] == status_to_select:
                            listbox.select_set(idx)
                update_selection_count()
            
            def remove_qa_failed_mark():
                selected = listbox.curselection()
                if not selected:
                    messagebox.showwarning("No Selection", "Please select at least one chapter.")
                    return
                
                # Filter for QA failed chapters only
                selected_chapters = [chapter_display_info[i] for i in selected]
                qa_failed_chapters = [ch for ch in selected_chapters if ch['status'] == 'qa_failed']
                
                if not qa_failed_chapters:
                    messagebox.showwarning("No QA Failed Chapters", 
                                         "None of the selected chapters have 'qa_failed' status.\n"
                                         "Only QA failed chapters can have their mark removed.")
                    return
                
                # Build confirmation message
                count = len(qa_failed_chapters)
                if count > 10:
                    confirm_msg = f"This will remove QA failed mark from {count} chapters.\n\nContinue?"
                else:
                    chapters_text = [f"Chapter {info['num']}" for info in qa_failed_chapters]
                    confirm_msg = f"This will remove QA failed mark from:\n\n{', '.join(chapters_text)}\n\nContinue?"
                
                # Add warning if non-QA-failed chapters are also selected
                non_qa_failed = [ch for ch in selected_chapters if ch['status'] != 'qa_failed']
                if non_qa_failed:
                    confirm_msg += f"\n\nNote: {len(non_qa_failed)} non-QA-failed chapters in selection will be ignored."
                
                if not messagebox.askyesno("Confirm Remove QA Failed Mark", confirm_msg):
                    return
                
                # Remove QA failed mark from chapters
                cleared_count = 0
                for idx in selected:
                    info = chapter_display_info[idx]
                    
                    # Only process QA failed chapters
                    if info['status'] != 'qa_failed':
                        continue
                    
                    output_file = info['output_file']
                    
                    # Update ALL duplicate entries for this file from progress
                    if output_file in files_to_entries:
                        for chapter_key, chapter_info in files_to_entries[output_file]:
                            if chapter_key in prog["chapters"]:
                                # Change status from qa_failed back to completed
                                prog["chapters"][chapter_key]["status"] = "completed"
                                
                                # Remove all QA-related fields
                                prog["chapters"][chapter_key].pop("qa_issues", None)
                                prog["chapters"][chapter_key].pop("qa_timestamp", None)
                                prog["chapters"][chapter_key].pop("qa_issues_found", None)
                                prog["chapters"][chapter_key].pop("duplicate_confidence", None)
                                
                                cleared_count += 1
                
                # Save updated progress
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(prog, f, ensure_ascii=False, indent=2)
                
                messagebox.showinfo("Success", 
                    f"Removed QA failed mark from {cleared_count} chapters.\n\n"
                    "They are now marked as completed.")
                
                dialog.destroy()
            
            def retranslate_selected():
                selected = listbox.curselection()
                if not selected:
                    messagebox.showwarning("No Selection", "Please select at least one chapter.")
                    return
                
                # Build confirmation message for retranslation
                count = len(selected)
                if count > 10:
                    confirm_msg = f"This will delete {count} translated chapters and mark them for retranslation.\n\nContinue?"
                else:
                    chapters = [f"Chapter {chapter_display_info[i]['num']}" for i in selected]
                    confirm_msg = f"This will delete and retranslate:\n\n{', '.join(chapters)}\n\nContinue?"
                
                if not messagebox.askyesno("Confirm Retranslation", confirm_msg):
                    return
                
                # Regular retranslation logic
                deleted_count = 0
                for idx in selected:
                    info = chapter_display_info[idx]
                    output_file = info['output_file']
                    
                    # Delete the output file
                    if output_file:
                        output_path = os.path.join(output_dir, output_file)
                        try:
                            if os.path.exists(output_path):
                                os.remove(output_path)
                                deleted_count += 1
                        except Exception as e:
                            print(f"Failed to delete {output_path}: {e}")
                    
                    # Remove ALL duplicate entries for this file from progress
                    if output_file in files_to_entries:
                        for chapter_key, _ in files_to_entries[output_file]:
                            if chapter_key in prog["chapters"]:
                                # Get content hash before deleting
                                content_hash = prog["chapters"][chapter_key].get("content_hash")
                                
                                # Delete from chapters
                                del prog["chapters"][chapter_key]
                                
                                # Clean up related data
                                if content_hash and content_hash in prog.get("content_hashes", {}):
                                    del prog["content_hashes"][content_hash]
                                
                                if content_hash and content_hash in prog.get("chapter_chunks", {}):
                                    del prog["chapter_chunks"][content_hash]
                                
                                if chapter_key in prog.get("chapter_chunks", {}):
                                    del prog["chapter_chunks"][chapter_key]
                
                # Save updated progress
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(prog, f, ensure_ascii=False, indent=2)
                
                messagebox.showinfo("Success", 
                    f"Deleted {deleted_count} files and cleared {len(selected)} chapters from tracking.\n\n"
                    "They will be retranslated on the next run.")
                
                dialog.destroy()
            
            # Add buttons with improved layout
            # Configure column weights for better distribution
            for i in range(4):
                button_frame.columnconfigure(i, weight=1)
            
            # Row 1: Selection buttons
            tb.Button(button_frame, text="Select All", command=select_all, bootstyle="info").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
            tb.Button(button_frame, text="Clear Selection", command=clear_selection, bootstyle="secondary").grid(row=0, column=1, padx=5, pady=5, sticky="ew")
            tb.Button(button_frame, text="Select Completed", command=lambda: select_status('completed'), bootstyle="success").grid(row=0, column=2, padx=5, pady=5, sticky="ew")
            tb.Button(button_frame, text="Select Failed", command=lambda: select_status('failed'), bootstyle="danger").grid(row=0, column=3, padx=5, pady=5, sticky="ew")
            
            # Row 2: Action buttons - spanning multiple columns for better balance
            tb.Button(button_frame, text="Retranslate Selected", command=retranslate_selected, 
                     bootstyle="warning").grid(row=1, column=0, columnspan=2, padx=5, pady=10, sticky="ew")
            tb.Button(button_frame, text="Remove QA Failed Mark", command=remove_qa_failed_mark, 
                     bootstyle="success").grid(row=1, column=2, columnspan=1, padx=5, pady=10, sticky="ew")
            tb.Button(button_frame, text="Cancel", command=dialog.destroy, 
                     bootstyle="secondary").grid(row=1, column=3, columnspan=1, padx=5, pady=10, sticky="ew")

    
    def glossary_manager(self):
       """Open comprehensive glossary management dialog"""
       # Create scrollable dialog (stays hidden)
       dialog, scrollable_frame, canvas = self.wm.setup_scrollable(
           self.master, 
           "Glossary Manager",
           width=0,  # Will be auto-sized
           height=None,
           max_width_ratio=0.8,
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
               # Manual glossary fields
               for field, var in self.manual_field_vars.items():
                   self.config[f'manual_extract_{field}'] = var.get()
               
               self.config['custom_glossary_fields'] = self.custom_glossary_fields
               
               # Prompts
               self.manual_glossary_prompt = self.manual_prompt_text.get('1.0', tk.END).strip()
               self.auto_glossary_prompt = self.auto_prompt_text.get('1.0', tk.END).strip()
               self.config['manual_glossary_prompt'] = self.manual_glossary_prompt
               self.config['enable_auto_glossary'] = self.enable_auto_glossary_var.get()
               self.config['append_glossary'] = self.append_glossary_var.get()
               self.config['auto_glossary_prompt'] = self.auto_glossary_prompt
               self.append_glossary_prompt = self.append_prompt_text.get('1.0', tk.END).strip()
               self.config['append_glossary_prompt'] = self.append_glossary_prompt
               
               # Temperature and context limit
               try:
                   self.config['manual_glossary_temperature'] = float(self.manual_temp_var.get())
                   self.config['manual_context_limit'] = int(self.manual_context_var.get())
               except ValueError:
                   messagebox.showwarning("Invalid Input", 
                       "Please enter valid numbers for temperature and context limit")
                   return
               
               # Update environment variables
               os.environ['GLOSSARY_SYSTEM_PROMPT'] = self.manual_glossary_prompt
               os.environ['AUTO_GLOSSARY_PROMPT'] = self.auto_glossary_prompt
               
               # Set field extraction flags
               enabled_fields = []
               for field, var in self.manual_field_vars.items():
                   env_key = f'GLOSSARY_EXTRACT_{field.upper()}'
                   os.environ[env_key] = '1' if var.get() else '0'
                   if var.get():
                       enabled_fields.append(field)
               
               if self.custom_glossary_fields:
                   os.environ['GLOSSARY_CUSTOM_FIELDS'] = json.dumps(self.custom_glossary_fields)
               
               # Save config
               with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                   json.dump(self.config, f, ensure_ascii=False, indent=2)
               
               self.append_log("✅ Glossary settings saved successfully")
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
       self.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.8, max_height_ratio=1.43)
       
       dialog.protocol("WM_DELETE_WINDOW", 
                      lambda: [dialog._cleanup_scrolling(), dialog.destroy()])

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
       
       self.manual_prompt_text = self.ui.setup_scrollable_text(
           prompt_frame, height=12, wrap=tk.WORD
       )
       self.manual_prompt_text.pack(fill=tk.BOTH, expand=True)
       self.manual_prompt_text.insert('1.0', self.manual_glossary_prompt)
       self.manual_prompt_text.edit_reset()
       
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
        
        # Master toggle
        master_toggle_frame = tk.Frame(auto_container)
        master_toggle_frame.pack(fill=tk.X, pady=(0, 15))
        
        tb.Checkbutton(master_toggle_frame, text="Enable Automatic Glossary Generation", 
                      variable=self.enable_auto_glossary_var,
                      bootstyle="round-toggle").pack(side=tk.LEFT)
        
        tk.Label(master_toggle_frame, text="(Automatically extracts and translates character names/terms during translation)",
                font=('TkDefaultFont', 10), fg='gray').pack(side=tk.LEFT, padx=(10, 0))
        
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
        
        
        self.append_prompt_text.insert('1.0', self.append_glossary_prompt)
        self.append_prompt_text.edit_reset()
        
        append_prompt_controls = tk.Frame(append_prompt_frame)
        append_prompt_controls.pack(fill=tk.X, pady=(5, 0))
        
        def reset_append_prompt():
            if messagebox.askyesno("Reset Prompt", "Reset to default glossary append format?"):
                self.append_prompt_text.delete('1.0', tk.END)
                self.append_prompt_text.insert('1.0', "Character/Term Glossary (use these translations consistently):")
        
        tb.Button(append_prompt_controls, text="Reset to Default", command=reset_append_prompt, 
                 bootstyle="warning").pack(side=tk.LEFT, padx=5)
        
        # Extraction settings
        settings_container = tk.Frame(auto_container)
        settings_container.pack(fill=tk.BOTH, expand=True)
        
        settings_label_frame = tk.LabelFrame(settings_container, text="Targeted Extraction Settings", padx=10, pady=10)
        settings_label_frame.pack(fill=tk.X, pady=(0, 15))
        
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
        
        # Help text
        help_frame = tk.Frame(settings_container)
        help_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(help_frame, text="💡 Settings Guide:", font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W)
        help_texts = [
            "• Min frequency: How many times a name must appear (lower = more terms)",
            "• Max names/titles: Limits to prevent huge glossaries",
            "• Translation batch: Terms per API call (larger = faster but may reduce quality)"
        ]
        for txt in help_texts:
            tk.Label(help_frame, text=txt, font=('TkDefaultFont', 11), fg='gray').pack(anchor=tk.W, padx=20)
        
        # Auto prompt section
        auto_prompt_frame = tk.LabelFrame(settings_container, text="Extraction Prompt Template", padx=10, pady=10)
        auto_prompt_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(auto_prompt_frame, text="Available placeholders: {language}, {min_frequency}, {max_names}, {max_titles}",
                font=('TkDefaultFont', 9), fg='blue').pack(anchor=tk.W, pady=(0, 5))
        
        self.auto_prompt_text = self.ui.setup_scrollable_text(
            auto_prompt_frame, height=12, wrap=tk.WORD
        )
        self.auto_prompt_text.pack(fill=tk.BOTH, expand=True)
        self.auto_prompt_text.insert('1.0', self.auto_glossary_prompt)
        self.auto_prompt_text.edit_reset()
        
        auto_prompt_controls = tk.Frame(settings_container)
        auto_prompt_controls.pack(fill=tk.X, pady=(10, 0))
        
        def reset_auto_prompt():
            if messagebox.askyesno("Reset Prompt", "Reset automatic glossary prompt to default?"):
                self.auto_prompt_text.delete('1.0', tk.END)
                self.auto_prompt_text.insert('1.0', self.default_auto_glossary_prompt)
        
        tb.Button(auto_prompt_controls, text="Reset to Default", command=reset_auto_prompt, 
                 bootstyle="warning").pack(side=tk.LEFT, padx=5)
        
        # Update states function with proper error handling
        def update_auto_glossary_state():
            try:
                if not extraction_grid.winfo_exists():
                    return
                state = tk.NORMAL if self.enable_auto_glossary_var.get() else tk.DISABLED
                for widget in extraction_grid.winfo_children():
                    if isinstance(widget, (tb.Entry, ttk.Entry)):
                        widget.config(state=state)
                if self.auto_prompt_text.winfo_exists():
                    self.auto_prompt_text.config(state=state)
                for widget in auto_prompt_controls.winfo_children():
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
            
            # Use WindowManager's setup_scrollable for unified scrolling
            dialog, scrollable_frame, canvas = self.wm.setup_scrollable(
                self.master,
                "Smart Trim Glossary",
                width=600,
                height=None,  # Will use default height calculation
                max_width_ratio=0.9,
                max_height_ratio=0.85
            )
            
            main_frame = scrollable_frame
            
            # Title and description
            tk.Label(main_frame, text="Smart Glossary Trimming", 
                    font=('TkDefaultFont', 14, 'bold')).pack(pady=(20, 5))
            
            tk.Label(main_frame, text="Optimize your glossary by removing unnecessary data and limiting field sizes",
                    font=('TkDefaultFont', 10), fg='gray', wraplength=550).pack(pady=(0, 15))
            
            # Analyze current glossary to detect all fields
            all_fields = set()
            list_fields = set()
            standard_list_fields = ['traits', 'locations', 'group_affiliation']
            
            if self.current_glossary_format == 'list':
                for entry in self.current_glossary_data:
                    for field, value in entry.items():
                        all_fields.add(field)
                        if isinstance(value, list):
                            list_fields.add(field)
            elif self.current_glossary_format == 'dict':
                for key, entry in self.current_glossary_data.get('entries', {}).items():
                    for field, value in entry.items():
                        all_fields.add(field)
                        if isinstance(value, list):
                            list_fields.add(field)
            
            # Detect custom fields
            standard_fields = {'original_name', 'name', 'gender', 'title', 'group_affiliation', 
                              'traits', 'how_they_refer_to_others', 'locations'}
            custom_fields = all_fields - standard_fields
            
            # Display current glossary stats
            stats_frame = tk.LabelFrame(main_frame, text="Current Glossary Statistics", padx=15, pady=10)
            stats_frame.pack(fill=tk.X, pady=(0, 15), padx=20)
            
            entry_count = len(self.current_glossary_data) if self.current_glossary_format == 'list' else len(self.current_glossary_data.get('entries', {}))
            tk.Label(stats_frame, text=f"Total entries: {entry_count}", font=('TkDefaultFont', 10)).pack(anchor=tk.W)
            tk.Label(stats_frame, text=f"Total fields detected: {len(all_fields)}", font=('TkDefaultFont', 10)).pack(anchor=tk.W)
            if custom_fields:
                tk.Label(stats_frame, text=f"Custom fields: {', '.join(sorted(custom_fields))}", 
                        font=('TkDefaultFont', 10), fg='blue').pack(anchor=tk.W)
            
            # Entry limit section
            limit_frame = tk.LabelFrame(main_frame, text="Entry Limit", padx=15, pady=10)
            limit_frame.pack(fill=tk.X, pady=(0, 15), padx=20)
            
            tk.Label(limit_frame, text="Limit the total number of glossary entries to reduce size and improve performance",
                    font=('TkDefaultFont', 9), fg='gray', wraplength=520).pack(anchor=tk.W, pady=(0, 10))
            
            top_frame = tk.Frame(limit_frame)
            top_frame.pack(fill=tk.X, pady=5)
            tk.Label(top_frame, text="Keep top").pack(side=tk.LEFT)
            top_var = tk.StringVar(value=str(min(100, entry_count)))
            tb.Entry(top_frame, textvariable=top_var, width=10).pack(side=tk.LEFT, padx=5)
            tk.Label(top_frame, text=f"entries (out of {entry_count})").pack(side=tk.LEFT)
            
            # Field-specific limits section
            field_limit_frame = tk.LabelFrame(main_frame, text="List Field Limits", padx=15, pady=10)
            field_limit_frame.pack(fill=tk.X, pady=(0, 15), padx=20)
            
            tk.Label(field_limit_frame, text="Limit the number of items in list fields to reduce redundancy",
                    font=('TkDefaultFont', 9), fg='gray', wraplength=520).pack(anchor=tk.W, pady=(0, 10))
            
            field_vars = {}
            
            # Standard list fields
            if any(field in list_fields for field in standard_list_fields):
                tk.Label(field_limit_frame, text="Standard Fields:", 
                        font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(5, 5))
                
                standard_field_defaults = {
                    'traits': ("Character traits", "5"),
                    'locations': ("Associated locations", "10"),
                    'group_affiliation': ("Group memberships", "3")
                }
                
                for field, (description, default) in standard_field_defaults.items():
                    if field in list_fields:
                        frame = tk.Frame(field_limit_frame)
                        frame.pack(fill=tk.X, pady=2, padx=(20, 0))
                        tk.Label(frame, text=f"{description}:", width=25, anchor=tk.W).pack(side=tk.LEFT)
                        var = tk.StringVar(value=default)
                        tb.Entry(frame, textvariable=var, width=10).pack(side=tk.LEFT, padx=5)
                        tk.Label(frame, text="items max", font=('TkDefaultFont', 9), fg='gray').pack(side=tk.LEFT)
                        field_vars[field] = var
            
            # Custom list fields
            custom_list_fields = [f for f in custom_fields if f in list_fields]
            if custom_list_fields:
                tk.Label(field_limit_frame, text="Custom Fields:", 
                        font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))
                
                for field in sorted(custom_list_fields):
                    frame = tk.Frame(field_limit_frame)
                    frame.pack(fill=tk.X, pady=2, padx=(20, 0))
                    tk.Label(frame, text=f"{field}:", width=25, anchor=tk.W).pack(side=tk.LEFT)
                    var = tk.StringVar(value="5")  # Default limit for custom fields
                    tb.Entry(frame, textvariable=var, width=10).pack(side=tk.LEFT, padx=5)
                    tk.Label(frame, text="items max", font=('TkDefaultFont', 9), fg='gray').pack(side=tk.LEFT)
                    field_vars[field] = var
            
            if not field_vars:
                tk.Label(field_limit_frame, text="No list fields detected in the glossary",
                        font=('TkDefaultFont', 10, 'italic'), fg='gray').pack(pady=10)
            
            # Remove fields section
            remove_frame = tk.LabelFrame(main_frame, text="Remove Fields", padx=15, pady=10)
            remove_frame.pack(fill=tk.X, pady=(0, 15), padx=20)
            
            tk.Label(remove_frame, text="Remove entire fields from all entries to reduce glossary size",
                    font=('TkDefaultFont', 9), fg='gray', wraplength=520).pack(anchor=tk.W, pady=(0, 10))
            
            remove_vars = {}
            
            # Group fields by type for better organization
            if all_fields:
                # Standard fields that are commonly removed
                removable_standard = ['title', 'how_they_refer_to_others', 'gender']
                existing_removable = [f for f in removable_standard if f in all_fields]
                
                if existing_removable:
                    tk.Label(remove_frame, text="Commonly Removed Fields:", 
                            font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(5, 5))
                    
                    for field in existing_removable:
                        var = tk.BooleanVar(value=False)
                        cb = tb.Checkbutton(remove_frame, text=f"Remove {field.replace('_', ' ')}", 
                                          variable=var)
                        cb.pack(anchor=tk.W, padx=20, pady=1)
                        remove_vars[field] = var
                
                # Other standard fields
                other_standard = [f for f in all_fields if f in standard_fields and f not in removable_standard]
                if other_standard:
                    tk.Label(remove_frame, text="Other Standard Fields:", 
                            font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))
                    
                    for field in sorted(other_standard):
                        var = tk.BooleanVar(value=False)
                        cb = tb.Checkbutton(remove_frame, text=f"Remove {field.replace('_', ' ')}", 
                                          variable=var)
                        cb.pack(anchor=tk.W, padx=20, pady=1)
                        remove_vars[field] = var
                
                # Custom fields
                if custom_fields:
                    tk.Label(remove_frame, text="Custom Fields:", 
                            font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))
                    
                    for field in sorted(custom_fields):
                        var = tk.BooleanVar(value=False)
                        cb = tb.Checkbutton(remove_frame, text=f"Remove {field}", 
                                          variable=var)
                        cb.pack(anchor=tk.W, padx=20, pady=1)
                        remove_vars[field] = var
            
            # Preview section
            preview_frame = tk.LabelFrame(main_frame, text="Preview", padx=15, pady=10)
            preview_frame.pack(fill=tk.X, pady=(0, 15), padx=20)
            
            preview_label = tk.Label(preview_frame, text="Click 'Preview Changes' to see the effect of your settings",
                                   font=('TkDefaultFont', 10), fg='gray')
            preview_label.pack(pady=5)
            
            def preview_changes():
                try:
                    # Count changes
                    top_n = int(top_var.get())
                    entries_to_remove = max(0, entry_count - top_n)
                    
                    fields_to_remove = sum(1 for var in remove_vars.values() if var.get())
                    
                    # Calculate approximate size reduction
                    items_to_trim = 0
                    for field, var in field_vars.items():
                        limit = int(var.get())
                        if self.current_glossary_format == 'list':
                            for entry in self.current_glossary_data[:top_n]:
                                if field in entry and isinstance(entry[field], list):
                                    items_to_trim += max(0, len(entry[field]) - limit)
                        
                    preview_text = f"Preview of changes:\n"
                    preview_text += f"• Entries: {entry_count} → {top_n} ({entries_to_remove} removed)\n"
                    if fields_to_remove > 0:
                        preview_text += f"• Fields to remove: {fields_to_remove}\n"
                    if items_to_trim > 0:
                        preview_text += f"• List items to trim: ~{items_to_trim}\n"
                    
                    preview_label.config(text=preview_text, fg='blue')
                    
                except ValueError:
                    preview_label.config(text="Please enter valid numbers", fg='red')
            
            tb.Button(preview_frame, text="Preview Changes", command=preview_changes,
                     bootstyle="info").pack()
            
            # Action buttons
            button_frame = tk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=(10, 20), padx=20)
            
            def apply_smart_trim():
                try:
                    top_n = int(top_var.get())
                    
                    if self.current_glossary_format == 'list':
                        # Keep only top N entries
                        if top_n < len(self.current_glossary_data):
                            self.current_glossary_data = self.current_glossary_data[:top_n]
                        
                        # Apply field limits
                        for entry in self.current_glossary_data:
                            # Limit list fields
                            for field, var in field_vars.items():
                                if field in entry and isinstance(entry[field], list):
                                    limit = int(var.get())
                                    if len(entry[field]) > limit:
                                        entry[field] = entry[field][:limit]
                            
                            # Remove selected fields
                            for field, var in remove_vars.items():
                                if var.get() and field in entry:
                                    entry.pop(field)
                    
                    elif self.current_glossary_format == 'dict':
                        # For dict format, only support entry limit
                        entries = list(self.current_glossary_data['entries'].items())
                        if top_n < len(entries):
                            self.current_glossary_data['entries'] = dict(entries[:top_n])
                        
                        # Apply field operations to dict entries
                        for key, entry in self.current_glossary_data['entries'].items():
                            # Limit list fields
                            for field, var in field_vars.items():
                                if field in entry and isinstance(entry[field], list):
                                    limit = int(var.get())
                                    if len(entry[field]) > limit:
                                        entry[field] = entry[field][:limit]
                            
                            # Remove selected fields
                            for field, var in remove_vars.items():
                                if var.get() and field in entry:
                                    entry.pop(field)
                    
                    if save_current_glossary():
                        load_glossary_for_editing()
                        
                        # Generate summary of changes
                        summary = "Smart trim applied successfully!\n\n"
                        summary += f"• Kept top {top_n} entries\n"
                        
                        if field_vars:
                            summary += "• Applied list limits:\n"
                            for field, var in field_vars.items():
                                summary += f"  - {field}: max {var.get()} items\n"
                        
                        removed_fields = [field for field, var in remove_vars.items() if var.get()]
                        if removed_fields:
                            summary += f"• Removed fields: {', '.join(removed_fields)}\n"
                        
                        messagebox.showinfo("Success", summary)
                        dialog.destroy()
                        
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid numbers")
            
            # Create inner frame for buttons
            button_inner_frame = tk.Frame(button_frame)
            button_inner_frame.pack()  # This centers it

            # Now pack the buttons in the inner frame
            tb.Button(button_inner_frame, text="Apply Trim", command=apply_smart_trim,
                     bootstyle="success", width=15).pack(side=tk.LEFT, padx=5)
            tb.Button(button_inner_frame, text="Cancel", command=dialog.destroy,
                     bootstyle="secondary", width=15).pack(side=tk.LEFT, padx=5)
            
            # Info section at bottom
            info_frame = tk.Frame(main_frame)
            info_frame.pack(fill=tk.X, pady=(0, 20), padx=20)
            
            tk.Label(info_frame, text="💡 Tip: Always backup your glossary before applying major changes!",
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
                height=None,  # Will use default height calculation
                max_width_ratio=0.9,
                max_height_ratio=0.85
            )
            
            main_frame = scrollable_frame
            
            # Title and description
            tk.Label(main_frame, text="Filter Glossary Entries", 
                    font=('TkDefaultFont', 14, 'bold')).pack(pady=(20, 5))
            
            tk.Label(main_frame, text="Create complex filters to find specific entries in your glossary",
                    font=('TkDefaultFont', 10), fg='gray', wraplength=550).pack(pady=(0, 15))
            
            # Analyze current glossary to detect all fields
            all_fields = set()
            field_types = {}  # Track field types: 'text', 'list', 'dict'
            field_samples = {}  # Store sample values for each field
            
            if self.current_glossary_format == 'list':
                for entry in self.current_glossary_data:
                    for field, value in entry.items():
                        all_fields.add(field)
                        if isinstance(value, list):
                            field_types[field] = 'list'
                        elif isinstance(value, dict):
                            field_types[field] = 'dict'
                        else:
                            field_types[field] = 'text'
                        
                        # Store sample values for text fields
                        if field_types[field] == 'text' and value and field not in field_samples:
                            field_samples[field] = str(value)[:50]  # First 50 chars
            elif self.current_glossary_format == 'dict':
                for key, entry in self.current_glossary_data.get('entries', {}).items():
                    for field, value in entry.items():
                        all_fields.add(field)
                        if isinstance(value, list):
                            field_types[field] = 'list'
                        elif isinstance(value, dict):
                            field_types[field] = 'dict'
                        else:
                            field_types[field] = 'text'
                        
                        if field_types[field] == 'text' and value and field not in field_samples:
                            field_samples[field] = str(value)[:50]
            
            # Separate standard and custom fields
            standard_fields = {'original_name', 'name', 'gender', 'title', 'group_affiliation', 
                              'traits', 'how_they_refer_to_others', 'locations'}
            custom_fields = all_fields - standard_fields
            
            # Current stats
            entry_count = len(self.current_glossary_data) if self.current_glossary_format == 'list' else len(self.current_glossary_data.get('entries', {}))
            
            stats_frame = tk.LabelFrame(main_frame, text="Current Status", padx=15, pady=10)
            stats_frame.pack(fill=tk.X, pady=(0, 15), padx=20)
            tk.Label(stats_frame, text=f"Total entries: {entry_count}", font=('TkDefaultFont', 10)).pack(anchor=tk.W)
            tk.Label(stats_frame, text=f"Available fields: {len(all_fields)}", font=('TkDefaultFont', 10)).pack(anchor=tk.W)
            
            # Filter type selection
            filter_type_frame = tk.LabelFrame(main_frame, text="Filter Mode", padx=15, pady=10)
            filter_type_frame.pack(fill=tk.X, pady=(0, 15), padx=20)
            
            filter_mode = tk.StringVar(value="all")
            tk.Radiobutton(filter_type_frame, text="Match ALL conditions (AND)", 
                          variable=filter_mode, value="all").pack(anchor=tk.W)
            tk.Radiobutton(filter_type_frame, text="Match ANY condition (OR)", 
                          variable=filter_mode, value="any").pack(anchor=tk.W)
            
            # Filter conditions
            conditions_frame = tk.LabelFrame(main_frame, text="Filter Conditions", padx=15, pady=10)
            conditions_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15), padx=20)
            
            tk.Label(conditions_frame, text="Select which entries to keep based on field conditions:",
                    font=('TkDefaultFont', 9), fg='gray').pack(anchor=tk.W, pady=(0, 10))
            
            # Store filter conditions
            filter_conditions = []
            
            # Presence filters (check if field exists and has content)
            presence_frame = tk.LabelFrame(conditions_frame, text="Field Presence Filters", padx=10, pady=10)
            presence_frame.pack(fill=tk.X, pady=(0, 10))
            
            presence_vars = {}
            
            # Common presence checks
            common_checks = [
                ('has_name', "Has name or original_name", True),
                ('has_translation', "Has English translation (name field)", False),
                ('has_both_names', "Has BOTH original_name AND name", False),
            ]
            
            for key, label, default in common_checks:
                var = tk.BooleanVar(value=default)
                tb.Checkbutton(presence_frame, text=label, variable=var).pack(anchor=tk.W)
                presence_vars[key] = var
            
            # Field-specific presence checks
            if all_fields:
                tk.Label(presence_frame, text="Must have these fields:", 
                        font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))
                
                # Group by standard/custom
                if standard_fields.intersection(all_fields):
                    tk.Label(presence_frame, text="Standard Fields:", 
                            font=('TkDefaultFont', 9, 'italic')).pack(anchor=tk.W, padx=20)
                    
                    field_presence_frame = tk.Frame(presence_frame)
                    field_presence_frame.pack(fill=tk.X, padx=20)
                    
                    col = 0
                    row = 0
                    for field in sorted(standard_fields.intersection(all_fields)):
                        if field not in ['name', 'original_name']:  # Already handled above
                            var = tk.BooleanVar(value=False)
                            cb = tb.Checkbutton(field_presence_frame, text=field.replace('_', ' '), 
                                              variable=var)
                            cb.grid(row=row, column=col, sticky=tk.W, padx=5, pady=1)
                            presence_vars[f'has_{field}'] = var
                            
                            col += 1
                            if col > 2:
                                col = 0
                                row += 1
                
                if custom_fields:
                    tk.Label(presence_frame, text="Custom Fields:", 
                            font=('TkDefaultFont', 9, 'italic')).pack(anchor=tk.W, padx=20, pady=(5, 0))
                    
                    custom_presence_frame = tk.Frame(presence_frame)
                    custom_presence_frame.pack(fill=tk.X, padx=20)
                    
                    col = 0
                    row = 0
                    for field in sorted(custom_fields):
                        var = tk.BooleanVar(value=False)
                        cb = tb.Checkbutton(custom_presence_frame, text=field, variable=var)
                        cb.grid(row=row, column=col, sticky=tk.W, padx=5, pady=1)
                        presence_vars[f'has_{field}'] = var
                        
                        col += 1
                        if col > 2:
                            col = 0
                            row += 1
            
            # Text content filters
            text_filter_frame = tk.LabelFrame(conditions_frame, text="Text Content Filters", padx=10, pady=10)
            text_filter_frame.pack(fill=tk.X, pady=(0, 10))
            
            tk.Label(text_filter_frame, text="Filter by text content (case-insensitive):",
                    font=('TkDefaultFont', 9), fg='gray').pack(anchor=tk.W, pady=(0, 5))
            
            text_filters = {}
            text_fields = [f for f in all_fields if field_types.get(f) == 'text']
            
            if text_fields:
                for field in sorted(text_fields):
                    frame = tk.Frame(text_filter_frame)
                    frame.pack(fill=tk.X, pady=2)
                    
                    tk.Label(frame, text=f"{field}:", width=20, anchor=tk.W).pack(side=tk.LEFT)
                    
                    contains_var = tk.StringVar()
                    entry = tb.Entry(frame, textvariable=contains_var, width=30)
                    entry.pack(side=tk.LEFT, padx=5)
                    
                    if field in field_samples:
                        tk.Label(frame, text=f"(e.g., {field_samples[field][:20]}...)", 
                                font=('TkDefaultFont', 8), fg='gray').pack(side=tk.LEFT)
                    
                    text_filters[field] = contains_var
            else:
                tk.Label(text_filter_frame, text="No text fields found",
                        font=('TkDefaultFont', 10, 'italic'), fg='gray').pack(pady=10)
            
            # List size filters
            list_filter_frame = tk.LabelFrame(conditions_frame, text="List Size Filters", padx=10, pady=10)
            list_filter_frame.pack(fill=tk.X, pady=(0, 10))
            
            list_size_filters = {}
            list_fields = [f for f in all_fields if field_types.get(f) == 'list']
            
            if list_fields:
                tk.Label(list_filter_frame, text="Filter by number of items in list fields:",
                        font=('TkDefaultFont', 9), fg='gray').pack(anchor=tk.W, pady=(0, 5))
                
                for field in sorted(list_fields):
                    frame = tk.Frame(list_filter_frame)
                    frame.pack(fill=tk.X, pady=2)
                    
                    tk.Label(frame, text=f"{field}:", width=20, anchor=tk.W).pack(side=tk.LEFT)
                    tk.Label(frame, text="Min:").pack(side=tk.LEFT)
                    
                    min_var = tk.StringVar()
                    tb.Entry(frame, textvariable=min_var, width=5).pack(side=tk.LEFT, padx=2)
                    
                    tk.Label(frame, text="Max:").pack(side=tk.LEFT, padx=(10, 0))
                    
                    max_var = tk.StringVar()
                    tb.Entry(frame, textvariable=max_var, width=5).pack(side=tk.LEFT, padx=2)
                    
                    tk.Label(frame, text="items", font=('TkDefaultFont', 9), fg='gray').pack(side=tk.LEFT, padx=5)
                    
                    list_size_filters[field] = (min_var, max_var)
            else:
                tk.Label(list_filter_frame, text="No list fields found",
                        font=('TkDefaultFont', 10, 'italic'), fg='gray').pack(pady=10)
            
            # Preview section
            preview_frame = tk.LabelFrame(main_frame, text="Preview", padx=15, pady=10)
            preview_frame.pack(fill=tk.X, pady=(0, 15), padx=20)
            
            preview_label = tk.Label(preview_frame, text="Click 'Preview Filter' to see how many entries match",
                                   font=('TkDefaultFont', 10), fg='gray')
            preview_label.pack(pady=5)
            
            def check_entry_matches(entry):
                """Check if an entry matches the filter conditions"""
                matches = []
                
                # Check presence conditions
                if presence_vars['has_name'].get():
                    matches.append(bool(entry.get('name') or entry.get('original_name')))
                
                if presence_vars['has_translation'].get():
                    matches.append(bool(entry.get('name')))
                
                if presence_vars['has_both_names'].get():
                    matches.append(bool(entry.get('name') and entry.get('original_name')))
                
                # Check field-specific presence
                for key, var in presence_vars.items():
                    if key.startswith('has_') and key not in ['has_name', 'has_translation', 'has_both_names']:
                        if var.get():
                            field = key[4:]  # Remove 'has_' prefix
                            matches.append(bool(entry.get(field)))
                
                # Check text content filters
                for field, var in text_filters.items():
                    search_text = var.get().strip().lower()
                    if search_text:
                        field_value = str(entry.get(field, '')).lower()
                        matches.append(search_text in field_value)
                
                # Check list size filters
                for field, (min_var, max_var) in list_size_filters.items():
                    try:
                        field_list = entry.get(field, [])
                        if isinstance(field_list, list):
                            list_len = len(field_list)
                            
                            if min_var.get():
                                min_size = int(min_var.get())
                                matches.append(list_len >= min_size)
                            
                            if max_var.get():
                                max_size = int(max_var.get())
                                matches.append(list_len <= max_size)
                    except ValueError:
                        pass
                
                # Apply filter mode
                if not matches:
                    return True  # No conditions set, keep all
                
                if filter_mode.get() == "all":
                    return all(matches)
                else:  # "any"
                    return any(matches)
            
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
                    self.current_glossary_data[:] = filtered
                    
                    if save_current_glossary():
                        load_glossary_for_editing()
                        messagebox.showinfo("Success", 
                            f"Filter applied!\n\nKept: {len(filtered)} entries\nRemoved: {removed} entries")
                        dialog.destroy()
                
                elif self.current_glossary_format == 'dict':
                    filtered_entries = {}
                    for key, entry in self.current_glossary_data.get('entries', {}).items():
                        if check_entry_matches(entry):
                            filtered_entries[key] = entry
                    
                    removed = len(self.current_glossary_data.get('entries', {})) - len(filtered_entries)
                    self.current_glossary_data['entries'] = filtered_entries
                    
                    if save_current_glossary():
                        load_glossary_for_editing()
                        messagebox.showinfo("Success", 
                            f"Filter applied!\n\nKept: {len(filtered_entries)} entries\nRemoved: {removed} entries")
                        dialog.destroy()
            
            def invert_filter():
                """Apply inverted filter (remove matching entries instead of keeping them)"""
                if messagebox.askyesno("Invert Filter", 
                                      "This will REMOVE matching entries instead of keeping them. Continue?"):
                    if self.current_glossary_format == 'list':
                        filtered = []
                        for entry in self.current_glossary_data:
                            if not check_entry_matches(entry):  # Inverted logic
                                filtered.append(entry)
                        
                        removed = len(self.current_glossary_data) - len(filtered)
                        self.current_glossary_data[:] = filtered
                        
                        if save_current_glossary():
                            load_glossary_for_editing()
                            messagebox.showinfo("Success", 
                                f"Inverted filter applied!\n\nKept: {len(filtered)} entries\nRemoved: {removed} entries")
                            dialog.destroy()
            
            # Create inner frame for buttons
            button_inner_frame = tk.Frame(button_frame)
            button_inner_frame.pack()  # This centers it

            # Now pack the buttons in the inner frame
            tb.Button(button_inner_frame, text="Apply Filter", command=apply_filter,
                     bootstyle="success", width=15).pack(side=tk.LEFT, padx=5)
            tb.Button(button_inner_frame, text="Invert Filter", command=invert_filter,
                     bootstyle="warning", width=15).pack(side=tk.LEFT, padx=5)
            tb.Button(button_inner_frame, text="Cancel", command=dialog.destroy,
                     bootstyle="secondary", width=15).pack(side=tk.LEFT, padx=5)
            
            # Info section
            info_frame = tk.Frame(main_frame)
            info_frame.pack(fill=tk.X, pady=(0, 20), padx=20)
            
            tk.Label(info_frame, text="💡 Tip: Use text filters to find specific characters or content patterns",
                    font=('TkDefaultFont', 9, 'italic'), fg='#666').pack()
            
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
       
       dialog = self.wm.create_simple_dialog(
           self.master,
           f"Edit {col_name.replace('_', ' ').title()}",
           width=400,
           height=150
       )
       
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
       
       dialog.deiconify()

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
               
               # Validate glossary path before passing to backend
               if hasattr(self, 'manual_glossary_path') and self.manual_glossary_path:
                   # If this is an auto-loaded glossary, check if it matches the current file
                   if (hasattr(self, 'auto_loaded_glossary_path') and 
                       self.manual_glossary_path == self.auto_loaded_glossary_path):
                       # This is an auto-loaded glossary
                       if (hasattr(self, 'auto_loaded_glossary_for_file') and 
                           hasattr(self, 'file_path') and 
                           self.file_path == self.auto_loaded_glossary_for_file):
                           # The glossary matches the current file
                           os.environ['MANUAL_GLOSSARY'] = self.manual_glossary_path
                           self.append_log(f"📑 Using auto-loaded glossary: {os.path.basename(self.manual_glossary_path)}")
                       else:
                           # The glossary is from a different file, don't use it
                           self.append_log("📑 Skipping auto-loaded glossary (different novel)")
                   else:
                       # This is a manually loaded glossary, always use it
                       os.environ['MANUAL_GLOSSARY'] = self.manual_glossary_path
                       self.append_log(f"📑 Using manual glossary: {os.path.basename(self.manual_glossary_path)}")
               
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
           'APPEND_GLOSSARY_PROMPT': self.append_glossary_prompt,
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
           'IMAGE_MAX_TOKENS': str(self.max_output_tokens),
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
           'CHAPTER_NUMBER_OFFSET': str(self.chapter_number_offset_var.get()), 
           'USE_HEADER_AS_OUTPUT': "1" if self.use_header_as_output_var.get() else "0",
           'ENABLE_DECIMAL_CHAPTERS': "1" if self.enable_decimal_chapters_var.get() else "0",
           'ENABLE_WATERMARK_REMOVAL': "1" if self.enable_watermark_removal_var.get() else "0",
           'ADVANCED_WATERMARK_REMOVAL': "1" if self.advanced_watermark_removal_var.get() else "0",
           'SAVE_CLEANED_IMAGES': "1" if self.save_cleaned_images_var.get() else "0",
           'COMPRESSION_FACTOR': self.compression_factor_var.get()
           
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
                'check_encoding_issues': True,
                'check_repetition': True,
                'check_translation_artifacts': True,
                'min_file_length': 100,
                'report_format': 'detailed',
                'auto_save_report': True
            })
            
            # ALWAYS show mode selection dialog with settings
            mode_dialog = self.wm.create_simple_dialog(
                self.master,
                "Select QA Scanner Mode",
                width=1500,  # Optimal width for 4 cards
                height=650,  # Compact height to ensure buttons are visible
                hide_initially=True
            )
            
            # Set minimum size to prevent dialog from being too small
            mode_dialog.minsize(1200, 600)
            
            # Variables
            selected_mode_value = None
            
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
            
            # Create mode cards
            for idx, mode in enumerate(mode_data):
                # Main card frame with initial background
                card = tk.Frame(modes_container, 
                               bg=mode["bg_color"],
                               highlightbackground=mode["border_color"],
                               highlightthickness=2,
                               relief='flat')
                card.grid(row=0, column=idx, padx=10, pady=5, sticky='nsew')  # Minimal padding
                modes_container.columnconfigure(idx, weight=1)
                
                # Configure row to not expand too much
                modes_container.rowconfigure(0, weight=0)  # Don't expand row vertically
                
                # Content frame
                content_frame = tk.Frame(card, bg=mode["bg_color"], cursor='hand2')
                content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)  # Further reduced padding
                
                # Emoji
                emoji_label = tk.Label(content_frame, text=mode["emoji"], 
                                      font=('Arial', 48), bg=mode["bg_color"])  # Further reduced
                emoji_label.pack(pady=(0, 5))  # Minimal padding
                
                # Title
                title_label = tk.Label(content_frame, text=mode["title"], 
                                      font=('Arial', 24, 'bold'),  # Further reduced
                                      fg='white', bg=mode["bg_color"])
                title_label.pack()
                
                # Subtitle
                tk.Label(content_frame, text=mode["subtitle"], 
                        font=('Arial', 14), fg=mode["accent_color"],  # Further reduced
                        bg=mode["bg_color"]).pack(pady=(3, 10))  # Minimal padding
                
                # Features
                features_frame = tk.Frame(content_frame, bg=mode["bg_color"])
                features_frame.pack(fill=tk.X)
                
                for feature in mode["features"]:
                    feature_label = tk.Label(features_frame, text=feature, 
                                           font=('Arial', 11), fg='#e0e0e0',  # Further reduced from 12
                                           bg=mode["bg_color"], justify=tk.LEFT)
                    feature_label.pack(anchor=tk.W, pady=1)  # Minimal padding
                
                # Recommendation badge if present
                if mode["recommendation"]:
                    rec_frame = tk.Frame(content_frame, bg=mode["accent_color"])
                    rec_frame.pack(pady=(10, 0), fill=tk.X)  # Further reduced
                    
                    rec_label = tk.Label(rec_frame, text=mode["recommendation"],
                                       font=('Arial', 11, 'bold'),  # Further reduced
                                       fg='white', bg=mode["accent_color"],
                                       padx=8, pady=4)  # Minimal padding
                    rec_label.pack()
                
                # Click handler
                def make_click_handler(mode_value):
                    def handler(event=None):
                        nonlocal selected_mode_value
                        selected_mode_value = mode_value
                        mode_dialog.destroy()
                    return handler
                
                click_handler = make_click_handler(mode["value"])
                
                # Hover effects function
                def create_hover_handlers(card_widget, content_widget, mode_info, all_widgets):
                    def on_enter(event=None):
                        # Apply hover color to ALL widgets
                        for widget in all_widgets:
                            try:
                                if hasattr(widget, 'config'):
                                    widget.config(bg=mode_info["hover_color"])
                            except:
                                pass
                    
                    def on_leave(event=None):
                        # Restore original color to ALL widgets
                        for widget in all_widgets:
                            try:
                                if hasattr(widget, 'config'):
                                    widget.config(bg=mode_info["bg_color"])
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
                                                padding=20, bootstyle="secondary")
                threshold_frame.pack(fill='x', padx=20, pady=(0, 15))
                
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
                    # Pass the QA settings to scan_html_folder
                    scan_html_folder(
                        folder_path, 
                        log=self.append_log, 
                        stop_flag=lambda: self.stop_requested, 
                        mode=mode,
                        qa_settings=qa_settings  # Pass settings to the scanner
                    )
                    self.append_log("✅ QA scan completed successfully.")
                except Exception as e:
                    self.append_log(f"❌ QA scan error: {e}")
                    self.append_log(f"Traceback: {traceback.format_exc()}")
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
        check_encoding_var = tk.BooleanVar(value=qa_settings.get('check_encoding_issues', True))
        check_repetition_var = tk.BooleanVar(value=qa_settings.get('check_repetition', True))
        check_artifacts_var = tk.BooleanVar(value=qa_settings.get('check_translation_artifacts', True))
        
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
        
        min_length_var = tk.IntVar(value=qa_settings.get('min_file_length', 100))
        min_length_spinbox = tb.Spinbox(
            min_length_frame,
            from_=10,
            to=10000,
            textvariable=min_length_var,
            width=10,
            bootstyle="primary"
        )
        min_length_spinbox.pack(side=tk.LEFT, padx=(10, 0))
        
        # Report Settings Section
        report_section = tk.LabelFrame(
            main_frame,
            text="Report Settings",
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=15
        )
        report_section.pack(fill=tk.X, pady=(0, 20))
        
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
            qa_settings['foreign_char_threshold'] = threshold_var.get()
            qa_settings['excluded_characters'] = excluded_text.get(1.0, tk.END).strip()
            qa_settings['check_encoding_issues'] = check_encoding_var.get()
            qa_settings['check_repetition'] = check_repetition_var.get()
            qa_settings['check_translation_artifacts'] = check_artifacts_var.get()
            qa_settings['min_file_length'] = min_length_var.get()
            qa_settings['report_format'] = format_var.get()
            qa_settings['auto_save_report'] = auto_save_var.get()
            
            # Save to main config
            self.config['qa_scanner_settings'] = qa_settings
            self.save_config()
            
            self.append_log("✅ QA Scanner settings saved")
            dialog._cleanup_scrolling()  # Clean up scrolling bindings
            dialog.destroy()
        
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
                check_encoding_var.set(True)
                check_repetition_var.set(True)
                check_artifacts_var.set(True)
                min_length_var.set(100)
                format_var.set('detailed')
                auto_save_var.set(True)
        
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
       if qa_running:
           self.qa_button.config(text="Stop Scan", command=self.stop_qa_scan, 
                                 bootstyle="danger", state=tk.NORMAL)
       else:
           self.qa_button.config(text="QA Scan", command=self.run_qa_scan, 
                                 bootstyle="warning", state=tk.NORMAL if scan_html_folder and not any_process_running else tk.DISABLED)   

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
        # Clear previous auto-loaded glossary if switching files
        if file_path != self.auto_loaded_glossary_for_file:
            if self.auto_loaded_glossary_path and self.manual_glossary_path == self.auto_loaded_glossary_path:
                self.manual_glossary_path = None
                self.append_log("📑 Cleared auto-loaded glossary from previous novel")
            self.auto_loaded_glossary_path = None
            self.auto_loaded_glossary_for_file = None
        
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
                        self.auto_loaded_glossary_path = glossary_path
                        self.auto_loaded_glossary_for_file = file_path
                        self.append_log(f"📑 Auto-loaded glossary for {file_base}: {os.path.basename(glossary_path)}")
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
            
            # Store the selected file path for tracking
            self.file_path = path
            
            # Clear previous auto-loaded glossary if switching files
            if hasattr(self, 'auto_loaded_glossary_for_file') and path != self.auto_loaded_glossary_for_file:
                if hasattr(self, 'auto_loaded_glossary_path') and self.manual_glossary_path == self.auto_loaded_glossary_path:
                    self.manual_glossary_path = None
                    self.append_log("📑 Cleared auto-loaded glossary from previous file")
                self.auto_loaded_glossary_path = None
                self.auto_loaded_glossary_for_file = None
            
            # Auto-load glossary for epub files
            if path.lower().endswith('.epub'):
                self.auto_load_glossary_for_file(path)
            else:
                # For non-epub files, clear any auto-loaded glossary
                if hasattr(self, 'auto_loaded_glossary_path') and self.manual_glossary_path == self.auto_loaded_glossary_path:
                    self.manual_glossary_path = None
                    self.auto_loaded_glossary_path = None
                    self.auto_loaded_glossary_for_file = None
                    self.append_log("📑 Cleared auto-loaded glossary (non-EPUB file selected)")

    def toggle_api_visibility(self):
       show = self.api_key_entry.cget('show')
       self.api_key_entry.config(show='' if show == '*' else '*')
       
    def configure_translation_chunk_prompt(self):
        """Configure the prompt template for translation chunks"""
        dialog = self.wm.create_simple_dialog(
            self.master,
            "Configure Translation Chunk Prompt",
            width=700,
            height=None
        )
        
        main_frame = tk.Frame(dialog, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="Translation Chunk Prompt Template", 
                font=('TkDefaultFont', 14, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        tk.Label(main_frame, text="Configure how chunks are presented to the AI when chapters are split.",
                font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, pady=(0, 10))
        
        # Instructions
        instructions_frame = tk.LabelFrame(main_frame, text="Available Placeholders", padx=10, pady=10)
        instructions_frame.pack(fill=tk.X, pady=(0, 15))
        
        placeholders = [
            ("{chunk_idx}", "Current chunk number (1-based)"),
            ("{total_chunks}", "Total number of chunks"),
            ("{chunk_html}", "The actual HTML content to translate")
        ]
        
        for placeholder, desc in placeholders:
            placeholder_frame = tk.Frame(instructions_frame)
            placeholder_frame.pack(anchor=tk.W, pady=2)
            tk.Label(placeholder_frame, text=f"• {placeholder}:", font=('Courier', 10, 'bold')).pack(side=tk.LEFT)
            tk.Label(placeholder_frame, text=f" {desc}", font=('TkDefaultFont', 10)).pack(side=tk.LEFT)
        
        # Prompt input
        prompt_frame = tk.LabelFrame(main_frame, text="Chunk Prompt Template", padx=10, pady=10)
        prompt_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.chunk_prompt_text = self.ui.setup_scrollable_text(
            prompt_frame, height=8, wrap=tk.WORD
        )
        self.chunk_prompt_text.pack(fill=tk.BOTH, expand=True)
        self.chunk_prompt_text.insert('1.0', self.translation_chunk_prompt)
        
        # Example
        example_frame = tk.LabelFrame(main_frame, text="Example Output", padx=10, pady=10)
        example_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(example_frame, text="With chunk 2 of 5, the prompt would be:",
                font=('TkDefaultFont', 10)).pack(anchor=tk.W)
        
        self.example_label = tk.Label(example_frame, text="", 
                                     font=('Courier', 9), fg='blue', 
                                     wraplength=650, justify=tk.LEFT)
        self.example_label.pack(anchor=tk.W, pady=(5, 0))
        
        def update_example(*args):
            try:
                template = self.chunk_prompt_text.get('1.0', tk.END).strip()
                example = template.replace('{chunk_idx}', '2').replace('{total_chunks}', '5').replace('{chunk_html}', '<p>Chapter content here...</p>')
                self.example_label.config(text=example[:200] + "..." if len(example) > 200 else example)
            except:
                self.example_label.config(text="[Invalid template]")
        
        self.chunk_prompt_text.bind('<KeyRelease>', update_example)
        update_example()
        
        # Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        def save_chunk_prompt():
            self.translation_chunk_prompt = self.chunk_prompt_text.get('1.0', tk.END).strip()
            self.config['translation_chunk_prompt'] = self.translation_chunk_prompt
            messagebox.showinfo("Success", "Translation chunk prompt saved!")
            dialog.destroy()
        
        def reset_chunk_prompt():
            if messagebox.askyesno("Reset Prompt", "Reset to default chunk prompt?"):
                self.chunk_prompt_text.delete('1.0', tk.END)
                self.chunk_prompt_text.insert('1.0', self.default_translation_chunk_prompt)
                update_example()
        
        tb.Button(button_frame, text="Save", command=save_chunk_prompt, 
                 bootstyle="success", width=15).pack(side=tk.LEFT, padx=5)
        tb.Button(button_frame, text="Reset to Default", command=reset_chunk_prompt, 
                 bootstyle="warning", width=15).pack(side=tk.LEFT, padx=5)
        tb.Button(button_frame, text="Cancel", command=dialog.destroy, 
                 bootstyle="secondary", width=15).pack(side=tk.LEFT, padx=5)
        
        dialog.deiconify()

    def configure_image_chunk_prompt(self):
        """Configure the prompt template for image chunks"""
        dialog = self.wm.create_simple_dialog(
            self.master,
            "Configure Image Chunk Prompt",
            width=700,
            height=None
        )
        
        main_frame = tk.Frame(dialog, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="Image Chunk Context Template", 
                font=('TkDefaultFont', 14, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        tk.Label(main_frame, text="Configure the context provided when tall images are split into chunks.",
                font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, pady=(0, 10))
        
        # Instructions
        instructions_frame = tk.LabelFrame(main_frame, text="Available Placeholders", padx=10, pady=10)
        instructions_frame.pack(fill=tk.X, pady=(0, 15))
        
        placeholders = [
            ("{chunk_idx}", "Current chunk number (1-based)"),
            ("{total_chunks}", "Total number of chunks"),
            ("{context}", "Additional context (e.g., chapter info)")
        ]
        
        for placeholder, desc in placeholders:
            placeholder_frame = tk.Frame(instructions_frame)
            placeholder_frame.pack(anchor=tk.W, pady=2)
            tk.Label(placeholder_frame, text=f"• {placeholder}:", font=('Courier', 10, 'bold')).pack(side=tk.LEFT)
            tk.Label(placeholder_frame, text=f" {desc}", font=('TkDefaultFont', 10)).pack(side=tk.LEFT)
        
        # Prompt input
        prompt_frame = tk.LabelFrame(main_frame, text="Image Chunk Prompt Template", padx=10, pady=10)
        prompt_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.image_chunk_prompt_text = self.ui.setup_scrollable_text(
            prompt_frame, height=8, wrap=tk.WORD
        )
        self.image_chunk_prompt_text.pack(fill=tk.BOTH, expand=True)
        self.image_chunk_prompt_text.insert('1.0', self.image_chunk_prompt)
        
        # Example
        example_frame = tk.LabelFrame(main_frame, text="Example Output", padx=10, pady=10)
        example_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(example_frame, text="With chunk 3 of 7 and chapter context, the prompt would be:",
                font=('TkDefaultFont', 10)).pack(anchor=tk.W)
        
        self.image_example_label = tk.Label(example_frame, text="", 
                                           font=('Courier', 9), fg='blue', 
                                           wraplength=650, justify=tk.LEFT)
        self.image_example_label.pack(anchor=tk.W, pady=(5, 0))
        
        def update_image_example(*args):
            try:
                template = self.image_chunk_prompt_text.get('1.0', tk.END).strip()
                example = template.replace('{chunk_idx}', '3').replace('{total_chunks}', '7').replace('{context}', 'Chapter 5: The Great Battle')
                self.image_example_label.config(text=example)
            except:
                self.image_example_label.config(text="[Invalid template]")
        
        self.image_chunk_prompt_text.bind('<KeyRelease>', update_image_example)
        update_image_example()
        
        # Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        def save_image_chunk_prompt():
            self.image_chunk_prompt = self.image_chunk_prompt_text.get('1.0', tk.END).strip()
            self.config['image_chunk_prompt'] = self.image_chunk_prompt
            messagebox.showinfo("Success", "Image chunk prompt saved!")
            dialog.destroy()
        
        def reset_image_chunk_prompt():
            if messagebox.askyesno("Reset Prompt", "Reset to default image chunk prompt?"):
                self.image_chunk_prompt_text.delete('1.0', tk.END)
                self.image_chunk_prompt_text.insert('1.0', self.default_image_chunk_prompt)
                update_image_example()
        
        tb.Button(button_frame, text="Save", command=save_image_chunk_prompt, 
                 bootstyle="success", width=15).pack(side=tk.LEFT, padx=5)
        tb.Button(button_frame, text="Reset to Default", command=reset_image_chunk_prompt, 
                 bootstyle="warning", width=15).pack(side=tk.LEFT, padx=5)
        tb.Button(button_frame, text="Cancel", command=dialog.destroy, 
                 bootstyle="secondary", width=15).pack(side=tk.LEFT, padx=5)
        
        dialog.deiconify()
    
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
       
       # Save & Close buttons
       self._create_settings_buttons(scrollable_frame, dialog, canvas)
       
       # Auto-resize and show
       self.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.9, max_height_ratio=1.6)
       
       dialog.protocol("WM_DELETE_WINDOW", lambda: [dialog._cleanup_scrolling(), dialog.destroy()])

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
        section_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=5)
        
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
        tk.Label(compression_frame, text="(0.7-1.0)").pack(side=tk.LEFT)
        
        tb.Button(compression_frame, text=" Chunk Prompt", 
                 command=self.configure_translation_chunk_prompt,
                 bootstyle="info-outline", width=15).pack(side=tk.LEFT, padx=(15, 0))

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
        
        tb.Button(title_frame, text="Configure Title Prompt", 
                 command=self.configure_title_prompt,
                 bootstyle="info-outline", width=20).pack(side=tk.LEFT, padx=(10, 0))
        
        tk.Label(section_frame, text="When enabled: Book titles will be translated to English\n"
                    "When disabled: Book titles remain in original language",
                    font=('TkDefaultFont', 11), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
            
        # EPUB Validation
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
        
        tk.Label(section_frame, text="Always use chapter ranges as specified\n(don't force adjust to chapter 1)",
                font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
                
        tb.Checkbutton(section_frame, text="Use Header as Output Name", 
                      variable=self.use_header_as_output_var,
                      bootstyle="round-toggle").pack(anchor=tk.W, pady=2)

        tk.Label(section_frame, text="Use chapter headers/titles as output filenames",
                font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
                

         # NEW: Chapter number offset
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
        
        # Right column - existing settings
        settings_frame = tk.Frame(right_column)
        settings_frame.pack(fill=tk.X)
        
        settings_frame.grid_columnconfigure(1, minsize=80)
        
        settings = [
            ("Min Image height (px):", self.webnovel_min_height_var),
            ("Max Images per chapter:", self.max_images_per_chapter_var),
            ("Chunk height:", self.image_chunk_height_var)
        ]
        
        for row, (label, var) in enumerate(settings):
            tk.Label(settings_frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
            tb.Entry(settings_frame, width=10, textvariable=var).grid(row=row, column=1, sticky=tk.W, pady=3)
        
        tb.Button(settings_frame, text="Image Chunk Prompt", 
                 command=self.configure_image_chunk_prompt,
                 bootstyle="info-outline", width=20).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
         
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
                    'chapter_number_offset': safe_int(self.chapter_number_offset_var.get(), 0),
                    'enable_decimal_chapters': self.enable_decimal_chapters_var.get(),
                    'use_header_as_output': self.use_header_as_output_var.get()
                })
                
                # Validate numeric fields
                numeric_fields = [
                    ('webnovel_min_height', self.webnovel_min_height_var, 1000),
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
                    "ENABLE_DECIMAL_CHAPTERS": "1" if self.enable_decimal_chapters_var.get() else "0",
                    'ENABLE_WATERMARK_REMOVAL': "1" if self.enable_watermark_removal_var.get() else "0",
                    'SAVE_CLEANED_IMAGES': "1" if self.save_cleaned_images_var.get() else "0",
                    'TRANSLATION_CHUNK_PROMPT': self.translation_chunk_prompt,
                    'IMAGE_CHUNK_PROMPT': self.image_chunk_prompt,
                }
                os.environ.update(env_updates)
                
                with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                
                self.append_log("✅ Other Settings saved successfully")
                dialog.destroy()
                
            except Exception as e:
                print(f"❌ Failed to save Other Settings: {e}")
                messagebox.showerror("Error", f"Failed to save settings: {e}")
        
        tb.Button(button_container, text="💾 Save Settings", command=save_and_close, 
                 bootstyle="success", width=20).pack(side=tk.LEFT, padx=5)
        
        tb.Button(button_container, text="❌ Cancel", command=lambda: [dialog._cleanup_scrolling(), dialog.destroy()], 
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
        
        # Clear auto-loaded tracking when manually loading
        self.auto_loaded_glossary_path = None
        self.auto_loaded_glossary_for_file = None
        
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
            self.config['max_images_per_chapter'] = safe_int(self.max_images_per_chapter_var.get(), 1)
            self.config['batch_translation'] = self.batch_translation_var.get()
            self.config['batch_size'] = safe_int(self.batch_size_var.get(), 3)
            self.config['translation_history_rolling'] = self.translation_history_rolling_var.get()
            self.config['glossary_history_rolling'] = self.glossary_history_rolling_var.get()
            self.config['disable_epub_gallery'] = self.disable_epub_gallery_var.get()
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
    
    print("🚀 Starting Glossarion v2.9.4...")
    
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
