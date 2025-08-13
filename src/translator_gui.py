#translator_gui.py
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

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
from api_key_encryption import encrypt_config, decrypt_config
from metadata_batch_translator import MetadataBatchTranslatorUI

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
            
            # Sanity check - if we got the full desktop width, try another method
            total_width = reference_window.winfo_screenwidth()
            if primary_width >= total_width * 0.9:
                # Likely got the full desktop, not just primary monitor
                # Use aspect ratio method as fallback
                screen_height = reference_window.winfo_screenheight()
                
                # Common aspect ratios
                aspect_ratios = [16/9, 16/10, 21/9, 4/3]
                for ratio in aspect_ratios:
                    test_width = int(screen_height * ratio)
                    if test_width < total_width * 0.7:  # Reasonable for primary monitor
                        primary_width = test_width
                        break
                else:
                    # Default to half of total if nothing else works
                    primary_width = total_width // 2
            
            self._primary_monitor_width = primary_width
            print(f"Detected primary monitor width: {primary_width}")
            return primary_width
            
        except Exception as e:
            print(f"Error detecting monitor: {e}")
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
        """Center a window on primary screen with auto-detection"""
        def do_center():
            if window.winfo_exists():
                window.update_idletasks()
                width = window.winfo_width()
                height = window.winfo_height()
                screen_height = window.winfo_screenheight()
                
                # Auto-detect primary monitor width
                primary_width = self.detect_primary_monitor_width(window)
                
                # Center on primary monitor
                x = (primary_width - width) // 2
                y = (screen_height - height) // 2
                
                # Move up by reducing Y (adjust this value as needed)
                y = max(30, y - 340)  # Move up by 340 pixels, but keep at least 30 from top
                
                # Ensure it stays on primary monitor
                x = max(0, min(x, primary_width - width))
                
                window.geometry(f"+{x}+{y}")
        
        window.after_idle(do_center)
    
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
        __version__ = "3.9.0"
        self.__version__ = __version__  # Store as instance variable
        master.title(f"Glossarion v{__version__}")
        
        # Get screen dimensions
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        
        # Set window size as ratio of screen (e.g., 0.8 = 80% of screen)
        width_ratio = 1.2  # 120% of screen width
        height_ratio = 1.2  # 120% of screen height
        
        window_width = int(screen_width * width_ratio)
        window_height = int(screen_height * height_ratio)
        
        # Apply size
        master.geometry(f"{window_width}x{window_height}")
        
        # Set minimum size as ratio too
        min_width = int(screen_width * 0.6)  # 60% minimum
        min_height = int(screen_height * 0.6)  # 60% minimum
        master.minsize(min_width, min_height)
        
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
        self.remove_ai_artifacts = os.getenv("REMOVE_AI_ARTIFACTS", "0") == "1"
        print(f"   🎨 Remove AI Artifacts: {'ENABLED' if self.remove_ai_artifacts else 'DISABLED'}")
        self.disable_chapter_merging_var = tk.BooleanVar(value=self.config.get('disable_chapter_merging', False))
        self.selected_files = []
        self.current_file_index = 0
        self.use_gemini_openai_endpoint_var = tk.BooleanVar(value=self.config.get('use_gemini_openai_endpoint', False))
        self.gemini_openai_endpoint_var = tk.StringVar(value=self.config.get('gemini_openai_endpoint', ''))
        # Initialize fuzzy threshold variable
        if not hasattr(self, 'fuzzy_threshold_var'):
            self.fuzzy_threshold_var = tk.DoubleVar(value=self.config.get('glossary_fuzzy_threshold', 0.90))

        
        # Initialize the variables with default values
        self.enable_parallel_extraction_var = tk.BooleanVar(value=self.config.get('enable_parallel_extraction', True))
        self.extraction_workers_var = tk.IntVar(value=self.config.get('extraction_workers', 4))

        # Set initial environment variable
        if self.enable_parallel_extraction_var.get():
            os.environ["EXTRACTION_WORKERS"] = str(self.extraction_workers_var.get())
        else:
            os.environ["EXTRACTION_WORKERS"] = "1"


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
        # Glossary-related variables (existing)
        self.enable_auto_glossary_var = tk.BooleanVar(value=self.config.get('enable_auto_glossary', False))
        self.append_glossary_var = tk.BooleanVar(value=self.config.get('append_glossary', False))
        self.glossary_min_frequency_var = tk.StringVar(value=str(self.config.get('glossary_min_frequency', 2)))
        self.glossary_max_names_var = tk.StringVar(value=str(self.config.get('glossary_max_names', 50)))
        self.glossary_max_titles_var = tk.StringVar(value=str(self.config.get('glossary_max_titles', 30)))
        self.glossary_batch_size_var = tk.StringVar(value=str(self.config.get('glossary_batch_size', 50)))
        self.glossary_max_text_size_var = tk.IntVar(value=self.config.get('glossary_max_text_size', 50000))

        
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
        self.attach_css_to_chapters_var = tk.BooleanVar(value=self.config.get('attach_css_to_chapters', False))

        
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
                "- Match the translation tone to the character’s expression.\n"
                "- If a character looks angry, use appropriately intense language.\n"
                "- If a character looks shy or embarrassed, reflect that in the translation.\n"
                "- Keep speech patterns consistent with the character’s appearance and demeanor.\n"
                "- Retain honorifics and onomatopoeia in Romaji.\n\n"

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
                "- Match the translation tone to the character’s expression.\n"
                "- If a character looks angry, use appropriately intense language.\n"
                "- If a character looks shy or embarrassed, reflect that in the translation.\n"
                "- Keep speech patterns consistent with the character’s appearance and demeanor.\n"
                "- Retain honorifics and onomatopoeia in Romaji.\n\n"

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
                "- Match the translation tone to the character’s expression.\n"
                "- If a character looks angry, use appropriately intense language.\n"
                "- If a character looks shy or embarrassed, reflect that in the translation.\n"
                "- Keep speech patterns consistent with the character’s appearance and demeanor.\n"
                "- Retain honorifics and onomatopoeia in Romaji.\n\n"

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
        """Check for updates on startup with debug logging"""
        print("[DEBUG] Running startup update check...")
        if self.update_manager:
            try:
                update_available, release_info = self.update_manager.check_for_updates(silent=True)
                print(f"[DEBUG] Update check result: available={update_available}")
                if release_info:
                    print(f"[DEBUG] Latest version: {release_info.get('tag_name', 'unknown')}")
            except Exception as e:
                print(f"[DEBUG] Update check failed: {e}")
        else:
            print("[DEBUG] Update manager is None")
        
    def check_for_updates_manual(self):
        """Manually check for updates from the Other Settings dialog"""
        if hasattr(self, 'update_manager') and self.update_manager:
            self.update_manager.check_for_updates(silent=False)
        else:
            messagebox.showerror("Update Check", 
                               "Update manager is not available.\n"
                               "Please check the GitHub releases page manually:\n"
                               "https://github.com/Shirochi-stack/Glossarion/releases")
                               
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
        """Open manga translator in a new window using WindowManager"""
        if not MANGA_SUPPORT:
            messagebox.showwarning("Not Available", "Manga translation modules not found.")
            return
        
        # Use WindowManager to create scrollable dialog
        dialog, scrollable_frame, canvas = self.wm.setup_scrollable(
            self.master,
            "Manga Panel Translator",
            width=900,
            height=700,
            max_width_ratio=0.9,
            max_height_ratio=1.45
        )
        
        # Initialize the manga translator interface on the scrollable frame
        self.manga_translator = MangaTranslationTab(scrollable_frame, self, dialog, canvas)
        
        # Auto-resize to fit content
        self.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.9, max_height_ratio=1.6)
        
        # Handle window close
        def on_close():
            dialog._cleanup_scrolling()
            dialog.destroy()
            self.manga_translator = None
        
        dialog.protocol("WM_DELETE_WINDOW", on_close)
      
        
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
            ('reset_failed_chapters_var', 'reset_failed_chapters', True),
            ('retry_truncated_var', 'retry_truncated', False),
            ('retry_duplicate_var', 'retry_duplicate_bodies', False),
            ('enable_image_translation_var', 'enable_image_translation', False),
            ('process_webnovel_images_var', 'process_webnovel_images', True),
            # REMOVED: ('comprehensive_extraction_var', 'comprehensive_extraction', False),
            ('hide_image_translation_label_var', 'hide_image_translation_label', True),
            ('retry_timeout_var', 'retry_timeout', True),
            ('batch_translation_var', 'batch_translation', False),
            ('disable_epub_gallery_var', 'disable_epub_gallery', False),
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
        self.append_log("🚀 Glossarion v3.9.0 - Ready to use!")
        self.append_log("💡 Click any function button to load modules automatically")
    
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
        self.browse_menu.add_command(label="📄 Select Single File", command=self.browse_file)
        self.browse_menu.add_command(label="📑 Select Multiple Files", command=self.browse_multiple_files)
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
        options_frame.grid(row=3, column=0, columnspan=1, sticky=tk.EW, padx=5, pady=5)
        
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
        
        # Show Google Cloud Credentials button for Vertex AI models
        if '@' in model or model.startswith('vertex/') or model.startswith('vertex_ai/'):
            self.gcloud_button.config(state=tk.NORMAL)
            self.vertex_location_entry.grid() 
            
            # Update API key label to indicate it's optional for Vertex
            # Find your api_key_label and update it
            # self.api_key_label.config(text="Project ID (optional):")
            
            # Check if credentials are already loaded
            if self.config.get('google_cloud_credentials'):
                creds_path = self.config['google_cloud_credentials']
                if os.path.exists(creds_path):
                    try:
                        with open(creds_path, 'r') as f:
                            creds_data = json.load(f)
                            self.gcloud_status_label.config(
                                text=f"✓ Credentials: {os.path.basename(creds_path)} (Project: {creds_data.get('project_id', 'Unknown')})",
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
                self.gcloud_status_label.config(
                    text="⚠ No Google Cloud credentials selected",
                    foreground='orange'
                )
        else:
            self.gcloud_button.config(state=tk.DISABLED)
            self.vertex_location_entry.grid_remove()
            self.gcloud_status_label.config(text="")
            # Reset API key label if you changed it
            # self.api_key_label.config(text="API Key:")

    # Also add this to bind manual typing events to the combobox
    def setup_model_combobox_bindings(self):
        """Setup bindings for manual model input in combobox"""
        # Bind to key release events to detect manual typing
        self.model_combo.bind('<KeyRelease>', self.on_model_change)
        # Also bind to FocusOut to catch when user clicks away after typing
        self.model_combo.bind('<FocusOut>', self.on_model_change)
        # Keep the existing binding for dropdown selection
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
    def _create_model_section(self):
        """Create model selection section"""
        tb.Label(self.frame, text="Model:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        default_model = self.config.get('model', 'gemini-2.0-flash')
        self.model_var = tk.StringVar(value=default_model)
        models = [
            # OpenAI Models
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1",
            "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k",
            "gpt-5-mini","gpt-5","gpt-5-nano",
            "o1-preview", "o1-mini", "o3", "o4-mini",
            
            # Google Gemini Models
            "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash","gemini-2.0-flash-lite",
            "gemini-2.5-flash","gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-pro", "gemini-pro-vision",
            
            # Anthropic Claude Models
            "claude-opus-4-20250514", "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219",
            "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
            "claude-2.1", "claude-2", "claude-instant-1.2",
            
            # Grok Models
            "grok-grok-4-0709", "grok-3", "grok-3-mini",
            
            # Vertex AI Model Garden - Claude models (confirmed)
            "claude-4-opus@20250514",
            "claude-4-sonnet@20250514",
            "claude-opus-4@20250514",
            "claude-sonnet-4@20250514",
            "claude-3-7-sonnet@20250219",
            "claude-3-5-sonnet@20240620",
            "claude-3-5-sonnet-v2@20241022",
            "claude-3-opus@20240229",
            "claude-3-sonnet@20240229",
            "claude-3-haiku@20240307",

            
            # Alternative format with vertex_ai prefix
            "vertex/claude-3-7-sonnet@20250219",
            "vertex/claude-3-5-sonnet@20240620",
            "vertex/claude-3-opus@20240229",
            "vertex/claude-4-opus@20250514",
            "vertex/claude-4-sonnet@20250514",
            "vertex/gemini-1.5-pro",
            "vertex/gemini-1.5-flash",
            "vertex/gemini-2.0-flash",
            "vertex/gemini-2.5-pro",
            "vertex/gemini-2.5-flash",
            "vertex/gemini-2.5-flash-lite",
            
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
            
            # For POE, prefix with 'poe/'
            "poe/gpt-4", "poe/gpt-4o", "poe/gpt-4.5", "poe/gpt-4.1",
            "poe/claude-3-opus", "poe/claude-4-opus", "poe/claude-3-sonnet", "poe/claude-4-sonnet",
            "poe/claude", "poe/Assistant",
            "poe/gemini-2.5-flash", "poe/gemini-2.5-pro",
            
            # For ElectronHub, prefix with 'eh/'
            "eh/gpt-4", "eh/gpt-3.5-turbo", "eh/claude-3-opus", "eh/claude-3-sonnet",
            "eh/llama-2-70b-chat", "eh/yi-34b-chat-200k", "eh/mistral-large",
            "eh/gemini-pro", "eh/deepseek-coder-33b",
        ]
        self.model_combo = tb.Combobox(self.frame, textvariable=self.model_var, values=models, state="normal")
        self.model_combo.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        
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
            # API delay (left side)
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
        Shared logic for force retranslation of EPUB/text files
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
        
        if not prog.get("chapters"):
            if not parent_dialog:
                messagebox.showinfo("Info", "No chapters found in progress tracking.")
            return None
        
        # If no parent dialog or tab frame, create standalone dialog
        if not parent_dialog and not tab_frame:
            dialog = self.wm.create_simple_dialog(
                self.master,
                "Force Retranslation",
                width=900,
                height=600
            )
            container = dialog
        else:
            container = tab_frame or parent_dialog
            dialog = parent_dialog
        
        # Process chapter data (same logic as before)
        files_to_entries = {}
        for chapter_key, chapter_info in prog.get("chapters", {}).items():
            output_file = chapter_info.get("output_file", "")
            if output_file:
                if output_file not in files_to_entries:
                    files_to_entries[output_file] = []
                files_to_entries[output_file].append((chapter_key, chapter_info))
        
        # Build display info
        chapter_display_info = []
        for output_file, entries in files_to_entries.items():
            chapter_key, chapter_info = entries[0]
            
            # Extract chapter number
            from TransateKRtoEN import extract_chapter_number_from_filename
            chapter_num, _ = extract_chapter_number_from_filename(output_file)
            
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
        
        # Create UI elements
        if not tab_frame:
            tk.Label(container, text="Select chapters to retranslate (scroll horizontally if needed):", 
                    font=('Arial', 12)).pack(pady=10)
        else:
            tk.Label(container, text="Select chapters to retranslate:", 
                    font=('Arial', 11)).pack(pady=5)
        
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
            selectmode=tk.MULTIPLE, 
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set,
            width=100
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
            'unknown': '❓'
        }
        
        for info in chapter_display_info:
            chapter_num = info['num']
            status = info['status']
            output_file = info['output_file']
            icon = status_icons.get(status, '❓')
            
            # Format display
            if isinstance(chapter_num, float) and chapter_num.is_integer():
                display = f"Chapter {int(chapter_num):03d} | {icon} {status} | {output_file}"
            elif isinstance(chapter_num, float):
                display = f"Chapter {chapter_num:06.1f} | {icon} {status} | {output_file}"
            else:
                display = f"Chapter {chapter_num:03d} | {icon} {status} | {output_file}"
            
            if info['duplicate_count'] > 1:
                display += f" | ({info['duplicate_count']} entries)"
            
            listbox.insert(tk.END, display)
        
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
            'files_to_entries': files_to_entries,
            'chapter_display_info': chapter_display_info,
            'listbox': listbox,
            'selection_count_label': selection_count_label,
            'dialog': dialog,
            'container': container
        }
        
        # If standalone (no parent), add buttons
        if not parent_dialog or tab_frame:
            self._add_retranslation_buttons(result)
        
        return result


    def _add_retranslation_buttons(self, data, button_frame=None):
        """Add the standard button set for retranslation dialogs"""
        
        if not button_frame:
            button_frame = tk.Frame(data['container'])
            button_frame.pack(pady=10)
        
        # Configure column weights
        for i in range(4):
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
                for chapter_key, _ in info['entries']:
                    if chapter_key in data['prog']["chapters"]:
                        data['prog']["chapters"][chapter_key]["status"] = "completed"
                        data['prog']["chapters"][chapter_key].pop("qa_issues", None)
                        data['prog']["chapters"][chapter_key].pop("qa_timestamp", None)
                        data['prog']["chapters"][chapter_key].pop("qa_issues_found", None)
                        data['prog']["chapters"][chapter_key].pop("duplicate_confidence", None)
                        cleared_count += 1
            
            # Save
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
            
            count = len(selected)
            if count > 10:
                confirm_msg = f"This will delete {count} translated chapters and mark them for retranslation.\n\nContinue?"
            else:
                chapters = [f"Chapter {data['chapter_display_info'][i]['num']}" for i in selected]
                confirm_msg = f"This will delete and retranslate:\n\n{', '.join(chapters)}\n\nContinue?"
            
            if not messagebox.askyesno("Confirm Retranslation", confirm_msg):
                return
            
            # Delete files and update progress
            deleted_count = 0
            for idx in selected:
                info = data['chapter_display_info'][idx]
                output_file = info['output_file']
                
                # Delete file
                if output_file:
                    output_path = os.path.join(data['output_dir'], output_file)
                    try:
                        if os.path.exists(output_path):
                            os.remove(output_path)
                            deleted_count += 1
                    except Exception as e:
                        print(f"Failed to delete {output_path}: {e}")
                
                # Remove from progress
                for chapter_key, _ in info['entries']:
                    if chapter_key in data['prog']["chapters"]:
                        content_hash = data['prog']["chapters"][chapter_key].get("content_hash")
                        del data['prog']["chapters"][chapter_key]
                        
                        if content_hash and content_hash in data['prog'].get("content_hashes", {}):
                            del data['prog']["content_hashes"][content_hash]
                        
                        if content_hash and content_hash in data['prog'].get("chapter_chunks", {}):
                            del data['prog']["chapter_chunks"][content_hash]
            
            # Save
            with open(data['progress_file'], 'w', encoding='utf-8') as f:
                json.dump(data['prog'], f, ensure_ascii=False, indent=2)
            
            messagebox.showinfo("Success", 
                f"Deleted {deleted_count} files and cleared {len(selected)} chapters from tracking.")
            
            if data.get('dialog'):
                data['dialog'].destroy()
        
        # Add buttons
        tb.Button(button_frame, text="Select All", command=select_all, 
                  bootstyle="info").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        tb.Button(button_frame, text="Clear Selection", command=clear_selection, 
                  bootstyle="secondary").grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tb.Button(button_frame, text="Select Completed", command=lambda: select_status('completed'), 
                  bootstyle="success").grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        tb.Button(button_frame, text="Select Failed", command=lambda: select_status('failed'), 
                  bootstyle="danger").grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        
        tb.Button(button_frame, text="Retranslate Selected", command=retranslate_selected, 
                  bootstyle="warning").grid(row=1, column=0, columnspan=2, padx=5, pady=10, sticky="ew")
        tb.Button(button_frame, text="Remove QA Failed Mark", command=remove_qa_failed_mark, 
                  bootstyle="success").grid(row=1, column=2, columnspan=1, padx=5, pady=10, sticky="ew")
        tb.Button(button_frame, text="Cancel", command=lambda: data['dialog'].destroy() if data.get('dialog') else None, 
                  bootstyle="secondary").grid(row=1, column=3, columnspan=1, padx=5, pady=10, sticky="ew")


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
            selectmode=tk.MULTIPLE,
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
                        if file.endswith('.html') and base_name in file:
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
            selectmode=tk.MULTIPLE,
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
        for file in sorted(os.listdir(output_dir)):
            if file.endswith('.html'):
                match = re.match(r'response_(\d+)_(.+)\.html', file)
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
                        if any(f.endswith('.html') for f in files):
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
            selectmode=tk.MULTIPLE, 
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
                    # Expected format: response_001_imagename.html
                    html_file = item['file']
                    match = re.match(r'response_\d+_(.+)\.html', html_file)
                    
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

                
                # Honorifics and other settings
                if hasattr(self, 'strip_honorifics_var'):
                    self.config['strip_honorifics'] = self.strip_honorifics_var.get()
                if hasattr(self, 'disable_honorifics_var'):
                    self.config['glossary_disable_honorifics_filter'] = self.disable_honorifics_var.get()
                
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

        # Update label when slider moves - DEFINE AFTER CREATING THE LABEL
        def update_fuzzy_label(*args):
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

        # Set up the trace AFTER creating the label
        self.fuzzy_threshold_var.trace('w', update_fuzzy_label)

        # Initialize description by calling the function
        update_fuzzy_label()
        
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
        
        tk.Label(master_toggle_frame, text="(NOT RECOMMENDED, Automatic extraction and translation of character names)",
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
        
        # Row 3 - Max text size
        tk.Label(extraction_grid, text="Max text size:").grid(row=3, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        tb.Entry(extraction_grid, textvariable=self.glossary_max_text_size_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=(0, 20), pady=(5, 0))

        tk.Label(extraction_grid, text="(0 = analyze entire text)").grid(row=3, column=2, columnspan=2, sticky=tk.W, padx=(0, 5), pady=(5, 0))

        # Move honorifics to Row 4 (was Row 2)
        tk.Label(extraction_grid, text="Strip honorifics:").grid(row=4, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        tb.Checkbutton(extraction_grid, text="Remove honorifics from extracted names", 
                      variable=self.strip_honorifics_var,
                      bootstyle="round-toggle").grid(row=4, column=1, columnspan=3, sticky=tk.W, pady=(5, 0))
                
        # Initialize the variable if not exists
        if not hasattr(self, 'strip_honorifics_var'):
            self.strip_honorifics_var = tk.BooleanVar(value=True)
        
        tb.Checkbutton(extraction_grid, text="Remove honorifics from extracted names", 
                      variable=self.strip_honorifics_var,
                      bootstyle="round-toggle").grid(row=2, column=1, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Help text
        help_frame = tk.Frame(extraction_tab)
        help_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        
        tk.Label(help_frame, text="💡 Settings Guide:", font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W)
        help_texts = [
            "• Min frequency: How many times a name must appear (lower = more terms)",
            "• Max names/titles: Limits to prevent huge glossaries",
            "• Translation batch: Terms per API call (larger = faster but may reduce quality)",
            "• Max text size: Characters to analyze (0 = entire text, 50000 = first 50k chars)",
            "• Strip honorifics: Extract clean names without suffixes (e.g., '김' instead of '김님')"
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
    
    def run_translation_thread(self):
        """Start translation in a separate thread"""
        if hasattr(self, 'glossary_thread') and self.glossary_thread and self.glossary_thread.is_alive():
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
        
        # Start thread IMMEDIATELY - no heavy operations here
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
            # Show initial feedback immediately
            self.append_log("🚀 Starting translation process...")
            
            # Load modules in background thread (not main thread!)
            if not self._modules_loaded:
                self.append_log("📦 Loading translation modules...")
                
                # Create a progress callback that uses append_log
                def module_progress(msg):
                    self.append_log(f"   {msg}")
                
                # Load modules with progress feedback
                if not self._lazy_load_modules(splash_callback=module_progress):
                    self.append_log("❌ Failed to load required modules")
                    return
            
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
                                max_workers = self.config.get('extraction_workers', 4)
                            else:
                                # Fallback to environment variable or default
                                max_workers = int(os.environ.get('EXTRACTION_WORKERS', '4'))
                            
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
                            if file_count > 300:
                                os.environ['SKIP_VALIDATION'] = '1'
                                os.environ['LAZY_LOAD_CONTENT'] = '1'
                                self.append_log("🚀 Enabled aggressive optimization for very large file")
                            
                except Exception as e:
                    # If we can't check, just continue
                    pass
            
            # Now run the actual translation
            self.run_translation_direct()
            
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
                return
            
            # DON'T CALL _lazy_load_modules HERE!
            # Modules are already loaded in the wrapper
            # Just verify they're loaded
            if not self._modules_loaded:
                self.append_log("❌ Translation modules not loaded")
                return

            # Check stop after verification
            if self.stop_requested:
                return

            # SET GLOSSARY IN ENVIRONMENT
            if hasattr(self, 'manual_glossary_path') and self.manual_glossary_path:
                os.environ['MANUAL_GLOSSARY'] = self.manual_glossary_path
                self.append_log(f"📑 Set glossary in environment: {os.path.basename(self.manual_glossary_path)}")
            else:
                # Clear any previous glossary from environment
                if 'MANUAL_GLOSSARY' in os.environ:
                    del os.environ['MANUAL_GLOSSARY']
                self.append_log(f"ℹ️ No glossary loaded")

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
                    return
                    
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
                return
                
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
            
        except Exception as e:
            self.append_log(f"❌ Translation setup error: {e}")
            import traceback
            self.append_log(f"❌ Full error: {traceback.format_exc()}")
        
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
                        
                        # Check for duplicate content
                        if content_hash in self.prog.get("content_hashes", {}):
                            duplicate_info = self.prog["content_hashes"][content_hash]
                            duplicate_output = duplicate_info.get("output_file")
                            if duplicate_output and os.path.exists(duplicate_output):
                                return False, f"Duplicate of {duplicate_info.get('original_name')}", duplicate_output
                        
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
                            
                            with open(manual_glossary_path, 'r', encoding='utf-8') as f:
                                glossary_data = json.load(f)
                                
                            output_glossary_path = os.path.join(output_dir, "glossary.json")
                            with open(output_glossary_path, 'w', encoding='utf-8') as f:
                                json.dump(glossary_data, f, ensure_ascii=False, indent=2)
                            self.append_log(f"💾 Saved glossary to output folder for auto-loading")
                            
                            # Format glossary for prompt
                            formatted_entries = {}
                            
                            if isinstance(glossary_data, list):
                                # List format (from glossary extractor)
                                for char in glossary_data:
                                    if not isinstance(char, dict):
                                        continue
                                        
                                    original = char.get('original_name', '')
                                    translated = char.get('name', original)
                                    if original and translated:
                                        formatted_entries[original] = translated
                                    
                                    # Include titles if present
                                    title = char.get('title')
                                    if title and original:
                                        formatted_entries[f"{original} ({title})"] = f"{translated} ({title})"
                                    
                                    # Include reference mappings
                                    refer_map = char.get('how_they_refer_to_others', {})
                                    if isinstance(refer_map, dict):
                                        for other_name, reference in refer_map.items():
                                            if other_name and reference:
                                                formatted_entries[f"{original} → {other_name}"] = f"{translated} → {reference}"
                            
                            elif isinstance(glossary_data, dict):
                                # Dictionary format
                                if "entries" in glossary_data and isinstance(glossary_data["entries"], dict):
                                    formatted_entries = glossary_data["entries"]
                                else:
                                    # Direct dictionary, exclude metadata
                                    formatted_entries = {k: v for k, v in glossary_data.items() if k != "metadata"}
                            
                            # Append glossary to system prompt if we have entries
                            if formatted_entries:
                                glossary_block = json.dumps(formatted_entries, ensure_ascii=False, indent=2)
                                
                                # Add newlines if system prompt already has content
                                if system_prompt:
                                    system_prompt += "\n\n"
                                
                                # Get custom glossary prompt or use default
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
            'BOOK_TITLE_SYSTEM_PROMPT': self.config.get('book_title_system_prompt', 
                "You are a translator. Respond with only the translated text, nothing else. Do not add any explanation or additional content."),
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
            'GLOSSARY_STRIP_HONORIFICS': "1" if self.strip_honorifics_var.get() else "0",
            'AUTO_GLOSSARY_PROMPT': self.auto_glossary_prompt if hasattr(self, 'auto_glossary_prompt') else '',
            'APPEND_GLOSSARY_PROMPT': self.append_glossary_prompt if hasattr(self, 'append_glossary_prompt') else '',
            'GLOSSARY_TRANSLATION_PROMPT': self.glossary_translation_prompt if hasattr(self, 'glossary_translation_prompt') else '',
            'GLOSSARY_FORMAT_INSTRUCTIONS': self.glossary_format_instructions if hasattr(self, 'glossary_format_instructions') else '',
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
            'BATCH_TRANSLATION': "1" if self.batch_translation_var.get() else "0",
            'BATCH_SIZE': self.batch_size_var.get(),
            'DISABLE_ZERO_DETECTION': "1" if self.disable_zero_detection_var.get() else "0",
            'TRANSLATION_HISTORY_ROLLING': "1" if self.translation_history_rolling_var.get() else "0",
            'USE_GEMINI_OPENAI_ENDPOINT': '1' if self.use_gemini_openai_endpoint_var.get() else '0',
            'GEMINI_OPENAI_ENDPOINT': self.gemini_openai_endpoint_var.get() if self.gemini_openai_endpoint_var.get() else '',
            "ATTACH_CSS_TO_CHAPTERS": "1" if self.attach_css_to_chapters_var.get() else "0",
            'GLOSSARY_FUZZY_THRESHOLD': str(self.config.get('glossary_fuzzy_threshold', 0.90)),
            'GLOSSARY_MAX_TEXT_SIZE': self.glossary_max_text_size_var.get(),

            # Extraction settings
            "EXTRACTION_MODE": extraction_mode,
            "ENHANCED_FILTERING": enhanced_filtering,
            "ENHANCED_PRESERVE_STRUCTURE": "1" if getattr(self, 'enhanced_preserve_structure_var', tk.BooleanVar(value=True)).get() else "0",
            
            # For new UI
            "TEXT_EXTRACTION_METHOD": extraction_method if hasattr(self, 'text_extraction_method_var') else ('enhanced' if extraction_mode == 'enhanced' else 'standard'),
            "FILE_FILTERING_LEVEL": filtering_level if hasattr(self, 'file_filtering_level_var') else extraction_mode,
            'DISABLE_CHAPTER_MERGING': '1' if self.disable_chapter_merging_var.get() else '0',
            'DISABLE_EPUB_GALLERY': "1" if self.disable_epub_gallery_var.get() else "0",
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
            
           # Multi API Key support
            'USE_MULTI_API_KEYS': "1" if self.config.get('use_multi_api_keys', False) else "0",
            'MULTI_API_KEYS': json.dumps(self.config.get('multi_api_keys', [])) if self.config.get('use_multi_api_keys', False) else '[]',
            'FORCE_KEY_ROTATION': '1' if self.config.get('force_key_rotation', True) else '0',
            'ROTATION_FREQUENCY': str(self.config.get('rotation_frequency', 1)),
           
       }
        print(f"[DEBUG] DISABLE_CHAPTER_MERGING = '{os.getenv('DISABLE_CHAPTER_MERGING', '0')}'")
        
    def run_glossary_extraction_thread(self):
        """Start glossary extraction in a separate thread"""
        if hasattr(self, 'translation_thread') and self.translation_thread and self.translation_thread.is_alive():
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
        try:
            api_key = self.api_key_entry.get()
            model = self.model_var.get()
            
            # Validate Vertex AI credentials if needed
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
                    'GLOSSARY_DUPLICATE_KEY_MODE': 'skip',  # Always use skip mode for new format
                    'SEND_INTERVAL_SECONDS': str(self.delay_entry.get()),
                    'CONTEXTUAL': '1' if self.contextual_var.get() else '0',
                    'GOOGLE_APPLICATION_CREDENTIALS': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''),
                    
                    # NEW GLOSSARY ADDITIONS
                    'GLOSSARY_MIN_FREQUENCY': str(self.glossary_min_frequency_var.get()),
                    'GLOSSARY_MAX_NAMES': str(self.glossary_max_names_var.get()),
                    'GLOSSARY_MAX_TITLES': str(self.glossary_max_titles_var.get()),
                    'GLOSSARY_BATCH_SIZE': str(self.glossary_batch_size_var.get()),
                    'ENABLE_AUTO_GLOSSARY': "1" if self.enable_auto_glossary_var.get() else "0",
                    'DISABLE_AUTO_GLOSSARY': "0" if self.enable_auto_glossary_var.get() else "1",  # Inverted!
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
            self.stop_requested = False
            # Schedule GUI update on main thread
            self.master.after(0, self.update_run_button)

                
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
                'check_encoding_issues': False,
                'check_repetition': True,
                'check_translation_artifacts': True,
                'min_file_length': 0,
                'report_format': 'detailed',
                'auto_save_report': True,
                'check_missing_html_tag': True,     
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
            # Check if word count cross-reference is enabled but no EPUB is selected
            check_word_count = qa_settings.get('check_word_count_ratio', False)
            epub_path = None
            
            if check_word_count:
                print("[DEBUG] Word count check is enabled, looking for EPUB...")
                epub_path = self.get_current_epub_path()
                print(f"[DEBUG] get_current_epub_path returned: {epub_path}")
                    
                if not epub_path:
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
                        else:
                            self.selected_epub_path = epub_path
                            self.config['last_epub_path'] = epub_path
                            self.save_config(show_message=False)
                            self.append_log(f"✅ Selected EPUB: {os.path.basename(epub_path)}")
                    else:  # Yes - Continue without word count
                        qa_settings = qa_settings.copy()
                        qa_settings['check_word_count_ratio'] = False
                        self.append_log("ℹ️ Proceeding without word count analysis.")
            # Now get the folder
            folder_path = filedialog.askdirectory(title="Select Folder with HTML Files")
            if not folder_path:
                self.append_log("⚠️ QA scan canceled.")
                return

            # Check for EPUB/folder name mismatch
            if epub_path and qa_settings.get('check_word_count_ratio', False) and qa_settings.get('warn_name_mismatch', True):
                epub_name = os.path.splitext(os.path.basename(epub_path))[0]
                folder_name = os.path.basename(folder_path.rstrip('/\\'))
                
                if not check_epub_folder_match(epub_name, folder_name, qa_settings.get('custom_output_suffixes', '')):
                    result = messagebox.askyesnocancel(
                        "EPUB/Folder Name Mismatch",
                        f"The source EPUB and output folder names don't match:\n\n" +
                        f"📖 EPUB: {epub_name}\n" +
                        f"📁 Folder: {folder_name}\n\n" +
                        "This might mean you're comparing the wrong files.\n" +
                        "Common issues:\n" +
                        "• 'Novel123' vs 'Novel124' (different books)\n" +
                        "• 'Book_1' vs 'Book_2' (different volumes)\n\n" +
                        "Would you like to:\n" +
                        "• YES - Continue anyway (I'm sure these match)\n" +
                        "• NO - Select a different EPUB file\n" +
                        "• CANCEL - Select a different folder",
                        icon='warning'
                    )
                    
                    if result is None:  # Cancel - select different folder
                        new_folder_path = filedialog.askdirectory(
                            title="Select Different Folder with HTML Files"
                        )
                        if new_folder_path:
                            folder_path = new_folder_path
                        else:
                            self.append_log("⚠️ QA scan canceled.")
                            return
                            
                    elif result is False:  # No - select different EPUB
                        new_epub_path = filedialog.askopenfilename(
                            title="Select Different Source EPUB File",
                            filetypes=[("EPUB files", "*.epub"), ("All files", "*.*")]
                        )
                        
                        if new_epub_path:
                            epub_path = new_epub_path
                            self.selected_epub_path = epub_path
                            self.config['last_epub_path'] = epub_path
                            self.save_config(show_message=False)
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
                                qa_settings = qa_settings.copy()
                                qa_settings['check_word_count_ratio'] = False
                                epub_path = None
                                self.append_log("ℹ️ Proceeding without word count analysis.")
                    else:
                        self.append_log(f"⚠️ Warning: EPUB/folder name mismatch - {epub_name} vs {folder_name}")
            
            mode = selected_mode_value
            self.append_log(f"🔍 Starting QA scan in {mode.upper()} mode for folder: {folder_path}")
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
                self.master.after(0, self.update_run_button)
                self.qa_button.config(text="Stop Scan", command=self.stop_qa_scan, bootstyle="danger")
                
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
                    
                    # Pass the QA settings to scan_html_folder (without custom_settings)
                    scan_html_folder(
                        folder_path, 
                        log=self.append_log, 
                        stop_flag=lambda: self.stop_requested, 
                        mode=mode,
                        qa_settings=qa_settings,  # Keep existing qa_settings parameter
                        epub_path=epub_path
                    )
                    
                    # If show_stats is enabled, log cache statistics
                    if qa_settings.get('cache_show_stats', False):
                        from scan_html_folder import get_cache_info
                        cache_stats = get_cache_info()
                        self.append_log("\n📊 Cache Performance Statistics:")
                        for name, info in cache_stats.items():
                            if info:  # Check if info exists
                                hit_rate = info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0
                                self.append_log(f"  {name}: {info.hits} hits, {info.misses} misses ({hit_rate:.1%} hit rate)")
                    
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
            check_artifacts_var = tk.BooleanVar(value=qa_settings.get('check_translation_artifacts', True))
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

            current_epub = self.get_current_epub_path()
            if current_epub:
                status_text = f"📖 Current EPUB: {os.path.basename(current_epub)}"
                status_color = 'green'
            else:
                status_text = "📖 No EPUB file selected"
                status_color = 'red'

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
                    check_artifacts_var.set(True)
                    check_glossary_var.set(True)
                    min_length_var.set(0)
                    format_var.set('detailed')
                    auto_save_var.set(True)
                    check_word_count_var.set(False)
                    check_multiple_headers_var.set(True)
                    warn_mismatch_var.set(False)
                    check_missing_html_tag_var.set(True)
                    check_paragraph_structure_var.set(True)
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
        
        try:
            import unified_api_client
            if hasattr(unified_api_client, 'set_stop_flag'):
                unified_api_client.set_stop_flag(True)
            # If there's a global client instance, stop it too
            if hasattr(unified_api_client, 'global_stop_flag'):
                unified_api_client.global_stop_flag = True
        except:
            pass        
        
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
                        self.manual_glossary_manually_loaded = False  # This is auto-loaded
                        self.append_log(f"📑 Auto-loaded glossary for {file_base}: {os.path.basename(glossary_path)}")
                        return True
                except Exception:
                    continue
        
        return False

    # File Selection Methods
    def browse_file(self):
        """Select a single file"""
        path = filedialog.askopenfilename(
            title="Select File",
            filetypes=[
                ("Supported files", "*.epub;*.txt;*.json;*.png;*.jpg;*.jpeg;*.gif;*.bmp;*.webp"),
                ("EPUB files", "*.epub"),
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
        if path:
            self._handle_file_selection([path])

    def browse_multiple_files(self):
        """Select multiple files at once"""
        paths = filedialog.askopenfilenames(
            title="Select Multiple Files (Ctrl+Click or Shift+Click)",
            filetypes=[
                ("Supported files", "*.epub;*.txt;*.json;*.png;*.jpg;*.jpeg;*.gif;*.bmp;*.webp"),
                ("EPUB files", "*.epub"),
                ("Text files", "*.txt;*.json"),
                ("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp;*.webp"),
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
            supported_extensions = {'.epub', '.txt', '.json', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
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
            if path.lower().endswith('.json'):
                # Convert JSON to TXT
                txt_path = self._convert_json_to_txt(path)
                if txt_path:
                    processed_paths.append(txt_path)
                    # Track the conversion for potential reverse conversion later
                    self.json_conversions[txt_path] = path
                    self.append_log(f"📄 Converted JSON to TXT: {os.path.basename(path)}")
                else:
                    self.append_log(f"❌ Failed to convert JSON: {os.path.basename(path)}")
            else:
                # Non-JSON files pass through unchanged
                processed_paths.append(path)
        
        # Store the list of selected files (using processed paths)
        self.selected_files = processed_paths
        self.current_file_index = 0
        
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
            elif len(epub_files) > 1:
                # Multiple EPUBs - clear glossary
                if hasattr(self, 'auto_loaded_glossary_path'):
                    self.manual_glossary_path = None
                    self.auto_loaded_glossary_path = None
                    self.auto_loaded_glossary_for_file = None
                    self.append_log("📑 Multiple files selected - glossary auto-loading disabled")

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

    def configure_image_compression(self):
        """Open the image compression configuration dialog"""
        dialog, scrollable_frame, canvas = self.wm.setup_scrollable(
            self.master,
            "Image Compression Settings",
            width=None,
            height=None,
            max_width_ratio=0.6,
            max_height_ratio=1.2
        )
        
        # Main container with padding
        main_frame = tk.Frame(scrollable_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text="🗜️ Image Compression Settings", 
                              font=('TkDefaultFont', 14, 'bold'))
        title_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Enable compression toggle
        enable_frame = tk.Frame(main_frame)
        enable_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.enable_image_compression_var = tk.BooleanVar(
            value=self.config.get('enable_image_compression', False)
        )
        tb.Checkbutton(enable_frame, text="Enable Image Compression", 
                      variable=self.enable_image_compression_var,
                      bootstyle="round-toggle",
                      command=lambda: self._toggle_compression_options()).pack(anchor=tk.W)
        
        # Create container for all compression options
        self.compression_options_frame = tk.Frame(main_frame)
        self.compression_options_frame.pack(fill=tk.BOTH, expand=True)
        
        # Auto Compression Section
        auto_section = tk.LabelFrame(self.compression_options_frame, text="Automatic Compression", 
                                    padx=15, pady=10)
        auto_section.pack(fill=tk.X, pady=(0, 15))
        
        self.auto_compress_enabled_var = tk.BooleanVar(
            value=self.config.get('auto_compress_enabled', True)
        )
        tb.Checkbutton(auto_section, text="Auto-compress to fit token limits", 
                      variable=self.auto_compress_enabled_var,
                      bootstyle="round-toggle").pack(anchor=tk.W)
        
        # Token limit setting
        token_frame = tk.Frame(auto_section)
        token_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(token_frame, text="Target tokens per image:").pack(side=tk.LEFT)
        
        self.target_image_tokens_var = tk.StringVar(
            value=str(self.config.get('target_image_tokens', '1000'))
        )
        tb.Entry(token_frame, width=10, textvariable=self.target_image_tokens_var).pack(side=tk.LEFT, padx=(10, 0))
        
        tk.Label(token_frame, text="(Gemini uses ~258 tokens per image)", 
                font=('TkDefaultFont', 9), fg='gray').pack(side=tk.LEFT, padx=(10, 0))
        
        # Format Selection Section
        format_section = tk.LabelFrame(self.compression_options_frame, text="Output Format", 
                                      padx=15, pady=10)
        format_section.pack(fill=tk.X, pady=(0, 15))
        
        self.image_format_var = tk.StringVar(
            value=self.config.get('image_compression_format', 'auto')
        )
        
        formats = [
            ("Auto (Best quality/size ratio)", "auto"),
            ("WebP (Best compression)", "webp"),
            ("JPEG (Wide compatibility)", "jpeg"),
            ("PNG (Lossless)", "png")
        ]
        
        for text, value in formats:
            tb.Radiobutton(format_section, text=text, variable=self.image_format_var, 
                          value=value).pack(anchor=tk.W, pady=2)
        
        # Quality Settings Section
        quality_section = tk.LabelFrame(self.compression_options_frame, text="Quality Settings", 
                                       padx=15, pady=10)
        quality_section.pack(fill=tk.X, pady=(0, 15))
        
        # WebP Quality
        webp_frame = tk.Frame(quality_section)
        webp_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(webp_frame, text="WebP Quality:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        
        self.webp_quality_var = tk.IntVar(value=self.config.get('webp_quality', 85))
        webp_scale = tk.Scale(webp_frame, from_=1, to=100, orient=tk.HORIZONTAL, 
                             variable=self.webp_quality_var, length=200)
        webp_scale.pack(side=tk.LEFT, padx=(10, 10))
        
        self.webp_quality_label = tk.Label(webp_frame, text=f"{self.webp_quality_var.get()}%")
        self.webp_quality_label.pack(side=tk.LEFT)
        
        webp_scale.config(command=lambda v: self.webp_quality_label.config(text=f"{int(float(v))}%"))
        
        # JPEG Quality
        jpeg_frame = tk.Frame(quality_section)
        jpeg_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(jpeg_frame, text="JPEG Quality:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        
        self.jpeg_quality_var = tk.IntVar(value=self.config.get('jpeg_quality', 85))
        jpeg_scale = tk.Scale(jpeg_frame, from_=1, to=100, orient=tk.HORIZONTAL, 
                             variable=self.jpeg_quality_var, length=200)
        jpeg_scale.pack(side=tk.LEFT, padx=(10, 10))
        
        self.jpeg_quality_label = tk.Label(jpeg_frame, text=f"{self.jpeg_quality_var.get()}%")
        self.jpeg_quality_label.pack(side=tk.LEFT)
        
        jpeg_scale.config(command=lambda v: self.jpeg_quality_label.config(text=f"{int(float(v))}%"))
        
        # PNG Compression
        png_frame = tk.Frame(quality_section)
        png_frame.pack(fill=tk.X)
        
        tk.Label(png_frame, text="PNG Compression:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        
        self.png_compression_var = tk.IntVar(value=self.config.get('png_compression', 6))
        png_scale = tk.Scale(png_frame, from_=0, to=9, orient=tk.HORIZONTAL, 
                            variable=self.png_compression_var, length=200)
        png_scale.pack(side=tk.LEFT, padx=(10, 10))
        
        self.png_compression_label = tk.Label(png_frame, text=f"Level {self.png_compression_var.get()}")
        self.png_compression_label.pack(side=tk.LEFT)
        
        png_scale.config(command=lambda v: self.png_compression_label.config(text=f"Level {int(float(v))}"))
        
        # Resolution Limits Section
        resolution_section = tk.LabelFrame(self.compression_options_frame, text="Resolution Limits", 
                                          padx=15, pady=10)
        resolution_section.pack(fill=tk.X, pady=(0, 15))
        
        # Max dimension
        max_dim_frame = tk.Frame(resolution_section)
        max_dim_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(max_dim_frame, text="Max dimension (px):").pack(side=tk.LEFT)
        
        self.max_image_dimension_var = tk.StringVar(
            value=str(self.config.get('max_image_dimension', '2048'))
        )
        tb.Entry(max_dim_frame, width=10, textvariable=self.max_image_dimension_var).pack(side=tk.LEFT, padx=(10, 0))
        
        tk.Label(max_dim_frame, text="(Images larger than this will be resized)", 
                font=('TkDefaultFont', 9), fg='gray').pack(side=tk.LEFT, padx=(10, 0))
        
        # Max file size
        max_size_frame = tk.Frame(resolution_section)
        max_size_frame.pack(fill=tk.X)
        
        tk.Label(max_size_frame, text="Max file size (MB):").pack(side=tk.LEFT)
        
        self.max_image_size_mb_var = tk.StringVar(
            value=str(self.config.get('max_image_size_mb', '10'))
        )
        tb.Entry(max_size_frame, width=10, textvariable=self.max_image_size_mb_var).pack(side=tk.LEFT, padx=(10, 0))
        
        tk.Label(max_size_frame, text="(Larger files will be compressed)", 
                font=('TkDefaultFont', 9), fg='gray').pack(side=tk.LEFT, padx=(10, 0))
        
        # Advanced Options Section
        advanced_section = tk.LabelFrame(self.compression_options_frame, text="Advanced Options", 
                                        padx=15, pady=10)
        advanced_section.pack(fill=tk.X, pady=(0, 15))
        
        self.preserve_transparency_var = tk.BooleanVar(
            value=self.config.get('preserve_transparency', False)  # Changed default to False
        )
        tb.Checkbutton(advanced_section, text="Preserve transparency (PNG/WebP only)", 
                      variable=self.preserve_transparency_var).pack(anchor=tk.W, pady=2)
        
        self.preserve_original_format_var = tk.BooleanVar(
            value=self.config.get('preserve_original_format', False)
        )
        tb.Checkbutton(advanced_section, text="Preserve original image format", 
                      variable=self.preserve_original_format_var).pack(anchor=tk.W, pady=2)
        
        self.optimize_for_ocr_var = tk.BooleanVar(
            value=self.config.get('optimize_for_ocr', True)
        )
        tb.Checkbutton(advanced_section, text="Optimize for OCR (maintain text clarity)", 
                      variable=self.optimize_for_ocr_var).pack(anchor=tk.W, pady=2)
        
        self.progressive_encoding_var = tk.BooleanVar(
            value=self.config.get('progressive_encoding', True)
        )
        tb.Checkbutton(advanced_section, text="Progressive encoding (JPEG)", 
                      variable=self.progressive_encoding_var).pack(anchor=tk.W, pady=2)
        
        self.save_compressed_images_var = tk.BooleanVar(
            value=self.config.get('save_compressed_images', False)
        )
        tb.Checkbutton(advanced_section, text="Save compressed images to disk", 
                      variable=self.save_compressed_images_var).pack(anchor=tk.W, pady=2)
        
        # Info Section
        info_frame = tk.Frame(self.compression_options_frame)
        info_frame.pack(fill=tk.X)
        
        info_text = ("💡 Tips:\n"
                    "• WebP offers the best compression with good quality\n"
                    "• Use 'Auto' format for intelligent format selection\n"
                    "• Higher quality = larger file size\n"
                    "• OCR optimization maintains text readability")
        
        tk.Label(info_frame, text=info_text, justify=tk.LEFT, 
                font=('TkDefaultFont', 9), fg='#666').pack(anchor=tk.W)
        
        # Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        def save_image_compression():
            try:
                # Validate numeric inputs
                try:
                    int(self.target_image_tokens_var.get())
                    int(self.max_image_dimension_var.get())
                    float(self.max_image_size_mb_var.get())
                except ValueError:
                    messagebox.showerror("Invalid Input", "Please enter valid numbers for numeric fields")
                    return
                
                # Save all settings
                self.config['enable_image_compression'] = self.enable_image_compression_var.get()
                self.config['auto_compress_enabled'] = self.auto_compress_enabled_var.get()
                self.config['target_image_tokens'] = int(self.target_image_tokens_var.get())
                self.config['image_compression_format'] = self.image_format_var.get()
                self.config['webp_quality'] = self.webp_quality_var.get()
                self.config['jpeg_quality'] = self.jpeg_quality_var.get()
                self.config['png_compression'] = self.png_compression_var.get()
                self.config['max_image_dimension'] = int(self.max_image_dimension_var.get())
                self.config['max_image_size_mb'] = float(self.max_image_size_mb_var.get())
                self.config['preserve_transparency'] = self.preserve_transparency_var.get()
                self.config['preserve_original_format'] = self.preserve_original_format_var.get()
                self.config['optimize_for_ocr'] = self.optimize_for_ocr_var.get()
                self.config['progressive_encoding'] = self.progressive_encoding_var.get()
                self.config['save_compressed_images'] = self.save_compressed_images_var.get()
                
                self.append_log("✅ Image compression settings saved")
                dialog._cleanup_scrolling()
                dialog.destroy()
                
            except Exception as e:
                print(f"❌ Failed to save compression settings: {e}")
                messagebox.showerror("Error", f"Failed to save settings: {e}")
        
        tb.Button(button_frame, text="💾 Save Settings", command=save_image_compression, 
                 bootstyle="success", width=20).pack(side=tk.LEFT, padx=5)
        
        tb.Button(button_frame, text="❌ Cancel", 
                 command=lambda: [dialog._cleanup_scrolling(), dialog.destroy()], 
                 bootstyle="secondary", width=20).pack(side=tk.LEFT, padx=5)
        
        # Toggle function for enable/disable
        def _toggle_compression_options():
            state = tk.NORMAL if self.enable_image_compression_var.get() else tk.DISABLED
            for widget in self.compression_options_frame.winfo_children():
                if isinstance(widget, (tk.LabelFrame, tk.Frame)):
                    for child in widget.winfo_children():
                        if isinstance(child, (tb.Checkbutton, tb.Entry, tb.Radiobutton, tk.Scale)):
                            child.config(state=state)
                        elif isinstance(child, tk.Frame):
                            for subchild in child.winfo_children():
                                if isinstance(subchild, (tb.Checkbutton, tb.Entry, tb.Radiobutton, tk.Scale)):
                                    subchild.config(state=state)
        
        self._toggle_compression_options = _toggle_compression_options
        
        # Set initial state
        _toggle_compression_options()
        
        # Auto-resize and show
        self.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.6, max_height_ratio=1.2)
        
        dialog.protocol("WM_DELETE_WINDOW", lambda: [dialog._cleanup_scrolling(), dialog.destroy()])
    
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

    def toggle_thinking_budget(self):
        """Enable/disable thinking budget entry based on checkbox state"""
        if hasattr(self, 'thinking_budget_entry'):
            if self.enable_gemini_thinking_var.get():
                self.thinking_budget_entry.config(state='normal')
            else:
                self.thinking_budget_entry.config(state='disabled')

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
       
       # Section 6: Anti-Duplicate Parameters
       self._create_anti_duplicate_section(scrollable_frame)
       
       # Section 7: Custom API Endpoints (NEW)
       self._create_custom_api_endpoints_section(scrollable_frame)
       
       # Save & Close buttons
       self._create_settings_buttons(scrollable_frame, dialog, canvas)
       
       # Auto-resize and show
       self.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.78, max_height_ratio=1.82)
       
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

        ttk.Separator(section_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 10))
               
        
        tk.Label(section_frame, text="Application Updates:", font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W, pady=(5, 5))
        
        # Create a frame for update-related controls
        update_frame = tk.Frame(section_frame)
        update_frame.pack(anchor=tk.W, fill=tk.X)

        tb.Button(update_frame, text="🔄 Check for Updates", 
                 command=lambda: self.check_for_updates_manual(), 
                 bootstyle="info-outline",
                 width=25).pack(side=tk.LEFT, pady=2)

        # Add auto-update checkbox
        tb.Checkbutton(update_frame, text="Check on startup", 
                      variable=self.auto_update_check_var,
                      bootstyle="round-toggle").pack(side=tk.LEFT, padx=(10, 0))

        tk.Label(section_frame, text="Check GitHub for new Glossarion releases\nand download updates",
                font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 5))


    def _create_response_handling_section(self, parent):
        """Create response handling section with AI Hunter additions"""
        section_frame = tk.LabelFrame(parent, text="Response Handling & Retry Logic", padx=10, pady=10)
        section_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=5)
        
        # Add Thinking Tokens Toggle with Budget Control (NEW)
        tk.Label(section_frame, text="Gemini Thinking Mode", 
                font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W)
        
        thinking_frame = tk.Frame(section_frame)
        thinking_frame.pack(anchor=tk.W, padx=20, pady=(5, 0))
        
        tb.Checkbutton(thinking_frame, text="Enable Gemini Thinking", 
                      variable=self.enable_gemini_thinking_var,
                      bootstyle="round-toggle",
                      command=self.toggle_thinking_budget).pack(side=tk.LEFT)
        
        tk.Label(thinking_frame, text="Budget:").pack(side=tk.LEFT, padx=(20, 5))
        self.thinking_budget_entry = tb.Entry(thinking_frame, width=8, textvariable=self.thinking_budget_var)
        self.thinking_budget_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(thinking_frame, text="tokens").pack(side=tk.LEFT)
        
        tk.Label(section_frame, text="Control Gemini's thinking process. 0 = disabled,\n512-24576 = limited thinking, -1 = dynamic (auto)",
               font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
        
        # Add separator after thinking toggle
        ttk.Separator(section_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # ADD EXTRACTION WORKERS CONFIGURATION HERE
        tk.Label(section_frame, text="Parallel Extraction", 
                font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W)
        
        extraction_frame = tk.Frame(section_frame)
        extraction_frame.pack(anchor=tk.W, padx=20, pady=(5, 0))
        
        tb.Checkbutton(extraction_frame, text="Enable Parallel Processing", 
                      variable=self.enable_parallel_extraction_var,
                      bootstyle="round-toggle",
                      command=self.toggle_extraction_workers).pack(side=tk.LEFT)
        
        tk.Label(extraction_frame, text="Workers:").pack(side=tk.LEFT, padx=(20, 5))
        self.extraction_workers_entry = tb.Entry(extraction_frame, width=6, textvariable=self.extraction_workers_var)
        self.extraction_workers_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(extraction_frame, text="threads").pack(side=tk.LEFT)
        
        tk.Label(section_frame, text="Speed up EPUB extraction using multiple threads.\nRecommended: 4-8 workers (set to 1 to disable)",
               font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
        
        # Add separator after extraction workers
        ttk.Separator(section_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Multi API Key Management Section
        multi_key_frame = tk.Frame(section_frame)
        multi_key_frame.pack(anchor=tk.W, fill=tk.X, pady=(0, 15))
        
        # Multi-key indicator and button in same row
        multi_key_row = tk.Frame(multi_key_frame)
        multi_key_row.pack(fill=tk.X)
        
        # Show status if multi-key is enabled
        if self.config.get('use_multi_api_keys', False):
            multi_keys = self.config.get('multi_api_keys', [])
            active_keys = sum(1 for k in multi_keys if k.get('enabled', True))
            
            status_frame = tk.Frame(multi_key_row)
            status_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            tk.Label(status_frame, text="🔑 Multi-Key Mode:", 
                    font=('TkDefaultFont', 11, 'bold')).pack(side=tk.LEFT)
            
            tk.Label(status_frame, text=f"ACTIVE ({active_keys}/{len(multi_keys)} keys)", 
                    font=('TkDefaultFont', 11, 'bold'), fg='green').pack(side=tk.LEFT, padx=(5, 0))
        else:
            tk.Label(multi_key_row, text="🔑 Multi-Key Mode: DISABLED", 
                    font=('TkDefaultFont', 11), fg='gray').pack(side=tk.LEFT)
        
        # Multi API Key Manager button
        tb.Button(multi_key_row, text="Configure API Keys", 
                  command=self.open_multi_api_key_manager,
                  bootstyle="primary-outline",
                  width=20).pack(side=tk.RIGHT)
        
        tk.Label(section_frame, text="Manage multiple API keys with automatic rotation and rate limit handling",
                 font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
        
        # Add separator after Multi API Key section
        ttk.Separator(section_frame, orient='horizontal').pack(fill='x', pady=10)
     
        # Retry Truncated
        tb.Checkbutton(section_frame, text="Auto-retry Truncated Responses", 
                          variable=self.retry_truncated_var,
                          bootstyle="round-toggle").pack(anchor=tk.W)
        retry_frame = tk.Frame(section_frame)
        retry_frame.pack(anchor=tk.W, padx=20, pady=(5, 5))
        tk.Label(retry_frame, text="Token constraint:").pack(side=tk.LEFT)
        tb.Entry(retry_frame, width=8, textvariable=self.max_retry_tokens_var).pack(side=tk.LEFT, padx=5)
        tk.Label(section_frame, text="Retry when truncated. Acts as min/max constraint:\nbelow value = minimum, above value = maximum",
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
        tk.Label(compression_frame, text="(0.7-1.0)", font=('TkDefaultFont', 11)).pack(side=tk.LEFT)
        
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
        
        # Update thinking budget entry state based on initial toggle state
        self.toggle_thinking_budget()

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
                
    def toggle_gemini_endpoint(self):
        """Enable/disable Gemini endpoint entry based on toggle"""
        if self.use_gemini_openai_endpoint_var.get():
            self.gemini_endpoint_entry.config(state='normal')
        else:
            self.gemini_endpoint_entry.config(state='disabled')

    def open_multi_api_key_manager(self):
        """Open the multi API key manager dialog"""
        # Import here to avoid circular imports
        try:
            from multi_api_key_manager import MultiAPIKeyDialog
            
            # Create and show dialog
            dialog = MultiAPIKeyDialog(self.master, self)
            
            # Wait for dialog to close
            self.master.wait_window(dialog.dialog)
            
            # Refresh the settings display if in settings dialog
            if hasattr(self, 'current_settings_dialog'):
                # Close and reopen settings to refresh
                self.current_settings_dialog.destroy()
                self.show_settings()  # or open_other_settings()
                
        except ImportError as e:
            messagebox.showerror("Error", f"Failed to load Multi API Key Manager: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error opening Multi API Key Manager: {str(e)}")
            import traceback
            traceback.print_exc()

    def _create_multi_key_row(self, parent):
        """Create a compact multi-key configuration row"""
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)
        
        # Status indicator
        if self.config.get('use_multi_api_keys', False):
            keys = self.config.get('multi_api_keys', [])
            active = sum(1 for k in keys if k.get('enabled', True))
            
            # Checkbox to enable/disable
            tb.Checkbutton(frame, text="Multi API Key Mode", 
                          variable=self.use_multi_api_keys_var,
                          bootstyle="round-toggle",
                          command=self._toggle_multi_key_setting).pack(side=tk.LEFT)
            
            # Status
            tk.Label(frame, text=f"({active}/{len(keys)} active)", 
                    font=('TkDefaultFont', 10), fg='green').pack(side=tk.LEFT, padx=(5, 0))
        else:
            tb.Checkbutton(frame, text="Multi API Key Mode", 
                          variable=self.use_multi_api_keys_var,
                          bootstyle="round-toggle",
                          command=self._toggle_multi_key_setting).pack(side=tk.LEFT)
        
        # Configure button
        tb.Button(frame, text="Configure Keys...", 
                  command=self.open_multi_api_key_manager,
                  bootstyle="primary-outline").pack(side=tk.LEFT, padx=(20, 0))
        
        return frame
                
    def _toggle_multi_key_setting(self):
        """Toggle multi-key mode from settings dialog"""
        self.config['use_multi_api_keys'] = self.use_multi_api_keys_var.get()
        # Don't save immediately, let the dialog's save button handle it

    def toggle_extraction_workers(self):
        """Enable/disable extraction workers entry based on toggle"""
        if self.enable_parallel_extraction_var.get():
            self.extraction_workers_entry.config(state='normal')
            # Set environment variable
            os.environ["EXTRACTION_WORKERS"] = str(self.extraction_workers_var.get())
        else:
            self.extraction_workers_entry.config(state='disabled')
            # Set to 1 worker (sequential) when disabled
            os.environ["EXTRACTION_WORKERS"] = "1"
        
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
        
        # CHANGED: New button text and command
        tb.Button(title_frame, text="Configure All", 
                 command=self.metadata_batch_ui.configure_translation_prompts,
                 bootstyle="info-outline", width=12).pack(side=tk.LEFT, padx=(10, 5))
        
        # NEW: Custom Metadata Fields button
        tb.Button(title_frame, text="Custom Metadata", 
                 command=self.metadata_batch_ui.configure_metadata_fields,
                 bootstyle="info-outline", width=15).pack(side=tk.LEFT, padx=(5, 0))
        
        tk.Label(section_frame, text="When enabled: Book titles and selected metadata will be translated",
                    font=('TkDefaultFont', 11), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 10))
        
        # NEW: Batch Header Translation Section
        ttk.Separator(section_frame, orient='horizontal').pack(fill=tk.X, pady=(5, 10))
        
        tk.Label(section_frame, text="Chapter Header Translation:", 
                font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W, pady=(5, 5))
        
        header_frame = tk.Frame(section_frame)
        header_frame.pack(anchor=tk.W, fill=tk.X, pady=(5, 10))
        
        tb.Checkbutton(header_frame, text="Batch Translate Headers", 
                      variable=self.batch_translate_headers_var,
                      bootstyle="round-toggle").pack(side=tk.LEFT)
        
        tk.Label(header_frame, text="Headers per batch:").pack(side=tk.LEFT, padx=(20, 5))
        
        batch_entry = tk.Entry(header_frame, textvariable=self.headers_per_batch_var, width=10)
        batch_entry.pack(side=tk.LEFT)
        
        # Options for header translation
        update_frame = tk.Frame(section_frame)
        update_frame.pack(anchor=tk.W, fill=tk.X, padx=20)
        
        tb.Checkbutton(update_frame, text="Update headers in HTML files", 
                      variable=self.update_html_headers_var,
                      bootstyle="round-toggle").pack(side=tk.LEFT)
        
        tb.Checkbutton(update_frame, text="Save translations to .txt file", 
                      variable=self.save_header_translations_var,
                      bootstyle="round-toggle").pack(side=tk.LEFT, padx=(20, 0))
        
        tk.Label(section_frame, 
                text="• OFF: Use existing headers from translated chapters\n"
                     "• ON: Extract all headers → Translate in batch → Update files",
                font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(5, 10))
        
        # EPUB Validation (keep existing)
        ttk.Separator(section_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 10))
        
        tk.Label(section_frame, text="EPUB Utilities:", font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W, pady=(5, 5))
        
        tb.Button(section_frame, text="🔍 Validate EPUB Structure", 
                 command=self.validate_epub_structure_gui, 
                 bootstyle="success-outline",
                 width=25).pack(anchor=tk.W, pady=2)
        
        tk.Label(section_frame, text="Check if all required EPUB files are present for compilation",
                font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 5))
        
        # NCX-only navigation toggle
        tb.Checkbutton(section_frame, text="Use NCX-only Navigation (Compatibility Mode)", 
                      variable=self.force_ncx_only_var,
                      bootstyle="round-toggle").pack(anchor=tk.W, pady=(5, 5))
                      
        # CSS Attachment toggle - NEW!
        tb.Checkbutton(section_frame, text="Attach CSS to Chapters (Fixes styling issues)", 
                      variable=self.attach_css_to_chapters_var,
              bootstyle="round-toggle").pack(anchor=tk.W, pady=(5, 5))      

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
        
        # === CHAPTER EXTRACTION SETTINGS ===
        # Main extraction frame
        extraction_frame = tk.LabelFrame(section_frame, text="Chapter Extraction Settings", padx=10, pady=5)
        extraction_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Initialize variables if not exists
        if not hasattr(self, 'text_extraction_method_var'):
            # Check if using old enhanced mode
            if self.config.get('extraction_mode') == 'enhanced':
                self.text_extraction_method_var = tk.StringVar(value='enhanced')
                # Set filtering from enhanced_filtering or default to smart
                self.file_filtering_level_var = tk.StringVar(
                    value=self.config.get('enhanced_filtering', 'smart')
                )
            else:
                self.text_extraction_method_var = tk.StringVar(value='standard')
                self.file_filtering_level_var = tk.StringVar(
                    value=self.config.get('extraction_mode', 'smart')
                )
        
        if not hasattr(self, 'enhanced_preserve_structure_var'):
            self.enhanced_preserve_structure_var = tk.BooleanVar(
                value=self.config.get('enhanced_preserve_structure', True)
            )
        
        # --- Text Extraction Method Section ---
        method_frame = tk.Frame(extraction_frame)
        method_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(method_frame, text="Text Extraction Method:", 
                font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        # Standard extraction
        tb.Radiobutton(method_frame, text="Standard (BeautifulSoup)", 
                      variable=self.text_extraction_method_var, value="standard",
                      bootstyle="round-toggle",
                      command=self.on_extraction_method_change).pack(anchor=tk.W, pady=2)
        
        tk.Label(method_frame, text="Traditional HTML parsing - fast and reliable",
                font=('TkDefaultFont', 9), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        # Enhanced extraction
        tb.Radiobutton(method_frame, text="🚀 Enhanced (html2text)", 
                      variable=self.text_extraction_method_var, value="enhanced",
                      bootstyle="success-round-toggle",
                      command=self.on_extraction_method_change).pack(anchor=tk.W, pady=2)

        tk.Label(method_frame, text="Superior Unicode handling, cleaner text extraction",
                font=('TkDefaultFont', 9), fg='dark green', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        # Enhanced options (shown when enhanced is selected)
        self.enhanced_options_frame = tk.Frame(method_frame)
        self.enhanced_options_frame.pack(fill=tk.X, padx=20, pady=(5, 0))
        
        # Structure preservation
        tb.Checkbutton(self.enhanced_options_frame, text="Preserve Markdown Structure", 
                      variable=self.enhanced_preserve_structure_var,
                      bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
        
        tk.Label(self.enhanced_options_frame, text="Keep formatting (bold, headers, lists) for better AI context",
                font=('TkDefaultFont', 8), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 3))
        
        # Requirements note
        requirements_frame = tk.Frame(self.enhanced_options_frame)
        requirements_frame.pack(anchor=tk.W, pady=(5, 0))
        
        # Separator
        ttk.Separator(method_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 10))
        
        # --- File Filtering Level Section ---
        filtering_frame = tk.Frame(extraction_frame)
        filtering_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(filtering_frame, text="File Filtering Level:", 
                font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        # Smart filtering
        tb.Radiobutton(filtering_frame, text="Smart (Aggressive Filtering)", 
                      variable=self.file_filtering_level_var, value="smart",
                      bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
        
        tk.Label(filtering_frame, text="Skips navigation, TOC, copyright files\nBest for clean EPUBs with clear chapter structure",
                font=('TkDefaultFont', 9), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        # Comprehensive filtering
        tb.Radiobutton(filtering_frame, text="Comprehensive (Moderate Filtering)", 
                      variable=self.file_filtering_level_var, value="comprehensive",
                      bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
        
        tk.Label(filtering_frame, text="Only skips obvious navigation files\nGood when Smart mode misses chapters",
                font=('TkDefaultFont', 9), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        # Full extraction
        tb.Radiobutton(filtering_frame, text="Full (No Filtering)", 
                      variable=self.file_filtering_level_var, value="full",
                      bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
        
        tk.Label(filtering_frame, text="Extracts ALL HTML/XHTML files\nUse when other modes skip important content",
                font=('TkDefaultFont', 9), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        # Chapter merging option
        ttk.Separator(extraction_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 10))
        
        # Initialize disable_chapter_merging_var if not exists
        if not hasattr(self, 'disable_chapter_merging_var'):
            self.disable_chapter_merging_var = tk.BooleanVar(
                value=self.config.get('disable_chapter_merging', False)
            )
        
        tb.Checkbutton(extraction_frame, text="Disable Chapter Merging", 
                      variable=self.disable_chapter_merging_var,
                      bootstyle="round-toggle").pack(anchor=tk.W, pady=2)
        
        tk.Label(extraction_frame, text="Disable automatic merging of Section/Chapter pairs.\nEach file will be treated as a separate chapter.",
                font=('TkDefaultFont', 9), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        # === REMAINING OPTIONS ===
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
                
        # Chapter number offset
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
                
        # Add separator before API safety settings
        ttk.Separator(section_frame, orient='horizontal').pack(fill=tk.X, pady=(15, 10))
        
        # API Safety Settings subsection
        tk.Label(section_frame, text="API Safety Settings", 
                 font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W, pady=(5, 5))
        
        # Create the Gemini safety checkbox
        if not hasattr(self, 'disable_gemini_safety_var'):
            self.disable_gemini_safety_var = tk.BooleanVar(
                value=self.config.get('disable_gemini_safety', False)
            )
        
        tb.Checkbutton(
            section_frame,
            text="Disable Gemini API Safety Filters",
            variable=self.disable_gemini_safety_var,
            bootstyle="round-toggle"
        ).pack(anchor=tk.W, pady=(5, 0))
        
        # Add warning text
        warning_text = ("⚠️ Disables ALL content safety filters for Gemini models.\n"
                       "This sets all harm categories to BLOCK_NONE.\n")
        tk.Label(
            section_frame,
            text=warning_text,
            font=('TkDefaultFont', 9),
            fg='#ff6b6b',
            justify=tk.LEFT
        ).pack(anchor=tk.W, padx=(20, 0), pady=(0, 5))
        
        # Add note about affected models
        tk.Label(
            section_frame,
            text="Does NOT affect ElectronHub Gemini models (eh/gemini-*)",
            font=('TkDefaultFont', 8),
            fg='gray',
            justify=tk.LEFT
        ).pack(anchor=tk.W, padx=(20, 0))
        
        # Initial state - show/hide enhanced options
        self.on_extraction_method_change()

    def on_extraction_method_change(self):
        """Handle extraction method changes and show/hide Enhanced options"""
        if hasattr(self, 'text_extraction_method_var') and hasattr(self, 'enhanced_options_frame'):
            if self.text_extraction_method_var.get() == 'enhanced':
                self.enhanced_options_frame.pack(fill=tk.X, padx=20, pady=(5, 0))
            else:
                self.enhanced_options_frame.pack_forget()
                
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

        # Buttons for prompts and compression
        tb.Button(settings_frame, text="Image Chunk Prompt", 
                 command=self.configure_image_chunk_prompt,
                 bootstyle="info-outline", width=20).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        
        # Add Image Compression button
        tb.Button(settings_frame, text="🗜️ Image Compression", 
                 command=self.configure_image_compression,
                 bootstyle="info-outline", width=25).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        # Add the toggle here in the right column with some spacing
        tk.Frame(right_column, height=15).pack()  # Add some spacing

        tb.Checkbutton(right_column, text="Send tall image chunks in single API call (NOT RECOMMENDED)", 
                      variable=self.single_api_image_chunks_var,
                      bootstyle="round-toggle").pack(anchor=tk.W)

        tk.Label(right_column, text="All image chunks sent to 1 API call (Most AI models don't like this)",
                font=('TkDefaultFont', 10), fg='gray').pack(anchor=tk.W, padx=20, pady=(0, 10))

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
        
    def on_extraction_mode_change(self):
        """Handle extraction mode changes and show/hide Enhanced options"""
        if self.extraction_mode_var.get() == 'enhanced':
            # Show enhanced options
            if hasattr(self, 'enhanced_options_separator'):
                self.enhanced_options_separator.pack(fill=tk.X, pady=(5, 5))
            if hasattr(self, 'enhanced_options_frame'):
                self.enhanced_options_frame.pack(fill=tk.X, padx=20)
        else:
            # Hide enhanced options
            if hasattr(self, 'enhanced_options_separator'):
                self.enhanced_options_separator.pack_forget()
            if hasattr(self, 'enhanced_options_frame'):
                self.enhanced_options_frame.pack_forget()
                
    def _create_anti_duplicate_section(self, parent):
        """Create comprehensive anti-duplicate parameter controls with tabs"""
        # Anti-Duplicate Parameters section
        ad_frame = tk.LabelFrame(parent, text="🎯 Anti-Duplicate Parameters", padx=15, pady=10)
        ad_frame.grid(row=6, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 15))
        
        # Description
        desc_label = tk.Label(ad_frame, 
            text="Configure parameters to reduce duplicate translations across all AI providers.",
            font=('TkDefaultFont', 9), fg='gray', wraplength=520)
        desc_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Enable/Disable toggle
        self.enable_anti_duplicate_var = tk.BooleanVar(value=self.config.get('enable_anti_duplicate', False))
        enable_cb = tb.Checkbutton(ad_frame, text="Enable Anti-Duplicate Parameters", 
                                  variable=self.enable_anti_duplicate_var,
                                  command=self._toggle_anti_duplicate_controls)
        enable_cb.pack(anchor=tk.W, pady=(0, 10))
        
        # Create notebook for organized parameters
        self.anti_duplicate_notebook = ttk.Notebook(ad_frame)
        self.anti_duplicate_notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Tab 1: Core Parameters
        core_frame = tk.Frame(self.anti_duplicate_notebook)
        self.anti_duplicate_notebook.add(core_frame, text="Core Parameters")
        
        # Top-P (Nucleus Sampling)
        top_p_frame = tk.Frame(core_frame)
        top_p_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(top_p_frame, text="Top-P (Nucleus Sampling):", width=25, anchor='w').pack(side=tk.LEFT)
        self.top_p_var = tk.DoubleVar(value=self.config.get('top_p', 1.0))
        top_p_scale = tk.Scale(top_p_frame, from_=0.1, to=1.0, resolution=0.01, 
                              orient=tk.HORIZONTAL, variable=self.top_p_var, length=200)
        top_p_scale.pack(side=tk.LEFT, padx=5)
        self.top_p_value_label = tk.Label(top_p_frame, text="", width=8)
        self.top_p_value_label.pack(side=tk.LEFT, padx=5)
        
        def update_top_p_label(*args):
            val = self.top_p_var.get()
            self.top_p_value_label.config(text=f"{val:.2f}")
        self.top_p_var.trace('w', update_top_p_label)
        update_top_p_label()
        
        # Top-K (Vocabulary Limit)
        top_k_frame = tk.Frame(core_frame)
        top_k_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(top_k_frame, text="Top-K (Vocabulary Limit):", width=25, anchor='w').pack(side=tk.LEFT)
        self.top_k_var = tk.IntVar(value=self.config.get('top_k', 0))
        top_k_scale = tk.Scale(top_k_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                              variable=self.top_k_var, length=200)
        top_k_scale.pack(side=tk.LEFT, padx=5)
        self.top_k_value_label = tk.Label(top_k_frame, text="", width=8)
        self.top_k_value_label.pack(side=tk.LEFT, padx=5)
        
        def update_top_k_label(*args):
            val = self.top_k_var.get()
            self.top_k_value_label.config(text=f"{val}" if val > 0 else "OFF")
        self.top_k_var.trace('w', update_top_k_label)
        update_top_k_label()
        
        # Frequency Penalty
        freq_penalty_frame = tk.Frame(core_frame)
        freq_penalty_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(freq_penalty_frame, text="Frequency Penalty:", width=25, anchor='w').pack(side=tk.LEFT)
        self.frequency_penalty_var = tk.DoubleVar(value=self.config.get('frequency_penalty', 0.0))
        freq_scale = tk.Scale(freq_penalty_frame, from_=0.0, to=2.0, resolution=0.1, 
                             orient=tk.HORIZONTAL, variable=self.frequency_penalty_var, length=200)
        freq_scale.pack(side=tk.LEFT, padx=5)
        self.freq_penalty_value_label = tk.Label(freq_penalty_frame, text="", width=8)
        self.freq_penalty_value_label.pack(side=tk.LEFT, padx=5)
        
        def update_freq_label(*args):
            val = self.frequency_penalty_var.get()
            self.freq_penalty_value_label.config(text=f"{val:.1f}" if val > 0 else "OFF")
        self.frequency_penalty_var.trace('w', update_freq_label)
        update_freq_label()
        
        # Presence Penalty
        pres_penalty_frame = tk.Frame(core_frame)
        pres_penalty_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(pres_penalty_frame, text="Presence Penalty:", width=25, anchor='w').pack(side=tk.LEFT)
        self.presence_penalty_var = tk.DoubleVar(value=self.config.get('presence_penalty', 0.0))
        pres_scale = tk.Scale(pres_penalty_frame, from_=0.0, to=2.0, resolution=0.1, 
                             orient=tk.HORIZONTAL, variable=self.presence_penalty_var, length=200)
        pres_scale.pack(side=tk.LEFT, padx=5)
        self.pres_penalty_value_label = tk.Label(pres_penalty_frame, text="", width=8)
        self.pres_penalty_value_label.pack(side=tk.LEFT, padx=5)
        
        def update_pres_label(*args):
            val = self.presence_penalty_var.get()
            self.pres_penalty_value_label.config(text=f"{val:.1f}" if val > 0 else "OFF")
        self.presence_penalty_var.trace('w', update_pres_label)
        update_pres_label()
        
        # Tab 2: Advanced Parameters
        advanced_frame = tk.Frame(self.anti_duplicate_notebook)
        self.anti_duplicate_notebook.add(advanced_frame, text="Advanced")
        
        # Repetition Penalty
        rep_penalty_frame = tk.Frame(advanced_frame)
        rep_penalty_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(rep_penalty_frame, text="Repetition Penalty:", width=25, anchor='w').pack(side=tk.LEFT)
        self.repetition_penalty_var = tk.DoubleVar(value=self.config.get('repetition_penalty', 1.0))
        rep_scale = tk.Scale(rep_penalty_frame, from_=1.0, to=2.0, resolution=0.05, 
                            orient=tk.HORIZONTAL, variable=self.repetition_penalty_var, length=200)
        rep_scale.pack(side=tk.LEFT, padx=5)
        self.rep_penalty_value_label = tk.Label(rep_penalty_frame, text="", width=8)
        self.rep_penalty_value_label.pack(side=tk.LEFT, padx=5)
        
        def update_rep_label(*args):
            val = self.repetition_penalty_var.get()
            self.rep_penalty_value_label.config(text=f"{val:.2f}" if val > 1.0 else "OFF")
        self.repetition_penalty_var.trace('w', update_rep_label)
        update_rep_label()
        
        # Candidate Count (Gemini)
        candidate_frame = tk.Frame(advanced_frame)
        candidate_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(candidate_frame, text="Candidate Count (Gemini):", width=25, anchor='w').pack(side=tk.LEFT)
        self.candidate_count_var = tk.IntVar(value=self.config.get('candidate_count', 1))
        candidate_scale = tk.Scale(candidate_frame, from_=1, to=4, orient=tk.HORIZONTAL, 
                                  variable=self.candidate_count_var, length=200)
        candidate_scale.pack(side=tk.LEFT, padx=5)
        self.candidate_value_label = tk.Label(candidate_frame, text="", width=8)
        self.candidate_value_label.pack(side=tk.LEFT, padx=5)
        
        def update_candidate_label(*args):
            val = self.candidate_count_var.get()
            self.candidate_value_label.config(text=f"{val}")
        self.candidate_count_var.trace('w', update_candidate_label)
        update_candidate_label()
        
        # Tab 3: Stop Sequences
        stop_frame = tk.Frame(self.anti_duplicate_notebook)
        self.anti_duplicate_notebook.add(stop_frame, text="Stop Sequences")
        
        # Custom Stop Sequences
        stop_seq_frame = tk.Frame(stop_frame)
        stop_seq_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(stop_seq_frame, text="Custom Stop Sequences:", width=25, anchor='w').pack(side=tk.LEFT)
        self.custom_stop_sequences_var = tk.StringVar(value=self.config.get('custom_stop_sequences', ''))
        stop_entry = tb.Entry(stop_seq_frame, textvariable=self.custom_stop_sequences_var, width=30)
        stop_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(stop_seq_frame, text="(comma-separated)", font=('TkDefaultFont', 8), fg='gray').pack(side=tk.LEFT)
        
        # Tab 4: Logit Bias (OpenAI)
        bias_frame = tk.Frame(self.anti_duplicate_notebook)
        self.anti_duplicate_notebook.add(bias_frame, text="Logit Bias")
        
        # Logit Bias Enable
        self.logit_bias_enabled_var = tk.BooleanVar(value=self.config.get('logit_bias_enabled', False))
        bias_cb = tb.Checkbutton(bias_frame, text="Enable Logit Bias (OpenAI only)", 
                                variable=self.logit_bias_enabled_var)
        bias_cb.pack(anchor=tk.W, pady=5)
        
        # Logit Bias Strength
        bias_strength_frame = tk.Frame(bias_frame)
        bias_strength_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(bias_strength_frame, text="Bias Strength:", width=25, anchor='w').pack(side=tk.LEFT)
        self.logit_bias_strength_var = tk.DoubleVar(value=self.config.get('logit_bias_strength', -0.5))
        bias_scale = tk.Scale(bias_strength_frame, from_=-2.0, to=2.0, resolution=0.1, 
                             orient=tk.HORIZONTAL, variable=self.logit_bias_strength_var, length=200)
        bias_scale.pack(side=tk.LEFT, padx=5)
        self.bias_strength_value_label = tk.Label(bias_strength_frame, text="", width=8)
        self.bias_strength_value_label.pack(side=tk.LEFT, padx=5)
        
        def update_bias_strength_label(*args):
            val = self.logit_bias_strength_var.get()
            self.bias_strength_value_label.config(text=f"{val:.1f}")
        self.logit_bias_strength_var.trace('w', update_bias_strength_label)
        update_bias_strength_label()
        
        # Preset bias targets
        preset_frame = tk.Frame(bias_frame)
        preset_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(preset_frame, text="Preset Bias Targets:", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W)
        
        self.bias_common_words_var = tk.BooleanVar(value=self.config.get('bias_common_words', False))
        tb.Checkbutton(preset_frame, text="Bias against common words (the, and, said)", 
                      variable=self.bias_common_words_var).pack(anchor=tk.W)
        
        self.bias_repetitive_phrases_var = tk.BooleanVar(value=self.config.get('bias_repetitive_phrases', False))
        tb.Checkbutton(preset_frame, text="Bias against repetitive phrases", 
                      variable=self.bias_repetitive_phrases_var).pack(anchor=tk.W)
        
        # Provider compatibility info
        compat_frame = tk.Frame(ad_frame)
        compat_frame.pack(fill=tk.X, pady=(15, 0))

        tk.Label(compat_frame, text="Parameter Compatibility:", 
                font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W)

        compat_text = tk.Label(compat_frame, 
            text="• Core: Most providers • Advanced: DeepSeek, Mistral, Groq • Logit Bias: OpenAI only",
            font=('TkDefaultFont', 8), fg='gray', justify=tk.LEFT)
        compat_text.pack(anchor=tk.W, pady=(5, 0))

        # Reset button
        reset_frame = tk.Frame(ad_frame)
        reset_frame.pack(fill=tk.X, pady=(10, 0))

        tb.Button(reset_frame, text="🔄 Reset to Defaults", 
                 command=self._reset_anti_duplicate_defaults,
                 bootstyle="secondary", width=20).pack(side=tk.LEFT)

        tk.Label(reset_frame, text="Reset all anti-duplicate parameters to default values", 
                font=('TkDefaultFont', 8), fg='gray').pack(side=tk.LEFT, padx=(10, 0))

        # Store all tab frames for enable/disable
        self.anti_duplicate_tabs = [core_frame, advanced_frame, stop_frame, bias_frame]

        # Initial state
        self._toggle_anti_duplicate_controls()

    def _toggle_anti_duplicate_controls(self):
        """Enable/disable anti-duplicate parameter controls"""
        state = tk.NORMAL if self.enable_anti_duplicate_var.get() else tk.DISABLED
        
        # Disable/enable the notebook itself
        if hasattr(self, 'anti_duplicate_notebook'):
            try:
                self.anti_duplicate_notebook.config(state=state)
            except tk.TclError:
                pass
        
        # Disable/enable all controls in tabs
        if hasattr(self, 'anti_duplicate_tabs'):
            for tab_frame in self.anti_duplicate_tabs:
                for widget in tab_frame.winfo_children():
                    for child in widget.winfo_children():
                        if hasattr(child, 'config'):
                            try:
                                child.config(state=state)
                            except tk.TclError:
                                pass
                                
    def _reset_anti_duplicate_defaults(self):
        """Reset all anti-duplicate parameters to their default values"""
        import tkinter.messagebox as messagebox
        
        # Ask for confirmation
        if not messagebox.askyesno("Reset Anti-Duplicate Parameters", 
                                  "Are you sure you want to reset all anti-duplicate parameters to their default values?"):
            return
        
        # Reset all variables to defaults
        if hasattr(self, 'enable_anti_duplicate_var'):
            self.enable_anti_duplicate_var.set(False)
        
        if hasattr(self, 'top_p_var'):
            self.top_p_var.set(1.0)  # Default = no effect
        
        if hasattr(self, 'top_k_var'):
            self.top_k_var.set(0)  # Default = disabled
        
        if hasattr(self, 'frequency_penalty_var'):
            self.frequency_penalty_var.set(0.0)  # Default = no penalty
        
        if hasattr(self, 'presence_penalty_var'):
            self.presence_penalty_var.set(0.0)  # Default = no penalty
        
        if hasattr(self, 'repetition_penalty_var'):
            self.repetition_penalty_var.set(1.0)  # Default = no penalty
        
        if hasattr(self, 'candidate_count_var'):
            self.candidate_count_var.set(1)  # Default = single response
        
        if hasattr(self, 'custom_stop_sequences_var'):
            self.custom_stop_sequences_var.set("")  # Default = empty
        
        if hasattr(self, 'logit_bias_enabled_var'):
            self.logit_bias_enabled_var.set(False)  # Default = disabled
        
        if hasattr(self, 'logit_bias_strength_var'):
            self.logit_bias_strength_var.set(-0.5)  # Default strength
        
        if hasattr(self, 'bias_common_words_var'):
            self.bias_common_words_var.set(False)  # Default = disabled
        
        if hasattr(self, 'bias_repetitive_phrases_var'):
            self.bias_repetitive_phrases_var.set(False)  # Default = disabled
        
        # Update enable/disable state
        self._toggle_anti_duplicate_controls()
        
        # Show success message
        messagebox.showinfo("Reset Complete", "All anti-duplicate parameters have been reset to their default values.")
        
        # Log the reset
        if hasattr(self, 'append_log'):
            self.append_log("🔄 Anti-duplicate parameters reset to defaults")        

    def _create_custom_api_endpoints_section(self, parent_frame):
        """Create the Custom API Endpoints section"""
        # Custom API Endpoints Section
        endpoints_frame = tb.LabelFrame(parent_frame, text="Custom API Endpoints", padding=10)
        endpoints_frame.grid(row=7, column=0, columnspan=2, sticky=tk.NSEW, padx=5, pady=5)
        
        # Checkbox to enable/disable custom endpoint (MOVED TO TOP)
        custom_endpoint_checkbox_frame = tb.Frame(endpoints_frame)
        custom_endpoint_checkbox_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        self.use_custom_endpoint_checkbox = tb.Checkbutton(
            custom_endpoint_checkbox_frame,
            text="Enable Custom OpenAI Endpoint",
            variable=self.use_custom_openai_endpoint_var,
            command=self.toggle_custom_endpoint_ui,
            bootstyle="primary"
        )
        self.use_custom_endpoint_checkbox.pack(side=tk.LEFT)

        # Main OpenAI Base URL
        openai_url_frame = tb.Frame(endpoints_frame)
        openai_url_frame.pack(fill=tk.X, padx=5, pady=5)

        tb.Label(openai_url_frame, text="Override API Endpoint:").pack(side=tk.LEFT, padx=(0, 5))
        self.openai_base_url_var = tk.StringVar(value=self.config.get('openai_base_url', ''))
        self.openai_base_url_entry = tb.Entry(openai_url_frame, textvariable=self.openai_base_url_var, width=50)
        self.openai_base_url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Clear button
        self.openai_clear_button = tb.Button(openai_url_frame, text="Clear", 
                 command=lambda: self.openai_base_url_var.set(""),
                 bootstyle="secondary", width=8)
        self.openai_clear_button.pack(side=tk.LEFT)
        
        # Set initial state based on checkbox
        if not self.use_custom_openai_endpoint_var.get():
            self.openai_base_url_entry.configure(state='disabled')
            self.openai_clear_button.configure(state='disabled')

        # Help text for main field
        help_text = tb.Label(endpoints_frame, 
                            text="Enable checkbox to use custom endpoint. For Ollama: http://localhost:11434/v1",
                            font=('TkDefaultFont', 8), foreground='gray')
        help_text.pack(anchor=tk.W, padx=5, pady=(0, 10))

        # Show More Fields button
        self.show_more_endpoints = False
        self.more_fields_button = tb.Button(endpoints_frame, 
                                           text="▼ Show More Fields", 
                                           command=self.toggle_more_endpoints,
                                           bootstyle="link")
        self.more_fields_button.pack(anchor=tk.W, padx=5, pady=5)

        # Container for additional fields (initially hidden)
        self.additional_endpoints_frame = tb.Frame(endpoints_frame)
        # Don't pack it initially - it's hidden

        # Inside the additional_endpoints_frame:
        # Groq/Local Base URL
        groq_url_frame = tb.Frame(self.additional_endpoints_frame)
        groq_url_frame.pack(fill=tk.X, padx=5, pady=5)

        tb.Label(groq_url_frame, text="Groq/Local Base URL:").pack(side=tk.LEFT, padx=(0, 5))
        self.groq_base_url_var = tk.StringVar(value=self.config.get('groq_base_url', ''))
        self.groq_base_url_entry = tb.Entry(groq_url_frame, textvariable=self.groq_base_url_var, width=50)
        self.groq_base_url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        tb.Button(groq_url_frame, text="Clear", 
                 command=lambda: self.groq_base_url_var.set(""),
                 bootstyle="secondary", width=8).pack(side=tk.LEFT)

        groq_help = tb.Label(self.additional_endpoints_frame, 
                            text="For vLLM: http://localhost:8000/v1 | For LM Studio: http://localhost:1234/v1",
                            font=('TkDefaultFont', 8), foreground='gray')
        groq_help.pack(anchor=tk.W, padx=5, pady=(0, 5))

        # Fireworks Base URL
        fireworks_url_frame = tb.Frame(self.additional_endpoints_frame)
        fireworks_url_frame.pack(fill=tk.X, padx=5, pady=5)

        tb.Label(fireworks_url_frame, text="Fireworks Base URL:").pack(side=tk.LEFT, padx=(0, 5))
        self.fireworks_base_url_var = tk.StringVar(value=self.config.get('fireworks_base_url', ''))
        self.fireworks_base_url_entry = tb.Entry(fireworks_url_frame, textvariable=self.fireworks_base_url_var, width=50)
        self.fireworks_base_url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        tb.Button(fireworks_url_frame, text="Clear", 
                 command=lambda: self.fireworks_base_url_var.set(""),
                 bootstyle="secondary", width=8).pack(side=tk.LEFT)

        # Info about multiple endpoints
        info_frame = tb.Frame(self.additional_endpoints_frame)
        info_frame.pack(fill=tk.X, padx=5, pady=10)

        info_text = """💡 Advanced: Use multiple endpoints to run different local LLM servers simultaneously.
        • Use model prefix 'groq/' to route through Groq endpoint
        • Use model prefix 'fireworks/' to route through Fireworks endpoint
        • Most users only need the main OpenAI endpoint above"""

        tb.Label(info_frame, text=info_text, 
                font=('TkDefaultFont', 8), foreground='#0dcaf0',  # Light blue color
                wraplength=600, justify=tk.LEFT).pack(anchor=tk.W)

        # Test Connection button (always visible)
        test_button = tb.Button(endpoints_frame, text="Test Connection", 
                               command=self.test_api_connections,
                               bootstyle="info")
        test_button.pack(pady=10)

        # Gemini OpenAI-Compatible Endpoint (inside additional_endpoints_frame)
        gemini_frame = tb.Frame(self.additional_endpoints_frame)
        gemini_frame.pack(fill=tk.X, padx=5, pady=5)

        # Checkbox for enabling Gemini endpoint
        self.gemini_checkbox = tb.Checkbutton(
            gemini_frame,
            text="Enable Gemini OpenAI-Compatible Endpoint",
            variable=self.use_gemini_openai_endpoint_var,
            command=self.toggle_gemini_endpoint,  # Add the command
            bootstyle="primary"
        )
        self.gemini_checkbox.pack(anchor=tk.W, pady=(5, 5))

        # Gemini endpoint URL input
        gemini_url_frame = tb.Frame(self.additional_endpoints_frame)
        gemini_url_frame.pack(fill=tk.X, padx=5, pady=5)

        tb.Label(gemini_url_frame, text="Gemini OpenAI Endpoint:").pack(side=tk.LEFT, padx=(0, 5))
        self.gemini_endpoint_entry = tb.Entry(gemini_url_frame, textvariable=self.gemini_openai_endpoint_var, width=50)
        self.gemini_endpoint_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.gemini_clear_button = tb.Button(gemini_url_frame, text="Clear", 
                 command=lambda: self.gemini_openai_endpoint_var.set(""),
                 bootstyle="secondary", width=8)
        self.gemini_clear_button.pack(side=tk.LEFT)

        # Help text
        gemini_help = tb.Label(self.additional_endpoints_frame, 
                              text="For Gemini rate limit optimization with proxy services (e.g., OpenRouter, LiteLLM)",
                              font=('TkDefaultFont', 8), foreground='gray')
        gemini_help.pack(anchor=tk.W, padx=5, pady=(0, 5))

        # Set initial state based on checkbox
        if not self.use_gemini_openai_endpoint_var.get():
            self.gemini_endpoint_entry.configure(state='disabled')
            self.gemini_clear_button.configure(state='disabled')

    def toggle_gemini_endpoint(self):
        """Enable/disable Gemini endpoint entry based on toggle"""
        if self.use_gemini_openai_endpoint_var.get():
            self.gemini_endpoint_entry.configure(state='normal')
            self.gemini_clear_button.configure(state='normal')
        else:
            self.gemini_endpoint_entry.configure(state='disabled')
            self.gemini_clear_button.configure(state='disabled')
        
    def toggle_custom_endpoint_ui(self):
        """Enable/disable the OpenAI base URL entry and clear button based on checkbox"""
        if self.use_custom_openai_endpoint_var.get():
            self.openai_base_url_entry.configure(state='normal')
            self.openai_clear_button.configure(state='normal')
            print("✅ Custom OpenAI endpoint enabled")
        else:
            self.openai_base_url_entry.configure(state='disabled')
            self.openai_clear_button.configure(state='disabled')
            print("❌ Custom OpenAI endpoint disabled - using default OpenAI API")

    def toggle_more_endpoints(self):
        """Toggle visibility of additional endpoint fields"""
        self.show_more_endpoints = not self.show_more_endpoints
        
        if self.show_more_endpoints:
            self.additional_endpoints_frame.pack(fill=tk.BOTH, expand=True, after=self.more_fields_button)
            self.more_fields_button.configure(text="▲ Show Fewer Fields")
        else:
            self.additional_endpoints_frame.pack_forget()
            self.more_fields_button.configure(text="▼ Show More Fields")
        
        # Update dialog scrolling if needed
        if hasattr(self, 'current_dialog') and self.current_dialog:
            self.current_dialog.update_idletasks()
            self.current_dialog.canvas.configure(scrollregion=self.current_dialog.canvas.bbox("all"))
                 
    def test_api_connections(self):
        """Test all configured API connections"""
        # Show immediate feedback
        progress_dialog = tk.Toplevel(self.current_dialog if hasattr(self, 'current_dialog') else self.master)
        progress_dialog.title("Testing Connections...")
        
        # Set icon
        try:
            progress_dialog.iconbitmap("halgakos.ico")
        except:
            pass  # Icon setting failed, continue without icon
        
        # Center the dialog
        progress_dialog.update_idletasks()
        width = 300
        height = 150
        x = (progress_dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (progress_dialog.winfo_screenheight() // 2) - (height // 2)
        progress_dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        # Add progress message
        progress_label = tb.Label(progress_dialog, text="Testing API connections...\nPlease wait...", 
                                 font=('TkDefaultFont', 10))
        progress_label.pack(pady=50)
        
        # Force update to show dialog immediately
        progress_dialog.update()
        
        try:
            # Ensure we have the openai module
            import openai
        except ImportError:
            progress_dialog.destroy()
            messagebox.showerror("Error", "OpenAI library not installed")
            return
        
        # Get API key from the main GUI
        api_key = self.api_key_entry.get() if hasattr(self, 'api_key_entry') else self.config.get('api_key', '')
        if not api_key:
            api_key = "sk-dummy-key"  # For local models
        
        # Collect all configured endpoints
        endpoints_to_test = []
        
        # OpenAI endpoint - only test if checkbox is enabled
        if self.use_custom_openai_endpoint_var.get():
            openai_url = self.openai_base_url_var.get()
            if openai_url:
                endpoints_to_test.append(("OpenAI (Custom)", openai_url, self.model_var.get() if hasattr(self, 'model_var') else "gpt-3.5-turbo"))
            else:
                # Use default OpenAI endpoint if checkbox is on but no custom URL provided
                endpoints_to_test.append(("OpenAI (Default)", "https://api.openai.com/v1", self.model_var.get() if hasattr(self, 'model_var') else "gpt-3.5-turbo"))
        
        # Groq endpoint
        if hasattr(self, 'groq_base_url_var'):
            groq_url = self.groq_base_url_var.get()
            if groq_url:
                # For Groq, we need a groq-prefixed model
                current_model = self.model_var.get() if hasattr(self, 'model_var') else "llama-3-70b"
                groq_model = current_model if current_model.startswith('groq/') else current_model.replace('groq/', '')
                endpoints_to_test.append(("Groq/Local", groq_url, groq_model))
        
        # Fireworks endpoint
        if hasattr(self, 'fireworks_base_url_var'):
            fireworks_url = self.fireworks_base_url_var.get()
            if fireworks_url:
                # For Fireworks, we need the accounts/ prefix
                current_model = self.model_var.get() if hasattr(self, 'model_var') else "llama-v3-70b-instruct"
                fw_model = current_model if current_model.startswith('accounts/') else f"accounts/fireworks/models/{current_model.replace('fireworks/', '')}"
                endpoints_to_test.append(("Fireworks", fireworks_url, fw_model))
        
        # Gemini OpenAI-Compatible endpoint
        if hasattr(self, 'use_gemini_openai_endpoint_var') and self.use_gemini_openai_endpoint_var.get():
            gemini_url = self.gemini_openai_endpoint_var.get()
            if gemini_url:
                # Ensure the endpoint ends with /openai/ for compatibility
                if not gemini_url.endswith('/openai/'):
                    if gemini_url.endswith('/'):
                        gemini_url = gemini_url + 'openai/'
                    else:
                        gemini_url = gemini_url + '/openai/'
                
                # For Gemini OpenAI-compatible endpoints, use the current model or a suitable default
                current_model = self.model_var.get() if hasattr(self, 'model_var') else "gemini-2.0-flash-exp"
                # Remove any 'gemini/' prefix for the OpenAI-compatible endpoint
                gemini_model = current_model.replace('gemini/', '') if current_model.startswith('gemini/') else current_model
                endpoints_to_test.append(("Gemini (OpenAI-Compatible)", gemini_url, gemini_model))
        
        if not endpoints_to_test:
            messagebox.showinfo("Info", "No custom endpoints configured. Using default API endpoints.")
            return
        
        # Test each endpoint
        results = []
        for name, base_url, model in endpoints_to_test:
            try:
                # Create client for this endpoint
                test_client = openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=5.0  # Short timeout for testing
                )
                
                # Try a minimal completion
                response = test_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5
                )
                
                results.append(f"✅ {name}: Connected successfully! (Model: {model})")
            except Exception as e:
                error_msg = str(e)
                # Simplify common error messages
                if "404" in error_msg:
                    error_msg = "404 - Endpoint not found. Check URL and model name."
                elif "401" in error_msg or "403" in error_msg:
                    error_msg = "Authentication failed. Check API key."
                elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                    error_msg = f"Model '{model}' not found at this endpoint."
                
                results.append(f"❌ {name}: {error_msg}")
        
        # Show results
        result_message = "Connection Test Results:\n\n" + "\n\n".join(results)
        
        # Close progress dialog
        progress_dialog.destroy()
        
        # Determine if all succeeded
        all_success = all("✅" in r for r in results)
        
        if all_success:
            messagebox.showinfo("Success", result_message)
        else:
            messagebox.showwarning("Test Results", result_message)
        
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
                    'attach_css_to_chapters': self.attach_css_to_chapters_var.get(),
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
                    'disable_chapter_merging': self.disable_chapter_merging_var.get(),
                    'disable_epub_gallery': self.disable_epub_gallery_var.get(),
                    'disable_zero_detection': self.disable_zero_detection_var.get(),
                    'enable_image_translation': self.enable_image_translation_var.get(),
                    'process_webnovel_images': self.process_webnovel_images_var.get(),
                    'hide_image_translation_label': self.hide_image_translation_label_var.get(),
                    'duplicate_detection_mode': self.duplicate_detection_mode_var.get(),
                    'chapter_number_offset': safe_int(self.chapter_number_offset_var.get(), 0),
                    'enable_decimal_chapters': self.enable_decimal_chapters_var.get(),
                    'use_header_as_output': self.use_header_as_output_var.get(),
                    'disable_gemini_safety': self.disable_gemini_safety_var.get(),
                    'auto_update_check': self.auto_update_check_var.get(),
                    'force_ncx_only': self.force_ncx_only_var.get(),
                    'single_api_image_chunks': self.single_api_image_chunks_var.get(),
                    'enable_gemini_thinking': self.enable_gemini_thinking_var.get(),
                    'thinking_budget': int(self.thinking_budget_var.get()) if self.thinking_budget_var.get().lstrip('-').isdigit() else 0,
                    'openai_base_url': self.openai_base_url_var.get(),
                    'groq_base_url': self.groq_base_url_var.get() if hasattr(self, 'groq_base_url_var') else '',
                    'fireworks_base_url': self.fireworks_base_url_var.get() if hasattr(self, 'fireworks_base_url_var') else '',
                    'use_custom_openai_endpoint': self.use_custom_openai_endpoint_var.get(),
                    'text_extraction_method': self.text_extraction_method_var.get() if hasattr(self, 'text_extraction_method_var') else 'standard',
                    'file_filtering_level': self.file_filtering_level_var.get() if hasattr(self, 'file_filtering_level_var') else 'smart',
                    'extraction_mode': 'enhanced' if self.text_extraction_method_var.get() == 'enhanced' else self.file_filtering_level_var.get(), 
                    'enhanced_filtering': self.file_filtering_level_var.get() if self.text_extraction_method_var.get() == 'enhanced' else 'smart', 
                    'use_gemini_openai_endpoint': self.use_gemini_openai_endpoint_var.get(),
                    'gemini_openai_endpoint': self.gemini_openai_endpoint_var.get(),
                                        
                    # ALL Anti-duplicate parameters (moved below other settings)
                    'enable_anti_duplicate': getattr(self, 'enable_anti_duplicate_var', type('', (), {'get': lambda: False})).get(),
                    'top_p': float(getattr(self, 'top_p_var', type('', (), {'get': lambda: 1.0})).get()),
                    'top_k': safe_int(getattr(self, 'top_k_var', type('', (), {'get': lambda: 0})).get(), 0),
                    'frequency_penalty': float(getattr(self, 'frequency_penalty_var', type('', (), {'get': lambda: 0.0})).get()),
                    'presence_penalty': float(getattr(self, 'presence_penalty_var', type('', (), {'get': lambda: 0.0})).get()),
                    'repetition_penalty': float(getattr(self, 'repetition_penalty_var', type('', (), {'get': lambda: 1.0})).get()),
                    'candidate_count': safe_int(getattr(self, 'candidate_count_var', type('', (), {'get': lambda: 1})).get(), 1),
                    'custom_stop_sequences': getattr(self, 'custom_stop_sequences_var', type('', (), {'get': lambda: ''})).get(),
                    'logit_bias_enabled': getattr(self, 'logit_bias_enabled_var', type('', (), {'get': lambda: False})).get(),
                    'logit_bias_strength': float(getattr(self, 'logit_bias_strength_var', type('', (), {'get': lambda: -0.5})).get()),
                    'bias_common_words': getattr(self, 'bias_common_words_var', type('', (), {'get': lambda: False})).get(),
                    'bias_repetitive_phrases': getattr(self, 'bias_repetitive_phrases_var', type('', (), {'get': lambda: False})).get(),
                    'enable_parallel_extraction': self.enable_parallel_extraction_var.get(),
                    'extraction_workers': safe_int(self.extraction_workers_var.get(), 4),
                    
                    # Batch header translation settings
                    'batch_translate_headers': self.batch_translate_headers_var.get(),
                    'headers_per_batch': safe_int(self.headers_per_batch_var.get(), 500),
                    'update_html_headers': self.update_html_headers_var.get(),
                    'save_header_translations': self.save_header_translations_var.get(),
                    
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
                    "ATTACH_CSS_TO_CHAPTERS": "1" if self.attach_css_to_chapters_var.get() else "0",
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
                    'DISABLE_CHAPTER_MERGING': '1' if self.disable_chapter_merging_var.get() else '0',
                    "ENABLE_IMAGE_TRANSLATION": "1" if self.enable_image_translation_var.get() else "0",
                    "PROCESS_WEBNOVEL_IMAGES": "1" if self.process_webnovel_images_var.get() else "0",
                    "WEBNOVEL_MIN_HEIGHT": str(self.config['webnovel_min_height']),
                    "MAX_IMAGES_PER_CHAPTER": str(self.config['max_images_per_chapter']),
                    "IMAGE_CHUNK_HEIGHT": str(self.config['image_chunk_height']),
                    "HIDE_IMAGE_TRANSLATION_LABEL": "1" if self.hide_image_translation_label_var.get() else "0",
                    "DISABLE_EPUB_GALLERY": "1" if self.disable_epub_gallery_var.get() else "0",
                    "DISABLE_ZERO_DETECTION": "1" if self.disable_zero_detection_var.get() else "0",
                    "DUPLICATE_DETECTION_MODE": self.duplicate_detection_mode_var.get(),
                    "ENABLE_DECIMAL_CHAPTERS": "1" if self.enable_decimal_chapters_var.get() else "0",
                    'ENABLE_WATERMARK_REMOVAL': "1" if self.enable_watermark_removal_var.get() else "0",
                    'SAVE_CLEANED_IMAGES': "1" if self.save_cleaned_images_var.get() else "0",
                    'TRANSLATION_CHUNK_PROMPT': str(getattr(self, 'translation_chunk_prompt', '')),  # FIXED: Convert to string
                    'IMAGE_CHUNK_PROMPT': str(getattr(self, 'image_chunk_prompt', '')),  # FIXED: Convert to string
                    "DISABLE_GEMINI_SAFETY": str(self.config.get('disable_gemini_safety', False)).lower(),
                    'auto_update_check': str(self.auto_update_check_var.get()),
                    'FORCE_NCX_ONLY': '1' if self.force_ncx_only_var.get() else '0',
                    'SINGLE_API_IMAGE_CHUNKS': "1" if self.single_api_image_chunks_var.get() else "0",
                    'ENABLE_GEMINI_THINKING': "1" if self.enable_gemini_thinking_var.get() else "0",
                    'THINKING_BUDGET': self.thinking_budget_var.get() if self.enable_gemini_thinking_var.get() else '0',
                    # Custom API endpoints
                    'OPENAI_CUSTOM_BASE_URL': self.openai_base_url_var.get() if self.openai_base_url_var.get() else '',
                    'GROQ_API_URL': self.groq_base_url_var.get() if hasattr(self, 'groq_base_url_var') and self.groq_base_url_var.get() else '',
                    'FIREWORKS_API_URL': self.fireworks_base_url_var.get() if hasattr(self, 'fireworks_base_url_var') and self.fireworks_base_url_var.get() else '',
                    'USE_CUSTOM_OPENAI_ENDPOINT': '1' if self.use_custom_openai_endpoint_var.get() else '0',
                    'USE_GEMINI_OPENAI_ENDPOINT': '1' if self.use_gemini_openai_endpoint_var.get() else '0',
                    'GEMINI_OPENAI_ENDPOINT': self.gemini_openai_endpoint_var.get() if self.gemini_openai_endpoint_var.get() else '',
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
                    'PRESERVE_TRANSPARENCY': "1" if self.config.get('preserve_transparency', True) else "0",
                    'OPTIMIZE_FOR_OCR': "1" if self.config.get('optimize_for_ocr', True) else "0",
                    'PROGRESSIVE_ENCODING': "1" if self.config.get('progressive_encoding', True) else "0",
                    'SAVE_COMPRESSED_IMAGES': "1" if self.config.get('save_compressed_images', False) else "0",
                    
                    # Metadata and batch header settings
                    'TRANSLATE_METADATA_FIELDS': json.dumps(self.translate_metadata_fields),
                    'METADATA_TRANSLATION_MODE': self.config.get('metadata_translation_mode', 'together'),
                    'BATCH_TRANSLATE_HEADERS': "1" if self.batch_translate_headers_var.get() else "0",
                    'HEADERS_PER_BATCH': str(self.config.get('headers_per_batch', 400)),
                    'UPDATE_HTML_HEADERS': "1" if self.update_html_headers_var.get() else "0",
                    'SAVE_HEADER_TRANSLATIONS': "1" if self.save_header_translations_var.get() else "0",
                    # EXTRACTION_MODE:
                    "TEXT_EXTRACTION_METHOD": self.text_extraction_method_var.get() if hasattr(self, 'text_extraction_method_var') else 'standard',
                    "FILE_FILTERING_LEVEL": self.file_filtering_level_var.get() if hasattr(self, 'file_filtering_level_var') else 'smart',
                    "EXTRACTION_MODE": 'enhanced' if self.text_extraction_method_var.get() == 'enhanced' else self.file_filtering_level_var.get(), 
                    "ENHANCED_FILTERING": self.file_filtering_level_var.get() if self.text_extraction_method_var.get() == 'enhanced' else 'smart', 
                    
                    # ALL Anti-duplicate environment variables (moved below other settings)
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
                    'EXTRACTION_WORKERS': str(self.extraction_workers_var.get()) if self.enable_parallel_extraction_var.get() else '1',
                    
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
            messagebox.showerror("Error", "Profile cannot be empty.")
            return
        content = self.prompt_text.get('1.0', tk.END).strip()
        self.prompt_profiles[name] = content
        self.config['prompt_profiles'] = self.prompt_profiles
        self.config['active_profile'] = name
        self.profile_menu['values'] = list(self.prompt_profiles.keys())
        messagebox.showinfo("Saved", f"Profile '{name}' saved.")
        self.save_profiles()

    def delete_profile(self):
        """Delete the selected profile."""
        name = self.profile_var.get()
        if name not in self.prompt_profiles:
            messagebox.showerror("Error", f"Profile '{name}' not found.")
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
            self.config['delay'] = safe_float(delay_val, 2)
            
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
            self.config['translation_history_limit'] = safe_int(trans_history_val, 2)
            
            # Add fuzzy matching threshold
            if hasattr(self, 'fuzzy_threshold_var'):
                fuzzy_val = self.fuzzy_threshold_var.get()
                if 0.5 <= fuzzy_val <= 1.0:
                    self.config['glossary_fuzzy_threshold'] = fuzzy_val
                else:
                    self.config['glossary_fuzzy_threshold'] = 0.90  # default
                    
             # Add after saving translation_prompt_text:
            if hasattr(self, 'format_instructions_text'):
                try:
                    self.config['glossary_format_instructions'] = self.format_instructions_text.get('1.0', tk.END).strip()
                except:
                    pass 
                    
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
            self.config['force_ncx_only'] = self.force_ncx_only_var.get()
            self.config['vertex_ai_location'] = self.vertex_location_var.get()
            self.config['batch_translate_headers'] = self.batch_translate_headers_var.get()
            self.config['headers_per_batch'] = self.headers_per_batch_var.get()
            self.config['update_html_headers'] = self.update_html_headers_var.get() 
            self.config['save_header_translations'] = self.save_header_translations_var.get()
            self.config['single_api_image_chunks'] = self.single_api_image_chunks_var.get()
            self.config['enable_gemini_thinking'] = self.enable_gemini_thinking_var.get()
            self.config['thinking_budget'] = int(self.thinking_budget_var.get()) if self.thinking_budget_var.get().lstrip('-').isdigit() else 0
            self.config['openai_base_url'] = self.openai_base_url_var.get()
            self.config['fireworks_base_url'] = self.fireworks_base_url_var.get()
            self.config['use_custom_openai_endpoint'] = self.use_custom_openai_endpoint_var.get()
            self.config['disable_chapter_merging'] = self.disable_chapter_merging_var.get()
            self.config['use_gemini_openai_endpoint'] = self.use_gemini_openai_endpoint_var.get()
            self.config['gemini_openai_endpoint'] = self.gemini_openai_endpoint_var.get()
            # Save extraction worker settings
            self.config['enable_parallel_extraction'] = self.enable_parallel_extraction_var.get()
            self.config['extraction_workers'] = self.extraction_workers_var.get()
            self.config['glossary_max_text_size'] = self.glossary_max_text_size_var.get()


            # NEW: Save strip honorifics setting
            self.config['strip_honorifics'] = self.strip_honorifics_var.get()
            
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
                
            # New cleaner UI variables
            if hasattr(self, 'text_extraction_method_var'):
                self.config['text_extraction_method'] = self.text_extraction_method_var.get()
                self.config['file_filtering_level'] = self.file_filtering_level_var.get()
                
                # Update extraction_mode for backwards compatibility
                if self.text_extraction_method_var.get() == 'enhanced':
                    self.config['extraction_mode'] = 'enhanced'
                    self.config['enhanced_filtering'] = self.file_filtering_level_var.get()
                else:
                    self.config['extraction_mode'] = self.file_filtering_level_var.get()
            else:
                # Fallback for old UI - keep existing behavior
                self.config['extraction_mode'] = self.extraction_mode_var.get()

            # Enhanced mode settings (these already exist in your code but ensure they're saved)
            self.config['enhanced_filtering'] = getattr(self, 'enhanced_filtering_var', tk.StringVar(value='smart')).get()
            self.config['enhanced_preserve_structure'] = getattr(self, 'enhanced_preserve_structure_var', tk.BooleanVar(value=True)).get()

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

            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(encrypted_config, f, ensure_ascii=False, indent=2) 
            
            # Only show message if requested
            if show_message:
                messagebox.showinfo("Saved", "Configuration saved.")
                
        except Exception as e:
            # Always show error messages regardless of show_message
            messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def log_debug(self, message):
        self.append_log(f"[DEBUG] {message}")

if __name__ == "__main__":
    import time
    
    print("🚀 Starting Glossarion v3.9.0...")
    
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
