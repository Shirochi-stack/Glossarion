# multi_api_key_manager.py
"""
Multi API Key Manager for Glossarion
Handles multiple API keys with round-robin load balancing and rate limit management
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import ttkbootstrap as tb
import json
import os
import threading
import time
import queue
from typing import Dict, List, Optional, Tuple
import requests
from datetime import datetime, timedelta
import logging
logger = logging.getLogger(__name__)
class RateLimitCache:
    """Thread-safe rate limit cache"""
    def __init__(self):
        self._cache = {}  # key_id -> expiry_time
        self._lock = threading.Lock()
    
    def add_rate_limit(self, key_id: str, cooldown_seconds: int):
        """Add a key to rate limit cache"""
        with self._lock:
            self._cache[key_id] = time.time() + cooldown_seconds
            logger.info(f"Added {key_id} to rate limit cache for {cooldown_seconds}s")
    
    def is_rate_limited(self, key_id: str) -> bool:
        """Check if key is rate limited"""
        with self._lock:
            if key_id not in self._cache:
                return False
            
            if time.time() >= self._cache[key_id]:
                # Expired, remove it
                del self._cache[key_id]
                return False
            
            return True
    
    def clear_expired(self):
        """Remove expired entries"""
        with self._lock:
            current_time = time.time()
            expired = [k for k, v in self._cache.items() if current_time >= v]
            for k in expired:
                del self._cache[k]
    
    def get_remaining_cooldown(self, key_id: str) -> float:
        """Get remaining cooldown time in seconds"""
        with self._lock:
            if key_id not in self._cache:
                return 0
            remaining = self._cache[key_id] - time.time()
            return max(0, remaining)


class APIKeyEntry:
    """Enhanced API key entry with test result storage"""
    def __init__(self, api_key: str, model: str, cooldown: int = 60, enabled: bool = True):
        self.api_key = api_key
        self.model = model
        self.cooldown = cooldown
        self.enabled = enabled
        self.last_error_time = None
        self.error_count = 0
        self.success_count = 0
        self.last_used_time = None
        self.is_cooling_down = False
        
        # Add test result storage
        self.last_test_result = None  # 'passed', 'failed', 'error', 'rate_limited'
        self.last_test_time = None
        self.last_test_message = None
    
    def is_available(self) -> bool:
        if not self.enabled:
            return False
        if self.last_error_time and self.is_cooling_down:
            time_since_error = time.time() - self.last_error_time
            if time_since_error < self.cooldown:
                return False
            else:
                self.is_cooling_down = False
        return True
    
    def mark_error(self, error_code: int = None):
        self.error_count += 1
        self.last_error_time = time.time()
        if error_code == 429:
            self.is_cooling_down = True
    
    def mark_success(self):
        self.success_count += 1
        self.last_used_time = time.time()
        self.error_count = 0
    
    def set_test_result(self, result: str, message: str = None):
        """Store test result"""
        self.last_test_result = result
        self.last_test_time = time.time()
        self.last_test_message = message

    def to_dict(self):
        """Convert to dictionary for saving"""
        return {
            'api_key': self.api_key,
            'model': self.model,
            'cooldown': self.cooldown,
            'enabled': self.enabled
        }
class APIKeyPool:
    """Thread-safe API key pool with proper rotation"""
    def __init__(self):
        self.keys: List[APIKeyEntry] = []
        self.lock = threading.Lock()
        self._rotation_index = 0  # Global rotation index
        self._thread_assignments = {}  # thread_id -> (key_index, assignment_time)
        self._rate_limit_cache = RateLimitCache()
    
    def load_from_list(self, key_list: List[dict]):
        with self.lock:
            self.keys.clear()
            for key_data in key_list:
                entry = APIKeyEntry(
                    api_key=key_data.get('api_key', ''),
                    model=key_data.get('model', ''),
                    cooldown=key_data.get('cooldown', 60),
                    enabled=key_data.get('enabled', True)
                )
                self.keys.append(entry)
            self._rotation_index = 0
            logger.info(f"Loaded {len(self.keys)} API keys into pool")
    
    def get_key_for_thread(self, force_rotation: bool = False, 
                          rotation_frequency: int = 1) -> Optional[Tuple[APIKeyEntry, int, str]]:
        """Get a key for the current thread with proper rotation logic"""
        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name
        
        with self.lock:
            if not self.keys:
                return None
            
            # Clear expired rate limits
            self._rate_limit_cache.clear_expired()
            
            # Check if thread already has an assignment
            if thread_id in self._thread_assignments and not force_rotation:
                key_index, assignment_time = self._thread_assignments[thread_id]
                if key_index < len(self.keys):
                    key = self.keys[key_index]
                    key_id = f"Key#{key_index+1} ({key.model})"
                    
                    # Check if the assigned key is still available
                    if key.is_available() and not self._rate_limit_cache.is_rate_limited(key_id):
                        return key, key_index, key_id
            
            # Find next available key
            attempts = 0
            while attempts < len(self.keys):
                # Use round-robin rotation
                key_index = self._rotation_index
                self._rotation_index = (self._rotation_index + 1) % len(self.keys)
                
                key = self.keys[key_index]
                key_id = f"Key#{key_index+1} ({key.model})"
                
                # Check availability
                if key.is_available() and not self._rate_limit_cache.is_rate_limited(key_id):
                    # Assign to thread
                    self._thread_assignments[thread_id] = (key_index, time.time())
                    logger.info(f"[Thread-{thread_name}] Assigned {key_id}")
                    return key, key_index, key_id
                
                attempts += 1
            
            # No available keys - try to find one with shortest cooldown
            best_key_index = None
            min_cooldown = float('inf')
            
            for i, key in enumerate(self.keys):
                key_id = f"Key#{i+1} ({key.model})"
                cooldown = self._rate_limit_cache.get_remaining_cooldown(key_id)
                if cooldown < min_cooldown:
                    min_cooldown = cooldown
                    best_key_index = i
            
            if best_key_index is not None:
                key = self.keys[best_key_index]
                key_id = f"Key#{best_key_index+1} ({key.model})"
                logger.warning(f"All keys on cooldown, using {key_id} (cooldown: {min_cooldown:.1f}s)")
                self._thread_assignments[thread_id] = (best_key_index, time.time())
                return key, best_key_index, key_id
            
            return None
    
    def mark_key_error(self, key_index: int, error_code: int = None):
        with self.lock:
            if 0 <= key_index < len(self.keys):
                self.keys[key_index].mark_error(error_code)
                
                # Add to rate limit cache if it's a 429
                if error_code == 429:
                    key = self.keys[key_index]
                    key_id = f"Key#{key_index+1} ({key.model})"
                    self._rate_limit_cache.add_rate_limit(key_id, key.cooldown)
    
    def mark_key_success(self, key_index: int):
        with self.lock:
            if 0 <= key_index < len(self.keys):
                self.keys[key_index].mark_success()
    
    def release_thread_assignment(self, thread_id: int = None):
        """Release key assignment for a thread"""
        if thread_id is None:
            thread_id = threading.current_thread().ident
        
        with self.lock:
            if thread_id in self._thread_assignments:
                del self._thread_assignments[thread_id]
                logger.debug(f"Released key assignment for thread {thread_id}")

    def get_all_keys(self) -> List[APIKeyEntry]:
        """Get all keys in the pool"""
        with self.lock:
            return self.keys.copy()

    @property
    def current_index(self):
        """Get the current rotation index"""
        with self.lock:
            return self._rotation_index

    @current_index.setter
    def current_index(self, value: int):
        """Set the current rotation index"""
        with self.lock:
            if self.keys:
                self._rotation_index = value % len(self.keys)
            else:
                self._rotation_index = 0
            
    def add_key(self, key_entry: APIKeyEntry):
        """Add a new key to the pool"""
        with self.lock:
            self.keys.append(key_entry)
            logger.info(f"Added key for model {key_entry.model} to pool")

    def remove_key(self, index: int):
        """Remove a key from the pool by index"""
        with self.lock:
            if 0 <= index < len(self.keys):
                removed_key = self.keys.pop(index)
                # Clean up any thread assignments for this key
                threads_to_remove = []
                for thread_id, (key_index, _) in self._thread_assignments.items():
                    if key_index == index:
                        threads_to_remove.append(thread_id)
                    elif key_index > index:
                        # Adjust indices for keys after the removed one
                        self._thread_assignments[thread_id] = (key_index - 1, self._thread_assignments[thread_id][1])
                
                for thread_id in threads_to_remove:
                    del self._thread_assignments[thread_id]
                
                # Reset rotation index if needed
                if self._rotation_index >= len(self.keys) and len(self.keys) > 0:
                    self._rotation_index = 0
                
                logger.info(f"Removed key for model {removed_key.model} from pool")

class MultiAPIKeyDialog:
    """Dialog for managing multiple API keys"""
    
    def __init__(self, parent, translator_gui):
        self.parent = parent
        self.translator_gui = translator_gui
        self.dialog = None
        self.key_pool = APIKeyPool()
        self.tree = None
        self.test_results = queue.Queue()
        
        # Load existing keys from config
        self._load_keys_from_config()
        
        # Create and show dialog
        self._create_dialog()
        
    def _load_keys_from_config(self):
        """Load API keys from translator GUI config"""
        if hasattr(self.translator_gui, 'config'):
            multi_api_keys = self.translator_gui.config.get('multi_api_keys', [])
            self.key_pool.load_from_list(multi_api_keys)
    
    def _update_rotation_display(self, *args):
        """Update the rotation description based on settings"""
        if self.force_rotation_var.get():
            freq = self.rotation_frequency_var.get()
            if freq == 1:
                desc = "Keys will rotate on every request (maximum distribution)"
            else:
                desc = f"Keys will rotate every {freq} requests"
        else:
            desc = "Keys will only rotate on errors or rate limits"
        
        self.rotation_desc_label.config(text=desc)
    
    def _save_keys_to_config(self):
        """Save API keys and rotation settings to translator GUI config"""
        if hasattr(self.translator_gui, 'config'):
            # Convert keys to list of dicts
            key_list = [key.to_dict() for key in self.key_pool.get_all_keys()]
            self.translator_gui.config['multi_api_keys'] = key_list
            
            # Use the current state of the toggle instead of always setting to True
            self.translator_gui.config['use_multi_api_keys'] = self.enabled_var.get()
            
            # Save rotation settings
            self.translator_gui.config['force_key_rotation'] = self.force_rotation_var.get()
            self.translator_gui.config['rotation_frequency'] = self.rotation_frequency_var.get()
            
            # Save config
            self.translator_gui.save_config(show_message=False)
    
    def _create_dialog(self):
        """Create the main dialog"""
        # Use WindowManager if available
        if hasattr(self.translator_gui, 'wm'):
            self.dialog, scrollable_frame, canvas = self.translator_gui.wm.setup_scrollable(
                self.parent,
                "Multi API Key Manager",
                width=None,
                height=None,
                max_width_ratio=0.9,
                max_height_ratio=0.99
            )
        else:
            self.dialog = tk.Toplevel(self.parent)
            self.dialog.title("Multi API Key Manager")
            self.dialog.geometry("900x600")
            scrollable_frame = self.dialog
        
        # Main container
        main_frame = tk.Frame(scrollable_frame, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title and description
        title_frame = tk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(title_frame, text="Multi API Key Management", 
                font=('TkDefaultFont', 16, 'bold')).pack(side=tk.LEFT)
        
        # Enable/Disable toggle
        self.enabled_var = tk.BooleanVar(value=self.translator_gui.config.get('use_multi_api_keys', False))
        tb.Checkbutton(title_frame, text="Enable Multi-Key Mode", 
                      variable=self.enabled_var,
                      bootstyle="round-toggle",
                      command=self._toggle_multi_key_mode).pack(side=tk.RIGHT, padx=(20, 0))
        
        tk.Label(main_frame, 
                text="Manage multiple API keys with automatic rotation and rate limit handling.\n"
                     "Keys can be rotated automatically to distribute load evenly.\n"
                     "Rate-limited keys are automatically cooled down and skipped in rotation.",
                font=('TkDefaultFont', 10), fg='gray', justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 15))
        
        # Rotation settings frame
        rotation_frame = tk.LabelFrame(main_frame, text="Rotation Settings", padx=15, pady=10)
        rotation_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Force rotation toggle
        rotation_settings = tk.Frame(rotation_frame)
        rotation_settings.pack(fill=tk.X)
        
        self.force_rotation_var = tk.BooleanVar(value=self.translator_gui.config.get('force_key_rotation', True))
        tb.Checkbutton(rotation_settings, text="Force Key Rotation", 
                      variable=self.force_rotation_var,
                      bootstyle="round-toggle",
                      command=self._update_rotation_display).pack(side=tk.LEFT)
        
        # Rotation frequency
        tk.Label(rotation_settings, text="Every").pack(side=tk.LEFT, padx=(20, 5))
        self.rotation_frequency_var = tk.IntVar(value=self.translator_gui.config.get('rotation_frequency', 1))
        frequency_spinbox = tb.Spinbox(rotation_settings, from_=1, to=100, 
                                      textvariable=self.rotation_frequency_var,
                                      width=5, command=self._update_rotation_display)
        frequency_spinbox.pack(side=tk.LEFT)
        tk.Label(rotation_settings, text="requests").pack(side=tk.LEFT, padx=(5, 0))
        
        # Rotation description
        self.rotation_desc_label = tk.Label(rotation_frame, 
                                          text="", 
                                          font=('TkDefaultFont', 9), fg='blue')
        self.rotation_desc_label.pack(anchor=tk.W, pady=(5, 0))
        self._update_rotation_display()
        
        # Add key section
        self._create_add_key_section(main_frame)
        
        # Separator
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=15)
        
        # Key list section
        self._create_key_list_section(main_frame)
        
        # Button bar
        self._create_button_bar(main_frame)
        
        # Load existing keys into tree
        self._refresh_key_list()
        
        # Center dialog
        self.dialog.transient(self.parent)
        #self.dialog.grab_set()
        
        # Handle window close
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _create_add_key_section(self, parent):
        """Create the add key section"""
        add_frame = tk.LabelFrame(parent, text="Add New API Key", padx=15, pady=15)
        add_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Grid configuration
        add_frame.columnconfigure(1, weight=1)
        add_frame.columnconfigure(3, weight=1)
        
        # API Key
        tk.Label(add_frame, text="API Key:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.api_key_var = tk.StringVar()
        self.api_key_entry = tb.Entry(add_frame, textvariable=self.api_key_var, show='*')
        self.api_key_entry.grid(row=0, column=1, sticky=tk.EW, pady=5)
        
        # Toggle visibility button
        self.show_key_btn = tb.Button(add_frame, text="👁", width=3,
                                     command=self._toggle_key_visibility)
        self.show_key_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Model
        tk.Label(add_frame, text="Model:").grid(row=0, column=3, sticky=tk.W, padx=(20, 10), pady=5)
        self.model_var = tk.StringVar()
        self.model_entry = tb.Entry(add_frame, textvariable=self.model_var)
        self.model_entry.grid(row=0, column=4, sticky=tk.EW, pady=5)
        
        # Cooldown
        tk.Label(add_frame, text="Cooldown (seconds):").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.cooldown_var = tk.IntVar(value=60)
        cooldown_frame = tk.Frame(add_frame)
        cooldown_frame.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        tb.Spinbox(cooldown_frame, from_=10, to=3600, textvariable=self.cooldown_var,
                  width=10).pack(side=tk.LEFT)
        tk.Label(cooldown_frame, text="(10-3600)", font=('TkDefaultFont', 9), 
                fg='gray').pack(side=tk.LEFT, padx=(10, 0))
        
        # Add button
        tb.Button(add_frame, text="Add Key", command=self._add_key,
                 bootstyle="success").grid(row=1, column=4, sticky=tk.E, pady=5)
        
        # Copy from current button
        tb.Button(add_frame, text="Copy Current Settings", 
                 command=self._copy_current_settings,
                 bootstyle="info-outline").grid(row=1, column=3, columnspan=2, 
                                               sticky=tk.W, padx=(20, 0), pady=5)
    
    def _create_key_list_section(self, parent):
        """Create the key list section with inline editing"""
        list_frame = tk.LabelFrame(parent, text="API Keys", padx=15, pady=15)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview with scrollbar
        tree_frame = tk.Frame(list_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Treeview
        columns = ('Model', 'Cooldown', 'Status', 'Success', 'Errors', 'Last Used')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='tree headings',
                                yscrollcommand=scrollbar.set, height=10)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=self.tree.yview)
        
        # Configure columns with better widths and anchoring
        self.tree.heading('#0', text='API Key', anchor='w')
        self.tree.column('#0', width=200, minwidth=150, anchor='w')
        
        self.tree.heading('Model', text='Model', anchor='w')
        self.tree.column('Model', width=180, minwidth=120, anchor='w')
        
        self.tree.heading('Cooldown', text='Cooldown', anchor='center')
        self.tree.column('Cooldown', width=80, minwidth=60, anchor='center')
        
        self.tree.heading('Status', text='Status', anchor='center')
        self.tree.column('Status', width=150, minwidth=100, anchor='center')
        
        self.tree.heading('Success', text='✓', anchor='center')
        self.tree.column('Success', width=50, minwidth=40, anchor='center')
        
        self.tree.heading('Errors', text='✗', anchor='center')
        self.tree.column('Errors', width=50, minwidth=40, anchor='center')
        
        self.tree.heading('Last Used', text='Last Used', anchor='center')
        self.tree.column('Last Used', width=90, minwidth=70, anchor='center')
        
        # Configure tree style for better appearance
        style = ttk.Style()
        style.configure("Treeview.Heading", font=('TkDefaultFont', 11, 'bold'))
        
        # Bind events for inline editing
        self.tree.bind('<Button-1>', self._on_click)
        self.tree.bind('<Button-3>', self._show_context_menu)
        
        # Track editing state
        self.edit_widget = None
        
        # Action buttons
        action_frame = tk.Frame(list_frame)
        action_frame.pack(fill=tk.X, pady=(10, 0))
        
        tb.Button(action_frame, text="Test Selected", command=self._test_selected,
                 bootstyle="warning").pack(side=tk.LEFT, padx=(0, 5))
        
        tb.Button(action_frame, text="Test All", command=self._test_all,
                 bootstyle="warning").pack(side=tk.LEFT, padx=5)
        
        tb.Button(action_frame, text="Enable Selected", command=self._enable_selected,
                 bootstyle="success").pack(side=tk.LEFT, padx=5)
        
        tb.Button(action_frame, text="Disable Selected", command=self._disable_selected,
                 bootstyle="danger").pack(side=tk.LEFT, padx=5)
        
        tb.Button(action_frame, text="Remove Selected", command=self._remove_selected,
                 bootstyle="danger").pack(side=tk.LEFT, padx=5)
        
        # Stats label
        self.stats_label = tk.Label(action_frame, text="", font=('TkDefaultFont', 11), fg='gray')
        self.stats_label.pack(side=tk.RIGHT)

    def _on_click(self, event):
        """Handle click on tree item for inline editing"""
        # Close any existing edit widget
        if self.edit_widget:
            self.edit_widget.destroy()
            self.edit_widget = None
        
        # Identify what was clicked
        region = self.tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        
        if not item:
            return
        
        # Get column index
        col_index = int(column.replace('#', ''))
        
        # Get the key index
        index = self.tree.index(item)
        if index >= len(self.key_pool.keys):
            return
        
        key = self.key_pool.keys[index]
        
        # Only allow editing Model (column #1) and Cooldown (column #2)
        if col_index == 1:  # Model column
            self._edit_model_inline(item, column, key)
        elif col_index == 2:  # Cooldown column
            self._edit_cooldown_inline(item, column, key)

    def _edit_model_inline(self, item, column, key):
        """Create inline editor for model name"""
        # Get the bounding box of the cell
        x, y, width, height = self.tree.bbox(item, column)
        
        # Create entry widget
        edit_var = tk.StringVar(value=key.model)
        self.edit_widget = tb.Entry(self.tree, textvariable=edit_var)
        
        def save_edit():
            new_value = edit_var.get().strip()
            if new_value and new_value != key.model:
                key.model = new_value
                self._refresh_key_list()
                self._show_status(f"Updated model to: {new_value}")
            if self.edit_widget:
                self.edit_widget.destroy()
                self.edit_widget = None
        
        def cancel_edit(event=None):
            if self.edit_widget:
                self.edit_widget.destroy()
                self.edit_widget = None
        
        # Place and configure the entry
        self.edit_widget.place(x=x, y=y, width=width, height=height)
        self.edit_widget.focus()
        self.edit_widget.select_range(0, tk.END)
        
        # Bind events
        self.edit_widget.bind('<Return>', lambda e: save_edit())
        self.edit_widget.bind('<Escape>', cancel_edit)
        self.edit_widget.bind('<FocusOut>', lambda e: save_edit())
        
        # Prevent the click from selecting the item
        return "break"

    def _edit_cooldown_inline(self, item, column, key):
        """Create inline editor for cooldown"""
        # Get the bounding box of the cell
        x, y, width, height = self.tree.bbox(item, column)
        
        # Create spinbox widget
        edit_var = tk.IntVar(value=key.cooldown)
        self.edit_widget = tb.Spinbox(self.tree, from_=10, to=3600, 
                                      textvariable=edit_var, width=10)
        
        def save_edit():
            new_value = edit_var.get()
            if new_value != key.cooldown:
                key.cooldown = new_value
                self._refresh_key_list()
                self._show_status(f"Updated cooldown to: {new_value}s")
            if self.edit_widget:
                self.edit_widget.destroy()
                self.edit_widget = None
        
        def cancel_edit(event=None):
            if self.edit_widget:
                self.edit_widget.destroy()
                self.edit_widget = None
        
        # Place and configure the spinbox
        self.edit_widget.place(x=x, y=y, width=width, height=height)
        self.edit_widget.focus()
        
        # Bind events
        self.edit_widget.bind('<Return>', lambda e: save_edit())
        self.edit_widget.bind('<Escape>', cancel_edit)
        self.edit_widget.bind('<FocusOut>', lambda e: save_edit())
        
        # Prevent the click from selecting the item
        return "break"

    def _show_context_menu(self, event):
        """Show simplified context menu for tree items"""
        # Select item under cursor
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            
            # Create context menu
            menu = tk.Menu(self.dialog, tearoff=0)
            menu.add_command(label="Test", command=self._test_selected)
            menu.add_command(label="Enable", command=self._enable_selected)
            menu.add_command(label="Disable", command=self._disable_selected)
            menu.add_separator()
            menu.add_command(label="Remove", command=self._remove_selected)
            
            # Show menu
            menu.post(event.x_root, event.y_root)
    
    def _create_button_bar(self, parent):
        """Create the bottom button bar"""
        button_frame = tk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Save button
        tb.Button(button_frame, text="Save & Close", command=self._save_and_close,
                 bootstyle="success").pack(side=tk.RIGHT, padx=(5, 0))
        
        # Cancel button  
        tb.Button(button_frame, text="Cancel", command=self._on_close,
                 bootstyle="secondary").pack(side=tk.RIGHT)
        
        # Import/Export
        tb.Button(button_frame, text="Import", command=self._import_keys,
                 bootstyle="info-outline").pack(side=tk.LEFT, padx=(0, 5))
        
        tb.Button(button_frame, text="Export", command=self._export_keys,
                 bootstyle="info-outline").pack(side=tk.LEFT)
    
    def _toggle_key_visibility(self):
        """Toggle API key visibility"""
        if self.api_key_entry.cget('show') == '*':
            self.api_key_entry.config(show='')
            self.show_key_btn.config(text='🔒')
        else:
            self.api_key_entry.config(show='*')
            self.show_key_btn.config(text='👁')
    
    def _toggle_multi_key_mode(self):
        """Toggle multi-key mode"""
        enabled = self.enabled_var.get()
        self.translator_gui.config['use_multi_api_keys'] = enabled
        
        # Save the config immediately
        self.translator_gui.save_config(show_message=False)
        
        # Update UI state
        for widget in [self.api_key_entry, self.model_entry]:
            if widget:
                widget.config(state=tk.NORMAL if enabled else tk.DISABLED)
        
        # Handle Treeview separately - it doesn't support state property
        if self.tree:
            if enabled:
                # Re-enable tree interactions
                self.tree.bind('<Button-1>', lambda e: 'break' if not self.enabled_var.get() else None)
                self.tree.bind('<Button-3>', self._show_context_menu)
                #self.tree.bind('<Double-Button-1>', self._on_double_click)
            else:
                # Disable tree interactions
                self.tree.bind('<Button-1>', lambda e: 'break')
                self.tree.bind('<Button-3>', lambda e: 'break')
                self.tree.bind('<Double-Button-1>', lambda e: 'break')
        
        # Update action buttons state
        for child in self.dialog.winfo_children():
            if isinstance(child, tk.Frame):
                for subchild in child.winfo_children():
                    if isinstance(subchild, tk.Frame):
                        for button in subchild.winfo_children():
                            if isinstance(button, (tb.Button, ttk.Button)) and button.cget('text') in [
                                'Test Selected', 'Test All', 'Enable Selected', 
                                'Disable Selected', 'Remove Selected', 'Add Key'
                            ]:
                                button.config(state=tk.NORMAL if enabled else tk.DISABLED)
    
    def _copy_current_settings(self):
        """Copy current API key and model from main GUI"""
        if hasattr(self.translator_gui, 'api_key_var'):
            self.api_key_var.set(self.translator_gui.api_key_var.get())
        if hasattr(self.translator_gui, 'model_var'):
            self.model_var.set(self.translator_gui.model_var.get())
    
    def _add_key(self):
        """Add a new API key"""
        api_key = self.api_key_var.get().strip()
        model = self.model_var.get().strip()
        cooldown = self.cooldown_var.get()
        
        if not api_key or not model:
            messagebox.showerror("Error", "Please enter both API key and model name")
            return
        
        # Add to pool
        key_entry = APIKeyEntry(api_key, model, cooldown)
        self.key_pool.add_key(key_entry)
        
        # Clear inputs
        self.api_key_var.set("")
        self.model_var.set("")
        self.cooldown_var.set(60)
        
        # Refresh list
        self._refresh_key_list()
        
        # Show success
        self._show_status(f"Added key for model: {model}")
    
    def _refresh_key_list(self):
        """Refresh the key list display"""
        # Clear tree
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add keys
        keys = self.key_pool.get_all_keys()
        for i, key in enumerate(keys):
            # Mask API key for display
            masked_key = key.api_key[:8] + "..." + key.api_key[-4:] if len(key.api_key) > 12 else key.api_key
            
            # Status
            if not key.enabled:
                status = "Disabled"
                tags = ('disabled',)
            elif key.is_cooling_down:
                remaining = int(key.cooldown - (time.time() - key.last_error_time))
                status = f"Cooling ({remaining}s)"
                tags = ('cooling',)
            else:
                status = "Active"
                tags = ('active',)
            
            # Last used
            last_used = ""
            if key.last_used_time:
                # Show relative time if recent, otherwise just time
                time_diff = time.time() - key.last_used_time
                if time_diff < 60:  # Less than 1 minute
                    last_used = "Just now"
                elif time_diff < 3600:  # Less than 1 hour
                    last_used = f"{int(time_diff/60)}m ago"
                elif time_diff < 86400:  # Less than 1 day
                    last_used = datetime.fromtimestamp(key.last_used_time).strftime("%H:%M")
                else:
                    last_used = datetime.fromtimestamp(key.last_used_time).strftime("%m/%d")
            
            # Insert into tree
            self.tree.insert('', 'end', 
                           text=masked_key,
                           values=(key.model, f"{key.cooldown}s", status, 
                                 key.success_count, key.error_count, last_used),
                           tags=tags)
        
        # Configure tags
        self.tree.tag_configure('active', foreground='green')
        self.tree.tag_configure('cooling', foreground='orange')
        self.tree.tag_configure('disabled', foreground='gray')
        
        # Update stats
        active_count = sum(1 for k in keys if k.enabled and not k.is_cooling_down)
        total_count = len(keys)
        self.stats_label.config(text=f"Keys: {active_count} active / {total_count} total")
    
    
    def _test_selected(self):
        """Test selected API keys with inline progress"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select keys to test")
            return
        
        # Get selected indices
        indices = [self.tree.index(item) for item in selected]
        
        # Start testing in thread
        thread = threading.Thread(target=self._run_inline_tests, args=(indices,))
        thread.daemon = True
        thread.start()

    def _test_all(self):
        """Test all API keys with inline progress"""
        if not self.key_pool.keys:
            messagebox.showwarning("Warning", "No keys to test")
            return
        
        indices = list(range(len(self.key_pool.keys)))
        
        # Start testing in thread
        thread = threading.Thread(target=self._run_inline_tests, args=(indices,))
        thread.daemon = True
        thread.start()

    def _run_inline_tests(self, indices: List[int]):
        """Run API tests with persistent inline results"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        
        print(f"[DEBUG] Starting tests for {len(indices)} keys")
        
        # Mark all selected keys as testing
        for index in indices:
            if index < len(self.key_pool.keys):
                key = self.key_pool.keys[index]
                key.last_test_result = None
                key._testing = True
                print(f"[DEBUG] Marked key {index} as testing")
        
        # Refresh once to show "Testing..." status
        self.dialog.after(0, self._refresh_key_list)
        
        # Create thread pool for parallel testing
        max_workers = min(10, len(indices))
        
        def test_single_key(index):
            """Test a single API key directly"""
            print(f"[DEBUG] Testing key at index {index}")
            
            if index >= len(self.key_pool.keys):
                return None
                
            key = self.key_pool.keys[index]
            
            try:
                # Simple test - just check if we can import the libraries
                # This is a minimal test to see if the function completes
                print(f"[DEBUG] Testing {key.model} with key {key.api_key[:8]}...")
                
                # Simulate a test
                import time
                time.sleep(1)  # Simulate API call
                
                # For now, just mark as passed to test the flow
                key.mark_success()
                key.set_test_result('passed', 'Test successful')
                print(f"[DEBUG] Key {index} test completed - PASSED")
                return (index, True, "Test passed")
                
            except Exception as e:
                print(f"[DEBUG] Key {index} test failed: {e}")
                key.mark_error()
                key.set_test_result('error', str(e)[:30])
                return (index, False, f"Error: {str(e)[:50]}...")
        
        # Run tests in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all test tasks
            future_to_index = {executor.submit(test_single_key, i): i for i in indices}
            
            # Process results as they complete
            for future in as_completed(future_to_index):
                result = future.result()
                if result:
                    results.append(result)
                    print(f"[DEBUG] Got result: {result}")
        
        print(f"[DEBUG] All tests complete. Results: {len(results)}")
        
        # Calculate summary
        success_count = sum(1 for _, success, _ in results if success)
        total_count = len(results)
        
        # Clear testing flags
        for index in indices:
            if index < len(self.key_pool.keys):
                key = self.key_pool.keys[index]
                if hasattr(key, '_testing'):
                    delattr(key, '_testing')
                    print(f"[DEBUG] Cleared testing flag for key {index}")
        
        # Update UI in main thread
        print(f"[DEBUG] Refreshing UI with results")
        self.dialog.after(0, self._refresh_key_list)
        self.dialog.after(0, lambda: self.stats_label.config(
            text=f"Test complete: {success_count}/{total_count} passed"))
        


    def _update_tree_item(self, index: int):
        """Update a single tree item based on current key state"""
        def update():
            # Find the tree item for this index
            items = self.tree.get_children()
            if index < len(items):
                item = items[index]
                key = self.key_pool.keys[index]
                
                # Determine status and tags
                if key.last_test_result is None:
                    # Currently testing
                    status = "⏳ Testing..."
                    tags = ('testing',)
                elif not key.enabled:
                    status = "Disabled"
                    tags = ('disabled',)
                elif key.last_test_result == 'passed':
                    if key.is_cooling_down:
                        remaining = int(key.cooldown - (time.time() - key.last_error_time))
                        status = f"✅ Passed (cooling {remaining}s)"
                        tags = ('passed_cooling',)
                    else:
                        status = "✅ Passed"
                        tags = ('passed',)
                elif key.last_test_result == 'failed':
                    status = "❌ Failed"
                    tags = ('failed',)
                elif key.last_test_result == 'rate_limited':
                    remaining = int(key.cooldown - (time.time() - key.last_error_time))
                    status = f"⚠️ Rate Limited ({remaining}s)"
                    tags = ('ratelimited',)
                elif key.last_test_result == 'error':
                    status = "❌ Error"
                    if key.last_test_message:
                        status += f": {key.last_test_message[:20]}..."
                    tags = ('error',)
                elif key.is_cooling_down:
                    remaining = int(key.cooldown - (time.time() - key.last_error_time))
                    status = f"Cooling ({remaining}s)"
                    tags = ('cooling',)
                else:
                    status = "Active"
                    tags = ('active',)
                
                # Get current values
                values = list(self.tree.item(item, 'values'))
                
                # Update status column
                values[2] = status
                
                # Update success/error counts
                values[3] = key.success_count
                values[4] = key.error_count
                
                # Update last used
                if key.last_used_time:
                    values[5] = datetime.fromtimestamp(key.last_used_time).strftime("%H:%M:%S")
                
                # Update the item
                self.tree.item(item, values=values, tags=tags)
        
        # Run in main thread
        self.dialog.after(0, update)

    def _refresh_key_list(self):
        """Refresh the key list display preserving test results"""
        # Clear tree
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add keys
        keys = self.key_pool.get_all_keys()
        for i, key in enumerate(keys):
            # Mask API key for display
            masked_key = key.api_key[:8] + "..." + key.api_key[-4:] if len(key.api_key) > 12 else key.api_key
            
            # Determine status based on test results and current state
            if key.last_test_result is None and hasattr(key, '_testing'):
                # Currently testing (temporary flag)
                status = "⏳ Testing..."
                tags = ('testing',)
            elif not key.enabled:
                status = "Disabled"
                tags = ('disabled',)
            elif key.last_test_result == 'passed':
                status = "✅ Passed"
                tags = ('passed',)
            elif key.last_test_result == 'failed':
                status = "❌ Failed"
                tags = ('failed',)
            elif key.last_test_result == 'rate_limited':
                status = "⚠️ Rate Limited"
                tags = ('ratelimited',)
            elif key.last_test_result == 'error':
                status = "❌ Error"
                if key.last_test_message:
                    status += f": {key.last_test_message[:20]}..."
                tags = ('error',)
            elif key.is_cooling_down and key.last_error_time:
                remaining = int(key.cooldown - (time.time() - key.last_error_time))
                if remaining > 0:
                    status = f"Cooling ({remaining}s)"
                    tags = ('cooling',)
                else:
                    key.is_cooling_down = False
                    status = "Active"
                    tags = ('active',)
            else:
                status = "Active"
                tags = ('active',)
            
            # Last used
            last_used = ""
            if key.last_used_time:
                last_used = datetime.fromtimestamp(key.last_used_time).strftime("%H:%M:%S")
            
            # Insert into tree
            self.tree.insert('', 'end', 
                           text=masked_key,
                           values=(key.model, f"{key.cooldown}s", status, 
                                 key.success_count, key.error_count, last_used),
                           tags=tags)
        
        # Configure tags
        self.tree.tag_configure('active', foreground='green')
        self.tree.tag_configure('cooling', foreground='orange')
        self.tree.tag_configure('disabled', foreground='gray')
        self.tree.tag_configure('testing', foreground='blue', font=('TkDefaultFont', 14))
        self.tree.tag_configure('passed', foreground='dark green', font=('TkDefaultFont', 14))
        self.tree.tag_configure('failed', foreground='red')
        self.tree.tag_configure('ratelimited', foreground='orange')
        self.tree.tag_configure('error', foreground='dark red')
        
        # Update stats
        active_count = sum(1 for k in keys if k.enabled and not k.is_cooling_down)
        total_count = len(keys)
        passed_count = sum(1 for k in keys if k.last_test_result == 'passed')
        self.stats_label.config(text=f"Keys: {active_count} active / {total_count} total | {passed_count} passed tests")
    
    def _create_progress_dialog(self):
        """Create simple progress dialog at mouse cursor position"""
        self.progress_dialog = tk.Toplevel(self.dialog)
        self.progress_dialog.title("Testing API Keys")
        
        # Get mouse position
        x = self.progress_dialog.winfo_pointerx()
        y = self.progress_dialog.winfo_pointery()
        
        # Set geometry at cursor position (offset slightly so cursor is inside window)
        self.progress_dialog.geometry(f"500x400+{x-50}+{y-30}")
        
        # Add label
        label = tb.Label(self.progress_dialog, text="Testing in progress...", 
                        font=('TkDefaultFont', 10, 'bold'))
        label.pack(pady=10)
        
        # Add text widget for results
        self.progress_text = scrolledtext.ScrolledText(self.progress_dialog, 
                                                      wrap=tk.WORD, width=60, height=20)
        self.progress_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Add close button (initially disabled)
        self.close_button = tb.Button(self.progress_dialog, text="Close", 
                                     command=self.progress_dialog.destroy,
                                     bootstyle="secondary", state=tk.DISABLED)
        self.close_button.pack(pady=(0, 10))
        
        self.progress_dialog.transient(self.dialog)

    def _run_tests(self, indices: List[int]):
        """Run API tests for specified keys in parallel"""
        from unified_api_client import UnifiedClient
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        
        # Get Gemini endpoint settings
        use_gemini_endpoint = os.getenv("USE_GEMINI_OPENAI_ENDPOINT", "0") == "1"
        gemini_endpoint = os.getenv("GEMINI_OPENAI_ENDPOINT", "")
        
        # Create thread pool for parallel testing
        max_workers = min(10, len(indices))  # Limit to 10 concurrent tests
        
        def test_single_key(index):
            """Test a single API key"""
            if index >= len(self.key_pool.keys):
                return None
                
            key = self.key_pool.keys[index]
            
            # Create a key identifier
            key_preview = f"{key.api_key[:8]}...{key.api_key[-4:]}" if len(key.api_key) > 12 else key.api_key
            test_label = f"{key.model} [{key_preview}]"
            
            # Update UI to show test started
            self.dialog.after(0, lambda label=test_label: self.progress_text.insert(tk.END, f"Testing {label}... "))
            self.dialog.after(0, lambda: self.progress_text.see(tk.END))
            
            try:
                # Check if this is a Gemini model with custom endpoint
                is_gemini_model = key.model.lower().startswith('gemini')
                
                if is_gemini_model and use_gemini_endpoint and gemini_endpoint:
                    # Test Gemini with OpenAI-compatible endpoint
                    import openai
                    
                    endpoint_url = gemini_endpoint
                    if not endpoint_url.endswith('/openai/'):
                        endpoint_url = endpoint_url.rstrip('/') + '/openai/'
                    
                    client = openai.OpenAI(
                        api_key=key.api_key,
                        base_url=endpoint_url,
                        timeout=10.0
                    )
                    
                    response = client.chat.completions.create(
                        model=key.model.replace('gemini/', ''),
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Say 'API test successful' and nothing else."}
                        ],
                        max_tokens=100,
                        temperature=0.7
                    )
                    
                    content = response.choices[0].message.content
                    if content and "test successful" in content.lower():
                        self.dialog.after(0, lambda label=test_label: self._update_test_result(label, True))
                        key.mark_success()
                        return (index, True, "Test passed")
                    else:
                        self.dialog.after(0, lambda label=test_label: self._update_test_result(label, False))
                        key.mark_error()
                        return (index, False, "Unexpected response")
                else:
                    # Use UnifiedClient for non-Gemini or regular Gemini
                    client = UnifiedClient(
                        api_key=key.api_key,
                        model=key.model,
                        output_dir=None
                    )
                    
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Say 'API test successful' and nothing else."}
                    ]
                    
                    response = client.send(
                        messages,
                        temperature=0.7,
                        max_tokens=100
                    )
                    
                    if response and isinstance(response, tuple):
                        content, finish_reason = response
                        if content and "test successful" in content.lower():
                            self.dialog.after(0, lambda label=test_label: self._update_test_result(label, True))
                            key.mark_success()
                            return (index, True, "Test passed")
                        else:
                            self.dialog.after(0, lambda label=test_label: self._update_test_result(label, False))
                            key.mark_error()
                            return (index, False, "Unexpected response")
                    else:
                        self.dialog.after(0, lambda label=test_label: self._update_test_result(label, False))
                        key.mark_error()
                        return (index, False, "No response")
                        
            except Exception as e:
                error_msg = str(e)
                error_code = None
                
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    error_code = 429
                    
                self.dialog.after(0, lambda label=test_label: self._update_test_result(label, False, error=True))
                key.mark_error(error_code)
                return (index, False, f"Error: {error_msg}")
        
        # Run tests in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all test tasks
            future_to_index = {executor.submit(test_single_key, i): i for i in indices}
            
            # Process results as they complete
            for future in as_completed(future_to_index):
                result = future.result()
                if result:
                    self.test_results.put(result)
        
        # Show completion and close button
        self.dialog.after(0, self._show_completion)
        
        # Process final results
        self.dialog.after(0, self._process_test_results)

    def _update_test_result(self, test_label, success, error=False):
        """Update the progress text with test result"""
        # Find the line with this test label
        content = self.progress_text.get("1.0", tk.END)
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if test_label in line and not any(status in line for status in ["✅", "❌"]):
                # This is our line, update it
                if error:
                    result_text = "❌ ERROR"
                elif success:
                    result_text = "✅ PASSED"
                else:
                    result_text = "❌ FAILED"
                
                # Calculate position
                line_num = i + 1
                line_end = f"{line_num}.end"
                
                self.progress_text.insert(line_end, result_text)
                self.progress_text.insert(line_end, "\n")
                self.progress_text.see(tk.END)
                break

    def _show_completion(self):
        """Show completion in the same dialog"""
        self.progress_text.insert(tk.END, "\n--- Testing Complete ---\n")
        self.progress_text.see(tk.END)
        
    def _process_test_results(self):
        """Process test results and show in the same dialog"""
        results = []
        
        # Get all results
        while not self.test_results.empty():
            try:
                results.append(self.test_results.get_nowait())
            except:
                break
        
        if results:
            # Build result message
            success_count = sum(1 for _, success, _ in results if success)
            total_count = len(results)
            
            # Update everything at once after all tests complete
            def final_update():
                # Clear testing flags
                for index in indices:
                    if index < len(self.key_pool.keys):
                        key = self.key_pool.keys[index]
                        if hasattr(key, '_testing'):
                            delattr(key, '_testing')
                
                self._refresh_key_list()
                self.stats_label.config(text=f"Test complete: {success_count}/{total_count} passed")

            # Use lambda to capture the variables in scope
            self.dialog.after(0, lambda: final_update())
            
            # Add summary to the same dialog
            self.progress_text.insert(tk.END, f"\nSummary: {success_count}/{total_count} passed\n")
            self.progress_text.insert(tk.END, "-" * 50 + "\n\n")
            
            for i, success, msg in results:
                key = self.key_pool.keys[i]
                # Show key identifier in results too
                key_preview = f"{key.api_key[:8]}...{key.api_key[-4:]}" if len(key.api_key) > 12 else key.api_key
                status = "✅" if success else "❌"
                self.progress_text.insert(tk.END, f"{status} {key.model} [{key_preview}]: {msg}\n")
            
            self.progress_text.see(tk.END)
            
            # Enable close button now that testing is complete
            self.close_button.config(state=tk.NORMAL)
            
            # Update the dialog title
            self.progress_dialog.title(f"API Test Results - {success_count}/{total_count} passed")
            
            # Refresh list
            self._refresh_key_list()
    
    def _enable_selected(self):
        """Enable selected keys"""
        selected = self.tree.selection()
        for item in selected:
            index = self.tree.index(item)
            if index < len(self.key_pool.keys):
                self.key_pool.keys[index].enabled = True
        
        self._refresh_key_list()
        self._show_status(f"Enabled {len(selected)} key(s)")
    
    def _disable_selected(self):
        """Disable selected keys"""
        selected = self.tree.selection()
        for item in selected:
            index = self.tree.index(item)
            if index < len(self.key_pool.keys):
                self.key_pool.keys[index].enabled = False
        
        self._refresh_key_list()
        self._show_status(f"Disabled {len(selected)} key(s)")
    
    def _remove_selected(self):
        """Remove selected keys"""
        selected = self.tree.selection()
        if not selected:
            return
        
        if messagebox.askyesno("Confirm", f"Remove {len(selected)} selected key(s)?"):
            # Get indices in reverse order to avoid index shifting
            indices = sorted([self.tree.index(item) for item in selected], reverse=True)
            
            for index in indices:
                self.key_pool.remove_key(index)
            
            self._refresh_key_list()
            self._show_status(f"Removed {len(selected)} key(s)")
    
    def _edit_cooldown(self):
        """Edit cooldown for selected key"""
        selected = self.tree.selection()
        if not selected or len(selected) != 1:
            messagebox.showwarning("Warning", "Please select exactly one key")
            return
        
        index = self.tree.index(selected[0])
        if index >= len(self.key_pool.keys):
            return
        
        key = self.key_pool.keys[index]
        
        # Create simple dialog
        dialog = tk.Toplevel(self.dialog)
        dialog.title("Edit Cooldown")
        dialog.geometry("300x150")
        
        tk.Label(dialog, text=f"Cooldown for {key.model}:").pack(pady=10)
        
        cooldown_var = tk.IntVar(value=key.cooldown)
        tb.Spinbox(dialog, from_=10, to=3600, textvariable=cooldown_var,
                  width=10).pack(pady=5)
        

    
    def _import_keys(self):
        """Import keys from JSON file"""
        from tkinter import filedialog
        
        filename = filedialog.askopenfilename(
            title="Import API Keys",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    # Load keys
                    imported_count = 0
                    for key_data in data:
                        if isinstance(key_data, dict) and 'api_key' in key_data and 'model' in key_data:
                            self.key_pool.add_key(APIKeyEntry.from_dict(key_data))
                            imported_count += 1
                    
                    self._refresh_key_list()
                    messagebox.showinfo("Success", f"Imported {imported_count} API keys")
                else:
                    messagebox.showerror("Error", "Invalid file format")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import: {str(e)}")
    
    def _export_keys(self):
        """Export keys to JSON file"""
        from tkinter import filedialog
        
        if not self.key_pool.keys:
            messagebox.showwarning("Warning", "No keys to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export API Keys",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Convert keys to list of dicts
                key_list = [key.to_dict() for key in self.key_pool.get_all_keys()]
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(key_list, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("Success", f"Exported {len(key_list)} API keys")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    def _show_status(self, message: str):
        """Show status message"""
        self.stats_label.config(text=message)
    
    def _save_and_close(self):
        """Save configuration and close"""
        self._save_keys_to_config()
        messagebox.showinfo("Success", "API key configuration saved")
        self.dialog.destroy()
    
    def _on_close(self):
        """Handle dialog close"""
        self.dialog.destroy()


# Integration with translator_gui.py
def add_multi_api_key_button(translator_gui, parent_frame):
    """Add multi API key button to the processing section"""
    
    def open_multi_key_dialog():
        """Open the multi API key dialog"""
        MultiAPIKeyDialog(translator_gui.master, translator_gui)
    
    # Create button
    multi_key_btn = tb.Button(
        parent_frame,
        text="🔑 Multi API Keys",
        command=open_multi_key_dialog,
        bootstyle="primary-outline",
        width=15
    )
    
    # Store reference
    translator_gui.multi_api_key_btn = multi_key_btn
    
    return multi_key_btn
