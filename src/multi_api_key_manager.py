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
    """Simple API key entry for multi-key support"""
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

    def to_dict(self):
        """Convert to dictionary for saving"""
        return {
            'api_key': self.api_key,
            'model': self.model,
            'cooldown': self.cooldown,
            'enabled': self.enabled
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary"""
        return cls(
            api_key=data.get('api_key', ''),
            model=data.get('model', ''),
            cooldown=data.get('cooldown', 60),
            enabled=data.get('enabled', True)
        )
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
            
            # Enable multi-key mode
            self.translator_gui.config['use_multi_api_keys'] = True
            
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
        self.dialog.grab_set()
        
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
        """Create the key list section"""
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
        
        # Configure columns
        self.tree.heading('#0', text='API Key')
        self.tree.column('#0', width=300)
        
        self.tree.heading('Model', text='Model')
        self.tree.column('Model', width=150)
        
        self.tree.heading('Cooldown', text='Cooldown')
        self.tree.column('Cooldown', width=80)
        
        self.tree.heading('Status', text='Status')
        self.tree.column('Status', width=100)
        
        self.tree.heading('Success', text='Success')
        self.tree.column('Success', width=70)
        
        self.tree.heading('Errors', text='Errors')
        self.tree.column('Errors', width=70)
        
        self.tree.heading('Last Used', text='Last Used')
        self.tree.column('Last Used', width=120)
        
        # Context menu
        self.tree.bind('<Button-3>', self._show_context_menu)
        
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
        self.stats_label = tk.Label(action_frame, text="", font=('TkDefaultFont', 9), fg='gray')
        self.stats_label.pack(side=tk.RIGHT)
    
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
        
        # Update UI state
        for widget in [self.tree, self.api_key_entry, self.model_entry]:
            if widget:
                widget.config(state=tk.NORMAL if enabled else tk.DISABLED)
    
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
        
        # Update stats
        active_count = sum(1 for k in keys if k.enabled and not k.is_cooling_down)
        total_count = len(keys)
        self.stats_label.config(text=f"Keys: {active_count} active / {total_count} total")
    
    def _show_context_menu(self, event):
        """Show context menu for tree items"""
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
            menu.add_command(label="Edit Cooldown", command=self._edit_cooldown)
            menu.add_separator()
            menu.add_command(label="Remove", command=self._remove_selected)
            
            # Show menu
            menu.post(event.x_root, event.y_root)
    
    def _test_selected(self):
        """Test selected API keys"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select keys to test")
            return
        
        # Get selected indices
        indices = [self.tree.index(item) for item in selected]
        
        # Run tests in thread
        thread = threading.Thread(target=self._run_tests, args=(indices,))
        thread.daemon = True
        thread.start()
    
    def _test_all(self):
        """Test all API keys"""
        if not self.key_pool.keys:
            messagebox.showwarning("Warning", "No keys to test")
            return
        
        indices = list(range(len(self.key_pool.keys)))
        
        # Run tests in thread
        thread = threading.Thread(target=self._run_tests, args=(indices,))
        thread.daemon = True
        thread.start()
    
    def _run_tests(self, indices: List[int]):
        """Run API tests for specified keys"""
        from unified_api_client import UnifiedClient
        
        for i in indices:
            if i >= len(self.key_pool.keys):
                continue
                
            key = self.key_pool.keys[i]
            
            try:
                # Create client - NO temperature in __init__!
                client = UnifiedClient(
                    api_key=key.api_key,
                    model=key.model,
                    output_dir=None  # Not needed for testing
                )
                
                # Simple test message
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'API test successful' and nothing else."}
                ]
                
                # Send request - temperature goes HERE in the send method
                response = client.send(
                    messages,
                    temperature=0.7,  # Temperature goes in send(), not __init__
                    max_tokens=100
                )
                
                # Check response
                if response and isinstance(response, tuple):
                    content, finish_reason = response
                    if content and "test successful" in content.lower():
                        # Success
                        self.test_results.put((i, True, "Test passed"))
                        key.mark_success()
                    else:
                        # Got response but not expected content
                        self.test_results.put((i, False, f"Unexpected response: {content[:100]}"))
                        key.mark_error()
                else:
                    # Failed
                    self.test_results.put((i, False, "No response"))
                    key.mark_error()
                    
            except Exception as e:
                error_msg = str(e)
                error_code = None
                
                # Check for rate limit
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    error_code = 429
                    
                self.test_results.put((i, False, f"Error: {error_msg}"))
                key.mark_error(error_code)
        
        # Update UI
        self.dialog.after(0, self._process_test_results)
    
    def _process_test_results(self):
        """Process test results from queue"""
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
            
            message = f"Test Results: {success_count}/{total_count} passed\n\n"
            
            for i, success, msg in results:
                key = self.key_pool.keys[i]
                status = "✅" if success else "❌"
                message += f"{status} {key.model}: {msg}\n"
            
            # Show results
            self._show_test_results(message)
            
            # Refresh list
            self._refresh_key_list()
    
    def _show_test_results(self, message: str):
        """Show test results in a dialog"""
        dialog = tk.Toplevel(self.dialog)
        dialog.title("API Test Results")
        dialog.geometry("500x400")
        
        # Text widget
        text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, width=60, height=20)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text.insert('1.0', message)
        text.config(state=tk.DISABLED)
        
        # Close button
        tb.Button(dialog, text="Close", command=dialog.destroy,
                 bootstyle="secondary").pack(pady=(0, 10))
        
        dialog.transient(self.dialog)
        dialog.grab_set()
    
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
        
        def save_cooldown():
            key.cooldown = cooldown_var.get()
            self._refresh_key_list()
            dialog.destroy()
        
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=20)
        
        tb.Button(button_frame, text="Save", command=save_cooldown,
                 bootstyle="success").pack(side=tk.LEFT, padx=5)
        tb.Button(button_frame, text="Cancel", command=dialog.destroy,
                 bootstyle="secondary").pack(side=tk.LEFT)
        
        dialog.transient(self.dialog)
        dialog.grab_set()
    
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
        # Could add a status bar, for now just refresh the stats
        self.stats_label.config(text=message)
        self.dialog.after(3000, self._refresh_key_list)  # Refresh after 3 seconds
    
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
