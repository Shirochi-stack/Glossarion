# multi_api_key_manager.py
"""
Multi API Key Manager for Glossarion
Handles multiple API keys with round-robin load balancing and rate limit management
"""

from PySide6.QtCore import QMetaObject, Q_ARG
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit, 
    QTextEdit, QScrollArea, QFileDialog, QMessageBox, QComboBox, QCheckBox, 
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QSpinBox,
    QTreeWidget, QTreeWidgetItem, QAbstractItemView, QHeaderView, QMenu, QFrame,
    QCompleter
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QIcon, QFont, QPixmap, QShortcut, QKeySequence
import json
import os
import threading
import time
import queue
from typing import Dict, List, Optional, Tuple
import requests
from datetime import datetime, timedelta
import logging
from model_options import get_model_options
# Dialog for configuring per-key endpoint
try:
    from individual_endpoint_dialog import IndividualEndpointDialog
except Exception:
    IndividualEndpointDialog = None

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
    """Enhanced API key entry with thread-safe operations"""
    def __init__(self, api_key: str, model: str, cooldown: int = 60, enabled: bool = True, 
                 google_credentials: str = None, azure_endpoint: str = None, google_region: str = None,
                 azure_api_version: str = None, use_individual_endpoint: bool = False):
        self.api_key = api_key
        self.model = model
        self.cooldown = cooldown
        self.enabled = enabled
        self.google_credentials = google_credentials  # Path to Google service account JSON
        self.azure_endpoint = azure_endpoint  # Azure endpoint URL (only used if use_individual_endpoint is True)
        self.google_region = google_region  # Google Cloud region (e.g., us-east5, us-central1)
        self.azure_api_version = azure_api_version or '2025-01-01-preview'  # Azure API version
        self.use_individual_endpoint = use_individual_endpoint  # Toggle to enable/disable individual endpoint
        self.last_error_time = None
        self.error_count = 0
        self.success_count = 0
        self.last_used_time = None
        self.times_used = 0  # Incremented whenever this key is assigned/used
        self.is_cooling_down = False
        
        # Add lock for thread-safe modifications
        self._lock = threading.Lock()
        
        # Add test result storage
        self.last_test_result = None
        self.last_test_time = None
        self.last_test_message = None
    
    def is_available(self) -> bool:
        with self._lock:
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
        with self._lock:
            self.error_count += 1
            self.times_used = getattr(self, 'times_used', 0) + 1
            self.last_error_time = time.time()
            if error_code == 429:
                self.is_cooling_down = True
    
    def mark_success(self):
        with self._lock:
            self.success_count += 1
            self.times_used = getattr(self, 'times_used', 0) + 1
            self.last_used_time = time.time()
            self.error_count = 0
    
    def set_test_result(self, result: str, message: str = None):
        """Store test result"""
        with self._lock:
            self.last_test_result = result
            self.last_test_time = time.time()
            self.last_test_message = message
            
    def to_dict(self):
        """Convert to dictionary for saving"""
        return {
            'api_key': self.api_key,
            'model': self.model,
            'cooldown': self.cooldown,
            'enabled': self.enabled,
            'google_credentials': self.google_credentials,
            'azure_endpoint': self.azure_endpoint,
            'google_region': self.google_region,
            'azure_api_version': self.azure_api_version,
            'use_individual_endpoint': self.use_individual_endpoint,
            # Persist times used and test results
            'times_used': getattr(self, 'times_used', 0),
            'last_test_result': getattr(self, 'last_test_result', None),
            'last_test_time': getattr(self, 'last_test_time', None),
            'last_test_message': getattr(self, 'last_test_message', None)
        }
class APIKeyPool:
    """Thread-safe API key pool with proper rotation"""
    def __init__(self):
        self.keys: List[APIKeyEntry] = []
        self.lock = threading.Lock()  # This already exists
        self._rotation_index = 0
        self._thread_assignments = {}
        self._rate_limit_cache = RateLimitCache()
        
        # NEW LOCKS:
        self.key_locks = {}  # Will be populated when keys are loaded
        self.key_selection_lock = threading.Lock()  # For coordinating key selection across threads
        
        # Track which keys are currently being used by which threads
        self._keys_in_use = {}  # key_index -> set of thread_ids
        self._usage_lock = threading.Lock()
    
    def load_from_list(self, key_list: List[dict]):
        with self.lock:
            # Preserve existing counters by mapping old entries by (api_key, model)
            old_map = {}
            for old in getattr(self, 'keys', []):
                key = (getattr(old, 'api_key', ''), getattr(old, 'model', ''))
                old_map[key] = old
            
            self.keys.clear()
            self.key_locks.clear()  # Clear existing locks
            
            for i, key_data in enumerate(key_list):
                api_key = key_data.get('api_key', '')
                model = key_data.get('model', '')
                entry = APIKeyEntry(
                    api_key=api_key,
                    model=model,
                    cooldown=key_data.get('cooldown', 60),
                    enabled=key_data.get('enabled', True),
                    google_credentials=key_data.get('google_credentials'),
                    azure_endpoint=key_data.get('azure_endpoint'),
                    google_region=key_data.get('google_region'),
                    azure_api_version=key_data.get('azure_api_version'),
                    use_individual_endpoint=key_data.get('use_individual_endpoint', False)
                )
                # Load saved test results and usage counter
                entry.times_used = key_data.get('times_used', 0)
                entry.last_test_result = key_data.get('last_test_result', None)
                entry.last_test_time = key_data.get('last_test_time', None)
                entry.last_test_message = key_data.get('last_test_message', None)
                # Restore counters if we had this key before
                old = old_map.get((api_key, model))
                if old is not None:
                    try:
                        entry.success_count = getattr(old, 'success_count', entry.success_count)
                        entry.error_count = getattr(old, 'error_count', entry.error_count)
                        entry.times_used = getattr(old, 'times_used', getattr(old, 'success_count', 0) + getattr(old, 'error_count', 0))
                        entry.last_used_time = getattr(old, 'last_used_time', None)
                        entry.last_error_time = getattr(old, 'last_error_time', None)
                        entry.is_cooling_down = getattr(old, 'is_cooling_down', False)
                        entry.last_test_result = getattr(old, 'last_test_result', None)
                        entry.last_test_time = getattr(old, 'last_test_time', None)
                        entry.last_test_message = getattr(old, 'last_test_message', None)
                    except Exception:
                        pass
                self.keys.append(entry)
                # Create a lock for each key
                self.key_locks[i] = threading.Lock()
            
            # Keep rotation index if possible
            if getattr(self, '_rotation_index', 0) >= len(self.keys):
                self._rotation_index = 0
            else:
                self._rotation_index = getattr(self, '_rotation_index', 0)
            self._keys_in_use.clear()
            logger.info(f"Loaded {len(self.keys)} API keys into pool with individual locks (preserved counters where possible)")
    
    def get_key_for_thread(self, force_rotation: bool = False, 
                          rotation_frequency: int = 1) -> Optional[Tuple[APIKeyEntry, int, str]]:
        """Get a key for the current thread with proper rotation logic"""
        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name
        
        # Clear expired rate limits first
        self._rate_limit_cache.clear_expired()
        
        # Use key_selection_lock for the entire selection process
        with self.key_selection_lock:
            if not self.keys:
                return None
            
            # Check if thread already has an assignment
            if thread_id in self._thread_assignments and not force_rotation:
                key_index, assignment_time = self._thread_assignments[thread_id]
                if key_index < len(self.keys):
                    key = self.keys[key_index]
                    key_id = f"Key#{key_index+1} ({key.model})"
                    
                    # Check if the assigned key is still available
                    # Use the key-specific lock for checking availability
                    with self.key_locks.get(key_index, threading.Lock()):
                        if key.is_available() and not self._rate_limit_cache.is_rate_limited(key_id):
                            logger.debug(f"[Thread-{thread_name}] Reusing assigned {key_id}")
                            
                            # Track usage
                            with self._usage_lock:
                                if key_index not in self._keys_in_use:
                                    self._keys_in_use[key_index] = set()
                                self._keys_in_use[key_index].add(thread_id)
                            
                            return key, key_index, key_id
                        else:
                            # Remove invalid assignment
                            del self._thread_assignments[thread_id]
            
            # Find next available key using round-robin
            start_index = self._rotation_index
            attempts = 0
            
            while attempts < len(self.keys):
                # Get current index and immediately increment for next thread
                key_index = self._rotation_index
                self._rotation_index = (self._rotation_index + 1) % len(self.keys)
                
                key = self.keys[key_index]
                key_id = f"Key#{key_index+1} ({key.model})"
                
                # Use key-specific lock when checking and modifying key state
                with self.key_locks.get(key_index, threading.Lock()):
                    if key.is_available() and not self._rate_limit_cache.is_rate_limited(key_id):
                        # Assign to thread
                        self._thread_assignments[thread_id] = (key_index, time.time())
                        
                        # Increment usage counter on assignment
                        try:
                            key.times_used += 1
                        except Exception:
                            pass
                        
                        # Track usage
                        with self._usage_lock:
                            if key_index not in self._keys_in_use:
                                self._keys_in_use[key_index] = set()
                            self._keys_in_use[key_index].add(thread_id)
                        
                        # Clean up old assignments
                        current_time = time.time()
                        expired_threads = [
                            tid for tid, (_, ts) in self._thread_assignments.items()
                            if current_time - ts > 300  # 5 minutes
                        ]
                        for tid in expired_threads:
                            del self._thread_assignments[tid]
                            # Remove from usage tracking
                            with self._usage_lock:
                                for k_idx in list(self._keys_in_use.keys()):
                                    self._keys_in_use[k_idx].discard(tid)
                                    if not self._keys_in_use[k_idx]:
                                        del self._keys_in_use[k_idx]
                        
                        logger.info(f"[Thread-{thread_name}] Assigned {key_id}")
                        time.sleep(0.5)  # Brief pause to improve retry responsiveness
                        logger.debug("💤 Pausing briefly to improve retry responsiveness after key assignment")
                        return key, key_index, key_id
                
                attempts += 1
            
            # No available keys - find one with shortest cooldown
            best_key_index = None
            min_cooldown = float('inf')
            
            for i, key in enumerate(self.keys):
                if key.enabled:  # At least check if enabled
                    key_id = f"Key#{i+1} ({key.model})"
                    remaining = self._rate_limit_cache.get_remaining_cooldown(key_id)
                    
                    # Also check key's own cooldown
                    if key.is_cooling_down and key.last_error_time:
                        key_cooldown = key.cooldown - (time.time() - key.last_error_time)
                        remaining = max(remaining, key_cooldown)
                    
                    if remaining < min_cooldown:
                        min_cooldown = remaining
                        best_key_index = i
            
            if best_key_index is not None:
                key = self.keys[best_key_index]
                key_id = f"Key#{best_key_index+1} ({key.model})"
                logger.warning(f"[Thread-{thread_name}] All keys on cooldown, using {key_id} (cooldown: {min_cooldown:.1f}s)")
                self._thread_assignments[thread_id] = (best_key_index, time.time())
                time.sleep(0.5)  # Brief pause to improve retry responsiveness
                logger.debug("💤 Pausing briefly to improve retry responsiveness after cooldown key selection")
                return key, best_key_index, key_id
            
            logger.error(f"[Thread-{thread_name}] No keys available at all")
            return None
    
    def mark_key_error(self, key_index: int, error_code: int = None):
        """Mark a key as having an error (thread-safe with key-specific lock)"""
        if 0 <= key_index < len(self.keys):
            # Use key-specific lock for this operation
            with self.key_locks.get(key_index, threading.Lock()):
                # Mark error on the key itself
                self.keys[key_index].mark_error(error_code)
                
                # Add to rate limit cache if it's a 429
                if error_code == 429:
                    key = self.keys[key_index]
                    key_id = f"Key#{key_index+1} ({key.model})"
                    self._rate_limit_cache.add_rate_limit(key_id, key.cooldown)
                    
                    print(f"Marked key {key_id} with an error code")
                    time.sleep(0.5)  # Brief pause to improve retry responsiveness
                    logger.debug("💤 Pausing briefly to improve retry responsiveness after marking key error")

    def mark_key_success(self, key_index: int):
        """Mark a key as successful (thread-safe with key-specific lock)"""
        if 0 <= key_index < len(self.keys):
            # Use key-specific lock for this operation
            with self.key_locks.get(key_index, threading.Lock()):
                self.keys[key_index].mark_success()
                
                key = self.keys[key_index]
                print(f"Marked key {key_index} ({key.model}) as successful")
    
    def release_thread_assignment(self, thread_id: int = None):
        """Release key assignment for a thread"""
        if thread_id is None:
            thread_id = threading.current_thread().ident
        
        with self.key_selection_lock:
            # Remove from assignments
            if thread_id in self._thread_assignments:
                key_index, _ = self._thread_assignments[thread_id]
                del self._thread_assignments[thread_id]
                
                # Remove from usage tracking
                with self._usage_lock:
                    if key_index in self._keys_in_use:
                        self._keys_in_use[key_index].discard(thread_id)
                        if not self._keys_in_use[key_index]:
                            del self._keys_in_use[key_index]
                
                print(f"Released key assignment for thread {thread_id}")

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

class MultiAPIKeyDialog(QDialog):
    """Dialog for managing multiple API keys"""
    
    @staticmethod
    def show_dialog(parent, translator_gui):
        """Static method to create and show the dialog modally.
        
        This ensures proper PySide6 application context and returns after dialog closes.
        """
        # Ensure QApplication exists
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Create and execute dialog
        dialog = MultiAPIKeyDialog(parent, translator_gui)
        dialog.exec_()  # Modal execution - blocks until closed
        
        return dialog
    
    def __init__(self, parent, translator_gui):
        # PySide6 dialogs need QWidget parents or None
        # Create as standalone top-level dialog
        super().__init__(None)  # Create as top-level dialog
        
        self.translator_gui = translator_gui
        self.tree = None
        self.test_results = queue.Queue()
        
        self.key_pool = APIKeyPool()
        
        # Attempt to bind to UnifiedClient's shared pool so UI reflects live usage
        self._bind_shared_pool()
        
        # Load existing keys from config
        self._load_keys_from_config()
        
        # Create and show dialog
        self._create_dialog()
        
        # Make it a window (not just a dialog)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        
    def _set_icon(self, window):
        """Set Halgakos.ico as window icon if available."""
        try:
            base_dir = getattr(self.translator_gui, 'base_dir', os.getcwd())
            ico_path = os.path.join(base_dir, 'Halgakos.ico')
            if os.path.isfile(ico_path):
                try:
                    window.setWindowIcon(QIcon(ico_path))
                except Exception:
                    pass
        except Exception:
            pass
    
    def _create_styled_checkbox(self, text):
        """Create a checkbox with proper checkmark using text overlay"""
        checkbox = QCheckBox(text)
        
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
        checkmark.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        def position_checkmark():
            try:
                if checkmark:
                    checkmark.setGeometry(2, 1, 14, 14)
            except RuntimeError:
                pass
        
        def update_checkmark():
            try:
                if checkbox and checkmark:
                    if checkbox.isChecked():
                        position_checkmark()
                        checkmark.show()
                    else:
                        checkmark.hide()
            except RuntimeError:
                pass
        
        checkbox.stateChanged.connect(update_checkmark)
        
        def safe_init():
            try:
                position_checkmark()
                update_checkmark()
            except RuntimeError:
                pass
        
        QTimer.singleShot(0, safe_init)
        
        return checkbox
    
    def _disable_spinbox_mousewheel(self, spinbox):
        """Disable mousewheel scrolling on a spinbox (PySide6)"""
        spinbox.wheelEvent = lambda event: None
    
    def _disable_combobox_mousewheel(self, combobox):
        """Disable mousewheel scrolling on a combobox (PySide6)"""
        combobox.wheelEvent = lambda event: None

    def _bind_shared_pool(self):
        """Bind this dialog to the UnifiedClient's shared APIKeyPool if available.
        If UnifiedClient has no pool yet, register our pool as the shared pool.
        This keeps Times Used and other counters in sync across UI and runtime.
        """
        try:
            from unified_api_client import UnifiedClient
            # If UC already has a pool, use it; otherwise share ours
            if getattr(UnifiedClient, '_api_key_pool', None) is not None:
                self.key_pool = UnifiedClient._api_key_pool
            else:
                UnifiedClient._api_key_pool = self.key_pool
        except Exception:
            # If import fails (early load), continue with local pool
            pass

    def _load_keys_from_config(self):
        """Load API keys from translator GUI config"""
        if hasattr(self.translator_gui, 'config'):
            multi_api_keys = self.translator_gui.config.get('multi_api_keys', [])
            self.key_pool.load_from_list(multi_api_keys)
    
    def _update_rotation_display(self, *args):
        """Update the rotation description based on settings"""
        # Read current state from widgets
        if hasattr(self, 'force_rotation_checkbox') and self.force_rotation_checkbox.isChecked():
            freq = self.frequency_spinbox.value() if hasattr(self, 'frequency_spinbox') else 1
            if freq == 1:
                desc = "Keys will rotate on every request (maximum distribution)"
            else:
                desc = f"Keys will rotate every {freq} requests"
        else:
            desc = "Keys will only rotate on errors or rate limits"
        
        if hasattr(self, 'rotation_desc_label'):
            self.rotation_desc_label.setText(desc)
    
    def _save_keys_to_config(self):
        """Save API keys and rotation settings to translator GUI config"""
        if hasattr(self.translator_gui, 'config'):
            # Convert keys to list of dicts
            key_list = [key.to_dict() for key in self.key_pool.get_all_keys()]
            self.translator_gui.config['multi_api_keys'] = key_list
            
            # Save fallback settings - read from checkbox
            use_fallback = self.use_fallback_checkbox.isChecked() if hasattr(self, 'use_fallback_checkbox') else False
            self.translator_gui.config['use_fallback_keys'] = use_fallback
            # Update the parent GUI's variable to stay in sync
            if hasattr(self.translator_gui, 'use_fallback_keys_var'):
                try:
                    self.translator_gui.use_fallback_keys_var.set(use_fallback)
                except:
                    self.translator_gui.use_fallback_keys_var = use_fallback
            # Fallback keys are already saved when added/removed
            
            # Use the current state of the toggle - read from checkbox
            enabled = self.enabled_checkbox.isChecked() if hasattr(self, 'enabled_checkbox') else False
            self.translator_gui.config['use_multi_api_keys'] = enabled
            
            # Save rotation settings - read from widgets
            force_rotation = self.force_rotation_checkbox.isChecked() if hasattr(self, 'force_rotation_checkbox') else True
            rotation_freq = self.frequency_spinbox.value() if hasattr(self, 'frequency_spinbox') else 1
            self.translator_gui.config['force_key_rotation'] = force_rotation
            self.translator_gui.config['rotation_frequency'] = rotation_freq
            
            # Save config
            self.translator_gui.save_config(show_message=False)
            
            # Update multi-key status label in other settings if it exists
            if hasattr(self.translator_gui, '_update_multi_key_status_label'):
                try:
                    self.translator_gui._update_multi_key_status_label()
                except Exception:
                    pass
    
    def _create_dialog(self):
        """Create the main dialog using PySide6"""
        # Set window properties
        self.setWindowTitle("Multi API Key Manager")
        self.resize(900, 700)
        
        # Apply comprehensive stylesheet matching other settings dialogs
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
            }
            QGroupBox {
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #5a9fd4;
            }
            QPushButton {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                padding: 5px 15px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
                border-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #1a1a1a;
            }
            QPushButton:disabled {
                background-color: #1a1a1a;
                color: #666666;
                border-color: #2a2a2a;
            }
            QLineEdit, QTextEdit, QSpinBox, QComboBox {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                padding: 4px;
            }
            QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QComboBox:focus {
                border-color: #5a5a5a;
            }
            QLineEdit:disabled, QTextEdit:disabled, QSpinBox:disabled, QComboBox:disabled {
                background-color: #1a1a1a;
                color: #666666;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #7a7a7a;
                width: 0;
                height: 0;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                color: #e0e0e0;
                selection-background-color: #4a7ba7;
                selection-color: #ffffff;
                border: 1px solid #3a3a3a;
            }
            QCheckBox {
                color: #e0e0e0;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #5a5a5a;
                border-radius: 2px;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background-color: #4a7ba7;
                border-color: #4a7ba7;
            }
            QCheckBox::indicator:hover {
                border-color: #6a6a6a;
            }
            QCheckBox:disabled {
                color: #666666;
            }
            QCheckBox::indicator:disabled {
                background-color: #1a1a1a;
                border-color: #3a3a3a;
            }
            QTreeWidget {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                alternate-background-color: #252525;
            }
            QTreeWidget::item {
                padding: 4px;
            }
            QTreeWidget::item:selected {
                background-color: #3a3a3a;
                color: #e0e0e0;
            }
            QTreeWidget::item:hover {
                background-color: #3a3a3a;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: none;
                border-right: 1px solid #3a3a3a;
                border-bottom: 1px solid #3a3a3a;
                padding: 4px;
                font-weight: bold;
            }
            QScrollBar:vertical {
                background-color: #1e1e1e;
                width: 12px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background-color: #4a7ba7;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #5a8ab7;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background-color: #1e1e1e;
                height: 12px;
                border: none;
            }
            QScrollBar::handle:horizontal {
                background-color: #4a7ba7;
                border-radius: 6px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #5a8ab7;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QFrame[frameShape="4"] {
                /* HLine */
                background-color: #3a3a3a;
                max-height: 1px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #2d2d2d;
                border: none;
                width: 16px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #3a3a3a;
            }
            QSpinBox::up-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 6px solid #7a7a7a;
                width: 0;
                height: 0;
            }
            QSpinBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #7a7a7a;
                width: 0;
                height: 0;
            }
        """)
        
        # Create scroll area
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create scrollable content widget
        scrollable_widget = QWidget()
        scrollable_layout = QVBoxLayout(scrollable_widget)
        scrollable_layout.setContentsMargins(20, 10, 20, 20)  # Reduced top margin from 20 to 10
        scrollable_layout.setSpacing(10)  # Set spacing between widgets
        scroll_area.setWidget(scrollable_widget)
        
        # Set main layout - will have scroll area AND button bar
        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(0, 0, 0, 0)
        dialog_layout.setSpacing(0)
        dialog_layout.addWidget(scroll_area)
        
        # Store references
        self.main_frame = scrollable_widget
        self.scrollable_frame = scrollable_widget
        self.main_layout = scrollable_layout
        self.scrollable_layout = scrollable_layout  # Store for adjusting spacing
        
        # Title and description
        title_frame = QWidget()
        title_layout = QHBoxLayout(title_frame)
        title_layout.setContentsMargins(0, 0, 0, 5)  # Reduced bottom margin from 10 to 5
        
        title_label = QLabel("Multi API Key Management")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_layout.addWidget(title_label)
        
        # Enable/Disable toggle
        self.enabled_var = self.translator_gui.config.get('use_multi_api_keys', False)
        self.enabled_checkbox = self._create_styled_checkbox("Enable Multi-Key Mode")
        self.enabled_checkbox.setChecked(self.enabled_var)
        self.enabled_checkbox.toggled.connect(self._toggle_multi_key_mode)
        title_layout.addStretch()
        title_layout.addWidget(self.enabled_checkbox)
        
        scrollable_layout.addWidget(title_frame)
        
        # Store reference to description label for hide/show
        self.multikey_desc_label = QLabel(
                "Manage multiple API keys with automatic rotation and rate limit handling.\n"
                "Keys can be rotated automatically to distribute load evenly.\n"
                "Rate-limited keys are automatically cooled down and skipped in rotation.")
        self.multikey_desc_label.setStyleSheet("color: gray; padding-bottom: 10px;")
        self.multikey_desc_label.setWordWrap(True)
        scrollable_layout.addWidget(self.multikey_desc_label)
        
        # Rotation settings frame - store reference for enabling/disabling
        self.rotation_frame = QGroupBox("Rotation Settings")
        rotation_frame_layout = QVBoxLayout(self.rotation_frame)
        rotation_frame_layout.setContentsMargins(15, 10, 15, 10)
        
        # Force rotation toggle
        rotation_settings = QWidget()
        rotation_settings_layout = QHBoxLayout(rotation_settings)
        rotation_settings_layout.setContentsMargins(0, 0, 0, 0)
        
        self.force_rotation_var = self.translator_gui.config.get('force_key_rotation', True)
        self.force_rotation_checkbox = self._create_styled_checkbox("Force Key Rotation")
        self.force_rotation_checkbox.setChecked(self.force_rotation_var)
        self.force_rotation_checkbox.toggled.connect(self._update_rotation_display)
        rotation_settings_layout.addWidget(self.force_rotation_checkbox)
        
        # Rotation frequency
        rotation_settings_layout.addSpacing(20)
        rotation_settings_layout.addWidget(QLabel("Every"))
        self.rotation_frequency_var = self.translator_gui.config.get('rotation_frequency', 1)
        self.frequency_spinbox = QSpinBox()
        self.frequency_spinbox.setRange(1, 100)
        self.frequency_spinbox.setValue(self.rotation_frequency_var)
        self.frequency_spinbox.setMaximumWidth(60)
        self.frequency_spinbox.valueChanged.connect(self._update_rotation_display)
        self._disable_spinbox_mousewheel(self.frequency_spinbox)  # Disable mousewheel
        rotation_settings_layout.addWidget(self.frequency_spinbox)
        rotation_settings_layout.addWidget(QLabel("requests"))
        rotation_settings_layout.addStretch()
        
        rotation_frame_layout.addWidget(rotation_settings)
        
        # Rotation description
        self.rotation_desc_label = QLabel()
        self.rotation_desc_label.setStyleSheet("color: #5a9fd4; font-style: italic;")
        rotation_frame_layout.addWidget(self.rotation_desc_label)
        self._update_rotation_display()
        
        scrollable_layout.addWidget(self.rotation_frame)
        
        # Add key section
        self._create_add_key_section(scrollable_layout)
        
        # Separator - store reference to hide when multi-key mode is off
        self.multikey_separator = QFrame()
        self.multikey_separator.setFrameShape(QFrame.HLine)
        self.multikey_separator.setFrameShadow(QFrame.Sunken)
        scrollable_layout.addWidget(self.multikey_separator)
        
        # Key list section
        self._create_key_list_section(scrollable_layout)
        
        # Add stretch before fallback that only appears when multi-key is enabled
        # This will be removed when multi-key is disabled to bring fallback closer
        scrollable_layout.addStretch(0)
        
        # Create fallback container (hidden by default)
        self._create_fallback_section(scrollable_layout)
        
        # Add stretch to fill remaining space in scroll area
        scrollable_layout.addStretch(1)
        
        # Button bar at the bottom - moved outside scroll area below
        # (will be added to dialog_layout instead)
        
        # Load existing keys into tree
        self._refresh_key_list()
        
        # Create button bar outside scroll area (fixed at bottom)
        self._create_button_bar(dialog_layout)
        
        # Set icon
        self._set_icon(self)
        
        # Apply initial state for multi-key mode toggle
        self._toggle_multi_key_mode()

    def _create_fallback_section(self, parent_layout):
        """Create the fallback keys section at the bottom"""
        # Container that can be hidden
        self.fallback_container = QWidget()
        fallback_container_layout = QVBoxLayout(self.fallback_container)
        fallback_container_layout.setContentsMargins(0, 5, 0, 0)  # Reduced top margin from 10 to 5
        
        # Separator - store reference to hide when multi-key mode is off
        self.fallback_separator = QFrame()
        self.fallback_separator.setFrameShape(QFrame.HLine)
        self.fallback_separator.setFrameShadow(QFrame.Sunken)
        fallback_container_layout.addWidget(self.fallback_separator)
        
        # Main fallback frame
        fallback_frame = QGroupBox("Fallback Keys (For Prohibited Content)")
        fallback_frame_layout = QVBoxLayout(fallback_frame)
        fallback_frame_layout.setContentsMargins(15, 15, 15, 15)
        
        # Description
        desc_label = QLabel(
                "Configure fallback keys that will be used when content is blocked.\n"
                "These should use different API keys or models that are less restrictive.\n"
                "In Multi-Key Mode: tried when main rotation encounters prohibited content.\n"
                "In Single-Key Mode: tried directly when main key fails, bypassing main key retry.")
        desc_label.setStyleSheet("color: gray;")
        desc_label.setWordWrap(True)
        fallback_frame_layout.addWidget(desc_label)
        
        # Enable fallback checkbox
        self.use_fallback_var = self.translator_gui.config.get('use_fallback_keys', False)
        self.use_fallback_checkbox = self._create_styled_checkbox("Enable Fallback Keys")
        self.use_fallback_checkbox.setChecked(self.use_fallback_var)
        self.use_fallback_checkbox.toggled.connect(self._toggle_fallback_section)
        fallback_frame_layout.addWidget(self.use_fallback_checkbox)
        
        # Add fallback key section - store reference for opacity effect
        self.add_fallback_frame = QWidget()
        add_fallback_grid = QGridLayout(self.add_fallback_frame)
        add_fallback_grid.setContentsMargins(0, 0, 0, 10)
        
        # Row 0: Fallback API Key and Model
        add_fallback_grid.addWidget(QLabel("Fallback API Key:"), 0, 0, Qt.AlignLeft)
        self.fallback_key_entry = QLineEdit()
        self.fallback_key_entry.setEchoMode(QLineEdit.Password)
        add_fallback_grid.addWidget(self.fallback_key_entry, 0, 1)
        
        # Toggle fallback visibility
        self.show_fallback_btn = QPushButton("👁")
        self.show_fallback_btn.setFixedWidth(40)
        self.show_fallback_btn.clicked.connect(self._toggle_fallback_visibility)
        add_fallback_grid.addWidget(self.show_fallback_btn, 0, 2)
        
        # Fallback Model
        add_fallback_grid.addWidget(QLabel("Model:"), 0, 3, Qt.AlignLeft)
        fallback_models = get_model_options()
        self.fallback_model_combo = QComboBox()
        self.fallback_model_combo.addItems(fallback_models)
        self.fallback_model_combo.setEditable(True)
        self._disable_combobox_mousewheel(self.fallback_model_combo)  # Disable mousewheel
        add_fallback_grid.addWidget(self.fallback_model_combo, 0, 4)
        
        # Add fallback button
        add_fallback_btn = QPushButton("Add Fallback Key")
        add_fallback_btn.clicked.connect(self._add_fallback_key)
        add_fallback_grid.addWidget(add_fallback_btn, 0, 5, Qt.AlignRight)
        
        # Set column stretch
        add_fallback_grid.setColumnStretch(1, 1)
        add_fallback_grid.setColumnStretch(4, 1)
        
        fallback_frame_layout.addWidget(self.add_fallback_frame)
        
        # Row 1: Google Credentials (optional, discretely styled)
        google_creds_label = QLabel("Google Creds:")
        google_creds_label.setStyleSheet("color: gray; font-size: 8pt;")
        add_fallback_grid.addWidget(google_creds_label, 1, 0, Qt.AlignLeft)
        self.fallback_google_creds_entry = QLineEdit()
        self.fallback_google_creds_entry.setStyleSheet("font-size: 7pt;")
        add_fallback_grid.addWidget(self.fallback_google_creds_entry, 1, 1)
        
        # Google credentials browse button
        browse_google_btn = QPushButton("📁")
        browse_google_btn.setFixedWidth(40)
        browse_google_btn.clicked.connect(self._browse_fallback_google_credentials)
        add_fallback_grid.addWidget(browse_google_btn, 1, 2)
        
        # Google region field for fallback
        region_label = QLabel("Region:")
        region_label.setStyleSheet("color: gray;")
        add_fallback_grid.addWidget(region_label, 1, 3, Qt.AlignLeft)
        self.fallback_google_region_entry = QLineEdit("us-east5")
        self.fallback_google_region_entry.setStyleSheet("font-size: 7pt;")
        self.fallback_google_region_entry.setMaximumWidth(100)
        add_fallback_grid.addWidget(self.fallback_google_region_entry, 1, 4, 1, 1, Qt.AlignLeft)
        
        # Row 2: Individual Endpoint Toggle for fallback
        self.fallback_use_individual_endpoint_var = False
        self.fallback_individual_endpoint_toggle = self._create_styled_checkbox("Use Individual Endpoint")
        self.fallback_individual_endpoint_toggle.setChecked(False)
        self.fallback_individual_endpoint_toggle.toggled.connect(self._toggle_fallback_individual_endpoint_fields)
        add_fallback_grid.addWidget(self.fallback_individual_endpoint_toggle, 2, 0, 1, 2, Qt.AlignLeft)
        
        # Row 3: Individual Endpoint (initially hidden)
        self.fallback_individual_endpoint_label = QLabel("Individual Endpoint:")
        self.fallback_individual_endpoint_label.setStyleSheet("color: gray; font-size: 9pt;")
        add_fallback_grid.addWidget(self.fallback_individual_endpoint_label, 3, 0, Qt.AlignLeft)
        self.fallback_azure_endpoint_entry = QLineEdit()
        self.fallback_azure_endpoint_entry.setStyleSheet("font-size: 8pt;")
        add_fallback_grid.addWidget(self.fallback_azure_endpoint_entry, 3, 1, 1, 2)
        
        # Individual Endpoint API Version (small dropdown, initially hidden)
        self.fallback_individual_api_version_label = QLabel("API Ver:")
        self.fallback_individual_api_version_label.setStyleSheet("color: gray;")
        add_fallback_grid.addWidget(self.fallback_individual_api_version_label, 3, 3, Qt.AlignLeft)
        fallback_azure_versions = [
            '2025-01-01-preview',
            '2024-12-01-preview', 
            '2024-10-01-preview',
            '2024-08-01-preview',
            '2024-06-01',
            '2024-02-01',
            '2023-12-01-preview'
        ]
        self.fallback_azure_api_version_combo = QComboBox()
        self.fallback_azure_api_version_combo.addItems(fallback_azure_versions)
        self.fallback_azure_api_version_combo.setCurrentText('2025-01-01-preview')
        self.fallback_azure_api_version_combo.setStyleSheet("font-size: 7pt;")
        self.fallback_azure_api_version_combo.setMaximumWidth(180)
        self._disable_combobox_mousewheel(self.fallback_azure_api_version_combo)  # Disable mousewheel
        add_fallback_grid.addWidget(self.fallback_azure_api_version_combo, 3, 4, 1, 1, Qt.AlignLeft)
        
        # Initially hide the endpoint fields when toggle is off
        self._toggle_fallback_individual_endpoint_fields()
        
        # Fallback keys list
        self._create_fallback_list(fallback_frame_layout)
        
        # Add fallback container to parent
        fallback_container_layout.addWidget(fallback_frame)
        parent_layout.addWidget(self.fallback_container)
        
        # Initially disable if checkbox is unchecked
        self._toggle_fallback_section()

    def _create_fallback_list(self, parent_layout):
        """Create the fallback keys list"""
        # Label - store reference for hide/show
        self.fallback_list_label = QLabel("Fallback Keys (tried in order):")
        list_label_font = QFont()
        list_label_font.setBold(True)
        self.fallback_list_label.setFont(list_label_font)
        parent_layout.addWidget(self.fallback_list_label)
        
        # Container for tree and buttons
        self.fallback_tree_container = QWidget()
        container_layout = QHBoxLayout(self.fallback_tree_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left side: Move buttons - store reference for hide/show
        self.fallback_move_frame = QWidget()
        move_layout = QVBoxLayout(self.fallback_move_frame)
        move_layout.setContentsMargins(0, 0, 5, 0)
        
        order_label = QLabel("Order")
        order_font = QFont()
        order_font.setBold(True)
        order_label.setFont(order_font)
        move_layout.addWidget(order_label)
        
        up_btn = QPushButton("↑")
        up_btn.setFixedWidth(40)
        up_btn.clicked.connect(lambda: self._move_fallback_key('up'))
        move_layout.addWidget(up_btn)
        
        down_btn = QPushButton("↓")
        down_btn.setFixedWidth(40)
        down_btn.clicked.connect(lambda: self._move_fallback_key('down'))
        move_layout.addWidget(down_btn)
        
        move_layout.addStretch()
        container_layout.addWidget(self.fallback_move_frame)
        
        # Right side: TreeWidget with drag and drop
        self.fallback_tree = QTreeWidget()
        self.fallback_tree.setHeaderLabels(['API Key', 'Model', 'Status', 'Times Used'])
        self.fallback_tree.setColumnWidth(0, 220)
        self.fallback_tree.setColumnWidth(1, 220)
        self.fallback_tree.setColumnWidth(2, 120)
        self.fallback_tree.setColumnWidth(3, 100)
        self.fallback_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.fallback_tree.customContextMenuRequested.connect(self._show_fallback_context_menu)
        self.fallback_tree.setMinimumHeight(150)
        
        # Enable drag and drop for fallback tree
        self.fallback_tree.setDragEnabled(True)
        self.fallback_tree.setAcceptDrops(True)
        self.fallback_tree.setDragDropMode(QAbstractItemView.InternalMove)
        self.fallback_tree.setDefaultDropAction(Qt.MoveAction)
        
        # Store original dropEvent and override
        self._fallback_tree_original_dropEvent = self.fallback_tree.dropEvent
        self.fallback_tree.dropEvent = self._on_fallback_tree_drop
        
        container_layout.addWidget(self.fallback_tree)
        
        parent_layout.addWidget(self.fallback_tree_container)
        
        # Action buttons - store reference for toggling
        self.fallback_action_frame = QWidget()
        fallback_action_layout = QHBoxLayout(self.fallback_action_frame)
        fallback_action_layout.setContentsMargins(0, 10, 0, 0)
        
        test_selected_btn = QPushButton("Test Selected")
        test_selected_btn.clicked.connect(self._test_selected_fallback)
        fallback_action_layout.addWidget(test_selected_btn)

        test_all_btn = QPushButton("Test All")
        test_all_btn.clicked.connect(self._test_all_fallbacks)
        fallback_action_layout.addWidget(test_all_btn)
    
        remove_selected_btn = QPushButton("Remove Selected")
        remove_selected_btn.clicked.connect(self._remove_selected_fallback)
        fallback_action_layout.addWidget(remove_selected_btn)
        
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self._clear_all_fallbacks)
        fallback_action_layout.addWidget(clear_all_btn)
        
        fallback_action_layout.addStretch()
        parent_layout.addWidget(self.fallback_action_frame)
        
        # Load existing fallback keys
        self._load_fallback_keys()

    def _test_all_fallbacks(self):
        """Test all fallback keys in parallel"""
        fallback_keys = self.translator_gui.config.get('fallback_keys', [])
        
        if not fallback_keys:
            QMessageBox.warning(self, "Warning", "No fallback keys to test")
            return
        
        # Update UI to show testing status for all keys
        for i in range(self.fallback_tree.topLevelItemCount()):
            item = self.fallback_tree.topLevelItem(i)
            if item:
                item.setText(2, "⏳ Testing...")
        
        # Ensure UnifiedClient uses the same shared pool instance
        try:
            from unified_api_client import UnifiedClient
            UnifiedClient._api_key_pool = self.key_pool
        except Exception:
            pass
        
        # Submit all tests to executor in parallel
        for i, key_data in enumerate(fallback_keys):
            self._test_single_fallback_key(key_data, i)

    def _show_fallback_context_menu(self, position):
        """Show context menu for fallback keys - includes model name editing"""
        # Get item at position
        item = self.fallback_tree.itemAt(position)
        if not item:
            return
        
        # Select item if not already selected
        if item not in self.fallback_tree.selectedItems():
            self.fallback_tree.setCurrentItem(item)
        
        # Create context menu
        menu = QMenu(self)
        
        # Get index for position info
        index = self.fallback_tree.indexOfTopLevelItem(item)
        fallback_keys = self.translator_gui.config.get('fallback_keys', [])
        total = len(fallback_keys)
        
        # Reorder submenu
        if total > 1:  # Only show reorder if there's more than one key
            reorder_menu = menu.addMenu("Reorder")
            if index > 0:
                up_action = reorder_menu.addAction("Move Up")
                up_action.triggered.connect(lambda: self._move_fallback_key('up'))
            if index < total - 1:
                down_action = reorder_menu.addAction("Move Down")
                down_action.triggered.connect(lambda: self._move_fallback_key('down'))
            menu.addSeparator()
        
        # Add Change Model option
        selected_count = len(self.fallback_tree.selectedItems())
        if selected_count > 1:
            change_model_action = menu.addAction(f"Change Model ({selected_count} selected)")
        else:
            change_model_action = menu.addAction("Change Model")
        change_model_action.triggered.connect(self._change_fallback_model_for_selected)
        
        menu.addSeparator()
        
        # Individual Endpoint options for fallback keys
        if index < len(fallback_keys):
            key_data = fallback_keys[index]
            endpoint_enabled = key_data.get('use_individual_endpoint', False)
            endpoint_url = key_data.get('azure_endpoint', '')
            
            if endpoint_enabled and endpoint_url:
                config_action = menu.addAction("✅ Individual Endpoint")
                config_action.triggered.connect(lambda: self._configure_fallback_individual_endpoint(index))
                disable_action = menu.addAction("Disable Individual Endpoint")
                disable_action.triggered.connect(lambda: self._toggle_fallback_individual_endpoint(index, False))
            else:
                config_action = menu.addAction("🔧 Configure Individual Endpoint")
                config_action.triggered.connect(lambda: self._configure_fallback_individual_endpoint(index))
        
        menu.addSeparator()
        
        # Test and Remove options
        test_action = menu.addAction("Test")
        test_action.triggered.connect(self._test_selected_fallback)
        menu.addSeparator()
        remove_action = menu.addAction("Remove")
        remove_action.triggered.connect(self._remove_selected_fallback)
        
        if total > 1:
            clear_action = menu.addAction("Clear All")
            clear_action.triggered.connect(self._clear_all_fallbacks)
        
        # Show menu
        menu.exec_(self.fallback_tree.viewport().mapToGlobal(position))


    def _change_fallback_model_for_selected(self):
        """Change model name for selected fallback keys"""
        selected = self.fallback_tree.selectedItems()
        if not selected:
            return
        
        # Get fallback keys
        fallback_keys = self.translator_gui.config.get('fallback_keys', [])
        
        # Create simple dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Change Model for {len(selected)} Fallback Keys")
        dialog.resize(400, 130)
        self._set_icon(dialog)
        
        # Main layout
        main_layout = QVBoxLayout(dialog)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Label
        label = QLabel("Enter new model name (press Enter to apply):")
        main_layout.addWidget(label)
        
        # Full model list
        all_models = get_model_options()
        
        model_combo = QComboBox()
        model_combo.addItems(all_models)
        model_combo.setEditable(True)
        main_layout.addWidget(model_combo)
        
        # Get current model from first selected item as default
        selected_indices = [self.fallback_tree.indexOfTopLevelItem(item) for item in selected]
        if selected_indices and selected_indices[0] < len(fallback_keys):
            current_model = fallback_keys[selected_indices[0]].get('model', '')
            model_combo.setCurrentText(current_model)
            model_combo.lineEdit().selectAll()
        
        def apply_change():
            new_model = model_combo.currentText().strip()
            if new_model:
                # Update all selected fallback keys
                for item in selected:
                    index = self.fallback_tree.indexOfTopLevelItem(item)
                    if index < len(fallback_keys):
                        fallback_keys[index]['model'] = new_model
                
                # Save to config
                self.translator_gui.config['fallback_keys'] = fallback_keys
                self.translator_gui.save_config(show_message=False)
                
                # Reload the list
                self._load_fallback_keys()
                
                # Show status
                self._show_status(f"Changed model to '{new_model}' for {len(selected)} fallback keys")
                
                dialog.accept()
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(apply_change)
        apply_btn.setDefault(True)
        button_layout.addWidget(apply_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        main_layout.addLayout(button_layout)
        
        # Focus on the combobox
        model_combo.setFocus()
        
        # Show dialog
        dialog.exec_()

    def _load_fallback_keys(self):
        """Load fallback keys from config"""
        fallback_keys = self.translator_gui.config.get('fallback_keys', [])
        
        # Clear tree
        self.fallback_tree.clear()
        
        # Add keys to tree
        for key_data in fallback_keys:
            api_key = key_data.get('api_key', '')
            model = key_data.get('model', '')
            times_used = int(key_data.get('times_used', 0))
            
            # Mask API key
            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else api_key
            
            # Insert into tree
            item = QTreeWidgetItem([masked_key, model, "Not tested", str(times_used)])
            item.setForeground(0, Qt.gray)  # Untested styling
            item.setForeground(1, Qt.gray)
            item.setForeground(2, Qt.gray)
            item.setForeground(3, Qt.gray)
            self.fallback_tree.addTopLevelItem(item)

    def _add_fallback_key(self):
        """Add a new fallback key with optional Google credentials and Azure endpoint"""
        api_key = self.fallback_key_entry.text().strip()
        model = self.fallback_model_combo.currentText().strip()
        google_credentials = self.fallback_google_creds_entry.text().strip() or None
        google_region = self.fallback_google_region_entry.text().strip() or None
        
        # Only use individual endpoint if toggle is enabled
        use_individual_endpoint = self.fallback_individual_endpoint_toggle.isChecked()
        azure_endpoint = self.fallback_azure_endpoint_entry.text().strip() if use_individual_endpoint else None
        azure_api_version = self.fallback_azure_api_version_combo.currentText().strip() if use_individual_endpoint else None
        
        if not api_key or not model:
            QMessageBox.critical(self, "Error", "Please enter both API key and model name")
            return
        
        # Get current fallback keys
        fallback_keys = self.translator_gui.config.get('fallback_keys', [])
        
        # Add new key with additional fields
        fallback_keys.append({
            'api_key': api_key,
            'model': model,
            'google_credentials': google_credentials,
            'azure_endpoint': azure_endpoint,
            'google_region': google_region,
            'azure_api_version': azure_api_version,
            'use_individual_endpoint': use_individual_endpoint,
            'times_used': 0
        })
        
        # Save to config
        self.translator_gui.config['fallback_keys'] = fallback_keys
        self.translator_gui.save_config(show_message=False)
        
        # Clear inputs
        self.fallback_key_entry.clear()
        self.fallback_model_combo.setCurrentText("")
        self.fallback_google_creds_entry.clear()
        self.fallback_azure_endpoint_entry.clear()
        self.fallback_google_region_entry.setText("us-east5")
        self.fallback_azure_api_version_combo.setCurrentText('2025-01-01-preview')
        self.fallback_individual_endpoint_toggle.setChecked(False)
        # Update the UI to disable endpoint fields
        self._toggle_fallback_individual_endpoint_fields()
        
        # Reload list
        self._load_fallback_keys()
        
        # Show success
        extras = []
        if google_credentials:
            extras.append(f"Google: {os.path.basename(google_credentials)}")
        if azure_endpoint:
            extras.append(f"Azure: {azure_endpoint[:30]}...")
        
        extra_info = f" ({', '.join(extras)})" if extras else ""
        self._show_status(f"Added fallback key for model: {model}{extra_info}")

    def _move_fallback_key(self, direction):
        """Move selected fallback key up or down"""
        selected = self.fallback_tree.selectedItems()
        if not selected:
            return
        
        item = selected[0]
        index = self.fallback_tree.indexOfTopLevelItem(item)
        
        # Get current fallback keys
        fallback_keys = self.translator_gui.config.get('fallback_keys', [])
        
        if index >= len(fallback_keys):
            return
        
        new_index = index
        if direction == 'up' and index > 0:
            new_index = index - 1
        elif direction == 'down' and index < len(fallback_keys) - 1:
            new_index = index + 1
        
        if new_index != index:
            # Swap keys
            fallback_keys[index], fallback_keys[new_index] = fallback_keys[new_index], fallback_keys[index]
            
            # Save to config
            self.translator_gui.config['fallback_keys'] = fallback_keys
            self.translator_gui.save_config(show_message=False)
            
            # Reload list
            self._load_fallback_keys()
            
            # Reselect item
            if new_index < self.fallback_tree.topLevelItemCount():
                item = self.fallback_tree.topLevelItem(new_index)
                if item:
                    self.fallback_tree.setCurrentItem(item)
                    item.setSelected(True)

    def _test_selected_fallback(self):
        """Test selected fallback key"""
        selected = self.fallback_tree.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select a fallback key to test")
            return
        
        index = self.fallback_tree.indexOfTopLevelItem(selected[0])
        fallback_keys = self.translator_gui.config.get('fallback_keys', [])
        
        if index >= len(fallback_keys):
            return
        
        # Update UI to show testing status immediately
        if index < self.fallback_tree.topLevelItemCount():
            item = self.fallback_tree.topLevelItem(index)
            if item:
                item.setText(2, "⏳ Testing...")
        
        key_data = fallback_keys[index]
        
        # Ensure UnifiedClient uses the same shared pool instance
        try:
            from unified_api_client import UnifiedClient
            UnifiedClient._api_key_pool = self.key_pool
        except Exception:
            pass
        
        # Run test on main thread using QTimer (non-blocking)
        QTimer.singleShot(100, lambda: self._test_single_fallback_key(key_data, index))

    def _update_fallback_test_result(self, index, success):
        """Update fallback tree item with test result and bump times used"""
        # Increment times_used in config
        fallback_keys = self.translator_gui.config.get('fallback_keys', [])
        if index < len(fallback_keys):
            try:
                fallback_keys[index]['times_used'] = int(fallback_keys[index].get('times_used', 0)) + 1
                # Persist
                self.translator_gui.config['fallback_keys'] = fallback_keys
                self.translator_gui.save_config(show_message=False)
            except Exception:
                pass
        
        if index < self.fallback_tree.topLevelItemCount():
            item = self.fallback_tree.topLevelItem(index)
            if item:
                # Update status (column 2)
                item.setText(2, "✅ Passed" if success else "❌ Failed")
                # Update times used cell (column 3)
                try:
                    current_times = int(item.text(3))
                    item.setText(3, str(current_times + 1))
                except Exception:
                    item.setText(3, "1")

    def _test_single_fallback_key(self, key_data, index):
        """Test a single fallback key - REAL API TEST"""
        api_key = key_data.get('api_key', '')
        model = key_data.get('model', '')
        
        print(f"[DEBUG] Starting REAL fallback key test for {model}")
        
        # Run REAL API test using executor like translation does
        from concurrent.futures import ThreadPoolExecutor
        from unified_api_client import UnifiedClient
        
        # Use shared executor from main GUI
        if hasattr(self.translator_gui, '_ensure_executor'):
            self.translator_gui._ensure_executor()
        executor = getattr(self.translator_gui, 'executor', None)
        
        def run_api_test():
            try:
                client = UnifiedClient(
                    api_key=api_key,
                    model=model,
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
                    content, _ = response
                    if content and "test successful" in content.lower():
                        print(f"[DEBUG] Fallback key test completed for {model}: PASSED")
                        # Update directly - we're in executor thread
                        self._update_fallback_test_result(index, True)
                        return
                
                # Failed
                print(f"[DEBUG] Fallback key test completed for {model}: FAILED")
                self._update_fallback_test_result(index, False)
            except Exception as e:
                print(f"[DEBUG] Fallback key test error for {model}: {e}")
                self._update_fallback_test_result(index, False)
        
        # Submit to shared executor like translation does
        if executor:
            executor.submit(run_api_test)
        else:
            # Fallback to thread if no executor
            thread = threading.Thread(target=run_api_test, daemon=True)
            thread.start()

    def _remove_selected_fallback(self):
        """Remove selected fallback key"""
        selected = self.fallback_tree.selectedItems()
        if not selected:
            return
        
        reply = QMessageBox.question(self, "Confirm", "Remove selected fallback key?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            index = self.fallback_tree.indexOfTopLevelItem(selected[0])
            
            # Get current fallback keys
            fallback_keys = self.translator_gui.config.get('fallback_keys', [])
            
            if index < len(fallback_keys):
                del fallback_keys[index]
                
                # Save to config
                self.translator_gui.config['fallback_keys'] = fallback_keys
                self.translator_gui.save_config(show_message=False)
                
                # Reload list
                self._load_fallback_keys()
                
                self._show_status("Removed fallback key")

    def _clear_all_fallbacks(self):
        """Clear all fallback keys"""
        if self.fallback_tree.topLevelItemCount() == 0:
            return
        
        reply = QMessageBox.question(self, "Confirm", "Remove ALL fallback keys?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Clear fallback keys
            self.translator_gui.config['fallback_keys'] = []
            self.translator_gui.save_config(show_message=False)
            
            # Reload list
            self._load_fallback_keys()
            
            self._show_status("Cleared all fallback keys")

    def _toggle_fallback_section(self):
        """Toggle fallback section - simply hide/show elements"""
        enabled = self.use_fallback_checkbox.isChecked()
        
        # Show/hide the input frame
        if hasattr(self, 'add_fallback_frame'):
            self.add_fallback_frame.setVisible(enabled)
        
        # Show/hide the list label
        if hasattr(self, 'fallback_list_label'):
            self.fallback_list_label.setVisible(enabled)
        
        # Show/hide the tree container (includes move buttons and tree)
        if hasattr(self, 'fallback_tree_container'):
            self.fallback_tree_container.setVisible(enabled)
        
        # Show/hide action buttons
        if hasattr(self, 'fallback_action_frame'):
            self.fallback_action_frame.setVisible(enabled)
        
        # Clear selection when disabled
        if hasattr(self, 'fallback_tree') and not enabled:
            self.fallback_tree.clearSelection()
        
        # === Check if multi-key is also disabled ===
        multikey_enabled = self.enabled_checkbox.isChecked() if hasattr(self, 'enabled_checkbox') else False
        both_disabled = not enabled and not multikey_enabled
        
        # === Adjust fallback container spacing ===
        if hasattr(self, 'fallback_container'):
            layout = self.fallback_container.layout()
            if layout:
                if both_disabled:
                    layout.setContentsMargins(0, 0, 0, 0)  # No top margin when both are off
                else:
                    layout.setContentsMargins(0, 5, 0, 0)  # Normal spacing otherwise
        
        # === Hide fallback separator when both are off ===
        if hasattr(self, 'fallback_separator'):
            self.fallback_separator.setVisible(not both_disabled)
        
        # === Adjust layout spacing ===
        if hasattr(self, 'scrollable_layout'):
            if both_disabled:
                self.scrollable_layout.setSpacing(0)  # No spacing when both disabled
            else:
                self.scrollable_layout.setSpacing(10)  # Normal spacing otherwise

    def _toggle_fallback_visibility(self):
        """Toggle fallback key visibility"""
        if self.fallback_key_entry.echoMode() == QLineEdit.Password:
            self.fallback_key_entry.setEchoMode(QLineEdit.Normal)
            self.show_fallback_btn.setText('🔒')
        else:
            self.fallback_key_entry.setEchoMode(QLineEdit.Password)
            self.show_fallback_btn.setText('👁')
    
    def _toggle_fallback_individual_endpoint_fields(self):
        """Toggle visibility and state of fallback individual endpoint fields"""
        enabled = self.fallback_individual_endpoint_toggle.isChecked()
        
        # Show/hide and enable/disable endpoint fields
        self.fallback_individual_endpoint_label.setVisible(enabled)
        self.fallback_azure_endpoint_entry.setVisible(enabled)
        self.fallback_individual_api_version_label.setVisible(enabled)
        self.fallback_azure_api_version_combo.setVisible(enabled)
        
        self.fallback_azure_endpoint_entry.setEnabled(enabled)
        self.fallback_azure_api_version_combo.setEnabled(enabled)
        
        if not enabled:
            # Clear the fields when disabled
            self.fallback_azure_endpoint_entry.clear()
            self.fallback_azure_api_version_combo.setCurrentText('2025-01-01-preview')
    
    def _configure_fallback_individual_endpoint(self, fallback_index):
        """Configure individual endpoint for a fallback key"""
        fallback_keys = self.translator_gui.config.get('fallback_keys', [])
        if fallback_index >= len(fallback_keys):
            return
        
        key_data = fallback_keys[fallback_index]
        
        # Create a temporary APIKeyEntry object for compatibility with IndividualEndpointDialog
        temp_key = APIKeyEntry(
            api_key=key_data.get('api_key', ''),
            model=key_data.get('model', ''),
            cooldown=60,  # Not used for fallback keys
            enabled=True,
            google_credentials=key_data.get('google_credentials'),
            azure_endpoint=key_data.get('azure_endpoint'),
            google_region=key_data.get('google_region'),
            azure_api_version=key_data.get('azure_api_version'),
            use_individual_endpoint=key_data.get('use_individual_endpoint', False)
        )
        
        # Define callback to update config after dialog closes
        def on_endpoint_configured():
            # Update the fallback key with new values
            fallback_keys[fallback_index]['azure_endpoint'] = temp_key.azure_endpoint
            fallback_keys[fallback_index]['azure_api_version'] = temp_key.azure_api_version
            fallback_keys[fallback_index]['use_individual_endpoint'] = temp_key.use_individual_endpoint
            
            # Save to config
            self.translator_gui.config['fallback_keys'] = fallback_keys
            self.translator_gui.save_config(show_message=False)
            
            # Reload the fallback list
            self._load_fallback_keys()
            
            # Show status
            status = "configured" if temp_key.use_individual_endpoint else "disabled"
            self._show_status(f"Individual endpoint {status} for fallback key")
        
        # Create individual endpoint dialog using the class
        if IndividualEndpointDialog is None:
            QMessageBox.critical(self, "Error", "IndividualEndpointDialog is not available.")
            return
        dialog = IndividualEndpointDialog(self, self.translator_gui, temp_key, on_endpoint_configured, self._show_status)
        dialog.exec_()
    
    def _toggle_fallback_individual_endpoint(self, fallback_index, enabled):
        """Quick toggle individual endpoint on/off for fallback key"""
        fallback_keys = self.translator_gui.config.get('fallback_keys', [])
        if fallback_index >= len(fallback_keys):
            return
        
        fallback_keys[fallback_index]['use_individual_endpoint'] = enabled
        
        # Save to config
        self.translator_gui.config['fallback_keys'] = fallback_keys
        self.translator_gui.save_config(show_message=False)
        
        # Reload fallback list
        self._load_fallback_keys()
        
        # Show status
        status = "enabled" if enabled else "disabled"
        model = fallback_keys[fallback_index].get('model', 'unknown')
        self._show_status(f"Individual endpoint {status} for fallback key ({model})")

    def _create_button_bar(self, parent_layout):
        """Create the bottom button bar as a fixed section"""
        # Create container for separator and buttons
        button_container = QWidget()
        button_container_layout = QVBoxLayout(button_container)
        button_container_layout.setContentsMargins(0, 0, 0, 0)
        button_container_layout.setSpacing(0)
        
        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("QFrame { color: #3a3a3a; background-color: #3a3a3a; }")
        button_container_layout.addWidget(separator)
        
        # Create button frame
        self.button_frame = QWidget()
        self.button_frame.setStyleSheet("""
            QWidget {
                background-color: #252525;
                padding: 0px;
            }
        """)
        button_layout = QHBoxLayout(self.button_frame)
        button_layout.setContentsMargins(20, 15, 20, 15)
        
        # Import/Export
        import_btn = QPushButton("Import")
        import_btn.setMinimumHeight(40)
        import_btn.setStyleSheet("QPushButton { font-size: 11pt; padding: 8px 20px; }")
        import_btn.clicked.connect(self._import_keys)
        button_layout.addWidget(import_btn)
        
        export_btn = QPushButton("Export")
        export_btn.setMinimumHeight(40)
        export_btn.setStyleSheet("QPushButton { font-size: 11pt; padding: 8px 20px; }")
        export_btn.clicked.connect(self._export_keys)
        button_layout.addWidget(export_btn)
        
        button_layout.addStretch()
        
        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumHeight(40)
        cancel_btn.setStyleSheet("QPushButton { font-size: 11pt; padding: 8px 20px; }")
        cancel_btn.clicked.connect(self._on_close)
        button_layout.addWidget(cancel_btn)
        
        # Save button
        save_btn = QPushButton("Save & Close")
        save_btn.setMinimumHeight(40)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a7ba7;
                color: white;
                font-weight: bold;
                font-size: 11pt;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #5a9fd4;
            }
        """)
        save_btn.clicked.connect(self._save_and_close)
        button_layout.addWidget(save_btn)
        
        # Add button frame to container
        button_container_layout.addWidget(self.button_frame)
        
        # Add button container to parent layout
        parent_layout.addWidget(button_container)
 
    def _create_key_list_section(self, parent_layout):
        """Create the key list section with inline editing and rearrangement controls"""
        # Store reference for enabling/disabling
        self.key_list_frame = QGroupBox("API Keys")
        list_frame_layout = QVBoxLayout(self.key_list_frame)
        list_frame_layout.setContentsMargins(15, 15, 15, 15)
        
        # Add primary key indicator frame at the top with improved styling
        primary_frame = QFrame()
        primary_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2d5a7b, stop:0.5 #4a7ba7, stop:1 #2d5a7b);
                border: none;
                border-radius: 6px;
            }
        """)
        primary_frame_layout = QVBoxLayout(primary_frame)
        primary_frame_layout.setContentsMargins(8, 8, 8, 8)
        
        self.primary_key_label = QLabel("⭐ PRIMARY KEY: Position #1 will be used first in rotation ⭐")
        self.primary_key_label.setStyleSheet("""
            QLabel {
                background: transparent;
                color: #ffffff;
                font-weight: bold;
                font-size: 11pt;
                padding: 2px;
            }
        """)
        primary_label_font = QFont()
        primary_label_font.setBold(True)
        primary_label_font.setPointSize(11)
        self.primary_key_label.setFont(primary_label_font)
        self.primary_key_label.setAlignment(Qt.AlignCenter)
        primary_frame_layout.addWidget(self.primary_key_label)
        
        list_frame_layout.addWidget(primary_frame)
        
        # Main container with treeview and controls
        main_container = QWidget()
        main_container_layout = QHBoxLayout(main_container)
        main_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left side: Move buttons
        move_frame = QWidget()
        move_layout = QVBoxLayout(move_frame)
        move_layout.setContentsMargins(0, 0, 5, 0)
        
        reorder_label = QLabel("Reorder")
        reorder_font = QFont()
        reorder_font.setBold(True)
        reorder_label.setFont(reorder_font)
        move_layout.addWidget(reorder_label)
        
        # Move to top button with expanded width
        top_btn = QPushButton("⬆⬆")
        top_btn.setFixedSize(55, 32)
        top_btn.setStyleSheet("QPushButton { font-size: 14pt; padding: 2px; }")
        top_btn.clicked.connect(lambda: self._move_key('top'))
        move_layout.addWidget(top_btn)
        
        # Move up button with expanded width
        up_btn = QPushButton("⬆")
        up_btn.setFixedSize(55, 32)
        up_btn.setStyleSheet("QPushButton { font-size: 16pt; padding: 2px; }")
        up_btn.clicked.connect(lambda: self._move_key('up'))
        move_layout.addWidget(up_btn)
        
        # Move down button with expanded width
        down_btn = QPushButton("⬇")
        down_btn.setFixedSize(55, 32)
        down_btn.setStyleSheet("QPushButton { font-size: 16pt; padding: 2px; }")
        down_btn.clicked.connect(lambda: self._move_key('down'))
        move_layout.addWidget(down_btn)
        
        # Move to bottom button with expanded width
        bottom_btn = QPushButton("⬇⬇")
        bottom_btn.setFixedSize(55, 32)
        bottom_btn.setStyleSheet("QPushButton { font-size: 14pt; padding: 2px; }")
        bottom_btn.clicked.connect(lambda: self._move_key('bottom'))
        move_layout.addWidget(bottom_btn)
        
        # Spacer
        move_layout.addSpacing(10)
        
        # Position label
        self.position_label = QLabel()
        self.position_label.setStyleSheet("color: gray;")
        move_layout.addWidget(self.position_label)
        move_layout.addStretch()
        
        main_container_layout.addWidget(move_frame)
        
        # Right side: TreeWidget with drag and drop support
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['API Key', 'Model', 'Cooldown', 'Status', 'Success', 'Errors', 'Times Used'])
        # Adjusted column widths: balanced distribution
        self.tree.setColumnWidth(0, 125)  # API Key (decreased from 140)
        self.tree.setColumnWidth(1, 230)  # Model (decreased from 320)
        self.tree.setColumnWidth(2, 80)   # Cooldown
        self.tree.setColumnWidth(3, 80)   # Status
        self.tree.setColumnWidth(4, 65)   # Success (increased from 40)
        self.tree.setColumnWidth(5, 60)   # Errors (increased from 40)
        self.tree.setColumnWidth(6, 90)   # Times Used
        
        # Set header font
        header = self.tree.header()
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(11)
        header.setFont(header_font)
        
        # Set context menu
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        
        # Set selection mode
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        # Enable drag and drop
        self.tree.setDragEnabled(True)
        self.tree.setAcceptDrops(True)
        self.tree.setDragDropMode(QAbstractItemView.InternalMove)
        self.tree.setDefaultDropAction(Qt.MoveAction)
        
        # Connect drop event to custom handler
        # Store original dropEvent to wrap it
        self._tree_original_dropEvent = self.tree.dropEvent
        self.tree.dropEvent = self._on_tree_drop
        
        # Connect signals
        self.tree.itemClicked.connect(self._on_click)
        self.tree.itemSelectionChanged.connect(self._on_selection_change)
        
        # Track editing state
        self.edit_widget = None
        
        main_container_layout.addWidget(self.tree)
        list_frame_layout.addWidget(main_container)
        
        # Action buttons
        action_frame = QWidget()
        action_layout = QHBoxLayout(action_frame)
        action_layout.setContentsMargins(0, 10, 0, 0)
        
        # Store references to action buttons for enabling/disabling
        self.test_selected_btn = QPushButton("Test Selected")
        self.test_selected_btn.clicked.connect(self._test_selected)
        action_layout.addWidget(self.test_selected_btn)
        
        self.test_all_btn = QPushButton("Test All")
        self.test_all_btn.clicked.connect(self._test_all)
        action_layout.addWidget(self.test_all_btn)
        
        self.enable_selected_btn = QPushButton("Enable Selected")
        self.enable_selected_btn.clicked.connect(self._enable_selected)
        action_layout.addWidget(self.enable_selected_btn)
        
        self.disable_selected_btn = QPushButton("Disable Selected")
        self.disable_selected_btn.clicked.connect(self._disable_selected)
        action_layout.addWidget(self.disable_selected_btn)
        
        self.remove_selected_btn = QPushButton("Remove Selected")
        self.remove_selected_btn.clicked.connect(self._remove_selected)
        action_layout.addWidget(self.remove_selected_btn)
        
        action_layout.addStretch()
        
        # Stats label
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("color: gray;")
        stats_font = QFont()
        stats_font.setPointSize(11)
        self.stats_label.setFont(stats_font)
        action_layout.addWidget(self.stats_label)
        
        list_frame_layout.addWidget(action_frame)
        parent_layout.addWidget(self.key_list_frame)
 
    def _create_add_key_section(self, parent_layout):
        """Create the add key section with Google credentials and Azure endpoint support"""
        # Store reference for enabling/disabling
        self.add_key_frame = QGroupBox("Add New API Key")
        add_grid = QGridLayout(self.add_key_frame)
        add_grid.setContentsMargins(15, 15, 15, 15)
        
        # Row 0: API Key and Model
        add_grid.addWidget(QLabel("API Key:"), 0, 0, Qt.AlignLeft)
        self.api_key_entry = QLineEdit()
        self.api_key_entry.setEchoMode(QLineEdit.Password)
        add_grid.addWidget(self.api_key_entry, 0, 1)
        
        # Toggle visibility button
        self.show_key_btn = QPushButton("👁")
        self.show_key_btn.setFixedWidth(40)
        self.show_key_btn.clicked.connect(self._toggle_key_visibility)
        add_grid.addWidget(self.show_key_btn, 0, 2)
        
        # Model
        add_grid.addWidget(QLabel("Model:"), 0, 3, Qt.AlignLeft)
        add_models = get_model_options()
        self.model_combo = QComboBox()
        self.model_combo.addItems(add_models)
        self.model_combo.setEditable(True)
        self._disable_combobox_mousewheel(self.model_combo)  # Disable mousewheel
        add_grid.addWidget(self.model_combo, 0, 4)
        
        # Row 1: Cooldown and buttons
        add_grid.addWidget(QLabel("Cooldown (s):"), 1, 0, Qt.AlignLeft)
        cooldown_widget = QWidget()
        cooldown_layout = QHBoxLayout(cooldown_widget)
        cooldown_layout.setContentsMargins(0, 0, 0, 0)
        
        self.cooldown_spinbox = QSpinBox()
        self.cooldown_spinbox.setRange(10, 3600)
        self.cooldown_spinbox.setValue(60)
        self.cooldown_spinbox.setMaximumWidth(100)
        self._disable_spinbox_mousewheel(self.cooldown_spinbox)  # Disable mousewheel
        cooldown_layout.addWidget(self.cooldown_spinbox)
        
        cooldown_hint = QLabel("(10-3600)")
        cooldown_hint.setStyleSheet("color: gray;")
        cooldown_layout.addWidget(cooldown_hint)
        cooldown_layout.addStretch()
        add_grid.addWidget(cooldown_widget, 1, 1)
        
        # Add button and Copy Current Key button
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addStretch()
        
        # Store references for enabling/disabling
        self.add_key_btn = QPushButton("Add Key")
        self.add_key_btn.clicked.connect(self._add_key)
        button_layout.addWidget(self.add_key_btn)
        
        self.copy_current_btn = QPushButton("Copy Current Key")
        self.copy_current_btn.clicked.connect(self._copy_current_settings)
        button_layout.addWidget(self.copy_current_btn)
        
        add_grid.addWidget(button_widget, 1, 4, Qt.AlignRight)
        
        # Row 2: Google Credentials (optional, discretely styled)
        google_creds_label = QLabel("Google Creds:")
        google_creds_label.setStyleSheet("color: gray; font-size: 9pt;")
        add_grid.addWidget(google_creds_label, 2, 0, Qt.AlignLeft)
        self.google_creds_entry = QLineEdit()
        self.google_creds_entry.setStyleSheet("font-size: 8pt;")
        add_grid.addWidget(self.google_creds_entry, 2, 1)
        
        # Google credentials browse button
        google_browse_btn = QPushButton("📁")
        google_browse_btn.setFixedWidth(40)
        google_browse_btn.clicked.connect(self._browse_google_credentials)
        add_grid.addWidget(google_browse_btn, 2, 2)
        
        # Google region field
        region_label = QLabel("Region:")
        region_label.setStyleSheet("color: gray;")
        add_grid.addWidget(region_label, 2, 3, Qt.AlignLeft)
        self.google_region_entry = QLineEdit("us-east5")
        self.google_region_entry.setStyleSheet("font-size: 8pt;")
        self.google_region_entry.setMaximumWidth(120)
        add_grid.addWidget(self.google_region_entry, 2, 4, 1, 1, Qt.AlignLeft)
        
        # Row 3: Individual Endpoint Toggle
        self.use_individual_endpoint_var = False
        self.individual_endpoint_toggle = self._create_styled_checkbox("Use Individual Endpoint")
        self.individual_endpoint_toggle.setChecked(False)
        self.individual_endpoint_toggle.toggled.connect(self._toggle_individual_endpoint_fields)
        add_grid.addWidget(self.individual_endpoint_toggle, 3, 0, 1, 2, Qt.AlignLeft)
        
        # Row 4: Individual Endpoint (initially hidden)
        self.individual_endpoint_label = QLabel("Individual Endpoint:")
        self.individual_endpoint_label.setStyleSheet("color: gray; font-size: 9pt;")
        add_grid.addWidget(self.individual_endpoint_label, 4, 0, Qt.AlignLeft)
        self.azure_endpoint_entry = QLineEdit()
        self.azure_endpoint_entry.setStyleSheet("font-size: 8pt;")
        self.azure_endpoint_entry.setEnabled(False)
        add_grid.addWidget(self.azure_endpoint_entry, 4, 1, 1, 2)
        
        # Individual Endpoint API Version (small dropdown, initially hidden)
        self.individual_api_version_label = QLabel("API Ver:")
        self.individual_api_version_label.setStyleSheet("color: gray;")
        add_grid.addWidget(self.individual_api_version_label, 4, 3, Qt.AlignLeft)
        azure_versions = [
            '2025-01-01-preview',
            '2024-12-01-preview', 
            '2024-10-01-preview',
            '2024-08-01-preview',
            '2024-06-01',
            '2024-02-01',
            '2023-12-01-preview'
        ]
        self.azure_api_version_combo = QComboBox()
        self.azure_api_version_combo.addItems(azure_versions)
        self.azure_api_version_combo.setCurrentText('2025-01-01-preview')
        self.azure_api_version_combo.setStyleSheet("font-size: 7pt;")
        self.azure_api_version_combo.setMaximumWidth(180)
        self.azure_api_version_combo.setEnabled(False)
        self._disable_combobox_mousewheel(self.azure_api_version_combo)  # Disable mousewheel
        add_grid.addWidget(self.azure_api_version_combo, 4, 4, 1, 1, Qt.AlignLeft)
        
        # Set column stretch
        add_grid.setColumnStretch(1, 1)
        add_grid.setColumnStretch(4, 1)
        
        # Initially hide the endpoint fields
        self._toggle_individual_endpoint_fields()
        
        parent_layout.addWidget(self.add_key_frame)
    
    def _toggle_individual_endpoint_fields(self):
        """Toggle visibility and state of individual endpoint fields"""
        enabled = self.individual_endpoint_toggle.isChecked()
        
        # Show/hide and enable/disable endpoint fields
        self.individual_endpoint_label.setVisible(enabled)
        self.azure_endpoint_entry.setVisible(enabled)
        self.individual_api_version_label.setVisible(enabled)
        self.azure_api_version_combo.setVisible(enabled)
        
        self.azure_endpoint_entry.setEnabled(enabled)
        self.azure_api_version_combo.setEnabled(enabled)
        
        if not enabled:
            # Clear the fields when disabled
            self.azure_endpoint_entry.clear()
            self.azure_api_version_combo.setCurrentText('2025-01-01-preview')
    
    def _move_key(self, direction):
        """Move selected key in the specified direction"""
        selected = self.tree.selectedItems()
        if not selected or len(selected) != 1:
            return
        
        item = selected[0]
        index = self.tree.indexOfTopLevelItem(item)
        
        if index >= len(self.key_pool.keys):
            return
        
        new_index = index
        
        if direction == 'up' and index > 0:
            new_index = index - 1
        elif direction == 'down' and index < len(self.key_pool.keys) - 1:
            new_index = index + 1
        elif direction == 'top':
            new_index = 0
        elif direction == 'bottom':
            new_index = len(self.key_pool.keys) - 1
        
        if new_index != index:
            # Swap keys in the pool
            with self.key_pool.lock:
                self.key_pool.keys[index], self.key_pool.keys[new_index] = \
                    self.key_pool.keys[new_index], self.key_pool.keys[index]
            
            # Refresh display
            self._refresh_key_list()
            
            # Reselect the moved item
            if new_index < self.tree.topLevelItemCount():
                new_item = self.tree.topLevelItem(new_index)
                if new_item:
                    self.tree.setCurrentItem(new_item)
                    new_item.setSelected(True)
                    self.tree.scrollToItem(new_item)
            
            # Show status
            self._show_status(f"Moved key to position {new_index + 1}")
            
    def _on_selection_change(self):
        """Update position label when selection changes"""
        selected = self.tree.selectedItems()
        if selected:
            index = self.tree.indexOfTopLevelItem(selected[0])
            total = len(self.key_pool.keys)
            self.position_label.setText(f"#{index + 1}/{total}")
        else:
            self.position_label.setText("")

    def _on_tree_drop(self, event):
        """Handle drop event for reordering keys"""
        # Get the item being dropped and its target position
        drop_indicator = self.tree.dropIndicatorPosition()
        target_item = self.tree.itemAt(event.pos())
        
        # Get selected items (items being dragged)
        selected_items = self.tree.selectedItems()
        if not selected_items:
            # Call original drop event if nothing selected
            self._tree_original_dropEvent(event)
            return
        
        # Get indices of selected items
        selected_indices = []
        for item in selected_items:
            index = self.tree.indexOfTopLevelItem(item)
            if index >= 0 and index < len(self.key_pool.keys):
                selected_indices.append(index)
        
        if not selected_indices:
            event.ignore()
            return
        
        selected_indices.sort()
        
        # Determine target index
        if target_item is None:
            # Dropped at the end
            target_index = len(self.key_pool.keys)
        else:
            target_index = self.tree.indexOfTopLevelItem(target_item)
            
            # Adjust based on drop indicator position
            if drop_indicator == QAbstractItemView.BelowItem:
                target_index += 1
            elif drop_indicator == QAbstractItemView.OnItem:
                # Treat as above item
                pass
        
        # Don't do anything if dropping in the same position
        if len(selected_indices) == 1 and selected_indices[0] == target_index:
            event.ignore()
            return
        
        # Reorder keys in the pool
        with self.key_pool.lock:
            # Extract the selected keys
            selected_keys = [self.key_pool.keys[i] for i in selected_indices]
            
            # Remove selected keys from their original positions (in reverse order to maintain indices)
            for index in reversed(selected_indices):
                del self.key_pool.keys[index]
            
            # Adjust target index if items were removed before it
            adjusted_target = target_index
            for index in selected_indices:
                if index < target_index:
                    adjusted_target -= 1
            
            # Insert selected keys at the new position
            for i, key in enumerate(selected_keys):
                self.key_pool.keys.insert(adjusted_target + i, key)
        
        # Refresh the display
        self._refresh_key_list()
        
        # Reselect the moved items
        new_start_index = adjusted_target
        for i in range(len(selected_keys)):
            item = self.tree.topLevelItem(new_start_index + i)
            if item:
                item.setSelected(True)
        
        # Show status
        if len(selected_indices) == 1:
            self._show_status(f"Moved key to position {adjusted_target + 1}")
        else:
            self._show_status(f"Moved {len(selected_indices)} keys to position {adjusted_target + 1}")
        
        event.accept()
    
    def _on_fallback_tree_drop(self, event):
        """Handle drop event for reordering fallback keys"""
        # Get the item being dropped and its target position
        drop_indicator = self.fallback_tree.dropIndicatorPosition()
        target_item = self.fallback_tree.itemAt(event.pos())
        
        # Get selected items (items being dragged)
        selected_items = self.fallback_tree.selectedItems()
        if not selected_items:
            self._fallback_tree_original_dropEvent(event)
            return
        
        # Get current fallback keys
        fallback_keys = self.translator_gui.config.get('fallback_keys', [])
        
        # Get indices of selected items
        selected_indices = []
        for item in selected_items:
            index = self.fallback_tree.indexOfTopLevelItem(item)
            if index >= 0 and index < len(fallback_keys):
                selected_indices.append(index)
        
        if not selected_indices:
            event.ignore()
            return
        
        selected_indices.sort()
        
        # Determine target index
        if target_item is None:
            target_index = len(fallback_keys)
        else:
            target_index = self.fallback_tree.indexOfTopLevelItem(target_item)
            
            if drop_indicator == QAbstractItemView.BelowItem:
                target_index += 1
            elif drop_indicator == QAbstractItemView.OnItem:
                pass
        
        # Don't do anything if dropping in the same position
        if len(selected_indices) == 1 and selected_indices[0] == target_index:
            event.ignore()
            return
        
        # Reorder keys in the fallback list
        selected_keys = [fallback_keys[i] for i in selected_indices]
        
        # Remove selected keys from their original positions (in reverse)
        for index in reversed(selected_indices):
            del fallback_keys[index]
        
        # Adjust target index
        adjusted_target = target_index
        for index in selected_indices:
            if index < target_index:
                adjusted_target -= 1
        
        # Insert selected keys at the new position
        for i, key in enumerate(selected_keys):
            fallback_keys.insert(adjusted_target + i, key)
        
        # Save to config
        self.translator_gui.config['fallback_keys'] = fallback_keys
        self.translator_gui.save_config(show_message=False)
        
        # Reload the list
        self._load_fallback_keys()
        
        # Reselect the moved items
        for i in range(len(selected_keys)):
            item = self.fallback_tree.topLevelItem(adjusted_target + i)
            if item:
                item.setSelected(True)
        
        # Show status
        if len(selected_indices) == 1:
            self._show_status(f"Moved fallback key to position {adjusted_target + 1}")
        else:
            self._show_status(f"Moved {len(selected_indices)} fallback keys to position {adjusted_target + 1}")
        
        event.accept()

    def _refresh_key_list(self):
        """Refresh the key list display preserving test results and highlighting key #1"""
        # Clear tree
        self.tree.clear()
        
        # Update primary key label if it exists
        if hasattr(self, 'primary_key_label'):
            keys = self.key_pool.get_all_keys()
            if keys:
                first_key = keys[0]
                masked = first_key.api_key[:8] + "..." + first_key.api_key[-4:] if len(first_key.api_key) > 12 else first_key.api_key
                self.primary_key_label.setText(f"⭐ PRIMARY KEY: {first_key.model} ({masked}) ⭐")
        
        # Add keys
        keys = self.key_pool.get_all_keys()
        for i, key in enumerate(keys):
            # Mask API key for display
            masked_key = key.api_key[:8] + "..." + key.api_key[-4:] if len(key.api_key) > 12 else key.api_key
            
            # Position indicator
            position = f"#{i+1}"
            if i == 0:
                position = "⭐ #1"
            
            # Determine status based on test results and current state
            if key.last_test_result is None and hasattr(key, '_testing'):
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
            
            # Times used (counter)
            times_used = getattr(key, 'times_used', key.success_count + key.error_count)
            
            # Insert into tree with position column
            item = QTreeWidgetItem([
                masked_key, key.model, f"{key.cooldown}s", status,
                str(key.success_count), str(key.error_count), str(times_used)
            ])
            
            # Set colors based on status
            if tags == ('active',):
                for col in range(7):
                    item.setForeground(col, Qt.green)
            elif tags == ('cooling',):
                for col in range(7):
                    item.setForeground(col, Qt.darkYellow)
            elif tags == ('disabled',):
                for col in range(7):
                    item.setForeground(col, Qt.gray)
            elif tags == ('testing',):
                for col in range(7):
                    item.setForeground(col, Qt.blue)
            elif tags == ('passed',):
                for col in range(7):
                    item.setForeground(col, Qt.darkGreen)
            elif tags == ('failed',):
                for col in range(7):
                    item.setForeground(col, Qt.red)
            elif tags == ('ratelimited',):
                for col in range(7):
                    item.setForeground(col, Qt.darkYellow)
            elif tags == ('error',):
                for col in range(7):
                    item.setForeground(col, Qt.darkRed)
            
            self.tree.addTopLevelItem(item)
        
        # Update stats
        active_count = sum(1 for k in keys if k.enabled and not k.is_cooling_down)
        total_count = len(keys)
        passed_count = sum(1 for k in keys if k.last_test_result == 'passed')
        self.stats_label.setText(f"Keys: {active_count} active / {total_count} total | {passed_count} passed tests")

    def _on_click(self, item, column):
        """Handle click on tree item for inline editing"""
        if not item:
            return
        
        index = self.tree.indexOfTopLevelItem(item)
        if index >= len(self.key_pool.keys):
            return
        
        key = self.key_pool.keys[index]
        
        # Column 1 = Model (editable)
        # Column 2 = Cooldown (editable)
        if column == 1:  # Model column
            # Create inline editor for model
            old_value = item.text(1)
            new_value, ok = self._show_model_edit_dialog(old_value)
            if ok and new_value and new_value != old_value:
                key.model = new_value
                self._refresh_key_list()
                self._show_status(f"Updated model to: {new_value}")
        elif column == 2:  # Cooldown column
            # Create inline editor for cooldown
            old_value = key.cooldown
            new_value, ok = self._show_cooldown_edit_dialog(old_value)
            if ok and new_value != old_value:
                key.cooldown = new_value
                self._refresh_key_list()
                self._show_status(f"Updated cooldown to: {new_value}s")
    
    def _show_model_edit_dialog(self, current_value):
        """Show dialog for editing model name"""
        from PySide6.QtWidgets import QInputDialog
        all_models = get_model_options()
        
        # Use QComboBox-based input dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Model")
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel("Model Name:"))
        combo = QComboBox()
        combo.addItems(all_models)
        combo.setEditable(True)
        combo.setCurrentText(current_value)
        combo.lineEdit().selectAll()
        layout.addWidget(combo)
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        cancel_btn = QPushButton("Cancel")
        
        def accept_dialog():
            dialog.accept()
        
        ok_btn.clicked.connect(accept_dialog)
        cancel_btn.clicked.connect(dialog.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        combo.setFocus()
        
        result = dialog.exec_()
        return (combo.currentText(), result == QDialog.Accepted)
    
    def _show_cooldown_edit_dialog(self, current_value):
        """Show dialog for editing cooldown"""
        from PySide6.QtWidgets import QInputDialog
        value, ok = QInputDialog.getInt(
            self, "Edit Cooldown", "Cooldown (seconds):",
            current_value, 10, 3600, 10
        )
        return (value, ok)

    def _show_context_menu(self, position):
        """Show context menu with reorder options"""
        # Get item at position
        item = self.tree.itemAt(position)
        if not item:
            return
        
        # Select item if not already selected
        if item not in self.tree.selectedItems():
            self.tree.setCurrentItem(item)
        
        # Create context menu
        menu = QMenu(self)
        
        # Reorder submenu
        reorder_menu = menu.addMenu("Reorder")
        top_action = reorder_menu.addAction("Move to Top")
        top_action.triggered.connect(lambda: self._move_key('top'))
        up_action = reorder_menu.addAction("Move Up")
        up_action.triggered.connect(lambda: self._move_key('up'))
        down_action = reorder_menu.addAction("Move Down")
        down_action.triggered.connect(lambda: self._move_key('down'))
        bottom_action = reorder_menu.addAction("Move to Bottom")
        bottom_action.triggered.connect(lambda: self._move_key('bottom'))
        
        menu.addSeparator()
        
        # Add change model option
        selected_count = len(self.tree.selectedItems())
        if selected_count > 1:
            change_action = menu.addAction(f"Change Model ({selected_count} selected)")
        else:
            change_action = menu.addAction("Change Model")
        change_action.triggered.connect(self._change_model_for_selected)
        
        menu.addSeparator()
        
        # Individual Endpoint options
        index = self.tree.indexOfTopLevelItem(item)
        if index < len(self.key_pool.keys):
            key = self.key_pool.keys[index]
            endpoint_enabled = getattr(key, 'use_individual_endpoint', False)
            endpoint_url = getattr(key, 'azure_endpoint', '')
            
            if endpoint_enabled and endpoint_url:
                config_action = menu.addAction("✅ Individual Endpoint")
                config_action.triggered.connect(lambda: self._configure_individual_endpoint(index))
                disable_action = menu.addAction("Disable Individual Endpoint")
                disable_action.triggered.connect(lambda: self._toggle_individual_endpoint(index, False))
            else:
                config_action = menu.addAction("🔧 Configure Individual Endpoint")
                config_action.triggered.connect(lambda: self._configure_individual_endpoint(index))
        
        menu.addSeparator()
        test_action = menu.addAction("Test")
        test_action.triggered.connect(self._test_selected)
        enable_action = menu.addAction("Enable")
        enable_action.triggered.connect(self._enable_selected)
        disable_action = menu.addAction("Disable")
        disable_action.triggered.connect(self._disable_selected)
        menu.addSeparator()
        remove_action = menu.addAction("Remove")
        remove_action.triggered.connect(self._remove_selected)
        
        # Show menu
        menu.exec_(self.tree.viewport().mapToGlobal(position))

    def _change_model_for_selected(self):
        """Change model for all selected entries"""
        selected = self.tree.selectedItems()
        if not selected:
            return
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Change Model for {len(selected)} Keys")
        dialog.setMinimumWidth(400)
        
        # Set icon
        try:
            icon_path = os.path.join(os.path.dirname(__file__), 'icon.ico')
            if os.path.exists(icon_path):
                dialog.setWindowIcon(QIcon(icon_path))
        except Exception:
            pass
        
        # Main layout
        main_layout = QVBoxLayout(dialog)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Label
        label = QLabel("Enter new model name (press Enter to apply):")
        label_font = QFont()
        label_font.setPointSize(10)
        label.setFont(label_font)
        main_layout.addWidget(label)
        
        # Model combo box
        all_models = get_model_options()
        model_combo = QComboBox()
        model_combo.addItems(all_models)
        model_combo.setEditable(True)
        
        # Get current model from first selected item as default
        selected_indices = [self.tree.indexOfTopLevelItem(item) for item in selected]
        if selected_indices and selected_indices[0] < len(self.key_pool.keys):
            current_model = self.key_pool.keys[selected_indices[0]].model
            model_combo.setCurrentText(current_model)
            if model_combo.lineEdit():
                model_combo.lineEdit().selectAll()
        
        main_layout.addWidget(model_combo)
        
        def apply_change():
            new_model = model_combo.currentText().strip()
            if new_model:
                # Update all selected keys
                for item in selected:
                    index = self.tree.indexOfTopLevelItem(item)
                    if index < len(self.key_pool.keys):
                        self.key_pool.keys[index].model = new_model
                
                # Refresh the display
                self._refresh_key_list()
                
                # Show status
                self._show_status(f"Changed model to '{new_model}' for {len(selected)} keys")
                
                dialog.accept()
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        ok_btn = QPushButton("Apply")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(apply_change)
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        main_layout.addLayout(button_layout)
        
        # Set up keyboard shortcuts
        from PySide6.QtGui import QShortcut, QKeySequence
        return_shortcut = QShortcut(QKeySequence(Qt.Key_Return), dialog)
        return_shortcut.activated.connect(apply_change)
        enter_shortcut = QShortcut(QKeySequence(Qt.Key_Enter), dialog)
        enter_shortcut.activated.connect(apply_change)
        escape_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), dialog)
        escape_shortcut.activated.connect(dialog.reject)
        
        # Focus on the combobox
        model_combo.setFocus()
        
        # Show dialog
        dialog.exec_()
    
    def _configure_individual_endpoint(self, key_index):
        """Configure individual endpoint for a specific key"""
        if key_index >= len(self.key_pool.keys):
            return
        
        key = self.key_pool.keys[key_index]
        
        # Create individual endpoint dialog using the class
        if IndividualEndpointDialog is None:
            QMessageBox.critical(self, "Error", "IndividualEndpointDialog is not available.")
            return
        dialog = IndividualEndpointDialog(self, self.translator_gui, key, self._refresh_key_list, self._show_status)
        dialog.exec_()
    
    def _toggle_endpoint_fields(self, enable_checkbox, endpoint_entry, version_combo):
        """Toggle endpoint configuration fields based on enable state"""
        enabled = enable_checkbox.isChecked()
        endpoint_entry.setEnabled(enabled)
        version_combo.setEnabled(enabled)
    
    def _toggle_individual_endpoint(self, key_index, enabled):
        """Quick toggle individual endpoint on/off"""
        if key_index >= len(self.key_pool.keys):
            return
        
        key = self.key_pool.keys[key_index]
        key.use_individual_endpoint = enabled
        
        # Refresh display
        self._refresh_key_list()
        
        # Show status
        status = "enabled" if enabled else "disabled"
        self._show_status(f"Individual endpoint {status} for {key.model}")

    # Additional helper method to swap keys programmatically
    def swap_keys(self, index1: int, index2: int):
        """Swap two keys by their indices"""
        with self.key_pool.lock:
            if 0 <= index1 < len(self.key_pool.keys) and 0 <= index2 < len(self.key_pool.keys):
                self.key_pool.keys[index1], self.key_pool.keys[index2] = \
                    self.key_pool.keys[index2], self.key_pool.keys[index1]
                self._refresh_key_list()
                return True
        return False

    # Method to move a key to a specific position
    def move_key_to_position(self, from_index: int, to_index: int):
        """Move a key from one position to another"""
        with self.key_pool.lock:
            if 0 <= from_index < len(self.key_pool.keys) and 0 <= to_index < len(self.key_pool.keys):
                key = self.key_pool.keys.pop(from_index)
                self.key_pool.keys.insert(to_index, key)
                self._refresh_key_list()
                return True
        return False
    
    # Note: Duplicate _create_button_bar removed - PySide6 version defined earlier at line 1491
    
    def _browse_google_credentials(self):
        """Browse for Google Cloud credentials JSON file"""
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
                        self.google_creds_entry.setText(filename)
                        self._show_status(f"Selected Google credentials: {os.path.basename(filename)}")
                    else:
                        QMessageBox.critical(
                            self,
                            "Error", 
                            "Invalid Google Cloud credentials file. Please select a valid service account JSON file."
                        )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load credentials: {str(e)}")
    
    def _browse_fallback_google_credentials(self):
        """Browse for Google Cloud credentials JSON file for fallback keys"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Google Cloud Credentials JSON for Fallback",
            "",
            "JSON files (*.json);;All files (*.*)"
        )
        
        if filename:
            try:
                # Validate it's a valid Google Cloud credentials file
                with open(filename, 'r') as f:
                    creds_data = json.load(f)
                    if 'type' in creds_data and 'project_id' in creds_data:
                        self.fallback_google_creds_entry.setText(filename)
                        self._show_status(f"Selected fallback Google credentials: {os.path.basename(filename)}")
                    else:
                        QMessageBox.critical(
                            self,
                            "Error", 
                            "Invalid Google Cloud credentials file. Please select a valid service account JSON file."
                        )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load credentials: {str(e)}")
    
    def _attach_model_autofill(self, combo: QComboBox, on_change=None):
        """Attach gentle autofill/autocomplete behavior to a QComboBox.
        
        PySide6 version using QCompleter with similar behavior to the tkinter version:
        - Shows suggestions as user types
        - Prefix-based matching with fallback to contains matching
        - Respects backspace/delete (no forced autocomplete)
        """
        # Set up completer for the combobox
        completer = QCompleter(combo.model())
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setFilterMode(Qt.MatchContains)  # Match anywhere in string
        combo.setCompleter(completer)
        
        # Store callback for changes
        if on_change:
            combo.currentTextChanged.connect(lambda: on_change())
            combo.editTextChanged.connect(lambda: on_change())
        
        # Enable completion while typing
        if combo.lineEdit():
            line_edit = combo.lineEdit()
            
            # Connect to text changed signal for gentle autofill
            def on_text_edited(text):
                if not text:
                    return
                
                # Find first match (prefix first, then contains)
                all_items = [combo.itemText(i) for i in range(combo.count())]
                
                # Try prefix match first
                prefix_matches = [item for item in all_items if item.lower().startswith(text.lower())]
                if prefix_matches:
                    first_match = prefix_matches[0]
                else:
                    # Try contains match
                    contains_matches = [item for item in all_items if text.lower() in item.lower()]
                    first_match = contains_matches[0] if contains_matches else None
                
                # Gentle autofill: only if cursor is at end and we found a match
                if first_match and line_edit.cursorPosition() == len(text):
                    # Check if text is growing (not backspacing)
                    if len(text) > len(getattr(line_edit, '_prev_text', '')):
                        # Only autocomplete if the match starts with what was typed
                        if first_match.lower().startswith(text.lower()) and first_match != text:
                            # Save cursor position
                            cursor_pos = len(text)
                            # Set the full match
                            line_edit.setText(first_match)
                            # Select the auto-filled part
                            line_edit.setSelection(cursor_pos, len(first_match) - cursor_pos)
                
                # Store current text for next comparison
                line_edit._prev_text = text
            
            line_edit.textEdited.connect(on_text_edited)
            line_edit._prev_text = ""

    def _toggle_key_visibility(self):
        """Toggle API key visibility"""
        if self.api_key_entry.echoMode() == QLineEdit.Password:
            self.api_key_entry.setEchoMode(QLineEdit.Normal)
            self.show_key_btn.setText('🔒')
        else:
            self.api_key_entry.setEchoMode(QLineEdit.Password)
            self.show_key_btn.setText('👁')
    
    def _toggle_multi_key_mode(self):
        """Toggle multi-key mode - simply hide/show sections"""
        enabled = self.enabled_checkbox.isChecked()
        self.translator_gui.config['use_multi_api_keys'] = enabled
        self.enabled_var = enabled
        
        # Save the config immediately
        self.translator_gui.save_config(show_message=False)
        
        # === Rotation Settings Frame ===
        if hasattr(self, 'rotation_frame'):
            if enabled:
                self.rotation_frame.setMaximumHeight(16777215)  # Reset to max
                self.rotation_frame.show()
            else:
                self.rotation_frame.hide()
                self.rotation_frame.setMaximumHeight(0)
        
        # === Add Key Section ===
        if hasattr(self, 'add_key_frame'):
            if enabled:
                self.add_key_frame.setMaximumHeight(16777215)  # Reset to max
                self.add_key_frame.show()
            else:
                self.add_key_frame.hide()
                self.add_key_frame.setMaximumHeight(0)
        
        # === Separator ===
        if hasattr(self, 'multikey_separator'):
            if enabled:
                self.multikey_separator.setMaximumHeight(16777215)  # Reset to max
                self.multikey_separator.show()
            else:
                self.multikey_separator.hide()
                self.multikey_separator.setMaximumHeight(0)
        
        # === Key List Section ===
        if hasattr(self, 'key_list_frame'):
            if enabled:
                self.key_list_frame.setMaximumHeight(16777215)  # Reset to max
                self.key_list_frame.show()
            else:
                self.key_list_frame.hide()
                self.key_list_frame.setMaximumHeight(0)
        
        # === Check if fallback is also disabled ===
        fallback_enabled = self.use_fallback_checkbox.isChecked() if hasattr(self, 'use_fallback_checkbox') else False
        both_disabled = not enabled and not fallback_enabled
        
        # === Adjust fallback container spacing ===
        # When both toggles are disabled, eliminate all top spacing
        if hasattr(self, 'fallback_container'):
            layout = self.fallback_container.layout()
            if layout:
                if both_disabled:
                    layout.setContentsMargins(0, 0, 0, 0)  # No top margin when both are off
                else:
                    layout.setContentsMargins(0, 5, 0, 0)  # Normal spacing otherwise
        
        # === Hide fallback separator when both are off ===
        if hasattr(self, 'fallback_separator'):
            self.fallback_separator.setVisible(not both_disabled)
        
        # === Adjust layout spacing ===
        # Reduce spacing between widgets when both toggles are off
        if hasattr(self, 'scrollable_layout'):
            if both_disabled:
                self.scrollable_layout.setSpacing(0)  # No spacing when both disabled
            else:
                self.scrollable_layout.setSpacing(10)  # Normal spacing otherwise
        
        # Show status message
        status = "enabled" if enabled else "disabled"
        self._show_status(f"Multi-Key Mode {status}")
    
    def _copy_current_settings(self):
        """Copy current API key and model from main GUI"""
        # Get current API key and model from translator GUI
        if hasattr(self.translator_gui, 'api_key_entry'):
            api_key = self.translator_gui.api_key_entry.text()
            if api_key:
                self.api_key_entry.setText(api_key)
        
        if hasattr(self.translator_gui, 'model_combo'):
            model = self.translator_gui.model_combo.currentText()
            if model:
                self.model_combo.setCurrentText(model)
    
    def _add_key(self):
        """Add a new API key with optional Google credentials and individual endpoint"""
        api_key = self.api_key_entry.text().strip()
        model = self.model_combo.currentText().strip()
        cooldown = self.cooldown_spinbox.value()
        google_credentials = self.google_creds_entry.text().strip() or None
        google_region = self.google_region_entry.text().strip() or None
        
        # Only use individual endpoint if toggle is enabled
        use_individual_endpoint = self.individual_endpoint_toggle.isChecked()
        azure_endpoint = self.azure_endpoint_entry.text().strip() if use_individual_endpoint else None
        azure_api_version = self.azure_api_version_combo.currentText().strip() if use_individual_endpoint else None
        
        if not api_key or not model:
            QMessageBox.critical(self, "Error", "Please enter both API key and model name")
            return
        
        # Add to pool with new fields
        key_entry = APIKeyEntry(api_key, model, cooldown, enabled=True, 
                               google_credentials=google_credentials, 
                               azure_endpoint=azure_endpoint,
                               google_region=google_region,
                               azure_api_version=azure_api_version,
                               use_individual_endpoint=use_individual_endpoint)
        self.key_pool.add_key(key_entry)
        
        # Clear inputs
        self.api_key_entry.clear()
        self.model_combo.setCurrentText("")
        self.cooldown_spinbox.setValue(60)
        self.google_creds_entry.clear()
        self.azure_endpoint_entry.clear()
        self.google_region_entry.setText("us-east5")
        self.azure_api_version_combo.setCurrentText('2025-01-01-preview')
        self.individual_endpoint_toggle.setChecked(False)
        # Update the UI to hide endpoint fields
        self._toggle_individual_endpoint_fields()
        
        # Refresh list
        self._refresh_key_list()
        
        # Show success
        extras = []
        if google_credentials:
            extras.append(f"Google: {os.path.basename(google_credentials)}")
        if azure_endpoint:
            extras.append(f"Azure: {azure_endpoint[:30]}...")
        
        extra_info = f" ({', '.join(extras)})" if extras else ""
        self._show_status(f"Added key for model: {model}{extra_info}")
    
    # Note: _refresh_key_list is defined earlier in the file (PySide6 version)
    
    def _test_selected(self):
        """Test selected API keys with inline progress"""
        selected = self.tree.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select keys to test")
            return
        
        # Get selected indices
        indices = [self.tree.indexOfTopLevelItem(item) for item in selected]
        
        # Mark keys as testing BEFORE starting thread (in main thread)
        for index in indices:
            if index < len(self.key_pool.keys):
                key = self.key_pool.keys[index]
                key.last_test_result = None
                key._testing = True
                print(f"[DEBUG] Pre-marked key {index} as testing")
        
        # Refresh UI immediately to show testing status
        self._refresh_key_list()
        QApplication.processEvents()  # Force UI update
        
        # Ensure UnifiedClient uses the same shared pool instance
        try:
            from unified_api_client import UnifiedClient
            UnifiedClient._api_key_pool = self.key_pool
        except Exception:
            pass
        
        # Run all tests in parallel using executor
        self._test_results = []
        self._total_tests = len(indices)
        self._completed_tests = 0
        
        # Submit all tests to executor at once
        for index in indices:
            self._submit_single_test(index)

    def _test_all(self):
        """Test all API keys with inline progress"""
        if not self.key_pool.keys:
            QMessageBox.warning(self, "Warning", "No keys to test")
            return
        
        indices = list(range(len(self.key_pool.keys)))
        
        # Mark keys as testing BEFORE starting thread (in main thread)
        for index in indices:
            if index < len(self.key_pool.keys):
                key = self.key_pool.keys[index]
                key.last_test_result = None
                key._testing = True
                print(f"[DEBUG] Pre-marked key {index} as testing")
        
        # Refresh UI immediately to show testing status
        self._refresh_key_list()
        QApplication.processEvents()  # Force UI update
        
        # Ensure UnifiedClient uses the same shared pool instance
        try:
            from unified_api_client import UnifiedClient
            UnifiedClient._api_key_pool = self.key_pool
        except Exception:
            pass
        
        # Run all tests in parallel using executor
        self._test_results = []
        self._total_tests = len(indices)
        self._completed_tests = 0
        
        # Submit all tests to executor at once
        for index in indices:
            self._submit_single_test(index)

    def _submit_single_test(self, index):
        """Submit a single test to executor for parallel execution"""
        if index >= len(self.key_pool.keys):
            return
        
        key = self.key_pool.keys[index]
        print(f"[DEBUG] Submitting test for key {index}: {key.model}")
        
        # Run REAL API test using executor like translation does
        from concurrent.futures import ThreadPoolExecutor
        from unified_api_client import UnifiedClient
        
        # Use shared executor from main GUI
        if hasattr(self.translator_gui, '_ensure_executor'):
            self.translator_gui._ensure_executor()
        executor = getattr(self.translator_gui, 'executor', None)
        
        def run_api_test():
            try:
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
                    content, _ = response
                    if content and "test successful" in content.lower():
                        # Success - update directly from executor thread
                        self._handle_test_result(index, True, "Test passed")
                        return
                
                # Failed - update directly
                self._handle_test_result(index, False, "Unexpected response")
            except Exception as e:
                error_msg = str(e)[:50]
                self._handle_test_result(index, False, f"Error: {error_msg}")
        
        # Submit to shared executor like translation does
        if executor:
            executor.submit(run_api_test)
        else:
            # Fallback to thread if no executor
            thread = threading.Thread(target=run_api_test, daemon=True)
            thread.start()
    
    def _handle_test_result(self, index, success, message):
        """Handle test result from background thread"""
        if index < len(self.key_pool.keys):
            key = self.key_pool.keys[index]
            if success:
                key.mark_success()
                key.set_test_result('passed', message)
            else:
                key.mark_error()
                key.set_test_result('failed', message)
            
            self._test_results.append((index, success, message))
            print(f"[DEBUG] Key {index} test completed - {'PASSED' if success else 'FAILED'}: {message}")
            
            # Update UI
            self._refresh_key_list()
        
        # Track completion
        self._completed_tests += 1
        print(f"[DEBUG] Completed {self._completed_tests}/{self._total_tests} tests")
        
        # If all tests done, finalize
        if self._completed_tests >= self._total_tests:
            self._finalize_tests()
    
    def _finalize_tests(self):
        """Finalize after all tests complete"""
        # Clear testing flags
        for i, key in enumerate(self.key_pool.keys):
            if hasattr(key, '_testing'):
                delattr(key, '_testing')
        
        # Calculate summary
        success_count = sum(1 for _, success, _ in self._test_results if success)
        total_count = len(self._test_results)
        
        # Final UI update
        self._refresh_key_list()
        self.stats_label.setText(f"Test complete: {success_count}/{total_count} passed")
        
        # Auto-save to persist test results
        self._save_keys_to_config()
        print(f"[DEBUG] All tests completed and saved: {success_count}/{total_count} passed")
    
    def _run_inline_tests(self, indices: List[int]):
        """Run API tests with persistent inline results"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        
        print(f"[DEBUG] Starting tests for {len(indices)} keys")
        
        # Keys are already marked as testing in the main thread before this function is called
        # Just verify they're still marked
        for index in indices:
            if index < len(self.key_pool.keys):
                key = self.key_pool.keys[index]
                print(f"[DEBUG] Key {index} testing state: _testing={hasattr(key, '_testing')}, last_test_result={key.last_test_result}")
        
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
                time.sleep(0.5)  # Brief pause to improve retry responsiveness
                logger.debug("💤 Pausing briefly to improve retry responsiveness after test completion")
                return (index, True, "Test passed")
                
            except Exception as e:
                print(f"[DEBUG] Key {index} test failed: {e}")
                key.mark_error()
                key.set_test_result('error', str(e)[:30])
                return (index, False, f"Error: {str(e)[:50]}...")
        
        # Run tests sequentially to avoid threading issues
        results = []
        for index in indices:
            print(f"[DEBUG] Starting test for key {index}")
            result = test_single_key(index)
            if result:
                results.append(result)
                print(f"[DEBUG] Got result: {result}")
                # Update UI immediately
                self._refresh_key_list()
                QApplication.processEvents()
        
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
        
        # Final update - we're already in a thread, just update directly
        print(f"[DEBUG] Refreshing UI with results")
        self._refresh_key_list()
        self.stats_label.setText(f"Test complete: {success_count}/{total_count} passed")
        # Auto-save to persist test results
        self._save_keys_to_config()
        QApplication.processEvents()
        print(f"[DEBUG] UI refresh and save completed")
        


    def _update_tree_item(self, index: int):
        """Update a single tree item based on current key state"""
        def update():
            # Find the tree item for this index
            if index >= self.tree.topLevelItemCount() or index >= len(self.key_pool.keys):
                return
            
            item = self.tree.topLevelItem(index)
            if not item:
                return
            
            key = self.key_pool.keys[index]
            
            # Determine status
            if key.last_test_result is None:
                status = "⏳ Testing..."
            elif not key.enabled:
                status = "Disabled"
            elif key.last_test_result == 'passed':
                if key.is_cooling_down:
                    remaining = int(key.cooldown - (time.time() - key.last_error_time))
                    status = f"✅ Passed (cooling {remaining}s)"
                else:
                    status = "✅ Passed"
            elif key.last_test_result == 'failed':
                status = "❌ Failed"
            elif key.last_test_result == 'rate_limited':
                remaining = int(key.cooldown - (time.time() - key.last_error_time))
                status = f"⚠️ Rate Limited ({remaining}s)"
            elif key.last_test_result == 'error':
                status = "❌ Error"
                if key.last_test_message:
                    status += f": {key.last_test_message[:20]}..."
            elif key.is_cooling_down:
                remaining = int(key.cooldown - (time.time() - key.last_error_time))
                status = f"Cooling ({remaining}s)"
            else:
                status = "Active"
            
            # Update status column (column 3)
            item.setText(3, status)
            
            # Update success/error counts
            item.setText(4, str(key.success_count))
            item.setText(5, str(key.error_count))
            
            # Update times used
            times_used = getattr(key, 'times_used', key.success_count + key.error_count)
            item.setText(6, str(times_used))
        
        # Run in main thread
        QTimer.singleShot(0, update)

    # Note: Another duplicate _refresh_key_list removed - using PySide6 version defined earlier
    
    def _create_progress_dialog(self):
        """Create simple progress dialog at mouse cursor position"""
        from PySide6.QtGui import QCursor
        
        self.progress_dialog = QDialog(self)
        self.progress_dialog.setWindowTitle("Testing API Keys")
        self.progress_dialog.resize(500, 400)
        
        # Get mouse position and move dialog there
        cursor_pos = QCursor.pos()
        self.progress_dialog.move(cursor_pos.x() - 50, cursor_pos.y() - 30)
        
        # Layout
        layout = QVBoxLayout(self.progress_dialog)
        
        # Add label
        label = QLabel("Testing in progress...")
        label_font = QFont()
        label_font.setPointSize(10)
        label_font.setBold(True)
        label.setFont(label_font)
        layout.addWidget(label)
        
        # Add text widget for results
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMinimumWidth(500)
        self.progress_text.setMinimumHeight(300)
        layout.addWidget(self.progress_text)
        
        # Add close button (initially disabled)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.progress_dialog.close)
        self.close_button.setEnabled(False)
        layout.addWidget(self.close_button)
        
        # Show dialog
        self.progress_dialog.show()

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
            QTimer.singleShot(0, lambda label=test_label: self.progress_text.append(f"Testing {label}... "))
            QTimer.singleShot(0, lambda: self.progress_text.moveCursor(self.progress_text.textCursor().End))
            
            try:
                # Count this usage for times used in testing as well
                try:
                    key.times_used += 1
                except Exception:
                    pass
                
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
                        QTimer.singleShot(0, lambda label=test_label: self._update_test_result(label, True))
                        key.mark_success()
                        key.set_test_result('passed', 'Test successful')
                        return (index, True, "Test passed")
                    else:
                        QTimer.singleShot(0, lambda label=test_label: self._update_test_result(label, False))
                        key.mark_error()
                        key.set_test_result('failed', 'Unexpected response')
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
                            QTimer.singleShot(0, lambda label=test_label: self._update_test_result(label, True))
                            key.mark_success()
                            key.set_test_result('passed', 'Test successful')
                            return (index, True, "Test passed")
                        else:
                            QTimer.singleShot(0, lambda label=test_label: self._update_test_result(label, False))
                            key.mark_error()
                            key.set_test_result('failed', 'Unexpected response')
                            return (index, False, "Unexpected response")
                    else:
                        QTimer.singleShot(0, lambda label=test_label: self._update_test_result(label, False))
                        key.mark_error()
                        key.set_test_result('failed', 'No response')
                        return (index, False, "No response")
                        
            except Exception as e:
                error_msg = str(e)
                error_code = None
                
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    error_code = 429
                    key.set_test_result('rate_limited', error_msg[:30])
                else:
                    key.set_test_result('error', error_msg[:30])
                    
                QTimer.singleShot(0, lambda label=test_label: self._update_test_result(label, False, error=True))
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
        QTimer.singleShot(0, self._show_completion)
        
        # Process final results
        QTimer.singleShot(0, self._process_test_results)

    def _update_test_result(self, test_label, success, error=False):
        """Update the progress text with test result"""
        # Find the line with this test label
        content = self.progress_text.toPlainText()
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
                
                # Update the line
                lines[i] = line + result_text
                
                # Set updated text
                self.progress_text.setPlainText('\n'.join(lines))
                
                # Scroll to end
                cursor = self.progress_text.textCursor()
                cursor.movePosition(cursor.End)
                self.progress_text.setTextCursor(cursor)
                break

    def _show_completion(self):
        """Show completion in the same dialog"""
        self.progress_text.append("\n--- Testing Complete ---\n")
        cursor = self.progress_text.textCursor()
        cursor.movePosition(cursor.End)
        self.progress_text.setTextCursor(cursor)
        self.close_button.setEnabled(True)
        
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
                # Get indices from results
                indices = [i for i, _, _ in results]
                # Clear testing flags
                for index in indices:
                    if index < len(self.key_pool.keys):
                        key = self.key_pool.keys[index]
                        if hasattr(key, '_testing'):
                            delattr(key, '_testing')
                
                self._refresh_key_list()
                self.stats_label.setText(f"Test complete: {success_count}/{total_count} passed")

            # Use QTimer to update in main thread
            QTimer.singleShot(0, lambda: final_update())
            
            # Add summary to the same dialog
            self.progress_text.append(f"\nSummary: {success_count}/{total_count} passed\n")
            self.progress_text.append("-" * 50 + "\n\n")
            
            for i, success, msg in results:
                key = self.key_pool.keys[i]
                # Show key identifier in results too
                key_preview = f"{key.api_key[:8]}...{key.api_key[-4:]}" if len(key.api_key) > 12 else key.api_key
                status = "✅" if success else "❌"
                self.progress_text.append(f"{status} {key.model} [{key_preview}]: {msg}\n")
            
            # Scroll to end
            cursor = self.progress_text.textCursor()
            cursor.movePosition(cursor.End)
            self.progress_text.setTextCursor(cursor)
            
            # Enable close button now that testing is complete
            self.close_button.setEnabled(True)
            
            # Update the dialog title
            self.progress_dialog.setWindowTitle(f"API Test Results - {success_count}/{total_count} passed")
            
            # Refresh list
            self._refresh_key_list()
    
    def _enable_selected(self):
        """Enable selected keys"""
        selected = self.tree.selectedItems()
        for item in selected:
            index = self.tree.indexOfTopLevelItem(item)
            if index < len(self.key_pool.keys):
                self.key_pool.keys[index].enabled = True
        
        self._refresh_key_list()
        self._show_status(f"Enabled {len(selected)} key(s)")
    
    def _disable_selected(self):
        """Disable selected keys"""
        selected = self.tree.selectedItems()
        for item in selected:
            index = self.tree.indexOfTopLevelItem(item)
            if index < len(self.key_pool.keys):
                self.key_pool.keys[index].enabled = False
        
        self._refresh_key_list()
        self._show_status(f"Disabled {len(selected)} key(s)")
    
    def _remove_selected(self):
        """Remove selected keys"""
        selected = self.tree.selectedItems()
        if not selected:
            return
        
        reply = QMessageBox.question(self, "Confirm", f"Remove {len(selected)} selected key(s)?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Get indices in reverse order to avoid index shifting
            indices = sorted([self.tree.indexOfTopLevelItem(item) for item in selected], reverse=True)
            
            for index in indices:
                self.key_pool.remove_key(index)
            
            self._refresh_key_list()
            self._show_status(f"Removed {len(selected)} key(s)")
    
    def _edit_cooldown(self):
        """Edit cooldown for selected key"""
        selected = self.tree.selectedItems()
        if not selected or len(selected) != 1:
            QMessageBox.warning(self, "Warning", "Please select exactly one key")
            return
        
        index = self.tree.indexOfTopLevelItem(selected[0])
        if index >= len(self.key_pool.keys):
            return
        
        key = self.key_pool.keys[index]
        
        # Use QInputDialog for simplicity
        from PySide6.QtWidgets import QInputDialog
        value, ok = QInputDialog.getInt(
            self, "Edit Cooldown", f"Cooldown for {key.model} (seconds):",
            key.cooldown, 10, 3600, 10
        )
        
        if ok:
            key.cooldown = value
            self._refresh_key_list()
            self._show_status(f"Updated cooldown to {value}s")
    
    def _import_keys(self):
        """Import keys from JSON file"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import API Keys",
            "",
            "JSON files (*.json);;All files (*.*)"
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
                    QMessageBox.information(self, "Success", f"Imported {imported_count} API keys")
                else:
                    QMessageBox.critical(self, "Error", "Invalid file format")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to import: {str(e)}")
    
    def _export_keys(self):
        """Export keys to JSON file"""
        if not self.key_pool.keys:
            QMessageBox.warning(self, "Warning", "No keys to export")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export API Keys",
            "",
            "JSON files (*.json);;All files (*.*)"
        )
        
        if filename:
            try:
                # Convert keys to list of dicts
                key_list = [key.to_dict() for key in self.key_pool.get_all_keys()]
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(key_list, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "Success", f"Exported {len(key_list)} API keys")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")
    
    def _show_status(self, message: str):
        """Show status message"""
        if hasattr(self, 'stats_label'):
            self.stats_label.setText(message)
    
    def _save_and_close(self):
        """Save configuration and close"""
        self._save_keys_to_config()
        QMessageBox.information(self, "Success", "API key configuration saved")
        self.accept()
    
    def _on_close(self):
        """Handle dialog close"""
        self.reject()
