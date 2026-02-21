# multi_api_key_manager.py
"""
Multi API Key Manager for Glossarion
Handles multiple API keys with round-robin load balancing and rate limit management
"""

# GUI imports - optional for Discord bot
try:
    from PySide6.QtCore import QMetaObject, Q_ARG
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit, 
        QTextEdit, QScrollArea, QFileDialog, QMessageBox, QComboBox, QCheckBox, 
        QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QSpinBox,
        QTreeWidget, QTreeWidgetItem, QAbstractItemView, QHeaderView, QMenu, QFrame,
        QCompleter
    )
    from PySide6.QtCore import Qt, QTimer, Signal, QObject, QPropertyAnimation, QEasingCurve, Slot
    from PySide6.QtGui import QIcon, QFont, QPixmap, QShortcut, QKeySequence, QTransform
    from spinning import create_icon_label, animate_icon
    HAS_GUI = True
except ImportError:
    HAS_GUI = False
    # Fallback dummy classes for non-GUI usage
    QObject = object
    QDialog = object
    QWidget = object
    QComboBox = object
    QLabel = object
    QPushButton = object
    QLineEdit = object
    QCheckBox = object
    QSpinBox = object
    QTreeWidget = object
    QTreeWidgetItem = object
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
                 azure_api_version: str = None, use_individual_endpoint: bool = False, individual_output_token_limit: Optional[int] = None):
        self.api_key = api_key
        self.model = model
        self.cooldown = cooldown
        self.enabled = enabled
        self.google_credentials = google_credentials  # Path to Google service account JSON
        self.azure_endpoint = azure_endpoint  # Azure endpoint URL (only used if use_individual_endpoint is True)
        self.google_region = google_region  # Google Cloud region (e.g., us-east5, us-central1)
        self.azure_api_version = azure_api_version or '2025-01-01-preview'  # Azure API version
        self.use_individual_endpoint = use_individual_endpoint  # Toggle to enable/disable individual endpoint
        # Individual output token limit for this key (overrides global limit when set)
        try:
            if individual_output_token_limit is not None and individual_output_token_limit != "" and int(individual_output_token_limit) > 0:
                self.individual_output_token_limit = int(individual_output_token_limit)
            else:
                self.individual_output_token_limit = None
        except (ValueError, TypeError):
            self.individual_output_token_limit = None
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
            'individual_output_token_limit': self.individual_output_token_limit,
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
        
        # Stop flag callback - can be set by UnifiedClient to check for stop requests
        self._stop_check_callback = None
    
    def set_stop_check_callback(self, callback):
        """Set a callback function to check if stop has been requested.
        The callback should return True if stop is requested, False otherwise.
        """
        self._stop_check_callback = callback
    
    def _is_stop_requested(self) -> bool:
        """Check if stop has been requested via callback or global flag."""
        # Check callback first if set
        if self._stop_check_callback is not None:
            try:
                if self._stop_check_callback():
                    return True
            except Exception:
                pass
        
        # Also check the module-level stop flag from unified_api_client if available
        try:
            from unified_api_client import is_stop_requested
            if is_stop_requested():
                return True
        except ImportError:
            pass
        
        return False
    
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
                    use_individual_endpoint=key_data.get('use_individual_endpoint', False),
                    individual_output_token_limit=key_data.get('individual_output_token_limit')
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
        # Check for stop request at start
        if self._is_stop_requested():
            logger.info("Stop requested during key selection, returning None")
            return None
        
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
                # Check for stop request in loop
                if self._is_stop_requested():
                    logger.info("Stop requested during key rotation loop")
                    return None
                
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
                            if current_time - ts > 310  # 5 minutes
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
                    
                    logger.debug(f"Marked key {key_id} with an error code")
                    time.sleep(0.5)  # Brief pause to improve retry responsiveness
                    logger.debug("💤 Pausing briefly to improve retry responsiveness after marking key error")

    def mark_key_success(self, key_index: int):
        """Mark a key as successful (thread-safe with key-specific lock)"""
        if 0 <= key_index < len(self.keys):
            # Use key-specific lock for this operation
            with self.key_locks.get(key_index, threading.Lock()):
                self.keys[key_index].mark_success()
                
                key = self.keys[key_index]
                logger.debug(f"Marked key {key_index} ({key.model}) as successful")
    
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


class RefusalPatternsDialog(QDialog):
    """Dialog for managing AI refusal patterns"""
    
    def __init__(self, parent, translator_gui):
        super().__init__(parent)
        self.translator_gui = translator_gui
        self.setWindowTitle("Manage Refusal Patterns")
        
        # Set dialog size based on screen size (35% width, 40% height)
        from PySide6.QtWidgets import QApplication
        screen = QApplication.primaryScreen().geometry()
        dialog_width = int(screen.width() * 0.25)
        dialog_height = int(screen.height() * 0.4)
        self.resize(dialog_width, dialog_height)
        
        # Set window icon
        self._set_icon()
        
        # Load patterns from config
        self.patterns = self._load_patterns()
        self.disable_refusal_checks = self._load_disable_refusal_checks()
        self.refusal_length_limit = self._load_refusal_length_limit()
        
        self._create_dialog()
    
    def _set_icon(self):
        """Set Halgakos.ico as window icon if available."""
        try:
            base_dir = getattr(self.translator_gui, 'base_dir', os.getcwd())
            ico_path = os.path.join(base_dir, 'Halgakos.ico')
            if os.path.isfile(ico_path):
                # Load icon with high quality preservation
                icon = QIcon(ico_path)
                # Enable high DPI pixmaps for better quality
                if hasattr(icon, 'setIsMask'):
                    icon.setIsMask(False)
                self.setWindowIcon(icon)
        except Exception:
            pass
    
    def _load_patterns(self):
        """Load refusal patterns from config"""
        if hasattr(self.translator_gui, 'config'):
            return self.translator_gui.config.get('refusal_patterns', self._get_default_patterns())
        return self._get_default_patterns()
    
    def _get_default_patterns(self):
        """Get default refusal patterns"""
        return [
            "i cannot assist", "i can't assist", "i'm not able to assist",
            "i cannot help", "i can't help", "i'm unable to help",
            "i'm afraid i cannot help with that", "designed to ensure appropriate use",
            "as an ai", "as a language model", "as an ai language model",
            "i don't feel comfortable", "i apologize, but i cannot",
            "i'm sorry, but i can't assist", "i'm sorry, but i cannot assist",
            "against my programming", "against my guidelines",
            "violates content policy", "i'm not programmed to",
            "cannot provide that kind", "unable to provide that",
            "i cannot assist with this request",
            "that's not within my capabilities to appropriately assist with",
            "is there something different i can help you with",
            "careful ethical considerations",
            "i could help you with a different question or task",
            "what other topics or questions can i help you explore",
            "i cannot and will not translate",
            "i cannot translate this content",
            "i can't translate this content",
        ]
    
    def _load_disable_refusal_checks(self):
        """Load refusal check disable toggle from config"""
        try:
            if hasattr(self.translator_gui, 'config'):
                return bool(self.translator_gui.config.get('disable_refusal_checks', False))
        except Exception:
            pass
        return False
    
    def _load_refusal_length_limit(self):
        """Load refusal length limit from config"""
        try:
            if hasattr(self.translator_gui, 'config'):
                return int(self.translator_gui.config.get('refusal_pattern_length_limit', 1000))
        except Exception:
            pass
        return 1000
    
    def _reload_from_config(self):
        """Reload patterns/toggles from config and refresh UI controls."""
        try:
            self.patterns = self._load_patterns()
            self.disable_refusal_checks = self._load_disable_refusal_checks()
            self.refusal_length_limit = self._load_refusal_length_limit()
            self._refresh_tree()
            if hasattr(self, 'disable_refusal_checks_cb'):
                self.disable_refusal_checks_cb.setChecked(bool(self.disable_refusal_checks))
            if hasattr(self, 'refusal_length_limit_entry'):
                self.refusal_length_limit_entry.setText(str(self.refusal_length_limit))
        except Exception:
            pass
    
    def _load_halgakos_pixmap(self, logical_size: int = 36):
        """Load the Halgakos icon as a HiDPI-aware pixmap scaled to logical_size."""
        try:
            base_dir = getattr(self.translator_gui, 'base_dir', os.getcwd())
            ico_path = os.path.join(base_dir, 'Halgakos.ico')
            if not os.path.isfile(ico_path):
                return None
            pixmap = QPixmap(ico_path)
            if pixmap.isNull():
                return None
            screen = QApplication.primaryScreen()
            dpr = screen.devicePixelRatio() if screen else 1.0
            target = int(logical_size * dpr)
            scaled = pixmap.scaled(
                target,
                target,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            scaled.setDevicePixelRatio(dpr)
            return scaled
        except Exception:
            return None
    
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
    
    def _create_dialog(self):
        """Create the dialog UI"""
        # Apply stylesheet matching other dialogs
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
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
            QLineEdit {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                padding: 4px;
            }
            QTreeWidget {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                selection-background-color: #4a7ba7;
            }
            QTreeWidget::item:hover {
                background-color: #3a3a3a;
            }
            QHeaderView::section {
                background-color: #252525;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                padding: 4px;
            }
        """)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title with icons on both sides
        title_container = QWidget()
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(15)
        title_layout.setAlignment(Qt.AlignCenter)
        
        # Left Icon
        icon_label_left = QLabel()
        left_pixmap = self._load_halgakos_pixmap(36)
        if left_pixmap:
            icon_label_left.setPixmap(left_pixmap)
            icon_label_left.setFixedSize(36, 36)
        icon_label_left.setAlignment(Qt.AlignCenter)
        
        title_layout.addWidget(icon_label_left, 0, Qt.AlignVCenter)
        
        # Title text
        title_label = QLabel("Refusal Pattern Management")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title_label, 0, Qt.AlignVCenter)
        
        # Right Icon (mirror of left)
        icon_label_right = QLabel()
        right_pixmap = self._load_halgakos_pixmap(36)
        if right_pixmap:
            icon_label_right.setPixmap(right_pixmap)
            icon_label_right.setFixedSize(36, 36)
        icon_label_right.setAlignment(Qt.AlignCenter)
        
        title_layout.addWidget(icon_label_right, 0, Qt.AlignVCenter)
        
        main_layout.addWidget(title_container)
        
        # Description
        desc_label = QLabel(
            "Patterns used to detect AI refusals in responses. "
            "When detected, fallback keys will be attempted.\n"
            "Patterns are checked on responses under a configurable length (case-insensitive).")
        desc_label.setStyleSheet("color: gray; padding: 5px 0px 10px 0px;")
        desc_label.setWordWrap(True)
        main_layout.addWidget(desc_label)
        
        # Controls row: disable toggle + length limit
        controls_row = QWidget()
        controls_h = QHBoxLayout(controls_row)
        controls_h.setContentsMargins(0, 0, 0, 10)
        
        self.disable_refusal_checks_cb = self._create_styled_checkbox("Disable refusal pattern checks")
        try:
            self.disable_refusal_checks_cb.setChecked(bool(self.disable_refusal_checks))
        except Exception:
            pass
        controls_h.addWidget(self.disable_refusal_checks_cb)
        
        controls_h.addSpacing(12)
        controls_h.addWidget(QLabel("Length limit:"))
        self.refusal_length_limit_entry = QLineEdit()
        self.refusal_length_limit_entry.setFixedWidth(80)
        try:
            self.refusal_length_limit_entry.setText(str(self.refusal_length_limit))
        except Exception:
            self.refusal_length_limit_entry.setText("1000")
        controls_h.addWidget(self.refusal_length_limit_entry)
        controls_h.addWidget(QLabel("chars"))
        controls_h.addStretch()
        main_layout.addWidget(controls_row)
        
        # Tree widget for patterns
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Pattern"])
        self.tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.tree.header().setStretchLastSection(True)
        main_layout.addWidget(self.tree)
        
        # Load patterns into tree
        self._refresh_tree()
        
        # Add/Edit/Delete buttons
        button_frame = QWidget()
        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(0, 10, 0, 0)
        
        add_btn = QPushButton("➕ Add Pattern")
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a5c1b;
                color: #ffffff;
                padding: 6px 12px;
                border: 1px solid #145016;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #237024;
                border-color: #1a5c1b;
            }
            QPushButton:pressed {
                background-color: #0f3d10;
            }
        """)
        add_btn.clicked.connect(self._add_pattern)
        button_layout.addWidget(add_btn)
        
        edit_btn = QPushButton("✏️ Edit Selected")
        edit_btn.setStyleSheet("""
            QPushButton {
                background-color: #2c5278;
                color: #ffffff;
                padding: 6px 12px;
                border: 1px solid #1e3a54;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a6a94;
                border-color: #2c5278;
            }
            QPushButton:pressed {
                background-color: #1e3a54;
            }
        """)
        edit_btn.clicked.connect(self._edit_pattern)
        button_layout.addWidget(edit_btn)
        
        delete_btn = QPushButton("🗑️ Delete Selected")
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b2e2e;
                color: #ffffff;
                padding: 6px 12px;
                border: 1px solid #6b1f1f;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #a73a3a;
                border-color: #8b2e2e;
            }
            QPushButton:pressed {
                background-color: #5c1e1e;
            }
        """)
        delete_btn.clicked.connect(self._delete_patterns)
        button_layout.addWidget(delete_btn)
        
        button_layout.addStretch()
        
        reset_btn = QPushButton("🔄 Reset to Defaults")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #6b5a1e;
                color: #ffffff;
                padding: 6px 12px;
                border: 1px solid #4d4116;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #8b7528;
                border-color: #6b5a1e;
            }
            QPushButton:pressed {
                background-color: #4d4116;
            }
        """)
        reset_btn.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(reset_btn)
        
        main_layout.addWidget(button_frame)
        
        # Save/Cancel buttons
        save_cancel_frame = QWidget()
        save_cancel_layout = QHBoxLayout(save_cancel_frame)
        save_cancel_layout.setContentsMargins(0, 10, 0, 0)
        save_cancel_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumHeight(32)
        cancel_btn.clicked.connect(self.reject)
        save_cancel_layout.addWidget(cancel_btn)
        
        save_btn = QPushButton("Save & Close")
        save_btn.setMinimumHeight(32)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a7ba7;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a9fd4;
            }
        """)
        save_btn.clicked.connect(self._save_and_close)
        save_cancel_layout.addWidget(save_btn)
        
        main_layout.addWidget(save_cancel_frame)
    
    def _refresh_tree(self):
        """Refresh the tree widget with current patterns"""
        self.tree.clear()
        for pattern in self.patterns:
            item = QTreeWidgetItem([pattern])
            self.tree.addTopLevelItem(item)
    
    def _add_pattern(self):
        """Add a new pattern"""
        # Create input dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Refusal Pattern")
        dialog.setMinimumWidth(500)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        
        label = QLabel("Enter refusal pattern (case-insensitive):")
        layout.addWidget(label)
        
        entry = QLineEdit()
        layout.addWidget(entry)
        
        hint_label = QLabel("Examples: \"i cannot assist\", \"as an ai\", \"violates content policy\"")
        hint_label.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(hint_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        ok_btn = QPushButton("Add")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            pattern = entry.text().strip().lower()
            if pattern and pattern not in self.patterns:
                self.patterns.append(pattern)
                self._refresh_tree()
    
    def _edit_pattern(self):
        """Edit the selected pattern"""
        selected = self.tree.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select a pattern to edit")
            return
        
        if len(selected) > 1:
            QMessageBox.warning(self, "Warning", "Please select only one pattern to edit")
            return
        
        item = selected[0]
        old_pattern = item.text(0)
        
        # Create input dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Refusal Pattern")
        dialog.setMinimumWidth(500)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        
        label = QLabel("Edit refusal pattern:")
        layout.addWidget(label)
        
        entry = QLineEdit()
        entry.setText(old_pattern)
        layout.addWidget(entry)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        ok_btn = QPushButton("Save")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            new_pattern = entry.text().strip().lower()
            if new_pattern and new_pattern != old_pattern:
                if new_pattern in self.patterns:
                    QMessageBox.warning(self, "Warning", "Pattern already exists")
                else:
                    index = self.patterns.index(old_pattern)
                    self.patterns[index] = new_pattern
                    self._refresh_tree()
    
    def _delete_patterns(self):
        """Delete selected patterns"""
        selected = self.tree.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select patterns to delete")
            return
        
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Delete {len(selected)} pattern(s)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            for item in selected:
                pattern = item.text(0)
                if pattern in self.patterns:
                    self.patterns.remove(pattern)
            self._refresh_tree()
    
    def _reset_to_defaults(self):
        """Reset patterns to defaults"""
        reply = QMessageBox.question(
            self,
            "Confirm Reset",
            "Reset all patterns to defaults? This will replace your current patterns.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.patterns = self._get_default_patterns()
            self._refresh_tree()
            # Reset toggle + limit to defaults
            try:
                if hasattr(self, 'disable_refusal_checks_cb'):
                    self.disable_refusal_checks_cb.setChecked(False)
            except Exception:
                pass
            try:
                if hasattr(self, 'refusal_length_limit_entry'):
                    self.refusal_length_limit_entry.setText("1000")
            except Exception:
                pass
    
    def _save_and_close(self):
        """Save patterns and close dialog"""
        if hasattr(self.translator_gui, 'config'):
            self.translator_gui.config['refusal_patterns'] = self.patterns
            # Save refusal controls
            try:
                self.translator_gui.config['disable_refusal_checks'] = bool(self.disable_refusal_checks_cb.isChecked())
            except Exception:
                pass
            try:
                raw_limit = self.refusal_length_limit_entry.text().strip()
                limit_val = int(raw_limit) if raw_limit.isdigit() else 1000
                if limit_val <= 0:
                    limit_val = 1000
                self.translator_gui.config['refusal_pattern_length_limit'] = limit_val
            except Exception:
                self.translator_gui.config['refusal_pattern_length_limit'] = 1000
            self.translator_gui.save_config(show_message=False)
            QMessageBox.information(self, "Success", f"Saved {len(self.patterns)} refusal patterns")
        self.accept()


class MultiAPIKeyDialog(QDialog):
    """Dialog for managing multiple API keys"""
    
    @staticmethod
    def show_dialog(parent, translator_gui):
        """Static method to create and show the dialog non-modally.
        
        This ensures proper PySide6 application context.
        """
        # Ensure QApplication exists
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Check if dialog already exists on the translator_gui
        if not hasattr(translator_gui, '_multi_api_key_dialog') or translator_gui._multi_api_key_dialog is None:
            # Create and show dialog non-modally
            dialog = MultiAPIKeyDialog(parent, translator_gui)
            dialog.setWindowModality(Qt.NonModal)
            # Make dialog stay on top of other windows
            dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowStaysOnTopHint)
            translator_gui._multi_api_key_dialog = dialog
        else:
            dialog = translator_gui._multi_api_key_dialog
        
        # Show and raise the dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        
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
            # Save main-key-fallback toggle (persist in config)
            if hasattr(self, 'use_main_key_fallback_checkbox'):
                self.translator_gui.config['use_main_key_fallback'] = self.use_main_key_fallback_checkbox.isChecked()
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
        # Use screen ratios for sizing
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.52)  # 52% of screen width
        height = int(screen.height() * 0.68)  # 68% of screen height
        self.resize(width, height)
        
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
            QLineEdit, QTextEdit {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                padding: 4px;
            }
            QLineEdit:focus, QTextEdit:focus {
                border-color: #5a5a5a;
            }
            QLineEdit:disabled, QTextEdit:disabled {
                background-color: #1a1a1a;
                color: #666666;
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
        
        # Enable/Disable toggle with spinning icon
        self.enabled_var = self.translator_gui.config.get('use_multi_api_keys', False)
        self.enabled_checkbox = self._create_styled_checkbox("Enable Multi-Key Mode")
        self.enabled_checkbox.setChecked(self.enabled_var)
        self.enabled_checkbox.toggled.connect(self._toggle_multi_key_mode)
        
        # Add spinning icon next to multi-key mode checkbox (HiDPI-aware like Extract Glossary)
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Halgakos.ico")
        self.multikey_icon = QLabel()
        self.multikey_icon.setStyleSheet("background-color: transparent;")
        if os.path.exists(icon_path):
            from PySide6.QtGui import QIcon, QPixmap
            from PySide6.QtCore import QSize
            icon = QIcon(icon_path)
            try:
                dpr = self.devicePixelRatioF()
            except Exception:
                dpr = 1.0
            logical_px = 16
            dev_px = int(logical_px * max(1.0, dpr))
            pm = icon.pixmap(QSize(dev_px, dev_px))
            if pm.isNull():
                raw = QPixmap(icon_path)
                img = raw.toImage().scaled(dev_px, dev_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                pm = QPixmap.fromImage(img)
            try:
                pm.setDevicePixelRatio(dpr)
            except Exception:
                pass
            self.multikey_icon.setPixmap(pm)
        self.multikey_icon.setFixedSize(36, 36)
        self.multikey_icon.setAlignment(Qt.AlignCenter)
        self.enabled_checkbox.toggled.connect(lambda: animate_icon(self.multikey_icon))
        
        title_layout.addStretch()
        title_layout.addWidget(self.multikey_icon)
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
        
        # Enable fallback checkbox with spinning icon
        fallback_checkbox_container = QWidget()
        fallback_checkbox_layout = QHBoxLayout(fallback_checkbox_container)
        fallback_checkbox_layout.setContentsMargins(0, 0, 0, 0)
        fallback_checkbox_layout.setSpacing(8)
        
        self.use_fallback_var = self.translator_gui.config.get('use_fallback_keys', False)
        self.use_fallback_checkbox = self._create_styled_checkbox("Enable Fallback Keys")
        self.use_fallback_checkbox.setChecked(self.use_fallback_var)
        self.use_fallback_checkbox.toggled.connect(self._toggle_fallback_section)
        
        # Add spinning icon next to fallback keys checkbox (HiDPI-aware like Extract Glossary)
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Halgakos.ico")
        self.fallback_icon = QLabel()
        self.fallback_icon.setStyleSheet("background-color: transparent;")
        if os.path.exists(icon_path):
            from PySide6.QtGui import QIcon, QPixmap
            from PySide6.QtCore import QSize
            icon = QIcon(icon_path)
            try:
                dpr = self.devicePixelRatioF()
            except Exception:
                dpr = 1.0
            logical_px = 16
            dev_px = int(logical_px * max(1.0, dpr))
            pm = icon.pixmap(QSize(dev_px, dev_px))
            if pm.isNull():
                raw = QPixmap(icon_path)
                img = raw.toImage().scaled(dev_px, dev_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                pm = QPixmap.fromImage(img)
            try:
                pm.setDevicePixelRatio(dpr)
            except Exception:
                pass
            self.fallback_icon.setPixmap(pm)
        self.fallback_icon.setFixedSize(36, 36)
        self.fallback_icon.setAlignment(Qt.AlignCenter)
        self.use_fallback_checkbox.toggled.connect(lambda: animate_icon(self.fallback_icon))
        
        fallback_checkbox_layout.addWidget(self.fallback_icon)
        fallback_checkbox_layout.addWidget(self.use_fallback_checkbox)
        fallback_checkbox_layout.addStretch()
        
        fallback_frame_layout.addWidget(fallback_checkbox_container)

        # Toggle: use main GUI key as first fallback entry
        self.use_main_key_fallback_var = self.translator_gui.config.get('use_main_key_fallback', True)
        self.use_main_key_fallback_checkbox = self._create_styled_checkbox("Use Main GUI Key as Fallback #1")
        self.use_main_key_fallback_checkbox.setChecked(self.use_main_key_fallback_var)
        fallback_frame_layout.addWidget(self.use_main_key_fallback_checkbox)
        
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
        
        # Row 4: Individual Output Token Limit (fallback)
        fallback_output_label = QLabel("Output Token Limit:")
        fallback_output_label.setStyleSheet("color: gray; font-size: 9pt;")
        add_fallback_grid.addWidget(fallback_output_label, 4, 0, Qt.AlignLeft)
        self.fallback_output_token_spinbox = QSpinBox()
        self.fallback_output_token_spinbox.setRange(0, 2000000)
        self.fallback_output_token_spinbox.setValue(0)
        self.fallback_output_token_spinbox.setMaximumWidth(120)
        self._disable_spinbox_mousewheel(self.fallback_output_token_spinbox)
        add_fallback_grid.addWidget(self.fallback_output_token_spinbox, 4, 1, Qt.AlignLeft)
        fallback_output_hint = QLabel("0 = use global limit")
        fallback_output_hint.setStyleSheet("color: gray; font-size: 8pt;")
        add_fallback_grid.addWidget(fallback_output_hint, 4, 2, 1, 2, Qt.AlignLeft)
        
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
        # Add explicit column for per-key output token limit
        self.fallback_tree.setHeaderLabels(['API Key', 'Model', 'Output Limit', 'Status', 'Times Used'])
        self.fallback_tree.setColumnWidth(0, 220)
        self.fallback_tree.setColumnWidth(1, 220)
        self.fallback_tree.setColumnWidth(2, 110)  # Output Limit
        self.fallback_tree.setColumnWidth(3, 120)  # Status
        self.fallback_tree.setColumnWidth(4, 100)  # Times Used
        self.fallback_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.fallback_tree.customContextMenuRequested.connect(self._show_fallback_context_menu)
        self.fallback_tree.setMinimumHeight(150)
        
        # Enable drag and drop for fallback tree using InternalMove (like profile manager)
        # This is the simplest approach - let Qt handle everything
        self.fallback_tree.setDragDropMode(QAbstractItemView.InternalMove)
        self.fallback_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        # Connect to model's rowsMoved signal to sync data after Qt moves items
        self.fallback_tree.model().rowsMoved.connect(self._on_fallback_rows_moved)
        
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
        # Status label specific to fallback actions
        self.fallback_status_label = QLabel()
        self.fallback_status_label.setStyleSheet("color: gray;")
        fallback_action_layout.addWidget(self.fallback_status_label)
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
                # Status column is index 3 (after Output Limit)
                item.setText(3, "⏳ Testing...")
        
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
        
        # Per-key output token limit options for fallback keys
        selected_items = self.fallback_tree.selectedItems()
        selected_count = len(selected_items)
        if selected_count > 1:
            set_limit_action = menu.addAction(f"Set Output Token Limit ({selected_count} selected)")
        else:
            set_limit_action = menu.addAction("Set Output Token Limit")
        set_limit_action.triggered.connect(self._set_fallback_output_token_limit_for_selected)
        clear_limit_action = menu.addAction("Clear Output Token Limit")
        clear_limit_action.triggered.connect(self._clear_fallback_output_token_limit_for_selected)
        
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
        # Use screen ratios for sizing
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.21)  # 21% of screen width
        height = int(screen.height() * 0.13)  # 13% of screen height
        dialog.resize(width, height)
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
        
        # Save scroll position and selection
        v_scroll = self.fallback_tree.verticalScrollBar().value()
        h_scroll = self.fallback_tree.horizontalScrollBar().value()
        
        # Save selection (by index)
        selected_indices = []
        for item in self.fallback_tree.selectedItems():
            selected_indices.append(self.fallback_tree.indexOfTopLevelItem(item))
        
        # Clear tree
        self.fallback_tree.clear()
        
        # Add keys to tree
        for key_data in fallback_keys:
            api_key = key_data.get('api_key', '')
            model = key_data.get('model', '')
            times_used = int(key_data.get('times_used', 0))
            
            # Mask API key
            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else api_key
            
            # Determine per-key output token limit display value
            try:
                raw_limit = key_data.get('individual_output_token_limit')
                per_key_limit = int(raw_limit) if raw_limit not in (None, "") else None
            except Exception:
                per_key_limit = None
            if per_key_limit and per_key_limit > 0:
                output_limit_str = str(per_key_limit)
            else:
                output_limit_str = "global"
            
            # Insert into tree
            item = QTreeWidgetItem([masked_key, model, output_limit_str, "Not tested", str(times_used)])
            # Untested styling
            for col in range(item.columnCount()):
                item.setForeground(col, Qt.gray)
            
            # Tooltip for per-key output token limit
            if per_key_limit and per_key_limit > 0:
                tooltip = f"Individual Output Token Limit: {per_key_limit}"
            else:
                tooltip = "Using global output token limit"
            for col in range(item.columnCount()):
                item.setToolTip(col, tooltip)
            
            # Disable drop on this item to prevent nesting (keep it a flat list)
            item.setFlags(item.flags() & ~Qt.ItemIsDropEnabled)
            self.fallback_tree.addTopLevelItem(item)
            
        # Restore selection
        for index in selected_indices:
            if index < self.fallback_tree.topLevelItemCount():
                item = self.fallback_tree.topLevelItem(index)
                item.setSelected(True)
        
        # Restore scroll position
        self.fallback_tree.verticalScrollBar().setValue(v_scroll)
        self.fallback_tree.horizontalScrollBar().setValue(h_scroll)

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
        
        # Determine per-key output token limit (0 = use global limit)
        individual_output_token_limit = None
        if hasattr(self, 'fallback_output_token_spinbox'):
            try:
                val = int(self.fallback_output_token_spinbox.value())
                if val > 0:
                    individual_output_token_limit = val
            except Exception:
                individual_output_token_limit = None
        
        # Add new key with additional fields
        fallback_keys.append({
            'api_key': api_key,
            'model': model,
            'google_credentials': google_credentials,
            'azure_endpoint': azure_endpoint,
            'google_region': google_region,
            'azure_api_version': azure_api_version,
            'use_individual_endpoint': use_individual_endpoint,
            'individual_output_token_limit': individual_output_token_limit,
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
        if hasattr(self, 'fallback_output_token_spinbox'):
            self.fallback_output_token_spinbox.setValue(0)
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
                # Status column is index 3 (after Output Limit)
                item.setText(3, "⏳ Testing...")
        
        key_data = fallback_keys[index]
        
        # Ensure UnifiedClient uses the same shared pool instance
        try:
            from unified_api_client import UnifiedClient
            UnifiedClient._api_key_pool = self.key_pool
        except Exception:
            pass
        
        # Run test on main thread using QTimer (non-blocking)
        QTimer.singleShot(100, lambda: self._test_single_fallback_key(key_data, index))

    # Decorate _update_fallback_test_result as a slot for invokeMethod
    @Slot(int, bool) if HAS_GUI else lambda x: x
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
                # Update status (column 3)
                item.setText(3, "✅ Passed" if success else "❌ Failed")
                # Update times used cell (column 4)
                try:
                    current_times = int(item.text(4))
                    item.setText(4, str(current_times + 1))
                except Exception:
                    item.setText(4, "1")

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
                
                # Force 1 retries for testing to speed up failure detection
                try:
                    tls = client._get_thread_local_client()
                    tls.max_retries_override = 1
                    print(f"[DEBUG] Set max_retries_override=1 for fallback key test")
                except Exception:
                    pass
                
                
                # Set Google credentials and other key-specific settings
                google_credentials = key_data.get('google_credentials')
                if google_credentials:
                    client.current_key_google_creds = google_credentials
                    client.google_creds_path = google_credentials
                    print(f"[DEBUG] Set Google credentials for fallback test: {os.path.basename(google_credentials)}")
                
                google_region = key_data.get('google_region')
                if google_region:
                    client.current_key_google_region = google_region
                    print(f"[DEBUG] Set Google region for fallback test: {google_region}")
                
                # Set Azure endpoint settings if configured
                use_individual_endpoint = key_data.get('use_individual_endpoint', False)
                if use_individual_endpoint:
                    azure_endpoint = key_data.get('azure_endpoint')
                    if azure_endpoint:
                        client.current_key_azure_endpoint = azure_endpoint
                        client.current_key_use_individual_endpoint = True
                        print(f"[DEBUG] Set Azure endpoint for fallback test: {azure_endpoint[:50]}...")
                    
                    azure_api_version = key_data.get('azure_api_version')
                    if azure_api_version:
                        client.current_key_azure_api_version = azure_api_version
                
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'API test successful' and nothing else."}
                ]
                
                response = client.send(
                    messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                if response and isinstance(response, tuple):
                    content, _ = response
                    if content and "test successful" in content.lower():
                        print(f"[DEBUG] Fallback key test completed for {model}: PASSED")
                        # Update directly - we're in executor thread, so use invokeMethod or signals
                        if HAS_GUI:
                            QMetaObject.invokeMethod(self, "_update_fallback_test_result", Qt.QueuedConnection, Q_ARG(int, index), Q_ARG(bool, True))
                        else:
                            self._update_fallback_test_result(index, True)
                        return
                
                # Failed
                print(f"[DEBUG] Fallback key test completed for {model}: FAILED")
                if HAS_GUI:
                    QMetaObject.invokeMethod(self, "_update_fallback_test_result", Qt.QueuedConnection, Q_ARG(int, index), Q_ARG(bool, False))
                else:
                    self._update_fallback_test_result(index, False)
            except Exception as e:
                print(f"[DEBUG] Fallback key test error for {model}: {e}")
                if HAS_GUI:
                    QMetaObject.invokeMethod(self, "_update_fallback_test_result", Qt.QueuedConnection, Q_ARG(int, index), Q_ARG(bool, False))
                else:
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
    
    def _set_fallback_output_token_limit_for_selected(self):
        """Set per-key output token limit for selected fallback keys"""
        from PySide6.QtWidgets import QInputDialog
        selected = self.fallback_tree.selectedItems()
        if not selected:
            return
        
        fallback_keys = self.translator_gui.config.get('fallback_keys', [])
        selected_indices = [self.fallback_tree.indexOfTopLevelItem(item) for item in selected]
        if not selected_indices:
            return
        
        # Determine default from first selected fallback key or global setting
        default_val = None
        first_idx = selected_indices[0]
        if 0 <= first_idx < len(fallback_keys):
            try:
                raw = fallback_keys[first_idx].get('individual_output_token_limit')
                if raw not in (None, ""):
                    iv = int(raw)
                    if iv > 0:
                        default_val = iv
            except Exception:
                default_val = None
        if default_val is None:
            try:
                default_val = int(getattr(self.translator_gui, 'max_output_tokens', 8192))
            except Exception:
                default_val = 8192
        
        value, ok = QInputDialog.getInt(
            self,
            "Set Fallback Output Token Limit",
            "Max output tokens for selected fallback key(s):",
            default_val,
            1,
            2000000,
            512,
        )
        if not ok or value <= 0:
            return
        
        for idx in selected_indices:
            if 0 <= idx < len(fallback_keys):
                fallback_keys[idx]['individual_output_token_limit'] = int(value)
        self.translator_gui.config['fallback_keys'] = fallback_keys
        self.translator_gui.save_config(show_message=False)
        self._load_fallback_keys()
        self._show_status(f"Set fallback output token limit to {value} for {len(selected_indices)} key(s)")
    
    def _clear_fallback_output_token_limit_for_selected(self):
        """Clear per-key output token limit for selected fallback keys"""
        selected = self.fallback_tree.selectedItems()
        if not selected:
            return
        
        fallback_keys = self.translator_gui.config.get('fallback_keys', [])
        selected_indices = [self.fallback_tree.indexOfTopLevelItem(item) for item in selected]
        for idx in selected_indices:
            if 0 <= idx < len(fallback_keys):
                if 'individual_output_token_limit' in fallback_keys[idx]:
                    del fallback_keys[idx]['individual_output_token_limit']
        self.translator_gui.config['fallback_keys'] = fallback_keys
        self.translator_gui.save_config(show_message=False)
        self._load_fallback_keys()
        self._show_status(f"Cleared fallback output token limit for {len(selected_indices)} key(s)")
    
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
        
        # Manage Refusal Patterns button
        refusal_btn = QPushButton("🚫 Manage Refusal Patterns")
        refusal_btn.setMinimumHeight(40)
        refusal_btn.setStyleSheet("""
            QPushButton {
                background-color: #5c2e5b;
                color: #ffffff;
                font-size: 11pt;
                padding: 8px 20px;
                border: 1px solid #42213f;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #733d71;
                border-color: #5c2e5b;
            }
            QPushButton:pressed {
                background-color: #3d1d3c;
            }
        """)
        refusal_btn.clicked.connect(self._manage_refusal_patterns)
        button_layout.addWidget(refusal_btn)
        
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
        top_btn = QPushButton("↑ ↑")
        top_btn.setFixedSize(55, 32)
        top_btn.setStyleSheet("QPushButton { font-size: 14pt; padding: 2px; }")
        top_btn.clicked.connect(lambda: self._move_key('top'))
        move_layout.addWidget(top_btn)
        
        # Move up button with expanded width
        up_btn = QPushButton("↑")
        up_btn.setFixedSize(55, 32)
        up_btn.setStyleSheet("QPushButton { font-size: 16pt; padding: 2px; }")
        up_btn.clicked.connect(lambda: self._move_key('up'))
        move_layout.addWidget(up_btn)
        
        # Move down button with expanded width
        down_btn = QPushButton("↓")
        down_btn.setFixedSize(55, 32)
        down_btn.setStyleSheet("QPushButton { font-size: 16pt; padding: 2px; }")
        down_btn.clicked.connect(lambda: self._move_key('down'))
        move_layout.addWidget(down_btn)
        
        # Move to bottom button with expanded width
        bottom_btn = QPushButton("↓ ↓")
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
        # Add explicit column for per-key output token limit
        self.tree.setHeaderLabels(['API Key', 'Model', 'Cooldown', 'Output Limit', 'Status', 'Success', 'Errors', 'Times Used'])
        # Adjusted column widths: balanced distribution
        self.tree.setColumnWidth(0, 125)  # API Key (decreased from 140)
        self.tree.setColumnWidth(1, 230)  # Model (decreased from 320)
        self.tree.setColumnWidth(2, 80)   # Cooldown
        self.tree.setColumnWidth(3, 100)  # Output Limit
        self.tree.setColumnWidth(4, 80)   # Status
        self.tree.setColumnWidth(5, 65)   # Success (increased from 40)
        self.tree.setColumnWidth(6, 60)   # Errors (increased from 40)
        self.tree.setColumnWidth(7, 90)   # Times Used
        
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
        
        # Enable drag and drop using InternalMove (like profile manager)
        # This is the simplest approach - let Qt handle everything
        self.tree.setDragDropMode(QAbstractItemView.InternalMove)
        
        # Connect to model's rowsMoved signal to sync data after Qt moves items
        self.tree.model().rowsMoved.connect(self._on_tree_rows_moved)
        
        # Connect signals
        self.tree.itemDoubleClicked.connect(self._on_click)
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
        
        # Row 5: Individual Output Token Limit
        output_label = QLabel("Output Token Limit:")
        output_label.setStyleSheet("color: gray; font-size: 9pt;")
        add_grid.addWidget(output_label, 5, 0, Qt.AlignLeft)
        self.output_token_spinbox = QSpinBox()
        self.output_token_spinbox.setRange(0, 2000000)
        self.output_token_spinbox.setValue(0)
        self.output_token_spinbox.setMaximumWidth(120)
        self._disable_spinbox_mousewheel(self.output_token_spinbox)
        add_grid.addWidget(self.output_token_spinbox, 5, 1, Qt.AlignLeft)
        output_hint = QLabel("0 = use global limit")
        output_hint.setStyleSheet("color: gray; font-size: 8pt;")
        add_grid.addWidget(output_hint, 5, 2, 1, 2, Qt.AlignLeft)
        
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

    def _on_tree_rows_moved(self):
        """Sync key_pool.keys with tree order after Qt moves rows via drag-drop"""
        # Read the new order from the tree and rebuild key_pool.keys
        new_order = []
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item:
                # Find the matching key by masked API key and model
                masked_key = item.text(0)
                model = item.text(1)
                for key in self.key_pool.keys:
                    key_masked = key.api_key[:8] + "..." + key.api_key[-4:] if len(key.api_key) > 12 else key.api_key
                    if key_masked == masked_key and key.model == model and key not in new_order:
                        new_order.append(key)
                        break
        
        # Update the pool if we got a valid reordering
        if len(new_order) == len(self.key_pool.keys):
            with self.key_pool.lock:
                self.key_pool.keys = new_order
            
            # Save to config
            self._save_keys_to_config()
            
            # Update primary key label
            if hasattr(self, 'primary_key_label') and new_order:
                first_key = new_order[0]
                masked = first_key.api_key[:8] + "..." + first_key.api_key[-4:] if len(first_key.api_key) > 12 else first_key.api_key
                self.primary_key_label.setText(f"⭐ PRIMARY KEY: {first_key.model} ({masked}) ⭐")
            
            self._show_status("Reordered keys")

    def _show_fallback_status(self, message: str):
        """Show status message in the fallback section."""
        if hasattr(self, 'fallback_status_label'):
            self.fallback_status_label.setText(message)
    
    def _on_fallback_rows_moved(self):
        """Sync fallback_keys config with tree order after Qt moves rows via drag-drop"""
        fallback_keys = self.translator_gui.config.get('fallback_keys', [])
        
        # Read the new order from the tree
        new_order = []
        for i in range(self.fallback_tree.topLevelItemCount()):
            item = self.fallback_tree.topLevelItem(i)
            if item:
                # Find the matching key by masked API key and model
                masked_key = item.text(0)
                model = item.text(1)
                for key_data in fallback_keys:
                    api_key = key_data.get('api_key', '')
                    key_masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else api_key
                    if key_masked == masked_key and key_data.get('model', '') == model and key_data not in new_order:
                        new_order.append(key_data)
                        break
        
        # Update config if we got a valid reordering
        if len(new_order) == len(fallback_keys):
            self.translator_gui.config['fallback_keys'] = new_order
            self.translator_gui.save_config(show_message=False)
            self._show_fallback_status("Reordered fallback keys")

    def _refresh_key_list(self):
        """Refresh the key list display preserving test results and highlighting key #1"""
        # Save scroll position and selection
        v_scroll = self.tree.verticalScrollBar().value()
        h_scroll = self.tree.horizontalScrollBar().value()
        
        # Save selection (by index)
        selected_indices = []
        for item in self.tree.selectedItems():
            selected_indices.append(self.tree.indexOfTopLevelItem(item))
            
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
            
            # Determine per-key output token limit display value
            per_key_limit = getattr(key, 'individual_output_token_limit', None)
            if per_key_limit and per_key_limit > 0:
                output_limit_str = str(per_key_limit)
            else:
                output_limit_str = "global"
            
            # Position indicator (not currently shown in a column, but kept for potential future use)
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
            
            # Insert into tree (now includes explicit Output Limit column)
            item = QTreeWidgetItem([
                masked_key,
                key.model,
                f"{key.cooldown}s",
                output_limit_str,
                status,
                str(key.success_count),
                str(key.error_count),
                str(times_used),
            ])
            
            # Tooltip for per-key output token limit
            if per_key_limit and per_key_limit > 0:
                tooltip = f"Individual Output Token Limit: {per_key_limit}"
            else:
                tooltip = "Using global output token limit"
            for col in range(item.columnCount()):
                item.setToolTip(col, tooltip)
            
            # Set colors based on status
            if tags == ('active',):
                for col in range(item.columnCount()):
                    item.setForeground(col, Qt.green)
            elif tags == ('cooling',):
                for col in range(item.columnCount()):
                    item.setForeground(col, Qt.darkYellow)
            elif tags == ('disabled',):
                for col in range(item.columnCount()):
                    item.setForeground(col, Qt.gray)
            elif tags == ('testing',):
                for col in range(item.columnCount()):
                    item.setForeground(col, Qt.blue)
            elif tags == ('passed',):
                for col in range(item.columnCount()):
                    item.setForeground(col, Qt.darkGreen)
            elif tags == ('failed',):
                for col in range(item.columnCount()):
                    item.setForeground(col, Qt.red)
            elif tags == ('ratelimited',):
                for col in range(item.columnCount()):
                    item.setForeground(col, Qt.darkYellow)
            elif tags == ('error',):
                for col in range(item.columnCount()):
                    item.setForeground(col, Qt.darkRed)
            
            # Disable drop on this item to prevent nesting (keep it a flat list)
            item.setFlags(item.flags() & ~Qt.ItemIsDropEnabled)
            self.tree.addTopLevelItem(item)
        
        # Update stats
        active_count = sum(1 for k in keys if k.enabled and not k.is_cooling_down)
        total_count = len(keys)
        passed_count = sum(1 for k in keys if k.last_test_result == 'passed')
        self.stats_label.setText(f"Keys: {active_count} active / {total_count} total | {passed_count} passed tests")
        
        # Restore selection
        for index in selected_indices:
            if index < self.tree.topLevelItemCount():
                item = self.tree.topLevelItem(index)
                item.setSelected(True)
        
        # Restore scroll position
        self.tree.verticalScrollBar().setValue(v_scroll)
        self.tree.horizontalScrollBar().setValue(h_scroll)

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
        """Show context menu with reorder and per-key settings"""
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
        selected_items = self.tree.selectedItems()
        selected_count = len(selected_items)
        if selected_count > 1:
            change_action = menu.addAction(f"Change Model ({selected_count} selected)")
        else:
            change_action = menu.addAction("Change Model")
        change_action.triggered.connect(self._change_model_for_selected)
        
        # Per-key output token limit options
        if selected_count > 1:
            set_limit_action = menu.addAction(f"Set Output Token Limit ({selected_count} selected)")
        else:
            set_limit_action = menu.addAction("Set Output Token Limit")
        set_limit_action.triggered.connect(self._set_output_token_limit_for_selected)
        clear_limit_action = menu.addAction("Clear Output Token Limit")
        clear_limit_action.triggered.connect(self._clear_output_token_limit_for_selected)
        
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
                disable_endpoint_action = menu.addAction("Disable Individual Endpoint")
                disable_endpoint_action.triggered.connect(lambda: self._toggle_individual_endpoint(index, False))
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

    def _set_output_token_limit_for_selected(self):
        """Set per-key output token limit for selected multi-key entries"""
        from PySide6.QtWidgets import QInputDialog
        selected = self.tree.selectedItems()
        if not selected:
            return
        
        # Determine default value from first selected key or global setting
        selected_indices = [self.tree.indexOfTopLevelItem(item) for item in selected]
        default_val = None
        for idx in selected_indices:
            if 0 <= idx < len(self.key_pool.keys):
                key = self.key_pool.keys[idx]
                per_key = getattr(key, 'individual_output_token_limit', None)
                if per_key and per_key > 0:
                    default_val = per_key
                    break
        if default_val is None:
            try:
                default_val = int(getattr(self.translator_gui, 'max_output_tokens', 8192))
            except Exception:
                default_val = 8192
        
        value, ok = QInputDialog.getInt(
            self,
            "Set Output Token Limit",
            "Max output tokens for selected key(s):",
            default_val,
            1,
            2000000,
            512,
        )
        if not ok or value <= 0:
            return
        
        for idx in selected_indices:
            if 0 <= idx < len(self.key_pool.keys):
                self.key_pool.keys[idx].individual_output_token_limit = value
        self._refresh_key_list()
        self._show_status(f"Set output token limit to {value} for {len(selected_indices)} key(s)")
    
    def _clear_output_token_limit_for_selected(self):
        """Clear per-key output token limit for selected multi-key entries"""
        selected = self.tree.selectedItems()
        if not selected:
            return
        selected_indices = [self.tree.indexOfTopLevelItem(item) for item in selected]
        for idx in selected_indices:
            if 0 <= idx < len(self.key_pool.keys):
                self.key_pool.keys[idx].individual_output_token_limit = None
        self._refresh_key_list()
        self._show_status(f"Cleared output token limit for {len(selected_indices)} key(s)")
    
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
    
    def _manage_refusal_patterns(self):
        """Open dialog to manage refusal patterns (non-modal)"""
        # Keep reference to prevent garbage collection
        if not hasattr(self, '_refusal_patterns_dialog') or self._refusal_patterns_dialog is None:
            self._refusal_patterns_dialog = RefusalPatternsDialog(self, self.translator_gui)
            self._refusal_patterns_dialog.setWindowModality(Qt.NonModal)
        else:
            try:
                self._refusal_patterns_dialog._reload_from_config()
            except Exception:
                pass
        
        # Show and raise the dialog
        self._refusal_patterns_dialog.show()
        self._refusal_patterns_dialog.raise_()
        self._refusal_patterns_dialog.activateWindow()
    
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
        
        # Determine per-key output token limit (0 = use global limit)
        individual_output_token_limit = None
        try:
            if hasattr(self, 'output_token_spinbox'):
                val = int(self.output_token_spinbox.value())
                if val > 0:
                    individual_output_token_limit = val
        except Exception:
            individual_output_token_limit = None
        
        # Add to pool with new fields
        key_entry = APIKeyEntry(
            api_key,
            model,
            cooldown,
            enabled=True,
            google_credentials=google_credentials,
            azure_endpoint=azure_endpoint,
            google_region=google_region,
            azure_api_version=azure_api_version,
            use_individual_endpoint=use_individual_endpoint,
            individual_output_token_limit=individual_output_token_limit,
        )
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
        if hasattr(self, 'output_token_spinbox'):
            self.output_token_spinbox.setValue(0)
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
        """Test all enabled API keys with inline progress"""
        if not self.key_pool.keys:
            QMessageBox.warning(self, "Warning", "No keys to test")
            return
        
        # Only test enabled keys
        indices = [i for i, key in enumerate(self.key_pool.keys) if key.enabled]
        
        if not indices:
            QMessageBox.warning(self, "Warning", "No enabled keys to test")
            return
        
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
                
                # Force 1 retries for testing to speed up failure detection
                try:
                    tls = client._get_thread_local_client()
                    tls.max_retries_override = 1
                    print(f"[DEBUG] Set max_retries_override=1 for key test")
                except Exception:
                    pass
                
                
                # Set Google credentials and other key-specific settings
                if hasattr(key, 'google_credentials') and key.google_credentials:
                    client.current_key_google_creds = key.google_credentials
                    client.google_creds_path = key.google_credentials
                    print(f"[DEBUG] Set Google credentials for test: {os.path.basename(key.google_credentials)}")
                
                if hasattr(key, 'google_region') and key.google_region:
                    client.current_key_google_region = key.google_region
                    print(f"[DEBUG] Set Google region for test: {key.google_region}")
                
                # Set Azure endpoint settings if configured
                if hasattr(key, 'use_individual_endpoint') and key.use_individual_endpoint:
                    if hasattr(key, 'azure_endpoint') and key.azure_endpoint:
                        client.current_key_azure_endpoint = key.azure_endpoint
                        client.current_key_use_individual_endpoint = True
                        print(f"[DEBUG] Set Azure endpoint for test: {key.azure_endpoint[:50]}...")
                    
                    if hasattr(key, 'azure_api_version') and key.azure_api_version:
                        client.current_key_azure_api_version = key.azure_api_version
                
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'API test successful' and nothing else."}
                ]
                
                response = client.send(
                    messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                if response and isinstance(response, tuple):
                    content, _ = response
                    if content and "test successful" in content.lower():
                        # Success - update via signal/slot for thread safety
                        if HAS_GUI:
                            QMetaObject.invokeMethod(self, "_handle_test_result", Qt.QueuedConnection, Q_ARG(int, index), Q_ARG(bool, True), Q_ARG(str, "Test passed"))
                        else:
                            self._handle_test_result(index, True, "Test passed")
                        return
                
                # Failed - update via signal/slot
                if HAS_GUI:
                    QMetaObject.invokeMethod(self, "_handle_test_result", Qt.QueuedConnection, Q_ARG(int, index), Q_ARG(bool, False), Q_ARG(str, "Unexpected response"))
                else:
                    self._handle_test_result(index, False, "Unexpected response")
            except Exception as e:
                error_msg = str(e)[:50]
                if HAS_GUI:
                    QMetaObject.invokeMethod(self, "_handle_test_result", Qt.QueuedConnection, Q_ARG(int, index), Q_ARG(bool, False), Q_ARG(str, f"Error: {error_msg}"))
                else:
                    self._handle_test_result(index, False, f"Error: {error_msg}")
        
        # Submit to shared executor like translation does
        if executor:
            executor.submit(run_api_test)
        else:
            # Fallback to thread if no executor
            thread = threading.Thread(target=run_api_test, daemon=True)
            thread.start()
    
    # Decorate _handle_test_result as a slot
    @Slot(int, bool, str) if HAS_GUI else lambda x: x
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
        # Use screen ratios for sizing
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.26)  # 26% of screen width
        height = int(screen.height() * 0.39)  # 39% of screen height
        self.progress_dialog.resize(width, height)
        
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
                        max_tokens=1000,
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
                    
                    # Set Google credentials and other key-specific settings
                    if hasattr(key, 'google_credentials') and key.google_credentials:
                        client.current_key_google_creds = key.google_credentials
                        client.google_creds_path = key.google_credentials
                    
                    if hasattr(key, 'google_region') and key.google_region:
                        client.current_key_google_region = key.google_region
                    
                    # Set Azure endpoint settings if configured
                    if hasattr(key, 'use_individual_endpoint') and key.use_individual_endpoint:
                        if hasattr(key, 'azure_endpoint') and key.azure_endpoint:
                            client.current_key_azure_endpoint = key.azure_endpoint
                            client.current_key_use_individual_endpoint = True
                        
                        if hasattr(key, 'azure_api_version') and key.azure_api_version:
                            client.current_key_azure_api_version = key.azure_api_version
                    
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Say 'API test successful' and nothing else."}
                    ]
                    
                    response = client.send(
                        messages,
                        temperature=0.7,
                        max_tokens=1000
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
