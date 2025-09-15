import json
import os
import time
import tempfile
import shutil
from threading import Lock
from contextlib import contextmanager

class HistoryManager:
    """Thread-safe history management with file locking"""
    
    def __init__(self, payloads_dir):
        self.payloads_dir = payloads_dir
        self.hist_path = os.path.join(payloads_dir, "translation_history.json")
        self.lock = Lock()
        self._file_locks = {}
    
    @contextmanager
    def _file_lock(self, filepath):
        """Simple file locking mechanism"""
        lock_file = filepath + '.lock'
        acquired = False
        try:
            # Try to acquire lock with timeout
            start_time = time.time()
            while time.time() - start_time < 30:  # 30 second timeout
                try:
                    # Create lock file atomically
                    fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                    acquired = True
                    break
                except FileExistsError:
                    time.sleep(0.1)
            
            if not acquired:
                raise TimeoutError(f"Could not acquire lock for {filepath}")
                
            yield
            
        finally:
            if acquired and os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except:
                    pass
    
    def load_history(self):
        """Load history with retry logic and file locking"""
        with self.lock:
            for attempt in range(3):
                try:
                    with self._file_lock(self.hist_path):
                        if os.path.exists(self.hist_path):
                            with open(self.hist_path, "r", encoding="utf-8") as f:
                                return json.load(f)
                        return []
                except (json.JSONDecodeError, IOError) as e:
                    print(f"[WARNING] Failed to load history (attempt {attempt + 1}): {e}")
                    if attempt < 2:
                        time.sleep(0.5)
                    else:
                        # Return empty history if all attempts fail
                        return []
        return []
    
    def save_history(self, history):
        """Save history atomically with file locking"""
        with self.lock:
            with self._file_lock(self.hist_path):
                # Write to temporary file first
                temp_fd, temp_path = tempfile.mkstemp(dir=self.payloads_dir, text=True)
                try:
                    with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                        json.dump(history, f, ensure_ascii=False, indent=2)
                    
                    # Atomically replace the old file
                    shutil.move(temp_path, self.hist_path)
                    
                except Exception as e:
                    # Clean up temp file on error
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise e
    
    def append_to_history(self, user_content, assistant_content, hist_limit, reset_on_limit=True, rolling_window=False):
        """
        Append to history with automatic reset or rolling window when limit is reached
        
        Args:
            user_content: User message content
            assistant_content: Assistant message content
            hist_limit: Maximum number of exchanges to keep (0 = no history)
            reset_on_limit: Whether to reset when limit is reached (old behavior)
            rolling_window: Whether to use rolling window mode (new behavior)
        """
        # CRITICAL FIX: If hist_limit is 0 or negative, don't maintain any history
        if hist_limit <= 0:
            # Don't load, save, or maintain any history when contextual is disabled
            return []
        
        history = self.load_history()
        
        # Count current exchanges (each exchange = 2 messages: user + assistant)
        current_exchanges = len(history) // 2
        
        # Handle limit reached
        if current_exchanges >= hist_limit:
            if rolling_window:
                # Rolling window mode: keep only the most recent (limit-1) exchanges
                # We keep limit-1 to make room for the new exchange
                messages_to_keep = (hist_limit - 1) * 2
                if messages_to_keep > 0:
                    history = history[-messages_to_keep:]
                    print(f"🔄 Rolling history window: keeping last {hist_limit-1} exchanges")
                else:
                    history = []
            elif reset_on_limit:
                # Old behavior: complete reset
                history = []
                print(f"🔄 Reset history after reaching limit of {hist_limit} exchanges")
        
        # Append new entries
        history.append({"role": "user", "content": user_content})
        history.append({"role": "assistant", "content": assistant_content})
        
        self.save_history(history)
        return history

    def will_reset_on_next_append(self, hist_limit, rolling_window=False):
        """Check if the next append will trigger a reset or rolling window"""
        if hist_limit <= 0:
            return False
        history = self.load_history()
        current_exchanges = len(history) // 2
        return current_exchanges >= hist_limit
