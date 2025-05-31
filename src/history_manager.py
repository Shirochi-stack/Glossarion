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
    
    def append_to_history(self, user_content, assistant_content, hist_limit):
        """Append to history with automatic trimming"""
        history = self.load_history()
        
        # Trim old entries if needed
        history = history[-(hist_limit * 2 - 2):]  # Keep room for new entry
        
        # Append new entries
        history.append({"role": "user", "content": user_content})
        history.append({"role": "assistant", "content": assistant_content})
        
        self.save_history(history)
        return history