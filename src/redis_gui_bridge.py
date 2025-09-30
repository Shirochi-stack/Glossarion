# redis_gui_bridge.py
"""
Redis-based IPC bridge for tkinter and PySide6 communication.

This module enables separate tkinter and Qt processes to communicate
by synchronizing state through Redis.
"""

import json
import time
import threading
from typing import Any, Dict, Optional, Callable
import subprocess
import sys
import os

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️ Redis not available. Install with: pip install redis")


class RedisGUIBridge:
    """Bridge for synchronizing state between tkinter and Qt processes via Redis."""
    
    def __init__(self, host='localhost', port=6379, db=0, key_prefix='glossarion:gui:'):
        """
        Initialize Redis bridge.
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            key_prefix: Prefix for all Redis keys
        """
        self.host = host
        self.port = port
        self.db = db
        self.key_prefix = key_prefix
        self.redis_client = None
        self._connected = False
        self._watchers = {}  # key -> callback
        self._watch_thread = None
        self._stop_watching = threading.Event()
        
        # Try to connect
        self.connect()
    
    def connect(self) -> bool:
        """Connect to Redis server."""
        if not REDIS_AVAILABLE:
            print("❌ Redis module not installed")
            return False
        
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            # Test connection
            self.redis_client.ping()
            self._connected = True
            print(f"✅ Connected to Redis at {self.host}:{self.port}")
            return True
        except redis.ConnectionError as e:
            print(f"❌ Failed to connect to Redis: {e}")
            print("💡 Make sure Redis is running. See redis_setup.md for instructions.")
            self._connected = False
            return False
        except Exception as e:
            print(f"❌ Redis error: {e}")
            self._connected = False
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        return self._connected and self.redis_client is not None
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.key_prefix}{key}"
    
    # ========== State Management ==========
    
    def set_state(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a state value in Redis.
        
        Args:
            key: State key
            value: Value to store (will be JSON serialized)
            ttl: Optional time-to-live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            return False
        
        try:
            full_key = self._make_key(key)
            serialized = json.dumps(value)
            
            if ttl:
                self.redis_client.setex(full_key, ttl, serialized)
            else:
                self.redis_client.set(full_key, serialized)
            
            return True
        except Exception as e:
            print(f"❌ Failed to set state {key}: {e}")
            return False
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get a state value from Redis.
        
        Args:
            key: State key
            default: Default value if key doesn't exist
            
        Returns:
            Deserialized value or default
        """
        if not self.is_connected():
            return default
        
        try:
            full_key = self._make_key(key)
            value = self.redis_client.get(full_key)
            
            if value is None:
                return default
            
            return json.loads(value)
        except Exception as e:
            print(f"❌ Failed to get state {key}: {e}")
            return default
    
    def delete_state(self, key: str) -> bool:
        """Delete a state value."""
        if not self.is_connected():
            return False
        
        try:
            full_key = self._make_key(key)
            self.redis_client.delete(full_key)
            return True
        except Exception as e:
            print(f"❌ Failed to delete state {key}: {e}")
            return False
    
    # ========== Pub/Sub for Real-time Updates ==========
    
    def publish_event(self, channel: str, message: Any) -> bool:
        """
        Publish an event to a channel.
        
        Args:
            channel: Channel name
            message: Message to publish (will be JSON serialized)
            
        Returns:
            True if successful
        """
        if not self.is_connected():
            return False
        
        try:
            full_channel = self._make_key(channel)
            serialized = json.dumps(message)
            self.redis_client.publish(full_channel, serialized)
            return True
        except Exception as e:
            print(f"❌ Failed to publish to {channel}: {e}")
            return False
    
    def subscribe_event(self, channel: str, callback: Callable[[Any], None]):
        """
        Subscribe to a channel and call callback on messages.
        
        Args:
            channel: Channel name
            callback: Function to call with deserialized messages
        """
        if not self.is_connected():
            return
        
        def listen():
            try:
                pubsub = self.redis_client.pubsub()
                full_channel = self._make_key(channel)
                pubsub.subscribe(full_channel)
                
                print(f"📻 Subscribed to channel: {channel}")
                
                for message in pubsub.listen():
                    if message['type'] == 'message':
                        try:
                            data = json.loads(message['data'])
                            callback(data)
                        except Exception as e:
                            print(f"❌ Error processing message: {e}")
            except Exception as e:
                print(f"❌ Subscription error: {e}")
        
        thread = threading.Thread(target=listen, daemon=True)
        thread.start()
    
    # ========== State Watching (Polling) ==========
    
    def watch_state(self, key: str, callback: Callable[[Any], None], interval: float = 0.5):
        """
        Watch a state key and call callback when it changes.
        
        Args:
            key: State key to watch
            callback: Function to call with new value
            interval: Polling interval in seconds
        """
        self._watchers[key] = (callback, None)  # (callback, last_value)
        
        if not self._watch_thread or not self._watch_thread.is_alive():
            self._start_watch_thread(interval)
    
    def _start_watch_thread(self, interval: float):
        """Start background thread for watching state changes."""
        def watch_loop():
            while not self._stop_watching.is_set():
                for key, (callback, last_value) in list(self._watchers.items()):
                    try:
                        current_value = self.get_state(key)
                        if current_value != last_value:
                            callback(current_value)
                            self._watchers[key] = (callback, current_value)
                    except Exception as e:
                        print(f"❌ Error watching {key}: {e}")
                
                time.sleep(interval)
        
        self._watch_thread = threading.Thread(target=watch_loop, daemon=True)
        self._watch_thread.start()
    
    def stop_watching(self):
        """Stop all state watching."""
        self._stop_watching.set()
        if self._watch_thread:
            self._watch_thread.join(timeout=1.0)
    
    # ========== Process Management ==========
    
    def launch_qt_process(self, script_path: str, **kwargs) -> Optional[subprocess.Popen]:
        """
        Launch a separate PySide6 process.
        
        Args:
            script_path: Path to Python script to run
            **kwargs: Additional arguments to pass as JSON via Redis
            
        Returns:
            Subprocess handle or None if failed
        """
        if not os.path.exists(script_path):
            print(f"❌ Script not found: {script_path}")
            return None
        
        # Store launch arguments in Redis
        if kwargs:
            self.set_state('launch_args', kwargs, ttl=60)
        
        try:
            # Launch separate Python process
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            )
            print(f"🚀 Launched Qt process (PID: {process.pid})")
            return process
        except Exception as e:
            print(f"❌ Failed to launch process: {e}")
            return None
    
    # ========== Manga-Specific State ==========
    
    def set_manga_state(self, state: Dict[str, Any]):
        """Set manga translator state."""
        self.set_state('manga:state', state)
    
    def get_manga_state(self) -> Dict[str, Any]:
        """Get manga translator state."""
        return self.get_state('manga:state', {})
    
    def set_manga_files(self, files: list):
        """Set selected manga files."""
        self.set_state('manga:files', files)
    
    def get_manga_files(self) -> list:
        """Get selected manga files."""
        return self.get_state('manga:files', [])
    
    def set_manga_config(self, config: Dict[str, Any]):
        """Set manga configuration."""
        self.set_state('manga:config', config)
    
    def get_manga_config(self) -> Dict[str, Any]:
        """Get manga configuration."""
        return self.get_state('manga:config', {})
    
    def cleanup(self):
        """Cleanup bridge resources."""
        self.stop_watching()
        if self.redis_client:
            try:
                # Clean up manga-specific keys
                keys = self.redis_client.keys(self._make_key('manga:*'))
                if keys:
                    self.redis_client.delete(*keys)
            except:
                pass


# ========== Singleton Instance ==========

_bridge_instance = None

def get_bridge(auto_connect=True) -> Optional[RedisGUIBridge]:
    """Get singleton bridge instance."""
    global _bridge_instance
    if _bridge_instance is None and auto_connect:
        _bridge_instance = RedisGUIBridge()
    return _bridge_instance


# ========== Example Usage ==========

if __name__ == '__main__':
    # Test the bridge
    bridge = RedisGUIBridge()
    
    if bridge.is_connected():
        print("✅ Redis bridge is working!")
        
        # Test state management
        bridge.set_state('test:key', {'hello': 'world'})
        value = bridge.get_state('test:key')
        print(f"Retrieved value: {value}")
        
        # Test manga state
        bridge.set_manga_files(['test1.png', 'test2.png'])
        files = bridge.get_manga_files()
        print(f"Manga files: {files}")
        
        # Cleanup
        bridge.cleanup()
    else:
        print("❌ Redis bridge not available")
        print("Make sure Redis is running on localhost:6379")