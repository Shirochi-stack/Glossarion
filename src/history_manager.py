import json
import os
import time
import tempfile
import shutil
from threading import Lock
from contextlib import contextmanager

class HistoryManager:
    """Thread-safe history management with file locking"""
    
    def __init__(self, payloads_dir, history_filename="translation_history.json"):
        self.payloads_dir = payloads_dir
        self.hist_path = os.path.join(payloads_dir, history_filename)
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
    
    def _has_thought_signature(self, obj) -> bool:
        """
        Detect presence of a thought signature on raw Gemini content objects.
        Supports both snake_case (thought_signature) and camelCase (thoughtSignature)
        forms that appear across Gemini 3 SDK / wire formats.
        """
        try:
            # Google SDK Content object
            if hasattr(obj, "parts"):
                for p in getattr(obj, "parts", []) or []:
                    if getattr(p, "thought_signature", None) or getattr(p, "thoughtSignature", None):
                        return True
                    # Some SDKs may expose signatures inside dict-like parts
                    if hasattr(p, "__dict__") and (
                        p.__dict__.get("thought_signature") or p.__dict__.get("thoughtSignature")
                    ):
                        return True

            # Dict / JSON-serialized form
            if isinstance(obj, dict):
                if obj.get("thought_signature") or obj.get("thoughtSignature"):
                    return True
                for part in obj.get("parts", []) or []:
                    if isinstance(part, dict) and (
                        part.get("thought_signature") or part.get("thoughtSignature")
                    ):
                        return True
            return False
        except Exception:
            return False

    def _make_json_serializable(self, obj):
        """Recursively convert objects to JSON-serializable format"""
        if isinstance(obj, bytes):
            import base64
            return {'_type': 'bytes', 'data': base64.b64encode(obj).decode('ascii')}
        elif isinstance(obj, dict):
            # Special handling for raw_content_object-like dicts with parts
            if 'parts' in obj and isinstance(obj.get('parts'), list):
                new_parts = []
                for p in obj.get('parts') or []:
                    if p is None:
                        continue
                    # Drop empty text fields before serialization
                    if isinstance(p, dict) and p.get('text') == "":
                        p = {k: v for k, v in p.items() if k != 'text'}
                    serialized_part = self._make_json_serializable(p)
                    if isinstance(serialized_part, dict):
                        # Remove empty text after serialization
                        if serialized_part.get('text') == "":
                            serialized_part.pop('text', None)
                        # Drop empty dicts
                        if not serialized_part:
                            continue
                    if serialized_part:
                        new_parts.append(serialized_part)
                cleaned = {k: self._make_json_serializable(v) for k, v in obj.items() if k != 'parts' and v is not None}
                if new_parts:
                    cleaned['parts'] = new_parts
                return cleaned

            cleaned = {}
            for k, v in obj.items():
                if v is None:
                    continue  # drop nulls
                serialized = self._make_json_serializable(v)
                if serialized is None:
                    continue
                cleaned[k] = serialized
            return cleaned
        elif isinstance(obj, list):
            cleaned_list = []
            for v in obj:
                if v is None:
                    continue  # drop null entries
                serialized = self._make_json_serializable(v)
                if serialized is None:
                    continue
                cleaned_list.append(serialized)
            return cleaned_list
        elif isinstance(obj, (str, int, float, bool, type(None))):
            # Primitive types are already serializable
            return obj
        elif hasattr(obj, 'to_dict'):
            # Objects with to_dict method (common in SDKs)
            return self._make_json_serializable(obj.to_dict())
        elif hasattr(obj, 'model_dump'):
            # Pydantic models
            return self._make_json_serializable(obj.model_dump())
        elif hasattr(obj, '__dict__'):
            # Google SDK Content objects and similar - extract parts
            result = {}
            
            # Special handling for Google SDK Content objects
            if hasattr(obj, 'parts'):
                parts = []
                try:
                    for part in obj.parts:
                        part_dict = {}
                        # Check for common attributes including text
                        for attr in ['text', 'thought', 'thought_signature', 'thoughtSignature', 'inline_data', 'function_call', 'function_response']:
                            if hasattr(part, attr):
                                value = getattr(part, attr, None)
                                if value is not None:
                                    part_dict[attr] = self._make_json_serializable(value)
                        # Drop empty text fields
                        if 'text' in part_dict and part_dict['text'] == "":
                            part_dict.pop('text', None)
                        if part_dict:
                            parts.append(part_dict)
                    if parts:
                        result['parts'] = parts
                except Exception:
                    # If we can't iterate parts, skip
                    pass
            
            # Try to extract other attributes
            if hasattr(obj, 'role'):
                try:
                    result['role'] = getattr(obj, 'role', None)
                except:
                    pass
            
            # Mark as from_vertex if it is
            class_name = obj.__class__.__name__
            # print(f"[DEBUG HistoryManager] Serializing object with class name: {class_name}")
            if class_name == 'Content':
                result['_from_vertex'] = True
                # print(f"[DEBUG HistoryManager] Marked as _from_vertex=True")
            
            if result:
                return result
            
            # Fallback: try to serialize __dict__ items
            try:
                obj_dict = {}
                for key, value in obj.__dict__.items():
                    if not key.startswith('_'):
                        obj_dict[key] = self._make_json_serializable(value)
                if obj_dict:
                    return obj_dict
            except:
                pass
            
            # Last resort: convert to string
            return str(obj)
        else:
            # For any other type, convert to string
            return str(obj)

    def save_history(self, history):
        """Save history atomically with file locking"""
        with self.lock:
            with self._file_lock(self.hist_path):
                # Clean history: drop raw_content_object without thought_signature; drop empty content
                cleaned_history = []
                for msg in history:
                    m = msg
                    try:
                        if isinstance(msg, dict):
                            m = dict(msg)
                            if m.get('role') == 'assistant':
                                if '_raw_content_object' in m and not self._has_thought_signature(m['_raw_content_object']):
                                    m.pop('_raw_content_object', None)
                                if m.get('content') == "":
                                    m['content'] = None
                    except Exception:
                        pass
                    cleaned_history.append(m)
                history = cleaned_history
                # Ensure everything is serializable
                safe_history = self._make_json_serializable(history)
                
                # Write to temporary file first
                temp_fd, temp_path = tempfile.mkstemp(dir=self.payloads_dir, text=True)
                try:
                    with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                        json.dump(safe_history, f, ensure_ascii=False, indent=2)
                    
                    # Atomically replace the old file
                    shutil.move(temp_path, self.hist_path)
                    
                except Exception as e:
                    # Clean up temp file on error
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise e
    
    def append_to_history(self, user_content, assistant_content, hist_limit, reset_on_limit=True, rolling_window=False, raw_assistant_object=None):
        """
        Append to history with automatic reset or rolling window when limit is reached
        
        Args:
            user_content: User message content
            assistant_content: Assistant message content
            hist_limit: Maximum number of exchanges to keep (0 = no history)
            reset_on_limit: Whether to reset when limit is reached (old behavior)
            rolling_window: Whether to use rolling window mode (new behavior)
            raw_assistant_object: Optional raw content object (e.g. for thought signatures)
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
                    print(f"📌 Rolling history window: keeping last {hist_limit-1} exchanges")
                else:
                    history = []
            elif reset_on_limit:
                # Old behavior: complete reset
                history = []
                print(f"🔄 Reset history after reaching limit of {hist_limit} exchanges")
        
        # Append new entries
        history.append({"role": "user", "content": user_content})
        
        assistant_msg = {"role": "assistant", "content": assistant_content}


        if raw_assistant_object is not None:
            try:
                # print(f"🔍 [HistoryManager] Attempting to save thought signature (Type: {type(raw_assistant_object)})")
                # Attempt to serialize raw object for thought signatures
                if hasattr(raw_assistant_object, 'to_dict'):
                    # print("   -> Using .to_dict()")
                    assistant_msg['_raw_content_object'] = raw_assistant_object.to_dict()
                elif hasattr(raw_assistant_object, 'model_dump'):
                    # print("   -> Using .model_dump() (Pydantic/New SDK)")
                    assistant_msg['_raw_content_object'] = raw_assistant_object.model_dump()
                elif hasattr(raw_assistant_object, '__dict__'):
                    # Try to extract from __dict__ for Google SDK objects
                    # print("   -> Trying __dict__ serialization (Google SDK)")
                    obj_dict = {}
                    for key, value in raw_assistant_object.__dict__.items():
                        if not key.startswith('_'):  # Skip private attributes
                            # Try to serialize nested objects
                            if hasattr(value, '__dict__'):
                                # Recursively handle nested objects
                                nested = {}
                                for k, v in value.__dict__.items():
                                    if not k.startswith('_'):
                                        # Convert to primitive types
                                        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                                            nested[k] = v
                                        elif hasattr(v, '__str__'):
                                            nested[k] = str(v)
                                obj_dict[key] = nested
                            elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                obj_dict[key] = value
                            elif hasattr(value, '__str__'):
                                obj_dict[key] = str(value)
                    if obj_dict:
                        # print(f"   -> Successfully extracted {len(obj_dict)} fields from Google SDK object")
                        assistant_msg['_raw_content_object'] = obj_dict
                    else:
                        # print("   -> No serializable fields found in __dict__")
                        pass
                elif isinstance(raw_assistant_object, (dict, list, str, int, float, bool)):
                    # print("   -> Object is already primitive")
                    assistant_msg['_raw_content_object'] = raw_assistant_object
                else:
                    # Try to convert to string representation as last resort
                    # print(f"   -> Attempting str() conversion as last resort")
                    str_repr = str(raw_assistant_object)
                    if str_repr and str_repr != str(type(raw_assistant_object)):
                        assistant_msg['_raw_content_object'] = {"_type": "string_repr", "value": str_repr}
                        # print("   -> Saved as string representation")
                    else:
                        print(f"   ⚠️ [HistoryManager] Object has no known serialization method. Type: {type(raw_assistant_object).__module__}.{type(raw_assistant_object).__name__}")
                        # For complex objects that are not JSON serializable, we skip to avoid breaking history
                        pass
            except Exception as e:
                print(f"   ❌ [HistoryManager] Serialization failed: {e}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
                pass
        # Drop raw_content_object if it lacks thought_signature
        if '_raw_content_object' in assistant_msg and not self._has_thought_signature(assistant_msg['_raw_content_object']):
            assistant_msg.pop('_raw_content_object', None)

        # If raw_content_object carries text or thought signatures, clear duplicate assistant content
        try:
            ro = assistant_msg.get('_raw_content_object')
            if ro:
                parts = []
                if hasattr(ro, 'parts'):
                    parts = ro.parts or []
                elif isinstance(ro, dict):
                    parts = ro.get('parts', []) or []
                has_text_or_sig = False
                for p in parts:
                    if hasattr(p, 'text') and getattr(p, 'text', None):
                        has_text_or_sig = True
                        break
                    if isinstance(p, dict):
                        if p.get('text'):
                            has_text_or_sig = True
                            break
                        if p.get('thought_signature') or p.get('thoughtSignature'):
                            has_text_or_sig = True
                            break
                if has_text_or_sig:
                    assistant_msg['content'] = ""
        except Exception:
            pass
            pass
        # If content is now empty, set to None so it gets omitted when serialized
        if assistant_msg.get('content') == "":
            assistant_msg['content'] = None
                
        history.append(assistant_msg)
        
        self.save_history(history)
        return history

    def will_reset_on_next_append(self, hist_limit, rolling_window=False):
        """Check if the next append will trigger a reset or rolling window"""
        if hist_limit <= 0:
            return False
        history = self.load_history()
        current_exchanges = len(history) // 2
        return current_exchanges >= hist_limit
