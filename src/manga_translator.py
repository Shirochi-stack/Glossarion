# manga_translator.py
"""
Enhanced Manga Translation Pipeline with improved text visibility controls
Handles OCR, translation, and advanced text rendering for manga panels
Now with proper history management and full page context support
"""

import os
import json
import base64
import logging
import time
import traceback
import cv2
from PIL import ImageEnhance, ImageFilter
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from bubble_detector import BubbleDetector
from TransateKRtoEN import send_with_interrupt

# Google Cloud Vision imports
try:
    from google.cloud import vision
    GOOGLE_CLOUD_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_VISION_AVAILABLE = False
    print("Warning: Google Cloud Vision not installed. Install with: pip install google-cloud-vision")

# Import HistoryManager for proper context management
try:
    from history_manager import HistoryManager
except ImportError:
    HistoryManager = None
    print("Warning: HistoryManager not available. Context tracking will be limited.")

logger = logging.getLogger(__name__)

@dataclass
class TextRegion:
    """Represents a detected text region (speech bubble, narration box, etc.)"""
    text: str
    vertices: List[Tuple[int, int]]  # Polygon vertices from Cloud Vision
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    region_type: str  # 'text_block' from Cloud Vision
    translated_text: Optional[str] = None
    
    def to_dict(self):
        return {
            'text': self.text,
            'vertices': self.vertices,
            'bounding_box': self.bounding_box,
            'confidence': self.confidence,
            'region_type': self.region_type,
            'translated_text': self.translated_text
        }

class MangaTranslator:
    """Main class for manga translation pipeline using Google Cloud Vision + API Key"""
    
    # Global, process-wide registry to make local inpainting init safe across threads
    # Only dictionary operations are locked (microseconds); heavy work happens outside the lock.
    _inpaint_pool_lock = threading.Lock()
    _inpaint_pool = {}  # (method, model_path) -> {'inpainter': obj|None, 'loaded': bool, 'event': threading.Event()}
    
    # Detector preloading pool for non-singleton bubble detector instances
    _detector_pool_lock = threading.Lock()
    _detector_pool = {}  # (detector_type, model_id_or_path) -> {'spares': list[BubbleDetector]}

    # Bubble detector singleton loading coordination
    _singleton_bd_event = threading.Event()
    _singleton_bd_loading = False

    # SINGLETON PATTERN: Shared model instances across all translators
    _singleton_lock = threading.Lock()
    _singleton_bubble_detector = None
    _singleton_local_inpainter = None
    _singleton_refs = 0  # Reference counter for singleton instances
    
    # Class-level cancellation flag for all instances
    _global_cancelled = False
    _global_cancel_lock = threading.RLock()
    
    @classmethod
    def set_global_cancellation(cls, cancelled: bool):
        """Set global cancellation flag for all translator instances"""
        with cls._global_cancel_lock:
            cls._global_cancelled = cancelled
    
    @classmethod
    def is_globally_cancelled(cls) -> bool:
        """Check if globally cancelled"""
        with cls._global_cancel_lock:
            return cls._global_cancelled
    
    @classmethod
    def reset_global_flags(cls):
        """Reset global cancellation flags when starting new translation"""
        with cls._global_cancel_lock:
            cls._global_cancelled = False
    
    def _return_inpainter_to_pool(self):
        """Return a checked-out inpainter instance back to the pool for reuse."""
        if not hasattr(self, '_checked_out_inpainter') or not hasattr(self, '_inpainter_pool_key'):
            return  # Nothing checked out
        
        try:
            with MangaTranslator._inpaint_pool_lock:
                key = self._inpainter_pool_key
                rec = MangaTranslator._inpaint_pool.get(key)
                if rec and 'checked_out' in rec:
                    checked_out = rec['checked_out']
                    if self._checked_out_inpainter in checked_out:
                        checked_out.remove(self._checked_out_inpainter)
                        self._log(f"🔄 Returned inpainter to pool ({len(checked_out)}/{len(rec.get('spares', []))} still in use)", "info")
            # Clear the references
            self._checked_out_inpainter = None
            self._inpainter_pool_key = None
        except Exception as e:
            # Non-critical - just log
            try:
                self._log(f"⚠️ Failed to return inpainter to pool: {e}", "debug")
            except:
                pass
    
    def _return_bubble_detector_to_pool(self):
        """Return a checked-out bubble detector instance back to the pool for reuse."""
        if not hasattr(self, '_checked_out_bubble_detector') or not hasattr(self, '_bubble_detector_pool_key'):
            return  # Nothing checked out
        
        try:
            with MangaTranslator._detector_pool_lock:
                key = self._bubble_detector_pool_key
                rec = MangaTranslator._detector_pool.get(key)
                if rec and 'checked_out' in rec:
                    checked_out = rec['checked_out']
                    if self._checked_out_bubble_detector in checked_out:
                        checked_out.remove(self._checked_out_bubble_detector)
                        self._log(f"🔄 Returned bubble detector to pool ({len(checked_out)}/{len(rec.get('spares', []))} still in use)", "info")
            # Clear the references
            self._checked_out_bubble_detector = None
            self._bubble_detector_pool_key = None
        except Exception as e:
            # Non-critical - just log
            try:
                self._log(f"⚠️ Failed to return bubble detector to pool: {e}", "debug")
            except:
                pass
    
    @classmethod
    def cleanup_singletons(cls, force=False):
        """Clean up singleton instances when no longer needed
        
        Args:
            force: If True, cleanup even if references exist (for app shutdown)
        """
        with cls._singleton_lock:
            if force or cls._singleton_refs == 0:
                # Cleanup singleton bubble detector
                if cls._singleton_bubble_detector is not None:
                    try:
                        if hasattr(cls._singleton_bubble_detector, 'unload'):
                            cls._singleton_bubble_detector.unload(release_shared=True)
                        cls._singleton_bubble_detector = None
                        print("🤖 Singleton bubble detector cleaned up")
                    except Exception as e:
                        print(f"Failed to cleanup singleton bubble detector: {e}")
                
                # Cleanup singleton local inpainter  
                if cls._singleton_local_inpainter is not None:
                    try:
                        if hasattr(cls._singleton_local_inpainter, 'unload'):
                            cls._singleton_local_inpainter.unload()
                        cls._singleton_local_inpainter = None
                        print("🎨 Singleton local inpainter cleaned up")
                    except Exception as e:
                        print(f"Failed to cleanup singleton local inpainter: {e}")
                
                cls._singleton_refs = 0
    
    def __init__(self, ocr_config: dict, unified_client, main_gui, log_callback=None):
        """Initialize with OCR configuration and API client from main GUI
        
        Args:
            ocr_config: Dictionary with OCR provider settings:
                {
                    'provider': 'google' or 'azure',
                    'google_credentials_path': str (if google),
                    'azure_key': str (if azure),
                    'azure_endpoint': str (if azure)
                }
        """
        # CRITICAL: Set thread limits FIRST before any heavy library operations
        # This must happen before cv2, torch, numpy operations
        try:
            parallel_enabled = main_gui.config.get('manga_settings', {}).get('advanced', {}).get('parallel_processing', False)
            if not parallel_enabled:
                # Force single-threaded mode for all computational libraries
                os.environ['OMP_NUM_THREADS'] = '1'
                os.environ['MKL_NUM_THREADS'] = '1'
                os.environ['OPENBLAS_NUM_THREADS'] = '1'
                os.environ['NUMEXPR_NUM_THREADS'] = '1'
                os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
                os.environ['ONNXRUNTIME_NUM_THREADS'] = '1'
                # Set torch and cv2 thread limits if already imported
                try:
                    import torch
                    torch.set_num_threads(1)
                except (ImportError, RuntimeError):
                    pass
                try:
                    cv2.setNumThreads(1)
                except (AttributeError, NameError):
                    pass
        except Exception:
            pass  # Silently fail if config not available
        
        # Set up logging first
        self.log_callback = log_callback
        self.main_gui = main_gui
        
        # Set up stdout capture to redirect prints to GUI
        self._setup_stdout_capture()
        
        # Pass log callback to unified client
        self.client = unified_client
        if hasattr(self.client, 'log_callback'):
            self.client.log_callback = log_callback
        elif hasattr(self.client, 'set_log_callback'):
            self.client.set_log_callback(log_callback)
        self.ocr_config = ocr_config
        self.main_gui = main_gui
        self.log_callback = log_callback
        self.config = main_gui.config
        self.manga_settings = self.config.get('manga_settings', {})
        # Concise logging flag from Advanced settings
        try:
            self.concise_logs = bool(self.manga_settings.get('advanced', {}).get('concise_logs', True))
        except Exception:
            self.concise_logs = True

        # Ensure all GUI environment variables are set
        self._sync_environment_variables()
        
        # Initialize attributes
        self.current_image = None
        self.current_mask = None
        self.text_regions = []
        self.translated_regions = []
        self.final_image = None
        
        # Initialize inpainter attributes
        self.local_inpainter = None
        self.hybrid_inpainter = None
        self.inpainter = None
        
        # Initialize bubble detector (will check singleton mode later)
        self.bubble_detector = None
        # Default: do NOT use singleton models unless explicitly enabled
        self.use_singleton_models = self.manga_settings.get('advanced', {}).get('use_singleton_models', False)
        
        # For bubble detector specifically, prefer a singleton so it stays resident in RAM
        self.use_singleton_bubble_detector = self.manga_settings.get('advanced', {}).get('use_singleton_bubble_detector', True)
        
        # Processing flags
        self.is_processing = False
        self.cancel_requested = False
        self.stop_flag = None  # Initialize stop_flag attribute

        # Initialize batch mode attributes (API parallelism) from environment, not GUI local toggles
        # BATCH_TRANSLATION controls whether UnifiedClient allows concurrent API calls across threads.
        try:
            self.batch_mode = os.getenv('BATCH_TRANSLATION', '0') == '1'
        except Exception:
            self.batch_mode = False
        
        # OCR ROI cache - PER IMAGE ONLY (cleared aggressively to prevent text leakage)
        # CRITICAL: This cache MUST be cleared before every new image to prevent text contamination
        # THREAD-SAFE: Each translator instance has its own cache (safe for parallel panel translation)
        self.ocr_roi_cache = {}
        self._current_image_hash = None  # Track current image to force cache invalidation
        
        # Thread-safe lock for cache operations (critical for parallel panel translation)
        import threading
        self._cache_lock = threading.Lock()
        try:
            self.batch_size = int(os.getenv('BATCH_SIZE', '1'))
        except Exception:
            # Fallback to GUI entry if present; otherwise default to 1
            try:
                self.batch_size = int(main_gui.batch_size_var.get()) if hasattr(main_gui, 'batch_size_var') else 1
            except Exception:
                self.batch_size = 1
        self.batch_current = 1 
        
        if self.batch_mode:
            self._log(f"📦 BATCH MODE: Processing {self.batch_size} images")
            self._log(f"⏱️ Keeping API delay for rate limit protection")
            
            # NOTE: We NO LONGER preload models here!
            # Models should only be loaded when actually needed
            # This was causing unnecessary RAM usage
            ocr_settings = self.manga_settings.get('ocr', {})
            bubble_detection_enabled = ocr_settings.get('bubble_detection_enabled', False)
            if bubble_detection_enabled:
                self._log("📦 BATCH MODE: Bubble detection will be loaded on first use")
            else:
                self._log("📦 BATCH MODE: Bubble detection is disabled")
        
        # Cache for processed images - DEPRECATED/UNUSED (kept for backward compatibility)
        # DO NOT USE THIS FOR TEXT DATA - IT CAN LEAK BETWEEN IMAGES
        self.cache = {}
        # Determine OCR provider
        self.ocr_provider = ocr_config.get('provider', 'google')

        if self.ocr_provider == 'google':
            if not GOOGLE_CLOUD_VISION_AVAILABLE:
                raise ImportError("Google Cloud Vision required. Install with: pip install google-cloud-vision")
            
            google_path = ocr_config.get('google_credentials_path')
            if not google_path:
                raise ValueError("Google credentials path required")
                
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_path
            self.vision_client = vision.ImageAnnotatorClient()
            
        elif self.ocr_provider == 'azure':
            # Import Azure libraries
            try:
                from azure.cognitiveservices.vision.computervision import ComputerVisionClient
                from msrest.authentication import CognitiveServicesCredentials
                self.azure_cv = ComputerVisionClient
                self.azure_creds = CognitiveServicesCredentials
            except ImportError:
                raise ImportError("Azure Computer Vision required. Install with: pip install azure-cognitiveservices-vision-computervision")
            
            azure_key = ocr_config.get('azure_key')
            azure_endpoint = ocr_config.get('azure_endpoint')
            
            if not azure_key or not azure_endpoint:
                raise ValueError("Azure key and endpoint required")
                
            self.vision_client = self.azure_cv(
                azure_endpoint,
                self.azure_creds(azure_key)
            )
        else:
            # New OCR providers handled by OCR manager
            try:
                from ocr_manager import OCRManager
                self.ocr_manager = OCRManager(log_callback=log_callback)
                print(f"Initialized OCR Manager for {self.ocr_provider}")
                # Initialize OCR manager with stop flag awareness
                if hasattr(self.ocr_manager, 'reset_stop_flags'):
                    self.ocr_manager.reset_stop_flags()
            except Exception as _e:
                self.ocr_manager = None
                self._log(f"Failed to initialize OCRManager: {str(_e)}", "error")
        
        self.client = unified_client
        self.main_gui = main_gui
        self.log_callback = log_callback
        
        # Prefer allocator that can return memory to OS (effective before torch loads)
        try:
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        except Exception:
            pass
        
        # Get all settings from GUI
        self.api_delay = float(self.main_gui.delay_entry.get() if hasattr(main_gui, 'delay_entry') else 2.0)
        # Propagate API delay to unified_api_client via env var so its internal pacing/logging matches GUI
        try:
            os.environ["SEND_INTERVAL_SECONDS"] = str(self.api_delay)
        except Exception:
            pass
        self.temperature = float(main_gui.trans_temp.get() if hasattr(main_gui, 'trans_temp') else 0.3)
        self.max_tokens = int(main_gui.max_output_tokens if hasattr(main_gui, 'max_output_tokens') else 4000)
        if hasattr(main_gui, 'token_limit_disabled') and main_gui.token_limit_disabled:
            self.input_token_limit = None  # None means no limit
            self._log("📊 Input token limit: DISABLED (unlimited)")
        else:
            token_limit_value = main_gui.token_limit_entry.get() if hasattr(main_gui, 'token_limit_entry') else '120000'
            if token_limit_value and token_limit_value.strip().isdigit():
                self.input_token_limit = int(token_limit_value.strip())
            else:
                self.input_token_limit = 120000  # Default
            self._log(f"📊 Input token limit: {self.input_token_limit} tokens")
        
        # Get contextual settings from GUI
        self.contextual_enabled = main_gui.contextual_var.get() if hasattr(main_gui, 'contextual_var') else False
        self.translation_history_limit = int(main_gui.trans_history.get() if hasattr(main_gui, 'trans_history') else 3)
        self.rolling_history_enabled = main_gui.translation_history_rolling_var.get() if hasattr(main_gui, 'translation_history_rolling_var') else False
        
        # Initialize HistoryManager placeholder
        self.history_manager = None
        self.history_manager_initialized = False
        self.history_output_dir = None
        
        # Full page context translation settings
        self.full_page_context_enabled = True
        
        # Default prompt for full page context mode
        self.full_page_context_prompt = (
            "You will receive multiple text segments from a manga page. "
            "Translate each segment considering the context of all segments together. "
            "Maintain consistency in character names, tone, and style across all translations.\n\n"
            "IMPORTANT: Return your response as a valid JSON object where each key is the EXACT original text "
            "(without the [0], [1] index prefixes) and each value is the translation.\n"
            "Make sure to properly escape any special characters in the JSON:\n"
            "- Use \\n for newlines\n"
            "- Use \\\" for quotes\n"
            "- Use \\\\ for backslashes\n\n"
            "Example:\n"
            '{\n'
            '  こんにちは: Hello,\n'
            '  ありがとう: Thank you\n'
            '}\n\n'
            'Do NOT include the [0], [1], etc. prefixes in the JSON keys.'
        )

        # Visual context setting (for non-vision model support)
        self.visual_context_enabled = main_gui.config.get('manga_visual_context_enabled', True)
        
        # Store context for contextual translation (backwards compatibility)
        self.translation_context = []
        
        # Font settings for text rendering
        self.font_path = self._find_font()
        self.min_font_size = 10
        self.max_font_size = 60
        try:
            _ms = main_gui.config.get('manga_settings', {}) or {}
            _rend = _ms.get('rendering', {}) or {}
            _font = _ms.get('font_sizing', {}) or {}
            self.min_readable_size = int(_rend.get('auto_min_size', _font.get('min_size', 16)))
        except Exception:
            self.min_readable_size = int(main_gui.config.get('manga_min_readable_size', 16))
        self.max_font_size_limit = main_gui.config.get('manga_max_font_size', 24)
        self.strict_text_wrapping = main_gui.config.get('manga_strict_text_wrapping', False)
        
        # Enhanced text rendering settings - Load from config if available
        config = main_gui.config if hasattr(main_gui, 'config') else {}
        
        self.text_bg_opacity = config.get('manga_bg_opacity', 255)  # 0-255, default fully opaque
        self.text_bg_style = config.get('manga_bg_style', 'box')  # 'box', 'circle', 'wrap'
        self.text_bg_reduction = config.get('manga_bg_reduction', 1.0)  # Size reduction factor (0.5-1.0)
        self.constrain_to_bubble = config.get('manga_constrain_to_bubble', True) 
        
        # Text color from config
        manga_text_color = config.get('manga_text_color', [0, 0, 0])
        self.text_color = tuple(manga_text_color)  # Convert list to tuple
        
        self.outline_color = (255, 255, 255)  # White outline
        self.outline_width_factor = 15  # Divider for font_size to get outline width
        self.selected_font_style = config.get('manga_font_path', None)  # Will store selected font path
        self.custom_font_size = config.get('manga_font_size', None) if config.get('manga_font_size', 0) > 0 else None
        
        # Text shadow settings from config
        self.shadow_enabled = config.get('manga_shadow_enabled', False)
        manga_shadow_color = config.get('manga_shadow_color', [128, 128, 128])
        self.shadow_color = tuple(manga_shadow_color)  # Convert list to tuple
        self.shadow_offset_x = config.get('manga_shadow_offset_x', 2)
        self.shadow_offset_y = config.get('manga_shadow_offset_y', 2)
        self.shadow_blur = config.get('manga_shadow_blur', 0)  # 0 = sharp shadow, higher = more blur
        self.force_caps_lock = config.get('manga_force_caps_lock', False)
        self.skip_inpainting = config.get('manga_skip_inpainting', True)

        # Font size multiplier mode - Load from config
        self.font_size_mode = config.get('manga_font_size_mode', 'fixed')  # 'fixed' or 'multiplier'
        self.font_size_multiplier = config.get('manga_font_size_multiplier', 1.0)  # Default multiplierr
        
        #inpainting quality
        self.inpaint_quality = config.get('manga_inpaint_quality', 'high')  # 'high' or 'fast'
        
        self._log("\n🔧 MangaTranslator initialized with settings:")
        self._log(f"   API Delay: {self.api_delay}s")
        self._log(f"   Temperature: {self.temperature}")
        self._log(f"   Max Output Tokens: {self.max_tokens}")
        self._log(f"   Input Token Limit: {'DISABLED' if self.input_token_limit is None else self.input_token_limit}")
        self._log(f"   Contextual Translation: {'ENABLED' if self.contextual_enabled else 'DISABLED'}")
        self._log(f"   Translation History Limit: {self.translation_history_limit}")
        self._log(f"   Rolling History: {'ENABLED' if self.rolling_history_enabled else 'DISABLED'}")
        self._log(f"   Font Path: {self.font_path or 'Default'}")
        self._log(f"   Text Rendering: BG {self.text_bg_style}, Opacity {int(self.text_bg_opacity/255*100)}%")
        self._log(f"   Shadow: {'ENABLED' if self.shadow_enabled else 'DISABLED'}\n")
        
        self.manga_settings = config.get('manga_settings', {})

        # Initialize local inpainter if configured (respects singleton mode)
        if self.manga_settings.get('inpainting', {}).get('method') == 'local':
            if self.use_singleton_models:
                self._initialize_singleton_local_inpainter()
            else:
                self._initialize_local_inpainter()
            
        # advanced settings
        self.debug_mode = self.manga_settings.get('advanced', {}).get('debug_mode', False)
        self.save_intermediate = self.manga_settings.get('advanced', {}).get('save_intermediate', False)
        self.parallel_processing = self.manga_settings.get('advanced', {}).get('parallel_processing', True)
        self.max_workers = self.manga_settings.get('advanced', {}).get('max_workers', 2)
        # Deep cleanup control: if True, release models after every image (aggressive)
        self.force_deep_cleanup_each_image = self.manga_settings.get('advanced', {}).get('force_deep_cleanup_each_image', False)
        
        # RAM cap
        adv = self.manga_settings.get('advanced', {})
        self.ram_cap_enabled = bool(adv.get('ram_cap_enabled', False))
        self.ram_cap_mb = int(adv.get('ram_cap_mb', 0) or 0)
        self.ram_cap_mode = str(adv.get('ram_cap_mode', 'soft'))
        self.ram_check_interval_sec = float(adv.get('ram_check_interval_sec', 1.0))
        self.ram_recovery_margin_mb = int(adv.get('ram_recovery_margin_mb', 256))
        self._mem_over_cap = False
        self._mem_stop_event = threading.Event()
        self._mem_thread = None
        # Advanced RAM gate tuning
        self.ram_gate_timeout_sec = float(adv.get('ram_gate_timeout_sec', 10.0))
        self.ram_min_floor_over_baseline_mb = int(adv.get('ram_min_floor_over_baseline_mb', 128))
        # Measure baseline at init
        try:
            self.ram_baseline_mb = self._get_process_rss_mb() or 0
        except Exception:
            self.ram_baseline_mb = 0
        if self.ram_cap_enabled and self.ram_cap_mb > 0:
            self._init_ram_cap()
            
            
    def set_stop_flag(self, stop_flag):
        """Set the stop flag for checking interruptions"""
        self.stop_flag = stop_flag
        self.cancel_requested = False
    
    def reset_stop_flags(self):
        """Reset all stop flags when starting new translation"""
        self.cancel_requested = False
        self.is_processing = False
        # Reset global flags
        self.reset_global_flags()
        self._log("🔄 Stop flags reset for new translation", "debug")

    def _check_stop(self):
        """Check if stop has been requested using multiple sources"""
        # Check global cancellation first
        if self.is_globally_cancelled():
            self.cancel_requested = True
            return True
            
        # Check local stop flag (only if it exists and is set)
        if hasattr(self, 'stop_flag') and self.stop_flag and self.stop_flag.is_set():
            self.cancel_requested = True
            return True
            
        # Check processing flag
        if hasattr(self, 'cancel_requested') and self.cancel_requested:
            return True
            
        return False

    def _setup_stdout_capture(self):
        """Set up stdout capture to redirect print statements to GUI"""
        import sys
        import builtins
        
        # Store original print function
        self._original_print = builtins.print
        
        # Create custom print function
        def gui_print(*args, **kwargs):
            """Custom print that redirects to GUI"""
            # Convert args to string
            message = ' '.join(str(arg) for arg in args)
            
            # Check if this is one of the specific messages we want to capture
            if any(marker in message for marker in ['🔍', '✅', '⏳', 'INFO:', 'ERROR:', 'WARNING:']):
                if self.log_callback:
                    # Clean up the message
                    message = message.strip()
                    
                    # Determine level
                    level = 'info'
                    if 'ERROR:' in message or '❌' in message:
                        level = 'error'
                    elif 'WARNING:' in message or '⚠️' in message:
                        level = 'warning'
                    
                    # Remove prefixes like "INFO:" if present
                    for prefix in ['INFO:', 'ERROR:', 'WARNING:', 'DEBUG:']:
                        message = message.replace(prefix, '').strip()
                    
                    # Send to GUI
                    self.log_callback(message, level)
                    return  # Don't print to console
            
            # For other messages, use original print
            self._original_print(*args, **kwargs)
        
        # Replace the built-in print
        builtins.print = gui_print
    
    def __del__(self):
        """Restore original print when MangaTranslator is destroyed"""
        if hasattr(self, '_original_print'):
            import builtins
            builtins.print = self._original_print
        # Best-effort shutdown in case caller forgot to call shutdown()
        try:
            self.shutdown()
        except Exception:
            pass

    def _cleanup_thread_locals(self):
        """Aggressively release thread-local heavy objects (onnx sessions, detectors)."""
        try:
            if hasattr(self, '_thread_local'):
                tl = self._thread_local
                # Release thread-local inpainters
                if hasattr(tl, 'local_inpainters') and isinstance(tl.local_inpainters, dict):
                    try:
                        for inp in list(tl.local_inpainters.values()):
                            try:
                                if hasattr(inp, 'unload'):
                                    inp.unload()
                            except Exception:
                                pass
                    finally:
                        try:
                            tl.local_inpainters.clear()
                        except Exception:
                            pass
                # Return thread-local bubble detector to pool (DO NOT unload)
                if hasattr(tl, 'bubble_detector') and tl.bubble_detector is not None:
                    try:
                        # Instead of unloading, return to pool for reuse
                        self._return_bubble_detector_to_pool()
                        # Keep thread-local reference intact for reuse in next image
                        # Only clear if we're truly shutting down the thread
                    except Exception:
                        pass
        except Exception:
            # Best-effort cleanup only
            pass

    def shutdown(self):
        """Fully release resources for MangaTranslator (models, detectors, torch caches, threads)."""
        try:
            # Decrement singleton reference counter if using singleton mode
            if hasattr(self, 'use_singleton_models') and self.use_singleton_models:
                with MangaTranslator._singleton_lock:
                    MangaTranslator._singleton_refs = max(0, MangaTranslator._singleton_refs - 1)
                    self._log(f"Singleton refs: {MangaTranslator._singleton_refs}", "debug")
            
            # Stop memory watchdog thread if running
            if hasattr(self, '_mem_stop_event') and getattr(self, '_mem_stop_event', None) is not None:
                try:
                    self._mem_stop_event.set()
                except Exception:
                    pass
            # Perform deep cleanup, then try to teardown torch
            try:
                self._deep_cleanup_models()
            except Exception:
                pass
            try:
                self._force_torch_teardown()
            except Exception:
                pass
            try:
                self._huggingface_teardown()
            except Exception:
                pass
            try:
                self._trim_working_set()
            except Exception:
                pass
            # Null out heavy references
            for attr in [
                'client', 'vision_client', 'local_inpainter', 'hybrid_inpainter', 'inpainter',
                'bubble_detector', 'ocr_manager', 'history_manager', 'current_image', 'current_mask',
                'text_regions', 'translated_regions', 'final_image'
            ]:
                try:
                    if hasattr(self, attr):
                        setattr(self, attr, None)
                except Exception:
                    pass
        except Exception as e:
            try:
                self._log(f"⚠️ shutdown() encountered: {e}", "warning")
            except Exception:
                pass

    def _sync_environment_variables(self):
        """Sync all GUI environment variables to ensure manga translation respects GUI settings
        This ensures settings like RETRY_TRUNCATED, THINKING_BUDGET, etc. are properly set
        """
        try:
            # Get config from main_gui if available
            if not hasattr(self, 'main_gui') or not self.main_gui:
                return
            
            # Use the main_gui's set_all_environment_variables method if available
            if hasattr(self.main_gui, 'set_all_environment_variables'):
                self.main_gui.set_all_environment_variables()
            else:
                # Fallback: manually set key variables
                config = self.main_gui.config if hasattr(self.main_gui, 'config') else {}
                
                # Thinking settings (most important for speed)
                thinking_enabled = config.get('enable_gemini_thinking', True)
                thinking_budget = config.get('gemini_thinking_budget', -1)
                
                # CRITICAL FIX: If thinking is disabled, force budget to 0 regardless of config value
                if not thinking_enabled:
                    thinking_budget = 0
                
                os.environ['ENABLE_GEMINI_THINKING'] = '1' if thinking_enabled else '0'
                os.environ['GEMINI_THINKING_BUDGET'] = str(thinking_budget)
                os.environ['THINKING_BUDGET'] = str(thinking_budget)  # Also set for unified_api_client
                
                # Retry settings
                retry_truncated = config.get('retry_truncated', False)
                max_retry_tokens = config.get('max_retry_tokens', 16384)
                max_retries = config.get('max_retries', 7)
                os.environ['RETRY_TRUNCATED'] = '1' if retry_truncated else '0'
                os.environ['MAX_RETRY_TOKENS'] = str(max_retry_tokens)
                os.environ['MAX_RETRIES'] = str(max_retries)
                
                # Safety settings
                disable_gemini_safety = config.get('disable_gemini_safety', False)
                os.environ['DISABLE_GEMINI_SAFETY'] = '1' if disable_gemini_safety else '0'
                
        except Exception as e:
            self._log(f"⚠️ Failed to sync environment variables: {e}", "warning")
    
    def _force_torch_teardown(self):
        """Best-effort teardown of PyTorch CUDA context and caches to drop closer to baseline.
        Safe to call even if CUDA is not available.
        """
        try:
            import torch, os, gc
            # CPU: free cached tensors
            try:
                gc.collect()
            except Exception:
                pass
            # CUDA path
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                # Try to clear cuBLAS workspaces (not always available)
                try:
                    getattr(torch._C, "_cuda_clearCublasWorkspaces")()
                except Exception:
                    pass
                # Optional hard reset via CuPy if present
                reset_done = False
                try:
                    import cupy
                    try:
                        cupy.cuda.runtime.deviceReset()
                        reset_done = True
                        self._log("CUDA deviceReset via CuPy", "debug")
                    except Exception:
                        pass
                except Exception:
                    pass
                # Fallback: attempt to call cudaDeviceReset from cudart on Windows
                if os.name == 'nt' and not reset_done:
                    try:
                        import ctypes
                        candidates = [
                            "cudart64_12.dll", "cudart64_120.dll", "cudart64_110.dll",
                            "cudart64_102.dll", "cudart64_101.dll", "cudart64_100.dll", "cudart64_90.dll"
                        ]
                        for name in candidates:
                            try:
                                dll = ctypes.CDLL(name)
                                dll.cudaDeviceReset.restype = ctypes.c_int
                                rc = dll.cudaDeviceReset()
                                self._log(f"cudaDeviceReset via {name} rc={rc}", "debug")
                                reset_done = True
                                break
                            except Exception:
                                continue
                    except Exception:
                        pass
        except Exception:
            pass

    def _huggingface_teardown(self):
        """Best-effort teardown of HuggingFace/transformers/tokenizers state.
        - Clears on-disk model cache for known repos (via _clear_hf_cache)
        - Optionally purges relevant modules from sys.modules (AGGRESSIVE_HF_UNLOAD=1)
        """
        try:
            import os, sys, gc
            # Clear disk cache for detectors (and any default repo) to avoid growth across runs
            try:
                self._clear_hf_cache()
            except Exception:
                pass
            # Optional aggressive purge of modules to free Python-level caches
            if os.getenv('AGGRESSIVE_HF_UNLOAD', '1') == '1':
                prefixes = (
                    'transformers',
                    'huggingface_hub',
                    'tokenizers',
                    'safetensors',
                    'accelerate',
                )
                to_purge = [m for m in list(sys.modules.keys()) if m.startswith(prefixes)]
                for m in to_purge:
                    try:
                        del sys.modules[m]
                    except Exception:
                        pass
                gc.collect()
        except Exception:
            pass

    def _deep_cleanup_models(self):
        """Release ALL model references and caches to reduce RAM after translation.
        This is the COMPREHENSIVE cleanup that ensures all models are unloaded from RAM.
        """
        self._log("🧹 Starting comprehensive model cleanup to free RAM...", "info")
        
        try:
            # ========== 1. CLEANUP OCR MODELS ==========
            try:
                if hasattr(self, 'ocr_manager'):
                    ocr_manager = getattr(self, 'ocr_manager', None)
                    if ocr_manager:
                        self._log("   Cleaning up OCR models...", "debug")
                        # Clear all loaded OCR providers
                        if hasattr(ocr_manager, 'providers'):
                            for provider_name, provider in ocr_manager.providers.items():
                                try:
                                    # Unload the model
                                    if hasattr(provider, 'model'):
                                        provider.model = None
                                    if hasattr(provider, 'processor'):
                                        provider.processor = None
                                    if hasattr(provider, 'tokenizer'):
                                        provider.tokenizer = None
                                    if hasattr(provider, 'reader'):
                                        provider.reader = None
                                    if hasattr(provider, 'is_loaded'):
                                        provider.is_loaded = False
                                    self._log(f"      ✓ Unloaded {provider_name} OCR provider", "debug")
                                except Exception as e:
                                    self._log(f"      Warning: Failed to unload {provider_name}: {e}", "debug")
                        # Clear the entire OCR manager
                        self.ocr_manager = None
                        self._log("   ✓ OCR models cleaned up", "debug")
            except Exception as e:
                self._log(f"   Warning: OCR cleanup failed: {e}", "debug")

            # ========== 2. CLEANUP BUBBLE DETECTOR (YOLO/RT-DETR) ==========
            try:
                # Instance-level bubble detector
                if hasattr(self, 'bubble_detector') and self.bubble_detector is not None:
                    # Check if using singleton mode - don't unload shared instance
                    if (getattr(self, 'use_singleton_bubble_detector', False)) or (hasattr(self, 'use_singleton_models') and self.use_singleton_models):
                        self._log("   Skipping bubble detector cleanup (singleton mode)", "debug")
                        # Just clear our reference, don't unload the shared instance
                        self.bubble_detector = None
                    else:
                        self._log("   Cleaning up bubble detector (YOLO/RT-DETR)...", "debug")
                        bd = self.bubble_detector
                        try:
                            if hasattr(bd, 'unload'):
                                bd.unload(release_shared=True)  # This unloads YOLO and RT-DETR models
                                self._log("      ✓ Called bubble detector unload", "debug")
                        except Exception as e:
                            self._log(f"      Warning: Bubble detector unload failed: {e}", "debug")
                        self.bubble_detector = None
                        self._log("   ✓ Bubble detector cleaned up", "debug")
                    
                # Also clean class-level shared RT-DETR models unless keeping singleton warm
                if not getattr(self, 'use_singleton_bubble_detector', False):
                    try:
                        from bubble_detector import BubbleDetector
                        if hasattr(BubbleDetector, '_rtdetr_shared_model'):
                            BubbleDetector._rtdetr_shared_model = None
                        if hasattr(BubbleDetector, '_rtdetr_shared_processor'):
                            BubbleDetector._rtdetr_shared_processor = None
                        if hasattr(BubbleDetector, '_rtdetr_loaded'):
                            BubbleDetector._rtdetr_loaded = False
                        self._log("      ✓ Cleared shared RT-DETR cache", "debug")
                    except Exception:
                        pass
                # Clear preloaded detector spares
                try:
                    with MangaTranslator._detector_pool_lock:
                        for rec in MangaTranslator._detector_pool.values():
                            try:
                                rec['spares'] = []
                            except Exception:
                                pass
                except Exception:
                    pass
            except Exception as e:
                self._log(f"   Warning: Bubble detector cleanup failed: {e}", "debug")

            # ========== 3. CLEANUP INPAINTERS ==========
            try:
                self._log("   Cleaning up inpainter models...", "debug")
                
                # Instance-level inpainter
                if hasattr(self, 'local_inpainter') and self.local_inpainter is not None:
                    # Check if using singleton mode - don't unload shared instance
                    if hasattr(self, 'use_singleton_models') and self.use_singleton_models:
                        self._log("      Skipping local inpainter cleanup (singleton mode)", "debug")
                        # Just clear our reference, don't unload the shared instance
                        self.local_inpainter = None
                    else:
                        try:
                            if hasattr(self.local_inpainter, 'unload'):
                                self.local_inpainter.unload()
                                self._log("      ✓ Unloaded local inpainter", "debug")
                        except Exception:
                            pass
                        self.local_inpainter = None
                
                # Hybrid inpainter
                if hasattr(self, 'hybrid_inpainter') and self.hybrid_inpainter is not None:
                    try:
                        if hasattr(self.hybrid_inpainter, 'unload'):
                            self.hybrid_inpainter.unload()
                            self._log("      ✓ Unloaded hybrid inpainter", "debug")
                    except Exception:
                        pass
                    self.hybrid_inpainter = None
                
                # Generic inpainter reference
                if hasattr(self, 'inpainter') and self.inpainter is not None:
                    try:
                        if hasattr(self.inpainter, 'unload'):
                            self.inpainter.unload()
                            self._log("      ✓ Unloaded inpainter", "debug")
                    except Exception:
                        pass
                    self.inpainter = None

                # Release any shared inpainters in the global pool
                with MangaTranslator._inpaint_pool_lock:
                    for key, rec in list(MangaTranslator._inpaint_pool.items()):
                        try:
                            inp = rec.get('inpainter') if isinstance(rec, dict) else None
                            if inp is not None:
                                try:
                                    if hasattr(inp, 'unload'):
                                        inp.unload()
                                        self._log(f"      ✓ Unloaded pooled inpainter: {key}", "debug")
                                except Exception:
                                    pass
                            # Drop any spare instances as well
                            try:
                                for spare in rec.get('spares') or []:
                                    try:
                                        if hasattr(spare, 'unload'):
                                            spare.unload()
                                    except Exception:
                                        pass
                                rec['spares'] = []
                            except Exception:
                                pass
                        except Exception:
                            pass
                    MangaTranslator._inpaint_pool.clear()
                    self._log("      ✓ Cleared inpainter pool", "debug")

                # Release process-wide shared inpainter
                if hasattr(MangaTranslator, '_shared_local_inpainter'):
                    shared = getattr(MangaTranslator, '_shared_local_inpainter', None)
                    if shared is not None:
                        try:
                            if hasattr(shared, 'unload'):
                                shared.unload()
                                self._log("      ✓ Unloaded shared inpainter", "debug")
                        except Exception:
                            pass
                        setattr(MangaTranslator, '_shared_local_inpainter', None)
                
                self._log("   ✓ Inpainter models cleaned up", "debug")
            except Exception as e:
                self._log(f"   Warning: Inpainter cleanup failed: {e}", "debug")

            # ========== 4. CLEANUP THREAD-LOCAL MODELS ==========
            try:
                if hasattr(self, '_thread_local') and self._thread_local is not None:
                    self._log("   Cleaning up thread-local models...", "debug")
                    tl = self._thread_local
                    
                    # Thread-local inpainters
                    if hasattr(tl, 'local_inpainters') and isinstance(tl.local_inpainters, dict):
                        for key, inp in list(tl.local_inpainters.items()):
                            try:
                                if hasattr(inp, 'unload'):
                                    inp.unload()
                                    self._log(f"      ✓ Unloaded thread-local inpainter: {key}", "debug")
                            except Exception:
                                pass
                        tl.local_inpainters.clear()
                    
                    # Thread-local bubble detector
                    if hasattr(tl, 'bubble_detector') and tl.bubble_detector is not None:
                        try:
                            if hasattr(tl.bubble_detector, 'unload'):
                                tl.bubble_detector.unload(release_shared=False)
                                self._log("      ✓ Unloaded thread-local bubble detector", "debug")
                        except Exception:
                            pass
                        tl.bubble_detector = None
                    
                    self._log("   ✓ Thread-local models cleaned up", "debug")
            except Exception as e:
                self._log(f"   Warning: Thread-local cleanup failed: {e}", "debug")

            # ========== 5. CLEAR PYTORCH/CUDA CACHE ==========
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    self._log("   ✓ Cleared CUDA cache", "debug")
            except Exception:
                pass

            # ========== 6. FORCE GARBAGE COLLECTION ==========
            try:
                import gc
                gc.collect()
                # Multiple passes for stubborn references
                gc.collect()
                gc.collect()
                self._log("   ✓ Forced garbage collection", "debug")
            except Exception:
                pass
            
            self._log("✅ Model cleanup complete - RAM should be freed", "info")
            
        except Exception as e:
            # Never raise from deep cleanup
            self._log(f"⚠️ Model cleanup encountered error: {e}", "warning")
            pass

    def _clear_hf_cache(self, repo_id: str = None):
        """Best-effort: clear Hugging Face cache for a specific repo (RT-DETR by default).
        This targets disk cache; it won’t directly reduce RAM but helps avoid growth across runs.
        """
        try:
            # Determine repo_id from BubbleDetector if not provided
            if repo_id is None:
                try:
                    import bubble_detector as _bdmod
                    BD = getattr(_bdmod, 'BubbleDetector', None)
                    if BD is not None and hasattr(BD, '_rtdetr_repo_id'):
                        repo_id = getattr(BD, '_rtdetr_repo_id') or 'ogkalu/comic-text-and-bubble-detector'
                    else:
                        repo_id = 'ogkalu/comic-text-and-bubble-detector'
                except Exception:
                    repo_id = 'ogkalu/comic-text-and-bubble-detector'

            # Try to use huggingface_hub to delete just the matching repo cache
            try:
                from huggingface_hub import scan_cache_dir
                info = scan_cache_dir()
                repos = getattr(info, 'repos', [])
                to_delete = []
                for repo in repos:
                    rid = getattr(repo, 'repo_id', None) or getattr(repo, 'id', None)
                    if rid == repo_id:
                        to_delete.append(repo)
                if to_delete:
                    # Prefer the high-level deletion API if present
                    if hasattr(info, 'delete_repos'):
                        info.delete_repos(to_delete)
                    else:
                        import shutil
                        for repo in to_delete:
                            repo_dir = getattr(repo, 'repo_path', None) or getattr(repo, 'repo_dir', None)
                            if repo_dir and os.path.exists(repo_dir):
                                shutil.rmtree(repo_dir, ignore_errors=True)
            except Exception:
                # Fallback: try removing default HF cache dir for this repo pattern
                try:
                    from pathlib import Path
                    hf_home = os.environ.get('HF_HOME')
                    if hf_home:
                        base = Path(hf_home)
                    else:
                        base = Path.home() / '.cache' / 'huggingface' / 'hub'
                    # Repo cache dirs are named like models--{org}--{name}
                    safe_name = repo_id.replace('/', '--')
                    candidates = list(base.glob(f'models--{safe_name}*'))
                    import shutil
                    for c in candidates:
                        shutil.rmtree(str(c), ignore_errors=True)
                except Exception:
                    pass
        except Exception:
            # Best-effort only
            pass

    def _trim_working_set(self):
        """Release freed memory back to the OS where possible.
        - On Windows: use EmptyWorkingSet on current process
        - On Linux: attempt malloc_trim(0)
        - On macOS: no direct API; rely on GC
        """
        import sys
        import platform
        try:
            system = platform.system()
            if system == 'Windows':
                import ctypes
                psapi = ctypes.windll.psapi
                kernel32 = ctypes.windll.kernel32
                h_process = kernel32.GetCurrentProcess()
                psapi.EmptyWorkingSet(h_process)
            elif system == 'Linux':
                import ctypes
                libc = ctypes.CDLL('libc.so.6')
                try:
                    libc.malloc_trim(0)
                except Exception:
                    pass
        except Exception:
            pass

    def _get_process_rss_mb(self) -> int:
        """Return current RSS in MB (cross-platform best-effort)."""
        try:
            import psutil, os as _os
            return int(psutil.Process(_os.getpid()).memory_info().rss / (1024*1024))
        except Exception:
            # Windows fallback
            try:
                import ctypes, os as _os
                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("cb", ctypes.c_uint),
                        ("PageFaultCount", ctypes.c_uint),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]
                GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
                GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                GetProcessMemoryInfo(GetCurrentProcess(), ctypes.byref(counters), counters.cb)
                return int(counters.WorkingSetSize / (1024*1024))
            except Exception:
                return 0

    def _apply_windows_job_memory_limit(self, cap_mb: int) -> bool:
        """Apply a hard memory cap using Windows Job Objects. Returns True on success."""
        try:
            import ctypes
            from ctypes import wintypes
            JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
            JobObjectExtendedLimitInformation = 9

            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_void_p),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_ulonglong),
                    ("WriteOperationCount", ctypes.c_ulonglong),
                    ("OtherOperationCount", ctypes.c_ulonglong),
                    ("ReadTransferCount", ctypes.c_ulonglong),
                    ("WriteTransferCount", ctypes.c_ulonglong),
                    ("OtherTransferCount", ctypes.c_ulonglong),
                ]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            CreateJobObject = kernel32.CreateJobObjectW
            CreateJobObject.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            CreateJobObject.restype = wintypes.HANDLE
            SetInformationJobObject = kernel32.SetInformationJobObject
            SetInformationJobObject.argtypes = [wintypes.HANDLE, wintypes.INT, ctypes.c_void_p, wintypes.DWORD]
            SetInformationJobObject.restype = wintypes.BOOL
            AssignProcessToJobObject = kernel32.AssignProcessToJobObject
            AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            AssignProcessToJobObject.restype = wintypes.BOOL
            GetCurrentProcess = kernel32.GetCurrentProcess
            GetCurrentProcess.restype = wintypes.HANDLE

            hJob = CreateJobObject(None, None)
            if not hJob:
                return False

            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY
            info.JobMemoryLimit = ctypes.c_size_t(int(cap_mb) * 1024 * 1024)

            ok = SetInformationJobObject(hJob, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info))
            if not ok:
                return False

            ok = AssignProcessToJobObject(hJob, GetCurrentProcess())
            if not ok:
                return False
            return True
        except Exception:
            return False

    def _memory_watchdog(self):
        try:
            import time
            while not self._mem_stop_event.is_set():
                if not self.ram_cap_enabled or self.ram_cap_mb <= 0:
                    break
                rss = self._get_process_rss_mb()
                if rss and rss > self.ram_cap_mb:
                    self._mem_over_cap = True
                    # Aggressive attempt to reduce memory
                    try:
                        self._deep_cleanup_models()
                    except Exception:
                        pass
                    try:
                        self._trim_working_set()
                    except Exception:
                        pass
                    # Wait a bit before re-checking
                    time.sleep(max(0.2, self.ram_check_interval_sec / 2))
                    time.sleep(0.1)  # Brief pause for stability
                    self._log("💤 Memory watchdog pausing briefly for stability", "debug")
                else:
                    # Below cap or couldn't read RSS
                    self._mem_over_cap = False
                    time.sleep(self.ram_check_interval_sec)
        except Exception:
            pass

    def _init_ram_cap(self):
        # Hard cap via Windows Job Object if selected and on Windows
        try:
            import platform
            if self.ram_cap_mode.startswith('hard') or self.ram_cap_mode == 'hard':
                if platform.system() == 'Windows':
                    if not self._apply_windows_job_memory_limit(self.ram_cap_mb):
                        self._log("⚠️ Failed to apply hard RAM cap; falling back to soft mode", "warning")
                        self.ram_cap_mode = 'soft'
                else:
                    self._log("⚠️ Hard RAM cap only supported on Windows; using soft mode", "warning")
                    self.ram_cap_mode = 'soft'
        except Exception:
            self.ram_cap_mode = 'soft'
        # Start watchdog regardless of mode to proactively stay under cap during operations
        try:
            self._mem_thread = threading.Thread(target=self._memory_watchdog, daemon=True)
            self._mem_thread.start()
        except Exception:
            pass

    def _block_if_over_cap(self, context_msg: str = ""):
        # If over cap, block until we drop under cap - margin
        if not self.ram_cap_enabled or self.ram_cap_mb <= 0:
            return
        import time
        # Never require target below baseline + floor margin
        baseline = max(0, getattr(self, 'ram_baseline_mb', 0))
        floor = baseline + max(0, self.ram_min_floor_over_baseline_mb)
        # Compute target below cap by recovery margin, but not below floor
        target = self.ram_cap_mb - max(64, min(self.ram_recovery_margin_mb, self.ram_cap_mb // 4))
        target = max(target, floor)
        start = time.time()
        waited = False
        last_log = 0
        while True:
            rss = self._get_process_rss_mb()
            now = time.time()
            if rss and rss <= target:
                break
            # Timeout to avoid deadlock when baseline can't go lower than target
            if now - start > max(2.0, self.ram_gate_timeout_sec):
                self._log(f"⌛ RAM gate timeout for {context_msg}: RSS={rss} MB, target={target} MB; proceeding in low-memory mode", "warning")
                break
            waited = True
            # Periodic log to help diagnose
            if now - last_log > 3.0 and rss:
                self._log(f"⏳ Waiting for RAM drop: RSS={rss} MB, target={target} MB ({context_msg})", "info")
                last_log = now
            # Attempt cleanup while waiting
            try:
                self._deep_cleanup_models()
            except Exception:
                pass
            try:
                self._trim_working_set()
            except Exception:
                pass
            if self._check_stop():
                break
            time.sleep(0.1)  # Brief pause for stability
            self._log("💤 RAM gate pausing briefly for stability", "debug")
        if waited and context_msg:
            self._log(f"🧹 Proceeding with {context_msg} (RSS now {self._get_process_rss_mb()} MB; target {target} MB)", "info")

    def set_batch_mode(self, enabled: bool, batch_size: int = 1):
        """Enable or disable batch mode optimizations"""
        self.batch_mode = enabled
        self.batch_size = batch_size
        
        if enabled:
            # Check if bubble detection is actually enabled before considering preload
            ocr_settings = self.manga_settings.get('ocr', {}) if hasattr(self, 'manga_settings') else {}
            bubble_detection_enabled = ocr_settings.get('bubble_detection_enabled', False)
            
            # Only suggest preloading if bubble detection is actually going to be used
            if bubble_detection_enabled:
                self._log("📦 BATCH MODE: Bubble detection models will load on first use")
                # NOTE: We don't actually preload anymore to save RAM
                # Models are loaded on-demand when first needed
            
            # Similarly for OCR models - they load on demand
            if hasattr(self, 'ocr_manager') and self.ocr_manager:
                self._log(f"📦 BATCH MODE: {self.ocr_provider} will load on first use")
                # NOTE: We don't preload OCR models either
            
            self._log(f"📦 BATCH MODE ENABLED: Processing {batch_size} images")
            self._log(f"⏱️ API delay: {self.api_delay}s (preserved for rate limiting)")
        else:
            self._log("📝 BATCH MODE DISABLED")

    def _ensure_bubble_detector_ready(self, ocr_settings):
        """Ensure a usable BubbleDetector for current thread, auto-reloading models after cleanup."""
        try:
            bd = self._get_thread_bubble_detector()
            detector_type = ocr_settings.get('detector_type', 'rtdetr_onnx')
            if detector_type == 'rtdetr_onnx':
                if not getattr(bd, 'rtdetr_onnx_loaded', False):
                    model_id = ocr_settings.get('rtdetr_model_url') or ocr_settings.get('bubble_model_path')
                    if not bd.load_rtdetr_onnx_model(model_id=model_id):
                        return None
            elif detector_type == 'rtdetr':
                if not getattr(bd, 'rtdetr_loaded', False):
                    model_id = ocr_settings.get('rtdetr_model_url') or ocr_settings.get('bubble_model_path')
                    if not bd.load_rtdetr_model(model_id=model_id):
                        return None
            elif detector_type == 'yolo':
                model_path = ocr_settings.get('bubble_model_path')
                if model_path and not getattr(bd, 'model_loaded', False):
                    if not bd.load_model(model_path):
                        return None
            else:  # auto
                # Prefer RT-DETR if available, else YOLO if configured
                if not getattr(bd, 'rtdetr_loaded', False):
                    bd.load_rtdetr_model(model_id=ocr_settings.get('rtdetr_model_url') or ocr_settings.get('bubble_model_path'))
            return bd
        except Exception:
            return None

    def _merge_with_bubble_detection(self, regions: List[TextRegion], image_path: str) -> List[TextRegion]:
        """Merge text regions by bubble and filter based on RT-DETR class settings"""
        try:
            # Get detector settings from config
            ocr_settings = self.main_gui.config.get('manga_settings', {}).get('ocr', {})
            detector_type = ocr_settings.get('detector_type', 'rtdetr_onnx')
            
            # Ensure detector is ready (auto-reload after cleanup)
            bd = self._ensure_bubble_detector_ready(ocr_settings)
            if bd is None:
                self._log("⚠️ Bubble detector unavailable after cleanup; falling back to proximity merge", "warning")
                return self._merge_nearby_regions(regions)
            
            # Check if bubble detection is enabled
            if not ocr_settings.get('bubble_detection_enabled', False):
                self._log("📦 Bubble detection is disabled in settings", "info")
                return self._merge_nearby_regions(regions)
            
            # Initialize thread-local detector
            bd = self._get_thread_bubble_detector()
            
            bubbles = None
            rtdetr_detections = None
            
            if detector_type == 'rtdetr_onnx':
                if not self.batch_mode:
                    self._log("🤖 Using RTEDR_onnx for bubble detection", "info")
                if self.batch_mode and getattr(bd, 'rtdetr_onnx_loaded', False):
                    pass
                elif not getattr(bd, 'rtdetr_onnx_loaded', False):
                    self._log("📥 Loading RTEDR_onnx model...", "info")
                    if not bd.load_rtdetr_onnx_model():
                        self._log("⚠️ Failed to load RTEDR_onnx, falling back to traditional merging", "warning")
                        return self._merge_nearby_regions(regions)
                    else:
                        # Model loaded successfully - mark in pool for reuse
                        try:
                            model_id = ocr_settings.get('rtdetr_model_url') or ocr_settings.get('bubble_model_path') or ''
                            key = ('rtdetr_onnx', model_id)
                            with MangaTranslator._detector_pool_lock:
                                if key not in MangaTranslator._detector_pool:
                                    MangaTranslator._detector_pool[key] = {'spares': []}
                                # Mark this detector type as loaded for next run
                                MangaTranslator._detector_pool[key]['loaded'] = True
                        except Exception:
                            pass
                rtdetr_confidence = ocr_settings.get('rtdetr_confidence', 0.3)
                detect_empty = ocr_settings.get('detect_empty_bubbles', True)
                detect_text_bubbles = ocr_settings.get('detect_text_bubbles', True)
                detect_free_text = ocr_settings.get('detect_free_text', True)
                if not self.batch_mode:
                    self._log(f"📋 RTEDR_onnx class filters:", "info")
                    self._log(f"   Empty bubbles: {'✓' if detect_empty else '✗'}", "info")
                    self._log(f"   Text bubbles: {'✓' if detect_text_bubbles else '✗'}", "info")
                    self._log(f"   Free text: {'✓' if detect_free_text else '✗'}", "info")
                    self._log(f"🎯 RTEDR_onnx confidence threshold: {rtdetr_confidence:.2f}", "info")
                rtdetr_detections = bd.detect_with_rtdetr_onnx(
                    image_path=image_path,
                    confidence=rtdetr_confidence,
                    return_all_bubbles=False
                )
                # Combine enabled bubble types for merging
                bubbles = []
                if detect_empty and 'bubbles' in rtdetr_detections:
                    bubbles.extend(rtdetr_detections['bubbles'])
                if detect_text_bubbles and 'text_bubbles' in rtdetr_detections:
                    bubbles.extend(rtdetr_detections['text_bubbles'])
                # Store free text locations for filtering later
                free_text_regions = rtdetr_detections.get('text_free', []) if detect_free_text else []
                self._log(f"✅ RTEDR_onnx detected:", "success")
                self._log(f"   {len(rtdetr_detections.get('bubbles', []))} empty bubbles", "info")
                self._log(f"   {len(rtdetr_detections.get('text_bubbles', []))} text bubbles", "info")
                self._log(f"   {len(rtdetr_detections.get('text_free', []))} free text regions", "info")
            elif detector_type == 'rtdetr':
                # BATCH OPTIMIZATION: Less verbose logging
                if not self.batch_mode:
                    self._log("🤖 Using RT-DETR for bubble detection", "info")
                
                # BATCH OPTIMIZATION: Don't reload if already loaded
                if self.batch_mode and bd.rtdetr_loaded:
                    # Model already loaded, skip the loading step entirely
                    pass
                elif not bd.rtdetr_loaded:
                    self._log("📥 Loading RT-DETR model...", "info")
                    if not bd.load_rtdetr_model():
                        self._log("⚠️ Failed to load RT-DETR, falling back to traditional merging", "warning")
                        return self._merge_nearby_regions(regions)
                    else:
                        # Model loaded successfully - mark in pool for reuse
                        try:
                            model_id = ocr_settings.get('rtdetr_model_url') or ocr_settings.get('bubble_model_path') or ''
                            key = ('rtdetr', model_id)
                            with MangaTranslator._detector_pool_lock:
                                if key not in MangaTranslator._detector_pool:
                                    MangaTranslator._detector_pool[key] = {'spares': []}
                                # Mark this detector type as loaded for next run
                                MangaTranslator._detector_pool[key]['loaded'] = True
                        except Exception:
                            pass
                
                # Get settings
                rtdetr_confidence = ocr_settings.get('rtdetr_confidence', 0.3)
                detect_empty = ocr_settings.get('detect_empty_bubbles', True)
                detect_text_bubbles = ocr_settings.get('detect_text_bubbles', True)
                detect_free_text = ocr_settings.get('detect_free_text', True)
                
                # BATCH OPTIMIZATION: Reduce logging
                if not self.batch_mode:
                    self._log(f"📋 RT-DETR class filters:", "info")
                    self._log(f"   Empty bubbles: {'✓' if detect_empty else '✗'}", "info")
                    self._log(f"   Text bubbles: {'✓' if detect_text_bubbles else '✗'}", "info")
                    self._log(f"   Free text: {'✓' if detect_free_text else '✗'}", "info")
                    self._log(f"🎯 RT-DETR confidence threshold: {rtdetr_confidence:.2f}", "info")

                # Get FULL RT-DETR detections (not just bubbles)
                rtdetr_detections = bd.detect_with_rtdetr(
                    image_path=image_path,
                    confidence=rtdetr_confidence,
                    return_all_bubbles=False  # Get dict with all classes
                )
                
                # Combine enabled bubble types for merging
                bubbles = []
                if detect_empty and 'bubbles' in rtdetr_detections:
                    bubbles.extend(rtdetr_detections['bubbles'])
                if detect_text_bubbles and 'text_bubbles' in rtdetr_detections:
                    bubbles.extend(rtdetr_detections['text_bubbles'])
                
                # Store free text locations for filtering later
                free_text_regions = rtdetr_detections.get('text_free', []) if detect_free_text else []

                # Helper to test if a point lies in any bbox
                def _point_in_any_bbox(cx, cy, boxes):
                    try:
                        for (bx, by, bw, bh) in boxes or []:
                            if bx <= cx <= bx + bw and by <= cy <= by + bh:
                                return True
                    except Exception:
                        pass
                    return False
                
                self._log(f"✅ RT-DETR detected:", "success")
                self._log(f"   {len(rtdetr_detections.get('bubbles', []))} empty bubbles", "info")
                self._log(f"   {len(rtdetr_detections.get('text_bubbles', []))} text bubbles", "info")
                self._log(f"   {len(rtdetr_detections.get('text_free', []))} free text regions", "info")
                
            elif detector_type == 'yolo':
                # Use YOLOv8 (existing code)
                self._log("🤖 Using YOLOv8 for bubble detection", "info")
                
                model_path = ocr_settings.get('bubble_model_path')
                if not model_path:
                    self._log("⚠️ No YOLO model configured, falling back to traditional merging", "warning")
                    return self._merge_nearby_regions(regions)
                
                if not bd.model_loaded:
                    self._log(f"📥 Loading YOLO model: {os.path.basename(model_path)}")
                    if not bd.load_model(model_path):
                        self._log("⚠️ Failed to load YOLO model, falling back to traditional merging", "warning")
                        return self._merge_nearby_regions(regions)
                
                confidence = ocr_settings.get('bubble_confidence', 0.3)
                self._log(f"🎯 Detecting bubbles with YOLO (confidence >= {confidence:.2f})")
                bubbles = bd.detect_bubbles(image_path, confidence=confidence, use_rtdetr=False)
                
            else:
                # Unknown detector type
                self._log(f"❌ Unknown detector type: {detector_type}", "error")
                self._log("   Valid options: rtdetr_onnx, rtdetr, yolo", "error")
                return self._merge_nearby_regions(regions)
            
            if not bubbles:
                self._log("⚠️ No bubbles detected, using traditional merging", "warning")
                return self._merge_nearby_regions(regions)
            
            self._log(f"✅ Found {len(bubbles)} bubbles for grouping", "success")
            
            # Merge regions within bubbles
            merged_regions = []
            used_indices = set()
            
            # Build lookup of free text regions for exclusion
            free_text_bboxes = free_text_regions if detector_type in ('rtdetr', 'rtdetr_onnx') else []
            
            # DEBUG: Log free text bboxes
            if free_text_bboxes:
                self._log(f"🔍 Free text exclusion zones: {len(free_text_bboxes)} regions", "debug")
                for idx, (fx, fy, fw, fh) in enumerate(free_text_bboxes):
                    self._log(f"   Free text zone {idx + 1}: x={fx:.0f}, y={fy:.0f}, w={fw:.0f}, h={fh:.0f}", "debug")
            else:
                self._log(f"⚠️ No free text exclusion zones detected by RT-DETR", "warning")
            
            # Helper to check if a point is in any free text region
            def _point_in_free_text(cx, cy, free_boxes):
                try:
                    for idx, (fx, fy, fw, fh) in enumerate(free_boxes or []):
                        if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                            self._log(f"      ✓ Point ({cx:.0f}, {cy:.0f}) is in free text zone {idx + 1}", "debug")
                            return True
                except Exception as e:
                    self._log(f"      ⚠️ Error checking free text: {e}", "debug")
                    pass
                return False
            
            for bubble_idx, (bx, by, bw, bh) in enumerate(bubbles):
                bubble_regions = []
                self._log(f"\n   Processing bubble {bubble_idx + 1}: x={bx:.0f}, y={by:.0f}, w={bw:.0f}, h={bh:.0f}", "debug")
                
                for idx, region in enumerate(regions):
                    if idx in used_indices:
                        continue
                        
                    rx, ry, rw, rh = region.bounding_box
                    region_center_x = rx + rw / 2
                    region_center_y = ry + rh / 2
                    
                    # Check if center is inside this bubble
                    if (bx <= region_center_x <= bx + bw and 
                        by <= region_center_y <= by + bh):
                        
                        self._log(f"      Region '{region.text[:20]}...' center ({region_center_x:.0f}, {region_center_y:.0f}) is in bubble", "debug")
                        
                        # CRITICAL: Don't merge if this region is in a free text area
                        # Free text should stay separate from bubbles
                        if _point_in_free_text(region_center_x, region_center_y, free_text_bboxes):
                            # This region is in a free text area, don't merge it into bubble
                            self._log(f"      ❌ SKIPPING: Region overlaps with free text area", "debug")
                            continue
                        
                        self._log(f"      ✓ Adding region to bubble {bubble_idx + 1}", "debug")
                        bubble_regions.append(region)
                        used_indices.add(idx)
                
                if bubble_regions:
                    merged_text = " ".join(r.text for r in bubble_regions)
                    
                    min_x = min(r.bounding_box[0] for r in bubble_regions)
                    min_y = min(r.bounding_box[1] for r in bubble_regions)
                    max_x = max(r.bounding_box[0] + r.bounding_box[2] for r in bubble_regions)
                    max_y = max(r.bounding_box[1] + r.bounding_box[3] for r in bubble_regions)
                    
                    all_vertices = []
                    for r in bubble_regions:
                        if hasattr(r, 'vertices') and r.vertices:
                            all_vertices.extend(r.vertices)
                    
                    if not all_vertices:
                        all_vertices = [
                            (min_x, min_y),
                            (max_x, min_y),
                            (max_x, max_y),
                            (min_x, max_y)
                        ]
                    
                    merged_region = TextRegion(
                        text=merged_text,
                        vertices=all_vertices,
                        bounding_box=(min_x, min_y, max_x - min_x, max_y - min_y),
                        confidence=0.95,
                        region_type='bubble_detected'
                    )
                    
                    # Store original regions for masking
                    merged_region.original_regions = bubble_regions
                    merged_region.bubble_bounds = (bx, by, bw, bh)
                    # Classify as text bubble for downstream rendering/masking
                    merged_region.bubble_type = 'text_bubble'
                    # Mark that this should be inpainted
                    merged_region.should_inpaint = True
                    
                    merged_regions.append(merged_region)
                    self._log(f"   Bubble {bubble_idx + 1}: Merged {len(bubble_regions)} text regions", "info")
            
            # Handle text outside bubbles based on RT-DETR settings
            for idx, region in enumerate(regions):
                if idx not in used_indices:
                    # This text is outside any bubble
                    
                    # For RT-DETR mode, check if we should include free text
                    if detector_type in ('rtdetr', 'rtdetr_onnx'):
                        # If "Free Text" checkbox is checked, include ALL text outside bubbles
                        # Don't require RT-DETR to specifically detect it as free text
                        if ocr_settings.get('detect_free_text', True):
                            region.should_inpaint = True
                            # If RT-DETR detected free text box covering this region's center, mark explicitly
                            try:
                                cx = region.bounding_box[0] + region.bounding_box[2] / 2
                                cy = region.bounding_box[1] + region.bounding_box[3] / 2
                                if _point_in_free_text(cx, cy, free_text_bboxes):
                                    region.bubble_type = 'free_text'
                                    self._log(f"   Free text region INCLUDED: '{region.text[:30]}...'", "debug")
                                else:
                                    # Text outside bubbles but not in free text box - still mark as free text
                                    region.bubble_type = 'free_text'
                                    self._log(f"   Text outside bubbles INCLUDED (as free text): '{region.text[:30]}...'", "debug")
                            except Exception:
                                # Default to free text if check fails
                                region.bubble_type = 'free_text'
                        else:
                            region.should_inpaint = False
                            self._log(f"   Text outside bubbles EXCLUDED (Free Text unchecked): '{region.text[:30]}...'", "info")
                    else:
                        # For YOLO/auto, include all text by default
                        region.should_inpaint = True
                    
                    merged_regions.append(region)

            # Log summary
            regions_to_inpaint = sum(1 for r in merged_regions if getattr(r, 'should_inpaint', True))
            regions_to_skip = len(merged_regions) - regions_to_inpaint

            self._log(f"📊 Bubble detection complete: {len(regions)} → {len(merged_regions)} regions", "success")
            if detector_type == 'rtdetr':
                self._log(f"   {regions_to_inpaint} regions will be inpainted", "info")
                if regions_to_skip > 0:
                    self._log(f"   {regions_to_skip} regions will be preserved (Free Text unchecked)", "info")

            return merged_regions
            
        except Exception as e:
            self._log(f"❌ Bubble detection error: {str(e)}", "error")
            self._log("   Falling back to traditional merging", "warning")
            return self._merge_nearby_regions(regions)
        
    def set_full_page_context(self, enabled: bool, custom_prompt: str = None):
        """Configure full page context translation mode
        
        Args:
            enabled: Whether to translate all text regions in a single contextual request
            custom_prompt: Optional custom prompt for full page context mode
        """
        self.full_page_context_enabled = enabled
        if custom_prompt:
            self.full_page_context_prompt = custom_prompt
        
        self._log(f"📄 Full page context mode: {'ENABLED' if enabled else 'DISABLED'}")
        if enabled:
            self._log("   All text regions will be sent together for contextual translation")
        else:
            self._log("   Text regions will be translated individually")
    
    def update_text_rendering_settings(self, 
                                     bg_opacity: int = None,
                                     bg_style: str = None,
                                     bg_reduction: float = None,
                                     font_style: str = None,
                                     font_size: int = None,
                                     text_color: tuple = None,
                                     shadow_enabled: bool = None,
                                     shadow_color: tuple = None,
                                     shadow_offset_x: int = None,
                                     shadow_offset_y: int = None,
                                     shadow_blur: int = None,
                                     force_caps_lock: bool = None):  # ADD THIS PARAMETER
        """Update text rendering settings"""
        self._log("📐 Updating text rendering settings:", "info")
        
        if bg_opacity is not None:
            self.text_bg_opacity = max(0, min(255, bg_opacity))
            self._log(f"  Background opacity: {int(self.text_bg_opacity/255*100)}%", "info")
        if bg_style is not None and bg_style in ['box', 'circle', 'wrap']:
            self.text_bg_style = bg_style
            self._log(f"  Background style: {bg_style}", "info")
        if bg_reduction is not None:
            self.text_bg_reduction = max(0.5, min(2.0, bg_reduction))
            self._log(f"  Background size: {int(self.text_bg_reduction*100)}%", "info")
        if font_style is not None:
            self.selected_font_style = font_style
            font_name = os.path.basename(font_style) if font_style else 'Default'
            self._log(f"  Font: {font_name}", "info")
        if font_size is not None:
            if font_size < 0:
                # Negative value indicates multiplier mode
                self.font_size_mode = 'multiplier'
                self.font_size_multiplier = abs(font_size)
                self.custom_font_size = None  # Clear fixed size
                self._log(f"  Font size mode: Dynamic multiplier ({self.font_size_multiplier:.1f}x)", "info")
            else:
                # Positive value or 0 indicates fixed mode
                self.font_size_mode = 'fixed'
                self.custom_font_size = font_size if font_size > 0 else None
                self._log(f"  Font size mode: Fixed ({font_size if font_size > 0 else 'Auto'})", "info")
        if text_color is not None:
            self.text_color = text_color
            self._log(f"  Text color: RGB{text_color}", "info")
        if shadow_enabled is not None:
            self.shadow_enabled = shadow_enabled
            self._log(f"  Shadow: {'Enabled' if shadow_enabled else 'Disabled'}", "info")
        if shadow_color is not None:
            self.shadow_color = shadow_color
            self._log(f"  Shadow color: RGB{shadow_color}", "info")
        if shadow_offset_x is not None:
            self.shadow_offset_x = shadow_offset_x
        if shadow_offset_y is not None:
            self.shadow_offset_y = shadow_offset_y
        if shadow_blur is not None:
            self.shadow_blur = max(0, shadow_blur)
        if force_caps_lock is not None:  # ADD THIS BLOCK
            self.force_caps_lock = force_caps_lock
            self._log(f"  Force Caps Lock: {'Enabled' if force_caps_lock else 'Disabled'}", "info")
            
        self._log("✅ Rendering settings updated", "info")
    
    def _log(self, message: str, level: str = "info"):
        """Log message to GUI or console, and also to file logger.
        The file logger is configured in translator_gui._setup_file_logging().
        Enhanced with comprehensive stop suppression.
        """
        # Enhanced stop suppression - allow only essential stop confirmation messages
        if self._check_stop() or self.is_globally_cancelled():
            # Only allow very specific stop confirmation messages - nothing else
            essential_stop_keywords = [
                "⏹️ Translation stopped by user",
                "🧹 Cleaning up models to free RAM",
                "✅ Model cleanup complete - RAM should be freed",
                "✅ All models cleaned up - RAM freed!"
            ]
            # Suppress ALL other messages when stopped - be very restrictive
            if not any(keyword in message for keyword in essential_stop_keywords):
                return
            
        # Concise pipeline logs: keep only high-level messages and errors/warnings
        if getattr(self, 'concise_logs', False):
            if level in ("error", "warning"):
                pass
            else:
                keep_prefixes = (
                    # Pipeline boundaries and IO
                    "📷 STARTING", "📁 Input", "📁 Output",
                    # Step markers
                    "📍 [STEP",
                    # Step 1 essentials
                    "🔍 Detecting text regions",  # start of detection on file
                    "📄 Detected",                # format detected
                    "Using OCR provider:",       # provider line
                    "Using Azure Read API",      # azure-specific run mode
                    "⚠️ Converting image to PNG", # azure PNG compatibility
                    "🤖 Using AI bubble detection", # BD merge mode
                    "🤖 Using RTEDR_onnx",         # selected BD
                    "✅ Detected",                # detected N regions after merging
                    # Detectors/inpainter readiness
                    "🤖 Using bubble detector", "🎨 Using local inpainter",
                    # Step 2: key actions
                    "🔀 Running",  # Running translation and inpainting concurrently
                    "📄 Using FULL PAGE CONTEXT",  # Explicit mode notice
                    "📄 Full page context mode",   # Alternate phrasing
                    "📄 Full page context translation",  # Start/summary
                    "🎭 Creating text mask", "📊 Mask breakdown", "📏 Applying",
                    "🎨 Inpainting", "🧽 Using local inpainting",
                    # Detection and summary
                    "📊 Bubble detection complete", "✅ Detection complete",
                    # Mapping/translation summary
                    "📊 Mapping", "📊 Full page context translation complete",
                    # Rendering
                    "✍️ Rendering", "✅ ENHANCED text rendering complete",
                    # Output and final summary
                    "💾 Saved output", "✅ TRANSLATION PIPELINE COMPLETE",
                    "📊 Translation Summary", "✅ Successful", "❌ Failed",
                    # Cleanup
                    "🔑 Auto cleanup", "🔑 Translator instance preserved"
                )
                _msg = message.lstrip() if isinstance(message, str) else message
                if not any(_msg.startswith(p) for p in keep_prefixes):
                    return
        
        # In batch mode, only log important messages
        if self.batch_mode:
            # Skip verbose/debug messages in batch mode
            if level == "debug" or "DEBUG:" in message:
                return
            # Skip repetitive messages
            if any(skip in message for skip in [
                "Using vertex-based", "Using", "Applying", "Font size", 
                "Region", "Found text", "Style:"
            ]):
                return
        
        # Send to GUI if available
        if self.log_callback:
            try:
                self.log_callback(message, level)
            except Exception:
                # Fall back to print if GUI callback fails
                print(message)
        else:
            print(message)
        
        # Always record to the Python logger (file)
        try:
            _logger = logging.getLogger(__name__)
            if level == "error":
                _logger.error(message)
            elif level == "warning":
                _logger.warning(message)
            elif level == "debug":
                _logger.debug(message)
            else:
                # Map custom levels like 'success' to INFO
                _logger.info(message)
        except Exception:
            pass

    def _is_primarily_english(self, text: str) -> bool:
        """Heuristic: treat text as English if it has no CJK and a high ASCII ratio.
        Conservative by default to avoid dropping legitimate content.
        Tunable via manga_settings.ocr:
          - english_exclude_threshold (float, default 0.70)
          - english_exclude_min_chars (int, default 4)
          - english_exclude_short_tokens (bool, default False)
        """
        if not text:
            return False
        
        # Pull tuning knobs from settings (with safe defaults)
        ocr_settings = {}
        try:
            ocr_settings = self.main_gui.config.get('manga_settings', {}).get('ocr', {})
        except Exception:
            pass
        threshold = float(ocr_settings.get('english_exclude_threshold', 0.70))
        min_chars = int(ocr_settings.get('english_exclude_min_chars', 4))
        exclude_short = bool(ocr_settings.get('english_exclude_short_tokens', False))
        
        # 1) If text contains any CJK or full-width characters, do NOT treat as English
        has_cjk = any(
            '\u4e00' <= char <= '\u9fff' or  # Chinese
            '\u3040' <= char <= '\u309f' or  # Hiragana  
            '\u30a0' <= char <= '\u30ff' or  # Katakana
            '\uac00' <= char <= '\ud7af' or  # Korean
            '\uff00' <= char <= '\uffef'     # Full-width characters
            for char in text
        )
        if has_cjk:
            return False
        
        text_stripped = text.strip()
        non_space_len = sum(1 for c in text_stripped if not c.isspace())
        
        # 2) By default, do not exclude very short tokens to avoid losing interjections like "Ah", "Eh?", etc.
        if not exclude_short and non_space_len < max(1, min_chars):
            return False
        
        # Optional legacy behavior: aggressively drop very short pure-ASCII tokens
        if exclude_short:
            if len(text_stripped) == 1 and text_stripped.isalpha() and ord(text_stripped) < 128:
                self._log(f"   Excluding single English letter: '{text_stripped}'", "debug")
                return True
            if len(text_stripped) <= 3:
                ascii_letters = sum(1 for char in text_stripped if char.isalpha() and ord(char) < 128)
                if ascii_letters >= len(text_stripped) * 0.5:
                    self._log(f"   Excluding short English text: '{text_stripped}'", "debug")
                    return True
        
        # 3) Compute ASCII ratio (exclude spaces)
        ascii_chars = sum(1 for char in text if 33 <= ord(char) <= 126)
        total_chars = sum(1 for char in text if not char.isspace())
        if total_chars == 0:
            return False
        ratio = ascii_chars / total_chars
        
        if ratio > threshold:
            self._log(f"   Excluding English text ({ratio:.0%} ASCII, threshold {threshold:.0%}, len={non_space_len}): '{text[:30]}...'", "debug")
            return True
        return False

    def _load_bubble_detector(self, ocr_settings, image_path):
        """Load bubble detector with appropriate model based on settings
        
        Returns:
            dict: Detection results or None if failed
        """
        detector_type = ocr_settings.get('detector_type', 'rtdetr_onnx')
        model_path = ocr_settings.get('bubble_model_path', '')
        confidence = ocr_settings.get('bubble_confidence', 0.3)
        
        bd = self._get_thread_bubble_detector()
        
        if detector_type == 'rtdetr_onnx' or 'RTEDR_onnx' in str(detector_type):
            # Load RT-DETR ONNX model
            if bd.load_rtdetr_onnx_model(model_id=ocr_settings.get('rtdetr_model_url') or model_path):
                return bd.detect_with_rtdetr_onnx(
                    image_path=image_path,
                    confidence=ocr_settings.get('rtdetr_confidence', confidence),
                    return_all_bubbles=False
                )
        elif detector_type == 'rtdetr' or 'RT-DETR' in str(detector_type):
            # Load RT-DETR (PyTorch) model
            if bd.load_rtdetr_model(model_id=ocr_settings.get('rtdetr_model_url') or model_path):
                return bd.detect_with_rtdetr(
                    image_path=image_path,
                    confidence=ocr_settings.get('rtdetr_confidence', confidence),
                    return_all_bubbles=False
                )
        elif detector_type == 'custom':
            # Custom model - try to determine type from path
            custom_path = ocr_settings.get('custom_model_path', model_path)
            if 'rtdetr' in custom_path.lower():
                # Custom RT-DETR model
                if bd.load_rtdetr_model(model_id=custom_path):
                    return bd.detect_with_rtdetr(
                        image_path=image_path,
                        confidence=confidence,
                        return_all_bubbles=False
                    )
            else:
                # Assume YOLO format for other custom models
                if custom_path and bd.load_model(custom_path):
                    detections = bd.detect_bubbles(
                        image_path,
                        confidence=confidence
                    )
                    return {
                        'text_bubbles': detections if detections else [],
                        'text_free': [],
                        'bubbles': []
                    }
        else:
            # Standard YOLO model
            if model_path and bd.load_model(model_path):
                detections = bd.detect_bubbles(
                    image_path,
                    confidence=confidence
                )
                return {
                    'text_bubbles': detections if detections else [],
                    'text_free': [],
                    'bubbles': []
                }
        return None
            
    def _ensure_google_client(self):
        try:
            if getattr(self, 'vision_client', None) is None:
                from google.cloud import vision
                google_path = self.ocr_config.get('google_credentials_path') if hasattr(self, 'ocr_config') else None
                if google_path:
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_path
                self.vision_client = vision.ImageAnnotatorClient()
                self._log("✅ Reinitialized Google Vision client", "debug")
        except Exception as e:
            self._log(f"❌ Failed to initialize Google Vision client: {e}", "error")

    def _ensure_azure_client(self):
        try:
            if getattr(self, 'vision_client', None) is None:
                from azure.cognitiveservices.vision.computervision import ComputerVisionClient
                from msrest.authentication import CognitiveServicesCredentials
                key = None
                endpoint = None
                try:
                    key = (self.ocr_config or {}).get('azure_key')
                    endpoint = (self.ocr_config or {}).get('azure_endpoint')
                except Exception:
                    pass
                if not key:
                    key = self.main_gui.config.get('azure_vision_key', '') if hasattr(self, 'main_gui') else None
                if not endpoint:
                    endpoint = self.main_gui.config.get('azure_vision_endpoint', '') if hasattr(self, 'main_gui') else None
                if not key or not endpoint:
                    raise ValueError("Azure credentials missing for client init")
                self.vision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))
                self._log("✅ Reinitialized Azure Computer Vision client", "debug")
        except Exception as e:
            self._log(f"❌ Failed to initialize Azure CV client: {e}", "error")

    def detect_text_regions(self, image_path: str) -> List[TextRegion]:
        """Detect text regions using configured OCR provider"""
        # Reduce logging in batch mode
        if not self.batch_mode:
            self._log(f"🔍 Detecting text regions in: {os.path.basename(image_path)}")
            self._log(f"   Using OCR provider: {self.ocr_provider.upper()}")
        else:
            # Only show batch progress if batch_current is set properly
            if hasattr(self, 'batch_current') and hasattr(self, 'batch_size'):
                self._log(f"🔍 [{self.batch_current}/{self.batch_size}] {os.path.basename(image_path)}")
            else:
                self._log(f"🔍 Detecting text: {os.path.basename(image_path)}")
        
        try:
            # ============================================================
            # CRITICAL: FORCE CLEAR ALL TEXT-RELATED CACHES
            # This MUST happen for EVERY image to prevent text contamination
            # NO EXCEPTIONS - batch mode or not, ALL caches get cleared
            # ============================================================
            
            # 1. Clear OCR ROI cache (prevents text from previous images leaking)
            # THREAD-SAFE: Use lock to prevent race conditions in parallel panel translation
            if hasattr(self, 'ocr_roi_cache'):
                with self._cache_lock:
                    self.ocr_roi_cache.clear()
                self._log("🧹 Cleared OCR ROI cache", "debug")
            
            # 2. Clear OCR manager caches (multiple potential cache locations)
            if hasattr(self, 'ocr_manager') and self.ocr_manager:
                # Clear last_results (can contain text from previous image)
                if hasattr(self.ocr_manager, 'last_results'):
                    self.ocr_manager.last_results = None
                # Clear generic cache
                if hasattr(self.ocr_manager, 'cache'):
                    self.ocr_manager.cache.clear()
                # Clear provider-level caches
                if hasattr(self.ocr_manager, 'providers'):
                    for provider_name, provider in self.ocr_manager.providers.items():
                        if hasattr(provider, 'last_results'):
                            provider.last_results = None
                        if hasattr(provider, 'cache'):
                            provider.cache.clear()
                self._log("🧹 Cleared OCR manager caches", "debug")
            
            # 3. Clear bubble detector cache (can contain text region info)
            if hasattr(self, 'bubble_detector') and self.bubble_detector:
                if hasattr(self.bubble_detector, 'last_detections'):
                    self.bubble_detector.last_detections = None
                if hasattr(self.bubble_detector, 'cache'):
                    self.bubble_detector.cache.clear()
                self._log("🧹 Cleared bubble detector cache", "debug")
            
            # Get manga settings from main_gui config
            manga_settings = self.main_gui.config.get('manga_settings', {})
            preprocessing = manga_settings.get('preprocessing', {})
            ocr_settings = manga_settings.get('ocr', {})
            
            # Get text filtering settings
            min_text_length = ocr_settings.get('min_text_length', 2)
            exclude_english = ocr_settings.get('exclude_english_text', True)
            confidence_threshold = ocr_settings.get('confidence_threshold', 0.1)
            
            # Load and preprocess image if enabled
            if preprocessing.get('enabled', False):
                self._log("📐 Preprocessing enabled - enhancing image quality")
                processed_image_data = self._preprocess_image(image_path, preprocessing)
            else:
                # Read image with optional compression (separate from preprocessing)
                try:
                    comp_cfg = (self.main_gui.config.get('manga_settings', {}) or {}).get('compression', {})
                    if comp_cfg.get('enabled', False):
                        processed_image_data = self._load_image_with_compression_only(image_path, comp_cfg)
                    else:
                        with open(image_path, 'rb') as image_file:
                            processed_image_data = image_file.read()
                except Exception:
                    with open(image_path, 'rb') as image_file:
                        processed_image_data = image_file.read()
            
            # Compute per-image hash for caching (based on uploaded bytes)
            # CRITICAL FIX #1: Never allow None page_hash to prevent cache key collisions
            try:
                import hashlib
                page_hash = hashlib.sha1(processed_image_data).hexdigest()
                
                # CRITICAL: Never allow None page_hash
                if page_hash is None:
                    # Fallback: use image path + timestamp for uniqueness
                    import time
                    import uuid
                    page_hash = hashlib.sha1(
                        f"{image_path}_{time.time()}_{uuid.uuid4()}".encode()
                    ).hexdigest()
                    self._log("⚠️ Using fallback page hash for cache isolation", "warning")
                
                # CRITICAL: If image hash changed, force clear ROI cache
                # THREAD-SAFE: Use lock for parallel panel translation
                if hasattr(self, '_current_image_hash') and self._current_image_hash != page_hash:
                    if hasattr(self, 'ocr_roi_cache'):
                        with self._cache_lock:
                            self.ocr_roi_cache.clear()
                        self._log("🧹 Image changed - cleared ROI cache", "debug")
                self._current_image_hash = page_hash
            except Exception as e:
                # Emergency fallback - never let page_hash be None
                import uuid
                page_hash = str(uuid.uuid4())
                self._current_image_hash = page_hash
                self._log(f"⚠️ Page hash generation failed: {e}, using UUID fallback", "error")
            
            regions = []
            
            # Route to appropriate provider
            if self.ocr_provider == 'google':
                # === GOOGLE CLOUD VISION ===
                # Ensure client exists (it might have been cleaned up between runs)
                try:
                    self._ensure_google_client()
                except Exception:
                    pass
                
                # Check if we should use RT-DETR for text region detection (NEW FEATURE)
                # IMPORTANT: bubble_detection_enabled should default to True for optimal detection
                if ocr_settings.get('bubble_detection_enabled', True) and ocr_settings.get('use_rtdetr_for_ocr_regions', True):
                    self._log("🎯 Using RT-DETR to guide Google Cloud Vision OCR")
                    
                    # Run RT-DETR to detect text regions first
                    _ = self._get_thread_bubble_detector()
                    rtdetr_detections = self._load_bubble_detector(ocr_settings, image_path)
                    
                    if rtdetr_detections:
                        # Collect all text-containing regions WITH TYPE TRACKING
                        all_regions = []
                        # Track region type to assign bubble_type later
                        region_types = {}
                        idx = 0
                        if 'text_bubbles' in rtdetr_detections:
                            for bbox in rtdetr_detections.get('text_bubbles', []):
                                all_regions.append(bbox)
                                region_types[idx] = 'text_bubble'
                                idx += 1
                        if 'text_free' in rtdetr_detections:
                            for bbox in rtdetr_detections.get('text_free', []):
                                all_regions.append(bbox)
                                region_types[idx] = 'free_text'
                                idx += 1
                        
                        if all_regions:
                            self._log(f"📊 RT-DETR detected {len(all_regions)} text regions, OCR-ing each with Google Vision")
                            
                            # Load image for cropping
                            import cv2
                            cv_image = cv2.imread(image_path)
                            if cv_image is None:
                                self._log("⚠️ Failed to load image, falling back to full-page OCR", "warning")
                            else:
                                # Define worker function for concurrent OCR
                                def ocr_region_google(region_data):
                                    i, region_idx, x, y, w, h = region_data
                                    try:
                                        # RATE LIMITING: Add small delay to avoid potential rate limits
                                        # Google has high limits (1,800/min paid tier) but being conservative
                                        import time
                                        import random
                                        time.sleep(0.1 + random.random() * 0.2)  # 0.1-0.3s random delay
                                        
                                        # Crop region
                                        cropped = self._safe_crop_region(cv_image, x, y, w, h)
                                        if cropped is None:
                                            return None
                                        
                                        # Validate and resize crop if needed (Google Vision requires minimum dimensions)
                                        h_crop, w_crop = cropped.shape[:2]
                                        MIN_SIZE = 50  # Minimum dimension (increased from 10 for better OCR)
                                        MIN_AREA = 2500  # Minimum area (50x50)
                                        
                                        if h_crop < MIN_SIZE or w_crop < MIN_SIZE or h_crop * w_crop < MIN_AREA:
                                            # Region too small - try to resize it
                                            scale_w = MIN_SIZE / w_crop if w_crop < MIN_SIZE else 1.0
                                            scale_h = MIN_SIZE / h_crop if h_crop < MIN_SIZE else 1.0
                                            scale = max(scale_w, scale_h)
                                            
                                            if scale > 1.0:
                                                new_w = int(w_crop * scale)
                                                new_h = int(h_crop * scale)
                                                cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                                                self._log(f"🔍 Region {i} resized from {w_crop}x{h_crop}px to {new_w}x{new_h}px for OCR", "debug")
                                                h_crop, w_crop = new_h, new_w
                                        
                                        # Final validation
                                        if h_crop < 10 or w_crop < 10:
                                            self._log(f"⚠️ Region {i} too small even after resize ({w_crop}x{h_crop}px), skipping", "debug")
                                            return None
                                        
                                        # Encode cropped image
                                        _, encoded = cv2.imencode('.jpg', cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                                        region_image_data = encoded.tobytes()
                                        
                                        # Create Vision API image object
                                        vision_image = vision.Image(content=region_image_data)
                                        image_context = vision.ImageContext(
                                            language_hints=ocr_settings.get('language_hints', ['ja', 'ko', 'zh'])
                                        )
                                        
                                        # Detect text in this region
                                        detection_mode = ocr_settings.get('text_detection_mode', 'document')
                                        if detection_mode == 'document':
                                            response = self.vision_client.document_text_detection(
                                                image=vision_image,
                                                image_context=image_context
                                            )
                                        else:
                                            response = self.vision_client.text_detection(
                                                image=vision_image,
                                                image_context=image_context
                                            )
                                        
                                        if response.error.message:
                                            self._log(f"⚠️ Region {i} error: {response.error.message}", "warning")
                                            return None
                                        
                                        # Extract text from this region
                                        region_text = response.full_text_annotation.text if response.full_text_annotation else ""
                                        if region_text.strip():
                                            # Clean the text
                                            region_text = self._fix_encoding_issues(region_text)
                                            region_text = self._sanitize_unicode_characters(region_text)
                                            region_text = region_text.strip()
                                            
                                            # Create TextRegion with original image coordinates
                                            region = TextRegion(
                                                text=region_text,
                                                vertices=[(x, y), (x+w, y), (x+w, y+h), (x, y+h)],
                                                bounding_box=(x, y, w, h),
                                                confidence=0.9,  # RT-DETR confidence
                                                region_type='text_block'
                                            )
                                            # Assign bubble_type from RT-DETR detection
                                            region.bubble_type = region_types.get(region_idx, 'text_bubble')
                                            if not getattr(self, 'concise_logs', False):
                                                self._log(f"✅ Region {i}/{len(all_regions)} ({region.bubble_type}): {region_text[:50]}...")
                                            return region
                                        return None
                                    
                                    except Exception as e:
                                        # Provide more detailed error info for debugging
                                        error_msg = str(e)
                                        if 'Bad Request' in error_msg or 'invalid' in error_msg.lower():
                                            self._log(f"⏭️ Skipping region {i}: Too small or invalid for Google Vision (dimensions < 10x10px or area < 100px²)", "debug")
                                        else:
                                            self._log(f"⚠️ Error OCR-ing region {i}: {e}", "warning")
                                        return None
                                
                                # Process regions concurrently with RT-DETR concurrency control
                                from concurrent.futures import ThreadPoolExecutor, as_completed
                                # Use rtdetr_max_concurrency setting (default 12) to control parallel OCR calls
                                max_workers = min(ocr_settings.get('rtdetr_max_concurrency', 12), len(all_regions))
                                
                                region_data_list = [(i+1, i, x, y, w, h) for i, (x, y, w, h) in enumerate(all_regions)]
                                
                                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                                    futures = {executor.submit(ocr_region_google, rd): rd for rd in region_data_list}
                                    for future in as_completed(futures):
                                        try:
                                            result = future.result()
                                            if result:
                                                regions.append(result)
                                        finally:
                                            # Clean up future to free memory
                                            del future
                                
                                # If we got results, sort and post-process
                                if regions:
                                    # CRITICAL: Sort regions by position (top-to-bottom, left-to-right)
                                    # Concurrent processing returns them in completion order, not detection order
                                    regions.sort(key=lambda r: (r.bounding_box[1], r.bounding_box[0]))
                                    self._log(f"✅ RT-DETR + Google Vision: {len(regions)} text regions detected (sorted by position)")
                                    
                                    # POST-PROCESS: Check for text_bubbles that overlap with free_text regions
                                    # If a text_bubble's center is within a free_text bbox, reclassify it as free_text
                                    free_text_bboxes = rtdetr_detections.get('text_free', [])
                                    if free_text_bboxes:
                                        reclassified_count = 0
                                        for region in regions:
                                            if getattr(region, 'bubble_type', None) == 'text_bubble':
                                                # Get region center
                                                x, y, w, h = region.bounding_box
                                                cx = x + w / 2
                                                cy = y + h / 2
                                                
                                                self._log(f"   Checking text_bubble '{region.text[:30]}...' at center ({cx:.0f}, {cy:.0f})", "debug")
                                                
                                                # Check if center is in any free_text bbox
                                                for bbox_idx, (fx, fy, fw, fh) in enumerate(free_text_bboxes):
                                                    in_x = fx <= cx <= fx + fw
                                                    in_y = fy <= cy <= fy + fh
                                                    self._log(f"      vs free_text bbox {bbox_idx+1}: in_x={in_x}, in_y={in_y}", "debug")
                                                    
                                                    if in_x and in_y:
                                                        # Reclassify as free text
                                                        old_type = region.bubble_type
                                                        region.bubble_type = 'free_text'
                                                        reclassified_count += 1
                                                        self._log(f"      ✅ RECLASSIFIED '{region.text[:30]}...' from {old_type} to free_text", "info")
                                                        break
                                        
                                        if reclassified_count > 0:
                                            self._log(f"🔄 Reclassified {reclassified_count} overlapping regions as free_text", "info")
                                            
                                            # MERGE: Combine free_text regions that are within the same free_text bbox
                                            # Group free_text regions by which free_text bbox they belong to
                                            free_text_groups = {}
                                            other_regions = []
                                            
                                            for region in regions:
                                                if getattr(region, 'bubble_type', None) == 'free_text':
                                                    # Find which free_text bbox this region belongs to
                                                    x, y, w, h = region.bounding_box
                                                    cx = x + w / 2
                                                    cy = y + h / 2
                                                    
                                                    for bbox_idx, (fx, fy, fw, fh) in enumerate(free_text_bboxes):
                                                        if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                                                            if bbox_idx not in free_text_groups:
                                                                free_text_groups[bbox_idx] = []
                                                            free_text_groups[bbox_idx].append(region)
                                                            break
                                                    else:
                                                        # Free text region not in any bbox (shouldn't happen, but handle it)
                                                        other_regions.append(region)
                                                else:
                                                    other_regions.append(region)
                                            
                                            # Merge each group of free_text regions
                                            merged_free_text = []
                                            for bbox_idx, group in free_text_groups.items():
                                                if len(group) > 1:
                                                    # Merge multiple free text regions in same bbox
                                                    merged_text = " ".join(r.text for r in group)
                                                    
                                                    min_x = min(r.bounding_box[0] for r in group)
                                                    min_y = min(r.bounding_box[1] for r in group)
                                                    max_x = max(r.bounding_box[0] + r.bounding_box[2] for r in group)
                                                    max_y = max(r.bounding_box[1] + r.bounding_box[3] for r in group)
                                                    
                                                    all_vertices = []
                                                    for r in group:
                                                        if hasattr(r, 'vertices') and r.vertices:
                                                            all_vertices.extend(r.vertices)
                                                    
                                                    if not all_vertices:
                                                        all_vertices = [
                                                            (min_x, min_y),
                                                            (max_x, min_y),
                                                            (max_x, max_y),
                                                            (min_x, max_y)
                                                        ]
                                                    
                                                    merged_region = TextRegion(
                                                        text=merged_text,
                                                        vertices=all_vertices,
                                                        bounding_box=(min_x, min_y, max_x - min_x, max_y - min_y),
                                                        confidence=0.95,
                                                        region_type='text_block'
                                                    )
                                                    merged_region.bubble_type = 'free_text'
                                                    merged_region.should_inpaint = True
                                                    merged_free_text.append(merged_region)
                                                    self._log(f"🔀 Merged {len(group)} free_text regions into one: '{merged_text[:50]}...'", "debug")
                                                else:
                                                    # Single region, keep as-is
                                                    merged_free_text.extend(group)
                                            
                                            # Combine all regions
                                            regions = other_regions + merged_free_text
                                            self._log(f"✅ Final: {len(regions)} regions after reclassification and merging", "info")
                                    
                                    # Skip merging section and return directly
                                    return regions
                                else:
                                    self._log("⚠️ No text found in RT-DETR regions, falling back to full-page OCR", "warning")
                
                # If bubble detection is enabled and batch variables suggest batching, do ROI-based batched OCR
                try:
                    use_roi_locality = ocr_settings.get('bubble_detection_enabled', False) and ocr_settings.get('roi_locality_enabled', False)
                    # Determine OCR batching enable
                    if 'ocr_batch_enabled' in ocr_settings:
                        ocr_batch_enabled = bool(ocr_settings.get('ocr_batch_enabled'))
                    else:
                        ocr_batch_enabled = (os.getenv('BATCH_OCR', '0') == '1') or (os.getenv('BATCH_TRANSLATION', '0') == '1') or getattr(self, 'batch_mode', False)
                    # Determine OCR batch size
                    bs = int(ocr_settings.get('ocr_batch_size') or 0)
                    if bs <= 0:
                        bs = int(os.getenv('OCR_BATCH_SIZE', '0') or 0)
                    if bs <= 0:
                        bs = int(os.getenv('BATCH_SIZE', str(getattr(self, 'batch_size', 1))) or 1)
                    ocr_batch_size = max(1, bs)
                except Exception:
                    use_roi_locality = False
                    ocr_batch_enabled = False
                    ocr_batch_size = 1
                if use_roi_locality and (ocr_batch_enabled or ocr_batch_size > 1):
                    rois = self._prepare_ocr_rois_from_bubbles(image_path, ocr_settings, preprocessing, page_hash)
                    if rois:
                        # Determine concurrency for Google: OCR_MAX_CONCURRENCY env or min(BATCH_SIZE,2)
                        try:
                            max_cc = int(ocr_settings.get('ocr_max_concurrency') or 0)
                            if max_cc <= 0:
                                max_cc = int(os.getenv('OCR_MAX_CONCURRENCY', '0') or 0)
                            if max_cc <= 0:
                                max_cc = min(max(1, ocr_batch_size), 2)
                        except Exception:
                            max_cc = min(max(1, ocr_batch_size), 2)
                        regions = self._google_ocr_rois_batched(rois, ocr_settings, max(1, ocr_batch_size), max_cc, page_hash)
                        self._log(f"✅ Google OCR batched over {len(rois)} ROIs → {len(regions)} regions (cc={max_cc})", "info")
                        
                        # Force garbage collection after concurrent OCR to reduce memory spikes
                        try:
                            import gc
                            gc.collect()
                        except Exception:
                            pass
                        
                        return regions

                # Start local inpainter preload while Google OCR runs (background; multiple if panel-parallel)
                try:
                    if not getattr(self, 'skip_inpainting', False) and not getattr(self, 'use_cloud_inpainting', False):
                        already_loaded, _lm = self._is_local_inpainter_loaded()
                        if not already_loaded:
                            import threading as _threading
                            local_method = (self.manga_settings.get('inpainting', {}) or {}).get('local_method', 'anime')
                            model_path = self.main_gui.config.get(f'manga_{local_method}_model_path', '') if hasattr(self, 'main_gui') else ''
                            adv = self.main_gui.config.get('manga_settings', {}).get('advanced', {}) if hasattr(self, 'main_gui') else {}
                            # Determine desired instances from panel-parallel settings
                            desired = 1
                            if adv.get('parallel_panel_translation', False):
                                try:
                                    desired = max(1, int(adv.get('panel_max_workers', 2)))
                                except Exception:
                                    desired = 2
                            # Honor advanced toggle for panel-local preload; for non-panel (desired==1) always allow
                            allow = True if desired == 1 else bool(adv.get('preload_local_inpainting_for_panels', True))
                            if allow:
                                self._inpaint_preload_event = _threading.Event()
                                def _preload_inp_many():
                                    try:
                                        self.preload_local_inpainters_concurrent(local_method, model_path, desired)
                                    finally:
                                        try:
                                            self._inpaint_preload_event.set()
                                        except Exception:
                                            pass
                                _threading.Thread(target=_preload_inp_many, name="InpaintPreload@GoogleOCR", daemon=True).start()
                except Exception:
                    pass

                # Create Vision API image object (full-page fallback)
                image = vision.Image(content=processed_image_data)
                
                # Build image context with all parameters
                image_context = vision.ImageContext(
                    language_hints=ocr_settings.get('language_hints', ['ja', 'ko', 'zh'])
                )
                
                # Add text detection params if available in your API version
                if hasattr(vision, 'TextDetectionParams'):
                    image_context.text_detection_params = vision.TextDetectionParams(
                        enable_text_detection_confidence_score=True
                    )
                
                # Configure text detection based on settings
                detection_mode = ocr_settings.get('text_detection_mode', 'document')
                
                if detection_mode == 'document':
                    response = self.vision_client.document_text_detection(
                        image=image,
                        image_context=image_context
                    )
                else:
                    response = self.vision_client.text_detection(
                        image=image,
                        image_context=image_context
                    )
                
                if response.error.message:
                    raise Exception(f"Cloud Vision API error: {response.error.message}")
                
                # Process each page (usually just one for manga)
                for page in response.full_text_annotation.pages:
                    for block in page.blocks:
                        # Extract text first to check if it's worth processing
                        block_text = ""
                        total_confidence = 0.0
                        word_count = 0
                        
                        for paragraph in block.paragraphs:
                            for word in paragraph.words:
                                # Get word-level confidence (more reliable than block level)
                                word_confidence = getattr(word, 'confidence', 0.0)  # Default to 0 if not available
                                word_text = ''.join([symbol.text for symbol in word.symbols])
                                
                                # Only include words above threshold
                                if word_confidence >= confidence_threshold:
                                    block_text += word_text + " "
                                    total_confidence += word_confidence
                                    word_count += 1
                                else:
                                    if not getattr(self, 'concise_logs', False):
                                        self._log(f"   Skipping low confidence word ({word_confidence:.2f}): {word_text}")
                        
                        block_text = block_text.strip()
                        
                        # CLEAN ORIGINAL OCR TEXT - Fix cube characters and encoding issues
                        original_text = block_text
                        block_text = self._fix_encoding_issues(block_text)
                        block_text = self._sanitize_unicode_characters(block_text)
                        
                        # Log cleaning if changes were made
                        if block_text != original_text:
                            self._log(f"🧹 Cleaned OCR text: '{original_text[:30]}...' → '{block_text[:30]}...'", "debug")
                        
                        # TEXT FILTERING SECTION
                        # Skip if text is too short (after cleaning)
                        if len(block_text.strip()) < min_text_length:
                            if not getattr(self, 'concise_logs', False):
                                self._log(f"   Skipping short text ({len(block_text)} chars): {block_text}")
                            continue
                        
                        # Skip if primarily English and exclude_english is enabled
                        if exclude_english and self._is_primarily_english(block_text):
                            if not getattr(self, 'concise_logs', False):
                                self._log(f"   Skipping English text: {block_text[:50]}...")
                            continue
                        
                        # Skip if no confident words found
                        if word_count == 0 or not block_text:
                            if not getattr(self, 'concise_logs', False):
                                self._log(f"   Skipping block - no words above threshold {confidence_threshold}")
                            continue
                        
                        # Calculate average confidence for the block
                        avg_confidence = total_confidence / word_count if word_count > 0 else 0.0
                        
                        # Extract vertices and create region
                        vertices = [(v.x, v.y) for v in block.bounding_box.vertices]
                        
                        # Calculate bounding box
                        xs = [v[0] for v in vertices]
                        ys = [v[1] for v in vertices]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        
                        region = TextRegion(
                            text=block_text,
                            vertices=vertices,
                            bounding_box=(x_min, y_min, x_max - x_min, y_max - y_min),
                            confidence=avg_confidence,  # Use average confidence
                            region_type='text_block'
                        )
                        regions.append(region)
                        if not getattr(self, 'concise_logs', False):
                            self._log(f"   Found text region ({avg_confidence:.2f}): {block_text[:50]}...")
                        
            elif self.ocr_provider == 'azure':
                # === AZURE COMPUTER VISION ===
                # Ensure client exists (it might have been cleaned up between runs)
                try:
                    self._ensure_azure_client()
                except Exception:
                    pass
                import io
                import time
                from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
                
                # Check if we should use RT-DETR for text region detection (NEW FEATURE)
                if ocr_settings.get('bubble_detection_enabled', False) and ocr_settings.get('use_rtdetr_for_ocr_regions', True):
                    self._log("🎯 Using RT-DETR to guide Azure Computer Vision OCR")
                    
                    # Run RT-DETR to detect text regions first
                    _ = self._get_thread_bubble_detector()
                    rtdetr_detections = self._load_bubble_detector(ocr_settings, image_path)
                    
                    if rtdetr_detections:
                        # Collect all text-containing regions WITH TYPE TRACKING
                        all_regions = []
                        # Track region type to assign bubble_type later
                        region_types = {}
                        idx = 0
                        if 'text_bubbles' in rtdetr_detections:
                            for bbox in rtdetr_detections.get('text_bubbles', []):
                                all_regions.append(bbox)
                                region_types[idx] = 'text_bubble'
                                idx += 1
                        if 'text_free' in rtdetr_detections:
                            for bbox in rtdetr_detections.get('text_free', []):
                                all_regions.append(bbox)
                                region_types[idx] = 'free_text'
                                idx += 1
                        
                        if all_regions:
                            self._log(f"📊 RT-DETR detected {len(all_regions)} text regions, OCR-ing each with Azure Vision")
                            
                            # Load image for cropping
                            import cv2
                            cv_image = cv2.imread(image_path)
                            if cv_image is None:
                                self._log("⚠️ Failed to load image, falling back to full-page OCR", "warning")
                            else:
                                ocr_results = []
                                
                                # Get Azure settings
                                azure_reading_order = ocr_settings.get('azure_reading_order', 'natural')
                                azure_model_version = ocr_settings.get('azure_model_version', 'latest')
                                azure_max_wait = ocr_settings.get('azure_max_wait', 60)
                                azure_poll_interval = ocr_settings.get('azure_poll_interval', 1.0)
                                
                                # Define worker function for concurrent OCR
                                def ocr_region_azure(region_data):
                                    i, region_idx, x, y, w, h = region_data
                                    try:
                                        # Crop region
                                        cropped = self._safe_crop_region(cv_image, x, y, w, h)
                                        if cropped is None:
                                            return None
                                        
                                        # Validate and resize crop if needed (Azure Vision requires minimum dimensions)
                                        h_crop, w_crop = cropped.shape[:2]
                                        MIN_SIZE = 50  # Minimum dimension (Azure requirement)
                                        MIN_AREA = 2500  # Minimum area (50x50)
                                        
                                        if h_crop < MIN_SIZE or w_crop < MIN_SIZE or h_crop * w_crop < MIN_AREA:
                                            # Region too small - try to resize it
                                            scale_w = MIN_SIZE / w_crop if w_crop < MIN_SIZE else 1.0
                                            scale_h = MIN_SIZE / h_crop if h_crop < MIN_SIZE else 1.0
                                            scale = max(scale_w, scale_h)
                                            
                                            if scale > 1.0:
                                                new_w = int(w_crop * scale)
                                                new_h = int(h_crop * scale)
                                                cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                                                self._log(f"🔍 Region {i} resized from {w_crop}x{h_crop}px to {new_w}x{new_h}px for Azure OCR", "debug")
                                                h_crop, w_crop = new_h, new_w
                                        
                                        # Final validation
                                        if h_crop < 10 or w_crop < 10:
                                            self._log(f"⚠️ Region {i} too small even after resize ({w_crop}x{h_crop}px), skipping", "debug")
                                            return None
                                        
                                        # RATE LIMITING: Add delay between Azure API calls to avoid "Too Many Requests"
                                        # Azure Free tier: 20 calls/minute = 1 call per 3 seconds
                                        # Azure Standard tier: Higher limits but still needs throttling
                                        import time
                                        import random
                                        # Stagger requests with randomized delay (0.1-0.3 seconds)
                                        time.sleep(0.1 + random.random() * 0.2)  # 0.1-0.3s random delay
                                        
                                        # Encode cropped image
                                        _, encoded = cv2.imencode('.jpg', cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                                        region_image_bytes = encoded.tobytes()
                                        
                                        # Call Azure Read API
                                        read_response = self.vision_client.read_in_stream(
                                            io.BytesIO(region_image_bytes),
                                            language=ocr_settings.get('language_hints', ['ja'])[0] if ocr_settings.get('language_hints') else 'ja',
                                            model_version=azure_model_version,
                                            reading_order=azure_reading_order,
                                            raw=True
                                        )
                                        
                                        # Get operation location
                                        operation_location = read_response.headers['Operation-Location']
                                        operation_id = operation_location.split('/')[-1]
                                        
                                        # Poll for result
                                        start_time = time.time()
                                        while True:
                                            result = self.vision_client.get_read_result(operation_id)
                                            if result.status not in [OperationStatusCodes.not_started, OperationStatusCodes.running]:
                                                break
                                            if time.time() - start_time > azure_max_wait:
                                                self._log(f"⚠️ Azure timeout for region {i}", "warning")
                                                break
                                            time.sleep(azure_poll_interval)
                                        
                                        if result.status == OperationStatusCodes.succeeded:
                                            # Extract text from result
                                            region_text = ""
                                            for text_result in result.analyze_result.read_results:
                                                for line in text_result.lines:
                                                    region_text += line.text + "\n"
                                            
                                            region_text = region_text.strip()
                                            if region_text:
                                                # Clean the text
                                                region_text = self._fix_encoding_issues(region_text)
                                                region_text = self._sanitize_unicode_characters(region_text)
                                                
                                                # Create TextRegion with original image coordinates
                                                region = TextRegion(
                                                    text=region_text,
                                                    vertices=[(x, y), (x+w, y), (x+w, y+h), (x, y+h)],
                                                    bounding_box=(x, y, w, h),
                                                    confidence=0.9,  # RT-DETR confidence
                                                    region_type='text_block'
                                                )
                                                # Assign bubble_type from RT-DETR detection
                                                region.bubble_type = region_types.get(region_idx, 'text_bubble')
                                                if not getattr(self, 'concise_logs', False):
                                                    self._log(f"✅ Region {i}/{len(all_regions)} ({region.bubble_type}): {region_text[:50]}...")
                                                return region
                                        return None
                                    
                                    except Exception as e:
                                        # Provide more detailed error info for debugging
                                        error_msg = str(e)
                                        if 'Bad Request' in error_msg or 'invalid' in error_msg.lower() or 'Too Many Requests' in error_msg:
                                            if 'Too Many Requests' in error_msg:
                                                self._log(f"⏸️ Region {i}: Azure rate limit hit, consider increasing delays", "warning")
                                            else:
                                                self._log(f"⏭️ Skipping region {i}: Too small or invalid for Azure Vision", "debug")
                                        else:
                                            self._log(f"⚠️ Error OCR-ing region {i}: {e}", "warning")
                                        return None
                                
                                # Process regions concurrently with RT-DETR concurrency control
                                from concurrent.futures import ThreadPoolExecutor, as_completed
                                # Use rtdetr_max_concurrency setting (default 12)
                                # Note: Rate limiting is handled via 0.1-0.3s delays per request
                                max_workers = min(ocr_settings.get('rtdetr_max_concurrency', 12), len(all_regions))
                                
                                region_data_list = [(i+1, i, x, y, w, h) for i, (x, y, w, h) in enumerate(all_regions)]
                                
                                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                                    futures = {executor.submit(ocr_region_azure, rd): rd for rd in region_data_list}
                                    for future in as_completed(futures):
                                        try:
                                            result = future.result()
                                            if result:
                                                regions.append(result)
                                        finally:
                                            # Clean up future to free memory
                                            del future
                                
                                # If we got results, sort and post-process
                                if regions:
                                    # CRITICAL: Sort regions by position (top-to-bottom, left-to-right)
                                    # Concurrent processing returns them in completion order, not detection order
                                    regions.sort(key=lambda r: (r.bounding_box[1], r.bounding_box[0]))
                                    self._log(f"✅ RT-DETR + Azure Vision: {len(regions)} text regions detected (sorted by position)")
                                    
                                    # POST-PROCESS: Check for text_bubbles that overlap with free_text regions
                                    # If a text_bubble's center is within a free_text bbox, reclassify it as free_text
                                    free_text_bboxes = rtdetr_detections.get('text_free', [])
                                    
                                    # DEBUG: Log what we have
                                    self._log(f"🔍 POST-PROCESS: Found {len(free_text_bboxes)} free_text bboxes from RT-DETR", "debug")
                                    for idx, (fx, fy, fw, fh) in enumerate(free_text_bboxes):
                                        self._log(f"   Free text bbox {idx+1}: x={fx:.0f}, y={fy:.0f}, w={fw:.0f}, h={fh:.0f}", "debug")
                                    
                                    text_bubble_count = sum(1 for r in regions if getattr(r, 'bubble_type', None) == 'text_bubble')
                                    free_text_count = sum(1 for r in regions if getattr(r, 'bubble_type', None) == 'free_text')
                                    self._log(f"🔍 Before reclassification: {text_bubble_count} text_bubbles, {free_text_count} free_text", "debug")
                                    
                                    if free_text_bboxes:
                                        reclassified_count = 0
                                        for region in regions:
                                            if getattr(region, 'bubble_type', None) == 'text_bubble':
                                                # Get region center
                                                x, y, w, h = region.bounding_box
                                                cx = x + w / 2
                                                cy = y + h / 2
                                                
                                                self._log(f"   Checking text_bubble '{region.text[:30]}...' at center ({cx:.0f}, {cy:.0f})", "debug")
                                                
                                                # Check if center is in any free_text bbox
                                                for bbox_idx, (fx, fy, fw, fh) in enumerate(free_text_bboxes):
                                                    in_x = fx <= cx <= fx + fw
                                                    in_y = fy <= cy <= fy + fh
                                                    self._log(f"      vs free_text bbox {bbox_idx+1}: in_x={in_x}, in_y={in_y}", "debug")
                                                    
                                                    if in_x and in_y:
                                                        # Reclassify as free text
                                                        old_type = region.bubble_type
                                                        region.bubble_type = 'free_text'
                                                        reclassified_count += 1
                                                        self._log(f"      ✅ RECLASSIFIED '{region.text[:30]}...' from {old_type} to free_text", "info")
                                                        break
                                        
                                        if reclassified_count > 0:
                                            self._log(f"🔄 Reclassified {reclassified_count} overlapping regions as free_text", "info")
                                            
                                            # MERGE: Combine free_text regions that are within the same free_text bbox
                                            # Group free_text regions by which free_text bbox they belong to
                                            free_text_groups = {}
                                            other_regions = []
                                            
                                            for region in regions:
                                                if getattr(region, 'bubble_type', None) == 'free_text':
                                                    # Find which free_text bbox this region belongs to
                                                    x, y, w, h = region.bounding_box
                                                    cx = x + w / 2
                                                    cy = y + h / 2
                                                    
                                                    for bbox_idx, (fx, fy, fw, fh) in enumerate(free_text_bboxes):
                                                        if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                                                            if bbox_idx not in free_text_groups:
                                                                free_text_groups[bbox_idx] = []
                                                            free_text_groups[bbox_idx].append(region)
                                                            break
                                                    else:
                                                        # Free text region not in any bbox (shouldn't happen, but handle it)
                                                        other_regions.append(region)
                                                else:
                                                    other_regions.append(region)
                                            
                                            # Merge each group of free_text regions
                                            merged_free_text = []
                                            for bbox_idx, group in free_text_groups.items():
                                                if len(group) > 1:
                                                    # Merge multiple free text regions in same bbox
                                                    merged_text = " ".join(r.text for r in group)
                                                    
                                                    min_x = min(r.bounding_box[0] for r in group)
                                                    min_y = min(r.bounding_box[1] for r in group)
                                                    max_x = max(r.bounding_box[0] + r.bounding_box[2] for r in group)
                                                    max_y = max(r.bounding_box[1] + r.bounding_box[3] for r in group)
                                                    
                                                    all_vertices = []
                                                    for r in group:
                                                        if hasattr(r, 'vertices') and r.vertices:
                                                            all_vertices.extend(r.vertices)
                                                    
                                                    if not all_vertices:
                                                        all_vertices = [
                                                            (min_x, min_y),
                                                            (max_x, min_y),
                                                            (max_x, max_y),
                                                            (min_x, max_y)
                                                        ]
                                                    
                                                    merged_region = TextRegion(
                                                        text=merged_text,
                                                        vertices=all_vertices,
                                                        bounding_box=(min_x, min_y, max_x - min_x, max_y - min_y),
                                                        confidence=0.95,
                                                        region_type='text_block'
                                                    )
                                                    merged_region.bubble_type = 'free_text'
                                                    merged_region.should_inpaint = True
                                                    merged_free_text.append(merged_region)
                                                    self._log(f"🔀 Merged {len(group)} free_text regions into one: '{merged_text[:50]}...'", "debug")
                                                else:
                                                    # Single region, keep as-is
                                                    merged_free_text.extend(group)
                                            
                                            # Combine all regions
                                            regions = other_regions + merged_free_text
                                            self._log(f"✅ Final: {len(regions)} regions after reclassification and merging", "info")
                                    
                                    # Skip merging section and return directly
                                    return regions
                                else:
                                    self._log("⚠️ No text found in RT-DETR regions, falling back to full-page OCR", "warning")
                
                # ROI-based concurrent OCR when bubble detection is enabled and batching is requested
                try:
                    use_roi_locality = ocr_settings.get('bubble_detection_enabled', False) and ocr_settings.get('roi_locality_enabled', False)
                    if 'ocr_batch_enabled' in ocr_settings:
                        ocr_batch_enabled = bool(ocr_settings.get('ocr_batch_enabled'))
                    else:
                        ocr_batch_enabled = (os.getenv('BATCH_OCR', '0') == '1') or (os.getenv('BATCH_TRANSLATION', '0') == '1') or getattr(self, 'batch_mode', False)
                    bs = int(ocr_settings.get('ocr_batch_size') or 0)
                    if bs <= 0:
                        bs = int(os.getenv('OCR_BATCH_SIZE', '0') or 0)
                    if bs <= 0:
                        bs = int(os.getenv('BATCH_SIZE', str(getattr(self, 'batch_size', 1))) or 1)
                    ocr_batch_size = max(1, bs)
                except Exception:
                    use_roi_locality = False
                    ocr_batch_enabled = False
                    ocr_batch_size = 1
                if use_roi_locality and (ocr_batch_enabled or ocr_batch_size > 1):
                    rois = self._prepare_ocr_rois_from_bubbles(image_path, ocr_settings, preprocessing, page_hash)
                    if rois:
                        # AZURE RATE LIMITING: Force low concurrency to prevent "Too Many Requests"
                        # Azure has strict rate limits that vary by tier:
                        # - Free tier: 20 requests/minute
                        # - Standard tier: Higher but still limited
                        try:
                            azure_workers = int(ocr_settings.get('ocr_max_concurrency') or 0)
                            if azure_workers <= 0:
                                azure_workers = 1  # Force sequential by default
                            else:
                                azure_workers = min(2, max(1, azure_workers))  # Cap at 2 max
                        except Exception:
                            azure_workers = 1  # Safe default
                        regions = self._azure_ocr_rois_concurrent(rois, ocr_settings, azure_workers, page_hash)
                        self._log(f"✅ Azure OCR concurrent over {len(rois)} ROIs → {len(regions)} regions (workers={azure_workers})", "info")
                        
                        # Force garbage collection after concurrent OCR to reduce memory spikes
                        try:
                            import gc
                            gc.collect()
                        except Exception:
                            pass
                        
                        return regions

                # Start local inpainter preload while Azure OCR runs (background; multiple if panel-parallel)
                try:
                    if not getattr(self, 'skip_inpainting', False) and not getattr(self, 'use_cloud_inpainting', False):
                        already_loaded, _lm = self._is_local_inpainter_loaded()
                        if not already_loaded:
                            import threading as _threading
                            local_method = (self.manga_settings.get('inpainting', {}) or {}).get('local_method', 'anime')
                            model_path = self.main_gui.config.get(f'manga_{local_method}_model_path', '') if hasattr(self, 'main_gui') else ''
                            adv = self.main_gui.config.get('manga_settings', {}).get('advanced', {}) if hasattr(self, 'main_gui') else {}
                            desired = 1
                            if adv.get('parallel_panel_translation', False):
                                try:
                                    desired = max(1, int(adv.get('panel_max_workers', 2)))
                                except Exception:
                                    desired = 2
                            allow = True if desired == 1 else bool(adv.get('preload_local_inpainting_for_panels', True))
                            if allow:
                                self._inpaint_preload_event = _threading.Event()
                                def _preload_inp_many():
                                    try:
                                        self.preload_local_inpainters_concurrent(local_method, model_path, desired)
                                    finally:
                                        try:
                                            self._inpaint_preload_event.set()
                                        except Exception:
                                            pass
                                _threading.Thread(target=_preload_inp_many, name="InpaintPreload@AzureOCR", daemon=True).start()
                except Exception:
                    pass
                
                # Ensure Azure-supported format for the BYTES we are sending.
                # If compression is enabled and produced an Azure-supported format (JPEG/PNG/BMP/TIFF),
                # DO NOT force-convert to PNG. Only convert when the current bytes are in an unsupported format.
                file_ext = os.path.splitext(image_path)[1].lower()
                azure_supported_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.pdf', '.tiff']
                azure_supported_fmts = ['jpeg', 'jpg', 'png', 'bmp', 'tiff']

                # Probe the actual byte format we will upload
                try:
                    from PIL import Image as _PILImage
                    img_probe = _PILImage.open(io.BytesIO(processed_image_data))
                    fmt = (img_probe.format or '').lower()
                except Exception:
                    fmt = ''

                # If original is a PDF, allow as-is (Azure supports PDF streams)
                if file_ext == '.pdf':
                    needs_convert = False
                else:
                    # Decide based on the detected format of the processed bytes
                    needs_convert = fmt not in azure_supported_fmts

                if needs_convert:
                    # If compression settings are enabled and target format is Azure-supported, prefer that
                    try:
                        comp_cfg = (self.main_gui.config.get('manga_settings', {}) or {}).get('compression', {})
                    except Exception:
                        comp_cfg = {}

                    # Determine if conversion is actually needed based on compression and current format
                    try:
                        from PIL import Image as _PILImage
                        img2 = _PILImage.open(io.BytesIO(processed_image_data))
                        fmt_lower = (img2.format or '').lower()
                    except Exception:
                        img2 = None
                        fmt_lower = ''

                    accepted = {'jpeg', 'jpg', 'png', 'bmp', 'tiff'}
                    convert_needed = False
                    target_fmt = None

                    if comp_cfg.get('enabled', False):
                        cf = str(comp_cfg.get('format', '')).lower()
                        desired = None
                        if cf in ('jpeg', 'jpg'):
                            desired = 'JPEG'
                        elif cf == 'png':
                            desired = 'PNG'
                        elif cf == 'bmp':
                            desired = 'BMP'
                        elif cf == 'tiff':
                            desired = 'TIFF'
                        # If WEBP or others, desired remains None and we fall back to PNG only if unsupported

                        if desired is not None:
                            # Skip conversion if already in the desired supported format
                            already_matches = ((fmt_lower in ('jpeg', 'jpg') and desired == 'JPEG') or (fmt_lower == desired.lower()))
                            if not already_matches:
                                convert_needed = True
                                target_fmt = desired
                        else:
                            # Compression format not supported by Azure (e.g., WEBP); convert only if unsupported
                            if fmt_lower not in accepted:
                                convert_needed = True
                                target_fmt = 'PNG'
                    else:
                        # No compression preference; convert only if unsupported by Azure
                        if fmt_lower not in accepted:
                            convert_needed = True
                            target_fmt = 'PNG'

                    if convert_needed:
                        self._log(f"⚠️ Converting image to {target_fmt} for Azure compatibility")
                        try:
                            if img2 is None:
                                from PIL import Image as _PILImage
                                img2 = _PILImage.open(io.BytesIO(processed_image_data))
                            buffer = io.BytesIO()
                            if target_fmt == 'JPEG' and img2.mode != 'RGB':
                                img2 = img2.convert('RGB')
                            img2.save(buffer, format=target_fmt)
                            processed_image_data = buffer.getvalue()
                        except Exception:
                            pass
                
                # Create stream from image data
                image_stream = io.BytesIO(processed_image_data)
                
                # Get Azure-specific settings
                reading_order = ocr_settings.get('azure_reading_order', 'natural')
                model_version = ocr_settings.get('azure_model_version', 'latest')
                max_wait = ocr_settings.get('azure_max_wait', 60)
                poll_interval = ocr_settings.get('azure_poll_interval', 0.5)
                
                # Map language hints to Azure language codes
                language_hints = ocr_settings.get('language_hints', ['ja', 'ko', 'zh'])
                
                # Build parameters dictionary
                read_params = {
                    'raw': True,
                    'readingOrder': reading_order
                }
                
                # Add model version if not using latest
                if model_version != 'latest':
                    read_params['model-version'] = model_version
                
                # Use language parameter only if single language is selected
                if len(language_hints) == 1:
                    azure_lang = language_hints[0]
                    # Map to Azure language codes
                    lang_mapping = {
                        'zh': 'zh-Hans',
                        'zh-TW': 'zh-Hant',
                        'zh-CN': 'zh-Hans',
                        'ja': 'ja',
                        'ko': 'ko',
                        'en': 'en'
                    }
                    azure_lang = lang_mapping.get(azure_lang, azure_lang)
                    read_params['language'] = azure_lang
                    self._log(f"   Using Azure Read API with language: {azure_lang}, order: {reading_order}")
                else:
                    self._log(f"   Using Azure Read API (auto-detect for {len(language_hints)} languages, order: {reading_order})")
                
                # Start Read operation with error handling and rate limit retry
                # Use max_retries from config (default 7, configurable in Other Settings)
                max_retries = self.main_gui.config.get('max_retries', 7)
                retry_delay = 60  # Start with 60 seconds for rate limits
                read_response = None
                
                for retry_attempt in range(max_retries):
                    try:
                        # Ensure client is alive before starting
                        if getattr(self, 'vision_client', None) is None:
                            self._log("⚠️ Azure client missing before read; reinitializing...", "warning")
                            self._ensure_azure_client()
                        if getattr(self, 'vision_client', None) is None:
                            raise RuntimeError("Azure Computer Vision client is not initialized. Check your key/endpoint and azure-cognitiveservices-vision-computervision installation.")

                        # Reset stream position for retry
                        image_stream.seek(0)
                        
                        read_response = self.vision_client.read_in_stream(
                            image_stream,
                            **read_params
                        )
                        # Success! Break out of retry loop
                        break
                        
                    except Exception as e:
                        error_msg = str(e)
                        
                        # Handle rate limit errors with fixed 60s wait
                        if 'Too Many Requests' in error_msg or '429' in error_msg:
                            if retry_attempt < max_retries - 1:
                                wait_time = retry_delay  # Fixed 60s wait each time
                                self._log(f"⚠️ Azure rate limit hit. Waiting {wait_time}s before retry {retry_attempt + 1}/{max_retries}...", "warning")
                                time.sleep(wait_time)
                                continue
                            else:
                                self._log(f"❌ Azure rate limit: Exhausted {max_retries} retries", "error")
                                raise
                        
                        # Handle bad request errors
                        elif 'Bad Request' in error_msg:
                            self._log("⚠️ Azure Read API Bad Request - likely invalid image format or too small. Retrying without language parameter...", "warning")
                            # Retry without language parameter
                            image_stream.seek(0)
                            read_params.pop('language', None)
                            if getattr(self, 'vision_client', None) is None:
                                self._ensure_azure_client()
                            read_response = self.vision_client.read_in_stream(
                                image_stream,
                                **read_params
                            )
                            break
                        else:
                            raise
                
                if read_response is None:
                    raise RuntimeError("Failed to get response from Azure Read API after retries")
                
                # Get operation ID
                operation_location = read_response.headers.get("Operation-Location") if hasattr(read_response, 'headers') else None
                if not operation_location:
                    raise RuntimeError("Azure Read API did not return Operation-Location header")
                operation_id = operation_location.split("/")[-1]
                
                # Poll for results with configurable timeout
                self._log(f"   Waiting for Azure OCR to complete (max {max_wait}s)...")
                wait_time = 0
                last_status = None
                result = None
                
                while wait_time < max_wait:
                    try:
                        if getattr(self, 'vision_client', None) is None:
                            # Client got cleaned up mid-poll; reinitialize and continue
                            self._log("⚠️ Azure client became None during polling; reinitializing...", "warning")
                            self._ensure_azure_client()
                            if getattr(self, 'vision_client', None) is None:
                                raise AttributeError("Azure client lost and could not be reinitialized")
                        result = self.vision_client.get_read_result(operation_id)
                    except AttributeError as e:
                        # Defensive: reinitialize once and retry this iteration
                        self._log(f"⚠️ {e} — reinitializing Azure client and retrying once", "warning")
                        self._ensure_azure_client()
                        if getattr(self, 'vision_client', None) is None:
                            raise
                        result = self.vision_client.get_read_result(operation_id)
                    
                    # Log status changes
                    if result.status != last_status:
                        self._log(f"   Status: {result.status}")
                        last_status = result.status
                    
                    if result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
                        break
                    
                    time.sleep(poll_interval)
                    self._log("💤 Azure OCR polling pausing briefly for stability", "debug")
                    wait_time += poll_interval
                
                if not result:
                    raise RuntimeError("Azure Read API polling did not return a result")
                if result.status == OperationStatusCodes.succeeded:
                    # Track statistics
                    total_lines = 0
                    handwritten_lines = 0
                    
                    for page_num, page in enumerate(result.analyze_result.read_results):
                        if len(result.analyze_result.read_results) > 1:
                            self._log(f"   Processing page {page_num + 1}/{len(result.analyze_result.read_results)}")
                        
                        for line in page.lines:
                            # CLEAN ORIGINAL OCR TEXT FOR AZURE - Fix cube characters and encoding issues
                            original_azure_text = line.text
                            cleaned_line_text = self._fix_encoding_issues(line.text)
                            cleaned_line_text = self._sanitize_unicode_characters(cleaned_line_text)
                            
                            # Log cleaning if changes were made
                            if cleaned_line_text != original_azure_text:
                                self._log(f"🧹 Cleaned Azure OCR text: '{original_azure_text[:30]}...' → '{cleaned_line_text[:30]}...'", "debug")
                            
                            # TEXT FILTERING FOR AZURE
                            # Skip if text is too short (after cleaning)
                            if len(cleaned_line_text.strip()) < min_text_length:
                                if not getattr(self, 'concise_logs', False):
                                    self._log(f"   Skipping short text ({len(cleaned_line_text)} chars): {cleaned_line_text}")
                                continue
                            
                            # Skip if primarily English and exclude_english is enabled (use cleaned text)
                            if exclude_english and self._is_primarily_english(cleaned_line_text):
                                if not getattr(self, 'concise_logs', False):
                                    self._log(f"   Skipping English text: {cleaned_line_text[:50]}...")
                                continue
                            
                            # Azure provides 8-point bounding box
                            bbox = line.bounding_box
                            vertices = [
                                (bbox[0], bbox[1]),
                                (bbox[2], bbox[3]),
                                (bbox[4], bbox[5]),
                                (bbox[6], bbox[7])
                            ]
                            
                            # Calculate rectangular bounding box
                            xs = [v[0] for v in vertices]
                            ys = [v[1] for v in vertices]
                            x_min, x_max = min(xs), max(xs)
                            y_min, y_max = min(ys), max(ys)
                            
                            # Calculate confidence from word-level data
                            confidence = 0.95  # Default high confidence
                            
                            if hasattr(line, 'words') and line.words:
                                # Calculate average confidence from words
                                confidences = []
                                for word in line.words:
                                    if hasattr(word, 'confidence'):
                                        confidences.append(word.confidence)
                                
                                if confidences:
                                    confidence = sum(confidences) / len(confidences)
                                    if not getattr(self, 'concise_logs', False):
                                        self._log(f"   Line has {len(line.words)} words, avg confidence: {confidence:.3f}")
                            
                            # Check for handwriting style (if available)
                            style = 'print'  # Default
                            style_confidence = None
                            
                            if hasattr(line, 'appearance') and line.appearance:
                                if hasattr(line.appearance, 'style'):
                                    style_info = line.appearance.style
                                    if hasattr(style_info, 'name'):
                                        style = style_info.name
                                        if style == 'handwriting':
                                            handwritten_lines += 1
                                    if hasattr(style_info, 'confidence'):
                                        style_confidence = style_info.confidence
                                        if not getattr(self, 'concise_logs', False):
                                            self._log(f"   Style: {style} (confidence: {style_confidence:.2f})")
                            
                            # Apply confidence threshold filtering
                            if confidence >= confidence_threshold:
                                region = TextRegion(
                                    text=cleaned_line_text,  # Use cleaned text instead of original
                                    vertices=vertices,
                                    bounding_box=(x_min, y_min, x_max - x_min, y_max - y_min),
                                    confidence=confidence,
                                    region_type='text_line'
                                )
                                
                                # Add extra attributes for Azure-specific info
                                region.style = style
                                region.style_confidence = style_confidence
                                
                                regions.append(region)
                                total_lines += 1
                                
                                # More detailed logging (use cleaned text)
                                if not getattr(self, 'concise_logs', False):
                                    if style == 'handwriting':
                                        self._log(f"   Found handwritten text ({confidence:.2f}): {cleaned_line_text[:50]}...")
                                    else:
                                        self._log(f"   Found text region ({confidence:.2f}): {cleaned_line_text[:50]}...")
                            else:
                                if not getattr(self, 'concise_logs', False):
                                    self._log(f"   Skipping low confidence text ({confidence:.2f}): {cleaned_line_text[:30]}...")
                    
                    # Log summary statistics
                    if total_lines > 0 and not getattr(self, 'concise_logs', False):
                        self._log(f"   Total lines detected: {total_lines}")
                        if handwritten_lines > 0:
                            self._log(f"   Handwritten lines: {handwritten_lines} ({handwritten_lines/total_lines*100:.1f}%)")
                    
                elif result.status == OperationStatusCodes.failed:
                    # More detailed error handling
                    error_msg = "Azure OCR failed"
                    if hasattr(result, 'message'):
                        error_msg += f": {result.message}"
                    if hasattr(result.analyze_result, 'errors') and result.analyze_result.errors:
                        for error in result.analyze_result.errors:
                            self._log(f"   Error: {error}", "error")
                    raise Exception(error_msg)
                else:
                    # Timeout or other status
                    raise Exception(f"Azure OCR ended with status: {result.status} after {wait_time}s")
                    
            else:
                # === NEW OCR PROVIDERS ===
                import cv2
                import numpy as np
                from ocr_manager import OCRManager
                
                # Load image as numpy array
                if isinstance(processed_image_data, bytes):
                    # Convert bytes to numpy array
                    nparr = np.frombuffer(processed_image_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    # Load from file path
                    image = cv2.imread(image_path)
                    if image is None:
                        # Try with PIL for Unicode paths
                        from PIL import Image as PILImage
                        pil_image = PILImage.open(image_path)
                        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
                # Ensure OCR manager is available
                if not hasattr(self, 'ocr_manager') or self.ocr_manager is None:
                    try:
                        # Prefer GUI-provided manager if available
                        if hasattr(self, 'main_gui') and hasattr(self.main_gui, 'ocr_manager') and self.main_gui.ocr_manager is not None:
                            self.ocr_manager = self.main_gui.ocr_manager
                        else:
                            from ocr_manager import OCRManager
                            self.ocr_manager = OCRManager(log_callback=self.log_callback)
                            self._log("Initialized internal OCRManager instance", "info")
                    except Exception as _e:
                        self.ocr_manager = None
                        self._log(f"Failed to initialize OCRManager: {str(_e)}", "error")
                if self.ocr_manager is None:
                    raise RuntimeError("OCRManager is not available; cannot proceed with OCR provider.")
                
                # Check provider status and load if needed
                provider_status = self.ocr_manager.check_provider_status(self.ocr_provider)

                if not provider_status['installed']:
                    self._log(f"❌ {self.ocr_provider} is not installed", "error")
                    self._log(f"   Please install it from the GUI settings", "error")
                    raise Exception(f"{self.ocr_provider} OCR provider is not installed")
                
                # Start local inpainter preload while provider is being readied/used (non-cloud path only; background)
                try:
                    if not getattr(self, 'skip_inpainting', False) and not getattr(self, 'use_cloud_inpainting', False):
                        already_loaded, _lm = self._is_local_inpainter_loaded()
                        if not already_loaded:
                            import threading as _threading
                            local_method = (self.manga_settings.get('inpainting', {}) or {}).get('local_method', 'anime')
                            model_path = self.main_gui.config.get(f'manga_{local_method}_model_path', '') if hasattr(self, 'main_gui') else ''
                            adv = self.main_gui.config.get('manga_settings', {}).get('advanced', {}) if hasattr(self, 'main_gui') else {}
                            desired = 1
                            if adv.get('parallel_panel_translation', False):
                                try:
                                    desired = max(1, int(adv.get('panel_max_workers', 2)))
                                except Exception:
                                    desired = 2
                            allow = True if desired == 1 else bool(adv.get('preload_local_inpainting_for_panels', True))
                            if allow:
                                self._inpaint_preload_event = _threading.Event()
                                def _preload_inp_many():
                                    try:
                                        self.preload_local_inpainters_concurrent(local_method, model_path, desired)
                                    finally:
                                        try:
                                            self._inpaint_preload_event.set()
                                        except Exception:
                                            pass
                                _threading.Thread(target=_preload_inp_many, name="InpaintPreload@OCRProvider", daemon=True).start()
                except Exception:
                    pass
                
                if not provider_status['loaded']:
                    # Check if Qwen2-VL - if it's supposedly not loaded but actually is, skip
                    if self.ocr_provider == 'Qwen2-VL':
                        provider = self.ocr_manager.get_provider('Qwen2-VL')
                        if provider and hasattr(provider, 'model') and provider.model is not None:
                            self._log("✅ Qwen2-VL model actually already loaded, skipping reload")
                            success = True
                        else:
                            # Only actually load if truly not loaded
                            model_size = self.ocr_config.get('model_size', '2') if hasattr(self, 'ocr_config') else '2'
                            self._log(f"Loading Qwen2-VL with model_size={model_size}")
                            success = self.ocr_manager.load_provider(self.ocr_provider, model_size=model_size)
                            if not success:
                                raise Exception(f"Failed to load {self.ocr_provider} model")
                    elif self.ocr_provider == 'custom-api':
                        # Custom API needs to initialize UnifiedClient with credentials
                        self._log("📡 Loading custom-api provider...")
                        # Try to get API key and model from GUI if available
                        load_kwargs = {}
                        if hasattr(self, 'main_gui'):
                            # Get API key from GUI
                            if hasattr(self.main_gui, 'api_key_entry'):
                                api_key = self.main_gui.api_key_entry.get()
                                if api_key:
                                    load_kwargs['api_key'] = api_key
                            # Get model from GUI  
                            if hasattr(self.main_gui, 'model_var'):
                                model = self.main_gui.model_var.get()
                                if model:
                                    load_kwargs['model'] = model
                        success = self.ocr_manager.load_provider(self.ocr_provider, **load_kwargs)
                        if not success:
                            raise Exception(f"Failed to initialize {self.ocr_provider}")
                    else:
                        # Other providers
                        success = self.ocr_manager.load_provider(self.ocr_provider)
                        if not success:
                            raise Exception(f"Failed to load {self.ocr_provider} model")
                    
                    if not success:
                        raise Exception(f"Failed to load {self.ocr_provider} model")
                
                # Initialize ocr_results here before any provider-specific code
                ocr_results = []
                
                # Special handling for manga-ocr (needs region detection first)
                if self.ocr_provider == 'manga-ocr':
                    # IMPORTANT: Initialize fresh results list
                    ocr_results = []
                    
                    # Check if we should use bubble detection for regions
                    if ocr_settings.get('bubble_detection_enabled', False):
                        self._log("📝 Using bubble detection regions for manga-ocr...")
                        
                        # Run bubble detection to get regions
                        if self.bubble_detector is None:
                            from bubble_detector import BubbleDetector
                            self.bubble_detector = BubbleDetector()
                        
                        # Get regions from bubble detector
                        rtdetr_detections = self._load_bubble_detector(ocr_settings, image_path)
                        if rtdetr_detections:
                            
                            # Process detections immediately and don't store
                            all_regions = []
                            
                            # ONLY ADD TEXT-CONTAINING REGIONS
                            # Skip empty bubbles since they shouldn't have text
                            if 'text_bubbles' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_bubbles', []))
                            if 'text_free' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_free', []))
                            
                            # DO NOT ADD empty bubbles - they're duplicates of text_bubbles
                            # if 'bubbles' in rtdetr_detections:  # <-- REMOVE THIS
                            #     all_regions.extend(rtdetr_detections.get('bubbles', []))
                            
                            self._log(f"📊 Processing {len(all_regions)} text-containing regions (skipping empty bubbles)")
                            
                            # Clear detection results after extracting regions
                            rtdetr_detections = None
                            
                            # Check if parallel processing is enabled
                            if self.parallel_processing and len(all_regions) > 1:
                                self._log(f"🚀 Using PARALLEL OCR for {len(all_regions)} regions with manga-ocr")
                                ocr_results = self._parallel_ocr_regions(image, all_regions, 'manga-ocr', confidence_threshold)
                            else:
                                # Process each region with manga-ocr
                                for i, (x, y, w, h) in enumerate(all_regions):
                                    cropped = self._safe_crop_region(image, x, y, w, h)
                                    if cropped is None:
                                        continue 
                                    result = self.ocr_manager.detect_text(cropped, 'manga-ocr', confidence=confidence_threshold)
                                    if result and len(result) > 0 and result[0].text.strip():
                                        result[0].bbox = (x, y, w, h)
                                        result[0].vertices = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                                        ocr_results.append(result[0])
                                        self._log(f"🔍 Processing region {i+1}/{len(all_regions)} with manga-ocr...")
                                        self._log(f"✅ Detected text: {result[0].text[:50]}...")

                            # Clear regions list after processing
                            all_regions = None
                    else:
                        # NO bubble detection - just process full image
                        self._log("📝 Processing full image with manga-ocr (no bubble detection)")
                        ocr_results = self.ocr_manager.detect_text(image, self.ocr_provider, confidence=confidence_threshold)  
                    
                elif self.ocr_provider == 'Qwen2-VL':
                    # Initialize results list
                    ocr_results = []
                    
                    # Configure Qwen2-VL for Korean text
                    language_hints = ocr_settings.get('language_hints', ['ko'])
                    self._log("🍩 Qwen2-VL OCR for Korean text recognition")
                    
                    # Check if we should use bubble detection for regions
                    if ocr_settings.get('bubble_detection_enabled', False):
                        self._log("📝 Using bubble detection regions for Qwen2-VL...")
                        
                        # Run bubble detection to get regions (thread-local)
                        _ = self._get_thread_bubble_detector()
                        
                        # Get regions from bubble detector
                        rtdetr_detections = self._load_bubble_detector(ocr_settings, image_path)
                        if rtdetr_detections:
                            
                            # Process only text-containing regions
                            all_regions = []
                            if 'text_bubbles' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_bubbles', []))
                            if 'text_free' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_free', []))
                            
                            self._log(f"📊 Processing {len(all_regions)} text regions with Qwen2-VL")
                            
                            # Check if parallel processing is enabled
                            if self.parallel_processing and len(all_regions) > 1:
                                self._log(f"🚀 Using PARALLEL OCR for {len(all_regions)} regions with Qwen2-VL")
                                ocr_results = self._parallel_ocr_regions(image, all_regions, 'Qwen2-VL', confidence_threshold)
                            else:
                                # Process each region with Qwen2-VL
                                for i, (x, y, w, h) in enumerate(all_regions):
                                    cropped = self._safe_crop_region(image, x, y, w, h)
                                    if cropped is None:
                                        continue 
                                    result = self.ocr_manager.detect_text(cropped, 'Qwen2-VL', confidence=confidence_threshold)
                                    if result and len(result) > 0 and result[0].text.strip():
                                        result[0].bbox = (x, y, w, h)
                                        result[0].vertices = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                                        ocr_results.append(result[0])
                                        self._log(f"✅ Region {i+1}: {result[0].text[:50]}...")
                    else:
                        # Process full image without bubble detection
                        self._log("📝 Processing full image with Qwen2-VL")
                        ocr_results = self.ocr_manager.detect_text(image, self.ocr_provider)

                elif self.ocr_provider == 'custom-api':
                    # Initialize results list
                    ocr_results = []
                    
                    # Configure Custom API for text extraction
                    self._log("🔌 Using Custom API for OCR")
                    
                    # Check if we should use bubble detection for regions
                    if ocr_settings.get('bubble_detection_enabled', False):
                        self._log("📝 Using bubble detection regions for Custom API...")
                        
                        # Run bubble detection to get regions (thread-local)
                        _ = self._get_thread_bubble_detector()
                        
                        # Get regions from bubble detector
                        rtdetr_detections = self._load_bubble_detector(ocr_settings, image_path)
                        if rtdetr_detections:
                            
                            # Process only text-containing regions
                            all_regions = []
                            if 'text_bubbles' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_bubbles', []))
                            if 'text_free' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_free', []))
                            
                            self._log(f"📊 Processing {len(all_regions)} text regions with Custom API")
                            
                            # Clear detections after extracting regions
                            rtdetr_detections = None
                            
                            # Decide parallelization for custom-api:
                            # Use API batch mode OR local parallel toggle so that API calls can run in parallel
                            if (getattr(self, 'batch_mode', False) or self.parallel_processing) and len(all_regions) > 1:
                                self._log(f"🚀 Using PARALLEL OCR for {len(all_regions)} regions (custom-api; API batch mode honored)")
                                ocr_results = self._parallel_ocr_regions(image, all_regions, 'custom-api', confidence_threshold)
                            else:
                                # Original sequential processing
                                for i, (x, y, w, h) in enumerate(all_regions):
                                    cropped = self._safe_crop_region(image, x, y, w, h)
                                    if cropped is None:
                                        continue 
                                    result = self.ocr_manager.detect_text(
                                        cropped, 
                                        'custom-api', 
                                        confidence=confidence_threshold
                                    )
                                    if result and len(result) > 0 and result[0].text.strip():
                                        result[0].bbox = (x, y, w, h)
                                        result[0].vertices = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                                        ocr_results.append(result[0])
                                        self._log(f"🔍 Region {i+1}/{len(all_regions)}: {result[0].text[:50]}...")
                            
                            # Clear regions list after processing
                            all_regions = None
                    else:
                        # Process full image without bubble detection
                        self._log("📝 Processing full image with Custom API")
                        ocr_results = self.ocr_manager.detect_text(
                            image, 
                            'custom-api',
                            confidence=confidence_threshold
                        )

                elif self.ocr_provider == 'easyocr':
                    # Initialize results list
                    ocr_results = []
                    
                    # Configure EasyOCR languages
                    language_hints = ocr_settings.get('language_hints', ['ja', 'en'])
                    validated_languages = self._validate_easyocr_languages(language_hints)
                    
                    easyocr_provider = self.ocr_manager.get_provider('easyocr')
                    if easyocr_provider:
                        if easyocr_provider.languages != validated_languages:
                            easyocr_provider.languages = validated_languages
                            easyocr_provider.is_loaded = False
                            self._log(f"🔥 Reloading EasyOCR with languages: {validated_languages}")
                            self.ocr_manager.load_provider('easyocr')
                    
                    # Check if we should use bubble detection
                    if ocr_settings.get('bubble_detection_enabled', False):
                        self._log("📝 Using bubble detection regions for EasyOCR...")
                        
                        # Run bubble detection to get regions (thread-local)
                        _ = self._get_thread_bubble_detector()
                        
                        # Get regions from bubble detector
                        rtdetr_detections = self._load_bubble_detector(ocr_settings, image_path)
                        if rtdetr_detections:
                            
                            # Process only text-containing regions
                            all_regions = []
                            if 'text_bubbles' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_bubbles', []))
                            if 'text_free' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_free', []))
                            
                            self._log(f"📊 Processing {len(all_regions)} text regions with EasyOCR")
                            
                            # Check if parallel processing is enabled
                            if self.parallel_processing and len(all_regions) > 1:
                                self._log(f"🚀 Using PARALLEL OCR for {len(all_regions)} regions with EasyOCR")
                                ocr_results = self._parallel_ocr_regions(image, all_regions, 'easyocr', confidence_threshold)
                            else:
                                # Process each region with EasyOCR
                                for i, (x, y, w, h) in enumerate(all_regions):
                                    cropped = self._safe_crop_region(image, x, y, w, h)
                                    if cropped is None:
                                        continue 
                                    result = self.ocr_manager.detect_text(cropped, 'easyocr', confidence=confidence_threshold)
                                    if result and len(result) > 0 and result[0].text.strip():
                                        result[0].bbox = (x, y, w, h)
                                        result[0].vertices = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                                        ocr_results.append(result[0])
                                        self._log(f"✅ Region {i+1}: {result[0].text[:50]}...")
                    else:
                        # Process full image without bubble detection
                        self._log("📝 Processing full image with EasyOCR")
                        ocr_results = self.ocr_manager.detect_text(image, self.ocr_provider)

                elif self.ocr_provider == 'paddleocr':
                    # Initialize results list
                    ocr_results = []
                    
                    # Configure PaddleOCR language
                    language_hints = ocr_settings.get('language_hints', ['ja'])
                    lang_map = {'ja': 'japan', 'ko': 'korean', 'zh': 'ch', 'en': 'en'}
                    paddle_lang = lang_map.get(language_hints[0] if language_hints else 'ja', 'japan')
                    
                    # Reload if language changed
                    paddle_provider = self.ocr_manager.get_provider('paddleocr')
                    if paddle_provider and paddle_provider.is_loaded:
                        if hasattr(paddle_provider.model, 'lang') and paddle_provider.model.lang != paddle_lang:
                            from paddleocr import PaddleOCR
                            paddle_provider.model = PaddleOCR(
                                use_angle_cls=True,
                                lang=paddle_lang,
                                use_gpu=True,
                                show_log=False
                            )
                            self._log(f"🔥 Reloaded PaddleOCR with language: {paddle_lang}")
                    
                    # Check if we should use bubble detection
                    if ocr_settings.get('bubble_detection_enabled', False):
                        self._log("📝 Using bubble detection regions for PaddleOCR...")
                        
                        # Run bubble detection to get regions (thread-local)
                        _ = self._get_thread_bubble_detector()
                        
                        # Get regions from bubble detector
                        rtdetr_detections = self._load_bubble_detector(ocr_settings, image_path)
                        if rtdetr_detections:
                            
                            # Process only text-containing regions
                            all_regions = []
                            if 'text_bubbles' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_bubbles', []))
                            if 'text_free' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_free', []))
                            
                            self._log(f"📊 Processing {len(all_regions)} text regions with PaddleOCR")
                            
                            # Check if parallel processing is enabled
                            if self.parallel_processing and len(all_regions) > 1:
                                self._log(f"🚀 Using PARALLEL OCR for {len(all_regions)} regions with PaddleOCR")
                                ocr_results = self._parallel_ocr_regions(image, all_regions, 'paddleocr', confidence_threshold)
                            else:
                                # Process each region with PaddleOCR
                                for i, (x, y, w, h) in enumerate(all_regions):
                                    cropped = self._safe_crop_region(image, x, y, w, h)
                                    if cropped is None:
                                        continue 
                                    result = self.ocr_manager.detect_text(cropped, 'paddleocr', confidence=confidence_threshold)
                                    if result and len(result) > 0 and result[0].text.strip():
                                        result[0].bbox = (x, y, w, h)
                                        result[0].vertices = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                                        ocr_results.append(result[0])
                                        self._log(f"✅ Region {i+1}: {result[0].text[:50]}...")
                    else:
                        # Process full image without bubble detection
                        self._log("📝 Processing full image with PaddleOCR")
                        ocr_results = self.ocr_manager.detect_text(image, self.ocr_provider)

                elif self.ocr_provider == 'doctr':
                    # Initialize results list
                    ocr_results = []
                    
                    self._log("📄 DocTR OCR for document text recognition")
                    
                    # Check if we should use bubble detection
                    if ocr_settings.get('bubble_detection_enabled', False):
                        self._log("📝 Using bubble detection regions for DocTR...")
                        
                        # Run bubble detection to get regions (thread-local)
                        _ = self._get_thread_bubble_detector()
                        
                        # Get regions from bubble detector
                        rtdetr_detections = self._load_bubble_detector(ocr_settings, image_path)
                        if rtdetr_detections:
                            
                            # Process only text-containing regions
                            all_regions = []
                            if 'text_bubbles' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_bubbles', []))
                            if 'text_free' in rtdetr_detections:
                                all_regions.extend(rtdetr_detections.get('text_free', []))
                            
                            self._log(f"📊 Processing {len(all_regions)} text regions with DocTR")
                            
                            # Check if parallel processing is enabled
                            if self.parallel_processing and len(all_regions) > 1:
                                self._log(f"🚀 Using PARALLEL OCR for {len(all_regions)} regions with DocTR")
                                ocr_results = self._parallel_ocr_regions(image, all_regions, 'doctr', confidence_threshold)
                            else:
                                # Process each region with DocTR
                                for i, (x, y, w, h) in enumerate(all_regions):
                                    cropped = self._safe_crop_region(image, x, y, w, h)
                                    if cropped is None:
                                        continue 
                                    result = self.ocr_manager.detect_text(cropped, 'doctr', confidence=confidence_threshold)
                                    if result and len(result) > 0 and result[0].text.strip():
                                        result[0].bbox = (x, y, w, h)
                                        result[0].vertices = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                                        ocr_results.append(result[0])
                                        self._log(f"✅ Region {i+1}: {result[0].text[:50]}...")
                    else:
                        # Process full image without bubble detection
                        self._log("📝 Processing full image with DocTR")
                        ocr_results = self.ocr_manager.detect_text(image, self.ocr_provider)

                elif self.ocr_provider == 'rapidocr':
                    # Initialize results list
                    ocr_results = []
                    
                    # Get RapidOCR settings
                    use_recognition = self.main_gui.config.get('rapidocr_use_recognition', True)
                    language = self.main_gui.config.get('rapidocr_language', 'auto')
                    detection_mode = self.main_gui.config.get('rapidocr_detection_mode', 'document')
                    
                    self._log(f"⚡ RapidOCR - Recognition: {'Full' if use_recognition else 'Detection Only'}")
                    
                    # ALWAYS process full image with RapidOCR for best results
                    self._log("📊 Processing full image with RapidOCR")
                    ocr_results = self.ocr_manager.detect_text(
                        image, 
                        'rapidocr',
                        confidence=confidence_threshold,
                        use_recognition=use_recognition,
                        language=language,
                        detection_mode=detection_mode
                    )
                    
                    # RT-DETR detection only affects merging, not OCR
                    if ocr_settings.get('bubble_detection_enabled', False):
                        self._log("🤖 RT-DETR will be used for bubble-based merging")

                else:
                    # Default processing for any other providers
                    ocr_results = self.ocr_manager.detect_text(image, self.ocr_provider)

                # Convert OCR results to TextRegion format
                for result in ocr_results:
                    # CLEAN ORIGINAL OCR TEXT - Fix cube characters and encoding issues
                    original_ocr_text = result.text
                    cleaned_result_text = self._fix_encoding_issues(result.text)
                    cleaned_result_text = self._normalize_unicode_width(cleaned_result_text)
                    cleaned_result_text = self._sanitize_unicode_characters(cleaned_result_text)
                    
                    # Log cleaning if changes were made
                    if cleaned_result_text != original_ocr_text:
                        self._log(f"🧹 Cleaned OCR manager text: '{original_ocr_text[:30]}...' → '{cleaned_result_text[:30]}...'", "debug")
                    
                    # Apply filtering (use cleaned text)
                    if len(cleaned_result_text.strip()) < min_text_length:
                        if not getattr(self, 'concise_logs', False):
                            self._log(f"   Skipping short text ({len(cleaned_result_text)} chars): {cleaned_result_text}")
                        continue
                    
                    if exclude_english and self._is_primarily_english(cleaned_result_text):
                        if not getattr(self, 'concise_logs', False):
                            self._log(f"   Skipping English text: {cleaned_result_text[:50]}...")
                        continue
                    
                    if result.confidence < confidence_threshold:
                        if not getattr(self, 'concise_logs', False):
                            self._log(f"   Skipping low confidence ({result.confidence:.2f}): {cleaned_result_text[:30]}...")
                        continue
                    
                    # Create TextRegion (use cleaned text)
                    region = TextRegion(
                        text=cleaned_result_text,  # Use cleaned text instead of original
                        vertices=result.vertices if result.vertices else [
                            (result.bbox[0], result.bbox[1]),
                            (result.bbox[0] + result.bbox[2], result.bbox[1]),
                            (result.bbox[0] + result.bbox[2], result.bbox[1] + result.bbox[3]),
                            (result.bbox[0], result.bbox[1] + result.bbox[3])
                        ],
                        bounding_box=result.bbox,
                        confidence=result.confidence,
                        region_type='text_block'
                    )
                    regions.append(region)
                    if not getattr(self, 'concise_logs', False):
                        self._log(f"   Found text ({result.confidence:.2f}): {cleaned_result_text[:50]}...")
            
            # MERGING SECTION (applies to all providers)
            # Check if bubble detection is enabled
            if ocr_settings.get('bubble_detection_enabled', False):
                self._log("🤖 Using AI bubble detection for merging")
                regions = self._merge_with_bubble_detection(regions, image_path)
            else:
                # Traditional merging
                merge_threshold = ocr_settings.get('merge_nearby_threshold', 20)
                
                # Apply provider-specific adjustments
                if self.ocr_provider == 'azure':
                    azure_multiplier = ocr_settings.get('azure_merge_multiplier', 2.0)
                    merge_threshold = int(merge_threshold * azure_multiplier)
                    self._log(f"📋 Using Azure-adjusted merge threshold: {merge_threshold}px")
                    
                    # Pre-group Azure lines if the method exists
                    if hasattr(self, '_pregroup_azure_lines'):
                        regions = self._pregroup_azure_lines(regions, merge_threshold)
                
                elif self.ocr_provider in ['paddleocr', 'easyocr', 'doctr']:
                    # These providers often return smaller text segments
                    line_multiplier = ocr_settings.get('line_ocr_merge_multiplier', 1.5)
                    merge_threshold = int(merge_threshold * line_multiplier)
                    self._log(f"📋 Using line-based OCR adjusted threshold: {merge_threshold}px")
                
                # Apply standard merging
                regions = self._merge_nearby_regions(regions, threshold=merge_threshold)
            
            self._log(f"✅ Detected {len(regions)} text regions after merging")
            
            # Save debug images only if 'Save intermediate images' is enabled
            advanced_settings = manga_settings.get('advanced', {})
            if advanced_settings.get('save_intermediate', False):
                self._save_debug_image(image_path, regions)
            
            return regions
            
        except Exception as e:
            self._log(f"❌ Error detecting text: {str(e)}", "error")
            import traceback
            self._log(traceback.format_exc(), "error")
            raise

    def _validate_easyocr_languages(self, languages):
        """Validate EasyOCR language combinations"""
        # EasyOCR compatibility rules
        incompatible_sets = [
            {'ja', 'ko'},  # Japanese + Korean
            {'ja', 'zh'},  # Japanese + Chinese  
            {'ko', 'zh'}   # Korean + Chinese
        ]
        
        lang_set = set(languages)
        
        for incompatible in incompatible_sets:
            if incompatible.issubset(lang_set):
                # Conflict detected - keep first language + English
                primary_lang = languages[0] if languages else 'en'
                result = [primary_lang, 'en'] if primary_lang != 'en' else ['en']
                
                self._log(f"⚠️ EasyOCR: {' + '.join(incompatible)} not compatible", "warning")
                self._log(f"🔧 Auto-adjusted from {languages} to {result}", "info")
                return result
        
        return languages

    def _parallel_ocr_regions(self, image: np.ndarray, regions: List, provider: str, confidence_threshold: float) -> List:
        """Process multiple regions in parallel using ThreadPoolExecutor"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        ocr_results = []
        results_lock = threading.Lock()
        
        def process_single_region(index: int, bbox: Tuple[int, int, int, int]):
            """Process a single region with OCR"""
            x, y, w, h = bbox
            try:
                # Use the safe crop method
                cropped = self._safe_crop_region(image, x, y, w, h)
                
                # Skip if crop failed
                if cropped is None:
                    self._log(f"⚠️ Skipping region {index} - invalid crop", "warning")
                    return
                
                # Run OCR on this region
                result = self.ocr_manager.detect_text(
                    cropped, 
                    provider,
                    confidence=confidence_threshold
                )
                
                if result and len(result) > 0 and result[0].text.strip():
                    # Adjust coordinates to full image space
                    result[0].bbox = (x, y, w, h)
                    result[0].vertices = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                    return (index, result[0])
                return (index, None)
                
            except Exception as e:
                self._log(f"Error processing region {index}: {str(e)}", "error")
                return (index, None)
        
        # Process regions in parallel
        max_workers = self.manga_settings.get('advanced', {}).get('max_workers', 4)
        # For custom-api, treat OCR calls as API calls: use batch size when batch mode is enabled
        try:
            if provider == 'custom-api':
                # prefer MangaTranslator.batch_size (from env BATCH_SIZE)
                bs = int(getattr(self, 'batch_size', 0) or int(os.getenv('BATCH_SIZE', '0')))
                if bs and bs > 0:
                    max_workers = bs
        except Exception:
            pass
        # Never spawn more workers than regions
        max_workers = max(1, min(max_workers, len(regions)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for i, bbox in enumerate(regions):
                future = executor.submit(process_single_region, i, bbox)
                future_to_index[future] = i
            
            # Collect results
            results_dict = {}
            completed = 0
            for future in as_completed(future_to_index):
                try:
                    index, result = future.result(timeout=30)
                    if result:
                        results_dict[index] = result
                        completed += 1
                        self._log(f"✅ [{completed}/{len(regions)}] Processed region {index+1}")
                except Exception as e:
                    self._log(f"Failed to process region: {str(e)}", "error")
            
            # Sort results by index to maintain order
            for i in range(len(regions)):
                if i in results_dict:
                    ocr_results.append(results_dict[i])
        
        self._log(f"📊 Parallel OCR complete: {len(ocr_results)}/{len(regions)} regions extracted")
        return ocr_results
    
    def _pregroup_azure_lines(self, lines: List[TextRegion], base_threshold: int) -> List[TextRegion]:
        """Pre-group Azure lines that are obviously part of the same text block
        This makes them more like Google's blocks before the main merge logic"""
        
        if len(lines) <= 1:
            return lines
        
        # Sort by vertical position first, then horizontal
        lines.sort(key=lambda r: (r.bounding_box[1], r.bounding_box[0]))
        
        pregrouped = []
        i = 0
        
        while i < len(lines):
            current_group = [lines[i]]
            current_bbox = list(lines[i].bounding_box)
            
            # Look ahead for lines that should obviously be grouped
            j = i + 1
            while j < len(lines):
                x1, y1, w1, h1 = current_bbox
                x2, y2, w2, h2 = lines[j].bounding_box
                
                # Calculate gaps
                vertical_gap = y2 - (y1 + h1) if y2 > y1 + h1 else 0
                
                # Check horizontal alignment
                center_x1 = x1 + w1 / 2
                center_x2 = x2 + w2 / 2
                horizontal_offset = abs(center_x1 - center_x2)
                avg_width = (w1 + w2) / 2
                
                # Group if:
                # 1. Lines are vertically adjacent (small gap)
                # 2. Lines are well-aligned horizontally (likely same bubble)
                if (vertical_gap < h1 * 0.5 and  # Less than half line height gap
                    horizontal_offset < avg_width * 0.5):  # Well centered
                    
                    # Add to group
                    current_group.append(lines[j])
                    
                    # Update bounding box to include new line
                    min_x = min(x1, x2)
                    min_y = min(y1, y2)
                    max_x = max(x1 + w1, x2 + w2)
                    max_y = max(y1 + h1, y2 + h2)
                    current_bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
                    
                    j += 1
                else:
                    break
            
            # Create merged region from group
            if len(current_group) > 1:
                merged_text = " ".join([line.text for line in current_group])
                all_vertices = []
                for line in current_group:
                    all_vertices.extend(line.vertices)
                
                merged_region = TextRegion(
                    text=merged_text,
                    vertices=all_vertices,
                    bounding_box=tuple(current_bbox),
                    confidence=0.95,
                    region_type='pregrouped_lines'
                )
                pregrouped.append(merged_region)
                
                self._log(f"   Pre-grouped {len(current_group)} Azure lines into block")
            else:
                # Single line, keep as is
                pregrouped.append(lines[i])
            
            i = j if j > i + 1 else i + 1
        
        self._log(f"   Azure pre-grouping: {len(lines)} lines → {len(pregrouped)} blocks")
        return pregrouped

    def _safe_crop_region(self, image, x, y, w, h):
        """Safely crop a region from image with validation"""
        img_h, img_w = image.shape[:2]
        
        # Validate and clamp coordinates
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        x2 = min(x + w, img_w)
        y2 = min(y + h, img_h)
        
        # Ensure valid region
        if x2 <= x or y2 <= y:
            self._log(f"⚠️ Invalid crop region: ({x},{y},{w},{h}) for image {img_w}x{img_h}", "warning")
            return None
        
        # Minimum size check
        if (x2 - x) < 5 or (y2 - y) < 5:
            self._log(f"⚠️ Region too small: {x2-x}x{y2-y} pixels", "warning")
            return None
        
        cropped = image[y:y2, x:x2]
        
        if cropped.size == 0:
            self._log(f"⚠️ Empty crop result", "warning")
            return None
        
        return cropped

    def _prepare_ocr_rois_from_bubbles(self, image_path: str, ocr_settings: Dict, preprocessing: Dict, page_hash: str) -> List[Dict[str, Any]]:
        """Prepare ROI crops (bytes) from bubble detection to use with OCR locality.
        - Enhancements/resizing are gated by preprocessing['enabled'].
        - Compression/encoding is controlled by manga_settings['compression'] independently.
        Returns list of dicts: {id, bbox, bytes, type}
        """
        try:
            # Run bubble detector and collect text-containing boxes
            detections = self._load_bubble_detector(ocr_settings, image_path)
            if not detections:
                return []
            regions = []
            for key in ('text_bubbles', 'text_free'):
                for i, (bx, by, bw, bh) in enumerate(detections.get(key, []) or []):
                    regions.append({'type': 'text_bubble' if key == 'text_bubbles' else 'free_text',
                                    'bbox': (int(bx), int(by), int(bw), int(bh)),
                                    'id': f"{key}_{i}"})
            if not regions:
                return []

            # Open original image once
            pil = Image.open(image_path)
            if pil.mode != 'RGB':
                pil = pil.convert('RGB')

            pad_ratio = float(ocr_settings.get('roi_padding_ratio', 0.08))  # 8% padding default
            preproc_enabled = bool(preprocessing.get('enabled', False))
            # Compression settings (separate from preprocessing)
            comp = {}
            try:
                comp = (self.main_gui.config.get('manga_settings', {}) or {}).get('compression', {})
            except Exception:
                comp = {}
            comp_enabled = bool(comp.get('enabled', False))
            comp_format = str(comp.get('format', 'jpeg')).lower()
            jpeg_q = int(comp.get('jpeg_quality', 85))
            png_lvl = int(comp.get('png_compress_level', 6))
            webp_q = int(comp.get('webp_quality', 85))

            out = []
            W, H = pil.size
            # Pre-filter tiny ROIs (skip before cropping)
            min_side_px = int(ocr_settings.get('roi_min_side_px', 12))
            min_area_px = int(ocr_settings.get('roi_min_area_px', 100))
            for rec in regions:
                x, y, w, h = rec['bbox']
                if min(w, h) < max(1, min_side_px) or (w * h) < max(1, min_area_px):
                    # Skip tiny ROI
                    continue
                # Apply padding
                px = int(w * pad_ratio)
                py = int(h * pad_ratio)
                x1 = max(0, x - px)
                y1 = max(0, y - py)
                x2 = min(W, x + w + px)
                y2 = min(H, y + h + py)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = pil.crop((x1, y1, x2, y2))

                # Quality-affecting steps only when preprocessing enabled
                if preproc_enabled:
                    try:
                        # Enhance contrast/sharpness/brightness if configured
                        c = float(preprocessing.get('contrast_threshold', 0.4))
                        s = float(preprocessing.get('sharpness_threshold', 0.3))
                        g = float(preprocessing.get('enhancement_strength', 1.5))
                        if c:
                            crop = ImageEnhance.Contrast(crop).enhance(1 + c)
                        if s:
                            crop = ImageEnhance.Sharpness(crop).enhance(1 + s)
                        if g and g != 1.0:
                            crop = ImageEnhance.Brightness(crop).enhance(g)
                        # Optional ROI resize limit (short side cap)
                        roi_max_side = int(ocr_settings.get('roi_max_side', 0) or 0)
                        if roi_max_side and (crop.width > roi_max_side or crop.height > roi_max_side):
                            ratio = min(roi_max_side / crop.width, roi_max_side / crop.height)
                            crop = crop.resize((max(1, int(crop.width * ratio)), max(1, int(crop.height * ratio))), Image.Resampling.LANCZOS)
                    except Exception:
                        pass
                # Encoding/Compression independent of preprocessing
                from io import BytesIO
                buf = BytesIO()
                try:
                    if comp_enabled:
                        if comp_format in ('jpeg', 'jpg'):
                            if crop.mode != 'RGB':
                                crop = crop.convert('RGB')
                            crop.save(buf, format='JPEG', quality=max(1, min(95, jpeg_q)), optimize=True, progressive=True)
                        elif comp_format == 'png':
                            crop.save(buf, format='PNG', optimize=True, compress_level=max(0, min(9, png_lvl)))
                        elif comp_format == 'webp':
                            crop.save(buf, format='WEBP', quality=max(1, min(100, webp_q)))
                        else:
                            crop.save(buf, format='PNG', optimize=True)
                    else:
                        # Default lossless PNG
                        crop.save(buf, format='PNG', optimize=True)
                    img_bytes = buf.getvalue()
                except Exception:
                    buf = BytesIO()
                    crop.save(buf, format='PNG', optimize=True)
                    img_bytes = buf.getvalue()

                out.append({
                    'id': rec['id'],
                    'bbox': (x, y, w, h),  # keep original bbox without padding for placement
                    'bytes': img_bytes,
                    'type': rec['type'],
                    'page_hash': page_hash
                })
            return out
        except Exception as e:
            self._log(f"⚠️ ROI preparation failed: {e}", "warning")
            return []

    def _google_ocr_rois_batched(self, rois: List[Dict[str, Any]], ocr_settings: Dict, batch_size: int, max_concurrency: int, page_hash: str) -> List[TextRegion]:
        """Batch OCR of ROI crops using Google Vision batchAnnotateImages.
        - Uses bounded concurrency for multiple batches in flight.
        - Consults and updates an in-memory ROI OCR cache.
        """
        try:
            from google.cloud import vision as _vision
        except Exception:
            self._log("❌ Google Vision SDK not available for ROI batching", "error")
            return []

        lang_hints = ocr_settings.get('language_hints', ['ja', 'ko', 'zh'])
        detection_mode = ocr_settings.get('text_detection_mode', 'document')
        feature_type = _vision.Feature.Type.DOCUMENT_TEXT_DETECTION if detection_mode == 'document' else _vision.Feature.Type.TEXT_DETECTION
        feature = _vision.Feature(type=feature_type)

        results: List[TextRegion] = []
        min_text_length = int(ocr_settings.get('min_text_length', 2))
        exclude_english = bool(ocr_settings.get('exclude_english_text', True))

        # Check cache first and build work list of uncached ROIs
        work_rois = []
        for roi in rois:
            x, y, w, h = roi['bbox']
            # Include region type in cache key to prevent mismapping
            cache_key = ("google", page_hash, x, y, w, h, tuple(lang_hints), detection_mode, roi.get('type', 'unknown'))
            # THREAD-SAFE: Use lock for cache access in parallel panel translation
            with self._cache_lock:
                cached_text = self.ocr_roi_cache.get(cache_key)
            if cached_text:
                region = TextRegion(
                    text=cached_text,
                    vertices=[(x, y), (x+w, y), (x+w, y+h), (x, y+h)],
                    bounding_box=(x, y, w, h),
                    confidence=0.95,
                    region_type='ocr_roi'
                )
                try:
                    region.bubble_type = 'free_text' if roi.get('type') == 'free_text' else 'text_bubble'
                    region.should_inpaint = True
                except Exception:
                    pass
                results.append(region)
            else:
                roi['cache_key'] = cache_key
                work_rois.append(roi)

        if not work_rois:
            return results

        # Create batches
        batch_size = max(1, batch_size)
        batches = [work_rois[i:i+batch_size] for i in range(0, len(work_rois), batch_size)]
        max_concurrency = max(1, int(max_concurrency or 1))

        def do_batch(batch):
            # RATE LIMITING: Add small delay before batch submission
            import time
            import random
            time.sleep(0.1 + random.random() * 0.2)  # 0.1-0.3s random delay
            
            requests = []
            for roi in batch:
                img = _vision.Image(content=roi['bytes'])
                ctx = _vision.ImageContext(language_hints=list(lang_hints))
                req = _vision.AnnotateImageRequest(image=img, features=[feature], image_context=ctx)
                requests.append(req)
            return self.vision_client.batch_annotate_images(requests=requests), batch

        # Execute with concurrency
        if max_concurrency == 1 or len(batches) == 1:
            iter_batches = [(self.vision_client.batch_annotate_images(requests=[
                _vision.AnnotateImageRequest(image=_vision.Image(content=roi['bytes']), features=[feature], image_context=_vision.ImageContext(language_hints=list(lang_hints)))
                for roi in batch
            ]), batch) for batch in batches]
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            iter_batches = []
            with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
                futures = [ex.submit(do_batch, b) for b in batches]
                for fut in as_completed(futures):
                    try:
                        iter_batches.append(fut.result())
                    except Exception as e:
                        self._log(f"⚠️ Google batch failed: {e}", "warning")
                        continue

        # Consume responses and update cache
        for resp, batch in iter_batches:
            for roi, ann in zip(batch, resp.responses):
                if getattr(ann, 'error', None) and ann.error.message:
                    self._log(f"⚠️ ROI OCR error: {ann.error.message}", "warning")
                    continue
                text = ''
                try:
                    if getattr(ann, 'full_text_annotation', None) and ann.full_text_annotation.text:
                        text = ann.full_text_annotation.text
                    elif ann.text_annotations:
                        text = ann.text_annotations[0].description
                except Exception:
                    text = ''
                text = (text or '').strip()
                text_clean = self._sanitize_unicode_characters(self._fix_encoding_issues(text))
                if len(text_clean.strip()) < min_text_length:
                    continue
                if exclude_english and self._is_primarily_english(text_clean):
                    continue
                x, y, w, h = roi['bbox']
                # Update cache
                # THREAD-SAFE: Use lock for cache write in parallel panel translation
                try:
                    ck = roi.get('cache_key') or ("google", page_hash, x, y, w, h, tuple(lang_hints), detection_mode)
                    with self._cache_lock:
                        self.ocr_roi_cache[ck] = text_clean
                except Exception:
                    pass
                region = TextRegion(
                    text=text_clean,
                    vertices=[(x, y), (x+w, y), (x+w, y+h), (x, y+h)],
                    bounding_box=(x, y, w, h),
                    confidence=0.95,
                    region_type='ocr_roi'
                )
                try:
                    region.bubble_type = 'free_text' if roi.get('type') == 'free_text' else 'text_bubble'
                    region.should_inpaint = True
                except Exception:
                    pass
                results.append(region)
        return results

    def _azure_ocr_rois_concurrent(self, rois: List[Dict[str, Any]], ocr_settings: Dict, max_workers: int, page_hash: str) -> List[TextRegion]:
        """Concurrent ROI OCR for Azure Read API. Each ROI is sent as a separate call.
        Concurrency is bounded by max_workers. Consults/updates cache.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
        import io
        results: List[TextRegion] = []

        # Read settings
        reading_order = ocr_settings.get('azure_reading_order', 'natural')
        model_version = ocr_settings.get('azure_model_version', 'latest')
        language_hints = ocr_settings.get('language_hints', ['ja'])
        read_params = {'raw': True, 'readingOrder': reading_order}
        if model_version != 'latest':
            read_params['model-version'] = model_version
        if len(language_hints) == 1:
            lang_mapping = {'zh': 'zh-Hans', 'zh-TW': 'zh-Hant', 'zh-CN': 'zh-Hans', 'ja': 'ja', 'ko': 'ko', 'en': 'en'}
            read_params['language'] = lang_mapping.get(language_hints[0], language_hints[0])

        min_text_length = int(ocr_settings.get('min_text_length', 2))
        exclude_english = bool(ocr_settings.get('exclude_english_text', True))

        # Check cache first and split into cached vs work rois
        cached_regions: List[TextRegion] = []
        work_rois: List[Dict[str, Any]] = []
        for roi in rois:
            x, y, w, h = roi['bbox']
            # Include region type in cache key to prevent mismapping
            cache_key = ("azure", page_hash, x, y, w, h, reading_order, roi.get('type', 'unknown'))
            # THREAD-SAFE: Use lock for cache access in parallel panel translation
            with self._cache_lock:
                text_cached = self.ocr_roi_cache.get(cache_key)
            if text_cached:
                region = TextRegion(
                    text=text_cached,
                    vertices=[(x, y), (x+w, y), (x+w, y+h), (x, y+h)],
                    bounding_box=(x, y, w, h),
                    confidence=0.95,
                    region_type='ocr_roi'
                )
                try:
                    region.bubble_type = 'free_text' if roi.get('type') == 'free_text' else 'text_bubble'
                    region.should_inpaint = True
                except Exception:
                    pass
                cached_regions.append(region)
            else:
                roi['cache_key'] = cache_key
                work_rois.append(roi)

        def ocr_one(roi):
            try:
                # RATE LIMITING: Add delay between Azure API calls to avoid "Too Many Requests"
                import time
                import random
                # Stagger requests with randomized delay
                time.sleep(0.1 + random.random() * 0.2)  # 0.1-0.3s random delay
                
                # Ensure Azure-supported format for ROI bytes; honor compression preference when possible
                data = roi['bytes']
                try:
                    from PIL import Image as _PILImage
                    im = _PILImage.open(io.BytesIO(data))
                    fmt = (im.format or '').lower()
                    if fmt not in ['jpeg', 'jpg', 'png', 'bmp', 'tiff']:
                        # Choose conversion target based on compression settings if available
                        try:
                            comp_cfg = (self.main_gui.config.get('manga_settings', {}) or {}).get('compression', {})
                        except Exception:
                            comp_cfg = {}
                        target_fmt = 'PNG'
                        try:
                            if comp_cfg.get('enabled', False):
                                cf = str(comp_cfg.get('format', '')).lower()
                                if cf in ('jpeg', 'jpg'):
                                    target_fmt = 'JPEG'
                                elif cf == 'png':
                                    target_fmt = 'PNG'
                                elif cf == 'bmp':
                                    target_fmt = 'BMP'
                                elif cf == 'tiff':
                                    target_fmt = 'TIFF'
                        except Exception:
                            pass
                        buf2 = io.BytesIO()
                        if target_fmt == 'JPEG' and im.mode != 'RGB':
                            im = im.convert('RGB')
                        im.save(buf2, format=target_fmt)
                        data = buf2.getvalue()
                except Exception:
                    pass
                stream = io.BytesIO(data)
                read_response = self.vision_client.read_in_stream(stream, **read_params)
                op_loc = read_response.headers.get('Operation-Location') if hasattr(read_response, 'headers') else None
                if not op_loc:
                    return None
                op_id = op_loc.split('/')[-1]
                # Poll
                import time
                waited = 0.0
                poll_interval = float(ocr_settings.get('azure_poll_interval', 0.5))
                max_wait = float(ocr_settings.get('azure_max_wait', 60))
                while waited < max_wait:
                    result = self.vision_client.get_read_result(op_id)
                    if result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
                        break
                    time.sleep(poll_interval)
                    waited += poll_interval
                if result.status != OperationStatusCodes.succeeded:
                    return None
                # Aggregate text lines
                texts = []
                for page in result.analyze_result.read_results:
                    for line in page.lines:
                        t = self._sanitize_unicode_characters(self._fix_encoding_issues(line.text or ''))
                        if t:
                            texts.append(t)
                text_all = ' '.join(texts).strip()
                if len(text_all) < min_text_length:
                    return None
                if exclude_english and self._is_primarily_english(text_all):
                    return None
                x, y, w, h = roi['bbox']
                # Update cache
                # THREAD-SAFE: Use lock for cache write in parallel panel translation
                try:
                    ck = roi.get('cache_key')
                    if ck:
                        with self._cache_lock:
                            self.ocr_roi_cache[ck] = text_all
                except Exception:
                    pass
                region = TextRegion(
                    text=text_all,
                    vertices=[(x, y), (x+w, y), (x+w, y+h), (x, y+h)],
                    bounding_box=(x, y, w, h),
                    confidence=0.95,
                    region_type='ocr_roi'
                )
                try:
                    region.bubble_type = 'free_text' if roi.get('type') == 'free_text' else 'text_bubble'
                    region.should_inpaint = True
                except Exception:
                    pass
                return region
            except Exception:
                return None

        # Combine cached and new results
        results.extend(cached_regions)
        
        if work_rois:
            max_workers = max(1, min(max_workers, len(work_rois)))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                fut_map = {ex.submit(ocr_one, r): r for r in work_rois}
                for fut in as_completed(fut_map):
                    reg = fut.result()
                    if reg is not None:
                        results.append(reg)
        return results

    def _detect_text_azure(self, image_data: bytes, ocr_settings: dict) -> List[TextRegion]:
        """Detect text using Azure Computer Vision"""
        import io
        from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
        
        stream = io.BytesIO(image_data)
        
        # Use Read API for better manga text detection
        read_result = self.vision_client.read_in_stream(
            stream,
            raw=True,
            language='ja'  # or from ocr_settings
        )
        
        # Get operation ID from headers
        operation_location = read_result.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]
        
        # Wait for completion
        import time
        while True:
            result = self.vision_client.get_read_result(operation_id)
            if result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
                break
            time.sleep(0.1)  # Brief pause for stability
            logger.debug("💤 Azure text detection pausing briefly for stability")
        
        regions = []
        confidence_threshold = ocr_settings.get('confidence_threshold', 0.6)
        
        if result.status == OperationStatusCodes.succeeded:
            for page in result.analyze_result.read_results:
                for line in page.lines:
                    # Azure returns bounding box as 8 coordinates
                    bbox = line.bounding_box
                    vertices = [
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        (bbox[4], bbox[5]),
                        (bbox[6], bbox[7])
                    ]
                    
                    xs = [v[0] for v in vertices]
                    ys = [v[1] for v in vertices]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    
                    # Azure doesn't provide per-line confidence in Read API
                    confidence = 0.95  # Default high confidence
                    
                    if confidence >= confidence_threshold:
                        region = TextRegion(
                            text=line.text,
                            vertices=vertices,
                            bounding_box=(x_min, y_min, x_max - x_min, y_max - y_min),
                            confidence=confidence,
                            region_type='text_line'
                        )
                        regions.append(region)
        
        return regions

    def _load_image_with_compression_only(self, image_path: str, comp: Dict) -> bytes:
        """Load image and apply compression settings only (no enhancements/resizing)."""
        from io import BytesIO
        pil = Image.open(image_path)
        if pil.mode != 'RGB':
            pil = pil.convert('RGB')
        buf = BytesIO()
        try:
            fmt = str(comp.get('format', 'jpeg')).lower()
            if fmt in ('jpeg', 'jpg'):
                q = max(1, min(95, int(comp.get('jpeg_quality', 85))))
                pil.save(buf, format='JPEG', quality=q, optimize=True, progressive=True)
            elif fmt == 'png':
                lvl = max(0, min(9, int(comp.get('png_compress_level', 6))))
                pil.save(buf, format='PNG', optimize=True, compress_level=lvl)
            elif fmt == 'webp':
                wq = max(1, min(100, int(comp.get('webp_quality', 85))))
                pil.save(buf, format='WEBP', quality=wq)
            else:
                pil.save(buf, format='PNG', optimize=True)
        except Exception:
            pil.save(buf, format='PNG', optimize=True)
        return buf.getvalue()

    def _preprocess_image(self, image_path: str, preprocessing_settings: Dict) -> bytes:
        """Preprocess image for better OCR results
        - Enhancements/resizing controlled by preprocessing_settings
        - Compression controlled by manga_settings['compression'] independently
        """
        try:
            # Open image with PIL
            pil_image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Auto-detect quality issues if enabled
            if preprocessing_settings.get('auto_detect_quality', True):
                needs_enhancement = self._detect_quality_issues(pil_image, preprocessing_settings)
                if needs_enhancement:
                    self._log("   Auto-detected quality issues - applying enhancements")
            else:
                needs_enhancement = True
            
            if needs_enhancement:
                # Apply contrast enhancement
                contrast_threshold = preprocessing_settings.get('contrast_threshold', 0.4)
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1 + contrast_threshold)
                
                # Apply sharpness enhancement
                sharpness_threshold = preprocessing_settings.get('sharpness_threshold', 0.3)
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(1 + sharpness_threshold)
                
                # Apply general enhancement strength
                enhancement_strength = preprocessing_settings.get('enhancement_strength', 1.5)
                if enhancement_strength != 1.0:
                    # Brightness adjustment
                    enhancer = ImageEnhance.Brightness(pil_image)
                    pil_image = enhancer.enhance(enhancement_strength)
            
            # Resize if too large
            max_dimension = preprocessing_settings.get('max_image_dimension', 2000)
            if pil_image.width > max_dimension or pil_image.height > max_dimension:
                ratio = min(max_dimension / pil_image.width, max_dimension / pil_image.height)
                new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                self._log(f"   Resized image to {new_size[0]}x{new_size[1]}")
            
            # Convert back to bytes with compression settings from global config
            from io import BytesIO
            buffered = BytesIO()
            comp = {}
            try:
                comp = (self.main_gui.config.get('manga_settings', {}) or {}).get('compression', {})
            except Exception:
                comp = {}
            try:
                if comp.get('enabled', False):
                    fmt = str(comp.get('format', 'jpeg')).lower()
                    if fmt in ('jpeg', 'jpg'):
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                        quality = max(1, min(95, int(comp.get('jpeg_quality', 85))))
                        pil_image.save(buffered, format='JPEG', quality=quality, optimize=True, progressive=True)
                        self._log(f"   Compressed image as JPEG (q={quality})")
                    elif fmt == 'png':
                        level = max(0, min(9, int(comp.get('png_compress_level', 6))))
                        pil_image.save(buffered, format='PNG', optimize=True, compress_level=level)
                        self._log(f"   Compressed image as PNG (level={level})")
                    elif fmt == 'webp':
                        q = max(1, min(100, int(comp.get('webp_quality', 85))))
                        pil_image.save(buffered, format='WEBP', quality=q)
                        self._log(f"   Compressed image as WEBP (q={q})")
                    else:
                        pil_image.save(buffered, format='PNG', optimize=True)
                        self._log("   Unknown compression format; saved as optimized PNG")
                else:
                    pil_image.save(buffered, format='PNG', optimize=True)
            except Exception as _e:
                self._log(f"   ⚠️ Compression failed ({_e}); saved as optimized PNG", "warning")
                pil_image.save(buffered, format='PNG', optimize=True)
            return buffered.getvalue()
            
        except Exception as e:
            self._log(f"⚠️ Preprocessing failed: {str(e)}, using original image", "warning")
            with open(image_path, 'rb') as f:
                return f.read()

    def _detect_quality_issues(self, image: Image.Image, settings: Dict) -> bool:
        """Auto-detect if image needs quality enhancement"""
        # Convert to grayscale for analysis
        gray = image.convert('L')
        
        # Get histogram
        hist = gray.histogram()
        
        # Calculate contrast (simplified)
        pixels = sum(hist)
        mean = sum(i * hist[i] for i in range(256)) / pixels
        variance = sum(hist[i] * (i - mean) ** 2 for i in range(256)) / pixels
        std_dev = variance ** 0.5
        
        # Low contrast if std deviation is low
        contrast_threshold = settings.get('contrast_threshold', 0.4) * 100
        if std_dev < contrast_threshold:
            self._log("   Low contrast detected")
            return True
        
        # Check for blur using Laplacian variance
        import numpy as np
        gray_array = np.array(gray)
        laplacian = cv2.Laplacian(gray_array, cv2.CV_64F)
        variance = laplacian.var()
        
        sharpness_threshold = settings.get('sharpness_threshold', 0.3) * 100
        if variance < sharpness_threshold:
            self._log("   Blur detected")
            return True
        
        return False

    def _save_debug_image(self, image_path: str, regions: List[TextRegion], debug_base_dir: str = None):
        """Save debug image with detected regions highlighted, respecting save_intermediate toggle.
        All files are written under <translated_images>/debug (or provided debug_base_dir)."""
        advanced_settings = self.manga_settings.get('advanced', {})
        # Skip debug images in batch mode unless explicitly requested
        if self.batch_mode and not advanced_settings.get('force_debug_batch', False):
            return
        # Respect the 'Save intermediate images' toggle only
        if not advanced_settings.get('save_intermediate', False):
            return
        # Compute debug directory under translated_images
        if debug_base_dir is None:
            translated_dir = os.path.join(os.path.dirname(image_path), 'translated_images')
            debug_dir = os.path.join(translated_dir, 'debug')
        else:
            debug_dir = os.path.join(debug_base_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        try:
            import cv2
            import numpy as np
            from PIL import Image as PILImage
            
            # Handle Unicode paths
            try:
                img = cv2.imread(image_path)
                if img is None:
                    # Fallback to PIL for Unicode paths
                    pil_image = PILImage.open(image_path)
                    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                self._log(f"   Failed to load image for debug: {str(e)}", "warning")
                return
            
            # Debug directory prepared earlier; compute base name
            # base_name already computed above
            
            # Draw rectangles around detected text regions
            overlay = img.copy()
            
            # Calculate statistics
            total_chars = sum(len(r.text) for r in regions)
            avg_confidence = np.mean([r.confidence for r in regions]) if regions else 0
            
            for i, region in enumerate(regions):
                x, y, w, h = region.bounding_box
                
                # Color based on confidence
                if region.confidence > 0.95:
                    color = (0, 255, 0)  # Green - high confidence
                elif region.confidence > 0.8:
                    color = (0, 165, 255)  # Orange - medium confidence
                else:
                    color = (0, 0, 255)  # Red - low confidence
                
                # Draw rectangle
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                
                # Add region info
                info_text = f"#{i} ({region.confidence:.2f})"
                cv2.putText(overlay, info_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1, cv2.LINE_AA)
                
                # Add character count
                char_count = len(region.text.strip())
                cv2.putText(overlay, f"{char_count} chars", (x, y + h + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                
                # Add detected text preview if in verbose debug mode
                if self.manga_settings.get('advanced', {}).get('save_intermediate', False):
                    text_preview = region.text[:20] + "..." if len(region.text) > 20 else region.text
                    cv2.putText(overlay, text_preview, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.4, color, 1, cv2.LINE_AA)
            
            # Add overall statistics to the image
            stats_bg = overlay.copy()
            cv2.rectangle(stats_bg, (10, 10), (300, 90), (0, 0, 0), -1)
            cv2.addWeighted(stats_bg, 0.7, overlay, 0.3, 0, overlay)
            
            stats_text = [
                f"Regions: {len(regions)}",
                f"Total chars: {total_chars}",
                f"Avg confidence: {avg_confidence:.2f}"
            ]
            
            for i, text in enumerate(stats_text):
                cv2.putText(overlay, text, (20, 35 + i*20), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Save main debug image (always under translated_images/debug when enabled)
            debug_path = os.path.join(debug_dir, f"{base_name}_debug_regions.png")
            cv2.imwrite(debug_path, overlay)
            self._log(f"   📸 Saved debug image: {debug_path}")
            
            # Save text mask
            mask = self.create_text_mask(img, regions)
            mask_debug_path = debug_path.replace('_debug', '_mask')
            cv2.imwrite(mask_debug_path, mask)
            mask_percentage = ((mask > 0).sum() / mask.size) * 100
            self._log(f"   🎭 Saved mask image: {mask_debug_path}", "info")
            self._log(f"   📊 Mask coverage: {mask_percentage:.1f}% of image", "info")
                        
            # If save_intermediate is enabled, save additional debug images
            if self.manga_settings.get('advanced', {}).get('save_intermediate', False):
                # Save confidence heatmap
                heatmap = self._create_confidence_heatmap(img, regions)
                heatmap_path = os.path.join(debug_dir, f"{base_name}_confidence_heatmap.png")
                cv2.imwrite(heatmap_path, heatmap)
                self._log(f"   🌡️ Saved confidence heatmap: {heatmap_path}")
                
                # Save polygon visualization with safe text areas
                if any(hasattr(r, 'vertices') and r.vertices for r in regions):
                    polygon_img = img.copy()
                    for region in regions:
                        if hasattr(region, 'vertices') and region.vertices:
                            # Draw polygon
                            pts = np.array(region.vertices, np.int32)
                            pts = pts.reshape((-1, 1, 2))
                            
                            # Fill with transparency
                            overlay_poly = polygon_img.copy()
                            cv2.fillPoly(overlay_poly, [pts], (0, 255, 255))
                            cv2.addWeighted(overlay_poly, 0.2, polygon_img, 0.8, 0, polygon_img)
                            
                            # Draw outline
                            cv2.polylines(polygon_img, [pts], True, (255, 0, 0), 2)
                            
                            # Draw safe text area
                            try:
                                safe_x, safe_y, safe_w, safe_h = self.get_safe_text_area(region)
                                cv2.rectangle(polygon_img, (safe_x, safe_y), 
                                            (safe_x + safe_w, safe_y + safe_h), 
                                            (0, 255, 0), 1)
                            except:
                                pass  # Skip if get_safe_text_area fails
                    
                    polygon_path = os.path.join(debug_dir, f"{base_name}_polygons.png")
                    cv2.imwrite(polygon_path, polygon_img)
                    self._log(f"   🔷 Saved polygon visualization: {polygon_path}")
                
                # Save individual region crops with more info
                regions_dir = os.path.join(debug_dir, 'regions')
                os.makedirs(regions_dir, exist_ok=True)
                
                for i, region in enumerate(regions[:10]):  # Limit to first 10 regions
                    x, y, w, h = region.bounding_box
                    # Add padding
                    pad = 10
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(img.shape[1], x + w + pad)
                    y2 = min(img.shape[0], y + h + pad)
                    
                    region_crop = img[y1:y2, x1:x2].copy()
                    
                    # Draw bounding box on crop
                    cv2.rectangle(region_crop, (pad, pad), 
                                (pad + w, pad + h), (0, 255, 0), 2)
                    
                    # Add text info on the crop
                    info = f"Conf: {region.confidence:.2f} | Chars: {len(region.text)}"
                    cv2.putText(region_crop, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                               0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Save with meaningful filename
                    safe_text = region.text[:20].replace('/', '_').replace('\\', '_').strip()
                    region_path = os.path.join(regions_dir, f"region_{i:03d}_{safe_text}.png")
                    cv2.imwrite(region_path, region_crop)
                
                self._log(f"   📁 Saved individual region crops to: {regions_dir}")
            
        except Exception as e:
            self._log(f"   ❌ Failed to save debug image: {str(e)}", "warning")
            if self.manga_settings.get('advanced', {}).get('debug_mode', False):
                # If debug mode is on, log the full traceback
                import traceback
                self._log(traceback.format_exc(), "warning")

    def _create_confidence_heatmap(self, img, regions):
        """Create a heatmap showing OCR confidence levels"""
        heatmap = np.zeros_like(img[:, :, 0], dtype=np.float32)
        
        for region in regions:
            x, y, w, h = region.bounding_box
            confidence = region.confidence
            heatmap[y:y+h, x:x+w] = confidence
        
        # Convert to color heatmap
        heatmap_normalized = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        
        # Blend with original image
        result = cv2.addWeighted(img, 0.7, heatmap_colored, 0.3, 0)
        return result

    def _get_translation_history_context(self) -> List[Dict[str, str]]:
        """Get translation history context from HistoryManager"""
        if not self.history_manager or not self.contextual_enabled:
            return []
        
        try:
            # Load full history
            full_history = self.history_manager.load_history()
            
            if not full_history:
                return []
            
            # Extract only the contextual messages up to the limit
            context = []
            exchange_count = 0
            
            # Process history in pairs (user + assistant messages)
            for i in range(0, len(full_history), 2):
                if i + 1 < len(full_history):
                    user_msg = full_history[i]
                    assistant_msg = full_history[i + 1]
                    
                    if user_msg.get("role") == "user" and assistant_msg.get("role") == "assistant":
                        context.extend([user_msg, assistant_msg])
                        exchange_count += 1
                        
                        # Only keep up to the history limit
                        if exchange_count >= self.translation_history_limit:
                            # Get only the most recent exchanges
                            context = context[-(self.translation_history_limit * 2):]
                            break
            
            return context
            
        except Exception as e:
            self._log(f"⚠️ Error loading history context: {str(e)}", "warning")
            return []
    
    def translate_text(self, text: str, context: Optional[List[Dict]] = None, image_path: str = None, region: TextRegion = None) -> str:
        """Translate text using API with GUI system prompt and full image context"""
        try:
            # Build per-request log prefix for clearer parallel logs
            try:
                import threading
                thread_name = threading.current_thread().name
            except Exception:
                thread_name = "MainThread"
            bbox_info = ""
            try:
                if region and hasattr(region, 'bounding_box') and region.bounding_box:
                    x, y, w, h = region.bounding_box
                    bbox_info = f" [bbox={x},{y},{w}x{h}]"
            except Exception:
                pass
            prefix = f"[{thread_name}]{bbox_info}"
            
            self._log(f"\n{prefix} 🌐 Starting translation for text: '{text[:50]}...'")
            # CHECK 1: Before starting
            if self._check_stop():
                self._log("⏹️ Translation stopped before full page context processing", "warning")
                return {}
            
            # Get system prompt from GUI profile
            profile_name = self.main_gui.profile_var.get()

            # Get the prompt from prompt_profiles dictionary
            system_prompt = ''
            if hasattr(self.main_gui, 'prompt_profiles') and profile_name in self.main_gui.prompt_profiles:
                system_prompt = self.main_gui.prompt_profiles[profile_name]
                self._log(f"📋 Using profile: {profile_name}")
            else:
                self._log(f"⚠️ Profile '{profile_name}' not found in prompt_profiles", "warning")

            self._log(f"{prefix} 📝 System prompt: {system_prompt[:100]}..." if system_prompt else f"{prefix} 📝 No system prompt configured")

            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}]
            else:
                messages = []
            
            
            # Add contextual translations if enabled
            if self.contextual_enabled and self.history_manager:
                # Get history from HistoryManager
                history_context = self._get_translation_history_context()
                
                if history_context:
                    context_count = len(history_context) // 2  # Each exchange is 2 messages
                    self._log(f"🔗 Adding {context_count} previous exchanges from history (limit: {self.translation_history_limit})")
                    messages.extend(history_context)
                else:
                    self._log(f"🔗 Contextual enabled but no history available yet")
            else:
                self._log(f"{prefix} 🔗 Contextual: {'Disabled' if not self.contextual_enabled else 'No HistoryManager'}")
            
            # Add full image context if available AND visual context is enabled
            if image_path and self.visual_context_enabled:
                try:
                    import base64
                    from PIL import Image as PILImage
                    
                    self._log(f"{prefix} 📷 Adding full page visual context for translation")
                    
                    # Read and encode the full image
                    with open(image_path, 'rb') as img_file:
                        img_data = img_file.read()
                    
                    # Check image size
                    img_size_mb = len(img_data) / (1024 * 1024)
                    self._log(f"{prefix} 📊 Image size: {img_size_mb:.2f} MB")
                    
                    # Optionally resize if too large (Gemini has limits)
                    if img_size_mb > 10:  # If larger than 10MB
                        self._log(f"📉 Resizing large image for API limits...")
                        pil_image = PILImage.open(image_path)
                        
                        # Calculate new size (max 2048px on longest side)
                        max_size = 2048
                        ratio = min(max_size / pil_image.width, max_size / pil_image.height)
                        if ratio < 1:
                            new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
                            pil_image = pil_image.resize(new_size, PILImage.Resampling.LANCZOS)
                            
                            # Re-encode
                            from io import BytesIO
                            buffered = BytesIO()
                            pil_image.save(buffered, format="PNG", optimize=True)
                            img_data = buffered.getvalue()
                            self._log(f"{prefix} ✅ Resized to {new_size[0]}x{new_size[1]}px ({len(img_data)/(1024*1024):.2f} MB)")
                    
                    # Encode to base64
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    
                    # Build the message with image and text location info
                    location_description = ""
                    if region:
                        x, y, w, h = region.bounding_box
                        # Describe where on the page this text is located
                        page_width = PILImage.open(image_path).width
                        page_height = PILImage.open(image_path).height
                        
                        # Determine position
                        h_pos = "left" if x < page_width/3 else "center" if x < 2*page_width/3 else "right"
                        v_pos = "top" if y < page_height/3 else "middle" if y < 2*page_height/3 else "bottom"
                        
                        location_description = f"\n\nThe text to translate is located in the {v_pos}-{h_pos} area of the page, "
                        location_description += f"at coordinates ({x}, {y}) with size {w}x{h} pixels."
                    
                    # Add image and text to translate
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": f"Looking at this full manga page, translate the following text: '{text}'{location_description}"
                            }
                        ]
                    })
                    
                    self._log(f"{prefix} ✅ Added full page image as visual context")
                    
                except Exception as e:
                    self._log(f"⚠️ Failed to add image context: {str(e)}", "warning")
                    self._log(f"   Error type: {type(e).__name__}", "warning")
                    import traceback
                    self._log(traceback.format_exc(), "warning")
                    # Fall back to text-only translation
                    messages.append({"role": "user", "content": text})
            elif image_path and not self.visual_context_enabled:
                # Visual context disabled - text-only mode
                self._log(f"{prefix} 📝 Text-only mode (visual context disabled)")
                messages.append({"role": "user", "content": text})
            else:
                # No image path provided - text-only translation
                messages.append({"role": "user", "content": text})
            
            # Check input token limit
            text_tokens = 0
            image_tokens = 0

            for msg in messages:
                if isinstance(msg.get("content"), str):
                    # Simple text message
                    text_tokens += len(msg["content"]) // 4
                elif isinstance(msg.get("content"), list):
                    # Message with mixed content (text + image)
                    for content_part in msg["content"]:
                        if content_part.get("type") == "text":
                            text_tokens += len(content_part.get("text", "")) // 4
                        elif content_part.get("type") == "image_url":
                            # Only count image tokens if visual context is enabled
                            if self.visual_context_enabled:
                                image_tokens += 258

            estimated_tokens = text_tokens + image_tokens

            # Check token limit only if it's enabled
            if self.input_token_limit is None:
                self._log(f"{prefix} 📊 Token estimate - Text: {text_tokens}, Images: {image_tokens} (Total: {estimated_tokens} / unlimited)")
            else:
                self._log(f"{prefix} 📊 Token estimate - Text: {text_tokens}, Images: {image_tokens} (Total: {estimated_tokens} / {self.input_token_limit})")
                
                if estimated_tokens > self.input_token_limit:
                    self._log(f"⚠️ Token limit exceeded, trimming context", "warning")
                    # Keep system prompt, image, and current text only
                    if image_path:
                        messages = [messages[0], messages[-1]]  
                    else:
                        messages = [messages[0], {"role": "user", "content": text}]
                    # Recalculate tokens after trimming
                    text_tokens = len(messages[0]["content"]) // 4
                    if isinstance(messages[-1].get("content"), str):
                        text_tokens += len(messages[-1]["content"]) // 4
                    else:
                        text_tokens += len(messages[-1]["content"][0]["text"]) // 4
                    estimated_tokens = text_tokens + image_tokens
                    self._log(f"📊 Trimmed token estimate: {estimated_tokens}")
            
            start_time = time.time()
            api_time = 0  # Initialize to avoid NameError
            
            try:
                response = send_with_interrupt(
                    messages=messages,
                    client=self.client,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop_check_fn=self._check_stop

                )
                api_time = time.time() - start_time
                self._log(f"{prefix} ✅ API responded in {api_time:.2f} seconds")

                # Normalize response to plain text (handle tuples and bytes)
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = response

                # Handle tuple response like (text, 'stop') from some clients
                if isinstance(response_text, tuple):
                    response_text = response_text[0]

                # Decode bytes/bytearray
                if isinstance(response_text, (bytes, bytearray)):
                    try:
                        response_text = response_text.decode('utf-8', errors='replace')
                    except Exception:
                        response_text = str(response_text)

                # Ensure string
                if not isinstance(response_text, str):
                    response_text = str(response_text)

                response_text = response_text.strip()

                # If it's a stringified tuple like "('text', 'stop')", extract the first element
                if response_text.startswith("('") or response_text.startswith('("'):
                    import ast, re
                    try:
                        parsed_tuple = ast.literal_eval(response_text)
                        if isinstance(parsed_tuple, tuple) and parsed_tuple:
                            response_text = str(parsed_tuple[0])
                            self._log("📦 Extracted response from tuple literal", "debug")
                    except Exception:
                        match = re.match(r"^\('(.+?)',\s*'.*'\)$", response_text, re.DOTALL)
                        if match:
                            tmp = match.group(1)
                            tmp = tmp.replace('\\n', '\n').replace("\\'", "'").replace('\\\"', '"').replace('\\\\', '\\')
                            response_text = tmp
                            self._log("📦 Extracted response using regex from tuple literal", "debug")

                self._log(f"{prefix} 📥 Received response ({len(response_text)} chars)")
                
            except Exception as api_error:
                api_time = time.time() - start_time
                error_str = str(api_error).lower()
                error_type = type(api_error).__name__
                
                # Check for specific error types
                if "429" in error_str or "rate limit" in error_str:
                    self._log(f"⚠️ RATE LIMIT ERROR (429) after {api_time:.2f}s", "error")
                    self._log(f"   The API rate limit has been exceeded", "error")
                    self._log(f"   Please wait before retrying or reduce request frequency", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    raise Exception(f"Rate limit exceeded (429): {str(api_error)}")
                    
                elif "401" in error_str or "unauthorized" in error_str:
                    self._log(f"❌ AUTHENTICATION ERROR (401) after {api_time:.2f}s", "error")
                    self._log(f"   Invalid API key or authentication failed", "error")
                    self._log(f"   Please check your API key in settings", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    raise Exception(f"Authentication failed (401): {str(api_error)}")
                    
                elif "403" in error_str or "forbidden" in error_str:
                    self._log(f"❌ FORBIDDEN ERROR (403) after {api_time:.2f}s", "error")
                    self._log(f"   Access denied - check API permissions", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    raise Exception(f"Access forbidden (403): {str(api_error)}")
                    
                elif "400" in error_str or "bad request" in error_str:
                    self._log(f"❌ BAD REQUEST ERROR (400) after {api_time:.2f}s", "error")
                    self._log(f"   Invalid request format or parameters", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    raise Exception(f"Bad request (400): {str(api_error)}")
                    
                elif "timeout" in error_str:
                    self._log(f"⏱️ TIMEOUT ERROR after {api_time:.2f}s", "error")
                    self._log(f"   API request timed out", "error")
                    self._log(f"   Consider increasing timeout or retry", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    raise Exception(f"Request timeout: {str(api_error)}")
                    
                else:
                    # Generic API error
                    self._log(f"❌ API ERROR ({error_type}) after {api_time:.2f}s", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    self._log(f"   Full traceback:", "error")
                    self._log(traceback.format_exc(), "error")
                    raise
            
            

            # Initialize translated with extracted response text to avoid UnboundLocalError
            if response_text is None:
                translated = ""
            elif isinstance(response_text, str):
                translated = response_text
            elif isinstance(response_text, (bytes, bytearray)):
                try:
                    translated = response_text.decode('utf-8', errors='replace')
                except Exception:
                    translated = str(response_text)
            else:
                translated = str(response_text)

            # ADD THIS DEBUG CODE:
            self._log(f"🔍 RAW API RESPONSE DEBUG:", "debug")
            self._log(f"  Type: {type(translated)}", "debug")
            #self._log(f"  Raw content length: {len(translated)}", "debug")
            #self._log(f"  First 200 chars: {translated[:200]}", "debug")
            #self._log(f"  Last 200 chars: {translated[-200:]}", "debug")

            # Check if both Japanese and English are present
            has_japanese = any('\u3040' <= c <= '\u9fff' or '\uac00' <= c <= '\ud7af' for c in translated)
            has_english = any('a' <= c.lower() <= 'z' for c in translated)

            if has_japanese and has_english:
                self._log(f"  ⚠️ WARNING: Response contains BOTH Japanese AND English!", "warning")
                self._log(f"  This might be causing the duplicate text issue", "warning")

            # Check if response looks like JSON (contains both { and } and : characters)
            if '{' in translated and '}' in translated and ':' in translated:
                try:
                    # It might be JSON, try to fix and parse it
                    fixed_json = self._fix_json_response(translated)
                    import json
                    parsed = json.loads(fixed_json)
                    
                    # If it's a dict with a single translation, extract it
                    if isinstance(parsed, dict) and len(parsed) == 1:
                        translated = list(parsed.values())[0]
                        translated = self._clean_translation_text(translated)
                        self._log("📦 Extracted translation from JSON response", "debug")
                except:
                    # Not JSON or failed to parse, use as-is
                    pass
            
            self._log(f"{prefix} 🔍 Raw response type: {type(translated)}")
            self._log(f"{prefix} 🔍 Raw response content: '{translated[:5000]}...'")
            
            # Check if the response looks like a Python literal (tuple/string representation)
            if translated.startswith("('") or translated.startswith('("') or translated.startswith("('''"):
                self._log(f"⚠️ Detected Python literal in response, attempting to extract actual text", "warning")
                original = translated
                try:
                    # Try to evaluate it as a Python literal
                    import ast
                    evaluated = ast.literal_eval(translated)
                    self._log(f"📦 Evaluated type: {type(evaluated)}")
                    
                    if isinstance(evaluated, tuple):
                        # Take the first element of the tuple
                        translated = str(evaluated[0])
                        self._log(f"📦 Extracted from tuple: '{translated[:50]}...'")
                    elif isinstance(evaluated, str):
                        translated = evaluated
                        self._log(f"📦 Extracted string: '{translated[:50]}...'")
                    else:
                        self._log(f"⚠️ Unexpected type after eval: {type(evaluated)}", "warning")
                        
                except Exception as e:
                    self._log(f"⚠️ Failed to parse Python literal: {e}", "warning")
                    self._log(f"⚠️ Original content: {original[:200]}", "warning")
                    
                    # Try multiple levels of unescaping
                    temp = translated
                    for i in range(5):  # Try up to 5 levels of unescaping
                        if temp.startswith("('") or temp.startswith('("'):
                            # Try regex as fallback
                            import re
                            match = re.search(r"^\(['\"](.+)['\"]\)$", temp, re.DOTALL)
                            if match:
                                temp = match.group(1)
                                self._log(f"📦 Regex extracted (level {i+1}): '{temp[:50]}...'")
                            else:
                                break
                        else:
                            break
                    translated = temp

            # Additional check for escaped content
            #if '\\\\' in translated or '\\n' in translated or "\\'" in translated or '\\"' in translated:
            #    self._log(f"⚠️ Detected escaped content, unescaping...", "warning")
            #    try:
            #        before = translated
            #        
            #        # Handle quotes and apostrophes
            #        translated = translated.replace("\\'", "'")
            #        translated = translated.replace('\\"', '"')
            #        translated = translated.replace("\\`", "`")
                    
                    # DON'T UNESCAPE NEWLINES BEFORE JSON PARSING!
                    # translated = translated.replace('\\n', '\n')  # COMMENT THIS OUT
                    
            #        translated = translated.replace('\\\\', '\\')
            #        translated = translated.replace('\\/', '/')
                    # translated = translated.replace('\\t', '\t')  # COMMENT THIS OUT TOO
                    # translated = translated.replace('\\r', '\r')  # AND THIS
                    
            #        self._log(f"📦 Unescaped safely: '{before[:50]}...' -> '{translated[:50]}...'")
            #    except Exception as e:
            #        self._log(f"⚠️ Failed to unescape: {e}", "warning")
            
            # Clean up unwanted trailing apostrophes/quotes
            import re
            response_text = translated
            response_text = re.sub(r"['''\"`]$", "", response_text.strip())  # Remove trailing
            response_text = re.sub(r"^['''\"`]", "", response_text.strip())   # Remove leading
            response_text = re.sub(r"\s+['''\"`]\s+", " ", response_text)     # Remove isolated
            translated = response_text
            translated = self._clean_translation_text(translated)
            
            # Apply glossary if available
            if hasattr(self.main_gui, 'manual_glossary') and self.main_gui.manual_glossary:
                glossary_count = len(self.main_gui.manual_glossary)
                self._log(f"📚 Applying glossary with {glossary_count} entries")
                
                replacements = 0
                for entry in self.main_gui.manual_glossary:
                    if 'source' in entry and 'target' in entry:
                        if entry['source'] in translated:
                            translated = translated.replace(entry['source'], entry['target'])
                            replacements += 1
                
                if replacements > 0:
                    self._log(f"   ✏️ Made {replacements} glossary replacements")
            
            translated = self._clean_translation_text(translated)
            
            # Store in history if HistoryManager is available
            if self.history_manager and self.contextual_enabled:
                try:
                    # Append to history with proper limit handling
                    self.history_manager.append_to_history(
                        user_content=text,
                        assistant_content=translated,
                        hist_limit=self.translation_history_limit,
                        reset_on_limit=not self.rolling_history_enabled,
                        rolling_window=self.rolling_history_enabled
                    )
                    
                    # Check if we're about to hit the limit
                    if self.history_manager.will_reset_on_next_append(
                        self.translation_history_limit, 
                        self.rolling_history_enabled
                    ):
                        mode = "roll over" if self.rolling_history_enabled else "reset"
                        self._log(f"📚 History will {mode} on next translation (at limit: {self.translation_history_limit})")
                    
                except Exception as e:
                    self._log(f"⚠️ Failed to save to history: {str(e)}", "warning")
            
            # Also store in legacy context for compatibility
            self.translation_context.append({
                "original": text,
                "translated": translated
            })
            
            return translated
            
        except Exception as e:
            self._log(f"❌ Translation error: {str(e)}", "error")
            self._log(f"   Error type: {type(e).__name__}", "error")
            import traceback
            self._log(f"   Traceback: {traceback.format_exc()}", "error")
            return text

    def translate_full_page_context(self, regions: List[TextRegion], image_path: str) -> Dict[str, str]:
        """Translate all text regions with full page context in a single request"""
        try:
            import time
            import traceback
            import json
            
            # Initialize response_text at the start
            response_text = ""
            
            self._log(f"\n📄 Full page context translation of {len(regions)} text regions")
            
            # Get system prompt from GUI profile
            profile_name = self.main_gui.profile_var.get()
            
            # Ensure visual_context_enabled exists (temporary fix)
            if not hasattr(self, 'visual_context_enabled'):
                self.visual_context_enabled = self.main_gui.config.get('manga_visual_context_enabled', True)
            
            # Try to get the prompt from prompt_profiles dictionary (for all profiles including custom ones)
            system_prompt = ''
            if hasattr(self.main_gui, 'prompt_profiles') and profile_name in self.main_gui.prompt_profiles:
                system_prompt = self.main_gui.prompt_profiles[profile_name]
                self._log(f"📋 Using profile: {profile_name}")
            else:
                # Fallback to check if it's stored as a direct attribute (legacy support)
                system_prompt = getattr(self.main_gui, profile_name.replace(' ', '_'), '')
                if system_prompt:
                    self._log(f"📋 Using profile (legacy): {profile_name}")
                else:
                    self._log(f"⚠️ Profile '{profile_name}' not found, using empty prompt", "warning")
            
            # Combine with full page context instructions
            if system_prompt:
                system_prompt = f"{system_prompt}\n\n{self.full_page_context_prompt}"
            else:
                system_prompt = self.full_page_context_prompt
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # CHECK 2: Before adding context
            if self._check_stop():
                self._log("⏹️ Translation stopped during context preparation", "warning")
                return {}
            
            # Add contextual translations if enabled
            if self.contextual_enabled and self.history_manager:
                history_context = self._get_translation_history_context()
                if history_context:
                    context_count = len(history_context) // 2
                    self._log(f"🔗 Adding {context_count} previous exchanges from history")
                    messages.extend(history_context)
            
            # Prepare text segments with indices
            all_texts = {}
            text_list = []
            for i, region in enumerate(regions):
                # Use index-based key to handle duplicate texts
                key = f"[{i}] {region.text}"
                all_texts[key] = region.text
                text_list.append(f"{key}")
                
            # CHECK 3: Before image processing
            if self._check_stop():
                self._log("⏹️ Translation stopped before image processing", "warning")
                return {}
            
            # Create the full context message text
            context_text = "\n".join(text_list)
            
            # Log text content info
            total_chars = sum(len(region.text) for region in regions)
            self._log(f"📝 Text content: {len(regions)} regions, {total_chars} total characters")
            
            # Process image if visual context is enabled
            if self.visual_context_enabled:
                try:
                    import base64
                    from PIL import Image as PILImage
                    
                    self._log(f"📷 Adding full page visual context for translation")
                    
                    # Read and encode the image
                    with open(image_path, 'rb') as img_file:
                        img_data = img_file.read()
                    
                    # Check image size
                    img_size_mb = len(img_data) / (1024 * 1024)
                    self._log(f"📊 Image size: {img_size_mb:.2f} MB")
                    
                    # Get image dimensions
                    pil_image = PILImage.open(image_path)
                    self._log(f"   Image dimensions: {pil_image.width}x{pil_image.height}")
                    
                    # CHECK 4: Before resizing (which can take time)
                    if self._check_stop():
                        self._log("⏹️ Translation stopped during image preparation", "warning")
                        return {}
                    
                    # Resize if needed
                    if img_size_mb > 10:
                        self._log(f"📉 Resizing large image for API limits...")
                        max_size = 2048
                        ratio = min(max_size / pil_image.width, max_size / pil_image.height)
                        if ratio < 1:
                            new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
                            pil_image = pil_image.resize(new_size, PILImage.Resampling.LANCZOS)
                            from io import BytesIO
                            buffered = BytesIO()
                            pil_image.save(buffered, format="PNG", optimize=True)
                            img_data = buffered.getvalue()
                            self._log(f"✅ Resized to {new_size[0]}x{new_size[1]}px ({len(img_data)/(1024*1024):.2f} MB)")
                    
                    # Convert to base64
                    img_b64 = base64.b64encode(img_data).decode('utf-8')
                    
                    # Create message with both text and image
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": context_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]
                    })
                    
                    self._log(f"✅ Added full page image as visual context")
                    
                except Exception as e:
                    self._log(f"⚠️ Failed to add image context: {str(e)}", "warning")
                    self._log(f"   Error type: {type(e).__name__}", "warning")
                    import traceback
                    self._log(traceback.format_exc(), "warning")
                    self._log(f"   Falling back to text-only translation", "warning")
                    
                    # Fall back to text-only translation
                    messages.append({"role": "user", "content": context_text})
            else:
                # Visual context disabled - send text only
                self._log(f"📝 Text-only mode (visual context disabled for non-vision models)")
                messages.append({"role": "user", "content": context_text})
            
            # CHECK 5: Before API call
            if self._check_stop():
                self._log("⏹️ Translation stopped before API call", "warning")
                return {}
            
            # Check input token limit
            text_tokens = 0
            image_tokens = 0

            for msg in messages:
                if isinstance(msg.get("content"), str):
                    # Simple text message
                    text_tokens += len(msg["content"]) // 4
                elif isinstance(msg.get("content"), list):
                    # Message with mixed content (text + image)
                    for content_part in msg["content"]:
                        if content_part.get("type") == "text":
                            text_tokens += len(content_part.get("text", "")) // 4
                        elif content_part.get("type") == "image_url":
                            # Only count image tokens if visual context is enabled
                            if self.visual_context_enabled:
                                image_tokens += 258

            estimated_tokens = text_tokens + image_tokens

            # Check token limit only if it's enabled
            if self.input_token_limit is None:
                self._log(f"📊 Token estimate - Text: {text_tokens}, Images: {image_tokens} (Total: {estimated_tokens} / unlimited)")
            else:
                self._log(f"📊 Token estimate - Text: {text_tokens}, Images: {image_tokens} (Total: {estimated_tokens} / {self.input_token_limit})")
                
                if estimated_tokens > self.input_token_limit:
                    self._log(f"⚠️ Token limit exceeded, trimming context", "warning")
                    # Keep system prompt and current message only
                    messages = [messages[0], messages[-1]]  
                    # Recalculate tokens
                    text_tokens = len(messages[0]["content"]) // 4
                    if isinstance(messages[-1]["content"], str):
                        text_tokens += len(messages[-1]["content"]) // 4
                    else:
                        for content_part in messages[-1]["content"]:
                            if content_part.get("type") == "text":
                                text_tokens += len(content_part.get("text", "")) // 4
                    estimated_tokens = text_tokens + image_tokens
                    self._log(f"📊 Trimmed token estimate: {estimated_tokens}")
            
            # Make API call using the client's send method (matching translate_text)
            self._log(f"🌐 Sending full page context to API...")
            self._log(f"   API Model: {self.client.model if hasattr(self.client, 'model') else 'unknown'}")
            self._log(f"   Temperature: {self.temperature}")
            self._log(f"   Max Output Tokens: {self.max_tokens}")
            
            start_time = time.time()
            api_time = 0  # Initialize to avoid NameError
            
            try:
                response = send_with_interrupt(
                    messages=messages,
                    client=self.client,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop_check_fn=self._check_stop
                )
                api_time = time.time() - start_time

                # Extract content from response
                if hasattr(response, 'content'):
                    response_text = response.content
                    # Check if it's a tuple representation
                    if isinstance(response_text, tuple):
                        response_text = response_text[0]  # Get first element of tuple
                    response_text = response_text.strip()
                elif hasattr(response, 'text'):
                    # Gemini responses have .text attribute
                    response_text = response.text.strip()
                elif hasattr(response, 'candidates') and response.candidates:
                    # Handle Gemini GenerateContentResponse structure
                    try:
                        response_text = response.candidates[0].content.parts[0].text.strip()
                    except (IndexError, AttributeError):
                        response_text = str(response).strip()
                else:
                    # If response is a string or other format
                    response_text = str(response).strip()
                    
                    # Check if it's a stringified tuple
                    if response_text.startswith("('") or response_text.startswith('("'):
                        # It's a tuple converted to string, extract the JSON part
                        import ast
                        try:
                            parsed_tuple = ast.literal_eval(response_text)
                            if isinstance(parsed_tuple, tuple):
                                response_text = parsed_tuple[0]  # Get first element
                                self._log("📦 Extracted response from tuple format", "debug")
                        except:
                            # If literal_eval fails, try regex
                            import re
                            match = re.match(r"^\('(.+)', '.*'\)$", response_text, re.DOTALL)
                            if match:
                                response_text = match.group(1)
                                # Unescape the string
                                response_text = response_text.replace('\\n', '\n')
                                response_text = response_text.replace("\\'", "'")
                                response_text = response_text.replace('\\"', '"')
                                response_text = response_text.replace('\\\\', '\\')
                                self._log("📦 Extracted response using regex from tuple string", "debug")
                
                # CHECK 6: Immediately after API response
                if self._check_stop():
                    self._log(f"⏹️ Translation stopped after API call ({api_time:.2f}s)", "warning")
                    return {}
                
                self._log(f"✅ API responded in {api_time:.2f} seconds")
                self._log(f"📥 Received response ({len(response_text)} chars)")
                
            except Exception as api_error:
                api_time = time.time() - start_time
                
                # CHECK 7: After API error
                if self._check_stop():
                    self._log(f"⏹️ Translation stopped during API error handling", "warning")
                    return {}
                
                error_str = str(api_error).lower()
                error_type = type(api_error).__name__
                
                # Check for specific error types
                if "429" in error_str or "rate limit" in error_str:
                    self._log(f"⚠️ RATE LIMIT ERROR (429) after {api_time:.2f}s", "error")
                    self._log(f"   The API rate limit has been exceeded", "error")
                    self._log(f"   Please wait before retrying or reduce request frequency", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    raise Exception(f"Rate limit exceeded (429): {str(api_error)}")
                    
                elif "401" in error_str or "unauthorized" in error_str:
                    self._log(f"❌ AUTHENTICATION ERROR (401) after {api_time:.2f}s", "error")
                    self._log(f"   Invalid API key or authentication failed", "error")
                    self._log(f"   Please check your API key in settings", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    raise Exception(f"Authentication failed (401): {str(api_error)}")
                    
                elif "403" in error_str or "forbidden" in error_str:
                    self._log(f"❌ FORBIDDEN ERROR (403) after {api_time:.2f}s", "error")
                    self._log(f"   Access denied - check API permissions", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    raise Exception(f"Access forbidden (403): {str(api_error)}")
                    
                elif "400" in error_str or "bad request" in error_str:
                    self._log(f"❌ BAD REQUEST ERROR (400) after {api_time:.2f}s", "error")
                    self._log(f"   Invalid request format or parameters", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    raise Exception(f"Bad request (400): {str(api_error)}")
                    
                elif "timeout" in error_str:
                    self._log(f"⏱️ TIMEOUT ERROR after {api_time:.2f}s", "error")
                    self._log(f"   API request timed out", "error")
                    self._log(f"   Consider increasing timeout or retry", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    raise Exception(f"Request timeout: {str(api_error)}")
                    
                else:
                    # Generic API error
                    self._log(f"❌ API ERROR ({error_type}) after {api_time:.2f}s", "error")
                    self._log(f"   Error details: {str(api_error)}", "error")
                    self._log(f"   Full traceback:", "error")
                    self._log(traceback.format_exc(), "error")
                    raise
            
            # CHECK 8: Before parsing response
            if self._check_stop():
                self._log("⏹️ Translation stopped before parsing response", "warning")
                return {}
            
            # Check if we got a response
            if not response_text:
                self._log("❌ Empty response from API", "error")
                return {}
            
            self._log(f"🔍 Raw response type: {type(response_text)}")
            self._log(f"🔍 Raw response preview: '{response_text[:2000]}...'")
            
            # Clean up response_text (handle Python literals, escapes, etc.)
            if response_text.startswith("('") or response_text.startswith('("') or response_text.startswith("('''"):
                self._log(f"⚠️ Detected Python literal in response, attempting to extract actual text", "warning")
                try:
                    import ast
                    evaluated = ast.literal_eval(response_text)
                    if isinstance(evaluated, tuple):
                        response_text = str(evaluated[0])
                    elif isinstance(evaluated, str):
                        response_text = evaluated
                except Exception as e:
                    self._log(f"⚠️ Failed to parse Python literal: {e}", "warning")
            
            # Handle escaped content
            #if '\\\\' in response_text or '\\n' in response_text or "\\'" in response_text or '\\"' in response_text:
            #    self._log(f"⚠️ Detected escaped content, unescaping...", "warning")
            #    response_text = response_text.replace("\\'", "'")
            #    response_text = response_text.replace('\\"', '"')
            #    response_text = response_text.replace('\\n', '\n')
            #    response_text = response_text.replace('\\\\', '\\')
            #    response_text = response_text.replace('\\/', '/')
            #    response_text = response_text.replace('\\t', '\t')
            #    response_text = response_text.replace('\\r', '\r')
            
            # Clean up quotes
            import re
            response_text = re.sub(r"['''\"`]$", "", response_text.strip())
            response_text = re.sub(r"^['''\"`]", "", response_text.strip())
            response_text = re.sub(r"\s+['''\"`]\s+", " ", response_text)
            
            # Try to parse as JSON
            translations = {}
            try:
                # Strip markdown blocks more aggressively
                import re
                import json
                
                # Method 1: Find JSON object directly (most reliable)
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                    try:
                        translations = json.loads(json_text)
                        self._log(f"✅ Successfully parsed {len(translations)} translations (direct extraction)")
                    except json.JSONDecodeError:
                        # Try to fix the extracted JSON
                        json_text = self._fix_json_response(json_text)
                        translations = json.loads(json_text)
                        self._log(f"✅ Successfully parsed {len(translations)} translations (after fix)")
                else:
                    # Method 2: Try stripping markdown if no JSON found
                    cleaned = response_text
                    
                    # Remove markdown code blocks
                    if '```' in cleaned:
                        # This pattern handles ```json, ``json, ``` or ``
                        patterns = [
                            r'```json\s*\n?(.*?)```',
                            r'``json\s*\n?(.*?)``',
                            r'```\s*\n?(.*?)```',
                            r'``\s*\n?(.*?)``'
                        ]
                        
                        for pattern in patterns:
                            match = re.search(pattern, cleaned, re.DOTALL)
                            if match:
                                cleaned = match.group(1).strip()
                                break
                    
                    # Try to parse the cleaned text
                    translations = json.loads(cleaned)
                    self._log(f"✅ Successfully parsed {len(translations)} translations (after markdown strip)")
                
                # Handle different response formats
                if isinstance(translations, list):
                    # Array of translations only - map by position
                    temp = {}
                    for i, region in enumerate(regions):
                        if i < len(translations):
                            temp[region.text] = translations[i]
                    translations = temp
                
                self._log(f"📊 Total translations: {len(translations)}")
                
            except Exception as e:
                self._log(f"❌ Failed to parse JSON: {str(e)}", "error")
                self._log(f"Response preview: {response_text[:500]}...", "warning")
                
                # CRITICAL: Check if this is a refusal message BEFORE regex fallback
                # OpenAI and other APIs refuse certain content with text responses instead of JSON
                # ONLY check if response looks like plain text refusal (not malformed JSON with translations)
                import re
                response_lower = response_text.lower()
                
                # Skip refusal check if response contains valid-looking JSON structure with translations
                # (indicates malformed JSON that should go to regex fallback, not a refusal)
                has_json_structure = (
                    (response_text.strip().startswith('{') and ':' in response_text and '"' in response_text) or
                    (response_text.strip().startswith('[') and ':' in response_text and '"' in response_text)
                )
                
                # Also check if response contains short translations (not refusal paragraphs)
                # Refusals are typically long paragraphs, translations are short
                avg_value_length = 0
                if has_json_structure:
                    # Quick estimate: count chars between quotes
                    import re
                    values = re.findall(r'"([^"]{1,200})"\s*[,}]', response_text)
                    if values:
                        avg_value_length = sum(len(v) for v in values) / len(values)
                
                # If looks like JSON with short values, skip refusal check (go to regex fallback)
                if has_json_structure and avg_value_length > 0 and avg_value_length < 150:
                    self._log(f"🔍 Detected malformed JSON with translations (avg len: {avg_value_length:.0f}), trying regex fallback", "debug")
                    # Skip refusal detection, go straight to regex fallback
                    pass
                else:
                    # Check for refusal patterns
                    # Refusal patterns - both simple strings and regex patterns
                    # Must be strict to avoid false positives on valid translations
                    refusal_patterns = [
                        "i cannot assist",
                        "i can't assist",
                        "i cannot help",
                        "i can't help",
                        r"sorry.{0,10}i can't (assist|help|translate)",  # OpenAI specific
                        "i'm unable to translate",
                        "i am unable to translate",
                        "i apologize, but i cannot",
                        "i'm sorry, but i cannot",
                        "i don't have the ability to",
                        "this request cannot be",
                        "unable to process this",
                        "cannot complete this",
                        r"against.{0,20}(content )?policy",  # "against policy" or "against content policy"
                        "violates.*policy",
                        r"(can't|cannot).{0,30}(sexual|explicit|inappropriate)",  # "can't translate sexual"
                        "appears to sexualize",
                        "who appear to be",
                        "prohibited content",
                        "content blocked",
                    ]
                    
                    # Check both simple string matching and regex patterns
                    is_refusal = False
                    for pattern in refusal_patterns:
                        if '.*' in pattern or r'.{' in pattern:
                            # It's a regex pattern
                            if re.search(pattern, response_lower):
                                is_refusal = True
                                break
                        else:
                            # Simple string match
                            if pattern in response_lower:
                                is_refusal = True
                                break
                    
                    if is_refusal:
                        # Raise UnifiedClientError with prohibited_content type
                        # Fallback mechanism will handle this automatically
                        from unified_api_client import UnifiedClientError
                        raise UnifiedClientError(
                            f"Content refused by API",
                            error_type="prohibited_content",
                            details={"refusal_message": response_text[:500]}
                        )
                
                # Fallback: try regex extraction (handles both quoted and unquoted keys)
                try:
                    import re
                    translations = {}
                    
                    # Try 1: Standard quoted keys and values
                    pattern1 = r'"([^"]+)"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
                    matches = re.findall(pattern1, response_text)
                    
                    if matches:
                        for key, value in matches:
                            value = value.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                            translations[key] = value
                        self._log(f"✅ Recovered {len(translations)} translations using regex (quoted keys)")
                    else:
                        # Try 2: Unquoted keys (for invalid JSON like: key: "value")
                        pattern2 = r'([^\s:{}]+)\s*:\s*([^\n}]+)'
                        matches = re.findall(pattern2, response_text)
                        
                        for key, value in matches:
                            # Clean up key and value
                            key = key.strip()
                            value = value.strip().rstrip(',')
                            # Remove quotes from value if present
                            if value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                            elif value.startswith("'") and value.endswith("'"):
                                value = value[1:-1]
                            translations[key] = value
                        
                        if translations:
                            self._log(f"✅ Recovered {len(translations)} translations using regex (unquoted keys)")
                    
                    if not translations:
                        self._log("❌ All parsing attempts failed", "error")
                        return {}
                except Exception as e:
                    self._log(f"❌ Failed to recover JSON: {e}", "error")
                    return {}
            
            # Map translations back to regions
            result = {}
            all_originals = []
            all_translations = []

            # Extract translation values in order
            translation_values = list(translations.values()) if translations else []

            # DEBUG: Log what we extracted
            self._log(f"📊 Extracted {len(translation_values)} translation values", "debug")
            for i, val in enumerate(translation_values[:1000]):  # First 1000 for debugging
                # Safely handle None values
                val_str = str(val) if val is not None else ""
                self._log(f"  Translation {i}: '{val_str[:1000]}...'", "debug")

            # Clean all translation values to remove quotes
            translation_values = [self._clean_translation_text(t) for t in translation_values]

            self._log(f"🔍 DEBUG: translation_values after cleaning:", "debug")
            for i, val in enumerate(translation_values):
                self._log(f"  [{i}]: {repr(val)}", "debug")
            
            # CRITICAL: Check if translation values are actually refusal messages
            # API sometimes returns valid JSON where each "translation" is a refusal
            if translation_values:
                # Check first few translations for refusal patterns
                import re
                refusal_patterns = [
                    "i cannot",
                    "i can't",
                    r"sorry.{0,5}i can't help",
                    r"sorry.{0,5}i can't",
                    "sexually explicit",
                    "content policy",
                    "prohibited content",
                    "appears to be",
                    "who appear to be",
                ]
                
                # Sample first 3 translations (or all if fewer)
                sample_size = min(3, len(translation_values))
                refusal_count = 0
                
                for sample_val in translation_values[:sample_size]:
                    if sample_val:
                        val_lower = sample_val.lower()
                        for pattern in refusal_patterns:
                            if '.*' in pattern or r'.{' in pattern:
                                if re.search(pattern, val_lower):
                                    refusal_count += 1
                                    break
                            else:
                                if pattern in val_lower:
                                    refusal_count += 1
                                    break
                
                # If most translations are refusals, treat as refusal
                if refusal_count >= sample_size * 0.5:  # 50% threshold
                    # Raise UnifiedClientError with prohibited_content type
                    # Fallback mechanism will handle this automatically
                    from unified_api_client import UnifiedClientError
                    raise UnifiedClientError(
                        f"Content refused by API",
                        error_type="prohibited_content",
                        details={"refusal_message": translation_values[0][:500]}
                    )

            # Position-based mapping
            self._log(f"📋 Mapping {len(translation_values)} translations to {len(regions)} regions")
            
            for i, region in enumerate(regions):
                if i % 10 == 0 and self._check_stop():
                    self._log(f"⏹️ Translation stopped during mapping (processed {i}/{len(regions)} regions)", "warning")
                    return result
                
                # Get translation by position or key
                translated = ""
                
                # Try position-based first
                if i < len(translation_values):
                    translated = translation_values[i]
                # Try key-based fallback
                elif region.text in translations:
                    translated = self._clean_translation_text(translations[region.text])
                # Try indexed key
                else:
                    key = f"[{i}] {region.text}"
                    if key in translations:
                        translated = self._clean_translation_text(translations[key])
                
                # Only mark as missing if we genuinely have no translation
                # NOTE: Keep translation even if it matches original (e.g., numbers, names, SFX)
                if not translated:
                    self._log(f"  ⚠️ No translation for region {i}, leaving empty", "warning")
                    translated = ""
                
                # Apply glossary if we have a translation
                if translated and hasattr(self.main_gui, 'manual_glossary') and self.main_gui.manual_glossary:
                    for entry in self.main_gui.manual_glossary:
                        if 'source' in entry and 'target' in entry:
                            if entry['source'] in translated:
                                translated = translated.replace(entry['source'], entry['target'])
                
                result[region.text] = translated
                region.translated_text = translated
                
                if translated:
                    all_originals.append(f"[{i+1}] {region.text}")
                    all_translations.append(f"[{i+1}] {translated}")
                    self._log(f"  ✅ Translated: '{region.text[:30]}...' → '{translated[:30]}...'", "debug")
            
            # Save history if enabled
            if self.history_manager and self.contextual_enabled and all_originals:
                try:
                    combined_original = "\n".join(all_originals)
                    combined_translation = "\n".join(all_translations)
                    
                    self.history_manager.append_to_history(
                        user_content=combined_original,
                        assistant_content=combined_translation,
                        hist_limit=self.translation_history_limit,
                        reset_on_limit=not self.rolling_history_enabled,
                        rolling_window=self.rolling_history_enabled
                    )
                    
                    self._log(f"📚 Saved {len(all_originals)} translations as 1 combined history entry", "success")
                except Exception as e:
                    self._log(f"⚠️ Failed to save page to history: {str(e)}", "warning")
            
            return result
            
        except Exception as e:
            if self._check_stop():
                self._log("⏹️ Translation stopped due to user request", "warning")
                return {}
            
            # Check if this is a prohibited_content error - re-raise it for fallback handling
            from unified_api_client import UnifiedClientError
            if isinstance(e, UnifiedClientError) and getattr(e, "error_type", None) == "prohibited_content":
                # Re-raise silently for fallback mechanism
                raise
                
            self._log(f"❌ Full page context translation error: {str(e)}", "error")
            self._log(traceback.format_exc(), "error")
            return {}

    def _fix_json_response(self, response_text: str) -> str:
        import re
        import json
        
        # Debug: Show what we received
        self._log(f"DEBUG: Original length: {len(response_text)}", "debug")
        self._log(f"DEBUG: First 50 chars: [{response_text[:50]}]", "debug")
        
        cleaned = response_text
        if "```json" in cleaned:
            match = re.search(r'```json\s*(.*?)```', cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
                self._log(f"DEBUG: Extracted {len(cleaned)} chars from markdown", "debug")
            else:
                self._log("DEBUG: Regex didn't match!", "warning")
        
        # Try to parse
        try:
            result = json.loads(cleaned)
            self._log(f"✅ Parsed JSON with {len(result)} entries", "info")
            return cleaned
        except json.JSONDecodeError as e:
            self._log(f"⚠️ JSON invalid: {str(e)}", "warning")
            self._log(f"DEBUG: Cleaned text starts with: [{cleaned[:20]}]", "debug")
            return cleaned

    def _clean_translation_text(self, text: str) -> str:
        """Remove unnecessary quotation marks, dots, and invalid characters from translated text"""
        if not text:
            return text
        
        # Log what we're cleaning
        original = text
        
        # First, fix encoding issues
        text = self._fix_encoding_issues(text)
        
        # Normalize width/compatibility (e.g., fullwidth → ASCII, circled numbers → digits)
        text = self._normalize_unicode_width(text)
        
        # Remove Unicode replacement characters and invalid symbols
        text = self._sanitize_unicode_characters(text)
        
        # Remove leading and trailing whitespace
        text = text.strip()
        
        # Remove ALL types of quotes and dots from start/end
        # Keep removing until no more quotes/dots at edges
        while len(text) > 0:
            old_len = len(text)
            
            # Remove from start
            text = text.lstrip('"\'`''""「」『』【】《》〈〉.·•°')
            
            # Remove from end (but preserve ... and !!)
            if not text.endswith('...') and not text.endswith('!!'):
                text = text.rstrip('"\'`''""「」『』【】《》〈〉.·•°')
            
            # If nothing changed, we're done
            if len(text) == old_len:
                break
        
        # Final strip
        text = text.strip()
        
        # Log if we made changes
        if text != original:
            self._log(f"🧹 Cleaned text: '{original}' → '{text}'", "debug")
        
        return text
    
    def _sanitize_unicode_characters(self, text: str) -> str:
        """Remove invalid Unicode characters, replacement characters, and box symbols.
        Also more aggressively exclude square-like glyphs that leak as 'cubes' in some fonts.
        """
        if not text:
            return text
        
        import re
        original = text
        
        
        # Remove Unicode replacement character (�) and similar invalid symbols
        text = text.replace('\ufffd', '')  # Unicode replacement character
        
        # Geometric squares and variants (broad sweep)
        geo_squares = [
            '□','■','▢','▣','▤','▥','▦','▧','▨','▩','◻','⬛','⬜',
            '\u25a1','\u25a0','\u2b1c','\u2b1b'
        ]
        for s in geo_squares:
            text = text.replace(s, '')
        
        # Extra cube-like CJK glyphs commonly misrendered in non-CJK fonts
        # (unconditionally removed per user request)
        cube_likes = [
            '口',  # U+53E3
            '囗',  # U+56D7
            '日',  # U+65E5 (often boxy)
            '曰',  # U+66F0
            '田',  # U+7530
            '回',  # U+56DE
            'ロ',  # U+30ED (Katakana RO)
            'ﾛ',  # U+FF9B (Halfwidth RO)
            'ㅁ',  # U+3141 (Hangul MIEUM)
            '丨',  # U+4E28 (CJK radical two) tall bar
        ]
        for s in cube_likes:
            text = text.replace(s, '')
        
        # Remove entire ranges that commonly render as boxes/blocks
        # Box Drawing, Block Elements, Geometric Shapes (full range), plus a common white/black large square range already handled
        text = re.sub(r'[\u2500-\u257F\u2580-\u259F\u25A0-\u25FF]', '', text)
        
        # Optional debug: log culprits found in original text (before removal)
        try:
            culprits = re.findall(r'[\u2500-\u257F\u2580-\u259F\u25A0-\u25FF\u2B1B\u2B1C\u53E3\u56D7\u65E5\u66F0\u7530\u56DE\u30ED\uFF9B\u3141\u4E28]', original)
            if culprits:
                as_codes = [f'U+{ord(c):04X}' for c in culprits]
                self._log(f"🧊 Removed box-like glyphs: {', '.join(as_codes)}", "debug")
        except Exception:
            pass
        
        # If line is mostly ASCII, strip any remaining single CJK ideographs that stand alone
        try:
            ascii_count = sum(1 for ch in text if ord(ch) < 128)
            ratio = ascii_count / max(1, len(text))
            if ratio >= 0.8:
                text = re.sub(r'(?:(?<=\s)|^)[\u3000-\u303F\u3040-\u30FF\u3400-\u9FFF\uFF00-\uFFEF](?=(?:\s)|$)', '', text)
        except Exception:
            pass
        
        # Remove invisible and zero-width characters
        text = re.sub(r'[\u200b-\u200f\u2028-\u202f\u205f-\u206f\ufeff]', '', text)
        
        # Remove remaining control characters (except common ones like newline, tab)
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        # Remove any remaining characters that can't be properly encoded
        try:
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
        except UnicodeError:
            pass
        
        if text != original:
            try:
                self._log(f"🔧 Sanitized Unicode: '{original}' → '{text}'", "debug")
            except Exception:
                pass
        
        return text
    
    def _normalize_unicode_width(self, text: str) -> str:
        """Normalize Unicode to NFKC to 'unsquare' fullwidth/stylized forms while preserving CJK text"""
        if not text:
            return text
        try:
            import unicodedata
            original = text
            # NFKC folds compatibility characters (fullwidth forms, circled digits, etc.) to standard forms
            text = unicodedata.normalize('NFKC', text)
            if text != original:
                try:
                    self._log(f"🔤 Normalized width/compat: '{original[:30]}...' → '{text[:30]}...'", "debug")
                except Exception:
                    pass
            return text
        except Exception:
            return text
    
    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues in text, especially for Korean"""
        if not text:
            return text
        
        # Check for mojibake indicators (UTF-8 misinterpreted as Latin-1)
        mojibake_indicators = ['ë', 'ì', 'ê°', 'ã', 'Ã', 'â', 'ä', 'ð', 'í', 'ë­', 'ì´']
        
        if any(indicator in text for indicator in mojibake_indicators):
            self._log("🔧 Detected mojibake encoding issue, attempting fixes...", "debug")
            
            # Try multiple encoding fixes
            encodings_to_try = [
                ('latin-1', 'utf-8'),
                ('windows-1252', 'utf-8'),
                ('iso-8859-1', 'utf-8'),
                ('cp1252', 'utf-8')
            ]
            
            for from_enc, to_enc in encodings_to_try:
                try:
                    fixed = text.encode(from_enc, errors='ignore').decode(to_enc, errors='ignore')
                    
                    # Check if the fix actually improved things
                    # Should have Korean characters (Hangul range) or be cleaner
                    if any('\uAC00' <= c <= '\uD7AF' for c in fixed) or fixed.count('�') < text.count('�'):
                        self._log(f"✅ Fixed encoding using {from_enc} -> {to_enc}", "debug")
                        return fixed
                except:
                    continue
        
        # Clean up any remaining control characters and replacement characters
        import re
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        # Additional cleanup for common encoding artifacts
        # Remove sequences that commonly appear from encoding errors
        text = re.sub(r'\ufffd+', '', text)  # Remove multiple replacement characters
        text = re.sub(r'[\u25a0-\u25ff]+', '', text)  # Remove geometric shapes (common fallbacks)
        
        # Clean up double spaces and normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
                
    def create_text_mask(self, image: np.ndarray, regions: List[TextRegion]) -> np.ndarray:
        """Create mask with comprehensive per-text-type dilation settings"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        regions_masked = 0
        regions_skipped = 0
        
        self._log(f"🎭 Creating text mask for {len(regions)} regions", "info")
        
        # Get manga settings
        manga_settings = self.main_gui.config.get('manga_settings', {})
        
        # Get dilation settings
        base_dilation_size = manga_settings.get('mask_dilation', 15)
        
        # If Auto Iterations is enabled, auto-set dilation by OCR provider and RT-DETR guide status
        auto_iterations = manga_settings.get('auto_iterations', True)
        if auto_iterations:
            try:
                ocr_settings = manga_settings.get('ocr', {})
                use_rtdetr_guide = ocr_settings.get('use_rtdetr_for_ocr_regions', True)
                bubble_detection_enabled = ocr_settings.get('bubble_detection_enabled', False)
                
                # If RT-DETR guide is enabled for Google/Azure, force dilation to 0
                if (getattr(self, 'ocr_provider', '').lower() in ('azure', 'google') and 
                    bubble_detection_enabled and use_rtdetr_guide):
                    base_dilation_size = 0
                    self._log(f"📏 Auto dilation (RT-DETR guided): 0px (using iterations only)", "info")
                elif getattr(self, 'ocr_provider', '').lower() in ('azure', 'google'):
                    base_dilation_size = 15
                    self._log(f"📏 Auto dilation by provider ({self.ocr_provider}): {base_dilation_size}px", "info")
                else:
                    base_dilation_size = 0
                    self._log(f"📏 Auto dilation by provider ({self.ocr_provider}): {base_dilation_size}px", "info")
            except Exception:
                pass
        
        # Auto iterations: decide by image color vs B&W
        auto_iterations = manga_settings.get('auto_iterations', True)
        if auto_iterations:
            try:
                # Heuristic: consider image B&W if RGB channels are near-equal
                if len(image.shape) < 3 or image.shape[2] == 1:
                    is_bw = True
                else:
                    # Compute mean absolute differences between channels
                    ch0 = image[:, :, 0].astype(np.int16)
                    ch1 = image[:, :, 1].astype(np.int16)
                    ch2 = image[:, :, 2].astype(np.int16)
                    diff01 = np.mean(np.abs(ch0 - ch1))
                    diff12 = np.mean(np.abs(ch1 - ch2))
                    diff02 = np.mean(np.abs(ch0 - ch2))
                    # If channels are essentially the same, treat as B&W
                    is_bw = max(diff01, diff12, diff02) < 2.0
                if is_bw:
                    text_bubble_iterations = 2
                    empty_bubble_iterations = 2
                    free_text_iterations = 0
                    self._log("📏 Auto iterations (B&W): text=2, empty=2, free=0", "info")
                else:
                    text_bubble_iterations = 4
                    empty_bubble_iterations = 4
                    free_text_iterations = 4
                    self._log("📏 Auto iterations (Color): all=3", "info")
            except Exception:
                # Fallback to configured behavior on any error
                auto_iterations = False
        
        if not auto_iterations:
            # Check if using uniform iterations for all text types
            use_all_iterations = manga_settings.get('use_all_iterations', False)
            
            if use_all_iterations:
                # Use the same iteration count for all text types
                all_iterations = manga_settings.get('all_iterations', 2)
                text_bubble_iterations = all_iterations
                empty_bubble_iterations = all_iterations
                free_text_iterations = all_iterations
                self._log(f"📏 Using uniform iterations: {all_iterations} for all text types", "info")
            else:
                # Use individual iteration settings
                text_bubble_iterations = manga_settings.get('text_bubble_dilation_iterations',
                                                           manga_settings.get('bubble_dilation_iterations', 2))
                empty_bubble_iterations = manga_settings.get('empty_bubble_dilation_iterations', 3)
                free_text_iterations = manga_settings.get('free_text_dilation_iterations', 0)
                self._log(f"📏 Using individual iterations - Text bubbles: {text_bubble_iterations}, "
                         f"Empty bubbles: {empty_bubble_iterations}, Free text: {free_text_iterations}", "info")
        
        # Create separate masks for different text types
        text_bubble_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        empty_bubble_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        free_text_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        text_bubble_count = 0
        empty_bubble_count = 0
        free_text_count = 0
        
        for i, region in enumerate(regions):
            # CHECK: Should this region be inpainted?
            if not getattr(region, 'should_inpaint', True):
                # Skip this region - it shouldn't be inpainted
                regions_skipped += 1
                self._log(f"   Region {i+1}: SKIPPED (filtered by settings)", "debug")
                continue
            
            regions_masked += 1
            
            # Determine text type
            text_type = 'free_text'  # default
            
            # Check if region has bubble_type attribute (from bubble detection)
            if hasattr(region, 'bubble_type'):
                # RT-DETR classifications
                if region.bubble_type == 'empty_bubble':
                    text_type = 'empty_bubble'
                elif region.bubble_type == 'text_bubble':
                    text_type = 'text_bubble'
                else:  # 'free_text' or others
                    text_type = 'free_text'
            else:
                # Fallback: use simple heuristics if no bubble detection
                x, y, w, h = region.bounding_box
                x, y, w, h = int(x), int(y), int(w), int(h)
                aspect_ratio = w / h if h > 0 else 1
                
                # Check if region has text
                has_text = hasattr(region, 'text') and region.text and len(region.text.strip()) > 0
                
                # Heuristic: bubbles tend to be more square-ish or tall
                # Free text tends to be wide and short
                if aspect_ratio < 2.5 and w > 50 and h > 50:
                    if has_text:
                        text_type = 'text_bubble'
                    else:
                        # Could be empty bubble if it's round/oval shaped
                        text_type = 'empty_bubble'
                else:
                    text_type = 'free_text'
            
            # Select appropriate mask and increment counter
            if text_type == 'text_bubble':
                target_mask = text_bubble_mask
                text_bubble_count += 1
                mask_type = "TEXT BUBBLE"
            elif text_type == 'empty_bubble':
                target_mask = empty_bubble_mask
                empty_bubble_count += 1
                mask_type = "EMPTY BUBBLE"
            else:
                target_mask = free_text_mask
                free_text_count += 1
                mask_type = "FREE TEXT"
            
            # Check if this is a merged region with original regions
            if hasattr(region, 'original_regions') and region.original_regions:
                # Use original regions for precise masking
                self._log(f"   Region {i+1} ({mask_type}): Using {len(region.original_regions)} original regions", "debug")
                
                for orig_region in region.original_regions:
                    if hasattr(orig_region, 'vertices') and orig_region.vertices:
                        pts = np.array(orig_region.vertices, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.fillPoly(target_mask, [pts], 255)
                    else:
                        x, y, w, h = orig_region.bounding_box
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        cv2.rectangle(target_mask, (x, y), (x + w, y + h), 255, -1)
            else:
                # Normal region
                if hasattr(region, 'vertices') and region.vertices and len(region.vertices) <= 8:
                    pts = np.array(region.vertices, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(target_mask, [pts], 255)
                    self._log(f"   Region {i+1} ({mask_type}): Using polygon", "debug")
                else:
                    x, y, w, h = region.bounding_box
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    cv2.rectangle(target_mask, (x, y), (x + w, y + h), 255, -1)
                    self._log(f"   Region {i+1} ({mask_type}): Using bounding box", "debug")
        
        self._log(f"📊 Mask breakdown: {text_bubble_count} text bubbles, {empty_bubble_count} empty bubbles, "
                 f"{free_text_count} free text regions, {regions_skipped} skipped", "info")
        
        # Apply different dilation settings to each mask type
        if base_dilation_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_dilation_size, base_dilation_size))
            
            # Apply dilation to text bubble mask
            if text_bubble_count > 0 and text_bubble_iterations > 0:
                self._log(f"📏 Applying text bubble dilation: {base_dilation_size}px, {text_bubble_iterations} iterations", "info")
                text_bubble_mask = cv2.dilate(text_bubble_mask, kernel, iterations=text_bubble_iterations)
            
            # Apply dilation to empty bubble mask
            if empty_bubble_count > 0 and empty_bubble_iterations > 0:
                self._log(f"📏 Applying empty bubble dilation: {base_dilation_size}px, {empty_bubble_iterations} iterations", "info")
                empty_bubble_mask = cv2.dilate(empty_bubble_mask, kernel, iterations=empty_bubble_iterations)
            
            # Apply dilation to free text mask
            if free_text_count > 0 and free_text_iterations > 0:
                self._log(f"📏 Applying free text dilation: {base_dilation_size}px, {free_text_iterations} iterations", "info")
                free_text_mask = cv2.dilate(free_text_mask, kernel, iterations=free_text_iterations)
            elif free_text_count > 0 and free_text_iterations == 0:
                self._log(f"📏 No dilation for free text (iterations=0, perfect for B&W panels)", "info")
        
        # Combine all masks
        mask = cv2.bitwise_or(text_bubble_mask, empty_bubble_mask)
        mask = cv2.bitwise_or(mask, free_text_mask)
        
        coverage_percent = (np.sum(mask > 0) / mask.size) * 100
        self._log(f"📊 Final mask coverage: {coverage_percent:.1f}% of image", "info")
        
        return mask
    
    def _get_or_init_shared_local_inpainter(self, local_method: str, model_path: str, force_reload: bool = False):
        """Return a shared LocalInpainter for (local_method, model_path) with minimal locking.
        If another thread is loading the same model, wait on its event instead of competing.
        Set force_reload=True only when the method or model_path actually changed.
        
        If spare instances are available in the pool, check one out for use.
        The instance will stay assigned to this translator until cleanup.
        """
        from local_inpainter import LocalInpainter
        key = (local_method, model_path or '')
        
        # FIRST: Try to check out a spare instance if available (for true parallelism)
        # Don't pop it - instead mark it as 'in use' so it stays in memory
        with MangaTranslator._inpaint_pool_lock:
            rec = MangaTranslator._inpaint_pool.get(key)
            if rec and rec.get('spares'):
                spares = rec.get('spares') or []
                # Initialize checked_out list if it doesn't exist
                if 'checked_out' not in rec:
                    rec['checked_out'] = []
                checked_out = rec['checked_out']
                
                # Look for an available spare (not checked out)
                for spare in spares:
                    if spare not in checked_out and spare and getattr(spare, 'model_loaded', False):
                        # Mark as checked out
                        checked_out.append(spare)
                        self._log(f"🧰 Checked out spare inpainter ({len(checked_out)}/{len(spares)} in use)", "debug")
                        # Store reference for later return
                        self._checked_out_inpainter = spare
                        self._inpainter_pool_key = key
                        return spare
        
        # FALLBACK: Use the shared instance
        rec = MangaTranslator._inpaint_pool.get(key)
        if rec and rec.get('loaded') and rec.get('inpainter'):
            # Already loaded - do NOT force reload!
            return rec['inpainter']
        # Create or wait for loader
        with MangaTranslator._inpaint_pool_lock:
            rec = MangaTranslator._inpaint_pool.get(key)
            if rec and rec.get('loaded') and rec.get('inpainter'):
                # Already loaded - do NOT force reload!
                return rec['inpainter']
            if not rec:
                # Register loading record
                rec = {'inpainter': None, 'loaded': False, 'event': threading.Event()}
                MangaTranslator._inpaint_pool[key] = rec
                is_loader = True
            else:
                is_loader = False
            event = rec['event']
        # Loader performs heavy work without holding the lock
        if is_loader:
            try:
                inp = LocalInpainter()
                # Apply tiling settings once to the shared instance
                tiling_settings = self.manga_settings.get('tiling', {})
                inp.tiling_enabled = tiling_settings.get('enabled', False)
                inp.tile_size = tiling_settings.get('tile_size', 512)
                inp.tile_overlap = tiling_settings.get('tile_overlap', 64)
                # Ensure model path
                if not model_path or not os.path.exists(model_path):
                    try:
                        model_path = inp.download_jit_model(local_method)
                    except Exception as e:
                        self._log(f"⚠️ JIT download failed: {e}", "warning")
                        model_path = None
                # Load model - NEVER force reload for first-time shared pool loading
                loaded_ok = False
                if model_path and os.path.exists(model_path):
                    try:
                        self._log(f"📦 Loading inpainter model...", "debug")
                        self._log(f"   Method: {local_method}", "debug")
                        self._log(f"   Path: {model_path}", "debug")
                        # Only force reload if explicitly requested AND this is not the first load
                        # For shared pool, we should never force reload on initial load
                        loaded_ok = inp.load_model_with_retry(local_method, model_path, force_reload=force_reload)
                        if not loaded_ok:
                            # Retry with force_reload if initial load failed
                            self._log(f"🔄 Initial load failed, retrying with force_reload=True", "warning")
                            loaded_ok = inp.load_model_with_retry(local_method, model_path, force_reload=True)
                            if not loaded_ok:
                                self._log(f"❌ Both load attempts failed", "error")
                                # Check file validity
                                try:
                                    size_mb = os.path.getsize(model_path) / (1024 * 1024)
                                    self._log(f"   File size: {size_mb:.2f} MB", "info")
                                    if size_mb < 1:
                                        self._log(f"   ⚠️ File may be corrupted (too small)", "warning")
                                except Exception:
                                    self._log(f"   ⚠️ Could not read model file", "warning")
                    except Exception as e:
                        self._log(f"⚠️ Inpainter load exception: {e}", "warning")
                        import traceback
                        self._log(traceback.format_exc(), "debug")
                        loaded_ok = False
                elif not model_path:
                    self._log(f"⚠️ No model path configured for {local_method}", "warning")
                elif not os.path.exists(model_path):
                    self._log(f"⚠️ Model file does not exist: {model_path}", "warning")
                # Publish result
                with MangaTranslator._inpaint_pool_lock:
                    rec = MangaTranslator._inpaint_pool.get(key) or rec
                    rec['inpainter'] = inp
                    rec['loaded'] = bool(loaded_ok)
                    rec['event'].set()
                return inp
            except Exception as e:
                with MangaTranslator._inpaint_pool_lock:
                    rec = MangaTranslator._inpaint_pool.get(key) or rec
                    rec['inpainter'] = None
                    rec['loaded'] = False
                    rec['event'].set()
                self._log(f"⚠️ Shared inpainter setup failed: {e}", "warning")
                return None
        else:
            # Wait for loader to finish (without holding the lock)
            success = event.wait(timeout=120)
            if not success:
                self._log(f"⏱️ Timeout waiting for inpainter to load (120s)", "warning")
                return None
            
            # Check if load was successful
            rec2 = MangaTranslator._inpaint_pool.get(key)
            if not rec2:
                self._log(f"⚠️ Inpainter pool record disappeared after load", "warning")
                return None
            
            inp = rec2.get('inpainter')
            loaded = rec2.get('loaded', False)
            
            if inp and loaded:
                # Successfully loaded by another thread
                return inp
            elif inp and not loaded:
                # Inpainter created but model failed to load
                # Try to load it ourselves
                self._log(f"⚠️ Inpainter exists but model not loaded, attempting to load", "debug")
                if model_path and os.path.exists(model_path):
                    try:
                        loaded_ok = inp.load_model_with_retry(local_method, model_path, force_reload=True)
                        if loaded_ok:
                            # Update the pool record
                            with MangaTranslator._inpaint_pool_lock:
                                rec2['loaded'] = True
                            self._log(f"✅ Successfully loaded model on retry in waiting thread", "info")
                            return inp
                    except Exception as e:
                        self._log(f"❌ Failed to load in waiting thread: {e}", "warning")
                return inp  # Return anyway, inpaint will no-op
            else:
                self._log(f"⚠️ Loader thread failed to create inpainter", "warning")
                return None

    @classmethod
    def _count_preloaded_inpainters(cls) -> int:
        try:
            with cls._inpaint_pool_lock:
                total = 0
                for rec in cls._inpaint_pool.values():
                    try:
                        total += len(rec.get('spares') or [])
                    except Exception:
                        pass
                return total
        except Exception:
            return 0

    def preload_local_inpainters(self, local_method: str, model_path: str, count: int) -> int:
        """Preload N local inpainting instances sequentially into the shared pool for parallel panel translation.
        Returns the number of instances successfully preloaded.
        """
        # Respect singleton mode: do not create extra instances/spares
        if getattr(self, 'use_singleton_models', False):
            try:
                self._log("🧰 Skipping local inpainting preload (singleton mode)", "debug")
            except Exception:
                pass
            return 0
        try:
            from local_inpainter import LocalInpainter
        except Exception:
            self._log("❌ Local inpainter module not available for preloading", "error")
            return 0
        key = (local_method, model_path or '')
        created = 0
        
        # FIRST: Ensure the shared instance is initialized and ready
        # This prevents race conditions when spare instances run out
        with MangaTranslator._inpaint_pool_lock:
            rec = MangaTranslator._inpaint_pool.get(key)
            if not rec or not rec.get('loaded') or not rec.get('inpainter'):
                # Need to create the shared instance
                if not rec:
                    rec = {'inpainter': None, 'loaded': False, 'event': threading.Event(), 'spares': []}
                    MangaTranslator._inpaint_pool[key] = rec
                    need_init_shared = True
                else:
                    need_init_shared = not (rec.get('loaded') and rec.get('inpainter'))
            else:
                need_init_shared = False
        
        if need_init_shared:
            self._log(f"📦 Initializing shared inpainter instance first...", "info")
            try:
                shared_inp = self._get_or_init_shared_local_inpainter(local_method, model_path, force_reload=False)
                if shared_inp and getattr(shared_inp, 'model_loaded', False):
                    self._log(f"✅ Shared instance initialized and model loaded", "info")
                    # Verify the pool record is updated
                    with MangaTranslator._inpaint_pool_lock:
                        rec_check = MangaTranslator._inpaint_pool.get(key)
                        if rec_check:
                            self._log(f"   Pool record: loaded={rec_check.get('loaded')}, has_inpainter={rec_check.get('inpainter') is not None}", "debug")
                else:
                    self._log(f"⚠️ Shared instance initialization returned but model not loaded", "warning")
                    if shared_inp:
                        self._log(f"   Instance exists but model_loaded={getattr(shared_inp, 'model_loaded', 'ATTR_MISSING')}", "debug")
            except Exception as e:
                self._log(f"⚠️ Shared instance initialization failed: {e}", "warning")
                import traceback
                self._log(traceback.format_exc(), "debug")
        
        # Ensure pool record and spares list exist
        with MangaTranslator._inpaint_pool_lock:
            rec = MangaTranslator._inpaint_pool.get(key)
            if not rec:
                rec = {'inpainter': None, 'loaded': False, 'event': threading.Event(), 'spares': []}
                MangaTranslator._inpaint_pool[key] = rec
            if 'spares' not in rec or rec['spares'] is None:
                rec['spares'] = []
            spares = rec.get('spares')
        # Prepare tiling settings
        tiling_settings = self.manga_settings.get('tiling', {}) if hasattr(self, 'manga_settings') else {}
        desired = max(0, int(count) - len(spares))
        if desired <= 0:
            return 0
        ctx = " for parallel panels" if int(count) > 1 else ""
        self._log(f"🧰 Preloading {desired} local inpainting instance(s){ctx}", "info")
        for i in range(desired):
            try:
                inp = LocalInpainter()
                inp.tiling_enabled = tiling_settings.get('enabled', False)
                inp.tile_size = tiling_settings.get('tile_size', 512)
                inp.tile_overlap = tiling_settings.get('tile_overlap', 64)
                # Resolve model path if needed
                resolved = model_path
                if not resolved or not os.path.exists(resolved):
                    try:
                        resolved = inp.download_jit_model(local_method)
                    except Exception as e:
                        self._log(f"⚠️ Preload JIT download failed: {e}", "warning")
                        resolved = None
                if resolved and os.path.exists(resolved):
                    ok = inp.load_model_with_retry(local_method, resolved, force_reload=False)
                    if ok and getattr(inp, 'model_loaded', False):
                        with MangaTranslator._inpaint_pool_lock:
                            rec = MangaTranslator._inpaint_pool.get(key) or {'spares': []}
                            if 'spares' not in rec or rec['spares'] is None:
                                rec['spares'] = []
                            rec['spares'].append(inp)
                            MangaTranslator._inpaint_pool[key] = rec
                        created += 1
                    elif ok and not getattr(inp, 'model_loaded', False):
                        self._log(f"⚠️ Preload: load_model_with_retry returned True but model_loaded is False", "warning")
                    elif not ok:
                        self._log(f"⚠️ Preload: load_model_with_retry returned False", "warning")
                else:
                    self._log("⚠️ Preload skipped: no model path available", "warning")
            except Exception as e:
                self._log(f"⚠️ Preload error: {e}", "warning")
        self._log(f"✅ Preloaded {created} local inpainting instance(s)", "info")
        return created

    def preload_local_inpainters_concurrent(self, local_method: str, model_path: str, count: int, max_parallel: int = None) -> int:
        """Preload N local inpainting instances concurrently into the shared pool.
        Honors advanced toggles for panel/region parallelism to pick a reasonable parallelism.
        Returns number of instances successfully preloaded.
        """
        # Respect singleton mode: do not create extra instances/spares
        if getattr(self, 'use_singleton_models', False):
            try:
                self._log("🧰 Skipping concurrent local inpainting preload (singleton mode)", "debug")
            except Exception:
                pass
            return 0
        try:
            from local_inpainter import LocalInpainter
        except Exception:
            self._log("❌ Local inpainter module not available for preloading", "error")
            return 0
        key = (local_method, model_path or '')
        # Determine desired number based on existing spares
        with MangaTranslator._inpaint_pool_lock:
            rec = MangaTranslator._inpaint_pool.get(key)
            if not rec:
                rec = {'inpainter': None, 'loaded': False, 'event': threading.Event(), 'spares': []}
                MangaTranslator._inpaint_pool[key] = rec
            spares = (rec.get('spares') or [])
        desired = max(0, int(count) - len(spares))
        if desired <= 0:
            return 0
        # Determine max_parallel from advanced settings if not provided
        if max_parallel is None:
            adv = {}
            try:
                adv = self.main_gui.config.get('manga_settings', {}).get('advanced', {}) if hasattr(self, 'main_gui') else {}
            except Exception:
                adv = {}
            if adv.get('parallel_panel_translation', False):
                try:
                    max_parallel = max(1, int(adv.get('panel_max_workers', 2)))
                except Exception:
                    max_parallel = 2
            elif adv.get('parallel_processing', False):
                try:
                    max_parallel = max(1, int(adv.get('max_workers', 4)))
                except Exception:
                    max_parallel = 2
            else:
                max_parallel = 1
        max_parallel = max(1, min(int(max_parallel), int(desired)))
        ctx = " for parallel panels" if int(count) > 1 else ""
        self._log(f"🧰 Preloading {desired} local inpainting instance(s){ctx} (parallel={max_parallel})", "info")
        # Resolve model path once
        resolved_path = model_path
        if not resolved_path or not os.path.exists(resolved_path):
            try:
                probe_inp = LocalInpainter()
                resolved_path = probe_inp.download_jit_model(local_method)
            except Exception as e:
                self._log(f"⚠️ JIT download failed for concurrent preload: {e}", "warning")
                resolved_path = None
        tiling_settings = self.manga_settings.get('tiling', {}) if hasattr(self, 'manga_settings') else {}
        from concurrent.futures import ThreadPoolExecutor, as_completed
        created = 0
        def _one():
            try:
                inp = LocalInpainter()
                inp.tiling_enabled = tiling_settings.get('enabled', False)
                inp.tile_size = tiling_settings.get('tile_size', 512)
                inp.tile_overlap = tiling_settings.get('tile_overlap', 64)
                if resolved_path and os.path.exists(resolved_path):
                    ok = inp.load_model_with_retry(local_method, resolved_path, force_reload=False)
                    if ok and getattr(inp, 'model_loaded', False):
                        with MangaTranslator._inpaint_pool_lock:
                            rec2 = MangaTranslator._inpaint_pool.get(key) or {'spares': []}
                            if 'spares' not in rec2 or rec2['spares'] is None:
                                rec2['spares'] = []
                            rec2['spares'].append(inp)
                            MangaTranslator._inpaint_pool[key] = rec2
                        return True
            except Exception as e:
                self._log(f"⚠️ Concurrent preload error: {e}", "warning")
            return False
        with ThreadPoolExecutor(max_workers=max_parallel) as ex:
            futs = [ex.submit(_one) for _ in range(desired)]
            for f in as_completed(futs):
                try:
                    if f.result():
                        created += 1
                except Exception:
                    pass
        self._log(f"✅ Preloaded {created} local inpainting instance(s)", "info")
        return created
        return created

    @classmethod
    def _count_preloaded_detectors(cls) -> int:
        try:
            with cls._detector_pool_lock:
                return sum(len((rec or {}).get('spares') or []) for rec in cls._detector_pool.values())
        except Exception:
            return 0

    @classmethod
    def get_preload_counters(cls) -> Dict[str, int]:
        """Return current counters for preloaded instances (for diagnostics/logging)."""
        try:
            with cls._inpaint_pool_lock:
                inpaint_spares = sum(len((rec or {}).get('spares') or []) for rec in cls._inpaint_pool.values())
                inpaint_keys = len(cls._inpaint_pool)
            with cls._detector_pool_lock:
                detector_spares = sum(len((rec or {}).get('spares') or []) for rec in cls._detector_pool.values())
                detector_keys = len(cls._detector_pool)
            return {
                'inpaint_spares': inpaint_spares,
                'inpaint_keys': inpaint_keys,
                'detector_spares': detector_spares,
                'detector_keys': detector_keys,
            }
        except Exception:
            return {'inpaint_spares': 0, 'inpaint_keys': 0, 'detector_spares': 0, 'detector_keys': 0}

    def preload_bubble_detectors(self, ocr_settings: Dict[str, Any], count: int) -> int:
        """Preload N bubble detector instances (non-singleton) for panel parallelism.
        Only applies when not using singleton models.
        """
        try:
            from bubble_detector import BubbleDetector
        except Exception:
            self._log("❌ BubbleDetector module not available for preloading", "error")
            return 0
        # Skip if singleton mode
        if getattr(self, 'use_singleton_models', False):
            return 0
        det_type = (ocr_settings or {}).get('detector_type', 'rtdetr_onnx')
        model_id = (ocr_settings or {}).get('rtdetr_model_url') or (ocr_settings or {}).get('bubble_model_path') or ''
        key = (det_type, model_id)
        created = 0
        with MangaTranslator._detector_pool_lock:
            rec = MangaTranslator._detector_pool.get(key)
            if not rec:
                rec = {'spares': []}
                MangaTranslator._detector_pool[key] = rec
            spares = rec.get('spares')
            if spares is None:
                spares = []
                rec['spares'] = spares
        desired = max(0, int(count) - len(spares))
        if desired <= 0:
            return 0
        self._log(f"🧰 Preloading {desired} bubble detector instance(s) [{det_type}]", "info")
        for i in range(desired):
            try:
                bd = BubbleDetector()
                ok = False
                if det_type == 'rtdetr_onnx':
                    ok = bool(bd.load_rtdetr_onnx_model(model_id=model_id))
                elif det_type == 'rtdetr':
                    ok = bool(bd.load_rtdetr_model(model_id=model_id))
                elif det_type == 'yolo':
                    if model_id:
                        ok = bool(bd.load_model(model_id))
                else:
                    # auto: prefer RT-DETR
                    ok = bool(bd.load_rtdetr_model(model_id=model_id))
                if ok:
                    with MangaTranslator._detector_pool_lock:
                        rec = MangaTranslator._detector_pool.get(key) or {'spares': []}
                        if 'spares' not in rec or rec['spares'] is None:
                            rec['spares'] = []
                        rec['spares'].append(bd)
                        MangaTranslator._detector_pool[key] = rec
                    created += 1
            except Exception as e:
                self._log(f"⚠️ Bubble detector preload error: {e}", "warning")
        self._log(f"✅ Preloaded {created} bubble detector instance(s)", "info")
        return created

    def _initialize_local_inpainter(self):
        """Initialize local inpainting if configured"""
        try:
            from local_inpainter import LocalInpainter, HybridInpainter, AnimeMangaInpaintModel
            
            # LOAD THE SETTINGS FROM CONFIG FIRST
            # The dialog saves it as 'manga_local_inpaint_model' at root level
            saved_local_method = self.main_gui.config.get('manga_local_inpaint_model', 'anime')
            saved_inpaint_method = self.main_gui.config.get('manga_inpaint_method', 'cloud')
            
            # MIGRATION: Ensure manga_ prefixed model path keys exist for ONNX methods
            # This fixes compatibility where model paths were saved without manga_ prefix
            for method_variant in ['anime', 'anime_onnx', 'lama', 'lama_onnx', 'aot', 'aot_onnx']:
                non_prefixed_key = f'{method_variant}_model_path'
                prefixed_key = f'manga_{method_variant}_model_path'
                # If we have the non-prefixed but not the prefixed, migrate it
                if non_prefixed_key in self.main_gui.config and prefixed_key not in self.main_gui.config:
                    self.main_gui.config[prefixed_key] = self.main_gui.config[non_prefixed_key]
                    self._log(f"🔄 Migrated model path config: {non_prefixed_key} → {prefixed_key}", "debug")
            
            # Update manga_settings with the saved values
            # ALWAYS use the top-level saved config to ensure correct model is loaded
            if 'inpainting' not in self.manga_settings:
                self.manga_settings['inpainting'] = {}
            
            # Always override with saved values from top-level config
            # This ensures the user's model selection in the settings dialog is respected
            self.manga_settings['inpainting']['method'] = saved_inpaint_method
            self.manga_settings['inpainting']['local_method'] = saved_local_method
            
            # Now get the values (they'll be correct now)
            inpaint_method = self.manga_settings.get('inpainting', {}).get('method', 'cloud')
            
            if inpaint_method == 'local':
                # This will now get the correct saved value
                local_method = self.manga_settings.get('inpainting', {}).get('local_method', 'anime')
                
                # Model path is saved with manga_ prefix - try both key formats for compatibility
                model_path = self.main_gui.config.get(f'manga_{local_method}_model_path', '')
                if not model_path:
                    # Fallback to non-prefixed key (older format)
                    model_path = self.main_gui.config.get(f'{local_method}_model_path', '')
                
                self._log(f"Using local method: {local_method} (loaded from config)", "info")
                
                # Check if we already have a loaded instance in the shared pool
                # This avoids unnecessary tracking and reloading
                inp_shared = self._get_or_init_shared_local_inpainter(local_method, model_path, force_reload=False)
                
                # Only track changes AFTER getting the shared instance
                # This prevents spurious reloads on first initialization
                if not hasattr(self, '_last_local_method'):
                    self._last_local_method = local_method
                    self._last_local_model_path = model_path
                else:
                    # Check if settings actually changed and we need to force reload
                    need_reload = False
                    if self._last_local_method != local_method:
                        self._log(f"🔄 Local method changed from {self._last_local_method} to {local_method}", "info")
                        need_reload = True
                        # If method changed, we need a different model - get it with force_reload
                        inp_shared = self._get_or_init_shared_local_inpainter(local_method, model_path, force_reload=True)
                    elif self._last_local_model_path != model_path:
                        self._log(f"🔄 Model path changed", "info")
                        if self._last_local_model_path:
                            self._log(f"   Old: {os.path.basename(self._last_local_model_path)}", "debug")
                        if model_path:
                            self._log(f"   New: {os.path.basename(model_path)}", "debug")
                        need_reload = True
                        # If path changed, reload the model
                        inp_shared = self._get_or_init_shared_local_inpainter(local_method, model_path, force_reload=True)
                    
                    # Update tracking only if changes were made
                    if need_reload:
                        self._last_local_method = local_method
                        self._last_local_model_path = model_path
                if inp_shared is not None:
                    self.local_inpainter = inp_shared
                    if getattr(self.local_inpainter, 'model_loaded', False):
                        self._log(f"✅ Using shared {local_method.upper()} inpainting model", "info")
                        return True
                    else:
                        self._log(f"⚠️ Shared inpainter created but model not loaded", "warning")
                        self._log(f"🔄 Attempting to retry model loading...", "info")
                        
                        # Retry loading the model
                        if model_path and os.path.exists(model_path):
                            self._log(f"📦 Model path: {model_path}", "info")
                            self._log(f"📋 Method: {local_method}", "info")
                            try:
                                loaded_ok = inp_shared.load_model_with_retry(local_method, model_path, force_reload=True)
                                if loaded_ok and getattr(inp_shared, 'model_loaded', False):
                                    self._log(f"✅ Model loaded successfully on retry", "info")
                                    return True
                                else:
                                    self._log(f"❌ Model still not loaded after retry", "error")
                                    # Check if model file exists and is valid
                                    try:
                                        size_mb = os.path.getsize(model_path) / (1024 * 1024)
                                        self._log(f"📊 Model file size: {size_mb:.2f} MB", "info")
                                        if size_mb < 1:
                                            self._log(f"⚠️ Model file seems too small (< 1 MB) - may be corrupted", "warning")
                                    except Exception:
                                        pass
                            except Exception as e:
                                self._log(f"❌ Retry load failed: {e}", "error")
                                import traceback
                                self._log(traceback.format_exc(), "debug")
                        elif not model_path:
                            self._log(f"❌ No model path provided", "error")
                        elif not os.path.exists(model_path):
                            self._log(f"❌ Model path does not exist: {model_path}", "error")
                            self._log(f"📥 Tip: Try downloading the model from the Manga Settings dialog", "info")
                        
                        # If retry failed, fall through to fallback logic below
                
                # Fall back to instance-level init only if shared init completely failed
                self._log("⚠️ Shared inpainter init failed, falling back to instance creation", "warning")
                try:
                    from local_inpainter import LocalInpainter
                    
                    # Create local inpainter instance
                    self.local_inpainter = LocalInpainter()
                    tiling_settings = self.manga_settings.get('tiling', {})
                    self.local_inpainter.tiling_enabled = tiling_settings.get('enabled', False)
                    self.local_inpainter.tile_size = tiling_settings.get('tile_size', 512)
                    self.local_inpainter.tile_overlap = tiling_settings.get('tile_overlap', 64)
                    self._log(f"✅ Set tiling: enabled={self.local_inpainter.tiling_enabled}, size={self.local_inpainter.tile_size}, overlap={self.local_inpainter.tile_overlap}", "info")
                    
                    # If no model path or doesn't exist, try to find or download one
                    if not model_path or not os.path.exists(model_path):
                        self._log(f"⚠️ Model path not found: {model_path}", "warning")
                        self._log("📥 Attempting to download JIT model...", "info")
                        try:
                            downloaded_path = self.local_inpainter.download_jit_model(local_method)
                        except Exception as e:
                            self._log(f"⚠️ JIT download failed: {e}", "warning")
                            downloaded_path = None
                        if downloaded_path:
                            model_path = downloaded_path
                            self._log(f"✅ Downloaded JIT model to: {model_path}")
                        else:
                            self._log("⚠️ JIT model download did not return a path", "warning")
                    
                    # Load model with retry to avoid transient file/JSON issues under parallel init
                    loaded_ok = False
                    if model_path and os.path.exists(model_path):
                        for attempt in range(2):
                            try:
                                self._log(f"📥 Loading {local_method} model... (attempt {attempt+1})", "info")
                                if self.local_inpainter.load_model(local_method, model_path, force_reload=need_reload):
                                    loaded_ok = True
                                    break
                            except Exception as e:
                                self._log(f"⚠️ Load attempt {attempt+1} failed: {e}", "warning")
                                time.sleep(0.5)
                        if loaded_ok:
                            self._log(f"✅ Local inpainter loaded with {local_method.upper()} (fallback instance)")
                        else:
                            self._log(f"⚠️ Failed to load model, but inpainter is ready", "warning")
                    else:
                        self._log(f"⚠️ No model available, but inpainter is initialized", "warning")
                    
                    return True
                    
                except Exception as e:
                    self._log(f"❌ Local inpainter module not available: {e}", "error")
                    return False
            
            elif inpaint_method == 'hybrid':
                # Track hybrid settings changes
                if not hasattr(self, '_last_hybrid_config'):
                    self._last_hybrid_config = None
                    
                    # Set tiling from tiling section
                    tiling_settings = self.manga_settings.get('tiling', {})
                    self.local_inpainter.tiling_enabled = tiling_settings.get('enabled', False)
                    self.local_inpainter.tile_size = tiling_settings.get('tile_size', 512)
                    self.local_inpainter.tile_overlap = tiling_settings.get('tile_overlap', 64)
                    
                    self._log(f"✅ Set tiling: enabled={self.local_inpainter.tiling_enabled}, size={self.local_inpainter.tile_size}, overlap={self.local_inpainter.tile_overlap}", "info")
                        
                current_hybrid_config = self.manga_settings.get('inpainting', {}).get('hybrid_methods', [])
                
                # Check if hybrid config changed
                need_reload = self._last_hybrid_config != current_hybrid_config
                if need_reload:
                    self._log("🔄 Hybrid configuration changed, reloading...", "info")
                    self.hybrid_inpainter = None  # Clear old instance
                
                self._last_hybrid_config = current_hybrid_config.copy() if current_hybrid_config else []
                
                if self.hybrid_inpainter is None:
                    self.hybrid_inpainter = HybridInpainter()
                    # REMOVED: No longer override tiling settings for HybridInpainter
                
                # Load multiple methods
                methods = self.manga_settings.get('inpainting', {}).get('hybrid_methods', [])
                loaded = 0
                
                for method_config in methods:
                    method = method_config.get('method')
                    model_path = method_config.get('model_path')
                    
                    if method and model_path:
                        if self.hybrid_inpainter.add_method(method, method, model_path):
                            loaded += 1
                            self._log(f"✅ Added {method.upper()} to hybrid inpainter")
                
                if loaded > 0:
                    self._log(f"✅ Hybrid inpainter ready with {loaded} methods")
                else:
                    self._log("⚠️ Hybrid inpainter initialized but no methods loaded", "warning")
                
                return True
            
            return False
            
        except ImportError:
            self._log("❌ Local inpainter module not available", "error")
            return False
        except Exception as e:
            self._log(f"❌ Error initializing inpainter: {e}", "error")
            return False


    def inpaint_regions(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint using configured method (cloud, local, or hybrid)"""
        # Primary source of truth is the runtime flags set by the UI.
        if getattr(self, 'skip_inpainting', False):
            self._log("   ⏭️ Skipping inpainting (preserving original art)", "info")
            return image.copy()
        
        # Cloud mode explicitly selected in UI
        if getattr(self, 'use_cloud_inpainting', False):
            return self._cloud_inpaint(image, mask)
        
        # Hybrid mode if UI requested it (fallback to settings key if present)
        mode = getattr(self, 'inpaint_mode', None) or self.manga_settings.get('inpainting', {}).get('method')
        if mode == 'hybrid' and hasattr(self, 'hybrid_inpainter'):
            self._log("   🔄 Using hybrid ensemble inpainting", "info")
            return self.hybrid_inpainter.inpaint_ensemble(image, mask)
        
        # If a background preload is running, wait until it's finished before inpainting
        try:
            if hasattr(self, '_inpaint_preload_event') and self._inpaint_preload_event and not self._inpaint_preload_event.is_set():
                self._log("   ⏳ Waiting for local inpainting models to finish preloading...", "info")
                # Wait with a generous timeout, but proceed afterward regardless
                self._inpaint_preload_event.wait(timeout=300)
        except Exception:
            pass
        
        # Default to local inpainting
        local_method = self.manga_settings.get('inpainting', {}).get('local_method', 'anime')
        model_path = self.main_gui.config.get(f'manga_{local_method}_model_path', '')
        
        # Use a thread-local inpainter instance
        inp = self._get_thread_local_inpainter(local_method, model_path)
        if inp and getattr(inp, 'model_loaded', False):
            self._log("   🧽 Using local inpainting", "info")
            return inp.inpaint(image, mask)
        else:
            # Conservative fallback: try shared instance only; do not attempt risky reloads that can corrupt output
            try:
                shared_inp = self._get_or_init_shared_local_inpainter(local_method, model_path)
                if shared_inp and getattr(shared_inp, 'model_loaded', False):
                    self._log("   ✅ Using shared inpainting instance", "info")
                    return shared_inp.inpaint(image, mask)
            except Exception:
                pass
            self._log("   ⚠️ Local inpainting model not loaded; returning original image", "warning")
            return image.copy()
            
    def _cloud_inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
            """Use Replicate API for inpainting"""
            try:
                import requests
                import base64
                from io import BytesIO
                from PIL import Image as PILImage
                import cv2
                
                self._log("   ☁️ Cloud inpainting via Replicate API", "info")
                
                # Convert to PIL
                image_pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                mask_pil = PILImage.fromarray(mask).convert('L')
                
                # Convert to base64
                img_buffer = BytesIO()
                image_pil.save(img_buffer, format='PNG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                mask_buffer = BytesIO()
                mask_pil.save(mask_buffer, format='PNG')
                mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()
                
                # Get cloud settings
                cloud_settings = self.main_gui.config.get('manga_settings', {})
                model_type = cloud_settings.get('cloud_inpaint_model', 'ideogram-v2')
                timeout = cloud_settings.get('cloud_timeout', 60)
                
                # Determine model identifier based on model type
                if model_type == 'ideogram-v2':
                    model = 'ideogram-ai/ideogram-v2'
                    self._log(f"   Using Ideogram V2 inpainting model", "info")
                elif model_type == 'sd-inpainting':
                    model = 'stability-ai/stable-diffusion-inpainting'
                    self._log(f"   Using Stable Diffusion inpainting model", "info")
                elif model_type == 'flux-inpainting':
                    model = 'zsxkib/flux-dev-inpainting'
                    self._log(f"   Using FLUX inpainting model", "info")
                elif model_type == 'custom':
                    model = cloud_settings.get('cloud_custom_version', '')
                    if not model:
                        raise Exception("No custom model identifier specified")
                    self._log(f"   Using custom model: {model}", "info")
                else:
                    # Default to Ideogram V2
                    model = 'ideogram-ai/ideogram-v2'
                    self._log(f"   Using default Ideogram V2 model", "info")
                
                # Build input data based on model type
                input_data = {
                    'image': f'data:image/png;base64,{img_base64}',
                    'mask': f'data:image/png;base64,{mask_base64}'
                }
                
                # Add prompt settings for models that support them
                if model_type in ['ideogram-v2', 'sd-inpainting', 'flux-inpainting', 'custom']:
                    prompt = cloud_settings.get('cloud_inpaint_prompt', 'clean background, smooth surface')
                    input_data['prompt'] = prompt
                    self._log(f"   Prompt: {prompt}", "info")
                    
                    # SD-specific parameters
                    if model_type == 'sd-inpainting':
                        negative_prompt = cloud_settings.get('cloud_negative_prompt', 'text, writing, letters')
                        input_data['negative_prompt'] = negative_prompt
                        input_data['num_inference_steps'] = cloud_settings.get('cloud_inference_steps', 20)
                        self._log(f"   Negative prompt: {negative_prompt}", "info")
                
                # Get the latest version of the model
                headers = {
                    'Authorization': f'Token {self.replicate_api_key}',
                    'Content-Type': 'application/json'
                }
                
                # First, get the latest version of the model
                model_response = requests.get(
                    f'https://api.replicate.com/v1/models/{model}',
                    headers=headers
                )
                
                if model_response.status_code != 200:
                    # If model lookup fails, try direct prediction with model identifier
                    self._log(f"   Model lookup returned {model_response.status_code}, trying direct prediction", "warning")
                    version = None
                else:
                    model_info = model_response.json()
                    version = model_info.get('latest_version', {}).get('id')
                    if not version:
                        raise Exception(f"Could not get version for model {model}")
                
                # Create prediction
                prediction_data = {
                    'input': input_data
                }
                
                if version:
                    prediction_data['version'] = version
                else:
                    # For custom models, try extracting version from model string
                    if ':' in model:
                        # Format: owner/model:version
                        model_name, version_id = model.split(':', 1)
                        prediction_data['version'] = version_id
                    else:
                        raise Exception(f"Could not determine version for model {model}. Try using format: owner/model:version")
                
                response = requests.post(
                    'https://api.replicate.com/v1/predictions',
                    headers=headers,
                    json=prediction_data
                )
                
                if response.status_code != 201:
                    raise Exception(f"API error: {response.text}")
                    
                # Get prediction URL
                prediction = response.json()
                prediction_url = prediction.get('urls', {}).get('get') or prediction.get('id')
                
                if not prediction_url:
                    raise Exception("No prediction URL returned")
                
                # If we only got an ID, construct the URL
                if not prediction_url.startswith('http'):
                    prediction_url = f'https://api.replicate.com/v1/predictions/{prediction_url}'
                
                # Poll for result with configured timeout
                import time
                for i in range(timeout):
                    response = requests.get(prediction_url, headers=headers)
                    result = response.json()
                    
                    # Log progress every 5 seconds
                    if i % 5 == 0 and i > 0:
                        self._log(f"   ⏳ Still processing... ({i}s elapsed)", "info")
                    
                    if result['status'] == 'succeeded':
                        # Download result image (handle both single URL and list)
                        output = result.get('output')
                        if not output:
                            raise Exception("No output returned from model")
                        
                        if isinstance(output, list):
                            output_url = output[0] if output else None
                        else:
                            output_url = output
                        
                        if not output_url:
                            raise Exception("No output URL in result")
                            
                        img_response = requests.get(output_url)
                        
                        # Convert back to numpy
                        result_pil = PILImage.open(BytesIO(img_response.content))
                        result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                        
                        self._log("   ✅ Cloud inpainting completed", "success")
                        return result_bgr
                        
                    elif result['status'] == 'failed':
                        error_msg = result.get('error', 'Unknown error')
                        # Check for common errors
                        if 'version' in error_msg.lower():
                            error_msg += f" (Try using the model identifier '{model}' in the custom field)"
                        raise Exception(f"Inpainting failed: {error_msg}")
                        
                    time.sleep(1)
                    
                raise Exception(f"Timeout waiting for inpainting (>{timeout}s)")
                
            except Exception as e:
                self._log(f"   ❌ Cloud inpainting failed: {str(e)}", "error")
                return image.copy()         
            

    def _regions_overlap(self, region1: TextRegion, region2: TextRegion) -> bool:
        """Check if two regions overlap"""
        x1, y1, w1, h1 = region1.bounding_box
        x2, y2, w2, h2 = region2.bounding_box
        
        # Check if rectangles overlap
        if (x1 + w1 < x2 or x2 + w2 < x1 or 
            y1 + h1 < y2 or y2 + h2 < y1):
            return False
        
        return True

    def _calculate_overlap_area(self, region1: TextRegion, region2: TextRegion) -> float:
        """Calculate the area of overlap between two regions"""
        x1, y1, w1, h1 = region1.bounding_box
        x2, y2, w2, h2 = region2.bounding_box
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        return (x_right - x_left) * (y_bottom - y_top)

    def _adjust_overlapping_regions(self, regions: List[TextRegion], image_width: int, image_height: int) -> List[TextRegion]:
        """Adjust positions of overlapping regions to prevent overlap while preserving text mapping"""
        if len(regions) <= 1:
            return regions
        
        # Create a copy of regions with preserved indices
        adjusted_regions = []
        for idx, region in enumerate(regions):
            # Create a new TextRegion with copied values
            adjusted_region = TextRegion(
                text=region.text,
                vertices=list(region.vertices),
                bounding_box=list(region.bounding_box),
                confidence=region.confidence,
                region_type=region.region_type
            )
            if hasattr(region, 'translated_text'):
                adjusted_region.translated_text = region.translated_text
            
            # IMPORTANT: Preserve original index to maintain text mapping
            adjusted_region.original_index = idx
            adjusted_region.original_bbox = tuple(region.bounding_box)  # Store original position
            
            adjusted_regions.append(adjusted_region)
        
        # DON'T SORT - This breaks the text-to-region mapping!
        # Process in original order to maintain associations
        
        # Track which regions have been moved to avoid cascade effects
        moved_regions = set()
        
        # Adjust overlapping regions
        for i in range(len(adjusted_regions)):
            if i in moved_regions:
                continue  # Skip if already moved
                
            for j in range(i + 1, len(adjusted_regions)):
                if j in moved_regions:
                    continue  # Skip if already moved
                    
                region1 = adjusted_regions[i]
                region2 = adjusted_regions[j]
                
                if self._regions_overlap(region1, region2):
                    x1, y1, w1, h1 = region1.bounding_box
                    x2, y2, w2, h2 = region2.bounding_box
                    
                    # Calculate centers using ORIGINAL positions for better logic
                    orig_x1, orig_y1, _, _ = region1.original_bbox
                    orig_x2, orig_y2, _, _ = region2.original_bbox
                    
                    # Determine which region to move based on original positions
                    # Move the one that's naturally "later" in reading order
                    if orig_y2 > orig_y1 + h1/2:  # region2 is below
                        # Move region2 down slightly
                        min_gap = 10
                        new_y2 = y1 + h1 + min_gap
                        if new_y2 + h2 <= image_height:
                            region2.bounding_box = (x2, new_y2, w2, h2)
                            moved_regions.add(j)
                            self._log(f"  📍 Adjusted region {j} down (preserving order)", "debug")
                    elif orig_y1 > orig_y2 + h2/2:  # region1 is below
                        # Move region1 down slightly
                        min_gap = 10
                        new_y1 = y2 + h2 + min_gap
                        if new_y1 + h1 <= image_height:
                            region1.bounding_box = (x1, new_y1, w1, h1)
                            moved_regions.add(i)
                            self._log(f"  📍 Adjusted region {i} down (preserving order)", "debug")
                    elif orig_x2 > orig_x1 + w1/2:  # region2 is to the right
                        # Move region2 right slightly
                        min_gap = 10
                        new_x2 = x1 + w1 + min_gap
                        if new_x2 + w2 <= image_width:
                            region2.bounding_box = (new_x2, y2, w2, h2)
                            moved_regions.add(j)
                            self._log(f"  📍 Adjusted region {j} right (preserving order)", "debug")
                    else:
                        # Minimal adjustment - just separate them slightly
                        # without changing their relative order
                        min_gap = 5
                        if y2 >= y1:  # region2 is lower or same level
                            new_y2 = y2 + min_gap
                            if new_y2 + h2 <= image_height:
                                region2.bounding_box = (x2, new_y2, w2, h2)
                                moved_regions.add(j)
                        else:  # region1 is lower
                            new_y1 = y1 + min_gap
                            if new_y1 + h1 <= image_height:
                                region1.bounding_box = (x1, new_y1, w1, h1)
                                moved_regions.add(i)
        
        # IMPORTANT: Return in ORIGINAL order to preserve text mapping
        # Sort by original_index to restore the original order
        adjusted_regions.sort(key=lambda r: r.original_index)
        
        return adjusted_regions

    # Emote-only mixed font fallback (Meiryo) — primary font remains unchanged
    def _get_emote_fallback_font(self, font_size: int):
        """Return a Meiryo Bold fallback font if available (preferred), else Meiryo.
        Does not change the primary font; used only for emote glyphs.
        """
        try:
            from PIL import ImageFont as _ImageFont
            import os as _os
            # Prefer Meiryo Bold TTC first; try common face indices, then regular Meiryo
            candidates = [
                ("C:/Windows/Fonts/meiryob.ttc", [0,1,2,3]),  # Meiryo Bold (and variants) TTC
                ("C:/Windows/Fonts/meiryo.ttc",  [1,0,2,3]),  # Try bold-ish index first if present
            ]
            for path, idxs in candidates:
                if _os.path.exists(path):
                    for idx in idxs:
                        try:
                            return _ImageFont.truetype(path, font_size, index=idx)
                        except Exception:
                            continue
            return None
        except Exception:
            return None

    def _is_emote_char(self, ch: str) -> bool:
        # Strict whitelist of emote-like symbols to render with Meiryo
        EMOTES = set([
            '\u2661', # ♡
            '\u2665', # ♥
            '\u2764', # ❤
            '\u2605', # ★
            '\u2606', # ☆
            '\u266A', # ♪
            '\u266B', # ♫
            '\u203B', # ※
        ])
        return ch in EMOTES

    def _line_width_emote_mixed(self, draw, text: str, primary_font, emote_font) -> int:
        if not emote_font:
            bbox = draw.textbbox((0, 0), text, font=primary_font)
            return (bbox[2] - bbox[0])
        w = 0
        i = 0
        while i < len(text):
            ch = text[i]
            # Treat VS16/VS15 as zero-width modifiers
            if ch in ('\ufe0f', '\ufe0e'):
                i += 1
                continue
            f = emote_font if self._is_emote_char(ch) else primary_font
            try:
                bbox = draw.textbbox((0, 0), ch, font=f)
                w += (bbox[2] - bbox[0])
            except Exception:
                w += max(1, int(getattr(primary_font, 'size', 12) * 0.6))
            i += 1
        return w

    def _draw_text_line_emote_mixed(self, draw, line: str, x: int, y: int, primary_font, emote_font,
                                    fill_rgba, outline_rgba, outline_width: int,
                                    shadow_enabled: bool, shadow_color_rgba, shadow_off):
        cur_x = x
        i = 0
        while i < len(line):
            ch = line[i]
            if ch in ('\ufe0f', '\ufe0e'):
                i += 1
                continue
            f = emote_font if (emote_font and self._is_emote_char(ch)) else primary_font
            # measure
            try:
                bbox = draw.textbbox((0, 0), ch, font=f)
                cw = bbox[2] - bbox[0]
            except Exception:
                cw = max(1, int(getattr(primary_font, 'size', 12) * 0.6))
            # shadow
            if shadow_enabled:
                sx, sy = shadow_off
                draw.text((cur_x + sx, y + sy), ch, font=f, fill=shadow_color_rgba)
            # outline
            if outline_width > 0:
                for dx in range(-outline_width, outline_width + 1):
                    for dy in range(-outline_width, outline_width + 1):
                        if dx == 0 and dy == 0:
                            continue
                        draw.text((cur_x + dx, y + dy), ch, font=f, fill=outline_rgba)
            # main
            draw.text((cur_x, y), ch, font=f, fill=fill_rgba)
            cur_x += cw
            i += 1

    
    def render_translated_text(self, image: np.ndarray, regions: List[TextRegion]) -> np.ndarray:
        """Enhanced text rendering with customizable backgrounds and styles"""
        self._log(f"\n🎨 Starting ENHANCED text rendering with custom settings:", "info")
        self._log(f"  ✅ Using ENHANCED renderer (not the simple version)", "info")
        self._log(f"  Background: {self.text_bg_style} @ {int(self.text_bg_opacity/255*100)}% opacity", "info")
        self._log(f"  Text color: RGB{self.text_color}", "info")
        self._log(f"  Shadow: {'Enabled' if self.shadow_enabled else 'Disabled'}", "info")
        self._log(f"  Font: {os.path.basename(self.selected_font_style) if self.selected_font_style else 'Default'}", "info")
        if self.force_caps_lock:  
            self._log(f"  Force Caps Lock: ENABLED", "info")
        
        # Convert to PIL for text rendering
        import cv2
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Get image dimensions for boundary checking
        image_height, image_width = image.shape[:2]  # <-- Add this line
        
        # Only adjust overlapping regions if constraining to bubbles
        if self.constrain_to_bubble:
            adjusted_regions = self._adjust_overlapping_regions(regions, image_width, image_height)
        else:
            # Skip adjustment when not constraining (allows overflow)
            adjusted_regions = regions
            self._log("  📝 Using original regions (overflow allowed)", "info")
        
        # Check if any regions still overlap after adjustment (shouldn't happen, but let's verify)
        has_overlaps = False
        for i, region1 in enumerate(adjusted_regions):
            for region2 in adjusted_regions[i+1:]:
                if self._regions_overlap(region1, region2):
                    has_overlaps = True
                    self._log("  ⚠️ Regions still overlap after adjustment", "warning")
                    break
            if has_overlaps:
                break
        
        # Handle transparency settings based on overlaps
        if has_overlaps and self.text_bg_opacity < 255 and self.text_bg_opacity > 0:
            self._log("  ⚠️ Overlapping regions detected with partial transparency", "warning")
            self._log("  ℹ️ Rendering with requested transparency level", "info")
        
        region_count = 0
        
        # Decide rendering path based on transparency needs
        # For full transparency (opacity = 0) or no overlaps, use RGBA rendering
        # For overlaps with partial transparency, we still use RGBA to honor user settings
        use_rgba_rendering = True  # Always use RGBA for consistent transparency support
        
        if use_rgba_rendering:
            # Transparency-enabled rendering path
            pil_image = pil_image.convert('RGBA')

            # Decide parallel rendering from advanced settings
            try:
                adv = getattr(self, 'manga_settings', {}).get('advanced', {}) if hasattr(self, 'manga_settings') else {}
            except Exception:
                adv = {}
            render_parallel = bool(adv.get('render_parallel', True))
            max_workers = None
            try:
                max_workers = int(adv.get('max_workers', 4))
            except Exception:
                max_workers = 4

            def _render_one(region, idx):
                # Build a separate overlay for this region
                from PIL import Image as _PIL
                overlay = _PIL.new('RGBA', pil_image.size, (0,0,0,0))
                draw = ImageDraw.Draw(overlay)
                # Work on local copy of text for caps lock
                tr_text = region.translated_text or ''
                if self.force_caps_lock:
                    tr_text = tr_text.upper()
                x, y, w, h = region.bounding_box
                # Fit text
                if self.custom_font_size:
                    font_size = self.custom_font_size
                    if hasattr(region, 'vertices') and region.vertices:
                        _, _, safe_w, safe_h = self.get_safe_text_area(region)
                        lines = self._wrap_text(tr_text, self._get_font(font_size), safe_w, draw)
                    else:
                        lines = self._wrap_text(tr_text, self._get_font(font_size), int(w*0.8), draw)
                elif self.font_size_mode == 'multiplier':
                    font_size, lines = self._fit_text_to_region(tr_text, w, h, draw, region)
                else:
                    font_size, lines = self._fit_text_to_region(tr_text, w, h, draw, region)
                # Fonts
                font = self._get_font(font_size)
                emote_font = self._get_emote_fallback_font(font_size)
                # Layout
                line_height = font_size * 1.2
                total_height = len(lines) * line_height
                start_y = y + (h - total_height) // 2
                # BG
                draw_bg = self.text_bg_opacity > 0
                try:
                    if draw_bg and getattr(self, 'free_text_only_bg_opacity', False):
                        draw_bg = self._is_free_text_region(region)
                except Exception:
                    pass
                if draw_bg:
                    self._draw_text_background(draw, x, y, w, h, lines, font, font_size, start_y, emote_font)
                # Text
                for i, line in enumerate(lines):
                    if emote_font is not None:
                        text_width = self._line_width_emote_mixed(draw, line, font, emote_font)
                    else:
                        tb = draw.textbbox((0,0), line, font=font)
                        text_width = tb[2]-tb[0]
                    tx = x + (w - text_width)//2
                    ty = start_y + i*line_height
                    ow = max(1, font_size // self.outline_width_factor)
                    if emote_font is not None:
                        self._draw_text_line_emote_mixed(draw, line, tx, ty, font, emote_font,
                                                         self.text_color + (255,), self.outline_color + (255,), ow,
                                                         self.shadow_enabled,
                                                         self.shadow_color + (255,) if isinstance(self.shadow_color, tuple) and len(self.shadow_color)==3 else (0,0,0,255),
                                                         (self.shadow_offset_x, self.shadow_offset_y))
                    else:
                        if self.shadow_enabled:
                            self._draw_text_shadow(draw, tx, ty, line, font)
                        for dx in range(-ow, ow+1):
                            for dy in range(-ow, ow+1):
                                if dx!=0 or dy!=0:
                                    draw.text((tx+dx, ty+dy), line, font=font, fill=self.outline_color + (255,))
                        draw.text((tx, ty), line, font=font, fill=self.text_color + (255,))
                return overlay

            overlays = []
            if render_parallel and len(adjusted_regions) > 1:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                workers = max(1, min(max_workers, len(adjusted_regions)))
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    fut_to_idx = {ex.submit(_render_one, r, i): i for i, r in enumerate(adjusted_regions) if r.translated_text}
                    # Collect in order
                    temp = {}
                    for fut in as_completed(fut_to_idx):
                        i = fut_to_idx[fut]
                        try:
                            temp[i] = fut.result()
                        except Exception:
                            temp[i] = None
                    overlays = [temp.get(i) for i in range(len(adjusted_regions))]
            else:
                for i, r in enumerate(adjusted_regions):
                    if not r.translated_text:
                        overlays.append(None)
                        continue
                    overlays.append(_render_one(r, i))

            # Composite overlays sequentially
            for ov in overlays:
                if ov is not None:
                    pil_image = Image.alpha_composite(pil_image, ov)

            # Convert back to RGB
            pil_image = pil_image.convert('RGB')
        
        else:
            # This path is now deprecated but kept for backwards compatibility
            # Direct rendering without transparency layers
            draw = ImageDraw.Draw(pil_image)
            
            for region in adjusted_regions:
                if not region.translated_text:
                    continue
                    
                self._log(f"DEBUG: Rendering - Original: '{region.text[:30]}...' -> Translated: '{region.translated_text[:30]}...'", "debug")

                
                # APPLY CAPS LOCK TRANSFORMATION HERE
                if self.force_caps_lock:
                    region.translated_text = region.translated_text.upper()
                
                region_count += 1
                self._log(f"  Rendering region {region_count}: {region.translated_text[:30]}...", "info")
                
                x, y, w, h = region.bounding_box
                
                # Find optimal font size
                if self.custom_font_size:
                    font_size = self.custom_font_size
                    lines = self._wrap_text(region.translated_text, 
                                          self._get_font(font_size), 
                                          int(w * 0.8), draw)
                else:
                    font_size, lines = self._fit_text_to_region(
                        region.translated_text, w, h, draw
                    )
                
                # Load font
                font = self._get_font(font_size)
                
                # Calculate text layout
                line_height = font_size * 1.2
                total_height = len(lines) * line_height
                start_y = y + (h - total_height) // 2
                
                # Draw opaque background (optionally only for free text)
                draw_bg = self.text_bg_opacity > 0
                try:
                    if draw_bg and getattr(self, 'free_text_only_bg_opacity', False):
                        draw_bg = self._is_free_text_region(region)
                except Exception:
                    pass
                if draw_bg:
                    self._draw_text_background(draw, x, y, w, h, lines, font, 
                                             font_size, start_y)
                
                # Draw text
                for i, line in enumerate(lines):
                    # Mixed fallback not supported in legacy path; keep primary measurement
                    text_bbox = draw.textbbox((0, 0), line, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    
                    text_x = x + (w - text_width) // 2
                    text_y = start_y + i * line_height
                    
                    if self.shadow_enabled:
                        self._draw_text_shadow(draw, text_x, text_y, line, font)
                    
                    outline_width = max(1, font_size // self.outline_width_factor)
                    
                    # Draw outline
                    for dx in range(-outline_width, outline_width + 1):
                        for dy in range(-outline_width, outline_width + 1):
                            if dx != 0 or dy != 0:
                                draw.text((text_x + dx, text_y + dy), line, 
                                        font=font, fill=self.outline_color)
                    
                    # Draw main text
                    draw.text((text_x, text_y), line, font=font, fill=self.text_color)
        
        # Convert back to numpy array
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        self._log(f"✅ ENHANCED text rendering complete - rendered {region_count} regions", "info")
        return result
    
    def _is_free_text_region(self, region) -> bool:
        """Heuristic: determine if the region is free text (not a bubble).
        Uses bubble_type when available; otherwise falls back to aspect ratio heuristics.
        """
        try:
            if hasattr(region, 'bubble_type') and region.bubble_type:
                return region.bubble_type == 'free_text'
            # Fallback heuristic
            x, y, w, h = region.bounding_box
            w, h = int(w), int(h)
            if h <= 0:
                return True
            aspect = w / max(1, h)
            # Wider, shorter regions are often free text
            return aspect >= 2.5 or h < 50
        except Exception:
            return False

    def _draw_text_background(self, draw: ImageDraw, x: int, y: int, w: int, h: int,
                            lines: List[str], font: ImageFont, font_size: int, 
                            start_y: int, emote_font: ImageFont = None):
        """Draw background behind text with selected style.
        If emote_font is provided, measure lines with emote-only mixing.
        """
        # Early return if opacity is 0 (fully transparent)
        if self.text_bg_opacity == 0:
            return
        
        # Calculate actual text bounds
        line_height = font_size * 1.2
        max_width = 0
        
        for line in lines:
            if emote_font is not None:
                line_width = self._line_width_emote_mixed(draw, line, font, emote_font)
            else:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
            max_width = max(max_width, line_width)
        
        # Apply size reduction
        padding = int(font_size * 0.3)
        bg_width = int((max_width + padding * 2) * self.text_bg_reduction)
        bg_height = int((len(lines) * line_height + padding * 2) * self.text_bg_reduction)
        
        # Center background
        bg_x = x + (w - bg_width) // 2
        bg_y = int(start_y - padding)
        
        # Create semi-transparent color
        bg_color = (255, 255, 255, self.text_bg_opacity)
        
        if self.text_bg_style == 'box':
            # Rounded rectangle
            radius = min(20, bg_width // 10, bg_height // 10)
            self._draw_rounded_rectangle(draw, bg_x, bg_y, bg_x + bg_width, 
                                       bg_y + bg_height, radius, bg_color)
            
        elif self.text_bg_style == 'circle':
            # Ellipse that encompasses the text
            center_x = bg_x + bg_width // 2
            center_y = bg_y + bg_height // 2
            # Make it slightly wider to look more natural
            ellipse_width = int(bg_width * 1.2)
            ellipse_height = bg_height
            
            draw.ellipse([center_x - ellipse_width // 2, center_y - ellipse_height // 2,
                         center_x + ellipse_width // 2, center_y + ellipse_height // 2],
                        fill=bg_color)
            
        elif self.text_bg_style == 'wrap':
            # Individual background for each line
            for i, line in enumerate(lines):
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                
                line_bg_width = int((line_width + padding) * self.text_bg_reduction)
                line_bg_x = x + (w - line_bg_width) // 2
                line_bg_y = int(start_y + i * line_height - padding // 2)
                line_bg_height = int(line_height + padding // 2)
                
                # Draw rounded rectangle for each line
                radius = min(10, line_bg_width // 10, line_bg_height // 10)
                self._draw_rounded_rectangle(draw, line_bg_x, line_bg_y, 
                                           line_bg_x + line_bg_width,
                                           line_bg_y + line_bg_height, radius, bg_color)
    
    def _draw_text_shadow(self, draw: ImageDraw, x: int, y: int, text: str, font: ImageFont):
        """Draw text shadow with optional blur effect"""
        if self.shadow_blur == 0:
            # Simple sharp shadow
            shadow_x = x + self.shadow_offset_x
            shadow_y = y + self.shadow_offset_y
            draw.text((shadow_x, shadow_y), text, font=font, fill=self.shadow_color)
        else:
            # Blurred shadow (simulated with multiple layers)
            blur_range = self.shadow_blur
            opacity_step = 80 // (blur_range + 1)  # Distribute opacity across blur layers
            
            for blur_offset in range(blur_range, 0, -1):
                layer_opacity = opacity_step * (blur_range - blur_offset + 1)
                shadow_color_with_opacity = self.shadow_color + (layer_opacity,)
                
                # Draw shadow at multiple positions for blur effect
                for dx in range(-blur_offset, blur_offset + 1):
                    for dy in range(-blur_offset, blur_offset + 1):
                        if dx*dx + dy*dy <= blur_offset*blur_offset:  # Circular blur
                            shadow_x = x + self.shadow_offset_x + dx
                            shadow_y = y + self.shadow_offset_y + dy
                            draw.text((shadow_x, shadow_y), text, font=font, 
                                    fill=shadow_color_with_opacity)
    
    def _draw_rounded_rectangle(self, draw: ImageDraw, x1: int, y1: int, 
                               x2: int, y2: int, radius: int, fill):
        """Draw a rounded rectangle"""
        # Draw the main rectangle
        draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill)
        draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill)
        
        # Draw the corners
        draw.pieslice([x1, y1, x1 + 2 * radius, y1 + 2 * radius], 180, 270, fill=fill)
        draw.pieslice([x2 - 2 * radius, y1, x2, y1 + 2 * radius], 270, 360, fill=fill)
        draw.pieslice([x1, y2 - 2 * radius, x1 + 2 * radius, y2], 90, 180, fill=fill)
        draw.pieslice([x2 - 2 * radius, y2 - 2 * radius, x2, y2], 0, 90, fill=fill)
    
    def _get_font(self, font_size: int) -> ImageFont:
        """Get font with specified size, using selected style if available"""
        font_path = self.selected_font_style or self.font_path
        
        if font_path:
            try:
                return ImageFont.truetype(font_path, font_size)
            except:
                pass
        
        return ImageFont.load_default()
     
    def _pil_word_wrap(self, text: str, font_path: str, roi_width: int, roi_height: int,
                       init_font_size: int, min_font_size: int, draw: ImageDraw) -> Tuple[str, int]:
        """Comic-translate's pil_word_wrap algorithm - top-down font sizing with column wrapping.
        
        Break long text to multiple lines, and reduce point size until all text fits within bounds.
        This is a direct port from comic-translate for better text fitting.
        """
        from hyphen_textwrap import wrap as hyphen_wrap
        
        mutable_message = text
        font_size = init_font_size
        
        def eval_metrics(txt, font):
            """Calculate width/height of multiline text."""
            lines = txt.split('\n')
            max_width = 0
            total_height = 0
            
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                line_height = bbox[3] - bbox[1]
                max_width = max(max_width, line_width)
                total_height += line_height
            
            return (max_width, total_height)
        
        # Get initial font
        try:
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        
        # Top-down algorithm: start with large font, shrink until it fits
        while font_size > min_font_size:
            try:
                if font_path:
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    font = ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()
            
            width, height = eval_metrics(mutable_message, font)
            
            if height > roi_height:
                # Text is too tall, reduce font size
                font_size -= 0.75
                mutable_message = text  # Restore original text
            elif width > roi_width:
                # Text is too wide, try wrapping with column optimization
                columns = len(mutable_message)
                
                # Search for optimal column width
                while columns > 0:
                    columns -= 1
                    if columns == 0:
                        break
                    
                    # Use hyphen_wrap for smart wrapping
                    try:
                        wrapped = '\n'.join(hyphen_wrap(
                            text, columns,
                            break_on_hyphens=False,
                            break_long_words=False,
                            hyphenate_broken_words=True
                        ))
                        wrapped_width, _ = eval_metrics(wrapped, font)
                        if wrapped_width <= roi_width:
                            mutable_message = wrapped
                            break
                    except Exception:
                        # Fallback to simple wrapping if hyphen_wrap fails
                        break
                
                if columns < 1:
                    # Couldn't find good column width, reduce font size
                    font_size -= 0.75
                    mutable_message = text  # Restore original text
            else:
                # Text fits!
                break
        
        # If we hit minimum font size, do brute-force optimization
        if font_size <= min_font_size:
            font_size = min_font_size
            mutable_message = text
            
            try:
                if font_path:
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    font = ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()
            
            # Brute force: minimize cost function (width - roi_width)^2 + (height - roi_height)^2
            min_cost = 1e9
            min_text = text
            
            for columns in range(1, min(len(text) + 1, 100)):  # Limit iterations for performance
                try:
                    wrapped_text = '\n'.join(hyphen_wrap(
                        text, columns,
                        break_on_hyphens=False,
                        break_long_words=False,
                        hyphenate_broken_words=True
                    ))
                    wrapped_width, wrapped_height = eval_metrics(wrapped_text, font)
                    cost = (wrapped_width - roi_width)**2 + (wrapped_height - roi_height)**2
                    
                    if cost < min_cost:
                        min_cost = cost
                        min_text = wrapped_text
                except Exception:
                    continue
            
            mutable_message = min_text
        
        return mutable_message, int(font_size)
    
    def get_safe_text_area(self, region: TextRegion) -> Tuple[int, int, int, int]:
        """Get safe text area with less conservative margins for readability"""
        if not hasattr(region, 'vertices') or not region.vertices:
            x, y, w, h = region.bounding_box
            margin_factor = 0.85  # Less conservative default
            safe_width = int(w * margin_factor)
            safe_height = int(h * margin_factor)
            safe_x = x + (w - safe_width) // 2
            safe_y = y + (h - safe_height) // 2
            return safe_x, safe_y, safe_width, safe_height
        
        try:
            # Convert vertices to numpy array with correct dtype
            vertices = np.array(region.vertices, dtype=np.int32)
            hull = cv2.convexHull(vertices)
            hull_area = cv2.contourArea(hull)
            poly_area = cv2.contourArea(vertices)
            
            if poly_area > 0:
                convexity = hull_area / poly_area
            else:
                convexity = 1.0
            
            # LESS CONSERVATIVE margins for better readability
            if convexity < 0.85:  # Speech bubble with tail
                margin_factor = 0.75
                self._log(f"  Speech bubble detected, using 75% of area", "info")
            elif convexity > 0.98:  # Rectangular
                margin_factor = 0.9
                self._log(f"  Rectangular bubble, using 90% of area", "info")
            else:  # Regular bubble
                margin_factor = 0.8
                self._log(f"  Regular bubble, using 80% of area", "info")
        except:
            margin_factor = 0.95
        
        # Convert vertices to numpy array for boundingRect
        vertices_np = np.array(region.vertices, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(vertices_np)
        
        safe_width = int(w * margin_factor)
        safe_height = int(h * margin_factor)
        safe_x = x + (w - safe_width) // 2
        safe_y = y + (h - safe_height) // 2
        
        return safe_x, safe_y, safe_width, safe_height
    
    def _fit_text_to_region(self, text: str, max_width: int, max_height: int, draw: ImageDraw, region: TextRegion = None) -> Tuple[int, List[str]]:
        """Find optimal font size using comic-translate's pil_word_wrap algorithm"""
        
        # Get usable area
        if region and hasattr(region, 'vertices') and region.vertices:
            safe_x, safe_y, safe_width, safe_height = self.get_safe_text_area(region)
            usable_width = safe_width
            usable_height = safe_height
        else:
            # Use 85% of bubble area
            margin = 0.85
            usable_width = int(max_width * margin)
            usable_height = int(max_height * margin)
        
        # Font size limits (GUI settings)
        min_font_size = max(10, self.min_readable_size)
        init_font_size = min(40, self.max_font_size_limit)
        
        # Use comic-translate's pil_word_wrap algorithm
        wrapped_text, final_font_size = self._pil_word_wrap(
            text=text,
            font_path=self.selected_font_style or self.font_path,
            roi_width=usable_width,
            roi_height=usable_height,
            init_font_size=init_font_size,
            min_font_size=min_font_size,
            draw=draw
        )
        
        # Convert wrapped text to lines
        lines = wrapped_text.split('\n') if wrapped_text else [text]
        
        # Apply multiplier if in multiplier mode
        if self.font_size_mode == 'multiplier':
            target_size = int(final_font_size * self.font_size_multiplier)
            
            # Check if multiplied size still fits (if constrained)
            if self.constrain_to_bubble:
                # Re-wrap at target size to check fit
                test_wrapped, _ = self._pil_word_wrap(
                    text=text,
                    font_path=self.selected_font_style or self.font_path,
                    roi_width=usable_width,
                    roi_height=usable_height,
                    init_font_size=target_size,
                    min_font_size=target_size,  # Force this size
                    draw=draw
                )
                test_lines = test_wrapped.split('\n') if test_wrapped else [text]
                test_height = len(test_lines) * target_size * 1.2
                
                if test_height <= usable_height:
                    final_font_size = target_size
                    lines = test_lines
                else:
                    self._log(f"  Multiplier {self.font_size_multiplier}x would exceed bubble", "debug")
            else:
                # Not constrained, use multiplied size
                final_font_size = target_size
                lines = wrapped_text.split('\n') if wrapped_text else [text]
        
        self._log(f"  Font sizing: text_len={len(text)}, size={final_font_size}, lines={len(lines)}", "debug")
        
        return final_font_size, lines

    def _fit_text_simple_topdown(self, text: str, usable_width: int, usable_height: int, 
                                 draw: ImageDraw, min_size: int, max_size: int) -> Tuple[int, List[str]]:
        """Simple top-down approach - start large and shrink only if needed"""
        # Start from a reasonable large size
        start_size = int(max_size * 0.8)
        
        for font_size in range(start_size, min_size - 1, -2):  # Step by 2 for speed
            font = self._get_font(font_size)
            lines = self._wrap_text(text, font, usable_width, draw)
            
            line_height = font_size * 1.2  # Tighter for overlaps
            total_height = len(lines) * line_height
            
            if total_height <= usable_height:
                return font_size, lines
        
        # If nothing fits, use minimum
        font = self._get_font(min_size)
        lines = self._wrap_text(text, font, usable_width, draw)
        return min_size, lines

    def _check_potential_overlap(self, region: TextRegion) -> bool:
        """Check if this region might overlap with others based on position"""
        if not region or not hasattr(region, 'bounding_box'):
            return False
        
        x, y, w, h = region.bounding_box
        
        # Simple heuristic: small regions or regions at edges might overlap
        # You can make this smarter based on your needs
        if w < 100 or h < 50:  # Small bubbles often overlap
            return True
        
        # Add more overlap detection logic here if needed
        # For now, default to no overlap for larger bubbles
        return False
    
    def _wrap_text(self, text: str, font: ImageFont, max_width: int, draw: ImageDraw) -> List[str]:
        """Wrap text to fit within max_width with optional strict wrapping"""
        # Handle empty text
        if not text.strip():
            return []
        
        # Only enforce width check if constrain_to_bubble is enabled
        if self.constrain_to_bubble and max_width <= 0:
            self._log(f"  ⚠️ Invalid max_width: {max_width}, using fallback", "warning")
            return [text[:20] + "..."] if len(text) > 20 else [text]
        
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            # Check if word alone is too long
            word_bbox = draw.textbbox((0, 0), word, font=font)
            word_width = word_bbox[2] - word_bbox[0]
            
            if word_width > max_width and len(word) > 1:
                # Word is too long for the bubble
                if current_line:
                    # Save current line first
                    lines.append(' '.join(current_line))
                    current_line = []
                
                if self.strict_text_wrapping:
                    # STRICT MODE: Force break the word to fit within bubble
                    # This is the original behavior that ensures text stays within bounds
                    broken_parts = self._force_break_word(word, font, max_width, draw)
                    lines.extend(broken_parts)
                else:
                    # RELAXED MODE: Keep word whole (may exceed bubble)
                    lines.append(word)
                    # self._log(f"  ⚠️ Word '{word}' exceeds bubble width, keeping whole", "warning")
            else:
                # Normal word processing
                if current_line:
                    test_line = ' '.join(current_line + [word])
                else:
                    test_line = word
                
                text_bbox = draw.textbbox((0, 0), test_line, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                
                if text_width <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        # Single word that fits
                        lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines

    # Keep the existing _force_break_word method as is (the complete version from earlier):
    def _force_break_word(self, word: str, font: ImageFont, max_width: int, draw: ImageDraw) -> List[str]:
        """Force break a word that's too long to fit"""
        lines = []
        
        # Binary search to find how many characters fit
        low = 1
        high = len(word)
        chars_that_fit = 1
        
        while low <= high:
            mid = (low + high) // 2
            test_text = word[:mid]
            bbox = draw.textbbox((0, 0), test_text, font=font)
            width = bbox[2] - bbox[0]
            
            if width <= max_width:
                chars_that_fit = mid
                low = mid + 1
            else:
                high = mid - 1
        
        # Break the word into pieces
        remaining = word
        while remaining:
            if len(remaining) <= chars_that_fit:
                # Last piece
                lines.append(remaining)
                break
            else:
                # Find the best break point
                break_at = chars_that_fit
                
                # Try to break at a more natural point if possible
                # Look for vowel-consonant boundaries for better hyphenation
                for i in range(min(chars_that_fit, len(remaining) - 1), max(1, chars_that_fit - 5), -1):
                    if i < len(remaining) - 1:
                        current_char = remaining[i].lower()
                        next_char = remaining[i + 1].lower()
                        
                        # Good hyphenation points:
                        # - Between consonant and vowel
                        # - After prefix (un-, re-, pre-, etc.)
                        # - Before suffix (-ing, -ed, -er, etc.)
                        if (current_char in 'bcdfghjklmnpqrstvwxyz' and next_char in 'aeiou') or \
                           (current_char in 'aeiou' and next_char in 'bcdfghjklmnpqrstvwxyz'):
                            break_at = i + 1
                            break
                
                # Add hyphen if we're breaking in the middle of a word
                if break_at < len(remaining):
                    # Check if adding hyphen still fits
                    test_with_hyphen = remaining[:break_at] + '-'
                    bbox = draw.textbbox((0, 0), test_with_hyphen, font=font)
                    width = bbox[2] - bbox[0]
                    
                    if width <= max_width:
                        lines.append(remaining[:break_at] + '-')
                    else:
                        # Hyphen doesn't fit, break without it
                        lines.append(remaining[:break_at])
                else:
                    lines.append(remaining[:break_at])
                
                remaining = remaining[break_at:]
        
        return lines
    
    def _estimate_font_size_for_region(self, region: TextRegion) -> int:
        """Estimate the likely font size for a text region based on its dimensions and text content"""
        x, y, w, h = region.bounding_box
        text_length = len(region.text.strip())
        
        if text_length == 0:
            return self.max_font_size // 2  # Default middle size
        
        # Calculate area per character
        area = w * h
        area_per_char = area / text_length
        
        # Estimate font size based on area per character
        # These ratios are approximate and based on typical manga text
        if area_per_char > 800:
            estimated_size = int(self.max_font_size * 0.8)
        elif area_per_char > 400:
            estimated_size = int(self.max_font_size * 0.6)
        elif area_per_char > 200:
            estimated_size = int(self.max_font_size * 0.4)
        elif area_per_char > 100:
            estimated_size = int(self.max_font_size * 0.3)
        else:
            estimated_size = int(self.max_font_size * 0.2)
        
        # Clamp to reasonable bounds
        return max(self.min_font_size, min(estimated_size, self.max_font_size))


    def _likely_different_bubbles(self, region1: TextRegion, region2: TextRegion) -> bool:
        """Detect if regions are likely in different speech bubbles based on spatial patterns"""
        x1, y1, w1, h1 = region1.bounding_box
        x2, y2, w2, h2 = region2.bounding_box
        
        # Calculate gaps and positions
        horizontal_gap = 0
        if x1 + w1 < x2:
            horizontal_gap = x2 - (x1 + w1)
        elif x2 + w2 < x1:
            horizontal_gap = x1 - (x2 + w2)
        
        vertical_gap = 0
        if y1 + h1 < y2:
            vertical_gap = y2 - (y1 + h1)
        elif y2 + h2 < y1:
            vertical_gap = y1 - (y2 + h2)
        
        # Calculate relative positions
        center_x1 = x1 + w1 / 2
        center_x2 = x2 + w2 / 2
        center_y1 = y1 + h1 / 2
        center_y2 = y2 + h2 / 2
        
        horizontal_center_diff = abs(center_x1 - center_x2)
        avg_width = (w1 + w2) / 2
        
        # FIRST CHECK: Very small gaps always indicate same bubble
        if horizontal_gap < 15 and vertical_gap < 15:
            return False  # Definitely same bubble
        
        # STRICTER CHECK: For regions that are horizontally far apart
        # Even if they pass the gap threshold, check if they're likely different bubbles
        if horizontal_gap > 40:  # Significant horizontal gap
            # Unless they're VERY well aligned vertically, they're different bubbles
            vertical_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
            min_height = min(h1, h2)
            
            if vertical_overlap < min_height * 0.8:  # Need 80% overlap to be same bubble
                return True
        
        # SPECIFIC FIX: Check for multi-line text pattern
        # If regions are well-aligned horizontally, they're likely in the same bubble
        if horizontal_center_diff < avg_width * 0.35:  # Relaxed from 0.2 to 0.35
            # Additional checks for multi-line text:
            # 1. Similar widths (common in speech bubbles)
            width_ratio = max(w1, w2) / min(w1, w2) if min(w1, w2) > 0 else 999
            
            # 2. Reasonable vertical spacing (not too far apart)
            avg_height = (h1 + h2) / 2
            
            if width_ratio < 2.0 and vertical_gap < avg_height * 1.5:
                # This is very likely multi-line text in the same bubble
                return False
        
        # Pattern 1: Side-by-side bubbles (common in manga)
        # Characteristics: Significant horizontal gap, similar vertical position
        if horizontal_gap > 50:  # Increased from 25 to avoid false positives
            vertical_overlap = min(y1 + h1, y2 + h2) - max(y1, y2)
            min_height = min(h1, h2)
            
            # If they have good vertical overlap, they're likely side-by-side bubbles
            if vertical_overlap > min_height * 0.5:
                return True
        
        # Pattern 2: Stacked bubbles
        # Characteristics: Significant vertical gap, similar horizontal position
        if vertical_gap > 25:  # Back to original threshold
            horizontal_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
            min_width = min(w1, w2)
            
            # If they have good horizontal overlap, they're likely stacked bubbles
            if horizontal_overlap > min_width * 0.5:
                return True
        
        # Pattern 3: Diagonal arrangement (different speakers)
        # If regions are separated both horizontally and vertically
        if horizontal_gap > 20 and vertical_gap > 20:
            return True
        
        # Pattern 4: Large gap relative to region size
        avg_height = (h1 + h2) / 2
        
        if horizontal_gap > avg_width * 0.6 or vertical_gap > avg_height * 0.6:
            return True
        
        return False

    def _regions_should_merge(self, region1: TextRegion, region2: TextRegion, threshold: int = 50) -> bool:
        """Determine if two regions should be merged - with bubble detection"""
        
        # First check if they're close enough spatially
        if not self._regions_are_nearby(region1, region2, threshold):
            return False
        
        x1, y1, w1, h1 = region1.bounding_box
        x2, y2, w2, h2 = region2.bounding_box
        
        # ONLY apply special handling if regions are from Azure
        if hasattr(region1, 'from_azure') and region1.from_azure:
            # Azure lines are typically small - be more lenient
            avg_height = (h1 + h2) / 2
            if avg_height < 50:  # Likely single lines
                self._log(f"   Azure lines detected, using lenient merge criteria", "info")
                
                center_x1 = x1 + w1 / 2
                center_x2 = x2 + w2 / 2
                horizontal_center_diff = abs(center_x1 - center_x2)
                avg_width = (w1 + w2) / 2
                
                # If horizontally aligned and nearby, merge them
                if horizontal_center_diff < avg_width * 0.7:
                    return True
        
        # GOOGLE LOGIC - unchanged from your original
        # SPECIAL CASE: If one region is very small, bypass strict checks
        area1 = w1 * h1
        area2 = w2 * h2
        if area1 < 500 or area2 < 500:
            self._log(f"   Small text region (area: {min(area1, area2)}), bypassing strict alignment checks", "info")
            return True
        
        # Calculate actual gaps between regions
        horizontal_gap = 0
        if x1 + w1 < x2:
            horizontal_gap = x2 - (x1 + w1)
        elif x2 + w2 < x1:
            horizontal_gap = x1 - (x2 + w2)
        
        vertical_gap = 0
        if y1 + h1 < y2:
            vertical_gap = y2 - (y1 + h1)
        elif y2 + h2 < y1:
            vertical_gap = y1 - (y2 + h2)
        
        # Calculate centers for alignment checks
        center_x1 = x1 + w1 / 2
        center_x2 = x2 + w2 / 2
        center_y1 = y1 + h1 / 2
        center_y2 = y2 + h2 / 2
        
        horizontal_center_diff = abs(center_x1 - center_x2)
        vertical_center_diff = abs(center_y1 - center_y2)
        
        avg_width = (w1 + w2) / 2
        avg_height = (h1 + h2) / 2
        
        # Determine text orientation and layout
        is_horizontal_text = horizontal_gap > vertical_gap or (horizontal_center_diff < avg_width * 0.5)
        is_vertical_text = vertical_gap > horizontal_gap or (vertical_center_diff < avg_height * 0.5)
        
        # PRELIMINARY CHECK: If regions overlap or are extremely close, merge them
        # This handles text that's clearly in the same bubble
        
        # Check for overlap
        overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        has_overlap = overlap_x > 0 and overlap_y > 0
        
        if has_overlap:
            self._log(f"   Regions overlap - definitely same bubble, merging", "info")
            return True
        
        # If gaps are tiny (< 10 pixels), merge regardless of other factors
        if horizontal_gap < 10 and vertical_gap < 10:
            self._log(f"   Very small gaps ({horizontal_gap}, {vertical_gap}) - merging", "info")
            return True
        
        # BUBBLE BOUNDARY CHECK: Use spatial patterns to detect different bubbles
        # But be less aggressive if gaps are small
        if horizontal_gap < 20 and vertical_gap < 20:
            # Very close regions are almost certainly in the same bubble
            self._log(f"   Regions very close, skipping bubble boundary check", "info")
        elif self._likely_different_bubbles(region1, region2):
            self._log(f"   Regions likely in different speech bubbles", "info")
            return False
        
        # CHECK 1: For well-aligned text with small gaps, merge immediately
        # This catches multi-line text in the same bubble
        if is_horizontal_text and vertical_center_diff < avg_height * 0.4:
            # Horizontal text that's well-aligned vertically
            if horizontal_gap <= threshold and vertical_gap <= threshold * 0.5:
                self._log(f"   Well-aligned horizontal text with acceptable gaps, merging", "info")
                return True
        
        if is_vertical_text and horizontal_center_diff < avg_width * 0.4:
            # Vertical text that's well-aligned horizontally
            if vertical_gap <= threshold and horizontal_gap <= threshold * 0.5:
                self._log(f"   Well-aligned vertical text with acceptable gaps, merging", "info")
                return True
        
        # ADDITIONAL CHECK: Multi-line text in speech bubbles
        # Even if not perfectly aligned, check for typical multi-line patterns
        if horizontal_center_diff < avg_width * 0.5 and vertical_gap <= threshold:
            # Lines that are reasonably centered and within threshold should merge
            self._log(f"   Multi-line text pattern detected, merging", "info")
            return True
        
        # CHECK 2: Check alignment quality
        # Poor alignment often indicates different bubbles
        if is_horizontal_text:
            # For horizontal text, check vertical alignment
            if vertical_center_diff > avg_height * 0.6:
                self._log(f"   Poor vertical alignment for horizontal text", "info")
                return False
        elif is_vertical_text:
            # For vertical text, check horizontal alignment
            if horizontal_center_diff > avg_width * 0.6:
                self._log(f"   Poor horizontal alignment for vertical text", "info")
                return False
        
        # CHECK 3: Font size check (but be reasonable)
        font_size1 = self._estimate_font_size_for_region(region1)
        font_size2 = self._estimate_font_size_for_region(region2)
        size_ratio = max(font_size1, font_size2) / max(min(font_size1, font_size2), 1)
        
        # Allow some variation for emphasis or stylistic choices
        if size_ratio > 2.0:
            self._log(f"   Font sizes too different ({font_size1} vs {font_size2})", "info")
            return False
        
        # CHECK 4: Final sanity check on merged area
        merged_width = max(x1 + w1, x2 + w2) - min(x1, x2)
        merged_height = max(y1 + h1, y2 + h2) - min(y1, y2)
        merged_area = merged_width * merged_height
        combined_area = (w1 * h1) + (w2 * h2)
        
        # If merged area is way larger than combined areas, they're probably far apart
        if merged_area > combined_area * 2.5:
            self._log(f"   Merged area indicates regions are too far apart", "info")
            return False
        
        # If we get here, apply standard threshold checks
        if horizontal_gap <= threshold and vertical_gap <= threshold:
            self._log(f"   Standard threshold check passed, merging", "info")
            return True
        
        self._log(f"   No merge conditions met", "info")
        return False

    def _merge_nearby_regions(self, regions: List[TextRegion], threshold: int = 50) -> List[TextRegion]:
        """Merge text regions that are likely part of the same speech bubble - with debug logging"""
        if len(regions) <= 1:
            return regions
        
        self._log(f"\n=== MERGE DEBUG: Starting merge analysis ===", "info")
        self._log(f"  Total regions: {len(regions)}", "info")
        self._log(f"  Threshold: {threshold}px", "info")
        
        # First, let's log what regions we have
        for i, region in enumerate(regions):
            x, y, w, h = region.bounding_box
            self._log(f"  Region {i}: pos({x},{y}) size({w}x{h}) text='{region.text[:20]}...'", "info")
        
        # Sort regions by area (largest first) to handle contained regions properly
        sorted_indices = sorted(range(len(regions)), 
                              key=lambda i: regions[i].bounding_box[2] * regions[i].bounding_box[3], 
                              reverse=True)
        
        merged = []
        used = set()
        
        # Process each region in order of size (largest first)
        for idx in sorted_indices:
            i = idx
            if i in used:
                continue
            
            region1 = regions[i]
            
            # Start with this region
            merged_text = region1.text
            merged_vertices = list(region1.vertices) if hasattr(region1, 'vertices') else []
            regions_merged = [i]  # Track which regions were merged
            
            self._log(f"\n  Checking region {i} for merges:", "info")
            
            # Check against all other unused regions
            for j in range(len(regions)):
                if j == i or j in used:
                    continue
                
                region2 = regions[j]
                self._log(f"    Testing merge with region {j}:", "info")
                
                # Check if region2 is contained within region1
                x1, y1, w1, h1 = region1.bounding_box
                x2, y2, w2, h2 = region2.bounding_box
                
                # Check if region2 is fully contained within region1
                if (x2 >= x1 and y2 >= y1 and 
                    x2 + w2 <= x1 + w1 and y2 + h2 <= y1 + h1):
                    self._log(f"      ✓ Region {j} is INSIDE region {i} - merging!", "success")
                    merged_text += " " + region2.text
                    if hasattr(region2, 'vertices'):
                        merged_vertices.extend(region2.vertices)
                    used.add(j)
                    regions_merged.append(j)
                    continue
                
                # Check if region1 is contained within region2 (shouldn't happen due to sorting, but be safe)
                if (x1 >= x2 and y1 >= y2 and 
                    x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2):
                    self._log(f"      ✓ Region {i} is INSIDE region {j} - merging!", "success")
                    merged_text += " " + region2.text
                    if hasattr(region2, 'vertices'):
                        merged_vertices.extend(region2.vertices)
                    used.add(j)
                    regions_merged.append(j)
                    # Update region1's bounding box to the larger region
                    region1 = TextRegion(
                        text=merged_text,
                        vertices=merged_vertices,
                        bounding_box=region2.bounding_box,
                        confidence=region1.confidence,
                        region_type='temp_merge'
                    )
                    continue
                
                # FIX: Always check proximity against ORIGINAL regions, not the expanded one
                # This prevents cascade merging across bubble boundaries
                if self._regions_are_nearby(regions[i], region2, threshold):  # Use regions[i] not region1
                    #self._log(f"      ✓ Regions are nearby", "info")
                    
                    # Then check if they should merge (also use original region)
                    if self._regions_should_merge(regions[i], region2, threshold):  # Use regions[i] not region1
                        #self._log(f"      ✓ Regions should merge!", "success")
                        
                        # Actually perform the merge
                        merged_text += " " + region2.text
                        if hasattr(region2, 'vertices'):
                            merged_vertices.extend(region2.vertices)
                        used.add(j)
                        regions_merged.append(j)
                        
                        # DON'T update region1 for proximity checks - keep using original regions
                    else:
                        self._log(f"      ✗ Regions should not merge", "warning")
                else:
                    self._log(f"      ✗ Regions not nearby", "warning")
            
            # Log if we merged multiple regions
            if len(regions_merged) > 1:
                self._log(f"  ✅ MERGED regions {regions_merged} into one bubble", "success")
            else:
                self._log(f"  ℹ️ Region {i} not merged with any other", "info")
            
            # Create final merged region with all the merged vertices
            if merged_vertices:
                xs = [v[0] for v in merged_vertices]
                ys = [v[1] for v in merged_vertices]
            else:
                # Fallback: calculate from all merged regions
                all_xs = []
                all_ys = []
                for idx in regions_merged:
                    x, y, w, h = regions[idx].bounding_box
                    all_xs.extend([x, x + w])
                    all_ys.extend([y, y + h])
                xs = all_xs
                ys = all_ys
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            merged_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
            
            merged_region = TextRegion(
                text=merged_text,
                vertices=merged_vertices,
                bounding_box=merged_bbox,
                confidence=regions[i].confidence,
                region_type='merged_text_block' if len(regions_merged) > 1 else regions[i].region_type
            )
            
            # Copy over any additional attributes
            if hasattr(regions[i], 'translated_text'):
                merged_region.translated_text = regions[i].translated_text
            
            merged.append(merged_region)
            used.add(i)
        
        self._log(f"\n=== MERGE DEBUG: Complete ===", "info")
        self._log(f"  Final region count: {len(merged)} (was {len(regions)})", "info")
        
        # Verify the merge worked
        if len(merged) == len(regions):
            self._log(f"  ⚠️ WARNING: No regions were actually merged!", "warning")
        
        return merged
    
    def _regions_are_nearby(self, region1: TextRegion, region2: TextRegion, threshold: int = 50) -> bool:
        """Check if two regions are close enough to be in the same bubble - WITH DEBUG"""
        x1, y1, w1, h1 = region1.bounding_box
        x2, y2, w2, h2 = region2.bounding_box
        
        #self._log(f"\n    === NEARBY CHECK DEBUG ===", "info")
        #self._log(f"    Region 1: pos({x1},{y1}) size({w1}x{h1})", "info")
        #self._log(f"    Region 2: pos({x2},{y2}) size({w2}x{h2})", "info")
        #self._log(f"    Threshold: {threshold}", "info")
        
        # Calculate gaps between closest edges
        horizontal_gap = 0
        if x1 + w1 < x2:  # region1 is to the left
            horizontal_gap = x2 - (x1 + w1)
        elif x2 + w2 < x1:  # region2 is to the left
            horizontal_gap = x1 - (x2 + w2)
        
        vertical_gap = 0
        if y1 + h1 < y2:  # region1 is above
            vertical_gap = y2 - (y1 + h1)
        elif y2 + h2 < y1:  # region2 is above
            vertical_gap = y1 - (y2 + h2)
        
        #self._log(f"    Horizontal gap: {horizontal_gap}", "info")
        #self._log(f"    Vertical gap: {vertical_gap}", "info")
        
        # Detect if regions are likely vertical text based on aspect ratio
        aspect1 = w1 / max(h1, 1)
        aspect2 = w2 / max(h2, 1)
        
        # More permissive vertical text detection
        # Vertical text typically has aspect ratio < 1.0 (taller than wide)
        is_vertical_text = (aspect1 < 1.0 and aspect2 < 1.0) or (aspect1 < 0.5 or aspect2 < 0.5)
        
        # Also check if text is arranged vertically (one above the other with minimal horizontal offset)
        center_x1 = x1 + w1 / 2
        center_x2 = x2 + w2 / 2
        horizontal_center_diff = abs(center_x1 - center_x2)
        avg_width = (w1 + w2) / 2
        
        # If regions are vertically stacked with aligned centers, treat as vertical text
        is_vertically_stacked = (horizontal_center_diff < avg_width * 1.5) and (vertical_gap >= 0)
        
        #self._log(f"    Is vertical text: {is_vertical_text}", "info")
        #self._log(f"    Is vertically stacked: {is_vertically_stacked}", "info")
        #self._log(f"    Horizontal center diff: {horizontal_center_diff:.1f}", "info")
        
        # SIMPLE APPROACH: Just check if gaps are within threshold
        # Don't overthink it
        if horizontal_gap <= threshold and vertical_gap <= threshold:
            #self._log(f"    ✅ NEARBY: Both gaps within threshold", "success")
            return True
        
        # SPECIAL CASE: Vertically stacked text with good alignment
        # This is specifically for multi-line text in bubbles
        if horizontal_center_diff < avg_width * 0.8 and vertical_gap <= threshold * 1.5:
            #self._log(f"    ✅ NEARBY: Vertically aligned text in same bubble", "success")
            return True
        
        # If one gap is small and the other is slightly over, still consider nearby
        if (horizontal_gap <= threshold * 0.5 and vertical_gap <= threshold * 1.5) or \
           (vertical_gap <= threshold * 0.5 and horizontal_gap <= threshold * 1.5):
            #self._log(f"    ✅ NEARBY: One small gap, other slightly over", "success")
            return True
        
        # Special case: Wide bubbles with text on sides
        # If regions are at nearly the same vertical position, they might be in a wide bubble
        if abs(y1 - y2) < 10:  # Nearly same vertical position
            # Check if this could be a wide bubble spanning both regions
            if horizontal_gap <= threshold * 3:  # Allow up to 3x threshold for wide bubbles
                #self._log(f"    ✅ NEARBY: Same vertical level, possibly wide bubble", "success")
                return True
        
        #self._log(f"    ❌ NOT NEARBY: Gaps exceed threshold", "warning")
        return False
    
    def _find_font(self) -> str:
        """Find a suitable font for text rendering"""
        font_candidates = [
            "C:/Windows/Fonts/comicbd.ttf",  # Comic Sans MS Bold as first choice
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf", 
            "C:/Windows/Fonts/tahoma.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]
        
        for font_path in font_candidates:
            if os.path.exists(font_path):
                return font_path
        
        return None  # Will use default font
    
    def _get_singleton_bubble_detector(self):
        """Get or initialize the singleton bubble detector instance with load coordination."""
        start_time = None
        with MangaTranslator._singleton_lock:
            if MangaTranslator._singleton_bubble_detector is not None:
                self._log("🤖 Using bubble detector (already loaded)", "info")
                MangaTranslator._singleton_refs += 1
                return MangaTranslator._singleton_bubble_detector
            # If another thread is loading, wait for it
            if MangaTranslator._singleton_bd_loading:
                self._log("⏳ Waiting for bubble detector to finish loading (singleton)", "debug")
                evt = MangaTranslator._singleton_bd_event
                # Drop the lock while waiting
                pass
            else:
                # Mark as loading and proceed to load outside lock
                MangaTranslator._singleton_bd_loading = True
                MangaTranslator._singleton_bd_event.clear()
                start_time = time.time()
                # Release lock and perform heavy load
                pass
        # Outside the lock: perform load or wait
        if start_time is None:
            # We are a waiter
            try:
                MangaTranslator._singleton_bd_event.wait(timeout=300)
            except Exception:
                pass
            with MangaTranslator._singleton_lock:
                if MangaTranslator._singleton_bubble_detector is not None:
                    MangaTranslator._singleton_refs += 1
                return MangaTranslator._singleton_bubble_detector
        else:
            # We are the loader
            try:
                from bubble_detector import BubbleDetector
                bd = None
                
                # First, try to get a preloaded detector from the pool
                try:
                    ocr_settings = self.main_gui.config.get('manga_settings', {}).get('ocr', {}) if hasattr(self, 'main_gui') else {}
                    det_type = ocr_settings.get('detector_type', 'rtdetr_onnx')
                    model_id = ocr_settings.get('rtdetr_model_url') or ocr_settings.get('bubble_model_path') or ''
                    key = (det_type, model_id)
                    self._log(f"[DEBUG] Looking for detector in pool with key: {key}", "debug")
                    with MangaTranslator._detector_pool_lock:
                        self._log(f"[DEBUG] Pool keys available: {list(MangaTranslator._detector_pool.keys())}", "debug")
                        rec = MangaTranslator._detector_pool.get(key)
                        if rec and isinstance(rec, dict):
                            spares = rec.get('spares') or []
                            self._log(f"[DEBUG] Found pool record with {len(spares)} spares", "debug")
                            # For singleton mode, we can use a pool instance without checking it out
                            # since the singleton will keep it loaded permanently
                            if spares:
                                # Just use the first spare (don't pop or check out)
                                # Singleton will keep it loaded, pool can still track it
                                bd = spares[0]
                                self._log(f"🤖 Using pool bubble detector for singleton (no check-out needed)", "info")
                        else:
                            self._log(f"[DEBUG] No pool record found for key: {key}", "debug")
                except Exception as e:
                    self._log(f"Could not fetch preloaded detector: {e}", "debug")
                
                # If no preloaded detector, create a new one
                if bd is None:
                    bd = BubbleDetector()
                    self._log("🤖 Created new bubble detector instance", "info")
                
                # Optionally: defer model load until first actual call inside BD; keeping instance resident
                with MangaTranslator._singleton_lock:
                    MangaTranslator._singleton_bubble_detector = bd
                    MangaTranslator._singleton_refs += 1
                    MangaTranslator._singleton_bd_loading = False
                    try:
                        MangaTranslator._singleton_bd_event.set()
                    except Exception:
                        pass
                elapsed = time.time() - start_time
                self._log(f"🤖 Singleton bubble detector ready (took {elapsed:.2f}s)", "info")
                return bd
            except Exception as e:
                with MangaTranslator._singleton_lock:
                    MangaTranslator._singleton_bd_loading = False
                    try:
                        MangaTranslator._singleton_bd_event.set()
                    except Exception:
                        pass
                self._log(f"Failed to create singleton bubble detector: {e}", "error")
                return None
    
    def _initialize_singleton_local_inpainter(self):
        """Initialize singleton local inpainter instance"""
        with MangaTranslator._singleton_lock:
            was_existing = MangaTranslator._singleton_local_inpainter is not None
            if MangaTranslator._singleton_local_inpainter is None:
                try:
                    from local_inpainter import LocalInpainter
                    local_method = self.manga_settings.get('inpainting', {}).get('local_method', 'anime')
                    # LocalInpainter only accepts config_path, not method
                    MangaTranslator._singleton_local_inpainter = LocalInpainter()
                    # Now load the model with the specified method
                    if local_method:
                        # Try to load the model
                        model_path = self.manga_settings.get('inpainting', {}).get('local_model_path')
                        if not model_path:
                            # Try to download if no path specified
                            try:
                                model_path = MangaTranslator._singleton_local_inpainter.download_jit_model(local_method)
                            except Exception as e:
                                self._log(f"⚠️ Failed to download model for {local_method}: {e}", "warning")
                        
                        if model_path and os.path.exists(model_path):
                            success = MangaTranslator._singleton_local_inpainter.load_model_with_retry(local_method, model_path)
                            if success:
                                self._log(f"🎨 Created singleton local inpainter with {local_method} model", "info")
                            else:
                                self._log(f"⚠️ Failed to load {local_method} model", "warning")
                        else:
                            self._log(f"🎨 Created singleton local inpainter (no model loaded yet)", "info")
                    else:
                        self._log(f"🎨 Created singleton local inpainter (default)", "info")
                except Exception as e:
                    self._log(f"Failed to create singleton local inpainter: {e}", "error")
                    return
            # Use the singleton instance
            self.local_inpainter = MangaTranslator._singleton_local_inpainter
            self.inpainter = self.local_inpainter
            MangaTranslator._singleton_refs += 1
            if was_existing:
                self._log("🎨 Using local inpainter (already loaded)", "info")
    
    def _get_thread_bubble_detector(self):
        """Get or initialize bubble detector (singleton or thread-local based on settings).
        Will consume a preloaded detector if available for current settings.
        """
        if getattr(self, 'use_singleton_bubble_detector', False) or (hasattr(self, 'use_singleton_models') and self.use_singleton_models):
            # Use singleton instance (preferred)
            if self.bubble_detector is None:
                self.bubble_detector = self._get_singleton_bubble_detector()
            return self.bubble_detector
        else:
            # Use thread-local instance (original behavior for parallel processing)
            if not hasattr(self, '_thread_local') or getattr(self, '_thread_local', None) is None:
                self._thread_local = threading.local()
            if not hasattr(self._thread_local, 'bubble_detector') or self._thread_local.bubble_detector is None:
                from bubble_detector import BubbleDetector
                # Try to check out a preloaded spare for the current detector settings
                try:
                    ocr_settings = self.main_gui.config.get('manga_settings', {}).get('ocr', {}) if hasattr(self, 'main_gui') else {}
                    det_type = ocr_settings.get('detector_type', 'rtdetr_onnx')
                    model_id = ocr_settings.get('rtdetr_model_url') or ocr_settings.get('bubble_model_path') or ''
                    key = (det_type, model_id)
                    with MangaTranslator._detector_pool_lock:
                        rec = MangaTranslator._detector_pool.get(key)
                        if rec and isinstance(rec, dict):
                            spares = rec.get('spares') or []
                            # Initialize checked_out list if it doesn't exist
                            if 'checked_out' not in rec:
                                rec['checked_out'] = []
                            checked_out = rec['checked_out']
                            
                            # Look for an available spare (not checked out)
                            if spares:
                                for spare in spares:
                                    if spare not in checked_out and spare:
                                        # Check out this spare instance
                                        checked_out.append(spare)
                                        self._thread_local.bubble_detector = spare
                                        # Store references for later return
                                        self._checked_out_bubble_detector = spare
                                        self._bubble_detector_pool_key = key
                                        self._log(f"🤖 Checked out bubble detector from pool ({len(checked_out)}/{len(spares)} in use)", "info")
                                        break
                except Exception:
                    pass
                # If still not set, create a fresh detector and store it for future use
                if not hasattr(self._thread_local, 'bubble_detector') or self._thread_local.bubble_detector is None:
                    self._thread_local.bubble_detector = BubbleDetector()
                    self._log("🤖 Created thread-local bubble detector", "debug")
                    
                    # Store this new detector in the pool for future reuse
                    try:
                        with MangaTranslator._detector_pool_lock:
                            if key not in MangaTranslator._detector_pool:
                                MangaTranslator._detector_pool[key] = {'spares': [], 'checked_out': []}
                            # Add this new detector to spares and immediately check it out
                            rec = MangaTranslator._detector_pool[key]
                            if 'spares' not in rec:
                                rec['spares'] = []
                            if 'checked_out' not in rec:
                                rec['checked_out'] = []
                            rec['spares'].append(self._thread_local.bubble_detector)
                            rec['checked_out'].append(self._thread_local.bubble_detector)
                            # Store references for later return
                            self._checked_out_bubble_detector = self._thread_local.bubble_detector
                            self._bubble_detector_pool_key = key
                    except Exception:
                        pass
            return self._thread_local.bubble_detector
    
    def _get_thread_local_inpainter(self, local_method: str, model_path: str):
        """Get or create a LocalInpainter (singleton or thread-local based on settings).
        Loads the requested model if needed.
        """
        if hasattr(self, 'use_singleton_models') and self.use_singleton_models:
            # Use singleton instance
            if self.local_inpainter is None:
                self._initialize_singleton_local_inpainter()
            return self.local_inpainter
        
        # Use thread-local instance (original behavior for parallel processing)
        # Ensure thread-local storage exists and has a dict
        tl = getattr(self, '_thread_local', None)
        if tl is None:
            self._thread_local = threading.local()
            tl = self._thread_local
        if not hasattr(tl, 'local_inpainters') or getattr(tl, 'local_inpainters', None) is None:
            tl.local_inpainters = {}
        key = (local_method or 'anime', model_path or '')
        if key not in tl.local_inpainters or tl.local_inpainters[key] is None:
            # First, try to use a preloaded spare instance from the shared pool
            try:
                rec = MangaTranslator._inpaint_pool.get(key)
                if rec and isinstance(rec, dict):
                    spares = rec.get('spares') or []
                    if spares:
                        tl.local_inpainters[key] = spares.pop(0)
                        self._log("🎨 Using preloaded local inpainting instance", "info")
                        return tl.local_inpainters[key]
                    # If there's a fully loaded shared instance but no spares, use it as a last resort
                    if rec.get('loaded') and rec.get('inpainter') is not None:
                        tl.local_inpainters[key] = rec.get('inpainter')
                        self._log("🎨 Using shared preloaded inpainting instance", "info")
                        return tl.local_inpainters[key]
            except Exception:
                pass
            
            # No preloaded instance available: create and load thread-local instance
            try:
                from local_inpainter import LocalInpainter
                # Use a per-thread config path to avoid concurrent JSON writes
                try:
                    import tempfile
                    thread_cfg = os.path.join(tempfile.gettempdir(), f"gl_inpainter_{threading.get_ident()}.json")
                except Exception:
                    thread_cfg = "config_thread_local.json"
                inp = LocalInpainter(config_path=thread_cfg)
                # Apply tiling settings
                tiling_settings = self.manga_settings.get('tiling', {}) if hasattr(self, 'manga_settings') else {}
                inp.tiling_enabled = tiling_settings.get('enabled', False)
                inp.tile_size = tiling_settings.get('tile_size', 512)
                inp.tile_overlap = tiling_settings.get('tile_overlap', 64)
                
                # Ensure model is available
                resolved_model_path = model_path
                if not resolved_model_path or not os.path.exists(resolved_model_path):
                    try:
                        resolved_model_path = inp.download_jit_model(local_method)
                    except Exception as e:
                        self._log(f"⚠️ JIT model download failed for {local_method}: {e}", "warning")
                        resolved_model_path = None
                
                # Load model for this thread's instance
                if resolved_model_path and os.path.exists(resolved_model_path):
                    try:
                        self._log(f"📥 Loading {local_method} inpainting model (thread-local)", "info")
                        inp.load_model_with_retry(local_method, resolved_model_path, force_reload=False)
                    except Exception as e:
                        self._log(f"⚠️ Thread-local inpainter load error: {e}", "warning")
                else:
                    self._log("⚠️ No model path available for thread-local inpainter", "warning")
                
                # Re-check thread-local and publish ONLY if model loaded successfully
                tl2 = getattr(self, '_thread_local', None)
                if tl2 is None:
                    self._thread_local = threading.local()
                    tl2 = self._thread_local
                if not hasattr(tl2, 'local_inpainters') or getattr(tl2, 'local_inpainters', None) is None:
                    tl2.local_inpainters = {}
                if getattr(inp, 'model_loaded', False):
                    tl2.local_inpainters[key] = inp
                    
                    # Store this loaded instance info in the pool for future reuse
                    try:
                        with MangaTranslator._inpaint_pool_lock:
                            if key not in MangaTranslator._inpaint_pool:
                                MangaTranslator._inpaint_pool[key] = {'inpainter': None, 'loaded': False, 'event': threading.Event(), 'spares': []}
                            # Mark that we have a loaded instance available
                            MangaTranslator._inpaint_pool[key]['loaded'] = True
                            MangaTranslator._inpaint_pool[key]['inpainter'] = inp  # Store reference
                            if MangaTranslator._inpaint_pool[key].get('event'):
                                MangaTranslator._inpaint_pool[key]['event'].set()
                    except Exception:
                        pass
                else:
                    # Ensure future calls will attempt a fresh init instead of using a half-initialized instance
                    tl2.local_inpainters[key] = None
            except Exception as e:
                self._log(f"❌ Failed to create thread-local inpainter: {e}", "error")
                try:
                    tl3 = getattr(self, '_thread_local', None)
                    if tl3 is None:
                        self._thread_local = threading.local()
                        tl3 = self._thread_local
                    if not hasattr(tl3, 'local_inpainters') or getattr(tl3, 'local_inpainters', None) is None:
                        tl3.local_inpainters = {}
                    tl3.local_inpainters[key] = None
                except Exception:
                    pass
        return getattr(self._thread_local, 'local_inpainters', {}).get(key)
    
    def translate_regions(self, regions: List[TextRegion], image_path: str) -> List[TextRegion]:
        """Translate all text regions with API delay"""
        self._log(f"\n📝 Translating {len(regions)} text regions...")
        
        # Check stop before even starting
        if self._check_stop():
            self._log(f"\n⏹️ Translation stopped before processing any regions", "warning")
            return regions
        
        # Check if parallel processing is enabled
        parallel_enabled = self.manga_settings.get('advanced', {}).get('parallel_processing', False)
        max_workers = self.manga_settings.get('advanced', {}).get('max_workers', 4)
        
        if parallel_enabled and len(regions) > 1:
            self._log(f"🚀 Using PARALLEL processing with {max_workers} workers")
            return self._translate_regions_parallel(regions, image_path, max_workers)
        else:
            # SEQUENTIAL CODE
            for i, region in enumerate(regions):
                if self._check_stop():
                    self._log(f"\n⏹️ Translation stopped by user after {i}/{len(regions)} regions", "warning")
                    break            
                if region.text.strip():
                    self._log(f"\n[{i+1}/{len(regions)}] Original: {region.text}")
                    
                    # Get context for translation
                    context = self.translation_context[-5:] if self.contextual_enabled else None
                    
                    # Translate with image context
                    translated = self.translate_text(
                        region.text, 
                        context,
                        image_path=image_path,
                        region=region
                    )
                    region.translated_text = translated
                    
                    self._log(f"Translated: {translated}")
                    
                    # SAVE TO HISTORY HERE
                    if self.history_manager and self.contextual_enabled and translated:
                        try:
                            self.history_manager.append_to_history(
                                user_content=region.text,
                                assistant_content=translated,
                                hist_limit=self.translation_history_limit,
                                reset_on_limit=not self.rolling_history_enabled,
                                rolling_window=self.rolling_history_enabled
                            )
                            self._log(f"📚 Saved to history (exchange {i+1})")
                        except Exception as e:
                            self._log(f"⚠️ Failed to save history: {e}", "warning")
                    
                    # Apply API delay
                    if i < len(regions) - 1:  # Don't delay after last translation
                        self._log(f"⏳ Waiting {self.api_delay}s before next translation...")
                        # Check stop flag every 0.1 seconds during delay
                        for _ in range(int(self.api_delay * 10)):
                            if self._check_stop():
                                self._log(f"\n⏹️ Translation stopped during delay", "warning")
                                return regions
                            time.sleep(0.1)
            
            return regions

    #  parallel processing:

    def _wait_for_api_slot(self, min_interval=None, jitter_max=0.25):
        """Global, thread-safe front-edge rate limiter for API calls.
        Ensures parallel requests are spaced out before dispatch, avoiding tail latency.
        """
        import time
        import random
        import threading

        if min_interval is None:
            try:
                min_interval = float(getattr(self, "api_delay", 0.0))
            except Exception:
                min_interval = 0.0
        if min_interval < 0:
            min_interval = 0.0

        # Lazy init shared state
        if not hasattr(self, "_api_rl_lock"):
            self._api_rl_lock = threading.Lock()
            self._api_next_allowed = 0.0  # monotonic seconds

        while True:
            now = time.monotonic()
            with self._api_rl_lock:
                # If we're allowed now, book the next slot and proceed
                if now >= self._api_next_allowed:
                    jitter = random.uniform(0.0, max(jitter_max, 0.0)) if jitter_max else 0.0
                    self._api_next_allowed = now + min_interval + jitter
                    return

                # Otherwise compute wait time (don’t hold the lock while sleeping)
                wait = self._api_next_allowed - now

            # Sleep outside the lock in short increments so stop flags can be honored
            if wait > 0:
                try:
                    if self._check_stop():
                        return
                except Exception:
                    pass
                time.sleep(min(wait, 0.05))

    def _translate_regions_parallel(self, regions: List[TextRegion], image_path: str, max_workers: int = None) -> List[TextRegion]:
        """Translate regions using parallel processing"""
        # Get max_workers from settings if not provided
        if max_workers is None:
            max_workers = self.manga_settings.get('advanced', {}).get('max_workers', 4)
        
        # Override with API batch size when batch mode is enabled — these are API calls.
        try:
            if getattr(self, 'batch_mode', False):
                bs = int(getattr(self, 'batch_size', 0) or int(os.getenv('BATCH_SIZE', '0')))
                if bs and bs > 0:
                    max_workers = bs
        except Exception:
            pass
        # Bound to number of regions
        max_workers = max(1, min(max_workers, len(regions)))
        
        # Thread-safe storage for results
        results_lock = threading.Lock()
        translated_regions = {}
        failed_indices = []
        
        # Filter out empty regions
        valid_regions = [(i, region) for i, region in enumerate(regions) if region.text.strip()]
        
        if not valid_regions:
            return regions
        
        # Create a thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all translation tasks
            future_to_data = {}
            
            for i, region in valid_regions:
                # Check for stop signal before submitting
                if self._check_stop():
                    self._log(f"\n⏹️ Translation stopped before submitting region {i+1}", "warning")
                    break
                
                # Submit translation task
                future = executor.submit(
                    self._translate_single_region_parallel, 
                    region, 
                    i, 
                    len(valid_regions),
                    image_path
                )
                future_to_data[future] = (i, region)
            
            # Process completed translations
            completed = 0
            for future in as_completed(future_to_data):
                i, region = future_to_data[future]
                
                # Check for stop signal
                if self._check_stop():
                    self._log(f"\n⏹️ Translation stopped at {completed}/{len(valid_regions)} completed", "warning")
                    # Cancel remaining futures
                    for f in future_to_data:
                        f.cancel()
                    break
                
                try:
                    translated_text = future.result()
                    if translated_text:
                        with results_lock:
                            translated_regions[i] = translated_text
                        completed += 1
                        self._log(f"✅ [{completed}/{len(valid_regions)}] Completed region {i+1}")
                    else:
                        with results_lock:
                            failed_indices.append(i)
                        self._log(f"❌ [{completed}/{len(valid_regions)}] Failed region {i+1}", "error")
                
                except Exception as e:
                    with results_lock:
                        failed_indices.append(i)
                    self._log(f"❌ Error in region {i+1}: {str(e)}", "error")
        
        # Apply translations back to regions
        for i, region in enumerate(regions):
            if i in translated_regions:
                region.translated_text = translated_regions[i]
        
        # Report summary
        success_count = len(translated_regions)
        fail_count = len(failed_indices)
        self._log(f"\n📊 Parallel translation complete: {success_count} succeeded, {fail_count} failed")
        
        return regions

    def reset_for_new_image(self):
        """Reset internal state for processing a new image"""
        # ============================================================
        # CRITICAL: COMPREHENSIVE CACHE CLEARING FOR NEW IMAGE
        # This ensures NO text data leaks between images
        # ============================================================
        
        # Clear any cached detection results
        if hasattr(self, 'last_detection_results'):
            del self.last_detection_results
        
        # FORCE clear OCR ROI cache (main text contamination source)
        # THREAD-SAFE: Use lock for parallel panel translation
        if hasattr(self, 'ocr_roi_cache'):
            with self._cache_lock:
                self.ocr_roi_cache.clear()
        self._current_image_hash = None
        
        # Clear OCR manager and ALL provider caches
        if hasattr(self, 'ocr_manager') and self.ocr_manager:
            if hasattr(self.ocr_manager, 'last_results'):
                self.ocr_manager.last_results = None
            if hasattr(self.ocr_manager, 'cache'):
                self.ocr_manager.cache.clear()
            # Clear ALL provider-level caches
            if hasattr(self.ocr_manager, 'providers'):
                for provider_name, provider in self.ocr_manager.providers.items():
                    if hasattr(provider, 'last_results'):
                        provider.last_results = None
                    if hasattr(provider, 'cache'):
                        provider.cache.clear()
        
        # Clear bubble detector cache
        if hasattr(self, 'bubble_detector') and self.bubble_detector:
            if hasattr(self.bubble_detector, 'last_detections'):
                self.bubble_detector.last_detections = None
            if hasattr(self.bubble_detector, 'cache'):
                self.bubble_detector.cache.clear()
        
        # Don't clear translation context if using rolling history
        if not self.rolling_history_enabled:
            self.translation_context = []
        
        # Clear any cached regions
        if hasattr(self, '_cached_regions'):
            del self._cached_regions
        
        self._log("🔄 Reset translator state for new image (ALL text caches cleared)", "debug")

    def _translate_single_region_parallel(self, region: TextRegion, index: int, total: int, image_path: str) -> Optional[str]:
        """Translate a single region for parallel processing"""
        try:
            thread_name = threading.current_thread().name
            self._log(f"\n[{thread_name}] [{index+1}/{total}] Original: {region.text}")
            
            # Note: Context is not used in parallel mode to avoid race conditions
            # Pass None for context to maintain compatibility with your translate_text method
            # Front-edge rate limiting across threads
            self._wait_for_api_slot()

            translated = self.translate_text(
                region.text,
                None,  # No context in parallel mode
                image_path=image_path,
                region=region
            )
            
            if translated:
                self._log(f"[{thread_name}] Translated: {translated}")
                return translated
            else:
                self._log(f"[{thread_name}] Translation failed", "error")
                return None
                
        except Exception as e:
            self._log(f"[{thread_name}] Error: {str(e)}", "error")
            return None


    def _is_bubble_detector_loaded(self, ocr_settings: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if the configured bubble detector's model is already loaded.
        Returns (loaded, detector_type). Safe: does not trigger a load.
        """
        try:
            bd = self._get_thread_bubble_detector()
        except Exception:
            return False, ocr_settings.get('detector_type', 'rtdetr_onnx')
        det = ocr_settings.get('detector_type', 'rtdetr_onnx')
        try:
            if det == 'rtdetr_onnx':
                return bool(getattr(bd, 'rtdetr_onnx_loaded', False)), det
            elif det == 'rtdetr':
                return bool(getattr(bd, 'rtdetr_loaded', False)), det
            elif det == 'yolo':
                return bool(getattr(bd, 'model_loaded', False)), det
            else:
                # Auto or unknown – consider any ready model as loaded
                ready = bool(getattr(bd, 'rtdetr_loaded', False) or getattr(bd, 'rtdetr_onnx_loaded', False) or getattr(bd, 'model_loaded', False))
                return ready, det
        except Exception:
            return False, det

    def _is_local_inpainter_loaded(self) -> Tuple[bool, Optional[str]]:
        """Check if a local inpainter model is already loaded for current settings.
        Returns (loaded, local_method) or (False, None).
        This respects UI flags: skip_inpainting / use_cloud_inpainting.
        """
        try:
            # If skipping or using cloud, this does not apply
            if getattr(self, 'skip_inpainting', False) or getattr(self, 'use_cloud_inpainting', False):
                return False, None
        except Exception:
            pass
        inpaint_cfg = self.manga_settings.get('inpainting', {}) if hasattr(self, 'manga_settings') else {}
        local_method = inpaint_cfg.get('local_method', 'anime')
        try:
            model_path = self.main_gui.config.get(f'manga_{local_method}_model_path', '') if hasattr(self, 'main_gui') else ''
        except Exception:
            model_path = ''
        # Singleton path
        if getattr(self, 'use_singleton_models', False):
            inp = getattr(MangaTranslator, '_singleton_local_inpainter', None)
            return (bool(getattr(inp, 'model_loaded', False)), local_method)
        # Thread-local/pooled path
        inp = getattr(self, 'local_inpainter', None)
        if inp is not None and getattr(inp, 'model_loaded', False):
            return True, local_method
        try:
            key = (local_method, model_path or '')
            rec = MangaTranslator._inpaint_pool.get(key)
            # Consider the shared 'inpainter' loaded or any spare that is model_loaded
            if rec:
                if rec.get('loaded') and rec.get('inpainter') is not None and getattr(rec['inpainter'], 'model_loaded', False):
                    return True, local_method
                for spare in rec.get('spares') or []:
                    if getattr(spare, 'model_loaded', False):
                        return True, local_method
        except Exception:
            pass
        return False, local_method

    def _log_model_status(self):
        """Emit concise status lines for already-loaded heavy models to avoid confusing 'loading' logs."""
        try:
            ocr_settings = self.manga_settings.get('ocr', {}) if hasattr(self, 'manga_settings') else {}
            if ocr_settings.get('bubble_detection_enabled', False):
                loaded, det = self._is_bubble_detector_loaded(ocr_settings)
                det_name = 'YOLO' if det == 'yolo' else ('RT-DETR' if det == 'rtdetr' else 'RTEDR_onnx')
                if loaded:
                    self._log("🤖 Using bubble detector (already loaded)", "info")
                else:
                    self._log("🤖 Bubble detector will load on first use", "debug")
        except Exception:
            pass
        try:
            loaded, local_method = self._is_local_inpainter_loaded()
            if local_method:
                label = local_method.upper()
                if loaded:
                    self._log("🎨 Using local inpainter (already loaded)", "info")
                else:
                    self._log("🎨 Local inpainter will load on first use", "debug")
        except Exception:
            pass

    def process_image(self, image_path: str, output_path: Optional[str] = None, 
                     batch_index: int = None, batch_total: int = None) -> Dict[str, Any]:
        """Process a single manga image through the full pipeline"""
        # Ensure local references exist for cleanup in finally
        image = None
        inpainted = None
        final_image = None
        mask = None
        mask_viz = None
        pil_image = None
        heatmap = None

        # Set batch tracking if provided
        if batch_index is not None and batch_total is not None:
            self.batch_current = batch_index
            self.batch_size = batch_total
            self.batch_mode = True
        
        # Simplified header for batch mode
        if not self.batch_mode:
            self._log(f"\n{'='*60}")
            self._log(f"📷 STARTING MANGA TRANSLATION PIPELINE")
            self._log(f"📁 Input: {image_path}")
            self._log(f"📁 Output: {output_path or 'Auto-generated'}")
            self._log(f"{'='*60}\n")
        else:
            self._log(f"\n[{batch_index}/{batch_total}] Processing: {os.path.basename(image_path)}")
        
        # Before heavy work, report model status to avoid confusing 'loading' logs later
        try:
            self._log_model_status()
        except Exception:
            pass
        
        result = {
            'success': False,
            'input_path': image_path,
            'output_path': output_path,
            'regions': [],
            'errors': [],
            'interrupted': False,
            'format_info': {}
        }
        
        try:
            # RAM cap gating before heavy processing
            try:
                self._block_if_over_cap("processing image")
            except Exception:
                pass

            # Determine the output directory from output_path
            if output_path:
                output_dir = os.path.dirname(output_path)
            else:
                # If no output path specified, use default
                output_dir = os.path.join(os.path.dirname(image_path), "translated_images")
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Initialize HistoryManager with the output directory
            if self.contextual_enabled and not self.history_manager_initialized:
                # Only initialize if we're in a new output directory
                if output_dir != getattr(self, 'history_output_dir', None):
                    try:
                        self.history_manager = HistoryManager(output_dir)
                        self.history_manager_initialized = True
                        self.history_output_dir = output_dir
                        self._log(f"📚 Initialized HistoryManager in output directory: {output_dir}")
                    except Exception as e:
                        self._log(f"⚠️ Failed to initialize history manager: {str(e)}", "warning")
                        self.history_manager = None
            
            # Check for stop signal
            if self._check_stop():
                result['interrupted'] = True
                self._log("⏹️ Translation stopped before processing", "warning")
                return result
            
            # Format detection if enabled
            if self.manga_settings.get('advanced', {}).get('format_detection', False):
                self._log("🔍 Analyzing image format...")
                img = Image.open(image_path)
                width, height = img.size
                aspect_ratio = height / width
                
                # Detect format type
                format_info = {
                    'width': width,
                    'height': height,
                    'aspect_ratio': aspect_ratio,
                    'is_webtoon': aspect_ratio > 3.0,
                    'is_spread': width > height * 1.3,
                    'format': 'unknown'
                }
                
                if format_info['is_webtoon']:
                    format_info['format'] = 'webtoon'
                    self._log("📱 Detected WEBTOON format - vertical scroll manga")
                elif format_info['is_spread']:
                    format_info['format'] = 'spread'
                    self._log("📖 Detected SPREAD format - two-page layout")
                else:
                    format_info['format'] = 'single_page'
                    self._log("📄 Detected SINGLE PAGE format")
                
                result['format_info'] = format_info
                
                # Handle webtoon mode if detected and enabled
                webtoon_mode = self.manga_settings.get('advanced', {}).get('webtoon_mode', 'auto')
                if format_info['is_webtoon'] and webtoon_mode != 'disabled':
                    if webtoon_mode == 'auto' or webtoon_mode == 'force':
                        self._log("🔄 Webtoon mode active - will process in chunks for better OCR")
                        # Process webtoon in chunks
                        return self._process_webtoon_chunks(image_path, output_path, result)
            
            # Step 1: Detect text regions using Google Cloud Vision
            self._log(f"📍 [STEP 1] Text Detection Phase")
            regions = self.detect_text_regions(image_path)
            
            if not regions:
                error_msg = "No text regions detected by Cloud Vision"
                self._log(f"⚠️ {error_msg}", "warning")
                result['errors'].append(error_msg)
                # Still save the original image as "translated" if no text found
                if output_path:
                    import shutil
                    shutil.copy2(image_path, output_path)
                    result['output_path'] = output_path
                result['success'] = True
                return result
            
            self._log(f"\n✅ Detection complete: {len(regions)} regions found")
            
            # Save debug outputs only if 'Save intermediate images' is enabled
            if self.manga_settings.get('advanced', {}).get('save_intermediate', False):
                self._save_debug_image(image_path, regions, debug_base_dir=output_dir)
            
            # Step 2: Translation & Inpainting (concurrent)
            self._log(f"\n📍 [STEP 2] Translation & Inpainting Phase (concurrent)")
            
            # Load image once (used by inpainting task); keep PIL fallback for Unicode paths
            import cv2
            self._log(f"🖼️ Loading image with OpenCV...")
            try:
                image = cv2.imread(image_path)
                if image is None:
                    self._log(f"   Using PIL to handle Unicode path...", "info")
                    from PIL import Image as PILImage
                    import numpy as np
                    pil_image = PILImage.open(image_path)
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    self._log(f"   ✅ Successfully loaded with PIL", "info")
            except Exception as e:
                error_msg = f"Failed to load image: {image_path} - {str(e)}"
                self._log(f"❌ {error_msg}", "error")
                result['errors'].append(error_msg)
                return result
            
            self._log(f"   Image dimensions: {image.shape[1]}x{image.shape[0]}")
            
            # Save intermediate original image if enabled
            if self.manga_settings.get('advanced', {}).get('save_intermediate', False):
                self._save_intermediate_image(image_path, image, "original", debug_base_dir=output_dir)
            
            # Check if we should continue before kicking off tasks
            if self._check_stop():
                result['interrupted'] = True
                self._log("⏹️ Translation stopped before concurrent phase", "warning")
                return result
            
            # Helper tasks
            def _task_translate():
                try:
                    if self.full_page_context_enabled:
                        # Full page context translation mode
                        self._log(f"\n📄 Using FULL PAGE CONTEXT mode")
                        self._log("   This mode sends all text together for more consistent translations", "info")
                        if self._check_stop():
                            return False
                        translations = self.translate_full_page_context(regions, image_path)
                        if translations:
                            translated_count = sum(1 for r in regions if getattr(r, 'translated_text', None) and r.translated_text and r.translated_text != r.text)
                            self._log(f"\n📊 Full page context translation complete: {translated_count}/{len(regions)} regions translated")
                            return True
                        else:
                            self._log("❌ Full page context translation failed", "error")
                            result['errors'].append("Full page context translation failed")
                            return False
                    else:
                        # Individual translation mode with parallel processing support
                        self._log(f"\n📝 Using INDIVIDUAL translation mode")
                        if self.manga_settings.get('advanced', {}).get('parallel_processing', False):
                            self._log("⚡ Parallel processing ENABLED")
                            _ = self._translate_regions_parallel(regions, image_path)
                        else:
                            _ = self.translate_regions(regions, image_path)
                        return True
                except Exception as te:
                    self._log(f"❌ Translation task error: {te}", "error")
                    return False
            
            def _task_inpaint():
                try:
                    if getattr(self, 'skip_inpainting', False):
                        self._log(f"🎨 Skipping inpainting (preserving original art)", "info")
                        return image.copy()
                    
                    self._log(f"🎭 Creating text mask...")
                    try:
                        self._block_if_over_cap("mask creation")
                    except Exception:
                        pass
                    mask_local = self.create_text_mask(image, regions)
                    
                    # Save mask and overlay only if 'Save intermediate images' is enabled
                    if self.manga_settings.get('advanced', {}).get('save_intermediate', False):
                        try:
                            debug_dir = os.path.join(output_dir, 'debug')
                            os.makedirs(debug_dir, exist_ok=True)
                            base_name = os.path.splitext(os.path.basename(image_path))[0]
                            mask_path = os.path.join(debug_dir, f"{base_name}_mask.png")
                            cv2.imwrite(mask_path, mask_local)
                            mask_percentage = ((mask_local > 0).sum() / mask_local.size) * 100
                            self._log(f"   🎭 DEBUG: Saved mask to {mask_path}", "info")
                            self._log(f"   📊 Mask coverage: {mask_percentage:.1f}% of image", "info")
                            
                            # Save mask overlay visualization
                            mask_viz_local = image.copy()
                            mask_viz_local[mask_local > 0] = [0, 0, 255]
                            viz_path = os.path.join(debug_dir, f"{base_name}_mask_overlay.png")
                            cv2.imwrite(viz_path, mask_viz_local)
                            self._log(f"   🎭 DEBUG: Saved mask overlay to {viz_path}", "info")
                        except Exception as e:
                            self._log(f"   ❌ Failed to save mask debug: {str(e)}", "error")
                            
                        # Also save intermediate copies
                        try:
                            self._save_intermediate_image(image_path, mask_local, "mask", debug_base_dir=output_dir)
                        except Exception:
                            pass
                    
                    self._log(f"🎨 Inpainting to remove original text")
                    try:
                        self._block_if_over_cap("inpainting")
                    except Exception:
                        pass
                    inpainted_local = self.inpaint_regions(image, mask_local)
                    
                    if self.manga_settings.get('advanced', {}).get('save_intermediate', False):
                        try:
                            self._save_intermediate_image(image_path, inpainted_local, "inpainted", debug_base_dir=output_dir)
                        except Exception:
                            pass
                    return inpainted_local
                except Exception as ie:
                    self._log(f"❌ Inpainting task error: {ie}", "error")
                    return image.copy()
            
            # Gate on advanced setting (default enabled)
            adv = self.manga_settings.get('advanced', {})
            run_concurrent = adv.get('concurrent_inpaint_translate', True)
            
            if run_concurrent:
                self._log("🔀 Running translation and inpainting concurrently", "info")
                with ThreadPoolExecutor(max_workers=2) as _executor:
                    fut_translate = _executor.submit(_task_translate)
                    fut_inpaint = _executor.submit(_task_inpaint)
                    # Wait for completion
                    try:
                        translate_ok = fut_translate.result()
                    except Exception:
                        translate_ok = False
                    try:
                        inpainted = fut_inpaint.result()
                    except Exception:
                        inpainted = image.copy()
            else:
                self._log("↪️ Concurrent mode disabled — running sequentially", "info")
                translate_ok = _task_translate()
                inpainted = _task_inpaint()
            
            # After concurrent phase, validate translation
            if self._check_stop():
                result['interrupted'] = True
                self._log("⏹️ Translation cancelled before rendering", "warning")
                result['regions'] = [r.to_dict() for r in regions]
                return result
            
            if not any(getattr(region, 'translated_text', None) for region in regions):
                result['interrupted'] = True
                self._log("⏹️ No regions were translated - translation was interrupted", "warning")
                result['regions'] = [r.to_dict() for r in regions]
                return result
            
            # Render translated text
            self._log(f"✍️ Rendering translated text...")
            self._log(f"   Using enhanced renderer with custom settings", "info")
            final_image = self.render_translated_text(inpainted, regions)
            
            # Save output
            try:
                if not output_path:
                    base, ext = os.path.splitext(image_path)
                    output_path = f"{base}_translated{ext}"
                
                success = cv2.imwrite(output_path, final_image)
                
                if not success:
                    self._log(f"   Using PIL to save with Unicode path...", "info")
                    from PIL import Image as PILImage
                    
                    rgb_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
                    pil_image = PILImage.fromarray(rgb_image)
                    pil_image.save(output_path)
                    self._log(f"   ✅ Successfully saved with PIL", "info")
                
                result['output_path'] = output_path
                self._log(f"\n💾 Saved output to: {output_path}")
                
            except Exception as e:
                error_msg = f"Failed to save output image: {str(e)}"
                self._log(f"❌ {error_msg}", "error")
                result['errors'].append(error_msg)
                result['success'] = False
                return result
            
            # Update result
            result['regions'] = [r.to_dict() for r in regions]
            if not result.get('interrupted', False):
                result['success'] = True
                self._log(f"\n✅ TRANSLATION PIPELINE COMPLETE", "success")
            else:
                self._log(f"\n⚠️ TRANSLATION INTERRUPTED - Partial output saved", "warning")
            
            self._log(f"{'='*60}\n")
            
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}\n{traceback.format_exc()}"
            self._log(f"\n❌ PIPELINE ERROR:", "error")
            self._log(f"   {str(e)}", "error")
            self._log(f"   Type: {type(e).__name__}", "error")
            self._log(traceback.format_exc(), "error")
            result['errors'].append(error_msg)
        finally:
            # Per-image memory cleanup to reduce RAM growth across pages
            try:
                # Clear self-held large attributes
                try:
                    self.current_image = None
                    self.current_mask = None
                    self.final_image = None
                    self.text_regions = []
                    self.translated_regions = []
                except Exception:
                    pass

                # Clear local large objects if present
                locs = locals()
                for name in [
                    'image', 'inpainted', 'final_image', 'mask', 'mask_viz', 'pil_image', 'heatmap'
                ]:
                    try:
                        if name in locs:
                            # Explicitly delete reference from locals
                            del locs[name]
                    except Exception:
                        pass

                # Reset caches for the next image (non-destructive to loaded models)
                try:
                    self.reset_for_new_image()
                except Exception:
                    pass

                # Encourage release of native resources
                try:
                    import cv2 as _cv2
                    try:
                        _cv2.destroyAllWindows()
                    except Exception:
                        pass
                except Exception:
                    pass

                # Free CUDA memory if torch is available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                # Release thread-local heavy objects to curb RAM growth across runs
                try:
                    self._cleanup_thread_locals()
                except Exception:
                    pass

                # Deep cleanup control - respects user settings and parallel processing
                try:
                    # Check if auto cleanup is enabled in settings
                    auto_cleanup_enabled = False  # Default disabled by default
                    try:
                        if hasattr(self, 'manga_settings'):
                            auto_cleanup_enabled = self.manga_settings.get('advanced', {}).get('auto_cleanup_models', False)
                    except Exception:
                        pass
                    
                    if not auto_cleanup_enabled:
                        # User has disabled automatic cleanup
                        self._log("🔑 Auto cleanup disabled - models will remain in RAM", "debug")
                    else:
                        # Determine if we should cleanup now
                        should_cleanup_now = True
                        
                        # Check if we're in batch mode
                        is_last_in_batch = False
                        try:
                            if getattr(self, 'batch_mode', False):
                                bc = getattr(self, 'batch_current', None)
                                bt = getattr(self, 'batch_size', None)
                                if bc is not None and bt is not None:
                                    is_last_in_batch = (bc >= bt)
                                    # In batch mode, only cleanup at the end
                                    should_cleanup_now = is_last_in_batch
                        except Exception:
                            pass
                        
                        # For parallel panel translation, cleanup is handled differently
                        # (it's handled in manga_integration.py after all panels complete)
                        is_parallel_panel = False
                        try:
                            if hasattr(self, 'manga_settings'):
                                is_parallel_panel = self.manga_settings.get('advanced', {}).get('parallel_panel_translation', False)
                        except Exception:
                            pass
                        
                        if is_parallel_panel:
                            # Don't cleanup here - let manga_integration handle it after all panels
                            self._log("🎯 Deferring cleanup until all parallel panels complete", "debug")
                            should_cleanup_now = False
                        
                        if should_cleanup_now:
                            # Perform the cleanup
                            self._deep_cleanup_models()
                            
                            # Also clear HF cache for RT-DETR (best-effort)
                            if is_last_in_batch or not getattr(self, 'batch_mode', False):
                                try:
                                    self._clear_hf_cache()
                                except Exception:
                                    pass
                except Exception:
                    pass

                # Force a garbage collection cycle
                try:
                    import gc
                    gc.collect()
                except Exception:
                    pass

                # Aggressively trim process working set (Windows) or libc heap (Linux)
                try:
                    self._trim_working_set()
                except Exception:
                    pass
            except Exception:
                # Never let cleanup fail the pipeline
                pass
        
        return result

    def reset_history_manager(self):
        """Reset history manager for new translation batch"""
        self.history_manager = None
        self.history_manager_initialized = False
        self.history_output_dir = None
        self.translation_context = []
        self._log("📚 Reset history manager for new batch", "debug")
    
    def cleanup_all_models(self):
        """Public method to force cleanup of all models - call this after translation!
        This ensures all models (YOLO, RT-DETR, inpainters, OCR) are unloaded from RAM.
        """
        self._log("🧹 Forcing cleanup of all models to free RAM...", "info")
        
        # Call the comprehensive cleanup
        self._deep_cleanup_models()
        
        # Also cleanup thread locals
        try:
            self._cleanup_thread_locals()
        except Exception:
            pass
        
        # Clear HF cache
        try:
            self._clear_hf_cache()
        except Exception:
            pass
        
        # Trim working set
        try:
            self._trim_working_set()
        except Exception:
            pass
        
        self._log("✅ All models cleaned up - RAM freed!", "info")
    
    def clear_internal_state(self):
        """Clear all internal state and cached data to free memory.
        This is called when the translator instance is being reset.
        Ensures OCR manager, inpainters, and bubble detector are also cleaned.
        """
        try:
            # Clear image data
            self.current_image = None
            self.current_mask = None
            self.final_image = None
            
            # Clear text regions
            if hasattr(self, 'text_regions'):
                self.text_regions = []
            if hasattr(self, 'translated_regions'):
                self.translated_regions = []
            
            # Clear ALL caches (including text caches)
            # THREAD-SAFE: Use lock for parallel panel translation
            if hasattr(self, 'cache'):
                self.cache.clear()
            if hasattr(self, 'ocr_roi_cache'):
                with self._cache_lock:
                    self.ocr_roi_cache.clear()
            self._current_image_hash = None
            
            # Clear history and context
            if hasattr(self, 'translation_context'):
                self.translation_context = []
            if hasattr(self, 'history_manager'):
                self.history_manager = None
            self.history_manager_initialized = False
            self.history_output_dir = None
            
            # IMPORTANT: Properly unload OCR manager
            if hasattr(self, 'ocr_manager') and self.ocr_manager:
                try:
                    ocr = self.ocr_manager
                    if hasattr(ocr, 'providers'):
                        for provider_name, provider in ocr.providers.items():
                            # Clear all model references
                            if hasattr(provider, 'model'):
                                provider.model = None
                            if hasattr(provider, 'processor'):
                                provider.processor = None
                            if hasattr(provider, 'tokenizer'):
                                provider.tokenizer = None
                            if hasattr(provider, 'reader'):
                                provider.reader = None
                            if hasattr(provider, 'client'):
                                provider.client = None
                            if hasattr(provider, 'is_loaded'):
                                provider.is_loaded = False
                        ocr.providers.clear()
                    self.ocr_manager = None
                    self._log("   ✓ OCR manager cleared", "debug")
                except Exception as e:
                    self._log(f"   Warning: OCR cleanup failed: {e}", "debug")
            
            # IMPORTANT: Handle local inpainter cleanup carefully
            # DO NOT unload if it's a shared/checked-out instance from the pool
            if hasattr(self, 'local_inpainter') and self.local_inpainter:
                try:
                    # Only unload if this is NOT a checked-out or shared instance
                    is_from_pool = hasattr(self, '_checked_out_inpainter') or hasattr(self, '_inpainter_pool_key')
                    if not is_from_pool and hasattr(self.local_inpainter, 'unload'):
                        self.local_inpainter.unload()
                        self._log("   ✓ Local inpainter unloaded", "debug")
                    else:
                        self._log("   ✓ Local inpainter reference cleared (pool instance preserved)", "debug")
                    self.local_inpainter = None
                except Exception as e:
                    self._log(f"   Warning: Inpainter cleanup failed: {e}", "debug")
            
            # Also clear hybrid and generic inpainter references
            if hasattr(self, 'hybrid_inpainter'):
                if self.hybrid_inpainter and hasattr(self.hybrid_inpainter, 'unload'):
                    try:
                        self.hybrid_inpainter.unload()
                    except Exception:
                        pass
                self.hybrid_inpainter = None
            
            if hasattr(self, 'inpainter'):
                if self.inpainter and hasattr(self.inpainter, 'unload'):
                    try:
                        self.inpainter.unload()
                    except Exception:
                        pass
                self.inpainter = None
            
            # IMPORTANT: Handle bubble detector cleanup carefully
            # DO NOT unload if it's a singleton or from a preloaded pool
            if hasattr(self, 'bubble_detector') and self.bubble_detector:
                try:
                    is_singleton = getattr(self, 'use_singleton_bubble_detector', False)
                    # Check if it's from thread-local which might have gotten it from the pool
                    is_from_pool = hasattr(self, '_thread_local') and hasattr(self._thread_local, 'bubble_detector')
                    
                    if not is_singleton and not is_from_pool:
                        if hasattr(self.bubble_detector, 'unload'):
                            self.bubble_detector.unload(release_shared=True)
                        self._log("   ✓ Bubble detector unloaded", "debug")
                    else:
                        self._log("   ✓ Bubble detector reference cleared (pool/singleton instance preserved)", "debug")
                    # In all cases, clear our instance reference
                    self.bubble_detector = None
                except Exception as e:
                    self._log(f"   Warning: Bubble detector cleanup failed: {e}", "debug")
            
            # Clear any file handles or temp data
            if hasattr(self, '_thread_local'):
                try:
                    self._cleanup_thread_locals()
                except Exception:
                    pass
            
            # Clear processing flags
            self.is_processing = False
            self.cancel_requested = False
            
            self._log("🧹 Internal state and all components cleared", "debug")
            
        except Exception as e:
            self._log(f"⚠️ Warning: Failed to clear internal state: {e}", "warning")
    
    def _process_webtoon_chunks(self, image_path: str, output_path: str, result: Dict) -> Dict:
        """Process webtoon in chunks for better OCR"""
        import cv2
        import numpy as np
        from PIL import Image as PILImage
        
        try:
            self._log("📱 Processing webtoon in chunks for better OCR", "info")
            
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                pil_image = PILImage.open(image_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            height, width = image.shape[:2]
            
            # Get chunk settings from config
            chunk_height = self.manga_settings.get('preprocessing', {}).get('chunk_height', 1000)
            chunk_overlap = self.manga_settings.get('preprocessing', {}).get('chunk_overlap', 100)
            
            self._log(f"   Image dimensions: {width}x{height}", "info")
            self._log(f"   Chunk height: {chunk_height}px, Overlap: {chunk_overlap}px", "info")
            
            # Calculate number of chunks needed
            effective_chunk_height = chunk_height - chunk_overlap
            num_chunks = max(1, (height - chunk_overlap) // effective_chunk_height + 1)
            
            self._log(f"   Will process in {num_chunks} chunks", "info")
            
            # Process each chunk
            all_regions = []
            chunk_offsets = []
            
            for i in range(num_chunks):
                # Calculate chunk boundaries
                start_y = i * effective_chunk_height
                end_y = min(start_y + chunk_height, height)
                
                # Make sure we don't miss the bottom part
                if i == num_chunks - 1:
                    end_y = height
                
                self._log(f"\n   📄 Processing chunk {i+1}/{num_chunks} (y: {start_y}-{end_y})", "info")
                
                # Extract chunk
                chunk = image[start_y:end_y, 0:width]
                
                # Save chunk temporarily for OCR
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    chunk_path = tmp.name
                    cv2.imwrite(chunk_path, chunk)
                
                try:
                    # Detect text in this chunk
                    chunk_regions = self.detect_text_regions(chunk_path)
                    
                    # Adjust region coordinates to full image space
                    for region in chunk_regions:
                        # Adjust bounding box
                        x, y, w, h = region.bounding_box
                        region.bounding_box = (x, y + start_y, w, h)
                        
                        # Adjust vertices if present
                        if hasattr(region, 'vertices') and region.vertices:
                            adjusted_vertices = []
                            for vx, vy in region.vertices:
                                adjusted_vertices.append((vx, vy + start_y))
                            region.vertices = adjusted_vertices
                        
                        # Mark which chunk this came from (for deduplication)
                        region.chunk_index = i
                        region.chunk_y_range = (start_y, end_y)
                    
                    all_regions.extend(chunk_regions)
                    chunk_offsets.append(start_y)
                    
                    self._log(f"   Found {len(chunk_regions)} text regions in chunk {i+1}", "info")
                    
                finally:
                    # Clean up temp file
                    import os
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
            
            # Remove duplicate regions from overlapping areas
            self._log(f"\n   🔍 Deduplicating regions from overlaps...", "info")
            unique_regions = self._deduplicate_chunk_regions(all_regions, chunk_overlap)
            
            self._log(f"   Total regions: {len(all_regions)} → {len(unique_regions)} after deduplication", "info")
            
            if not unique_regions:
                self._log("⚠️ No text regions detected in webtoon", "warning")
                result['errors'].append("No text regions detected")
                return result
            
            # Now process the regions as normal
            self._log(f"\n📍 Translating {len(unique_regions)} unique regions", "info")
            
            # Translate regions
            if self.full_page_context_enabled:
                translations = self.translate_full_page_context(unique_regions, image_path)
                for region in unique_regions:
                    if region.text in translations:
                        region.translated_text = translations[region.text]
            else:
                unique_regions = self.translate_regions(unique_regions, image_path)
            
            # Create mask and inpaint
            self._log(f"\n🎨 Creating mask and inpainting...", "info")
            mask = self.create_text_mask(image, unique_regions)
            
            if self.skip_inpainting:
                inpainted = image.copy()
            else:
                inpainted = self.inpaint_regions(image, mask)
            
            # Render translated text
            self._log(f"✍️ Rendering translated text...", "info")
            final_image = self.render_translated_text(inpainted, unique_regions)
            
            # Save output
            if not output_path:
                base, ext = os.path.splitext(image_path)
                output_path = f"{base}_translated{ext}"
            
            cv2.imwrite(output_path, final_image)
            
            result['output_path'] = output_path
            result['regions'] = [r.to_dict() for r in unique_regions]
            result['success'] = True
            result['format_info']['chunks_processed'] = num_chunks
            
            self._log(f"\n✅ Webtoon processing complete: {output_path}", "success")
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing webtoon chunks: {str(e)}"
            self._log(f"❌ {error_msg}", "error")
            result['errors'].append(error_msg)
            return result

    def _deduplicate_chunk_regions(self, regions: List, overlap_height: int) -> List:
        """Remove duplicate regions from overlapping chunk areas"""
        if not regions:
            return regions
        
        # Sort regions by y position
        regions.sort(key=lambda r: r.bounding_box[1])
        
        unique_regions = []
        used_indices = set()
        
        for i, region1 in enumerate(regions):
            if i in used_indices:
                continue
            
            # Check if this region is in an overlap zone
            x1, y1, w1, h1 = region1.bounding_box
            chunk_idx = region1.chunk_index if hasattr(region1, 'chunk_index') else 0
            chunk_y_start, chunk_y_end = region1.chunk_y_range if hasattr(region1, 'chunk_y_range') else (0, float('inf'))
            
            # Check if region is near chunk boundary (in overlap zone)
            in_overlap_zone = (y1 < chunk_y_start + overlap_height) and chunk_idx > 0
            
            if in_overlap_zone:
                # Look for duplicate in previous chunk's regions
                found_duplicate = False
                
                for j, region2 in enumerate(regions):
                    if j >= i or j in used_indices:
                        continue
                    
                    if hasattr(region2, 'chunk_index') and region2.chunk_index == chunk_idx - 1:
                        x2, y2, w2, h2 = region2.bounding_box
                        
                        # Check if regions are the same (similar position and size)
                        if (abs(x1 - x2) < 20 and 
                            abs(y1 - y2) < 20 and 
                            abs(w1 - w2) < 20 and 
                            abs(h1 - h2) < 20):
                            
                            # Check text similarity
                            if region1.text == region2.text:
                                # This is a duplicate
                                found_duplicate = True
                                used_indices.add(i)
                                self._log(f"   Removed duplicate: '{region1.text[:30]}...'", "debug")
                                break
                
                if not found_duplicate:
                    unique_regions.append(region1)
                    used_indices.add(i)
            else:
                # Not in overlap zone, keep it
                unique_regions.append(region1)
                used_indices.add(i)
        
        return unique_regions

    def _save_intermediate_image(self, original_path: str, image, stage: str, debug_base_dir: str = None):
        """Save intermediate processing stages under translated_images/debug or provided base dir"""
        if debug_base_dir is None:
            translated_dir = os.path.join(os.path.dirname(original_path), 'translated_images')
            debug_dir = os.path.join(translated_dir, 'debug')
        else:
            debug_dir = os.path.join(debug_base_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        output_path = os.path.join(debug_dir, f"{base_name}_{stage}.png")
        
        cv2.imwrite(output_path, image)
        self._log(f"   💾 Saved {stage} image: {output_path}")
