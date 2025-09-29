# manga_integration.py
"""
Enhanced GUI Integration module for Manga Translation with text visibility controls
Integrates with TranslatorGUI using WindowManager and existing infrastructure
Now includes full page context mode with customizable prompt
"""
import sys
import os
import json
import threading
import time
import hashlib
import traceback
import concurrent.futures
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk
import ttkbootstrap as tb
from typing import List, Dict, Optional, Any
from queue import Queue
import logging
import sys
import threading
from manga_translator import MangaTranslator, GOOGLE_CLOUD_VISION_AVAILABLE
from manga_settings_dialog import MangaSettingsDialog


# Try to import UnifiedClient for API initialization
try:
    from unified_api_client import UnifiedClient
except ImportError:
    UnifiedClient = None

class _MangaGuiLogHandler(logging.Handler):
    """Forward logging records into MangaTranslationTab._log."""
    def __init__(self, gui_ref, level=logging.INFO):
        super().__init__(level)
        self.gui_ref = gui_ref
        self._last_msg = None
        self.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))

    def emit(self, record: logging.LogRecord) -> None:
        # Avoid looping/duplicates from this module's own messages or when stdio is redirected
        try:
            if getattr(self.gui_ref, '_stdio_redirect_active', False):
                return
            if record and isinstance(record.name, str) and record.name.startswith(('manga_integration',)):
                return
        except Exception:
            pass
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        # Deduplicate identical consecutive messages
        if msg == self._last_msg:
            return
        self._last_msg = msg
        try:
            if hasattr(self.gui_ref, '_log'):
                # Map logging levels to our tag levels
                lvl = record.levelname.lower()
                tag = 'info'
                if lvl.startswith('warn'):
                    tag = 'warning'
                elif lvl.startswith('err') or lvl.startswith('crit'):
                    tag = 'error'
                elif lvl.startswith('debug'):
                    tag = 'debug'
                elif lvl.startswith('info'):
                    tag = 'info'
                self.gui_ref._log(msg, tag)
        except Exception:
            pass

class _StreamToGuiLog:
    """A minimal file-like stream that forwards lines to _log."""
    def __init__(self, write_cb):
        self._write_cb = write_cb
        self._buf = ''

    def write(self, s: str):
        try:
            self._buf += s
            while '\n' in self._buf:
                line, self._buf = self._buf.split('\n', 1)
                if line.strip():
                    self._write_cb(line)
        except Exception:
            pass

    def flush(self):
        try:
            if self._buf.strip():
                self._write_cb(self._buf)
            self._buf = ''
        except Exception:
            pass

class MangaTranslationTab:
    """GUI interface for manga translation integrated with TranslatorGUI"""
    
    # Class-level cancellation flag for all instances
    _global_cancelled = False
    _global_cancel_lock = threading.RLock()
    
    @classmethod
    def set_global_cancellation(cls, cancelled: bool):
        """Set global cancellation flag for all translation instances"""
        with cls._global_cancel_lock:
            cls._global_cancelled = cancelled
    
    @classmethod
    def is_globally_cancelled(cls) -> bool:
        """Check if globally cancelled"""
        with cls._global_cancel_lock:
            return cls._global_cancelled
    
    def __init__(self, parent_frame: tk.Frame, main_gui, dialog, canvas):
        """Initialize manga translation interface
        
        Args:
            parent_frame: The scrollable frame from WindowManager
            main_gui: Reference to TranslatorGUI instance
            dialog: The dialog window
            canvas: The canvas for scrolling
        """
        self.parent_frame = parent_frame
        self.main_gui = main_gui
        self.dialog = dialog
        self.canvas = canvas
        
        # Translation state
        self.translator = None
        self.is_running = False
        self.stop_flag = threading.Event()
        self.translation_thread = None
        self.translation_future = None
        # Shared executor from main GUI if available
        try:
            if hasattr(self.main_gui, 'executor') and self.main_gui.executor:
                self.executor = self.main_gui.executor
            else:
                self.executor = None
        except Exception:
            self.executor = None
        self.selected_files = []
        self.current_file_index = 0
        self.font_mapping = {}  # Initialize font mapping dictionary

        
        # Progress tracking
        self.total_files = 0
        self.completed_files = 0
        self.failed_files = 0
        self.qwen2vl_model_size = self.main_gui.config.get('qwen2vl_model_size', '1')
        
        # Advanced performance toggles
        try:
            adv_cfg = self.main_gui.config.get('manga_settings', {}).get('advanced', {})
        except Exception:
            adv_cfg = {}
        # In singleton mode, reduce OpenCV thread usage to avoid CPU spikes
        try:
            if bool(adv_cfg.get('use_singleton_models', True)):
                import cv2 as _cv2
                try:
                    _cv2.setNumThreads(1)
                except Exception:
                    pass
        except Exception:
            pass
        # Do NOT preload big local models by default to avoid startup crashes
        self.preload_local_models_on_open = bool(adv_cfg.get('preload_local_models_on_open', False))
        
        # Queue for thread-safe GUI updates
        self.update_queue = Queue()
        
        # Flags for stdio redirection to avoid duplicate GUI logs
        self._stdout_redirect_on = False
        self._stderr_redirect_on = False
        self._stdio_redirect_active = False
        
        # Flag to prevent saving during initialization
        self._initializing = True
        
        # IMPORTANT: Load settings BEFORE building interface
        # This ensures all variables are initialized before they're used in the GUI
        self._load_rendering_settings()
        
        # Initialize the full page context prompt
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

        # Initialize the OCR system prompt
        self.ocr_prompt = self.main_gui.config.get('manga_ocr_prompt', 
            "YOU ARE AN OCR SYSTEM. YOUR ONLY JOB IS TEXT EXTRACTION.\n\n"
            "CRITICAL RULES:\n"
            "1. DO NOT TRANSLATE ANYTHING\n"
            "2. DO NOT MODIFY THE TEXT\n"
            "3. DO NOT EXPLAIN OR COMMENT\n"
            "4. ONLY OUTPUT THE EXACT TEXT YOU SEE\n"
            "5. PRESERVE NATURAL TEXT FLOW - DO NOT ADD UNNECESSARY LINE BREAKS\n\n"
            "If you see Korean text, output it in Korean.\n"
            "If you see Japanese text, output it in Japanese.\n"
            "If you see Chinese text, output it in Chinese.\n"
            "If you see English text, output it in English.\n\n"
            "IMPORTANT: Only use line breaks where they naturally occur in the original text "
            "(e.g., between dialogue lines or paragraphs). Do not break text mid-sentence or "
            "between every word/character.\n\n"
            "For vertical text common in manga/comics, transcribe it as a continuous line unless "
            "there are clear visual breaks.\n\n"
            "NEVER translate. ONLY extract exactly what is written.\n"
            "Output ONLY the raw text, nothing else."
        )
        
        # flag to skip status checks during init
        self._initializing_gui = True
        
        # Build interface AFTER loading settings
        self._build_interface()

        # Now allow status checks
        self._initializing_gui = False
        
        # Do one status check after everything is built
        self.dialog.after(100, self._check_provider_status)
        
        # Now that everything is initialized, allow saving
        self._initializing = False
        
        # Attach logging bridge so library logs appear in our log area
        self._attach_logging_bridge()

        # Start update loop
        self._process_updates()
    
    def _is_stop_requested(self) -> bool:
        """Check if stop has been requested using multiple sources"""
        # Check global cancellation first
        if self.is_globally_cancelled():
            return True
            
        # Check local stop flag
        if hasattr(self, 'stop_flag') and self.stop_flag.is_set():
            return True
            
        # Check running state
        if hasattr(self, 'is_running') and not self.is_running:
            return True
            
        return False
    
    def _reset_global_cancellation(self):
        """Reset all global cancellation flags for new translation"""
        # Reset local class flag
        self.set_global_cancellation(False)
        
        # Reset MangaTranslator class flag
        try:
            from manga_translator import MangaTranslator
            MangaTranslator.set_global_cancellation(False)
        except ImportError:
            pass
            
        # Reset UnifiedClient flag
        try:
            from unified_api_client import UnifiedClient
            UnifiedClient.set_global_cancellation(False)
        except ImportError:
            pass
    
    def reset_stop_flags(self):
        """Reset all stop flags when starting new translation"""
        self.is_running = False
        if hasattr(self, 'stop_flag'):
            self.stop_flag.clear()
        self._reset_global_cancellation()
        self._log("🔄 Stop flags reset for new translation", "debug")
    
    def _distribute_stop_flags(self):
        """Distribute stop flags to all manga translation components"""
        if not hasattr(self, 'translator') or not self.translator:
            return
        
        # Set stop flag on translator
        if hasattr(self.translator, 'set_stop_flag'):
            self.translator.set_stop_flag(self.stop_flag)
        
        # Set stop flag on OCR manager and all providers
        if hasattr(self.translator, 'ocr_manager') and self.translator.ocr_manager:
            if hasattr(self.translator.ocr_manager, 'set_stop_flag'):
                self.translator.ocr_manager.set_stop_flag(self.stop_flag)
        
        # Set stop flag on bubble detector if available
        if hasattr(self.translator, 'bubble_detector') and self.translator.bubble_detector:
            if hasattr(self.translator.bubble_detector, 'set_stop_flag'):
                self.translator.bubble_detector.set_stop_flag(self.stop_flag)
                
        # Set stop flag on local inpainter if available
        if hasattr(self.translator, 'local_inpainter') and self.translator.local_inpainter:
            if hasattr(self.translator.local_inpainter, 'set_stop_flag'):
                self.translator.local_inpainter.set_stop_flag(self.stop_flag)
                
        # Also try to set on thread-local components if accessible
        if hasattr(self.translator, '_thread_local'):
            thread_local = self.translator._thread_local
            # Set on thread-local bubble detector
            if hasattr(thread_local, 'bubble_detector') and thread_local.bubble_detector:
                if hasattr(thread_local.bubble_detector, 'set_stop_flag'):
                    thread_local.bubble_detector.set_stop_flag(self.stop_flag)
            
            # Set on thread-local inpainters
            if hasattr(thread_local, 'local_inpainters') and isinstance(thread_local.local_inpainters, dict):
                for inpainter in thread_local.local_inpainters.values():
                    if hasattr(inpainter, 'set_stop_flag'):
                        inpainter.set_stop_flag(self.stop_flag)
        
        self._log("🔄 Stop flags distributed to all components", "debug")
    
    def _preflight_bubble_detector(self, ocr_settings: dict) -> bool:
        """Try to ensure the bubble detector is ready before starting translation.
        Returns True if a detector is ready (loaded), False otherwise.
        This mitigates aggressive cleanup by retrying RT-DETR load briefly.
        """
        try:
            # Only act when bubble detection is enabled
            if not ocr_settings.get('bubble_detection_enabled', False):
                return False
            detector_type = ocr_settings.get('detector_type', 'rtdetr')

            # Use translator helper if available
            for attempt in range(3):
                try:
                    bd = None
                    if hasattr(self, 'translator') and self.translator and hasattr(self.translator, '_ensure_bubble_detector_ready'):
                        bd = self.translator._ensure_bubble_detector_ready(ocr_settings)
                        if bd is not None:
                            self.translator.bubble_detector = bd
                            if getattr(bd, 'rtdetr_loaded', False) or getattr(bd, 'model_loaded', False):
                                if attempt > 0:
                                    self._log("✅ Bubble detector ready after retry", "info")
                                else:
                                    self._log("🤖 Bubble detector ready", "debug")
                                return True
                    # Fallback: try loading explicitly
                    try:
                        from bubble_detector import BubbleDetector
                    except Exception:
                        return False
                    bd = BubbleDetector()
                    if detector_type == 'rtdetr':
                        model_id = ocr_settings.get('rtdetr_model_url') or ocr_settings.get('bubble_model_path') or 'ogkalu/comic-text-and-bubble-detector'
                        ok = bd.load_rtdetr_model(model_id=model_id)
                    else:
                        model_path = ocr_settings.get('bubble_model_path')
                        ok = bd.load_model(model_path) if model_path else False
                    if ok:
                        if hasattr(self, 'translator') and self.translator:
                            self.translator.bubble_detector = bd
                        self._log(f"🤖 Bubble detector preloaded ({'RT-DETR' if detector_type=='rtdetr' else 'YOLO'})", "info")
                        return True
                except Exception as e:
                    self._log(f"⚠️ Bubble detector preflight attempt {attempt+1} failed: {e}", "debug")
                # Brief wait before retry
                try:
                    time.sleep(0.4)
                except Exception:
                    pass
            # After retries
            self._log("⚠️ Bubble detector preflight failed; will fall back to proximity merge if needed", "warning")
            return False
        except Exception:
            return False
    
    def _disable_spinbox_mousewheel(self, spinbox):
        """Disable mousewheel scrolling on a spinbox"""
        spinbox.bind("<MouseWheel>", lambda e: "break")
        spinbox.bind("<Button-4>", lambda e: "break")  # Linux scroll up
        spinbox.bind("<Button-5>", lambda e: "break")  # Linux scroll down

    def _download_hf_model(self):
        """Download HuggingFace models with progress tracking"""
        provider = self.ocr_provider_var.get()
        
        # Model sizes (approximate in MB)
        model_sizes = {
            'manga-ocr': 450,
            'Qwen2-VL': {
                '2B': 4000,
                '7B': 14000,
                '72B': 144000,
                'custom': 10000  # Default estimate for custom models
            }
        }
        
        # For Qwen2-VL, show model selection dialog first
        if provider == 'Qwen2-VL':
            # Use window manager from main_gui
            selection_dialog, scrollable_frame, canvas = self.main_gui.wm.setup_scrollable(
                self.dialog,
                "Select Qwen2-VL Model Size",
                width=None,
                height=None,
                max_width_ratio=0.6,
                max_height_ratio=0.3
            )
            
            # Title
            title_frame = tk.Frame(scrollable_frame)
            title_frame.pack(fill=tk.X, pady=(10, 20))
            tk.Label(title_frame, text="Select Qwen2-VL Model Size", 
                    font=('Arial', 14, 'bold')).pack()
            
            # Model selection frame
            model_frame = tk.LabelFrame(
                scrollable_frame,
                text="Model Options",
                font=('Arial', 11, 'bold'),
                padx=15,
                pady=10
            )
            model_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            model_options = {
                "2B": {
                    "title": "2B Model",
                    "desc": "• Smallest model (~4GB download, 4-8GB VRAM)\n• Fast but less accurate\n• Good for quick testing"
                },
                "7B": {
                    "title": "7B Model", 
                    "desc": "• Medium model (~14GB download, 12-16GB VRAM)\n• Best balance of speed and quality\n• Recommended for most users"
                },
                "72B": {
                    "title": "72B Model",
                    "desc": "• Largest model (~144GB download, 80GB+ VRAM)\n• Highest quality but very slow\n• Requires high-end GPU"
                },
                "custom": {
                    "title": "Custom Model",
                    "desc": "• Enter any Hugging Face model ID\n• For advanced users\n• Size varies by model"
                }
            }
            
            selected_model = tk.StringVar(value="2B")
            custom_model_id = tk.StringVar()
            
            for key, info in model_options.items():
                option_frame = tk.Frame(model_frame)
                option_frame.pack(fill=tk.X, pady=5)
                
                rb = tk.Radiobutton(option_frame, text=info["title"], 
                                   variable=selected_model, value=key, 
                                   font=('Arial', 11, 'bold'))
                rb.pack(anchor='w')
                
                desc_label = tk.Label(option_frame, text=info["desc"], 
                                     font=('Arial', 9), justify=tk.LEFT, fg='#666666')
                desc_label.pack(anchor='w', padx=(20, 0))
                
                if key != "custom":
                    ttk.Separator(option_frame, orient='horizontal').pack(fill=tk.X, pady=(5, 0))
            
            # Custom model ID frame
            custom_frame = tk.LabelFrame(
                scrollable_frame,
                text="Custom Model ID",
                font=('Arial', 11, 'bold'),
                padx=15,
                pady=10
            )
            
            entry_frame = tk.Frame(custom_frame)
            entry_frame.pack(fill=tk.X, pady=5)
            tk.Label(entry_frame, text="Model ID:", font=('Arial', 10)).pack(side=tk.LEFT, padx=(0, 10))
            custom_entry = tk.Entry(entry_frame, textvariable=custom_model_id, width=40, font=('Arial', 10))
            custom_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            def toggle_custom_frame(*args):
                if selected_model.get() == "custom":
                    custom_frame.pack(fill=tk.X, padx=20, pady=10, after=model_frame)
                else:
                    custom_frame.pack_forget()
            
            selected_model.trace('w', toggle_custom_frame)
            
            # GPU status frame
            gpu_frame = tk.LabelFrame(
                scrollable_frame,
                text="System Status",
                font=('Arial', 11, 'bold'),
                padx=15,
                pady=10
            )
            gpu_frame.pack(fill=tk.X, padx=20, pady=10)
            
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                    gpu_text = f"✓ GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f}GB)"
                    gpu_color = '#4CAF50'
                else:
                    gpu_text = "✗ No GPU detected - will use CPU (very slow)"
                    gpu_color = '#f44336'
            except:
                gpu_text = "? GPU status unknown - install torch with CUDA"
                gpu_color = '#FF9800'
            
            tk.Label(gpu_frame, text=gpu_text, font=('Arial', 10), fg=gpu_color).pack(anchor='w')
            
            # Buttons
            button_frame = tk.Frame(scrollable_frame)
            button_frame.pack(fill=tk.X, pady=20)
            
            model_confirmed = {'value': False, 'model_key': None, 'model_id': None}
            
            def confirm_selection():
                selected = selected_model.get()
                if selected == "custom":
                    if not custom_model_id.get().strip():
                        messagebox.showerror("Error", "Please enter a model ID")
                        return
                    model_confirmed['model_key'] = selected
                    model_confirmed['model_id'] = custom_model_id.get().strip()
                else:
                    model_confirmed['model_key'] = selected
                    model_confirmed['model_id'] = f"Qwen/Qwen2-VL-{selected}-Instruct"
                model_confirmed['value'] = True
                selection_dialog.destroy()
            
            # Center the buttons by creating an inner frame
            button_inner_frame = tk.Frame(button_frame)
            button_inner_frame.pack()

            proceed_btn = tk.Button(
                button_inner_frame, text="Continue", command=confirm_selection,
                bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                padx=20, pady=8, cursor='hand2'
            )
            proceed_btn.pack(side=tk.LEFT, padx=5)

            cancel_btn = tk.Button(
                button_inner_frame, text="Cancel", command=selection_dialog.destroy,
                bg='#9E9E9E', fg='white', font=('Arial', 10),
                padx=20, pady=8, cursor='hand2'
            )
            cancel_btn.pack(side=tk.LEFT, padx=5)
            
            # Auto-resize and wait
            self.main_gui.wm.auto_resize_dialog(selection_dialog, canvas, max_width_ratio=0.5, max_height_ratio=0.6)
            self.dialog.wait_window(selection_dialog)
            
            if not model_confirmed['value']:
                return
            
            selected_model_key = model_confirmed['model_key']
            model_id = model_confirmed['model_id']
            total_size_mb = model_sizes['Qwen2-VL'][selected_model_key]
        elif provider == 'rapidocr':
            total_size_mb = 50  # Approximate size for display
            model_id = None
            selected_model_key = None
        else:
            total_size_mb = model_sizes.get(provider, 500)
            model_id = None
            selected_model_key = None
        
        # Create download dialog with window manager
        download_dialog, scrollable_frame, canvas = self.main_gui.wm.setup_scrollable(
            self.dialog,
            f"Download {provider} Model",
            width=600,
            height=450,
            max_width_ratio=0.6,
            max_height_ratio=0.6
        )
        
        # Info section
        info_frame = tk.LabelFrame(
            scrollable_frame,
            text="Model Information",
            font=('Arial', 11, 'bold'),
            padx=15,
            pady=10
        )
        info_frame.pack(fill=tk.X, padx=20, pady=10)
        
        if provider == 'Qwen2-VL':
            info_text = f"📚 Qwen2-VL {selected_model_key} Model\n"
            info_text += f"Model ID: {model_id}\n"
            info_text += f"Estimated size: ~{total_size_mb/1000:.1f}GB\n"
            info_text += "Vision-Language model for Korean OCR"
        else:
            info_text = f"📚 {provider} Model\nOptimized for manga/manhwa text detection"
        
        tk.Label(info_frame, text=info_text, font=('Arial', 10), justify=tk.LEFT).pack(anchor='w')
        
        # Progress section
        progress_frame = tk.LabelFrame(
            scrollable_frame,
            text="Download Progress",
            font=('Arial', 11, 'bold'),
            padx=15,
            pady=10
        )
        progress_frame.pack(fill=tk.X, padx=20, pady=10)
        
        progress_label = tk.Label(progress_frame, text="Ready to download", font=('Arial', 10))
        progress_label.pack(pady=(5, 10))
        
        progress_var = tk.DoubleVar()
        try:
            # Try to use our custom progress bar style
            progress_bar = ttk.Progressbar(progress_frame, length=550, mode='determinate', 
                                          variable=progress_var,
                                          style="MangaProgress.Horizontal.TProgressbar")
        except Exception:
            # Fallback to default if style not available yet
            progress_bar = ttk.Progressbar(progress_frame, length=550, mode='determinate', 
                                          variable=progress_var)
        progress_bar.pack(pady=(0, 5))
        
        size_label = tk.Label(progress_frame, text="", font=('Arial', 9), fg='#666666')
        size_label.pack()
        
        speed_label = tk.Label(progress_frame, text="", font=('Arial', 9), fg='#666666')
        speed_label.pack()
        
        status_label = tk.Label(progress_frame, text="Click 'Download' to begin", 
                              font=('Arial', 9), fg='#666666')
        status_label.pack(pady=(5, 0))
        
        # Log section
        log_frame = tk.LabelFrame(
            scrollable_frame,
            text="Download Log",
            font=('Arial', 11, 'bold'),
            padx=15,
            pady=10
        )
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Create a frame to hold the text widget and scrollbar
        text_frame = tk.Frame(log_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        details_text = tk.Text(
            text_frame, 
            height=12, 
            width=70, 
            font=('Courier', 9), 
            bg='#f5f5f5'
        )
        details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Attach scrollbar to the frame, not the text widget
        scrollbar = ttk.Scrollbar(text_frame, command=details_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        details_text.config(yscrollcommand=scrollbar.set)
            
        def add_log(message):
            """Add message to log"""
            details_text.insert(tk.END, f"{message}\n")
            details_text.see(tk.END)
            details_text.update()
        
        # Buttons frame
        button_frame = tk.Frame(download_dialog)
        button_frame.pack(pady=15)
        
        # Download tracking variables
        download_active = {'value': False}
        
        def get_dir_size(path):
            """Get total size of directory"""
            total = 0
            try:
                for dirpath, dirnames, filenames in os.walk(path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if os.path.exists(filepath):
                            total += os.path.getsize(filepath)
            except:
                pass
            return total
        
        def download_with_progress():
            """Download model with real progress tracking"""
            import time
            
            download_active['value'] = True
            total_size = total_size_mb * 1024 * 1024
            
            try:
                if provider == 'manga-ocr':
                    progress_label.config(text="Loading manga-ocr model...")
                    add_log("Initializing manga-ocr...")
                    progress_var.set(10)
                    
                    from manga_ocr import MangaOcr
                    
                    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                    initial_size = get_dir_size(cache_dir) if os.path.exists(cache_dir) else 0
                    
                    def init_model_with_progress():
                        start_time = time.time()
                        
                        import threading
                        model_ready = threading.Event()
                        model_instance = [None]
                        
                        def init_model():
                            model_instance[0] = MangaOcr()
                            model_ready.set()
                        
                        init_thread = threading.Thread(target=init_model)
                        init_thread.start()
                        
                        while not model_ready.is_set() and download_active['value']:
                            current_size = get_dir_size(cache_dir) if os.path.exists(cache_dir) else 0
                            downloaded = current_size - initial_size
                            
                            if downloaded > 0:
                                progress = min((downloaded / total_size) * 100, 99)
                                progress_var.set(progress)
                                
                                elapsed = time.time() - start_time
                                if elapsed > 0:
                                    speed = downloaded / elapsed
                                    speed_mb = speed / (1024 * 1024)
                                    speed_label.config(text=f"Speed: {speed_mb:.1f} MB/s")
                                
                                mb_downloaded = downloaded / (1024 * 1024)
                                mb_total = total_size / (1024 * 1024)
                                size_label.config(text=f"{mb_downloaded:.1f} MB / {mb_total:.1f} MB")
                                progress_label.config(text=f"Downloading: {progress:.1f}%")
                            
                            time.sleep(0.5)
                        
                        init_thread.join(timeout=1)
                        return model_instance[0]
                    
                    model = init_model_with_progress()
                    
                    if model:
                        progress_var.set(100)
                        size_label.config(text=f"{total_size_mb} MB / {total_size_mb} MB")
                        progress_label.config(text="✅ Download complete!")
                        status_label.config(text="Model ready to use!")
                        
                        self.ocr_manager.get_provider('manga-ocr').model = model
                        self.ocr_manager.get_provider('manga-ocr').is_loaded = True
                        self.ocr_manager.get_provider('manga-ocr').is_installed = True
                        
                        self.dialog.after(0, self._check_provider_status)
                        
                elif provider == 'Qwen2-VL':
                    try:
                        from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq
                        import torch
                    except ImportError as e:
                        progress_label.config(text="❌ Missing dependencies")
                        status_label.config(text="Install dependencies first")
                        add_log(f"ERROR: {str(e)}")
                        add_log("Please install manually:")
                        add_log("pip install transformers torch torchvision")
                        return
                    
                    progress_label.config(text=f"Downloading model...")
                    add_log(f"Starting download of {model_id}")
                    progress_var.set(10)
                    
                    add_log("Downloading processor...")
                    status_label.config(text="Downloading processor...")
                    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                    progress_var.set(30)
                    add_log("✓ Processor downloaded")
                    
                    add_log("Downloading tokenizer...")
                    status_label.config(text="Downloading tokenizer...")
                    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                    progress_var.set(50)
                    add_log("✓ Tokenizer downloaded")
                    
                    add_log("Downloading model weights (this may take several minutes)...")
                    status_label.config(text="Downloading model weights...")
                    progress_label.config(text="Downloading model weights...")
                    
                    if torch.cuda.is_available():
                        add_log(f"Using GPU: {torch.cuda.get_device_name(0)}")
                        model = AutoModelForVision2Seq.from_pretrained(
                            model_id,
                            dtype=torch.float16,
                            device_map="auto",
                            trust_remote_code=True
                        )
                    else:
                        add_log("No GPU detected, will load on CPU")
                        model = AutoModelForVision2Seq.from_pretrained(
                            model_id,
                            dtype=torch.float32,
                            trust_remote_code=True
                        )
                    
                    progress_var.set(90)
                    add_log("✓ Model weights downloaded")
                    
                    add_log("Initializing model...")
                    status_label.config(text="Initializing...")
                    
                    qwen_provider = self.ocr_manager.get_provider('Qwen2-VL')
                    if qwen_provider:
                        qwen_provider.processor = processor
                        qwen_provider.tokenizer = tokenizer  
                        qwen_provider.model = model
                        qwen_provider.model.eval()
                        qwen_provider.is_loaded = True
                        qwen_provider.is_installed = True
                        
                        if selected_model_key:
                            qwen_provider.loaded_model_size = selected_model_key
                    
                    progress_var.set(100)
                    progress_label.config(text="✅ Download complete!")
                    status_label.config(text="Model ready for Korean OCR!")
                    add_log("✓ Model ready to use!")
                    
                    self.dialog.after(0, self._check_provider_status)
                    
                elif provider == 'rapidocr':
                    progress_label.config(text="📦 RapidOCR Installation Instructions")
                    add_log("RapidOCR requires manual pip installation")
                    progress_var.set(20)
                    
                    add_log("Command to run:")
                    add_log("pip install rapidocr-onnxruntime")
                    progress_var.set(50)
                    
                    add_log("")
                    add_log("After installation:")
                    add_log("1. Close this dialog")
                    add_log("2. Click 'Load Model' to initialize RapidOCR")
                    add_log("3. Status should show '✅ Model loaded'")
                    progress_var.set(100)
                    
                    progress_label.config(text="📦 Installation instructions shown")
                    status_label.config(text="Manual pip install required")
                    
                    download_btn.config(state=tk.DISABLED)
                    cancel_btn.config(text="Close")                        
                        
            except Exception as e:
                progress_label.config(text="❌ Download failed")
                status_label.config(text=f"Error: {str(e)[:50]}")
                add_log(f"ERROR: {str(e)}")
                self._log(f"Download error: {str(e)}", "error")
                
            finally:
                download_active['value'] = False
        
        def start_download():
            """Start download in background thread or executor"""
            download_btn.config(state=tk.DISABLED)
            cancel_btn.config(text="Cancel")
            
            try:
                if hasattr(self.main_gui, '_ensure_executor'):
                    self.main_gui._ensure_executor()
                execu = getattr(self.main_gui, 'executor', None)
                if execu:
                    execu.submit(download_with_progress)
                else:
                    import threading
                    download_thread = threading.Thread(target=download_with_progress, daemon=True)
                    download_thread.start()
            except Exception:
                import threading
                download_thread = threading.Thread(target=download_with_progress, daemon=True)
                download_thread.start()
        
        def cancel_download():
            """Cancel or close dialog"""
            if download_active['value']:
                download_active['value'] = False
                status_label.config(text="Cancelling...")
            else:
                download_dialog.destroy()
        
        download_btn = tb.Button(button_frame, text="Download", command=start_download, bootstyle="primary")
        download_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = tb.Button(button_frame, text="Close", command=cancel_download, bootstyle="secondary")
        cancel_btn.pack(side=tk.LEFT, padx=5)
    
        # Auto-resize
        self.main_gui.wm.auto_resize_dialog(download_dialog, canvas, max_width_ratio=0.5, max_height_ratio=0.6)
    
    def _check_provider_status(self):
        """Check and display OCR provider status"""
        # Skip during initialization to prevent lag
        if hasattr(self, '_initializing_gui') and self._initializing_gui:
            self.provider_status_label.config(text="", fg="black")
            return
        provider = self.ocr_provider_var.get()
        
        # Hide ALL buttons first
        if hasattr(self, 'provider_setup_btn'):
            self.provider_setup_btn.pack_forget()
        if hasattr(self, 'download_model_btn'):
            self.download_model_btn.pack_forget()
        
        if provider == 'google':
            # Google - check for credentials file
            google_creds = self.main_gui.config.get('google_vision_credentials', '')
            if google_creds and os.path.exists(google_creds):
                self.provider_status_label.config(text="✅ Ready", fg="green")
            else:
                self.provider_status_label.config(text="❌ Credentials needed", fg="red")
            
        elif provider == 'azure':
            # Azure - check for API key
            azure_key = self.main_gui.config.get('azure_vision_key', '')
            if azure_key:
                self.provider_status_label.config(text="✅ Ready", fg="green")
            else:
                self.provider_status_label.config(text="❌ Key needed", fg="red")

        elif provider == 'custom-api':
            # Custom API - check for main API key
            api_key = None
            if hasattr(self.main_gui, 'api_key_entry') and self.main_gui.api_key_entry.get().strip():
                api_key = self.main_gui.api_key_entry.get().strip()
            elif hasattr(self.main_gui, 'config') and self.main_gui.config.get('api_key'):
                api_key = self.main_gui.config.get('api_key')
            
            # Check if AI bubble detection is enabled
            manga_settings = self.main_gui.config.get('manga_settings', {})
            ocr_settings = manga_settings.get('ocr', {})
            bubble_detection_enabled = ocr_settings.get('bubble_detection_enabled', False)
            
            if api_key:
                if bubble_detection_enabled:
                    self.provider_status_label.config(text="✅ Ready", fg="green")
                else:
                    self.provider_status_label.config(text="⚠️ Enable AI bubble detection for best results", fg="orange")
            else:
                self.provider_status_label.config(text="❌ API key needed", fg="red")
     
        elif provider == 'Qwen2-VL':
            # Load saved model size if available
            if hasattr(self, 'qwen2vl_model_size'):
                saved_model_size = self.qwen2vl_model_size
            else:
                saved_model_size = self.main_gui.config.get('qwen2vl_model_size', '1')
            
            # When displaying status for loaded model
            if status['loaded']:
                # Map the saved size to display name
                size_names = {'1': '2B', '2': '7B', '3': '72B', '4': 'custom'}
                display_size = size_names.get(saved_model_size, saved_model_size)
                self.provider_status_label.config(text=f"✅ {display_size} model loaded", fg="green")
 
        else:
            # Local OCR providers
            if not hasattr(self, 'ocr_manager'):
                from ocr_manager import OCRManager
                self.ocr_manager = OCRManager(log_callback=self._log)
                
            status = self.ocr_manager.check_provider_status(provider)
            
            if status['loaded']:
                # Model is loaded and ready
                if provider == 'Qwen2-VL':
                    # Check which model size is loaded
                    qwen_provider = self.ocr_manager.get_provider('Qwen2-VL')
                    if qwen_provider and hasattr(qwen_provider, 'loaded_model_size'):
                        model_size = qwen_provider.loaded_model_size
                        status_text = f"✅ {model_size} model loaded"
                    else:
                        status_text = "✅ Model loaded"
                    self.provider_status_label.config(text=status_text, fg="green")
                else:
                    self.provider_status_label.config(text="✅ Model loaded", fg="green")
                
                # Show reload button for all local providers
                self.provider_setup_btn.config(text="Reload", bootstyle="secondary")
                self.provider_setup_btn.pack(side=tk.LEFT, padx=(5, 0))
                
            elif status['installed']:
                # Dependencies installed but model not loaded
                self.provider_status_label.config(text="📦 Dependencies ready", fg="orange")
                
                # Show Load button for all providers
                self.provider_setup_btn.config(text="Load Model", bootstyle="primary")
                self.provider_setup_btn.pack(side=tk.LEFT, padx=(5, 0))
                
                # Also show Download button for models that need downloading
                if provider in ['Qwen2-VL', 'manga-ocr']:
                    self.download_model_btn.config(text="📥 Download Model")
                    self.download_model_btn.pack(side=tk.LEFT, padx=(5, 0))
                
            else:
                # Not installed
                self.provider_status_label.config(text="❌ Not installed", fg="red")
                
                # Categorize providers
                huggingface_providers = ['manga-ocr', 'Qwen2-VL', 'rapidocr']  # Move rapidocr here
                pip_providers = ['easyocr', 'paddleocr', 'doctr']  # Remove rapidocr from here

                if provider in huggingface_providers:
                    # For HuggingFace models, show BOTH buttons
                    self.provider_setup_btn.config(text="Load Model", bootstyle="primary")
                    self.provider_setup_btn.pack(side=tk.LEFT, padx=(5, 0))
                    
                    # Download button
                    if provider == 'rapidocr':
                        self.download_model_btn.config(text="📥 Install RapidOCR")
                    else:
                        self.download_model_btn.config(text=f"📥 Download {provider}")
                    self.download_model_btn.pack(side=tk.LEFT, padx=(5, 0))

                elif provider in pip_providers:
                    # Check if running as .exe
                    if getattr(sys, 'frozen', False):
                        # Running as .exe - can't pip install
                        self.provider_status_label.config(
                            text="❌ Not available in .exe", 
                            fg="red"
                        )
                        self._log(f"⚠️ {provider} cannot be installed in standalone .exe version", "warning")
                    else:
                        # Running from Python - can pip install
                        self.provider_setup_btn.config(text="Install", bootstyle="success")
                        self.provider_setup_btn.pack(side=tk.LEFT, padx=(5, 0))
                    
                elif provider in pip_providers:
                    # Check if running as .exe
                    if getattr(sys, 'frozen', False):
                        # Running as .exe - can't pip install
                        self.provider_status_label.config(
                            text="❌ Not available in .exe", 
                            fg="red"
                        )
                        self._log(f"⚠️ {provider} cannot be installed in standalone .exe version", "warning")
                    else:
                        # Running from Python - can pip install
                        self.provider_setup_btn.config(text="Install", bootstyle="success")
                        self.provider_setup_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # Additional GPU status check for Qwen2-VL
        if provider == 'Qwen2-VL' and not status['loaded']:
            try:
                import torch
                if not torch.cuda.is_available():
                    self._log("⚠️ No GPU detected - Qwen2-VL will run slowly on CPU", "warning")
            except ImportError:
                pass

    def _setup_ocr_provider(self):
        """Setup/install/load OCR provider"""
        provider = self.ocr_provider_var.get()
        
        if provider in ['google', 'azure']:
            return  # Cloud providers don't need setup

        # your own api key
        if provider == 'custom-api':
            # Open configuration dialog for custom API
            try:
                from custom_api_config_dialog import CustomAPIConfigDialog
                dialog = CustomAPIConfigDialog(
                    self.manga_window,
                    self.main_gui.config,
                    self.main_gui.save_config
                )
                # After dialog closes, refresh status
                self.dialog.after(100, self._check_provider_status)
            except ImportError:
                # If dialog not available, show message
                messagebox.showinfo(
                    "Custom API Configuration",
                    "This mode uses your own API key in the main GUI:\n\n"
                    "- Make sure your API supports vision\n"
                    "- api_key: Your API key\n"
                    "- model: Model name\n"
                    "- custom url: You can override API endpoint under Other settings"
                )
            return
        
        status = self.ocr_manager.check_provider_status(provider)
        
        # For Qwen2-VL, check if we need to select model size first
        model_size = None
        if provider == 'Qwen2-VL' and status['installed'] and not status['loaded']:
            # Use window manager for dialog
            selection_dialog, scrollable_frame, canvas = self.main_gui.wm.setup_scrollable(
                self.dialog,
                "Select Qwen2-VL Model Size",
                width=None,
                height=None,
                max_width_ratio=0.5,
                max_height_ratio=0.3
            )
            
            # Title
            title_frame = tk.Frame(scrollable_frame)
            title_frame.pack(fill=tk.X, pady=(10, 20))
            tk.Label(title_frame, text="Select Model Size to Load", 
                    font=('Arial', 12, 'bold')).pack()
            
            # Model selection frame
            model_frame = tk.LabelFrame(
                scrollable_frame,
                text="Available Models",
                font=('Arial', 11, 'bold'),
                padx=15,
                pady=10
            )
            model_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            # Model options
            model_options = {
                "1": {"name": "Qwen2-VL 2B", "desc": "Smallest (4-8GB VRAM)"},
                "2": {"name": "Qwen2-VL 7B", "desc": "Medium (12-16GB VRAM)"},
                "3": {"name": "Qwen2-VL 72B", "desc": "Largest (80GB+ VRAM)"},
                "4": {"name": "Custom Model", "desc": "Enter any HF model ID"},
            }
            
            selected_model = tk.StringVar(value="1")
            custom_model_id = tk.StringVar()
            
            for key, info in model_options.items():
                option_frame = tk.Frame(model_frame)
                option_frame.pack(fill=tk.X, pady=5)
                
                rb = tk.Radiobutton(
                    option_frame, 
                    text=f"{info['name']} - {info['desc']}", 
                    variable=selected_model, 
                    value=key,
                    font=('Arial', 10),
                    anchor='w'
                )
                rb.pack(anchor='w')
                
                if key != "4":
                    ttk.Separator(option_frame, orient='horizontal').pack(fill=tk.X, pady=(5, 0))
            
            # Custom model ID frame
            custom_frame = tk.LabelFrame(
                scrollable_frame,
                text="Custom Model Configuration",
                font=('Arial', 11, 'bold'),
                padx=15,
                pady=10
            )
            
            entry_frame = tk.Frame(custom_frame)
            entry_frame.pack(fill=tk.X, pady=5)
            tk.Label(entry_frame, text="Model ID:", font=('Arial', 10)).pack(side=tk.LEFT, padx=(0, 10))
            custom_entry = tk.Entry(entry_frame, textvariable=custom_model_id, width=35, font=('Arial', 10))
            custom_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            def toggle_custom_frame(*args):
                if selected_model.get() == "4":
                    custom_frame.pack(fill=tk.X, padx=20, pady=10, after=model_frame)
                else:
                    custom_frame.pack_forget()
            
            selected_model.trace('w', toggle_custom_frame)
            
            # Buttons with centering
            button_frame = tk.Frame(scrollable_frame)
            button_frame.pack(fill=tk.X, pady=20)
            
            button_inner_frame = tk.Frame(button_frame)
            button_inner_frame.pack()
            
            model_confirmed = {'value': False, 'size': None}
            
            def confirm_selection():
                selected = selected_model.get()
                self._log(f"DEBUG: Radio button selection = {selected}")
                if selected == "4":
                    if not custom_model_id.get().strip():
                        messagebox.showerror("Error", "Please enter a model ID")
                        return
                    model_confirmed['size'] = f"custom:{custom_model_id.get().strip()}"
                else:
                    model_confirmed['size'] = selected
                model_confirmed['value'] = True
                selection_dialog.destroy()
            
            load_btn = tk.Button(
                button_inner_frame, text="Load", command=confirm_selection,
                bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                padx=20, pady=8, cursor='hand2', width=12
            )
            load_btn.pack(side=tk.LEFT, padx=5)
            
            cancel_btn = tk.Button(
                button_inner_frame, text="Cancel", command=selection_dialog.destroy,
                bg='#9E9E9E', fg='white', font=('Arial', 10),
                padx=20, pady=8, cursor='hand2', width=12
            )
            cancel_btn.pack(side=tk.LEFT, padx=5)
            
            # Auto-resize and wait
            self.main_gui.wm.auto_resize_dialog(selection_dialog, canvas, max_width_ratio=0.5, max_height_ratio=0.35)
            self.dialog.wait_window(selection_dialog)
            
            if not model_confirmed['value']:
                return
            
            model_size = model_confirmed['size']
            self._log(f"DEBUG: Dialog closed, model_size set to: {model_size}")
        
        # Create progress dialog with window manager
        progress_dialog, progress_frame, canvas = self.main_gui.wm.setup_scrollable(
            self.dialog,
            f"Setting up {provider}",
            width=400,
            height=200,
            max_width_ratio=0.4,
            max_height_ratio=0.3
        )
        
        # Progress section
        progress_section = tk.LabelFrame(
            progress_frame,
            text="Setup Progress",
            font=('Arial', 11, 'bold'),
            padx=15,
            pady=10
        )
        progress_section.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        progress_label = tk.Label(progress_section, text="Initializing...", font=('Arial', 10))
        progress_label.pack(pady=(10, 15))
        
        try:
            # Try to use our custom progress bar style
            progress_bar = ttk.Progressbar(
                progress_section,
                length=350,
                mode='indeterminate',
                style="MangaProgress.Horizontal.TProgressbar"
            )
        except Exception:
            # Fallback to default if style not available yet
            progress_bar = ttk.Progressbar(
                progress_section,
                length=350,
                mode='indeterminate'
            )
        progress_bar.pack(pady=(0, 10))
        progress_bar.start(10)
        
        status_label = tk.Label(progress_section, text="", font=('Arial', 9), fg='#666666')
        status_label.pack(pady=(0, 10))
        
        def update_progress(message, percent=None):
            """Update progress display"""
            progress_label.config(text=message)
            if percent is not None:
                progress_bar.stop()
                progress_bar.config(mode='determinate', value=percent)
        
        def setup_thread():
            """Run setup in background thread"""
            nonlocal model_size
            try:
                success = False
                
                if not status['installed']:
                    # Install provider
                    update_progress(f"Installing {provider}...")
                    success = self.ocr_manager.install_provider(provider, update_progress)
                    
                    if not success:
                        update_progress("❌ Installation failed!", 0)
                        self._log(f"Failed to install {provider}", "error")
                        return
                
                # Load model
                update_progress(f"Loading {provider} model...")
                
                # Special handling for Qwen2-VL - pass model_size
                if provider == 'Qwen2-VL':
                    if success and model_size:
                        # Save the model size to config
                        self.qwen2vl_model_size = model_size
                        self.main_gui.config['qwen2vl_model_size'] = model_size
                        
                        # Save config immediately
                        if hasattr(self.main_gui, 'save_config'):
                            self.main_gui.save_config(show_message=False)
                    self._log(f"DEBUG: In thread, about to load with model_size={model_size}")
                    if model_size:
                        success = self.ocr_manager.load_provider(provider, model_size=model_size)
                        
                        if success:
                            provider_obj = self.ocr_manager.get_provider('Qwen2-VL')
                            if provider_obj:
                                provider_obj.loaded_model_size = {
                                    "1": "2B",
                                    "2": "7B", 
                                    "3": "72B",
                                    "4": "custom"
                                }.get(model_size, model_size)
                    else:
                        self._log("Warning: No model size specified for Qwen2-VL, defaulting to 2B", "warning")
                        success = self.ocr_manager.load_provider(provider, model_size="1")
                else:
                    success = self.ocr_manager.load_provider(provider)
                
                if success:
                    update_progress(f"✅ {provider} ready!", 100)
                    self._log(f"✅ {provider} is ready to use", "success")
                    self.dialog.after(0, self._check_provider_status)
                else:
                    update_progress("❌ Failed to load model!", 0)
                    self._log(f"Failed to load {provider} model", "error")
                
            except Exception as e:
                update_progress(f"❌ Error: {str(e)}", 0)
                self._log(f"Setup error: {str(e)}", "error")
                import traceback
                self._log(traceback.format_exc(), "debug")
            
            finally:
                self.dialog.after(2000, progress_dialog.destroy)
        
        # Auto-resize
        self.main_gui.wm.auto_resize_dialog(progress_dialog, canvas, max_width_ratio=0.4, max_height_ratio=0.3)
        
        # Start setup in background via executor if available
        try:
            if hasattr(self.main_gui, '_ensure_executor'):
                self.main_gui._ensure_executor()
            execu = getattr(self.main_gui, 'executor', None)
            if execu:
                execu.submit(setup_thread)
            else:
                import threading
                threading.Thread(target=setup_thread, daemon=True).start()
        except Exception:
            import threading
            threading.Thread(target=setup_thread, daemon=True).start()

    def _on_ocr_provider_change(self, event=None):
        """Handle OCR provider change"""
        provider = self.ocr_provider_var.get()
        
        # Hide ALL provider-specific frames first
        if hasattr(self, 'google_creds_frame'):
            self.google_creds_frame.pack_forget()
        if hasattr(self, 'azure_frame'):
            self.azure_frame.pack_forget()
        
        # Update the API label based on provider
        api_label_text = {
            'custom-api': "OCR: Custom API | Translation: API Key",
            'google': "OCR: Google Cloud Vision | Translation: API Key",
            'azure': "OCR: Azure Computer Vision | Translation: API Key",
            'rapidocr': "OCR: RapidOCR (Local) | Translation: API Key", 
            'manga-ocr': "OCR: Manga OCR (Japanese) | Translation: API Key",
            'Qwen2-VL': "OCR: Qwen2-VL (Korean) | Translation: API Key",
            'easyocr': "OCR: EasyOCR (Multi-lang) | Translation: API Key",
            'paddleocr': "OCR: PaddleOCR | Translation: API Key",
            'doctr': "OCR: DocTR | Translation: API Key"
        }.get(provider, f"OCR: {provider} | Translation: API Key")
        
        # Update the label in the UI
        for widget in self.parent_frame.winfo_children():
            if isinstance(widget, tk.LabelFrame) and "Translation Settings" in widget.cget("text"):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Frame):
                        for subchild in child.winfo_children():
                            if isinstance(subchild, tk.Label) and "OCR:" in subchild.cget("text"):
                                subchild.config(text=api_label_text)
                                break
        
        # Show only the relevant settings frame for the selected provider
        if provider == 'google':
            # Show Google credentials frame
            self.google_creds_frame.pack(fill=tk.X, pady=(0, 10), after=self.ocr_provider_frame)
            
        elif provider == 'azure':
            # Show Azure settings frame  
            self.azure_frame.pack(fill=tk.X, pady=(0, 10), after=self.ocr_provider_frame)
            
        # For all other providers (manga-ocr, Qwen2-VL, easyocr, paddleocr, doctr)
        # Don't show any cloud credential frames - they use local models
        
        # Check provider status to show appropriate buttons
        self._check_provider_status()
        
        # Log the change
        provider_descriptions = {
            'custom-api': "Custom API - use your own vision model",
            'google': "Google Cloud Vision (requires credentials)",
            'azure': "Azure Computer Vision (requires API key)",
            'manga-ocr': "Manga OCR - optimized for Japanese manga",
            'rapidocr': "RapidOCR - fast local OCR with region detection",
            'Qwen2-VL': "Qwen2-VL - a big model", 
            'easyocr': "EasyOCR - multi-language support",
            'paddleocr': "PaddleOCR - CJK language support",
            'doctr': "DocTR - document text recognition"
        }
        
        self._log(f"📋 OCR provider changed to: {provider_descriptions.get(provider, provider)}", "info")
        
        # Save the selection
        self.main_gui.config['manga_ocr_provider'] = provider
        if hasattr(self.main_gui, 'save_config'):
            self.main_gui.save_config(show_message=False)
        
        # IMPORTANT: Reset translator to force recreation with new OCR provider
        if hasattr(self, 'translator') and self.translator:
            self._log(f"OCR provider changed to {provider.upper()}. Translator will be recreated on next run.", "info")
            self.translator = None  # Force recreation on next translation
    
    def _build_interface(self):
        """Build the enhanced manga translation interface"""
        # Title
        title_frame = tk.Frame(self.parent_frame)
        title_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        title_label = tk.Label(
            title_frame,
            text="🎌 Manga Translation",
            font=('Arial', 16, 'bold')
        )
        title_label.pack(side=tk.LEFT)
        
        # Requirements check
        has_api_key = bool(self.main_gui.api_key_entry.get().strip())
        has_vision = os.path.exists(self.main_gui.config.get('google_vision_credentials', ''))
        
        status_text = "✅ Ready" if (has_api_key and has_vision) else "❌ Setup Required"
        status_color = "green" if (has_api_key and has_vision) else "red"
        
        status_label = tk.Label(
            title_frame,
            text=status_text,
            font=('Arial', 12),
            fg=status_color
        )
        status_label.pack(side=tk.RIGHT)
        
        # Store reference for updates
        self.status_label = status_label
        
        # Add instructions and Google Cloud setup
        if not (has_api_key and has_vision):
            req_frame = tk.Frame(self.parent_frame)
            req_frame.pack(fill=tk.X, padx=20, pady=5)
            
            req_text = []
            if not has_api_key:
                req_text.append("• API Key not configured")
            if not has_vision:
                req_text.append("• Google Cloud Vision credentials not set")
            
            tk.Label(
                req_frame,
                text="\n".join(req_text),
                font=('Arial', 10),
                fg='red',
                justify=tk.LEFT
            ).pack(anchor=tk.W)
        
        # File selection frame
        file_frame = tk.LabelFrame(
            self.parent_frame,
            text="Select Manga Images",
            font=('Arial', 12, 'bold'),
            padx=15,
            pady=10
        )
        file_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # File listbox with scrollbar
        list_frame = tk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.file_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            height=8,
            selectmode=tk.EXTENDED
        )
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.file_listbox.yview)
        
        # File buttons
        file_btn_frame = tk.Frame(file_frame)
        file_btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        tb.Button(
            file_btn_frame,
            text="Add Files",
            command=self._add_files,
            bootstyle="primary"
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        tb.Button(
            file_btn_frame,
            text="Add Folder",
            command=self._add_folder,
            bootstyle="primary"
        ).pack(side=tk.LEFT, padx=5)
        
        tb.Button(
            file_btn_frame,
            text="Remove Selected",
            command=self._remove_selected,
            bootstyle="danger"
        ).pack(side=tk.LEFT, padx=5)
        
        tb.Button(
            file_btn_frame,
            text="Clear All",
            command=self._clear_all,
            bootstyle="warning"
        ).pack(side=tk.LEFT, padx=5)
        
        # Settings frame
        settings_frame = tk.LabelFrame(
            self.parent_frame,
            text="Translation Settings",
            font=('Arial', 12, 'bold'),
            padx=15,
            pady=10
        )
      
        settings_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # API Settings - Hybrid approach
        api_frame = tk.Frame(settings_frame)
        api_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(api_frame, text="OCR: Google Cloud Vision | Translation: API Key", 
                font=('Arial', 10, 'italic'), fg='gray').pack(side=tk.LEFT)
        
        # Show current model
        current_model = 'Unknown'
        if hasattr(self.main_gui, 'model_var'):
            current_model = self.main_gui.model_var.get()
        elif hasattr(self.main_gui, 'model_combo'):
            current_model = self.main_gui.model_combo.get()
        elif hasattr(self.main_gui, 'config'):
            current_model = self.main_gui.config.get('model', 'Unknown')
        
        tk.Label(api_frame, text=f"Model: {current_model}", 
                font=('Arial', 10, 'italic'), fg='gray').pack(side=tk.RIGHT)

        # OCR Provider Selection - ENHANCED VERSION
        self.ocr_provider_frame = tk.Frame(settings_frame)
        self.ocr_provider_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(self.ocr_provider_frame, text="OCR Provider:", width=20, anchor='w').pack(side=tk.LEFT)

        # Expanded provider list with descriptions
        ocr_providers = [
            ('custom-api', 'Your Own key'),
            ('google', 'Google Cloud Vision'),
            ('azure', 'Azure Computer Vision'),
            ('rapidocr', '⚡ RapidOCR (Fast & Local)'),
            ('manga-ocr', '🇯🇵 Manga OCR (Japanese)'),
            ('Qwen2-VL', '🇰🇷 Qwen2-VL (Korean)'),
            ('easyocr', '🌏 EasyOCR (Multi-lang)'),
            #('paddleocr', '🐼 PaddleOCR'),
            ('doctr', '📄 DocTR'),
        ]

        # Just the values for the combobox
        provider_values = [p[0] for p in ocr_providers]
        provider_display = [f"{p[0]} - {p[1]}" for p in ocr_providers]

        self.ocr_provider_var = tk.StringVar(value=self.main_gui.config.get('manga_ocr_provider', 'custom-api'))
        provider_combo = ttk.Combobox(
            self.ocr_provider_frame,
            textvariable=self.ocr_provider_var,
            values=provider_values,
            state='readonly',
            width=15
        )
        provider_combo.pack(side=tk.LEFT, padx=10)
        provider_combo.bind('<<ComboboxSelected>>', self._on_ocr_provider_change)
        # Prevent mouse wheel from changing selection while scrolling the page
        provider_combo.bind("<MouseWheel>", lambda e: "break")
        provider_combo.bind("<Button-4>", lambda e: "break")  # Linux scroll up
        provider_combo.bind("<Button-5>", lambda e: "break")  # Linux scroll down

        # Provider status indicator with more detail
        self.provider_status_label = tk.Label(
            self.ocr_provider_frame,
            text="",
            font=('Arial', 9),
            width=40
        )
        self.provider_status_label.pack(side=tk.LEFT, padx=(10, 0))

        # Setup/Install button for non-cloud providers - ALWAYS VISIBLE for local providers
        self.provider_setup_btn = tb.Button(
            self.ocr_provider_frame,
            text="Setup",
            command=self._setup_ocr_provider,
            bootstyle="info",
            width=12
        )
        # Don't pack yet, let _check_provider_status handle it

        # Add explicit download button for Hugging Face models
        self.download_model_btn = tb.Button(
            self.ocr_provider_frame,
            text="📥 Download",
            command=self._download_hf_model,
            bootstyle="success",
            width=22
        )
        # Don't pack yet

        # Initialize OCR manager
        from ocr_manager import OCRManager
        self.ocr_manager = OCRManager(log_callback=self._log)

        # Check initial provider status
        self._check_provider_status()

        # Google Cloud Credentials section (now in a frame that can be hidden)
        self.google_creds_frame = tk.Frame(settings_frame)
        self.google_creds_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(self.google_creds_frame, text="Google Cloud Credentials:", width=20, anchor='w').pack(side=tk.LEFT)

        # Show current credentials file
        google_creds_path = self.main_gui.config.get('google_vision_credentials', '') or self.main_gui.config.get('google_cloud_credentials', '')
        creds_display = os.path.basename(google_creds_path) if google_creds_path else "Not Set"

        self.creds_label = tk.Label(self.google_creds_frame, text=creds_display, 
                                   font=('Arial', 9), fg='green' if google_creds_path else 'red')
        self.creds_label.pack(side=tk.LEFT, padx=10)

        tb.Button(
            self.google_creds_frame,
            text="Browse",
            command=self._browse_google_credentials_permanent,
            bootstyle="primary"
        ).pack(side=tk.LEFT)

        # Azure settings frame (hidden by default)
        self.azure_frame = tk.Frame(settings_frame)

        # Azure Key
        azure_key_frame = tk.Frame(self.azure_frame)
        azure_key_frame.pack(fill=tk.X, pady=(0, 5))

        tk.Label(azure_key_frame, text="Azure Key:", width=20, anchor='w').pack(side=tk.LEFT)
        self.azure_key_entry = tk.Entry(azure_key_frame, show='*', width=30)
        self.azure_key_entry.pack(side=tk.LEFT, padx=10)

        # Show/Hide button for Azure key
        self.show_azure_key_var = tk.BooleanVar(value=False)
        tb.Checkbutton(
            azure_key_frame,
            text="Show",
            variable=self.show_azure_key_var,
            command=lambda: self.azure_key_entry.config(show='' if self.show_azure_key_var.get() else '*'),
            bootstyle="secondary"
        ).pack(side=tk.LEFT, padx=5)

        # Azure Endpoint
        azure_endpoint_frame = tk.Frame(self.azure_frame)
        azure_endpoint_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(azure_endpoint_frame, text="Azure Endpoint:", width=20, anchor='w').pack(side=tk.LEFT)
        self.azure_endpoint_entry = tk.Entry(azure_endpoint_frame, width=40)
        self.azure_endpoint_entry.pack(side=tk.LEFT, padx=10)

        # Load saved Azure settings
        saved_key = self.main_gui.config.get('azure_vision_key', '')
        saved_endpoint = self.main_gui.config.get('azure_vision_endpoint', 'https://YOUR-RESOURCE.cognitiveservices.azure.com/')
        self.azure_key_entry.insert(0, saved_key)
        self.azure_endpoint_entry.insert(0, saved_endpoint)

        # Initially show/hide based on saved provider
        self._on_ocr_provider_change()

        # Separator for context settings
        ttk.Separator(settings_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 10))
        
        # Context and Full Page Mode Settings
        context_frame = tk.LabelFrame(
            settings_frame,
            text="🔄 Context & Translation Mode",
            font=('Arial', 11, 'bold'),
            padx=10,
            pady=10
        )
        context_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Show current contextual settings from main GUI
        context_info = tk.Frame(context_frame)
        context_info.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(
            context_info,
            text="Main GUI Context Settings:",
            font=('Arial', 10, 'bold')
        ).pack(anchor=tk.W)
        
        # Display current settings
        settings_frame_display = tk.Frame(context_info)
        settings_frame_display.pack(fill=tk.X, padx=(20, 0))
        
        # Contextual enabled status
        contextual_status = "Enabled" if self.main_gui.contextual_var.get() else "Disabled"
        self.contextual_status_label = tk.Label(
            settings_frame_display,
            text=f"• Contextual Translation: {contextual_status}",
            font=('Arial', 10)
        )
        self.contextual_status_label.pack(anchor=tk.W)
        
        # History limit
        history_limit = self.main_gui.trans_history.get() if hasattr(self.main_gui, 'trans_history') else "3"
        self.history_limit_label = tk.Label(
            settings_frame_display,
            text=f"• Translation History Limit: {history_limit} exchanges",
            font=('Arial', 10)
        )
        self.history_limit_label.pack(anchor=tk.W)
        
        # Rolling history status
        rolling_status = "Enabled (Rolling Window)" if self.main_gui.translation_history_rolling_var.get() else "Disabled (Reset on Limit)"
        self.rolling_status_label = tk.Label(
            settings_frame_display,
            text=f"• Rolling History: {rolling_status}",
            font=('Arial', 10)
        )
        self.rolling_status_label.pack(anchor=tk.W)

        # Refresh button to update from main GUI
        tb.Button(
            context_frame,
            text="↻ Refresh from Main GUI",
            command=self._refresh_context_settings,
            bootstyle="secondary"
        ).pack(pady=(10, 0))
        
        # Separator
        ttk.Separator(context_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 10))
        
        # Full Page Context Translation Settings
        full_page_frame = tk.Frame(context_frame)
        full_page_frame.pack(fill=tk.X)

        tk.Label(
            full_page_frame,
            text="Full Page Context Mode (Manga-specific):",
            font=('Arial', 10, 'bold')
        ).pack(anchor=tk.W, pady=(0, 5))

        # Enable/disable toggle
        self.full_page_context_var = tk.BooleanVar(
            value=self.main_gui.config.get('manga_full_page_context', True)
        )

        toggle_frame = tk.Frame(full_page_frame)
        toggle_frame.pack(fill=tk.X, padx=(20, 0))

        self.context_checkbox = tb.Checkbutton(
            toggle_frame,
            text="Enable Full Page Context Translation",
            variable=self.full_page_context_var,
            command=self._on_context_toggle,
            bootstyle="round-toggle"
        )
        self.context_checkbox.pack(side=tk.LEFT)

        # Edit prompt button
        tb.Button(
            toggle_frame,
            text="Edit Prompt",
            command=self._edit_context_prompt,
            bootstyle="secondary"
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Help button for full page context
        tb.Button(
            toggle_frame,
            text="?",
            command=lambda: self._show_help_dialog(
                "Full Page Context Mode",
                "Full page context sends all text regions from the page together in a single request.\n\n"
                "This allows the AI to see all text at once for more contextually accurate translations, "
                "especially useful for maintaining character name consistency and understanding "
                "conversation flow across multiple speech bubbles.\n\n"
                "✅ Pros:\n"
                "• Better context awareness\n"
                "• Consistent character names\n"
                "• Understanding of conversation flow\n"
                "• Maintains tone across bubbles\n\n"
                "❌ Cons:\n"
                "• Single API call failure affects all text\n"
                "• May use more tokens\n"
                "• Slower for pages with many text regions"
            ),
            bootstyle="info",
            width=2
        ).pack(side=tk.LEFT, padx=(5, 0))

        # Separator
        ttk.Separator(context_frame, orient='horizontal').pack(fill=tk.X, pady=(10, 10))

        # Visual Context Settings (for non-vision model support)
        visual_frame = tk.Frame(context_frame)
        visual_frame.pack(fill=tk.X)

        tk.Label(
            visual_frame,
            text="Visual Context (Image Support):",
            font=('Arial', 10, 'bold')
        ).pack(anchor=tk.W, pady=(0, 5))

        # Visual context toggle
        self.visual_context_enabled_var = tk.BooleanVar(
            value=self.main_gui.config.get('manga_visual_context_enabled', True)
        )

        visual_toggle_frame = tk.Frame(visual_frame)
        visual_toggle_frame.pack(fill=tk.X, padx=(20, 0))

        self.visual_context_checkbox = tb.Checkbutton(
            visual_toggle_frame,
            text="Include page image in translation requests",
            variable=self.visual_context_enabled_var,
            command=self._on_visual_context_toggle,
            bootstyle="round-toggle"
        )
        self.visual_context_checkbox.pack(side=tk.LEFT)

        # Help button for visual context
        tb.Button(
            visual_toggle_frame,
            text="?",
            command=lambda: self._show_help_dialog(
                "Visual Context Settings",
                "Visual context includes the manga page image with translation requests.\n\n"
                "⚠️ WHEN TO DISABLE:\n"
                "• Using text-only models (Claude, GPT-3.5, standard Gemini)\n"
                "• Model doesn't support images\n"
                "• Want to reduce token usage\n"
                "• Testing text-only translation\n\n"
                "✅ WHEN TO ENABLE:\n"
                "• Using vision models (Gemini Vision, GPT-4V, Claude 3)\n"
                "• Want spatial awareness of text position\n"
                "• Need visual context for better translation\n\n"
                "Impact:\n"
                "• Disabled: Only text is sent (compatible with any model)\n"
                "• Enabled: Text + image sent (requires vision model)\n\n"
                "Note: Disabling may reduce translation quality as the AI won't see\n"
                "the artwork context or spatial layout of the text."
            ),
            bootstyle="info",
            width=2
        ).pack(side=tk.LEFT, padx=(5, 0))
        
        # Text Rendering Settings Frame
        render_frame = tk.LabelFrame(
            self.parent_frame,
            text="Text Visibility Settings",
            font=('Arial', 12, 'bold'),
            padx=15,
            pady=10
        )
        render_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Advanced Settings button at the top of render_frame
        advanced_button_frame = tk.Frame(render_frame)
        advanced_button_frame.pack(fill=tk.X, pady=(0, 10))

        tb.Button(
            advanced_button_frame,
            text="⚙️ Advanced Settings",
            command=self._open_advanced_settings,
            bootstyle="info"
        ).pack(side=tk.RIGHT)

        tk.Label(
            advanced_button_frame,
            text="Configure OCR, preprocessing, and performance options",
            font=('Arial', 9),
            fg='gray'
        ).pack(side=tk.LEFT)

        # Force Caps Lock Checkbox
        force_caps_check = ttk.Checkbutton(
            render_frame,
            text="Force CAPS LOCK (All text in uppercase)",
            variable=self.force_caps_lock_var,
            command=self._apply_rendering_settings
        )
        force_caps_check.pack(anchor=tk.W, padx=20, pady=(0, 5))
        
        # Background opacity slider
        opacity_frame = tk.Frame(render_frame)
        opacity_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(opacity_frame, text="Background Opacity:", width=20, anchor='w').pack(side=tk.LEFT)
        
        opacity_scale = tk.Scale(
            opacity_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self.bg_opacity_var,
            command=self._update_opacity_label,
            length=200
        )
        opacity_scale.pack(side=tk.LEFT, padx=10)
        
        self.opacity_label = tk.Label(opacity_frame, text="100%", width=5)
        self.opacity_label.pack(side=tk.LEFT)
        
        # Initialize the label with the loaded value
        self._update_opacity_label(self.bg_opacity_var.get())

        # Free text only background opacity toggle (applies BG opacity only to free-text regions)
        ft_only_frame = tk.Frame(render_frame)
        ft_only_frame.pack(fill=tk.X, pady=(0, 5))
        tb.Checkbutton(
            ft_only_frame,
            text="Free text only background opacity",
            variable=self.free_text_only_bg_opacity_var,
            bootstyle="round-toggle",
            command=self._apply_rendering_settings
        ).pack(anchor='w')

        # Background style selection
        style_frame = tk.Frame(render_frame)
        style_frame.pack(fill=tk.X, pady=5)

        tk.Label(style_frame, text="Background Style:", width=20, anchor='w').pack(side=tk.LEFT)

        # Radio buttons for background style
        style_selection_frame = tk.Frame(style_frame)
        style_selection_frame.pack(side=tk.LEFT, padx=10)

        tb.Radiobutton(
            style_selection_frame,
            text="Box",
            variable=self.bg_style_var,
            value="box",
            command=self._save_rendering_settings,
            bootstyle="primary"
        ).pack(side=tk.LEFT, padx=(0, 10))

        tb.Radiobutton(
            style_selection_frame,
            text="Circle",
            variable=self.bg_style_var,
            value="circle",
            command=self._save_rendering_settings,
            bootstyle="primary"
        ).pack(side=tk.LEFT, padx=(0, 10))

        tb.Radiobutton(
            style_selection_frame,
            text="Wrap",
            variable=self.bg_style_var,
            value="wrap",
            command=self._save_rendering_settings,
            bootstyle="primary"
        ).pack(side=tk.LEFT)

        # Add tooltips or descriptions
        style_help = tk.Label(
            style_frame,
            text="(Box: rounded rectangle, Circle: ellipse, Wrap: per-line)",
            font=('Arial', 9),
            fg='gray'
        )
        style_help.pack(side=tk.LEFT, padx=(10, 0))
 
        # Skip inpainting toggle - store as instance variable
        self.skip_inpainting_checkbox = tb.Checkbutton(
            render_frame, 
            text="Skip Inpainter", 
            variable=self.skip_inpainting_var,
            bootstyle="round-toggle",
            command=self._toggle_inpaint_visibility
        )
        self.skip_inpainting_checkbox.pack(anchor='w', pady=5)

        # Inpainting method selection (only visible when inpainting is enabled)
        self.inpaint_method_frame = tk.Frame(render_frame)
        self.inpaint_method_frame.pack(fill=tk.X, pady=5)

        tk.Label(self.inpaint_method_frame, text="Inpaint Method:", width=20, anchor='w').pack(side=tk.LEFT)

        # Radio buttons for inpaint method
        method_selection_frame = tk.Frame(self.inpaint_method_frame)
        method_selection_frame.pack(side=tk.LEFT, padx=5)

        self.inpaint_method_var = tk.StringVar(value=self.main_gui.config.get('manga_inpaint_method', 'cloud'))

        tb.Radiobutton(
            method_selection_frame,
            text="Cloud API",
            variable=self.inpaint_method_var,
            value="cloud",
            command=self._on_inpaint_method_change,
            bootstyle="primary"
        ).pack(side=tk.LEFT, padx=(0, 10))

        tb.Radiobutton(
            method_selection_frame,
            text="Local Model",
            variable=self.inpaint_method_var,
            value="local",
            command=self._on_inpaint_method_change,
            bootstyle="primary"
        ).pack(side=tk.LEFT, padx=(0, 10))

        tb.Radiobutton(
            method_selection_frame,
            text="Hybrid",
            variable=self.inpaint_method_var,
            value="hybrid",
            command=self._on_inpaint_method_change,
            bootstyle="primary"
        ).pack(side=tk.LEFT)

        # Cloud settings frame
        self.cloud_inpaint_frame = tk.Frame(render_frame)
        self.cloud_inpaint_frame.pack(fill=tk.X, pady=5)

        # Quality selection for cloud
        quality_frame = tk.Frame(self.cloud_inpaint_frame)
        quality_frame.pack(fill=tk.X)

        tk.Label(quality_frame, text="Cloud Quality:", width=20, anchor='w').pack(side=tk.LEFT)

        quality_options = [('high', 'High Quality'), ('fast', 'Fast')]
        for value, text in quality_options:
            tb.Radiobutton(
                quality_frame,
                text=text,
                variable=self.inpaint_quality_var,
                value=value,
                bootstyle="primary",
                command=self._save_rendering_settings
            ).pack(side=tk.LEFT, padx=10)

        # Conditional separator
        self.inpaint_separator = ttk.Separator(render_frame, orient='horizontal')
        if not self.skip_inpainting_var.get():
            self.inpaint_separator.pack(fill=tk.X, pady=(10, 10))

        # Cloud API status
        api_status_frame = tk.Frame(self.cloud_inpaint_frame)
        api_status_frame.pack(fill=tk.X, pady=(10, 0))

        # Check if API key exists
        saved_api_key = self.main_gui.config.get('replicate_api_key', '')
        if saved_api_key:
            status_text = "✅ Cloud API configured"
            status_color = 'green'
        else:
            status_text = "❌ Cloud API not configured"
            status_color = 'red'

        self.inpaint_api_status_label = tk.Label(
            api_status_frame, 
            text=status_text,
            font=('Arial', 9),
            fg=status_color
        )
        self.inpaint_api_status_label.pack(side=tk.LEFT)

        tb.Button(
            api_status_frame,
            text="Configure API Key",
            command=self._configure_inpaint_api,
            bootstyle="info"
        ).pack(side=tk.LEFT, padx=(10, 0))

        if saved_api_key:
            tb.Button(
                api_status_frame,
                text="Clear",
                command=self._clear_inpaint_api,
                bootstyle="secondary"
            ).pack(side=tk.LEFT, padx=(5, 0))

        # Local inpainting settings frame
        self.local_inpaint_frame = tk.Frame(render_frame)

        # Local model selection
        local_model_frame = tk.Frame(self.local_inpaint_frame)
        local_model_frame.pack(fill=tk.X)

        tk.Label(local_model_frame, text="Local Model:", width=20, anchor='w').pack(side=tk.LEFT)

        self.local_model_type_var = tk.StringVar(value=self.main_gui.config.get('manga_local_inpaint_model', 'anime_onnx'))
        local_model_combo = ttk.Combobox(
            local_model_frame,
            textvariable=self.local_model_type_var,
            values= ['aot', 'aot_onnx', 'lama', 'lama_onnx', 'anime', 'anime_onnx', 'mat', 'ollama', 'sd_local'],
            state='readonly',
            width=15
        )
        local_model_combo.pack(side=tk.LEFT, padx=10)
        local_model_combo.bind('<<ComboboxSelected>>', self._on_local_model_change)
        self._disable_spinbox_mousewheel(local_model_combo)

        # Model descriptions
        model_desc = {
            'lama': 'LaMa (Best quality)',
            'aot': 'AOT GAN (Fast)',
            'aot_onnx': 'AOT ONNX (Optimized)',
            'mat': 'MAT (High-res)',
            'sd_local': 'Stable Diffusion (Anime)',
            'anime': 'Anime/Manga Inpainting',
            'anime_onnx': 'Anime ONNX (Fast/Optimized)',
            'lama_onnx': 'LaMa ONNX (Optimized)',
        }
        self.model_desc_label = tk.Label(
            local_model_frame,
            text=model_desc.get(self.local_model_type_var.get(), ''),
            font=('Arial', 9),
            fg='gray'
        )
        self.model_desc_label.pack(side=tk.LEFT, padx=(10, 0))

        # Model file selection
        model_path_frame = tk.Frame(self.local_inpaint_frame)
        model_path_frame.pack(fill=tk.X, pady=(5, 0))

        tk.Label(model_path_frame, text="Model File:", width=20, anchor='w').pack(side=tk.LEFT)

        self.local_model_path_var = tk.StringVar(
            value=self.main_gui.config.get(f'manga_{self.local_model_type_var.get()}_model_path', '')
        )
        self.local_model_entry = tk.Entry(
            model_path_frame,
            textvariable=self.local_model_path_var,
            width=30,
            state='readonly',
            bg='#2b2b2b',  # Dark gray background
            fg='#ffffff',  # White text
            readonlybackground='#2b2b2b'  # Gray even when readonly
        )
        self.local_model_entry.pack(side=tk.LEFT, padx=(10, 5))

        tb.Button(
            model_path_frame,
            text="Browse",
            command=self._browse_local_model,
            bootstyle="primary"
        ).pack(side=tk.LEFT)
        
        # Manual load button to avoid auto-loading on dialog open
        tb.Button(
            model_path_frame,
            text="Load",
            command=self._click_load_local_model,
            bootstyle="success"
        ).pack(side=tk.LEFT, padx=(5, 0))

        # Model status
        self.local_model_status_label = tk.Label(
            self.local_inpaint_frame,
            text="",
            font=('Arial', 9)
        )
        self.local_model_status_label.pack(anchor='w', pady=(5, 0))

        # Download model button
        tb.Button(
            self.local_inpaint_frame,
            text="📥 Download Model",
            command=self._download_model,
            bootstyle="info"
        ).pack(anchor='w', pady=(5, 0))

        # Model info button
        tb.Button(
            self.local_inpaint_frame,
            text="ℹ️ Model Info",
            command=self._show_model_info,
            bootstyle="secondary"
        ).pack(anchor='w', pady=(5, 0))

        # Try to load saved model for current type on dialog open
        initial_model_type = self.local_model_type_var.get()
        initial_model_path = self.main_gui.config.get(f'manga_{initial_model_type}_model_path', '')

        if initial_model_path and os.path.exists(initial_model_path):
            self.local_model_path_var.set(initial_model_path)
            if getattr(self, 'preload_local_models_on_open', False):
                self.local_model_status_label.config(text="⏳ Loading saved model...", fg='orange')
                # Auto-load after dialog is ready
                self.dialog.after(500, lambda: self._try_load_model(initial_model_type, initial_model_path))
            else:
                # Do not auto-load large models at startup to avoid crashes on some systems
                self.local_model_status_label.config(
                    text="💤 Saved model detected (not loaded). Click 'Load' to initialize.",
                    fg='blue'
                )
        else:
            self.local_model_status_label.config(text="No model loaded", fg='gray')

        # Initialize visibility based on current settings
        self._toggle_inpaint_visibility()
        
        # Background size reduction
        reduction_frame = tk.Frame(render_frame)
        reduction_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(reduction_frame, text="Background Size:", width=20, anchor='w').pack(side=tk.LEFT)
        
        reduction_scale = tk.Scale(
            reduction_frame,
            from_=0.5,
            to=2.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.bg_reduction_var,
            command=self._update_reduction_label,
            length=200
        )
        reduction_scale.pack(side=tk.LEFT, padx=10)
        
        self.reduction_label = tk.Label(reduction_frame, text="100%", width=5)
        self.reduction_label.pack(side=tk.LEFT)
        
        # Initialize the label with the loaded value
        self._update_reduction_label(self.bg_reduction_var.get())
        
        # Font size selection with mode toggle
        font_frame = tk.Frame(render_frame)
        font_frame.pack(fill=tk.X, pady=5)
        
        # Mode selection frame
        mode_frame = tk.Frame(font_frame)
        mode_frame.pack(fill=tk.X)

        tk.Label(mode_frame, text="Font Size Mode:", width=20, anchor='w').pack(side=tk.LEFT)

        # Radio buttons for mode selection
        mode_selection_frame = tk.Frame(mode_frame)
        mode_selection_frame.pack(side=tk.LEFT, padx=10)

        tb.Radiobutton(
            mode_selection_frame,
            text="Fixed Size",
            variable=self.font_size_mode_var,
            value="fixed",
            command=self._toggle_font_size_mode,
            bootstyle="primary"
        ).pack(side=tk.LEFT, padx=(0, 10))

        tb.Radiobutton(
            mode_selection_frame,
            text="Dynamic Multiplier",
            variable=self.font_size_mode_var,
            value="multiplier",
            command=self._toggle_font_size_mode,
            bootstyle="primary"
        ).pack(side=tk.LEFT)

        # Fixed font size frame
        self.fixed_size_frame = tk.Frame(font_frame)
        # Don't pack yet - let _toggle_font_size_mode handle it

        tk.Label(self.fixed_size_frame, text="Font Size:", width=20, anchor='w').pack(side=tk.LEFT)

        font_size_spinbox = tb.Spinbox(
            self.fixed_size_frame,
            from_=0,
            to=72,
            textvariable=self.font_size_var,
            width=10,
            command=self._save_rendering_settings
        )
        font_size_spinbox.pack(side=tk.LEFT, padx=10)
        # Also bind to save on manual entry
        font_size_spinbox.bind('<Return>', lambda e: self._save_rendering_settings())
        font_size_spinbox.bind('<FocusOut>', lambda e: self._save_rendering_settings())
        self._disable_spinbox_mousewheel(font_size_spinbox)

        tk.Label(self.fixed_size_frame, text="(0 = Auto)", font=('Arial', 9), fg='gray').pack(side=tk.LEFT)

        # Dynamic multiplier frame
        self.multiplier_frame = tk.Frame(font_frame)
        # Don't pack yet - let _toggle_font_size_mode handle it

        tk.Label(self.multiplier_frame, text="Size Multiplier:", width=20, anchor='w').pack(side=tk.LEFT)

        multiplier_scale = tk.Scale(
            self.multiplier_frame,
            from_=0.5,
            to=2.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.font_size_multiplier_var,
            command=self._update_multiplier_label,
            length=200
        )
        multiplier_scale.pack(side=tk.LEFT, padx=10)

        self.multiplier_label = tk.Label(self.multiplier_frame, text="1.0x", width=5)
        self.multiplier_label.pack(side=tk.LEFT)

        tk.Label(self.multiplier_frame, text="(Scales with panel size)", font=('Arial', 9), fg='gray').pack(side=tk.LEFT, padx=5)

        # Constraint checkbox frame (only visible in multiplier mode)
        self.constraint_frame = tk.Frame(font_frame)
        # Don't pack yet - let _toggle_font_size_mode handle it
        
        self.constrain_checkbox = tb.Checkbutton(
            self.constraint_frame,
            text="Constrain text to bubble boundaries",
            variable=self.constrain_to_bubble_var,
            command=self._save_rendering_settings,
            bootstyle="primary"
        )
        self.constrain_checkbox.pack(side=tk.LEFT, padx=(20, 0))

        tk.Label(
            self.constraint_frame, 
            text="(Unchecked allows text to exceed bubbles)", 
            font=('Arial', 9), 
            fg='gray'
        ).pack(side=tk.LEFT, padx=5)

        # Initialize visibility AFTER all frames are created
        self._toggle_font_size_mode()

        # Minimum font size setting (for auto mode)
        min_size_frame = tk.Frame(render_frame)
        min_size_frame.pack(fill=tk.X, pady=5)

        tk.Label(min_size_frame, text="Minimum Font Size:", width=20, anchor='w').pack(side=tk.LEFT)

        min_size_spinbox = ttk.Spinbox(
            min_size_frame,
            from_=10,
            to=24,
            textvariable=self.min_readable_size_var,
            width=10,
            command=self._save_rendering_settings
        )
        min_size_spinbox.pack(side=tk.LEFT, padx=10)
        self._disable_spinbox_mousewheel(min_size_spinbox)


        tk.Label(
            min_size_frame, 
            text="(Auto mode won't go below this)", 
            font=('Arial', 9), 
            fg='gray'
        ).pack(side=tk.LEFT, padx=5)
    
        # Maximum font size setting
        max_size_frame = tk.Frame(render_frame)
        max_size_frame.pack(fill=tk.X, pady=5)

        tk.Label(max_size_frame, text="Maximum Font Size:", width=20, anchor='w').pack(side=tk.LEFT)

        max_size_spinbox = ttk.Spinbox(
            max_size_frame,
            from_=20,
            to=100,
            textvariable=self.max_font_size_var,
            width=10,
            command=self._save_rendering_settings
        )
        max_size_spinbox.pack(side=tk.LEFT, padx=10)
        self._disable_spinbox_mousewheel(max_size_spinbox)

        tk.Label(
            max_size_frame, 
            text="(Limits maximum text size)", 
            font=('Arial', 9), 
            fg='gray'
        ).pack(side=tk.LEFT, padx=5)

        # Text wrapping mode
        wrap_frame = tk.Frame(render_frame)
        wrap_frame.pack(fill=tk.X, pady=5)

        self.strict_wrap_checkbox = tb.Checkbutton(
            wrap_frame,
            text="Strict text wrapping (force text to fit within bubbles)",
            variable=self.strict_text_wrapping_var,
            command=self._save_rendering_settings,
            bootstyle="primary"
        )
        self.strict_wrap_checkbox.pack(side=tk.LEFT)

        tk.Label(
            wrap_frame, 
            text="(Break words with hyphens if needed)", 
            font=('Arial', 9), 
            fg='gray'
        ).pack(side=tk.LEFT, padx=5)
    
        # Update multiplier label with loaded value
        self._update_multiplier_label(self.font_size_multiplier_var.get())
        
        font_size_spinbox.pack(side=tk.LEFT, padx=10)
        # Also bind to save on manual entry
        font_size_spinbox.bind('<Return>', lambda e: self._save_rendering_settings())
        font_size_spinbox.bind('<FocusOut>', lambda e: self._save_rendering_settings())
        
        tk.Label(font_frame, text="(0 = Auto)", font=('Arial', 9), fg='gray').pack(side=tk.LEFT)
        
        # Font style selection
        font_style_frame = tk.Frame(render_frame)
        font_style_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(font_style_frame, text="Font Style:", width=20, anchor='w').pack(side=tk.LEFT)
        
        # Font style will be set from loaded config in _load_rendering_settings
        self.font_combo = ttk.Combobox(
            font_style_frame,
            textvariable=self.font_style_var,
            values=self._get_available_fonts(),
            width=30,
            state='readonly'
        )
        self.font_combo.pack(side=tk.LEFT, padx=10)
        self.font_combo.bind('<<ComboboxSelected>>', self._on_font_selected)

        # Disable mousewheel scrolling completely
        self.font_combo.bind("<MouseWheel>", lambda e: "break")
        self.font_combo.bind("<Button-4>", lambda e: "break")  # Linux scroll up
        self.font_combo.bind("<Button-5>", lambda e: "break")  # Linux scroll down
        
        # Font color selection
        color_frame = tk.Frame(render_frame)
        color_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(color_frame, text="Font Color:", width=20, anchor='w').pack(side=tk.LEFT)
        
        # Color button and preview
        color_button_frame = tk.Frame(color_frame)
        color_button_frame.pack(side=tk.LEFT, padx=10)
        
        # Color preview
        self.color_preview = tk.Canvas(color_button_frame, width=40, height=30, 
                                     highlightthickness=1, highlightbackground="gray")
        self.color_preview.pack(side=tk.LEFT, padx=(0, 10))
        
        # RGB display label
        r, g, b = self.text_color_r.get(), self.text_color_g.get(), self.text_color_b.get()
        self.rgb_label = tk.Label(color_button_frame, text=f"RGB({r},{g},{b})", width=12)
        self.rgb_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Color picker button
        def pick_font_color():
            from tkinter import colorchooser
            # Get current color
            current_color = (self.text_color_r.get(), self.text_color_g.get(), self.text_color_b.get())
            color_hex = f'#{current_color[0]:02x}{current_color[1]:02x}{current_color[2]:02x}'
            
            # Open color dialog
            color = colorchooser.askcolor(initialcolor=color_hex, parent=self.dialog, 
                                         title="Choose Font Color")
            if color[0]:  # If a color was chosen (not cancelled)
                # Update RGB values
                self.text_color_r.set(int(color[0][0]))
                self.text_color_g.set(int(color[0][1]))
                self.text_color_b.set(int(color[0][2]))
                # Update display
                self.rgb_label.config(text=f"RGB({int(color[0][0])},{int(color[0][1])},{int(color[0][2])})")
                self._update_color_preview(None)
        
        tb.Button(
            color_button_frame,
            text="Choose Color",
            command=pick_font_color,
            bootstyle="info"
        ).pack(side=tk.LEFT)
        
        self._update_color_preview(None)  # Initialize with loaded colors
        
        # Shadow settings frame
        shadow_frame = tk.LabelFrame(render_frame, text="Text Shadow", padx=10, pady=5)
        shadow_frame.pack(fill=tk.X, pady=10)
        
        # Shadow enabled checkbox
        tb.Checkbutton(
            shadow_frame,
            text="Enable Shadow",
            variable=self.shadow_enabled_var,
            bootstyle="round-toggle",
            command=self._toggle_shadow_controls
        ).pack(anchor='w')
        
        # Shadow controls container
        self.shadow_controls = tk.Frame(shadow_frame)
        self.shadow_controls.pack(fill=tk.X, pady=5)
        
        # Shadow color
        shadow_color_frame = tk.Frame(self.shadow_controls)
        shadow_color_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(shadow_color_frame, text="Shadow Color:", width=15, anchor='w').pack(side=tk.LEFT)
        
        # Shadow color button and preview
        shadow_button_frame = tk.Frame(shadow_color_frame)
        shadow_button_frame.pack(side=tk.LEFT, padx=10)
        
        # Shadow color preview
        self.shadow_preview = tk.Canvas(shadow_button_frame, width=30, height=25, 
                                      highlightthickness=1, highlightbackground="gray")
        self.shadow_preview.pack(side=tk.LEFT, padx=(0, 10))
        
        # Shadow RGB display label
        sr, sg, sb = self.shadow_color_r.get(), self.shadow_color_g.get(), self.shadow_color_b.get()
        self.shadow_rgb_label = tk.Label(shadow_button_frame, text=f"RGB({sr},{sg},{sb})", width=15)
        self.shadow_rgb_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Shadow color picker button
        def pick_shadow_color():
            from tkinter import colorchooser
            # Get current color
            current_color = (self.shadow_color_r.get(), self.shadow_color_g.get(), self.shadow_color_b.get())
            color_hex = f'#{current_color[0]:02x}{current_color[1]:02x}{current_color[2]:02x}'
            
            # Open color dialog
            color = colorchooser.askcolor(initialcolor=color_hex, parent=self.dialog, 
                                         title="Choose Shadow Color")
            if color[0]:  # If a color was chosen (not cancelled)
                # Update RGB values
                self.shadow_color_r.set(int(color[0][0]))
                self.shadow_color_g.set(int(color[0][1]))
                self.shadow_color_b.set(int(color[0][2]))
                # Update display
                self.shadow_rgb_label.config(text=f"RGB({int(color[0][0])},{int(color[0][1])},{int(color[0][2])})")
                self._update_shadow_preview(None)
        
        tb.Button(
            shadow_button_frame,
            text="Choose Color",
            command=pick_shadow_color,
            bootstyle="info",
            width=12
        ).pack(side=tk.LEFT)
        
        self._update_shadow_preview(None)  # Initialize with loaded colors
        
        # Shadow offset
        offset_frame = tk.Frame(self.shadow_controls)
        offset_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(offset_frame, text="Shadow Offset:", width=15, anchor='w').pack(side=tk.LEFT)
        
        # X offset
        x_frame = tk.Frame(offset_frame)
        x_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(x_frame, text="X:", width=2).pack(side=tk.LEFT)
        x_spinbox = tb.Spinbox(x_frame, from_=-10, to=10, textvariable=self.shadow_offset_x_var,
                  width=5, command=self._save_rendering_settings)
        x_spinbox.pack(side=tk.LEFT)
        x_spinbox.bind('<Return>', lambda e: self._save_rendering_settings())
        x_spinbox.bind('<FocusOut>', lambda e: self._save_rendering_settings())
        self._disable_spinbox_mousewheel(x_spinbox)
        
        # Y offset
        y_frame = tk.Frame(offset_frame)
        y_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(y_frame, text="Y:", width=2).pack(side=tk.LEFT)
        y_spinbox = tb.Spinbox(y_frame, from_=-10, to=10, textvariable=self.shadow_offset_y_var,
                  width=5, command=self._save_rendering_settings)
        y_spinbox.pack(side=tk.LEFT)
        y_spinbox.bind('<Return>', lambda e: self._save_rendering_settings())
        y_spinbox.bind('<FocusOut>', lambda e: self._save_rendering_settings())
        self._disable_spinbox_mousewheel(y_spinbox)
        
        # Shadow blur
        blur_frame = tk.Frame(self.shadow_controls)
        blur_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(blur_frame, text="Shadow Blur:", width=15, anchor='w').pack(side=tk.LEFT)
        tk.Scale(blur_frame, from_=0, to=10, orient=tk.HORIZONTAL, variable=self.shadow_blur_var,
                length=150, command=lambda v: self._save_rendering_settings()).pack(side=tk.LEFT, padx=10)
        tk.Label(blur_frame, text="(0=sharp, 10=blurry)", font=('Arial', 9), fg='gray').pack(side=tk.LEFT)
        
        # Initially disable shadow controls
        self._toggle_shadow_controls()
        
        # Output settings
        output_frame = tk.Frame(settings_frame)
        output_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.create_subfolder_var = tk.BooleanVar(value=self.main_gui.config.get('manga_create_subfolder', True))
        tb.Checkbutton(
            output_frame,
            text="Create 'translated' subfolder for output",
            variable=self.create_subfolder_var,
            bootstyle="round-toggle",
            command=self._save_rendering_settings
        ).pack(side=tk.LEFT)
        
        # Control buttons
        control_frame = tk.Frame(self.parent_frame)
        control_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Check if ready based on selected provider
        has_api_key = bool(self.main_gui.api_key_entry.get().strip()) if hasattr(self.main_gui, 'api_key_entry') else False
        provider = self.ocr_provider_var.get()

        # Determine readiness based on provider
        if provider == 'google':
            has_vision = os.path.exists(self.main_gui.config.get('google_vision_credentials', ''))
            is_ready = has_api_key and has_vision
        elif provider == 'azure':
            has_azure = bool(self.main_gui.config.get('azure_vision_key', ''))
            is_ready = has_api_key and has_azure
        elif provider == 'custom-api':
            is_ready = has_api_key  # Only needs API key
        else:
            # Local providers (manga-ocr, easyocr, etc.) only need API key for translation
            is_ready = has_api_key

        self.start_button = tb.Button(
            control_frame,
            text="Start Translation",
            command=self._start_translation,
            bootstyle="success",
            state=tk.NORMAL if is_ready else tk.DISABLED
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))

        # Add tooltip to show why button is disabled
        if not is_ready:
            reasons = []
            if not has_api_key:
                reasons.append("API key not configured")
            if provider == 'google' and not os.path.exists(self.main_gui.config.get('google_vision_credentials', '')):
                reasons.append("Google Vision credentials not set")
            elif provider == 'azure' and not self.main_gui.config.get('azure_vision_key', ''):
                reasons.append("Azure credentials not configured")
            tooltip_text = "Cannot start: " + ", ".join(reasons)
            # You can add a tooltip here if you have a tooltip library
        
        self.stop_button = tb.Button(
            control_frame,
            text="Stop",
            command=self._stop_translation,
            bootstyle="danger",
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT)
        
        # Progress frame
        progress_frame = tk.LabelFrame(
            self.parent_frame,
            text="Progress",
            font=('Arial', 12, 'bold'),
            padx=15,
            pady=10
        )
        progress_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Overall progress
        self.progress_label = tk.Label(
            progress_frame,
            text="Ready to start",
            font=('Arial', 11),
            fg='white'  # White text for dark mode
        )
        self.progress_label.pack(anchor=tk.W)
        
        # Create and configure progress bar with custom styling
        try:
            # Create a custom style for the progress bar
            style = ttk.Style()
            
            # Configure the progress bar for dark mode with white fill
            style.configure("MangaProgress.Horizontal.TProgressbar",
                          troughcolor='#2d3748',      # Dark gray background
                          background='white',          # White progress fill color
                          lightcolor='#f7fafc',       # Light white highlight
                          darkcolor='#e2e8f0',        # Light gray shadow
                          bordercolor='#4a5568',      # Border color
                          relief='flat')
            
            # Apply the style to our progress bar
            self.progress_bar = ttk.Progressbar(
                progress_frame,
                mode='determinate',
                length=400,
                style="MangaProgress.Horizontal.TProgressbar"
            )
        except Exception:
            # Fallback to default progress bar if styling fails
            self.progress_bar = ttk.Progressbar(
                progress_frame,
                mode='determinate',
                length=400
            )
        
        self.progress_bar.pack(fill=tk.X, pady=(5, 10))
        
        # Current file status
        self.current_file_label = tk.Label(
            progress_frame,
            text="",
            font=('Arial', 10),
            fg='lightgray'  # Light gray for dark mode visibility
        )
        self.current_file_label.pack(anchor=tk.W)
        
        # Log frame
        log_frame = tk.LabelFrame(
            self.parent_frame,
            text="Translation Log",
            font=('Arial', 12, 'bold'),
            padx=15,
            pady=10
        )
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
        
        # Log text with scrollbar
        log_scroll_frame = tk.Frame(log_frame)
        log_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        log_scrollbar = tk.Scrollbar(log_scroll_frame)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(
            log_scroll_frame,
            height=28,
            wrap=tk.WORD,
            yscrollcommand=log_scrollbar.set
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.config(command=self.log_text.yview)

        # Make log read-only (programmatic inserts will toggle state)
        self.log_text.config(state='disabled', cursor='arrow')
        try:
            # Prevent focus via Tab key
            self.log_text.configure(takefocus=0)
        except Exception:
            pass
        
        # Configure text tags for colored output
        self.log_text.tag_config('info', foreground='white')
        self.log_text.tag_config('success', foreground='green')
        self.log_text.tag_config('warning', foreground='orange')
        self.log_text.tag_config('error', foreground='red')

    def _show_help_dialog(self, title: str, message: str):
        """Show a help dialog with the given title and message"""
        # Create a simple dialog
        help_dialog = tk.Toplevel(self.dialog)
        help_dialog.title(title)
        help_dialog.geometry("500x400")
        help_dialog.transient(self.dialog)
        help_dialog.grab_set()
        
        # Center the dialog
        help_dialog.update_idletasks()
        x = (help_dialog.winfo_screenwidth() // 2) - (help_dialog.winfo_width() // 2)
        y = (help_dialog.winfo_screenheight() // 2) - (help_dialog.winfo_height() // 2)
        help_dialog.geometry(f"+{x}+{y}")
        
        # Main frame with padding
        main_frame = tk.Frame(help_dialog, padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)
        
        # Icon and title
        title_frame = tk.Frame(main_frame)
        title_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(
            title_frame,
            text="ℹ️",
            font=('Arial', 20)
        ).pack(side='left', padx=(0, 10))
        
        tk.Label(
            title_frame,
            text=title,
            font=('Arial', 12, 'bold')
        ).pack(side='left')
        
        # Help text in a scrollable frame
        text_frame = tk.Frame(main_frame)
        text_frame.pack(fill='both', expand=True)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side='right', fill='y')
        
        # Text widget
        text_widget = tk.Text(
            text_frame,
            wrap='word',
            yscrollcommand=scrollbar.set,
            font=('Arial', 10),
            padx=10,
            pady=10
        )
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=text_widget.yview)
        
        # Insert the help text
        text_widget.insert('1.0', message)
        text_widget.config(state='disabled')  # Make read-only
        
        # Close button
        tb.Button(
            main_frame,
            text="Close",
            command=help_dialog.destroy,
            bootstyle="secondary"
        ).pack(pady=(10, 0))
        
        # Bind Escape key to close
        help_dialog.bind('<Escape>', lambda e: help_dialog.destroy())

    def _on_visual_context_toggle(self):
        """Handle visual context toggle"""
        enabled = self.visual_context_enabled_var.get()
        self.main_gui.config['manga_visual_context_enabled'] = enabled
        
        # Update translator if it exists
        if self.translator:
            self.translator.visual_context_enabled = enabled
        
        # Save config
        if hasattr(self.main_gui, 'save_config'):
            self.main_gui.save_config(show_message=False)
        
        # Log the change
        if enabled:
            self._log("📷 Visual context ENABLED - Images will be sent to API", "info")
            self._log("   Make sure you're using a vision-capable model", "warning")
        else:
            self._log("📝 Visual context DISABLED - Text-only mode", "info")
            self._log("   Compatible with non-vision models (Claude, GPT-3.5, etc.)", "success")
 
    def _open_advanced_settings(self):
        """Open the manga advanced settings dialog"""
        try:
            def on_settings_saved(settings):
                """Callback when settings are saved"""
                # Update config with new settings
                self.main_gui.config['manga_settings'] = settings
                
                # Reload settings in translator if it exists
                if self.translator:
                    self._log("📋 Reloading settings in translator...", "info")
                    # The translator will pick up new settings on next operation
                
                self._log("✅ Advanced settings saved and applied", "success")
            
            # Open the settings dialog
            MangaSettingsDialog(
                parent=self.dialog,
                main_gui=self.main_gui,
                config=self.main_gui.config,
                callback=on_settings_saved
            )
            
        except Exception as e:
            self._log(f"❌ Error opening settings dialog: {str(e)}", "error")
            messagebox.showerror("Error", f"Failed to open settings dialog:\n{str(e)}")
        
    def _toggle_font_size_mode(self):
        """Toggle between fixed size and multiplier modes"""
        mode = self.font_size_mode_var.get()
        
        # Check if frames exist before trying to pack/unpack them
        if hasattr(self, 'fixed_size_frame') and hasattr(self, 'multiplier_frame'):
            if mode == "fixed":
                self.fixed_size_frame.pack(fill=tk.X, pady=(5, 0))
                self.multiplier_frame.pack_forget()
                # Hide constraint frame if it exists
                if hasattr(self, 'constraint_frame'):
                    self.constraint_frame.pack_forget()
            else:
                self.fixed_size_frame.pack_forget()
                self.multiplier_frame.pack(fill=tk.X, pady=(5, 0))
                # Show constraint frame if it exists
                if hasattr(self, 'constraint_frame'):
                    self.constraint_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Only save if we're not initializing
        if not hasattr(self, '_initializing') or not self._initializing:
            self._save_rendering_settings()

    def _update_multiplier_label(self, value):
        """Update multiplier label"""
        self.multiplier_label.config(text=f"{float(value):.1f}x")
        # Auto-save on change
        self._save_rendering_settings()
    
    def _update_color_preview(self, event):
        """Update the font color preview"""
        r = self.text_color_r.get()
        g = self.text_color_g.get()
        b = self.text_color_b.get()
        color = f'#{r:02x}{g:02x}{b:02x}'
        self.color_preview.configure(bg=color)
        # Auto-save on change
        if event is not None:  # Only save on user interaction, not initial load
            self._save_rendering_settings()
    
    def _update_shadow_preview(self, event):
        """Update the shadow color preview"""
        r = self.shadow_color_r.get()
        g = self.shadow_color_g.get()
        b = self.shadow_color_b.get()
        color = f'#{r:02x}{g:02x}{b:02x}'
        self.shadow_preview.configure(bg=color)
        # Auto-save on change
        if event is not None:  # Only save on user interaction, not initial load
            self._save_rendering_settings()
    
    def _toggle_shadow_controls(self):
        """Enable/disable shadow controls based on checkbox"""
        if self.shadow_enabled_var.get():
            for widget in self.shadow_controls.winfo_children():
                self._enable_widget_tree(widget)
        else:
            for widget in self.shadow_controls.winfo_children():
                self._disable_widget_tree(widget)
        # Auto-save on change (but not during initialization)
        if not getattr(self, '_initializing', False):
            self._save_rendering_settings()
    
    def _enable_widget_tree(self, widget):
        """Recursively enable a widget and its children"""
        try:
            widget.configure(state=tk.NORMAL)
        except:
            pass
        for child in widget.winfo_children():
            self._enable_widget_tree(child)
    
    def _disable_widget_tree(self, widget):
        """Recursively disable a widget and its children"""
        try:
            widget.configure(state=tk.DISABLED)
        except:
            pass
        for child in widget.winfo_children():
            self._disable_widget_tree(child)
        
    def _load_rendering_settings(self):
        """Load text rendering settings from config"""
        config = self.main_gui.config
        
        # Get inpainting settings from the nested location
        manga_settings = config.get('manga_settings', {})
        inpaint_settings = manga_settings.get('inpainting', {})
        
        # Load inpaint method from the correct location
        self.inpaint_method_var = tk.StringVar(value=inpaint_settings.get('method', 'local'))
        self.local_model_type_var = tk.StringVar(value=inpaint_settings.get('local_method', 'anime_onnx'))
        
        # Load model paths
        for model_type in  ['aot', 'aot_onnx', 'lama', 'lama_onnx', 'anime', 'anime_onnx', 'mat', 'ollama', 'sd_local']:
            path = inpaint_settings.get(f'{model_type}_model_path', '')
            if model_type == self.local_model_type_var.get():
                self.local_model_path_var = tk.StringVar(value=path)
        
        # Initialize with defaults
        self.bg_opacity_var = tk.IntVar(value=config.get('manga_bg_opacity', 130))
        # Free-text-only background opacity (default off)
        self.free_text_only_bg_opacity_var = tk.BooleanVar(value=config.get('manga_free_text_only_bg_opacity', False))
        # Persist on change like other controls
        self.free_text_only_bg_opacity_var.trace('w', lambda *args: self._save_rendering_settings())
        self.bg_opacity_var.trace('w', lambda *args: self._save_rendering_settings())  # Add trace right after creation
        
        self.bg_style_var = tk.StringVar(value=config.get('manga_bg_style', 'circle'))
        self.bg_style_var.trace('w', lambda *args: self._save_rendering_settings())
        
        self.bg_reduction_var = tk.DoubleVar(value=config.get('manga_bg_reduction', 1.0))
        self.bg_reduction_var.trace('w', lambda *args: self._save_rendering_settings())
        
        self.font_size_var = tk.IntVar(value=config.get('manga_font_size', 0))
        self.font_size_var.trace('w', lambda *args: self._save_rendering_settings())
        
        self.selected_font_path = config.get('manga_font_path', None)
        self.skip_inpainting_var = tk.BooleanVar(value=config.get('manga_skip_inpainting', True))
        self.inpaint_quality_var = tk.StringVar(value=config.get('manga_inpaint_quality', 'high'))
        self.inpaint_dilation_var = tk.IntVar(value=config.get('manga_inpaint_dilation', 15))
        self.inpaint_passes_var = tk.IntVar(value=config.get('manga_inpaint_passes', 2))
        
        self.font_size_mode_var = tk.StringVar(value=config.get('manga_font_size_mode', 'fixed'))
        self.font_size_mode_var.trace('w', lambda *args: self._save_rendering_settings())
        
        self.font_size_multiplier_var = tk.DoubleVar(value=config.get('manga_font_size_multiplier', 1.0))
        self.font_size_multiplier_var.trace('w', lambda *args: self._save_rendering_settings())
        
        self.force_caps_lock_var = tk.BooleanVar(value=config.get('manga_force_caps_lock', False))
        self.force_caps_lock_var.trace('w', lambda *args: self._save_rendering_settings())
        
        self.constrain_to_bubble_var = tk.BooleanVar(value=config.get('manga_constrain_to_bubble', True))
        self.constrain_to_bubble_var.trace('w', lambda *args: self._save_rendering_settings())
        
        self.min_readable_size_var = tk.IntVar(value=config.get('manga_min_readable_size', 16))
        self.min_readable_size_var.trace('w', lambda *args: self._save_rendering_settings()) 
        
        self.max_font_size_var = tk.IntVar(value=config.get('manga_max_font_size', 24))
        self.max_font_size_var.trace('w', lambda *args: self._save_rendering_settings())  
        
        self.strict_text_wrapping_var = tk.BooleanVar(value=config.get('manga_strict_text_wrapping', False))
        self.strict_text_wrapping_var.trace('w', lambda *args: self._save_rendering_settings())
        
        # Font color settings
        manga_text_color = config.get('manga_text_color', [102, 0, 0])
        self.text_color_r = tk.IntVar(value=manga_text_color[0])
        self.text_color_r.trace('w', lambda *args: self._save_rendering_settings())
        
        self.text_color_g = tk.IntVar(value=manga_text_color[1])
        self.text_color_g.trace('w', lambda *args: self._save_rendering_settings())
        
        self.text_color_b = tk.IntVar(value=manga_text_color[2])
        self.text_color_b.trace('w', lambda *args: self._save_rendering_settings())
        
        # Shadow settings
        self.shadow_enabled_var = tk.BooleanVar(value=config.get('manga_shadow_enabled', True))
        self.shadow_enabled_var.trace('w', lambda *args: self._save_rendering_settings())
        
        manga_shadow_color = config.get('manga_shadow_color', [204, 128, 128])
        self.shadow_color_r = tk.IntVar(value=manga_shadow_color[0])
        self.shadow_color_r.trace('w', lambda *args: self._save_rendering_settings())
        
        self.shadow_color_g = tk.IntVar(value=manga_shadow_color[1])
        self.shadow_color_g.trace('w', lambda *args: self._save_rendering_settings())
        
        self.shadow_color_b = tk.IntVar(value=manga_shadow_color[2])
        self.shadow_color_b.trace('w', lambda *args: self._save_rendering_settings())
        
        self.shadow_offset_x_var = tk.IntVar(value=config.get('manga_shadow_offset_x', 2))
        self.shadow_offset_x_var.trace('w', lambda *args: self._save_rendering_settings())
        
        self.shadow_offset_y_var = tk.IntVar(value=config.get('manga_shadow_offset_y', 2))
        self.shadow_offset_y_var.trace('w', lambda *args: self._save_rendering_settings())
        
        self.shadow_blur_var = tk.IntVar(value=config.get('manga_shadow_blur', 0))
        self.shadow_blur_var.trace('w', lambda *args: self._save_rendering_settings())
        
        # Initialize font_style_var with saved value or default
        saved_font_style = config.get('manga_font_style', 'Default')
        self.font_style_var = tk.StringVar(value=saved_font_style)
        self.font_style_var.trace('w', lambda *args: self._save_rendering_settings())
        
        # Full page context settings
        self.full_page_context_var = tk.BooleanVar(value=config.get('manga_full_page_context', False))
        self.full_page_context_var.trace('w', lambda *args: self._save_rendering_settings())
        
        self.full_page_context_prompt = config.get('manga_full_page_context_prompt', 
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
 
        # Load OCR prompt
        self.ocr_prompt = config.get('manga_ocr_prompt', 
            "YOU ARE AN OCR SYSTEM. YOUR ONLY JOB IS TEXT EXTRACTION.\n\n"
            "CRITICAL RULES:\n"
            "1. DO NOT TRANSLATE ANYTHING\n"
            "2. DO NOT MODIFY THE TEXT\n"
            "3. DO NOT EXPLAIN OR COMMENT\n"
            "4. ONLY OUTPUT THE EXACT TEXT YOU SEE\n"
            "5. PRESERVE NATURAL TEXT FLOW - DO NOT ADD UNNECESSARY LINE BREAKS\n\n"
            "If you see Korean text, output it in Korean.\n"
            "If you see Japanese text, output it in Japanese.\n"
            "If you see Chinese text, output it in Chinese.\n"
            "If you see English text, output it in English.\n\n"
            "IMPORTANT: Only use line breaks where they naturally occur in the original text "
            "(e.g., between dialogue lines or paragraphs). Do not break text mid-sentence or "
            "between every word/character.\n\n"
            "For vertical text common in manga/comics, transcribe it as a continuous line unless "
            "there are clear visual breaks.\n\n"
            "NEVER translate. ONLY extract exactly what is written.\n"
            "Output ONLY the raw text, nothing else."
        ) 
        # Visual context setting
        self.visual_context_enabled_var = tk.BooleanVar(
            value=self.main_gui.config.get('manga_visual_context_enabled', True)
        )
        self.visual_context_enabled_var.trace('w', lambda *args: self._save_rendering_settings())
        self.qwen2vl_model_size = config.get('qwen2vl_model_size', '1')  # Default to '1' (2B)
        
        # Initialize RapidOCR settings
        self.rapidocr_use_recognition_var = tk.BooleanVar(
            value=self.main_gui.config.get('rapidocr_use_recognition', True)
        )
        self.rapidocr_language_var = tk.StringVar(
            value=self.main_gui.config.get('rapidocr_language', 'auto')
        )
        self.rapidocr_detection_mode_var = tk.StringVar(
            value=self.main_gui.config.get('rapidocr_detection_mode', 'document')
        )

        # Output settings
        self.create_subfolder_var = tk.BooleanVar(value=config.get('manga_create_subfolder', True))
        self.create_subfolder_var.trace('w', lambda *args: self._save_rendering_settings())
    
    def _save_rendering_settings(self):
        """Save rendering settings with validation"""
        # Don't save during initialization
        if hasattr(self, '_initializing') and self._initializing:
            return
        
        # Validate that variables exist and have valid values before saving
        try:
            # Ensure manga_settings structure exists
            if 'manga_settings' not in self.main_gui.config:
                self.main_gui.config['manga_settings'] = {}
            if 'inpainting' not in self.main_gui.config['manga_settings']:
                self.main_gui.config['manga_settings']['inpainting'] = {}
            
            # Save to nested location
            inpaint = self.main_gui.config['manga_settings']['inpainting']
            if hasattr(self, 'inpaint_method_var'):
                inpaint['method'] = self.inpaint_method_var.get()
            if hasattr(self, 'local_model_type_var'):
                inpaint['local_method'] = self.local_model_type_var.get()
                model_type = self.local_model_type_var.get()
                if hasattr(self, 'local_model_path_var'):
                    inpaint[f'{model_type}_model_path'] = self.local_model_path_var.get()
            
            # Add new inpainting settings
            if hasattr(self, 'inpaint_method_var'):
                self.main_gui.config['manga_inpaint_method'] = self.inpaint_method_var.get()
            if hasattr(self, 'local_model_type_var'):
                self.main_gui.config['manga_local_inpaint_model'] = self.local_model_type_var.get()
            
            # Save model paths for each type
            for model_type in  ['aot', 'lama', 'lama_onnx', 'anime', 'mat', 'ollama', 'sd_local']:
                if hasattr(self, 'local_model_type_var'):
                    if model_type == self.local_model_type_var.get():
                        if hasattr(self, 'local_model_path_var'):
                            path = self.local_model_path_var.get()
                            if path:
                                self.main_gui.config[f'manga_{model_type}_model_path'] = path
            
            # Save all other settings with validation
            if hasattr(self, 'bg_opacity_var'):
                self.main_gui.config['manga_bg_opacity'] = self.bg_opacity_var.get()
            if hasattr(self, 'bg_style_var'):
                self.main_gui.config['manga_bg_style'] = self.bg_style_var.get()
            if hasattr(self, 'bg_reduction_var'):
                self.main_gui.config['manga_bg_reduction'] = self.bg_reduction_var.get()
            
            # Save free-text-only background opacity toggle
            if hasattr(self, 'free_text_only_bg_opacity_var'):
                try:
                    self.main_gui.config['manga_free_text_only_bg_opacity'] = bool(self.free_text_only_bg_opacity_var.get())
                except tk.TclError:
                    pass
            
            # CRITICAL: Font size settings - validate before saving
            if hasattr(self, 'font_size_var'):
                try:
                    value = self.font_size_var.get()
                    self.main_gui.config['manga_font_size'] = value
                except tk.TclError:
                    pass  # Skip if variable is in invalid state
            
            if hasattr(self, 'min_readable_size_var'):
                try:
                    value = self.min_readable_size_var.get()
                    # Validate the value is reasonable
                    if 0 <= value <= 100:
                        self.main_gui.config['manga_min_readable_size'] = value
                except (tk.TclError, ValueError):
                    pass  # Skip if variable is in invalid state
            
            if hasattr(self, 'max_font_size_var'):
                try:
                    value = self.max_font_size_var.get()
                    # Validate the value is reasonable
                    if 0 <= value <= 200:
                        self.main_gui.config['manga_max_font_size'] = value
                except (tk.TclError, ValueError):
                    pass  # Skip if variable is in invalid state
            
            # Continue with other settings
            self.main_gui.config['manga_font_path'] = self.selected_font_path
            
            if hasattr(self, 'skip_inpainting_var'):
                self.main_gui.config['manga_skip_inpainting'] = self.skip_inpainting_var.get()
            if hasattr(self, 'inpaint_quality_var'):
                self.main_gui.config['manga_inpaint_quality'] = self.inpaint_quality_var.get()
            if hasattr(self, 'inpaint_dilation_var'):
                self.main_gui.config['manga_inpaint_dilation'] = self.inpaint_dilation_var.get()
            if hasattr(self, 'inpaint_passes_var'):
                self.main_gui.config['manga_inpaint_passes'] = self.inpaint_passes_var.get()
            if hasattr(self, 'font_size_mode_var'):
                self.main_gui.config['manga_font_size_mode'] = self.font_size_mode_var.get()
            if hasattr(self, 'font_size_multiplier_var'):
                self.main_gui.config['manga_font_size_multiplier'] = self.font_size_multiplier_var.get()
            if hasattr(self, 'font_style_var'):
                self.main_gui.config['manga_font_style'] = self.font_style_var.get()
            if hasattr(self, 'constrain_to_bubble_var'):
                self.main_gui.config['manga_constrain_to_bubble'] = self.constrain_to_bubble_var.get()
            if hasattr(self, 'strict_text_wrapping_var'):
                self.main_gui.config['manga_strict_text_wrapping'] = self.strict_text_wrapping_var.get()
            if hasattr(self, 'force_caps_lock_var'):
                self.main_gui.config['manga_force_caps_lock'] = self.force_caps_lock_var.get()
            
            # Save font color as list
            if hasattr(self, 'text_color_r') and hasattr(self, 'text_color_g') and hasattr(self, 'text_color_b'):
                self.main_gui.config['manga_text_color'] = [
                    self.text_color_r.get(),
                    self.text_color_g.get(),
                    self.text_color_b.get()
                ]
            
            # Save shadow settings
            if hasattr(self, 'shadow_enabled_var'):
                self.main_gui.config['manga_shadow_enabled'] = self.shadow_enabled_var.get()
            if hasattr(self, 'shadow_color_r') and hasattr(self, 'shadow_color_g') and hasattr(self, 'shadow_color_b'):
                self.main_gui.config['manga_shadow_color'] = [
                    self.shadow_color_r.get(),
                    self.shadow_color_g.get(),
                    self.shadow_color_b.get()
                ]
            if hasattr(self, 'shadow_offset_x_var'):
                self.main_gui.config['manga_shadow_offset_x'] = self.shadow_offset_x_var.get()
            if hasattr(self, 'shadow_offset_y_var'):
                self.main_gui.config['manga_shadow_offset_y'] = self.shadow_offset_y_var.get()
            if hasattr(self, 'shadow_blur_var'):
                self.main_gui.config['manga_shadow_blur'] = self.shadow_blur_var.get()
            
            # Save output settings
            if hasattr(self, 'create_subfolder_var'):
                self.main_gui.config['manga_create_subfolder'] = self.create_subfolder_var.get()
            
            # Save full page context settings
            if hasattr(self, 'full_page_context_var'):
                self.main_gui.config['manga_full_page_context'] = self.full_page_context_var.get()
            if hasattr(self, 'full_page_context_prompt'):
                self.main_gui.config['manga_full_page_context_prompt'] = self.full_page_context_prompt
            
            # OCR prompt
            if hasattr(self, 'ocr_prompt'):
                self.main_gui.config['manga_ocr_prompt'] = self.ocr_prompt
             
            # Qwen and custom models             
            if hasattr(self, 'qwen2vl_model_size'):
                self.main_gui.config['qwen2vl_model_size'] = self.qwen2vl_model_size

            # RapidOCR specific settings
            if hasattr(self, 'rapidocr_use_recognition_var'):
                self.main_gui.config['rapidocr_use_recognition'] = self.rapidocr_use_recognition_var.get()
            if hasattr(self, 'rapidocr_detection_mode_var'):
                self.main_gui.config['rapidocr_detection_mode'] = self.rapidocr_detection_mode_var.get()
            if hasattr(self, 'rapidocr_language_var'):
                self.main_gui.config['rapidocr_language'] = self.rapidocr_language_var.get()

            # Call main GUI's save_config to persist to file
            if hasattr(self.main_gui, 'save_config'):
                self.main_gui.save_config(show_message=False)
                
        except Exception as e:
            # Log error but don't crash
            print(f"Error saving manga settings: {e}")
    
    def _on_context_toggle(self):
        """Handle full page context toggle"""
        enabled = self.full_page_context_var.get()
        self._save_rendering_settings()
    
    def _edit_context_prompt(self):
        """Open dialog to edit full page context prompt and OCR prompt"""
        # Store parent canvas for scroll restoration
        parent_canvas = self.canvas
        
        # Use WindowManager to create scrollable dialog
        dialog, scrollable_frame, canvas = self.main_gui.wm.setup_scrollable(
            self.dialog,  # parent window
            "Edit Prompts",
            width=700,
            height=600,
            max_width_ratio=0.7,
            max_height_ratio=0.85
        )
        
        # Instructions
        instructions = tk.Label(
            scrollable_frame,
            text="Edit the prompt used for full page context translation.\n"
                 "This will be appended to the main translation system prompt.",
            font=('Arial', 10),
            justify=tk.LEFT
        )
        instructions.pack(padx=20, pady=(20, 10))
        
        # Full Page Context label
        context_label = tk.Label(
            scrollable_frame,
            text="Full Page Context Prompt:",
            font=('Arial', 10, 'bold')
        )
        context_label.pack(padx=20, pady=(10, 5), anchor='w')
        
        # Text editor for context
        text_editor = self.main_gui.ui.setup_scrollable_text(
            scrollable_frame,
            wrap=tk.WORD,
            height=10
        )
        text_editor.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Insert current prompt
        text_editor.insert(1.0, self.full_page_context_prompt)
        
        # OCR Prompt label
        ocr_label = tk.Label(
            scrollable_frame,
            text="OCR System Prompt:",
            font=('Arial', 10, 'bold')
        )
        ocr_label.pack(padx=20, pady=(10, 5), anchor='w')
        
        # Text editor for OCR
        ocr_editor = self.main_gui.ui.setup_scrollable_text(
            scrollable_frame,
            wrap=tk.WORD,
            height=10
        )
        ocr_editor.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
        # Get current OCR prompt
        if hasattr(self, 'ocr_prompt'):
            ocr_editor.insert(1.0, self.ocr_prompt)
        else:
            ocr_editor.insert(1.0, "")
        
        # Button frame
        button_frame = tk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        def close_dialog():
            """Properly close dialog and restore parent scrolling"""
            try:
                # Clean up this dialog's scrolling
                if hasattr(dialog, '_cleanup_scrolling') and callable(dialog._cleanup_scrolling):
                    dialog._cleanup_scrolling()
            except:
                pass
            
            # Destroy the dialog
            dialog.destroy()
        
        def save_prompt():
            self.full_page_context_prompt = text_editor.get(1.0, tk.END).strip()
            self.ocr_prompt = ocr_editor.get(1.0, tk.END).strip()  # Save to self.ocr_prompt
            
            # Save to config
            self.main_gui.config['manga_full_page_context_prompt'] = self.full_page_context_prompt
            self.main_gui.config['manga_ocr_prompt'] = self.ocr_prompt
            
            self._save_rendering_settings()
            self._log("✅ Updated prompts", "success")
            close_dialog()
        
        def reset_prompt():
            default_prompt = (
                "You will receive multiple text segments from a manga page. "
                "Translate each segment considering the context of all segments together. "
                "Maintain consistency in character names, tone, and style across all translations.\n\n"
                "IMPORTANT: Return your response as a JSON object where each key is the EXACT original text "
                "(without the [0], [1] index prefixes) and each value is the translation. Example:\n"
                '{\n'
                '  こんにちは: Hello,\n'
                '  ありがとう: Thank you\n'
                '}\n\n'
                'Do NOT include the [0], [1], etc. prefixes in the JSON keys.'
            )
            text_editor.delete(1.0, tk.END)
            text_editor.insert(1.0, default_prompt)
            
            default_ocr = (
                "YOU ARE AN OCR SYSTEM. YOUR ONLY JOB IS TEXT EXTRACTION.\n\n"
                "CRITICAL RULES:\n"
                "1. DO NOT TRANSLATE ANYTHING\n"
                "2. DO NOT MODIFY THE TEXT\n"
                "3. DO NOT EXPLAIN OR COMMENT\n"
                "4. ONLY OUTPUT THE EXACT TEXT YOU SEE\n"
                "5. PRESERVE NATURAL TEXT FLOW - DO NOT ADD UNNECESSARY LINE BREAKS\n\n"
                "If you see Korean text, output it in Korean.\n"
                "If you see Japanese text, output it in Japanese.\n"
                "If you see Chinese text, output it in Chinese.\n"
                "If you see English text, output it in English.\n\n"
                "IMPORTANT: Only use line breaks where they naturally occur in the original text "
                "(e.g., between dialogue lines or paragraphs). Do not break text mid-sentence or "
                "between every word/character.\n\n"
                "For vertical text common in manga/comics, transcribe it as a continuous line unless "
                "there are clear visual breaks.\n\n"
                "NEVER translate. ONLY extract exactly what is written.\n"
                "Output ONLY the raw text, nothing else."
            )
            ocr_editor.delete(1.0, tk.END)
            ocr_editor.insert(1.0, default_ocr)
        
        # Buttons
        tb.Button(
            button_frame,
            text="Save",
            command=save_prompt,
            bootstyle="primary"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        tb.Button(
            button_frame,
            text="Reset to Default",
            command=reset_prompt,
            bootstyle="secondary"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        tb.Button(
            button_frame,
            text="Cancel",
            command=close_dialog,
            bootstyle="secondary"
        ).pack(side=tk.LEFT)
        
        # Auto-resize dialog to fit content
        self.main_gui.wm.auto_resize_dialog(dialog, canvas, max_width_ratio=0.7, max_height_ratio=0.6)
        
        # Handle window close
        dialog.protocol("WM_DELETE_WINDOW", close_dialog)
    
    def _refresh_context_settings(self):
        """Refresh context settings from main GUI"""
        # Actually fetch the current values from main GUI
        if hasattr(self.main_gui, 'contextual_var'):
            contextual_enabled = self.main_gui.contextual_var.get()
            self.contextual_status_label.config(text=f"• Contextual Translation: {'Enabled' if contextual_enabled else 'Disabled'}")
        
        if hasattr(self.main_gui, 'trans_history'):
            history_limit = self.main_gui.trans_history.get()
            self.history_limit_label.config(text=f"• Translation History Limit: {history_limit} exchanges")
        
        if hasattr(self.main_gui, 'translation_history_rolling_var'):
            rolling_enabled = self.main_gui.translation_history_rolling_var.get()
            rolling_status = "Enabled (Rolling Window)" if rolling_enabled else "Disabled (Reset on Limit)"
            self.rolling_status_label.config(text=f"• Rolling History: {rolling_status}")
        
        # Get and update model from main GUI
        current_model = None
        model_changed = False
        
        if hasattr(self.main_gui, 'model_var'):
            current_model = self.main_gui.model_var.get()
        elif hasattr(self.main_gui, 'model_combo'):
            current_model = self.main_gui.model_combo.get()
        elif hasattr(self.main_gui, 'config'):
            current_model = self.main_gui.config.get('model', 'Unknown')
        
        # Update model display in the API Settings frame
        for widget in self.parent_frame.winfo_children():
            if isinstance(widget, tk.LabelFrame) and "Translation Settings" in widget.cget("text"):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Frame):
                        for subchild in child.winfo_children():
                            if isinstance(subchild, tk.Label) and "Model:" in subchild.cget("text"):
                                old_model_text = subchild.cget("text")
                                old_model = old_model_text.split("Model: ")[-1] if "Model: " in old_model_text else None
                                if old_model != current_model:
                                    model_changed = True
                                subchild.config(text=f"Model: {current_model}")
                                break
        
        # If model changed, reset translator and client to force recreation
        if model_changed and current_model:
            if self.translator:
                self._log(f"Model changed to {current_model}. Translator will be recreated on next run.", "info")
                self.translator = None  # Force recreation on next translation
            
            # Also reset the client if it exists to ensure new model is used
            if hasattr(self.main_gui, 'client') and self.main_gui.client:
                if hasattr(self.main_gui.client, 'model') and self.main_gui.client.model != current_model:
                    self.main_gui.client = None  # Force recreation with new model
        
        # If translator exists, update its history manager settings
        if self.translator and hasattr(self.translator, 'history_manager'):
            try:
                # Update the history manager with current main GUI settings
                if hasattr(self.main_gui, 'contextual_var'):
                    self.translator.history_manager.contextual_enabled = self.main_gui.contextual_var.get()
                
                if hasattr(self.main_gui, 'trans_history'):
                    self.translator.history_manager.max_history = int(self.main_gui.trans_history.get())
                
                if hasattr(self.main_gui, 'translation_history_rolling_var'):
                    self.translator.history_manager.rolling_enabled = self.main_gui.translation_history_rolling_var.get()
                
                # Reset the history to apply new settings
                self.translator.history_manager.reset()
                
                self._log("✅ Refreshed context settings from main GUI and updated translator", "success")
            except Exception as e:
                self._log(f"✅ Refreshed context settings display (translator will update on next run)", "success")
        else:
            log_message = "✅ Refreshed context settings from main GUI"
            if model_changed:
                log_message += f" (Model: {current_model})"
            self._log(log_message, "success")
    
    def _browse_google_credentials_permanent(self):
        """Browse and set Google Cloud Vision credentials from the permanent button"""
        file_path = filedialog.askopenfilename(
            title="Select Google Cloud Service Account JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            # Save to config with both keys for compatibility
            self.main_gui.config['google_vision_credentials'] = file_path
            self.main_gui.config['google_cloud_credentials'] = file_path
            
            # Save configuration
            if hasattr(self.main_gui, 'save_config'):
                self.main_gui.save_config(show_message=False)

            
            # Update button state immediately
            self.start_button.config(state=tk.NORMAL)
            
            # Update credentials display
            self.creds_label.config(text=os.path.basename(file_path), fg='green')
            
            # Update status if we have a reference
            if hasattr(self, 'status_label'):
                self.status_label.config(text="✅ Ready", fg="green")
            
            messagebox.showinfo("Success", "Google Cloud credentials set successfully!")
    
    def _update_status_display(self):
        """Update the status display after credentials change"""
        # This would update the status label if we had a reference to it
        # For now, we'll just ensure the button is enabled
        google_creds_path = self.main_gui.config.get('google_vision_credentials', '') or self.main_gui.config.get('google_cloud_credentials', '')
        has_vision = os.path.exists(google_creds_path) if google_creds_path else False
        
        if has_vision:
            self.start_button.config(state=tk.NORMAL)
    
    def _get_available_fonts(self):
        """Get list of available fonts from system and custom directories"""
        fonts = ["Default"]  # Default option
        
        # Reset font mapping
        self.font_mapping = {}
        
        # Comprehensive map of Windows font filenames to proper display names
        font_name_map = {
            # === BASIC LATIN FONTS ===
            # Arial family
            'arial': 'Arial',
            'ariali': 'Arial Italic',
            'arialbd': 'Arial Bold',
            'arialbi': 'Arial Bold Italic',
            'ariblk': 'Arial Black',
            
            # Times New Roman
            'times': 'Times New Roman',
            'timesbd': 'Times New Roman Bold',
            'timesi': 'Times New Roman Italic',
            'timesbi': 'Times New Roman Bold Italic',
            
            # Calibri family
            'calibri': 'Calibri',
            'calibrib': 'Calibri Bold',
            'calibrii': 'Calibri Italic',
            'calibriz': 'Calibri Bold Italic',
            'calibril': 'Calibri Light',
            'calibrili': 'Calibri Light Italic',
            
            # Comic Sans family
            'comic': 'Comic Sans MS',
            'comici': 'Comic Sans MS Italic',
            'comicbd': 'Comic Sans MS Bold',
            'comicz': 'Comic Sans MS Bold Italic',
            
            # Segoe UI family
            'segoeui': 'Segoe UI',
            'segoeuib': 'Segoe UI Bold',
            'segoeuii': 'Segoe UI Italic',
            'segoeuiz': 'Segoe UI Bold Italic',
            'segoeuil': 'Segoe UI Light',
            'segoeuisl': 'Segoe UI Semilight',
            'seguisb': 'Segoe UI Semibold',
            'seguisbi': 'Segoe UI Semibold Italic',
            'seguisli': 'Segoe UI Semilight Italic',
            'seguili': 'Segoe UI Light Italic',
            'seguibl': 'Segoe UI Black',
            'seguibli': 'Segoe UI Black Italic',
            'seguihis': 'Segoe UI Historic',
            'seguiemj': 'Segoe UI Emoji',
            'seguisym': 'Segoe UI Symbol',
            
            # Courier
            'cour': 'Courier New',
            'courbd': 'Courier New Bold',
            'couri': 'Courier New Italic',
            'courbi': 'Courier New Bold Italic',
            
            # Verdana
            'verdana': 'Verdana',
            'verdanab': 'Verdana Bold',
            'verdanai': 'Verdana Italic',
            'verdanaz': 'Verdana Bold Italic',
            
            # Georgia
            'georgia': 'Georgia',
            'georgiab': 'Georgia Bold',
            'georgiai': 'Georgia Italic',
            'georgiaz': 'Georgia Bold Italic',
            
            # Tahoma
            'tahoma': 'Tahoma',
            'tahomabd': 'Tahoma Bold',
            
            # Trebuchet
            'trebuc': 'Trebuchet MS',
            'trebucbd': 'Trebuchet MS Bold',
            'trebucit': 'Trebuchet MS Italic',
            'trebucbi': 'Trebuchet MS Bold Italic',
            
            # Impact
            'impact': 'Impact',
            
            # Consolas
            'consola': 'Consolas',
            'consolab': 'Consolas Bold',
            'consolai': 'Consolas Italic',
            'consolaz': 'Consolas Bold Italic',
            
            # Sitka family (from your screenshot)
            'sitka': 'Sitka Small',
            'sitkab': 'Sitka Small Bold',
            'sitkai': 'Sitka Small Italic',
            'sitkaz': 'Sitka Small Bold Italic',
            'sitkavf': 'Sitka Text',
            'sitkavfb': 'Sitka Text Bold',
            'sitkavfi': 'Sitka Text Italic',
            'sitkavfz': 'Sitka Text Bold Italic',
            'sitkasubheading': 'Sitka Subheading',
            'sitkasubheadingb': 'Sitka Subheading Bold',
            'sitkasubheadingi': 'Sitka Subheading Italic',
            'sitkasubheadingz': 'Sitka Subheading Bold Italic',
            'sitkaheading': 'Sitka Heading',
            'sitkaheadingb': 'Sitka Heading Bold',
            'sitkaheadingi': 'Sitka Heading Italic',
            'sitkaheadingz': 'Sitka Heading Bold Italic',
            'sitkadisplay': 'Sitka Display',
            'sitkadisplayb': 'Sitka Display Bold',
            'sitkadisplayi': 'Sitka Display Italic',
            'sitkadisplayz': 'Sitka Display Bold Italic',
            'sitkabanner': 'Sitka Banner',
            'sitkabannerb': 'Sitka Banner Bold',
            'sitkabanneri': 'Sitka Banner Italic',
            'sitkabannerz': 'Sitka Banner Bold Italic',
            
            # Ink Free (from your screenshot)
            'inkfree': 'Ink Free',
            
            # Lucida family
            'l_10646': 'Lucida Sans Unicode',
            'lucon': 'Lucida Console',
            'ltype': 'Lucida Sans Typewriter',
            'ltypeb': 'Lucida Sans Typewriter Bold',
            'ltypei': 'Lucida Sans Typewriter Italic',
            'ltypebi': 'Lucida Sans Typewriter Bold Italic',

            # Palatino Linotype
            'pala': 'Palatino Linotype',
            'palab': 'Palatino Linotype Bold',
            'palabi': 'Palatino Linotype Bold Italic',
            'palai': 'Palatino Linotype Italic',

            # Noto fonts
            'notosansjp': 'Noto Sans JP',
            'notoserifjp': 'Noto Serif JP',

            # UD Digi Kyokasho (Japanese educational font)
            'uddigikyokashon-b': 'UD Digi Kyokasho NK-B',
            'uddigikyokashon-r': 'UD Digi Kyokasho NK-R',
            'uddigikyokashonk-b': 'UD Digi Kyokasho NK-B',
            'uddigikyokashonk-r': 'UD Digi Kyokasho NK-R',

            # Urdu Typesetting
            'urdtype': 'Urdu Typesetting',
            'urdtypeb': 'Urdu Typesetting Bold',

            # Segoe variants
            'segmdl2': 'Segoe MDL2 Assets',
            'segoeicons': 'Segoe Fluent Icons',
            'segoepr': 'Segoe Print',
            'segoeprb': 'Segoe Print Bold',
            'segoesc': 'Segoe Script',
            'segoescb': 'Segoe Script Bold',
            'seguivar': 'Segoe UI Variable',

            # Sans Serif Collection
            'sansserifcollection': 'Sans Serif Collection',

            # Additional common Windows 10/11 fonts
            'holomdl2': 'HoloLens MDL2 Assets',
            'gadugi': 'Gadugi',
            'gadugib': 'Gadugi Bold',

            # Cascadia Code (developer font)
            'cascadiacode': 'Cascadia Code',
            'cascadiacodepl': 'Cascadia Code PL',
            'cascadiamono': 'Cascadia Mono',
            'cascadiamonopl': 'Cascadia Mono PL',

            # More Segoe UI variants
            'seguibli': 'Segoe UI Black Italic',
            'segoeuiblack': 'Segoe UI Black',

            # Other fonts
            'aldhabi': 'Aldhabi',
            'andiso': 'Andalus',  # This is likely Andalus font
            'arabtype': 'Arabic Typesetting',
            'mstmc': 'Myanmar Text',  # Alternate file name
            'monbaiti': 'Mongolian Baiti',  # Shorter filename variant
            'leeluisl': 'Leelawadee UI Semilight',  # Missing variant
            'simsunextg': 'SimSun-ExtG',  # Extended SimSun variant
            'ebrima': 'Ebrima',
            'ebrimabd': 'Ebrima Bold',
            'gabriola': 'Gabriola',

            # Bahnschrift variants
            'bahnschrift': 'Bahnschrift',
            'bahnschriftlight': 'Bahnschrift Light',
            'bahnschriftsemibold': 'Bahnschrift SemiBold',
            'bahnschriftbold': 'Bahnschrift Bold',

            # Majalla (African language font)
            'majalla': 'Sakkal Majalla',
            'majallab': 'Sakkal Majalla Bold',

            # Additional fonts that might be missing
            'amiri': 'Amiri',
            'amiri-bold': 'Amiri Bold',
            'amiri-slanted': 'Amiri Slanted',
            'amiri-boldslanted': 'Amiri Bold Slanted',
            'aparaj': 'Aparajita',
            'aparajb': 'Aparajita Bold',
            'aparaji': 'Aparajita Italic',
            'aparajbi': 'Aparajita Bold Italic',
            'kokila': 'Kokila',
            'kokilab': 'Kokila Bold',
            'kokilai': 'Kokila Italic',
            'kokilabi': 'Kokila Bold Italic',
            'utsaah': 'Utsaah',
            'utsaahb': 'Utsaah Bold',
            'utsaahi': 'Utsaah Italic',
            'utsaahbi': 'Utsaah Bold Italic',
            'vani': 'Vani',
            'vanib': 'Vani Bold',
            
            # === JAPANESE FONTS ===
            'msgothic': 'MS Gothic',
            'mspgothic': 'MS PGothic',
            'msmincho': 'MS Mincho',
            'mspmincho': 'MS PMincho',
            'meiryo': 'Meiryo',
            'meiryob': 'Meiryo Bold',
            'yugothic': 'Yu Gothic',
            'yugothb': 'Yu Gothic Bold',
            'yugothl': 'Yu Gothic Light',
            'yugothm': 'Yu Gothic Medium',
            'yugothr': 'Yu Gothic Regular',
            'yumin': 'Yu Mincho',
            'yumindb': 'Yu Mincho Demibold',
            'yuminl': 'Yu Mincho Light',
            
            # === KOREAN FONTS ===
            'malgun': 'Malgun Gothic',
            'malgunbd': 'Malgun Gothic Bold',
            'malgunsl': 'Malgun Gothic Semilight',
            'gulim': 'Gulim',
            'gulimche': 'GulimChe',
            'dotum': 'Dotum',
            'dotumche': 'DotumChe',
            'batang': 'Batang',
            'batangche': 'BatangChe',
            'gungsuh': 'Gungsuh',
            'gungsuhche': 'GungsuhChe',
            
            # === CHINESE FONTS ===
            # Simplified Chinese
            'simsun': 'SimSun',
            'simsunb': 'SimSun Bold',
            'simsunextb': 'SimSun ExtB',
            'nsimsun': 'NSimSun',
            'simhei': 'SimHei',
            'simkai': 'KaiTi',
            'simfang': 'FangSong',
            'simli': 'LiSu',
            'simyou': 'YouYuan',
            'stcaiyun': 'STCaiyun',
            'stfangsong': 'STFangsong',
            'sthupo': 'STHupo',
            'stkaiti': 'STKaiti',
            'stliti': 'STLiti',
            'stsong': 'STSong',
            'stxihei': 'STXihei',
            'stxingkai': 'STXingkai',
            'stxinwei': 'STXinwei',
            'stzhongsong': 'STZhongsong',
            
            # Traditional Chinese  
            'msjh': 'Microsoft JhengHei',
            'msjhbd': 'Microsoft JhengHei Bold',
            'msjhl': 'Microsoft JhengHei Light',
            'mingliu': 'MingLiU',
            'pmingliu': 'PMingLiU',
            'mingliub': 'MingLiU Bold',
            'mingliuhk': 'MingLiU_HKSCS',
            'mingliuextb': 'MingLiU ExtB',
            'pmingliuextb': 'PMingLiU ExtB',
            'mingliuhkextb': 'MingLiU_HKSCS ExtB',
            'kaiu': 'DFKai-SB',
            
            # Microsoft YaHei
            'msyh': 'Microsoft YaHei',
            'msyhbd': 'Microsoft YaHei Bold',
            'msyhl': 'Microsoft YaHei Light',
            
            # === THAI FONTS ===
            'leelawui': 'Leelawadee UI',
            'leelauib': 'Leelawadee UI Bold',
            'leelauisl': 'Leelawadee UI Semilight',
            'leelawad': 'Leelawadee',
            'leelawdb': 'Leelawadee Bold',
            
            # === INDIC FONTS ===
            'mangal': 'Mangal',
            'vrinda': 'Vrinda',
            'raavi': 'Raavi',
            'shruti': 'Shruti',
            'tunga': 'Tunga',
            'gautami': 'Gautami',
            'kartika': 'Kartika',
            'latha': 'Latha',
            'kalinga': 'Kalinga',
            'vijaya': 'Vijaya',
            'nirmala': 'Nirmala UI',
            'nirmalab': 'Nirmala UI Bold',
            'nirmalas': 'Nirmala UI Semilight',
            
            # === ARABIC FONTS ===
            'arial': 'Arial',
            'trado': 'Traditional Arabic',
            'tradbdo': 'Traditional Arabic Bold',
            'simpo': 'Simplified Arabic',
            'simpbdo': 'Simplified Arabic Bold',
            'simpfxo': 'Simplified Arabic Fixed',
            
            # === OTHER ASIAN FONTS ===
            'javatext': 'Javanese Text',
            'himalaya': 'Microsoft Himalaya',
            'mongolianbaiti': 'Mongolian Baiti',
            'msuighur': 'Microsoft Uighur',
            'msuighub': 'Microsoft Uighur Bold',
            'msyi': 'Microsoft Yi Baiti',
            'taileb': 'Microsoft Tai Le Bold',
            'taile': 'Microsoft Tai Le',
            'ntailu': 'Microsoft New Tai Lue',
            'ntailub': 'Microsoft New Tai Lue Bold',
            'phagspa': 'Microsoft PhagsPa',
            'phagspab': 'Microsoft PhagsPa Bold',
            'mmrtext': 'Myanmar Text',
            'mmrtextb': 'Myanmar Text Bold',
            
            # === SYMBOL FONTS ===
            'symbol': 'Symbol',
            'webdings': 'Webdings',
            'wingding': 'Wingdings',
            'wingdng2': 'Wingdings 2',
            'wingdng3': 'Wingdings 3',
            'mtextra': 'MT Extra',
            'marlett': 'Marlett',
            
            # === OTHER FONTS ===
            'mvboli': 'MV Boli',
            'sylfaen': 'Sylfaen',
            'estrangelo': 'Estrangelo Edessa',
            'euphemia': 'Euphemia',
            'plantagenet': 'Plantagenet Cherokee',
            'micross': 'Microsoft Sans Serif',
            
            # Franklin Gothic
            'framd': 'Franklin Gothic Medium',
            'framdit': 'Franklin Gothic Medium Italic',
            'fradm': 'Franklin Gothic Demi',
            'fradmcn': 'Franklin Gothic Demi Cond',
            'fradmit': 'Franklin Gothic Demi Italic',
            'frahv': 'Franklin Gothic Heavy',
            'frahvit': 'Franklin Gothic Heavy Italic',
            'frabook': 'Franklin Gothic Book',
            'frabookit': 'Franklin Gothic Book Italic',
            
            # Cambria
            'cambria': 'Cambria',
            'cambriab': 'Cambria Bold',
            'cambriai': 'Cambria Italic',
            'cambriaz': 'Cambria Bold Italic',
            'cambria&cambria math': 'Cambria Math',
            
            # Candara
            'candara': 'Candara',
            'candarab': 'Candara Bold',
            'candarai': 'Candara Italic',
            'candaraz': 'Candara Bold Italic',
            'candaral': 'Candara Light',
            'candarali': 'Candara Light Italic',
            
            # Constantia
            'constan': 'Constantia',
            'constanb': 'Constantia Bold',
            'constani': 'Constantia Italic',
            'constanz': 'Constantia Bold Italic',
            
            # Corbel
            'corbel': 'Corbel',
            'corbelb': 'Corbel Bold',
            'corbeli': 'Corbel Italic',
            'corbelz': 'Corbel Bold Italic',
            'corbell': 'Corbel Light',
            'corbelli': 'Corbel Light Italic',
            
            # Bahnschrift
            'bahnschrift': 'Bahnschrift',
            
            # Garamond
            'gara': 'Garamond',
            'garabd': 'Garamond Bold',
            'garait': 'Garamond Italic',
            
            # Century Gothic
            'gothic': 'Century Gothic',
            'gothicb': 'Century Gothic Bold',
            'gothici': 'Century Gothic Italic',
            'gothicz': 'Century Gothic Bold Italic',
            
            # Bookman Old Style
            'bookos': 'Bookman Old Style',
            'bookosb': 'Bookman Old Style Bold',
            'bookosi': 'Bookman Old Style Italic',
            'bookosbi': 'Bookman Old Style Bold Italic',
        }
        
        # Dynamically discover all Windows fonts
        windows_fonts = []
        windows_font_dir = "C:/Windows/Fonts"
        
        if os.path.exists(windows_font_dir):
            for font_file in os.listdir(windows_font_dir):
                font_path = os.path.join(windows_font_dir, font_file)
                
                # Check if it's a font file
                if os.path.isfile(font_path) and font_file.lower().endswith(('.ttf', '.ttc', '.otf')):
                    # Get base name without extension
                    base_name = os.path.splitext(font_file)[0]
                    base_name_lower = base_name.lower()
                    
                    # Check if we have a proper name mapping
                    if base_name_lower in font_name_map:
                        display_name = font_name_map[base_name_lower]
                    else:
                        # Generic cleanup for unmapped fonts
                        display_name = base_name.replace('_', ' ').replace('-', ' ')
                        display_name = ' '.join(word.capitalize() for word in display_name.split())
                    
                    windows_fonts.append((display_name, font_path))
        
        # Sort alphabetically
        windows_fonts.sort(key=lambda x: x[0])
        
        # Add all discovered fonts to the list
        for font_name, font_path in windows_fonts:
            fonts.append(font_name)
            self.font_mapping[font_name] = font_path
        
        # Check for custom fonts directory (keep your existing code)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fonts_dir = os.path.join(script_dir, "fonts")
        
        if os.path.exists(fonts_dir):
            for root, dirs, files in os.walk(fonts_dir):
                for font_file in files:
                    if font_file.endswith(('.ttf', '.ttc', '.otf')):
                        font_path = os.path.join(root, font_file)
                        font_name = os.path.splitext(font_file)[0]
                        # Add category from folder
                        category = os.path.basename(root)
                        if category != "fonts":
                            font_name = f"{font_name} ({category})"
                        fonts.append(font_name)
                        self.font_mapping[font_name] = font_path
        
        # Load previously saved custom fonts (keep your existing code)
        if 'custom_fonts' in self.main_gui.config:
            for custom_font in self.main_gui.config['custom_fonts']:
                if os.path.exists(custom_font['path']):
                    # Check if this font is already in the list
                    if custom_font['name'] not in fonts:
                        fonts.append(custom_font['name'])
                        self.font_mapping[custom_font['name']] = custom_font['path']
        
        # Add custom fonts option at the end
        fonts.append("Browse Custom Font...")
        
        return fonts
    
    def _on_font_selected(self, event):
        """Handle font selection"""
        selected = self.font_style_var.get()
        
        if selected == "Default":
            self.selected_font_path = None
        elif selected == "Browse Custom Font...":
            # Open file dialog to select custom font
            font_path = filedialog.askopenfilename(
                title="Select Font File",
                filetypes=[
                    ("Font files", "*.ttf *.ttc *.otf"),
                    ("TrueType fonts", "*.ttf"),
                    ("TrueType collections", "*.ttc"),
                    ("OpenType fonts", "*.otf"),
                    ("All files", "*.*")
                ]
            )
            
            if font_path:
                # Add to combo box
                font_name = os.path.basename(font_path)
                current_values = list(self.font_combo['values'])
                
                # Insert before "Browse Custom Font..." option
                if font_name not in [n for n in self.font_mapping.keys()]:
                    current_values.insert(-1, font_name)
                    self.font_combo['values'] = current_values
                    self.font_combo.set(font_name)
                    
                    # Update font mapping
                    self.font_mapping[font_name] = font_path
                    self.selected_font_path = font_path
                    
                    # Save custom font to config
                    if 'custom_fonts' not in self.main_gui.config:
                        self.main_gui.config['custom_fonts'] = []
                    
                    custom_font_entry = {'name': font_name, 'path': font_path}
                    # Check if this exact entry already exists
                    font_exists = False
                    for existing_font in self.main_gui.config['custom_fonts']:
                        if existing_font['path'] == font_path:
                            font_exists = True
                            break
                    
                    if not font_exists:
                        self.main_gui.config['custom_fonts'].append(custom_font_entry)
                        # Save config immediately to persist custom fonts
                        if hasattr(self.main_gui, 'save_config'):
                            self.main_gui.save_config(show_message=False)
                else:
                    # Font already exists, just select it
                    self.font_combo.set(font_name)
                    self.selected_font_path = self.font_mapping[font_name]
            else:
                # User cancelled, revert to previous selection
                if hasattr(self, 'previous_font_selection'):
                    self.font_combo.set(self.previous_font_selection)
                else:
                    self.font_combo.set("Default")
                return
        else:
            # Check if it's in the font mapping
            if selected in self.font_mapping:
                self.selected_font_path = self.font_mapping[selected]
            else:
                # This shouldn't happen, but just in case
                self.selected_font_path = None
        
        # Store current selection for next time
        self.previous_font_selection = selected
        
        # Auto-save on change
        self._save_rendering_settings()
    
    def _update_opacity_label(self, value):
        """Update opacity percentage label"""
        percentage = int((float(value) / 255) * 100)
        self.opacity_label.config(text=f"{percentage}%")
        # Auto-save on change
        self._save_rendering_settings()
    
    def _update_reduction_label(self, value):
        """Update size reduction percentage label"""
        percentage = int(float(value) * 100)
        self.reduction_label.config(text=f"{percentage}%")
        # Auto-save on change
        self._save_rendering_settings()
        
    def _toggle_inpaint_quality_visibility(self):
        """Show/hide inpaint quality options based on skip_inpainting setting"""
        if hasattr(self, 'inpaint_quality_frame'):
            if self.skip_inpainting_var.get():
                # Hide quality options when inpainting is skipped
                self.inpaint_quality_frame.pack_forget()
            else:
                # Show quality options when inpainting is enabled
                self.inpaint_quality_frame.pack(fill=tk.X, pady=5, after=self.skip_inpainting_checkbox)

    def _toggle_inpaint_visibility(self):
        """Show/hide inpainting options based on skip toggle"""
        if self.skip_inpainting_var.get():
            # Hide all inpainting options
            self.inpaint_method_frame.pack_forget()
            self.cloud_inpaint_frame.pack_forget()
            self.local_inpaint_frame.pack_forget()
            self.inpaint_separator.pack_forget()  # Hide separator
        else:
            # Show method selection
            self.inpaint_method_frame.pack(fill=tk.X, pady=5, after=self.skip_inpainting_checkbox)
            self.inpaint_separator.pack(fill=tk.X, pady=(10, 10))  # Show separator
            self._on_inpaint_method_change()
        
        self._save_rendering_settings()

    def _on_inpaint_method_change(self):
        """Show appropriate inpainting settings based on method"""
        method = self.inpaint_method_var.get()
        
        if method == 'cloud':
            self.cloud_inpaint_frame.pack(fill=tk.X, pady=5, after=self.inpaint_method_frame)
            self.local_inpaint_frame.pack_forget()
        elif method == 'local':
            self.local_inpaint_frame.pack(fill=tk.X, pady=10, after=self.inpaint_method_frame)
            self.cloud_inpaint_frame.pack_forget()
        elif method == 'hybrid':
            # Show both frames for hybrid
            self.local_inpaint_frame.pack(fill=tk.X, pady=5, after=self.inpaint_method_frame)
            self.cloud_inpaint_frame.pack(fill=tk.X, pady=5, after=self.local_inpaint_frame)
        
        self._save_rendering_settings()

    def _on_local_model_change(self, event=None):
        """Handle model type change and auto-load if model exists"""
        model_type = self.local_model_type_var.get()
        
        # Update description
        model_desc = {
            'lama': 'LaMa (Best quality)',
            'aot': 'AOT GAN (Fast)',
            'aot_onnx': 'AOT ONNX (Optimized)',
            'mat': 'MAT (High-res)',
            'sd_local': 'Stable Diffusion (Anime)',
            'anime': 'Anime/Manga Inpainting',
            'anime_onnx': 'Anime ONNX (Fast/Optimized)',
        }
        self.model_desc_label.config(text=model_desc.get(model_type, ''))
        
        # Check for saved path for this model type
        saved_path = self.main_gui.config.get(f'manga_{model_type}_model_path', '')
        
        if saved_path and os.path.exists(saved_path):
            # Update the path display
            self.local_model_path_var.set(saved_path)
            self.local_model_status_label.config(text="⏳ Loading saved model...", fg='orange')
            
            # Auto-load the model after a short delay
            self.dialog.after(100, lambda: self._try_load_model(model_type, saved_path))
        else:
            # Clear the path display
            self.local_model_path_var.set("")
            self.local_model_status_label.config(text="No model loaded", fg='gray')
        
        self._save_rendering_settings()

    def _browse_local_model(self):
        """Browse for local inpainting model and auto-load"""
        model_type = self.local_model_type_var.get()
        
        if model_type == 'sd_local':
            filetypes = [
                ("Model files", "*.safetensors *.pt *.pth *.ckpt *.onnx"),
                ("SafeTensors", "*.safetensors"),
                ("Checkpoint files", "*.ckpt"),
                ("PyTorch models", "*.pt *.pth"),
                ("ONNX models", "*.onnx"),
                ("All files", "*.*")
            ]
        else:
            filetypes = [
                ("Model files", "*.pt *.pth *.ckpt *.onnx"),
                ("Checkpoint files", "*.ckpt"),
                ("PyTorch models", "*.pt *.pth"),
                ("ONNX models", "*.onnx"),
                ("All files", "*.*")
            ]
        
        path = filedialog.askopenfilename(
            title=f"Select {model_type.upper()} Model",
            filetypes=filetypes
        )
        
        if path:
            self.local_model_path_var.set(path)
            # Save to config
            self.main_gui.config[f'manga_{model_type}_model_path'] = path
            self._save_rendering_settings()
            
            # Update status first
            self._update_local_model_status()
            
            # Auto-load the selected model
            self.dialog.after(100, lambda: self._try_load_model(model_type, path))

    def _click_load_local_model(self):
        """Manually trigger loading of the selected local inpainting model"""
        try:
            model_type = self.local_model_type_var.get() if hasattr(self, 'local_model_type_var') else None
            path = self.local_model_path_var.get() if hasattr(self, 'local_model_path_var') else ''
            if not model_type or not path:
                messagebox.showinfo("Load Model", "Please select a model file first using the Browse button.")
                return
            # Defer to keep UI responsive
            self.dialog.after(50, lambda: self._try_load_model(model_type, path))
        except Exception:
            pass

    def _try_load_model(self, method: str, model_path: str):
        """Try to load a model and update status (runs loading on a background thread)."""
        try:
            # Show loading status immediately
            self.local_model_status_label.config(text="⏳ Loading model...", fg='orange')
            self.dialog.update_idletasks()
            self.main_gui.append_log(f"⏳ Loading {method.upper()} model...")

            def do_load():
                from local_inpainter import LocalInpainter
                ok = False
                try:
                    test_inpainter = LocalInpainter()
                    ok = test_inpainter.load_model_with_retry(method, model_path, force_reload=True)
                except Exception as e:
                    self.main_gui.append_log(f"❌ Error loading model: {e}")
                    ok = False
                # Update UI on main thread
                def _after():
                    if ok:
                        self._update_local_model_status()
                        self.local_model_status_label.config(
                            text=f"✅ {method.upper()} model loaded successfully!",
                            fg='green'
                        )
                        self.main_gui.append_log(f"✅ {method.upper()} model loaded successfully!")
                        if hasattr(self, 'translator') and self.translator:
                            for attr in ('local_inpainter', '_last_local_method', '_last_local_model_path'):
                                if hasattr(self.translator, attr):
                                    try:
                                        delattr(self.translator, attr)
                                    except Exception:
                                        pass
                        self.dialog.after(3000, self._update_local_model_status)
                    else:
                        self.local_model_status_label.config(
                            text="⚠️ Model file found but failed to load",
                            fg='orange'
                        )
                        self.main_gui.append_log("⚠️ Model file found but failed to load")
                try:
                    self.dialog.after(0, _after)
                except Exception:
                    pass
            # Fire background loader
            threading.Thread(target=do_load, daemon=True).start()
            return True
        except Exception as e:
            try:
                self.local_model_status_label.config(text=f"❌ Error: {str(e)[:50]}", fg='red')
            except Exception:
                pass
            self.main_gui.append_log(f"❌ Error loading model: {e}")
            return False
            self.local_model_status_label.config(
                text=f"❌ Error: {str(e)[:50]}",
                fg='red'
            )
            self.main_gui.append_log(f"❌ Error loading model: {str(e)}")
            return False
        
    def _update_local_model_status(self):
        """Update local model status display"""
        path = self.local_model_path_var.get()
        
        if not path:
            self.local_model_status_label.config(text="⚠️ No model selected", fg='orange')
            return
        
        if not os.path.exists(path):
            self.local_model_status_label.config(text="❌ Model file not found", fg='red')
            return
        
        # Check for ONNX cache
        if path.endswith(('.pt', '.pth', '.safetensors')):
            onnx_dir = os.path.join(os.path.dirname(path), 'models')
            if os.path.exists(onnx_dir):
                # Check if ONNX file exists for this model
                model_hash = hashlib.md5(path.encode()).hexdigest()[:8]
                onnx_files = [f for f in os.listdir(onnx_dir) if model_hash in f]
                if onnx_files:
                    self.local_model_status_label.config(
                        text="✅ Model ready (ONNX cached)",
                        fg='green'
                    )
                else:
                    self.local_model_status_label.config(
                        text="ℹ️ Will convert to ONNX on first use",
                        fg='blue'
                    )
            else:
                self.local_model_status_label.config(
                    text="ℹ️ Will convert to ONNX on first use",
                    fg='blue'
                )
        else:
            self.local_model_status_label.config(
                text="✅ ONNX model ready",
                fg='green'
            )

    def _download_model(self):
        """Actually download the model for the selected type"""
        model_type = self.local_model_type_var.get()
        
        # Define URLs for each model type
        model_urls = {
            'aot': 'https://huggingface.co/ogkalu/aot-inpainting-jit/resolve/main/aot_traced.pt',
            'aot_onnx': 'https://huggingface.co/ogkalu/aot-inpainting/resolve/main/aot.onnx',
            'lama': 'https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt',
            'lama_onnx': 'https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx',  
            'anime': 'https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt',
            'anime_onnx': 'https://huggingface.co/ogkalu/lama-manga-onnx-dynamic/resolve/main/lama-manga-dynamic.onnx',
            'mat': '',  # User must provide
            'ollama': '',  # Not applicable
            'sd_local': ''  # User must provide
        }
        
        url = model_urls.get(model_type, '')
        
        if not url:
            messagebox.showinfo("Manual Download", 
                f"Please manually download and browse for {model_type} model")
            return
        
        # Determine filename
        filename_map = {
            'aot': 'aot_traced.pt',
            'aot_onnx': 'aot.onnx',
            'lama': 'big-lama.pt',
            'anime': 'anime-manga-big-lama.pt',
            'anime_onnx': 'lama-manga-dynamic.onnx',
            'lama_onnx': 'lama_fp32.onnx',
            'fcf_onnx': 'fcf.onnx',
            'sd_inpaint_onnx': 'sd_inpaint_unet.onnx'
        }
        
        filename = filename_map.get(model_type, f'{model_type}.pt')
        save_path = os.path.join('models', filename)
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Check if already exists
        if os.path.exists(save_path):
            self.local_model_path_var.set(save_path)
            self.local_model_status_label.config(text="✅ Model already downloaded")
            messagebox.showinfo("Model Ready", f"Model already exists at:\n{save_path}")
            return
        
        # Download the model
        self._perform_download(url, save_path, model_type)

    def _perform_download(self, url: str, save_path: str, model_name: str):
        """Perform the actual download with progress indication"""
        import threading
        import requests
        
        # Create a progress dialog
        progress_dialog = tk.Toplevel(self.dialog)
        progress_dialog.title(f"Downloading {model_name.upper()} Model")
        progress_dialog.geometry("400x150")
        progress_dialog.transient(self.dialog)
        progress_dialog.grab_set()
        
        # Center the dialog
        progress_dialog.update_idletasks()
        x = (progress_dialog.winfo_screenwidth() // 2) - (progress_dialog.winfo_width() // 2)
        y = (progress_dialog.winfo_screenheight() // 2) - (progress_dialog.winfo_height() // 2)
        progress_dialog.geometry(f"+{x}+{y}")
        
        # Progress label
        progress_label = tk.Label(progress_dialog, text="⏳ Downloading...", font=('Arial', 10))
        progress_label.pack(pady=20)
        
        # Progress bar
        progress_var = tk.DoubleVar()
        try:
            # Try to use our custom progress bar style
            progress_bar = ttk.Progressbar(
                progress_dialog,
                length=350,
                mode='determinate',
                variable=progress_var,
                style="MangaProgress.Horizontal.TProgressbar"
            )
        except Exception:
            # Fallback to default if style not available yet
            progress_bar = ttk.Progressbar(
                progress_dialog,
                length=350,
                mode='determinate',
                variable=progress_var
            )
        progress_bar.pack(pady=10)
        
        # Status label
        status_label = tk.Label(progress_dialog, text="0%", font=('Arial', 9))
        status_label.pack()
        
        # Cancel flag
        cancel_download = {'value': False}
        
        def on_cancel():
            cancel_download['value'] = True
            progress_dialog.destroy()
        
        progress_dialog.protocol("WM_DELETE_WINDOW", on_cancel)
        
        def download_thread():
            try:
                # Download with progress
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if cancel_download['value']:
                            # Clean up partial file
                            f.close()
                            if os.path.exists(save_path):
                                os.remove(save_path)
                            return
                        
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Update progress
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                progress_dialog.after(0, lambda p=progress: [
                                    progress_var.set(p),
                                    status_label.config(text=f"{p:.1f}%"),
                                    progress_label.config(text=f"⏳ Downloading... {downloaded//1024//1024}MB / {total_size//1024//1024}MB")
                                ])
                
                # Success - update UI in main thread
                progress_dialog.after(0, lambda: [
                    progress_dialog.destroy(),
                    self._download_complete(save_path, model_name)
                ])
                
            except requests.exceptions.RequestException as e:
                # Error - update UI in main thread
                if not cancel_download['value']:
                    error_msg = str(e)  # Capture error before lambda
                    progress_dialog.after(0, lambda: [
                        progress_dialog.destroy(),
                        self._download_failed(error_msg)
                    ])
            except Exception as e:
                if not cancel_download['value']:
                    error_msg = str(e)  # Capture error before lambda
                    progress_dialog.after(0, lambda: [
                        progress_dialog.destroy(),
                        self._download_failed(error_msg)
                    ])
        
        # Start download in background thread
        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()

    def _download_complete(self, save_path: str, model_name: str):
        """Handle successful download"""
        # Update the model path entry
        self.local_model_path_var.set(save_path)
        
        # Save to config
        self.main_gui.config[f'manga_{model_name}_model_path'] = save_path
        self._save_rendering_settings()
        
        # Auto-load the downloaded model
        self.local_model_status_label.config(text="⏳ Loading downloaded model...", fg='orange')
        
        def load_after_download():
            if self._try_load_model(model_name, save_path):
                messagebox.showinfo("Success", f"{model_name.upper()} model downloaded and loaded!")
            else:
                messagebox.showinfo("Download Complete", f"{model_name.upper()} model downloaded but needs manual loading")
        
        self.dialog.after(100, load_after_download)
        
        # Log to main GUI
        self.main_gui.append_log(f"✅ Downloaded {model_name} model to: {save_path}")

    def _download_failed(self, error: str):
        """Handle download failure"""
        messagebox.showerror("Download Failed", f"Failed to download model:\n{error}")
        self.main_gui.append_log(f"❌ Model download failed: {error}")

    def _show_model_info(self):
        """Show information about models"""
        model_type = self.local_model_type_var.get()
        
        info = {
            'aot': "AOT GAN Model:\n\n"
                   "• Auto-downloads from HuggingFace\n"
                   "• Traced PyTorch JIT model\n"
                   "• Good for general inpainting\n"
                   "• Fast processing speed\n"
                   "• File size: ~100MB",
            
            'aot_onnx': "AOT ONNX Model:\n\n"
                        "• Optimized ONNX version\n"
                        "• Auto-downloads from HuggingFace\n"
                        "• 2-3x faster than PyTorch version\n"
                        "• Great for batch processing\n"
                        "• Lower memory usage\n"
                        "• File size: ~100MB",
            
            'lama': "LaMa Model:\n\n"
                    "• Auto-downloads anime-optimized version\n"
                    "• Best quality for manga/anime\n"
                    "• Large model (~200MB)\n"
                    "• Excellent at removing text from bubbles\n"
                    "• Preserves art style well",
            
            'anime': "Anime-Specific Model:\n\n"
                     "• Same as LaMa anime version\n"
                     "• Optimized for manga/anime art\n"
                     "• Auto-downloads from GitHub\n"
                     "• Recommended for manga translation\n"
                     "• Preserves screen tones and patterns",
            
            'anime_onnx': "Anime ONNX Model:\n\n"
                          "• Optimized ONNX version for speed\n"
                          "• Auto-downloads from HuggingFace\n"
                          "• 2-3x faster than PyTorch version\n"
                          "• Perfect for batch processing\n"
                          "• Same quality as anime model\n"
                          "• File size: ~190MB\n"
                          "• DEFAULT for inpainting",
            
            'mat': "MAT Model:\n\n"
                   "• Manual download required\n"
                   "• Get from: github.com/fenglinglwb/MAT\n"
                   "• Good for high-resolution images\n"
                   "• Slower but high quality\n"
                   "• File size: ~500MB",
            
            'ollama': "Ollama:\n\n"
                      "• Uses local Ollama server\n"
                      "• No model download needed here\n"
                      "• Run: ollama pull llava\n"
                      "• Context-aware inpainting\n"
                      "• Requires Ollama running locally",
            
            'sd_local': "Stable Diffusion:\n\n"
                        "• Manual download required\n"
                        "• Get from HuggingFace\n"
                        "• Requires significant VRAM (4-8GB)\n"
                        "• Best quality but slowest\n"
                        "• Can use custom prompts"
        }
        
        # Create info dialog
        info_dialog = tk.Toplevel(self.dialog)
        info_dialog.title(f"{model_type.upper()} Model Information")
        info_dialog.geometry("450x350")
        info_dialog.transient(self.dialog)
        
        # Center the dialog
        info_dialog.update_idletasks()
        x = (info_dialog.winfo_screenwidth() // 2) - (info_dialog.winfo_width() // 2)
        y = (info_dialog.winfo_screenheight() // 2) - (info_dialog.winfo_height() // 2)
        info_dialog.geometry(f"+{x}+{y}")
        
        # Info text
        text_widget = tk.Text(info_dialog, wrap=tk.WORD, padx=20, pady=20)
        text_widget.pack(fill='both', expand=True)
        text_widget.insert(1.0, info.get(model_type, "Please select a model type first"))
        text_widget.config(state='disabled')
        
        # Close button
        tb.Button(
            info_dialog,
            text="Close",
            command=info_dialog.destroy,
            bootstyle="secondary"
        ).pack(pady=10)

    def _toggle_inpaint_controls_visibility(self):
            """Toggle visibility of inpaint controls (mask expansion and passes) based on skip inpainting setting"""
            # Just return if the frame doesn't exist - prevents AttributeError
            if not hasattr(self, 'inpaint_controls_frame'):
                return
                
            if self.skip_inpainting_var.get():
                self.inpaint_controls_frame.pack_forget()
            else:
                # Pack it back in the right position
                self.inpaint_controls_frame.pack(fill=tk.X, pady=5, after=self.inpaint_quality_frame)

    def _configure_inpaint_api(self):
        """Configure cloud inpainting API"""
        # Show instructions
        result = messagebox.askyesno(
            "Configure Cloud Inpainting",
            "Cloud inpainting uses Replicate API for questionable results.\n\n"
            "1. Go to replicate.com and sign up (free tier available?)\n"
            "2. Get your API token from Account Settings\n"
            "3. Enter it here\n\n"
            "Pricing: ~$0.0023 per image?\n"
            "Free tier: ~100 images per month?\n\n"
            "Would you like to proceed?"
        )
        
        if not result:
            return
        
        # Open Replicate page
        import webbrowser
        webbrowser.open("https://replicate.com/account/api-tokens")
        
        # Get API key from user using WindowManager
        dialog = self.main_gui.wm.create_simple_dialog(
            self.main_gui.master,
            "Replicate API Key",
            width=None,
            height=None,
            hide_initially=True  # Hide initially so we can position it
        )
        
        # Force the height by overriding after creation
        dialog.update_idletasks()  # Process pending geometry
        dialog.minsize(None, None)   # Set minimum size
        dialog.maxsize(None, None)   # Set maximum size to lock it
        
        # Get cursor position
        cursor_x = self.main_gui.master.winfo_pointerx()
        cursor_y = self.main_gui.master.winfo_pointery()
        
        # Offset the dialog slightly so it doesn't appear directly under cursor
        # This prevents the cursor from immediately being over a button
        offset_x = 10
        offset_y = 10
        
        # Ensure dialog doesn't go off-screen
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()
        
        # Adjust position if it would go off-screen
        if cursor_x + 400 + offset_x > screen_width:
            cursor_x = screen_width - 400 - offset_x
        if cursor_y + 150 + offset_y > screen_height:
            cursor_y = screen_height - 150 - offset_y
        
        # Set position and show
        dialog.geometry(f"400x150+{cursor_x + offset_x}+{cursor_y + offset_y}")
        dialog.deiconify()  # Show the dialog
        
        # Variables
        api_key_var = tk.StringVar()
        result = {'key': None}
        
        # Content
        frame = tk.Frame(dialog, padx=20, pady=20)
        frame.pack(fill='both', expand=True)
        
        tk.Label(frame, text="Enter your Replicate API key:").pack(anchor='w', pady=(0, 10))
        
        # Entry with show/hide
        entry_frame = tk.Frame(frame)
        entry_frame.pack(fill='x')
        
        entry = tk.Entry(entry_frame, textvariable=api_key_var, show='*', width=20)
        entry.pack(side='left', fill='x', expand=True)
        
        # Toggle show/hide
        def toggle_show():
            current = entry.cget('show')
            entry.config(show='' if current else '*')
            show_btn.config(text='Hide' if current else 'Show')
        
        show_btn = tb.Button(entry_frame, text="Show", command=toggle_show, width=8)
        show_btn.pack(side='left', padx=(10, 0))
        
        # Buttons
        btn_frame = tk.Frame(frame)
        btn_frame.pack(fill='x', pady=(20, 0))
        
        def on_ok():
            result['key'] = api_key_var.get().strip()
            dialog.destroy()
        
        tb.Button(btn_frame, text="OK", command=on_ok, bootstyle="success").pack(side='right', padx=(5, 0))
        tb.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side='right')
        
        # Focus and bindings
        entry.focus_set()
        dialog.bind('<Return>', lambda e: on_ok())
        dialog.bind('<Escape>', lambda e: dialog.destroy())
        
        # Wait for dialog
        dialog.wait_window()
        
        api_key = result['key']
        
        if api_key:
            try:
                # Save the API key
                self.main_gui.config['replicate_api_key'] = api_key
                self.main_gui.save_config(show_message=False)
                
                # Update UI
                self.inpaint_api_status_label.config(
                    text="✅ Cloud inpainting configured",
                    fg='green'
                )
                
                # Add clear button if it doesn't exist
                clear_button_exists = False
                for widget in self.inpaint_api_status_label.master.winfo_children():
                    if isinstance(widget, tb.Button) and widget.cget('text') == 'Clear':
                        clear_button_exists = True
                        break
                
                if not clear_button_exists:
                    tb.Button(
                        self.inpaint_api_status_label.master,
                        text="Clear",
                        command=self._clear_inpaint_api,
                        bootstyle="secondary"
                    ).pack(side=tk.LEFT, padx=(5, 0))
                
                # Set flag on translator
                if self.translator:
                    self.translator.use_cloud_inpainting = True
                    self.translator.replicate_api_key = api_key
                    
                self._log("✅ Cloud inpainting API configured", "success")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save API key:\n{str(e)}")

    def _clear_inpaint_api(self):
        """Clear the inpainting API configuration"""
        self.main_gui.config['replicate_api_key'] = ''
        self.main_gui.save_config(show_message=False)
        
        self.inpaint_api_status_label.config(
            text="❌ Inpainting API not configured", 
            fg='red'
        )
        
        if hasattr(self, 'translator') and self.translator:
            self.translator.use_cloud_inpainting = False
            self.translator.replicate_api_key = None
            
        self._log("🗑️ Cleared inpainting API configuration", "info")
        
        # Find and destroy the clear button
        for widget in self.inpaint_api_status_label.master.winfo_children():
            if isinstance(widget, tb.Button) and widget.cget('text') == 'Clear':
                widget.destroy()
                break 
            
    def _add_files(self):
        """Add image files (and CBZ archives) to the list"""
        files = filedialog.askopenfilenames(
            title="Select Manga Images or CBZ",
            filetypes=[
                ("Images / CBZ", "*.png *.jpg *.jpeg *.gif *.bmp *.webp *.cbz"),
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.webp"),
                ("Comic Book Zip", "*.cbz"),
                ("All files", "*.*")
            ]
        )
        
        if not files:
            return
        
        # Ensure temp root for CBZ extraction lives for the session
        cbz_temp_root = getattr(self, 'cbz_temp_root', None)
        if cbz_temp_root is None:
            try:
                import tempfile
                cbz_temp_root = tempfile.mkdtemp(prefix='glossarion_cbz_')
                self.cbz_temp_root = cbz_temp_root
            except Exception:
                cbz_temp_root = None
        
        for path in files:
            lower = path.lower()
            if lower.endswith('.cbz'):
                # Extract images from CBZ and add them in natural sort order
                try:
                    import zipfile, shutil
                    base = os.path.splitext(os.path.basename(path))[0]
                    extract_dir = os.path.join(self.cbz_temp_root or os.path.dirname(path), base)
                    os.makedirs(extract_dir, exist_ok=True)
                    with zipfile.ZipFile(path, 'r') as zf:
                        # Extract all to preserve subfolders and avoid name collisions
                        zf.extractall(extract_dir)
                    # Initialize CBZ job tracking
                    if not hasattr(self, 'cbz_jobs'):
                        self.cbz_jobs = {}
                    if not hasattr(self, 'cbz_image_to_job'):
                        self.cbz_image_to_job = {}
                    # Prepare output dir next to source CBZ
                    out_dir = os.path.join(os.path.dirname(path), f"{base}_translated")
                    self.cbz_jobs[path] = {
                        'extract_dir': extract_dir,
                        'out_dir': out_dir,
                    }
                    # Collect all images recursively from extract_dir
                    added = 0
                    for root, _, files_in_dir in os.walk(extract_dir):
                        for fn in sorted(files_in_dir):
                            if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')):
                                target_path = os.path.join(root, fn)
                                if target_path not in self.selected_files:
                                    self.selected_files.append(target_path)
                                    self.file_listbox.insert(tk.END, os.path.basename(target_path))
                                    added += 1
                                # Map extracted image to its CBZ job
                                self.cbz_image_to_job[target_path] = path
                    self._log(f"📦 Added {added} images from CBZ: {os.path.basename(path)}", "info")
                except Exception as e:
                    self._log(f"❌ Failed to read CBZ {os.path.basename(path)}: {e}", "error")
            else:
                if path not in self.selected_files:
                    self.selected_files.append(path)
                    self.file_listbox.insert(tk.END, os.path.basename(path))
    
    def _add_folder(self):
        """Add all images (and CBZ archives) from a folder"""
        folder = filedialog.askdirectory(title="Select Folder with Manga Images or CBZ")
        if not folder:
            return
        
        # Extensions
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        cbz_ext = '.cbz'
        
        # Ensure temp root for CBZ extraction lives for the session
        cbz_temp_root = getattr(self, 'cbz_temp_root', None)
        if cbz_temp_root is None:
            try:
                import tempfile
                cbz_temp_root = tempfile.mkdtemp(prefix='glossarion_cbz_')
                self.cbz_temp_root = cbz_temp_root
            except Exception:
                cbz_temp_root = None
        
        for filename in sorted(os.listdir(folder)):
            filepath = os.path.join(folder, filename)
            if not os.path.isfile(filepath):
                continue
            lower = filename.lower()
            if any(lower.endswith(ext) for ext in image_extensions):
                if filepath not in self.selected_files:
                    self.selected_files.append(filepath)
                    self.file_listbox.insert(tk.END, filename)
            elif lower.endswith(cbz_ext):
                # Extract images from CBZ archive
                try:
                    import zipfile, shutil
                    base = os.path.splitext(os.path.basename(filepath))[0]
                    extract_dir = os.path.join(self.cbz_temp_root or folder, base)
                    os.makedirs(extract_dir, exist_ok=True)
                    with zipfile.ZipFile(filepath, 'r') as zf:
                        zf.extractall(extract_dir)
                    # Initialize CBZ job tracking
                    if not hasattr(self, 'cbz_jobs'):
                        self.cbz_jobs = {}
                    if not hasattr(self, 'cbz_image_to_job'):
                        self.cbz_image_to_job = {}
                    # Prepare output dir next to source CBZ
                    out_dir = os.path.join(os.path.dirname(filepath), f"{base}_translated")
                    self.cbz_jobs[filepath] = {
                        'extract_dir': extract_dir,
                        'out_dir': out_dir,
                    }
                    # Collect all images recursively
                    added = 0
                    for root, _, files_in_dir in os.walk(extract_dir):
                        for fn in sorted(files_in_dir):
                            if fn.lower().endswith(tuple(image_extensions)):
                                target_path = os.path.join(root, fn)
                                if target_path not in self.selected_files:
                                    self.selected_files.append(target_path)
                                    self.file_listbox.insert(tk.END, os.path.basename(target_path))
                                    added += 1
                                # Map extracted image to its CBZ job
                                self.cbz_image_to_job[target_path] = filepath
                    self._log(f"📦 Added {added} images from CBZ: {filename}", "info")
                except Exception as e:
                    self._log(f"❌ Failed to read CBZ {filename}: {e}", "error")
    
    def _remove_selected(self):
        """Remove selected files from the list"""
        selected_indices = list(self.file_listbox.curselection())
        
        # Remove in reverse order to maintain indices
        for index in reversed(selected_indices):
            self.file_listbox.delete(index)
            del self.selected_files[index]
    
    def _clear_all(self):
        """Clear all files from the list"""
        self.file_listbox.delete(0, tk.END)
        self.selected_files.clear()
    
    def _finalize_cbz_jobs(self):
        """Package translated outputs back into .cbz for each imported CBZ.
        - Always creates a CLEAN archive with only final translated pages.
        - If save_intermediate is enabled in settings, also creates a DEBUG archive that
          contains the same final pages at root plus debug/raw artifacts under subfolders.
        """
        try:
            if not hasattr(self, 'cbz_jobs') or not self.cbz_jobs:
                return
            import zipfile
            # Read debug flag from settings
            save_debug = False
            try:
                save_debug = bool(self.main_gui.config.get('manga_settings', {}).get('advanced', {}).get('save_intermediate', False))
            except Exception:
                save_debug = False
            image_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
            text_exts = ('.txt', '.json', '.csv', '.log')
            excluded_patterns = ('_mask', '_overlay', '_debug', '_raw', '_ocr', '_regions', '_chunk', '_clean', '_cleaned', '_inpaint', '_inpainted')

            for cbz_path, job in self.cbz_jobs.items():
                out_dir = job.get('out_dir')
                if not out_dir or not os.path.isdir(out_dir):
                    continue
                parent = os.path.dirname(cbz_path)
                base = os.path.splitext(os.path.basename(cbz_path))[0]

                # Compute original basenames from extracted images mapping
                original_basenames = set()
                try:
                    if hasattr(self, 'cbz_image_to_job'):
                        for img_path, job_path in self.cbz_image_to_job.items():
                            if job_path == cbz_path:
                                original_basenames.add(os.path.basename(img_path))
                except Exception:
                    pass

                # Helper to iterate files in out_dir
                all_files = []
                for root, _, files in os.walk(out_dir):
                    for fn in files:
                        fp = os.path.join(root, fn)
                        rel = os.path.relpath(fp, out_dir)
                        all_files.append((fp, rel, fn))

                # 1) CLEAN ARCHIVE: only final images matching original basenames
                clean_zip = os.path.join(parent, f"{base}_translated.cbz")
                clean_count = 0
                with zipfile.ZipFile(clean_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for fp, rel, fn in all_files:
                        fn_lower = fn.lower()
                        if not fn_lower.endswith(image_exts):
                            continue
                        if original_basenames and fn not in original_basenames:
                            # Only include pages corresponding to original entries
                            continue
                        # Also skip obvious debug artifacts by pattern (extra safeguard)
                        if any(p in fn_lower for p in excluded_patterns):
                            continue
                        zf.write(fp, fn)  # place at root with page filename
                        clean_count += 1
                self._log(f"📦 Compiled CLEAN {clean_count} pages into {os.path.basename(clean_zip)}", "success")

                # 2) DEBUG ARCHIVE: include final pages + extras under subfolders
                if save_debug:
                    debug_zip = os.path.join(parent, f"{base}_translated_debug.cbz")
                    dbg_count = 0
                    raw_count = 0
                    page_count = 0
                    with zipfile.ZipFile(debug_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for fp, rel, fn in all_files:
                            fn_lower = fn.lower()
                            # Final pages at root
                            if fn_lower.endswith(image_exts) and (not original_basenames or fn in original_basenames) and not any(p in fn_lower for p in excluded_patterns):
                                zf.write(fp, fn)
                                page_count += 1
                                continue
                            # Raw text/logs
                            if fn_lower.endswith(text_exts):
                                zf.write(fp, os.path.join('raw', rel))
                                raw_count += 1
                                continue
                            # Other images or artifacts -> debug/
                            zf.write(fp, os.path.join('debug', rel))
                            dbg_count += 1
                    self._log(f"📦 Compiled DEBUG archive: pages={page_count}, debug_files={dbg_count}, raw={raw_count} -> {os.path.basename(debug_zip)}", "info")
        except Exception as e:
            self._log(f"⚠️ Failed to compile CBZ packages: {e}", "warning")

    def _attach_logging_bridge(self):
        """Attach a root logging handler that forwards records into the GUI log."""
        try:
            if getattr(self, '_gui_log_handler', None) is None:
                handler = _MangaGuiLogHandler(self, level=logging.INFO)
                root_logger = logging.getLogger()
                # Avoid duplicates
                if all(not isinstance(h, _MangaGuiLogHandler) for h in root_logger.handlers):
                    root_logger.addHandler(handler)
                self._gui_log_handler = handler
                # Ensure common module loggers propagate
                for name in ['bubble_detector', 'local_inpainter', 'manga_translator']:
                    try:
                        lg = logging.getLogger(name)
                        lg.setLevel(logging.INFO)
                        lg.propagate = True
                    except Exception:
                        pass
        except Exception:
            pass

    def _redirect_stderr(self, enable: bool):
        """Temporarily redirect stderr to the GUI log (captures tqdm/HF progress)."""
        try:
            if enable:
                if not hasattr(self, '_old_stderr') or self._old_stderr is None:
                    self._old_stderr = sys.stderr
                    sys.stderr = _StreamToGuiLog(lambda s: self._log(s, 'info'))
                self._stderr_redirect_on = True
            else:
                if hasattr(self, '_old_stderr') and self._old_stderr is not None:
                    sys.stderr = self._old_stderr
                    self._old_stderr = None
                self._stderr_redirect_on = False
            # Update combined flag to avoid double-forwarding with logging handler
            self._stdio_redirect_active = bool(self._stdout_redirect_on or self._stderr_redirect_on)
        except Exception:
            pass

    def _redirect_stdout(self, enable: bool):
        """Temporarily redirect stdout to the GUI log."""
        try:
            if enable:
                if not hasattr(self, '_old_stdout') or self._old_stdout is None:
                    self._old_stdout = sys.stdout
                    sys.stdout = _StreamToGuiLog(lambda s: self._log(s, 'info'))
                self._stdout_redirect_on = True
            else:
                if hasattr(self, '_old_stdout') and self._old_stdout is not None:
                    sys.stdout = self._old_stdout
                    self._old_stdout = None
                self._stdout_redirect_on = False
            # Update combined flag to avoid double-forwarding with logging handler
            self._stdio_redirect_active = bool(self._stdout_redirect_on or self._stderr_redirect_on)
        except Exception:
            pass

    def _log(self, message: str, level: str = "info"):
        """Log message to GUI text widget or console with enhanced stop suppression"""
        # Enhanced stop suppression - allow only essential stop confirmation messages
        if self._is_stop_requested() or self.is_globally_cancelled():
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
        
        # Lightweight deduplication: ignore identical lines within a short interval
        try:
            now = time.time()
            last_msg = getattr(self, '_last_log_msg', None)
            last_ts = getattr(self, '_last_log_time', 0)
            if last_msg == message and (now - last_ts) < 0.7:
                return
        except Exception:
            pass
            
        # Check if log_text widget exists yet
        if hasattr(self, 'log_text') and self.log_text:
            # Thread-safe logging to GUI
            if threading.current_thread() == threading.main_thread():
                # We're in the main thread, update directly
                try:
                    self.log_text.config(state='normal')
                    self.log_text.insert(tk.END, message + '\n', level)
                    self.log_text.see(tk.END)
                finally:
                    self.log_text.config(state='disabled')
            else:
                # We're in a background thread, use queue
                self.update_queue.put(('log', message, level))
        else:
            # Widget doesn't exist yet or we're in initialization, print to console
            print(message)
        
        # Update deduplication state
        try:
            self._last_log_msg = message
            self._last_log_time = time.time()
        except Exception:
            pass
    
    def _update_progress(self, current: int, total: int, status: str):
        """Thread-safe progress update"""
        self.update_queue.put(('progress', current, total, status))
    
    def _update_current_file(self, filename: str):
        """Thread-safe current file update"""
        self.update_queue.put(('current_file', filename))
    
    def _start_startup_heartbeat(self):
        """Show a small spinner in the progress label during startup so there is no silence."""
        try:
            self._startup_heartbeat_running = True
            self._heartbeat_idx = 0
            chars = ['|', '/', '-', '\\']
            def tick():
                if not getattr(self, '_startup_heartbeat_running', False):
                    return
                try:
                    c = chars[self._heartbeat_idx % len(chars)]
                    if hasattr(self, 'progress_label'):
                        self.progress_label.config(text=f"Starting… {c}", fg='white')
                except Exception:
                    pass
                self._heartbeat_idx += 1
                try:
                    self.parent_frame.after(250, tick)
                except Exception:
                    pass
            # Kick off on main thread
            self.parent_frame.after(0, tick)
        except Exception:
            pass

    def _stop_startup_heartbeat(self):
        try:
            self._startup_heartbeat_running = False
        except Exception:
            pass
    
    def _process_updates(self):
        """Process queued GUI updates"""
        try:
            while True:
                update = self.update_queue.get_nowait()
                
                if update[0] == 'log':
                    _, message, level = update
                    try:
                        self.log_text.config(state='normal')
                        self.log_text.insert(tk.END, message + '\n', level)
                        self.log_text.see(tk.END)
                    finally:
                        self.log_text.config(state='disabled')
                    
                elif update[0] == 'progress':
                    _, current, total, status = update
                    if total > 0:
                        percentage = (current / total) * 100
                        self.progress_bar['value'] = percentage
                    
                    # Check if this is a stopped status and style accordingly
                    if "stopped" in status.lower() or "cancelled" in status.lower():
                        # Make the status more prominent for stopped translations
                        self.progress_label.config(text=f"⏹️ {status}", fg='orange')
                    elif "complete" in status.lower() or "finished" in status.lower():
                        # Success status
                        self.progress_label.config(text=f"✅ {status}", fg='green')
                    elif "error" in status.lower() or "failed" in status.lower():
                        # Error status
                        self.progress_label.config(text=f"❌ {status}", fg='red')
                    else:
                        # Normal status - white for dark mode
                        self.progress_label.config(text=status, fg='white')
                    
                elif update[0] == 'current_file':
                    _, filename = update
                    # Style the current file display based on the status
                    if "stopped" in filename.lower() or "cancelled" in filename.lower():
                        self.current_file_label.config(text=f"⏹️ {filename}", fg='orange')
                    elif "complete" in filename.lower() or "finished" in filename.lower():
                        self.current_file_label.config(text=f"✅ {filename}", fg='green')
                    elif "error" in filename.lower() or "failed" in filename.lower():
                        self.current_file_label.config(text=f"❌ {filename}", fg='red')
                    else:
                        self.current_file_label.config(text=f"Current: {filename}", fg='lightgray')
                    
        except:
            pass
        
        # Schedule next update
        self.parent_frame.after(100, self._process_updates)

    def load_local_inpainting_model(self, model_path):
        """Load a local inpainting model
        
        Args:
            model_path: Path to the model file
            
        Returns:
            bool: True if successful
        """
        try:
            # Store the model path
            self.local_inpaint_model_path = model_path
            
            # If using diffusers/torch models, load them here
            if model_path.endswith('.safetensors') or model_path.endswith('.ckpt'):
                # Initialize your inpainting pipeline
                # This depends on your specific inpainting implementation
                # Example:
                # from diffusers import StableDiffusionInpaintPipeline
                # self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_single_file(model_path)
                pass
                
            return True
        except Exception as e:
            self._log(f"Failed to load inpainting model: {e}", "error")
            return False
            
    def _start_translation(self):
        """Start the translation process"""
        # Mirror console output to GUI during startup for immediate feedback
        self._redirect_stdout(True)
        self._redirect_stderr(True)
        if not self.selected_files:
            messagebox.showwarning("No Files", "Please select manga images to translate.")
            return
        
        # Immediately disable Start to prevent double-clicks
        try:
            if hasattr(self, 'start_button') and self.start_button and self.start_button.winfo_exists():
                self.start_button.config(state=tk.DISABLED)
        except Exception:
            pass
        
        # Clear existing log immediately (main thread) so the next lines are visible
        try:
            if hasattr(self, 'log_text') and self.log_text:
                self.log_text.config(state='normal')
                self.log_text.delete('1.0', tk.END)
                self.log_text.config(state='disabled')
        except Exception:
            pass
        
        # Immediate minimal feedback
        self._log("starting translation", "info")
        try:
            self.dialog.update_idletasks()
        except Exception:
            pass
        # Start heartbeat spinner so there's visible activity until logs stream
        self._start_startup_heartbeat()
        
        # Reset all stop flags at the start of new translation
        self.reset_stop_flags()
        self._log("🚀 Starting new manga translation batch", "info")
        try:
            # Let the GUI render the above log immediately
            self.dialog.update_idletasks()
        except Exception:
            pass
        
        # Run the heavy preparation and kickoff in a background thread to avoid GUI freeze
        threading.Thread(target=self._start_translation_heavy, name="MangaStartHeavy", daemon=True).start()
        return
    
    def _start_translation_heavy(self):
        """Heavy part of start: build configs, init client/translator, and launch worker (runs off-main-thread)."""
        # Early feedback
        self._log("⏳ Preparing configuration...", "info")
        # Build OCR configuration
        ocr_config = {'provider': self.ocr_provider_var.get()}

        if ocr_config['provider'] == 'Qwen2-VL':
            qwen_provider = self.ocr_manager.get_provider('Qwen2-VL')
            if qwen_provider:
                # Set model size configuration
                if hasattr(qwen_provider, 'loaded_model_size'):
                    if qwen_provider.loaded_model_size == "Custom":
                        ocr_config['model_size'] = f"custom:{qwen_provider.model_id}"
                    else:
                        size_map = {'2B': '1', '7B': '2', '72B': '3'}
                        ocr_config['model_size'] = size_map.get(qwen_provider.loaded_model_size, '2')
                    self._log(f"Setting ocr_config['model_size'] = {ocr_config['model_size']}", "info")
                
                # Set OCR prompt if available
                if hasattr(self, 'ocr_prompt'):
                    # Set it via environment variable (Qwen2VL will read this)
                    os.environ['OCR_SYSTEM_PROMPT'] = self.ocr_prompt
                    
                    # Also set it directly on the provider if it has the method
                    if hasattr(qwen_provider, 'set_ocr_prompt'):
                        qwen_provider.set_ocr_prompt(self.ocr_prompt)
                    else:
                        # If no setter method, set it directly
                        qwen_provider.ocr_prompt = self.ocr_prompt
                    
                    self._log("✅ Set custom OCR prompt for Qwen2-VL", "info")
       
        elif ocr_config['provider'] == 'google':
            import os
            google_creds = self.main_gui.config.get('google_vision_credentials', '') or self.main_gui.config.get('google_cloud_credentials', '')
            if not google_creds or not os.path.exists(google_creds):
                try:
                    self.dialog.after(0, lambda: messagebox.showerror("Error", "Google Cloud Vision credentials not found.\nPlease set up credentials in the main settings."))
                except Exception:
                    pass
                try:
                    self.parent_frame.after(0, self._stop_startup_heartbeat)
                    self.parent_frame.after(0, self._reset_ui_state)
                except Exception:
                    pass
                return
            ocr_config['google_credentials_path'] = google_creds
            
        elif ocr_config['provider'] == 'azure':
            azure_key = self.azure_key_entry.get().strip()
            azure_endpoint = self.azure_endpoint_entry.get().strip()
            
            if not azure_key or not azure_endpoint:
                try:
                    self.dialog.after(0, lambda: messagebox.showerror("Error", "Azure credentials not configured."))
                except Exception:
                    pass
                try:
                    self.parent_frame.after(0, self._stop_startup_heartbeat)
                    self.parent_frame.after(0, self._reset_ui_state)
                except Exception:
                    pass
                return
            
            # Save Azure settings
            self.main_gui.config['azure_vision_key'] = azure_key
            self.main_gui.config['azure_vision_endpoint'] = azure_endpoint
            if hasattr(self.main_gui, 'save_config'):
                self.main_gui.save_config(show_message=False)
            
            ocr_config['azure_key'] = azure_key
            ocr_config['azure_endpoint'] = azure_endpoint
        
        # Get current API key and model for translation
        api_key = None
        model = 'gemini-2.5-flash'  # default
        
        # Try to get API key from various sources
        if hasattr(self.main_gui, 'api_key_entry') and self.main_gui.api_key_entry.get().strip():
            api_key = self.main_gui.api_key_entry.get().strip()
        elif hasattr(self.main_gui, 'config') and self.main_gui.config.get('api_key'):
            api_key = self.main_gui.config.get('api_key')
        
        # Try to get model - ALWAYS get the current selection from GUI
        if hasattr(self.main_gui, 'model_var'):
            model = self.main_gui.model_var.get()
        elif hasattr(self.main_gui, 'config') and self.main_gui.config.get('model'):
            model = self.main_gui.config.get('model')
        
        if not api_key:
            try:
                self.dialog.after(0, lambda: messagebox.showerror("Error", "API key not found.\nPlease configure your API key in the main settings."))
            except Exception:
                pass
            try:
                self.parent_frame.after(0, self._stop_startup_heartbeat)
                self.parent_frame.after(0, self._reset_ui_state)
            except Exception:
                pass
            return
        
        # Check if we need to create or update the client
        needs_new_client = False
        self._log("🔎 Checking API client...", "debug")
        
        if not hasattr(self.main_gui, 'client') or not self.main_gui.client:
            needs_new_client = True
            self._log(f"🛠 Creating new API client with model: {model}", "info")
        elif hasattr(self.main_gui.client, 'model') and self.main_gui.client.model != model:
            needs_new_client = True
            self._log(f"🛠 Model changed from {self.main_gui.client.model} to {model}, creating new client", "info")
        else:
            self._log("♻️ Reusing existing API client", "debug")
        
        if needs_new_client:
            # Apply multi-key settings from config so UnifiedClient picks them up
            try:
                import os  # Import os here
                use_mk = bool(self.main_gui.config.get('use_multi_api_keys', False))
                mk_list = self.main_gui.config.get('multi_api_keys', [])
                if use_mk and mk_list:
                    os.environ['USE_MULTI_API_KEYS'] = '1'
                    os.environ['USE_MULTI_KEYS'] = '1'  # backward-compat for retry paths
                    os.environ['MULTI_API_KEYS'] = json.dumps(mk_list)
                    os.environ['FORCE_KEY_ROTATION'] = '1' if self.main_gui.config.get('force_key_rotation', True) else '0'
                    os.environ['ROTATION_FREQUENCY'] = str(self.main_gui.config.get('rotation_frequency', 1))
                    self._log("🔑 Multi-key mode ENABLED for manga translator", "info")
                else:
                    # Explicitly disable if not configured
                    os.environ['USE_MULTI_API_KEYS'] = '0'
                    os.environ['USE_MULTI_KEYS'] = '0'
                # Fallback keys (optional)
                if self.main_gui.config.get('use_fallback_keys', False):
                    os.environ['USE_FALLBACK_KEYS'] = '1'
                    os.environ['FALLBACK_KEYS'] = json.dumps(self.main_gui.config.get('fallback_keys', []))
                else:
                    os.environ['USE_FALLBACK_KEYS'] = '0'
                    os.environ['FALLBACK_KEYS'] = '[]'
            except Exception as env_err:
                self._log(f"⚠️ Failed to apply multi-key settings: {env_err}", "warning")
            
            # Create the unified client with the current model
            try:
                from unified_api_client import UnifiedClient
                self._log("⏳ Creating API client (network/model handshake)...", "debug")
                self.main_gui.client = UnifiedClient(model=model, api_key=api_key)
                self._log(f"✅ API client ready (model: {model})", "info")
                try:
                    time.sleep(0.05)
                except Exception:
                    pass
            except Exception as e:
                try:
                    self.dialog.after(0, lambda e=e: messagebox.showerror("Error", f"Failed to create API client:\n{str(e)}"))
                except Exception:
                    pass
                try:
                    self.parent_frame.after(0, self._stop_startup_heartbeat)
                    self.parent_frame.after(0, self._reset_ui_state)
                except Exception:
                    pass
                return
        
        # Reset the translator's history manager for new batch
        if hasattr(self, 'translator') and self.translator and hasattr(self.translator, 'reset_history_manager'):
            self.translator.reset_history_manager()

        # Set environment variables for custom-api provider
        if ocr_config['provider'] == 'custom-api':
            import os  # Import os for environment variables
            env_vars = self.main_gui._get_environment_variables(
                epub_path='',  # Not needed for manga
                api_key=api_key
            )
            
            # Apply all environment variables EXCEPT SYSTEM_PROMPT
            for key, value in env_vars.items():
                if key == 'SYSTEM_PROMPT':
                    # DON'T SET THE TRANSLATION SYSTEM PROMPT FOR OCR
                    continue
                os.environ[key] = str(value)
            
            # Set a VERY EXPLICIT OCR prompt that OpenAI can't ignore
            os.environ['OCR_SYSTEM_PROMPT'] = (
            "YOU ARE AN OCR SYSTEM. YOUR ONLY JOB IS TEXT EXTRACTION.\n\n"
            "CRITICAL RULES:\n"
            "1. DO NOT TRANSLATE ANYTHING\n"
            "2. DO NOT MODIFY THE TEXT\n"
            "3. DO NOT EXPLAIN OR COMMENT\n"
            "4. ONLY OUTPUT THE EXACT TEXT YOU SEE\n"
            "5. PRESERVE NATURAL TEXT FLOW - DO NOT ADD UNNECESSARY LINE BREAKS\n\n"
            "If you see Korean text, output it in Korean.\n"
            "If you see Japanese text, output it in Japanese.\n"
            "If you see Chinese text, output it in Chinese.\n"
            "If you see English text, output it in English.\n\n"
            "IMPORTANT: Only use line breaks where they naturally occur in the original text "
            "(e.g., between dialogue lines or paragraphs). Do not break text mid-sentence or "
            "between every word/character.\n\n"
            "For vertical text common in manga/comics, transcribe it as a continuous line unless "
            "there are clear visual breaks.\n\n"
            "NEVER translate. ONLY extract exactly what is written.\n"
            "Output ONLY the raw text, nothing else."
            )
            
            self._log("✅ Set environment variables for custom-api OCR (excluded SYSTEM_PROMPT)")

            # Respect user settings: only set default detector values when bubble detection is OFF.
            try:
                ms = self.main_gui.config.setdefault('manga_settings', {})
                ocr_set = ms.setdefault('ocr', {})
                changed = False
                bubble_enabled = bool(ocr_set.get('bubble_detection_enabled', False))

                if not bubble_enabled:
                    # User has bubble detection OFF -> set non-intrusive defaults only
                    if 'detector_type' not in ocr_set:
                        ocr_set['detector_type'] = 'rtdetr_onnx'
                        changed = True
                    if not ocr_set.get('rtdetr_model_url') and not ocr_set.get('bubble_model_path'):
                        # Default HF repo (detector.onnx lives here)
                        ocr_set['rtdetr_model_url'] = 'ogkalu/comic-text-and-bubble-detector'
                        changed = True
                    if changed and hasattr(self.main_gui, 'save_config'):
                        self.main_gui.save_config(show_message=False)
                    # Do NOT preload when detection is off
                    self._preloaded_bd = None
                else:
                    # Bubble detection is ON → do not override user's detector choice
                    # Optionally warm up based on current detector type
                    detector_type = str(ocr_set.get('detector_type', '')).lower()
                    try:
                        from bubble_detector import BubbleDetector
                        self._preloaded_bd = BubbleDetector()
                        if detector_type == 'rtdetr_onnx':
                            self._log("📥 Warming up RTEDR_onnx for custom-api OCR", "info")
                            model_id = ocr_set.get('rtdetr_model_url') or ocr_set.get('bubble_model_path')
                            self._preloaded_bd.load_rtdetr_onnx_model(model_id=model_id)
                            # Stagger subsequent heavy initializations to prevent CPU spike
                            try:
                                import time as _time
                                _time.sleep(1.0)
                            except Exception:
                                pass
                        elif detector_type == 'rtdetr':
                            self._log("📥 Warming up RT-DETR (PyTorch) for custom-api OCR", "info")
                            model_id = ocr_set.get('rtdetr_model_url') or ocr_set.get('bubble_model_path')
                            self._preloaded_bd.load_rtdetr_model(model_id=model_id)
                            try:
                                import time as _time
                                _time.sleep(1.0)
                            except Exception:
                                pass
                        else:
                            # YOLO or custom – no preload here
                            self._preloaded_bd = None
                    except Exception:
                        self._preloaded_bd = None
            except Exception:
                self._preloaded_bd = None
        
        # Initialize translator if needed (or if it was reset or client was cleared during shutdown)
        needs_new_translator = (not hasattr(self, 'translator')) or (self.translator is None)
        if not needs_new_translator:
            try:
                needs_new_translator = getattr(self.translator, 'client', None) is None
                if needs_new_translator:
                    self._log("♻️ Translator exists but client was cleared — reinitializing translator", "debug")
            except Exception:
                needs_new_translator = True
        if needs_new_translator:
            self._log("⚙️ Initializing translator...", "info")
            try:
                self.translator = MangaTranslator(
                    ocr_config,
                    self.main_gui.client,
                    self.main_gui,
                    log_callback=self._log
                )
                
                # Fix 4: Safely set OCR manager
                if hasattr(self, 'ocr_manager'):
                    self.translator.ocr_manager = self.ocr_manager
                else:
                    from ocr_manager import OCRManager
                    self.ocr_manager = OCRManager(log_callback=self._log)
                    self.translator.ocr_manager = self.ocr_manager
                    
                    # Attach preloaded RT-DETR if available
                    try:
                        if hasattr(self, '_preloaded_bd') and self._preloaded_bd:
                            self.translator.bubble_detector = self._preloaded_bd
                            self._log("🤖 RT-DETR preloaded and attached to translator", "debug")
                    except Exception:
                        pass
                    
                    # Distribute stop flags to all components
                    self._distribute_stop_flags()
                    
                    # Set cloud inpainting if configured (only once!)
                saved_api_key = self.main_gui.config.get('replicate_api_key', '')
                if saved_api_key:
                    self.translator.use_cloud_inpainting = True
                    self.translator.replicate_api_key = saved_api_key
                
                # Apply text rendering settings
                self._apply_rendering_settings()
                
                try:
                    time.sleep(0.05)
                except Exception:
                    pass
                self._log("✅ Translator ready", "info")
                
            except Exception as e:
                try:
                    self.dialog.after(0, lambda e=e: messagebox.showerror("Error", f"Failed to initialize translator:\n{str(e)}"))
                except Exception:
                    pass
                self._log(f"Initialization error: {str(e)}", "error")
                import traceback
                self._log(traceback.format_exc(), "error")
                try:
                    self.parent_frame.after(0, self._stop_startup_heartbeat)
                    self.parent_frame.after(0, self._reset_ui_state)
                except Exception:
                    pass
                return
        else:
            # Update the translator with the new client if model changed
            if needs_new_client and hasattr(self.translator, 'client'):
                self.translator.client = self.main_gui.client
                self._log(f"Updated translator with new API client", "info")
            
            # Distribute stop flags to all components
            self._distribute_stop_flags()
            
            # Update rendering settings
            self._apply_rendering_settings()
            
            # Ensure inpainting settings are properly synchronized
            if hasattr(self, 'inpainting_mode_var'):
                inpainting_mode = self.inpainting_mode_var.get()
                
                if inpainting_mode == 'skip':
                    self.translator.skip_inpainting = True
                    self.translator.use_cloud_inpainting = False
                    self._log("Inpainting: SKIP", "debug")
                    
                elif inpainting_mode == 'local':
                    self.translator.skip_inpainting = False
                    self.translator.use_cloud_inpainting = False
                    
                    # IMPORTANT: Load the local inpainting model if not already loaded
                    if hasattr(self, 'local_model_var'):
                        selected_model = self.local_model_var.get()
                        if selected_model and selected_model != "None":
                            # Get model path from available models
                            model_info = self.available_local_models.get(selected_model)
                            if model_info:
                                model_path = model_info['path']
                                # Load the model into translator
                                if hasattr(self.translator, 'load_local_inpainting_model'):
                                    success = self.translator.load_local_inpainting_model(model_path)
                                    if success:
                                        self._log(f"Inpainting: LOCAL - Loaded {selected_model}", "info")
                                    else:
                                        self._log(f"Inpainting: Failed to load local model {selected_model}", "error")
                                else:
                                    # Set the model path directly if no load method
                                    self.translator.local_inpaint_model_path = model_path
                                    self._log(f"Inpainting: LOCAL - Set model path for {selected_model}", "info")
                            else:
                                self._log("Inpainting: LOCAL - No model selected", "warning")
                        else:
                            self._log("Inpainting: LOCAL - No model configured", "warning")
                    else:
                        self._log("Inpainting: LOCAL (default)", "debug")
                    
                elif inpainting_mode == 'cloud':
                    self.translator.skip_inpainting = False
                    saved_api_key = self.main_gui.config.get('replicate_api_key', '')
                    if saved_api_key:
                        self.translator.use_cloud_inpainting = True
                        self.translator.replicate_api_key = saved_api_key
                        self._log("Inpainting: CLOUD (Replicate)", "debug")
                    else:
                        # Fallback to local if no API key
                        self.translator.use_cloud_inpainting = False
                        self._log("Inpainting: LOCAL (no Replicate key, fallback)", "warning")
            else:
                # Default to local inpainting if variable doesn't exist
                self.translator.skip_inpainting = False
                self.translator.use_cloud_inpainting = False
                self._log("Inpainting: LOCAL (default)", "debug")

            # Double-check the settings are applied correctly
            self._log(f"Inpainting final status:", "debug")
            self._log(f"  - Skip: {self.translator.skip_inpainting}", "debug")
            self._log(f"  - Cloud: {self.translator.use_cloud_inpainting}", "debug")
            self._log(f"  - Mode: {'SKIP' if self.translator.skip_inpainting else 'CLOUD' if self.translator.use_cloud_inpainting else 'LOCAL'}", "debug")
        
        # Preflight RT-DETR to avoid first-page fallback after aggressive cleanup
        try:
            ocr_set = self.main_gui.config.get('manga_settings', {}).get('ocr', {}) or {}
            if ocr_set.get('bubble_detection_enabled', False):
                # Ensure a default RT-DETR model id exists when required
                if ocr_set.get('detector_type', 'rtdetr') in ('rtdetr', 'auto'):
                    if not ocr_set.get('rtdetr_model_url') and not ocr_set.get('bubble_model_path'):
                        ocr_set['rtdetr_model_url'] = 'ogkalu/comic-text-and-bubble-detector'
                        if hasattr(self.main_gui, 'save_config'):
                            self.main_gui.save_config(show_message=False)
                self._preflight_bubble_detector(ocr_set)
        except Exception:
            pass
        
        # Reset progress
        self.total_files = len(self.selected_files)
        self.completed_files = 0
        self.failed_files = 0
        self.current_file_index = 0
        
        # Reset all global cancellation flags for new translation
        self._reset_global_cancellation()
        
        # Update UI state (schedule on main thread)
        self.is_running = True
        self.stop_flag.clear()
        if threading.current_thread() == threading.main_thread():
            try:
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                self.file_listbox.config(state=tk.DISABLED)
            except Exception:
                pass
        else:
            try:
                self.parent_frame.after(0, lambda: self.start_button.config(state=tk.DISABLED))
                self.parent_frame.after(0, lambda: self.stop_button.config(state=tk.NORMAL))
                self.parent_frame.after(0, lambda: self.file_listbox.config(state=tk.DISABLED))
            except Exception:
                pass
        
        # Log start message
        self._log(f"Starting translation of {self.total_files} files...", "info")
        self._log(f"Using OCR provider: {ocr_config['provider'].upper()}", "info")
        if ocr_config['provider'] == 'google':
            self._log(f"Using Google Vision credentials: {os.path.basename(ocr_config['google_credentials_path'])}", "info")
        elif ocr_config['provider'] == 'azure':
            self._log(f"Using Azure endpoint: {ocr_config['azure_endpoint']}", "info")
        else:
            self._log(f"Using local OCR provider: {ocr_config['provider'].upper()}", "info")
            # Report effective API routing/model with multi-key awareness
            try:
                c = getattr(self.main_gui, 'client', None)
                if c is not None:
                    if getattr(c, 'use_multi_keys', False):
                        total_keys = 0
                        try:
                            stats = c.get_stats()
                            total_keys = stats.get('total_keys', 0)
                        except Exception:
                            pass
                        self._log(
                            f"API routing: Multi-key pool enabled — starting model '{getattr(c, 'model', 'unknown')}', keys={total_keys}, rotation={getattr(c, '_rotation_frequency', 1)}",
                            "info"
                        )
                    else:
                        self._log(f"API model: {getattr(c, 'model', 'unknown')}", "info")
            except Exception:
                pass
            self._log(f"Contextual: {'Enabled' if self.main_gui.contextual_var.get() else 'Disabled'}", "info")
            self._log(f"History limit: {self.main_gui.trans_history.get()} exchanges", "info")
            self._log(f"Rolling history: {'Enabled' if self.main_gui.translation_history_rolling_var.get() else 'Disabled'}", "info")
            self._log(f"  Full Page Context: {'Enabled' if self.full_page_context_var.get() else 'Disabled'}", "info")
        
        # Stop heartbeat before launching worker; now regular progress takes over
        try:
            self.parent_frame.after(0, self._stop_startup_heartbeat)
        except Exception:
            pass
        
        # Start translation via executor
        try:
            # Sync with main GUI executor if possible and update EXTRACTION_WORKERS
            if hasattr(self.main_gui, '_ensure_executor'):
                self.main_gui._ensure_executor()
                self.executor = self.main_gui.executor
            # Ensure env var reflects current worker setting from main GUI
            try:
                os.environ["EXTRACTION_WORKERS"] = str(self.main_gui.extraction_workers_var.get())
            except Exception:
                pass
            
            if self.executor:
                self.translation_future = self.executor.submit(self._translation_worker)
            else:
                # Fallback to dedicated thread
                self.translation_thread = threading.Thread(
                    target=self._translation_worker,
                    daemon=True
                )
                self.translation_thread.start()
        except Exception:
            # Last resort fallback to thread
            self.translation_thread = threading.Thread(
                target=self._translation_worker,
                daemon=True
            )
            self.translation_thread.start()
    
    def _apply_rendering_settings(self):
        """Apply current rendering settings to translator"""
        if self.translator:
            # Get text color as tuple
            text_color = (
                self.text_color_r.get(),
                self.text_color_g.get(),
                self.text_color_b.get()
            )
            
            # Get shadow color as tuple
            shadow_color = (
                self.shadow_color_r.get(),
                self.shadow_color_g.get(),
                self.shadow_color_b.get()
            )
            
            # Determine font size value based on mode
            if self.font_size_mode_var.get() == 'multiplier':
                # Pass negative value to indicate multiplier mode
                font_size = -self.font_size_multiplier_var.get()
            else:
                # Fixed mode - use the font size value directly
                font_size = self.font_size_var.get() if self.font_size_var.get() > 0 else None
            
            self.translator.update_text_rendering_settings(
                bg_opacity=self.bg_opacity_var.get(),
                bg_style=self.bg_style_var.get(),
                bg_reduction=self.bg_reduction_var.get(),
                font_style=self.selected_font_path,
                font_size=font_size,
                text_color=text_color,
                shadow_enabled=self.shadow_enabled_var.get(),
                shadow_color=shadow_color,
                shadow_offset_x=self.shadow_offset_x_var.get(),
                shadow_offset_y=self.shadow_offset_y_var.get(),
                shadow_blur=self.shadow_blur_var.get(),
                force_caps_lock=self.force_caps_lock_var.get()
            )

            # Free-text-only background opacity toggle -> pass through to translator
            try:
                if hasattr(self, 'free_text_only_bg_opacity_var'):
                    self.translator.free_text_only_bg_opacity = bool(self.free_text_only_bg_opacity_var.get())
            except Exception:
                pass
            
            # Update font mode and multiplier explicitly
            self.translator.font_size_mode = self.font_size_mode_var.get()
            self.translator.font_size_multiplier = self.font_size_multiplier_var.get()
            self.translator.min_readable_size = self.min_readable_size_var.get()
            self.translator.max_font_size_limit = self.max_font_size_var.get()
            self.translator.strict_text_wrapping = self.strict_text_wrapping_var.get()
            self.translator.force_caps_lock = self.force_caps_lock_var.get()
            
            # Update constrain to bubble setting
            if hasattr(self, 'constrain_to_bubble_var'):
                self.translator.constrain_to_bubble = self.constrain_to_bubble_var.get()
            
            # Handle inpainting mode (3 radio buttons: skip/local/cloud)
            if hasattr(self, 'inpainting_mode_var'):
                mode = self.inpainting_mode_var.get()
                
                if mode == 'skip':
                    self.translator.skip_inpainting = True
                    self.translator.use_cloud_inpainting = False
                    self._log("  Inpainting: Skipped", "info")
                elif mode == 'local':
                    self.translator.skip_inpainting = False
                    self.translator.use_cloud_inpainting = False
                    self._log("  Inpainting: Local", "info")
                elif mode == 'cloud':
                    self.translator.skip_inpainting = False
                    saved_api_key = self.main_gui.config.get('replicate_api_key', '')
                    if saved_api_key:
                        self.translator.use_cloud_inpainting = True
                        self.translator.replicate_api_key = saved_api_key
                        self._log("  Inpainting: Cloud (Replicate)", "info")
                    else:
                        # Fallback to local if no API key
                        self.translator.use_cloud_inpainting = False
                        self._log("  Inpainting: Local (no Replicate key, fallback)", "warning")
            
            # Set full page context mode
            self.translator.set_full_page_context(
                enabled=self.full_page_context_var.get(),
                custom_prompt=self.full_page_context_prompt
            )
            
        # Update logging to include new settings
            # Persist free-text-only BG opacity setting to config as well
            try:
                if hasattr(self, 'free_text_only_bg_opacity_var'):
                    self.main_gui.config['manga_free_text_only_bg_opacity'] = bool(self.free_text_only_bg_opacity_var.get())
            except Exception:
                pass

            self._log(f"Applied rendering settings:", "info")
            self._log(f"  Background: {self.bg_style_var.get()} @ {int(self.bg_opacity_var.get()/255*100)}% opacity", "info")
            self._log(f"  Font: {os.path.basename(self.selected_font_path) if self.selected_font_path else 'Default'}", "info")
            self._log(f"  Minimum Font Size: {self.min_readable_size_var.get()}pt", "info")
            self._log(f"  Maximum Font Size: {self.max_font_size_var.get()}pt", "info")
            self._log(f"  Strict Text Wrapping: {'Enabled (force fit)' if self.strict_text_wrapping_var.get() else 'Disabled (allow overflow)'}", "info")
            
            # Log font size mode
            if self.font_size_mode_var.get() == 'multiplier':
                self._log(f"  Font Size: Dynamic multiplier ({self.font_size_multiplier_var.get():.1f}x)", "info")
                if hasattr(self, 'constrain_to_bubble_var'):
                    constraint_status = "constrained" if self.constrain_to_bubble_var.get() else "unconstrained"
                    self._log(f"  Text Constraint: {constraint_status}", "info")
            else:
                size_text = f"{self.font_size_var.get()}pt" if self.font_size_var.get() > 0 else "Auto"
                self._log(f"  Font Size: Fixed ({size_text})", "info")
            
            self._log(f"  Text Color: RGB({text_color[0]}, {text_color[1]}, {text_color[2]})", "info")
            self._log(f"  Shadow: {'Enabled' if self.shadow_enabled_var.get() else 'Disabled'}", "info")
            try:
                self._log(f"  Free-text-only BG opacity: {'Enabled' if getattr(self, 'free_text_only_bg_opacity_var').get() else 'Disabled'}", "info")
            except Exception:
                pass
            self._log(f"  Full Page Context: {'Enabled' if self.full_page_context_var.get() else 'Disabled'}", "info")
    
    def _translation_worker(self):
        """Worker thread for translation"""
        try:
            # Defensive: ensure translator exists before using it (legacy callers may start this worker early)
            if not hasattr(self, 'translator') or self.translator is None:
                self._log("⚠️ Translator not initialized yet; skipping worker start", "warning")
                return
            if hasattr(self.translator, 'set_stop_flag'):
                self.translator.set_stop_flag(self.stop_flag)
            
            # Ensure API parallelism (batch API calls) is controlled independently of local parallel processing.
            # Propagate the GUI "Batch Translation" toggle into environment so Unified API Client applies it globally
            # for all providers (including custom endpoints).
            try:
                import os as _os
                _os.environ['BATCH_TRANSLATION'] = '1' if getattr(self.main_gui, 'batch_translation_var', None) and self.main_gui.batch_translation_var.get() else '0'
                # Use GUI batch size if available; default to 3 to match existing default
                bs_val = None
                try:
                    bs_val = str(int(self.main_gui.batch_size_var.get())) if hasattr(self.main_gui, 'batch_size_var') else None
                except Exception:
                    bs_val = None
                _os.environ['BATCH_SIZE'] = bs_val or _os.environ.get('BATCH_SIZE', '3')
            except Exception:
                # Non-fatal if env cannot be set
                pass
            
            # Panel-level parallelization setting (LOCAL threading for panels)
            advanced = self.main_gui.config.get('manga_settings', {}).get('advanced', {})
            panel_parallel = bool(advanced.get('parallel_panel_translation', False))
            requested_panel_workers = int(advanced.get('panel_max_workers', 2))

            # Decouple from global parallel processing: panel concurrency is governed ONLY by panel settings
            effective_workers = requested_panel_workers if (panel_parallel and len(self.selected_files) > 1) else 1

            if panel_parallel and len(self.selected_files) > 1 and effective_workers > 1:
                self._log(f"🚀 Parallel PANEL translation ENABLED ({effective_workers} workers)", "info")
                
                import concurrent.futures
                import threading as _threading
                progress_lock = _threading.Lock()
                counters = {
                    'started': 0,
                    'done': 0,
                    'failed': 0
                }
                total = self.total_files
                
                def process_single(idx, filepath):
                    # Check stop flag at the very beginning
                    if self.stop_flag.is_set():
                        return False
                    
                    # Create an isolated translator instance per panel
                    try:
                        # Check again before starting expensive work
                        if self.stop_flag.is_set():
                            return False
                        from manga_translator import MangaTranslator
                        import os
                        # Build full OCR config for this thread (mirror _start_translation)
                        ocr_config = {'provider': self.ocr_provider_var.get()}
                        if ocr_config['provider'] == 'google':
                            google_creds = self.main_gui.config.get('google_vision_credentials', '') or \
                                           self.main_gui.config.get('google_cloud_credentials', '')
                            if google_creds and os.path.exists(google_creds):
                                ocr_config['google_credentials_path'] = google_creds
                            else:
                                self._log("⚠️ Google Cloud Vision credentials not found for parallel task", "warning")
                        elif ocr_config['provider'] == 'azure':
                            azure_key = self.main_gui.config.get('azure_vision_key', '')
                            azure_endpoint = self.main_gui.config.get('azure_vision_endpoint', '')
                            if azure_key and azure_endpoint:
                                ocr_config['azure_key'] = azure_key
                                ocr_config['azure_endpoint'] = azure_endpoint
                            else:
                                self._log("⚠️ Azure credentials not found for parallel task", "warning")

                        translator = MangaTranslator(ocr_config, self.main_gui.client, self.main_gui, log_callback=self._log)
                        translator.set_stop_flag(self.stop_flag)
                        
                        # Also propagate global cancellation to isolated translator
                        from manga_translator import MangaTranslator as MTClass
                        if MTClass.is_globally_cancelled():
                            return False
                        
                        # Check stop flag before configuration
                        if self.stop_flag.is_set():
                            return False
                            
                        # Apply inpainting and rendering options roughly matching current translator
                        try:
                            translator.constrain_to_bubble = getattr(self, 'constrain_to_bubble_var').get() if hasattr(self, 'constrain_to_bubble_var') else True
                        except Exception:
                            pass
                        
                        # Set full page context based on UI
                        try:
                            translator.set_full_page_context(
                                enabled=self.full_page_context_var.get(),
                                custom_prompt=self.full_page_context_prompt
                            )
                        except Exception:
                            pass
                            
                        # Another check before path setup
                        if self.stop_flag.is_set():
                            return False
                        
                        # Determine output path (route CBZ images to job out_dir)
                        filename = os.path.basename(filepath)
                        output_path = None
                        try:
                            if hasattr(self, 'cbz_image_to_job') and filepath in self.cbz_image_to_job:
                                cbz_file = self.cbz_image_to_job[filepath]
                                job = getattr(self, 'cbz_jobs', {}).get(cbz_file)
                                if job:
                                    output_dir = job.get('out_dir')
                                    os.makedirs(output_dir, exist_ok=True)
                                    output_path = os.path.join(output_dir, filename)
                        except Exception:
                            output_path = None
                        if not output_path:
                            if self.create_subfolder_var.get():
                                output_dir = os.path.join(os.path.dirname(filepath), 'translated')
                                os.makedirs(output_dir, exist_ok=True)
                                output_path = os.path.join(output_dir, filename)
                            else:
                                base, ext = os.path.splitext(filepath)
                                output_path = f"{base}_translated{ext}"
                        
                        # Announce start
                        self._update_current_file(filename)
                        with progress_lock:
                            counters['started'] += 1
                            self._update_progress(counters['done'], total, f"Processing {counters['started']}/{total}: {filename}")
                        
                        # Final check before expensive processing
                        if self.stop_flag.is_set():
                            return False
                            
                        # Process image
                        result = translator.process_image(filepath, output_path, batch_index=idx+1, batch_total=total)
                        
                        # Update counters only if not stopped
                        with progress_lock:
                            if self.stop_flag.is_set():
                                # Don't update counters if translation was stopped
                                return False
                            
                            # Check if translation actually produced valid output
                            translation_successful = False
                            if result.get('success', False) and not result.get('interrupted', False):
                                # Verify there's an actual output file and translated regions
                                output_exists = result.get('output_path') and os.path.exists(result.get('output_path', ''))
                                has_translations = any(r.get('translated_text', '') for r in result.get('regions', []))
                                translation_successful = output_exists and has_translations
                            
                            if translation_successful:
                                self.completed_files += 1
                                self._log(f"✅ Translation completed: {filename}", "success")
                                time.sleep(0.1)  # Brief pause for stability
                                self._log("💤 Panel completion pausing briefly for stability", "debug")
                            else:
                                self.failed_files += 1
                                # Log the specific reason for failure
                                if result.get('interrupted', False):
                                    self._log(f"⚠️ Translation interrupted: {filename}", "warning")
                                elif not result.get('success', False):
                                    self._log(f"❌ Translation failed: {filename}", "error")
                                elif not result.get('output_path') or not os.path.exists(result.get('output_path', '')):
                                    self._log(f"❌ Output file not created: {filename}", "error")
                                else:
                                    self._log(f"❌ No text was translated: {filename}", "error")
                                counters['failed'] += 1
                            counters['done'] += 1
                            self._update_progress(counters['done'], total, f"Completed {counters['done']}/{total}")
                        
                        return result.get('success', False)
                    except Exception as e:
                        with progress_lock:
                            # Don't update error counters if stopped
                            if not self.stop_flag.is_set():
                                self.failed_files += 1
                                counters['failed'] += 1
                                counters['done'] += 1
                        if not self.stop_flag.is_set():
                            self._log(f"❌ Error in panel task: {str(e)}", "error")
                            self._log(traceback.format_exc(), "error")
                        return False
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, effective_workers)) as executor:
                    futures = []
                    stagger_ms = int(advanced.get('panel_start_stagger_ms', 100))
                    for idx, filepath in enumerate(self.selected_files):
                        if self.stop_flag.is_set():
                            break
                        futures.append(executor.submit(process_single, idx, filepath))
                        if stagger_ms > 0:
                            time.sleep(stagger_ms / 1000.0)
                            time.sleep(0.1)  # Brief pause for stability
                            self._log("💤 Staggered submission pausing briefly for stability", "debug")
                    
                    # Handle completion and stop behavior
                    try:
                        for f in concurrent.futures.as_completed(futures):
                            if self.stop_flag.is_set():
                                # More aggressive cancellation
                                for rem in futures:
                                    rem.cancel()
                                # Try to shutdown executor immediately
                                try:
                                    executor.shutdown(wait=False)
                                except Exception:
                                    pass
                                break
                            try:
                                # Consume future result to let it raise exceptions or return
                                f.result(timeout=0.1)  # Very short timeout
                            except Exception:
                                # Ignore; counters are updated inside process_single
                                pass
                    except Exception:
                        # If as_completed fails due to shutdown, that's ok
                        pass
                    
                    # If stopped during parallel processing, do not log panel completion
                    if self.stop_flag.is_set():
                        pass
                    else:
                        # After parallel processing, skip sequential loop
                        pass
                
                # After parallel processing, skip sequential loop
                
                # Finalize CBZ packaging after parallel mode finishes
                try:
                    self._finalize_cbz_jobs()
                except Exception:
                    pass
                
            else:
                # Sequential processing (or panel parallel requested but capped to 1 by global setting)
                for index, filepath in enumerate(self.selected_files):
                    if self.stop_flag.is_set():
                        self._log("\n⏹️ Translation stopped by user", "warning")
                        break
                    
                    # IMPORTANT: Reset translator state for each new image
                    if hasattr(self.translator, 'reset_for_new_image'):
                        self.translator.reset_for_new_image()
                    
                    self.current_file_index = index
                    filename = os.path.basename(filepath)
                    
                    self._update_current_file(filename)
                    self._update_progress(
                        index,
                        self.total_files,
                        f"Processing {index + 1}/{self.total_files}: {filename}"
                    )
                    
                    try:
                        # Determine output path (route CBZ images to job out_dir)
                        job_output_path = None
                        try:
                            if hasattr(self, 'cbz_image_to_job') and filepath in self.cbz_image_to_job:
                                cbz_file = self.cbz_image_to_job[filepath]
                                job = getattr(self, 'cbz_jobs', {}).get(cbz_file)
                                if job:
                                    output_dir = job.get('out_dir')
                                    os.makedirs(output_dir, exist_ok=True)
                                    job_output_path = os.path.join(output_dir, filename)
                        except Exception:
                            job_output_path = None
                        if job_output_path:
                            output_path = job_output_path
                        else:
                            if self.create_subfolder_var.get():
                                output_dir = os.path.join(os.path.dirname(filepath), 'translated')
                                os.makedirs(output_dir, exist_ok=True)
                                output_path = os.path.join(output_dir, filename)
                            else:
                                base, ext = os.path.splitext(filepath)
                                output_path = f"{base}_translated{ext}"
                        
                        # Process the image
                        result = self.translator.process_image(filepath, output_path)
                        
                        # Check if translation was interrupted
                        if result.get('interrupted', False):
                            self._log(f"⏸️ Translation of {filename} was interrupted", "warning")
                            self.failed_files += 1
                            if self.stop_flag.is_set():
                                break
                        elif result.get('success', False):
                            # Verify translation actually produced valid output
                            output_exists = result.get('output_path') and os.path.exists(result.get('output_path', ''))
                            has_translations = any(r.get('translated_text', '') for r in result.get('regions', []))
                            
                            if output_exists and has_translations:
                                self.completed_files += 1
                                self._log(f"✅ Translation completed: {filename}", "success")
                                time.sleep(0.1)  # Brief pause for stability
                                self._log("💤 Sequential completion pausing briefly for stability", "debug")
                            else:
                                self.failed_files += 1
                                if not output_exists:
                                    self._log(f"❌ Output file not created: {filename}", "error")
                                else:
                                    self._log(f"❌ No text was translated: {filename}", "error")
                        else:
                            self.failed_files += 1
                            errors = '\n'.join(result.get('errors', ['Unknown error']))
                            self._log(f"❌ Translation failed: {filename}\n{errors}", "error")
                            
                            # Check for specific error types in the error messages
                            errors_lower = errors.lower()
                            if '429' in errors or 'rate limit' in errors_lower:
                                self._log(f"⚠️ RATE LIMIT DETECTED - Please wait before continuing", "error")
                                self._log(f"   The API provider is limiting your requests", "error")
                                self._log(f"   Consider increasing delay between requests in settings", "error")
                                
                                # Optionally pause for a bit
                                self._log(f"   Pausing for 60 seconds...", "warning")
                                for sec in range(60):
                                    if self.stop_flag.is_set():
                                        break
                                    time.sleep(1)
                                    if sec % 10 == 0:
                                        self._log(f"   Waiting... {60-sec} seconds remaining", "warning")
                        
                    except Exception as e:
                        self.failed_files += 1
                        error_str = str(e)
                        error_type = type(e).__name__
                        
                        self._log(f"❌ Error processing {filename}:", "error")
                        self._log(f"   Error type: {error_type}", "error")
                        self._log(f"   Details: {error_str}", "error")
                        
                        # Check for specific API errors
                        if "429" in error_str or "rate limit" in error_str.lower():
                            self._log(f"⚠️ RATE LIMIT ERROR (429) - API is throttling requests", "error")
                            self._log(f"   Please wait before continuing or reduce request frequency", "error")
                            self._log(f"   Consider increasing the API delay in settings", "error")
                            
                            # Pause for rate limit
                            self._log(f"   Pausing for 60 seconds...", "warning")
                            for sec in range(60):
                                if self.stop_flag.is_set():
                                    break
                                time.sleep(1)
                                if sec % 10 == 0:
                                    self._log(f"   Waiting... {60-sec} seconds remaining", "warning")
                            
                        elif "401" in error_str or "unauthorized" in error_str.lower():
                            self._log(f"❌ AUTHENTICATION ERROR (401) - Check your API key", "error")
                            self._log(f"   The API key appears to be invalid or expired", "error")
                            
                        elif "403" in error_str or "forbidden" in error_str.lower():
                            self._log(f"❌ FORBIDDEN ERROR (403) - Access denied", "error")
                            self._log(f"   Check your API subscription and permissions", "error")
                            
                        elif "timeout" in error_str.lower():
                            self._log(f"⏱️ TIMEOUT ERROR - Request took too long", "error")
                            self._log(f"   Consider increasing timeout settings", "error")
                            
                        else:
                            # Generic error with full traceback
                            self._log(f"   Full traceback:", "error")
                            self._log(traceback.format_exc(), "error")
                        
            
            # Finalize CBZ packaging (both modes)
            try:
                self._finalize_cbz_jobs()
            except Exception:
                pass
            
            # Final summary - only if not stopped
            if not self.stop_flag.is_set():
                self._log(f"\n{'='*60}", "info")
                self._log(f"📊 Translation Summary:", "info")
                self._log(f"   Total files: {self.total_files}", "info")
                self._log(f"   ✅ Successful: {self.completed_files}", "success")
                self._log(f"   ❌ Failed: {self.failed_files}", "error" if self.failed_files > 0 else "info")
                self._log(f"{'='*60}\n", "info")
                
                self._update_progress(
                    self.total_files,
                    self.total_files,
                    f"Complete! {self.completed_files} successful, {self.failed_files} failed"
                )
            
        except Exception as e:
            self._log(f"\n❌ Translation error: {str(e)}", "error")
            self._log(traceback.format_exc(), "error")
        
        finally:
            # Check if auto cleanup is enabled in settings
            auto_cleanup_enabled = False  # Default disabled by default
            try:
                advanced_settings = self.main_gui.config.get('manga_settings', {}).get('advanced', {})
                auto_cleanup_enabled = advanced_settings.get('auto_cleanup_models', False)
            except Exception:
                pass
            
            if auto_cleanup_enabled:
                # Clean up all models to free RAM
                try:
                    # For parallel panel translation, cleanup happens here after ALL panels complete
                    is_parallel_panel = False
                    try:
                        advanced_settings = self.main_gui.config.get('manga_settings', {}).get('advanced', {})
                        is_parallel_panel = advanced_settings.get('parallel_panel_translation', False)
                    except Exception:
                        pass
                    
                    # Skip the "all parallel panels complete" message if stopped
                    if is_parallel_panel and not self.stop_flag.is_set():
                        self._log("\n🧹 All parallel panels complete - cleaning up models to free RAM...", "info")
                    elif not is_parallel_panel:
                        self._log("\n🧹 Cleaning up models to free RAM...", "info")
                    
                    # Clean up the shared translator if parallel processing was used
                    if 'translator' in locals():
                        translator.cleanup_all_models()
                        self._log("✅ Shared translator models cleaned up!", "info")
                    
                    # Also clean up the instance translator if it exists
                    if hasattr(self, 'translator') and self.translator:
                        self.translator.cleanup_all_models()
                        # Set to None to ensure it's released
                        self.translator = None
                        self._log("✅ Instance translator models cleaned up!", "info")
                    
                    self._log("✅ All models cleaned up - RAM freed!", "info")
                    
                except Exception as e:
                    self._log(f"⚠️ Warning: Model cleanup failed: {e}", "warning")
                
                # Force garbage collection to ensure memory is freed
                try:
                    import gc
                    gc.collect()
                except Exception:
                    pass
            else:
                self._log("🔑 Auto cleanup disabled - models will remain in RAM for faster subsequent translations", "info")
            
            # IMPORTANT: Reset the entire translator instance to free ALL memory
            # Controlled by a separate "Unload models after translation" toggle
            try:
                # Check if we should reset the translator instance
                reset_translator = False  # default disabled
                try:
                    advanced_settings = self.main_gui.config.get('manga_settings', {}).get('advanced', {})
                    reset_translator = bool(advanced_settings.get('unload_models_after_translation', False))
                except Exception:
                    reset_translator = False
                
                if reset_translator:
                    self._log("\n🗑️ Resetting translator instance to free all memory...", "info")
                    
                    # Clear the instance translator completely
                    if hasattr(self, 'translator'):
                        # First ensure models are cleaned if not already done
                        try:
                            if self.translator and hasattr(self.translator, 'cleanup_all_models'):
                                self.translator.cleanup_all_models()
                        except Exception:
                            pass
                        
                        # Clear all internal state using the dedicated method
                        try:
                            if self.translator and hasattr(self.translator, 'clear_internal_state'):
                                self.translator.clear_internal_state()
                        except Exception:
                            pass
                        
                        # Clear remaining references with proper cleanup
                        try:
                            if self.translator:
                                # Properly unload OCR manager and all its providers
                                if hasattr(self.translator, 'ocr_manager') and self.translator.ocr_manager:
                                    try:
                                        ocr_manager = self.translator.ocr_manager
                                        # Clear all loaded OCR providers
                                        if hasattr(ocr_manager, 'providers'):
                                            for provider_name, provider in ocr_manager.providers.items():
                                                # Unload each provider's models
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
                                                self._log(f"      ✓ Unloaded OCR provider: {provider_name}", "debug")
                                            ocr_manager.providers.clear()
                                        self._log("   ✓ OCR manager fully unloaded", "debug")
                                    except Exception as e:
                                        self._log(f"   Warning: OCR manager cleanup failed: {e}", "debug")
                                    finally:
                                        self.translator.ocr_manager = None
                                
                                # Properly unload local inpainter
                                if hasattr(self.translator, 'local_inpainter') and self.translator.local_inpainter:
                                    try:
                                        if hasattr(self.translator.local_inpainter, 'unload'):
                                            self.translator.local_inpainter.unload()
                                            self._log("   ✓ Local inpainter unloaded", "debug")
                                    except Exception as e:
                                        self._log(f"   Warning: Local inpainter cleanup failed: {e}", "debug")
                                    finally:
                                        self.translator.local_inpainter = None
                                
                                # Properly unload bubble detector
                                if hasattr(self.translator, 'bubble_detector') and self.translator.bubble_detector:
                                    try:
                                        if hasattr(self.translator.bubble_detector, 'unload'):
                                            self.translator.bubble_detector.unload(release_shared=True)
                                            self._log("   ✓ Bubble detector unloaded", "debug")
                                    except Exception as e:
                                        self._log(f"   Warning: Bubble detector cleanup failed: {e}", "debug")
                                    finally:
                                        self.translator.bubble_detector = None
                                
                                # Clear API clients
                                if hasattr(self.translator, 'client'):
                                    self.translator.client = None
                                if hasattr(self.translator, 'vision_client'):
                                    self.translator.vision_client = None
                        except Exception:
                            pass
                        
                        # Call translator shutdown to free all resources
                        try:
                            if translator and hasattr(translator, 'shutdown'):
                                translator.shutdown()
                        except Exception:
                            pass
                        # Finally, delete the translator instance entirely
                        self.translator = None
                        self._log("✅ Translator instance reset - all memory freed!", "info")
                    
                    # Also clear the shared translator from parallel processing if it exists
                    if 'translator' in locals():
                        try:
                            # Clear internal references
                            if hasattr(translator, 'cache'):
                                translator.cache = None
                            if hasattr(translator, 'text_regions'):
                                translator.text_regions = None
                            if hasattr(translator, 'translated_regions'):
                                translator.translated_regions = None
                            # Delete the local reference
                            del translator
                        except Exception:
                            pass
                    
                    # Clear standalone OCR manager if it exists in manga_integration
                    if hasattr(self, 'ocr_manager') and self.ocr_manager:
                        try:
                            ocr_manager = self.ocr_manager
                            # Clear all loaded OCR providers
                            if hasattr(ocr_manager, 'providers'):
                                for provider_name, provider in ocr_manager.providers.items():
                                    # Unload each provider's models
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
                                ocr_manager.providers.clear()
                            self.ocr_manager = None
                            self._log("   ✓ Standalone OCR manager cleared", "debug")
                        except Exception as e:
                            self._log(f"   Warning: Standalone OCR manager cleanup failed: {e}", "debug")
                    
                    # Force multiple garbage collection passes to ensure everything is freed
                    try:
                        import gc
                        gc.collect()
                        gc.collect()  # Multiple passes for stubborn references
                        gc.collect()
                        self._log("✅ Memory fully reclaimed", "debug")
                    except Exception:
                        pass
                else:
                    self._log("🔑 Translator instance preserved for faster subsequent translations", "debug")
                    
            except Exception as e:
                self._log(f"⚠️ Warning: Failed to reset translator instance: {e}", "warning")
            
            # Check if parent frame still exists before scheduling callback
            if hasattr(self, 'parent_frame') and self.parent_frame.winfo_exists():
                self.parent_frame.after(0, self._reset_ui_state)
    
    def _stop_translation(self):
        """Stop the translation process"""
        if self.is_running:
            # Set local stop flag
            self.stop_flag.set()
            
            # Set global cancellation flags for coordinated stopping
            self.set_global_cancellation(True)
            
            # Also propagate to MangaTranslator class
            try:
                from manga_translator import MangaTranslator
                MangaTranslator.set_global_cancellation(True)
            except ImportError:
                pass
            
            # Also propagate to UnifiedClient if available
            try:
                from unified_api_client import UnifiedClient
                UnifiedClient.set_global_cancellation(True)
            except ImportError:
                pass
            
            # Update progress to show stopped status
            self._update_progress(
                self.completed_files,
                self.total_files, 
                f"Stopped - {self.completed_files}/{self.total_files} completed"
            )
            
            # Try to style the progress bar to indicate stopped status
            try:
                # Set progress bar to a distinctive value and try to change appearance
                if hasattr(self, 'progress_bar'):
                    # You could also set a custom style here if needed
                    # For now, we'll rely on the text indicators
                    pass
            except Exception:
                pass
            
            # Update current file display to show stopped
            self._update_current_file("Translation stopped")
            
            # Kick off immediate resource shutdown to free RAM
            try:
                tr = getattr(self, 'translator', None)
                if tr and hasattr(tr, 'shutdown'):
                    import threading
                    threading.Thread(target=tr.shutdown, name="MangaTranslatorShutdown", daemon=True).start()
                    self._log("🧹 Initiated translator resource shutdown", "info")
                    # Important: clear the stale translator reference so the next Start creates a fresh instance
                    self.translator = None
            except Exception as e:
                self._log(f"⚠️ Failed to start shutdown: {e}", "warning")
            
            # Immediately reset UI state to re-enable start button
            self._reset_ui_state()
            self._log("\n⏹️ Translation stopped by user", "warning")
    
    def _reset_ui_state(self):
        """Reset UI to ready state - with widget existence checks"""
        # Restore stdio redirection if active
        self._redirect_stderr(False)
        self._redirect_stdout(False)
        # Stop any startup heartbeat if still running
        try:
            self._stop_startup_heartbeat()
        except Exception:
            pass
        try:
            # Check if the dialog still exists
            if not hasattr(self, 'dialog') or not self.dialog or not self.dialog.winfo_exists():
                return
                
            # Reset running flag
            self.is_running = False
            
            # Check and update start_button if it exists - only if not already enabled
            if hasattr(self, 'start_button') and self.start_button and self.start_button.winfo_exists():
                if str(self.start_button.cget('state')) == 'disabled':
                    self.start_button.config(state=tk.NORMAL)
            
            # Check and update stop_button if it exists - only if not already disabled
            if hasattr(self, 'stop_button') and self.stop_button and self.stop_button.winfo_exists():
                if str(self.stop_button.cget('state')) == 'normal':
                    self.stop_button.config(state=tk.DISABLED)
            
            # Re-enable file modification - check if listbox exists
            if hasattr(self, 'file_listbox') and self.file_listbox and self.file_listbox.winfo_exists():
                if str(self.file_listbox.cget('state')) == 'disabled':
                    self.file_listbox.config(state=tk.NORMAL)
                
        except tk.TclError:
            # Widget has been destroyed, nothing to do
            pass
        except Exception as e:
            # Log the error but don't crash
            if hasattr(self, '_log'):
                self._log(f"Error resetting UI state: {str(e)}", "warning")
