"""
bubble_detector.py - Modified version that works in frozen PyInstaller executables
Replace your bubble_detector.py with this version
"""
import os
import sys
import json
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
import logging
import traceback
import hashlib
import gc
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if we're running in a frozen environment
IS_FROZEN = getattr(sys, 'frozen', False)
if IS_FROZEN:
    # In frozen environment, set proper paths for ML libraries
    MEIPASS = sys._MEIPASS
    os.environ['TORCH_HOME'] = MEIPASS
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(MEIPASS, 'transformers')
    os.environ['HF_HOME'] = os.path.join(MEIPASS, 'huggingface')
    logger.info(f"Running in frozen environment: {MEIPASS}")

# Modified import checks for frozen environment
YOLO_AVAILABLE = False
YOLO = None
torch = None
TORCH_AVAILABLE = False
ONNX_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
RTDetrForObjectDetection = None
RTDetrImageProcessor = None
PIL_AVAILABLE = False

# Try to import YOLO dependencies with better error handling
if IS_FROZEN:
    # In frozen environment, try harder to import
    try:
        # First try to import torch components individually
        import torch
        import torch.nn
        import torch.cuda
        TORCH_AVAILABLE = True
        logger.info("✓ PyTorch loaded in frozen environment")
    except Exception as e:
        logger.warning(f"PyTorch not available in frozen environment: {e}")
        TORCH_AVAILABLE = False
        torch = None
    
    # Try ultralytics after torch
    if TORCH_AVAILABLE:
        try:
            from ultralytics import YOLO
            YOLO_AVAILABLE = True
            logger.info("✓ Ultralytics YOLO loaded in frozen environment")
        except Exception as e:
            logger.warning(f"Ultralytics not available in frozen environment: {e}")
            YOLO_AVAILABLE = False
    
    # Try transformers
    try:
        import transformers
        # Try specific imports
        try:
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
            TRANSFORMERS_AVAILABLE = True
            logger.info("✓ Transformers RT-DETR loaded in frozen environment")
        except ImportError:
            # Try alternative import
            try:
                from transformers import AutoModel, AutoImageProcessor
                RTDetrForObjectDetection = AutoModel
                RTDetrImageProcessor = AutoImageProcessor
                TRANSFORMERS_AVAILABLE = True
                logger.info("✓ Transformers loaded with AutoModel fallback")
            except:
                TRANSFORMERS_AVAILABLE = False
                logger.warning("Transformers RT-DETR not available in frozen environment")
    except Exception as e:
        logger.warning(f"Transformers not available in frozen environment: {e}")
        TRANSFORMERS_AVAILABLE = False
else:
    # Normal environment - original import logic
    try:
        from ultralytics import YOLO
        YOLO_AVAILABLE = True
    except:
        YOLO_AVAILABLE = False
        logger.warning("Ultralytics YOLO not available")

    try:
        import torch
        # Test if cuda attribute exists
        _ = torch.cuda
        TORCH_AVAILABLE = True
    except (ImportError, AttributeError):
        TORCH_AVAILABLE = False
        torch = None
        logger.warning("PyTorch not available or incomplete")

    try:
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        try:
            from transformers import RTDetrV2ForObjectDetection
            RTDetrForObjectDetection = RTDetrV2ForObjectDetection
        except ImportError:
            pass
        TRANSFORMERS_AVAILABLE = True
    except:
        TRANSFORMERS_AVAILABLE = False
        logger.info("Transformers not available for RT-DETR")

# ONNX Runtime - works well in frozen environments
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    logger.info("✓ ONNX Runtime available")
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available")

# PIL
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.info("PIL not available")


class BubbleDetector:
    """
    Combined YOLOv8 and RT-DETR speech bubble detector for comics and manga.
    Supports multiple model formats and provides configurable detection.
    Backward compatible with existing code while adding RT-DETR support.
    """
    
    def __init__(self, config_path: str = "bubble_detector_config.json"):
        """
        Initialize the bubble detector.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # YOLOv8 components (original)
        self.model = None
        self.model_loaded = False
        self.model_type = None  # 'yolo', 'onnx', or 'torch'
        self.onnx_session = None
        
        # RT-DETR components (new)
        self.rtdetr_model = None
        self.rtdetr_processor = None
        self.rtdetr_loaded = False
        self.rtdetr_repo = 'ogkalu/comic-text-and-bubble-detector'
        
        # RT-DETR class definitions
        self.CLASS_BUBBLE = 0      # Empty speech bubble
        self.CLASS_TEXT_BUBBLE = 1 # Bubble with text
        self.CLASS_TEXT_FREE = 2   # Text without bubble
        
        # Detection settings
        self.default_confidence = 0.5
        self.default_iou_threshold = 0.45
        self.default_max_detections = 100
        
        # Cache directory for ONNX conversions
        self.cache_dir = os.environ.get('BUBBLE_CACHE_DIR', 'bubble_model_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # GPU availability
        self.use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'
        
        logger.info(f"🗨️ BubbleDetector initialized")
        logger.info(f"   GPU: {'Available' if self.use_gpu else 'Not available'}")
        logger.info(f"   YOLO: {'Available' if YOLO_AVAILABLE else 'Not installed'}")
        logger.info(f"   ONNX: {'Available' if ONNX_AVAILABLE else 'Not installed'}")
        logger.info(f"   RT-DETR: {'Available' if TRANSFORMERS_AVAILABLE else 'Not installed'}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        return {}
    
    def _save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
        
    def load_model(self, model_path: str, force_reload: bool = False) -> bool:
        """
        Load a YOLOv8 model for bubble detection.
        
        Args:
            model_path: Path to model file (.pt, .onnx, or .torchscript)
            force_reload: Force reload even if model is already loaded
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Check if it's the same model already loaded
            if self.model_loaded and not force_reload:
                last_path = self.config.get('last_model_path', '')
                if last_path == model_path:
                    logger.info("Model already loaded (same path)")
                    return True
                else:
                    logger.info(f"Model path changed from {last_path} to {model_path}, reloading...")
                    force_reload = True
            
            # Clear previous model if force reload
            if force_reload:
                logger.info(f"🔄 Force reloading {method} model...")
                
                # Properly clear existing model
                if self.model is not None:
                    if TORCH_AVAILABLE and torch is not None and hasattr(self.model, 'cpu'):
                        try:
                            self.model.cpu()
                        except:
                            pass
                    del self.model
                    self.model = None
                
                # Clear ONNX session
                if self.onnx_session is not None:
                    del self.onnx_session
                    self.onnx_session = None
                
                # Clear GPU cache if using PyTorch
                if TORCH_AVAILABLE and torch is not None:
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            logger.info("   Cleared GPU cache")
                    except:
                        pass
                
                # Force garbage collection for CPU memory
                import gc
                gc.collect()
                
                # Reset flags
                self.model_loaded = False
                self.is_jit_model = False
                self.use_onnx = False
                self.current_method = None
                
                # Small delay to ensure cleanup
                import time
                time.sleep(0.1)
            
            logger.info(f"📥 Loading bubble detection model: {model_path}")
            
            # Determine model type by extension
            ext = Path(model_path).suffix.lower()
            
            if ext in ['.pt', '.pth']:
                if not YOLO_AVAILABLE:
                    logger.warning("Ultralytics package not available in this build")
                    logger.info("Bubble detection will be disabled - this is normal for lightweight builds")
                    # Don't return False immediately, try other fallbacks
                    self.model_loaded = False
                    return False
                
                # Load YOLOv8 model
                try:
                    self.model = YOLO(model_path)
                    self.model_type = 'yolo'
                    
                    # Set to eval mode
                    if hasattr(self.model, 'model'):
                        self.model.model.eval()
                    
                    # Move to GPU if available
                    if self.use_gpu and TORCH_AVAILABLE:
                        try:
                            self.model.to('cuda')
                        except Exception as gpu_error:
                            logger.warning(f"Could not move model to GPU: {gpu_error}")
                            
                    logger.info("✅ YOLOv8 model loaded successfully")
                    
                except Exception as yolo_error:
                    logger.error(f"Failed to load YOLO model: {yolo_error}")
                    return False
                    
            elif ext == '.onnx':
                if not ONNX_AVAILABLE:
                    logger.warning("ONNX Runtime not available in this build")
                    logger.info("ONNX model support disabled - this is normal for lightweight builds")
                    return False
                
                try:
                    # Load ONNX model
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
                    self.onnx_session = ort.InferenceSession(model_path, providers=providers)
                    self.model_type = 'onnx'
                    
                    logger.info("✅ ONNX model loaded successfully")
                    
                except Exception as onnx_error:
                    logger.error(f"Failed to load ONNX model: {onnx_error}")
                    return False
                    
            elif ext == '.torchscript':
                if not TORCH_AVAILABLE:
                    logger.warning("PyTorch not available in this build")
                    logger.info("TorchScript model support disabled - this is normal for lightweight builds")
                    return False
                
                try:
                    # Add safety check for torch being None
                    if torch is None:
                        logger.error("PyTorch module is None - cannot load TorchScript model")
                        return False
                        
                    # Load TorchScript model
                    self.model = torch.jit.load(model_path, map_location='cpu')
                    self.model.eval()
                    self.model_type = 'torch'
                    
                    if self.use_gpu:
                        try:
                            self.model = self.model.cuda()
                        except Exception as gpu_error:
                            logger.warning(f"Could not move TorchScript model to GPU: {gpu_error}")
                    
                    logger.info("✅ TorchScript model loaded successfully")
                    
                except Exception as torch_error:
                    logger.error(f"Failed to load TorchScript model: {torch_error}")
                    return False
                    
            else:
                logger.error(f"Unsupported model format: {ext}")
                logger.info("Supported formats: .pt/.pth (YOLOv8), .onnx (ONNX), .torchscript (TorchScript)")
                return False
            
            # Only set loaded if we actually succeeded
            self.model_loaded = True
            self.config['last_model_path'] = model_path
            self.config['model_type'] = self.model_type
            self._save_config()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(traceback.format_exc())
            self.model_loaded = False
            
            # Provide helpful context for .exe users
            logger.info("Note: If running from .exe, some ML libraries may not be included")
            logger.info("This is normal for lightweight builds - bubble detection will be disabled")
            
            return False

    def cleanup(self):
        """Cleanup all loaded models and free memory"""
        logger.info("🧹 Cleaning up BubbleDetector...")
        
        # Clear YOLO model
        if self.model is not None:
            if TORCH_AVAILABLE and hasattr(self.model, 'to'):
                try:
                    self.model.to('cpu')
                except:
                    pass
            del self.model
            self.model = None
        
        # Clear ONNX
        if self.onnx_session is not None:
            del self.onnx_session
            self.onnx_session = None
        
        # Clear RT-DETR
        if self.rtdetr_model is not None:
            if TORCH_AVAILABLE and hasattr(self.rtdetr_model, 'cpu'):
                try:
                    self.rtdetr_model.cpu()
                except:
                    pass
            del self.rtdetr_model
            self.rtdetr_model = None
        
        if self.rtdetr_processor is not None:
            del self.rtdetr_processor
            self.rtdetr_processor = None
        
        # Clear GPU cache
        if TORCH_AVAILABLE and torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
        
        self.model_loaded = False
        self.rtdetr_loaded = False
        logger.info("✅ BubbleDetector cleanup complete")
    
    def load_rtdetr_model(self, model_path: str = None, model_id: str = None, force_reload: bool = False) -> bool:
        """
        Load RT-DETR model for advanced bubble and text detection.
        
        Args:
            model_path: Optional path to local model
            model_id: Optional HuggingFace model ID (default: 'ogkalu/comic-text-and-bubble-detector')
            force_reload: Force reload even if already loaded
            
        Returns:
            True if successful, False otherwise
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library required for RT-DETR. Install with: pip install transformers")
            return False
        
        if not PIL_AVAILABLE:
            logger.error("PIL required for RT-DETR. Install with: pip install pillow")
            return False
        
        if self.rtdetr_loaded and not force_reload:
            logger.info("RT-DETR model already loaded")
            return True
        
        try:
            # Use custom model_id if provided, otherwise use default
            repo_id = model_id if model_id else self.rtdetr_repo
            
            logger.info(f"📥 Loading RT-DETR model from {repo_id}...")
            
            # Load processor
            self.rtdetr_processor = RTDetrImageProcessor.from_pretrained(
                repo_id,
                size={"width": 640, "height": 640},
                cache_dir=self.cache_dir if not model_path else None
            )
            
            # Load model
            self.rtdetr_model = RTDetrForObjectDetection.from_pretrained(
                model_path if model_path else repo_id,
                cache_dir=self.cache_dir if not model_path else None
            )
            
            # Move to device
            if self.device == 'cuda':
                self.rtdetr_model = self.rtdetr_model.to('cuda')
                logger.info("   RT-DETR model moved to GPU")
            
            self.rtdetr_model.eval()
            self.rtdetr_loaded = True
            
            # Save the model ID that was used
            self.config['rtdetr_loaded'] = True
            self.config['rtdetr_model_id'] = repo_id
            self._save_config()
            
            logger.info("✅ RT-DETR model loaded successfully")
            logger.info("   Classes: Empty bubbles, Text bubbles, Free text")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load RT-DETR: {e}")
            self.rtdetr_loaded = False
            return False
 
    def check_rtdetr_available(self, model_id: str = None) -> bool:
        """
        Check if RT-DETR model is available (cached).
        
        Args:
            model_id: Optional HuggingFace model ID
            
        Returns:
            True if model is cached and available
        """
        try:
            from pathlib import Path
            
            # Use provided model_id or default
            repo_id = model_id if model_id else self.rtdetr_repo
            
            # Check HuggingFace cache
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_id_formatted = repo_id.replace("/", "--")
            
            # Look for model folder
            model_folders = list(cache_dir.glob(f"models--{model_id_formatted}*"))
            
            if model_folders:
                for folder in model_folders:
                    if (folder / "snapshots").exists():
                        snapshots = list((folder / "snapshots").iterdir())
                        if snapshots:
                            return True
            
            return False
            
        except Exception:
            return False
        
    def detect_bubbles(self, 
                      image_path: str, 
                      confidence: float = None,
                      iou_threshold: float = None,
                      max_detections: int = None,
                      use_rtdetr: bool = None) -> List[Tuple[int, int, int, int]]:
        """
        Detect speech bubbles in an image (backward compatible method).
        
        Args:
            image_path: Path to image file
            confidence: Minimum confidence threshold (0-1)
            iou_threshold: IOU threshold for NMS (0-1)
            max_detections: Maximum number of detections to return
            use_rtdetr: If True, use RT-DETR instead of YOLOv8 (if available)
            
        Returns:
            List of bubble bounding boxes as (x, y, width, height) tuples
        """
        # Decide which model to use
        if use_rtdetr is None:
            # Auto-select: prefer RT-DETR if available
            use_rtdetr = self.rtdetr_loaded
        
        if use_rtdetr and self.rtdetr_loaded:
            # Use RT-DETR
            results = self.detect_with_rtdetr(
                image_path=image_path,
                confidence=confidence,
                return_all_bubbles=True
            )
            return results
        
        # Original YOLOv8 detection
        if not self.model_loaded:
            logger.error("No model loaded. Call load_model() first.")
            return []
        
        # Use defaults if not specified
        confidence = confidence or self.default_confidence
        iou_threshold = iou_threshold or self.default_iou_threshold
        max_detections = max_detections or self.default_max_detections
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return []
            
            h, w = image.shape[:2]
            logger.info(f"🔍 Detecting bubbles in {w}x{h} image")
            
            if self.model_type == 'yolo':
                # YOLOv8 inference
                results = self.model(
                    image_path,
                    conf=confidence,
                    iou=iou_threshold,
                    max_det=max_detections,
                    verbose=False
                )
                
                bubbles = []
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x, y = int(x1), int(y1)
                            width = int(x2 - x1)
                            height = int(y2 - y1)
                            
                            # Get confidence
                            conf = float(box.conf[0])
                            
                            # Add to list
                            bubbles.append((x, y, width, height))
                            
                            logger.debug(f"   Bubble: ({x},{y}) {width}x{height} conf={conf:.2f}")
                
            elif self.model_type == 'onnx':
                # ONNX inference
                bubbles = self._detect_with_onnx(image, confidence, iou_threshold, max_detections)
                
            elif self.model_type == 'torch':
                # TorchScript inference
                bubbles = self._detect_with_torchscript(image, confidence, iou_threshold, max_detections)
            
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                return []
            
            logger.info(f"✅ Detected {len(bubbles)} speech bubbles")
            return bubbles
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def detect_with_rtdetr(self,
                          image_path: str = None,
                          image: np.ndarray = None,
                          confidence: float = None,
                          return_all_bubbles: bool = False) -> Any:
        """
        Detect using RT-DETR model with 3-class detection.
        
        Args:
            image_path: Path to image file
            image: Image array (BGR format)
            confidence: Confidence threshold
            return_all_bubbles: If True, return list of bubble boxes (for compatibility)
                               If False, return dict with all classes
        
        Returns:
            List of bubbles if return_all_bubbles=True, else dict with classes
        """
        if not self.rtdetr_loaded:
            logger.warning("RT-DETR not loaded. Call load_rtdetr_model() first.")
            if return_all_bubbles:
                return []
            return {'bubbles': [], 'text_bubbles': [], 'text_free': []}
        
        confidence = confidence or self.default_confidence
        
        try:
            # Load image
            if image_path:
                image = cv2.imread(image_path)
            elif image is None:
                logger.error("No image provided")
                if return_all_bubbles:
                    return []
                return {'bubbles': [], 'text_bubbles': [], 'text_free': []}
            
            # Convert BGR to RGB for PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Prepare image for model
            inputs = self.rtdetr_processor(images=pil_image, return_tensors="pt")
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.rtdetr_model(**inputs)
            
            # Post-process results
            target_sizes = torch.tensor([pil_image.size[::-1]])
            if self.device == "cuda":
                target_sizes = target_sizes.to("cuda")
            
            results = self.rtdetr_processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=confidence
            )[0]
            
            logger.info(f"📊 RT-DETR found {len(results['boxes'])} detections above {confidence:.2f} confidence")

            # Organize detections by class
            detections = {
                'bubbles': [],       # Empty speech bubbles
                'text_bubbles': [],  # Bubbles with text
                'text_free': []      # Text without bubbles
            }
            
            for box, score, label in zip(results['boxes'], results['scores'], results['labels']):
                x1, y1, x2, y2 = map(int, box.tolist())
                width = x2 - x1
                height = y2 - y1
                
                # Store as (x, y, width, height) to match YOLOv8 format
                bbox = (x1, y1, width, height)
                
                label_id = label.item()
                if label_id == self.CLASS_BUBBLE:
                    detections['bubbles'].append(bbox)
                elif label_id == self.CLASS_TEXT_BUBBLE:
                    detections['text_bubbles'].append(bbox)
                elif label_id == self.CLASS_TEXT_FREE:
                    detections['text_free'].append(bbox)
            
            # Log results
            total = len(detections['bubbles']) + len(detections['text_bubbles']) + len(detections['text_free'])
            logger.info(f"✅ RT-DETR detected {total} objects:")
            logger.info(f"   - Empty bubbles: {len(detections['bubbles'])}")
            logger.info(f"   - Text bubbles: {len(detections['text_bubbles'])}")
            logger.info(f"   - Free text: {len(detections['text_free'])}")
            
            # Return format based on compatibility mode
            if return_all_bubbles:
                # Return all bubbles (empty + with text) for backward compatibility
                all_bubbles = detections['bubbles'] + detections['text_bubbles']
                return all_bubbles
            else:
                return detections
            
        except Exception as e:
            logger.error(f"RT-DETR detection failed: {e}")
            logger.error(traceback.format_exc())
            if return_all_bubbles:
                return []
            return {'bubbles': [], 'text_bubbles': [], 'text_free': []}
    
    def detect_all_text_regions(self, image_path: str = None, image: np.ndarray = None) -> List[Tuple[int, int, int, int]]:
        """
        Detect all text regions using RT-DETR (both in bubbles and free text).
        
        Returns:
            List of bounding boxes for all text regions
        """
        if not self.rtdetr_loaded:
            logger.warning("RT-DETR required for text detection")
            return []
        
        detections = self.detect_with_rtdetr(image_path=image_path, image=image, return_all_bubbles=False)
        
        # Combine text bubbles and free text
        all_text = detections['text_bubbles'] + detections['text_free']
        
        logger.info(f"📝 Found {len(all_text)} text regions total")
        return all_text
    
    def _detect_with_onnx(self, image: np.ndarray, confidence: float, 
                         iou_threshold: float, max_detections: int) -> List[Tuple[int, int, int, int]]:
        """Run detection using ONNX model."""
        # Preprocess image
        img_size = 640  # Standard YOLOv8 input size
        img_resized = cv2.resize(image, (img_size, img_size))
        img_norm = img_resized.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_norm, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        # Run inference
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: img_batch})
        
        # Process outputs (YOLOv8 format)
        predictions = outputs[0][0]  # Remove batch dimension
        
        # Filter by confidence and apply NMS
        bubbles = []
        boxes = []
        scores = []
        
        for pred in predictions.T:  # Transpose to get predictions per detection
            if len(pred) >= 5:
                x_center, y_center, width, height, obj_conf = pred[:5]
                
                if obj_conf >= confidence:
                    # Convert to corner coordinates
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    
                    # Scale to original image size
                    h, w = image.shape[:2]
                    x1 = int(x1 * w / img_size)
                    y1 = int(y1 * h / img_size)
                    width = int(width * w / img_size)
                    height = int(height * h / img_size)
                    
                    boxes.append([x1, y1, x1 + width, y1 + height])
                    scores.append(float(obj_conf))
        
        # Apply NMS
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, confidence, iou_threshold)
            if len(indices) > 0:
                indices = indices.flatten()[:max_detections]
                for i in indices:
                    x1, y1, x2, y2 = boxes[i]
                    bubbles.append((x1, y1, x2 - x1, y2 - y1))
        
        return bubbles
    
    def _detect_with_torchscript(self, image: np.ndarray, confidence: float,
                                 iou_threshold: float, max_detections: int) -> List[Tuple[int, int, int, int]]:
        """Run detection using TorchScript model."""
        # Similar to ONNX but using PyTorch tensors
        img_size = 640
        img_resized = cv2.resize(image, (img_size, img_size))
        img_norm = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)
        
        if self.use_gpu:
            img_tensor = img_tensor.cuda()
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
        
        # Process outputs similar to ONNX
        # Implementation depends on exact model output format
        # This is a placeholder - adjust based on your model
        return []
    
    def visualize_detections(self, image_path: str, bubbles: List[Tuple[int, int, int, int]] = None, 
                            output_path: str = None, use_rtdetr: bool = False) -> np.ndarray:
        """
        Visualize detected bubbles on the image.
        
        Args:
            image_path: Path to original image
            bubbles: List of bubble bounding boxes (if None, will detect)
            output_path: Optional path to save visualization
            use_rtdetr: Use RT-DETR for visualization with class colors
            
        Returns:
            Image with drawn bounding boxes
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        vis_image = image.copy()
        
        if use_rtdetr and self.rtdetr_loaded:
            # RT-DETR visualization with different colors per class
            detections = self.detect_with_rtdetr(image_path=image_path, return_all_bubbles=False)
            
            # Colors for each class
            colors = {
                'bubbles': (0, 255, 0),       # Green for empty bubbles
                'text_bubbles': (255, 0, 0),  # Blue for text bubbles
                'text_free': (0, 0, 255)      # Red for free text
            }
            
            # Draw detections
            for class_name, bboxes in detections.items():
                color = colors[class_name]
                
                for i, (x, y, w, h) in enumerate(bboxes):
                    # Draw rectangle
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                    
                    # Add label
                    label = f"{class_name.replace('_', ' ').title()} {i+1}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(vis_image, (x, y - label_size[1] - 4), 
                                (x + label_size[0], y), color, -1)
                    cv2.putText(vis_image, label, (x, y - 2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # Original YOLOv8 visualization
            if bubbles is None:
                bubbles = self.detect_bubbles(image_path)
            
            # Draw bounding boxes
            for i, (x, y, w, h) in enumerate(bubbles):
                # Draw rectangle
                color = (0, 255, 0)  # Green
                thickness = 2
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, thickness)
                
                # Add label
                label = f"Bubble {i+1}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis_image, (x, y - label_size[1] - 4), (x + label_size[0], y), color, -1)
                cv2.putText(vis_image, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, vis_image)
            logger.info(f"💾 Visualization saved to: {output_path}")
        
        return vis_image
    
    def convert_to_onnx(self, model_path: str, output_path: str = None) -> bool:
        """
        Convert a YOLOv8 model to ONNX format.
        
        Args:
            model_path: Path to YOLOv8 .pt model
            output_path: Path for ONNX output (auto-generated if None)
            
        Returns:
            True if conversion successful, False otherwise
        """
        if not YOLO_AVAILABLE:
            logger.error("Ultralytics package required for conversion")
            return False
        
        try:
            logger.info(f"🔄 Converting {model_path} to ONNX...")
            
            # Load model
            model = YOLO(model_path)
            
            # Generate output path if not provided
            if output_path is None:
                base_name = Path(model_path).stem
                output_path = os.path.join(self.cache_dir, f"{base_name}.onnx")
            
            # Export to ONNX
            model.export(format='onnx', imgsz=640, simplify=True)
            
            logger.info(f"✅ ONNX model saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return False
    
    def batch_detect(self, image_paths: List[str], **kwargs) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """
        Detect bubbles in multiple images.
        
        Args:
            image_paths: List of image paths
            **kwargs: Detection parameters (confidence, iou_threshold, max_detections, use_rtdetr)
            
        Returns:
            Dictionary mapping image paths to bubble lists
        """
        results = {}
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            bubbles = self.detect_bubbles(image_path, **kwargs)
            results[image_path] = bubbles
        
        return results
    
    def get_bubble_masks(self, image_path: str, bubbles: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Create a mask image with bubble regions.
        
        Args:
            image_path: Path to original image
            bubbles: List of bubble bounding boxes
            
        Returns:
            Binary mask with bubble regions as white (255)
        """
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Fill bubble regions
        for x, y, bw, bh in bubbles:
            cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
        
        return mask
    
    def filter_bubbles_by_size(self, bubbles: List[Tuple[int, int, int, int]], 
                              min_area: int = 100, 
                              max_area: int = None) -> List[Tuple[int, int, int, int]]:
        """
        Filter bubbles by area.
        
        Args:
            bubbles: List of bubble bounding boxes
            min_area: Minimum area in pixels
            max_area: Maximum area in pixels (None for no limit)
            
        Returns:
            Filtered list of bubbles
        """
        filtered = []
        
        for x, y, w, h in bubbles:
            area = w * h
            if area >= min_area and (max_area is None or area <= max_area):
                filtered.append((x, y, w, h))
        
        return filtered
    
    def merge_overlapping_bubbles(self, bubbles: List[Tuple[int, int, int, int]], 
                                 overlap_threshold: float = 0.1) -> List[Tuple[int, int, int, int]]:
        """
        Merge overlapping bubble detections.
        
        Args:
            bubbles: List of bubble bounding boxes
            overlap_threshold: Minimum overlap ratio to merge
            
        Returns:
            Merged list of bubbles
        """
        if not bubbles:
            return []
        
        # Convert to numpy array for easier manipulation
        boxes = np.array([(x, y, x+w, y+h) for x, y, w, h in bubbles])
        
        merged = []
        used = set()
        
        for i, box1 in enumerate(boxes):
            if i in used:
                continue
            
            # Start with current box
            x1, y1, x2, y2 = box1
            
            # Check for overlaps with remaining boxes
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                
                box2 = boxes[j]
                
                # Calculate intersection
                ix1 = max(x1, box2[0])
                iy1 = max(y1, box2[1])
                ix2 = min(x2, box2[2])
                iy2 = min(y2, box2[3])
                
                if ix1 < ix2 and iy1 < iy2:
                    # Calculate overlap ratio
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    overlap = intersection / min(area1, area2)
                    
                    if overlap >= overlap_threshold:
                        # Merge boxes
                        x1 = min(x1, box2[0])
                        y1 = min(y1, box2[1])
                        x2 = max(x2, box2[2])
                        y2 = max(y2, box2[3])
                        used.add(j)
            
            merged.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        
        return merged


# Standalone utility functions
def download_model_from_huggingface(repo_id: str = "ogkalu/comic-speech-bubble-detector-yolov8m",
                                   filename: str = "comic-speech-bubble-detector-yolov8m.pt",
                                   cache_dir: str = "models") -> str:
    """
    Download model from Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repository ID
        filename: Model filename in the repository
        cache_dir: Local directory to cache the model
        
    Returns:
        Path to downloaded model file
    """
    try:
        from huggingface_hub import hf_hub_download
        
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"📥 Downloading {filename} from {repo_id}...")
        
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            local_dir=cache_dir
        )
        
        logger.info(f"✅ Model downloaded to: {model_path}")
        return model_path
        
    except ImportError:
        logger.error("huggingface-hub package required. Install with: pip install huggingface-hub")
        return None
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


def download_rtdetr_model(cache_dir: str = "models") -> bool:
    """
    Download RT-DETR model for advanced detection.
    
    Args:
        cache_dir: Directory to cache the model
        
    Returns:
        True if successful
    """
    if not TRANSFORMERS_AVAILABLE:
        logger.error("Transformers required. Install with: pip install transformers")
        return False
    
    try:
        logger.info("📥 Downloading RT-DETR model...")
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        
        # This will download and cache the model
        processor = RTDetrImageProcessor.from_pretrained(
            "ogkalu/comic-text-and-bubble-detector",
            cache_dir=cache_dir
        )
        model = RTDetrForObjectDetection.from_pretrained(
            "ogkalu/comic-text-and-bubble-detector",
            cache_dir=cache_dir
        )
        
        logger.info("✅ RT-DETR model downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Create detector
    detector = BubbleDetector()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "download":
            # Download model from Hugging Face
            model_path = download_model_from_huggingface()
            if model_path:
                print(f"YOLOv8 model downloaded to: {model_path}")
            
            # Also download RT-DETR
            if download_rtdetr_model():
                print("RT-DETR model downloaded")
        
        elif sys.argv[1] == "detect" and len(sys.argv) > 3:
            # Detect bubbles in an image
            model_path = sys.argv[2]
            image_path = sys.argv[3]
            
            # Load appropriate model
            if 'rtdetr' in model_path.lower():
                if detector.load_rtdetr_model():
                    # Use RT-DETR
                    results = detector.detect_with_rtdetr(image_path)
                    print(f"RT-DETR Detection:")
                    print(f"  Empty bubbles: {len(results['bubbles'])}")
                    print(f"  Text bubbles: {len(results['text_bubbles'])}")
                    print(f"  Free text: {len(results['text_free'])}")
            else:
                if detector.load_model(model_path):
                    bubbles = detector.detect_bubbles(image_path, confidence=0.5)
                    print(f"YOLOv8 detected {len(bubbles)} bubbles:")
                    for i, (x, y, w, h) in enumerate(bubbles):
                        print(f"  Bubble {i+1}: position=({x},{y}) size=({w}x{h})")
            
            # Optionally visualize
            if len(sys.argv) > 4:
                output_path = sys.argv[4]
                detector.visualize_detections(image_path, output_path=output_path, 
                                             use_rtdetr='rtdetr' in model_path.lower())
        
        elif sys.argv[1] == "test-both" and len(sys.argv) > 2:
            # Test both models
            image_path = sys.argv[2]
            
            # Load YOLOv8
            yolo_path = "models/comic-speech-bubble-detector-yolov8m.pt"
            if os.path.exists(yolo_path):
                detector.load_model(yolo_path)
                yolo_bubbles = detector.detect_bubbles(image_path, use_rtdetr=False)
                print(f"YOLOv8: {len(yolo_bubbles)} bubbles")
            
            # Load RT-DETR
            if detector.load_rtdetr_model():
                rtdetr_bubbles = detector.detect_bubbles(image_path, use_rtdetr=True)
                print(f"RT-DETR: {len(rtdetr_bubbles)} bubbles")
        
        else:
            print("Usage:")
            print("  python bubble_detector.py download")
            print("  python bubble_detector.py detect <model_path> <image_path> [output_path]")
            print("  python bubble_detector.py test-both <image_path>")
    
    else:
        print("Bubble Detector Module (YOLOv8 + RT-DETR)")
        print("Usage:")
        print("  python bubble_detector.py download")
        print("  python bubble_detector.py detect <model_path> <image_path> [output_path]")
        print("  python bubble_detector.py test-both <image_path>")

