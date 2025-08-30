# bubble_detector.py
"""
Local LLM bubble detector with automatic PT to ONNX conversion
No model bundling required - users provide their own
"""

import os
import json
import hashlib
from typing import Optional, List, Tuple
import numpy as np
import cv2

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

class BubbleDetector:
    """YOLOv8 bubble detector with automatic model conversion"""
    
    def __init__(self, config_path: str = "bubble_detector_config.json"):
        """Initialize detector
        
        Args:
            config_path: Path to configuration file storing model paths
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.session = None
        self.model_loaded = False
        
    def _load_config(self) -> dict:
        """Load configuration"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_config(self):
        """Save configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_model(self, pt_path: str, force_reconvert: bool = False) -> bool:
        """Load YOLO model, converting from PT to ONNX if needed
        
        Args:
            pt_path: Path to .pt file
            force_reconvert: Force reconversion even if ONNX exists
            
        Returns:
            True if successful
        """
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Model file not found: {pt_path}")
        
        if not pt_path.endswith('.pt'):
            raise ValueError("Model file must be a .pt file")
        
        # Generate ONNX path based on PT file
        onnx_dir = os.path.join(os.path.dirname(pt_path), 'onnx_cache')
        os.makedirs(onnx_dir, exist_ok=True)
        
        # Use hash to handle model updates
        pt_hash = self._get_file_hash(pt_path)
        onnx_filename = f"{os.path.splitext(os.path.basename(pt_path))[0]}_{pt_hash[:8]}.onnx"
        onnx_path = os.path.join(onnx_dir, onnx_filename)
        
        # Check if we need to convert
        if not os.path.exists(onnx_path) or force_reconvert:
            print(f"Converting {os.path.basename(pt_path)} to ONNX format...")
            onnx_path = self._convert_to_onnx(pt_path, onnx_path)
            if not onnx_path:
                return False
        else:
            print(f"Using cached ONNX model: {onnx_filename}")
        
        # Load ONNX model
        try:
            if not ONNX_AVAILABLE:
                raise ImportError("onnxruntime not installed")
            
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(onnx_path, providers=providers)
            
            # Get model info
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # Save to config
            self.config['last_pt_path'] = pt_path
            self.config['last_onnx_path'] = onnx_path
            self.config['model_hash'] = pt_hash
            self._save_config()
            
            self.model_loaded = True
            print(f"✅ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
            return False
    
    def _get_file_hash(self, filepath: str) -> str:
        """Get file hash for caching"""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _convert_to_onnx(self, pt_path: str, onnx_path: str) -> Optional[str]:
        """Convert PT model to ONNX
        
        Args:
            pt_path: Input .pt file
            onnx_path: Output .onnx file
            
        Returns:
            Path to ONNX file or None if failed
        """
        try:
            # Check if ultralytics is available
            try:
                from ultralytics import YOLO
            except ImportError:
                print("❌ ultralytics not installed. Cannot convert model.")
                print("   Install with: pip install ultralytics")
                return None
            
            # Load and convert
            print(f"Loading YOLO model...")
            model = YOLO(pt_path)
            
            print(f"Converting to ONNX (this may take a minute)...")
            exported_path = model.export(
                format='onnx',
                simplify=True,
                dynamic=False,
                imgsz=640,
                half=False
            )
            
            # Move to desired location if different
            if exported_path and exported_path != onnx_path:
                import shutil
                shutil.move(exported_path, onnx_path)
            
            print(f"✅ Converted to: {onnx_path}")
            return onnx_path
            
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            return None
    
    def detect_bubbles(self, image_path: str, confidence_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """Detect speech bubbles in image
        
        Args:
            image_path: Path to image
            confidence_threshold: Minimum confidence
            
        Returns:
            List of (x, y, width, height) tuples
        """
        if not self.model_loaded:
            raise RuntimeError("No model loaded. Call load_model() first")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            from PIL import Image
            pil_image = Image.open(image_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        original_height, original_width = image.shape[:2]
        
        # Preprocess
        input_tensor = self._preprocess(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Postprocess
        boxes = self._postprocess(outputs[0], original_width, original_height, confidence_threshold)
        
        return boxes
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for YOLO"""
        # Resize to 640x640
        resized = cv2.resize(image, (640, 640))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW
        chw = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batch = np.expand_dims(chw, axis=0)
        
        return batch
    
    def _postprocess(self, output: np.ndarray, orig_w: int, orig_h: int, conf_thresh: float) -> List[Tuple[int, int, int, int]]:
        """Process YOLO output"""
        boxes = []
        
        # DEBUG: Initialize counters at the start
        total_detections = 0
        filtered_out = 0
        
        # Handle different output formats
        if len(output.shape) == 3:
            predictions = output[0].T
        else:
            predictions = output
        
        # Scale factors
        x_scale = orig_w / 640
        y_scale = orig_h / 640
        
        # DEBUG: Log prediction count
        print(f"\n   YOLO OUTPUT DEBUG:")
        print(f"   - Raw predictions shape: {predictions.shape}")
        print(f"   - Confidence threshold: {conf_thresh:.3f}")
        
        for i, pred in enumerate(predictions):
            if len(pred) >= 5:
                confidence = pred[4]
                total_detections += 1
                
                # DEBUG: Only log first 10 to avoid spam
                if total_detections <= 10:
                    print(f"   Detection {total_detections}: confidence={confidence:.3f}", end="")
                
                if confidence >= conf_thresh:
                    cx, cy, w, h = pred[:4]
                    
                    # Convert to top-left
                    x = (cx - w / 2) * x_scale
                    y = (cy - h / 2) * y_scale
                    width = w * x_scale
                    height = h * y_scale
                    
                    # Clamp to image bounds
                    x = max(0, min(x, orig_w))
                    y = max(0, min(y, orig_h))
                    width = min(width, orig_w - x)
                    height = min(height, orig_h - y)
                    
                    boxes.append((int(x), int(y), int(width), int(height)))
                    
                    if total_detections <= 10:
                        print(f" ✓ ACCEPTED")
                else:
                    filtered_out += 1
                    if total_detections <= 10:
                        print(f" ✗ REJECTED")
        
        # DEBUG: Summary
        print(f"\n   CONFIDENCE FILTER SUMMARY:")
        print(f"   - Total YOLO detections: {total_detections}")
        print(f"   - Accepted (>={conf_thresh:.2f}): {len(boxes)}")
        print(f"   - Filtered out: {filtered_out}")
        if total_detections > 0:
            print(f"   - Acceptance rate: {len(boxes)/total_detections*100:.1f}%")
        print("")
        
        return boxes
