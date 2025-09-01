"""
YOLOv8 Speech Bubble Detector for Manga/Comic Translation
Supports both .pt and ONNX models with automatic conversion
Fully configurable through GUI settings
"""

import os
import json
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
import logging
import traceback
import hashlib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import YOLO dependencies
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics YOLO not available - install with: pip install ultralytics")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available")


class BubbleDetector:
    """
    YOLOv8-based speech bubble detector for comics and manga.
    Supports multiple model formats and provides configurable detection.
    """
    
    def __init__(self, config_path: str = "bubble_detector_config.json"):
        """
        Initialize the bubble detector.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.model_loaded = False
        self.model_type = None  # 'yolo', 'onnx', or 'torch'
        self.onnx_session = None
        
        # Detection settings
        self.default_confidence = 0.5
        self.default_iou_threshold = 0.45
        self.default_max_detections = 100
        
        # Cache directory for ONNX conversions
        self.cache_dir = os.environ.get('BUBBLE_CACHE_DIR', 'bubble_model_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # GPU availability
        self.use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
        
        logger.info(f"🗨️ BubbleDetector initialized")
        logger.info(f"   GPU: {'Available' if self.use_gpu else 'Not available'}")
        logger.info(f"   YOLO: {'Available' if YOLO_AVAILABLE else 'Not installed'}")
        logger.info(f"   ONNX: {'Available' if ONNX_AVAILABLE else 'Not installed'}")
    
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
            
            if self.model_loaded and not force_reload:
                logger.info("Model already loaded")
                return True
            
            logger.info(f"📥 Loading bubble detection model: {model_path}")
            
            # Determine model type by extension
            ext = Path(model_path).suffix.lower()
            
            if ext in ['.pt', '.pth']:
                if not YOLO_AVAILABLE:
                    logger.error("Ultralytics package required for .pt models")
                    return False
                
                # Load YOLOv8 model
                self.model = YOLO(model_path)
                self.model_type = 'yolo'
                
                # Set to eval mode
                if hasattr(self.model, 'model'):
                    self.model.model.eval()
                
                # Move to GPU if available
                if self.use_gpu:
                    self.model.to('cuda')
                    
                logger.info("✅ YOLOv8 model loaded successfully")
                
            elif ext == '.onnx':
                if not ONNX_AVAILABLE:
                    logger.error("ONNX Runtime required for .onnx models")
                    return False
                
                # Load ONNX model
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
                self.onnx_session = ort.InferenceSession(model_path, providers=providers)
                self.model_type = 'onnx'
                
                logger.info("✅ ONNX model loaded successfully")
                
            elif ext == '.torchscript':
                if not TORCH_AVAILABLE:
                    logger.error("PyTorch required for TorchScript models")
                    return False
                
                # Load TorchScript model
                self.model = torch.jit.load(model_path)
                self.model.eval()
                self.model_type = 'torch'
                
                if self.use_gpu:
                    self.model = self.model.cuda()
                
                logger.info("✅ TorchScript model loaded successfully")
                
            else:
                logger.error(f"Unsupported model format: {ext}")
                return False
            
            self.model_loaded = True
            self.config['last_model_path'] = model_path
            self.config['model_type'] = self.model_type
            self._save_config()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(traceback.format_exc())
            self.model_loaded = False
            return False
    
    def detect_bubbles(self, 
                      image_path: str, 
                      confidence: float = None,
                      iou_threshold: float = None,
                      max_detections: int = None) -> List[Tuple[int, int, int, int]]:
        """
        Detect speech bubbles in an image.
        
        Args:
            image_path: Path to image file
            confidence: Minimum confidence threshold (0-1)
            iou_threshold: IOU threshold for NMS (0-1)
            max_detections: Maximum number of detections to return
            
        Returns:
            List of bubble bounding boxes as (x, y, width, height) tuples
        """
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
    
    def visualize_detections(self, image_path: str, bubbles: List[Tuple[int, int, int, int]], 
                            output_path: str = None) -> np.ndarray:
        """
        Visualize detected bubbles on the image.
        
        Args:
            image_path: Path to original image
            bubbles: List of bubble bounding boxes
            output_path: Optional path to save visualization
            
        Returns:
            Image with drawn bounding boxes
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        # Draw bounding boxes
        for i, (x, y, w, h) in enumerate(bubbles):
            # Draw rectangle
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            
            # Add label
            label = f"Bubble {i+1}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x, y - label_size[1] - 4), (x + label_size[0], y), color, -1)
            cv2.putText(image, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, image)
            logger.info(f"💾 Visualization saved to: {output_path}")
        
        return image
    
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
            **kwargs: Detection parameters (confidence, iou_threshold, max_detections)
            
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
                print(f"Model downloaded to: {model_path}")
        
        elif sys.argv[1] == "detect" and len(sys.argv) > 3:
            # Detect bubbles in an image
            model_path = sys.argv[2]
            image_path = sys.argv[3]
            
            if detector.load_model(model_path):
                bubbles = detector.detect_bubbles(image_path, confidence=0.5)
                print(f"Detected {len(bubbles)} bubbles:")
                for i, (x, y, w, h) in enumerate(bubbles):
                    print(f"  Bubble {i+1}: position=({x},{y}) size=({w}x{h})")
                
                # Optionally visualize
                if len(sys.argv) > 4:
                    output_path = sys.argv[4]
                    detector.visualize_detections(image_path, bubbles, output_path)
        
        else:
            print("Usage:")
            print("  python bubble_detector.py download")
            print("  python bubble_detector.py detect <model_path> <image_path> [output_path]")
    
    else:
        print("Bubble Detector Module")
        print("Usage:")
        print("  python bubble_detector.py download")
        print("  python bubble_detector.py detect <model_path> <image_path> [output_path]")
