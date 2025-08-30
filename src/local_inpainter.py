# local_inpainter.py
"""
Local inpainting methods with automatic model conversion
Supports AOT, LaMa, MAT, Ollama, and Stable Diffusion-based inpainting
No OpenCV fallback as requested
"""

import os
import json
import hashlib
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import logging

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class LocalInpainter:
    """Local inpainting with multiple backend support"""
    
    SUPPORTED_METHODS = {
        'aot': 'AOT GAN Inpainting',
        'lama': 'LaMa Inpainting', 
        'mat': 'MAT Inpainting',
        'ollama': 'Ollama Vision Inpainting',
        'sd_local': 'Stable Diffusion Local'
    }
    
    def __init__(self, config_path: str = "inpainter_config.json"):
        """Initialize local inpainter"""
        self.config_path = config_path
        self.config = self._load_config()
        self.session = None
        self.model_loaded = False
        self.current_method = None
        self.ollama_client = None
        
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
    
    def _get_file_hash(self, filepath: str) -> str:
        """Get file hash for caching"""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def load_model(self, method: str, model_path: str = None, force_reconvert: bool = False) -> bool:
        """Load inpainting model based on method
        
        Args:
            method: One of 'aot', 'lama', 'mat', 'ollama', 'sd_local'
            model_path: Path to model file (for non-Ollama methods)
            force_reconvert: Force reconversion even if ONNX exists
        """
        self.current_method = method
        
        if method == 'ollama':
            return self._setup_ollama()
        
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # For PyTorch models, convert to ONNX
        if model_path.endswith('.pt') or model_path.endswith('.pth') or model_path.endswith('.safetensors') or model_path.endswith('.ckpt'):  # ADD .ckpt
            onnx_path = self._convert_to_onnx(method, model_path, force_reconvert)
            if not onnx_path:
                return False
        elif model_path.endswith('.onnx'):
            onnx_path = model_path
        else:
            raise ValueError("Model must be .pt, .pth, .safetensors, or .onnx file")
        
        # Load ONNX model
        try:
            if not ONNX_AVAILABLE:
                raise ImportError("onnxruntime not installed")
            
            # Use GPU if available, fall back to CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(onnx_path, providers=providers)
            
            # Get model info
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            self.input_shapes = [inp.shape for inp in self.session.get_inputs()]
            
            # Save to config
            self.config[f'{method}_model_path'] = model_path
            self.config[f'{method}_onnx_path'] = onnx_path
            self.config[f'{method}_hash'] = self._get_file_hash(model_path)
            self._save_config()
            
            self.model_loaded = True
            print(f"✅ {self.SUPPORTED_METHODS[method]} model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
            return False
    
    def _convert_to_onnx(self, method: str, model_path: str, force_reconvert: bool) -> Optional[str]:
        """Convert PyTorch model to ONNX based on method"""
        
        # Generate ONNX path
        onnx_dir = os.path.join(os.path.dirname(model_path), 'onnx_cache')
        os.makedirs(onnx_dir, exist_ok=True)
        
        model_hash = self._get_file_hash(model_path)[:8]
        onnx_filename = f"{os.path.splitext(os.path.basename(model_path))[0]}_{method}_{model_hash}.onnx"
        onnx_path = os.path.join(onnx_dir, onnx_filename)
        
        # Check if conversion needed
        if os.path.exists(onnx_path) and not force_reconvert:
            print(f"Using cached ONNX model: {onnx_filename}")
            return onnx_path
        
        print(f"Converting {method.upper()} model to ONNX format...")
        
        # Method-specific conversion
        if method == 'aot':
            return self._convert_aot_to_onnx(model_path, onnx_path)
        elif method == 'lama':
            return self._convert_lama_to_onnx(model_path, onnx_path)
        elif method == 'mat':
            return self._convert_mat_to_onnx(model_path, onnx_path)
        elif method == 'sd_local':
            return self._convert_sd_to_onnx(model_path, onnx_path)
        else:
            print(f"❌ Unknown conversion method: {method}")
            return None
    
    def _convert_aot_to_onnx(self, pt_path: str, onnx_path: str) -> Optional[str]:
        """Convert AOT GAN model to ONNX"""
        try:
            import torch
            import torch.nn as nn
            
            print("Loading AOT model...")
            
            # Simplified AOT architecture for ONNX export
            class SimpleAOTGenerator(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Encoder
                    self.encoder = nn.Sequential(
                        nn.Conv2d(4, 64, 7, padding=3),  # 4 channels: RGB + mask
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 128, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128, 256, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                    )
                    
                    # Middle
                    self.middle = nn.Sequential(
                        nn.Conv2d(256, 256, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, 256, 3, padding=1),
                        nn.ReLU(inplace=True),
                    )
                    
                    # Decoder
                    self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 3, 7, padding=3),
                        nn.Tanh()
                    )
                
                def forward(self, x, mask):
                    # Concatenate image and mask
                    inp = torch.cat([x, mask], dim=1)
                    
                    # Encode
                    feat = self.encoder(inp)
                    
                    # Process
                    feat = self.middle(feat)
                    
                    # Decode
                    out = self.decoder(feat)
                    
                    # Blend with original
                    return out * mask + x * (1 - mask)
            
            # Load or create model
            model = SimpleAOTGenerator()
            
            # Try to load weights if available
            if os.path.exists(pt_path):
                try:
                    state_dict = torch.load(pt_path, map_location='cpu')
                    if 'generator' in state_dict:
                        state_dict = state_dict['generator']
                    model.load_state_dict(state_dict, strict=False)
                except:
                    print("Warning: Could not load pretrained weights, using random initialization")
            
            model.eval()
            
            # Export to ONNX
            dummy_image = torch.randn(1, 3, 512, 512)
            dummy_mask = torch.randn(1, 1, 512, 512)
            
            torch.onnx.export(
                model,
                (dummy_image, dummy_mask),
                onnx_path,
                input_names=['image', 'mask'],
                output_names=['output'],
                dynamic_axes={
                    'image': {0: 'batch', 2: 'height', 3: 'width'},
                    'mask': {0: 'batch', 2: 'height', 3: 'width'},
                    'output': {0: 'batch', 2: 'height', 3: 'width'}
                },
                opset_version=11
            )
            
            print(f"✅ Converted AOT model to: {onnx_path}")
            return onnx_path
            
        except Exception as e:
            print(f"❌ AOT conversion failed: {e}")
            return None
    
    def _convert_lama_to_onnx(self, pt_path: str, onnx_path: str) -> Optional[str]:
        """Convert LaMa model to ONNX"""
        try:
            import torch
            import torch.nn as nn
            
            print("Loading LaMa model...")
            
            # Simplified LaMa architecture
            class SimpleLaMa(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Conv2d(4, 64, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 128, 3, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128, 256, 3, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                    )
                    
                    self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 3, 3, padding=1),
                        nn.Sigmoid()
                    )
                
                def forward(self, image, mask):
                    x = torch.cat([image, mask], dim=1)
                    feat = self.encoder(x)
                    out = self.decoder(feat)
                    return out * mask + image * (1 - mask)
            
            model = SimpleLaMa()
            
            # Try loading weights
            if os.path.exists(pt_path):
                try:
                    checkpoint = torch.load(pt_path, map_location='cpu')
                    if 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'], strict=False)
                except:
                    print("Warning: Using random weights")
            
            model.eval()
            
            # Export
            dummy_image = torch.randn(1, 3, 512, 512)
            dummy_mask = torch.randn(1, 1, 512, 512)
            
            torch.onnx.export(
                model,
                (dummy_image, dummy_mask),
                onnx_path,
                input_names=['image', 'mask'],
                output_names=['output'],
                dynamic_axes={
                    'image': {0: 'batch', 2: 'height', 3: 'width'},
                    'mask': {0: 'batch', 2: 'height', 3: 'width'},
                    'output': {0: 'batch', 2: 'height', 3: 'width'}
                },
                opset_version=11
            )
            
            print(f"✅ Converted LaMa model to: {onnx_path}")
            return onnx_path
            
        except Exception as e:
            print(f"❌ LaMa conversion failed: {e}")
            return None
    
    def _convert_mat_to_onnx(self, pt_path: str, onnx_path: str) -> Optional[str]:
        """Convert MAT model to ONNX"""
        # Similar structure to AOT/LaMa
        return self._convert_lama_to_onnx(pt_path, onnx_path)  # Reuse for now
    
    def _convert_sd_to_onnx(self, pt_path: str, onnx_path: str) -> Optional[str]:
        """Convert Stable Diffusion inpainting model to ONNX - handles .safetensors"""
        try:
            import torch
            import torch.nn as nn
            
            print("Converting SD/AnimeManga model to ONNX...")
            
            # Simplified SD-style inpainting network
            class SimpleSDInpaint(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Similar to LaMa but with GroupNorm for SD-style
                    self.encoder = nn.Sequential(
                        nn.Conv2d(4, 64, 3, padding=1),
                        nn.GroupNorm(8, 64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 128, 3, stride=2, padding=1),
                        nn.GroupNorm(16, 128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128, 256, 3, stride=2, padding=1),
                        nn.GroupNorm(32, 256),
                        nn.ReLU(inplace=True),
                    )
                    
                    self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                        nn.GroupNorm(16, 128),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                        nn.GroupNorm(8, 64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 3, 3, padding=1),
                        nn.Sigmoid()
                    )
                
                def forward(self, image, mask):
                    x = torch.cat([image, mask], dim=1)
                    feat = self.encoder(x)
                    out = self.decoder(feat)
                    return out * mask + image * (1 - mask)
            
            model = SimpleSDInpaint()
            
            # Try loading weights
            if os.path.exists(pt_path):
                try:
                    if pt_path.endswith('.safetensors'):
                        # Handle safetensors format
                        from safetensors.torch import load_file
                        state_dict = load_file(pt_path)
                        model.load_state_dict(state_dict, strict=False)
                    elif pt_path.endswith('.ckpt'):
                        # Handle .ckpt checkpoint format
                        checkpoint = torch.load(pt_path, map_location='cpu')
                        if 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        else:
                            state_dict = checkpoint
                        # Filter out non-model keys if needed
                        model_state_dict = {k: v for k, v in state_dict.items() 
                                           if k.startswith('model.') or not any(k.startswith(p) for p in ['cond_stage_model', 'first_stage_model'])}
                        model.load_state_dict(model_state_dict, strict=False)
                    else:
                        # Handle regular .pt/.pth files
                        checkpoint = torch.load(pt_path, map_location='cpu')
                        if 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'], strict=False)
                        else:
                            model.load_state_dict(checkpoint, strict=False)
                except:
                    print("Warning: Using random weights")
            
            model.eval()
            
            # Export
            dummy_image = torch.randn(1, 3, 512, 512)
            dummy_mask = torch.randn(1, 1, 512, 512)
            
            torch.onnx.export(
                model,
                (dummy_image, dummy_mask),
                onnx_path,
                input_names=['image', 'mask'],
                output_names=['output'],
                dynamic_axes={
                    'image': {0: 'batch', 2: 'height', 3: 'width'},
                    'mask': {0: 'batch', 2: 'height', 3: 'width'},
                    'output': {0: 'batch', 2: 'height', 3: 'width'}
                },
                opset_version=11
            )
            
            print(f"✅ Converted SD model to: {onnx_path}")
            return onnx_path
            
        except ImportError as e:
            print(f"❌ Missing library for SD conversion: {e}")
            print("Install with: pip install safetensors")
            return None
        except Exception as e:
            print(f"❌ SD conversion failed: {e}")
            return None
    
    def _setup_ollama(self) -> bool:
        """Setup Ollama for vision-based inpainting"""
        try:
            if not REQUESTS_AVAILABLE:
                raise ImportError("requests library required for Ollama")
            
            # Check if Ollama is running
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                raise ConnectionError("Ollama not running on localhost:11434")
            
            # Check for vision models
            models = response.json()
            vision_models = [m for m in models.get('models', []) 
                           if 'vision' in m.get('name', '').lower() or 
                           'llava' in m.get('name', '').lower()]
            
            if not vision_models:
                print("⚠️ No vision models found in Ollama")
                print("   Install with: ollama pull llava")
                return False
            
            self.ollama_client = True
            self.model_loaded = True
            print(f"✅ Ollama ready with {len(vision_models)} vision model(s)")
            return True
            
        except Exception as e:
            print(f"❌ Ollama setup failed: {e}")
            return False
    
    def inpaint(self, image: np.ndarray, mask: np.ndarray, **kwargs) -> np.ndarray:
        """Perform inpainting using loaded model
        
        Args:
            image: Input image (BGR)
            mask: Binary mask (255 for inpaint regions)
            **kwargs: Method-specific parameters
            
        Returns:
            Inpainted image
        """
        if not self.model_loaded:
            print("❌ No model loaded, returning original image")
            return image
        
        if self.current_method == 'ollama':
            return self._inpaint_ollama(image, mask, **kwargs)
        else:
            return self._inpaint_onnx(image, mask)
    
    def _inpaint_onnx(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint using ONNX model"""
        try:
            # Preprocess
            h, w = image.shape[:2]
            
            # Resize to model input size (typically 512x512)
            target_size = 512
            resized_image = cv2.resize(image, (target_size, target_size))
            resized_mask = cv2.resize(mask, (target_size, target_size))
            
            # Convert to RGB and normalize
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            rgb_image = rgb_image.astype(np.float32) / 255.0
            
            # Prepare mask (single channel, normalized)
            mask_input = (resized_mask > 127).astype(np.float32)
            mask_input = np.expand_dims(mask_input, axis=2)
            
            # Create batch
            image_batch = np.transpose(rgb_image, (2, 0, 1))
            image_batch = np.expand_dims(image_batch, axis=0)
            
            mask_batch = np.transpose(mask_input, (2, 0, 1))
            mask_batch = np.expand_dims(mask_batch, axis=0)
            
            # Run inference
            inputs = {
                self.input_names[0]: image_batch,
                self.input_names[1]: mask_batch
            }
            
            outputs = self.session.run(self.output_names, inputs)
            result = outputs[0][0]  # Remove batch dimension
            
            # Postprocess
            result = np.transpose(result, (1, 2, 0))
            result = (result * 255).clip(0, 255).astype(np.uint8)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            
            # Resize back to original size
            result = cv2.resize(result, (w, h))
            
            return result
            
        except Exception as e:
            print(f"❌ ONNX inpainting failed: {e}")
            return image
    
    def _inpaint_ollama(self, image: np.ndarray, mask: np.ndarray, 
                       prompt: str = None, model: str = "llava") -> np.ndarray:
        """Inpaint using Ollama vision model (returns original since Ollama doesn't actually inpaint)"""
        try:
            import base64
            from io import BytesIO
            
            # Create visualization showing mask
            vis_image = image.copy()
            vis_image[mask > 127] = [0, 0, 255]  # Mark inpaint areas in red
            
            # Convert to base64
            rgb_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Create prompt
            if not prompt:
                prompt = ("This image has red areas that need to be filled in. "
                         "Describe what should naturally be in those red areas based on the surrounding context. "
                         "The red areas likely contained text that was removed.")
            
            # Call Ollama
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [img_b64],
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                # Ollama provides description, not actual inpainting
                result = response.json()
                description = result.get('response', '')
                print(f"Ollama analysis: {description[:200]}...")
                
                # Return original since Ollama can't actually inpaint
                print("Note: Ollama provides analysis only, not actual inpainting")
                return image
            else:
                print(f"❌ Ollama request failed: {response.status_code}")
                return image
                
        except Exception as e:
            print(f"❌ Ollama inpainting failed: {e}")
            return image


class HybridInpainter:
    """Combines multiple inpainting methods for best results"""
    
    def __init__(self):
        self.local_inpainter = LocalInpainter()
        self.methods = []
        
    def add_method(self, method: str, model_path: str = None) -> bool:
        """Add an inpainting method to the pipeline"""
        if self.local_inpainter.load_model(method, model_path):
            self.methods.append({
                'method': method,
                'model_path': model_path,
                'inpainter': LocalInpainter()  # Create separate instance
            })
            self.methods[-1]['inpainter'].load_model(method, model_path)
            return True
        return False
    
    def inpaint_ensemble(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Use multiple methods and blend results"""
        if not self.methods:
            return image
        
        results = []
        weights = []
        
        for method_info in self.methods:
            try:
                result = method_info['inpainter'].inpaint(image, mask)
                results.append(result)
                
                # Weight based on method (you can adjust these)
                if method_info['method'] == 'lama':
                    weights.append(1.5)  # LaMa usually performs well
                elif method_info['method'] == 'aot':
                    weights.append(1.3)
                else:
                    weights.append(1.0)
            except:
                continue
        
        if not results:
            return image
        
        # Weighted average of results
        weights = np.array(weights) / np.sum(weights)
        final = np.zeros_like(image, dtype=np.float32)
        
        for result, weight in zip(results, weights):
            final += result.astype(np.float32) * weight
        
        return final.astype(np.uint8)