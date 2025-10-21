"""
Fast Overlay Compositor for instant text region updates
Caches individual region overlays and composites them on demand
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FastOverlayCompositor:
    """Manages cached text overlays for instant updates when moving regions"""
    
    def __init__(self):
        self.base_image = None  # Cleaned base image (RGB)
        self.region_overlays = {}  # region_index -> PIL RGBA overlay
        self.region_positions = {}  # region_index -> (x, y, w, h)
        self.image_size = None  # (width, height)
        
    def set_base_image(self, image_bgr: np.ndarray):
        """Set the base cleaned image (BGR numpy array)"""
        try:
            from image_utils_cpp import convert_rgb_bgr, is_available as cpp_available
            if cpp_available():
                self.base_image = Image.fromarray(convert_rgb_bgr(image_bgr))
            else:
                self.base_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        except Exception:
            self.base_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        
        self.image_size = self.base_image.size
        logger.info(f"⚡ Set base image: {self.image_size[0]}x{self.image_size[1]}")
    
    def cache_region_overlay(self, region_index: int, overlay: Image.Image, position: Tuple[int, int, int, int]):
        """
        Cache a pre-rendered region overlay
        
        Args:
            region_index: Index of the region
            overlay: PIL RGBA image of the rendered text (full image size)
            position: (x, y, w, h) bounding box
        """
        self.region_overlays[region_index] = overlay
        self.region_positions[region_index] = position
        logger.debug(f"Cached overlay for region {region_index} at {position}")
    
    def update_region_position(self, region_index: int, new_position: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Update a single region's position and return composited result
        
        Args:
            region_index: Index of region to update
            new_position: New (x, y, w, h) position
        
        Returns:
            BGR numpy array of composited result, or None if cache miss
        """
        if region_index not in self.region_overlays:
            logger.warning(f"Region {region_index} not in cache, full re-render needed")
            return None
        
        # Update cached position
        self.region_positions[region_index] = new_position
        
        # Composite all overlays
        return self.composite_all()
    
    def composite_all(self) -> np.ndarray:
        """
        Composite all cached overlays onto base image
        
        Returns:
            BGR numpy array of final image
        """
        if self.base_image is None:
            raise RuntimeError("No base image set")
        
        # Start with base
        result = self.base_image.convert('RGBA')
        
        # Composite each overlay in order
        for region_index in sorted(self.region_overlays.keys()):
            overlay = self.region_overlays[region_index]
            result = Image.alpha_composite(result, overlay)
        
        # Convert back to BGR
        result_rgb = result.convert('RGB')
        result_array = np.array(result_rgb)
        
        try:
            from image_utils_cpp import convert_rgb_bgr, is_available as cpp_available
            if cpp_available():
                return convert_rgb_bgr(result_array)
            else:
                return cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
        except Exception:
            return cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
    
    def render_single_region_fast(self, region_index: int, text: str, new_position: Tuple[int, int, int, int],
                                  font: ImageFont, text_color: Tuple[int, int, int], 
                                  outline_color: Tuple[int, int, int], outline_width: int,
                                  shadow_enabled: bool = False, shadow_color: Tuple[int, int, int] = (0, 0, 0),
                                  shadow_offset: Tuple[int, int] = (2, 2)) -> np.ndarray:
        """
        Quickly render a single region with new position and composite
        
        Args:
            region_index: Index of region
            text: Text to render
            new_position: (x, y, w, h) position
            font: PIL ImageFont
            text_color: RGB tuple
            outline_color: RGB tuple
            outline_width: Width of outline
            shadow_enabled: Whether to draw shadow
            shadow_color: RGB tuple
            shadow_offset: (dx, dy) offset
        
        Returns:
            BGR numpy array of composited result
        """
        if self.base_image is None or self.image_size is None:
            raise RuntimeError("No base image set")
        
        # Create transparent overlay for this region
        overlay = Image.new('RGBA', self.image_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        x, y, w, h = new_position
        
        # Simple text positioning (centered)
        lines = text.split('\n')
        line_height = int(font.size * 1.2)
        total_height = len(lines) * line_height
        start_y = y + (h - total_height) // 2
        
        # Render each line
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            # Get text width
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
            except Exception:
                text_width = len(line) * font.size * 0.6
            
            tx = x + (w - text_width) // 2
            ty = start_y + i * line_height
            
            # Clamp to image bounds
            tx = max(0, min(tx, self.image_size[0] - 10))
            ty = max(0, min(ty, self.image_size[1] - 10))
            
            # Draw shadow
            if shadow_enabled:
                sx, sy = shadow_offset
                draw.text((tx + sx, ty + sy), line, font=font, fill=shadow_color + (255,))
            
            # OPTIMIZED: Use stroke parameter for outline
            try:
                draw.text(
                    (tx, ty), line, font=font,
                    fill=text_color + (255,),
                    stroke_width=outline_width,
                    stroke_fill=outline_color + (255,)
                )
            except TypeError:
                # Fallback for older PIL
                if outline_width > 0:
                    for dx in range(-outline_width, outline_width + 1):
                        for dy in range(-outline_width, outline_width + 1):
                            if dx != 0 or dy != 0:
                                draw.text((tx + dx, ty + dy), line, font=font, fill=outline_color + (255,))
                draw.text((tx, ty), line, font=font, fill=text_color + (255,))
        
        # Cache the overlay
        self.cache_region_overlay(region_index, overlay, new_position)
        
        # Composite and return
        return self.composite_all()
    
    def invalidate_region(self, region_index: int):
        """Remove a region from cache (e.g., when deleted)"""
        if region_index in self.region_overlays:
            del self.region_overlays[region_index]
        if region_index in self.region_positions:
            del self.region_positions[region_index]
        logger.debug(f"Invalidated cache for region {region_index}")
    
    def clear_cache(self):
        """Clear all cached overlays"""
        self.region_overlays.clear()
        self.region_positions.clear()
        logger.info("Cleared overlay cache")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'cached_regions': len(self.region_overlays),
            'base_image_loaded': self.base_image is not None,
            'image_size': self.image_size
        }
