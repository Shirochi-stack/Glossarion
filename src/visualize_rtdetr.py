"""Visualize RT-DETR detections to debug bubble/free text overlap"""
import sys
import os
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bubble_detector import BubbleDetector

def visualize_detections(image_path, confidence=0.3):
    """Visualize RT-DETR detections with different colors"""
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Initialize detector
    detector = BubbleDetector()
    
    # Load RT-DETR ONNX model
    print("Loading RT-DETR ONNX model...")
    if not detector.load_rtdetr_onnx_model():
        print("❌ Failed to load RT-DETR ONNX model")
        return
    
    # Detect all regions
    print(f"Detecting with confidence >= {confidence}...")
    detections = detector.detect_with_rtdetr_onnx(
        image_path=image_path,
        confidence=confidence,
        return_all_bubbles=False
    )
    
    # Color map
    colors = {
        'bubbles': ('red', 'Empty Bubbles'),
        'text_bubbles': ('blue', 'Text Bubbles'),
        'text_free': ('green', 'Free Text')
    }
    
    # Draw each detection type
    for det_type, (color, label) in colors.items():
        bboxes = detections.get(det_type, [])
        print(f"\\n{label}: {len(bboxes)} detected")
        
        for idx, (x, y, w, h) in enumerate(bboxes):
            # Draw rectangle
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            
            # Draw label
            text = f"{label} {idx + 1}"
            draw.text((x + 5, y + 5), text, fill=color)
            
            print(f"  {idx + 1}. x={x:.0f}, y={y:.0f}, w={w:.0f}, h={h:.0f}")
    
    # Save visualization
    output_path = image_path.replace('.png', '_rtdetr_visualization.png')
    img.save(output_path)
    print(f"\\n✅ Visualization saved to: {output_path}")
    
    # Summary
    print(f"\\n{'='*60}")
    print("DETECTION SUMMARY:")
    print(f"{'='*60}")
    total_bubbles = len(detections.get('bubbles', [])) + len(detections.get('text_bubbles', []))
    total_free_text = len(detections.get('text_free', []))
    print(f"Total bubbles (empty + text): {total_bubbles}")
    print(f"Total free text regions: {total_free_text}")
    
    if total_free_text == 0:
        print(f"\\n⚠️ WARNING: No free text regions detected!")
        print("   This means the free text will NOT be excluded from bubble merging.")
        print("   Possible causes:")
        print("   1. Free text confidence is too low")
        print("   2. RT-DETR model doesn't detect this type of text as 'free text'")
        print("   3. The 'detect_free_text' setting is disabled")
        print(f"\\n   Try lowering the confidence threshold (current: {confidence})")
    
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_rtdetr.py <image_path> [confidence]")
        print("Example: python visualize_rtdetr.py test.png 0.3")
        sys.exit(1)
    
    image_path = sys.argv[1]
    confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    visualize_detections(image_path, confidence)
