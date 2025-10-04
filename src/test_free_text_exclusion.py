"""Test script to debug free text exclusion during bubble merging"""
import os
import sys

# Test the free text exclusion logic
def test_point_in_bbox():
    """Test the bounding box overlap logic"""
    
    # Example from your image:
    # Speech bubble: "ANY OTHER BIDDERS?!" at top-left
    # Free text: "ONE MILLION! AN PACHI HAS BEEN MADE!" in center
    
    # Let's say bubble is at x=50, y=50, w=200, h=100
    bubble = (50, 50, 200, 100)
    bx, by, bw, bh = bubble
    
    # And free text region is at x=100, y=150, w=150, h=80
    free_text = (100, 150, 150, 80)
    fx, fy, fw, fh = free_text
    
    # OCR might detect text in free text area at x=120, y=170, w=50, h=20
    ocr_region = (120, 170, 50, 20)
    rx, ry, rw, rh = ocr_region
    
    # Calculate center
    region_center_x = rx + rw / 2  # 145
    region_center_y = ry + rh / 2  # 180
    
    print(f"Bubble bbox: x={bx}, y={by}, w={bw}, h={bh}")
    print(f"   Range: x=[{bx}, {bx+bw}], y=[{by}, {by+bh}]")
    print()
    print(f"Free text bbox: x={fx}, y={fy}, w={fw}, h={fh}")
    print(f"   Range: x=[{fx}, {fx+fw}], y=[{fy}, {fy+fh}]")
    print()
    print(f"OCR region: x={rx}, y={ry}, w={rw}, h={rh}")
    print(f"   Center: ({region_center_x}, {region_center_y})")
    print()
    
    # Check if center is in bubble
    in_bubble = (bx <= region_center_x <= bx + bw and 
                 by <= region_center_y <= by + bh)
    print(f"Is OCR center in bubble? {in_bubble}")
    
    # Check if center is in free text
    in_free_text = (fx <= region_center_x <= fx + fw and 
                    fy <= region_center_y <= fy + fh)
    print(f"Is OCR center in free text? {in_free_text}")
    print()
    
    if in_bubble and in_free_text:
        print("❌ PROBLEM: Center is in BOTH bubble and free text!")
        print("   This should be EXCLUDED from bubble merging")
    elif in_bubble:
        print("✅ Center is only in bubble - merge is OK")
    elif in_free_text:
        print("✅ Center is only in free text - stays separate")
    else:
        print("⚠️ Center is outside both regions")
    
    print()
    print("="*60)
    print("EXPECTED BEHAVIOR:")
    print("="*60)
    print("1. RT-DETR detects:")
    print("   - text_bubbles: Speech bubble bboxes")
    print("   - text_free: Free text bboxes")
    print()
    print("2. OCR detects all text (doesn't know types)")
    print()
    print("3. Merging logic:")
    print("   - For each bubble bbox:")
    print("     - Find OCR regions with center in bubble")
    print("     - CHECK: Is center also in free_text bbox?")
    print("       - YES → SKIP (don't merge)")
    print("       - NO → Merge into bubble")
    print()
    print("4. Result:")
    print("   - Bubble text merged together")
    print("   - Free text stays separate")

if __name__ == "__main__":
    test_point_in_bbox()
    
    print()
    print("="*60)
    print("DEBUGGING CHECKLIST:")
    print("="*60)
    print("1. Check RT-DETR detection logs:")
    print("   ✓ How many text_bubbles detected?")
    print("   ✓ How many text_free detected?")
    print()
    print("2. Check free text exclusion zone logs:")
    print("   ✓ Are free text bboxes populated?")
    print("   ✓ What are their coordinates?")
    print()
    print("3. Check bubble merging logs:")
    print("   ✓ Which regions are being checked?")
    print("   ✓ Are any being skipped due to free text overlap?")
    print()
    print("4. If free text is STILL being merged:")
    print("   → RT-DETR might not be detecting the free text region")
    print("   → Check RT-DETR confidence threshold")
    print("   → Check if 'detect_free_text' setting is enabled")
    print("   → Visualize RT-DETR detections to verify")
