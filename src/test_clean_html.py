import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from pdf_extractor import _extract_absolute_html

def test_clean_html():
    """Test that clean HTML is generated without positioning styles"""
    
    # Use a sample PDF if available
    test_pdf = input("Enter path to test PDF file: ").strip().strip('"')
    
    if not os.path.exists(test_pdf):
        print("❌ PDF file not found")
        return
    
    output_dir = os.path.join(os.path.dirname(test_pdf), "test_output")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Testing Dual HTML Extraction")
    print("="*60 + "\n")
    
    # Test page-by-page mode
    print("📄 Testing page-by-page mode...")
    positioned_pages, images_by_page, clean_pages = _extract_absolute_html(
        test_pdf, 
        output_dir, 
        page_by_page=True
    )
    
    print(f"\n✓ Extracted {len(positioned_pages)} positioned pages")
    print(f"✓ Extracted {len(clean_pages)} clean pages")
    print(f"✓ Found {sum(len(imgs) for imgs in images_by_page.values())} images")
    
    # Show first page comparison
    if positioned_pages and clean_pages:
        page_num_pos, pos_html = positioned_pages[0]
        page_num_clean, clean_html = clean_pages[0]
        
        print(f"\n--- Page {page_num_pos} Positioned HTML Sample (first 500 chars) ---")
        print(pos_html[:500])
        print("\n--- Page {page_num_clean} Clean HTML Sample (first 500 chars) ---")
        print(clean_html[:500])
        
        # Check for positioning styles in each
        has_pos_styles = 'position:absolute' in pos_html
        has_clean_styles = 'position:absolute' in clean_html
        
        print(f"\n📊 Analysis:")
        print(f"  Positioned HTML has absolute positioning: {has_pos_styles} {'✓' if has_pos_styles else '❌'}")
        print(f"  Clean HTML has absolute positioning: {has_clean_styles} {'✓ FAIL!' if has_clean_styles else '❌ PASS'}")
        
        # Check payload efficiency
        pos_size = len(pos_html)
        clean_size = len(clean_html)
        reduction = ((pos_size - clean_size) / pos_size * 100) if pos_size > 0 else 0
        
        print(f"\n💾 Size Comparison (Page {page_num_pos}):")
        print(f"  Positioned HTML: {pos_size:,} bytes")
        print(f"  Clean HTML: {clean_size:,} bytes")
        print(f"  Reduction: {reduction:.1f}%")
        
        # Write samples to files
        with open(os.path.join(output_dir, "sample_positioned.html"), 'w', encoding='utf-8') as f:
            f.write(pos_html)
        with open(os.path.join(output_dir, "sample_clean.html"), 'w', encoding='utf-8') as f:
            f.write(clean_html)
        
        print(f"\n✓ Sample files written to: {output_dir}")
    
    print("\n" + "="*60)
    print("✅ Test completed!")
    print("="*60)

if __name__ == "__main__":
    test_clean_html()
