#!/usr/bin/env python3
"""
Test script for PDF HTML extraction with images and CSS generation
"""

import os
import sys
import tempfile

# Test the new PDF extraction features
def test_pdf_extraction():
    print("=" * 60)
    print("PDF HTML Extraction Test")
    print("=" * 60)
    
    # You can test with your own PDF file
    test_pdf = input("Enter path to a test PDF file (or press Enter to skip): ").strip()
    
    if not test_pdf or not os.path.exists(test_pdf):
        print("\n⚠️  No valid PDF file provided. Test skipped.")
        print("\nTo test the new features, run this script again with a PDF file.")
        return
    
    # Create a temporary output directory
    output_dir = tempfile.mkdtemp(prefix="pdf_test_")
    print(f"\n📁 Output directory: {output_dir}")
    
    # Test 1: Extract images
    print("\n" + "=" * 60)
    print("TEST 1: Image Extraction")
    print("=" * 60)
    
    try:
        from pdf_extractor import extract_images_from_pdf
        images = extract_images_from_pdf(test_pdf, output_dir)
        
        if images:
            print(f"✅ Successfully extracted images from {len(images)} pages")
            total_imgs = sum(len(imgs) for imgs in images.values())
            print(f"   Total images: {total_imgs}")
        else:
            print("ℹ️  No images found in PDF")
    except Exception as e:
        print(f"❌ Image extraction failed: {e}")
    
    # Test 2: Generate CSS
    print("\n" + "=" * 60)
    print("TEST 2: CSS Generation")
    print("=" * 60)
    
    try:
        from pdf_extractor import generate_css_from_pdf
        css = generate_css_from_pdf(test_pdf)
        
        css_path = os.path.join(output_dir, "test_styles.css")
        with open(css_path, 'w', encoding='utf-8') as f:
            f.write(css)
        
        print(f"✅ Generated CSS file: {css_path}")
        print(f"   CSS length: {len(css)} characters")
    except Exception as e:
        print(f"❌ CSS generation failed: {e}")
    
    # Test 3: Extract with formatting
    print("\n" + "=" * 60)
    print("TEST 3: HTML Extraction with Formatting")
    print("=" * 60)
    
    try:
        from pdf_extractor import extract_pdf_with_formatting
        html, images = extract_pdf_with_formatting(test_pdf, output_dir, extract_images=True)
        
        html_path = os.path.join(output_dir, "test_output.html")
        
        # Create full HTML document
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test PDF Extraction</title>
    <link rel="stylesheet" href="test_styles.css">
</head>
<body>
{html}
</body>
</html>"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        print(f"✅ Generated HTML file: {html_path}")
        print(f"   HTML length: {len(html)} characters")
        print(f"   Images extracted: {sum(len(imgs) for imgs in images.values())}")
    except Exception as e:
        print(f"❌ HTML extraction failed: {e}")
    
    # Test 4: Test with txt_processor
    print("\n" + "=" * 60)
    print("TEST 4: Full Pipeline Test with txt_processor")
    print("=" * 60)
    
    try:
        # Set environment variables for testing
        os.environ["PDF_OUTPUT_FORMAT"] = "html"
        os.environ["PDF_EXTRACT_IMAGES"] = "1"
        os.environ["PDF_GENERATE_CSS"] = "1"
        os.environ["USE_HTML2TEXT"] = "0"
        
        from txt_processor import TextFileProcessor
        
        processor = TextFileProcessor(test_pdf, output_dir)
        chapters = processor.extract_chapters()
        
        print(f"✅ Extracted {len(chapters)} chapter(s)")
        
        # Check word_count folder
        word_count_dir = os.path.join(output_dir, 'word_count')
        if os.path.exists(word_count_dir):
            files = os.listdir(word_count_dir)
            html_files = [f for f in files if f.endswith('.html')]
            print(f"   HTML section files in word_count: {len(html_files)}")
            
            # Check for images folder
            images_dir = os.path.join(word_count_dir, 'images')
            if os.path.exists(images_dir):
                img_files = os.listdir(images_dir)
                print(f"   Images in word_count/images: {len(img_files)}")
        
        # Check CSS
        css_path = os.path.join(output_dir, 'styles.css')
        if os.path.exists(css_path):
            print(f"   ✅ CSS file generated: styles.css")
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Test markdown conversion
    print("\n" + "=" * 60)
    print("TEST 5: Markdown Conversion Test")
    print("=" * 60)
    
    try:
        os.environ["PDF_OUTPUT_FORMAT"] = "markdown"
        os.environ["USE_HTML2TEXT"] = "1"
        
        from txt_processor import TextFileProcessor
        
        md_output_dir = tempfile.mkdtemp(prefix="pdf_test_md_")
        processor = TextFileProcessor(test_pdf, md_output_dir)
        chapters = processor.extract_chapters()
        
        print(f"✅ Extracted {len(chapters)} chapter(s) in Markdown format")
        
        # Check word_count folder
        word_count_dir = os.path.join(md_output_dir, 'word_count')
        if os.path.exists(word_count_dir):
            files = os.listdir(word_count_dir)
            md_files = [f for f in files if f.endswith('.md')]
            print(f"   Markdown section files in word_count: {len(md_files)}")
            
            if md_files:
                # Show sample content
                sample_file = os.path.join(word_count_dir, md_files[0])
                with open(sample_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"   Sample content length: {len(content)} characters")
                print(f"   Contains <img> tags: {'<img' in content}")
        
        print(f"\n📁 Markdown output directory: {md_output_dir}")
        
    except Exception as e:
        print(f"❌ Markdown conversion test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Tests completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"\nYou can now:")
    print(f"  1. Open {os.path.join(output_dir, 'test_output.html')} in a browser")
    print(f"  2. Check the images folder: {os.path.join(output_dir, 'images')}")
    print(f"  3. Review the CSS file: {os.path.join(output_dir, 'test_styles.css')}")
    print(f"  4. Check word_count sections: {os.path.join(output_dir, 'word_count')}")


if __name__ == "__main__":
    try:
        test_pdf_extraction()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
