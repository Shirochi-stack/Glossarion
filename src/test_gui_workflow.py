#!/usr/bin/env python3
"""
Test script to verify the GUI translation workflow properly applies existing translations
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, 'C:\\Users\\ADMIN\\Dev\\Glossarion\\src')

def test_apply_existing_translations():
    """Test that the GUI workflow correctly applies existing translations"""
    
    print("=" * 80)
    print("Testing GUI workflow with existing translated_headers.txt")
    print("=" * 80)
    
    # Create a temporary test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n📁 Test directory: {temp_dir}")
        
        # Create sample translated_headers.txt
        translations_file = os.path.join(temp_dir, "translated_headers.txt")
        with open(translations_file, 'w', encoding='utf-8') as f:
            f.write("""Chapter 1:
  Original:   The Beginning
  Translated: El Comienzo
----------------------------------------
Chapter 2:
  Original:   The Journey
  Translated: El Viaje
----------------------------------------
Chapter 3:
  Original:   The End
  Translated: El Final
----------------------------------------
""")
        print(f"✅ Created sample translated_headers.txt")
        
        # Create sample HTML files to update
        html_dir = os.path.join(temp_dir, "html")
        os.makedirs(html_dir)
        
        # Create chapter HTML files with titles to be replaced
        for i, (orig, trans) in enumerate([
            ("The Beginning", "El Comienzo"),
            ("The Journey", "El Viaje"),
            ("The End", "El Final")
        ], 1):
            html_file = os.path.join(html_dir, f"chapter{i:03d}.html")
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Chapter {i}</title>
</head>
<body>
    <h1>{orig}</h1>
    <p>This is chapter {i} content.</p>
</body>
</html>""")
        print(f"✅ Created {i} sample HTML files")
        
        # Test loading translations
        print("\n📖 Testing load_translations_from_file...")
        from translate_headers_standalone import load_translations_from_file
        
        chapters, translations = load_translations_from_file(translations_file)
        print(f"  Loaded {len(chapters)} chapters, {len(translations)} translations")
        for num, title in translations.items():
            print(f"    Chapter {num}: {title}")
        
        # Simulate the GUI workflow code path
        print("\n🔄 Simulating GUI workflow...")
        
        # Set up environment like GUI does
        os.environ['EPUB_PATH'] = 'dummy_path.epub'  # Would be real in actual workflow
        
        # Import the key functions
        from translate_headers_standalone import (
            apply_existing_translations,
            extract_source_chapters_with_opf_mapping,
            match_output_to_source_chapters
        )
        
        # Since we don't have a real EPUB, we'll test the HTML update directly
        print("\n📝 Testing direct HTML update...")
        from metadata_batch_translator import BatchHeaderTranslator
        
        # Create dummy translator
        class DummyClient:
            def translate(self, *args, **kwargs):
                return "Dummy translation"
        
        translator = BatchHeaderTranslator(DummyClient(), {})
        
        # Prepare current titles map like the actual code does
        current_titles_map = {
            1: {'title': 'The Beginning', 'filename': 'chapter001.html'},
            2: {'title': 'The Journey', 'filename': 'chapter002.html'},
            3: {'title': 'The End', 'filename': 'chapter003.html'}
        }
        
        # Apply translations using the translator's method
        translator._update_html_headers_exact(html_dir, translations, current_titles_map)
        
        # Verify HTML files were updated
        print("\n✅ Verifying HTML updates...")
        for i, expected_title in enumerate(["El Comienzo", "El Viaje", "El Final"], 1):
            html_file = os.path.join(html_dir, f"chapter{i:03d}.html")
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if expected_title in content:
                    print(f"  ✓ Chapter {i}: Found '{expected_title}' in HTML")
                else:
                    print(f"  ✗ Chapter {i}: '{expected_title}' NOT found in HTML!")
                    print(f"    Content preview: {content[:200]}...")
        
        print("\n" + "=" * 80)
        print("Test completed!")
        print("=" * 80)

if __name__ == "__main__":
    try:
        test_apply_existing_translations()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()