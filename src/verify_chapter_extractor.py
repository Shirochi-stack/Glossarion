#!/usr/bin/env python3
"""Quick verification that Chapter_Extractor loads with new debug code"""
import sys
import os

print("=" * 60)
print("VERIFICATION: Testing Chapter_Extractor module loading")
print("=" * 60)

# Clear any cached imports
if 'Chapter_Extractor' in sys.modules:
    print("⚠️ Chapter_Extractor already in sys.modules, removing...")
    del sys.modules['Chapter_Extractor']

# Now import
print("\n🔄 Importing Chapter_Extractor...")
import Chapter_Extractor

print("\n✅ Import successful!")
print(f"📍 Module location: {Chapter_Extractor.__file__}")

# Check for the new debug function
if hasattr(Chapter_Extractor, 'extract_chapters'):
    print("✅ extract_chapters function found")
else:
    print("❌ extract_chapters function NOT found")

print("\n" + "=" * 60)
print("If you see '[DEBUG_LOAD] Chapter_Extractor.py loaded - CODE_VERSION: 2025.01.07.001'")
print("above this message, the new code is loaded!")
print("=" * 60)
