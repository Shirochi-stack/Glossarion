#!/usr/bin/env python3
"""Test that importlib.reload actually reloads Chapter_Extractor"""
import sys
import importlib

print("=" * 60)
print("TEST: Verifying importlib.reload() functionality")
print("=" * 60)

# First import
print("\n1️⃣ First import:")
import Chapter_Extractor
print(f"   Module location: {Chapter_Extractor.__file__}")

# Second import (should use cache - NO reload message)
print("\n2️⃣ Second import (no reload):")
import Chapter_Extractor
print(f"   Module location: {Chapter_Extractor.__file__}")

# Third import WITH reload (should show reload message)
print("\n3️⃣ Third import WITH reload:")
importlib.reload(Chapter_Extractor)
print(f"   Module location: {Chapter_Extractor.__file__}")

print("\n" + "=" * 60)
print("EXPECTED OUTPUT:")
print("- You should see '[DEBUG_LOAD] Chapter_Extractor.py loaded...'")
print("  TWO times above (once for first import, once for reload)")
print("=" * 60)
