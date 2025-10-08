#!/usr/bin/env python3
"""
Test Script for Save Config Debugging
This script simulates calling save_config to test the debugging output.
"""

import os
import sys

def test_environment_vars():
    """Test current environment variable state"""
    print("🧪 Testing current environment variable state...")
    
    # Critical variables that should be set by save_config
    test_vars = {
        'OPENROUTER_USE_HTTP_ONLY': 'OpenRouter HTTP setting',
        'OPENROUTER_ACCEPT_IDENTITY': 'OpenRouter identity setting', 
        'EXTRACTION_WORKERS': 'Number of extraction workers',
        'ENABLE_GUI_YIELD': 'GUI yield setting',
        'RETAIN_SOURCE_EXTENSION': 'Source extension retention',
        'GLOSSARY_SYSTEM_PROMPT': 'Manual glossary prompt',
        'AUTO_GLOSSARY_PROMPT': 'Auto glossary prompt',
        'GLOSSARY_CUSTOM_ENTRY_TYPES': 'Custom entry types JSON',
    }
    
    print("\n📊 Current Environment State:")
    print("-" * 50)
    
    set_count = 0
    for var_name, description in test_vars.items():
        value = os.environ.get(var_name)
        
        if value is None:
            print(f"❌ {var_name}: NOT SET")
        elif not value.strip():
            print(f"⚠️  {var_name}: EMPTY")
        else:
            print(f"✅ {var_name}: {len(value)} chars")
            set_count += 1
    
    print(f"\n📈 Summary: {set_count}/{len(test_vars)} variables set")
    
    if set_count == len(test_vars):
        print("🎉 All critical environment variables are set!")
        return True
    else:
        print("⚠️  Some environment variables are missing.")
        print("💡 Run Glossarion and click 'Save Config' to see debugging output.")
        return False

def simulate_save_config_debug():
    """Simulate what the new debugging should show"""
    print("\n🔍 Simulating Save Config Debug Output:")
    print("=" * 50)
    
    # This simulates what you should see when clicking Save Config
    debug_messages = [
        "🔍 [SAVE_CONFIG] Starting comprehensive config save with environment variable debugging...",
        "🔍 [DEBUG] Setting OpenRouter environment variables...",
        "🔍 [SAVE_CONFIG] Verifying environment variables after config save...",
    ]
    
    for msg in debug_messages:
        print(msg)
    
    # Test actual environment state
    test_environment_vars()

if __name__ == "__main__":
    print("🚀 Save Config Debug Test")
    print("=" * 30)
    print("This script tests the environment variable debugging system.")
    print("After the fix, clicking 'Save Config' should show detailed debugging.")
    
    simulate_save_config_debug()
    
    print(f"\n💡 What you should see when clicking Save Config:")
    print("   1. 🔍 [SAVE_CONFIG] Starting comprehensive config save...")
    print("   2. 🔍 [DEBUG] Setting OpenRouter environment variables...")
    print("   3. Environment variable before/after comparisons")
    print("   4. 🔍 [SAVE_CONFIG] Verifying environment variables after save...")
    print("   5. ✅ or ❌ status for each critical variable")
    print("   6. Summary with total issues found")
    
    print(f"\n🔧 If you don't see this output:")
    print("   1. Make sure you're clicking the 'Save Config' button in the main GUI")
    print("   2. Check that the debugging code was properly added to translator_gui.py")
    print("   3. Try enabling debug mode: python enable_debug_mode.py enable")
    print("   4. Use the debug button in Other Settings for more detailed output")