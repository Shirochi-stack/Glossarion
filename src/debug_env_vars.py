#!/usr/bin/env python3
"""
Environment Variable Debugging Helper
This script helps you debug environment variable initialization issues.
"""

import os
import sys
import json

def print_env_var_status():
    """Print the current status of all critical environment variables."""
    print("🔍 Current Environment Variable Status:")
    print("=" * 60)
    
    # Critical environment variables that should always be set
    critical_env_vars = {
        # Glossary-related
        'GLOSSARY_SYSTEM_PROMPT': 'Manual glossary extraction prompt',
        'AUTO_GLOSSARY_PROMPT': 'Auto glossary generation prompt', 
        'GLOSSARY_CUSTOM_ENTRY_TYPES': 'Custom entry types configuration (JSON)',
        'GLOSSARY_DISABLE_HONORIFICS_FILTER': 'Honorifics filter disable flag',
        'GLOSSARY_STRIP_HONORIFICS': 'Strip honorifics flag',
        'GLOSSARY_FUZZY_THRESHOLD': 'Fuzzy matching threshold',
        'GLOSSARY_USE_LEGACY_CSV': 'Legacy CSV format flag',
        'GLOSSARY_MAX_SENTENCES': 'Maximum sentences for glossary processing',
        
        # OpenRouter settings
        'OPENROUTER_USE_HTTP_ONLY': 'OpenRouter HTTP-only transport',
        'OPENROUTER_ACCEPT_IDENTITY': 'OpenRouter identity encoding',
        'OPENROUTER_PREFERRED_PROVIDER': 'OpenRouter preferred provider',
        
        # General application settings
        'EXTRACTION_WORKERS': 'Number of extraction worker threads',
        'ENABLE_GUI_YIELD': 'GUI yield during processing',
        'RETAIN_SOURCE_EXTENSION': 'Retain source file extension',
    }
    
    # Optional environment variables
    optional_env_vars = {
        'GLOSSARY_CUSTOM_FIELDS': 'Custom glossary fields (JSON)',
        'GLOSSARY_TRANSLATION_PROMPT': 'Glossary translation prompt',
        'GLOSSARY_FORMAT_INSTRUCTIONS': 'Glossary formatting instructions',
    }
    
    missing_critical = []
    empty_critical = []
    set_critical = []
    
    print("\n📋 CRITICAL ENVIRONMENT VARIABLES:")
    print("-" * 40)
    
    for var_name, description in critical_env_vars.items():
        value = os.environ.get(var_name)
        
        if value is None:
            missing_critical.append(var_name)
            print(f"❌ MISSING: {var_name}")
            print(f"   Description: {description}")
        elif not value.strip():
            empty_critical.append(var_name)
            print(f"⚠️  EMPTY: {var_name}")
            print(f"   Description: {description}")
        else:
            set_critical.append(var_name)
            value_preview = str(value)[:80] + ('...' if len(str(value)) > 80 else '')
            print(f"✅ {var_name}: {value_preview}")
    
    print(f"\n📋 OPTIONAL ENVIRONMENT VARIABLES:")
    print("-" * 40)
    
    for var_name, description in optional_env_vars.items():
        value = os.environ.get(var_name)
        if value is None:
            print(f"🔍 Not set: {var_name}")
        elif not value.strip():
            print(f"⚠️  Empty: {var_name}")
        else:
            value_preview = str(value)[:80] + ('...' if len(str(value)) > 80 else '')
            print(f"🔍 {var_name}: {value_preview}")
    
    # Summary
    total_critical = len(critical_env_vars)
    print(f"\n📊 SUMMARY:")
    print("-" * 20)
    print(f"Critical variables set: {len(set_critical)}/{total_critical}")
    
    if missing_critical:
        print(f"❌ Missing ({len(missing_critical)}): {', '.join(missing_critical)}")
        
    if empty_critical:
        print(f"⚠️  Empty ({len(empty_critical)}): {', '.join(empty_critical)}")
    
    if not missing_critical and not empty_critical:
        print("✅ All critical environment variables are properly set!")
        return True
    else:
        print("\n🔧 RECOMMENDATIONS:")
        print("1. Run the Glossary Manager and save settings")
        print("2. Check that all GUI variables are properly initialized")
        print("3. Call self.initialize_environment_variables() on app startup")
        print("4. Call self.debug_environment_variables() to see detailed debugging")
        return False

def test_json_env_vars():
    """Test JSON environment variables for validity."""
    print("\n🧪 JSON Environment Variable Validation:")
    print("=" * 50)
    
    json_vars = ['GLOSSARY_CUSTOM_ENTRY_TYPES', 'GLOSSARY_CUSTOM_FIELDS']
    
    for var_name in json_vars:
        value = os.environ.get(var_name)
        
        if not value:
            print(f"🔍 {var_name}: Not set")
            continue
            
        try:
            parsed_json = json.loads(value)
            print(f"✅ {var_name}: Valid JSON ({len(value)} chars)")
            print(f"   Content type: {type(parsed_json).__name__}")
            if isinstance(parsed_json, dict):
                print(f"   Keys: {list(parsed_json.keys())}")
            elif isinstance(parsed_json, list):
                print(f"   Items: {len(parsed_json)}")
        except json.JSONDecodeError as e:
            print(f"❌ {var_name}: Invalid JSON")
            print(f"   Error: {e}")
            print(f"   Value preview: {value[:100]}...")

if __name__ == "__main__":
    print("🚀 Environment Variable Debugging Tool")
    print("This tool helps debug Glossarion environment variable issues")
    
    # Check current status
    success = print_env_var_status()
    
    # Test JSON variables
    test_json_env_vars()
    
    if not success:
        print(f"\n💡 TIP: Add these methods to your TranslatorGUI class:")
        print("  • self.initialize_environment_variables() - Call on startup")
        print("  • self.debug_environment_variables() - Call for debugging")
        
    print("\n" + "=" * 60)
    print("🎯 To use the new debugging features in your app:")
    print("   1. self.initialize_environment_variables()  # On startup")
    print("   2. self.debug_environment_variables()       # For debugging")  
    print("   3. Enhanced save_glossary_settings() now has full debugging")
    print("   4. Enhanced save_config() now has full debugging")