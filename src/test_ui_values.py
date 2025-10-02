#!/usr/bin/env python3
"""Test that UI values are loaded correctly from decrypted config"""

import json
from app import GlossarionWeb

def test_ui_values():
    print("Testing UI value loading...")
    
    # Create instance
    app = GlossarionWeb()
    
    # Check that values are properly loaded from decrypted config
    print(f"\nModel: {app.get_config_value('model', 'default')}")
    print(f"API Key (decrypted): {app.get_config_value('api_key', 'None')[:10]}..." if app.get_config_value('api_key') else "API Key: None")
    print(f"OCR Provider: {app.get_config_value('ocr_provider', 'custom-api')}")
    print(f"Bubble Detection: {app.get_config_value('bubble_detection_enabled', True)}")
    print(f"Batch Size: {app.get_config_value('batch_size', 10)}")
    
    # Test setting a new value and saving
    print("\n📝 Testing save with new value...")
    new_config = app.load_config()
    new_config['model'] = 'gpt-4-turbo'
    new_config['batch_size'] = 5
    app.save_config(new_config)
    
    # Reload and check
    print("\n🔄 Reloading...")
    app2 = GlossarionWeb()
    print(f"Model after reload: {app2.get_config_value('model')}")
    print(f"Batch size after reload: {app2.get_config_value('batch_size')}")
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    test_ui_values()