#!/usr/bin/env python3
"""Test config saving and loading"""

import json
import os
from app import GlossarionWeb

def test_config():
    print("Testing config save/load...")
    
    # Create instance
    app = GlossarionWeb()
    
    # Show current config
    print(f"\nCurrent model: {app.get_config_value('model')}")
    print(f"Current API key (encrypted): {app.config.get('api_key', 'None')}")
    print(f"Current API key (decrypted): {app.get_config_value('api_key', 'None')}")
    
    # Test saving a new value
    print("\nTesting save...")
    test_config = app.load_config()
    test_config['model'] = 'test-model-123'
    test_config['test_field'] = 'test_value'
    result = app.save_config(test_config)
    print(f"Save result: {result}")
    
    # Reload and check
    print("\nReloading config...")
    new_config = app.load_config()
    print(f"New model value: {new_config.get('model')}")
    print(f"Test field value: {new_config.get('test_field')}")
    
    # Check file directly
    print("\nChecking config file directly...")
    with open('config_web.json', 'r', encoding='utf-8') as f:
        file_config = json.load(f)
        print(f"Model in file: {file_config.get('model')}")
        print(f"Test field in file: {file_config.get('test_field')}")
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    test_config()