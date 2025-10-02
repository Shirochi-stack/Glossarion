#!/usr/bin/env python3
"""Test that values actually persist when changed through the UI handlers"""

import json
import time
from app import GlossarionWeb

def test_real_persistence():
    print("Testing REAL persistence with UI handlers...")
    
    # 1. Create instance and check initial values
    print("\n1️⃣ Initial load:")
    app1 = GlossarionWeb()
    initial_model = app1.get_config_value('model')
    initial_api_key = app1.get_config_value('api_key')
    initial_batch_size = app1.get_config_value('batch_size')
    initial_contextual = app1.get_config_value('contextual')
    
    print(f"  Model: {initial_model}")
    print(f"  API Key: {initial_api_key[:10] if initial_api_key else 'None'}...")
    print(f"  Batch Size: {initial_batch_size}")
    print(f"  Contextual: {initial_contextual}")
    
    # 2. Change values using the actual save functions from the UI
    print("\n2️⃣ Changing values using UI save functions...")
    
    # Simulate manga credentials save (like the UI does)
    def save_manga_credentials(model, api_key):
        try:
            current_config = app1.load_config()
            current_config['model'] = model
            if api_key:
                current_config['api_key'] = api_key
            app1.save_config(current_config)
            return "Saved"
        except Exception as e:
            return f"Error: {e}"
    
    # Change model and API key
    new_model = f"test-model-{int(time.time())}"
    new_api_key = "sk-test-key-12345"
    result = save_manga_credentials(new_model, new_api_key)
    print(f"  Save manga credentials: {result}")
    
    # Simulate settings tab save
    def save_settings_tab():
        try:
            current_config = app1.load_config()
            current_config['batch_size'] = 25
            current_config['contextual'] = True
            current_config['token_limit'] = 150000
            current_config['temperature'] = 0.7
            app1.save_config(current_config)
            return "Settings saved"
        except Exception as e:
            return f"Error: {e}"
    
    result = save_settings_tab()
    print(f"  Save settings tab: {result}")
    
    # 3. Create a NEW instance (like restarting the app)
    print("\n3️⃣ Creating NEW instance (simulating app restart)...")
    app2 = GlossarionWeb()
    
    # 4. Check if the new values persisted
    print("\n4️⃣ Checking if values persisted:")
    loaded_model = app2.get_config_value('model')
    loaded_api_key = app2.get_config_value('api_key')
    loaded_batch_size = app2.get_config_value('batch_size')
    loaded_contextual = app2.get_config_value('contextual')
    loaded_token_limit = app2.get_config_value('token_limit')
    loaded_temperature = app2.get_config_value('temperature')
    
    print(f"  Model: {loaded_model}")
    print(f"  API Key: {loaded_api_key[:10] if loaded_api_key else 'None'}...")
    print(f"  Batch Size: {loaded_batch_size}")
    print(f"  Contextual: {loaded_contextual}")
    print(f"  Token Limit: {loaded_token_limit}")
    print(f"  Temperature: {loaded_temperature}")
    
    # 5. Verify the values actually changed
    print("\n5️⃣ Verification:")
    success = True
    
    if loaded_model != new_model:
        print(f"  ❌ Model didn't persist! Expected: {new_model}, Got: {loaded_model}")
        success = False
    else:
        print(f"  ✅ Model persisted correctly: {loaded_model}")
    
    if loaded_api_key != new_api_key:
        print(f"  ❌ API Key didn't persist! Expected: {new_api_key}, Got: {loaded_api_key}")
        success = False
    else:
        print(f"  ✅ API Key persisted correctly")
    
    if loaded_batch_size != 25:
        print(f"  ❌ Batch size didn't persist! Expected: 25, Got: {loaded_batch_size}")
        success = False
    else:
        print(f"  ✅ Batch size persisted correctly: {loaded_batch_size}")
    
    if loaded_contextual != True:
        print(f"  ❌ Contextual didn't persist! Expected: True, Got: {loaded_contextual}")
        success = False
    else:
        print(f"  ✅ Contextual persisted correctly: {loaded_contextual}")
    
    if loaded_token_limit != 150000:
        print(f"  ❌ Token limit didn't persist! Expected: 150000, Got: {loaded_token_limit}")
        success = False
    else:
        print(f"  ✅ Token limit persisted correctly: {loaded_token_limit}")
    
    # 6. Check the raw file to see what's actually saved
    print("\n6️⃣ Checking raw config file:")
    with open('config_web.json', 'r', encoding='utf-8') as f:
        file_config = json.load(f)
        print(f"  Model in file: {file_config.get('model')}")
        print(f"  API key in file: {file_config.get('api_key')[:30]}..." if file_config.get('api_key') else "  API key in file: None")
        print(f"  Batch size in file: {file_config.get('batch_size')}")
        print(f"  Contextual in file: {file_config.get('contextual')}")
    
    # 7. Final result
    print("\n" + "="*50)
    if success:
        print("✅ ALL TESTS PASSED! Persistence is working correctly!")
    else:
        print("❌ SOME TESTS FAILED! Persistence has issues!")
    print("="*50)

if __name__ == "__main__":
    test_real_persistence()