#!/usr/bin/env python3
"""Test with actual UI handlers like they are called in Gradio"""

import json
import time
from app import GlossarionWeb

def test_actual_ui():
    print("Testing with ACTUAL UI handler simulation...")
    
    # 1. Start the app
    print("\n1️⃣ Starting app...")
    web_app = GlossarionWeb()
    
    print(f"Initial model: {web_app.get_config_value('model')}")
    print(f"Initial batch_size: {web_app.get_config_value('batch_size')}")
    
    # 2. Simulate the ACTUAL manga model save handler from line ~2092
    print("\n2️⃣ Simulating manga_model.change handler...")
    def save_manga_credentials(model, api_key):
        """This is the ACTUAL function from the UI"""
        try:
            current_config = web_app.load_config()
            # Don't decrypt - just update the fields we need
            current_config['model'] = model
            if api_key:  # Only save if not empty
                current_config['api_key'] = api_key
            web_app.save_config(current_config)
            return None  # No output needed
        except Exception as e:
            print(f"Failed to save manga credentials: {e}")
            return None
    
    # Call it like the UI would
    new_model = "claude-3-opus-20240229"
    new_api_key = "sk-ant-test-key-456"
    save_manga_credentials(new_model, new_api_key)
    
    # 3. Simulate the ACTUAL settings tab save handler from line ~3326
    print("\n3️⃣ Simulating Settings tab save...")
    def save_settings_tab(thread_delay_val, api_delay_val, chapter_range_val, token_limit_val, 
                          disable_token_limit_val, output_token_limit_val, contextual_val, 
                          history_limit_val, rolling_history_val, batch_translation_val, 
                          batch_size_val, save_api_key_val):
        """This is the ACTUAL function from Settings tab"""
        try:
            current_config = web_app.load_config()
            # Don't decrypt - just update non-encrypted fields
            
            # Update settings
            current_config['thread_submission_delay'] = float(thread_delay_val)
            current_config['delay'] = float(api_delay_val)
            current_config['chapter_range'] = str(chapter_range_val)
            current_config['token_limit'] = int(token_limit_val)
            current_config['token_limit_disabled'] = bool(disable_token_limit_val)
            current_config['max_output_tokens'] = int(output_token_limit_val)
            current_config['contextual'] = bool(contextual_val)
            current_config['translation_history_limit'] = int(history_limit_val)
            current_config['translation_history_rolling'] = bool(rolling_history_val)
            current_config['batch_translation'] = bool(batch_translation_val)
            current_config['batch_size'] = int(batch_size_val)
            
            # Save to file
            web_app.save_config(current_config)
            
            return "✅ Settings saved successfully", ""
        except Exception as e:
            return f"❌ Failed to save: {str(e)}", ""
    
    # Call with test values
    result = save_settings_tab(
        0.2,    # thread_delay
        1.5,    # api_delay  
        "1-10", # chapter_range
        175000, # token_limit
        False,  # disable_token_limit
        20000,  # output_token_limit
        True,   # contextual
        3,      # history_limit
        True,   # rolling_history
        False,  # batch_translation
        15,     # batch_size
        True    # save_api_key
    )
    print(f"Settings save result: {result[0]}")
    
    # 4. Check what's in memory RIGHT NOW without reloading
    print("\n4️⃣ Checking in-memory values (without restart):")
    print(f"  Model in memory: {web_app.get_config_value('model')}")
    print(f"  Batch size in memory: {web_app.get_config_value('batch_size')}")
    print(f"  Token limit in memory: {web_app.get_config_value('token_limit')}")
    
    # 5. Check what's in the FILE
    print("\n5️⃣ Checking what's actually in the file:")
    with open('config_web.json', 'r', encoding='utf-8') as f:
        file_config = json.load(f)
        print(f"  Model in file: {file_config.get('model')}")
        print(f"  Batch size in file: {file_config.get('batch_size')}")
        print(f"  Token limit in file: {file_config.get('token_limit')}")
        print(f"  API key in file (encrypted?): {file_config.get('api_key', 'None')[:30]}...")
    
    # 6. Create NEW instance to test persistence
    print("\n6️⃣ Creating NEW app instance (like restarting)...")
    web_app2 = GlossarionWeb()
    
    print(f"  Loaded model: {web_app2.get_config_value('model')}")
    print(f"  Loaded batch size: {web_app2.get_config_value('batch_size')}")
    print(f"  Loaded token limit: {web_app2.get_config_value('token_limit')}")
    print(f"  Loaded contextual: {web_app2.get_config_value('contextual')}")
    print(f"  Loaded API key: {web_app2.get_config_value('api_key')[:20]}..." if web_app2.get_config_value('api_key') else "None")
    
    # 7. Verify
    print("\n7️⃣ Verification:")
    if web_app2.get_config_value('model') == new_model:
        print(f"  ✅ Model persisted: {new_model}")
    else:
        print(f"  ❌ Model NOT persisted! Expected: {new_model}, Got: {web_app2.get_config_value('model')}")
    
    if web_app2.get_config_value('batch_size') == 15:
        print(f"  ✅ Batch size persisted: 15")
    else:
        print(f"  ❌ Batch size NOT persisted! Expected: 15, Got: {web_app2.get_config_value('batch_size')}")
    
    if web_app2.get_config_value('token_limit') == 175000:
        print(f"  ✅ Token limit persisted: 175000")
    else:
        print(f"  ❌ Token limit NOT persisted! Expected: 175000, Got: {web_app2.get_config_value('token_limit')}")

if __name__ == "__main__":
    test_actual_ui()