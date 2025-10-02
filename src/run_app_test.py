#!/usr/bin/env python3
"""Run the actual app and monitor what happens"""

import subprocess
import time

# First, let's see what the current config has
print("Current config before starting app:")
with open('config_web.json', 'r', encoding='utf-8') as f:
    import json
    config = json.load(f)
    print(f"  Model: {config.get('model')}")
    print(f"  Batch size: {config.get('batch_size')}")

print("\nStarting actual app.py...")
print("Please manually test changing settings in the browser at http://localhost:7860")
print("Then close the browser and press Ctrl+C here\n")

# Run the actual app
try:
    subprocess.run(['python', 'app.py'], check=True)
except KeyboardInterrupt:
    print("\n\nApp stopped. Checking config after changes...")
    
    with open('config_web.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
        print(f"  Model: {config.get('model')}")
        print(f"  Batch size: {config.get('batch_size')}")