#!/usr/bin/env python3
"""Test that the app starts correctly"""

from app import GlossarionWeb

app = GlossarionWeb()
print("✅ App initialized successfully")
print(f"Current model: {app.get_config_value('model')}")
print(f"Batch size: {app.get_config_value('batch_size')}")
print(f"API key (first 10 chars): {app.get_config_value('api_key')[:10]}..." if app.get_config_value('api_key') else "No API key")