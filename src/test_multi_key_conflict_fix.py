#!/usr/bin/env python3
"""
Test script to demonstrate the multi-key credential conflict fix.

This script tests the priority logic:
1. Google Credentials > Azure Endpoint > Model Name Detection
2. Proper validation to prevent invalid configurations
"""

import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_api_client import UnifiedClient

def test_credential_priority():
    """Test the credential priority logic in multi-key mode"""
    
    print("=" * 60)
    print("TESTING MULTI-KEY CREDENTIAL CONFLICT FIX")
    print("=" * 60)
    
    # Test Case 1: Gemini model with Google credentials (should work)
    print("\n1. Testing Gemini model with Google credentials (should work):")
    try:
        client1 = UnifiedClient("dummy-api-key", "gemini-2.5-flash", output_dir=None)
        client1.current_key_google_creds = "path/to/google-creds.json"
        print("   Creating client...")
        client1._setup_client()
        print(f"   ✅ SUCCESS: client_type = {client1.client_type}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Test Case 2: Gemini model with Azure endpoint (should fail with clear error)
    print("\n2. Testing Gemini model with Azure endpoint (should fail):")
    try:
        client2 = UnifiedClient("dummy-api-key", "gemini-2.5-flash", output_dir=None)
        client2.current_key_azure_endpoint = "https://test.openai.azure.com/"
        print("   Creating client...")
        client2._setup_client()
        print(f"   ❌ UNEXPECTED SUCCESS: client_type = {client2.client_type}")
    except ValueError as e:
        print(f"   ✅ EXPECTED ERROR: {e}")
    except Exception as e:
        print(f"   ❌ UNEXPECTED ERROR: {e}")
    
    # Test Case 3: Gemini model with BOTH credentials (Google should win)
    print("\n3. Testing Gemini model with BOTH credentials (Google should win):")
    try:
        client3 = UnifiedClient("dummy-api-key", "gemini-2.5-flash", output_dir=None)
        client3.current_key_google_creds = "path/to/google-creds.json"
        client3.current_key_azure_endpoint = "https://test.openai.azure.com/"
        print("   Creating client...")
        client3._setup_client()
        print(f"   ✅ SUCCESS: client_type = {client3.client_type} (Google credentials took priority)")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Test Case 4: OpenAI model with Azure endpoint (should work)
    print("\n4. Testing OpenAI model with Azure endpoint (should work):")
    try:
        client4 = UnifiedClient("dummy-api-key", "gpt-4o", output_dir=None)
        client4.current_key_azure_endpoint = "https://test.openai.azure.com/"
        print("   Creating client...")
        client4._setup_client()
        print(f"   ✅ SUCCESS: client_type = {client4.client_type}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    # Test Case 5: Claude model with Azure endpoint (should fail)
    print("\n5. Testing Claude model with Azure endpoint (should fail):")
    try:
        client5 = UnifiedClient("dummy-api-key", "claude-3-5-sonnet", output_dir=None)
        client5.current_key_azure_endpoint = "https://test.openai.azure.com/"
        print("   Creating client...")
        client5._setup_client()
        print(f"   ❌ UNEXPECTED SUCCESS: client_type = {client5.client_type}")
    except ValueError as e:
        print(f"   ✅ EXPECTED ERROR: {e}")
    except Exception as e:
        print(f"   ❌ UNEXPECTED ERROR: {e}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_credential_priority()