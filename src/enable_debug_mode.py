#!/usr/bin/env python3
"""
Debug Mode Enabler for Glossarion
This script enables debug buttons and features for developers/troubleshooting.
"""

import json
import os
import sys

CONFIG_FILE = 'config.json'

def enable_debug_mode():
    """Enable debug mode by adding show_debug_buttons to config"""
    
    if not os.path.exists(CONFIG_FILE):
        print("❌ config.json not found. Please run Glossarion at least once to create the config file.")
        return False
    
    try:
        # Load existing config
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"📖 Loaded config file: {CONFIG_FILE}")
        
        # Check if already enabled
        if config.get('show_debug_buttons', False):
            print("✅ Debug mode is already enabled!")
            return True
        
        # Enable debug mode
        config['show_debug_buttons'] = True
        
        # Create backup
        backup_file = f"{CONFIG_FILE}.backup"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"💾 Created backup: {backup_file}")
        
        # Save updated config
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print("✅ Debug mode enabled successfully!")
        print("🔧 Debug features now available:")
        print("   • Debug Mode toggle button in Other Settings (GUI)")
        print("   • Enhanced debugging in all save functions")
        print("   • Comprehensive environment variable checking")
        print("💡 You can also toggle debug mode directly in the GUI: Other Settings → Debug Mode toggle")
        
        return True
        
    except Exception as e:
        print(f"❌ Error enabling debug mode: {e}")
        return False

def disable_debug_mode():
    """Disable debug mode by removing show_debug_buttons from config"""
    
    if not os.path.exists(CONFIG_FILE):
        print("❌ config.json not found.")
        return False
    
    try:
        # Load existing config
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"📖 Loaded config file: {CONFIG_FILE}")
        
        # Check if already disabled
        if not config.get('show_debug_buttons', False):
            print("✅ Debug mode is already disabled!")
            return True
        
        # Disable debug mode
        config['show_debug_buttons'] = False
        
        # Save updated config
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print("✅ Debug mode disabled successfully!")
        print("🔒 Debug buttons will be hidden from users")
        
        return True
        
    except Exception as e:
        print(f"❌ Error disabling debug mode: {e}")
        return False

def check_debug_status():
    """Check current debug mode status"""
    
    if not os.path.exists(CONFIG_FILE):
        print("❌ config.json not found.")
        return
    
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        debug_enabled = config.get('show_debug_buttons', False)
        
        print("🔍 Current Debug Mode Status:")
        print(f"   show_debug_buttons: {debug_enabled}")
        
        if debug_enabled:
            print("✅ Debug mode is ENABLED")
            print("   • Debug buttons will appear in Other Settings")
            print("   • Enhanced debugging is active in save functions")
        else:
            print("🔒 Debug mode is DISABLED")
            print("   • Debug buttons are hidden from users")
            print("   • Standard logging only")
            
    except Exception as e:
        print(f"❌ Error checking debug status: {e}")

if __name__ == "__main__":
    print("🚀 Glossarion Debug Mode Manager")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
        
        if action in ['enable', 'on', 'true']:
            enable_debug_mode()
        elif action in ['disable', 'off', 'false']:
            disable_debug_mode()
        elif action in ['status', 'check']:
            check_debug_status()
        else:
            print(f"❌ Unknown action: {action}")
            print("Valid actions: enable, disable, status")
    else:
        # Interactive mode
        print("Choose an action:")
        print("1. Enable debug mode")
        print("2. Disable debug mode") 
        print("3. Check current status")
        print("4. Exit")
        
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                enable_debug_mode()
            elif choice == '2':
                disable_debug_mode()
            elif choice == '3':
                check_debug_status()
            elif choice == '4':
                print("👋 Goodbye!")
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 40)
    print("💡 Usage:")
    print("  python enable_debug_mode.py enable   # Enable debug mode")
    print("  python enable_debug_mode.py disable  # Disable debug mode")
    print("  python enable_debug_mode.py status   # Check status")
    print("  python enable_debug_mode.py          # Interactive mode")
