#!/usr/bin/env python3
"""
Glossarion Web Launcher
Starts the Gradio app and automatically opens it in Chrome
"""

import os
import sys
import time
import webbrowser
import threading
import socket
import subprocess
from pathlib import Path

def find_free_port(start_port=7860):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except:
            continue
    return start_port

def open_browser(url, delay=3):
    """Open browser with improved error handling"""
    time.sleep(delay)
    
    # Try Chrome with subprocess to avoid shell conflicts
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe"),
    ]
    
    for chrome_path in chrome_paths:
        if os.path.exists(chrome_path):
            try:
                # Use subprocess to avoid shell conflicts
                subprocess.Popen(
                    [chrome_path, url],
                    shell=False,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                )
                print(f"✅ Opened in Google Chrome")
                return
            except Exception as e:
                print(f"⚠️ Chrome launch failed: {e}")
                continue
    
    # Fallback to default browser
    try:
        webbrowser.open(url)
        print("✅ Opened in default browser")
    except Exception as e:
        print(f"⚠️ Browser launch failed: {e}")
        print(f"Please open manually: {url}")

def main():
    """Main launcher with improved error handling"""
    print("""
╔══════════════════════════════════════════╗
║         GLOSSARION WEB LAUNCHER          ║
║       AI-Powered Translation Tool        ║
╚══════════════════════════════════════════╝
    """)
    
    try:
        # Set environment variables
        os.environ['GLOSSARION_EXE_MODE'] = '1'
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # Add src directory to path
        src_dir = Path(__file__).parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        
        # Find free port
        port = find_free_port(7860)
        url = f"http://127.0.0.1:{port}"
        
        print(f"🌐 Starting server on port {port}...")
        print(f"🔗 URL: {url}")
        
        # Import with better error handling
        print("📦 Loading application modules...")
        try:
            from app import GlossarionWeb
            print("✅ Core modules loaded successfully")
        except ImportError as e:
            print(f"❌ Failed to import core modules: {e}")
            print("\nMake sure all required packages are installed:")
            print("  pip install gradio")
            print("\nPress Enter to exit...")
            input()
            return
        
        # Create web interface
        print("🚀 Initializing web interface...")
        try:
            web_app = GlossarionWeb()
            interface = web_app.create_interface()
            print("✅ Interface created successfully")
        except Exception as e:
            print(f"❌ Failed to create interface: {e}")
            print("\nPress Enter to exit...")
            input()
            return
        
        # Start browser in background
        browser_thread = threading.Thread(
            target=open_browser,
            args=(url,),
            daemon=True
        )
        browser_thread.start()
        
        print(f"✨ Launching at {url}")
        print("\n" + "="*50)
        print("Press Ctrl+C to stop the server")
        print("Close this window to exit")
        print("="*50 + "\n")
        
        # Launch with error handling
        interface.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            quiet=True,  # Reduce console noise
            show_error=True,
            prevent_thread_lock=False,
            inbrowser=False
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Glossarion Web...")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("\nPress Enter to exit...")
        input()
    finally:
        sys.exit(0)
if __name__ == "__main__":
    main()