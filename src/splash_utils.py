import subprocess
import os
import sys
import atexit
import threading
import time

class SplashManager:
    """Manages the splash screen for both development and packaged executable"""
    
    def __init__(self):
        self.splash_process = None
        self.splash_thread = None
        self._splash_active = False
        
    def start_splash(self):
        """Start the splash screen - works for both .py and .exe versions"""
        try:
            # Check if we're running as a packaged executable
            if getattr(sys, 'frozen', False):
                # Running as .exe - use embedded splash screen
                print("🎨 Starting embedded splash screen...")
                self._start_embedded_splash()
            else:
                # Running as .py files - use subprocess
                print("🎨 Starting splash screen subprocess...")
                self._start_subprocess_splash()
                
            # Register cleanup
            atexit.register(self.close_splash)
            return True
                
        except Exception as e:
            print(f"⚠️ Could not start splash: {e}")
            return False
    
    def _start_embedded_splash(self):
        """Start splash screen in the same process (for .exe)"""
        def run_splash():
            try:
                # Import here to avoid circular imports
                from splash_screen import SplashScreen
                self._splash_active = True
                
                splash = SplashScreen()
                
                # Run splash in separate thread
                def splash_loop():
                    try:
                        # Start the splash screen
                        splash.root.after(0, splash.root.mainloop)
                    except Exception as e:
                        print(f"Splash screen error: {e}")
                    finally:
                        self._splash_active = False
                
                # Start splash loop in thread
                splash_thread = threading.Thread(target=splash_loop, daemon=True)
                splash_thread.start()
                
                # Keep reference to splash for cleanup
                self.embedded_splash = splash
                
            except Exception as e:
                print(f"⚠️ Embedded splash failed: {e}")
                self._splash_active = False
        
        # Start splash in background thread
        self.splash_thread = threading.Thread(target=run_splash, daemon=True)
        self.splash_thread.start()
    
    def _start_subprocess_splash(self):
        """Start splash screen as subprocess (for .py development)"""
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            splash_script = os.path.join(script_dir, 'splash_screen.py')
            
            # Start splash process with proper python executable
            if os.path.exists(splash_script):
                # Use python.exe explicitly for development
                python_exe = sys.executable
                if 'python' not in os.path.basename(python_exe).lower():
                    # If sys.executable doesn't look like python, try to find it
                    import shutil
                    python_exe = shutil.which('python') or shutil.which('python3') or sys.executable
                
                self.splash_process = subprocess.Popen([
                    python_exe, splash_script
                ], cwd=script_dir, 
                   creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
                
                print("🎨 Splash screen subprocess started")
            else:
                print("⚠️ Splash script not found, continuing without splash")
                
        except Exception as e:
            print(f"⚠️ Subprocess splash failed: {e}")
    
    def update_status(self, message):
        """Update the splash screen status"""
        try:
            # For subprocess version
            with open('splash_status.txt', 'w', encoding='utf-8') as f:
                f.write(message)
        except Exception:
            pass
        
        # For embedded version
        if hasattr(self, 'embedded_splash'):
            try:
                def update_ui():
                    if hasattr(self.embedded_splash, 'status_label'):
                        self.embedded_splash.status_label.config(text=message)
                
                if hasattr(self.embedded_splash, 'root'):
                    self.embedded_splash.root.after(0, update_ui)
            except Exception:
                pass
    
    def close_splash(self):
        """Close the splash screen"""
        try:
            # Signal splash to close (for subprocess version)
            with open('splash_close.signal', 'w') as f:
                f.write('close')
            
            # Close subprocess if running
            if self.splash_process:
                try:
                    self.splash_process.terminate()
                    self.splash_process.wait(timeout=2)
                except:
                    try:
                        self.splash_process.kill()
                    except:
                        pass
                self.splash_process = None
            
            # Close embedded splash if running
            if hasattr(self, 'embedded_splash'):
                try:
                    if hasattr(self.embedded_splash, 'close'):
                        self.embedded_splash.close()
                    elif hasattr(self.embedded_splash, 'root'):
                        self.embedded_splash.root.quit()
                        self.embedded_splash.root.destroy()
                except Exception as e:
                    print(f"Error closing embedded splash: {e}")
            
            self._splash_active = False
            
        except Exception as e:
            print(f"Error during splash cleanup: {e}")
        finally:
            # Clean up files
            try:
                for file in ['splash_status.txt', 'splash_close.signal']:
                    if os.path.exists(file):
                        os.remove(file)
            except:
                pass
    
    def is_active(self):
        """Check if splash screen is still active"""
        if getattr(sys, 'frozen', False):
            return self._splash_active
        else:
            return self.splash_process is not None and self.splash_process.poll() is None
