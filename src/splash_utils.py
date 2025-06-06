import subprocess
import os
import sys
import atexit

class SplashManager:
    """Manages the separate splash screen process"""
    
    def __init__(self):
        self.splash_process = None
        self.splash_script = None
        
    def start_splash(self):
        """Start the splash screen in a separate process"""
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.splash_script = os.path.join(script_dir, 'splash_screen.py')
            
            # Start splash process
            if os.path.exists(self.splash_script):
                self.splash_process = subprocess.Popen([
                    sys.executable, self.splash_script
                ], cwd=script_dir)
                
                # Register cleanup
                atexit.register(self.close_splash)
                
                print("🎨 Splash screen started")
                return True
            else:
                print("⚠️ Splash script not found, continuing without splash")
                return False
                
        except Exception as e:
            print(f"⚠️ Could not start splash: {e}")
            return False
    
    def update_status(self, message):
        """Update the splash screen status"""
        try:
            with open('splash_status.txt', 'w', encoding='utf-8') as f:
                f.write(message)
        except Exception:
            pass
    
    def close_splash(self):
        """Close the splash screen"""
        try:
            # Signal splash to close
            with open('splash_close.signal', 'w') as f:
                f.write('close')
            
            # Wait for process to finish
            if self.splash_process:
                self.splash_process.wait(timeout=2)
                
        except Exception:
            # Force kill if needed
            if self.splash_process:
                try:
                    self.splash_process.terminate()
                except:
                    pass
        finally:
            # Clean up files
            try:
                if os.path.exists('splash_status.txt'):
                    os.remove('splash_status.txt')
                if os.path.exists('splash_close.signal'):
                    os.remove('splash_close.signal')
            except:
                pass


