import tkinter as tk
import os
import sys
import time
import threading

class SplashScreen:
    def __init__(self, embedded_mode=False):
        self.running = True  # Initialize running flag FIRST
        self.progress_pos = 0  # Initialize progress position
        self.embedded_mode = embedded_mode  # Track if running embedded in main process
        
        self.root = tk.Tk()
        self.root.title("Loading Glossarion...")
        self.root.geometry("450x350")
        self.root.configure(bg='#2b2b2b')
        self.root.resizable(False, False)
        self.root.overrideredirect(True)
        
        # Center the splash screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - 225
        y = (self.root.winfo_screenheight() // 2) - 175
        self.root.geometry(f"450x350+{x}+{y}")
        
        self.setup_ui()
        
        # Start monitoring for status updates
        if not embedded_mode:
            self.monitor_status()
        
    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2b2b2b', relief='raised', bd=2)
        main_frame.pack(fill='both', expand=True, padx=2, pady=2)
        
        # Try to load icon
        self.load_icon(main_frame)
        
        # Title
        title_label = tk.Label(main_frame, text="Glossarion v1.6.6", 
                              bg='#2b2b2b', fg='#4a9eff', 
                              font=('Arial', 20, 'bold'))
        title_label.pack(pady=(10, 5))
        
        # Subtitle
        subtitle_label = tk.Label(main_frame, text="Advanced EPUB Translation Suite", 
                                 bg='#2b2b2b', fg='#cccccc', 
                                 font=('Arial', 12))
        subtitle_label.pack(pady=(0, 15))
        
        # Status label
        self.status_label = tk.Label(main_frame, text="Initializing...", 
                                    bg='#2b2b2b', fg='#ffffff', 
                                    font=('Arial', 11))
        self.status_label.pack(pady=(10, 10))
        
        # Progress bar
        progress_frame = tk.Frame(main_frame, bg='#2b2b2b')
        progress_frame.pack(pady=(5, 20))
        
        self.progress_canvas = tk.Canvas(progress_frame, width=300, height=20, 
                                        bg='#404040', highlightthickness=0)
        self.progress_canvas.pack()
        
        # Version info
        version_label = tk.Label(main_frame, text="Starting up...", 
                                bg='#2b2b2b', fg='#888888', 
                                font=('Arial', 9))
        version_label.pack(side='bottom', pady=(0, 15))
        
        # Start progress animation
        self.animate_progress()
    
    def load_icon(self, parent):
        """Load the application icon"""
        try:
            # Get icon path - handle both development and packaged modes
            if getattr(sys, 'frozen', False):
                # Running as .exe
                base_dir = sys._MEIPASS
            else:
                # Running as .py files
                base_dir = os.path.dirname(os.path.abspath(__file__))
            
            ico_path = os.path.join(base_dir, 'Halgakos.ico')
            
            if os.path.isfile(ico_path):
                try:
                    # Try PIL first for better quality
                    from PIL import Image, ImageTk
                    pil_image = Image.open(ico_path)
                    pil_image = pil_image.resize((128, 128), Image.Resampling.LANCZOS)
                    icon_photo = ImageTk.PhotoImage(pil_image, master=self.root)
                    icon_label = tk.Label(parent, image=icon_photo, bg='#2b2b2b')
                    icon_label.image = icon_photo  # Keep reference
                    icon_label.pack(pady=(20, 10))
                    return
                except ImportError:
                    # Fallback to basic tkinter
                    try:
                        icon_image = tk.PhotoImage(file=ico_path)
                        icon_label = tk.Label(parent, image=icon_image, bg='#2b2b2b')
                        icon_label.image = icon_image
                        icon_label.pack(pady=(20, 10))
                        return
                    except tk.TclError:
                        pass
        except Exception:
            pass
        
        # Fallback emoji
        icon_frame = tk.Frame(parent, bg='#4a9eff', width=128, height=128)
        icon_frame.pack(pady=(20, 10))
        icon_frame.pack_propagate(False)
        
        icon_label = tk.Label(icon_frame, text="📚", font=('Arial', 64), 
                             bg='#4a9eff', fg='white')
        icon_label.pack(expand=True)
    
    def animate_progress(self):
        """Animate the progress bar"""
        if self.running and hasattr(self, 'root') and self.root.winfo_exists():
            try:
                self.progress_canvas.delete("progress")
                self.progress_pos = (self.progress_pos + 12) % 320
                self.progress_canvas.create_rectangle(0, 0, self.progress_pos, 20, 
                                                    fill='#4a9eff', tags="progress")
                self.root.after(120, self.animate_progress)
            except (tk.TclError, AttributeError):
                pass  # Window was destroyed or canvas doesn't exist
    
    def monitor_status(self):
        """Monitor for status updates from main application"""
        def check_status():
            if not self.running:
                return
                
            try:
                # Check for status file
                if os.path.exists('splash_status.txt'):
                    with open('splash_status.txt', 'r', encoding='utf-8') as f:
                        status = f.read().strip()
                    
                    if status == "CLOSE":
                        self.close()
                        return
                    elif status and hasattr(self, 'status_label'):
                        self.status_label.config(text=status)
                
                # Check for close signal
                if os.path.exists('splash_close.signal'):
                    self.close()
                    return
                    
            except Exception:
                pass
            
            # Continue monitoring
            if self.running and hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(100, check_status)
        
        # Start monitoring
        self.root.after(100, check_status)
    
    def update_status(self, message):
        """Update status label (for embedded mode)"""
        if hasattr(self, 'status_label'):
            try:
                self.status_label.config(text=message)
            except:
                pass
    
    def close(self):
        """Close the splash screen"""
        self.running = False
        try:
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.quit()
                self.root.destroy()
        except (tk.TclError, AttributeError):
            pass
    
    def run(self):
        """Run the splash screen"""
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"Splash screen error: {e}")
        finally:
            # Clean up status files (only in subprocess mode)
            if not self.embedded_mode:
                try:
                    for file in ['splash_status.txt', 'splash_close.signal']:
                        if os.path.exists(file):
                            os.remove(file)
                except:
                    pass

# Prevent execution when imported
if __name__ == "__main__":
    # Only run if this file is executed directly, not when imported
    # And only if not running as part of a packaged executable
    if not getattr(sys, 'frozen', False):
        try:
            splash = SplashScreen(embedded_mode=False)
            splash.run()
        except Exception as e:
            print(f"Failed to start splash screen: {e}")
            sys.exit(1)
    else:
        # If running as .exe, this should not be executed directly
        print("Splash screen should not be run directly from packaged executable")
        sys.exit(0)
