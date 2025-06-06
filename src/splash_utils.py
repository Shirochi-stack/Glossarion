import time
import atexit

class SplashManager:
    """Simple splash screen manager that works with main thread"""
    
    def __init__(self):
        self.splash_window = None
        self._status_text = "Initializing..."
        
    def start_splash(self):
        """Create splash window on main thread"""
        try:
            import tkinter as tk
            
            print("🎨 Starting splash screen...")
            
            # Create splash window on main thread
            self.splash_window = tk.Tk()
            self.splash_window.title("Loading Glossarion...")
            self.splash_window.geometry("450x350")
            self.splash_window.configure(bg='#2b2b2b')
            self.splash_window.resizable(False, False)
            self.splash_window.overrideredirect(True)
            
            # Center the window
            self.splash_window.update_idletasks()
            x = (self.splash_window.winfo_screenwidth() // 2) - 225
            y = (self.splash_window.winfo_screenheight() // 2) - 175
            self.splash_window.geometry(f"450x350+{x}+{y}")
            
            # Add content
            main_frame = tk.Frame(self.splash_window, bg='#2b2b2b', relief='raised', bd=2)
            main_frame.pack(fill='both', expand=True, padx=2, pady=2)
            
            # Load the actual Halgakos.ico icon
            self._load_icon(main_frame)
            
            # Title
            title_label = tk.Label(main_frame, text="Glossarion v1.6.6", 
                                  bg='#2b2b2b', fg='#4a9eff', font=('Arial', 20, 'bold'))
            title_label.pack(pady=(10, 5))
            
            # Subtitle
            subtitle_label = tk.Label(main_frame, text="Advanced EPUB Translation Suite", 
                                     bg='#2b2b2b', fg='#cccccc', font=('Arial', 12))
            subtitle_label.pack(pady=(0, 15))
            
            # Status
            self.status_label = tk.Label(main_frame, text=self._status_text, 
                                        bg='#2b2b2b', fg='#ffffff', font=('Arial', 11))
            self.status_label.pack(pady=(10, 10))
            
            # Progress bar
            progress_frame = tk.Frame(main_frame, bg='#2b2b2b')
            progress_frame.pack(pady=(5, 20))
            self.progress_canvas = tk.Canvas(progress_frame, width=300, height=20, 
                                           bg='#404040', highlightthickness=0)
            self.progress_canvas.pack()
            
            # Version info
            version_label = tk.Label(main_frame, text="Starting up...", 
                                   bg='#2b2b2b', fg='#888888', font=('Arial', 9))
            version_label.pack(side='bottom', pady=(0, 15))
            
            # Start progress animation
            self.progress_pos = 0
            self._animate_progress()
            
            # Update the display
            self.splash_window.update()
            
            # Register cleanup
            atexit.register(self.close_splash)
            return True
            
        except Exception as e:
            print(f"⚠️ Could not start splash: {e}")
            return False
    
    def _load_icon(self, parent):
        """Load the Halgakos.ico icon"""
        try:
            # Get icon path - handle both development and packaged modes
            import os
            import sys
            import tkinter as tk
            
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
                    icon_photo = ImageTk.PhotoImage(pil_image, master=self.splash_window)
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
        
        # Fallback emoji if icon loading fails
        import tkinter as tk
        icon_frame = tk.Frame(parent, bg='#4a9eff', width=128, height=128)
        icon_frame.pack(pady=(20, 10))
        icon_frame.pack_propagate(False)
        
        icon_label = tk.Label(icon_frame, text="📚", font=('Arial', 64), 
                             bg='#4a9eff', fg='white')
        icon_label.pack(expand=True)

    def _animate_progress(self):
        """Animate progress bar"""
        if self.splash_window and self.splash_window.winfo_exists():
            try:
                self.progress_canvas.delete("progress")
                self.progress_pos = (self.progress_pos + 12) % 320
                self.progress_canvas.create_rectangle(0, 0, self.progress_pos, 20, fill='#4a9eff', tags="progress")
                self.splash_window.after(120, self._animate_progress)
            except:
                pass
    
    def update_status(self, message):
        """Update splash status"""
        self._status_text = message
        try:
            if self.splash_window and hasattr(self, 'status_label'):
                self.status_label.config(text=message)
                self.splash_window.update()
        except:
            pass
    
    def close_splash(self):
        """Close the splash screen"""
        try:
            if self.splash_window and self.splash_window.winfo_exists():
                self.splash_window.destroy()
                self.splash_window = None
        except:
            pass
