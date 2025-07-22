import time
import atexit

class SplashManager:
    """Simple splash screen manager that works with main thread"""
    
    def __init__(self):
        self.splash_window = None
        self._status_text = "Initializing..."
        self.progress_value = 0  # Track actual progress 0-100
        self.canvas_width = 320  # Progress bar dimensions (increased from 300)
        self.canvas_height = 36  # Increased from 30
        self._after_id = None
        
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
            title_label = tk.Label(main_frame, text="Glossarion v3.5.2", 
                                  bg='#2b2b2b', fg='#4a9eff', font=('Arial', 20, 'bold'))
            title_label.pack(pady=(10, 5))
            
            # Subtitle
            subtitle_label = tk.Label(main_frame, text="Advanced AI Translation Suite", 
                                     bg='#2b2b2b', fg='#cccccc', font=('Arial', 12))
            subtitle_label.pack(pady=(0, 15))
            
            # Status
            self.status_label = tk.Label(main_frame, text=self._status_text, 
                                        bg='#2b2b2b', fg='#ffffff', font=('Arial', 11))
            self.status_label.pack(pady=(10, 10))
            
            # Progress bar container
            progress_frame = tk.Frame(main_frame, bg='#2b2b2b')
            progress_frame.pack(pady=(5, 15))  # Adjusted padding for larger bar
            
            # Progress bar background
            self.progress_bg = tk.Canvas(progress_frame, width=self.canvas_width, height=self.canvas_height, 
                                        bg='#2b2b2b', highlightthickness=0)
            self.progress_bg.pack()
            
            # Create border
            self.progress_bg.create_rectangle(1, 1, self.canvas_width-1, self.canvas_height-1, 
                                            outline='#666666', width=2)
            
            # Create background
            self.progress_bg.create_rectangle(3, 3, self.canvas_width-3, self.canvas_height-3, 
                                            fill='#1a1a1a', outline='')
            
            # Progress bar fill (will be updated)
            self.progress_fill = None
            
            # Progress percentage text - moved up and with better font
            text_x = self.canvas_width // 2  # 160 for 320px width
            text_y = 13.5  # Positioned slightly above center for visual balance
            
            # Use a cleaner, more modern font
            progress_font = ('Montserrat', 12, 'bold')  # Increased size to 12
            
            # Create outline for better readability
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        self.progress_bg.create_text(text_x + dx, text_y + dy, text="0%", 
                                                   fill='#000000', font=progress_font,
                                                   tags="outline", anchor='center')
            
            # Main text on top (white)
            self.progress_text = self.progress_bg.create_text(text_x, text_y, text="0%", 
                                                             fill='#ffffff', font=progress_font,
                                                             anchor='center')
            
            # Version info
            version_label = tk.Label(main_frame, text="Starting up...", 
                                   bg='#2b2b2b', fg='#888888', font=('Arial', 9))
            version_label.pack(side='bottom', pady=(0, 15))
            
            # Start progress animation
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
        """Animate progress bar filling up"""
        # Cancel any existing after callback first
        if self._after_id:
            try:
                self.splash_window.after_cancel(self._after_id)
            except:
                pass
            self._after_id = None
            
        if self.splash_window and self.splash_window.winfo_exists():
            try:
                # Auto-increment progress for visual effect during startup
                if self.progress_value < 100:
                    # Increment at different rates for different phases
                    if self.progress_value < 30:
                        self.progress_value += 8  # Fast initial progress
                    elif self.progress_value < 70:
                        self.progress_value += 4  # Medium progress
                    elif self.progress_value < 90:
                        self.progress_value += 2  # Slow progress
                    else:
                        self.progress_value += 1  # Very slow final progress
                    
                    # Cap at 99% until explicitly set to 100%
                    if self.progress_value >= 99:
                        self.progress_value = 99
                
                # Update progress bar fill
                if self.progress_fill:
                    self.progress_bg.delete(self.progress_fill)
                # Also delete old highlight
                self.progress_bg.delete("highlight")
                
                # Calculate fill width (3 to canvas_width-3)
                fill_width = int((self.progress_value / 100) * (self.canvas_width - 6))  # -6 for borders
                if fill_width > 0:
                    # Create gradient effect
                    self.progress_fill = self.progress_bg.create_rectangle(
                        3, 3, 3 + fill_width, self.canvas_height - 3, 
                        fill='#4a9eff', outline=''
                    )
                    
                    # Add a highlight effect (adjusted for new height)
                    if fill_width > 10:
                        self.progress_bg.create_rectangle(
                            3, 3, min(13, 3 + fill_width), 12,
                            fill='#6bb6ff', outline='', tags="highlight"
                        )
                
                # Update percentage text without changing position
                percent_text = f"{self.progress_value}%"
                
                # Update main text
                self.progress_bg.itemconfig(self.progress_text, text=percent_text)
                
                # Update all outline layers
                for item in self.progress_bg.find_withtag("outline"):
                    self.progress_bg.itemconfig(item, text=percent_text)
                
                # Ensure text stays on top of progress fill
                self.progress_bg.tag_raise("outline")
                self.progress_bg.tag_raise(self.progress_text)

                # Store the after ID so we can cancel it later
                self._after_id = self.splash_window.after(100, self._animate_progress)
                
            except Exception:
                self._after_id = None
                pass
    
    def update_status(self, message):
            """Update splash status and progress with enhanced module loading support"""
            self._status_text = message
            try:
                if self.splash_window and hasattr(self, 'status_label'):
                    self.status_label.config(text=message)
                    
                    # Enhanced progress mapping starting module loading at 10%
                    progress_map = {
                        "Loading theme framework...": 5,
                        "Loading UI framework...": 8,
                        
                        # Module loading phase - starts at 10% and goes to 85%
                        "Loading translation modules...": 10,
                        "Initializing module system...": 15,
                        "Loading translation engine...": 20,
                        "Validating translation engine...": 30,
                        "✅ translation engine loaded": 40,
                        "Loading glossary extractor...": 45,
                        "Validating glossary extractor...": 55,
                        "✅ glossary extractor loaded": 65,
                        "Loading EPUB converter...": 70,
                        "✅ EPUB converter loaded": 75,
                        "Loading QA scanner...": 78,
                        "✅ QA scanner loaded": 82,
                        "Finalizing module initialization...": 85,
                        "✅ All modules loaded successfully": 88,
                        
                        "Creating main window...": 92,
                        "Ready!": 100
                    }
                    
                    # Check for exact matches first
                    if message in progress_map:
                        self.set_progress(progress_map[message])
                    else:
                        # Check for partial matches
                        for key, value in progress_map.items():
                            if key in message:
                                self.set_progress(value)
                                break
                    
                    self.splash_window.update()
            except:
                pass
    
    def set_progress(self, value):
        """Manually set progress value (0-100)"""
        self.progress_value = max(0, min(100, value))
    
    def close_splash(self):
            """Close the splash screen with proper text visibility"""
            try:
                # IMPORTANT: Cancel the animation first
                if self._after_id and self.splash_window:
                    try:
                        self.splash_window.after_cancel(self._after_id)
                    except:
                        pass
                    self._after_id = None
                
                if self.splash_window and self.splash_window.winfo_exists():
                    # Set to 100% and ensure text is visible
                    self.progress_value = 100
                    
                    # Update display one last time without scheduling another callback
                    if hasattr(self, 'progress_fill') and self.progress_fill:
                        self.progress_bg.delete(self.progress_fill)
                    self.progress_bg.delete("highlight")
                    
                    # Create the 100% progress bar (but leave space for text)
                    fill_width = int((self.progress_value / 100) * (self.canvas_width - 6))
                    if fill_width > 0:
                        # Create progress fill that doesn't cover the text area
                        self.progress_fill = self.progress_bg.create_rectangle(
                            3, 3, 3 + fill_width, self.canvas_height - 3, 
                            fill='#4a9eff', outline=''
                        )
                        
                        # Add highlight effect
                        if fill_width > 10:
                            self.progress_bg.create_rectangle(
                                3, 3, min(13, 3 + fill_width), 12,
                                fill='#6bb6ff', outline='', tags="highlight"
                            )
                    
                    # CRITICAL: Make sure text stays on top and is visible
                    if hasattr(self, 'progress_text'):
                        self.progress_bg.itemconfig(self.progress_text, text="100%", fill='#ffffff')
                    
                    # Update all outline layers for better visibility
                    for item in self.progress_bg.find_withtag("outline"):
                        self.progress_bg.itemconfig(item, text="100%", fill='#000000')
                    
                    # Ensure text layers are on top of progress fill
                    self.progress_bg.tag_raise("outline")
                    if hasattr(self, 'progress_text'):
                        self.progress_bg.tag_raise(self.progress_text)
                    
                    self.splash_window.update()
                    time.sleep(0.1)
                    
                    self.splash_window.destroy()
                    self.splash_window = None
            except:
                # Ensure cleanup even on error
                self._after_id = None
                self.splash_window = None
