#!/usr/bin/env python3
"""
Glossarion Launcher - Main Entry Point
This is the primary entry point that shows the splash screen immediately 
before loading heavy modules and launching the main application.
"""

# Minimal imports for fastest startup
import sys
import os

# Show immediate feedback if running as exe
if getattr(sys, 'frozen', False):
    print("Starting Glossarion...", end='', flush=True)

# Fix Python path for Windows file association issues
if sys.platform == "win32":
    # Add user site-packages to path
    import site
    user_site = site.getusersitepackages()
    if user_site and os.path.exists(user_site) and user_site not in sys.path:
        sys.path.insert(0, user_site)
    
    # Also add the specific Python 3.13 user packages
    py313_packages = os.path.expanduser(r"~\AppData\Roaming\Python\Python313\site-packages")
    if os.path.exists(py313_packages) and py313_packages.lower() not in [p.lower() for p in sys.path]:
        sys.path.insert(0, py313_packages)

# Now import the rest
import time
import atexit
import threading
import traceback


class SplashManager:
    """Enhanced splash screen manager that serves as the application launcher"""
    
    def __init__(self):
        self.splash_window = None
        self._status_text = "Initializing..."
        self.progress_value = 0
        self._target_progress = 0
        self.canvas_width = 320
        self.canvas_height = 36
        self._after_id = None
        self._launch_success = False
        self._progress_lock = threading.Lock()
        
    def start_splash(self):
        """Create splash window on main thread"""
        try:
            import tkinter as tk
            
            print("🎨 Starting splash screen...")
            
            # Create splash window
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
            
            # Load icon
            self._load_icon(main_frame)
            
            # Title
            title_label = tk.Label(main_frame, text="Glossarion v2.6.9", 
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
            
            # Progress bar container
            progress_frame = tk.Frame(main_frame, bg='#2b2b2b')
            progress_frame.pack(pady=(5, 15))
            
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
            
            # Progress percentage text
            text_x = self.canvas_width // 2
            text_y = self.canvas_height // 2
            
            progress_font = ('Arial', 11, 'bold')
            
            # Create text with white color
            self.progress_text = self.progress_bg.create_text(text_x, text_y, text="0%",
                                                             fill='#ffffff', font=progress_font,
                                                             anchor='center')
            
            # Version info
            self.version_label = tk.Label(main_frame, text="Starting up...", 
                                   bg='#2b2b2b', fg='#888888', font=('Arial', 9))
            self.version_label.pack(side='bottom', pady=(0, 15))
            
            # Start progress animation
            self._animate_progress()
            
            # Update the display
            self.splash_window.update()
            
            # Register cleanup
            atexit.register(self.cleanup)
            return True
            
        except Exception as e:
            print(f"⚠️ Could not start splash: {e}")
            return False
    
    def _load_icon(self, parent):
        """Load the Halgakos.ico icon"""
        try:
            import os
            import sys
            import tkinter as tk
            
            if getattr(sys, 'frozen', False):
                base_dir = sys._MEIPASS
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))
            
            ico_path = os.path.join(base_dir, 'Halgakos.ico')
            
            if os.path.isfile(ico_path):
                try:
                    from PIL import Image, ImageTk
                    pil_image = Image.open(ico_path)
                    pil_image = pil_image.resize((128, 128), Image.Resampling.LANCZOS)
                    icon_photo = ImageTk.PhotoImage(pil_image, master=self.splash_window)
                    icon_label = tk.Label(parent, image=icon_photo, bg='#2b2b2b')
                    icon_label.image = icon_photo
                    icon_label.pack(pady=(20, 10))
                    return
                except ImportError:
                    pass
        except Exception:
            pass
        
        # Fallback emoji
        import tkinter as tk
        icon_frame = tk.Frame(parent, bg='#4a9eff', width=128, height=128)
        icon_frame.pack(pady=(20, 10))
        icon_frame.pack_propagate(False)
        
        icon_label = tk.Label(icon_frame, text="📚", font=('Arial', 64), 
                             bg='#4a9eff', fg='white')
        icon_label.pack(expand=True)

    def _animate_progress(self):
        """Smooth progress bar animation with interpolation"""
        if self._after_id:
            try:
                self.splash_window.after_cancel(self._after_id)
            except:
                pass
            self._after_id = None
            
        if self.splash_window:
            try:
                # Check if window still exists
                if not self.splash_window.winfo_exists():
                    return
                
                with self._progress_lock:
                    # Very fast interpolation for instant visual updates
                    if self.progress_value < self._target_progress:
                        # Jump directly to target if close enough
                        distance = self._target_progress - self.progress_value
                        if distance < 5:
                            self.progress_value = self._target_progress
                        else:
                            # Very fast steps - 50% of remaining distance
                            step = distance * 0.5
                            self.progress_value = min(self.progress_value + step, self._target_progress)
                    
                    # Update progress bar fill
                    if self.progress_fill:
                        self.progress_bg.delete(self.progress_fill)
                    self.progress_bg.delete("highlight")
                    
                    # Calculate fill width
                    fill_width = int((self.progress_value / 100) * (self.canvas_width - 6))
                    if fill_width > 0:
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
                    
                    # Update percentage text
                    percent_text = f"{int(self.progress_value)}%"
                    self.progress_bg.itemconfig(self.progress_text, text=percent_text)
                    
                    # Ensure text stays on top
                    self.progress_bg.tag_raise(self.progress_text)

                # Schedule next update - very fast for responsive animation
                self._after_id = self.splash_window.after(16, self._animate_progress)  # ~60fps
                
            except Exception:
                self._after_id = None
    
    def set_progress(self, value):
        """Set target progress value for smooth animation"""
        with self._progress_lock:
            self._target_progress = max(0, min(100, value))
    
    def update_status(self, message, progress=None):
        """Update splash status and optionally set specific progress"""
        self._status_text = message
        try:
            if self.splash_window and hasattr(self, 'status_label'):
                self.status_label.config(text=message)
                
                # If specific progress provided, use it
                if progress is not None:
                    self.set_progress(progress)
                
                # Update version label for certain statuses
                if "Ready!" in message:
                    self.version_label.config(text="✅ Launch successful!")
                elif "Loading" in message and "module" in message:
                    # Extract module number if present
                    import re
                    match = re.search(r'(\d+)/(\d+)', message)
                    if match:
                        current, total = int(match.group(1)), int(match.group(2))
                        self.version_label.config(text=f"Module {current} of {total}")
                elif "Checking" in message:
                    self.version_label.config(text="Verifying installation...")
                
                self.splash_window.update()
        except:
            pass
    
    def close_splash(self):
        """Close the splash screen"""
        try:
            # Cancel any pending animations first
            if self._after_id:
                try:
                    if self.splash_window and self.splash_window.winfo_exists():
                        self.splash_window.after_cancel(self._after_id)
                except:
                    pass
                self._after_id = None
            
            if self.splash_window:
                try:
                    if self.splash_window.winfo_exists():
                        # Just close without forcing to 100%
                        self.splash_window.quit()
                        self.splash_window.destroy()
                except:
                    pass
                finally:
                    self.splash_window = None
                
        except Exception as e:
            self._after_id = None
            self.splash_window = None
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self._after_id:
                try:
                    if self.splash_window and self.splash_window.winfo_exists():
                        self.splash_window.after_cancel(self._after_id)
                except:
                    pass
                self._after_id = None
            
            self.close_splash()
        except:
            pass
    
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        missing_deps = []
        
        # Check for ttkbootstrap
        try:
            import ttkbootstrap
            print(f"   ✅ Found ttkbootstrap at: {ttkbootstrap.__file__}")
        except ImportError as e:
            print(f"   ❌ Could not import ttkbootstrap: {e}")
            missing_deps.append("ttkbootstrap")
        
        # Check for PIL/Pillow (optional but recommended)
        try:
            from PIL import Image
        except ImportError:
            print("   ⚠️ Warning: PIL/Pillow not installed. Icon loading may be limited.")
        
        return missing_deps
    
    def show_dependency_error(self, missing_deps):
        """Show error dialog for missing dependencies"""
        try:
            import tkinter as tk
            from tkinter import messagebox
            
            if self.splash_window:
                self.update_status("❌ Missing dependencies", 0)
                self.version_label.config(text="Installation required", fg='#ff5555')
                
                # Create error message
                error_msg = "The following required packages are not installed:\n\n"
                for dep in missing_deps:
                    error_msg += f"• {dep}\n"
                
                error_msg += "\nPlease install them using:\n"
                for dep in missing_deps:
                    error_msg += f"pip install {dep}\n"
                
                # Show error dialog
                messagebox.showerror("Missing Dependencies", error_msg)
                
                # Keep splash visible for a moment
                time.sleep(2)
        except Exception as e:
            print(f"Error showing dependency dialog: {e}")
    
    def launch_main_app(self):
        """Launch the main application with detailed progress tracking"""
        try:
            # Phase 1: Check dependencies (0-10%)
            self.update_status("Checking dependencies...", 5)
            missing_deps = self.check_dependencies()
            
            if missing_deps:
                print(f"❌ Missing required dependencies: {', '.join(missing_deps)}")
                self.show_dependency_error(missing_deps)
                self.cleanup()
                sys.exit(1)
            
            self.set_progress(10)
            
            # Phase 2: Load ttkbootstrap (10-20%)
            self.update_status("Loading UI framework...", 15)
            import ttkbootstrap as tb
            self.set_progress(20)
            
            # Phase 3: Import translator_gui module (20-30%)
            self.update_status("Loading translator GUI...", 25)
            import translator_gui
            self.set_progress(30)
            
            # Phase 4: Load modules before closing splash (30-80%)
            self.update_status("Loading translation engine...", 40)
            try:
                import TransateKRtoEN
            except ImportError:
                pass
            
            self.update_status("Loading glossary extractor...", 50)
            try:
                import extract_glossary_from_epub
            except ImportError:
                pass
            
            self.update_status("Loading EPUB converter...", 60)
            try:
                import epub_converter
            except ImportError:
                pass
            
            self.update_status("Loading QA scanner...", 70)
            try:
                import scan_html_folder
            except ImportError:
                pass
            
            self.update_status("Preparing workspace...", 80)
            self.update_status("Finalizing...", 90)
            self.update_status("Ready!", 100)
            
            # Force immediate 100% display
            with self._progress_lock:
                self.progress_value = 100  # Jump directly to 100
            
            # Manually update the progress bar to 100% immediately
            if self.progress_fill:
                self.progress_bg.delete(self.progress_fill)
            self.progress_bg.delete("highlight")
            
            # Draw full progress bar
            self.progress_fill = self.progress_bg.create_rectangle(
                3, 3, self.canvas_width - 3, self.canvas_height - 3,
                fill='#4a9eff', outline=''
            )
            
            # Update text to 100%
            self.progress_bg.itemconfig(self.progress_text, text="100%")
            self.progress_bg.tag_raise(self.progress_text)
            
            # Force update
            self.splash_window.update()
            
            time.sleep(0.1)  # Just 100ms to see 100%
            
            # Get screen dimensions BEFORE closing splash
            screen_width = self.splash_window.winfo_screenwidth()
            screen_height = self.splash_window.winfo_screenheight()
            
            # Calculate window position
            window_width = 1550
            window_height = 1000
            
            if window_width > screen_width:
                window_width = int(screen_width * 0.9)
            if window_height > screen_height:
                window_height = int(screen_height * 0.9)
            
            x = max(0, (screen_width - window_width) // 2)
            y = max(0, (screen_height - window_height) // 2)
            
            # NOW close splash
            self.close_splash()
            
            # Small delay to ensure splash is fully closed
            time.sleep(0.05)
            
            # NOW create ttkbootstrap window (no more Tk conflicts)
            root = tb.Window(themename="darkly")
            root.withdraw()
            root.geometry(f"{window_width}x{window_height}+{x}+{y}")
            root.minsize(1600, 1000)
            
            # Initialize GUI application
            # Disable module loading on init
            original_lazy_load = translator_gui.TranslatorGUI._lazy_load_modules
            translator_gui.TranslatorGUI._lazy_load_modules = lambda self, cb=None: True
            
            app = translator_gui.TranslatorGUI(root)
            
            # Restore original method and load modules
            translator_gui.TranslatorGUI._lazy_load_modules = original_lazy_load
            app._lazy_load_modules()
            
            # Prepare and show main window
            root.update_idletasks()
            root.deiconify()
            root.lift()
            root.focus_force()
            
            print("✅ Ready to use!")
            
            # Mark launch as successful
            self._launch_success = True
            
            # Start main loop
            root.mainloop()
            
        except ImportError as e:
            module_name = str(e).split("'")[1] if "'" in str(e) else "unknown module"
            print(f"❌ Failed to import required module: {module_name}")
            traceback.print_exc()
            
            # Show error in splash
            try:
                if self.splash_window and self.splash_window.winfo_exists():
                    self.update_status(f"❌ Missing module: {module_name}", 0)
                    if hasattr(self, 'version_label'):
                        self.version_label.config(text="Please install required dependencies", fg='#ff5555')
                    
                    # Show installation instructions
                    import tkinter as tk
                    from tkinter import messagebox
                    
                    install_msg = f"Required module '{module_name}' is not installed.\n\n"
                    install_msg += f"Please install it using:\npip install {module_name}"
                    
                    messagebox.showerror("Module Not Found", install_msg)
                    time.sleep(1)
            except:
                pass
            
            self.cleanup()
            sys.exit(1)
            
        except Exception as e:
            print(f"❌ Failed to launch application: {e}")
            traceback.print_exc()
            
            # Show error in splash if possible
            try:
                if self.splash_window and self.splash_window.winfo_exists():
                    self.update_status(f"❌ Launch failed: {str(e)[:50]}...", 0)
                    if hasattr(self, 'version_label'):
                        self.version_label.config(text="See console for details", fg='#ff5555')
                    time.sleep(3)
            except:
                pass
            
            self.cleanup()
            sys.exit(1)


def main():
    """Main entry point for Glossarion"""
    print("🚀 Starting Glossarion v2.6.9...")
    print(f"📂 Working directory: {os.getcwd()}")
    print(f"🐍 Python version: {sys.version.split()[0]}")
    print(f"🐍 Python executable: {sys.executable}")
    
    # Add user site-packages to path if needed
    import site
    user_site = site.getusersitepackages()
    if user_site and os.path.exists(user_site) and user_site not in sys.path:
        sys.path.insert(0, user_site)
        print(f"📦 Added user site-packages to path: {user_site}")
    
    # Create and start splash manager
    splash_manager = SplashManager()
    
    # Start splash screen
    splash_started = splash_manager.start_splash()
    
    if not splash_started:
        print("⚠️ Failed to start splash screen, launching directly...")
        # Fallback: launch without splash
        try:
            import ttkbootstrap as tb
            import translator_gui
            
            root = tb.Window(themename="darkly")
            app = translator_gui.TranslatorGUI(root)
            root.mainloop()
        except Exception as e:
            print(f"❌ Failed to start application: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Launch main app with splash screen
        splash_manager.launch_main_app()


if __name__ == "__main__":
    # Handle multiprocessing for frozen executables
    if getattr(sys, 'frozen', False):
        try:
            import multiprocessing
            multiprocessing.freeze_support()
        except:
            pass
    
    # Launch the application
    main()
