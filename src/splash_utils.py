# splash_utils.py - PySide6 Version
import sys
import time
import atexit
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QProgressBar, QVBoxLayout
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QPalette, QColor, QPixmap

class SplashManager:
    """PySide6 splash screen manager - shows faster than Tkinter"""
    
    def __init__(self):
        self.splash_window = None
        self.app = None
        self._status_text = "Initializing..."
        self.progress_value = 0  # Track actual progress 0-100
        self.timer = None
        self.status_label = None
        self.progress_bar = None
        self.progress_label = None
        
    def start_splash(self):
        """Create splash window with PySide6"""
        try:
            print("🎨 Starting PySide6 splash screen...")
            
            # Create QApplication if it doesn't exist
            if not QApplication.instance():
                self.app = QApplication(sys.argv)
            else:
                self.app = QApplication.instance()
            
            # Create main splash widget
            self.splash_window = QWidget()
            self.splash_window.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
            self.splash_window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
            # Use screen ratios for sizing
            screen = self.app.primaryScreen().geometry()
            width = int(screen.width() * 0.24)  # 24% of screen width
            height = int(screen.height() * 0.30)  # 30% of screen height (reduced from 36%)
            self.splash_window.setFixedSize(width, height)
            
            # Set dark background with border
            palette = self.splash_window.palette()
            palette.setColor(QPalette.ColorRole.Window, QColor('#2b2b2b'))
            self.splash_window.setPalette(palette)
            self.splash_window.setAutoFillBackground(True)
            
            # Add border using stylesheet (only to main window, not children)
            self.splash_window.setStyleSheet("""
                QWidget#splash_main {
                    background-color: #2b2b2b;
                    border: 1px solid #3d4450;
                    border-radius: 8px;
                }
            """)
            self.splash_window.setObjectName("splash_main")
            
            # Main layout with tighter spacing
            layout = QVBoxLayout()
            layout.setContentsMargins(15, 15, 15, 15)  # Reduced margins from 20 to 15
            layout.setSpacing(8)  # Reduced spacing from 10 to 8
            
            # Load and add icon
            self._load_icon(layout)
            
            # Title
            title_label = QLabel("Glossarion v6.5.1")
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_font = QFont("Arial", 20, QFont.Weight.Bold)
            title_label.setFont(title_font)
            title_label.setStyleSheet("color: #4a9eff; background: transparent;")
            layout.addWidget(title_label)
            
            # Subtitle
            subtitle_label = QLabel("Advanced AI Translation Suite")
            subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            subtitle_font = QFont("Arial", 12)
            subtitle_label.setFont(subtitle_font)
            subtitle_label.setStyleSheet("color: #cccccc; background: transparent;")
            layout.addWidget(subtitle_label)
            
            layout.addSpacing(6)  # Reduced from 10 to 6
            
            # Status label with fixed height to prevent shaking
            self.status_label = QLabel(self._status_text)
            self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_font = QFont("Arial", 11)
            self.status_label.setFont(status_font)
            self.status_label.setFixedHeight(30)  # Fixed height to prevent layout shifts
            self.status_label.setStyleSheet("color: #ffffff; background: transparent;")
            layout.addWidget(self.status_label)
            
            # Progress bar
            self.progress_bar = QProgressBar()
            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(100)
            self.progress_bar.setValue(0)
            self.progress_bar.setTextVisible(False)
            self.progress_bar.setFixedHeight(36)
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #666666;
                    border-radius: 5px;
                    background-color: #1a1a1a;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                      stop:0 #6bb6ff, stop:1 #4a9eff);
                    border-radius: 3px;
                }
            """)
            layout.addWidget(self.progress_bar)
            
            # Progress percentage label (overlaid on progress bar)
            self.progress_label = QLabel("0%", self.progress_bar)
            self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            progress_font = QFont("Montserrat", 12, QFont.Weight.Bold)
            self.progress_label.setFont(progress_font)
            self.progress_label.setStyleSheet("""
                color: #ffffff;
                background: transparent;
                border: none;
            """)
            # Position label to match progress bar size dynamically
            # Use a timer to ensure the progress bar has been laid out first
            def position_progress_label():
                if self.progress_bar and self.progress_label:
                    bar_width = self.progress_bar.width()
                    bar_height = self.progress_bar.height()
                    self.progress_label.setGeometry(0, 0, bar_width, bar_height)
            QTimer.singleShot(0, position_progress_label)
            self.progress_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            
            layout.addSpacing(3)  # Reduced from 5 to 3
            
            # Version info
            version_label = QLabel("Starting up...")
            version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            version_font = QFont("Arial", 9)
            version_label.setFont(version_font)
            version_label.setStyleSheet("color: #888888; background: transparent;")
            layout.addWidget(version_label)
            
            layout.addStretch()
            
            self.splash_window.setLayout(layout)
            
            # Center the window
            screen = self.app.primaryScreen().geometry()
            x = (screen.width() - self.splash_window.width()) // 2
            y = (screen.height() - self.splash_window.height()) // 2
            self.splash_window.move(x, y)
            
            # Show splash
            self.splash_window.show()
            
            # Start progress animation
            self._animate_progress()
            
            # Process events to show immediately
            self.app.processEvents()
            
            # Register cleanup
            atexit.register(self.close_splash)
            
            print("✅ PySide6 splash screen displayed")
            return True
            
        except Exception as e:
            print(f"⚠️ Could not start splash: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_icon(self, layout):
        """Load the Halgakos.ico icon"""
        try:
            import os
            from PySide6.QtGui import QIcon
            
            if getattr(sys, 'frozen', False):
                # Running as .exe
                base_dir = sys._MEIPASS
            else:
                # Running as .py files
                base_dir = os.path.dirname(os.path.abspath(__file__))
            
            ico_path = os.path.join(base_dir, 'Halgakos.ico')
            
            if os.path.isfile(ico_path):
                # Load icon with Qt and get the highest resolution available
                icon = QIcon(ico_path)
                available_sizes = icon.availableSizes()
                
                if available_sizes:
                    # Get the largest available size from the ICO file
                    largest_size = max(available_sizes, key=lambda s: s.width() * s.height())
                    print(f"📐 Loading icon at native size: {largest_size.width()}x{largest_size.height()}")
                    
                    # Get pixmap at the largest native size
                    pixmap = icon.pixmap(largest_size)
                    
                    # Only scale if the native size is larger than 110x110
                    # This preserves quality by using native resolution when possible
                    if largest_size.width() > 110 or largest_size.height() > 110:
                        pixmap = pixmap.scaled(110, 110, Qt.AspectRatioMode.KeepAspectRatio, 
                                              Qt.TransformationMode.SmoothTransformation)
                else:
                    # Fallback to direct load
                    pixmap = QPixmap(ico_path)
                    if pixmap.width() > 110 or pixmap.height() > 110:
                        pixmap = pixmap.scaled(110, 110, Qt.AspectRatioMode.KeepAspectRatio, 
                                              Qt.TransformationMode.SmoothTransformation)
                
                if not pixmap.isNull():
                    icon_label = QLabel()
                    icon_label.setPixmap(pixmap)
                    icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    icon_label.setStyleSheet("background: transparent;")
                    layout.addWidget(icon_label)
                    return
        except Exception as e:
            print(f"⚠️ Could not load icon: {e}")
            import traceback
            traceback.print_exc()
        
        # Fallback emoji if icon loading fails
        icon_label = QLabel("📚")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_font = QFont("Arial", 56)  # Reduced from 64
        icon_label.setFont(icon_font)
        icon_label.setStyleSheet("background: #4a9eff; color: white; border-radius: 10px;")
        icon_label.setFixedSize(110, 110)  # Reduced from 128x128
        layout.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignCenter)

    def _animate_progress(self):
        """Animate progress bar filling up"""
        if not self.timer:
            self.timer = QTimer()
            self.timer.timeout.connect(self._update_progress)
            self.timer.start(100)  # Update every 100ms
    
    def _update_progress(self):
        """Update progress animation"""
        try:
            # Skip auto-animation if manual control is active
            if getattr(self, '_manual_progress', False):
                return
                
            if self.splash_window and self.progress_value < 100:
                # Auto-increment progress for visual effect during startup
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
                
                # Update progress bar
                if self.progress_bar:
                    self.progress_bar.setValue(self.progress_value)
                
                # Update percentage text
                if self.progress_label:
                    self.progress_label.setText(f"{self.progress_value}%")
                
                # Process events
                if self.app:
                    self.app.processEvents()
        except Exception:
            pass
    
    def update_status(self, message):
        """Update splash status and progress"""
        self._status_text = message
        try:
            if self.splash_window and self.status_label:
                self.status_label.setText(message)
                
                # Enhanced progress mapping
                progress_map = {
                    "Loading theme framework...": 5,
                    "Loading UI framework...": 8,
                    
                    # Script validation phase - 10-25%
                    "Scanning Python modules...": 10,
                    "Validating": 12,  # Partial match for "Validating X Python scripts..."
                    "✅ All scripts validated": 25,
                    
                    # Module loading phase - 30-85%
                    "Loading translation modules...": 30,
                    "Initializing module system...": 35,
                    "Loading translation engine...": 40,
                    "Validating translation engine...": 45,
                    "✅ translation engine loaded": 50,
                    "Loading glossary extractor...": 55,
                    "Validating glossary extractor...": 60,
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
                
                # Process events
                if self.app:
                    self.app.processEvents()
        except Exception:
            pass
    
    def set_progress(self, value):
        """Manually set progress value (0-100)"""
        self.progress_value = max(0, min(100, value))
        if self.progress_bar:
            self.progress_bar.setValue(self.progress_value)
        if self.progress_label:
            self.progress_label.setText(f"{self.progress_value}%")
        
        # Process events to ensure smooth UI updates
        if self.app:
            self.app.processEvents()
    
    def validate_all_scripts(self, base_dir=None):
        """Validate that all Python scripts in the project compile without syntax errors
        
        Args:
            base_dir: Directory to scan for Python files. Defaults to script directory.
            
        Returns:
            tuple: (success_count, total_count, failed_scripts)
        """
        import os
        import py_compile
        
        # Enable manual progress control and stop auto-animation timer
        self._manual_progress = True
        if self.timer:
            self.timer.stop()
        
        if base_dir is None:
            if getattr(sys, 'frozen', False):
                base_dir = sys._MEIPASS
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.update_status("Scanning Python modules...")
        self.set_progress(10)  # Explicitly set starting progress
        
        # Find all Python files
        python_files = []
        try:
            for file in os.listdir(base_dir):
                if file.endswith('.py') and not file.startswith('.'):
                    python_files.append(os.path.join(base_dir, file))
        except Exception as e:
            print(f"⚠️ Could not scan directory: {e}")
            return (0, 0, [])
        
        total_count = len(python_files)
        success_count = 0
        failed_scripts = []
        
        if total_count == 0:
            return (0, 0, [])
        
        self.update_status(f"Validating {total_count} Python scripts...")
        print(f"🔍 Validating {total_count} Python scripts for compilation errors...")
        
        # Check each file
        for idx, filepath in enumerate(python_files, 1):
            filename = os.path.basename(filepath)
            
            # Log progress for debugging hangs
            print(f"📂 Scanning files: [{idx}/{total_count}] {filename}")
            
            try:
                # Try to compile the file
                py_compile.compile(filepath, doraise=True)
                success_count += 1
                print(f"✅ Validated {idx}/{total_count}: {filename}")
                
                # Update progress based on validation progress
                # Map 0-100% of files to 15-25% of total progress
                progress_pct = 15 + int((idx / total_count) * 10)
                # Only update if progress actually changed (avoid animation churn)
                if progress_pct != self.progress_value:
                    self.set_progress(progress_pct)
                
            except SyntaxError as e:
                failed_scripts.append((filename, str(e)))
                print(f"❌ Syntax error in {filename}: {e}")
                import traceback
                print(f"Full traceback:\n{traceback.format_exc()}")
            except Exception as e:
                failed_scripts.append((filename, str(e)))
                print(f"⚠️ Could not validate {filename}: {e}")
                import traceback
                print(f"Full traceback:\n{traceback.format_exc()}")
            
            # Process events to keep UI responsive
            if self.app:
                self.app.processEvents()
        
        # Report results
        if failed_scripts:
            self.update_status(f"⚠️ {success_count}/{total_count} scripts valid")
            print(f"\n⚠️ Validation complete: {success_count}/{total_count} scripts compiled successfully")
            print(f"Failed scripts:")
            for script, error in failed_scripts:
                print(f"  • {script}: {error}")
        else:
            self.update_status("✅ All scripts validated")
            print(f"✅ All {total_count} Python scripts validated successfully")
        
        # Re-enable auto-animation after validation completes
        self._manual_progress = False
        
        return (success_count, total_count, failed_scripts)
    
    def close_splash(self):
        """Close the splash screen"""
        try:
            # Stop timer first
            if self.timer:
                self.timer.stop()
                self.timer = None
            
            # Update progress to 100% before closing (only if window still exists)
            if self.splash_window:
                try:
                    self.progress_value = 100
                    if self.progress_bar:
                        self.progress_bar.setValue(100)
                    if self.progress_label:
                        self.progress_label.setText("100%")
                    
                    # Process events one last time
                    if self.app:
                        self.app.processEvents()
                    
                    time.sleep(0.1)
                except RuntimeError:
                    # Widget already deleted by Qt, that's okay
                    pass
                
                # Close splash window
                try:
                    self.splash_window.close()
                except RuntimeError:
                    # Already deleted
                    pass
                
                self.splash_window = None
            
            print("✅ Splash screen closed")
            
        except Exception as e:
            print(f"⚠️ Error closing splash: {e}")
        finally:
            # Ensure cleanup
            self.timer = None
            self.splash_window = None
            self.progress_bar = None
            self.progress_label = None
            self.status_label = None
