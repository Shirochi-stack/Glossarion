# splash_utils.py - PySide6 Version
import sys
import time
import atexit
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QProgressBar
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QPalette, QColor, QFont

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
            self.splash_window.setFixedSize(450, 370)
            
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
            
            # Main layout
            layout = QVBoxLayout()
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(10)
            
            # Load and add icon
            self._load_icon(layout)
            
            # Title
            title_label = QLabel("Glossarion v6.0.1")
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
            
            layout.addSpacing(10)
            
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
            # Position label over progress bar with fixed geometry
            self.progress_label.setGeometry(0, 0, 400, 36)  # Fixed width to match progress bar
            self.progress_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            
            layout.addSpacing(5)
            
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
            
            if getattr(sys, 'frozen', False):
                # Running as .exe
                base_dir = sys._MEIPASS
            else:
                # Running as .py files
                base_dir = os.path.dirname(os.path.abspath(__file__))
            
            ico_path = os.path.join(base_dir, 'Halgakos.ico')
            
            if os.path.isfile(ico_path):
                # Load icon with Qt
                pixmap = QPixmap(ico_path)
                if not pixmap.isNull():
                    # Scale to 128x128 with smooth transformation
                    pixmap = pixmap.scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio, 
                                          Qt.TransformationMode.SmoothTransformation)
                    icon_label = QLabel()
                    icon_label.setPixmap(pixmap)
                    icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    icon_label.setStyleSheet("background: transparent;")
                    layout.addWidget(icon_label)
                    return
        except Exception as e:
            print(f"⚠️ Could not load icon: {e}")
        
        # Fallback emoji if icon loading fails
        icon_label = QLabel("📚")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_font = QFont("Arial", 64)
        icon_label.setFont(icon_font)
        icon_label.setStyleSheet("background: #4a9eff; color: white; border-radius: 10px;")
        icon_label.setFixedSize(128, 128)
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
        if self.app:
            self.app.processEvents()
    
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
