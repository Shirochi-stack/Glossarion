# pyside6_loading_dialog.py
"""PySide6-based loading dialog for manga translator model preloading.

This module provides a modern, responsive loading dialog using PySide6
that can run independently from the main tkinter application.
"""

import sys
from threading import Thread, Event
from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QLabel, 
    QProgressBar, QWidget
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, Slot
from PySide6.QtGui import QFont, QIcon


class LoadingSignals(QObject):
    """Signals for thread-safe communication with the loading dialog."""
    update_status = Signal(str)
    close_dialog = Signal()


class PySide6LoadingDialog(QDialog):
    """Modern loading dialog for model preloading using PySide6."""
    
    def __init__(self, parent=None, icon_path=None):
        super().__init__(parent)
        self.setWindowTitle("Loading Manga Translator")
        self.setFixedSize(400, 200)
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        
        # Set icon if provided
        if icon_path:
            try:
                self.setWindowIcon(QIcon(icon_path))
            except:
                pass
        
        # Setup UI
        self._setup_ui()
        
        # Center on screen
        self._center_on_screen()
        
        # Setup signals for thread-safe updates
        self.signals = LoadingSignals()
        self.signals.update_status.connect(self._update_status_label)
        self.signals.close_dialog.connect(self.accept)
    
    def _setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)
        
        # Title
        title_label = QLabel("Loading Models")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        status_font = QFont()
        status_font.setPointSize(10)
        self.status_label.setFont(status_font)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        # Spacer
        layout.addStretch()
        
        # Progress bar (indeterminate)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)  # Indeterminate mode
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)
    
    def _center_on_screen(self):
        """Center the dialog on the screen."""
        screen_geometry = QApplication.primaryScreen().geometry()
        dialog_geometry = self.frameGeometry()
        center_point = screen_geometry.center()
        dialog_geometry.moveCenter(center_point)
        self.move(dialog_geometry.topLeft())
    
    @Slot(str)
    def _update_status_label(self, text):
        """Update the status label (thread-safe)."""
        self.status_label.setText(text)
    
    def update_status(self, text):
        """Update status from any thread."""
        self.signals.update_status.emit(text)
    
    def close_dialog(self):
        """Close the dialog from any thread."""
        self.signals.close_dialog.emit()


class LoadingDialogManager:
    """Manager for running the PySide6 loading dialog in a separate thread."""
    
    def __init__(self, icon_path=None):
        self.icon_path = icon_path
        self.dialog = None
        self.app = None
        self.thread = None
        self.ready_event = Event()
        self._closed = False
        self._lock = __import__('threading').Lock()
    
    def _run_dialog(self):
        """Run the dialog in its own Qt event loop."""
        try:
            # Create QApplication if it doesn't exist
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            self.app = app
            
            # Create and show dialog
            self.dialog = PySide6LoadingDialog(icon_path=self.icon_path)
            
            # Signal that dialog is ready
            self.ready_event.set()
            
            # Show dialog and run event loop
            self.dialog.exec()
            
        except Exception as e:
            print(f"Loading dialog error: {e}")
        finally:
            with self._lock:
                self._closed = True
    
    def start(self):
        """Start the loading dialog in a separate thread."""
        self.thread = Thread(target=self._run_dialog, daemon=True)
        self.thread.start()
        
        # Wait for dialog to be ready
        self.ready_event.wait(timeout=2.0)
        
        return self
    
    def update_status(self, text):
        """Update the status text (thread-safe)."""
        with self._lock:
            if self.dialog and not self._closed:
                try:
                    self.dialog.update_status(text)
                except:
                    pass
    
    def close(self):
        """Close the loading dialog."""
        with self._lock:
            if self.dialog and not self._closed:
                try:
                    self.dialog.close_dialog()
                except:
                    pass
        
        # Wait a bit for the dialog to close
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


# Example usage and testing
if __name__ == "__main__":
    import time
    
    # Create and show loading dialog
    manager = LoadingDialogManager()
    manager.start()
    
    # Simulate model loading
    statuses = [
        "Checking configuration...",
        "Loading RT-DETR ONNX...",
        "✓ RT-DETR ONNX ready",
        "Loading LaMa...",
        "✓ LaMa ready",
        "✓ All models loaded!"
    ]
    
    for status in statuses:
        time.sleep(1.5)
        manager.update_status(status)
    
    time.sleep(1)
    manager.close()