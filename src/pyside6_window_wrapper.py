# pyside6_window_wrapper.py
"""PySide6 window wrappers that provide a tkinter-compatible API.

This module allows WindowManager to create PySide6-based windows and dialogs
with a similar API to tkinter, enabling seamless migration.
"""

import sys
import os
from threading import Thread, Event, Lock
from typing import Optional, Callable, Tuple

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QDialog, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QScrollArea, QFrame, QSizePolicy
    )
    from PySide6.QtCore import Qt, Signal, QObject, Slot, QTimer, QSize
    from PySide6.QtGui import QFont, QIcon, QScreen
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False


class PySide6Bridge(QObject):
    """Signal bridge for thread-safe communication with PySide6 widgets."""
    close_signal = Signal()
    show_signal = Signal()
    hide_signal = Signal()
    geometry_signal = Signal(int, int, int, int)  # x, y, width, height
    title_signal = Signal(str)


class PySide6Dialog(QDialog):
    """PySide6 dialog that mimics tkinter Toplevel interface."""
    
    def __init__(self, parent=None, title="Dialog", width=400, height=300, 
                 modal=True, icon_path=None):
        super().__init__(parent)
        
        self.setWindowTitle(title)
        self.setModal(modal)
        
        # Set icon if provided
        if icon_path and os.path.exists(icon_path):
            try:
                self.setWindowIcon(QIcon(icon_path))
            except:
                pass
        
        # Setup signal bridge for thread-safe operations
        self.bridge = PySide6Bridge()
        self.bridge.close_signal.connect(self.accept)
        self.bridge.show_signal.connect(self.show)
        self.bridge.hide_signal.connect(self.hide)
        self.bridge.geometry_signal.connect(self._set_geometry)
        self.bridge.title_signal.connect(self.setWindowTitle)
        
        # Create main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        
        # Set initial size
        self.resize(width, height)
        
        # Center on screen
        self._center_on_screen()
    
    def _center_on_screen(self):
        """Center the dialog on the screen."""
        screen = QApplication.primaryScreen().geometry()
        dialog_geometry = self.frameGeometry()
        center_point = screen.center()
        dialog_geometry.moveCenter(center_point)
        self.move(dialog_geometry.topLeft())
    
    @Slot(int, int, int, int)
    def _set_geometry(self, x, y, width, height):
        """Set geometry (thread-safe)."""
        self.setGeometry(x, y, width, height)
    
    def set_title(self, title):
        """Set window title (thread-safe)."""
        self.bridge.title_signal.emit(title)
    
    def close_dialog(self):
        """Close the dialog (thread-safe)."""
        self.bridge.close_signal.emit()
    
    def show_dialog(self):
        """Show the dialog (thread-safe)."""
        self.bridge.show_signal.emit()
    
    def hide_dialog(self):
        """Hide the dialog (thread-safe)."""
        self.bridge.hide_signal.emit()
    
    def add_widget(self, widget):
        """Add a widget to the main layout."""
        self.main_layout.addWidget(widget)
    
    def add_layout(self, layout):
        """Add a layout to the main layout."""
        self.main_layout.addLayout(layout)


class PySide6ScrollableDialog(PySide6Dialog):
    """Scrollable PySide6 dialog that mimics tkinter scrollable frame."""
    
    def __init__(self, parent=None, title="Dialog", width=600, height=500, 
                 modal=True, icon_path=None):
        # Don't call parent __init__ yet - we need to set up scroll area first
        QDialog.__init__(self, parent)
        
        self.setWindowTitle(title)
        self.setModal(modal)
        
        # Set icon if provided
        if icon_path and os.path.exists(icon_path):
            try:
                self.setWindowIcon(QIcon(icon_path))
            except:
                pass
        
        # Setup signal bridge
        self.bridge = PySide6Bridge()
        self.bridge.close_signal.connect(self.accept)
        self.bridge.show_signal.connect(self.show)
        self.bridge.hide_signal.connect(self.hide)
        self.bridge.geometry_signal.connect(self._set_geometry)
        self.bridge.title_signal.connect(self.setWindowTitle)
        
        # Create scroll area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        
        # Create scrollable content widget
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.content_layout.setSpacing(10)
        
        # Set content widget in scroll area
        self.scroll_area.setWidget(self.content_widget)
        
        # Create main layout and add scroll area
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.scroll_area)
        self.setLayout(main_layout)
        
        # Set initial size
        self.resize(width, height)
        
        # Center on screen
        self._center_on_screen()
    
    def add_widget(self, widget):
        """Add a widget to the scrollable content."""
        self.content_layout.addWidget(widget)
    
    def add_layout(self, layout):
        """Add a layout to the scrollable content."""
        self.content_layout.addLayout(layout)
    
    def get_content_widget(self):
        """Get the scrollable content widget for adding custom widgets."""
        return self.content_widget


class PySide6DialogManager:
    """Manager for PySide6 dialogs running in a separate thread."""
    
    def __init__(self):
        self.app = None
        self.dialog = None
        self.thread = None
        self.ready_event = Event()
        self._lock = Lock()
        self._closed = False
        self._result = None
    
    def create_dialog(self, dialog_class, parent=None, **kwargs):
        """Create and show a dialog in a separate thread."""
        self.dialog_class = dialog_class
        self.dialog_kwargs = kwargs
        
        # Start dialog thread
        self.thread = Thread(target=self._run_dialog, daemon=True)
        self.thread.start()
        
        # Wait for dialog to be ready
        self.ready_event.wait(timeout=2.0)
        
        return self.dialog
    
    def _run_dialog(self):
        """Run the dialog in its own Qt event loop."""
        try:
            # Create QApplication if it doesn't exist
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            self.app = app
            
            # Create dialog
            self.dialog = self.dialog_class(**self.dialog_kwargs)
            
            # Signal that dialog is ready
            self.ready_event.set()
            
            # Show dialog and run event loop
            result = self.dialog.exec()
            
            with self._lock:
                self._result = result
                self._closed = True
            
        except Exception as e:
            print(f"PySide6 dialog error: {e}")
            with self._lock:
                self._closed = True
    
    def wait_for_close(self, timeout=None):
        """Wait for dialog to close."""
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
        return self._result
    
    def close(self):
        """Close the dialog."""
        with self._lock:
            if self.dialog and not self._closed:
                try:
                    self.dialog.close_dialog()
                except:
                    pass
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


class PySide6ProgressDialog(PySide6Dialog):
    """Progress dialog with progress bar and status text."""
    
    def __init__(self, parent=None, title="Progress", width=400, height=200,
                 modal=True, icon_path=None):
        super().__init__(parent, title, width, height, modal, icon_path)
        
        # Remove default layout
        layout = self.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        
        # Create new layout
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)
        
        # Title label
        self.title_label = QLabel("Loading...")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
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
        
        # Progress bar
        from PySide6.QtWidgets import QProgressBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)  # Indeterminate
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)
        
        # Signal for updating status
        self._status_signal = Signal(str)
        self._status_signal = self._create_status_signal()
        self._status_signal.connect(self._update_status_label)
    
    def _create_status_signal(self):
        """Create a signal for updating status."""
        class StatusSignal(QObject):
            update = Signal(str)
        return StatusSignal()
    
    @Slot(str)
    def _update_status_label(self, text):
        """Update status label (thread-safe)."""
        self.status_label.setText(text)
    
    def update_status(self, text):
        """Update status text (thread-safe)."""
        try:
            self._status_signal.update.emit(text)
        except:
            # Fallback to direct update if signal fails
            self.status_label.setText(text)


def create_loading_dialog(parent=None, title="Loading", icon_path=None):
    """Helper function to create a loading dialog."""
    return PySide6ProgressDialog(
        parent=parent,
        title=title,
        width=400,
        height=200,
        modal=True,
        icon_path=icon_path
    )


def create_scrollable_dialog(parent=None, title="Dialog", width=600, height=500,
                            modal=True, icon_path=None):
    """Helper function to create a scrollable dialog."""
    return PySide6ScrollableDialog(
        parent=parent,
        title=title,
        width=width,
        height=height,
        modal=modal,
        icon_path=icon_path
    )


# Example usage and testing
if __name__ == "__main__":
    import time
    
    if PYSIDE6_AVAILABLE:
        # Test progress dialog
        manager = PySide6DialogManager()
        dialog = manager.create_dialog(
            PySide6ProgressDialog,
            title="Test Progress",
            width=400,
            height=200
        )
        
        # Simulate progress updates
        statuses = [
            "Loading component 1...",
            "Loading component 2...",
            "Loading component 3...",
            "Complete!"
        ]
        
        for status in statuses:
            time.sleep(1)
            if dialog:
                dialog.update_status(status)
        
        time.sleep(1)
        manager.close()
        
        print("PySide6 dialog test completed!")
    else:
        print("PySide6 not available for testing.")