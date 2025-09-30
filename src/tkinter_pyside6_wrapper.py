# tkinter_pyside6_wrapper.py
"""
Automatic tkinter-to-PySide6 compatibility wrapper.

This module provides a drop-in replacement for tkinter that uses PySide6 under the hood.
Import this instead of tkinter to automatically convert tkinter code to use Qt widgets.

Usage:
    # Instead of: import tkinter as tk
    # Use:        import tkinter_pyside6_wrapper as tk
"""

import sys
from typing import Any, Callable, Optional, Dict, List, Union

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QDialog, QLabel, QPushButton,
        QLineEdit, QTextEdit, QCheckBox, QRadioButton, QComboBox, QSpinBox,
        QListWidget, QScrollArea, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout,
        QFileDialog, QMessageBox, QColorDialog, QProgressBar, QSlider
    )
    from PySide6.QtCore import Qt, Signal, QObject, QTimer
    from PySide6.QtGui import QFont, QColor, QPixmap, QIcon
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False
    print("⚠️ PySide6 not available, tkinter wrapper will fall back to real tkinter")
    import tkinter as _real_tk
    import tkinter.ttk as _real_ttk


# Global Qt Application instance
_qt_app = None


def _ensure_qt_app():
    """Ensure QApplication exists."""
    global _qt_app
    if not _qt_app:
        _qt_app = QApplication.instance()
        if not _qt_app:
            _qt_app = QApplication(sys.argv)
    return _qt_app


# ==================== Variable Wrappers ====================

class StringVar:
    """Wrapper for tk.StringVar using Python value storage."""
    def __init__(self, value=''):
        self._value = str(value)
        self._callbacks = []
    
    def get(self):
        return self._value
    
    def set(self, value):
        self._value = str(value)
        for callback in self._callbacks:
            callback()
    
    def trace(self, mode, callback):
        """Add trace callback."""
        self._callbacks.append(callback)


class BooleanVar:
    """Wrapper for tk.BooleanVar using Python value storage."""
    def __init__(self, value=False):
        self._value = bool(value)
        self._callbacks = []
    
    def get(self):
        return self._value
    
    def set(self, value):
        self._value = bool(value)
        for callback in self._callbacks:
            callback()
    
    def trace(self, mode, callback):
        """Add trace callback."""
        self._callbacks.append(callback)


class IntVar:
    """Wrapper for tk.IntVar using Python value storage."""
    def __init__(self, value=0):
        self._value = int(value)
        self._callbacks = []
    
    def get(self):
        return self._value
    
    def set(self, value):
        self._value = int(value)
        for callback in self._callbacks:
            callback()
    
    def trace(self, mode, callback):
        """Add trace callback."""
        self._callbacks.append(callback)


class DoubleVar:
    """Wrapper for tk.DoubleVar using Python value storage."""
    def __init__(self, value=0.0):
        self._value = float(value)
        self._callbacks = []
    
    def get(self):
        return self._value
    
    def set(self, value):
        self._value = float(value)
        for callback in self._callbacks:
            callback()
    
    def trace(self, mode, callback):
        """Add trace callback."""
        self._callbacks.append(callback)


# ==================== Widget Wrappers ====================

class WidgetBase:
    """Base class for all widget wrappers."""
    def __init__(self, parent=None):
        self.parent = parent
        self._qt_widget = None
        self._pack_info = {}
        self._grid_info = {}
        self._children = []
    
    def pack(self, **kwargs):
        """Store pack info for later layout."""
        self._pack_info = kwargs
        if self.parent and hasattr(self.parent, '_add_child_packed'):
            self.parent._add_child_packed(self, kwargs)
        return self
    
    def grid(self, **kwargs):
        """Store grid info for later layout."""
        self._grid_info = kwargs
        if self.parent and hasattr(self.parent, '_add_child_grid'):
            self.parent._add_child_grid(self, kwargs)
        return self
    
    def config(self, **kwargs):
        """Configure widget properties."""
        self.configure(**kwargs)
    
    def configure(self, **kwargs):
        """Configure widget properties."""
        for key, value in kwargs.items():
            self._configure_property(key, value)
    
    def _configure_property(self, key, value):
        """Override in subclasses."""
        pass
    
    def bind(self, sequence, func):
        """Bind event (basic implementation)."""
        # Qt signals will be connected in subclasses
        pass
    
    def after(self, ms, func):
        """Schedule function call after ms milliseconds."""
        QTimer.singleShot(ms, func)
    
    def winfo_exists(self):
        """Check if widget still exists."""
        return self._qt_widget is not None


class Frame(WidgetBase):
    """Wrapper for tk.Frame using QWidget."""
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        parent_widget = parent._qt_widget if parent else None
        self._qt_widget = QWidget(parent_widget)
        self._layout = QVBoxLayout(self._qt_widget)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._configure_kwargs(kwargs)
    
    def _configure_kwargs(self, kwargs):
        """Process kwargs during initialization."""
        for key, value in kwargs.items():
            self._configure_property(key, value)
    
    def _add_child_packed(self, child, pack_info):
        """Add child widget with pack layout."""
        if child._qt_widget:
            self._layout.addWidget(child._qt_widget)
            # Handle pack options
            if pack_info.get('side') == 'left':
                # Change to horizontal layout
                if not isinstance(self._layout, QHBoxLayout):
                    # Convert layout
                    pass
    
    def pack_forget(self):
        """Remove widget from layout."""
        if self._qt_widget and self.parent:
            self._qt_widget.setVisible(False)


class Label(WidgetBase):
    """Wrapper for tk.Label using QLabel."""
    def __init__(self, parent=None, text='', **kwargs):
        super().__init__(parent)
        parent_widget = parent._qt_widget if parent else None
        self._qt_widget = QLabel(text, parent_widget)
        self._configure_kwargs(kwargs)
    
    def _configure_kwargs(self, kwargs):
        for key, value in kwargs.items():
            self._configure_property(key, value)
    
    def _configure_property(self, key, value):
        if key == 'text':
            self._qt_widget.setText(str(value))
        elif key == 'font':
            if isinstance(value, tuple):
                font = QFont(value[0], value[1] if len(value) > 1 else 10)
                if len(value) > 2 and 'bold' in value[2]:
                    font.setBold(True)
                self._qt_widget.setFont(font)
        elif key == 'fg' or key == 'foreground':
            self._qt_widget.setStyleSheet(f"color: {value};")
        elif key == 'bg' or key == 'background':
            self._qt_widget.setStyleSheet(f"background-color: {value};")


class Button(WidgetBase):
    """Wrapper for tk.Button using QPushButton."""
    def __init__(self, parent=None, text='', command=None, **kwargs):
        super().__init__(parent)
        parent_widget = parent._qt_widget if parent else None
        self._qt_widget = QPushButton(text, parent_widget)
        if command:
            self._qt_widget.clicked.connect(command)
        self._configure_kwargs(kwargs)
    
    def _configure_kwargs(self, kwargs):
        for key, value in kwargs.items():
            if key != 'command':  # Already handled
                self._configure_property(key, value)
    
    def _configure_property(self, key, value):
        if key == 'text':
            self._qt_widget.setText(str(value))
        elif key == 'command':
            self._qt_widget.clicked.connect(value)
        elif key == 'state':
            self._qt_widget.setEnabled(value != 'disabled')


class Entry(WidgetBase):
    """Wrapper for tk.Entry using QLineEdit."""
    def __init__(self, parent=None, textvariable=None, **kwargs):
        super().__init__(parent)
        parent_widget = parent._qt_widget if parent else None
        self._qt_widget = QLineEdit(parent_widget)
        self._textvariable = textvariable
        if textvariable:
            self._qt_widget.setText(textvariable.get())
            self._qt_widget.textChanged.connect(lambda text: textvariable.set(text))
        self._configure_kwargs(kwargs)
    
    def _configure_kwargs(self, kwargs):
        for key, value in kwargs.items():
            if key != 'textvariable':
                self._configure_property(key, value)
    
    def get(self):
        return self._qt_widget.text()
    
    def insert(self, index, text):
        current = self._qt_widget.text()
        self._qt_widget.setText(current + text)


class Checkbutton(WidgetBase):
    """Wrapper for tk.Checkbutton using QCheckBox."""
    def __init__(self, parent=None, text='', variable=None, command=None, **kwargs):
        super().__init__(parent)
        parent_widget = parent._qt_widget if parent else None
        self._qt_widget = QCheckBox(text, parent_widget)
        self._variable = variable
        if variable:
            self._qt_widget.setChecked(variable.get())
            self._qt_widget.toggled.connect(lambda checked: variable.set(checked))
        if command:
            self._qt_widget.toggled.connect(command)
        self._configure_kwargs(kwargs)
    
    def _configure_kwargs(self, kwargs):
        for key, value in kwargs.items():
            if key not in ('variable', 'command'):
                self._configure_property(key, value)


class Listbox(WidgetBase):
    """Wrapper for tk.Listbox using QListWidget."""
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        parent_widget = parent._qt_widget if parent else None
        self._qt_widget = QListWidget(parent_widget)
        self._configure_kwargs(kwargs)
    
    def _configure_kwargs(self, kwargs):
        for key, value in kwargs.items():
            self._configure_property(key, value)
    
    def insert(self, index, *items):
        for item in items:
            self._qt_widget.addItem(str(item))
    
    def delete(self, first, last=None):
        if last is None:
            self._qt_widget.takeItem(first)
        else:
            for i in range(last, first - 1, -1):
                self._qt_widget.takeItem(i)
    
    def get(self, first, last=None):
        if last is None:
            item = self._qt_widget.item(first)
            return item.text() if item else None
        else:
            return [self._qt_widget.item(i).text() for i in range(first, last + 1)]


class Text(WidgetBase):
    """Wrapper for tk.Text using QTextEdit."""
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        parent_widget = parent._qt_widget if parent else None
        self._qt_widget = QTextEdit(parent_widget)
        self._configure_kwargs(kwargs)
    
    def _configure_kwargs(self, kwargs):
        for key, value in kwargs.items():
            self._configure_property(key, value)
    
    def insert(self, index, text):
        self._qt_widget.append(text)
    
    def get(self, start, end):
        return self._qt_widget.toPlainText()
    
    def delete(self, start, end):
        self._qt_widget.clear()


class Toplevel(WidgetBase):
    """Wrapper for tk.Toplevel using QDialog."""
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        _ensure_qt_app()
        parent_widget = parent._qt_widget if parent else None
        self._qt_widget = QDialog(parent_widget)
        self._layout = QVBoxLayout(self._qt_widget)
        self._configure_kwargs(kwargs)
    
    def _configure_kwargs(self, kwargs):
        for key, value in kwargs.items():
            self._configure_property(key, value)
    
    def _configure_property(self, key, value):
        if key == 'title':
            self._qt_widget.setWindowTitle(str(value))
    
    def title(self, text=None):
        if text:
            self._qt_widget.setWindowTitle(text)
        return self._qt_widget.windowTitle()
    
    def geometry(self, geom=None):
        if geom:
            # Parse "WxH+X+Y" format
            parts = geom.replace('+', 'x').split('x')
            if len(parts) >= 2:
                w, h = int(parts[0]), int(parts[1])
                self._qt_widget.resize(w, h)
                if len(parts) >= 4:
                    x, y = int(parts[2]), int(parts[3])
                    self._qt_widget.move(x, y)
    
    def protocol(self, name, func):
        """Set window protocol handler."""
        if name == "WM_DELETE_WINDOW":
            self._qt_widget.closeEvent = lambda event: func()
    
    def destroy(self):
        """Destroy the window."""
        if self._qt_widget:
            self._qt_widget.close()
            self._qt_widget.deleteLater()
            self._qt_widget = None


class Tk(WidgetBase):
    """Wrapper for tk.Tk using QMainWindow."""
    def __init__(self):
        super().__init__(None)
        _ensure_qt_app()
        self._qt_widget = QMainWindow()
        central = QWidget()
        self._qt_widget.setCentralWidget(central)
        self._layout = QVBoxLayout(central)
    
    def title(self, text):
        self._qt_widget.setWindowTitle(text)
    
    def geometry(self, geom):
        # Parse "WxH" format
        parts = geom.split('x')
        if len(parts) >= 2:
            w, h = int(parts[0]), int(parts[1])
            self._qt_widget.resize(w, h)
    
    def mainloop(self):
        """Start Qt event loop."""
        self._qt_widget.show()
        return _qt_app.exec()


# ==================== Constants ====================

# Layout constants
X = 'x'
Y = 'y'
BOTH = 'both'
LEFT = 'left'
RIGHT = 'right'
TOP = 'top'
BOTTOM = 'bottom'
CENTER = 'center'

# Widget states
NORMAL = 'normal'
DISABLED = 'disabled'
ACTIVE = 'active'

# Selection modes
SINGLE = 'single'
BROWSE = 'browse'
MULTIPLE = 'multiple'
EXTENDED = 'extended'

# Anchor positions
W = 'w'
E = 'e'
N = 'n'
S = 's'
NW = 'nw'
NE = 'ne'
SW = 'sw'
SE = 'se'


# ==================== Dialog Functions ====================

class filedialog:
    @staticmethod
    def askopenfilename(**kwargs):
        _ensure_qt_app()
        filename, _ = QFileDialog.getOpenFileName(
            None,
            kwargs.get('title', 'Open File'),
            kwargs.get('initialdir', ''),
            kwargs.get('filetypes', 'All Files (*)')
        )
        return filename
    
    @staticmethod
    def askopenfilenames(**kwargs):
        _ensure_qt_app()
        filenames, _ = QFileDialog.getOpenFileNames(
            None,
            kwargs.get('title', 'Open Files'),
            kwargs.get('initialdir', ''),
            kwargs.get('filetypes', 'All Files (*)')
        )
        return filenames
    
    @staticmethod
    def askdirectory(**kwargs):
        _ensure_qt_app()
        dirname = QFileDialog.getExistingDirectory(
            None,
            kwargs.get('title', 'Select Directory'),
            kwargs.get('initialdir', '')
        )
        return dirname


class messagebox:
    @staticmethod
    def showinfo(title, message):
        _ensure_qt_app()
        QMessageBox.information(None, title, message)
    
    @staticmethod
    def showwarning(title, message):
        _ensure_qt_app()
        QMessageBox.warning(None, title, message)
    
    @staticmethod
    def showerror(title, message):
        _ensure_qt_app()
        QMessageBox.critical(None, title, message)
    
    @staticmethod
    def askyesno(title, message):
        _ensure_qt_app()
        reply = QMessageBox.question(None, title, message,
                                     QMessageBox.Yes | QMessageBox.No)
        return reply == QMessageBox.Yes


class colorchooser:
    @staticmethod
    def askcolor(**kwargs):
        _ensure_qt_app()
        color = QColorDialog.getColor()
        if color.isValid():
            rgb = (color.red(), color.green(), color.blue())
            hex_color = color.name()
            return (rgb, hex_color)
        return (None, None)


# ==================== Export API ====================

# If PySide6 is not available, export real tkinter
if not PYSIDE6_AVAILABLE:
    # Fall back to real tkinter
    Frame = _real_tk.Frame
    Label = _real_tk.Label
    Button = _real_tk.Button
    Entry = _real_tk.Entry
    Checkbutton = _real_tk.Checkbutton
    Listbox = _real_tk.Listbox
    Text = _real_tk.Text
    Toplevel = _real_tk.Toplevel
    Tk = _real_tk.Tk
    StringVar = _real_tk.StringVar
    BooleanVar = _real_tk.BooleanVar
    IntVar = _real_tk.IntVar
    DoubleVar = _real_tk.DoubleVar
    filedialog = _real_tk.filedialog
    messagebox = _real_tk.messagebox
    colorchooser = _real_tk.colorchooser


__all__ = [
    'Tk', 'Frame', 'Label', 'Button', 'Entry', 'Checkbutton', 'Listbox', 'Text', 'Toplevel',
    'StringVar', 'BooleanVar', 'IntVar', 'DoubleVar',
    'filedialog', 'messagebox', 'colorchooser',
    'X', 'Y', 'BOTH', 'LEFT', 'RIGHT', 'TOP', 'BOTTOM', 'CENTER',
    'NORMAL', 'DISABLED', 'ACTIVE',
    'SINGLE', 'BROWSE', 'MULTIPLE', 'EXTENDED',
    'W', 'E', 'N', 'S', 'NW', 'NE', 'SW', 'SE'
]