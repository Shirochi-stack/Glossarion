# Custom hook for PySide6.QtNetwork to avoid subprocess crash during SSL check
# This bypasses the problematic _check_if_openssl_enabled() call

from PyInstaller.utils.hooks.qt import get_qt_library_info

# Get PySide6 info
pyside6_library_info = get_qt_library_info('PySide6')

# Collect Qt plugins without triggering SSL check
datas = []
binaries = []

# Just add basic hiddenimports - skip the problematic SSL detection
hiddenimports = ['PySide6.QtNetwork']
